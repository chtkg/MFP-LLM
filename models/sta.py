from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stallm.lib.utils import cheb_polynomial, scaled_laplacian


def _as_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


class SpatialAttention(nn.Module):
    """
    Spatial attention S_normalized: (B, N, N)
    Input x: (B, N, F, T)
    """

    def __init__(self, in_channels: int, num_nodes: int, num_timesteps: int, device: torch.device):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(num_timesteps, device=device))
        self.W2 = nn.Parameter(torch.empty(in_channels, num_timesteps, device=device))
        self.W3 = nn.Parameter(torch.empty(in_channels, device=device))
        self.bs = nn.Parameter(torch.empty(1, num_nodes, num_nodes, device=device))
        self.Vs = nn.Parameter(torch.empty(num_nodes, num_nodes, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (B,N,T)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (B,T,N)
        product = torch.matmul(lhs, rhs)  # (B,N,N)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (B,N,N)
        return F.softmax(S, dim=1)


class TemporalAttention(nn.Module):
    """
    Temporal attention E_normalized: (B, T, T)
    Input x: (B, N, F, T)
    """

    def __init__(self, in_channels: int, num_nodes: int, num_timesteps: int, device: torch.device):
        super().__init__()
        self.U1 = nn.Parameter(torch.empty(num_nodes, device=device))
        self.U2 = nn.Parameter(torch.empty(in_channels, num_nodes, device=device))
        self.U3 = nn.Parameter(torch.empty(in_channels, device=device))
        self.be = nn.Parameter(torch.empty(1, num_timesteps, num_timesteps, device=device))
        self.Ve = nn.Parameter(torch.empty(num_timesteps, num_timesteps, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)  # (B,T,N)
        rhs = torch.matmul(self.U3, x)  # (B,N,T)
        product = torch.matmul(lhs, rhs)  # (B,T,T)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B,T,T)
        return F.softmax(E, dim=1)


class ChebConvWithSpatialAttention(nn.Module):
    """
    K-order Chebyshev graph convolution with spatial attention.

    x: (B, N, F_in, T)
    spatial_attention: (B, N, N)
    out: (B, N, F_out, T)
    """

    def __init__(self, K: int, cheb_polynomials: Sequence[torch.Tensor], in_channels: int, out_channels: int):
        super().__init__()
        self.K = K
        self.cheb_polynomials = list(cheb_polynomials)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = self.cheb_polynomials[0].device
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.empty(in_channels, out_channels, device=self.device)) for _ in range(K)]
        )

    def forward(self, x: torch.Tensor, spatial_attention: torch.Tensor) -> torch.Tensor:
        B, N, _, T = x.shape
        outputs: list[torch.Tensor] = []

        for t in range(T):
            graph_signal = x[:, :, :, t]  # (B,N,F_in)
            out_t = torch.zeros(B, N, self.out_channels, device=self.device)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)
                T_k_with_at = T_k.mul(spatial_attention)  # (B,N,N)
                theta_k = self.Theta[k]  # (F_in,F_out)
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (B,N,F_in)
                out_t = out_t + rhs.matmul(theta_k)  # (B,N,F_out)

            outputs.append(out_t.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class STABlock(nn.Module):
    """
    One STA block:
      TemporalAttention -> apply to x
      SpatialAttention  -> dynamic adjacency
      ChebConvWithSAt
      TimeConv (Conv2d with kernel (1,3))
      Residual + LayerNorm

    Input/Output: (B, N, C, T)
    """

    def __init__(
        self,
        device: torch.device,
        in_channels: int,
        K: int,
        nb_chev_filter: int,
        nb_time_filter: int,
        time_strides: int,
        cheb_polynomials: Sequence[torch.Tensor],
        num_nodes: int,
        num_timesteps: int,
    ):
        super().__init__()
        self.TAt = TemporalAttention(in_channels, num_nodes, num_timesteps, device=device)
        self.SAt = SpatialAttention(in_channels, num_nodes, num_timesteps, device=device)
        self.cheb_conv_SAt = ChebConvWithSpatialAttention(K, cheb_polynomials, in_channels, nb_chev_filter)

        self.time_conv = nn.Conv2d(
            nb_chev_filter,
            nb_time_filter,
            kernel_size=(1, 3),
            stride=(1, time_strides),
            padding=(0, 1),
        )
        self.residual_conv = nn.Conv2d(
            in_channels,
            nb_time_filter,
            kernel_size=(1, 1),
            stride=(1, time_strides),
        )
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, _, T = x.shape

        temporal_at = self.TAt(x)  # (B,T,T)
        x_tat = torch.matmul(x.reshape(B, -1, T), temporal_at).reshape_as(x)

        spatial_at = self.SAt(x_tat)  # (B,N,N)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_at)  # (B,N,nb_chev_filter,T)

        time_conv_out = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (B,F,N,T')
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (B,F,N,T')

        out = F.relu(x_residual + time_conv_out).permute(0, 3, 2, 1)  # (B,T',N,F)
        out = self.ln(out).permute(0, 2, 3, 1)  # (B,N,F,T')
        return out


class STA(nn.Module):
    """
    Stack of STA blocks + final conv head.

    Input:  (B, N, in_features, in_steps)
    Output: (B, N, out_features, out_steps)
    """

    def __init__(
        self,
        device: torch.device,
        nb_block: int,
        in_features: int,
        K: int,
        nb_chev_filter: int,
        nb_time_filter: int,
        time_strides: int,
        cheb_polynomials: Sequence[torch.Tensor],
        out_steps: int,
        out_features: int,
        in_steps: int,
        num_nodes: int,
    ):
        super().__init__()
        if nb_block <= 0:
            raise ValueError("nb_block must be positive")

        blocks = [
            STABlock(
                device=device,
                in_channels=in_features,
                K=K,
                nb_chev_filter=nb_chev_filter,
                nb_time_filter=nb_time_filter,
                time_strides=time_strides,
                cheb_polynomials=cheb_polynomials,
                num_nodes=num_nodes,
                num_timesteps=in_steps,
            )
        ]

        steps_after_first = in_steps // time_strides
        for _ in range(nb_block - 1):
            blocks.append(
                STABlock(
                    device=device,
                    in_channels=nb_time_filter,
                    K=K,
                    nb_chev_filter=nb_chev_filter,
                    nb_time_filter=nb_time_filter,
                    time_strides=1,
                    cheb_polynomials=cheb_polynomials,
                    num_nodes=num_nodes,
                    num_timesteps=steps_after_first,
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.final_conv = nn.Conv2d(
            in_channels=steps_after_first,
            out_channels=out_steps * out_features,
            kernel_size=(1, nb_time_filter),
        )
        self.out_steps = out_steps
        self.out_features = out_features
        self.num_nodes = num_nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)

        z = x.permute(0, 3, 1, 2)  # (B,T',N,C)
        z = self.final_conv(z)  # (B,out_steps*out_features,N,1)
        z = z[:, :, :, -1]  # (B,out_steps*out_features,N)

        B = z.shape[0]
        z = z.view(B, self.out_steps, self.out_features, self.num_nodes)  # (B,T_out,C_out,N)
        z = z.permute(0, 3, 2, 1).contiguous()  # (B,N,C_out,T_out)
        return z

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return spatio-temporal features before the final prediction head.

        Input:
          x: (B, N, in_features, in_steps)
        Output:
          feats: (B, N, F', T') where F' == nb_time_filter and T' == in_steps // time_strides
        """
        for block in self.blocks:
            x = block(x)
        return x


@dataclass(frozen=True)
class STAConfig:
    num_nodes: int
    in_features: int
    in_steps: int
    out_steps: int
    out_features: int = 2
    nb_block: int = 2
    K: int = 3
    nb_chev_filter: int = 64
    nb_time_filter: int = 64
    time_strides: int = 1


def make_sta(
    *,
    num_nodes: int,
    in_features: int,
    in_steps: int,
    out_steps: int,
    out_features: int = 2,
    nb_block: int = 2,
    K: int = 3,
    nb_chev_filter: int = 64,
    nb_time_filter: int = 64,
    time_strides: int = 1,
    adj_mx: torch.Tensor | np.ndarray,
    device: str | torch.device | None = None,
    init: Literal["xavier_uniform", "none"] = "xavier_uniform",
) -> STA:
    dev = _as_device(device)

    if isinstance(adj_mx, torch.Tensor):
        adj_np = adj_mx.detach().cpu().numpy()
    else:
        adj_np = np.asarray(adj_mx)

    L_tilde = scaled_laplacian(adj_np)
    cheb = cheb_polynomial(L_tilde, K)
    cheb_tensors = [torch.from_numpy(a).to(dev, dtype=torch.float32) for a in cheb]

    model = STA(
        device=dev,
        nb_block=nb_block,
        in_features=in_features,
        K=K,
        nb_chev_filter=nb_chev_filter,
        nb_time_filter=nb_time_filter,
        time_strides=time_strides,
        cheb_polynomials=cheb_tensors,
        out_steps=out_steps,
        out_features=out_features,
        in_steps=in_steps,
        num_nodes=num_nodes,
    ).to(dev)

    if init == "xavier_uniform":
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    elif init == "none":
        pass
    else:
        raise ValueError(f"Unknown init={init}")

    return model


__all__ = [
    "STA",
    "STABlock",
    "STAConfig",
    "make_sta",
    "SpatialAttention",
    "TemporalAttention",
    "ChebConvWithSpatialAttention",
]

