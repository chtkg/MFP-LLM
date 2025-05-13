import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial


class SpatialAttention(nn.Module):
    def __init__(self, device, input_dim, num_nodes, num_steps):
        super().__init__()
        self.weight_a = nn.Parameter(torch.randn(num_steps).to(device))
        self.weight_b = nn.Parameter(torch.randn(input_dim, num_steps).to(device))
        self.weight_c = nn.Parameter(torch.randn(input_dim).to(device))
        self.bias_s = nn.Parameter(torch.randn(1, num_nodes, num_nodes).to(device))
        self.proj_s = nn.Parameter(torch.randn(num_nodes, num_nodes).to(device))

    def forward(self, x):
        lhs = torch.matmul(torch.matmul(x, self.weight_a), self.weight_b)
        rhs = torch.matmul(self.weight_c, x).transpose(-1, -2)
        attention_scores = torch.matmul(lhs, rhs)
        attn = torch.sigmoid(attention_scores + self.bias_s)
        S = torch.matmul(self.proj_s, attn)
        return F.softmax(S, dim=1)


class TemporalAttention(nn.Module):
    def __init__(self, device, input_dim, num_nodes, num_steps):
        super().__init__()
        self.temp_w1 = nn.Parameter(torch.randn(num_nodes).to(device))
        self.temp_w2 = nn.Parameter(torch.randn(input_dim, num_nodes).to(device))
        self.temp_w3 = nn.Parameter(torch.randn(input_dim).to(device))
        self.bias_t = nn.Parameter(torch.randn(1, num_steps, num_steps).to(device))
        self.proj_t = nn.Parameter(torch.randn(num_steps, num_steps).to(device))

    def forward(self, x):
        bsz, N, F, T = x.size()
        x_reshaped = x.permute(0, 3, 2, 1)
        lhs = torch.matmul(torch.matmul(x_reshaped, self.temp_w1), self.temp_w2)
        rhs = torch.matmul(self.temp_w3, x)
        attn_product = torch.matmul(lhs, rhs)
        E = torch.matmul(self.proj_t, torch.sigmoid(attn_product + self.bias_t))
        return F.softmax(E, dim=1)


class ChebGraphConv(nn.Module):
    def __init__(self, k_order, cheb_basis, input_dim, output_dim):
        super().__init__()
        self.k = k_order
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cheb_kernels = cheb_basis
        self.device = cheb_basis[0].device
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, output_dim).to(self.device)) for _ in range(k_order)
        ])

    def forward(self, x, attn=None):
        bsz, N, F_in, T = x.size()
        results = []

        for t in range(T):
            x_slice = x[:, :, :, t]
            agg = torch.zeros(bsz, N, self.output_dim).to(self.device)
            for k in range(self.k):
                T_k = self.cheb_kernels[k]
                if attn is not None:
                    T_k = T_k * attn
                rhs = T_k.permute(0, 2, 1) @ x_slice
                agg += rhs @ self.weights[k]
            results.append(agg.unsqueeze(-1))
        return F.relu(torch.cat(results, dim=-1))


class SpatioTemporalBlock(nn.Module):
    def __init__(self, device, input_dim, k_order, gcn_hidden, temporal_hidden, stride, cheb_basis, num_nodes, seq_len):
        super().__init__()
        self.temp_attn = TemporalAttention(device, input_dim, num_nodes, seq_len)
        self.spat_attn = SpatialAttention(device, input_dim, num_nodes, seq_len)
        self.dynamic_gcn = ChebGraphConv(k_order, cheb_basis, input_dim, gcn_hidden)
        self.time_conv = nn.Conv2d(gcn_hidden, temporal_hidden, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1))
        self.skip_proj = nn.Conv2d(input_dim, temporal_hidden, kernel_size=(1, 1), stride=(1, stride))
        self.norm_layer = nn.LayerNorm(temporal_hidden)

    def forward(self, x):
        B, N, F, T = x.size()
        temp_weights = self.temp_attn(x)
        x_temp = torch.matmul(x.view(B, -1, T), temp_weights).view(B, N, F, T)
        spat_weights = self.spat_attn(x_temp)
        gcn_out = self.dynamic_gcn(x, spat_weights)
        tcn_out = self.time_conv(gcn_out.permute(0, 2, 1, 3))
        residual = self.skip_proj(x.permute(0, 2, 1, 3))
        out = F.relu(tcn_out + residual).permute(0, 3, 2, 1)
        return self.norm_layer(out).permute(0, 2, 3, 1)


class STGNNEncoder(nn.Module):
    def __init__(self, device, num_blocks, input_dim, k_order, gcn_hidden, temporal_hidden, stride,
                 cheb_basis, pred_len, input_len, num_nodes):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(
            SpatioTemporalBlock(device, input_dim, k_order, gcn_hidden, temporal_hidden, stride,
                                cheb_basis, num_nodes, input_len)
        )
        for _ in range(num_blocks - 1):
            self.blocks.append(
                SpatioTemporalBlock(device, temporal_hidden, k_order, gcn_hidden, temporal_hidden, 1,
                                    cheb_basis, num_nodes, input_len // stride)
            )
        self.output_layer = nn.Conv2d(in_channels=input_len // stride, out_channels=pred_len,
                                      kernel_size=(1, temporal_hidden))
        self.device = device
        self.to(device)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        out = self.output_layer(x.permute(0, 3, 1, 2))[:, :, :, -1]
        return out.permute(0, 2, 1)


def build_stgnn_model(device, num_blocks, input_dim, k_order, gcn_hidden, temporal_hidden,
                      stride, adj_matrix, pred_len, input_len, num_nodes):
    L_norm = scaled_Laplacian(adj_matrix)
    cheb_basis = cheb_polynomial(L_norm, k_order)
    cheb_tensors = [torch.tensor(basis, dtype=torch.float32, device=device) for basis in cheb_basis]

    model = STGNNEncoder(
        device=device,
        num_blocks=num_blocks,
        input_dim=input_dim,
        k_order=k_order,
        gcn_hidden=gcn_hidden,
        temporal_hidden=temporal_hidden,
        stride=stride,
        cheb_basis=cheb_tensors,
        pred_len=pred_len,
        input_len=input_len,
        num_nodes=num_nodes
    )

    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.uniform_(param)

    return model