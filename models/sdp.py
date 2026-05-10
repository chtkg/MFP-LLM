from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SDPConfig:
    """
    Soft Dictionary Projection (SDP) config.

    F_prime: feature channels from STA features (F')
    vocab_size: LLM tokenizer vocab size (V)
    d_model: LLM hidden size (d_model), e.g. LLaMA-7B is typically 4096
    """

    F_prime: int
    vocab_size: int
    d_model: int
    prefix_len: Optional[int] = None
    temperature: float = 1.0
    softmax_dim: Literal["vocab"] = "vocab"


class SDPAligner(nn.Module):
    """
    Soft Dictionary Projection layer.

    Implements (3.13)-(3.15), with optional prefix pooling:
      X' (B,N,F',T') -> X_flat (B,L_src,F')  where L_src=N*T'
      (optional) X_flat -> X_pool (B,prefix_len,F') via learnable attention pooling
      logits = X_flat @ W + b               -> (B,L_src,V)
      P = softmax(logits)                  -> (B,L_src,V)
      H_graph = P @ E                      -> (B,L_src,d_model)

    Notes:
    - E is the *frozen* token embedding matrix from the LLM: (V, d_model)
    - W,b are learnable parameters.
    """

    def __init__(
        self,
        F_prime: int,
        vocab_size: int,
        d_model: int,
        *,
        prefix_len: Optional[int] = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.F_prime = F_prime
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.prefix_len = int(prefix_len) if prefix_len is not None else None
        self.temperature = float(temperature)

        self.W = nn.Parameter(torch.empty(F_prime, vocab_size))
        self.b = nn.Parameter(torch.empty(vocab_size))

        self.prefix_queries: Optional[nn.Parameter]
        if self.prefix_len is not None:
            if self.prefix_len <= 0:
                raise ValueError("prefix_len must be positive")
            # Learnable queries for attention pooling in feature space (F')
            self.prefix_queries = nn.Parameter(torch.empty(self.prefix_len, F_prime))
            nn.init.xavier_uniform_(self.prefix_queries)
        else:
            self.prefix_queries = None

        # Cached attention weights for interpretability (prefix pooling only):
        # last_prefix_attn: (B, M, L_src) where M=prefix_len, L_src=N*T'
        self.last_prefix_attn: Optional[torch.Tensor] = None

        # init
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def _prefix_pool(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Learnable attention pooling from (B,L,F') -> (B,prefix_len,F')
        """
        if self.prefix_queries is None:
            self.last_prefix_attn = None
            return x_flat
        B, L, Fp = x_flat.shape
        # Q: (B,prefix_len,F'), K/V: (B,L,F')
        Q = self.prefix_queries.unsqueeze(0).expand(B, -1, -1)
        # attn scores: (B,prefix_len,L)
        scores = torch.matmul(Q, x_flat.transpose(1, 2)) / (Fp**0.5)
        attn = F.softmax(scores, dim=-1)
        # Cache attn for mapping evidence tokens back to original spatio-temporal positions.
        # Shape: (B, prefix_len, L_src)
        self.last_prefix_attn = attn.detach()
        return torch.matmul(attn, x_flat)

    def forward(self, x_prime: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x_prime: (B, N, F', T')
          token_embedding: (V, d_model) frozen LLM embedding matrix

        Returns:
          H_graph: (B, L_src, d_model) where L_src = N*T'
        """
        if x_prime.ndim != 4:
            raise ValueError(f"x_prime must be (B,N,F',T'), got {tuple(x_prime.shape)}")
        B, N, Fp, Tp = x_prime.shape
        if Fp != self.F_prime:
            raise ValueError(f"Expected F'={self.F_prime}, got {Fp}")
        if token_embedding.ndim != 2:
            raise ValueError(f"token_embedding must be (V,d_model), got {tuple(token_embedding.shape)}")
        V, d_model = token_embedding.shape
        if V != self.vocab_size or d_model != self.d_model:
            raise ValueError(
                f"Expected token_embedding ({self.vocab_size},{self.d_model}), got ({V},{d_model})"
            )

        # (3.13) time-first, then flatten to tokens: (B,T',N,F') -> (B,L_src,F')
        x_flat = x_prime.permute(0, 3, 1, 2).reshape(B, N * Tp, Fp)
        x_flat = self._prefix_pool(x_flat)  # (B,prefix_len,F') or (B,L_src,F')

        # (3.14) logits over vocab: (B,L_src,F')@(F',V) + (V) -> (B,L_src,V)
        logits = x_flat.matmul(self.W) + self.b
        if self.temperature != 1.0:
            logits = logits / self.temperature
        P = F.softmax(logits, dim=-1)

        # (3.15) weighted sum of token embeddings: (B,L_src,V)@(V,d_model)->(B,L_src,d_model)
        H_graph = P.matmul(token_embedding)
        return H_graph


__all__ = ["SDPConfig", "SDPAligner"]

