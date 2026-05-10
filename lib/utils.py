from __future__ import annotations

import numpy as np


def scaled_laplacian(adj_mx: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Compute scaled Laplacian: L_tilde = (2 / lambda_max) * L - I

    adj_mx: (N, N) adjacency (weighted or binary). Diagonal will be ignored.
    """
    if adj_mx.ndim != 2 or adj_mx.shape[0] != adj_mx.shape[1]:
        raise ValueError(f"adj_mx must be square (N,N), got {adj_mx.shape}")

    adj = adj_mx.astype(np.float64).copy()
    np.fill_diagonal(adj, 0.0)

    # Degree matrix
    d = np.sum(adj, axis=1)
    L = np.diag(d) - adj

    # Largest eigenvalue (real part; Laplacian should be real symmetric if adj symmetric)
    try:
        lambda_max = np.max(np.linalg.eigvals(L).real)
    except np.linalg.LinAlgError:
        lambda_max = np.linalg.norm(L, 2)

    if lambda_max < eps:
        lambda_max = eps

    n = adj.shape[0]
    L_tilde = (2.0 / lambda_max) * L - np.eye(n, dtype=np.float64)
    return L_tilde.astype(np.float32)


def cheb_polynomial(l_tilde: np.ndarray, K: int) -> list[np.ndarray]:
    """
    Compute Chebyshev polynomials T_k(L_tilde) for k=0..K-1.

    Returns list of (N, N) arrays.
    """
    if K <= 0:
        raise ValueError("K must be positive")
    if l_tilde.ndim != 2 or l_tilde.shape[0] != l_tilde.shape[1]:
        raise ValueError(f"l_tilde must be square (N,N), got {l_tilde.shape}")

    n = l_tilde.shape[0]
    t_k = [np.eye(n, dtype=np.float32)]
    if K == 1:
        return t_k

    t_k.append(l_tilde.astype(np.float32))
    for k in range(2, K):
        t_k.append((2.0 * l_tilde @ t_k[k - 1] - t_k[k - 2]).astype(np.float32))
    return t_k

