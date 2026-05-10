from __future__ import annotations

import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = torch.abs(target)
    mask = denom > eps
    if not torch.any(mask):
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    ape = torch.abs(pred - target)[mask] / denom[mask]
    return torch.mean(ape)


def wmape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    numerator = torch.sum(torch.abs(pred - target))
    denominator = torch.sum(torch.abs(target))
    if denominator <= eps:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return numerator / denominator


def compute_forecast_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "mae": float(mae(pred, target).item()),
        "rmse": float(rmse(pred, target).item()),
        "mape": float(mape(pred, target).item()),
        "wmape": float(wmape(pred, target).item()),
    }


__all__ = ["compute_forecast_metrics", "mae", "mape", "rmse", "wmape"]

