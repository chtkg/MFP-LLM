from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


Split = Literal["train", "val", "test"]
GraphType = Literal["conn", "cor", "sml"]


@dataclass(frozen=True)
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + self.eps) + self.mean

    def transform_model_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform tensors in model layout (..., F, T), e.g. (B, N, F, T).
        """
        mean = torch.as_tensor(self.mean.reshape(1, 1, -1, 1), device=x.device, dtype=x.dtype)
        std = torch.as_tensor(self.std.reshape(1, 1, -1, 1), device=x.device, dtype=x.dtype)
        return (x - mean) / (std + self.eps)

    def inverse_transform_model_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse-transform tensors in model layout (..., F, T), e.g. (B, N, F, T).
        """
        mean = torch.as_tensor(self.mean.reshape(1, 1, -1, 1), device=x.device, dtype=x.dtype)
        std = torch.as_tensor(self.std.reshape(1, 1, -1, 1), device=x.device, dtype=x.dtype)
        return x * (std + self.eps) + mean


def fit_hangzhou_scaler(train_data_path: str | Path) -> StandardScaler:
    """
    Fit feature-wise standardization stats from train split.

    train['x'] shape: (num_samples, T_in, N, F)
    We compute mean/std over (samples, time, nodes), preserving feature dim F.
    """
    data = load_hangzhou_split(train_data_path)
    x = data["x"].astype(np.float32)
    mean = x.mean(axis=(0, 1, 2), keepdims=True)  # (1,1,1,F)
    std = x.std(axis=(0, 1, 2), keepdims=True)  # (1,1,1,F)
    return StandardScaler(mean=mean, std=std)


def load_hangzhou_split(path: str | Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    required = {"x", "y", "xtime", "ytime"}
    missing = required.difference(data.keys())
    if missing:
        raise ValueError(f"Missing keys in {path}: {sorted(missing)}")
    return data


def load_hangzhou_graph(root_dir: str | Path, graph_type: GraphType = "conn") -> np.ndarray:
    root = Path(root_dir)
    graph_path = root / f"graph_hz_{graph_type}.pkl"
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    graph = np.asarray(graph, dtype=np.float32)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError(f"Expected square adjacency, got {graph.shape}")
    return graph


def load_daily_weather(weather_csv_path: str | Path) -> dict[str, dict[str, str]]:
    weather_csv_path = Path(weather_csv_path)
    weather_by_date: dict[str, dict[str, str]] = {}
    with weather_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"date", "temp_min_c", "temp_max_c", "weather_desc", "has_precip"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing weather columns in {weather_csv_path}: {sorted(missing)}")
        for row in reader:
            weather_by_date[row["date"]] = row
    return weather_by_date


def format_daily_weather_text(weather_row: dict[str, str]) -> str:
    precip_text = "precipitation" if weather_row["has_precip"].lower() == "true" else "no precipitation"
    return (
        f"Daily weather: {weather_row['weather_desc']}, "
        f"low {weather_row['temp_min_c']}C, high {weather_row['temp_max_c']}C, {precip_text}."
    )


class HangzhouDataset(Dataset):
    """
    Hangzhou metro flow dataset.

    Original split format:
      x: (num_samples, T_in, N, F)
      y: (num_samples, T_out, N, F)

    Converted sample format returned here:
      x: (N, F, T_in)
      y: (N, F, T_out)
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: Split,
        *,
        scaler: StandardScaler | None = None,
        normalize_target: bool = True,
        dataset_name: str = "hangzhou",
        weather_csv_path: str | Path | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.dataset_name = dataset_name
        self.data = load_hangzhou_split(self.root_dir / f"{split}.pkl")
        self.scaler = scaler
        self.normalize_target = normalize_target
        self.weather_by_date = load_daily_weather(weather_csv_path) if weather_csv_path is not None else None

        self.x = self.data["x"].astype(np.float32)  # (S,T,N,F)
        self.y = self.data["y"].astype(np.float32)  # (S,T,N,F)
        self.xtime = self.data["xtime"]
        self.ytime = self.data["ytime"]

        if self.scaler is not None:
            self.x = self.scaler.transform(self.x).astype(np.float32)
            if self.normalize_target:
                self.y = self.scaler.transform(self.y).astype(np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | list[str]]:
        x = self.x[idx]  # (T_in,N,F)
        y = self.y[idx]  # (T_out,N,F)

        # Convert to model format: (N,F,T)
        x = np.transpose(x, (1, 2, 0))
        y = np.transpose(y, (1, 2, 0))

        xtime = [np.datetime_as_string(t, unit="m") for t in self.xtime[idx]]
        ytime = [np.datetime_as_string(t, unit="m") for t in self.ytime[idx]]
        forecast_date = ytime[0][:10]
        weather_text = None
        if self.weather_by_date is not None:
            weather_row = self.weather_by_date.get(forecast_date)
            if weather_row is not None:
                weather_text = format_daily_weather_text(weather_row)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "xtime": xtime,
            "ytime": ytime,
            # Use the first forecast timestamp as the prompt timestamp.
            "timestamp": ytime[0],
            "weather_text": weather_text,
            "dataset_name": self.dataset_name,
        }


__all__ = [
    "GraphType",
    "HangzhouDataset",
    "Split",
    "StandardScaler",
    "fit_hangzhou_scaler",
    "format_daily_weather_text",
    "load_daily_weather",
    "load_hangzhou_graph",
    "load_hangzhou_split",
]

