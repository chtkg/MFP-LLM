from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from .hangzhou import StandardScaler, Split, format_daily_weather_text, load_daily_weather


GraphType = Literal["conn", "cor", "sml"]


def load_shanghai_split(path: str | Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    required = {"x", "y", "xtime", "ytime"}
    missing = required.difference(data.keys())
    if missing:
        raise ValueError(f"Missing keys in {path}: {sorted(missing)}")
    return data


def fit_shanghai_scaler(train_data_path: str | Path) -> StandardScaler:
    """
    Fit feature-wise standardization stats from Shanghai train split.
    """
    data = load_shanghai_split(train_data_path)
    x = data["x"].astype(np.float32)
    mean = x.mean(axis=(0, 1, 2), keepdims=True)  # (1,1,1,F)
    std = x.std(axis=(0, 1, 2), keepdims=True)  # (1,1,1,F)
    return StandardScaler(mean=mean, std=std)


def load_shanghai_graph(root_dir: str | Path, graph_type: GraphType = "conn") -> np.ndarray:
    """
    Load Shanghai adjacency matrix.

    Assumption: file is named like graph_sh_{graph_type}.pkl under root_dir,
    e.g. graph_sh_conn.pkl / graph_sh_cor.pkl / graph_sh_sml.pkl
    """
    root = Path(root_dir)
    graph_path = root / f"graph_sh_{graph_type}.pkl"
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    graph = np.asarray(graph, dtype=np.float32)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError(f"Expected square adjacency, got {graph.shape}")
    return graph


class ShanghaiDataset(Dataset):
    """
    Shanghai metro flow dataset.

    Expected split format (same as Hangzhou):
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
        dataset_name: str = "shanghai",
        weather_csv_path: str | Path | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.dataset_name = dataset_name
        self.data = load_shanghai_split(self.root_dir / f"{split}.pkl")
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
        # Always return a string for weather_text to avoid None in DataLoader collation.
        weather_text = ""
        if self.weather_by_date is not None:
            weather_row = self.weather_by_date.get(forecast_date)
            if weather_row is not None:
                weather_text = format_daily_weather_text(weather_row)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "xtime": xtime,
            "ytime": ytime,
            "timestamp": ytime[0],
            "weather_text": weather_text,
            "dataset_name": self.dataset_name,
        }


__all__ = [
    "GraphType",
    "ShanghaiDataset",
    "StandardScaler",
    "Split",
    "fit_shanghai_scaler",
    "load_shanghai_graph",
    "load_shanghai_split",
]

