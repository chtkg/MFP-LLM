from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stallm.data.shanghai import (
    ShanghaiDataset,
    StandardScaler,
    fit_shanghai_scaler,
    load_shanghai_graph,
)
from stallm.lib.metrics import compute_forecast_metrics
from stallm.models.llm_forecast import (
    STALLMForecaster,
    STALLMForecasterNoLLM,
    STALLMForecasterNoSDP,
    apply_lora,
    load_llama_local,
)
from stallm.models.ablation_encoders import TemporalConvEncoder
from stallm.models.sdp import SDPAligner
from stallm.models.sta import make_sta


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def make_criterion(loss_name: str, huber_delta: float) -> nn.Module:
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    if loss_name == "huber":
        return nn.HuberLoss(delta=huber_delta)
    raise ValueError(f"Unsupported loss: {loss_name}")


def build_shanghai_datasets(
    *,
    data_root: str | Path,
    weather_csv: str | Path | None,
) -> tuple[StandardScaler, ShanghaiDataset, ShanghaiDataset, ShanghaiDataset]:
    data_root = Path(data_root)
    # Treat empty string as "no weather".
    if weather_csv is not None and str(weather_csv) == "":
        weather_csv = None
    scaler = fit_shanghai_scaler(data_root / "train.pkl")
    train_set = ShanghaiDataset(
        data_root,
        "train",
        scaler=scaler,
        dataset_name="shanghai",
        weather_csv_path=weather_csv,
    )
    val_set = ShanghaiDataset(
        data_root,
        "val",
        scaler=scaler,
        dataset_name="shanghai",
        weather_csv_path=weather_csv,
    )
    test_set = ShanghaiDataset(
        data_root,
        "test",
        scaler=scaler,
        dataset_name="shanghai",
        weather_csv_path=weather_csv,
    )
    return scaler, train_set, val_set, test_set


def build_shanghai_model(
    *,
    device: torch.device,
    sample: dict[str, Any],
    data_root: str | Path,
    llm_path: str,
    graph_type: str,
    prefix_len: int,
    mlp_hidden: int,
    nb_block: int,
    K: int,
    nb_chev_filter: int,
    nb_time_filter: int,
    time_strides: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    ablation_no_llm: bool = False,
    d_model_ablation: int = 2048,
    ablation_no_sta: bool = False,
    ablation_no_sdp: bool = False,
    ablation_no_mlp: bool = False,
) -> STALLMForecaster | STALLMForecasterNoLLM | STALLMForecasterNoSDP:
    graph = load_shanghai_graph(data_root, graph_type)
    num_nodes = int(sample["x"].shape[0])
    in_features = int(sample["x"].shape[1])
    in_steps = int(sample["x"].shape[2])
    out_steps = int(sample["y"].shape[2])

    if ablation_no_sta:
        sta = TemporalConvEncoder(
            in_features=in_features,
            hidden_channels=nb_time_filter,
            num_layers=max(1, nb_block),
            time_strides=time_strides,
            dropout=0.1,
        ).to(device)
    else:
        sta = make_sta(
            num_nodes=num_nodes,
            in_features=in_features,
            in_steps=in_steps,
            out_steps=out_steps,
            out_features=in_features,
            nb_block=nb_block,
            K=K,
            nb_chev_filter=nb_chev_filter,
            nb_time_filter=nb_time_filter,
            time_strides=time_strides,
            adj_mx=graph,
            device=device,
        )

    if ablation_no_llm:
        feat_sample = sta.forward_features(sample["x"].unsqueeze(0).to(device=device, dtype=torch.float32))
        feat_dim = int(feat_sample.shape[2])
        model = STALLMForecasterNoLLM(
            sta=sta,
            num_nodes=num_nodes,
            out_steps=out_steps,
            out_features=in_features,
            mlp_hidden=mlp_hidden,
            feat_dim=feat_dim,
            d_model=d_model_ablation,
            use_linear_head=ablation_no_mlp,
        ).to(device)
        return model

    feat_sample = sta.forward_features(sample["x"].unsqueeze(0).to(device=device, dtype=torch.float32))
    feat_dim = int(feat_sample.shape[2])

    llm, tokenizer = load_llama_local(llm_path, device=device)
    llm = apply_lora(
        llm,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    if ablation_no_sdp:
        model = STALLMForecasterNoSDP(
            sta=sta,
            llm=llm,
            tokenizer=tokenizer,
            num_nodes=num_nodes,
            out_steps=out_steps,
            out_features=in_features,
            mlp_hidden=mlp_hidden,
            feat_dim=feat_dim,
            prefix_len=prefix_len,
            use_linear_head=ablation_no_mlp,
        ).to(device)
        return model

    sdp = SDPAligner(
        F_prime=feat_dim,
        vocab_size=int(llm.get_input_embeddings().weight.shape[0]),
        d_model=int(llm.config.hidden_size),
        prefix_len=prefix_len,
    ).to(device)

    model = STALLMForecaster(
        sta=sta,
        sdp=sdp,
        llm=llm,
        tokenizer=tokenizer,
        num_nodes=num_nodes,
        out_steps=out_steps,
        out_features=in_features,
        mlp_hidden=mlp_hidden,
        freeze_llm=False,
        use_linear_head=ablation_no_mlp,
    ).to(device)
    return model


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    scaler: StandardScaler,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    # Weighted by batch size so result is independent of batch_size (global mean over all samples).
    # Here, \"mape\" is defined as a truncated MAPE that only considers positions with |y| > 10.
    metric_sums = {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "wmape": 0.0}
    horizon_metric_sums: dict[int, dict[str, float]] = {}
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device, dtype=torch.float32)
            xtimes = batch["xtime"]
            ytimes = batch["ytime"]
            weather_texts = batch["weather_text"]
            dataset_name = batch["dataset_name"][0]

            pred = model(
                x,
                dataset_name=dataset_name,
                in_steps=x.shape[-1],
                out_steps=y.shape[-1],
                xtimes=xtimes,
                ytimes=ytimes,
                weather_texts=weather_texts,
            )
            loss = criterion(pred, y)
            total_loss += float(loss.item()) * x.shape[0]
            total_count += x.shape[0]

            pred_denorm = scaler.inverse_transform_model_tensor(pred)
            y_denorm = scaler.inverse_transform_model_tensor(y)
            batch_size = x.shape[0]
            metrics = compute_forecast_metrics(pred_denorm, y_denorm)
            # Accumulate MAE/RMSE/WMAPE as usual (global mean over all samples).
            for key, value in metrics.items():
                if key in metric_sums and key != "mape":
                    metric_sums[key] += value * batch_size

            # Truncated MAPE over elements with sufficiently large target (for Shanghai evaluation only).
            # This mitigates the effect of many near-zero flows on MAPE, and we treat it as \"mape\".
            denom = torch.abs(y_denorm)
            mask = denom > 10.0  # threshold = 10 passengers
            if torch.any(mask):
                ape_trunc = torch.abs(pred_denorm - y_denorm)[mask] / denom[mask]
                batch_mape_trunc = float(ape_trunc.mean().item())
            else:
                batch_mape_trunc = 0.0
            metric_sums["mape"] += batch_mape_trunc * batch_size

            # Per-horizon metrics over cumulative windows 1..H, where H = out_steps.
            H = int(y_denorm.shape[-1])
            for h in range(1, H + 1):
                if h not in horizon_metric_sums:
                    horizon_metric_sums[h] = {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "wmape": 0.0}
                pred_h = pred_denorm[..., :h]
                y_h = y_denorm[..., :h]
                m_h = compute_forecast_metrics(pred_h, y_h)
                for key, value in m_h.items():
                    if key in horizon_metric_sums[h] and key != "mape":
                        horizon_metric_sums[h][key] += value * batch_size

                # Truncated per-horizon MAPE (interpreted as horizon-wise MAPE for Shanghai).
                denom_h = torch.abs(y_h)
                mask_h = denom_h > 10.0
                if torch.any(mask_h):
                    ape_trunc_h = torch.abs(pred_h - y_h)[mask_h] / denom_h[mask_h]
                    batch_mape_trunc_h = float(ape_trunc_h.mean().item())
                else:
                    batch_mape_trunc_h = 0.0
                horizon_metric_sums[h]["mape"] += batch_mape_trunc_h * batch_size

            if max_batches > 0 and step >= max_batches:
                break
    n = max(total_count, 1)
    results = {"loss": total_loss / n}
    for key, value in metric_sums.items():
        results[key] = value / n
    for h, sums in horizon_metric_sums.items():
        for key, value in sums.items():
            results[f"{key}_h{h}"] = value / n
    return results


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_val_loss: float,
    args: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "args": args,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    return payload


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=device)


__all__ = [
    "build_shanghai_datasets",
    "build_shanghai_model",
    "checkpoint_payload",
    "evaluate",
    "load_checkpoint",
    "make_criterion",
    "make_loader",
    "save_checkpoint",
    "set_seed",
]

