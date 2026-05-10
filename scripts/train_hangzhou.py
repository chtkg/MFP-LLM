import argparse
import os
import sys
from pathlib import Path

import torch

# Allow running without installing the package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stallm.experiments.hangzhou_pipeline import (
    build_hangzhou_datasets,
    build_hangzhou_model,
    checkpoint_payload,
    evaluate,
    load_checkpoint,
    make_criterion,
    make_loader,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STALLM on Hangzhou dataset")
    parser.add_argument("--data-root", type=str, default=r"/home/pc/WorkSpace/wangchang/Data/hangzhou")
    parser.add_argument("--llm-path", type=str, default=r"/home/pc/WorkSpace/wangchang/models/Llama-3.2-1B")
    parser.add_argument("--weather-csv", type=str, default=r"/home/pc/WorkSpace/wangchang/STALLM/stallm/data/hangzhou_weather_daily.csv")
    parser.add_argument("--graph-type", type=str, default="sml", choices=["conn", "cor", "sml"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--loss", type=str, default="huber", choices=["mse", "mae", "huber"])
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--prefix-len", type=int, default=128)
    parser.add_argument("--mlp-hidden", type=int, default=512)
    parser.add_argument("--nb-block", type=int, default=2)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--nb-chev-filter", type=int, default=64)
    parser.add_argument("--nb-time-filter", type=int, default=128)
    parser.add_argument("--time-strides", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=r"/home/pc/WorkSpace/wangchang/STALLM/outputs/hangzhou_run5")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max-train-batches", type=int, default=0, help="0 means full epoch")
    parser.add_argument("--max-val-batches", type=int, default=0, help="0 means full validation")
    parser.add_argument(
        "--ablation-no-llm",
        action="store_true",
        help="Ablation: w/o LLM (and w/o SDP). STA -> pool -> Linear -> RegressionHead only.",
    )
    parser.add_argument(
        "--d-model-ablation",
        type=int,
        default=2048,
        help="Hidden size for fusion/regression head when --ablation-no-llm (default 2048).",
    )
    parser.add_argument(
        "--ablation-no-sta",
        action="store_true",
        help="Ablation: w/o STA (no graph/spatial modeling). Use a temporal-only conv encoder as STA replacement.",
    )
    parser.add_argument(
        "--ablation-no-sdp",
        action="store_true",
        help="Ablation: w/o SDP. Use pooled STA features with a linear projection as continuous LLM prefix instead of SDP.",
    )
    parser.add_argument(
        "--ablation-no-mlp",
        action="store_true",
        help="Ablation: w/o deep MLP regression head. Replace it with a single linear layer from hidden to outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    scaler, train_set, val_set, _ = build_hangzhou_datasets(
        data_root=args.data_root,
        weather_csv=args.weather_csv,
    )
    train_loader = make_loader(train_set, args.batch_size, shuffle=True)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False)

    sample = train_set[0]
    model = build_hangzhou_model(
        device=device,
        sample=sample,
        data_root=args.data_root,
        llm_path=args.llm_path,
        graph_type=args.graph_type,
        prefix_len=args.prefix_len,
        mlp_hidden=args.mlp_hidden,
        nb_block=args.nb_block,
        K=args.K,
        nb_chev_filter=args.nb_chev_filter,
        nb_time_filter=args.nb_time_filter,
        time_strides=args.time_strides,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        ablation_no_llm=args.ablation_no_llm,
        d_model_ablation=args.d_model_ablation,
        ablation_no_sta=args.ablation_no_sta,
        ablation_no_sdp=args.ablation_no_sdp,
        ablation_no_mlp=args.ablation_no_mlp,
    )

    criterion = make_criterion(args.loss, args.huber_delta)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = load_checkpoint(args.resume, device)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        if missing or unexpected:
            print("Warning: non-strict checkpoint load (resume).")
            if missing:
                print(f"  Missing keys (initialized randomly): {missing}")
            if unexpected:
                print(f"  Unexpected keys (ignored): {unexpected}")
        if "optimizer_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except ValueError as e:
                print(
                    "Warning: optimizer_state could not be loaded (parameter groups mismatch). "
                    "Continuing with a freshly initialized optimizer."
                )
                print(f"  Details: {e}")
        start_epoch = int(ckpt.get("epoch", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    print(f"Input shape per sample: x={tuple(sample['x'].shape)}, y={tuple(sample['y'].shape)}")
    print(f"Loss: {args.loss}")
    print(f"Ablation w/o LLM: {args.ablation_no_llm}")
    print(f"Save dir: {save_dir}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        seen = 0

        for step, batch in enumerate(train_loader, start=1):
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device, dtype=torch.float32)
            xtimes = batch["xtime"]
            ytimes = batch["ytime"]
            weather_texts = batch["weather_text"]
            dataset_name = batch["dataset_name"][0]

            optimizer.zero_grad()
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
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * x.shape[0]
            seen += x.shape[0]

            if args.max_train_batches > 0 and step >= args.max_train_batches:
                break

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion=criterion,
            scaler=scaler,
            max_batches=args.max_val_batches,
        )

        print(
            f"epoch={epoch + 1} train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f} "
            f"val_mape={val_metrics['mape']:.6f} "
            f"val_wmape={val_metrics['wmape']:.6f}"
        )

        latest_payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            best_val_loss=best_val_loss,
            args=vars(args),
        )
        save_checkpoint(save_dir / "latest.pt", latest_payload)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_val_loss=best_val_loss,
                args=vars(args),
            )
            save_checkpoint(save_dir / "best.pt", best_payload)
            print(f"Saved new best checkpoint to {save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

