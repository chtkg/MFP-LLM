import argparse
import os
import sys
from pathlib import Path

import torch

# Allow running without installing the package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stallm.experiments.shanghai_pipeline import (
    build_shanghai_datasets,
    build_shanghai_model,
    evaluate,
    load_checkpoint,
    make_criterion,
    make_loader,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ST-ALLM on Shanghai test split")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-test-batches", type=int, default=0, help="0 means full test set")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    device = torch.device(cli_args.device)

    ckpt = load_checkpoint(cli_args.checkpoint, device)
    args = ckpt["args"]
    set_seed(int(args.get("seed", 42)))

    scaler, train_set, _, test_set = build_shanghai_datasets(
        data_root=args["data_root"],
        weather_csv=args.get("weather_csv", "") or None,
    )
    test_loader = make_loader(test_set, cli_args.batch_size, shuffle=False)

    sample = train_set[0]
    model = build_shanghai_model(
        device=device,
        sample=sample,
        data_root=args["data_root"],
        llm_path=args["llm_path"],
        graph_type=args["graph_type"],
        prefix_len=int(args["prefix_len"]),
        mlp_hidden=int(args["mlp_hidden"]),
        nb_block=int(args["nb_block"]),
        K=int(args["K"]),
        nb_chev_filter=int(args["nb_chev_filter"]),
        nb_time_filter=int(args["nb_time_filter"]),
        time_strides=int(args["time_strides"]),
        lora_r=int(args["lora_r"]),
        lora_alpha=int(args["lora_alpha"]),
        lora_dropout=float(args["lora_dropout"]),
        ablation_no_llm=bool(args.get("ablation_no_llm", False)),
        d_model_ablation=int(args.get("d_model_ablation", 2048)),
        ablation_no_sta=bool(args.get("ablation_no_sta", False)),
        ablation_no_sdp=bool(args.get("ablation_no_sdp", False)),
        ablation_no_mlp=bool(args.get("ablation_no_mlp", False)),
    )
    model.load_state_dict(ckpt["model_state"])

    criterion = make_criterion(args["loss"], float(args["huber_delta"]))
    metrics = evaluate(
        model,
        test_loader,
        device,
        criterion=criterion,
        scaler=scaler,
        max_batches=cli_args.max_test_batches,
    )

    ckpt_path = Path(cli_args.checkpoint)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Test loss:  {metrics['loss']:.6f}")
    print(f"Test MAE:   {metrics['mae']:.6f}")
    print(f"Test RMSE:  {metrics['rmse']:.6f}")
    # Here MAPE is already the truncated MAPE (|y|>10) computed in the Shanghai evaluator.
    print(f"Test MAPE:  {metrics['mape']:.6f}")
    print(f"Test WMAPE: {metrics['wmape']:.6f}")

    # Horizon-wise metrics if available, for horizons 1..4 (or shorter if out_steps < 4).
    for h in range(1, 5):
        mae_key = f"mae_h{h}"
        if mae_key not in metrics:
            break
        print(f"\nHorizon 1..{h}:")
        print(f"  MAE:   {metrics.get(f'mae_h{h}', float('nan')):.6f}")
        print(f"  RMSE:  {metrics.get(f'rmse_h{h}', float('nan')):.6f}")
        # Per-horizon MAPE is also truncated (|y|>10), see Shanghai evaluator.
        print(f"  MAPE:  {metrics.get(f'mape_h{h}', float('nan')):.6f}")
        print(f"  WMAPE: {metrics.get(f'wmape_h{h}', float('nan')):.6f}")


if __name__ == "__main__":
    main()

