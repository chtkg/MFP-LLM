from __future__ import annotations

import argparse
import json
import os
import sys
import gc
import re
import random
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Allow running without installing the package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stallm.lib.explain import (
    ExplainPromptConfig,
    build_explain_prompt,
    evidence_lines_from_set,
    summarize_predictions,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate evidence-grounded explanations with an LLM.")
    p.add_argument("--city", type=str, choices=["hangzhou", "shanghai"], required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    # Evidence selection / description
    p.add_argument("--evidence-topk", type=int, default=10)
    p.add_argument("--stride", type=int, default=4, help="Original time steps per encoder time step t'.")

    # Prediction summary
    p.add_argument("--summary-topk", type=int, default=3)

    # LLM generation
    p.add_argument("--llm-path", type=str, default="", help="Override LLM path; default uses checkpoint args.")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--min-new-tokens", type=int, default=80)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--max-sentences", type=int, default=6)

    p.add_argument("--out", type=str, default="explanations.jsonl")

    # Baselines / ablations (for comparing explanation quality).
    p.add_argument("--no-evidence", action="store_true", help="Remove key spatio-temporal evidence from the prompt.")
    p.add_argument("--no-summary", action="store_true", help="Remove numeric prediction summary from the prompt.")
    p.add_argument(
        "--random-evidence",
        action="store_true",
        help="Shuffle evidence items before building the evidence list in the prompt.",
    )
    p.add_argument(
        "--evidence-random-seed",
        type=int,
        default=0,
        help="Seed for shuffling evidence when --random-evidence is enabled (0 means use --seed).",
    )
    return p.parse_args()


def _load_city_pipeline(city: str):
    if city == "hangzhou":
        from stallm.experiments.hangzhou_pipeline import (  # type: ignore
            build_hangzhou_datasets as build_datasets,
            build_hangzhou_model as build_model,
            load_checkpoint as load_ckpt,
            make_loader as make_loader,
            set_seed as set_seed,
        )

        return build_datasets, build_model, load_ckpt, make_loader, set_seed

    from stallm.experiments.shanghai_pipeline import (  # type: ignore
        build_shanghai_datasets as build_datasets,
        build_shanghai_model as build_model,
        load_checkpoint as load_ckpt,
        make_loader as make_loader,
        set_seed as set_seed,
    )

    return build_datasets, build_model, load_ckpt, make_loader, set_seed


def _load_text_llm(llm_path: str, device: torch.device):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # pragma: no cover
        raise ImportError("transformers is required for explanation generation") from e

    # If llm_path looks like a local path but doesn't exist, give a clearer error.
    p = Path(llm_path)
    if ("/" in llm_path or "\\" in llm_path) and not p.exists():
        raise FileNotFoundError(
            f"LLM path does not exist: {llm_path}\n"
            "If you want to load from HuggingFace Hub, pass a repo id like 'Qwen/Qwen2.5-7B-Instruct'.\n"
            "If you want to load from local files, pass the correct directory that contains config.json."
        )

    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in half precision on CUDA to reduce VRAM.
    model_kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
    if device.type == "cuda":
        # Prefer bfloat16 if available; otherwise fall back to float16.
        use_bf16 = torch.cuda.is_bf16_supported()
        model_kwargs["torch_dtype"] = torch.bfloat16 if use_bf16 else torch.float16
        # Avoid an fp32 model then .to(cuda) peak; place directly on GPU.
        model_kwargs["device_map"] = {"": 0}

    llm = AutoModelForCausalLM.from_pretrained(llm_path, **model_kwargs)
    if device.type != "cuda":
        llm.to(device)
    llm.eval()
    return llm, tokenizer


def _generate_text(
    llm,
    tokenizer,
    prompt: str,
    *,
    device: torch.device,
    temperature: float,
    top_p: float,
    min_new_tokens: int,
    max_new_tokens: int,
    max_sentences: int,
) -> str:
    def _count_sentences(s: str) -> int:
        # Rough sentence count for English/Chinese punctuation.
        parts = re.split(r"[\.!\?。！？]+", s.strip())
        return len([p for p in parts if p.strip()])

    def _looks_incomplete(s: str) -> bool:
        s = s.strip()
        if not s:
            return True
        # If it does not end with a sentence terminator, treat as incomplete.
        return not re.search(r"[\.!\?。！？]\s*$", s)

    # Prefer chat template if available (instruction-tuned LLaMA models).
    # This greatly improves instruction following compared to raw continuation.
    use_chat = hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None)

    def _encode(p: str):
        if use_chat:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that explains metro passenger flow forecasts using the provided evidence only.",
                },
                {"role": "user", "content": p},
            ]
            ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
            return ids, None
        tok = tokenizer(p, return_tensors="pt")
        ids = tok["input_ids"].to(device)
        am = tok.get("attention_mask", None)
        if am is not None:
            am = am.to(device)
        return ids, am

    do_sample = temperature > 1e-6

    text_out = ""
    base_prompt = prompt
    # Generate, then optionally continue up to 2 times if incomplete.
    for step in range(3):
        input_ids, attn = _encode(base_prompt)
        gen = llm.generate(
            input_ids=input_ids,
            attention_mask=attn,
            min_new_tokens=int(min_new_tokens) if step == 0 else 0,
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else 1.0,
            top_p=float(top_p) if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_ids = gen[0, input_ids.shape[1] :]
        chunk = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if chunk:
            text_out = (text_out + " " + chunk).strip()

        # Stop if complete enough.
        if _count_sentences(text_out) >= max_sentences or not _looks_incomplete(text_out):
            break

        # Ask for continuation without repeating.
        base_prompt = (
            prompt
            + "\n\nYour current draft explanation is:\n"
            + text_out
            + "\n\nContinue to finish the explanation. Do not repeat earlier sentences."
        )

    # Hard truncate to max_sentences for paper-friendly output.
    # Keep splitting delimiters by punctuation.
    sents = re.split(r"(?<=[\.!\?。！？])\s+", text_out.strip())
    sents = [s for s in sents if s.strip()]
    if len(sents) > max_sentences:
        text_out = " ".join(sents[:max_sentences]).strip()

    # Light post-processing: enforce evidence tag prefix per sentence.
    sents2 = re.split(r"(?<=[\.!\?。！？])\s+", text_out.strip())
    sents2 = [s.strip() for s in sents2 if s.strip()]
    fixed: list[str] = []
    for s in sents2:
        if not re.match(r"^\s*(\[[Ee]\d+\])+", s):
            s = "[E1] " + s
        fixed.append(s)
    text_out = " ".join(fixed).strip()
    return text_out


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    build_datasets, build_model, load_ckpt, make_loader, set_seed = _load_city_pipeline(args.city)
    set_seed(int(args.seed))
    random_seed = int(args.seed)
    if args.evidence_random_seed != 0:
        random_seed = int(args.evidence_random_seed)
    random.seed(random_seed)

    ckpt = load_ckpt(args.checkpoint, device)
    ckpt_args: Dict[str, Any] = ckpt["args"]

    llm_path = args.llm_path or str(ckpt_args.get("llm_path", ""))
    if not llm_path:
        raise ValueError("Missing llm_path; pass --llm-path or ensure it's stored in checkpoint args.")

    # Build datasets
    if args.city == "hangzhou":
        scaler, train_set, val_set, test_set = build_datasets(
            data_root=ckpt_args["data_root"],
            weather_csv=ckpt_args.get("weather_csv", None),
        )
    else:
        scaler, train_set, val_set, test_set = build_datasets(
            data_root=ckpt_args["data_root"],
            weather_csv=ckpt_args.get("weather_csv", "") or None,
        )

    dataset = test_set if args.split == "test" else val_set
    if dataset is None:
        raise ValueError(f"Split {args.split} not available.")

    loader = make_loader(dataset, batch_size=1, shuffle=False)

    # Build forecasting model and load weights
    sample0 = train_set[0]
    model = build_model(
        device=device,
        sample=sample0,
        data_root=ckpt_args["data_root"],
        llm_path=str(ckpt_args["llm_path"]),
        graph_type=str(ckpt_args["graph_type"]),
        prefix_len=int(ckpt_args["prefix_len"]),
        mlp_hidden=int(ckpt_args["mlp_hidden"]),
        nb_block=int(ckpt_args["nb_block"]),
        K=int(ckpt_args["K"]),
        nb_chev_filter=int(ckpt_args["nb_chev_filter"]),
        nb_time_filter=int(ckpt_args["nb_time_filter"]),
        time_strides=int(ckpt_args["time_strides"]),
        lora_r=int(ckpt_args["lora_r"]),
        lora_alpha=int(ckpt_args["lora_alpha"]),
        lora_dropout=float(ckpt_args["lora_dropout"]),
        ablation_no_llm=bool(ckpt_args.get("ablation_no_llm", False)),
        d_model_ablation=int(ckpt_args.get("d_model_ablation", 2048)),
        ablation_no_sta=bool(ckpt_args.get("ablation_no_sta", False)),
        ablation_no_sdp=bool(ckpt_args.get("ablation_no_sdp", False)),
        ablation_no_mlp=bool(ckpt_args.get("ablation_no_mlp", False)),
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print("Warning: non-strict checkpoint load for explanation generation.")
        if missing:
            print(f"  Missing keys (initialized randomly): {missing}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")
    model.to(device)
    model.eval()

    cfg = ExplainPromptConfig(
        city=args.city,
        task_desc="Predict inflow and outflow for all stations over the next horizon.",
    )

    variant = []
    if args.no_evidence:
        variant.append("no_evidence")
    if args.no_summary:
        variant.append("no_summary")
    if args.random_evidence:
        variant.append("random_evidence")
    variant_name = "+".join(variant) if variant else "full"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1: run forecasting model and build prompts/evidence on GPU.
    pending: list[Dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(pending) >= int(args.num_samples):
                break

            x = batch["x"].to(device=device, dtype=torch.float32)  # (1,N,F,T_in)
            y = batch["y"].to(device=device, dtype=torch.float32)
            xtimes = batch["xtime"]  # list[list[str]] length=1
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

            # Denormalize for human-readable stats
            x_denorm = scaler.inverse_transform_model_tensor(x)
            pred_denorm = scaler.inverse_transform_model_tensor(pred)

            # Build evidence ε (requires STALLMForecaster; skip if not available)
            evidences = []
            if hasattr(model, "build_key_evidence_set"):
                evid_batch = model.build_key_evidence_set(
                    x_flow=x_denorm,
                    xtimes=xtimes,
                    topk=int(args.evidence_topk),
                    stride=int(args.stride),
                )
                evidences = evid_batch[0] if evid_batch else []

            # Apply ablations/baselines to evidence & summary.
            if args.no_evidence:
                evidences = []
            elif args.random_evidence and evidences:
                # Deterministic shuffle for fair comparisons.
                r = random.Random(random_seed + int(batch_idx))
                evidences = list(evidences)
                r.shuffle(evidences)

            evidence_lines = evidence_lines_from_set(evidences, max_items=int(args.evidence_topk))

            # Time fields
            xt = xtimes[0]
            yt = ytimes[0]
            input_window = f"{xt[0]} to {xt[-1]}" if len(xt) >= 2 else str(xt[0])
            forecast_window = f"{yt[0]} to {yt[-1]}" if len(yt) >= 2 else str(yt[0])
            forecast_start = str(yt[0])

            weather_text: Optional[str] = None
            if weather_texts is not None and len(weather_texts) > 0:
                weather_text = weather_texts[0]

            if args.no_summary:
                pred_summary = "(summary omitted)"
            else:
                pred_summary = summarize_predictions(
                    pred=pred_denorm[0, :, :2, :],
                    x_hist=x_denorm[0, :, :2, :],
                    topk=int(args.summary_topk),
                )

            prompt = build_explain_prompt(
                cfg,
                forecast_start=forecast_start,
                input_window=input_window,
                forecast_window=forecast_window,
                weather_text=weather_text,
                pred_summary=pred_summary,
                evidence_lines=evidence_lines,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_sentences=int(args.max_sentences),
            )

            pending.append(
                {
                    "city": args.city,
                    "split": args.split,
                    "sample_index": batch_idx,
                    "station_count": int(x.shape[1]),
                    "in_steps": int(x.shape[-1]),
                    "out_steps": int(y.shape[-1]),
                    "forecast_start": forecast_start,
                    "prompt": prompt,
                    "evidence": evidences,
                    "variant": variant_name,
                }
            )

    # Free forecasting model GPU memory before loading the text LLM.
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    # Phase 2: load text LLM and generate explanations.
    text_llm, text_tok = _load_text_llm(llm_path, device)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in pending:
            text = _generate_text(
                text_llm,
                text_tok,
                rec["prompt"],
                device=device,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                min_new_tokens=int(args.min_new_tokens),
                max_new_tokens=int(args.max_new_tokens),
                max_sentences=int(args.max_sentences),
            )
            rec_out: Dict[str, Any] = dict(rec)
            rec_out["explanation"] = text
            f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

    print(f"Wrote {len(pending)} explanations to {out_path}")


if __name__ == "__main__":
    main()

