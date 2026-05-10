from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class ExplainPromptConfig:
    city: str
    task_desc: str


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", ""))


def _time_slot_label(hour: int) -> str:
    if 5 <= hour < 10:
        return "morning peak"
    if 10 <= hour < 17:
        return "daytime off-peak"
    if 17 <= hour < 21:
        return "evening peak"
    return "night/off-service period"


def summarize_predictions(
    *,
    pred: torch.Tensor,
    x_hist: Optional[torch.Tensor] = None,
    topk: int = 3,
) -> str:
    """
    Build a compact numeric prediction summary for the prompt.

    Args:
      pred: (N,2,T_out) in real units (denormalized).
      x_hist: optional (N,2,T_in) history in real units for relative change.
      topk: number of high/low stations to report per direction (kept small).
    """
    if pred.ndim != 3 or pred.shape[1] < 2:
        raise ValueError(f"pred must be (N,2,T_out), got {tuple(pred.shape)}")
    N = int(pred.shape[0])
    T_out = int(pred.shape[2])
    k = max(1, min(int(topk), N))

    mean_pred = pred[:, :2, :].mean(dim=-1)  # (N,2)
    lines: List[str] = []
    for d, name in [(0, "inflow"), (1, "outflow")]:
        vals = mean_pred[:, d]  # (N,)
        hi = torch.topk(vals, k=k, largest=True).indices.tolist()
        lo = torch.topk(vals, k=k, largest=False).indices.tolist()

        def _fmt(idx: int) -> str:
            v = float(vals[idx].item())
            if x_hist is not None:
                base = float(x_hist[idx, d, :].mean().item())
                delta = v - base
                rel = (delta / (abs(base) + 1e-6)) * 100.0
                return f"station {idx}: {v:.1f} ({delta:+.1f}, {rel:+.1f}%)"
            return f"station {idx}: {v:.1f}"

        lines.append(f"- {name} high: " + "; ".join(_fmt(i) for i in hi))
        lines.append(f"- {name} low: " + "; ".join(_fmt(i) for i in lo))

    horizon_line = f"- forecast horizon: {T_out} steps (values summarized by mean over horizon)"
    return "\n".join([horizon_line] + lines)


def evidence_lines_from_set(
    evidences: Sequence[Dict[str, Any]],
    *,
    max_items: int = 10,
) -> List[str]:
    """
    Convert structured evidence dicts into one-line natural language evidence items.
    """
    out: List[str] = []
    for i, e in enumerate(list(evidences)[: max_items]):
        n = e.get("station_idx")
        t_start = e.get("t_start")
        t_end = e.get("t_end")
        w = e.get("weight", None)
        inflow = e.get("inflow", {})
        outflow = e.get("outflow", {})

        def _dir_text(stats: Dict[str, Any], name: str) -> str:
            mean = stats.get("mean", None)
            trend = stats.get("trend", None)
            above = stats.get("above_daily_mean", None)
            if mean is None or trend is None or above is None:
                return f"{name}=unknown"
            above_txt = "above daily mean" if bool(above) else "below daily mean"
            return f"{name} mean={float(mean):.1f}, trend={trend}, {above_txt}"

        w_txt = f", weight={float(w):.3f}" if w is not None else ""
        out.append(
            f"[E{i+1}] station {n}, {t_start} to {t_end}{w_txt}: "
            f"{_dir_text(inflow, 'inflow')}; {_dir_text(outflow, 'outflow')}"
        )
    return out


def build_explain_prompt(
    cfg: ExplainPromptConfig,
    *,
    forecast_start: str,
    input_window: str,
    forecast_window: str,
    weather_text: Optional[str],
    pred_summary: str,
    evidence_lines: Sequence[str],
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_sentences: int = 6,
) -> str:
    """
    Multi-source structured prompt template for explanation generation.
    """
    dt = _parse_iso(forecast_start)
    weekday = dt.strftime("%A")
    is_weekend = weekday in {"Saturday", "Sunday"}
    slot = _time_slot_label(dt.hour)
    weather_line = f"Weather: {weather_text}\n" if weather_text else ""

    evidence_block = "\n".join(evidence_lines) if evidence_lines else "(none)"

    return (
        "You are an expert analyst for metro passenger flow forecasting.\n"
        f"City: {cfg.city}\n"
        f"Task: {cfg.task_desc}\n\n"
        "## Time context\n"
        f"Forecast start: {forecast_start}\n"
        f"Input window: {input_window}\n"
        f"Forecast window: {forecast_window}\n"
        f"Weekday: {weekday}\n"
        f"Weekend: {str(is_weekend).lower()}\n"
        f"Time slot: {slot}\n"
        f"{weather_line}\n"
        "## Numeric prediction summary\n"
        f"{pred_summary}\n\n"
        "## Key spatio-temporal evidence (must cite by [E#])\n"
        f"{evidence_block}\n\n"
        "## Output requirements\n"
        f"- Write at most {max_sentences} sentences.\n"
        "- Every sentence MUST start with one or more evidence IDs in square brackets, e.g., [E1] ... or [E1][E3] ...\n"
        "- Do not introduce external facts not in the prompt.\n"
        "- Describe a clear causal chain from historical patterns/external conditions to the predicted outcome.\n\n"
        "## Strict output format example\n"
        "[E1] <statement about a pattern supported by E1>.\n"
        "[E2] <statement about another factor supported by E2>.\n"
        "[E1][E3] <causal conclusion linked to the prediction>.\n\n"
        "Generate the explanation now.\n"
        f"(decoding hint: temperature={temperature}, top_p={top_p})"
    )


def build_template_reference(
    *,
    city: str,
    forecast_start: str,
    weather_text: Optional[str],
    pred_summary: str,
    evidence_lines: Sequence[str],
    max_sentences: int = 6,
) -> str:
    """
    Build a deterministic template reference explanation.

    This is a weak reference for BLEU/ROUGE when human references are unavailable.
    It only uses information present in the prompt: time/weather/summary/evidence.
    """
    dt = _parse_iso(forecast_start)
    weekday = dt.strftime("%A")
    is_weekend = weekday in {"Saturday", "Sunday"}
    slot = _time_slot_label(dt.hour)
    weather_clause = f" Weather indicates: {weather_text}." if weather_text else ""

    # Use first 2 evidence items for the causal chain by default.
    ev = list(evidence_lines)
    ev1 = ev[0] if len(ev) > 0 else "[E1] (no evidence provided)"
    ev2 = ev[1] if len(ev) > 1 else ""
    ev2_clause = f" Another key factor is {ev2}" if ev2 else ""

    # Extract a short horizon note from the summary (first line is horizon_line).
    summary_lines = [ln.strip() for ln in pred_summary.splitlines() if ln.strip()]
    horizon_note = summary_lines[0].lstrip("- ").strip() if summary_lines else "forecast horizon summary"

    sents: List[str] = []
    sents.append(
        f"In {city}, the forecast starts on {forecast_start} ({weekday}, weekend={str(is_weekend).lower()}, {slot}).{weather_clause}"
    )
    sents.append(f"The numeric prediction summary suggests notable station-level variations ({horizon_note}).")
    sents.append(f"Evidence-driven factors include {ev1}.")
    if ev2_clause:
        sents.append(f"{ev2_clause.strip()}.")
    sents.append(
        "Therefore, the predicted inflow/outflow patterns are consistent with these historical spatio-temporal cues and external conditions."
    )
    sents.append("Uncertainty remains due to unobserved factors beyond the provided evidence.")

    return " ".join(sents[: max(1, int(max_sentences))]).strip()


__all__ = [
    "ExplainPromptConfig",
    "summarize_predictions",
    "evidence_lines_from_set",
    "build_explain_prompt",
    "build_template_reference",
]

