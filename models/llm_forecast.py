from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from transformers import AutoModel, AutoTokenizer
    from transformers import LlamaConfig, LlamaModel
except Exception:  # pragma: no cover
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    LlamaConfig = None  # type: ignore
    LlamaModel = None  # type: ignore

from stallm.models.sdp import SDPAligner
from stallm.models.sta import STA


PRED_TOKEN = "[PRED]"


@dataclass(frozen=True)
class PromptConfig:
    dataset_name: str
    in_steps: int
    out_steps: int
    num_nodes: int


def _parse_iso_timestamp(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", ""))


def _time_slot_label(hour: int) -> str:
    if 5 <= hour < 10:
        return "morning peak"
    if 10 <= hour < 17:
        return "daytime off-peak"
    if 17 <= hour < 21:
        return "evening peak"
    return "night/off-service period"


def build_prompt(
    cfg: PromptConfig,
    *,
    forecast_start: str,
    xtimes: Optional[Sequence[str]] = None,
    ytimes: Optional[Sequence[str]] = None,
    weather_text: Optional[str] = None,
) -> str:
    # Deterministic template with richer temporal context.
    dt = _parse_iso_timestamp(forecast_start)
    weekday = dt.strftime("%A")
    is_weekend = weekday in {"Saturday", "Sunday"}
    slot = _time_slot_label(dt.hour)

    input_window = f"{xtimes[0]} to {xtimes[-1]}" if xtimes else f"past {cfg.in_steps} time steps"
    forecast_window = f"{ytimes[0]} to {ytimes[-1]}" if ytimes else f"next {cfg.out_steps} time steps"
    weather_line = f"{weather_text}\n" if weather_text else ""

    return (
        "You are a forecasting model.\n"
        f"Dataset: {cfg.dataset_name}\n"
        f"Forecast start time: {forecast_start}\n"
        f"Input window: {input_window}\n"
        f"Forecast window: {forecast_window}\n"
        f"Temporal context: weekday={weekday}, weekend={str(is_weekend).lower()}, time_slot={slot}.\n"
        f"{weather_line}"
        f"Input: spatio-temporal features of metro passenger flows for {cfg.num_nodes} stations "
        f"over {cfg.in_steps} observed time steps.\n"
        f"Task: predict inflow and outflow for each station over the next {cfg.out_steps} time steps.\n"
        "Output format requirement: output a compact latent vector at the [PRED] position for regression.\n"
        f"{PRED_TOKEN}"
    )


class RegressionHead(nn.Module):
    def __init__(self, d_model: int, hidden: int, out_dim: int):
        super().__init__()
        # Deeper MLP improves capacity for multi-output regression.
        # Keep the same constructor signature to avoid changing training scripts.
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearRegressionHead(nn.Module):
    """
    Ablation: w/o deep MLP.

    Replace the multi-layer RegressionHead with a single linear projection
    from LLM hidden size to the full output dimension.
    """

    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class STALLMForecaster(nn.Module):
    """
    End-to-end model:
      x -> STA.forward_features -> SDPAligner(prefix_len) -> concat prompt embeds -> LLM -> [PRED] hidden -> MLP -> y_hat

    Output y_hat: (B, N, out_features(=2), T_out)
    """

    def __init__(
        self,
        *,
        sta: nn.Module,
        sdp: SDPAligner,
        llm: nn.Module,
        tokenizer,
        num_nodes: int,
        out_steps: int,
        out_features: int = 2,
        mlp_hidden: int = 1024,
        freeze_llm: bool = True,
        use_linear_head: bool = False,
        evidence_score_dim: Optional[int] = None,
    ):
        super().__init__()
        if not hasattr(sta, "forward_features"):
            raise TypeError("sta must implement forward_features(x) -> (B,N,F',T')")
        self.sta = sta
        self.sdp = sdp
        self.llm = llm
        self.tokenizer = tokenizer
        self.num_nodes = num_nodes
        self.out_steps = out_steps
        self.out_features = out_features

        d_model = self._llm_hidden_size()
        d_k = int(evidence_score_dim) if evidence_score_dim is not None else d_model
        self.evidence_key = nn.Linear(d_model, d_k, bias=False)
        self._evidence_dk = d_k
        self.last_evidence_scores: Optional[torch.Tensor] = None  # (B, M), set in forward
        self.last_src_weights: Optional[torch.Tensor] = None  # (B, L_src), set in forward
        self.last_src_shape: Optional[Tuple[int, int]] = None  # (N, T')

        out_dim = num_nodes * out_features * out_steps
        if use_linear_head:
            self.reg = LinearRegressionHead(d_model=d_model, out_dim=out_dim)
        else:
            self.reg = RegressionHead(d_model=d_model, hidden=mlp_hidden, out_dim=out_dim)

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

    def _llm_hidden_size(self) -> int:
        if hasattr(self.llm, "config") and hasattr(self.llm.config, "hidden_size"):
            return int(self.llm.config.hidden_size)
        raise ValueError("Cannot infer LLM hidden size; expected llm.config.hidden_size")

    def _token_embedding_matrix(self) -> torch.Tensor:
        emb = self.llm.get_input_embeddings()
        return emb.weight

    @staticmethod
    def _normalize_time_sequences(
        seq_batch: Optional[Sequence[Sequence[str]] | Sequence[str]],
        batch_size: int,
        *,
        name: str,
    ) -> Optional[List[List[str]]]:
        if seq_batch is None:
            return None

        # Case 1: already sample-major, e.g. List[List[str]] with len == B
        if len(seq_batch) == batch_size and all(not isinstance(item, str) for item in seq_batch):
            return [list(item) for item in seq_batch]  # type: ignore[arg-type]

        # Case 2: already plain per-sample timestamps, e.g. List[str] with len == B
        if len(seq_batch) == batch_size and all(isinstance(item, str) for item in seq_batch):
            return [[str(item)] for item in seq_batch]  # type: ignore[list-item]

        # Case 3: DataLoader default collate on List[str]:
        # turns List[List[str]] into time-major structure like List[Tuple[str, ...]]
        if all(not isinstance(item, str) for item in seq_batch):
            seq_list = [list(item) for item in seq_batch]  # type: ignore[arg-type]
            if seq_list and all(len(item) == batch_size for item in seq_list):
                return [list(items) for items in zip(*seq_list)]

        raise ValueError(f"{name} could not be normalized to batch-major sequences for B={batch_size}")

    @staticmethod
    def _normalize_optional_strings(
        items: Optional[Sequence[Optional[str]] | Sequence[str] | Sequence[tuple]],
        batch_size: int,
        *,
        name: str,
    ) -> Optional[List[Optional[str]]]:
        if items is None:
            return None
        if len(items) == batch_size and all(isinstance(item, (str, type(None))) for item in items):
            return [None if item is None else str(item) for item in items]
        # DataLoader may wrap batch_size==1 strings as single-item tuples
        if len(items) == batch_size and all(isinstance(item, tuple) and len(item) == 1 for item in items):
            return [None if item[0] is None else str(item[0]) for item in items]  # type: ignore[index]
        raise ValueError(f"{name} could not be normalized to a batch-major string list for B={batch_size}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        dataset_name: str,
        in_steps: int,
        out_steps: int,
        timestamps: Optional[Sequence[str]] = None,
        xtimes: Optional[Sequence[Sequence[str]]] = None,
        ytimes: Optional[Sequence[Sequence[str]]] = None,
        weather_texts: Optional[Sequence[Optional[str]]] = None,
    ) -> torch.Tensor:
        """
        Args:
          x: (B,N,F,T_in)
          timestamps: len==B, each a string. Backward-compatible fallback.
          xtimes: optional batch of input timestamp sequences
          ytimes: optional batch of forecast timestamp sequences
          weather_texts: optional batch of daily weather strings
        """
        B = x.shape[0]
        if out_steps != self.out_steps:
            raise ValueError(f"out_steps mismatch: got {out_steps}, model out_steps={self.out_steps}")
        if timestamps is not None and len(timestamps) != B:
            raise ValueError(f"timestamps must have length B={B}, got {len(timestamps)}")

        xtimes = self._normalize_time_sequences(xtimes, B, name="xtimes")
        ytimes = self._normalize_time_sequences(ytimes, B, name="ytimes")
        weather_texts = self._normalize_optional_strings(weather_texts, B, name="weather_texts")

        feats = self.sta.forward_features(x)  # (B,N,F',T')
        H_graph = self.sdp(feats, token_embedding=self._token_embedding_matrix())  # (B,prefix_len,d_model) or (B,L,d)

        # Build prompts
        prompt_cfg = PromptConfig(dataset_name=dataset_name, in_steps=in_steps, out_steps=out_steps, num_nodes=self.num_nodes)
        prompts: List[str] = []
        for i in range(B):
            x_seq = xtimes[i] if xtimes is not None else None
            y_seq = ytimes[i] if ytimes is not None else None
            weather_text = weather_texts[i] if weather_texts is not None else None
            forecast_start = y_seq[0] if y_seq is not None else (timestamps[i] if timestamps is not None else None)
            if forecast_start is None:
                raise ValueError("Provide either ytimes or timestamps for prompt construction")
            prompts.append(
                build_prompt(
                    prompt_cfg,
                    forecast_start=forecast_start,
                    xtimes=x_seq,
                    ytimes=y_seq,
                    weather_text=weather_text,
                )
            )

        tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        input_ids = tok["input_ids"].to(x.device)
        attn_text = tok["attention_mask"].to(x.device)  # (B,L_text)

        text_emb = self.llm.get_input_embeddings()(input_ids)  # (B,L_text,d_model)
        inputs_embeds = torch.cat([H_graph, text_emb], dim=1)

        prefix_len = H_graph.shape[1]
        attn_prefix = torch.ones(B, prefix_len, device=x.device, dtype=attn_text.dtype)
        attention_mask = torch.cat([attn_prefix, attn_text], dim=1)  # (B,L_total)

        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
        hidden = out.last_hidden_state  # (B,L_total,d_model)

        # Index of [PRED] token: last non-pad token of text part
        last_text_index = attn_text.sum(dim=1) - 1  # (B,)
        pred_index = prefix_len + last_text_index  # (B,)
        o_pred = hidden[torch.arange(B, device=x.device), pred_index]  # (B,d_model)

        # Evidence scoring (Eq. 4.1-4.2): q=[PRED] hidden, K=H_graph W_K, s=softmax(q^T K / sqrt(d_k)).
        # This does not affect prediction; the scores are cached for interpretability modules.
        # H_graph: (B,M,d_model); q: (B,d_model) -> K: (B,M,d_k) -> s: (B,M)
        K = self.evidence_key(H_graph)  # (B,M,d_k)
        qk = torch.bmm(K, o_pred.unsqueeze(-1)).squeeze(-1)  # (B,M), equivalent to q^T K_i
        scores = F.softmax(qk / math.sqrt(self._evidence_dk), dim=1)  # (B,M)
        self.last_evidence_scores = scores.detach()

        # Back-projection to source spatio-temporal positions (Eq. 4.3-4.4).
        # If SDP used prefix pooling, it caches A = last_prefix_attn of shape (B,M,L_src).
        # Then w_src = sum_j s_j * a_{j,src} = s @ A.
        A = getattr(self.sdp, "last_prefix_attn", None)
        if A is None:
            # No pooling: evidence tokens already correspond to flattened source positions.
            w_src = scores
            N = int(feats.shape[1])
            Tp = int(feats.shape[3])
        else:
            # (B,1,M)@(B,M,L_src)->(B,1,L_src)->(B,L_src)
            w_src = torch.bmm(scores.unsqueeze(1), A).squeeze(1)
            N = int(feats.shape[1])
            Tp = int(feats.shape[3])
        self.last_src_weights = w_src.detach()
        self.last_src_shape = (N, Tp)

        y = self.reg(o_pred)  # (B, N*out_features*out_steps)
        y = y.view(B, self.num_nodes, self.out_features, self.out_steps)
        return y

    def get_topk_spatiotemporal_positions(
        self, k: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return top-k source positions by back-projected weight.

        Returns:
          station_idx: (B,k) long
          time_idx: (B,k) long, index in T' (encoder time steps)
          weight: (B,k) float
        """
        if self.last_src_weights is None or self.last_src_shape is None:
            raise RuntimeError("Call forward() once before requesting evidence positions.")
        if k <= 0:
            raise ValueError("k must be positive")
        N, Tp = self.last_src_shape
        w = self.last_src_weights  # (B,L_src)
        L_src = N * Tp
        if w.shape[1] != L_src:
            raise RuntimeError(f"Unexpected src weight length: got {w.shape[1]}, expected {L_src}")

        topw, topi = torch.topk(w, k=min(k, L_src), dim=1)
        station = topi // Tp
        tprime = topi % Tp
        return station.long(), tprime.long(), topw

    @staticmethod
    def _trend_label(delta: float, mean_val: float) -> str:
        # Relative threshold to avoid labeling tiny noise as trend.
        thr = max(1e-6, 0.05 * abs(mean_val))
        if delta > thr:
            return "increasing"
        if delta < -thr:
            return "decreasing"
        return "stable"

    def build_key_evidence_set(
        self,
        *,
        x_flow: torch.Tensor,
        xtimes: Optional[Sequence[Sequence[str]]] = None,
        topk: int = 10,
        stride: int = 4,
    ) -> List[List[Dict[str, Any]]]:
        """
        Build a human-readable key evidence set ε for each sample in the batch.

        This implements Eq. (4.5) with a fixed mapping from encoder time index t'
        back to the original input time steps using a stride (default=4).

        Args:
          x_flow: (B,N,F,T_in) input flows in *real units* (recommended: denormalized).
                  F should be 2 (inflow,outflow) or more; we use indices 0/1.
          xtimes: optional batch of input timestamp sequences, each length T_in.
          topk: number of key spatio-temporal locations to select.
          stride: number of original time steps represented by one encoder time step t'.

        Returns:
          A list of length B; each element is a list of evidence dicts with fields:
            station_idx, tprime_idx, t_start, t_end, inflow_desc, outflow_desc
        """
        if x_flow.ndim != 4:
            raise ValueError(f"x_flow must be (B,N,F,T_in), got {tuple(x_flow.shape)}")
        B, N, F, T_in = x_flow.shape
        if F < 2:
            raise ValueError("x_flow must have at least 2 features: inflow/outflow")
        if stride <= 0:
            raise ValueError("stride must be positive")

        station_idx, tprime_idx, weights = self.get_topk_spatiotemporal_positions(topk)
        # station_idx/tprime_idx/weights: (B,k)

        # Daily mean per sample, per station, per direction over the entire input window.
        daily_mean = x_flow[:, :, :2, :].mean(dim=-1)  # (B,N,2)

        # Normalize xtimes to batch-major if provided (reuse existing normalizer).
        xtimes_norm = None
        if xtimes is not None:
            xtimes_norm = self._normalize_time_sequences(xtimes, B, name="xtimes")

        out: List[List[Dict[str, Any]]] = []
        k_eff = int(station_idx.shape[1])
        for b in range(B):
            evidences: List[Dict[str, Any]] = []
            for r in range(k_eff):
                n = int(station_idx[b, r].item())
                tp = int(tprime_idx[b, r].item())

                t_start_idx = tp * stride
                t_end_idx = min((tp + 1) * stride - 1, T_in - 1)
                if t_start_idx > T_in - 1:
                    continue

                # Time labels (prefer timestamps if available).
                if xtimes_norm is not None and len(xtimes_norm[b]) == T_in:
                    t_start = str(xtimes_norm[b][t_start_idx])
                    t_end = str(xtimes_norm[b][t_end_idx])
                else:
                    t_start = f"t={t_start_idx}"
                    t_end = f"t={t_end_idx}"

                def describe(direction: int) -> Dict[str, Any]:
                    seg = x_flow[b, n, direction, t_start_idx : t_end_idx + 1]
                    seg_mean = float(seg.mean().item())
                    seg_delta = float((seg[-1] - seg[0]).item())
                    seg_trend = self._trend_label(seg_delta, seg_mean)
                    base = float(daily_mean[b, n, direction].item())
                    above = seg_mean > base
                    return {
                        "mean": seg_mean,
                        "trend": seg_trend,
                        "above_daily_mean": bool(above),
                        "daily_mean": base,
                    }

                inflow_stats = describe(0)
                outflow_stats = describe(1)

                evidences.append(
                    {
                        "station_idx": n,
                        "tprime_idx": tp,
                        "t_start": t_start,
                        "t_end": t_end,
                        "weight": float(weights[b, r].item()),
                        "inflow": inflow_stats,
                        "outflow": outflow_stats,
                    }
                )
            out.append(evidences)
        return out


class STALLMForecasterNoLLM(nn.Module):
    """
    Ablation: w/o LLM (and w/o SDP).
    STA features -> mean pool -> Linear -> RegressionHead -> y_hat.
    Forward accepts the same kwargs as STALLMForecaster but ignores prompt/timestamps.
    """

    def __init__(
        self,
        *,
        sta: STA,
        num_nodes: int,
        out_steps: int,
        out_features: int = 2,
        mlp_hidden: int = 1024,
        feat_dim: int,
        d_model: int = 2048,
        use_linear_head: bool = False,
    ):
        super().__init__()
        self.sta = sta
        self.num_nodes = num_nodes
        self.out_steps = out_steps
        self.out_features = out_features
        self.fusion = nn.Linear(feat_dim, d_model)
        out_dim = num_nodes * out_features * out_steps
        if use_linear_head:
            self.reg = LinearRegressionHead(d_model=d_model, out_dim=out_dim)
        else:
            self.reg = RegressionHead(d_model=d_model, hidden=mlp_hidden, out_dim=out_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        dataset_name: Optional[str] = None,
        in_steps: Optional[int] = None,
        out_steps: Optional[int] = None,
        timestamps: Optional[Sequence[str]] = None,
        xtimes: Optional[Sequence[Sequence[str]]] = None,
        ytimes: Optional[Sequence[Sequence[str]]] = None,
        weather_texts: Optional[Sequence[Optional[str]]] = None,
    ) -> torch.Tensor:
        # (B,N,F,T_in) -> (B,N,F',T') -> mean over N,T' -> (B,F') -> fusion -> (B,d_model) -> reg -> (B,N*2*T_out)
        feats = self.sta.forward_features(x)  # (B,N,F',T')
        pooled = feats.mean(dim=(1, 3))  # (B,F')
        h = self.fusion(pooled)  # (B,d_model)
        out = self.reg(h)  # (B, N*out_features*out_steps)
        B = x.shape[0]
        return out.view(B, self.num_nodes, self.out_features, self.out_steps)


class STALLMForecasterNoSDP(nn.Module):
    """
    Ablation: w/o SDP.

    STA (or temporal encoder) -> forward_features -> mean pool over (N,T')
    -> Linear(F'->d_model) -> repeated as prefix_len evidence tokens
    -> LLM + RegressionHead -> y_hat.

    与 STALLMForecaster 共享相同的 prompt/LLM/回归头部分，仅改变前缀构造方式。
    """

    def __init__(
        self,
        *,
        sta: nn.Module,
        llm: nn.Module,
        tokenizer,
        num_nodes: int,
        out_steps: int,
        out_features: int = 2,
        mlp_hidden: int = 1024,
        feat_dim: int,
        prefix_len: int,
        use_linear_head: bool = False,
    ):
        super().__init__()
        if not hasattr(sta, "forward_features"):
            raise TypeError("sta must implement forward_features(x) -> (B,N,F',T')")
        self.sta = sta
        self.llm = llm
        self.tokenizer = tokenizer
        self.num_nodes = num_nodes
        self.out_steps = out_steps
        self.out_features = out_features
        self.prefix_len = prefix_len

        if not hasattr(self.llm, "config") or not hasattr(self.llm.config, "hidden_size"):
            raise ValueError("Cannot infer LLM hidden size; expected llm.config.hidden_size")
        d_model = int(self.llm.config.hidden_size)

        # F' -> d_model, then expand to prefix_len tokens
        self.fusion = nn.Linear(feat_dim, d_model)
        out_dim = num_nodes * out_features * out_steps
        if use_linear_head:
            self.reg = LinearRegressionHead(d_model=d_model, out_dim=out_dim)
        else:
            self.reg = RegressionHead(d_model=d_model, hidden=mlp_hidden, out_dim=out_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        dataset_name: str,
        in_steps: int,
        out_steps: int,
        timestamps: Optional[Sequence[str]] = None,
        xtimes: Optional[Sequence[Sequence[str]]] = None,
        ytimes: Optional[Sequence[Sequence[str]]] = None,
        weather_texts: Optional[Sequence[Optional[str]]] = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        if out_steps != self.out_steps:
            raise ValueError(f"out_steps mismatch: got {out_steps}, model out_steps={self.out_steps}")
        if timestamps is not None and len(timestamps) != B:
            raise ValueError(f"timestamps must have length B={B}, got {len(timestamps)}")

        xtimes = STALLMForecaster._normalize_time_sequences(xtimes, B, name="xtimes")
        ytimes = STALLMForecaster._normalize_time_sequences(ytimes, B, name="ytimes")
        weather_texts = STALLMForecaster._normalize_optional_strings(weather_texts, B, name="weather_texts")

        # (B,N,F',T') from encoder, then mean over (N,T') -> (B,F')
        feats = self.sta.forward_features(x)
        pooled = feats.mean(dim=(1, 3))  # (B,F')
        h = self.fusion(pooled)  # (B,d_model)
        H_prefix = h.unsqueeze(1).expand(B, self.prefix_len, -1)  # (B,prefix_len,d_model)

        # 构造解释/预测 prompt，与完整模型共用模板。
        prompt_cfg = PromptConfig(dataset_name=dataset_name, in_steps=in_steps, out_steps=out_steps, num_nodes=self.num_nodes)
        prompts: List[str] = []
        for i in range(B):
            x_seq = xtimes[i] if xtimes is not None else None
            y_seq = ytimes[i] if ytimes is not None else None
            weather_text = weather_texts[i] if weather_texts is not None else None
            forecast_start = y_seq[0] if y_seq is not None else (timestamps[i] if timestamps is not None else None)
            if forecast_start is None:
                raise ValueError("Provide either ytimes or timestamps for prompt construction")
            prompts.append(
                build_prompt(
                    prompt_cfg,
                    forecast_start=forecast_start,
                    xtimes=x_seq,
                    ytimes=y_seq,
                    weather_text=weather_text,
                )
            )

        tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        input_ids = tok["input_ids"].to(x.device)
        attn_text = tok["attention_mask"].to(x.device)  # (B,L_text)

        text_emb = self.llm.get_input_embeddings()(input_ids)  # (B,L_text,d_model)
        inputs_embeds = torch.cat([H_prefix, text_emb], dim=1)

        attn_prefix = torch.ones(B, self.prefix_len, device=x.device, dtype=attn_text.dtype)
        attention_mask = torch.cat([attn_prefix, attn_text], dim=1)  # (B,L_total)

        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
        hidden = out.last_hidden_state  # (B,L_total,d_model)

        # [PRED] 位置与完整模型一致：文本部分最后一个非 pad token。
        last_text_index = attn_text.sum(dim=1) - 1  # (B,)
        pred_index = self.prefix_len + last_text_index  # (B,)
        o_pred = hidden[torch.arange(B, device=x.device), pred_index]  # (B,d_model)

        y = self.reg(o_pred)  # (B, N*out_features*out_steps)
        y = y.view(B, self.num_nodes, self.out_features, self.out_steps)
        return y

def load_llama7b(model_name_or_path: str, *, device: str | torch.device = "cpu"):
    if AutoModel is None or AutoTokenizer is None:
        raise ImportError("transformers is required to load LLaMA models")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # add [PRED] token
    if PRED_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [PRED_TOKEN]})

    llm = AutoModel.from_pretrained(model_name_or_path)
    llm.resize_token_embeddings(len(tokenizer))
    llm.to(device)
    return llm, tokenizer


def apply_lora(
    llm: nn.Module,
    *,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[Sequence[str]] = None,
):
    """
    Apply LoRA to a LLaMA-like backbone using PEFT.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as e:  # pragma: no cover
        raise ImportError("peft is required for LoRA fine-tuning") from e

    if target_modules is None:
        # Common projection modules for LLaMA-family models.
        target_modules = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )

    # Prefer feature extraction; fall back if older PEFT doesn't have it.
    task_type = getattr(TaskType, "FEATURE_EXTRACTION", TaskType.CAUSAL_LM)

    cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=task_type,
        target_modules=list(target_modules),
    )
    return get_peft_model(llm, cfg)


def load_llama_local(
    local_path: str,
    *,
    device: str | torch.device = "cpu",
):
    """
    Load a local LLaMA-family model (e.g., Llama-3.2-1B) and tokenizer, adding [PRED].
    """
    if AutoModel is None or AutoTokenizer is None:
        raise ImportError("transformers is required to load LLaMA models")
    tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if PRED_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [PRED_TOKEN]})

    llm = AutoModel.from_pretrained(local_path)
    llm.resize_token_embeddings(len(tokenizer))
    llm.to(device)
    return llm, tokenizer


def make_tiny_llama_for_test(*, vocab_size: int = 256, d_model: int = 64, n_layers: int = 2):
    """
    Create a tiny LLaMA-like model for local shape tests (no checkpoint download).
    """
    if LlamaConfig is None or LlamaModel is None:
        raise ImportError("transformers is required for tiny llama test model")
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=d_model,
        intermediate_size=d_model * 4,
        num_hidden_layers=n_layers,
        num_attention_heads=max(1, d_model // 16),
        max_position_embeddings=2048,
    )
    return LlamaModel(cfg)


__all__ = [
    "PRED_TOKEN",
    "STALLMForecaster",
    "STALLMForecasterNoLLM",
    "STALLMForecasterNoSDP",
    "load_llama7b",
    "load_llama_local",
    "apply_lora",
    "make_tiny_llama_for_test",
    "build_prompt",
    "PromptConfig",
]

