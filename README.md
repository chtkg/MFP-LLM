# TRACE: Traceable Reasoning with Aligned Chain of Evidence for Metro Flow Prediction


## Overview

TRACE makes LLM-based metro flow predictions verifiable by decomposing every forecast into a set of traceable spatio-temporal points. The framework consists of five stages:

1. **Spatio-Temporal Encoding** — Attention-based encoder with Chebyshev graph convolution
2. **Feature-to-Evidence Mapping** — Prefix Pooling + Soft Dictionary Projection (SDP)
3. **LLM-based Prediction** — Frozen LLaMA-2-7B with LoRA fine-tuning + MLP regression head
4. **Evidence Back-Projection** — Quantifying and tracing each evidence token's contribution to original station-time positions
5. **Explanation Generation** — Structured prompt template for causally grounded natural language explanations


## Datasets

| Dataset | Stations | Duration |
|---------|----------|----------|
| Hangzhou Metro (HZMetro) | 80 | Jan 1–25, 2019 |
| Shanghai Metro (SHMetro) | 288 | Jul 1–Sep 30, 2016 |

**Download:** [Baidu Netdisk](https://pan.baidu.com/s/1lesAk4WOfBQtg0a0XgDfvA) (Extraction code: `np5p`)

Place the downloaded data in `./data/` directory.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA 11.8+ (RTX 4090 recommended, 24GB VRAM)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation
Place the downloaded datasets in `./data/`.

### 2. Training
```bash
python run.py --dataset hzmetro --config configs/default.yaml
```

### 3. Evaluation
```bash
python run.py --dataset hzmetro --mode eval --checkpoint ./checkpoints/best.pt
```

### 4. Zero-Shot Transfer
```bash
python run.py --dataset shmetro --mode eval --checkpoint ./checkpoints/hzmetro_best.pt
```

## Reproducing Results

To reproduce all experimental results:

```bash
bash scripts/reproduce_all.sh
```

This script runs the following experiments sequentially:
- Overall Performance (Hangzhou & Shanghai)
- Ablation Study
- Zero-Shot Forecasting
- Explanation Quality Evaluation

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| LoRA rank | 8 |
| Learning rate | 2e-4 |
| Input length | 4 steps (1 hour) |
| Prefix tokens | 64 |
| LLM backbone | LLaMA-2-7B |



