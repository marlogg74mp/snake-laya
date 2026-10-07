# Stage 3: DAgger & In-Browser INT4 ONNX WebGPU

## Overview
In this stage, we solve the covariate shift problem (cascading errors) using **DAgger (Dataset Aggregation)**. We also test grid-size generalization and quantize the 149M model down to 71 MB for zero-server in-browser WebGPU execution.

## Key Files in this Branch
* `dagger_v10.py` — DAgger loop: model plays games, teacher labels states visited by the model.
* `run_v11.py` — Retraining pipeline for aggregated DAgger dataset.
* `export_onnx.py` / `export_onnx_web.py` — Vocabulary pruning (50,280 -> 296 tokens) + block INT4 quantization.
* `eval_grid_size.py` — Evaluates generalization on 15x15, 20x20, and 25x25 grids.
* `web3d/model/` — Pre-compiled client-side INT4 ONNX model (`laya_web.onnx`, ~71 MB).

## Step-by-Step Execution Guide

### 1. Collect DAgger Data (Model Mistakes)
```bash
python dagger_v10.py 40000 laya_snake_weights_v9
```
Runs 64 parallel environments, collects states where the model made mistakes, and labels them with teacher moves into `dataset_v10.jsonl`.

### 2. Train Model v10
```bash
python train_v7_full_context.py --init laya_snake_weights_v9 --dataset dataset_v10.jsonl --out laya_snake_weights_v10
```

### 3. Export to WebGPU INT4 ONNX
```bash
python export_onnx_web.py laya_snake_weights_v10 web3d/model/laya_web.onnx
```
Prunes embedding table to 296 tokens and applies block-32 INT4 quantization (599 MB -> 71 MB).

### 4. Test Board Generalization
```bash
python eval_grid_size.py
```

### 5. Play Serverless in Browser
Open `web3d/index.html` directly in Chrome or Edge — model runs locally on WebGPU!

## Key Insights & Failure Modes (Lesson 3 & 4)
* **DAgger tripled survival:** Survival steps grew from 45 to 127 steps, preventing cascading failure loops.
* **Vocabulary Pruning:** Game JSON only needs 296 unique tokens out of 50k in ModernBERT. Pruning saved tens of megabytes.
* **Next Stage:** Switch to `stage-4-duel-laya421` for Human-vs-AI duels and testing Convai Laya 421M.
