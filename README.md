# Stage 1: Game State as JSON & ModernBERT Imitation Learning

## Overview
Can a transformer text encoder play Snake? In this initial stage, the 20x20 Snake environment is serialized into a textual JSON document (~340 tokens). We train a `ModernBERT-base` (149M parameters) encoder as a 4-class classifier (`UP`, `DOWN`, `LEFT`, `RIGHT`) using imitation learning (Behavior Cloning) from an algorithmic A* teacher.

```
State: {"current_dir": "RIGHT", "food_dir": "DOWN_RIGHT", "head_pos": [9, 8], ...}
Output: Classification probabilities -> UP (0.02), DOWN (0.01), LEFT (0.01), RIGHT (0.96)
```

## Key Files in this Branch
* `snake_env.py` — Core 20x20 Snake environment simulation.
* `a_star_solver.py` — Algorithmic A* teacher that calculates the shortest path to food.
* `build_dataset_v7_patrols.py` — Generates pairs of `(game_state_json -> expert_move)`.
* `train_v7_full_context.py` — PyTorch training script fine-tuning ModernBERT-base.
* `eval_laya.py` — Evaluates model classification accuracy on hold-out test sets.
* `web_server.py` — FastAPI model server (port 9500) serving live move predictions.
* `EXPERIMENT_LOG.md` — Detailed research logs from early experiments.

## Step-by-Step Execution Guide

### 1. Generate Dataset
```bash
python build_dataset_v7_patrols.py
```
Generates `dataset_v7_patrols.jsonl`.

### 2. Train ModernBERT-base
```bash
python train_v7_full_context.py --dataset dataset_v7_patrols.jsonl --out laya_snake_weights --epochs 3 --batch-size 16
```
Trained weights will be saved to `laya_snake_weights/` along with `state_schema.json`.

### 3. Evaluate Offline Accuracy
```bash
python eval_laya.py --weights laya_snake_weights
```
Validation accuracy: **~98.2%**.

### 4. Serve Model for Game Interface
```bash
python web_server.py
```
Listens on `http://127.0.0.1:9500/predict_action`.

## Key Insights & Failure Modes (Lesson 1)
* **The Loop in the Corner:** Despite 98% accuracy on static test data, in the live game the snake looped indefinitely near the wall.
* **Root Cause:** Key order mismatch in JSON serialization between JavaScript and Python, and `"NONE"` vs `"UNKNOWN"` string representations.
* **Fix:** Introduced `state_schema.json` to strictly enforce byte-identical serialization across Python and JavaScript.
* **Next Stage:** Switch to `stage-2-drones-v9` to test the model against dynamic obstacles (patrol drones, phase walls, portals).
