# Stage 5: Egocentric Window Vision (v11w) & Strategic Routing

## Overview
This stage implements an **egocentric 11x11 sliding window** (`v11w`) to eliminate board-size dependency. It also explores multi-apple strategic planning (`v12`, `v13`) and metrics decomposition.

## Key Files in this Branch
* `build_dataset_v11w.py` / `run_v11w.py` / `eval_window.py` — 11x11 window centered on snake's head with relative offsets.
* `build_dataset_v12.py` / `run_v12.py` / `eval_v12.py` — Valued apples (+1 regular, +3 golden, +2 star).
* `teacher_route.py` / `build_dataset_v13.py` / `run_v13.py` / `eval_v13.py` — Multi-apple routing teacher.
* `versus/model_v11w/` — In-browser INT4 ONNX model for `v11w`.

## Step-by-Step Execution Guide

### 1. Train Egocentric Window Model (v11w)
```bash
python run_v11w.py
```
Generates egocentric dataset, trains ModernBERT, and exports INT4 web weights to `versus/model_v11w/`.

### 2. Verify Grid-Size Generalization
```bash
python eval_window.py
```
Tests on 15x15, 20x20, 25x25, and 50x50 grids: 0% boundary crashes across all board sizes!

### 3. Experiments with Valued Apples (v12 / v13)
```bash
python run_v12.py
python run_v13.py
```

## Key Insights & Failure Modes (Lesson 5)
* **The Optimization Blind Alley:** Versions v12 and v13 failed to improve performance over v11w.
* **Metric Decomposition:** Decomposing score showed points-per-step were already higher than the teacher (0.109 vs 0.094)! The real bottleneck was that 70% of deaths were drone strikes at step ~145. Optimizing food collection was polishing something that was not broken.
* **Next Stage:** Switch to `main` for the ultimate breakthrough: the hybrid Safety Shield.
