# Stage 2: Dynamic Threats — Drones, Phase Walls & 3D Web Arena

## Overview
This stage introduces complex dynamic hazards: patrolling drones, periodic phase walls (permeable every N ticks), portals, and temporary immortality apples. It also introduces a real-time 3D web arena built with Three.js.

## Key Files in this Branch
* `build_dataset_v8_patrol_body.py` — Adds snake body segments to state JSON (fixing body blindness).
* `build_dataset_v9.py` — Generates 60,000 steps with time-space look-ahead A* teacher dodging drones.
* `eval_v9.py` / `eval_patrol_survival.py` — Evaluates survival against drone attacks.
* `RULES_V9.md` — Complete formal rules of the v9 game mechanics.
* `web3d/` — Full 3D web client (Three.js, cyberpunk shaders, real-time WebSocket).

## Step-by-Step Execution Guide

### 1. Generate Dataset with Drone Trajectories
```bash
python build_dataset_v9.py 60000
```
Outputs `dataset_v9.jsonl` with drone flight paths and snake body coordinates.

### 2. Fine-tune ModernBERT (v9)
```bash
python train_v7_full_context.py --dataset dataset_v9.jsonl --out laya_snake_weights_v9 --epochs 3
```

### 3. Evaluate Drone Survival
```bash
python eval_v9.py
```

### 4. Play in 3D Web Arena
1. Start server: `python web_server.py`
2. Open `web3d/index.html` in your browser.

## Key Insights & Failure Modes (Lesson 2)
* **The Body Blindness Trap:** Drones collided into the snake's body/tail 100% of the time because body coordinates were originally omitted from the JSON.
* **Imbalance Masking:** Overall test accuracy was 91.7%, but drone evasion accuracy was only 26%. Adding full body coordinates and 4x oversampling of evasion scenarios boosted evasions to 54%.
* **Next Stage:** Switch to `stage-3-dagger-onnx` to tackle cascading errors (covariate shift) and browser deployment.
