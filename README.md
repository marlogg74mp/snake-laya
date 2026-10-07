# Stage 4: Human-vs-AI Duel & Convai Laya 421M Tournament

## Overview
This stage builds a symmetric 3D duel arena (human player vs Laya) and tests the official 421M parameter `Convai Laya` (ModernBERT-large) in zero-shot and fine-tuned modes against our 149M baseline.

## Key Files in this Branch
* `versus_server.py` — Local HTTP server for the duel arena (port 9700).
* `VERSUS_RULES.md` — Complete rules of the duel mode (fog of war, scores, kills).
* `laya_convai_snake.py` — Training and inference pipeline for Convai Laya 421M.
* `run_laya_compare.py` / `versus/tournament.mjs` — Head-to-head tournament runner (20 matches across 10 maps).
* `export_laya421_web.py` — Web export for Laya 421M.
* `versus/` — 3D duel arena web client (chase camera, mobile touch controls, fog of war).

## Step-by-Step Execution Guide

### 1. Launch Duel Arena
```bash
python versus_server.py
```
Open `http://127.0.0.1:9700/` to duel against the AI in real time.

### 2. Fine-tune Convai Laya 421M
```bash
python laya_convai_snake.py train
```
Fine-tunes the 421M parameter model using its native `[MASK]` classification head.

### 3. Run AI-vs-AI Tournament
```bash
python run_laya_compare.py
```
Plays 20 tournament matches between 149M v10 and Laya 421M. Result: 16:4 in favor of 421M.

## Key Insights
* **Zero-Shot Cowardice:** Untrained Laya 421M with prompt *"pick safest move"* lived the longest (165 steps) but scored only 0.2 points — running safely in circles!
* **Multiplayer Zero-Shot Adaptation:** The opponent's snake body was passed as a static wall to the neural net, allowing multiplayer duels without any multiplayer retraining.
* **Next Stage:** Switch to `stage-5-window-memory` to implement egocentric window vision and multi-apple routing.
