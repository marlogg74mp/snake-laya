# Snake played by a text encoder

Can a language model that reads JSON play Snake? This repository teaches a small text encoder
(ModernBERT-base, 149M parameters, in the spirit of [Convai's Laya](https://huggingface.co/convaiinnovations/laya)
and the closed [Jev](https://openrouter.ai/) decision models) to pick `UP / DOWN / LEFT / RIGHT` from a game state
written as text. It then plays against humans in a 3D browser duel, entirely client-side via ONNX Runtime Web.

```
State: {"current_dir": "RIGHT", "danger_UP": false, ..., "drones": [[-3, 2, 1, 0]], "walls": [[2, -1]], ...}
Question: What is the next safe move avoiding patrols and walls towards food?
Choices: UP, DOWN, LEFT, RIGHT                         ->  classifier  ->  {"UP": 0.02, "RIGHT": 0.95, ...}
```

**Play in the browser:** `https://<your-user>.github.io/<repo>/` (duel) and `.../solo/` (single player).
No server: the model (~70 MB, int4) is downloaded once and runs on your GPU (WebGPU) or CPU (WASM).

## What is inside

| | |
|---|---|
| Game | Snake with patrolling drones, walls that periodically turn transparent, portals, fog of war, immortality apples, poison |
| Teacher | A* + time-space look-ahead that dodges drones (`build_dataset_v9.py`) |
| Student | ModernBERT-base fine-tuned as a 4-class classifier on teacher moves (imitation learning) |
| DAgger | the model plays, the teacher labels the states it reached (`dagger_v10.py`) |
| Window model | egocentric 11×11 view + memory, works on any board size (`build_dataset_v11w.py`) |
| Safety shield | the model ranks moves, a 20-step look-ahead in a simulator vetoes moves into traps (`eval_shield.py`, `versus/js/shield.js`) |
| Browser | int4 ONNX with a pruned vocabulary (50 280 → ~300 tokens), onnxruntime-web, three.js |
| Real Laya | Convai Laya 421M fine-tuned on the same data for comparison (`laya_convai_snake.py`) |

## Results (100 episodes, multi-apple worlds, same seeds for every player)

| 20×20 board | score | steps survived | survived to the end | killed by a drone |
|---|---|---|---|---|
| v11w (model only) | 15.8 | 145 | 17% | 70% |
| **v11w + shield** | **23.1** | **257** | **75%** | **24%** |
| A* teacher | 22.1 | 236 | 55% | 44% |

The shield overrides the model in only ~3% of moves. More numbers: [`results/`](results/).

## Stages = branches

Each branch adds the scripts of one stage on top of the previous one, so you can rebuild the project step by step.
The web code is the final version in every branch where it appears.

| branch | adds | build it |
|---|---|---|
| `stage-1-text-state` | env, A* teacher, datasets v3–v7, trainer, model server | `python build_dataset_v7_patrols.py` → `python train_v7_full_context.py --dataset dataset_v7_patrols.jsonl --out laya_snake_weights` |
| `stage-2-drones-v9` | drones, phase walls, immortality, look-ahead teacher, 3D UI | `python build_dataset_v9.py 60000` → train with `--dataset dataset_v9.jsonl --out laya_snake_weights_v9` → `python eval_v9.py` |
| `stage-3-dagger-onnx` | DAgger, board-size test, browser export | `python dagger_v10.py 40000 laya_snake_weights_v9` → train `--init laya_snake_weights_v9 --dataset dataset_v10.jsonl --out laya_snake_weights_v10` → `python export_onnx_web.py laya_snake_weights_v10` |
| `stage-4-duel-laya421` | human-vs-AI duel, Convai Laya 421M | `python versus_server.py` (open http://127.0.0.1:9700) · `python laya_convai_snake.py train` |
| `stage-5-window-memory` | window + memory (v11w), apple values (v12/v13) | `python run_v11w.py` · `python run_v12.py` · `python run_v13.py` |
| `main` | safety shield, results, GitHub Pages | `python eval_shield.py 100 20 30` |

The `run_*.py` scripts chain dataset → training → evaluation and write a log (`run_v11w.log`, …).
Long runs: a ~45 min fine-tune per model on one consumer GPU (bf16, batch 16 × 4 accumulation, ~7.5 GB VRAM).

## Quick start

```bash
pip install -r requirements.txt
```

Play locally (no model server needed, the models run in the browser):

```bash
python versus_server.py
```

then open http://127.0.0.1:9700 and choose **v11w · БРАУЗЕР** (the UI is in Russian; the menu buttons pick the
model, the speed and the shield).

Serve the trained PyTorch models instead (GPU): `python web_server.py` (port 9500) next to `versus_server.py`.
`web_server.py` can also call the hosted Jev model through OpenRouter; set `OPENROUTER_API_KEY` for that.

## Weights

Trained weights are not in git (≈600 MB each). Get them from the release page / the model hub link in the
release notes, or train them with the stage scripts. Expected folders: `laya_snake_weights_v10/`,
`laya_snake_weights_v11w/`, … (each with `state_schema.json`: the JSON key order the model was trained on).

The browser builds of v10 (`web3d/model/`) and v11w (`versus/model_v11w/`) are in the repository.
The Laya 421M browser build (226 MB) is too large for GitHub; without it the duel hides that option.
To use it, run `python export_laya421_web.py` (needs the Convai Laya checkpoint in `laya_convai/` and the
fine-tuned `laya_convai_snake/`).

## Lessons learned

1. **The input format is part of the model.** Key order and spellings (`"NONE"` vs `"UNKNOWN"`) must match training exactly — `state_schema.json` is saved next to the weights.
2. **A model cannot learn what it cannot see.** Absolute full-board JSON breaks on larger boards (it does not fit into 512 tokens); an egocentric window does not.
3. **Imitation learning compounds errors; DAgger fixes much of it.**
4. **A task-specific vocabulary is tiny:** pruning 50k tokens to ~300 plus int4 gives a ~70 MB browser model.
5. **Find where the model actually loses.** Better apple choice (v12, v13) did not help — the model already scores as much per step as the teacher; it just dies to drones.
6. **Intuition + verification.** A cheap look-ahead shield over the network beats both halves alone (AlphaZero in miniature).

## Credits

* [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) (Answer.AI, LightOn)
* [Laya](https://huggingface.co/convaiinnovations/laya) by Convai Innovations (Apache 2.0)
* DAgger: Ross, Gordon, Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, AISTATS 2011
* Safety shields: Alshiekh et al., *Safe Reinforcement Learning via Shielding*, AAAI 2018
* [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/), [transformers.js](https://github.com/huggingface/transformers.js), [three.js](https://threejs.org/)

## License

MIT, see [LICENSE](LICENSE). Model weights derived from ModernBERT (Apache 2.0) and Laya (Apache 2.0) keep their licenses.
