"""
Evaluation under game rules V9 (RULES_V9.md):
1. closed loop: the model drives the snake in build_dataset_v9.World (moves deadly on the next cell are
   masked, like the web UI) — survival, causes of death, score; the teacher plays the same episodes;
2. imitation: accuracy on ordinary moves vs drone-dodge moves (teacher deviates from plain A*).

Usage: python eval_v9.py laya_snake_weights [--episodes 40] [--teacher]
"""

import argparse
import json
import os
import random
import time
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import build_dataset_v9 as v9
from a_star_solver import SnakeAStarSolver

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
solver = SnakeAStarSolver()

class Model:
    def __init__(self, model_dir):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()
        self.keys = json.load(open(os.path.join(model_dir, "state_schema.json")))["keys"]

    @torch.no_grad()
    def probs_batch(self, states):
        texts = [f"State: {json.dumps({k: s[k] for k in self.keys})}\nQuestion: {v9.QUESTION}\n"
                 f"Choices: UP, DOWN, LEFT, RIGHT" for s in states]
        enc = self.tok(texts, truncation=True, max_length=448, padding=True, return_tensors="pt").to("cuda")
        p = torch.softmax(self.model(**enc).logits, dim=-1).tolist()
        return [dict(zip(ACTIONS, row)) for row in p]

def episode(policy, seed, max_steps=300):
    rng = random.Random(seed)
    random.seed(seed)
    w = None
    while w is None:
        w = v9.new_world(rng)
    for t in range(max_steps):
        state = w.model_state()
        if policy == "teacher":
            action = v9.teacher_move(solver, w)
        else:
            p = policy.probs_batch([state])[0]
            legal = [a for a in ACTIONS if not state[f"danger_{a}"]
                     and not (len(w.body) > 1 and a == v9.OPP[w.direction])]
            action = max(legal or ACTIONS, key=p.get)
        immortal = w.immortal_steps > 0
        r = w.world_step(action)
        if r == "dead":
            hx, hy = w.head
            cause = "drone" if any(d.pos in set(w.body) or d.pos == (hx, hy) for d in w.drones) and not immortal \
                else "wall/body/border"
            return {"steps": t + 1, "score": w.score, "end": cause}
        if r == "win":
            break
    return {"steps": max_steps, "score": w.score, "end": "survived"}

def closed_loop(policy, name, episodes):
    t0 = time.time()
    res = [episode(policy, 50_000 + i) for i in range(episodes)]
    ends = Counter(r["end"] for r in res)
    n = len(res)
    print(f"{name:12} | survived 300: {ends['survived'] / n:4.0%} | drone deaths: {ends['drone'] / n:4.0%} | "
          f"other deaths: {ends['wall/body/border'] / n:4.0%} | avg steps {sum(r['steps'] for r in res) / n:4.0f} | "
          f"avg score {sum(r['score'] for r in res) / n:5.1f} | {time.time() - t0:.0f}s", flush=True)

def imitation(model, n_samples=3000):
    rng = random.Random(77)
    random.seed(77)
    rows = []
    while len(rows) < n_samples:
        w = v9.new_world(rng)
        if w is None:
            continue
        for _ in range(150):
            base, a = v9.base_move(solver, w), v9.teacher_move(solver, w)
            rows.append((w.model_state(), a, a != base, base))
            if w.world_step(a) != "ok":
                break
    st = Counter()
    for i in range(0, len(rows), 64):
        batch = rows[i:i + 64]
        for (s, a, dodge, base), p in zip(batch, model.probs_batch([r[0] for r in batch])):
            k = "dodge" if dodge else "normal"
            pred = max(p, key=p.get)
            st[k, "n"] += 1
            st[k, "ok"] += pred == a
            st[k, "base"] += pred == base
            k2 = "immortal" if s["immortal_steps"] > 0 else None
            if k2:
                st[k2, "n"] += 1
                st[k2, "ok"] += pred == a
    for k in ("normal", "dodge", "immortal"):
        n = st[k, "n"]
        if n:
            extra = f" | picked the plain A* move: {st[k, 'base'] / n:.0%}" if k == "dodge" else ""
            print(f"imitation {k:9} n={n:5} accuracy {st[k, 'ok'] / n:6.1%}{extra}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--teacher", action="store_true", help="also run the teacher on the same episodes")
    args = ap.parse_args()
    m = Model(args.model_dir)
    imitation(m)
    closed_loop(m, "model", args.episodes)
    if args.teacher:
        closed_loop("teacher", "teacher", args.episodes)
