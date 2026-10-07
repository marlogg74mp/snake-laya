"""
Board-size test: v10 (full-board absolute state) vs v11w (egocentric window + memory) vs the teacher,
on the same episodes (build_dataset_v11w.make_world: V9 rules, no portals, drones/walls scaled to the board).

Usage: python eval_window.py [episodes] [sizes...]      e.g.  python eval_window.py 40 15 20 25 50
"""

import json
import os
import random
import sys
import time
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import build_dataset_v11w as w11
import build_dataset_v9 as v9
from a_star_solver import SnakeAStarSolver

ROOT = os.path.dirname(os.path.abspath(__file__))
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
solver = SnakeAStarSolver()

class Model:
    def __init__(self, model_dir, window):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()
        self.keys = json.load(open(os.path.join(model_dir, "state_schema.json")))["keys"]
        self.window = window
        self.long = 0
        self.calls = 0

    @torch.no_grad()
    def probs(self, state):
        text = f"State: {json.dumps({k: state[k] for k in self.keys})}\nQuestion: {v9.QUESTION}\nChoices: UP, DOWN, LEFT, RIGHT"
        enc = self.tok(text, truncation=True, max_length=448, return_tensors="pt").to("cuda")
        self.calls += 1
        self.long += len(self.tok(text)["input_ids"]) > 448
        return dict(zip(ACTIONS, torch.softmax(self.model(**enc).logits, -1)[0].tolist()))

def episode(policy, n, seed, max_steps=300):
    rng = random.Random(seed)
    random.seed(seed)
    w = None
    while w is None:
        w = w11.make_world(rng, n)
    mem = w11.Memory()
    mem.update(w)
    for t in range(max_steps):
        if policy == "teacher":
            action = w11.teacher_move(solver, w, mem)
        else:
            full = w.model_state()
            state = w11.window_state(w, mem) if policy.window else full
            p = policy.probs(state)
            legal = [a for a in ACTIONS if not full[f"danger_{a}"] and not (len(w.body) > 1 and a == v9.OPP[w.direction])]
            action = max(legal or ACTIONS, key=p.get)
        immortal = w.immortal_steps > 0
        r = w.world_step(action)
        if r == "dead":
            drone = any(d.pos in set(w.body) or d.pos == w.head for d in w.drones) and not immortal
            return t + 1, w.score, "drone" if drone else "other"
        if r == "win":
            break
        mem.update(w)
    return max_steps, w.score, "survived"

def run(policy, name, n, episodes):
    t0 = time.time()
    res = [episode(policy, n, 120_000 + n * 1000 + i) for i in range(episodes)]
    ends = Counter(e for _, _, e in res)
    k = len(res)
    long = f" | inputs over 448 tokens {policy.long / max(1, policy.calls):.1%}" if policy != "teacher" else ""
    if policy != "teacher":
        policy.long = policy.calls = 0
    print(f"{n}x{n} {name:8} | survived 300: {ends['survived'] / k:4.0%} | drone deaths {ends['drone'] / k:4.0%} | "
          f"other deaths {ends['other'] / k:4.0%} | avg steps {sum(r[0] for r in res) / k:4.0f} | "
          f"avg score {sum(r[1] for r in res) / k:5.1f}{long} | {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    sizes = [int(x) for x in sys.argv[2:]] or [15, 20, 25, 50]
    v10 = Model(os.path.join(ROOT, "laya_snake_weights_v10"), window=False)
    v11w = Model(os.path.join(ROOT, "laya_snake_weights_v11w"), window=True)
    for n in sizes:
        run(v10, "v10", n, episodes)
        run(v11w, "v11w", n, episodes)
        run("teacher", "teacher", n, episodes)
