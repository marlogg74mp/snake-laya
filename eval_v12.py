"""
Multi-apple worlds (build_dataset_v12.World12): v11w (window, no apple values: it is shown the nearest known apple)
vs v12 (window + values + memory) vs the teacher, same episodes. The point of V12 is the score.

Usage: python eval_v12.py [episodes] [sizes...]      e.g.  python eval_v12.py 40 20 30
"""

import json
import os
import random
import sys
import time
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import build_dataset_v12 as v12
import build_dataset_v11w as w11
import build_dataset_v9 as v9
from a_star_solver import SnakeAStarSolver

ROOT = os.path.dirname(os.path.abspath(__file__))
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
solver = SnakeAStarSolver()

class Model:
    def __init__(self, model_dir, kind):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()
        self.keys = json.load(open(os.path.join(model_dir, "state_schema.json")))["keys"]
        self.kind = kind

    @torch.no_grad()
    def probs(self, state):
        text = f"State: {json.dumps({k: state[k] for k in self.keys})}\nQuestion: {v9.QUESTION}\nChoices: UP, DOWN, LEFT, RIGHT"
        enc = self.tok(text, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        return dict(zip(ACTIONS, torch.softmax(self.model(**enc).logits, -1)[0].tolist()))

def state_for(policy, w, mem12, mem11):
    if policy.kind == "v12":
        return v12.state12(w, mem12)
    # v11w knows one food: show it the nearest item it could know about
    saved = w.food
    w.food = v12.nearest_known(w, mem12)
    try:
        return w11.window_state(w, mem11)
    finally:
        w.food = saved

def episode(policy, n, seed, eaten, max_steps=300):
    rng = random.Random(seed)
    random.seed(seed)
    w = None
    while w is None:
        w = v12.make_world(rng, n)
    mem12, mem11 = v12.Memory12(), w11.Memory()
    mem12.update(w)
    for t in range(max_steps):
        if policy == "teacher":
            action, _ = v12.teacher12(solver, w, mem12)
        else:
            state = state_for(policy, w, mem12, mem11)
            danger = v12.state12(w, mem12)  # the same safety mask for everyone (incl. poison)
            p = policy.probs(state)
            legal = [a for a in ACTIONS if not danger[f"danger_{a}"] and not (len(w.body) > 1 and a == v9.OPP[w.direction])]
            action = max(legal or ACTIONS, key=p.get)
        before = dict(w.items)
        immortal = w.immortal_steps > 0
        r = w.world_step(action)
        for c, (kind, _) in before.items():
            if c not in w.items and c == w.head:
                eaten[kind] += 1
        if r == "dead":
            drone = any(d.pos in set(w.body) or d.pos == w.head for d in w.drones) and not immortal
            return t + 1, w.score, "drone" if drone else "other"
        if r == "win":
            break
        mem12.update(w)
        if policy != "teacher" and policy.kind == "v11w":
            mem11.update(w)
    return max_steps, w.score, "survived"

def run(policy, name, n, episodes):
    t0 = time.time()
    eaten = Counter()
    res = [episode(policy, n, 150_000 + n * 1000 + i, eaten) for i in range(episodes)]
    ends = Counter(e for _, _, e in res)
    k = len(res)
    total = sum(eaten.values()) or 1
    print(f"{n}x{n} {name:8} | avg score {sum(r[1] for r in res) / k:5.1f} | avg steps {sum(r[0] for r in res) / k:4.0f} | "
          f"survived {ends['survived'] / k:4.0%} | drone deaths {ends['drone'] / k:4.0%} | other deaths {ends['other'] / k:4.0%} | "
          f"eaten: golden {eaten['golden'] / total:4.0%} apple {eaten['apple'] / total:4.0%} immortal {eaten['immortal'] / total:4.0%} "
          f"shrink {eaten['shrink'] / total:4.0%} | {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    sizes = [int(x) for x in sys.argv[2:]] or [20, 30]
    m11 = Model(os.path.join(ROOT, "laya_snake_weights_v11w"), "v11w")
    m12 = Model(os.path.join(ROOT, "laya_snake_weights_v12"), "v12")
    for n in sizes:
        run(m11, "v11w", n, episodes)
        run(m12, "v12", n, episodes)
        run("teacher", "teacher", n, episodes)
