"""
Closed-loop evaluation of a Laya model in the drone arena with the strict rule
(a drone touching ANY body segment kills the snake, as with PATROL_BODY_HITS = true in the UI).

The model drives the snake directly; like the web UI, moves that are deadly on the very next cell
(wall, body, drone's current/next cell) are masked out and the most probable remaining move is taken.

Usage: python eval_patrol_survival.py laya_snake_weights_v7 laya_snake_weights [--episodes 40]
"""

import argparse
import json
import os
import random
import time
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from snake_env import SnakeEnv
import build_dataset_v8_patrol_body as v8

ID_TO_ACTION = ["UP", "DOWN", "LEFT", "RIGHT"]
V7_KEYS = ["current_dir", "recent_actions", "food_dir", "danger_UP", "danger_DOWN", "danger_LEFT",
           "danger_RIGHT", "head_pos", "food_pos", "obstacles", "portals", "fog_of_war", "sight_radius", "patrols"]

class Model:
    def __init__(self, model_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        schema = os.path.join(model_dir, "state_schema.json")
        self.keys = json.load(open(schema))["keys"] if os.path.exists(schema) else V7_KEYS

    @torch.no_grad()
    def probs(self, full_state):
        s = {k: full_state[k] for k in self.keys}
        if "snake_len" not in self.keys:
            s["patrols"] = [p[:2] for p in s["patrols"]]
        text = f"State: {json.dumps(s)}\nQuestion: {v8.QUESTION}\nChoices: UP, DOWN, LEFT, RIGHT"
        enc = self.tok(text, truncation=True, max_length=288, return_tensors="pt").to(self.device)
        p = torch.softmax(self.model(**enc).logits, dim=-1)[0].tolist()
        return dict(zip(ID_TO_ACTION, p))

def run_episode(model, seed, max_steps=300):
    rng = random.Random(seed)
    random.seed(seed)
    patrols = v8.make_ui_patrols() if rng.random() < 0.5 else v8.make_random_patrols(rng)
    env = SnakeEnv(20, 20, initial_length=rng.randint(3, 11), num_obstacles=rng.randint(0, 3),
                   num_portals=2 if rng.random() < 0.6 else 0, fog_of_war=False, seed=rng.randrange(1 << 30))
    if any(p.pos in set(env.body) for p in patrols):
        return None
    for t in range(max_steps):
        state = v8.build_state(env, patrols)
        p = model.probs(state)
        legal = [a for a in ID_TO_ACTION
                 if not state[f"danger_{a}"] and not (len(env.body) > 1 and a == SnakeEnv.OPPOSITES[env.direction])]
        action = max(legal, key=p.get) if legal else max(p, key=p.get)
        old_head, old_pos = env.head, [d.pos for d in patrols]
        _, _, done, _ = env.step(action)
        for d in patrols:
            d.step()
        if not done and v8.patrol_hit(env, patrols, old_head, old_pos):
            return {"steps": t + 1, "score": env.score, "end": "drone"}
        if done:
            return {"steps": t + 1, "score": env.score, "end": "wall/body"}
    return {"steps": max_steps, "score": env.score, "end": "survived"}

def evaluate(model_dir, episodes):
    model = Model(model_dir)
    results, seed = [], 0
    t0 = time.time()
    while len(results) < episodes:
        r = run_episode(model, 10_000 + seed)
        seed += 1
        if r:
            results.append(r)
    ends = Counter(r["end"] for r in results)
    n = len(results)
    print(f"{model_dir:30} | survived 300: {ends['survived']/n:5.0%} | drone deaths: {ends['drone']/n:5.0%} | "
          f"wall/body: {ends['wall/body']/n:5.0%} | avg steps: {sum(r['steps'] for r in results)/n:5.0f} | "
          f"avg score: {sum(r['score'] for r in results)/n:4.1f} | {time.time()-t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--episodes", type=int, default=40)
    args = ap.parse_args()
    for m in args.models:
        evaluate(m, args.episodes)
