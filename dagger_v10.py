"""
V10 = DAgger round on top of V9.
1. The current model plays in V9 worlds (64 in lockstep, batched GPU inference). With probability BETA
   the teacher's move is executed instead, so episodes do not collapse immediately.
2. Every visited state is labelled by the teacher (A* + time-space search) in a process pool.
3. The new samples (drone dodges repeated like in V9) are written to dataset_v10_dagger.jsonl and
   merged with dataset_v9.jsonl into dataset_v10.jsonl for fine-tuning.

Usage: python dagger_v10.py [n_states] [model_dir] [base_dataset] [out_name]
  round 1 (V10): defaults  -> dataset_v10_dagger.jsonl + dataset_v9.jsonl  = dataset_v10.jsonl
  round 2 (V11): 40000 laya_snake_weights_v10 dataset_v10.jsonl v11       = dataset_v11.jsonl
"""

import json
import os
import random
import sys
import time
from multiprocessing import Pool

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import build_dataset_v9 as v9
from a_star_solver import SnakeAStarSolver

ROOT = os.path.dirname(os.path.abspath(__file__))
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
BETA = 0.3
PARALLEL = 64
MAX_STEPS = 300

def load_model(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()
    keys = json.load(open(os.path.join(model_dir, "state_schema.json")))["keys"]
    return tok, model, keys

@torch.no_grad()
def predict(tok, model, keys, states):
    texts = [f"State: {json.dumps({k: s[k] for k in keys})}\nQuestion: {v9.QUESTION}\nChoices: UP, DOWN, LEFT, RIGHT"
             for s in states]
    enc = tok(texts, truncation=True, max_length=448, padding=True, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**enc).logits
    return torch.softmax(logits.float(), -1).tolist()

def collect(n_states, model_dir, seed=2024):
    tok, model, keys = load_model(model_dir)
    rng = random.Random(seed)
    random.seed(seed)
    solver = SnakeAStarSolver()
    worlds, visited = [], []
    stats = {"episodes": 0, "deaths": 0}

    def fresh():
        w = None
        while w is None:
            w = v9.new_world(rng)
        stats["episodes"] += 1
        return [w, 0]

    worlds = [fresh() for _ in range(PARALLEL)]
    t0 = time.time()
    while len(visited) < n_states:
        states = [w.model_state() for w, _ in worlds]
        probs = predict(tok, model, keys, states)
        for slot, (st, p) in enumerate(zip(states, probs)):
            w, steps = worlds[slot]
            visited.append(w.clone())
            if rng.random() < BETA:
                action = v9.teacher_move(solver, w)
            else:
                legal = [a for a in ACTIONS if not st[f"danger_{a}"] and not (len(w.body) > 1 and a == v9.OPP[w.direction])]
                pp = dict(zip(ACTIONS, p))
                action = max(legal or ACTIONS, key=pp.get)
            r = w.world_step(action)
            if r == "dead":
                stats["deaths"] += 1
            worlds[slot] = fresh() if r != "ok" or steps + 1 >= MAX_STEPS else [w, steps + 1]
        if len(visited) % (PARALLEL * 50) < PARALLEL:
            print(f"  collected {len(visited)} states ({time.time() - t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return visited[:n_states], stats

def label(w):
    solver = SnakeAStarSolver()
    random.seed(0)
    state = w.model_state()
    a = v9.teacher_move(solver, w.clone())
    dodge = a != v9.base_move(solver, w.clone())
    return state, a, dodge

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    model_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "laya_snake_weights")
    base = sys.argv[3] if len(sys.argv) > 3 else "dataset_v9.jsonl"
    name = sys.argv[4] if len(sys.argv) > 4 else "v10"
    t0 = time.time()
    print(f"[1/3] Model {model_dir} plays (beta={BETA}) to collect {n} states...", flush=True)
    worlds, st = collect(n, model_dir, seed=2024 if name == "v10" else 2024 + int(name.strip("v")))
    print(f"      {st['episodes']} episodes, {st['deaths']} deaths, {time.time() - t0:.0f}s", flush=True)

    print("[2/3] Teacher labels the visited states...", flush=True)
    t1 = time.time()
    with Pool(12) as pool:
        labelled = pool.map(label, worlds, chunksize=64)
    dodges = sum(d for _, _, d in labelled)
    print(f"      done in {time.time() - t1:.0f}s, drone-dodge labels: {dodges} ({dodges / len(labelled):.1%})", flush=True)

    print("[3/3] Writing datasets...", flush=True)
    new = []
    for s, a, d in labelled:
        sample = {"state": s, "question": v9.QUESTION, "choices": ACTIONS, "label": a}
        new.extend([sample] * (v9.DODGE_REPEAT if d else 1))
    with open(os.path.join(ROOT, f"dataset_{name}_dagger.jsonl"), "w", encoding="utf-8") as f:
        for s in new:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    merged = new + [json.loads(l) for l in open(os.path.join(ROOT, base), encoding="utf-8")]  # DAgger aggregates
    random.Random(11).shuffle(merged)
    with open(os.path.join(ROOT, f"dataset_{name}.jsonl"), "w", encoding="utf-8") as f:
        for s in merged:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"      dagger samples: {len(new)}, merged dataset_{name}.jsonl: {len(merged)} | total {time.time() - t0:.0f}s", flush=True)
