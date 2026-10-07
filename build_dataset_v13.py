"""
Dataset V13 — V12 world (several apples of different value, poison, fog, memory) with two changes:

1. The model sees exact relative coordinates of every apple it knows about (V12 showed far apples only as
   a direction + "mid"/"far" bucket):
     foods       [[dx, dy, value, ttl]]  known items sorted by distance, dx/dy clipped to ±20, ttl -1 = never expires
     memory_food [[dx, dy, value, age]]  items seen before and out of sight now (fog), as in V12
2. Teacher: the V12 one (value / distance). The route-planning teacher (teacher_route.pick_target_route, ROUTE = True)
   was checked with `gamma` on 80 new episodes and gave +3% at 20x20 and -3..5% at 30x30, i.e. noise — so V13
   changes only what the model sees, which also makes the comparison with V12 clean.

Then a DAgger round (as in V10): the trained model plays, the teacher labels the states it visits.

python build_dataset_v13.py data [n]                 -> dataset_v13.jsonl
python build_dataset_v13.py dagger [n] [model_dir]   -> dataset_v13d.jsonl (dagger samples + dataset_v13.jsonl)
python build_dataset_v13.py gamma [episodes]         -> pick GAMMA for the route teacher (CPU only)
"""

import copy
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import build_dataset_v9 as v9
import build_dataset_v12 as v12
import teacher_route as tr
from a_star_solver import SnakeAStarSolver

ROOT = os.path.dirname(os.path.abspath(__file__))
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
KEYS = ["current_dir", "recent_actions", "danger_UP", "danger_DOWN", "danger_LEFT", "danger_RIGHT",
        "foods", "memory_food", "poison", "edges", "snake_len", "immortal_steps",
        "body", "walls", "ghost_walls", "drones", "drone_path", "fog_of_war", "explored"]
MAX_FOODS = 5
ROUTE = False  # route-planning teacher instead of value / distance (see `gamma`)

def clip(v):
    return max(-20, min(20, v))

def state13(w, mem):
    s = v12.state12(w, mem)  # window parts, dangers incl. poison, memory_food, explored
    hx, hy = w.head
    foods = []
    for (x, y), (kind, ttl) in sorted(w.items.items(), key=lambda kv: abs(kv[0][0] - hx) + abs(kv[0][1] - hy)):
        if not w.fog_of_war or mem.visible(w, x, y):
            foods.append([clip(x - hx), clip(y - hy), v12.VALUE[kind], -1 if ttl is None else ttl])
    s["foods"] = foods[:MAX_FOODS]
    return {k: s[k] for k in KEYS}

def teacher13(solver, w, mem):
    """(teacher move, plain A* move to the same target); with ROUTE the target comes from the route planner."""
    saved = v12.pick_target
    v12.pick_target = tr.pick_target_route if ROUTE else saved
    try:
        return v12.teacher12(solver, w, mem)
    finally:
        v12.pick_target = saved

def generate_chunk(args):
    n_samples, seed = args
    rng = random.Random(seed)
    random.seed(seed)
    solver = SnakeAStarSolver()
    samples, st = [], {"episodes": 0, "dodges": 0, "memory_moves": 0, "deaths": 0, "score": 0}
    while len(samples) < n_samples:
        w = v12.make_world(rng, rng.randint(12, 40))
        if w is None:
            continue
        mem = v12.Memory12()
        mem.update(w)
        st["episodes"] += 1
        for _ in range(200):
            if len(samples) >= n_samples:
                break
            state = state13(w, mem)
            action, plain = teacher13(solver, w, mem)
            dodge = action != plain
            st["dodges"] += dodge
            st["memory_moves"] += bool(state["memory_food"])
            sample = {"state": state, "question": v9.QUESTION, "choices": ACTIONS, "label": action}
            samples.extend([sample] * (v9.DODGE_REPEAT if dodge else 1))
            r = w.world_step(action)
            if r == "dead":
                st["deaths"] += 1
            if r != "ok":
                break
            mem.update(w)
        st["score"] += w.score
    return samples[:n_samples], st

def write(path, samples):
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

def generate(target=80000, workers=12):
    jobs = [(target // workers + (1 if i < target % workers else 0), 13_000 + i) for i in range(workers)]
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(generate_chunk, jobs)
    samples = [s for ch, _ in res for s in ch]
    random.Random(7).shuffle(samples)
    tot = {k: sum(r[k] for _, r in res) for k in res[0][1]}
    write("dataset_v13.jsonl", samples)
    print(f"V13: {len(samples)} samples in {time.time() - t0:.0f}s | {tot}", flush=True)

# ---------- DAgger round ----------

def label(args):
    w, mem = args
    solver = SnakeAStarSolver()
    random.seed(0)
    state = state13(w, mem)
    a, plain = teacher13(solver, w, mem)
    return state, a, a != plain

def dagger(n_states=40000, model_dir="laya_snake_weights_v13", beta=0.3, parallel=64):
    import torch
    from dagger_v10 import load_model, predict
    tok, model, keys = load_model(os.path.join(ROOT, model_dir))
    rng = random.Random(2313)
    random.seed(2313)
    solver = SnakeAStarSolver()
    stats = {"episodes": 0, "deaths": 0}

    def fresh():
        w = None
        while w is None:
            w = v12.make_world(rng, rng.randint(12, 40))
        mem = v12.Memory12()
        mem.update(w)
        stats["episodes"] += 1
        return [w, mem, 0]

    worlds, visited, t0 = [fresh() for _ in range(parallel)], [], time.time()
    print(f"[1/3] {model_dir} plays (beta={beta}) to collect {n_states} states...", flush=True)
    while len(visited) < n_states:
        states = [state13(w, mem) for w, mem, _ in worlds]
        probs = predict(tok, model, keys, states)
        for slot, (st, p) in enumerate(zip(states, probs)):
            w, mem, steps = worlds[slot]
            visited.append((w.clone(), copy.deepcopy(mem)))
            if rng.random() < beta:
                action, _ = teacher13(solver, w, mem)
            else:
                legal = [a for a in ACTIONS if not st[f"danger_{a}"] and not (len(w.body) > 1 and a == v9.OPP[w.direction])]
                pp = dict(zip(ACTIONS, p))
                action = max(legal or ACTIONS, key=pp.get)
            r = w.world_step(action)
            stats["deaths"] += r == "dead"
            if r != "ok" or steps + 1 >= 300:
                worlds[slot] = fresh()
            else:
                mem.update(w)
                worlds[slot] = [w, mem, steps + 1]
        if len(visited) % (parallel * 50) < parallel:
            print(f"  collected {len(visited)} states ({time.time() - t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    print(f"      {stats['episodes']} episodes, {stats['deaths']} deaths", flush=True)
    print("[2/3] Teacher labels the visited states...", flush=True)
    with Pool(12) as pool:
        labelled = pool.map(label, visited[:n_states], chunksize=64)
    new = []
    for s, a, d in labelled:
        new.extend([{"state": s, "question": v9.QUESTION, "choices": ACTIONS, "label": a}] * (v9.DODGE_REPEAT if d else 1))
    print(f"[3/3] dagger samples {len(new)} (dodges {sum(d for *_, d in labelled)})", flush=True)
    merged = new + [json.loads(l) for l in open(os.path.join(ROOT, "dataset_v13.jsonl"), encoding="utf-8")]
    random.Random(11).shuffle(merged)
    write("dataset_v13d.jsonl", merged)
    print(f"      dataset_v13d.jsonl: {len(merged)} samples | {time.time() - t0:.0f}s", flush=True)

# ---------- GAMMA sweep for the route teacher ----------

def gamma_episode(args):
    gamma, n, seed = args
    tr.GAMMA = gamma
    solver = SnakeAStarSolver()
    rng = random.Random(seed)
    random.seed(seed)
    w = None
    while w is None:
        w = v12.make_world(rng, n)
    mem = v12.Memory12()
    mem.update(w)
    for t in range(300):
        if gamma is None:
            action, _ = v12.teacher12(solver, w, mem)
        else:
            v12.pick_target = tr.pick_target_route
            action, _ = v12.teacher12(solver, w, mem)
        if w.world_step(action) != "ok":
            break
        mem.update(w)
    return w.score, t + 1

def gamma_sweep(episodes=80):
    for n in (20, 30):
        for g in (None, 0.9, 0.95, 0.98):
            t0 = time.time()
            with Pool(12) as pool:
                res = pool.map(gamma_episode, [(g, n, 160_000 + n * 1000 + i) for i in range(episodes)])
            print(f"{n}x{n} {'greedy' if g is None else f'route g={g}':14} | avg score {sum(r[0] for r in res) / episodes:5.2f} | "
                  f"avg steps {sum(r[1] for r in res) / episodes:4.0f} | {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "data"
    if cmd == "data":
        generate(int(sys.argv[2]) if len(sys.argv) > 2 else 80000)
    elif cmd == "dagger":
        dagger(int(sys.argv[2]) if len(sys.argv) > 2 else 40000, sys.argv[3] if len(sys.argv) > 3 else "laya_snake_weights_v13")
    elif cmd == "gamma":
        gamma_sweep(int(sys.argv[2]) if len(sys.argv) > 2 else 80)
