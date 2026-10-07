"""
Safety shield on top of a model (no retraining): the model ranks the moves; going down that ranking, the first move
for which the simulator finds a way to stay alive for H more steps (build_dataset_v9.survives: the same
look-ahead the teacher uses) is played. If no move survives, the model's best legal move is played.

  H = "teacher": min(len + 3, 20), budget 400  (as the teacher)
  H = 4:         a short look-ahead, budget 60 (cheap enough for the browser)

Same multi-apple episodes as eval_v12 / eval_v13 (seeds 150_000 + n*1000 + i).
Usage: python eval_shield.py [episodes] [sizes...]
"""

import os
import random
import sys
import time
from collections import Counter

import build_dataset_v12 as v12
import build_dataset_v9 as v9
import eval_v13  # noqa: F401  (patches eval_v12.state_for for v13/v13d)
import eval_v12 as e12
from a_star_solver import SnakeAStarSolver

ROOT = os.path.dirname(os.path.abspath(__file__))
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
solver = SnakeAStarSolver()

def shielded_move(policy, w, mem12, mem11, depth, stats):
    state = e12.state_for(policy, w, mem12, mem11)
    danger = v12.state12(w, mem12)
    p = policy.probs(state)
    legal = [a for a in ACTIONS if not danger[f"danger_{a}"] and not (len(w.body) > 1 and a == v9.OPP[w.direction])]
    ranked = sorted(legal or ACTIONS, key=p.get, reverse=True)
    if depth is None:
        return ranked[0]
    horizon, budget = (min(len(w.body) + 3, 20), 400) if depth == "teacher" else (depth, 60)
    for i, a in enumerate(ranked):
        if v9.survives(solver, w, a, horizon, budget):
            stats["vetoes"] += i > 0
            return a
    stats["no_safe"] += 1
    return ranked[0]

def episode(policy, n, seed, depth, eaten, stats, max_steps=300):
    rng = random.Random(seed)
    random.seed(seed)
    w = None
    while w is None:
        w = v12.make_world(rng, n)
    mem12, mem11 = v12.Memory12(), e12.w11.Memory()
    mem12.update(w)
    for t in range(max_steps):
        action = shielded_move(policy, w, mem12, mem11, depth, stats)
        stats["moves"] += 1
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
        if policy.kind == "v11w":
            mem11.update(w)
    return max_steps, w.score, "survived"

def run(policy, name, n, episodes, depth):
    t0 = time.time()
    eaten, stats = Counter(), Counter()
    res = [episode(policy, n, 150_000 + n * 1000 + i, depth, eaten, stats) for i in range(episodes)]
    ends = Counter(e for _, _, e in res)
    k = len(res)
    label = f"{name}+shield{depth}" if depth else name
    print(f"{n}x{n} {label:20} | avg score {sum(r[1] for r in res) / k:5.1f} | avg steps {sum(r[0] for r in res) / k:4.0f} | "
          f"survived {ends['survived'] / k:4.0%} | drone deaths {ends['drone'] / k:4.0%} | other deaths {ends['other'] / k:4.0%} | "
          f"vetoes {stats['vetoes'] / max(1, stats['moves']):5.1%} | no safe move {stats['no_safe']} | {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    sizes = [int(x) for x in sys.argv[2:]] or [20, 30]
    models = {m: e12.Model(os.path.join(ROOT, d), m) for m, d in
              (("v11w", "laya_snake_weights_v11w"), ("v13d", "laya_snake_weights_v13d"))}
    t0 = time.time()
    for n in sizes:
        for m, pol in models.items():
            for depth in (4, "teacher"):
                run(pol, m, n, episodes, depth)
    print(f"total {time.time() - t0:.0f}s", flush=True)
