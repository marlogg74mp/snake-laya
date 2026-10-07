"""
How does a model trained only on 20x20 play on other board sizes?
Same V9 rules (build_dataset_v9.World); drones, walls and starting length are scaled to the board;
only linear drones (the circuit drone has the 20x20 bounds built in). The teacher plays the same
episodes, so a drop that the teacher shows too is the board, not the model.

Usage: python eval_grid_size.py [model_dir] [episodes] [sizes...]   e.g.  python eval_grid_size.py laya_snake_weights_v10 40 15 20 25
"""

import random
import sys
import time
from collections import Counter

import build_dataset_v9 as v9
import eval_v9
from a_star_solver import SnakeAStarSolver

solver = SnakeAStarSolver()
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

def make_world(rng, n):
    drones = []
    for _ in range(rng.choice([1, 2])):
        lo, hi = rng.randint(2, max(2, n // 4)), rng.randint(n - n // 4 - 1, n - 3)
        line = rng.randint(4, n - 5)
        drones.append(v9.Drone("h", lo, line, 1, 0, lo=lo, hi=hi) if rng.random() < 0.5
                      else v9.Drone("v", line, lo, 0, 1, lo=lo, hi=hi))
    area = (n * n) / 400
    w = v9.World(n, n, initial_length=rng.randint(3, min(11, n // 2 + 1)),
                 num_obstacles=rng.randint(max(1, round(1 * area)), max(1, round(4 * area))),
                 num_portals=2 if rng.random() < 0.6 else 0, fog_of_war=False, sight_radius=5,
                 seed=rng.randrange(1 << 30))
    w.setup(rng, drones)
    if any(d.pos in set(w.body) or d.pos in w.solid for d in drones):
        return None
    return w

def episode(policy, n, seed, stats, max_steps=300):
    rng = random.Random(seed)
    random.seed(seed)
    w = None
    while w is None:
        w = make_world(rng, n)
    for t in range(max_steps):
        state = w.model_state()
        if policy == "teacher":
            action = v9.teacher_move(solver, w)
        else:
            text = f"State: {eval_v9.json.dumps({k: state[k] for k in policy.keys})}"
            stats["states"] += 1
            stats["truncated"] += len(policy.tok(text)["input_ids"]) > 420  # + question/choices > 448
            p = policy.probs_batch([state])[0]
            legal = [a for a in ACTIONS if not state[f"danger_{a}"] and not (len(w.body) > 1 and a == v9.OPP[w.direction])]
            action = max(legal or ACTIONS, key=p.get)
        immortal = w.immortal_steps > 0
        r = w.world_step(action)
        if r == "dead":
            drone = any(d.pos in set(w.body) or d.pos == w.head for d in w.drones) and not immortal
            return t + 1, w.score, "drone" if drone else "other"
        if r == "win":
            break
    return max_steps, w.score, "survived"

def run(policy, name, n, episodes):
    stats = Counter()
    t0 = time.time()
    res = [episode(policy, n, 90_000 + n * 1000 + i, stats) for i in range(episodes)]
    ends = Counter(e for _, _, e in res)
    k = len(res)
    trunc = f" | inputs over 448 tokens: {stats['truncated'] / stats['states']:.1%}" if stats["states"] else ""
    print(f"{n}x{n} {name:8} | survived 300: {ends['survived'] / k:4.0%} | drone deaths: {ends['drone'] / k:4.0%} | "
          f"other deaths: {ends['other'] / k:4.0%} | avg steps {sum(r[0] for r in res) / k:4.0f} | "
          f"avg score {sum(r[1] for r in res) / k:5.1f}{trunc} | {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "laya_snake_weights_v10"
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    sizes = [int(x) for x in sys.argv[3:]] or [15, 20, 25]
    m = eval_v9.Model(model_dir)
    for n in sizes:
        run(m, "model", n, episodes)
        run("teacher", "teacher", n, episodes)
