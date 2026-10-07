"""
Dataset V8: Patrol drones that kill on ANY body contact (not only head-on).

Differences from V7:
- Game rule: a drone stepping onto any snake segment (or the head stepping onto a drone, or a swap
  crossing) ends the episode, matching the web UI with PATROL_BODY_HITS = true.
- Teacher: the A* move is checked by a time-space rollout (snake follows the teacher, drones follow
  their routes) for len(snake)+3 steps; if a drone would hit the body, the teacher picks the
  safest alternative (wait / detour) instead.
- State schema V8 adds what the model needs to predict body hits:
    "snake_len": int
    "patrols": [[x, y, dx, dy], ...]   (dx, dy = drone's next step)
- 1-2 drones per episode; ~35% of episodes use the exact drone routes of the web UI.
- Longer snakes (initial length 3-11, up to 150 steps per episode for more map variety).
"""

import copy
import json
import os
import random
import sys
import time
from multiprocessing import Pool

from snake_env import SnakeEnv
from a_star_solver import SnakeAStarSolver
from build_dataset_v7_patrols import PatrolSimulator

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_V8 = os.path.join(ROOT, "dataset_v8_patrol_body.jsonl")

STATE_KEYS_V8 = ["current_dir", "recent_actions", "food_dir", "danger_UP", "danger_DOWN", "danger_LEFT",
                 "danger_RIGHT", "head_pos", "food_pos", "snake_len", "obstacles", "portals",
                 "fog_of_war", "sight_radius", "patrols"]
QUESTION = "What is the next safe move avoiding patrols and walls towards food?"
DIRS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

def make_ui_patrols():
    """The two fixed drones of the web UI (web_server.py generatePatrols)."""
    h = PatrolSimulator(20, 20, "linear_horiz")
    h.x, h.y, h.dx, h.dy, h.min_x, h.max_x = 4, 5, 1, 0, 3, 16
    v = PatrolSimulator(20, 20, "linear_vert")
    v.x, v.y, v.dx, v.dy, v.min_y, v.max_y = 15, 12, 0, 1, 8, 17
    return [h, v]

def make_random_patrols(rng):
    return [PatrolSimulator(20, 20, rng.choice(["linear_horiz", "linear_vert", "circuit"]))
            for _ in range(rng.choice([1, 2]))]

def clone_patrol(p):
    c = copy.copy(p)
    if hasattr(p, "waypoints"):
        c.waypoints = list(p.waypoints)
    return c

def clone_env(env):
    e = copy.copy(env)
    e.body = list(env.body)
    e.obstacles = list(env.obstacles)
    e.portals = list(env.portals)
    e.recent_actions = list(env.recent_actions)
    e.rng = random.Random(0)  # rollouts must not consume the real episode's RNG
    return e

def patrol_hit(env, patrols, old_head, old_positions):
    """UI rule with PATROL_BODY_HITS = true. Called after both snake and drones moved."""
    body = set(env.body)
    for p, op in zip(patrols, old_positions):
        if p.pos in body:
            return True
        if env.head == op and old_head == p.pos:  # swap crossing
            return True
    return False

def base_move(solver, env, patrols):
    """V7 teacher: A* treating each drone's current and next cell as walls."""
    orig = env.obstacles
    env.obstacles = orig + [p.pos for p in patrols] + [p.predict_next(1) for p in patrols]
    try:
        return solver.get_best_move(env)
    finally:
        env.obstacles = orig

def simulate(env, patrols, move):
    """One step of snake + drones on copies. Returns (env, patrols) or None if the snake dies."""
    e = clone_env(env)
    ps = [clone_patrol(p) for p in patrols]
    old_head, old_pos = e.head, [p.pos for p in ps]
    _, _, done, info = e.step(move)
    if done and info.get("reason") != "win":
        return None
    for p in ps:
        p.step()
    if patrol_hit(e, ps, old_head, old_pos):
        return None
    return e, ps

def has_escape(solver, env, patrols, depth, budget, use_teacher):
    """
    True if SOME move sequence keeps the snake alive for `depth` steps (drones are deterministic).
    Tries the A* teacher's move first, so the common case costs ~depth steps.
    """
    if depth == 0:
        return True
    if budget[0] <= 0:
        return False
    budget[0] -= 1
    order = list(DIRS)
    if use_teacher:
        first = base_move(solver, env, patrols)
        order.remove(first)
        order.insert(0, first)
    for m in order:
        if len(env.body) > 1 and m == SnakeEnv.OPPOSITES[env.direction]:
            continue
        nxt = simulate(env, patrols, m)
        if nxt and has_escape(solver, nxt[0], nxt[1], depth - 1, budget, use_teacher=False):
            return True
    return False

def rollout_survives(solver, env, patrols, first_move, horizon, budget=400):
    nxt = simulate(env, patrols, first_move)
    if nxt is None:
        return False
    return has_escape(solver, nxt[0], nxt[1], horizon - 1, [budget], use_teacher=True)

def teacher_move(solver, env, patrols):
    move = base_move(solver, env, patrols)
    horizon = min(len(env.body) + 3, 20)
    hx, hy = env.head
    if all(abs(p.x - hx) + abs(p.y - hy) > horizon + 2 for p in patrols):
        return move  # no drone can reach the snake within the horizon

    if rollout_survives(solver, env, patrols, move, horizon):
        return move

    blocked = {p.pos for p in patrols} | {p.predict_next(1) for p in patrols}
    alternatives = []
    for a, (dx, dy) in DIRS.items():
        if a == move or (len(env.body) > 1 and a == SnakeEnv.OPPOSITES[env.direction]):
            continue
        nxt = (hx + dx, hy + dy)
        if env.is_collision(nxt) or nxt in blocked:
            continue
        food_dist = abs(nxt[0] - env.food[0]) + abs(nxt[1] - env.food[1]) if env.food else 0
        alternatives.append((food_dist, a))
    for _, a in sorted(alternatives):
        if rollout_survives(solver, env, patrols, a, horizon):
            return a
    return move

def build_state(env, patrols):
    s = env.get_compact_state()
    hx, hy = env.head
    danger_cells = {p.pos for p in patrols} | {p.predict_next(1) for p in patrols}
    for a, (dx, dy) in DIRS.items():
        s[f"danger_{a}"] = bool(s[f"danger_{a}"] or (hx + dx, hy + dy) in danger_cells)
    s["snake_len"] = len(env.body)
    s["patrols"] = []
    for p in patrols:
        nx, ny = p.predict_next(1)
        s["patrols"].append([p.x, p.y, nx - p.x, ny - p.y])
    return {k: s[k] for k in STATE_KEYS_V8}

def generate_chunk(args):
    n_samples, seed = args
    rng = random.Random(seed)
    random.seed(seed)  # PatrolSimulator uses the global RNG
    solver = SnakeAStarSolver()
    samples = []
    stats = {"episodes": 0, "overrides": 0, "patrol_deaths": 0}

    while len(samples) < n_samples:
        patrols = make_ui_patrols() if rng.random() < 0.35 else make_random_patrols(rng)
        env = SnakeEnv(width=20, height=20,
                       initial_length=rng.randint(3, 11),
                       num_obstacles=rng.randint(0, 3),
                       num_portals=2 if rng.random() < 0.6 else 0,
                       fog_of_war=rng.random() < 0.35, sight_radius=5,
                       seed=rng.randrange(1 << 30))
        start_cells = set(env.body)
        if any(p.pos in start_cells for p in patrols):
            continue
        stats["episodes"] += 1

        for _ in range(150):
            if len(samples) >= n_samples:
                break
            action = teacher_move(solver, env, patrols)
            if action != base_move(solver, env, patrols):
                stats["overrides"] += 1
            samples.append({"state": build_state(env, patrols), "question": QUESTION,
                            "choices": ["UP", "DOWN", "LEFT", "RIGHT"], "label": action})

            old_head, old_pos = env.head, [p.pos for p in patrols]
            _, _, done, _ = env.step(action)
            for p in patrols:
                p.step()
            if not done and patrol_hit(env, patrols, old_head, old_pos):
                stats["patrol_deaths"] += 1
                done = True
            if done:
                break
    return samples, stats

def generate_v8_dataset(target_samples=50000, workers=12):
    chunk = target_samples // workers
    jobs = [(chunk + (1 if i < target_samples % workers else 0), 1000 + i) for i in range(workers)]
    t0 = time.time()
    print(f"Generating {target_samples} V8 samples on {workers} processes...", flush=True)
    with Pool(workers) as pool:
        results = pool.map(generate_chunk, jobs)

    samples = [s for chunk_samples, _ in results for s in chunk_samples]
    random.Random(7).shuffle(samples)
    totals = {k: sum(st[k] for _, st in results) for k in results[0][1]}
    print(f"Done in {time.time() - t0:.0f}s: {len(samples)} samples, {totals['episodes']} episodes, "
          f"teacher overrides (drone avoidance): {totals['overrides']}, "
          f"episodes ended by drone: {totals['patrol_deaths']}", flush=True)

    with open(DATASET_V8, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {DATASET_V8}", flush=True)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    generate_v8_dataset(n)
