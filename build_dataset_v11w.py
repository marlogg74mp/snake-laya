"""
Dataset V11w — egocentric "window" state + an agent-side memory, so the model does not depend on the board size.

Rules = V9 (build_dataset_v9.World): drones bounce off walls and kill on body contact, phase walls,
immortality apple. Portals are off (on an unbounded board the exit would always be out of sight).
Boards are random 12x12 … 40x40.

What the model sees (no absolute coordinates anywhere, all cells relative to the head, x right / y down):
  current_dir, recent_actions, food_dir      as before (UNKNOWN in fog when the food is out of sight)
  danger_UP/DOWN/LEFT/RIGHT                  deadly right now (as V9)
  food_rel     [dx, dy] if the food is inside the window, else null
  edges        distance to the border up/down/left/right, 6 = "farther than the window"
  snake_len, immortal_steps
  body / walls / ghost_walls [dx, dy, steps_left] / drones [dx, dy, ddx, ddy] / drone_path
               only what is inside the window (drones within radius 8, their next 6 cells)
  fog_of_war
  memory_food  up to 3 apples seen earlier and now out of the window: [dx, dy, steps since seen]
  explored     how much of up/down/left/right was walked over the last 200 steps, 0..9

The teacher uses the same memory: in fog, if it remembers an apple it goes there instead of exploring.
"""

import json
import os
import random
import sys
import time
from collections import deque
from multiprocessing import Pool

import build_dataset_v9 as v9
from a_star_solver import SnakeAStarSolver

ROOT = os.path.dirname(os.path.abspath(__file__))
R = 5                 # window radius (11x11), same as the fog sight radius
DRONE_R = 8           # drones are reported a bit beyond the window: they move towards you
MEMORY_FOOD = 3
EXPLORE_STEPS = 200
QUESTION = v9.QUESTION
KEYS = ["current_dir", "recent_actions", "food_dir", "danger_UP", "danger_DOWN", "danger_LEFT", "danger_RIGHT",
        "food_rel", "edges", "snake_len", "immortal_steps", "body", "walls", "ghost_walls", "drones", "drone_path",
        "fog_of_war", "memory_food", "explored"]

class Memory:
    """What the game remembers for the snake (absolute positions inside, relative ones in the state)."""

    def __init__(self):
        self.food = {}                     # (x, y) -> step when last seen
        self.trail = deque(maxlen=EXPLORE_STEPS)
        self.step = 0

    def update(self, w):
        self.step += 1
        hx, hy = w.head
        self.trail.append((hx, hy))
        # visible = inside the window, and in fog also within the sight diamond
        visible = lambda x, y: abs(x - hx) <= R and abs(y - hy) <= R and (not w.fog_of_war or abs(x - hx) + abs(y - hy) <= R)
        for (x, y) in list(self.food):  # forget an apple whose cell is visibly empty now
            if visible(x, y) and w.food != (x, y):
                del self.food[(x, y)]
        if w.food and visible(*w.food):
            self.food[w.food] = self.step
        if len(self.food) > MEMORY_FOOD:
            for k in sorted(self.food, key=self.food.get)[:-MEMORY_FOOD]:
                del self.food[k]

    def remembered(self, w):
        """Apples remembered and currently out of the window, newest first."""
        hx, hy = w.head
        out = [(p, s) for p, s in self.food.items() if abs(p[0] - hx) > R or abs(p[1] - hy) > R]
        return sorted(out, key=lambda t: -t[1])

def window_state(w, mem):
    s = w.model_state()  # V9 state: dangers, food_dir, immortality, drones … in absolute form
    hx, hy = w.head
    inside = lambda x, y: abs(x - hx) <= R and abs(y - hy) <= R
    rel = lambda x, y: [x - hx, y - hy]
    food_rel = rel(*w.food) if (w.food and inside(*w.food) and s["food_dir"] != "UNKNOWN") else None
    edges = [min(hy, R + 1), min(w.height - 1 - hy, R + 1), min(hx, R + 1), min(w.width - 1 - hx, R + 1)]
    drones, paths = [], []
    for d, p, path in zip(w.drones, s["patrols"], s["drone_path"]):
        if abs(d.x - hx) <= DRONE_R and abs(d.y - hy) <= DRONE_R:
            drones.append(rel(p[0], p[1]) + [p[2], p[3]])
            paths.append([rel(x, y) for x, y in path])
    explored = []
    if mem.trail:
        cnt = [sum(1 for (x, y) in mem.trail if y < hy), sum(1 for (x, y) in mem.trail if y > hy),
               sum(1 for (x, y) in mem.trail if x < hx), sum(1 for (x, y) in mem.trail if x > hx)]
        explored = [min(9, round(9 * c / len(mem.trail))) for c in cnt]
    state = {
        "current_dir": s["current_dir"], "recent_actions": s["recent_actions"], "food_dir": s["food_dir"],
        "danger_UP": s["danger_UP"], "danger_DOWN": s["danger_DOWN"], "danger_LEFT": s["danger_LEFT"], "danger_RIGHT": s["danger_RIGHT"],
        "food_rel": food_rel, "edges": edges, "snake_len": s["snake_len"], "immortal_steps": s["immortal_steps"],
        "body": [rel(x, y) for x, y in w.body[1:] if inside(x, y)][:30],
        "walls": [rel(x, y) for x, y in w.obstacles if inside(x, y)],
        "ghost_walls": [rel(x, y) + [t] for x, y, t in s["phase_walls"] if t > 0 and inside(x, y)],
        "drones": drones, "drone_path": paths,
        "fog_of_war": s["fog_of_war"],
        "memory_food": [rel(*p) + [mem.step - st] for p, st in mem.remembered(w)],
        "explored": explored or [0, 0, 0, 0],
    }
    return {k: state[k] for k in KEYS}

def teacher_move(solver, w, mem):
    """V9 teacher; in fog with the food out of sight it walks to a remembered apple instead of exploring."""
    hx, hy = w.head
    hidden = w.fog_of_war and (not w.food or abs(w.food[0] - hx) + abs(w.food[1] - hy) > w.sight_radius)
    rem = mem.remembered(w)
    if hidden and rem:
        real_food, w.food, w.fog_of_war = w.food, rem[0][0], False
        try:
            return v9.teacher_move(solver, w)
        finally:
            w.food, w.fog_of_war = real_food, True
    return v9.teacher_move(solver, w)

def make_world(rng, n):
    drones = []
    for _ in range(rng.choice([1, 2]) * max(1, round(n * n / 400))):
        lo, hi = rng.randint(2, max(2, n // 4)), rng.randint(n - n // 4 - 1, n - 3)
        line = rng.randint(3, n - 4)
        drones.append(v9.Drone("h", lo, line, 1, 0, lo=lo, hi=hi) if rng.random() < 0.5
                      else v9.Drone("v", line, lo, 0, 1, lo=lo, hi=hi))
    area = n * n / 400
    w = v9.World(n, n, initial_length=rng.randint(3, min(11, n // 2 + 1)),
                 num_obstacles=rng.randint(max(1, round(area)), max(1, round(4 * area))),
                 num_portals=0, fog_of_war=rng.random() < 0.3, sight_radius=R, seed=rng.randrange(1 << 30))
    w.setup(rng, drones)
    if any(d.pos in set(w.body) or d.pos in w.solid for d in drones):
        return None
    if rng.random() < v9.IMMORTAL_APPLE_CHANCE:
        w.food_kind, w.food_ttl = "immortal", v9.IMMORTAL_APPLE_TTL
    return w

def generate_chunk(args):
    n_samples, seed = args
    rng = random.Random(seed)
    random.seed(seed)
    solver = SnakeAStarSolver()
    samples, st = [], {"episodes": 0, "dodges": 0, "memory_moves": 0, "deaths": 0}
    while len(samples) < n_samples:
        w = make_world(rng, rng.randint(12, 40))
        if w is None:
            continue
        mem = Memory()
        mem.update(w)
        st["episodes"] += 1
        for _ in range(200):
            if len(samples) >= n_samples:
                break
            state = window_state(w, mem)
            action = teacher_move(solver, w, mem)
            dodge = action != v9.base_move(solver, w)
            st["dodges"] += dodge
            st["memory_moves"] += bool(state["fog_of_war"] and state["memory_food"])
            sample = {"state": state, "question": QUESTION, "choices": ["UP", "DOWN", "LEFT", "RIGHT"], "label": action}
            samples.extend([sample] * (v9.DODGE_REPEAT if dodge else 1))
            r = w.world_step(action)
            if r == "dead":
                st["deaths"] += 1
            if r != "ok":
                break
            mem.update(w)
    return samples[:n_samples], st

def generate(target=70000, workers=12):
    jobs = [(target // workers + (1 if i < target % workers else 0), 11_000 + i) for i in range(workers)]
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(generate_chunk, jobs)
    samples = [s for ch, _ in res for s in ch]
    random.Random(7).shuffle(samples)
    tot = {k: sum(r[k] for _, r in res) for k in res[0][1]}
    with open(os.path.join(ROOT, "dataset_v11w.jsonl"), "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"V11w: {len(samples)} samples in {time.time() - t0:.0f}s | {tot}", flush=True)

if __name__ == "__main__":
    generate(int(sys.argv[1]) if len(sys.argv) > 1 else 70000)
