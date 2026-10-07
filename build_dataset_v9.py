"""
Dataset V9 — rules in RULES_V9.md:
- drones bounce off solid walls and kill on any body contact
- phase walls: SOLID -> BLINKING (3) -> GHOST (15-25) -> SOLID, ghost walls block nobody
- immortality apple: 50 steps immune to drones and own body

Teacher = A* + time-space search (a move is accepted only if some continuation survives
min(len+3, 20) steps with the deterministic drones and wall timers).
"""

import copy
import json
import os
import random
import sys
import time
from collections import deque
from multiprocessing import Pool

from snake_env import SnakeEnv
from a_star_solver import SnakeAStarSolver

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_V9 = os.path.join(ROOT, "dataset_v9.jsonl")

STATE_KEYS_V9 = ["current_dir", "recent_actions", "food_dir", "danger_UP", "danger_DOWN", "danger_LEFT",
                 "danger_RIGHT", "head_pos", "body", "food_pos", "snake_len", "immortal_steps", "obstacles",
                 "phase_walls", "portals", "fog_of_war", "sight_radius", "patrols", "drone_path"]
QUESTION = "What is the next safe move avoiding patrols and walls towards food?"
DIRS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
OPP = SnakeEnv.OPPOSITES

IMMORTAL_STEPS = 50
IMMORTAL_APPLE_TTL = 60
IMMORTAL_APPLE_CHANCE = 0.15
BLINK_STEPS = 3
MAX_BODY_CELLS = 30      # body cells shown to the model (neck -> tail)
DRONE_PATH_STEPS = 6     # future drone cells shown to the model
DODGE_REPEAT = 4         # drone-dodge samples are rare; repeat them so the model does not treat them as noise

# ---------------------------------------------------------------- drones

class Drone:
    """Linear ping-pong ('h'/'v') or rectangular circuit; bounces off solid walls (RULES_V9 §1)."""

    def __init__(self, kind, x, y, dx, dy, lo=None, hi=None, waypoints=None):
        self.kind, self.x, self.y, self.dx, self.dy = kind, x, y, dx, dy
        self.lo, self.hi = lo, hi
        self.waypoints = waypoints
        self.wp_idx, self.wp_sign = 0, 1

    @property
    def pos(self):
        return (self.x, self.y)

    def _in_route(self, c):
        if self.kind == "h":
            return c[1] == self.y and self.lo <= c[0] <= self.hi
        if self.kind == "v":
            return c[0] == self.x and self.lo <= c[1] <= self.hi
        return 0 <= c[0] < 20 and 0 <= c[1] < 20

    def _circuit_target(self, sign):
        n = len(self.waypoints)
        return self.waypoints[(self.wp_idx + (1 if sign > 0 else 0)) % n] if sign > 0 else \
            self.waypoints[self.wp_idx % n]

    def _circuit_step(self, sign):
        tx, ty = self._circuit_target(sign)
        if (self.x, self.y) == (tx, ty):
            return (0, 0)
        if self.x != tx:
            return (1 if tx > self.x else -1, 0)
        return (0, 1 if ty > self.y else -1)

    def plan(self, solid):
        """(dx, dy) of the next move given the set of solid wall cells."""
        if self.kind in ("h", "v"):
            for d in ((self.dx, self.dy), (-self.dx, -self.dy)):
                c = (self.x + d[0], self.y + d[1])
                if self._in_route(c) and c not in solid:
                    return d
            return (0, 0)
        for sign in (self.wp_sign, -self.wp_sign):
            d = self._circuit_step(sign)
            c = (self.x + d[0], self.y + d[1])
            if d != (0, 0) and c not in solid:
                return d
        return (0, 0)

    def step(self, solid):
        d = self.plan(solid)
        if self.kind in ("h", "v"):
            if d != (0, 0):
                self.dx, self.dy = d
        else:
            if d != (0, 0) and d != self._circuit_step(self.wp_sign):
                self.wp_sign = -self.wp_sign
                if self.wp_sign < 0:
                    pass
        self.x += d[0]
        self.y += d[1]
        if self.kind == "circuit":
            n = len(self.waypoints)
            if self.wp_sign > 0 and self.pos == self.waypoints[(self.wp_idx + 1) % n]:
                self.wp_idx = (self.wp_idx + 1) % n
            elif self.wp_sign < 0 and self.pos == self.waypoints[self.wp_idx % n]:
                self.wp_idx = (self.wp_idx - 1) % n

def ui_drones():
    """The two drones of the web UI."""
    return [Drone("h", 4, 5, 1, 0, lo=3, hi=16), Drone("v", 15, 12, 0, 1, lo=8, hi=17)]

def random_drones(rng):
    drones = []
    for _ in range(rng.choice([1, 2])):
        kind = rng.choice(["h", "v", "circuit"])
        if kind == "h":
            y, lo, hi = rng.randint(4, 15), rng.randint(2, 5), rng.randint(14, 17)
            drones.append(Drone("h", lo, y, 1, 0, lo=lo, hi=hi))
        elif kind == "v":
            x, lo, hi = rng.randint(4, 15), rng.randint(2, 5), rng.randint(14, 17)
            drones.append(Drone("v", x, lo, 0, 1, lo=lo, hi=hi))
        else:
            cx, cy = rng.randint(5, 13), rng.randint(5, 13)
            wps = [(cx, cy), (cx + 5, cy), (cx + 5, cy + 5), (cx, cy + 5)]
            drones.append(Drone("circuit", cx, cy, 1, 0, waypoints=wps))
    return drones

# ---------------------------------------------------------------- world

class World(SnakeEnv):
    """SnakeEnv + drones + phase walls + immortality (RULES_V9)."""

    def setup(self, rng, drones):
        self.wrng = rng
        self.drones = drones
        self.immortal_steps = 0
        self.food_kind, self.food_ttl = "standard", None
        # group wall cells into structures (4-connected components)
        cells, self.structures = set(self.obstacles), []
        while cells:
            start = cells.pop()
            comp, q = [start], deque([start])
            while q:
                cx, cy = q.popleft()
                for dx, dy in DIRS.values():
                    n = (cx + dx, cy + dy)
                    if n in cells:
                        cells.remove(n)
                        comp.append(n)
                        q.append(n)
            self.structures.append({"cells": comp, "phase": "SOLID", "timer": rng.randint(15, 80)})
        self._sync_walls()

    def _sync_walls(self):
        self.obstacles = [c for s in self.structures if s["phase"] != "GHOST" for c in s["cells"]]
        self.solid = set(self.obstacles)

    def is_collision(self, point):
        x, y = point[0], point[1]
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        if (x, y) in getattr(self, "solid", set(self.obstacles)):
            return True
        if getattr(self, "immortal_steps", 0) <= 0 and (x, y) in self.body:
            return True
        return False

    def _spawn_food(self):
        occupied = set(self.body) | {c for s in getattr(self, "structures", []) for c in s["cells"]}
        occupied |= set(getattr(self, "obstacles", []))
        free = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in occupied]
        food = self.rng.choice(free) if free else None
        wrng = getattr(self, "wrng", None)
        if wrng is not None:
            if wrng.random() < IMMORTAL_APPLE_CHANCE:
                self.food_kind, self.food_ttl = "immortal", IMMORTAL_APPLE_TTL
            else:
                self.food_kind, self.food_ttl = "standard", None
        return food

    def clone(self):
        w = copy.copy(self)
        w.body = list(self.body)
        w.recent_actions = list(self.recent_actions)
        w.structures = [dict(s, cells=s["cells"]) for s in self.structures]
        w.obstacles = list(self.obstacles)
        w.solid = set(self.solid)
        w.drones = [copy.copy(d) for d in self.drones]
        w.rng = random.Random(0)
        w.wrng = random.Random(0)
        return w

    def drone_danger_cells(self):
        cells = set()
        for d in self.drones:
            dx, dy = d.plan(self.solid)
            cells.add(d.pos)
            cells.add((d.x + dx, d.y + dy))
        return cells

    def world_step(self, action):
        """Full V9 step. Returns 'dead', 'ok' or 'win'."""
        immortal = self.immortal_steps > 0
        hx, hy = self.head
        d = DIRS[action if not (len(self.body) > 1 and action == OPP[self.direction]) else self.direction]
        target = (hx + d[0], hy + d[1])
        if len(self.portals) >= 2:
            target = self.portals[1] if target == self.portals[0] else self.portals[0] if target == self.portals[1] else target
        if not immortal and any(dr.pos == target for dr in self.drones):
            return "dead"
        old_head = self.head
        old_drones = [dr.pos for dr in self.drones]
        food_kind = self.food_kind
        old_score = self.score
        _, _, done, info = self.step(action)
        if done and info.get("reason") != "win":
            return "dead"
        ate = self.score > old_score
        if ate and food_kind == "immortal":
            self.immortal_steps = IMMORTAL_STEPS + 1  # decremented below, so 50 remain
        for dr in self.drones:
            dr.step(self.solid)
        if not immortal and self.immortal_steps <= 0:
            body = set(self.body)
            for dr, op in zip(self.drones, old_drones):
                if dr.pos in body or (self.head == op and old_head == dr.pos):
                    return "dead"
        occupied = set(self.body) | {dr.pos for dr in self.drones}
        for s in self.structures:
            if s["phase"] == "GHOST" and s["timer"] <= 1:
                if not any(c in occupied for c in s["cells"]):
                    s["phase"], s["timer"] = "SOLID", self.wrng.randint(40, 80)
                continue
            s["timer"] -= 1
            if s["timer"] <= 0:
                if s["phase"] == "SOLID":
                    s["phase"], s["timer"] = "BLINKING", BLINK_STEPS
                elif s["phase"] == "BLINKING":
                    s["phase"], s["timer"] = "GHOST", self.wrng.randint(15, 25)
        self._sync_walls()
        if self.immortal_steps > 0:
            self.immortal_steps -= 1
        if self.food_kind == "immortal" and not ate:
            self.food_ttl -= 1
            if self.food_ttl <= 0:
                self.food = self._spawn_food()
        return "win" if done else "ok"

    def model_state(self):
        s = self.get_compact_state()
        hx, hy = self.head
        if self.immortal_steps <= 1:
            danger = self.drone_danger_cells()
            for a, (dx, dy) in DIRS.items():
                s[f"danger_{a}"] = bool(s[f"danger_{a}"] or (hx + dx, hy + dy) in danger)
        s["snake_len"] = len(self.body)
        s["immortal_steps"] = self.immortal_steps
        s["obstacles"] = [[x, y] for (x, y) in self.obstacles]
        phase = []
        for st in self.structures:
            if st["phase"] == "GHOST":
                phase += [[x, y, st["timer"]] for (x, y) in st["cells"]]
            elif st["phase"] == "BLINKING":
                phase += [[x, y, -st["timer"]] for (x, y) in st["cells"]]
        s["phase_walls"] = phase
        s["patrols"] = []
        s["drone_path"] = []
        for dr in self.drones:
            dx, dy = dr.plan(self.solid)
            s["patrols"].append([dr.x, dr.y, dx, dy])
            ghost = copy.copy(dr)
            path = []
            for _ in range(DRONE_PATH_STEPS):
                ghost.step(self.solid)
                path.append([ghost.x, ghost.y])
            s["drone_path"].append(path)
        s["body"] = [[x, y] for (x, y) in self.body[1:MAX_BODY_CELLS + 1]]
        return {k: s[k] for k in STATE_KEYS_V9}

# ---------------------------------------------------------------- teacher

def base_move(solver, w):
    hx, hy = w.head
    if w.immortal_steps > 2 and w.food is not None:
        path = solver._a_star_search(w.head, w.food, set(w.obstacles), w.width, w.height, portals=w.portals)
        if path and len(path) > 1 and len(path) - 1 < w.immortal_steps - 1:
            a = solver._get_action(w.head, path[1])
            if not (len(w.body) > 1 and a == OPP[w.direction]):
                return a
    orig = w.obstacles
    if w.immortal_steps <= 1:
        w.obstacles = orig + list(w.drone_danger_cells())
    try:
        return solver.get_best_move(w)
    finally:
        w.obstacles = orig

def has_escape(solver, w, depth, budget, use_teacher):
    if depth == 0:
        return True
    if budget[0] <= 0:
        return False
    budget[0] -= 1
    order = list(DIRS)
    if use_teacher:
        first = base_move(solver, w)
        order.remove(first)
        order.insert(0, first)
    for m in order:
        if len(w.body) > 1 and m == OPP[w.direction]:
            continue
        nxt = w.clone()
        r = nxt.world_step(m)
        if r == "win":
            return True
        if r == "ok" and has_escape(solver, nxt, depth - 1, budget, False):
            return True
    return False

def survives(solver, w, move, horizon, budget=400):
    nxt = w.clone()
    r = nxt.world_step(move)
    if r == "dead":
        return False
    return r == "win" or has_escape(solver, nxt, horizon - 1, [budget], True)

def teacher_move(solver, w):
    move = base_move(solver, w)
    horizon = min(len(w.body) + 3, 20)
    hx, hy = w.head
    near_drone = any(abs(d.x - hx) + abs(d.y - hy) <= horizon + 2 for d in w.drones)
    changing_walls = any(s["phase"] != "SOLID" for s in w.structures)
    if not (near_drone or changing_walls or w.immortal_steps > 0):
        return move
    if survives(solver, w, move, horizon):
        return move
    alts = []
    for a, (dx, dy) in DIRS.items():
        if a == move or (len(w.body) > 1 and a == OPP[w.direction]):
            continue
        n = (hx + dx, hy + dy)
        fd = abs(n[0] - w.food[0]) + abs(n[1] - w.food[1]) if w.food else 0
        alts.append((fd, a))
    for _, a in sorted(alts):
        if survives(solver, w, a, horizon):
            return a
    return move

# ---------------------------------------------------------------- generation

def new_world(rng):
    drones = ui_drones() if rng.random() < 0.35 else random_drones(rng)
    w = World(20, 20, initial_length=rng.randint(3, 11), num_obstacles=rng.randint(1, 4),
              num_portals=2 if rng.random() < 0.6 else 0, fog_of_war=rng.random() < 0.3,
              sight_radius=5, seed=rng.randrange(1 << 30))
    w.setup(rng, drones)
    if any(d.pos in set(w.body) or d.pos in w.solid for d in drones):
        return None
    if rng.random() < IMMORTAL_APPLE_CHANCE:
        w.food_kind, w.food_ttl = "immortal", IMMORTAL_APPLE_TTL
    return w

def generate_chunk(args):
    n_samples, seed = args
    rng = random.Random(seed)
    random.seed(seed)
    solver = SnakeAStarSolver()
    samples = []
    st = {"episodes": 0, "overrides": 0, "deaths": 0, "immortal_samples": 0, "ghost_samples": 0}
    while len(samples) < n_samples:
        w = new_world(rng)
        if w is None:
            continue
        st["episodes"] += 1
        for _ in range(150):
            if len(samples) >= n_samples:
                break
            state = w.model_state()
            action = teacher_move(solver, w)
            dodge = action != base_move(solver, w)
            st["overrides"] += dodge
            st["immortal_samples"] += state["immortal_steps"] > 0
            st["ghost_samples"] += any(p[2] > 0 for p in state["phase_walls"])
            sample = {"state": state, "question": QUESTION,
                      "choices": ["UP", "DOWN", "LEFT", "RIGHT"], "label": action}
            samples.extend([sample] * (DODGE_REPEAT if dodge else 1))
            r = w.world_step(action)
            if r == "dead":
                st["deaths"] += 1
            if r != "ok":
                break
    return samples, st

def generate(target=60000, workers=12):
    chunk = target // workers
    jobs = [(chunk + (1 if i < target % workers else 0), 9000 + i) for i in range(workers)]
    t0 = time.time()
    print(f"Generating {target} V9 samples on {workers} processes...", flush=True)
    with Pool(workers) as pool:
        results = pool.map(generate_chunk, jobs)
    samples = [s for ch, _ in results for s in ch]
    random.Random(7).shuffle(samples)
    tot = {k: sum(r[k] for _, r in results) for k in results[0][1]}
    print(f"Done in {time.time() - t0:.0f}s: {len(samples)} samples | {tot}", flush=True)
    with open(DATASET_V9, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {DATASET_V9}", flush=True)

if __name__ == "__main__":
    generate(int(sys.argv[1]) if len(sys.argv) > 1 else 60000)
