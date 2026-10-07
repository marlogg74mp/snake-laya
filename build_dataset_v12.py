"""
Dataset V12 — window state (V11w) + several apples of different value + poison + a memory that matters.

Apples (as in the games): apple +1 · golden +3 · immortal +2 and 50 steps of immortality ·
shrink +2 and the snake gets 2 cells shorter · poison: deadly unless immortal.
Non-apple items live 60 steps. 2..4 good items and 0..2 poisons are on the board at once.

The teacher picks the most profitable item, not the nearest one:
    utility = effective value / (A* path length + 3), items it cannot reach before they expire are skipped;
    immortality is worth more with drones nearby, shrink is worth more for a long snake.
Then the V9 teacher (A* + time-space drone check) walks there. In fog the teacher only knows items in sight
or in memory, exactly like the model.

State (all relative to the head; KEYS below):
  foods_near  [[dx, dy, value, ttl]]  items inside the 11x11 window (ttl -1 = never expires)
  foods_far   [[dir, dist, value]]   items out of the window (not in fog); dist bucket: "mid" <= 15 < "far"
  memory_food [[dx, dy, value, age]] items seen before, out of sight now (fog), dx/dy clipped to ±20
  poison      [[dx, dy]]             poisons inside the window
  + everything V11w has except food_dir/food_rel (edges, body, walls, ghost walls, drones, explored, …)
"""

import json
import os
import random
import sys
import time
from collections import deque
from multiprocessing import Pool

import build_dataset_v9 as v9
import build_dataset_v11w as w11
from a_star_solver import SnakeAStarSolver

ROOT = os.path.dirname(os.path.abspath(__file__))
R = w11.R
VALUE = {"apple": 1, "golden": 3, "immortal": 2, "shrink": 2}
ITEM_TTL = 60
KEYS = ["current_dir", "recent_actions", "danger_UP", "danger_DOWN", "danger_LEFT", "danger_RIGHT",
        "foods_near", "foods_far", "memory_food", "poison", "edges", "snake_len", "immortal_steps",
        "body", "walls", "ghost_walls", "drones", "drone_path", "fog_of_war", "explored"]

def dir8(dx, dy):
    v = "UP" if dy < 0 else "DOWN" if dy > 0 else ""
    h = "LEFT" if dx < 0 else "RIGHT" if dx > 0 else ""
    return f"{v}_{h}" if v and h else v or h or "SAME"

class World12(v9.World):
    """V9 world with several food items and poisons (positions in self.items: {(x, y): [kind, ttl]})."""

    def setup12(self, rng):
        self.items = {}
        self.poisons = {}
        for _ in range(rng.randint(2, 4)):
            self._add_item(rng)
        for _ in range(rng.choice([0, 0, 1, 1, 2])):
            self._add_poison(rng)

    def _free(self, rng):
        occ = set(self.body) | {c for s in self.structures for c in s["cells"]} | set(self.items) | set(self.poisons)
        occ |= {d.pos for d in self.drones}
        for _ in range(200):
            c = (rng.randrange(self.width), rng.randrange(self.height))
            if c not in occ:
                return c
        return None

    def _add_item(self, rng):
        c = self._free(rng)
        if c:
            kind = rng.choices(["apple", "golden", "immortal", "shrink"], [55, 22, 11, 12])[0]
            self.items[c] = [kind, None if kind == "apple" else ITEM_TTL]

    def _add_poison(self, rng):
        c = self._free(rng)
        if c:
            self.poisons[c] = ITEM_TTL

    def is_collision(self, point):
        if super().is_collision(point):
            return True
        return getattr(self, "immortal_steps", 0) <= 0 and tuple(point) in getattr(self, "poisons", {})

    def _spawn_food(self):
        return None  # food is managed in self.items

    def placeholder(self):
        """A stand-in goal for the A* solver when nothing is known (always used together with fog = hidden)."""
        return (self.width // 2, self.height // 2)

    def clone(self):
        w = super().clone()
        w.items = {k: list(v) for k, v in self.items.items()}
        w.poisons = dict(self.poisons)
        return w

    def world_step(self, action):
        """V9 step, eating whichever item the head lands on; then item lifetimes and respawns."""
        hx, hy = self.head
        d = v9.DIRS[action if not (len(self.body) > 1 and action == v9.OPP[self.direction]) else self.direction]
        target = (hx + d[0], hy + d[1])
        eaten = self.items.get(target)
        goal = self.food  # the teacher's target (kept for its look-ahead rollouts)
        self.food = target if eaten else None
        self.food_kind = "immortal" if eaten and eaten[0] == "immortal" else "standard"
        poisoned = target in self.poisons
        r = super().world_step(action)
        if r == "dead":
            return r
        if poisoned:  # immortal: poison is eaten without effect
            self.poisons.pop(target, None)
        if eaten:
            self.items.pop(target, None)
            self.score += VALUE[eaten[0]] - 1  # SnakeEnv already added +1
            if eaten[0] == "shrink":
                for _ in range(2):
                    if len(self.body) > 3:
                        self.body.pop()
        self.food = goal if goal and goal != target else self.placeholder()
        for c in list(self.items):
            if self.items[c][1] is not None:
                self.items[c][1] -= 1
                if self.items[c][1] <= 0:
                    del self.items[c]
        for c in list(self.poisons):
            self.poisons[c] -= 1
            if self.poisons[c] <= 0:
                del self.poisons[c]
        rng = self.wrng
        while len(self.items) < 2 or (len(self.items) < 4 and rng.random() < 0.03):
            self._add_item(rng)
        if len(self.poisons) < 2 and rng.random() < 0.02:
            self._add_poison(rng)
        return r

class Memory12:
    """Items the snake has seen: {(x, y): [kind, step seen]}; forgotten when their cell is seen empty."""

    def __init__(self):
        self.items = {}
        self.trail = deque(maxlen=w11.EXPLORE_STEPS)
        self.step = 0

    def visible(self, w, x, y):
        hx, hy = w.head
        return abs(x - hx) <= R and abs(y - hy) <= R and (not w.fog_of_war or abs(x - hx) + abs(y - hy) <= R)

    def update(self, w):
        self.step += 1
        self.trail.append(w.head)
        for c in list(self.items):
            if self.visible(w, *c) and c not in w.items:
                del self.items[c]
        for c, (kind, _) in w.items.items():
            if self.visible(w, *c):
                self.items[c] = [kind, self.step]
        if len(self.items) > 4:
            for c in sorted(self.items, key=lambda c: self.items[c][1])[:-4]:
                del self.items[c]

def known_items(w, mem):
    """Items the snake may know about: everything without fog, otherwise what is in sight or remembered."""
    if not w.fog_of_war:
        return {c: (k, t) for c, (k, t) in w.items.items()}
    out = {c: (k, t) for c, (k, t) in w.items.items() if mem.visible(w, *c)}
    for c, (k, _) in mem.items.items():
        out.setdefault(c, (k, None))
    return out

def state12(w, mem):
    s = w11.window_state(w, w11.Memory())  # window parts (walls, body, drones, edges, dangers …)
    hx, hy = w.head
    clip = lambda v: max(-20, min(20, v))
    near, far, remembered = [], [], []
    for (x, y), (kind, ttl) in sorted(w.items.items(), key=lambda kv: abs(kv[0][0] - hx) + abs(kv[0][1] - hy)):
        dx, dy = x - hx, y - hy
        if mem.visible(w, x, y):
            near.append([dx, dy, VALUE[kind], -1 if ttl is None else ttl])
        elif not w.fog_of_war:
            dist = abs(dx) + abs(dy)
            far.append([dir8(dx, dy), "mid" if dist <= 15 else "far", VALUE[kind]])
    if w.fog_of_war:
        for (x, y), (kind, seen) in sorted(mem.items.items(), key=lambda kv: -kv[1][1]):
            if not mem.visible(w, x, y):
                remembered.append([clip(x - hx), clip(y - hy), VALUE[kind], mem.step - seen])
    danger = {}
    for a, (dx, dy) in v9.DIRS.items():
        danger[a] = s[f"danger_{a}"] or (w.immortal_steps <= 1 and (hx + dx, hy + dy) in w.poisons)
    explored = s["explored"]
    if mem.trail:
        cnt = [sum(1 for (x, y) in mem.trail if y < hy), sum(1 for (x, y) in mem.trail if y > hy),
               sum(1 for (x, y) in mem.trail if x < hx), sum(1 for (x, y) in mem.trail if x > hx)]
        explored = [min(9, round(9 * c / len(mem.trail))) for c in cnt]
    state = dict(s, foods_near=near[:4], foods_far=far[:4], memory_food=remembered[:4],
                 poison=[[x - hx, y - hy] for (x, y) in w.poisons if abs(x - hx) <= R and abs(y - hy) <= R],
                 explored=explored, **{f"danger_{a}": v for a, v in danger.items()})
    return {k: state[k] for k in KEYS}

def pick_target(solver, w, mem):
    """The most profitable known item: value / (path length + 3), reachable before it expires."""
    known = known_items(w, mem)
    if not known:
        return None
    hx, hy = w.head
    blocked = set(w.body) | set(w.obstacles) | (set(w.poisons) if w.immortal_steps <= 1 else set())
    drones_near = any(abs(d.x - hx) + abs(d.y - hy) <= 8 for d in w.drones)
    best, best_u = None, -1.0
    for c, (kind, ttl) in known.items():
        path = solver._a_star_search(w.head, c, blocked, w.width, w.height, portals=w.portals)
        if not path:
            continue
        steps = len(path) - 1
        if ttl is not None and steps >= ttl:
            continue
        value = VALUE[kind] + (1.5 if kind == "immortal" and drones_near else 0) + (1.0 if kind == "shrink" and len(w.body) >= 12 else 0)
        u = value / (steps + 3)
        if u > best_u:
            best, best_u = c, u
    return best

def teacher12(solver, w, mem):
    """Returns (teacher move, plain A* move to the same target): they differ when the teacher dodges a drone."""
    target = pick_target(solver, w, mem)
    saved = (w.food, w.fog_of_war, w.obstacles)
    w.obstacles = w.obstacles + (list(w.poisons) if w.immortal_steps <= 1 else [])
    if target:
        w.food, w.fog_of_war = target, False
    else:
        w.food = w.placeholder()
        w.fog_of_war = True  # nothing known: explore (the placeholder counts as hidden unless it is close)
    try:
        return v9.teacher_move(solver, w), v9.base_move(solver, w)
    finally:
        w.food, w.fog_of_war, w.obstacles = saved

def nearest_known(w, mem):
    known = known_items(w, mem)
    hx, hy = w.head
    return min(known, key=lambda c: abs(c[0] - hx) + abs(c[1] - hy)) if known else None

def make_world(rng, n):
    w0 = w11.make_world(rng, n)
    if w0 is None:
        return None
    w = World12.__new__(World12)
    w.__dict__.update(w0.__dict__)
    w.setup12(rng)
    return w

def generate_chunk(args):
    n_samples, seed = args
    rng = random.Random(seed)
    random.seed(seed)
    solver = SnakeAStarSolver()
    samples, st = [], {"episodes": 0, "dodges": 0, "memory_moves": 0, "not_nearest": 0, "deaths": 0, "score": 0}
    while len(samples) < n_samples:
        w = make_world(rng, rng.randint(12, 40))
        if w is None:
            continue
        mem = Memory12()
        mem.update(w)
        st["episodes"] += 1
        for _ in range(200):
            if len(samples) >= n_samples:
                break
            state = state12(w, mem)
            action, plain = teacher12(solver, w, mem)
            dodge = action != plain
            st["not_nearest"] += bool(state["foods_near"] or state["foods_far"]) and pick_target(solver, w, mem) != nearest_known(w, mem)
            st["dodges"] += dodge
            st["memory_moves"] += bool(state["memory_food"])
            sample = {"state": state, "question": v9.QUESTION, "choices": ["UP", "DOWN", "LEFT", "RIGHT"], "label": action}
            samples.extend([sample] * (v9.DODGE_REPEAT if dodge else 1))
            r = w.world_step(action)
            if r == "dead":
                st["deaths"] += 1
            if r != "ok":
                break
            mem.update(w)
        st["score"] += w.score
    return samples[:n_samples], st

def generate(target=80000, workers=12):
    jobs = [(target // workers + (1 if i < target % workers else 0), 12_000 + i) for i in range(workers)]
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(generate_chunk, jobs)
    samples = [s for ch, _ in res for s in ch]
    random.Random(7).shuffle(samples)
    tot = {k: sum(r[k] for _, r in res) for k in res[0][1]}
    with open(os.path.join(ROOT, "dataset_v12.jsonl"), "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"V12: {len(samples)} samples in {time.time() - t0:.0f}s | {tot}", flush=True)

if __name__ == "__main__":
    generate(int(sys.argv[1]) if len(sys.argv) > 1 else 80000)
