"""
Route-planning target choice for the V12 teacher (orienteering): instead of the single best value/distance item,
try every order of up to 3 known items and take the first item of the route with the highest
time-discounted value  sum(value_i * GAMMA ** arrival_i), skipping routes that reach an item after it expires.
Head -> item distances are A* paths around the body; item -> item distances are BFS over the walls only.

python teacher_route.py [episodes]   -> greedy vs route teacher on the same multi-apple episodes (CPU only)
"""

import itertools
import random
import sys
import time
from collections import Counter, deque

import build_dataset_v12 as v12
import build_dataset_v9 as v9
from a_star_solver import SnakeAStarSolver

GAMMA = 0.95
MAX_ROUTE = 3

def bfs_dist(w, start, goals, blocked):
    """Grid distances from start to each goal cell (walls only), None if unreachable."""
    goals, seen, q, out = set(goals), {start}, deque([(start, 0)]), {}
    while q and len(out) < len(goals):
        (x, y), d = q.popleft()
        if (x, y) in goals:
            out[(x, y)] = d
        for dx, dy in v9.DIRS.values():
            n = (x + dx, y + dy)
            if 0 <= n[0] < w.width and 0 <= n[1] < w.height and n not in seen and n not in blocked:
                seen.add(n)
                q.append((n, d + 1))
    return out

def value_of(w, kind):
    hx, hy = w.head
    drones_near = any(abs(d.x - hx) + abs(d.y - hy) <= 8 for d in w.drones)
    return v12.VALUE[kind] + (1.5 if kind == "immortal" and drones_near else 0) + (1.0 if kind == "shrink" and len(w.body) >= 12 else 0)

def pick_target_route(solver, w, mem):
    known = v12.known_items(w, mem)
    if not known:
        return None
    blocked_head = set(w.body) | set(w.obstacles) | (set(w.poisons) if w.immortal_steps <= 1 else set())
    walls = set(w.obstacles) | (set(w.poisons) if w.immortal_steps <= 1 else set())
    first = {}
    for c in known:
        path = solver._a_star_search(w.head, c, blocked_head, w.width, w.height, portals=w.portals)
        if path:
            first[c] = len(path) - 1
    if not first:
        return None
    items = list(first)
    between = {c: bfs_dist(w, c, [d for d in items if d != c], walls) for c in items}
    best, best_v = None, -1.0
    for k in range(1, min(MAX_ROUTE, len(items)) + 1):
        for route in itertools.permutations(items, k):
            t, total, ok = first[route[0]], 0.0, True
            for i, c in enumerate(route):
                if i:
                    step = between[route[i - 1]].get(c)
                    if step is None:
                        ok = False
                        break
                    t += step
                kind, ttl = known[c]
                if ttl is not None and t >= ttl:
                    ok = False
                    break
                total += value_of(w, kind) * GAMMA ** t
            if ok and total > best_v:
                best, best_v = route[0], total
    return best

def compare(episodes=40, sizes=(20, 30)):
    solver = SnakeAStarSolver()
    for n in sizes:
        for name, picker in (("greedy", v12.pick_target), ("route", pick_target_route)):
            v12.pick_target = picker if name == "route" else ORIGINAL
            scores, steps, eaten, t0 = [], [], Counter(), time.time()
            for i in range(episodes):
                rng = random.Random(150_000 + n * 1000 + i)
                random.seed(150_000 + n * 1000 + i)
                w = None
                while w is None:
                    w = v12.make_world(rng, n)
                mem = v12.Memory12()
                mem.update(w)
                for t in range(300):
                    action, _ = v12.teacher12(solver, w, mem)
                    before = dict(w.items)
                    r = w.world_step(action)
                    for c, (kind, _) in before.items():
                        if c not in w.items and c == w.head:
                            eaten[kind] += 1
                    if r != "ok":
                        break
                    mem.update(w)
                scores.append(w.score)
                steps.append(t + 1)
            tot = sum(eaten.values()) or 1
            print(f"{n}x{n} {name:6} | avg score {sum(scores) / episodes:5.2f} | avg steps {sum(steps) / episodes:4.0f} | "
                  f"eaten golden {eaten['golden'] / tot:4.0%} apple {eaten['apple'] / tot:4.0%} immortal {eaten['immortal'] / tot:4.0%} "
                  f"shrink {eaten['shrink'] / tot:4.0%} | {time.time() - t0:.0f}s", flush=True)
    v12.pick_target = ORIGINAL

ORIGINAL = v12.pick_target

if __name__ == "__main__":
    compare(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
