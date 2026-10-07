"""
Rule-parity harness, part 1 (Python): runs random V9 worlds through build_dataset_v9.World and records
the initial world + per-step (model_state, action, result). part 2 (parity_v9.mjs) replays the same worlds
and actions in web3d/js/game.js and compares every state byte-for-byte.

Random choices that differ by implementation are pinned on both sides: new wall timers = lower bound
(randint(a, b) -> a), respawned food = first free cell, always a standard apple.

Usage:  python web3d/tools/parity_v9.py [worlds] [out.json]   (CPU only, does not touch the model)
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import build_dataset_v9 as v9  # noqa: E402


class PinnedRng:
    def randint(self, a, b):
        return a

    def random(self):
        return 0.99  # never an immortality apple on respawn

    def choice(self, seq):
        return seq[0]


def export_world(w):
    drones = []
    for d in w.drones:
        e = {"kind": d.kind, "x": d.x, "y": d.y, "dx": d.dx, "dy": d.dy}
        if d.kind in ("h", "v"):
            e.update(lo=d.lo, hi=d.hi)
        else:
            e.update(waypoints=[list(p) for p in d.waypoints], wpIdx=d.wp_idx, wpSign=d.wp_sign)
        drones.append(e)
    food = []
    if w.food is not None:
        food.append({"x": w.food[0], "y": w.food[1], "type": "immortal" if w.food_kind == "immortal" else "standard",
                     "ttl": w.food_ttl if w.food_kind == "immortal" else None})
    return {
        "snake": [list(c) for c in w.body], "direction": w.direction, "recentActions": list(w.recent_actions),
        "structures": [{"cells": [list(c) for c in s["cells"]], "phase": s["phase"], "timer": s["timer"]} for s in w.structures],
        "portals": [list(p) for p in w.portals], "drones": drones, "foods": food,
        "immortalSteps": w.immortal_steps, "fog": bool(w.fog_of_war), "sightRadius": w.sight_radius,
    }


def pick_action(w, rng):
    """Mostly survive, often greedy toward food, sometimes random (also exercises deaths)."""
    moves = [m for m in v9.DIRS if not (len(w.body) > 1 and m == v9.OPP[w.direction])]
    if rng.random() < 0.05:
        return rng.choice(moves)
    safe = [m for m in moves if w.clone().world_step(m) != "dead"]
    if not safe:
        return rng.choice(moves)
    if w.food is not None and rng.random() < 0.7:
        hx, hy = w.head
        return min(safe, key=lambda m: abs(hx + v9.DIRS[m][0] - w.food[0]) + abs(hy + v9.DIRS[m][1] - w.food[1]))
    return rng.choice(safe)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity_v9_traces.json")
    rng = random.Random(12345)
    traces = []
    while len(traces) < n:
        w = v9.new_world(rng)
        if w is None:
            continue
        w.wrng = PinnedRng()
        w.rng = PinnedRng()
        if rng.random() < 0.4:
            w.immortal_steps = rng.randint(1, 50)          # start some worlds already immortal
        if rng.random() < 0.3:                               # and put an immortality apple on the board
            w.food_kind, w.food_ttl = "immortal", rng.randint(1, 60)
        world = export_world(w)
        steps = []
        for _ in range(250):
            state = w.model_state()
            action = pick_action(w, rng)
            result = w.world_step(action)
            steps.append({"state": state, "action": action, "result": result})
            if result != "ok":
                break
        traces.append({"world": world, "steps": steps, "final": w.model_state() if steps[-1]["result"] == "ok" else None})
    with open(out, "w", encoding="utf-8") as f:
        json.dump(traces, f)
    tot = sum(len(t["steps"]) for t in traces)
    deaths = sum(t["steps"][-1]["result"] == "dead" for t in traces)
    print(f"wrote {len(traces)} traces, {tot} steps, {deaths} deaths -> {out}")


if __name__ == "__main__":
    main()
