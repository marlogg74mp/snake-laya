"""Rebuilds every snapshot from window_parity.mjs with the training code (build_dataset_v12.state12) and compares."""
import json, os, sys
from collections import deque
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import build_dataset_v9 as v9
import build_dataset_v12 as v12

DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

class DuelWorld(v12.World12):
    """In the duel the opponent is a wall for the model (state + dangers) but drones fly through it:
    drones bounce off structure walls only (self.solid), the opponent cells are in self.opp."""

    def is_collision(self, point):
        return super().is_collision(point) or tuple(point) in self.opp

def world(snap):
    w = DuelWorld.__new__(DuelWorld)
    sn = snap["snake"]
    w.width, w.height = snap["width"], snap["height"]
    w.body = [tuple(c) for c in sn["cells"]]
    w.head = w.body[0]
    w.direction, w.recent_actions = sn["dir"], list(sn["recent"])
    w.immortal_steps, w.fog_of_war, w.sight_radius = sn["immortal"], snap["fog"], 5
    w.portals, w.food, w.score, w.done, w.steps = [tuple(q) for q in snap["portals"]], None, 0, False, 0
    w.structures = [{"cells": [tuple(c) for c in st["cells"]], "phase": st["phase"], "timer": st["timer"]} for st in snap["structures"]]
    walls = [c for st in w.structures if st["phase"] != "GHOST" for c in st["cells"]]
    opp = []
    o = snap["opp"]
    if o["alive"]:
        opp = [tuple(c) for c in o["cells"]]
        dx, dy = DELTA[o["dir"]]
        f = (o["cells"][0][0] + dx, o["cells"][0][1] + dy)
        if 0 <= f[0] < w.width and 0 <= f[1] < w.height:
            opp.append(f)
    w.solid, w.opp = set(walls), set(opp)
    w.obstacles = walls + opp  # what the model sees as walls
    w.drones = [v9.Drone(d["kind"], d["x"], d["y"], d["dx"], d["dy"], lo=d["lo"], hi=d["hi"]) for d in snap["drones"]]
    w.items = {(f["x"], f["y"]): [f["type"], f["ttl"]] for f in snap["foods"] if f["type"] != "poison"}
    w.poisons = {(f["x"], f["y"]): f["ttl"] for f in snap["foods"] if f["type"] == "poison"}
    mem = v12.Memory12()
    mem.items = {tuple(map(int, k.split(","))): list(v) for k, v in snap["memory"]["items"]}
    mem.step = snap["memory"]["step"]
    mem.trail = deque((tuple(p) for p in snap["memory"]["trail"]), maxlen=200)
    return w, mem

snaps = json.load(open(os.path.join(os.path.dirname(__file__), "window_parity.json")))
bad = {}
for i, snap in enumerate(snaps):
    w, mem = world(snap)
    py = json.loads(json.dumps(v12.state12(w, mem)))
    for k in py:
        if py[k] != snap["js"][k]:
            bad.setdefault(k, []).append(i)
print(f"{len(snaps)} snapshots | identical: {sum(1 for i in range(len(snaps)) if not any(i in v for v in bad.values()))}")
for k, idx in bad.items():
    i = idx[0]
    w, mem = world(snaps[i])
    print(f"  {k}: {len(idx)} mismatches, e.g. #{i}\n    py {json.loads(json.dumps(v12.state12(w, mem)))[k]}\n    js {snaps[i]['js'][k]}")
