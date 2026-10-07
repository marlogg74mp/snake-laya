"""
Multi-apple worlds (V12 rules), same episodes for everyone: window models v11w / v12 / v13 (/ v13d after DAgger)
vs the greedy teacher (value / distance) and the route teacher (V13).

Usage: python eval_v13.py [episodes] [sizes...] [--models=v11w,v13,v13d]
"""

import os
import sys
import time
from collections import Counter

import build_dataset_v12 as v12
import build_dataset_v13 as v13
import eval_v12 as e12
import teacher_route as tr

ROOT = os.path.dirname(os.path.abspath(__file__))
DIRS = {"v11w": "laya_snake_weights_v11w", "v12": "laya_snake_weights_v12",
        "v13": "laya_snake_weights_v13", "v13d": "laya_snake_weights_v13d"}

_state_for = e12.state_for
def state_for(policy, w, mem12, mem11):
    if policy.kind in ("v13", "v13d"):
        return v13.state13(w, mem12)
    return _state_for(policy, w, mem12, mem11)
e12.state_for = state_for

def run(policy, name, n, episodes):
    # eval_v12 plays the teacher for the string "teacher"; the route teacher is the same with the route picker
    route = name == "route"
    saved = v12.pick_target
    if route:
        v12.pick_target = tr.pick_target_route
    try:
        e12.run("teacher" if name in ("greedy", "route") else policy, name, n, episodes)
    finally:
        v12.pick_target = saved

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = [a for a in sys.argv[1:] if a.startswith("--models=")]
    names = opt[0].split("=", 1)[1].split(",") if opt else ["v11w", "v13", "v13d"]
    episodes = int(args[0]) if args else 100
    sizes = [int(x) for x in args[1:]] or [20, 30]
    models = {m: e12.Model(os.path.join(ROOT, DIRS[m]), m) for m in names if os.path.isdir(os.path.join(ROOT, DIRS[m]))}
    t0 = time.time()
    for n in sizes:
        for m, pol in models.items():
            run(pol, m, n, episodes)
        run(None, "greedy", n, episodes)
        run(None, "route", n, episodes)
    print(f"total {time.time() - t0:.0f}s", flush=True)
