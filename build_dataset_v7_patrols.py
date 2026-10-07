"""
Dataset V7 Generator & Fine-Tuning Pipeline for Dynamic Cyber Patrols (Guard Drones).
Trains Laya ModernBERT to handle autonomous moving obstacles:
- Linear Ping-Pong Patrol Drones (🚨) and Circuit Waypoint Sentinels
- Lookahead Collision Avoidance (evaluating where drones will be on step t+1)
- Timing Choke Points and Lane Crossings behind Patrols
- Combined with Portals (🌀), Multi-Cell Walls (🧱), and Fog of War (🌫️)
"""

import json
import os
import random
import time
import requests
import torch
from snake_env import SnakeEnv
from a_star_solver import SnakeAStarSolver
from train_laya import train_laya_model

CRASHES_FILE = os.path.join(os.path.dirname(__file__), "crashes.jsonl")
DATASET_V7 = os.path.join(os.path.dirname(__file__), "dataset_v7_patrols.jsonl")
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "laya_snake_weights")

class PatrolSimulator:
    """
    Simulates linear and rectangular patrol bots inside the Snake arena.
    """
    def __init__(self, grid_w=20, grid_h=20, patrol_type="linear"):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.patrol_type = patrol_type

        if patrol_type == "linear_horiz":
            self.y = random.randint(4, grid_h - 5)
            self.min_x = random.randint(2, 5)
            self.max_x = random.randint(grid_w - 6, grid_w - 3)
            self.x = self.min_x
            self.dx = 1
            self.dy = 0
        elif patrol_type == "linear_vert":
            self.x = random.randint(4, grid_w - 5)
            self.min_y = random.randint(2, 5)
            self.max_y = random.randint(grid_h - 6, grid_h - 3)
            self.y = self.min_y
            self.dx = 0
            self.dy = 1
        else: # circuit
            self.cx = random.randint(5, grid_w - 7)
            self.cy = random.randint(5, grid_h - 7)
            self.w = 5
            self.h = 5
            self.waypoints = [
                (self.cx, self.cy),
                (self.cx + self.w, self.cy),
                (self.cx + self.w, self.cy + self.h),
                (self.cx, self.cy + self.h)
            ]
            self.wp_idx = 0
            self.x, self.y = self.waypoints[0]
            self.dx = 1
            self.dy = 0

    def step(self):
        if self.patrol_type == "linear_horiz":
            self.x += self.dx
            if self.x >= self.max_x:
                self.x = self.max_x
                self.dx = -1
            elif self.x <= self.min_x:
                self.x = self.min_x
                self.dx = 1
        elif self.patrol_type == "linear_vert":
            self.y += self.dy
            if self.y >= self.max_y:
                self.y = self.max_y
                self.dy = -1
            elif self.y <= self.min_y:
                self.y = self.min_y
                self.dy = 1
        else: # circuit waypoint follow
            target = self.waypoints[(self.wp_idx + 1) % len(self.waypoints)]
            if self.x < target[0]: self.x += 1
            elif self.x > target[0]: self.x -= 1
            elif self.y < target[1]: self.y += 1
            elif self.y > target[1]: self.y -= 1
            
            if (self.x, self.y) == target:
                self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)

    @property
    def pos(self):
        return (self.x, self.y)

    def predict_next(self, steps=1):
        """Simulates future coordinates for lookahead danger checks."""
        sim = PatrolSimulator(self.grid_w, self.grid_h, self.patrol_type)
        sim.x = self.x; sim.y = self.y; sim.dx = self.dx; sim.dy = self.dy
        if hasattr(self, "waypoints"):
            sim.waypoints = list(self.waypoints)
            sim.wp_idx = self.wp_idx
        for _ in range(steps):
            sim.step()
        return (sim.x, sim.y)

def generate_v7_dataset(target_samples: int = 40000):
    solver = SnakeAStarSolver()
    samples = []

    print(f"Generating {target_samples} Patrol-Aware samples for Dataset V7...")

    while len(samples) < target_samples:
        p_type = random.choice(["linear_horiz", "linear_vert", "circuit"])
        patrol = PatrolSimulator(20, 20, p_type)

        is_fog = random.random() < 0.35
        num_obs = random.randint(0, 3)
        num_ports = 2 if random.random() < 0.6 else 0

        env = SnakeEnv(width=20, height=20, num_obstacles=num_obs, num_portals=num_ports, fog_of_war=is_fog, sight_radius=5)
        # Inject dynamic patrol into environment
        env.patrol_pos = patrol.pos

        compact_state = env.get_compact_state()
        done = False
        step_count = 0

        while not done and len(samples) < target_samples and step_count < 140:
            # Predict patrol's next location for safety reservation
            next_patrol_pos = patrol.predict_next(steps=1)
            
            # Augment obstacles with current and next patrol coordinates
            orig_obs = list(env.obstacles)
            env.obstacles = orig_obs + [patrol.pos, next_patrol_pos]

            action = solver.get_best_move(env)

            # Restore original static obstacles
            env.obstacles = orig_obs

            # Enrich state representation for training sample
            p_state = dict(compact_state)
            p_state["patrols"] = [[patrol.x, patrol.y]]
            
            # Check lookahead danger including moving patrol
            hx, hy = env.head
            p_state["danger_UP"] = p_state["danger_UP"] or ((hx, hy - 1) in [patrol.pos, next_patrol_pos])
            p_state["danger_DOWN"] = p_state["danger_DOWN"] or ((hx, hy + 1) in [patrol.pos, next_patrol_pos])
            p_state["danger_LEFT"] = p_state["danger_LEFT"] or ((hx - 1, hy) in [patrol.pos, next_patrol_pos])
            p_state["danger_RIGHT"] = p_state["danger_RIGHT"] or ((hx + 1, hy) in [patrol.pos, next_patrol_pos])

            sample = {
                "state": p_state,
                "question": "What is the next safe move avoiding patrols and walls towards food?",
                "choices": ["UP", "DOWN", "LEFT", "RIGHT"],
                "label": action
            }
            samples.append(sample)

            # Advance environment and patrol
            next_state, reward, done, info = env.step(action)
            patrol.step()
            env.patrol_pos = patrol.pos

            # Check collision with patrol drone
            if env.head == patrol.pos:
                done = True

            compact_state = next_state
            step_count += 1

    print(f"Generated {len(samples)} dynamic patrol training samples.")

    print(f"Writing {len(samples)} samples to {DATASET_V7}...")
    with open(DATASET_V7, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return DATASET_V7

def run_v7_patrol_training():
    v7_file = generate_v7_dataset(target_samples=40000)
    print("\n=== Retraining Laya Model V7 (Dynamic Patrol Avoidance & Lane Timing) on CUDA ===")
    train_laya_model(
        dataset_path=v7_file,
        output_dir=MODEL_OUTPUT_DIR,
        base_model_name="answerdotai/ModernBERT-base",
        epochs=3,
        batch_size=64,
        learning_rate=3e-5
    )
    print("\n[SUCCESS] Laya V7 Patrol Model Retraining Complete!")

    try:
        resp = requests.post("http://localhost:9500/reload_weights", timeout=10)
        print(f"[INFO] Automatic weights hot-reload response: {resp.json()}")
    except Exception as e:
        print(f"[WARNING] Could not auto-trigger hot-reload: {e}")

if __name__ == "__main__":
    run_v7_patrol_training()
