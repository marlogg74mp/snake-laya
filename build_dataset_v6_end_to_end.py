"""
End-to-End Pure Neural Pipeline (Dataset V6) for Laya ModernBERT.
Trains the neural network to handle ALL mechanics autonomously:
- Food Seeking with Wormhole Portals (🌀) and Multi-Cell Wall Obstacles (🧱)
- Fog of War (🌫️) Partial Observability Active Exploration & Wide Arena Sweeping
- Long-Snake Anti-Trapping, Tail Following, and Self-Preservation
- Hard Negative Crash Telemetry Corrections
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
DATASET_V6 = os.path.join(os.path.dirname(__file__), "dataset_v6.jsonl")
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "laya_snake_weights")

def generate_v6_dataset(target_samples: int = 35000):
    solver = SnakeAStarSolver()
    samples = []

    print(f"Generating {target_samples} end-to-end multi-behavior training samples for Dataset V6...")

    while len(samples) < target_samples:
        # Diversity mix:
        # 40% Standard food navigation with walls and portals
        # 35% Fog of War exploration & target interception
        # 25% Longer snake survival & tail following
        mode_roll = random.random()
        num_obs = random.randint(0, 4)
        num_ports = 2 if random.random() < 0.8 else 0

        if mode_roll < 0.40:
            # Standard Mode
            env = SnakeEnv(width=20, height=20, num_obstacles=num_obs, num_portals=num_ports, fog_of_war=False)
        elif mode_roll < 0.75:
            # Fog of War Mode (tests active exploration when food is unknown)
            env = SnakeEnv(width=20, height=20, num_obstacles=num_obs, num_portals=num_ports, fog_of_war=True, sight_radius=5)
        else:
            # Long Snake Mode (starts with length 8-14)
            init_len = random.randint(8, 14)
            env = SnakeEnv(width=20, height=20, initial_length=init_len, num_obstacles=num_obs, num_portals=num_ports, fog_of_war=random.choice([True, False]))

        compact_state = env.get_compact_state()
        done = False
        step_in_game = 0

        while not done and len(samples) < target_samples and step_in_game < 120:
            action = solver.get_best_move(env)

            sample = {
                "state": compact_state,
                "question": "What is the next safe move towards food?",
                "choices": ["UP", "DOWN", "LEFT", "RIGHT"],
                "label": action
            }
            samples.append(sample)

            next_state, reward, done, info = env.step(action)
            compact_state = next_state
            step_in_game += 1

    print(f"Generated {len(samples)} core end-to-end samples (Standard, Fog Exploration, Long Snake).")

    # Integrate crash telemetries if present
    if os.path.exists(CRASHES_FILE):
        print(f"Processing crash telemetry states from {CRASHES_FILE}...")
        corrected_crashes = 0
        with open(CRASHES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    c_state = item.get("state", {})

                    temp_env = SnakeEnv(width=20, height=20)
                    head = c_state.get("head_pos", [10, 10])
                    food = c_state.get("food_pos", [15, 10])
                    obs = c_state.get("obstacles", [])
                    ports = c_state.get("portals", [])
                    fog = c_state.get("fog_of_war", False)
                    s_rad = c_state.get("sight_radius", 5)

                    temp_env.head = (head[0], head[1])
                    temp_env.body = [(head[0], head[1]), (head[0] - 1, head[1]), (head[0] - 2, head[1])]
                    temp_env.food = (food[0], food[1]) if food[0] != -1 else None
                    temp_env.obstacles = [(o[0], o[1]) for o in obs]
                    temp_env.portals = [(p[0], p[1]) for p in ports]
                    temp_env.fog_of_war = fog
                    temp_env.sight_radius = s_rad

                    correct_action = solver.get_best_move(temp_env)

                    sample = {
                        "state": c_state,
                        "question": "What is the next safe move towards food?",
                        "choices": ["UP", "DOWN", "LEFT", "RIGHT"],
                        "label": correct_action
                    }
                    samples.append(sample)
                    corrected_crashes += 1

        print(f"Added {corrected_crashes} corrected crash samples.")

    print(f"Writing total {len(samples)} samples to {DATASET_V6}...")
    with open(DATASET_V6, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return DATASET_V6

def run_v6_training():
    v6_file = generate_v6_dataset(target_samples=35000)
    print("\n=== Retraining Laya Model V6 (End-to-End Pure Neural Controller) on CUDA ===")
    train_laya_model(
        dataset_path=v6_file,
        output_dir=MODEL_OUTPUT_DIR,
        base_model_name="answerdotai/ModernBERT-base",
        epochs=3,
        batch_size=64,
        learning_rate=3e-5
    )
    print("\n[SUCCESS] Laya V6 End-to-End Pure Neural Model Training Complete!")

    # Hot-reload weights on web server
    try:
        resp = requests.post("http://localhost:9500/reload_weights", timeout=10)
        print(f"[INFO] Automatic weights hot-reload response: {resp.json()}")
    except Exception as e:
        print(f"[WARNING] Could not auto-trigger hot-reload: {e}")

if __name__ == "__main__":
    run_v6_training()
