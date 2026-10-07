"""
Balanced Dataset V5 Builder and Fine-Tuning Pipeline for Laya Model with Portals & Speed Boost Orbs.
Generates 30,000 multi-apple, obstacle-aware, and portal-aware optimal moves with Portals (🌀) and Walls (🧱).
"""

import json
import os
import random
import time
import torch
from snake_env import SnakeEnv
from a_star_solver import SnakeAStarSolver
from train_laya import train_laya_model

CRASHES_FILE = os.path.join(os.path.dirname(__file__), "crashes.jsonl")
DATASET_V5 = os.path.join(os.path.dirname(__file__), "dataset_v5.jsonl")
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "laya_snake_weights")

def generate_v5_balanced_dataset(target_samples: int = 30000):
    solver = SnakeAStarSolver()
    samples = []

    print(f"Generating {target_samples} A* food-seeking, portal-aware & obstacle-aware samples for Dataset V5...")

    while len(samples) < target_samples:
        # Vary obstacles count (0 to 4) and portals (2 = 1 pair)
        num_obs = random.randint(0, 4)
        env = SnakeEnv(width=20, height=20, num_obstacles=num_obs, num_portals=2)
        compact_state = env.get_compact_state()
        done = False
        
        while not done and len(samples) < target_samples:
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

    print(f"Generated {len(samples)} core A* food, portal, & obstacle-aware samples.")

    # Process crash telemetries if present
    if os.path.exists(CRASHES_FILE):
        print(f"Processing crash states from {CRASHES_FILE}...")
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
                    
                    temp_env.head = (head[0], head[1])
                    temp_env.body = [(head[0], head[1]), (head[0]-1, head[1]), (head[0]-2, head[1])]
                    temp_env.food = (food[0], food[1])
                    temp_env.obstacles = [(o[0], o[1]) for o in obs]
                    temp_env.portals = [(p[0], p[1]) for p in ports]
                    
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

    print(f"Writing total {len(samples)} balanced samples to {DATASET_V5}...")
    with open(DATASET_V5, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return DATASET_V5

def run_v5_retraining():
    v5_file = generate_v5_balanced_dataset(target_samples=30000)
    print("\n=== Retraining Laya Model V5 (Food-Seeking + Portal & Obstacle-Aware) on CUDA ===")
    train_laya_model(
        dataset_path=v5_file,
        output_dir=MODEL_OUTPUT_DIR,
        base_model_name="answerdotai/ModernBERT-base",
        epochs=3,
        batch_size=64,
        learning_rate=3e-5
    )
    print("\n[SUCCESS] Laya V5 Model Retraining Complete!")
    
    # Automatically notify local web server to hot-reload new PyTorch weights into CUDA
    try:
        import requests
        resp = requests.post("http://localhost:9500/reload_weights", timeout=10)
        print(f"[INFO] Automatic weights hot-reload response: {resp.json()}")
    except Exception as e:
        print(f"[WARNING] Could not auto-trigger hot-reload: {e}")

if __name__ == "__main__":
    run_v5_retraining()
