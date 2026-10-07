"""
Balanced Dataset V3 Builder and Fine-Tuning Pipeline for Laya Model.
Fixes overfitting on corner-avoidance by pairing all crash states with true A* food-seeking safe actions
and generating 30,000 multi-apple/poison aware optimal moves.
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
DATASET_V3 = os.path.join(os.path.dirname(__file__), "dataset_v3.jsonl")
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "laya_snake_weights")

def generate_v3_balanced_dataset(target_samples: int = 30000):
    solver = SnakeAStarSolver()
    samples = []

    print(f"Generating {target_samples} A* food-seeking & hazard-aware samples for Dataset V3...")
    
    # 1. Generate core A* optimal games
    env = SnakeEnv(width=20, height=20)
    
    while len(samples) < target_samples:
        compact_state = env.reset()
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

    print(f"Generated {len(samples)} core A* food-seeking samples.")

    # 2. Process crashes.jsonl by recalculating TRUE A* food-seeking actions for crash states
    if os.path.exists(CRASHES_FILE):
        print(f"Fixing crash states from {CRASHES_FILE} using A* solver...")
        corrected_crashes = 0
        
        with open(CRASHES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    c_state = item.get("state", {})
                    
                    # Create temporary environment matching crash head and food positions
                    temp_env = SnakeEnv(width=20, height=20)
                    head = c_state.get("head_pos", [10, 10])
                    food = c_state.get("food_pos", [15, 10])
                    
                    temp_env.head = (head[0], head[1])
                    temp_env.body = [(head[0], head[1]), (head[0]-1, head[1]), (head[0]-2, head[1])]
                    temp_env.food = (food[0], food[1])
                    
                    correct_action = solver.get_best_move(temp_env)
                    
                    sample = {
                        "state": c_state,
                        "question": "What is the next safe move towards food?",
                        "choices": ["UP", "DOWN", "LEFT", "RIGHT"],
                        "label": correct_action
                    }
                    samples.append(sample)
                    corrected_crashes += 1

        print(f"Corrected {corrected_crashes} crash states with true A* food-seeking targets.")

    print(f"Writing total {len(samples)} balanced samples to {DATASET_V3}...")
    with open(DATASET_V3, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return DATASET_V3

def run_v3_retraining():
    v3_file = generate_v3_balanced_dataset(target_samples=30000)
    print("\n=== Retraining Laya Model V3 (Food-Seeking + Hazard-Aware) on CUDA ===")
    train_laya_model(
        dataset_path=v3_file,
        output_dir=MODEL_OUTPUT_DIR,
        base_model_name="answerdotai/ModernBERT-base",
        epochs=3,
        batch_size=64,
        learning_rate=3e-5
    )
    print("\n[SUCCESS] Laya V3 Model Retraining Complete!")

if __name__ == "__main__":
    run_v3_retraining()
