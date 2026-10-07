"""
Script for generating dataset of Snake game states and optimal moves from A* solver.
Saves samples to dataset.jsonl in a format compatible with Laya fine-tuning and Jev API.
"""

import json
import os
import random
import time
from snake_env import SnakeEnv
from a_star_solver import SnakeAStarSolver

def generate_snake_dataset(output_file: str, target_samples: int = 20000):
    env = SnakeEnv(width=20, height=20)
    solver = SnakeAStarSolver()
    
    samples = []
    games_played = 0
    total_steps = 0
    max_score = 0
    
    print(f"Starting dataset generation. Target samples: {target_samples}")
    start_time = time.time()
    
    while len(samples) < target_samples:
        compact_state = env.reset()
        done = False
        game_steps = 0
        
        while not done and len(samples) < target_samples:
            action = solver.get_best_move(env)
            
            # Record current compact state and target optimal action
            sample = {
                "state": compact_state,
                "question": "What is the next safe move towards food?",
                "choices": ["UP", "DOWN", "LEFT", "RIGHT"],
                "label": action
            }
            samples.append(sample)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            compact_state = next_state
            game_steps += 1
            total_steps += 1
            
        games_played += 1
        score = info.get("score", 0)
        if score > max_score:
            max_score = score
            
        if games_played % 50 == 0 or len(samples) >= target_samples:
            elapsed = time.time() - start_time
            print(f"Games: {games_played} | Collected: {len(samples)}/{target_samples} | "
                  f"Max Score: {max_score} | Speed: {total_steps / elapsed:.1f} steps/s")

    # Write to JSONL
    print(f"Writing {len(samples)} samples to {output_file}...")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Dataset successfully created at {output_file}!")

if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "dataset.jsonl")
    generate_snake_dataset(output_path, target_samples=20000)
