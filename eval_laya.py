"""
Evaluation script to test fine-tuned Laya model on SnakeEnv game simulation.
Measures decision accuracy, game score, and inference latency in milliseconds.
"""

import json
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from snake_env import SnakeEnv

ID_TO_ACTION = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

class LayaSnakeAgent:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading fine-tuned Laya model from {model_dir} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def get_action(self, compact_state: dict) -> tuple[str, float]:
        """
        Returns predicted action string and inference time in milliseconds.
        """
        prompt_text = (
            f"State: {json.dumps(compact_state)}\n"
            "Question: What is the next safe move towards food?\n"
            "Choices: UP, DOWN, LEFT, RIGHT"
        )
        
        t0 = time.time()
        encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=-1).item()

        latency_ms = (time.time() - t0) * 1000.0
        return ID_TO_ACTION[pred_id], latency_ms

def run_laya_evaluation(model_dir: str, num_games: int = 5, max_steps_per_game: int = 200):
    agent = LayaSnakeAgent(model_dir)
    env = SnakeEnv(width=20, height=20)

    total_score = 0
    total_latency = 0.0
    total_steps = 0

    print(f"\n=== Evaluating Laya Model on {num_games} Snake Games ===")

    for game_idx in range(num_games):
        state = env.reset()
        game_steps = 0
        
        while game_steps < max_steps_per_game:
            action, latency = agent.get_action(state)
            total_latency += latency
            total_steps += 1
            game_steps += 1

            state, reward, done, info = env.step(action)
            score = info.get("score", 0)

            if done:
                print(f"Game {game_idx+1}: Ended at step {game_steps} | Score: {score} | Reason: {info.get('reason')}")
                break

        if not done:
            print(f"Game {game_idx+1}: Reached max steps ({max_steps_per_game}) | Score: {score}")

        total_score += score

    avg_score = total_score / num_games
    avg_latency = total_latency / total_steps if total_steps > 0 else 0.0
    print(f"\nEvaluation Summary:")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Average Inference Latency: {avg_latency:.2f} ms/step")

if __name__ == "__main__":
    weights_path = os.path.join(os.path.dirname(__file__), "laya_snake_weights")
    if os.path.exists(weights_path):
        run_laya_evaluation(weights_path)
    else:
        print(f"[ERROR] Fine-tuned model directory not found at {weights_path}.")
