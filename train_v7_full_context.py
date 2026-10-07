"""
Train ModernBERT on Dataset V7 with Full Context Window (max_length = 256).
Zero truncation: 100% of state, sensors, obstacles, portals, patrols, questions and choices preserved.

Memory-efficient version (fits a 16 GB GPU without spilling into shared system RAM):
- bf16 autocast for forward/backward
- dynamic padding: each batch is padded only to its longest sample (~225 tokens), not to 256
- micro-batch 32 x 2 gradient accumulation steps = effective batch 64
- warm start from the previous Laya weights instead of the raw base model
- weights saved after every epoch
"""

import os
import sys
import json
import argparse
import time
import requests
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(ROOT, "dataset_v7_patrols.jsonl")
OUTPUT_DIR = os.path.join(ROOT, "laya_snake_weights")
# Previous Laya (V6) weights; falls back to the raw base model if missing.
INIT_MODEL = os.path.join(ROOT, "laya_snake_weights_v6_backup")
BASE_MODEL = "answerdotai/ModernBERT-base"
MAX_LENGTH = 256

ACTION_TO_ID = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
ID_TO_ACTION = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

class FullContextDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256):
        texts = []
        labels = []
        self.state_keys = None

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if self.state_keys is None:
                    self.state_keys = list(item["state"].keys())
                prompt_text = (
                    f"State: {json.dumps(item['state'])}\n"
                    f"Question: {item['question']}\n"
                    f"Choices: {', '.join(item['choices'])}"
                )
                texts.append(prompt_text)
                labels.append(ACTION_TO_ID[item["label"]])

        print(f"[DATASET] Tokenizing {len(texts)} samples with max_length={max_length} (no padding)...", flush=True)
        encodings = tokenizer(texts, truncation=True, max_length=max_length)
        self.input_ids = encodings["input_ids"]
        self.labels = labels
        lengths = [len(x) for x in self.input_ids]
        truncated = sum(1 for n in lengths if n >= max_length)
        print(f"[DATASET] Done! Samples: {len(lengths)}, max tokens: {max(lengths)}, truncated: {truncated}", flush=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}

def save_model(model, tokenizer, out_dir, state_keys):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # The server (and the future ONNX demo) must feed the state in exactly this key order
    with open(os.path.join(out_dir, "state_schema.json"), "w", encoding="utf-8") as f:
        json.dump({"keys": state_keys}, f)

def run_training(epochs=2, batch_size=32, grad_accum=2, learning_rate=3e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"=== ModernBERT V7 Full-Context Training (256 Tokens, bf16={use_bf16}) ===", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    init_path = INIT_MODEL if os.path.isdir(INIT_MODEL) else BASE_MODEL
    print(f"Loading tokenizer and model from '{init_path}'...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(init_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(init_path, num_labels=4, local_files_only=True)
    model.to(device)

    dataset = FullContextDataset(DATASET_PATH, tokenizer, max_length=MAX_LENGTH)
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42))

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size * 2, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    opt_steps_per_epoch = (len(train_loader) + grad_accum - 1) // grad_accum

    print(f"Starting Training: {len(train_dataset)} train samples, {len(eval_dataset)} eval samples", flush=True)
    print(f"Epochs: {epochs}, Batch: {batch_size} x {grad_accum} accum = {batch_size * grad_accum}, "
          f"Optimizer: AdamW (lr={learning_rate}), {opt_steps_per_epoch} optimizer steps/epoch", flush=True)

    t_start = time.time()
    accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        t_epoch_start = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch).loss
            (loss / grad_accum).backward()
            total_train_loss += loss.item()

            is_last = (step + 1) == len(train_loader)
            if (step + 1) % grad_accum == 0 or is_last:
                optimizer.step()
                optimizer.zero_grad()
                opt_step = (step + 1 + grad_accum - 1) // grad_accum
                if opt_step % 50 == 0 or is_last:
                    elapsed = time.time() - t_epoch_start
                    peak_gb = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                    print(f"Epoch {epoch+1}/{epochs} | Step {opt_step}/{opt_steps_per_epoch} | "
                          f"Train Loss: {total_train_loss/(step+1):.4f} | Elapsed: {elapsed:.1f}s | "
                          f"Peak VRAM: {peak_gb:.1f} GB", flush=True)

        model.eval()
        total_eval_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    outputs = model(**batch)
                total_eval_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

        eval_loss = total_eval_loss / len(eval_loader)
        accuracy = (correct / total) * 100.0
        print(f"==> Epoch {epoch+1} Completed | Train Loss: {total_train_loss/len(train_loader):.4f} | "
              f"Eval Loss: {eval_loss:.4f} | Accuracy: {accuracy:.2f}%", flush=True)

        print(f"Saving epoch {epoch+1} weights to {OUTPUT_DIR}...", flush=True)
        save_model(model, tokenizer, OUTPUT_DIR, dataset.state_keys)

    total_time = time.time() - t_start
    print(f"\n[DONE] Training complete in {total_time:.1f}s! Final Accuracy: {accuracy:.2f}%", flush=True)

    try:
        resp = requests.post("http://localhost:9500/reload_weights", timeout=10)
        print(f"[HOT-RELOAD] Server response: {resp.json()}", flush=True)
    except Exception as e:
        print(f"[HOT-RELOAD WARNING] Could not auto-trigger hot reload: {e}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--init", default=INIT_MODEL, help="weights to warm-start from")
    parser.add_argument("--out", default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    args = parser.parse_args()
    DATASET_PATH, INIT_MODEL, OUTPUT_DIR, MAX_LENGTH = args.dataset, args.init, args.out, args.max_length
    run_training(epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum)
