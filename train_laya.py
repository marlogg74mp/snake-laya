"""
Fine-tuning script for Laya (ModernBERT / BERT-based decision model) on Snake game dataset using pure PyTorch.
Does not require HuggingFace 'accelerate' library.
"""

import json
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ACTION_TO_ID = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
ID_TO_ACTION = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

class SnakeDecisionDataset(Dataset):
    """
    Dataset class for converting JSONL game states into tokenized inputs for Laya.
    Uses fast batch tokenization for high performance.
    """
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256):
        texts = []
        labels = []
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    prompt_text = (
                        f"State: {json.dumps(item['state'])}\n"
                        f"Question: {item['question']}\n"
                        f"Choices: {', '.join(item['choices'])}"
                    )
                    label_id = ACTION_TO_ID[item["label"]]
                    texts.append(prompt_text)
                    labels.append(label_id)

        print(f"Batch tokenizing {len(texts)} samples...")
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

def train_laya_model(
    dataset_path: str,
    output_dir: str = "./laya_snake_weights",
    base_model_name: str = "answerdotai/ModernBERT-base",
    epochs: int = 2,
    batch_size: int = 64,
    learning_rate: float = 3e-5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_path = output_dir if os.path.exists(output_dir) else base_model_name
    print(f"Loading tokenizer and base model from '{model_path}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=os.path.exists(output_dir))
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=4,
            local_files_only=os.path.exists(output_dir)
        )
    except Exception as e:
        print(f"[WARN] Local load failed ({e}), falling back to base model: {base_model_name}...")
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['PYTHONHTTPSVERIFY'] = '0'
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=4)

    model.to(device)

    print("Loading and tokenizing dataset...")
    full_dataset = SnakeDecisionDataset(dataset_path, tokenizer)
    
    train_size = int(0.9 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Starting PyTorch training loop for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            
            if steps % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {steps}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / steps

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        eval_loss = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                eval_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"--> Epoch {epoch+1} Completed | Train Loss: {avg_train_loss:.4f} | "
              f"Eval Loss: {eval_loss/len(eval_loader):.4f} | Accuracy: {accuracy*100:.2f}%")

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s.")

    print(f"Saving fine-tuned model and tokenizer to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("All weights saved successfully!")

if __name__ == "__main__":
    dataset_file = os.path.join(os.path.dirname(__file__), "dataset.jsonl")
    output_weights_dir = os.path.join(os.path.dirname(__file__), "laya_snake_weights")
    if os.path.exists(dataset_file):
        train_laya_model(dataset_file, output_dir=output_weights_dir)
    else:
        print(f"[ERROR] Dataset file not found at {dataset_file}.")
