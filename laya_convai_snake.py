"""
The real Laya (convaiinnovations/laya: ModernBERT-large + decision head, 421M) on our Snake task.

Laya's native format is used, with its own code (laya_convai/rl_common.py):
  [CLS] choice question: <instructions> [SEP] [MASK] UP: … [MASK] DOWN: … [MASK] LEFT: … [MASK] RIGHT: … [SEP] <state JSON> [SEP]
and the head scores the four [MASK] markers.

  python laya_convai_snake.py zeroshot            # Laya as released, asked like Jev (no training)
  python laya_convai_snake.py train [dataset]     # fine-tune encoder + head, 1 epoch -> laya_convai_snake/
  python laya_convai_snake.py eval <model dir>    # same benchmark as eval_v9.py (imitation + 40 closed-loop episodes)
"""

import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "laya_convai"))
from rl_common import build_model, build_sequence, collate_items, QTYPES  # noqa: E402

import eval_v9  # noqa: E402

LAYA_DIR = os.path.join(ROOT, "laya_convai")
OUT_DIR = os.path.join(ROOT, "laya_convai_snake")
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
# the same question the Jev version of the game asked (web_server.py, query_jev_113)
QUESTION = {
    "t": "choice",
    "ins": "Select the safest move towards food while strictly avoiding walls, body, drones and poison",
    "crit": {"UP": "Move upwards (y - 1)", "DOWN": "Move downwards (y + 1)",
             "LEFT": "Move left (x - 1)", "RIGHT": "Move right (x + 1)"},
}

class LayaPolicy:
    """Wraps Laya so eval_v9.py can use it: probs_batch(states) -> [{UP: p, …}]."""

    def __init__(self, weights=os.path.join(LAYA_DIR, "model.safetensors"), train=False):
        self.cfg = json.load(open(os.path.join(LAYA_DIR, "rl_agent_config.json")))
        self.tok = AutoTokenizer.from_pretrained(os.path.join(LAYA_DIR, "tokenizer"))
        self.model = build_model(self.cfg, encoder_dir=os.path.join(LAYA_DIR, "encoder"))
        self.model.load_state_dict(load_file(weights), strict=True)
        self.model.encoder.config.reference_compile = False
        self.model.cuda().train(train)
        self.keys = json.load(open(os.path.join(ROOT, "laya_snake_weights_v10", "state_schema.json")))["keys"]
        self.truncated = 0

    def item(self, state, label=-1):
        s = {k: state[k] for k in self.keys}
        ids, markers = build_sequence(self.tok, s, QUESTION, self.cfg["max_len"], self.cfg["head_max_len"])
        if len(ids) >= self.cfg["max_len"]:
            self.truncated += 1
        return {"ids": ids, "markers": markers, "qtype": QTYPES["choice"], "target": [0.0] * 4, "label": label,
                "episode": 0, "ep_step": 0, "ep_len": 1, "src": "snake"}

    def forward(self, items):
        b = collate_items([items], self.tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = self.model(b["input_ids"].cuda(), b["attention_mask"].cuda(), b["marker_pos"].cuda(),
                                   b["marker_mask"].cuda(), b["qtype"].cuda())
        return logits[:, :4].float()

    @torch.no_grad()
    def probs_batch(self, states):
        p = torch.softmax(self.forward([self.item(s) for s in states]), -1).tolist()
        return [dict(zip(ACTIONS, row)) for row in p]

def train(dataset, epochs=1, micro=16, accum=4, lr=2e-5):
    pol = LayaPolicy(train=True)
    pol.model.encoder.gradient_checkpointing_enable()
    rows = [json.loads(l) for l in open(os.path.join(ROOT, dataset), encoding="utf-8")]
    random.Random(1).shuffle(rows)
    n_eval = len(rows) // 10
    ev, tr = rows[:n_eval], rows[n_eval:]
    opt = torch.optim.AdamW(pol.model.parameters(), lr=lr, weight_decay=0.01)
    total = math.ceil(len(tr) / (micro * accum)) * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * max(0.0, 1 - s / total))
    print(f"[LAYA] train {len(tr)} / eval {len(ev)} samples, {total} optimizer steps, micro {micro} x accum {accum}", flush=True)
    t0, step, run = time.time(), 0, 0.0
    for ep in range(epochs):
        for i in range(0, len(tr), micro):
            batch = tr[i:i + micro]
            logits = pol.forward([pol.item(r["state"]) for r in batch])
            y = torch.tensor([ACTIONS.index(r["label"]) for r in batch], device="cuda")
            loss = F.cross_entropy(logits, y)
            (loss / accum).backward()
            run += loss.item()
            if (i // micro + 1) % accum == 0 or i + micro >= len(tr):
                torch.nn.utils.clip_grad_norm_(pol.model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 50 == 0 or step == total:
                    print(f"[LAYA] step {step}/{total} | loss {run / (accum * 50 if step % 50 == 0 else accum):.4f} | "
                          f"{time.time() - t0:.0f}s | peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
                    run = 0.0
    pol.model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(ev), 32):
            batch = ev[i:i + 32]
            pred = pol.forward([pol.item(r["state"]) for r in batch]).argmax(-1).tolist()
            correct += sum(ACTIONS[p] == r["label"] for p, r in zip(pred, batch))
    print(f"[LAYA] eval accuracy {correct / len(ev):.2%} | truncated inputs {pol.truncated} | {time.time() - t0:.0f}s", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    save_file({k: v.contiguous() for k, v in pol.model.state_dict().items()}, os.path.join(OUT_DIR, "model.safetensors"))
    print(f"[LAYA] saved {OUT_DIR}", flush=True)

def evaluate(weights, name, episodes=40):
    pol = LayaPolicy(weights)
    eval_v9.imitation(pol)
    eval_v9.closed_loop(pol, name, episodes)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "zeroshot":
        evaluate(os.path.join(LAYA_DIR, "model.safetensors"), "laya-zeroshot")
    elif cmd == "train":
        train(sys.argv[2] if len(sys.argv) > 2 else "dataset_v10.jsonl")
    elif cmd == "eval":
        evaluate(os.path.join(sys.argv[2], "model.safetensors"), os.path.basename(sys.argv[2].rstrip("/\\")))
