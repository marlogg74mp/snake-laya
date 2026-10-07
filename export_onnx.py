"""
Export Laya (ModernBERT sequence classifier) to ONNX for in-browser play (onnxruntime-web).

Writes to web3d/model/:
  laya.onnx        fp32 (reference)
  laya_int8.onnx   dynamic int8 quantization (~4x smaller, what the browser loads)
  tokenizer.json, tokenizer_config.json, special_tokens_map.json, config.json, state_schema.json

Then checks on real V9 states that ONNX (fp32 and int8) picks the same move as PyTorch.
Runs on CPU so it does not disturb training on the GPU.

Usage: python export_onnx.py [model_dir]
"""

import json
import os
import random
import shutil
import sys
import time

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "web3d", "model")
QUESTION = "What is the next safe move avoiding patrols and walls towards food?"

def prompt(state, keys):
    return (f"State: {json.dumps({k: state[k] for k in keys})}\n"
            f"Question: {QUESTION}\nChoices: UP, DOWN, LEFT, RIGHT")

def main(model_dir):
    os.makedirs(OUT, exist_ok=True)
    keys = json.load(open(os.path.join(model_dir, "state_schema.json")))["keys"]
    tok = AutoTokenizer.from_pretrained(model_dir)
    # eager attention: no flash-attn / unpadding kernels, plain ops that export cleanly
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, attn_implementation="eager").eval()

    samples = [json.loads(l) for l in open(os.path.join(ROOT, "dataset_v9.jsonl"), encoding="utf-8")]
    random.Random(5).shuffle(samples)
    samples = samples[:200]

    fp32 = os.path.join(OUT, "laya.onnx")
    enc = tok(prompt(samples[0]["state"], keys), return_tensors="pt")
    t0 = time.time()
    torch.onnx.export(
        model, (enc["input_ids"], enc["attention_mask"]), fp32,
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"exported {fp32} ({os.path.getsize(fp32) / 1e6:.0f} MB, {time.time() - t0:.0f}s)", flush=True)

    int8 = os.path.join(OUT, "laya_int8.onnx")
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    print(f"quantized {int8} ({os.path.getsize(int8) / 1e6:.0f} MB)", flush=True)

    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", "state_schema.json"):
        shutil.copy(os.path.join(model_dir, f), OUT)

    # parity check: same move as PyTorch?
    sessions = {name: ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for name, p in (("fp32", fp32), ("int8", int8))}
    agree = {k: 0 for k in sessions}
    max_diff = {k: 0.0 for k in sessions}
    times = {k: [] for k in sessions}
    for s in samples:
        e = tok(prompt(s["state"], keys), return_tensors="pt")
        with torch.no_grad():
            ref = torch.softmax(model(**e).logits, -1)[0].numpy()
        feed = {"input_ids": e["input_ids"].numpy(), "attention_mask": e["attention_mask"].numpy()}
        for k, sess in sessions.items():
            t = time.time()
            logits = sess.run(None, feed)[0][0]
            times[k].append((time.time() - t) * 1000)
            p = np.exp(logits - logits.max()); p /= p.sum()
            agree[k] += int(p.argmax() == ref.argmax())
            max_diff[k] = max(max_diff[k], float(np.abs(p - ref).max()))
    for k in sessions:
        print(f"{k}: same move as PyTorch {agree[k]}/{len(samples)}, max prob diff {max_diff[k]:.4f}, "
              f"CPU latency median {np.median(times[k]):.0f} ms", flush=True)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "laya_snake_weights_v9"))
