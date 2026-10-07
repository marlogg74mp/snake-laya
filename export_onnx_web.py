"""
Browser build of Laya: pruned vocabulary + int4 weights.

Our prompts are JSON game states: they use only ~150 of ModernBERT's 50k tokens, and the token
embedding table is 26% of the model (155 MB in fp32). We keep only the rows that can occur,
export to ONNX and quantize the matmuls to 4 bit (MatMulNBits, supported by onnxruntime-web).

Output (web3d/model/):
  laya_web.onnx      pruned + int4 model (inputs: input_ids remapped to the pruned vocab, attention_mask)
  vocab_map.json     {"map": {original_id: new_id}, "unk": new_unk_id}
  tokenizer.json, tokenizer_config.json, special_tokens_map.json, state_schema.json
Intermediate fp32 files go to onnx_build/ (not served).

Usage: python export_onnx_web.py [model_dir] [out_dir] [dataset.jsonl]
  e.g. python export_onnx_web.py laya_snake_weights_v11w versus/model_v11w dataset_v11w.jsonl
"""

import json
import os
import random
import shutil
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "web3d", "model")
BUILD = os.path.join(ROOT, "onnx_build")
QUESTION = "What is the next safe move avoiding patrols and walls towards food?"

def prompt(state, keys):
    return (f"State: {json.dumps({k: state[k] for k in keys})}\n"
            f"Question: {QUESTION}\nChoices: UP, DOWN, LEFT, RIGHT")

def used_token_ids(tok, keys, datasets):
    ids = set(tok.all_special_ids)
    for path in datasets:
        for i, line in enumerate(open(path, encoding="utf-8")):
            if i % 5 == 0:
                ids.update(tok(prompt(json.loads(line)["state"], keys))["input_ids"])
    # safety margin: every number, word and value spelling that a state can contain
    words = keys + ["UP", "DOWN", "LEFT", "RIGHT", "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT", "UNKNOWN",
                    "SAME", "NONE", "true", "false", "null"]
    extra = " ".join(f"[{n}, {-n}], {n}" for n in range(0, 301)) + " " + json.dumps({w: [w, 0] for w in words})
    ids.update(tok(extra)["input_ids"])
    return sorted(ids)

def main(model_dir, out=OUT, data=None):
    OUT = out
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)
    keys = json.load(open(os.path.join(model_dir, "state_schema.json")))["keys"]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, attn_implementation="eager").eval()
    full = AutoModelForSequenceClassification.from_pretrained(model_dir, attn_implementation="eager").eval()

    datasets = [data] if data else [p for p in ("dataset_v9.jsonl", "dataset_v10.jsonl") if os.path.exists(os.path.join(ROOT, p))]
    keep = used_token_ids(tok, keys, [os.path.join(ROOT, p) for p in datasets])
    remap = {old: new for new, old in enumerate(keep)}
    unk = remap[tok.unk_token_id]
    print(f"vocab: {tok.vocab_size} -> {len(keep)} tokens", flush=True)

    emb = model.model.embeddings.tok_embeddings
    pruned = torch.nn.Embedding(len(keep), emb.embedding_dim)
    pruned.weight.data = emb.weight.data[torch.tensor(keep)].clone()
    model.model.embeddings.tok_embeddings = pruned
    model.config.vocab_size = len(keep)

    def encode(state):
        e = tok(prompt(state, keys), return_tensors="pt")
        ids = torch.tensor([[remap.get(int(i), unk) for i in e["input_ids"][0]]])
        return e, ids

    samples = [json.loads(l) for l in open(os.path.join(ROOT, datasets[0]), encoding="utf-8")]
    random.Random(33).shuffle(samples)
    samples = samples[:300]

    e, ids = encode(samples[0]["state"])
    fp32 = os.path.join(BUILD, "laya_pruned_fp32.onnx")
    torch.onnx.export(model, (ids, e["attention_mask"]), fp32,
                      input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    q = MatMul4BitsQuantizer(onnx.load(fp32), block_size=32, is_symmetric=True)
    q.process()
    web = os.path.join(OUT, "laya_web.onnx")
    q.model.save_model_to_file(web, use_external_data_format=False)
    print(f"fp32 pruned: {os.path.getsize(fp32) / 1e6:.0f} MB -> int4 web model: {os.path.getsize(web) / 1e6:.0f} MB", flush=True)

    json.dump({"map": {str(k): v for k, v in remap.items()}, "unk": unk}, open(os.path.join(OUT, "vocab_map.json"), "w"))
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "state_schema.json"):
        shutil.copy(os.path.join(model_dir, f), OUT)
    for old in ("laya.onnx", "laya_int8.onnx", "laya_int8_per_channel.onnx", "laya_int8_per_channel_matmul_only.onnx",
                "laya_int4_b32.onnx", "laya_int4_b128.onnx", "config.json"):
        p = os.path.join(OUT, old)
        if os.path.exists(p):
            shutil.move(p, os.path.join(BUILD, old))  # experiments stay out of the served folder

    # parity: pruned fp32 must equal the original exactly; int4 is compared by chosen move
    sessions = {"pruned fp32": ort.InferenceSession(fp32, providers=["CPUExecutionProvider"]),
                "pruned int4 (web)": ort.InferenceSession(web, providers=["CPUExecutionProvider"])}
    stats = {k: [0, 0.0, []] for k in sessions}
    for s in samples:
        e, ids = encode(s["state"])
        with torch.no_grad():
            ref = torch.softmax(full(**e).logits, -1)[0].numpy()
        for k, sess in sessions.items():
            t = time.time()
            lg = sess.run(None, {"input_ids": ids.numpy(), "attention_mask": e["attention_mask"].numpy()})[0][0]
            stats[k][2].append((time.time() - t) * 1000)
            p = np.exp(lg - lg.max()); p /= p.sum()
            stats[k][0] += int(p.argmax() == ref.argmax())
            stats[k][1] = max(stats[k][1], float(np.abs(p - ref).max()))
    for k, (agree, diff, ts) in stats.items():
        print(f"{k:18}: same move as original {agree}/{len(samples)}, max prob diff {diff:.4f}, CPU {np.median(ts):.0f} ms", flush=True)

if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else os.path.join(ROOT, "laya_snake_weights_v9"), os.path.join(ROOT, a[1]) if len(a) > 1 else OUT, a[2] if len(a) > 2 else None)
