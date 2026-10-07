"""
Browser build of the real Laya (421M) fine-tuned on Snake (laya_convai_snake/): pruned vocabulary + int4.

Laya's input is built in the browser exactly like laya_convai/rl_common.build_sequence:
  [CLS] choice question: <ins> [SEP] [MASK] UP: … [MASK] DOWN: … [MASK] LEFT: … [MASK] RIGHT: … [SEP] <state> [SEP]
The ONNX graph takes input_ids (pruned vocab), attention_mask and marker_pos (positions of the 4 [MASK]
markers) and returns the 4 option logits.

Output: web3d/model_laya421/ (laya421_web.onnx, vocab_map.json, laya_format.json, tokenizer files, state_schema.json)
Usage (CPU, does not disturb the GPU): python export_laya421_web.py
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
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
LAYA = os.path.join(ROOT, "laya_convai")
sys.path.insert(0, LAYA)
from rl_common import DecisionModel, build_sequence  # noqa: E402
import laya_convai_snake as LS  # noqa: E402  (QUESTION, ACTIONS)

OUT = os.path.join(ROOT, "web3d", "model_laya421")
BUILD = os.path.join(ROOT, "onnx_build")

class Options4(torch.nn.Module):
    """DecisionModel for one 4-option choice question: (ids, mask, marker_pos) -> 4 logits."""

    def __init__(self, dm):
        super().__init__()
        self.dm = dm

    @staticmethod
    def head_layer(layer, h):
        """nn.TransformerEncoderLayer (norm_first, relu, eval) written out with shape-agnostic reshapes:
        traced nn.MultiheadAttention bakes the sequence length into the graph."""
        sa = layer.self_attn
        nh = sa.num_heads
        d = h.shape[-1]
        y = layer.norm1(h)
        q, k, v = torch.nn.functional.linear(y, sa.in_proj_weight, sa.in_proj_bias).chunk(3, dim=-1)
        split = lambda t: t.reshape(t.shape[0], -1, nh, d // nh).transpose(1, 2)
        o = torch.nn.functional.scaled_dot_product_attention(split(q), split(k), split(v))
        o = sa.out_proj(o.transpose(1, 2).reshape(h.shape[0], -1, d))
        h = h + o
        return h + layer.linear2(layer.activation(layer.linear1(layer.norm2(h))))

    def forward(self, input_ids, attention_mask, marker_pos):
        # one unpadded sequence per call in the browser, so the head needs no padding mask
        dm = self.dm
        h = dm.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + dm.type_emb.weight[0]  # question type "choice"
        for layer in dm.head.layers:
            h = self.head_layer(layer, h)
        idx = marker_pos[:, :, None].expand(-1, -1, h.shape[-1])
        return dm.scorer(torch.gather(h, 1, idx)).squeeze(-1)

def main():
    torch.backends.mha.set_fastpath_enabled(False)  # plain ops for TransformerEncoderLayer, exportable
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)
    cfg = json.load(open(os.path.join(LAYA, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(LAYA, "tokenizer"))
    keys = json.load(open(os.path.join(ROOT, "laya_snake_weights_v10", "state_schema.json")))["keys"]

    def build(state):
        s = {k: state[k] for k in keys}
        return build_sequence(tok, s, LS.QUESTION, cfg["max_len"], cfg["head_max_len"])

    def model(attn):
        enc = AutoModel.from_config(AutoConfig.from_pretrained(os.path.join(LAYA, "encoder")), attn_implementation=attn)
        dm = DecisionModel(enc, cfg["head_layers"], len(cfg["act_costs"]) + 1)
        dm.load_state_dict(load_file(os.path.join(ROOT, "laya_convai_snake", "model.safetensors")), strict=True)
        return dm.eval()

    full = model("eager")
    pruned = model("eager")

    # vocabulary that can occur: the fixed question/options + every state in the datasets + all small numbers
    ids_used = set(tok.all_special_ids)
    rows = []
    for path in ("dataset_v9.jsonl", "dataset_v10.jsonl"):
        for i, line in enumerate(open(os.path.join(ROOT, path), encoding="utf-8")):
            if i % 5 == 0:
                st = json.loads(line)["state"]
                ids, _ = build(st)
                ids_used.update(ids)
                if len(rows) < 300 and i % 175 == 0:
                    rows.append(st)
    words = keys + ["UP", "DOWN", "LEFT", "RIGHT", "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT", "UNKNOWN", "SAME",
                    "NONE", "true", "false", "null"]
    extra = " ".join(f"[{n}, {-n}], {n}" for n in range(0, 101)) + " " + json.dumps({w: [w, 0] for w in words})
    ids_used.update(tok(extra, add_special_tokens=False)["input_ids"])
    keep = sorted(ids_used)
    remap = {old: new for new, old in enumerate(keep)}
    unk = remap[tok.unk_token_id]
    emb = pruned.encoder.embeddings.tok_embeddings
    new_emb = torch.nn.Embedding(len(keep), emb.embedding_dim)
    new_emb.weight.data = emb.weight.data[torch.tensor(keep)].clone()
    pruned.encoder.embeddings.tok_embeddings = new_emb
    print(f"vocab {tok.vocab_size} -> {len(keep)}", flush=True)

    def enc(state):
        ids, markers = build(state)
        x = torch.tensor([[remap.get(i, unk) for i in ids]])
        return torch.tensor([ids]), x, torch.ones_like(x), torch.tensor([markers])

    _, x, att, mp = enc(rows[0])
    fp32 = os.path.join(BUILD, "laya421_pruned_fp32.onnx")
    t0 = time.time()
    torch.onnx.export(Options4(pruned), (x, att, mp), fp32, input_names=["input_ids", "attention_mask", "marker_pos"],
                      output_names=["logits"], dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                                                             "marker_pos": {0: "b"}, "logits": {0: "b"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    # >2 GB protobuf limit is not hit (fp32 ~1.5 GB); quantize matmuls to 4 bit
    q = MatMul4BitsQuantizer(onnx.load(fp32), block_size=32, is_symmetric=True)
    q.process()
    web = os.path.join(OUT, "laya421_web.onnx")
    q.model.save_model_to_file(web, use_external_data_format=False)
    print(f"exported in {time.time() - t0:.0f}s: fp32 {os.path.getsize(fp32) / 1e6:.0f} MB -> int4 {os.path.getsize(web) / 1e6:.0f} MB", flush=True)

    json.dump({"map": {str(k): v for k, v in remap.items()}, "unk": unk}, open(os.path.join(OUT, "vocab_map.json"), "w"))
    json.dump({"max_len": cfg["max_len"], "head_max_len": cfg["head_max_len"], "question": LS.QUESTION,
               "actions": LS.ACTIONS}, open(os.path.join(OUT, "laya_format.json"), "w"), ensure_ascii=False, indent=1)
    shutil.copy(os.path.join(ROOT, "laya_snake_weights_v10", "state_schema.json"), OUT)
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(LAYA, "tokenizer", f), OUT)

    # parity on real states: same move as the PyTorch fine-tuned model?
    sess = ort.InferenceSession(web, providers=["CPUExecutionProvider"])
    agree, ts = 0, []
    with torch.no_grad():
        for st in rows[:200]:
            full_ids, x, att, mp = enc(st)
            mm, qt = torch.ones_like(mp, dtype=torch.bool), torch.zeros(1, dtype=torch.long)
            ref = full(full_ids, att, mp, mm, qt)[0][0].numpy()  # the original DecisionModel forward
            t = time.time()
            lg = sess.run(None, {"input_ids": x.numpy(), "attention_mask": att.numpy(), "marker_pos": mp.numpy()})[0][0]
            ts.append((time.time() - t) * 1000)
            agree += int(lg.argmax() == ref.argmax())
    print(f"int4 web model: same move as PyTorch {agree}/{min(200, len(rows))}, CPU median {np.median(ts):.0f} ms", flush=True)

if __name__ == "__main__":
    main()
