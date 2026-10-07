// onnx_policy.js — models in the browser for Snake Duel (onnxruntime-web: WebGPU if available, else WASM).
//   OnnxPolicy      — our v10 (ModernBERT-base classifier), files in model/          (export_onnx_web.py)
//   OnnxPolicy("model_v11w/", "Laya v11w") — the window model v11w, files in versus/model_v11w/ (same exporter)
//   Laya421Policy   — the real Laya 421M fine-tuned on Snake, files in model_laya421/ (export_laya421_web.py)
// Both use a pruned vocabulary: the original tokenizer runs here, ids are remapped with vocab_map.json.

const QUESTION = "What is the next safe move avoiding patrols and walls towards food?";
const DIRS = ["UP", "DOWN", "LEFT", "RIGHT"];
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const TRANSFORMERS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.2";

// One WebGPU device, one inference at a time: concurrent session.run calls (two models in AI-vs-AI)
// queue up on the GPU and can stall the page, so every run goes through this chain.
let gpuChain = Promise.resolve();
function exclusive(fn) {
  const run = gpuChain.then(fn, fn);
  gpuChain = run.catch(() => {});
  return run;
}

/** Python json.dumps (", " / ": " separators) — the models were trained on exactly this text. */
function pyJsonDumps(v) {
  if (v === null || v === undefined) return "null";
  if (v === true) return "true";
  if (v === false) return "false";
  if (typeof v === "number") return String(v);
  if (typeof v === "string") return JSON.stringify(v);
  if (Array.isArray(v)) return "[" + v.map(pyJsonDumps).join(", ") + "]";
  return "{" + Object.keys(v).map((k) => pyJsonDumps(k) + ": " + pyJsonDumps(v[k])).join(", ") + "}";
}

/** Shared loading: runtime, tokenizer, vocab map, schema, the .onnx file with progress, session, warm-up. */
class BrowserModel {
  constructor(base, file) {
    this.base = new URL(base, location.href).href;
    this.file = file;
    this.available = false;
    this.backend = null;
  }

  async load(onProgress, extraJson = []) {
    const ort = (this.ort = await import(`${ORT}ort.webgpu.min.mjs`));
    ort.env.wasm.wasmPaths = ORT;
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
    const tf = await import(TRANSFORMERS);
    tf.env.allowRemoteModels = false;
    tf.env.allowLocalModels = true;
    tf.env.localModelPath = new URL("..", this.base).href;
    this.tokenizer = await tf.AutoTokenizer.from_pretrained(new URL(this.base).pathname.split("/").filter(Boolean).pop());
    const [vmap, schema, ...extra] = await Promise.all(
      ["vocab_map.json", "state_schema.json", ...extraJson].map((f) => fetch(this.base + f).then((r) => r.json())));
    this.vmap = vmap.map;
    this.unk = vmap.unk;
    this.keys = schema.keys;
    const resp = await fetch(this.base + this.file);
    if (!resp.ok) throw new Error(`model file: HTTP ${resp.status}`);
    const total = +resp.headers.get("content-length") || 0;
    const reader = resp.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      onProgress(total ? got / total : 0);
    }
    const bytes = new Uint8Array(got);
    let o = 0;
    for (const c of chunks) { bytes.set(c, o); o += c.length; }
    try {
      if (!navigator.gpu) throw new Error("no WebGPU");
      this.session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu", "wasm"] });
      this.backend = "WebGPU";
    } catch {
      this.session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      this.backend = "WASM";
    }
    return extra;
  }

  project(state) {
    const s = {};
    for (const k of this.keys) s[k] = state ? state[k] : k === "food_pos" ? [-1, -1] : null;
    return s;
  }

  remap(ids) {
    const x = new BigInt64Array(ids.length);
    for (let i = 0; i < ids.length; i++) x[i] = BigInt(this.vmap[ids[i]] ?? this.unk);
    return x;
  }

  result(logits, t0, label) {
    const latency = performance.now() - t0;
    const lg = Array.from(logits).slice(0, 4), m = Math.max(...lg), e = lg.map((x) => Math.exp(x - m)), z = e.reduce((a, b) => a + b, 0);
    const probabilities = { UP: e[0] / z, DOWN: e[1] / z, LEFT: e[2] / z, RIGHT: e[3] / z };
    const action = DIRS.reduce((a, b) => (probabilities[b] > probabilities[a] ? b : a), "UP");
    return { action, probabilities, latency_ms: latency, engine: `${label} в браузере (${this.backend === "WebGPU" ? "видеокарта" : "процессор"})` };
  }
}

/** Our v10: "State: {json}\nQuestion: …\nChoices: …" -> 4-class classifier. */
export class OnnxPolicy extends BrowserModel {
  constructor(base = "model/", label = "Laya v10") { super(base, "laya_web.onnx"); this.label = label; }

  async init(onProgress = () => {}) {
    if (this.available) return;
    await this.load(onProgress);
    await this.decide(null); // warm-up (first WebGPU run compiles shaders, ~2 s)
    this.available = true;
  }

  async decide(state) {
    const text = `State: ${pyJsonDumps(this.project(state))}\nQuestion: ${QUESTION}\nChoices: UP, DOWN, LEFT, RIGHT`;
    const ids = this.tokenizer.encode(text).slice(0, 512); // adds [CLS] … [SEP] like the Python tokenizer
    const T = this.ort.Tensor;
    let t0 = 0;
    const out = await exclusive(() => {
      t0 = performance.now();
      return this.session.run({
        input_ids: new T("int64", this.remap(ids), [1, ids.length]),
        attention_mask: new T("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
      });
    });
    return this.result(out.logits.data, t0, this.label);
  }
}

/** The real Laya 421M: its own input format (laya_convai/rl_common.build_sequence), the head scores 4 [MASK] markers. */
export class Laya421Policy extends BrowserModel {
  constructor(base = "model_laya421/") { super(base, "laya421_web.onnx"); }

  async init(onProgress = () => {}) {
    if (this.available) return;
    [this.fmt] = await this.load(onProgress, ["laya_format.json"]);
    const id = (t) => this.tokenizer.model.tokens_to_ids.get(t);
    this.special = { cls: id("[CLS]"), sep: id("[SEP]"), mask: id("[MASK]") };
    await this.decide(null);
    this.available = true;
  }

  enc(text) { return this.tokenizer.encode(text, { add_special_tokens: false }); }

  /** Port of build_sequence: [CLS] choice question: ins [SEP] [MASK] opt0 … [SEP] state [SEP]. */
  build(state) {
    const { max_len, head_max_len, question: q, actions } = this.fmt, sp = this.special;
    let head = this.enc(`${q.t} question: ${q.ins}`);
    const opts = actions.map((a) => [sp.mask, ...this.enc(" " + (q.crit[a] ? `${a}: ${q.crit[a]}` : a)).slice(0, 48)]);
    const budget = head_max_len - opts.reduce((n, o) => n + o.length, 0);
    head = head.slice(0, Math.max(8, budget));
    const ids = [sp.cls, ...head, sp.sep], markers = [];
    for (const o of opts) { markers.push(ids.length); ids.push(...o); }
    ids.push(sp.sep);
    const room = Math.max(0, max_len - ids.length - 1);
    ids.push(...this.enc(pyJsonDumps(this.project(state))).slice(0, room), sp.sep);
    return { ids: ids.slice(0, max_len), markers };
  }

  async decide(state) {
    const { ids, markers } = this.build(state);
    const T = this.ort.Tensor;
    let t0 = 0;
    const out = await exclusive(() => {
      t0 = performance.now();
      return this.session.run({
        input_ids: new T("int64", this.remap(ids), [1, ids.length]),
        attention_mask: new T("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
        marker_pos: new T("int64", BigInt64Array.from(markers.map(BigInt)), [1, markers.length]),
      });
    });
    return this.result(out.logits.data, t0, "Laya 421M");
  }
}
