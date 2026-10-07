// policy.js — абстракция выбора хода.
//
// Любой провайдер реализует интерфейс:
//   id: string, label: string
//   async init(): Promise<void>                 — подготовка (загрузка модели и т.п.)
//   async decide(state, game): Promise<Decision> — state = game.getModelState(schema)
//   dispose(): void
//
// Decision = {
//   action: "UP"|"DOWN"|"LEFT"|"RIGHT"|null,
//   probabilities: {UP, DOWN, LEFT, RIGHT},   // 0..1
//   confidence: number,                        // 0..1
//   latencyMs: number,                         // время инференса, которое сообщил движок
//   roundTripMs: number,                       // полное время вызова decide()
//   engine: string,                            // человекочитаемое имя движка
// }

import { DIRS, ACTIONS, OPPOSITES, GRID, droneStep, cloneDrone, STATE_KEYS_V9, MAX_BODY_CELLS } from "./game.js";

export const QUESTION = "What is the next safe move avoiding patrols and walls towards food?";

/** json.dumps(obj) из Python (separators=(', ', ': '), ensure_ascii=True) — нужно для ONNX/токенизатора. */
export function pyJsonDumps(v) {
  if (v === null || v === undefined) return "null";
  if (v === true) return "true";
  if (v === false) return "false";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : (Number.isFinite(v) ? String(v) : "NaN");
  if (typeof v === "string") return JSON.stringify(v).replace(/[\u007f-￿]/g, (c) => "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0"));
  if (Array.isArray(v)) return "[" + v.map(pyJsonDumps).join(", ") + "]";
  return "{" + Object.keys(v).map((key) => pyJsonDumps(key) + ": " + pyJsonDumps(v[key])).join(", ") + "}";
}

/** Порядок ключей V9 (RULES_V9 §5 = STATE_KEYS_V9 в build_dataset_v9.py / state_schema.json модели V9). */
export const TRAINING_KEYS = STATE_KEYS_V9;

/**
 * JS-копия to_training_schema(state, keys) из web_server.py. Laya читает состояние как ТЕКСТ, поэтому порядок
 * ключей, лишние ключи и написание значений должны совпадать с обучающими данными, иначе модель «плывёт».
 * keys — список из state_schema.json модели (по умолчанию V9).
 */
export function toTrainingSchema(state, keys = TRAINING_KEYS) {
  const s = {};
  for (const k of keys) s[k] = state[k] ?? null;
  if (s.food_dir === null || s.food_dir === "NONE") s.food_dir = "UNKNOWN";
  else if (s.food_dir === "ON_FOOD") s.food_dir = "SAME"; // snake_env._get_food_dir() пишет "SAME"
  if (s.food_pos === null) s.food_pos = [-1, -1];
  for (const k of ["obstacles", "portals", "patrols"]) s[k] = s[k] || [];
  for (const k of ["body", "phase_walls", "drone_path"]) if (keys.includes(k)) s[k] = s[k] || [];
  if (keys.includes("immortal_steps")) s.immortal_steps = Math.trunc(s.immortal_steps || 0);
  if (keys.includes("body")) s.body = s.body.map((c) => c.slice(0, 2)).slice(0, MAX_BODY_CELLS);
  s.recent_actions = s.recent_actions || [];
  if (s.sight_radius === null) s.sight_radius = 5;
  s.fog_of_war = Boolean(s.fog_of_war);
  if (keys.includes("snake_len")) {
    s.snake_len = Math.trunc(s.snake_len || 0);
    s.patrols = s.patrols.map((p) => (p.length === 2 ? [...p, 0, 0] : p.slice(0, 4)));
  } else {
    s.patrols = s.patrols.map((p) => p.slice(0, 2));
  }
  return s;
}

/** Точный текст промпта, который видит ModernBERT в web_server.py / train_laya.py (через toTrainingSchema). */
export function buildPrompt(state) {
  return `State: ${pyJsonDumps(toTrainingSchema(state))}\nQuestion: ${QUESTION}\nChoices: UP, DOWN, LEFT, RIGHT`;
}

const uniform = () => ({ UP: 0.25, DOWN: 0.25, LEFT: 0.25, RIGHT: 0.25 });
const oneHot = (a) => ({ UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0, ...(a ? { [a]: 1 } : {}) });

// ======================================================================
// 1. ServerPolicy — POST /predict к FastAPI (web_server.py), формат как раньше
// ======================================================================
export class ServerPolicy {
  constructor({ baseUrl = "", model = "laya", timeoutMs = 150 } = {}) {
    this.id = "server";
    this.label = "Нейросеть на сервере";
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.model = model;
    this.timeoutMs = timeoutMs;
  }

  /** GET /health → {ok, laya_loaded, ...}. Не трогает модель. */
  static async probe(baseUrl = "", timeoutMs = 1200) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const r = await fetch(`${baseUrl.replace(/\/$/, "")}/health`, { signal: ctrl.signal, cache: "no-store" });
      if (!r.ok) return { ok: false };
      const j = await r.json();
      return { ok: j && j.status === "healthy", ...j };
    } catch {
      return { ok: false };
    } finally {
      clearTimeout(t);
    }
  }

  async init() {}

  async decide(state) {
    const t0 = performance.now();
    const ctrl = new AbortController();
    const tid = setTimeout(() => ctrl.abort(), this.timeoutMs);
    try {
      const resp = await fetch(`${this.baseUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: this.model, state }), // the server normalizes to the loaded model's schema
        signal: ctrl.signal,
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      return {
        action: data.action,
        probabilities: data.probabilities || uniform(),
        confidence: data.confidence ?? 0,
        latencyMs: data.latency_ms ?? 0,
        roundTripMs: performance.now() - t0,
        engine: data.engine || "Server",
      };
    } finally {
      clearTimeout(tid);
    }
  }

  async recordCrash({ state, wrongAction, reason }) {
    try {
      await fetch(`${this.baseUrl}/record_crash`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state, wrong_action: wrongAction, crash_reason: reason, model_used: this.model }),
      });
    } catch (e) { console.warn("record_crash failed", e); }
  }

  dispose() {}
}

// ======================================================================
// 2. OnnxPolicy — ЗАГЛУШКА для будущего инференса прямо в браузере
// ======================================================================
export class OnnxPolicy {
  /**
   * Laya in the browser: pruned-vocab int4 ModernBERT (export_onnx_web.py) on onnxruntime-web.
   * WebGPU when available, otherwise WASM. The tokenizer is the original tokenizer.json (transformers.js);
   * its ids are remapped to the pruned vocabulary with vocab_map.json.
   */
  constructor({ base = "./model/" } = {}) {
    this.id = "onnx";
    this.label = "Нейросеть в браузере";
    this.base = new URL(base, location.href).href;
    this.available = false;
    this.backend = null;
  }

  async init(onProgress = () => {}) {
    if (this.available) return;
    const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    const ort = (this.ort = await import(`${ORT}ort.webgpu.min.mjs`));
    ort.env.wasm.wasmPaths = ORT;
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
    const tf = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.2");
    tf.env.allowRemoteModels = false;
    tf.env.allowLocalModels = true;
    tf.env.localModelPath = new URL("..", this.base).href; // tokenizer files live in <base> = ".../model/"
    this.tokenizer = await tf.AutoTokenizer.from_pretrained(new URL(this.base).pathname.split("/").filter(Boolean).pop());
    const [vmap, schema] = await Promise.all([
      fetch(this.base + "vocab_map.json").then((r) => r.json()),
      fetch(this.base + "state_schema.json").then((r) => r.json()),
    ]);
    this.vmap = vmap.map;
    this.unk = vmap.unk;
    this.keys = schema.keys;

    // download with progress (~70 MB, cached by the browser afterwards)
    const resp = await fetch(this.base + "laya_web.onnx");
    const total = +resp.headers.get("content-length") || 0;
    const reader = resp.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      onProgress(total ? got / total : 0, got);
    }
    const bytes = new Uint8Array(got);
    let o = 0;
    for (const c of chunks) { bytes.set(c, o); o += c.length; }

    const providers = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
    try {
      this.session = await ort.InferenceSession.create(bytes, { executionProviders: providers, graphOptimizationLevel: "all" });
      this.backend = providers[0];
    } catch (e) {
      console.warn("WebGPU session failed, falling back to WASM", e);
      this.session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      this.backend = "wasm";
    }
    this.available = true;
  }

  async decide(state) {
    if (!this.available) throw new Error("OnnxPolicy не инициализирован");
    const t0 = performance.now();
    const text = `State: ${pyJsonDumps(toTrainingSchema(state, this.keys))}
Question: ${QUESTION}
Choices: UP, DOWN, LEFT, RIGHT`;
    const ids = this.tokenizer.encode(text); // adds [CLS] ... [SEP] like the Python tokenizer
    const n = Math.min(ids.length, 512);
    const inputIds = new BigInt64Array(n), mask = new BigInt64Array(n);
    for (let i = 0; i < n; i++) { inputIds[i] = BigInt(this.vmap[ids[i]] ?? this.unk); mask[i] = 1n; }
    const T = this.ort.Tensor;
    const t1 = performance.now();
    const out = await this.session.run({ input_ids: new T("int64", inputIds, [1, n]), attention_mask: new T("int64", mask, [1, n]) });
    const latency = performance.now() - t1;
    const logits = Array.from(out.logits.data);
    const m = Math.max(...logits);
    const e = logits.map((x) => Math.exp(x - m));
    const z = e.reduce((a, b) => a + b, 0);
    const probabilities = { UP: e[0] / z, DOWN: e[1] / z, LEFT: e[2] / z, RIGHT: e[3] / z };
    const action = DIRS.reduce((a, b) => (probabilities[b] > probabilities[a] ? b : a), "UP");
    return { action, probabilities, confidence: probabilities[action], latencyMs: latency, roundTripMs: performance.now() - t0,
      engine: `Laya в браузере (${this.backend === "webgpu" ? "видеокарта" : "процессор"})` };
  }

  dispose() { this.session?.release?.(); }
}

// ======================================================================
// 3. LocalBfsPolicy — детерминированный фолбэк (порт findBestFoodMove из web_server.py)
// ======================================================================
export class LocalBfsPolicy {
  constructor() {
    this.id = "bfs";
    this.label = "Алгоритм без ИИ";
  }
  async init() {}

  async decide(state, game) {
    const t0 = performance.now();
    const action = LocalBfsPolicy.compute(game);
    const dt = performance.now() - t0;
    return { action, probabilities: oneHot(action), confidence: action ? 1 : 0, latencyMs: dt, roundTripMs: dt, engine: "Алгоритм поиска пути (BFS)" };
  }

  // --- вспомогательные функции; работают только на чтение состояния game ---
  static _blocked(game, x, y, bodySet) {
    if (x < 0 || x >= GRID || y < 0 || y >= GRID) return true;
    const key = `${x},${y}`;
    if (game.immortalSteps <= 0 && bodySet.has(key)) return true;
    if (game.foods.some((f) => f.x === x && f.y === y && f.type === "poison")) return true;
    // Улучшение относительно оригинала: стены тоже не считаются свободным пространством.
    if (game.obstacles.some((o) => o.x === x && o.y === y)) return true;
    return false;
  }

  static floodFillSpace(game, start, customSnake) {
    const visited = new Set([`${start.x},${start.y}`]);
    const queue = [start];
    const body = new Set(customSnake.map((s) => `${s.x},${s.y}`));
    while (queue.length) {
      const c = queue.shift();
      for (const d of DIRS) {
        const nx = c.x + ACTIONS[d].x, ny = c.y + ACTIONS[d].y, key = `${nx},${ny}`;
        if (!visited.has(key) && !LocalBfsPolicy._blocked(game, nx, ny, body)) { visited.add(key); queue.push({ x: nx, y: ny }); }
      }
    }
    return visited.size;
  }

  static canReachTail(game, start, customSnake) {
    const tail = customSnake[customSnake.length - 1];
    const visited = new Set([`${start.x},${start.y}`]);
    const queue = [start];
    const body = new Set(customSnake.slice(0, -1).map((s) => `${s.x},${s.y}`));
    while (queue.length) {
      const c = queue.shift();
      if (c.x === tail.x && c.y === tail.y) return true;
      for (const d of DIRS) {
        const nx = c.x + ACTIONS[d].x, ny = c.y + ACTIONS[d].y, key = `${nx},${ny}`;
        if (!visited.has(key) && !LocalBfsPolicy._blocked(game, nx, ny, body)) { visited.add(key); queue.push({ x: nx, y: ny }); }
      }
    }
    return false;
  }

  /**
   * Прогноз дронов: клетка → тики, когда в ней будет дрон (с отскоками от текущих твёрдых стен).
   * Дрон, въехавший в ЛЮБОЙ сегмент тела, убивает змею, поэтому фолбэк не пересекает полосу перед дроном,
   * а проходит за ним. Плюс фазовые стены: призрачная клетка, которая затвердеет к моменту прихода, тоже закрыта.
   * Возвращает unsafe(key, d): голова войдёт в клетку через d тиков и тело пробудет там до ~d + длина.
   */
  static patrolGuard(game, horizon = 48) {
    const times = new Map();
    for (const p0 of game.drones) {
      const p = cloneDrone(p0);
      for (let t = 1; t <= horizon; t++) {
        droneStep(p, game.solid);
        const key = `${p.x},${p.y}`;
        if (!times.has(key)) times.set(key, []);
        times.get(key).push(t);
      }
    }
    const ghostLeft = new Map();
    for (const s of game.structures) if (s.phase === "GHOST") s.cells.forEach((c) => ghostLeft.set(`${c.x},${c.y}`, s.timer));
    if (!times.size && !ghostLeft.size) return null;
    const L = game.snake.length, imm = game.immortalSteps;
    return (key, d) => {
      const g = ghostLeft.get(key);
      if (g !== undefined && g <= d + 1) return true; // может затвердеть, пока мы там
      const ts = times.get(key);
      if (!ts) return false;
      for (const t of ts) if (t > imm && t >= d - 1 && t <= d + L) return true; // пока бессмертны — дрон не страшен
      return false;
    };
  }

  static validMoves(game, guard = null) {
    const h = game.snake[0];
    return DIRS.filter((d) => {
      if (d === OPPOSITES[game.currentDir] && game.snake.length > 1) return false;
      const np = { x: h.x + ACTIONS[d].x, y: h.y + ACTIONS[d].y };
      if (guard && guard(`${np.x},${np.y}`, 1)) return false;
      return !game.isHazard(np); // правила V9: стены/граница всегда, тело/яд/дроны — если не бессмертны
    });
  }

  static tailChasing(game, moves) {
    moves = moves || LocalBfsPolicy.validMoves(game);
    if (!moves.length) return null;
    const head = game.snake[0], tail = game.snake[game.snake.length - 1];
    let best = moves[0], bestScore = -Infinity;
    for (const d of moves) {
      const np = { x: head.x + ACTIONS[d].x, y: head.y + ACTIONS[d].y };
      const vs = [np, ...game.snake.slice(0, -1)];
      const space = LocalBfsPolicy.floodFillSpace(game, np, vs);
      const reach = LocalBfsPolicy.canReachTail(game, np, vs) ? 100 : 0;
      const s = space * 10 + reach - (Math.abs(np.x - tail.x) + Math.abs(np.y - tail.y));
      if (s > bestScore) { bestScore = s; best = d; }
    }
    return best;
  }

  static exploration(game, moves) {
    const head = game.snake[0], tail = game.snake[game.snake.length - 1];
    const cands = [];
    for (const d of moves) {
      const delta = ACTIONS[d];
      const np = { x: head.x + delta.x, y: head.y + delta.y };
      const vs = [np, ...game.snake.slice(0, -1)];
      const space = LocalBfsPolicy.floodFillSpace(game, np, vs);
      const reach = LocalBfsPolicy.canReachTail(game, np, vs);
      if (space < Math.min(game.snake.length, 10) && !reach) continue;
      const momentum = d === game.currentDir ? 50 : 0;
      const last = game.cellLastVisit[`${np.x},${np.y}`] ?? -999;
      const ago = game.stepCount - last;
      const recency = ago < 35 ? (35 - ago) * 25 : 0;
      let patch = 0;
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
        const ns = game.cellLastVisit[`${np.x + dx},${np.y + dy}`] ?? -999;
        if (game.stepCount - ns < 15) patch += 8;
      }
      const tailDist = Math.abs(np.x - tail.x) + Math.abs(np.y - tail.y);
      const tailScore = game.snake.length <= 8 ? tailDist * 20 : reach ? 40 : 0;
      const spaceBonus = Math.min(space, 50) * 2;
      const nn = { x: np.x + delta.x, y: np.y + delta.y };
      const look = nn.x >= 0 && nn.x < GRID && nn.y >= 0 && nn.y < GRID && !game.isHazard(nn) ? 15 : 0;
      cands.push({ d, s: momentum - recency - patch + tailScore + spaceBonus + look });
    }
    if (!cands.length) return LocalBfsPolicy.tailChasing(game, moves);
    cands.sort((a, b) => b.s - a.s);
    return cands[0].d;
  }

  /** Порт findBestFoodMove(): BFS к ближайшей видимой еде + защита от запирания в собственном хвосте. */
  static compute(game) {
    // Если давно не ели (еда на полосе дрона, а змея длинная) — рискуем и идём без «охраны».
    const desperate = game.stepCount - game.lastEatStep > 120;
    const guard = desperate ? null : LocalBfsPolicy.patrolGuard(game);
    return LocalBfsPolicy._compute(game, guard) ?? LocalBfsPolicy._compute(game, null);
  }

  static _compute(game, guard) {
    const head = game.snake[0];
    const food = game.getNearestGoodFood();
    const moves = LocalBfsPolicy.validMoves(game, guard);
    if (!moves.length) return null;
    if (food.x === -1) return LocalBfsPolicy.exploration(game, moves);

    const visited = new Set([`${head.x},${head.y}`]);
    const queue = [];
    for (const d of moves) {
      const np = { x: head.x + ACTIONS[d].x, y: head.y + ACTIONS[d].y };
      visited.add(`${np.x},${np.y}`);
      queue.push({ x: np.x, y: np.y, first: d, depth: 1 });
    }
    let pathDir = null;
    while (queue.length) {
      const c = queue.shift();
      if (c.x === food.x && c.y === food.y) { pathDir = c.first; break; }
      for (const d of DIRS) {
        const nx = c.x + ACTIONS[d].x, ny = c.y + ACTIONS[d].y, key = `${nx},${ny}`;
        if (nx >= 0 && nx < GRID && ny >= 0 && ny < GRID && !visited.has(key) && !(guard && guard(key, c.depth + 1)) && !game.isHazard({ x: nx, y: ny }) &&
            !(game.immortalSteps > 0 && c.depth + 1 >= game.immortalSteps - 1 && game.snake.some((b) => b.x === nx && b.y === ny))) {
          visited.add(key);
          queue.push({ x: nx, y: ny, first: c.first, depth: c.depth + 1 });
        }
      }
    }
    if (pathDir) {
      const np = { x: head.x + ACTIONS[pathDir].x, y: head.y + ACTIONS[pathDir].y };
      const vs = [np, ...game.snake.slice(0, -1)];
      if (LocalBfsPolicy.floodFillSpace(game, np, vs) >= game.snake.length || LocalBfsPolicy.canReachTail(game, np, vs)) return pathDir;
    }
    return LocalBfsPolicy.tailChasing(game, moves);
  }

  dispose() {}
}

// ======================================================================
// Итоговый выбор хода: «маска безопасности» из старого executeStep()
// ======================================================================
/**
 * @param {object|null} decision  ответ провайдера (null, если провайдер упал/таймаут)
 * @param {SnakeGame} game
 * @param {string|null} bfsAction  ход локального BFS (фолбэк)
 * @param {{safetyMask:boolean}} opts
 * @returns {{action:string, masked:boolean, fallback:boolean}}
 */
export function resolveAction(decision, game, bfsAction, { safetyMask = true } = {}) {
  const current = game.currentDir;
  if (!decision || !decision.action) return { action: bfsAction || current, masked: false, fallback: true };
  if (!safetyMask) return { action: decision.action, masked: false, fallback: false };
  const probs = decision.probabilities || {};
  const cands = DIRS.map((d) => {
    const deadly = game.isDeadlyMove(d);
    return { d, p: deadly ? -1 : probs[d] ?? 0, safe: !deadly };
  }).sort((a, b) => b.p - a.p);
  if (cands[0].safe) return { action: cands[0].d, masked: cands[0].d !== decision.action, fallback: false };
  if (decision.action !== OPPOSITES[current]) return { action: decision.action, masked: false, fallback: false };
  return { action: bfsAction || current, masked: true, fallback: true };
}
