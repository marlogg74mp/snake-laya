// main.js — связывает SnakeGame (логика), политику (выбор хода) и Renderer3D (картинка) + HUD.
import { SnakeGame, DIRS, ACTIONS, CRASH_REASONS, IMMORTAL_STEPS } from "./game.js";
import { ServerPolicy, OnnxPolicy, LocalBfsPolicy, resolveAction, toTrainingSchema } from "./policy.js";
import { Renderer3D } from "./render3d.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

// ---------------------------------------------------------------- настройки
const cfg = {
  serverBase: params.get("server") || "",
  timeoutMs: +(params.get("timeout") || 400),
  schema: ["legacy", "v7"].includes(params.get("schema")) ? params.get("schema") : "v9", // v9 = RULES_V9 §5
  forcedProvider: params.get("provider"),
  quality: params.get("quality") === "low" ? "low" : "high",
  qualityForced: params.has("quality"),
};

const ui = {
  running: true,
  speedMul: 1,
  safetyMask: true,
  autoReboot: true,
  crashTelemetry: false,
};

const REASON_RU = {
  [CRASH_REASONS.LOOP]: "зацикливание",
  [CRASH_REASONS.PATROL]: "столкновение с дроном",
  [CRASH_REASONS.POISON]: "съеден яд",
  [CRASH_REASONS.WALL]: "удар о стену",
  [CRASH_REASONS.BOUNDARY]: "граница или собственное тело",
};
const ARROWS = { UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→" };
const DIR_RU = { UP: "ВВЕРХ", DOWN: "ВНИЗ", LEFT: "ВЛЕВО", RIGHT: "ВПРАВО" };

// ---------------------------------------------------------------- ядро
const game = new SnakeGame();
let renderer;
try {
  renderer = new Renderer3D($("stage"), { quality: cfg.quality });
} catch (e) {
  console.error(e);
  $("fatal").hidden = false;
  throw e;
}
renderer.handleEvents(game.drainEvents(), game);

const policies = {
  server: new ServerPolicy({ baseUrl: cfg.serverBase, model: "laya", timeoutMs: cfg.timeoutMs }),
  onnx: new OnnxPolicy(),
  bfs: new LocalBfsPolicy(),
};
let provider = policies.bfs;
let serverHealth = { ok: false };
let serverFailStreak = 0;
let autoProvider = !params.get("provider"); // false once the user picks a provider by hand

// ---------------------------------------------------------------- HUD-хелперы
const setText = (el, v) => { v = String(v); if (el.textContent !== v) el.textContent = v; };
let bannerLock = 0;
function banner(text, tone = "ok", holdMs = 0) {
  const el = $("banner");
  setText($("bannerText"), text);
  el.dataset.tone = tone;
  if (holdMs) bannerLock = performance.now() + holdMs;
}
function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove("show"), 3200);
}

let onnxLoading = false;
async function setProvider(id) {
  if (id === "onnx" && !policies.onnx.available) {
    if (onnxLoading) return;
    onnxLoading = true;
    toast("Загружаю модель Laya в браузер…");
    try {
      await policies.onnx.init((frac, bytes) => toast(`Загрузка модели: ${frac ? Math.round(frac * 100) + "%" : (bytes / 1e6).toFixed(0) + " МБ"}`));
      toast(`Laya работает в браузере (${policies.onnx.backend === "webgpu" ? "WebGPU" : "WASM"}) — сервер не нужен`);
    } catch (e) {
      console.error(e);
      toast("Не удалось запустить ONNX: " + e.message);
      onnxLoading = false;
      return;
    }
    onnxLoading = false;
  }
  provider = policies[id];
  document.querySelectorAll("[data-provider]").forEach((b) => b.classList.toggle("active", b.dataset.provider === id));
  $("serverOpts").hidden = id !== "server";
  $("telemetryRow").hidden = id !== "server";
  serverFailStreak = 0;
  updateProviderStatus();
}

function updateProviderStatus() {
  const el = $("providerStatus");
  if (provider.id === "server") {
    if (serverFailStreak > 2) { el.dataset.tone = "warn"; setText(el, "сервер не отвечает → ходит алгоритм без ИИ"); }
    else if (serverHealth.ok) { el.dataset.tone = "ok"; setText(el, `сервер на связи · ${serverHealth.laya_loaded ? "Laya готова" : "Laya не загружена"}${serverHealth.cuda_available ? " · считает видеокарта" : " · считает процессор"}`); }
    else { el.dataset.tone = "warn"; setText(el, "сервер не найден → ходит алгоритм без ИИ"); }
  } else if (provider.id === "onnx") {
    el.dataset.tone = "ok";
    setText(el, `Laya считает в этом браузере · ${policies.onnx.backend === "webgpu" ? "на видеокарте (WebGPU)" : "на процессоре (WASM)"} · сервер не нужен`);
  } else {
    el.dataset.tone = "ok";
    setText(el, "поиск пути (BFS) прямо в браузере, без нейросети");
  }
}

// ---------------------------------------------------------------- шаг симуляции
let busy = false;
let lastStepAt = 0;
let stepAppliedAt = performance.now();
const latHist = [];
let lastState = null;

// sub-millisecond times (BFS) in microseconds; browsers round performance.now() to ~0.1 ms without isolation
function fmtMs(ms) {
  if (ms < 0.1) return "<100 мкс";
  if (ms < 1) return `${Math.round(ms * 1000)} мкс`;
  return `${ms.toFixed(ms < 10 ? 2 : 0)} мс`;
}

async function doStep() {
  if (busy || game.over) return;
  busy = true;
  const epoch = game.epoch;
  const state = game.getModelState(cfg.schema);
  lastState = state;
  const tb = performance.now();
  const bfsAction = LocalBfsPolicy.compute(game); // also the fallback for the other providers
  const bfsMs = performance.now() - tb;
  let decision = null;
  const t0 = performance.now();
  if (provider.id === "bfs") {
    const dt = bfsMs;
    decision = { action: bfsAction, probabilities: { UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0, ...(bfsAction ? { [bfsAction]: 1 } : {}) }, confidence: bfsAction ? 1 : 0, latencyMs: dt, roundTripMs: dt, engine: "Алгоритм поиска пути (BFS)" };
  } else {
    try {
      decision = await provider.decide(state, game);
      serverFailStreak = 0;
    } catch (e) {
      serverFailStreak++;
      decision = null;
    }
    updateProviderStatus();
  }
  if (epoch !== game.epoch || game.over) { busy = false; return; }

  const res = resolveAction(decision, game, bfsAction, { safetyMask: ui.safetyMask });
  updateDecisionHud(state, decision, res);
  const out = game.step(res.action);
  stepAppliedAt = performance.now();
  if (out.crashed) onCrash(state, res.action, out.reason);
  busy = false;
}

function onCrash(state, action, reason) {
  banner(`АВАРИЯ: ${REASON_RU[reason] || reason} [${game.crashedCell.x}, ${game.crashedCell.y}]`, "bad", 2500);
  document.body.classList.add("crashed");
  setTimeout(() => document.body.classList.remove("crashed"), 700);
  if (ui.crashTelemetry && provider.id === "server") policies.server.recordCrash({ state, wrongAction: action, reason });
  const epoch = game.epoch;
  if (ui.autoReboot) setTimeout(() => { if (game.over && game.epoch === epoch) game.reset(); }, 2000);
}

// ---------------------------------------------------------------- HUD: решение
function updateDecisionHud(state, d, res) {
  // No model answer (timeout / server down): show dashes instead of fake 25% bars
  const probs = d?.probabilities || { UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0 };
  setText($("actionArrow"), ARROWS[res.action] || "·");
  setText($("actionText"), DIR_RU[res.action] || res.action);
  setText($("confVal"), d ? `${(d.confidence * 100).toFixed(1)}%` : "—");
  setText($("engineName"), d ? d.engine : "Локальный BFS (фолбэк)");
  $("flagMask").hidden = !res.masked;
  $("flagFallback").hidden = !res.fallback;
  for (const dir of DIRS) {
    const p = Math.max(0, probs[dir] || 0);
    $(`bar-${dir}`).style.transform = `scaleX(${p})`;
    setText($(`pct-${dir}`), d ? `${(p * 100).toFixed(1)}%` : "—");
    $(`prob-${dir}`).classList.toggle("winner", dir === res.action);
    const cell = $(`radar-${dir}`);
    const h = game.snake[0], nx = h.x + ACTIONS[dir].x, ny = h.y + ACTIONS[dir].y;
    const poison = game.foods.some((f) => f.x === nx && f.y === ny && f.type === "poison");
    cell.dataset.state = state[`danger_${dir}`] ? (poison ? "poison" : "danger") : "safe";
  }
  if (d) {
    latHist.push(d.roundTripMs);
    if (latHist.length > 80) latHist.shift();
    setText($("latVal"), fmtMs(d.latencyMs));
    setText($("rttVal"), fmtMs(d.roundTripMs));
  }
  if ($("stateBox").open) {
    const ts = cfg.schema === "v9" ? toTrainingSchema(state) : state;
    setText($("stateJson"), "{\n" + Object.keys(ts).map((k) => `  "${k}": ${JSON.stringify(ts[k])}`).join(",\n") + "\n}");
  }
}

function drawSparkline() {
  const c = $("spark"), g = c.getContext("2d");
  const w = c.width, h = c.height;
  g.clearRect(0, 0, w, h);
  if (latHist.length < 2) return;
  const max = Math.max(5, ...latHist) * 1.15;
  g.beginPath();
  latHist.forEach((v, i) => { const x = (i / (80 - 1)) * w, y = h - (v / max) * (h - 4) - 2; i ? g.lineTo(x, y) : g.moveTo(x, y); });
  g.strokeStyle = "#3dffa8"; g.lineWidth = 1.5; g.shadowColor = "#3dffa8"; g.shadowBlur = 6; g.stroke();
  g.shadowBlur = 0;
  g.lineTo(((latHist.length - 1) / 79) * w, h); g.lineTo(0, h); g.closePath();
  g.fillStyle = "rgba(61,255,168,0.08)"; g.fill();
}

// ---------------------------------------------------------------- события игры → HUD
function handleHudEvents(events) {
  for (const e of events) {
    if (e.type === "reset") { banner("СИМУЛЯЦИЯ АКТИВНА", "ok"); bannerLock = 0; }
    if (e.type === "eat") {
      const ft = e.food.type;
      if (e.neutralized) { /* см. ниже */ }
      else if (ft === "shrink") banner("МАГИЧЕСКОЕ ЯБЛОКО: ДЛИНА −2", "cyan", 1600);
      else if (ft === "golden") banner("ЗОЛОТОЕ ЯБЛОКО +3", "gold", 1200);
      $("scoreCard").classList.remove("bump"); void $("scoreCard").offsetWidth; $("scoreCard").classList.add("bump");
    }
    if (e.type === "speed") {
      banner(e.effect === "turbo" ? "ТУРБО: СКОРОСТЬ ×2" : "ЗАМОРОЗКА: СКОРОСТЬ ×0.5", e.effect === "turbo" ? "gold" : "cyan", 1800);
      document.body.dataset.effect = e.effect;
    }
    if (e.type === "speedEnd") document.body.dataset.effect = "";
    if (e.type === "teleport") banner("ПОРТАЛ: ТЕЛЕПОРТАЦИЯ", "violet", 700);
    if (e.type === "immortal") banner(`БЕССМЕРТИЕ: ${e.steps} ХОДОВ — ДРОНЫ, ТЕЛО И ЯД НЕ СТРАШНЫ`, "gold", 2200);
    if (e.type === "immortalEnd") banner("БЕССМЕРТИЕ ЗАКОНЧИЛОСЬ", "dim", 1200);
    if (e.type === "eat" && e.neutralized) banner("ЯД НЕЙТРАЛИЗОВАН БЕССМЕРТИЕМ", "violet", 1200);
  }
}

function updateStatsHud(fps) {
  setText($("scoreVal"), game.score);
  setText($("lenVal"), game.snake.length);
  setText($("bestVal"), game.maxScore);
  setText($("stepVal"), game.stepCount);
  setText($("crashVal"), game.crashes);
  setText($("fpsVal"), fps ? `${fps} FPS` : "");
  const imm = game.immortalSteps;
  $("immortalBar").hidden = imm <= 0;
  if (imm > 0) {
    setText($("immortalVal"), imm);
    $("immortalFill").style.transform = `scaleX(${Math.min(1, imm / IMMORTAL_STEPS)})`;
    $("immortalBar").classList.toggle("warn", imm <= 8);
  }
  const tick = Math.round(game.tickInterval / ui.speedMul);
  setText($("tickVal"), `${tick} мс/шаг`);
  if (!game.over && performance.now() > bannerLock) {
    if (game.settings.fog && game.getNearestGoodFood().x === -1) banner("ТУМАН ВОЙНЫ: ПАТРУЛИРОВАНИЕ АРЕНЫ", "cyan");
    else if (!ui.running) banner("ПАУЗА", "dim");
    else banner("ЦЕЛЬ ЗАХВАЧЕНА // ИДЁМ К ЕДЕ", "ok");
  }
}

// ---------------------------------------------------------------- главный цикл
let prevT = performance.now();
let fpsFrames = 0, fpsT = prevT, fps = 0, slowSeconds = 0;
const startT = prevT;

function loop(now) {
  requestAnimationFrame(loop);
  const dt = Math.min(0.1, (now - prevT) / 1000);
  prevT = now;
  const interval = game.tickInterval / ui.speedMul;

  if (ui.running && !game.over) {
    game.update(dt * 1000 * ui.speedMul);
    if (!busy && now - lastStepAt >= interval) { lastStepAt = now; doStep(); }
  }
  const events = game.drainEvents();
  if (events.length) { renderer.handleEvents(events, game); handleHudEvents(events); }

  renderer.tickMs = interval;
  const alpha = (now - stepAppliedAt) / interval;
  renderer.frame(game, dt, alpha);

  // FPS + авто-деградация качества
  fpsFrames++;
  if (now - fpsT >= 1000) {
    fps = Math.round((fpsFrames * 1000) / (now - fpsT));
    fpsFrames = 0; fpsT = now;
    updateStatsHud(fps);
    drawSparkline();
    if (!cfg.qualityForced && renderer.quality === "high" && now - startT > 3000 && !document.hidden) {
      slowSeconds = fps < 40 ? slowSeconds + 1 : 0;
      if (slowSeconds >= 3) { applyQuality("low"); toast("Низкий FPS: включён облегчённый режим (без bloom)"); }
    }
  } else if (fpsFrames % 6 === 0) updateStatsHud(fps);
}

// ---------------------------------------------------------------- управление
function applyQuality(q) {
  renderer.setQuality(q);
  $("bloomChk").checked = renderer.bloomOn;
  document.querySelectorAll("[data-quality]").forEach((b) => b.classList.toggle("active", b.dataset.quality === q));
}
function setCamera(m) {
  renderer.setCameraMode(m);
  document.querySelectorAll("[data-cam]").forEach((b) => b.classList.toggle("active", b.dataset.cam === m));
}
function togglePause() {
  ui.running = !ui.running;
  setText($("btnPlay"), ui.running ? "Пауза" : "Пуск");
  $("btnPlay").classList.toggle("active", !ui.running);
}
function stepOnce() { if (!ui.running && !game.over) doStep(); }
function bindToggle(id, fn) { const el = $(id); el.addEventListener("change", () => fn(el.checked)); }

document.querySelectorAll("[data-provider]").forEach((b) => b.addEventListener("click", () => { autoProvider = false; setProvider(b.dataset.provider); }));
document.querySelectorAll("[data-cam]").forEach((b) => b.addEventListener("click", () => setCamera(b.dataset.cam)));
document.querySelectorAll("[data-quality]").forEach((b) => b.addEventListener("click", () => { cfg.qualityForced = true; applyQuality(b.dataset.quality); }));
$("serverModel").addEventListener("change", (e) => {
  policies.server.model = e.target.value;
  // Jev goes through OpenRouter (~0.4-1 s); the local Laya answers in tens of ms
  policies.server.timeoutMs = e.target.value === "jev" ? Math.max(3000, cfg.timeoutMs) : cfg.timeoutMs;
});
$("btnPlay").addEventListener("click", togglePause);
$("btnStep").addEventListener("click", stepOnce);
$("btnReset").addEventListener("click", () => game.reset());
$("speed").addEventListener("input", (e) => { ui.speedMul = +e.target.value; setText($("speedVal"), `×${ui.speedMul.toFixed(2).replace(/0$/, "")}`); });
bindToggle("wallsChk", (v) => game.setSetting("walls", v));
bindToggle("portalsChk", (v) => game.setSetting("portals", v));
bindToggle("patrolsChk", (v) => game.setSetting("patrols", v));
bindToggle("fogChk", (v) => game.setSetting("fog", v));
bindToggle("maskChk", (v) => (ui.safetyMask = v));
bindToggle("rebootChk", (v) => { ui.autoReboot = v; if (v && game.over) game.reset(); });
bindToggle("telemetryChk", (v) => (ui.crashTelemetry = v));
bindToggle("bloomChk", (v) => renderer.setBloom(v));
bindToggle("autorotChk", (v) => renderer.setAutoRotate(v));
$("hudToggle").addEventListener("click", () => document.body.classList.toggle("hud-hidden"));
document.querySelectorAll(".panel-toggle").forEach((b) => b.addEventListener("click", () => b.closest(".panel").classList.toggle("collapsed")));

const FOOD_CYCLE = ["standard", "golden", "shrink", "turbo", "freeze", "poison", "immortal"];
let foodCycleIdx = -1;
renderer.onCellClick((x, y, e) => {
  // клик по полю — подбросить еду (как в старом UI), Shift+клик — типы по очереди
  const type = e?.shiftKey ? FOOD_CYCLE[(foodCycleIdx = (foodCycleIdx + 1) % FOOD_CYCLE.length)] : null;
  if (!game.spawnFood({ x, y }, type)) toast("Клетка занята");
});

window.addEventListener("keydown", (e) => {
  if (e.target.matches("input, select, textarea")) return;
  const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); togglePause(); }
  else if (k === "n" || k === "arrowright") stepOnce();
  else if (k === "r") game.reset();
  else if (k === "1") setCamera("orbit");
  else if (k === "2") setCamera("top");
  else if (k === "3") setCamera("chase");
  else if (k === "f") { $("fogChk").click(); }
  else if (k === "b") { $("bloomChk").click(); }
  else if (k === "h") document.body.classList.toggle("hud-hidden");
});

// ---------------------------------------------------------------- старт
if (innerWidth < 820) document.querySelectorAll(".panel").forEach((p) => p.classList.add("collapsed"));
applyQuality(cfg.quality);
setProvider("bfs");
(async () => {
  serverHealth = await ServerPolicy.probe(cfg.serverBase);
  const want = cfg.forcedProvider || (serverHealth.ok && serverHealth.laya_loaded ? "server" : "bfs");
  if (want === "server" || want === "bfs") setProvider(want);
  if (serverHealth.ok) toast(`Сервер найден: ${serverHealth.laya_loaded ? "модель Laya готова" : "модель не загружена"}`);
  updateProviderStatus();
})();
// If the page started while the server was down, keep checking and switch to it once it is up
setInterval(async () => {
  if (!autoProvider || provider.id !== "bfs") return;
  serverHealth = await ServerPolicy.probe(cfg.serverBase);
  if (serverHealth.ok && serverHealth.laya_loaded) {
    setProvider("server");
    toast("Сервер снова доступен: модель Laya подключена");
  }
}, 5000);
requestAnimationFrame(loop);
window.__snake = { game, renderer, policies }; // для отладки из консоли
