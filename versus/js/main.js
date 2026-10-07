// main.js — game loop, input, AI calls and HUD for Snake Duel (human vs Laya).
import { VersusMatch, MATCH_STEPS, DIRS, OPP } from "./versus_game.js";
import { Renderer2D } from "./render2d.js";
import { Renderer3D } from "./render3d.js";
import { OnnxPolicy, Laya421Policy } from "./onnx_policy.js";
import { windowState } from "./window_state.js";
import { shieldMove } from "./shield.js";

// Laya's brain: the model server (/api/predict) or the same model in the browser (ONNX, no server needed)
const onnx = new OnnxPolicy("model/");
const onnx421 = new Laya421Policy("model_laya421/");
const onnxV11w = new OnnxPolicy("model_v11w/", "Laya v11w");
// brain: which model plays Laya's snake and where it runs
//   server = v10 on the model server · onnx = v10 in the browser · convai = Laya 421M on the server · onnx421 = Laya 421M in the browser
let brain = "server";
const inBrowser = () => brain === "onnx" || brain === "onnx421" || brain === "onnx_v11w";
const usesLaya421 = () => brain === "convai" || brain === "onnx421";
// window models (egocentric view + memory, any board size): v11w, v12 (+ apple values) — server only for now
const WINDOW_MODELS = { v11w: "laya_v11w", v12: "laya_v12", onnx_v11w: "laya_v11w" }; // onnx_v11w = v11w in the browser
const isWindowModel = (model) => model === "laya_v11w" || model === "laya_v12";
// safety shield: a look-ahead in a small simulator vetoes the model's move if it leads into a trap (shield.js)
let shieldOn = true;
const vetoes = { ai: 0, human: 0 };
let rttEma = 0; // smoothed model answer time; in-browser models get more time before the BFS fallback

const $ = (id) => document.getElementById(id);
const SPEEDS = { slow: 250, normal: 150, fast: 100 }; // ms per step; the match is always MATCH_STEPS steps
const SPEED_RU = { slow: "Тренировка", normal: "Норма", fast: "Турбо" };
let speed = "normal";
let TICK_MS = SPEEDS[speed];
let AI_WAIT_MS = 250;            // max extra wait for the model before the BFS fallback is used
const ARROWS = { UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→" };
const DIR_RU = { UP: "ВВЕРХ", DOWN: "ВНИЗ", LEFT: "ВЛЕВО", RIGHT: "ВПРАВО" };
const REASON_RU = {
  border: "в границу", wall: "в стену", self: "в себя", drone: "от дрона", poison: "от яда",
  "head-on": "лоб в лоб", opponent: "в соперника",
};
const NAME = { human: "Вы", ai: "Laya" };

// 3D by default; 2D with ?render=2d, on the toggle, or when WebGL is unavailable
let renderer;
function useRenderer(kind) {
  if (renderer?.dispose) renderer.dispose();
  renderer = null;
  if (kind === "3d") {
    try {
      renderer = new Renderer3D($("board").parentElement);
      $("board").hidden = true;
    } catch (e) {
      console.warn("WebGL unavailable, using 2D", e);
      kind = "2d";
    }
  }
  if (!renderer) {
    $("board").hidden = false;
    renderer = new Renderer2D($("board"));
  }
  renderer.setIdle?.(mode !== "play");
  try {
    const v = localStorage.getItem("duelView");
    for (let i = 0; i < 3 && renderer.view && ["tilt", "top", "chase"].includes(v) && renderer.view !== v; i++) renderer.toggleView();
  } catch {}
  $("viewBtn").textContent = kind === "3d" ? "2D" : "3D";
  $("camBtn").hidden = kind !== "3d";
  $("viewBtn").dataset.kind = kind;
  try { localStorage.setItem("duelRender", kind); } catch {}
}
let match = new VersusMatch();
let mode = "menu"; // menu | countdown | play | pause | end
let savedRender = null;
try { savedRender = localStorage.getItem("duelRender"); } catch {}
useRenderer(new URLSearchParams(location.search).get("render") || savedRender || "3d");
let lastTick = 0;
let inputQueue = [];
let aiPending = null; // {promise, result, startedAt}
let leftPending = null; // the left snake's model move in AI-vs-AI mode
let watch = false; // AI vs AI: left = our v10, right = Laya 421M (or v11w when the 421M files are not hosted)
let has421 = true; // Laya 421M (226 MB) does not fit GitHub Pages; without its files the watch mode uses v11w
const watchModel = () => (has421 ? "laya_convai" : "laya_v11w");
const watchName = () => (has421 ? "LAYA 421M" : "LAYA v11w");
let names = { human: "ВЫ", ai: "LAYA" };
let stepping = false;

// ---------------------------------------------------------------- AI
// model: "laya" = our v10 (ModernBERT-base), "laya_convai" = the real Laya 421M fine-tuned on Snake
async function askModel(state, model = "laya") {
  const t0 = performance.now();
  if (inBrowser()) {
    const r = await (model === "laya_convai" ? onnx421 : model === "laya_v11w" ? onnxV11w : onnx).decide(state);
    const rtt = performance.now() - t0;
    rttEma = rttEma ? rttEma * 0.8 + rtt * 0.2 : rtt;
    return { ...r, rtt };
  }
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 1500);
  try {
    const r = await fetch("api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state, model }), signal: ctrl.signal,
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    return { ...d, rtt: performance.now() - t0 };
  } finally {
    clearTimeout(timer);
  }
}

function moveJob(snake, model) {
  if (!snake.alive || match.over) return null;
  const state = isWindowModel(model) ? windowState(match, snake, model === "laya_v12" ? "v12" : "v11w") : match.modelState(snake);
  const job = { state, result: null, done: false, startedAt: performance.now() };
  job.promise = askModel(state, model).then((r) => { job.result = r; }).catch(() => {}).finally(() => { job.done = true; });
  return job;
}

function requestAiMove() {
  aiPending = moveJob(match.ai, watch ? watchModel() : WINDOW_MODELS[brain] || (usesLaya421() ? "laya_convai" : "laya"));
  leftPending = watch ? moveJob(match.human, "laya") : null;
}

function chooseAiAction(job, s = match.ai, panel = true) {
  const state = job?.state ?? match.modelState(s);
  const legal = DIRS.filter((d) => !state[`danger_${d}`] && d !== OPP[s.dir]);
  const r = job?.result;
  if (!r) {
    const a = match.fallbackMove(s, match.modelState(s)); // the BFS fallback reads the full-board state
    if (panel) showBrain(null, a, "алгоритм без ИИ (Laya не успела ответить)");
    return a;
  }
  const p = r.probabilities;
  const byP = (a, b) => p[b] - p[a];
  const ranked = (legal.length ? legal : DIRS.filter((d) => d !== OPP[s.dir])).sort(byP);
  let a = ranked[0] ?? r.action, note = "";
  if (shieldOn && ranked.length) {
    // the model's order: legal moves first, then the rest (the shield may find that a "dangerous" move is the only way out)
    const rest = DIRS.filter((d) => d !== OPP[s.dir] && !ranked.includes(d)).sort(byP);
    const sh = shieldMove(match, s, [...ranked, ...rest]);
    if (sh.vetoed) { vetoes[s.id] += 1; note = `🛡 щит: ${ARROWS[a]} ведёт в ловушку → ${ARROWS[sh.move]}`; }
    else if (sh.doomed) note = "🛡 щит: спасения нет";
    a = sh.move;
  }
  if (panel) showBrain(r, a, r.engine || "Laya", note);
  return a;
}

// ---------------------------------------------------------------- step
async function doStep() {
  stepping = true;
  const waiting = [aiPending, leftPending].filter((j) => j && !j.done).map((j) => j.promise);
  // in-browser models set the pace: wait for their answer (a timeout would pile new GPU runs on unfinished ones)
  const wait = inBrowser() ? 4000 : AI_WAIT_MS;
  if (waiting.length) await Promise.race([Promise.all(waiting), new Promise((res) => setTimeout(res, wait))]);
  if (mode !== "play") { stepping = false; return; }
  let human = match.human.dir;
  const md = mouseDir();
  if (md && !inputQueue.length) human = md;
  while (inputQueue.length) {
    const d = inputQueue.shift();
    if (d !== OPP[match.human.dir] && d !== match.human.dir) { human = d; break; }
  }
  if (watch) human = match.human.alive ? chooseAiAction(leftPending, match.human, false) : match.human.dir;
  const ai = match.ai.alive ? chooseAiAction(aiPending) : match.ai.dir;
  renderer.snapshot(match);
  match.advance({ human, ai });
  const events = match.drainEvents();
  renderer.handleEvents(events);
  events.forEach(feedEvent);
  lastTick = performance.now();
  updateHud();
  if (match.over) finish();
  else requestAiMove();
  stepping = false;
}

function loop(now) {
  if (mode === "play" && !stepping && now - lastTick >= TICK_MS) doStep();
  const t = mode === "play" ? Math.min(1, (now - lastTick) / TICK_MS) : 1;
  renderer.draw(match, t, now);
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- HUD
function updateHud() {
  const h = match.human, a = match.ai;
  $("scoreHuman").textContent = h.score;
  $("scoreAi").textContent = a.score;
  $("metaHuman").textContent = `⚔ ${h.kills} · ☠ ${h.deaths}${h.alive ? "" : ` · ⟳ ${h.respawnIn}`}`;
  $("metaAi").textContent = `⚔ ${a.kills} · ☠ ${a.deaths}${a.alive ? "" : ` · ⟳ ${a.respawnIn}`}`;
  $("immHuman").firstElementChild.style.width = `${Math.min(100, (h.immortal / 50) * 100)}%`;
  $("immAi").firstElementChild.style.width = `${Math.min(100, (a.immortal / 50) * 100)}%`;
  const left = Math.max(0, MATCH_STEPS - match.step);
  const secs = Math.ceil((left * TICK_MS) / 1000);
  $("timer").textContent = `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
  $("timer").classList.toggle("low", secs <= 10);
  $("timebar").style.width = `${(left / MATCH_STEPS) * 100}%`;
}

const bars = {};
for (const d of DIRS) {
  const row = document.createElement("div");
  row.className = "bar";
  row.innerHTML = `<span>${ARROWS[d]} ${DIR_RU[d]}</span><div class="track"><div class="fill"></div></div><b>—</b>`;
  $("aiBars").append(row);
  bars[d] = row;
}
function showBrain(r, action, source, note = "") {
  $("aiArrow").textContent = ARROWS[action] || "·";
  $("aiMove").textContent = DIR_RU[action] || "—";
  for (const d of DIRS) {
    const p = r ? r.probabilities[d] : 0;
    bars[d].querySelector(".fill").style.width = `${p * 100}%`;
    bars[d].querySelector("b").textContent = r ? `${(p * 100).toFixed(1)}%` : "—";
    bars[d].classList.toggle("win", d === action);
  }
  $("aiConf").textContent = r ? `${(r.probabilities[action] * 100).toFixed(1)}%` : "—";
  $("aiLat").textContent = r ? `${r.latency_ms.toFixed(0)} мс модель · ${r.rtt.toFixed(0)} мс всего` : "—";
  $("aiSrc").textContent = source;
  if (note) $("aiShield").textContent = note;
  $("aiVetoes").textContent = shieldOn ? `${vetoes.ai} вето за матч` : "выключен";
}

function feed(text, cls) {
  const li = document.createElement("li");
  li.className = cls;
  li.textContent = text;
  $("feed").prepend(li);
  while ($("feed").children.length > 9) $("feed").lastChild.remove();
}
const VIEW_RU = { tilt: "Обзор арены", top: "Вид сверху", chase: "Камера за змейкой · ←/→ = поворот налево/направо" };
function cycleView() {
  if (!renderer.toggleView) return;
  const v = renderer.toggleView();
  toast(VIEW_RU[v]);
  try { localStorage.setItem("duelView", v); } catch {}
}

function feedEvent(e) {
  if (e.type === "fogWarning") { toast("Туман опускается через 10 ходов…"); return; }
  if (e.type === "fog") {
    feed(e.active ? "Туман: видна еда только рядом (5 клеток)" : "Туман рассеялся", "both");
    toast(e.active ? "🌫 Туман! Еду видно только рядом" : "Туман рассеялся");
    return;
  }
  if (e.type === "eat") {
    const what = { apple: "яблоко +1", golden: "золотое яблоко +3", immortal: "★ бессмертие +2" }[e.food];
    if (e.food !== "apple") feed(`${NAME[e.who]}: ${what}`, e.who);
  } else if (e.type === "death") {
    if (e.who === "human") buzz(120);
    feed(`${NAME[e.who]}: гибель ${REASON_RU[e.reason] || ""} (−3)`, e.who);
  }
  else if (e.type === "kill") { buzz([40, 40, 40]); feed(`${NAME[e.who]} +5 за то, что соперник врезался`, e.who); toast(e.who === "human" ? "Laya врезалась в вас: +5!" : "Вы врезались в Laya: ей +5"); }
  else if (e.type === "wall" && e.phase === "GHOST") feed("Стены стали прозрачными", "both");
}

function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.classList.add("show");
  clearTimeout(toast.t);
  toast.t = setTimeout(() => el.classList.remove("show"), 2200);
}

// ---------------------------------------------------------------- flow
function showCard(id) {
  renderer.setIdle?.(id === "menuCard" || id === "endCard");
  $("overlay").classList.toggle("hidden", id === null);
  for (const c of ["menuCard", "countCard", "pauseCard", "endCard"]) $(c).classList.toggle("hidden", c !== id);
}

// Phones: go fullscreen and lock landscape (Android/Chrome); iOS ignores this, the CSS hint covers it
async function goLandscape() {
  if (!matchMedia("(pointer: coarse)").matches) return;
  try {
    if (!document.fullscreenElement) await document.documentElement.requestFullscreen?.({ navigationUI: "hide" });
    await screen.orientation?.lock?.("landscape");
  } catch {}
}

function setBrain(b) {
  brain = b;
  document.querySelectorAll("#brains button").forEach((x) => x.classList.toggle("active", x.dataset.brain === b));
  try { localStorage.setItem("duelBrain", b); } catch {}
}

async function ensureBrain() {
  if (!inBrowser()) return true;
  // the browser models this match needs: watch mode = both, otherwise the chosen one
  const need = watch ? [["v10", onnx], has421 ? ["421M", onnx421] : ["v11w", onnxV11w]]
    : [brain === "onnx421" ? ["421M", onnx421] : brain === "onnx_v11w" ? ["v11w", onnxV11w] : ["v10", onnx]];
  const todo = need.filter(([, p]) => !p.available);
  if (!todo.length) return true;
  const btns = [$("startBtn"), $("watchBtn")], labels = btns.map((b) => b.textContent);
  btns.forEach((b) => (b.disabled = true));
  try {
    for (const [name, pol] of todo) {
      await pol.init((f) => btns.forEach((b) => (b.textContent = `ЗАГРУЗКА LAYA ${name} ${Math.round(f * 100)}%`)));
      btns.forEach((b) => (b.textContent = "ПРОГРЕВ…"));
    }
    toast(`Модели загружены в браузер (${todo.map(([n, p]) => `${n}: ${p.backend}`).join(", ")})`);
    return true;
  } catch (e) {
    console.error(e);
    toast("ONNX не запустился, играет сервер: " + e.message);
    setBrain("server");
    return false;
  } finally {
    btns.forEach((b, i) => { b.disabled = false; b.textContent = labels[i]; });
  }
}

function applyNames() {
  names = watch ? { human: "v10", ai: watchName() }
    : { human: "ВЫ", ai: usesLaya421() ? "LAYA 421M" : brain === "v11w" || brain === "onnx_v11w" ? "LAYA v11w" : brain === "v12" ? "LAYA v12" : "LAYA" };
  document.querySelector(".side.human .who").textContent = names.human;
  document.querySelector(".side.ai .who").textContent = names.ai;
  NAME.human = watch ? "v10" : "Вы";
  NAME.ai = { "LAYA": "Laya", "LAYA 421M": "Laya 421M", "LAYA v11w": "Laya v11w", "LAYA v12": "Laya v12" }[names.ai];
  renderer.setNames?.(names);
}

async function start() {
  goLandscape();
  applyNames();
  await ensureBrain();
  const name = $("playerName").value.trim();
  try { localStorage.setItem("duelName", name); localStorage.setItem("duelSpeed", speed); } catch {}
  TICK_MS = SPEEDS[speed];
  AI_WAIT_MS = Math.max(120, Math.min(250, TICK_MS));
  match = new VersusMatch();
  vetoes.ai = vetoes.human = 0;
  $("aiShield").textContent = "—";
  renderer.prev = null;
  inputQueue = [];
  $("feed").innerHTML = "";
  updateHud();
  mode = "countdown";
  showCard("countCard");
  let n = 3;
  $("count").textContent = n;
  requestAiMove();
  const iv = setInterval(() => {
    n -= 1;
    if (n > 0) { $("count").textContent = n; return; }
    clearInterval(iv);
    $("count").textContent = "GO";
    setTimeout(() => { mode = "play"; showCard(null); lastTick = performance.now(); }, 300);
  }, 650);
}

function togglePause() {
  if (mode === "play") { mode = "pause"; showCard("pauseCard"); }
  else if (mode === "pause") { mode = "play"; showCard(null); lastTick = performance.now(); }
}

let submitted = false;
function finish() {
  mode = "end";
  const h = match.human, a = match.ai;
  $("endTitle").textContent = watch
    ? (h.score > a.score ? "v10 ПОБЕДИЛА" : h.score < a.score ? `${watchName()} ПОБЕДИЛА` : "НИЧЬЯ")
    : (h.score > a.score ? "ПОБЕДА!" : h.score < a.score ? names.ai + " ПОБЕДИЛА" : "НИЧЬЯ");
  $("finalHuman").textContent = h.score;
  $("finalAi").textContent = a.score;
  $("endStats").innerHTML = `
    <span class="c-human">${NAME.human}: яблок ${h.apples}</span><span class="c-ai">${NAME.ai}: яблок ${a.apples}</span>
    <span class="c-human">убийств ${h.kills}, смертей ${h.deaths}</span><span class="c-ai">убийств ${a.kills}, смертей ${a.deaths}</span>
    ${shieldOn ? `<span class="c-human">${watch ? `щит: ${vetoes.human} вето` : ""}</span><span class="c-ai">щит: ${vetoes.ai} вето</span>` : ""}`;
  submitted = false;
  $("submitRow").hidden = watch; // AI-vs-AI results do not go to the human leaderboard
  showCard("endCard");
  loadLeaders();
}

async function loadLeaders(highlight) {
  try {
    $("leadersTitle").textContent = `Рекорды · ${SPEED_RU[speed]}`;
    const r = await fetch(`api/leaderboard?speed=${speed}`);
    const rows = await r.json();
    $("leaders").innerHTML = rows.length ? rows.map((x) =>
      `<li class="${x.id === highlight ? "me" : ""}">${escapeHtml(x.name)} — ${x.human}:${x.ai} ${x.human > x.ai ? "🏆" : ""}</li>`).join("")
      : "<li>пока пусто — будьте первым</li>";
  } catch {
    $("leaders").innerHTML = "<li>таблица рекордов недоступна</li>";
  }
}

async function submit() {
  if (submitted) return;
  submitted = true;
  const name = $("playerName").value.trim() || "Игрок";
  try {
    const r = await fetch("api/leaderboard", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, speed, human: match.human.score, ai: match.ai.score, kills: match.human.kills, deaths: match.human.deaths, seed: match.seed }),
    });
    const d = await r.json();
    $("submitRow").hidden = true;
    loadLeaders(d.id);
  } catch {
    submitted = false;
    toast("Не удалось сохранить результат");
  }
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------------------------------------------------------------- input
const KEYS = { ArrowUp: "UP", KeyW: "UP", ArrowDown: "DOWN", KeyS: "DOWN", ArrowLeft: "LEFT", KeyA: "LEFT", ArrowRight: "RIGHT", KeyD: "RIGHT" };
const TURN_LEFT = { UP: "LEFT", LEFT: "DOWN", DOWN: "RIGHT", RIGHT: "UP" };
const TURN_RIGHT = { UP: "RIGHT", RIGHT: "DOWN", DOWN: "LEFT", LEFT: "UP" };
function queueDir(d) {
  if (mode !== "play" && mode !== "countdown") return;
  const last = inputQueue[inputQueue.length - 1] ?? match.human.dir;
  // chase camera turns with the snake, so input is relative: left/right = turn, up/down = nothing
  if (renderer.view === "chase") {
    if (d === "LEFT") d = TURN_LEFT[last];
    else if (d === "RIGHT") d = TURN_RIGHT[last];
    else return;
  }
  if (d === last || d === OPP[last]) return;
  if (inputQueue.length < 3) inputQueue.push(d);
}
// Mouse steering: the snake heads for the cell under the cursor (a ring marks it).
// Arrow keys switch it off until the mouse moves again. Not used in the chase camera (relative controls).
let mouseTarget = null, mouseOn = false;
function mouseDir() {
  if (!mouseOn || !mouseTarget || !match.human.alive || renderer.view === "chase") return null;
  const h = match.human.head, dx = mouseTarget.x - h.x, dy = mouseTarget.y - h.y;
  const horiz = dx > 0 ? "RIGHT" : dx < 0 ? "LEFT" : null, vert = dy > 0 ? "DOWN" : dy < 0 ? "UP" : null;
  const cur = match.human.dir;
  if (cur === horiz || cur === vert) return cur; // still closing in on this axis: no zig-zag, L-shaped path
  const order = Math.abs(dx) >= Math.abs(dy) ? [horiz, vert] : [vert, horiz];
  return order.find((d) => d && d !== OPP[cur]) ?? null; // on the target or behind: keep going
}
$("board").parentElement.addEventListener("pointermove", (e) => {
  if (e.pointerType !== "mouse") return;
  mouseTarget = renderer.cellAt?.(e.clientX, e.clientY) ?? null;
  mouseOn = !!mouseTarget;
  renderer.setTarget?.(mouseOn && renderer.view !== "chase" ? mouseTarget : null);
});
$("board").parentElement.addEventListener("pointerleave", () => { mouseOn = false; renderer.setTarget?.(null); });

addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") { if (e.key === "Enter" && mode === "menu") start(); return; }
  if (KEYS[e.code]) { e.preventDefault(); mouseOn = false; renderer.setTarget?.(null); queueDir(KEYS[e.code]); }
  else if (e.code === "Space") { e.preventDefault(); togglePause(); }
  else if (e.code === "KeyV") cycleView();
  else if (e.code === "Enter" && (mode === "menu" || mode === "end")) start();
});
// Swipe anywhere on the screen. A turn fires as soon as the finger has moved ~18 px, and the origin
// resets, so one continuous zig-zag gives several turns without lifting the finger.
const SWIPE_PX = 18;
let touch = null;
const inGame = () => mode === "play" || mode === "countdown";
document.addEventListener("touchstart", (e) => {
  if (!inGame() || e.target.closest?.("#pad, button, input")) return;
  const t = e.touches[0];
  touch = { x: t.clientX, y: t.clientY };
}, { passive: true });
document.addEventListener("touchmove", (e) => {
  if (!touch || !inGame()) return;
  e.preventDefault(); // no page scroll / pull-to-refresh while steering
  const t = e.touches[0], dx = t.clientX - touch.x, dy = t.clientY - touch.y;
  if (Math.max(Math.abs(dx), Math.abs(dy)) < SWIPE_PX) return;
  queueDir(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? "RIGHT" : "LEFT") : (dy > 0 ? "DOWN" : "UP"));
  touch = { x: t.clientX, y: t.clientY };
}, { passive: false });
document.addEventListener("touchend", () => { touch = null; });
const buzz = (ms) => { try { navigator.vibrate?.(ms); } catch {} };document.querySelectorAll("#pad button").forEach((b) => b.addEventListener("pointerdown", (e) => { e.preventDefault(); queueDir(b.dataset.dir); }));
document.addEventListener("visibilitychange", () => { if (document.hidden && mode === "play") togglePause(); });
$("startBtn").addEventListener("click", () => { watch = false; start(); });
$("watchBtn").addEventListener("click", () => { watch = true; start(); });
$("againBtn").addEventListener("click", start); // keeps the current mode (play or watch)
$("submitBtn").addEventListener("click", submit);
function setSpeed(s) {
  if (!SPEEDS[s]) return;
  speed = s;
  document.querySelectorAll("#speeds button").forEach((b) => b.classList.toggle("active", b.dataset.speed === s));
  if (mode === "menu") { TICK_MS = SPEEDS[s]; updateHud(); }
}
document.querySelectorAll("#speeds button").forEach((b) => b.addEventListener("click", () => setSpeed(b.dataset.speed)));
document.querySelectorAll("#brains button").forEach((b) => b.addEventListener("click", () => setBrain(b.dataset.brain)));
function setShield(on) {
  shieldOn = on;
  document.querySelectorAll("#shields button").forEach((b) => b.classList.toggle("active", (b.dataset.shield === "on") === on));
  $("aiVetoes").textContent = on ? "включён" : "выключен";
  try { localStorage.setItem("duelShield", on ? "on" : "off"); } catch {}
}
document.querySelectorAll("#shields button").forEach((b) => b.addEventListener("click", () => setShield(b.dataset.shield === "on")));
try { setShield(localStorage.getItem("duelShield") !== "off"); } catch { setShield(true); }
// default: what the player chose last time; without a working model server -> the browser model
(async () => {
  let saved = null;
  try { saved = localStorage.getItem("duelBrain"); } catch {}
  let serverOk = false;
  try { serverOk = (await (await fetch("api/health")).json()).model_ready; } catch {}
  // without a server the best browser model plays: v11w (+ the shield)
  const browserOnly = ["onnx", "onnx421", "onnx_v11w"];
  setBrain(!serverOk ? (browserOnly.includes(saved) ? saved : "onnx_v11w") : saved || "server");
  // window models are offered only when the model server has them loaded
  let have = [];
  try { have = (await (await fetch("api/health")).json()).window_models || []; } catch {}
  for (const [b, m] of Object.entries(WINDOW_MODELS)) {
    if (b.startsWith("onnx")) continue;
    const btn = document.querySelector(`#brains [data-brain="${b}"]`);
    if (btn && !have.includes(m)) { btn.disabled = true; btn.title = "модель ещё не загружена на сервер"; }
  }
  // server brains are useless without a server (static hosting, e.g. GitHub Pages)
  if (!serverOk) for (const b of ["server", "convai", "v11w", "v12"]) {
    const btn = document.querySelector(`#brains [data-brain="${b}"]`);
    if (btn) { btn.disabled = true; btn.title = "нужен сервер модели"; }
  }
  // Laya 421M in the browser only if its files are hosted next to the page
  try { has421 = (await fetch("model_laya421/state_schema.json", { method: "HEAD" })).ok; } catch { has421 = false; }
  if (!has421) {
    const b = document.querySelector('#brains [data-brain="onnx421"]');
    if (b) { b.disabled = true; b.title = "файлы Laya 421M не выложены рядом с игрой"; }
    if (brain === "onnx421") setBrain("onnx_v11w");
    if (!serverOk) $("watchBtn").textContent = "СМОТРЕТЬ: v10 ПРОТИВ v11w";
  }
  if (!serverOk) toast("Сервер модели недоступен — Laya будет играть в браузере");
})();
try {
  $("playerName").value = localStorage.getItem("duelName") || "";
  setSpeed(localStorage.getItem("duelSpeed") || "normal");
} catch {}

updateHud();
$("camBtn").addEventListener("click", cycleView);
$("portraitBtn").addEventListener("click", () => document.body.classList.add("portrait-ok"));
$("viewBtn").addEventListener("click", () => {
  const prev = renderer.prev;
  useRenderer($("viewBtn").dataset.kind === "3d" ? "2d" : "3d");
  renderer.prev = prev;
});
showCard("menuCard");
requestAnimationFrame(loop);
window.__duel = { get match() { return match; }, get renderer() { return renderer; }, mouseDir, get queue() { return inputQueue; } }; // debugging
