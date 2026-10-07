// game.js — чистая игровая логика Snake (без DOM и без рендеринга).
// Правила: RULES_V9.md. Эталонная реализация: build_dataset_v9.py (классы Drone и World,
// World.world_step / World.model_state). Порядок событий внутри шага и логика отскока дронов
// повторяют эталон один в один — модель обучена на нём.
// Модуль можно переиспользовать: в браузере, в Node (tools/parity-тест), в будущем ONNX-демо.

/** true = дрон, наехавший на ЛЮБОЙ сегмент тела, убивает змею (RULES_V9 §1). */
export const PATROL_BODY_HITS = true;

export const GRID = 20;
export const DIRS = ["UP", "DOWN", "LEFT", "RIGHT"];
export const ACTIONS = { UP: { x: 0, y: -1 }, DOWN: { x: 0, y: 1 }, LEFT: { x: -1, y: 0 }, RIGHT: { x: 1, y: 0 } };
export const OPPOSITES = { UP: "DOWN", DOWN: "UP", LEFT: "RIGHT", RIGHT: "LEFT" };

// ---- константы V9 (как в build_dataset_v9.py) ----
export const IMMORTAL_STEPS = 50;
export const IMMORTAL_APPLE_TTL = 60;
export const IMMORTAL_APPLE_CHANCE_UI = 0.08; // в обучении 0.15
export const BLINK_STEPS = 3;
export const MAX_BODY_CELLS = 30;
export const DRONE_PATH_STEPS = 6;
export const STATE_KEYS_V9 = ["current_dir", "recent_actions", "food_dir", "danger_UP", "danger_DOWN", "danger_LEFT",
  "danger_RIGHT", "head_pos", "body", "food_pos", "snake_len", "immortal_steps", "obstacles",
  "phase_walls", "portals", "fog_of_war", "sight_radius", "patrols", "drone_path"];

// Скорость отрисовки (только темп показа; правила считаются в шагах)
export const BASE_TICK_MS = 120;
export const TURBO_TICK_MS = 60;
export const FREEZE_TICK_MS = 220;
export const SPEED_EFFECT_STEPS = 42; // ≈ 5000 мс старого setTimeout при 120 мс/шаг

// value: очки, ttl: жизнь в ШАГАХ (null = вечная). Старые мс пересчитаны при 120 мс/шаг.
export const FOOD_TYPES = {
  standard: { value: 1, ttl: null },
  golden: { value: 3, ttl: null },
  shrink: { value: 2, ttl: 67 },
  turbo: { value: 1, ttl: 50 },
  freeze: { value: 1, ttl: 50 },
  poison: { value: -5, ttl: 50 },
  immortal: { value: 2, ttl: IMMORTAL_APPLE_TTL },
};

export const SHAPE_TEMPLATES = [
  [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 2, y: 0 }], // горизонтальная 3
  [{ x: 0, y: 0 }, { x: 0, y: 1 }, { x: 0, y: 2 }], // вертикальная 3
  [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 1 }], // квадрат 2x2
  [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 2, y: 0 }, { x: 0, y: 1 }], // L
  [{ x: 0, y: 0 }, { x: 1, y: 0 }], // горизонтальная 2
  [{ x: 0, y: 0 }, { x: 0, y: 1 }], // вертикальная 2
];

export const CRASH_REASONS = {
  LOOP: "Infinite Loop / Corner Deadlock",
  PATROL: "Patrol Drone Collision",
  POISON: "Ate Poison Apple",
  WALL: "Hit Internal Obstacle Wall",
  BOUNDARY: "Boundary/Body Collision",
};

const k = (x, y) => `${x},${y}`;
const manhattan = (a, b) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
const mod = (a, n) => ((a % n) + n) % n;
const sgn = (v) => (v > 0 ? 1 : -1);

// =====================================================================
// Дроны — порт класса Drone (build_dataset_v9.py). kind: "h" | "v" | "circuit".
// h/v: {kind, x, y, dx, dy, lo, hi}; circuit: {kind, x, y, dx, dy, waypoints:[[x,y]..], wpIdx, wpSign}
// =====================================================================
function inRoute(p, x, y) {
  if (p.kind === "h") return y === p.y && p.lo <= x && x <= p.hi;
  if (p.kind === "v") return x === p.x && p.lo <= y && y <= p.hi;
  return x >= 0 && x < GRID && y >= 0 && y < GRID;
}
function circuitTarget(p, sign) {
  const n = p.waypoints.length;
  return sign > 0 ? p.waypoints[mod(p.wpIdx + 1, n)] : p.waypoints[mod(p.wpIdx, n)];
}
function circuitStep(p, sign) {
  const [tx, ty] = circuitTarget(p, sign);
  if (p.x === tx && p.y === ty) return [0, 0];
  if (p.x !== tx) return [sgn(tx - p.x), 0];
  return [0, sgn(ty - p.y)];
}

/** (dx, dy) следующего хода дрона при данном множестве клеток твёрдых стен (Set "x,y"). */
export function dronePlan(p, solid) {
  if (p.kind === "h" || p.kind === "v") {
    for (const [dx, dy] of [[p.dx, p.dy], [-p.dx || 0, -p.dy || 0]]) {
      const cx = p.x + dx, cy = p.y + dy;
      if (inRoute(p, cx, cy) && !solid.has(k(cx, cy))) return [dx, dy];
    }
    return [0, 0];
  }
  for (const sign of [p.wpSign, -p.wpSign]) {
    const d = circuitStep(p, sign);
    if ((d[0] || d[1]) && !solid.has(k(p.x + d[0], p.y + d[1]))) return d;
  }
  return [0, 0];
}

/** Один шаг дрона (мутирует p) — Drone.step. */
export function droneStep(p, solid) {
  const d = dronePlan(p, solid);
  if (p.kind === "h" || p.kind === "v") {
    if (d[0] || d[1]) { p.dx = d[0]; p.dy = d[1]; }
  } else if ((d[0] || d[1])) {
    const fwd = circuitStep(p, p.wpSign);
    if (d[0] !== fwd[0] || d[1] !== fwd[1]) p.wpSign = -p.wpSign;
  }
  p.x += d[0];
  p.y += d[1];
  if (p.kind === "circuit") {
    const n = p.waypoints.length;
    const nxt = p.waypoints[mod(p.wpIdx + 1, n)], cur = p.waypoints[mod(p.wpIdx, n)];
    if (p.wpSign > 0 && p.x === nxt[0] && p.y === nxt[1]) p.wpIdx = mod(p.wpIdx + 1, n);
    else if (p.wpSign < 0 && p.x === cur[0] && p.y === cur[1]) p.wpIdx = mod(p.wpIdx - 1, n);
  }
  return p;
}

export const cloneDrone = (p) => ({ ...p });

// =====================================================================
export class SnakeGame {
  /**
   * @param {object} [opts]
   * @param {() => number} [opts.rng] источник случайности (по умолчанию Math.random)
   * @param {object} [opts.settings] {walls, portals, fog, patrols, fogRadius, loopDetector}
   */
  constructor(opts = {}) {
    this.rng = opts.rng || Math.random;
    this.settings = { walls: true, portals: true, fog: false, patrols: true, fogRadius: 5, loopDetector: true, ...(opts.settings || {}) };
    this.score = 0;
    this.maxScore = 0;
    this.crashes = 0;
    this.epoch = 0;
    this.clock = 0; // мс реального времени симуляции — только для анимаций
    this._events = [];
    if (!opts.noReset) this.reset();
  }

  // ---------- события для рендера/HUD ----------
  emit(type, data = {}) { this._events.push({ type, ...data }); }
  drainEvents() { const e = this._events; this._events = []; return e; }

  // ---------- RNG-хуки (переопределяются в parity-тесте) ----------
  _randInt(n) { return Math.floor(this.rng() * n); }
  /** Целое в [lo, hi] включительно — как random.randint. */
  _randRange(lo, hi) { return lo + this._randInt(hi - lo + 1); }
  _pickCell(cells) { return cells[this._randInt(cells.length)]; }

  // ---------- жизненный цикл ----------
  _resetCommon() {
    this.epoch++;
    this.over = false;
    this.crashedCell = null;
    this.crashReason = null;
    this.crashedAction = null;
    this.recentHeadPositions = [];
    this.cellLastVisit = {};
    this.score = 0;
    this.stepCount = 0;
    this.lastEatStep = 0;
    this.immortalSteps = 0;
    this.foods = [];
    this.speedEffect = null; // {type:'turbo'|'freeze', steps}
  }

  reset() {
    this._resetCommon();
    this.recentActions = ["RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"];
    this.currentDir = "RIGHT";
    this.snake = [{ x: 10, y: 10 }, { x: 9, y: 10 }, { x: 8, y: 10 }];
    this.generateObstacles();
    this.generatePortals();
    this.generatePatrols();
    this.spawnFood();
    this._snapshotPrev();
    this.emit("reset");
  }

  /**
   * Загрузить произвольное состояние мира (для тестов паритета и сценариев).
   * structures: [{cells:[[x,y]..], phase, timer}], drones: в формате дронов выше, foods: [{x,y,type,ttl}].
   */
  loadWorld(w) {
    this._resetCommon();
    this.snake = w.snake.map(([x, y]) => ({ x, y }));
    this.currentDir = w.direction;
    this.recentActions = [...w.recentActions];
    this.structures = w.structures.map((s, i) => ({ id: i, cells: s.cells.map(([x, y]) => ({ x, y })), phase: s.phase, timer: s.timer }));
    this._syncWalls();
    this.portals = (w.portals || []).map(([x, y]) => ({ x, y }));
    this.patrols = (w.drones || []).map((d) => ({ ...d }));
    this.settings.portals = this.portals.length >= 2;
    this.settings.patrols = true;
    this.settings.fog = !!w.fog;
    if (w.sightRadius != null) this.settings.fogRadius = w.sightRadius;
    this.immortalSteps = w.immortalSteps || 0;
    for (const f of w.foods || []) this._addFood({ x: f.x, y: f.y }, f.type, f.ttl);
    this._events = [];
    this._snapshotPrev();
    this.emit("reset");
  }

  _snapshotPrev() {
    this.prevSnake = this.snake.map((s) => ({ ...s }));
    this.prevPatrols = this.patrols.map((p) => ({ x: p.x, y: p.y }));
  }

  setSetting(name, value) {
    this.settings[name] = value;
    // как в старом UI: стены/порталы/патрули пересоздают игру, туман — нет
    if (name === "walls" || name === "portals" || name === "patrols") this.reset();
    else this.emit("settings");
  }

  get tickInterval() {
    if (!this.speedEffect) return BASE_TICK_MS;
    return this.speedEffect.type === "turbo" ? TURBO_TICK_MS : FREEZE_TICK_MS;
  }

  // ---------- генерация уровня ----------
  generateObstacles() {
    this.structures = [];
    if (this.settings.walls) {
      const numStructures = 2 + this._randInt(2);
      const head = this.snake[0];
      const forbidden = new Set(this.snake.map((s) => k(s.x, s.y)));
      for (let dx = -3; dx <= 3; dx++) for (let dy = -3; dy <= 3; dy++) forbidden.add(k(head.x + dx, head.y + dy));
      // стартовые клетки дронов UI (как new_world отбрасывает миры, где дрон внутри стены)
      if (this.settings.patrols) { forbidden.add(k(4, 5)); forbidden.add(k(15, 12)); }
      const all = [];
      for (let i = 0; i < numStructures; i++) {
        for (let attempts = 0; attempts < 100; attempts++) {
          const tmpl = SHAPE_TEMPLATES[this._randInt(SHAPE_TEMPLATES.length)];
          const ox = 1 + this._randInt(GRID - 4);
          const oy = 1 + this._randInt(GRID - 4);
          const cells = tmpl.map((p) => ({ x: ox + p.x, y: oy + p.y }));
          if (cells.every((c) => c.x >= 0 && c.x < GRID && c.y >= 0 && c.y < GRID && !forbidden.has(k(c.x, c.y)))) {
            cells.forEach((c) => { all.push(c); forbidden.add(k(c.x, c.y)); });
            break;
          }
        }
      }
      // Как World.setup: структура = компонента 4-связности; стартовый таймер randint(15, 80)
      const left = new Map(all.map((c) => [k(c.x, c.y), c]));
      for (const c0 of all) {
        if (!left.has(k(c0.x, c0.y))) continue;
        left.delete(k(c0.x, c0.y));
        const comp = [c0], q = [c0];
        while (q.length) {
          const c = q.shift();
          for (const d of DIRS) {
            const key = k(c.x + ACTIONS[d].x, c.y + ACTIONS[d].y);
            if (left.has(key)) { const n = left.get(key); left.delete(key); comp.push(n); q.push(n); }
          }
        }
        this.structures.push({ id: this.structures.length, cells: comp, phase: "SOLID", timer: this._randRange(15, 80) });
      }
    }
    this._syncWalls();
  }

  /** obstacles = клетки SOLID и BLINKING (смертельны), solid = то же множеством. */
  _syncWalls() {
    this.obstacles = [];
    for (const s of this.structures) if (s.phase !== "GHOST") this.obstacles.push(...s.cells);
    this.solid = new Set(this.obstacles.map((c) => k(c.x, c.y)));
    this._allWallCells = new Set(this.structures.flatMap((s) => s.cells.map((c) => k(c.x, c.y))));
  }

  generatePortals() {
    this.portals = [];
    if (!this.settings.portals) return;
    const forbidden = new Set(this.snake.map((s) => k(s.x, s.y)));
    this._allWallCells.forEach((key) => forbidden.add(key));
    for (let attempts = 0; attempts < 200; attempts++) {
      const pa = { x: 1 + this._randInt(GRID - 2), y: 1 + this._randInt(GRID - 2) };
      const pb = { x: 1 + this._randInt(GRID - 2), y: 1 + this._randInt(GRID - 2) };
      if (!forbidden.has(k(pa.x, pa.y)) && !forbidden.has(k(pb.x, pb.y)) && manhattan(pa, pb) >= 6) {
        this.portals = [pa, pb];
        break;
      }
    }
  }

  /** Дроны веб-UI — ui_drones() из build_dataset_v9.py. */
  generatePatrols() {
    this.patrols = [];
    if (!this.settings.patrols) return;
    this.patrols = [
      { kind: "h", x: 4, y: 5, dx: 1, dy: 0, lo: 3, hi: 16 },
      { kind: "v", x: 15, y: 12, dx: 0, dy: 1, lo: 8, hi: 17 },
    ];
  }

  /** Активные дроны (пустой список, если патрули выключены). */
  get drones() { return this.settings.patrols ? this.patrols : []; }

  /** Планы дронов [dx, dy] на следующий шаг при текущих стенах. */
  dronePlans() { return this.drones.map((p) => dronePlan(p, this.solid)); }

  /** Будущие клетки каждого дрона (drone_path), с отскоками от текущих стен. */
  dronePaths(steps = DRONE_PATH_STEPS) {
    return this.drones.map((p) => {
      const g = cloneDrone(p), path = [];
      for (let i = 0; i < steps; i++) { droneStep(g, this.solid); path.push([g.x, g.y]); }
      return path;
    });
  }

  /** drone_danger_cells: текущая и следующая клетка каждого дрона. */
  droneDangerCells() {
    // кэш на текущий шаг: BFS спрашивает это сотни раз за ход
    const key = `${this.epoch}:${this.stepCount}:${this.settings.patrols}`;
    if (this._ddcKey === key) return this._ddc;
    const cells = (this._ddc = new Set());
    this._ddcKey = key;
    for (const p of this.drones) {
      const [dx, dy] = dronePlan(p, this.solid);
      cells.add(k(p.x, p.y));
      cells.add(k(p.x + dx, p.y + dy));
    }
    return cells;
  }

  // ---------- еда ----------
  _rollFoodType() {
    let r = this.rng();
    if (r < IMMORTAL_APPLE_CHANCE_UI) return "immortal";
    r = (r - IMMORTAL_APPLE_CHANCE_UI) / (1 - IMMORTAL_APPLE_CHANCE_UI); // остальное — как в старом UI
    if (r < 0.08) return "shrink";
    if (r < 0.16) return "turbo";
    if (r < 0.24) return "freeze";
    if (r < 0.45) return "golden";
    if (r < 0.6) return "poison";
    return "standard";
  }

  /** Занята ли клетка для спавна: тело, еда, ЛЮБАЯ клетка стены (в т.ч. призрачной), порталы. */
  _occupied(x, y) {
    return this.snake.some((s) => s.x === x && s.y === y) ||
      this.foods.some((f) => f.x === x && f.y === y) ||
      this._allWallCells.has(k(x, y)) ||
      this.portals.some((p) => p.x === x && p.y === y);
  }

  _addFood(cell, fType, ttl) {
    const spec = FOOD_TYPES[fType];
    const t = ttl !== undefined ? ttl : spec.ttl;
    const food = {
      id: (this._foodId = (this._foodId || 0) + 1),
      x: cell.x, y: cell.y, type: fType, value: spec.value,
      ttl: t, maxTtl: spec.ttl || t, bornStep: this.stepCount,
    };
    this.foods.push(food);
    this.emit("spawn", { food });
    return food;
  }

  /** Спавн еды. pos — конкретная клетка (клик пользователя), type — принудительный тип. */
  spawnFood(pos = null, type = null) {
    const fType = type || this._rollFoodType();
    let cell = null;
    if (pos) {
      if (pos.x < 0 || pos.x >= GRID || pos.y < 0 || pos.y >= GRID) return null;
      if (this._occupied(pos.x, pos.y)) return null;
      cell = pos;
    } else {
      const empty = [];
      for (let x = 0; x < GRID; x++) for (let y = 0; y < GRID; y++) if (!this._occupied(x, y)) empty.push({ x, y });
      if (!empty.length) return null;
      cell = this._pickCell(empty);
    }
    return this._addFood(cell, fType);
  }

  _ensureGoodFood() {
    if (!this.foods.some((f) => f.type !== "poison")) this.spawnFood();
  }

  /** Продвинуть «настенные» часы (мс) — только для анимаций; правила живут в шагах. */
  update(dtMs) {
    this.clock += dtMs;
    this._ensureGoodFood();
  }

  // ---------- сенсоры ----------
  isFoodVisible(f) {
    if (!this.settings.fog) return true;
    return manhattan(f, this.snake[0]) <= this.settings.fogRadius;
  }

  getNearestGoodFood() {
    const head = this.snake[0];
    const good = this.foods.filter((f) => f.type !== "poison" && this.isFoodVisible(f));
    if (!good.length) return { x: -1, y: -1, type: "none" };
    let best = good[0], bestD = Infinity;
    for (const f of good) { const d = manhattan(f, head); if (d < bestD) { bestD = d; best = f; } }
    return best;
  }

  warp(pos) {
    if (this.settings.portals && this.portals.length >= 2) {
      const [a, b] = this.portals;
      if (pos.x === a.x && pos.y === a.y) return { x: b.x, y: b.y };
      if (pos.x === b.x && pos.y === b.y) return { x: a.x, y: a.y };
    }
    return pos;
  }

  _onBody(t) { return this.snake.some((s) => s.x === t.x && s.y === t.y); }
  _poisonAt(t) { return this.foods.some((f) => f.x === t.x && f.y === t.y && f.type === "poison"); }
  _oob(t) { return t.x < 0 || t.x >= GRID || t.y < 0 || t.y >= GRID; }

  /**
   * danger_* как в World.model_state: check_danger (портал → граница / твёрдая стена / тело, если immortal_steps <= 0)
   * + яд (только в UI, тоже при immortal_steps <= 0) + клетки дронов (текущая и следующая, по сырой соседней
   * клетке, без портала), если immortal_steps <= 1.
   */
  dangers() {
    const h = this.snake[0];
    const st = this.immortalSteps;
    const droneCells = st <= 1 ? this.droneDangerCells() : null;
    const out = {};
    for (const d of DIRS) {
      const n = { x: h.x + ACTIONS[d].x, y: h.y + ACTIONS[d].y };
      const t = this.warp(n);
      let danger = this._oob(t) || this.solid.has(k(t.x, t.y)) || (st <= 0 && (this._onBody(t) || this._poisonAt(t)));
      if (droneCells && droneCells.has(k(n.x, n.y))) danger = true;
      out[d] = danger;
    }
    return out;
  }

  /**
   * Убьёт ли ход прямо сейчас (для маски безопасности и BFS). Точнее, чем danger_*: учитывает портал для дронов
   * и реальную неуязвимость (immortal_steps > 0 = иммунитет на весь следующий шаг).
   */
  isDeadlyMove(dir) {
    if (dir === OPPOSITES[this.currentDir] && this.snake.length > 1) return true;
    const h = this.snake[0];
    return this.isDeadlyCell(this.warp({ x: h.x + ACTIONS[dir].x, y: h.y + ACTIONS[dir].y }));
  }

  /** Клетка, куда голова попадёт в следующем шаге (уже после портала). */
  isDeadlyCell(t) {
    if (this._oob(t) || this.solid.has(k(t.x, t.y))) return true;
    if (this.immortalSteps > 0) return false;
    if (this._onBody(t) || this._poisonAt(t)) return true;
    return this.droneDangerCells().has(k(t.x, t.y));
  }

  /** Совместимость со старым API (BFS): опасна ли клетка соседа (pos — до портала). */
  isHazard(pos) { return this.isDeadlyCell(this.warp(pos)); }

  _foodDir(food, head, vocab) {
    if (food.x === -1) return vocab.none;
    const vert = food.y < head.y ? "UP" : food.y > head.y ? "DOWN" : "";
    const horiz = food.x < head.x ? "LEFT" : food.x > head.x ? "RIGHT" : "";
    return vert && horiz ? `${vert}_${horiz}` : vert || horiz || vocab.same;
  }

  /** phase_walls: GHOST → [x, y, steps_left], BLINKING → [x, y, -steps_left]. */
  phaseWalls() {
    const out = [];
    for (const s of this.structures) {
      if (s.phase === "GHOST") s.cells.forEach((c) => out.push([c.x, c.y, s.timer]));
      else if (s.phase === "BLINKING") s.cells.forEach((c) => out.push([c.x, c.y, -s.timer]));
    }
    return out;
  }

  /**
   * Состояние для модели (порядок ключей важен — Laya читает json.dumps(state) как текст).
   * "v9" (по умолчанию) — World.model_state из build_dataset_v9.py, ключи STATE_KEYS_V9.
   * "v7" — v7/v8-совместимый набор (сервер сам урежет под state_schema.json модели).
   * "legacy" — то, что отправлял старый UI.
   */
  getModelState(schema = "v9") {
    const head = this.snake[0];
    const food = this.getNearestGoodFood();
    const d = this.dangers();
    const obstacles = this.obstacles.map((o) => [o.x, o.y]);
    const portals = this.settings.portals ? this.portals.map((p) => [p.x, p.y]) : [];
    const plans = this.dronePlans();
    const patrols = this.drones.map((p, i) => [p.x, p.y, plans[i][0], plans[i][1]]);
    if (schema === "legacy") {
      return {
        danger_UP: d.UP, danger_DOWN: d.DOWN, danger_LEFT: d.LEFT, danger_RIGHT: d.RIGHT,
        current_dir: this.currentDir,
        food_dir: this._foodDir(food, head, { none: "NONE", same: "ON_FOOD" }),
        food_type: food.type || "standard",
        recent_actions: [...this.recentActions],
        head_pos: [head.x, head.y],
        food_pos: [food.x, food.y],
        snake_len: this.snake.length,
        obstacles, portals, patrols: patrols.map((p) => p.slice(0, 2)),
        fog_of_war: this.settings.fog,
        sight_radius: this.settings.fogRadius,
      };
    }
    const base = {
      current_dir: this.currentDir,
      recent_actions: [...this.recentActions],
      food_dir: this._foodDir(food, head, { none: "UNKNOWN", same: "SAME" }),
      danger_UP: d.UP, danger_DOWN: d.DOWN, danger_LEFT: d.LEFT, danger_RIGHT: d.RIGHT,
      head_pos: [head.x, head.y],
    };
    if (schema === "v7") {
      return { ...base, food_pos: [food.x, food.y], snake_len: this.snake.length, obstacles, portals,
        fog_of_war: this.settings.fog, sight_radius: this.settings.fogRadius, patrols };
    }
    return {
      ...base,
      body: this.snake.slice(1, MAX_BODY_CELLS + 1).map((c) => [c.x, c.y]),
      food_pos: [food.x, food.y],
      snake_len: this.snake.length,
      immortal_steps: this.immortalSteps,
      obstacles,
      phase_walls: this.phaseWalls(),
      portals,
      fog_of_war: this.settings.fog,
      sight_radius: this.settings.fogRadius,
      patrols,
      drone_path: this.dronePaths(),
    };
  }

  // ---------- шаг симуляции (порядок = World.world_step, RULES_V9 §4) ----------
  _crash(cell, reason, action, extra = {}) {
    this.over = true;
    this.crashedCell = cell;
    this.crashReason = reason;
    this.crashedAction = action;
    this.crashes++;
    this.emit("crash", { cell, reason, action, ...extra });
    return { crashed: true, reason, ...extra };
  }

  /** Применить действие. Возвращает {crashed, reason?, ate?, teleported}. */
  step(action) {
    if (this.over) return { crashed: true, reason: this.crashReason };
    this._snapshotPrev();
    const immortal = this.immortalSteps > 0;
    const drones = this.drones;

    if (ACTIONS[action] && (action !== OPPOSITES[this.currentDir] || this.snake.length === 1)) this.currentDir = action;
    this.recentActions.push(this.currentDir);
    if (this.recentActions.length > 5) this.recentActions.shift();

    this.stepCount++;
    const delta = ACTIONS[this.currentDir];
    const oldHead = { x: this.snake[0].x, y: this.snake[0].y };
    const entry = { x: oldHead.x + delta.x, y: oldHead.y + delta.y };
    const newHead = this.warp(entry);
    const teleported = newHead !== entry;

    // Детектор зацикливания — только UI (в правилах V9 его нет), можно выключить settings.loopDetector
    this.recentHeadPositions.push(k(newHead.x, newHead.y));
    if (this.recentHeadPositions.length > 16) this.recentHeadPositions.shift();
    let isLooping = false;
    if (this.settings.loopDetector) {
      const counts = {};
      for (const s of this.recentHeadPositions) { counts[s] = (counts[s] || 0) + 1; if (counts[s] >= 4) { isLooping = true; break; } }
    }

    // 1. Голова: дрон в целевой клетке, граница, твёрдая стена, тело, яд
    if (!immortal && drones.some((p) => p.x === newHead.x && p.y === newHead.y)) return this._crash(newHead, CRASH_REASONS.PATROL, action, { teleported });
    if (this._oob(newHead)) return this._crash(newHead, CRASH_REASONS.BOUNDARY, action, { teleported });
    if (this.solid.has(k(newHead.x, newHead.y))) return this._crash(newHead, CRASH_REASONS.WALL, action, { teleported });
    if (!immortal && this._onBody(newHead)) return this._crash(newHead, CRASH_REASONS.BOUNDARY, action, { teleported }); // тело вкл. хвост
    if (!immortal && this._poisonAt(newHead)) return this._crash(newHead, CRASH_REASONS.POISON, action, { teleported });
    if (isLooping) return this._crash(newHead, CRASH_REASONS.LOOP, action, { teleported });

    if (teleported) this.emit("teleport", { from: entry, to: newHead });
    this.snake.unshift(newHead);
    this.cellLastVisit[k(newHead.x, newHead.y)] = this.stepCount;

    // 2. Еда
    let ate = null;
    const idx = this.foods.findIndex((f) => f.x === newHead.x && f.y === newHead.y);
    if (idx !== -1) {
      ate = this.foods[idx];
      this.foods.splice(idx, 1);
      if (ate.type === "poison") {
        // бессмертная змея съедает яд без эффекта (без очков и роста)
        this.snake.pop();
        this.emit("eat", { food: ate, cell: newHead, neutralized: true });
      } else {
        this.score += ate.value;
        if (ate.type === "immortal") {
          this.immortalSteps = IMMORTAL_STEPS + 1; // −1 ниже → останется 50
          this.emit("immortal", { steps: IMMORTAL_STEPS });
        } else if (ate.type === "shrink") {
          if (this.snake.length > 3) this.snake.pop();
          if (this.snake.length > 3) this.snake.pop();
        } else if (ate.type === "turbo" || ate.type === "freeze") {
          this.speedEffect = { type: ate.type, steps: SPEED_EFFECT_STEPS + 1 };
          this.emit("speed", { effect: ate.type });
        }
        this.recentHeadPositions = [];
        this.lastEatStep = this.stepCount;
        if (this.score > this.maxScore) this.maxScore = this.score;
        this.emit("eat", { food: ate, cell: newHead });
      }
      this._ensureGoodFood();
    } else {
      this.snake.pop();
    }

    // 3. Дроны ходят по стенам НАЧАЛА шага; смерть, если дрон въехал в тело или поменялся с головой местами
    const oldDrones = drones.map((p) => ({ x: p.x, y: p.y }));
    const solidAtStart = this.solid;
    drones.forEach((p) => droneStep(p, solidAtStart));
    if (!immortal && this.immortalSteps <= 0) {
      const head = this.snake[0];
      for (let i = 0; i < drones.length; i++) {
        const p = drones[i], op = oldDrones[i];
        const onBody = PATROL_BODY_HITS ? this._onBody(p) : p.x === head.x && p.y === head.y;
        const swap = head.x === op.x && head.y === op.y && oldHead.x === p.x && oldHead.y === p.y;
        if (onBody || swap) return this._crash({ x: p.x, y: p.y }, CRASH_REASONS.PATROL, action, { teleported, ate });
      }
    }

    // 4. Фазы стен (призрак твердеет, только если его клетки свободны от змеи и дронов)
    const occupied = new Set([...this.snake.map((c) => k(c.x, c.y)), ...drones.map((p) => k(p.x, p.y))]);
    let wallsChanged = false;
    for (const s of this.structures) {
      if (s.phase === "GHOST" && s.timer <= 1) {
        if (!s.cells.some((c) => occupied.has(k(c.x, c.y)))) {
          s.phase = "SOLID"; s.timer = this._randRange(40, 80);
          wallsChanged = true;
          this.emit("wallPhase", { structure: s, phase: "SOLID" });
        }
        continue;
      }
      s.timer -= 1;
      if (s.timer <= 0) {
        if (s.phase === "SOLID") { s.phase = "BLINKING"; s.timer = BLINK_STEPS; }
        else if (s.phase === "BLINKING") { s.phase = "GHOST"; s.timer = this._randRange(15, 25); }
        wallsChanged = true;
        this.emit("wallPhase", { structure: s, phase: s.phase });
      }
    }
    if (wallsChanged) this._syncWalls();

    // 5. Счётчики: бессмертие и время жизни еды (кроме съеденной/только что появившейся)
    if (this.immortalSteps > 0) {
      this.immortalSteps -= 1;
      if (this.immortalSteps === 0) this.emit("immortalEnd");
    }
    const expired = [];
    for (const f of this.foods) {
      if (f.ttl == null || f.bornStep >= this.stepCount) continue;
      f.ttl -= 1;
      if (f.ttl <= 0) expired.push(f);
    }
    if (expired.length) {
      this.foods = this.foods.filter((f) => !expired.includes(f));
      expired.forEach((food) => this.emit("expire", { food }));
      this._ensureGoodFood();
    }
    if (this.speedEffect && --this.speedEffect.steps <= 0) {
      this.emit("speedEnd", { effect: this.speedEffect.type });
      this.speedEffect = null;
    }
    return { crashed: false, ate, teleported };
  }
}
