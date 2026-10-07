// versus_game.js — Human vs AI match logic (VERSUS_RULES.md on top of RULES_V9.md). No DOM, no rendering.

export const GRID = 20;
export const MATCH_STEPS = 600;
export const DIRS = ["UP", "DOWN", "LEFT", "RIGHT"];
export const DELTA = { UP: [0, -1], DOWN: [0, 1], LEFT: [-1, 0], RIGHT: [1, 0] };
export const OPP = { UP: "DOWN", DOWN: "UP", LEFT: "RIGHT", RIGHT: "LEFT" };

const IMMORTAL_STEPS = 50;
const SPAWN_SHIELD_STEPS = 10;
const RESPAWN_DELAY = 10;
const KILL_BONUS = 5;
const DEATH_PENALTY = 3;
const ITEM_TTL = 60;
const BLINK_STEPS = 3;
const DRONE_PATH_STEPS = 6;
const MAX_BODY_CELLS = 30;
const FOOD_SCORE = { apple: 1, golden: 3, immortal: 2 };
export const FOG_RADIUS = 5;           // Manhattan, same as training (sight_radius)
const FOG_WARNING = 10;                // steps of warning before the fog falls

const key = (x, y) => `${x},${y}`;
const inGrid = (x, y) => x >= 0 && x < GRID && y >= 0 && y < GRID;

export function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------- drones (same as build_dataset_v9.Drone, linear lanes)
export class Drone {
  constructor(kind, x, y, dx, dy, lo, hi) {
    Object.assign(this, { kind, x, y, dx, dy, lo, hi });
  }
  inRoute(x, y) {
    return this.kind === "h" ? y === this.y && x >= this.lo && x <= this.hi : x === this.x && y >= this.lo && y <= this.hi;
  }
  plan(solid) {
    for (const [dx, dy] of [[this.dx, this.dy], [-this.dx, -this.dy]]) {
      const nx = this.x + dx, ny = this.y + dy;
      if (this.inRoute(nx, ny) && !solid.has(key(nx, ny))) return [dx, dy];
    }
    return [0, 0];
  }
  step(solid) {
    const [dx, dy] = this.plan(solid);
    if (dx || dy) { this.dx = dx; this.dy = dy; }
    this.x += dx; this.y += dy;
  }
  clone() { return new Drone(this.kind, this.x, this.y, this.dx, this.dy, this.lo, this.hi); }
}

// ---------------------------------------------------------------- snake
// Agent-side memory for the window models (port of build_dataset_v12.Memory12):
// items seen {"x,y": [kind, step seen]}, forgotten when their cell is seen empty; last 200 head positions.
export const WINDOW_R = 5;
export class SnakeMemory {
  constructor() { this.items = new Map(); this.trail = []; this.step = 0; }
  visible(match, s, x, y) {
    const h = s.head;
    return Math.abs(x - h.x) <= WINDOW_R && Math.abs(y - h.y) <= WINDOW_R &&
      (!match.fog.active || Math.abs(x - h.x) + Math.abs(y - h.y) <= WINDOW_R);
  }
  update(match, s) {
    this.step += 1;
    this.trail.push([s.head.x, s.head.y]);
    if (this.trail.length > 200) this.trail.shift();
    const good = match.foods.filter((f) => f.type !== "poison");
    for (const k of [...this.items.keys()]) {
      const [x, y] = k.split(",").map(Number);
      if (this.visible(match, s, x, y) && !good.some((f) => f.x === x && f.y === y)) this.items.delete(k);
    }
    for (const f of good) {
      if (this.visible(match, s, f.x, f.y)) {
        this.items.set(`${f.x},${f.y}`, [f.type, this.step]); // like a Python dict: existing keys keep their place
      }
    }
    if (this.items.size > 4) {
      const oldest = [...this.items.entries()].sort((a, b) => a[1][1] - b[1][1]).slice(0, this.items.size - 4);
      for (const [k] of oldest) this.items.delete(k);
    }
  }
}

class Snake {
  constructor(id, cells, dir) {
    this.id = id;
    this.score = 0;
    this.kills = 0;
    this.deaths = 0;
    this.apples = 0;
    this.place(cells, dir, 0);
  }
  place(cells, dir, shield) {
    this.memory = new SnakeMemory(); // a respawned snake starts with an empty memory
    this.cells = cells.map(([x, y]) => ({ x, y }));
    this.dir = dir;
    this.recent = Array(5).fill(dir);
    this.alive = true;
    this.respawnIn = 0;
    this.immortal = shield;
    this.grow = 0;
  }
  get head() { return this.cells[0]; }
  occupies(x, y) { return this.alive && this.cells.some((c) => c.x === x && c.y === y); }
}

// ---------------------------------------------------------------- match
export class VersusMatch {
  constructor(seed = (Math.random() * 2 ** 31) | 0) {
    this.seed = seed;
    this.rand = mulberry32(seed);
    this.width = this.height = GRID;
    this.step = 0;
    this.over = false;
    this.events = [];
    this.human = new Snake("human", [[4, 10], [3, 10], [2, 10]], "RIGHT");
    this.ai = new Snake("ai", [[15, 10], [16, 10], [17, 10]], "LEFT");
    this.snakes = [this.human, this.ai];
    this.drones = [new Drone("h", 2, 4, 1, 0, 2, 17), new Drone("h", 17, 15, -1, 0, 2, 17)];
    this._buildMap();
    this.foods = [];
    // Periodic fog of war (VERSUS_RULES.md): falls every 120-200 steps for 50-70 steps
    this.fog = { active: false, timer: this.ri(120, 200) };
    this._ensureFood();
    for (const sn of this.snakes) sn.memory.update(this, sn);
  }

  ri(lo, hi) { return lo + Math.floor(this.rand() * (hi - lo + 1)); }

  // Mirror-symmetric walls and portals (x <-> 19 - x)
  _buildMap() {
    const templates = [
      [[0, 0], [1, 0], [2, 0]], [[0, 0], [0, 1], [0, 2]], [[0, 0], [1, 0], [0, 1], [1, 1]],
      [[0, 0], [1, 0], [2, 0], [0, 1]], [[0, 0], [1, 0]], [[0, 0], [0, 1]],
    ];
    const forbidden = new Set();
    for (let x = 0; x < GRID; x++) for (let y = 7; y <= 13; y++) if (x <= 7 || x >= 12) forbidden.add(key(x, y));
    for (const s of this.snakes) s.cells.forEach((c) => forbidden.add(key(c.x, c.y)));
    this.structures = [];
    const count = this.ri(2, 3);
    for (let n = 0, tries = 0; n < count && tries < 300; tries++) {
      const t = templates[Math.floor(this.rand() * templates.length)];
      const ox = this.ri(1, 7), oy = this.ri(1, 17);
      const cells = t.map(([dx, dy]) => [ox + dx, oy + dy]);
      const mirror = cells.map(([x, y]) => [GRID - 1 - x, y]);
      const all = [...cells, ...mirror];
      if (all.every(([x, y]) => inGrid(x, y) && x !== 9 && x !== 10 && !forbidden.has(key(x, y)))) {
        all.forEach(([x, y]) => {
          forbidden.add(key(x, y));
          for (const [dx, dy] of Object.values(DELTA)) forbidden.add(key(x + dx, y + dy)); // keep structures apart
        });
        const timer = this.ri(30, 80);
        this.structures.push({ cells, phase: "SOLID", timer }, { cells: mirror, phase: "SOLID", timer });
        n++;
      }
    }
    this.portals = [];
    for (let tries = 0; tries < 200; tries++) {
      const x = this.ri(1, 6), y = this.ri(1, 18);
      if (!forbidden.has(key(x, y)) && !forbidden.has(key(GRID - 1 - x, y)) && y !== 4 && y !== 15) {
        this.portals = [{ x, y }, { x: GRID - 1 - x, y }];
        break;
      }
    }
    this._syncWalls();
  }

  _syncWalls() {
    this.solid = new Set();
    for (const s of this.structures) if (s.phase !== "GHOST") s.cells.forEach(([x, y]) => this.solid.add(key(x, y)));
  }

  warp(x, y) {
    if (this.portals.length === 2) {
      const [a, b] = this.portals;
      if (x === a.x && y === a.y) return [b.x, b.y];
      if (x === b.x && y === b.y) return [a.x, a.y];
    }
    return [x, y];
  }

  opponent(s) { return s === this.human ? this.ai : this.human; }

  // ------------------------------------------------------------ food
  _freeCell() {
    const blocked = new Set();
    this.structures.forEach((s) => s.cells.forEach(([x, y]) => blocked.add(key(x, y))));
    this.snakes.forEach((s) => s.alive && s.cells.forEach((c) => blocked.add(key(c.x, c.y))));
    this.drones.forEach((d) => blocked.add(key(d.x, d.y)));
    this.foods.forEach((f) => blocked.add(key(f.x, f.y)));
    this.portals.forEach((p) => blocked.add(key(p.x, p.y)));
    for (let tries = 0; tries < 400; tries++) {
      const x = this.ri(0, GRID - 1), y = this.ri(0, GRID - 1);
      if (!blocked.has(key(x, y)) && y !== 4 && y !== 15) return { x, y };
    }
    return null;
  }

  _spawn(type) {
    const c = this._freeCell();
    if (c) this.foods.push({ ...c, type, ttl: type === "apple" ? null : ITEM_TTL, born: this.step });
  }

  _ensureFood() {
    while (this.foods.filter((f) => f.type === "apple").length < 2) this._spawn("apple");
    const has = (t) => this.foods.some((f) => f.type === t);
    if (!has("golden") && this.rand() < 1 / 90) this._spawn("golden");
    if (!has("immortal") && this.rand() < 1 / 160) this._spawn("immortal");
    if (this.foods.filter((f) => f.type === "poison").length < 2 && this.rand() < 1 / 70) this._spawn("poison");
  }

  // ------------------------------------------------------------ one simultaneous step
  /** actions: {human: dir, ai: dir}. Reversals are ignored (keep current direction). */
  advance(actions) {
    if (this.over) return;
    const live = this.snakes.filter((s) => s.alive);
    const plan = new Map();
    for (const s of live) {
      let a = actions[s.id] || s.dir;
      if (a === OPP[s.dir]) a = s.dir;
      const [dx, dy] = DELTA[a];
      const [nx, ny] = this.warp(s.head.x + dx, s.head.y + dy);
      plan.set(s, { a, x: nx, y: ny, teleported: nx !== s.head.x + dx || ny !== s.head.y + dy });
    }

    // 1. collisions on the new head cells (all current cells count, tails included)
    const deaths = new Map(); // snake -> {reason, killer}
    for (const s of live) {
      const p = plan.get(s);
      const o = this.opponent(s);
      const imm = s.immortal > 0;
      if (!inGrid(p.x, p.y)) deaths.set(s, { reason: "border" });
      else if (this.solid.has(key(p.x, p.y))) deaths.set(s, { reason: "wall" });
      else if (!imm && s.cells.some((c) => c.x === p.x && c.y === p.y)) deaths.set(s, { reason: "self" });
      else if (!imm && this.drones.some((d) => d.x === p.x && d.y === p.y)) deaths.set(s, { reason: "drone" });
      else if (!imm && this.foods.some((f) => f.type === "poison" && f.x === p.x && f.y === p.y)) deaths.set(s, { reason: "poison" });
      else if (!imm && o.alive && o.cells.some((c) => c.x === p.x && c.y === p.y)) {
        const headSwap = plan.has(o) && o.head.x === p.x && o.head.y === p.y &&
          plan.get(o).x === s.head.x && plan.get(o).y === s.head.y;
        deaths.set(s, headSwap ? { reason: "head-on" } : { reason: "opponent", killer: o });
      }
    }
    if (live.length === 2) {
      const [a, b] = live, pa = plan.get(a), pb = plan.get(b);
      if (pa.x === pb.x && pa.y === pb.y) {
        if (!(a.immortal > 0) && !deaths.has(a)) deaths.set(a, { reason: "head-on" });
        if (!(b.immortal > 0) && !deaths.has(b)) deaths.set(b, { reason: "head-on" });
      }
    }

    // 2. move survivors, eat
    for (const s of live) {
      if (deaths.has(s)) continue;
      const p = plan.get(s);
      s.dir = p.a;
      s.recent.push(p.a);
      s.recent.shift();
      s.cells.unshift({ x: p.x, y: p.y });
      if (p.teleported) this.events.push({ type: "teleport", who: s.id, x: p.x, y: p.y });
      const fi = this.foods.findIndex((f) => f.x === p.x && f.y === p.y);
      if (fi >= 0) {
        const f = this.foods[fi];
        this.foods.splice(fi, 1);
        if (f.type === "poison") {
          this.events.push({ type: "shrug", who: s.id, x: p.x, y: p.y }); // immortal: poison has no effect
        } else {
          s.score += FOOD_SCORE[f.type];
          s.apples += 1;
          s.grow += 1;
          if (f.type === "immortal") s.immortal = IMMORTAL_STEPS + 1;
          this.events.push({ type: "eat", who: s.id, food: f.type, x: p.x, y: p.y });
        }
      }
      if (s.grow > 0) s.grow -= 1; else s.cells.pop();
    }
    for (const [s, d] of deaths) this._kill(s, d);

    // 3. drones (walls as at the start of the step)
    const oldDrones = this.drones.map((d) => [d.x, d.y]);
    this.drones.forEach((d) => d.step(this.solid));
    for (const s of this.snakes) {
      if (!s.alive || s.immortal > 0) continue;
      const prevHead = plan.get(s);
      this.drones.forEach((d, i) => {
        if (!s.alive) return;
        const onBody = s.cells.some((c) => c.x === d.x && c.y === d.y);
        const swap = prevHead && s.head.x === oldDrones[i][0] && s.head.y === oldDrones[i][1] &&
          d.x === s.cells[1]?.x && d.y === s.cells[1]?.y;
        if (onBody || swap) this._kill(s, { reason: "drone" });
      });
    }

    // 4. wall phases
    const occupied = new Set();
    this.snakes.forEach((s) => s.alive && s.cells.forEach((c) => occupied.add(key(c.x, c.y))));
    this.drones.forEach((d) => occupied.add(key(d.x, d.y)));
    for (const st of this.structures) {
      if (st.phase === "GHOST" && st.timer <= 1) {
        if (!st.cells.some(([x, y]) => occupied.has(key(x, y)))) {
          st.phase = "SOLID"; st.timer = this.ri(40, 80);
          this.events.push({ type: "wall", phase: "SOLID", cells: st.cells });
        }
        continue;
      }
      st.timer -= 1;
      if (st.timer <= 0) {
        if (st.phase === "SOLID") { st.phase = "BLINKING"; st.timer = BLINK_STEPS; }
        else if (st.phase === "BLINKING") {
          st.phase = "GHOST"; st.timer = this.ri(15, 25);
          this.events.push({ type: "wall", phase: "GHOST", cells: st.cells });
        }
      }
    }
    this._syncWalls();

    // 5. timers, respawns, food, fog
    if (--this.fog.timer <= 0) {
      this.fog.active = !this.fog.active;
      this.fog.timer = this.fog.active ? this.ri(50, 70) : this.ri(120, 200);
      this.events.push({ type: "fog", active: this.fog.active });
    } else if (!this.fog.active && this.fog.timer === FOG_WARNING) {
      this.events.push({ type: "fogWarning" });
    }
    for (const s of this.snakes) {
      if (s.alive && s.immortal > 0) s.immortal -= 1;
      if (!s.alive && --s.respawnIn <= 0) this._respawn(s);
    }
    for (const f of this.foods) if (f.ttl !== null) f.ttl -= 1;
    this.foods = this.foods.filter((f) => f.ttl === null || f.ttl > 0);
    this._ensureFood();

    for (const sn of this.snakes) if (sn.alive) sn.memory.update(this, sn);
    this.step += 1;
    if (this.step >= MATCH_STEPS) {
      this.over = true;
      const h = this.human.score, a = this.ai.score;
      this.events.push({ type: "end", winner: h > a ? "human" : a > h ? "ai" : "draw" });
    }
  }

  _kill(s, { reason, killer }) {
    if (!s.alive) return;
    this.events.push({ type: "death", who: s.id, reason, cells: s.cells.map((c) => ({ ...c })) });
    s.alive = false;
    s.deaths += 1;
    s.score = Math.max(0, s.score - DEATH_PENALTY);
    s.respawnIn = RESPAWN_DELAY;
    s.immortal = 0;
    if (killer && killer.alive) {
      killer.score += KILL_BONUS;
      killer.kills += 1;
      this.events.push({ type: "kill", who: killer.id, victim: s.id });
    }
  }

  _respawn(s) {
    const o = this.opponent(s);
    const blocked = new Set([...this.solid]);
    this.structures.forEach((st) => st.cells.forEach(([x, y]) => blocked.add(key(x, y))));
    if (o.alive) o.cells.forEach((c) => blocked.add(key(c.x, c.y)));
    this.drones.forEach((d) => blocked.add(key(d.x, d.y)));
    this.foods.forEach((f) => blocked.add(key(f.x, f.y)));
    this.portals.forEach((p) => blocked.add(key(p.x, p.y)));
    let best = null;
    for (let y = 1; y < GRID - 1; y++) {
      for (let x = 1; x < GRID - 1; x++) {
        for (const dir of DIRS) {
          const [dx, dy] = DELTA[dir];
          const cells = [[x, y], [x - dx, y - dy], [x - 2 * dx, y - 2 * dy]];
          const ahead = [x + dx, y + dy];
          if (!cells.every(([cx, cy]) => inGrid(cx, cy) && !blocked.has(key(cx, cy)))) continue;
          if (!inGrid(...ahead) || blocked.has(key(...ahead))) continue;
          const dOpp = o.alive ? Math.abs(o.head.x - x) + Math.abs(o.head.y - y) : 20;
          const dDrone = Math.min(...this.drones.map((d) => Math.abs(d.x - x) + Math.abs(d.y - y)));
          const score = Math.min(dOpp, 12) * 2 + Math.min(dDrone, 6) * 3 + this.rand();
          if (!best || score > best.score) best = { score, cells, dir };
        }
      }
    }
    if (!best) { s.respawnIn = 1; return; }
    s.place(best.cells, best.dir, SPAWN_SHIELD_STEPS);
    this.events.push({ type: "respawn", who: s.id, x: best.cells[0][0], y: best.cells[0][1] });
  }

  /** Fog: is cell (x, y) visible to snake s? Dead snakes see nothing while the fog is down. */
  visibleTo(s, x, y) {
    if (!this.fog.active) return true;
    if (!s.alive) return false;
    return Math.abs(x - s.head.x) + Math.abs(y - s.head.y) <= FOG_RADIUS;
  }

  drainEvents() { const e = this.events; this.events = []; return e; }

  // ------------------------------------------------------------ model input (RULES_V9 §5) for snake s
  modelState(s) {
    const o = this.opponent(s);
    const head = s.head;
    const imm = s.immortal;
    // in fog only food within FOG_RADIUS of the head is visible (as in training)
    const goods = this.foods.filter((f) => f.type !== "poison" && this.visibleTo(s, f.x, f.y));
    let food = null;
    for (const f of goods) {
      const d = Math.abs(f.x - head.x) + Math.abs(f.y - head.y);
      if (!food || d < food.d) food = { f, d };
    }
    let foodDir = "UNKNOWN", foodPos = [-1, -1];
    if (food) {
      const { x, y } = food.f;
      const v = y < head.y ? "UP" : y > head.y ? "DOWN" : "";
      const h = x < head.x ? "LEFT" : x > head.x ? "RIGHT" : "";
      foodDir = v && h ? `${v}_${h}` : v || h || "SAME";
      foodPos = [x, y];
    }
    // the opponent is presented to the model as walls (+ the cell in front of its head)
    const oppCells = new Set();
    if (o.alive) {
      o.cells.forEach((c) => oppCells.add(key(c.x, c.y)));
      const [dx, dy] = DELTA[o.dir];
      if (inGrid(o.head.x + dx, o.head.y + dy)) oppCells.add(key(o.head.x + dx, o.head.y + dy));
    }
    const droneDanger = new Set();
    for (const d of this.drones) {
      const [dx, dy] = d.plan(this.solid);
      droneDanger.add(key(d.x, d.y));
      droneDanger.add(key(d.x + dx, d.y + dy));
    }
    const danger = {};
    for (const dir of DIRS) {
      const [dx, dy] = DELTA[dir];
      const [x, y] = this.warp(head.x + dx, head.y + dy);
      danger[dir] = !inGrid(x, y) || this.solid.has(key(x, y)) || oppCells.has(key(x, y)) ||
        // as in the training data: own body is a danger once immortality is over (imm <= 0),
        // drones and poison already on its last step (imm <= 1)
        (imm <= 0 && s.cells.some((c) => c.x === x && c.y === y)) ||
        (imm <= 1 && (droneDanger.has(key(x, y)) || this.foods.some((f) => f.type === "poison" && f.x === x && f.y === y)));
    }
    const obstacles = [...this.solid].map((k) => k.split(",").map(Number));
    oppCells.forEach((k) => obstacles.push(k.split(",").map(Number)));
    const phaseWalls = [];
    for (const st of this.structures) {
      if (st.phase === "GHOST") st.cells.forEach(([x, y]) => phaseWalls.push([x, y, st.timer]));
      else if (st.phase === "BLINKING") st.cells.forEach(([x, y]) => phaseWalls.push([x, y, -st.timer]));
    }
    return {
      current_dir: s.dir,
      recent_actions: [...s.recent],
      food_dir: foodDir,
      danger_UP: danger.UP, danger_DOWN: danger.DOWN, danger_LEFT: danger.LEFT, danger_RIGHT: danger.RIGHT,
      head_pos: [head.x, head.y],
      body: s.cells.slice(1, MAX_BODY_CELLS + 1).map((c) => [c.x, c.y]),
      food_pos: foodPos,
      snake_len: s.cells.length,
      immortal_steps: imm,
      obstacles,
      phase_walls: phaseWalls,
      portals: this.portals.map((p) => [p.x, p.y]),
      fog_of_war: this.fog.active,
      sight_radius: 5,
      patrols: this.drones.map((d) => { const [dx, dy] = d.plan(this.solid); return [d.x, d.y, dx, dy]; }),
      drone_path: this.drones.map((d) => {
        const g = d.clone(), path = [];
        for (let i = 0; i < DRONE_PATH_STEPS; i++) { g.step(this.solid); path.push([g.x, g.y]); }
        return path;
      }),
    };
  }

  /** Simple BFS fallback for the AI when the model does not answer in time. */
  fallbackMove(s, state) {
    const safe = DIRS.filter((d) => !state[`danger_${d}`] && d !== OPP[s.dir]);
    if (!safe.length) return s.dir;
    if (state.food_pos[0] < 0) return safe.includes(s.dir) ? s.dir : safe[0];
    const blocked = new Set(state.obstacles.map(([x, y]) => key(x, y)));
    s.cells.forEach((c) => blocked.add(key(c.x, c.y)));
    let best = safe[0], bestD = Infinity;
    for (const d of safe) {
      const [dx, dy] = DELTA[d];
      const start = this.warp(s.head.x + dx, s.head.y + dy);
      const seen = new Set([key(...start)]);
      const q = [[...start, 0]];
      while (q.length) {
        const [x, y, n] = q.shift();
        if (x === state.food_pos[0] && y === state.food_pos[1]) { if (n < bestD) { bestD = n; best = d; } break; }
        for (const [ex, ey] of Object.values(DELTA)) {
          const [nx, ny] = this.warp(x + ex, y + ey);
          if (inGrid(nx, ny) && !blocked.has(key(nx, ny)) && !seen.has(key(nx, ny))) { seen.add(key(nx, ny)); q.push([nx, ny, n + 1]); }
        }
      }
    }
    return best;
  }
}
