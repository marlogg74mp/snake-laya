// shield.js — safety shield on top of the model (port of build_dataset_v9.survives / has_escape, eval_shield.py).
// The model ranks the moves; going down that ranking, the first move after which the snake can still stay alive
// for `horizon` more steps (searched in a small simulator: own body, drones, walls) is played.
// The opponent is frozen in place (as the model sees it: walls + the cell in front of its head).
import { DIRS, DELTA, OPP, GRID } from "./versus_game.js";

const key = (x, y) => `${x},${y}`;
const inGrid = (x, y) => x >= 0 && x < GRID && y >= 0 && y < GRID;

/** A light copy of what matters for survival of snake s. */
function snapshot(match, s) {
  const o = match.opponent(s);
  const fixed = new Set(); // cells that kill regardless of time: the opponent (+ cell ahead), poison
  if (o.alive) {
    o.cells.forEach((c) => fixed.add(key(c.x, c.y)));
    const [dx, dy] = DELTA[o.dir];
    if (inGrid(o.head.x + dx, o.head.y + dy)) fixed.add(key(o.head.x + dx, o.head.y + dy));
  }
  const poison = new Set(match.foods.filter((f) => f.type === "poison").map((f) => key(f.x, f.y)));
  const food = new Set(match.foods.filter((f) => f.type !== "poison").map((f) => key(f.x, f.y)));
  // walls: SOLID/BLINKING stay solid; a GHOST wall becomes solid again after its timer (conservative)
  const walls = match.structures.map((st) => ({ cells: st.cells.map(([x, y]) => key(x, y)), ghostFor: st.phase === "GHOST" ? st.timer : 0 }));
  return {
    match, fixed, poison, food, walls,
    body: s.cells.map((c) => [c.x, c.y]), dir: s.dir, imm: s.immortal, t: 0,
    drones: match.drones.map((d) => d.clone()),
  };
}

function solidAt(w) {
  const solid = new Set();
  for (const st of w.walls) if (w.t >= st.ghostFor - 1) st.cells.forEach((c) => solid.add(c));
  return solid;
}

/** One step of the simulator; returns the next world or null if the snake dies. */
function step(w, move) {
  const a = move === OPP[w.dir] ? w.dir : move;
  const [dx, dy] = DELTA[a];
  const [hx, hy] = w.body[0];
  const [x, y] = w.match.warp(hx + dx, hy + dy);
  const k = key(x, y), imm = w.imm > 0;
  const solid = solidAt(w);
  if (!inGrid(x, y) || solid.has(k) || w.fixed.has(k)) return null;
  if (!imm && (w.poison.has(k) || w.body.some(([bx, by]) => bx === x && by === y) || w.drones.some((d) => d.x === x && d.y === y))) return null;
  const grow = w.food.has(k);
  const body = [[x, y], ...(grow ? w.body : w.body.slice(0, -1))];
  const drones = w.drones.map((d) => d.clone());
  const old = drones.map((d) => [d.x, d.y]);
  drones.forEach((d) => d.step(solid));
  if (!imm) {
    for (let i = 0; i < drones.length; i++) {
      const d = drones[i];
      if (body.some(([bx, by]) => bx === d.x && by === d.y)) return null;
      if (x === old[i][0] && y === old[i][1] && body[1] && d.x === body[1][0] && d.y === body[1][1]) return null;
    }
  }
  const food = grow ? new Set([...w.food].filter((f) => f !== k)) : w.food;
  return { ...w, body, dir: a, imm: Math.max(0, w.imm - 1), t: w.t + 1, drones, food };
}

function hasEscape(w, depth, budget) {
  if (depth === 0) return true;
  if (budget.n <= 0) return false;
  budget.n -= 1;
  const order = [w.dir, ...DIRS.filter((d) => d !== w.dir && d !== OPP[w.dir])];
  for (const m of order) {
    const n = step(w, m);
    if (n && hasEscape(n, depth - 1, budget)) return true;
  }
  return false;
}

/**
 * ranked: moves in the model's order of preference. Returns {move, vetoed, doomed}.
 * vetoed = the model's first choice was replaced; doomed = no move survives (the first choice is kept).
 */
export function shieldMove(match, s, ranked, { budget = 400 } = {}) {
  if (!ranked.length) return { move: s.dir, vetoed: false, doomed: false };
  const horizon = Math.min(s.cells.length + 3, 20);
  if (s.immortal > horizon) return { move: ranked[0], vetoed: false, doomed: false };
  const w = snapshot(match, s);
  for (let i = 0; i < ranked.length; i++) {
    const n = step(w, ranked[i]);
    if (n && hasEscape(n, horizon - 1, { n: budget })) return { move: ranked[i], vetoed: i > 0, doomed: false };
  }
  return { move: ranked[0], vetoed: false, doomed: true };
}
