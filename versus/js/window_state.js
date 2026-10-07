// window_state.js — egocentric model input for the window models, port of
//   build_dataset_v11w.window_state (schema "v11w") and build_dataset_v12.state12 (schema "v12").
// Everything is relative to the snake's head; the opponent is presented as walls (as for the older models).
// Checked against Python by versus/tools/window_parity.mjs + window_parity.py.
import { DELTA, WINDOW_R } from "./versus_game.js";

const R = WINDOW_R;
const DRONE_R = 8;
const VALUE = { apple: 1, golden: 3, immortal: 2, shrink: 2 };
const KEYS = {
  v11w: ["current_dir", "recent_actions", "food_dir", "danger_UP", "danger_DOWN", "danger_LEFT", "danger_RIGHT",
    "food_rel", "edges", "snake_len", "immortal_steps", "body", "walls", "ghost_walls", "drones", "drone_path",
    "fog_of_war", "memory_food", "explored"],
  v12: ["current_dir", "recent_actions", "danger_UP", "danger_DOWN", "danger_LEFT", "danger_RIGHT",
    "foods_near", "foods_far", "memory_food", "poison", "edges", "snake_len", "immortal_steps",
    "body", "walls", "ghost_walls", "drones", "drone_path", "fog_of_war", "explored"],
};

function dir8(dx, dy) {
  const v = dy < 0 ? "UP" : dy > 0 ? "DOWN" : "";
  const h = dx < 0 ? "LEFT" : dx > 0 ? "RIGHT" : "";
  return v && h ? `${v}_${h}` : v || h || "SAME";
}

/** The parts shared by both schemas. base = match.modelState(s) (dangers, recent actions, …). */
function common(match, s, base) {
  const h = s.head, o = match.opponent(s);
  const inside = (x, y) => Math.abs(x - h.x) <= R && Math.abs(y - h.y) <= R;
  const rel = (x, y) => [x - h.x, y - h.y];
  // walls = solid/blinking structure cells in structure order, then the opponent (+ the cell in front of its head)
  const wallCells = [];
  for (const st of match.structures) if (st.phase !== "GHOST") for (const [x, y] of st.cells) wallCells.push([x, y]);
  if (o.alive) {
    for (const c of o.cells) wallCells.push([c.x, c.y]);
    const [dx, dy] = DELTA[o.dir];
    const fx = o.head.x + dx, fy = o.head.y + dy;
    if (fx >= 0 && fy >= 0 && fx < match.width && fy < match.height) wallCells.push([fx, fy]);
  }
  const ghosts = [];
  for (const st of match.structures) if (st.phase === "GHOST") for (const [x, y] of st.cells) if (st.timer > 0 && inside(x, y)) ghosts.push([...rel(x, y), st.timer]);
  const drones = [], paths = [];
  base.patrols.forEach((p, i) => {
    if (Math.abs(p[0] - h.x) <= DRONE_R && Math.abs(p[1] - h.y) <= DRONE_R) {
      drones.push([...rel(p[0], p[1]), p[2], p[3]]);
      paths.push(base.drone_path[i].map(([x, y]) => rel(x, y)));
    }
  });
  const mem = s.memory, trail = mem.trail;
  const explored = trail.length
    ? [trail.filter(([, y]) => y < h.y).length, trail.filter(([, y]) => y > h.y).length,
      trail.filter(([x]) => x < h.x).length, trail.filter(([x]) => x > h.x).length].map((c) => Math.min(9, pyRound((9 * c) / trail.length)))
    : [0, 0, 0, 0];
  return {
    inside, rel, explored,
    edges: [Math.min(h.y, R + 1), Math.min(match.height - 1 - h.y, R + 1), Math.min(h.x, R + 1), Math.min(match.width - 1 - h.x, R + 1)],
    body: s.cells.slice(1).filter((c) => inside(c.x, c.y)).map((c) => rel(c.x, c.y)).slice(0, 30),
    walls: wallCells.filter(([x, y]) => inside(x, y)).map(([x, y]) => rel(x, y)),
    ghost_walls: ghosts, drones, drone_path: paths,
  };
}

/** Python's round(): halves go to the even neighbour. */
function pyRound(v) {
  const f = Math.floor(v), d = v - f;
  if (Math.abs(d - 0.5) < 1e-9) return f % 2 === 0 ? f : f + 1;
  return Math.round(v);
}

/** Items the snake may know about: all without fog, otherwise in sight or remembered. */
function knownItems(match, s) {
  const mem = s.memory, good = match.foods.filter((f) => f.type !== "poison");
  if (!match.fog.active) return good.map((f) => ({ x: f.x, y: f.y, type: f.type }));
  const out = good.filter((f) => mem.visible(match, s, f.x, f.y)).map((f) => ({ x: f.x, y: f.y, type: f.type }));
  for (const [k, [type]] of mem.items) {
    const [x, y] = k.split(",").map(Number);
    if (!out.some((f) => f.x === x && f.y === y)) out.push({ x, y, type });
  }
  return out;
}

export function windowState(match, s, schema = "v12") {
  const base = match.modelState(s);
  const h = s.head, mem = s.memory;
  const c = common(match, s, base);
  const st = {
    current_dir: base.current_dir, recent_actions: base.recent_actions,
    danger_UP: base.danger_UP, danger_DOWN: base.danger_DOWN, danger_LEFT: base.danger_LEFT, danger_RIGHT: base.danger_RIGHT,
    edges: c.edges, snake_len: base.snake_len, immortal_steps: base.immortal_steps,
    body: c.body, walls: c.walls, ghost_walls: c.ghost_walls, drones: c.drones, drone_path: c.drone_path,
    fog_of_war: base.fog_of_war, explored: c.explored,
  };
  if (schema === "v11w") {
    // one food: the nearest known item (the V11w model never saw apple values)
    const known = knownItems(match, s);
    let food = null;
    for (const f of known) if (!food || Math.abs(f.x - h.x) + Math.abs(f.y - h.y) < Math.abs(food.x - h.x) + Math.abs(food.y - h.y)) food = f;
    let foodDir = "SAME";
    if (food) {
      const d = Math.abs(food.x - h.x) + Math.abs(food.y - h.y);
      foodDir = match.fog.active && d > R ? "UNKNOWN" : dir8(food.x - h.x, food.y - h.y);
    }
    st.food_dir = foodDir;
    st.food_rel = food && c.inside(food.x, food.y) && foodDir !== "UNKNOWN" ? c.rel(food.x, food.y) : null;
    st.memory_food = [];
    if (match.fog.active) {
      for (const [k, [, seen]] of [...mem.items].sort((a, b) => b[1][1] - a[1][1])) {
        const [x, y] = k.split(",").map(Number);
        if (!c.inside(x, y)) st.memory_food.push([...c.rel(x, y), mem.step - seen]);
      }
      st.memory_food = st.memory_food.slice(0, 3);
    }
  } else {
    const near = [], far = [], remembered = [];
    const dist = (f) => Math.abs(f.x - h.x) + Math.abs(f.y - h.y);
    const items = match.foods.filter((f) => f.type !== "poison").slice().sort((a, b) => dist(a) - dist(b));
    for (const f of items) {
      const [dx, dy] = c.rel(f.x, f.y);
      if (mem.visible(match, s, f.x, f.y)) near.push([dx, dy, VALUE[f.type], f.ttl === null ? -1 : f.ttl]);
      else if (!match.fog.active) far.push([dir8(dx, dy), dist(f) <= 15 ? "mid" : "far", VALUE[f.type]]);
    }
    if (match.fog.active) {
      const clip = (v) => Math.max(-20, Math.min(20, v));
      for (const [k, [type, seen]] of [...mem.items].sort((a, b) => b[1][1] - a[1][1])) {
        const [x, y] = k.split(",").map(Number);
        if (!mem.visible(match, s, x, y)) remembered.push([clip(x - h.x), clip(y - h.y), VALUE[type], mem.step - seen]);
      }
    }
    st.foods_near = near.slice(0, 4);
    st.foods_far = far.slice(0, 4);
    st.memory_food = remembered.slice(0, 4);
    st.poison = match.foods.filter((f) => f.type === "poison" && c.inside(f.x, f.y)).map((f) => c.rel(f.x, f.y));
  }
  const out = {};
  for (const k of KEYS[schema]) out[k] = st[k];
  return out;
}
