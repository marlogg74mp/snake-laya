// Dumps duel snapshots + the JS window state (schema v12) for window_parity.py to rebuild with the training code.
import { VersusMatch } from "../js/versus_game.js";
import { windowState } from "../js/window_state.js";
import { writeFileSync } from "node:fs";

const out = [];
for (let seed = 1; seed <= 6; seed++) {
  const m = new VersusMatch(500 + seed);
  for (let t = 0; t < 400 && !m.over; t++) {
    if (t === 60 + seed * 10) { m.fog.active = true; m.fog.timer = 80; } // make sure fog + memory get covered
    for (const s of m.snakes) {
      if (!s.alive || t % 5 !== seed % 5) continue;
      const o = m.opponent(s);
      out.push({
        width: m.width, height: m.height, fog: m.fog.active, portals: m.portals.map((q) => [q.x, q.y]),
        snake: { cells: s.cells.map((c) => [c.x, c.y]), dir: s.dir, recent: [...s.recent], immortal: s.immortal },
        opp: { alive: o.alive, cells: o.cells.map((c) => [c.x, c.y]), dir: o.dir },
        structures: m.structures.map((st) => ({ cells: st.cells, phase: st.phase, timer: st.timer })),
        drones: m.drones.map((d) => ({ kind: d.kind, x: d.x, y: d.y, dx: d.dx, dy: d.dy, lo: d.lo, hi: d.hi })),
        foods: m.foods.map((f) => ({ x: f.x, y: f.y, type: f.type, ttl: f.ttl })),
        memory: { items: [...s.memory.items.entries()].map(([k, v]) => [k, [...v]]), step: s.memory.step, trail: s.memory.trail.map((p) => [...p]) },
        js: windowState(m, s, "v12"),
      });
    }
    const act = (s) => (s.alive ? m.fallbackMove(s, m.modelState(s)) : s.dir);
    m.advance({ human: act(m.human), ai: act(m.ai) });
    m.drainEvents();
  }
}
writeFileSync(new URL("./window_parity.json", import.meta.url), JSON.stringify(out));
console.log("snapshots", out.length, "with fog", out.filter((x) => x.fog).length, "with memory", out.filter((x) => x.js.memory_food.length).length);
