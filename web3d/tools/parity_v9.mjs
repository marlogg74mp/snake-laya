// Rule-parity harness, part 2 (Node): replays traces from parity_v9.py in web3d/js/game.js and compares
// every model_state (as JSON text, i.e. what Laya sees) and every step result with build_dataset_v9.World.
// Usage: node web3d/tools/parity_v9.mjs [parity_v9_traces.json]
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { SnakeGame } from "../js/game.js";

const here = dirname(fileURLToPath(import.meta.url));
const traces = JSON.parse(readFileSync(process.argv[2] || join(here, "parity_v9_traces.json"), "utf8"));

/** Same pinned randomness as PinnedRng in parity_v9.py. */
class PinnedGame extends SnakeGame {
  _randRange(lo) { return lo; }
  _rollFoodType() { return "standard"; }
  _pickCell(cells) { return cells[0]; }
  // Python's World._spawn_food excludes body + every wall cell only (not portals / other food)
  _occupied(x, y) { return this.snake.some((s) => s.x === x && s.y === y) || this._allWallCells.has(`${x},${y}`); }
}

let steps = 0, mismatches = 0, deathsChecked = 0, immortalSteps = 0, ghostSteps = 0, blinkSteps = 0;
const firstFail = [];
for (const [ti, tr] of traces.entries()) {
  const g = new PinnedGame({ noReset: true, settings: { loopDetector: false, walls: true } });
  g.loadWorld(tr.world);
  for (const [si, st] of tr.steps.entries()) {
    const js = JSON.stringify(g.getModelState("v9"));
    const py = JSON.stringify(st.state);
    steps++;
    if (st.state.immortal_steps > 0) immortalSteps++;
    if (st.state.phase_walls.some((p) => p[2] > 0)) ghostSteps++;
    if (st.state.phase_walls.some((p) => p[2] < 0)) blinkSteps++;
    if (js !== py) {
      mismatches++;
      if (firstFail.length < 3) firstFail.push({ trace: ti, step: si, py, js });
      break;
    }
    const r = g.step(st.action);
    const jsResult = r.crashed ? "dead" : "ok";
    if (st.result === "dead") deathsChecked++;
    if (jsResult !== (st.result === "win" ? "ok" : st.result)) {
      mismatches++;
      if (firstFail.length < 3) firstFail.push({ trace: ti, step: si, action: st.action, py: st.result, js: jsResult, reason: r.reason });
      break;
    }
  }
  if (tr.final && !g.over && JSON.stringify(g.getModelState("v9")) !== JSON.stringify(tr.final)) {
    mismatches++;
    if (firstFail.length < 3) firstFail.push({ trace: ti, step: "final" });
  }
}
console.log(`traces=${traces.length} steps=${steps} deaths=${deathsChecked} immortal_steps>0=${immortalSteps} ghost=${ghostSteps} blinking=${blinkSteps}`);
console.log(mismatches ? `MISMATCHES: ${mismatches}` : "PARITY OK: every state and step result identical");
for (const f of firstFail) {
  console.log(JSON.stringify(f).slice(0, 4000));
  if (f.py && f.js && f.py.startsWith("{")) {
    const a = JSON.parse(f.py), b = JSON.parse(f.js);
    for (const key of Object.keys(a)) if (JSON.stringify(a[key]) !== JSON.stringify(b[key])) console.log(`  key ${key}: py=${JSON.stringify(a[key])} js=${JSON.stringify(b[key])}`);
  }
}
process.exit(mismatches ? 1 : 0);
