// Speed and effect of the shield (shield.js): a greedy BFS snake with and without the shield, both sides.
import { VersusMatch, DIRS, OPP } from "../js/versus_game.js";
import { shieldMove } from "../js/shield.js";
function play(seed, shield) {
  const m = new VersusMatch(seed); let t = 0, n = 0, vet = 0;
  while (!m.over) {
    const acts = {};
    for (const s of m.snakes) {
      if (!s.alive) continue;
      const st = m.modelState(s);
      let a = m.fallbackMove(s, st);
      if (shield) {
        const ranked = [a, ...DIRS.filter((d) => d !== a && d !== OPP[s.dir])];
        const t0 = performance.now(); const r = shieldMove(m, s, ranked); t = Math.max(t, performance.now() - t0); n++;
        vet += r.vetoed; a = r.move;
      }
      acts[s.id] = a;
    }
    m.advance(acts);
  }
  return { deaths: m.human.deaths + m.ai.deaths, score: m.human.score + m.ai.score, maxMs: t, vet };
}
for (const shield of [false, true]) {
  let d = 0, sc = 0, mx = 0, v = 0; const t0 = performance.now();
  for (let i = 0; i < 20; i++) { const r = play(1000 + i, shield); d += r.deaths; sc += r.score; mx = Math.max(mx, r.maxMs); v += r.vet; }
  console.log(`shield ${shield ? "on " : "off"} | deaths/match ${(d / 20).toFixed(1)} | score/match ${(sc / 20).toFixed(1)} | vetoes ${v} | worst shield call ${mx.toFixed(1)} ms | ${(performance.now() - t0).toFixed(0)} ms`);
}
