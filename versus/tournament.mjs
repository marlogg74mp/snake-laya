// Headless tournament: our v10 (ModernBERT-base) vs the real Laya 421M fine-tuned on Snake.
// Same match logic and rules as the Snake Duel page (js/versus_game.js), moves from the model server.
// Each map is played twice with sides swapped, so neither model profits from a side or a seed.
//   node versus/tournament.mjs [maps=10] [server=http://127.0.0.1:9500]
import { VersusMatch, DIRS, OPP } from "./js/versus_game.js";
import { writeFileSync } from "node:fs";

const MAPS = +(process.argv[2] || 10);
const SERVER = process.argv[3] || "http://127.0.0.1:9500";
const MODELS = { v10: "laya", laya421: "laya_convai" };

async function decide(match, snake, model) {
  const state = match.modelState(snake);
  const legal = DIRS.filter((d) => !state[`danger_${d}`] && d !== OPP[snake.dir]);
  try {
    const r = await fetch(`${SERVER}/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, state }), signal: AbortSignal.timeout(3000),
    });
    const p = (await r.json()).probabilities;
    // same safety mask as the duel page: the most probable move that is not deadly on the next cell
    const pool = legal.length ? legal : DIRS.filter((d) => d !== OPP[snake.dir]);
    return { action: pool.sort((a, b) => p[b] - p[a])[0], fallback: false };
  } catch {
    return { action: match.fallbackMove(snake, state), fallback: true };
  }
}

async function playMatch(seed, left, right) {
  const m = new VersusMatch(seed);
  const side = { human: left, ai: right }; // "human" = left snake, "ai" = right snake
  const stats = { [left]: { deathReasons: {}, fallbacks: 0 }, [right]: { deathReasons: {}, fallbacks: 0 } };
  while (!m.over) {
    const moves = await Promise.all(m.snakes.map((s) =>
      s.alive ? decide(m, s, MODELS[side[s.id]]) : Promise.resolve({ action: s.dir, fallback: false })));
    m.snakes.forEach((s, i) => { if (moves[i].fallback) stats[side[s.id]].fallbacks++; });
    m.advance({ human: moves[0].action, ai: moves[1].action });
    for (const e of m.drainEvents()) {
      if (e.type === "death") {
        const d = stats[side[e.who]].deathReasons;
        d[e.reason] = (d[e.reason] || 0) + 1;
      }
    }
  }
  const res = {};
  for (const s of m.snakes) {
    const name = side[s.id];
    res[name] = { score: s.score, kills: s.kills, deaths: s.deaths, apples: s.apples, side: s.id === "human" ? "left" : "right", ...stats[name] };
  }
  return { seed, ...res };
}

const results = [];
const t0 = Date.now();
for (let i = 0; i < MAPS; i++) {
  const seed = 70_000 + i;
  for (const [left, right] of [["v10", "laya421"], ["laya421", "v10"]]) {
    const r = await playMatch(seed, left, right);
    results.push(r);
    const w = r.v10.score > r.laya421.score ? "v10" : r.laya421.score > r.v10.score ? "laya421" : "draw";
    console.log(`map ${i + 1}/${MAPS} seed ${seed} | left ${left} | v10 ${r.v10.score} : ${r.laya421.score} laya421 | winner ${w} | ${((Date.now() - t0) / 60000).toFixed(1)} min`);
  }
}

const sum = (k, f) => results.reduce((a, r) => a + f(r[k]), 0);
const wins = { v10: 0, laya421: 0, draw: 0 };
for (const r of results) wins[r.v10.score > r.laya421.score ? "v10" : r.laya421.score > r.v10.score ? "laya421" : "draw"]++;
const summary = { matches: results.length, wins };
for (const k of ["v10", "laya421"]) {
  const deaths = {};
  results.forEach((r) => Object.entries(r[k].deathReasons).forEach(([d, n]) => (deaths[d] = (deaths[d] || 0) + n)));
  summary[k] = {
    avgScore: +(sum(k, (x) => x.score) / results.length).toFixed(2),
    avgApples: +(sum(k, (x) => x.apples) / results.length).toFixed(2),
    avgDeaths: +(sum(k, (x) => x.deaths) / results.length).toFixed(2),
    kills: sum(k, (x) => x.kills), fallbacks: sum(k, (x) => x.fallbacks), deathsByReason: deaths,
  };
}
console.log("\nSUMMARY", JSON.stringify(summary, null, 2));
writeFileSync(new URL("./tournament_results.json", import.meta.url), JSON.stringify({ summary, results }, null, 2));
