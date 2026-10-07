// render2d.js — neon canvas renderer for the versus match (interpolated, particles, shake).
import { GRID, DELTA, FOG_RADIUS } from "./versus_game.js";

const COLORS = {
  human: { body: "#22d3ee", glow: "#67e8f9", dark: "#0e7490" },
  ai: { body: "#f472b6", glow: "#f9a8d4", dark: "#9d174d" },
  wall: "#ff2e63", ghost: "#a78bfa", drone: "#facc15", portal: ["#22d3ee", "#c084fc"],
  food: { apple: "#ff4d4d", golden: "#facc15", immortal: "#fde047", poison: "#84cc16" },
};

export class Renderer2D {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.particles = [];
    this.rings = [];
    this.shake = 0;
    this.prev = null;
    this.resize();
    addEventListener("resize", () => this.resize());
  }

  resize() {
    const box = this.canvas.parentElement.getBoundingClientRect();
    const size = Math.floor(Math.min(box.width, box.height || box.width));
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.canvas.style.width = this.canvas.style.height = `${size}px`;
    this.canvas.width = this.canvas.height = Math.floor(size * dpr);
    this.cell = this.canvas.width / GRID;
  }

  /** Remember positions before a step so movement can be interpolated. */
  /** Grid cell under a screen point, or null. */
  cellAt(clientX, clientY) {
    const r = this.canvas.getBoundingClientRect();
    const x = Math.floor(((clientX - r.left) / r.width) * GRID), y = Math.floor(((clientY - r.top) / r.height) * GRID);
    return x >= 0 && y >= 0 && x < GRID && y < GRID ? { x, y } : null;
  }
  setTarget(cell) { this.target = cell; }
  setNames(names) { this.names = names; }

  snapshot(match) {
    this.prev = {
      human: match.human.alive ? match.human.cells.map((c) => ({ ...c })) : null,
      ai: match.ai.alive ? match.ai.cells.map((c) => ({ ...c })) : null,
      drones: match.drones.map((d) => ({ x: d.x, y: d.y })),
    };
  }

  handleEvents(events) {
    const c = this.cell;
    for (const e of events) {
      if (e.type === "eat") {
        const col = COLORS.food[e.food];
        this.burst((e.x + 0.5) * c, (e.y + 0.5) * c, col, e.food === "immortal" ? 60 : 24);
        this.rings.push({ x: (e.x + 0.5) * c, y: (e.y + 0.5) * c, r: 0, col, life: 1 });
      } else if (e.type === "death") {
        const col = COLORS[e.who].body;
        e.cells.forEach((p, i) => i % 1 === 0 && this.burst((p.x + 0.5) * c, (p.y + 0.5) * c, col, 8));
        this.shake = Math.max(this.shake, 14);
      } else if (e.type === "respawn") {
        this.rings.push({ x: (e.x + 0.5) * c, y: (e.y + 0.5) * c, r: 0, col: COLORS[e.who].glow, life: 1 });
      } else if (e.type === "teleport") {
        this.rings.push({ x: (e.x + 0.5) * c, y: (e.y + 0.5) * c, r: 0, col: COLORS.portal[1], life: 1 });
      } else if (e.type === "wall") {
        e.cells.forEach(([x, y]) => this.burst((x + 0.5) * c, (y + 0.5) * c, e.phase === "GHOST" ? COLORS.ghost : COLORS.wall, 5));
      }
    }
  }

  burst(x, y, col, n) {
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2, v = (0.5 + Math.random() * 2.5) * this.cell * 0.06;
      this.particles.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: 1, col, size: 1 + Math.random() * 2.5 });
    }
  }

  draw(match, t, now) {
    const { ctx, cell: c } = this;
    const W = this.canvas.width;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (this.shake > 0.3) {
      ctx.translate((Math.random() - 0.5) * this.shake, (Math.random() - 0.5) * this.shake);
      this.shake *= 0.86;
    }
    // background + grid
    ctx.fillStyle = "#060912";
    ctx.fillRect(-20, -20, W + 40, W + 40);
    ctx.strokeStyle = "rgba(56, 189, 248, 0.07)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= GRID; i++) {
      ctx.beginPath(); ctx.moveTo(i * c, 0); ctx.lineTo(i * c, W); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * c); ctx.lineTo(W, i * c); ctx.stroke();
    }
    // drone lanes
    for (const d of match.drones) {
      ctx.fillStyle = "rgba(250, 204, 21, 0.05)";
      ctx.fillRect(d.lo * c, d.y * c, (d.hi - d.lo + 1) * c, c);
    }
    this.drawPortals(match, now);
    this.drawWalls(match, now);
    this.drawFood(match, now);
    for (const s of match.snakes) this.drawSnake(match, s, t, now);
    this.drawDrones(match, t, now);
    this.drawFog(match, t, now);
    if (this.target) {
      const { ctx, cell: c } = this;
      const x = (this.target.x + 0.5) * c, y = (this.target.y + 0.5) * c;
      ctx.strokeStyle = "rgba(103, 232, 249, 0.85)";
      ctx.lineWidth = Math.max(2, c * 0.07);
      ctx.beginPath(); ctx.arc(x, y, c * (0.36 + 0.05 * Math.sin(now / 150)), 0, Math.PI * 2); ctx.stroke();
    }
    this.drawFx();
    ctx.restore();
  }

  glow(col, blur) { this.ctx.shadowColor = col; this.ctx.shadowBlur = blur * (this.cell / 30); }

  drawPortals(match, now) {
    const { ctx, cell: c } = this;
    match.portals.forEach((p, i) => {
      const cx = (p.x + 0.5) * c, cy = (p.y + 0.5) * c;
      this.glow(COLORS.portal[i], 18);
      ctx.strokeStyle = COLORS.portal[i];
      ctx.lineWidth = c * 0.08;
      for (let k = 0; k < 3; k++) {
        ctx.beginPath();
        const a0 = now / 400 + (k * Math.PI * 2) / 3 + i * Math.PI;
        ctx.arc(cx, cy, c * (0.18 + k * 0.1), a0, a0 + Math.PI * 1.2);
        ctx.stroke();
      }
    });
    ctx.shadowBlur = 0;
  }

  drawWalls(match, now) {
    const { ctx, cell: c } = this;
    for (const st of match.structures) {
      for (const [x, y] of st.cells) {
        if (st.phase === "GHOST") {
          ctx.globalAlpha = 0.18 + 0.08 * Math.sin(now / 150 + x);
          ctx.strokeStyle = COLORS.ghost;
          ctx.setLineDash([c * 0.12, c * 0.1]);
          ctx.lineWidth = c * 0.06;
          ctx.strokeRect(x * c + c * 0.12, y * c + c * 0.12, c * 0.76, c * 0.76);
          ctx.setLineDash([]);
          if (st.timer <= 3) { ctx.globalAlpha = 0.5; ctx.fillStyle = COLORS.wall; ctx.fillRect(x * c + c * 0.3, y * c + c * 0.3, c * 0.4, c * 0.4); }
          ctx.globalAlpha = 1;
          continue;
        }
        const blink = st.phase === "BLINKING" && Math.floor(now / 90) % 2 === 0;
        this.glow(COLORS.wall, blink ? 4 : 14);
        ctx.fillStyle = blink ? "rgba(255, 46, 99, 0.35)" : "rgba(255, 46, 99, 0.85)";
        ctx.fillRect(x * c + c * 0.08, y * c + c * 0.08, c * 0.84, c * 0.84);
        ctx.shadowBlur = 0;
        ctx.strokeStyle = "rgba(255,255,255,0.35)";
        ctx.lineWidth = 1;
        ctx.strokeRect(x * c + c * 0.2, y * c + c * 0.2, c * 0.6, c * 0.6);
      }
    }
  }

  drawFood(match, now) {
    const { ctx, cell: c } = this;
    for (const f of match.foods) {
      if (!match.visibleTo(match.human, f.x, f.y)) continue;
      const cx = (f.x + 0.5) * c, cy = (f.y + 0.5) * c;
      const pulse = 1 + 0.08 * Math.sin(now / 160 + f.x * 3);
      const fading = f.ttl !== null && f.ttl < 12 && Math.floor(now / 120) % 2 === 0;
      ctx.globalAlpha = fading ? 0.35 : 1;
      this.glow(COLORS.food[f.type], f.type === "immortal" ? 26 : 14);
      ctx.fillStyle = COLORS.food[f.type];
      if (f.type === "immortal") {
        this.star(cx, cy, c * 0.42 * pulse, c * 0.18 * pulse, now / 600);
      } else if (f.type === "poison") {
        ctx.beginPath(); ctx.arc(cx, cy, c * 0.28 * pulse, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0; ctx.fillStyle = "#0b1020";
        ctx.font = `${Math.floor(c * 0.36)}px sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText("☠", cx, cy + 1);
      } else {
        ctx.beginPath(); ctx.arc(cx, cy, c * (f.type === "golden" ? 0.32 : 0.27) * pulse, 0, Math.PI * 2); ctx.fill();
      }
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1;
    }
  }

  star(cx, cy, R, r, rot) {
    const ctx = this.ctx;
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const a = rot + (i * Math.PI) / 5 - Math.PI / 2, rad = i % 2 ? r : R;
      ctx.lineTo(cx + Math.cos(a) * rad, cy + Math.sin(a) * rad);
    }
    ctx.closePath();
    ctx.fill();
  }

  lerpCells(s, t) {
    const prev = this.prev?.[s.id];
    return s.cells.map((p, i) => {
      const q = prev?.[i] ?? prev?.[prev.length - 1] ?? p;
      if (!prev || Math.abs(q.x - p.x) + Math.abs(q.y - p.y) > 1) return { x: p.x, y: p.y }; // portal jump: no lerp
      return { x: q.x + (p.x - q.x) * t, y: q.y + (p.y - q.y) * t };
    });
  }

  drawSnake(match, s, t, now) {
    if (!s.alive) return;
    const { ctx, cell: c } = this;
    const col = COLORS[s.id];
    const pts = this.lerpCells(s, t);
    const immortal = s.immortal > 0;
    const ending = immortal && s.immortal <= 8 && Math.floor(now / 100) % 2 === 0;
    // body
    ctx.lineCap = ctx.lineJoin = "round";
    for (let pass = 0; pass < 2; pass++) {
      ctx.beginPath();
      pts.forEach((p, i) => {
        const x = (p.x + 0.5) * c, y = (p.y + 0.5) * c;
        if (i === 0) { ctx.moveTo(x, y); return; }
        const prev = pts[i - 1];
        if (Math.abs(prev.x - p.x) + Math.abs(prev.y - p.y) > 1.01) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      if (pass === 0) {
        const aura = immortal && !ending ? `hsl(${(now / 6) % 360}, 95%, 65%)` : col.glow;
        this.glow(aura, immortal ? 30 : 16);
        ctx.strokeStyle = immortal && !ending ? aura : col.dark;
        ctx.lineWidth = c * (immortal ? 0.86 : 0.74);
      } else {
        ctx.shadowBlur = 0;
        ctx.strokeStyle = col.body;
        ctx.lineWidth = c * 0.52;
      }
      ctx.stroke();
    }
    // head
    const h = pts[0];
    const hx = (h.x + 0.5) * c, hy = (h.y + 0.5) * c;
    this.glow(col.glow, 22);
    ctx.fillStyle = "#f8fafc";
    ctx.beginPath(); ctx.arc(hx, hy, c * 0.36, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = col.body;
    ctx.beginPath(); ctx.arc(hx, hy, c * 0.27, 0, Math.PI * 2); ctx.fill();
    const [dx, dy] = DELTA[s.dir];
    ctx.fillStyle = "#020617";
    for (const side of [-1, 1]) {
      const ex = hx + dx * c * 0.12 + -dy * side * c * 0.12, ey = hy + dy * c * 0.12 + dx * side * c * 0.12;
      ctx.beginPath(); ctx.arc(ex, ey, c * 0.06, 0, Math.PI * 2); ctx.fill();
    }
    // label
    ctx.font = `600 ${Math.floor(c * 0.34)}px "JetBrains Mono", monospace`;
    ctx.textAlign = "center";
    ctx.fillStyle = col.glow;
    ctx.globalAlpha = 0.85;
    ctx.fillText((this.names || { human: "ВЫ", ai: "LAYA" })[s.id], hx, hy - c * 0.62);
    ctx.globalAlpha = 1;
  }

  drawDrones(match, t, now) {
    const { ctx, cell: c } = this;
    match.drones.forEach((d, i) => {
      const p = this.prev?.drones[i] ?? d;
      const x = (p.x + (d.x - p.x) * t + 0.5) * c, y = (p.y + (d.y - p.y) * t + 0.5) * c;
      const [ndx, ndy] = d.plan(match.solid);
      ctx.fillStyle = "rgba(250, 204, 21, 0.18)";
      ctx.fillRect((d.x + ndx) * c + 2, (d.y + ndy) * c + 2, c - 4, c - 4); // where it goes next
      this.glow(COLORS.drone, 20);
      ctx.fillStyle = COLORS.drone;
      ctx.beginPath(); ctx.arc(x, y, c * 0.3, 0, Math.PI * 2); ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = "rgba(250, 204, 21, 0.8)";
      ctx.lineWidth = c * 0.05;
      for (let k = 0; k < 4; k++) {
        const a = now / 60 + (k * Math.PI) / 2;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + Math.cos(a) * c * 0.45, y + Math.sin(a) * c * 0.45); ctx.stroke();
      }
      ctx.fillStyle = "#7f1d1d";
      ctx.beginPath(); ctx.arc(x, y, c * 0.11, 0, Math.PI * 2); ctx.fill();
    });
  }

  drawFog(match, t) {
    this.fogAmt = (this.fogAmt ?? 0) + ((match.fog.active ? 1 : 0) - (this.fogAmt ?? 0)) * 0.06;
    if (this.fogAmt < 0.01) return;
    const { ctx, cell: c } = this, W = this.canvas.width;
    ctx.save();
    ctx.globalAlpha = 0.88 * this.fogAmt;
    ctx.fillStyle = "#03050c";
    ctx.beginPath();
    ctx.rect(-20, -20, W + 40, W + 40);
    if (match.human.alive) {
      const h = this.lerpCells(match.human, t)[0], R = (FOG_RADIUS + 0.5) * c;
      const x = (h.x + 0.5) * c, y = (h.y + 0.5) * c;
      ctx.moveTo(x, y - R); ctx.lineTo(x - R, y); ctx.lineTo(x, y + R); ctx.lineTo(x + R, y); ctx.closePath();
    }
    ctx.fill("evenodd");
    ctx.restore();
  }

  drawFx() {
    const ctx = this.ctx;
    for (const p of this.particles) {
      p.x += p.vx; p.y += p.vy; p.vx *= 0.94; p.vy *= 0.94; p.life -= 0.025;
      ctx.globalAlpha = Math.max(0, p.life);
      ctx.fillStyle = p.col;
      ctx.fillRect(p.x, p.y, p.size, p.size);
    }
    this.particles = this.particles.filter((p) => p.life > 0);
    for (const r of this.rings) {
      r.r += this.cell * 0.12; r.life -= 0.035;
      ctx.globalAlpha = Math.max(0, r.life);
      ctx.strokeStyle = r.col;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(r.x, r.y, r.r, 0, Math.PI * 2); ctx.stroke();
    }
    this.rings = this.rings.filter((r) => r.life > 0);
    ctx.globalAlpha = 1;
  }
}
