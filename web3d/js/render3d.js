// render3d.js — 3D-визуализация SnakeGame на three.js (неоновая кибер-арена).
// Ничего не знает о политике/сервере: получает game + события + коэффициент интерполяции.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { GRID, ACTIONS } from "./game.js";

const HALF = (GRID - 1) / 2;
const UP = new THREE.Vector3(0, 1, 0);
const tmpV = new THREE.Vector3();
const tmpV2 = new THREE.Vector3();

export const PALETTE = {
  grid: new THREE.Color("#00e5ff"),
  headCol: new THREE.Color("#3dffa8"),
  tailCol: new THREE.Color("#0a84ff"),
  wallEdge: new THREE.Color("#ff2a6d"),
  portalA: new THREE.Color("#00e5ff"),
  portalB: new THREE.Color("#b026ff"),
  drone: new THREE.Color("#ff3355"),
  food: {
    standard: new THREE.Color("#ff3358"),
    golden: new THREE.Color("#ffc933"),
    shrink: new THREE.Color("#4df0ff"),
    turbo: new THREE.Color("#ffe14d"),
    freeze: new THREE.Color("#a8f0ff"),
    poison: new THREE.Color("#b44dff"),
    immortal: new THREE.Color("#fff27a"),
  },
};

const cellX = (x) => x - HALF;
const cellZ = (y) => y - HALF;
const lerp = (a, b, t) => a + (b - a) * t;
const clamp01 = (t) => Math.min(1, Math.max(0, t));
const damp = (k, dt) => 1 - Math.exp(-k * dt);
const easeOutBack = (t) => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); };
function lerpAngle(a, b, t) { let d = ((b - a + Math.PI) % (Math.PI * 2)) - Math.PI; if (d < -Math.PI) d += Math.PI * 2; return a + d * t; }

const GLSL_NOISE = /* glsl */ `
float sq(float x){ return x * x; }
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.0-2.0*f);
  return mix(mix(hash12(i), hash12(i+vec2(1,0)), u.x), mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y); }
float fbm(vec2 p){ float v = 0.0, a = 0.5; for(int i=0;i<4;i++){ v += a*vnoise(p); p *= 2.03; a *= 0.5; } return v; }
`;
const WORLD_VS = /* glsl */ `
varying vec3 vW; varying vec2 vUv;
void main(){ vUv = uv; vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;
const OUT_FS = `\n#include <tonemapping_fragment>\n#include <colorspace_fragment>\n`;

function makeGlowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, "rgba(255,255,255,1)");
  grd.addColorStop(0.25, "rgba(255,255,255,0.45)");
  grd.addColorStop(0.6, "rgba(255,255,255,0.1)");
  grd.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// ---------------------------------------------------------------------------
// Система частиц (CPU-интеграция, один draw call)
// ---------------------------------------------------------------------------
class Particles {
  constructor(capacity, scaleUniform) {
    this.cap = capacity;
    this.pos = new Float32Array(capacity * 3);
    this.col = new Float32Array(capacity * 3);
    this.size = new Float32Array(capacity);
    this.alpha = new Float32Array(capacity);
    this.vel = new Float32Array(capacity * 3);
    this.life = new Float32Array(capacity);
    this.maxLife = new Float32Array(capacity);
    this.baseSize = new Float32Array(capacity);
    this.grav = new Float32Array(capacity);
    this.next = 0;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("aColor", new THREE.BufferAttribute(this.col, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("aSize", new THREE.BufferAttribute(this.size, 1).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("aAlpha", new THREE.BufferAttribute(this.alpha, 1).setUsage(THREE.DynamicDrawUsage));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e4);
    const m = new THREE.ShaderMaterial({
      uniforms: { uScale: scaleUniform },
      vertexShader: /* glsl */ `
        attribute vec3 aColor; attribute float aSize; attribute float aAlpha; uniform float uScale;
        varying vec3 vC; varying float vA;
        void main(){ vC = aColor; vA = aAlpha; vec4 mv = modelViewMatrix * vec4(position,1.0);
          gl_Position = projectionMatrix * mv; gl_PointSize = aSize * uScale / max(0.1, -mv.z); }`,
      fragmentShader: /* glsl */ `
        varying vec3 vC; varying float vA;
        void main(){ vec2 c = gl_PointCoord - 0.5; float d = length(c); if (d > 0.5) discard;
          float a = smoothstep(0.5, 0.0, d); gl_FragColor = vec4(vC * (1.0 + 2.0*a*a), vA * a); }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false,
    });
    this.points = new THREE.Points(g, m);
    this.points.frustumCulled = false;
    this.points.renderOrder = 20;
    this.active = 0;
  }

  burst(origin, color, count, o = {}) {
    const speed = o.speed ?? 3, up = o.up ?? 2.5, life = o.life ?? 0.9, size = o.size ?? 0.14, grav = o.gravity ?? -7, spread = o.spread ?? 0.15;
    for (let n = 0; n < count; n++) {
      const i = this.next; this.next = (this.next + 1) % this.cap;
      const th = Math.random() * Math.PI * 2, ph = Math.acos(2 * Math.random() - 1);
      const sp = speed * (0.35 + Math.random() * 0.65);
      this.pos[i * 3] = origin.x + (Math.random() - 0.5) * spread;
      this.pos[i * 3 + 1] = origin.y + (Math.random() - 0.5) * spread;
      this.pos[i * 3 + 2] = origin.z + (Math.random() - 0.5) * spread;
      this.vel[i * 3] = Math.sin(ph) * Math.cos(th) * sp;
      this.vel[i * 3 + 1] = Math.abs(Math.cos(ph)) * sp * 0.6 + up * Math.random();
      this.vel[i * 3 + 2] = Math.sin(ph) * Math.sin(th) * sp;
      const tint = 0.75 + Math.random() * 0.5;
      this.col[i * 3] = color.r * tint; this.col[i * 3 + 1] = color.g * tint; this.col[i * 3 + 2] = color.b * tint;
      this.maxLife[i] = this.life[i] = life * (0.6 + Math.random() * 0.6);
      this.baseSize[i] = size * (0.5 + Math.random());
      this.grav[i] = grav;
    }
  }

  update(dt) {
    let active = 0;
    for (let i = 0; i < this.cap; i++) {
      if (this.life[i] <= 0) { if (this.alpha[i] !== 0) { this.alpha[i] = 0; this.size[i] = 0; } continue; }
      active++;
      this.life[i] -= dt;
      const t = Math.max(0, this.life[i] / this.maxLife[i]);
      this.vel[i * 3 + 1] += this.grav[i] * dt;
      const drag = Math.exp(-1.8 * dt);
      this.vel[i * 3] *= drag; this.vel[i * 3 + 2] *= drag;
      this.pos[i * 3] += this.vel[i * 3] * dt;
      this.pos[i * 3 + 1] = Math.max(0.03, this.pos[i * 3 + 1] + this.vel[i * 3 + 1] * dt);
      this.pos[i * 3 + 2] += this.vel[i * 3 + 2] * dt;
      if (this.pos[i * 3 + 1] <= 0.03) this.vel[i * 3 + 1] *= -0.4;
      this.alpha[i] = t * t;
      this.size[i] = this.baseSize[i] * (0.4 + 0.6 * t);
    }
    this.active = active;
    const a = this.points.geometry.attributes;
    a.position.needsUpdate = a.aColor.needsUpdate = a.aSize.needsUpdate = a.aAlpha.needsUpdate = true;
  }
}

// ---------------------------------------------------------------------------
// Трубка змеи с переиспользуемыми буферами
// ---------------------------------------------------------------------------
const RADIAL = 14;
class TubeChain {
  constructor(material) {
    this.geo = new THREE.BufferGeometry();
    this.cap = 0;
    this.mesh = new THREE.Mesh(this.geo, material);
    this.mesh.frustumCulled = false;
    this.ensure(256);
  }
  ensure(rings) {
    if (rings <= this.cap) return;
    let cap = Math.max(this.cap || 256, 256);
    while (cap < rings) cap *= 2;
    this.cap = cap;
    this.pos = new Float32Array(cap * RADIAL * 3);
    this.nor = new Float32Array(cap * RADIAL * 3);
    this.us = new Float32Array(cap * RADIAL * 2);
    const idx = new Uint32Array((cap - 1) * RADIAL * 6);
    let o = 0;
    for (let i = 0; i < cap - 1; i++) for (let j = 0; j < RADIAL; j++) {
      const a = i * RADIAL + j, b = i * RADIAL + ((j + 1) % RADIAL), c = (i + 1) * RADIAL + j, d = (i + 1) * RADIAL + ((j + 1) % RADIAL);
      idx[o++] = a; idx[o++] = b; idx[o++] = c;
      idx[o++] = b; idx[o++] = d; idx[o++] = c;
    }
    this.geo.setIndex(new THREE.BufferAttribute(idx, 1));
    this.geo.setAttribute("position", new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute("normal", new THREE.BufferAttribute(this.nor, 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute("aUS", new THREE.BufferAttribute(this.us, 2).setUsage(THREE.DynamicDrawUsage));
  }
  /** samples: Float32Array [x,y,z,r,u,s]*n */
  build(samples, n) {
    this.ensure(n);
    const P = this.pos, N = this.nor, U = this.us;
    for (let i = 0; i < n; i++) {
      const i0 = Math.max(0, i - 1) * 6, i1 = Math.min(n - 1, i + 1) * 6, c = i * 6;
      let tx = samples[i1] - samples[i0], tz = samples[i1 + 2] - samples[i0 + 2];
      const tl = Math.hypot(tx, tz) || 1; tx /= tl; tz /= tl;
      const bx = -tz, bz = tx; // B = T x UP ; N = UP
      const cx = samples[c], cy = samples[c + 1], cz = samples[c + 2], r = samples[c + 3];
      for (let j = 0; j < RADIAL; j++) {
        const th = (j / RADIAL) * Math.PI * 2, co = Math.cos(th), si = Math.sin(th);
        const nx = si * bx, ny = co, nz = si * bz;
        const v = (i * RADIAL + j) * 3;
        P[v] = cx + nx * r; P[v + 1] = cy + ny * r; P[v + 2] = cz + nz * r;
        N[v] = nx; N[v + 1] = ny; N[v + 2] = nz;
        const w = (i * RADIAL + j) * 2;
        U[w] = samples[c + 4]; U[w + 1] = samples[c + 5];
      }
    }
    this.geo.attributes.position.needsUpdate = true;
    this.geo.attributes.normal.needsUpdate = true;
    this.geo.attributes.aUS.needsUpdate = true;
    this.geo.setDrawRange(0, Math.max(0, n - 1) * RADIAL * 6);
    this.mesh.visible = n > 1;
  }
}

// ---------------------------------------------------------------------------
export class Renderer3D {
  constructor(container, { quality = "high" } = {}) {
    this.container = container;
    this.time = 0;
    this.cameraMode = "orbit";
    this.autoRotate = true;
    this.lastInteract = -1e9;
    this.shake = 0;
    this.deadAmt = 0;
    this.effect = { amt: 0, color: new THREE.Color(1, 1, 1) };
    this.headYaw = Math.PI / 2;
    this.lumps = [];
    this.gulp = 0;
    this.pulses = [];
    this._clickCb = null;
    this.headWorld = new THREE.Vector3();
    this.headFwd = new THREE.Vector3(1, 0, 0);
    this.lookTarget = new THREE.Vector3(0, 0, 0);

    const renderer = (this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" }));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.95;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0x03040a, 1);
    container.appendChild(renderer.domElement);
    renderer.domElement.classList.add("gl");

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x070414, 0.011);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1200);
    this.camera.position.set(0, 17, 21);

    const pmrem = new THREE.PMREMGenerator(renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(renderer), 0.04).texture;
    pmrem.dispose();

    this.controls = new OrbitControls(this.camera, renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.maxPolarAngle = 1.42;
    this.controls.minDistance = 7;
    this.controls.maxDistance = 70;
    this.controls.autoRotateSpeed = 0.35;
    this.controls.addEventListener("start", () => { this.lastInteract = performance.now(); this.userMovedCamera = true; });

    this.uScale = { value: 500 };
    this.glowTex = makeGlowTexture();

    this._buildLights();
    this._buildEnvironment();
    this._buildArena();
    this._buildFog();
    this._buildSnake();
    this.particles = new Particles(2600, this.uScale);
    this.scene.add(this.particles.points);

    this.foodPool = new Map();
    this.levelGroup = new THREE.Group();
    this.scene.add(this.levelGroup);
    this.drones = [];
    this.portalViews = [];

    this._initComposer();
    this.setQuality(quality);
    this._initPicking();
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(container);
    this.resize();
  }

  // ======================= сцена =======================
  _buildLights() {
    this.scene.add(new THREE.HemisphereLight(0x5a6cff, 0x12001a, 0.9));
    const key = new THREE.DirectionalLight(0xc8d4ff, 1.4);
    key.position.set(-8, 18, 10);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xff3cac, 0.8);
    rim.position.set(10, 6, -14);
    this.scene.add(rim);
    this.headLight = new THREE.PointLight(PALETTE.headCol, 4, 6, 2);
    this.headLight.position.set(0, 1.2, 0);
    this.scene.add(this.headLight);
  }

  _buildEnvironment() {
    // Небо-градиент
    const sky = new THREE.Mesh(
      new THREE.SphereGeometry(600, 32, 16),
      new THREE.ShaderMaterial({
        side: THREE.BackSide, depthWrite: false, fog: false,
        uniforms: { uTime: { value: 0 } },
        vertexShader: `varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
        fragmentShader: /* glsl */ `varying vec3 vP; uniform float uTime;
          void main(){ float h = normalize(vP).y;
            vec3 top = vec3(0.004,0.004,0.018), hor = vec3(0.10,0.025,0.16), bot = vec3(0.0);
            vec3 c = mix(hor, top, smoothstep(0.0, 0.55, h));
            c = mix(c, bot, smoothstep(0.0, -0.25, h));
            c += vec3(0.85,0.12,0.55) * exp(-abs(h - 0.02) * 22.0) * 0.32;
            c += vec3(0.1,0.5,0.9) * exp(-abs(h - 0.05) * 60.0) * 0.12;
            gl_FragColor = vec4(c,1.0); ${OUT_FS} }`,
      })
    );
    sky.renderOrder = -10;
    this.sky = sky;
    this.scene.add(sky);

    // Звёзды
    const N = 1600, sp = new Float32Array(N * 3), ss = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const th = Math.random() * Math.PI * 2, y = 0.05 + Math.random() * 0.95, r = Math.sqrt(1 - y * y);
      sp.set([Math.cos(th) * r * 500, y * 500, Math.sin(th) * r * 500], i * 3);
      ss[i] = Math.random();
    }
    const sg = new THREE.BufferGeometry();
    sg.setAttribute("position", new THREE.BufferAttribute(sp, 3));
    sg.setAttribute("aSeed", new THREE.BufferAttribute(ss, 1));
    this.starMat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 } }, transparent: true, depthWrite: false, fog: false, blending: THREE.AdditiveBlending,
      vertexShader: `attribute float aSeed; uniform float uTime; varying float vA;
        void main(){ vec4 mv = modelViewMatrix * vec4(position,1.0); gl_Position = projectionMatrix * mv;
          vA = 0.35 + 0.65 * (0.5 + 0.5*sin(uTime*(0.6+aSeed*2.0) + aSeed*40.0)); gl_PointSize = 1.0 + aSeed*2.2; }`,
      fragmentShader: `varying float vA; void main(){ float d = length(gl_PointCoord-0.5); if(d>0.5) discard; gl_FragColor = vec4(vec3(0.75,0.85,1.0)*vA, vA*(1.0-d*2.0)); }`,
    });
    this.scene.add(new THREE.Points(sg, this.starMat));

    // Синтвейв-сетка до горизонта
    this.outerMat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 } }, fog: false,
      vertexShader: WORLD_VS,
      fragmentShader: /* glsl */ `varying vec3 vW; uniform float uTime;
        void main(){ vec2 p = vW.xz; float d = length(p);
          vec2 g = abs(fract(p / 2.0 - 0.5) - 0.5) / fwidth(p / 2.0);
          float line = 1.0 - min(min(g.x, g.y), 1.0);
          float fade = exp(-max(d - 12.0, 0.0) * 0.025);
          vec3 c = vec3(0.01,0.005,0.03);
          vec3 lc = mix(vec3(0.9,0.15,0.9), vec3(0.1,0.6,1.0), 0.5 + 0.5*sin(d*0.08 - uTime*0.6));
          c += lc * line * 0.32 * fade;
          c += vec3(0.0,0.4,0.6) * exp(-max(d-11.0,0.0)*0.35) * 0.18;
          c = mix(c, vec3(0.10,0.025,0.16), smoothstep(60.0, 420.0, d));
          gl_FragColor = vec4(c,1.0); ${OUT_FS} }`,
    });
    const outer = new THREE.Mesh(new THREE.PlaneGeometry(1200, 1200), this.outerMat);
    outer.rotation.x = -Math.PI / 2;
    outer.position.y = -0.62;
    this.scene.add(outer);

    // Пыль в воздухе
    const D = 420, dp = new Float32Array(D * 3), ds = new Float32Array(D);
    for (let i = 0; i < D; i++) { dp.set([(Math.random() - 0.5) * 34, Math.random() * 9, (Math.random() - 0.5) * 34], i * 3); ds[i] = Math.random(); }
    const dg = new THREE.BufferGeometry();
    dg.setAttribute("position", new THREE.BufferAttribute(dp, 3));
    dg.setAttribute("aSeed", new THREE.BufferAttribute(ds, 1));
    this.dustMat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 }, uScale: this.uScale }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      vertexShader: `attribute float aSeed; uniform float uTime; uniform float uScale; varying float vA; varying float vS;
        void main(){ vec3 p = position; p.y = mod(p.y + uTime*(0.15+aSeed*0.25), 9.0);
          p.x += sin(uTime*0.3 + aSeed*20.0)*0.6; p.z += cos(uTime*0.25 + aSeed*13.0)*0.6;
          vec4 mv = modelViewMatrix * vec4(p,1.0); gl_Position = projectionMatrix * mv;
          gl_PointSize = (0.03 + aSeed*0.05) * uScale / max(0.1,-mv.z); vS = aSeed;
          vA = smoothstep(0.0,1.0,p.y) * smoothstep(9.0,6.0,p.y) * 0.55; }`,
      fragmentShader: `varying float vA; varying float vS; void main(){ float d = length(gl_PointCoord-0.5); if(d>0.5) discard;
          vec3 c = mix(vec3(0.2,0.9,1.0), vec3(0.9,0.3,1.0), vS); gl_FragColor = vec4(c*1.5, vA*(1.0-d*2.0)); }`,
    });
    this.dust = new THREE.Points(dg, this.dustMat);
    this.scene.add(this.dust);
  }

  _buildArena() {
    // Платформа
    const plat = new THREE.Mesh(
      new RoundedBoxGeometry(GRID + 1.6, 0.6, GRID + 1.6, 3, 0.12),
      new THREE.MeshStandardMaterial({ color: 0x0b0f1f, metalness: 0.8, roughness: 0.35, envMapIntensity: 0.6 })
    );
    plat.position.y = -0.302;
    this.scene.add(plat);

    // Пол арены (шейдер: сетка, световые пятна, волны, туман войны)
    const glows = [], glowCols = [], pulses = [], pulseCols = [];
    for (let i = 0; i < 8; i++) { glows.push(new THREE.Vector4()); glowCols.push(new THREE.Color()); }
    for (let i = 0; i < 4; i++) { pulses.push(new THREE.Vector4(0, 0, -100, 0)); pulseCols.push(new THREE.Color()); }
    this.floorU = {
      uTime: { value: 0 }, uHead: { value: new THREE.Vector2() }, uFog: { value: 0 }, uFogR: { value: 5 },
      uGlow: { value: glows }, uGlowCol: { value: glowCols }, uGlowN: { value: 0 },
      uPulse: { value: pulses }, uPulseCol: { value: pulseCols }, uDead: { value: 0 },
    };
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(GRID, GRID, 1, 1),
      new THREE.ShaderMaterial({
        uniforms: this.floorU,
        vertexShader: WORLD_VS,
        fragmentShader: /* glsl */ `
          uniform float uTime, uFog, uFogR, uDead; uniform vec2 uHead;
          uniform vec4 uGlow[8]; uniform vec3 uGlowCol[8]; uniform int uGlowN;
          uniform vec4 uPulse[4]; uniform vec3 uPulseCol[4];
          varying vec3 vW;
          ${GLSL_NOISE}
          void main(){
            vec2 p = vW.xz; vec2 c = p + ${(GRID / 2).toFixed(1)};
            vec2 id = floor(c); vec2 f = fract(c);
            vec2 g = min(f, 1.0 - f); float edge = min(g.x, g.y);
            float line = 1.0 - smoothstep(0.0, 0.035, edge);
            float halo = exp(-edge * 16.0);
            float chk = mod(id.x + id.y, 2.0);
            vec3 col = mix(vec3(0.011,0.015,0.034), vec3(0.016,0.021,0.046), chk);
            float h = hash12(id);
            col += vec3(0.0,0.035,0.06) * pow(max(0.0, 0.5 + 0.5*sin(uTime*0.7 + h*6.2831)), 12.0);
            float sweep = exp(-sq(fract((p.x + p.y)*0.03 - uTime*0.08) - 0.5) * 300.0);
            vec3 gc = mix(vec3(0.0,0.75,1.0), vec3(1.0,0.12,0.3), uDead);
            col += gc * (line * 0.13 + halo * 0.025) * (1.0 + sweep * 1.4);
            for (int i = 0; i < 8; i++) { if (i >= uGlowN) break;
              vec4 gl = uGlow[i]; float d = length(p - gl.xy);
              col += uGlowCol[i] * gl.w * exp(-d*d / (gl.z*gl.z)) * (0.55 + line * 2.2); }
            for (int i = 0; i < 4; i++) { vec4 pu = uPulse[i]; float age = uTime - pu.z;
              if (pu.w <= 0.0 || age < 0.0 || age > 1.4) continue;
              float d = length(p - pu.xy); float r = age * 8.0;
              col += uPulseCol[i] * exp(-sq((d - r) * 2.2)) * (1.0 - age / 1.4) * pu.w * (0.35 + line * 2.5); }
            if (uFog > 0.001) {
              float dh = length(p - uHead);
              vec2 hc = floor(uHead + ${(GRID / 2).toFixed(1)});
              float man = abs(id.x - hc.x) + abs(id.y - hc.y);
              float vis = 1.0 - step(uFogR + 0.5, man);
              col += vec3(0.0,0.35,0.45) * vis * uFog * (0.03 + 0.025 * sin(uTime*3.0 - dh*2.0)) * (1.0 + line*3.0);
              float dark = max(smoothstep(uFogR - 0.4, uFogR + 2.2, dh), (1.0 - vis) * 0.55);
              col *= mix(1.0, 0.035, dark * uFog);
              float ang = atan(p.y - uHead.y, p.x - uHead.x + 1e-4);
              float ring = exp(-sq((dh - (uFogR + 0.5)) * 5.0)) * step(0.45, fract(ang * 5.0 / 3.14159 + uTime * 0.12));
              col += vec3(0.0,0.9,1.0) * ring * 0.55 * uFog;
            }
            gl_FragColor = vec4(col, 1.0);${OUT_FS}
          }`,
      })
    );
    floor.rotation.x = -Math.PI / 2;
    this.floor = floor;
    this.scene.add(floor);

    // Неоновые бортики (граница поля = смерть)
    this.railMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.0, 0.9, 1.3), toneMapped: false });
    const L = GRID + 0.24, off = GRID / 2 + 0.06;
    const railGeoX = new THREE.BoxGeometry(L, 0.1, 0.1), railGeoZ = new THREE.BoxGeometry(0.1, 0.1, L);
    [[0, -off, railGeoX], [0, off, railGeoX], [-off, 0, railGeoZ], [off, 0, railGeoZ]].forEach(([x, z, g]) => {
      const m = new THREE.Mesh(g, this.railMat); m.position.set(x, 0.05, z); this.scene.add(m);
    });
    // Нижний неоновый кант платформы
    const trimMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.6, 0.15, 1.2), toneMapped: false });
    const tl = GRID + 1.62, toff = (GRID + 1.6) / 2;
    [[0, -toff, new THREE.BoxGeometry(tl, 0.04, 0.04)], [0, toff, new THREE.BoxGeometry(tl, 0.04, 0.04)],
     [-toff, 0, new THREE.BoxGeometry(0.04, 0.04, tl)], [toff, 0, new THREE.BoxGeometry(0.04, 0.04, tl)]].forEach(([x, z, g]) => {
      const m = new THREE.Mesh(g, trimMat); m.position.set(x, -0.58, z); this.scene.add(m);
    });
    // Пилоны по углам
    const pyGeo = new RoundedBoxGeometry(0.5, 1.6, 0.5, 2, 0.06), pyMat = new THREE.MeshStandardMaterial({ color: 0x111633, metalness: 0.9, roughness: 0.3 });
    const capGeo = new THREE.BoxGeometry(0.52, 0.08, 0.52), capMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0, 0.85, 1.2), toneMapped: false });
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
      const g = new THREE.Group();
      const b = new THREE.Mesh(pyGeo, pyMat); b.position.y = 0.5; g.add(b);
      const c1 = new THREE.Mesh(capGeo, capMat); c1.position.y = 1.32; g.add(c1);
      const c2 = new THREE.Mesh(capGeo, capMat); c2.position.y = 0.2; c2.scale.set(1, 0.5, 1); g.add(c2);
      g.position.set(sx * (GRID / 2 + 0.55), 0, sz * (GRID / 2 + 0.55));
      this.scene.add(g);
    }

    // Общие ресурсы для уровня
    this.wallGeo = new RoundedBoxGeometry(0.9, 1.0, 0.9, 3, 0.08);
    this.wallMat = new THREE.MeshStandardMaterial({ color: 0x151a36, metalness: 0.75, roughness: 0.28, envMapIntensity: 0.8 });
    this.wallEdgeGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(0.92, 1.02, 0.92));
    this.wallEdgeMat = new THREE.LineBasicMaterial({ color: PALETTE.wallEdge.clone().multiplyScalar(2.0), toneMapped: false });
    this.wallCapGeo = new THREE.PlaneGeometry(0.62, 0.62);
    this.wallCapMat = new THREE.MeshBasicMaterial({ color: PALETTE.wallEdge.clone().multiplyScalar(0.9), toneMapped: false, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false });
    this.decalGeo = new THREE.PlaneGeometry(1, 1);
  }

  _decal(color, size, intensity = 1) {
    const m = new THREE.Mesh(this.decalGeo, new THREE.MeshBasicMaterial({
      map: this.glowTex, color: color.clone().multiplyScalar(intensity), transparent: true, depthWrite: false,
      blending: THREE.AdditiveBlending, toneMapped: false,
    }));
    m.rotation.x = -Math.PI / 2;
    m.scale.set(size, size, 1);
    m.position.y = 0.012;
    m.renderOrder = 2;
    return m;
  }

  _buildFog() {
    this.fogGroup = new THREE.Group();
    this.fogLayers = [];
    const heights = [0.1, 0.42, 0.8, 1.25, 1.85], alphas = [0.55, 0.42, 0.42, 0.5, 0.86];
    heights.forEach((h, i) => {
      const u = { uTime: { value: 0 }, uFog: { value: 0 }, uFogR: { value: 5 }, uHead: { value: new THREE.Vector2() }, uLayer: { value: i }, uAlpha: { value: alphas[i] } };
      const m = new THREE.Mesh(new THREE.PlaneGeometry(GRID + 3, GRID + 3), new THREE.ShaderMaterial({
        uniforms: u, transparent: true, depthWrite: false, fog: false,
        vertexShader: WORLD_VS,
        fragmentShader: /* glsl */ `uniform float uTime, uFog, uFogR, uLayer, uAlpha; uniform vec2 uHead; varying vec3 vW;
          ${GLSL_NOISE}
          void main(){ vec2 p = vW.xz; float dh = length(p - uHead);
            float m = smoothstep(uFogR - 0.3 + uLayer*0.15, uFogR + 2.4, dh);
            float edge = 1.0 - smoothstep(${(GRID / 2 + 0.3).toFixed(2)}, ${(GRID / 2 + 1.5).toFixed(2)}, max(abs(p.x), abs(p.y)));
            float n = fbm(p * 0.28 + vec2(uTime*0.05 + uLayer*3.1, uTime*0.035 - uLayer*1.7));
            float a = m * edge * uFog * uAlpha * (0.5 + 0.75*n);
            vec3 c = mix(vec3(0.004,0.006,0.016), vec3(0.03,0.05,0.12), n*n);
            gl_FragColor = vec4(c, clamp(a, 0.0, 0.97)); ${OUT_FS} }`,
      }));
      m.rotation.x = -Math.PI / 2;
      m.position.y = h;
      m.renderOrder = 10 + i;
      this.fogLayers.push(m);
      this.fogGroup.add(m);
    });
    this.fogGroup.visible = false;
    this.fogAmt = 0;
    this.scene.add(this.fogGroup);
  }

  _buildSnake() {
    this.snakeU = { uTime: { value: 0 }, uHeadCol: { value: PALETTE.headCol.clone() }, uTailCol: { value: PALETTE.tailCol.clone() }, uDead: { value: 0 }, uEffect: { value: 0 }, uEffectCol: { value: new THREE.Color(1, 1, 1) }, uImmortal: { value: 0 }, uWarn: { value: 0 } };
    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.35, metalness: 0.2, envMapIntensity: 0.3 });
    mat.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, this.snakeU);
      sh.vertexShader = "attribute vec2 aUS;\nvarying vec2 vUS;\n" + sh.vertexShader.replace("#include <begin_vertex>", "#include <begin_vertex>\nvUS = aUS;");
      sh.fragmentShader = "uniform float uTime; uniform vec3 uHeadCol; uniform vec3 uTailCol; uniform float uDead; uniform float uEffect; uniform vec3 uEffectCol; uniform float uImmortal; uniform float uWarn; float sq(float x){ return x * x; }\n" +
        "vec3 hsv2rgb(vec3 c){ vec3 p = abs(fract(c.xxx + vec3(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0); return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y); }\nvarying vec2 vUS;\n" +
        sh.fragmentShader
          .replace("#include <color_fragment>", /* glsl */ `#include <color_fragment>
            vec3 sc = mix(uHeadCol, uTailCol, smoothstep(0.0, 1.0, vUS.x));
            sc = mix(sc, vec3(1.0, 0.1, 0.22), uDead);
            float groove = 1.0 - 0.45 * exp(-sq((fract(vUS.y) - 0.5) * 7.0));
            diffuseColor.rgb = sc * 0.42 * groove;`)
          .replace("#include <emissivemap_fragment>", /* glsl */ `
            float band = pow(max(0.0, 0.5 + 0.5 * sin(vUS.y * 1.6 - uTime * (6.0 + uEffect * 8.0))), 12.0);
            float scale = 0.5 + 0.5 * cos(vUS.y * 6.2831);
            vec3 ec = mix(sc, uEffectCol, uEffect * 0.7);
            totalEmissiveRadiance = (sc * (0.1 + 0.08 * scale) + ec * 0.95 * band * (1.0 - vUS.x * 0.7)) * groove;
            // Бессмертие: радужный перелив по телу + блики; в последние шаги — тревожное мигание
            vec3 rainbow = hsv2rgb(vec3(fract(vUS.y * 0.09 - uTime * 0.45), 0.95, 1.0));
            float shimmer = 0.25 + 0.75 * pow(max(0.0, 0.5 + 0.5 * sin(vUS.y * 2.6 - uTime * 11.0)), 6.0);
            float warn = 1.0 - uWarn * step(0.5, fract(uTime * 5.0)) * 0.85;
            totalEmissiveRadiance = mix(totalEmissiveRadiance, rainbow * shimmer * 0.7 * warn, uImmortal * 0.85);
            diffuseColor.rgb = mix(diffuseColor.rgb, rainbow * 0.25, uImmortal * 0.6);`);
    };
    this.snakeMat = mat;
    this.snakeGroup = new THREE.Group();
    this.scene.add(this.snakeGroup);
    this.tubes = [];
    this.samples = new Float32Array(6 * 4096);

    // Голова
    const head = (this.head = new THREE.Group());
    this.headMat = new THREE.MeshStandardMaterial({ color: PALETTE.headCol, emissive: PALETTE.headCol, emissiveIntensity: 0.22, roughness: 0.3, metalness: 0.2, envMapIntensity: 0.3 });
    const skull = new THREE.Mesh(new THREE.SphereGeometry(0.4, 40, 28), this.headMat);
    skull.scale.set(1, 0.78, 1.2);
    head.add(skull);
    this.eyeMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(2.6, 3.0, 3.0), toneMapped: false });
    const eyeGeo = new THREE.SphereGeometry(0.075, 16, 12);
    for (const s of [-1, 1]) {
      const e = new THREE.Mesh(eyeGeo, this.eyeMat);
      e.position.set(s * 0.17, 0.15, 0.3);
      e.scale.set(1, 0.8, 0.7);
      head.add(e);
    }
    // «Визор»-гребень
    const crest = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.05, 0.5), new THREE.MeshBasicMaterial({ color: PALETTE.headCol.clone().multiplyScalar(2.2), toneMapped: false }));
    crest.position.set(0, 0.31, -0.02);
    this.crestMat = crest.material;
    head.add(crest);
    this.headDecal = this._decal(PALETTE.headCol, 2.4, 0.22);
    this.immortalAmt = 0;
    this.aura = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glowTex, color: new THREE.Color(1.6, 1.3, 0.5), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false }));
    this.aura.scale.setScalar(2.2);
    this.aura.visible = false;
    this.scene.add(this.aura);
    this.scene.add(this.headDecal);
    this.snakeGroup.add(head);

    // Маркер места аварии
    this.crashMarker = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: new THREE.Color(3, 0.2, 0.4), wireframe: true, toneMapped: false }));
    this.crashMarker.visible = false;
    this.scene.add(this.crashMarker);
  }

  _initComposer() {
    const rt = new THREE.WebGLRenderTarget(2, 2, { type: THREE.HalfFloatType, samples: 4 });
    this.composer = new EffectComposer(this.renderer, rt);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.72, 0.42, 0.86);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
  }

  setQuality(q) {
    this.quality = q;
    this.bloomOn = q === "high";
    const dpr = window.devicePixelRatio || 1;
    this.renderer.setPixelRatio(q === "high" ? Math.min(dpr, 2) : Math.min(dpr, 1));
    this.dust && (this.dust.visible = q === "high");
    this.fogLayers.forEach((l, i) => (l.visible = q === "high" || i % 2 === 0 || i === this.fogLayers.length - 1));
    this.resize();
  }
  setBloom(on) { this.bloomOn = on; }

  setCameraMode(m) {
    if (this.cameraMode === m) return;
    if (m === "orbit") this.controls.target.copy(this.lookTarget);
    this.cameraMode = m;
  }
  setAutoRotate(on) { this.autoRotate = on; }
  onCellClick(cb) { this._clickCb = cb; }

  _initPicking() {
    const el = this.renderer.domElement, ray = new THREE.Raycaster(), plane = new THREE.Plane(UP, 0), ndc = new THREE.Vector2(), hit = new THREE.Vector3();
    let down = null;
    el.addEventListener("pointerdown", (e) => (down = { x: e.clientX, y: e.clientY }));
    el.addEventListener("pointerup", (e) => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 6 || e.button !== 0) return;
      const r = el.getBoundingClientRect();
      ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ndc, this.camera);
      if (!ray.ray.intersectPlane(plane, hit)) return;
      const gx = Math.round(hit.x + HALF), gy = Math.round(hit.z + HALF);
      if (gx >= 0 && gx < GRID && gy >= 0 && gy < GRID && this._clickCb) this._clickCb(gx, gy, e);
    });
  }

  resize() {
    const w = Math.max(1, this.container.clientWidth), h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    const pr = this.renderer.getPixelRatio();
    this.composer.setPixelRatio(pr);
    this.composer.setSize(w, h);
    this.bloom.resolution.set(w * pr * 0.5, h * pr * 0.5);
    this.uScale.value = (h * pr) / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2));
    this.viewW = w; this.viewH = h;
    if (!this.userMovedCamera && this.cameraMode === "orbit") {
      // Подогнать дистанцию орбиты под соотношение сторон, чтобы арена влезала целиком
      const dist = 27 * Math.pow(Math.max(1, 1.25 / this.camera.aspect), 0.9);
      const off = this.camera.position.clone().sub(this.controls.target).setLength(dist);
      this.camera.position.copy(this.controls.target).add(off);
    }
  }

  // ======================= уровень =======================
  _clearLevel() {
    const shared = new Set([this.wallMat, this.wallEdgeMat, this.wallCapMat, this.wallGeo, this.wallEdgeGeo, this.wallCapGeo, this.decalGeo]);
    this.levelGroup.traverse((o) => {
      if (o.material && !shared.has(o.material)) o.material.dispose?.();
      if (o.geometry && !shared.has(o.geometry)) o.geometry.dispose?.();
    });
    this.linkU = null;
    this.levelGroup.clear();
    this.drones.forEach((d) => this.scene.remove(d.light));
    this.drones = [];
    this.portalViews = [];
    this.wallViews = [];
    (this.structViews || []).forEach((sv) => sv.labelTex.dispose());
    this.structViews = [];
  }

  _buildLevel(game) {
    this._clearLevel();
    const t0 = this.time;
    // Стены: каждая структура — свои материалы (фазы SOLID / BLINKING / GHOST, RULES_V9 §2)
    let wi = 0;
    for (const s of game.structures) {
      const mats = { box: this.wallMat.clone(), edge: this.wallEdgeMat.clone(), cap: this.wallCapMat.clone() };
      mats.box.transparent = true;
      const labelCanvas = document.createElement("canvas"); labelCanvas.width = 128; labelCanvas.height = 64;
      const labelTex = new THREE.CanvasTexture(labelCanvas); labelTex.colorSpace = THREE.SRGBColorSpace;
      const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: labelTex, transparent: true, depthWrite: false, depthTest: false, toneMapped: false }));
      label.scale.set(1.4, 0.7, 1);
      label.renderOrder = 30;
      const cx = s.cells.reduce((a, c) => a + c.x, 0) / s.cells.length, cy = s.cells.reduce((a, c) => a + c.y, 0) / s.cells.length;
      label.position.set(cellX(cx), 1.75, cellZ(cy));
      label.visible = false;
      this.levelGroup.add(label);
      const sv = { s, mats, ghost: s.phase === "GHOST" ? 1 : 0, groups: [], label, labelCanvas, labelTex, labelText: "" };
      for (const o of s.cells) {
        const g = new THREE.Group();
        const box = new THREE.Mesh(this.wallGeo, mats.box); box.position.y = 0.5; g.add(box);
        const edges = new THREE.LineSegments(this.wallEdgeGeo, mats.edge); edges.position.y = 0.5; g.add(edges);
        const cap = new THREE.Mesh(this.wallCapGeo, mats.cap); cap.rotation.x = -Math.PI / 2; cap.position.y = 1.012; g.add(cap);
        const dec = this._decal(PALETTE.wallEdge, 1.9, 0.2); g.add(dec);
        g.position.set(cellX(o.x), 0, cellZ(o.y));
        this.levelGroup.add(g);
        this.wallViews.push({ g, born: t0 + 0.04 * wi++, dec });
        sv.groups.push(g);
      }
      this.structViews.push(sv);
    }
    // Порталы
    if (game.settings.portals && game.portals.length >= 2) {
      game.portals.forEach((p, i) => this.portalViews.push(this._makePortal(p, i === 0 ? PALETTE.portalA : PALETTE.portalB, i)));
      this._makePortalLink(game.portals[0], game.portals[1]);
    }
    // Дроны
    if (game.settings.patrols) game.patrols.forEach((p, i) => this.drones.push(this._makeDrone(p, i)));
  }

  _makePortal(p, color, idx) {
    const g = new THREE.Group();
    g.position.set(cellX(p.x), 0, cellZ(p.y));
    const u = { uTime: { value: 0 }, uColor: { value: color.clone() }, uFlash: { value: 0 }, uScale: this.uScale };
    const disc = new THREE.Mesh(new THREE.CircleGeometry(0.66, 64), new THREE.ShaderMaterial({
      uniforms: u, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false,
      vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: /* glsl */ `varying vec2 vUv; uniform float uTime; uniform vec3 uColor; uniform float uFlash;
        void main(){ vec2 q = vUv * 2.0 - 1.0; float r = length(q); if (r > 1.0) discard; float a = atan(q.y, q.x + 1e-4);
          float arms = 0.5 + 0.5 * sin(a * 3.0 + log(r + 0.05) * 6.0 - uTime * 5.0);
          arms = pow(max(arms, 0.0), 3.0);
          float core = pow(max(1.0 - r, 0.0), 3.0);
          float rim = exp(-(r - 0.93) * (r - 0.93) * 324.0);
          vec3 c = uColor * (arms * (1.0 - r) * 2.2 + rim * 3.0 + core * 1.5) + vec3(1.0) * core * core * 1.2;
          c *= 1.0 + uFlash * 3.0;
          gl_FragColor = vec4(c, smoothstep(1.0, 0.9, r)); }`,
    }));
    disc.rotation.x = -Math.PI / 2;
    disc.position.y = 0.02;
    disc.renderOrder = 3;
    g.add(disc);

    const ringMat = new THREE.MeshBasicMaterial({ color: color.clone().multiplyScalar(2.5), toneMapped: false });
    const r1 = new THREE.Mesh(new THREE.TorusGeometry(0.62, 0.045, 12, 64), ringMat);
    r1.rotation.x = -Math.PI / 2; r1.position.y = 0.08; g.add(r1);
    const r2 = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.025, 8, 48), ringMat);
    r2.position.y = 0.55; g.add(r2);

    const col = new THREE.Mesh(new THREE.CylinderGeometry(0.58, 0.64, 2.2, 32, 1, true), new THREE.ShaderMaterial({
      uniforms: u, transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, toneMapped: false,
      vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: `varying vec2 vUv; uniform float uTime; uniform vec3 uColor; uniform float uFlash;
        void main(){ float h = vUv.y; float streak = 0.5 + 0.5*sin(vUv.x*62.83 + uTime*3.0 + h*8.0);
          float a = pow(max(1.0 - h, 0.0), 2.2) * (0.18 + 0.25*streak) * (1.0 + uFlash*2.0);
          gl_FragColor = vec4(uColor * 1.6, a); }`,
    }));
    col.position.y = 1.1;
    g.add(col);

    const N = 70, ph = new Float32Array(N), pp = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) ph[i] = Math.random();
    const pg = new THREE.BufferGeometry();
    pg.setAttribute("position", new THREE.BufferAttribute(pp, 3));
    pg.setAttribute("aPhase", new THREE.BufferAttribute(ph, 1));
    const pts = new THREE.Points(pg, new THREE.ShaderMaterial({
      uniforms: u, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false,
      vertexShader: `attribute float aPhase; uniform float uTime; uniform float uScale; varying float vA;
        void main(){ float h = fract(uTime*0.35 + aPhase); float ang = aPhase*43.0 + uTime*2.4 + h*5.0; float r = 0.6*(1.0 - h*0.55);
          vec3 p = vec3(cos(ang)*r, h*2.0, sin(ang)*r); vec4 mv = modelViewMatrix * vec4(p,1.0);
          gl_Position = projectionMatrix * mv; gl_PointSize = (0.05 + 0.06*(1.0-h)) * uScale / max(0.1,-mv.z);
          vA = (1.0 - h) * smoothstep(0.0, 0.08, h); }`,
      fragmentShader: `varying float vA; uniform vec3 uColor; void main(){ float d = length(gl_PointCoord-0.5); if(d>0.5) discard; gl_FragColor = vec4(uColor*2.5, vA*(1.0-d*2.0)); }`,
    }));
    pts.frustumCulled = false;
    g.add(pts);
    g.add(this._decal(color, 2.6, 0.32));
    this.levelGroup.add(g);
    return { g, u, r1, r2, idx, color };
  }

  _makePortalLink(a, b) {
    const A = new THREE.Vector3(cellX(a.x), 0.3, cellZ(a.y)), B = new THREE.Vector3(cellX(b.x), 0.3, cellZ(b.y));
    const mid = A.clone().add(B).multiplyScalar(0.5); mid.y = 3 + A.distanceTo(B) * 0.22;
    const curve = new THREE.QuadraticBezierCurve3(A, mid, B);
    this.linkU = { uTime: { value: 0 }, uA: { value: PALETTE.portalA }, uB: { value: PALETTE.portalB }, uFlash: { value: 0 } };
    const m = new THREE.Mesh(new THREE.TubeGeometry(curve, 96, 0.03, 6, false), new THREE.ShaderMaterial({
      uniforms: this.linkU, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false,
      vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: `varying vec2 vUv; uniform float uTime; uniform vec3 uA; uniform vec3 uB; uniform float uFlash;
        void main(){ float d = smoothstep(0.55, 0.75, fract(vUv.x*14.0 - uTime*1.4)); float e = smoothstep(0.0,0.08,vUv.x)*smoothstep(1.0,0.92,vUv.x);
          gl_FragColor = vec4(mix(uA, uB, vUv.x) * 2.0, (0.06 + 0.32*d + uFlash*0.6) * e); }`,
    }));
    this.levelGroup.add(m);
  }

  _makeDrone(p, i) {
    const g = new THREE.Group();
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x1a1d2e, metalness: 0.9, roughness: 0.25 });
    const glowMat = new THREE.MeshBasicMaterial({ color: PALETTE.drone.clone().multiplyScalar(2.6), toneMapped: false });
    const body = new THREE.Mesh(new THREE.SphereGeometry(0.3, 28, 18), bodyMat); body.scale.set(1, 0.55, 1); g.add(body);
    const dome = new THREE.Mesh(new THREE.SphereGeometry(0.16, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0x220008, emissive: PALETTE.drone, emissiveIntensity: 1.2, roughness: 0.1 }));
    dome.position.y = 0.1; g.add(dome);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.43, 0.035, 10, 48), glowMat); ring.rotation.x = Math.PI / 2; g.add(ring);
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.07, 12, 8), new THREE.MeshBasicMaterial({ color: new THREE.Color(3, 2.2, 2.2), toneMapped: false }));
    eye.position.set(0, -0.02, 0.29); g.add(eye);
    const rotors = new THREE.Group();
    const armGeo = new THREE.BoxGeometry(0.9, 0.03, 0.05), rotorGeo = new THREE.CircleGeometry(0.15, 20);
    const rotorMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.2, 0.3, 0.4), transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthWrite: false, toneMapped: false });
    [Math.PI / 4, -Math.PI / 4].forEach((a) => { const arm = new THREE.Mesh(armGeo, bodyMat); arm.rotation.y = a; arm.position.y = 0.05; g.add(arm); });
    const rotorMeshes = [];
    for (const [x, z] of [[0.32, 0.32], [-0.32, 0.32], [0.32, -0.32], [-0.32, -0.32]]) {
      const r = new THREE.Mesh(rotorGeo, rotorMat); r.rotation.x = -Math.PI / 2; r.position.set(x, 0.09, z); rotors.add(r); rotorMeshes.push(r);
    }
    g.add(rotors);
    // Конус сканера
    const coneGeo = new THREE.ConeGeometry(0.75, 2.3, 28, 1, true); coneGeo.translate(0, -1.15, 0);
    const coneU = { uTime: { value: 0 } };
    const cone = new THREE.Mesh(coneGeo, new THREE.ShaderMaterial({
      uniforms: coneU, transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, toneMapped: false,
      vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: `varying vec2 vUv; uniform float uTime; void main(){ float a = pow(max(vUv.y, 0.0), 1.6) * (0.55 + 0.45*sin(vUv.y*40.0 - uTime*10.0));
        gl_FragColor = vec4(vec3(1.0,0.12,0.25)*1.4, a*0.22); }`,
    }));
    cone.rotation.x = -(Math.PI / 2 - 0.5);
    cone.position.set(0, -0.02, 0.28);
    g.add(cone);
    g.position.set(cellX(p.x), 0.85, cellZ(p.y));
    this.levelGroup.add(g);

    // Маркеры опасных клеток (текущая + следующая — как в danger-сенсорах)
    const mkU = { uTime: { value: 0 } };
    const mkMat = (strength) => new THREE.ShaderMaterial({
      uniforms: { ...mkU, uS: { value: strength } }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false,
      vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: `varying vec2 vUv; uniform float uTime; uniform float uS; void main(){ vec2 q = abs(vUv - 0.5); float e = max(q.x, q.y);
        float border = smoothstep(0.40, 0.46, e) * (1.0 - smoothstep(0.47, 0.5, e));
        float fill = (1.0 - smoothstep(0.0, 0.46, e)) * 0.12;
        float pulse = 0.6 + 0.4*sin(uTime*8.0);
        gl_FragColor = vec4(vec3(1.0,0.1,0.25)*2.0, (border + fill) * pulse * uS); }`,
    });
    const curMk = new THREE.Mesh(this.decalGeo, mkMat(0.9)); curMk.rotation.x = -Math.PI / 2; curMk.position.y = 0.016;
    const nextMk = new THREE.Mesh(this.decalGeo, mkMat(0.5)); nextMk.rotation.x = -Math.PI / 2; nextMk.position.y = 0.016;
    curMk.renderOrder = nextMk.renderOrder = 4;
    this.levelGroup.add(curMk, nextMk);
    // Маршрут (h / v / прямоугольный circuit)
    const segs = p.kind === "h" ? [[[p.lo, p.y], [p.hi, p.y]]] : p.kind === "v" ? [[[p.x, p.lo], [p.x, p.hi]]]
      : (p.waypoints || []).map((a, j, w) => [a, w[(j + 1) % w.length]]);
    for (const [[ax, ay], [bx, by]] of segs) {
      const len = Math.abs(bx - ax) + Math.abs(by - ay), horiz = ay === by;
      const route = new THREE.Mesh(new THREE.PlaneGeometry(horiz ? len : 0.06, horiz ? 0.06 : len), new THREE.MeshBasicMaterial({ color: new THREE.Color(0.9, 0.08, 0.2), transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false }));
      route.rotation.x = -Math.PI / 2;
      route.position.set(cellX((ax + bx) / 2), 0.011, cellZ((ay + by) / 2));
      this.levelGroup.add(route);
    }
    // drone_path: следующие 6 клеток дрона — ровно то, что видит модель
    const pathMk = [];
    for (let j = 0; j < 6; j++) {
      const m = this._decal(PALETTE.drone, 0.55 - j * 0.04, 0.55 - j * 0.07);
      m.position.y = 0.014;
      this.levelGroup.add(m);
      pathMk.push(m);
    }
    const decal = this._decal(PALETTE.drone, 2.0, 0.28);
    this.levelGroup.add(decal);
    const light = new THREE.PointLight(PALETTE.drone, 4, 4.5, 2);
    this.scene.add(light);
    const yaw = Math.atan2(p.dx || 0, p.dy || 0);
    return { g, rotorMeshes, coneU, mkU: [curMk.material.uniforms, nextMk.material.uniforms], curMk, nextMk, pathMk, decal, light, yaw, i };
  }

  // ======================= еда =======================
  _foodShared() {
    if (this._fs) return this._fs;
    const pts = [];
    const N = 28;
    for (let i = 0; i <= N; i++) {
      const a = (i / N) * Math.PI;
      const r = Math.sin(a) * (1 + 0.12 * Math.cos(a));
      let y = -Math.cos(a) * 0.92;
      y += 0.16 * Math.exp(-Math.pow(a / 0.45, 2));
      y -= 0.28 * Math.exp(-Math.pow((Math.PI - a) / 0.5, 2));
      pts.push(new THREE.Vector2(Math.max(r, 0.0001) * 0.5, y * 0.5));
    }
    const apple = new THREE.LatheGeometry(pts, 40);
    apple.computeVertexNormals();
    const stem = new THREE.CylinderGeometry(0.025, 0.04, 0.24, 8); stem.translate(0, 0.42, 0);
    const leaf = new THREE.SphereGeometry(0.5, 14, 8); leaf.scale(0.2, 0.03, 0.09); leaf.rotateZ(-0.45); leaf.translate(0.12, 0.44, 0);

    const spike = new THREE.IcosahedronGeometry(0.28, 1);
    const base = new THREE.IcosahedronGeometry(1, 0).attributes.position;
    const dirs = []; for (let i = 0; i < base.count; i++) dirs.push(new THREE.Vector3().fromBufferAttribute(base, i).normalize());
    const sp = spike.attributes.position;
    for (let i = 0; i < sp.count; i++) {
      tmpV.fromBufferAttribute(sp, i);
      const d = tmpV.clone().normalize();
      if (dirs.some((q) => q.dot(d) > 0.999)) { tmpV.copy(d).multiplyScalar(0.5); sp.setXYZ(i, tmpV.x, tmpV.y, tmpV.z); }
    }
    spike.computeVertexNormals();

    const crystal = new THREE.OctahedronGeometry(0.27, 0); crystal.scale(1, 1.6, 1);
    const bolt = new THREE.Shape();
    bolt.moveTo(0.06, 0.5); bolt.lineTo(-0.22, 0.02); bolt.lineTo(-0.03, 0.02); bolt.lineTo(-0.1, -0.5); bolt.lineTo(0.23, 0.07); bolt.lineTo(0.03, 0.07); bolt.closePath();
    const boltGeo = new THREE.ExtrudeGeometry(bolt, { depth: 0.1, bevelEnabled: true, bevelThickness: 0.03, bevelSize: 0.025, bevelSegments: 2 }); boltGeo.center();
    const starShape = new THREE.Shape();
    for (let i = 0; i < 10; i++) {
      const a = Math.PI / 2 + (i * Math.PI) / 5, r = i % 2 ? 0.17 : 0.42;
      if (i) starShape.lineTo(Math.cos(a) * r, Math.sin(a) * r); else starShape.moveTo(Math.cos(a) * r, Math.sin(a) * r);
    }
    starShape.closePath();
    const starGeo = new THREE.ExtrudeGeometry(starShape, { depth: 0.1, bevelEnabled: true, bevelThickness: 0.04, bevelSize: 0.03, bevelSegments: 2 }); starGeo.center();
    const arm = new THREE.BoxGeometry(0.72, 0.055, 0.055), twig = new THREE.BoxGeometry(0.17, 0.04, 0.04);
    const ring = new THREE.PlaneGeometry(1.25, 1.25);
    const mats = {
      standard: new THREE.MeshPhysicalMaterial({ color: PALETTE.food.standard, emissive: PALETTE.food.standard, emissiveIntensity: 0.35, roughness: 0.22, clearcoat: 1, clearcoatRoughness: 0.1 }),
      golden: new THREE.MeshStandardMaterial({ color: 0xffd257, metalness: 1, roughness: 0.15, emissive: 0xffa400, emissiveIntensity: 0.55, envMapIntensity: 1.6 }),
      stem: new THREE.MeshStandardMaterial({ color: 0x4a2b12, roughness: 0.8 }),
      leaf: new THREE.MeshStandardMaterial({ color: 0x29d16b, emissive: 0x0d7a34, emissiveIntensity: 0.6, roughness: 0.5 }),
      poison: new THREE.MeshStandardMaterial({ color: 0x2c0848, emissive: 0x8a00ff, emissiveIntensity: 0.7, roughness: 0.35, metalness: 0.4, flatShading: true }),
      toxic: new THREE.MeshBasicMaterial({ color: new THREE.Color(0.3, 1.5, 0.2), toneMapped: false }),
      shrink: new THREE.MeshPhysicalMaterial({ color: 0x7ff6ff, emissive: 0x00bfff, emissiveIntensity: 0.45, roughness: 0.05, metalness: 0.1, flatShading: true, transparent: true, opacity: 0.88 }),
      shrinkCore: new THREE.MeshBasicMaterial({ color: new THREE.Color(1.3, 0.6, 1.7), toneMapped: false }),
      turbo: new THREE.MeshStandardMaterial({ color: 0xffe14d, emissive: 0xffb000, emissiveIntensity: 0.65, roughness: 0.3, metalness: 0.3 }),
      turboRing: new THREE.MeshBasicMaterial({ color: new THREE.Color(1.5, 1.15, 0.2), toneMapped: false }),
      freeze: new THREE.MeshStandardMaterial({ color: 0xd8fbff, emissive: 0x5cd6ff, emissiveIntensity: 0.55, roughness: 0.2, metalness: 0.2 }),
      white: new THREE.MeshBasicMaterial({ color: new THREE.Color(1.3, 1.5, 1.7), toneMapped: false }),
      immortal: new THREE.MeshStandardMaterial({ color: 0xfff1a0, emissive: 0xffc830, emissiveIntensity: 0.55, metalness: 0.9, roughness: 0.18, envMapIntensity: 1.4 }),
      rainbowRing: new THREE.MeshBasicMaterial({ color: new THREE.Color(1.4, 1.2, 0.4), toneMapped: false }),
    };
    this._fs = { apple, stem, leaf, spike, crystal, boltGeo, starGeo, arm, twig, ring, mats,
      torus: new THREE.TorusGeometry(0.36, 0.018, 8, 48), small: new THREE.SphereGeometry(0.05, 10, 8), core: new THREE.IcosahedronGeometry(0.1, 0), octa: new THREE.OctahedronGeometry(0.12, 0) };
    return this._fs;
  }

  _makeFood(f) {
    const S = this._foodShared(), M = S.mats;
    const root = new THREE.Group(), model = new THREE.Group(), spin = [];
    root.add(model);
    switch (f.type) {
      case "standard":
      case "golden": {
        const m = f.type === "golden" ? M.golden : M.standard;
        model.add(new THREE.Mesh(S.apple, m), new THREE.Mesh(S.stem, M.stem), new THREE.Mesh(S.leaf, M.leaf));
        model.scale.setScalar(0.66);
        if (f.type === "golden") {
          const halo = new THREE.Mesh(S.torus, M.turboRing); halo.rotation.x = Math.PI / 2; halo.scale.setScalar(1.25); root.add(halo); spin.push([halo, "z", 1.5]);
        }
        break;
      }
      case "poison": {
        model.add(new THREE.Mesh(S.spike, M.poison));
        const r1 = new THREE.Mesh(S.torus, M.toxic); r1.rotation.x = 1.1; root.add(r1); spin.push([r1, "y", 2.2]);
        const r2 = new THREE.Mesh(S.torus, M.toxic); r2.rotation.x = -1.1; r2.scale.setScalar(0.85); root.add(r2); spin.push([r2, "y", -1.7]);
        break;
      }
      case "shrink": {
        model.add(new THREE.Mesh(S.crystal, M.shrink));
        const core = new THREE.Mesh(S.octa, M.shrinkCore); core.scale.set(1, 1.6, 1); model.add(core);
        const orb = new THREE.Group();
        for (let i = 0; i < 3; i++) { const s = new THREE.Mesh(S.small, M.shrinkCore); s.position.set(Math.cos(i * 2.094) * 0.42, 0, Math.sin(i * 2.094) * 0.42); orb.add(s); }
        root.add(orb); spin.push([orb, "y", 2.6]);
        break;
      }
      case "turbo": {
        model.add(new THREE.Mesh(S.boltGeo, M.turbo));
        const r = new THREE.Mesh(S.torus, M.turboRing); r.scale.setScalar(1.15); root.add(r); spin.push([r, "x", 3.5]);
        break;
      }
      case "immortal": {
        model.add(new THREE.Mesh(S.starGeo, M.immortal));
        const halo = new THREE.Mesh(S.torus, M.rainbowRing); halo.rotation.x = Math.PI / 2; halo.scale.setScalar(1.3); root.add(halo); spin.push([halo, "z", 2.4]);
        const halo2 = new THREE.Mesh(S.torus, M.rainbowRing); halo2.scale.setScalar(1.05); root.add(halo2); spin.push([halo2, "y", -1.9]);
        break;
      }
      case "freeze": {
        for (let k = 0; k < 3; k++) {
          const a = new THREE.Group(); a.rotation.z = (k * Math.PI) / 3;
          a.add(new THREE.Mesh(S.arm, M.freeze));
          for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
            const t = new THREE.Mesh(S.twig, M.freeze); t.position.set(sx * 0.22, sy * 0.05, 0); t.rotation.z = sx * sy * 0.8; a.add(t);
          }
          model.add(a);
        }
        model.add(new THREE.Mesh(S.core, M.white));
        break;
      }
    }
    const color = PALETTE.food[f.type];
    const base = new THREE.Group();
    base.position.set(cellX(f.x), 0, cellZ(f.y));
    const decal = this._decal(color, 1.7, 0.32);
    base.add(decal);
    let ringU = null;
    if (f.ttl != null) {
      ringU = { uProg: { value: 1 }, uColor: { value: color.clone().multiplyScalar(1.1) }, uTime: { value: 0 } };
      const ring = new THREE.Mesh(S.ring, new THREE.ShaderMaterial({
        uniforms: ringU, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false,
        vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
        fragmentShader: `varying vec2 vUv; uniform float uProg; uniform vec3 uColor; uniform float uTime;
          void main(){ vec2 q = vUv - 0.5; float r = length(q); float a = fract(atan(q.x, -q.y) / 6.28318 + 1.0);
            float band = smoothstep(0.40, 0.415, r) * (1.0 - smoothstep(0.455, 0.47, r));
            float on = step(a, uProg); float warn = uProg < 0.3 ? 0.5 + 0.5*sin(uTime*18.0) : 1.0;
            gl_FragColor = vec4(uColor, band * (on * 0.75 * warn + 0.1)); }`,
      }));
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.02;
      ring.renderOrder = 5;
      base.add(ring);
    }
    root.position.set(cellX(f.x), 0.55, cellZ(f.y));
    root.scale.setScalar(0.001);
    base.scale.setScalar(0.001);
    this.scene.add(root, base);
    return { root, base, model, spin, decal, ringU, f, born: this.time, phase: Math.random() * 6.28, vis: 1, removing: null };
  }

  _disposeFoodView(v) {
    this.scene.remove(v.root, v.base);
    v.base.traverse((o) => { if (o.material && (o.material.isShaderMaterial || o.material.map === this.glowTex)) o.material.dispose(); });
  }

  // ======================= события =======================
  handleEvents(events, game) {
    for (const e of events) {
      switch (e.type) {
        case "reset":
          this._buildLevel(game);
          for (const v of this.foodPool.values()) this._disposeFoodView(v);
          this.foodPool.clear();
          this.lumps = [];
          this.crashMarker.visible = false;
          this.effect.amt = 0;
          this._resetAt = this.time;
          game.snake.forEach((s, i) => this.particles.burst(tmpV.set(cellX(s.x), 0.4, cellZ(s.y)), PALETTE.headCol, 10, { speed: 1.5, up: 1.5, life: 0.7, size: 0.1 }));
          break;
        case "spawn":
          this.particles.burst(tmpV.set(cellX(e.food.x), 0.5, cellZ(e.food.y)), PALETTE.food[e.food.type], 14, { speed: 1.4, up: 1.6, life: 0.6, size: 0.1, gravity: -2 });
          break;
        case "expire": {
          const v = this.foodPool.get(e.food.id);
          if (v && !v.removing) v.removing = { kind: "expire", t: this.time };
          this.particles.burst(tmpV.set(cellX(e.food.x), 0.5, cellZ(e.food.y)), new THREE.Color(0.5, 0.55, 0.7), 18, { speed: 1.2, up: 1.2, life: 0.7, size: 0.13, gravity: -1 });
          break;
        }
        case "eat": {
          const v = this.foodPool.get(e.food.id);
          if (v) v.removing = { kind: "eat", t: this.time };
          const c = PALETTE.food[e.food.type];
          const at = tmpV.set(cellX(e.cell.x), 0.5, cellZ(e.cell.y));
          this.particles.burst(at, c, e.food.type === "golden" ? 90 : 55, { speed: 4.2, up: 3, life: 1.0, size: 0.16 });
          this.particles.burst(at, new THREE.Color(1, 1, 1), 16, { speed: 2.5, up: 2, life: 0.5, size: 0.1 });
          this._pulse(e.cell, c, 1.2);
          this.gulp = 1;
          if (e.food.type !== "shrink") this.lumps.push({ t: this.time, color: c });
          if (e.food.type === "shrink") {
            const tail = game.snake[game.snake.length - 1];
            this.particles.burst(tmpV.set(cellX(tail.x), 0.4, cellZ(tail.y)), PALETTE.food.shrink, 50, { speed: 3, up: 2, life: 0.9 });
          }
          break;
        }
        case "teleport":
          [e.from, e.to].forEach((p, i) => {
            this.particles.burst(tmpV.set(cellX(p.x), 0.4, cellZ(p.y)), i ? PALETTE.portalB : PALETTE.portalA, 40, { speed: 3, up: 3.5, life: 0.8, size: 0.13, gravity: -2 });
          });
          this.portalViews.forEach((pv) => (pv.u.uFlash.value = 1));
          if (this.linkU) this.linkU.uFlash.value = 1;
          this._pulse(e.to, PALETTE.portalB, 0.8);
          break;
        case "crash": {
          const at = tmpV.set(cellX(e.cell.x), 0.45, cellZ(e.cell.y));
          this.particles.burst(at, new THREE.Color(1, 0.18, 0.25), 160, { speed: 6.5, up: 4, life: 1.4, size: 0.18 });
          this.particles.burst(at, new THREE.Color(1, 0.7, 0.2), 70, { speed: 4.5, up: 5, life: 1.1, size: 0.12 });
          this.particles.burst(at, new THREE.Color(1, 1, 1), 30, { speed: 7, up: 2, life: 0.5, size: 0.08 });
          this.shake = 1;
          this._pulse(e.cell, new THREE.Color(1, 0.1, 0.25), 2);
          this.crashMarker.position.set(cellX(e.cell.x), 0.5, cellZ(e.cell.y));
          this.crashMarker.visible = true;
          break;
        }
        case "immortal": {
          const at = tmpV.copy(this.headWorld);
          for (let h = 0; h < 6; h++) this.particles.burst(at, tmpCol2.setHSL(h / 6, 0.9, 0.6), 22, { speed: 4.5, up: 3.5, life: 1.2, size: 0.15 });
          this._pulse(game.snake[0], PALETTE.food.immortal, 0.9);
          break;
        }
        case "immortalEnd":
          this.particles.burst(tmpV.copy(this.headWorld), new THREE.Color(0.7, 0.7, 0.8), 30, { speed: 2, up: 1.5, life: 0.7, size: 0.1 });
          break;
        case "wallPhase": {
          const col = e.phase === "GHOST" ? tmpCol2.setRGB(0.3, 0.9, 1) : e.phase === "SOLID" ? PALETTE.wallEdge : null;
          if (col) for (const c of e.structure.cells) this.particles.burst(tmpV.set(cellX(c.x), 0.6, cellZ(c.y)), col, e.phase === "SOLID" ? 14 : 8, { speed: 1.6, up: 1.2, life: 0.6, size: 0.1, gravity: -1 });
          break;
        }
        case "speed":
          this.effect.color.copy(e.effect === "turbo" ? PALETTE.food.turbo : PALETTE.food.freeze);
          this.effect.target = 1;
          break;
        case "speedEnd":
          this.effect.target = 0;
          break;
      }
    }
  }

  _pulse(cell, color, strength) {
    const U = this.floorU, i = (this._pulseIdx = ((this._pulseIdx || 0) + 1) % 4);
    U.uPulse.value[i].set(cellX(cell.x), cellZ(cell.y), this.time, strength);
    U.uPulseCol.value[i].copy(color);
  }

  // ======================= кадр =======================
  /**
   * @param {SnakeGame} game
   * @param {number} dt секунды с прошлого кадра
   * @param {number} alpha 0..1 — прогресс интерполяции между prevSnake и snake
   */
  frame(game, dt, alpha) {
    this.time += dt;
    const t = this.time;
    alpha = clamp01(alpha);

    // ---- змея ----
    this._updateSnake(game, alpha, dt);

    // ---- туман войны ----
    const fogTarget = game.settings.fog ? 1 : 0;
    this.fogAmt += (fogTarget - this.fogAmt) * damp(3, dt);
    this.fogGroup.visible = this.fogAmt > 0.01;
    const R = game.settings.fogRadius;
    this.floorU.uFog.value = this.fogAmt;
    this.floorU.uFogR.value = R;
    this.floorU.uHead.value.set(this.headWorld.x, this.headWorld.z);
    this.floorU.uTime.value = t;
    this.fogLayers.forEach((l) => { const u = l.material.uniforms; u.uTime.value = t; u.uFog.value = this.fogAmt; u.uFogR.value = R; u.uHead.value.set(this.headWorld.x, this.headWorld.z); });

    // ---- стены (анимация подъёма) ----
    for (const w of this.wallViews || []) {
      const k = clamp01((t - w.born) / 0.7);
      w.g.position.y = -1.1 * (1 - easeOutBack(k));
      w.g.visible = k > 0;
    }
    this._updateWallPhases(dt);

    // ---- порталы ----
    for (const p of this.portalViews) {
      p.u.uTime.value = t;
      p.u.uFlash.value *= Math.exp(-dt * 3);
      p.r1.rotation.z = t * (p.idx ? -1.2 : 1.2);
      p.r2.rotation.x = Math.PI / 2 + Math.sin(t * 1.3 + p.idx) * 0.35;
      p.r2.rotation.y = t * 1.7;
      p.r2.position.y = 0.55 + Math.sin(t * 2 + p.idx * 2) * 0.12;
    }
    if (this.linkU) { this.linkU.uTime.value = t; this.linkU.uFlash.value *= Math.exp(-dt * 2.5); }

    // ---- дроны ----
    const plans = game.dronePlans(), paths = game.dronePaths();
    game.drones.forEach((p, i) => {
      const d = this.drones[i]; if (!d) return;
      const pp = game.prevPatrols[i] || p;
      const x = lerp(pp.x, p.x, alpha), y = lerp(pp.y, p.y, alpha);
      d.g.position.set(cellX(x), 0.85 + Math.sin(t * 3 + i) * 0.07, cellZ(y));
      const mvx = p.x - pp.x, mvy = p.y - pp.y;
      const [pdx, pdy] = plans[i];
      const targetYaw = mvx || mvy ? Math.atan2(mvx, mvy) : pdx || pdy ? Math.atan2(pdx, pdy) : d.yaw;
      d.yaw = lerpAngle(d.yaw, targetYaw, damp(10, dt));
      d.g.rotation.y = d.yaw;
      d.g.rotation.z = Math.sin(t * 2.1 + i) * 0.06;
      d.rotorMeshes.forEach((r, k) => (r.rotation.z = t * (k % 2 ? 40 : -40)));
      d.coneU.uTime.value = t;
      d.mkU.forEach((u) => (u.uTime.value = t));
      d.curMk.position.set(cellX(p.x), 0.016, cellZ(p.y));
      d.nextMk.visible = !!(pdx || pdy);
      d.nextMk.position.set(cellX(p.x + pdx), 0.016, cellZ(p.y + pdy));
      paths[i].forEach(([px, py], j) => d.pathMk[j].position.set(cellX(px), 0.014, cellZ(py)));
      d.decal.position.set(d.g.position.x, 0.013, d.g.position.z);
      d.light.position.copy(d.g.position).y -= 0.2;
    });

    // ---- еда ----
    this._updateFoods(game, dt);

    // ---- частицы и окружение ----
    this.particles.update(dt);
    this.sky.material.uniforms.uTime.value = t;
    this.starMat.uniforms.uTime.value = t;
    this.outerMat.uniforms.uTime.value = t;
    this.dustMat.uniforms.uTime.value = t;
    const deadTarget = game.over ? 1 : 0;
    this.deadAmt += (deadTarget - this.deadAmt) * damp(game.over ? 6 : 3, dt);
    this.floorU.uDead.value = this.deadAmt * 0.7;
    this.railMat.color.setRGB(lerp(0, 2.0, this.deadAmt), lerp(0.9, 0.12, this.deadAmt), lerp(1.3, 0.3, this.deadAmt));
    if (this.crashMarker.visible) {
      this.crashMarker.rotation.y = t * 1.5;
      this.crashMarker.scale.setScalar(1 + Math.sin(t * 10) * 0.06);
      if (!game.over) this.crashMarker.visible = false;
    }

    // ---- floor glows: голова, порталы, дроны ----
    const U = this.floorU; let n = 0;
    const glow = (x, z, r, inten, col) => { if (n >= 8) return; U.uGlow.value[n].set(x, z, r, inten); U.uGlowCol.value[n].copy(col); n++; };
    glow(this.headWorld.x, this.headWorld.z, 1.5, 0.28 * (1 - this.deadAmt * 0.5), game.over ? tmpCol.setRGB(1, 0.1, 0.2) : PALETTE.headCol);
    this.portalViews.forEach((p) => glow(p.g.position.x, p.g.position.z, 1.3, 0.3 + p.u.uFlash.value, p.color));
    this.drones.forEach((d) => glow(d.g.position.x, d.g.position.z, 1.0, 0.3, PALETTE.drone));
    U.uGlowN.value = n;

    this._updateCamera(game, dt);
    this._render();
  }

  /** Фазовые стены: мигание-предупреждение, голограмма-призрак, плавное «затвердевание», счётчик шагов. */
  _updateWallPhases(dt) {
    const t = this.time;
    for (const sv of this.structViews || []) {
      const s = sv.s;
      const ghostTarget = s.phase === "GHOST" ? 1 : 0;
      sv.ghost += (ghostTarget - sv.ghost) * damp(ghostTarget ? 7 : 4, dt);
      const gh = sv.ghost;
      const blinking = s.phase === "BLINKING";
      const blink = blinking ? (Math.sin(t * 26) > 0 ? 1 : 0.15) : 1;
      const holo = 0.75 + 0.25 * Math.sin(t * 9 + sv.groups.length);
      sv.mats.box.opacity = lerp(1, 0.1 + 0.05 * holo, gh);
      sv.mats.box.depthWrite = gh < 0.5;
      sv.mats.box.color.setRGB(lerp(0.082, 0.2, gh), lerp(0.1, 0.6, gh), lerp(0.21, 0.9, gh));
      sv.mats.box.emissive.copy(PALETTE.wallEdge).multiplyScalar(blinking && blink > 0.5 ? 0.6 : 0);
      tmpCol.copy(PALETTE.wallEdge).multiplyScalar(2.0).lerp(tmpCol2.setRGB(0.25, 1.4, 1.9), gh);
      sv.mats.edge.color.copy(tmpCol).multiplyScalar(blinking ? blink * 1.4 : gh > 0.5 ? holo : 1);
      sv.mats.cap.opacity = lerp(0.85, 0.25 * holo, gh) * (blinking ? blink : 1);
      sv.mats.cap.color.copy(PALETTE.wallEdge).multiplyScalar(0.9).lerp(tmpCol2.setRGB(0.1, 0.6, 0.8), gh);
      for (const g of sv.groups) g.children[0].scale.y = lerp(1, 0.97, gh);
      // счётчик: шаги до исчезновения (мигание, «!N») или до затвердевания (призрак, «N»)
      const text = blinking ? `!${s.timer}` : s.phase === "GHOST" ? `${s.timer}` : "";
      if (text !== sv.labelText) {
        sv.labelText = text;
        const c = sv.labelCanvas.getContext("2d");
        c.clearRect(0, 0, 128, 64);
        if (text) {
          c.font = "bold 40px Orbitron, JetBrains Mono, monospace";
          c.textAlign = "center"; c.textBaseline = "middle";
          c.shadowColor = s.phase === "GHOST" ? "#4df0ff" : "#ff2a6d"; c.shadowBlur = 12;
          c.fillStyle = s.phase === "GHOST" ? "#bff8ff" : "#ffd0dc";
          c.fillText(text, 64, 34);
        }
        sv.labelTex.needsUpdate = true;
        sv.label.visible = !!text;
      }
      if (sv.label.visible) sv.label.position.y = 1.75 + Math.sin(t * 2.5) * 0.06;
    }
  }

  _updateSnake(game, alpha, dt) {
    const cur = game.snake, prev = game.prevSnake || cur, n = cur.length;
    const pts = (this._pts ||= []);
    const breaks = (this._breaks ||= []);
    pts.length = 0; breaks.length = 0;
    for (let i = 0; i < n; i++) {
      const c = cur[i], p = prev[Math.min(i, prev.length - 1)];
      let tx = c.x, ty = c.y;
      if (Math.abs(c.x - p.x) + Math.abs(c.y - p.y) > 1) {
        // телепорт: показываем, как сегмент «втягивается» во входной портал
        const entry = game.portals.find((q) => !(q.x === c.x && q.y === c.y)) || c;
        tx = entry.x; ty = entry.y;
      }
      pts.push(lerp(p.x, tx, alpha), lerp(p.y, ty, alpha));
    }
    // Голова
    const hx = cellX(pts[0]), hz = cellZ(pts[1]);
    this.headWorld.set(hx, 0.36, hz);
    let fx = ACTIONS[game.currentDir].x, fz = ACTIONS[game.currentDir].y;
    this.headYaw = lerpAngle(this.headYaw, Math.atan2(fx, fz), damp(14, dt));
    this.headFwd.set(Math.sin(this.headYaw), 0, Math.cos(this.headYaw));
    this.gulp *= Math.exp(-dt * 5);
    const teleHead = prev[0] && Math.abs(cur[0].x - prev[0].x) + Math.abs(cur[0].y - prev[0].y) > 1;
    const sink = teleHead ? 1 - 0.75 * Math.pow(alpha, 2) : 1;
    this.head.position.copy(this.headWorld);
    this.head.rotation.y = this.headYaw;
    this.head.scale.setScalar((1 + this.gulp * 0.25) * sink);
    this.headDecal.position.set(hx, 0.013, hz);
    this.headLight.position.set(hx, 1.0, hz);
    const deadCol = tmpCol.setRGB(1, 0.1, 0.22);
    this.headMat.color.copy(PALETTE.headCol).multiplyScalar(0.6).lerp(tmpCol2.setRGB(1, 0.8, 0.3), (this.immortalAmt || 0) * 0.6).lerp(deadCol, this.deadAmt);
    this.headMat.emissive.copy(this.headMat.color);
    this.headLight.color.copy(this.headMat.color);
    this.headDecal.material.color.copy(this.headMat.color).multiplyScalar(0.22);
    this.snakeU.uTime.value = this.time;
    this.snakeU.uDead.value = this.deadAmt;
    this.effect.amt += ((this.effect.target || 0) - this.effect.amt) * damp(4, dt);
    this.snakeU.uEffect.value = this.effect.amt;
    this.snakeU.uEffectCol.value.copy(this.effect.color);
    // ---- бессмертие ----
    const imm = game.immortalSteps;
    this.immortalAmt += ((imm > 0 ? 1 : 0) - this.immortalAmt) * damp(imm > 0 ? 6 : 3, dt);
    const warn = imm > 0 && imm <= 8 ? 1 : 0;
    this.snakeU.uImmortal.value = this.immortalAmt;
    this.snakeU.uWarn.value = warn;
    this.aura.visible = this.immortalAmt > 0.02;
    if (this.aura.visible) {
      const flick = warn && Math.sin(this.time * 31) > 0 ? 0.35 : 1;
      this.aura.position.set(this.headWorld.x, 0.55, this.headWorld.z);
      this.aura.scale.setScalar((2.1 + 0.25 * Math.sin(this.time * 6)) * this.immortalAmt);
      this.aura.material.color.setHSL((this.time * 0.25) % 1, 0.9, 0.62).multiplyScalar(1.6 * flick * this.immortalAmt);
      if (Math.random() < dt * 40 * this.immortalAmt && n > 1) {
        const seg = game.snake[Math.floor(Math.random() * n)];
        tmpV2.set(cellX(seg.x) + (Math.random() - 0.5) * 0.4, 0.5, cellZ(seg.y) + (Math.random() - 0.5) * 0.4);
        this.particles.burst(tmpV2, tmpCol2.setHSL(Math.random(), 0.9, 0.65), 1, { speed: 0.4, up: 1.6, life: 0.7, size: 0.1, gravity: 0.5 });
      }
    }

    // Разбиение на цепочки (по телепортам)
    const chains = [];
    let start = 0;
    for (let i = 1; i < n; i++) {
      if (Math.abs(pts[i * 2] - pts[i * 2 - 2]) + Math.abs(pts[i * 2 + 1] - pts[i * 2 - 1]) > 1.5) { chains.push([start, i - 1]); start = i; }
    }
    chains.push([start, n - 1]);

    // Комки проглоченной еды
    const tick = Math.max(0.03, (this.tickMs || 120) / 1000);
    this.lumps = this.lumps.filter((l) => (this.time - l.t) / tick < n + 1);
    const lumpS = this.lumps.map((l) => (this.time - l.t) / tick);

    const SUB = 6, S = this.samples, total = Math.max(1, n - 1);
    let ci = 0;
    for (const [a, b] of chains) {
      if (b - a < 1) continue;
      let m = 0;
      const P = (j) => { j = Math.max(a, Math.min(b, j)); return j; };
      for (let j = a; j <= b; j++) {
        const steps = j === b ? 1 : SUB;
        for (let s = 0; s < steps; s++) {
          const tt = s / SUB;
          const i0 = P(j - 1), i1 = j, i2 = P(j + 1), i3 = P(j + 2);
          const cr = (k) => {
            const p0 = pts[i0 * 2 + k], p1 = pts[i1 * 2 + k], p2 = pts[i2 * 2 + k], p3 = pts[i3 * 2 + k];
            const t2 = tt * tt, t3 = t2 * tt;
            return 0.5 * (2 * p1 + (-p0 + p2) * tt + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
          };
          const sx = cr(0), sy = cr(1);
          const segS = j + tt;
          const u = segS / total;
          let r = lerp(0.31, 0.15, Math.pow(u, 1.1));
          r *= 0.93 + 0.07 * Math.cos(tt * Math.PI * 2);
          const tailLeft = (n - 1) - segS;
          r *= 0.25 + 0.75 * clamp01(tailLeft / 0.9);
          for (const ls of lumpS) r *= 1 + 0.5 * Math.exp(-Math.pow((segS - ls) * 1.4, 2));
          if (m >= 4096) break;
          const o = m * 6;
          S[o] = cellX(sx); S[o + 1] = 0.34; S[o + 2] = cellZ(sy); S[o + 3] = r; S[o + 4] = u; S[o + 5] = segS;
          m++;
        }
      }
      if (!this.tubes[ci]) { this.tubes[ci] = new TubeChain(this.snakeMat); this.snakeGroup.add(this.tubes[ci].mesh); }
      this.tubes[ci].build(S, m);
      ci++;
    }
    for (let i = ci; i < this.tubes.length; i++) this.tubes[i].mesh.visible = false;
  }

  _updateFoods(game, dt) {
    const t = this.time;
    const alive = new Set();
    for (const f of game.foods) {
      alive.add(f.id);
      let v = this.foodPool.get(f.id);
      if (!v) { v = this._makeFood(f); this.foodPool.set(f.id, v); }
    }
    for (const [id, v] of this.foodPool) {
      if (!alive.has(id) && !v.removing) v.removing = { kind: "expire", t };
      const f = v.f;
      const visTarget = game.isFoodVisible(f) ? 1 : 0;
      v.vis += (visTarget - v.vis) * damp(6, dt);
      const pop = easeOutBack(clamp01((t - v.born) / 0.5));
      let rem = 1, yOff = 0;
      if (v.removing) {
        const k = clamp01((t - v.removing.t) / (v.removing.kind === "eat" ? 0.18 : 0.4));
        rem = 1 - k;
        if (v.removing.kind === "eat") yOff = k * 0.3;
        if (k >= 1) { this._disposeFoodView(v); this.foodPool.delete(id); continue; }
      }
      v.root.scale.setScalar(Math.max(0.001, pop * (0.15 + 0.85 * v.vis) * rem));
      v.base.scale.setScalar(Math.max(0.001, pop * v.vis * rem));
      v.root.position.y = 0.55 + Math.sin(t * 2.2 + v.phase) * 0.08 + yOff;
      v.model.rotation.y = t * (f.type === "turbo" ? 2.4 : 1.1) + v.phase;
      if (f.type === "poison") v.model.rotation.x = t * 0.7;
      if (f.type === "shrink") v.model.rotation.y = t * 1.8;
      for (const [o, ax, sp] of v.spin) o.rotation[ax] += sp * dt;
      v.decal.visible = v.vis > 0.05;
      if (v.ringU) {
        v.ringU.uProg.value = clamp01(f.ttl / (f.maxTtl || 1));
        v.ringU.uTime.value = t;
      }
      // Искры у золотого яблока / пузыри у яда
      if (v.vis > 0.5 && !v.removing && Math.random() < dt * (f.type === "golden" ? 10 : f.type === "immortal" ? 16 : f.type === "poison" ? 6 : 0)) {
        tmpV2.set(v.root.position.x + (Math.random() - 0.5) * 0.5, 0.6, v.root.position.z + (Math.random() - 0.5) * 0.5);
        const pc = f.type === "golden" ? PALETTE.food.golden : f.type === "immortal" ? tmpCol.setHSL(Math.random(), 0.9, 0.65) : tmpCol.setRGB(0.4, 1, 0.3);
        this.particles.burst(tmpV2, pc, 1, { speed: 0.2, up: 1.2, life: 0.9, size: 0.09, gravity: 0.6 });
      }
    }
  }

  _updateCamera(game, dt) {
    const cam = this.camera;
    const center = tmpV.set(0, 0, 0);
    if (this.cameraMode === "orbit") {
      this.controls.enabled = true;
      this.controls.autoRotate = this.autoRotate && performance.now() - this.lastInteract > 4000;
      const desired = tmpV2.copy(center).lerp(this.headWorld, 0.22); desired.y = 0;
      this.controls.target.lerp(desired, damp(1.5, dt));
      this.controls.update(dt);
      this.lookTarget.copy(this.controls.target);
    } else {
      this.controls.enabled = false;
      let pos, look;
      if (this.cameraMode === "top") {
        const fit = (GRID / 2 + 1.4) / Math.tan(THREE.MathUtils.degToRad(cam.fov / 2));
        const h = fit / Math.min(1, cam.aspect);
        pos = tmpV2.set(0, h, h * 0.12);
        look = new THREE.Vector3(0, 0, 0.2);
      } else {
        pos = tmpV2.copy(this.headWorld).addScaledVector(this.headFwd, -6.2); pos.y = 5.4;
        look = this.headWorld.clone().addScaledVector(this.headFwd, 3);
        look.y = 0;
      }
      cam.position.lerp(pos, damp(this.cameraMode === "chase" ? 3.2 : 2.5, dt));
      this.lookTarget.lerp(look, damp(5, dt));
      cam.lookAt(this.lookTarget);
    }
    this.shake *= Math.exp(-dt * 3.5);
  }

  _render() {
    const cam = this.camera;
    let ox = 0, oy = 0, oz = 0;
    if (this.shake > 0.01) {
      const s = this.shake * 0.35;
      ox = (Math.random() - 0.5) * s; oy = (Math.random() - 0.5) * s; oz = (Math.random() - 0.5) * s;
      cam.position.x += ox; cam.position.y += oy; cam.position.z += oz;
    }
    if (this.bloomOn) this.composer.render();
    else this.renderer.render(this.scene, cam);
    cam.position.x -= ox; cam.position.y -= oy; cam.position.z -= oz;
  }
}

const tmpCol = new THREE.Color();
const tmpCol2 = new THREE.Color();
