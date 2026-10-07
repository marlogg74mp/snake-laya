// render3d.js — three.js renderer for Snake Duel. Same interface as Renderer2D:
//   new Renderer3D(container) · snapshot(match) · handleEvents(events) · draw(match, t, now) · setIdle(bool)
// Particles / TubeChain are adapted from web3d/js/render3d.js so the duel stays self-contained.
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { GRID, DELTA, FOG_RADIUS } from "./versus_game.js";

const HALF = (GRID - 1) / 2;
const wx = (x) => x - HALF;
const wz = (y) => y - HALF;
const damp = (k, dt) => 1 - Math.exp(-k * dt);

const COL = {
  human: new THREE.Color("#22d3ee"), humanDark: new THREE.Color("#0e7490"),
  ai: new THREE.Color("#f472b6"), aiDark: new THREE.Color("#9d174d"),
  wall: new THREE.Color("#ff2e63"), ghost: new THREE.Color("#7dd3fc"),
  drone: new THREE.Color("#facc15"), portal: [new THREE.Color("#22d3ee"), new THREE.Color("#c084fc")],
  food: { apple: new THREE.Color("#ff3b5c"), golden: new THREE.Color("#ffc933"), immortal: new THREE.Color("#fff27a"), poison: new THREE.Color("#a3e635") },
};

// ---------------------------------------------------------------- particles (one draw call)
class Particles {
  constructor(cap, scale) {
    Object.assign(this, { cap, next: 0 });
    for (const k of ["pos", "col", "vel"]) this[k] = new Float32Array(cap * 3);
    for (const k of ["size", "alpha", "life", "max", "base", "grav"]) this[k] = new Float32Array(cap);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("aColor", new THREE.BufferAttribute(this.col, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("aSize", new THREE.BufferAttribute(this.size, 1).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("aAlpha", new THREE.BufferAttribute(this.alpha, 1).setUsage(THREE.DynamicDrawUsage));
    this.points = new THREE.Points(g, new THREE.ShaderMaterial({
      uniforms: { uScale: scale },
      vertexShader: `attribute vec3 aColor; attribute float aSize; attribute float aAlpha; uniform float uScale;
        varying vec3 vC; varying float vA;
        void main(){ vC = aColor; vA = aAlpha; vec4 mv = modelViewMatrix * vec4(position,1.0);
          gl_Position = projectionMatrix * mv; gl_PointSize = aSize * uScale / max(0.1, -mv.z); }`,
      fragmentShader: `varying vec3 vC; varying float vA;
        void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;
          float a = smoothstep(0.5, 0.0, d); gl_FragColor = vec4(vC * (1.0 + 2.0*a*a), vA * a); }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    this.points.frustumCulled = false;
  }
  burst(o, color, n, { speed = 3, up = 2.5, life = 0.9, size = 0.14, gravity = -7 } = {}) {
    for (let k = 0; k < n; k++) {
      const i = this.next; this.next = (i + 1) % this.cap;
      const th = Math.random() * 6.283, ph = Math.acos(2 * Math.random() - 1), sp = speed * (0.35 + Math.random() * 0.65);
      this.pos.set([o.x, o.y, o.z], i * 3);
      this.vel.set([Math.sin(ph) * Math.cos(th) * sp, Math.abs(Math.cos(ph)) * sp * 0.6 + up * Math.random(), Math.sin(ph) * Math.sin(th) * sp], i * 3);
      const tint = 0.75 + Math.random() * 0.5;
      this.col.set([color.r * tint, color.g * tint, color.b * tint], i * 3);
      this.max[i] = this.life[i] = life * (0.6 + Math.random() * 0.6);
      this.base[i] = size * (0.5 + Math.random());
      this.grav[i] = gravity;
    }
  }
  update(dt) {
    for (let i = 0; i < this.cap; i++) {
      if (this.life[i] <= 0) { this.alpha[i] = 0; continue; }
      this.life[i] -= dt;
      const t = Math.max(0, this.life[i] / this.max[i]), drag = Math.exp(-1.8 * dt);
      this.vel[i * 3 + 1] += this.grav[i] * dt;
      this.vel[i * 3] *= drag; this.vel[i * 3 + 2] *= drag;
      this.pos[i * 3] += this.vel[i * 3] * dt;
      this.pos[i * 3 + 1] = Math.max(0.03, this.pos[i * 3 + 1] + this.vel[i * 3 + 1] * dt);
      this.pos[i * 3 + 2] += this.vel[i * 3 + 2] * dt;
      if (this.pos[i * 3 + 1] <= 0.03) this.vel[i * 3 + 1] *= -0.4;
      this.alpha[i] = t * t;
      this.size[i] = this.base[i] * (0.4 + 0.6 * t);
    }
    const a = this.points.geometry.attributes;
    a.position.needsUpdate = a.aColor.needsUpdate = a.aSize.needsUpdate = a.aAlpha.needsUpdate = true;
  }
}

// ---------------------------------------------------------------- snake tube with reusable buffers
const RADIAL = 12;
class TubeChain {
  constructor(material) {
    this.geo = new THREE.BufferGeometry();
    this.mesh = new THREE.Mesh(this.geo, material);
    this.mesh.frustumCulled = false;
    this.cap = 0;
    this.ensure(256);
  }
  ensure(rings) {
    if (rings <= this.cap) return;
    let cap = Math.max(this.cap, 256);
    while (cap < rings) cap *= 2;
    this.cap = cap;
    this.P = new Float32Array(cap * RADIAL * 3);
    this.N = new Float32Array(cap * RADIAL * 3);
    const idx = new Uint32Array((cap - 1) * RADIAL * 6);
    let o = 0;
    for (let i = 0; i < cap - 1; i++) for (let j = 0; j < RADIAL; j++) {
      const a = i * RADIAL + j, b = i * RADIAL + ((j + 1) % RADIAL), c = a + RADIAL, d = b + RADIAL;
      idx.set([a, b, c, b, d, c], o); o += 6;
    }
    this.geo.setIndex(new THREE.BufferAttribute(idx, 1));
    this.geo.setAttribute("position", new THREE.BufferAttribute(this.P, 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute("normal", new THREE.BufferAttribute(this.N, 3).setUsage(THREE.DynamicDrawUsage));
  }
  /** pts: [{x,y,z,r}] */
  build(pts) {
    const n = pts.length;
    this.ensure(n);
    for (let i = 0; i < n; i++) {
      const p0 = pts[Math.max(0, i - 1)], p1 = pts[Math.min(n - 1, i + 1)], c = pts[i];
      let tx = p1.x - p0.x, tz = p1.z - p0.z;
      const tl = Math.hypot(tx, tz) || 1; tx /= tl; tz /= tl;
      for (let j = 0; j < RADIAL; j++) {
        const th = (j / RADIAL) * 6.283, nx = Math.sin(th) * -tz, ny = Math.cos(th), nz = Math.sin(th) * tx;
        const v = (i * RADIAL + j) * 3;
        this.P[v] = c.x + nx * c.r; this.P[v + 1] = c.y + ny * c.r; this.P[v + 2] = c.z + nz * c.r;
        this.N[v] = nx; this.N[v + 1] = ny; this.N[v + 2] = nz;
      }
    }
    this.geo.attributes.position.needsUpdate = this.geo.attributes.normal.needsUpdate = true;
    this.geo.setDrawRange(0, Math.max(0, n - 1) * RADIAL * 6);
    this.mesh.visible = n > 1;
  }
}

function labelSprite(text, color) {
  const c = document.createElement("canvas");
  c.width = 256; c.height = 96;
  const g = c.getContext("2d");
  g.font = "900 54px Orbitron, sans-serif";
  g.textAlign = "center"; g.textBaseline = "middle";
  g.shadowColor = color; g.shadowBlur = 18;
  g.fillStyle = color;
  g.fillText(text, 128, 50);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  s.scale.set(1.6, 0.6, 1);
  s.renderOrder = 30;
  return s;
}

function starShape(R = 0.42, r = 0.18) {
  const s = new THREE.Shape();
  for (let i = 0; i < 10; i++) {
    const a = (i * Math.PI) / 5 + Math.PI / 2, rad = i % 2 ? r : R;
    i ? s.lineTo(Math.cos(a) * rad, Math.sin(a) * rad) : s.moveTo(Math.cos(a) * rad, Math.sin(a) * rad);
  }
  return s;
}

// ---------------------------------------------------------------- renderer
export class Renderer3D {
  constructor(container) {
    this.container = container;
    this.prev = null;
    this.shake = 0;
    this.flash = 0;
    this.idle = true;
    this.view = "tilt";
    this.lastNow = performance.now();
    this.fx = []; // rings and beams
    this.fps = { frames: 0, since: performance.now(), low: false };

    const r = (this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" }));
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.0;
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.setClearColor(0x04060d, 1);
    r.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    r.domElement.className = "gl";
    container.prepend(r.domElement);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x060914, 0.012);
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 400);
    this.uScale = { value: 500 };

    this.scene.add(new THREE.HemisphereLight(0x6a7cff, 0x14001c, 1.1));
    const key = new THREE.DirectionalLight(0xdbe4ff, 1.5);
    key.position.set(-6, 18, 12);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xff3cac, 0.7);
    rim.position.set(10, 6, -14);
    this.scene.add(rim);

    this._buildArena();
    this.particles = new Particles(2400, this.uScale);
    this.scene.add(this.particles.points);
    this.level = new THREE.Group();
    this.scene.add(this.level);
    this.levelSeed = null;
    this.foodViews = new Map();
    this._buildSnakes();

    const rt = new THREE.WebGLRenderTarget(2, 2, { type: THREE.HalfFloatType, samples: 4 });
    this.composer = new EffectComposer(r, rt);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.55, 0.35, 0.9);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.bloomOn = !new URLSearchParams(location.search).has("lowfx");

    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(container);
    this.resize();
  }

  dispose() {
    this._ro.disconnect();
    this.renderer.domElement.remove();
    this.renderer.dispose();
  }

  setIdle(on) { this.idle = on; }
  setNames(names) {
    for (const id of ["human", "ai"]) {
      const v = this.snakeViews[id], old = v.label;
      v.label = labelSprite(names[id], id === "human" ? "#67e8f9" : "#f9a8d4");
      if (names[id].length > 4) v.label.scale.set(2.6, 0.6, 1);
      this.scene.remove(old);
      old.material.map.dispose(); old.material.dispose();
      this.scene.add(v.label);
    }
  }
  /** tilt -> top -> chase (camera behind the human snake, controls become relative) */
  toggleView() {
    this.view = { tilt: "top", top: "chase", chase: "tilt" }[this.view];
    return this.view;
  }

  resize() {
    const w = Math.max(1, this.container.clientWidth), h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = `${w}px`;
    this.renderer.domElement.style.height = `${h}px`;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    const pr = this.renderer.getPixelRatio();
    this.composer.setPixelRatio(pr);
    this.composer.setSize(w, h);
    this.bloom.resolution.set(w * pr * 0.5, h * pr * 0.5);
    this.uScale.value = (h * pr) / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2));
  }

  // ------------------------------------------------------------ static arena
  _buildArena() {
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(GRID + 0.4, GRID + 0.4), new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 } },
      vertexShader: `varying vec2 vP; void main(){ vP = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: `varying vec2 vP; uniform float uTime;
        void main(){
          vec2 g = abs(fract(vP + 0.5) - 0.5);
          float line = smoothstep(0.035, 0.0, min(g.x, g.y));
          float side = smoothstep(-2.0, 2.0, vP.x);               // left = human (cyan), right = AI (magenta)
          vec3 tint = mix(vec3(0.13, 0.83, 0.93), vec3(0.96, 0.45, 0.71), side);
          float pulse = 0.55 + 0.45 * sin(uTime * 1.3 - length(vP) * 0.5);
          vec3 base = vec3(0.012, 0.02, 0.045);
          gl_FragColor = vec4(base + tint * line * (0.25 + 0.2 * pulse), 1.0);
          #include <colorspace_fragment>
        }`,
    }));
    floor.rotation.x = -Math.PI / 2;
    this.floor = floor;
    this.scene.add(floor);
    // border rails: cyan on the human side, magenta on the AI side
    const railGeo = new THREE.BoxGeometry(1, 0.12, 0.12);
    const mk = (color, x, z, len, rotY) => {
      const m = new THREE.Mesh(railGeo, new THREE.MeshBasicMaterial({ color: color.clone().multiplyScalar(2.2) }));
      m.scale.x = len; m.position.set(x, 0.06, z); m.rotation.y = rotY;
      this.scene.add(m);
    };
    const e = GRID / 2 + 0.1;
    mk(COL.human, -e, 0, GRID + 0.3, Math.PI / 2);
    mk(COL.ai, e, 0, GRID + 0.3, Math.PI / 2);
    mk(COL.human, -GRID / 4, -e, GRID / 2, 0); mk(COL.ai, GRID / 4, -e, GRID / 2, 0);
    mk(COL.human, -GRID / 4, e, GRID / 2, 0); mk(COL.ai, GRID / 4, e, GRID / 2, 0);
    // backdrop
    const stars = new THREE.BufferGeometry();
    const sp = new Float32Array(900 * 3);
    for (let i = 0; i < 900; i++) {
      const a = Math.random() * 6.283, d = 40 + Math.random() * 90;
      sp.set([Math.cos(a) * d, -5 + Math.random() * 60, Math.sin(a) * d], i * 3);
    }
    stars.setAttribute("position", new THREE.BufferAttribute(sp, 3));
    this.scene.add(new THREE.Points(stars, new THREE.PointsMaterial({ color: 0x8aa0ff, size: 0.25, transparent: true, opacity: 0.6 })));
    this.cellBox = new RoundedBoxGeometry(0.86, 0.9, 0.86, 3, 0.12);
    // fog of war: dark sheet above the board with a Manhattan "diamond" window around the human head
    this.fogMat = new THREE.ShaderMaterial({
      uniforms: { uCenter: { value: new THREE.Vector2() }, uAmt: { value: 0 }, uR: { value: FOG_RADIUS + 0.5 }, uTime: { value: 0 } },
      vertexShader: `varying vec2 vP; void main(){ vec4 w = modelMatrix * vec4(position,1.0); vP = w.xz; gl_Position = projectionMatrix * viewMatrix * w; }`,
      fragmentShader: `varying vec2 vP; uniform vec2 uCenter; uniform float uAmt; uniform float uR; uniform float uTime;
        void main(){
          vec2 d = abs(vP - uCenter);
          float m = d.x + d.y;
          float swirl = 0.08 * sin(vP.x * 0.9 + uTime * 0.7) * sin(vP.y * 0.8 - uTime * 0.5);
          float a = uAmt * (0.88 + swirl) * smoothstep(uR - 0.7, uR + 0.5, m);
          float rim = uAmt * 0.35 * (1.0 - smoothstep(0.0, 0.25, abs(m - uR)));
          gl_FragColor = vec4(mix(vec3(0.015, 0.02, 0.05), vec3(0.2, 0.6, 1.0), rim / max(0.001, a + rim)), clamp(a + rim, 0.0, 1.0));
        }`,
      transparent: true, depthWrite: false,
    });
    this.fogSheet = new THREE.Mesh(new THREE.PlaneGeometry(GRID + 40, GRID + 40), this.fogMat);
    this.fogSheet.rotation.x = -Math.PI / 2;
    this.fogSheet.position.y = 1.7;
    this.fogSheet.renderOrder = 25;
    this.scene.add(this.fogSheet);
    this.fogAmt = 0;
  }

  // ------------------------------------------------------------ per-match level (walls, portals, drones)
  _buildLevel(match) {
    this.level.clear();
    this.levelSeed = match.seed;
    this.wallViews = match.structures.map((st) => {
      const solidMat = new THREE.MeshStandardMaterial({ color: 0x2a0612, emissive: COL.wall, emissiveIntensity: 1.1, roughness: 0.35, metalness: 0.2, transparent: true });
      const wireMat = new THREE.LineBasicMaterial({ color: COL.ghost.clone().multiplyScalar(1.8), transparent: true, opacity: 0 });
      const edges = new THREE.EdgesGeometry(this.cellBox, 30);
      const group = new THREE.Group();
      for (const [x, y] of st.cells) {
        const m = new THREE.Mesh(this.cellBox, solidMat);
        m.position.set(wx(x), 0.45, wz(y));
        const w = new THREE.LineSegments(edges, wireMat);
        w.position.copy(m.position);
        group.add(m, w);
      }
      this.level.add(group);
      return { st, solidMat, wireMat, group, vis: 1 };
    });
    this.portalViews = match.portals.map((p, i) => {
      const g = new THREE.Group();
      g.position.set(wx(p.x), 0.05, wz(p.y));
      const ringMat = new THREE.MeshBasicMaterial({ color: COL.portal[i].clone().multiplyScalar(2.5) });
      for (let k = 0; k < 3; k++) {
        const ring = new THREE.Mesh(new THREE.TorusGeometry(0.22 + k * 0.1, 0.025, 8, 40, Math.PI * 1.3), ringMat);
        ring.rotation.x = -Math.PI / 2;
        ring.position.y = 0.05 + k * 0.08;
        g.add(ring);
      }
      const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.42, 2.2, 24, 1, true),
        new THREE.MeshBasicMaterial({ color: COL.portal[i], transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending }));
      beam.position.y = 1.1;
      g.add(beam);
      this.level.add(g);
      return g;
    });
    for (const d of match.drones) {
      const lane = new THREE.Mesh(new THREE.PlaneGeometry(d.hi - d.lo + 1, 0.9),
        new THREE.MeshBasicMaterial({ color: COL.drone, transparent: true, opacity: 0.05, depthWrite: false }));
      lane.rotation.x = -Math.PI / 2;
      lane.position.set(wx((d.lo + d.hi) / 2), 0.01, wz(d.y));
      this.level.add(lane);
    }
    this.droneViews = match.drones.map(() => {
      const g = new THREE.Group();
      const body = new THREE.Mesh(new THREE.SphereGeometry(0.26, 24, 16),
        new THREE.MeshStandardMaterial({ color: 0x3a2a00, emissive: COL.drone, emissiveIntensity: 1.6, roughness: 0.3 }));
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.09, 12, 8), new THREE.MeshBasicMaterial({ color: 0xff1f3d }));
      eye.position.set(0, -0.08, 0.2);
      const rotor = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.03, 6, 32), new THREE.MeshBasicMaterial({ color: COL.drone.clone().multiplyScalar(2) }));
      rotor.rotation.x = Math.PI / 2;
      rotor.position.y = 0.12;
      g.add(body, eye, rotor);
      const marker = new THREE.Mesh(new THREE.RingGeometry(0.3, 0.42, 4), new THREE.MeshBasicMaterial({ color: COL.drone, transparent: true, opacity: 0.6, side: THREE.DoubleSide }));
      marker.rotation.x = -Math.PI / 2;
      marker.rotation.z = Math.PI / 4;
      const dots = [];
      for (let k = 0; k < 6; k++) {
        const dot = new THREE.Mesh(new THREE.CircleGeometry(0.08, 12), new THREE.MeshBasicMaterial({ color: COL.drone, transparent: true, opacity: 0.5 - k * 0.07 }));
        dot.rotation.x = -Math.PI / 2;
        this.level.add(dot);
        dots.push(dot);
      }
      this.level.add(g, marker);
      return { g, rotor, marker, dots };
    });
    for (const v of this.foodViews.values()) this.scene.remove(v.obj);
    this.foodViews.clear();
  }

  // ------------------------------------------------------------ snakes
  _buildSnakes() {
    this.snakeViews = {};
    for (const id of ["human", "ai"]) {
      const color = COL[id], dark = COL[id + "Dark"];
      const mat = new THREE.MeshStandardMaterial({ color: dark, emissive: color, emissiveIntensity: 0.6, roughness: 0.25, metalness: 0.3 });
      const chains = [new TubeChain(mat), new TubeChain(mat), new TubeChain(mat)];
      chains.forEach((c) => this.scene.add(c.mesh));
      const head = new THREE.Group();
      const headMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: color, emissiveIntensity: 1.3, roughness: 0.2 });
      head.add(new THREE.Mesh(new THREE.SphereGeometry(0.38, 28, 20), headMat));
      for (const side of [-1, 1]) {
        const eye = new THREE.Mesh(new THREE.SphereGeometry(0.075, 12, 8), new THREE.MeshBasicMaterial({ color: 0x020617 }));
        eye.position.set(side * 0.15, 0.17, 0.27);
        head.add(eye);
      }
      const light = new THREE.PointLight(color, 3, 5, 2);
      light.position.y = 0.8;
      head.add(light);
      const label = labelSprite(id === "human" ? "ВЫ" : "LAYA", id === "human" ? "#67e8f9" : "#f9a8d4");
      this.scene.add(head, label);
      this.snakeViews[id] = { mat, headMat, chains, head, label, color, dark, yaw: 0 };
    }
  }

  /** Grid cell under a screen point (ray vs. floor plane), or null. */
  cellAt(clientX, clientY) {
    const r = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    const hit = new THREE.Vector3();
    if (!ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), hit)) return null;
    const x = Math.round(hit.x + HALF), y = Math.round(hit.z + HALF);
    return x >= 0 && y >= 0 && x < GRID && y < GRID ? { x, y } : null;
  }
  setTarget(cell) {
    if (!this.targetRing) {
      this.targetRing = new THREE.Mesh(new THREE.RingGeometry(0.32, 0.42, 40),
        new THREE.MeshBasicMaterial({ color: COL.human.clone().multiplyScalar(2.2), transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false }));
      this.targetRing.rotation.x = -Math.PI / 2;
      this.scene.add(this.targetRing);
    }
    this.targetRing.visible = !!cell;
    if (cell) this.targetRing.position.set(wx(cell.x), 0.05, wz(cell.y));
  }

  snapshot(match) {
    this.prev = {
      human: match.human.alive ? match.human.cells.map((c) => ({ ...c })) : null,
      ai: match.ai.alive ? match.ai.cells.map((c) => ({ ...c })) : null,
      drones: match.drones.map((d) => ({ x: d.x, y: d.y })),
    };
  }

  _lerpCells(s, t) {
    const prev = this.prev?.[s.id];
    return s.cells.map((p, i) => {
      const q = prev?.[i] ?? prev?.[prev.length - 1] ?? p;
      if (!prev || Math.abs(q.x - p.x) + Math.abs(q.y - p.y) > 1) return { x: p.x, y: p.y };
      return { x: q.x + (p.x - q.x) * t, y: q.y + (p.y - q.y) * t };
    });
  }

  _updateSnake(s, t, now) {
    const v = this.snakeViews[s.id];
    const visible = s.alive;
    v.head.visible = v.label.visible = visible;
    if (!visible) { v.chains.forEach((c) => (c.mesh.visible = false)); return; }
    const pts = this._lerpCells(s, t);
    // split into contiguous runs (portal jumps break the tube)
    const runs = [[]];
    pts.forEach((p, i) => {
      if (i && Math.abs(p.x - pts[i - 1].x) + Math.abs(p.y - pts[i - 1].y) > 1.01) runs.push([]);
      runs[runs.length - 1].push({ p, i });
    });
    const n = pts.length;
    v.chains.forEach((chain, k) => {
      const run = runs[k];
      if (!run || run.length < 2) { chain.mesh.visible = false; return; }
      const curve = new THREE.CatmullRomCurve3(run.map(({ p }) => new THREE.Vector3(wx(p.x), 0.32, wz(p.y))), false, "centripetal");
      const samples = curve.getPoints(Math.max(4, run.length * 5)).map((q, j, arr) => {
        const idx = run[0].i + (j / (arr.length - 1)) * (run.length - 1);
        return { x: q.x, y: 0.32, z: q.z, r: 0.3 - 0.13 * (idx / Math.max(1, n - 1)) };
      });
      chain.build(samples);
    });
    // head
    const h = pts[0], h1 = pts[1] ?? { x: h.x - DELTA[s.dir][0], y: h.y - DELTA[s.dir][1] };
    v.head.position.set(wx(h.x), 0.36, wz(h.y));
    const yaw = Math.atan2(h.x - h1.x, h.y - h1.y);
    const target = Math.abs(h.x - h1.x) + Math.abs(h.y - h1.y) > 1.01 ? Math.atan2(DELTA[s.dir][0], DELTA[s.dir][1]) : yaw;
    let d = target - v.yaw;
    d = ((d + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    v.yaw += d * 0.35;
    v.head.rotation.y = v.yaw;
    v.label.position.set(wx(h.x), 1.25, wz(h.y));
    // immortality: rainbow shimmer, flicker in the last 8 steps
    if (s.immortal > 0) {
      const ending = s.immortal <= 8 && Math.floor(now / 110) % 2 === 0;
      const hue = ((now / 900) % 1);
      v.mat.emissive.setHSL(ending ? 0 : hue, 0.95, ending ? 0.35 : 0.6);
      v.mat.emissiveIntensity = ending ? 0.6 : 1.6;
      v.headMat.emissive.setHSL(hue, 0.95, 0.65);
      if (Math.random() < 0.35) this.particles.burst(new THREE.Vector3(wx(h.x), 0.5, wz(h.y)), new THREE.Color().setHSL(hue, 1, 0.65), 1, { speed: 0.6, up: 1.2, life: 0.6, size: 0.1, gravity: 0 });
    } else {
      v.mat.emissive.copy(v.color);
      v.mat.emissiveIntensity = 0.6;
      v.headMat.emissive.copy(v.color);
    }
  }

  // ------------------------------------------------------------ food
  _foodObj(f) {
    const color = COL.food[f.type];
    let obj;
    if (f.type === "immortal") {
      obj = new THREE.Mesh(new THREE.ExtrudeGeometry(starShape(), { depth: 0.12, bevelEnabled: true, bevelSize: 0.03, bevelThickness: 0.03, bevelSegments: 2 }),
        new THREE.MeshStandardMaterial({ color: 0x664400, emissive: color, emissiveIntensity: 2.2, roughness: 0.2, metalness: 0.6 }));
      obj.geometry.center();
    } else if (f.type === "poison") {
      obj = new THREE.Mesh(new THREE.IcosahedronGeometry(0.27, 0),
        new THREE.MeshStandardMaterial({ color: 0x1a2e05, emissive: color, emissiveIntensity: 0.9, flatShading: true, roughness: 0.6 }));
    } else {
      obj = new THREE.Mesh(new THREE.SphereGeometry(f.type === "golden" ? 0.3 : 0.25, 24, 16),
        new THREE.MeshStandardMaterial({ color: color.clone().multiplyScalar(0.3), emissive: color, emissiveIntensity: f.type === "golden" ? 1.8 : 1.3, roughness: 0.3 }));
      if (f.type === "golden") {
        const ring = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.02, 6, 40), new THREE.MeshBasicMaterial({ color: color.clone().multiplyScalar(2) }));
        ring.rotation.x = Math.PI / 2;
        obj.add(ring);
      }
    }
    obj.position.set(wx(f.x), 0.45, wz(f.y));
    obj.scale.setScalar(0.01);
    this.scene.add(obj);
    return { obj, f, born: performance.now() };
  }

  _updateFoods(match, now) {
    const alive = new Set();
    for (const f of match.foods) {
      const k = `${f.x},${f.y},${f.type},${f.born}`;
      alive.add(k);
      let v = this.foodViews.get(k);
      if (!v) { v = this._foodObj(f); this.foodViews.set(k, v); }
      v.obj.visible = match.visibleTo(match.human, f.x, f.y);
      const age = Math.min(1, (now - v.born) / 350);
      const fading = f.ttl !== null && f.ttl < 12 && Math.floor(now / 120) % 2 === 0;
      v.obj.scale.setScalar((0.2 + 0.8 * age) * (fading ? 0.55 : 1));
      v.obj.position.y = 0.45 + Math.sin(now / 300 + f.x) * 0.07;
      v.obj.rotation.y = now / (f.type === "immortal" ? 500 : 1400);
      if (f.type === "poison") v.obj.rotation.x = now / 900;
    }
    for (const [k, v] of this.foodViews) {
      if (!alive.has(k)) { this.scene.remove(v.obj); v.obj.geometry.dispose(); this.foodViews.delete(k); }
    }
  }

  // ------------------------------------------------------------ events
  handleEvents(events) {
    for (const e of events) {
      const at = (x, y, h = 0.4) => new THREE.Vector3(wx(x), h, wz(y));
      if (e.type === "eat") {
        const c = COL.food[e.food];
        this.particles.burst(at(e.x, e.y), c, e.food === "immortal" ? 90 : 34, { speed: e.food === "immortal" ? 4.5 : 3 });
        this._ring(e.x, e.y, c);
      } else if (e.type === "death") {
        const c = COL[e.who];
        e.cells.forEach((p) => this.particles.burst(at(p.x, p.y, 0.35), c, 10, { speed: 4, up: 4 }));
        this.shake = Math.max(this.shake, 0.45);
        this.flash = Math.max(this.flash, 0.6);
      } else if (e.type === "kill") {
        this.flash = 1;
      } else if (e.type === "respawn") {
        this._beam(e.x, e.y, COL[e.who]);
      } else if (e.type === "teleport") {
        this._ring(e.x, e.y, COL.portal[1]);
        this.particles.burst(at(e.x, e.y), COL.portal[1], 24);
      } else if (e.type === "wall") {
        e.cells.forEach(([x, y]) => this.particles.burst(at(x, y, 0.5), e.phase === "GHOST" ? COL.ghost : COL.wall, 8, { speed: 2 }));
      } else if (e.type === "shrug") {
        this.particles.burst(at(e.x, e.y), COL.food.poison, 20);
      }
    }
  }

  _ring(x, y, color) {
    const m = new THREE.Mesh(new THREE.RingGeometry(0.3, 0.38, 48), new THREE.MeshBasicMaterial({ color: color.clone().multiplyScalar(2), transparent: true, side: THREE.DoubleSide, depthWrite: false }));
    m.rotation.x = -Math.PI / 2;
    m.position.set(wx(x), 0.04, wz(y));
    this.scene.add(m);
    this.fx.push({ obj: m, life: 1, kind: "ring" });
  }

  _beam(x, y, color) {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.45, 8, 24, 1, true),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.5, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending }));
    m.position.set(wx(x), 4, wz(y));
    this.scene.add(m);
    this.fx.push({ obj: m, life: 1, kind: "beam" });
  }

  _updateFx(dt) {
    for (const f of this.fx) {
      f.life -= dt * (f.kind === "beam" ? 1.2 : 1.8);
      if (f.kind === "ring") f.obj.scale.setScalar(1 + (1 - f.life) * 4);
      else f.obj.scale.set(1 - (1 - f.life) * 0.7, 1, 1 - (1 - f.life) * 0.7);
      f.obj.material.opacity = Math.max(0, f.life) * (f.kind === "beam" ? 0.5 : 1);
    }
    for (const f of this.fx.filter((f) => f.life <= 0)) { this.scene.remove(f.obj); f.obj.geometry.dispose(); f.obj.material.dispose(); }
    this.fx = this.fx.filter((f) => f.life > 0);
  }

  // ------------------------------------------------------------ frame
  draw(match, t, now) {
    const dt = Math.min(0.05, (now - this.lastNow) / 1000);
    this.lastNow = now;
    if (this.levelSeed !== match.seed) this._buildLevel(match);
    this.floor.material.uniforms.uTime.value = now / 1000;

    // walls: smooth SOLID <-> GHOST transition, strobe while BLINKING
    for (const w of this.wallViews) {
      const target = w.st.phase === "GHOST" ? 0 : 1;
      w.vis += (target - w.vis) * damp(10, dt);
      const blink = w.st.phase === "BLINKING" && Math.floor(now / 90) % 2 === 0;
      w.solidMat.opacity = 0.08 + 0.92 * w.vis;
      w.solidMat.emissiveIntensity = blink ? 0.25 : 1.1 * w.vis + 0.1;
      w.wireMat.opacity = (1 - w.vis) * (0.55 + 0.25 * Math.sin(now / 160));
      w.group.children.forEach((m, i) => { if (i % 2 === 0) m.scale.y = 0.25 + 0.75 * w.vis; });
      w.group.children.forEach((m, i) => { if (i % 2 === 0) m.position.y = 0.45 * (0.25 + 0.75 * w.vis); });
    }
    this.portalViews.forEach((g, i) => g.children.forEach((c, k) => { if (k < 3) c.rotation.z = (now / 500) * (k % 2 ? -1 : 1) + i; }));

    // drones
    match.drones.forEach((d, i) => {
      const v = this.droneViews[i], p = this.prev?.drones[i] ?? d;
      v.g.position.set(wx(p.x + (d.x - p.x) * t), 0.55 + Math.sin(now / 250 + i) * 0.06, wz(p.y + (d.y - p.y) * t));
      v.rotor.rotation.z = now / 40;
      v.g.rotation.y = Math.atan2(d.dx, d.dy);
      const [ndx, ndy] = d.plan(match.solid);
      v.marker.visible = !!(ndx || ndy);
      v.marker.position.set(wx(d.x + ndx), 0.03, wz(d.y + ndy));
      const ghost = d.clone();
      v.dots.forEach((dot) => { ghost.step(match.solid); dot.position.set(wx(ghost.x), 0.025, wz(ghost.y)); });
    });

    this._updateFoods(match, now);
    for (const s of match.snakes) this._updateSnake(s, t, now);
    this.fogAmt += ((match.fog.active ? 1 : 0) - this.fogAmt) * damp(2.5, dt);
    this.fogSheet.visible = this.fogAmt > 0.01;
    const hh = this.snakeViews.human.head.position;
    if (match.human.alive) this.fogMat.uniforms.uCenter.value.set(hh.x, hh.z);
    this.fogMat.uniforms.uR.value = match.human.alive ? FOG_RADIUS + 0.5 : -1;
    this.fogMat.uniforms.uAmt.value = this.fogAmt;
    this.fogMat.uniforms.uTime.value = now / 1000;
    this.humanAlive = match.human.alive;
    this.particles.update(dt);
    this._updateFx(dt);
    if (this.targetRing?.visible) this.targetRing.scale.setScalar(1 + 0.12 * Math.sin(now / 150));
    this._updateCamera(dt, now);

    this.flash *= Math.exp(-4 * dt);
    this.bloom.strength = 0.55 + this.flash * 1.2;
    if (this.bloomOn) this.composer.render(); else this.renderer.render(this.scene, this.camera);
    this._watchFps(now);
  }

  _updateCamera(dt, now) {
    if (this.view === "chase" && this.humanAlive) {
      const v = this.snakeViews.human, h = v.head.position;
      const fwd = new THREE.Vector3(Math.sin(v.yaw), 0, Math.cos(v.yaw));
      const pos = h.clone().addScaledVector(fwd, -6.5).add(new THREE.Vector3(0, 7.2, 0));
      const look = h.clone().addScaledVector(fwd, 4.5);
      this.camPos = this.camPos ? this.camPos.lerp(pos, damp(5, dt)) : pos;
      this.camLook = this.camLook ? this.camLook.lerp(look, damp(6, dt)) : look;
      this.camera.position.copy(this.camPos);
      if (this.shake > 0.01) { this.camera.position.x += (Math.random() - 0.5) * this.shake; this.shake *= Math.exp(-6 * dt); }
      this.camera.lookAt(this.camLook);
      return;
    }
    const fovV = THREE.MathUtils.degToRad(this.camera.fov);
    const half = GRID / 2 + 1.2;
    // radians from vertical; wide screens (landscape phones) tilt more so the board can use the width
    const tilt = this.view === "top" ? 0.001 : this.camera.aspect > 1.6 ? 0.85 : 0.62;
    // distance so the whole board fits both vertically (foreshortened) and horizontally
    const dV = (half * (Math.cos(tilt) + Math.sin(tilt) * 0.55)) / Math.tan(fovV / 2);
    const dH = half / (Math.tan(fovV / 2) * this.camera.aspect);
    let dist = Math.max(dV, dH) * 0.95;
    let yaw = 0;
    if (this.idle) yaw = Math.sin(now / 6000) * 0.35;
    const target = new THREE.Vector3(0, 0, this.view === "top" ? 0 : 0.4);
    const pos = new THREE.Vector3(Math.sin(yaw) * Math.sin(tilt) * dist, Math.cos(tilt) * dist, Math.cos(yaw) * Math.sin(tilt) * dist).add(target);
    this.camPos = this.camPos ? this.camPos.lerp(pos, damp(4, dt)) : pos;
    this.camera.position.copy(this.camPos);
    if (this.shake > 0.01) {
      this.camera.position.x += (Math.random() - 0.5) * this.shake;
      this.camera.position.y += (Math.random() - 0.5) * this.shake;
      this.shake *= Math.exp(-6 * dt);
    }
    this.camLook = target.clone();
    this.camera.lookAt(target);
  }

  _watchFps(now) {
    const f = this.fps;
    f.frames++;
    if (now - f.since > 3000) {
      const fps = (f.frames * 1000) / (now - f.since);
      if (fps < 40 && !f.low) { // weak device: no bloom, pixel ratio 1
        f.low = true;
        this.bloomOn = false;
        this.renderer.setPixelRatio(1);
        this.resize();
      }
      f.frames = 0;
      f.since = now;
    }
  }
}
