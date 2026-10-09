import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';

const BLUE = new THREE.Color('#5b9bff'), ORANGE = new THREE.Color('#ff6b2c'), WHITE = new THREE.Color('#e9edf5');
const ease = (x) => 1 - Math.pow(1 - Math.min(1, Math.max(0, x)), 3);

function dotTex() {
  const c = document.createElement('canvas'); c.width = c.height = 64; const g = c.getContext('2d');
  const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.3, 'rgba(255,255,255,.8)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = gr; g.fillRect(0, 0, 64, 64); return new THREE.CanvasTexture(c);
}

function base(el, fov) {
  const r = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  r.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  r.domElement.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block';
  el.appendChild(r.domElement);
  const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(fov, 1, 0.1, 100);
  const fit = () => { const w = el.clientWidth || 1, h = el.clientHeight || 1; r.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); };
  fit(); const ro = new ResizeObserver(fit); ro.observe(el);
  const labels = [], v = new THREE.Vector3();
  const label = (text, pos, css = '', parent = null) => {
    const d = document.createElement('div'); d.textContent = text;
    d.style.cssText = 'position:absolute;left:0;top:0;pointer-events:none;white-space:nowrap;font:500 10.5px/1.2 "Geist Mono",ui-monospace,monospace;letter-spacing:.08em;text-transform:uppercase;color:#a7adb8;' + css;
    el.appendChild(d); const o = { d, pos: pos.clone(), parent }; labels.push(o); return o;
  };
  const place = () => {
    const w = el.clientWidth, h = el.clientHeight;
    labels.forEach((o) => { v.copy(o.pos); if (o.parent) o.parent.localToWorld(v); v.project(cam);
      o.d.style.transform = `translate(${(v.x * 0.5 + 0.5) * w}px,${(-v.y * 0.5 + 0.5) * h}px) translate(-50%,-50%)`; });
  };
  let raf = 0, alive = true;
  const loop = (fn) => { let last = performance.now();
    const tick = (t) => { if (!alive) return; const dt = Math.min(0.05, (t - last) / 1000); last = t; fn(dt, t / 1000); place(); r.render(scene, cam); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick); };
  const dispose = () => { alive = false; cancelAnimationFrame(raf); ro.disconnect(); r.dispose(); r.domElement.remove(); labels.forEach((o) => o.d.remove()); };
  return { scene, cam, label, loop, dispose };
}

export function mountHero(el, opts = {}) {
  const B = base(el, 32), { scene, cam } = B, tex = dotTex();
  let mode = opts.mode || 'published', speed = opts.motion === false ? 0.2 : 1;
  const grid = new THREE.GridHelper(26, 52, 0x232a34, 0x151a20); grid.position.y = -2.4; scene.add(grid);
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-9, -2.38, 3), new THREE.Vector3(9, -2.38, 3)]), new THREE.LineBasicMaterial({ color: 0x4a525e })));
  const PH = 4.6, PD = 6.4;
  const planeMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.03, side: THREE.DoubleSide, depthWrite: false });
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(PD, PH), planeMat); plane.rotation.y = Math.PI / 2; plane.position.y = -0.1; scene.add(plane);
  const edgeMat = new THREE.LineBasicMaterial({ color: ORANGE.clone(), transparent: true, opacity: 0.9 });
  const edge = new THREE.LineSegments(new THREE.EdgesGeometry(plane.geometry), edgeMat); edge.rotation.copy(plane.rotation); edge.position.copy(plane.position); scene.add(edge);
  const hp = []; for (let i = 1; i < 10; i++) { const y = -PH / 2 + i * PH / 10; hp.push(new THREE.Vector3(0, y, -PD / 2), new THREE.Vector3(0, y, PD / 2)); }
  const hatchMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.05 });
  const hatch = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(hp), hatchMat); hatch.position.y = plane.position.y; scene.add(hatch);

  const CORE = new THREE.Vector3(-3.6, 0, 0), core = new THREE.Group(); core.position.copy(CORE); scene.add(core);
  const wireMat = new THREE.LineBasicMaterial({ color: BLUE.clone(), transparent: true, opacity: 0.8 });
  const wire = new THREE.LineSegments(new THREE.WireframeGeometry(new THREE.IcosahedronGeometry(1.05, 1)), wireMat); core.add(wire);
  const innerMat = new THREE.MeshBasicMaterial({ color: BLUE.clone(), transparent: true, opacity: 0.2 });
  const inner = new THREE.Mesh(new THREE.IcosahedronGeometry(0.6, 0), innerMat); core.add(inner);
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, color: BLUE.clone(), transparent: true, opacity: 0.32, depthWrite: false, blending: THREE.AdditiveBlending })); halo.scale.set(4.2, 4.2, 1); core.add(halo);

  const N = 560, posA = new Float32Array(N * 3), colA = new Float32Array(N * 3), P = [];
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(posA, 3)); geo.setAttribute('color', new THREE.BufferAttribute(colA, 3));
  scene.add(new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.14, map: tex, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending })));
  const rnd = (a, b) => a + Math.random() * (b - a);
  const spawn = (p, init) => {
    p.future = Math.random() < 0.46; p.state = 0; p.a = 0; p.s = rnd(0.7, 1.15);
    if (p.future) p.p.set(init ? rnd(0.4, 8.5) : rnd(6.5, 9), rnd(-2, 2), rnd(-3, 3));
    else { do { p.p.set(rnd(-9, -0.5), rnd(-2, 2), rnd(-3, 3)); } while (p.p.distanceTo(CORE) < 2.8); }
    p.vel.set(0, 0, 0);
  };
  for (let i = 0; i < N; i++) { const p = { p: new THREE.Vector3(), vel: new THREE.Vector3() }; spawn(p, true); p.a = Math.random(); P.push(p); }

  const rings = [...Array(30)].map(() => { const m = new THREE.Mesh(new THREE.RingGeometry(0.06, 0.1, 40), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending }));
    m.rotation.y = Math.PI / 2; m.visible = false; scene.add(m); return { m, life: 0, s: 1 }; });
  let ri = 0;
  const ring = (p, color, s = 1) => { const r = rings[ri++ % rings.length]; r.m.position.set(0.01, p.y, p.z); r.m.material.color.copy(color); r.life = 1; r.s = s; r.m.visible = true; };

  B.label('Past · features', new THREE.Vector3(-5.8, -2.38, 3.4));
  B.label('Prediction cut-off', new THREE.Vector3(0, PH / 2 + 0.2, 0), 'color:#eceef2');
  B.label('Future · after cut-off', new THREE.Vector3(5.4, -2.38, 3.4), 'color:#ff8a55');
  B.label('Model', new THREE.Vector3(-3.6, 1.65, 0), 'color:#eceef2');

  let tx = 0, ty = 0, mx = 0, my = 0, contam = mode === 'published' ? 0.6 : 0, pulse = 0, kk = mode === 'honest' ? 1 : 0;
  const onMove = (e) => { const r = el.getBoundingClientRect(); tx = ((e.clientX - r.left) / r.width) * 2 - 1; ty = ((e.clientY - r.top) / r.height) * 2 - 1; };
  el.addEventListener('pointermove', onMove);
  const d = new THREE.Vector3(), sw = new THREE.Vector3(), tmp = new THREE.Color(), look = new THREE.Vector3(-0.9, -0.3, 0);

  B.loop((dt0, t) => {
    const dt = dt0 * speed;
    for (let i = 0; i < N; i++) {
      const p = P[i];
      d.subVectors(CORE, p.p); const dist = d.length(); d.divideScalar(dist || 1);
      if (p.state === 1) { p.p.addScaledVector(p.vel, dt); p.a -= dt * 0.8; if (p.a <= 0) spawn(p); }
      else {
        sw.set(-d.z, 0, d.x).multiplyScalar(0.35);
        d.multiplyScalar(p.s).add(sw); p.vel.lerp(d, Math.min(1, dt * 1.5));
        const px = p.p.x; p.p.addScaledVector(p.vel, dt); p.a = Math.min(1, p.a + dt * 1.2);
        if (p.future && p.state === 0 && px > 0 && p.p.x <= 0) {
          if (mode === 'honest') { p.state = 1; p.p.x = 0.02; p.vel.set(Math.abs(p.vel.x) * 0.5 + 0.5, p.vel.y * 0.3, p.vel.z * 0.3); ring(p.p, WHITE); }
          else { p.state = 2; if (Math.random() < 0.4) ring(p.p, ORANGE, 0.6); }
        }
        if (dist < 0.85) { if (p.future) { contam = Math.min(1, contam + 0.07); pulse = 1; } spawn(p); }
      }
      const c = p.future ? ORANGE : BLUE, a = Math.max(0, p.a) * (p.state === 2 ? 1.2 : 0.9);
      posA[i * 3] = p.p.x; posA[i * 3 + 1] = p.p.y; posA[i * 3 + 2] = p.p.z;
      colA[i * 3] = c.r * a; colA[i * 3 + 1] = c.g * a; colA[i * 3 + 2] = c.b * a;
    }
    geo.attributes.position.needsUpdate = true; geo.attributes.color.needsUpdate = true;
    rings.forEach((r) => { if (!r.m.visible) return; r.life -= dt0 * 1.3; if (r.life <= 0) { r.m.visible = false; return; }
      r.m.scale.setScalar(1 + (1 - r.life) * 6 * r.s); r.m.material.opacity = r.life * 0.9; });
    contam = Math.max(0, contam - dt * 0.14); pulse = Math.max(0, pulse - dt0 * 3);
    tmp.copy(BLUE).lerp(ORANGE, contam); wireMat.color.copy(tmp); innerMat.color.copy(tmp); halo.material.color.copy(tmp);
    core.scale.setScalar(1 + pulse * 0.07); wire.rotation.y += dt0 * 0.22 * speed; wire.rotation.x += dt0 * 0.08 * speed; inner.rotation.y -= dt0 * 0.3 * speed;
    kk += ((mode === 'honest' ? 1 : 0) - kk) * Math.min(1, dt0 * 3);
    planeMat.opacity = 0.02 + kk * 0.08; hatchMat.opacity = 0.04 + kk * 0.12; edgeMat.color.copy(ORANGE).lerp(WHITE, kk);
    mx += (tx - mx) * dt0 * 2; my += (ty - my) * dt0 * 2;
    cam.position.set(5.4 + mx * 1.4 + Math.sin(t * 0.12) * 0.5 * speed, 2.7 - my * 0.7, 11.8); cam.lookAt(look);
  });
  return {
    setMode(m) { mode = m; },
    setMotion(on) { speed = on ? 1 : 0.2; },
    dispose() { el.removeEventListener('pointermove', onMove); B.dispose(); },
  };
}

export function mountMatrix(el, rows, opts = {}) {
  const B = base(el, 30), { scene, cam } = B;
  let auto = opts.autoRotate !== false, motion = opts.motion !== false, t0 = null;
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const dl = new THREE.DirectionalLight(0xffffff, 1.2); dl.position.set(4, 9, 6); scene.add(dl);
  const dl2 = new THREE.DirectionalLight(0x8fb4ff, 0.4); dl2.position.set(-6, 3, -5); scene.add(dl2);
  const g = new THREE.Group(); scene.add(g);
  const S = 1.55, K = 5.2, nr = rows.length, nc = Math.max(...rows.map((r) => r.feats.length));
  const ox = ((nc - 1) * S) / 2, oz = ((nr - 1) * S) / 2;
  const plateGeo = new THREE.BoxGeometry(nc * S + 0.8, 0.08, nr * S + 0.8);
  const plate = new THREE.Mesh(plateGeo, new THREE.MeshStandardMaterial({ color: 0x12161c, roughness: 0.9 })); plate.position.y = -0.05; g.add(plate);
  const pe = new THREE.LineSegments(new THREE.EdgesGeometry(plateGeo), new THREE.LineBasicMaterial({ color: 0x2a313b })); pe.position.y = -0.05; g.add(pe);
  const gp = []; for (let i = 0; i <= nc; i++) { const x = i * S - ox - S / 2; gp.push(new THREE.Vector3(x, 0.001, -oz - S / 2), new THREE.Vector3(x, 0.001, oz + S / 2)); }
  for (let j = 0; j <= nr; j++) { const z = j * S - oz - S / 2; gp.push(new THREE.Vector3(-ox - S / 2, 0.001, z), new THREE.Vector3(ox + S / 2, 0.001, z)); }
  g.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(gp), new THREE.LineBasicMaterial({ color: 0x1e242c })));
  const bars = []; let idx = 0;
  rows.forEach((r, ri) => {
    B.label('Case ' + r.key, new THREE.Vector3(-ox - S * 1.05, 0.05, ri * S - oz), 'color:#eceef2', g);
    r.feats.forEach((f, ci) => {
      const x = ci * S - ox, z = ri * S - oz, box = new THREE.BoxGeometry(0.8, 1, 0.8);
      const hb = new THREE.Mesh(box, new THREE.MeshStandardMaterial({ color: 0x5b9bff, roughness: 0.45, metalness: 0.1 }));
      const lb = new THREE.Mesh(box, new THREE.MeshStandardMaterial({ color: 0xff6b2c, roughness: 0.25, transparent: true, opacity: 0.5, emissive: 0xff6b2c, emissiveIntensity: 0.3, depthWrite: false }));
      lb.add(new THREE.LineSegments(new THREE.EdgesGeometry(box), new THREE.LineBasicMaterial({ color: 0xff8a55, transparent: true, opacity: 0.95 })));
      hb.position.set(x, 0, z); lb.position.set(x, 0, z); g.add(hb, lb);
      const lab = B.label(`${f.short}  −${(f.pub - f.hon).toFixed(2)}`, new THREE.Vector3(x, 0, z), 'color:#ffb08a;text-transform:none;letter-spacing:0;font-size:10px;', g);
      bars.push({ hb, lb, f, x, z, lab, i: idx++ });
    });
  });
  let rotY = -0.6, drag = null;
  el.style.cursor = 'grab'; el.style.touchAction = 'pan-y';
  const down = (e) => { drag = { x: e.clientX, r: rotY }; el.style.cursor = 'grabbing'; el.setPointerCapture(e.pointerId); };
  const move = (e) => { if (drag) rotY = drag.r + (e.clientX - drag.x) * 0.008; };
  const up = () => { drag = null; el.style.cursor = 'grab'; };
  el.addEventListener('pointerdown', down); el.addEventListener('pointermove', move); el.addEventListener('pointerup', up); el.addEventListener('pointercancel', up);
  const look = new THREE.Vector3(0, 1.1, 0);
  B.loop((dt) => {
    if (!drag && auto && motion) rotY += dt * 0.14;
    g.rotation.y = rotY;
    const el2 = t0 === null ? 0 : motion ? (performance.now() - t0) / 1000 : 99;
    bars.forEach((b) => {
      const e1 = ease((el2 - b.i * 0.07) / 0.9), e2 = ease((el2 - 0.8 - b.i * 0.07) / 0.9);
      const h1 = Math.max(0.001, b.f.hon * K * e1), h2 = Math.max(0.001, (b.f.pub - b.f.hon) * K * e2);
      b.hb.scale.y = h1; b.hb.position.y = h1 / 2; b.lb.scale.y = h2; b.lb.position.y = h1 + h2 / 2;
      b.lab.pos.set(b.x, h1 + h2 + 0.35, b.z); b.lab.d.style.opacity = e2;
    });
    cam.position.set(7.2, 6.2, 8.4); cam.lookAt(look);
  });
  return {
    play() { if (t0 === null) t0 = performance.now(); },
    setAutoRotate(v) { auto = v; },
    dispose() { el.removeEventListener('pointerdown', down); el.removeEventListener('pointermove', move); el.removeEventListener('pointerup', up); el.removeEventListener('pointercancel', up); B.dispose(); },
  };
}
