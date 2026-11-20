// Moteur d'inférence maison sur la carte graphique (WebGPU).
// Mêmes calculs que hanzi-net.js, mais tous les caractères d'une image passent en un seul lot :
// chaque couche est un petit programme (« shader ») exécuté en parallèle par des milliers de cœurs.

const WG = 64;           // taille d'un groupe de travail
const MAX_BATCH = 64;    // caractères par lot (raisonnable aussi pour la mémoire d'un téléphone)

const HEADER = `
struct P { B: u32, C: u32, H: u32, W: u32, Co: u32, relu: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
fn index(g: vec3<u32>, n: vec3<u32>) -> u32 { return g.x + g.y * n.x * ${WG}u; }
`;

// Convolution 3×3, remplissage 1, + biais + ReLU. Une invocation = une valeur de sortie.
const CONV = HEADER + `
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
  let idx = index(g, n);
  let plane = p.H * p.W;
  if (idx >= p.B * p.Co * plane) { return; }
  let x = i32(idx % p.W);
  let y = i32((idx / p.W) % p.H);
  let co = (idx / plane) % p.Co;
  let b = idx / (plane * p.Co);
  var s = bias[co];
  for (var ci = 0u; ci < p.C; ci++) {
    let ib = (b * p.C + ci) * plane;
    let wb = (co * p.C + ci) * 9u;
    for (var dy = 0; dy < 3; dy++) {
      let yy = y + dy - 1;
      if (yy < 0 || yy >= i32(p.H)) { continue; }
      for (var dx = 0; dx < 3; dx++) {
        let xx = x + dx - 1;
        if (xx < 0 || xx >= i32(p.W)) { continue; }
        s += wt[wb + u32(dy * 3 + dx)] * inp[ib + u32(yy) * p.W + u32(xx)];
      }
    }
  }
  outp[idx] = max(s, 0.0);
}`;

// Max-pooling 2×2 (p.H, p.W = taille d'entrée).
const POOL = HEADER + `
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
  let idx = index(g, n);
  let h = p.H / 2u; let w = p.W / 2u;
  if (idx >= p.B * p.C * h * w) { return; }
  let x = idx % w; let y = (idx / w) % h; let bc = idx / (w * h);
  let i = bc * p.H * p.W + 2u * y * p.W + 2u * x;
  outp[idx] = max(max(inp[i], inp[i + 1u]), max(inp[i + p.W], inp[i + p.W + 1u]));
}`;

// Couche dense (p.C entrées, p.Co sorties) + biais (+ ReLU si p.relu).
const DENSE = HEADER + `
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
  let idx = index(g, n);
  if (idx >= p.B * p.Co) { return; }
  let o = idx % p.Co; let b = idx / p.Co;
  var s = bias[o];
  let wb = o * p.C; let ib = b * p.C;
  for (var i = 0u; i < p.C; i++) { s += wt[wb + i] * inp[ib + i]; }
  if (p.relu == 1u) { s = max(s, 0.0); }
  outp[idx] = s;
}`;

export async function createGpuNet(net) {
  if (!navigator.gpu) throw new Error("WebGPU absent");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("pas de carte graphique WebGPU");
  const device = await adapter.requestDevice();
  const S = 32 * 48 * 48; // plus grande activation par image
  const mk = (size, usage) => device.createBuffer({ size: Math.max(16, size), usage });
  const upload = (arr) => {
    const b = mk(arr.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(b, 0, arr);
    return b;
  };
  const pipe = (code) => device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const pipes = { conv: pipe(CONV), pool: pipe(POOL), dense: pipe(DENSE) };
  const nOut = net.layers[net.layers.length - 1].out;
  const bufA = mk(MAX_BATCH * S * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const bufB = mk(MAX_BATCH * S * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const read = mk(MAX_BATCH * nOut * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const layers = net.layers.map((l) => (l.type === "pool" ? l : { ...l, gw: upload(l.wf), gb: upload(l.bf) }));

  async function run(inputs) {
    const B = inputs.length;
    const x = new Float32Array(B * 48 * 48);
    inputs.forEach((img, k) => { for (let i = 0; i < img.length; i++) x[k * 2304 + i] = (img[i] - net.mean) / net.std; });
    device.queue.writeBuffer(bufA, 0, x);
    const enc = device.createCommandEncoder();
    let src = bufA, dst = bufB, C = 1, H = 48, W = 48;
    const params = [];
    for (const l of layers) {
      let kind, n, u;
      if (l.type === "conv") { kind = "conv"; n = B * l.cout * H * W; u = [B, C, H, W, l.cout, 1, 0, 0]; }
      else if (l.type === "pool") { kind = "pool"; n = B * C * (H >> 1) * (W >> 1); u = [B, C, H, W, C, 0, 0, 0]; }
      else { kind = "dense"; n = B * l.out; u = [B, l.in, 1, 1, l.out, l.relu ? 1 : 0, 0, 0]; }
      const ub = mk(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(ub, 0, new Uint32Array(u));
      params.push(ub);
      // Le max-pooling n'a pas de poids : sa disposition automatique ne contient pas les liaisons 2 et 3.
      const entries = [
        { binding: 0, resource: { buffer: ub } },
        { binding: 1, resource: { buffer: src } },
        { binding: 4, resource: { buffer: dst } },
      ];
      if (kind !== "pool") entries.push({ binding: 2, resource: { buffer: l.gw } }, { binding: 3, resource: { buffer: l.gb } });
      const bind = device.createBindGroup({ layout: pipes[kind].getBindGroupLayout(0), entries });
      const pass = enc.beginComputePass();
      pass.setPipeline(pipes[kind]);
      pass.setBindGroup(0, bind);
      const groups = Math.ceil(n / WG);
      pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
      pass.end();
      [src, dst] = [dst, src];
      if (l.type === "conv") C = l.cout;
      else if (l.type === "pool") { H >>= 1; W >>= 1; }
    }
    enc.copyBufferToBuffer(src, 0, read, 0, B * nOut * 4);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const logits = new Float32Array(read.getMappedRange(0, B * nOut * 4).slice(0));
    read.unmap();
    params.forEach((b) => b.destroy());
    return logits;
  }

  // Renvoie, pour chaque image, les 3 meilleures classes [[indice, probabilité] × 3].
  async function classify(inputs) {
    const out = [];
    for (let s = 0; s < inputs.length; s += MAX_BATCH) {
      const part = inputs.slice(s, s + MAX_BATCH);
      const logits = await run(part);
      for (let k = 0; k < part.length; k++) {
        const z = logits.subarray(k * nOut, (k + 1) * nOut);
        let m = -Infinity;
        for (const v of z) if (v > m) m = v;
        let sum = 0;
        const best = [[-1, -1], [-1, -1], [-1, -1]];
        for (let i = 0; i < nOut; i++) {
          const e = Math.exp(z[i] - m);
          sum += e;
          if (e > best[2][1]) { best[2] = [i, e]; best.sort((a, b) => b[1] - a[1]); }
        }
        out.push(best.map(([i, e]) => [i, e / sum]));
      }
    }
    return out;
  }

  // Un seul lot à la fois sur la carte (la détection auto et une lecture manuelle peuvent se croiser).
  let queue = Promise.resolve();
  const serial = (inputs) => {
    const job = queue.then(() => classify(inputs));
    queue = job.catch(() => {});
    return job;
  };
  return { classify: serial, name: adapter.info?.vendor || "GPU", destroy: () => device.destroy() };
}
