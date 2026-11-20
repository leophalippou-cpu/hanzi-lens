// Moteur d'inférence maison : exécute notre réseau de neurones (entraîné avec ml/train.py,
// exporté par ml/export.py) en JavaScript pur, sans aucune bibliothèque.
// Fonctionne dans un Web Worker du navigateur et dans Node (pour les tests).

export const SIZE = 48;
const INNER = 40;

// ---------- Chargement des poids ----------

// manifest : objet JSON (models/hanzi.json). buffer : ArrayBuffer de models/hanzi.bin.
// Les poids sont stockés en int8 avec une échelle par neurone de sortie ; on les remet en float32.
export function loadNet(manifest, buffer) {
  const layers = manifest.layers.map((l) => {
    if (l.type === "pool") return l;
    const n = l.type === "conv" ? l.cout * l.cin * 9 : l.out * l.in;
    const per = n / (l.type === "conv" ? l.cout : l.out);
    const q = new Int8Array(buffer, l.w, n);
    const scale = new Float32Array(buffer, l.scale, l.type === "conv" ? l.cout : l.out);
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) w[i] = q[i] * scale[(i / per) | 0];
    const b = new Float32Array(buffer, l.b, l.type === "conv" ? l.cout : l.out).slice();
    return { ...l, wf: w, bf: b };
  });
  return { ...manifest, layers };
}

// ---------- Couches ----------

// Convolution 3×3 (remplissage 1) + ReLU, calculée comme un produit de matrices :
// on range d'abord chaque voisinage 3×3 de chaque pixel dans une ligne (« im2col »), puis on
// multiplie par les poids en blocs de 4 neurones × 2 pixels pour réutiliser chaque valeur lue.
function conv3x3(inp, C, H, W, l) {
  const P = H * W, K = C * 9, Cout = l.cout, wt = l.wf, b = l.bf;
  const col = new Float32Array(P * K);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let o = (y * W + x) * K;
      for (let c = 0; c < C; c++) {
        const base = c * P;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy;
          for (let dx = -1; dx <= 1; dx++, o++) {
            const xx = x + dx;
            col[o] = yy < 0 || yy >= H || xx < 0 || xx >= W ? 0 : inp[base + yy * W + xx];
          }
        }
      }
    }
  const out = new Float32Array(Cout * P);
  for (let c = 0; c < Cout; c += 4) {
    const a0 = c * K, a1 = a0 + K, a2 = a1 + K, a3 = a2 + K;
    for (let p = 0; p < P; p += 2) {
      const b0 = p * K, b1 = b0 + K;
      let s00 = b[c], s01 = s00, s10 = b[c + 1], s11 = s10, s20 = b[c + 2], s21 = s20, s30 = b[c + 3], s31 = s30;
      for (let k = 0; k < K; k++) {
        const x0 = col[b0 + k], x1 = col[b1 + k];
        let w = wt[a0 + k]; s00 += w * x0; s01 += w * x1;
        w = wt[a1 + k]; s10 += w * x0; s11 += w * x1;
        w = wt[a2 + k]; s20 += w * x0; s21 += w * x1;
        w = wt[a3 + k]; s30 += w * x0; s31 += w * x1;
      }
      const o = c * P + p;
      out[o] = s00 > 0 ? s00 : 0; out[o + 1] = s01 > 0 ? s01 : 0;
      out[o + P] = s10 > 0 ? s10 : 0; out[o + P + 1] = s11 > 0 ? s11 : 0;
      out[o + 2 * P] = s20 > 0 ? s20 : 0; out[o + 2 * P + 1] = s21 > 0 ? s21 : 0;
      out[o + 3 * P] = s30 > 0 ? s30 : 0; out[o + 3 * P + 1] = s31 > 0 ? s31 : 0;
    }
  }
  return out;
}

function maxpool2(inp, C, H, W) {
  const h = H >> 1, w = W >> 1, out = new Float32Array(C * h * w);
  for (let c = 0; c < C; c++)
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = c * H * W + 2 * y * W + 2 * x;
        out[c * h * w + y * w + x] = Math.max(inp[i], inp[i + 1], inp[i + W], inp[i + W + 1]);
      }
  return out;
}

function dense(inp, l) {
  const out = new Float32Array(l.out), wt = l.wf, n = l.in, N = l.out;
  let o = 0;
  for (; o + 4 <= N; o += 4) {
    const a0 = o * n, a1 = a0 + n, a2 = a1 + n, a3 = a2 + n;
    let s0 = l.bf[o], s1 = l.bf[o + 1], s2 = l.bf[o + 2], s3 = l.bf[o + 3];
    for (let i = 0; i < n; i++) {
      const x = inp[i];
      s0 += wt[a0 + i] * x; s1 += wt[a1 + i] * x; s2 += wt[a2 + i] * x; s3 += wt[a3 + i] * x;
    }
    out[o] = s0; out[o + 1] = s1; out[o + 2] = s2; out[o + 3] = s3;
  }
  for (; o < N; o++) {
    let s = l.bf[o];
    for (let i = 0; i < n; i++) s += wt[o * n + i] * inp[i];
    out[o] = s;
  }
  if (l.relu) for (let i = 0; i < N; i++) if (out[i] < 0) out[i] = 0;
  return out;
}

// img : Float32Array 48×48 (encre claire, 0..1). Renvoie les probabilités (Float32Array).
export function forward(net, img) {
  let x = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < x.length; i++) x[i] = (img[i] - net.mean) / net.std;
  let C = 1, H = SIZE, W = SIZE;
  for (const l of net.layers) {
    if (l.type === "conv") { x = conv3x3(x, C, H, W, l); C = l.cout; }
    else if (l.type === "pool") { x = maxpool2(x, C, H, W); H >>= 1; W >>= 1; }
    else x = dense(x, l);
  }
  let m = -Infinity;
  for (const v of x) if (v > m) m = v;
  let s = 0;
  for (let i = 0; i < x.length; i++) s += (x[i] = Math.exp(x[i] - m));
  for (let i = 0; i < x.length; i++) x[i] /= s;
  return x;
}

export function topK(probs, k = 3) {
  const best = [];
  for (let i = 0; i < probs.length; i++) {
    if (best.length < k || probs[i] > best[best.length - 1].p) {
      best.push({ i, p: probs[i] });
      best.sort((a, b) => b.p - a.p);
      if (best.length > k) best.pop();
    }
  }
  return best;
}

// ---------- Normalisation (identique à ml/prepare.py : normalize) ----------

// Redimensionne un tableau float (w×h) vers (nw×nh) : moyenne de surface si on réduit,
// interpolation bilinéaire si on agrandit.
function resize(src, w, h, nw, nh) {
  const out = new Float32Array(nw * nh);
  const fx = w / nw, fy = h / nh;
  if (fx >= 1 && fy >= 1) {
    for (let y = 0; y < nh; y++) {
      const y0 = y * fy, y1 = y0 + fy;
      for (let x = 0; x < nw; x++) {
        const x0 = x * fx, x1 = x0 + fx;
        let s = 0, a = 0;
        for (let sy = Math.floor(y0); sy < Math.ceil(y1); sy++) {
          const wy = Math.min(y1, sy + 1) - Math.max(y0, sy);
          for (let sx = Math.floor(x0); sx < Math.ceil(x1); sx++) {
            const wx = Math.min(x1, sx + 1) - Math.max(x0, sx);
            s += src[sy * w + sx] * wx * wy;
            a += wx * wy;
          }
        }
        out[y * nw + x] = s / a;
      }
    }
    return out;
  }
  for (let y = 0; y < nh; y++) {
    const sy = Math.min(h - 1, Math.max(0, (y + 0.5) * fy - 0.5));
    const y0 = Math.floor(sy), y1 = Math.min(h - 1, y0 + 1), dy = sy - y0;
    for (let x = 0; x < nw; x++) {
      const sx = Math.min(w - 1, Math.max(0, (x + 0.5) * fx - 0.5));
      const x0 = Math.floor(sx), x1 = Math.min(w - 1, x0 + 1), dx = sx - x0;
      out[y * nw + x] = (src[y0 * w + x0] * (1 - dx) + src[y0 * w + x1] * dx) * (1 - dy)
        + (src[y1 * w + x0] * (1 - dx) + src[y1 * w + x1] * dx) * dy;
    }
  }
  return out;
}

// ink : Float32Array w×h, encre claire sur fond ~0. Renvoie l'entrée 48×48 du réseau (0..1) ou null.
export function normalize(ink, w, h) {
  let m = 0;
  for (const v of ink) if (v > m) m = v;
  if (m <= 0) return null;
  const t = m * 0.2;
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (ink[y * w + x] > t) {
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
  const cw = x1 - x0 + 1, ch = y1 - y0 + 1;
  const crop = new Float32Array(cw * ch);
  for (let y = 0; y < ch; y++)
    for (let x = 0; x < cw; x++) crop[y * cw + x] = Math.min(1, ink[(y + y0) * w + x + x0] / m);
  const s = INNER / Math.max(cw, ch);
  const nw = Math.max(1, Math.round(cw * s)), nh = Math.max(1, Math.round(ch * s));
  const small = resize(crop, cw, ch, nw, nh);
  const out = new Float32Array(SIZE * SIZE);
  const oy = (SIZE - nh) >> 1, ox = (SIZE - nw) >> 1;
  for (let y = 0; y < nh; y++) out.set(small.subarray(y * nw, y * nw + nw), (y + oy) * SIZE + ox);
  let peak = 0;
  for (const v of out) if (v > peak) peak = v;
  const vals = [];
  for (const v of out) if (v > peak * 0.2) vals.push(v);
  vals.sort((a, b) => a - b);
  const p = vals.length ? vals[Math.min(vals.length - 1, Math.floor(0.9 * (vals.length - 1)))] : peak;
  if (p <= 0) return null;
  for (let i = 0; i < out.length; i++) out[i] = Math.min(1, out[i] / p);
  return out;
}
