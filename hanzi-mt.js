// Moteur maison du traducteur chinois -> français (Transformer, voir ml/mt_model.py).
// Exécute les mêmes calculs que PyTorch, en JavaScript pur, dans un Web Worker (et dans Node pour les tests).

export function loadMT(manifest, buffer) {
  const T = {};
  for (const [name, t] of Object.entries(manifest.tensors)) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    if ("q" in t) {
      const [rows, cols] = t.shape;
      const q = new Int8Array(buffer, t.q, n), s = new Float32Array(buffer, t.scale, rows), w = new Float32Array(n);
      for (let i = 0; i < rows; i++) { const si = s[i], o = i * cols; for (let j = 0; j < cols; j++) w[o + j] = q[o + j] * si; }
      T[name] = w;
    } else T[name] = new Float32Array(buffer, t.f, n).slice();
  }
  const src = new Map(manifest.src_vocab.map((c, i) => [c, i]));
  return { cfg: manifest, T, src, pieces: manifest.tgt_pieces };
}

const PAD = 0, BOS = 1, EOS = 2, UNK = 3;

// y = W x + b, W rangée par lignes [sorties × entrées].
export function linear(m, name, x) {
  const W = m.T[name + ".weight"], b = m.T[name + ".bias"], n = x.length, N = b.length;
  const out = new Float32Array(N);
  let o = 0;
  for (; o + 4 <= N; o += 4) {
    const a0 = o * n, a1 = a0 + n, a2 = a1 + n, a3 = a2 + n;
    let s0 = b[o], s1 = b[o + 1], s2 = b[o + 2], s3 = b[o + 3];
    for (let i = 0; i < n; i++) { const v = x[i]; s0 += W[a0 + i] * v; s1 += W[a1 + i] * v; s2 += W[a2 + i] * v; s3 += W[a3 + i] * v; }
    out[o] = s0; out[o + 1] = s1; out[o + 2] = s2; out[o + 3] = s3;
  }
  for (; o < N; o++) { let s = b[o]; for (let i = 0; i < n; i++) s += W[o * n + i] * x[i]; out[o] = s; }
  return out;
}

function layerNorm(m, name, x) {
  const g = m.T[name + ".weight"], b = m.T[name + ".bias"], n = x.length;
  let mean = 0; for (let i = 0; i < n; i++) mean += x[i]; mean /= n;
  let v = 0; for (let i = 0; i < n; i++) v += (x[i] - mean) ** 2; v /= n;
  const inv = 1 / Math.sqrt(v + 1e-5), out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * g[i] + b[i];
  return out;
}

function addInto(x, y) { for (let i = 0; i < x.length; i++) x[i] += y[i]; return x; }

function feedForward(m, pre, h) {
  const a = linear(m, pre + "ff1", h);
  for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0;
  return linear(m, pre + "ff2", a);
}

// Attention à plusieurs têtes : q (d), K et V (listes de vecteurs d).
function attend(q, K, V, heads) {
  const d = q.length, dk = d / heads, out = new Float32Array(d), sc = 1 / Math.sqrt(dk);
  const w = new Float32Array(K.length);
  for (let h = 0; h < heads; h++) {
    const o = h * dk;
    let mx = -Infinity;
    for (let j = 0; j < K.length; j++) {
      let s = 0; const k = K[j];
      for (let i = 0; i < dk; i++) s += q[o + i] * k[o + i];
      w[j] = s * sc; if (w[j] > mx) mx = w[j];
    }
    let sum = 0;
    for (let j = 0; j < K.length; j++) { w[j] = Math.exp(w[j] - mx); sum += w[j]; }
    for (let j = 0; j < K.length; j++) {
      const p = w[j] / sum, v = V[j];
      for (let i = 0; i < dk; i++) out[o + i] += p * v[o + i];
    }
  }
  return out;
}

function embed(m, table, id, pos) {
  const d = m.cfg.d, E = m.T[table], v = new Float32Array(d), s = Math.sqrt(d);
  for (let i = 0; i < d; i += 2) {
    const ang = pos / Math.pow(10000, i / d);
    v[i] = E[id * d + i] * s + Math.sin(ang);
    v[i + 1] = E[id * d + i + 1] * s + Math.cos(ang);
  }
  return v;
}

export function encode(m, ids) {
  const { heads, enc } = m.cfg;
  let X = ids.map((id, p) => embed(m, "src_emb.weight", id, p));
  for (let l = 0; l < enc; l++) {
    const pre = `enc.${l}.`;
    const H = X.map((x) => layerNorm(m, pre + "n1", x));
    const Q = H.map((h) => linear(m, pre + "att.q", h));
    const K = H.map((h) => linear(m, pre + "att.k", h));
    const V = H.map((h) => linear(m, pre + "att.v", h));
    X = X.map((x, i) => addInto(x, linear(m, pre + "att.o", attend(Q[i], K, V, heads))));
    X = X.map((x) => addInto(x, feedForward(m, pre, layerNorm(m, pre + "n2", x))));
  }
  return X.map((x) => layerNorm(m, "enc_norm", x));
}

// Un pas du décodeur pour une hypothèse : renvoie les log-probabilités du morceau suivant.
export function decodeStep(m, cache, cross, tok, pos) {
  const { heads, dec, n_tgt, d } = m.cfg;
  let y = embed(m, "tgt_emb.weight", tok, pos);
  for (let l = 0; l < dec; l++) {
    const pre = `dec.${l}.`;
    let h = layerNorm(m, pre + "n1", y);
    cache[l].K.push(linear(m, pre + "self_att.k", h));
    cache[l].V.push(linear(m, pre + "self_att.v", h));
    y = addInto(y, linear(m, pre + "self_att.o", attend(linear(m, pre + "self_att.q", h), cache[l].K, cache[l].V, heads)));
    h = layerNorm(m, pre + "n2", y);
    y = addInto(y, linear(m, pre + "cross.o", attend(linear(m, pre + "cross.q", h), cross[l].K, cross[l].V, heads)));
    y = addInto(y, feedForward(m, pre, layerNorm(m, pre + "n3", y)));
  }
  y = layerNorm(m, "dec_norm", y);
  const E = m.T["tgt_emb.weight"], logits = new Float32Array(n_tgt);
  let mx = -Infinity;
  for (let t = 0; t < n_tgt; t++) {
    let s = 0; const o = t * d;
    for (let i = 0; i < d; i++) s += E[o + i] * y[i];
    logits[t] = s; if (s > mx) mx = s;
  }
  let sum = 0;
  for (let t = 0; t < n_tgt; t++) sum += Math.exp(logits[t] - mx);
  const lse = mx + Math.log(sum);
  for (let t = 0; t < n_tgt; t++) logits[t] -= lse;
  return logits;
}

// Interdit de répéter un triplet de morceaux déjà écrit (les petits modèles bégaient parfois).
function repeats(tokens, next) {
  const n = tokens.length;
  if (n < 2) return false;
  const a = tokens[n - 2], b = tokens[n - 1];
  for (let i = 0; i + 2 < n; i++) if (tokens[i] === a && tokens[i + 1] === b && tokens[i + 2] === next) return true;
  return false;
}

export function detok(m, tokens) {
  return tokens.map((t) => (t > UNK ? m.pieces[t] : "")).join("").replace(/▁/g, " ").trim();
}

// Recherche en faisceau : on garde les `beam` meilleures traductions partielles à chaque pas.
export function translate(m, text, opts) { return translateScored(m, text, opts).text; }

// Renvoie { text, logp } : logp = log-probabilité moyenne par morceau (plus c'est bas, moins c'est sûr).
export function translateScored(m, text, { beam = 4, alpha = 0.7, noRepeat = true } = {}) {
  const ids = [...text].filter((c) => c.trim()).map((c) => m.src.get(c) ?? UNK).slice(0, 64);
  if (!ids.length) return { text: "", logp: 0 };
  const mem = encode(m, ids);
  const cross = [];
  for (let l = 0; l < m.cfg.dec; l++) {
    cross.push({ K: mem.map((x) => linear(m, `dec.${l}.cross.k`, x)), V: mem.map((x) => linear(m, `dec.${l}.cross.v`, x)) });
  }
  const maxLen = Math.min(96, 2 * ids.length + 12);
  let beams = [{ tokens: [], logp: 0, cache: cross.map(() => ({ K: [], V: [] })) }];
  const finished = [];
  const score = (h) => h.logp / Math.pow((5 + h.tokens.length) / 6, alpha);
  for (let pos = 0; pos < maxLen && beams.length; pos++) {
    const cands = [];
    for (const b of beams) {
      const lp = decodeStep(m, b.cache, cross, pos === 0 ? BOS : b.tokens[b.tokens.length - 1], pos);
      // Les `beam` meilleurs morceaux suivants pour cette hypothèse.
      const top = [];
      for (let t = 1; t < lp.length; t++) {
        if (top.length < beam + 2 || lp[t] > top[top.length - 1][1]) {
          top.push([t, lp[t]]); top.sort((x, y) => y[1] - x[1]); if (top.length > beam + 2) top.pop();
        }
      }
      for (const [t, l] of top) if (t === EOS || !noRepeat || !repeats(b.tokens, t)) cands.push({ parent: b, t, logp: b.logp + l });
    }
    cands.sort((x, y) => y.logp - x.logp);
    const next = [];
    for (const c of cands) {
      if (next.length >= beam) break;
      const tokens = [...c.parent.tokens, c.t];
      if (c.t === EOS) { finished.push({ tokens: tokens.slice(0, -1), logp: c.logp }); continue; }
      next.push({ tokens, logp: c.logp, cache: c.parent.cache.map((k) => ({ K: k.K.slice(), V: k.V.slice() })) });
    }
    beams = next;
    // Arrêt : la meilleure traduction finie ne peut plus être battue.
    if (finished.length >= beam && beams.length && Math.max(...finished.map(score)) > score(beams[0])) break;
  }
  const pool = finished.length ? finished : beams;
  pool.sort((x, y) => score(y) - score(x));
  const best = pool[0];
  // Tiret de dialogue hérité des sous-titres : inutile ici.
  const out = detok(m, best.tokens).replace(/^[-–—]\s*/, "");
  return { text: out, logp: best.logp / Math.max(1, best.tokens.length + 1) };
}

// Mots qui ne doivent jamais apparaître dans une traduction (restes de pages web indésirables).
const BAD = /(sexe?s?|sexy|sexuel\w*|porn\w*|xxx|escort\w*|[ée]rotique\w*|salope\w*|nichon\w*|chatte\w*|bites?|p[ée]nis|vagin\w*|masturb\w*|orgasm\w*|viagra|cialis|casino\w*|putes?|prostitu\w*)/i;
export function isClean(text) { return !BAD.test(text); }
