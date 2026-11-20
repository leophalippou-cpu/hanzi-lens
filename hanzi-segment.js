// Découpage maison : trouve les caractères dans une image (photo, webcam) avant de les reconnaître.
//
// 1. Binarisation locale (méthode de Sauvola) : un pixel est de l'encre s'il est nettement plus
//    sombre que son voisinage. On essaie aussi l'image inversée (texte clair sur fond foncé).
// 2. Composantes connexes : chaque tache d'encre d'un seul tenant (un trait, un morceau de caractère).
// 3. Regroupement : les morceaux proches dont l'ensemble garde la taille d'un caractère sont réunis.
// 4. Lignes : les caractères alignés horizontalement forment une ligne, lue de gauche à droite.
// Le réseau tranche ensuite les cas ambigus (ex. 好 = 女 + 子 ?) dans hanzi-reader.js.

import { normalize } from "./hanzi-net.js";

export function toGray(rgba, n) {
  const g = new Float32Array(n);
  for (let i = 0, j = 0; j < n; i += 4, j++) g[j] = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
  return g;
}

// Renvoie { ink: Float32Array (contraste de l'encre, 0 ailleurs), mask: Uint8Array }.
function binarize(gray, w, h, invert) {
  const n = w * h;
  const W1 = w + 1;
  const sum = new Float64Array(W1 * (h + 1)), sq = new Float64Array(W1 * (h + 1));
  for (let y = 0; y < h; y++) {
    let rs = 0, rq = 0;
    for (let x = 0; x < w; x++) {
      const v = invert ? 255 - gray[y * w + x] : gray[y * w + x];
      rs += v; rq += v * v;
      sum[(y + 1) * W1 + x + 1] = sum[y * W1 + x + 1] + rs;
      sq[(y + 1) * W1 + x + 1] = sq[y * W1 + x + 1] + rq;
    }
  }
  const r = Math.max(7, Math.round(Math.min(w, h) / 24));
  const ink = new Float32Array(n), mask = new Uint8Array(n);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
      const a = (y1 - y0) * (x1 - x0);
      const s = sum[y1 * W1 + x1] - sum[y0 * W1 + x1] - sum[y1 * W1 + x0] + sum[y0 * W1 + x0];
      const q = sq[y1 * W1 + x1] - sq[y0 * W1 + x1] - sq[y1 * W1 + x0] + sq[y0 * W1 + x0];
      const m = s / a, sd = Math.sqrt(Math.max(0, q / a - m * m));
      const v = invert ? 255 - gray[y * w + x] : gray[y * w + x];
      const c = m - v;
      if (c > 0) ink[y * w + x] = c;
      if (v < m * (1 + 0.25 * (sd / 128 - 1)) && c > 18) mask[y * w + x] = 1;
    }
  }
  return { ink, mask };
}

// Composantes connexes (8-voisinage). Renvoie les étiquettes et la liste des composantes.
function components(mask, w, h) {
  const label = new Int32Array(w * h);
  const parent = [0];
  const find = (a) => { while (parent[a] !== a) a = parent[a] = parent[parent[a]]; return a; };
  let next = 1;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      const nb = [];
      if (x > 0 && label[i - 1]) nb.push(label[i - 1]);
      if (y > 0) {
        if (label[i - w]) nb.push(label[i - w]);
        if (x > 0 && label[i - w - 1]) nb.push(label[i - w - 1]);
        if (x < w - 1 && label[i - w + 1]) nb.push(label[i - w + 1]);
      }
      if (!nb.length) { parent.push(next); label[i] = next++; continue; }
      let m = find(nb[0]);
      for (let k = 1; k < nb.length; k++) {
        const r = find(nb[k]);
        if (r !== m) { if (r < m) { parent[m] = r; m = r; } else parent[r] = m; }
      }
      label[i] = m;
    }
  const boxes = new Map();
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!label[i]) continue;
      const r = find(label[i]);
      label[i] = r;
      let b = boxes.get(r);
      if (!b) boxes.set(r, (b = { id: r, x0: x, y0: y, x1: x, y1: y, area: 0 }));
      if (x < b.x0) b.x0 = x; if (x > b.x1) b.x1 = x;
      if (y < b.y0) b.y0 = y; if (y > b.y1) b.y1 = y;
      b.area++;
    }
  return { label, comps: [...boxes.values()] };
}

const side = (b) => Math.max(b.x1 - b.x0 + 1, b.y1 - b.y0 + 1);
const union = (a, b) => ({ x0: Math.min(a.x0, b.x0), y0: Math.min(a.y0, b.y0), x1: Math.max(a.x1, b.x1), y1: Math.max(a.y1, b.y1) });
const gapX = (a, b) => Math.max(0, Math.max(a.x0, b.x0) - Math.min(a.x1, b.x1) - 1);
const gapY = (a, b) => Math.max(0, Math.max(a.y0, b.y0) - Math.min(a.y1, b.y1) - 1);

function percentile(arr, p) {
  const s = [...arr].sort((a, b) => a - b);
  return s.length ? s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))] : 0;
}

// Regroupe les composantes en « grappes » de la taille d'un caractère, puis les grappes en lignes.
function group(comps, w, h) {
  const minSide = Math.max(4, Math.round(Math.min(w, h) / 160));
  let keep = comps.filter((c) => {
    const bw = c.x1 - c.x0 + 1, bh = c.y1 - c.y0 + 1;
    if (Math.max(bw, bh) < minSide && c.area < minSide * 2) return false;      // poussière
    if (c.x0 === 0 || c.y0 === 0 || c.x1 === w - 1 || c.y1 === h - 1) return false; // coupé par le bord
    if (bw > w * 0.6 || bh > h * 0.6) return false;                               // fond, bord de feuille
    if (c.area / (bw * bh) > 0.75 && Math.max(bw, bh) > 14) return false;       // tache pleine
    return true;
  });
  if (keep.length > 1500) keep = keep.sort((a, b) => b.area - a.area).slice(0, 1500);
  if (!keep.length) return { S: 0, lines: [] };

  // Taille typique d'un caractère : les plus gros morceaux sont souvent des caractères entiers.
  const big = keep.map(side).filter((s) => s >= minSide * 2);
  const S0 = Math.max(minSide * 3, percentile(big.length ? big : keep.map(side), 0.85));

  // On ne réunit ici que les morceaux EMPILÉS (même colonne : 三, 字, 是…) ou imbriqués (回, 心…).
  // Les morceaux côte à côte (女 + 子 = 好 ?) sont départagés plus tard par le réseau.
  const linked = (a, b, S) => {
    const u = union(a, b);
    if (side(u) > 1.2 * S) return false;
    const wa = a.x1 - a.x0 + 1, wb = b.x1 - b.x0 + 1;
    const xo = Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0) + 1;
    if (xo >= 0.5 * Math.min(wa, wb) && gapY(a, b) <= 0.6 * S) return true;
    const yo = Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0) + 1;
    if (xo > 0 && yo > 0) {
      const small = Math.min(wa * (a.y1 - a.y0 + 1), wb * (b.y1 - b.y0 + 1));
      if (xo * yo >= 0.3 * small) return true;
    }
    return false;
  };
  // Fusion gloutonne (paires dont l'ensemble est le plus petit d'abord), pour une taille S donnée.
  const mergeStacked = (S) => {
    const parent = keep.map((_, i) => i);
    const find = (a) => { while (parent[a] !== a) a = parent[a] = parent[parent[a]]; return a; };
    const box = keep.map((c) => ({ x0: c.x0, y0: c.y0, x1: c.x1, y1: c.y1 }));
    const cell = Math.max(8, Math.ceil(1.2 * S));
    const grid = new Map();
    keep.forEach((c, i) => {
      for (let gy = Math.floor(c.y0 / cell); gy <= Math.floor(c.y1 / cell); gy++)
        for (let gx = Math.floor(c.x0 / cell); gx <= Math.floor(c.x1 / cell); gx++) {
          const k = gy * 100000 + gx;
          if (!grid.has(k)) grid.set(k, []);
          grid.get(k).push(i);
        }
    });
    const pairs = [], seen = new Set();
    keep.forEach((c, i) => {
      for (let gy = Math.floor(c.y0 / cell) - 1; gy <= Math.floor(c.y1 / cell) + 1; gy++)
        for (let gx = Math.floor(c.x0 / cell) - 1; gx <= Math.floor(c.x1 / cell) + 1; gx++)
          for (const j of grid.get(gy * 100000 + gx) || []) {
            if (j <= i || seen.has(i * 1e6 + j)) continue;
            seen.add(i * 1e6 + j);
            if (linked(keep[i], keep[j], S)) pairs.push([side(union(keep[i], keep[j])), i, j]);
          }
    });
    pairs.sort((a, b) => a[0] - b[0]);
    for (const [, i, j] of pairs) {
      const ri = find(i), rj = find(j);
      if (ri === rj || !linked(box[ri], box[rj], S)) continue;
      parent[rj] = ri;
      box[ri] = union(box[ri], box[rj]);
    }
    const clusters = new Map();
    keep.forEach((c, i) => {
      const r = find(i);
      if (!clusters.has(r)) clusters.set(r, { ...box[r], ids: [] });
      clusters.get(r).ids.push(c.id);
    });
    return [...clusters.values()];
  };

  // Une écriture en traits séparés fait sous-estimer la taille (S0 ≈ longueur d'un trait).
  // On la corrige avec la hauteur des bandes de texte : rangées de pixels qui contiennent de l'encre.
  const prof = new Int32Array(h + 1);
  for (const c of keep) { prof[c.y0]++; prof[c.y1 + 1]--; }
  const bands = [];
  for (let y = 0, run = 0, start = -1, n = 0; y <= h; y++) {
    run += y < h ? prof[y] : 0;
    if (y < h && run > 0) {
      if (start < 0) start = y;
      n = Math.max(n, run);
    } else if (start >= 0) {
      const last = bands[bands.length - 1];
      if (last && start - last.y1 < 0.3 * S0) { last.y1 = y - 1; last.n += n; }
      else bands.push({ y0: start, y1: y - 1, n });
      start = -1; n = 0;
    }
  }
  const heights = bands.flatMap((b) => Array(Math.min(b.n, 50)).fill(b.y1 - b.y0 + 1));
  const S = Math.min(1.8 * S0, Math.max(S0, percentile(heights, 0.5)));
  const clusters = mergeStacked(S);
  // Les petits morceaux (points, traits courts) restent : ils peuvent appartenir au caractère voisin.
  const cl = clusters.filter((c) => side(c) >= 0.08 * S && side(c) <= 1.8 * S);

  // Lignes : on parcourt les grappes de gauche à droite et on les accroche à la ligne alignée.
  cl.sort((a, b) => a.x0 - b.x0);
  const lines = [];
  for (const c of cl) {
    const cy = (c.y0 + c.y1) / 2;
    let best = null, bestD = Infinity;
    for (const l of lines) {
      const last = l.items[l.items.length - 1];
      const ly = (l.y0 + l.y1) / 2, lh = l.y1 - l.y0 + 1;
      const d = Math.abs(cy - ly);
      if (d < 0.5 * Math.max(lh, S * 0.6) && c.x0 - last.x1 < 1.2 * S && d < bestD) { best = l; bestD = d; }
    }
    if (best) {
      best.items.push(c);
      best.y0 = Math.min(best.y0, c.y0); best.y1 = Math.max(best.y1, c.y1);
    } else lines.push({ items: [c], y0: c.y0, y1: c.y1 });
  }
  lines.sort((a, b) => a.y0 - b.y0);
  return { S, lines };
}

// Image du réseau (48×48) pour un ensemble de grappes : seule l'encre de leurs composantes est gardée.
function crop(ink, label, w, items) {
  let b = items[0];
  for (const it of items) b = union(b, it);
  const ids = new Set(items.flatMap((it) => it.ids));
  const x0 = Math.max(0, b.x0 - 1), y0 = Math.max(0, b.y0 - 1);
  const x1 = Math.min(w - 1, b.x1 + 1), y1 = Math.min(label.length / w - 1, b.y1 + 1);
  const cw = x1 - x0 + 1, ch = y1 - y0 + 1;
  const patch = new Float32Array(cw * ch);
  for (let y = y0; y <= y1; y++)
    for (let x = x0; x <= x1; x++) {
      // On garde aussi le pixel voisin d'un trait (bord adouci du trait).
      let mine = false;
      for (let dy = -1; dy <= 1 && !mine; dy++)
        for (let dx = -1; dx <= 1 && !mine; dx++) {
          const yy = y + dy, xx = x + dx;
          if (yy >= 0 && xx >= 0 && xx < w && yy * w + xx < label.length && ids.has(label[yy * w + xx])) mine = true;
        }
      if (mine) patch[(y - y0) * cw + (x - x0)] = ink[y * w + x];
    }
  return { box: b, input: normalize(patch, cw, ch) };
}

// Point d'entrée : gray (Float32Array w×h). Renvoie, pour chaque polarité, les lignes et
// les « candidats » à faire reconnaître (un candidat = 1 à 3 grappes consécutives d'une ligne).
export function segment(gray, w, h, invert = false) {
  const { ink, mask } = binarize(gray, w, h, invert);
  const { label, comps } = components(mask, w, h);
  const { S, lines } = group(comps, w, h);
  const out = [];
  for (const l of lines.slice(0, 12)) {
    const items = l.items.slice(0, 40);
    // Hauteur de ligne : son étendue verticale, bornée par la taille typique d'un caractère.
    const lh = Math.min(Math.max(l.y1 - l.y0 + 1, S * 0.6), S * 1.3);
    const cands = [];
    for (let i = 0; i < items.length; i++)
      for (let n = 1; n <= 4 && i + n <= items.length; n++) {
        const part = items.slice(i, i + n);
        let u = part[0];
        for (const p of part) u = union(u, p);
        const uw = u.x1 - u.x0 + 1, uh = u.y1 - u.y0 + 1;
        if (n > 1 && uw > 1.3 * Math.max(uh, lh * 0.8)) break;
        // Un morceau minuscule seul n'est pas un caractère : il ne sert qu'avec ses voisins.
        if (n === 1 && Math.max(uw, uh) < 0.3 * lh) continue;
        const c = crop(ink, label, w, part);
        if (c.input) cands.push({ start: i, n, box: c.box, input: c.input, aspect: uw / lh, size: Math.max(uw, uh) / lh });
      }
    out.push({ count: items.length, height: lh, cands });
  }
  return { S, lines: out };
}
