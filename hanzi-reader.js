// Lecteur maison : découpe l'image, fait reconnaître chaque caractère par notre réseau
// (en parallèle sur plusieurs cœurs), puis choisit le meilleur découpage de chaque ligne.
import { segment, toGray } from "./hanzi-segment.js";
import { loadNet, forward, topK } from "./hanzi-net.js";
import { createGpuNet } from "./hanzi-gpu.js";

const SKIP = 2.3;          // coût pour ignorer une tache (bruit, ponctuation…)
const CHAR_COST = 0.7;     // coût par caractère : à certitude égale, on préfère moins de caractères (好 plutôt que 女 + 子)
const NARROW = 0.45;       // haut et étroit (largeur/hauteur < 0.45, ex. 亻 氵) : sans doute la moitié d'un caractère

export class Reader {
  constructor(base = "/models/") {
    this.base = base;
    this.workers = [];
    this.manifest = null;
    this.jobs = new Map();
    this.nextId = 1;
  }

  async init() {
    const [manifest, weights] = await Promise.all([
      fetch(this.base + "hanzi.json").then((r) => { if (!r.ok) throw new Error("modèle introuvable"); return r.json(); }),
      fetch(this.base + "hanzi.bin").then((r) => { if (!r.ok) throw new Error("poids introuvables"); return r.arrayBuffer(); }),
    ]);
    this.manifest = manifest;
    this.charset = [...manifest.charset];
    this.reject = manifest.reject;

    // 1er choix : la carte graphique (WebGPU), si elle donne exactement les mêmes résultats que le processeur.
    try {
      const net = loadNet(manifest, weights.slice(0));
      const gpu = await createGpuNet(net);
      const test = new Float32Array(48 * 48).map((_, i) => ((i % 48) > 8 && (i % 48) < 40 && (i * 7919) % 5 < 2 ? 1 : 0));
      const [g] = await gpu.classify([test]);
      const c = topK(forward(net, test), 1)[0];
      if (g[0][0] !== c.i || Math.abs(g[0][1] - c.p) > 1e-3) throw new Error("résultats différents du processeur");
      this.gpu = gpu;
      this.backend = "la carte graphique";
      return;
    } catch (err) {
      console.info("Calcul sur le processeur (WebGPU indisponible : " + err.message + ")");
    }

    // Sinon : plusieurs travailleurs en parallèle sur les cœurs du processeur.
    this.backend = "le processeur";
    // Peu de mémoire (téléphone) : moins de travailleurs, chacun garde une copie du réseau (27 Mo).
    const cap = (navigator.deviceMemory || 8) <= 4 ? 2 : 6;
    const n = Math.max(2, Math.min(cap, (navigator.hardwareConcurrency || 4) - 2));
    const url = new URL("./hanzi-worker.js", import.meta.url);
    await Promise.all(Array.from({ length: n }, () => new Promise((resolve, reject) => {
      const w = new Worker(url, { type: "module" });
      w.onerror = (e) => reject(new Error(e.message || "travailleur en échec"));
      w.onmessage = (e) => {
        if (e.data.type === "ready") { this.workers.push(w); w.onmessage = (ev) => this.#done(ev.data); resolve(); }
      };
      w.postMessage({ type: "init", manifest, weights: weights.slice(0) });
    })));
  }

  #done(msg) {
    const job = this.jobs.get(msg.id);
    if (job) { this.jobs.delete(msg.id); job(msg.out); }
  }

  // Reconnaît une liste d'images 48×48 ; renvoie pour chacune [[indice, probabilité] × 3].
  async classify(inputs) {
    if (!inputs.length) return [];
    if (this.gpu) return this.gpu.classify(inputs);
    const k = this.workers.length, per = Math.ceil(inputs.length / k);
    const parts = await Promise.all(this.workers.map((w, j) => {
      const slice = inputs.slice(j * per, (j + 1) * per);
      if (!slice.length) return [];
      const id = this.nextId++;
      return new Promise((resolve) => { this.jobs.set(id, resolve); w.postMessage({ type: "classify", id, inputs: slice }); });
    }));
    return parts.flat();
  }

  // Lit une image (ImageData ou { data, width, height }). Renvoie { lines, ms, polarity }.
  async read(img, { tryInverted = true } = {}) {
    const t0 = performance.now();
    const gray = toGray(img.data, img.width * img.height);
    let best = null;
    for (const invert of tryInverted ? [false, true] : [false]) {
      const res = await this.#readPolarity(gray, img.width, img.height, invert);
      // L'image inversée ne gagne que si elle contient au moins un caractère sûr (sinon : texture du papier).
      if (!best || (res.strong >= 1 && res.score > best.score)) best = res;
      if (best.strong >= 2) break; // texte foncé sur fond clair trouvé : inutile d'essayer l'inverse
    }
    return { lines: best.lines, polarity: best.invert ? "clair sur foncé" : "foncé sur clair", ms: Math.round(performance.now() - t0) };
  }

  async #readPolarity(gray, w, h, invert) {
    const seg = segment(gray, w, h, invert);
    const all = seg.lines.flatMap((l) => l.cands);
    const tops = await this.classify(all.map((c) => c.input));
    all.forEach((c, i) => { c.top = tops[i]; });
    const lines = [];
    let strong = 0, score = 0;
    for (const l of seg.lines) {
      let { chars, skipped } = this.#bestPath(l);
      // Ligne où les formes laissées de côté sont surtout « pas du chinois » (lettres latines, chiffres…) :
      // c'est sans doute du texte latin, on n'y garde que les caractères quasi certains (un « l » ressemble à 工).
      const rejected = l.cands.filter((c) => c.n === 1 && skipped.has(c.start)
        && c.top[0][0] === this.reject && c.top[0][1] >= 0.6).length;
      if (rejected >= 2 && rejected >= chars.length) chars = chars.filter((c) => c.p >= 0.97);
      if (!chars.length) continue;
      strong += chars.filter((c) => c.p >= 0.8).length;
      score += chars.reduce((s, c) => s + (c.p >= 0.6 ? c.p : 0), 0);
      lines.push({ chars });
    }
    return { lines, strong, score, invert };
  }

  // Programmation dynamique : parmi tous les découpages possibles de la ligne, garde celui dont
  // les caractères sont reconnus avec le plus de certitude (et qui ont une forme plausible).
  #bestPath(line) {
    const n = line.count;
    const best = new Array(n + 1).fill(-Infinity), back = new Array(n + 1).fill(null);
    best[0] = 0;
    const byStart = new Map();
    for (const c of line.cands) {
      if (!byStart.has(c.start)) byStart.set(c.start, []);
      byStart.get(c.start).push(c);
    }
    for (let i = 0; i < n; i++) {
      if (best[i] === -Infinity) continue;
      if (best[i] - SKIP > best[i + 1]) { best[i + 1] = best[i] - SKIP; back[i + 1] = { from: i, cand: null }; }
      for (const c of byStart.get(i) || []) {
        const [idx, p] = c.top[0];
        if (idx === this.reject) continue;
        const bw = c.box.x1 - c.box.x0 + 1, bh = c.box.y1 - c.box.y0 + 1;
        const halfLike = bw / bh < NARROW && bh >= 0.7 * line.height;
        const v = Math.log(Math.max(p, 1e-3)) - CHAR_COST - (halfLike ? 0.8 : 0);
        const j = i + c.n;
        if (best[i] + v > best[j]) { best[j] = best[i] + v; back[j] = { from: i, cand: c }; }
      }
    }
    const chars = [], skipped = new Set();
    for (let j = n; j > 0 && back[j]; j = back[j].from) {
      const c = back[j].cand;
      if (!c) { skipped.add(back[j].from); continue; }
      const alts = c.top.filter(([i]) => i !== this.reject).map(([i, p]) => ({ ch: this.charset[i], p }));
      chars.unshift({ box: c.box, ch: alts[0].ch, p: alts[0].p, alts });
    }
    return { chars, skipped };
  }
}
