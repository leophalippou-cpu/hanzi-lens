// Travailleur de calcul : reçoit des images 48×48 et renvoie les 3 caractères les plus probables.
// Plusieurs travailleurs tournent en parallèle (un par cœur du processeur) pour aller plus vite.
import { loadNet, forward, topK } from "./hanzi-net.js";

let net = null;

self.onmessage = (e) => {
  const m = e.data;
  if (m.type === "init") {
    net = loadNet(m.manifest, m.weights);
    self.postMessage({ type: "ready" });
  } else if (m.type === "classify") {
    const out = m.inputs.map((x) => topK(forward(net, x), 3).map((t) => [t.i, t.p]));
    self.postMessage({ type: "result", id: m.id, out });
  }
};
