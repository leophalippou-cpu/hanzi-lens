// Travailleur de traduction : garde le traducteur en mémoire et traduit sans bloquer la page.
import { loadMT, translateScored, isClean } from "./hanzi-mt.js";

let model = null;

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "init") {
    model = loadMT(msg.manifest, msg.weights);
    self.postMessage({ type: "ready" });
  } else if (msg.type === "translate") {
    let out = { text: "", logp: -99 };
    try { out = translateScored(model, msg.text, { beam: msg.beam || 4 }); } catch (err) { /* pas de traduction */ }
    // Traduction indésirable : on n'affiche rien, le mot à mot du dictionnaire reste visible.
    if (!isClean(out.text)) out = { text: "", logp: -99 };
    self.postMessage({ type: "result", id: msg.id, text: out.text, logp: out.logp });
  }
};
