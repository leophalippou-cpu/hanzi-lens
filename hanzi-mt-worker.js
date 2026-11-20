// Travailleur de traduction : garde le traducteur en mémoire et traduit sans bloquer la page.
import { loadMT, translate } from "./hanzi-mt.js";

let model = null;

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "init") {
    model = loadMT(msg.manifest, msg.weights);
    self.postMessage({ type: "ready" });
  } else if (msg.type === "translate") {
    let text = "";
    try { text = translate(model, msg.text, { beam: msg.beam || 4 }); } catch (err) { text = ""; }
    self.postMessage({ type: "result", id: msg.id, text });
  }
};
