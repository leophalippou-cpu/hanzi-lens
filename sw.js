// Service worker : garde une copie de l'app sur le téléphone pour qu'elle marche sans connexion
// (dehors, devant une enseigne). Changer VERSION à chaque publication pour forcer la mise à jour.
const VERSION = "hanzi-lens-2026-11-23a";
const FILES = [
  "./", "./index.html", "./manifest.webmanifest", "./icon-192.png", "./icon-512.png",
  "./hanzi-net.js", "./hanzi-segment.js", "./hanzi-reader.js", "./hanzi-gpu.js", "./hanzi-worker.js", "./hanzi-dict.js",
  "./hanzi-mt.js", "./hanzi-mt-worker.js",
  "./models/hanzi.json", "./models/hanzi.bin", "./models/mt.json", "./models/mt.bin", "./dict/zh.json",
];

self.addEventListener("install", (e) => {
  // Fichier par fichier : un fichier absent (ex. traducteur pas encore publié) n'empêche pas le reste.
  e.waitUntil(caches.open(VERSION)
    .then((c) => Promise.all(FILES.map((f) => c.add(f).catch(() => {}))))
    .then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

// Réseau d'abord (pour avoir la dernière version), copie locale si pas de connexion.
self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET" || new URL(req.url).origin !== location.origin) return;
  e.respondWith(
    fetch(req)
      .then((res) => {
        if (res.ok) { const copy = res.clone(); caches.open(VERSION).then((c) => c.put(req, copy)); }
        return res;
      })
      .catch(() => caches.match(req, { ignoreSearch: true }).then((hit) => hit || caches.match("./index.html"))),
  );
});
