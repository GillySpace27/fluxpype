// Plumber service worker: no caching. Its only job is a friendly page when the Plumber server is not
// running, which is what a Dock icon or bookmark would otherwise show as a bare connection error.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

const OFFLINE = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Plumber is not running</title>
<style>
:root{color-scheme:dark light;--bg:#0b0d12;--card:#141821;--ink:#e8ebf2;--muted:#9aa3b5;--line:#262c3a;--accent:#ff9f43}
@media (prefers-color-scheme: light){:root{--bg:#f6f7fb;--card:#fff;--ink:#141821;--muted:#5b6475;--line:#dde1ea}}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--ink);
font:14px/1.55 -apple-system,BlinkMacSystemFont,"Inter",system-ui,sans-serif;padding:16px}
.card{max-width:460px;background:var(--card);border:1px solid var(--line);border-radius:18px;padding:28px}
h1{font-size:18px;margin:0 0 6px}p{color:var(--muted);margin:0 0 12px}
code{white-space:nowrap;font:12px ui-monospace,Menlo,monospace;background:var(--bg);padding:2px 6px;border-radius:6px}
button{margin-top:6px;padding:9px 16px;border:0;border-radius:10px;background:var(--accent);color:#1b1206;font-weight:600;cursor:pointer}
</style></head><body><div class="card"><h1>Plumber is not running</h1>
<p>Nothing answered at this address. Start Plumber with <code>flux-plumb</code>, or have it start at login with
<code>flux-plumb --install</code>. Running jobs are not affected; they continue on their own.</p>
<p>Check its state with <code>flux-plumb --status</code>.</p>
<button onclick="location.reload()">Try again</button></div>
<script>setTimeout(function(){location.reload()},10000)</script></body></html>`;

self.addEventListener("fetch", (event) => {
  if (event.request.mode !== "navigate") return;
  event.respondWith(
    fetch(event.request).catch(
      () => new Response(OFFLINE, { status: 503, headers: { "Content-Type": "text/html; charset=utf-8" } }),
    ),
  );
});
