const CACHE = 'atlas-v1';
const ASSETS = [
  '/',
  '/index.html',
  '/app.js',
  '/style.css',
  '/manifest.json',
  'https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600;700&display=swap'
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  if (e.request.url.includes('financialmodelingprep') || e.request.url.includes('api.')) {
    // Network first for API calls
    e.respondWith(
      fetch(e.request).catch(() => caches.match(e.request))
    );
  } else {
    // Cache first for assets
    e.respondWith(
      caches.match(e.request).then(r => r || fetch(e.request).then(res => {
        const clone = res.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone));
        return res;
      }))
    );
  }
});

// Background sync for portfolio updates
self.addEventListener('sync', e => {
  if (e.tag === 'portfolio-sync') {
    e.waitUntil(syncPortfolio());
  }
});

async function syncPortfolio() {
  // Placeholder for background portfolio data refresh
  console.log('[ATLAS SW] Background sync triggered');
}
