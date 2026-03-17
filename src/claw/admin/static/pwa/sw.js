/**
 * Service worker for The Claw PWA.
 *
 * Caches the app shell for offline loading (the UI works offline,
 * but voice/chat require server connectivity through WireGuard).
 */
const CACHE_NAME = 'claw-v2';  // Bump on each deploy to invalidate stale caches
const APP_SHELL = [
  '/app',
  '/static/pwa/audio-processor.js',
  '/static/pwa/manifest.json',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(
        names
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  // Only cache GET requests for app shell
  if (event.request.method !== 'GET') return;

  // Don't cache API calls or WebSocket upgrades
  const url = new URL(event.request.url);
  if (url.pathname.startsWith('/api/')) return;

  event.respondWith(
    caches.match(event.request).then((cached) => {
      // Network-first for app shell, fallback to cache
      return fetch(event.request)
        .then((response) => {
          if (response.ok && APP_SHELL.includes(url.pathname)) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return response;
        })
        .catch(() => cached);
    })
  );
});
