// Minimal service worker: its ONLY job is to catch files shared into the app
// via the OS share sheet (manifest share_target POSTs here). Every other request
// is left untouched — no caching, no interception — so this can't affect normal
// page loads. Shared files are stashed in the Cache API and the page at
// /share-target?shared=1 reads them and uploads with the user's auth token.

const SHARE_CACHE = 'lifelog-share-v1';
const SHARE_PATH = '/share-target';

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (event.request.method === 'POST' && url.pathname === SHARE_PATH) {
    event.respondWith(handleShare(event.request));
  }
  // No else: all other requests fall through to the network normally.
});

async function handleShare(request) {
  try {
    const formData = await request.formData();
    const files = formData.getAll('files').filter((f) => f && typeof f !== 'string');
    const cache = await caches.open(SHARE_CACHE);
    const keys = [];
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const key = `/__shared/${Date.now()}-${i}`;
      await cache.put(
        new Request(key),
        new Response(file, {
          headers: {
            'content-type': file.type || 'application/octet-stream',
            'x-filename': encodeURIComponent(file.name || `file-${i}`),
          },
        }),
      );
      keys.push(key);
    }
    await cache.put(
      new Request('/__shared/index'),
      new Response(JSON.stringify(keys), { headers: { 'content-type': 'application/json' } }),
    );
  } catch (err) {
    // Swallow — the page will just show "nothing to upload".
  }
  return Response.redirect(`${SHARE_PATH}?shared=1`, 303);
}
