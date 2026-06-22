'use client';

import { useEffect } from 'react';

// Registers the service worker that powers the OS share target. Best-effort —
// failures are ignored and never affect the app.
export default function PWARegister() {
  useEffect(() => {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js').catch(() => {});
    }
  }, []);
  return null;
}
