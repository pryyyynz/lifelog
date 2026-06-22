'use client';

import { useEffect, useState } from 'react';
import { CheckCircle2, Loader2, UploadCloud } from 'lucide-react';
import { uploadIngest } from '@/lib/api';

const SHARE_CACHE = 'lifelog-share-v1';

// Pull the files the service worker stashed (from an OS share), rebuild them as
// File objects, and ingest them with auto type-detection.
async function readSharedFiles(): Promise<File[]> {
  if (typeof caches === 'undefined') return [];
  const cache = await caches.open(SHARE_CACHE);
  const index = await cache.match('/__shared/index');
  if (!index) return [];
  const keys: string[] = await index.json();
  const files: File[] = [];
  for (const key of keys) {
    const res = await cache.match(key);
    if (!res) continue;
    const blob = await res.blob();
    const name = decodeURIComponent(res.headers.get('x-filename') || 'file');
    files.push(new File([blob], name, { type: blob.type }));
    await cache.delete(key);
  }
  await cache.delete('/__shared/index');
  return files;
}

export default function ShareTargetPage() {
  const [state, setState] = useState<'working' | 'done' | 'empty' | 'error'>('working');
  const [detail, setDetail] = useState('');

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const files = await readSharedFiles();
        if (files.length === 0) {
          if (!cancelled) setState('empty');
          return;
        }
        const result = await uploadIngest('auto', files);
        if (cancelled) return;
        const types = result.by_type
          ? Object.entries(result.by_type).map(([k, v]) => `${v} ${k}`).join(', ')
          : `${result.saved}`;
        setDetail(`Ingesting ${types}.`);
        setState('done');
      } catch (err) {
        if (!cancelled) {
          setDetail(err instanceof Error ? err.message : 'Upload failed');
          setState('error');
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="flex h-screen flex-col items-center justify-center gap-4 px-6 text-center text-gray-100">
      {state === 'working' && (
        <>
          <Loader2 className="h-8 w-8 animate-spin text-indigo-400" />
          <p className="text-sm text-gray-400">Adding shared files to your lifelog…</p>
        </>
      )}
      {state === 'done' && (
        <>
          <CheckCircle2 className="h-10 w-10 text-emerald-400" />
          <p className="text-base font-medium">Shared to Lifelog</p>
          <p className="text-sm text-gray-400">{detail}</p>
          <a href="/" className="mt-2 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 px-4 py-2 text-sm font-medium text-white">
            Open Lifelog
          </a>
        </>
      )}
      {state === 'empty' && (
        <>
          <UploadCloud className="h-10 w-10 text-gray-500" />
          <p className="text-sm text-gray-400">Nothing to add. Share files from another app to ingest them.</p>
          <a href="/" className="mt-2 text-sm text-indigo-300">Open Lifelog</a>
        </>
      )}
      {state === 'error' && (
        <>
          <p className="text-base font-medium text-red-300">Couldn’t add shared files</p>
          <p className="text-sm text-gray-400">{detail}</p>
          <a href="/" className="mt-2 text-sm text-indigo-300">Open Lifelog</a>
        </>
      )}
    </div>
  );
}
