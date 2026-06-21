'use client';

import { useState } from 'react';
import { Maximize2 } from 'lucide-react';
import type { HitOut } from '@/lib/types';
import { mediaUrl } from '@/lib/api';
import Lightbox from './Lightbox';

// Renders a hit's media so it can be viewed/played right inside the chat.
// Images open a full-screen lightbox; video and audio play inline.
export default function MediaPreview({ hit, onOpenOriginal }: { hit: HitOut; onOpenOriginal?: () => void }) {
  const [lightbox, setLightbox] = useState(false);
  const url = hit.preview_url ? mediaUrl(hit.preview_url) : null;
  const name = hit.file_path.split(/[\\/]/).pop() ?? 'file';

  if (!url) return null;

  if (hit.preview_type === 'image') {
    // Inline preview uses a downscaled thumbnail; the lightbox keeps the full-res original.
    const thumbUrl = `${url}&thumb=1`;
    return (
      <>
        <button
          type="button"
          onClick={() => setLightbox(true)}
          title="Click to enlarge"
          className="group/media relative block overflow-hidden rounded-xl border border-white/10 bg-black/20"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={thumbUrl}
            alt={name}
            loading="lazy"
            decoding="async"
            className="max-h-72 w-full bg-black/40 object-contain transition duration-300 group-hover/media:scale-[1.02]"
          />
          <span className="pointer-events-none absolute right-2 top-2 rounded-md bg-black/50 p-1.5 text-white opacity-0 backdrop-blur-sm transition group-hover/media:opacity-100">
            <Maximize2 className="h-4 w-4" />
          </span>
        </button>
        {lightbox && (
          <Lightbox type="image" url={url} alt={name} onClose={() => setLightbox(false)} onOpenOriginal={onOpenOriginal} />
        )}
      </>
    );
  }

  if (hit.preview_type === 'video') {
    return (
      <video
        src={url}
        controls
        preload="metadata"
        className="max-h-72 w-full rounded-xl border border-white/10 bg-black/40"
      />
    );
  }

  if (hit.preview_type === 'audio') {
    return <audio src={url} controls preload="metadata" className="w-full" />;
  }

  return null;
}
