'use client';

import { useState } from 'react';
import { ExternalLink } from 'lucide-react';
import type { HitOut } from '@/lib/types';
import { openFile } from '@/lib/api';
import { sourceMeta } from '@/lib/sources';
import MediaPreview from './MediaPreview';

interface HitItemProps {
  hit: HitOut;
  isPrimary?: boolean;
}

export default function HitItem({ hit, isPrimary = false }: HitItemProps) {
  const [opening, setOpening] = useState(false);
  const meta = sourceMeta(hit.source_type);
  const Icon = meta.Icon;

  const handleOpen = async () => {
    setOpening(true);
    try {
      await openFile(hit.file_path);
    } finally {
      setOpening(false);
    }
  };

  // For media, the snippet is the behind-the-scenes OCR/VLM description used for
  // search — show the media itself, not that caption.
  const isMedia =
    hit.preview_type === 'image' || hit.preview_type === 'video' || hit.preview_type === 'audio';

  return (
    <div
      className={`group rounded-xl border p-3 transition ${
        isPrimary
          ? 'border-white/10 bg-white/[0.05]'
          : 'border-white/5 bg-white/[0.02] hover:bg-white/[0.04]'
      }`}
    >
      <div className="flex items-start gap-3">
        <span className={`mt-0.5 inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-lg ${meta.bg} ${meta.text}`}>
          <Icon className="h-4 w-4" />
        </span>

        <div className="min-w-0 flex-1 space-y-2">
          {hit.snippet && !isMedia && (
            <p className="text-sm leading-relaxed text-gray-200 line-clamp-4">{hit.snippet}</p>
          )}

          <MediaPreview hit={hit} onOpenOriginal={handleOpen} />
        </div>

        <button
          type="button"
          onClick={handleOpen}
          disabled={opening}
          title="Open original file"
          className="shrink-0 rounded-lg p-1.5 text-gray-500 opacity-0 transition hover:bg-white/10 hover:text-indigo-300 group-hover:opacity-100 focus:opacity-100 disabled:opacity-30"
        >
          <ExternalLink className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
