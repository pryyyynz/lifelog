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

  const time = hit.timestamp_utc
    ? new Date(hit.timestamp_utc).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : null;
  const fileName = hit.file_path.split(/[\\/]/).pop();
  const rationale = (hit.rationale ?? []).slice(0, 2).filter(Boolean);
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

          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-gray-500">
            <span className={`rounded-full px-2 py-0.5 ${meta.bg} ${meta.text}`}>{meta.label}</span>
            {time && <span>{time}</span>}
            {hit.place_name && <span>· {hit.place_name}</span>}
            {fileName && (
              <span className="max-w-[12rem] truncate" title={hit.file_path}>
                · {fileName}
              </span>
            )}
            {rationale.length > 0 && <span className="text-gray-600">· {rationale.join(', ')}</span>}
          </div>
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
