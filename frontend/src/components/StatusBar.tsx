'use client';

import type { StatusResponse } from '@/lib/types';
import { sourceMeta } from '@/lib/sources';

const MODALITY_ORDER = ['text', 'email', 'photo', 'audio', 'video', 'calendar', 'browser_history'];

export default function StatusBar({ status }: { status: StatusResponse }) {
  const lastSync = status.last_ingest_timestamp
    ? new Date(status.last_ingest_timestamp).toLocaleString()
    : 'never';

  return (
    <div className="scrollbar-thin flex shrink-0 items-center gap-3 overflow-x-auto border-t border-white/10 bg-gray-950/70 px-6 py-2 text-xs text-gray-500 backdrop-blur">
      <span className="shrink-0">
        Last sync: <span className="text-gray-400">{lastSync}</span>
      </span>
      <span className="h-3 w-px shrink-0 bg-white/10" />
      {MODALITY_ORDER.map((key) => {
        const meta = sourceMeta(key);
        return (
          <span key={key} className="flex shrink-0 items-center gap-1.5">
            <span className={`h-1.5 w-1.5 rounded-full ${meta.dot}`} />
            <span className="text-gray-400">{meta.label}</span>
            <span className="text-gray-300">{(status.files_by_modality?.[key] ?? 0).toLocaleString()}</span>
          </span>
        );
      })}
    </div>
  );
}
