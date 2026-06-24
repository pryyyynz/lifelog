'use client';

import { useState } from 'react';
import { ChevronDown, MapPin } from 'lucide-react';
import type { SessionCardOut } from '@/lib/types';
import { sourceMeta } from '@/lib/sources';
import HitItem from './HitItem';

interface SessionCardProps {
  card: SessionCardOut;
}

export default function SessionCard({ card }: SessionCardProps) {
  const [expanded, setExpanded] = useState(false);

  const startDate = card.start_utc
    ? new Date(card.start_utc).toLocaleDateString(undefined, {
        weekday: 'short',
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      })
    : null;

  const startTime = card.start_utc
    ? new Date(card.start_utc).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : null;

  const endTime =
    card.end_utc && card.start_utc && card.end_utc !== card.start_utc
      ? new Date(card.end_utc).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      : null;

  const place = card.primary.place_name;
  const sourceTypes = card.modalities?.length
    ? card.modalities
    : Array.from(new Set([card.primary, ...card.secondary].map((hit) => hit.source_type)));

  return (
    <div className="overflow-hidden rounded-2xl border border-white/10 bg-white/[0.03] shadow-sm">
      <div className="flex items-center justify-between gap-2 border-b border-white/10 bg-white/[0.02] px-4 py-2.5">
        <div className="flex min-w-0 items-center gap-2.5">
          {startDate && <span className="text-sm font-medium text-gray-100">{startDate}</span>}
          {startTime && (
            <span className="text-xs text-gray-400">
              {startTime}
              {endTime ? ` – ${endTime}` : ''}
            </span>
          )}
          {place && (
            <span className="flex items-center gap-1 truncate text-xs text-gray-400">
              <MapPin className="h-3 w-3 shrink-0" />
              {place}
            </span>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <div className="flex items-center gap-1">
            {sourceTypes.map((st) => {
              const meta = sourceMeta(st);
              return (
                <span
                  key={st}
                  title={meta.label}
                  className={`inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-[10px] ${meta.bg} ${meta.text}`}
                >
                  <span className={`h-1.5 w-1.5 rounded-full ${meta.dot}`} />
                  {meta.label}
                </span>
              );
            })}
          </div>
        </div>
      </div>

      {card.title && (
        <div className="px-4 pt-3 text-sm font-semibold text-gray-100">{card.title}</div>
      )}
      {card.summary && (
        <div className="px-4 pt-1 text-xs leading-relaxed text-gray-400">{card.summary}</div>
      )}

      <div className="p-3">
        <HitItem hit={card.primary} isPrimary />
      </div>

      {card.secondary.length > 0 && (
        <div className="px-3 pb-3">
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="inline-flex items-center gap-1 text-xs text-indigo-300 transition hover:text-indigo-200"
          >
            <ChevronDown className={`h-3.5 w-3.5 transition-transform ${expanded ? 'rotate-180' : ''}`} />
            {expanded ? 'Hide' : `${card.secondary.length} more from this session`}
          </button>
          {expanded && (
            <div className="mt-2 space-y-2">
              {card.secondary.map((hit) => (
                <HitItem key={hit.chunk_id} hit={hit} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
