'use client';

import { useEffect, useState } from 'react';
import { CheckCircle2, Database, FileInput, FolderOpen, FolderSearch, Loader2, X } from 'lucide-react';
import { getIngestStatus, getStatus, previewSource, selectLocalPath, triggerIngest } from '@/lib/api';
import type { IngestStatusResponse, SourcePreviewResponse, StatusResponse } from '@/lib/types';

const SOURCE_TYPES = [
  { value: 'text', label: 'Text' },
  { value: 'photos', label: 'Photos' },
  { value: 'audio', label: 'Audio' },
  { value: 'video', label: 'Video' },
  { value: 'email', label: 'Email' },
  { value: 'calendar', label: 'Calendar' },
  { value: 'browser_history', label: 'Browser history' },
];

interface IngestDataButtonProps {
  onStatusChange: (status: StatusResponse) => void;
  onError: (message: string) => void;
}

export default function IngestDataButton({ onStatusChange, onError }: IngestDataButtonProps) {
  const [open, setOpen] = useState(false);
  const [sourceType, setSourceType] = useState('text');
  const [path, setPath] = useState('');
  const [full, setFull] = useState(false);
  const [preview, setPreview] = useState<SourcePreviewResponse | null>(null);
  const [ingestStatus, setIngestStatus] = useState<IngestStatusResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [selectingPath, setSelectingPath] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;

    let cancelled = false;
    const poll = async () => {
      try {
        const next = await getIngestStatus();
        if (!cancelled) setIngestStatus(next);
        if (next.state !== 'running') {
          const status = await getStatus();
          if (!cancelled) onStatusChange(status);
        }
      } catch {
        return;
      }
    };

    poll();
    const id = window.setInterval(poll, 1500);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [open, onStatusChange]);

  const handlePreview = async () => {
    if (!path.trim()) {
      setMessage('Enter a local file or folder path.');
      return;
    }

    setBusy(true);
    setMessage(null);
    try {
      const result = await previewSource(sourceType, path.trim());
      setPreview(result);
      if (!result.ok) setMessage(result.errors.join('; '));
    } catch (err) {
      const text = err instanceof Error ? err.message : 'Preview failed';
      setMessage(text);
      onError(text);
    } finally {
      setBusy(false);
    }
  };

  const handleSelectPath = async (target: 'file' | 'folder') => {
    setSelectingPath(true);
    setMessage(null);
    try {
      const selected = await selectLocalPath(sourceType, target);
      if (selected.path) {
        setPath(selected.path);
        setPreview(null);
      }
    } catch (err) {
      const text = err instanceof Error ? err.message : 'Path selection failed';
      setMessage(text);
      onError(text);
    } finally {
      setSelectingPath(false);
    }
  };

  const handleIngest = async () => {
    if (!path.trim()) {
      setMessage('Enter a local file or folder path.');
      return;
    }

    setBusy(true);
    setMessage(null);
    try {
      await triggerIngest(full, { sourceType, path: path.trim() });
      setIngestStatus({
        state: 'running',
        message: 'Ingest running',
        mode: full ? 'full' : 'incremental',
        source_id: preview?.source_id ?? null,
        started_at: new Date().toISOString(),
        finished_at: null,
        processed_items: 0,
        skipped_items: 0,
        failed_items: 0,
      });
    } catch (err) {
      const text = err instanceof Error ? err.message : 'Ingest failed';
      setMessage(text);
      onError(text);
    } finally {
      setBusy(false);
    }
  };

  const running = ingestStatus?.state === 'running';
  const done = ingestStatus?.state === 'done' || ingestStatus?.state === 'done_with_errors';

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.06] px-3 py-1.5 text-xs text-gray-200 transition hover:bg-white/10"
      >
        <Database className="h-4 w-4" />
        Ingest data
      </button>

      {open && (
        <div className="absolute right-0 top-9 z-20 w-[420px] rounded-xl border border-white/10 bg-gray-900/95 shadow-2xl backdrop-blur">
          <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
            <div className="text-sm font-medium text-gray-100">Ingest data</div>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded p-1 text-gray-500 hover:bg-gray-800 hover:text-gray-200"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <div className="space-y-4 px-4 py-4">
            <div className="grid grid-cols-2 gap-3">
              <label className="space-y-1 text-xs text-gray-400">
                Data type
                <select
                  value={sourceType}
                  onChange={(event) => {
                    setSourceType(event.target.value);
                    setPreview(null);
                    setMessage(null);
                  }}
                  className="w-full rounded-lg border border-white/10 bg-white/5 px-2 py-2 text-sm text-gray-100"
                >
                  {SOURCE_TYPES.map((type) => (
                    <option key={type.value} value={type.value}>
                      {type.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="space-y-1 text-xs text-gray-400">
                Sync mode
                <select
                  value={full ? 'full' : 'incremental'}
                  onChange={(event) => setFull(event.target.value === 'full')}
                  className="w-full rounded-lg border border-white/10 bg-white/5 px-2 py-2 text-sm text-gray-100"
                >
                  <option value="incremental">Partial sync</option>
                  <option value="full">Full ingest</option>
                </select>
              </label>
            </div>

            <label className="space-y-1 text-xs text-gray-400">
              Local file or folder path
              <div className="flex gap-2">
                <input
                  value={path}
                  onChange={(event) => {
                    setPath(event.target.value);
                    setPreview(null);
                    setMessage(null);
                  }}
                  onDoubleClick={() => handleSelectPath('folder')}
                  placeholder="C:\\Users\\pryyy\\Pictures\\Screenshots"
                  className="min-w-0 flex-1 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-gray-100 placeholder-gray-600"
                />
                <button
                  type="button"
                  onClick={() => handleSelectPath('folder')}
                  disabled={busy || running || selectingPath}
                  title="Browse for folder"
                  className="shrink-0 rounded-lg border border-white/10 bg-white/5 px-2.5 text-gray-300 hover:bg-white/10 disabled:opacity-50"
                >
                  {selectingPath ? <Loader2 className="h-4 w-4 animate-spin" /> : <FolderOpen className="h-4 w-4" />}
                </button>
                <button
                  type="button"
                  onClick={() => handleSelectPath('file')}
                  disabled={busy || running || selectingPath}
                  title="Browse for file"
                  className="shrink-0 rounded-lg border border-white/10 bg-white/5 px-2.5 text-gray-300 hover:bg-white/10 disabled:opacity-50"
                >
                  <FileInput className="h-4 w-4" />
                </button>
              </div>
            </label>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handlePreview}
                disabled={busy || running || selectingPath}
                className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-gray-200 transition hover:bg-white/10 disabled:opacity-50"
              >
                <FolderSearch className="h-4 w-4" />
                Preview
              </button>
              <button
                type="button"
                onClick={handleIngest}
                disabled={busy || running || selectingPath}
                className="flex items-center gap-2 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-500 px-3 py-2 text-xs font-medium text-white shadow-md transition hover:from-indigo-400 hover:to-violet-400 disabled:opacity-50"
              >
                {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Database className="h-4 w-4" />}
                {full ? 'Run full ingest' : 'Run partial sync'}
              </button>
            </div>

            {preview && (
              <div className="rounded-lg border border-white/10 bg-black/20 p-3">
                <div className="flex items-center justify-between text-xs text-gray-400">
                  <span>{preview.item_count} matching files</span>
                  <span>{preview.ok ? 'Ready' : 'Invalid source'}</span>
                </div>
                {preview.warnings.length > 0 && (
                  <div className="mt-2 text-xs text-amber-300">{preview.warnings.join('; ')}</div>
                )}
                {sourceType === 'photos' ? (
                  <div className="mt-3 grid grid-cols-4 gap-2">
                    {preview.files.map((file) => (
                      <div key={file.path} className="aspect-square overflow-hidden rounded border border-gray-800 bg-gray-800">
                        {file.preview_url ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img src={file.preview_url} alt={file.name} className="h-full w-full object-cover" />
                        ) : null}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="mt-3 max-h-36 space-y-1 overflow-y-auto text-xs text-gray-500">
                    {preview.files.map((file) => (
                      <div key={file.path} className="truncate" title={file.path}>
                        {file.name}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {(running || done || ingestStatus?.state === 'error') && (
              <div className="rounded-lg border border-white/10 bg-black/20 px-3 py-2 text-xs text-gray-300">
                <div className="flex items-center gap-2">
                  {running && <Loader2 className="h-4 w-4 animate-spin text-indigo-300" />}
                  {done && <CheckCircle2 className="h-4 w-4 text-emerald-300" />}
                  <span>{ingestStatus?.message}</span>
                </div>
              </div>
            )}

            {message && <div className="text-xs text-red-300">{message}</div>}
          </div>
        </div>
      )}
    </div>
  );
}
