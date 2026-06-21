import type {
  IngestStatusResponse,
  LocalPathSelectResponse,
  QueryRequest,
  QueryResponse,
  SourcePreviewResponse,
  StatusResponse,
} from './types';

const API_BASE = '/api';

export async function runQuery(req: QueryRequest): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Query failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<QueryResponse>;
}

export async function runImageQuery(params: {
  image: Blob;
  query?: string;
  top_k?: number;
  conversation_id?: string;
}): Promise<QueryResponse> {
  const form = new FormData();
  const type = params.image.type || 'image/png';
  const ext = type.includes('jpeg') || type.includes('jpg') ? 'jpg' : type.split('/')[1] || 'png';
  form.append('image', params.image, `query.${ext}`);
  if (params.query) form.append('query', params.query);
  form.append('top_k', String(params.top_k ?? 5));
  if (params.conversation_id) form.append('conversation_id', params.conversation_id);

  const res = await fetch(`${API_BASE}/query/image`, { method: 'POST', body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Image search failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<QueryResponse>;
}

export async function getStatus(): Promise<StatusResponse> {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) throw new Error(`Status failed (${res.status})`);
  return res.json() as Promise<StatusResponse>;
}

export async function triggerIngest(
  full: boolean,
  options?: { sourceType?: string; path?: string; sourceId?: string },
): Promise<void> {
  const res = await fetch(`${API_BASE}/ingest/trigger`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      full,
      source_id: options?.sourceId,
      source_type: options?.sourceType,
      path: options?.path,
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Ingest failed (${res.status}): ${text}`);
  }
}

export async function previewSource(sourceType: string, path: string): Promise<SourcePreviewResponse> {
  const res = await fetch(`${API_BASE}/sources/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_type: sourceType, path }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Preview failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<SourcePreviewResponse>;
}

export async function getIngestStatus(): Promise<IngestStatusResponse> {
  const res = await fetch(`${API_BASE}/ingest/status`);
  if (!res.ok) throw new Error(`Ingest status failed (${res.status})`);
  return res.json() as Promise<IngestStatusResponse>;
}

export async function selectLocalPath(
  sourceType: string,
  target: 'file' | 'folder',
): Promise<LocalPathSelectResponse> {
  const res = await fetch(`${API_BASE}/local-path/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_type: sourceType, target }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Path selection failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<LocalPathSelectResponse>;
}

export async function transcribeAudio(blob: Blob): Promise<{ text: string; language: string | null }> {
  const form = new FormData();
  const ext = blob.type.includes('mp4') ? 'mp4' : blob.type.includes('ogg') ? 'ogg' : 'webm';
  form.append('audio', blob, `query.${ext}`);
  const res = await fetch(`${API_BASE}/transcribe`, { method: 'POST', body: form });
  if (!res.ok) {
    let detail = await res.text();
    try {
      detail = JSON.parse(detail).detail ?? detail;
    } catch {
      // keep raw text
    }
    throw new Error(detail || `Transcription failed (${res.status})`);
  }
  return res.json() as Promise<{ text: string; language: string | null }>;
}

export async function openFile(filePath: string, timestampSec?: number): Promise<void> {
  await fetch(`${API_BASE}/open-file`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_path: filePath, timestamp_sec: timestampSec ?? null }),
  });
}
