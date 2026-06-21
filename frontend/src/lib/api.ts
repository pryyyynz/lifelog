import type {
  IngestStatusResponse,
  LocalPathSelectResponse,
  QueryRequest,
  QueryResponse,
  SourcePreviewResponse,
  StatusResponse,
} from './types';

const API_BASE = '/api';
const TOKEN_KEY = 'lifelog_token';

export function getToken(): string | null {
  if (typeof window === 'undefined') return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

function setToken(token: string): void {
  window.localStorage.setItem(TOKEN_KEY, token);
}

// Clear the session and notify the AuthGate to show the login screen.
export function logout(): void {
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem(TOKEN_KEY);
  window.dispatchEvent(new Event('lifelog:unauthorized'));
}

// Append the session token to a media URL. <img>/<video>/<audio> tags can't
// send an Authorization header, so token-protected previews ride a query param.
export function mediaUrl(url: string): string {
  const token = getToken();
  if (!token) return url;
  const sep = url.includes('?') ? '&' : '?';
  return `${url}${sep}token=${encodeURIComponent(token)}`;
}

// Shared fetch: injects the bearer token and funnels every 401 through logout()
// so an expired/invalid session bounces the user back to the login screen.
async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (res.status === 401) {
    logout();
    throw new Error('Session expired — please log in again.');
  }
  return res;
}

export async function getAuthStatus(): Promise<{ auth_required: boolean }> {
  const res = await fetch(`${API_BASE}/auth/status`);
  if (!res.ok) throw new Error(`Auth status failed (${res.status})`);
  return res.json() as Promise<{ auth_required: boolean }>;
}

export async function login(password: string): Promise<void> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password }),
  });
  if (!res.ok) {
    if (res.status === 401) throw new Error('Incorrect password');
    const text = await res.text();
    throw new Error(`Login failed (${res.status}): ${text}`);
  }
  const data = (await res.json()) as { token: string };
  setToken(data.token);
}

export async function runQuery(req: QueryRequest): Promise<QueryResponse> {
  const res = await apiFetch('/query', {
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

  const res = await apiFetch('/query/image', { method: 'POST', body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Image search failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<QueryResponse>;
}

export async function getStatus(): Promise<StatusResponse> {
  const res = await apiFetch('/status');
  if (!res.ok) throw new Error(`Status failed (${res.status})`);
  return res.json() as Promise<StatusResponse>;
}

export async function triggerIngest(
  full: boolean,
  options?: { sourceType?: string; path?: string; sourceId?: string },
): Promise<void> {
  const res = await apiFetch('/ingest/trigger', {
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
  const res = await apiFetch('/sources/preview', {
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
  const res = await apiFetch('/ingest/status');
  if (!res.ok) throw new Error(`Ingest status failed (${res.status})`);
  return res.json() as Promise<IngestStatusResponse>;
}

// Upload files from this device (e.g. a phone) to be ingested on the backend.
export async function uploadIngest(
  sourceType: string,
  files: File[],
): Promise<{ saved: number; skipped: string[]; source_id: string }> {
  const form = new FormData();
  form.append('source_type', sourceType);
  for (const file of files) form.append('files', file);
  const res = await apiFetch('/ingest/upload', { method: 'POST', body: form });
  if (!res.ok) {
    let detail = await res.text();
    try {
      detail = JSON.parse(detail).detail ?? detail;
    } catch {
      // keep raw text
    }
    throw new Error(detail || `Upload failed (${res.status})`);
  }
  return res.json() as Promise<{ saved: number; skipped: string[]; source_id: string }>;
}

export async function selectLocalPath(
  sourceType: string,
  target: 'file' | 'folder',
): Promise<LocalPathSelectResponse> {
  const res = await apiFetch('/local-path/select', {
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
  const res = await apiFetch('/transcribe', { method: 'POST', body: form });
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
  await apiFetch('/open-file', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_path: filePath, timestamp_sec: timestampSec ?? null }),
  });
}
