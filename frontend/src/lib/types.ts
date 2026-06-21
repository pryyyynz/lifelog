// TypeScript types matching the FastAPI response schemas

export interface QueryFilters {
  source_type?: string;
  session_id?: string;
  date_from?: string;
  date_to?: string;
}

export interface QueryRequest {
  query: string;
  filters?: QueryFilters;
  top_k?: number;
  conversation_id?: string;
  chronological?: boolean;
}

export interface HitOut {
  chunk_id: string;
  source_type: string;
  file_path: string;
  score: number;
  rank: number;
  rationale: string[];
  match_reasons: string[];
  timestamp_utc: string | null;
  timestamp_display: string | null;
  session_id: string | null;
  snippet: string | null;
  place_name: string | null;
  thumbnail_path: string | null;
  preview_url?: string | null;
  preview_type?: 'image' | 'video' | 'audio' | null;
}

export interface SessionCardOut {
  session_id: string;
  score: number;
  start_utc: string | null;
  end_utc: string | null;
  modalities: string[];
  primary: HitOut;
  secondary: HitOut[];
  title?: string | null;
  summary?: string | null;
}

export interface QueryDebug {
  intent?: 'conversational' | 'search';
  temporal_range?: string[] | null;
  place_names?: string[];
  person_names?: string[];
  visual_intent?: boolean;
  visual_keyword_count?: number;
  total_hits_before_grouping?: number;
  clarification_needed?: boolean;
  options?: string[];
}

export interface QueryResponse {
  sessions: SessionCardOut[];
  conversation_id: string;
  clarification_prompt?: string | null;
  chat_message?: string | null;
  answer?: string | null;
  answer_citations?: string[];
  query_debug: QueryDebug;
}

export interface StatusResponse {
  status: string;
  environment: string;
  api_host: string;
  api_port: number;
  total_chunks: number;
  chunks_by_modality: Record<string, number>;
  files_by_modality: Record<string, number>;
  last_ingest_timestamp: string | null;
  sqlite_path?: string;
  vector_store_mode?: string;
  qdrant_url?: string;
}

export interface ConversationTurn {
  id: string;
  query: string;
  response?: QueryResponse;
  timestamp: string;
  status?: 'pending' | 'complete' | 'error';
  error?: string;
  filters?: QueryFilters;
  chronological?: boolean;
  /** Data URL of an image attached to this turn (photo search). */
  imageDataUrl?: string;
}

export interface ChatRecord {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  conversationId?: string;
  turns: ConversationTurn[];
}

export interface PreviewFileOut {
  path: string;
  name: string;
  extension: string;
  preview_url: string | null;
}

export interface SourcePreviewResponse {
  source_id: string;
  source_type: string;
  path: string;
  ok: boolean;
  item_count: number;
  errors: string[];
  warnings: string[];
  files: PreviewFileOut[];
}

export interface IngestStatusResponse {
  state: 'idle' | 'running' | 'done' | 'done_with_errors' | 'error';
  message: string;
  mode: string | null;
  source_id: string | null;
  started_at: string | null;
  finished_at: string | null;
  processed_items: number;
  skipped_items: number;
  failed_items: number;
}

export interface LocalPathSelectResponse {
  path: string | null;
}
