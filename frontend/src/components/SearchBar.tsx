'use client';

import { useEffect, useState, useRef, type KeyboardEvent } from 'react';
import { ChevronDown, Filter, ImagePlus, Loader2, Mic, Send, Square, X } from 'lucide-react';
import { transcribeAudio } from '@/lib/api';
import type { QueryFilters } from '@/lib/types';

interface SearchBarProps {
  onQuery: (query: string, filters: QueryFilters, chronological: boolean, image?: File) => void;
  loading: boolean;
  text: string;
  onTextChange: (text: string) => void;
  focusToken?: number;
}

const SOURCE_TYPES = ['text', 'email', 'photo', 'audio', 'video', 'calendar', 'browser_history'];

const AUTO_SUBMIT_KEY = 'lifelog.voiceAutoSubmit.v1';
// Discard accidental taps shorter than this (ms) so they never hit the backend.
const MIN_RECORDING_MS = 300;
// Hard cap so a forgotten recording can't run away.
const MAX_RECORDING_MS = 60_000;

type RecState = 'idle' | 'recording' | 'transcribing';

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === 'undefined') return undefined;
  for (const type of ['audio/webm', 'audio/mp4', 'audio/ogg']) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return undefined;
}

export default function SearchBar({ onQuery, loading, text, onTextChange, focusToken }: SearchBarProps) {
  const [showFilters, setShowFilters] = useState(false);
  const [sourceType, setSourceType] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [chronological, setChronological] = useState(false);
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Voice search state
  const [voiceSupported, setVoiceSupported] = useState(false);
  const [recState, setRecState] = useState<RecState>('idle');
  const [autoSubmit, setAutoSubmit] = useState(false);
  const [voiceError, setVoiceError] = useState('');
  const [voiceHint, setVoiceHint] = useState('');
  const [elapsed, setElapsed] = useState(0);
  const [showVoiceMenu, setShowVoiceMenu] = useState(false);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startedAtRef = useRef(0);
  const capTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Latest text, so the recorder's onstop closure appends to current input.
  const textRef = useRef(text);
  textRef.current = text;
  const autoSubmitRef = useRef(autoSubmit);
  autoSubmitRef.current = autoSubmit;

  useEffect(() => {
    if (focusToken === undefined) return;
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.focus();
    textarea.setSelectionRange(text.length, text.length);
  }, [focusToken, text.length]);

  // Feature-detect mic support and restore the auto-submit preference.
  useEffect(() => {
    setVoiceSupported(
      typeof navigator !== 'undefined' &&
        !!navigator.mediaDevices?.getUserMedia &&
        typeof MediaRecorder !== 'undefined',
    );
    try {
      setAutoSubmit(window.localStorage.getItem(AUTO_SUBMIT_KEY) === '1');
    } catch {
      // ignore unavailable storage
    }
  }, []);

  // Clean up any live recording resources on unmount.
  useEffect(() => {
    return () => {
      if (capTimerRef.current) clearTimeout(capTimerRef.current);
      if (tickRef.current) clearInterval(tickRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const setAutoSubmitPersisted = (value: boolean) => {
    setAutoSubmit(value);
    try {
      window.localStorage.setItem(AUTO_SUBMIT_KEY, value ? '1' : '0');
    } catch {
      // ignore
    }
  };

  const buildFilters = (): QueryFilters => {
    const filters: QueryFilters = {};
    if (sourceType) filters.source_type = sourceType;
    if (dateFrom) filters.date_from = dateFrom;
    if (dateTo) filters.date_to = dateTo;
    return filters;
  };

  const clearImage = () => {
    setImage(null);
    setImagePreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return '';
    });
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const onImagePick = (file: File | undefined) => {
    if (!file) return;
    setImage(file);
    setImagePreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });
  };

  // Revoke the last preview URL on unmount.
  useEffect(() => () => {
    if (imagePreview) URL.revokeObjectURL(imagePreview);
  }, [imagePreview]);

  const submitText = (value: string) => {
    const q = value.trim();
    if ((!q && !image) || loading) return;
    onQuery(q, buildFilters(), chronological, image ?? undefined);
    onTextChange('');
    clearImage();
    textareaRef.current?.focus();
  };

  const submit = () => submitText(text);

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const stopTimers = () => {
    if (capTimerRef.current) {
      clearTimeout(capTimerRef.current);
      capTimerRef.current = null;
    }
    if (tickRef.current) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
  };

  const handleTranscript = (spoken: string) => {
    const clean = spoken.trim();
    if (!clean) {
      setVoiceHint("Didn't catch that — try again");
      return;
    }
    const existing = textRef.current.trim();
    const combined = existing ? `${existing} ${clean}` : clean;
    if (autoSubmitRef.current) {
      submitText(combined);
    } else {
      onTextChange(combined);
      textareaRef.current?.focus();
    }
  };

  const finishRecording = async () => {
    const blob = new Blob(chunksRef.current, {
      type: recorderRef.current?.mimeType || 'audio/webm',
    });
    chunksRef.current = [];
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    recorderRef.current = null;

    const durationMs = Date.now() - startedAtRef.current;
    if (durationMs < MIN_RECORDING_MS || blob.size === 0) {
      setRecState('idle');
      return;
    }

    setRecState('transcribing');
    try {
      const { text: spoken } = await transcribeAudio(blob);
      handleTranscript(spoken);
    } catch (err) {
      setVoiceError(err instanceof Error ? err.message : 'Transcription failed');
    } finally {
      setRecState('idle');
    }
  };

  const startRecording = async () => {
    setVoiceError('');
    setVoiceHint('');
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      const name = err instanceof DOMException ? err.name : '';
      setVoiceError(
        name === 'NotAllowedError'
          ? 'Microphone access blocked. Allow it in your browser to use voice search.'
          : 'No microphone available.',
      );
      return;
    }

    const mimeType = pickMimeType();
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    chunksRef.current = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.onstop = () => {
      stopTimers();
      void finishRecording();
    };

    streamRef.current = stream;
    recorderRef.current = recorder;
    startedAtRef.current = Date.now();
    recorder.start();
    setElapsed(0);
    setRecState('recording');

    tickRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAtRef.current) / 1000));
    }, 250);
    capTimerRef.current = setTimeout(() => {
      if (recorderRef.current?.state === 'recording') recorderRef.current.stop();
    }, MAX_RECORDING_MS);
  };

  const toggleRecording = () => {
    if (recState === 'transcribing' || loading) return;
    if (recState === 'recording') {
      recorderRef.current?.stop();
    } else {
      void startRecording();
    }
  };

  const micBusy = recState === 'transcribing';
  const recording = recState === 'recording';

  return (
    <div className="space-y-2">
      {showFilters && (
        <div className="flex flex-wrap gap-3 text-sm rounded-xl border border-white/10 bg-white/[0.04] px-4 py-3 backdrop-blur">
          <label className="flex items-center gap-2 text-gray-300">
            Type:
            <select
              value={sourceType}
              onChange={(e) => setSourceType(e.target.value)}
              className="rounded-md border border-white/10 bg-white/5 px-2 py-1 text-gray-100"
            >
              <option value="">all</option>
              {SOURCE_TYPES.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-gray-300">
            From:
            <input
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
              className="rounded-md border border-white/10 bg-white/5 px-2 py-1 text-gray-100"
            />
          </label>
          <label className="flex items-center gap-2 text-gray-300">
            To:
            <input
              type="date"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
              className="rounded-md border border-white/10 bg-white/5 px-2 py-1 text-gray-100"
            />
          </label>
          <label className="flex items-center gap-2 text-gray-300 select-none cursor-pointer">
            <input
              type="checkbox"
              checked={chronological}
              onChange={(e) => setChronological(e.target.checked)}
              className="accent-indigo-500"
            />
            Chronological
          </label>
        </div>
      )}

      {(voiceError || voiceHint) && (
        <div
          className={`text-xs px-1 ${voiceError ? 'text-red-400' : 'text-gray-400'}`}
          role="status"
        >
          {voiceError || voiceHint}
        </div>
      )}

      {imagePreview && (
        <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={imagePreview} alt="attached" className="h-12 w-12 rounded-lg border border-white/10 object-cover" />
          <div className="min-w-0 flex-1 text-xs text-gray-300">
            <div className="truncate font-medium">{image?.name}</div>
            <div className="text-gray-500">Photo search — add text to refine, or send to match this image</div>
          </div>
          <button
            type="button"
            onClick={clearImage}
            title="Remove image"
            className="shrink-0 rounded-lg p-1.5 text-gray-400 transition hover:bg-white/10 hover:text-gray-200"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => onImagePick(e.target.files?.[0] ?? undefined)}
      />

      <div className="flex items-end gap-2 rounded-2xl border border-white/10 bg-white/[0.04] p-1.5 shadow-lg backdrop-blur focus-within:border-indigo-500/60 focus-within:ring-1 focus-within:ring-indigo-500/40 transition">
        <button
          type="button"
          onClick={() => setShowFilters((v) => !v)}
          title="Toggle filters"
          className={`shrink-0 rounded-xl p-2.5 transition ${
            showFilters
              ? 'bg-indigo-500/20 text-indigo-300'
              : 'text-gray-400 hover:bg-white/10 hover:text-gray-200'
          }`}
        >
          <Filter className="h-4 w-4" />
        </button>

        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={loading || recording}
          title="Search by photo"
          className={`shrink-0 rounded-xl p-2.5 transition disabled:opacity-50 ${
            image
              ? 'bg-indigo-500/20 text-indigo-300'
              : 'text-gray-400 hover:bg-white/10 hover:text-gray-200'
          }`}
        >
          <ImagePlus className="h-4 w-4" />
        </button>

        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => onTextChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={
            recording
              ? `Listening… ${elapsed}s (tap mic to stop)`
              : 'Ask about your memories or say hi... (Enter to send)'
          }
          rows={1}
          className="flex-1 resize-none border-0 bg-transparent px-2 py-2.5 text-sm leading-relaxed text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-0"
          style={{ minHeight: '44px', maxHeight: '160px' }}
          onInput={(e) => {
            const el = e.currentTarget;
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 160) + 'px';
          }}
          disabled={loading || recording}
        />

        {voiceSupported && (
          <div className="relative shrink-0 flex">
            <button
              type="button"
              onClick={toggleRecording}
              disabled={loading || micBusy}
              title={recording ? 'Stop recording' : micBusy ? 'Transcribing…' : 'Search by voice'}
              aria-pressed={recording}
              className={`rounded-l-xl p-2.5 transition disabled:opacity-60 ${
                recording
                  ? 'animate-pulse bg-red-600 text-white hover:bg-red-500'
                  : 'text-gray-400 hover:bg-white/10 hover:text-gray-200'
              }`}
            >
              {micBusy ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : recording ? (
                <Square className="h-4 w-4" />
              ) : (
                <Mic className="h-4 w-4" />
              )}
            </button>
            <button
              type="button"
              onClick={() => setShowVoiceMenu((v) => !v)}
              disabled={loading || recording || micBusy}
              title="Voice options"
              aria-label="Voice options"
              aria-expanded={showVoiceMenu}
              className="rounded-r-xl px-1.5 text-gray-400 transition hover:bg-white/10 hover:text-gray-200 disabled:opacity-60"
            >
              <ChevronDown className={`h-3.5 w-3.5 transition-transform ${showVoiceMenu ? 'rotate-180' : ''}`} />
            </button>

            {showVoiceMenu && (
              <>
                <div className="fixed inset-0 z-10" onClick={() => setShowVoiceMenu(false)} />
                <div className="absolute bottom-full right-0 mb-2 z-20 w-60 rounded-xl border border-white/10 bg-gray-900/95 p-3 shadow-xl backdrop-blur">
                  <label className="flex items-center gap-2 text-sm text-gray-100 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={autoSubmit}
                      onChange={(e) => setAutoSubmitPersisted(e.target.checked)}
                      className="accent-indigo-500"
                    />
                    Auto-send after voice
                  </label>
                  <p className="mt-1.5 text-xs leading-relaxed text-gray-400">
                    {autoSubmit
                      ? 'Voice queries run immediately after transcription.'
                      : 'Voice queries fill the box so you can edit, then press Enter.'}
                  </p>
                </div>
              </>
            )}
          </div>
        )}

        <button
          type="button"
          onClick={submit}
          disabled={loading || (!text.trim() && !image)}
          title="Send"
          className="shrink-0 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 p-2.5 text-white shadow-md transition hover:from-indigo-400 hover:to-violet-400 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? (
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </button>
      </div>
    </div>
  );
}
