'use client';

import { useState, useRef, useCallback, useEffect, useMemo, type KeyboardEvent, type ReactNode } from 'react';
import {
  AlertCircle,
  Bot,
  Check,
  Clock,
  Copy,
  Layers,
  LogOut,
  MessageSquare,
  Pencil,
  Plus,
  RotateCcw,
  Search,
  Sparkles,
} from 'lucide-react';
import { runQuery, runImageQuery, getStatus, getToken, logout } from '@/lib/api';
import { chatTitleFromQuery, createChat, loadChats, saveChats } from '@/lib/chatStorage';
import type { ChatRecord, ConversationTurn, QueryResponse, StatusResponse, QueryFilters } from '@/lib/types';
import IngestDataButton from '@/components/IngestDataButton';
import SearchBar from '@/components/SearchBar';
import SessionCard from '@/components/SessionCard';
import StatusBar from '@/components/StatusBar';

function formatChatTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function sortChats(chats: ChatRecord[]): ChatRecord[] {
  return [...chats].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}

function newTurnId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `turn_${Date.now()}_${Math.random().toString(36).slice(2)}`;
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

async function dataUrlToBlob(dataUrl: string): Promise<Blob> {
  const res = await fetch(dataUrl);
  return res.blob();
}

function responseToText(response: QueryResponse): string {
  const parts: string[] = [];
  if (response.chat_message) parts.push(response.chat_message);
  if (response.answer) parts.push(response.answer);
  if (parts.length === 0) {
    response.sessions.forEach((card, index) => {
      const title = card.title || card.primary.session_id || `Result ${index + 1}`;
      const snippet = card.summary || card.primary.snippet || '';
      parts.push(snippet ? `${index + 1}. ${title}\n${snippet}` : `${index + 1}. ${title}`);
    });
  }
  return parts.join('\n\n').trim();
}

const DRAFT_CHAT: ChatRecord = {
  id: 'draft',
  title: 'New chat',
  createdAt: '',
  updatedAt: '',
  turns: [],
};

const EXAMPLE_PROMPTS = [
  'What did I do last summer?',
  'Show me photos from the beach',
  'Find emails about my flight',
  'What do you do?',
];

function FormattedText({ text }: { text: string }) {
  const paragraphs = text
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean);

  return (
    <div className="space-y-3 text-sm leading-6 text-gray-200">
      {paragraphs.map((paragraph, index) => {
        const lines = paragraph.split('\n').map((line) => line.trim()).filter(Boolean);
        const isList = lines.length > 1 && lines.every((line) => /^[-*]\s+/.test(line));
        if (isList) {
          return (
            <ul key={index} className="list-disc space-y-1 pl-5">
              {lines.map((line) => (
                <li key={line}>{line.replace(/^[-*]\s+/, '')}</li>
              ))}
            </ul>
          );
        }
        return <p key={index} className="whitespace-pre-line">{paragraph}</p>;
      })}
    </div>
  );
}

function StreamingText({ text }: { text: string }) {
  const [visible, setVisible] = useState('');

  useEffect(() => {
    setVisible('');
    let index = 0;
    const step = Math.max(3, Math.ceil(text.length / 90));
    const interval = window.setInterval(() => {
      index = Math.min(text.length, index + step);
      setVisible(text.slice(0, index));
      if (index >= text.length) window.clearInterval(interval);
    }, 16);
    return () => window.clearInterval(interval);
  }, [text]);

  return <FormattedText text={visible || text.slice(0, 1)} />;
}

// Left-aligned assistant row: gradient avatar + content column.
function AssistantShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex justify-start gap-3">
      <span className="mt-0.5 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 text-white shadow-md">
        <Bot className="h-4 w-4" />
      </span>
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}

function UserMessage({
  turn,
  onEdit,
  loading,
}: {
  turn: ConversationTurn;
  onEdit: (turnId: string, query: string) => void;
  loading: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(turn.query);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!editing) return;
    const el = textareaRef.current;
    if (!el) return;
    el.focus();
    el.setSelectionRange(el.value.length, el.value.length);
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }, [editing]);

  const startEdit = () => {
    setDraft(turn.query);
    setEditing(true);
  };

  const cancel = () => {
    setEditing(false);
    setDraft(turn.query);
  };

  const save = () => {
    const next = draft.trim();
    if (!next) return;
    setEditing(false);
    if (next !== turn.query.trim()) onEdit(turn.id, next);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      save();
    } else if (event.key === 'Escape') {
      event.preventDefault();
      cancel();
    }
  };

  if (editing) {
    return (
      <div className="flex justify-end">
        <div className="w-full max-w-2xl space-y-2">
          <textarea
            ref={textareaRef}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={onKeyDown}
            rows={1}
            className="w-full resize-none rounded-2xl border border-indigo-500/60 bg-white/[0.04] px-4 py-3 text-sm leading-6 text-gray-100 shadow-lg backdrop-blur focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
            style={{ maxHeight: '200px' }}
            onInput={(event) => {
              const el = event.currentTarget;
              el.style.height = 'auto';
              el.style.height = Math.min(el.scrollHeight, 200) + 'px';
            }}
          />
          <div className="flex items-center justify-end gap-2 text-sm">
            <button
              type="button"
              onClick={cancel}
              className="rounded-lg px-3 py-1.5 text-gray-300 transition hover:bg-white/10 hover:text-gray-100"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={save}
              disabled={loading || !draft.trim()}
              className="rounded-lg bg-gradient-to-br from-indigo-500 to-violet-500 px-3 py-1.5 font-medium text-white shadow-md transition hover:from-indigo-400 hover:to-violet-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Save &amp; submit
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-end">
      <div className="group max-w-2xl">
        <div className="flex items-center justify-end gap-2 pb-1 text-xs text-gray-500">
          <span>{formatChatTime(turn.timestamp)}</span>
          <button
            type="button"
            onClick={startEdit}
            className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-gray-400 opacity-0 transition hover:bg-white/10 hover:text-gray-200 group-hover:opacity-100 focus:opacity-100"
            title="Edit request"
          >
            <Pencil className="h-3.5 w-3.5" />
            Edit
          </button>
        </div>
        <div className="rounded-2xl rounded-tr-md bg-gradient-to-br from-indigo-500 to-violet-500 px-4 py-3 text-sm leading-6 text-white shadow-md">
          {turn.imageDataUrl && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={turn.imageDataUrl}
              alt="attached"
              className={`max-h-60 w-full rounded-lg bg-black/20 object-contain ${turn.query ? 'mb-2' : ''}`}
            />
          )}
          {turn.query && <div className="whitespace-pre-line">{turn.query}</div>}
        </div>
      </div>
    </div>
  );
}

function MessageActions({
  onRetry,
  copyText,
  loading,
}: {
  onRetry: () => void;
  copyText: string;
  loading: boolean;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (!copyText) return;
    try {
      await navigator.clipboard.writeText(copyText);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard unavailable — ignore
    }
  };

  const buttonClass =
    'inline-flex items-center gap-1.5 rounded-full border border-white/10 px-2.5 py-1 transition hover:bg-white/10 hover:text-gray-200 disabled:cursor-not-allowed disabled:opacity-40';

  return (
    <div className="flex items-center gap-2 pt-1 text-xs text-gray-400">
      <button type="button" onClick={onRetry} disabled={loading} className={buttonClass} title="Retry search">
        <RotateCcw className="h-3.5 w-3.5" />
        Retry
      </button>
      {copyText && (
        <button type="button" onClick={handleCopy} className={buttonClass} title="Copy result">
          {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      )}
    </div>
  );
}

function AssistantPending() {
  return (
    <AssistantShell>
      <div className="inline-flex max-w-2xl items-center gap-3 rounded-2xl rounded-tl-md border border-white/10 bg-white/[0.04] px-4 py-3 text-sm text-gray-300 shadow-sm">
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 animate-bounce rounded-full bg-indigo-400 [animation-delay:-0.3s]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-indigo-400 [animation-delay:-0.15s]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-indigo-400" />
        </span>
        Searching your lifelog…
      </div>
    </AssistantShell>
  );
}

function AssistantError({
  message,
  onRetry,
  loading,
}: {
  message: string;
  onRetry: () => void;
  loading: boolean;
}) {
  return (
    <AssistantShell>
      <div className="max-w-2xl space-y-2">
        <div className="rounded-2xl rounded-tl-md border border-red-500/40 bg-red-950/40 px-4 py-3 text-sm text-red-100">
          <div className="flex items-start gap-2">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{message}</span>
          </div>
        </div>
        <button
          type="button"
          onClick={onRetry}
          disabled={loading}
          className="inline-flex items-center gap-1.5 rounded-full border border-white/10 px-2.5 py-1 text-xs text-gray-400 transition hover:bg-white/10 hover:text-gray-200 disabled:cursor-not-allowed disabled:opacity-40"
          title="Retry search"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Retry
        </button>
      </div>
    </AssistantShell>
  );
}

function AssistantResponse({
  response,
  onRetry,
  loading,
}: {
  response: QueryResponse;
  onRetry: () => void;
  loading: boolean;
}) {
  const hasMessage = Boolean(response.chat_message);
  const hasPrompt = Boolean(response.clarification_prompt);
  const hasSessions = response.sessions.length > 0;

  return (
    <AssistantShell>
      <div className="w-full max-w-3xl space-y-3">
        {hasMessage && (
          <div className="rounded-2xl rounded-tl-md border border-white/10 bg-white/[0.04] px-4 py-3 shadow-sm">
            <StreamingText text={response.chat_message ?? ''} />
          </div>
        )}

        {hasPrompt && (
          <div className="rounded-2xl border border-amber-500/40 bg-amber-950/30 px-4 py-3 text-sm leading-6 text-amber-100">
            <FormattedText text={response.clarification_prompt ?? ''} />
          </div>
        )}

        {response.answer && (
          <div className="rounded-2xl rounded-tl-md border border-indigo-500/40 bg-indigo-950/30 px-4 py-3 shadow-sm">
            <div className="mb-1.5 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-indigo-300">
              <Sparkles className="h-3.5 w-3.5" />
              Answer
            </div>
            <FormattedText text={response.answer} />
            <div className="mt-2 text-xs text-gray-500">
              Grounded in the matches below — bracketed numbers refer to results in order.
            </div>
          </div>
        )}

        {!hasMessage && !hasSessions && !hasPrompt && (
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-gray-300">
            No results found.
          </div>
        )}

        {!hasMessage && hasSessions && (
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-3 px-1">
              <div className="flex items-center gap-2 text-sm font-medium text-gray-200">
                <Layers className="h-4 w-4 text-indigo-400" />
                Best matches
              </div>
              <div className="text-xs text-gray-500">
                {response.sessions.length} {response.sessions.length === 1 ? 'session' : 'sessions'}
              </div>
            </div>
            {response.sessions.map((card) => (
              <SessionCard key={card.session_id} card={card} />
            ))}
          </div>
        )}

        <MessageActions onRetry={onRetry} copyText={responseToText(response)} loading={loading} />
      </div>
    </AssistantShell>
  );
}

function ConversationTurnView({
  turn,
  onEdit,
  onRetry,
  loading,
}: {
  turn: ConversationTurn;
  onEdit: (turnId: string, query: string) => void;
  onRetry: (turnId: string) => void;
  loading: boolean;
}) {
  const status = turn.status ?? (turn.response ? 'complete' : 'pending');

  return (
    <div className="space-y-4">
      <UserMessage turn={turn} onEdit={onEdit} loading={loading} />
      {status === 'pending' && <AssistantPending />}
      {status === 'error' && (
        <AssistantError message={turn.error ?? 'Request failed'} onRetry={() => onRetry(turn.id)} loading={loading} />
      )}
      {status === 'complete' && turn.response && (
        <AssistantResponse response={turn.response} onRetry={() => onRetry(turn.id)} loading={loading} />
      )}
    </div>
  );
}

export default function Home() {
  const [chats, setChats] = useState<ChatRecord[]>([DRAFT_CHAT]);
  const [activeChatId, setActiveChatId] = useState<string | undefined>(DRAFT_CHAT.id);
  const [hydrated, setHydrated] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [composerText, setComposerText] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const saved = sortChats(loadChats());
    const initial = saved.length > 0 ? saved : [createChat()];
    setChats(initial);
    setActiveChatId(initial[0]?.id);
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (hydrated) saveChats(chats);
  }, [chats, hydrated]);

  useEffect(() => {
    getStatus().then(setStatus).catch(() => null);
  }, []);

  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeChatId) ?? chats[0],
    [activeChatId, chats],
  );
  const turns = activeChat?.turns ?? [];

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeChatId, turns.length, turns.at(-1)?.status]);

  // Write a completed response back onto a specific turn.
  const applyTurnResponse = useCallback((chatId: string, turnId: string, response: QueryResponse) => {
    const doneAt = new Date().toISOString();
    setChats((prev) => sortChats(
      prev.map((chat) => {
        if (chat.id !== chatId) return chat;
        return {
          ...chat,
          updatedAt: doneAt,
          conversationId: response.conversation_id,
          turns: chat.turns.map((turn) => (
            turn.id === turnId
              ? { ...turn, response, status: 'complete' as const, error: undefined }
              : turn
          )),
        };
      }),
    ));
  }, []);

  // Mark a specific turn as failed.
  const applyTurnError = useCallback((chatId: string, turnId: string, message: string) => {
    setError(message);
    setChats((prev) => sortChats(
      prev.map((chat) => {
        if (chat.id !== chatId) return chat;
        return {
          ...chat,
          updatedAt: new Date().toISOString(),
          turns: chat.turns.map((turn) => (
            turn.id === turnId
              ? { ...turn, status: 'error' as const, error: message }
              : turn
          )),
        };
      }),
    ));
  }, []);

  // Runs a text query for an already-existing turn and writes the result back.
  const runTurn = useCallback(
    async (
      targetChatId: string,
      turnId: string,
      query: string,
      filters: QueryFilters | undefined,
      chronological: boolean | undefined,
      conversationId: string | undefined,
    ) => {
      setLoading(true);
      setError(null);
      try {
        const response = await runQuery({
          query,
          filters,
          top_k: 5,
          conversation_id: conversationId,
          chronological,
        });
        applyTurnResponse(targetChatId, turnId, response);
      } catch (err) {
        applyTurnError(targetChatId, turnId, err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    },
    [applyTurnResponse, applyTurnError],
  );

  // Runs a photo search (image + optional text) for an existing turn.
  const runImageTurn = useCallback(
    async (
      targetChatId: string,
      turnId: string,
      query: string,
      imageDataUrl: string,
      conversationId: string | undefined,
    ) => {
      setLoading(true);
      setError(null);
      try {
        const image = await dataUrlToBlob(imageDataUrl);
        const response = await runImageQuery({
          image,
          query: query || undefined,
          top_k: 5,
          conversation_id: conversationId,
        });
        applyTurnResponse(targetChatId, turnId, response);
      } catch (err) {
        applyTurnError(targetChatId, turnId, err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    },
    [applyTurnResponse, applyTurnError],
  );

  const handleQuery = useCallback(
    async (query: string, filters: QueryFilters, chronological: boolean, image?: File) => {
      const now = new Date().toISOString();
      const turnId = newTurnId();
      const targetChatId = activeChat?.id ?? createChat(now).id;
      const chatConversationId = activeChat?.conversationId;

      const imageDataUrl = image ? await fileToDataUrl(image) : undefined;
      const titleText = query || (image ? 'Photo search' : '');
      const pendingTurn: ConversationTurn = {
        id: turnId,
        query,
        timestamp: now,
        status: 'pending',
        filters,
        chronological,
        imageDataUrl,
      };

      setActiveChatId(targetChatId);
      setChats((prev) => {
        const existing = prev.find((chat) => chat.id === targetChatId);
        if (!existing) {
          return sortChats([
            {
              id: targetChatId,
              title: chatTitleFromQuery(titleText),
              createdAt: now,
              updatedAt: now,
              turns: [pendingTurn],
            },
            ...prev,
          ]);
        }

        return sortChats(
          prev.map((chat) => {
            if (chat.id !== targetChatId) return chat;
            return {
              ...chat,
              title: chat.turns.length === 0 ? chatTitleFromQuery(titleText) : chat.title,
              updatedAt: now,
              turns: [...chat.turns, pendingTurn],
            };
          }),
        );
      });

      if (imageDataUrl) {
        await runImageTurn(targetChatId, turnId, query, imageDataUrl, chatConversationId);
      } else {
        await runTurn(targetChatId, turnId, query, filters, chronological, chatConversationId);
      }
    },
    [activeChat, runTurn, runImageTurn],
  );

  const handleNewConversation = () => {
    const chat = createChat();
    setChats((prev) => sortChats([chat, ...prev]));
    setActiveChatId(chat.id);
    setComposerText('');
    setError(null);
  };

  const handleSelectChat = (chatId: string) => {
    setActiveChatId(chatId);
    setError(null);
  };

  // Edit a past request in place: replace its query, drop the turns that came
  // after it (they belonged to the old branch), and re-run from that point.
  const handleEditTurn = useCallback(
    async (turnId: string, newQuery: string) => {
      const chat = activeChat;
      if (!chat) return;
      const target = chat.turns.find((turn) => turn.id === turnId);
      if (!target) return;

      const now = new Date().toISOString();
      setChats((prev) => sortChats(
        prev.map((c) => {
          if (c.id !== chat.id) return c;
          const index = c.turns.findIndex((turn) => turn.id === turnId);
          if (index === -1) return c;
          const editedTurn: ConversationTurn = {
            ...c.turns[index],
            query: newQuery,
            timestamp: now,
            status: 'pending',
            response: undefined,
            error: undefined,
          };
          return { ...c, updatedAt: now, turns: [...c.turns.slice(0, index), editedTurn] };
        }),
      ));

      if (target.imageDataUrl) {
        await runImageTurn(chat.id, turnId, newQuery, target.imageDataUrl, chat.conversationId);
      } else {
        await runTurn(chat.id, turnId, newQuery, target.filters, target.chronological, chat.conversationId);
      }
    },
    [activeChat, runTurn, runImageTurn],
  );

  // Re-run a turn's existing query and replace its result.
  const handleRetryTurn = useCallback(
    async (turnId: string) => {
      const chat = activeChat;
      if (!chat) return;
      const target = chat.turns.find((turn) => turn.id === turnId);
      if (!target) return;

      const now = new Date().toISOString();
      setChats((prev) => sortChats(
        prev.map((c) => {
          if (c.id !== chat.id) return c;
          return {
            ...c,
            updatedAt: now,
            turns: c.turns.map((turn) => (
              turn.id === turnId
                ? { ...turn, status: 'pending' as const, response: undefined, error: undefined }
                : turn
            )),
          };
        }),
      ));

      if (target.imageDataUrl) {
        await runImageTurn(chat.id, turnId, target.query, target.imageDataUrl, chat.conversationId);
      } else {
        await runTurn(chat.id, turnId, target.query, target.filters, target.chronological, chat.conversationId);
      }
    },
    [activeChat, runTurn, runImageTurn],
  );

  return (
    <div className="flex h-screen bg-transparent text-gray-100">
      <aside className="flex w-72 shrink-0 flex-col border-r border-white/10 bg-gray-950/60 backdrop-blur">
        <div className="flex items-center gap-2.5 px-4 py-4">
          <span className="inline-flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 text-white shadow-md">
            <Sparkles className="h-4 w-4" />
          </span>
          <div className="text-sm font-semibold tracking-tight text-gray-100">Lifelog</div>
        </div>

        <div className="px-3 pb-3">
          <button
            type="button"
            onClick={handleNewConversation}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 px-3 py-2 text-sm font-medium text-white shadow-md transition hover:from-indigo-400 hover:to-violet-400"
          >
            <Plus className="h-4 w-4" />
            New chat
          </button>
        </div>

        <div className="flex items-center gap-2 px-4 pb-1 pt-2 text-[11px] font-semibold uppercase tracking-wide text-gray-500">
          <MessageSquare className="h-3.5 w-3.5" />
          All chats
        </div>

        <div className="scrollbar-thin flex-1 space-y-1 overflow-y-auto px-2 py-2">
          {chats.length === 0 ? (
            <div className="px-3 py-3 text-sm text-gray-500">No saved chats</div>
          ) : (
            chats.map((chat) => {
              const selected = chat.id === activeChat?.id;
              const pendingCount = chat.turns.filter((turn) => turn.status === 'pending').length;
              return (
                <button
                  key={chat.id}
                  type="button"
                  onClick={() => handleSelectChat(chat.id)}
                  className={`w-full rounded-xl px-3 py-2.5 text-left transition ${
                    selected
                      ? 'bg-white/10 text-gray-100 ring-1 ring-white/10'
                      : 'text-gray-400 hover:bg-white/5 hover:text-gray-200'
                  }`}
                >
                  <div className="truncate text-sm font-medium">{chat.title}</div>
                  <div className="mt-1 flex items-center justify-between gap-2 text-[11px] text-gray-500">
                    <span>{chat.turns.length} {chat.turns.length === 1 ? 'turn' : 'turns'}</span>
                    <span className="flex items-center gap-1">
                      {pendingCount > 0 && <Clock className="h-3 w-3 text-indigo-400" />}
                      {formatChatTime(chat.updatedAt)}
                    </span>
                  </div>
                </button>
              );
            })
          )}
        </div>

        {hydrated && getToken() && (
          <div className="border-t border-white/10 p-3">
            <button
              type="button"
              onClick={() => logout()}
              className="flex w-full items-center gap-2 rounded-xl px-3 py-2 text-sm text-gray-400 transition hover:bg-white/5 hover:text-gray-200"
            >
              <LogOut className="h-4 w-4" />
              Log out
            </button>
          </div>
        )}
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex shrink-0 items-center justify-between gap-3 border-b border-white/10 bg-gray-950/40 px-6 py-3 backdrop-blur">
          <div className="min-w-0">
            <div className="truncate text-base font-semibold text-gray-100">
              {activeChat?.title ?? 'Life Log Search'}
            </div>
            {status && (
              <div className="text-xs text-gray-500">
                {status.total_chunks.toLocaleString()} chunks · {status.environment}
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            <IngestDataButton onStatusChange={setStatus} onError={setError} />
          </div>
        </header>

        <main className="scrollbar-thin flex-1 overflow-y-auto px-4 py-6">
          <div className="mx-auto w-full max-w-4xl space-y-8">
            {turns.length === 0 && !loading && (
              <div className="flex min-h-[55vh] flex-col items-center justify-center gap-4 text-center">
                <span className="inline-flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500/20 to-violet-500/20 text-indigo-300 ring-1 ring-white/10">
                  <Search className="h-7 w-7" />
                </span>
                <div>
                  <p className="text-lg font-medium text-gray-200">Search your personal history</p>
                  <p className="mx-auto mt-1 max-w-md text-sm text-gray-500">
                    Ask in natural language — photos, audio, video, email and more, answered from your own data.
                  </p>
                </div>
                <div className="flex flex-wrap items-center justify-center gap-2">
                  {EXAMPLE_PROMPTS.map((example) => (
                    <button
                      key={example}
                      type="button"
                      onClick={() => setComposerText(example)}
                      className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs text-gray-300 transition hover:bg-white/10 hover:text-gray-100"
                    >
                      {example}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {turns.map((turn) => (
              <ConversationTurnView
                key={turn.id}
                turn={turn}
                onEdit={handleEditTurn}
                onRetry={handleRetryTurn}
                loading={loading}
              />
            ))}

            {error && (
              <div className="flex items-start gap-2 rounded-xl border border-red-500/40 bg-red-950/40 px-4 py-3 text-sm text-red-200">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        </main>

        {status && <StatusBar status={status} />}

        <div className="shrink-0 border-t border-white/10 bg-gray-950/40 px-4 py-4 backdrop-blur">
          <div className="mx-auto max-w-4xl">
            <SearchBar
              onQuery={handleQuery}
              loading={loading}
              text={composerText}
              onTextChange={setComposerText}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
