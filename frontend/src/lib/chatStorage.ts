import type { ChatRecord } from './types';

const STORAGE_KEY = 'lifelog.chats.v1';
const MAX_CHATS = 100;

function newId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `chat_${Date.now()}_${Math.random().toString(36).slice(2)}`;
}

export function createChat(now = new Date().toISOString()): ChatRecord {
  return {
    id: newId(),
    title: 'New chat',
    createdAt: now,
    updatedAt: now,
    turns: [],
  };
}

export function loadChats(): ChatRecord[] {
  if (typeof window === 'undefined') return [];

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((chat): chat is ChatRecord => {
        return (
          chat &&
          typeof chat.id === 'string' &&
          typeof chat.title === 'string' &&
          typeof chat.createdAt === 'string' &&
          typeof chat.updatedAt === 'string' &&
          Array.isArray(chat.turns)
        );
      })
      .slice(0, MAX_CHATS);
  } catch {
    return [];
  }
}

export function saveChats(chats: ChatRecord[]): void {
  if (typeof window === 'undefined') return;
  const ordered = [...chats]
    .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
    .slice(0, MAX_CHATS);
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(ordered));
}

export function chatTitleFromQuery(query: string): string {
  const compact = query.trim().replace(/\s+/g, ' ');
  if (!compact) return 'New chat';
  return compact.length > 48 ? `${compact.slice(0, 45)}...` : compact;
}
