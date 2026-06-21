'use client';

import { useEffect, useState, type FormEvent, type ReactNode } from 'react';
import { Loader2, Lock, Sparkles } from 'lucide-react';
import { getAuthStatus, getToken, login } from '@/lib/api';

// Wraps the app: renders children only once the user is authenticated (or when
// the backend has no password configured). Otherwise shows the login screen.
export default function AuthGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<'checking' | 'authed' | 'login'>('checking');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getAuthStatus()
      .then((status) => {
        if (cancelled) return;
        // No password on the server, or we already hold a token → let the app in.
        // An invalid/expired token surfaces as a 401, which flips us to login.
        setState(!status.auth_required || getToken() ? 'authed' : 'login');
      })
      .catch(() => {
        // Backend unreachable — load the app so it can show its own error state.
        if (!cancelled) setState('authed');
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const onUnauthorized = () => {
      setState('login');
      setPassword('');
    };
    window.addEventListener('lifelog:unauthorized', onUnauthorized);
    return () => window.removeEventListener('lifelog:unauthorized', onUnauthorized);
  }, []);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!password.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await login(password);
      setPassword('');
      setState('authed');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setSubmitting(false);
    }
  };

  if (state === 'authed') return <>{children}</>;

  if (state === 'checking') {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-950 text-gray-400">
        <Loader2 className="h-6 w-6 animate-spin" />
      </div>
    );
  }

  return (
    <div className="flex h-screen items-center justify-center bg-gray-950 px-4 text-gray-100">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-sm space-y-5 rounded-2xl border border-white/10 bg-white/[0.03] p-6 shadow-2xl backdrop-blur"
      >
        <div className="flex flex-col items-center gap-2 text-center">
          <span className="inline-flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 text-white shadow-md">
            <Sparkles className="h-5 w-5" />
          </span>
          <div className="text-lg font-semibold tracking-tight">Lifelog</div>
          <div className="text-sm text-gray-400">Enter your password to continue</div>
        </div>

        <label className="block space-y-1.5">
          <span className="text-xs font-medium text-gray-400">Password</span>
          <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 focus-within:ring-2 focus-within:ring-indigo-500/50">
            <Lock className="h-4 w-4 shrink-0 text-gray-500" />
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoFocus
              autoComplete="current-password"
              className="w-full bg-transparent py-2.5 text-sm text-gray-100 placeholder-gray-600 focus:outline-none"
              placeholder="••••••••"
            />
          </div>
        </label>

        {error && <div className="text-sm text-red-400">{error}</div>}

        <button
          type="submit"
          disabled={submitting || !password.trim()}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 px-4 py-2.5 text-sm font-medium text-white shadow-md transition hover:from-indigo-400 hover:to-violet-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Lock className="h-4 w-4" />}
          Log in
        </button>
      </form>
    </div>
  );
}
