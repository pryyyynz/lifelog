'use client';

import { useEffect } from 'react';
import { createPortal } from 'react-dom';
import { ExternalLink, X } from 'lucide-react';

interface LightboxProps {
  type: 'image' | 'video';
  url: string;
  alt?: string;
  onClose: () => void;
  onOpenOriginal?: () => void;
}

// Full-screen overlay for taking a closer look at an image or video result.
export default function Lightbox({ type, url, alt, onClose, onOpenOriginal }: LightboxProps) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = previousOverflow;
    };
  }, [onClose]);

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div className="absolute right-4 top-4 flex items-center gap-2">
        {onOpenOriginal && (
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              onOpenOriginal();
            }}
            title="Open original file"
            className="rounded-full bg-white/10 p-2 text-gray-200 transition hover:bg-white/20"
          >
            <ExternalLink className="h-5 w-5" />
          </button>
        )}
        <button
          type="button"
          onClick={onClose}
          title="Close"
          className="rounded-full bg-white/10 p-2 text-gray-200 transition hover:bg-white/20"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      <div className="max-h-full max-w-6xl" onClick={(event) => event.stopPropagation()}>
        {type === 'image' ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={url} alt={alt ?? 'preview'} className="max-h-[88vh] max-w-full rounded-xl object-contain shadow-2xl" />
        ) : (
          <video src={url} controls autoPlay className="max-h-[88vh] max-w-full rounded-xl shadow-2xl" />
        )}
      </div>
    </div>,
    document.body,
  );
}
