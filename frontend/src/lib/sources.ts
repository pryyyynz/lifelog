import {
  Calendar,
  FileText,
  Globe,
  Image as ImageIcon,
  Mail,
  Music,
  Video,
  type LucideIcon,
} from 'lucide-react';

// Shared per-source-type presentation (icon + colour) so the status bar, result
// cards, and hits all read from one place.
export interface SourceMeta {
  label: string;
  Icon: LucideIcon;
  text: string;
  bg: string;
  dot: string;
}

export const SOURCE_META: Record<string, SourceMeta> = {
  text: { label: 'Text', Icon: FileText, text: 'text-sky-300', bg: 'bg-sky-500/10', dot: 'bg-sky-400' },
  email: { label: 'Email', Icon: Mail, text: 'text-violet-300', bg: 'bg-violet-500/10', dot: 'bg-violet-400' },
  photo: { label: 'Photo', Icon: ImageIcon, text: 'text-pink-300', bg: 'bg-pink-500/10', dot: 'bg-pink-400' },
  audio: { label: 'Audio', Icon: Music, text: 'text-amber-300', bg: 'bg-amber-500/10', dot: 'bg-amber-400' },
  video: { label: 'Video', Icon: Video, text: 'text-orange-300', bg: 'bg-orange-500/10', dot: 'bg-orange-400' },
  calendar: { label: 'Calendar', Icon: Calendar, text: 'text-emerald-300', bg: 'bg-emerald-500/10', dot: 'bg-emerald-400' },
  browser_history: { label: 'Browser', Icon: Globe, text: 'text-teal-300', bg: 'bg-teal-500/10', dot: 'bg-teal-400' },
};

export function sourceMeta(type: string): SourceMeta {
  return (
    SOURCE_META[type] ?? {
      label: type,
      Icon: FileText,
      text: 'text-gray-300',
      bg: 'bg-gray-500/10',
      dot: 'bg-gray-400',
    }
  );
}
