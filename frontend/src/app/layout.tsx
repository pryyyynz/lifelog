import type { Metadata } from 'next';
import './globals.css';
import AuthGate from '@/components/AuthGate';

export const metadata: Metadata = {
  title: 'Life Log Search',
  description: 'Local personal search engine',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <AuthGate>{children}</AuthGate>
      </body>
    </html>
  );
}
