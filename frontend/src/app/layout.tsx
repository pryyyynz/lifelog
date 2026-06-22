import type { Metadata, Viewport } from 'next';
import './globals.css';
import AuthGate from '@/components/AuthGate';
import PWARegister from '@/components/PWARegister';

export const metadata: Metadata = {
  title: 'Life Log Search',
  description: 'Local personal search engine',
  manifest: '/manifest.webmanifest',
  appleWebApp: { capable: true, statusBarStyle: 'black-translucent', title: 'Lifelog' },
  icons: { icon: '/icon-192.png', apple: '/icon-192.png' },
};

export const viewport: Viewport = {
  themeColor: '#0a0a0f',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <PWARegister />
        <AuthGate>{children}</AuthGate>
      </body>
    </html>
  );
}
