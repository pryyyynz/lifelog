import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Life Log Search',
  description: 'Local personal search engine',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
