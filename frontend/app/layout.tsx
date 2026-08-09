import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'LedgerLens — GST compliance agent',
  description: 'Citation-grounded Indian-GST agent. FastAPI + LangGraph + pgvector.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
