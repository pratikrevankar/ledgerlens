import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'LedgerLens — GST compliance agent',
  description: 'Citation-grounded Indian-GST agent. FastAPI + LangGraph + pgvector.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // suppressHydrationWarning: browser extensions (password managers, Grammarly…)
  // inject attributes into <html>/<body> before React hydrates, which can otherwise
  // surface as a "client-side exception" on load. This makes hydration resilient.
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
