'use client';

import { useRef, useState } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type Msg = { role: 'user' | 'assistant'; content: string; citations?: string[] };

const NODE_LABEL: Record<string, string> = {
  classify: 'Classifying intent…',
  retrieve: 'Retrieving provisions…',
  compute: 'Computing GST…',
  confirm: 'Awaiting your approval…',
  answer: 'Writing the answer…',
};

const SAMPLES = [
  'GST on a ₹50,000 inter-state consulting sale from Karnataka to Maharashtra?',
  'Under what conditions can I claim input tax credit?',
  'What is the GST registration turnover threshold?',
  'Record a ₹50,000 sale to Maharashtra',
];

/** Read an SSE POST response and dispatch each `event:`/`data:` frame. */
async function readSSE(res: Response, on: (event: string, data: any) => void) {
  const reader = res.body!.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const frames = buf.split('\n\n');
    buf = frames.pop() || '';
    for (const f of frames) {
      const ev = /event: (.*)/.exec(f)?.[1]?.trim();
      const dm = /data: (.*)/.exec(f)?.[1];
      if (ev && dm) on(ev, JSON.parse(dm));
    }
  }
}

export default function Home() {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [node, setNode] = useState<string | null>(null);
  const [pending, setPending] = useState<{ threadId: string; summary: string } | null>(null);
  const scroller = useRef<HTMLDivElement>(null);

  const patchLast = (fn: (m: Msg) => Msg) =>
    setMsgs((cur) => cur.map((m, i) => (i === cur.length - 1 ? fn(m) : m)));

  function handlers(threadId: string) {
    return (ev: string, data: any) => {
      if (ev === 'node') setNode(data.node);
      else if (ev === 'citations') patchLast((m) => ({ ...m, citations: data.citations }));
      else if (ev === 'token') patchLast((m) => ({ ...m, content: m.content + data.text }));
      else if (ev === 'interrupt') setPending({ threadId, summary: data.summary });
      else if (ev === 'done') {
        patchLast((m) => ({ content: data.answer || m.content, citations: data.citations || m.citations, role: 'assistant' }));
        setNode(null);
      }
      requestAnimationFrame(() => scroller.current?.scrollTo({ top: 1e9, behavior: 'smooth' }));
    };
  }

  async function send(q: string) {
    if (!q.trim() || busy) return;
    setBusy(true);
    setInput('');
    const threadId = crypto.randomUUID();
    setMsgs((c) => [...c, { role: 'user', content: q }, { role: 'assistant', content: '' }]);
    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ query: q, thread_id: threadId }),
      });
      await readSSE(res, handlers(threadId));
    } catch {
      patchLast((m) => ({ ...m, content: 'Could not reach the backend. Is it running on :8000?' }));
    } finally {
      setBusy(false); setNode(null);
    }
  }

  async function decide(approved: boolean) {
    if (!pending) return;
    const { threadId } = pending;
    setPending(null); setBusy(true);
    try {
      const res = await fetch(`${API}/resume`, {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ thread_id: threadId, approved }),
      });
      await readSSE(res, handlers(threadId));
    } finally {
      setBusy(false); setNode(null);
    }
  }

  return (
    <main style={{ maxWidth: 760, margin: '0 auto', padding: '28px 20px', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header style={{ marginBottom: 18 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 30, height: 30, borderRadius: 8, background: 'linear-gradient(135deg,#4f8cff,#34d399)' }} />
          <h1 style={{ fontSize: 20, fontWeight: 800, margin: 0, letterSpacing: '-0.02em' }}>LedgerLens</h1>
        </div>
        <p style={{ color: 'var(--muted)', fontSize: 13.5, margin: '8px 0 0' }}>
          A citation-grounded GST agent — hybrid RAG + tool-use + human-in-the-loop, streamed from FastAPI/LangGraph.
        </p>
      </header>

      <div ref={scroller} style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 14, paddingBottom: 8 }}>
        {msgs.length === 0 && (
          <div style={{ display: 'grid', gap: 8, marginTop: 8 }}>
            {SAMPLES.map((s) => (
              <button key={s} onClick={() => send(s)}
                style={{ textAlign: 'left', background: 'var(--panel)', border: '1px solid var(--border)', color: 'var(--text)', borderRadius: 10, padding: '11px 14px', cursor: 'pointer', fontSize: 13.5 }}>
                {s}
              </button>
            ))}
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start' }}>
            <div style={{
              maxWidth: '86%', borderRadius: 12, padding: '11px 14px', fontSize: 14.5, whiteSpace: 'pre-wrap',
              background: m.role === 'user' ? 'var(--accent-dim)' : 'var(--panel)',
              border: `1px solid ${m.role === 'user' ? '#274064' : 'var(--border)'}`,
            }}>
              {m.content || (busy && i === msgs.length - 1 ? <span style={{ color: 'var(--muted)' }}>▍</span> : '')}
              {m.citations && m.citations.length > 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 10 }}>
                  {m.citations.map((c) => (
                    <span key={c} style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--green)', background: '#10241d', border: '1px solid #1c3a2e', borderRadius: 999, padding: '2px 9px' }}>
                      {c}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {node && (
          <div style={{ color: 'var(--muted)', fontSize: 12.5, fontFamily: 'var(--mono)', display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ width: 6, height: 6, borderRadius: 99, background: 'var(--accent)' }} /> {NODE_LABEL[node] ?? node}
          </div>
        )}

        {pending && (
          <div style={{ background: '#241f10', border: '1px solid #4a3d15', borderRadius: 12, padding: 14 }}>
            <div style={{ fontSize: 13.5, color: 'var(--amber)', fontWeight: 600, marginBottom: 4 }}>⏸ Human-in-the-loop</div>
            <div style={{ fontSize: 13.5, marginBottom: 12 }}>{pending.summary}</div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button onClick={() => decide(true)} style={{ background: 'var(--green)', color: '#04120c', border: 0, borderRadius: 8, padding: '7px 16px', fontWeight: 700, cursor: 'pointer' }}>Approve</button>
              <button onClick={() => decide(false)} style={{ background: 'transparent', color: 'var(--muted)', border: '1px solid var(--border)', borderRadius: 8, padding: '7px 16px', cursor: 'pointer' }}>Reject</button>
            </div>
          </div>
        )}
      </div>

      <form onSubmit={(e) => { e.preventDefault(); send(input); }}
        style={{ display: 'flex', gap: 8, marginTop: 12, borderTop: '1px solid var(--border)', paddingTop: 12 }}>
        <textarea value={input} onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input); } }}
          rows={1} placeholder="Ask a GST question…" disabled={busy}
          style={{ flex: 1, resize: 'none', background: 'var(--panel-2)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 10, padding: '11px 13px', fontSize: 14.5 }} />
        <button type="submit" disabled={busy || !input.trim()}
          style={{ background: 'var(--accent)', color: '#fff', border: 0, borderRadius: 10, padding: '0 18px', fontWeight: 700, cursor: busy ? 'not-allowed' : 'pointer', opacity: busy || !input.trim() ? 0.5 : 1 }}>
          Send
        </button>
      </form>
    </main>
  );
}
