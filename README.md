# LedgerLens

[![CI](https://github.com/pratikrevankar/ledgerlens/actions/workflows/ci.yml/badge.svg)](https://github.com/pratikrevankar/ledgerlens/actions/workflows/ci.yml)

**A citation-grounded GST compliance agent.** Ask a real Indian-tax question in
plain English; LedgerLens retrieves the governing provision, runs the exact tax
math in code (never the model), and answers **with a citation after every claim —
or refuses if it has no source.** State-changing actions pause for human approval.

> **Honest framing:** I run the production version of this pattern in TypeScript on
> Vercel (a multi-jurisdiction financial platform with a live pgvector RAG). This
> repo is a **self-contained Python / FastAPI / LangGraph reference implementation**
> of the same agent — built to show the applied-AI stack end to end, not as a toy.

---

## What it does

> *"We sold ₹50,000 of consulting to a client in another state. What's the GST, and
> which section decides place of supply?"*

1. **Classify** — question, calculation, or a state-changing action?
2. **Retrieve** — hybrid search (dense pgvector + sparse full-text, fused with RRF)
   over 25 real GST provisions → returns the text **and its citation**.
3. **Compute** — a deterministic Python tool returns IGST ₹9,000 (inter-state,
   IGST Act s.7); place of supply = recipient's location (s.12).
4. **Human-in-the-loop** — if you say *"record this,"* the graph **interrupts** and
   waits for your approval before it would commit anything.
5. **Answer** — streamed token-by-token, **every claim cited**; unsupported → *"I
   don't have a source for that."*

## Architecture

```
Next.js UI ──SSE──► FastAPI (Python) ──► LangGraph agent ──► Postgres + pgvector
                                              │
              classify ─► retrieve ─┬─► compute ─►┐
                                    ├─► confirm ──►┤  (interrupt: human approval)
                                    └─────────────►┴─► answer  (grounded, streamed)
```

- **Retrieval is hybrid** — dense embeddings catch paraphrase, keyword catches exact
  statutory terms (`section 16`, `LUT`, `ITC`); [Reciprocal Rank Fusion](backend/app/retrieval.py) merges them.
- **Embeddings run locally** (BAAI/bge-small, 384-dim, via fastembed) — retrieval
  needs **no API key**. Only generation needs one LLM key.
- **The math is code, not the model** — [`compute_gst`](backend/app/tools.py) is pure
  and unit-tested; the LLM only decides *when* to call it and with what inputs.

## Quickstart

```bash
cp backend/.env.example backend/.env      # add your ANTHROPIC_API_KEY
docker compose up --build                 # db + backend + UI; seeds the corpus on boot
```

Open the chat at **http://localhost:3000**, or hit the streaming API directly:

```bash
curl -N -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"query":"GST on a 50000 inter-state consulting sale from Karnataka to Maharashtra?","thread_id":"t1"}'
```

You'll see SSE frames: `node` (which step is running) → `citations` → streamed
`token`s → `done`. Ask *"record a 50000 sale to Maharashtra"* and you'll get an
`interrupt` frame; approve it via `POST /resume {"thread_id":"t1","approved":true}`.

## Evals

```bash
cd backend
pytest evals/test_gst_tool.py             # the math guardrails (no infra needed)
python -m evals.run_evals                 # retrieval recall@4 over labelled cases (needs db up)
```

`run_evals.py` exits non-zero if retrieval recall drops below threshold — a CI gate
against silent RAG regressions.

## Observability

Set `LANGSMITH_TRACING=true` + a `LANGSMITH_API_KEY` in `.env` to trace every run,
node transition and tool call in LangSmith.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Agent | **LangGraph** | Explicit state machine: per-step trust boundaries + human-in-the-loop `interrupt` |
| API | **FastAPI** (async, SSE) | Token + tool-event streaming without Edge tricks |
| Vector | **Postgres + pgvector** | Real hybrid retrieval, mirrors production |
| Embeddings | **fastembed / bge-small** | Local, key-less, offline-runnable retrieval |
| LLM | **Anthropic Claude** | Classify, extract, and write the grounded answer |
| UI | **Next.js (App Router)** | Streaming chat client |

## What this demonstrates

Grounded RAG with citations · a real agent state machine · tool-use with a
deterministic calculator · human-in-the-loop approval · SSE streaming · evals as a
regression gate · one-command Docker repro. A regulated-domain agent, not a
PDF-chat demo.

*Corpus: illustrative summaries of real CGST/IGST Act 2017 provisions — verify
against the bare Act before any production use.*
