# Deploying LedgerLens on Vercel

LedgerLens deploys as **two Vercel projects from this one repo** (a monorepo), plus
one **Postgres database with the `pgvector` extension**. Both the Next.js frontend
and the FastAPI backend run on Vercel — the backend as a Python function on Fluid
Compute, which supports the app's SSE streaming and its durable, Postgres-backed
human-in-the-loop.

```
repo ─┬─ frontend/   → Vercel project "ledgerlens"          (Next.js)
      └─ backend/    → Vercel project "ledgerlens-backend"  (Python / FastAPI)
                       ↳ needs a pgvector Postgres (Vercel Marketplace)
```

## 0. Provision the database (once)

Add a **Postgres with `pgvector`** from the **Vercel Marketplace** (several
providers on the Marketplace support the `vector` extension). Copy its connection
string — prefer the **pooled / serverless** endpoint if the provider offers one, so
many function instances don't exhaust the connection limit. This is `DATABASE_URL`.

## 1. Backend project (root directory = `backend/`)

- **New Project → import this repo → set Root Directory to `backend`.** Framework
  preset: *Other* (the included `vercel.json` wires the Python function).
- `requirements.txt`, `api/index.py`, and `vercel.json` are already in `backend/`.
  The rewrite sends every request to the FastAPI app, whose routes (`/chat`,
  `/resume`, `/health`) receive the original path.
- **Environment variables:**
  | var | value |
  |-----|-------|
  | `DATABASE_URL` | the pgvector Postgres DSN from step 0 |
  | `ANTHROPIC_API_KEY` | your Anthropic key *(you add this — needed for classify/answer; retrieval works without it)* |
  | `LLM_MODEL` | `claude-sonnet-5` (default) |
  | `FASTEMBED_CACHE_DIR` | `/tmp/fastembed` **(required on Vercel — the FS is read-only except `/tmp`, so the embedding model can only download there)** |
  | `DB_POOL_MAX` | `4` (keep small on serverless; optional) |
- Deploy. Note the backend URL, e.g. `https://ledgerlens-backend.vercel.app`.
- **Health check:** `GET /health` → `{"status":"ok"}`.

## 2. Seed the corpus (once, after the DB exists)

The corpus is embedded into pgvector by a **one-time** job — it must NOT run on
every serverless cold start, so it isn't in the request path. Run it once against
the remote DB (locally or from CI):

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
DATABASE_URL='<the pgvector DSN>' python -m app.ingest
```

`app.ingest` is idempotent: it `CREATE EXTENSION IF NOT EXISTS vector`, applies
`schema.sql`, truncates, and re-inserts the 25 provisions with embeddings. Re-run
any time the corpus changes.

## 3. Frontend project (root directory = `frontend/`)

- **New Project → same repo → Root Directory `frontend`.** Framework preset:
  *Next.js* (auto-detected).
- **Environment variable (Production + Preview):**
  | var | value |
  |-----|-------|
  | `NEXT_PUBLIC_API_URL` | the backend URL from step 1, **no trailing slash** |
- ⚠ `NEXT_PUBLIC_*` is inlined at **build time**, so set it *before* the first
  build; changing it later needs a redeploy.
- Deploy. Open the frontend URL and ask a GST question.

## Notes / gotchas

- **First request is slow.** On a cold start the backend downloads the ~130 MB
  `bge-small` embedding model into `/tmp/fastembed` before it can answer. Subsequent
  requests on a warm instance are fast. (To eliminate it, pre-bundle the model into
  the deployment — a later optimization.)
- **Connections.** Use the DB's pooled DSN and keep `DB_POOL_MAX` small; serverless
  fans out into many instances.
- **CORS** is already open (`allow_origins=["*"]`) so the cross-origin
  frontend→backend calls work.
- **Durable HITL** needs a reachable Postgres; if `DATABASE_URL` is unset/unreachable
  the backend silently falls back to an in-memory checkpointer (interrupts won't
  survive a cold start) — so verify `/health` and a real `/chat` after deploy.
- **Package size** (fastembed + onnxruntime + langgraph) is well under Vercel's 5 GB
  function limit.
