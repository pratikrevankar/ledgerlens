-- LedgerLens — pgvector schema for the GST grounding corpus.
-- Mirrors the production shape (a chunks table with both a dense embedding and a
-- full-text vector) so retrieval can be HYBRID: dense (semantic) + sparse (keyword).
-- Embedding dim = 384 (BAAI/bge-small-en-v1.5, run locally — no API key needed).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS gst_chunks (
  id            BIGSERIAL PRIMARY KEY,
  provision_id  TEXT        NOT NULL,          -- e.g. 'cgst-s16'
  act           TEXT        NOT NULL,          -- 'CGST Act 2017'
  section       TEXT        NOT NULL,          -- '16'
  title         TEXT        NOT NULL,
  citation_ref  TEXT        NOT NULL,          -- 'CGST Act, s.16'  → shown as the citation
  topics        TEXT[]      NOT NULL DEFAULT '{}',
  content       TEXT        NOT NULL,
  embedding     vector(384),                   -- dense semantic vector
  content_tsv   tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(title,'') || ' ' || coalesce(content,''))) STORED,
  created_at    timestamptz NOT NULL DEFAULT now()
);

-- Dense ANN index (cosine). ivfflat is plenty for a small demo corpus.
CREATE INDEX IF NOT EXISTS gst_chunks_embedding_idx
  ON gst_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);

-- Sparse keyword index.
CREATE INDEX IF NOT EXISTS gst_chunks_tsv_idx
  ON gst_chunks USING gin (content_tsv);
