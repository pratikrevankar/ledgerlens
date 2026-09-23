"""Chunk + embed the GST corpus into pgvector. Idempotent: re-run any time.

    python -m app.ingest
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .db import get_pool
from .embeddings import embed_documents

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "gst_provisions.json"
SCHEMA = Path(__file__).resolve().parent / "schema.sql"


async def main() -> None:
    from .contextualize import build_context, mode

    provisions = json.loads(CORPUS.read_text())["provisions"]

    # Contextual Retrieval: prepend a situating context to each provision before
    # embedding, so the vector + keyword index carry document-level context. `off`
    # reproduces the original title+content embedding.
    ctx_mode = mode()
    contexts = [await build_context(p, ctx_mode) for p in provisions]

    # Embed context + title + body together — short provisions, so one chunk each.
    texts = [
        (f"{ctx}\n{p['title']}. {p['content']}" if ctx else f"{p['title']}. {p['content']}")
        for p, ctx in zip(provisions, contexts)
    ]
    vectors = embed_documents(texts)

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA.read_text())
        # Backfill the column on a table created before Contextual Retrieval existed.
        await conn.execute("ALTER TABLE gst_chunks ADD COLUMN IF NOT EXISTS context TEXT NOT NULL DEFAULT ''")
        await conn.execute("TRUNCATE gst_chunks RESTART IDENTITY")
        await conn.executemany(
            """
            INSERT INTO gst_chunks
              (provision_id, act, section, title, citation_ref, topics, content, context, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            [
                (p["id"], p["act"], p["section"], p["title"], p["citation_ref"],
                 p["topics"], p["content"], ctx, v)
                for p, ctx, v in zip(provisions, contexts, vectors)
            ],
        )
        await conn.execute("ANALYZE gst_chunks")
    tag = f" (contextual: {ctx_mode})" if ctx_mode != "off" else ""
    print(f"✓ ingested {len(provisions)} GST provisions into pgvector{tag}")


if __name__ == "__main__":
    asyncio.run(main())
