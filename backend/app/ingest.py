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
    provisions = json.loads(CORPUS.read_text())["provisions"]
    # Embed the title + body together — short provisions, so one chunk each.
    texts = [f"{p['title']}. {p['content']}" for p in provisions]
    vectors = embed_documents(texts)

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA.read_text())
        await conn.execute("TRUNCATE gst_chunks RESTART IDENTITY")
        await conn.executemany(
            """
            INSERT INTO gst_chunks
              (provision_id, act, section, title, citation_ref, topics, content, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            [
                (p["id"], p["act"], p["section"], p["title"], p["citation_ref"],
                 p["topics"], p["content"], v)
                for p, v in zip(provisions, vectors)
            ],
        )
        await conn.execute("ANALYZE gst_chunks")
    print(f"✓ ingested {len(provisions)} GST provisions into pgvector")


if __name__ == "__main__":
    asyncio.run(main())
