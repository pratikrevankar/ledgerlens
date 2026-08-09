"""Hybrid retrieval over the GST corpus.

Two independent searches — dense (pgvector cosine) and sparse (Postgres
full-text ts_rank) — fused with Reciprocal Rank Fusion. Hybrid beats either
alone: dense catches paraphrase ("what tax on out-of-state sale" → IGST /
place-of-supply), sparse nails exact statutory terms ("section 16", "LUT",
"ITC") that embeddings sometimes blur.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .db import get_pool
from .embeddings import embed_query

RRF_K = 60  # standard RRF damping constant


@dataclass
class Chunk:
    provision_id: str
    act: str
    section: str
    title: str
    citation_ref: str
    content: str
    score: float

    def citation(self) -> str:
        return self.citation_ref


async def _dense(query: str, limit: int) -> List[str]:
    vec = embed_query(query)
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT provision_id
        FROM gst_chunks
        ORDER BY embedding <=> $1
        LIMIT $2
        """,
        vec, limit,
    )
    return [r["provision_id"] for r in rows]


async def _sparse(query: str, limit: int) -> List[str]:
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT provision_id
        FROM gst_chunks
        WHERE content_tsv @@ plainto_tsquery('english', $1)
        ORDER BY ts_rank(content_tsv, plainto_tsquery('english', $1)) DESC
        LIMIT $2
        """,
        query, limit,
    )
    return [r["provision_id"] for r in rows]


def _rrf(*ranked_lists: List[str]) -> List[str]:
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, pid in enumerate(lst):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(scores, key=lambda p: scores[p], reverse=True)


async def retrieve(query: str, top_k: int = 4, pool_size: int = 10) -> List[Chunk]:
    """Return the top_k most relevant provisions for a query, fused + hydrated."""
    dense, sparse = await _dense(query, pool_size), await _sparse(query, pool_size)
    order = _rrf(dense, sparse)[:top_k]
    if not order:
        return []
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT provision_id, act, section, title, citation_ref, content
        FROM gst_chunks WHERE provision_id = ANY($1)
        """,
        order,
    )
    by_id = {r["provision_id"]: r for r in rows}
    out: List[Chunk] = []
    for rank, pid in enumerate(order):
        r = by_id.get(pid)
        if r:
            out.append(Chunk(
                provision_id=r["provision_id"], act=r["act"], section=r["section"],
                title=r["title"], citation_ref=r["citation_ref"], content=r["content"],
                score=1.0 / (rank + 1),
            ))
    return out
