"""Hybrid retrieval over the GST corpus.

Two independent searches — dense (pgvector cosine) and sparse (Postgres
full-text ts_rank) — fused with Reciprocal Rank Fusion. Hybrid beats either
alone: dense catches paraphrase ("what tax on out-of-state sale" → IGST /
place-of-supply), sparse nails exact statutory terms ("section 16", "LUT",
"ITC") that embeddings sometimes blur.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .db import get_pool
from .embeddings import embed_query
from .reranker import rerank, rerank_enabled

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


async def retrieve(
    query: str,
    top_k: int = 4,
    pool_size: int = 10,
    rerank_pool: int = 12,
    use_rerank: Optional[bool] = None,
) -> List[Chunk]:
    """Return the top_k most relevant provisions for a query.

    Pipeline: dense + sparse → RRF fusion → (optional) cross-encoder rerank → top_k.
    When reranking, we hydrate a WIDER shortlist (`rerank_pool`) because the
    cross-encoder needs each candidate's text to re-score it; the reranker then
    cuts back to top_k. `use_rerank` defaults to the RERANK_ENABLED env flag — the
    eval harness flips it to measure recall@k / MRR with rerank off vs on.
    """
    do_rerank = rerank_enabled() if use_rerank is None else use_rerank
    dense, sparse = await _dense(query, pool_size), await _sparse(query, pool_size)
    order = _rrf(dense, sparse)
    if not order:
        return []
    # Hydrate a shortlist: just top_k without rerank; a wider pool with it.
    shortlist = order[: (rerank_pool if do_rerank else top_k)]
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT provision_id, act, section, title, citation_ref, content
        FROM gst_chunks WHERE provision_id = ANY($1)
        """,
        shortlist,
    )
    by_id = {r["provision_id"]: r for r in rows}
    candidates: List[Chunk] = []
    for rank, pid in enumerate(shortlist):
        r = by_id.get(pid)
        if r:
            candidates.append(Chunk(
                provision_id=r["provision_id"], act=r["act"], section=r["section"],
                title=r["title"], citation_ref=r["citation_ref"], content=r["content"],
                score=1.0 / (rank + 1),  # fusion rank; re-stamped below after rerank
            ))

    if do_rerank:
        candidates = rerank(
            query, candidates,
            text_of=lambda c: f"{c.title}. {c.content}",
            top_k=top_k,
        )
    else:
        candidates = candidates[:top_k]

    # Re-stamp score to reflect FINAL rank (post-rerank), so the UI/eval see the
    # order the agent actually used.
    for i, c in enumerate(candidates):
        c.score = 1.0 / (i + 1)
    return candidates
