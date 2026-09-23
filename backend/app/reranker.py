"""Cross-encoder reranker over the hybrid-retrieval shortlist.

Bi-encoder retrieval (dense pgvector) embeds the query and each document
*independently*, so it can't model fine query↔document interaction — it's fast
and recall-friendly but its top-of-list ordering is coarse. A cross-encoder
scores the (query, document) PAIR jointly in one forward pass, which reorders a
shortlist far more precisely. The standard production pattern, used here:

    hybrid retrieval casts a wide cheap net (top ~N by RRF)
        → the cross-encoder re-scores those N jointly
            → keep the best top_k

Kept LOCAL + keyless via fastembed (same rationale as the bge-small embedder), so
the whole retrieval stack still runs with no API key. Default model is the small,
fast ms-marco MiniLM; set RERANK_MODEL=BAAI/bge-reranker-base for a stronger (and
heavier) cross-encoder. The measured recall@k / MRR lift, rerank off vs on, is in
evals/run_evals.py.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable, List, Optional, Sequence, TypeVar

RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")


def rerank_enabled() -> bool:
    """On by default; RERANK_ENABLED=0/false turns it off (used by the eval A/B)."""
    return os.getenv("RERANK_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")


@lru_cache(maxsize=1)
def _encoder():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    # Read-only serverless FS → the model can only download to /tmp (see embeddings.py).
    cache_dir = os.getenv("FASTEMBED_CACHE_DIR") or None
    return TextCrossEncoder(model_name=RERANK_MODEL, cache_dir=cache_dir)


def score_pairs(query: str, docs: Sequence[str]) -> List[float]:
    """Cross-encoder relevance score per doc (higher = more relevant)."""
    if not docs:
        return []
    return [float(s) for s in _encoder().rerank(query, list(docs))]


T = TypeVar("T")


def rerank(
    query: str,
    candidates: Sequence[T],
    text_of: Callable[[T], str],
    top_k: int,
    scorer: Optional[Callable[[str, Sequence[str]], List[float]]] = None,
) -> List[T]:
    """Reorder `candidates` by cross-encoder score against `query`, keep top_k.

    `scorer` is injectable so the ordering logic is unit-testable without loading a
    model; it defaults to the fastembed cross-encoder. Reranking is an enhancement,
    never a hard dependency — if the model can't load/score, we degrade gracefully
    to the input order (which is already the fused hybrid ranking)."""
    if not candidates:
        return []
    score = scorer or score_pairs
    try:
        scores = score(query, [text_of(c) for c in candidates])
        if len(scores) != len(candidates):
            return list(candidates)[:top_k]
        ranked = [c for c, _ in sorted(
            zip(candidates, scores), key=lambda z: z[1], reverse=True)]
        return ranked[:top_k]
    except Exception:  # noqa: BLE001 — never let a reranker failure break retrieval
        return list(candidates)[:top_k]
