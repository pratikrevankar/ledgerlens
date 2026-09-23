"""Local dense embeddings — BAAI/bge-small-en-v1.5 (384-dim) via fastembed.

Runs on CPU in-process with no API key, so `docker compose up` gives working
semantic retrieval out of the box. bge models want a short instruction prefixed
to the QUERY (not the documents) for best retrieval quality.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    # Downloaded once and cached. On a normal box fastembed's default cache dir is
    # fine; on a read-only serverless filesystem (Vercel) only /tmp is writable, so
    # FASTEMBED_CACHE_DIR must point there (e.g. /tmp/fastembed) or the download
    # fails at import time. Passing cache_dir=None keeps the default locally.
    cache_dir = os.getenv("FASTEMBED_CACHE_DIR") or None
    return TextEmbedding(model_name=MODEL_NAME, cache_dir=cache_dir)


def embed_documents(texts: List[str]) -> List[List[float]]:
    return [list(map(float, v)) for v in _model().embed(list(texts))]


def embed_query(text: str) -> List[float]:
    vec = next(_model().embed([_QUERY_INSTRUCTION + text]))
    return list(map(float, vec))
