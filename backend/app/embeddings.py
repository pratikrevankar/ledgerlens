"""Local dense embeddings — BAAI/bge-small-en-v1.5 (384-dim) via fastembed.

Runs on CPU in-process with no API key, so `docker compose up` gives working
semantic retrieval out of the box. bge models want a short instruction prefixed
to the QUERY (not the documents) for best retrieval quality.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    # Downloaded once and cached in the image/volume.
    return TextEmbedding(model_name=MODEL_NAME)


def embed_documents(texts: List[str]) -> List[List[float]]:
    return [list(map(float, v)) for v in _model().embed(list(texts))]


def embed_query(text: str) -> List[float]:
    vec = next(_model().embed([_QUERY_INSTRUCTION + text]))
    return list(map(float, vec))
