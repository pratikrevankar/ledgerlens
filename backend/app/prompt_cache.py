"""Anthropic prompt caching helpers.

`cache_control` marks a stable prompt PREFIX as cacheable: a later request that
shares that prefix reads it from cache instead of re-processing it — lower latency
and ~10× cheaper on the cached tokens. Two concrete wins in LedgerLens:

  • the answer node's static system prompt + retrieved context are IDENTICAL across
    the first draft and the Reflexion self-correction pass on one query, so the
    second call reads them from cache;
  • the ingest-time contextualizer reuses the corpus overview across all 25
    provisions.

Anthropic only caches a prefix above a per-model minimum (~1024 tokens), so on a
short context the marker is simply a no-op — correct and harmless either way.
Toggle with PROMPT_CACHE (default on).
"""
from __future__ import annotations

import os


def enabled() -> bool:
    return os.getenv("PROMPT_CACHE", "true").strip().lower() not in ("0", "false", "no", "off")


def text_block(text: str, cache: bool = True) -> dict:
    """A Claude content block; marked cacheable when `cache` and PROMPT_CACHE are on.

    Used as list-of-blocks message content (langchain-anthropic passes the
    cache_control through to the Anthropic API)."""
    block = {"type": "text", "text": text}
    if cache and enabled():
        block["cache_control"] = {"type": "ephemeral"}
    return block
