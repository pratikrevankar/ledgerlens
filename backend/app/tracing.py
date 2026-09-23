"""LangSmith tracing wiring.

Tracing itself is automatic once the env is set — LangChain/LangGraph emit every
node, tool and LLM call as a nested run. This module adds the two things that make
those traces actually useful:

  1. `configure()` normalises the newer LANGSMITH_* vars to the LANGCHAIN_* names
     older SDKs read, so tracing turns on regardless of version (current
     langchain-core reads LANGSMITH_* natively; this is belt-and-suspenders).
  2. `run_config()` attaches a run name + tags + metadata to each graph invocation,
     so in LangSmith you can filter traces by intent and by which retrieval levers
     were on (rerank / contextual retrieval / prompt cache) — the difference
     between "a pile of runs" and "I can find the bad answer and see why."
"""
from __future__ import annotations

import os
from typing import Optional


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def configure() -> bool:
    """Mirror LANGSMITH_* → LANGCHAIN_* when tracing is on. Idempotent; returns
    whether tracing is enabled (no-op + False when LANGSMITH_TRACING is off)."""
    if not _truthy(os.getenv("LANGSMITH_TRACING", "")):
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_API_KEY", os.environ["LANGSMITH_API_KEY"])
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "ledgerlens"))
    if os.getenv("LANGSMITH_ENDPOINT"):
        os.environ.setdefault("LANGCHAIN_ENDPOINT", os.environ["LANGSMITH_ENDPOINT"])
    return True


def run_config(thread_id: str, extra: Optional[dict] = None) -> dict:
    """A LangGraph config with tracing-friendly run name, tags and metadata.

    Keeps `configurable.thread_id` so durable checkpointing / resume still work;
    the run name, tags and metadata are what LangSmith indexes for filtering."""
    from .contextualize import mode as ctx_mode
    from .prompt_cache import enabled as cache_on
    from .reranker import rerank_enabled

    md = {
        "thread_id": thread_id,
        "rerank": rerank_enabled(),
        "contextual_retrieval": ctx_mode(),
        "prompt_cache": cache_on(),
    }
    if extra:
        md.update(extra)
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "ledgerlens-chat",
        "tags": [
            "ledgerlens",
            f"rerank:{str(md['rerank']).lower()}",
            f"ctx:{md['contextual_retrieval']}",
        ],
        "metadata": md,
    }
