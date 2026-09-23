"""Contextual Retrieval (Anthropic's technique).

Prepend a short situating context to each provision BEFORE embedding it, so the
dense vector — and the sparse keyword index — carry document-level context the
bare chunk text lacks. On fragmented chunks from large documents this sharply cuts
retrieval failures; on this small, already-self-contained GST corpus the lift is
modest (near the recall ceiling), so it's here to demonstrate the technique and to
be measured, not to fake a big number.

Modes via CONTEXTUAL_RETRIEVAL:
  • off       (default) — embed title + content, as before.
  • template  — deterministic, KEYLESS: situate the chunk from its own metadata.
                Measurable with no API key (see evals/run_evals ingest A/B).
  • llm       — Anthropic-style: the model writes the one-sentence context. Needs a
                key; the corpus overview is prompt-CACHED across all provisions.
"""
from __future__ import annotations

import os
from typing import Optional

from .prompt_cache import text_block

CORPUS_OVERVIEW = (
    "This corpus is the core of Indian GST law — the CGST Act 2017, the IGST Act "
    "2017, and the e-invoicing / e-way-bill rules — one statutory provision per chunk."
)


def mode() -> str:
    return os.getenv("CONTEXTUAL_RETRIEVAL", "off").strip().lower()


def template_context(p: dict) -> str:
    """Deterministic, keyless situating context from the provision's metadata."""
    topics = ", ".join(p.get("topics", []))
    base = (f"In Indian GST, {p['act']} section {p['section']} ({p['citation_ref']}) "
            f"concerns {p['title'].lower()}")
    return base + (f"; topics: {topics}." if topics else ".")


def _context_llm():
    from langchain_anthropic import ChatAnthropic
    model = os.getenv("CONTEXT_MODEL") or os.getenv("LLM_MODEL", "claude-sonnet-5")
    return ChatAnthropic(model=model, max_tokens=200)


def _as_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content
                       if isinstance(b, dict) and b.get("type") == "text")
    return str(content)


async def llm_context(p: dict, llm=None) -> str:
    """Anthropic-style: ask the model for a one-sentence search context. The stable
    system + corpus overview is prompt-cached, so it's paid for once across all 25
    provisions rather than re-sent each call."""
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = llm or _context_llm()
    system = [text_block(
        "You situate a single legal provision within the whole GST corpus so a "
        "search engine can retrieve it for the right question. Reply with ONE short "
        f"sentence of context — no preamble.\n\nCorpus: {CORPUS_OVERVIEW}")]
    msg = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Provision {p['citation_ref']} — {p['title']}:\n{p['content']}\n\n"
                             "One-sentence search context:"),
    ])
    return _as_text(msg.content).strip()


async def build_context(p: dict, m: Optional[str] = None, llm=None) -> str:
    """Return the situating context for a provision under the active (or given) mode."""
    m = (m or mode())
    if m == "template":
        return template_context(p)
    if m == "llm":
        return await llm_context(p, llm=llm)
    return ""  # off
