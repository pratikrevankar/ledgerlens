"""Reflexion critic — verify the drafted answer is grounded before returning it.

Two checks, cheapest-first:
  1. DETERMINISTIC (no LLM): no invented citations — every "[Act, s.X]" the answer
     cites must be one of the retrieved provisions. This catches the specific
     failure that matters most in a compliance tool: a confident, fabricated
     section number.
  2. LLM entailment: every remaining factual claim (a rate, a condition) must
     follow from the retrieved context; the critic returns any that don't.

If either fails, the graph loops back to the answer node ONCE with the critique so
the model can self-correct (bounded by MAX_REVISIONS in graph.py). "Don't just
cite — verify the citation supports the claim, and rewrite if not" is the point.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from .grounding import citation_validity, is_refusal


def _critic_llm():
    from langchain_anthropic import ChatAnthropic
    model = os.getenv("CRITIC_MODEL") or os.getenv("LLM_MODEL", "claude-sonnet-5")
    return ChatAnthropic(model=model, max_tokens=1024)


class _Grounding(BaseModel):
    grounded: bool = Field(..., description="true iff every factual claim in the answer is supported by the context")
    unsupported_claims: List[str] = Field(default_factory=list)


async def verify_grounded(
    answer: str,
    context: str,
    retrieved_citations: List[str],
    llm=None,
) -> Tuple[bool, str]:
    """Return (grounded, critique). Deterministic citation check first, then LLM
    entailment. `critique` is empty when grounded, else actionable feedback for the
    answer node to self-correct against."""
    # A refusal cites nothing and asserts nothing — it is safe by construction.
    if is_refusal(answer):
        return True, ""

    # 1. Deterministic — invented citations short-circuit before we spend an LLM call.
    cv = citation_validity(answer, retrieved_citations)
    if cv["invalid"]:
        return False, (
            f"These cited sources are NOT in the retrieved context and must be "
            f"removed or replaced with a real one from the context: {cv['invalid']}."
        )

    # 2. LLM entailment — every claim must follow from the context.
    judge = (llm or _critic_llm()).with_structured_output(_Grounding)
    r: _Grounding = await judge.ainvoke(
        "Check whether EVERY factual claim in the answer is directly supported by "
        "the context. Set grounded=false if any claim — a rate, a section number, a "
        "condition — is not present in the context.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}")
    if r.grounded:
        return True, ""
    return False, (
        "Remove or correct these claims the context does not support: "
        + "; ".join(r.unsupported_claims)
        + ". Answer ONLY from the context, or say \"I don't have a source for that.\""
    )
