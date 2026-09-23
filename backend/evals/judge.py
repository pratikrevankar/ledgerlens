"""Generation-quality scorers — LLM-as-judge (RAGAS-style) + deterministic checks.

Three signals per answer, cheapest-first:

  • citation_validity — DETERMINISTIC, no LLM: every "[Act, s.X]" the answer cites
    must be one of the provisions actually retrieved. Catches invented citations —
    the specific failure mode that matters most in a compliance tool.
  • faithfulness      — LLM judge: is every claim entailed by the retrieved
    context? Returns the unsupported claims so a failure is inspectable.
  • answer_relevance  — LLM judge: does the answer actually address the question?

The judge runs as a SEPARATE LLM pass (JUDGE_MODEL) from the answer model, so it
isn't grading its own generation in the same call. The deterministic pieces are
pure functions — unit-tested without a key or a database (see test_reranker.py).
"""
from __future__ import annotations

import os
from typing import List

from pydantic import BaseModel, Field

# Groundedness helpers live in app.grounding so the runtime critic and this eval
# judge score citations identically. Re-exported here for existing importers.
from app.grounding import (  # noqa: F401
    REFUSAL, citation_validity, extract_citations, is_refusal,
)


def _judge_llm():
    from langchain_anthropic import ChatAnthropic
    model = os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL", "claude-sonnet-5")
    return ChatAnthropic(model=model, max_tokens=1024)


class _Faithfulness(BaseModel):
    score: float = Field(..., ge=0, le=1,
                         description="fraction of the answer's factual claims directly supported by the context")
    unsupported_claims: List[str] = Field(default_factory=list)


class _Relevance(BaseModel):
    score: float = Field(..., ge=0, le=1, description="how well the answer addresses the question")
    reason: str = ""


async def faithfulness(answer: str, context: str, llm=None) -> dict:
    """LLM judge: is every claim in `answer` supported by `context`?"""
    judge = (llm or _judge_llm()).with_structured_output(_Faithfulness)
    r: _Faithfulness = await judge.ainvoke(
        "You are grading an answer for FAITHFULNESS to the provided context. Every "
        "factual claim in the answer must be directly supported by the context. "
        "Score = fraction of claims that are supported (1.0 = all supported). List "
        "any claims NOT supported by the context.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}")
    return {"score": r.score, "unsupported_claims": r.unsupported_claims}


async def answer_relevance(question: str, answer: str, llm=None) -> dict:
    """LLM judge: does `answer` actually address `question`?"""
    judge = (llm or _judge_llm()).with_structured_output(_Relevance)
    r: _Relevance = await judge.ainvoke(
        "Grade how well the ANSWER addresses the QUESTION (0 = off-topic, 1 = fully "
        "on-point). A correct refusal ('I don't have a source for that.') to a "
        "question the context can't answer counts as RELEVANT (1.0).\n\n"
        f"QUESTION:\n{question}\n\nANSWER:\n{answer}")
    return {"score": r.score, "reason": r.reason}
