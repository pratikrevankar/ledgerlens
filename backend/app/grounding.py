"""Pure grounding checks — citation extraction + validity, refusal detection.

Shared by the runtime Reflexion critic (app/critic.py) and the offline eval judge
(evals/judge.py), so the deployed self-correction and the eval score groundedness
the SAME way — no drift between what the agent enforces and what the harness
measures. No LLM, no I/O; trivially unit-tested.
"""
from __future__ import annotations

import re
from typing import List

REFUSAL = "I don't have a source for that."
_CITE_RE = re.compile(r"\[([^\]\[]+)\]")


def is_refusal(answer: str) -> bool:
    return REFUSAL.lower() in (answer or "").lower()


def extract_citations(answer: str) -> List[str]:
    """Pull bracketed citations like '[CGST Act, s.16]' out of an answer."""
    return [m.strip() for m in _CITE_RE.findall(answer or "")]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def citation_validity(answer: str, retrieved_citations: List[str]) -> dict:
    """Fraction of the answer's citations that match a retrieved provision.

    An invented/hallucinated citation (not in the retrieved context) drags the
    score down. A refusal or an uncited answer scores 1.0 — nothing false was
    asserted."""
    cited = extract_citations(answer)
    if not cited:
        return {"score": 1.0, "n_cited": 0, "invalid": []}
    allowed = {_norm(c) for c in retrieved_citations}
    invalid = [c for c in cited if _norm(c) not in allowed]
    return {
        "score": (len(cited) - len(invalid)) / len(cited),
        "n_cited": len(cited),
        "invalid": invalid,
    }
