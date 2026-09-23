"""Pure unit tests — reranker ordering + deterministic citation checks.

Run with `pytest`, no DB / no API key / no model download: the reranker takes an
injectable scorer, and the citation checks are pure string logic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.reranker import rerank  # noqa: E402
from evals.judge import (  # noqa: E402
    REFUSAL, citation_validity, extract_citations, is_refusal,
)


# ── reranker ordering (stub scorer, no model) ────────────────────────
def test_rerank_reorders_by_score_and_cuts_to_top_k():
    cands = ["a", "b", "c"]
    # scores line up with a,b,c → c is best, then a, then b
    out = rerank("q", cands, text_of=lambda x: x, top_k=2,
                 scorer=lambda q, docs: [0.2, 0.1, 0.9])
    assert out == ["c", "a"]


def test_rerank_degrades_to_input_order_on_scorer_failure():
    def boom(q, docs):
        raise RuntimeError("model unavailable")
    out = rerank("q", ["a", "b", "c"], text_of=lambda x: x, top_k=2, scorer=boom)
    assert out == ["a", "b"]  # falls back to fused order, still cut to top_k


def test_rerank_degrades_when_scores_length_mismatches():
    out = rerank("q", ["a", "b"], text_of=lambda x: x, top_k=2,
                 scorer=lambda q, docs: [0.5])  # wrong length
    assert out == ["a", "b"]


def test_rerank_empty_candidates():
    assert rerank("q", [], text_of=lambda x: x, top_k=3, scorer=lambda q, d: []) == []


# ── citation validity (deterministic) ────────────────────────────────
def test_citation_validity_all_valid():
    r = citation_validity("ITC needs conditions [CGST Act, s.16].", ["CGST Act, s.16"])
    assert r["score"] == 1.0 and r["invalid"] == [] and r["n_cited"] == 1


def test_citation_validity_flags_invented_citation():
    r = citation_validity("The rate is 18% [CGST Act, s.99].", ["CGST Act, s.16"])
    assert r["score"] == 0.0 and r["invalid"] == ["CGST Act, s.99"]


def test_citation_validity_partial():
    r = citation_validity("[CGST Act, s.16] and [Fake Act, s.1]", ["CGST Act, s.16"])
    assert r["score"] == 0.5 and r["invalid"] == ["Fake Act, s.1"]


def test_citation_validity_refusal_is_clean():
    r = citation_validity(REFUSAL, [])
    assert r["score"] == 1.0 and r["n_cited"] == 0


def test_citation_validity_is_whitespace_insensitive():
    r = citation_validity("[CGST Act,  s.16]", ["CGST Act, s.16"])
    assert r["score"] == 1.0


# ── helpers ──────────────────────────────────────────────────────────
def test_extract_citations():
    assert extract_citations("a [X] b [Y, s.1] c") == ["X", "Y, s.1"]
    assert extract_citations("no citations here") == []


def test_is_refusal():
    assert is_refusal("Sorry — I don't have a source for that.")
    assert not is_refusal("The rate is 18% [rates].")
