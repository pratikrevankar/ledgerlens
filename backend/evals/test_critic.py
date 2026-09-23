"""Unit tests for the Reflexion critic — no DB, no API key.

The LLM entailment step is stubbed; the deterministic checks (refusal, invented
citation) run for real and must short-circuit before any LLM call.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.critic import verify_grounded, _Grounding  # noqa: E402


class _FakeStructured:
    def __init__(self, result):
        self._r = result

    async def ainvoke(self, *a, **k):
        return self._r


class _FakeLLM:
    def __init__(self, result):
        self._r = result

    def with_structured_output(self, schema):
        return _FakeStructured(self._r)


def test_refusal_is_grounded_without_llm():
    grounded, critique = asyncio.run(verify_grounded(
        "I don't have a source for that.", "some context", []))
    assert grounded is True and critique == ""


def test_invented_citation_fails_before_any_llm_call():
    class _Boom:
        def with_structured_output(self, schema):
            raise AssertionError("LLM must not be called once a citation is invented")

    grounded, critique = asyncio.run(verify_grounded(
        "The rate is 18% [CGST Act, s.99].", "context", ["CGST Act, s.16"], llm=_Boom()))
    assert grounded is False
    assert "CGST Act, s.99" in critique


def test_valid_citation_grounded_via_llm():
    grounded, critique = asyncio.run(verify_grounded(
        "ITC needs conditions [CGST Act, s.16].", "context", ["CGST Act, s.16"],
        llm=_FakeLLM(_Grounding(grounded=True))))
    assert grounded is True and critique == ""


def test_valid_citation_ungrounded_via_llm_returns_critique():
    grounded, critique = asyncio.run(verify_grounded(
        "The rate is 40% [CGST Act, s.16].", "context", ["CGST Act, s.16"],
        llm=_FakeLLM(_Grounding(grounded=False, unsupported_claims=["40% rate"]))))
    assert grounded is False
    assert "40% rate" in critique
