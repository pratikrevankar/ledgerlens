"""Pure unit tests — contextual-retrieval template + prompt-cache blocks.

No DB, no key, no model: the template context is deterministic and cache_control
is pure dict-shaping.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import prompt_cache  # noqa: E402
from app.contextualize import mode, template_context  # noqa: E402


def test_template_context_situates_from_metadata():
    p = {"act": "CGST Act 2017", "section": "16", "citation_ref": "CGST Act, s.16",
         "title": "Input tax credit", "topics": ["itc", "credit"]}
    c = template_context(p)
    assert "CGST Act, s.16" in c
    assert "input tax credit" in c.lower()
    assert "itc" in c


def test_template_context_without_topics():
    p = {"act": "IGST Act 2017", "section": "16", "citation_ref": "IGST Act, s.16",
         "title": "Zero-rated supply", "topics": []}
    c = template_context(p)
    assert c.endswith(".") and "topics:" not in c


def test_text_block_caches_by_default():
    b = prompt_cache.text_block("hello")
    assert b["type"] == "text" and b["text"] == "hello"
    assert b.get("cache_control") == {"type": "ephemeral"}


def test_text_block_opts_out_with_cache_false():
    assert "cache_control" not in prompt_cache.text_block("hi", cache=False)


def test_prompt_cache_disabled_via_env():
    old = os.environ.get("PROMPT_CACHE")
    os.environ["PROMPT_CACHE"] = "off"
    try:
        assert "cache_control" not in prompt_cache.text_block("x")
    finally:
        os.environ.pop("PROMPT_CACHE", None) if old is None else os.environ.__setitem__("PROMPT_CACHE", old)


def test_contextual_mode_defaults_off():
    old = os.environ.get("CONTEXTUAL_RETRIEVAL")
    os.environ.pop("CONTEXTUAL_RETRIEVAL", None)
    try:
        assert mode() == "off"
    finally:
        if old is not None:
            os.environ["CONTEXTUAL_RETRIEVAL"] = old
