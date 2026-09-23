"""Pure unit tests — LangSmith env normalisation + run_config shaping. No network."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import tracing  # noqa: E402

_TRACE_VARS = [
    "LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT",
    "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT", "LANGCHAIN_ENDPOINT",
]


def _clear_env():
    for k in _TRACE_VARS:
        os.environ.pop(k, None)


def test_configure_off_is_noop():
    saved = {k: os.environ.get(k) for k in _TRACE_VARS}
    try:
        _clear_env()
        os.environ["LANGSMITH_TRACING"] = "false"
        assert tracing.configure() is False
        assert "LANGCHAIN_TRACING_V2" not in os.environ
    finally:
        _clear_env()
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_configure_on_mirrors_vars():
    saved = {k: os.environ.get(k) for k in _TRACE_VARS}
    try:
        _clear_env()
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = "ls-key"
        os.environ["LANGSMITH_PROJECT"] = "ledgerlens"
        assert tracing.configure() is True
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGCHAIN_API_KEY"] == "ls-key"
        assert os.environ["LANGCHAIN_PROJECT"] == "ledgerlens"
    finally:
        _clear_env()
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_run_config_carries_thread_tags_and_metadata():
    cfg = tracing.run_config("thread-123", {"eval": True})
    assert cfg["configurable"]["thread_id"] == "thread-123"
    assert cfg["run_name"] == "ledgerlens-chat"
    assert "ledgerlens" in cfg["tags"]
    assert any(t.startswith("rerank:") for t in cfg["tags"])
    assert any(t.startswith("ctx:") for t in cfg["tags"])
    assert cfg["metadata"]["thread_id"] == "thread-123"
    assert cfg["metadata"]["eval"] is True
    # the feature-flag metadata is present for filtering
    for key in ("rerank", "contextual_retrieval", "prompt_cache"):
        assert key in cfg["metadata"]
