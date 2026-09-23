"""LedgerLens API — FastAPI with Server-Sent Events.

Streams the agent run as it happens: which node is active, the retrieved
citations, the answer tokens, and any human-in-the-loop interrupt. Plain async
SSE over the LangGraph event stream — no Edge runtime required for streaming.

The human-in-the-loop checkpoint is persisted in Postgres (not in memory), so a
run that paused for approval survives a backend restart / serverless cold start —
the /resume can land on a different process and still continue.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from .db import close_pool
from .graph import build_graph
from .tracing import configure as configure_tracing, run_config

log = logging.getLogger("ledgerlens")
# Turn on LangSmith tracing if the env asks for it (normalises var names). Every
# node/tool/LLM call is then a nested run; the per-request run_config below tags
# and adds metadata so traces are filterable.
_TRACING = configure_tracing()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ledgerlens:ledgerlens@db:5432/ledgerlens")
NODE_NAMES = {"classify", "retrieve", "compute", "confirm", "answer", "verify"}

_graph = None  # compiled graph, built lazily by _ensure_graph()
_graph_lock = asyncio.Lock()
_stack: Optional[AsyncExitStack] = None  # keeps the Postgres saver context open


async def _ensure_graph():
    """Build (once) the graph with a PERSISTENT Postgres checkpointer, falling back
    to in-memory only if Postgres is unreachable so the demo still runs.

    Called lazily from the request handlers rather than only from the lifespan: a
    long-lived server (uvicorn/docker) runs the lifespan, but a serverless platform
    (Vercel) may never fire ASGI lifespan events, so we cannot rely on it to set
    `_graph`. The lock makes concurrent first-requests build it exactly once, and
    the AsyncExitStack keeps the saver's connection open for the process lifetime."""
    global _graph, _stack
    if _graph is not None:
        return _graph
    async with _graph_lock:
        if _graph is not None:  # another request won the race
            return _graph
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            stack = AsyncExitStack()
            saver = await stack.enter_async_context(AsyncPostgresSaver.from_conn_string(DATABASE_URL))
            await saver.setup()  # idempotent — creates the checkpoint tables
            _graph = build_graph(checkpointer=saver)
            _stack = stack
            log.info("checkpointer: postgres (durable human-in-the-loop)")
        except Exception as e:  # noqa: BLE001 — degrade gracefully, never fail to boot
            log.warning("postgres checkpointer unavailable (%s) — using in-memory", e)
            from langgraph.checkpoint.memory import MemorySaver
            _graph = build_graph(checkpointer=MemorySaver())
    return _graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the graph when a long-lived server runs the lifespan (uvicorn/docker).
    Harmless on serverless where the handlers lazy-init anyway."""
    await _ensure_graph()
    yield
    if _stack is not None:
        await _stack.aclose()
    await close_pool()


app = FastAPI(title="LedgerLens", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    thread_id: str


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _run(graph, payload, config) -> AsyncIterator[str]:
    """Drive the graph and translate LangGraph events into SSE frames."""
    async for ev in graph.astream_events(payload, config=config, version="v2"):
        kind = ev["event"]
        if kind == "on_chain_start" and ev.get("name") in NODE_NAMES:
            yield _sse("node", {"node": ev["name"]})
        elif kind == "on_chain_end" and ev.get("name") == "retrieve":
            out = ev["data"].get("output") or {}
            yield _sse("citations", {"citations": out.get("citations", [])})
        elif kind == "on_chat_model_stream" and ev.get("metadata", {}).get("langgraph_node") == "answer":
            text = getattr(ev["data"].get("chunk"), "content", "")
            if isinstance(text, list):  # some providers chunk content as blocks
                text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
            if text:
                yield _sse("token", {"text": text})

    # After the stream drains, surface a human-in-the-loop interrupt if we paused.
    snap = await graph.aget_state(config)
    interrupts = getattr(snap, "interrupts", None) or []
    if interrupts:
        yield _sse("interrupt", interrupts[0].value)
    else:
        answer = (snap.values or {}).get("answer", "")
        citations = (snap.values or {}).get("citations", [])
        yield _sse("done", {"answer": answer, "citations": citations})


@app.post("/chat")
async def chat(req: ChatRequest):
    graph = await _ensure_graph()
    config = run_config(req.thread_id, {"query": req.query[:200]})
    return StreamingResponse(_run(graph, {"query": req.query}, config), media_type="text/event-stream")


@app.post("/resume")
async def resume(req: ResumeRequest):
    """Continue a run that paused for human confirmation — even after a restart."""
    graph = await _ensure_graph()
    config = run_config(req.thread_id, {"resumed": True, "approved": req.approved})
    return StreamingResponse(_run(graph, Command(resume=req.approved), config), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}
