"""LedgerLens API — FastAPI with Server-Sent Events.

Streams the agent run as it happens: which node is active, the retrieved
citations, the answer tokens, and any human-in-the-loop interrupt. Plain async
SSE over the LangGraph event stream — no Edge runtime required for streaming.

The human-in-the-loop checkpoint is persisted in Postgres (not in memory), so a
run that paused for approval survives a backend restart / serverless cold start —
the /resume can land on a different process and still continue.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from .db import close_pool
from .graph import build_graph

log = logging.getLogger("ledgerlens")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ledgerlens:ledgerlens@db:5432/ledgerlens")
NODE_NAMES = {"classify", "retrieve", "compute", "confirm", "answer"}

_graph = None  # compiled graph, set on startup by the lifespan below


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Compile the graph with a PERSISTENT Postgres checkpointer for the app's
    lifetime. Falls back to in-memory only if Postgres is unreachable, so the demo
    still runs — but the default, and the point, is durable interrupt/resume."""
    global _graph
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as saver:
            await saver.setup()  # idempotent — creates the checkpoint tables
            _graph = build_graph(checkpointer=saver)
            log.info("checkpointer: postgres (durable human-in-the-loop)")
            yield
            await close_pool()
            return
    except Exception as e:  # noqa: BLE001 — degrade gracefully, never fail to boot
        log.warning("postgres checkpointer unavailable (%s) — using in-memory", e)

    from langgraph.checkpoint.memory import MemorySaver
    _graph = build_graph(checkpointer=MemorySaver())
    yield
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


async def _run(payload, config) -> AsyncIterator[str]:
    """Drive the graph and translate LangGraph events into SSE frames."""
    async for ev in _graph.astream_events(payload, config=config, version="v2"):
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
    snap = await _graph.aget_state(config)
    interrupts = getattr(snap, "interrupts", None) or []
    if interrupts:
        yield _sse("interrupt", interrupts[0].value)
    else:
        answer = (snap.values or {}).get("answer", "")
        citations = (snap.values or {}).get("citations", [])
        yield _sse("done", {"answer": answer, "citations": citations})


@app.post("/chat")
async def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    return StreamingResponse(_run({"query": req.query}, config), media_type="text/event-stream")


@app.post("/resume")
async def resume(req: ResumeRequest):
    """Continue a run that paused for human confirmation — even after a restart."""
    config = {"configurable": {"thread_id": req.thread_id}}
    return StreamingResponse(_run(Command(resume=req.approved), config), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}
