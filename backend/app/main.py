"""LedgerLens API — FastAPI with Server-Sent Events.

Streams the agent run as it happens: which node is active, the retrieved
citations, the answer tokens, and any human-in-the-loop interrupt. Plain async
SSE over the LangGraph event stream — no Edge runtime required for streaming.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from .db import close_pool
from .graph import build_graph

NODE_NAMES = {"classify", "retrieve", "compute", "confirm", "answer"}

app = FastAPI(title="LedgerLens", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_checkpointer = MemorySaver()
_graph = build_graph(checkpointer=_checkpointer)


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

    # After the stream drains, check whether we paused on a human-in-the-loop interrupt.
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
    return StreamingResponse(
        _run({"query": req.query}, config), media_type="text/event-stream",
    )


@app.post("/resume")
async def resume(req: ResumeRequest):
    """Continue a run that paused for human confirmation."""
    config = {"configurable": {"thread_id": req.thread_id}}
    return StreamingResponse(
        _run(Command(resume=req.approved), config), media_type="text/event-stream",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.on_event("shutdown")
async def _shutdown():
    await close_pool()
