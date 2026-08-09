"""LedgerLens agent — a LangGraph state machine.

    classify ─► retrieve ─┬─► compute ─►┐
                          ├─► confirm ──►┤ (human-in-the-loop interrupt)
                          └─────────────►┴─► answer (grounded, cite-or-refuse)

Why a graph and not a single prompt: the steps have different trust levels.
Retrieval and the GST math are deterministic and must not be hallucinated, so
they live in their own nodes; the LLM only classifies, extracts inputs, and
writes the final grounded answer. `confirm` uses LangGraph's interrupt() to pause
before any state-changing "action" until a human approves.
"""
from __future__ import annotations

import os
import re
from typing import List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from .retrieval import retrieve as hybrid_retrieve, Chunk
from .tools import GstInput, compute_gst

LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5")  # override with any current Claude model id


def _llm(streaming: bool = False):
    from langchain_anthropic import ChatAnthropic
    # Note: newer Claude models reject an explicit `temperature`, so we don't set it.
    return ChatAnthropic(model=LLM_MODEL, streaming=streaming, max_tokens=2048)


# ── State ────────────────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    query: str
    intent: Literal["question", "compute", "action"]
    chunks: List[dict]        # serialisable retrieved provisions
    tool_result: Optional[dict]
    confirmed: Optional[bool]
    answer: str
    citations: List[str]


# ── Structured LLM schemas ───────────────────────────────────────────
class _Intent(BaseModel):
    intent: Literal["question", "compute", "action"] = Field(
        ..., description="'compute' = needs a GST calculation; 'action' = asks to record/commit "
                         "something to the books; 'question' = everything else (explain a rule).")


class _GstExtract(BaseModel):
    taxable_value: Optional[float] = None
    rate_percent: float = 18.0
    supplier_state: Optional[str] = None
    place_of_supply_state: Optional[str] = None
    supply_type: Literal["goods", "services"] = "services"


# ── Nodes ────────────────────────────────────────────────────────────
# A clear "commit something to the books" phrasing is always an ACTION — pin it
# deterministically so the human-in-the-loop gate never depends on an LLM guess.
_ACTION_RE = re.compile(
    r"\b(record|book|post|create|save|add|enter|register|log)\b[\w\s,₹.]{0,40}\b"
    r"(sale|purchase|invoice|voucher|entry|bill|payment|receipt|transaction|expense)\b",
    re.I,
)


async def classify_node(state: AgentState) -> AgentState:
    if _ACTION_RE.search(state["query"]):
        return {"intent": "action"}
    llm = _llm().with_structured_output(_Intent)
    res: _Intent = await llm.ainvoke(
        f"Classify this user request for a GST assistant.\n\nRequest: {state['query']}")
    return {"intent": res.intent}


async def retrieve_node(state: AgentState) -> AgentState:
    chunks: List[Chunk] = await hybrid_retrieve(state["query"], top_k=4)
    return {
        "chunks": [{"citation": c.citation(), "title": c.title, "act": c.act,
                    "section": c.section, "content": c.content} for c in chunks],
        "citations": [c.citation() for c in chunks],
    }


async def compute_node(state: AgentState) -> AgentState:
    llm = _llm().with_structured_output(_GstExtract)
    ext: _GstExtract = await llm.ainvoke(
        "Extract GST calculation inputs from the request. Leave a field null if not stated.\n\n"
        f"Request: {state['query']}")
    if ext.taxable_value is None or not ext.supplier_state or not ext.place_of_supply_state:
        return {"tool_result": {"error": "missing_inputs",
                                "need": "taxable value, supplier state, and place-of-supply state"}}
    result = compute_gst(GstInput(
        taxable_value=ext.taxable_value, rate_percent=ext.rate_percent,
        supplier_state=ext.supplier_state, place_of_supply_state=ext.place_of_supply_state,
        supply_type=ext.supply_type,
    ))
    return {"tool_result": result.model_dump()}


async def confirm_node(state: AgentState) -> AgentState:
    """Human-in-the-loop gate before any state-changing action."""
    from langgraph.types import interrupt
    decision = interrupt({
        "type": "confirm_action",
        "summary": f"You asked to record: “{state['query']}”. Approve before I commit anything.",
    })
    return {"confirmed": bool(decision)}


def _route_after_retrieve(state: AgentState) -> str:
    return {"compute": "compute", "action": "confirm"}.get(state.get("intent", "question"), "answer")


_ANSWER_SYS = (
    "You are LedgerLens, an Indian-GST assistant. Answer ONLY from the provided context provisions. "
    "Cite the section after each claim in square brackets, e.g. [CGST Act, s.16]. "
    "If the context does not support an answer, say exactly: \"I don't have a source for that.\" "
    "Never invent a section number or a rate. Be concise."
)


def _as_text(content) -> str:
    """Normalise an LLM message's content to plain text. Reasoning models (e.g.
    claude-sonnet-5) return a LIST of blocks (thinking + text) — keep only text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


async def answer_node(state: AgentState):
    # Action path is deterministic — never grounded, never an LLM call. Approving
    # confirms; rejecting cancels. (Demo: there is no real ledger to write to.)
    if state.get("intent") == "action":
        if state.get("confirmed"):
            return {"answer": f"✓ Approved — I'd post this entry to the ledger now: “{state['query']}”.\n"
                              f"(Demo build: no persistent write.)"}
        return {"answer": "No problem — I haven't recorded anything."}

    ctx = "\n\n".join(f"[{c['citation']}] {c['title']}\n{c['content']}" for c in state.get("chunks", []))
    tool = state.get("tool_result")
    tool_txt = ""
    if tool and "error" not in tool:
        tool_txt = f"\n\nComputed result (trust these numbers, they are exact):\n{tool}"
    elif tool and tool.get("error") == "missing_inputs":
        tool_txt = f"\n\nThe user asked for a calculation but did not give: {tool['need']}. Ask for them."

    llm = _llm(streaming=True)
    from langchain_core.messages import SystemMessage, HumanMessage
    msg = await llm.ainvoke([
        SystemMessage(content=_ANSWER_SYS),
        HumanMessage(content=f"Context:\n{ctx}{tool_txt}\n\nUser question: {state['query']}"),
    ])
    return {"answer": _as_text(msg.content)}


# ── Assembly ─────────────────────────────────────────────────────────
def build_graph(checkpointer=None):
    from langgraph.graph import StateGraph, END
    g = StateGraph(AgentState)
    g.add_node("classify", classify_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("compute", compute_node)
    g.add_node("confirm", confirm_node)
    g.add_node("answer", answer_node)
    g.set_entry_point("classify")
    g.add_edge("classify", "retrieve")
    g.add_conditional_edges("retrieve", _route_after_retrieve,
                            {"compute": "compute", "confirm": "confirm", "answer": "answer"})
    g.add_edge("compute", "answer")
    g.add_edge("confirm", "answer")
    g.add_edge("answer", END)
    return g.compile(checkpointer=checkpointer)
