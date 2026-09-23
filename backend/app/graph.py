"""LedgerLens agent — a LangGraph state machine.

    classify ─┬─► retrieve ─┬─► compute ─►┐
              │             └─────────────┴─► answer ─► verify ─┬─► END
              │                              ▲   (Reflexion critic) │
              │                              └── self-correct once ─┘
              └─► confirm ─► answer ─► END      (human-in-the-loop interrupt)

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
from .critic import verify_grounded
from .prompt_cache import text_block

LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5")  # override with any current Claude model id
MAX_REVISIONS = 1  # Reflexion: at most one self-correction pass after the critic


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
    verified: Optional[bool]   # critic verdict on the current answer
    critique: Optional[str]    # critic feedback the answer node self-corrects against
    revision: int              # self-correction attempts so far (bounded by MAX_REVISIONS)


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


def _route_after_classify(state: AgentState) -> str:
    # An action goes straight to the human-in-the-loop gate — no need to retrieve
    # tax law to record an entry (that's why the confirmation carried stray citations).
    return "confirm" if state.get("intent") == "action" else "retrieve"


def _route_after_retrieve(state: AgentState) -> str:
    return "compute" if state.get("intent") == "compute" else "answer"


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

    # On a self-correction pass the critic left feedback — fold it into the prompt
    # so the rewrite fixes the ungrounded claims rather than repeating them.
    fix = ""
    if state.get("critique"):
        fix = (f"\n\nYOUR PREVIOUS DRAFT WAS NOT FULLY GROUNDED. {state['critique']}\n"
               "Rewrite the answer to fix this — every claim must trace to the context above.")

    # Prompt caching: the system prompt + retrieved context are a stable prefix,
    # IDENTICAL on the first draft and the Reflexion self-correction pass for one
    # query — mark them cacheable so the second call reads them from cache. The
    # question (which varies, and carries the critique on a rewrite) is not cached.
    llm = _llm(streaming=True)
    from langchain_core.messages import SystemMessage, HumanMessage
    msg = await llm.ainvoke([
        SystemMessage(content=[text_block(_ANSWER_SYS)]),
        HumanMessage(content=[
            text_block(f"Context:\n{ctx}{tool_txt}"),
            text_block(f"\n\nUser question: {state['query']}{fix}", cache=False),
        ]),
    ])
    return {"answer": _as_text(msg.content)}


async def verify_node(state: AgentState) -> AgentState:
    """Reflexion critic — is the drafted answer grounded in the retrieved context?

    Deterministic invented-citation check first, then an LLM entailment check. If
    ungrounded, bump the revision counter and hand back a critique; the router
    loops to `answer` once (MAX_REVISIONS) so the model can self-correct."""
    grounded, critique = await verify_grounded(
        state.get("answer", ""),
        "\n\n".join(f"[{c['citation']}] {c['title']}\n{c['content']}" for c in state.get("chunks", [])),
        state.get("citations", []),
    )
    if grounded:
        return {"verified": True, "critique": None}
    return {"verified": False, "critique": critique, "revision": state.get("revision", 0) + 1}


def _route_after_answer(state: AgentState) -> str:
    # Action answers are deterministic (no LLM, no citations) — nothing to verify.
    return "end" if state.get("intent") == "action" else "verify"


def _route_after_verify(state: AgentState) -> str:
    # Stop when grounded, or once we've spent our self-correction budget.
    if state.get("verified") or state.get("revision", 0) > MAX_REVISIONS:
        return "end"
    return "answer"


# ── Assembly ─────────────────────────────────────────────────────────
def build_graph(checkpointer=None):
    from langgraph.graph import StateGraph, END
    g = StateGraph(AgentState)
    g.add_node("classify", classify_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("compute", compute_node)
    g.add_node("confirm", confirm_node)
    g.add_node("answer", answer_node)
    g.add_node("verify", verify_node)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", _route_after_classify,
                            {"confirm": "confirm", "retrieve": "retrieve"})
    g.add_conditional_edges("retrieve", _route_after_retrieve,
                            {"compute": "compute", "answer": "answer"})
    g.add_edge("compute", "answer")
    g.add_edge("confirm", "answer")
    # answer → verify (Reflexion critic) → self-correct once, or finish.
    g.add_conditional_edges("answer", _route_after_answer,
                            {"verify": "verify", "end": END})
    g.add_conditional_edges("verify", _route_after_verify,
                            {"answer": "answer", "end": END})
    return g.compile(checkpointer=checkpointer)
