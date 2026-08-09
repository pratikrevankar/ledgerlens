# Design decisions

Why LedgerLens is built the way it is — the reasoning behind each choice, and the
trade-offs. (Written to be defensible in a technical conversation, not marketing.)

## The one idea: trust boundaries

An LLM is unreliable at three things that matter in a regulated domain: **facts**
(it hallucinates section numbers), **arithmetic** (it mis-adds), and **memory**
(it has no durable state). So the design pushes each of those *out* of the model:

| Concern | Who owns it | Why not the LLM |
|---|---|---|
| Facts / law | Retrieval + cite-or-refuse | The model invents plausible-but-wrong citations |
| Arithmetic | `compute_gst()` (pure Python, unit-tested) | The model mis-computes; tax must be exact |
| State / resume | Postgres checkpointer | The model has no memory across turns/restarts |

The LLM is left to do only what it's good at: **classify intent, extract inputs,
and write grounded prose.** That division is the whole architecture.

## Why LangGraph (not a single prompt or a plain loop)

The steps have *different trust levels and control flow* — retrieval and math must
run deterministically, and an action must pause for a human. A state machine makes
that explicit and inspectable: each node is a unit you can test, trace, and reason
about, and `interrupt()` gives first-class human-in-the-loop. A single mega-prompt
would bury all of this inside the model where you can't verify or gate it.

- **Q: Why not just function-calling in a loop?** You can, but you lose the
  explicit graph — the routing, the interrupt/resume, and the per-node
  observability. LangGraph is that control plane.

## Why hybrid retrieval + Reciprocal Rank Fusion

Dense (embedding) search catches **paraphrase** — "tax on an out-of-state sale" →
IGST / place-of-supply — where the words don't match. Sparse (Postgres full-text)
search nails **exact statutory tokens** — `section 16`, `LUT`, `ITC` — that
embeddings blur together. Each is blind to the other's strength, so I run both and
merge with **RRF** (rank-based, no score-scale tuning). On the eval set, hybrid
beats either alone.

## Why local embeddings (bge-small via fastembed)

Retrieval runs with **no API key** — `docker compose up` gives working semantic
search offline, and CI runs the retrieval eval with zero secrets. Trade-off: a
384-dim local model is weaker than a large hosted embedding; at real scale I'd
A/B a hosted embedding + a cross-encoder **re-ranker**. For a grounded demo over a
curated corpus, local is the right call.

## Why the GST math is deterministic code

`compute_gst()` is pure and unit-tested; the LLM only decides *when* to call it and
extracts the inputs. Tax numbers can't be "probably right." This is also the
cleanest interview extension point: *"add cess," "handle a composition dealer"* —
it's ordinary code, fully in your control.

## Why cite-or-refuse

In compliance, a confident wrong answer is a liability. The answer prompt is
constrained to the retrieved context and must append a citation per claim or say
*"I don't have a source for that."* Grounding + refusal is the difference between a
chatbot and something you'd let near a client's books.

## Why a Postgres checkpointer for human-in-the-loop

The `confirm` node calls `interrupt()`; the run **pauses** and its state is written
to a checkpoint keyed by `thread_id`. With an in-memory saver that state dies on a
restart — and on serverless it dies between invocations, so a `/resume` landing on
a different process finds nothing. Persisting the checkpoint in **Postgres** makes
the pause durable: resume works across restarts and horizontal scale. That's the
production-grade version of HITL.

## Why SSE (and why this doesn't need "Edge")

The UI needs token + node-transition streaming. Server-Sent Events over a normal
async endpoint does that — no special runtime. A common misconception is that
streaming requires an Edge runtime; it doesn't, on Node or Python.

## What I'd do before this is "production"

- **Re-ranker** (cross-encoder) after hybrid retrieval; larger, versioned corpus.
- **Eval depth**: faithfulness / answer-groundedness (e.g. RAGAS), not just
  retrieval recall; adversarial "make it cite something false" cases.
- **Auth + rate limiting** on the API (a public LLM endpoint is a cost/abuse vector).
- **Observability**: LangSmith tracing on by default; per-node latency + token cost.
- **Corpus freshness**: ingestion pipeline with source dates + supersession, since
  tax law changes.

## Honest scope

This is a **reference implementation**, not a product: 25 curated provisions, a
5-node graph, no real users. Its job is to demonstrate the pattern — grounded RAG +
tool-use + durable HITL + evals — cleanly and runnably, in Python. The production
version of this pattern runs elsewhere in TypeScript.
