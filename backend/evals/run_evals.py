"""Retrieval evals — recall@k and MRR over the labelled cases in cases.jsonl,
measured with the cross-encoder reranker OFF vs ON so the lift is quantified.

Measures whether the retrieval stack surfaces the right GST provision for a real
question. Needs the DB up + corpus ingested (docker compose up, or a provisioned
pgvector Postgres with `python -m app.ingest` run once). Run from backend/:

    python -m evals.run_evals

Prints a before/after table and exits non-zero if the reranked recall@k falls
below THRESHOLD — so it doubles as a CI gate. This is the number you quote:
"hybrid → +reranker took recall@4 from X% to Y%, MRR from A to B."
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval import retrieve  # noqa: E402

TOP_K = 4
THRESHOLD = 0.85  # fail CI if reranked recall@k falls below this


def _load_cases() -> List[dict]:
    lines = (Path(__file__).parent / "cases.jsonl").read_text().splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


async def _score(cases: List[dict], *, use_rerank: bool):
    """Return (recall@k, MRR, misses) for one retrieval configuration."""
    hits = 0
    rr_sum = 0.0
    misses = []
    for c in cases:
        chunks = await retrieve(c["query"], top_k=TOP_K, use_rerank=use_rerank)
        got = [ch.provision_id for ch in chunks]
        expected = set(c["expect"])
        # recall@k: did any expected provision make the top_k?
        if expected & set(got):
            hits += 1
        else:
            misses.append({"query": c["query"], "expected": c["expect"], "got": got})
        # reciprocal rank: 1/(rank of first expected hit), else 0
        rr = 0.0
        for rank, pid in enumerate(got):
            if pid in expected:
                rr = 1.0 / (rank + 1)
                break
        rr_sum += rr
    n = len(cases)
    return hits / n, rr_sum / n, misses


async def main() -> int:
    cases = _load_cases()
    base_recall, base_mrr, _ = await _score(cases, use_rerank=False)
    rr_recall, rr_mrr, rr_misses = await _score(cases, use_rerank=True)

    def pct(x: float) -> str:
        return f"{x:.0%}"

    print(f"\n  Retrieval eval — {len(cases)} labelled cases, top_k={TOP_K}\n")
    print(f"  {'config':<22}{'recall@k':>10}{'MRR':>8}")
    print(f"  {'-'*40}")
    print(f"  {'hybrid (RRF)':<22}{pct(base_recall):>10}{base_mrr:>8.3f}")
    print(f"  {'hybrid + reranker':<22}{pct(rr_recall):>10}{rr_mrr:>8.3f}")
    print(f"  {'-'*40}")
    print(f"  {'Δ from reranker':<22}"
          f"{f'{(rr_recall-base_recall)*100:+.0f} pts':>10}"
          f"{rr_mrr-base_mrr:>+8.3f}\n")

    for m in rr_misses:
        print(f"  MISS (reranked): {m['query']}\n"
              f"        expected {m['expected']}, got {m['got']}")

    ok = rr_recall >= THRESHOLD
    print(f"\n  {'PASS' if ok else 'FAIL'} — reranked recall@{TOP_K} "
          f"{pct(rr_recall)} vs threshold {pct(THRESHOLD)}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
