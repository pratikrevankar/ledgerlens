"""Retrieval evals — recall@k over the labelled cases in cases.jsonl.

Measures whether hybrid retrieval surfaces the right GST provision for a real
question. Needs the DB up + corpus ingested (docker compose up). Run:

    python -m evals.run_evals            # from the backend/ dir

Exits non-zero if recall@k drops below THRESHOLD — so it doubles as a CI gate.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval import retrieve  # noqa: E402

TOP_K = 4
THRESHOLD = 0.85  # fail CI if recall@k falls below this


async def main() -> int:
    cases = [json.loads(line) for line in (Path(__file__).parent / "cases.jsonl").read_text().splitlines() if line.strip()]
    hits, misses = 0, []
    for c in cases:
        chunks = await retrieve(c["query"], top_k=TOP_K)
        got = [ch.provision_id for ch in chunks]
        if any(pid in got for pid in c["expect"]):
            hits += 1
        else:
            misses.append({"query": c["query"], "expected": c["expect"], "got": got})

    recall = hits / len(cases)
    print(f"\n  Retrieval recall@{TOP_K}: {hits}/{len(cases)} = {recall:.0%}\n")
    for m in misses:
        print(f"  MISS: {m['query']}\n        expected {m['expected']}, got {m['got']}")
    ok = recall >= THRESHOLD
    print(f"\n  {'PASS' if ok else 'FAIL'} (threshold {THRESHOLD:.0%})\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
