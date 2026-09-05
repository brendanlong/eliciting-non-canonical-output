#!/usr/bin/env python3
"""Randomly selected non-canonical spans for the writeup, as a Markdown table.

    uv run python scripts/random_examples.py --per-cell 3 --seed 0 > figures/random_examples.md

Draws uniformly from every event in each cell's metrics/examples.jsonl
(byte fragments included), so the sample is not curated. Tokens are shown
with · for a space and ⏎ for a newline.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CELLS = [("OLMo-3-7B Think-DPO", "think-dpo"), ("OLMo-3-7B Think RL final", "think-main"),
         ("OLMo-3-7B RL-Zero-Math step 2000", "rlzero-math"), ("OLMo-3-7B Instruct-DPO", "instruct-dpo")]


def show(tokens: list[str] | None) -> str:
    if tokens is None:
        return "(incomplete UTF-8 bytes, no text form)"
    return " ".join("`" + t.replace(" ", "·").replace("\n", "⏎").replace("`", "\\`") + "`" for t in tokens)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-cell", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="untruncated")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    print("| cell | preceding text | emitted tokens | canonical tokens | position |")
    print("|---|---|---|---|--:|")
    for label, run in CELLS:
        events = [json.loads(l) for l in (Path("out") / run / "dapo_sample500" / "metrics" / "examples.jsonl").open()]
        events = [e for e in events if e["file"].startswith(args.arm)]
        for e in rng.sample(events, args.per_cell):
            ctx = e["context"].replace("\n", "⏎").replace("|", "\\|").replace("`", "\\`")
            print(f"| {label} | `…{ctx}` | {show(e['emitted'])} | {show(e['canonical'])} | {e['pos']:,} |")


if __name__ == "__main__":
    main()
