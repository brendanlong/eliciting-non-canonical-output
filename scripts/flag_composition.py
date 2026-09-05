#!/usr/bin/env python3
"""What the flagged-rollout fraction is made of: rollouts flagged only by byte fragments, only by
bare-space spans (a standalone whitespace token followed by the deviation), or by one recurring
span text, and the fraction left when those are excluded.

    uv run python scripts/flag_composition.py --arm untruncated "Think-SFT=out/think-sft/dapo_sample500" ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from noncanon.compare import load_rows, parse_spec, wilson


def kind(e: dict) -> str:
    if e["canonical"] is None:
        return "fragment"
    if e["emitted"][0].strip() == "" and len(e["emitted"]) > 1:
        return "bare-space"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells", nargs="+", help="label=run_dir[:arm]")
    ap.add_argument("--arm", default=None)
    args = ap.parse_args()
    print("| cell | rollouts | flagged | events: fragment / bare-space / other | most common span text (rollouts) | flagged excluding fragments | excluding fragments and bare-space | excluding fragments, bare-space and the most common text |")
    print("|---|--:|--:|---|---|--:|--:|--:|")
    for spec in args.cells:
        label, run, arm = parse_spec(spec, args.arm)
        rows = load_rows(run, arm)
        arm = arm or rows[0]["file"].removesuffix(".parquet")
        events = [json.loads(l) for l in (run / "metrics" / "examples.jsonl").open()]
        events = [e for e in events if e["file"].startswith(arm)]
        by_rollout = defaultdict(list)
        for e in events:
            by_rollout[(e["prompt_id"], e["sample"])].append(e)
        kinds = Counter(kind(e) for e in events)
        texts = Counter("".join(e["emitted"]) for e in events if kind(e) == "other")
        top, _ = texts.most_common(1)[0] if texts else ("", 0)
        top_rollouts = sum(any(kind(e) == "other" and "".join(e["emitted"]) == top for e in es) for es in by_rollout.values())
        n = len(rows)
        flagged = sum(r["nc_events"] > 0 for r in rows)
        def excl(pred) -> str:
            k = sum(any(pred(e) for e in es) for es in by_rollout.values())
            lo, hi = wilson(k, n)
            return f"{100 * k / n:.1f}% ({k}) [{100 * lo:.1f}–{100 * hi:.1f}]"
        print(f"| {label} | {n} | {100 * flagged / n:.1f}% ({flagged}) | {kinds['fragment']} / {kinds['bare-space']} / {kinds['other']} | `{top.replace(' ', '·')}` ({top_rollouts}) | "
              f"{excl(lambda e: kind(e) != 'fragment')} | {excl(lambda e: kind(e) == 'other')} | {excl(lambda e: kind(e) == 'other' and ''.join(e['emitted']) != top)} |")


if __name__ == "__main__":
    main()
