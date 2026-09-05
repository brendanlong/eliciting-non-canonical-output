"""Text-level statistics behind the DPO bare-space spans (Brendan, 2026-09-05).

    uv run python -m noncanon.textstats --arm untruncated "Think-SFT=out/think-sft/dapo_sample500" ...

Per cell, from the transcripts and span examples (no model, no tokenizer):
CJK characters per 100k characters of output and the rollouts containing
any; spans whose first emitted token is whitespace-only (the bare space is
canonical before a digit, so the span is the non-digit token sampled after
it) and what that second token is (a byte fragment starting a multi-byte
character, a CJK character, or something else); and standalone
byte-fragment events whose preceding context ends in a space.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from noncanon.compare import load_rows, parse_spec

CJK = re.compile(r"[㐀-鿿]")


def stats(run: Path, arm: str | None) -> dict:
    rows = load_rows(run, arm)
    arm_name = rows[0]["file"].removesuffix(".parquet")
    chars = cjk = with_cjk = 0
    for line in (run / "metrics" / "transcripts.jsonl").open():
        t = json.loads(line)
        if not t["file"].startswith(arm_name):
            continue
        n = len(CJK.findall(t["transcript"]))
        chars += len(t["transcript"]); cjk += n; with_cjk += n > 0
    ws_first, second, frag_after_space = 0, Counter(), 0
    for line in (run / "metrics" / "examples.jsonl").open():
        e = json.loads(line)
        if not e["file"].startswith(arm_name):
            continue
        if e["canonical"] is None:
            frag_after_space += e["context"].endswith(" ")
            continue
        if e["emitted"][0].strip() == "" and len(e["emitted"]) > 1:
            ws_first += 1
            nxt = e["emitted"][1]
            second["byte fragment" if "�" in nxt else "CJK" if CJK.search(nxt) else "other"] += 1
    return {"rollouts": len(rows), "cjk_per_100k": 1e5 * cjk / max(1, chars), "with_cjk": with_cjk, "ws_first": ws_first, "second": second, "frag_after_space": frag_after_space}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells", nargs="+", help="label=run_dir[:arm]")
    ap.add_argument("--arm", default=None)
    args = ap.parse_args()
    print("| cell | CJK characters per 100k | rollouts with any CJK | spans starting with a whitespace-only token | second token: byte fragment / CJK / other | standalone byte-fragment events preceded by a space |")
    print("|---|--:|--:|--:|---|--:|")
    for spec in args.cells:
        label, run, arm = parse_spec(spec, args.arm)
        s = stats(run, arm)
        sec = s["second"]
        print(f"| {label} | {s['cjk_per_100k']:.1f} | {s['with_cjk']}/{s['rollouts']} | {s['ws_first']} | {sec['byte fragment']} / {sec['CJK']} / {sec['other']} | {s['frag_after_space']} |")


if __name__ == "__main__":
    main()
