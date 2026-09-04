#!/usr/bin/env python3
"""Verify (or regenerate) the generated tables in RESULTS.md.

    uv run python scripts/check_results.py          # rerun every block's command and diff; exit 1 on any mismatch
    uv run python scripts/check_results.py --write  # replace each block with fresh output

A generated block is

    <!-- generated: <shell command, may continue over lines> -->
    ...table...
    <!-- end generated -->

The command runs from the repository root with the metrics of every cell
under out/ (download them from the HF dataset or rerun
scripts/recompute_metrics.sh first). Its stdout, stripped, must equal the
block's content, stripped.
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from pathlib import Path

BLOCK = re.compile(r"<!-- generated: (?P<cmd>.*?) -->\n(?P<body>.*?)<!-- end generated -->", re.S)


def run(cmd: str) -> str:
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
    if proc.returncode:
        sys.exit(f"command failed ({proc.returncode}): {cmd}\n{proc.stderr}")
    return proc.stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, default=Path(__file__).resolve().parent.parent / "RESULTS.md")
    ap.add_argument("--write", action="store_true", help="replace block contents with fresh output instead of checking")
    args = ap.parse_args()
    text = args.file.read_text()
    blocks = list(BLOCK.finditer(text))
    if not blocks:
        sys.exit(f"no generated blocks in {args.file}")
    failures, out, last = 0, [], 0
    for m in blocks:
        cmd = " ".join(line.strip().rstrip("\\").strip() for line in m.group("cmd").splitlines())  # join continuation lines
        fresh = run(cmd)
        recorded = m.group("body").strip()
        if fresh == recorded:
            print(f"ok       {cmd[:100]}")
        else:
            failures += 1
            print(f"MISMATCH {cmd[:100]}")
            print("\n".join(difflib.unified_diff(recorded.splitlines(), fresh.splitlines(), "RESULTS.md", "regenerated", lineterm="")))
        out.append(text[last:m.start("body")] + fresh + "\n")
        last = m.end("body")
    out.append(text[last:])
    if args.write:
        args.file.write_text("".join(out))
        print(f"wrote {len(blocks)} blocks to {args.file}")
    elif failures:
        sys.exit(f"{failures} of {len(blocks)} generated blocks differ from their command's output")
    else:
        print(f"all {len(blocks)} generated blocks match")


if __name__ == "__main__":
    main()
