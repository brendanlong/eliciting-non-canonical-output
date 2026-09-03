"""Build the held-out prompt sets used by every run.

    uv run python -m noncanon.prompts dapo --pilot 50   # prompts/dapo_heldout.jsonl (+ pilot sample)
    uv run python -m noncanon.prompts aime              # prompts/aime_2024_2025.jsonl

DAPO: ``open-r1/DAPO-Math-17k-Processed`` (English config) minus every problem
whose normalized text matches a prompt in the OLMo-3 RL training sets
(``allenai/Dolci-RL-Zero-Math-7B``, ``allenai/Dolci-Think-RL-7B``), either by
exact normalized match or by a shared 80-character normalized prefix. The
filter report (counts per source and match type) is written next to the
prompt file so the surviving count is on record.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfFileSystem

PROMPT_DIR = Path("prompts")
PREFIX_CHARS = 80  # of the alphanumeric-only normalized text
TRAINING_SETS = {
    "Dolci-RL-Zero-Math-7B": "datasets/allenai/Dolci-RL-Zero-Math-7B/data/*.parquet",
    "Dolci-Think-RL-7B": "datasets/allenai/Dolci-Think-RL-7B/data/*.parquet",
}


def normalize(text: str) -> str:
    """Lowercase alphanumerics only: LaTeX spacing and punctuation differ
    between copies of the same problem, so whitespace-collapsing alone misses
    matches."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"^\s*user:\s*", "", text)  # Dolci-Think-RL stores "user: <problem>"
    return re.sub(r"[^a-z0-9]", "", text)


def training_prompts() -> dict[str, list[str]]:
    """Normalized prompt texts of the OLMo-3 RL training sets, per source.

    Dolci-Think-RL-7B is 1.9 GB because it stores tokenized columns; parquet
    column projection over the Hub filesystem reads only the ``prompt`` column.
    """
    fs = HfFileSystem()
    out: dict[str, list[str]] = {}
    for name, pattern in TRAINING_SETS.items():
        texts: list[str] = []
        for f in sorted(fs.glob(pattern)):
            table = pq.read_table(f, columns=["prompt"], filesystem=fs)
            texts.extend(normalize(p) for p in table.column("prompt").to_pylist())
        out[name] = texts
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_dapo(pilot: int | None, seed: int) -> None:
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    train = training_prompts()
    exact = {src: set(v) for src, v in train.items()}
    prefix = {src: {t[:PREFIX_CHARS] for t in v} for src, v in train.items()}

    report: dict = {
        "source": "open-r1/DAPO-Math-17k-Processed (config en, split train)",
        "total_rows": len(ds),
        "prefix_chars": PREFIX_CHARS,
        "training_prompts": {k: len(v) for k, v in train.items()},
        "matched": {},
        "duplicates_within_dapo": 0,
        "non_integer_answer": 0,
    }
    kept: list[dict] = []
    seen: set[str] = set()
    for row in ds:
        norm = normalize(row["prompt"])
        if norm in seen:
            report["duplicates_within_dapo"] += 1
            continue
        seen.add(norm)
        hit = None
        for src in train:
            if norm in exact[src]:
                hit = f"{src}:exact"
                break
            if norm[:PREFIX_CHARS] in prefix[src]:
                hit = f"{src}:prefix"
                break
        if hit:
            report["matched"][hit] = report["matched"].get(hit, 0) + 1
            continue
        answer = str(row["solution"]).strip()
        if not re.fullmatch(r"-?\d+", answer):
            report["non_integer_answer"] += 1
            continue
        kept.append({"id": row["extra_info"]["index"], "problem": row["prompt"], "answer": answer})
    report["kept"] = len(kept)

    write_jsonl(PROMPT_DIR / "dapo_heldout.jsonl", kept)
    (PROMPT_DIR / "dapo_filter_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if pilot:
        sample = random.Random(seed).sample(kept, pilot)
        write_jsonl(PROMPT_DIR / f"dapo_pilot{pilot}.jsonl", sample)
        print(f"wrote {pilot}-prompt pilot sample (seed {seed})")


def build_aime() -> None:
    rows = []
    for r in load_dataset("Maxwell-Jia/AIME_2024", split="train"):
        rows.append({"id": f"aime2024-{r['ID']}", "problem": r["Problem"], "answer": str(r["Answer"])})
    for r in load_dataset("MathArena/aime_2025", split="train"):
        rows.append({"id": f"aime2025-{r['problem_idx']}", "problem": r["problem"], "answer": str(r["answer"])})
    write_jsonl(PROMPT_DIR / "aime_2024_2025.jsonl", rows)
    print(f"wrote {len(rows)} AIME problems")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dapo")
    d.add_argument("--pilot", type=int, default=None, help="also write a random pilot sample of this size")
    d.add_argument("--seed", type=int, default=0)
    sub.add_parser("aime")
    args = ap.parse_args()
    if args.cmd == "dapo":
        build_dapo(args.pilot, args.seed)
    else:
        build_aime()


if __name__ == "__main__":
    main()
