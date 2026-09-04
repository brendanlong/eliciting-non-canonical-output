"""Build the held-out prompt sets used by every run.

    uv run python -m noncanon.prompts dapo --sample 500 # prompts/dapo_heldout.jsonl (+ fixed-seed sample)
    uv run python -m noncanon.prompts aime              # prompts/aime_2024_2025.jsonl
    uv run python -m noncanon.prompts overlap prompts/dapo_sample500.jsonl allenai/RLVR-MATH ...  # same rule, other RL sets

DAPO: ``open-r1/DAPO-Math-17k-Processed`` (English config) minus every problem
that appears in the OLMo-3 RL training sets (``allenai/Dolci-RL-Zero-Math-7B``,
``allenai/Dolci-Think-RL-7B``). Matching is on the lowercase alphanumeric
characters of the problem text (LaTeX spacing and punctuation differ between
copies of the same problem), either the whole string or its first
``PREFIX_CHARS`` characters (catches a training copy with an instruction or
constraint appended). The filter report, including how many *distinct* DAPO
problems share a prefix with each other (the false-positive risk of the prefix
rule), is written next to the prompt file.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfFileSystem

PROMPT_DIR = Path("prompts")
PREFIX_CHARS = 80  # of the alphanumeric-only text; collision count is in the report
TRAINING_SETS = {
    "Dolci-RL-Zero-Math-7B": "datasets/allenai/Dolci-RL-Zero-Math-7B/data/*.parquet",
    "Dolci-Think-RL-7B": "datasets/allenai/Dolci-Think-RL-7B/data/*.parquet",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"^\s*user:\s*", "", text)  # Dolci-Think-RL stores "user: <problem>"
    return re.sub(r"[^a-z0-9]", "", text)


class TrainingFilter:
    """Answers "was this problem in the RL training data?" by normalized text."""

    def __init__(self, training: dict[str, list[str]]):
        self.exact = {src: set(map(normalize, texts)) for src, texts in training.items()}
        self.prefix = {src: {t[:PREFIX_CHARS] for t in s} for src, s in self.exact.items()}

    def match(self, problem: str) -> str | None:
        norm = normalize(problem)
        for src in self.exact:
            if norm in self.exact[src]:
                return f"{src}:exact"
            if norm[:PREFIX_CHARS] in self.prefix[src]:
                return f"{src}:prefix"
        return None


def training_prompts() -> dict[str, list[str]]:
    """Prompt texts of the OLMo-3 RL training sets, per source.

    Dolci-Think-RL-7B is 1.9 GB because it stores tokenized columns; parquet
    column projection over the Hub filesystem reads only the ``prompt`` column.
    """
    fs = HfFileSystem()
    return {
        name: [p for f in sorted(fs.glob(pattern)) for p in pq.read_table(f, columns=["prompt"], filesystem=fs).column("prompt").to_pylist()]
        for name, pattern in TRAINING_SETS.items()
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_dapo(sample: int | None, seed: int) -> None:
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    training = training_prompts()
    filt = TrainingFilter(training)

    report: dict = {
        "source": "open-r1/DAPO-Math-17k-Processed (config en, split train)",
        "total_rows": len(ds),
        "prefix_chars": PREFIX_CHARS,
        "training_prompts": {k: len(v) for k, v in training.items()},
        "matched": Counter(),
        "duplicates_within_dapo": 0,
        "non_integer_answer": 0,
    }
    kept, seen = [], set()
    for row in ds:
        norm = normalize(row["prompt"])
        if norm in seen:
            report["duplicates_within_dapo"] += 1
            continue
        seen.add(norm)
        if hit := filt.match(row["prompt"]):
            report["matched"][hit] += 1
            continue
        answer = str(row["solution"]).strip()
        if not re.fullmatch(r"-?\d+", answer):
            report["non_integer_answer"] += 1
            continue
        kept.append({"id": row["extra_info"]["index"], "problem": row["prompt"], "answer": answer})
    # False-positive risk of the prefix rule: distinct DAPO problems sharing a prefix.
    prefixes = Counter(t[:PREFIX_CHARS] for t in seen)
    report["distinct_dapo_problems_sharing_a_prefix"] = sum(c for c in prefixes.values() if c > 1)
    report["kept"] = len(kept)

    write_jsonl(PROMPT_DIR / "dapo_heldout.jsonl", kept)
    (PROMPT_DIR / "dapo_filter_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if sample:
        write_jsonl(PROMPT_DIR / f"dapo_sample{sample}.jsonl", random.Random(seed).sample(kept, sample))
        print(f"wrote {sample}-prompt sample (seed {seed})")


def build_aime() -> None:
    rows = [
        {"id": f"aime2024-{r['ID']}", "problem": r["Problem"], "answer": str(r["Answer"])}
        for r in load_dataset("Maxwell-Jia/AIME_2024", split="train")
    ] + [
        {"id": f"aime2025-{r['problem_idx']}", "problem": r["problem"], "answer": str(r["answer"])}
        for r in load_dataset("MathArena/aime_2025", split="train")
    ]
    write_jsonl(PROMPT_DIR / "aime_2024_2025.jsonl", rows)
    print(f"wrote {len(rows)} AIME problems")


def user_turn(row: dict) -> str:
    """Prompt text of a row from a chat-format RL dataset (first user message), or its ``prompt`` column."""
    if "messages" in row:
        return next(m["content"] for m in row["messages"] if m["role"] == "user")
    return row["prompt"]


def report_overlap(prompt_file: Path, datasets: list[str]) -> None:
    """How many prompts of ``prompt_file`` appear in other RL datasets, by the DAPO filter's rule."""
    training = {name: [user_turn(r) for r in load_dataset(name, split="train")] for name in datasets}
    filt = TrainingFilter(training)
    rows = [json.loads(line) for line in prompt_file.open()]
    matches = Counter(m for r in rows if (m := filt.match(r["problem"])))
    print(json.dumps({"prompts": len(rows), "rows": {n: len(t) for n, t in training.items()}, "matches": dict(matches)}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dapo")
    d.add_argument("--sample", type=int, default=None, help="also write a fixed-seed random sample of this size")
    d.add_argument("--seed", type=int, default=0)
    sub.add_parser("aime")
    o = sub.add_parser("overlap", help="count prompts of a file that appear in other RL datasets (same matching rule)")
    o.add_argument("prompt_file", type=Path)
    o.add_argument("datasets", nargs="+", help="HF dataset names with a `messages` or `prompt` column")
    args = ap.parse_args()
    if args.cmd == "dapo":
        build_dapo(args.sample, args.seed)
    elif args.cmd == "aime":
        build_aime()
    else:
        report_overlap(args.prompt_file, args.datasets)


if __name__ == "__main__":
    main()
