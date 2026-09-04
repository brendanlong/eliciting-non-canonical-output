"""Where in the next-token distribution do non-canonical spans come from?

    uv run python -m noncanon.tail --tokenizer allenai/Olmo-3-7B-Think out/think-main/dapo_sample500 ...

From the stored top-10 logprobs of each run: sharpness of the distribution
(mass beyond the top-k, entropy), the rank of the emitted token at every
position, the rank of the first token of each non-canonical span, and the
special case of spans that begin with a bare space token (canonical only
before a digit in cl100k-style tokenizers), where the deviation is the
token after the space. Byte fragments are excluded from the span
statistics. Descriptive only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from noncanon.metrics import Analyzer
from noncanon.records import iter_records

BANDS = ((1, 1), (2, 3), (4, 10), (11, 10**9))
KS = (1, 2, 3, 5, 10)


def band_shares(ranks: np.ndarray) -> str:
    if not len(ranks):
        return "n/a"
    return " / ".join(f"{100 * ((ranks >= lo) & (ranks <= hi)).mean():.1f}%" for lo, hi in BANDS)


def analyze_run(an: Analyzer, run_dir: Path) -> None:
    ranks, beyond = [], {k: [] for k in KS}
    first, second_after_space, next_piece = [], [], Counter()
    entropies = []
    for rec in iter_records(run_dir / "untruncated.parquet") if (run_dir / "untruncated.parquet").exists() else iter_records(run_dir / "recommended.parquet"):
        tk = rec["topk_logprobs"]
        if not tk:
            continue
        lps = np.array([sorted(r, reverse=True)[:10] for r in tk], dtype=np.float64)
        probs = np.exp(lps)
        cum = np.cumsum(probs, axis=1)
        for k in KS:
            beyond[k].append(1 - cum[:, k - 1])
        norm = lps - np.logaddexp.reduce(lps, axis=1, keepdims=True)
        entropies.append(-(np.exp(norm) * norm).sum(1))
        rank = 1 + (lps > np.array(rec["logprobs"], dtype=np.float64)[:, None] + 1e-9).sum(1)
        ranks.append(rank)
        a = an.analyze(rec)
        for sp in a["spans"]:
            if sp["shape"] == "byte-fragment" or sp["pos"] >= len(rank):
                continue
            if sp["emitted"][0].strip() == "" and len(sp["emitted"]) > 1 and sp["pos"] + 1 < len(rank):
                second_after_space.append(rank[sp["pos"] + 1])
                next_piece[sp["emitted"][1]] += 1
            else:
                first.append(rank[sp["pos"]])
    ranks = np.concatenate(ranks)
    first, second_after_space = np.array(first), np.array(second_after_space)
    ent = np.concatenate(entropies)
    print(f"\n{run_dir}: {len(ranks):,} positions, {len(first) + len(second_after_space)} spans")
    print("  mean mass beyond top-k:      " + "  ".join(f"k={k}: {np.concatenate(beyond[k]).mean():.4f}" for k in KS))
    print(f"  entropy (top-10, nats): mean {ent.mean():.4f}, p90 {np.percentile(ent, 90):.4f}, frac > 1: {100 * (ent > 1).mean():.2f}%")
    print("  all sampled tokens at rank 1 / 2-3 / 4-10 / >10:            " + band_shares(ranks))
    print(f"  spans starting with a bare space: {len(second_after_space)}; rank of the token after it: " + band_shares(second_after_space))
    if next_piece:
        print("    most common token after the space:", next_piece.most_common(6))
    print(f"  other spans: {len(first)}; first-token rank at 1 / 2-3 / 4-10 / >10: " + band_shares(first))
    print("  other spans per 1M samples in each rank band:               " + " / ".join(
        f"{1e6 * ((first >= lo) & (first <= hi)).sum() / max(1, ((ranks >= lo) & (ranks <= hi)).sum()):.1f}" for lo, hi in BANDS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("runs", type=Path, nargs="+", help="run directories holding <arm>.parquet")
    args = ap.parse_args()
    an = Analyzer(args.tokenizer, args.revision)
    for run_dir in args.runs:
        analyze_run(an, run_dir)


if __name__ == "__main__":
    main()
