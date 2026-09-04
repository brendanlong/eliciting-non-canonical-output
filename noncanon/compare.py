"""Compare cells per the plan's analysis specification.

    uv run python -m noncanon.compare out/think-dpo/dapo_sample500 out/think-main/dapo_sample500

For each pair of run directories (each holding ``metrics/analysis.jsonl``),
reports the pooled per-token rate of each cell with a rollout-bootstrap 95%
CI, the difference, a per-token two-proportion z test on pooled counts, and
a per-rollout permutation test (rollouts as the unit), under both the
headline counting rule and the segmentation-only rate. Deterministic
(seed 0, 5,000 resamples/permutations).
"""

from __future__ import annotations

import argparse
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np

CONVENTIONS = {"headline": ("nc_events", "n_units"), "segmentation only": ("nc_canonical", "n_canonical")}


def load(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = [json.loads(line) for line in (run_dir / "metrics" / "analysis.jsonl").open()]
    return {name: (np.array([r[nc] for r in rows]), np.array([r[n] for r in rows])) for name, (nc, n) in CONVENTIONS.items()}


def pooled(nc: np.ndarray, n: np.ndarray) -> float:
    return nc.sum() / n.sum()


def bootstrap_ci(nc: np.ndarray, n: np.ndarray, rng: np.random.Generator, B: int = 5000) -> tuple[float, float]:
    idx = rng.integers(0, len(nc), (B, len(nc)))
    return tuple(np.percentile(nc[idx].sum(1) / n[idx].sum(1), [2.5, 97.5]))


def permutation_test(a, b, rng: np.random.Generator, B: int = 5000) -> tuple[float, float]:
    (nc1, n1), (nc2, n2) = a, b
    observed = pooled(nc2, n2) - pooled(nc1, n1)
    nc, n, k = np.concatenate([nc1, nc2]), np.concatenate([n1, n2]), len(nc1)
    hits = 0
    for _ in range(B):
        p = rng.permutation(len(nc))
        hits += abs(pooled(nc[p[k:]], n[p[k:]]) - pooled(nc[p[:k]], n[p[:k]])) >= abs(observed)
    return observed, hits / B


def token_test(a, b) -> tuple[float, float]:
    """Two-proportion z on pooled counts (ignores clustering within rollouts)."""
    (nc1, n1), (nc2, n2) = a, b
    total = nc1.sum() + nc2.sum()
    expected = n1.sum() / (n1.sum() + n2.sum())
    z = (nc1.sum() / total - expected) / sqrt(expected * (1 - expected) / total)
    return z, 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="run directory of the first cell")
    ap.add_argument("b", type=Path, help="run directory of the second cell")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    A, B_ = load(args.a), load(args.b)
    for conv in CONVENTIONS:
        rng = np.random.default_rng(args.seed)
        print(f"== {conv} ==")
        for name, cell in ((args.a, A), (args.b, B_)):
            nc, n = cell[conv]
            lo, hi = bootstrap_ci(nc, n, rng)
            print(f"  {str(name):40s} {100 * pooled(nc, n):.4f}%  95% CI [{100 * lo:.4f}, {100 * hi:.4f}]  rollouts with >=1: {(nc > 0).sum()}/{len(nc)}")
        z, p_tok = token_test(A[conv], B_[conv])
        diff, p_roll = permutation_test(A[conv], B_[conv], rng)
        print(f"  b - a = {100 * diff:+.4f} pp | per-token z = {z:+.1f}, p = {p_tok:.2e} | per-rollout permutation p = {p_roll:.4f}")


if __name__ == "__main__":
    main()
