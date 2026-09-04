"""Compare cells per the plan's analysis specification.

    uv run python -m noncanon.compare out/think-dpo/dapo_sample500 out/think-main/dapo_sample500

For each pair of run directories (each holding ``metrics/analysis.jsonl``),
reports the pooled per-token rate of each cell with a rollout-bootstrap 95%
CI, the difference b − a, a per-token z test on pooled counts (the
conditional-binomial form, numerically the two-proportion z; signed so
that z > 0 means b is higher), and a per-rollout permutation test
(rollouts as the unit), under both the headline counting rule and the
segmentation-only rate. Deterministic and order-invariant: each cell's
bootstrap and the permutation test get their own seeded generator.
Permutation p is reported as "< 1/B" when no permutation reached the
observed difference.
"""

from __future__ import annotations

import argparse
import json
from math import erfc, sqrt
from pathlib import Path

import numpy as np

CONVENTIONS = {"headline": ("nc_events", "n_units"), "segmentation only": ("nc_canonical", "n_canonical")}


def load(run_dir: Path, arm: str | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = [json.loads(line) for line in (run_dir / "metrics" / "analysis.jsonl").open()]
    if arm:
        rows = [r for r in rows if r["file"].startswith(arm)]
    files = {r["file"] for r in rows}
    assert len(files) == 1, f"{run_dir} mixes sampling arms {sorted(files)}; pass --arm"
    return {name: (np.array([r[nc] for r in rows]), np.array([r[n] for r in rows])) for name, (nc, n) in CONVENTIONS.items()}


def pooled(nc: np.ndarray, n: np.ndarray) -> float:
    return nc.sum() / n.sum()


def bootstrap_ci(nc: np.ndarray, n: np.ndarray, seed: int, B: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(nc), (B, len(nc)))
    lo, hi = np.percentile(nc[idx].sum(1) / n[idx].sum(1), [2.5, 97.5])
    return float(lo), float(hi)


def permutation_test(a, b, seed: int, B: int = 20000) -> tuple[float, int, int]:
    """Two-sided; returns (observed b − a, permutations at least as extreme, B)."""
    (nc1, n1), (nc2, n2) = a, b
    observed = pooled(nc2, n2) - pooled(nc1, n1)
    nc, n, k = np.concatenate([nc1, nc2]), np.concatenate([n1, n2]), len(nc1)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(B):
        p = rng.permutation(len(nc))
        hits += abs(pooled(nc[p[k:]], n[p[k:]]) - pooled(nc[p[:k]], n[p[:k]])) >= abs(observed)
    return float(observed), int(hits), B


def token_test(a, b) -> tuple[float, float]:
    """z on pooled counts, signed so z > 0 means b has the higher rate.

    Conditional-binomial form (nc_b | total ~ Bin(total, n_b / (n_a + n_b)));
    agrees with the textbook two-proportion z to 5 significant figures at
    these rates. Ignores clustering within rollouts.
    """
    (nc1, n1), (nc2, n2) = a, b
    total = nc1.sum() + nc2.sum()
    expected = n2.sum() / (n1.sum() + n2.sum())
    z = (nc2.sum() / total - expected) / sqrt(expected * (1 - expected) / total)
    return float(z), erfc(abs(z) / sqrt(2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="run directory of the first cell")
    ap.add_argument("b", type=Path, help="run directory of the second cell")
    ap.add_argument("--arm", default=None, help="restrict to one sampling arm when a run directory holds several")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    A, B_ = load(args.a, args.arm), load(args.b, args.arm)
    for conv in CONVENTIONS:
        print(f"== {conv} ==")
        for name, cell in ((args.a, A), (args.b, B_)):
            nc, n = cell[conv]
            # Each cell's CI is seeded from its own data size and the seed, so
            # `compare a b` and `compare b a` print identical numbers.
            lo, hi = bootstrap_ci(nc, n, seed=[args.seed, len(nc), int(nc.sum())])
            print(f"  {str(name):40s} {100 * pooled(nc, n):.4f}%  95% CI [{100 * lo:.4f}, {100 * hi:.4f}]  rollouts with >=1: {(nc > 0).sum()}/{len(nc)}")
        z, p_tok = token_test(A[conv], B_[conv])
        diff, hits, B = permutation_test(A[conv], B_[conv], seed=[args.seed, 1])
        p_roll = f"< {1 / B:.5f}" if hits == 0 else f"= {hits / B:.4f}"
        print(f"  b - a = {100 * diff:+.4f} pp | per-token z = {z:+.1f}, p = {p_tok:.1e} | per-rollout permutation p {p_roll} ({hits}/{B})")


if __name__ == "__main__":
    main()
