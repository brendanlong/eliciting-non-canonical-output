"""Compare cells per the plan's analysis specification.

    uv run python -m noncanon.compare out/think-dpo/dapo_sample500 out/think-main/dapo_sample500
    uv run python -m noncanon.compare --table out/think-sft/dapo_sample500 out/think-dpo/dapo_sample500 ...

Rollouts are the unit (Brendan, 2026-09-04: tokens within a rollout are
not independent, so the per-token rate is reported but demoted). For each
pair of run directories (each holding ``metrics/analysis.jsonl``), in order:

1. Fraction of rollouts with at least one non-canonical event, with a
   Wilson 95% interval, and Fisher's exact test (two-sided) on the 2×2
   table of flagged rollouts (both from scipy.stats).
2. The same flag restricted to the first L tokens, over rollouts that
   reached L tokens (``SEQ_LENGTHS``), which conditions on length; a
   window with fewer than ``MIN_ELIGIBLE`` such rollouts in either cell
   is marked † (the p-value is exact but uninformative).
3. The pooled per-token rate with a rollout-bootstrap 95% CI, a per-token
   z on pooled counts (conditional-binomial form; ignores clustering) and a
   per-rollout permutation test on the difference of pooled rates; under
   the headline and the segmentation-only counting rules.

``--outcome`` restricts both cells to one outcome bucket (correct,
incorrect, parsed = correct + incorrect, unparsed, truncated).
``--table`` prints the rollout-level numbers of every run directory given.
Deterministic and order-invariant: each cell's bootstrap and the permutation
test get their own seeded generator; permutation p is "< 1/B" when no
permutation reached the observed difference.
"""

from __future__ import annotations

import argparse
import json
from math import erfc, sqrt
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, fisher_exact as _fisher

MIN_ELIGIBLE = 10  # a window with fewer eligible rollouts in either cell is reported as n/a, not tested

from noncanon.metrics import SEQ_LENGTHS, outcome

CONVENTIONS = {"headline": ("nc_events", "n_units"), "segmentation only": ("nc_canonical", "n_canonical")}
OUTCOMES = {"all": None, "parsed": {"correct", "incorrect"}, "correct": {"correct"}, "incorrect": {"incorrect"}, "unparsed": {"unparsed"}, "truncated": {"truncated"}}


def load_rows(run_dir: Path, arm: str | None = None, outcome_filter: str = "all") -> list[dict]:
    rows = [json.loads(line) for line in (run_dir / "metrics" / "analysis.jsonl").open()]
    if arm:
        rows = [r for r in rows if r["file"].startswith(arm)]
    files = {r["file"] for r in rows}
    assert len(files) == 1, f"{run_dir} mixes sampling arms {sorted(files)}; pass --arm"
    keep = OUTCOMES[outcome_filter]
    return rows if keep is None else [r for r in rows if outcome(r) in keep]


def load(run_dir: Path, arm: str | None = None, outcome_filter: str = "all") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = load_rows(run_dir, arm, outcome_filter)
    return {name: (np.array([r[nc] for r in rows]), np.array([r[n] for r in rows])) for name, (nc, n) in CONVENTIONS.items()}


# --- rollout level -----------------------------------------------------------------
def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    ci = binomtest(k, n).proportion_ci(method="wilson")
    return float(ci.low), float(ci.high)


def fisher_exact(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p for flagged counts k1/n1 vs k2/n2."""
    return float(_fisher([[k1, n1 - k1], [k2, n2 - k2]]).pvalue)


def flags(rows: list[dict], L: int | None = None) -> tuple[int, int]:
    """(flagged, eligible) rollouts: any event, or an event within the first L tokens among rollouts that reached L.

    Uses ``event_positions`` (ordinal of every event start, standalone
    fragments included), so metrics written before that field existed fail
    loudly instead of silently giving the older, fragment-free flags.
    """
    if L is None:
        return sum(r["nc_events"] > 0 for r in rows), len(rows)
    eligible = [r for r in rows if r["n_tokens"] >= L]
    return sum(bool(r["event_positions"]) and r["event_positions"][0] < L for r in eligible), len(eligible)


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.2f}%" if n else "n/a"


def rollout_line(name: str, rows: list[dict], L: int | None = None) -> str:
    k, n = flags(rows, L)
    lo, hi = wilson(k, n)
    return f"  {name:40s} {k:4d}/{n:<4d} = {pct(k, n):>8s}  Wilson 95% [{100 * lo:.2f}, {100 * hi:.2f}]"


# --- token level (secondary) ---------------------------------------------------
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


# --- CLI ----------------------------------------------------------------------------
def print_table(runs: list[Path], arm: str | None, outcome_filter: str) -> None:
    print("| cell | rollouts | with ≥1 event | Wilson 95% | " + " | ".join(f"within first {L} (of those ≥ {L})" for L in SEQ_LENGTHS) + " | per-token rate |")
    print("|---|--:|--:|---|" + "---|" * len(SEQ_LENGTHS) + "--:|")
    for run in runs:
        rows = load_rows(run, arm, outcome_filter)
        k, n = flags(rows)
        lo, hi = wilson(k, n)
        cells = []
        for L in SEQ_LENGTHS:
            kL, nL = flags(rows, L)
            cells.append(f"{kL}/{nL} = {100 * kL / nL:.1f}%" + ("" if nL >= MIN_ELIGIBLE else " †") if nL else "—")
        rate = sum(r["nc_events"] for r in rows) / max(1, sum(r["n_units"] for r in rows))
        frac = f"{k} ({100 * k / n:.1f}%)" if n else "0 (n/a)"
        print(f"| {run} | {n} | {frac} | {100 * lo:.1f}–{100 * hi:.1f}% | " + " | ".join(cells) + f" | {100 * rate:.4f}% |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+", help="two run directories (a b), or any number with --table")
    ap.add_argument("--arm", default=None, help="restrict to one sampling arm when a run directory holds several")
    ap.add_argument("--outcome", default="all", choices=list(OUTCOMES), help="restrict to one outcome bucket")
    ap.add_argument("--table", action="store_true", help="print rollout-level numbers for every run given, no tests")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.table:
        print_table(args.runs, args.arm, args.outcome)
        return
    assert len(args.runs) == 2, "pass exactly two run directories (or --table)"
    a, b = args.runs
    ra, rb = load_rows(a, args.arm, args.outcome), load_rows(b, args.arm, args.outcome)
    print(f"== rollouts with ≥1 non-canonical event (outcome: {args.outcome}) ==")
    print(rollout_line(str(a), ra))
    print(rollout_line(str(b), rb))
    (k1, n1), (k2, n2) = flags(ra), flags(rb)
    if not (n1 and n2):
        print("  one cell has no rollouts in this outcome bucket; nothing to test")
        return
    print(f"  b - a = {100 * (k2 / n2 - k1 / n1):+.2f} pp | Fisher exact p = {fisher_exact(k1, n1, k2, n2):.2e}")
    for L in SEQ_LENGTHS:
        (k1, n1), (k2, n2) = flags(ra, L), flags(rb, L)
        counts = f"{k1}/{n1} = {pct(k1, n1)} vs {k2}/{n2} = {pct(k2, n2)}"
        if not (n1 and n2):
            print(f"  within the first {L} tokens: {counts} | no eligible rollouts in a cell")
        elif min(n1, n2) >= MIN_ELIGIBLE:
            print(f"  within the first {L} tokens: {counts} | Fisher exact p = {fisher_exact(k1, n1, k2, n2):.2e}")
        else:
            print(f"  within the first {L} tokens: {counts} | Fisher exact p = {fisher_exact(k1, n1, k2, n2):.2e} † (fewer than {MIN_ELIGIBLE} eligible rollouts in a cell; exact but uninformative)")
    A, B_ = load(a, args.arm, args.outcome), load(b, args.arm, args.outcome)
    for conv in CONVENTIONS:
        print(f"== per-token rate, {conv} (secondary) ==")
        for name, cell in ((a, A), (b, B_)):
            nc, n = cell[conv]
            # Each cell's CI is seeded from its own data size and the seed, so
            # `compare a b` and `compare b a` print identical numbers.
            lo, hi = bootstrap_ci(nc, n, seed=[args.seed, len(nc), int(nc.sum())])
            print(f"  {str(name):40s} {100 * pooled(nc, n):.4f}%  95% CI [{100 * lo:.4f}, {100 * hi:.4f}]")
        z, p_tok = token_test(A[conv], B_[conv])
        diff, hits, B = permutation_test(A[conv], B_[conv], seed=[args.seed, 1])
        p_roll = f"< {1 / B:.5f}" if hits == 0 else f"= {hits / B:.4f}"
        print(f"  b - a = {100 * diff:+.4f} pp | per-token z = {z:+.1f}, p = {p_tok:.1e} | per-rollout permutation p {p_roll} ({hits}/{B})")


if __name__ == "__main__":
    main()
