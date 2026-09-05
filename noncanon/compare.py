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
``--table`` prints the rollout-level numbers of every run directory given
(``label=run_dir[:arm]``); ``--pairs`` prints the pairwise-test table for
``"A vs B=run_a[:arm],run_b[:arm][;outcome]"`` specs. Both are the
generated blocks of RESULTS.md, checked by ``scripts/check_results.py``.
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


# --- tables (the generated blocks in RESULTS.md) -------------------------------------
def parse_spec(spec: str, default_arm: str | None) -> tuple[str, Path, str | None]:
    """``[label=]run_dir[:arm]`` → (label, run_dir, arm); the label defaults to the run_dir."""
    label, sep, cell = spec.partition("=")
    if not sep:
        label, cell = "", label
    run, _, arm = cell.partition(":")
    return label or run, Path(run), arm or default_arm


def fmt_p(p: float) -> str:
    return f"{p:.0e}" if p < 0.001 else (f"{p:.4f}" if p < 0.01 else (f"{p:.3f}" if p < 0.1 else f"{p:.2f}"))


def without_span(rows: list[dict], run: Path, arm_name: str, outcome_filter: str, text: str) -> list[dict]:
    """Rows with every span whose emitted text equals ``text`` removed from the events (flags only; rates are not recomputed).

    Spans (examples.jsonl, raw-index order) and event ordinals are both monotone in position, so they pair by rank;
    a rollout whose counts differ (a fragment adjacent to a span) is left unchanged."""
    by_rollout = span_texts_by_rollout(run, arm_name, outcome_filter)
    out = []
    for r in rows:
        es = sorted(by_rollout.get((r["prompt_id"], r["sample"]), []), key=lambda e: e["pos"])
        positions = sorted(r["event_positions"])
        if len(es) != len(positions):
            out.append(r)
            continue
        kept = [pos for pos, e in zip(positions, es) if "".join(e["emitted"]) != text]
        out.append({**r, "event_positions": kept, "nc_events": len(kept), "n_units": r["n_units"]})
    return out


def print_table(specs: list[str], arm: str | None, outcome_filter: str, drop_span: str | None = None) -> None:
    rate_col = " | per-token rate |" if drop_span is None else " |"
    print("| cell | rollouts | with ≥1 event | Wilson 95% | " + " | ".join(f"within first {L} (of those ≥ {L})" for L in SEQ_LENGTHS) + rate_col)
    print("|---|--:|--:|---|" + "---|" * len(SEQ_LENGTHS) + ("--:|" if drop_span is None else ""))
    for spec in specs:
        label, run, cell_arm = parse_spec(spec, arm)
        rows = load_rows(run, cell_arm, outcome_filter)
        if drop_span is not None:
            rows = without_span(rows, run, rows[0]["file"].removesuffix(".parquet"), outcome_filter, drop_span)
        k, n = flags(rows)
        lo, hi = wilson(k, n)
        cells = []
        for L in SEQ_LENGTHS:
            kL, nL = flags(rows, L)
            cells.append(f"{kL}/{nL} = {100 * kL / nL:.1f}%" + ("" if nL >= MIN_ELIGIBLE else " †") if nL else "—")
        rate = sum(r["nc_events"] for r in rows) / max(1, sum(r["n_units"] for r in rows))
        frac = f"{k} ({100 * k / n:.1f}%)" if n else "0 (n/a)"
        tail = f" | {100 * rate:.4f}% |" if drop_span is None else " |"
        print(f"| {label} | {n} | {frac} | {100 * lo:.1f}–{100 * hi:.1f}% | " + " | ".join(cells) + tail)


def print_pairs(specs: list[str], arm: str | None, seed: int, default_outcome: str = "all") -> None:
    """One row per ``"A vs B=run_a[:arm],run_b[:arm][;outcome]"`` (an optional ``prefix: `` before ``A vs B`` is kept in the label).

    Columns: flagged fraction of each cell; Fisher exact p overall and within
    each window; the per-rollout permutation p on pooled per-token rates
    (headline convention). A window result whose direction differs from the
    overall one, or where the overall test is not significant, is annotated
    with the higher cell; the permutation p is annotated when its direction
    differs from the flag direction.
    """
    print("| pair (a vs b) | a | b | Fisher, all | " + " | ".join(f"first {L:,}" for L in SEQ_LENGTHS) + " | permutation (rates) |")
    print("|---|--:|--:|--:|" + "--:|" * len(SEQ_LENGTHS) + "--:|")
    for spec in specs:
        label, _, rest = spec.partition("=")
        cells, _, outcome_filter = rest.partition(";")
        outcome_filter = outcome_filter or default_outcome
        names = label.rsplit(": ", 1)[-1].split(" vs ")
        assert len(names) == 2, f"label must read 'A vs B': {label}"
        (_, run_a, arm_a), (_, run_b, arm_b) = (parse_spec(c, arm) for c in cells.split(","))
        ra, rb = load_rows(run_a, arm_a, outcome_filter), load_rows(run_b, arm_b, outcome_filter)
        (k1, n1), (k2, n2) = flags(ra), flags(rb)
        if not (n1 and n2):
            print(f"| {label} | — | — | no rollouts in this outcome bucket for one cell |" + " — |" * (len(SEQ_LENGTHS) + 1))
            continue
        p_all = fisher_exact(k1, n1, k2, n2)
        higher_all = k2 / n2 > k1 / n1
        out = [f"{100 * k1 / n1:.1f}%", f"{100 * k2 / n2:.1f}%", fmt_p(p_all)]
        for L in SEQ_LENGTHS:
            (a1, m1), (a2, m2) = flags(ra, L), flags(rb, L)
            if not (m1 and m2):
                out.append("—")
                continue
            p = fisher_exact(a1, m1, a2, m2)
            cell = fmt_p(p) + (" †" if min(m1, m2) < MIN_ELIGIBLE else "")
            higher = a2 / m2 > a1 / m1
            if p < 0.05 and (higher != higher_all or p_all >= 0.05):
                cell += f" ({names[higher]} higher)"
            out.append(cell)
        A = (np.array([r["nc_events"] for r in ra]), np.array([r["n_units"] for r in ra]))
        B_ = (np.array([r["nc_events"] for r in rb]), np.array([r["n_units"] for r in rb]))
        diff, hits, B = permutation_test(A, B_, seed=[seed, 1])
        perm = f"< {1 / B:.5f}" if hits == 0 else f"{hits / B:.4f}"
        if hits / B < 0.05 and (diff > 0) != higher_all:
            perm += f" ({names[diff > 0]} higher rate)"
        out.append(perm)
        print(f"| {label} | " + " | ".join(out) + " |")


def pieces_md(pieces: list[str]) -> str:
    """Token pieces as adjacent code spans joined by +, safe inside a GFM table cell."""
    return "+".join("`" + p.replace("|", "\\|").replace("\n", "⏎") + "`" for p in pieces)


def span_texts_by_rollout(run: Path, arm_name: str, outcome_filter: str) -> dict[tuple, list[dict]]:
    """Span examples (fragments excluded) per rollout of one arm, restricted to the outcome bucket."""
    keep = {(r["prompt_id"], r["sample"]) for r in load_rows(run, arm_name, outcome_filter)}
    out: dict[tuple, list[dict]] = {}
    for line in (run / "metrics" / "examples.jsonl").open():
        e = json.loads(line)
        if e["file"].startswith(arm_name) and e["canonical"] is not None and (e["prompt_id"], e["sample"]) in keep:
            out.setdefault((e["prompt_id"], e["sample"]), []).append(e)
    return out


def print_top_spans(specs: list[str], arm: str | None, n: int, outcome_filter: str) -> None:
    """The most common non-canonical spans of each cell: emitted pieces → canonical pieces, count, rollouts.

    Spans only: rollouts whose sole events are byte fragments are not counted here, so the
    rollout column can be slightly below the flagged count of the cell table."""
    from collections import Counter

    print("| cell | spans | rollouts with spans | most common spans: emitted → canonical (count, rollouts) |")
    print("|---|--:|--:|---|")
    for spec in specs:
        label, run, cell_arm = parse_spec(spec, arm)
        arm_name = load_rows(run, cell_arm)[0]["file"].removesuffix(".parquet")
        by_rollout = span_texts_by_rollout(run, arm_name, outcome_filter)
        counts, rollouts = Counter(), {}
        for key, es in by_rollout.items():
            for e in es:
                k = (tuple(e["emitted"]), tuple(e["canonical"]))
                counts[k] += 1
                rollouts.setdefault(k, set()).add(key)
        top = "; ".join(f"{pieces_md(list(em))} → {pieces_md(list(ca))} ({cnt}, {len(rollouts[(em, ca)])})" for (em, ca), cnt in counts.most_common(n))
        print(f"| {label} | {sum(counts.values())} | {len(by_rollout)} | {top} |")


# --- CLI ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="two run directories (a b); with --table any number of [label=]run_dir[:arm]; with --pairs 'A vs B=run_a,run_b[;outcome]' specs")
    ap.add_argument("--arm", default=None, help="restrict to one sampling arm when a run directory holds several")
    ap.add_argument("--outcome", default="all", choices=list(OUTCOMES), help="restrict to one outcome bucket")
    ap.add_argument("--table", action="store_true", help="print rollout-level numbers for every run given, no tests")
    ap.add_argument("--pairs", action="store_true", help="print the pairwise-test table for the pair specs given")
    ap.add_argument("--top-spans", type=int, default=0, metavar="N", help="print the N most common spans of every run given")
    ap.add_argument("--without-span", default=None, metavar="TEXT", help="with --table: drop every span whose emitted text equals TEXT before computing the flags")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.table:
        print_table(args.runs, args.arm, args.outcome, args.without_span)
        return
    if args.pairs:
        print_pairs(args.runs, args.arm, args.seed, args.outcome)
        return
    if args.top_spans:
        print_top_spans(args.runs, args.arm, args.top_spans, args.outcome)
        return
    assert len(args.runs) == 2, "pass exactly two run directories (or --table / --pairs)"
    a, b = map(Path, args.runs)
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
