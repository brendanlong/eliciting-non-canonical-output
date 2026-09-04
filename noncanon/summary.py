"""High-level summary tables (Brendan's three comparisons, 2026-09-04).

    uv run python -m noncanon.summary ladder --arm untruncated "Think:SFT=out/think-sft/dapo_sample500" "Think:DPO=..." ...
    uv run python -m noncanon.summary pairs --labels DAPO AIME "Think RL final=out/think-main/dapo_sample500:untruncated,out/think-main/aime_2024_2025" ...

Per cell: rollouts, parsed (finished with an integer answer), correct,
truncated, mean emitted tokens, rollouts with ≥1 non-canonical event
(Wilson 95%), the same among correct rollouts, the flag within the first
1,024 tokens among rollouts that long, and the per-token rate (secondary).

``ladder`` rows are ``family:stage=run_dir[:arm]`` (``--arm`` is the default
arm for specs without one); Fisher exact p-values are given against the
previous stage of the same family and against the family's first stage,
followed by one omnibus chi-square per family across all its stages.
``pairs`` rows are ``label=run_dir_a[:arm],run_dir_b[:arm]``; p-values
compare a with b, and the sampling settings come from each cell's
``<arm>.meta.json``. A test where either cell has fewer than MIN_ELIGIBLE
eligible rollouts is still printed, marked with a dagger: the p-value is
exact but uninformative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from scipy.stats import chi2_contingency

from noncanon.compare import MIN_ELIGIBLE, fisher_exact, flags, load_rows, wilson
from noncanon.metrics import outcome

SPARSE = "†"


def parse_cell(spec: str, default_arm: str | None = None) -> tuple[Path, str | None]:
    run, _, arm = spec.partition(":")
    return Path(run), arm or default_arm


def stats(run: Path, arm: str | None) -> dict:
    rows = load_rows(run, arm)
    correct = [r for r in rows if outcome(r) == "correct"]
    parsed = [r for r in rows if outcome(r) in ("correct", "incorrect")]
    return {
        "run": run, "arm": rows[0]["file"].removesuffix(".parquet"),
        "n": len(rows), "parsed": len(parsed), "correct": len(correct), "truncated": sum(r["finish_reason"] == "length" for r in rows),
        "tokens": fmean(r["n_tokens"] for r in rows),
        "flag": flags(rows), "flag_correct": flags(correct), "flag_1024": flags(rows, 1024),
        "rate": sum(r["nc_events"] for r in rows) / max(1, sum(r["n_units"] for r in rows)),
    }


def settings(s: dict) -> str:
    meta = s["run"] / f"{s['arm']}.meta.json"
    if not meta.exists():
        raise FileNotFoundError(f"{meta}: needed for the sampling settings of {s['run']}")
    sp = json.load(meta.open())["sampling"]
    return f"T={sp['temperature']}, top-p={sp['top_p']}"


def frac(k: int, n: int, ci: bool = True) -> str:
    if n == 0:
        return "—"
    lo, hi = wilson(k, n)
    mark = SPARSE if n < MIN_ELIGIBLE else ""
    return f"{100 * k / n:.1f}% ({k}/{n}){mark}" + (f" [{100 * lo:.1f}–{100 * hi:.1f}]" if ci else "")


def fmt_p(p: float) -> str:
    return "< 1e-10" if p < 1e-10 else (f"{p:.1e}" if p < 0.001 else f"{p:.3f}")


def p_str(a: tuple[int, int], b: tuple[int, int]) -> str:
    if not (a[1] and b[1]):
        return "—"
    return fmt_p(fisher_exact(a[0], a[1], b[0], b[1])) + (SPARSE if min(a[1], b[1]) < MIN_ELIGIBLE else "")


def omnibus(cells: list[tuple[int, int]]) -> str:
    """Chi-square test of independence on the stages × (flagged, not flagged) table."""
    table = [[k, n - k] for k, n in cells if n]
    if len(table) < 2 or sum(r[0] for r in table) == 0:
        return "—"
    p = chi2_contingency(table).pvalue
    return fmt_p(p) + (SPARSE if min(n for _, n in cells if n) < MIN_ELIGIBLE else "")


def ladder(specs: list[str], default_arm: str | None) -> None:
    print("| family | stage | rollouts | parsed | correct | truncated | mean tokens | rollouts with ≥1 NC event [Wilson 95%] | p vs previous stage | p vs first stage | correct rollouts with ≥1 NC | p vs previous | p vs first | within first 1,024 tokens (of rollouts ≥ 1,024) | per-token rate |")
    print("|---|---|--:|--:|--:|--:|--:|---|--:|--:|---|--:|--:|---|--:|")
    per_family: dict[str, list[dict]] = {}
    for spec in specs:
        name, _, cell = spec.partition("=")
        family, _, stage = name.partition(":")
        s = stats(*parse_cell(cell, default_arm))
        seen = per_family.setdefault(family, [])
        prev, first = (seen[-1], seen[0]) if seen else (None, None)
        pv = lambda key, ref: p_str(ref[key], s[key]) if ref else "—"
        print(f"| {family} | {stage} | {s['n']} | {s['parsed']} | {s['correct']} | {s['truncated']} | {s['tokens']:,.0f} | {frac(*s['flag'])} | {pv('flag', prev)} | {pv('flag', first)} "
              f"| {frac(*s['flag_correct'])} | {pv('flag_correct', prev)} | {pv('flag_correct', first)} | {frac(*s['flag_1024'], ci=False)} | {100 * s['rate']:.4f}% |")
        seen.append(s)
    print()
    print("Omnibus chi-square across each family's stages (all rollouts / correct rollouts / within first 1,024 tokens): "
          + "; ".join(f"{fam} {omnibus([c['flag'] for c in cells])} / {omnibus([c['flag_correct'] for c in cells])} / {omnibus([c['flag_1024'] for c in cells])}" for fam, cells in per_family.items() if len(cells) > 1))


def pairs(specs: list[str], labels: tuple[str, str], default_arm: str | None) -> None:
    a_l, b_l = labels
    print(f"| model | settings ({a_l} / {b_l}) | rollouts ({a_l} / {b_l}) | parsed | correct | truncated | mean tokens | ≥1 NC event, {a_l} | ≥1 NC event, {b_l} | p | correct rollouts ≥1 NC, {a_l} | {b_l} | p | within first 1,024 tokens, {a_l} | {b_l} | p | per-token rate, {a_l} / {b_l} |")
    print("|---|---|--:|--:|--:|--:|--:|---|---|--:|---|---|--:|---|---|--:|--:|")
    for spec in specs:
        name, _, cells = spec.partition("=")
        a, b = (stats(*parse_cell(c, default_arm)) for c in cells.split(","))
        both = lambda key: f"{a[key]} / {b[key]}"
        print(f"| {name} | {settings(a)} / {settings(b)} | {both('n')} | {both('parsed')} | {both('correct')} | {both('truncated')} | {a['tokens']:,.0f} / {b['tokens']:,.0f} "
              f"| {frac(*a['flag'])} | {frac(*b['flag'])} | {p_str(a['flag'], b['flag'])} | {frac(*a['flag_correct'], ci=False)} | {frac(*b['flag_correct'], ci=False)} | {p_str(a['flag_correct'], b['flag_correct'])} "
              f"| {frac(*a['flag_1024'], ci=False)} | {frac(*b['flag_1024'], ci=False)} | {p_str(a['flag_1024'], b['flag_1024'])} | {100 * a['rate']:.4f}% / {100 * b['rate']:.4f}% |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default=None, help="default sampling arm for cells given without a :arm suffix")
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("ladder")
    l.add_argument("cells", nargs="+", help="family:stage=run_dir[:arm]")
    p = sub.add_parser("pairs")
    p.add_argument("--labels", nargs=2, default=("a", "b"))
    p.add_argument("cells", nargs="+", help="label=run_dir_a[:arm],run_dir_b[:arm]")
    args = ap.parse_args()
    if args.cmd == "ladder":
        ladder(args.cells, args.arm)
    else:
        pairs(args.cells, tuple(args.labels), args.arm)


if __name__ == "__main__":
    main()
