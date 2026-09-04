"""Does one non-canonical event make another more likely? (Brendan, 2026-09-04)

    uv run python -m noncanon.clustering --arm untruncated "Think-DPO=out/think-dpo/dapo_sample500" ...

Two things can make events cluster in rollouts: some rollouts are prone
(propensity), or an event itself raises the chance of the next one
(contagion). Per cell, from ``metrics/analysis.jsonl`` (event ordinals) and
``metrics/examples.jsonl`` (span text):

- rollouts with ≥1 and ≥2 events, and the number with ≥2 expected if events
  were Poisson at the cell's per-token rate given each rollout's length
  (excess = propensity or contagion);
- P(another event within W tokens | an event), against a depth-matched
  baseline: the same window at the same position in other rollouts of the
  cell (excess = propensity or contagion, length-controlled);
- the median gap between consecutive events within a rollout, against
  the same number of events placed uniformly at random in the same rollout
  (shuffle test, B permutations; keeps each rollout's count and length, so
  a smaller observed gap is contagion, not propensity);
- the fraction of consecutive events whose emitted span text repeats the
  previous one, and the gap test restricted to pairs with different text.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import exp
from pathlib import Path

import numpy as np

from noncanon.compare import load_rows, parse_spec

WINDOW = 64


def poisson_multi(rows: list[dict]) -> tuple[int, int, float]:
    """(rollouts with ≥1, with ≥2, expected with ≥2 given ≥1 under a Poisson at the cell's rate)."""
    rate = sum(r["nc_events"] for r in rows) / sum(r["n_units"] for r in rows)
    flagged = [r for r in rows if r["nc_events"] > 0]
    expected = 0.0
    for r in flagged:
        lam = rate * r["n_units"]
        expected += 1 - lam * exp(-lam) / (1 - exp(-lam)) if lam > 0 else 0.0
    return len(flagged), sum(len(r["event_positions"]) >= 2 for r in flagged), expected


def hazard(rows: list[dict], W: int, rng: np.random.Generator, draws: int = 20) -> tuple[float, float, int]:
    """P(event in (t, t+W] | event at t) and the same window at depth t in other rollouts of the cell."""
    obs, base, n = 0, 0.0, 0
    lengths = np.array([r["n_tokens"] for r in rows])
    for i, r in enumerate(rows):
        for t in r["event_positions"]:
            if t + W > r["n_tokens"]:
                continue
            n += 1
            obs += any(t < p <= t + W for p in r["event_positions"])
            eligible = np.flatnonzero((lengths >= t + W) & (np.arange(len(rows)) != i))
            if len(eligible):
                picks = rng.choice(eligible, size=min(draws, len(eligible)), replace=False)
                base += np.mean([any(t < p <= t + W for p in rows[j]["event_positions"]) for j in picks])
    return (obs / n if n else float("nan")), (base / n if n else float("nan")), n


def gap_test(rows: list[dict], rng: np.random.Generator, B: int, texts: dict | None = None) -> tuple[float, float, float, int]:
    """(observed median gap, shuffled median gap, one-sided p that shuffles give a median ≤ observed, gaps)."""
    multi = [r for r in rows if len(r["event_positions"]) >= 2]
    def gaps(positions_by_rollout):
        out = []
        for r, ps in positions_by_rollout:
            ps = sorted(ps)
            for a, b in zip(ps, ps[1:]):
                if texts is None or texts.get((r["prompt_id"], r["sample"], a)) != texts.get((r["prompt_id"], r["sample"], b)):
                    out.append(b - a)
        return out
    observed = gaps([(r, r["event_positions"]) for r in multi])
    if not observed:
        return float("nan"), float("nan"), float("nan"), 0
    obs_med = float(np.median(observed))
    shuffled, hits = [], 0
    for _ in range(B):
        g = gaps([(r, rng.choice(r["n_tokens"], size=len(r["event_positions"]), replace=False)) for r in multi])
        med = float(np.median(g)) if g else float("nan")
        shuffled.append(med)
        hits += med <= obs_med
    return obs_med, float(np.nanmedian(shuffled)), hits / B, len(observed)


def span_texts(run_dir: Path, arm: str, rows: list[dict]) -> tuple[dict, float]:
    """Map (prompt_id, sample, event ordinal) → emitted text, and the fraction of consecutive events repeating the previous text."""
    by_rollout = defaultdict(list)
    for line in (run_dir / "metrics" / "examples.jsonl").open():
        e = json.loads(line)
        if e["file"].startswith(arm):
            by_rollout[(e["prompt_id"], e["sample"])].append(e)
    texts, same, pairs = {}, 0, 0
    for r in rows:
        es = sorted(by_rollout.get((r["prompt_id"], r["sample"]), []), key=lambda e: e["pos"])
        # examples are in raw-index order and event_positions in ordinal order; both list every event, so zip by rank
        if len(es) != len(r["event_positions"]):
            continue
        for ordinal, e in zip(sorted(r["event_positions"]), es):
            texts[(r["prompt_id"], r["sample"], ordinal)] = "".join(e["emitted"])
        for a, b in zip(es, es[1:]):
            pairs += 1
            same += "".join(a["emitted"]) == "".join(b["emitted"])
    return texts, (same / pairs if pairs else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells", nargs="+", help="label=run_dir[:arm]")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("-B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    W = args.window
    print(f"| cell | rollouts ≥1 / ≥2 events | ≥2 expected if Poisson | P(another within {W} tokens ∣ event): observed / depth-matched baseline (n) | median gap, observed / shuffled (p; n gaps) | consecutive events with the same text | median gap, different text only: observed / shuffled (p; n) |")
    print("|---|--:|--:|---|---|--:|---|")
    for spec in args.cells:
        label, run, arm = parse_spec(spec, args.arm)
        rows = load_rows(run, arm)
        rng = np.random.default_rng(args.seed)
        n1, n2, exp2 = poisson_multi(rows)
        h_obs, h_base, h_n = hazard(rows, W, rng)
        g_obs, g_shuf, g_p, g_n = gap_test(rows, rng, args.B)
        texts, same = span_texts(run, arm or rows[0]["file"].removesuffix(".parquet"), rows)
        d_obs, d_shuf, d_p, d_n = gap_test(rows, rng, args.B, texts)
        print(f"| {label} | {n1} / {n2} | {exp2:.1f} | {100 * h_obs:.1f}% / {100 * h_base:.1f}% ({h_n}) | {g_obs:.0f} / {g_shuf:.0f} (p = {g_p:.3f}; {g_n}) | {100 * same:.1f}% | {d_obs:.0f} / {d_shuf:.0f} (p = {d_p:.3f}; {d_n}) |")


if __name__ == "__main__":
    main()
