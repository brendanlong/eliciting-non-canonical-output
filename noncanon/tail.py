"""Where in the next-token distribution do non-canonical spans come from?

    uv run python -m noncanon.tail --tokenizer allenai/Olmo-3-7B-Think out/think-main/dapo_sample500 ...

From the stored top-10 logprobs of each run: sharpness of the distribution
(mass beyond the top-k, entropy), the rank of the emitted token at every
position, the rank of the first token of each non-canonical span, and the
special case of spans that begin with a bare space token (canonical only
before a digit in cl100k-style tokenizers), where the deviation is the
token after the space. Byte fragments are excluded from the span
statistics (as are out-of-vocabulary ids). Descriptive only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from noncanon.metrics import Analyzer
from noncanon.records import iter_records

BANDS = ((1, 1), (2, 3), (4, 10), (11, 10**9))  # ">10" is "outside the stored top-k" when k = 10
KS = (1, 2, 3, 5, 10)


def band_shares(ranks: np.ndarray) -> str:
    if not len(ranks):
        return "n/a"
    return " / ".join(f"{100 * ((ranks >= lo) & (ranks <= hi)).mean():.0f}%" for lo, hi in BANDS)


def per_million(span_ranks: np.ndarray, ranks: np.ndarray) -> str:
    return " / ".join(f"{1e6 * ((span_ranks >= lo) & (span_ranks <= hi)).sum() / max(1, ((ranks >= lo) & (ranks <= hi)).sum()):.1f}" for lo, hi in BANDS)


def analyze_run(an: Analyzer, run_dir: Path) -> None:
    parquet = next(p for p in (run_dir / "untruncated.parquet", run_dir / "recommended.parquet") if p.exists())
    ranks, beyond = [], {}
    all_first, other_first, second_after_space, next_piece = [], [], [], Counter()
    entropies, beyond_k_all, beyond_k_first, p_top1 = [], [], [], []
    k_stored = None
    for rec in iter_records(parquet):
        tk = rec["topk_logprobs"]
        if not tk:
            continue
        # vLLM stores the top-k plus the sampled token when it fell outside the
        # top-k; use the smallest row length as k so every position uses the same k.
        k = min(len(r) for r in tk)
        k_stored = k if k_stored is None else min(k_stored, k)
        lps = np.array([sorted(r, reverse=True)[:k] for r in tk], dtype=np.float64)
        cum = np.cumsum(np.exp(lps), axis=1)  # raw logprobs: a full-vocabulary softmax
        for kk in KS:
            if kk <= k:
                beyond.setdefault(kk, []).append(1 - cum[:, kk - 1])
        p_top1.append(np.exp(lps[:, 0]))
        norm = lps - np.logaddexp.reduce(lps, axis=1, keepdims=True)
        entropies.append(-(np.exp(norm) * norm).sum(1))
        # Rank of the emitted token by vLLM's own ordering of the stored ids
        # (ties in bf16 logprobs would otherwise be broken optimistically).
        rank = np.array([ids.index(t) + 1 if t in ids else k + 1 for t, ids in zip(rec["token_ids"], rec["topk_ids"])])
        rank = np.minimum(rank, k + 1)
        ranks.append(rank)
        beyond_k_all.append(1 - cum[:, k - 1])
        a = an.analyze(rec)
        for sp in a["spans"]:
            if sp["canonical"] is None or sp["pos"] >= len(rank):  # byte fragments and OOV ids are not spans
                continue
            all_first.append(rank[sp["pos"]])
            beyond_k_first.append(1 - cum[sp["pos"], k - 1])
            if sp["emitted"][0].strip() == "" and len(sp["emitted"]) > 1 and sp["pos"] + 1 < len(rank):
                second_after_space.append(rank[sp["pos"] + 1])
                next_piece[sp["emitted"][1]] += 1
            else:
                other_first.append(rank[sp["pos"]])
    ranks = np.concatenate(ranks)
    all_first, other_first, second_after_space = np.array(all_first), np.array(other_first), np.array(second_after_space)
    ent = np.concatenate(entropies)
    print(f"\n{run_dir}: {len(ranks):,} positions, {len(all_first)} spans (byte fragments excluded), k = {k_stored}")
    print("  mean mass beyond top-k:      " + "  ".join(f"k={kk}: {np.concatenate(v).mean():.4f}" for kk, v in sorted(beyond.items())))
    print(f"  mass beyond top-{k_stored}: all positions {np.concatenate(beyond_k_all).mean():.4f}, at span-first positions {np.mean(beyond_k_first) if len(beyond_k_first) else float('nan'):.4f}")
    print(f"  entropy (top-{k_stored}, nats): mean {ent.mean():.4f}, p90 {np.percentile(ent, 90):.4f}, frac > 1: {100 * (ent > 1).mean():.2f}%")
    p1 = np.concatenate(p_top1)
    print(f"  p(top-1): mean {p1.mean():.4f}, frac < 0.5: {100 * (p1 < 0.5).mean():.2f}%, frac < 0.9: {100 * (p1 < 0.9).mean():.2f}%")
    print(f"  all sampled tokens at rank 1 / 2-3 / 4-10 / >10:            " + band_shares(ranks))
    print(f"  all spans, first-token rank at 1 / 2-3 / 4-10 / >10:        " + band_shares(all_first))
    print(f"  all spans per 1M samples in each rank band:                 " + per_million(all_first, ranks))
    print(f"  spans starting with a whitespace-only token: {len(second_after_space)}; rank of the token after it: " + band_shares(second_after_space))
    if next_piece:
        print("    most common token after it:", next_piece.most_common(6))
    print(f"  other spans: {len(other_first)}; first-token rank at 1 / 2-3 / 4-10 / >10: " + band_shares(other_first))
    print(f"  other spans per 1M samples in each rank band:               " + per_million(other_first, ranks))


def deviation_index(an: Analyzer, span_ids: list[int]) -> int:
    """Index of the first emitted token after which the span's tokens so far no longer re-encode to themselves.

    The first token of a minimal-diff span is often a legitimate prefix (a bare
    space before a digit, ` light` before `house`), so the decision that made
    the string non-canonical is usually the second token."""
    for k in range(len(span_ids)):
        if an.tok.encode(an.tok.decode(span_ids[: k + 1]), add_special_tokens=False) != span_ids[: k + 1]:
            return k
    return len(span_ids) - 1


def print_deviation_table(an: Analyzer, specs: list[str], default_arm: str | None) -> None:
    """Per cell: rank bands of the span's first token and of its deviating token, and where the deviation sits."""
    import json

    from noncanon.compare import load_rows, parse_spec

    print("| cell | spans | first-token rank: 1 / 2–3 / 4–10 / beyond top-10 | deviating-token rank: 1 / 2–3 / 4–10 / beyond top-10 | deviation at token 1 / 2 / later |")
    print("|---|--:|---|---|---|")
    for spec in specs:
        label, run, arm = parse_spec(spec, default_arm)
        arm_name = load_rows(run, arm)[0]["file"].removesuffix(".parquet")
        recs = {(r["prompt_id"], r["sample"]): r for r in iter_records(run / f"{arm_name}.parquet")}
        first, dev, where = [], [], Counter()
        for line in (run / "metrics" / "examples.jsonl").open():
            e = json.loads(line)
            if not e["file"].startswith(arm_name) or e["canonical"] is None:
                continue
            rec = recs[(e["prompt_id"], e["sample"])]
            ids, pos, L = rec["token_ids"], e["pos"], len(e["emitted"])
            k = deviation_index(an, ids[pos: pos + L])
            if pos + k >= len(rec["topk_ids"]):
                continue
            rank = lambda i: rec["topk_ids"][i].index(ids[i]) + 1 if ids[i] in rec["topk_ids"][i][:10] else 11
            first.append(rank(pos)); dev.append(rank(pos + k)); where["1" if k == 0 else "2" if k == 1 else "later"] += 1
        n = len(first)
        pct = lambda c: f"{100 * c / n:.0f}%" if n else "n/a"
        print(f"| {label} | {n} | {band_shares(np.array(first))} | {band_shares(np.array(dev))} | {pct(where['1'])} / {pct(where['2'])} / {pct(where['later'])} |")


def print_deviation_examples(an: Analyzer, specs: list[str], default_arm: str | None, n: int, seed: int) -> None:
    """Random spans per cell with the token that breaks canonicity marked, its rank and probability
    at that position (stored top-10), and the model's top choice there."""
    import json
    import random

    from noncanon.compare import load_rows, parse_spec

    show = lambda t: "`" + t.replace("|", "\\|").replace("\n", "⏎").replace("`", "'") + "`"
    print("| cell | context (…before the span) | span: emitted (breaking token in **bold**) → canonical | breaking token: rank, p | model's top choice there, p |")
    print("|---|---|---|---|---|")
    for spec in specs:
        label, run, arm = parse_spec(spec, default_arm)
        arm_name = load_rows(run, arm)[0]["file"].removesuffix(".parquet")
        recs = {(r["prompt_id"], r["sample"]): r for r in iter_records(run / f"{arm_name}.parquet")}
        spans = [json.loads(l) for l in (run / "metrics" / "examples.jsonl").open()]
        spans = [e for e in spans if e["file"].startswith(arm_name) and e["canonical"] is not None]
        rng = random.Random(seed)
        rng.shuffle(spans)
        for e in spans[:n]:
            rec = recs[(e["prompt_id"], e["sample"])]
            ids, pos, L = rec["token_ids"], e["pos"], len(e["emitted"])
            k = deviation_index(an, ids[pos: pos + L])
            i = pos + k
            if i >= len(rec["topk_ids"]):
                continue
            tk, lp = rec["topk_ids"][i], rec["topk_logprobs"][i]
            rank = tk.index(ids[i]) + 1 if ids[i] in tk[:10] else None
            p_tok = np.exp(lp[tk.index(ids[i])]) if ids[i] in tk else float("nan")
            best = int(np.argmax(lp[:10]))
            pieces = " ".join(("**" + show(x) + "**") if j == k else show(x) for j, x in enumerate(e["emitted"]))
            ctx = an.tok.decode(ids[max(0, pos - 20): pos])[-50:]
            print(f"| {label} | …{show(ctx)} | {pieces} → {' '.join(show(x) for x in e['canonical'])} | {rank if rank else '>10'}, {p_tok:.3f} | {show(an.piece(tk[best]))} {np.exp(lp[best]):.2f} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("runs", nargs="+", help="run directories holding <arm>.parquet; with --deviation-table, label=run_dir[:arm] specs")
    ap.add_argument("--deviation-table", action="store_true", help="table of first-token vs deviating-token ranks per cell")
    ap.add_argument("--deviation-examples", type=int, default=0, metavar="N", help="N random spans per cell with the breaking token, its rank and probability")
    ap.add_argument("--arm", default=None, help="default arm for the table/examples specs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    an = Analyzer(args.tokenizer, args.revision)
    if args.deviation_table:
        print_deviation_table(an, args.runs, args.arm)
        return
    if args.deviation_examples:
        print_deviation_examples(an, args.runs, args.arm, args.deviation_examples, args.seed)
        return
    for run_dir in args.runs:
        analyze_run(an, Path(run_dir))


if __name__ == "__main__":
    main()
