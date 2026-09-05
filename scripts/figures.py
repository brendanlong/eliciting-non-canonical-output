#!/usr/bin/env python3
"""Figures for the writeup, from the metrics under out/ (download them from the HF dataset first).

    uv run python scripts/figures.py                # all figures into figures/, plus figures/data.json
    uv run python scripts/figures.py 1 4            # only figures 1 and 4
    uv run python scripts/figures.py 5              # figure 1 at the recommended sampling settings
    uv run python scripts/figures.py 6              # DAPO vs AIME, same model
    uv run python scripts/figures.py 7              # correct vs incorrect rollouts

Figure 3 needs the rollout parquet (with logprobs) of the OLMo-3 Think and
RL-Zero-Math cells; figure 4 needs out/divergence/<run>/untruncated.parquet.
Every number drawn is also written to figures/data.json.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from noncanon.compare import flags, load_rows, wilson
from noncanon.divergence import DISTANCE_BINS, load_table

matplotlib.use("Agg")

OUT = Path("figures")
D = "dapo_sample500"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]  # blue ramp steps 250 / 350 / 450 / 600
STAGE_COLOR = {"SFT": BLUE, "DPO": ORANGE, "RL final": AQUA, "RLVR (PPO)": AQUA, "3.1 RLVR (GRPO)": YELLOW}

# (stage label, run holding the temperature-1 arm, run holding the recommended-settings arm)
LADDERS = {
    "OLMo-3-7B Think": [("SFT", "think-sft", None), ("DPO", "think-dpo", "think-dpo-recommended"), ("RL final", "think-main", "think-main-recommended")],
    "OLMo-3-7B Instruct": [("SFT", "instruct-sft", "instruct-sft"), ("DPO", "instruct-dpo", "instruct-dpo"), ("RL final", "instruct-main", "instruct-main")],
    "OLMo-3-7B RL-Zero-Math": [("step 300", "rlzero-math-step300", None), ("step 2000", "rlzero-math", None)],
    "Tulu-3-8B": [("SFT", "tulu3-sft", "tulu3-sft"), ("DPO", "tulu3-dpo", "tulu3-dpo"), ("RLVR (PPO)", "tulu3-rlvr", "tulu3-rlvr"), ("3.1 RLVR (GRPO)", "tulu31-rlvr", "tulu31-rlvr")],
}
OLMO = {k: v for k, v in LADDERS.items() if k != "Tulu-3-8B"}
DATA: dict = {}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
})


def rows(run: str, arm: str, prompt_set: str = D) -> list[dict]:
    return load_rows(Path("out") / run / prompt_set, arm)


def flag_pct(rs: list[dict], L: int | None = None) -> tuple[float, float, float, int, int]:
    """(percent, lower, upper, flagged, eligible) with Wilson 95% bounds."""
    k, n = flags(rs, L)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0, 0
    lo, hi = wilson(k, n)
    return 100 * k / n, 100 * lo, 100 * hi, k, n


def errbar(ax, x, pct, lo, hi, **kw):
    ax.errorbar(x, pct, yerr=[[p - l for p, l in zip(pct, lo)], [h - p for p, h in zip(pct, hi)]], fmt="none", ecolor=MUTED, elinewidth=1, capsize=2, capthick=1, zorder=3, **kw)


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


# --- 1. training stages, length-matched --------------------------------------------
WINDOWS = [(256, "first 256 tokens"), (1024, "first 1,024 tokens"), (4096, "first 4,096 tokens")]


def fig1(arm: str = "untruncated") -> None:
    """Length-matched ladder; ``arm="recommended"`` uses each checkpoint's recommended settings (cells without one are dropped)."""
    idx = 1 if arm == "untruncated" else 2
    ladders = {fam: [st for st in stages if st[idx]] for fam, stages in LADDERS.items()}
    ladders = {fam: stages for fam, stages in ladders.items() if stages}
    widths = [len(v) for v in ladders.values()]
    fig, axes = plt.subplots(1, len(ladders), figsize=(12.5 * sum(widths) / 12, 3.8), gridspec_kw={"width_ratios": widths}, sharey=True)
    key = "fig1" if arm == "untruncated" else "fig1_recommended"
    DATA[key] = {}
    for ax, (family, stages) in zip(axes, ladders.items()):
        xs = np.arange(len(stages))
        for j, ((L, wlabel), color) in enumerate(zip(WINDOWS, ORDINAL[:3])):
            pts = []
            for i, (stage, *runs) in enumerate(stages):
                run = runs[idx - 1]
                p, lo, hi, k, n = flag_pct(rows(run, arm), L)
                DATA[key][f"{family} | {stage} | {wlabel}"] = {"pct": p, "lo": lo, "hi": hi, "k": k, "n": n}
                if n >= 10:
                    pts.append((i, p, lo, hi, n))
            if not pts:
                continue
            x = np.array([i for i, *_ in pts]) + (j - 1) * 0.27
            ax.bar(x, [p[1] for p in pts], width=0.25, color=color, label=wlabel, zorder=2)
            errbar(ax, x, [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts])
            for xi, (_, p, _, hi, n) in zip(x, pts):
                ax.text(xi, hi + 1, f"{p:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(xs, [s[0] for s in stages], rotation=20, ha="right")
        ax.set_title(family)
        ax.set_xlim(-0.6, len(stages) - 0.4)
    axes[0].set_ylabel("rollouts with a non-canonical event in the window,\namong rollouts at least that long (%)")
    axes[0].set_ylim(0, 62 if arm == "untruncated" else 25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3)
    settings = "temperature 1 / top-p 1" if arm == "untruncated" else "each checkpoint's recommended settings"
    fig.suptitle(f"Non-canonical output by post-training stage, length-matched (DAPO held-out, {settings}, Wilson 95%; windows with <10 eligible rollouts omitted)", fontsize=9.5, color=INK, y=1.09)
    save(fig, "fig1_stages.png" if arm == "untruncated" else "fig1_stages_recommended.png")


# --- 2. RL-Zero-Math over training steps ---------------------------------------------
def zero_steps() -> list[tuple[int, str]]:
    steps = [(2000, "rlzero-math")]
    for m in glob.glob(f"out/rlzero-math-step*/{D}/metrics/analysis.jsonl"):
        steps.append((int(re.search(r"step(\d+)", m).group(1)), m.split("/")[1]))
    return sorted(steps)


HABIT = " $($"  # the span RL-Zero-Math learns at the start of nearly every rollout


def flag_pct_excluding(run: str, text: str, L: int | None = None) -> tuple[float, float, float, int, int]:
    """Flagged fraction counting only events whose emitted text is not ``text`` (byte fragments still count),
    within the first L tokens among rollouts that reached L when L is given."""
    rs = rows(run, "untruncated")
    if L is not None:
        rs = [r for r in rs if r["n_tokens"] >= L]
    events = [json.loads(l) for l in (Path("out") / run / D / "metrics" / "examples.jsonl").open()]
    keep = {(e["prompt_id"], e["sample"]) for e in events
            if e["file"].startswith("untruncated") and "".join(e["emitted"]) != text and (L is None or e["pos"] < L)}
    k, n = sum((r["prompt_id"], r["sample"]) in keep for r in rs), len(rs)
    lo, hi = wilson(k, n)
    return 100 * k / n, 100 * lo, 100 * hi, k, n


DEPTH_BANDS = [(0, 1024), (1024, 4096), (4096, 8192), (8192, 10**9)]


def events_per_million_by_depth(run: str, text: str) -> list[tuple[float, int]]:
    """(events per million tokens, event count) in each depth band, counting events whose emitted text is not ``text``."""
    lengths = [r["n_tokens"] for r in rows(run, "untruncated")]
    events = [json.loads(l) for l in (Path("out") / run / D / "metrics" / "examples.jsonl").open()]
    events = [e for e in events if e["file"].startswith("untruncated") and "".join(e["emitted"]) != text]
    out = []
    for lo, hi in DEPTH_BANDS:
        n_ev = sum(lo <= e["pos"] < hi for e in events)
        toks = sum(min(max(l - lo, 0), hi - lo) for l in lengths)
        out.append((1e6 * n_ev / max(toks, 1), n_ev))
    return out


def fig2() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    DATA["fig2"] = {}
    habit_label = HABIT.replace(" ", "·").replace("$", r"\$")  # matplotlib reads bare $ as mathtext
    for ax, exclude in zip(axes[:2], (False, True)):
        for L, color, label, marker in [(None, BLUE, "whole rollout", "o"), (1024, ORANGE, "within the first 1,024 tokens", "s")]:
            xs, ps, los, his = [], [], [], []
            for step, run in zero_steps():
                p, lo, hi, k, n = flag_pct_excluding(run, HABIT, L) if exclude else flag_pct(rows(run, "untruncated"), L)
                xs.append(step); ps.append(p); los.append(lo); his.append(hi)
                DATA["fig2"][f"step {step} | {'all' if L is None else L} | {'excluding habit' if exclude else 'all spans'}"] = {"pct": p, "lo": lo, "hi": hi, "k": k, "n": n}
            ax.plot(xs, ps, color=color, lw=2, marker=marker, ms=7, mec=SURFACE, mew=1.5, label=label, zorder=3)
            errbar(ax, xs, ps, los, his)
            ax.annotate(f"{ps[-1]:.1f}%", (xs[-1], ps[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
        ax.set_title(f"excluding the {habit_label} span" if exclude else "all spans")
        ax.set_xlabel("RL step")
        ax.set_xlim(0, 2300)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("rollouts with ≥1 non-canonical event (%)")
    for ax in axes[:2]:
        ax.set_ylim(0, 70)
    ax = axes[2]
    steps = zero_steps()
    per_step = {step: events_per_million_by_depth(run, HABIT) for step, run in steps}
    band_labels = [f"{lo:,}–{hi:,}" if hi < 10**9 else f"{lo:,}+" for lo, hi in DEPTH_BANDS]
    for b, (label, color) in enumerate(zip(band_labels, ORDINAL)):
        ys = [per_step[step][b][0] for step, _ in steps]
        ax.plot([st for st, _ in steps], ys, color=color, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label=f"tokens {label}", zorder=3)
        for step, _ in steps:
            DATA["fig2"][f"step {step} | depth {label} | excluding habit"] = {"per_million": per_step[step][b][0], "events": per_step[step][b][1]}
    ax.set_title(f"excluding the {habit_label} span, by depth in the rollout")
    ax.set_xlabel("RL step")
    ax.set_xlim(0, 2300)
    ax.set_ylabel("non-canonical events per million tokens")
    ax.set_ylim(0, 230)
    ax.legend(loc="upper left", ncol=2)
    fig.suptitle("OLMo-3-7B RL-Zero-Math over training (RL from the base model; DAPO held-out, temperature 1)", fontsize=10, color=INK, y=1.02)
    save(fig, "fig2_rlzero_steps.png")


# --- 3. rank of the token that breaks canonicity -------------------------------------------
def fig3() -> None:
    from noncanon.metrics import Analyzer
    from noncanon.tail import BANDS, deviation_ranks

    an = Analyzer("allenai/Olmo-3-7B-Think")
    cells = [("Think-SFT", "think-sft"), ("Think-DPO", "think-dpo"), ("Think RL final", "think-main"),
             ("Instruct-SFT", "instruct-sft"), ("Instruct-DPO", "instruct-dpo"), ("Instruct RL final", "instruct-main"),
             ("RL-Zero step 300", "rlzero-math-step300"), ("RL-Zero step 2000", "rlzero-math")]
    labels, first_shares, dev_shares, ns = [], [], [], []
    DATA["fig3"] = {}
    for label, run in cells:
        first, dev, where = deviation_ranks(an, Path("out") / run / D, "untruncated")
        f_sh = [100 * ((first >= lo) & (first <= hi)).mean() for lo, hi in BANDS]
        d_sh = [100 * ((dev >= lo) & (dev <= hi)).mean() for lo, hi in BANDS]
        labels.append(label); first_shares.append(f_sh); dev_shares.append(d_sh); ns.append(len(first))
        DATA["fig3"][label] = {"spans": len(first), "deviating_token_rank_1_2-3_4-10_gt10": d_sh, "first_token_rank_1_2-3_4-10_gt10": f_sh,
                              "deviation_at_token": {k: where[k] for k in ("1", "2", "later")}}
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(labels))[::-1]
    band_names = ["rank 1 (the argmax)", "rank 2–3", "rank 4–10", "beyond the top 10"]
    for shares in (dev_shares,):
        ax.grid(False)
        left = np.zeros(len(labels))
        for b, (name, color) in enumerate(zip(band_names, ORDINAL)):
            vals = np.array([s[b] for s in shares])
            ax.barh(y, vals, left=left, color=color, height=0.62, label=name, edgecolor=SURFACE, linewidth=1.5, zorder=2)
            for yi, l, v in zip(y, left, vals):
                if b in (0, 3) and v >= 7:
                    ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=7.5, color=SURFACE if b == 3 else INK)
            left += vals
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("share of spans by the rank of the token that breaks canonicity (%)")
    for yi, n in zip(y, ns):
        ax.text(101, yi, f"n = {n}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, labels)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=4)
    ax.set_title("Rank of the token that breaks canonicity in the model's next-token distribution (DAPO held-out, temperature 1)", pad=28)
    save(fig, "fig3_span_rank.png")


# --- 4. what a span does downstream ------------------------------------------------------
def fig4() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    DATA["fig4"] = {}
    bin_labels = [f"{lo}" if lo == hi else (f"{lo}–{hi}" if hi < 10**9 else f"{lo}+") for lo, hi in DISTANCE_BINS]
    for col, (family, stages) in enumerate(OLMO.items()):
        top = axes[col]
        series = [(stage, run, STAGE_COLOR.get(stage, ORDINAL[1 + i]), None) for i, (stage, run, _) in enumerate(stages)]
        if family.endswith("RL-Zero-Math"):
            series.append((f"step 2000 without {HABIT.replace(' ', '·').replace('$', chr(92) + '$')}", "rlzero-math", AQUA, HABIT))
        for stage, run, color, exclude in series:
            c = load_table(Path("out/divergence") / run, "untruncated")
            m = c["kind"] == "span"
            if exclude is not None:
                events = [json.loads(l) for l in (Path("out") / run / D / "metrics" / "examples.jsonl").open()]
                habit = {(e["prompt_id"], e["sample"], e["pos"]) for e in events if "".join(e["emitted"]) == exclude}
                m = m & ~np.array([k in habit for k in zip(c["prompt_id"], c["sample"], c["span_pos"])])
            differs, med_kl = [], []
            for lo, hi in DISTANCE_BINS:
                y = m & (c["distance_tokens"] >= lo) & (c["distance_tokens"] <= hi)
                differs.append(100 * (1 - c["top1_agree"][y].mean()) if y.any() else float("nan"))
                med_kl.append(float(np.median(c["kl_ab"][y])) if y.any() else float("nan"))
            n_spans = len(set(zip(c["prompt_id"][m], c["sample"][m], c["span_pos"][m])))
            top.plot(range(len(DISTANCE_BINS)), differs, color=color, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label=f"{stage} ({n_spans} spans)", zorder=3)
            DATA["fig4"][f"{family} | {stage}"] = {"spans": n_spans, "top1_differs_pct_by_bin": dict(zip(bin_labels, differs)), "median_kl_by_bin": dict(zip(bin_labels, med_kl))}
        top.set_title(family)
        top.set_yscale("log")
        top.set_ylim(0.2, 120)
        top.set_yticks([0.3, 1, 3, 10, 30], ["0.3", "1", "3", "10", "30"])
        top.legend(loc="upper right", fontsize=7.5)
        top.set_xticks(range(len(DISTANCE_BINS)), bin_labels)
        top.set_xlabel("tokens after the span")
    axes[0].set_ylabel("next-token argmax differs,\nemitted vs canonical prefix (%)")
    fig.suptitle("How often the next-token argmax differs after a non-canonical span (teacher-forced, same text, emitted vs re-tokenized ids)", fontsize=10, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "fig4_divergence.png")


# --- 6. DAPO vs AIME, same model -------------------------------------------------------
AIME = "aime_2024_2025"
AIME_CELLS = [("OLMo-3-7B Think-DPO", "think-dpo"), ("OLMo-3-7B Think RL final", "think-main"), ("OLMo-3-7B RL-Zero-Math step 2000", "rlzero-math")]


def fig6() -> None:
    """Whole-rollout and length-window flag rates on DAPO (500 × 1) and AIME 2024/2025 (60 × 8), temperature 1."""
    DATA["fig6"] = {}
    windows = [(None, "whole rollout")] + WINDOWS
    fig, axes = plt.subplots(1, len(AIME_CELLS), figsize=(12.5, 3.8), sharey=True)
    for ax, (label, run) in zip(axes, AIME_CELLS):
        means = []
        for j, (prompt_set, name, color) in enumerate((("dapo_sample500", "DAPO held-out", BLUE), (AIME, "AIME 2024/2025", ORANGE))):
            rs = rows(run, "untruncated", prompt_set)
            mean_len = np.mean([r["n_tokens"] for r in rs])
            means.append(mean_len)
            pts = []
            for i, (L, wlabel) in enumerate(windows):
                pct, lo, hi, k, n = flag_pct(rs, L)
                DATA["fig6"][f"{label} | {name} | {wlabel}"] = {"pct": pct, "lo": lo, "hi": hi, "k": k, "n": n, "mean_tokens": float(mean_len)}
                if n >= 10:
                    pts.append((i, pct, lo, hi))
            x = np.array([i for i, *_ in pts]) + (j - 0.5) * 0.36
            ax.bar(x, [p[1] for p in pts], width=0.34, color=color, label=name, zorder=2)
            errbar(ax, x, [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts])
            for xi, (_, pct, _, hi) in zip(x, pts):
                ax.text(xi, hi + 1, f"{pct:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(range(len(windows)), [w[1] for w in windows], rotation=20, ha="right")
        ax.set_title(f"{label}\nmean length: DAPO {means[0]:,.0f}, AIME {means[1]:,.0f} tokens", fontsize=9)
    axes[0].set_ylabel("rollouts with a non-canonical event (%)")
    axes[0].set_ylim(0, 80)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=2)
    fig.suptitle("Same model, two prompt sets: whole-rollout rates differ, length-matched windows do not (DAPO 500 × 1, AIME 60 × 8; temperature 1 / top-p 1, Wilson 95%)", fontsize=9.5, color=INK, y=1.09)
    save(fig, "fig6_dapo_vs_aime.png")


# --- 7. correct vs incorrect rollouts ------------------------------------------------
def fig7() -> None:
    """Whole-rollout flag rate among correct and among incorrect rollouts, every stage, temperature 1 (buckets with <10 rollouts omitted)."""
    from noncanon.metrics import outcome

    DATA["fig7"] = {}
    ladders = {fam: [st for st in stages if st[1]] for fam, stages in LADDERS.items()}
    widths = [len(v) for v in ladders.values()]
    fig, axes = plt.subplots(1, len(ladders), figsize=(12.5, 3.8), gridspec_kw={"width_ratios": widths}, sharey=True)
    for ax, (family, stages) in zip(axes, ladders.items()):
        for j, (bucket, color) in enumerate((("correct", AQUA), ("incorrect", ORANGE))):
            pts = []
            for i, (stage, run, _) in enumerate(stages):
                rs = [r for r in rows(run, "untruncated") if outcome(r) == bucket]
                pct, lo, hi, k, n = flag_pct(rs)
                mean_len = float(np.mean([r["n_tokens"] for r in rs])) if rs else float("nan")
                DATA["fig7"][f"{family} | {stage} | {bucket}"] = {"pct": pct, "lo": lo, "hi": hi, "k": k, "n": n, "mean_tokens": mean_len}
                if n >= 10:
                    pts.append((i, pct, lo, hi, n))
            if not pts:
                continue
            x = np.array([i for i, *_ in pts]) + (j - 0.5) * 0.36
            ax.bar(x, [p[1] for p in pts], width=0.34, color=color, label=bucket, zorder=2)
            errbar(ax, x, [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts])
            for xi, (_, pct, _, hi, n) in zip(x, pts):
                ax.text(xi, hi + 1, f"{pct:.0f}\nn={n}", ha="center", va="bottom", fontsize=6.5, color=INK2, linespacing=0.9)
        ax.set_xticks(range(len(stages)), [s[0] for s in stages], rotation=20, ha="right")
        ax.set_title(family)
        ax.set_xlim(-0.6, len(stages) - 0.4)
    axes[0].set_ylabel("rollouts with a non-canonical event (%)")
    axes[0].set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=2)
    fig.suptitle("Correct vs incorrect rollouts, whole rollout (DAPO held-out, temperature 1 / top-p 1, Wilson 95%; buckets with <10 rollouts omitted)", fontsize=9.5, color=INK, y=1.09)
    save(fig, "fig7_correct_vs_incorrect.png")


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: lambda: fig1("recommended"), 6: fig6, 7: fig7}


def main() -> None:
    which = [int(a) for a in sys.argv[1:]] or sorted(FIGS)
    for n in which:
        FIGS[n]()
    OUT.mkdir(exist_ok=True)
    existing = json.loads((OUT / "data.json").read_text()) if (OUT / "data.json").exists() else {}
    existing.update(DATA)
    (OUT / "data.json").write_text(json.dumps(existing, indent=1))
    print("wrote", OUT / "data.json")


if __name__ == "__main__":
    main()
