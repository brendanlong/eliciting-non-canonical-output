#!/usr/bin/env python3
"""Figures for the writeup, from the metrics under out/ (download them from the HF dataset first).

    uv run python scripts/figures.py                # all figures into figures/, plus figures/data.json
    uv run python scripts/figures.py 1 4            # only figures 1 and 4

Figure 3 needs the rollout parquet (with logprobs) of the OLMo-3 Think and
RL-Zero-Math cells; figure 6 needs out/divergence/<run>/untruncated.parquet.
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
from noncanon import clustering
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
    "OLMo-3 Think": [("SFT", "think-sft", None), ("DPO", "think-dpo", "think-dpo-recommended"), ("RL final", "think-main", "think-main-recommended")],
    "OLMo-3 Instruct": [("SFT", "instruct-sft", "instruct-sft"), ("DPO", "instruct-dpo", "instruct-dpo"), ("RL final", "instruct-main", "instruct-main")],
    "OLMo-3 RL-Zero-Math": [("step 300", "rlzero-math-step300", None), ("step 2000", "rlzero-math", None)],
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


# --- 1. training stages, both sampling arms -------------------------------------------
def fig1() -> None:
    widths = [len(v) for v in LADDERS.values()]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), gridspec_kw={"width_ratios": widths}, sharey=True)
    DATA["fig1"] = {}
    for ax, (family, stages) in zip(axes, LADDERS.items()):
        xs = np.arange(len(stages))
        for j, (arm, run_idx, color, label) in enumerate([("untruncated", 1, BLUE, "temperature 1, top-p 1"), ("recommended", 2, ORANGE, "recommended settings")]):
            pts = []
            for i, st in enumerate(stages):
                run = st[run_idx]
                if run is None:
                    continue
                p, lo, hi, k, n = flag_pct(rows(run, arm))
                pts.append((i, p, lo, hi))
                DATA["fig1"][f"{family} | {st[0]} | {arm}"] = {"pct": p, "lo": lo, "hi": hi, "k": k, "n": n}
            if not pts:
                continue
            x = np.array([i for i, *_ in pts]) + (j - 0.5) * 0.38
            ax.bar(x, [p for _, p, *_ in pts], width=0.34, color=color, label=label, zorder=2)
            errbar(ax, x, [p for _, p, *_ in pts], [lo for *_, lo, _ in pts], [hi for *_, hi in pts])
            for xi, (_, p, _, hi) in zip(x, pts):
                ax.text(xi, hi + 1, f"{p:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(xs, [s[0] for s in stages], rotation=20, ha="right")
        ax.set_title(family)
        ax.set_xlim(-0.6, len(stages) - 0.4)
    axes[0].set_ylabel("rollouts with ≥1 non-canonical event (%)")
    axes[0].set_ylim(0, 72)
    axes[0].legend(loc="upper left")
    fig.suptitle("Non-canonical output by post-training stage (DAPO held-out, 500 rollouts per cell, Wilson 95%)", fontsize=10, color=INK, y=1.02)
    save(fig, "fig1_stages.png")


# --- 2. RL-Zero-Math over training steps ---------------------------------------------
def zero_steps() -> list[tuple[int, str]]:
    steps = [(2000, "rlzero-math")]
    for m in glob.glob(f"out/rlzero-math-step*/{D}/metrics/analysis.jsonl"):
        steps.append((int(re.search(r"step(\d+)", m).group(1)), m.split("/")[1]))
    return sorted(steps)


HABIT = " $($"  # the span RL-Zero-Math learns at the start of nearly every rollout


def flag_pct_excluding(run: str, text: str) -> tuple[float, float, float, int, int]:
    """Flagged fraction counting only spans whose emitted text is not ``text`` (byte fragments still count)."""
    rs = rows(run, "untruncated")
    events = [json.loads(l) for l in (Path("out") / run / D / "metrics" / "examples.jsonl").open()]
    keep = {(e["prompt_id"], e["sample"]) for e in events if e["file"].startswith("untruncated") and "".join(e["emitted"]) != text}
    k, n = sum((r["prompt_id"], r["sample"]) in keep for r in rs), len(rs)
    lo, hi = wilson(k, n)
    return 100 * k / n, 100 * lo, 100 * hi, k, n


def fig2() -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    DATA["fig2"] = {}
    habit_label = HABIT.replace(" ", "·").replace("$", r"\$")  # matplotlib reads bare $ as mathtext
    series = [(None, BLUE, "whole rollout", "o"), (1024, ORANGE, "within the first 1,024 tokens", "s"), ("habit", AQUA, f"whole rollout, excluding the {habit_label} span", "^")]
    for L, color, label, marker in series:
        xs, ps, los, his = [], [], [], []
        for step, run in zero_steps():
            p, lo, hi, k, n = flag_pct_excluding(run, HABIT) if L == "habit" else flag_pct(rows(run, "untruncated"), L)
            xs.append(step); ps.append(p); los.append(lo); his.append(hi)
            DATA["fig2"][f"step {step} | {'all' if L is None else L}"] = {"pct": p, "lo": lo, "hi": hi, "k": k, "n": n}
        ax.plot(xs, ps, color=color, lw=2, marker=marker, ms=7, mec=SURFACE, mew=1.5, label=label, zorder=3)
        errbar(ax, xs, ps, los, his)
        ax.annotate(f"{ps[-1]:.1f}%", (xs[-1], ps[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("RL step (Olmo-3-7B-RL-Zero-Math, RL from the base model)")
    ax.set_ylabel("rollouts with ≥1 non-canonical event (%)")
    ax.set_ylim(0, 70)
    ax.set_xlim(0, 2250)
    ax.legend(loc="upper left")
    ax.set_title("RL-Zero-Math over training (DAPO held-out, temperature 1)")
    save(fig, "fig2_rlzero_steps.png")


# --- 3. rank of the first token of each span -----------------------------------------
def fig3() -> None:
    from noncanon.metrics import Analyzer
    from noncanon.tail import BANDS, collect

    an = Analyzer("allenai/Olmo-3-7B-Think")
    cells = [("Think-SFT", "think-sft"), ("Think-DPO", "think-dpo"), ("Think RL final", "think-main"),
             ("RL-Zero step 300", "rlzero-math-step300"), ("RL-Zero step 2000", "rlzero-math"),
             ("Think-DPO, recommended settings", "think-dpo-recommended"), ("Think RL final, recommended settings", "think-main-recommended")]
    labels, shares, ns = [], [], []
    DATA["fig3"] = {}
    for label, run in cells:
        d = collect(an, Path("out") / run / D)
        for name, arr in [(label, d["all_first"])] + ([(label + ", bare-space spans excluded", d["other_first"])] if label == "Think-DPO" else []):
            sh = [100 * ((arr >= lo) & (arr <= hi)).mean() for lo, hi in BANDS]
            labels.append(name); shares.append(sh); ns.append(len(arr))
            DATA["fig3"][name] = {"spans": len(arr), "share_rank_1_2-3_4-10_gt10": sh, "sampled_tokens_rank_1": 100 * (d["ranks"] == 1).mean()}
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.grid(False)
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    band_names = ["rank 1 (the argmax)", "rank 2–3", "rank 4–10", "beyond the top 10"]
    for b, (name, color) in enumerate(zip(band_names, ORDINAL)):
        vals = np.array([s[b] for s in shares])
        ax.barh(y, vals, left=left, color=color, height=0.62, label=name, edgecolor=SURFACE, linewidth=1.5, zorder=2)
        for yi, l, v in zip(y, left, vals):
            if b in (0, 3) and v >= 6:
                ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=7.5, color=SURFACE if b == 3 else INK)
        left += vals
    for yi, n in zip(y, ns):
        ax.text(101, yi, f"n = {n}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of non-canonical spans by the rank of their first emitted token (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=4)
    ax.set_title("Where spans start in the next-token distribution (DAPO held-out)", pad=28)
    save(fig, "fig3_span_rank.png")


# --- 4. length control: flag within the first L tokens ------------------------------
def fig4() -> None:
    windows = [(256, "first 256"), (1024, "first 1,024"), (4096, "first 4,096"), (None, "whole rollout")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    DATA["fig4"] = {}
    for ax, (family, stages) in zip(axes, OLMO.items()):
        for i, (stage, run, _) in enumerate(stages):
            rs = rows(run, "untruncated")
            pts = [(x, *flag_pct(rs, L)) for x, (L, _) in enumerate(windows)]
            pts = [p for p in pts if p[5] >= 10]
            color = STAGE_COLOR.get(stage, ORDINAL[1 + i])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label=stage, zorder=3)
            errbar(ax, [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts])
            for x, p, lo, hi, k, n in pts:
                DATA["fig4"][f"{family} | {stage} | {windows[x][1]}"] = {"pct": p, "lo": lo, "hi": hi, "k": k, "n": n}
        ax.set_xticks(range(len(windows)), [w[1] for w in windows], rotation=20, ha="right")
        ax.set_title(family)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("rollouts with an event in the window,\namong rollouts that long (%)")
    axes[0].set_ylim(0, 65)
    fig.suptitle("Flagged fraction within a fixed token budget (DAPO held-out, temperature 1; points with <10 eligible rollouts omitted)", fontsize=10, color=INK, y=1.02)
    save(fig, "fig4_window.png")


# --- 5. clustering: propensity vs contagion --------------------------------------------
def fig5(B: int = 2000, seed: int = 0) -> None:
    cells = [(f"{fam.removeprefix('OLMo-3 ')} {st}" if fam != "OLMo-3 RL-Zero-Math" else f"RL-Zero {st}", run) for fam, stages in OLMO.items() for st, run, _ in stages]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    DATA["fig5"] = {}
    hz_obs, hz_base, ratios = [], [], []
    for label, run in cells:
        rs = rows(run, "untruncated")
        rng = np.random.default_rng(seed)
        n1, n2, exp2 = clustering.poisson_multi(rs)
        h_obs, h_base, h_n = clustering.hazard(rs, clustering.WINDOW, rng)
        g_obs, g_shuf, g_p, g_n = clustering.gap_test(rs, rng, B)
        texts, same = clustering.span_texts(Path("out") / run / D, "untruncated", rs)
        d_obs, d_shuf, d_p, d_n = clustering.gap_test(rs, rng, B, texts)
        hz_obs.append(100 * h_obs); hz_base.append(100 * h_base)
        ratios.append(((g_obs / g_shuf, g_p, g_n), (d_obs / d_shuf if d_n else float("nan"), d_p, d_n)))
        DATA["fig5"][label] = {"rollouts_ge1": n1, "rollouts_ge2": n2, "expected_ge2": exp2, "hazard_obs": h_obs, "hazard_base": h_base, "hazard_n": h_n,
                               "gap_obs": g_obs, "gap_shuf": g_shuf, "gap_p": g_p, "gaps": g_n, "same_text": same,
                               "gap_diff_obs": d_obs, "gap_diff_shuf": d_shuf, "gap_diff_p": d_p, "gaps_diff": d_n}
    x = np.arange(len(cells))
    a.bar(x - 0.19, hz_obs, width=0.36, color=BLUE, label="observed", zorder=2)
    a.bar(x + 0.19, hz_base, width=0.36, color=ORANGE, label="same window, same depth, other rollouts", zorder=2)
    for xi, v in zip(x, hz_obs):
        a.text(xi - 0.19, v + 0.6, f"{v:.0f}", ha="center", fontsize=7, color=INK2)
    a.set_xticks(x, [c[0] for c in cells], rotation=30, ha="right")
    a.set_ylabel("P(another event within 64 tokens | event) (%)")
    a.set_title("Hazard after an event", pad=26)
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    for j, (name, color, marker) in enumerate([("all consecutive pairs", BLUE, "o"), ("pairs with different span text", ORANGE, "s")]):
        vals = [r[j] for r in ratios]
        xs = x + (j - 0.5) * 0.3
        filled = [v[1] < 0.05 for v in vals]
        b.scatter(xs, [v[0] for v in vals], s=48, marker=marker, facecolors=[color if f else SURFACE for f in filled], edgecolors=color, linewidths=1.5, label=name, zorder=3)
    b.axhline(1, color=AXIS, lw=1, zorder=1)
    b.set_xticks(x, [c[0] for c in cells], rotation=30, ha="right")
    b.set_ylabel("median gap between consecutive events,\nobserved ÷ random placement")
    b.set_ylim(0, 1.3)
    b.set_title("Are consecutive events closer than chance? (filled: p < 0.05)", pad=26)
    b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.suptitle("Does one event make the next more likely? (OLMo-3 cells, DAPO held-out, temperature 1)", fontsize=10, color=INK, y=1.12)
    save(fig, "fig5_clustering.png")


# --- 6. what a span does downstream ------------------------------------------------------
def fig6() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    DATA["fig6"] = {}
    bin_labels = [f"{lo}" if lo == hi else (f"{lo}–{hi}" if hi < 10**9 else f"{lo}+") for lo, hi in DISTANCE_BINS]
    for col, (family, stages) in enumerate(OLMO.items()):
        top, bot = axes[0, col], axes[1, col]
        for i, (stage, run, _) in enumerate(stages):
            c = load_table(Path("out/divergence") / run, "untruncated")
            m = c["kind"] == "span"
            color = STAGE_COLOR.get(stage, ORDINAL[1 + i])
            differs, med_kl = [], []
            for lo, hi in DISTANCE_BINS:
                y = m & (c["distance_tokens"] >= lo) & (c["distance_tokens"] <= hi)
                differs.append(100 * (1 - c["top1_agree"][y].mean()) if y.any() else float("nan"))
                med_kl.append(float(np.median(c["kl_ab"][y])) if y.any() else float("nan"))
            n_spans = len(set(zip(c["prompt_id"][m], c["sample"][m], c["span_pos"][m])))
            top.plot(range(len(DISTANCE_BINS)), differs, color=color, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label=f"{stage} ({n_spans} spans)", zorder=3)
            near = m & (c["distance_tokens"] <= 16)
            layers = c["lens_layers"][:-1]
            lens = c["lens_kl"][near].mean(0)[: len(layers)]
            bot.plot(layers, lens, color=color, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label=stage, zorder=3)
            DATA["fig6"][f"{family} | {stage}"] = {"spans": n_spans, "top1_differs_pct_by_bin": dict(zip(bin_labels, differs)), "median_kl_by_bin": dict(zip(bin_labels, med_kl)),
                                                  "lens_kl_by_layer_within_16": dict(zip([int(l) for l in layers], [float(v) for v in lens])), "final_kl_within_16": float(c["kl_ab"][near].mean())}
        top.set_title(family)
        top.set_yscale("log")
        top.set_ylim(0.2, 60)
        top.set_yticks([0.3, 1, 3, 10, 30], ["0.3", "1", "3", "10", "30"])
        top.set_xticks(range(len(DISTANCE_BINS)), bin_labels)
        top.set_xlabel("tokens after the span")
        top.legend(loc="upper right")
        bot.set_xlabel("layer (logit lens, boundaries ≤ 16 tokens after the span)" if col == 1 else "layer")
        bot.set_ylim(0, 0.2)
        bot.legend(loc="upper left")
    axes[0, 0].set_ylabel("next-token argmax differs,\nemitted vs canonical prefix (%)")
    axes[1, 0].set_ylabel("mean logit-lens KL(emitted ‖ canonical)")
    fig.suptitle("What a non-canonical span does to the computation after it (teacher-forced, same text, emitted vs re-tokenized ids)", fontsize=10, color=INK, y=1.0)
    fig.tight_layout()
    save(fig, "fig6_divergence.png")


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6}


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
