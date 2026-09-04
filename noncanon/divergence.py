"""Downstream divergence: what a non-canonical span does to the model's later computation.

    uv run python -m noncanon.divergence fetch think-dpo                       # records + spans from the HF dataset into out/
    uv run python -m noncanon.divergence run --model allenai/Olmo-3-7B-Think-DPO out/think-dpo/dapo_sample500
    uv run python -m noncanon.divergence summarize out/divergence/think-dpo ...

For each non-canonical span two teacher-forced forward passes are run over
the same text with the model's real context: the prompt and every emitted
token before the span, verbatim, in both. They differ only from the span
on: A continues with the token ids the model emitted (the span and up to
``--after`` following tokens), B with the canonical re-tokenization of
that same text (checked to splice onto the prefix exactly as the
canonical tokenization of the whole text would). At every byte boundary
after the span that both share, we record the KL divergence between the
two next-token distributions (final logits, and the logit lens at a
subset of layers) and the cosine distance of the residual stream at those
layers. Spans with another event inside the following window are skipped
so the measured divergence is the span's own.

Rows are written to ``out/divergence/<run>/<arm>.parquet``. Beyond the
span the two sequences differ only in position index (the canonical span
usually has a different token count), which sets a small floor far from
the span.

``contagion``: for consecutive spans in a rollout, the log-probability of
the second span's first emitted token under the context with the first
span as emitted (A) versus canonically re-tokenized (B); Δ > 0 means the
first span's tokenization makes the second more likely.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from noncanon.metrics import Analyzer, cumulative_ends
from noncanon.records import iter_records

LENS_LAYERS = (4, 8, 12, 16, 20, 24, 28)  # plus the final layer, added at run time
DISTANCE_BINS = ((0, 0), (1, 4), (5, 16), (17, 64), (65, 256), (257, 10**9))  # tokens after the span end; 0 = at the span end
SCHEMA = pa.schema([
    ("kind", pa.string()), ("prompt_id", pa.string()), ("sample", pa.int32()), ("span_pos", pa.int32()), ("shape", pa.string()),
    ("span_emitted", pa.int16()), ("span_canonical", pa.int16()), ("prefix_tokens", pa.int32()), ("prefix_truncated", pa.bool_()),
    ("distance_tokens", pa.int32()), ("distance_bytes", pa.int32()),
    ("kl_ab", pa.float32()), ("kl_ba", pa.float32()), ("top1_agree", pa.bool_()), ("p_top1_a_under_b", pa.float32()),
    ("lens_layers", pa.list_(pa.int16())), ("lens_kl", pa.list_(pa.float32())), ("hidden_cos_dist", pa.list_(pa.float32())),
])


# --- alignment (CPU) ----------------------------------------------------------------
def canonical_suffix(an: Analyzer, prefix: list[int], max_len: int = 64) -> list[int]:
    """The longest suffix of ``prefix`` (up to ``max_len`` tokens) that re-encodes to itself."""
    for k in range(min(max_len, len(prefix)), 0, -1):
        seg = prefix[-k:]
        if an.tok.encode(an.tok.decode(seg), add_special_tokens=False) == seg:
            return seg
    return []


def canonical_tail(an: Analyzer, prefix: list[int], tail: list[int]) -> list[int] | None:
    """Canonical tokenization of ``tail``'s text that splices onto ``prefix`` exactly as the
    canonical tokenization of the whole text would; None if the join would merge across."""
    suffix = canonical_suffix(an, prefix)
    if not suffix:
        return None
    canon = an.tok.encode(an.tok.decode(tail), add_special_tokens=False)
    joined = an.tok.encode(an.tok.decode(suffix + tail), add_special_tokens=False)
    if joined != suffix + canon:
        return None
    if b"".join(an.token_bytes(tail)) != b"".join(an.token_bytes(canon)):
        return None  # the tail ends inside a multi-byte character; decode() would pad it and misalign byte offsets
    return canon


def shared_boundaries(bytes_a: list[bytes], bytes_b: list[bytes], after_byte: int) -> list[tuple[int, int, int]]:
    """(byte offset, index in a, index in b) of every token end at or beyond ``after_byte`` present in both tokenizations.

    The end of the span itself is included (distance 0): the prediction made
    there is the first one conditioned on the span."""
    ends_a = {e: i for i, e in enumerate(cumulative_ends(bytes_a))}
    ends_b = {e: i for i, e in enumerate(cumulative_ends(bytes_b))}
    return [(e, ends_a[e], ends_b[e]) for e in sorted(ends_a) if e >= after_byte and e in ends_b]


def build_pair(an: Analyzer, prompt_ids: list[int], ids: list[int], pos: int, span_len: int, after: int) -> dict | None:
    """Token sequences A (emitted) and B (canonical from the span on) for one span, with the shared boundaries after it.

    Both keep the prompt and every emitted token before the span verbatim."""
    end = min(len(ids), pos + span_len + after)
    prefix, tail = ids[:pos], ids[pos:end]
    canon = canonical_tail(an, prefix, tail)
    if canon is None or canon == tail:
        return None
    bytes_a, bytes_b = an.token_bytes(tail), an.token_bytes(canon)
    span_end_byte = sum(len(b) for b in bytes_a[:span_len])
    bounds = shared_boundaries(bytes_a, bytes_b, span_end_byte)
    if not bounds:
        return None
    offset = len(prompt_ids) + len(prefix)
    return {
        "a": prompt_ids + prefix + tail, "b": prompt_ids + prefix + canon, "offset": offset, "span_end_a": span_len,
        "span_end_byte": span_end_byte, "prefix_tokens": len(prefix), "prefix_truncated": False, "bounds": bounds,
    }


# --- model (GPU) --------------------------------------------------------------------
class Model:
    def __init__(self, name: str, revision: str):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype, attn_implementation="sdpa").to(self.device).eval()
        n_layers = self.model.config.num_hidden_layers
        self.layers = [l for l in LENS_LAYERS if l < n_layers] + [n_layers]

    def hidden(self, ids: list[int]):
        """All hidden states of the base model (the last one already normed); no full-vocabulary logits."""
        torch = self.torch
        with torch.no_grad():
            return self.model.model(input_ids=torch.tensor([ids], device=self.device), output_hidden_states=True).hidden_states

    def forward(self, ids: list[int], positions: list[int]) -> tuple:
        """Final log-probs, logit-lens log-probs per selected layer, and residual states at ``positions``."""
        torch = self.torch
        n_layers = self.model.config.num_hidden_layers
        with torch.no_grad():
            hs = self.hidden(ids)
            idx = torch.tensor(positions, device=self.device)
            hidden = [hs[l][0, idx] for l in self.layers]
            # hs[-1] is already passed through the model's norm; applying it again would double-normalise.
            lens = [torch.log_softmax(self.model.lm_head(h if l == n_layers else self.model.model.norm(h)).float(), -1) for l, h in zip(self.layers, hidden)]
            final = lens[-1]
        return final, lens, hidden


def measure(m: Model, pair: dict, meta: dict, kind: str) -> list[dict]:
    torch = m.torch
    pos_a = [pair["offset"] + ia for _, ia, _ in pair["bounds"]]
    pos_b = [pair["offset"] + ib for _, _, ib in pair["bounds"]]
    fa, la, ha = m.forward(pair["a"], pos_a)
    fb, lb, hb = m.forward(pair["b"], pos_b)
    kl = lambda p, q: (p.exp() * (p - q)).sum(-1)
    kl_ab, kl_ba = kl(fa, fb).cpu().numpy(), kl(fb, fa).cpu().numpy()
    top_a, top_b = fa.argmax(-1), fb.argmax(-1)
    p_top1 = fb.gather(-1, top_a[:, None])[:, 0].exp().cpu().numpy()
    lens_kl = torch.stack([kl(x, y) for x, y in zip(la, lb)], 1).cpu().numpy()
    cos = torch.stack([1 - torch.nn.functional.cosine_similarity(x.float(), y.float(), dim=-1) for x, y in zip(ha, hb)], 1).cpu().numpy()
    rows = []
    for j, (byte, ia, _) in enumerate(pair["bounds"]):
        rows.append({
            "kind": kind, "prompt_id": meta["prompt_id"], "sample": int(meta["sample"]), "span_pos": int(meta["pos"]), "shape": meta["shape"],
            "span_emitted": len(meta["emitted"]), "span_canonical": len(meta["canonical"]), "prefix_tokens": pair["prefix_tokens"], "prefix_truncated": pair["prefix_truncated"],
            "distance_tokens": ia + 1 - pair["span_end_a"], "distance_bytes": byte - pair["span_end_byte"],  # 0 at the span's last token
            "kl_ab": float(kl_ab[j]), "kl_ba": float(kl_ba[j]), "top1_agree": bool((top_a[j] == top_b[j]).item()), "p_top1_a_under_b": float(p_top1[j]),
            "lens_layers": m.layers, "lens_kl": lens_kl[j].tolist(), "hidden_cos_dist": cos[j].tolist(),
        })
    return rows


def run_cell(run_dir: Path, arm: str, model_name: str, revision: str, out_dir: Path, after: int, max_spans: int, per_rollout: int, seed: int) -> None:
    an = Analyzer(model_name, revision)
    records = {(r["prompt_id"], r["sample"]): r for r in iter_records(run_dir / f"{arm}.parquet")}
    spans = defaultdict(list)
    for line in (run_dir / "metrics" / "examples.jsonl").open():
        e = json.loads(line)
        if e["file"].startswith(arm):
            spans[(e["prompt_id"], e["sample"])].append(e)
    rng = random.Random(seed)
    candidates, skipped = [], Counter()
    for key, es in spans.items():
        es.sort(key=lambda e: e["pos"])
        starts = [e["pos"] for e in es]
        kept = 0
        for i, e in enumerate(es):
            if e["canonical"] is None:
                skipped["fragment"] += 1
                continue
            lo, hi = e["pos"], e["pos"] + len(e["emitted"]) + after
            if any(lo <= s < hi for j, s in enumerate(starts) if j != i):
                skipped["another event in the following window"] += 1
                continue
            if kept >= per_rollout:
                skipped["per-rollout cap"] += 1
                continue
            candidates.append((key, e))
            kept += 1
    rng.shuffle(candidates)
    candidates = candidates[:max_spans]
    m = Model(model_name, revision)
    rows = []
    for key, e in candidates:
        rec = records[key]
        pair = build_pair(an, list(rec["prompt_token_ids"]), list(rec["token_ids"]), e["pos"], len(e["emitted"]), after)
        if pair is None:
            skipped["canonical tail does not splice / no shared boundary"] += 1
            continue
        rows.extend(measure(m, pair, e, "span"))
    n_spans = len({(r["prompt_id"], r["sample"], r["span_pos"]) for r in rows})
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out_dir / f"{arm}.parquet")
    meta = {"model": model_name, "revision": revision, "run_dir": str(run_dir), "arm": arm, "prefix": "full", "after": after, "seed": seed,
            "candidate_spans": len(candidates), "measured_spans": n_spans, "skipped": dict(skipped), "lens_layers": m.layers, "rows": len(rows)}
    (out_dir / f"{arm}.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


# --- summary (CPU) ------------------------------------------------------------------
def load_table(d: Path, arm: str) -> dict[str, np.ndarray]:
    """Rows of one cell, duplicates dropped."""
    t = pq.read_table(d / f"{arm}.parquet")
    keys = list(zip(*(t.column(c).to_pylist() for c in ("kind", "prompt_id", "sample", "span_pos", "distance_tokens"))))
    seen, keep = set(), []
    for i, k in enumerate(keys):
        if k not in seen:
            seen.add(k)
            keep.append(i)
    t = t.take(keep)
    cols = {c: t.column(c).to_numpy(zero_copy_only=False) for c in ("kind", "prompt_id", "sample", "span_pos", "distance_tokens", "kl_ab", "top1_agree")}
    cols["lens_layers"] = t.column("lens_layers")[0].as_py() if len(t) else []
    cols["lens_kl"] = np.array(t.column("lens_kl").to_pylist(), dtype=float) if len(t) else np.zeros((0, 0))
    cols["hidden_cos_dist"] = np.array(t.column("hidden_cos_dist").to_pylist(), dtype=float) if len(t) else np.zeros((0, 0))
    return cols


def n_spans(c: dict[str, np.ndarray], mask: np.ndarray) -> int:
    return len(set(zip(c["prompt_id"][mask], c["sample"][mask], c["span_pos"][mask])))


def summarize(dirs: list[str], arm: str) -> None:
    print("| cell | kind | spans | boundaries | " + " | ".join(f"KL(A‖B) at {lo}–{hi if hi < 10**9 else '∞'} tokens after: mean / median / top-1 differs" for lo, hi in DISTANCE_BINS) + " |")
    print("|---|---|--:|--:|" + "---|" * len(DISTANCE_BINS))
    for spec in dirs:
        label, _, d = spec.rpartition("=")
        d = Path(d); label = label or d.name
        c = load_table(d, arm)
        for kind in ("span",):
            m = c["kind"] == kind
            if not m.any():
                continue
            cells = []
            for lo, hi in DISTANCE_BINS:
                y = m & (c["distance_tokens"] >= lo) & (c["distance_tokens"] <= hi)
                cells.append(f"{c['kl_ab'][y].mean():.3f} / {np.median(c['kl_ab'][y]):.4f} / {100 * (1 - c['top1_agree'][y].mean()):.1f}% (n={y.sum()})" if y.any() else "—")
            print(f"| {label} | {kind} | {n_spans(c, m)} | {m.sum()} | " + " | ".join(cells) + " |")


def summarize_lens(dirs: list[str], arm: str, max_distance: int = 16) -> None:
    """Per-layer table over boundaries near the span. The last entry of ``lens_layers`` is the model's
    final layer, whose stored state is post-norm; its logit lens is the final logits, so the final
    column reports KL(A‖B) of the final distribution and no cosine."""
    print(f"| cell | kind | layer: mean logit-lens KL(A‖B) / mean residual cosine distance (pre-norm), boundaries ≤ {max_distance} tokens after the span | final logits: mean KL(A‖B) |")
    print("|---|---|---|--:|")
    for spec in dirs:
        label, _, d = spec.rpartition("=")
        d = Path(d); label = label or d.name
        c = load_table(d, arm)
        for kind in ("span",):
            m = (c["kind"] == kind) & (c["distance_tokens"] <= max_distance)
            if not m.any():
                continue
            kl, cos = c["lens_kl"][m].mean(0), c["hidden_cos_dist"][m].mean(0)
            layers = c["lens_layers"][:-1]  # drop the final entry: see docstring
            print(f"| {label} | {kind} | " + "; ".join(f"L{l}: {k:.3f} / {x:.4f}" for l, k, x in zip(layers, kl, cos)) + f" | {c['kl_ab'][m].mean():.3f} |")


# --- contagion: is the second span more likely because the first was emitted non-canonically? -------
CONTAGION_SCHEMA = pa.schema([
    ("kind", pa.string()), ("prompt_id", pa.string()), ("sample", pa.int32()), ("first_pos", pa.int32()), ("target_pos", pa.int32()),
    ("gap", pa.int32()), ("same_text", pa.bool_()), ("target_id", pa.int32()), ("canonical_id", pa.int32()),
    ("logp_a", pa.float32()), ("logp_b", pa.float32()), ("logp_a_canonical", pa.float32()), ("logp_b_canonical", pa.float32()),
    ("rank_a", pa.int32()), ("rank_b", pa.int32()),
])


def build_contagion_pair(an: Analyzer, prompt_ids: list[int], ids: list[int], p1: int, p2: int) -> dict | None:
    """A = the emitted ids up to (not including) the target position p2; B = the same with ids[p1:p2] canonically re-tokenized.

    Both keep the prompt and every emitted token before p1 verbatim and end at
    the byte where the target token starts, so each next-token distribution is
    the model's prediction for that token."""
    prefix, tail = ids[:p1], ids[p1:p2]
    canon = canonical_tail(an, prefix, tail)
    if canon is None or canon == tail:
        return None
    return {"a": prompt_ids + prefix + tail, "b": prompt_ids + prefix + canon, "first_pos": p1, "target_pos": p2}


def measure_contagion(m: Model, pair: dict, target_id: int, canonical_id: int) -> dict:
    torch = m.torch
    out = {}
    for key in ("a", "b"):
        with torch.no_grad():
            logp = torch.log_softmax(m.model.lm_head(m.hidden(pair[key])[-1][0, -1]).float(), -1)
        out[f"logp_{key}"] = float(logp[target_id])
        out[f"logp_{key}_canonical"] = float(logp[canonical_id])
        out[f"rank_{key}"] = int((logp > logp[target_id]).sum().item()) + 1
    return out


def run_contagion(run_dir: Path, arm: str, model_name: str, revision: str, out_dir: Path, max_gap: int, max_pairs: int, seed: int) -> None:
    an = Analyzer(model_name, revision)
    records = {(r["prompt_id"], r["sample"]): r for r in iter_records(run_dir / f"{arm}.parquet")}
    spans = defaultdict(list)
    for line in (run_dir / "metrics" / "examples.jsonl").open():
        e = json.loads(line)
        if e["file"].startswith(arm):
            spans[(e["prompt_id"], e["sample"])].append(e)
    rng = random.Random(seed)
    pairs, skipped = [], Counter()
    for key, es in spans.items():
        es.sort(key=lambda e: e["pos"])
        for e1, e2 in zip(es, es[1:]):  # consecutive events: nothing non-canonical between them
            if e1["canonical"] is None or e2["canonical"] is None:
                skipped["fragment"] += 1
                continue
            gap = e2["pos"] - (e1["pos"] + len(e1["emitted"]))
            if gap < 1 or gap > max_gap:
                skipped["gap out of range"] += 1
                continue
            pairs.append((key, e1, e2))
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    m = Model(model_name, revision)
    rows = []
    for key, e1, e2 in pairs:
        rec = records[key]
        ids = list(rec["token_ids"])
        pair = build_contagion_pair(an, list(rec["prompt_token_ids"]), ids, e1["pos"], e2["pos"])
        if pair is None:
            skipped["canonical tail does not splice"] += 1
            continue
        target = ids[e2["pos"]]
        canonical = an.tok.encode("".join(e2["canonical"][:1]), add_special_tokens=False)
        canonical_id = canonical[0] if len(canonical) == 1 else target
        r = measure_contagion(m, pair, target, canonical_id)
        rows.append({"kind": "pair", "prompt_id": key[0], "sample": int(key[1]), "first_pos": e1["pos"], "target_pos": e2["pos"], "gap": e2["pos"] - (e1["pos"] + len(e1["emitted"])),
                     "same_text": "".join(e1["emitted"]) == "".join(e2["emitted"]), "target_id": target, "canonical_id": canonical_id, **r})
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=CONTAGION_SCHEMA), out_dir / f"{arm}.contagion.parquet")
    meta = {"model": model_name, "revision": revision, "run_dir": str(run_dir), "arm": arm, "prefix": "full", "max_gap": max_gap, "seed": seed,
            "candidate_pairs": len(pairs), "measured_pairs": sum(r["kind"] == "pair" for r in rows), "skipped": dict(skipped)}
    (out_dir / f"{arm}.contagion.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def summarize_contagion(dirs: list[str], arm: str) -> None:
    """Δ = log p(target | emitted context) − log p(target | re-tokenized context); positive favours the non-canonical continuation."""
    from scipy.stats import wilcoxon

    print("| cell | kind | pairs | median gap | mean Δ (nats) | median Δ | Δ > 0 | Wilcoxon p | mean Δ for the canonical alternative | rank of target: median A / B |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for spec in dirs:
        label, _, d = spec.rpartition("=")
        d = Path(d); label = label or d.name
        t = pq.read_table(d / f"{arm}.contagion.parquet")
        cols = {c: t.column(c).to_numpy(zero_copy_only=False) for c in t.column_names}
        delta = cols["logp_a"] - cols["logp_b"]
        delta_c = cols["logp_a_canonical"] - cols["logp_b_canonical"]
        groups = [("same text", (cols["kind"] == "pair") & cols["same_text"]), ("different text", (cols["kind"] == "pair") & ~cols["same_text"])]
        for name, mask in groups:
            if mask.sum() == 0:
                continue
            dm = delta[mask]
            p = wilcoxon(dm).pvalue if len(dm) >= 6 and np.any(dm != 0) else float("nan")
            print(f"| {label} | {name} | {mask.sum()} | {int(np.median(cols['gap'][mask]))} | {dm.mean():+.3f} | {np.median(dm):+.3f} | {100 * (dm > 0).mean():.0f}% | {p:.2g} | {delta_c[mask].mean():+.3f} | {int(np.median(cols['rank_a'][mask]))} / {int(np.median(cols['rank_b'][mask]))} |")


def fetch(run: str, prompt_set: str, repo: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo, repo_type="dataset", local_dir="out", allow_patterns=[f"{run}/{prompt_set}/*.parquet", f"{run}/{prompt_set}/metrics/examples.jsonl"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("run"); f.add_argument("--prompt-set", default="dapo_sample500"); f.add_argument("--repo", default="brendanlong/noncanonical-post-training")
    r = sub.add_parser("run")
    r.add_argument("run_dir", type=Path); r.add_argument("--model", required=True); r.add_argument("--revision", default="main"); r.add_argument("--arm", default="untruncated")
    r.add_argument("--out-dir", type=Path, default=None); r.add_argument("--after", type=int, default=512)
    r.add_argument("--max-spans", type=int, default=400); r.add_argument("--per-rollout", type=int, default=3); r.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("summarize"); s.add_argument("dirs", nargs="+", help="[label=]out/divergence/<run>"); s.add_argument("--arm", default="untruncated"); s.add_argument("--lens", action="store_true"); s.add_argument("--contagion", action="store_true")
    c = sub.add_parser("contagion", help="second-span probability under the emitted vs re-tokenized first span")
    c.add_argument("run_dir", type=Path); c.add_argument("--model", required=True); c.add_argument("--revision", default="main"); c.add_argument("--arm", default="untruncated")
    c.add_argument("--out-dir", type=Path, default=None); c.add_argument("--max-gap", type=int, default=4096)
    c.add_argument("--max-pairs", type=int, default=400); c.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch(args.run, args.prompt_set, args.repo)
    elif args.cmd == "run":
        out_dir = args.out_dir or Path("out/divergence") / args.run_dir.parent.name
        run_cell(args.run_dir, args.arm, args.model, args.revision, out_dir, args.after, args.max_spans, args.per_rollout, args.seed)
    elif args.cmd == "contagion":
        out_dir = args.out_dir or Path("out/divergence") / args.run_dir.parent.name
        run_contagion(args.run_dir, args.arm, args.model, args.revision, out_dir, args.max_gap, args.max_pairs, args.seed)
    else:
        (summarize_contagion if args.contagion else summarize_lens if args.lens else summarize)(args.dirs, args.arm)


if __name__ == "__main__":
    main()
