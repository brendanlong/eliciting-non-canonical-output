"""Downstream divergence: what a non-canonical span does to the model's later computation.

    uv run python -m noncanon.divergence fetch think-dpo                       # records + spans from the HF dataset into out/
    uv run python -m noncanon.divergence run --model allenai/Olmo-3-7B-Think-DPO out/think-dpo/dapo_sample500
    uv run python -m noncanon.divergence summarize out/divergence/think-dpo ...

For each non-canonical span (from ``metrics/examples.jsonl``) two teacher-forced
forward passes are run over the same text: the token ids the model actually
emitted (A) and the canonical re-tokenization of the decoded text (B). The
two share the prompt and a prefix window of up to ``--before`` emitted tokens
(shifted forward until it re-encodes to itself, so A and B are identical up
to the span), then the span and up to ``--after`` following tokens. At every
byte boundary after the span that both tokenizations share, we record the KL
divergence between the two next-token distributions (final logits, and the
logit lens at a subset of layers) and the cosine distance of the residual
stream at those layers. Spans with another event inside the window are
skipped so the measured divergence is the span's own.

Control: in rollouts of the same cell with no event, a random alphabetic word
token at a matched depth is split into two in-vocabulary pieces (an
arbitrary non-canonical tokenization of the same text) and measured the same
way. Rows are written to ``out/divergence/<run>/<arm>.parquet``.
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
def canonical_start(an: Analyzer, ids: list[int], start: int, end: int, max_shift: int = 16) -> int | None:
    """First index in [start, start + max_shift] from which ids[start:end] re-encodes to itself."""
    for s in range(start, min(end, start + max_shift + 1)):
        seg = ids[s:end]
        if an.tok.encode(an.tok.decode(seg), add_special_tokens=False) == seg:
            return s
    return None


def shared_boundaries(bytes_a: list[bytes], bytes_b: list[bytes], after_byte: int) -> list[tuple[int, int, int]]:
    """(byte offset, index in a, index in b) of every token end at or beyond ``after_byte`` present in both tokenizations.

    The end of the span itself is included (distance 0): the prediction made
    there is the first one conditioned on the span."""
    ends_a = {e: i for i, e in enumerate(cumulative_ends(bytes_a))}
    ends_b = {e: i for i, e in enumerate(cumulative_ends(bytes_b))}
    return [(e, ends_a[e], ends_b[e]) for e in sorted(ends_a) if e >= after_byte and e in ends_b]


def build_pair(an: Analyzer, prompt_ids: list[int], ids: list[int], pos: int, span_len: int, before: int, after: int) -> dict | None:
    """Token sequences A (emitted) and B (canonical) for one span, with the shared boundaries after it."""
    start = canonical_start(an, ids, max(0, pos - before), pos)
    if start is None:
        return None
    end = min(len(ids), pos + span_len + after)
    window = ids[start:end]
    prefix = ids[start:pos]
    canon = an.tok.encode(an.tok.decode(window), add_special_tokens=False)
    if canon[: len(prefix)] != prefix or canon == window:
        return None  # prefix not reproduced (a merge across the window edge) or nothing differs
    bytes_a, bytes_b = an.token_bytes(window), an.token_bytes(canon)
    span_end_byte = sum(len(b) for b in bytes_a[: pos - start + span_len])
    bounds = shared_boundaries(bytes_a, bytes_b, span_end_byte)
    if not bounds:
        return None
    return {
        "a": prompt_ids + window, "b": prompt_ids + canon, "offset": len(prompt_ids), "span_end_a": pos - start + span_len,
        "span_end_byte": span_end_byte, "prefix_tokens": len(prefix), "prefix_truncated": start > 0,
        "bounds": bounds,
    }


def split_word(an: Analyzer, t: int, rng: random.Random) -> list[int] | None:
    """Two in-vocabulary tokens whose bytes concatenate to token ``t``'s bytes (an arbitrary non-canonical tokenization)."""
    b = an.token_bytes([t])[0]
    if len(b) < 5 or not b.decode(errors="replace").strip().isalpha():
        return None
    vocab = an.tok.get_vocab()
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    enc = bytes_to_unicode()
    to_str = lambda bs: "".join(enc[x] for x in bs)
    cuts = list(range(2, len(b) - 1))
    rng.shuffle(cuts)
    for c in cuts:
        left, right = to_str(b[:c]), to_str(b[c:])
        if left in vocab and right in vocab:
            return [vocab[left], vocab[right]]
    return None


def control_pairs(an: Analyzer, records: list[dict], depths: list[int], n: int, before: int, after: int, seed: int) -> list[tuple[dict, dict]]:
    """Arbitrary re-tokenizations at span-matched depths in event-free rollouts."""
    rng = random.Random(seed)
    out = []
    tries = 0
    while len(out) < n and tries < 20 * n:
        tries += 1
        rec, depth = rng.choice(records), rng.choice(depths)
        ids = list(rec["token_ids"])
        if len(ids) < 32:
            continue
        pos = min(depth, len(ids) - 16) + rng.randint(-24, 24)
        pos = max(1, min(len(ids) - 8, pos))
        for p in range(pos, min(pos + 48, len(ids) - 4)):
            if ids[p] in an.special:
                continue
            pieces = split_word(an, ids[p], rng)
            if pieces is None:
                continue
            perturbed = ids[:p] + pieces + ids[p + 1:]
            pair = build_pair(an, list(rec["prompt_token_ids"]), perturbed, p, len(pieces), before, after)
            if pair is not None:
                out.append((pair, {"prompt_id": rec["prompt_id"], "sample": rec["sample"], "pos": p, "shape": "control-split", "emitted": pieces, "canonical": [ids[p]]}))
                break
    return out


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

    def forward(self, ids: list[int], positions: list[int]) -> tuple:
        """Final log-probs, logit-lens log-probs per selected layer, and residual states at ``positions``."""
        torch = self.torch
        with torch.no_grad():
            out = self.model(torch.tensor([ids], device=self.device), output_hidden_states=True)
            idx = torch.tensor(positions, device=self.device)
            final = torch.log_softmax(out.logits[0, idx].float(), -1)
            hidden = [out.hidden_states[l][0, idx] for l in self.layers]
            lens = [torch.log_softmax(self.model.lm_head(self.model.model.norm(h)).float(), -1) for h in hidden]
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


def run_cell(run_dir: Path, arm: str, model_name: str, revision: str, out_dir: Path, before: int, after: int, max_spans: int, per_rollout: int, seed: int) -> None:
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
            lo, hi = e["pos"] - before, e["pos"] + len(e["emitted"]) + after
            if any(lo <= s < hi for j, s in enumerate(starts) if j != i):
                skipped["another event in window"] += 1
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
        pair = build_pair(an, list(rec["prompt_token_ids"]), list(rec["token_ids"]), e["pos"], len(e["emitted"]), before, after)
        if pair is None:
            skipped["prefix not canonical / no shared boundary"] += 1
            continue
        rows.extend(measure(m, pair, e, "span"))
    n_spans = len({(r["prompt_id"], r["sample"], r["span_pos"]) for r in rows})
    clean = [r for k, r in records.items() if k not in spans and r["finish_reason"] == "stop"]
    depths = [e["pos"] for _, e in candidates] or [256]
    for pair, meta in control_pairs(an, clean, depths, n_spans, before, after, seed):
        rows.extend(measure(m, pair, meta, "control"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out_dir / f"{arm}.parquet")
    meta = {"model": model_name, "revision": revision, "run_dir": str(run_dir), "arm": arm, "before": before, "after": after, "seed": seed,
            "candidate_spans": len(candidates), "measured_spans": n_spans, "controls": len({(r['prompt_id'], r['sample'], r['span_pos']) for r in rows if r['kind'] == 'control'}),
            "skipped": dict(skipped), "lens_layers": m.layers, "rows": len(rows)}
    (out_dir / f"{arm}.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


# --- summary (CPU) ------------------------------------------------------------------
def load_table(d: Path, arm: str) -> dict[str, np.ndarray]:
    t = pq.read_table(d / f"{arm}.parquet")
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
        for kind in ("span", "control"):
            m = c["kind"] == kind
            if not m.any():
                continue
            cells = []
            for lo, hi in DISTANCE_BINS:
                y = m & (c["distance_tokens"] >= lo) & (c["distance_tokens"] <= hi)
                cells.append(f"{c['kl_ab'][y].mean():.3f} / {np.median(c['kl_ab'][y]):.4f} / {100 * (1 - c['top1_agree'][y].mean()):.1f}% (n={y.sum()})" if y.any() else "—")
            print(f"| {label} | {kind} | {n_spans(c, m)} | {m.sum()} | " + " | ".join(cells) + " |")


def summarize_lens(dirs: list[str], arm: str, max_distance: int = 16) -> None:
    print(f"| cell | kind | layer: mean logit-lens KL(A‖B) / mean residual cosine distance, boundaries ≤ {max_distance} tokens after the span |")
    print("|---|---|---|")
    for spec in dirs:
        label, _, d = spec.rpartition("=")
        d = Path(d); label = label or d.name
        c = load_table(d, arm)
        for kind in ("span", "control"):
            m = (c["kind"] == kind) & (c["distance_tokens"] <= max_distance)
            if not m.any():
                continue
            kl, cos = c["lens_kl"][m].mean(0), c["hidden_cos_dist"][m].mean(0)
            print(f"| {label} | {kind} | " + "; ".join(f"L{l}: {k:.3f} / {x:.4f}" for l, k, x in zip(c["lens_layers"], kl, cos)) + " |")


def fetch(run: str, prompt_set: str, repo: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo, repo_type="dataset", local_dir="out", allow_patterns=[f"{run}/{prompt_set}/*.parquet", f"{run}/{prompt_set}/metrics/examples.jsonl"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("run"); f.add_argument("--prompt-set", default="dapo_sample500"); f.add_argument("--repo", default="brendanlong/noncanonical-post-training")
    r = sub.add_parser("run")
    r.add_argument("run_dir", type=Path); r.add_argument("--model", required=True); r.add_argument("--revision", default="main"); r.add_argument("--arm", default="untruncated")
    r.add_argument("--out-dir", type=Path, default=None); r.add_argument("--before", type=int, default=4096); r.add_argument("--after", type=int, default=512)
    r.add_argument("--max-spans", type=int, default=400); r.add_argument("--per-rollout", type=int, default=3); r.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("summarize"); s.add_argument("dirs", nargs="+", help="[label=]out/divergence/<run>"); s.add_argument("--arm", default="untruncated"); s.add_argument("--lens", action="store_true")
    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch(args.run, args.prompt_set, args.repo)
    elif args.cmd == "run":
        out_dir = args.out_dir or Path("out/divergence") / args.run_dir.parent.name
        run_cell(args.run_dir, args.arm, args.model, args.revision, out_dir, args.before, args.after, args.max_spans, args.per_rollout, args.seed)
    else:
        (summarize_lens if args.lens else summarize)(args.dirs, args.arm)


if __name__ == "__main__":
    main()
