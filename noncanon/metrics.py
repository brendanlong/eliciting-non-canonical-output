"""Round-trip metrics, recomputed from stored token IDs.

    uv run python -m noncanon.metrics --tokenizer allenai/Olmo-3-7B-Think \
        --records out/pilot/*.jsonl.gz --out-dir out/pilot/metrics

A rollout is split into segments between special tokens (EOS, ``<|im_end|>``
and the like are excluded from the measurement). For each segment the
emitted IDs are decoded and re-encoded; ``encode(decode(ids)) != ids`` is
the non-canonical condition. Because both tokenizations partition the same
byte string, the minimal diff is exact: token boundaries that both sides
share split the segment into stretches, and a stretch whose emitted tokens
differ from its canonical tokens is one non-canonical span. Per-token rate
is emitted tokens inside such spans divided by emitted tokens.

Also derived per rollout: think/answer split at ``</think>``, token classes,
verifier result against the stored integer answer, sequence-level flags at
fixed lengths, and top-k entropy per position.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

THINK_END = b"</think>"
SEQ_LENGTHS = (256, 1024, 4096)
CLASSES = ("whitespace", "digit", "word", "mixed", "symbol")

_BYTE_DECODER = {u: b for b, u in bytes_to_unicode().items()}


def token_class(piece: str) -> str:
    s = piece.strip()
    if not s:
        return "whitespace"
    if s.isdigit():
        return "digit"
    if s.isalpha():
        return "word"
    if any(c.isalnum() for c in s):
        return "mixed"
    return "symbol"


class Analyzer:
    def __init__(self, tokenizer_name: str, revision: str = "main"):
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision)
        # all_special_ids covers only eos/pad on OLMo-3; <|im_start|>, <|im_end|>
        # and the other control tokens are "added tokens" not flagged special.
        self.special = set(self.tok.all_special_ids) | set(self.tok.added_tokens_decoder)

    # --- byte-level bookkeeping -------------------------------------------------
    def token_bytes(self, ids: list[int]) -> list[bytes]:
        out = []
        for t, s in zip(ids, self.tok.convert_ids_to_tokens(ids)):
            if t in self.special:
                out.append(s.encode("utf-8"))
            else:
                out.append(bytes(_BYTE_DECODER[c] for c in s))
        return out

    def piece(self, t: int) -> str:
        return self.tok.decode([t], skip_special_tokens=False, clean_up_tokenization_spaces=False)

    # --- the round trip -----------------------------------------------------------
    def trim_incomplete_utf8(self, seg: list[int]) -> tuple[list[int], int, bool]:
        """Drop trailing tokens that leave the segment's bytes mid-character.

        A rollout cut by ``max_tokens`` can end inside a multi-byte character;
        those bytes cannot round-trip and must not be measured. Returns
        (trimmed segment, tokens dropped, still_invalid). ``still_invalid``
        means the bytes are undecodable even after dropping up to four
        tokens, so the caller excludes the whole segment.
        """
        eb = self.token_bytes(seg)
        for k in range(0, min(4, len(seg)) + 1):
            try:
                b"".join(eb[: len(seg) - k]).decode("utf-8")
                return seg[: len(seg) - k], k, False
            except UnicodeDecodeError:
                continue
        return seg, 0, True

    def roundtrip_segment(self, seg: list[int]) -> tuple[list[int], list[tuple[int, int, int, int]]]:
        """Return (canonical ids, spans) for a segment whose bytes are valid UTF-8.

        spans are (e_start, e_end, c_start, c_end) index ranges into the
        emitted and canonical sequences whose tokens differ.
        """
        eb = self.token_bytes(seg)
        text = b"".join(eb).decode("utf-8")
        canon = self.tok.encode(text, add_special_tokens=False)
        if seg == canon:
            return canon, []
        cb = self.token_bytes(canon)
        assert b"".join(eb) == b"".join(cb), "canonical re-encoding changed the bytes"
        e_ends, c_ends = _cum(eb), _cum(cb)
        common = sorted(set(e_ends) & set(c_ends))
        spans = []
        ei = ci = 0
        e_start = c_start = 0
        for boundary in common:
            while ei < len(seg) and e_ends[ei] <= boundary:
                ei += 1
            while ci < len(canon) and c_ends[ci] <= boundary:
                ci += 1
            if seg[e_start:ei] != canon[c_start:ci]:
                spans.append((e_start, ei, c_start, ci))
            e_start, c_start = ei, ci
        return canon, spans

    # --- one rollout -----------------------------------------------------------
    def analyze(self, rec: dict) -> dict:
        ids: list[int] = list(rec["token_ids"])
        # Trailing EOS / stop tokens are part of the stream but not of the text.
        while ids and ids[-1] in self.special:
            ids.pop()
        prompt_tail = self.tok.decode(rec["prompt_token_ids"][-8:], skip_special_tokens=False)
        has_think = prompt_tail.rstrip().endswith("<think>")

        pieces = [self.piece(t) for t in ids]
        classes = [token_class(p) if t not in self.special else "special" for t, p in zip(ids, pieces)]
        all_bytes = self.token_bytes(ids)
        ends = _cum(all_bytes)
        full = b"".join(all_bytes)

        think_end = None  # index of the first token after the one containing "</think>"
        answer_start_byte = 0 if not has_think else None
        if has_think:
            pos = full.find(THINK_END)
            if pos >= 0:
                answer_start_byte = pos + len(THINK_END)
                think_end = next(i + 1 for i, e in enumerate(ends) if e >= answer_start_byte)

        # Measured tokens: non-special, and not dropped as incomplete UTF-8.
        excluded: set[int] = set()
        nc_idx: list[int] = []
        spans_out: list[dict] = []
        for start, seg in _segments(ids, self.special):
            seg, dropped, invalid = self.trim_incomplete_utf8(seg)
            if invalid:
                excluded.update(range(start, start + len(seg)))
                continue
            excluded.update(range(start + len(seg), start + len(seg) + dropped))
            canon, spans = self.roundtrip_segment(seg)
            for es, ee, cs, ce in spans:
                nc_idx.extend(range(start + es, start + ee))
                spans_out.append(
                    {
                        "pos": start + es,
                        "region": _region(start + es, think_end, has_think),
                        "emitted": [pieces[i] for i in range(start + es, start + ee)],
                        "canonical": [self.piece(t) for t in canon[cs:ce]],
                        # Decode the context jointly: a multi-byte character split
                        # across tokens would otherwise render as replacement marks.
                        "context": b"".join(all_bytes[max(0, start + es - 8) : start + es]).decode("utf-8", errors="replace"),
                        "classes": [classes[i] for i in range(start + es, start + ee)],
                    }
                )
        measured = [i for i, t in enumerate(ids) if t not in self.special and i not in excluded]
        ordinal = {i: k for k, i in enumerate(measured)}  # raw index -> position among measured tokens

        n = len(measured)
        n_think = sum(1 for i in measured if _region(i, think_end, has_think) == "think")
        n_answer = n - n_think
        nc_think = sum(1 for i in nc_idx if _region(i, think_end, has_think) == "think")
        nc_answer = len(nc_idx) - nc_think

        answer_text = full[answer_start_byte:].decode("utf-8", errors="replace") if answer_start_byte is not None else ""
        pred, correct = verify(answer_text, rec.get("answer"))

        # Entropy over a fixed k: vLLM's dict holds top-k plus the sampled
        # token when it fell outside the top-k, so truncate to k entries.
        k = min((len(l) for l in rec.get("topk_logprobs", [])), default=0)
        entropies = [topk_entropy(sorted(lps, reverse=True)[:k]) for lps in rec.get("topk_logprobs", [])][: len(ids)]
        return {
            "prompt_id": rec.get("prompt_id"),
            "sample": rec.get("sample"),
            "arm": rec.get("arm"),
            "model": rec.get("model"),
            "revision": rec.get("revision"),
            "finish_reason": rec.get("finish_reason"),
            "has_think": has_think,
            "think_closed": (think_end is not None) if has_think else None,
            "n_tokens": n,
            "n_think": n_think,
            "n_answer": n_answer,
            "nc_tokens": len(nc_idx),
            "nc_think": nc_think,
            "nc_answer": nc_answer,
            "nc_spans": len(spans_out),
            "nc_positions": [ordinal[i] for i in nc_idx],  # ordinals among measured tokens
            "nc_classes": Counter(classes[i] for i in nc_idx),
            "all_classes": Counter(classes[i] for i in measured),
            "seq_flags": {str(L): any(ordinal[i] < L for i in nc_idx) for L in SEQ_LENGTHS},
            "excluded_tokens": len(excluded),
            "pred": pred,
            "correct": correct,
            "entropy_k": k,
            "entropy_mean": statistics.fmean(entropies) if entropies else None,
            "entropy_at_nc": [entropies[i] for i in nc_idx if i < len(entropies)],
            "entropies": entropies,
            "spans": spans_out,
        }


def _cum(chunks: list[bytes]) -> list[int]:
    out, s = [], 0
    for c in chunks:
        s += len(c)
        out.append(s)
    return out


def _segments(ids: list[int], special: set[int]):
    start = None
    for i, t in enumerate(ids):
        if t in special:
            if start is not None:
                yield start, ids[start:i]
                start = None
        elif start is None:
            start = i
    if start is not None:
        yield start, ids[start:]


def _region(i: int, think_end: int | None, has_think: bool) -> str:
    if not has_think:
        return "answer"
    if think_end is None or i < think_end:
        return "think"
    return "answer"


def topk_entropy(logprobs: list[float]) -> float:
    """Entropy (nats) of the top-k candidates, renormalized over the k."""
    if not logprobs:
        return float("nan")
    m = max(logprobs)
    ws = [math.exp(lp - m) for lp in logprobs]
    z = sum(ws)
    return -sum((w / z) * (lp - m - math.log(z)) for w, lp in zip(ws, logprobs))


# --- verifier -----------------------------------------------------------------------
_BOXED = re.compile(r"\\boxed\s*\{")
_ANSWER_LINE = re.compile(r"(?im)^\W*(?:final\s+)?answer\W*[:=]\s*(.+?)\s*$")
_ANSWER_INLINE = re.compile(r"(?i)(?:final\s+)?answer\s+is\W*(-?\d[\d,]*)")


def extract_boxed(text: str) -> str | None:
    last = None
    for m in _BOXED.finditer(text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            last = text[m.end() : i - 1]
    return last


def parse_int(s: str) -> int | None:
    s = s.strip().strip("*").strip()
    s = re.sub(r"^\\text\{(.*)\}$", r"\1", s)
    s = re.sub(r"\{,\}|\\[,!;]|[$,\s]", "", s).rstrip(".")
    m = re.fullmatch(r"(-?\d+)(?:\.0+)?", s)
    return int(m.group(1)) if m else None


def verify(answer_text: str, gold: str | None) -> tuple[str | None, bool | None]:
    """Return (prediction string, correct).

    ``correct`` is None only when no answer marker was found at all (or no
    gold answer is available); a marker whose content is not an integer is a
    wrong answer to an integer question, so it is False.
    """
    pred = extract_boxed(answer_text)
    if pred is None:
        lines = _ANSWER_LINE.findall(answer_text)
        pred = lines[-1] if lines else None
    if pred is None:
        inline = _ANSWER_INLINE.findall(answer_text)
        pred = inline[-1] if inline else None
    if gold is None or pred is None:
        return pred, None
    p, g = parse_int(pred), parse_int(gold)
    if g is None:
        return pred, None
    return pred, p == g


# --- aggregation -------------------------------------------------------------------
def iter_records(paths: list[Path]):
    for p in paths:
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 6) if den else None


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.3f}%"


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    tok = sum(r["n_tokens"] for r in rows)
    nc = sum(r["nc_tokens"] for r in rows)
    think_tok = sum(r["n_think"] for r in rows)
    think_nc = sum(r["nc_think"] for r in rows)
    ans_tok = sum(r["n_answer"] for r in rows)
    ans_nc = sum(r["nc_answer"] for r in rows)
    lengths = sorted(r["n_tokens"] for r in rows)
    cls_nc, cls_all = Counter(), Counter()
    for r in rows:
        cls_nc.update(r["nc_classes"])
        cls_all.update(r["all_classes"])
    outcome = defaultdict(lambda: [0, 0, 0])  # tokens, nc, rollouts
    for r in rows:
        key = "truncated" if r["finish_reason"] == "length" else {True: "correct", False: "incorrect", None: "unparsed"}[r["correct"]]
        outcome[key][0] += r["n_tokens"]
        outcome[key][1] += r["nc_tokens"]
        outcome[key][2] += 1
    # rate by rollout-length quartile (rank-based, so ties cannot collapse bins)
    quart = defaultdict(lambda: [0, 0, 0])
    for rank, r in enumerate(sorted(rows, key=lambda r: r["n_tokens"])):
        b = min(3, 4 * rank // max(1, n))
        quart[f"q{b + 1}"][0] += r["n_tokens"]
        quart[f"q{b + 1}"][1] += r["nc_tokens"]
        quart[f"q{b + 1}"][2] += 1
    # rate by relative position decile
    dec_tok, dec_nc = Counter(), Counter()
    for r in rows:
        L = r["n_tokens"]
        if not L:
            continue
        for d in range(10):
            lo, hi = int(L * d / 10), int(L * (d + 1) / 10)
            dec_tok[d] += hi - lo
        for i in r["nc_positions"]:
            dec_nc[min(9, int(10 * i / L))] += 1
    ent_all = [e for r in rows for e in r["entropies"] if not math.isnan(e)]
    ent_nc = [e for r in rows for e in r["entropy_at_nc"] if not math.isnan(e)]
    return {
        "rollouts": n,
        "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
        "think_closed": dict(Counter(str(r["think_closed"]) for r in rows)),
        "excluded_tokens": sum(r["excluded_tokens"] for r in rows),
        "tokens": tok,
        "nc_tokens": nc,
        "per_token_rate": _rate(nc, tok),
        "spans": sum(r["nc_spans"] for r in rows),
        "spans_per_1k_tokens": round(1000 * sum(r["nc_spans"] for r in rows) / tok, 3) if tok else None,
        "rollouts_with_nc": sum(r["nc_tokens"] > 0 for r in rows),
        "think": {"tokens": think_tok, "nc": think_nc, "rate": _rate(think_nc, think_tok)},
        "answer": {"tokens": ans_tok, "nc": ans_nc, "rate": _rate(ans_nc, ans_tok)},
        "length": {
            "mean": round(statistics.fmean(lengths), 1) if lengths else None,
            "median": lengths[len(lengths) // 2] if lengths else None,
            "p90": lengths[int(0.9 * (len(lengths) - 1))] if lengths else None,
            "max": lengths[-1] if lengths else None,
        },
        # Denominator: rollouts that reached L tokens (shorter ones cannot fire the flag).
        "seq_flag_rate": {
            str(L): _rate(sum(r["seq_flags"][str(L)] for r in rows if r["n_tokens"] >= L), sum(1 for r in rows if r["n_tokens"] >= L))
            for L in SEQ_LENGTHS
        },
        "classes": {c: {"tokens": cls_all[c], "nc": cls_nc[c], "rate": _rate(cls_nc[c], cls_all[c])} for c in CLASSES},
        "by_outcome": {k: {"rollouts": v[2], "tokens": v[0], "nc": v[1], "rate": _rate(v[1], v[0])} for k, v in sorted(outcome.items())},
        "by_length_quartile": {k: {"rollouts": v[2], "tokens": v[0], "nc": v[1], "rate": _rate(v[1], v[0])} for k, v in sorted(quart.items())},
        "by_position_decile": {str(d): _rate(dec_nc[d], dec_tok[d]) for d in range(10)},
        "entropy": {
            "mean_all_positions": round(statistics.fmean(ent_all), 4) if ent_all else None,
            "mean_at_nc_positions": round(statistics.fmean(ent_nc), 4) if ent_nc else None,
        },
        # Accuracy over finished rollouts only; truncated ones are their own outcome.
        "accuracy": _rate(
            sum(1 for r in rows if r["correct"] and r["finish_reason"] != "length"),
            sum(1 for r in rows if r["correct"] is not None and r["finish_reason"] != "length"),
        ),
    }


def to_markdown(name: str, s: dict) -> str:
    lines = [f"### {name}", ""]
    lines.append(f"- rollouts: {s['rollouts']}; finish: {s['finish_reasons']}; think closed: {s['think_closed']}; accuracy (finished, parsed): {_pct(s['accuracy'])}; tokens excluded as incomplete UTF-8: {s['excluded_tokens']}")
    lines.append(f"- length tokens: mean {s['length']['mean']}, median {s['length']['median']}, p90 {s['length']['p90']}, max {s['length']['max']}")
    lines.append(f"- **per-token non-canonical rate: {_pct(s['per_token_rate'])}** ({s['nc_tokens']} / {s['tokens']}), {s['spans']} spans, {s['spans_per_1k_tokens']} spans/1k tokens, {s['rollouts_with_nc']}/{s['rollouts']} rollouts with ≥1")
    lines.append(f"- think: {_pct(s['think']['rate'])} ({s['think']['nc']} / {s['think']['tokens']}); answer: {_pct(s['answer']['rate'])} ({s['answer']['nc']} / {s['answer']['tokens']})")
    lines.append(f"- sequence-level flag rate at L={list(s['seq_flag_rate'])}: {[_pct(v) for v in s['seq_flag_rate'].values()]}")
    lines.append(f"- entropy (top-k, nats): all positions {s['entropy']['mean_all_positions']}, at non-canonical positions {s['entropy']['mean_at_nc_positions']}")
    lines.append("")
    lines.append("| token class | tokens | non-canonical | rate |")
    lines.append("|---|--:|--:|--:|")
    for c, v in s["classes"].items():
        lines.append(f"| {c} | {v['tokens']} | {v['nc']} | {_pct(v['rate'])} |")
    lines.append("")
    lines.append("| outcome | rollouts | tokens | non-canonical | rate |")
    lines.append("|---|--:|--:|--:|--:|")
    for k, v in s["by_outcome"].items():
        lines.append(f"| {k} | {v['rollouts']} | {v['tokens']} | {v['nc']} | {_pct(v['rate'])} |")
    lines.append("")
    lines.append("| length quartile | rollouts | tokens | non-canonical | rate |")
    lines.append("|---|--:|--:|--:|--:|")
    for k, v in s["by_length_quartile"].items():
        lines.append(f"| {k} | {v['rollouts']} | {v['tokens']} | {v['nc']} | {_pct(v['rate'])} |")
    lines.append("")
    lines.append("| position decile | " + " | ".join(s["by_position_decile"]) + " |")
    lines.append("|---|" + "--:|" * 10)
    lines.append("| rate | " + " | ".join(_pct(v) for v in s["by_position_decile"].values()) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--records", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    an = Analyzer(args.tokenizer, args.revision)
    groups: dict[str, list[dict]] = defaultdict(list)
    with (args.out_dir / "analysis.jsonl").open("w") as fa, (args.out_dir / "examples.jsonl").open("w") as fe:
        for path in args.records:
            for rec in iter_records([path]):
                a = an.analyze(rec)
                groups[path.name].append(a)
                for sp in a["spans"]:
                    fe.write(json.dumps({"file": path.name, "prompt_id": a["prompt_id"], "sample": a["sample"], **sp}, ensure_ascii=False) + "\n")
                slim = {k: v for k, v in a.items() if k not in ("spans", "entropies", "entropy_at_nc", "nc_positions")}
                slim["nc_classes"] = dict(slim["nc_classes"])
                slim["all_classes"] = dict(slim["all_classes"])
                fa.write(json.dumps({"file": path.name, **slim}, ensure_ascii=False) + "\n")

    summary = {name: summarize(rows) for name, rows in groups.items()}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = "\n".join(to_markdown(name, s) for name, s in summary.items())
    (args.out_dir / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
