"""Round-trip metrics, recomputed from stored token IDs.

    uv run python -m noncanon.metrics --tokenizer allenai/Olmo-3-7B-Think \
        --records out/pilot/*.parquet --out-dir out/pilot/metrics

A rollout is split into runs of ordinary tokens between special tokens (EOS,
``<|im_end|>`` and the like are excluded). For each run the emitted IDs are
decoded and re-encoded; ``encode(decode(ids)) != ids`` is the non-canonical
condition. Because both tokenizations partition the same bytes, token
boundaries that both sides share split the run into stretches, and a stretch
whose emitted tokens differ from its canonical tokens is one non-canonical
*span*.

Headline per-token rate = canonical tokens inside spans / canonical tokens,
so numerator and denominator count the same thing (the two-token word
``c``+``at`` written as ``c``+``a``+``t`` is one non-canonical canonical
token out of two: 50%). The emitted-token count inside spans is reported
alongside.

Also derived per rollout: think/answer split at ``</think>``, token classes,
verifier result against the stored integer answer, sequence-level flags at
fixed lengths, and top-k entropy per position.
"""

from __future__ import annotations

import argparse
import codecs
import json
import re
import statistics
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from noncanon.records import iter_records

THINK_END = b"</think>"
SEQ_LENGTHS = (256, 1024, 4096)
CLASSES = ("whitespace", "digit", "word", "mixed", "symbol")
WHITESPACE_BYTES = (b" ", b"\n", b"\t", b"\r")

_BYTE_DECODER = {u: b for b, u in bytes_to_unicode().items()}


def span_shape(classes: list[str]) -> str:
    """Coarse shape of a span from its emitted tokens' classes: a whitespace
    run, an all-alphabetic span (two words emitted without the space between
    them, or a word split unusually), or anything involving symbols/digits."""
    if "whitespace" in classes:
        return "whitespace"
    if all(c == "word" for c in classes):
        return "alphabetic"
    return "symbolic"


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


@dataclass
class Span:
    """One stretch where the emitted tokens differ from the canonical ones."""

    start: int  # index into the rollout's token list
    emitted: list[int]
    canonical: list[int]


@dataclass
class Rollout:
    ids: list[int]
    excluded_utf8: int = 0  # tokens whose bytes are not valid UTF-8 (cut mid-character, or garbage)
    excluded_truncated: int = 0  # trailing tokens dropped because the cap cut a word
    spans: list[Span] = field(default_factory=list)
    n_canonical: int = 0  # canonical tokens over all measured runs
    excluded: set[int] = field(default_factory=set)  # indices of tokens not measured
    fragments: list[tuple[int, list[int]]] = field(default_factory=list)  # (start, ids) of each invalid/incomplete byte sequence


class Analyzer:
    def __init__(self, tokenizer_name: str, revision: str = "main"):
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision)
        # all_special_ids covers only eos/pad on OLMo-3; <|im_start|>, <|im_end|>
        # and the other control tokens are "added tokens" not flagged special.
        self.special = set(self.tok.all_special_ids) | set(self.tok.added_tokens_decoder)

    def token_bytes(self, ids: list[int]) -> list[bytes]:
        """Exact bytes of each token (byte-level BPE maps one char per byte)."""
        out = []
        for t, s in zip(ids, self.tok.convert_ids_to_tokens(ids)):
            out.append(s.encode() if t in self.special else bytes(_BYTE_DECODER[c] for c in s))
        return out

    def piece(self, t: int) -> str:
        return self.tok.decode([t], skip_special_tokens=False, clean_up_tokenization_spaces=False)

    # --- the round trip ---------------------------------------------------------
    def canonical_spans(self, run: list[int], offset: int) -> tuple[int, list[Span]]:
        """Diff one run of ordinary tokens against its canonical re-encoding.

        Returns (number of canonical tokens, spans). The bytes of the run
        must be valid UTF-8.
        """
        emitted_bytes = self.token_bytes(run)
        canonical = self.tok.encode(b"".join(emitted_bytes).decode(), add_special_tokens=False)
        if run == canonical:
            return len(canonical), []
        canonical_bytes = self.token_bytes(canonical)
        assert b"".join(emitted_bytes) == b"".join(canonical_bytes)

        # Walk both tokenizations in step. Whenever they reach the same byte
        # offset together, the stretch since the previous shared offset is
        # either identical or one span.
        e_ends, c_ends = cumulative_ends(emitted_bytes), cumulative_ends(canonical_bytes)
        shared = sorted(set(e_ends) & set(c_ends))
        spans, e_from, c_from = [], 0, 0
        for boundary in shared:
            e_to = bisect_right(e_ends, boundary)  # tokens ending at or before this offset
            c_to = bisect_right(c_ends, boundary)
            if run[e_from:e_to] != canonical[c_from:c_to]:
                spans.append(Span(offset + e_from, run[e_from:e_to], canonical[c_from:c_to]))
            e_from, c_from = e_to, c_to
        return len(canonical), spans

    def measure(self, ids: list[int], finish_reason: str) -> Rollout:
        ids = list(ids)
        while ids and ids[-1] in self.special:  # the stop token is not text
            ids.pop()
        r = Rollout(ids)
        if finish_reason == "length":
            # A cap can cut a word in half, and a half-word's tokens need not be
            # the canonical tokens of the half-word string. Drop the last word.
            cut = last_word_start(self.token_bytes(ids))
            r.excluded_truncated = len(ids) - cut
            ids = ids[:cut]
        for offset, run in ordinary_runs(ids, self.special):
            kept = set()
            for s, e in decodable_segments(self.token_bytes(run)):
                kept.update(range(s, e))
                n_canonical, spans = self.canonical_spans(run[s:e], offset + s)
                r.n_canonical += n_canonical
                r.spans.extend(spans)
            bad = [offset + i for i in range(len(run)) if i not in kept]
            r.excluded_utf8 += len(bad)
            r.excluded.update(bad)
            for group in consecutive_groups(bad):
                r.fragments.append((group[0], [ids[i] for i in group]))
        r.ids = ids
        return r

    # --- one rollout ------------------------------------------------------------
    def analyze(self, rec: dict) -> dict:
        r = self.measure(rec["token_ids"], rec["finish_reason"])
        ids = r.ids
        prompt_tail = self.tok.decode(rec["prompt_token_ids"][-8:], skip_special_tokens=False)
        has_think = prompt_tail.rstrip().endswith("<think>")

        all_bytes = self.token_bytes(ids)
        full = b"".join(all_bytes)
        ends = cumulative_ends(all_bytes)
        think_end = None  # index of the first token after the one containing </think>
        answer_from = 0 if not has_think else None
        if has_think and (pos := full.find(THINK_END)) >= 0:
            answer_from = pos + len(THINK_END)
            think_end = next(i + 1 for i, e in enumerate(ends) if e >= answer_from)

        def region(i: int) -> str:
            return "think" if has_think and (think_end is None or i < think_end) else "answer"

        measured = [i for i, t in enumerate(ids) if t not in self.special and i not in r.excluded]
        ordinal = {i: k for k, i in enumerate(measured)}
        pieces = {i: self.piece(ids[i]) for i in measured}
        classes = {i: token_class(pieces[i]) for i in measured}

        nc_emitted = [i for s in r.spans for i in range(s.start, s.start + len(s.emitted))]
        spans_out = [
            {
                "pos": s.start,
                "region": region(s.start),
                "emitted": [pieces[i] for i in range(s.start, s.start + len(s.emitted))],
                "canonical": [self.piece(t) for t in s.canonical],
                # Decode context jointly: a character split across tokens would
                # otherwise render as replacement marks.
                "context": b"".join(all_bytes[max(0, s.start - 8) : s.start]).decode(errors="replace"),
                "classes": [classes[i] for i in range(s.start, s.start + len(s.emitted))],
                "shape": span_shape([classes[i] for i in range(s.start, s.start + len(s.emitted))]),
            }
            for s in r.spans
        ]
        # Byte-fragment events: the model started a multi-byte character as
        # separate byte tokens and never completed it (or emitted stray bytes).
        # These bytes have no text form, so they cannot be scored canonical or
        # not; they are reported as their own event class.
        for start, frag_ids in r.fragments:
            spans_out.append(
                {
                    "pos": start,
                    "region": region(start),
                    "emitted": [self.piece(t) for t in frag_ids],
                    "emitted_bytes": [self.token_bytes([t])[0].hex() for t in frag_ids],
                    "canonical": None,
                    "context": b"".join(all_bytes[max(0, start - 8) : start]).decode(errors="replace"),
                    "classes": ["fragment"] * len(frag_ids),
                    "shape": "byte-fragment",
                }
            )
        # Counting rule for fragments (Brendan, 2026-09-03): a fragment adjacent
        # to a non-canonical span is part of that span (+0); a fragment with
        # canonical tokens on both sides is one more event (+1 to the numerator
        # and to the denominator, like a span of one canonical token).
        span_ranges = [(sp.start, sp.start + len(sp.emitted)) for sp in r.spans]
        standalone = [
            (start, frag) for start, frag in r.fragments
            if not any(e == start or b == start + len(frag) for b, e in span_ranges)
        ]
        nc_events = sum(len(sp.canonical) for sp in r.spans) + len(standalone)
        n_units = r.n_canonical + len(standalone)
        n_think = sum(region(i) == "think" for i in measured)
        answer_text = full[answer_from:].decode(errors="replace") if answer_from is not None else ""
        transcript = render_transcript(all_bytes, {i for start, frag in r.fragments for i in range(start, start + len(frag))})
        pred, correct = verify(answer_text, rec.get("answer"))
        entropies = topk_entropy(rec["topk_logprobs"])[: len(ids)]

        return {
            "prompt_id": rec.get("prompt_id"),
            "sample": rec.get("sample"),
            "finish_reason": rec["finish_reason"],
            "has_think": has_think,
            "think_closed": (think_end is not None) if has_think else None,
            "n_tokens": len(measured),
            "n_canonical": r.n_canonical,
            "n_units": n_units,  # canonical tokens + standalone fragment events (headline denominator)
            "n_think": n_think,
            "n_answer": len(measured) - n_think,
            "nc_canonical": sum(len(s.canonical) for s in r.spans),  # segmentation only
            "nc_events": nc_events,  # headline numerator: canonical tokens in spans + standalone fragments
            "nc_emitted": len(nc_emitted),
            "nc_events_think": sum(len(s.canonical) for s in r.spans if region(s.start) == "think")
            + sum(region(start) == "think" for start, _ in standalone),
            "nc_spans": len(r.spans),
            "fragment_events": len(r.fragments),
            "fragment_events_standalone": len(standalone),
            "span_shapes": Counter(sp["shape"] for sp in spans_out),
            "nc_positions": [ordinal[i] for i in nc_emitted],
            "nc_classes": Counter(classes[i] for i in nc_emitted),
            "all_classes": Counter(classes.values()),
            "seq_flags": {str(L): any(ordinal[i] < L for i in nc_emitted) for L in SEQ_LENGTHS},
            "excluded_utf8": r.excluded_utf8,
            "excluded_truncated": r.excluded_truncated,
            "pred": pred,
            "correct": correct,
            "entropy_mean": float(entropies.mean()) if len(entropies) else None,
            "entropy_at_nc": [float(entropies[i]) for i in nc_emitted if i < len(entropies)],
            "spans": spans_out,
            "transcript": transcript,
        }


def decodable_segments(chunks: list[bytes]) -> list[tuple[int, int]]:
    """Maximal token ranges whose concatenated bytes are valid UTF-8.

    A rollout can end mid-character (cut by the cap) or, rarely, emit bytes
    that are not UTF-8 at all. Tokens in an incomplete or invalid sequence
    are left out of every segment so they are not measured.
    """
    segments, start, clean_end = [], 0, 0
    dec = codecs.getincrementaldecoder("utf-8")()
    for i, b in enumerate(chunks):
        try:
            dec.decode(b)
        except UnicodeDecodeError:
            # The bytes buffered so far (an unfinished character) are not text;
            # close the segment before them. The token that tripped the error
            # may itself be fine (e.g. an ASCII word after an abandoned prefix),
            # so try it again on a fresh decoder.
            if clean_end > start:
                segments.append((start, clean_end))
            dec.reset()
            start = clean_end = i
            try:
                dec.decode(b)
            except UnicodeDecodeError:  # a stray continuation byte: exclude it too
                dec.reset()
                start = clean_end = i + 1
                continue
        if not dec.getstate()[0]:  # no bytes buffered: the character is complete
            clean_end = i + 1
    if clean_end > start:
        segments.append((start, clean_end))
    return segments


def render_transcript(chunks: list[bytes], fragment_indices: set[int]) -> str:
    """Decode the rollout for reading, showing byte-fragment tokens explicitly.

    A fragment is rendered as ``⟨bytes e2 88⟩`` so that a transcript with a
    fragment between two words is distinguishable from one without (vLLM's
    text field shows a replacement character, which the model can also emit
    legitimately).
    """
    out, buf = [], []
    for i, b in enumerate(chunks):
        if i in fragment_indices:
            if buf:
                out.append(b"".join(buf).decode(errors="replace"))
                buf = []
            out.append(f"⟨bytes {b.hex(' ')}⟩")
        else:
            buf.append(b)
    if buf:
        out.append(b"".join(buf).decode(errors="replace"))
    return "".join(out)


def consecutive_groups(indices: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for i in indices:
        if groups and i == groups[-1][-1] + 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def cumulative_ends(chunks: list[bytes]) -> list[int]:
    return np.cumsum([len(c) for c in chunks]).tolist()


def ordinary_runs(ids: list[int], special: set[int]):
    """Yield (offset, ids) for each maximal run of non-special tokens."""
    start = None
    for i, t in enumerate(ids + [next(iter(special))]):  # sentinel closes the last run
        if t in special:
            if start is not None:
                yield start, ids[start:i]
            start = None
        elif start is None:
            start = i


def last_word_start(chunks: list[bytes]) -> int:
    """Index of the last token that begins with whitespace (the start of the last word)."""
    for i in range(len(chunks) - 1, -1, -1):
        if chunks[i].startswith(WHITESPACE_BYTES):
            return i
    return len(chunks)


def topk_entropy(topk_logprobs: list[list[float]]) -> np.ndarray:
    """Entropy (nats) of the top-k candidates at each position, renormalized over k.

    vLLM stores top-k plus the sampled token when it fell outside the top-k,
    so rows are truncated to the smallest k so every position uses the same k.
    """
    if not topk_logprobs:
        return np.zeros(0)
    k = min(len(row) for row in topk_logprobs)
    lp = np.array([sorted(row, reverse=True)[:k] for row in topk_logprobs], dtype=np.float64)
    lp -= np.logaddexp.reduce(lp, axis=1, keepdims=True)  # renormalize
    return -(np.exp(lp) * lp).sum(axis=1)


# --- verifier ---------------------------------------------------------------------
# \boxed{...} allowing one level of nested braces (\frac{1}{2}, \text{...}).
_BOXED = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
_ANSWER_LINE = re.compile(r"(?im)^\W*(?:final\s+)?answer\W*[:=]\s*(.+?)\s*$")
_ANSWER_INLINE = re.compile(r"(?i)(?:final\s+)?answer\s+is\W*(-?\d[\d,]*)")


def extract_boxed(text: str) -> str | None:
    matches = _BOXED.findall(text)
    return matches[-1] if matches else None


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
    g = parse_int(gold)
    return pred, (parse_int(pred) == g) if g is not None else None


# --- aggregation -----------------------------------------------------------------
def _rate(num: int, den: int) -> float | None:
    return round(num / den, 6) if den else None


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.4f}%"


def _bucket(rows: list[dict], key) -> dict:
    """Totals per bucket; ``key(index, row)`` names the bucket."""
    out = defaultdict(lambda: {"rollouts": 0, "units": 0, "nc": 0})
    for i, r in enumerate(rows):
        b = out[key(i, r)]
        b["rollouts"] += 1
        b["units"] += r["n_units"]
        b["nc"] += r["nc_events"]
    return {k: {**v, "rate": _rate(v["nc"], v["units"])} for k, v in sorted(out.items())}


def outcome(r: dict) -> str:
    if r["finish_reason"] == "length":
        return "truncated"
    return {True: "correct", False: "incorrect", None: "unparsed"}[r["correct"]]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    canonical = sum(r["n_canonical"] for r in rows)
    units = sum(r["n_units"] for r in rows)
    nc = sum(r["nc_events"] for r in rows)
    nc_seg = sum(r["nc_canonical"] for r in rows)
    think_canonical = sum(r["n_think"] for r in rows)  # canonical ≈ emitted outside spans; think split is by emitted index
    lengths = sorted(r["n_tokens"] for r in rows)
    cls_nc, cls_all = Counter(), Counter()
    for r in rows:
        cls_nc.update(r["nc_classes"])
        cls_all.update(r["all_classes"])
    by_rank = sorted(range(n), key=lambda i: rows[i]["n_tokens"])
    quartile = {i: f"q{min(3, 4 * rank // max(1, n)) + 1}" for rank, i in enumerate(by_rank)}
    decile_tok, decile_nc = Counter(), Counter()
    for r in rows:
        L = r["n_tokens"]
        for d in range(10):
            decile_tok[d] += int(L * (d + 1) / 10) - int(L * d / 10)
        for i in r["nc_positions"]:
            decile_nc[min(9, 10 * i // max(1, L))] += 1
    ent_nc = [e for r in rows for e in r["entropy_at_nc"]]
    finished = [r for r in rows if r["finish_reason"] != "length"]
    return {
        "rollouts": n,
        "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
        "think_closed": dict(Counter(str(r["think_closed"]) for r in rows)),
        "excluded_utf8": sum(r["excluded_utf8"] for r in rows),
        "excluded_truncated": sum(r["excluded_truncated"] for r in rows),
        "emitted_tokens": sum(r["n_tokens"] for r in rows),
        "canonical_tokens": canonical,
        "units": units,  # canonical tokens + standalone fragment events
        "nc_events": nc,
        "nc_canonical": nc_seg,
        "nc_emitted": sum(r["nc_emitted"] for r in rows),
        "per_token_rate": _rate(nc, units),  # headline: spans + standalone fragments
        "per_token_rate_segmentation_only": _rate(nc_seg, canonical),
        "spans": sum(r["nc_spans"] for r in rows),
        "spans_per_1k_tokens": round(1000 * sum(r["nc_spans"] for r in rows) / canonical, 3) if canonical else None,
        "rollouts_with_nc": sum(r["nc_spans"] > 0 or r["fragment_events"] > 0 for r in rows),
        "rollouts_with_spans_only": sum(r["nc_spans"] > 0 for r in rows),
        "fragment_events": sum(r["fragment_events"] for r in rows),
        "fragment_events_standalone": sum(r["fragment_events_standalone"] for r in rows),
        "rollouts_with_fragments": sum(r["fragment_events"] > 0 for r in rows),
        "span_shapes": dict(sum((Counter(r["span_shapes"]) for r in rows), Counter())),
        "think": {"tokens": think_canonical, "nc": sum(r["nc_events_think"] for r in rows)},
        "answer": {"tokens": sum(r["n_answer"] for r in rows), "nc": nc - sum(r["nc_events_think"] for r in rows)},
        "length": {
            "mean": round(statistics.fmean(lengths), 1) if lengths else None,
            "median": lengths[len(lengths) // 2] if lengths else None,
            "p90": lengths[int(0.9 * (len(lengths) - 1))] if lengths else None,
            "max": lengths[-1] if lengths else None,
        },
        # Denominator: rollouts that reached L tokens (shorter ones cannot fire the flag).
        "seq_flag_rate": {
            str(L): _rate(sum(r["seq_flags"][str(L)] for r in rows if r["n_tokens"] >= L), sum(r["n_tokens"] >= L for r in rows))
            for L in SEQ_LENGTHS
        },
        "classes": {c: {"tokens": cls_all[c], "nc": cls_nc[c], "rate": _rate(cls_nc[c], cls_all[c])} for c in CLASSES},
        "by_outcome": _bucket(rows, lambda i, r: outcome(r)),
        "by_length_quartile": _bucket(rows, lambda i, r: quartile[i]),
        "by_position_decile": {str(d): _rate(decile_nc[d], decile_tok[d]) for d in range(10)},
        "entropy": {
            "mean_all_positions": round(statistics.fmean(r["entropy_mean"] for r in rows if r["entropy_mean"] is not None), 4) if rows else None,
            "mean_at_nc_positions": round(statistics.fmean(ent_nc), 4) if ent_nc else None,
        },
        "accuracy": _rate(sum(bool(r["correct"]) for r in finished), sum(r["correct"] is not None for r in finished)),
    }


def to_markdown(name: str, s: dict) -> str:
    think, answer = s["think"], s["answer"]
    lines = [
        f"### {name}",
        "",
        f"- rollouts: {s['rollouts']}; finish: {s['finish_reasons']}; think closed: {s['think_closed']}; "
        f"accuracy (finished, parsed): {_pct(s['accuracy'])}; excluded tokens: {s['excluded_utf8']} incomplete UTF-8, "
        f"{s['excluded_truncated']} cut last word",
        f"- length (emitted tokens): mean {s['length']['mean']}, median {s['length']['median']}, p90 {s['length']['p90']}, max {s['length']['max']}",
        f"- **per-token non-canonical rate: {_pct(s['per_token_rate'])}** ({s['nc_events']} of {s['units']} units = canonical tokens in "
        f"{s['spans']} spans + {s['fragment_events_standalone']} standalone byte-fragment events; {s['nc_emitted']} emitted tokens in spans, "
        f"{s['spans_per_1k_tokens']} spans/1k tokens; {s['rollouts_with_nc']}/{s['rollouts']} rollouts with ≥1 event; span shapes {s['span_shapes']})",
        f"- segmentation only (fragments excluded): {_pct(s['per_token_rate_segmentation_only'])} ({s['nc_canonical']} of {s['canonical_tokens']}), "
        f"{s['rollouts_with_spans_only']}/{s['rollouts']} rollouts; byte fragments: {s['fragment_events']} events ({s['fragment_events_standalone']} standalone, "
        f"the rest adjacent to a span), {s['excluded_utf8']} tokens, {s['rollouts_with_fragments']}/{s['rollouts']} rollouts",
        f"- think: {_pct(_rate(think['nc'], think['tokens']))} ({think['nc']} / {think['tokens']}); "
        f"answer: {_pct(_rate(answer['nc'], answer['tokens']))} ({answer['nc']} / {answer['tokens']})",
        f"- sequence-level flag rate at L={list(s['seq_flag_rate'])}: {[_pct(v) for v in s['seq_flag_rate'].values()]}",
        f"- entropy (top-k, nats): all positions {s['entropy']['mean_all_positions']}, at non-canonical positions {s['entropy']['mean_at_nc_positions']}",
        "",
        "| token class | tokens | non-canonical | rate |",
        "|---|--:|--:|--:|",
        *(f"| {c} | {v['tokens']} | {v['nc']} | {_pct(v['rate'])} |" for c, v in s["classes"].items()),
        "",
    ]
    for title, table in (("outcome", s["by_outcome"]), ("length quartile", s["by_length_quartile"])):
        lines += [f"| {title} | rollouts | units | non-canonical | rate |", "|---|--:|--:|--:|--:|"]
        lines += [f"| {k} | {v['rollouts']} | {v['units']} | {v['nc']} | {_pct(v['rate'])} |" for k, v in table.items()]
        lines.append("")
    lines += [
        "| position decile | " + " | ".join(s["by_position_decile"]) + " |",
        "|---|" + "--:|" * 10,
        "| rate | " + " | ".join(_pct(v) for v in s["by_position_decile"].values()) + " |",
        "",
    ]
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
    with (args.out_dir / "analysis.jsonl").open("w") as fa, (args.out_dir / "examples.jsonl").open("w") as fe, (args.out_dir / "transcripts.jsonl").open("w") as ft:
        for path in args.records:
            meta_path = path.with_suffix(".meta.json")
            run = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            stamp = {"file": path.name, "model": run.get("model"), "revision": run.get("revision"), "arm": run.get("arm")}
            for rec in iter_records(path):
                a = an.analyze(rec)
                groups[path.name].append(a)
                for sp in a["spans"]:
                    fe.write(json.dumps({**stamp, "prompt_id": a["prompt_id"], "sample": a["sample"], **sp}, ensure_ascii=False) + "\n")
                ft.write(json.dumps({**stamp, "prompt_id": a["prompt_id"], "sample": a["sample"], "problem": rec.get("problem"), "answer": rec.get("answer"), "finish_reason": a["finish_reason"], "correct": a["correct"], "nc_spans": a["nc_spans"], "fragment_events": a["fragment_events"], "transcript": a["transcript"]}, ensure_ascii=False) + "\n")
                slim = {k: v for k, v in a.items() if k not in ("spans", "entropy_at_nc", "nc_positions", "transcript")}
                for k in ("nc_classes", "all_classes", "span_shapes"):
                    slim[k] = dict(slim[k])
                fa.write(json.dumps({**stamp, **slim}, ensure_ascii=False) + "\n")

    summary = {name: summarize(rows) for name, rows in groups.items()}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = "\n".join(to_markdown(name, s) for name, s in summary.items())
    (args.out_dir / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
