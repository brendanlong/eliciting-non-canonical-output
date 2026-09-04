"""CPU tests for the round-trip metric, on the real OLMo-3 tokenizer."""

import pytest

from noncanon.metrics import Analyzer, extract_boxed, parse_int, summarize, token_class, verify

TOKENIZER = "allenai/Olmo-3-7B-Think"


@pytest.fixture(scope="module")
def an():
    return Analyzer(TOKENIZER)


def make_record(an, prompt_text, output_ids, answer=None, finish="stop"):
    return {
        "prompt_id": "t",
        "sample": 0,
        "prompt_token_ids": an.tok.encode(prompt_text, add_special_tokens=False),
        "token_ids": output_ids,
        "finish_reason": finish,
        "answer": answer,
        "topk_logprobs": [[-0.1, -3.0, -4.0]] * len(output_ids),
    }


def enc(an, text):
    return an.tok.encode(text, add_special_tokens=False)


def test_canonical_text_has_zero_rate(an):
    ids = enc(an, "The quick brown fox jumps over 12345 lazy dogs.\n\nDone, 2024-09-03.")
    a = an.analyze(make_record(an, "<|im_start|>assistant\n", ids))
    assert a["nc_canonical"] == a["nc_emitted"] == a["nc_spans"] == 0
    assert a["n_tokens"] == a["n_canonical"] == len(ids)
    assert not a["has_think"]


def test_split_word_is_one_span_counted_in_canonical_tokens(an):
    light, house = enc(an, "light"), enc(an, "house")
    assert light + house != enc(an, "lighthouse")
    # "The word is " ends in a bare space token that canonically merges into
    # " l", so the span covers the space plus both word pieces (3 emitted)
    # against " l" + "ighthouse" (2 canonical).
    ids = enc(an, "The word is ") + light + house
    a = an.analyze(make_record(an, "<|im_start|>assistant\n", ids))
    span = a["spans"][0]
    assert a["nc_spans"] == 1
    assert "".join(span["emitted"]) == "".join(span["canonical"]) == " lighthouse"
    assert a["nc_emitted"] == len(span["emitted"]) == 3
    assert a["nc_canonical"] == len(span["canonical"]) == 2
    assert a["n_canonical"] == len(enc(an, "The word is lighthouse"))
    assert a["seq_flags"]["256"] is True
    b = an.analyze(make_record(an, "<|im_start|>assistant\n", enc(an, "The word is lighthouse")))
    assert b["nc_canonical"] == 0


def test_special_tokens_are_excluded(an):
    body = enc(an, "Hello there.")
    ids = body + [an.tok.convert_tokens_to_ids("<|im_end|>"), an.tok.eos_token_id]
    a = an.analyze(make_record(an, "<|im_start|>assistant\n", ids))
    assert a["n_tokens"] == len(body) and a["nc_canonical"] == 0


def test_interior_special_token_splits_runs_and_positions_are_ordinals(an):
    light, house = enc(an, "light"), enc(an, "house")
    ids = light + house + [an.tok.convert_tokens_to_ids("<|im_end|>")] + light + house
    a = an.analyze(make_record(an, "<|im_start|>assistant\n", ids))
    assert a["n_tokens"] == 4 and a["nc_spans"] == 2 and a["nc_emitted"] == 4
    assert a["nc_positions"] == [0, 1, 2, 3]  # ordinals among measured tokens, not raw indices


def test_think_split_and_verifier(an):
    ids = enc(an, "Let me think. 6*7 is 42.</think>\n\nThe answer is \\boxed{42}.")
    a = an.analyze(make_record(an, "<|im_start|>assistant\n<think>", ids, answer="42"))
    assert a["has_think"] and a["think_closed"]
    assert a["n_think"] + a["n_answer"] == a["n_tokens"] and a["n_think"] > 0 and a["n_answer"] > 0
    assert a["pred"] == "42" and a["correct"] is True


def test_think_boundary_token_straddling_answer(an):
    # "></think>\" can be one token; the answer bytes after the tag must still be verified.
    ids = enc(an, "done</think>\\boxed{42}")
    a = an.analyze(make_record(an, "<|im_start|>assistant\n<think>", ids, answer="42"))
    assert a["think_closed"] and a["pred"] == "42" and a["correct"] is True


def test_unclosed_think_is_all_cot(an):
    ids = enc(an, "Still thinking about it")
    a = an.analyze(make_record(an, "<|im_start|>assistant\n<think>", ids, answer="1"))
    assert a["think_closed"] is False and a["n_think"] == a["n_tokens"] and a["correct"] is None


def test_incomplete_utf8_tail_is_excluded_not_counted(an):
    partial = next(t for t in range(1000, 100256) if (b := an.token_bytes([t])[0]) and (b[0] & 0xF0) == 0xF0 and len(b) < 4)
    ids = enc(an, "Some text and then an emoji") + [partial]
    a = an.analyze(make_record(an, "<|im_start|>assistant\n", ids))
    assert a["excluded_utf8"] == 1 and a["n_tokens"] == len(ids) - 1 and a["nc_canonical"] == 0


def test_invalid_utf8_in_the_middle_excludes_only_the_bad_bytes(an):
    # A lone continuation byte is invalid anywhere; the words on either side
    # must still be measured, and the bad token excluded.
    bad = next(t for t in range(0, 100256) if (b := an.token_bytes([t])[0]) and (b[0] & 0xC0) == 0x80 and len(b) == 1)
    left, right = enc(an, "The quick brown fox"), enc(an, " jumps over the lazy dog")
    a = an.analyze(make_record(an, "<|im_start|>assistant\n", left + [bad] + right))
    assert a["excluded_utf8"] == 1
    assert a["n_tokens"] == len(left) + len(right)
    assert a["nc_canonical"] == 0
    assert a["fragment_events"] == 1 and a["span_shapes"] == {"byte-fragment": 1}
    frag = a["spans"][0]
    assert frag["shape"] == "byte-fragment" and frag["canonical"] is None and frag["pos"] == len(left)


def test_truncated_rollout_drops_its_last_word(an):
    # The cap cuts " numbers" after " nu"; the half-word's tokens are not
    # measured, so a cut is not mistaken for a non-canonical span.
    ids = enc(an, "There are many") + enc(an, " nu")
    a = an.analyze(make_record(an, "<|im_start|>assistant\n", ids, finish="length"))
    assert a["excluded_truncated"] == 1 and a["n_tokens"] == len(ids) - 1 and a["nc_canonical"] == 0
    b = an.analyze(make_record(an, "<|im_start|>assistant\n", ids, finish="stop"))
    assert b["excluded_truncated"] == 0 and b["n_tokens"] == len(ids)


def test_verifier_formats():
    assert extract_boxed(r"so \boxed{\frac{1}{2}} then \boxed{1,234}") == "1,234"
    assert extract_boxed(r"\boxed{\text{42}}") == r"\text{42}"
    assert parse_int("1,234") == 1234 and parse_int("$-7$") == -7 and parse_int("abc") is None
    assert parse_int("1{,}234") == 1234 and parse_int("42.0") == 42 and parse_int("042") == 42
    assert verify("Answer: 99", "99") == ("99", True)
    assert verify("Answer: 98", "99") == ("98", False)
    assert verify("**Answer:** 42", "42")[1] is True
    assert verify("Answer: **42**", "42")[1] is True
    assert verify("So the answer is 42.", "42")[1] is True
    assert verify("Final Answer: 7", "7")[1] is True
    assert verify("no answer here", "99") == (None, None)
    assert verify(r"\boxed{x+1}", "3") == ("x+1", False)  # wrong, not "unparsed"


def test_token_classes():
    assert token_class(" the") == "word"
    assert token_class("123") == "digit"
    assert token_class("\n\n") == "whitespace"
    assert token_class(" 3x") == "mixed"
    assert token_class("),") == "symbol"


def test_summarize_denominators():
    def row(n, nc_positions, finish="stop", correct=True):
        return {
            "n_tokens": n, "n_canonical": n, "n_think": 0, "n_answer": n,
            "nc_canonical": len(nc_positions), "nc_emitted": len(nc_positions), "nc_canonical_think": 0,
            "nc_spans": len(nc_positions), "nc_positions": nc_positions, "nc_classes": {}, "all_classes": {"word": n},
            "seq_flags": {str(L): any(p < L for p in nc_positions) for L in (256, 1024, 4096)},
            "excluded_utf8": 0, "excluded_truncated": 0, "finish_reason": finish, "correct": correct,
            "think_closed": None, "entropy_mean": 0.5, "entropy_at_nc": [], "span_shapes": {}, "fragment_events": 0,
        }

    rows = [row(300, [5])] + [row(300, [])] * 3 + [row(5000, [], finish="length")]
    s = summarize(rows)
    assert s["seq_flag_rate"]["256"] == 0.2  # 1 of the 5 rollouts that reached 256
    assert s["seq_flag_rate"]["4096"] == 0.0  # only the truncated rollout reached 4096
    assert s["accuracy"] == 1.0 and s["by_outcome"]["truncated"]["rollouts"] == 1
    assert set(s["by_length_quartile"]) == {"q1", "q2", "q3", "q4"}
