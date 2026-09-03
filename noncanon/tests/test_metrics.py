"""CPU tests for the round-trip metric, on the real OLMo-3 tokenizer."""

import pytest

from noncanon.metrics import Analyzer, extract_boxed, parse_int, token_class, verify

TOKENIZER = "allenai/Olmo-3-7B-Think"


@pytest.fixture(scope="module")
def an():
    return Analyzer(TOKENIZER)


def make_record(an, prompt_text, output_ids, answer=None):
    return {
        "prompt_id": "t",
        "sample": 0,
        "arm": "test",
        "prompt_token_ids": an.tok.encode(prompt_text, add_special_tokens=False),
        "token_ids": output_ids,
        "finish_reason": "stop",
        "answer": answer,
        "topk_logprobs": [[-0.1, -3.0, -4.0]] * len(output_ids),
    }


def test_canonical_text_has_zero_rate(an):
    ids = an.tok.encode("The quick brown fox jumps over 12345 lazy dogs.\n\nDone, 2024-09-03.", add_special_tokens=False)
    a = an.analyze(make_record(an, "<|im_start|>assistant\n", ids))
    assert a["nc_tokens"] == 0 and a["nc_spans"] == 0
    assert a["n_tokens"] == len(ids)
    assert not a["has_think"]


def test_split_word_is_detected_as_one_span(an):
    light, house = an.tok.encode("light", add_special_tokens=False), an.tok.encode("house", add_special_tokens=False)
    canonical = an.tok.encode("lighthouse", add_special_tokens=False)
    assert light + house != canonical, "test needs a word whose split differs from canonical"
    # "The word is " ends in a bare space token that canonically merges into
    # " l", so the span covers the space plus both word pieces.
    prefix = an.tok.encode("The word is ", add_special_tokens=False)
    ids = prefix + light + house
    a = an.analyze(make_record(an, "<|im_start|>assistant\n", ids))
    assert a["nc_spans"] == 1
    assert a["n_tokens"] == len(ids)
    span = a["spans"][0]
    assert a["nc_tokens"] == len(span["emitted"]) >= len(light) + len(house)
    assert "".join(span["emitted"]) == "".join(span["canonical"]) == " lighthouse"
    assert span["emitted"] != span["canonical"]
    assert a["seq_flags"]["256"] is True
    # The same bytes tokenized canonically: zero.
    b = an.analyze(make_record(an, "<|im_start|>assistant\n", an.tok.encode("The word is lighthouse", add_special_tokens=False)))
    assert b["nc_tokens"] == 0


def test_special_tokens_are_excluded(an):
    body = an.tok.encode("Hello there.", add_special_tokens=False)
    ids = body + [an.tok.convert_tokens_to_ids("<|im_end|>"), an.tok.eos_token_id]
    a = an.analyze(make_record(an, "<|im_start|>assistant\n", ids))
    assert a["n_tokens"] == len(body)
    assert a["nc_tokens"] == 0


def test_interior_special_token_splits_segments_and_positions_are_ordinals(an):
    light, house = an.tok.encode("light", add_special_tokens=False), an.tok.encode("house", add_special_tokens=False)
    im_end = an.tok.convert_tokens_to_ids("<|im_end|>")
    ids = light + house + [im_end] + light + house
    a = an.analyze(make_record(an, "<|im_start|>assistant\n", ids))
    assert a["n_tokens"] == 4 and a["nc_spans"] == 2 and a["nc_tokens"] == 4
    assert a["nc_positions"] == [0, 1, 2, 3]  # ordinals among measured tokens, not raw indices


def test_think_split_and_verifier(an):
    text = "Let me think. 6*7 is 42.</think>\n\nThe answer is \\boxed{42}."
    ids = an.tok.encode(text, add_special_tokens=False)
    a = an.analyze(make_record(an, "<|im_start|>assistant\n<think>", ids, answer="42"))
    assert a["has_think"] and a["think_closed"]
    assert a["n_think"] + a["n_answer"] == a["n_tokens"]
    assert a["n_think"] > 0 and a["n_answer"] > 0
    assert a["pred"] == "42" and a["correct"] is True


def test_think_boundary_token_straddling_answer(an):
    # "></think>\" can be one token; the answer bytes after the tag must still be verified.
    text = "done</think>\\boxed{42}"
    ids = an.tok.encode(text, add_special_tokens=False)
    a = an.analyze(make_record(an, "<|im_start|>assistant\n<think>", ids, answer="42"))
    assert a["think_closed"]
    assert a["pred"] == "42" and a["correct"] is True


def test_incomplete_utf8_tail_is_excluded_not_counted(an):
    # Find a vocab token that is an incomplete 4-byte UTF-8 prefix (the first
    # bytes of an emoji), which is what a max_tokens cut mid-character leaves.
    partial = next(
        t for t in range(1000, 100256)
        if (b := an.token_bytes([t])[0]) and (b[0] & 0xF0) == 0xF0 and len(b) < 4
    )
    ids = an.tok.encode("Some text and then an emoji", add_special_tokens=False) + [partial]
    a = an.analyze(make_record(an, "<|im_start|>assistant\n", ids))
    assert a["excluded_tokens"] == 1
    assert a["n_tokens"] == len(ids) - 1
    assert a["nc_tokens"] == 0


def test_unclosed_think_is_all_cot(an):
    ids = an.tok.encode("Still thinking about it", add_special_tokens=False)
    a = an.analyze(make_record(an, "<|im_start|>assistant\n<think>", ids, answer="1"))
    assert a["has_think"] and a["think_closed"] is False
    assert a["n_think"] == a["n_tokens"] and a["n_answer"] == 0
    assert a["correct"] is None


def test_verifier_formats():
    assert extract_boxed(r"so \boxed{\frac{1}{2}} then \boxed{1,234}") == "1,234"
    assert parse_int("1,234") == 1234 and parse_int("$-7$") == -7 and parse_int("abc") is None
    assert parse_int("1{,}234") == 1234 and parse_int("42.0") == 42 and parse_int("042") == 42
    assert verify("Answer: 99", "99") == ("99", True)
    assert verify("Answer: 98", "99") == ("98", False)
    assert verify("**Answer:** 42", "42")[1] is True
    assert verify("Answer: **42**", "42")[1] is True
    assert verify("So the answer is 42.", "42")[1] is True
    assert verify("Final Answer: 7", "7")[1] is True
    assert verify("no answer here", "99") == (None, None)
    # A boxed non-integer is a wrong answer to an integer question, not "unparsed".
    assert verify(r"\boxed{x+1}", "3") == ("x+1", False)


def test_summarize_denominators():
    from noncanon.metrics import summarize

    def row(n_tokens, nc_positions, finish="stop", correct=True):
        return {
            "n_tokens": n_tokens, "n_think": 0, "n_answer": n_tokens, "nc_tokens": len(nc_positions),
            "nc_think": 0, "nc_answer": len(nc_positions), "nc_spans": len(nc_positions),
            "nc_positions": nc_positions, "nc_classes": {}, "all_classes": {"word": n_tokens},
            "seq_flags": {"256": any(p < 256 for p in nc_positions), "1024": any(p < 1024 for p in nc_positions), "4096": any(p < 4096 for p in nc_positions)},
            "excluded_tokens": 0, "finish_reason": finish, "correct": correct, "think_closed": None,
            "entropies": [], "entropy_at_nc": [],
        }

    rows = [row(300, [5])] + [row(300, [])] * 3 + [row(5000, [], finish="length", correct=True)]
    s = summarize(rows)
    assert s["seq_flag_rate"]["256"] == 0.2          # 1 of 5 rollouts reached 256
    assert s["seq_flag_rate"]["4096"] == 0.0         # only the truncated rollout reached 4096
    assert s["accuracy"] == 1.0 and s["by_outcome"]["truncated"]["rollouts"] == 1
    assert sum(v["rollouts"] for v in s["by_length_quartile"].values()) == 5
    assert set(s["by_length_quartile"]) == {"q1", "q2", "q3", "q4"}


def test_token_classes():
    assert token_class(" the") == "word"
    assert token_class("123") == "digit"
    assert token_class("\n\n") == "whitespace"
    assert token_class(" 3x") == "mixed"
    assert token_class("),") == "symbol"
