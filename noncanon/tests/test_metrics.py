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


def test_think_split_and_verifier(an):
    text = "Let me think. 6*7 is 42.</think>\n\nThe answer is \\boxed{42}."
    ids = an.tok.encode(text, add_special_tokens=False)
    a = an.analyze(make_record(an, "<|im_start|>assistant\n<think>", ids, answer="42"))
    assert a["has_think"] and a["think_closed"]
    assert a["n_think"] + a["n_answer"] == a["n_tokens"]
    assert a["n_think"] > 0 and a["n_answer"] > 0
    assert a["pred"] == "42" and a["correct"] is True


def test_unclosed_think_is_all_cot(an):
    ids = an.tok.encode("Still thinking about it", add_special_tokens=False)
    a = an.analyze(make_record(an, "<|im_start|>assistant\n<think>", ids, answer="1"))
    assert a["has_think"] and a["think_closed"] is False
    assert a["n_think"] == a["n_tokens"] and a["n_answer"] == 0
    assert a["correct"] is None


def test_verifier_formats():
    assert extract_boxed(r"so \boxed{\frac{1}{2}} then \boxed{1,234}") == "1,234"
    assert parse_int("1,234") == 1234 and parse_int("$-7$") == -7 and parse_int("abc") is None
    assert verify("Answer: 99", "99") == ("99", True)
    assert verify("Answer: 98", "99") == ("98", False)
    assert verify("no answer here", "99") == (None, None)
    assert verify(r"\boxed{x+1}", "3") == ("x+1", None)


def test_token_classes():
    assert token_class(" the") == "word"
    assert token_class("123") == "digit"
    assert token_class("\n\n") == "whitespace"
    assert token_class(" 3x") == "mixed"
    assert token_class("),") == "symbol"
