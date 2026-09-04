"""Alignment pieces of the divergence measurement (CPU, real tokenizer)."""

import random

import pytest

from noncanon.divergence import build_pair, canonical_start, shared_boundaries, split_word
from noncanon.metrics import Analyzer


@pytest.fixture(scope="module")
def an():
    return Analyzer("allenai/Olmo-3-7B-Think")


def enc(an, text):
    return an.tok.encode(text, add_special_tokens=False)


def test_shared_boundaries_are_common_token_ends_after_the_span():
    a = [b"ab", b"c", b"def", b"g"]  # ends 2, 3, 6, 7
    b = [b"abc", b"de", b"f", b"g"]  # ends 3, 5, 6, 7
    assert shared_boundaries(a, b, after_byte=2) == [(3, 1, 0), (6, 2, 2), (7, 3, 3)]
    assert shared_boundaries(a, b, after_byte=6) == [(6, 2, 2), (7, 3, 3)]  # the span end itself is included


def test_build_pair_aligns_emitted_and_canonical_windows(an):
    prefix = enc(an, "The quick brown fox jumps over the lazy dog. Then the word is")
    light, house = enc(an, " light"), enc(an, "house")
    tail = enc(an, " and the story continues for a while after that.")
    ids = prefix + light + house + tail
    pair = build_pair(an, [1, 2, 3], ids, len(prefix), 2, before=1000, after=100)
    assert pair is not None
    assert pair["a"][:3] == [1, 2, 3] and pair["a"][3:] == ids and pair["prefix_tokens"] == len(prefix) and not pair["prefix_truncated"]
    assert pair["b"][3:3 + len(prefix)] == prefix and pair["b"] != pair["a"]
    # every shared boundary lies after the span and both indices point at tokens ending on the same byte
    for byte, ia, ib in pair["bounds"]:
        assert byte >= pair["span_end_byte"]
        assert sum(len(x) for x in an.token_bytes(pair["a"][3:3 + ia + 1])) == byte == sum(len(x) for x in an.token_bytes(pair["b"][3:3 + ib + 1]))
    assert pair["bounds"][0][0] == pair["span_end_byte"] and len(pair["bounds"]) >= len(tail) - 1


def test_build_pair_returns_none_when_nothing_differs(an):
    ids = enc(an, "A perfectly ordinary sentence with nothing unusual in it at all.")
    assert build_pair(an, [], ids, 3, 1, 100, 100) is None


def test_canonical_start_shifts_past_a_word_cut(an):
    ids = enc(an, "Some words here and then the tokenization continues normally.")
    assert canonical_start(an, ids, 0, len(ids)) == 0
    light, house = enc(an, " light"), enc(an, "house")
    ids = enc(an, "Prefix text") + light + house + enc(an, " more")
    # starting inside the non-canonical pair does not re-encode to itself; the search moves past it
    s = canonical_start(an, ids, len(enc(an, "Prefix text")), len(ids))
    assert s is not None and s >= len(enc(an, "Prefix text")) + 1


def test_split_word_gives_two_vocab_tokens_with_the_same_bytes(an):
    t = enc(an, " tokenization")[0]
    pieces = split_word(an, t, random.Random(0))
    assert pieces is not None and len(pieces) == 2 and pieces != [t]
    assert b"".join(an.token_bytes(pieces)) == an.token_bytes([t])[0]
    assert split_word(an, enc(an, " a")[0], random.Random(0)) is None


def test_build_pair_rejects_a_window_cut_inside_a_character(an):
    # A window whose last token is a lone continuation byte decodes with U+FFFD, so the
    # canonical re-encoding has different bytes; the pair must be refused, not misaligned.
    bad = next(t for t in range(0, 100256) if (b := an.token_bytes([t])[0]) and (b[0] & 0xC0) == 0x80 and len(b) == 1)
    prefix, light, house = enc(an, "The word is"), enc(an, " light"), enc(an, "house")
    ids = prefix + light + house + enc(an, " and more text follows here") + [bad]
    assert build_pair(an, [], ids, len(prefix), 2, before=100, after=100) is None
    assert build_pair(an, [], ids[:-1], len(prefix), 2, before=100, after=100) is not None
