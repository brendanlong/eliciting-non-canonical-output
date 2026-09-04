"""Alignment pieces of the divergence measurement (CPU, real tokenizer)."""

import random

import pytest

from noncanon.divergence import build_pair, canonical_suffix, canonical_tail, shared_boundaries, split_word
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


def test_build_pair_keeps_the_full_prefix_and_retokenizes_from_the_span(an):
    prefix = enc(an, "The quick brown fox jumps over the lazy dog. Then the word is")
    light, house = enc(an, " light"), enc(an, "house")
    tail = enc(an, " and the story continues for a while after that.")
    ids = prefix + light + house + tail
    pair = build_pair(an, [1, 2, 3], ids, len(prefix), 2, after=100)
    assert pair is not None
    assert pair["a"] == [1, 2, 3] + ids and pair["b"][: 3 + len(prefix)] == [1, 2, 3] + prefix and pair["b"] != pair["a"]
    assert pair["prefix_tokens"] == len(prefix) and pair["offset"] == 3 + len(prefix)
    for byte, ia, ib in pair["bounds"]:  # shared boundaries are the same byte offset in both tails
        assert byte >= pair["span_end_byte"]
        assert sum(len(x) for x in an.token_bytes(pair["a"][pair["offset"]: pair["offset"] + ia + 1])) == byte == sum(len(x) for x in an.token_bytes(pair["b"][pair["offset"]: pair["offset"] + ib + 1]))
    assert pair["bounds"][0][0] == pair["span_end_byte"] and len(pair["bounds"]) >= len(tail) - 1
    # an earlier non-canonical span in the prefix is kept verbatim in both sequences
    ids2 = enc(an, "First") + light + house + enc(an, " then much later again the word is") + light + house + tail
    p2 = len(ids2) - len(tail) - 2
    pair2 = build_pair(an, [], ids2, p2, 2, after=100)
    assert pair2 is not None and pair2["a"][:p2] == ids2[:p2] == pair2["b"][:p2]


def test_build_pair_returns_none_when_nothing_differs(an):
    ids = enc(an, "A perfectly ordinary sentence with nothing unusual in it at all.")
    assert build_pair(an, [], ids, 3, 1, 100) is None


def test_canonical_suffix_and_tail_splice(an):
    ids = enc(an, "Some words here and then the tokenization continues normally.")
    assert canonical_suffix(an, ids) == ids[-min(64, len(ids)):]
    light, house = enc(an, " light"), enc(an, "house")
    prefix = enc(an, "Prefix text") + light + house  # ends with a non-canonical pair: the suffix stops before it
    assert canonical_suffix(an, prefix) == house  # stops before the non-canonical " light"+"house" pair
    assert canonical_tail(an, enc(an, "Prefix text"), light + house + enc(an, " more")) == enc(an, " lighthouse more")


def test_split_word_gives_two_vocab_tokens_with_the_same_bytes(an):
    t = enc(an, " tokenization")[0]
    pieces = split_word(an, t, random.Random(0))
    assert pieces is not None and len(pieces) == 2 and pieces != [t]
    assert b"".join(an.token_bytes(pieces)) == an.token_bytes([t])[0]
    assert split_word(an, enc(an, " a")[0], random.Random(0)) is None


def test_build_pair_rejects_a_tail_cut_inside_a_character(an):
    bad = next(t for t in range(0, 100256) if (b := an.token_bytes([t])[0]) and (b[0] & 0xC0) == 0x80 and len(b) == 1)
    prefix, light, house = enc(an, "The word is"), enc(an, " light"), enc(an, "house")
    ids = prefix + light + house + enc(an, " and more text follows here") + [bad]
    assert build_pair(an, [], ids, len(prefix), 2, after=100) is None
    assert build_pair(an, [], ids[:-1], len(prefix), 2, after=100) is not None


def test_contagion_pair_ends_where_the_target_token_starts(an):
    from noncanon.divergence import build_contagion_pair

    prefix, light, house = enc(an, "The word is"), enc(an, " light"), enc(an, "house")
    middle = enc(an, " and then a while later the next word is")
    ids = prefix + light + house + middle + enc(an, " sunflower")
    p2 = len(prefix) + 2 + len(middle)
    pair = build_contagion_pair(an, [9], ids, len(prefix), p2)
    assert pair is not None and pair["a"] == [9] + ids[:p2] and pair["b"] != pair["a"]
    assert b"".join(an.token_bytes(pair["a"][1:])) == b"".join(an.token_bytes(pair["b"][1:]))  # same text, so the next token starts at the same byte
    assert build_contagion_pair(an, [], enc(an, "Nothing non-canonical anywhere here at all"), 2, 6) is None
