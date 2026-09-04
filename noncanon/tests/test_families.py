"""The pieces that differ between model families: tokenizer byte mapping,
chat-template turn end, and the recommended sampling setting. CPU only;
imports vLLM-free helpers via a stub so generate.py's vllm import is not needed."""

import importlib
import sys
import types

import pytest
from transformers import AutoTokenizer, GenerationConfig

from noncanon.metrics import _BYTE_DECODER, Analyzer

FAMILIES = {
    "olmo3": ("allenai/Olmo-3-7B-Instruct", {"temperature": 0.6, "top_p": 0.95}, "<|im_end|>"),
    "tulu3": ("allenai/Llama-3.1-Tulu-3-8B-SFT", {"temperature": 0.6, "top_p": 0.9}, "<|end_of_text|>"),  # Tulu uses plain-text role markers; turns end with EOS
}


@pytest.fixture(scope="module")
def gen():
    # generate.py imports vllm/torch at module level; stub them so the pure
    # helpers (resolve_arms, stop_token_ids) can be tested on CPU.
    for name in ("vllm", "torch"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["vllm"].LLM = sys.modules["vllm"].SamplingParams = sys.modules["vllm"].TokensPrompt = object
    sys.modules["vllm"].__version__ = "stub"
    return importlib.import_module("noncanon.generate")


@pytest.mark.parametrize("family", list(FAMILIES))
def test_every_token_maps_through_the_byte_decoder(family):
    an = Analyzer(FAMILIES[family][0])
    bad = [t for t in range(len(an.tok)) if t not in an.special and any(c not in _BYTE_DECODER for c in an.tok.convert_ids_to_tokens(t))]
    assert bad == []


@pytest.mark.parametrize("family", list(FAMILIES))
def test_round_trip_and_split_word(family):
    an = Analyzer(FAMILIES[family][0])
    ids = an.tok.encode("The quick brown fox jumps over 12345 lazy dogs.\n\nDone, 2024-09-03.", add_special_tokens=False)
    a = an.analyze({"prompt_token_ids": [], "token_ids": ids, "finish_reason": "stop", "answer": None, "topk_logprobs": [[-0.1, -3.0]] * len(ids)})
    assert a["nc_canonical"] == 0 and a["fragment_events"] == 0 and a["n_tokens"] == len(ids)
    light, house = an.tok.encode("light", add_special_tokens=False), an.tok.encode("house", add_special_tokens=False)
    ids = an.tok.encode("The word is ", add_special_tokens=False) + light + house
    a = an.analyze({"prompt_token_ids": [], "token_ids": ids, "finish_reason": "stop", "answer": None, "topk_logprobs": [[-0.1, -3.0]] * len(ids)})
    assert a["nc_spans"] == 1


@pytest.mark.parametrize("family", list(FAMILIES))
def test_recommended_setting_and_stop_tokens_come_from_the_checkpoint(gen, family):
    model, expected, turn_end = FAMILIES[family]
    tok = AutoTokenizer.from_pretrained(model)
    cfg = GenerationConfig.from_pretrained(model)
    assert gen.resolve_arms(["recommended", "untruncated"], cfg)["recommended"] == expected
    stops = gen.stop_token_ids(tok, cfg)
    assert tok.convert_tokens_to_ids(turn_end) in stops
    assert tok.eos_token_id in stops
