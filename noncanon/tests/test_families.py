"""The pieces that differ between model families: tokenizer byte mapping,
chat-template turn end, and the recommended sampling setting. CPU only;
imports vLLM-free helpers via a stub so generate.py's vllm import is not needed."""

import importlib
import importlib.machinery
import importlib.util
import sys
import types
from unittest import mock

import pytest
from transformers import AutoTokenizer, GenerationConfig

from noncanon.metrics import _BYTE_DECODER, Analyzer

# model, recommended setting (None = generation_config sets none), expected stop token strings
FAMILIES = {
    "olmo3": ("allenai/Olmo-3-7B-Instruct", {"temperature": 0.6, "top_p": 0.95}, {"<|im_end|>", "<|endoftext|>"}),
    "olmo3-zero": ("allenai/Olmo-3-7B-RL-Zero-Math", None, {"<|im_end|>", "<|endoftext|>"}),  # no eos in generation_config, no specials in the template
    "tulu3": ("allenai/Llama-3.1-Tulu-3-8B-SFT", {"temperature": 0.6, "top_p": 0.9}, {"<|end_of_text|>"}),  # plain-text role markers; turns end with EOS
}


def _stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)  # find_spec() must not raise on it
    return m


@pytest.fixture(scope="module")
def gen():
    # generate.py imports vllm/torch at module level. In the CPU test env
    # they are absent; stub only what is missing, only for the duration of
    # the module, and never touch a real installation.
    stubs = {}
    for name in ("vllm", "torch"):
        if importlib.util.find_spec(name) is None:
            stubs[name] = _stub(name)
    if "vllm" in stubs:
        stubs["vllm"].LLM = stubs["vllm"].SamplingParams = stubs["vllm"].TokensPrompt = object
        stubs["vllm"].__version__ = "stub"
    with mock.patch.dict(sys.modules, stubs):
        sys.modules.pop("noncanon.generate", None)
        yield importlib.import_module("noncanon.generate")
        sys.modules.pop("noncanon.generate", None)


@pytest.mark.parametrize("family", ["olmo3", "tulu3"])
def test_every_token_maps_through_the_byte_decoder(family):
    an = Analyzer(FAMILIES[family][0])
    bad = [t for t in range(len(an.tok)) if t not in an.special and any(c not in _BYTE_DECODER for c in an.tok.convert_ids_to_tokens(t))]
    assert bad == []


@pytest.mark.parametrize("family", ["olmo3", "tulu3"])
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
    model, expected, stop_strings = FAMILIES[family]
    tok = AutoTokenizer.from_pretrained(model)
    cfg = GenerationConfig.from_pretrained(model)
    if expected is None:
        with pytest.raises(AssertionError):  # must not silently fall back to the class defaults (1.0 / 1.0)
            gen.resolve_arms(["recommended"], cfg)
    else:
        assert gen.resolve_arms(["recommended", "untruncated"], cfg)["recommended"] == expected
    assert set(gen.stop_token_ids(tok, cfg)) == {tok.convert_tokens_to_ids(t) for t in stop_strings}
