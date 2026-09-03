"""Generate rollouts with vLLM and keep the emitted token IDs.

    uv run python -m noncanon.generate --model allenai/Olmo-3-7B-Think \
        --prompts prompts/dapo_pilot50.jsonl --arms untruncated --out-dir out/pilot

Writes one Parquet file per sampling arm (``<arm>.parquet``) plus
``<arm>.meta.json`` with the run configuration. Every row carries the prompt
token IDs, the emitted token IDs, the finish reason, the sampled token's
logprob at each position and the top-k (id, logprob) candidates at each
position, so every metric can be recomputed offline from the file.

Sampling is pinned explicitly: ``top_k=-1``, ``min_p=0``, no repetition
penalty, bf16 weights and KV cache, no speculative decoding. Logprobs are
vLLM's raw model logprobs (before temperature/top-p), which is what the
entropy analysis wants.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import vllm
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams, TokensPrompt

from noncanon.records import write_records

# "recommended" is the decoding setting the OLMo 3 report evaluated with and
# every OLMo-3 checkpoint ships in generation_config.json
# (https://arxiv.org/html/2512.13961v2#S4.SS1.SSS1). "untruncated" is the
# full distribution.
ARMS = {
    "recommended": {"temperature": 0.6, "top_p": 0.95},
    "untruncated": {"temperature": 1.0, "top_p": 1.0},
}



def load_prompts(path: Path, field: str, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    rows = rows[:limit] if limit else rows
    for r in rows:
        assert field in r, f"prompt field {field!r} missing in {path}"
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--prompt-field", default="problem")
    ap.add_argument("--arms", default="untruncated")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1, help="samples per prompt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--max-model-len", type=int, default=34816)
    ap.add_argument("--logprobs", type=int, default=10, help="top-k candidates stored per position")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(ARMS)
    assert not unknown, f"unknown arms {unknown}; known: {list(ARMS)}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    eos = GenerationConfig.from_pretrained(args.model, revision=args.revision).eos_token_id
    eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]

    prompts = load_prompts(args.prompts, args.prompt_field, args.limit)
    prompt_ids = [
        tok.apply_chat_template([{"role": "user", "content": p[args.prompt_field]}], add_generation_prompt=True, tokenize=True)
        for p in prompts
    ]
    longest = max(len(ids) for ids in prompt_ids)
    assert longest + args.max_tokens <= args.max_model_len, (
        f"longest prompt {longest} + max_tokens {args.max_tokens} exceeds max_model_len {args.max_model_len}"
    )

    llm = LLM(
        model=args.model,
        revision=args.revision,
        tokenizer_revision=args.revision,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        enable_prefix_caching=True,
        disable_log_stats=False,  # periodic throughput / running / KV-usage lines in the log
    )

    for arm in arms:
        sampling = {**ARMS[arm], "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0, "max_tokens": args.max_tokens, "n": args.n}
        params = SamplingParams(logprobs=args.logprobs, stop_token_ids=eos_ids, skip_special_tokens=False, **sampling)
        t0 = time.time()
        outputs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids], params, use_tqdm=True)
        elapsed = time.time() - t0

        rows = []
        for p, ids, out in zip(prompts, prompt_ids, outputs):
            for j, comp in enumerate(out.outputs):
                # comp.logprobs[i] maps token id -> Logprob for the top-k candidates at
                # position i, plus the sampled token if it fell outside the top-k.
                topk = [sorted(lp.items(), key=lambda kv: kv[1].logprob, reverse=True) for lp in comp.logprobs]
                rows.append(
                    {
                        "prompt_id": p.get("id"),
                        "sample": j,
                        "problem": p[args.prompt_field],
                        "answer": p.get("answer"),
                        "prompt_token_ids": ids,
                        "token_ids": comp.token_ids,
                        "text": comp.text,
                        "finish_reason": comp.finish_reason,
                        "logprobs": [lp[t].logprob for t, lp in zip(comp.token_ids, comp.logprobs)],
                        "topk_ids": [[t for t, _ in cands] for cands in topk],
                        "topk_logprobs": [[v.logprob for _, v in cands] for cands in topk],
                    }
                )
        write_records(rows, args.out_dir / f"{arm}.parquet")

        n_tokens = sum(len(r["token_ids"]) for r in rows)
        meta = {
            "model": args.model,
            "revision": args.revision,
            "arm": arm,
            "sampling": sampling,
            "prompts": str(args.prompts),
            "n_prompts": len(prompts),
            "n_completions": len(rows),
            "output_tokens": n_tokens,
            "elapsed_s": round(elapsed, 1),
            "output_tokens_per_s": round(n_tokens / elapsed, 1),
            "eos_token_ids": eos_ids,
            "logprobs_topk": args.logprobs,
            "logprobs_mode": str(llm.llm_engine.model_config.logprobs_mode),
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
            "max_model_len": args.max_model_len,
            "seed": args.seed,
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
        }
        text = json.dumps(meta, indent=2)
        (args.out_dir / f"{arm}.meta.json").write_text(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
