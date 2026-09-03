"""Generate rollouts with vLLM and keep the emitted token IDs.

    uv run python -m noncanon.generate --model allenai/Olmo-3-7B-Think \
        --prompts prompts/dapo_pilot50.jsonl --arms recommended,untruncated \
        --out-dir out/pilot

Writes one ``<arm>.jsonl.gz`` per sampling arm plus ``<arm>.meta.json``. Every
record carries the prompt token IDs, the emitted token IDs, the finish
reason, the sampled token's logprob at each position and the top-k
(id, logprob) candidates at each position, so every metric can be recomputed
offline from the file without trusting anything computed here.

Sampling is pinned explicitly: ``top_k=-1``, ``min_p=0``, no repetition
penalty, bf16 weights and KV cache, no speculative decoding. Logprobs are
vLLM's raw model logprobs (before temperature/top-p), which is what the
entropy analysis wants.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams, TokensPrompt

# "recommended" is the generation_config.json shipped with every OLMo-3
# checkpoint (Think, Think-DPO, Instruct all carry temperature 0.6 / top_p 0.95).
# "untruncated" is the full distribution.
ARMS = {
    "recommended": {"temperature": 0.6, "top_p": 0.95},
    "untruncated": {"temperature": 1.0, "top_p": 1.0},
}


def load_prompts(path: Path, field: str, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if limit:
        rows = rows[:limit]
    for r in rows:
        assert field in r, f"prompt field {field!r} missing in {path}"
    return rows


def topk_lists(lp_dict) -> tuple[list[int], list[float]]:
    items = sorted(lp_dict.items(), key=lambda kv: kv[1].logprob, reverse=True)
    return [int(k) for k, _ in items], [float(v.logprob) for _, v in items]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--prompt-field", default="problem")
    ap.add_argument("--arms", default="recommended,untruncated")
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
    for a in arms:
        assert a in ARMS, f"unknown arm {a!r}; known: {list(ARMS)}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    gen_cfg = GenerationConfig.from_pretrained(args.model, revision=args.revision)
    eos = gen_cfg.eos_token_id
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
        disable_log_stats=False,  # periodic "Avg generation throughput / Running / KV usage" lines in the log
    )
    logprobs_mode = getattr(getattr(llm.llm_engine, "model_config", None), "logprobs_mode", "raw_logprobs (vllm default)")

    for arm in arms:
        sp = SamplingParams(
            n=args.n,
            max_tokens=args.max_tokens,
            logprobs=args.logprobs,
            stop_token_ids=eos_ids,
            skip_special_tokens=False,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            **ARMS[arm],
        )
        t0 = time.time()
        outputs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids], sp, use_tqdm=True)
        elapsed = time.time() - t0

        n_tokens = 0
        out_path = args.out_dir / f"{arm}.jsonl.gz"
        with gzip.open(out_path, "wt") as f:
            for p, ids, out in zip(prompts, prompt_ids, outputs):
                for j, comp in enumerate(out.outputs):
                    n_tokens += len(comp.token_ids)
                    sampled, topk_ids, topk_lps = [], [], []
                    for t, lp in zip(comp.token_ids, comp.logprobs or []):
                        sampled.append(float(lp[t].logprob))
                        ks, ls = topk_lists(lp)
                        topk_ids.append(ks)
                        topk_lps.append(ls)
                    rec = {
                        "prompt_id": p.get("id"),
                        "sample": j,
                        "model": args.model,
                        "revision": args.revision,
                        "arm": arm,
                        "sampling": {**ARMS[arm], "top_k": -1, "min_p": 0.0, "max_tokens": args.max_tokens, "n": args.n},
                        "problem": p[args.prompt_field],
                        "answer": p.get("answer"),
                        "prompt_token_ids": list(ids),
                        "token_ids": list(comp.token_ids),
                        "text": comp.text,
                        "finish_reason": comp.finish_reason,
                        "logprobs": sampled,
                        "topk_ids": topk_ids,
                        "topk_logprobs": topk_lps,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        import vllm  # noqa: PLC0415  (version only)

        meta = {
            "model": args.model,
            "revision": args.revision,
            "arm": arm,
            "sampling": {**ARMS[arm], "top_k": -1, "min_p": 0.0, "max_tokens": args.max_tokens, "n": args.n},
            "prompts": str(args.prompts),
            "n_prompts": len(prompts),
            "n_completions": sum(len(o.outputs) for o in outputs),
            "output_tokens": n_tokens,
            "elapsed_s": round(elapsed, 1),
            "output_tokens_per_s": round(n_tokens / elapsed, 1) if elapsed else None,
            "eos_token_ids": eos_ids,
            "logprobs_topk": args.logprobs,
            "logprobs_mode": str(logprobs_mode),
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
            "max_model_len": args.max_model_len,
            "seed": args.seed,
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(),
        }
        (args.out_dir / f"{arm}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
