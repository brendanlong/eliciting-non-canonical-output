# Results log

One entry per run, in order. Exact commands, configuration, and the numbers
as produced; prediction status against `EXPERIMENT_PLAN.md`. Interpretation
is kept out of this file.

## Prompt sets (2026-09-03)

```
uv run python -m noncanon.prompts dapo --pilot 50
```

`open-r1/DAPO-Math-17k-Processed` (config `en`, split `train`): 14,116 rows.
Filtered against the OLMo-3 RL training prompts by normalized text
(NFKC, lowercase, `user:` prefix stripped, then **alphanumerics only**),
exact or shared 80-character prefix:

| removed because | rows |
|---|--:|
| exact match in `Dolci-RL-Zero-Math-7B` (13,314 prompts) | 9,886 |
| prefix match in `Dolci-RL-Zero-Math-7B` | 154 |
| exact match in `Dolci-Think-RL-7B` (102,014 prompts) | 52 |
| prefix match in `Dolci-Think-RL-7B` | 35 |
| duplicate within DAPO | 735 |
| non-integer answer | 0 |
| **kept** (`prompts/dapo_heldout.jsonl`) | **3,254** |

Full report: `prompts/dapo_filter_report.json`. Pilot sample:
50 problems drawn with `random.Random(0)` from the kept set
(`prompts/dapo_pilot50.jsonl`).

Earlier version (commit `6bb02b2`, used for the pilot run below):
normalization was whitespace-collapsing only, which kept 3,430 problems;
code review found 77 of those were byte-identical to a training prompt
once punctuation was stripped (LaTeX spacing differs between copies), so
the normalization was tightened. The pilot's 50 prompts were drawn from
the 3,430-set and that file is stored with the pilot artifacts.

## Pilot: Olmo-3-7B-Think, 50 DAPO prompts, both arms (2026-09-03)

Purpose (from the plan): measure the rollout length distribution and
truncation rate, and exercise ID capture, the round-trip metric, the
verifier and the upload path end to end.

```
sky launch -c noncanon-pilot skypilot/pilot.yaml -i 20 --down -y -d --env HF_TOKEN
```

Configuration: `allenai/Olmo-3-7B-Think` @ `main`, bf16 weights and KV
cache, vLLM 0.11.0 / torch 2.8.0 (CUDA 12.8), `max_tokens=32768`,
`max_model_len=34816`, 1 sample per prompt, top-10 logprobs stored per
position, no speculative decoding. Arms: `recommended` (temperature 0.6,
top_p 0.95, from the checkpoint's `generation_config.json`) and
`untruncated` (temperature 1.0, top_p 1.0); `top_k=-1`, `min_p=0`,
no repetition penalty in both. Prompt = the bare problem as the user turn
under the model's default chat template (which opens the assistant turn
with `<think>`). GPU: RunPod A100-80GB (CA-MTL), $1.39/h.

Artifacts: `pilot/` on `brendanlong/noncanonical-post-training`
(`<arm>.jsonl.gz` rollouts with IDs and logprobs, `<arm>.meta.json`,
`metrics/summary.{json,md}`, `metrics/analysis.jsonl`,
`metrics/examples.jsonl`).

_Results pending._
