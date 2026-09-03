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
`dapo_pilot50.jsonl` as used, `metrics/summary.{json,md}`,
`metrics/analysis.jsonl`, `metrics/examples.jsonl`). The metrics in the
dataset were recomputed locally with the review-fixed `metrics.py`
(commit `556b56f` onward) and overwrite the on-box copy; the raw records
are unchanged. Job: SkyPilot job 1 on `noncanon-pilot`, SUCCEEDED,
41m 51s wall clock including setup, cluster torn down afterwards.

**Throughput** (A100-80GB, 50 concurrent requests submitted, vLLM KV
budget 119,312 tokens, max concurrency 6.65 at 34,816 tokens):

| arm | rollouts | output tokens | wall time | output tokens/s |
|---|--:|--:|--:|--:|
| recommended (0.6 / 0.95) | 50 | 468,362 | 1,178.5 s | 397.4 |
| untruncated (1.0 / 1.0) | 50 | 465,346 | 1,103.3 s | 421.8 |

**Rollout length and outcome** (measured tokens exclude the trailing stop
token; tokens excluded as incomplete UTF-8: 0 in both arms):

| arm | mean | median | p90 | max | finish | think closed | verified correct |
|---|--:|--:|--:|--:|---|---|---|
| recommended | 9,366 | 8,526 | 12,666 | 32,768 | 49 stop, 1 length | 49 / 50 | 49 / 49 finished |
| untruncated | 9,306 | 8,612 | 14,449 | 30,245 | 50 stop | 50 / 50 | 49 / 50 |

**Non-canonical tokens** (per-token rate = emitted tokens inside a
non-canonical span / measured tokens):

| arm | tokens | non-canonical | rate | spans | rollouts with ≥1 | in think | in answer |
|---|--:|--:|--:|--:|--:|---|---|
| recommended | 468,313 | 0 | 0.000% | 0 | 0 / 50 | 0 / 439,887 | 0 / 28,426 |
| untruncated | 465,296 | 20 | 0.0043% | 10 | 6 / 50 | 20 / 435,760 (0.0046%) | 0 / 29,536 |

Untruncated arm, further slices (20 events; small counts):

- by token class: whitespace 1 / 21,895; digit 0 / 77,219; word
  14 / 220,526; mixed 2 / 9,191; symbol 3 / 136,465.
- by outcome: correct 18 / 435,051 (49 rollouts); incorrect 2 / 30,245
  (1 rollout).
- by rollout-length quartile (rank-based): q1 4 / 65,503; q2 0 / 88,918;
  q3 0 / 126,421; q4 16 / 184,454.
- by relative position decile (0 to 9): 0, 0, 4, 0, 2, 2, 0, 4, 0, 8
  events; decile 9 rate 0.017%.
- sequence-level flag (≥1 event within the first L tokens, over
  rollouts that reached L): L=256 0 / 50; L=1024 0 / 50; L=4096 3 / 48.
- top-10 entropy (nats, renormalized): mean over all positions 0.359;
  mean at the 20 non-canonical positions 1.010.

**All ten spans in the untruncated arm** (emitted tokens → canonical
tokens; `·` marks a space):

| rollout | pos | preceding text | emitted | canonical |
|---|--:|---|---|---|
| b925913f | 1,303 | `Hundreds  Tens  Units\n` | `·····`, `WR` | `····`, `·WR` |
| 639f2d08 | 16,089 | `points below the linex=y and` | `·above`, `x` | `·ab`, `ov`, `ex` |
| 639f2d08 | 16,259 | `x can only be0;\n\n` | `as`, `y` | `asy` |
| 639f2d08 | 19,859 | `so x and y are roots` | `·of`, `t` | `·oft` |
| 639f2d08 | 19,984 | `andx+y-2xy cannot both` | `·equal`, `k` | `·equ`, `alk` |
| 0a947bc6 | 3,193 | `where we want A not B` | `_inter`, `sect` | `_intersect` |
| 607f851b | 3,419 | `with 5, divisible by 5` | `.`, `Composite` | `.Com`, `posite` |
| e4c4901d | 29,052 | `⇒m≡2 mod4.\n\n` | `Th`, `uem` | `Thu`, `em` |
| 6d7a828c | 6,345 | `= z^3\n\nz^3` | `·*`, `\` | `·*\` |
| 6d7a828c | 12,269 | `no, the dot product is:\n\n` | `(O`, `G` | `(`, `OG` |

Rollout 639f2d08 is the one verified incorrect (30,245 tokens, the
longest in the arm); it carries 4 of the 10 spans.

**Checks run on the pilot output** (all in the session log; none
committed as code beyond the tests):

- An independent round trip with plain tokenizer calls on the raw
  non-special IDs of every rollout agrees with the metric: 0 / 50
  mismatching rollouts in the recommended arm.
- An injected split of the token ` numbers` into ` nu`, `m`, `bers`
  inside a real rollout is detected as exactly one span of three tokens
  against the canonical single token.
- Verifier predictions were listed against gold for all 50 recommended
  rollouts: 49 matches, 1 `None` (the rollout that hit the cap with its
  think block open).
- One transcript (the eighth record of the recommended arm, a mean-line
  problem over five complex numbers, 5,689 tokens) was read at its start
  and end: coherent reasoning ending in a boxed answer and the
  end-of-text token. The ten spans above were read with their context.

**Prediction status.** The pilot is a single checkpoint, so none of the
before/after predictions are testable from it. Prediction 1's bound
(<0.1% per token on in-distribution text for a text-distilled-then-RL'd
model) holds for this checkpoint at both settings. Prediction 2's digit
bound holds (0 / 77,219 and 0 / 84,853 digit tokens).
