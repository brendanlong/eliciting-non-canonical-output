# Results log

One entry per run, in order. Exact commands, configuration, and the numbers
as produced; prediction status against `EXPERIMENT_PLAN.md`. Interpretation
is kept out of this file.

## Summary (2026-09-04)

The three comparisons the project set out to make, on the rollout-level
metric adopted on 2026-09-04 (see "Rollout-level reanalysis" below and the
plan amendment). "NC event" = a non-canonical span or a standalone byte
fragment; "parsed" = the rollout finished (not truncated at the cap) with
a boxed / "Answer:" integer; "correct" = it matched the gold integer.
Fisher exact p-values on flagged rollouts; a † marks a test or fraction
where a cell has fewer than 10 eligible rollouts (the p-value is exact but
uninformative). Each family's ladder also gets one omnibus chi-square
across all its stages ("do the stages differ at all"), under which the
pairwise p-values are post-hoc. The per-token rate is the preregistered
metric, reported but secondary. Every DAPO cell is 500 rollouts on the
same 500 held-out problems (one sample each); AIME is 60 problems × 8
samples. Tables are the output of `noncanon.summary` (commands in the
Reproduction section).

### 1. Training stages, DAPO, temperature 1 / top-p 1

| family | stage | rollouts | parsed | correct | truncated | mean tokens | rollouts with ≥1 NC event [Wilson 95%] | p vs previous stage | p vs first stage | correct rollouts with ≥1 NC | p vs previous | p vs first | within first 1,024 tokens (of rollouts ≥ 1,024) | per-token rate |
|---|---|--:|--:|--:|--:|--:|---|--:|--:|---|--:|--:|---|--:|
| OLMo-3 Think | SFT | 500 | 481 | 469 | 0 | 8,231 | 11.8% (59/500) [9.3–14.9] | — | — | 9.2% (43/469) [6.9–12.1] | — | — | 0.8% (4/500) | 0.0030% |
| OLMo-3 Think | DPO | 500 | 479 | 472 | 3 | 8,184 | 55.0% (275/500) [50.6–59.3] | < 1e-10 | < 1e-10 | 54.2% (256/472) [49.7–58.7] | < 1e-10 | < 1e-10 | 22.1% (110/497) | 0.0186% |
| OLMo-3 Think | RL final | 500 | 492 | 486 | 7 | 9,299 | 10.2% (51/500) [7.8–13.2] | < 1e-10 | 0.479 | 9.3% (45/486) [7.0–12.2] | < 1e-10 | 1.000 | 1.2% (6/500) | 0.0034% |
| OLMo-3 RL-Zero-Math | step 300 | 500 | 499 | 436 | 0 | 6,348 | 14.6% (73/500) [11.8–18.0] | — | — | 13.3% (58/436) [10.4–16.8] | — | — | 2.8% (14/493) | 0.0085% |
| OLMo-3 RL-Zero-Math | step 2000 (final) | 500 | 498 | 446 | 1 | 6,265 | 58.4% (292/500) [54.0–62.6] | < 1e-10 | < 1e-10 | 56.5% (252/446) [51.9–61.0] | < 1e-10 | < 1e-10 | 44.3% (221/499) | 0.0238% |
| OLMo-3 Instruct | SFT | 500 | 415 | 133 | 0 | 538 | 3.4% (17/500) [2.1–5.4] | — | — | 3.8% (5/133) [1.6–8.5] | — | — | 3.2% (1/31) | 0.0267% |
| OLMo-3 Instruct | DPO | 500 | 496 | 368 | 2 | 3,403 | 21.0% (105/500) [17.7–24.8] | < 1e-10 | < 1e-10 | 15.5% (57/368) [12.2–19.5] | 1.9e-04 | 1.9e-04 | 8.0% (28/348) | 0.0186% |
| OLMo-3 Instruct | RL final | 500 | 500 | 460 | 0 | 2,545 | 6.2% (31/500) [4.4–8.7] | < 1e-10 | 0.053 | 5.4% (25/460) [3.7–7.9] | 1.9e-06 | 0.509 | 2.6% (10/388) | 0.0047% |
| Tulu-3-8B | SFT | 500 | 478 | 20 | 0 | 812 | 5.2% (26/500) [3.6–7.5] | — | — | 0.0% (0/20) [0.0–16.1] | — | — | 13.2% (14/106) | 0.0342% |
| Tulu-3-8B | DPO | 500 | 498 | 54 | 0 | 943 | 3.0% (15/500) [1.8–4.9] | 0.110 | 0.110 | 1.9% (1/54) [0.3–9.8] | 1.000 | 1.000 | 3.6% (6/168) | 0.0085% |
| Tulu-3-8B | RLVR (PPO) | 500 | 498 | 79 | 0 | 1,112 | 2.8% (14/500) [1.7–4.6] | 1.000 | 0.075 | 2.5% (2/79) [0.7–8.8] | 1.000 | 1.000 | 3.2% (8/249) | 0.0065% |
| Tulu-3-8B | 3.1 RLVR (GRPO) | 500 | 493 | 87 | 0 | 1,104 | 3.4% (17/500) [2.1–5.4] | 0.716 | 0.212 | 2.3% (2/87) [0.6–8.0] | 1.000 | 1.000 | 0.5% (1/207) | 0.0060% |

Omnibus chi-square across each family's stages (all rollouts / correct rollouts / within first 1,024 tokens): OLMo-3 Think < 1e-10 / < 1e-10 / < 1e-10; OLMo-3 RL-Zero-Math < 1e-10 / < 1e-10 / < 1e-10; OLMo-3 Instruct < 1e-10 / 2.6e-07 / 0.003; Tulu-3-8B 0.159 / 0.911 / 9.6e-07

Reading: within each OLMo-3 track the DPO checkpoint flags more rollouts
than the SFT before it (4.7× Think, 6.2× Instruct), and the on-policy RL
final flags fewer than DPO (5.4× and 3.4×), ending level with SFT (Think,
p = 0.48) or slightly above it (Instruct, p = 0.053). RL-Zero-Math, which
starts from the base model with no DPO, rises 4× from step 300 to step
2000. On Tulu-3 no adjacent or first-vs-later stage differs on all
rollouts (p ≥ 0.075; omnibus p = 0.16), its correct buckets (20–87
rollouts, 0–2 events) do not support a correct-only comparison, and the
one significant Tulu omnibus (within 1,024 tokens, p = 1e-6) is the SFT
word-salad rollouts (13.2% vs ≤ 3.6%). For the OLMo-3 tracks the
correct-only columns give the same orderings as the all-rollout columns.

### 2. DAPO vs AIME, same model, temperature 1 / top-p 1

| model | settings (DAPO / AIME) | rollouts (DAPO / AIME) | parsed | correct | truncated | mean tokens | ≥1 NC event, DAPO | ≥1 NC event, AIME | p | correct rollouts ≥1 NC, DAPO | AIME | p | within first 1,024 tokens, DAPO | AIME | p | per-token rate, DAPO / AIME |
|---|---|--:|--:|--:|--:|--:|---|---|--:|---|---|--:|---|---|--:|--:|
| OLMo-3 Think-DPO | T=1.0, top-p=1.0 / T=1.0, top-p=1.0 | 500 / 480 | 479 / 393 | 472 / 317 | 3 / 46 | 8,184 / 17,384 | 55.0% (275/500) [50.6–59.3] | 68.8% (330/480) [64.5–72.7] | 1.0e-05 | 54.2% (256/472) | 62.8% (199/317) | 0.019 | 22.1% (110/497) | 24.7% (118/478) | 0.364 | 0.0186% / 0.0177% |
| OLMo-3 Think RL final | T=1.0, top-p=1.0 / T=1.0, top-p=1.0 | 500 / 480 | 492 / 416 | 486 / 327 | 7 / 54 | 9,299 / 19,127 | 10.2% (51/500) [7.8–13.2] | 22.9% (110/480) [19.4–26.9] | 7.1e-08 | 9.3% (45/486) | 15.3% (50/327) | 0.010 | 1.2% (6/500) | 1.5% (7/480) | 0.785 | 0.0034% / 0.0039% |
| OLMo-3 RL-Zero-Math final | T=1.0, top-p=1.0 / T=1.0, top-p=1.0 | 500 / 480 | 498 / 476 | 446 / 166 | 1 / 3 | 6,265 / 11,229 | 58.4% (292/500) [54.0–62.6] | 67.3% (323/480) [63.0–71.3] | 0.004 | 56.5% (252/446) | 56.0% (93/166) | 0.927 | 44.3% (221/499) | 43.1% (207/480) | 0.747 | 0.0238% / 0.0273% |

Reading: every model flags more AIME rollouts than DAPO rollouts overall
(AIME rollouts are 1.8–2.1× longer), and none differs within the first
1,024 tokens (p ≥ 0.36). The ordering DPO ≈ Zero ≫ RL final holds on both
sets.

### 3. Temperature 1 / top-p 1 vs each checkpoint's recommended settings, DAPO

| model | settings (untruncated / recommended) | rollouts (untruncated / recommended) | parsed | correct | truncated | mean tokens | ≥1 NC event, untruncated | ≥1 NC event, recommended | p | correct rollouts ≥1 NC, untruncated | recommended | p | within first 1,024 tokens, untruncated | recommended | p | per-token rate, untruncated / recommended |
|---|---|--:|--:|--:|--:|--:|---|---|--:|---|---|--:|---|---|--:|--:|
| OLMo-3 Think-DPO | T=1.0, top-p=1.0 / T=0.6, top-p=0.95 | 500 / 500 | 479 / 490 | 472 / 483 | 3 / 2 | 8,184 / 7,775 | 55.0% (275/500) [50.6–59.3] | 15.2% (76/500) [12.3–18.6] | < 1e-10 | 54.2% (256/472) | 15.1% (73/483) | < 1e-10 | 22.1% (110/497) | 7.0% (35/498) | < 1e-10 | 0.0186% / 0.0032% |
| OLMo-3 Think RL final | T=1.0, top-p=1.0 / T=0.6, top-p=0.95 | 500 / 500 | 492 / 494 | 486 / 491 | 7 / 6 | 9,299 / 8,956 | 10.2% (51/500) [7.8–13.2] | 2.0% (10/500) [1.1–3.6] | 4.0e-08 | 9.3% (45/486) | 2.0% (10/491) | 5.7e-07 | 1.2% (6/500) | 0.2% (1/500) | 0.124 | 0.0034% / 0.0008% |
| OLMo-3 Instruct-SFT | T=1.0, top-p=1.0 / T=0.6, top-p=0.95 | 500 / 500 | 415 / 447 | 133 / 174 | 0 / 18 | 538 / 1,826 | 3.4% (17/500) [2.1–5.4] | 0.4% (2/500) [0.1–1.4] | 6.5e-04 | 3.8% (5/133) | 0.6% (1/174) | 0.089 | 3.2% (1/31) | 0.0% (0/49) | 0.387 | 0.0267% / 0.0004% |
| OLMo-3 Instruct-DPO | T=1.0, top-p=1.0 / T=0.6, top-p=0.95 | 500 / 500 | 496 / 484 | 368 / 389 | 2 / 15 | 3,403 / 3,512 | 21.0% (105/500) [17.7–24.8] | 5.0% (25/500) [3.4–7.3] | < 1e-10 | 15.5% (57/368) | 3.1% (12/389) | 1.6e-09 | 8.0% (28/348) | 1.5% (5/328) | 5.6e-05 | 0.0186% / 0.0028% |
| OLMo-3 Instruct RL final | T=1.0, top-p=1.0 / T=0.6, top-p=0.95 | 500 / 500 | 500 / 500 | 460 / 468 | 0 / 0 | 2,545 / 2,507 | 6.2% (31/500) [4.4–8.7] | 1.0% (5/500) [0.4–2.3] | 9.2e-06 | 5.4% (25/460) | 0.6% (3/468) | 1.0e-05 | 2.6% (10/388) | 0.5% (2/387) | 0.037 | 0.0047% / 0.0008% |
| Tulu-3-SFT | T=1.0, top-p=1.0 / T=0.6, top-p=0.9 | 500 / 500 | 478 / 482 | 20 / 40 | 0 / 16 | 812 / 1,826 | 5.2% (26/500) [3.6–7.5] | 0.6% (3/500) [0.2–1.7] | 1.2e-05 | 0.0% (0/20) | 0.0% (0/40) | 1.000 | 13.2% (14/106) | 1.6% (2/127) | 4.8e-04 | 0.0342% / 0.0026% |
| Tulu-3-DPO | T=1.0, top-p=1.0 / T=0.6, top-p=0.9 | 500 / 500 | 498 / 496 | 54 / 64 | 0 / 2 | 943 / 1,064 | 3.0% (15/500) [1.8–4.9] | 0.8% (4/500) [0.3–2.0] | 0.018 | 1.9% (1/54) | 1.6% (1/64) | 1.000 | 3.6% (6/168) | 1.3% (2/149) | 0.290 | 0.0085% / 0.0034% |
| Tulu-3 RLVR | T=1.0, top-p=1.0 / T=0.6, top-p=0.9 | 500 / 500 | 498 / 493 | 79 / 79 | 0 / 3 | 1,112 / 1,244 | 2.8% (14/500) [1.7–4.6] | 1.4% (7/500) [0.7–2.9] | 0.185 | 2.5% (2/79) | 0.0% (0/79) | 0.497 | 3.2% (8/249) | 0.9% (2/219) | 0.113 | 0.0065% / 0.0042% |
| Tulu-3.1 | T=1.0, top-p=1.0 / T=0.6, top-p=0.9 | 500 / 500 | 493 / 475 | 87 / 93 | 0 / 20 | 1,104 / 2,430 | 3.4% (17/500) [2.1–5.4] | 2.2% (11/500) [1.2–3.9] | 0.338 | 2.3% (2/87) | 0.0% (0/93) | 0.232 | 0.5% (1/207) | 2.2% (5/230) | 0.219 | 0.0060% / 0.0021% |

Reading: the recommended settings lower the flagged fraction in every
cell, by 3.6–8.5× in the OLMo-3 cells (all p ≤ 7e-4) and by 1.5–8.7× in
Tulu (significant for SFT and DPO only), without changing the ordering of
stages (Think-DPO 15.2% vs RL final 2.0%; Instruct-DPO 5.0% vs RL final
1.0%). At the recommended settings Instruct-SFT, Instruct-DPO, Tulu-3-SFT
and Tulu-3.1 run to the 32k cap in 3–4% of rollouts, which lengthens their
mean tokens; the other short-answer cells stay under 1%. RL-Zero-Math has
no recommended setting (no `generation_config`), and Think-SFT was only
run at temperature 1, so those cells are absent.

*Notes from Claude:* the tables answer "how often does a shipped
checkpoint emit a non-canonical token in a math rollout" (table 3,
recommended column) and "which training stage moves it" (table 1), not
whether the temperature reduction alone explains table 3, since the
recommended settings also truncate the tail (top-p) and were not varied
one factor at a time. The correct-only columns are informative for the
OLMo-3 tracks (≥ 133 correct rollouts per cell) and not for Tulu. The AIME
cells are 8 samples per problem, and the flags cluster by problem
(intra-problem correlation about 0.1–0.2), so the AIME intervals and the
DAPO-vs-AIME p-values in table 2 are somewhat too narrow; the directions
are unaffected. The principled single-model alternative to these pairwise
tests is a logistic regression of the flag on stage plus log length (with
problem clusters for AIME), which adjusts for length instead of
conditioning on a window; not done here.

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
(`<arm>.parquet` rollouts with IDs and logprobs in the current record
schema, converted one-to-one from the `<arm>.jsonl.gz` files the pilot
wrote before the Parquet switch, which are kept alongside;
`<arm>.meta.json`; `dapo_pilot50.jsonl` as used; `metrics/summary.{json,md}`,
`metrics/analysis.jsonl`, `metrics/examples.jsonl`). The metrics in the
dataset were recomputed locally with the post-review `metrics.py` and
overwrite the on-box copy; the raw records are unchanged. Job: SkyPilot
job 1 on `noncanon-pilot`, SUCCEEDED, 41m 51s wall clock including setup,
cluster torn down afterwards.

**Throughput** (A100-80GB, 50 concurrent requests submitted, vLLM KV
budget 119,312 tokens, max concurrency 6.65 at 34,816 tokens):

| arm | rollouts | output tokens | wall time | output tokens/s |
|---|--:|--:|--:|--:|
| recommended (0.6 / 0.95) | 50 | 468,362 | 1,178.5 s | 397.4 |
| untruncated (1.0 / 1.0) | 50 | 465,346 | 1,103.3 s | 421.8 |

**Rollout length and outcome** (measured tokens exclude the trailing stop
token; excluded as incomplete UTF-8: 0 in both arms; excluded as the cut
last word of a truncated rollout: 3 tokens in the recommended arm):
(Unit counts throughout this file were recomputed on 2026-09-04 after the
cut-last-word exclusion was capped at 8 tokens — see the Think
recommended-settings cell — and only unit and excluded-token counts
moved; no rate, CI or p-value changed at the reported precision.)

| arm | mean | median | p90 | max | finish | think closed | verified correct |
|---|--:|--:|--:|--:|---|---|---|
| recommended | 9,366 | 8,526 | 12,666 | 32,765 | 49 stop, 1 length | 49 / 50 | 49 / 49 finished |
| untruncated | 9,306 | 8,612 | 14,449 | 30,245 | 50 stop | 50 / 50 | 49 / 50 |

**Non-canonical tokens.** Per-token rate = canonical tokens inside
non-canonical spans / canonical tokens; the emitted-token count inside
spans is given alongside.

| arm | canonical tokens | non-canonical (canonical) | rate | emitted in spans | spans | rollouts with ≥1 | think | answer |
|---|--:|--:|--:|--:|--:|--:|---|---|
| recommended | 468,310 | 0 | 0.0000% | 0 | 0 | 0 / 50 | 0 / 439,884 | 0 / 28,426 |
| untruncated | 465,293 | 17 | 0.0037% | 20 | 10 | 6 / 50 | 17 / 435,760 (0.0039%) | 0 / 29,536 |

Untruncated arm, further slices (17 canonical / 20 emitted tokens in 10
spans; small counts). Class and position slices count emitted tokens;
outcome and length slices count canonical tokens.

- by token class of the emitted token: whitespace 1 / 21,895; digit
  0 / 77,219; word 14 / 220,526; mixed 2 / 9,191; symbol 3 / 136,465.
- by outcome: correct 15 / 435,048 (49 rollouts); incorrect 2 / 30,245
  (1 rollout).
- by rollout-length quartile (rank-based): q1 3 / 65,502; q2 0 / 88,918;
  q3 0 / 126,421; q4 14 / 184,452.
- by relative position decile (0 to 9), emitted tokens: 0, 0, 4, 0, 2,
  2, 0, 4, 0, 8; decile 9 rate 0.017%.
- sequence-level flag (≥1 event within the first L tokens, over
  rollouts that reached L): L=256 0 / 50; L=1024 0 / 50; L=4096 3 / 48.
- top-10 entropy (nats, renormalized over a fixed k=10): mean over all
  positions 0.343; mean at the 20 non-canonical positions 1.010.

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

Think vs answer: under a common rate of 17 / 465,293 the expected number
of events in the 29,536 answer tokens is 1.1, and the probability of
observing none is 0.33, so the pilot does not distinguish the two
regions.

**Prediction status.** The pilot is a single checkpoint, so none of the
before/after predictions are testable from it. Predictions 10–12 in the
plan were added after reading these results. Prediction 1's bound
(<0.1% per token on in-distribution text for a text-distilled-then-RL'd
model) holds for this checkpoint at both settings. Prediction 2's digit
bound holds (0 / 77,219 and 0 / 84,853 digit tokens).

## Full run, in progress: Think RL final vs Think-DPO vs RL-Zero-Math (2026-09-03)

Untruncated arm only (temperature 1.0, top-p 1.0). Prompt sets:
`prompts/dapo_sample500.jsonl` (500 held-out DAPO problems, seed 0, from the
3,254-problem set) and `prompts/aime_2024_2025.jsonl` at 8 samples per
problem (480 rollouts). One B200 ($6.79/h) per checkpoint; `max_tokens`
32,768; otherwise as the pilot. Launch:

```
sky launch -c <cluster> skypilot/run.yaml --gpus B200:1 -i 20 --down -y -d --env HF_TOKEN \
    --env MODEL=<checkpoint> --env RUN_NAME=<run>
```

Incidents: the first Think B200 host had a dead network (0 bytes received
over 10 s while `uv sync` sat at 2% CPU for 20 min) and was replaced. The
DPO and Zero jobs finished DAPO generation and then failed in the on-box
metrics step on rollouts containing invalid UTF-8 mid-run (fixed in
`metrics.py` by measuring only decodable segments; the records were copied
off the boxes first and the metrics below were computed locally with the
fixed code). Their AIME halves were relaunched on the same clusters.
Throughput on the B200s ran 1,500–2,000 output tokens/s once contexts
lengthened (about 3.5–4× the A100 at the same phase), with a 300k-token KV
budget (2.5× the A100).

### DAPO 500, DPO and Zero (Think 500 pending)

| checkpoint | rollouts | finish | mean / median / max tokens | accuracy (finished, parsed) | excluded tokens (UTF-8 / cut word) |
|---|--:|---|---|--:|---|
| `Olmo-3-7B-Think-DPO` | 500 | 497 stop, 3 length; think closed 490 | 8,184 / 6,710 / 32,764 | 98.5% (18 unparsed) | 207 / 6 |
| `Olmo-3-7B-RL-Zero-Math` | 500 | 499 stop, 1 length; no think block | 6,265 / 5,360 / 32,767 | 89.6% (1 unparsed) | 5 / 1 |

Headline rate = (canonical tokens in spans + standalone byte-fragment
events) / (canonical tokens + standalone fragment events), per the
counting rule recorded in the plan; the segmentation-only rate excludes
fragments.

| checkpoint | units | non-canonical | **rate** | segmentation only | spans | fragments (standalone) | rollouts with ≥1 event | span shapes (whitespace / alphabetic / symbolic) |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| Think-DPO | 4,092,078 | 762 | **0.0186%** | 655 / 4,091,971 = 0.0160% | 403 | 107 (107) | 275 / 500 | 232 / 69 / 102 |
| RL-Zero-Math | 3,132,357 | 745 | **0.0238%** | 742 / 3,132,354 = 0.0237% | 543 | 3 (3) | 292 / 500 | 4 / 176 / 363 |
| (pilot, Think RL final, 50 prompts, old sample) | 465,293 | 17 | 0.0037% | same | 10 | 0 | 6 / 50 | 1 / 5 / 4 |

Think-DPO: think 735 / 3,750,384 (0.0196%), answer 27 / 341,751 (0.0079%);
segmentation only, think 631 (0.0168%), answer 24 (0.0070%).
Entropy (top-10, nats): all positions 0.286, at non-canonical positions
0.803. By outcome: correct 603 / 3.68M (472 rollouts), incorrect 8 / 118k
(7), truncated 18 / 98k (3), unparsed 26 / 194k (18). By length quartile
(q1 shortest): 0.025%, 0.022%, 0.016%, 0.012%. Token class of the emitted
tokens in spans: whitespace 234, digit 7, word 281, mixed 78, symbol 219.

RL-Zero-Math: no think block, so all tokens are "answer". Entropy 0.365 all
positions, 0.555 at non-canonical positions. By outcome: correct 550 / 2.51M
(446 rollouts), incorrect 190 / 572k (52), truncated 0 / 33k (1). By length
quartile: 0.029%, 0.017%, 0.028%, 0.023%. Token class of emitted tokens in
spans: whitespace 4, digit 135, word 407, mixed 18, symbol 527.

**Byte-fragment events** (the model starts a multi-byte character as
separate byte tokens and never completes it, e.g. `\xe2\x88` — the first two
bytes of `−`/`√` — followed by ` geological`; the bytes have no text form so
the canonical comparison is undefined): Think-DPO 107 events / 108 tokens
in 59 rollouts (53 of those rollouts finished and verified correct), all
standalone (none adjacent to a span); RL-Zero-Math 3 events / 3 tokens in 3
rollouts. An earlier count of 207 tokens for DPO included the valid token
following each abandoned prefix; fixed the same day. Transcripts
(`metrics/transcripts.jsonl`) render fragments as `⟨bytes e2 88⟩`.

**Digit splits.** All-digit spans: RL-Zero-Math 68, of which 52 are in one
rollout (45 of them `'3'`,`'5'` → `'35'` in `log_35`-style contexts) and 8
in another; DPO 7 events. Digit tokens in spans over digit tokens emitted:
RL-Zero-Math 135 / 527,858 (0.026%), Think-DPO 7 / 648,651 (0.001%). One
Zero example runs the other way: after `log₁₀` the model emitted `'10'` as
one token where the canonical encoding is `'1'`,`'0'` (the subscript digits
count toward the pre-tokenizer's three-digit grouping).

**Most common span patterns** (emitted → canonical; count):

Think-DPO (403 spans): `' '`,`'.'` → `' .'` ×16; `' '`,`'�'` (space then a
partial multi-byte character) → `' �'` ×14; `' '`,`'**'` → `' **'` ×7;
`' '`,`'.\n\n'` ×4; `'”'`,`'的'` → `'”的'` ×3; `'当'`,`'前'` → `'当前'`
(Chinese inside the English CoT). Most spans are a bare space token
followed by punctuation or a word piece. Densest rollouts: 8, 6, 5, 5, 5
spans.

RL-Zero-Math (543 spans): `' $'`,`'($'` → `' $($'` ×213; `'3'`,`'5'` →
`'35'` ×45 (a digit split); `' AE'`,`'FB'` → `' A'`,`'EF'`,`'B'` ×12;
dropped-space word joins with a variable name: `' than'`,`'k'` → `' thank'`
×10, `' when'`,`'g'` → `' wh'`,`'eng'` ×10, `'Thus'`,`'k'` ×9, `' for'`,`'k'`
×7, `' of'`,`'f'` ×6. Heavily clustered: one rollout carries 52 spans and
another 30.

**DPO vs Zero, rollouts as the unit** (events cluster, so token-level
tests overstate certainty). Headline rule (fragments counted):
rollout-bootstrap 95% CIs DPO 0.0164–0.0210%, Zero 0.0189–0.0298%;
permutation test on the pooled rate with rollouts permuted: Zero − DPO =
0.0052 pp, two-sided p = 0.060 (`noncanon.compare`, 20,000 permutations;
earlier inline runs of the same test with 5,000 permutations and other RNG
streams gave 0.053–0.057). Segmentation only: CIs DPO 0.0141–0.0180%,
Zero 0.0188–0.0295%; Zero − DPO = 0.0077 pp, p = 0.0023.
Rollouts with ≥1 span: 50.2% vs 58.0%, z = 2.5. Spans per rollout: DPO mean
0.81, variance 1.15, max 8; Zero mean 1.09, variance 8.63, max 52. Median
per-rollout rate: DPO 0.0052%, Zero 0.0138%. Dropping Zero's two densest
rollouts: Zero 0.0206%, difference 0.0046 pp, p = 0.011 (inline run,
5,000 permutations; not part of the specification). (Token-level
Poisson z ignoring clustering: 7.4.)

Brendan's note (2026-09-03) on the two conventions: byte fragments and
non-canonical-but-valid tokens are probably not the same thing (invalid
UTF-8 reads as "the model isn't smart enough", while other non-canonical
tokens are frequently reasonable); the headline rule's p = 0.057 is not
ideal but more defensible. The AIME cells will add evidence, with the
caveat that the same data must not be compared in different ways until
something comes out significant.

### DAPO 500, Think RL final (completes the DAPO comparison)

`Olmo-3-7B-Think` @ `main`, same prompts and settings. 500 rollouts: 493
stop, 7 length; think closed 495 / 500; accuracy (finished, parsed) 98.8%;
excluded 5 tokens incomplete UTF-8, 24 cut last word. Length: mean 9,299,
median 8,077, p90 15,702, max 32,767.

| checkpoint | units | non-canonical | **rate** | segmentation only | spans | fragments | rollouts with ≥1 event | span shapes (whitespace / alphabetic / symbolic) |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| Think RL final | 4,649,364 | 160 | **0.0034%** | 155 / 4,649,359 = 0.0033% | 108 | 5 | 51 / 500 | 1 / 25 / 82 |

Think 153 / 4,364,900 (0.0035%), answer 7 / 284,517 (0.0025%). Entropy
(top-10, nats): all positions 0.354, at non-canonical positions 0.472.
The pilot's 50-prompt Think number on the older sample was 0.0037%.

**Tests per the analysis specification in the plan** (DAPO vs DAPO;
per-token = two-proportion z on pooled counts; per-rollout = permutation
test with rollouts as units, 20,000 permutations, so p < 0.00005 reads as
0/20,000; z is reported unsigned here, its sign follows the difference; rollout-bootstrap 95% CIs):

| comparison | headline rule: difference | per-token z, p | per-rollout p | segmentation only: difference | per-token z, p | per-rollout p |
|---|--:|---|--:|--:|---|--:|
| Think RL final − Think-DPO | −0.0152 pp | 21.8, 2e-105 | <0.00005 | −0.0127 pp | 19.4, 5e-84 | <0.00005 |
| Think RL final − RL-Zero-Math | −0.0203 pp | 25.8, 8e-147 | <0.00005 | −0.0204 pp | 25.9, 3e-148 | <0.00005 |
| RL-Zero-Math − Think-DPO | +0.0052 pp | 4.8, 1.9e-6 | 0.060 | +0.0077 pp | 7.4, 1.9e-13 | 0.0023 |

Rollout-bootstrap 95% CIs, headline rule: Think RL final 0.0019–0.0059%,
Think-DPO 0.0164–0.0210%, RL-Zero-Math 0.0189–0.0298%.

**Prediction status (DAPO cell).** Prediction 3 (on-policy RL raises the
rate; Think RL final > Think-DPO): **refuted in this cell** — the RL final
checkpoint is about 5× below its DPO starting point under either
convention, at every reported p. The exclusively-on-policy model
(RL-Zero-Math) is above both Think checkpoints per-token, and above DPO
per-rollout at p = 0.060 (headline) / 0.0023 (segmentation only).
Prediction 4 (SFT ≈ DPO) is untested (no SFT cell yet). Prediction 5
(higher inside the CoT): Think RL final 0.0035% think vs 0.0025% answer,
DPO 0.0196% vs 0.0079% (descriptive). Prediction 11 (word joins grow with
RL): the alphabetic-span count is 25 for RL final vs 69 for DPO
(descriptive).

### AIME 2024/2025 (60 problems × 8 samples), RL-Zero-Math

480 rollouts: 477 stop, 3 length; accuracy 34.9% (1 unparsed); excluded 1
incomplete UTF-8, 18 cut last word. Length: mean 11,229, median 11,560,
p90 16,924, max 32,766. Headline rate **0.0273%** (1,470 of 5,389,306
units; 939 spans, 1 fragment; 323 / 480 rollouts with ≥1 event; shapes
whitespace 30 / alphabetic 431 / symbolic 478); segmentation only 0.0273%.
Entropy 0.397 all positions, 0.687 at non-canonical positions. DAPO for
the same checkpoint: 0.0238%. Within-model AIME − DAPO (per the spec):
headline +0.0035 pp, per-token z = 3.0 (p = 0.002), per-rollout
permutation p = 0.39; segmentation only +0.0036 pp, z = 3.1 (p = 0.002),
per-rollout p = 0.38. The Think and DPO AIME cells are still running.

### AIME 2024/2025, Think-DPO

480 rollouts: 434 stop, 46 length; think closed 433 / 480; accuracy
(finished, parsed) 80.7%; excluded 165 tokens incomplete UTF-8, 144 cut
last word. Length: mean 17,384, median 14,779, p90 32,539, max 32,767.
Headline rate **0.0177%** (1,474 of 8,344,397 units; 757 spans, 157
fragments of which 156 standalone; 330 / 480 rollouts with ≥1 event; shapes
whitespace 301 / alphabetic 222 / symbolic 234); segmentation only 0.0158%
(1,318 / 8,344,219). Think 1,422 / 7,961,992 (0.0179%), answer 52 / 382,429
(0.0136%). Entropy 0.393 all positions, 0.967 at non-canonical positions.
Rollout-bootstrap 95% CI (headline) 0.0158–0.0198%.

Within-model AIME − DAPO for DPO: headline −0.0010 pp, per-token z = 1.2
(p = 0.24), per-rollout p = 0.55; segmentation only −0.0002 pp, z = 0.3
(p = 0.78), per-rollout p = 0.88.

AIME, RL-Zero-Math − Think-DPO: headline +0.0096 pp, per-token z = 11.9
(p = 2e-32), per-rollout p < 0.00005; segmentation only +0.0115 pp,
z = 14.6 (p = 5e-48), per-rollout p < 0.00005. (The AIME Think cell is still running.)

### Next-token distribution sharpness, DAPO 500 (from the stored top-10 logprobs)

Computed over every generated position; p_top1 is the raw probability of
the model's most likely token, entropy is over the top-10 renormalized.

| checkpoint | positions | mean p_top1 | frac p_top1 < 0.5 | frac p_top1 < 0.9 | mean mass outside top-1 | mean entropy | entropy p90 | frac entropy > 1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Think RL final | 4,649,943 | 0.858 | 8.8% | 36.9% | 0.142 | 0.379 | 1.16 | 13.9% |
| Think-DPO | 4,092,746 | 0.874 | 7.9% | 32.6% | 0.126 | 0.338 | 1.10 | 12.2% |
| RL-Zero-Math | 3,133,206 | 0.864 | 8.8% | 34.8% | 0.136 | 0.368 | 1.18 | 14.2% |

(Think RL final has the highest average entropy and tail mass of the three
and the lowest non-canonical rate; DPO the lowest entropy and the middle
rate.)

**Tail depth and where spans start.** Mass beyond the top-10 (the deep
tail; raw logprobs are a full softmax so the top-10 sum is exact) and the
rank of the emitted token, overall and at the first token of each
non-canonical span (byte fragments excluded):

| checkpoint | mass beyond top-10, all positions | at span-first positions | positions sampled beyond top-10 | span-first token at rank 1 / 2–3 / 4–10 / >10 | spans per 1M samples at rank 1 / 2–3 / 4–10 / >10 |
|---|--:|--:|--:|---|---|
| Think RL final | 0.0024 | 0.0109 | 0.22% | 79.6 / 12.0 / 5.6 / 2.8% | 21 / 25 / 58 / 288 |
| Think-DPO | 0.0050 | 0.0347 | 0.49% | 76.4 / 10.2 / 4.2 / 9.2% | 86 / 103 / 208 / 1,848 |
| RL-Zero-Math | 0.0031 | 0.0186 | 0.31% | 78.6 / 15.3 / 3.7 / 2.4% | 157 / 249 / 264 / 1,358 |

Share of all sampled tokens at rank 1 is 86–88% for all three. Under
temperature 1 sampling, about three quarters of spans begin with the
model's argmax token in every checkpoint.

**Refinement: spans that begin with a bare space token.** In this
tokenizer a standalone space token is canonical only before a digit
(digits are pre-tokenized separately), so a bare space followed by a
non-digit is a span whose first token is innocuous and whose second token
is the deviation. DPO has 231 such spans (of 403); the token after the
space is at rank >10 in 81% of them (rank 1 in 3%); the most common
followers are a partial multi-byte character (18), `.` (16), `**` (7). Its
other 172 spans start at rank 1 in 54% and rank >10 in 22%. Think RL
final has 1 bare-space span and 107 others (80% rank 1, 3% rank >10);
RL-Zero-Math has 2 and 541 others (79% rank 1, 2% rank >10).

**Random examples** (context … emitted → canonical; p = probability of
the first emitted token; for tail cases, the model's top choice):

Think RL final, argmax-start: `The image link is to a cdn` `' art'`,`'of'` → `' ar'`,`'to'`,`'f'` (p 0.29); `factor the left-hand side` `'.'`,`'Rew'` → `'.R'`,`'ew'` (1.00); `r)(-s) = - (` `'t'`,`'rs'` → `'trs'` (0.68); `But W` `'_g'`,`'b'` → `'_gb'` (1.00); `based on案` `'例'`,`'如'` → `'例如'` (0.98). Tail-start: `or else the set` `' is'`,`'nt'` → `' isnt'` (rank 2, p 0.29; top `' isn'` 0.69); `which` `' simpl'`,`'ates'` → `' sim'`,`'plates'` (rank 2; top `' gives'` 0.73); `starts with 2.` `'abc'`,`'def'` → `'abcdef'` (rank 5, p 0.02; top `' followed'` 0.47).

Think-DPO, argmax-start (all bare-space): `15m + b =` `' '`,`'lic'` → `' lic'` (p 1.00); `is at most` `' '`,`'Counter'` → `' Counter'` (1.00); `legs of length 3 and` `' '`,`'.'` → `' .'` (1.00); `6k+2,` `' '`,`'-P'`,`'Finding'` → `' -'`,`'PF'`,`'inding'` (1.00). Tail-start: `counter BAL` `'ANC'`,`'Ed'` → `'ANCE'`,`'d'` (rank 4, p 0.001; top `'ANCED'` 0.98); `t_n = 1 +` `' Chronic'`,`'hidden'` → `' Chron'`,`'ich'`,`'idden'` (rank 11; top `'2'` 0.97); `So that cross` `'-bot'`,`'ton'` → `'-b'`,`'otton'` (rank 11; top `'-check'` 0.91).

RL-Zero-Math, argmax-start: `tag, in` `' $'`,`'($'` → `' $($'` (p 1.00, ×213); `which is original` `' term'`,`'m'` → `' ter'`,`'mm'` (0.99); `answer ask` `' for'`,`'k'` → `' fork'` (0.37); `We need to` `' compute'`,`'x'` → `' comput'`,`'ex'` (0.50). Tail-start: `Expand the second term:` `'(b'`,`'ig'` → `'(big'` (rank 2; top `'Multiply'` 0.54); `Alternatively,` `' plot'`,`'ting'` → `' plotting'` (rank 3, p 0.17; top `' plotting'` 0.31); `Total P` `'+'`,`'E'` → `'+E'` (rank 2; top `'+E'` 0.69). For orientation against the
recommended-settings pilot (Think RL final, temperature 0.6 / top-p 0.95,
50 prompts): 21 rank-1 spans per million samples would predict about 9
spans in its 468k tokens if rank-1 events were independent of the
sampling regime; it observed 0.

### Launched 2026-09-03: Think RL final at recommended settings, DAPO 500 only

Per plan prediction 15.

```
sky launch -c nc-think-rec skypilot/run.yaml --gpus B200:1 --retry-until-up -i 20 --down -y -d --env HF_TOKEN \
    --env MODEL=allenai/Olmo-3-7B-Think --env RUN_NAME=think-main-recommended --env ARMS=recommended \
    --env PROMPTS="prompts/dapo_sample500.jsonl"
```

### Launched 2026-09-03: Think-DPO at recommended settings, DAPO 500 only

Per plan prediction 16. Same command as the Think recommended run with
`MODEL=allenai/Olmo-3-7B-Think-DPO`, `RUN_NAME=think-dpo-recommended`.

### Launched 2026-09-03: Think-SFT and RL-Zero-Math step_300, DAPO 500 only

Per plan predictions 13 and 14. Same task and settings; DAPO only.

```
sky launch -c nc-sft     skypilot/run.yaml --gpus B200:1 --retry-until-up -i 20 --down -y -d --env HF_TOKEN \
    --env MODEL=allenai/Olmo-3-7B-Think-SFT --env RUN_NAME=think-sft --env PROMPTS="prompts/dapo_sample500.jsonl"
sky launch -c nc-zero300 skypilot/run.yaml --gpus B200:1 --retry-until-up -i 20 --down -y -d --env HF_TOKEN \
    --env MODEL=allenai/Olmo-3-7B-RL-Zero-Math --env REVISION=step_300 --env RUN_NAME=rlzero-math-step300 \
    --env PROMPTS="prompts/dapo_sample500.jsonl"
```

RL-Zero-Math step branches on the Hub run `step_100` to `step_1900`, and
`main` is the commit "Upload checkpoint from step 2000" (same run; the
later commits on `main` touch only the README and config). So the
RL-Zero-Math cells above are step 2000, and step_300 is the same run at
15% of its length.
Compliance and accuracy of step_300 are to be checked before its rate is
compared.

### DAPO 500, RL-Zero-Math step_300 (prediction 14)

`allenai/Olmo-3-7B-RL-Zero-Math` @ `step_300` (the same run as the
step-2000 `main` checkpoint, 15% of the way through). 500 rollouts, all
stop; accuracy (finished, parsed) 87.4% (step 2000: 89.6%); no excluded
tokens. Length: mean 6,348, median 5,154, p90 11,991, max 24,304.

| checkpoint | finished (stop / length) | verified correct | mean / median tokens | units | non-canonical | rate | rollouts with ≥1 event | spans | span shapes (whitespace / alphabetic / symbolic) |
|---|---|--:|---|--:|--:|--:|--:|--:|---|
| step_300 | 500 / 0 | 87.4% (436 / 499 parsed) | 6,348 / 5,154 | 3,173,716 | 270 | 0.0085% | 73 / 500 | 166 | 9 / 106 / 51 |
| step 2000 (`main`) | 499 / 1 | 89.6% (446 / 498 parsed) | 6,265 / 5,360 | 3,132,357 | 745 | 0.0238% | 292 / 500 | 543 | 4 / 176 / 363 |

By outcome (rate over units): step_300 correct 0.0092% (436 rollouts),
incorrect 0.0064% (63), unparsed 0 (1); step 2000 correct 0.0219% (446),
incorrect 0.0332% (52), truncated 0 (1), unparsed 0.0145% (1). Both
checkpoints finish, attempt and mostly solve the problems at the same
rollout lengths, so the rate difference is not a compliance difference.

Rollout-bootstrap 95% CI for step_300: 0.0043–0.0157%. step 2000 − step
300 = +0.0153 pp; per-token z = 15.1 (p = 1e-51); per-rollout permutation
p < 0.00005 (identical under segmentation only: no fragments at step 300).
Entropy (top-10): 0.411 all positions, 0.883 at non-canonical positions.

Span patterns at step_300: word-plus-variable joins dominate (`' where'`,`'x'`
×10; `' per'`,`'k'` ×7; `' to'`,`'x'` ×6; `' above'`,`'x'` ×6; `' since'`,`'x'`
×5; `' to'`,`'u'` ×5); the `' $'`,`'($'` pattern that accounts for 213 of step
2000's 543 spans does not occur at step_300 (0); all-digit spans 3 (step
2000: 68). One rollout carries 50 spans; the next densest 8.

**Prediction status.** Prediction 14 (direction to be observed): within
the RL-Zero-Math run the rate rises from step 300 to step 2000, by about
3× per token and from 15% to 58% of rollouts, at every reported p. This is
the opposite direction from the Think track, where the RL final checkpoint
sits below its DPO starting point.

### DAPO 500, Think-SFT (prediction 13)

`allenai/Olmo-3-7B-Think-SFT` @ `main`. 500 rollouts, all stop; think
closed 500 / 500; accuracy (finished, parsed) 97.5%; excluded 3 tokens
incomplete UTF-8. Length: mean 8,231, median 7,014, p90 13,810, max
30,996. Headline rate **0.0030%** (123 of 4,115,568 units; 71 spans, 3
fragments; 59 / 500 rollouts with ≥1 event; shapes whitespace 6 /
alphabetic 28 / symbolic 37); segmentation only 0.0029%. Think 123 /
3,853,668 (0.0032%), answer 0 / 261,922. Entropy (top-10) 0.571 all
positions, 1.097 at non-canonical positions. Rollout-bootstrap 95% CI
0.0022–0.0038%.

The Think ladder on DAPO 500, headline rule:

| stage | rate | 95% CI | rollouts with ≥1 event |
|---|--:|---|--:|
| SFT | 0.0030% | 0.0022–0.0038% | 59 / 500 |
| DPO | 0.0186% | 0.0164–0.0210% | 275 / 500 |
| RL final | 0.0034% | 0.0019–0.0058% | 51 / 500 |

SFT − DPO: −0.0156 pp, per-token z = 21.6 (p = 4e-103), per-rollout
p < 0.00005 (segmentation only: −0.0131 pp, z = 19.3, p < 0.00005).
SFT vs RL final: +0.0005 pp, per-token z = 1.2 (p = 0.24), per-rollout
p = 0.86 (segmentation only: z = 1.1, p = 0.27; per-rollout 0.88).

**Prediction status.** Prediction 13 (SFT above DPO because it has less
training): **refuted**; SFT is 6× below DPO and indistinguishable from RL
final. Prediction 4 (SFT ≈ DPO): refuted in the same cell; the DPO stage
is where the rate rises within this ladder, and the RL stage is where it
falls back.

### AIME 2024/2025, Think RL final (completes the AIME comparison)

480 rollouts: 426 stop, 54 length; think closed 428 / 480; accuracy
(finished, parsed) 78.6%; excluded 8 tokens incomplete UTF-8, 132 cut last
word. Length: mean 19,127, median 18,550, p90 32,761, max 32,767. Headline
rate **0.0039%** (355 of 9,180,709 units; 201 spans, 8 fragments; 110 / 480
rollouts with ≥1 event; shapes whitespace 1 / alphabetic 103 / symbolic
97); segmentation only 0.0038%. Think 353 / 8,916,561 (0.0040%), answer
2 / 264,186 (0.0008%). Entropy 0.423 all positions, 0.910 at non-canonical
positions. Rollout-bootstrap 95% CI 0.0029–0.0049%.

AIME, all three checkpoints, headline rule:

| checkpoint | rate | 95% CI | rollouts with ≥1 event | accuracy |
|---|--:|---|--:|--:|
| Think RL final | 0.0039% | 0.0029–0.0049% | 110 / 480 | 78.6% |
| Think-DPO | 0.0177% | 0.0157–0.0197% | 330 / 480 | 80.7% |
| RL-Zero-Math | 0.0273% | 0.0222–0.0330% | 323 / 480 | 34.9% |

Tests on AIME: RL final − DPO = −0.0138 pp, per-token z = 28.2, per-rollout
p < 0.00005 (segmentation only −0.0120 pp, z = 25.8, p < 0.00005). RL final −
Zero = −0.0234 pp, z = 38.5, per-rollout p < 0.00005 (segmentation only
−0.0235 pp, z = 38.8). Zero − DPO on AIME was reported above.

Within-model AIME − DAPO for Think RL final: +0.0004 pp, per-token z = 1.2
(p = 0.22), per-rollout p = 0.69 (segmentation only: z = 1.3, p = 0.19;
per-rollout 0.68).

**Prediction status (AIME cell).** Prediction 3: refuted on AIME as on
DAPO (RL final about 4.5× below DPO). Prediction 10 (AIME slightly higher
than DAPO): the direction is as predicted for all three checkpoints
(Think +0.0004 pp, DPO −0.0010 pp is the exception, Zero +0.0035 pp) and
none of the within-model differences is distinguishable at the rollout
level. Prediction 5 (higher inside the CoT), descriptive: Think RL final
think 0.0040% vs answer 0.0008% on AIME, 0.0035% vs 0.0025% on DAPO;
DPO 0.0179% vs 0.0136% on AIME.

### DAPO 500, Think RL final at the recommended settings (prediction 15)

`allenai/Olmo-3-7B-Think` @ `main`, temperature 0.6 / top-p 0.95 (the
checkpoint's `generation_config.json`), otherwise as the untruncated cell;
B200, 1,495 output tokens/s. 500 rollouts: 494 stop, 6 length; think
closed 495 / 500; accuracy (finished, parsed) 99.4%; excluded 0 tokens
incomplete UTF-8, 19 cut last word. Length: mean 8,897, median 7,552, p90
15,203, max 32,767.

| Think RL final, DAPO 500 | units | non-canonical | rate | 95% CI | spans | rollouts with ≥1 event | shapes (ws / alpha / sym) |
|---|--:|--:|--:|---|--:|--:|---|
| untruncated (1.0 / 1.0) | 4,649,364 | 160 | 0.0034% | 0.0019–0.0059% | 108 | 51 / 500 | 1 / 25 / 82 |
| recommended (0.6 / 0.95) | 4,477,830 | 34 | **0.0008%** | 0.0003–0.0013% | 17 | 10 / 500 | 0 / 11 / 6 |

Recommended − untruncated: −0.0027 pp, per-token z = 8.8 (p = 2e-18),
per-rollout permutation p = 0.0001 (3 / 20,000); segmentation only
identical (no fragments in the recommended arm). Think 30 / 4,162,998
(0.0007%), answer 4 / 285,300 (0.0014%). Entropy 0.304 all positions,
0.707 at non-canonical positions. Of the 17 spans, 14 begin at the argmax
token, 2 at rank 2–3, 1 at rank 4–10, none beyond the top-10; the sampler
reached beyond the top-10 at 0.0% of positions (93.7% of samples were the
argmax).

**Prediction status.** Prediction 15 (does the rate hit exactly 0 at the
recommended settings?): **no** — 34 units in 4.48M, 10 of 500 rollouts,
about 4× below the untruncated cell. The pilot's 0 in 468k was consistent
with this rate (expected ≈ 3.6 events). Prediction 16 (DPO at the
recommended settings; possible reversal) is pending that cell.

**Cut-last-word fix (2026-09-04).** This cell exposed a flaw in the
truncated-rollout exclusion: it dropped everything back to the last
whitespace-starting token, which for a truncated Chinese passage or a
symbol loop with no whitespace token was thousands of tokens (29,553 in
this cell's 6 truncated rollouts). The look-back is now capped at 8
tokens. All cells were recomputed; only unit and excluded-token counts
changed (this cell's units 4,448,296 → 4,477,830), no rate, CI or p at the
reported precision.

### DAPO 500, Think-DPO at the recommended settings (prediction 16)

`allenai/Olmo-3-7B-Think-DPO` @ `main`, temperature 0.6 / top-p 0.95. 500
rollouts: 498 stop, 2 length; think closed 493 / 500; accuracy (finished,
parsed) 98.6%; excluded 12 tokens incomplete UTF-8, 16 cut last word.
Length: mean 7,775, median 6,378, p90 13,811, max 32,760.

| DAPO 500, recommended settings | units | non-canonical | rate | 95% CI | spans | fragments | rollouts with ≥1 event | shapes (ws / alpha / sym) |
|---|--:|--:|--:|---|--:|--:|--:|---|
| Think-DPO | 3,887,498 | 126 | **0.0032%** | 0.0025–0.0040% | 98 | 10 | 76 / 500 | 73 / 9 / 16 |
| Think RL final | 4,477,830 | 34 | **0.0008%** | 0.0003–0.0013% | 17 | 0 | 10 / 500 | 0 / 11 / 6 |

DPO − RL final at the recommended settings: −0.0025 pp for RL, per-token
z = 8.2 (p = 3e-16), per-rollout permutation p < 0.00005 (segmentation
only: −0.0023 pp, z = 7.6, p < 0.00005). DPO recommended − DPO
untruncated: −0.0154 pp, z = 20.6, per-rollout p < 0.00005. Think 125 /
3,605,100 (0.0035%), answer 1 / 282,472. Entropy 0.256 all positions,
0.973 at non-canonical positions. Spans: 73 of 98 begin with a bare space;
the token after the space is now at rank 1 in 42%, rank 2–3 in 38%, beyond
the top-10 in 7% (untruncated: 3% / 9% / 81%). 92% of all spans begin at
the argmax token.

**Prediction status.** Prediction 16 (a reversal under the recommended
settings, Think RL final above DPO): **refuted** — DPO stays about 4× above
RL final. Truncation removed 83% of DPO's events and 79% of RL final's.

### Where spans start, all cells so far (from `noncanon.tail`)

Spans per million sampled tokens whose first token was the model's argmax
(rank 1) versus sampled from beyond the stored top-10, with the
distribution's sharpness. Byte fragments excluded. DPO's argmax figure is
dominated by its bare-space spans (space at p ≈ 1, deviation in the next
token); the figure excluding them is in brackets.

| cell (DAPO 500) | spans | argmax-start spans per 1M argmax samples | tail-start spans per 1M samples beyond top-10 | samples beyond top-10 | mass beyond top-10 | mean entropy (top-10) |
|---|--:|--:|--:|--:|--:|--:|
| Think-SFT | 71 | 10.9 | 166.5 | 1.2% | 0.0122 | 0.615 |
| Think-DPO | 403 | 85.7 (25.9) | 1,847.5 | 0.5% | 0.0050 | 0.338 |
| Think RL final | 108 | 21.4 | 288.0 | 0.2% | 0.0024 | 0.379 |
| Think RL final, recommended | 17 | 3.3 | — (never sampled) | 0.0% | 0.0012 | 0.314 |
| Think-DPO, recommended | 98 | 24.5 (5.2) | — (never sampled) | 0.0% | 0.0016 | 0.256 |
| RL-Zero-Math step 300 | 166 | 31.1 | 856.8 | 0.5% | 0.0048 | 0.433 |
| RL-Zero-Math step 2000 | 543 | 157.3 | 1,358.1 | 0.3% | 0.0031 | 0.368 |

Reading these against the two-process hypothesis in the plan
(descriptive): within the Zero run the argmax-start rate rises 5× from
step 300 to step 2000 and the tail-start rate 1.6×; within the Think
ladder the argmax-start rate goes SFT 10.9 → DPO 25.9 (bare-space spans
excluded) → RL final 21.4, and the tail-start rate 166 → 1,848 → 288.
SFT is the least sharp checkpoint by every measure (1.2% of samples
beyond the top-10, entropy 0.615) and has the lowest tail-start span rate,
so a flatter tail alone does not produce spans; DPO's tail samples are
non-canonical 11× more often than SFT's.

### Launched 2026-09-04: OLMo-3 Instruct and Tulu-3 ladders, DAPO 500, both arms

One box per family, checkpoints queued with `JOBS`; both arms
(`recommended` = each checkpoint's own `generation_config.json`: OLMo-3
0.6 / 0.95, Tulu-3 0.6 / 0.9; `untruncated` = 1.0 / 1.0).

```
sky launch -c nc-instruct skypilot/run.yaml --retry-until-up -i 20 --down -y -d --env HF_TOKEN \
    --env ARMS=recommended,untruncated --env PROMPTS="prompts/dapo_sample500.jsonl" \
    --env JOBS="allenai/Olmo-3-7B-Instruct-SFT:main:instruct-sft allenai/Olmo-3-7B-Instruct-DPO:main:instruct-dpo allenai/Olmo-3-7B-Instruct:main:instruct-main"
sky launch -c nc-tulu skypilot/run.yaml --retry-until-up -i 20 --down -y -d --env HF_TOKEN \
    --env ARMS=recommended,untruncated --env PROMPTS="prompts/dapo_sample500.jsonl" \
    --env JOBS="allenai/Llama-3.1-Tulu-3-8B-SFT:main:tulu3-sft allenai/Llama-3.1-Tulu-3-8B-DPO:main:tulu3-dpo allenai/Llama-3.1-Tulu-3-8B:main:tulu3-rlvr allenai/Llama-3.1-Tulu-3.1-8B:main:tulu31-rlvr"
```

Prompt overlap for Tulu: none of the 500 DAPO prompts matches a prompt in
Tulu 3's public RLVR sets (`allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`,
29,946 rows; `allenai/RLVR-MATH`, 7,500 rows), by the same normalized
exact/prefix rule used for the OLMo-3 filter. Tulu's SFT mixture was not
checked. The Instruct ladder shares the OLMo-3 filter already applied.

### OLMo-3 Instruct ladder, DAPO 500, both arms (2026-09-04)

Same base and code as the Think track, short-answer post-training (no
think block). Recommended = 0.6 / 0.95. "Parsed-only" = rate over the
correct + incorrect outcome buckets (excludes truncated and unparsed
rollouts), the stand-in for judge-conditioning until the judge exists.

| checkpoint | arm | rollouts with ≥1 event | units | non-canonical | **rate** | 95% CI | parsed-only rate | mean tokens | finish (stop / cap) | accuracy |
|---|---|--:|--:|--:|--:|---|--:|--:|---|--:|
| Instruct-SFT | untruncated | 17 / 500 | 269,205 | 72 | 0.0267% | 0.0100–0.0482% | 0.0261% | 538 | 500 / 0 | 32.0% (85 unparsed) |
| Instruct-SFT | recommended | 2 / 500 | 913,241 | 4 | 0.0004% | 0.0000–0.0013% | 0.0015% | 1,826 | 482 / 18 | 38.9% (35 unparsed) |
| Instruct-DPO | untruncated | 105 / 500 | 1,701,689 | 316 | 0.0186% | 0.0153–0.0221% | 0.0187% | 3,403 | 498 / 2 | 74.2% |
| Instruct-DPO | recommended | 25 / 500 | 1,755,763 | 50 | 0.0028% | 0.0018–0.0040% | 0.0021% | 3,512 | 485 / 15 | 80.4% |
| Instruct (RL final) | untruncated | 31 / 500 | 1,272,567 | 60 | 0.0047% | 0.0031–0.0065% | 0.0047% | 2,545 | 500 / 0 | 92.0% |
| Instruct (RL final) | recommended | 5 / 500 | 1,253,525 | 10 | 0.0008% | 0.0002–0.0016% | 0.0008% | 2,507 | 500 / 0 | 93.6% |

Notes on the cells: SFT answers are short (538 tokens at temperature 1)
and often unparsable (85 / 500), so its untruncated CI is wide; DPO
lengthens answers 6× and raises accuracy from 32% to 74%; at the
recommended settings SFT and DPO run to the 32k cap in 18 and 15 rollouts
(repetition loops), which is why those arms have more tokens than the
untruncated ones. Instruct-DPO's recommended arm has 19 byte-fragment
events among its 50 events; Instruct-SFT's untruncated arm 12 of its 72
events in unparsed rollouts. Instruct (RL final) has no truncations, no
unparsed rollouts, and 92–94% accuracy.

Tests per the specification (untruncated; `noncanon.compare --arm untruncated`):
- SFT vs DPO: DPO − SFT = −0.0082 pp, per-token z = 2.8 (p = 0.005),
  per-rollout permutation p = 0.044.
- DPO vs RL final: −0.0139 pp, per-token z = 10.5 (p = 7e-26),
  per-rollout p < 0.00005.
- SFT vs RL final: −0.0220 pp, per-token z = 11.2 (p = 3e-29),
  per-rollout p < 0.00005.
Recommended arm, headline convention: SFT vs DPO +0.0024 pp, z = 4.2
(p = 3e-5), per-rollout p = 0.002; DPO vs RL final −0.0021 pp, z = 3.9
(p = 9e-5), per-rollout p = 0.004; SFT vs RL final +0.0004 pp, z = 1.0
(p = 0.30), per-rollout p = 0.51. Segmentation-only convention: 19 of
Instruct-DPO's 50 recommended-arm events are byte fragments, so its rate
drops to 0.0018% (CI 0.0010–0.0027%) and the DPO vs RL final contrast
weakens to −0.0010 pp, z = 2.2 (p = 0.025), per-rollout p = 0.11 (SFT vs
DPO: +0.0013 pp, z = 2.8, p = 0.005, per-rollout p = 0.039). The
untruncated-arm tests are unchanged to the printed precision under the
segmentation-only convention.

**Prediction status.** Prediction 3 (RL final above DPO): refuted again
on this ladder, in both arms (RL final about 4× below DPO untruncated;
3.5× below at the recommended settings under the headline convention,
2.3× and only per-token significant under segmentation-only). Prediction 4 (SFT ≈ DPO): not
supported; SFT is above DPO per token at temperature 1 (p = 0.044 at the
rollout level) and below it at the recommended settings, but the SFT
cell's short, often-unparsable answers make it the least comparable cell
in the ladder. The Think-ladder pattern "DPO high, RL final low"
replicates here; the "SFT low" part does not.

### Tulu-3-8B ladder (Llama-3.1 base), DAPO 500, both arms (2026-09-04)

Different base and tokenizer (Llama-3 128k vocabulary, three-digit
number tokens), the recipe OLMo-3 Instruct descends from (SFT → DPO →
RLVR via open-instruct; Tulu 3.1 = the same RLVR stage redone with GRPO
instead of PPO plus hyperparameter retuning, per its model card, *not* a
longer run of the same recipe). Recommended = 0.6 / 0.9. This family solves few DAPO problems
(4–20%), so most rollouts land in the "incorrect" bucket; "parsed-only"
is as above (correct + incorrect buckets).

| checkpoint | arm | rollouts with ≥1 event | units | non-canonical | **rate** | 95% CI | parsed-only rate | mean tokens | finish (stop / cap) | accuracy |
|---|---|--:|--:|--:|--:|---|--:|--:|---|--:|
| Tulu-3-SFT | untruncated | 26 / 500 | 406,245 | 139 | 0.0342% | 0.0133–0.0632% | 0.0093% | 812 | 500 / 0 | 4.2% (22 unparsed) |
| Tulu-3-SFT | recommended | 3 / 500 | 913,159 | 24 | 0.0026% | 0.0000–0.0067% | 0.0062% | 1,826 | 484 / 16 | 8.3% (2 unparsed) |
| Tulu-3-DPO | untruncated | 15 / 500 | 471,312 | 40 | 0.0085% | 0.0035–0.0153% | 0.0085% | 943 | 500 / 0 | 10.8% (2 unparsed) |
| Tulu-3-DPO | recommended | 4 / 500 | 531,796 | 18 | 0.0034% | 0.0004–0.0078% | 0.0039% | 1,064 | 498 / 2 | 12.9% (2 unparsed) |
| Tulu-3 (RLVR) | untruncated | 14 / 500 | 555,914 | 36 | 0.0065% | 0.0031–0.0104% | 0.0065% | 1,112 | 500 / 0 | 15.9% (2 unparsed) |
| Tulu-3 (RLVR) | recommended | 7 / 500 | 621,759 | 26 | 0.0042% | 0.0010–0.0084% | 0.0048% | 1,244 | 497 / 3 | 16.0% (4 unparsed) |
| Tulu-3.1 (GRPO RLVR) | untruncated | 17 / 500 | 551,966 | 33 | 0.0060% | 0.0028–0.0101% | 0.0061% | 1,104 | 500 / 0 | 17.6% (7 unparsed) |
| Tulu-3.1 (GRPO RLVR) | recommended | 11 / 500 | 1,215,178 | 25 | 0.0021% | 0.0008–0.0039% | 0.0045% | 2,430 | 480 / 20 | 19.6% (5 unparsed) |

Notes on the cells: Tulu-3-SFT at temperature 1 collapses into word
salad in a subset of rollouts (the 22 unparsed rollouts, e.g. a 2,482-token
rollout ending in a run of underscores and asterisks, another ending in
"completechecks_my ({unit_requipiter refresh intuition stand …"); those 22
rollouts hold 103 of its 139 events, so its headline rate (0.0342%) is a
degenerate-text rate and the parsed-only rate (0.0093%) is the comparable
number. At the recommended settings the SFT and Tulu-3.1 checkpoints run
to the 32k cap in 16 and 20 rollouts (repetition loops; zero events in
them), which inflates those arms' unit counts and lowers their headline
rates below the parsed-only ones. Tulu-3-SFT untruncated has 2
byte-fragment events and 1 OOV-id event (token id 128262, an unused row of
the LM head; see the incident note above); every other Tulu cell has none,
so the two counting conventions differ only in the two SFT pairs (SFT
0.0342% headline vs 0.0335% segmentation-only) and agree exactly elsewhere.
Most common spans, untruncated: SFT `_g|iven` → `_given`; DPO `_be|ads` →
`_b|ead|s` (4×), `)]|:` → `)|]:` (3×); RLVR `б|ов`, `c|ulated`; 3.1
`_f|rant|ic` → `_fr|antic` (4×), `**|\n\n`. Whitespace-first spans: none in
DPO, RLVR or 3.1; 3 in SFT, one of them the OLMo-3 DPO shape (` |/result`
→ ` /|result`) and two starting with indentation tokens.

Tests per the specification (`noncanon.compare --arm untruncated`;
headline convention, with the segmentation-only p-values differing only
in the two SFT pairs, where they are 1.7e-16 / 0.013 and 1.3e-22 / 0.0001):
- SFT vs DPO: DPO − SFT = −0.0257 pp, per-token z = 8.4 (p = 4e-17),
  per-rollout permutation p = 0.013.
- DPO vs RLVR: −0.0020 pp, z = 1.2 (p = 0.24), per-rollout p = 0.59.
- SFT vs RLVR: −0.0277 pp, z = 10.0 (p = 2e-23), per-rollout p = 0.0001.
- RLVR vs 3.1 (longer RL): −0.0005 pp, z = 0.3 (p = 0.74), per-rollout p = 0.86.
- DPO vs 3.1: −0.0025 pp, z = 1.5 (p = 0.13), per-rollout p = 0.50.
Recommended arm: SFT vs DPO +0.0008 pp, z = 0.8 (p = 0.42), per-rollout
p = 0.79; DPO vs RLVR +0.0008 pp, z = 0.7 (p = 0.49), per-rollout p = 0.77;
SFT vs RLVR +0.0016 pp, z = 1.7 (p = 0.10), per-rollout p = 0.56; RLVR vs
3.1 −0.0021 pp, z = 2.6 (p = 0.01), per-rollout p = 0.21; DPO vs 3.1
−0.0013 pp, z = 1.6 (p = 0.10), per-rollout p = 0.43.

Distribution sharpness (`noncanon.tail`, untruncated): mass beyond top-1
is 0.177 for SFT, 0.096 DPO, 0.095 RLVR, 0.104 for 3.1; span-first tokens
at rank 1: SFT 14% (73% outside the top-10, i.e. the word-salad
rollouts), DPO 45%, RLVR 62%, 3.1 32% (byte fragments and the OOV id
excluded).

**Prediction status.** Prediction 3 (RL final above DPO): not supported
on this family either; RLVR and DPO are indistinguishable in both arms
(untruncated per-rollout p = 0.59). The GRPO rerun of the RL stage (3.1)
matches the PPO one at temperature 1 (p = 0.86) and is half its rate at
the recommended settings (0.0021% vs 0.0042%, per-token p = 0.01,
per-rollout p = 0.21). Prediction 4 (SFT ≈
DPO): SFT is above DPO per token at temperature 1 (per-rollout p = 0.013),
driven by the degenerate rollouts; on parsed rollouts only the SFT rate
(0.0093%) is at DPO's level (0.0085%), and at the recommended settings
the three Tulu-3 checkpoints are indistinguishable. The "DPO high, RL
final low" pattern of the two OLMo-3 ladders does not appear here: Tulu's
DPO is *not* elevated (0.0085% vs 0.0186% for both OLMo-3 DPO cells), and
there is no bare-space-then-tail span class in any Tulu cell, which was
the whole of the OLMo-3 DPO excess.

*Notes from Claude:* Tulu's DPO and OLMo-3's DPO are not the same recipe
(Tulu 3: length-normalized DPO on GPT-4o-judged pairs; OLMo 3 Instruct:
the Dolci preference mixture; the loss variants and data differ), and
Tulu's RLVR prompts (GSM8K, MATH, IF constraints) are much
easier than DAPO, which the 4–20% accuracy reflects. The family's rates
are also all low enough (33–40 events per cell outside the degenerate
rollouts) that a 2× difference reaches per-token significance at best
and never per-rollout significance at this sample size, as the
recommended-arm RLVR vs 3.1 pair shows. The clean claim from this family is the absence of the DPO
elevation, not an RL effect in either direction.

## Rollout-level reanalysis: primary metric changed (Brendan, 2026-09-04)

Per the plan amendment of the same date: the per-token rate is demoted
because tokens within a rollout are not independent, so its z-test
p-values overstate the evidence. Primary numbers from here on are the
fraction of rollouts with at least one event (Wilson 95% interval, Fisher
exact test between cells) and the same flag restricted to the first L
tokens among rollouts that reached L (length control). The per-token rate
and the per-rollout permutation test on pooled rates are kept for
continuity. Every full-run cell (not the pilot) was recomputed with the
current code so the windowed flags include standalone fragment events (no
other number changed). All tables from `noncanon.compare --table`; all pairs from
`scripts/compare_all.sh`.

### Rollouts with ≥1 event, DAPO 500, temperature 1 / top-p 1

| cell | with ≥1 event | Wilson 95% | within first 256 tokens (of rollouts ≥ 256) | within first 1,024 | within first 4,096 | per-token rate |
|---|--:|---|---|---|---|--:|
| Think-SFT | 59 (11.8%) | 9.3–14.9% | 2/500 = 0.4% | 4/500 = 0.8% | 11/440 = 2.5% | 0.0030% |
| Think-DPO | 275 (55.0%) | 50.6–59.3% | 37/498 = 7.4% | 110/497 = 22.1% | 213/400 = 53.2% | 0.0186% |
| Think RL final | 51 (10.2%) | 7.8–13.2% | 0/500 = 0.0% | 6/500 = 1.2% | 16/464 = 3.4% | 0.0034% |
| RL-Zero step 300 | 73 (14.6%) | 11.8–18.0% | 9/499 = 1.8% | 14/493 = 2.8% | 33/310 = 10.6% | 0.0085% |
| RL-Zero step 2000 | 292 (58.4%) | 54.0–62.6% | 215/500 = 43.0% | 221/499 = 44.3% | 191/359 = 53.2% | 0.0238% |
| Instruct-SFT | 17 (3.4%) | 2.1–5.4% | 7/410 = 1.7% | 1/31 = 3.2% | — | 0.0267% |
| Instruct-DPO | 105 (21.0%) | 17.7–24.8% | 8/497 = 1.6% | 28/348 = 8.0% | 31/102 = 30.4% | 0.0186% |
| Instruct RL final | 31 (6.2%) | 4.4–8.7% | 1/499 = 0.2% | 10/388 = 2.6% | 6/77 = 7.8% | 0.0047% |
| Tulu-3-SFT | 26 (5.2%) | 3.6–7.5% | 2/491 = 0.4% | 14/106 = 13.2% | — | 0.0342% |
| Tulu-3-DPO | 15 (3.0%) | 1.8–4.9% | 2/498 = 0.4% | 6/168 = 3.6% | — | 0.0085% |
| Tulu-3 RLVR | 14 (2.8%) | 1.7–4.6% | 4/498 = 0.8% | 8/249 = 3.2% | 0/1 † | 0.0065% |
| Tulu-3.1 | 17 (3.4%) | 2.1–5.4% | 4/496 = 0.8% | 1/207 = 0.5% | 0/3 † | 0.0060% |

### Rollouts with ≥1 event, DAPO 500, recommended settings

| cell | with ≥1 event | Wilson 95% | within first 256 | within first 1,024 | within first 4,096 | per-token rate |
|---|--:|---|---|---|---|--:|
| Think-DPO | 76 (15.2%) | 12.3–18.6% | 22/498 = 4.4% | 35/498 = 7.0% | 65/388 = 16.8% | 0.0032% |
| Think RL final | 10 (2.0%) | 1.1–3.6% | 1/500 = 0.2% | 1/500 = 0.2% | 3/457 = 0.7% | 0.0008% |
| Instruct-SFT | 2 (0.4%) | 0.1–1.4% | 1/425 = 0.2% | 0/49 = 0.0% | 0/23 = 0.0% | 0.0004% |
| Instruct-DPO | 25 (5.0%) | 3.4–7.3% | 0/498 = 0.0% | 5/328 = 1.5% | 5/86 = 5.8% | 0.0028% |
| Instruct RL final | 5 (1.0%) | 0.4–2.3% | 0/498 = 0.0% | 2/387 = 0.5% | 1/77 = 1.3% | 0.0008% |
| Tulu-3-SFT | 3 (0.6%) | 0.2–1.7% | 0/482 = 0.0% | 2/127 = 1.6% | 0/17 = 0.0% | 0.0026% |
| Tulu-3-DPO | 4 (0.8%) | 0.3–2.0% | 1/492 = 0.2% | 2/149 = 1.3% | 0/4 † | 0.0034% |
| Tulu-3 RLVR | 7 (1.4%) | 0.7–2.9% | 2/498 = 0.4% | 2/219 = 0.9% | 0/4 † | 0.0042% |
| Tulu-3.1 | 11 (2.2%) | 1.2–3.9% | 4/498 = 0.8% | 5/230 = 2.2% | 0/29 = 0.0% | 0.0021% |

### Rollouts with ≥1 event, AIME 2024/2025 (60 × 8), temperature 1

| cell | with ≥1 event | Wilson 95% | within first 256 | within first 1,024 | within first 4,096 | per-token rate |
|---|--:|---|---|---|---|--:|
| Think-DPO | 330/480 (68.8%) | 64.5–72.7% | 39/478 = 8.2% | 118/478 = 24.7% | 255/468 = 54.5% | 0.0177% |
| Think RL final | 110/480 (22.9%) | 19.4–26.9% | 1/480 = 0.2% | 7/480 = 1.5% | 22/480 = 4.6% | 0.0039% |
| RL-Zero step 2000 | 323/480 (67.3%) | 63.0–71.3% | 204/480 = 42.5% | 207/480 = 43.1% | 218/459 = 47.5% | 0.0273% |

### Correct rollouts only, DAPO 500, temperature 1

| cell | correct rollouts | with ≥1 event | Wilson 95% | within first 256 | within first 1,024 | within first 4,096 |
|---|--:|--:|---|---|---|---|
| Think-SFT | 469 | 43 (9.2%) | 6.9–12.1% | 2/469 = 0.4% | 4/469 = 0.9% | 11/409 = 2.7% |
| Think-DPO | 472 | 256 (54.2%) | 49.7–58.7% | 35/472 = 7.4% | 103/472 = 21.8% | 202/382 = 52.9% |
| Think RL final | 486 | 45 (9.3%) | 7.0–12.2% | 0/486 = 0.0% | 6/486 = 1.2% | 16/451 = 3.5% |
| RL-Zero step 300 | 436 | 58 (13.3%) | 10.4–16.8% | 7/436 = 1.6% | 10/431 = 2.3% | 27/252 = 10.7% |
| RL-Zero step 2000 | 446 | 252 (56.5%) | 51.9–61.0% | 187/446 = 41.9% | 193/445 = 43.4% | 162/305 = 53.1% |
| Instruct-SFT | 133 | 5 (3.8%) | 1.6–8.5% | 3/114 = 2.6% | 1/11 = 9.1% | — |
| Instruct-DPO | 368 | 57 (15.5%) | 12.2–19.5% | 7/365 = 1.9% | 20/246 = 8.1% | 15/51 = 29.4% |
| Instruct RL final | 460 | 25 (5.4%) | 3.7–7.9% | 1/459 = 0.2% | 8/353 = 2.3% | 4/58 = 6.9% |
| Tulu-3-SFT | 20 | 0 (0.0%) | 0.0–16.1% | 0/20 = 0.0% | — | — |
| Tulu-3-DPO | 54 | 1 (1.9%) | 0.3–9.8% | 0/54 = 0.0% | 0/15 = 0.0% | — |
| Tulu-3 RLVR | 79 | 2 (2.5%) | 0.7–8.8% | 1/78 = 1.3% | 2/32 = 6.2% | — |
| Tulu-3.1 | 87 | 2 (2.3%) | 0.6–8.0% | 1/86 = 1.2% | 0/28 = 0.0% | 0/1 † |

Parsed-only (correct + incorrect) for the short-answer families: Instruct
SFT 11/415 = 2.7%, DPO 103/496 = 20.8%, RL final 31/500 = 6.2%; Tulu SFT
18/478 = 3.8%, DPO 15/498 = 3.0%, RLVR 14/498 = 2.8%, 3.1 17/493 = 3.4%.

### Pairwise tests, rollouts as the unit

Fisher exact p on flagged rollouts (all, then within the first 256 /
1,024 / 4,096 tokens), and the per-rollout permutation p on pooled
per-token rates (headline convention) for continuity. Same pairs as the
per-token analysis above; no new pairs.

| pair (a vs b) | a | b | Fisher, all | first 256 | first 1,024 | first 4,096 | permutation (rates) |
|---|--:|--:|--:|--:|--:|--:|--:|
| Think-DPO vs Think RL final | 55.0% | 10.2% | 1e-54 | 3e-12 | 6e-29 | 3e-68 | < 0.00005 |
| Think-DPO vs RL-Zero 2000 | 55.0% | 58.4% | 0.31 | 6e-41 (Zero higher) | 1e-13 (Zero higher) | 1.00 | 0.060 |
| RL-Zero 2000 vs Think RL final | 58.4% | 10.2% | 9e-62 | 5e-78 | 5e-71 | 3e-65 | < 0.00005 |
| RL-Zero 300 vs RL-Zero 2000 | 14.6% | 58.4% | 8e-49 | 3e-64 | 1e-60 | 3e-33 | < 0.00005 |
| Think-SFT vs Think-DPO | 11.8% | 55.0% | 7e-50 | 7e-10 | 4e-31 | 2e-70 | < 0.00005 |
| Think-SFT vs Think RL final | 11.8% | 10.2% | 0.48 | 0.50 | 0.75 | 0.44 | 0.86 |
| RL-Zero 2000: DAPO vs AIME | 58.4% | 67.3% | 0.0045 | 0.90 | 0.75 | 0.12 | 0.39 |
| Think-DPO: DAPO vs AIME | 55.0% | 68.8% | 1e-5 | 0.72 | 0.36 | 0.73 | 0.55 |
| Think RL final: DAPO vs AIME | 10.2% | 22.9% | 7e-8 | 0.49 | 0.79 | 0.41 | 0.69 |
| AIME: Think-DPO vs RL-Zero 2000 | 68.8% | 67.3% | 0.68 | 2e-36 (Zero higher) | 2e-9 (Zero higher) | 0.036 (DPO higher) | < 0.00005 |
| AIME: Think-DPO vs Think RL final | 68.8% | 22.9% | 2e-47 | 2e-11 | 1e-30 | 2e-71 | < 0.00005 |
| AIME: RL-Zero 2000 vs Think RL final | 67.3% | 22.9% | 1e-44 | 9e-72 | 6e-64 | 1e-56 | < 0.00005 |
| Think RL final: untruncated vs recommended | 10.2% | 2.0% | 4e-8 | 1.00 | 0.12 | 0.0041 | 0.0001 |
| recommended: Think-DPO vs Think RL final | 15.2% | 2.0% | 1e-14 | 2e-6 | 3e-10 | 5e-20 | < 0.00005 |
| Think-DPO: untruncated vs recommended | 55.0% | 15.2% | 5e-41 | 0.059 | 7e-12 | 2e-27 | < 0.00005 |
| Instruct-SFT vs Instruct-DPO | 3.4% | 21.0% | 1e-18 | 1.00 | 0.49 | — | 0.044 (SFT higher rate) |
| Instruct-DPO vs Instruct RL final | 21.0% | 6.2% | 6e-12 | 0.021 | 0.0012 | 0.0002 | < 0.00005 |
| Instruct-SFT vs Instruct RL final | 3.4% | 6.2% | 0.053 | 0.026 (SFT higher) | 0.58 | — | < 0.00005 (SFT higher rate) |
| recommended: Instruct-SFT vs DPO | 0.4% | 5.0% | 4e-6 | 0.46 | 1.00 | 0.58 | 0.002 |
| recommended: Instruct-DPO vs RL final | 5.0% | 1.0% | 0.0003 | 1.00 | 0.26 | 0.21 | 0.0037 |
| recommended: Instruct-SFT vs RL final | 0.4% | 1.0% | 0.45 | 0.46 | 1.00 | 1.00 | 0.51 |
| Tulu-3-SFT vs Tulu-3-DPO | 5.2% | 3.0% | 0.11 | 1.00 | 0.0039 (SFT higher) | — | 0.013 |
| Tulu-3-DPO vs Tulu-3 RLVR | 3.0% | 2.8% | 1.00 | 0.69 | 1.00 | — | 0.59 |
| Tulu-3-SFT vs Tulu-3 RLVR | 5.2% | 2.8% | 0.075 | 0.69 | 0.0010 (SFT higher) | — | 0.0001 |
| Tulu-3 RLVR vs Tulu-3.1 | 2.8% | 3.4% | 0.72 | 1.00 | 0.044 (RLVR higher) | 1.00 † | 0.86 |
| Tulu-3-DPO vs Tulu-3.1 | 3.0% | 3.4% | 0.86 | 0.45 | 0.048 (DPO higher) | — | 0.50 |
| recommended: Tulu SFT vs DPO | 0.6% | 0.8% | 1.00 | 1.00 | 1.00 | 1.00 † | 0.79 |
| recommended: Tulu DPO vs RLVR | 0.8% | 1.4% | 0.55 | 1.00 | 1.00 | 1.00 † | 0.77 |
| recommended: Tulu SFT vs RLVR | 0.6% | 1.4% | 0.34 | 0.50 | 0.63 | 1.00 † | 0.56 |
| recommended: Tulu RLVR vs 3.1 | 1.4% | 2.2% | 0.48 | 0.69 | 0.45 | 1.00 † | 0.21 |
| recommended: Tulu DPO vs 3.1 | 0.8% | 2.2% | 0.12 | 0.37 | 0.71 | 1.00 † | 0.43 |
| correct only: Think-SFT vs Think-DPO | 9.2% | 54.2% | 2e-53 | 6e-9 | 2e-28 | 1e-64 | < 0.00005 |
| correct only: Think-DPO vs Think RL final | 54.2% | 9.3% | 5e-54 | 9e-12 | 3e-27 | 9e-65 | < 0.00005 |
| correct only: Think-SFT vs Think RL final | 9.2% | 9.3% | 1.00 | 0.24 | 0.75 | 0.56 | 0.61 |
| correct only: RL-Zero 300 vs 2000 | 13.3% | 56.5% | 4e-43 | 7e-56 | 2e-54 | 1e-27 | 0.0001 |
| correct only: Instruct-SFT vs DPO | 3.8% | 15.5% | 0.0002 | 0.71 | 1.00 | — | 0.0024 |
| correct only: Instruct-DPO vs RL final | 15.5% | 5.4% | 2e-6 | 0.025 | 0.0013 | 0.0024 | < 0.00005 |

Cells marked — have no rollouts that long in one cell (Instruct-SFT and
Tulu answers are short); † marks windows with fewer than 10 eligible
rollouts in a cell, where the p-value is exact but uninformative (the
tool prints the counts beside it).

**Prediction status under the rollout metric.** Prediction 3 (on-policy
RL raises the rate) remains refuted on the Think and Instruct ladders: RL
final flags fewer rollouts than DPO overall in both arms (Fisher p = 1e-54
and 1e-14 for Think, 6e-12 and 0.0003 for Instruct); the windowed tests
agree wherever they have power (Think, all windows, both arms; Instruct
untruncated, p = 0.021 / 0.0012 / 0.0002) and are not significant in the
sparse Instruct recommended-arm windows (p = 1.00 / 0.26 / 0.21). On Tulu
no RL checkpoint flags more rollouts than its DPO (overall p ≥ 0.5; the
only window differences, p = 0.04–0.05 at 1,024 tokens, go the other way
or involve SFT). Prediction 4 (SFT ≈ DPO): refuted on Think (11.8% vs
55.0%, p = 7e-50); on Instruct the overall difference (3.4% vs 21.0%,
p = 1e-18) is not seen within the first 256 or 1,024 tokens (p = 1.00 /
0.49, the latter over 31 SFT rollouts). Prediction 13 (Think-SFT above RL
final): indistinguishable in every window (p ≥ 0.44). Prediction 14
(RL-Zero rises with training): 14.6% → 58.4% (p = 8e-49), and within the
first 256 tokens 1.8% → 43.0%.

Two statuses recorded under the per-token metric change under this one:
Instruct-SFT vs Instruct-DPO reverses direction (per token SFT was
higher, permutation p = 0.044; by rollouts DPO flags 6× more, p = 1e-18),
and Tulu-3-SFT vs Tulu-3-DPO goes from distinguishable (permutation
p = 0.013) to not (5.2% vs 3.0%, p = 0.11). Everything else keeps its
status.

What the length windows show, beyond the earlier analysis:
- **RL-Zero's events come early; Think-DPO's accumulate with length.**
  Overall the two are level (58.4% vs 55.0%, p = 0.31), but within the
  first 256 tokens Zero flags 43.0% of rollouts against DPO's 7.4%
  (p = 6e-41), and by 4,096 tokens they meet (53.2% vs 53.2%). The same
  crossover appears on AIME (42.5% vs 8.2% at 256; 47.5% vs 54.5% at
  4,096).
- **Within a model, DAPO and AIME agree in every window** (p ≥ 0.12)
  while their overall flag rates differ (p ≤ 0.0045); AIME rollouts are
  longer. Consistent with the DAPO-vs-AIME difference being length,
  though the windows do not prove it.
- **Instruct-DPO vs RL final holds in all three windows** (p = 0.021,
  0.0012, 0.0002) and in correct rollouts only. Instruct-SFT vs DPO does
  not (p = 1.00 at 256; p = 0.49 at 1,024 over 31 SFT rollouts), so the
  windows cannot say whether that gap is anything but length.
- **Tulu-SFT vs DPO or RLVR is distinguishable only within the first
  1,024 tokens** (p = 0.004, 0.001), among the 106 SFT rollouts that
  reached that length, which are the word-salad ones.

*Notes from Claude:* the windowed flag is a fixed-horizon "has the first
event happened by token L" and is compared only among rollouts that
reached L, so it conditions on length rather than adjusting for it; a
cell whose long rollouts are its degenerate ones (Tulu-SFT) will look
worse in the long windows for that reason, and a cell with few long
rollouts (Instruct-SFT: 31 past 1,024 tokens, none past 4,096) gives the
long windows no power, so a null there is not evidence of a length
effect. The per-token permutation and the Fisher test disagree in sign
for Instruct-SFT vs DPO (SFT has the higher pooled rate because its few
events sit in short rollouts, while DPO flags 6× more rollouts); the
rollout numbers are the ones the amended plan treats as primary. The
pairwise table holds 37 pairs × up to 4 tests; at that count a few
p-values near 0.05 are expected by chance, and the Tulu 1,024-token
entries at p = 0.04–0.05 should be read that way. Windows with fewer
than 10 eligible rollouts in either cell are marked † and their counts
shown; those p-values are exact but carry no information.

## Reproduction

Every number above comes from these commands, run from a clean checkout
(`uv sync --group dev`; a GPU is needed only for generation). Artifacts for
each cell are on `brendanlong/noncanonical-post-training` under the run
name, so the analysis steps can be run without regenerating.

Prompt sets:

```
uv run python -m noncanon.prompts dapo --sample 500   # prompts/dapo_heldout.jsonl, dapo_sample500.jsonl, filter report
uv run python -m noncanon.prompts aime                # prompts/aime_2024_2025.jsonl
```

The pilot's `prompts/dapo_pilot50.jsonl` was drawn with the earlier
`--pilot 50` flag from the earlier (whitespace-only) filter's 3,430-problem
set and cannot be regenerated from the current filter; it is committed and
stored with the pilot artifacts.

Generation, one launch per cell (RunPod B200 via SkyPilot; `--gpus A100-80GB:1`
works too, at about a quarter of the throughput). `PROMPTS` entries take an
optional `:n` samples-per-prompt suffix; `ARMS` is `untruncated` (temperature
1, top-p 1) or `recommended` (the checkpoint's `generation_config.json`):

| cell | command |
|---|---|
| pilot | `sky launch -c noncanon-pilot skypilot/pilot.yaml -i 20 --down -y -d --env HF_TOKEN` (task since renamed `skypilot/run.yaml`; `PROMPTS=prompts/dapo_pilot50.jsonl ARMS=recommended,untruncated`) |
| think-main (DAPO + AIME) | `sky launch -c nc-think-b200 skypilot/run.yaml --gpus B200:1 -i 20 --down -y -d --env HF_TOKEN --env MODEL=allenai/Olmo-3-7B-Think --env RUN_NAME=think-main` |
| think-dpo, DAPO | same with `MODEL=allenai/Olmo-3-7B-Think-DPO RUN_NAME=think-dpo` (its AIME half was relaunched separately with `--env PROMPTS="prompts/aime_2024_2025.jsonl:8"` after the metrics failure) |
| rlzero-math, DAPO | same with `MODEL=allenai/Olmo-3-7B-RL-Zero-Math RUN_NAME=rlzero-math` (AIME half relaunched the same way) |
| think-sft, DAPO | `... --env MODEL=allenai/Olmo-3-7B-Think-SFT --env RUN_NAME=think-sft --env PROMPTS="prompts/dapo_sample500.jsonl"` |
| rlzero-math-step300, DAPO | `... --env MODEL=allenai/Olmo-3-7B-RL-Zero-Math --env REVISION=step_300 --env RUN_NAME=rlzero-math-step300 --env PROMPTS="prompts/dapo_sample500.jsonl"` |
| think-main-recommended, DAPO | `... --env MODEL=allenai/Olmo-3-7B-Think --env RUN_NAME=think-main-recommended --env ARMS=recommended --env PROMPTS="prompts/dapo_sample500.jsonl"` |
| think-dpo-recommended, DAPO | `... --env MODEL=allenai/Olmo-3-7B-Think-DPO --env RUN_NAME=think-dpo-recommended --env ARMS=recommended --env PROMPTS="prompts/dapo_sample500.jsonl"` |
| instruct-sft, instruct-dpo, instruct-main (both arms) | one launch with `JOBS`, given verbatim in "Launched 2026-09-04" above |
| tulu3-sft, tulu3-dpo, tulu3-rlvr, tulu31-rlvr (both arms) | one launch with `JOBS`, given verbatim in "Launched 2026-09-04" above |

Metrics for a cell (recomputes everything from the stored token IDs; the
on-box copies uploaded by early runs were produced by earlier versions of
`metrics.py` and are superseded by rerunning this):

```
uv run python -m noncanon.metrics --tokenizer <checkpoint> --revision <revision> --records out/<run>/<prompt set>/*.parquet --out-dir out/<run>/<prompt set>/metrics
```

(`--revision` matters only for step checkpoints such as `rlzero-math-step300`.
Within the OLMo-3 family the BPE vocabulary is identical and only the names
of a few added control tokens differ, all of which are excluded as special,
so any family member's tokenizer gives the same numbers; the recompute
script still uses each track's own.)

Comparisons per the analysis specification. Since 2026-09-04 `noncanon.compare`
prints the rollout-level numbers first (flag fraction with Wilson interval,
Fisher exact, fixed-window flags) and the per-token rate second; `--outcome
correct|parsed|...` restricts both cells; `--table` prints the rollout-level
row of every run directory given. `scripts/compare_all.sh` runs every pair
and table in this file; `scripts/recompute_metrics.sh` recomputes every
cell's metrics from the stored records first:

```
uv run python -m noncanon.compare out/think-dpo/dapo_sample500 out/think-main/dapo_sample500
uv run python -m noncanon.compare out/think-dpo/dapo_sample500 out/rlzero-math/dapo_sample500
uv run python -m noncanon.compare out/rlzero-math/dapo_sample500 out/think-main/dapo_sample500
uv run python -m noncanon.compare out/rlzero-math/dapo_sample500 out/rlzero-math/aime_2024_2025   # within-model
uv run python -m noncanon.compare out/think-dpo/dapo_sample500 out/think-dpo/aime_2024_2025
uv run python -m noncanon.compare out/think-dpo/aime_2024_2025 out/rlzero-math/aime_2024_2025
uv run python -m noncanon.compare out/rlzero-math-step300/dapo_sample500 out/rlzero-math/dapo_sample500
uv run python -m noncanon.compare out/think-sft/dapo_sample500 out/think-dpo/dapo_sample500
uv run python -m noncanon.compare out/think-sft/dapo_sample500 out/think-main/dapo_sample500
uv run python -m noncanon.compare out/think-main/dapo_sample500 out/think-main/aime_2024_2025
uv run python -m noncanon.compare out/think-dpo/aime_2024_2025 out/think-main/aime_2024_2025
uv run python -m noncanon.compare out/rlzero-math/aime_2024_2025 out/think-main/aime_2024_2025
uv run python -m noncanon.compare out/think-main/dapo_sample500 out/think-main-recommended/dapo_sample500
uv run python -m noncanon.compare out/think-dpo-recommended/dapo_sample500 out/think-main-recommended/dapo_sample500
uv run python -m noncanon.compare out/think-dpo/dapo_sample500 out/think-dpo-recommended/dapo_sample500
uv run python -m noncanon.compare out/instruct-sft/dapo_sample500 out/instruct-dpo/dapo_sample500 --arm untruncated   # and --arm recommended
uv run python -m noncanon.compare out/instruct-dpo/dapo_sample500 out/instruct-main/dapo_sample500 --arm untruncated
uv run python -m noncanon.compare out/instruct-sft/dapo_sample500 out/instruct-main/dapo_sample500 --arm untruncated
uv run python -m noncanon.compare out/tulu3-sft/dapo_sample500 out/tulu3-dpo/dapo_sample500 --arm untruncated      # and --arm recommended, for each Tulu pair
uv run python -m noncanon.compare out/tulu3-dpo/dapo_sample500 out/tulu3-rlvr/dapo_sample500 --arm untruncated
uv run python -m noncanon.compare out/tulu3-sft/dapo_sample500 out/tulu3-rlvr/dapo_sample500 --arm untruncated
uv run python -m noncanon.compare out/tulu3-rlvr/dapo_sample500 out/tulu31-rlvr/dapo_sample500 --arm untruncated
uv run python -m noncanon.compare out/tulu3-dpo/dapo_sample500 out/tulu31-rlvr/dapo_sample500 --arm untruncated
```

Summary tables (the Summary section at the top of this file):

```
uv run python -m noncanon.summary --arm untruncated ladder \
    "OLMo-3 Think:SFT=out/think-sft/dapo_sample500" "OLMo-3 Think:DPO=out/think-dpo/dapo_sample500" "OLMo-3 Think:RL final=out/think-main/dapo_sample500" \
    "OLMo-3 RL-Zero-Math:step 300=out/rlzero-math-step300/dapo_sample500" "OLMo-3 RL-Zero-Math:step 2000 (final)=out/rlzero-math/dapo_sample500" \
    "OLMo-3 Instruct:SFT=out/instruct-sft/dapo_sample500" "OLMo-3 Instruct:DPO=out/instruct-dpo/dapo_sample500" "OLMo-3 Instruct:RL final=out/instruct-main/dapo_sample500" \
    "Tulu-3-8B:SFT=out/tulu3-sft/dapo_sample500" "Tulu-3-8B:DPO=out/tulu3-dpo/dapo_sample500" "Tulu-3-8B:RLVR (PPO)=out/tulu3-rlvr/dapo_sample500" "Tulu-3-8B:3.1 RLVR (GRPO)=out/tulu31-rlvr/dapo_sample500"
uv run python -m noncanon.summary --arm untruncated pairs --labels DAPO AIME \
    "OLMo-3 Think-DPO=out/think-dpo/dapo_sample500,out/think-dpo/aime_2024_2025" "OLMo-3 Think RL final=out/think-main/dapo_sample500,out/think-main/aime_2024_2025" "OLMo-3 RL-Zero-Math final=out/rlzero-math/dapo_sample500,out/rlzero-math/aime_2024_2025"
uv run python -m noncanon.summary pairs --labels untruncated recommended \
    "OLMo-3 Think-DPO=out/think-dpo/dapo_sample500:untruncated,out/think-dpo-recommended/dapo_sample500:recommended" "OLMo-3 Think RL final=out/think-main/dapo_sample500:untruncated,out/think-main-recommended/dapo_sample500:recommended" \
    "OLMo-3 Instruct-SFT=out/instruct-sft/dapo_sample500:untruncated,out/instruct-sft/dapo_sample500:recommended" "OLMo-3 Instruct-DPO=out/instruct-dpo/dapo_sample500:untruncated,out/instruct-dpo/dapo_sample500:recommended" "OLMo-3 Instruct RL final=out/instruct-main/dapo_sample500:untruncated,out/instruct-main/dapo_sample500:recommended" \
    "Tulu-3-SFT=out/tulu3-sft/dapo_sample500:untruncated,out/tulu3-sft/dapo_sample500:recommended" "Tulu-3-DPO=out/tulu3-dpo/dapo_sample500:untruncated,out/tulu3-dpo/dapo_sample500:recommended" "Tulu-3 RLVR=out/tulu3-rlvr/dapo_sample500:untruncated,out/tulu3-rlvr/dapo_sample500:recommended" "Tulu-3.1=out/tulu31-rlvr/dapo_sample500:untruncated,out/tulu31-rlvr/dapo_sample500:recommended"
```

Tail depth, emitted-token ranks and bare-space spans:

```
uv run python -m noncanon.tail --tokenizer allenai/Olmo-3-7B-Think out/think-main/dapo_sample500 out/think-dpo/dapo_sample500 out/rlzero-math/dapo_sample500 \
    out/think-sft/dapo_sample500 out/rlzero-math-step300/dapo_sample500 out/think-main-recommended/dapo_sample500 out/think-dpo-recommended/dapo_sample500
uv run python -m noncanon.tail --tokenizer allenai/Llama-3.1-Tulu-3-8B out/tulu3-sft/dapo_sample500 out/tulu3-dpo/dapo_sample500 out/tulu3-rlvr/dapo_sample500 out/tulu31-rlvr/dapo_sample500
```

The prompt-set overlap with the OLMo-3 RL training data is computed inside
`noncanon.prompts dapo` and written to `prompts/dapo_filter_report.json`.
The Tulu 3 RLVR overlap check (row counts and zero matches) is:

```
uv run python -m noncanon.prompts overlap prompts/dapo_sample500.jsonl allenai/RLVR-GSM-MATH-IF-Mixed-Constraints allenai/RLVR-MATH
```
