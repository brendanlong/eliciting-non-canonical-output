# Experiment plan: non-canonical token output across post-training stages

**Status:** plan written 2026-09-03, before any runs. Predictions in this
file are Brendan's, recorded before data collection; items marked
*(to confirm)* came up in planning discussion and are awaiting his
explicit sign-off.

## Relationship to the earlier project

This is a follow-up broken out of
[tokenization-hidden-computation-experiment](https://github.com/brendanlong/tokenization-hidden-computation-experiment)
(a toy hidden-computation model, an in-the-wild rate survey, one prompted
elicitation attempt, and two small RL arms). That project's survey was
judged too broad and its data too thin to answer the questions below.
Time spent on the earlier project will be reported alongside this one;
how much of it counts toward the application's limit is the reviewer's
call.

## Question

Non-canonical tokens are emitted token sequences whose decoded text
re-encodes to a different sequence: `encode(decode(ids)) != ids`. They are
invisible in any transcript stored as text.

The main question for v1: **does on-policy RL raise a model's non-canonical
token rate relative to the SFT/DPO checkpoint it started from?** The
reason to expect it might: SFT and DPO train on stored text, whose
tokenization is canonical by construction, whereas on-policy RL computes
its loss on the token IDs the policy actually emitted, which is the only
training stage where a non-canonical sequence can be a training target.

Secondary questions, on the same rollouts:

- Is the rate higher inside chain-of-thought than in the final answer?
- Does the rate rise with rollout length?
- Is the rate higher on rollouts where the model does not get the answer?
- Which token classes (whitespace, words, digits, symbols) go
  non-canonical?
- Does the earlier survey's finding of very low rates on coherent
  in-distribution English (<0.1% per token) hold on longer, more
  realistic prompts?

## What is already known (from the earlier project)

- Every model tested produced non-canonical tokens at some non-zero rate.
- Most models produce them very rarely on in-distribution text (<0.1% of
  tokens once output is conditioned on judged compliance).
- Out-of-distribution prompts can push small models much higher (>1% for
  Llama-3.2-1B on questions in languages it does not handle).
- The non-canonical tokens are mostly whitespace/punctuation seams; GPT-2
  also split digits.
- Temperature and numerical precision both move the rate substantially,
  so sampling settings and dtype must be pinned and reported.
- Sequence-level rates grow with generation length and cannot be
  compared across lengths; per-token rate is the comparable quantity.

## Models

### v1: the OLMo-3-7B family

All v1 checkpoints share one pretrained base (`allenai/Olmo-3-1025-7B`),
one tokenizer, and one post-training codebase (OLMo-core for SFT,
open-instruct for DPO and RLVR). Each stage is initialised from the
previous one, so adjacent checkpoints differ by exactly one training
stage. This is the reason for choosing this family over larger or more
capable models: it is the only place where a matched before/after
comparison across post-training stages exists with the training code
public.

| Track | SFT | DPO | RLVR (final) | Notes |
|---|---|---|---|---|
| Think | `Olmo-3-7B-Think-SFT` | `Olmo-3-7B-Think-DPO` | `Olmo-3-7B-Think` | Long CoT in `<think>` tags. SFT traces are re-tokenized DeepSeek R1 / R1-0528 output. Largest RL run in the family (1375 steps). |
| Instruct | `Olmo-3-7B-Instruct-SFT` | `Olmo-3-7B-Instruct-DPO` | `Olmo-3-7B-Instruct` | Short answers, no think block. 400 RL steps. |

RL step checkpoints are published as HuggingFace branches: Think RL has
55 (steps 125 to 1375), Instruct RL has 8 (50 to 400), and Think-SFT has
43 SFT-step checkpoints. These allow a dose-response curve later; v1 uses
endpoints.

Because the RL stage is initialised from the DPO checkpoint, the
before/after pair for RL is **DPO vs RL final**. SFT is a third point that
separates the DPO stage from the RL stage.

Tokenizer facts that matter for the digit question: OLMo-3 has one-,
two-, and three-digit tokens, no four-digit tokens, and no space-prefixed
digit tokens. A four-digit number is canonically `3+1` and can be emitted
as `1+3` or `2+2`, so digits can be non-canonical in this family.

### Not in v1, available in the same family

- `Olmo-3-7B-RL-Zero-{Math,Code,IF,General}`: RL applied directly to the
  base with no SFT. There is no matched pre-RL checkpoint that attempts
  the task, so these are a cross-model comparison at best. Deferred.
- `Olmo-3-32B-Think` and `Olmo-3-32B-Think-SFT`: scale replication.

### Extension candidates (other families)

Considered for replication or for predictions v1 cannot test. Listed in
rough priority order.

| Family | Checkpoints | What it adds |
|---|---|---|
| Tulu-3-8B (Llama-3.1 base) | `Llama-3.1-Tulu-3-8B-SFT`, `-DPO`, `-8B` (RLVR), `Tulu-3.1-8B` (longer RL) | Same recipe and code on a different base and tokenizer (three-digit tokens); two RL doses |
| Phi-4 (14B) | `Phi-4-reasoning` (SFT on o3-mini traces) → `Phi-4-reasoning-plus` (+ math GRPO) | Most modern SFT→RL pair; short RL dose; training code closed |
| OLMo-2-32B | `OLMo-2-0325-32B-SFT`, `-DPO`, `-Instruct` | Older full ladder at 32B |
| Qwen2.5-32B base family | `DeepSeek-R1-Distill-Qwen-32B` (SFT-only CoT) vs `Open-Reasoner-Zero-32B` / `DAPO-Qwen-32B` (RL-only CoT), plus the base | Same base, SFT-only vs RL-only reasoning; the zero models have weak general compliance |
| Skywork-OR1, AceReason-Nemotron | RL applied on top of `DeepSeek-R1-Distill-Qwen-{7B,14B,32B}` | Thousands of RL steps added to a public SFT-only checkpoint; reasoning-specialised |
| Llama-3.2-1B/3B vs Llama-3.1-8B | released checkpoints | Logit-distilled (pruned from 8B) vs the from-scratch parent: tests the logit-distillation prediction |
| Qwen3 small models | released checkpoints | On-policy logit distillation from Qwen3-32B/235B; a hybrid case for the distillation predictions |

Dropped from consideration: Qwen3 and gpt-oss as primary subjects (no
staged siblings, only decode-mode controls); QwQ-32B vs R1-Distill-Qwen-32B
(different SFT data as well as different RL, so not matched); DeepSeek R1
/ R1-Zero (671B, reasoning tokens unavailable by API, not worth the cost
given the 7B–32B zero models).

An API survey of frontier models with the same prompt set may be run
later if the prompts prove useful; frontier models cannot be run locally
and have no staged siblings, so they do not serve the main question.

## Prompts

1. **Held-out DAPO-style math problems.** Integer-answer competition
   problems of the kind the Think track was RL-trained on. Filtered
   against the public RL training sets (`Dolci-RLZero-Math-7B`, the
   Dolci Think RL data) so no prompt in the evaluation set was an RL
   training prompt. Correctness is checked by a deterministic verifier on
   the boxed answer, not by a judge. Proof-style problems are dropped:
   the models were not trained on them and there is no verifier.
2. **Realistic English chat prompts.** Real single-turn and multi-turn
   conversation prefixes from public chat datasets (WildChat / LMSYS
   Chat-1M), English-filtered; the model generates only the final
   assistant turn. Used for the Instruct track and, once the math
   headline exists, for the Think track. Instruction-following and
   coherence are judged per generation.

Rollouts are given enough room to finish (32k-token cap for Think,
matching its RL training length); the finish reason is stored and
truncated rollouts are reported as their own category.

## Measurement

- **Token IDs are captured at generation** (vLLM, bf16 weights and bf16
  KV cache, no speculative decoding). The round-trip check runs on the
  emitted IDs.
- **Headline metric:** per-token non-canonical rate, counted from a
  minimal diff between the emitted sequence and the canonical
  re-encoding.
- **Secondary:** sequence-level rate at a stated fixed length (familiar,
  free, not the headline); expected non-canonical probability mass from
  top-k logprobs at each position (a low-variance additional estimate);
  dispersion of events within rollouts.
- **Slices:** inside `<think>` vs after it; token class (whitespace,
  word, digit, symbol); position within the rollout; rollout length bin
  × outcome (correct / incorrect / truncated); per-token entropy bin.
- **Sampling:** two arms, model-recommended settings and temperature 1.0
  with top-p 1.0; both reported.
- **Compliance judge:** `gpt-oss-120b` via OpenRouter, reading decoded
  text only (so it cannot see the outcome variable), grading coherence,
  staying on task, and instruction-following for chat prompts. Rates are
  reported raw and conditioned on judged compliance, with the compliance
  fraction. The judge is validated on a random ~200-item subsample
  against a stronger model, and ~30 items are read by hand; agreement is
  reported. Cheap heuristics (repeated n-gram fraction, non-ASCII ratio,
  truncation) are recorded alongside as a non-LLM degeneracy signal.

## Run order

1. **Pilot:** one checkpoint, ~50 held-out DAPO prompts, to measure the
   rollout length distribution and truncation rate and to exercise ID
   capture, the round-trip metric, the verifier, and the upload path end
   to end. Budgets below are re-derived from the pilot.
2. **Think DPO vs Think RL final** on the full math prompt set. The
   headline comparison.
3. **Think SFT final.**
4. **Instruct SFT, DPO, RL final** on chat prompts.
5. Think RL midpoints (roughly steps 350, 700, 1050) for a curve; Think
   on chat prompts; then the RL-Zero models and extension families as
   time allows.

Checkpoints are independent, so wall-clock is reduced by running several
single-GPU boxes over checkpoint slices rather than by using a larger
card. Each checkpoint's records are uploaded as it completes.

## Predictions (Brendan's, recorded before running)

Carried over from the earlier project's plan:

1. Text-distilled models (trained only on re-tokenized teacher output,
   e.g. Think-SFT) have very low non-canonical rates on in-distribution
   text: <0.1% per token.
2. Digits have very low non-canonical rates in modern models with sane
   tokenization (<0.1%); structurally 0% for single-digit tokenizers.
3. From-scratch-trained models have higher rates than text-distilled
   models.
4. Rates are higher during reasoning, especially inside the CoT.
5. On rollouts hard enough to trigger "desperation", reasoning rates are
   higher still.

Added during planning for this project:

6. Predictions 1 and 2 hold on the longer, more realistic prompt sets
   used here.
7. Two distillation mechanisms, pulling in opposite directions:
   teacher-forced **logit** distillation yields a student with *higher*
   rates than its teacher (it is a worse model); **sequence-level**
   distillation from text transcripts yields a student with *lower*
   rates than its teacher (the student trains only on canonical text
   while the teacher was also RL-trained). Only the second is testable in
   v1 (Think-SFT); the first needs the Llama-3.2 or Qwen3 extension
   models.
8. On-policy RL raises the rate: **Think RL final > Think DPO** (and
   likewise Instruct RL final > Instruct DPO, with a smaller expected
   effect given the shorter RL run).
9. *(to confirm)* DPO does not move the rate relative to SFT (DPO trains
   on stored text, not emitted IDs), so the pattern is SFT ≈ DPO < RL.
10. Non-canonical rates rise with rollout length, because entropy rises
    over long rollouts.
11. Non-canonical rates are higher when the model does not know the
    answer, at least for length-matched rollouts.

## Deferred and out of scope for v1

- **Elicitation** (prompting models into non-canonical output). Planned
  as a later step with a structured set of prompt families and fixed
  trial counts; not part of v1.
- **Confusion / non-English prompts.** Detected if it occurs (via the
  judge), not deliberately induced.
- **Simulated multi-turn conversations** and **agentic coding**. Real
  conversation prefixes cover the multi-turn case; agentic harnesses are
  out of budget.
- **Frontier models via API.** May reuse the prompt set later.

## Artifacts and reproducibility

- Every generation is stored with its emitted token IDs, prompt, sampling
  settings, checkpoint revision, finish reason, verifier result, judge
  verdict, and heuristic flags, in JSONL on a HuggingFace dataset.
- Every script that generates or processes an artifact lives in this
  repo and is runnable from a clean checkout; rates are recomputed from
  the stored IDs rather than trusted from stored flags.
- `RESULTS.md` records each run as it completes: exact command, config,
  outcome, and prediction status against this file.
