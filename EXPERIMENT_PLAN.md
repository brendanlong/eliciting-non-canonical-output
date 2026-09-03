# Experiment plan: non-canonical token output across post-training stages

**Status:** plan written 2026-09-03, before any runs. Predictions in this
file are Brendan's, recorded before data collection.

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

### Planned follow-up: the teacher/student question

Prediction 7 (below) compares a distilled student with its teacher and
needs pairs where both are runnable and share a tokenizer. Candidates:

| Distillation type | Teacher | Student | Notes |
|---|---|---|---|
| Logit (teacher-forced) | `Llama-3.1-8B-Instruct` | `Llama-3.2-3B-Instruct`, `Llama-3.2-1B-Instruct` | Per Meta's release notes the 3.2 models were pruned from 3.1-8B and pretrained with logit distillation from the 8B and 70B; post-training was done separately per model, so the pair is not stage-matched |
| Logit, incl. on-policy | `Qwen3-32B` | `Qwen3-8B`, `Qwen3-4B`, `Qwen3-1.7B` | Per the Qwen3 report the small models were strong-to-weak distilled (off-policy then on-policy logit distillation) from the 32B/235B. The on-policy stage scores the student's own sampled tokens against the teacher, so it is a hybrid case |
| Sequence-level (text) | `QwQ-32B` | `OpenThinker3-7B` | Student is `Qwen2.5-7B-Instruct` SFT'd on 1.2M QwQ-32B traces; same tokenizer; teacher is RL-trained and runnable. Cleaner than the v1 Think-SFT case, whose teacher (R1) is not runnable |

Dropped from consideration: Qwen3 and gpt-oss as primary subjects (no
staged siblings, only decode-mode controls); QwQ-32B vs R1-Distill-Qwen-32B
(different SFT data as well as different RL, so not matched); DeepSeek R1
/ R1-Zero (671B, reasoning tokens unavailable by API, not worth the cost
given the 7B–32B zero models).

An API survey of frontier models with the same prompt set may be run
later if the prompts prove useful; frontier models cannot be run locally
and have no staged siblings, so they do not serve the main question.

## Prompts

1. **Held-out DAPO-Math-17k problems.** Source:
   `open-r1/DAPO-Math-17k-Processed` (English config, ~17.4k unique
   problems; the original `BytedTsinghua-SIA/DAPO-Math-17k` repeats each
   prompt for training epochs). Answers are integers by DAPO's design, so
   correctness is a deterministic check on the extracted boxed answer,
   not a judge verdict. DAPO is the training distribution for
   `Olmo-3-7B-RL-Zero-Math` and likely part of the Think RL data, so the
   evaluation set is the remainder after filtering against the public RL
   training sets (`Dolci-RLZero-Math-7B`, `Dolci-Think-RL-7B`) by
   normalized exact match on the problem text. The filter script and the
   surviving count are recorded in the repo.
2. **AIME 2024 and 2025** (`Maxwell-Jia/AIME_2024`,
   `MathArena/aime_2025`; 30 integer-answer problems each), sampled
   several times per problem, as a harder tier for the
   "model does not know the answer" comparison. Same verifier.
3. **WildChat-1M chat prefixes** (`allenai/WildChat-1M`, ODC-BY). Real
   user conversations with GPT-3.5/4, with per-conversation language,
   toxicity and turn-count fields. Filtered to English and non-toxic, and
   to prompts whose original assistant reply was at least a few hundred
   tokens (a free proxy for "this prompt calls for a long answer"). A
   prefix is cut at a user turn and the model generates only that
   assistant turn, so both single-turn and multi-turn cases come from
   the same source. Tulu 3's SFT mixture drew ~100k prompts from
   WildChat and the Dolci Instruct data derives from Tulu 3, so prompts
   are also filtered against the Dolci Instruct SFT, DPO and RL prompt
   sets. Used for the Instruct track and, once the math headline exists,
   for the Think track. Instruction-following and coherence are judged
   per generation.

Rollouts are given enough room to finish (32k-token cap for Think,
matching its RL training length); the finish reason is stored and
truncated rollouts are reported as their own category.

## Measurement

- **Token IDs are captured at generation** (vLLM, bf16 weights and bf16
  KV cache, no speculative decoding). The round-trip check runs on the
  emitted IDs.
- **Metrics.** None of the available single numbers is fully
  satisfactory. The best comparison is the length-matched non-canonical
  rate on the same prompt across checkpoints, which depends on how well
  rollouts can be length-matched. All of the following are recorded and
  probably all reported; the per-token rate is the best single number:
  - per-token non-canonical rate: canonical tokens inside non-canonical
    spans divided by canonical tokens, from a minimal diff between the
    emitted sequence and its canonical re-encoding, so that numerator and
    denominator count the same thing (the emitted-token count inside
    spans is reported alongside);
  - length-matched rate on the same prompt across checkpoints;
  - sequence-level rate at a stated fixed length (familiar, free);
  - expected non-canonical probability mass from top-k logprobs at each
    position (a low-variance additional estimate);
  - dispersion of events within rollouts.
- **Slices:** inside `<think>` vs after it; token class (whitespace,
  word, digit, symbol); position within the rollout; rollout length bin
  × outcome (correct / incorrect / truncated); per-token entropy bin.
- **Sampling:** the primary arm is temperature 1.0 with top-p 1.0, the
  untruncated distribution, which measures what the model actually
  learned independent of how the provider thinks it should be run. The
  model-recommended settings (every OLMo-3 checkpoint ships
  `temperature=0.6, top_p=0.95` in its `generation_config.json`, which
  both sharpens the distribution and truncates the tail where
  non-canonical tokens live) are a second, less important run rather
  than dropped: if the recommended settings take the rate to ~0, that
  says a lot about the practical risk versus the theoretical risk.
  (Decision 2026-09-03, after the pilot's recommended-settings arm
  returned 0 in 468k tokens: untruncated results first.)
- **Compliance judge:** `gpt-oss-120b` via OpenRouter, reading decoded
  text only (so it cannot see the outcome variable), grading coherence,
  staying on task, and instruction-following for chat prompts. Rates are
  reported raw and conditioned on judged compliance, with the compliance
  fraction. The judge is validated on a random ~200-item subsample
  against a stronger model, and agreement is reported. Some transcripts
  and verdicts are read by hand; the number read is reported in the
  results rather than committed to here. Cheap heuristics (repeated
  n-gram fraction, non-ASCII ratio, truncation) are recorded alongside
  as a non-LLM degeneracy signal.

## Run order

1. **Pilot:** one checkpoint, ~50 filtered DAPO prompts, to measure the
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

Results from the earlier project, predicted to hold on the longer and
more realistic prompt sets used here:

1. Text-distilled models (trained only on re-tokenized teacher output,
   e.g. Think-SFT) have very low non-canonical rates on in-distribution
   text: <0.1% per token.
2. Digits have very low non-canonical rates in modern models with sane
   tokenization (<0.1%); structurally 0% for single-digit tokenizers.

Predictions for this project:

3. **On-policy RL raises the rate.** Models trained exclusively with
   on-policy RL (the RL-Zero models, if run as a follow-up) have higher
   rates than models whose post-training was SFT, and on-policy RL
   checkpoints trained on top of SFT (Think RL final, Instruct RL final)
   have higher rates than the earlier SFT endpoints (SFT, DPO). The
   expected effect is smaller for Instruct, given its shorter RL run and
   shorter rollouts. No from-scratch models are tested in v1.
4. DPO does not move the rate relative to SFT (DPO trains on stored
   text, not emitted IDs), so the pattern within a track is
   SFT ≈ DPO < RL.
5. Rates are higher during reasoning, especially inside the CoT.
   Reasoning is what the RL stage trains most directly, and the effect
   would be larger for the CoT if the CoT is intentionally not trained
   on; whether that is the case for this model is not known.
6. On rollouts hard enough to trigger "desperation", reasoning rates are
   higher still.
7. Two distillation mechanisms, pulling in opposite directions:
   teacher-forced **logit** distillation yields a student with *higher*
   rates than its teacher (it is a worse model); **sequence-level**
   distillation from text transcripts yields a student with *lower*
   rates than its teacher (the student trains only on canonical text
   while the teacher was also RL-trained). Only the second is testable in
   v1 (Think-SFT); the first needs the Llama-3.2 or Qwen3 extension
   models.
8. Non-canonical rates rise with rollout length, because entropy rises
   over long rollouts.
9. Non-canonical rates are higher when the model does not know the
   answer, at least for length-matched rollouts, also because entropy is
   higher when the model does not know the answer.

Added 2026-09-03 after the pilot (Think RL final, 50 DAPO prompts:
0.0037% at temperature 1, 0 at recommended settings; all events inside
the think block; several events are two words emitted without the space
between them, e.g. `above`+`x`, `as`+`y`, `of`+`t`):

10. Two hypotheses for the very low rate, not yet separable: models reach
    very low non-canonical rates after sufficient training, or the DAPO
    prompts are *too* in-distribution (the model was RL-trained on very
    similar problems). AIME 2024/2025 is shaped slightly differently from
    the training problems, which a reasoning model should generalize to:
    **the AIME rate will still be quite low but slightly higher than on
    DAPO.** Something farther out of distribution but still realistic
    should be higher again (dataset to be chosen).

    *Notes from Claude:* the held-out DAPO set is likely worse than
    merely in-distribution: it is what remained after AI2 selected their
    13.3k RL subset from DAPO, and the pilot's 98–100% accuracy suggests
    the remainder skews easy. AIME 2024/2025 are reported as benchmarks
    for these models in the OLMo 3 report, so AI2 decontaminated against
    them, which makes them genuinely held out rather than just differently
    worded.
11. The dropped-space word joins have the same shape as the earlier
    project's prompted `light`+`house` splits, i.e. non-canonical
    tokenization learned during reasoning. **These become more common
    over RL training** (checkpoint comparison), and the rate at which
    they grow is informative, if hackishly, for extrapolating to
    frontier models with much longer RL runs.

    *Notes from Claude:* in the pilot, five of the ten spans have this
    word-join shape, and four of those five are in one rollout, the
    longest in the arm and the only incorrect one. That hints the joins
    cluster in struggling rollouts rather than spreading evenly; the
    dispersion statistic in the metrics section is the check. The
    examples file can tag each span by shape (word join, whitespace run,
    symbol join) so the join rate is tracked separately from the total
    across checkpoints.
12. Some models are known to produce nearly incomprehensible reasoning.
    If any such model can be run, **its incomprehensible reasoning will
    have a surprisingly large non-canonical rate.**

    *Notes from Claude:* the nearest runnable candidate is already in the
    family: `Olmo-3-7B-RL-Zero-Math`, RL straight from the base with no
    SFT stage to canonicalize it, same tokenizer, and zero-style models
    are the ones with documented readability problems (DeepSeek reported
    language mixing and poor readability for R1-Zero).
    `Open-Reasoner-Zero-32B` and `DAPO-Qwen-32B` are the same recipe on
    Qwen2.5-32B. The judge's coherence verdict plus the degeneracy
    heuristics separate "incomprehensible but canonical" from
    non-canonical, which this prediction needs.

## Deferred and out of scope for v1

- **Elicitation** (prompting models into non-canonical output). Planned
  as a later step with a structured set of prompt families and fixed
  trial counts; not part of v1.
- **Proof-style math problems.** Considered as a second math prompt set
  and dropped: the models were not trained on them and there is no
  verifier for correctness.
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
