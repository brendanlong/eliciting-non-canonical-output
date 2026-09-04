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
rollout-bootstrap 95% CIs DPO 0.0164–0.0211%, Zero 0.0189–0.0297%;
permutation test on the pooled rate with rollouts permuted: Zero − DPO =
0.0052 pp, two-sided p = 0.053 (`noncanon.compare`; an earlier inline run
of the same test gave 0.057 and 0.055 from a different RNG stream). Segmentation only: CIs DPO 0.0142–0.0179%,
Zero 0.0189–0.0296%; Zero − DPO = 0.0077 pp, p = 0.0016.
Rollouts with ≥1 span: 50.2% vs 58.0%, z = 2.5. Spans per rollout: DPO mean
0.81, variance 1.15, max 8; Zero mean 1.09, variance 8.63, max 52. Median
per-rollout rate: DPO 0.0052%, Zero 0.0138%. Dropping Zero's two densest
rollouts: Zero 0.0206%, difference 0.0046 pp, p = 0.011. (Token-level
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
excluded 5 tokens incomplete UTF-8, 28 cut last word. Length: mean 9,299,
median 8,077, p90 15,702, max 32,767.

| checkpoint | units | non-canonical | **rate** | segmentation only | spans | fragments | rollouts with ≥1 event | span shapes (whitespace / alphabetic / symbolic) |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| Think RL final | 4,649,360 | 160 | **0.0034%** | 155 / 4,649,355 = 0.0033% | 108 | 5 | 51 / 500 | 1 / 25 / 82 |

Think 153 / 4,364,900 (0.0035%), answer 7 / 284,517 (0.0025%). Entropy
(top-10, nats): all positions 0.354, at non-canonical positions 0.472.
The pilot's 50-prompt Think number on the older sample was 0.0037%.

**Tests per the analysis specification in the plan** (DAPO vs DAPO;
per-token = two-proportion z on pooled counts; per-rollout = permutation
test with rollouts as units, 5,000 permutations, so p < 0.0002 reads as
0/5,000; rollout-bootstrap 95% CIs):

| comparison | headline rule: difference | per-token z, p | per-rollout p | segmentation only: difference | per-token z, p | per-rollout p |
|---|--:|---|--:|--:|---|--:|
| Think RL final − Think-DPO | −0.0152 pp | 21.8, <1e-15 | <0.0002 | −0.0127 pp | 19.4, <1e-15 | <0.0002 |
| Think RL final − RL-Zero-Math | −0.0203 pp | 25.8, <1e-15 | <0.0002 | −0.0204 pp | 25.9, <1e-15 | <0.0002 |
| RL-Zero-Math − Think-DPO | +0.0052 pp | 4.8, 1.9e-6 | 0.053 | +0.0077 pp | 7.4, 1.9e-13 | 0.0016 |

Rollout-bootstrap 95% CIs, headline rule: Think RL final 0.0019–0.0059%,
Think-DPO 0.0164–0.0210%, RL-Zero-Math 0.0189–0.0297%.

**Prediction status (DAPO cell).** Prediction 3 (on-policy RL raises the
rate; Think RL final > Think-DPO): **refuted in this cell** — the RL final
checkpoint is about 5× below its DPO starting point under either
convention, at every reported p. The exclusively-on-policy model
(RL-Zero-Math) is above both Think checkpoints per-token, and above DPO
per-rollout at p = 0.055 (headline) / 0.0016 (segmentation only).
Prediction 4 (SFT ≈ DPO) is untested (no SFT cell yet). Prediction 5
(higher inside the CoT): Think RL final 0.0035% think vs 0.0025% answer,
DPO 0.0196% vs 0.0079% (descriptive). Prediction 11 (word joins grow with
RL): the alphabetic-span count is 25 for RL final vs 69 for DPO
(descriptive).

### AIME 2024/2025 (60 problems × 8 samples), RL-Zero-Math

480 rollouts: 477 stop, 3 length; accuracy 34.9% (1 unparsed); excluded 1
incomplete UTF-8, 39 cut last word. Length: mean 11,229, median 11,560,
p90 16,924, max 32,766. Headline rate **0.0273%** (1,470 of 5,389,285
units; 939 spans, 1 fragment; 323 / 480 rollouts with ≥1 event; shapes
whitespace 30 / alphabetic 431 / symbolic 478); segmentation only 0.0273%.
Entropy 0.397 all positions, 0.687 at non-canonical positions. DAPO for
the same checkpoint: 0.0238%. Within-model AIME − DAPO (per the spec):
headline +0.0035 pp, per-token z = 3.0 (p = 0.002), per-rollout
permutation p = 0.41; segmentation only +0.0036 pp, z = 3.1 (p = 0.002),
per-rollout p = 0.38. The Think and DPO AIME cells are still running.

### AIME 2024/2025, Think-DPO

480 rollouts: 434 stop, 46 length; think closed 433 / 480; accuracy
(finished, parsed) 80.7%; excluded 165 tokens incomplete UTF-8, 166 cut
last word. Length: mean 17,384, median 14,779, p90 32,539, max 32,767.
Headline rate **0.0177%** (1,474 of 8,344,375 units; 757 spans, 157
fragments of which 156 standalone; 330 / 480 rollouts with ≥1 event; shapes
whitespace 301 / alphabetic 222 / symbolic 234); segmentation only 0.0158%
(1,318 / 8,344,219). Think 1,422 / 7,961,992 (0.0179%), answer 52 / 382,429
(0.0136%). Entropy 0.393 all positions, 0.967 at non-canonical positions.
Rollout-bootstrap 95% CI (headline) 0.0158–0.0198%.

Within-model AIME − DAPO for DPO: headline −0.0010 pp, per-token z = 1.2
(p = 0.24), per-rollout p = 0.55; segmentation only −0.0002 pp, z = 0.3
(p = 0.78), per-rollout p = 0.88.

AIME, RL-Zero-Math − Think-DPO: headline +0.0096 pp, per-token z = 11.9
(p < 1e-15), per-rollout p < 0.0002; segmentation only +0.0115 pp,
z = 14.6, per-rollout p < 0.0002. (The AIME Think cell is still running.)

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

Metrics for a cell (recomputes everything from the stored token IDs; the
on-box copies uploaded by early runs were produced by earlier versions of
`metrics.py` and are superseded by rerunning this):

```
uv run python -m noncanon.metrics --tokenizer allenai/Olmo-3-7B-Think --records out/<run>/<prompt set>/*.parquet --out-dir out/<run>/<prompt set>/metrics
```

Comparisons per the analysis specification (rollout-bootstrap CIs, per-token
z, per-rollout permutation; both conventions):

```
uv run python -m noncanon.compare out/think-dpo/dapo_sample500 out/think-main/dapo_sample500
uv run python -m noncanon.compare out/think-dpo/dapo_sample500 out/rlzero-math/dapo_sample500
uv run python -m noncanon.compare out/rlzero-math/dapo_sample500 out/think-main/dapo_sample500
uv run python -m noncanon.compare out/rlzero-math/dapo_sample500 out/rlzero-math/aime_2024_2025   # within-model
uv run python -m noncanon.compare out/think-dpo/dapo_sample500 out/think-dpo/aime_2024_2025
uv run python -m noncanon.compare out/think-dpo/aime_2024_2025 out/rlzero-math/aime_2024_2025
```

Tail depth, emitted-token ranks and bare-space spans:

```
uv run python -m noncanon.tail --tokenizer allenai/Olmo-3-7B-Think out/think-main/dapo_sample500 out/think-dpo/dapo_sample500 out/rlzero-math/dapo_sample500
```

The prompt-set overlap with the OLMo-3 RL training data is computed inside
`noncanon.prompts dapo` and written to `prompts/dapo_filter_report.json`.
