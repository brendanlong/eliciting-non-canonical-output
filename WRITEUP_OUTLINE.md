# Writeup outline (scaffolding, not prose)

Working document for the MATS writeup. Every number below is copied from
the summary tables in `RESULTS.md` as of 2026-09-04 and should be re-checked
against the regenerated tables before it goes into the doc. Prose is
deliberately left to Brendan (Neel reads LLM-written summaries as a negative
signal); each section lists the facts it should carry and the figure it
should hang off.

## Why restructure before writing

The current draft orders sections by the questions asked, and its headline
("varied by training, but not in an easy to explain way") undersells the one
result Neel is most likely to learn something from:

- The pre-registered prediction (`EXPERIMENT_PLAN.md` predictions 3 and 4:
  SFT ≈ DPO < RL, because only on-policy RL trains on emitted token IDs)
  **failed in both OLMo-3 tracks.** DPO, which trains on canonical text,
  raised the flagged-rollout fraction 4.7× (Think) and 6.2× (Instruct), and
  the on-policy RL stage brought it back down to the SFT level (Think,
  p = 0.48) or slightly above it (Instruct, 6.2% vs 3.4%, p = 0.053).
- RL from the base model with no SFT (RL-Zero-Math) went the other way,
  rising 4× between step 300 and step 2000.
- Tulu-3 (Llama-3.1 base, the recipe OLMo-3 Instruct descends from, but a
  different tokenizer and a different DPO loss and data) showed no stage
  effect on whole rollouts.

That is a concrete, surprising, pre-registered-and-falsified result with a
same-base same-tokenizer control. It should be the first takeaway and the
first figure, and the DAPO-vs-AIME and correct-vs-incorrect sections should
be demoted to robustness checks under it.

Second thing missing from the draft: the "where do spans start" analysis
(`noncanon.tail`) is the closest thing the project has to a mechanistic
finding and is the one that answers the practical question a reader will
ask first ("isn't this just tail sampling that top-p removes?"). About three
quarters of spans begin with the model's argmax token in the three OLMo-3
cells where this was measured (Think-DPO, Think RL, RL-Zero-Math),
which is why the recommended settings cut the rate 3.6–8.5× but do not
eliminate it (Think-DPO still flags 15.2% of rollouts at T=0.6 / top-p 0.95).

Third: Neel asks for randomly selected raw examples right after the
executive summary. `RESULTS.md` already has them ("Random examples" under
the sharpness section); they just need to be lifted into the doc.

## Proposed structure

```
Title
1. Executive summary (write last; ≤600 words; Fig 1 + Fig 2 inline)
2. Random examples (6–8 spans, stated selection rule)
3. Background: what a non-canonical token is and why it matters
4. Setup: models, prompts, sampling, metric, statistics
5. Results
   5.1 Training stage (headline; Fig 1)
   5.2 RL-Zero over checkpoints (Fig 2; pending)
   5.3 How much survives realistic sampling: recommended settings + where spans start (Fig 3)
   5.4 Robustness: length control, DAPO vs AIME, correct-only (Fig 4)
   5.5 Does one event make the next more likely? (clustering; Fig 5)
   5.6 What does a span do to the computation afterwards? (divergence; Fig 6; rerun pending)
6. What I think is going on (hypotheses, and what would separate them)
7. Limitations
8. Time spent and AI use
9. Code and reproduction
```

R1D1 (the "easy accept") is roughly this shape: summary with takeaways
and key experiments, then background, detailed method, results, limitations.
The CoT writeup's weakness per Neel was assuming too much context; section 3
is there to prevent that.

## Section-by-section contents

### 1. Executive summary

Neel's suggested sub-headings: what problem, why interesting; high-level
takeaways; one paragraph and graph per key experiment. Takeaways to draft
from (numbers are DAPO, temperature 1 / top-p 1, fraction of rollouts with
at least one non-canonical event, 500 rollouts per cell):

- Every checkpoint tested emits non-canonical tokens in ordinary math
  rollouts; the shipped OLMo-3 Think model does so in 10.2% of rollouts at
  temperature 1 and 2.0% at its recommended settings (mean ~9k tokens).
- Post-training stage moves the rate a lot, but not the way I predicted.
  Think: SFT 11.8% → DPO 55.0% → RL 10.2%. Instruct: 3.4% → 21.0% → 6.2%.
  Pre-registered prediction was SFT ≈ DPO < RL.
- RL straight from the base model rises with training: RL-Zero-Math
  14.6% at step 300 → 58.4% at step 2000 (more checkpoints pending).
- Tulu-3 (Llama tokenizer) is flat across stages on whole rollouts
  (2.8–5.2%, omnibus p = 0.16), so the DPO spike may be an OLMo-3 tokenizer
  or recipe effect, not a general one. Its DPO is not the same recipe as
  OLMo-3's, so this is a weak replication test.
- Most spans start at the model's argmax token (76–80% in the three OLMo-3
  cells measured; Tulu is lower and varies 14–62%), so this is not purely a
  sampling-tail artefact; recommended settings reduce but do not remove it.
- Restricting to correct rollouts leaves the stage ordering unchanged in
  the OLMo-3 tracks; the first-1,024-token window separates DPO and Zero
  from the rest early but has no power for the short Instruct-SFT and Tulu
  cells. AIME is higher than DAPO overall and indistinguishable within the
  first 1,024 tokens, consistent with (not proof of) a length effect.
- Events cluster within rollouts, but for two different reasons: in the
  on-policy RL cells the excess is the same span recurring (84% of
  consecutive events in Think RL final repeat the same text), while in
  Think-DPO events of different text still sit closer than random
  placement (p < 0.0005).
- A span changes the next-token distribution where it happens (argmax
  differs for 15–36% of spans in the OLMo-3 cells) and the effect is gone
  within about 16 tokens; so a re-tokenized transcript is wrong locally,
  not globally. First-build numbers; the full-prefix rerun is pending.

### 2. Random examples

Lift from `RESULTS.md` "Random examples" (sharpness section). Show
context, emitted tokens, canonical tokens, rank and probability of the first
emitted token. State the selection rule; RESULTS.md groups the examples by
argmax-start vs tail-start per checkpoint but does not record how they were
drawn, so check `noncanon.tail` and say what it does. Include at least one DPO bare-space span
because it is the shape behind the DPO spike. Neel explicitly wants these
"randomly selected, not cherry-picked".

### 3. Background

Move the footnote into the body; the reader may have mech interp
experience but not know the term. Contents:

- Definition: emitted IDs whose decoded text re-encodes differently,
  `encode(decode(ids)) != ids`; `light`+`house` vs `l`+`ighthouse`.
- Why it matters for interpretability: a transcript stored as text
  re-tokenizes canonically, so probes, logit lens, or attribution run on
  the text see positions and residual streams the model never computed.
  (Lead with this; the title is about interpretability.)
- Why it matters for safety: link to the non-canonical-input jailbreak
  paper; the worry that a model could reason in tokens that a text-based
  monitor does not see.
- Why RL specifically: SFT and DPO train on stored text, which is
  canonical by construction; on-policy RL is the only stage whose loss is
  computed on the IDs the policy actually emitted. Hence predictions 3/4.
- One sentence on the prior project and what it established (every model
  tested has a non-zero rate; <0.1% per token on in-distribution text;
  temperature matters), so the reader knows what this project adds.
- Related work: Geh et al. 2024 (canonicity under unconditional sampling
  from base models), Jain et al. 2026 (input-side retokenization on the
  OLMo-2 ladder). Two sentences each, from the notes in
  `EXPERIMENT_PLAN.md`.

### 4. Setup

Neel: "show me enough detail so I can follow along: how did you generate
your data or choose your prompts, how did you define your metrics, what
were your hyperparameters". Bullets:

**Models.** Table, one row per checkpoint, columns: family, base,
tokenizer, stage, HF id. Selection reason: only families with public
per-stage checkpoints from one base, one tokenizer, one training codebase,
so adjacent checkpoints differ by exactly one stage.

| family | base | stages run |
|---|---|---|
| OLMo-3-7B Think | Olmo-3-1025-7B | SFT → DPO → RL final |
| OLMo-3-7B Instruct | same | SFT → DPO → RL final |
| OLMo-3-7B RL-Zero-Math | same, no SFT | step 300, step 2000 (final); more pending |
| Tulu-3-8B | Llama-3.1-8B | SFT → DPO → RLVR (PPO); Tulu-3.1 (GRPO from the same DPO) |

Fix from the draft: these are OLMo-3, not Qwen, in every mention.

**Prompts.** `open-r1/DAPO-Math-17k-Processed`, 14,116 rows, filtered by
normalised exact/80-char-prefix match against the public OLMo-3 RL prompt
sets (Dolci-RL-Zero-Math-7B, 13,314 prompts; Dolci-Think-RL-7B, 102,014),
leaving 3,254 held-out problems, 500 sampled with a fixed seed, one rollout
each. AIME 2024 + 2025 (60 problems, 8 samples each, 480 rollouts) as a
harder, decontaminated set. Integer answers, so correctness is a
deterministic check on the boxed answer. Caveat worth stating: the held-out
DAPO remainder is what AI2 did not select for RL and skews easy (Think
ladder accuracy on finished, parsed rollouts 97.5–98.8%).

**Sampling.** vLLM 0.11, bf16 weights and KV cache, no speculative
decoding, 32k-token cap, default chat template (for RL-Zero-Math the
template injects a fixed "solve step by step ... Answer:" instruction, so
that cell's prompt is not the bare problem). Two arms: temperature 1 /
top-p 1 (the untruncated distribution, what the model learned) and each
checkpoint's `generation_config.json` (0.6 / 0.95 for OLMo-3, 0.6 / 0.9 for
Tulu; RL-Zero-Math ships none). Emitted token IDs and top-10 logprobs
stored per position.

**Metric.** Round-trip on emitted IDs; minimal diff between emitted and
canonical sequences gives spans; standalone byte fragments count as one
event. Primary statistic: fraction of rollouts with at least one event
(Wilson 95% interval, Fisher exact between cells, one omnibus chi-square
per family). Length control: the same flag restricted to the first L
tokens among rollouts that reached L. Per-token rate was the pre-registered
statistic and was demoted on 2026-09-04 because events cluster within
rollouts and the per-token test overstates the evidence; still reported.
Say this in the writeup: Neel likes seeing an analysis decision made for a
stated reason.

**Cost.** B200 at $6.79/h for the full runs, A100-80GB at $1.39/h for the
pilot (RunPod via SkyPilot); fill in the total from the RunPod bill, not
from memory.

### 5.1 Training stage (headline)

Figure 1: four panels (Think, Instruct, RL-Zero-Math, Tulu-3), x = stage,
y = % rollouts with ≥1 event, Wilson error bars, paired bars for T=1 and
recommended where both exist. Annotate the pre-registered prediction.

Text facts:

- Think: 11.8 → 55.0 → 10.2%; DPO vs SFT p = 7e-50, RL vs DPO p = 1e-54,
  RL vs SFT p = 0.48. Mean tokens 8.2k / 8.2k / 9.3k, so this is not length.
- Instruct: 3.4 → 21.0 → 6.2%; RL vs SFT p = 0.053. Mean tokens 538 /
  3,403 / 2,545, so the SFT number is on much shorter rollouts; within
  the first 1,024 tokens it is 1/31, i.e. uninformative.
- RL-Zero-Math: 14.6 → 58.4% (p = 8e-49); 44.3% within the first 1,024
  tokens, far above any other cell.
- Tulu-3: 5.2 / 3.0 / 2.8 / 3.4%; no pair significant; omnibus p = 0.16.
  The one significant Tulu omnibus (within 1,024 tokens, p = 1e-06) is
  driven by SFT rollouts that degenerate into word salad (13.2% vs ≤ 3.6%);
  the pairwise window comparisons at p = 0.04–0.05 are multiple-comparison
  noise per RESULTS.md.
- Correct-only columns give the same orderings for the OLMo-3 tracks.
- State plainly that predictions 3 and 4 were wrong.

### 5.2 RL-Zero over checkpoints (pending)

Figure 2: % rollouts flagged vs RL step, with steps 300 and 2000 measured
and steps 600, 1,000, 1,400 and 1,800 launched on 2026-09-04. If the curve is monotone, say so; if it is not,
say that too.

### 5.3 What survives realistic sampling

Figure 3: stacked bars per checkpoint of spans by rank of the first
emitted token (1 / 2–3 / 4–10 / >10), with DPO's bare-space spans shown
separately. Alternative or companion: T=1 vs recommended paired bars are
already in Figure 1.

Text facts:

- Recommended settings lower every OLMo-3 cell 3.6–8.5× (all p ≤ 7e-4).
  Think-DPO 55.0 → 15.2%, Think RL 10.2 → 2.0%, Instruct-DPO 21.0 → 5.0%,
  Instruct RL 6.2 → 1.0%. Stage ordering unchanged.
- Three quarters of spans start at the argmax token (Think RL 79.6%, DPO
  76.4%, Zero 78.6%). At the recommended settings the beyond-top-10 sample
  share rounds to 0.0%, and the remaining spans (Think RL 17, of which 14
  argmax-start; DPO 98, 92% argmax-start) are almost all argmax starts.
- DPO's excess is a specific shape: 231 of its 403 spans begin with a bare
  space token emitted at p ≈ 1, followed by a token at rank >10 in 81% of
  cases (in this tokenizer a standalone space is canonical only before a
  digit). Excluding those, DPO's argmax-start rate is 25.9 per million
  vs SFT 10.9 and RL 21.4. This is the fact that most needs a hypothesis
  in section 6.
- Sharpness does not explain it: Think-SFT is the least sharp checkpoint
  by every measure (top-10 entropy 0.615, 1.2% of samples beyond the
  top-10) and has the lowest tail-start span rate; DPO is the sharpest
  (0.338, 0.5%) and its beyond-top-10 samples are non-canonical 11× more
  often than SFT's. Among Think-DPO, Think RL and Zero, RL has the highest
  entropy (0.379) and the lowest rate.
- Think vs answer: Think RL 0.0035% inside `<think>` vs 0.0025% after,
  DPO 0.0196% vs 0.0079%. Prediction 5 (higher in CoT) holds weakly.

### 5.4 Robustness

Figure 4: % rollouts flagged within the first L tokens for
L = 256, 1024, 4096, all, one line per OLMo-3 checkpoint. Shows DPO and
Zero separating from the others early, and that the whole-rollout gaps
are not a length artefact.

- DAPO vs AIME: every model flags more AIME rollouts overall (DPO 68.8 vs
  55.0%, RL 22.9 vs 10.2%, Zero 67.3 vs 58.4%; p ≤ 0.0045), AIME rollouts
  are 1.8–2.1× longer, and within the first 1,024 tokens no model differs
  (p ≥ 0.36). The draft's "no difference" is wrong as stated; the correct
  claim is "no difference once length is controlled". Caveat: AIME is 8
  samples per problem and flags cluster by problem, so its intervals are
  somewhat narrow.
- Correct vs incorrect: the draft's "[or did I?]" is fair. What the tables
  support is that restricting to correct rollouts leaves the stage
  ordering unchanged (Think 9.2 → 54.2 → 9.3%). A direct correct-vs-
  incorrect test is underpowered in the Think track (6–12 incorrect
  rollouts per cell) and the per-token rates by outcome go both ways
  (Zero step 2000: 0.0219% correct vs 0.0332% incorrect). Recommend
  reporting it as "ordering robust to conditioning on correctness" rather
  than as a null result on prediction 9.

### 5.5 Does one event make the next more likely?

Source: `noncanon.clustering` table under "Follow-ups after the main
result" in `RESULTS.md`. DAPO 500, temperature 1. Four measurements, and
the writeup should say why there are four: propensity (some rollouts are
prone) and contagion (one event raises the chance of the next) both
produce clustering and need separating.

Figure 5: per cell, observed vs shuffled median gap between consecutive
events, all pairs and different-text pairs side by side. Or a simpler
paired bar of P(another event within 64 tokens) observed vs depth-matched
baseline.

- Propensity: the large cells have about as many ≥2-event rollouts as a
  Poisson process at the cell's rate given each rollout's length
  (Think-DPO 119 vs 118 expected, Instruct-DPO 37 vs 32) or fewer
  (RL-Zero step 2000 88 vs 130). Events are not concentrated in prone
  rollouts beyond what length predicts. Small cells exceed it
  (Tulu-3-SFT 9 vs 3), which is the word-salad rollouts.
- Hazard: P(another event within 64 tokens | event) exceeds the
  depth-matched baseline in 11 of 12 cells, by a lot in the on-policy RL
  cells (Think RL final 32% vs 0.1%, RL-Zero step 2000 18% vs 3%) and by
  little in the DPO cells (Think-DPO 2.0% vs 1.3%). The baseline does not
  condition on the rollout having any event, so this includes propensity.
- Gaps: consecutive events sit closer than random placement within the
  same rollout in 9 of 12 cells. Restricting to consecutive events with
  different span text splits the story: in Think RL final (84% of
  consecutive events repeat the same text) and RL-Zero step 2000 (41%)
  the clustering disappears (p = 0.33 and 0.58); in Think-DPO it stays
  (1,019 vs 1,725 tokens, p < 0.0005; its 18% repeats are almost all byte
  fragments, 41 of 43) and in Tulu-3-SFT too (47 vs 111, p < 0.0005).
- Reading for the writeup: for the on-policy RL checkpoints, "contagion"
  is a learned tokenization habit repeating (the ` $`+`($` span appears
  213 times in RL-Zero step 2000), not a destabilised rollout. For
  Think-DPO, different-text events cluster beyond what count and length
  predict, which is consistent with either a local state (a stretch where
  the tail gets sampled more) or true contagion; section 5.6 is the
  direct test.
- Caveat: the 64-token baseline controls for depth but not prompt, and
  for rare-event cells rests on few draws.

### 5.6 What does a span do to the computation afterwards?

Source: `noncanon.divergence` tables in the same section. **Status: the
numbers on main are from the first build (commit ada2362), which
truncated the prefix to the last 4,096 tokens and spliced it; a
full-prefix rerun of all 12 cells is in progress.** Either wait for the
rerun before writing this section or label the numbers as the first
build. Do not mix the two.

Method to state: for each span, two teacher-forced passes over the same
text, the emitted IDs (A) and the canonical re-tokenization of the decoded
text (B), identical up to the span; at every byte boundary after the span
that both tokenizations share, KL(A‖B) between the next-token
distributions, whether the argmax differs, and at layers 4, 8, ..., 28 the
logit-lens KL and residual cosine distance. At most 3 spans per rollout,
400 per cell; spans with another event inside the 512-token window
skipped.

Figure 6: fraction of boundaries where the argmax differs (and median KL)
vs distance after the span, one line per OLMo-3 cell, log-x. Second
panel: logit-lens KL by layer for boundaries within 16 tokens.

First-build numbers:

- At the span's end the argmax differs for 36% of Think-DPO spans, 31%
  Think-SFT, 23% Think RL final, 15% RL-Zero step 2000 (11% at step 300),
  21–31% on the Instruct ladder; median KL 0.008–0.30 nats in the OLMo-3
  cells.
- Decay: by 5–16 tokens after the span the argmax differs at 2.4–4.5% of
  boundaries in the OLMo-3 cells; from 17 tokens on, 0.3–1.7% with median
  KL below 0.001.
- Logit lens near the span: KL and residual cosine distance are largest
  at layers 24–28 in six of the eight OLMo-3 cells (Think-SFT and RL-Zero
  step 2000 peak at layer 4).
- Tulu cells have 8–17 spans and are too small to read beyond the same
  fast decay.
- Caveat to state: beyond the span the two sequences differ in position
  index (the canonical span usually has a different token count), which
  is a plausible source of the far-field floor.

Reading for the writeup: this is the number the title question turns on.
Running a probe or logit lens on a text transcript gives the wrong
distribution at the span itself, in a quarter to a third of cases a
different argmax, and is back in agreement within about 16 tokens. So
the interpretability damage is local to the span, not a corrupted
rollout. Pair that with the rate numbers: at recommended settings the
shipped Think model has a span in 2% of ~9k-token rollouts, so per token
the exposure is tiny; at DPO or RL-Zero checkpoints it is not.

### 6. What I think is going on

Neel's assessment of R1D1 was that "conceptual analysis was a bit
limited"; this section is the cheap way to score there. Candidate
hypotheses from `EXPERIMENT_PLAN.md` and `RESULTS.md`, with what would
separate them:

1. Training amount: with quality roughly held equal, more training lowers
   the rate (Think RL > DPO > SFT in steps taken from the base). Does not
   explain the DPO spike above SFT or RL-Zero rising with steps.
2. DPO's preference data, or the DPO objective itself, teaches a bare-space
   habit specific to this tokenizer's digit pre-tokenisation; on-policy RL
   then unlearns it because those tokens are sampled and scored. Testable
   with the bare-space split already in `noncanon.tail` and by looking at
   whether the DPO chosen/rejected data contains the pattern.
3. On-policy RL canonicalises rather than releases pressure, when it
   starts from an SFT model: the sampled IDs are mostly canonical, and
   reinforcing them sharpens the model onto canonical continuations. RL
   from the base has no such prior and drifts. This is the reverse of the
   pre-registered mechanism and is worth saying explicitly.
4. Tokenizer: Tulu's flat ladder suggests the effect depends on the
   tokenizer or recipe; one more family with per-stage checkpoints would
   settle it.

### 7. Limitations

Keep the draft's three; add:

- One tokenizer family carries the DPO result; Tulu does not replicate it.
- Recommended settings change temperature and top-p together, so the two
  are not separated.
- DAPO held-out set skews easy; AIME sampled 8× per problem with
  within-problem correlation.
- Correct-vs-incorrect underpowered in the Think track.
- The RL-Zero curve is two points until steps 600–1,800 land.
- The divergence numbers are from a truncated-prefix build; the
  full-prefix rerun may move them.
- Clustering's depth-matched baseline does not control for prompt.

### 8. Time spent and AI use

State the hours plainly and let Neel decide. The wiki estimate for the
earlier project is ~19.8 h at the keyboard (band 14–24 h) from transcript
timing; this project is 8 h so far. Say what carried over concretely (the
round-trip metric definition, the finding that temperature and dtype
matter, the tokenizer-class slices) versus what is new (all code, prompt
sets, models). The FAQ allows a timer reset only for a total change of
direction where earlier code and findings are "not particularly helpful";
this is a partial case, and saying so is better than "I'm unsure if this
counts".
Include the AI-use disclosure (agents ran the experiments; which numbers
you verified by hand and how: Neel explicitly rewards "I read 30
transcripts and confirmed ...").

### 9. Code

Repo link (rename before submitting; fix the `PROJECT_PLAN.md` link to
`EXPERIMENT_PLAN.md`), HF dataset id, one line on `check_results.py`
verifying the Summary and rollout-level tables in `RESULTS.md` against
the stored records.

## Corrections to the current draft

- "Qwen" → OLMo-3 throughout (Methods lists OLMo-3; Results says Qwen).
- AIME: 2024 and 2025, 60 problems × 8 samples; sources are
  `Maxwell-Jia/AIME_2024` and `MathArena/aime_2025`, not the BytedTsinghua
  set linked in the draft. DAPO source is `open-r1/DAPO-Math-17k-Processed`.
- "no difference between DAPO and AIME" → no difference within the first
  1,024 tokens; AIME is higher overall because rollouts are ~2× longer.
- "between 0.5% and 44% ... in the first 1024 tokens": correct as a range
  but the 0.5% end (Tulu-3.1, 1/207) and the Instruct-SFT cell (1/31) are
  on small denominators; quote the OLMo-3 Think and Zero numbers instead.
- "as low as ~0.5% (max ~15%)" at recommended settings: table says 0.4%
  (Instruct-SFT) to 15.2% (Think-DPO). Fine, but say which cells.
- Temperature section has a truncated sentence ("comparable to the .").
- The recommended-settings note ("would have been better to do matched
  settings") reads as an apology; better to state the design reason (arm 1
  measures what the model learned, arm 2 measures what a user sees) and
  move the one-factor-at-a-time gap to Limitations.
- `PROJECT_PLAN.md` link text vs `EXPERIMENT_PLAN.md` file.
