# Non-canonical token output across post-training stages

Does on-policy RL raise a model's rate of *non-canonical* tokens (emitted
token sequences whose decoded text re-encodes differently, so they are
invisible in any transcript stored as text) relative to the SFT/DPO
checkpoint it started from? Measured on the OLMo-3-7B Think and Instruct
ladders, whose stages share one base, one tokenizer and one codebase.

- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): design, models and why, prompt
  sets, metrics, run order, and pre-registered predictions.
- [RESULTS.md](RESULTS.md): one entry per run, in order, with the exact
  command and the numbers.
- Artifacts (every rollout with its emitted token IDs and logprobs) live on
  the HuggingFace dataset `brendanlong/noncanonical-post-training`.

## Layout

```
noncanon/
  prompts.py     build the held-out prompt sets (DAPO filtered against the OLMo-3 RL data; AIME)
  generate.py    vLLM rollouts keeping emitted token IDs, sampled + top-k logprobs, finish reason
  records.py     the Parquet record schema shared by generation and analysis
  metrics.py     round-trip metric (exact minimal diff), think/answer split, token classes,
                 integer verifier, length/outcome/position slices, top-k entropy
  compare.py     cell-vs-cell tests: rollouts flagged (Fisher exact, fixed-window), per-token rate (secondary)
  tail.py        where in the next-token distribution non-canonical spans start
  upload.py      push a run directory to the HF dataset
  gpu_check.py   fail fast if CUDA is not usable on the host
  tests/         CPU tests on the real OLMo-3 tokenizer
prompts/         generated prompt files and the DAPO filter report (committed)
scripts/         recompute every cell's metrics; run every comparison in RESULTS.md
skypilot/        RunPod task definitions
```

## Setup

```bash
uv sync --group dev          # CPU: prompts, metrics, tests
uv run pytest -q
uv run python -m noncanon.prompts dapo --sample 500
```

Generation needs a GPU and `uv sync --extra gpu` (vLLM 0.11, torch 2.8 CUDA
12.8). On RunPod via SkyPilot:

```bash
sky launch -c nc-think skypilot/run.yaml -i 20 --down -y -d --env HF_TOKEN --env MODEL=allenai/Olmo-3-7B-Think --env RUN_NAME=think-main
sky logs nc-think
```

Metrics recompute everything from the stored IDs:

```bash
uv run python -m noncanon.metrics --tokenizer allenai/Olmo-3-7B-Think \
    --records out/pilot/*.parquet --out-dir out/pilot/metrics
```
