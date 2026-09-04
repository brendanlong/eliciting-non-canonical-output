#!/bin/bash
# Recompute metrics for every cell under out/ with the current metrics.py (four cells at a time).
# Usage: scripts/recompute_metrics.sh   (records first: download each run from the HF dataset into out/<run>/)
jobs_list=""
for d in out/*/*/; do
  run=$(basename $(dirname $d)); set_=$(basename $d)
  [ "$run" = pilot ] && continue
  ls $d/*.parquet >/dev/null 2>&1 || continue
  jobs_list="$jobs_list $run/$set_"
done
echo "$jobs_list" | tr ' ' '\n' | grep . | xargs -P 4 -I{} bash -c '
  run=$(dirname {}); d=out/{}
  tok=$(case "$run" in tulu3*|tulu31*) echo allenai/Llama-3.1-Tulu-3-8B;; rlzero*) echo allenai/Olmo-3-7B-RL-Zero-Math;; *) echo allenai/Olmo-3-7B-Think;; esac)
  rev=$(case "$run" in rlzero-math-step300) echo step_300;; *) echo main;; esac)
  start=$(date +%s)
  uv run python -m noncanon.metrics --tokenizer $tok --revision $rev --records $d/*.parquet --out-dir $d/metrics > $d/metrics.log 2>&1 && echo "done {} in $(( $(date +%s) - start ))s" || echo "FAILED {}"
'
echo ALL_DONE
