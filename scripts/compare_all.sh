#!/bin/bash
# Every pairwise comparison and rollout-level table in RESULTS.md.
# Usage: scripts/compare_all.sh > compare_all.log   (needs out/<run>/<prompt set>/metrics/analysis.jsonl for every cell)
C="uv run python -m noncanon.compare"
D=dapo_sample500; A=aime_2024_2025
echo "##### TABLE untruncated DAPO"
$C --table --arm untruncated out/think-sft/$D out/think-dpo/$D out/think-main/$D out/rlzero-math-step300/$D out/rlzero-math/$D out/instruct-sft/$D out/instruct-dpo/$D out/instruct-main/$D out/tulu3-sft/$D out/tulu3-dpo/$D out/tulu3-rlvr/$D out/tulu31-rlvr/$D
echo "##### TABLE recommended DAPO"
$C --table --arm recommended out/think-dpo-recommended/$D out/think-main-recommended/$D out/instruct-sft/$D out/instruct-dpo/$D out/instruct-main/$D out/tulu3-sft/$D out/tulu3-dpo/$D out/tulu3-rlvr/$D out/tulu31-rlvr/$D
echo "##### TABLE AIME untruncated"
$C --table out/think-dpo/$A out/think-main/$A out/rlzero-math/$A
echo "##### TABLE correct-only untruncated DAPO"
$C --table --arm untruncated --outcome correct out/think-sft/$D out/think-dpo/$D out/think-main/$D out/rlzero-math-step300/$D out/rlzero-math/$D out/instruct-sft/$D out/instruct-dpo/$D out/instruct-main/$D out/tulu3-sft/$D out/tulu3-dpo/$D out/tulu3-rlvr/$D out/tulu31-rlvr/$D
echo "##### TABLE parsed-only untruncated DAPO"
$C --table --arm untruncated --outcome parsed out/instruct-sft/$D out/instruct-dpo/$D out/instruct-main/$D out/tulu3-sft/$D out/tulu3-dpo/$D out/tulu3-rlvr/$D out/tulu31-rlvr/$D
pair() { echo "##### PAIR $*"; $C "$@" | grep -v "per-token rate, segmentation" -A0 ; }
# Think ladder and Zero (untruncated DAPO)
pair out/think-dpo/$D out/think-main/$D
pair out/think-dpo/$D out/rlzero-math/$D
pair out/rlzero-math/$D out/think-main/$D
pair out/rlzero-math-step300/$D out/rlzero-math/$D
pair out/think-sft/$D out/think-dpo/$D
pair out/think-sft/$D out/think-main/$D
# within-model DAPO vs AIME, and AIME pairs
pair out/rlzero-math/$D out/rlzero-math/$A
pair out/think-dpo/$D out/think-dpo/$A
pair out/think-main/$D out/think-main/$A
pair out/think-dpo/$A out/rlzero-math/$A
pair out/think-dpo/$A out/think-main/$A
pair out/rlzero-math/$A out/think-main/$A
# recommended settings
pair out/think-main/$D out/think-main-recommended/$D
pair out/think-dpo-recommended/$D out/think-main-recommended/$D
pair out/think-dpo/$D out/think-dpo-recommended/$D
# Instruct
for arm in untruncated recommended; do
  pair out/instruct-sft/$D out/instruct-dpo/$D --arm $arm
  pair out/instruct-dpo/$D out/instruct-main/$D --arm $arm
  pair out/instruct-sft/$D out/instruct-main/$D --arm $arm
done
# Tulu
for arm in untruncated recommended; do
  pair out/tulu3-sft/$D out/tulu3-dpo/$D --arm $arm
  pair out/tulu3-dpo/$D out/tulu3-rlvr/$D --arm $arm
  pair out/tulu3-sft/$D out/tulu3-rlvr/$D --arm $arm
  pair out/tulu3-rlvr/$D out/tulu31-rlvr/$D --arm $arm
  pair out/tulu3-dpo/$D out/tulu31-rlvr/$D --arm $arm
done
# correct-only, where the bucket has events
pair out/think-sft/$D out/think-dpo/$D --outcome correct
pair out/think-dpo/$D out/think-main/$D --outcome correct
pair out/think-sft/$D out/think-main/$D --outcome correct
pair out/rlzero-math-step300/$D out/rlzero-math/$D --outcome correct
pair out/instruct-sft/$D out/instruct-dpo/$D --arm untruncated --outcome correct
pair out/instruct-dpo/$D out/instruct-main/$D --arm untruncated --outcome correct
echo ALL_DONE
