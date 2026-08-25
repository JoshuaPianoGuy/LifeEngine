#!/usr/bin/env bash
# landscape/run_slice_landscapes.sh
# ============================================================================
# Merge the companion slice shards, draw the landscapes, and compute the
# difficulty metrics — hard and baseline on ONE shared normalisation range.
#
# Everything downstream of the SLURM array lives here, so re-running after more
# shards land is a single command. merge_shards.py is idempotent (it dedups by
# job id, last wins), so it is safe to re-run over a partly-finished array.
#
#   bash landscape/run_slice_landscapes.sh
#
# Outputs land in landscape/out/slice_landscapes/:
#   normalisation.csv        the min-max constants both stages quote
#   <env>/slices_f_evo.png   the landscape slices, RL off
#   <env>/slices_f_learn.png the same cells with in-lifetime learning on
#   <env>/slices_lift.png    the paired difference
#   <env>/rl_effect.png      that difference as numbers
#   <env>/outcome_endpoints.png  where the runs end up, by outcome
#   <env>/endpoints.csv, <env>/slice_summary.csv
#   metrics/slice_metrics.{png,csv}   FDC, autocorrelation, neutrality
set -euo pipefail
cd "$(dirname "$0")/.."

HARD=landscape/out/slices_h128_companion
BASE=landscape/out/slices_h128_baseline_companion
OUT=landscape/out/slice_landscapes

echo "== merging shards =="
for root in "$HARD" "$BASE"; do
    for d in "$root"/*/; do python landscape/merge_shards.py --out-dir "$d"; done
done

echo
echo "== landscapes (one shared normalisation range across both environments) =="
python landscape/plot_slices.py \
    --slices-root "hard:$HARD" \
    --slices-root "baseline:$BASE" \
    --out-dir "$OUT"

# The ENV field carries the OUTCOME as well as the environment, so the metric
# report contrasts succeeded against failed rather than pooling them into one
# 'hard' column whose spread is really a group difference.
echo
echo "== difficulty metrics =="
ARGS=()
for d in "$HARD"/*/; do
    n=$(basename "$d"); seed=$(echo "$n" | grep -oE 'seed[0-9]+')
    case "$n" in *_succeed_*) g=succeeded;; *) g=failed;; esac
    ARGS+=(--slice "hard/$g:${seed#seed}:${d%/}")
done
for d in "$BASE"/*/; do
    n=$(basename "$d"); seed=$(echo "$n" | grep -oE 'seed[0-9]+')
    # 'weakest', not 'failed': the baseline set was split at a MEDIAN of 7.478,
    # so its underperformers still finish at ~1.7x replacement.
    case "$n" in *_succeed_*) g=succeeded;; *) g=weakest;; esac
    ARGS+=(--slice "baseline/$g:${seed#seed}:${d%/}")
done
python landscape/slice_metrics.py "${ARGS[@]}" \
    --norm-csv "$OUT/normalisation.csv" \
    --out-dir "$OUT/metrics"
