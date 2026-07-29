#!/bin/bash
# ============================================================================
# submit_stage6.sh — submit the 4 Stage-6 lambda (autocorrelation) arrays.
#
# Stage 6 = W independent mutational random walks per (environment, start), used
# to estimate the ruggedness scalar lambda and (via Stage 8) the NK effective K.
# After the methodology hardening (5 walks x 500 steps, detrend, LS-fit lambda),
# each of the 4 dirs holds 20,040 probes = 5 walks x 501 steps x R=8. Like Stage
# 3/4 the compute is the SAME generic runner (run_landscape_probe_array.slurm);
# only the jobs.json content changed. Stage 6 is RL-OFF, so PARAMS supplies only
# the base env config (map_seed etc.); the hard predator override lives in the
# hard dirs' jobs.json.
#
# PREREQUISITE: the 4 dirs (with jobs.json) exist under landscape/out/
#   stage6_baseline_random  stage6_baseline_centroid
#   stage6_hard_random      stage6_hard_centroid
# (regenerate locally with landscape/stage6_lambda.py ... jobs, then scp.)
#
# USAGE (from project root on the login node):
#   ./submit_stage6.sh                 # N_SHARDS=40 (default), all 4
#   N_SHARDS=60 ./submit_stage6.sh     # finer sharding
#   DIRS="stage6_hard_random" ./submit_stage6.sh
#   DRY_RUN=1 ./submit_stage6.sh       # print, submit nothing
#
# After all arrays finish, copy results_shard_*.csv back and locally run:
#   for d in stage6_baseline_random stage6_baseline_centroid \
#            stage6_hard_random stage6_hard_centroid; do
#     python landscape/merge_shards.py --out-dir landscape/out/$d
#     python landscape/stage6_lambda.py analyze --out-dir landscape/out/$d
#   done
#   python landscape/stage8_nk.py --out-dir landscape/out/stage6_*   # effective K
# ============================================================================
set -euo pipefail

N_SHARDS="${N_SHARDS:-40}"
DIRS="${DIRS:-stage6_baseline_random stage6_baseline_centroid stage6_hard_random stage6_hard_centroid}"
SLURM_SCRIPT="${SLURM_SCRIPT:-run_landscape_probe_array.slurm}"
DRY_RUN="${DRY_RUN:-0}"

BASELINE_PARAMS="${BASELINE_PARAMS:-logs/learning/standard/auto-run/baseline_g1k_learning_noeps_lr0.01_lscape_seed999_r2/params.json}"
HARD_PARAMS="${HARD_PARAMS:-logs/learning/standard/auto-run/roam60d5_g1k_learning_noeps_lr0.02_lscape_seed999_r2/params.json}"

ARRAY_MAX=$((N_SHARDS - 1))
[ -f "$SLURM_SCRIPT" ] || { echo "ERROR: $SLURM_SCRIPT not found. Run from project root." >&2; exit 1; }

params_for() {
    case "$1" in
        *baseline*) echo "$BASELINE_PARAMS" ;;
        *hard*)     echo "$HARD_PARAMS" ;;
        *) echo "ERROR: cannot infer env (baseline/hard) from '$1'" >&2; exit 1 ;;
    esac
}

echo "[submit_stage6] N_SHARDS=$N_SHARDS  array=0-$ARRAY_MAX  dirs='$DIRS'"
n=0
for name in $DIRS; do
    out_dir="landscape/out/${name}"
    params="$(params_for "$name")"
    [ -f "$out_dir/jobs.json" ] || { echo "ERROR: $out_dir/jobs.json not found — generate/scp it first." >&2; exit 1; }
    [ -f "$params" ] || { echo "ERROR: params.json not found: $params" >&2; exit 1; }
    echo "  -> submit $out_dir  (params=$params)"
    if [ "$DRY_RUN" = "1" ]; then
        echo "     [dry-run] OUT_DIR=$out_dir PARAMS=$params N_SHARDS=$N_SHARDS sbatch --array=0-$ARRAY_MAX $SLURM_SCRIPT"
    else
        OUT_DIR="$out_dir" PARAMS="$params" N_SHARDS="$N_SHARDS" \
            sbatch --array=0-"$ARRAY_MAX" "$SLURM_SCRIPT"
    fi
    n=$((n + 1))
done
echo "[submit_stage6] $n array job(s) $( [ "$DRY_RUN" = "1" ] && echo previewed || echo submitted )."
