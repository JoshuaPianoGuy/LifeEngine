#!/bin/bash
# ============================================================================
# submit_stage4.sh — one command to submit ALL Stage-4 random-plane arrays.
#
# Stage 4 = 3 random filter-normalised planes per environment (baseline, hard),
# each a full 25x25 x R=10 x {RL off, RL on} grid = 12,500 probes. This driver
# loops the 6 plane out-dirs and submits each as its own sharded SLURM array via
# the generic runner run_landscape_probe_array.slurm. The predator config for the
# hard planes is already baked into their jobs.json (set at generation time), so
# PARAMS here only supplies the RL hyperparameters for the RL-on pass.
#
# PREREQUISITE: the 6 out-dirs (with jobs.json) must already exist under
#   landscape/out/stage4_baseline/plane{0,1,2}
#   landscape/out/stage4_hard/plane{0,1,2}
# (generated locally with landscape/stage4_random.py, then scp'd to scratch).
#
# USAGE (run from the project root on the login node):
#   ./submit_stage4.sh                     # N_SHARDS=20 (default), submit all 6
#   N_SHARDS=40 ./submit_stage4.sh         # finer sharding, shorter wall-clock
#   ENVS="hard" ./submit_stage4.sh         # only the hard planes
#   PLANES="0" ./submit_stage4.sh          # only plane0 of each env (a quick look)
#   DRY_RUN=1 ./submit_stage4.sh           # print the sbatch commands, submit nothing
#
# After all arrays finish, copy results_shard_*.csv back and locally run:
#   for env in baseline hard; do for p in 0 1 2; do
#     python landscape/merge_shards.py --out-dir landscape/out/stage4_$env/plane$p
#     python landscape/stage7_plots.py --out-dir landscape/out/stage4_$env/plane$p \
#       --lambda-json landscape/out/stage6_${env}_random/lambda_summary.json
#   done; done
# ============================================================================
set -euo pipefail

# ── knobs (override via environment) ────────────────────────────────────────
N_SHARDS="${N_SHARDS:-20}"                 # array size; each shard ~= 12500/N jobs
ENVS="${ENVS:-baseline hard}"              # which environments to submit
PLANES="${PLANES:-0 1 2}"                  # which planes per environment
SLURM_SCRIPT="${SLURM_SCRIPT:-run_landscape_probe_array.slurm}"
DRY_RUN="${DRY_RUN:-0}"

# ── per-environment params.json (learning run = env + RL hyperparameters) ────
BASELINE_PARAMS="${BASELINE_PARAMS:-logs/learning/standard/auto-run/baseline_g1k_learning_noeps_lr0.01_lscape_seed999_r2/params.json}"
HARD_PARAMS="${HARD_PARAMS:-logs/learning/standard/auto-run/roam60d5_g1k_learning_noeps_lr0.02_lscape_seed999_r2/params.json}"

ARRAY_MAX=$((N_SHARDS - 1))

if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "ERROR: $SLURM_SCRIPT not found. Run this from the project root." >&2
    exit 1
fi

params_for() {
    case "$1" in
        baseline) echo "$BASELINE_PARAMS" ;;
        hard)     echo "$HARD_PARAMS" ;;
        *) echo "ERROR: unknown env '$1'" >&2; exit 1 ;;
    esac
}

echo "[submit_stage4] N_SHARDS=$N_SHARDS  array=0-$ARRAY_MAX  envs='$ENVS'  planes='$PLANES'"
n=0
for env in $ENVS; do
    params="$(params_for "$env")"
    if [ ! -f "$params" ]; then
        echo "ERROR: params.json not found for $env: $params" >&2
        exit 1
    fi
    for p in $PLANES; do
        out_dir="landscape/out/stage4_${env}/plane${p}"
        if [ ! -f "$out_dir/jobs.json" ]; then
            echo "ERROR: $out_dir/jobs.json not found — generate/scp it first." >&2
            exit 1
        fi
        echo "  -> submit $out_dir  (params=$params)"
        if [ "$DRY_RUN" = "1" ]; then
            echo "     [dry-run] OUT_DIR=$out_dir PARAMS=$params N_SHARDS=$N_SHARDS sbatch --array=0-$ARRAY_MAX $SLURM_SCRIPT"
        else
            OUT_DIR="$out_dir" PARAMS="$params" N_SHARDS="$N_SHARDS" \
                sbatch --array=0-"$ARRAY_MAX" "$SLURM_SCRIPT"
        fi
        n=$((n + 1))
    done
done
echo "[submit_stage4] $n array job(s) $( [ "$DRY_RUN" = "1" ] && echo 'previewed' || echo 'submitted' )."
