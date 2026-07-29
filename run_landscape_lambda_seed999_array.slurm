#!/bin/bash
# ============================================================================
# run_landscape_lambda_seed999_array.slurm — self-contained runner for ALL FOUR
# Stage-6 ruggedness (lambda) CONDITIONS (seed 999):
#   {hard, baseline} x {random-start, centroid-start}.
#
# WHY a dedicated script: bakes in the four out-dirs + their matching
# learning-run params so submitting is all that's needed:
#
#     N_SHARDS=32 sbatch --array=0-127%100 run_landscape_lambda_seed999_array.slurm
#
# Each condition's jobs.json is written LOCALLY first by
#   landscape/stage6_lambda.py jobs --start <random|centroid> --out-dir <dir> ...
# Genomes are embedded in jobs.json — no genome.csv needed on the cluster.
#
# lambda = correlation length from the random-walk autocorrelation (Weinberger
# 1990); headline estimator is the LS fit of ln rho(k) = -k/lambda over k=1..10.
# Running BOTH environments and BOTH start points is the "ruggedness is a
# property of the ENVIRONMENT, not a seed/start accident" comparison.
#
#   CONDITIONS (condition-major, shard varies fastest):
#     0  hard_random      landscape/out/stage6_hard_random       roam60d5 params
#     1  hard_centroid    landscape/out/stage6_hard_centroid     roam60d5 params
#     2  baseline_random  landscape/out/stage6_baseline_random   baseline params
#     3  baseline_centroid landscape/out/stage6_baseline_centroid baseline params
#   Each condition = 20,040 probes: 5 INTERNAL walks x 501 steps x 8 map repeats.
#   (The 5 internal walks are what gives lambda its cross-walk error bar; they
#   live inside one jobs.json and are NOT the four conditions above.)
#   TOTAL across the four conditions = 80,160 probes.
#
# CORE PACKING
# ------------
# One array task == one shard == one core. N_SHARDS is PER CONDITION; TOTAL tasks
# = N_CONDITIONS * N_SHARDS. With N_SHARDS=32 that is 4 * 32 = 128 tasks
# (--array=0-127), throttled to 100 concurrent with %100 to respect the ~100
# per-job core cap. Each shard is ~627 probes ~ 1-2 h at the ~5-10 s/probe the
# stage-3 planes measured; --time=08:00:00 leaves headroom on a slow node.
#   To go finer, raise N_SHARDS and set --array=0-(N_CONDITIONS*N_SHARDS - 1)%100
#   to match. Indices beyond TOTAL are skipped (harmless).
#
# Re-running is safe: run_probe.js skips job ids already present in a shard CSV,
# so a walltime kill just needs a resubmit.
#   *** BUT: if you REGENERATE a jobs.json, delete that out-dir's stale
#   results_shard_*.csv and results.csv FIRST. *** The resume check is by job id
#   only — it cannot tell that an id now refers to a different genome. The
#   pre-2026-07-21 jobs (100 steps x 10 repeats, single walk) also used the
#   legacy id format s{t}_r{r}; the current format is w{walk}_s{step}_r{repeat},
#   and merging the two breaks stage6_lambda.py analyze outright.
#
# AFTER the array finishes, merge + compute lambda LOCALLY (needs Python):
#   for OD in stage6_hard_random stage6_hard_centroid \
#             stage6_baseline_random stage6_baseline_centroid; do
#     python landscape/merge_shards.py     --out-dir landscape/out/$OD
#     python landscape/stage6_lambda.py analyze --out-dir landscape/out/$OD \
#            --kfit 10 --detrend linear
#     python landscape/stage8_nk.py        --out-dir landscape/out/$OD
#   done
#
# (sbatch SCHEDULES this file — you do not execute it yourself.)
# ============================================================================
#SBATCH --job-name="landscape_lambda_s999"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --array=0-127%100
#SBATCH --output=logs/landscape_lambda_s999_%A_%a.out
#SBATCH --error=logs/landscape_lambda_s999_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/LifeEngine}}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

# ── Conditions (baked in) ────────────────────────────────────────────────────
WALK_NAMES=(hard_random hard_centroid baseline_random baseline_centroid)
WALK_OUT=(
    landscape/out/stage6_hard_random
    landscape/out/stage6_hard_centroid
    landscape/out/stage6_baseline_random
    landscape/out/stage6_baseline_centroid
)
HARD_PARAMS=logs/learning/standard/auto-run/roam60d5_g1k_learning_noeps_lr0.02_lscape_seed999_r2/params.json
BASE_PARAMS=logs/learning/standard/auto-run/baseline_g1k_learning_noeps_lr0.01_lscape_seed999_r2/params.json
WALK_PARAMS=("$HARD_PARAMS" "$HARD_PARAMS" "$BASE_PARAMS" "$BASE_PARAMS")
N_WALKS=${#WALK_NAMES[@]}

# ── Sharding ─────────────────────────────────────────────────────────────────
N_SHARDS="${N_SHARDS:-32}"             # shards PER condition (must match --array)
TOTAL_TASKS=$(( N_WALKS * N_SHARDS ))  # 128 with defaults

# ── Preflight ────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR" logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

# ── Map this task -> (walk, shard) ───────────────────────────────────────────
TASK="${SLURM_ARRAY_TASK_ID}"
if [ "$TASK" -ge "$TOTAL_TASKS" ]; then
    echo "[array] task $TASK >= $TOTAL_TASKS total — nothing to do."; exit 0
fi
WALK_IDX=$(( TASK / N_SHARDS ))
SHARD=$((    TASK % N_SHARDS ))

WALK_NAME="${WALK_NAMES[$WALK_IDX]}"
OUT_DIR="${WALK_OUT[$WALK_IDX]}"
PARAMS="${WALK_PARAMS[$WALK_IDX]}"
OUT_CSV="$OUT_DIR/results_shard_${SHARD}.csv"

if [ ! -f "$OUT_DIR/jobs.json" ]; then
    echo "ERROR: $OUT_DIR/jobs.json not found. Generate it locally first with landscape/stage6_lambda.py jobs ... --out-dir $OUT_DIR"
    exit 1
fi
if [ ! -f "$PARAMS" ]; then
    echo "ERROR: params not found at $PARAMS (copy the learning run's params.json to the cluster)."
    exit 1
fi

echo "[lambda] task=$TASK  walk=$WALK_NAME  shard=${SHARD}/${N_SHARDS}  out_dir=$OUT_DIR"
echo "[lambda]   params=$PARAMS"
echo "[lambda]   -> $OUT_CSV"

"$NODE_BIN" src/eval/run_probe.js \
    --params "$PARAMS" \
    --jobs   "$OUT_DIR/jobs.json" \
    --out    "$OUT_CSV" \
    --shard  "${SHARD}/${N_SHARDS}"
rc=$?
echo "[lambda] task=$TASK done (exit $rc): $WALK_NAME shard $SHARD"
exit $rc
