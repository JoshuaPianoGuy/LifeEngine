#!/bin/bash
# ============================================================================
# run_landscape_stage3_seed999_array.slurm — self-contained runner for BOTH
# Stage-3 PCA fitness-landscape planes (seed 999): hard (roam60d5) + baseline.
#
# WHY a dedicated script: the generic run_landscape_probe_array.slurm needs
# OUT_DIR / PARAMS / N_SHARDS passed on the command line. This one BAKES IN the
# two planes + their matching learning-run params so submitting is all that's
# needed:
#
#     sbatch run_landscape_stage3_seed999_array.slurm
#
# It evaluates f(theta) over each plane's jobs.json (written LOCALLY first by
#   landscape/stage3_pca.py ... --out-dir landscape/out/stage3_<env>_coarse).
# Genomes are embedded in jobs.json, so no genome.csv is needed on the cluster —
# only src/eval/, the two out-dirs, and the two params.json files below.
#
#   PLANES (env-major, shard varies fastest):
#     0  hard      landscape/out/stage3_hard_coarse       roam60d5 learning r2 params
#     1  baseline  landscape/out/stage3_baseline_coarse   baseline  learning r2 params
#   Each plane = 12,500 probes (25x25 x R=10 x 2 passes) split into N_SHARDS.
#
# CORE PACKING
# ------------
# One array task == one shard == one core (single-threaded Node). N_SHARDS is
# PER PLANE; TOTAL tasks = N_PLANES * N_SHARDS. With N_SHARDS=20 that is
# 2 * 20 = 40 tasks (--array=0-39) = at most 40 concurrent cores — kept well
# under the ~100 per-job core cap. Each shard is ~625 jobs ~ 1.7 h (the
# --time=6h below leaves ~3x headroom for slower nodes).
#   Fewer/longer shards were chosen deliberately over 64 short ones so the array
#   never exceeds the core cap. To go finer, raise N_SHARDS and set
#   --array=0-(N_PLANES*N_SHARDS - 1) to match. Indices beyond TOTAL are skipped.
#
# Re-running is safe: run_probe.js skips job ids already present in a shard CSV,
# so a timed-out or failed shard can be resubmitted with --array=<i>.
#
# AFTER the array finishes, merge + plot LOCALLY (needs Python/matplotlib):
#   python landscape/merge_shards.py --out-dir landscape/out/stage3_hard_coarse
#   python landscape/stage7_plots.py --out-dir landscape/out/stage3_hard_coarse \
#     --lambda-json landscape/out/stage6_hard_random/lambda_summary.json
#   (likewise for stage3_baseline_coarse + stage6_baseline_random)
#
# (sbatch SCHEDULES this file — you do not execute it yourself.)
# ============================================================================
#SBATCH --job-name="landscape_stage3_s999"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --array=0-39
#SBATCH --output=logs/landscape_stage3_s999_%A_%a.out
#SBATCH --error=logs/landscape_stage3_s999_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/LifeEngine}}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

# ── Planes (baked in) ────────────────────────────────────────────────────────
PLANE_NAMES=(hard baseline)
PLANE_OUT=(
    landscape/out/stage3_hard_coarse
    landscape/out/stage3_baseline_coarse
)
PLANE_PARAMS=(
    logs/learning/standard/auto-run/roam60d5_g1k_learning_noeps_lr0.02_lscape_seed999_r2/params.json
    logs/learning/standard/auto-run/baseline_g1k_learning_noeps_lr0.01_lscape_seed999_r2/params.json
)
N_PLANES=${#PLANE_NAMES[@]}

# ── Sharding ─────────────────────────────────────────────────────────────────
N_SHARDS="${N_SHARDS:-20}"              # shards PER plane (must match --array below)
TOTAL_TASKS=$(( N_PLANES * N_SHARDS ))  # 40 with defaults

# ── Preflight ────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR" logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

# ── Map this task -> (plane, shard) ──────────────────────────────────────────
TASK="${SLURM_ARRAY_TASK_ID}"
if [ "$TASK" -ge "$TOTAL_TASKS" ]; then
    echo "[array] task $TASK >= $TOTAL_TASKS total — nothing to do."; exit 0
fi
PLANE_IDX=$(( TASK / N_SHARDS ))
SHARD=$((     TASK % N_SHARDS ))

ENV_NAME="${PLANE_NAMES[$PLANE_IDX]}"
OUT_DIR="${PLANE_OUT[$PLANE_IDX]}"
PARAMS="${PLANE_PARAMS[$PLANE_IDX]}"
OUT_CSV="$OUT_DIR/results_shard_${SHARD}.csv"

if [ ! -f "$OUT_DIR/jobs.json" ]; then
    echo "ERROR: $OUT_DIR/jobs.json not found. Generate it locally first with landscape/stage3_pca.py ... --out-dir $OUT_DIR"
    exit 1
fi
if [ ! -f "$PARAMS" ]; then
    echo "ERROR: params not found at $PARAMS (copy the learning run's params.json to the cluster)."
    exit 1
fi

echo "[stage3] task=$TASK  plane=$ENV_NAME  shard=${SHARD}/${N_SHARDS}  out_dir=$OUT_DIR"
echo "[stage3]   params=$PARAMS"
echo "[stage3]   -> $OUT_CSV"

"$NODE_BIN" src/eval/run_probe.js \
    --params "$PARAMS" \
    --jobs   "$OUT_DIR/jobs.json" \
    --out    "$OUT_CSV" \
    --shard  "${SHARD}/${N_SHARDS}"
rc=$?
echo "[stage3] task=$TASK done (exit $rc): $ENV_NAME shard $SHARD"
exit $rc
