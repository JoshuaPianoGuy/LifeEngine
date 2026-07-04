#!/bin/bash
# ============================================================================
# run_sweep_array.slurm — learning-rate sweep as a SINGLE SLURM job array.
#
# Replaces the sweep_lr.sh submit loop: instead of you running a wrapper
# script, SLURM itself fans this one job out into 9 array tasks (indices 0-8),
# one per (learning_rate, seed) combination. You only ever run ONE command:
#
#     sbatch run_sweep_array.slurm
#
# (sbatch SCHEDULES this file — you are not executing a shell script yourself.)
#
# Grid: 3 learning rates x 3 seeds = 9 tasks. Learning rate is the only factor
# that changes; the 3 seeds are shared across every LR (map = controlled
# replicate). 500x500 grid, learning condition, predators fully disabled.
#
# Each task writes CSVs + params.txt to
#   logs/learning/standard/auto-run/lr<LR>_seed<SEED>/
#
#   GENERATIONS=200 sbatch run_sweep_array.slurm   # override run length
# ============================================================================
#SBATCH --job-name="lr_sweep"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-8
#SBATCH --output=logs/lr_sweep_%A_%a.out
#SBATCH --error=logs/lr_sweep_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/LifeEngine}"

# ── Sweep grid ──────────────────────────────────────────────────────────────
LEARNING_RATES=(0.0001 0.001 0.01)   # 1e-4, 1e-3, 1e-2
SEEDS=(1 42 999)                     # 3 maps; cached pools already exist

# ── Fixed factors (identical for every task) ────────────────────────────────
WIDTH=500
HEIGHT=500
GENERATIONS="${GENERATIONS:-500}"

# ── Decompose the array index into (learning_rate, seed) ────────────────────
# Task IDs 0..8 laid out as LR-major: 0-2 = lr[0] x seeds, 3-5 = lr[1] x seeds,
# 6-8 = lr[2] x seeds.
N_SEEDS=${#SEEDS[@]}
LR_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
LEARNING_RATE="${LEARNING_RATES[$LR_IDX]}"
MAP_SEED="${SEEDS[$SEED_IDX]}"
RUN_NAME="lr${LEARNING_RATE}_seed${MAP_SEED}"

# ── Run ─────────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

# headless.js derives a unique run folder from --run-name (we pass one), but keep
# PBS_JOBID mapped for any other unique-naming path.
export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "[array] task=$SLURM_ARRAY_TASK_ID  lr=$LEARNING_RATE  seed=$MAP_SEED  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"

"$NODE_BIN" src/headless.js \
    --map-seed "$MAP_SEED" \
    --condition learning \
    --mode standard \
    --generations "$GENERATIONS" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --run-name "$RUN_NAME" \
    --learning-rate "$LEARNING_RATE" \
    --predators-per-patch 0 \
    --roaming-predators 0

echo "[array] task=$SLURM_ARRAY_TASK_ID done (exit $?)"
