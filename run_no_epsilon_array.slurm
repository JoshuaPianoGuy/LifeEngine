#!/bin/bash
# ============================================================================
# run_no_epsilon_array.slurm — the epsilon-OFF baseline for the epsilon sweep,
# as a single SLURM job array.
#
# Goal: isolate whether epsilon-greedy exploration adds anything on top of
# REINFORCE. With --no-epsilon, epsilon is pinned to 0 for the whole lifetime:
# organisms act purely on-policy (sample straight from the softmax) and
# REINFORCE still runs. Compare these runs against run_epsilon_sweep_array.slurm.
#
#     sbatch run_no_epsilon_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Grid: 3 seeds = 3 tasks (indices 0-2). Only the map seed varies — epsilon
# start / decay shape are meaningless when epsilon is disabled, so they are not
# swept here. Same 500x500 grid, learning/standard condition, predators
# disabled, and same seeds as the epsilon sweep — so the two are directly
# comparable.
#
# Each task writes CSVs + params.txt to
#   logs/learning/standard/auto-run/noeps_seed<SEED>/
#
#   GENERATIONS=200 sbatch run_no_epsilon_array.slurm       # override length
#   LEARNING_RATE=0.03 sbatch run_no_epsilon_array.slurm    # set LR winner
# ============================================================================
#SBATCH --job-name="no_epsilon"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-2
#SBATCH --output=logs/noeps_%A_%a.out
#SBATCH --error=logs/noeps_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/LifeEngine}"
# Base output dir. Point this at SCRATCH so runs do NOT fill the home quota
# under $PROJECT_DIR/logs. EDIT to match your cluster's scratch path.
LOG_DIR="${LOG_DIR:-/scratch/$USER/LifeEngine/logs}"

# ── Sweep grid ──────────────────────────────────────────────────────────────
# One factor only (seed). If you change the seed count, update --array above to
# N_SEEDS - 1 (3 seeds -> 0-2).
SEEDS=(1 42 999)                               # SAME 3 maps as the epsilon sweep

# ── Fixed factors (identical for every task, and to the epsilon sweep) ───────
WIDTH=500
HEIGHT=500
GENERATIONS="${GENERATIONS:-500}"
# In-lifetime learning rate. Set to the winner from the LR sweep; overridable.
LEARNING_RATE="${LEARNING_RATE:-0.01}"

# ── Decompose the array index into (seed) ────────────────────────────────────
N_SEEDS=${#SEEDS[@]}          # 3
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

MAP_SEED="${SEEDS[$SEED_IDX]}"
RUN_NAME="noeps_seed${MAP_SEED}"

# ── Run ─────────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR" logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

# headless.js derives a unique run folder from --run-name (we pass one), but keep
# PBS_JOBID mapped for any other unique-naming path.
export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "[array] task=$SLURM_ARRAY_TASK_ID  no-epsilon  seed=$MAP_SEED  lr=$LEARNING_RATE  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"

"$NODE_BIN" src/headless.js \
    --map-seed "$MAP_SEED" \
    --condition learning \
    --mode standard \
    --generations "$GENERATIONS" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --run-name "$RUN_NAME" \
    --log-dir "$LOG_DIR" \
    --learning-rate "$LEARNING_RATE" \
    --no-epsilon \
    --predators-per-patch 0 \
    --roaming-predators 0

echo "[array] task=$SLURM_ARRAY_TASK_ID done (exit $?)"
