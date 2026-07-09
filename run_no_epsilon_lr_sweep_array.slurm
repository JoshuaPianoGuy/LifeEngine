#!/bin/bash
# ============================================================================
# run_no_epsilon_lr_sweep_array.slurm — learning-rate sweep for the NO-EPSILON
# (pure on-policy REINFORCE) condition, as a single SLURM job array. This is
# run_lr_sweep_fine_array.slurm with --no-epsilon and a different set of rates.
#
# Goal: with epsilon disabled (epsilon pinned to 0 for the whole lifetime, so
# organisms sample straight from the softmax and REINFORCE runs with an
# importance ratio of exactly 1), find which learning rate works best. Directly
# comparable to the epsilon-on sweeps run under the same seeds/grid/condition.
#
#     sbatch run_no_epsilon_lr_sweep_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Grid: 4 learning rates x 3 seeds = 12 tasks (indices 0-11). Same 3 seeds, same
# 500x500 grid, same learning/standard condition, predators disabled — so these
# runs are directly comparable to the LR / epsilon sweeps.
#
# Each task writes CSVs + params.txt to
#   logs/learning/standard/auto-run/noeps_lr<LR>_seed<SEED>/
# The noeps_lr prefix keeps these folders distinct from the epsilon-on
# lr<LR>_seed* runs (a plain re-run into an existing folder would double its CSV
# rows, because Logger auto-save APPENDS and a forced --run-name is not
# collision-checked).
#
#   GENERATIONS=200 sbatch run_no_epsilon_lr_sweep_array.slurm   # override length
# ============================================================================
#SBATCH --job-name="no_epsilon_lr_sweep_fine"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-11
#SBATCH --output=logs/noeps_lr_sweep_%A_%a.out
#SBATCH --error=logs/noeps_lr_sweep_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
# Run from wherever you submit: cd into your SCRATCH clone, then `sbatch ...`.
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from; falls back to a home
# clone if run some other way. Keeping code + logs in scratch avoids the home
# quota and the "which clone am I running?" confusion entirely.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/LifeEngine}}"
# Logs live alongside the project — scratch, when you submit from scratch.
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

# ── Sweep grid ──────────────────────────────────────────────────────────────
# If you change the number of rates, also update --array above to match:
# N_LR*N_SEEDS - 1  (4 rates x 3 seeds -> 0-11).
LEARNING_RATES=(0.005 0.1 0.2 0.3)   # learning rates for the no-epsilon condition
SEEDS=(1 42 999)                     # SAME 3 maps as the LR / epsilon sweeps

# ── Fixed factors (identical for every task, and to run_lr_sweep_fine_array) ─
WIDTH=500
HEIGHT=500
GENERATIONS="${GENERATIONS:-500}"

# ── Decompose the array index into (learning_rate, seed) ────────────────────
# Task IDs 0..11 laid out as LR-major: 0-2 = lr[0] x seeds, 3-5 = lr[1] x seeds,
# 6-8 = lr[2] x seeds, 9-11 = lr[3] x seeds.
N_SEEDS=${#SEEDS[@]}
LR_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
LEARNING_RATE="${LEARNING_RATES[$LR_IDX]}"
MAP_SEED="${SEEDS[$SEED_IDX]}"
RUN_NAME="noeps_lr${LEARNING_RATE}_seed${MAP_SEED}"

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

echo "[array] task=$SLURM_ARRAY_TASK_ID  no-epsilon  lr=$LEARNING_RATE  seed=$MAP_SEED  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"

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
