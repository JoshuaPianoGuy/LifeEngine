#!/bin/bash
# ============================================================================
# run_lr_no_epsilon_sweep_roaming_array.slurm — learning-rate sweep for the
# NO-EPSILON (pure on-policy REINFORCE) condition in the ROAMING-PREDATOR
# environment, as a single SLURM job array.
#
# Same design as run_no_epsilon_lr_sweep_array.slurm (epsilon disabled — pinned
# to 0 for the whole lifetime, so organisms sample straight from the softmax and
# REINFORCE runs with an importance ratio of exactly 1), but running with roaming
# predators switched on. Goal: find the best learning rate for the no-epsilon
# condition under a constant predation pressure. Directly comparable to the
# epsilon-on roaming sweep and to the predator-free no-epsilon sweep.
#
#     sbatch run_lr_no_epsilon_sweep_roaming_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Grid: 4 learning rates x 3 seeds = 12 tasks (indices 0-11).
#     learning rate : 0.005, 0.01, 0.02, 0.03
#     seeds         : 1, 42, 999   (SAME 3 maps as every other sweep)
# 500x500 grid, learning/standard condition, 60 roaming predators with drain 5
# (patrol predators disabled).
#
# Each task writes CSVs + params.txt to
#   logs/learning/standard/auto-run/roam60d5_noeps_lr<LR>_seed<SEED>/
# The roam60d5_noeps_ prefix keeps these folders distinct from both the
# predator-free noeps_lr<LR>_seed* runs and the epsilon-on roaming runs.
#
#   GENERATIONS=200 sbatch run_lr_no_epsilon_sweep_roaming_array.slurm   # override
#   ROAMING_PREDATORS=45 PREDATOR_DRAIN=3 sbatch ...                     # override env
# ============================================================================
#SBATCH --job-name="lr_no_epsilon_sweep_roaming"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-11
#SBATCH --output=logs/lr_noeps_roaming_%A_%a.out
#SBATCH --error=logs/lr_noeps_roaming_%A_%a.err
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
LEARNING_RATES=(0.005 0.01 0.02 0.03)   # learning rates for the no-epsilon condition
SEEDS=(1 42 999)                        # SAME 3 maps as every other sweep

# ── Fixed factors (identical for every task) ────────────────────────────────
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-500}"
# Roaming-predator environment: 60 predators, drain 5, patrol predators off.
ROAMING_PREDATORS="${ROAMING_PREDATORS:-60}"
PREDATOR_DRAIN="${PREDATOR_DRAIN:-5}"

# ── Decompose the array index into (learning_rate, seed) ────────────────────
# Task IDs 0..11 laid out as LR-major: 0-2 = lr[0] x seeds, 3-5 = lr[1] x seeds,
# 6-8 = lr[2] x seeds, 9-11 = lr[3] x seeds (seed varies fastest).
N_SEEDS=${#SEEDS[@]}
LR_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
LEARNING_RATE="${LEARNING_RATES[$LR_IDX]}"
MAP_SEED="${SEEDS[$SEED_IDX]}"
RUN_NAME="roam60d5_noeps_lr${LEARNING_RATE}_seed${MAP_SEED}"

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
echo "[array]   predators: roaming=$ROAMING_PREDATORS  drain=$PREDATOR_DRAIN  patrol=0"

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
    --roaming-predators "$ROAMING_PREDATORS" \
    --predator-drain "$PREDATOR_DRAIN"

echo "[array] task=$SLURM_ARRAY_TASK_ID done (exit $?)"
