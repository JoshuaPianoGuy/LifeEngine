#!/bin/bash
# ============================================================================
# run_lr_sweep_fine_array.slurm — SECOND, finer learning-rate sweep around the
# 1e-2 region, as a single SLURM job array. Companion to run_sweep_array.slurm.
#
# run_sweep_array.slurm swept the decades {1e-4, 1e-3, 1e-2} and found fitness
# was ~flat with the most in-lifetime learning (MAD) at 1e-2. This sweep probes
# FINER/HIGHER rates around that end to see if pushing past 1e-2 keeps helping.
#
#     sbatch run_lr_sweep_fine_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Grid: 4 learning rates x 3 seeds = 12 tasks (indices 0-11). Same 3 seeds, same
# 500x500 grid, same learning/standard condition, predators disabled — so these
# runs are directly comparable to (and analysed together with) run_sweep_array's.
#
# NOTE — 0.01 is INCLUDED here. It was also swept by run_sweep_array.slurm, but
# the HPC folder structure has since been reorganised so these runs no longer
# collide with the earlier lr0.01_seed* folders (a plain re-run into an existing
# folder would double its CSV rows, because Logger auto-save APPENDS and a forced
# --run-name is not collision-checked). The analysis scripts discover all runs
# under the auto-run tree, so the combined LR curve spans 1e-4 ... 3e-2.
#
# Each task writes CSVs + params.txt to
#   logs/learning/standard/auto-run/lr<LR>_seed<SEED>/
#
#   GENERATIONS=200 sbatch run_lr_sweep_fine_array.slurm   # override run length
# ============================================================================
#SBATCH --job-name="lr_sweep_fine"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-11
#SBATCH --output=logs/lr_sweep_fine_%A_%a.out
#SBATCH --error=logs/lr_sweep_fine_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/LifeEngine}"

# ── Sweep grid ──────────────────────────────────────────────────────────────
# If you change the number of rates, also update --array above to match:
# N_LR*N_SEEDS - 1  (4 rates x 3 seeds -> 0-11).
LEARNING_RATES=(0.005 0.01 0.02 0.03)   # finer/higher rates around the 1e-2 optimum
SEEDS=(1 42 999)                     # SAME 3 maps as run_sweep_array.slurm

# ── Fixed factors (identical for every task, and to run_sweep_array.slurm) ───
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
