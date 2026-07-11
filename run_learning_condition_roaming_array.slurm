#!/bin/bash
# ============================================================================
# run_learning_condition_roaming_array.slurm — the LEARNING condition
# (GA + in-lifetime REINFORCE, NO epsilon exploration, learning rate 0.02) in
# the roaming-predator environment, as a single SLURM job array over 5 seeds.
#
# This is one of a matched trio of "final comparison" runs that share IDENTICAL
# run details (500x500 grid, 1000 generations, 60 roaming predators @ drain 5,
# 5 seeds) and differ ONLY in the learning/evolution mechanism:
#     run_learning_condition_roaming_array.slurm   <- THIS FILE (GA + RL)
#     run_evolution_condition_roaming_array.slurm  (GA only, no RL)
#     run_pure_rl_condition_roaming_array.slurm    (RL only, no GA)
# The no-epsilon + lr 0.02 setting is the one selected from the roaming
# learning-rate sweep (analyse_lr_sweep_noeps.py).
#
#     sbatch run_learning_condition_roaming_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Grid: 5 seeds = 5 tasks (indices 0-4).
#     seeds : 1, 42, 999   (the 3 existing maps, reused)
#           + 123, 456     (2 NEW maps — see MAP-SEED note below)
# 500x500 grid, learning/standard condition, 60 roaming predators with drain 5
# (patrol predators disabled).
#
# MAP-SEED note (new worlds are generated + stored automatically)
# ---------------------------------------------------------------
# headless.js builds each world deterministically from --map-seed. On first use
# of a seed it generates the map pool and caches it atomically to
# src/maps/map_pool_seed<SEED>.json (temp-write + rename, so parallel tasks that
# hit a brand-new seed at once can't corrupt it). So seeds 123 and 456 need no
# manual pre-generation — the first task to use each one creates and stores it,
# and every later run (any condition) reuses the same cached world.
#
# Each task writes CSVs + params.txt to
#   logs/learning/standard/auto-run/roam60d5_g1k_learning_noeps_lr0.02_seed<SEED>/
# The g1k (1000-gen) tag keeps these folders distinct from the 500-gen sweep
# runs in the same dir — a forced --run-name is NOT collision-checked and Logger
# auto-save APPENDS, so a name clash would double the CSV rows.
#
#   GENERATIONS=200 sbatch run_learning_condition_roaming_array.slurm   # override
#   ROAMING_PREDATORS=45 PREDATOR_DRAIN=3 sbatch ...                    # override env
# ============================================================================
#SBATCH --job-name="learning_condition_roaming"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-4
#SBATCH --output=logs/learning_cond_roaming_%A_%a.out
#SBATCH --error=logs/learning_cond_roaming_%A_%a.err
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

# ── Grid ────────────────────────────────────────────────────────────────────
# Seed is the only factor. If you change the seed count, also update --array
# above to match: N_SEEDS - 1  (5 seeds -> 0-4).
SEEDS=(1 42 999 123 456)   # 3 existing maps + 2 new maps (auto-generated)

# ── Fixed factors (identical for every task AND every condition in the trio) ─
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-0.02}"
# Roaming-predator environment: 60 predators, drain 5, patrol predators off.
ROAMING_PREDATORS="${ROAMING_PREDATORS:-60}"
PREDATOR_DRAIN="${PREDATOR_DRAIN:-5}"

# ── Decompose the array index into (seed) ───────────────────────────────────
MAP_SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="roam60d5_g1k_learning_noeps_lr${LEARNING_RATE}_seed${MAP_SEED}"

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

echo "[array] task=$SLURM_ARRAY_TASK_ID  LEARNING (GA+RL, no-epsilon)  lr=$LEARNING_RATE  seed=$MAP_SEED  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
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
