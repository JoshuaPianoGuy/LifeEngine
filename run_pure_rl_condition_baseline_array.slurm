#!/bin/bash
# ============================================================================
# run_pure_rl_condition_baseline_array.slurm — the PURE-RL condition
# (reinforcement learning ONLY, NO genetic algorithm; --mode pure_rl, NO epsilon,
# learning rate 0.01) in the PREDATOR-FREE BASELINE environment, as a single
# SLURM job array over 5 seeds.
#
# This is one of a matched trio of "baseline" (no-predator) comparison runs, the
# predator-free counterpart to the roaming-predator trio. They share IDENTICAL
# run details (500x500 grid, 1000 generations, NO predators, 5 seeds) and differ
# ONLY in the learning/evolution mechanism:
#     run_learning_condition_baseline_array.slurm   (GA + RL)
#     run_evolution_condition_baseline_array.slurm  (GA only, no RL)
#     run_pure_rl_condition_baseline_array.slurm    <- THIS FILE (RL only)
# --mode pure_rl swaps the GAManager for PureRLManager: agents learn by REINFORCE
# within their lifetime, but there is NO between-generation genetic
# mutation/selection. The RL settings (--no-epsilon, --learning-rate 0.01) are
# IDENTICAL to the baseline learning condition, so the only difference between
# the two is whether the GA runs — isolating the GA's contribution. LR is 0.01
# here (baseline-environment optimum); the roaming trio uses 0.02.
#
#     sbatch run_pure_rl_condition_baseline_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Grid: 5 seeds = 5 tasks (indices 0-4).
#     seeds : 1, 42, 999   (the 3 existing maps, reused)
#           + 123, 456     (2 NEW maps — see MAP-SEED note below)
# 500x500 grid, learning condition in pure_rl mode, NO predators (roaming + patrol off).
#
# MAP-SEED note (new worlds are generated + stored automatically)
# ---------------------------------------------------------------
# headless.js builds each world deterministically from --map-seed. On first use
# of a seed it generates the map pool and caches it atomically to
# src/maps/map_pool_seed<SEED>.json. So seeds 123 and 456 need no manual pre-
# generation — the first task to use each one creates and stores it, and every
# later run reuses the SAME cached world, keeping all comparisons on identical
# terrain.
#
# Each task writes CSVs + params.txt to
#   logs/learning/pure_rl/auto-run/baseline_g1k_purerl_noeps_lr0.01_seed<SEED>/
# (pure_rl is a distinct MODE sub-tree under logs/learning/, so there is no
# clash with the learning condition's logs/learning/standard/ runs.)
#
#   GENERATIONS=200 sbatch run_pure_rl_condition_baseline_array.slurm    # override
#   LEARNING_RATE=0.02 sbatch ...                                        # override env
# ============================================================================
#SBATCH --job-name="pure_rl_condition_baseline"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-4
#SBATCH --output=logs/pure_rl_cond_baseline_%A_%a.out
#SBATCH --error=logs/pure_rl_cond_baseline_%A_%a.err
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
LEARNING_RATE="${LEARNING_RATE:-0.01}"   # baseline-environment optimum (roaming uses 0.02)

# ── Decompose the array index into (seed) ───────────────────────────────────
MAP_SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="baseline_g1k_purerl_noeps_lr${LEARNING_RATE}_seed${MAP_SEED}"

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

echo "[array] task=$SLURM_ARRAY_TASK_ID  PURE-RL (RL only, no GA, no-epsilon)  lr=$LEARNING_RATE  seed=$MAP_SEED  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
echo "[array]   predators: NONE (baseline environment — roaming=0, patrol=0)"

# --condition learning enables RL; --mode pure_rl removes the GA (PureRLManager).
# Predator-free baseline: roaming + patrol both disabled.
"$NODE_BIN" src/headless.js \
    --map-seed "$MAP_SEED" \
    --condition learning \
    --mode pure_rl \
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
