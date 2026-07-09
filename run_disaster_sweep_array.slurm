#!/bin/bash
# ============================================================================
# run_disaster_sweep_array.slurm — natural-disaster CULLING-FRACTION sweep as a
# single SLURM job array, for the EVOLUTION condition.
#
# Goal: find at what point randomly culling the gene pool actually makes a
# difference to the population/genome dynamics. The disaster removes a FIXED
# fraction of the selection pool (fitness-blind) on the generations it strikes;
# this sweep steps that fraction from 20% up to 60% while holding everything
# else constant.
#
# Reproducibility / graph rationale:
#   - Every task shares ONE disaster seed (DISASTER_SEED). Disaster TIMING (the
#     per-gen Bernoulli draw) and the shuffle draws come from that dedicated
#     PRNG and are independent of population, so all fractions strike on the
#     SAME generations. Any population dip therefore lands at the same x-position
#     across fractions — averaging across seeds gives a clean, aligned graph.
#   - Same 3 map seeds (1, 42, 999) as every other sweep, so results are
#     directly comparable and cached map pools already exist.
#   - Cooldown of 100 generations: after a strike at gen G, no further strike
#     until gen >= G+100, letting the population/fitness recover between events.
#
#     sbatch run_disaster_sweep_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Grid: 5 culling fractions x 3 seeds = 15 tasks (indices 0-14).
#     culling fraction : 0.2, 0.3, 0.4, 0.5, 0.6
#     seeds            : 1, 42, 999
# Evolution condition (pure natural selection), 1000x1000 grid, predators
# disabled so the disaster is the only extrinsic pressure being probed.
#
# Each task writes CSVs + params.txt to
#   logs/evolution/standard/auto-run/evo_cull<F>_seed<S>/
#
#   GENERATIONS=2000 sbatch run_disaster_sweep_array.slurm   # override length
# ============================================================================
#SBATCH --job-name="single_disaster_sweep"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-14
#SBATCH --output=logs/disaster_sweep_%A_%a.out
#SBATCH --error=logs/disaster_sweep_%A_%a.err
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
# If you change the number of fractions, also update --array above to match:
# N_FRACTIONS*N_SEEDS - 1  (5 fractions x 3 seeds -> 0-14).
CULL_FRACTIONS=(0.2 0.3 0.4 0.5 0.6)  # fraction of the selection pool culled per strike
SEEDS=(1 42 999)                      # SAME 3 maps as every other sweep

# ── Fixed factors (identical for every task) ────────────────────────────────
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-1000}"

# Disaster controls shared by every task.
#   DISASTER_SEED : ONE seed for all runs so strikes are aligned across fractions.
#   DISASTER_PROB : per-generation chance a strike fires (headless default 0.1).
#   DISASTER_COOLDOWN : min generations between strikes.
DISASTER_SEED="${DISASTER_SEED:-777}"
DISASTER_PROB="${DISASTER_PROB:-0.1}"
DISASTER_COOLDOWN="${DISASTER_COOLDOWN:-100}"

# ── Decompose the array index into (fraction, seed) ─────────────────────────
# Task IDs 0..14 laid out as fraction-major: 0-2 = frac[0] x seeds,
# 3-5 = frac[1] x seeds, ... 12-14 = frac[4] x seeds (seed varies fastest).
N_SEEDS=${#SEEDS[@]}
FRAC_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
CULL_FRACTION="${CULL_FRACTIONS[$FRAC_IDX]}"
MAP_SEED="${SEEDS[$SEED_IDX]}"
RUN_NAME="evo_cull${CULL_FRACTION}_seed${MAP_SEED}"

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

echo "[array] task=$SLURM_ARRAY_TASK_ID  evolution  cull=$CULL_FRACTION  seed=$MAP_SEED  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
echo "[array]   disaster: prob=$DISASTER_PROB  cooldown=$DISASTER_COOLDOWN  disaster-seed=$DISASTER_SEED (shared)"

"$NODE_BIN" src/headless.js \
    --map-seed "$MAP_SEED" \
    --condition evolution \
    --mode standard \
    --generations "$GENERATIONS" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --run-name "$RUN_NAME" \
    --log-dir "$LOG_DIR" \
    --disaster \
    --disaster-prob "$DISASTER_PROB" \
    --disaster-fraction "$CULL_FRACTION" \
    --disaster-cooldown "$DISASTER_COOLDOWN" \
    --disaster-seed "$DISASTER_SEED" \
    --predators-per-patch 0 \
    --roaming-predators 0

echo "[array] task=$SLURM_ARRAY_TASK_ID done (exit $?)"
