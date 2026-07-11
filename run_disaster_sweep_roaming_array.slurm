#!/bin/bash
# ============================================================================
# run_disaster_sweep_roaming_array.slurm — natural-disaster CULLING-FRACTION
# sweep in the ROAMING-PREDATOR environment, as a single SLURM job array, for
# the EVOLUTION condition.
#
# Same design as run_disaster_sweep_array.slurm (a FIXED fraction of the
# selection pool is culled fitness-blind on the generations a strike fires), but
# with roaming predators switched on so the disaster acts ON TOP of a constant
# predation pressure rather than in isolation. Goal: see how randomly culling
# the gene pool interacts with steady predation.
#
# Reproducibility / graph rationale:
#   - Every task shares ONE disaster seed (DISASTER_SEED). Disaster TIMING (the
#     per-gen Bernoulli draw) and the shuffle draws come from that dedicated
#     PRNG, so all fractions strike on ~the SAME generations for clean averaging.
#   - Same 3 map seeds (1, 42, 999) as every other sweep; cached pools exist.
#   - Cooldown of 100 generations between strikes lets the population recover.
#
#     sbatch run_disaster_sweep_roaming_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Grid: 5 culling fractions x 3 seeds = 15 tasks (indices 0-14).
#     culling fraction : 0.2, 0.3, 0.4, 0.5, 0.6
#     seeds            : 1, 42, 999
# Evolution condition, 500x500 grid, 60 roaming predators with drain 5 (patrol
# predators disabled). The disaster and the predators are the two extrinsic
# pressures being probed together.
#
# Each task writes CSVs + params.txt to
#   logs/evolution/standard/auto-run/roam60d5_evo_cull<F>_seed<S>/
# The roam60d5_ prefix keeps these folders distinct from the predator-free
# evo_cull<F>_seed* runs.
#
#   GENERATIONS=2000 sbatch run_disaster_sweep_roaming_array.slurm   # override
#   ROAMING_PREDATORS=45 PREDATOR_DRAIN=3 sbatch ...                 # override env
# ============================================================================
#SBATCH --job-name="single_disaster_sweep_roaming"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-14
#SBATCH --output=logs/disaster_sweep_roaming_%A_%a.out
#SBATCH --error=logs/disaster_sweep_roaming_%A_%a.err
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
# Roaming-predator environment: 60 predators, drain 5, patrol predators off.
ROAMING_PREDATORS="${ROAMING_PREDATORS:-60}"
PREDATOR_DRAIN="${PREDATOR_DRAIN:-5}"

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
RUN_NAME="roam60d5_evo_cull${CULL_FRACTION}_seed${MAP_SEED}"

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
echo "[array]   predators: roaming=$ROAMING_PREDATORS  drain=$PREDATOR_DRAIN  patrol=0"
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
    --roaming-predators "$ROAMING_PREDATORS" \
    --predator-drain "$PREDATOR_DRAIN"

echo "[array] task=$SLURM_ARRAY_TASK_ID done (exit $?)"
