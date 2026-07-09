#!/bin/bash
# ============================================================================
# run_predator_sweep_array.slurm — roaming-predator DIFFICULTY sweep as a
# single SLURM job array.
#
# Goal: establish under what predator settings the dynamics become too hard
# (population collapses) or too easy (no selection pressure). Run for the
# EVOLUTION condition first (pure natural selection = the difficulty baseline);
# later re-run the SAME grid for learning / pure_rl to compare, once the
# learning rate has been chosen from the LR sweep.
#
# Grid (per condition): 4 roaming counts x 4 drain amounts x 3 seeds = 48 tasks
#     roaming predators : 15, 30, 45, 60
#     drain amount      : 1, 3, 5, 10
#     seeds             : 1, 42, 999   (cached map pools already exist)
# Roaming predators only — patrol predators are disabled (--predators-per-patch 0)
# so the roaming count is the sole predator-density knob.
#
#   sbatch run_predator_sweep_array.slurm                    # evolution (default)
#   CONDITION=learning LEARNING_RATE=0.001 sbatch run_predator_sweep_array.slurm
#   CONDITION=pure_rl  LEARNING_RATE=0.001 sbatch run_predator_sweep_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Each task writes CSVs + params.txt to
#   logs/<cond>/<mode>/auto-run/<CONDITION>_roam<N>_drain<D>_seed<S>/
# ============================================================================
#SBATCH --job-name="pred_sweep"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-47
#SBATCH --output=logs/pred_sweep_%A_%a.out
#SBATCH --error=logs/pred_sweep_%A_%a.err
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
ROAMING_COUNTS=(15 30 45 60)      # roaming predator population
DRAIN_AMOUNTS=(1 3 5 10)          # energy drained per predator contact
SEEDS=(1 42 999)                  # 3 maps; cached pools already exist

# ── Fixed factors (identical for every task) ────────────────────────────────
# 500x500 matches the LR sweep grid (clean comparison) and keeps the predator
# density high enough to produce a real difficulty gradient — 15-60 predators
# on a larger map are too sparse to threaten (60 @ 1000x1000 == 15 @ 500x500
# density). Overridable if you want to probe a different size.
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-1000}"

# Which experimental condition to run. Default evolution (difficulty baseline).
# Re-submit with CONDITION=learning / pure_rl to run the other two later.
CONDITION="${CONDITION:-evolution}"
# Only used by learning / pure_rl (evolution ignores it). Set once the LR sweep
# has picked a winner.
LEARNING_RATE="${LEARNING_RATE:-0.001}"

# Map the experimental condition onto headless.js --condition / --mode.
case "$CONDITION" in
    evolution) COND_FLAG=evolution; MODE_FLAG=standard ;;
    learning)  COND_FLAG=learning;  MODE_FLAG=standard ;;
    pure_rl)   COND_FLAG=learning;  MODE_FLAG=pure_rl  ;;
    *) echo "ERROR: CONDITION must be evolution|learning|pure_rl (got '$CONDITION')"; exit 1 ;;
esac

# ── Decompose the array index into (count, drain, seed) ─────────────────────
# Layout: count-major, then drain, then seed (seed varies fastest).
#   idx = count_idx*12 + drain_idx*3 + seed_idx      for idx in 0..47
N_SEEDS=${#SEEDS[@]}          # 3
N_DRAINS=${#DRAIN_AMOUNTS[@]} # 4
SEED_IDX=$((  SLURM_ARRAY_TASK_ID % N_SEEDS ))
DRAIN_IDX=$(( (SLURM_ARRAY_TASK_ID / N_SEEDS) % N_DRAINS ))
COUNT_IDX=$(( SLURM_ARRAY_TASK_ID / (N_SEEDS * N_DRAINS) ))

ROAMING_PREDATORS="${ROAMING_COUNTS[$COUNT_IDX]}"
PREDATOR_DRAIN="${DRAIN_AMOUNTS[$DRAIN_IDX]}"
MAP_SEED="${SEEDS[$SEED_IDX]}"
RUN_NAME="${CONDITION}_roam${ROAMING_PREDATORS}_drain${PREDATOR_DRAIN}_seed${MAP_SEED}"

# ── Run ─────────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR" logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "[array] task=$SLURM_ARRAY_TASK_ID  cond=$CONDITION (--condition $COND_FLAG --mode $MODE_FLAG)"
echo "[array]   roaming=$ROAMING_PREDATORS  drain=$PREDATOR_DRAIN  seed=$MAP_SEED  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"

"$NODE_BIN" src/headless.js \
    --map-seed "$MAP_SEED" \
    --condition "$COND_FLAG" \
    --mode "$MODE_FLAG" \
    --generations "$GENERATIONS" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --run-name "$RUN_NAME" \
    --log-dir "$LOG_DIR" \
    --learning-rate "$LEARNING_RATE" \
    --predators-per-patch 0 \
    --roaming-predators "$ROAMING_PREDATORS" \
    --predator-drain "$PREDATOR_DRAIN"

echo "[array] task=$SLURM_ARRAY_TASK_ID done (exit $?)"
