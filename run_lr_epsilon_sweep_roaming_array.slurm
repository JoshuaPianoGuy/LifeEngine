#!/bin/bash
# ============================================================================
# run_lr_epsilon_sweep_roaming_array.slurm — learning-rate x epsilon sweep in the
# ROAMING-PREDATOR environment, as a single SLURM job array.
#
# Same design as run_epsilon_sweep_array.slurm (epsilon START x DECAY SHAPE), but
# ALSO sweeping the in-lifetime LEARNING RATE and running with roaming predators
# switched on instead of a predator-free world. Goal: find the best (lr, epsilon)
# setting for the learning condition when organisms face a constant predation
# pressure — directly comparable to the predator-free lr/epsilon sweeps.
#
#     sbatch run_lr_epsilon_sweep_roaming_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# SUBMIT-LIMIT NOTE (why this array is only 48 tasks, not 144)
# ------------------------------------------------------------
# The full grid is 4 lr x 4 eps x 3 shapes x 3 seeds = 144 combinations. A 144-
# element array trips the account's AssocMaxSubmitJobLimit (~120 queued+running
# tasks max — array tasks count individually). So each task here runs ALL 3 SEEDS
# sequentially and the array indexes only (lr x eps x shape) = 48 tasks. Same 144
# runs get produced; 48 <= 120 keeps us under both the submit cap AND the 120-core
# concurrency cap. Wall-time is raised to 45h to cover three back-to-back runs.
# (The `%` array throttle would NOT help here — it caps concurrency, not the
# submitted task count that AssocMaxSubmitJobLimit checks.)
#
# Grid (array): 4 learning rates x 4 epsilon starts x 3 decay shapes
#       = 48 tasks (indices 0-47); each task loops the 3 seeds internally.
#     learning rate : 0.005, 0.01, 0.02, 0.03
#     epsilon start : 0.2, 0.3, 0.5, 0.7
#     decay shape   : sublinear, linear, quadratic
#     seeds         : 1, 42, 999   (SAME 3 maps as every other sweep)
# 500x500 grid, learning/standard condition, 60 roaming predators with drain 5
# (patrol predators disabled) — the roaming count/drain are the predation knobs.
#
# Each run writes CSVs + params.txt to
#   logs/learning/standard/auto-run/roam60d5_lr<LR>_eps<EPS>_<SHAPE>_seed<SEED>/
# The roam60d5_ prefix keeps these folders distinct from the predator-free
# eps<EPS>_<SHAPE>_seed* runs (a forced --run-name is not collision-checked and
# Logger auto-save APPENDS, so a name clash would double the CSV rows).
#
#   GENERATIONS=200 sbatch run_lr_epsilon_sweep_roaming_array.slurm     # override
#   ROAMING_PREDATORS=45 PREDATOR_DRAIN=3 sbatch ...                    # override env
# ============================================================================
#SBATCH --job-name="lr_epsilon_sweep_roaming"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-47
#SBATCH --output=logs/lr_eps_roaming_%A_%a.out
#SBATCH --error=logs/lr_eps_roaming_%A_%a.err
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
# If you change the count of an ARRAY factor (lr/eps/shape), also update --array
# above to match: N_LR*N_EPS*N_SHAPES - 1  (4 x 4 x 3 -> 0-47). Seeds are looped
# INSIDE each task, so changing the seed count does NOT change --array.
LEARNING_RATES=(0.005 0.01 0.02 0.03)          # in-lifetime learning rate
EPSILON_STARTS=(0.2 0.3 0.5 0.7)               # exploration epsilon at birth
DECAY_SHAPES=(sublinear linear quadratic)      # epsilon decay curve over life
SEEDS=(1 42 999)                               # SAME 3 maps as every other sweep (looped per task)

# ── Fixed factors (identical for every task) ────────────────────────────────
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-500}"
# Roaming-predator environment: 60 predators, drain 5, patrol predators off.
ROAMING_PREDATORS="${ROAMING_PREDATORS:-60}"
PREDATOR_DRAIN="${PREDATOR_DRAIN:-5}"

# ── Decompose the array index into (lr, epsilon_start, shape) ───────────────
# Layout: lr-major, then epsilon, then shape (shape varies fastest).
#   idx = lr_idx*(N_EPS*N_SHAPES) + eps_idx*N_SHAPES + shape_idx   for idx in 0..47
N_SHAPES=${#DECAY_SHAPES[@]}  # 3
N_EPS=${#EPSILON_STARTS[@]}   # 4
SHAPE_IDX=$(( SLURM_ARRAY_TASK_ID % N_SHAPES ))
EPS_IDX=$((   (SLURM_ARRAY_TASK_ID / N_SHAPES) % N_EPS ))
LR_IDX=$((    SLURM_ARRAY_TASK_ID / (N_SHAPES * N_EPS) ))

LEARNING_RATE="${LEARNING_RATES[$LR_IDX]}"
EPSILON_START="${EPSILON_STARTS[$EPS_IDX]}"
DECAY_SHAPE="${DECAY_SHAPES[$SHAPE_IDX]}"

# ── Run ─────────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR" logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

echo "[array] task=$SLURM_ARRAY_TASK_ID  learning  lr=$LEARNING_RATE  eps_start=$EPSILON_START  shape=$DECAY_SHAPE  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
echo "[array]   predators: roaming=$ROAMING_PREDATORS  drain=$PREDATOR_DRAIN  patrol=0"
echo "[array]   seeds (run sequentially in this task): ${SEEDS[*]}"

# Each task runs all 3 seeds back-to-back. A single seed failing does NOT abort
# the task — the others still run — but we track it so the task exits non-zero.
task_rc=0
for MAP_SEED in "${SEEDS[@]}"; do
    RUN_NAME="roam60d5_lr${LEARNING_RATE}_eps${EPSILON_START}_${DECAY_SHAPE}_seed${MAP_SEED}"
    # headless.js derives a unique run folder from --run-name (we pass one), but
    # keep PBS_JOBID mapped (seed-suffixed) for any other unique-naming path.
    export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_s${MAP_SEED}"
    echo "[array]   -> seed=$MAP_SEED  run=$RUN_NAME"

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
        --epsilon-start "$EPSILON_START" \
        --epsilon-decay-shape "$DECAY_SHAPE" \
        --predators-per-patch 0 \
        --roaming-predators "$ROAMING_PREDATORS" \
        --predator-drain "$PREDATOR_DRAIN"
    rc=$?
    echo "[array]   -> seed=$MAP_SEED done (exit $rc)"
    [ "$rc" -ne 0 ] && task_rc=$rc
done

echo "[array] task=$SLURM_ARRAY_TASK_ID all seeds done (exit $task_rc)"
exit "$task_rc"
