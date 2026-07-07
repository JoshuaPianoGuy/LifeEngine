#!/bin/bash
# ============================================================================
# run_epsilon_sweep_array.slurm — exploration sweep over the epsilon START value
# and the epsilon DECAY SHAPE, as a single SLURM job array.
#
# Goal: see how much early exploration (epsilon_start) and the SHAPE of its
# decay over an organism's lifetime affect in-lifetime learning and fitness.
# epsilon decays from epsilon_start -> epsilon_end (0.05, fixed) as
# lifetime_frac goes 0 -> 1, with the curve set by --epsilon-decay-shape:
#     sublinear (sqrt) : drops fast early, then flattens   (explore-early)
#     linear           : constant-rate decay
#     quadratic (sq)   : holds high early, drops sharply late (explore-late)
#
#     sbatch run_epsilon_sweep_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Grid: 4 epsilon starts x 3 decay shapes x 3 seeds = 36 tasks (indices 0-35).
#     epsilon start : 0.2, 0.3, 0.5, 0.7
#     decay shape   : sublinear, linear, quadratic
#     seeds         : 1, 42, 999   (SAME 3 maps as the LR / predator sweeps)
# Same 500x500 grid, learning/standard condition, predators disabled — so these
# runs are directly comparable to the LR sweep's.
#
# Each task writes CSVs + params.txt to
#   logs/learning/standard/auto-run/eps<EPS>_<SHAPE>_seed<SEED>/
#
#   GENERATIONS=200 sbatch run_epsilon_sweep_array.slurm       # override length
#   LEARNING_RATE=0.03 sbatch run_epsilon_sweep_array.slurm    # set LR winner
# ============================================================================
#SBATCH --job-name="epsilon_sweep"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-35
#SBATCH --output=logs/eps_sweep_%A_%a.out
#SBATCH --error=logs/eps_sweep_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/LifeEngine}"
# Base output dir. Point this at SCRATCH so runs do NOT fill the home quota
# under $PROJECT_DIR/logs. EDIT to match your cluster's scratch path.
LOG_DIR="${LOG_DIR:-/scratch/$USER/LifeEngine/logs}"

# ── Sweep grid ──────────────────────────────────────────────────────────────
# If you change the count of either factor, also update --array above to match:
# N_EPS*N_SHAPES*N_SEEDS - 1  (4 x 3 x 3 -> 0-35).
EPSILON_STARTS=(0.2 0.3 0.5 0.7)               # exploration epsilon at birth
DECAY_SHAPES=(sublinear linear quadratic)      # epsilon decay curve over life
SEEDS=(1 42 999)                               # SAME 3 maps as the LR sweep

# ── Fixed factors (identical for every task, and to the LR sweep) ────────────
WIDTH=500
HEIGHT=500
GENERATIONS="${GENERATIONS:-500}"
# In-lifetime learning rate. Set to the winner from the LR sweep; overridable.
LEARNING_RATE="${LEARNING_RATE:-0.01}"

# ── Decompose the array index into (epsilon_start, shape, seed) ─────────────
# Layout: epsilon-major, then shape, then seed (seed varies fastest).
#   idx = eps_idx*(N_SHAPES*N_SEEDS) + shape_idx*N_SEEDS + seed_idx   for 0..35
N_SEEDS=${#SEEDS[@]}          # 3
N_SHAPES=${#DECAY_SHAPES[@]}  # 3
SEED_IDX=$((  SLURM_ARRAY_TASK_ID % N_SEEDS ))
SHAPE_IDX=$(( (SLURM_ARRAY_TASK_ID / N_SEEDS) % N_SHAPES ))
EPS_IDX=$((   SLURM_ARRAY_TASK_ID / (N_SEEDS * N_SHAPES) ))

EPSILON_START="${EPSILON_STARTS[$EPS_IDX]}"
DECAY_SHAPE="${DECAY_SHAPES[$SHAPE_IDX]}"
MAP_SEED="${SEEDS[$SEED_IDX]}"
RUN_NAME="eps${EPSILON_START}_${DECAY_SHAPE}_seed${MAP_SEED}"

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

echo "[array] task=$SLURM_ARRAY_TASK_ID  eps_start=$EPSILON_START  shape=$DECAY_SHAPE  seed=$MAP_SEED  lr=$LEARNING_RATE  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"

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
    --roaming-predators 0

echo "[array] task=$SLURM_ARRAY_TASK_ID done (exit $?)"
