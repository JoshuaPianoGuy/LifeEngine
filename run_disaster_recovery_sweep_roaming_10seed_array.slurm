#!/bin/bash
# ============================================================================
# run_disaster_recovery_sweep_roaming_10seed_array.slurm — GRADUAL-RECOVERY
# disaster variant sweep in the ROAMING-PREDATOR environment, as a single SLURM
# job array, for the EVOLUTION condition. 10-SEED variant.
#
# Identical design to run_disaster_recovery_sweep_roaming_array.slurm: a strike
# at fraction F tapers the cull back to full population over successive
# generations (F, F-rate, F-2*rate, ... 0), all while roaming predators keep up a
# constant predation pressure. The ONLY change is the seed count: 10 map seeds
# per parameter instead of 3, for tighter confidence intervals when averaging.
#
# Reproducibility / graph rationale:
#   - Every task shares ONE disaster seed (DISASTER_SEED). Strike TIMING comes
#     from that dedicated PRNG and is independent of fraction AND recovery rate
#     (no PRNG is consumed during a taper), so runs strike on ~the SAME
#     generations. Dips stay aligned across the sweep for clean averaging.
#   - 10 map seeds: 1, 42, 999 (cached pools already exist) + 7 new ones. The
#     headless runner AUTO-GENERATES maps/map_pool_seed<N>.json deterministically
#     on first use (atomic temp+rename), so the new seeds need no pre-generation;
#     parallel tasks racing on the same new seed are safe.
#   - Cooldown of 100 generations, counted from the taper's LAST generation.
#
#     sbatch run_disaster_recovery_sweep_roaming_10seed_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Grid: 5 fractions x 2 recovery rates x 10 seeds = 100 tasks (indices 0-99).
#     starting fraction : 0.2, 0.3, 0.4, 0.5, 0.6
#     recovery rate     : 0.05, 0.10   (fraction shed per generation)
#     seeds             : 1, 42, 999, 7, 13, 21, 55, 88, 123, 314
# Evolution condition, 500x500 grid, 60 roaming predators with drain 5 (patrol
# predators disabled). The disaster and the predators are the two extrinsic
# pressures being probed together.
#
# Each task writes CSVs + params.txt to
#   logs/evolution/standard/auto-run/roam60d5_10s_evo_cull<F>_rec<R>_seed<S>/
# The roam60d5_10s_ prefix keeps these folders distinct from BOTH the predator-free
# runs and the original 3-seed roaming sweep (roam60d5_evo_cull<F>_rec<R>_seed*).
#
#   GENERATIONS=2000 sbatch run_disaster_recovery_sweep_roaming_10seed_array.slurm  # override
#   ROAMING_PREDATORS=45 PREDATOR_DRAIN=3 sbatch ...                                # override env
# ============================================================================
#SBATCH --job-name="disaster_recovery_sweep_roaming_10seed"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-99
#SBATCH --output=logs/disaster_recovery_sweep_roaming_10seed_%A_%a.out
#SBATCH --error=logs/disaster_recovery_sweep_roaming_10seed_%A_%a.err
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
# If you change the counts, also update --array above to match:
# N_FRACTIONS*N_RECOVERY*N_SEEDS - 1  (5 x 2 x 10 -> 0-99).
CULL_FRACTIONS=(0.2 0.3 0.4 0.5 0.6)          # STARTING fraction culled on the strike generation
RECOVERY_RATES=(0.05 0.10)                    # fraction shed per generation as the cull tapers
SEEDS=(1 42 999 7 13 21 55 88 123 314)        # 3 cached maps + 7 auto-generated

# ── Fixed factors (identical for every task) ────────────────────────────────
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-1000}"
# Roaming-predator environment: 60 predators, drain 5, patrol predators off.
ROAMING_PREDATORS="${ROAMING_PREDATORS:-60}"
PREDATOR_DRAIN="${PREDATOR_DRAIN:-5}"

# Disaster controls shared by every task.
#   DISASTER_SEED : ONE seed for all runs so strikes are aligned across the sweep.
#   DISASTER_PROB : per-generation chance a strike fires (headless default 0.1).
#   DISASTER_COOLDOWN : min generations between strikes (from the taper's last gen).
DISASTER_SEED="${DISASTER_SEED:-777}"
DISASTER_PROB="${DISASTER_PROB:-0.1}"
DISASTER_COOLDOWN="${DISASTER_COOLDOWN:-100}"

# ── Decompose the array index into (fraction, recovery, seed) ───────────────
# Layout: fraction-major, then recovery, then seed (seed varies fastest).
#   idx = frac_idx*(N_REC*N_SEEDS) + rec_idx*N_SEEDS + seed_idx   for idx in 0..99
N_SEEDS=${#SEEDS[@]}              # 10
N_REC=${#RECOVERY_RATES[@]}       # 2
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
REC_IDX=$(( (SLURM_ARRAY_TASK_ID / N_SEEDS) % N_REC ))
FRAC_IDX=$(( SLURM_ARRAY_TASK_ID / (N_SEEDS * N_REC) ))

CULL_FRACTION="${CULL_FRACTIONS[$FRAC_IDX]}"
RECOVERY_RATE="${RECOVERY_RATES[$REC_IDX]}"
MAP_SEED="${SEEDS[$SEED_IDX]}"
RUN_NAME="roam60d5_10s_evo_cull${CULL_FRACTION}_rec${RECOVERY_RATE}_seed${MAP_SEED}"

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

echo "[array] task=$SLURM_ARRAY_TASK_ID  evolution  cull=$CULL_FRACTION  recovery=$RECOVERY_RATE  seed=$MAP_SEED  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
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
    --disaster-recovery-rate "$RECOVERY_RATE" \
    --predators-per-patch 0 \
    --roaming-predators "$ROAMING_PREDATORS" \
    --predator-drain "$PREDATOR_DRAIN"

echo "[array] task=$SLURM_ARRAY_TASK_ID done (exit $?)"
