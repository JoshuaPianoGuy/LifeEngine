#!/bin/bash
# ============================================================================
# run_explore_bonus_sweep_roaming_array.slurm — EXPLORATION-BONUS sweep in the
# HARD (roaming-predator) environment.
#
# Same sweep as run_explore_bonus_sweep_array.slurm (the predator-free baseline
# version) but run in the HARD environment: 60 roaming predators, drain 5,
# patrol off. Measures how the per-tick REINFORCE exploration bonus trades off
# against predation — its effect on fitness and on cave entries during the day
# vs at night when roaming out to explore also means risking predator contact.
#
#     EXPLORE_BONUS values : 0, 0.02, 0.04, 0.08, 0.12, 0.15   (0.15 = old default)
#     seeds                : 1, 42, 999   (the 3 existing maps, reused)
#                          + 123, 456     (the same 2 new maps as the trio)
#     6 bonuses x 5 seeds  = 30 tasks (array indices 0-29).
#
# The bonus is wired through --explore-bonus (ExperimentParams.explore_bonus,
# read by AdvancedOrganism at module load). 500x500 grid, 1000 generations,
# 60 roaming predators + drain 5 (patrol off), no-epsilon (on-policy REINFORCE),
# lr 0.02 (the roaming-environment optimum from the LR sweep).
#
# APPLIES TO BOTH conditions — the ONLY difference between the "learning" and
# "pure RL" runs is the presence of the GA, selected here via the MODE env var:
#     sbatch run_explore_bonus_sweep_roaming_array.slurm               # learning (GA+RL)
#     MODE=pure_rl sbatch run_explore_bonus_sweep_roaming_array.slurm  # pure RL (no GA)
# Both pass --condition learning (which enables RL); MODE just toggles the GA
# (--mode standard = GAManager, --mode pure_rl = PureRLManager). Run-folder
# names embed the condition label so the two sets never collide.
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# MAP-SEED note (new worlds are generated + stored automatically)
# ---------------------------------------------------------------
# headless.js builds each world deterministically from --map-seed and caches it
# to src/maps/map_pool_seed<SEED>.json on first use, so seeds 123/456 need no
# manual pre-generation and every condition reuses identical terrain.
#
# Each task writes CSVs + params.txt to
#   logs/learning/<mode>/auto-run/roam60d5_g1k_<label>_noeps_lr0.02_eb<BONUS>_seed<SEED>/
# The eb<BONUS> tag keeps every sweep point in its own folder (a forced
# --run-name is NOT collision-checked and Logger auto-save APPENDS, so a clash
# would double rows). The roam60d5_ prefix keeps these distinct from the
# baseline_ (predator-free) sweep folders in the same dir.
#
#   GENERATIONS=200 sbatch run_explore_bonus_sweep_roaming_array.slurm    # override
#   LEARNING_RATE=0.03 sbatch ...                                         # override env
#   MODE=pure_rl sbatch ...                                               # pure-RL sweep
# ============================================================================
#SBATCH --job-name="explore_bonus_sweep_roaming"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-29
#SBATCH --output=logs/explore_bonus_sweep_roaming_%A_%a.out
#SBATCH --error=logs/explore_bonus_sweep_roaming_%A_%a.err
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

# ── Grid (2 factors: exploration bonus x seed) ──────────────────────────────
# array index = BONUS_IDX * N_SEEDS + SEED_IDX. If you change either list,
# update --array above to  (N_BONUSES * N_SEEDS - 1).
BONUSES=(0 0.02 0.04 0.08 0.12 0.15)   # exploration-bonus sweep points
SEEDS=(1 42 999 123 456)               # 3 existing maps + 2 new maps (auto-generated)
N_SEEDS=${#SEEDS[@]}

# ── Fixed factors (identical for every task) ────────────────────────────────
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-0.02}"   # roaming-environment optimum
# Roaming-predator (hard) environment: 60 predators, drain 5, patrol predators off.
ROAMING_PREDATORS="${ROAMING_PREDATORS:-60}"
PREDATOR_DRAIN="${PREDATOR_DRAIN:-5}"

# ── Condition: MODE toggles the GA (only difference learning vs pure RL) ─────
#   standard -> learning (GA + REINFORCE) ; pure_rl -> pure RL (REINFORCE only)
MODE="${MODE:-standard}"
case "$MODE" in
    standard) LABEL="learning" ;;
    pure_rl)  LABEL="purerl" ;;
    *) echo "ERROR: MODE must be 'standard' (learning) or 'pure_rl'."; exit 1 ;;
esac

# ── Decompose the array index into (bonus, seed) ────────────────────────────
BONUS_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$((  SLURM_ARRAY_TASK_ID % N_SEEDS ))
EXPLORE_BONUS="${BONUSES[$BONUS_IDX]}"
MAP_SEED="${SEEDS[$SEED_IDX]}"
RUN_NAME="roam60d5_g1k_${LABEL}_noeps_lr${LEARNING_RATE}_eb${EXPLORE_BONUS}_seed${MAP_SEED}"

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

echo "[array] task=$SLURM_ARRAY_TASK_ID  ${LABEL} (mode=$MODE, no-epsilon)  explore_bonus=$EXPLORE_BONUS  lr=$LEARNING_RATE  seed=$MAP_SEED  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
echo "[array]   predators: roaming=$ROAMING_PREDATORS  drain=$PREDATOR_DRAIN  patrol=0"

# --condition learning enables RL; --mode selects GA (standard) vs no-GA (pure_rl).
# Hard environment: 60 roaming predators + drain 5, patrol off.
"$NODE_BIN" src/headless.js \
    --map-seed "$MAP_SEED" \
    --condition learning \
    --mode "$MODE" \
    --generations "$GENERATIONS" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --run-name "$RUN_NAME" \
    --log-dir "$LOG_DIR" \
    --learning-rate "$LEARNING_RATE" \
    --explore-bonus "$EXPLORE_BONUS" \
    --no-epsilon \
    --predators-per-patch 0 \
    --roaming-predators "$ROAMING_PREDATORS" \
    --predator-drain "$PREDATOR_DRAIN"

echo "[array] task=$SLURM_ARRAY_TASK_ID done (exit $?)"
