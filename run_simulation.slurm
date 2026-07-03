#!/bin/bash
# ============================================================================
# run_simulation.slurm — SLURM job script for the learning-vs-evolution
# experiment on the UCT HPC cluster ("hex" / hpc.uct.ac.za, see "UCT_HEX_GUIDE.md").
#
# This is the SLURM counterpart of run_simulation.pbs (used on the national
# CHPC, which runs PBS). Same headless runner, same flags, different scheduler.
#
# Two ways to set parameters:
#   1. Edit the defaults below, then:           sbatch run_simulation.slurm
#   2. Override per job at submit time (ideal for sweeps / arrays):
#        sbatch --export=MAP_SEED=43,CONDITION=evolution run_simulation.slurm
#        sbatch --export=MAP_SEED=7,GENERATIONS=100,LEARNING_RATE=0.01,HIDDEN_SIZE=64 run_simulation.slurm
#   Every parameter below uses ${VAR:-default}, so an exported env var wins,
#   otherwise the default applies.
#
# Each run writes CSVs + a params.txt to logs/<condition>/<mode>/auto-run/run_N/.
# ============================================================================

#SBATCH --job-name="life_engine_lr_tuning"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/life_engine_%j.out
#SBATCH --error=logs/life_engine_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za
#

# Path to the Node 16 binary (run `nvm use 16 && which node` to find yours).
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
# Absolute path to the project directory on the cluster.
PROJECT_DIR="${PROJECT_DIR:-$HOME/LifeEngine}"

# ── Run control (override with --export) ────────────────────────────────────
MAP_SEED="${MAP_SEED:-42}"          # terrain seed (deterministic world)
CONDITION="${CONDITION:-learning}"  # learning | evolution
MODE="${MODE:-standard}"            # standard | frozen_pg | pure_rl
GENERATIONS="${GENERATIONS:-50}"    # number of generations to run
WIDTH="${WIDTH:-400}"               # grid columns
HEIGHT="${HEIGHT:-300}"             # grid rows
LOG_EVERY="${LOG_EVERY:-10000}"     # progress print interval (ticks)
RUN_NAME="${RUN_NAME:-}"            # output folder name (empty = seed<N>_job<id>)

# ── Tunable hyperparameters (override with --export) ────────────────────────
LEARNING_RATE="${LEARNING_RATE:-0.02}"
EPSILON_START="${EPSILON_START:-0.5}"
EPSILON_END="${EPSILON_END:-0.05}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
POPULATION_SIZE="${POPULATION_SIZE:-100}"
MUT_PROB="${MUT_PROB:-0.03}"
MUT_SIGMA="${MUT_SIGMA:-0.1}"
DISASTER="${DISASTER:-0}"                 # 1 = enable natural-disaster culls
DISASTER_PROB="${DISASTER_PROB:-0.1}"     # per-gen chance a disaster hits
DISASTER_FRACTION="${DISASTER_FRACTION:-0.2}"  # fixed fraction culled per event
DISASTER_COOLDOWN="${DISASTER_COOLDOWN:-0}"    # min generations between disasters (0=none)
DISASTER_SEED="${DISASTER_SEED:-0}"       # disaster PRNG seed (0 = unseeded)
PREDATOR_DRAIN="${PREDATOR_DRAIN:-1.0}"
PREDATORS_PER_PATCH="${PREDATORS_PER_PATCH:-2}"
ROAMING_PREDATORS="${ROAMING_PREDATORS:-0}"

# ── Run ─────────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

# headless.js only recognises PBS_JOBID for unique run-folder naming;
# map the SLURM job id onto it so runs on this cluster stay unique too.
export PBS_JOBID="${SLURM_JOB_ID}"

echo "[slurm] node=$NODE_BIN"
echo "[slurm] map_seed=$MAP_SEED condition=$CONDITION mode=$MODE generations=$GENERATIONS grid=${WIDTH}x${HEIGHT}"

"$NODE_BIN" src/headless.js \
    --map-seed "$MAP_SEED" \
    --condition "$CONDITION" \
    --mode "$MODE" \
    --generations "$GENERATIONS" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --log-every "$LOG_EVERY" \
    $( [ -n "$RUN_NAME" ] && printf -- "--run-name %s" "$RUN_NAME" ) \
    --learning-rate "$LEARNING_RATE" \
    --epsilon-start "$EPSILON_START" \
    --epsilon-end "$EPSILON_END" \
    --hidden-size "$HIDDEN_SIZE" \
    --population-size "$POPULATION_SIZE" \
    --mut-prob "$MUT_PROB" \
    --mut-sigma "$MUT_SIGMA" \
    $( [ "$DISASTER" = "1" ] && printf -- "--disaster" ) \
    --disaster-prob "$DISASTER_PROB" \
    --disaster-fraction "$DISASTER_FRACTION" \
    --disaster-cooldown "$DISASTER_COOLDOWN" \
    --disaster-seed "$DISASTER_SEED" \
    --predator-drain "$PREDATOR_DRAIN" \
    --predators-per-patch "$PREDATORS_PER_PATCH" \
    --roaming-predators "$ROAMING_PREDATORS"

echo "[slurm] done (exit $?)"
