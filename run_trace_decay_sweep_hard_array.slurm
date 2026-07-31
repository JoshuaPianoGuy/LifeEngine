#!/bin/bash
# ============================================================================
# run_trace_decay_sweep_hard_array.slurm — eligibility-trace decay (gamma)
# sweep in the HARD (roaming-predator) environment, LEARNING condition.
# 3-REPLICATES-PER-SEED variant.
#
# Goal: completeness — how far back in a lifetime credit is spread. NNBrain
# accumulates, each tick,
#     trace <- gamma * trace + grad log pi(a)
# so the trace's effective credit horizon is ~1/(1-gamma) ticks:
#     0.90 -> ~10 ticks   (credit stays local to the last few actions)
#     0.95 -> ~20 ticks
#     0.98 -> ~50 ticks
#     0.99 -> ~100 ticks  (long-delayed rewards reach far-back actions; noisier)
# 0.90 is the value every previous run used (the old hard-coded constant), so
# that arm doubles as the baseline this sweep is measured against.
#
# The hard environment is where a longer trace has the clearest story to tell:
# predator damage is a delayed, sparse penalty, so credit for the actions that
# walked an agent into a predator may sit tens of ticks back.
#
# Environment / operating point (matches run_learning_condition_roaming_array):
#   60 roaming predators, drain 5, patrol predators off, no-epsilon (pure
#   on-policy REINFORCE), LR 0.02 — the roaming-environment optimum from the LR
#   sweep. The BASELINE counterpart is run_trace_decay_sweep_baseline_array.slurm
#   (predator-free, LR 0.01); the two differ ONLY in those environment settings,
#   so the pair answers "does the best gamma depend on the environment?".
#
# WHY REPLICATES: only the map TERRAIN is seeded (--map-seed). GA mutation, RL
# action sampling and predator wandering are UNSEEDED, so the 3 replicates of a
# (gamma, seed) cell share identical terrain but differ purely by algorithmic /
# predator chance — their spread is the run-to-run noise, which is what you need
# to tell a real gamma effect apart from stochasticity. Expect MORE spread here
# than in the baseline environment: predators add a second source of chance.
# Aggregate across replicates BEFORE comparing gammas.
#
# Grid: 4 gammas x 3 seeds x 3 replicates = 36 runs.
#     trace decay : 0.90, 0.95, 0.98, 0.99
#     seeds       : 1, 42, 999   (SAME 3 maps as the LR / epsilon sweeps)
#     replicates  : 1..3
# 500x500 grid, learning/standard condition — directly comparable to the roaming
# LR sweeps and to the roaming condition runs.
#
# CORE PACKING (to stay under the simultaneous-core limit)
# --------------------------------------------------------
# Each array task runs RUNS_PER_JOB runs back-to-back on its single core.
#   RUNS_PER_JOB (env, default 1) -> set --array to  ceil(36 / RUNS_PER_JOB) - 1:
#       RUNS_PER_JOB=1 -> --array=0-35  (36 cores; no packing)   <- CURRENT DEFAULT
#       RUNS_PER_JOB=2 -> --array=0-17  (18 cores)
#       RUNS_PER_JOB=3 -> --array=0-11  (12 cores)
#   If you change RUNS_PER_JOB you MUST update the #SBATCH --array line to match.
#   Runs beyond index 35 are skipped, so an over-long array is harmless.
#
#     sbatch run_trace_decay_sweep_hard_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Each run writes CSVs + params.txt to
#   logs/learning/standard/auto-run/roam60d5_trace<G>_seed<S>_r<N>/
# The roam60d5_ prefix keeps these distinct from the baseline sweep, and the
# _r<N> replicate suffix (deliberately NOT _rep<N>) keeps them out of the
# condition-consistency analysis, which matches runs by params and requires a
# _rep<N> suffix. A forced --run-name is not collision-checked and Logger
# auto-save APPENDS, so a unique name per run is required to avoid doubled rows.
#
#   GENERATIONS=200 sbatch run_trace_decay_sweep_hard_array.slurm   # override length
#   N_REPS=5 sbatch --array=0-59 ...                                # more replicates
#   RUNS_PER_JOB=2 sbatch --array=0-17 ...                          # denser packing
#   ROAMING_PREDATORS=45 PREDATOR_DRAIN=3 sbatch ...                # override env
# ============================================================================
#SBATCH --job-name="trace_sweep_hard"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-35
#SBATCH --output=logs/trace_sweep_hard_%A_%a.out
#SBATCH --error=logs/trace_sweep_hard_%A_%a.err
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

# ── Sweep grid (3 factors: gamma x seed x replicate) ────────────────────────
TRACE_DECAYS=(0.90 0.95 0.98 0.99)   # eligibility-trace decay gamma
SEEDS=(1 42 999)                     # SAME 3 maps as the LR / epsilon sweeps
N_SEEDS=${#SEEDS[@]}
N_REPS="${N_REPS:-3}"                # independent replicate runs per (gamma, seed)
PER_TRACE=$(( N_SEEDS * N_REPS ))    # runs per gamma (9)
TOTAL_RUNS=$(( ${#TRACE_DECAYS[@]} * PER_TRACE ))   # 36

# ── Core packing: how many runs each array task executes back-to-back ────────
# CHANGING THIS REQUIRES updating the #SBATCH --array line (see header table).
RUNS_PER_JOB="${RUNS_PER_JOB:-1}"

# ── Fixed factors (identical for every run) ─────────────────────────────────
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-500}"
# Roaming/hard-environment optimum from the LR sweep (the baseline env uses 0.01).
LEARNING_RATE="${LEARNING_RATE:-0.02}"
# Hard environment: 60 roaming predators, drain 5, patrol predators off.
ROAMING_PREDATORS="${ROAMING_PREDATORS:-60}"
PREDATOR_DRAIN="${PREDATOR_DRAIN:-5}"

# ── Preflight ───────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR" logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

# ── One run, given a global run index ───────────────────────────────────────
# Decomposes the index into (gamma, seed, replicate). Layout is gamma-major,
# then seed, then replicate (replicate varies fastest):
#   g = TRACE_IDX*(N_SEEDS*N_REPS) + SEED_IDX*N_REPS + REP_IDX
run_one() {
    local GIDX="$1"
    local TRACE_IDX=$(( GIDX / PER_TRACE ))
    local REM=$((       GIDX % PER_TRACE ))
    local SEED_IDX=$((  REM / N_REPS ))
    local REP=$((       REM % N_REPS + 1 ))     # replicate number, 1..N_REPS
    local TRACE_DECAY="${TRACE_DECAYS[$TRACE_IDX]}"
    local MAP_SEED="${SEEDS[$SEED_IDX]}"
    local RUN_NAME="roam60d5_trace${TRACE_DECAY}_seed${MAP_SEED}_r${REP}"

    export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${GIDX}"

    echo "[run g$GIDX] LEARNING (GA+RL, no-epsilon)  trace_decay=$TRACE_DECAY  seed=$MAP_SEED  rep=$REP/$N_REPS  lr=$LEARNING_RATE  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
    echo "[run g$GIDX]   hard environment: roaming=$ROAMING_PREDATORS drain=$PREDATOR_DRAIN patrol=0"
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
        --trace-decay "$TRACE_DECAY" \
        --no-epsilon \
        --predators-per-patch 0 \
        --roaming-predators "$ROAMING_PREDATORS" \
        --predator-drain "$PREDATOR_DRAIN"
    local rc=$?
    echo "[run g$GIDX] done (exit $rc): $RUN_NAME"
    return $rc
}

# ── Run this task's slice of RUNS_PER_JOB consecutive global indices ────────
START=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_JOB ))
echo "[array] task=$SLURM_ARRAY_TASK_ID  runs global indices ${START}..$(( START + RUNS_PER_JOB - 1 ))  (RUNS_PER_JOB=$RUNS_PER_JOB, of $TOTAL_RUNS total)  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"

fail=0
for (( k = 0; k < RUNS_PER_JOB; k++ )); do
    GIDX=$(( START + k ))
    if [ "$GIDX" -ge "$TOTAL_RUNS" ]; then
        echo "[array] global index $GIDX >= $TOTAL_RUNS — nothing more to do."
        break
    fi
    # Don't let one failed run abort the rest of the job's packed runs.
    run_one "$GIDX" || { echo "[array] WARNING: run g$GIDX exited non-zero"; fail=1; }
done

echo "[array] task=$SLURM_ARRAY_TASK_ID complete (fail=$fail)"
exit $fail
