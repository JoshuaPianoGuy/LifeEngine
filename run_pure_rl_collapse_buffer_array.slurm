#!/bin/bash
# ============================================================================
# run_pure_rl_collapse_buffer_array.slurm — collapse-respawn buffer sweep
# (25 vs 100) for the PURE-RL condition, in BOTH environments.
#
# What the buffer is
# ------------------
# PureRLManager keeps a rolling buffer of active_weight snapshots taken from
# recently dead organisms (PureRLManager.js:_bufferDeadWeights). When the
# population collapses or drops below population_size, the shortfall is
# respawned by sampling that buffer and mutating around the drawn vectors
# (_respawnFromBuffer). With no GA in this condition, that buffer is the ONLY
# mechanism carrying learned strategies across a collapse — it is pure RL's
# entire inheritance channel.
#
# Its size sets a recency/diversity trade-off, and the two arms bracket it:
#     25  (default)  the pool tracks the newest learning closely, but is narrow:
#                    a collapse can refill the whole world from ~25 vectors that
#                    are often near-identical (they died at the same moment, of
#                    the same cause), so the post-collapse population is a tight
#                    cloud and diversity has to be rebuilt by mutation alone.
#                    This is the value every previous pure-RL run used (the old
#                    hard-coded constant), so this arm doubles as the baseline
#                    the sweep is measured against.
#     100            four times the pool. More genotypic spread at respawn, at
#                    the cost of reaching back to staler, less-learned weights.
# The hypothesis worth testing: if pure RL's known collapse-and-restart churn is
# a diversity problem, 100 should reduce the depth/frequency of collapses and
# raise the level it recovers to; if it is a recency problem, 100 should make it
# WORSE by reseeding from weights that have already been superseded.
#
# BOTH arms are run HERE rather than comparing against the existing pure-RL
# condition runs, even though those already sit at 25. Reason: they are 1
# replicate on 5 seeds, this is 5 replicates on 3 seeds, and pure RL's run-to-run
# spread is large enough that a 1-vs-5 comparison would be dominated by it.
# Running both arms in the same job also guarantees they share code version,
# node version and cluster conditions — the buffer becomes the only difference.
# Everything else (500x500, 1000 generations, --no-epsilon, per-environment LR,
# patrol predators off) still matches
#     run_pure_rl_condition_baseline_array.slurm  (baseline, lr 0.01)
#     run_pure_rl_condition_roaming_array.slurm   (hard,     lr 0.02)
# so those runs remain usable as extra seeds for the 25 arm if wanted.
#
# --collapse-buffer is a NEW TUNABLE. Before this sweep the value was a
# hard-coded constant, so it is now recorded in every run's params.txt as
# collapse_buffer_size — including for older-style runs, which report 25.
#
# WHY REPLICATES: only the map TERRAIN is seeded (--map-seed). Founder weights
# come from an unseeded Xavier draw and RL action sampling is unseeded too, so
# the 5 replicates of a (env, seed) cell share identical terrain and differ
# purely by algorithmic chance. Pure RL is the noisiest condition in this
# codebase — collapse timing is close to a coin flip — so the replicate spread
# IS the measurement here. Aggregate across replicates BEFORE comparing to the
# buffer-25 arm.
#
# Grid: 2 buffers x 2 environments x 3 seeds x 5 replicates = 60 runs.
#     buffers      : 25 (default / control), 100
#     environments : baseline (predator-free, lr 0.01)
#                    hard     (60 roaming predators, drain 5, lr 0.02)
#     seeds        : 1, 42, 999   (SAME 3 maps as every other sweep)
#     replicates   : 1..5
# 500x500 grid, 1000 generations, learning condition in pure_rl mode.
#
# CORE PACKING (to stay under the simultaneous-core limit)
# --------------------------------------------------------
# Each array task runs RUNS_PER_JOB runs back-to-back on its single core.
#   RUNS_PER_JOB (env, default 1) -> set --array to  ceil(60 / RUNS_PER_JOB) - 1:
#       RUNS_PER_JOB=1 -> --array=0-59  (60 cores; no packing)   <- CURRENT DEFAULT
#       RUNS_PER_JOB=2 -> --array=0-29  (30 cores)
#       RUNS_PER_JOB=3 -> --array=0-19  (20 cores)
#       RUNS_PER_JOB=4 -> --array=0-14  (15 cores)
#   If you change RUNS_PER_JOB you MUST update the #SBATCH --array line to match.
#   Runs beyond index 59 are skipped, so an over-long array is harmless.
#
#     sbatch run_pure_rl_collapse_buffer_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Each run writes CSVs + params.txt to
#   logs/learning/pure_rl/auto-run/{base,roam60d5}_purerl_buf<N>_seed<S>_r<N>/
# The buf<N> tag is what the analysis keys the sweep arm off, and the _r<N>
# replicate suffix (deliberately NOT _rep<N>) keeps these out of the condition-
# consistency analysis, which matches runs by params and requires _rep<N>. A
# forced --run-name is not collision-checked and Logger auto-save APPENDS, so a
# unique name per run is required to avoid doubled rows.
#
#   GENERATIONS=200 sbatch run_pure_rl_collapse_buffer_array.slurm     # shorter
#   COLLAPSE_BUFFERS="25 50 100 200" sbatch --array=0-119 ...          # more sizes
#   N_REPS=3 sbatch --array=0-35 ...                                   # fewer reps
#   RUNS_PER_JOB=2 sbatch --array=0-29 ...                             # denser packing
# ============================================================================
#SBATCH --job-name="purerl_buffer"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-59
#SBATCH --output=logs/purerl_buffer_%A_%a.out
#SBATCH --error=logs/purerl_buffer_%A_%a.err
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

# ── Sweep grid (4 factors: buffer x environment x seed x replicate) ─────────
# The factor under test: rolling dead-weight buffer sampled to respawn after a
# collapse. 25 is the value every previous pure-RL run used (the old hard-coded
# constant) and is included here as the matched control arm.
# shellcheck disable=SC2206
COLLAPSE_BUFFERS=(${COLLAPSE_BUFFERS:-25 100})
ENVS=(baseline hard)
SEEDS=(1 42 999)                     # SAME 3 maps as the LR / trace sweeps
N_SEEDS=${#SEEDS[@]}
N_REPS="${N_REPS:-5}"                # independent replicate runs per cell
PER_ENV=$(( N_SEEDS * N_REPS ))              # runs per (buffer, env)  = 15
PER_BUFFER=$(( ${#ENVS[@]} * PER_ENV ))      # runs per buffer         = 30
TOTAL_RUNS=$(( ${#COLLAPSE_BUFFERS[@]} * PER_BUFFER ))   # 60

# ── Core packing: how many runs each array task executes back-to-back ────────
# CHANGING THIS REQUIRES updating the #SBATCH --array line (see header table).
RUNS_PER_JOB="${RUNS_PER_JOB:-1}"

# ── Fixed factors (identical for every run) ─────────────────────────────────
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-1000}"
# Hard-environment predator settings (baseline runs disable predators entirely).
ROAMING_PREDATORS="${ROAMING_PREDATORS:-60}"
PREDATOR_DRAIN="${PREDATOR_DRAIN:-5}"
# Per-environment RL learning rate — the optimum each environment's LR sweep
# found, and the value its existing pure-RL condition run used. Overriding
# LEARNING_RATE forces the SAME rate on both environments (rarely what you want).
LEARNING_RATE_BASELINE="${LEARNING_RATE_BASELINE:-0.01}"
LEARNING_RATE_HARD="${LEARNING_RATE_HARD:-0.02}"

# ── Preflight ───────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR" logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

# --collapse-buffer is a recent addition (ExperimentParams.collapse_buffer_size).
# An older clone silently IGNORES the flag and runs the whole sweep at 25, which
# looks like a null result instead of a mistake — so fail loudly up front.
if ! grep -q 'collapse_buffer_size' src/ExperimentParams.js 2>/dev/null; then
    echo "ERROR: this clone predates --collapse-buffer (no collapse_buffer_size in"
    echo "       src/ExperimentParams.js). Pull the current code before submitting —"
    echo "       otherwise every run silently uses the default buffer of 25."
    exit 1
fi

# ── One run, given a global run index ───────────────────────────────────────
# Decomposes the index into (buffer, environment, seed, replicate). Layout is
# buffer-major, then env, then seed, then replicate (replicate varies fastest):
#   g = BUF_IDX*PER_BUFFER + ENV_IDX*PER_ENV + SEED_IDX*N_REPS + REP_IDX
run_one() {
    local GIDX="$1"
    local BUF_IDX=$((  GIDX / PER_BUFFER ))
    local BUF_REM=$((  GIDX % PER_BUFFER ))
    local ENV_IDX=$((  BUF_REM / PER_ENV ))
    local REM=$((      BUF_REM % PER_ENV ))
    local SEED_IDX=$(( REM / N_REPS ))
    local REP=$((      REM % N_REPS + 1 ))     # replicate number, 1..N_REPS
    local COLLAPSE_BUFFER="${COLLAPSE_BUFFERS[$BUF_IDX]}"
    local ENV_NAME="${ENVS[$ENV_IDX]}"
    local MAP_SEED="${SEEDS[$SEED_IDX]}"

    # Per-environment predator + LR settings.
    local ROAMING DRAIN LR PREFIX
    if [ "$ENV_NAME" = "hard" ]; then
        ROAMING="$ROAMING_PREDATORS"; DRAIN="$PREDATOR_DRAIN"
        LR="${LEARNING_RATE:-$LEARNING_RATE_HARD}"
        PREFIX="roam60d5"
    else
        ROAMING=0; DRAIN=1
        LR="${LEARNING_RATE:-$LEARNING_RATE_BASELINE}"
        PREFIX="base"
    fi
    local RUN_NAME="${PREFIX}_purerl_buf${COLLAPSE_BUFFER}_seed${MAP_SEED}_r${REP}"

    export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${GIDX}"

    echo "[run g$GIDX] PURE-RL (RL only, no GA, no-epsilon)  env=$ENV_NAME  buffer=$COLLAPSE_BUFFER  seed=$MAP_SEED  rep=$REP/$N_REPS  lr=$LR  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
    echo "[run g$GIDX]   predators: roaming=$ROAMING  drain=$DRAIN  patrol=0"
    # --condition learning enables RL; --mode pure_rl removes the GA (PureRLManager).
    "$NODE_BIN" src/headless.js \
        --map-seed "$MAP_SEED" \
        --condition learning \
        --mode pure_rl \
        --generations "$GENERATIONS" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --run-name "$RUN_NAME" \
        --log-dir "$LOG_DIR" \
        --learning-rate "$LR" \
        --collapse-buffer "$COLLAPSE_BUFFER" \
        --no-epsilon \
        --predators-per-patch 0 \
        --roaming-predators "$ROAMING" \
        --predator-drain "$DRAIN"
    local rc=$?
    echo "[run g$GIDX] done (exit $rc): $RUN_NAME"
    return $rc
}

# ── Run this task's slice of RUNS_PER_JOB consecutive global indices ────────
START=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_JOB ))
echo "[array] task=$SLURM_ARRAY_TASK_ID  runs global indices ${START}..$(( START + RUNS_PER_JOB - 1 ))  (RUNS_PER_JOB=$RUNS_PER_JOB, of $TOTAL_RUNS total)  buffers=${COLLAPSE_BUFFERS[*]}  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"

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
