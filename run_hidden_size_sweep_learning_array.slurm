#!/bin/bash
# ============================================================================
# run_hidden_size_sweep_learning_array.slurm — network-capacity sweep
# (hidden width 32 / 64 / 128) for the LEARNING condition (GA + RL), in BOTH
# environments.
#
# What the hidden size is
# -----------------------
# NNBrain is a single-hidden-layer policy network, 54 -> HIDDEN -> 6
# (NNBrain.js: STATE_SIZE 54, OUTPUT_SIZE 6). ExperimentParams.hidden_size sets
# HIDDEN at module load, so it resizes the flat genome every organism carries:
#     GENOME_SIZE = 54*H + H + 6*H + 6 = 61H + 6
#         H =  32  ->  1958 weights   (half the default capacity)
#         H =  64  ->  3910 weights   (the default every previous run used)
#         H = 128  ->  7814 weights   (double the default capacity)
# It is the one knob that changes WHAT the policy can represent rather than how
# it is trained, and it trades off against everything else in the model: a wider
# net can express more behaviour per organism, but it is a bigger search space
# for the GA, a higher-variance target for REINFORCE, and a longer genome to
# carry through mutation. 64 is the value every earlier experiment used, so that
# arm doubles as the matched control the other two are measured against.
#
# This file is one of a matched TRIO — the same sweep, same grid, same seeds and
# replicates, run once per condition so capacity can be compared BETWEEN
# mechanisms as well as within one:
#     run_hidden_size_sweep_evolution_array.slurm   (GA only, no RL)
#     run_hidden_size_sweep_learning_array.slurm    <- THIS FILE (GA + RL)
#     run_hidden_size_sweep_pure_rl_array.slurm     (RL only, no GA, buffer 100)
# The interesting question is whether the three disagree: extra capacity that
# helps a learner within its lifetime is not obviously the same extra capacity
# that helps a GA search across generations.
#
# ALL THREE hidden sizes are run HERE rather than comparing against the existing
# condition runs, even though those already sit at 64. Reason: they are 1
# replicate on 5 seeds, this is 5 replicates on 3 seeds, and the run-to-run
# spread in this model is large enough that a 1-vs-5 comparison would be
# dominated by it. Running all arms in the same job also guarantees they share
# code version, node version and cluster conditions — the width becomes the only
# difference.
#
# Grid: 3 widths x 2 environments x 3 seeds x 5 replicates = 90 runs.
#     widths       : 32, 64 (default / control), 128
#     environments : baseline (predator-free, lr 0.01)
#                    hard     (60 roaming predators, drain 5, lr 0.02)
#     seeds        : 1, 42, 999   (SAME 3 maps as every other sweep)
#     replicates   : 1..5
# 500x500 grid, 1000 generations, learning condition (GA + on-policy REINFORCE,
# --no-epsilon), patrol predators off. Everything except the width matches
#     run_learning_condition_baseline_array.slurm  (baseline, lr 0.01)
#     run_learning_condition_roaming_array.slurm   (hard,     lr 0.02)
#
# WHY REPLICATES: only the map TERRAIN is seeded (--map-seed). Founder weights
# come from an unseeded Xavier draw, and GA mutation and RL action sampling are
# unseeded too, so the 5 replicates of a (env, seed) cell share identical terrain
# and differ purely by algorithmic chance. Aggregate across ALL runs of an arm
# (seeds x replicates) before comparing widths — that is what
# analyse_buffer_sweep.py's aggregation does, and the width analysis should match
# it.
#
# CORES / TIME (read before changing --array or RUNS_PER_JOB)
# -----------------------------------------------------------
# 90 runs PACKED THREE PER TASK -> 30 array tasks, 90h wall clock each. All
# three condition scripts can therefore be submitted at once and still fit the
# 120-simultaneous-core limit with room to spare: 3 x 30 = 90 cores. No %N array
# throttle is needed or set.
#
# The three runs in a task are NOT adjacent grid cells. Task k runs global
# indices k, k+30 and k+60 (the stride is the number of TASKS). The grid is
# width-major (32 -> 64 -> 128) and each width block is 30 runs, so that stride
# lands exactly one run of EACH width in every task — the most balanced packing
# available, since width is what drives the cost:
#     tasks  0-14 : h32 baseline + h64 baseline + h128 baseline   (~51h)
#     tasks 15-29 : h32 hard     + h64 hard     + h128 hard       (~24h)
# Contiguous packing would instead have put three h128 runs in one task, the
# slowest cells in the grid. Timings from the buffer sweep's pure-RL logs (the
# only 500x500/1000-generation runs measured so far) put width 64 at ~7h hard
# and ~15h baseline — baseline runs are the slow ones, bigger surviving
# populations — and width 128 roughly doubles the per-tick network cost.
#
# THE BASELINE TASKS ARE THE ONES TO WATCH: ~51h estimated against a 90h wall
# clock. That is comfortable but it is an ESTIMATE — the GA+RL condition has
# never been timed at 500x500 x 1000 generations, and it is the sum of three runs
# rather than one, so a slow node hits all three. If tasks 0-14 start timing out,
# drop to RUNS_PER_JOB=2 (see below) rather than raising --time further.
#
# CHANGING RUNS_PER_JOB REQUIRES updating the #SBATCH --array line:
#     RUNS_PER_JOB=1 -> --array=0-89     (90 tasks, no packing — needs 90 cores)
#     RUNS_PER_JOB=2 -> --array=0-44     (45 tasks, ~35h; 3 scripts = 135 cores)
#     RUNS_PER_JOB=3 -> --array=0-29     (30 tasks, ~51h)   <- CURRENT DEFAULT
# Indices past the end of the grid are skipped, so an over-long array is harmless.
#
# SUBMITTING ALL THREE AT ONCE (the intended workflow):
#     sbatch run_hidden_size_sweep_evolution_array.slurm
#     sbatch run_hidden_size_sweep_learning_array.slurm
#     sbatch run_hidden_size_sweep_pure_rl_array.slurm
# 3 x 30 = 90 simultaneous tasks, inside the 120-core allowance, so all three
# arrays run flat out with nothing queued behind a throttle.
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Each run writes CSVs + params.txt to
#   logs/learning/standard/auto-run/{base,roam60d5}_learning_h<H>_seed<S>_r<N>/
# The h<H> tag is what the analysis keys the sweep arm off (hidden_size is also
# recorded in params.json), and the _r<N> replicate suffix (deliberately NOT
# _rep<N>) keeps these out of the condition-consistency analysis, which matches
# runs by params and requires _rep<N>. A forced --run-name is not
# collision-checked and Logger auto-save APPENDS, so a unique name per run is
# required to avoid doubled rows.
#
#   GENERATIONS=200 sbatch run_hidden_size_sweep_learning_array.slurm   # shorter
#   HIDDEN_SIZES="32 64 128 256" sbatch --array=0-39 ...             # 120 runs, 40 tasks
#   N_REPS=3 sbatch --array=0-17 ...                                   # 54 runs, 18 tasks
# ============================================================================
#SBATCH --job-name="hsweep_learning"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-29
#SBATCH --output=logs/hsweep_learning_%A_%a.out
#SBATCH --error=logs/hsweep_learning_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
# Run from wherever you submit: cd into your SCRATCH clone, then `sbatch ...`.
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from; falls back to a home
# clone if run some other way.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/LifeEngine}}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

# ── Sweep grid (4 factors: width x environment x seed x replicate) ──────────
# The factor under test: NNBrain hidden-layer width. 64 is the value every
# previous run used (the ExperimentParams default) and is included here as the
# matched control arm.
# shellcheck disable=SC2206
HIDDEN_SIZES=(${HIDDEN_SIZES:-32 64 128})
ENVS=(baseline hard)
SEEDS=(1 42 999)                     # SAME 3 maps as the LR / trace / buffer sweeps
N_SEEDS=${#SEEDS[@]}
N_REPS="${N_REPS:-5}"                # independent replicate runs per cell
PER_ENV=$(( N_SEEDS * N_REPS ))              # runs per (width, env)  = 15
PER_HIDDEN=$(( ${#ENVS[@]} * PER_ENV ))      # runs per width         = 30
TOTAL_RUNS=$(( ${#HIDDEN_SIZES[@]} * PER_HIDDEN ))   # 90

# ── Core packing: how many runs each array task executes back-to-back ────────
# CHANGING THIS REQUIRES updating the #SBATCH --array line (see header table).
RUNS_PER_JOB="${RUNS_PER_JOB:-3}"

# ── Fixed factors (identical for every run) ─────────────────────────────────
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-1000}"
# Hard-environment predator settings (baseline runs disable predators entirely).
ROAMING_PREDATORS="${ROAMING_PREDATORS:-60}"
PREDATOR_DRAIN="${PREDATOR_DRAIN:-5}"
# Per-environment RL learning rate — the optimum each environment's LR sweep
# found, and the value its existing condition run used. Overriding LEARNING_RATE
# forces the SAME rate on both environments (rarely what you want).
LEARNING_RATE_BASELINE="${LEARNING_RATE_BASELINE:-0.01}"
LEARNING_RATE_HARD="${LEARNING_RATE_HARD:-0.02}"

# ── Preflight ───────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR" logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

# An older clone without --hidden-size silently IGNORES the flag and runs every
# arm at the default 64, which looks like a null result instead of a mistake —
# so fail loudly up front.
if ! grep -q "'hidden-size'" src/headless.js 2>/dev/null; then
    echo "ERROR: this clone has no --hidden-size flag in src/headless.js. Pull the"
    echo "       current code before submitting — otherwise every run silently"
    echo "       uses the default width of 64."
    exit 1
fi

# ── One run, given a global run index ───────────────────────────────────────
# Decomposes the index into (width, environment, seed, replicate). Layout is
# width-major, then env, then seed, then replicate (replicate varies fastest):
#   g = H_IDX*PER_HIDDEN + ENV_IDX*PER_ENV + SEED_IDX*N_REPS + REP_IDX
run_one() {
    local GIDX="$1"
    local H_IDX=$((    GIDX / PER_HIDDEN ))
    local H_REM=$((    GIDX % PER_HIDDEN ))
    local ENV_IDX=$((  H_REM / PER_ENV ))
    local REM=$((      H_REM % PER_ENV ))
    local SEED_IDX=$(( REM / N_REPS ))
    local REP=$((      REM % N_REPS + 1 ))     # replicate number, 1..N_REPS
    local HIDDEN="${HIDDEN_SIZES[$H_IDX]}"
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
    local RUN_NAME="${PREFIX}_learning_h${HIDDEN}_seed${MAP_SEED}_r${REP}"

    export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${GIDX}"

    echo "[run g$GIDX] LEARNING (GA + RL, no-epsilon)  env=$ENV_NAME  hidden=$HIDDEN  seed=$MAP_SEED  rep=$REP/$N_REPS  lr=$LR  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
    echo "[run g$GIDX]   genome = 61*$HIDDEN + 6 = $(( 61 * HIDDEN + 6 )) weights;  predators: roaming=$ROAMING  drain=$DRAIN  patrol=0"
    "$NODE_BIN" src/headless.js \
        --map-seed "$MAP_SEED" \
        --condition learning \
        --mode standard \
        --generations "$GENERATIONS" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --run-name "$RUN_NAME" \
        --log-dir "$LOG_DIR" \
        --hidden-size "$HIDDEN" \
        --learning-rate "$LR" \
        --no-epsilon \
        --predators-per-patch 0 \
        --roaming-predators "$ROAMING" \
        --predator-drain "$DRAIN"
    local rc=$?
    echo "[run g$GIDX] done (exit $rc): $RUN_NAME"
    return $rc
}

# ── Run this task's STRIDED slice of RUNS_PER_JOB global indices ────────────
# Task k runs indices k, k+N_TASKS, k+2*N_TASKS, ... — NOT k*RUNS_PER_JOB and its
# neighbours. The grid is width-major, so a stride of N_TASKS pairs a narrow run
# with a wide one and keeps the two slowest cells out of the same task; see the
# CORES/TIME note in the header. N_TASKS is derived from RUNS_PER_JOB so the two
# stay consistent, but the #SBATCH --array line must be updated to match by hand.
N_TASKS=$(( (TOTAL_RUNS + RUNS_PER_JOB - 1) / RUNS_PER_JOB ))

INDICES=()
for (( k = 0; k < RUNS_PER_JOB; k++ )); do
    GIDX=$(( SLURM_ARRAY_TASK_ID + k * N_TASKS ))
    [ "$GIDX" -lt "$TOTAL_RUNS" ] && INDICES+=("$GIDX")
done

echo "[array] task=$SLURM_ARRAY_TASK_ID  runs global indices ${INDICES[*]:-none}  (RUNS_PER_JOB=$RUNS_PER_JOB, stride=$N_TASKS, of $TOTAL_RUNS total)  widths=${HIDDEN_SIZES[*]}  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"

if [ ${#INDICES[@]} -eq 0 ]; then
    echo "[array] task index $SLURM_ARRAY_TASK_ID is past the end of the grid — nothing to do."
    exit 0
fi

# Don't let one failed run abort the rest of the job's packed runs.
fail=0
for GIDX in "${INDICES[@]}"; do
    run_one "$GIDX" || { echo "[array] WARNING: run g$GIDX exited non-zero"; fail=1; }
done

echo "[array] task=$SLURM_ARRAY_TASK_ID complete (fail=$fail)"
exit $fail
