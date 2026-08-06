#!/bin/bash
# ============================================================================
# run_lr_sweep_learning_w1k_array.slurm — small LEARNING-RATE sweep for the
# LEARNING condition (GA + in-lifetime REINFORCE, no epsilon) on a 1000x1000
# world, in BOTH environments.
#
# What is new here vs every earlier sweep
# ---------------------------------------
#   grid        1000 x 1000   (4x the AREA of the 500x500 runs)
#   generations 500           (half the 1000-generation budget)
#   hidden_size 128           (the new standing default — 61*128 + 6 = 7814
#                              weights; see analyse_hidden_size_sweep.py)
#   buffer      100           (the new standing default; INERT in this mode —
#                              see the note below)
#   predators   240 roaming   (60 rescaled to constant DENSITY)
#   food        x4 per ref    (rescaled to constant DENSITY, --food-density-scale)
#
# Grid: 3 learning rates x 2 environments x 3 seeds x 5 replicates = 90 runs.
#     learning rates : 0.01, 0.02, 0.03
#                      (0.01 / 0.02 are the previously selected baseline / hard
#                      optima; 0.03 extends the sweep upward, since the longer
#                      journeys of a 4x larger world may reward faster
#                      within-life adaptation even at matched density)
#     environments   : baseline (predator-free)
#                      hard     (240 roaming predators, drain 5)
#     seeds          : 1, 42, 999   (SAME 3 maps as every other sweep)
#     replicates     : 1..5
# learning/standard condition, epsilon OFF (pure on-policy REINFORCE), patrol
# predators off.
#
# PREDATOR DENSITY IS HELD CONSTANT, NOT PREDATOR COUNT
# -----------------------------------------------------
# The hard environment has always meant 60 roaming predators on 500x500, i.e.
#     60 / 250,000 cells = 2.4e-4 predators per cell.
# 1000x1000 is 1,000,000 cells, so the same density needs
#     2.4e-4 * 1,000,000 = 240 roaming predators.
# Keeping the COUNT at 60 would have quartered the encounter rate and quietly
# turned "hard" into something closer to baseline; the whole point of the larger
# world is to keep the difficulty while giving the population room. Drain stays
# at 5 per contact — that is a per-encounter quantity and does not scale.
#
# FOOD DENSITY IS HELD CONSTANT THE SAME WAY  (--food-density-scale 4)
# ---------------------------------------------------------------------
# Map pools store cells as RELATIVE coordinates against a 400x300 reference grid
# and WorldEnvironment._loadMapFromPool rescales them to the runtime grid. Caves
# and landmark lines are drawn as filled footprints, so they scale with area —
# but FOOD was historically placed as single points (one runtime cell per
# reference cell, "an organic, scattered blob by design"), which fixes the food
# COUNT while the area grows. A map that put ~3,400 food cells on 250,000 cells
# (1.4% cover at 500x500) would put the same ~3,400 on 1,000,000 (0.34% cover) —
# a silent ~4x famine on top of the predators.
#
# --food-density-scale N places N runtime food cells per reference food cell,
# scattered inside that cell's runtime footprint (deterministically, seeded from
# the map, so a seed still reproduces its terrain exactly). Set to the AREA
# RATIO being matched:
#     (1000*1000) / (500*500) = 4
# so food per unit area matches the 500x500 runs, exactly as 240 predators match
# their density. Verify from the run's own log: the [WorldEnv] Map line reports
#     food_cells=13652 (x4/ref)     vs   food_cells=3413 (x1/ref)  unscaled
#
# The flag defaults to 1, which is the historical behaviour bit for bit, so no
# existing run or analysis is affected by its introduction.
#
# STILL DO NOT compare absolute fitness against the 500x500 runs. Density is
# matched, but a 4x larger world with the same density is not the same problem:
# travel distances, patch spacing and the reachable fraction of the map in one
# lifetime all change. Compare learning rates against each other WITHIN this
# sweep.
#
# ABOUT --collapse-buffer 100
# ---------------------------
# The collapse-respawn buffer is PureRLManager's mechanism (--collapse-buffer is
# documented "pure_rl:" in headless.js). This condition runs the GA, so the flag
# has NO EFFECT on the simulation here. It is passed anyway so params.json
# records 100 for every run of the new standard and an analysis that filters on
# collapse_buffer_size sees a consistent value across conditions.
#
# CORES / TIME (read before changing --array or RUNS_PER_JOB)
# -----------------------------------------------------------
# Estimated from the h128 arms of the network-capacity sweep, fitted against
# their real per-generation timings rather than assumed. Two measured facts:
#
#   1. Cost is LINEAR IN POPULATION with a negligible intercept:
#          learning baseline   sec/gen = -3.5 + 0.0894 * total_agents
#          learning hard       sec/gen = -0.1 + 0.0748 * total_agents
#      So x4 the organisms is x4 the cost per generation, near enough exactly.
#      Matching food density (above) means the 4x world carries ~4x the
#      population, so the area factor passes straight through to wall clock.
#
#   2. HALVING THE GENERATIONS DOES NOT HALVE THE COST. The population grows
#      over a run, so early generations are cheap: generations 1-500 are only
#      46.7% of a baseline run's total wall clock (43.6% for hard), not 50%.
#
# Applying both to the measured 500x500 x 1000-generation times:
#     baseline  31h x 4 x 0.467 = ~58h
#     hard      25h x 4 x 0.436 = ~44h
# against a 90h wall clock.
#
# WHY RUNS_PER_JOB IS 1 — AND WHY THAT IS ALSO THE FASTEST OPTION
# ---------------------------------------------------------------
# 90 tasks + the evolution job's 25 = 115 of the 120-core allowance, one run
# each. That is the MOST parallel layout available and finishes the whole grid
# in a single ~58h window. Packing runs per task does not save cores you can
# use for anything — it just serialises runs that would otherwise be concurrent,
# so the sweep finishes later. Pack only if you need cores back for a third job.
#
# It also does not fit here. The stride pairs one baseline run with one hard run
# per task (that is the point of the stride), so:
#     RUNS_PER_JOB=1 -> --array=0-89   90 tasks   worst task ~58h   <- DEFAULT
#     RUNS_PER_JOB=2 -> --array=0-44   45 tasks   every task ~102h  EXCEEDS 90h
#     RUNS_PER_JOB=3 -> --array=0-29   30 tasks   ~174h AND BADLY PACKED:
#                       at 3 the stride (30) equals PER_LR, so a task draws the
#                       same env slot from all three lr blocks — three BASELINE
#                       runs land in one task. Do not use 3 without changing the
#                       index layout.
# 2 runs per task only becomes viable if your partition allows > 102h.
# Indices past the end of the grid are skipped, so an over-long array is harmless.
#
# THE BASELINE TASKS ARE THE ONES TO WATCH: ~58h estimated against 90h. There is
# margin, but not a lot of it, and holding food density constant deliberately
# removes the effect that would otherwise have kept populations (and therefore
# cost) down. If baseline tasks start timing out, drop GENERATIONS rather than
# thinning the grid — 500 generations is already the short end.
#
# MEMORY: 24G, up from the 8G the 500x500 runs used. The grid is 4x the cells
# and carries ~4x the organisms at matched food density.
#
# SUBMITTING ALONGSIDE THE EVOLUTION JOB (the intended workflow):
#     sbatch run_lr_sweep_learning_w1k_array.slurm            #  90 tasks
#     sbatch run_evolution_condition_hard_w1k_array.slurm     #  25 tasks
# 90 + 25 = 115 simultaneous tasks, inside the 120-core allowance, so both arrays
# run flat out with nothing queued behind a throttle. Submit the evolution job
# FIRST if you want its (much shorter) runs to start without waiting.
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# DISK: ~4x the agents at half the generations is ~2x the organisms.csv of a
# 500x500/1000-gen run (~100 MB), so budget ~200 MB x 90 runs = ~18 GB.
#
# Each run writes CSVs + params.txt to
#   logs/learning/standard/auto-run/{base,roam240d5}_w1k_learning_h128_lr<LR>_seed<S>_r<N>/
# The w1k tag marks the 1000x1000 world so these never mix with the 500x500
# runs in the same directory, and the _r<N> replicate suffix (deliberately NOT
# _rep<N>) keeps them out of the condition-consistency analysis, which matches
# runs by params and requires _rep<N>. A forced --run-name is not
# collision-checked and Logger auto-save APPENDS, so a unique name per run is
# required to avoid doubled rows.
#
#   GENERATIONS=200 sbatch run_lr_sweep_learning_w1k_array.slurm      # shorter
#   LEARNING_RATES="0.005 0.01 0.02 0.03" sbatch --array=0-119 ...    # 120 runs
#   N_REPS=3 sbatch --array=0-53 ...                                  # 54 runs
# ============================================================================
#SBATCH --job-name="lrsweep_learn_w1k"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --array=0-89
#SBATCH --output=logs/lrsweep_learn_w1k_%A_%a.out
#SBATCH --error=logs/lrsweep_learn_w1k_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
# Run from wherever you submit: cd into your SCRATCH clone, then `sbatch ...`.
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from; falls back to a home
# clone if run some other way.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/LifeEngine}}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

# ── Sweep grid (4 factors: learning rate x environment x seed x replicate) ──
# shellcheck disable=SC2206
LEARNING_RATES=(${LEARNING_RATES:-0.01 0.02 0.03})
ENVS=(baseline hard)
SEEDS=(1 42 999)                     # SAME 3 maps as the LR / trace / buffer sweeps
N_SEEDS=${#SEEDS[@]}
N_REPS="${N_REPS:-5}"                # independent replicate runs per cell
PER_ENV=$(( N_SEEDS * N_REPS ))              # runs per (lr, env)  = 15
PER_LR=$(( ${#ENVS[@]} * PER_ENV ))          # runs per lr         = 30
TOTAL_RUNS=$(( ${#LEARNING_RATES[@]} * PER_LR ))     # 90

# ── Core packing: how many runs each array task executes back-to-back ────────
# CHANGING THIS REQUIRES updating the #SBATCH --array line (see header table).
RUNS_PER_JOB="${RUNS_PER_JOB:-1}"

# ── Fixed factors (identical for every run) ─────────────────────────────────
WIDTH="${WIDTH:-1000}"
HEIGHT="${HEIGHT:-1000}"
GENERATIONS="${GENERATIONS:-500}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
COLLAPSE_BUFFER="${COLLAPSE_BUFFER:-100}"    # recorded only; inert outside pure_rl
# Runtime food cells per REFERENCE food cell. 4 = (1000*1000)/(500*500), i.e. the
# area ratio that holds food density equal to the 500x500 runs (see the header).
# A 500x500 run would use 1 (the default and the historical behaviour).
FOOD_DENSITY_SCALE="${FOOD_DENSITY_SCALE:-4}"
# Hard-environment predator settings. 240 = the 500x500 count of 60 rescaled to
# 1000x1000 at CONSTANT DENSITY (see the header). Drain is per-contact and does
# not scale.
ROAMING_PREDATORS="${ROAMING_PREDATORS:-240}"
PREDATOR_DRAIN="${PREDATOR_DRAIN:-5}"

# ── Preflight ───────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR" logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

# An older clone without --hidden-size silently IGNORES the flag and runs every
# arm at the default 64, which would make this a width-64 sweep wearing h128
# folder names. Fail loudly up front.
if ! grep -q "'hidden-size'" src/headless.js 2>/dev/null; then
    echo "ERROR: this clone has no --hidden-size flag in src/headless.js. Pull the"
    echo "       current code before submitting — otherwise every run silently"
    echo "       uses the default width of 64."
    exit 1
fi

# ── One run, given a global run index ───────────────────────────────────────
# Decomposes the index into (learning rate, environment, seed, replicate).
# Layout is lr-major, then env, then seed, then replicate (replicate fastest):
#   g = LR_IDX*PER_LR + ENV_IDX*PER_ENV + SEED_IDX*N_REPS + REP_IDX
run_one() {
    local GIDX="$1"
    local LR_IDX=$((    GIDX / PER_LR ))
    local LR_REM=$((    GIDX % PER_LR ))
    local ENV_IDX=$((   LR_REM / PER_ENV ))
    local REM=$((       LR_REM % PER_ENV ))
    local SEED_IDX=$((  REM / N_REPS ))
    local REP=$((       REM % N_REPS + 1 ))    # replicate number, 1..N_REPS
    local LR="${LEARNING_RATES[$LR_IDX]}"
    local ENV_NAME="${ENVS[$ENV_IDX]}"
    local MAP_SEED="${SEEDS[$SEED_IDX]}"

    # Per-environment predator settings.
    local ROAMING DRAIN PREFIX
    if [ "$ENV_NAME" = "hard" ]; then
        ROAMING="$ROAMING_PREDATORS"; DRAIN="$PREDATOR_DRAIN"
        PREFIX="roam${ROAMING_PREDATORS}d${PREDATOR_DRAIN}"
    else
        ROAMING=0; DRAIN=1
        PREFIX="base"
    fi
    local RUN_NAME="${PREFIX}_w1k_learning_h${HIDDEN_SIZE}_lr${LR}_seed${MAP_SEED}_r${REP}"

    export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${GIDX}"

    echo "[run g$GIDX] LEARNING (GA + RL, no epsilon)  env=$ENV_NAME  lr=$LR  seed=$MAP_SEED  rep=$REP/$N_REPS  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
    echo "[run g$GIDX]   hidden=$HIDDEN_SIZE ($(( 61 * HIDDEN_SIZE + 6 )) weights);  predators: roaming=$ROAMING  drain=$DRAIN  patrol=0;  food_density_scale=$FOOD_DENSITY_SCALE"
    "$NODE_BIN" src/headless.js \
        --map-seed "$MAP_SEED" \
        --condition learning \
        --mode standard \
        --generations "$GENERATIONS" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --run-name "$RUN_NAME" \
        --log-dir "$LOG_DIR" \
        --hidden-size "$HIDDEN_SIZE" \
        --collapse-buffer "$COLLAPSE_BUFFER" \
        --food-density-scale "$FOOD_DENSITY_SCALE" \
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
# At the default RUNS_PER_JOB=1 this is simply "task k runs index k". The stride
# only matters if you repack: it is N_TASKS, not 1, so a task picks runs from
# DIFFERENT lr blocks rather than adjacent cells of the same one — the balanced
# packing, since cost varies far more between environments than within a cell.
N_TASKS=$(( (TOTAL_RUNS + RUNS_PER_JOB - 1) / RUNS_PER_JOB ))

INDICES=()
for (( k = 0; k < RUNS_PER_JOB; k++ )); do
    GIDX=$(( SLURM_ARRAY_TASK_ID + k * N_TASKS ))
    [ "$GIDX" -lt "$TOTAL_RUNS" ] && INDICES+=("$GIDX")
done

echo "[array] task=$SLURM_ARRAY_TASK_ID  runs global indices ${INDICES[*]:-none}  (RUNS_PER_JOB=$RUNS_PER_JOB, stride=$N_TASKS, of $TOTAL_RUNS total)  lrs=${LEARNING_RATES[*]}  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS  hidden=$HIDDEN_SIZE"

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
