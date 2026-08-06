#!/bin/bash
# ============================================================================
# run_evolution_condition_hard_w1k_array.slurm — the EVOLUTION condition
# (GA only, NO RL) in the HARD environment on a 1000x1000 world.
#
# What is new here vs every earlier evolution run
# -----------------------------------------------
#   grid        1000 x 1000   (4x the AREA of the 500x500 runs)
#   generations 500           (half the 1000-generation budget)
#   hidden_size 128           (the new standing default — 61*128 + 6 = 7814
#                              weights)
#   buffer      100           (the new standing default; INERT here — see below)
#   predators   240 roaming   (60 rescaled to constant DENSITY — see below)
#   food        x4 per ref    (rescaled to constant DENSITY, --food-density-scale)
#
# WHY 128 IS THE RIGHT DEFAULT FOR *THIS* CONDITION IN *THIS* ENVIRONMENT
# ----------------------------------------------------------------------
# The network-capacity sweep (analyse_hidden_size_sweep.py, hard environment)
# found the evolution condition's runs split into two groups: some take off and
# some plateau low and never recover. Width changed HOW MANY got going, not how
# high the ones that did got:
#     runs that plateaued low   h32 11/15    h64 8/15    h128 5/15
# Everything that took off landed at ~7.1 fitness regardless of width. So 128 is
# chosen here for CONSISTENCY, not for per-run performance — and 5 replicates on
# 5 seeds is sized to measure exactly that split, since with ~1/3 of runs still
# expected to stall, an arm of 5 would be mostly noise.
#
# Grid: 5 seeds x 5 replicates = 25 runs.
#     seeds      : 1, 42, 999   (the 3 existing maps)
#                + 123, 456     (2 further maps — auto-generated on first use)
#     replicates : 1..5
# evolution/standard condition (GA only — no RL, so no learning rate and no
# epsilon settings apply), patrol predators off.
#
# WHY REPLICATES: only the map TERRAIN is seeded (--map-seed). Founder weights
# come from an unseeded Xavier draw and GA mutation is unseeded too, so the 5
# replicates of a seed share identical terrain and differ purely by algorithmic
# chance. Aggregate across ALL 25 runs before drawing conclusions — and count the
# stalled ones separately, since the mean of a bimodal arm describes none of it.
#
# PREDATOR DENSITY IS HELD CONSTANT, NOT PREDATOR COUNT
# -----------------------------------------------------
# The hard environment has always meant 60 roaming predators on 500x500, i.e.
#     60 / 250,000 cells = 2.4e-4 predators per cell.
# 1000x1000 is 1,000,000 cells, so the same density needs
#     2.4e-4 * 1,000,000 = 240 roaming predators.
# Keeping the COUNT at 60 would have quartered the encounter rate and quietly
# turned "hard" into something closer to baseline. Drain stays at 5 per contact —
# that is a per-encounter quantity and does not scale.
#
# FOOD DENSITY IS HELD CONSTANT THE SAME WAY  (--food-density-scale 4)
# ---------------------------------------------------------------------
# Map pools store cells as RELATIVE coordinates against a 400x300 reference grid
# and WorldEnvironment._loadMapFromPool rescales them to the runtime grid. Caves
# and landmark lines are drawn as filled footprints, so they scale with area —
# but FOOD was historically placed as single points (one runtime cell per
# reference cell), which fixes the food COUNT while the area grows. A map that
# put ~3,400 food cells on 250,000 cells (1.4% cover at 500x500) would put the
# same ~3,400 on 1,000,000 (0.34% cover) — a silent ~4x famine on top of the
# predators, which for THIS condition would have been the difference between
# measuring capacity and measuring starvation.
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
# lifetime all change. Expect the stall/take-off split to move — that is the
# measurement, not a defect.
#
# ABOUT --collapse-buffer 100
# ---------------------------
# The collapse-respawn buffer is PureRLManager's mechanism (--collapse-buffer is
# documented "pure_rl:" in headless.js). This condition runs the GA and no RL at
# all, so the flag has NO EFFECT on the simulation here. It is passed anyway so
# params.json records 100 for every run of the new standard and an analysis that
# filters on collapse_buffer_size sees a consistent value across conditions.
#
# CORES / TIME (read before changing --array or RUNS_PER_JOB)
# -----------------------------------------------------------
# Estimated from the h128 evolution-hard arm at 500x500 x 1000 generations
# (6.3h median, 13.5h worst), fitted against its real per-generation timings
# rather than assumed. Two measured facts:
#
#   1. Cost is LINEAR IN POPULATION with a small intercept:
#          sec/gen = 5.9 + 0.0369 * total_agents
#      So x4 the organisms is ~x4 the cost per generation. Matching food density
#      (above) means the 4x world carries ~4x the population, so the area factor
#      passes straight through to wall clock.
#
#   2. HALVING THE GENERATIONS DOES NOT HALVE THE COST. The population grows
#      over a run, so early generations are cheap: generations 1-500 are only
#      38.0% of a full run's wall clock here — the lowest of the three
#      conditions, because a stalling GA population takes a long time to build.
#
# Applying both:  6.3h x 4 x 0.380 = ~10h median,  13.5h x 4 x 0.380 = ~21h worst,
# against a 60h wall clock. Comfortable.
#
# The spread is the thing to watch, not the median: a run that takes off carries
# a large population for the rest of its generations and costs several times one
# that stalls at a hundred-odd organisms. The 60h limit is set for the former.
#
# MEMORY: 16G, up from the 8G the 500x500 runs used. Hard-environment
# populations stay well below the baseline condition's, so this has more headroom
# than the learning sweep's 24G.
#
# PACKING (this is the job that CAN pack, if you want cores back)
# ---------------------------------------------------------------
# 25 tasks + the learning sweep's 90 = 115 of the 120-core allowance at one run
# per task, which is the most parallel layout and finishes everything in one
# window. Packing does not free cores for this grid — it frees them for a THIRD
# job, at the cost of this one finishing later. These all fit:
#     RUNS_PER_JOB=1 -> --array=0-24   25 tasks   ~10h  (worst ~21h)  <- DEFAULT
#     RUNS_PER_JOB=2 -> --array=0-12   13 tasks   ~20h  (worst ~41h)  fits 60h
#     RUNS_PER_JOB=3 -> --array=0-8     9 tasks   ~29h  (worst ~62h)  RAISE
#                                                       --time to 72:00:00 first
# The worst case assumes every packed run in a task takes off, which is the
# point of the 5x5 grid — do not size --time off the median.
# Indices past the end of the grid are skipped, so an over-long array is harmless.
#
# SUBMITTING ALONGSIDE THE LEARNING LR SWEEP (the intended workflow):
#     sbatch run_evolution_condition_hard_w1k_array.slurm   #  25 tasks
#     sbatch run_lr_sweep_learning_w1k_array.slurm          #  90 tasks
# 25 + 90 = 115 simultaneous tasks, inside the 120-core allowance. Submit THIS
# one first: its runs are much shorter, so it clears while the learning sweep is
# still going.
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Each run writes CSVs + params.txt to
#   logs/evolution/standard/auto-run/roam240d5_w1k_evolution_h128_seed<S>_r<N>/
# The w1k tag marks the 1000x1000 world so these never mix with the 500x500 runs
# in the same directory, and the _r<N> replicate suffix (deliberately NOT
# _rep<N>) keeps them out of the condition-consistency analysis, which matches
# runs by params and requires _rep<N>. A forced --run-name is not
# collision-checked and Logger auto-save APPENDS, so a unique name per run is
# required to avoid doubled rows.
#
# MAP-SEED note: headless.js builds each world deterministically from --map-seed
# and caches a brand-new seed's pool atomically to
# src/maps/map_pool_seed<SEED>.json (temp-write + rename), so parallel tasks that
# hit seeds 123/456 at the same time cannot corrupt them. No manual
# pre-generation is needed. The pools are resolution-independent, so the same
# cached seed gives the SAME terrain rescaled to 1000x1000.
#
#   GENERATIONS=200 sbatch run_evolution_condition_hard_w1k_array.slurm  # shorter
#   N_REPS=10 sbatch --array=0-49 ...                                    # 50 runs
#   SEEDS="1 42 999" sbatch --array=0-14 ...                             # 15 runs
# ============================================================================
#SBATCH --job-name="evo_hard_w1k"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-24
#SBATCH --output=logs/evo_hard_w1k_%A_%a.out
#SBATCH --error=logs/evo_hard_w1k_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
# Run from wherever you submit: cd into your SCRATCH clone, then `sbatch ...`.
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from; falls back to a home
# clone if run some other way.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/LifeEngine}}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

# ── Sweep grid (2 factors: seed x replicate) ────────────────────────────────
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-1 42 999 123 456})   # 3 existing maps + 2 auto-generated
N_SEEDS=${#SEEDS[@]}
N_REPS="${N_REPS:-5}"                # independent replicate runs per seed
TOTAL_RUNS=$(( N_SEEDS * N_REPS ))   # 25

# ── Core packing: how many runs each array task executes back-to-back ────────
# CHANGING THIS REQUIRES updating the #SBATCH --array line (see header table).
RUNS_PER_JOB="${RUNS_PER_JOB:-1}"

# ── Fixed factors (identical for every run) ─────────────────────────────────
WIDTH="${WIDTH:-1000}"
HEIGHT="${HEIGHT:-1000}"
GENERATIONS="${GENERATIONS:-500}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
COLLAPSE_BUFFER="${COLLAPSE_BUFFER:-100}"    # recorded only; inert outside pure_rl
# 240 = the 500x500 count of 60 rescaled to 1000x1000 at CONSTANT DENSITY (see
# the header). Drain is per-contact and does not scale.
ROAMING_PREDATORS="${ROAMING_PREDATORS:-240}"
PREDATOR_DRAIN="${PREDATOR_DRAIN:-5}"
# Runtime food cells per REFERENCE food cell. 4 = (1000*1000)/(500*500), i.e. the
# area ratio that holds food density equal to the 500x500 runs (see the header).
# A 500x500 run would use 1 (the default and the historical behaviour).
FOOD_DENSITY_SCALE="${FOOD_DENSITY_SCALE:-4}"
# NOTE: no learning rate / epsilon here — the evolution condition runs no RL at
# all, so those flags would be recorded in params.txt but have no effect.

# ── Preflight ───────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR" logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

# An older clone without --hidden-size silently IGNORES the flag and runs every
# arm at the default 64, which would make this a width-64 job wearing h128
# folder names. Fail loudly up front.
if ! grep -q "'hidden-size'" src/headless.js 2>/dev/null; then
    echo "ERROR: this clone has no --hidden-size flag in src/headless.js. Pull the"
    echo "       current code before submitting — otherwise every run silently"
    echo "       uses the default width of 64."
    exit 1
fi

# ── One run, given a global run index ───────────────────────────────────────
# Decomposes the index into (seed, replicate); replicate varies fastest:
#   g = SEED_IDX*N_REPS + REP_IDX
run_one() {
    local GIDX="$1"
    local SEED_IDX=$(( GIDX / N_REPS ))
    local REP=$((      GIDX % N_REPS + 1 ))    # replicate number, 1..N_REPS
    local MAP_SEED="${SEEDS[$SEED_IDX]}"
    local PREFIX="roam${ROAMING_PREDATORS}d${PREDATOR_DRAIN}"
    local RUN_NAME="${PREFIX}_w1k_evolution_h${HIDDEN_SIZE}_seed${MAP_SEED}_r${REP}"

    export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${GIDX}"

    echo "[run g$GIDX] EVOLUTION (GA only, no RL)  env=hard  seed=$MAP_SEED  rep=$REP/$N_REPS  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
    echo "[run g$GIDX]   hidden=$HIDDEN_SIZE ($(( 61 * HIDDEN_SIZE + 6 )) weights);  predators: roaming=$ROAMING_PREDATORS  drain=$PREDATOR_DRAIN  patrol=0;  food_density_scale=$FOOD_DENSITY_SCALE"
    "$NODE_BIN" src/headless.js \
        --map-seed "$MAP_SEED" \
        --condition evolution \
        --mode standard \
        --generations "$GENERATIONS" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --run-name "$RUN_NAME" \
        --log-dir "$LOG_DIR" \
        --hidden-size "$HIDDEN_SIZE" \
        --collapse-buffer "$COLLAPSE_BUFFER" \
        --food-density-scale "$FOOD_DENSITY_SCALE" \
        --predators-per-patch 0 \
        --roaming-predators "$ROAMING_PREDATORS" \
        --predator-drain "$PREDATOR_DRAIN"
    local rc=$?
    echo "[run g$GIDX] done (exit $rc): $RUN_NAME"
    return $rc
}

# ── Run this task's STRIDED slice of RUNS_PER_JOB global indices ────────────
# At the default RUNS_PER_JOB=1 this is simply "task k runs index k". The stride
# only matters if you repack: it is N_TASKS, not 1, so a task picks runs from
# DIFFERENT seeds rather than adjacent replicates of the same one — replicates of
# one seed are the runs most likely to cost the same, so pairing them is the
# worst packing available.
N_TASKS=$(( (TOTAL_RUNS + RUNS_PER_JOB - 1) / RUNS_PER_JOB ))

INDICES=()
for (( k = 0; k < RUNS_PER_JOB; k++ )); do
    GIDX=$(( SLURM_ARRAY_TASK_ID + k * N_TASKS ))
    [ "$GIDX" -lt "$TOTAL_RUNS" ] && INDICES+=("$GIDX")
done

echo "[array] task=$SLURM_ARRAY_TASK_ID  runs global indices ${INDICES[*]:-none}  (RUNS_PER_JOB=$RUNS_PER_JOB, stride=$N_TASKS, of $TOTAL_RUNS total)  seeds=${SEEDS[*]}  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS  hidden=$HIDDEN_SIZE"

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
