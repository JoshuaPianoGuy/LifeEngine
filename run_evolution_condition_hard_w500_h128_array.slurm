#!/bin/bash
# ============================================================================
# run_evolution_condition_hard_w500_h128_array.slurm — the EVOLUTION condition
# (GA only, NO RL) in the HARD environment on a 500x500 world, with founder-
# genome logging ON.
#
# One of a matched TRIO that share every run detail and differ ONLY in the
# learning/evolution mechanism:
#     run_evolution_condition_hard_w500_h128_array.slurm   <- THIS FILE (GA only)
#     run_learning_condition_hard_w500_h128_array.slurm    (GA + in-life RL)
#     run_pure_rl_condition_hard_w500_h128_array.slurm     (RL only, no GA)
# Submit them one after another (each one alone fills ~100 of the 120-core
# allowance; see CORES/TIME below).
#
# Grid: 10 seeds x 10 replicates = 100 runs, one per array task (--array=0-99).
#     seeds      : 2001..2010   (TEN BRAND-NEW maps — see FRESH SEEDS below)
#     replicates : 1..10
#
# WHY REPLICATES: only the map TERRAIN is seeded (--map-seed). Founder weights
# come from an unseeded Xavier draw and GA mutation is unseeded too, so the 10
# replicates of a seed share identical terrain and differ purely by algorithmic
# chance. This condition is BIMODAL at h128 in this environment (the
# hidden-size sweep saw 5/15 runs stall low and never recover), so aggregate
# across all 100 runs AND count the stalled ones separately — the mean of a
# bimodal arm describes none of it. 10 replicates is sized for exactly that.
#
# Run details (identical across the trio)
# ---------------------------------------
#   grid        500 x 500      (the historical reference scale)
#   generations 1000           (see GENERATIONS below if you want the short form)
#   hidden_size 128            (61*128 + 6 = 7814 weights)
#   predators   60 roaming @ drain 5, patrol off   = the HARD environment
#   food        x1 per reference cell (the historical 500x500 density)
#   buffer      100            (recorded only; PureRLManager's knob, inert here)
#
# GENOME LOGGING (--log-genomes)
# ------------------------------
# Writes genome.csv: the centroid + fittest founder genome EVERY generation, and
# EVERY founder's genome every 250 generations (gens 250/500/750/1000 at the
# default length — Logger.FULL_GENOME_DUMP_EVERY_N_GENS). Genomes are the
# inherited genome_weights at full float32 precision, Base64-encoded. This is
# the fitness-landscape source, and it is why this condition and the learning
# condition are worth the disk (see DISK below); pure RL cannot produce it at
# all, since PureRLManager keeps no founder list for Logger.logFounderGenomes.
#
# CORES / TIME / MEMORY (measured, not assumed)
# ----------------------------------------------
# From the 15 existing h128 evolution runs in THIS environment at THIS size and
# 1000 generations: median 6.3h wall clock, max 13.5h. --time=30:00:00 is ~2.2x
# the worst observed, which is the margin this arm needs because a run that
# takes off carries a large population for the rest of its generations and costs
# several times one that stalls. Size the limit off the MAX, not the median.
# 8G matches the hidden-size sweep, which ran this exact configuration.
#
# 100 tasks at one run each is the most parallel layout and fits the 120-core
# allowance. To trade wall clock for cores, pack:
#     RUNS_PER_JOB=1 -> --array=0-99   100 tasks  ~6h  (worst ~14h)  <- DEFAULT
#     RUNS_PER_JOB=2 -> --array=0-49    50 tasks ~13h  (worst ~27h)  fits 30h
#     RUNS_PER_JOB=4 -> --array=0-24    25 tasks ~25h  (worst ~54h)  RAISE --time
# Indices past the end of the grid are skipped, so an over-long array is harmless.
#
# DISK: ~135-205 MB per run (organisms.csv 33-105 MB + genome.csv ~100 MB at
# h128/1000 gens), so this array writes roughly 15-20 GB. Check your scratch
# quota before submitting the trio — all three together are ~45 GB.
#
#     sbatch run_evolution_condition_hard_w500_h128_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Each run writes CSVs + params.txt to
#   logs/evolution/standard/auto-run/roam60d5_w500_evolution_h128_seed<S>_r<N>/
# The _r<N> replicate suffix (deliberately NOT _rep<N>) keeps these out of the
# condition-consistency analysis, which matches runs by params and requires
# _rep<N>. A forced --run-name is not collision-checked and Logger auto-save
# APPENDS, so a unique name per run is required to avoid doubled rows.
#
# FRESH SEEDS (2001..2010) — deliberately disjoint from every earlier run
# ------------------------------------------------------------------------
# Every run under logs*/ so far used map_seed 1, 42, 123, 456 or 999, and the
# older job scripts additionally name 7/13/21/55/88/314. NONE of those appear
# here: all ten maps are new, so this trio is measured on terrain that no earlier
# tuning decision was ever fitted to. Sequential seed numbers are safe — the
# generator mixes the seed through mulberry32, so 2001 and 2002 give unrelated
# terrain rather than neighbouring variants of one map.
#
# MAP-SEED note: headless.js builds each world deterministically from --map-seed
# and caches a brand-new seed's pool atomically to
# src/maps/map_pool_seed<SEED>.json (temp-write + rename), so the many tasks that
# hit these ten brand-new seeds at once cannot corrupt them. No pre-generation is
# needed: whichever condition you submit FIRST creates all ten pools, and the
# other two then reuse the SAME cached terrain — which is what makes the three
# conditions comparable. To generate them up front instead (optional, avoids
# every task racing to build the same pool on the first submission):
#     for s in 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010; do \
#         node generate_map_pool.js --seed $s; done
#
#   GENERATIONS=500 sbatch run_evolution_condition_hard_w500_h128_array.slurm
#   N_REPS=5 sbatch --array=0-49 ...                       # 50 runs
#   SEEDS="2001 2002 2003" sbatch --array=0-29 ...         # 3 seeds x 10 reps
# ============================================================================
#SBATCH --job-name="evo_hard_w500_h128"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-99
#SBATCH --output=logs/evo_hard_w500_h128_%A_%a.out
#SBATCH --error=logs/evo_hard_w500_h128_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
# Run from wherever you submit: cd into your SCRATCH clone, then `sbatch ...`.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/LifeEngine}}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

# ── Sweep grid (2 factors: seed x replicate) ────────────────────────────────
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-2001 2002 2003 2004 2005 2006 2007 2008 2009 2010})   # 10 fresh maps
N_SEEDS=${#SEEDS[@]}
N_REPS="${N_REPS:-10}"               # independent replicate runs per seed
TOTAL_RUNS=$(( N_SEEDS * N_REPS ))   # 100

# ── Core packing: how many runs each array task executes back-to-back ────────
# CHANGING THIS REQUIRES updating the #SBATCH --array line (see header table).
RUNS_PER_JOB="${RUNS_PER_JOB:-1}"

# ── Fixed factors (identical for every run AND every condition in the trio) ─
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-1000}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
COLLAPSE_BUFFER="${COLLAPSE_BUFFER:-100}"    # recorded only; inert outside pure_rl
# Hard environment: 60 roaming predators at drain 5 on 500x500 (2.4e-4 per cell).
ROAMING_PREDATORS="${ROAMING_PREDATORS:-60}"
PREDATOR_DRAIN="${PREDATOR_DRAIN:-5}"
# 1 = one runtime food cell per REFERENCE food cell, i.e. the historical 500x500
# density. (The 1000x1000 runs pass 4 to hold that density at 4x the area.)
FOOD_DENSITY_SCALE="${FOOD_DENSITY_SCALE:-1}"
# NOTE: no learning rate / epsilon here — the evolution condition runs no RL at
# all, so those flags would be recorded in params.txt but have no effect.

# ── Preflight ───────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR" logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

# An older clone without --hidden-size silently IGNORES the flag and runs at the
# default 64, which would make this a width-64 job wearing h128 folder names.
if ! grep -q "'hidden-size'" src/headless.js 2>/dev/null; then
    echo "ERROR: this clone has no --hidden-size flag in src/headless.js. Pull the"
    echo "       current code before submitting — otherwise every run silently"
    echo "       uses the default width of 64."
    exit 1
fi

# Likewise --log-genomes: without it the runs finish fine but write NO genome.csv,
# which is the whole reason this arm costs what it costs. Fail up front.
if ! grep -q "'log-genomes'" src/headless.js 2>/dev/null; then
    echo "ERROR: this clone has no --log-genomes flag in src/headless.js. Pull the"
    echo "       current code — otherwise no genome.csv is written."
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
    local RUN_NAME="${PREFIX}_w500_evolution_h${HIDDEN_SIZE}_seed${MAP_SEED}_r${REP}"

    export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${GIDX}"

    echo "[run g$GIDX] EVOLUTION (GA only, no RL)  env=hard  seed=$MAP_SEED  rep=$REP/$N_REPS  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
    echo "[run g$GIDX]   hidden=$HIDDEN_SIZE ($(( 61 * HIDDEN_SIZE + 6 )) weights);  predators: roaming=$ROAMING_PREDATORS  drain=$PREDATOR_DRAIN  patrol=0;  genome-logging=ON"
    "$NODE_BIN" src/headless.js \
        --map-seed "$MAP_SEED" \
        --condition evolution \
        --mode standard \
        --generations "$GENERATIONS" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --run-name "$RUN_NAME" \
        --log-dir "$LOG_DIR" \
        --log-genomes \
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
