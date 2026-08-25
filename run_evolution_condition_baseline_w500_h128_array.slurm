#!/bin/bash
# ============================================================================
# run_evolution_condition_baseline_w500_h128_array.slurm — the EVOLUTION
# condition (GA only, NO RL) in the PREDATOR-FREE BASELINE environment on a
# 500x500 world, with founder-genome logging ON.
#
# One of a matched TRIO that share every run detail and differ ONLY in the
# learning/evolution mechanism:
#     run_evolution_condition_baseline_w500_h128_array.slurm  <- THIS FILE (GA only)
#     run_learning_condition_baseline_w500_h128_array.slurm   (GA + in-life RL)
#     run_pure_rl_condition_baseline_w500_h128_array.slurm    (RL only, no GA)
# Submit them one after another (each one alone fills ~100 of the 120-core
# allowance; see CORES/TIME below).
#
# This trio is the predator-free counterpart of the hard-environment trio
#     run_{evolution,learning,pure_rl}_condition_hard_w500_h128_array.slurm
# and is deliberately identical to it in EVERY respect except the environment:
# same grid, same generations, same hidden size, same seeds, same replicate
# count, same genome logging. Two things change, and only two:
#
#   1. PREDATORS OFF   --roaming-predators 0 --predators-per-patch 0
#                      (--predator-drain is inert with no predators; 1 is passed
#                      so params.json records the same value as BrowserPreset's
#                      baseline row rather than the hard arm's 5)
#   2. LEARNING RATE   0.01, not 0.02 — the baseline environment's own LR-sweep
#                      optimum (EXPERIMENTS.md 5.1). This arm runs no RL at all
#                      so it is unaffected, but the other two arms of this trio
#                      use 0.01 and the difference matters when reading them
#                      beside the hard trio.
#
# Grid: 10 seeds x 10 replicates = 100 runs, one per array task (--array=0-99).
#     seeds      : 2001..2010   (the SAME ten maps as the hard trio — see SEEDS)
#     replicates : 1..10
#
# WHY REPLICATES: only the map TERRAIN is seeded (--map-seed). Founder weights
# come from an unseeded Xavier draw and GA mutation is unseeded too, so the 10
# replicates of a seed share identical terrain and differ purely by algorithmic
# chance. The hard arm of this condition is BIMODAL at h128 (10 of its 100 runs
# never cleared the 4.375 replacement level, finishing at 3.60-3.97 against a
# 6.65 arm mean); whether the predator-free arm is
# unimodal is one of the things this job answers, so keep the replicate count
# and count any stalled runs separately rather than reporting only a mean.
#
# Run details (identical across the trio AND to the hard trio except predators)
# ----------------------------------------------------------------------------
#   grid        500 x 500      (the historical reference scale)
#   generations 1000           (see GENERATIONS below if you want the short form)
#   hidden_size 128            (61*128 + 6 = 7814 weights)
#   predators   NONE           roaming 0, patrol 0  = the BASELINE environment
#   food        x1 per reference cell (the historical 500x500 density)
#   buffer      100            (recorded only; PureRLManager's knob, inert here)
#
# GENOME LOGGING (--log-genomes)
# ------------------------------
# Writes genome.csv: the centroid + fittest founder genome EVERY generation, and
# EVERY founder's genome every 250 generations (gens 250/500/750/1000 at the
# default length — Logger.FULL_GENOME_DUMP_EVERY_N_GENS). Genomes are the
# inherited genome_weights at full float32 precision, Base64-encoded. This is
# the fitness-landscape source: landscape/make_slice_jobs.py reads these run
# directories directly, so keeping the flag on is what makes a baseline
# landscape-slice comparison possible at all.
#
# CORES / TIME / MEMORY (measured, then scaled — read before changing --time)
# ---------------------------------------------------------------------------
# The hard arm of this exact configuration is now finished, so its cost is
# MEASURED over 100 runs (sum of generations.csv wall_clock_s):
#     evolution hard h128 : median 12.0h, p90 22.1h, max 25.3h
# The baseline/hard cost ratio for this condition is measured on the h64
# landscape runs (500x500, 1000 generations, same three conditions):
#     evolution baseline h64 median 13.1h  /  hard h64 median 7.0h  =  x1.86
# Cost is linear in population (EXPERIMENTS.md 5.2b) and that is exactly what
# the ratio reflects: with no predators the surviving population is roughly
# double (mean total_agents 1248 baseline vs 592 hard at h64).
#     projected baseline h128: median ~22h, worst ~47h
# --time=72:00:00 is ~1.5x the projected worst case, and the projection itself
# is a cross-width extrapolation, hence the wider margin than the hard arm's.
#
# 8G is retained deliberately and is NOT a leftover. Memory tracks the PEAK
# CONCURRENT population, not the cumulative one, and that barely moves between
# environments: max peak_population is 723 across the 100 hard h128 runs vs 828
# across the h64 baseline runs. It is the cumulative agent-ticks — hence the
# wall clock — that doubles, not the footprint.
#
# 100 tasks at one run each is the most parallel layout and fits the 120-core
# allowance. To trade wall clock for cores, pack:
#     RUNS_PER_JOB=1 -> --array=0-99   100 tasks ~22h  (worst ~47h)  <- DEFAULT
#     RUNS_PER_JOB=2 -> --array=0-49    50 tasks ~45h  (worst ~94h)  RAISE --time
#     RUNS_PER_JOB=4 -> --array=0-24    25 tasks ~89h  (worst ~189h) RAISE --time
# Indices past the end of the grid are skipped, so an over-long array is harmless.
#
# DISK: the hard array wrote 210 MB per run (genome.csv ~100 MB, fixed by
# founders x generations + organisms.csv ~120 MB, which scales with population).
# Only organisms.csv grows here, by the same ~2x: expect ~350 MB per run and
# roughly 35 GB for this array. The baseline trio together is ~75 GB — MORE than
# the hard trio's 43 GB. Check your scratch quota before submitting.
#
#     sbatch run_evolution_condition_baseline_w500_h128_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Each run writes CSVs + params.txt to
#   logs/evolution/standard/auto-run/base_w500_evolution_h128_seed<S>_r<N>/
# The base_ prefix (vs the hard trio's roam60d5_) is what keeps the two
# environments apart in one logs tree, and matches the hidden-size sweep's
# existing {base,roam60d5} convention. The _r<N> replicate suffix (deliberately
# NOT _rep<N>) keeps these out of the condition-consistency analysis, which
# matches runs by params and requires _rep<N>. A forced --run-name is not
# collision-checked and Logger auto-save APPENDS, so a unique name per run is
# required to avoid doubled rows.
#
# SEEDS 2001..2010 — THE SAME TERRAIN AS THE HARD TRIO, ON PURPOSE
# -----------------------------------------------------------------
# The hard trio introduced these ten maps as fresh terrain no earlier tuning
# decision had been fitted to. Reusing them here is what makes the comparison a
# PAIRED one: seed 2004's baseline run and seed 2004's hard run inhabit the
# identical world and differ only in whether 60 predators are in it, so the
# environment effect is not confounded with terrain. Their pools are already
# cached in src/maps/map_pool_seed<SEED>.json from the hard trio, so no map
# generation happens on this submission at all. If you are starting from a clean
# clone, headless.js regenerates each pool deterministically and atomically
# (temp-write + rename), so concurrent tasks cannot corrupt them; to pre-build:
#     for s in 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010; do \
#         node generate_map_pool.js --seed $s; done
#
#   GENERATIONS=500 sbatch run_evolution_condition_baseline_w500_h128_array.slurm
#   N_REPS=5 sbatch --array=0-49 ...                       # 50 runs
#   SEEDS="2001 2002 2003" sbatch --array=0-29 ...         # 3 seeds x 10 reps
# ============================================================================
#SBATCH --job-name="evo_base_w500_h128"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-99
#SBATCH --output=logs/evo_base_w500_h128_%A_%a.out
#SBATCH --error=logs/evo_base_w500_h128_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
# Run from wherever you submit: cd into your SCRATCH clone, then `sbatch ...`.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/LifeEngine}}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

# ── Sweep grid (2 factors: seed x replicate) ────────────────────────────────
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-2001 2002 2003 2004 2005 2006 2007 2008 2009 2010})   # same maps as the hard trio
N_SEEDS=${#SEEDS[@]}
N_REPS="${N_REPS:-10}"               # independent replicate runs per seed
TOTAL_RUNS=$(( N_SEEDS * N_REPS ))   # 100

# ── Core packing: how many runs each array task executes back-to-back ────────
# CHANGING THIS REQUIRES updating the #SBATCH --array line AND --time (see header).
RUNS_PER_JOB="${RUNS_PER_JOB:-1}"

# ── Fixed factors (identical for every run AND every condition in the trio) ─
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-1000}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
COLLAPSE_BUFFER="${COLLAPSE_BUFFER:-100}"    # recorded only; inert outside pure_rl
# BASELINE environment: no predators of either kind. The drain is inert with
# zero predators; 1 matches BrowserPreset's baseline row so params.json reads
# the same in the browser preset and on the cluster.
ROAMING_PREDATORS="${ROAMING_PREDATORS:-0}"
PREDATOR_DRAIN="${PREDATOR_DRAIN:-1}"
# 1 = one runtime food cell per REFERENCE food cell, i.e. the historical 500x500
# density. (The 1000x1000 runs pass 4 to hold that density at 4x the area.)
FOOD_DENSITY_SCALE="${FOOD_DENSITY_SCALE:-1}"
# NOTE: no learning rate / epsilon here — the evolution condition runs no RL at
# all, so those flags would be recorded in params.txt but have no effect. The
# other two arms of this trio use lr 0.01 (the baseline optimum).

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
    local PREFIX="base"
    local RUN_NAME="${PREFIX}_w500_evolution_h${HIDDEN_SIZE}_seed${MAP_SEED}_r${REP}"

    export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${GIDX}"

    echo "[run g$GIDX] EVOLUTION (GA only, no RL)  env=baseline  seed=$MAP_SEED  rep=$REP/$N_REPS  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
    echo "[run g$GIDX]   hidden=$HIDDEN_SIZE ($(( 61 * HIDDEN_SIZE + 6 )) weights);  predators: roaming=$ROAMING_PREDATORS  patrol=0  (drain=$PREDATOR_DRAIN, inert);  genome-logging=ON"
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

echo "[array] task=$SLURM_ARRAY_TASK_ID  runs global indices ${INDICES[*]:-none}  (RUNS_PER_JOB=$RUNS_PER_JOB, stride=$N_TASKS, of $TOTAL_RUNS total)  seeds=${SEEDS[*]}  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS  hidden=$HIDDEN_SIZE  env=baseline"

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
