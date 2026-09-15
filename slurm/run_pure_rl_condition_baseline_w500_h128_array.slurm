#!/bin/bash
# ============================================================================
# run_pure_rl_condition_baseline_w500_h128_array.slurm — the PURE-RL condition
# (in-lifetime REINFORCE ONLY, NO genetic algorithm; --mode pure_rl, no epsilon,
# lr 0.01) in the PREDATOR-FREE BASELINE environment on a 500x500 world.
#
# One of a matched TRIO that share every run detail and differ ONLY in the
# learning/evolution mechanism:
#     run_evolution_condition_baseline_w500_h128_array.slurm  (GA only)
#     run_learning_condition_baseline_w500_h128_array.slurm   (GA + in-life RL)
#     run_pure_rl_condition_baseline_w500_h128_array.slurm    <- THIS FILE (RL only)
# --mode pure_rl swaps GAManager for PureRLManager: agents learn by REINFORCE
# within their lifetime, but there is NO between-generation genetic selection.
# The RL settings are deliberately IDENTICAL to the learning arm, so the only
# difference between the two is whether the GA runs.
#
# This trio is the predator-free counterpart of the hard-environment trio
#     run_{evolution,learning,pure_rl}_condition_hard_w500_h128_array.slurm
# and is deliberately identical to it in EVERY respect except the environment:
# same grid, same generations, same hidden size, same seeds, same replicate
# count. Two things change, and only two:
#
#   1. PREDATORS OFF   --roaming-predators 0 --predators-per-patch 0
#                      (--predator-drain is inert with no predators; 1 is passed
#                      so params.json records the same value as BrowserPreset's
#                      baseline row rather than the hard arm's 5)
#   2. LEARNING RATE   0.01, not 0.02 — the baseline environment's own LR-sweep
#                      optimum (EXPERIMENTS.md 5.1). Matching the learning arm's
#                      LR is what keeps "pure RL vs learning" a clean test of the
#                      GA rather than a test of the step size.
#
# Grid: 10 seeds x 10 replicates = 100 runs, one per array task (--array=0-99).
#     seeds      : 2001..2010   (the SAME ten maps as the hard trio — see SEEDS)
#     replicates : 1..10
#
# WHY REPLICATES: only the map TERRAIN is seeded (--map-seed). Initial weights,
# the RL policy's own sampling and the between-episode mutation are all
# UNSEEDED, so the 10 replicates of a seed share identical terrain and differ
# purely by algorithmic chance. Aggregate across all 100 runs.
#
# Run details (identical across the trio AND to the hard trio except predators)
# ----------------------------------------------------------------------------
#   grid        500 x 500      (the historical reference scale)
#   generations 1000           (episode WINDOWS here, not GA generations)
#   hidden_size 128            (61*128 + 6 = 7814 weights)
#   predators   NONE           roaming 0, patrol 0  = the BASELINE environment
#   food        x1 per reference cell (the historical 500x500 density)
#   buffer      100            (ACTIVE here: the rolling dead-weight buffer that
#                               PureRLManager samples to respawn after a collapse)
#
# NO GENOME LOGGING HERE — AND IT IS NOT AN OVERSIGHT
# ----------------------------------------------------
# --log-genomes is deliberately NOT passed. Logger.logFounderGenomes() returns
# early unless ga.founders is a non-empty array, and PureRLManager has no
# founders field at all (it tracks all_agents_this_window instead, since there
# is no GA selection step and its "generations" are episode windows). Passing
# the flag would cost nothing and produce nothing: no genome.csv, no rows.
# If founder genomes are ever needed for this condition, PureRLManager must
# first expose a founding cohort as this.founders — and note the semantics would
# differ from the other two arms, because pure RL syncs the learned
# active_weights INTO genome_weights at episode boundaries, so its "founder
# genome" is post-learning rather than pre-learning.
#
# CORES / TIME / MEMORY (measured, then scaled — read before changing --time)
# ---------------------------------------------------------------------------
# The hard arm of this exact configuration is now finished, so its cost is
# MEASURED over 100 runs (sum of generations.csv wall_clock_s):
#     pure RL hard h128 : median 11.9h, p90 15.0h, max 18.7h
# The baseline/hard cost ratio for this condition is measured on the h64
# landscape runs (500x500, 1000 generations, same three conditions):
#     pure RL baseline h64 median 13.2h  /  hard h64 median 6.7h  =  x1.98
# This is the LARGEST ratio of the trio. Cost is linear in population, and this
# condition carries the smallest population in the hard environment (no GA
# lifting the cohort), so removing the predators has the most room to grow it —
# mean total_agents 907 baseline vs 380 hard at h64.
#     projected baseline h128: median ~24h, worst ~37h
# --time=60:00:00 is ~1.6x the projected worst case, and the projection is a
# cross-width extrapolation, hence the margin. This is still the cheapest arm.
#
# 8G is retained deliberately and is NOT a leftover. Memory tracks the PEAK
# CONCURRENT population, not the cumulative one, and that barely moves between
# environments: max peak_population is 540 across the 100 hard h128 runs vs 616
# across the h64 baseline runs. It is the cumulative agent-ticks — hence the
# wall clock — that doubles, not the footprint.
#
# 100 tasks at one run each is the most parallel layout and fits the 120-core
# allowance. To trade wall clock for cores, pack:
#     RUNS_PER_JOB=1 -> --array=0-99   100 tasks ~24h  (worst ~37h)  <- DEFAULT
#     RUNS_PER_JOB=2 -> --array=0-49    50 tasks ~47h  (worst ~74h)  RAISE --time
#     RUNS_PER_JOB=3 -> --array=0-33    34 tasks ~71h  (worst ~111h) RAISE --time
# Indices past the end of the grid are skipped, so an over-long array is harmless.
#
# DISK: the hard array wrote 81 MB per run (organisms.csv only — no genome.csv
# in this condition), and organisms.csv scales with population, so expect the
# same ~2x: ~170 MB per run and roughly 17 GB for this array. Still the cheapest
# of the trio, but the baseline trio together is ~75 GB — MORE than the hard
# trio's 43 GB. Check your scratch quota before submitting.
#
#     sbatch run_pure_rl_condition_baseline_w500_h128_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Each run writes CSVs + params.txt to
#   logs/learning/pure_rl/auto-run/base_w500_purerl_h128_noeps_lr0.01_seed<S>_r<N>/
# (pure_rl is a distinct MODE sub-tree under logs/learning/, so there is no clash
# with the learning condition's logs/learning/standard/ runs.) The base_ prefix
# (vs the hard trio's roam60d5_) keeps the two environments apart in one logs
# tree, and matches the hidden-size sweep's existing {base,roam60d5} convention.
# The _r<N> replicate suffix (deliberately NOT _rep<N>) keeps these out of the
# condition-consistency analysis. A forced --run-name is not collision-checked
# and Logger auto-save APPENDS, so a unique name per run is required to avoid
# doubled rows.
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
#   GENERATIONS=500 sbatch run_pure_rl_condition_baseline_w500_h128_array.slurm
#   COLLAPSE_BUFFER=25 sbatch ...                          # the older buffer size
#   N_REPS=5 sbatch --array=0-49 ...                       # 50 runs
# ============================================================================
#SBATCH --job-name="purerl_base_w500_h128"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-99
#SBATCH --output=logs/purerl_base_w500_h128_%A_%a.out
#SBATCH --error=logs/purerl_base_w500_h128_%A_%a.err
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
# ACTIVE in this mode: the rolling dead-weight buffer sampled to respawn after a
# population collapse (PureRLManager). 100 is the current standing default.
COLLAPSE_BUFFER="${COLLAPSE_BUFFER:-100}"
# BASELINE environment: no predators of either kind. The drain is inert with
# zero predators; 1 matches BrowserPreset's baseline row so params.json reads
# the same in the browser preset and on the cluster.
ROAMING_PREDATORS="${ROAMING_PREDATORS:-0}"
PREDATOR_DRAIN="${PREDATOR_DRAIN:-1}"
# 1 = one runtime food cell per REFERENCE food cell, i.e. the historical 500x500
# density. (The 1000x1000 runs pass 4 to hold that density at 4x the area.)
FOOD_DENSITY_SCALE="${FOOD_DENSITY_SCALE:-1}"
# RL knobs — IDENTICAL to the learning arm, so the GA is the only difference.
# 0.01 is the BASELINE optimum; the hard trio uses 0.02 (see header).
LEARNING_RATE="${LEARNING_RATE:-0.01}"
EXPLORE_BONUS="${EXPLORE_BONUS:-0}"

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

# ── One run, given a global run index ───────────────────────────────────────
# Decomposes the index into (seed, replicate); replicate varies fastest:
#   g = SEED_IDX*N_REPS + REP_IDX
run_one() {
    local GIDX="$1"
    local SEED_IDX=$(( GIDX / N_REPS ))
    local REP=$((      GIDX % N_REPS + 1 ))    # replicate number, 1..N_REPS
    local MAP_SEED="${SEEDS[$SEED_IDX]}"
    local PREFIX="base"
    local RUN_NAME="${PREFIX}_w500_purerl_h${HIDDEN_SIZE}_noeps_lr${LEARNING_RATE}_seed${MAP_SEED}_r${REP}"

    export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${GIDX}"

    echo "[run g$GIDX] PURE RL (RL only, no GA, no epsilon)  env=baseline  seed=$MAP_SEED  rep=$REP/$N_REPS  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
    echo "[run g$GIDX]   hidden=$HIDDEN_SIZE ($(( 61 * HIDDEN_SIZE + 6 )) weights);  lr=$LEARNING_RATE  explore_bonus=$EXPLORE_BONUS  buffer=$COLLAPSE_BUFFER;  predators: roaming=$ROAMING_PREDATORS  patrol=0  (drain=$PREDATOR_DRAIN, inert);  genome-logging=N/A (no founders in pure_rl)"
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
        --hidden-size "$HIDDEN_SIZE" \
        --collapse-buffer "$COLLAPSE_BUFFER" \
        --food-density-scale "$FOOD_DENSITY_SCALE" \
        --learning-rate "$LEARNING_RATE" \
        --no-epsilon \
        --explore-bonus "$EXPLORE_BONUS" \
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

echo "[array] task=$SLURM_ARRAY_TASK_ID  runs global indices ${INDICES[*]:-none}  (RUNS_PER_JOB=$RUNS_PER_JOB, stride=$N_TASKS, of $TOTAL_RUNS total)  seeds=${SEEDS[*]}  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS  hidden=$HIDDEN_SIZE  lr=$LEARNING_RATE  env=baseline"

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
