#!/bin/bash
# ============================================================================
# run_random_floor_condition_hard_w500_h128_array.slurm — the RANDOM-POLICY
# FLOOR: the CHANCE ANCHOR for every fitness number in the experiment.
#
# This is the fourth arm of what was a trio. It is the evolution condition with
# exactly one thing removed — between-generation selection:
#     run_evolution_condition_hard_w500_h128_array.slurm      GA only
#     run_learning_condition_hard_w500_h128_array.slurm       GA + in-life RL
#     run_pure_rl_condition_hard_w500_h128_array.slurm        RL only, no GA
#     run_random_floor_condition_hard_w500_h128_array.slurm   <- THIS FILE
#
# WHY IT EXISTS
# -------------
# Without it every fitness number in the write-up is unanchored. A converged
# average fitness of 3.5 cannot be called "bad", and a trapped run cannot be
# told apart from a merely unlucky one, because nothing says what an agent that
# has learned and evolved NOTHING scores on the same maps. The trapped/escaped
# bimodality in the evolution arm only becomes an interpretable claim once it
# can be stated relative to chance: trapped runs are, or are not, above the floor.
#
# HOW IT WORKS (--random-floor)
# -----------------------------
# GAManager runs its full generation loop — 100 founders, the same anatomy, the
# same 50-map pool, the same predators, the same within-generation reproduction,
# the same fitness sort, the same tournament / crossover / mutation, the same
# logging — and then DISCARDS the gene pool it just built at the generation
# boundary, so the next generation's founders are freshly Xavier-initialised.
# RL is off (--condition evolution), so no adaptation of any kind survives a
# generation. Every other flag below is character-for-character the evolution
# arm's.
#
# It is deliberately NOT implemented as "skip selection": the selection block
# still runs, so PRNG consumption and the logged per-generation metrics stay on
# the same code path as a real run. See ExperimentParams.random_floor.
#
# READING THE OUTPUT — THE TRAP
# ------------------------------
# These runs record condition == "evolution" and mode == "standard" in
# params.json, because that is what they are minus selection. Analysis code
# matching on those two fields alone would pool them straight into the evolution
# arm and drag its mean down to chance. Two things prevent that:
#   * params.json carries random_floor: true
#   * generations.csv's `condition` column reads `random_floor`
# analyse_learning_hard_runs.discover() filters on the flag and EXCLUDES these
# by default; pass want['random_floor'] = True to select them instead.
# analyse_hard_stats.py already carries them as a fourth arm.
#
# GENERATIONS: 1000, to match the trio exactly so the floor can be drawn on the
# same axis. The floor is STATIONARY by construction, though — it has no
# mechanism by which to improve — so shortening it is a legitimate saving when
# the allocation is tight:
#     GENERATIONS=200 sbatch run_random_floor_condition_hard_w500_h128_array.slurm
# Check that stationarity rather than assuming it: a floor whose fitness trends
# upward over generations means something is leaking across the boundary, and
#     node src/eval/validate.js --only floor
# is the check for exactly that (it compares consecutive founder centroids).
#
# NO --log-genomes. The floor's genomes are Xavier noise by construction, there
# is no landscape to reconstruct from them, and it would cost ~100 MB a run.
#
# CORES / TIME / MEMORY
# ---------------------
# Cheaper than the evolution arm: with no selection the population never takes
# off, so per-generation cost stays near the low end instead of growing. Measured
# locally at 500x500 / h128 in this environment: ~5.6 s per generation, i.e.
# ~1.6 h for 1000 generations. --time=12:00:00 is ~7x that, which covers a run
# that happens to carry a large population by chance. 8G matches the trio.
#
# Grid: 10 seeds x 10 replicates = 100 runs, one per array task (--array=0-99).
#     seeds      : 2001..2010 — THE SAME TEN MAPS THE TRIO USES. The whole point
#                  of an anchor is that it is measured on identical terrain, so
#                  do not change these independently of the other three scripts.
#     replicates : 1..10
#
# DISK: ~35-110 MB per run (organisms.csv only, no genome.csv) — roughly 5-10 GB
# for the array, well under what any one arm of the trio costs.
#
#     sbatch run_random_floor_condition_hard_w500_h128_array.slurm
#
# Each run writes CSVs + params.txt to
#   logs/evolution/standard/auto-run/roam60d5_w500_randomfloor_h128_seed<S>_r<N>/
# ============================================================================
#SBATCH --job-name="floor_hard_w500_h128"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-99
#SBATCH --output=logs/floor_hard_w500_h128_%A_%a.out
#SBATCH --error=logs/floor_hard_w500_h128_%A_%a.err
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

# --random-floor is what makes this arm the floor rather than a second copy of
# the evolution arm. An older clone without it does not error on the unknown
# flag — headless.js just ignores it — so 100 runs would come back looking like
# ordinary evolution runs under floor folder names. Fail up front.
if ! grep -q "random-floor" src/headless.js 2>/dev/null; then
    echo "ERROR: this clone has no --random-floor flag in src/headless.js. Pull"
    echo "       the current code — otherwise these runs are ordinary EVOLUTION"
    echo "       runs wearing random-floor names, which is worse than no anchor."
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
    local RUN_NAME="${PREFIX}_w500_randomfloor_h${HIDDEN_SIZE}_seed${MAP_SEED}_r${REP}"

    export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${GIDX}"

    echo "[run g$GIDX] RANDOM FLOOR (no selection, no RL)  env=hard  seed=$MAP_SEED  rep=$REP/$N_REPS  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
    echo "[run g$GIDX]   hidden=$HIDDEN_SIZE ($(( 61 * HIDDEN_SIZE + 6 )) weights);  predators: roaming=$ROAMING_PREDATORS  drain=$PREDATOR_DRAIN  patrol=0;  genome-logging=OFF"
    "$NODE_BIN" src/headless.js \
        --map-seed "$MAP_SEED" \
        --condition evolution \
        --mode standard \
        --generations "$GENERATIONS" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --run-name "$RUN_NAME" \
        --log-dir "$LOG_DIR" \
        --random-floor \
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
