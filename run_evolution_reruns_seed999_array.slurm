#!/bin/bash
# ============================================================================
# run_evolution_reruns_seed999_array.slurm — extra EVOLUTION replicates on
# seed 999, both environments, WITH GENOME LOGGING, to source a low-fitness
# ("evolution struggled") run for the fitness-landscape comparison.
#
# WHY: per-seed stochasticity makes evolution converge to different levels
# across replicates on the same terrain (GA mutation + predator wandering are
# unseeded; only the map is seeded). The existing 3 replicates cluster tightly
# (roam60d5 ~15.7-16.1). Firing 10 more per environment widens the sample so the
# LOWEST-fitness run can be picked as the landscape's evolution anchor — the case
# where evolution stayed in a low basin (the interesting, rugged story). The
# ruggedness scalar (lambda) is an ENVIRONMENT property measured independently of
# any run; this only chooses which trajectory to illustrate on the surface.
#
#     2 environments x 10 replicates = 20 runs.
#         environments : baseline (no predators), roam60d5 (60 roaming, drain 5)
#         replicates   : 1..10
#     Evolution condition, 500x500 grid, 1000 generations, --log-genomes ON.
#
# CRITICAL: --log-genomes is passed on every run (off by default → no genome.csv,
# which would make the run useless for the landscape). This is the whole point.
#
# CORE PACKING
# ------------
# 20 runs as 20 array tasks = 20 cores at once. Each task instead runs
# RUNS_PER_JOB runs back-to-back on one core. RUNS_PER_JOB=2 packs 20 into
# ceil(20/2)=10 tasks → 10 cores. Two 1000-gen evolution runs fit in 45h
# (three may not — hence 2, matching the disaster-sweep scripts).
#
#   RUNS_PER_JOB (env, default 2) → set --array to ceil(20/RUNS_PER_JOB)-1:
#       RUNS_PER_JOB=1 → --array=0-19  (20 cores; no packing)
#       RUNS_PER_JOB=2 → --array=0-9   (10 cores)     <- CURRENT DEFAULT
#   If you change RUNS_PER_JOB you MUST update the #SBATCH --array line to match.
#   Indices beyond 19 are skipped, so an over-long array is harmless.
#
# (sbatch SCHEDULES this file — you do not execute it yourself.)
#
# Each run writes CSVs (incl. genome.csv) + params.txt to
#   logs/evolution/standard/auto-run/<ENV>_g1k_evolution_lscape_seed999_rerun<N>/
# The _rerun<N> suffix keeps these distinct from the original _r{1,2,3} runs and
# out of the _rep<N> condition-consistency matcher. Logger auto-save APPENDS, so a
# unique forced --run-name per run is required to avoid doubled rows.
#
#   GENERATIONS=2000 sbatch run_evolution_reruns_seed999_array.slurm   # override gens
#   N_REPS=5 sbatch --array=0-4 run_evolution_reruns_seed999_array.slurm  # fewer reps
# ============================================================================
#SBATCH --job-name="evo_reruns_seed999"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-9
#SBATCH --output=logs/evo_reruns_seed999_%A_%a.out
#SBATCH --error=logs/evo_reruns_seed999_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/LifeEngine}}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

# ── Sweep grid (2 environments x replicates) ────────────────────────────────
ENV_NAMES=(baseline roam60d5)         # environment key (also the run-name prefix)
ENV_ROAMING=(0 60)                    # roaming predators per environment
# Predator drain per environment. baseline=1 is INERT — with roaming=0 and
# patrol=0 there are no predators, so drainAmount is never read; 1 is used only
# to match the existing baseline runs' params.json (predator_drain: 1). Only
# roam60d5's drain=5 actually affects a simulation.
ENV_DRAIN=(1 5)
N_ENVS=${#ENV_NAMES[@]}
N_REPS="${N_REPS:-10}"                 # independent replicate runs per environment
TOTAL_RUNS=$(( N_ENVS * N_REPS ))      # 20

# ── Core packing ─────────────────────────────────────────────────────────────
RUNS_PER_JOB="${RUNS_PER_JOB:-2}"

# ── Fixed factors (identical for every run) ──────────────────────────────────
MAP_SEED="${MAP_SEED:-999}"            # cached pool map_pool_seed999.json exists
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-1000}"

# ── Preflight ────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR" logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

# ── One run, given a global run index ────────────────────────────────────────
# Layout is environment-major, replicate varies fastest:
#   g = ENV_IDX*N_REPS + REP_IDX      (REP_IDX 0..N_REPS-1)
run_one() {
    local GIDX="$1"
    local ENV_IDX=$(( GIDX / N_REPS ))
    local REP=$((     GIDX % N_REPS + 1 ))          # replicate number, 1..N_REPS
    local ENV_NAME="${ENV_NAMES[$ENV_IDX]}"
    local ROAMING="${ENV_ROAMING[$ENV_IDX]}"
    local DRAIN="${ENV_DRAIN[$ENV_IDX]}"
    # _g1k_ matches the existing seed-999 naming family (default 1000 gens); the
    # true generation count is always recorded in the run's params.json.
    local RUN_NAME="${ENV_NAME}_g1k_evolution_lscape_seed${MAP_SEED}_rerun${REP}"

    export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${GIDX}"

    echo "[run g$GIDX] evolution  env=$ENV_NAME  rep=$REP/$N_REPS  seed=$MAP_SEED  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
    echo "[run g$GIDX]   predators: roaming=$ROAMING drain=$DRAIN patrol=0   genome-logging=ON"
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
        --roaming-predators "$ROAMING" \
        --predator-drain "$DRAIN" \
        --predators-per-patch 0
    local rc=$?
    echo "[run g$GIDX] done (exit $rc): $RUN_NAME"
    return $rc
}

# ── Run this task's slice of RUNS_PER_JOB consecutive global indices ─────────
START=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_JOB ))
echo "[array] task=$SLURM_ARRAY_TASK_ID  runs global indices ${START}..$(( START + RUNS_PER_JOB - 1 ))  (RUNS_PER_JOB=$RUNS_PER_JOB, of $TOTAL_RUNS total)  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"

fail=0
for (( k = 0; k < RUNS_PER_JOB; k++ )); do
    GIDX=$(( START + k ))
    if [ "$GIDX" -ge "$TOTAL_RUNS" ]; then
        echo "[array] global index $GIDX >= $TOTAL_RUNS — nothing more to do."
        break
    fi
    run_one "$GIDX" || { echo "[array] WARNING: run g$GIDX exited non-zero"; fail=1; }
done

echo "[array] task=$SLURM_ARRAY_TASK_ID complete (fail=$fail)"
exit $fail
