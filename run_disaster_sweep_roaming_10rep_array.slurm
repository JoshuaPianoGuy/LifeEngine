#!/bin/bash
# ============================================================================
# run_disaster_sweep_roaming_10rep_array.slurm — natural-disaster CULLING-
# FRACTION sweep in the ROAMING-PREDATOR environment, EVOLUTION condition.
# 10-REPLICATES-PER-SEED variant.
#
# Same design as run_disaster_sweep_roaming_array.slurm / _10seed_array.slurm
# (a FIXED fraction of the selection pool is culled fitness-blind on the
# generations a strike fires, with roaming predators keeping constant predation
# pressure). The ONLY change here is the REPLICATE COUNT: each (fraction, seed)
# cell is run 10 INDEPENDENT times instead of once, to measure the run-to-run
# stochasticity of the disaster response on a fixed map.
#
#     5 culling fractions x 3 seeds x 10 replicates = 150 runs.
#
# Why replicates work as a stochasticity probe (cf. the consistency runs):
#   - Only the map TERRAIN is seeded (--map-seed) and the disaster TIMING is
#     seeded (--disaster-seed, shared) so strikes stay aligned across the sweep.
#     GA mutation and predator wandering are UNSEEDED, so the 10 replicates of a
#     (fraction, seed) cell share identical terrain AND identical strike timing
#     but differ purely by algorithmic/predator chance — their spread is the
#     run-to-run noise of the disaster response.
#
# CORE PACKING (to stay under the simultaneous-core limit)
# --------------------------------------------------------
# 150 runs as 150 array tasks would ask for 150 cores at once. Instead each array
# task runs RUNS_PER_JOB runs back-to-back on its single core. With RUNS_PER_JOB=2
# the 150 runs pack into ceil(150/2)=75 tasks -> 75 cores in flight. Two evolution
# 1000-gen runs fit comfortably in the 45h wall-time (three may not — hence 2).
#
#   RUNS_PER_JOB (env, default 2) -> set --array to  ceil(150 / RUNS_PER_JOB) - 1:
#       RUNS_PER_JOB=1 -> --array=0-149  (150 cores; no packing)
#       RUNS_PER_JOB=2 -> --array=0-74   (75 cores)    <- CURRENT DEFAULT
#       RUNS_PER_JOB=3 -> --array=0-49   (50 cores)
#       RUNS_PER_JOB=5 -> --array=0-29   (30 cores)
#   If you change RUNS_PER_JOB you MUST update the #SBATCH --array line to match.
#   Runs beyond index 149 are skipped, so an over-long array is harmless.
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Grid: 5 culling fractions x 3 seeds x 10 replicates = 150 runs.
#     culling fraction : 0.2, 0.3, 0.4, 0.5, 0.6
#     seeds            : 1, 42, 999   (cached pools already exist)
#     replicates       : 1..10
# Evolution condition, 500x500 grid, 60 roaming predators with drain 5 (patrol
# predators disabled).
#
# Each run writes CSVs + params.txt to
#   logs/evolution/standard/auto-run/roam60d5_10r_evo_cull<F>_seed<S>_r<N>/
# The roam60d5_10r_ prefix + _r<N> replicate suffix keep these folders distinct
# from the 3-seed (roam60d5_evo_cull<F>_seed*) and 10-seed (roam60d5_10s_...)
# sweeps, and the _r<N> suffix (deliberately NOT _rep<N>) keeps them out of the
# condition-consistency analysis (which matches evolution/roaming runs by params
# and requires a _rep<N> suffix). A forced --run-name is not collision-checked and
# Logger auto-save APPENDS, so a unique name per run is required to avoid doubled
# rows.
#
#   GENERATIONS=2000 sbatch run_disaster_sweep_roaming_10rep_array.slurm   # override
#   RUNS_PER_JOB=3 sbatch --array=0-49 ...                                 # denser packing
#   ROAMING_PREDATORS=45 PREDATOR_DRAIN=3 sbatch ...                       # override env
# ============================================================================
#SBATCH --job-name="disaster_sweep_roaming_10rep"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-74
#SBATCH --output=logs/disaster_sweep_roaming_10rep_%A_%a.out
#SBATCH --error=logs/disaster_sweep_roaming_10rep_%A_%a.err
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

# ── Sweep grid (3 factors: fraction x seed x replicate) ─────────────────────
CULL_FRACTIONS=(0.2 0.3 0.4 0.5 0.6)          # fraction of the selection pool culled per strike
SEEDS=(1 42 999)                              # cached maps (pools already exist)
N_SEEDS=${#SEEDS[@]}
N_REPS="${N_REPS:-10}"                          # independent replicate runs per (fraction, seed)
PER_FRAC=$(( N_SEEDS * N_REPS ))               # runs per culling fraction (30)
TOTAL_RUNS=$(( ${#CULL_FRACTIONS[@]} * PER_FRAC ))   # 150

# ── Core packing: how many runs each array task executes back-to-back ────────
# CHANGING THIS REQUIRES updating the #SBATCH --array line (see header table).
RUNS_PER_JOB="${RUNS_PER_JOB:-2}"

# ── Fixed factors (identical for every run) ──────────────────────────────────
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-1000}"
# Roaming-predator environment: 60 predators, drain 5, patrol predators off.
ROAMING_PREDATORS="${ROAMING_PREDATORS:-60}"
PREDATOR_DRAIN="${PREDATOR_DRAIN:-5}"

# Disaster controls shared by every run.
#   DISASTER_SEED : ONE seed for all runs so strikes are aligned across the sweep.
#   DISASTER_PROB : per-generation chance a strike fires (headless default 0.1).
#   DISASTER_COOLDOWN : min generations between strikes.
DISASTER_SEED="${DISASTER_SEED:-777}"
DISASTER_PROB="${DISASTER_PROB:-0.1}"
DISASTER_COOLDOWN="${DISASTER_COOLDOWN:-100}"

# ── Preflight ────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR" logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

# ── One run, given a global run index ────────────────────────────────────────
# Decomposes the index into (fraction, seed, replicate). Layout is fraction-major,
# then seed, then replicate (replicate varies fastest):
#   g = FRAC_IDX*(N_SEEDS*N_REPS) + SEED_IDX*N_REPS + REP_IDX
run_one() {
    local GIDX="$1"
    local FRAC_IDX=$(( GIDX / PER_FRAC ))
    local REM=$((      GIDX % PER_FRAC ))
    local SEED_IDX=$(( REM / N_REPS ))
    local REP=$((      REM % N_REPS + 1 ))     # replicate number, 1..N_REPS
    local CULL_FRACTION="${CULL_FRACTIONS[$FRAC_IDX]}"
    local MAP_SEED="${SEEDS[$SEED_IDX]}"
    local RUN_NAME="roam60d5_10r_evo_cull${CULL_FRACTION}_seed${MAP_SEED}_r${REP}"

    export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${GIDX}"

    echo "[run g$GIDX] evolution  cull=$CULL_FRACTION  seed=$MAP_SEED  rep=$REP/$N_REPS  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
    echo "[run g$GIDX]   predators: roaming=$ROAMING_PREDATORS drain=$PREDATOR_DRAIN patrol=0  disaster: prob=$DISASTER_PROB cooldown=$DISASTER_COOLDOWN seed=$DISASTER_SEED (shared)"
    "$NODE_BIN" src/headless.js \
        --map-seed "$MAP_SEED" \
        --condition evolution \
        --mode standard \
        --generations "$GENERATIONS" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --run-name "$RUN_NAME" \
        --log-dir "$LOG_DIR" \
        --disaster \
        --disaster-prob "$DISASTER_PROB" \
        --disaster-fraction "$CULL_FRACTION" \
        --disaster-cooldown "$DISASTER_COOLDOWN" \
        --disaster-seed "$DISASTER_SEED" \
        --predators-per-patch 0 \
        --roaming-predators "$ROAMING_PREDATORS" \
        --predator-drain "$PREDATOR_DRAIN"
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
    # Don't let one failed run abort the rest of the job's packed runs.
    run_one "$GIDX" || { echo "[array] WARNING: run g$GIDX exited non-zero"; fail=1; }
done

echo "[array] task=$SLURM_ARRAY_TASK_ID complete (fail=$fail)"
exit $fail
