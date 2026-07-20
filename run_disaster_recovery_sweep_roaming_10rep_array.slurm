#!/bin/bash
# ============================================================================
# run_disaster_recovery_sweep_roaming_10rep_array.slurm — GRADUAL-RECOVERY
# disaster variant sweep in the ROAMING-PREDATOR environment, EVOLUTION
# condition. 10-REPLICATES-PER-SEED variant.
#
# Same design as run_disaster_recovery_sweep_roaming_array.slurm / _10seed:
# a strike at fraction F tapers the cull back to full population over successive
# generations (F, F-rate, F-2*rate, ... 0), while roaming predators keep constant
# predation pressure. The ONLY change here is the REPLICATE COUNT: each
# (fraction, recovery, seed) cell is run 10 INDEPENDENT times instead of once, to
# measure the run-to-run stochasticity of the recovery response on a fixed map.
#
#     5 fractions x 2 recovery rates x 3 seeds x 10 replicates = 300 runs.
#
# Why replicates work as a stochasticity probe (cf. the consistency runs):
#   - Only the map TERRAIN is seeded (--map-seed) and the disaster TIMING is
#     seeded (--disaster-seed, shared; timing is independent of fraction AND
#     recovery rate — no PRNG is consumed during a taper). GA mutation and
#     predator wandering are UNSEEDED, so the 10 replicates of a cell share
#     identical terrain AND identical strike timing but differ purely by
#     algorithmic/predator chance — their spread is the run-to-run noise.
#
# CORE PACKING (to stay under the simultaneous-core limit)
# --------------------------------------------------------
# 300 runs as 300 array tasks would ask for 300 cores at once. Instead each array
# task runs RUNS_PER_JOB runs back-to-back on its single core. With RUNS_PER_JOB=3
# the 300 runs pack into ceil(300/3)=100 tasks -> 100 cores in flight (down from the
# 150 cores of the earlier RUNS_PER_JOB=2 default). THREE evolution 1000-gen runs
# per task need more wall-time than two, so --time was raised 45h -> 54h to keep a
# margin: on the header's own per-run model (two runs fit "comfortably" in 45h, so
# ~15-18h/run), three runs are ~45-54h, which would sit right at the old 45h edge.
# (If 100 concurrent cores is still over your limit, throttle with e.g. --array=0-99%50.)
#
#   RUNS_PER_JOB (env, default 3) -> set --array to  ceil(300 / RUNS_PER_JOB) - 1
#   AND size --time for RUNS_PER_JOB back-to-back runs (~18h each):
#       RUNS_PER_JOB=1 -> --array=0-299  (300 cores; no packing)   ~18h  -> --time=24:00:00
#       RUNS_PER_JOB=2 -> --array=0-149  (150 cores)               ~36h  -> --time=45:00:00
#       RUNS_PER_JOB=3 -> --array=0-99   (100 cores)  <- CURRENT   ~54h  -> --time=54:00:00
#       RUNS_PER_JOB=5 -> --array=0-59   (60 cores)                ~90h  -> --time=90:00:00
#       RUNS_PER_JOB=6 -> --array=0-49   (50 cores)                ~108h -> --time=108:00:00
#   If you change RUNS_PER_JOB you MUST update the #SBATCH --array AND --time lines
#   to match. Runs beyond index 299 are skipped, so an over-long array is harmless.
#   If the ada partition caps wall-time below what a given packing needs, sbatch will
#   reject it — drop back to a smaller RUNS_PER_JOB, or lower GENERATIONS, in that case.
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# Grid: 5 fractions x 2 recovery rates x 3 seeds x 10 replicates = 300 runs.
#     starting fraction : 0.2, 0.3, 0.4, 0.5, 0.6
#     recovery rate     : 0.05, 0.10   (fraction shed per generation)
#     seeds             : 1, 42, 999   (cached pools already exist)
#     replicates        : 1..10
# Evolution condition, 500x500 grid, 60 roaming predators with drain 5 (patrol
# predators disabled).
#
# Each run writes CSVs + params.txt to
#   logs/evolution/standard/auto-run/roam60d5_10r_evo_cull<F>_rec<R>_seed<S>_r<N>/
# The roam60d5_10r_ prefix + _r<N> replicate suffix keep these folders distinct
# from the 3-seed (roam60d5_evo_cull<F>_rec<R>_seed*) and 10-seed (roam60d5_10s_...)
# sweeps, and the _r<N> suffix (deliberately NOT _rep<N>) keeps them out of the
# condition-consistency analysis (which matches evolution/roaming runs by params
# and requires a _rep<N> suffix). A forced --run-name is not collision-checked and
# Logger auto-save APPENDS, so a unique name per run is required to avoid doubled
# rows.
#
#   GENERATIONS=2000 sbatch run_disaster_recovery_sweep_roaming_10rep_array.slurm  # override
#   RUNS_PER_JOB=6 sbatch --array=0-49 --time=108:00:00 ...                        # denser packing
#   sbatch --array=0-99%50 ...                                                     # throttle concurrency
#   ROAMING_PREDATORS=45 PREDATOR_DRAIN=3 sbatch ...                               # override env
# ============================================================================
#SBATCH --job-name="disaster_recovery_sweep_roaming_10rep"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=54:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-99
#SBATCH --output=logs/disaster_recovery_sweep_roaming_10rep_%A_%a.out
#SBATCH --error=logs/disaster_recovery_sweep_roaming_10rep_%A_%a.err
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

# ── Sweep grid (4 factors: fraction x recovery x seed x replicate) ──────────
CULL_FRACTIONS=(0.2 0.3 0.4 0.5 0.6)          # STARTING fraction culled on the strike generation
RECOVERY_RATES=(0.05 0.10)                    # fraction shed per generation as the cull tapers
SEEDS=(1 42 999)                              # cached maps (pools already exist)
N_SEEDS=${#SEEDS[@]}
N_REC=${#RECOVERY_RATES[@]}
N_REPS="${N_REPS:-10}"                          # independent replicate runs per cell
PER_REC=$((  N_SEEDS * N_REPS ))               # runs per (fraction, recovery) block (30)
PER_FRAC=$(( N_REC * PER_REC ))                # runs per fraction (60)
TOTAL_RUNS=$(( ${#CULL_FRACTIONS[@]} * PER_FRAC ))   # 300

# ── Core packing: how many runs each array task executes back-to-back ────────
# CHANGING THIS REQUIRES updating the #SBATCH --array line (see header table).
RUNS_PER_JOB="${RUNS_PER_JOB:-3}"

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
#   DISASTER_COOLDOWN : min generations between strikes (from the taper's last gen).
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
# Decomposes the index into (fraction, recovery, seed, replicate). Layout is
# fraction-major, then recovery, then seed, then replicate (replicate fastest):
#   g = FRAC_IDX*(N_REC*N_SEEDS*N_REPS) + REC_IDX*(N_SEEDS*N_REPS)
#         + SEED_IDX*N_REPS + REP_IDX
run_one() {
    local GIDX="$1"
    local FRAC_IDX=$(( GIDX / PER_FRAC ))
    local REM=$((      GIDX % PER_FRAC ))
    local REC_IDX=$((  REM / PER_REC ))
    local REM2=$((     REM % PER_REC ))
    local SEED_IDX=$(( REM2 / N_REPS ))
    local REP=$((      REM2 % N_REPS + 1 ))    # replicate number, 1..N_REPS
    local CULL_FRACTION="${CULL_FRACTIONS[$FRAC_IDX]}"
    local RECOVERY_RATE="${RECOVERY_RATES[$REC_IDX]}"
    local MAP_SEED="${SEEDS[$SEED_IDX]}"
    local RUN_NAME="roam60d5_10r_evo_cull${CULL_FRACTION}_rec${RECOVERY_RATE}_seed${MAP_SEED}_r${REP}"

    export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${GIDX}"

    echo "[run g$GIDX] evolution  cull=$CULL_FRACTION  recovery=$RECOVERY_RATE  seed=$MAP_SEED  rep=$REP/$N_REPS  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"
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
        --disaster-recovery-rate "$RECOVERY_RATE" \
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
