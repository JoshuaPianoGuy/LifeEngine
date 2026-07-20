#!/bin/bash
# ============================================================================
# run_shuffle_condition_sweep_array.slurm — CONDITION x SHUFFLE-PERIOD sweep on
# the non-stationary "shuffle" environment (predator-free baseline where the
# food-tier -> energy-value mapping periodically reshuffles; see FoodShuffle.js).
#
# Purpose: does a shorter reshuffle period (more within-lifetime non-stationarity)
# separate the three mechanisms? Evolution can't adapt in-life, so it should
# decay as P shrinks; learning / pure RL can re-taste and should hold up better.
#
#     3 conditions x 5 periods x 5 map seeds x 3 replicates = 225 runs.
#         conditions : learning, evolution, pure_rl
#         periods P  : 300, 1000, 3000, 10000, 100000  (ticks between reshuffles)
#         seeds      : 5 terrains (3 cached + 2 auto-generated on first use)
#         replicates : 3 independent runs per (condition, period, seed) cell
#     500x500 world, 1000 generations. Predators OFF; lr 0.01 (baseline optimum).
#
# The shuffle is ENVIRONMENT-level (applied to ALL conditions): evolution eats
# under the same non-stationary payoffs, it just has no in-life mechanism to
# exploit them. The reshuffle SCHEDULE is seeded (--food-shuffle-seed = map seed)
# so, like the terrain, it is FIXED across the 3 replicates of a cell — replicate
# spread is then pure GA/RL/predator stochasticity, not schedule luck.
#
# CORE PACKING (to stay under the simultaneous-core limit, ~100, keep lower)
# -------------------------------------------------------------------------
# Each array task runs RUNS_PER_JOB runs back-to-back on its single core (same
# scheme as run_condition_consistency / run_evolution_reruns). 225 runs:
#
#   RUNS_PER_JOB (env, default 2) -> set --array to  ceil(225 / RUNS_PER_JOB) - 1:
#       RUNS_PER_JOB=1 -> --array=0-224      (225 cores; no packing)
#       RUNS_PER_JOB=2 -> --array=0-112%80   (113 tasks, throttled to 80)  <- DEFAULT
#       RUNS_PER_JOB=3 -> --array=0-74       (75 tasks; RAISE --time, 3x runtime)
#       RUNS_PER_JOB=5 -> --array=0-44       (45 tasks; RAISE --time)
#   The %80 SUFFIX throttles CONCURRENT tasks to 80 (<=100 cap) while total tasks
#   stay 113 — needed because ceil(225/2)=113 exceeds the cap on its own. Raise/
#   lower it (e.g. %64) via `sbatch --array=0-112%64 ...` to taste.
#   Two 1000-gen runs fit in the 45h wall; THREE do NOT (see run_evolution_reruns
#   note) — so RUNS_PER_JOB>=3 REQUIRES a larger --time and only helps if your
#   partition allows >45h. If you change RUNS_PER_JOB you MUST update --array.
#   Indices >= 225 are skipped, so an over-long array is harmless.
#
# MAP-SEED note: headless.js builds each world deterministically from --map-seed
# and caches it to src/maps/map_pool_seed<SEED>.json on first use (atomic write,
# parallel-safe), so no manual pre-generation is required. To avoid 45 jobs
# regenerating the same pool at t=0, optionally pre-warm once before submitting:
#   for s in 1 42 999 123 456; do node generate_map_pool.js --seed $s; done
#
# Each task writes CSVs + params.txt (records food_shuffle_period/seed) to
#   logs/<cond>/<mode>/auto-run/shufP<P>_g1k_<mech>_seed<SEED>_rep<N>/
# The shufP<P> prefix encodes the shuffle period (the environment variant), the
# way roam60d5_/baseline_ encode their environments. The _rep<N> tag keeps every
# replicate in its own folder (a forced --run-name is NOT collision-checked and
# Logger auto-save APPENDS, so a clash would double rows).
#
#   GENERATIONS=200 sbatch run_shuffle_condition_sweep_array.slurm    # quick test
#   LEARNING_RATE=0.02 sbatch ...                                     # override lr
#   RUNS_PER_JOB=3 sbatch --array=0-74 --time=90:00:00 ...            # denser pack
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
# ============================================================================
#SBATCH --job-name="shuffle_cond_sweep"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-112%80
#SBATCH --output=logs/shuffle_cond_sweep_%A_%a.out
#SBATCH --error=logs/shuffle_cond_sweep_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/LifeEngine}}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

# ── Grid (4 factors: condition x period x seed x replicate) ─────────────────
# Global run index, condition slowest -> replicate fastest:
#   g = COND_IDX*PER_COND + P_IDX*PER_PERIOD + SEED_IDX*N_REPS + REP_IDX
CONDITIONS=(learning evolution pure_rl)      # the three mechanisms
PERIODS=(300 1000 3000 10000 100000)         # ticks between food payoff reshuffles
SEEDS=(1 42 999 123 456)                     # 5 terrains (3 cached + 2 auto-generated)
N_COND=${#CONDITIONS[@]}
N_P=${#PERIODS[@]}
N_SEEDS=${#SEEDS[@]}
N_REPS="${N_REPS:-3}"                         # independent replicates per cell
PER_PERIOD=$(( N_SEEDS * N_REPS ))            # runs per (cond,period)   = 15
PER_COND=$((   N_P * PER_PERIOD ))            # runs per condition       = 75
TOTAL_RUNS=$(( N_COND * PER_COND ))           # 225

# ── Core packing ─────────────────────────────────────────────────────────────
RUNS_PER_JOB="${RUNS_PER_JOB:-2}"

# ── Fixed factors (identical for every run) ─────────────────────────────────
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"        # baseline (predator-free) optimum;
                                              # applies only to RL conditions

# ── Preflight ────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR" logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

# ── One run, given a global run index ────────────────────────────────────────
run_one() {
    local GIDX="$1"
    local COND_IDX=$(( GIDX / PER_COND ))
    local REM=$((      GIDX % PER_COND ))
    local P_IDX=$((    REM / PER_PERIOD ))
    local REM2=$((     REM % PER_PERIOD ))
    local SEED_IDX=$(( REM2 / N_REPS ))
    local REP=$((      REM2 % N_REPS + 1 ))          # replicate number, 1..N_REPS
    local CONDITION_KEY="${CONDITIONS[$COND_IDX]}"
    local PERIOD="${PERIODS[$P_IDX]}"
    local MAP_SEED="${SEEDS[$SEED_IDX]}"

    # learning vs pure RL differ only in the GA (standard=GAManager, pure_rl=none);
    # evolution never learns in-life, so it takes no RL knobs.
    local HL_CONDITION HL_MODE LABEL USE_RL
    case "$CONDITION_KEY" in
        learning)  HL_CONDITION=learning;  HL_MODE=standard; LABEL=learning;  USE_RL=1 ;;
        pure_rl)   HL_CONDITION=learning;  HL_MODE=pure_rl;  LABEL=purerl;    USE_RL=1 ;;
        evolution) HL_CONDITION=evolution; HL_MODE=standard; LABEL=evolution; USE_RL=0 ;;
        *) echo "ERROR: unknown condition '$CONDITION_KEY'."; return 2 ;;
    esac

    local RL_ARGS NAME_MECH
    if [ "$USE_RL" = "1" ]; then
        RL_ARGS=(--no-epsilon --learning-rate "$LEARNING_RATE")
        NAME_MECH="${LABEL}_noeps_lr${LEARNING_RATE}"
    else
        RL_ARGS=()
        NAME_MECH="${LABEL}"
    fi
    # shufP<period> prefix marks the environment variant; seed the reshuffle
    # schedule to the map seed so it is fixed across this cell's replicates.
    local RUN_NAME="shufP${PERIOD}_g1k_${NAME_MECH}_seed${MAP_SEED}_rep${REP}"

    export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${GIDX}"

    echo "[run g$GIDX] ${LABEL} (cond=$HL_CONDITION mode=$HL_MODE)  P=$PERIOD  seed=$MAP_SEED  rep=$REP/$N_REPS  predators=OFF"
    "$NODE_BIN" src/headless.js \
        --map-seed "$MAP_SEED" \
        --condition "$HL_CONDITION" \
        --mode "$HL_MODE" \
        --generations "$GENERATIONS" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --run-name "$RUN_NAME" \
        --log-dir "$LOG_DIR" \
        --predators-per-patch 0 \
        --roaming-predators 0 \
        --food-shuffle-period "$PERIOD" \
        --food-shuffle-seed "$MAP_SEED" \
        "${RL_ARGS[@]}"
    local rc=$?
    echo "[run g$GIDX] done (exit $rc): $RUN_NAME"
    return $rc
}

# ── Run this task's slice of RUNS_PER_JOB consecutive global indices ─────────
START=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_JOB ))
echo "[array] task=$SLURM_ARRAY_TASK_ID  global indices ${START}..$(( START + RUNS_PER_JOB - 1 ))  (RUNS_PER_JOB=$RUNS_PER_JOB, of $TOTAL_RUNS total)  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"

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
