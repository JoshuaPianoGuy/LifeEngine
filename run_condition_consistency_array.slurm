#!/bin/bash
# ============================================================================
# run_condition_consistency_array.slurm — CONDITION CONSISTENCY / stochasticity
#
# Runs ALL THREE conditions (learning, evolution, pure RL) on ALL FIVE map
# seeds, with FIVE independent REPLICATE runs per (condition, seed) cell:
#
#     3 conditions x 5 seeds x 5 replicates = 75 runs total.
#
# e.g. evolution on seed 1 five times, evolution on seed 42 five times, ...,
# then learning x every seed x5, then pure RL x every seed x5.
#
# CORE PACKING (to stay under the simultaneous-core limit)
# --------------------------------------------------------
# Instead of 75 array tasks (75 cores at once), each array task runs
# RUNS_PER_JOB runs back-to-back on its single core. With RUNS_PER_JOB=2 the 75
# runs pack into ceil(75/2)=38 tasks -> only 38 cores in flight. Two 1000-gen
# runs fit comfortably in the 45h wall-time (evolution runs are cheaper still).
#
#   RUNS_PER_JOB (env, default 2) -> set --array to  ceil(75 / RUNS_PER_JOB) - 1:
#       RUNS_PER_JOB=1 -> --array=0-74   (75 cores; no packing)
#       RUNS_PER_JOB=2 -> --array=0-37   (38 cores)   <- CURRENT DEFAULT
#       RUNS_PER_JOB=3 -> --array=0-24   (25 cores)
#       RUNS_PER_JOB=5 -> --array=0-14   (15 cores)
#   If you change RUNS_PER_JOB you MUST update the #SBATCH --array line to match.
#   Runs beyond index 74 are skipped, so an over-long array is harmless.
#
# WHY replicates work as a stochasticity probe
# --------------------------------------------
# Only the map TERRAIN is seeded (--map-seed). GA mutation, RL exploration and
# predator wandering are deliberately UNSEEDED (Math.random). So the 5 replicate
# runs of a (condition, seed) cell share IDENTICAL terrain but differ purely by
# algorithmic/predator chance — their spread is the run-to-run noise of that
# condition on that map. Comparing spreads:
#   * variance ACROSS replicates within a seed  -> algorithm stochasticity
#   * variance ACROSS seeds                      -> map/terrain sensitivity
# The condition with the tightest replicate spread is the most CONSISTENT.
# (Each replicate gets its own run folder via a _rep<N> suffix; the unseeded
# RNG then makes them genuinely independent repeats.)
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# ENVIRONMENT — pick with the ENV var (default: roaming/hard) ─────────────────
#     sbatch run_condition_consistency_array.slurm             # roaming (hard): 60 preds, drain 5, lr 0.02
#     ENV=baseline sbatch run_condition_consistency_array.slurm # predator-free: no preds, lr 0.01
# Per-environment settings match the existing comparison trios exactly:
#   roaming  -> roam60d5_ prefix, lr 0.02, 60 roaming predators + drain 5, patrol off
#   baseline -> baseline_ prefix, lr 0.01, no predators
# The learning rate applies ONLY to the RL-using conditions (learning, pure RL);
# evolution passes no RL knobs (its network never learns within a lifetime).
#
# MAP-SEED note (new worlds are generated + stored automatically)
# ---------------------------------------------------------------
# headless.js builds each world deterministically from --map-seed and caches it
# to src/maps/map_pool_seed<SEED>.json on first use, so seeds 123/456 need no
# manual pre-generation and every condition/replicate reuses identical terrain.
#
# Each task writes CSVs + params.txt to
#   logs/<cond>/<mode>/auto-run/<prefix>_g1k_<label>[_noeps_lr<LR>]_seed<SEED>_rep<N>/
# The _rep<N> tag keeps every replicate in its own folder (a forced --run-name is
# NOT collision-checked and Logger auto-save APPENDS, so a clash would double rows).
#
#   GENERATIONS=200 sbatch run_condition_consistency_array.slurm   # override
#   ENV=baseline sbatch ...                                        # predator-free
#   RUNS_PER_JOB=3 sbatch --array=0-24 ...                          # denser packing
# ============================================================================
#SBATCH --job-name="condition_consistency"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-37
#SBATCH --output=logs/condition_consistency_%A_%a.out
#SBATCH --error=logs/condition_consistency_%A_%a.err
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

# ── Grid (3 factors: condition x seed x replicate) ──────────────────────────
# Global run index = COND_IDX*(N_SEEDS*N_REPS) + SEED_IDX*N_REPS + REP_IDX.
# Each array TASK then runs RUNS_PER_JOB consecutive global indices in sequence.
CONDITIONS=(learning evolution pure_rl)   # the three mechanisms
SEEDS=(1 42 999 123 456)                  # 3 existing maps + 2 new maps (auto-generated)
N_SEEDS=${#SEEDS[@]}
N_REPS=5                                  # independent replicate runs per (condition, seed)
PER_COND=$(( N_SEEDS * N_REPS ))          # runs per condition (25)
TOTAL_RUNS=$(( ${#CONDITIONS[@]} * PER_COND ))   # 75

# ── Core packing: how many runs each array task executes back-to-back ────────
# Cuts simultaneous cores from TOTAL_RUNS to ceil(TOTAL_RUNS/RUNS_PER_JOB).
# CHANGING THIS REQUIRES updating the #SBATCH --array line (see header table).
RUNS_PER_JOB="${RUNS_PER_JOB:-2}"

# ── Fixed factors (identical for every run) ─────────────────────────────────
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-1000}"

# ── Environment selection (predator-free baseline vs roaming hard) ──────────
ENVIRONMENT="${ENV:-roaming}"
case "$ENVIRONMENT" in
    baseline)
        ENV_PREFIX="baseline"
        LEARNING_RATE="${LEARNING_RATE:-0.01}"   # predator-free optimum
        ROAMING_PREDATORS=0
        PRED_ARGS=(--predators-per-patch 0 --roaming-predators 0)
        PRED_DESC="NONE (baseline environment)"
        ;;
    roaming)
        ENV_PREFIX="roam60d5"
        LEARNING_RATE="${LEARNING_RATE:-0.02}"   # roaming optimum
        ROAMING_PREDATORS="${ROAMING_PREDATORS:-60}"
        PREDATOR_DRAIN="${PREDATOR_DRAIN:-5}"
        PRED_ARGS=(--predators-per-patch 0 --roaming-predators "$ROAMING_PREDATORS" --predator-drain "$PREDATOR_DRAIN")
        PRED_DESC="roaming=$ROAMING_PREDATORS drain=$PREDATOR_DRAIN patrol=0"
        ;;
    *) echo "ERROR: ENV must be 'roaming' (hard) or 'baseline' (predator-free)."; exit 1 ;;
esac

# ── Preflight ────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR" logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

# ── One run, given a global run index ────────────────────────────────────────
# Decomposes the index into (condition, seed, replicate), maps the condition to
# its headless flags, and runs it. Returns headless's exit code.
run_one() {
    local GIDX="$1"
    local COND_IDX=$(( GIDX / PER_COND ))
    local REM=$((      GIDX % PER_COND ))
    local SEED_IDX=$(( REM / N_REPS ))
    local REP=$((      REM % N_REPS + 1 ))       # replicate number, 1..N_REPS
    local CONDITION_KEY="${CONDITIONS[$COND_IDX]}"
    local MAP_SEED="${SEEDS[$SEED_IDX]}"

    # Only difference learning vs pure RL is the GA: standard=GAManager, pure_rl=none.
    # Evolution never learns within a lifetime, so it takes no RL knobs.
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
    local RUN_NAME="${ENV_PREFIX}_g1k_${NAME_MECH}_seed${MAP_SEED}_rep${REP}"

    # headless.js derives its run folder from --run-name (we pass a unique one),
    # but keep PBS_JOBID unique per run for any other unique-naming path.
    export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${GIDX}"

    echo "[run g$GIDX] ${LABEL} (cond=$HL_CONDITION mode=$HL_MODE)  seed=$MAP_SEED  rep=$REP/$N_REPS  env=$ENVIRONMENT  predators: $PRED_DESC"
    "$NODE_BIN" src/headless.js \
        --map-seed "$MAP_SEED" \
        --condition "$HL_CONDITION" \
        --mode "$HL_MODE" \
        --generations "$GENERATIONS" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --run-name "$RUN_NAME" \
        --log-dir "$LOG_DIR" \
        "${RL_ARGS[@]}" \
        "${PRED_ARGS[@]}"
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
