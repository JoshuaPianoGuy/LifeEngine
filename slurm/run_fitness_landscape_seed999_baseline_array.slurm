#!/bin/bash
# ============================================================================
# run_fitness_landscape_seed999_baseline_array.slurm — FITNESS-LANDSCAPE source
# runs, PREDATOR-FREE BASELINE environment.
#
# Baseline counterpart of run_fitness_landscape_seed999_array.slurm (which
# defaults to the roaming/hard environment). Identical machinery — only the
# default environment differs (baseline here) — so the two landscapes can be
# compared: same three mechanisms, same map seed, with vs without predators.
#
# Runs the three conditions THREE times each on a single map seed (999), with
# founder-genome logging turned ON (--log-genomes), so the logged genomes can
# later be used to plot the filter-normalised fitness landscapes:
#
#     3 conditions x 3 replicates = 9 runs total, all on seed 999.
#
#   learning   rep 1..3   (--condition learning --mode standard  + RL knobs)
#   evolution  rep 1..3   (--condition evolution --mode standard, no RL knobs)
#   pure RL    rep 1..3   (--condition learning --mode pure_rl    + RL knobs)
#
# Nine independent runs, one per array task (--array=0-8): 500x500 world,
# 1000 generations, matching the other condition scripts. Only the map TERRAIN
# is seeded; GA mutation / RL exploration stay UNSEEDED, so the 3 replicates
# share identical terrain but are genuinely independent repeats (a per-condition
# landscape ensemble on one map).
#
# GENOME LOGGING (the reason these runs exist)
# --------------------------------------------
# --log-genomes makes headless write genome.csv (per-generation centroid +
# fittest founder genomes, all founders every 250 gens). It is OFF by default
# (tuning sweeps don't need it); these final runs turn it ON to feed the
# fitness-landscape evaluator. CSVs are otherwise identical to the other runs.
#
# (sbatch SCHEDULES this file — you do not execute a shell script yourself.)
#
# ENVIRONMENT — defaults to baseline; ENV var can override ─────────────────────
#     sbatch run_fitness_landscape_seed999_baseline_array.slurm             # predator-free: no preds, lr 0.01
#     ENV=roaming sbatch run_fitness_landscape_seed999_baseline_array.slurm # roaming (hard): 60 preds, drain 5, lr 0.02
# (For the roaming landscape prefer run_fitness_landscape_seed999_array.slurm,
# which defaults to roaming.) Per-environment settings match the existing trios:
#   baseline -> baseline_ prefix, lr 0.01, no predators
#   roaming  -> roam60d5_ prefix, lr 0.02, 60 roaming predators + drain 5, patrol off
# The learning rate applies ONLY to the RL-using conditions (learning, pure RL);
# evolution passes no RL knobs (its network never learns within a lifetime).
#
# MAP-SEED note (world generated + cached automatically)
# ------------------------------------------------------
# headless.js builds seed 999's world deterministically from --map-seed and
# caches it to src/maps/map_pool_seed999.json on first use, so every run reuses
# identical terrain with no manual pre-generation.
#
# Each task writes CSVs (incl. genome.csv) + params.txt to
#   logs/<cond>/<mode>/auto-run/<prefix>_g1k_<mech>_lscape_seed999_r<N>/
# The _lscape_..._r<N> tag is DELIBERATELY distinct from the consistency runs'
# _rep<N> tag and the explore-bonus sweep's _eb<val> tag, so these landscape runs
# (which share the same params) are NOT swept into those analyses. A forced
# --run-name is not collision-checked and Logger auto-save APPENDS, so a unique
# name per run is required to avoid doubled rows. (The baseline_ prefix also keeps
# these distinct from the roaming landscape set's roam60d5_ folders.)
#
#   GENERATIONS=200 sbatch run_fitness_landscape_seed999_baseline_array.slurm   # override
#   ENV=roaming sbatch ...                                                      # roaming instead
#   SEED=42 sbatch ...                                                          # different map
# ============================================================================
#SBATCH --job-name="fitness_landscape_s999_baseline"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-8
#SBATCH --output=logs/fitness_landscape_s999_baseline_%A_%a.out
#SBATCH --error=logs/fitness_landscape_s999_baseline_%A_%a.err
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

# ── Grid (2 factors: condition x replicate) on a single seed ────────────────
# Global run index = COND_IDX*N_REPS + REP_IDX. One run per array task.
CONDITIONS=(learning evolution pure_rl)   # the three mechanisms
MAP_SEED="${SEED:-999}"                   # single map seed for this landscape set
N_REPS="${N_REPS:-3}"                      # independent replicate runs per condition
TOTAL_RUNS=$(( ${#CONDITIONS[@]} * N_REPS ))   # 9

# ── Fixed factors (identical for every run) ─────────────────────────────────
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-1000}"

# ── Environment selection (predator-free baseline vs roaming hard) ──────────
# Defaults to baseline for this script; ENV=roaming overrides.
ENVIRONMENT="${ENV:-baseline}"
case "$ENVIRONMENT" in
    baseline)
        ENV_PREFIX="baseline"
        LEARNING_RATE="${LEARNING_RATE:-0.01}"   # predator-free optimum
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
    *) echo "ERROR: ENV must be 'baseline' (predator-free) or 'roaming' (hard)."; exit 1 ;;
esac

# ── Preflight ────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR" logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

# ── This task's run: decompose the array index into (condition, replicate) ───
GIDX="$SLURM_ARRAY_TASK_ID"
if [ "$GIDX" -ge "$TOTAL_RUNS" ]; then
    echo "[array] index $GIDX >= $TOTAL_RUNS — nothing to do."; exit 0
fi
COND_IDX=$(( GIDX / N_REPS ))
REP=$((      GIDX % N_REPS + 1 ))          # replicate number, 1..N_REPS
CONDITION_KEY="${CONDITIONS[$COND_IDX]}"

# Only difference learning vs pure RL is the GA: standard=GAManager, pure_rl=none.
# Evolution never learns within a lifetime, so it takes no RL knobs.
case "$CONDITION_KEY" in
    learning)  HL_CONDITION=learning;  HL_MODE=standard; LABEL=learning;  USE_RL=1 ;;
    pure_rl)   HL_CONDITION=learning;  HL_MODE=pure_rl;  LABEL=purerl;    USE_RL=1 ;;
    evolution) HL_CONDITION=evolution; HL_MODE=standard; LABEL=evolution; USE_RL=0 ;;
    *) echo "ERROR: unknown condition '$CONDITION_KEY'."; exit 2 ;;
esac

# Exploration bonus is OFF (0) for the RL conditions — the new project default
# (see ExperimentParams.explore_bonus). Passed explicitly so the run records it
# regardless of the code default.
if [ "$USE_RL" = "1" ]; then
    RL_ARGS=(--no-epsilon --learning-rate "$LEARNING_RATE" --explore-bonus 0)
    NAME_MECH="${LABEL}_noeps_lr${LEARNING_RATE}"
else
    RL_ARGS=()
    NAME_MECH="${LABEL}"
fi
# _lscape_..._r<N>: intentionally NOT the consistency _rep<N> nor the sweep _eb tag,
# so these genome-logged landscape runs stay out of those analyses.
RUN_NAME="${ENV_PREFIX}_g1k_${NAME_MECH}_lscape_seed${MAP_SEED}_r${REP}"

# headless.js derives its run folder from --run-name (we pass a unique one), but
# keep PBS_JOBID unique per run for any other unique-naming path.
export PBS_JOBID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "[array] task=$GIDX  ${LABEL} (cond=$HL_CONDITION mode=$HL_MODE)  seed=$MAP_SEED  rep=$REP/$N_REPS  env=$ENVIRONMENT  predators: $PRED_DESC"
echo "[array]   grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS  genome-logging=ON  run=$RUN_NAME"

# --log-genomes writes genome.csv (the fitness-landscape source). --condition +
# --mode select the mechanism; RL knobs apply only to the RL-using conditions.
"$NODE_BIN" src/headless.js \
    --map-seed "$MAP_SEED" \
    --condition "$HL_CONDITION" \
    --mode "$HL_MODE" \
    --generations "$GENERATIONS" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --run-name "$RUN_NAME" \
    --log-dir "$LOG_DIR" \
    --log-genomes \
    "${RL_ARGS[@]}" \
    "${PRED_ARGS[@]}"
rc=$?

echo "[array] task=$GIDX done (exit $rc): $RUN_NAME"
exit $rc
