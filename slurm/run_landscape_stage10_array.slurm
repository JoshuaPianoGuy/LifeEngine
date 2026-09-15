#!/bin/bash
# ============================================================================
# run_landscape_stage10_array.slurm — Stage 10 multi-trajectory landscapes.
#
#   ONE plane per ENVIRONMENT, carrying EVERY run of EVERY condition. The
#   trajectories are drawn individually; the only averaging anywhere is the
#   anchor genome (the mean of all pooled centroids). This array evaluates the
#   two fitness surfaces those trajectories will be drawn on.
#
#     sbatch run_landscape_stage10_array.slurm
#
# Both jobs.json files are written LOCALLY first and carry their genomes
# embedded as base64 plus their own config_overrides, so the cluster needs NO
# genome.csv and no per-plane predator flags — only Node and the matching
# learning-run params.json:
#
#   python landscape/stage10_strata.py planejobs \
#       --logs-root logs_landscapes --env baseline --plane-method joint \
#       --resolution 21 --repeats 5 --out-dir landscape/out/stage10_baseline_traj
#
#   python landscape/stage10_strata.py planejobs \
#       --logs-root logs_landscapes --env roam60d5 --plane-method within \
#       --resolution 21 --repeats 5 --out-dir landscape/out/stage10_hard_traj \
#       --config-overrides '{"roaming_predator_count":60,"predator_drain":5}'
#
# WHY THE TWO ENVIRONMENTS USE DIFFERENT PLANE METHODS
# ----------------------------------------------------
# Not a fudge — it is what the free preview showed, and the scores alone point
# the wrong way in BOTH directions (run `planecheck --fig` to reproduce):
#
#   BASELINE (3 evolution + 3 learning) -> 'joint'
#       Scores a poor 0.058 within-run capture yet draws the cleanest figure:
#       six paths fanning out from the shared gen-1 origin with the two
#       conditions cleanly separated.
#   HARD (13 evolution + 3 learning) -> 'within'
#       'joint' scores 0.010 and spends BOTH components on the 3 long learning
#       runs, squashing all 13 evolution paths into a blob at the origin.
#       'within' (PCA about each run's own mean) fans them out legibly.
#       'balanced' is worse still here: the two COLLAPSED runs are short, so
#       normalising run length amplifies them until they hijack the plane.
#
# The honest caveat to carry into the writeup: the best possible 2D plane
# fitted to a SINGLE run alone captures only ~0.61-0.66 of that run's own
# motion, so every 2D landscape figure in this project drops at least a third
# of the trajectory. That is a property of 3910-D weight space, not of pooling.
#
# WORK UNITS (unit-major, shard varies fastest)
#
#   idx  unit             out-dir                                probes  shards
#    0   s10_hard         landscape/out/stage10_hard_traj          4,410     15
#    1   s10_baseline     landscape/out/stage10_baseline_traj      4,410     15
#                                                        TOTAL     8,820     30
#
#   Each = 21 x 21 grid x 5 repeats x {RL off, RL on}.
#
# ALLOCATION / CORE PACKING
# -------------------------
# The allocation grants 120 cores at once and the scheduler charges the WHOLE
# submitted array against it, so sum(UNIT_SHARDS) must stay <= 120 (see
# run_landscape_stage46_array.slurm, which sits at 114). This one asks for 30 —
# it is a far smaller job than stage 4/6 (8,820 probes vs 155,160), so the core
# limit is nowhere near binding and there is ample headroom to raise it.
#
#   At the ~4-7 s/probe measured on the stage-3 planes:
#     30 tasks x 294 probes  ~ 20-35 min/task, one wave, ~13.5 core-hours total.
#
#   To go wider (e.g. 40 shards each = 80 tasks, ~8-13 min/task), change
#   UNIT_SHARDS and the --array upper bound TOGETHER — they must stay in sync
#   (upper bound = sum(UNIT_SHARDS) - 1); a mismatch silently drops shards.
#
# --time is PER TASK, not for the whole array. 4h against a ~35 min task is ~7x
# headroom; a task killed on walltime is cheap because resubmitting skips every
# probe already in its CSV.
#
# RESUMING
# --------
# run_probe.js skips job ids already present in its shard CSV, so a walltime
# kill just needs a resubmit.
#   *** If you REGENERATE a jobs.json, delete that out-dir's results_shard_*.csv
#   and results.csv FIRST. *** The resume check is by job id only; it cannot
#   tell that an id now refers to a different genome — and a new plane means
#   every id DOES refer to a different genome. The guard below catches the wrong
#   id FORMAT but cannot catch a same-format id from a superseded plane.
#
# AFTER the array finishes — merge + plot LOCALLY (needs Python, seconds):
#   for OD in stage10_hard_traj stage10_baseline_traj; do
#     python landscape/merge_shards.py --out-dir landscape/out/$OD
#   done
#   python landscape/stage10_strata.py trajplot \
#          --grid-dir landscape/out/stage10_baseline_traj --env baseline
#   python landscape/stage10_strata.py trajplot \
#          --grid-dir landscape/out/stage10_hard_traj --env hard
#
# (sbatch SCHEDULES this file — you do not execute it yourself.)
# ============================================================================
#SBATCH --job-name="landscape_s10_traj"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-29
#SBATCH --output=logs/landscape_s10_%A_%a.out
#SBATCH --error=logs/landscape_s10_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/LifeEngine}}"

# Where the transferred run tree lives on the cluster. The Stage 10 planejobs
# step reads this tree LOCALLY; the cluster only needs one params.json per
# environment out of it, so this is the sole reason it must be present here.
LOGS_ROOT="${LOGS_ROOT:-logs_landscapes}"

HARD_PARAMS=$LOGS_ROOT/learning/standard/auto-run/roam60d5_g1k_learning_noeps_lr0.02_lscape_seed999_r2/params.json
BASE_PARAMS=$LOGS_ROOT/learning/standard/auto-run/baseline_g1k_learning_noeps_lr0.01_lscape_seed999_r2/params.json

# ── Work units (baked in) ────────────────────────────────────────────────────
UNIT_NAMES=(s10_hard          s10_baseline)
UNIT_OUT=(
    landscape/out/stage10_hard_traj
    landscape/out/stage10_baseline_traj
)
UNIT_PARAMS=("$HARD_PARAMS" "$BASE_PARAMS")
UNIT_SHARDS=(15 15)   # sum (30) MUST equal the --array size
# Expected job-id prefix per unit — the stale-results guard below.
#   grid ids: p{i}_{j}_r{r}_{on|off}
UNIT_IDRE=(
    '^p[0-9]+_[0-9]+_r[0-9]+_(on|off)$'
    '^p[0-9]+_[0-9]+_r[0-9]+_(on|off)$'
)

# ── Preflight ────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi

# ── Map this task -> (unit, shard) via cumulative shard offsets ──────────────
TASK="${SLURM_ARRAY_TASK_ID}"
acc=0
UNIT_IDX=-1
for i in "${!UNIT_SHARDS[@]}"; do
    n="${UNIT_SHARDS[$i]}"
    if [ "$TASK" -lt $(( acc + n )) ]; then
        UNIT_IDX=$i
        SHARD=$(( TASK - acc ))
        N_SHARDS=$n
        break
    fi
    acc=$(( acc + n ))
done
if [ "$UNIT_IDX" -lt 0 ]; then
    echo "[array] task $TASK >= $acc total shards — nothing to do."; exit 0
fi

UNIT="${UNIT_NAMES[$UNIT_IDX]}"
OUT_DIR="${UNIT_OUT[$UNIT_IDX]}"
PARAMS="${UNIT_PARAMS[$UNIT_IDX]}"
ID_RE="${UNIT_IDRE[$UNIT_IDX]}"
OUT_CSV="$OUT_DIR/results_shard_${SHARD}.csv"

if [ ! -f "$OUT_DIR/jobs.json" ]; then
    echo "ERROR: $OUT_DIR/jobs.json not found. Generate it locally with"
    echo "       landscape/stage10_strata.py planejobs ... and transfer the out-dir."
    exit 1
fi
if [ ! -f "$PARAMS" ]; then
    echo "ERROR: params not found at $PARAMS"
    echo "       Copy the learning run's params.json to the cluster under \$LOGS_ROOT"
    echo "       (currently '$LOGS_ROOT'), or set LOGS_ROOT to where you put the tree."
    exit 1
fi

# Stale-results guard: an existing shard CSV whose ids do not match this unit's
# format is left over from an older jobs.json. Resuming on top of it would keep
# rows that no longer correspond to these genomes, so refuse rather than merge.
if [ -s "$OUT_CSV" ]; then
    FIRST_ID=$(awk -F',' 'NR==2{print $1; exit}' "$OUT_CSV")
    if [ -n "$FIRST_ID" ] && ! echo "$FIRST_ID" | grep -Eq "$ID_RE"; then
        echo "ERROR: $OUT_CSV holds ids like '$FIRST_ID', which do not match this unit's"
        echo "       format ($ID_RE) — it is stale output from an older jobs.json."
        echo "       Delete it first:  rm -f $OUT_DIR/results_shard_*.csv $OUT_DIR/results.csv"
        exit 1
    fi
fi

echo "[s10] task=$TASK  unit=$UNIT  shard=${SHARD}/${N_SHARDS}"
echo "[s10]   out_dir=$OUT_DIR"
echo "[s10]   params=$PARAMS"
echo "[s10]   -> $OUT_CSV"

"$NODE_BIN" src/eval/run_probe.js \
    --params "$PARAMS" \
    --jobs   "$OUT_DIR/jobs.json" \
    --out    "$OUT_CSV" \
    --shard  "${SHARD}/${N_SHARDS}"
rc=$?
echo "[s10] task=$TASK done (exit $rc): $UNIT shard $SHARD"
exit $rc
