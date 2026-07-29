#!/bin/bash
# ============================================================================
# run_landscape_stage46_array.slurm — ONE array that runs BOTH remaining probe
# stages of the fitness-landscape pipeline (seed 999) in a single submission:
#
#   Stage 6  ruggedness (lambda) mutational walks — {hard,baseline} x {random,centroid}
#   Stage 4  random filter-normalised planes      — {hard,baseline} x plane{0,1,2}
#
# ("Stage 5" is not a separate thing to submit — it is just *running* a jobs.json
#  through src/eval/run_probe.js, which is exactly what every task here does.)
#
#     sbatch run_landscape_stage46_array.slurm
#
# Every jobs.json is written LOCALLY first (landscape/stage{4,6}_*.py ... jobs)
# and carries its genomes embedded as base64 + its own config_overrides, so the
# cluster needs NO genome.csv and no per-plane predator flags — only Node and the
# matching learning-run params.json.
#
# WORK UNITS (unit-major, shard varies fastest; stage 6 first so the lambda
# headline lands in the first wave):
#
#   idx  unit                  out-dir                                probes  shards
#    0   s6_hard_random        landscape/out/stage6_hard_random       20,040    32
#    1   s6_hard_centroid      landscape/out/stage6_hard_centroid     20,040    32
#    2   s6_baseline_random    landscape/out/stage6_baseline_random   20,040    32
#    3   s6_baseline_centroid  landscape/out/stage6_baseline_centroid 20,040    32
#    4   s4_hard_plane0        landscape/out/stage4_hard/plane0       12,500    20
#    5   s4_hard_plane1        landscape/out/stage4_hard/plane1       12,500    20
#    6   s4_hard_plane2        landscape/out/stage4_hard/plane2       12,500    20
#    7   s4_baseline_plane0    landscape/out/stage4_baseline/plane0   12,500    20
#    8   s4_baseline_plane1    landscape/out/stage4_baseline/plane1   12,500    20
#    9   s4_baseline_plane2    landscape/out/stage4_baseline/plane2   12,500    20
#                                                        TOTAL 155,160   248
#
#   Stage 6 = 5 internal walks x 501 steps x 8 map repeats per condition. Those 5
#   walks live INSIDE one jobs.json (they are what gives lambda its cross-walk
#   error bar) and are not the four conditions above.
#   Stage 4 = 25 x 25 grid x 8 repeats x {RL off, RL on} per plane.
#
# ALLOCATION / CORE PACKING
# -------------------------
# Shard counts are chosen so EVERY task is ~625 probes regardless of stage
# (12500/20 = 625; 20040/32 = 626) — uniform tasks mean no straggler holds the
# tail. 248 tasks at ~625 probes x ~5-10 s = ~1-2 h per task.
#
# --time is PER TASK, not for the whole array. The stage-3 planes measured
# ~4-7 s/probe (625-probe shards finished in 40-75 min), so a task here should
# take ~1-1.5 h; stage-6 shards are RL-off only and should sit at the fast end.
# 12:00:00 is deliberate over-provisioning (~10x headroom, tolerating up to
# ~70 s/probe). The only cost is backfill priority — long walltime requests are
# harder for the scheduler to slot into gaps, so tasks may queue a little
# longer. Drop to 06:00:00 if the queue looks slow; a task killed on walltime
# is cheap anyway, since resubmitting skips every probe already in its CSV.
#
# CORES REQUESTED = the %N throttle on --array x --cpus-per-task (1). Nothing
# else in this file affects it — the task->unit offset math below only decides
# WHICH out-dir an already-running task reads.
#
# `--array=0-247%50` caps CONCURRENCY at 50 tasks = 50 cores, comfortably under
# the ~100 per-job cap (%100 sits exactly ON that ceiling and gets rejected once
# anything else of yours is queued). Stages 4 and 6 still run TOGETHER: SLURM
# feeds all 248 tasks through the 50 slots in ~5 waves.
#
#   throttle   concurrent cores   waves   approx wall clock
#     %32              32          7.8         ~10 h
#     %50              50          5.0          ~6 h     <- default
#     %64              64          3.9          ~5 h
#     %100            100          2.5          ~3 h     (at the cap; risky)
#
# Total work is fixed at ~300 core-hours either way. To change it, edit ONLY the
# number after % in --array below — the 248 upper bound must stay in sync with
# sum(UNIT_SHARDS), but the throttle is free.
#   If the rejection is about memory rather than cores, --mem=6G per task is
#   generous for a single-genome probe; 2-3G is ample.
#
# RESUMING
# --------
# run_probe.js skips job ids already present in its shard CSV, so a walltime kill
# just needs a resubmit — completed probes are not repeated.
#   *** If you REGENERATE a jobs.json, delete that out-dir's results_shard_*.csv
#   and results.csv FIRST. *** The resume check is by job id only; it cannot tell
#   that an id now refers to a different genome. Pre-2026-07-21 stage-6 jobs
#   (100 steps x 10 repeats, single walk) also used the legacy id format
#   s{t}_r{r} vs today's w{walk}_s{step}_r{repeat}, and merging the two breaks
#   stage6_lambda.py analyze outright. The preflight below hard-fails on that
#   rather than letting it through.
#
# AFTER the array finishes — merge + analyse LOCALLY (needs Python, seconds):
#   for OD in stage6_hard_random stage6_hard_centroid \
#             stage6_baseline_random stage6_baseline_centroid; do
#     python landscape/merge_shards.py          --out-dir landscape/out/$OD
#     python landscape/stage6_lambda.py analyze --out-dir landscape/out/$OD \
#            --kfit 10 --detrend linear
#     python landscape/stage8_nk.py             --out-dir landscape/out/$OD
#   done
#   for OD in stage4_hard/plane0 stage4_hard/plane1 stage4_hard/plane2 \
#             stage4_baseline/plane0 stage4_baseline/plane1 stage4_baseline/plane2; do
#     python landscape/merge_shards.py  --out-dir landscape/out/$OD
#     python landscape/stage7_plots.py  --out-dir landscape/out/$OD \
#            --lambda-json landscape/out/stage6_hard_random/lambda_summary.json
#   done
#
# (sbatch SCHEDULES this file — you do not execute it yourself.)
# ============================================================================
#SBATCH --job-name="landscape_s46_s999"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --array=0-247%50
#SBATCH --output=logs/landscape_s46_%A_%a.out
#SBATCH --error=logs/landscape_s46_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/LifeEngine}}"

HARD_PARAMS=logs/learning/standard/auto-run/roam60d5_g1k_learning_noeps_lr0.02_lscape_seed999_r2/params.json
BASE_PARAMS=logs/learning/standard/auto-run/baseline_g1k_learning_noeps_lr0.01_lscape_seed999_r2/params.json

# ── Work units (baked in) ────────────────────────────────────────────────────
UNIT_NAMES=(
    s6_hard_random  s6_hard_centroid  s6_baseline_random  s6_baseline_centroid
    s4_hard_plane0  s4_hard_plane1    s4_hard_plane2
    s4_baseline_plane0  s4_baseline_plane1  s4_baseline_plane2
)
UNIT_OUT=(
    landscape/out/stage6_hard_random
    landscape/out/stage6_hard_centroid
    landscape/out/stage6_baseline_random
    landscape/out/stage6_baseline_centroid
    landscape/out/stage4_hard/plane0
    landscape/out/stage4_hard/plane1
    landscape/out/stage4_hard/plane2
    landscape/out/stage4_baseline/plane0
    landscape/out/stage4_baseline/plane1
    landscape/out/stage4_baseline/plane2
)
UNIT_PARAMS=(
    "$HARD_PARAMS" "$HARD_PARAMS" "$BASE_PARAMS" "$BASE_PARAMS"
    "$HARD_PARAMS" "$HARD_PARAMS" "$HARD_PARAMS"
    "$BASE_PARAMS" "$BASE_PARAMS" "$BASE_PARAMS"
)
UNIT_SHARDS=(32 32 32 32  20 20 20  20 20 20)   # sum MUST equal the --array size
# Expected job-id prefix per unit — the stale-results guard below.
#   stage 6 ids: w{walk}_s{step}_r{repeat}    stage 4 ids: p{i}_{j}_r{r}_{on|off}
UNIT_IDRE=(
    '^w[0-9]+_s[0-9]+_r[0-9]+$' '^w[0-9]+_s[0-9]+_r[0-9]+$'
    '^w[0-9]+_s[0-9]+_r[0-9]+$' '^w[0-9]+_s[0-9]+_r[0-9]+$'
    '^p[0-9]+_[0-9]+_r[0-9]+_(on|off)$' '^p[0-9]+_[0-9]+_r[0-9]+_(on|off)$'
    '^p[0-9]+_[0-9]+_r[0-9]+_(on|off)$' '^p[0-9]+_[0-9]+_r[0-9]+_(on|off)$'
    '^p[0-9]+_[0-9]+_r[0-9]+_(on|off)$' '^p[0-9]+_[0-9]+_r[0-9]+_(on|off)$'
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
    echo "ERROR: $OUT_DIR/jobs.json not found. Generate it locally with landscape/stage${UNIT:1:1}_*.py ... jobs and transfer the out-dir."
    exit 1
fi
if [ ! -f "$PARAMS" ]; then
    echo "ERROR: params not found at $PARAMS (copy the learning run's params.json to the cluster)."
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

echo "[s46] task=$TASK  unit=$UNIT  shard=${SHARD}/${N_SHARDS}"
echo "[s46]   out_dir=$OUT_DIR"
echo "[s46]   params=$PARAMS"
echo "[s46]   -> $OUT_CSV"

"$NODE_BIN" src/eval/run_probe.js \
    --params "$PARAMS" \
    --jobs   "$OUT_DIR/jobs.json" \
    --out    "$OUT_CSV" \
    --shard  "${SHARD}/${N_SHARDS}"
rc=$?
echo "[s46] task=$TASK done (exit $rc): $UNIT shard $SHARD"
exit $rc
