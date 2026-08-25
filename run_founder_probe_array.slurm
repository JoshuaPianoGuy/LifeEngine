#!/bin/bash
# ============================================================================
# run_founder_probe_array.slurm — re-evaluate every LOGGED FOUNDER GENOME with
# in-lifetime learning OFF and ON, to test genetic assimilation.
#
# Build the units locally first (needs Python + the genome.csv files), then
# transfer only the out-root:
#
#     python assimilation/make_founder_jobs.py --logs-root logs_hard \
#         --env-label hard --out-root assimilation/out/founders_hard
#
# The defaults below already match the hard set (20 units x 4 shards = 80 tasks),
# so for that one it is simply:
#
#     sbatch run_founder_probe_array.slurm
#
# For the baseline set, point it at the other root — same shape, 20 units:
#
#     UNITS_ROOT=assimilation/out/founders_baseline sbatch run_founder_probe_array.slurm
#
# Both are 20 units -> 80 tasks at the default SHARDS_PER_UNIT=4, capped at 96
# concurrent by the %96 on the --array line. Run them one after the other, or
# together if the account can hold both.
#
# make_founder_jobs.py prints the exact --array line for the units it built.
# (sbatch SCHEDULES this file — you do not execute it yourself.)
#
# WHERE THE OUTPUT GOES
# ---------------------
#   $UNITS_ROOT/<unit>/results_shard_<i>.csv   one per task, beside that unit's
#                                              jobs.json — same layout as the
#                                              landscape slices
#   ./slurm-founder_<jobid>_<task>.out|.err    in the submit directory
#
# The result CSVs are SMALL — ~99 bytes/row, 600 rows per shard, so ~60 KB per
# shard and ~4.7 MB for a whole environment. Only those need to come back:
#
#   rsync -av --include='*/' --include='results_shard_*.csv' --exclude='*' \
#       <cluster>:<path>/assimilation/out/founders_hard/ \
#       assimilation/out/founders_hard/
#
# Do NOT pull the jobs.json files back — they are the 324 MB you sent up.
#
# WHAT ONE UNIT IS
# ----------------
# One RUN's founder genomes: 100 founders at each of generations 250/500/750/
# 1000, so 400 genomes, each probed as a monomorphic 100-clone cohort with the
# GA off, on R maps, twice — once with RL off (what the inherited weights do
# alone) and once with RL on (what they do while learning). 400 x R x 2 jobs.
#
# The unit dirs are SELF-CONTAINED: jobs.json carries every genome as base64 and
# the config_overrides (hidden_size, map_seed), and params.json is copied in
# beside it. Nothing here reads the logs tree, and the cluster runs Node only —
# no Python, no venv.
#
# h128 — THE THING THAT SILENTLY BREAKS IF IT IS WRONG
# ------------------------------------------------------
# These runs are hidden_size=128 (genome 7814). run_probe.js builds its network
# from ExperimentParams, so without config_overrides.hidden_size it builds a
# 64-wide brain and mis-reads every genome without erroring. The preflight below
# asserts the key is present before spending any walltime.
#
# SHARD COUNT IS FREE HERE, AND RL-OFF FINISHES FIRST
# -----------------------------------------------------
# run_probe.js shards with `idx % N`. In the LANDSCAPE jobs files the jobs
# alternate RL-off / RL-on, so an even N preserved parity and split the array
# into an all-RL-off half and an all-RL-on half; because RL-on is ~2.4x slower,
# half the tasks hit the walltime and truncated while the other half finished
# early. That cost the h128 companion run 8,468 probes.
#
# make_founder_jobs.py writes the jobs as TWO CONTIGUOUS BLOCKS — every RL-off
# job, then every RL-on job. Two consequences, both wanted:
#
#   * any N is balanced. Shard i takes every Nth index, and the off jobs are the
#     first half of the list, so it draws about half its work from each block
#     whatever N is. The parity trap cannot recur.
#   * a walltime kill costs RL-ON probes, never RL-off ones. The off jobs come
#     first inside every shard, so the INHERITED-genome measurement — the primary
#     one for the assimilation test — completes by construction, and only the
#     secondary pass degrades.
#
# The preflight below checks that ordering actually holds, because a hand-built
# or older jobs.json would not have it.
#
# CORES / TIME
# ------------
# At ~7 s per RL-off probe and ~17 s per RL-on one (measured on the h128
# companion slices at w500 / 2000 ticks), a unit of 400 genomes x 3 maps x 2
# passes = 2,400 probes is ~8 core-hours.
#
# THE ACCOUNT CAN HOLD 100 CORES AT A TIME. 20 units, so the task count is
# 20 x SHARDS_PER_UNIT and only these fit in one wave:
#
#   SHARDS_PER_UNIT=2 -> --array=0-39    40 tasks, 1200 probes, ~4.0 h
#   SHARDS_PER_UNIT=3 -> --array=0-59    60 tasks,  800 probes, ~2.7 h
#   SHARDS_PER_UNIT=4 -> --array=0-79    80 tasks,  600 probes, ~2.0 h  <- DEFAULT
#   SHARDS_PER_UNIT=5 -> --array=0-99   100 tasks,  480 probes, ~1.6 h  (AT the cap)
#
# 4 is the default because 80 stays clear of the cap with room for anything else
# queued, and ~2.0 h against the 6h --time is 3x headroom. Packing fewer, longer
# tasks costs cores-in-use, not finishing time, as long as they all start
# together — so there is nothing to gain by crowding the limit.
#
# ANYTHING LARGER MUST BE THROTTLED. The %N suffix on --array caps how many
# tasks run CONCURRENTLY, independent of how many exist, so a bigger array is
# fine as long as it carries one:
#
#   SHARDS_PER_UNIT=8  sbatch --array=0-159%96 run_founder_probe_array.slurm
#
# The #SBATCH line below already carries %96 so the default cannot exceed the
# cap even if SHARDS_PER_UNIT is raised without updating --array. If you override
# --array on the command line, CARRY THE %96 OVER — a command-line --array
# replaces this line entirely, throttle included, and that is exactly how a
# 220-core request gets made by accident.
#
# A task killed on walltime is cheap twice over: run_probe.js skips job ids
# already present in its shard CSV so resubmitting the SAME array resumes rather
# than restarts, and the RL-off-first ordering means what it lost was RL-on.
#
# AFTER
# -----
#     for d in assimilation/out/founders_hard/*/; do
#         python landscape/merge_shards.py --out-dir "$d"; done
#
# Run that LOCALLY after pulling the shard CSVs back — the cluster has no Python.
# ============================================================================
#SBATCH --job-name="founder_probe"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
# %96 is the CONCURRENCY cap, not the task count: at most 96 of these run at
# once, keeping the job inside the 100-core allowance. See CORES / TIME above.
#SBATCH --array=0-79%96
# Written to the SUBMIT DIR, not logs/, on purpose: SLURM opens these before the
# script runs and does NOT create missing directories, so a --output under a
# path that does not exist on the cluster fails the task before line 1.
#SBATCH --output=slurm-founder_%A_%a.out
#SBATCH --error=slurm-founder_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

set -u

UNITS_ROOT="${UNITS_ROOT:-assimilation/out/founders_hard}"
SHARDS_PER_UNIT="${SHARDS_PER_UNIT:-4}"
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

if [ ! -d "$UNITS_ROOT" ]; then
    echo "ERROR: $UNITS_ROOT not found. Build the units locally with"
    echo "       python assimilation/make_founder_jobs.py --logs-root <logs|logs_hard> \\"
    echo "           --env-label <baseline|hard> --out-root $UNITS_ROOT"
    echo "       then transfer that directory here."
    exit 1
fi

# Unit list, sorted so it is IDENTICAL between the submitting task and any
# resubmission — a reordering would send a shard's rows to the wrong unit.
UNITS=()
while IFS= read -r u; do
    [ -f "$u/jobs.json" ] && UNITS+=("$u")
done < <(find "$UNITS_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
N_UNITS=${#UNITS[@]}

if [ "$N_UNITS" -eq 0 ]; then
    echo "ERROR: no sub-dir of $UNITS_ROOT contains a jobs.json."
    exit 1
fi

EXPECTED_TASKS=$(( N_UNITS * SHARDS_PER_UNIT ))
ARRAY_COUNT="${SLURM_ARRAY_TASK_COUNT:-$EXPECTED_TASKS}"
if [ "$ARRAY_COUNT" -ne "$EXPECTED_TASKS" ]; then
    echo "ERROR: --array covers $ARRAY_COUNT tasks but $N_UNITS units x"
    echo "       SHARDS_PER_UNIT=$SHARDS_PER_UNIT needs $EXPECTED_TASKS."
    echo "       Fix the --array line to:  --array=0-$(( EXPECTED_TASKS - 1 ))"
    exit 1
fi

# Concurrency sanity check. SLURM does not expose the %N throttle to the job, so
# this cannot verify it directly — but a task count over the cap with no throttle
# is the mistake worth catching, and saying so once per task is cheap.
MAX_CONCURRENT="${MAX_CONCURRENT:-100}"
if [ "$EXPECTED_TASKS" -gt "$MAX_CONCURRENT" ]; then
    echo "[founder] NOTE: $EXPECTED_TASKS tasks exceeds the ${MAX_CONCURRENT}-core"
    echo "[founder]       allowance. That is fine ONLY if --array carries a %N"
    echo "[founder]       throttle (e.g. --array=0-$(( EXPECTED_TASKS - 1 ))%96)."
    echo "[founder]       Without one this job is asking for $EXPECTED_TASKS cores."
fi

TASK="${SLURM_ARRAY_TASK_ID}"
UNIT_IDX=$(( TASK / SHARDS_PER_UNIT ))
SHARD=$((    TASK % SHARDS_PER_UNIT ))

if [ "$UNIT_IDX" -ge "$N_UNITS" ]; then
    echo "[array] task $TASK is past the end of the unit list — nothing to do."
    exit 0
fi

OUT_DIR="${UNITS[$UNIT_IDX]}"
UNIT=$(basename "$OUT_DIR")
PARAMS="$OUT_DIR/params.json"
JOBS="$OUT_DIR/jobs.json"
OUT_CSV="$OUT_DIR/results_shard_${SHARD}.csv"

if [ ! -f "$PARAMS" ]; then
    echo "ERROR: $PARAMS not found — rebuild the units."
    exit 1
fi

# hidden_size guard: see the h128 note in the header. A missing override is the
# one failure mode that produces plausible-looking numbers from the wrong network.
if ! grep -q '"hidden_size"' "$JOBS"; then
    echo "ERROR: $JOBS has no hidden_size in config_overrides."
    echo "       These are h128 runs; without it run_probe.js builds a 64-wide"
    echo "       network and mis-reads every 7814-long genome. Rebuild."
    exit 1
fi

# metrics guard: the assimilation analysis needs mean_weight_drift, which
# run_probe.js only writes when the jobs file asks for it. Without this the run
# would finish and simply be missing the column the whole test turns on.
if ! grep -q '"metrics"[[:space:]]*:[[:space:]]*"founder"' "$JOBS"; then
    echo "ERROR: $JOBS does not request \"metrics\": \"founder\", so"
    echo "       mean_weight_drift would not be recorded. Rebuild with"
    echo "       assimilation/make_founder_jobs.py."
    exit 1
fi

# Ordering guard: see SHARD COUNT IS FREE above. The RL-off jobs must form a
# contiguous PREFIX of the list. If they alternate instead, an even
# SHARDS_PER_UNIT makes every shard single-pass and the slow (RL-on) half hits
# the walltime; if they came last, a truncation would eat the primary
# measurement rather than the secondary one.
ORDER_OK=$("$NODE_BIN" -e '
    const j = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8")).jobs;
    let seenOn = false, ok = true;
    for (const x of j) {
        const on = x.id.endsWith("_on");
        if (on) seenOn = true; else if (seenOn) { ok = false; break; }
    }
    process.stdout.write(ok ? "1" : "0");
' "$JOBS" 2>/dev/null || echo "1")
if [ "$ORDER_OK" != "1" ]; then
    echo "ERROR: $JOBS is not ordered RL-off-first — an RL-off job appears after"
    echo "       an RL-on one. With SHARDS_PER_UNIT even this splits the array"
    echo "       into a fast half and a slow half, and a walltime kill would cost"
    echo "       RL-off probes (the primary measurement) instead of RL-on ones."
    echo "       Rebuild with assimilation/make_founder_jobs.py."
    exit 1
fi

# Stale-results guard: a shard CSV whose ids are not founder probe ids is left
# over from an older jobs.json. Resuming on top of it would keep rows that no
# longer correspond to these genomes.
#   founder ids: g{gen}_{founder_index}_r{r}_{on|off}
if [ -s "$OUT_CSV" ]; then
    FIRST_ID=$(awk -F',' 'NR==2{print $1; exit}' "$OUT_CSV")
    if [ -n "$FIRST_ID" ] && ! echo "$FIRST_ID" | grep -Eq '^g[0-9]+_[0-9]+_r[0-9]+_(on|off)$'; then
        echo "ERROR: $OUT_CSV holds ids like '$FIRST_ID', which are not founder"
        echo "       probe ids — stale output from an older jobs.json. Delete first:"
        echo "       rm -f $OUT_DIR/results_shard_*.csv $OUT_DIR/results.csv"
        exit 1
    fi
fi

echo "[founder] task=$TASK  unit=$UNIT_IDX/$N_UNITS  shard=${SHARD}/${SHARDS_PER_UNIT}"
echo "[founder]   $UNIT"
echo "[founder]   -> $OUT_CSV"

"$NODE_BIN" src/eval/run_probe.js \
    --params "$PARAMS" \
    --jobs   "$JOBS" \
    --out    "$OUT_CSV" \
    --shard  "${SHARD}/${SHARDS_PER_UNIT}"
rc=$?
echo "[founder] task=$TASK done (exit $rc): $UNIT shard $SHARD"
exit $rc
