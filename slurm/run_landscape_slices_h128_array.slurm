#!/bin/bash
# ============================================================================
# run_landscape_slices_h128_array.slurm — probe every landscape SLICE under
# landscape/out/slices_h128_hard/.
#
# These are the fixed-extent / fixed-resolution slices written by
#   python landscape/make_slice_jobs.py --logs-root logs --condition evolution \
#       --n-per-group 3 --out-root landscape/out/slices_h128_hard
#   python landscape/make_slice_jobs.py --logs-root logs --condition learning  \
#       --n-per-group 3 --out-root landscape/out/slices_h128_hard
# Generate them LOCALLY (needs Python + the genome.csv files), then transfer only
# landscape/out/slices_h128_hard/ to the cluster. Nothing else from logs/ is
# needed — see SELF-CONTAINED below.
#
#     sbatch run_landscape_slices_h128_array.slurm
#
# (sbatch SCHEDULES this file — you do not execute it yourself.)
#
# WHAT ONE SLICE IS
# -----------------
# A 2D plane through the 7814-dim weight space, anchored on ONE run's trajectory
# mean and spanned by that run's own best two directions. Each grid cell is a
# genome (anchor + alpha*u + beta*v) probed for fitness with RL off and RL on.
# Slices differ ONLY in where they cut — extent and resolution are pinned across
# all of them so slice_metrics.py can attribute any difference to the cut itself.
#
#   9 slices x 31x31 cells x R=5 repeats x 2 RL passes = 9,610 probes each
#   86,490 probes total
#
# SELF-CONTAINED SLICE DIRS — no logs tree needed here
# -----------------------------------------------------
# Unlike run_landscape_stage10_array.slurm, this script does NOT need $LOGS_ROOT.
# make_slice_jobs.py copies the anchor run's params.json INTO each slice dir, so
# every dir carries jobs.json + params.json + meta.json + plane.json and needs
# nothing outside itself. Transfer the one directory and submit.
#
# h128 — THE THING THAT SILENTLY BREAKS IF IT IS WRONG
# ------------------------------------------------------
# These runs are hidden_size=128 (genome 7814); the older landscape work was all
# hidden_size=64 (genome 3910). jobs.json carries
# config_overrides.hidden_size=128, and run_probe.js applies it when building the
# network. If you ever hand-edit a jobs.json, do not drop that key: a 64-wide
# brain reading a 7814-long genome fails quietly rather than loudly. The
# preflight below asserts it is present before spending any walltime.
#
# CORES / TIME
# ------------
# At the 4-7 s/probe measured on the Stage-3 planes:
#     9 units x 11 shards = 99 tasks, 874 probes each
#     ~58-102 min per task, one wave, ~132 core-hours total.
# 99 tasks deliberately stays UNDER 100 concurrent cores. The array is still one
# wave, so the wall clock is one task's runtime — packing fewer, longer tasks
# costs cores-in-use, not finishing time, as long as they all start together.
# --time is PER TASK, not for the whole array; 6h against a ~102 min worst case
# is ~3.5x headroom. A task killed on walltime is cheap: run_probe.js skips job
# ids already present in its shard CSV, so resubmitting resumes, not restarts.
#
# SHARDS_PER_UNIT MUST BE ODD — this is not a preference, it is arithmetic
# ------------------------------------------------------------------------
# jobs.json alternates the two RL passes job by job:
#     idx 0 p0_0_r0_OFF, 1 p0_0_r0_ON, 2 p0_0_r1_OFF, 3 p0_0_r1_ON, ...
# so an even index is always an RL-off probe and an odd index always an RL-on
# one. run_probe.js shards with `idx % shardN`, and a modulus that is EVEN
# preserves parity: every even shard then draws ONLY RL-off jobs and every odd
# shard ONLY RL-on jobs.
#
# That splits the array into a fast half and a slow half instead of NP tasks of
# equal cost. RL-on carries in-lifetime learning and measured ~17 s/probe here
# against ~7 s for RL-off, so the odd tasks need roughly 2x the estimate below
# while the even ones finish early. It is not a crash — the RL-off grid comes
# back complete and only the RL-on pass is truncated, which looks like a data
# problem rather than a scheduling one.
#
# It has bitten this script once: the 2026-08 companion run at SHARDS_PER_UNIT=16
# returned every RL-off cell and lost 1.5-6.3% of the RL-on probes on 9 of 12
# slices, each odd shard stopping cleanly at ~87-92% on the 6h wall.
#
# An ODD modulus is coprime with 2, so each shard cycles through both parities
# and every task gets the same off/on mix. The preflight below refuses an even
# value.
#
# To repack, change SHARDS_PER_UNIT and the --array upper bound TOGETHER — they
# must stay in sync (the preflight refuses to run if they disagree):
#     SHARDS_PER_UNIT=11 -> --array=0-98     99 tasks ~58-102 min  <- DEFAULT
#     SHARDS_PER_UNIT=9  -> --array=0-80     81 tasks ~71-125 min
#     SHARDS_PER_UNIT=7  -> --array=0-62     63 tasks ~92-160 min
#     SHARDS_PER_UNIT=13 -> --array=0-116   117 tasks ~49-86 min  (over 100 cores)
# Every one of these fits inside the 6h --time; only drop below 7 shards if you
# raise it. (The earlier table offered 10, 8 and 6 — all even, all subject to the
# parity split above. Do not use them.)
#
# If you regenerated the slices with a different --n-per-group, the unit count
# changes and so must --array. The preflight prints the exact number to use.
#
# THE COMPANION SET IS A DIFFERENT SHAPE — pass SLICE_ROOT and resize the array
# ----------------------------------------------------------------------------
# make_slice_jobs.py --companion puts the matched run of the OTHER condition on
# each plane, which widens the grid to extent 43 / resolution 47 and takes a
# slice from 9,610 probes to 22,090 (2.3x). It also makes fewer units: only the
# evolution runs are anchors, so --n-per-group 3 gives 6 slices, not 9.
#
#     SLICE_ROOT=landscape/out/slices_h128_companion SHARDS_PER_UNIT=15 \
#         sbatch --array=0-89 run_landscape_slices_h128_array.slurm
#
# 15, not 16: see SHARDS_PER_UNIT MUST BE ODD above. 6 units x 15 shards = 90
# tasks of 1,473 probes, each an even mix of the fast and slow pass.
#
#     6 units x 16 shards = 96 tasks, 1,380 probes each, ~1.5-2.7 h per task,
#     ~202 core-hours total. Still one wave under 100 cores, still inside the 6h
#     --time. The env vars propagate because sbatch exports the submitting
#     environment by default, and --array on the command line overrides the
#     #SBATCH line below.
#
# Smoke-tested before submission: one 45-probe shard of the seed-2004 companion
# slice returned 45 finite rows at 100 clones each, f = 2.42-5.67.
#
# AFTER the array finishes — merge + analyse LOCALLY (needs Python, seconds):
#   for d in landscape/out/slices_h128_hard/*/; do \
#       python landscape/merge_shards.py --out-dir "$d"; done
#   python landscape/slice_metrics.py \
#       $(for d in landscape/out/slices_h128_hard/*/; do \
#           n=$(basename "$d"); echo "--slice hard:${n:0:24}:$d"; done) \
#       --k-json landscape/out/stage9_calib_hard_grid/calibration.json \
#       --out-dir landscape/out/slice_metrics_h128
# ============================================================================
#SBATCH --job-name="lscape_slices_h128"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-98
#SBATCH --output=logs/lscape_slices_%A_%a.out
#SBATCH --error=logs/lscape_slices_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/LifeEngine}}"

# Parent of the slice dirs. Every immediate sub-dir holding a jobs.json is a unit.
SLICE_ROOT="${SLICE_ROOT:-landscape/out/slices_h128_hard}"
SHARDS_PER_UNIT="${SHARDS_PER_UNIT:-11}"

# ── Preflight ────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi
if [ ! -d "$SLICE_ROOT" ]; then
    echo "ERROR: $SLICE_ROOT not found. Generate the slices locally with"
    echo "       landscape/make_slice_jobs.py and transfer that directory here."
    exit 1
fi

# Units: every sub-dir with a jobs.json, in a STABLE sorted order. The order must
# not change between the submitting task and any resubmission, or a shard would
# resume against a different slice's CSV.
UNITS=()
while IFS= read -r d; do UNITS+=("$d"); done < <(
    find "$SLICE_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
UNITS_WITH_JOBS=()
for d in "${UNITS[@]}"; do
    [ -f "$d/jobs.json" ] && UNITS_WITH_JOBS+=("$d")
done
UNITS=("${UNITS_WITH_JOBS[@]}")
N_UNITS=${#UNITS[@]}

if [ "$N_UNITS" -eq 0 ]; then
    echo "ERROR: no sub-dir of $SLICE_ROOT contains a jobs.json."
    exit 1
fi

# Parity guard: an even modulus makes every shard single-pass. See the
# SHARDS_PER_UNIT MUST BE ODD block in the header for why that truncates the
# RL-on grid and leaves the RL-off one looking perfectly healthy.
# ALLOW_EVEN_SHARDS=1 is the RESUME escape hatch, and it is the right thing to
# use when finishing a run that was already sharded evenly. run_probe.js skips
# ids already in ITS OWN shard CSV, so a resume only picks up where a task
# stopped when the modulus is unchanged. Switching to an odd modulus mid-run
# re-partitions the jobs, so completed probes land in a different shard's file,
# are not recognised as done, and are re-run — correct but expensive. Finish an
# even-sharded run even, then use an odd value for the next fresh one.
if [ $(( SHARDS_PER_UNIT % 2 )) -eq 0 ] && [ "${ALLOW_EVEN_SHARDS:-0}" != "1" ]; then
    echo "ERROR: SHARDS_PER_UNIT=$SHARDS_PER_UNIT is EVEN."
    echo "       jobs.json alternates RL-off / RL-on, and run_probe.js shards on"
    echo "       idx % $SHARDS_PER_UNIT, so an even modulus gives every even shard"
    echo "       only RL-off jobs and every odd shard only RL-on jobs. The RL-on"
    echo "       tasks then take ~2x as long and hit the walltime, silently"
    echo "       truncating the RL-on grid while RL-off comes back complete."
    echo "       Use an odd value, e.g. SHARDS_PER_UNIT=$(( SHARDS_PER_UNIT - 1 ))"
    echo "       with --array=0-$(( N_UNITS * (SHARDS_PER_UNIT - 1) - 1 ))."
    echo
    echo "       To RESUME a run that was already sharded evenly, keep the same"
    echo "       modulus and set ALLOW_EVEN_SHARDS=1 — changing it now would"
    echo "       re-partition the jobs and re-run probes that are already done."
    exit 1
fi
if [ $(( SHARDS_PER_UNIT % 2 )) -eq 0 ]; then
    echo "[slice] WARNING: SHARDS_PER_UNIT=$SHARDS_PER_UNIT is even — even shards"
    echo "[slice]          are RL-off only, odd shards RL-on only. Proceeding"
    echo "[slice]          because ALLOW_EVEN_SHARDS=1 (resume of an existing run)."
fi

EXPECTED_TASKS=$(( N_UNITS * SHARDS_PER_UNIT ))
ARRAY_COUNT="${SLURM_ARRAY_TASK_COUNT:-$EXPECTED_TASKS}"
if [ "$ARRAY_COUNT" -ne "$EXPECTED_TASKS" ]; then
    echo "ERROR: --array covers $ARRAY_COUNT tasks but $N_UNITS units x"
    echo "       SHARDS_PER_UNIT=$SHARDS_PER_UNIT needs $EXPECTED_TASKS."
    echo "       Fix the #SBATCH --array line to:  --array=0-$(( EXPECTED_TASKS - 1 ))"
    exit 1
fi

# ── Map this task -> (unit, shard) ───────────────────────────────────────────
# Unit-major: at the default 11 shards, tasks 0..10 are unit 0's, 11..21 unit
# 1's, and so on. Repacking is safe against a half-finished run: merge_shards.py
# dedups by job id, and every shard of the same jobs.json probes the same genomes
# under the same config, so rows written under a different SHARDS_PER_UNIT stay
# valid and are simply deduped.
TASK="${SLURM_ARRAY_TASK_ID}"
UNIT_IDX=$(( TASK / SHARDS_PER_UNIT ))
SHARD=$((    TASK % SHARDS_PER_UNIT ))

if [ "$UNIT_IDX" -ge "$N_UNITS" ]; then
    echo "[array] task $TASK is past the end of the grid — nothing to do."
    exit 0
fi

OUT_DIR="${UNITS[$UNIT_IDX]}"
UNIT=$(basename "$OUT_DIR")
PARAMS="$OUT_DIR/params.json"
OUT_CSV="$OUT_DIR/results_shard_${SHARD}.csv"

if [ ! -f "$PARAMS" ]; then
    echo "ERROR: $PARAMS not found. make_slice_jobs.py copies the anchor run's"
    echo "       params.json into each slice dir — regenerate the slices."
    exit 1
fi

# hidden_size guard: see the h128 note in the header. A missing override is the
# one failure mode that produces plausible-looking numbers from the wrong network.
if ! grep -q '"hidden_size"' "$OUT_DIR/jobs.json"; then
    echo "ERROR: $OUT_DIR/jobs.json has no hidden_size in config_overrides."
    echo "       These are h128 runs; without it run_probe.js builds a 64-wide"
    echo "       network and mis-reads every 7814-long genome. Regenerate."
    exit 1
fi

# Stale-results guard: a shard CSV whose ids do not look like grid probe ids is
# left over from an older jobs.json. Resuming on top of it would keep rows that
# no longer correspond to these genomes, so refuse rather than silently merge.
#   grid ids: p{i}_{j}_r{r}_{on|off}
if [ -s "$OUT_CSV" ]; then
    FIRST_ID=$(awk -F',' 'NR==2{print $1; exit}' "$OUT_CSV")
    if [ -n "$FIRST_ID" ] && ! echo "$FIRST_ID" | grep -Eq '^p[0-9]+_[0-9]+_r[0-9]+_(on|off)$'; then
        echo "ERROR: $OUT_CSV holds ids like '$FIRST_ID', which are not grid probe ids"
        echo "       — stale output from an older jobs.json. Delete it first:"
        echo "       rm -f $OUT_DIR/results_shard_*.csv $OUT_DIR/results.csv"
        exit 1
    fi
fi

echo "[slice] task=$TASK  unit=$UNIT_IDX/$N_UNITS  shard=${SHARD}/${SHARDS_PER_UNIT}"
echo "[slice]   $UNIT"
echo "[slice]   params=$PARAMS"
echo "[slice]   -> $OUT_CSV"

"$NODE_BIN" src/eval/run_probe.js \
    --params "$PARAMS" \
    --jobs   "$OUT_DIR/jobs.json" \
    --out    "$OUT_CSV" \
    --shard  "${SHARD}/${SHARDS_PER_UNIT}"
rc=$?
echo "[slice] task=$TASK done (exit $rc): $UNIT shard $SHARD"
exit $rc
