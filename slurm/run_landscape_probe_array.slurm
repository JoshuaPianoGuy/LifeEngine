#!/bin/bash
# ============================================================================
# run_landscape_probe_array.slurm — GENERIC sharded runner for the fitness-
# landscape / ruggedness pipeline (src/eval/run_probe.js).
#
# Every stage (Stage 2 calibration, Stage 3/4 grids, Stage 6 lambda walk) reduces
# to the SAME job: evaluate one jobs.json of f(theta) probes. This one array
# script runs any of them — you point it at an OUT_DIR that already contains a
# jobs.json (written by the corresponding landscape/stageN_*.py ... jobs step),
# and it splits the jobs across array tasks with --shard i/N.
#
# WORKFLOW
# --------
#   1) On the login node (or locally), generate the jobs file into a scratch dir:
#        cd $SCRATCH/LifeEngine
#        python landscape/stage3_pca.py --evo-genome-csv ... --learn-genome-csv ... \
#               --out-dir landscape/out/stage3_hard_coarse --grid coarse \
#               --resolution 25 --repeats 10 \
#               --config-overrides '{"roaming_predator_count":60,"predator_drain":5}'
#   2) Submit this array, one task per shard:
#        OUT_DIR=landscape/out/stage3_hard_coarse \
#        PARAMS=logs/learning/standard/auto-run/roam60d5_g1k_learning_noeps_lr0.02_lscape_seed999_r1/params.json \
#        N_SHARDS=64 sbatch --array=0-63 run_landscape_probe_array.slurm
#      (N_SHARDS MUST equal the array size. Each task writes results_shard_<i>.csv.)
#   3) After the array finishes, merge shards and analyse/plot:
#        python landscape/merge_shards.py --out-dir landscape/out/stage3_hard_coarse
#        python landscape/stage7_plots.py --out-dir landscape/out/stage3_hard_coarse
#
# PARAMS chooses the ENVIRONMENT + RL hyperparameters the probe runs under. Use
# the LEARNING run's params.json for grids (so the RL-on pass matches the learning
# condition); config_overrides inside jobs.json can retarget predators per plane.
# Re-running is safe: run_probe.js skips job ids already present in the shard CSV.
#
#   PACKING: one array task == one shard. Pick N_SHARDS so each shard is a few
#   hours (a shard of J/N jobs at ~10s/job). e.g. 12500 jobs / 64 shards ~ 195
#   jobs/shard ~ 33 min. Bigger sweeps -> more shards.
# ============================================================================
#SBATCH --job-name="landscape_probe"
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --array=0-63
#SBATCH --output=logs/landscape_probe_%A_%a.out
#SBATCH --error=logs/landscape_probe_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dgrjos001@myuct.ac.za

# ── Environment (EDIT for your account) ─────────────────────────────────────
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v16.20.2/bin/node}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/LifeEngine}}"

# ── Required inputs ──────────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:?set OUT_DIR to the dir containing jobs.json}"
PARAMS="${PARAMS:?set PARAMS to a run params.json (env + RL hyperparameters)}"
N_SHARDS="${N_SHARDS:?set N_SHARDS = array size (e.g. 64 for --array=0-63)}"

cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

if [ ! -x "$NODE_BIN" ] && ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "ERROR: Node binary not found at '$NODE_BIN'. Run 'nvm use 16 && which node' and set NODE_BIN."
    exit 1
fi
if [ ! -f "$OUT_DIR/jobs.json" ]; then
    echo "ERROR: $OUT_DIR/jobs.json not found. Generate it first with landscape/stageN_*.py ... jobs."
    exit 1
fi

SHARD="${SLURM_ARRAY_TASK_ID}"
OUT_CSV="$OUT_DIR/results_shard_${SHARD}.csv"

echo "[landscape] shard ${SHARD}/${N_SHARDS}  out_dir=$OUT_DIR  params=$PARAMS"
echo "[landscape] -> $OUT_CSV"

"$NODE_BIN" src/eval/run_probe.js \
    --params "$PARAMS" \
    --jobs   "$OUT_DIR/jobs.json" \
    --out    "$OUT_CSV" \
    --shard  "${SHARD}/${N_SHARDS}"
rc=$?
echo "[landscape] shard ${SHARD} done (exit $rc)"
exit $rc
