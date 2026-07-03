#!/usr/bin/env bash
# ============================================================================
# sweep_lr.sh — learning-rate sweep for the BASELINE (no-predator) condition.
#
# Submits one independent HPC job per (learning_rate, seed) combination:
#     3 learning rates  ×  3 seeds  =  9 jobs
# Learning rate is the ONLY variable that changes. The 3 seeds are shared
# across every learning rate, so each LR is evaluated on the SAME 3 maps
# (map = controlled replicate, not a confound). Grid is 500×500, learning
# condition, and predators are fully disabled (patrol + roaming = 0).
#
# The scheduler is auto-detected: sbatch (SLURM / UCT hex) or qsub (PBS / CHPC).
#
#   ./sweep_lr.sh                 # submit all 9 jobs (500 generations each)
#   GENERATIONS=200 ./sweep_lr.sh # override run length for every job
#   DRY_RUN=1 ./sweep_lr.sh       # print the submit commands without submitting
#
# Each job writes CSVs + params.txt to
#   logs/learning/standard/auto-run/lr<LR>_seed<SEED>/
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ── Sweep grid ──────────────────────────────────────────────────────────────
LEARNING_RATES=(0.0001 0.001 0.01)   # 1e-4, 1e-3, 1e-2
SEEDS=(1 42 999)                     # 3 maps; cached pools already exist

# ── Fixed factors (identical for every job) ─────────────────────────────────
WIDTH="${WIDTH:-500}"
HEIGHT="${HEIGHT:-500}"
GENERATIONS="${GENERATIONS:-500}"
CONDITION=learning
MODE=standard
PREDATORS_PER_PATCH=0                 # baseline: no patrol predators
ROAMING_PREDATORS=0                   # baseline: no roaming predators

# ── Detect scheduler ────────────────────────────────────────────────────────
if   command -v sbatch >/dev/null 2>&1; then SCHED=slurm; JOB=run_simulation.slurm
elif command -v qsub   >/dev/null 2>&1; then SCHED=pbs;   JOB=run_simulation.pbs
else echo "ERROR: neither sbatch nor qsub found on PATH." >&2; exit 1
fi
echo "[sweep] scheduler=$SCHED  job=$JOB  grid=${WIDTH}x${HEIGHT}  gens=$GENERATIONS"

# ── Submit loop ─────────────────────────────────────────────────────────────
for lr in "${LEARNING_RATES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    RUN_NAME="lr${lr}_seed${seed}"
    VARS="MAP_SEED=${seed},WIDTH=${WIDTH},HEIGHT=${HEIGHT},GENERATIONS=${GENERATIONS}"
    VARS="${VARS},CONDITION=${CONDITION},MODE=${MODE},LEARNING_RATE=${lr}"
    VARS="${VARS},PREDATORS_PER_PATCH=${PREDATORS_PER_PATCH},ROAMING_PREDATORS=${ROAMING_PREDATORS}"
    VARS="${VARS},RUN_NAME=${RUN_NAME}"

    if [ "$SCHED" = slurm ]; then
      CMD=(sbatch --job-name="$RUN_NAME" --export="ALL,${VARS}" "$JOB")
    else
      CMD=(qsub -N "$RUN_NAME" -v "$VARS" "$JOB")
    fi

    echo "[sweep] $RUN_NAME"
    if [ "${DRY_RUN:-0}" = 1 ]; then printf '   '; printf '%q ' "${CMD[@]}"; echo; else "${CMD[@]}"; fi
  done
done

echo "[sweep] submitted ${#LEARNING_RATES[@]}x${#SEEDS[@]} = $(( ${#LEARNING_RATES[@]} * ${#SEEDS[@]} )) jobs."
