#!/usr/bin/env bash
# ============================================================================
# run_local.sh — test the PBS job locally before submitting to the CHPC.
#
# The #PBS lines in run_simulation.pbs are just bash comments, so this runs the
# exact same command body locally — using `node` from your PATH, the current
# directory as the project dir, and a short generation count for a quick smoke
# test. If this works, the real PBS job will too.
#
#   ./run_local.sh                       # quick 2-generation smoke test
#   GENERATIONS=10 MODE=pure_rl ./run_local.sh
#   MAP_SEED=7 CONDITION=evolution HIDDEN_SIZE=64 ./run_local.sh
#
# Any parameter accepted by run_simulation.pbs can be set the same way here.
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")"

# Use the system node and this directory; keep the test short by default.
export NODE_BIN="${NODE_BIN:-node}"
export PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
export GENERATIONS="${GENERATIONS:-2}"
export WIDTH="${WIDTH:-400}"
export HEIGHT="${HEIGHT:-300}"
export LOG_EVERY="${LOG_EVERY:-5000}"

echo "[run_local] smoke test: ${GENERATIONS} generations, grid ${WIDTH}x${HEIGHT}"
bash run_simulation.pbs
