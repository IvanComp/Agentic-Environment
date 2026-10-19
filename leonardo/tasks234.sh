#!/usr/bin/env bash
set -euo pipefail

MODELS=(qwen3:8b qwen3:14b qwen3:32b)
NAME=tasks234b

cd "$(dirname "${BASH_SOURCE[0]}")/.."
git fetch -q origin
git reset -q --hard origin/main
if (( $# > 0 )); then
    MODELS=("$@")
fi
waves=10
echo "Waves 1-${waves} per model, completed waves are skipped."
AGENTIC_NIGHT_NAME="${NAME}" \
AGENTIC_EXPERIMENT_FILE=experiment-tasks234.args \
AGENTIC_KEEP_RESULTS=1 \
AGENTIC_NIGHT_WAVES="${waves}" \
AGENTIC_NIGHT_PER_JOB="${waves}" \
AGENTIC_NIGHT_TIME=24:00:00 \
exec ./leonardo/night.sh "${MODELS[@]}"
