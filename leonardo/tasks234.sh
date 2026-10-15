#!/usr/bin/env bash
set -euo pipefail

MODELS=(llama3.3:70b qwen3:32b mistral-small3.2:24b qwen3:14b qwen3:8b llama3.1:8b)

cd "$(dirname "${BASH_SOURCE[0]}")/.."
git fetch -q origin
git reset -q --hard origin/main
if (( $# > 0 )); then
    MODELS=("$@")
fi
AGENTIC_NIGHT_NAME=tasks234 \
AGENTIC_EXPERIMENT_FILE=experiment-tasks234.args \
AGENTIC_KEEP_RESULTS=1 \
AGENTIC_NIGHT_WAVES=10 \
AGENTIC_NIGHT_TIME=24:00:00 \
exec ./leonardo/night.sh "${MODELS[@]}"
