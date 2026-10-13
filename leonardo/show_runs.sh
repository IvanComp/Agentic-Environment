#!/usr/bin/env bash
set -euo pipefail

campaign="${1:?usage: leonardo/show_runs.sh <campaign-name> [architecture]}"
architecture="${2:-unprotected}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${AGENTIC_RESULTS_ROOT:-${WORK}/Agentic-Environment/campaigns}/${campaign}"
cache_base="${FAST:-${WORK}}"

export AGENTIC_SOURCE_DIR="${repo_dir}"
export AGENTIC_CACHE_HOME="${AGENTIC_CACHE_HOME:-${cache_base}/Agentic-Environment/model-cache}"
export AGENTIC_OUTPUT_ROOT="${root}"
mkdir -p "${AGENTIC_CACHE_HOME}"

"${repo_dir}/leonardo/container_exec.sh" python3 -m agentic_quant.show_runs "${root}" \
    --architecture "${architecture}" --out "${root}/runs.txt"
echo "Saved to ${root}/runs.txt"
