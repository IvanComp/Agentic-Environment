#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"
root="${WORK:?WORK is not defined}/Agentic-Environment/campaigns/${1:-night}"
export AGENTIC_SOURCE_DIR="${repo_dir}"
export AGENTIC_CACHE_HOME="${AGENTIC_CACHE_HOME:-${FAST:-${WORK}}/Agentic-Environment/model-cache}"
export AGENTIC_OUTPUT_ROOT="${root}"
clear
"${repo_dir}/leonardo/container_exec.sh" python3 leonardo/digest.py "${root}"
