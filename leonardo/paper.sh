#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"
campaigns="${WORK:?WORK is not defined}/Agentic-Environment/campaigns"
out="${campaigns}/paper"
mkdir -p "${out}"
export AGENTIC_SOURCE_DIR="${repo_dir}"
export AGENTIC_CACHE_HOME="${AGENTIC_CACHE_HOME:-${FAST:-${WORK}}/Agentic-Environment/model-cache}"
export AGENTIC_OUTPUT_ROOT="${campaigns}"
"${repo_dir}/leonardo/container_exec.sh" python3 -m agentic_quant.paper_results "${campaigns}" --out "${out}"
(cd "${campaigns}" && find . -path ./paper -prune -o -name results.jsonl -print | tar -czf "${out}/data.tar.gz" -T -)
echo
ls -l "${out}/tables" "${out}/data.tar.gz"
echo "Tables: ${out}/tables"
echo "Raw results for results/data on your Mac: ${out}/data.tar.gz"
