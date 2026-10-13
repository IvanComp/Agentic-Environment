#!/usr/bin/env bash
set -euo pipefail

campaign="${1:-test}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${AGENTIC_RESULTS_ROOT:-${WORK}/Agentic-Environment/campaigns}/${campaign}"

if [[ -f "${root}/jobs.txt" ]]; then
    read -r prepare_job wave_job cleanup_job <"${root}/jobs.txt" || true
    states="$(sacct -X -n -j "${prepare_job},${wave_job},${cleanup_job}" --format=JobName%10,State%12,Elapsed%10)"
    echo "${states}"
    if grep -qv COMPLETED <<<"${states}"; then
        for log in "leonardo/logs/model-${prepare_job}.err" "leonardo/logs/wave-${wave_job}_1.err" "leonardo/logs/cleanup-${cleanup_job}.err"; do
            if [[ -s "${repo_dir}/${log}" ]]; then
                echo "--- ${log}"
                tail -n 15 "${repo_dir}/${log}"
            fi
        done
    fi
fi
for metrics in "${root}"/*/metrics/metrics.md; do
    if [[ -f "${metrics}" ]]; then
        cat "${metrics}"
    fi
done

export AGENTIC_SOURCE_DIR="${repo_dir}"
export AGENTIC_CACHE_HOME="${AGENTIC_CACHE_HOME:-${FAST:-${WORK}}/Agentic-Environment/model-cache}"
export AGENTIC_OUTPUT_ROOT="${root}"
mkdir -p "${AGENTIC_CACHE_HOME}"
"${repo_dir}/leonardo/container_exec.sh" python3 -m agentic_quant.show_runs "${root}" --out "${root}/runs.txt" --brief
echo "Full details: ${root}/runs.txt"
