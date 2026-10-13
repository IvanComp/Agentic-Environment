#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
campaigns="${AGENTIC_RESULTS_ROOT:-${WORK}/Agentic-Environment/campaigns}"
campaign="${1:-$(ls -t "${campaigns}" 2>/dev/null | head -n 1)}"
root="${campaigns}/${campaign:?no campaign found in ${campaigns}}"

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
        for log in "${root}"/*/ollama-*.log; do
            if [[ -f "${log}" ]]; then
                echo "--- GPU seen by Ollama in $(basename "${log}")"
                grep "inference compute" "${log}" | head -n 4 || echo "none"
            fi
        done
    fi
fi
export AGENTIC_SOURCE_DIR="${repo_dir}"
export AGENTIC_CACHE_HOME="${AGENTIC_CACHE_HOME:-${FAST:-${WORK}}/Agentic-Environment/model-cache}"
export AGENTIC_OUTPUT_ROOT="${root}"
mkdir -p "${AGENTIC_CACHE_HOME}"

for model_dir in "${root}"/*/; do
    if compgen -G "${model_dir}wave_*/results.jsonl" >/dev/null; then
        "${repo_dir}/leonardo/container_exec.sh" python3 -m agentic_quant.metrics "${model_dir}" --out "${model_dir}metrics"
    fi
done
"${repo_dir}/leonardo/container_exec.sh" python3 -m agentic_quant.show_runs "${root}" --out "${root}/runs.txt" --brief
echo "Full details: ${root}/runs.txt"
