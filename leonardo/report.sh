#!/usr/bin/env bash
set -euo pipefail

main() {
    local repo_dir campaigns root folder
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "${repo_dir}"
    campaigns="${WORK}/Agentic-Environment/campaigns"
    root="${campaigns}/${1:-$(ls -t "${campaigns}" | head -n 1)}"
    for folder in "${root}"/*/; do
        [[ -f "${folder}jobs.txt" ]] || continue
        sacct -n -X -j "$(tr ' ' ',' <"${folder}jobs.txt")" --format=JobID%20,JobName%10,State%14,Elapsed%12,NodeList%14 >"${folder}sacct.txt" 2>&1 || true
    done
    export AGENTIC_SOURCE_DIR="${repo_dir}"
    export AGENTIC_CACHE_HOME="${AGENTIC_CACHE_HOME:-${FAST:-${WORK}}/Agentic-Environment/model-cache}"
    export AGENTIC_OUTPUT_ROOT="${root}"
    "${repo_dir}/leonardo/container_exec.sh" python3 leonardo/report.py "${root}" >"${root}/report.txt"
    head -n 20 "${root}/report.txt"
    echo "Full report: ${root}/report.txt"
}

main "$@"
