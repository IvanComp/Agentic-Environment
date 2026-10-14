#!/usr/bin/env bash
set -euo pipefail

main() {
    local repo_dir campaigns campaign
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "${repo_dir}"
    if [[ "${AGENTIC_INSPECT_UPDATED:-0}" != "1" ]]; then
        git fetch -q origin
        git reset -q --hard origin/main
        AGENTIC_INSPECT_UPDATED=1 exec "${repo_dir}/leonardo/inspect.sh" "$@"
    fi
    campaigns="${WORK}/Agentic-Environment/campaigns"
    campaign="${1:-$(ls -t "${campaigns}" | head -n 1)}"
    export AGENTIC_SOURCE_DIR="${repo_dir}"
    export AGENTIC_CACHE_HOME="${AGENTIC_CACHE_HOME:-${FAST:-${WORK}}/Agentic-Environment/model-cache}"
    export AGENTIC_OUTPUT_ROOT="${campaigns}/${campaign}"
    "${repo_dir}/leonardo/container_exec.sh" python3 -m agentic_quant.show_runs "${campaigns}/${campaign}" --inspect
}

main "$@"
