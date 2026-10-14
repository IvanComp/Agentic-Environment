#!/usr/bin/env bash
set -euo pipefail

main() {
    local repo_dir log cache
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    log="$(ls -t "${repo_dir}"/leonardo/logs/model-*.out 2>/dev/null | head -n 1 || true)"
    if [[ -z "${log}" ]]; then
        echo "No model download job yet."
        return
    fi
    echo "Log: ${log}"
    grep "model-download" "${log}" | tail -n 1 || echo "Download not started yet."
    grep -q "MODEL READY" "${log}" && echo "Model ready."
    tail -n 3 "${log%.out}.err" 2>/dev/null | tr '\r' '\n' | tail -n 3 || true
    cache="${AGENTIC_CACHE_HOME:-${FAST:-${WORK}}/Agentic-Environment/model-cache}"
    du -sh "${cache}/models" "${cache}/ollama" 2>/dev/null || true
}

main "$@"
