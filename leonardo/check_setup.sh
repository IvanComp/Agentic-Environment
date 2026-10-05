#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sif_path="${AGENTIC_SIF_PATH:-${repo_dir}/leonardo/agentic-quant.sif}"
failed=0

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        echo "OK: $1"
    else
        echo "MISSING: $1" >&2
        failed=1
    fi
}

echo "Agentic Quant Leonardo pre-flight"
echo "Repository: ${repo_dir}"
check_command sbatch
check_command squeue
check_command flock
if command -v singularity >/dev/null 2>&1; then
    runtime=singularity
elif command -v apptainer >/dev/null 2>&1; then
    runtime=apptainer
else
    echo "MISSING: Singularity or Apptainer" >&2
    failed=1
    runtime=""
fi
if [[ -f "${sif_path}" ]]; then
    echo "OK: ${sif_path}"
else
    echo "MISSING: ${sif_path}" >&2
    failed=1
fi
if [[ -n "${WORK:-}" ]]; then
    echo "OK: WORK=${WORK}"
else
    echo "MISSING: WORK" >&2
    failed=1
fi
if (( failed != 0 )); then
    echo "PRE-FLIGHT FAILED" >&2
    exit 1
fi

"${runtime}" inspect "${sif_path}" >/dev/null
"${runtime}" exec --bind "${repo_dir}:/workspace" --pwd /workspace \
    --env PYTHONPATH=/workspace/src "${sif_path}" \
    python3 -m agentic_quant.run_benchmark --list-models >/dev/null
echo "PRE-FLIGHT PASSED"
