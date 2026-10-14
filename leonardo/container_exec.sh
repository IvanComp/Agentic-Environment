#!/usr/bin/env bash
set -euo pipefail

source_dir="${AGENTIC_SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
sif_path="${AGENTIC_SIF_PATH:-${source_dir}/leonardo/agentic-quant.sif}"
cache_home="${AGENTIC_CACHE_HOME:?AGENTIC_CACHE_HOME must point to the shared model-cache root}"

if command -v singularity >/dev/null 2>&1; then
    runtime=singularity
elif command -v apptainer >/dev/null 2>&1; then
    runtime=apptainer
else
    echo "Singularity or Apptainer is required." >&2
    exit 2
fi

args=(exec)
if [[ "${AGENTIC_CONTAINER_NV:-0}" == "1" ]]; then
    args+=(--nv)
fi
args+=(
    --bind "${source_dir}:/workspace"
    --bind "${cache_home}:${cache_home}"
    --pwd /workspace
    --env "PYTHONPATH=/workspace/src"
    --env "AGENTIC_QUANT_HOME=${cache_home}"
    --env "AGENTIC_QUANT_MODELS_DIR=${cache_home}/models"
    --env "OLLAMA_MODELS=${cache_home}/ollama"
    --env "OLLAMA_HOST=${OLLAMA_HOST:-http://127.0.0.1:11434}"
)
if [[ -n "${AGENTIC_OLLAMA_CUDA_DIR:-}" ]]; then
    args+=(--bind "${AGENTIC_OLLAMA_CUDA_DIR}:/usr/lib/ollama/cuda_v13")
fi
if [[ -n "${AGENTIC_OUTPUT_ROOT:-}" ]]; then
    args+=(--bind "${AGENTIC_OUTPUT_ROOT}:${AGENTIC_OUTPUT_ROOT}")
fi

exec "${runtime}" "${args[@]}" "${sif_path}" "$@"
