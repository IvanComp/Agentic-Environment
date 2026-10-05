#!/usr/bin/env bash
#SBATCH --job-name=aq-model
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=leonardo/logs/model-%j.out
#SBATCH --error=leonardo/logs/model-%j.err

set -euo pipefail

repo_dir="${AGENTIC_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
repo_dir="$(cd "${repo_dir}" && pwd)"
model="${AGENTIC_MODEL_KEY:?AGENTIC_MODEL_KEY is required}"
cache_home="${AGENTIC_CACHE_HOME:?AGENTIC_CACHE_HOME is required}"
sif_path="${AGENTIC_SIF_PATH:-${repo_dir}/leonardo/agentic-quant.sif}"
slug="$(printf '%s' "${model}" | tr -c 'A-Za-z0-9._-' '_')"
lock_root="${cache_home}/locks"

mkdir -p "${lock_root}" "${cache_home}/models" "${cache_home}/ollama"
exec 9>"${lock_root}/${slug}.lock"
flock 9

export AGENTIC_SOURCE_DIR="${repo_dir}"
export AGENTIC_SIF_PATH="${sif_path}"
export AGENTIC_CACHE_HOME="${cache_home}"
export OLLAMA_HOST="http://127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))"

backend="$("${repo_dir}/leonardo/container_exec.sh" python3 leonardo/model_cache.py backend "${model}")"
ollama_pid=""
stop_ollama() {
    if [[ -n "${ollama_pid}" ]]; then
        kill -TERM "${ollama_pid}" 2>/dev/null || true
        wait "${ollama_pid}" 2>/dev/null || true
    fi
}
trap stop_ollama EXIT TERM INT

if [[ "${backend}" == "ollama" ]]; then
    "${repo_dir}/leonardo/container_exec.sh" ollama serve &
    ollama_pid=$!
    "${repo_dir}/leonardo/container_exec.sh" python3 leonardo/model_cache.py wait-server "${model}" --timeout 180
fi

"${repo_dir}/leonardo/container_exec.sh" python3 leonardo/model_cache.py prepare "${model}"
echo "MODEL READY: ${model}"
