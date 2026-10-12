#!/usr/bin/env bash
#SBATCH --job-name=aq-clean
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=leonardo/logs/cleanup-%j.out
#SBATCH --error=leonardo/logs/cleanup-%j.err

set -euo pipefail

repo_dir="${AGENTIC_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
repo_dir="$(cd "${repo_dir}" && pwd)"
model="${AGENTIC_MODEL_KEY:?AGENTIC_MODEL_KEY is required}"
cache_home="${AGENTIC_CACHE_HOME:?AGENTIC_CACHE_HOME is required}"
campaign_root="${AGENTIC_CAMPAIGN_ROOT:?AGENTIC_CAMPAIGN_ROOT is required}"
sif_path="${AGENTIC_SIF_PATH:-${repo_dir}/leonardo/agentic-quant.sif}"
seed_start="${AGENTIC_SEED_START:-0}"
seed_step="${AGENTIC_SEED_STEP:-1}"
slug="$(printf '%s' "${model}" | tr -c 'A-Za-z0-9._-' '_')"
lock_root="${cache_home}/locks"
active_dir="${lock_root}/${slug}.active"

python3 "${repo_dir}/leonardo/verify_campaign.py" "${campaign_root}" \
    --model "${model}" --waves "${AGENTIC_WAVES:-10}" --seed-start "${seed_start}" --seed-step "${seed_step}" \
    --write "${campaign_root}/campaign_verification.json" --require-complete

PYTHONPATH="${repo_dir}/src" python3 -m agentic_quant.metrics "${campaign_root}" --out "${campaign_root}/metrics" \
    || echo "Metrics not computed; run agentic-quant-metrics on ${campaign_root}." >&2

mkdir -p "${lock_root}"
exec 9>"${lock_root}/${slug}.lock"
flock 9

if [[ -d "${active_dir}" ]]; then
    for lease in "${active_dir}"/*; do
        [[ -e "${lease}" ]] || continue
        job_id="$(<"${lease}")"
        if squeue --noheader --jobs="${job_id}" 2>/dev/null | grep -q .; then
            echo "Model ${model} is still used by Slurm job ${job_id}; preserving it." >&2
            exit 1
        fi
        rm -f "${lease}"
    done
    rmdir "${active_dir}" 2>/dev/null || true
fi

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

"${repo_dir}/leonardo/container_exec.sh" python3 leonardo/model_cache.py delete "${model}"
echo "MODEL REMOVED AFTER VERIFIED CAMPAIGN: ${model}"
