#!/usr/bin/env bash
#SBATCH --job-name=aq-wave
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=leonardo/logs/wave-%A_%a.out
#SBATCH --error=leonardo/logs/wave-%A_%a.err

set -euo pipefail

repo_dir="${AGENTIC_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
repo_dir="$(cd "${repo_dir}" && pwd)"
model="${AGENTIC_MODEL_KEY:?AGENTIC_MODEL_KEY is required}"
cache_home="${AGENTIC_CACHE_HOME:?AGENTIC_CACHE_HOME is required}"
campaign_root="${AGENTIC_CAMPAIGN_ROOT:?AGENTIC_CAMPAIGN_ROOT is required}"
sif_path="${AGENTIC_SIF_PATH:-${repo_dir}/leonardo/agentic-quant.sif}"
seed_start="${AGENTIC_SEED_START:-0}"
seed_step="${AGENTIC_SEED_STEP:-1}"
total_waves="${AGENTIC_WAVES:-10}"
per_job="${AGENTIC_WAVES_PER_JOB:-1}"
task="${SLURM_ARRAY_TASK_ID}"
first_wave="$(( (task - 1) * per_job + 1 ))"
last_wave="$(( first_wave + per_job - 1 ))"
if (( last_wave > total_waves )); then
    last_wave="${total_waves}"
fi
slug="$(printf '%s' "${model}" | tr -c 'A-Za-z0-9._-' '_')"
lease_dir="${cache_home}/locks/${slug}.active"
lease="${lease_dir}/${SLURM_ARRAY_JOB_ID}_${task}"
task_repo="${TMPDIR:-/tmp}/agentic-quant-${SLURM_ARRAY_JOB_ID}-${task}"

wave_done() {
    local wave_dir="${campaign_root}/wave_$(printf '%02d' "$1")"
    [[ -f "${wave_dir}/.complete.json" && -s "${wave_dir}/summary.json" && -s "${wave_dir}/results.jsonl" ]]
}

pending=0
for (( wave = first_wave; wave <= last_wave; wave++ )); do
    wave_done "${wave}" || pending=1
done
if (( pending == 0 )); then
    echo "Waves ${first_wave}-${last_wave} already complete; skipping."
    exit 0
fi

mkdir -p "${lease_dir}" "${task_repo}"
printf '%s\n' "${SLURM_ARRAY_JOB_ID}_${task}" >"${lease}"
cp -a "${repo_dir}/src" "${repo_dir}/pyproject.toml" "${task_repo}/"
mkdir -p "${task_repo}/leonardo"
cp "${repo_dir}/leonardo/model_cache.py" "${repo_dir}/leonardo/experiment.args" "${task_repo}/leonardo/"

export AGENTIC_SOURCE_DIR="${task_repo}"
export AGENTIC_SIF_PATH="${sif_path}"
export AGENTIC_CACHE_HOME="${cache_home}"
export AGENTIC_OUTPUT_ROOT="${campaign_root}"
export AGENTIC_CONTAINER_NV=1
export OLLAMA_HOST="http://127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

ollama_pid=""
experiment_pid=""
cleanup() {
    if [[ -n "${experiment_pid}" ]]; then
        kill -TERM "${experiment_pid}" 2>/dev/null || true
        wait "${experiment_pid}" 2>/dev/null || true
    fi
    if [[ -n "${ollama_pid}" ]]; then
        kill -TERM "${ollama_pid}" 2>/dev/null || true
        wait "${ollama_pid}" 2>/dev/null || true
    fi
    rm -f "${lease}"
    rmdir "${lease_dir}" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

mkdir -p "${campaign_root}"
backend="$("${repo_dir}/leonardo/container_exec.sh" python3 leonardo/model_cache.py backend "${model}")"
if [[ "${backend}" == "ollama" ]]; then
    ollama_log="${campaign_root}/ollama-${SLURM_ARRAY_JOB_ID}_${task}.log"
    compat_dir="${cache_home}/cuda-compat-12-8"
    ollama_cuda_dir="${cache_home}/ollama-cuda12"
    cat "${ollama_cuda_dir}"/* "${compat_dir}"/* >/dev/null
    AGENTIC_OLLAMA_CUDA_DIR="${ollama_cuda_dir}" OLLAMA_DEBUG=1 OLLAMA_LLM_LIBRARY=cuda_v13 OLLAMA_LOAD_TIMEOUT=30m \
        "${repo_dir}/leonardo/container_exec.sh" \
        sh -c "LD_LIBRARY_PATH=${compat_dir}:\${LD_LIBRARY_PATH:-} exec ollama serve" >"${ollama_log}" 2>&1 &
    ollama_pid=$!
    "${repo_dir}/leonardo/container_exec.sh" python3 leonardo/model_cache.py wait "${model}" --timeout 180
    for _ in $(seq 60); do
        grep -q "inference compute" "${ollama_log}" && break
        sleep 1
    done
    "${repo_dir}/leonardo/container_exec.sh" nvidia-smi -L >&2 || true
    grep "inference compute" "${ollama_log}" >&2 || true
    if ! grep -qi "inference compute.*library=cuda" "${ollama_log}"; then
        "${repo_dir}/leonardo/container_exec.sh" nvidia-smi --query-gpu=driver_version --format=csv,noheader >&2 || true
        grep -E "failure during|timed out|failed|too old|cuInit|discovery took|skipping" "${ollama_log}" | cut -c 1-400 | head -n 20 >&2 || true
        echo "Ollama sees no GPU on ${SLURMD_NODENAME:-unknown}; stopping instead of running on CPU." >&2
        exit 3
    fi
fi

experiment_args=()
while IFS= read -r argument || [[ -n "${argument}" ]]; do
    [[ -z "${argument}" || "${argument}" == \#* ]] && continue
    experiment_args+=("${argument}")
done <"${task_repo}/leonardo/experiment.args"

for (( wave = first_wave; wave <= last_wave; wave++ )); do
    if wave_done "${wave}"; then
        echo "Wave ${wave} already complete; skipping."
        continue
    fi
    seed="$((seed_start + (wave - 1) * seed_step))"
    wave_dir="${campaign_root}/wave_$(printf '%02d' "${wave}")"
    mkdir -p "${wave_dir}"
    "${repo_dir}/leonardo/container_exec.sh" nvidia-smi -L >"${wave_dir}/gpus.txt"
    "${repo_dir}/leonardo/container_exec.sh" python3 leonardo/model_cache.py status "${model}" >"${wave_dir}/model_status_before.json"

    "${repo_dir}/leonardo/container_exec.sh" \
        python3 -u -m agentic_quant.run_benchmark \
        --model "${model}" \
        --seed "${seed}" \
        --out "${wave_dir}" \
        "${experiment_args[@]}" &
    experiment_pid=$!
    wait "${experiment_pid}"
    experiment_pid=""

    python3 - "${wave_dir}/.complete.json" "${model}" "${seed}" "${wave}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "model": sys.argv[2],
    "seed": int(sys.argv[3]),
    "wave": int(sys.argv[4]),
    "slurm_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
    "node": os.environ.get("SLURMD_NODENAME", ""),
}
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY

    echo "WAVE COMPLETE: model=${model} wave=${wave} seed=${seed} node=${SLURMD_NODENAME:-unknown}"
done
