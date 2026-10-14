#!/usr/bin/env bash
set -euo pipefail

MODELS=(llama3.3:70b qwen3:32b mistral-small3.2:24b qwen3:14b qwen3:8b llama3.1:8b)
WAVES=10
PER_JOB=5

main() {
    local repo_dir
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "${repo_dir}"
    if [[ "${AGENTIC_NIGHT_UPDATED:-0}" != "1" ]]; then
        git fetch -q origin
        git reset -q --hard origin/main
        AGENTIC_NIGHT_UPDATED=1 exec "${repo_dir}/leonardo/night.sh" "$@"
    fi
    echo "Code: $(git log -1 --format='%h %s')"
    source leonardo/models.sh
    local account="${AGENTIC_ACCOUNT:-$(basename "${WORK:?WORK is not defined}")}"
    local campaigns="${WORK}/Agentic-Environment/campaigns"
    local user="${USER:-$(id -un)}"
    local root="${campaigns}/night"

    for job_name in aq-model aq-wave aq-clean; do
        scancel -u "${user}" -n "${job_name}" 2>/dev/null || true
    done
    local cache="${AGENTIC_CACHE_HOME:-${FAST:-${WORK}}/Agentic-Environment/model-cache}"
    rm -rf "${campaigns}" "${cache}/models" "${cache}/ollama" "${cache}/locks"
    rm -f leonardo/logs/*.out leonardo/logs/*.err
    echo "Cleaned old jobs, results, logs and downloaded models."
    ./leonardo/prepare_cuda.sh

    unset AGENTIC_RESULTS_ROOT AGENTIC_SEED_START AGENTIC_SEED_STEP
    export AGENTIC_CAMPAIGN_NAME=night AGENTIC_WAVES="${WAVES}" AGENTIC_WAVES_PER_JOB="${PER_JOB}"
    mkdir -p "${root}"
    local cleanup_jobs=() model slug resources submitted jobs
    for model in "${MODELS[@]}"; do
        slug="$(printf '%s' "${model}" | tr -c 'A-Za-z0-9._-' '_')"
        resources="$(model_resources "${model}")"
        if [[ -n "${resources}" ]]; then
            export AGENTIC_WAVE_SBATCH_ARGS="--time=12:00:00 ${resources}"
        else
            export AGENTIC_WAVE_SBATCH_ARGS="--time=08:00:00"
        fi
        submitted="$(./leonardo/submit_model_campaign.sh "${account}" "${model}" 2)"
        jobs="$(printf '%s\n' "${submitted}" | sed -n 's/^\(prepare\|wave\|cleanup\)_job=//p' | tr '\n' ' ')"
        echo "${model}: jobs ${jobs}"
        mkdir -p "${root}/${slug}"
        echo ${jobs} >"${root}/${slug}/jobs.txt"
        cleanup_jobs+=("$(printf '%s\n' "${submitted}" | sed -n 's/^cleanup_job=//p')")
    done

    echo "Waiting for all models. You can close this window and later run ./leonardo/report.sh"
    local ids
    ids="$(IFS=,; echo "${cleanup_jobs[*]}")"
    while squeue -h -j "${ids}" 2>/dev/null | grep -q .; do
        echo "$(date +%H:%M) $(squeue -h -u "${user}" -o '%j=%T' | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"
        sleep 300
    done
    ./leonardo/report.sh night
}

main "$@"
