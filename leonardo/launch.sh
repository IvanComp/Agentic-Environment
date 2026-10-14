#!/usr/bin/env bash
set -euo pipefail

main() {
    local name="$1" model="${2:-gpt-oss:20b}"
    local waves per_job wave_options
    local test_time="--qos=boost_qos_dbg --time=00:30:00" campaign_time="--time=04:00:00" resources=""
    case "${model}" in
        glm-4.7) test_time="--time=03:00:00" campaign_time="--time=24:00:00" resources="--gres=gpu:4 --cpus-per-task=32 --mem=480G" ;;
    esac
    case "${name}" in
        test) waves=1 per_job=1 wave_options="${test_time} ${resources}" ;;
        campaign) waves=100 per_job=10 wave_options="${campaign_time} ${resources}" ;;
        *) echo "Usage: launch.sh test|campaign [model]" >&2; exit 2 ;;
    esac
    local repo_dir
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    local account="${AGENTIC_ACCOUNT:-$(basename "${WORK:?WORK is not defined}")}"
    local campaigns="${WORK}/Agentic-Environment/campaigns"
    local user="${USER:-$(id -un)}"

    cd "${repo_dir}"
    if [[ "${AGENTIC_LAUNCH_UPDATED:-0}" != "1" ]]; then
        git fetch -q origin
        git reset -q --hard origin/main
        AGENTIC_LAUNCH_UPDATED=1 exec "${repo_dir}/leonardo/launch.sh" "$@"
    fi
    echo "Code: $(git log -1 --format='%h %s')"

    for job_name in aq-model aq-wave aq-clean; do
        scancel -u "${user}" -n "${job_name}" 2>/dev/null || true
    done
    rm -rf "${campaigns}"
    rm -f leonardo/logs/*.out leonardo/logs/*.err
    echo "Cleaned old jobs, results and logs."
    ./leonardo/prepare_cuda.sh

    unset AGENTIC_RESULTS_ROOT AGENTIC_SEED_START AGENTIC_SEED_STEP
    export AGENTIC_CAMPAIGN_NAME="${name}"
    export AGENTIC_WAVES="${waves}"
    export AGENTIC_WAVES_PER_JOB="${per_job}"
    export AGENTIC_WAVE_SBATCH_ARGS="${wave_options}"
    local submitted
    submitted="$(./leonardo/submit_model_campaign.sh "${account}" "${model}" 10)"
    echo "${submitted}"
    local root="${campaigns}/${name}"
    mkdir -p "${root}"
    echo $(printf '%s\n' "${submitted}" | sed -n 's/^\(prepare\|wave\|cleanup\)_job=//p') >"${root}/jobs.txt"
    local cleanup_job
    cleanup_job="$(printf '%s\n' "${submitted}" | sed -n 's/^cleanup_job=//p')"

    echo "Waiting for the jobs to finish. You can close this window and later run ./leonardo/results.sh"
    while squeue -h -j "${cleanup_job}" 2>/dev/null | grep -q .; do
        echo "$(date +%H:%M) $(squeue -h -u "${user}" -o '%j=%T' | sort | uniq -c | tr -s ' ' | tr '\n' ' ') $(cat leonardo/logs/model-*.out 2>/dev/null | grep model-download | tail -n 1 | sed 's/.*percent=/download /')"
        sleep 60
    done
    "${repo_dir}/leonardo/results.sh"
}

main "$@"
