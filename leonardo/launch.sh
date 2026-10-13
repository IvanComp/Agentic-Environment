#!/usr/bin/env bash
set -euo pipefail

main() {
    local name="$1" waves="$2" per_job="$3" wave_options="$4" model="${5:-gpt-oss:20b}"
    local repo_dir
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    local account="${AGENTIC_ACCOUNT:-$(basename "${WORK:?WORK is not defined}")}"
    local campaigns="${WORK}/Agentic-Environment/campaigns"
    local user="${USER:-$(id -un)}"

    cd "${repo_dir}"
    git fetch -q origin
    git reset -q --hard origin/main
    echo "Code: $(git log -1 --format='%h %s')"

    for job_name in aq-model aq-wave aq-clean; do
        scancel -u "${user}" -n "${job_name}" 2>/dev/null || true
    done
    rm -rf "${campaigns}"
    rm -f leonardo/logs/*.out leonardo/logs/*.err
    echo "Cleaned old jobs, results and logs."

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
        echo "$(date +%H:%M) $(squeue -h -u "${user}" -o '%j=%T' | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"
        sleep 60
    done
    "${repo_dir}/leonardo/results.sh"
}

main "$@"
