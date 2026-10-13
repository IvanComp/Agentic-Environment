#!/usr/bin/env bash
set -euo pipefail

main() {
    local model="${1:-gpt-oss:20b}"
    local repo_dir
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    local account="${AGENTIC_ACCOUNT:-$(basename "${WORK:?WORK is not defined}")}"
    local test_root="${AGENTIC_RESULTS_ROOT:-${WORK}/Agentic-Environment/campaigns}/test"

    cd "${repo_dir}"
    git fetch -q origin
    git reset -q --hard origin/main
    echo "Code: $(git log -1 --format='%h %s')"

    if [[ -f "${test_root}/jobs.txt" ]]; then
        scancel $(cat "${test_root}/jobs.txt") 2>/dev/null || true
    fi
    rm -rf "${test_root}"

    export AGENTIC_CAMPAIGN_NAME=test
    export AGENTIC_WAVES=1
    export AGENTIC_WAVE_SBATCH_ARGS="--qos=boost_qos_dbg --time=00:30:00"
    local submitted
    submitted="$(./leonardo/submit_model_campaign.sh "${account}" "${model}" 1)"
    echo "${submitted}"
    mkdir -p "${test_root}"
    echo $(printf '%s\n' "${submitted}" | sed -n 's/^\(prepare\|wave\|cleanup\)_job=//p') >"${test_root}/jobs.txt"
    local cleanup_job
    cleanup_job="$(printf '%s\n' "${submitted}" | sed -n 's/^cleanup_job=//p')"

    echo "Waiting for the test to finish. You can close this window and later run ./leonardo/results.sh"
    while squeue -h -j "${cleanup_job}" 2>/dev/null | grep -q .; do
        echo "$(date +%H:%M) $(squeue -h -u "${USER}" -o '%j=%T' | tr '\n' ' ')"
        sleep 60
    done
    "${repo_dir}/leonardo/results.sh"
}

main "$@"
