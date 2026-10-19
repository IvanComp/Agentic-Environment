#!/usr/bin/env bash
set -euo pipefail

MODELS=(
    gemma4:31b granite4.1:30b nemotron-3-nano:30b ministral-3:14b
    gemma4:12b granite4.1:8b ministral-3:8b
    gemma4:e4b nemotron-3-nano:4b granite4.1:3b ministral-3:3b
)
NAME=families
WAVES=10
PER_JOB=5
LIMIT=15

cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "${AGENTIC_FAMILIES_UPDATED:-0}" != "1" ]]; then
    git fetch -q origin
    git reset -q --hard origin/main
    AGENTIC_FAMILIES_UPDATED=1 exec ./leonardo/families.sh "$@"
fi
source leonardo/models.sh
if (( $# > 0 )); then
    MODELS=("$@")
fi
account="${AGENTIC_ACCOUNT:-$(basename "${WORK:?WORK is not defined}")}"
user="${USER:-$(id -un)}"
root="${WORK}/Agentic-Environment/campaigns/${NAME}"
needed=$(( 2 + (WAVES + PER_JOB - 1) / PER_JOB ))
./leonardo/prepare_cuda.sh >/dev/null
unset AGENTIC_RESULTS_ROOT AGENTIC_SEED_START AGENTIC_SEED_STEP
export AGENTIC_CAMPAIGN_NAME="${NAME}" AGENTIC_WAVES="${WAVES}" AGENTIC_WAVES_PER_JOB="${PER_JOB}" AGENTIC_EXPERIMENT_FILE=experiment-all.args
waiting=0
for model in "${MODELS[@]}"; do
    slug="$(printf '%s' "${model}" | tr -c 'A-Za-z0-9._-' '_')"
    dir="${root}/${slug}"
    done_waves="$(find "${dir}" -mindepth 2 -maxdepth 2 -name .complete.json 2>/dev/null | wc -l)"
    if (( done_waves >= WAVES )); then
        echo "${model}: complete"
        continue
    fi
    if [[ -s "${dir}/jobs.txt" ]] && squeue -h -j "$(tr -s ' ' ',' <"${dir}/jobs.txt" | sed 's/,$//')" 2>/dev/null | grep -q .; then
        echo "${model}: running, ${done_waves}/${WAVES} waves done"
        continue
    fi
    queued="$(squeue -h -r -u "${user}" 2>/dev/null | wc -l)"
    if (( queued + needed > LIMIT )); then
        echo "${model}: waiting for free slots"
        waiting=$(( waiting + 1 ))
        continue
    fi
    export AGENTIC_WAVE_SBATCH_ARGS="--time=24:00:00 $(model_resources "${model}")"
    if ! submitted="$(./leonardo/submit_model_campaign.sh "${account}" "${model}" 2)"; then
        echo "${model}: not submitted"
        waiting=$(( waiting + 1 ))
        continue
    fi
    mkdir -p "${dir}"
    printf '%s\n' "${submitted}" | sed -n 's/^\(prepare\|wave\|cleanup\)_job=//p' | tr '\n' ' ' >"${dir}/jobs.txt"
    echo "${model}: submitted, jobs $(cat "${dir}/jobs.txt")"
done
if (( waiting > 0 )); then
    echo "${waiting} models still waiting: run ./leonardo/families.sh again later."
else
    echo "All models submitted or complete. Progress: ./leonardo/status.sh ${NAME}"
fi
