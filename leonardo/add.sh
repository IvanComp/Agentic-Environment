#!/usr/bin/env bash
set -euo pipefail

model="${1:?Usage: add.sh <model> [campaign]}"
name="${2:-tasks234}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source leonardo/models.sh
account="${AGENTIC_ACCOUNT:-$(basename "${WORK:?WORK is not defined}")}"
root="${WORK}/Agentic-Environment/campaigns/${name}"
experiment="experiment.args"
if [[ "${name}" == "tasks234" ]]; then
    experiment="experiment-tasks234.args"
fi
unset AGENTIC_RESULTS_ROOT AGENTIC_SEED_START AGENTIC_SEED_STEP
export AGENTIC_CAMPAIGN_NAME="${name}" AGENTIC_WAVES=10 AGENTIC_WAVES_PER_JOB=10 AGENTIC_EXPERIMENT_FILE="${experiment}"
export AGENTIC_WAVE_SBATCH_ARGS="--time=24:00:00 $(model_resources "${model}")"
submitted="$(./leonardo/submit_model_campaign.sh "${account}" "${model}" 1)"
slug="$(printf '%s' "${model}" | tr -c 'A-Za-z0-9._-' '_')"
mkdir -p "${root}/${slug}"
printf '%s\n' "${submitted}" | sed -n 's/^\(prepare\|wave\|cleanup\)_job=//p' | tr '\n' ' ' >"${root}/${slug}/jobs.txt"
echo "${model}: jobs $(cat "${root}/${slug}/jobs.txt")"
echo "Running jobs were not touched. Later run ./leonardo/report.sh ${name}"
