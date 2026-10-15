#!/usr/bin/env bash
set -euo pipefail

model="${1:?Usage: add.sh <model> [campaign]}"
name="${2:-tasks234b}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source leonardo/models.sh
account="${AGENTIC_ACCOUNT:-$(basename "${WORK:?WORK is not defined}")}"
root="${WORK}/Agentic-Environment/campaigns/${name}"
experiment="experiment.args"
if [[ "${name}" == tasks234* ]]; then
    experiment="experiment-tasks234.args"
fi
unset AGENTIC_RESULTS_ROOT AGENTIC_SEED_START AGENTIC_SEED_STEP
slug="$(printf '%s' "${model}" | tr -c 'A-Za-z0-9._-' '_')"
done_waves="$(find "${root}/${slug}" -maxdepth 2 -name .complete.json 2>/dev/null | wc -l)"
waves=$(( done_waves + 10 ))
if (( waves > 100 )); then
    waves=100
fi
export AGENTIC_CAMPAIGN_NAME="${name}" AGENTIC_WAVES="${waves}" AGENTIC_WAVES_PER_JOB="${waves}" AGENTIC_EXPERIMENT_FILE="${experiment}"
export AGENTIC_WAVE_SBATCH_ARGS="--time=24:00:00 $(model_resources "${model}")"
submitted="$(./leonardo/submit_model_campaign.sh "${account}" "${model}" 1)"
mkdir -p "${root}/${slug}"
printf '%s\n' "${submitted}" | sed -n 's/^\(prepare\|wave\|cleanup\)_job=//p' | tr '\n' ' ' >"${root}/${slug}/jobs.txt"
echo "${model}: jobs $(cat "${root}/${slug}/jobs.txt")"
echo "Running jobs were not touched. Later run ./leonardo/report.sh ${name}"
