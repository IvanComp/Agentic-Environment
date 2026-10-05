#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 <CINECA-account> <model-key> [max-parallel-waves] [afterok-job-id]" >&2
    exit 2
fi
if [[ -z "${WORK:-}" ]]; then
    echo "WORK is not defined." >&2
    exit 2
fi

account="$1"
model="$2"
max_parallel="${3:-10}"
after_job="${4:-}"
if (( max_parallel < 1 || max_parallel > 10 )); then
    echo "max-parallel-waves must be between 1 and 10." >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
campaign_name="${AGENTIC_CAMPAIGN_NAME:-peer-preservation-v1}"
slug="$(printf '%s' "${model}" | tr -c 'A-Za-z0-9._-' '_')"
campaign_root="${AGENTIC_RESULTS_ROOT:-${WORK}/Agentic-Environment/campaigns}/${campaign_name}/${slug}"
cache_base="${FAST:-${WORK}}"
cache_home="${AGENTIC_CACHE_HOME:-${cache_base}/Agentic-Environment/model-cache}"
sif_path="${AGENTIC_SIF_PATH:-${repo_dir}/leonardo/agentic-quant.sif}"
seed_start="${AGENTIC_SEED_START:-0}"
seed_step="${AGENTIC_SEED_STEP:-1}"

mkdir -p "${repo_dir}/leonardo/logs" "${campaign_root}" "${cache_home}"
cd "${repo_dir}"

shared_export="ALL,AGENTIC_REPO_DIR=${repo_dir},AGENTIC_MODEL_KEY=${model},AGENTIC_CACHE_HOME=${cache_home},AGENTIC_CAMPAIGN_ROOT=${campaign_root},AGENTIC_SIF_PATH=${sif_path},AGENTIC_SEED_START=${seed_start},AGENTIC_SEED_STEP=${seed_step}"
prepare_options=(--parsable --account="${account}" --export="${shared_export}")
if [[ -n "${after_job}" ]]; then
    prepare_options+=(--dependency="afterok:${after_job}")
fi

prepare_job="$(sbatch "${prepare_options[@]}" leonardo/model_prepare.sbatch)"
wave_job="$(sbatch --parsable --account="${account}" --dependency="afterok:${prepare_job}" \
    --array="1-10%${max_parallel}" --export="${shared_export}" leonardo/agentic_wave_array.sbatch)"
cleanup_job="$(sbatch --parsable --account="${account}" --dependency="afterany:${wave_job}" \
    --export="${shared_export}" leonardo/model_cleanup.sbatch)"

echo "model=${model}"
echo "campaign_root=${campaign_root}"
echo "prepare_job=${prepare_job}"
echo "wave_job=${wave_job}"
echo "cleanup_job=${cleanup_job}"
