#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 <CINECA-account> [max-parallel-waves] [models-file]" >&2
    exit 2
fi

account="$1"
max_parallel="${2:-10}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_file="${3:-${repo_dir}/leonardo/models.txt}"
previous_cleanup=""
submitted=0

while IFS= read -r model || [[ -n "${model}" ]]; do
    model="${model%%#*}"
    model="${model#"${model%%[![:space:]]*}"}"
    model="${model%"${model##*[![:space:]]}"}"
    [[ -z "${model}" ]] && continue
    output="$("${repo_dir}/leonardo/submit_model_campaign.sh" \
        "${account}" "${model}" "${max_parallel}" "${previous_cleanup}")"
    printf '%s\n' "${output}"
    previous_cleanup="$(printf '%s\n' "${output}" | awk -F= '$1 == "cleanup_job" {print $2}')"
    if [[ -z "${previous_cleanup}" ]]; then
        echo "Could not read cleanup job id for ${model}." >&2
        exit 1
    fi
    submitted=$((submitted + 1))
done <"${models_file}"

if (( submitted == 0 )); then
    echo "No model keys found in ${models_file}." >&2
    exit 2
fi
echo "Submitted ${submitted} model campaign(s) sequentially; final cleanup job=${previous_cleanup}."
