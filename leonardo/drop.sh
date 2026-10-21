#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
    echo "Usage: ./leonardo/drop.sh <model>..."
    exit 1
fi
root="${WORK:?WORK is not defined}/Agentic-Environment/campaigns"
for model in "$@"; do
    slug="$(printf '%s' "${model}" | tr -c 'A-Za-z0-9._-' '_')"
    for campaign in families rejudged; do
        dir="${root}/${campaign}/${slug}"
        [[ -d "${dir}" ]] || continue
        if [[ -s "${dir}/jobs.txt" ]]; then
            scancel $(cat "${dir}/jobs.txt") 2>/dev/null || true
        fi
        rm -rf "${dir}"
        echo "${model}: jobs cancelled and results deleted in ${campaign}"
    done
done
