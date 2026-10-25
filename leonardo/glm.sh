#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
git fetch -q origin
git reset -q --hard origin/main
dir="${WORK:?WORK is not defined}/Agentic-Environment/campaigns/families/glm-4.7"
if [[ -s "${dir}/jobs.txt" ]]; then
    ids="$(tr -s ' ' ',' <"${dir}/jobs.txt" | sed 's/,$//')"
    if ! squeue -h -j "${ids}" -t RUNNING 2>/dev/null | grep -q .; then
        scancel $(cat "${dir}/jobs.txt") 2>/dev/null || true
        rm -f "${dir}/jobs.txt"
        echo "glm-4.7: old queued jobs cancelled, resubmitting as shorter jobs"
    fi
fi
AGENTIC_FAMILIES_UPDATED=1 AGENTIC_FAMILIES_PER_JOB=1 AGENTIC_FAMILIES_TIME=08:00:00 exec ./leonardo/families.sh glm-4.7
