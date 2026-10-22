#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
count="${1:-8}"
for log in $(ls -t leonardo/logs/wave-*.err 2>/dev/null | head -n "${count}"); do
    job="$(basename "${log}" .err)"
    job="${job#wave-}"
    echo "== ${job}  $(date -r "${log}" '+%d/%m %H:%M')  $(sacct -X -n -j "${job%%_*}" --format=State%12,Elapsed%11 2>/dev/null | head -n 1 | tr -s ' ')"
    grep -v "gres/tmpfs" "${log}" | tail -n 6 | cut -c 1-180 | sed 's/^/    /'
done
