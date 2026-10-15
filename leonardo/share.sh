#!/usr/bin/env bash
set -euo pipefail

name="${1:-night}"
file="${WORK:?WORK is not defined}/Agentic-Environment/campaigns/${name}/report.txt"
if [[ ! -s "${file}" ]]; then
    echo "No report at ${file}. Run ./leonardo/report.sh ${name} first." >&2
    exit 1
fi
url="$(curl -sS -m 60 --data-binary "@${file}" https://paste.rs/ || true)"
if [[ "${url}" != https://* ]]; then
    url="$(curl -sS -m 60 -A curl/8 -F "file=@${file}" https://0x0.st || true)"
fi
if [[ "${url}" != https://* ]]; then
    echo "Upload failed: ${url}" >&2
    exit 1
fi
echo
echo "Open this link on your Mac:"
echo
echo "    ${url}"
echo
