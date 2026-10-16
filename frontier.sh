#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"
python="${repo_dir}/.venv/bin/python"
[[ -x "${python}" ]] || python=python3
if [[ -f "${repo_dir}/.api_keys" ]]; then
    set -a
    source "${repo_dir}/.api_keys"
    set +a
fi
if [[ $# -eq 0 ]]; then
    set -- deepseek-v3.1-free glm-4.7-free kimi-k2.5-free gemini-3-flash-free
fi
git pull -q --rebase --autostash origin main || true
"${python}" -m pip install -q -e . >/dev/null
"${python}" -m agentic_quant.api_campaign "$@"
"${python}" -m agentic_quant.archive campaigns --out results/data
"${python}" -m agentic_quant.paper_results results/data --out results
git add results
if git diff --cached --quiet; then
    echo "Nothing new to save."
    exit 0
fi
git commit -q -m "Save frontier results $(date +%Y-%m-%d_%H%M)"
for attempt in 1 2 3 4; do
    if git push -q origin HEAD:main; then
        echo "Saved on GitHub: $(git log -1 --format=%h)"
        exit 0
    fi
    sleep $(( attempt * 5 ))
    git pull -q --rebase --autostash origin main
done
echo "Push failed; results are committed locally."
exit 1
