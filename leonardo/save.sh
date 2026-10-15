#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"
if [[ "${AGENTIC_SAVE_UPDATED:-0}" != "1" ]]; then
    git fetch -q origin
    git reset -q --hard origin/main
    AGENTIC_SAVE_UPDATED=1 exec "${repo_dir}/leonardo/save.sh" "$@"
fi
campaigns="${WORK:?WORK is not defined}/Agentic-Environment/campaigns"
tools="${WORK}/Agentic-Environment/tools"
export PATH="${tools}/bin:${PATH}"
export AGENTIC_SOURCE_DIR="${repo_dir}"
export AGENTIC_CACHE_HOME="${AGENTIC_CACHE_HOME:-${FAST:-${WORK}}/Agentic-Environment/model-cache}"
export AGENTIC_OUTPUT_ROOT="${campaigns}"
"${repo_dir}/leonardo/container_exec.sh" python3 -m agentic_quant.archive "${campaigns}" --out results/data
"${repo_dir}/leonardo/container_exec.sh" python3 -m agentic_quant.paper_results results/data --out results
git add results
if git diff --cached --quiet; then
    echo "Nothing new to save."
    exit 0
fi
git -c user.name=IvanComp -c user.email=IvanComp@users.noreply.github.com commit -q -m "Save results $(date +%Y-%m-%d_%H%M)"
for attempt in 1 2 3 4; do
    if git push -q origin HEAD:main; then
        echo "Saved on GitHub: $(git log -1 --format=%h) $(git diff --stat HEAD~1 | tail -n 1)"
        exit 0
    fi
    sleep $(( attempt * 5 ))
    git pull -q --rebase origin main
done
echo "Push failed. Run ./leonardo/github_login.sh once, then ./leonardo/save.sh again."
exit 1
