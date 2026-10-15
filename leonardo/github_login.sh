#!/usr/bin/env bash
set -euo pipefail

version=2.62.0
tools="${WORK:?WORK is not defined}/Agentic-Environment/tools"
if [[ ! -x "${tools}/bin/gh" ]]; then
    mkdir -p "${tools}"
    curl -fsSL "https://github.com/cli/cli/releases/download/v${version}/gh_${version}_linux_amd64.tar.gz" | tar -xz -C "${tools}" --strip-components=1
fi
export PATH="${tools}/bin:${PATH}"
echo "Open github.com/login/device on your Mac and type the code shown below."
BROWSER=true gh auth login --hostname github.com --git-protocol https --web --scopes repo
gh auth setup-git --hostname github.com
gh auth status
