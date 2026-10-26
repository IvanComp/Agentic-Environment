#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
git fetch -q origin
git reset -q --hard origin/main
./leonardo/drop.sh glm-4.7
AGENTIC_FAMILIES_UPDATED=1 exec ./leonardo/families.sh glm-4.7-flash
