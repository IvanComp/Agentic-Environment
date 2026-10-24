#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
AGENTIC_FAMILIES_PER_JOB=2 exec ./leonardo/families.sh glm-4.7
