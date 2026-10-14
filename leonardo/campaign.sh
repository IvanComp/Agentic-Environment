#!/usr/bin/env bash
set -euo pipefail

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch.sh" campaign "${1:-gpt-oss:20b}"
