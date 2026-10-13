#!/usr/bin/env bash
set -euo pipefail

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch.sh" campaign 100 10 "--time=04:00:00" "${1:-gpt-oss:20b}"
