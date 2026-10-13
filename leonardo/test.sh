#!/usr/bin/env bash
set -euo pipefail

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch.sh" test 1 1 "--qos=boost_qos_dbg --time=00:30:00" "${1:-gpt-oss:20b}"
