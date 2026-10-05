#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_tag="agentic-quant-leonardo:latest"
sif_path="${1:-${repo_dir}/leonardo/agentic-quant.sif}"
archive_path="$(mktemp "${TMPDIR:-/tmp}/agentic-quant-image.XXXXXX.tar")"

cleanup() {
    rm -f "${archive_path}"
}
trap cleanup EXIT

if command -v singularity >/dev/null 2>&1; then
    runtime=singularity
elif command -v apptainer >/dev/null 2>&1; then
    runtime=apptainer
else
    echo "Install Singularity or Apptainer before building the SIF." >&2
    exit 1
fi

docker build -t "${image_tag}" -f "${repo_dir}/leonardo/Dockerfile" "${repo_dir}"
docker save -o "${archive_path}" "${image_tag}"
"${runtime}" build --force "${sif_path}" "docker-archive://${archive_path}"
echo "Created ${sif_path}"
