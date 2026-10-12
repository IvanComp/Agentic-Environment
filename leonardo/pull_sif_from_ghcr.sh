#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sif_path="${AGENTIC_SIF_PATH:-${repo_dir}/leonardo/agentic-quant.sif}"
image_uri="${AGENTIC_IMAGE_URI:-docker://ghcr.io/ivancomp/agentic-environment-leonardo:latest}"

if command -v singularity >/dev/null 2>&1; then
    runtime=singularity
elif command -v apptainer >/dev/null 2>&1; then
    runtime=apptainer
else
    echo "Singularity or Apptainer is required." >&2
    exit 1
fi

storage="${WORK:-${repo_dir}}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${storage}/.singularity-cache}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-${storage}/.singularity-tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SINGULARITY_CACHEDIR}}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SINGULARITY_TMPDIR}}"
mkdir -p "$(dirname "${sif_path}")" "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"
echo "Downloading ${image_uri}"
"${runtime}" pull --force "${sif_path}" "${image_uri}"
"${runtime}" inspect "${sif_path}" >/dev/null
echo "Created ${sif_path}"
