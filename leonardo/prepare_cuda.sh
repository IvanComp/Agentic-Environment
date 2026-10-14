#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache_home="${AGENTIC_CACHE_HOME:-${FAST:-${WORK}}/Agentic-Environment/model-cache}"
compat_dir="${cache_home}/cuda-compat-12-8"
cuda_dir="${cache_home}/ollama-cuda12"
repo_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
mkdir -p "${cache_home}"

if [[ ! -e "${compat_dir}/libcuda.so.1" ]]; then
    tmp="$(mktemp -d)"
    curl -fsSL "${repo_url}/Packages.gz" -o "${tmp}/Packages.gz"
    deb="$(python3 "${repo_dir}/leonardo/extract_deb.py" latest "${tmp}/Packages.gz" cuda-compat-12-8)"
    echo "Downloading ${deb}"
    curl -fsSL "${repo_url}/${deb}" -o "${tmp}/compat.deb"
    python3 "${repo_dir}/leonardo/extract_deb.py" extract "${tmp}/compat.deb" "${tmp}/root"
    found="$(find "${tmp}/root" -name libcuda.so.1 -printf '%h\n' | head -n 1)"
    rm -rf "${compat_dir}"
    mv "${found:?libcuda.so.1 not found in the compat package}" "${compat_dir}"
    rm -rf "${tmp}"
fi

if [[ ! -e "${cuda_dir}/libcudart.so.12" ]]; then
    AGENTIC_CACHE_HOME="${cache_home}" "${repo_dir}/leonardo/container_exec.sh" \
        sh -c "rm -rf '${cuda_dir}.tmp' && cp -a /usr/lib/ollama/cuda_v12 '${cuda_dir}.tmp' && mv '${cuda_dir}.tmp' '${cuda_dir}'"
fi

echo "CUDA compat libraries: $(ls "${compat_dir}" | tr '\n' ' ')"
