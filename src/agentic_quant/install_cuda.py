from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys


CUDA_WHEELS = (
    ((11, 8), "cu118"),
    ((12, 1), "cu121"),
    ((12, 2), "cu122"),
    ((12, 3), "cu123"),
    ((12, 4), "cu124"),
    ((12, 5), "cu125"),
    ((13, 0), "cu130"),
    ((13, 2), "cu132"),
)


def main() -> None:
    if platform.system() != "Linux":
        raise SystemExit("The CUDA installer is intended for Linux.")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit(
            "The prebuilt CUDA wheels require Linux x86_64. "
            "Use requirements-cuda.txt to compile from source on this architecture."
        )
    if sys.version_info[:2] not in {(3, 10), (3, 11), (3, 12)}:
        raise SystemExit(
            "Prebuilt CUDA wheels require Python 3.10, 3.11, or 3.12. "
            "Create the virtual environment with one of these Python versions."
        )

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise SystemExit("nvidia-smi was not found. Install or repair the NVIDIA driver first.")
    probe = subprocess.run(
        [nvidia_smi],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip()
        raise SystemExit(f"nvidia-smi failed: {detail or 'unknown driver error'}")

    cuda_version = parse_driver_cuda_version(probe.stdout)
    wheel_tag = select_cuda_wheel(cuda_version)
    runtime_packages = cuda_runtime_packages(wheel_tag)
    index_url = f"https://abetlen.github.io/llama-cpp-python/whl/{wheel_tag}"
    print(
        f"NVIDIA driver supports CUDA {cuda_version[0]}.{cuda_version[1]}; "
        f"installing the {wheel_tag} llama-cpp-python wheel."
    )
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--no-cache-dir",
        "--prefer-binary",
        "--only-binary=llama-cpp-python",
        "llama-cpp-python>=0.2.90",
        *runtime_packages,
        "--extra-index-url",
        index_url,
    ]
    subprocess.run(command, check=True)
    verify_cuda_backend()
    print("CUDA backend verified. Restart agentic-quant-ui and run a real model.")


def parse_driver_cuda_version(output: str) -> tuple[int, int]:
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output, flags=re.I)
    if not match:
        raise SystemExit("Could not read the supported CUDA version from nvidia-smi.")
    return int(match.group(1)), int(match.group(2))


def select_cuda_wheel(cuda_version: tuple[int, int]) -> str:
    compatible = [tag for version, tag in CUDA_WHEELS if version <= cuda_version]
    if not compatible:
        minimum = CUDA_WHEELS[0][0]
        raise SystemExit(
            f"The NVIDIA driver reports CUDA {cuda_version[0]}.{cuda_version[1]}, "
            f"but the oldest available wheel requires CUDA {minimum[0]}.{minimum[1]}."
        )
    return compatible[-1]


def cuda_runtime_packages(wheel_tag: str) -> list[str]:
    match = re.fullmatch(r"cu(11|12|13)\d", wheel_tag)
    if not match:
        raise ValueError(f"Unsupported CUDA wheel tag: {wheel_tag}")
    major = match.group(1)
    packages = [
        f"nvidia-cuda-runtime-cu{major}",
        f"nvidia-cublas-cu{major}",
    ]
    if major != "11":
        packages.append(f"nvidia-nvjitlink-cu{major}")
    return packages


def verify_cuda_backend() -> None:
    code = (
        "from agentic_quant.providers import _preload_cuda_libraries; "
        "_preload_cuda_libraries(); "
        "import llama_cpp; "
        "supported = bool(llama_cpp.llama_supports_gpu_offload()); "
        "info = llama_cpp.llama_print_system_info(); "
        "text = info.decode('utf-8', errors='replace') if isinstance(info, bytes) else str(info); "
        "assert supported and 'CUDA' in text.upper(), "
        "'Installed package does not expose the CUDA backend'; "
        "print(text)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    main()
