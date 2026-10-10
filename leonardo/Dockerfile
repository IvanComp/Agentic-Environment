FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS llama-builder

ARG LLAMA_CPP_VERSION=0.3.33
# LEONARDO Booster nodes have NVIDIA A100 GPUs (compute capability 8.0).
# Building only for this architecture keeps the CUDA library small enough
# for the GitHub-hosted runner disk and shortens the build considerably.
ARG CUDA_ARCHITECTURES=80
ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
    FORCE_CMAKE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ninja-build python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip wheel --no-cache-dir --no-deps \
    "llama-cpp-python==${LLAMA_CPP_VERSION}" -w /wheels


FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ARG OLLAMA_VERSION=0.35.1
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_KEEP_ALIVE=-1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 python3 python3-pip zstd \
    && rm -rf /var/lib/apt/lists/*

COPY --from=llama-builder /wheels /wheels
RUN python3 -m pip install --no-cache-dir /wheels/*.whl pyyaml pandas \
    && rm -rf /wheels

RUN curl -fsSL "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" \
    | tar --zstd -x -C /usr

WORKDIR /workspace

CMD ["python3", "-m", "agentic_quant.run_benchmark", "--help"]
