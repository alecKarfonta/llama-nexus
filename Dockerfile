# Multi-stage build for llama.cpp with CUDA support
# [2026-10-15] | INFO | DOCKERFILE.RUNTIME_BASE | global ARG so both FROM lines can use
# it; cpu variant swaps the CUDA runtime base for plain ubuntu:22.04 via build-arg
ARG RUNTIME_BASE=nvidia/cuda:13.0.0-runtime-ubuntu22.04
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    libcurl4-openssl-dev \
    pciutils \
    python3 \
    python3-pip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install huggingface_hub for model downloads
RUN pip3 install huggingface_hub hf_transfer

# Clone and build llama.cpp with CUDA support
# Pin a revision so Docker cache does not keep an old llama.cpp without newer arch support.
# [TIME] | INFO | DOCKERFILE.ARG_PINS | 2026-10-15: default b10723 — qwen4exp (Qwen3.8-Flash-Next,
# [TIME] | INFO | DOCKERFILE.ARG_PINS | PR #27742) is merged upstream, so the official tag suffices.
# MTP_PATCH: optional git-apply patch layered on top (qwen4exp draft-head graph + converter
# --mtp export) for MTP-converted GGUFs; verified against unslothai/llama.cpp@bea3b12da.
# MTP flavor build (patch applies to the fork snapshot, not to merged b10723):
#   docker compose build --build-arg LLAMA_CPP_REF=bea3b12daee45876b0129a3602dc8f534ce30bf0 \
#     --build-arg LLAMA_CPP_REMOTE=https://github.com/unslothai/llama.cpp.git \
#     --build-arg MTP_PATCH=patches/qwen4exp-mtp-draft-head.patch --build-arg LLAMACPP_BUILD_TAG=b10631-qwen4exp-mtp
# Override at build time: docker compose build --build-arg LLAMA_CPP_REF=b10723 llamacpp-api
ARG LLAMA_CPP_REF=b10723
ARG LLAMA_CPP_REMOTE=https://github.com/ggml-org/llama.cpp.git
ARG MTP_PATCH=
ARG LLAMACPP_BUILD_TAG=b10723
WORKDIR /build
ARG SKIP_BUILD_FROM_SOURCE=false
ARG LLAMACPP_VERSION=b10723
ARG ENABLE_TURBOQUANT=false
# Opt-in: requires Rust in builder; enables jump-forward grammar tier in llama-server
ARG ENABLE_LLGUIDANCE=false
# CUDA/nvcc compiles are memory-heavy — keep low to avoid OOM (override: CMAKE_BUILD_JOBS=4)
ARG CMAKE_BUILD_JOBS=2
RUN if [ "$ENABLE_LLGUIDANCE" = "true" ]; then \
        apt-get update && apt-get install -y rustc cargo && rm -rf /var/lib/apt/lists/*; \
    fi
# CUDA architecture(s) passed to CMake. Run on the host: nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# (use the major.minor as an integer SM, e.g. 8.6 -> 86). Common values: 75 (T4), 80 (A100), 86 (RTX 30xx),
# 89 (RTX 40xx), 90 (H100), 120 (RTX 50xx / Blackwell). Default 120 suits 4x 5070 Ti / RTX 5090.
# IMPORTANT: Do not use "native" here when building Docker images — the builder usually has no GPU, so CMake
# cannot detect SM and nvcc fails with: Unsupported gpu architecture 'compute_' .
ARG CUDA_ARCH=120

# [2026-10-15] | INFO | DOCKERFILE.RUNTIME_LIBS_DIR | always exists so the runtime
# COPY --from=builder works for both prebuilt (libs populated) and source-built (empty)
RUN mkdir -p /build/runtime-libs

# [TIME] | INFO | DOCKERFILE.PATCH_CONTEXT | the MTP patch lives in the build context
# (./patches/); COPY it into the builder so the build RUN can git-apply it.
COPY ${MTP_PATCH} /build-context/${MTP_PATCH}

# [2026-10-15] | FIX | DOCKERFILE.SKIP_BUILD | upstream ships .tar.gz (not .zip) and the
# prebuilt binaries are dynamically linked (RUNPATH=$ORIGIN) with bundled .so files that
# must sit beside them at runtime. curl now uses -f so a 404 asset fails the build; the
# old `A && B || fallback` precedence silently swallowed download failures.
# [2026-10-16] | FIX | DOCKERFILE.TARGETARCH | bare ARG (no default): buildx
# auto-populates TARGETARCH per build platform; a declared default would pin
# every platform to amd64 (bit me in the arm64 cross-build)
ARG TARGETARCH
RUN if [ "$SKIP_BUILD_FROM_SOURCE" = "true" ]; then \
        case "${TARGETARCH}" in \
            arm64) ASSET_ARCH=arm64 ;; \
            *) ASSET_ARCH=x64 ;; \
        esac && \
        echo "Downloading pre-built binaries: ${LLAMACPP_VERSION} ubuntu-${ASSET_ARCH}" && \
        mkdir -p /build/llama.cpp/build/bin /build/runtime-libs && \
        curl -fL -o llama-bin.tar.gz "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMACPP_VERSION}/llama-${LLAMACPP_VERSION}-bin-ubuntu-${ASSET_ARCH}.tar.gz" && \
        mkdir extracted_llama && \
        tar -xzf llama-bin.tar.gz -C extracted_llama --strip-components=1 && \
        cp extracted_llama/llama-server extracted_llama/llama-cli extracted_llama/llama-gguf-split /build/llama.cpp/build/bin/ && \
        cp extracted_llama/*.so* /build/runtime-libs/ && \
        echo "${LLAMACPP_BUILD_TAG}" > /build/llamacpp-build-tag.txt && \
        echo "false" > /build/llamacpp-llguidance-enabled.txt && \
        ( /build/llama.cpp/build/bin/llama-cli --version 2>/dev/null | head -1 > /build/llama-cli-version.txt; \
          [ -s /build/llama-cli-version.txt ] || echo "llama.cpp ${LLAMACPP_VERSION}" > /build/llama-cli-version.txt ) && \
        rm -rf llama-bin.tar.gz extracted_llama; \
    else \
        if [ "$ENABLE_TURBOQUANT" = "true" ]; then \
            echo "Building TurboQuant fork..." && \
            git clone https://github.com/spiritbuun/llama-cpp-turboquant-cuda.git llama.cpp && \
            cd llama.cpp && \
            git checkout feature/turboquant-kv-cache; \
        else \
            echo "Building standard llama.cpp..." && \
            git clone https://github.com/ggml-org/llama.cpp.git llama.cpp && \
            cd llama.cpp && \
            case "${LLAMA_CPP_REF}" in \
                pull/*) git fetch origin "${LLAMA_CPP_REF}" && git checkout FETCH_HEAD ;; \
                *) git fetch "${LLAMA_CPP_REMOTE}" "${LLAMA_CPP_REF}" && git checkout FETCH_HEAD ;; \
            esac && \
            if [ -n "${MTP_PATCH}" ] && [ -f "/build-context/${MTP_PATCH}" ]; then \
                echo "Applying MTP patch: ${MTP_PATCH}" && \
                git apply --check "/build-context/${MTP_PATCH}" && \
                git apply "/build-context/${MTP_PATCH}"; \
            fi && \
            echo "llama.cpp tree: $(git rev-parse --short HEAD) + ${MTP_PATCH:-no-patch}"; \
        fi && \
        echo "Building llama.cpp version: $(git describe --tags --always)" && \
        LLGUIDANCE_CMAKE="" && \
        if [ "$ENABLE_LLGUIDANCE" = "true" ]; then LLGUIDANCE_CMAKE="-DLLAMA_LLGUIDANCE=ON"; fi && \
        cmake . -B build \
            -DBUILD_SHARED_LIBS=OFF \
            -DGGML_CUDA=ON \
            -DGGML_CUDA_FA=ON \
            -DGGML_CUDA_FA_ALL_QUANTS=ON \
            -DLLAMA_CURL=ON \
            -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
            ${LLGUIDANCE_CMAKE} && \
        cmake --build build --config Release -j${CMAKE_BUILD_JOBS} \
            --target llama-server llama-cli llama-gguf-split && \
        echo "${LLAMACPP_BUILD_TAG}" > /build/llamacpp-build-tag.txt && \
        echo "${ENABLE_LLGUIDANCE}" > /build/llamacpp-llguidance-enabled.txt && \
        ./build/bin/llama-cli --version 2>/dev/null | head -1 > /build/llama-cli-version.txt || true && \
        echo "Built llama.cpp version: $(cat /build/llama-cli-version.txt 2>/dev/null || echo unknown)"; \
    fi

# [2026-10-15] | INFO | DOCKERFILE.RUNTIME_STAGE | uses the global RUNTIME_BASE ARG
# (declared before the first FROM); cpu variant passes ubuntu:22.04 to drop CUDA base
FROM ${RUNTIME_BASE}

ARG LLAMACPP_BUILD_TAG=b10723
LABEL org.opencontainers.image.version="${LLAMACPP_BUILD_TAG}"
ENV LLAMACPP_BUILD_TAG=${LLAMACPP_BUILD_TAG}

# Fix CUDA forward compatibility error on certain driver versions
RUN rm -rf /usr/local/cuda/compat

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
# [2026-10-15] | INFO | DOCKERFILE.RUNTIME_DEPS | libstdc++6 needed by upstream
# prebuilt binaries (dynamically linked); harmless when already present
RUN apt-get update && apt-get install -y \
    curl \
    libcurl4 \
    libstdc++6 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# [2026-10-16] | INFO | DOCKERFILE.PIP_PEP668 | ubuntu:24.04 runtime (needed by arm64
# prebuilts, glibc 2.38+) marks python externally-managed; this pip env var allows
# the install and is harmlessly ignored by 22.04's older pip
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# Install huggingface_hub for model downloads
RUN pip3 install huggingface_hub hf_transfer

# Create user for security
# [2026-10-16] | FIX | DOCKERFILE.USERADD | ubuntu:24.04 ships a default 'ubuntu'
# user at UID 1000; reuse the slot (useradd -o allows duplicate UID) so arm64
# builds don't die where 22.04 had no default user
RUN useradd -m -o -u 1000 llamacpp && \
    mkdir -p /home/llamacpp/models && \
    chown -R llamacpp:llamacpp /home/llamacpp

# Copy built binaries from builder stage
COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /build/llama.cpp/build/bin/llama-cli /usr/local/bin/
COPY --from=builder /build/llama.cpp/build/bin/llama-gguf-split /usr/local/bin/
# [2026-10-15] | INFO | DOCKERFILE.RUNTIME_LIBS | upstream prebuilts are dynamically
# linked with RUNPATH=$ORIGIN; bundled .so files must sit beside the binaries.
# Compiled-from-source builds leave /build/runtime-libs empty -> copy is a no-op.
# [2026-10-16] | FIX | DOCKERFILE.BACKEND_DIRS | ggml's dynamic backend loader
# (libggml-cpu-*.so) searches the EXECUTABLE directory — LD_LIBRARY_PATH only
# satisfies the linker. Upstream prebuilts ship dynamic backends, so they must
# sit beside llama-server in /usr/local/bin (the lib copy stays for the linker
# and any embedded consumers). Symptom without this: "no backends are loaded".
COPY --from=builder /build/runtime-libs/ /usr/local/lib/llama-nexus/
COPY --from=builder /build/runtime-libs/ /usr/local/bin/
# [2026-10-15] | INFO | DOCKERFILE.LD_PATH | plain value; ENV has no shell expansion
ENV LD_LIBRARY_PATH=/usr/local/lib/llama-nexus
COPY --from=builder /build/llamacpp-build-tag.txt /etc/llama-nexus/llamacpp-build-tag
COPY --from=builder /build/llama-cli-version.txt /etc/llama-nexus/llama-cli-version
COPY --from=builder /build/llamacpp-llguidance-enabled.txt /etc/llama-nexus/llamacpp-llguidance-enabled

# Copy configuration files
COPY scripts/start.sh /start.sh
COPY chat-templates/chat-template-oss.jinja /home/llamacpp/templates/chat-template-oss.jinja
COPY chat-templates/chat-template-basic.jinja /home/llamacpp/templates/chat-template-basic.jinja
RUN mkdir -p /home/llamacpp/templates && chmod +x /start.sh

# Set up environment for GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run as llamacpp user for security
USER llamacpp
WORKDIR /home/llamacpp

# Use start.sh as entrypoint
ENTRYPOINT ["/start.sh"]