# Multi-stage build for llama.cpp with CUDA support
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
# 2026-08-28: pinned to unslothai/llama.cpp@bea3b12da (the exact PR #27742 snapshot the
# qwen4exp MTP draft-head patch was verified against — dzannotti/Qwen3.8-Flash-Next-MTP-GGUF).
# The commit is orphaned on the fork (force-pushed past), so it is fetched by full SHA from
# a URL ref, which works even when no branch points at it.
# MTP_PATCH: optional git-apply patch layered on top (qwen4exp draft-head graph + converter
# --mtp export). Kept optional so a plain PR build needs no patch file present.
# Override at build time: docker compose build --build-arg LLAMA_CPP_REF=b9496 llamacpp-api
ARG LLAMA_CPP_REF=bea3b12daee45876b0129a3602dc8f534ce30bf0
ARG LLAMA_CPP_REMOTE=https://github.com/unslothai/llama.cpp.git
ARG MTP_PATCH=patches/qwen4exp-mtp-draft-head.patch
ARG LLAMACPP_BUILD_TAG=b10631-qwen4exp-mtp
WORKDIR /build
ARG SKIP_BUILD_FROM_SOURCE=false
ARG LLAMACPP_VERSION=b9193
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
# 89 (RTX 40xx), 90 (H100). Default 86 suits Ampere builds.
# IMPORTANT: Do not use "native" here when building Docker images — the builder usually has no GPU, so CMake
# cannot detect SM and nvcc fails with: Unsupported gpu architecture 'compute_' .
ARG CUDA_ARCH=86

# [TIME] | INFO | DOCKERFILE.PATCH_CONTEXT | the MTP patch lives in the build context
# (./patches/); COPY it into the builder so the build RUN can git-apply it.
COPY ${MTP_PATCH} /build-context/${MTP_PATCH}

RUN if [ "$SKIP_BUILD_FROM_SOURCE" = "true" ]; then \
        echo "Skipping build, downloading pre-built binaries version: ${LLAMACPP_VERSION}" && \
        mkdir -p /build/llama.cpp/build/bin && \
        curl -L -o llama.zip https://github.com/ggml-org/llama.cpp/releases/download/${LLAMACPP_VERSION}/llama-${LLAMACPP_VERSION}-bin-ubuntu-x64.zip && \
        unzip llama.zip -d extracted_llama && \
        find extracted_llama -name "llama-server" -exec cp {} /build/llama.cpp/build/bin/ \; && \
        find extracted_llama -name "llama-cli" -exec cp {} /build/llama.cpp/build/bin/ \; && \
        find extracted_llama -name "llama-gguf-split" -exec cp {} /build/llama.cpp/build/bin/ \; && \
        echo "${LLAMACPP_BUILD_TAG}" > /build/llamacpp-build-tag.txt && \
        echo "false" > /build/llamacpp-llguidance-enabled.txt && \
        /build/llama.cpp/build/bin/llama-cli --version 2>/dev/null | head -1 > /build/llama-cli-version.txt || \
            echo "llama.cpp ${LLAMACPP_VERSION}" > /build/llama-cli-version.txt && \
        rm -rf llama.zip extracted_llama; \
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

# Runtime stage
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

ARG LLAMACPP_BUILD_TAG=b9193
LABEL org.opencontainers.image.version="${LLAMACPP_BUILD_TAG}"
ENV LLAMACPP_BUILD_TAG=${LLAMACPP_BUILD_TAG}

# Fix CUDA forward compatibility error on certain driver versions
RUN rm -rf /usr/local/cuda/compat

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libcurl4 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install huggingface_hub for model downloads
RUN pip3 install huggingface_hub hf_transfer

# Create user for security
RUN useradd -m -u 1000 llamacpp && \
    mkdir -p /home/llamacpp/models && \
    chown -R llamacpp:llamacpp /home/llamacpp

# Copy built binaries from builder stage
COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /build/llama.cpp/build/bin/llama-cli /usr/local/bin/
COPY --from=builder /build/llama.cpp/build/bin/llama-gguf-split /usr/local/bin/
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