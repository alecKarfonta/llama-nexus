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
# b9193+ includes merged MTP (Multi-Token Prediction) via ggml-org/llama.cpp PR #22673.
# Qwen3.6 35B MoE GGUFs use architecture qwen35moe — requires current llama.cpp (see src/models/qwen35moe.cpp).
# Override at build time: docker compose build --build-arg LLAMA_CPP_REF=b9496 llamacpp-api
ARG LLAMA_CPP_REF=b9193
ARG LLAMACPP_BUILD_TAG=b9193
WORKDIR /build
ARG SKIP_BUILD_FROM_SOURCE=false
ARG LLAMACPP_VERSION=b9193
ARG ENABLE_TURBOQUANT=false
# CUDA architecture(s) passed to CMake. Run on the host: nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# (use the major.minor as an integer SM, e.g. 8.6 -> 86). Common values: 75 (T4), 80 (A100), 86 (RTX 30xx),
# 89 (RTX 40xx), 90 (H100). Default 86 suits Ampere builds.
# IMPORTANT: Do not use "native" here when building Docker images — the builder usually has no GPU, so CMake
# cannot detect SM and nvcc fails with: Unsupported gpu architecture 'compute_' .
ARG CUDA_ARCH=86

RUN if [ "$SKIP_BUILD_FROM_SOURCE" = "true" ]; then \
        echo "Skipping build, downloading pre-built binaries version: ${LLAMACPP_VERSION}" && \
        mkdir -p /build/llama.cpp/build/bin && \
        curl -L -o llama.zip https://github.com/ggml-org/llama.cpp/releases/download/${LLAMACPP_VERSION}/llama-${LLAMACPP_VERSION}-bin-ubuntu-x64.zip && \
        unzip llama.zip -d extracted_llama && \
        find extracted_llama -name "llama-server" -exec cp {} /build/llama.cpp/build/bin/ \; && \
        find extracted_llama -name "llama-cli" -exec cp {} /build/llama.cpp/build/bin/ \; && \
        find extracted_llama -name "llama-gguf-split" -exec cp {} /build/llama.cpp/build/bin/ \; && \
        echo "${LLAMACPP_BUILD_TAG}" > /build/llamacpp-build-tag.txt && \
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
            git checkout ${LLAMA_CPP_REF}; \
        fi && \
        echo "Building llama.cpp version: $(git describe --tags --always)" && \
        cmake . -B build \
            -DBUILD_SHARED_LIBS=OFF \
            -DGGML_CUDA=ON \
            -DGGML_CUDA_FA=ON \
            -DGGML_CUDA_FA_ALL_QUANTS=ON \
            -DLLAMA_CURL=ON \
            -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" && \
        cmake --build build --config Release -j8 --clean-first \
            --target llama-server llama-cli llama-gguf-split && \
        echo "${LLAMACPP_BUILD_TAG}" > /build/llamacpp-build-tag.txt && \
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