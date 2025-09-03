# Multi-stage build for llama.cpp with CUDA support
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

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
    && rm -rf /var/lib/apt/lists/*

# Install huggingface_hub for model downloads
RUN pip3 install huggingface_hub hf_transfer

# Clone and build llama.cpp with CUDA support
WORKDIR /build
RUN git clone https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    git checkout b6181 && \
    echo "Building llama.cpp version: $(git describe --tags --always)" && \
    cmake . -B build \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON \
        -DLLAMA_CURL=ON \
        -DLLAMA_SERVER_VERBOSE=ON && \
    cmake --build build --config Release -j$(nproc) --clean-first \
        --target llama-server llama-cli llama-gguf-split && \
    echo "Built llama.cpp version: $(./build/bin/llama-cli --version | head -1)"

# Runtime stage
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

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

# Copy configuration files
COPY start.sh /start.sh
COPY chat-template-oss.jinja /home/llamacpp/chat-template.jinja
RUN chmod +x /start.sh

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

# Start the service
CMD ["/start.sh"]