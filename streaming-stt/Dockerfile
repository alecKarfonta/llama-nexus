# NeMo 25.11 with native Blackwell (sm_120) support
# Requires NVIDIA Driver 580.82+
FROM nvcr.io/nvidia/nemo:25.11

# Set working directory
WORKDIR /app

# Install ffmpeg for pydub audio processing
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose WebSocket port
EXPOSE 8009

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8009/health || exit 1

# Run the streaming STT server
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8009", "--ws-ping-interval", "0"]
