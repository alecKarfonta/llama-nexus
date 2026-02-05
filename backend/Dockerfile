FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Docker CLI, Node.js for MCP servers, and NVIDIA tools
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    && mkdir -p /etc/apt/keyrings \
    # Docker CLI
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    # Node.js 20 LTS for MCP (Model Context Protocol) servers
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update \
    && apt-get install -y docker-ce-cli nodejs \
    && rm -rf /var/lib/apt/lists/* \
    # Verify Node.js and npm are installed
    && node --version && npm --version

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install lm-evaluation-harness for standardized benchmarks
RUN pip install --no-cache-dir "lm_eval[api]"

# Copy application code
COPY main.py enhanced_logger.py tool_call_tests.json training_worker.py ./
COPY modules/ modules/
COPY routes/ routes/

# Expose port
EXPOSE 8700

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8700/health || exit 1

# Create a logging configuration file
RUN echo '{"version": 1, "disable_existing_loggers": true, "formatters": {"minimal": {"format": "%(message)s"}}, "handlers": {"null": {"class": "logging.NullHandler"}}, "loggers": {"uvicorn": {"handlers": ["null"], "level": "WARNING"}, "uvicorn.error": {"handlers": ["null"], "level": "WARNING"}, "uvicorn.access": {"handlers": ["null"], "level": "WARNING"}}}' > /app/log_config.json

# Run the application with custom logging configuration
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8700", "--log-config", "/app/log_config.json"]
