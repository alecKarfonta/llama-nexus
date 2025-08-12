# Llama Nexus

> A comprehensive model management platform with web UI control over LlamaCPP deployments

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green.svg)](https://developer.nvidia.com/cuda-zone)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-orange.svg)](https://openai.com/blog/openai-api)

**Llama Nexus** is a complete model management system featuring real-time monitoring, configuration management, and full lifecycle control for LlamaCPP deployments. Built with modern web technologies and optimized for high-performance GPU inference.

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [✨ Features](#features)
- [🤖 GPT-OSS Model Details](#-gpt-oss-model-details)
- [🔌 API Endpoints](#-api-endpoints)
- [💻 Usage Examples](#-usage-examples)
- [💾 Model Caching & Storage](#-model-caching--storage)
- [⚙️ Configuration](#️-configuration)
- [🚀 Performance Optimization](#-performance-optimization)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [📚 Source & References](#-source--references)
- [🗺️ Next Steps](#️-next-steps)

## 🚀 Quick Start

```bash
# Build and start the complete system
docker compose up -d --build

# Check status
docker compose ps

# Access services
# Management UI: http://localhost:3000
# Backend API:   http://localhost:8700  
# LlamaCPP API:  http://localhost:8600

# Control LlamaCPP from the web UI
# 1. Open http://localhost:3000
# 2. Go to Dashboard
# 3. Click "Start" to launch LlamaCPP
# 4. Monitor logs and status in real-time
# 5. Use Configuration page to adjust settings
```

## ✨ Features

### 🎛️ Model Management
- **🔄 Service Control**: Start, stop, and restart LlamaCPP instances from UI
- **⚙️ Configuration Management**: Edit all LlamaCPP parameters with validation
- **📊 Real-time Monitoring**: Live resource usage (CPU, Memory, GPU, VRAM)
- **📋 Log Streaming**: Live log viewer with search and filtering
- **🎯 Configuration Presets**: Quick switching between optimized settings
- **💻 Command Line View**: See exact commands being executed

### 🚀 API Backend
- **🧠 GPT-OSS Support**: Proper support for OpenAI's gpt-oss-20b reasoning model
- **⚡ RTX 5090 Optimized**: Maximum GPU utilization with CUDA acceleration
- **🔌 OpenAI API Compatible**: Drop-in replacement for OpenAI API clients
- **📝 Reasoning Traces**: Access to full chain-of-thought reasoning process
- **💾 Persistent Storage**: Models cached in Docker volumes
- **🚀 High Performance**: Optimized llama.cpp backend with flash attention

## 🤖 GPT-OSS Model Details

**GPT-OSS-20B** is OpenAI's open-weight reasoning model designed for powerful reasoning, agentic tasks, and versatile developer use cases.

### Model Specifications
- **Parameters**: 21B total (3.6B active parameters MoE)
- **Context Length**: 131,072 tokens
- **License**: Apache 2.0 (commercial-friendly)
- **Quantization**: Q4_K_M (optimal quality/speed balance)
- **VRAM Usage**: ~12-16GB on RTX 5090

### Key Features
- **🧠 Advanced Reasoning**: Superior performance on complex problem-solving
- **🔍 Chain-of-Thought**: Full access to reasoning process via channels
- **🛠️ Tool Use**: Native function calling and structured outputs
- **📊 Channel System**: analysis, commentary, final channels for detailed reasoning
- **🎯 Fine-tunable**: Supports LoRA and full fine-tuning

### Optimized Settings
Per [Unsloth recommendations](https://docs.unsloth.ai/basics/gpt-oss-how-to-run-and-fine-tune#run-gpt-oss-20b):
- **Temperature**: 1.0 (default)
- **Top-P**: 1.0 (default)  
- **Top-K**: 0 (disabled)
- **Context**: 131K tokens
- **Reasoning Level**: medium (configurable in system prompt)

## 🔌 API Endpoints

### OpenAI Compatible API
- **Chat Completions**: `POST /v1/chat/completions`
- **Completions**: `POST /v1/completions`
- **Models**: `GET /v1/models`
- **Embeddings**: `POST /v1/embeddings`

### Llama.cpp Native API
- **Health**: `GET /health`
- **Metrics**: `GET /metrics`
- **Props**: `GET /props`

## 💻 Usage Examples

### Python OpenAI Client
```python
import openai

# Configure client
client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="llamacpp-gpt-oss"
)

# Chat completion with reasoning
response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[
        {
            "role": "system", 
            "content": "You are a helpful assistant. Use high-level reasoning. Reasoning: high"
        },
        {
            "role": "user", 
            "content": "Explain quantum entanglement in simple terms, showing your reasoning process."
        }
    ],
    max_tokens=800,
    temperature=1.0,
    top_p=1.0
)

print(response.choices[0].message.content)
```

### cURL Examples
```bash
# List available models
curl http://localhost:8080/v1/models \
    -H "Authorization: Bearer llamacpp-gpt-oss"

# Mathematical reasoning
curl -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer llamacpp-gpt-oss" \
    -d '{
        "model": "gpt-oss-20b",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert mathematician. Show your reasoning step by step. Reasoning: high"
            },
            {
                "role": "user", 
                "content": "If a train travels 120 km in 1.5 hours, then speeds up by 20 km/h for the next 2 hours, what is the total distance traveled?"
            }
        ],
        "max_tokens": 600,
        "temperature": 1.0,
        "top_p": 1.0
    }'
```

## 💾 Model Caching & Storage

Models are automatically cached in persistent Docker volumes:

- **Storage Location**: Docker volume `gpt_oss_models`
- **Automatic Caching**: Models download once, persist across restarts
- **Model Size**: ~11-13GB for Q4_K_M quantization

### Managing Model Cache
```bash
# View cached models and storage usage
docker compose exec llamacpp-api ls -la /home/llamacpp/models/
docker system df -v

# Check specific volume usage
docker volume inspect llamacpp-api_gpt_oss_models

# Clear model cache (if needed)
docker compose down
docker volume rm llamacpp-api_gpt_oss_models
docker compose up -d
```

## ⚙️ Configuration

### Environment Variables
Configure via `docker-compose.yml` environment section:

#### LlamaCPP API Settings
```yaml
environment:
  # Model settings
  - MODEL_NAME=gpt-oss-20b
  - MODEL_VARIANT=Q4_K_M        # Q4_K_M, F16, Q8_0, etc.
  - CONTEXT_SIZE=131072         # Maximum context length
  - GPU_LAYERS=99               # Number of layers on GPU
  
  # Inference settings (per Unsloth recommendations)  
  - TEMPERATURE=1.0
  - TOP_P=1.0
  - TOP_K=0
  - MIN_P=0.0
  
  # Performance tuning
  - THREADS=-1                  # CPU threads (-1 = auto)
  - BATCH_SIZE=2048            # Batch size for processing
  - UBATCH_SIZE=512            # Micro-batch size
```

#### Frontend Settings
```yaml
environment:
  # API connection settings
  - VITE_API_BASE_URL=http://192.168.1.77:8600  # LlamaCPP API URL
  - VITE_BACKEND_URL=http://192.168.1.77:8700   # Backend Management API URL
```

> **Important**: If you're experiencing API connectivity issues in the frontend, make sure these environment variables are correctly set in the docker-compose.yml file.

### Reasoning Levels
Configure reasoning level in system prompts:
- **Low**: `"Reasoning: low"` - Fast responses
- **Medium**: `"Reasoning: medium"` - Balanced (default)  
- **High**: `"Reasoning: high"` - Deep analysis

### Chat Template
The server uses a custom Jinja2 template (`gpt-oss-template.jinja`) optimized for gpt-oss models with proper channel support:
- **Channels**: `analysis`, `commentary`, `final`
- **Special Tokens**: `<|start|>`, `<|message|>`, `<|channel|>`, `<|end|>`
- **EOS Token**: `<|return|>`

## 🚀 Performance Optimization

### RTX 5090 Optimizations
- **CUDA Acceleration**: Full GPU acceleration with 99 layers
- **Flash Attention**: Enabled for better performance
- **Continuous Batching**: Improved throughput for multiple requests
- **Memory Optimization**: Optimized for 32GB VRAM

### Monitoring
```bash
# Server logs
docker compose logs -f llamacpp-api

# GPU usage
nvidia-smi -l 1

# API metrics
curl http://localhost:8080/metrics
```

## 🛠️ Troubleshooting

### Model Loading Issues
```bash
# Check model download
docker compose exec llamacpp-api ls -la /home/llamacpp/models/

# Verify GPU detection
docker compose exec llamacpp-api nvidia-smi

# Check server logs
docker compose logs llamacpp-api --tail=50
```

### Memory Issues
- Reduce `GPU_LAYERS` if out of VRAM
- Use smaller quantization (Q4_K_M → Q8_0 → F16)
- Reduce `CONTEXT_SIZE` for lower memory usage
- Adjust `BATCH_SIZE` and `UBATCH_SIZE`

### API Connection Issues
```bash
# Test health endpoint
curl http://localhost:8080/health

# Check if port is accessible
netstat -tlnp | grep 8080
```

## 📚 Source & References

- **GPT-OSS Model**: [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
- **GGUF Quantized**: [unsloth/gpt-oss-20b-GGUF](https://huggingface.co/unsloth/gpt-oss-20b-GGUF)
- **Llama.cpp**: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- **Unsloth Guide**: [GPT-OSS How to Run & Fine-tune](https://docs.unsloth.ai/basics/gpt-oss-how-to-run-and-fine-tune#run-gpt-oss-20b)

## 🗺️ Next Steps

1. **Model Variants**: Support for gpt-oss-120b (requires more VRAM)
2. **Function Calling**: Add structured outputs and tool use
3. **Fine-tuning**: Integration with Unsloth for custom training
4. **Multi-GPU**: Support for model parallelism across multiple GPUs
5. **Load Balancing**: Multiple instances for production scaling