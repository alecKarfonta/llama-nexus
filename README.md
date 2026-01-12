# Llama Nexus

> A comprehensive AI model orchestration platform with web-based deployment, monitoring, and management

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green.svg)](https://developer.nvidia.com/cuda-zone)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-orange.svg)](https://openai.com/blog/openai-api)

**Llama Nexus** is a full-stack model management platform that provides web UI control over multiple AI services including LLM inference (LlamaCPP/vLLM), text-to-speech, speech-to-text, embeddings, and knowledge graph systems. Built for production workloads with NVIDIA GPU acceleration.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Core Services](#core-services)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [GraphRAG Integration](#graphrag-integration)
- [Workflow Engine](#workflow-engine)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/alecKarfonta/llama-nexus.git
cd llama-nexus

# Build and start core services
docker compose up -d --build

# Check service status
docker compose ps

# Access the web interface
# Frontend UI:     http://localhost:3002
# Backend API:     http://localhost:8700
# LlamaCPP API:    http://localhost:8600
```

### First-Time Setup

1. Open the web interface at `http://localhost:3002`
2. Navigate to the **Deploy** page
3. Select a model from the registry or download a new one
4. Configure inference parameters (context size, GPU layers, sampling)
5. Click **Deploy** to start the LLM service
6. Monitor logs and resource usage in real-time

## Architecture Overview

Llama Nexus uses a microservices architecture with Docker containers:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend (React/Vite)                        │
│                         localhost:3002                               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Backend API (FastAPI)                           │
│                      localhost:8700                                  │
│  ┌─────────────┬──────────────┬──────────────┬──────────────┐       │
│  │   Deploy    │   Models     │   Workflows  │   GraphRAG   │       │
│  │   Manager   │   Registry   │   Engine     │   Proxy      │       │
│  └─────────────┴──────────────┴──────────────┴──────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  LlamaCPP   │ │   vLLM      │ │  Embedding  │ │   Qdrant    │
│  API        │ │   API       │ │   Server    │ │   Vector DB │
│  :8600      │ │   :8601     │ │   :8602     │ │   :6333     │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

### Optional Services (via Docker profiles)

| Service | Profile | Port | Description |
|---------|---------|------|-------------|
| vLLM | `vllm` | 8601 | High-throughput inference backend |
| Embedding Server | `embed` | 8602 | Dedicated embedding service |
| Streaming STT | `streaming-stt` | 8609 | NVIDIA Nemotron speech-to-text |
| TTS API | `tts` | 8605 | Text-to-speech synthesis |

Start optional services:
```bash
docker compose --profile streaming-stt up -d
docker compose --profile embed up -d
```

## Core Services

### LLM Inference (LlamaCPP)

The primary inference engine supporting:
- GGUF model format with multiple quantization levels
- Vision-language models (VLM) with multimodal projection
- Flash attention for optimized memory usage
- Continuous batching for throughput
- OpenAI-compatible API endpoints

### Model Registry

Web-based model management:
- Browse and download models from HuggingFace
- Track downloaded models with metadata
- Configure per-model sampling parameters
- Support for multi-part GGUF files

### Backend API

FastAPI service providing:
- Docker container orchestration for LlamaCPP instances
- Real-time log streaming via WebSocket
- Configuration management with persistence
- Health monitoring and metrics collection

## API Reference

### OpenAI-Compatible Endpoints

All endpoints are available at `http://localhost:8600/v1/`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion with messages |
| `/v1/completions` | POST | Text completion |
| `/v1/models` | GET | List available models |
| `/v1/embeddings` | POST | Generate embeddings |

### Management API

Available at `http://localhost:8700/`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/service/status` | GET | Current service status |
| `/v1/service/action` | POST | Start/stop/restart service |
| `/v1/service/config` | GET/PUT | Configuration management |
| `/v1/service/logs` | WebSocket | Real-time log streaming |
| `/api/models/` | GET | List downloaded models |
| `/api/models/download` | POST | Initiate model download |

### Usage Examples

**Python (OpenAI SDK)**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8600/v1",
    api_key="placeholder-api-key"
)

response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    max_tokens=500,
    temperature=0.7
)

print(response.choices[0].message.content)
```

**cURL**
```bash
curl -X POST http://localhost:8600/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer placeholder-api-key" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Configuration

### Environment Variables

Key configuration options in `docker-compose.yml`:

```yaml
environment:
  # Model Configuration
  - MODEL_NAME=Qwen_Qwen3-VL-4B-Thinking
  - MODEL_VARIANT=Q4_K_M
  - CONTEXT_SIZE=40960
  - GPU_LAYERS=999

  # Sampling Parameters
  - TEMPERATURE=0.7
  - TOP_P=0.8
  - TOP_K=20
  - REPEAT_PENALTY=1.05

  # Performance
  - BATCH_SIZE=2048
  - N_CPU_MOE=0
```

### Multimodal Models

For vision-language models, set the multimodal projection file via the Deploy UI. The `MMPROJ_FILE` parameter is only applied when explicitly configured for VL models.

### Model Storage

Models are stored in persistent Docker volumes:
- `gpt_oss_models` - GGUF model files
- `conversation_data` - Chat history and conversations

```bash
# View model storage
docker volume inspect llama-nexus_gpt_oss_models

# List downloaded models
docker compose exec backend-api ls -la /home/llamacpp/models/
```

## GraphRAG Integration

Llama Nexus includes integration with GraphRAG for knowledge graph-enhanced retrieval:

### Features

- **Entity Extraction**: Automatic entity and relationship extraction from documents
- **Knowledge Graph**: Neo4j-backed graph storage
- **Hybrid Search**: Combined vector and graph-based retrieval
- **Code Search**: Semantic code search with AST awareness
- **Reasoning Playground**: Interactive graph exploration

### GraphRAG Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/graphrag/entities` | Entity management |
| `/api/graphrag/search` | Hybrid search |
| `/api/graphrag/code-search` | Code-aware search |
| `/api/graphrag/reasoning` | Graph reasoning queries |

## Workflow Engine

Build automated pipelines with the visual workflow builder:

### Node Types

- **LLM Nodes**: Chat completion, text generation
- **GraphRAG Nodes**: Entity search, code search, reasoning
- **Document Nodes**: Ingestion, chunking, processing
- **Utility Nodes**: Transformers, conditionals, outputs

### Workflow Templates

Pre-built templates for common use cases:
- RAG Pipeline
- Multi-Model Comparison
- Document Processing
- Chat with Memory

## Performance Tuning

### GPU Optimization

For NVIDIA GPUs (tested on RTX 5090):

```yaml
environment:
  - GPU_LAYERS=999          # Offload all layers to GPU
  - BATCH_SIZE=2048         # Increase for throughput
  - UBATCH_SIZE=512         # Micro-batch size
  - CUDA_VISIBLE_DEVICES=0  # Specific GPU selection
```

### Memory Management

| Model Size | Recommended Quantization | VRAM Usage |
|------------|-------------------------|------------|
| 7B | Q4_K_M | ~6GB |
| 13B | Q4_K_M | ~10GB |
| 24B | Q8_0 | ~26GB |
| 70B | Q4_K_M | ~40GB |

### Monitoring

```bash
# GPU utilization
nvidia-smi -l 1

# Container logs
docker compose logs -f llamacpp-api

# API metrics
curl http://localhost:8600/metrics
```

## Troubleshooting

### Container Startup Issues

```bash
# Check container status
docker compose ps

# View startup logs
docker compose logs backend-api --tail=100

# Restart services
docker compose restart backend-api
```

### Model Loading Failures

```bash
# Verify model files exist
docker compose exec llamacpp-api ls -la /home/llamacpp/models/

# Check GPU availability
docker compose exec llamacpp-api nvidia-smi

# Review LlamaCPP logs
docker logs llamacpp-api --tail=50
```

### Connection Issues

```bash
# Test backend health
curl http://localhost:8700/health

# Test LlamaCPP health
curl http://localhost:8600/health

# Check network connectivity
docker network ls
docker network inspect llama-nexus_default
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | Model too large for VRAM | Reduce `GPU_LAYERS` or use smaller quantization |
| `Connection refused` | Service not started | Check `docker compose ps` and start services |
| `Model not found` | GGUF file missing | Download model via UI or check model path |
| `mmproj mismatch` | Wrong multimodal projector | Clear mmproj config for non-VL models |

## Development

### Local Development

```bash
# Start backend in development mode
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8700

# Start frontend in development mode
cd frontend
npm install
npm run dev
```

### Building Images

```bash
# Build all images
docker compose build

# Build specific service
docker compose build backend-api

# Build with no cache
docker compose build --no-cache
```

## License

This project is licensed under the MIT License.

## References

- [Llama.cpp](https://github.com/ggml-org/llama.cpp) - LLM inference engine
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [Vite](https://vitejs.dev/) - Frontend build tool
- [HuggingFace](https://huggingface.co/) - Model repository