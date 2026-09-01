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
| Qdrant Vector DB | `qdrant` | 6333/6334 | Vector storage for RAG and GraphRAG embeddings |
| Streaming STT | `streaming-stt` | 8609 | NVIDIA Nemotron speech-to-text |
| TTS API | `tts` | 8605 | Text-to-speech synthesis |

Start optional services:
```bash
docker compose --profile streaming-stt up -d
docker compose --profile embed up -d
```

If another local GraphRAG or Qdrant stack already owns `6333/6334`, keep the
container-internal ports unchanged and remap only the host ports:

```bash
QDRANT_HTTP_PORT=16333 QDRANT_GRPC_PORT=16334 docker compose --profile embed up -d qdrant llamacpp-embed
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
- **Tracked Document Pipelines**: Upload previews plus single-file and batch traced uploads with structure extraction, chunking, graph-candidate, balanced relationship highlights, embedding-input samples, interaction handoff lanes, and retrieval rehearsals for chat-ready graph work
- **Rich Document Understanding**: PDF/Office/OpenDocument layout, macro-enabled and template Office variants, best-effort legacy DOC/PPT/XLS OLE stream text with stream provenance, source-anchored hyperlinks, PDF links/forms/images/annotations/tables/OCR with readable page bboxes and zones, PowerPoint/Word/OpenDocument table cell coordinates/spans/merge state/styling, PowerPoint table-cell fills/borders/margins/text anchors, PowerPoint object geometry with readable slide bboxes and zones, charts with title/axis/legend/data-label context plus series/category/value graph edges, chart series/point visual cues and trendlines, workbook sheet visibility/source-part provenance, sheet-local spreadsheet chart/link/image anchors, spreadsheet formula dependency edges, conditional-formatting rule graph edges, pivot-table source and field graph edges, PowerPoint/OpenDocument presentation slide visibility/source-part provenance, hidden-slide graph edges, speaker notes with slide/source-part provenance, shape accessibility metadata, connector arrows, inherited slide layout/master text and images, Word drawing/VML text boxes, anchored DOCX/DOCM comments, anchored footnotes/endnotes, merged spreadsheet labels, Excel table definitions, named ranges, spreadsheet data-validation rules, and classic/threaded spreadsheet cell comments, OCR text, embedded PDF attachments, standalone images, ZIP bundle entries, and nested Office/PDF artifacts are surfaced as graph evidence
- **Graph-Augmented Chat Context**: Chat retrieval expands with discovered graph path terms, provenance from document structure, and labels graph-sourced evidence
- **Reasoning Playground**: Interactive graph exploration

### Deployment

```bash
# Start the llama-nexus GraphRAG helper stack.
GRAPHRAG_ENABLED=true docker compose --profile graphrag up -d

# Optional: point graph/search/ingest/reasoning routes at an external GraphRAG API.
GRAPHRAG_ENABLED=true GRAPHRAG_URL=http://host.docker.internal:18000 docker compose --profile graphrag up -d
```

When the backend runs in Docker with `GRAPHRAG_ENABLED=true`, it auto-discovers the bundled helper services at `graphrag-ner:8001`, `graphrag-rel:8002`, and `graphrag-code:8000` unless explicit URLs are provided. Health, capability, and deployment diagnostics endpoints report endpoint source provenance, distinguishing explicit env vars from Docker service defaults, plus partial readiness so NER/REL/Code-RAG can be used while an external graph API is still being wired up. `/api/v1/graphrag/services/status` also returns workflow readiness for local pipeline preview, remote ingest, graph search/reasoning, extraction, Code-RAG, and crawler routes so operators can see what works immediately; its `interaction_contract` summarizes the current Upload, Chat, and Search behavior, including the health preflight URL, adaptive ingest endpoint candidates, endpoint source, smoke-test probes, and trace fields where attempted URLs, endpoint attempts, and the selected endpoint are recorded. Its `operator_checklist` turns deployment state into ordered setup/upload/chat/search actions, and `deployment_quickstart` returns command-ready runbook cards for local trace/chat, helper extraction, remote GraphRAG ingest/search, and OCR runtime setup that the GraphRAG Workbench displays. Pipeline uploads preflight the configured graph API and then adapt across compatible remote ingest endpoints (`/ingest-documents`, `/process-document`, and `/add-to-vector-store`), recording the selected endpoint and every attempt in the trace. Pipeline trace history is stored at `GRAPHRAG_PIPELINE_RUNS_FILE` and defaults to `/data/graphrag/pipeline_runs.json` in Docker. Tracked pipeline runs include an evidence profile with archive entries, attachments, OCR blocks, distinct tables, workbook sheets and hidden sheets, presentation slides and hidden slides, pivot tables, chart relationship counts, PowerPoint shapes, connectors, inherited layout/master evidence, table-derived facts with table-cell visual cues, and layout/path relationships so operators can see exactly what will reach GraphRAG before remote ingestion. Each run also keeps exportable artifacts for structured ingest Markdown, local graph, local context, graph reasoning, samples, stages, and the interaction handoff, plus an `interaction_handoff` with Trace/Graph/Paths/Retrieval/Chat lanes, capability cards, graph anchors, starter prompts, retrieval rehearsals, and citation scopes; layout-aware traces expose a layout evidence card, layout starter prompt, and layout citation scope, while PPTX and ODP decks also expose a presentation-structure card and starter prompt for slide visibility/source-part review. Chat consumes those handoffs as graph-reasoning evidence, retrieval terms, answer-plan cards, and a `Pipeline Interaction Handoff` context section, allowing it to retrieve recent upload chunks and answer path/layout/chart/sheet-visibility/slide-visibility questions from pipeline traces even before remote GraphRAG ingestion is available. The `/api/v1/graphrag/search` route and dedicated `/api/v1/graphrag/reasoning/*` routes also fall back to recent local pipeline chunks, graph edges, and inferred graph paths when remote GraphRAG is unavailable or when `pipeline_run_ids`/`use_local_pipeline_context` are supplied. Retrieval rehearsals show which starter prompts already match graph anchors and run-scoped chunks, and batch traces roll those rehearsal counts into both summaries and upload manifests. Chat answer plans include an evidence checklist that turns graph hops, handoff anchors, and supporting relationships into citation-ready cards with document provenance, including readable layout bbox/zone provenance, PowerPoint table-cell visual provenance, workbook sheet visibility/source-part provenance, presentation slide visibility/source-part provenance, and spreadsheet formula, conditional-format, named-range, validation, and pivot-table lineage. Batch traces roll evidence counts up across files and return an `upload_manifest` with per-file readiness, layout-region counts, interaction mode counts, payload kind, byte counts, Upload/Trace/Chat/Remote Graph lanes, and the recommended next action. ZIP entries and PDF attachments can themselves be structured Office/PDF/OpenDocument/EPUB/image/text artifacts, with child evidence carried forward into chunks and graph relationships.

Chat requests with `pipeline_run_ids` remain the most precise way to continue from a specific upload, but graph/path/relationship questions without run IDs also probe recent pipeline traces in the matching domain so ad hoc chat stays useful after an upload.

Quickly validate the live upload-to-chat path with:
```bash
scripts/graphrag_pipeline_smoke.sh --domain manual-smoke /path/to/document.pptx
```
For small/simple files where a multi-hop path may not exist yet, run the same
command with `--allow-no-paths` or `GRAPHRAG_REQUIRE_PATHS=false` (or
`0`/`no`/`off`).
For staged local-only checks before remote ingest is configured, also set
`--local-only` or `GRAPHRAG_UPLOAD_TO_REMOTE=false` (or `0`/`no`/`off`).

Manual Workbench test path:
1. Open `http://localhost:3002/graphrag` and confirm **Service Smoke Check** reports ready workflows with no warnings.
2. In **Corpus**, choose a document and use **Preview Trace** for local samples or **Upload with Trace** for the full remote-ingest path.
3. Inspect the remembered run for stage samples, artifacts, graph anchors, retrieval rehearsals, and citation scopes.
4. Use **Ask Chat** from the run, then ask a follow-up graph/path question in the same domain to confirm recent pipeline traces are reused.

### GraphRAG Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/v1/graphrag/entity/link` and `/api/v1/graphrag/entity/disambiguate` | Entity linking and disambiguation |
| `/api/v1/graphrag/search` and `/api/v1/graphrag/search/intelligent` | Hybrid search with local pipeline graph/chunk fallback |
| `/api/v1/graphrag/code/search` | Code-aware search when Code-RAG is configured |
| `/api/v1/graphrag/reasoning/*` | Multi-hop, causal, comparative, advanced, and relationship explanation queries |
| `/api/v1/graphrag/ingest/pipeline/preview` | Local document extraction/chunking/graph-candidate preview |
| `/api/v1/graphrag/ingest/pipeline/upload` | Upload with remembered per-stage pipeline trace |
| `/api/v1/graphrag/ingest/pipeline/batch` | Multi-file upload with bounded concurrency, shared service preflight, remembered per-file traces, and an upload manifest |
| `/api/v1/graphrag/chat-context-v2` | Unified RAG plus graph context with graph-augmented retrieval signals |

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
# From the repo root, start the local GraphRAG document-chat backend with workspace-safe data paths
scripts/start-graphrag-backend-local.sh

# Or start the backend manually in development mode
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
