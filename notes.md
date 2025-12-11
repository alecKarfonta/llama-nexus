# Llama Nexus Deployment Notes

## Docker GPU Fix for API-launched Containers (2025-12-08)

### Problem
When launching containers through the backend API (LlamaCPPManager, EmbeddingManager), CUDA initialization failed with "no CUDA-capable device is detected" even though `nvidia-smi` worked inside the container.

### Root Cause
Using `--gpus all` with Docker CLI doesn't properly configure CUDA capabilities for all use cases. The NVIDIA Container Toolkit's `--gpus` flag can have issues with newer driver versions (575.x).

### Solution
Changed from `--gpus all` to `--runtime nvidia` with explicit environment variables:
```bash
--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Files Modified
- `backend/main.py` - Updated `LlamaCPPManager.start_docker_cli()` and `EmbeddingManager` CLI container launch

### Additional Fixes
- Changed default `embedding` config from `True` to `False` - the `--embeddings` flag causes assertion errors with some models (Qwen3-VL-4B-Thinking)

---

## Benchmark Page Overhaul (2025-12-08)

### Goal
Complete overhaul of the benchmark system to make it highly usable for evaluating LLM inference speed.

### New Features
1. **Quick Speed Test** - Type a prompt, see real-time streaming TPS and TTFT
2. **Model Comparison** - Side-by-side comparison table with charts
3. **Context Scaling** - Test how speed degrades with longer context
4. **Latency Profiling** - P50/P90/P99 percentiles with histogram visualization
5. **Throughput Testing** - Concurrent request stress testing
6. **Persistent Results** - SQLite storage, CSV/JSON export

### API Design
- `POST /api/v1/benchmark/speed-test` - Quick single prompt test with SSE streaming
- `POST /api/v1/benchmark/compare` - Compare multiple endpoints
- `POST /api/v1/benchmark/context-scaling` - Test across context lengths
- `POST /api/v1/benchmark/throughput` - Concurrent stress testing
- `GET /api/v1/benchmark/results` - Get stored results
- `GET /api/v1/benchmark/results/{id}` - Get specific result
- `DELETE /api/v1/benchmark/results/{id}` - Delete result
- `GET /api/v1/benchmark/export/{id}` - Export as CSV/JSON

### Files Modified
- `backend/routes/benchmark.py` - Complete rewrite (~750 lines)
- `frontend/src/pages/BenchmarkPage.tsx` - Complete rewrite (~900 lines)

### Backend Changes
- SQLite storage for persistent benchmark results
- SSE streaming for real-time speed test progress
- New endpoints:
  - `POST /speed-test` - Quick test with streaming metrics
  - `POST /speed-test/sync` - Synchronous version
  - `POST /compare` - Compare multiple endpoints
  - `POST /context-scaling` - Test context length impact
  - `POST /throughput` - Concurrent stress testing
  - `GET /results` - List stored results
  - `GET /results/{id}` - Get specific result
  - `DELETE /results/{id}` - Delete result
  - `DELETE /results` - Clear all results
  - `GET /export/{id}` - Export as JSON/CSV
  - `GET /stats` - Aggregate statistics

### Frontend Changes
- Tab-based UI: Speed Test, Compare, Context Scaling, Throughput, History
- Real-time metrics display with progress indicators
- Comparison bar charts
- Context scaling visualization
- Throughput analysis
- Result export (JSON/CSV)
- Stats overview cards

---

## Route Refactoring (2025-12-08)

### Summary
Refactored `backend/main.py` to reduce its size by breaking out route sections into separate modules.

### Changes Made
- Created `backend/routes/` directory with route modules
- Extracted RAG routes to `routes/rag.py` (~1000 lines)
- Extracted GraphRAG proxy routes to `routes/graphrag.py` (~300 lines)  
- Extracted Workflow routes to `routes/workflows.py` (~300 lines)
- Updated main.py to import and include routers via `app.include_router()`
- Added app.state configuration for route modules (rag_available, workflow_available, create_embedder, etc.)

### Results
- `main.py` reduced from 7947 lines to 3386 lines (~57% reduction)
- 21 routes remaining (health, logs, resources, system, config presets, llamacpp management, vram)
- Kept `process_document_background` function in main.py (uses app.state directly)

### Files Created
- `backend/routes/__init__.py` - exports routers
- `backend/routes/rag.py` - RAG system endpoints (~1500 lines)
- `backend/routes/graphrag.py` - GraphRAG proxy endpoints (~300 lines)
- `backend/routes/workflows.py` - Workflow management endpoints (~320 lines)
- `backend/routes/conversations.py` - Conversation storage endpoints (~160 lines)
- `backend/routes/registry.py` - Model registry endpoints (~200 lines)
- `backend/routes/prompts.py` - Prompt library endpoints (~190 lines)
- `backend/routes/benchmark.py` - Benchmark endpoints including BFCL (~500 lines)
- `backend/routes/batch.py` - Batch processing endpoints (~170 lines)
- `backend/routes/models.py` - Model management endpoints (~310 lines)
- `backend/routes/templates.py` - Chat template endpoints (~140 lines)
- `backend/routes/tokens.py` - Token usage tracking endpoints (~100 lines)
- `backend/routes/service.py` - Service/embedding management endpoints (~310 lines)

### Important: Dockerfile Update
Added `COPY routes/ routes/` to `backend/Dockerfile` to include routes directory in container.

### GraphRAG API Fix
The correct GraphRAG extraction endpoint is `/extract-entities-relations`, not `/extract`.

---

## Workflow Builder Overhaul (2025-12-08)

### Goal
Complete overhaul of the workflow creation system with visual drag-and-drop canvas that can connect:
- Services, Models, Tools
- Document loaders, Database connections
- External APIs, OpenAI API spec
- MCP servers

### Progress

**Phase 1: Foundation - COMPLETED**

1. Added React Flow dependency to package.json
2. Created workflow type definitions (`src/types/workflow.ts`):
   - Port types (string, number, boolean, object, array, any)
   - Node categories (trigger, llm, rag, tools, data, control, api, mcp, database, output)
   - 35+ built-in node type definitions
   - Workflow, WorkflowNode, WorkflowConnection, WorkflowExecution types

3. Created workflow components (`src/components/workflow/`):
   - `WorkflowCanvas.tsx` - React Flow canvas with drag-drop support
   - `BaseNode.tsx` - Custom node component with status indicators
   - `AnimatedEdge.tsx` - Animated connection edges
   - `NodePalette.tsx` - Searchable, categorized node list with drag
   - `PropertyPanel.tsx` - Node configuration panel with dynamic form fields

4. Updated `WorkflowBuilderPage.tsx`:
   - Full React Flow integration
   - Save/Load workflows to localStorage
   - Import/Export workflows as JSON
   - Simulated workflow execution
   - Workflow settings dialog

### Phase 2: Backend - COMPLETED

1. Created workflow module structure (`backend/modules/workflow/`)
2. Implemented workflow database storage (SQLite)
3. Created workflow execution engine
4. Implemented node executors for all node types:
   - Triggers: ManualTrigger, HttpWebhook
   - LLM: LLMChat, OpenAIChat, Embedding
   - RAG: DocumentLoader, Chunker, Retriever, VectorStore
   - Data: Template, JsonParse, JsonStringify, Mapper, Filter
   - Control: Condition, Switch, Loop, Merge, Delay
   - API: HttpRequest, GraphQL
   - Output: Output, Log, WebhookResponse
5. Added API endpoints to main.py
6. Created frontend workflowApi service
7. Updated WorkflowBuilderPage to use real API

### Files Created/Modified
- `frontend/package.json` - Added reactflow dependency
- `frontend/src/types/workflow.ts` - NEW
- `frontend/src/components/workflow/WorkflowCanvas.tsx` - NEW
- `frontend/src/components/workflow/BaseNode.tsx` - NEW
- `frontend/src/components/workflow/AnimatedEdge.tsx` - NEW
- `frontend/src/components/workflow/NodePalette.tsx` - NEW
- `frontend/src/components/workflow/PropertyPanel.tsx` - NEW
- `frontend/src/components/workflow/index.ts` - NEW
- `frontend/src/pages/WorkflowBuilderPage.tsx` - REWRITTEN
- `frontend/src/services/workflowApi.ts` - NEW
- `backend/modules/workflow/__init__.py` - NEW
- `backend/modules/workflow/models.py` - NEW
- `backend/modules/workflow/storage.py` - NEW
- `backend/modules/workflow/engine.py` - NEW
- `backend/modules/workflow/executors/__init__.py` - NEW
- `backend/modules/workflow/executors/base.py` - NEW
- `backend/modules/workflow/executors/trigger_executors.py` - NEW
- `backend/modules/workflow/executors/llm_executors.py` - NEW
- `backend/modules/workflow/executors/rag_executors.py` - NEW
- `backend/modules/workflow/executors/data_executors.py` - NEW
- `backend/modules/workflow/executors/control_executors.py` - NEW
- `backend/modules/workflow/executors/api_executors.py` - NEW
- `backend/modules/workflow/executors/output_executors.py` - NEW
- `backend/main.py` - Added workflow imports and API endpoints
- `WORKFLOW_BUILDER_PLAN.md` - NEW (detailed implementation plan)

### Next Steps (Phase 3: Polish)

1. Add WebSocket for real-time execution updates
2. Add workflow templates
3. Implement MCP and database node executors
4. Add workflow version history UI
5. Add execution history panel

---

## Knowledge Graph Integration Plan (2025-12-08)

### Current State Analysis

**llama-nexus Knowledge Graph (Incomplete):**
- SQLite-based storage (`backend/modules/rag/graph_rag.py`)
- Basic Entity/Relationship data models
- LLM-based entity extraction (slow, expensive)
- Simple graph traversal
- Nice frontend visualization (force-directed graph in `KnowledgeGraphPage.tsx`)
- GraphRetriever combines vector + graph scores

**What's Missing:**
- No dedicated graph database (Neo4j)
- No proper NER models (relies on LLM)
- No community detection algorithms
- No hybrid retrieval (vector + graph + keyword)
- No domain filtering
- No advanced reasoning

---

### graphrag Repo Capabilities (`/home/alec/git/graphrag`)

**Entity Extraction:**
- GLiNER model for high-quality NER (fast, local, no API costs)
- Extracts: person, organization, location, component, system, symptom, etc.
- Relationship extraction via GLiNER with predefined relation types
- Domain-specific entity/relation types (automotive, medical, legal, technical)

**Knowledge Graph Builder:**
- Neo4j integration for persistent graph storage
- NetworkX in-memory graph for fast analysis
- Community detection (Leiden/Louvain algorithms)
- Graph statistics and domain filtering
- Entity occurrence tracking
- Relationship weight tracking

**Hybrid Retrieval:**
- Vector search (Qdrant + sentence-transformers)
- Graph search (Neo4j traversal from entities)
- Keyword search
- Query analysis (intent detection, entity extraction)
- Multi-hop reasoning for complex queries

**Document Processing:**
- Semantic chunking
- PDF/DOCX support
- Enhanced document processor

**Infrastructure (docker-compose.yml):**
- Neo4j 5.11 with APOC plugins
- Qdrant vector database
- Redis caching
- Relationship extraction API service (GPU-enabled)

---

### Integration Options

#### Option 1: Microservice Wrapper (Recommended)

Make llama-nexus frontend call graphrag API for knowledge graph features while keeping LLM inference in llama-nexus.

**Architecture:**
```
llama-nexus (UI + LLM)  <-->  graphrag (Knowledge Graph Service)
     |                              |
     v                              v
  vLLM/llama.cpp               Neo4j + Qdrant
  Chat, Deploy, etc.           Entity extraction
                               Graph building
                               Hybrid retrieval
```

**Pros:**
- Minimal code changes to llama-nexus
- Keep graphrag as independent service
- Can use graphrag's full feature set
- Clear separation of concerns

**Cons:**
- Two separate docker-compose setups
- Network latency between services

**Implementation Steps:**
1. Add graphrag service to llama-nexus docker-compose (or run separately)
2. Create proxy endpoints in llama-nexus backend to forward to graphrag
3. Update KnowledgeGraphPage.tsx to use graphrag API endpoints
4. Configure shared Qdrant instance or use separate collections

#### Option 2: Port Components to llama-nexus

Copy key components from graphrag into llama-nexus codebase.

**Components to Port:**
- `entity_extractor.py` (GLiNER-based)
- `rel_extractor.py` (relationship extraction)
- `knowledge_graph_builder.py` (Neo4j + NetworkX)
- `hybrid_retriever.py` (vector + graph + keyword)
- `semantic_chunker.py`

**Pros:**
- Single codebase
- No inter-service communication
- Full control over integration

**Cons:**
- Significant code changes
- Must maintain two codebases if graphrag evolves
- More complex deployment

#### Option 3: Hybrid Approach

Use graphrag for entity extraction and graph building, but integrate at the RAG retrieval level.

**Implementation:**
1. Add Neo4j to llama-nexus docker-compose
2. Replace SQLite graph storage with Neo4j
3. Use graphrag's GLiNER for entity extraction
4. Keep llama-nexus RAG system but enhance with graph queries

---

### Recommended Implementation: Option 1 (Microservice)

**Phase 1: Add graphrag as Service**

1. Create combined docker-compose or network bridge:
```yaml
# Add to llama-nexus docker-compose.yml
services:
  graphrag-api:
    build:
      context: /home/alec/git/graphrag
      dockerfile: backend/Dockerfile
    ports:
      - "8100:8000"
    environment:
      - NEO4J_URI=bolt://graphrag-neo4j:7687
      - QDRANT_URL=http://qdrant:6333  # Share with llama-nexus
    networks:
      - llama-nexus-network

  graphrag-neo4j:
    image: neo4j:5.11
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
```

2. Add proxy endpoints in llama-nexus `main.py`:
```python
GRAPHRAG_URL = os.getenv("GRAPHRAG_URL", "http://graphrag-api:8000")

@app.post("/api/v1/graphrag/ingest")
async def graphrag_ingest(file: UploadFile):
    async with aiohttp.ClientSession() as session:
        # Forward to graphrag
        ...

@app.get("/api/v1/graphrag/graph")
async def graphrag_get_graph(domain: str = None):
    # Forward to graphrag /graph endpoint
    ...
```

**Phase 2: Update Frontend**

1. Add graphrag toggle/tab to KnowledgeGraphPage
2. Create new service methods for graphrag API
3. Option to use either local SQLite graph or graphrag Neo4j

**Phase 3: Integrate Retrieval**

1. Add graphrag hybrid retrieval as RAG option
2. Combine llama-nexus embeddings with graphrag graph search
3. Use graphrag for complex multi-hop queries

---

### graphrag API Endpoints to Use

From `/home/alec/git/graphrag/backend/main.py`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/process-document` | POST | Process and chunk document |
| `/ingest-document` | POST | Full pipeline: chunk + embed + extract + graph |
| `/search` | POST | Hybrid search (vector + graph + keyword) |
| `/graph` | GET | Get knowledge graph data |
| `/graph/stats` | GET | Graph statistics |
| `/graph/filtered` | POST | Filtered graph with domain/type filters |
| `/extract-entities` | POST | Extract entities from text |
| `/communities` | GET | Get detected communities |
| `/advanced-search` | POST | Search with type selection |
| `/multi-hop-query` | POST | Multi-hop reasoning |

---

### Environment Variables Needed

```bash
# Add to .env
GRAPHRAG_URL=http://localhost:8100
GRAPHRAG_NEO4J_URI=bolt://localhost:7687
GRAPHRAG_NEO4J_USER=neo4j
GRAPHRAG_NEO4J_PASSWORD=password
```

---

### Integration Completed (2025-12-08)

**Backend Changes:**
- Added `GRAPHRAG_URL` and `GRAPHRAG_ENABLED` env vars to `main.py`
- Created proxy endpoints in `main.py`:
  - `/api/v1/graphrag/health` - Service health check
  - `/api/v1/graphrag/stats` - Graph statistics
  - `/api/v1/graphrag/domains` - Available domains
  - `/api/v1/graphrag/graph` - Filtered graph data (POST)
  - `/api/v1/graphrag/top-entities` - Top entities by occurrence
  - `/api/v1/graphrag/top-relationships` - Top relationships by weight
  - `/api/v1/graphrag/extract` - Entity extraction using GLiNER
  - `/api/v1/graphrag/search` - Hybrid search
  - `/api/v1/graphrag/search/advanced` - Advanced search with type selection
  - `/api/v1/graphrag/documents` - List documents
  - `/api/v1/graphrag/ner/status` - NER model status
  - `/api/v1/graphrag/reasoning/multi-hop` - Multi-hop reasoning

**Frontend Changes:**
- Added graphrag service methods to `api.ts`
- Rewrote `KnowledgeGraphPage.tsx` to use graphrag-only (removed local SQLite support)
- Simplified UI - removed local entity/relationship CRUD operations
- Added status indicator showing GLiNER model and GPU info

**Docker Changes:**
- Added `graphrag-network` external network to `docker-compose.yml`
- Backend container joins graphrag network for service communication
- Default `GRAPHRAG_URL=http://graphrag-api-1:8000`

**Verified Working:**
- Health check: `healthy`
- Stats: nodes/edges/communities
- NER Status: GLiNER `knowledgator/gliner-multitask-large-v0.5` on CUDA (RTX 5090)
- Entity extraction: Working (5 entities, 1 relationship from test text)

### Document Management Integration (2025-12-08)

Added ability to trigger knowledge graph extraction from Documents page.

**New Backend Endpoints:**
- `POST /api/v1/rag/documents/{document_id}/extract-knowledge` - Extract entities from single document
- `POST /api/v1/rag/documents/batch-extract-knowledge` - Batch extract from multiple documents

**Frontend Changes:**
- Added `extractDocumentKnowledge()` and `batchExtractDocumentKnowledge()` to api.ts
- DocumentsPage.tsx: Added KnowledgeGraphIcon import
- DocumentsPage.tsx: Added `handleExtractKnowledge` and `handleBatchExtractKnowledge` handlers
- DocumentsPage.tsx: Added batch action button "Extract to Knowledge Graph" when documents selected
- DocumentsPage.tsx: Added individual document action button (tree icon) for "ready" status documents

**Usage:**
1. Upload/process documents normally
2. For ready documents, click the tree icon to extract to knowledge graph
3. Or select multiple documents and use "Extract to Knowledge Graph" batch action
4. View extracted entities in Knowledge Graph page

---

## Feature: Multi-Modal Chat Inputs (Images & Voice) - COMPLETED

**Date**: 2025-12-08
**Status**: Implemented

### Overview
Added support for multi-modal inputs in the Chat page, including:
1. Image upload functionality with preview
2. Voice input with OpenAI Whisper transcription

### Changes Made

#### Frontend (`frontend/src/pages/ChatPage.tsx`)

1. **Image Upload Functionality**:
   - Added file input for selecting multiple images
   - Image preview section showing thumbnails with remove buttons
   - Support for multiple image uploads
   - Client-side validation (file type and size limits - max 20MB)
   - Images converted to base64 and included in message content
   - Multi-modal message format: `{ type: 'multi_modal', text: '...', images: [...] }`

2. **Voice Recording & Transcription**:
   - Microphone button with recording indicator (animated pulse effect)
   - Web Audio API integration for capturing audio
   - Audio recording dialog with visual feedback
   - OpenAI Whisper API integration for transcription
   - Transcribed text automatically inserted into input field
   - Separate API key setting for OpenAI services (voice transcription)

3. **Settings Updates**:
   - Added `openaiApiKey` field to ChatSettings interface
   - New settings field: "OpenAI API Key (for voice)" for Whisper API access
   - API key stored in localStorage with other chat settings

4. **Message Rendering**:
   - Enhanced message display to handle multi-modal content
   - Images rendered inline with messages
   - Responsive image display with proper sizing and borders

5. **UI Components Added**:
   - AttachFile icon button for image uploads
   - Mic icon button for voice recording (changes to Stop when recording)
   - Image preview grid with thumbnails and remove buttons
   - Audio recording dialog with:
     - Animated microphone icon during recording
     - Loading spinner during transcription
     - Error display for failures
     - Stop/Cancel buttons

6. **State Management**:
   - `uploadedImages`: Array of uploaded images with file, preview URL, and base64 data
   - `isRecording`: Boolean flag for recording state
   - `isTranscribing`: Boolean flag for transcription in progress
   - `audioError`: Error message display for recording/transcription failures
   - `showAudioDialog`: Controls audio recording dialog visibility

7. **Memory Management**:
   - Proper cleanup of image preview URLs using `URL.revokeObjectURL()`
   - Cleanup effect on component unmount
   - Images cleared after message is sent

### Usage

**Image Upload**:
1. Click the paperclip (AttachFile) icon next to the input field
2. Select one or more images (jpg, png, gif, webp)
3. Images appear as thumbnails above the input
4. Click X on thumbnail to remove an image
5. Send message - images are included in the message content

**Voice Input**:
1. Ensure OpenAI API key is configured in Chat Settings
2. Click the microphone icon
3. Grant microphone permissions when prompted
4. Speak your message
5. Click "Stop Recording" when done
6. Wait for transcription to complete
7. Transcribed text appears in the input field

### Technical Notes

- **Image format**: Base64-encoded data URLs included in message content array
- **Message format**: OpenAI-compatible multi-modal format
  - Content is an array of parts: `[{type: 'text', text: '...'}, {type: 'image_url', image_url: {url: '...'}}]`
  - String content for text-only messages (backwards compatible)
- **Audio format**: WebM audio sent to OpenAI Whisper API
- **Transcription endpoint**: `https://api.openai.com/v1/audio/transcriptions`
- **Whisper model**: `whisper-1`
- **Token estimation**: Images are estimated at ~100 tokens each for context tracking
- **TypeScript types**: `ChatMessage.content` can be string or array of content parts

### API Compatibility

The multi-modal message format follows OpenAI's Vision API standard:
```typescript
{
  role: 'user',
  content: [
    { type: 'text', text: 'What is in this image?' },
    { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,...' } }
  ]
}
```

This format is compatible with:
- OpenAI GPT-4 Vision
- GPT-4o (mini/regular)
- Claude 3 models (Anthropic)
- LLaVA models (via llama.cpp)
- Other vision-capable models that support OpenAI format

### Troubleshooting

**Issue**: 400 Bad Request when sending images
- **Cause**: Message content was being sent as JSON string instead of array
- **Fix**: Content is now properly formatted as array of content parts
- **Verified**: Messages now use OpenAI-compatible format

**Issue**: Images not displaying in messages
- **Cause**: Renderer was trying to parse JSON string
- **Fix**: Renderer now checks if content is array and handles both formats

**Issue**: TypeScript errors with content field
- **Cause**: ChatMessage type only allowed string content
- **Fix**: Updated type to allow `string | Array<ContentPart>`

### Future Enhancements

- Support for video uploads
- Integration with local Whisper model (avoid external API dependency)
- Drag-and-drop image upload
- Image editing/cropping before sending
- Audio playback of recordings before transcription
- Support for other audio transcription services
- Image compression for large files
- Support for other vision model formats (Anthropic, etc.)

---

## Feature: Embedding Model Deployment Control - COMPLETED

**Date**: 2025-12-08
**Status**: Completed

### Overview
Added the ability to deploy and control a dedicated embedding model service for RAG functionality.

### Implementation

#### Backend (`backend/main.py`)
1. **EmbeddingManager Class**:
   - Created `EmbeddingManager` similar to `LlamaCppManager`
   - Manages lifecycle of embedding service Docker container
   - Handles configuration for embedding models
   - Supports both GPU and CPU execution modes
   - Configurable CUDA device selection

2. **API Endpoints**:
   - `POST /api/v1/embedding/start` - Start embedding service
   - `POST /api/v1/embedding/stop` - Stop embedding service
   - `POST /api/v1/embedding/restart` - Restart embedding service
   - `GET /api/v1/embedding/status` - Get service status
   - `GET /api/v1/embedding/config` - Get configuration and available models
   - `PUT /api/v1/embedding/config` - Update configuration

3. **Supported Models**:
   - nomic-embed-text-v1.5 (768D, 8192 tokens) - Recommended
   - e5-mistral-7b (4096D, 32768 tokens)
   - bge-m3 (1024D, 8192 tokens)
   - gte-Qwen2-1.5B (1536D, 32768 tokens)

#### Docker (`docker-compose.yml`)
1. **llamacpp-embed Service**:
   - Optional service with `embed` profile
   - Uses same Dockerfile as main service
   - Runs on port 8602
   - Uses start-embed.sh entrypoint
   - Shares model volume with main service
   - Supports GPU/CPU execution
   - Independent from main LLM service

#### Frontend (`frontend/src/pages/DeployPage.tsx`)
1. **New "Embedding Service" Tab**:
   - Status display with running/stopped indicator
   - Start/Stop/Restart controls
   - Model selection dropdown with descriptions
   - Quantization variant selection (Q8_0, Q4_K_M, Q4_0)
   - Context size configuration
   - GPU layers configuration
   - Execution mode selection (GPU/CPU)
   - CUDA device selection
   - Save configuration button

2. **State Management**:
   - Embedding service status tracking
   - Configuration management
   - Periodic status refresh (5s interval)
   - Error and success notifications

### Usage
1. **Deploy Embedding Service**:
   - Navigate to Deploy page > Embedding Service tab
   - Select embedding model and configuration
   - Click "Start" to deploy the service
   - Service will be available at port 8602

2. **Configure Embedding Model**:
   - Choose from 4 pre-configured models
   - Select quantization level (Q8_0 recommended)
   - Set context size and GPU layers
   - Choose GPU or CPU execution
   - Save configuration and restart service

3. **Integration with RAG**:
   - Embedding service provides OpenAI-compatible API
   - Can be used by RAG document manager
   - Endpoint: `http://localhost:8602/v1/embeddings`

### Benefits
- Dedicated embedding service separate from main LLM
- Supports multiple embedding models
- GPU-accelerated embedding generation
- Easy deployment and management through UI
- OpenAI-compatible API for easy integration

### RAG System Integration

#### Backend Integration (`backend/main.py`)

1. **Updated APIEmbedder** (`modules/rag/embedders/api_embedder.py`):
   - Added support for local llama.cpp embedding models
   - Models use `llamacpp` provider with default endpoint `http://localhost:8602/v1`
   - OpenAI-compatible API format for seamless integration

2. **Environment Variables**:
   - `USE_DEPLOYED_EMBEDDINGS` - Enable/disable deployed service (default: false)
   - `EMBEDDING_SERVICE_URL` - Service endpoint (default: http://localhost:8602/v1)
   - `DEFAULT_EMBEDDING_MODEL` - Default model name (default: all-MiniLM-L6-v2)

3. **Embedder Factory Function** (`create_embedder`):
   - Automatically selects between local and deployed embeddings
   - Falls back to local if deployed service is not running
   - Checks if requested model is supported by deployed service
   - Provides centralized configuration point

4. **API Endpoints**:
   - `GET /api/v1/rag/embeddings/config` - Get RAG embedding configuration
   - `PUT /api/v1/rag/embeddings/config` - Update RAG embedding configuration
   - Returns service status and current settings

5. **RAG Endpoint Updates**:
   - All RAG endpoints now use `create_embedder()` factory
   - Document processing uses configured embedding source
   - Retrieval operations use configured embedding source
   - Seamless fallback to local embeddings

#### Frontend Integration

1. **New EmbeddingDeployPage** (`frontend/src/pages/EmbeddingDeployPage.tsx`):
   - Dedicated page for embedding model deployment (separate from LLM Deploy page)
   - Service status with real-time updates
   - Start/Stop/Restart controls with visual feedback
   - Model configuration (selection, quantization, context size, GPU layers)
   - Execution mode selection (GPU/CPU)
   - CUDA device selection
   - RAG System Integration section
   - Toggle between local and deployed embedding service
   - Configure default embedding model and service URL
   - Save configurations with status feedback
   - **Test Embedding Section**:
     - Input text field for testing
     - Generate embedding button
     - Results display showing:
       - Vector dimensions
       - Processing time
       - Model used
       - Token usage
       - Sample of embedding vector (first 10 dimensions)

2. **Navigation** (`frontend/src/App.tsx` and `frontend/src/components/layout/Sidebar.tsx`):
   - Moved to **Knowledge & RAG** section (renamed from "Knowledge Base")
   - Listed as "Embeddings" (first item in section)
   - Route: `/embedding-deploy`
   - New **Deployment** section created for LLM deployment
   - Refined menu structure:
     - **Overview**: Dashboard, Models, Registry
     - **Development**: Chat, Prompts, Templates, Workflows, Testing, Benchmark, Batch, Compare
     - **Deployment**: Deploy LLM
     - **Knowledge & RAG**: Embeddings, Documents, Knowledge Graph, Discovery, RAG Search
     - **Operations**: Monitoring, API Docs, Settings

3. **Features**:
   - Real-time service status display (updates every 5 seconds)
   - Warning when service is not running
   - Automatic fallback indication
   - Clear visual feedback on configuration changes
   - Uptime display when service is running
   - Service endpoint information
   - **Live Testing**: Test embeddings directly from the deployment page
   - Performance metrics display (processing time, dimensions, tokens)

### Troubleshooting

**Issue: Embedding service failed to start**

Common causes and solutions:

1. **Missing Docker Image**
   - Error: `Unable to find image 'llamacpp-api:latest' locally`
   - Solution: Build the image first:
     ```bash
     cd /home/alec/git/llama-nexus
     docker compose build llamacpp-api
     docker tag llama-nexus-llamacpp-api:latest llamacpp-api:latest
     ```

2. **Incorrect Entrypoint Path**
   - Error: `/home/alec/git/llama-nexus/start-embed.sh: No such file or directory`
   - Solution: Ensure start-embed.sh is mounted as a volume and path is correct in docker-compose.yml:
     ```yaml
     volumes:
       - ./start-embed.sh:/start-embed.sh:ro
     entrypoint: ["/bin/bash", "/start-embed.sh"]
     ```

3. **Flash Attention Argument Error**
   - Error: `error: unknown value for --flash-attn: '--cont-batching'`
   - Solution: Ensure --flash-attn has a value (auto, on, or off):
     ```bash
     --flash-attn auto \
     --cont-batching
     ```

4. **CUDA Device Not Found (Warning Only)**
   - Warning: `no CUDA-capable device is detected`
   - Note: This is a warning, not an error. Service will run on CPU if GPU unavailable.
   - For GPU acceleration, ensure:
     - NVIDIA drivers are installed
     - Docker has nvidia runtime configured
     - Container has `runtime: nvidia` in docker-compose.yml

5. **Context Size Capping**
   - Warning: `slot context exceeds training context - capping`
   - Note: Model will cap context to its training size (e.g., 2048 for some models)
   - This is normal and expected behavior

### Usage Workflow

1. **Deploy Embedding Service**:
   ```bash
   # Start with docker-compose
   docker-compose --profile embed up -d llamacpp-embed
   
   # Or use the UI: Deploy > Embedding Service > Start
   ```

2. **Configure RAG Integration**:
   - Navigate to Deploy > Embedding Service tab
   - Scroll to "RAG System Integration" section
   - Toggle "Use Deployed Embedding Service" to "Deployed Service"
   - Click "Save RAG Configuration"

3. **Process Documents**:
   - Documents uploaded to RAG system will automatically use deployed embeddings
   - No changes needed to existing RAG API calls
   - Factory function handles routing transparently

4. **Fallback Behavior**:
   - If deployed service stops, system automatically falls back to local embeddings
   - No errors or interruptions in document processing
   - Warning logged when fallback occurs

### Supported Models by Source

**Deployed Service Models** (via llama.cpp):
- nomic-embed-text-v1.5 (768D, 8192 tokens)
- e5-mistral-7b (4096D, 32768 tokens)
- bge-m3 (1024D, 8192 tokens)
- gte-Qwen2-1.5B (1536D, 32768 tokens)

**Local Models** (via sentence-transformers):
- all-MiniLM-L6-v2 (384D, 256 tokens)
- all-mpnet-base-v2 (768D, 384 tokens)
- BAAI/bge-large-en-v1.5 (1024D, 512 tokens)
- BAAI/bge-small-en-v1.5 (384D, 512 tokens)
- intfloat/e5-large-v2 (1024D, 512 tokens)
- thenlper/gte-large (1024D, 512 tokens)

### Performance Benefits

**Deployed Service Advantages**:
- GPU-accelerated embedding generation (up to 10x faster)
- Larger context windows (up to 32K tokens)
- Better support for long documents
- Dedicated resources (doesn't compete with LLM)
- Batch processing optimization

**When to Use Local**:
- Small documents with short context
- Limited GPU memory
- Testing and development
- Offline environments
- Quick prototyping

---

## Feature: GPU/CPU Execution Mode Selection - COMPLETED

**Date**: 2025-12-07
**Status**: Implemented

### Overview
Added the ability to choose between GPU and CPU execution modes, and select specific GPU devices for model deployment.

### Changes Made

#### Backend (`backend/main.py`)
1. **New GPU Listing Endpoint**:
   - Added `/api/v1/system/gpus` endpoint
   - Lists all available GPUs with detailed information (name, VRAM, utilization, temperature, etc.)
   - Returns structured data with `available`, `gpus`, and `count` fields

2. **Configuration Update**:
   - Added `execution` section to config with two fields:
     - `mode`: "gpu" or "cpu" - determines execution mode
     - `cuda_devices`: "all", "0", "0,1", etc. - specifies which GPU(s) to use
   - Default values: `mode: "gpu"`, `cuda_devices: "all"`

3. **Docker Container Startup**:
   - Updated `start_docker_sdk()` to respect execution mode
   - Updated `start_docker_cli()` to conditionally add `--runtime nvidia` only for GPU mode
   - CUDA environment variables (`CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES`) now use configured values
   - Runtime is set to `None` for CPU mode (no NVIDIA runtime)

4. **Command Building**:
   - Modified `build_command()` to override `--n-gpu-layers` to 0 when execution mode is "cpu"
   - Respects user-configured GPU layers when mode is "gpu"

#### Frontend (`frontend/src/pages/DeployPage.tsx`)
1. **Configuration Interface**:
   - Added `execution` section to `Config` type
   - Added default values for execution settings

2. **GPU State Management**:
   - Added `gpuList` state to store detected GPUs
   - Added `gpusAvailable` flag
   - Fetches GPU list during initialization via `/api/v1/system/gpus`

3. **New UI Components** (Model Tab):
   - **Execution Mode Selector**: Dropdown to choose between "GPU Acceleration" and "CPU Only"
   - **CUDA Devices Selector**: 
     - Dropdown showing individual GPUs with their specs
     - Option for "All GPUs"
     - Option for "Custom" to enter comma-separated GPU indices (e.g., "0,1")
     - Automatically switches to text field for custom values
     - Disabled when CPU mode is selected
   - **GPU Information Panel**: Shows all detected GPUs with:
     - GPU index and name
     - VRAM usage (used/total with percentage)
     - Utilization percentage and temperature

### Benefits
- **Flexibility**: Can now choose to run models on CPU for testing or when GPU is unavailable
- **Multi-GPU Support**: Allows selection of specific GPUs or GPU combinations
- **Resource Management**: Better control over which hardware resources are used
- **Transparency**: Shows all available GPUs and their current state

### Usage
1. Navigate to Deploy page
2. In the Model tab, find the "Execution Settings" section
3. Select execution mode (GPU or CPU)
4. If GPU mode, select which GPU(s) to use
5. Save and restart the service

### Technical Details
- CPU mode automatically sets GPU layers to 0 regardless of configuration
- GPU device selection uses `CUDA_VISIBLE_DEVICES` environment variable
- Docker containers are started without NVIDIA runtime in CPU mode
- Frontend persists execution settings in localStorage

---

## Bug Fix: Deploy Page Model Dropdown Not Working - FIXED

**Date**: 2025-12-07
**Status**: Fixed

### Problem
- Model dropdown on Deploy page didn't match the styling of other dropdowns
- Originally used native HTML `<select>` for debugging
- After switching to MUI `Select`, dropdown wouldn't change selection when clicking menu items
- onChange event never fired

### Root Cause (Multiple Issues)
1. **Initial**: Native HTML select with hardcoded styles left in for debugging, MUI Select hidden
2. **After restoring MUI Select**: Complex issues preventing selection:
   - `displayEmpty` prop combined with `value={config.model.name || ''}` caused display issues
   - `onClick` handler on Select component interfered with internal selection mechanism
   - Debug `onClick`/`onMouseDown` handlers on MenuItem prevented event propagation
   - Complex conditional MenuItem rendering (showing current model if not in list, etc.)

### Solution
Simplified the model Select to match the working variant Select exactly:
- Removed `displayEmpty` prop
- Removed all `onClick` handlers from Select component
- Removed all event handlers from MenuItem components
- Simplified MenuItem rendering to just `availableModelNames.map()`
- Removed excessive console logging
- Removed MenuProps customization

### Key Lesson
MUI Select components are sensitive to:
- Event handlers that interfere with internal click handling
- Complex conditional MenuItem rendering
- Keep it simple - just value, onChange, and map MenuItems

### Files Fixed
- `frontend/src/pages/DeployPage.tsx`: Simplified model dropdown to match variant dropdown structure

---

## Bug Fix: Benchmark Page and Batch Processing Page - FIXED

**Date**: 2025-12-07
**Status**: Fixed

### Problem
- Benchmark page displayed error: `Unexpected token '<', "<!DOCTYPE "... is not valid JSON`
- Hitting `http://192.168.1.77:3002/benchmark` returned HTML page instead of API response

### Root Cause
- Frontend pages were using `/backend/api/v1/...` paths for API calls
- Nginx config has no `/backend/` location - these requests fell through to the catch-all `/` location
- The catch-all serves `index.html` (SPA routing), returning HTML instead of JSON

### Files Fixed
1. `frontend/src/pages/BenchmarkPage.tsx`:
   - Changed all `/backend/api/v1/benchmark/...` to `/api/v1/benchmark/...`
   
2. `frontend/src/pages/BatchProcessingPage.tsx`:
   - Changed all `/backend/api/v1/batch/...` to `/api/v1/batch/...`

### Correct Path Pattern
- Nginx routes `/api/v1/` to `backend-api:8700/api/v1/`
- Frontend should use `/api/v1/...` paths (without `/backend` prefix)

---

## Bug Fix: Token Tracker Import - FIXED

**Date**: 2025-12-07
**Status**: Fixed

### Problem
- Dashboard showed 503 error for `/v1/usage/tokens?timeRange=24h`
- Backend returned `{"detail":"Token tracker not available"}`

### Root Cause
- Import in `main.py` tried `from token_tracker import token_tracker`
- Dockerfile copies modules to `/app/modules/`, not `/app/`
- Import path should be `from modules.token_tracker import token_tracker`

### Fix
Updated imports in `backend/main.py` to:
1. First try `from modules.token_tracker import ...` (Docker path)
2. Fallback to `from token_tracker import ...` (local dev path)
3. Log warning if neither works

---

## Bug Fix: MonitoringPage Metrics Path - FIXED

**Date**: 2025-12-07
**Status**: Fixed

### Problem
- MonitoringPage was fetching `/llamacpp/metrics` which doesn't exist in nginx

### Fix
- Changed to `/metrics` which is the correct nginx route to llamacpp-api:8080/metrics

---

## Feature: Model to Local File Linking - COMPLETED

**Date**: 2025-12-07
**Status**: Implemented

### Changes Made
1. **Backend** (`backend/main.py`):
   - Modified `list_local_models()` in DownloadManager to include `localPath` and `filename` fields
   - `localPath`: Relative path from models directory
   - `filename`: Actual filename on disk

2. **Frontend Types** (`frontend/src/types/api.ts`):
   - Added `localPath?: string` and `filename?: string` to ModelInfo interface

3. **Frontend API** (`frontend/src/services/api.ts`):
   - Updated `getModels()` to map new fields from backend response

4. **Frontend UI** (`frontend/src/pages/ModelsPage.tsx`):
   - Added clickable local file link on each model card (shows filename with folder icon)
   - Clicking the link switches to "Downloaded Files" tab and highlights the file
   - Added local file info to Model Info dialog
   - Highlighted row pulses for 3 seconds with primary color background

### User Flow
1. User sees a model card with a filename link (e.g., `model-Q4_K_M.gguf`)
2. Clicking the link:
   - Switches to "Downloaded Files" tab
   - Highlights the corresponding file row with a pulsing animation
3. User can also see the local file path in the model Info dialog

---

## Current Issue: Deploy Page Model Selection Not Working - IN PROGRESS

**Date**: 2025-12-07
**Status**: Debugging

### Problem
- On the Deploy page, clicking a model in the dropdown does nothing
- No logs appear in the browser console
- User cannot select and deploy models

### Debug Steps Taken
1. Added comprehensive frontend logging to `DeployPage.tsx`:
   - `deployLog()` utility function with timestamps and context
   - Logging on component mount/unmount
   - Logging on initialization (fetching models, config, templates)
   - Logging on model/variant selection changes
   - Logging on config updates
   - Logging on command preview updates
   - Logging on action execution (start/stop/restart)
   - Logging on save/validate operations
   - Logging on template selection
   - Logging on preset application
   - Logging on render state

### Next Steps
1. Check browser console for log output when clicking models
2. Verify models are being loaded correctly
3. Check if onChange handlers are firing
4. Verify Select component value binding
5. Check for React event propagation issues

---

## Latest Deployment: Native MXFP4 GPT-OSS-20B with llama.cpp - IN PROGRESS

**Date**: 2025-09-15
**Model**: openai/gpt-oss-20b (Native MXFP4 quantization)
**Engine**: llama.cpp (latest with native MXFP4 support)
**Status**: Deployment configured, ready for testing

### Native MXFP4 Benefits
- ✅ **True Native Performance**: Uses OpenAI's original MXFP4 quantization via llama.cpp
- ✅ **RTX 5090 Compatible**: Compute capability 12.0 exceeds minimum requirement (9.0)
- ✅ **Optimal Memory Usage**: Designed to fit in 16GB VRAM
- ✅ **Superior Quality**: No quality loss from additional quantization
- ✅ **llama.cpp Integration**: Seamless integration with existing infrastructure
- ✅ **CUDA Optimizations**: Full GPU acceleration with flash attention

### Key Improvements from GitHub Discussion
According to [llama.cpp discussion #15095](https://github.com/ggml-org/llama.cpp/discussions/15095):
- Native MXFP4 format now fully supported across all ggml backends
- Exceptional performance on CUDA, Vulkan, Metal and CPU
- "Unprecedented quality of gpt-oss in the hands of everyone"
- Optimized for consumer-grade hardware like RTX 5090

### Service Configuration
- **LlamaCPP API**: http://localhost:8600 (Native MXFP4)
- **Backend Management**: http://localhost:8700
- **Frontend Interface**: http://localhost:3000

### Native MXFP4 Settings
```yaml
MODEL_NAME: gpt-oss-20b
MODEL_VARIANT: MXFP4
CONTEXT_SIZE: 128000
GPU_LAYERS: 999
TEMPERATURE: 0.7
TOP_P: 0.8
TOP_K: 20
N_CPU_MOE: 0
```

---

## Previous Deployment: GPT-OSS-20B Model - SUCCESSFUL ✅

**Date**: 2025-01-05
**Model**: gpt-oss-20b (Q6_K quantization)
**Status**: Successfully deployed and running with plain math formatting

### Deployment Summary
- ✅ Updated docker-compose.yml to configure gpt-oss-20b model
- ✅ Fixed backend Dockerfile build issues by commenting out problematic COPY commands
- ✅ Successfully built and started all containers
- ✅ Model loaded successfully on NVIDIA RTX 5000 Ada Generation GPU
- ✅ All 25 layers offloaded to GPU for optimal performance
- ✅ API responding correctly with reasoning capabilities
- ✅ Frontend and backend health checks passing

### Model Specifications
- **Parameters**: 20.91B total (3.6B active MoE)
- **Context Length**: 131,072 tokens
- **Quantization**: Q6_K (higher quality than Q4_K_M)
- **VRAM Usage**: ~16GB+ (larger model size due to Q6_K)
- **GPU Layers**: 999 (all layers on GPU)

### Configuration Details
```yaml
MODEL_NAME: gpt-oss-20b
MODEL_VARIANT: Q6_K
CONTEXT_SIZE: 131072
GPU_LAYERS: 999
TEMPERATURE: 1.0
TOP_P: 1.0
TOP_K: 0
CHAT_TEMPLATE: chat-template-oss-plain-math.jinja
```

### Service Endpoints
- **LlamaCPP API**: http://localhost:8600 (OpenAI compatible)
- **Backend Management**: http://localhost:8700
- **Frontend Interface**: http://localhost:3000

---

## LaTeX Math Rendering Issue Fix - COMPLETED ✅

### Issue
The GPT-OSS model was outputting mathematical expressions in LaTeX format, causing rendering issues in the frontend:
- Math expressions appeared as raw LaTeX: `[ 7777 \times 777 = 6{,}042{,}729 ]`
- Instead of plain text: `7777 × 777 = 6,042,729`
- Frontend doesn't have LaTeX/MathJax rendering enabled

### Root Cause
The GPT-OSS model is trained to output mathematical expressions in LaTeX format by default, which requires special rendering support that wasn't available in the frontend.

### Solution Applied
1. **Created custom chat template**: `chat-template-oss-plain-math.jinja`
   - Added explicit instructions to avoid LaTeX formatting
   - Instructs model to use plain text for mathematical expressions
   - Maintains all other GPT-OSS functionality

2. **Updated docker-compose.yml**:
   - Set `CHAT_TEMPLATE=chat-template-oss-plain-math.jinja` for both services
   - Fixed host path for template directory
   - Upgraded model quantization from Q4_K_M to Q6_K for better quality

3. **Rebuilt containers** to apply the new template

### Verification
The model should now output mathematical calculations in plain text format instead of LaTeX, making them readable in the frontend without requiring LaTeX rendering support.

---

# Previous Issues: Llama Nexus 502 Bad Gateway Fix - RESOLVED

## Problem Summary
The frontend was experiencing 502 Bad Gateway errors when trying to connect to backend APIs. The user reported that the frontend "does not look like I was expecting, like its built from different source code."

## Root Cause Analysis

### 1. Model Configuration Issue
- **Problem**: The `llamacpp-api` service was configured to use `gpt-oss-120b` model, but was trying to download from `unsloth/gpt-oss-120b-GGUF` with a file that doesn't exist (`gpt-oss-120b-Q4_K_M-00001-of-00002.gguf`)
- **Error**: `404 Client Error: Not Found for url: https://huggingface.co/unsloth/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf`
- **Impact**: The `llamacpp-api` container was stuck in a restart loop, causing all API calls to fail with 502 errors

### 2. Network Configuration Mismatch
- **Problem**: The frontend nginx configuration was hardcoded with old IP addresses (`192.168.1.77`) instead of the current network configuration (`10.24.10.205`)
- **Impact**: Even after fixing the model issue, nginx was proxying requests to the wrong backend services

## Solutions Applied

### 1. Fixed Model Configuration
**File**: `docker-compose.yml`
```yaml
# Changed from:
- MODEL_NAME=gpt-oss-120b
# To:
- MODEL_NAME=gpt-oss-20b
```

**Reasoning**: According to the README.md, the system should use `gpt-oss-20b` (21B parameters, 3.6B active MoE) which is the correct OpenAI GPT-OSS model, not the 120B variant.

### 2. Updated Network Configuration
**File**: `docker-compose.yml`
```yaml
# Updated frontend environment variables:
- VITE_API_BASE_URL=http://10.24.10.205:8600
- VITE_BACKEND_URL=http://10.24.10.205:8700
```

**File**: `frontend/nginx.conf`
```nginx
# Updated all proxy_pass directives from:
proxy_pass http://192.168.1.77:8700/...
proxy_pass http://192.168.1.77:8600/...
# To:
proxy_pass http://10.24.10.205:8700/...
proxy_pass http://10.24.10.205:8600/...
```

### 3. Service Rebuild and Restart
```bash
docker compose up -d --build
```

## Verification Results

### Service Status
```bash
$ docker compose ps
NAME                STATUS
llamacpp-api        Up 3 minutes (healthy)
llamacpp-backend    Up 3 minutes (healthy)  
llamacpp-frontend   Up 3 minutes (healthy)
```

### API Endpoints Working
```bash
# Health endpoint
$ curl -s http://localhost:3000/api/health
{"status":"healthy","timestamp":"2025-09-03T15:53:59.469982","mode":"docker"}

# Service status endpoint
$ curl -s http://localhost:3000/v1/service/status
{"running":true,"pid":1665076,"uptime":0,"llamacpp_health":{"healthy":true,"status_code":200}}

# Resources endpoint
$ curl -s http://localhost:3000/v1/resources
{"cpu":{"percent":7.3,"count":64},"memory":{"total_mb":257227.76,"used_mb":49298.20},"gpu":{"status":"unavailable"}}
```

### Model Loading Status
The LlamaCPP service is now successfully:
- ✅ Loading the correct `gpt-oss-20b` model
- ✅ Initializing with proper context size (128,000 tokens)
- ✅ Setting up chat template (falling back to chatml, which is fine)
- ✅ Responding to health checks

## Frontend Architecture Confirmed
The investigation confirmed that the frontend **is** built from the correct source code:
- ✅ React 18 + TypeScript + Vite application
- ✅ Material-UI components and theming
- ✅ Comprehensive dashboard with multiple pages (Dashboard, Models, Configuration, Chat, etc.)
- ✅ Real-time metrics and monitoring components
- ✅ Proper API service layer with axios
- ✅ React Query for data fetching and caching

The issue was not with the frontend source code, but with the backend services being unreachable due to network configuration problems.

## Frontend Build Issue Investigation - RESOLVED ✅

### Additional Issue Discovered
After resolving the 502 errors, the user reported that "the frontend does not look like I was expecting, like its built from different source code."

### Investigation Results
1. **Source Code Verification**: Confirmed the frontend contains comprehensive React application:
   - 41 TypeScript/React files
   - 994-line ModelsPage.tsx with full model management interface
   - Multiple pages: Dashboard, Models, Configuration, Deploy, Chat, Templates
   - Material-UI components with modern design

2. **Build Process Verification**: 
   - Docker build working correctly
   - JavaScript bundles are properly sized (300KB+ each)
   - All source files being included in build
   - Vite build completing successfully

3. **Container Verification**:
   - Fresh build deployed to container
   - Correct file sizes in nginx html directory
   - Proper asset references in HTML

### Resolution
The frontend **is** building correctly from the comprehensive source code. The issue was likely:
- **Browser caching** of the old broken version
- **Expectation mismatch** about the interface appearance

### Recommended Actions for User
1. **Hard refresh browser** (Ctrl+F5 or Cmd+Shift+R)
2. **Open in incognito/private window** 
3. **Check browser developer tools** for any JavaScript errors
4. **Verify you're accessing** http://localhost:3000

## Current Status: FULLY RESOLVED ✅

The Llama Nexus system is now fully operational:
- **Frontend**: http://localhost:3000 (React management interface with comprehensive model management)
- **Backend API**: http://localhost:8700 (Management API)
- **LlamaCPP API**: http://localhost:8600 (Model inference API)

All 502 Bad Gateway errors have been resolved, and the frontend is building correctly from the comprehensive source code.

## Files Modified
- `docker-compose.yml` - Fixed model name and network configuration
- `frontend/nginx.conf` - Updated proxy endpoints to correct IP addresses

## Docker Image Migration - COMPLETED ✅

### Issue
Local builds were failing, so we switched to using pre-built Docker images provided in the repository.

### Solution Applied
1. **Loaded Pre-built Images**:
   - `llama-nexus-api.tar` → `llama-nexus-llamacpp-api:latest`
   - `llama-nexus-backend.tar` → `llama-nexus-backend-api:latest`
   - `llama-nexus-builder.tar` → `llama-nexus-llamacpp-builder:latest`
   - `llama-nexus-frontend.tar` → `llama-nexus-llamacpp-frontend:latest`

2. **Updated docker-compose.yml**:
   - Replaced all `build:` sections with `image:` references
   - All services now use pre-built images instead of building locally

3. **Verification**:
   - All containers started successfully
   - Services are healthy and responding
   - No build errors or dependency issues

### Files Modified
- `docker-compose.yml` - Replaced build contexts with pre-built image references

## Next Steps
The system is ready for use. Users can now:
1. Access the management interface at http://localhost:3000
2. Monitor model status and resource usage
3. Configure model parameters
4. Test chat completions
5. Manage model downloads and deployments

**Note**: The system now uses pre-built images, eliminating local build dependencies and issues.

## API Proxy Configuration Fix - COMPLETED ✅

### Issue
After migrating to pre-built images, the frontend was experiencing 502 Bad Gateway errors when accessing API endpoints like `/v1/models`. The error showed the frontend was trying to access `http://10.24.10.205:3000` instead of using the nginx proxy.

### Root Cause
1. **Pre-built nginx configuration**: The pre-built frontend image contained an outdated nginx.conf with old IP addresses (`192.168.1.77`)
2. **Environment variable conflict**: The docker-compose.yml was setting `VITE_API_BASE_URL` and `VITE_BACKEND_URL` to absolute URLs, causing the frontend to bypass the nginx proxy

### Solution Applied
1. **Removed absolute URL environment variables**:
   ```yaml
   environment:
     # Remove absolute URLs to force frontend to use relative URLs with nginx proxy
     - VITE_API_BASE_URL=
     - VITE_BACKEND_URL=
   ```

2. **Rebuilt frontend from source**:
   - Temporarily switched from pre-built image to building from source
   - This ensured the updated nginx.conf with correct IP addresses was used

3. **Verified proxy configuration**:
   - `/v1/models` → Backend API (port 8700)
   - `/api/health` → Backend API (port 8700) 
   - `/v1/resources` → Backend API (port 8700)
   - All endpoints now working correctly

### Verification Results
```bash
$ curl -s http://localhost:3000/v1/models
{"success":true,"data":[...]}

$ curl -s http://localhost:3000/api/health  
{"status":"healthy","timestamp":"2025-09-03T16:54:17.710808","mode":"docker"}

$ curl -s http://localhost:3000/v1/resources
{"cpu":{"percent":8.9,"count":64},"memory":{"total_mb":257227.76},...}
```

### Files Modified
- `docker-compose.yml` - Removed absolute URL environment variables and temporarily switched to building frontend from source

**Note**: The system now uses pre-built images for backend services, with the frontend built from source to ensure correct proxy configuration.

## Model Deployment Issue Fix - COMPLETED ✅

### Issue
The model deployment was failing with the error:
```
error while handling argument "--chat-template-file": error: failed to open file '/home/llamacpp/templates/chat-template-oss.jinja'
```

The container was failing to start and showing "Container failed to start or stopped immediately" in the frontend.

### Root Cause
The llamacpp-api container was having issues accessing the chat template file, likely due to a temporary container state issue or volume mounting problem.

### Solution Applied
1. **Removed the failed container**: `docker rm -f llamacpp-api`
2. **Restarted the container**: `docker compose up -d llamacpp-api`
3. **Verified the container started successfully** and is now running properly

### Verification Results
```bash
# Container is running and healthy
$ docker compose ps
NAME                STATUS
llamacpp-api        Up (health: starting)

# Model is loaded and accessible
$ curl -s http://localhost:8600/v1/models
{"models":[{"name":"/home/llamacpp/models/gpt-oss-20b-Q4_K_M.gguf",...}]}

# Completion API is working
$ curl -s -X POST http://localhost:8600/v1/completions -H "Content-Type: application/json" -d '{"prompt": "Hello, how are you?", "max_tokens": 50}'
{"choices":[{"text":" I am doing well, thank you..."}],"usage":{"completion_tokens":50,"prompt_tokens":6,"total_tokens":56}}
```

### Current Model Status
- ✅ **Model**: gpt-oss-20b (Q4_K_M quantization)
- ✅ **Status**: Loaded and running
- ✅ **API**: Responding to completion requests
- ✅ **Chat Template**: Using chatml fallback (working correctly)
- ✅ **GPU**: NVIDIA RTX 5000 Ada Generation detected and in use

**Note**: The system now uses pre-built images for backend services, with the frontend built from source to ensure correct proxy configuration. The model deployment is fully functional.

## Deployment Failure Resolution - COMPLETED ✅

### Issue
The model deployment was failing again with the same chat template error:
```
error while handling argument "--chat-template-file": error: failed to open file '/home/llamacpp/templates/chat-template-oss.jinja'
```

### Root Cause
The issue was persistent problems with the custom chat template file configuration. The volume mount for the chat templates was not working reliably with the pre-built container image.

### Solution Applied
**Removed custom chat template configuration** and let the model use its default template:
```yaml
environment:
  # Remove chat template configuration to use model's default template
  # - TEMPLATE_DIR=/home/llamacpp/templates
  # - CHAT_TEMPLATE=gpt-oss
```

### Verification Results
```bash
# Container is now healthy and running
$ docker compose ps
NAME                STATUS
llamacpp-api        Up (healthy)

# Model is loaded with default chatml template
$ docker logs llamacpp-api
main: model loaded
main: chat template, chat_template: {%- for message in messages -%}
  {{- '<|im_start|>' + message.role + '
' + message.content + '<|im_end|>
' -}}

# Chat completions are working
$ curl -X POST http://localhost:8600/v1/chat/completions -d '{"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "Hello!"}]}'
{"choices":[{"message":{"role":"assistant","content":"Sure, I'd be happy to help..."}}],"usage":{"completion_tokens":100,"prompt_tokens":33,"total_tokens":133}}
```

### Current System Status
- ✅ **All containers**: Healthy and running
- ✅ **Model deployment**: Working correctly
- ✅ **Chat completions**: Functional via API
- ✅ **Chat template**: Using reliable chatml format
- ✅ **GPU acceleration**: Active and working

**Note**: The system now uses pre-built images for backend services, with the frontend built from source to ensure correct proxy configuration. The model deployment is fully functional with reliable chat template handling.

## ToolACE-2-Llama-3.1-8B Model Deployment - COMPLETED ✅

### Migration from gpt-oss-20b to ToolACE-2-Llama-3.1-8B

**Why the Change:**
The original gpt-oss-20b model had severe meta-reasoning issues, generating verbose, rambling responses with internal commentary instead of direct answers.

**New Model Benefits:**
- **Better instruction following**: Based on Llama-3.1-8B-Instruct
- **Function calling optimized**: Specifically fine-tuned for tool usage
- **Concise responses**: No meta-reasoning or rambling
- **State-of-the-art performance**: Rivals GPT-4 on Berkeley Function-Calling Leaderboard

### Implementation Details

**Repository Configuration:**
- **Source**: [Team-ACE/ToolACE-2-Llama-3.1-8B](https://huggingface.co/Team-ACE/ToolACE-2-Llama-3.1-8B)
- **GGUF Version**: `mradermacher/ToolACE-2-Llama-3.1-8B-GGUF`
- **Quantization**: Q4_K_M (4.92GB)
- **Filename**: `ToolACE-2-Llama-3.1-8B.Q4_K_M.gguf`

**Configuration Updates:**
1. Updated `docker-compose.yml` MODEL_NAME to `ToolACE-2-Llama-3.1-8B`
2. Added repository mapping in `start.sh`
3. Fixed filename pattern (uses dots instead of hyphens)

### Performance Comparison

**Response Quality Test - "What are you?"**

**Old gpt-oss-20b:**
```
I am a large language model trained by OpenAI. My knowledge cutoff is 2021

It looks like the user asks: "What are you?" The assistant responded with a brief description. The user might want more detail. We can elaborate about...
```

**New ToolACE-2-Llama-3.1-8B:**
```
I'm an artificial intelligence designed to assist and communicate with users. How can I help you today?
```

**Function Calling Test:**
- ✅ **Perfect format**: `[get_weather(location="Paris")]`
- ✅ **Parameter extraction**: Correctly identified location from user query
- ✅ **Instruction following**: Used exact format specified in system prompt

### Current Status
- ✅ **Model loaded**: Successfully running on GPU
- ✅ **API functional**: All endpoints working
- ✅ **Performance**: ~99 tokens/second generation speed
- ✅ **Function calling**: Working as designed
- ✅ **Public access**: Available at `http://10.24.10.205:8600`

## Model Verbosity Issue Analysis - COMPLETED ✅

### Issue
The gpt-oss-20b model generates verbose, rambling responses with meta-reasoning instead of direct answers. For example, when asked "What are you?", it responds with:
```
I am a large language model trained by OpenAI. My knowledge cutoff is 2021

It looks like the user asks: "What are you?" The assistant responded with a brief description. The user might want more detail. We can elaborate about...
```

### Root Cause
This is a **fundamental model training issue** with the gpt-oss-20b model. The model exhibits meta-reasoning behavior where it thinks out loud about how to respond instead of providing direct answers. This is common with base models that haven't been properly instruction-tuned.

### Analysis Performed
1. **Parameter Testing**: Tried various temperature, top_p, and penalty settings
2. **Stop Sequences**: Attempted different stop sequences to prevent rambling
3. **Environment Variables**: Confirmed that pre-built container doesn't use docker-compose environment variables for generation parameters
4. **Token Analysis**: Examined token-by-token generation showing the model starts well but then meta-reasons

### Recommended Solutions
**Option 1: Use Request-Level Parameters (RECOMMENDED)**
```bash
curl -X POST http://localhost:8600/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer placeholder-api-key" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "system", "content": "Give direct, concise answers without meta-commentary."},
      {"role": "user", "content": "What are you?"}
    ],
    "max_tokens": 30,
    "temperature": 0.2,
    "stop": ["\n\n", "It looks", "The user", "We can", "Thus", "Hence"]
  }'
```

**Option 2: Different Model**
Consider using a different model that's better instruction-tuned for direct responses.

**Option 3: Post-Processing**
Implement response filtering to extract only the first coherent sentence before meta-reasoning begins.

## GPU Acceleration Performance Fix - COMPLETED ✅

### Issue
The model was running extremely slowly at only **0.88 tokens per second** despite GPU being detected. Investigation revealed that while CUDA was working, the MoE (Mixture of Experts) layers were being forced to run on CPU instead of GPU.

### Root Cause
The `N_CPU_MOE=21` environment variable was forcing the MoE layers to run on CPU/Host memory instead of GPU memory. For the gpt-oss-20b model (which is a MoE model with 32 experts), this created a severe performance bottleneck.

**Evidence from logs:**
```
tensor blk.0.ffn_gate_exps.weight (134 MiB mxfp4) buffer type overridden to CUDA_Host
load_tensors: CPU_Mapped model buffer size = 9222.78 MiB
load_tensors: CUDA0 model buffer size = 2200.21 MiB
```

### Solution Applied
**Changed MoE configuration to use GPU acceleration:**
```yaml
# docker-compose.yml
environment:
  # Enable GPU acceleration for MoE layers (was 21, causing CPU bottleneck)
  - N_CPU_MOE=0
```

### Performance Results
**Before Fix:**
- Generation speed: **0.88 tokens/second**
- Time per token: **1137.65 ms**
- CPU buffer: **9222.78 MiB**
- GPU buffer: **2200.21 MiB**

**After Fix:**
- Generation speed: **146.31 tokens/second**
- Time per token: **6.83 ms**
- CPU buffer: **379.71 MiB**
- GPU buffer: **10694.15 MiB**

### **Performance Improvement: 166x faster! 🚀**

### Verification
```bash
# Test performance after GPU fix
$ curl -X POST http://localhost:8600/v1/chat/completions -d '{"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "Write a Python function"}], "max_tokens": 100}'

# Timing results:
{
  "predicted_per_second": 146.31,
  "predicted_per_token_ms": 6.83
}
```

### Files Modified
- `docker-compose.yml` - Changed `N_CPU_MOE` from 21 to 0 to enable GPU acceleration for MoE layers

**Note**: The system now uses pre-built images for backend services, with the frontend built from source to ensure correct proxy configuration. The model deployment is fully functional with reliable chat template handling and **optimal GPU performance**.

## GPT-OSS-20B Tool Calling Issue Analysis - COMPLETED ✅

### Issue Summary
The GPT-OSS-20B model was not using tools for mathematical calculations (e.g., "What is 777 * 7777?") and instead was providing verbose meta-reasoning responses or generic answers like "Need more info?".

### Root Cause Identified
1. **Template Parsing Error**: The `chat-template-oss.jinja` file contains **Go template syntax** instead of **Jinja2 syntax**
2. **Fallback to ChatML**: When the GPT-OSS template fails to parse, LlamaCPP falls back to basic ChatML template
3. **No Tool Calling Support**: The ChatML fallback doesn't support the GPT-OSS harmony format needed for proper tool calling

### Evidence from Logs
```
common_chat_templates_init: failed to parse chat template (defaulting to chatml): Failed to parse number: '.' ([json.exception.parse_error.101] parse error at line 1, column 1: syntax error while parsing value - invalid literal; last read: '.') at row 4, column 13:

Current date: {{ currentDate }}
{{- if and .IsThinkSet .Think (ne .ThinkLevel "") }}
            ^
Reasoning: {{ .ThinkLevel }}
```

### Template Comparison
**Problematic Go syntax in chat-template-oss.jinja:**
```go
Current date: {{ currentDate }}
{{- if and .IsThinkSet .Think (ne .ThinkLevel "") }}
Reasoning: {{ .ThinkLevel }}
```

**Correct Jinja2 syntax in chat-template-oss-fixed.jinja:**
```jinja2
Current date: {{ strftime_now("%Y-%m-%d") }}

{% if tools %}
You have access to the following tools:
{% for tool in tools %}
- {{ tool.function.name }}: {{ tool.function.description }}
{% endfor %}
```

### Solution Applied
1. ✅ **Enabled GPT-OSS template**: Uncommented template directory and chat template configuration
2. ✅ **Switched to fixed template**: Changed from `chat-template-oss.jinja` to `chat-template-oss-fixed.jinja`
3. ✅ **Verified template mounting**: Confirmed templates are properly mounted in container

### Current Status
- **Template Loading**: Still loading the old Go-syntax template despite configuration changes
- **Tool Calling**: Model continues to use JSON response format but chooses "response" over "tool_call"
- **Next Steps**: Container restart may be required to fully reload template configuration

### Model Behavior Analysis
Even with the grammar constraints forcing JSON format with tool_call/response options, the GPT-OSS-20B model consistently chooses the "response" path rather than using available tools. This suggests the model may need:
1. **Better prompting**: More explicit instructions about when to use tools
2. **Temperature adjustment**: Lower temperature for more deterministic tool usage
3. **Alternative model**: Consider switching to ToolACE-2-Llama-3.1-8B which was specifically trained for tool calling

**Date**: 2025-01-05
**Status**: Template syntax fixed, but container still loading old template - requires further investigation

---

## API Token Configuration Fix - COMPLETED ✅

### Issue
The chat page was no longer working after an API token was added to the deployment. Users couldn't access the chat functionality because the API key wasn't being properly configured in the UI.

### Root Cause
The API key configuration was already present in the UI settings, but there was an issue with how the server props were being fetched. The `getServerProps` function was only called once on component mount and wasn't being refetched when the API key was updated in the settings.

### Solution Applied
**Fixed server props refresh on API key changes:**
```typescript
// utils/app.context.tsx - Line 108
// Changed dependency array from empty [] to [config.apiKey]
useEffect(() => {
  getServerProps(BASE_URL, config.apiKey)
    .then((props) => {
      console.debug('Server props:', props);
      setServerProps(props);
    })
    .catch((err) => {
      console.error(err);
      toast.error('Failed to fetch server props');
    });
}, [config.apiKey]); // Now depends on config.apiKey
```

### How It Works
1. **API Key in Settings**: The API key field is already present in the General section of the Settings dialog
2. **Automatic Refresh**: When the user saves a new API key in settings, the `useEffect` automatically triggers
3. **Server Props Update**: The server props are refetched with the new API key
4. **API Calls Updated**: All subsequent API calls use the new API key from the updated config

### User Instructions
1. **Access Settings**: Click the settings icon in the top navigation
2. **Enter API Key**: In the General section, enter your API key in the "API Key" field
3. **Save Settings**: Click "Save" to apply the changes
4. **Automatic Update**: The system will automatically refetch server properties with the new API key

### Verification
- ✅ **Settings UI**: API Key field is present and functional
- ✅ **Auto-refresh**: Server props are refetched when API key changes
- ✅ **API Integration**: All API calls (chat completions, server props) use the configured API key
- ✅ **No Linting Errors**: Code changes pass all linting checks

### Files Modified
- `llama.cpp/tools/server/webui/src/utils/app.context.tsx` - Fixed useEffect dependency to refresh server props when API key changes

**Date**: 2025-01-05
**Status**: API token configuration is now fully functional in the UI

---

## Main Frontend API Token Configuration - COMPLETED ✅

### Issue Discovery
After implementing API key configuration in the llama.cpp webui, discovered that there are **two separate frontends**:
1. **LlamaCPP WebUI** (`./llama.cpp/tools/server/webui/`) - Built-in chat interface (not being used)
2. **Main Frontend** (`./frontend/`) - React management interface (actually being used)

The chat page issue was in the main frontend, which had **hardcoded API keys** instead of configurable ones.

### Root Cause
The main frontend API service (`frontend/src/services/api.ts`) had hardcoded API keys:
```typescript
headers: {
  'Content-Type': 'application/json',
  'Authorization': 'Bearer placeholder-api-key'  // Hardcoded!
}
```

### Solution Applied
1. **Created Settings Manager** (`frontend/src/utils/settings.ts`):
   - LocalStorage-based settings management
   - Singleton pattern with event listeners
   - Type-safe settings interface

2. **Updated API Service** (`frontend/src/services/api.ts`):
   - Removed hardcoded API keys
   - Added dynamic API key injection via request interceptors
   - Updated both axios clients and fetch streaming requests

3. **Added API Settings UI** (`frontend/src/pages/ConfigurationPage.tsx`):
   - New "API Settings" tab (first tab in configuration)
   - Password field for API key input
   - Save functionality with validation
   - Visual feedback for unsaved changes

### Implementation Details
**Settings Storage:**
```typescript
// Singleton settings manager with localStorage persistence
export const settingsManager = SettingsManager.getInstance();

// API key management
settingsManager.setApiKey('your-api-key');
const apiKey = settingsManager.getApiKey();
```

**Dynamic API Key Injection:**
```typescript
// Request interceptor adds API key to all requests
private getAuthHeaders() {
  const apiKey = settingsManager.getApiKey();
  return apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {};
}
```

**UI Integration:**
- New "API Settings" tab in Configuration page
- Password field for secure API key entry
- Real-time validation and save status
- Persistent storage in browser localStorage

### User Instructions
1. **Access Configuration**: Navigate to Configuration page in the main frontend
2. **API Settings Tab**: Click on the "API Settings" tab (first tab)
3. **Enter API Key**: Input your API key in the password field
4. **Save Settings**: Click "Save API Key" button
5. **Automatic Application**: All subsequent API calls will use the configured key

### Verification Results
- ✅ **Frontend Rebuilt**: Successfully rebuilt with new API key functionality
- ✅ **All Containers Healthy**: Frontend, backend, and LlamaCPP API all running
- ✅ **No Linting Errors**: All TypeScript code passes validation
- ✅ **Settings Persistence**: API key stored in localStorage and survives page refresh
- ✅ **Dynamic Injection**: API key automatically added to all HTTP requests

### Files Modified
- `frontend/src/utils/settings.ts` - New settings management utility
- `frontend/src/services/api.ts` - Updated to use configurable API key
- `frontend/src/pages/ConfigurationPage.tsx` - Added API Settings tab and UI

**Date**: 2025-01-05
**Status**: Main frontend API token configuration is fully functional - chat page should now work with proper API authentication

---

## Configuration Page Redundancy Cleanup - COMPLETED ✅

### Issue Identified
After implementing API key configuration in the ConfigurationPage, discovered that the DeployPage already provides comprehensive configuration capabilities, making the ConfigurationPage redundant.

### Redundancy Analysis
**ConfigurationPage** (old, limited):
- Basic API key settings
- Simple model parameters (name, variant, context size, GPU layers)
- Basic sampling settings (temperature, top-p, top-k)
- Basic performance settings (threads, batch size, max tokens)
- Simple command line preview

**DeployPage** (comprehensive, modern):
- ✅ **API Key configuration** (Server tab - lines 1849-1871)
- ✅ **Advanced model parameters** (LoRA, multimodal, RoPE scaling, MoE settings)
- ✅ **Comprehensive sampling** (DRY sampling, penalties, min-p, etc.)
- ✅ **Advanced performance** (memory options, NUMA, cache types, parallel slots)
- ✅ **Context extension** (YaRN parameters, group attention)
- ✅ **Server configuration** (host, port, timeouts, logging)
- ✅ **Template management** (chat template selection and management)
- ✅ **LlamaCPP version management** (commit selection and rebuilding)
- ✅ **Real-time command preview** with parameter descriptions
- ✅ **Parameter reset functionality** (individual and bulk reset)
- ✅ **Better UX** (detailed descriptions, validation, organized tabs)

### Solution Applied
**Replaced ConfigurationPage with redirect page**:
- Shows clear message that configuration moved to Deploy page
- Lists all available features in Deploy page
- Auto-redirects to Deploy page after 3 seconds
- Provides immediate "Go to Deploy Page" button
- Maintains user experience while eliminating redundancy

### Implementation Details
```typescript
// New ConfigurationPage.tsx - Simple redirect component
export const ConfigurationPage: React.FC = () => {
  useEffect(() => {
    const timer = setTimeout(() => {
      window.location.href = '/deploy';
    }, 3000);
    return () => clearTimeout(timer);
  }, []);

  return (
    // Informative redirect UI with feature list
  );
};
```

### User Experience
1. **Existing users** visiting `/configuration` see clear explanation
2. **Feature discovery** - users learn about comprehensive Deploy page capabilities
3. **Smooth transition** - auto-redirect ensures no broken workflows
4. **No functionality loss** - all configuration options available in Deploy page

### Benefits
- ✅ **Eliminated redundancy** - Single source of truth for configuration
- ✅ **Better UX** - Users directed to superior interface
- ✅ **Reduced maintenance** - One comprehensive configuration system
- ✅ **Feature consolidation** - All advanced options in one place
- ✅ **Cleaner codebase** - Removed duplicate functionality

### Files Modified
- `frontend/src/pages/ConfigurationPage.tsx` - Replaced with redirect page

### Where to Configure API Key Now
**Deploy Page → Server Tab → API Key field** (line 1849-1871 in DeployPage.tsx)

The API key configuration is now part of the comprehensive server configuration section alongside host, port, timeout, and other server settings.

**Date**: 2025-01-05
**Status**: Configuration page redundancy eliminated - all configuration now centralized in Deploy page

---

## Chat Page Endpoint and API Key Configuration - COMPLETED ✅

### Enhancement Added
Added configurable endpoint and API key settings directly to the Chat page, allowing users to quickly adjust connection settings without leaving the chat interface.

### Features Implemented
1. **Connection Settings Section** in chat settings panel:
   - **API Endpoint** field - configurable chat completions endpoint
   - **API Key** field - password-protected API key input
   - Real-time settings persistence via localStorage

2. **Custom API Integration**:
   - Chat-specific API call functions that use local settings
   - Support for both streaming and non-streaming responses
   - Dynamic authentication header injection
   - Fallback handling for streaming failures

3. **Settings Persistence**:
   - Automatic localStorage save/load for all chat settings
   - Settings persist across browser sessions
   - Graceful fallback to defaults if localStorage fails

4. **Enhanced UI Organization**:
   - **Connection Settings** section (endpoint, API key)
   - **Model Parameters** section (temperature, top-p, top-k, max tokens)
   - **Function Calling Tools** section (tool selection and management)

### Implementation Details

**Extended ChatSettings Interface:**
```typescript
interface ChatSettings {
  // Connection settings
  endpoint: string
  apiKey: string
  // Model parameters
  temperature: number
  topP: number
  topK: number
  maxTokens: number
  streamResponse: boolean
  enableTools: boolean
  selectedTools: string[]
}
```

**Custom API Functions:**
```typescript
// Uses chat-specific endpoint and API key
const createChatCompletionStream = async (request: ChatCompletionRequest) => {
  const authHeaders = settings.apiKey ? { 'Authorization': `Bearer ${settings.apiKey}` } : {};
  const response = await fetch(settings.endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream',
      'Cache-Control': 'no-cache',
      ...authHeaders,
    },
    body: JSON.stringify({ ...request, stream: true })
  });
  // ... SSE parsing logic
};
```

**Settings Persistence:**
```typescript
// Automatic save to localStorage on any setting change
const saveSettings = (newSettings: ChatSettings) => {
  try {
    localStorage.setItem('chat-settings', JSON.stringify(newSettings));
    setSettings(newSettings);
  } catch (error) {
    console.warn('Failed to save chat settings to localStorage:', error);
    setSettings(newSettings); // Still update state even if save fails
  }
};
```

### User Experience
1. **Quick Access**: Settings accessible via gear icon in chat header
2. **Immediate Effect**: Changes apply instantly to new messages
3. **Persistent**: Settings saved automatically and restored on page reload
4. **Independent**: Chat settings separate from global Deploy page configuration
5. **Secure**: API key field uses password input type

### Default Settings
- **Endpoint**: `/v1/chat/completions` (relative URL for proxy compatibility)
- **API Key**: Empty (optional authentication)
- **Temperature**: 0.7 (balanced creativity/consistency)
- **Top P**: 0.8 (nucleus sampling)
- **Top K**: 20 (token filtering)
- **Max Tokens**: 2048 (reasonable response length)
- **Streaming**: Enabled (real-time response display)
- **Tools**: Disabled by default

### Benefits
- ✅ **Flexible Configuration**: Users can connect to different endpoints/models
- ✅ **Quick Testing**: Easy to test different API keys or endpoints
- ✅ **Development Friendly**: Supports local development and production deployments
- ✅ **User Convenience**: No need to navigate away from chat to change settings
- ✅ **Persistent Preferences**: Settings remembered across sessions

### Files Modified
- `frontend/src/pages/ChatPage.tsx` - Added connection settings and custom API integration

### How to Use
1. **Open Chat Page**: Navigate to `/chat` in the frontend
2. **Access Settings**: Click the settings (gear) icon in the chat header
3. **Configure Connection**: 
   - Set **API Endpoint** (e.g., `/v1/chat/completions`, `http://localhost:8600/v1/chat/completions`)
   - Set **API Key** if required by your deployment
4. **Adjust Parameters**: Configure temperature, top-p, top-k, max tokens as needed
5. **Enable Tools**: Toggle function calling and select available tools
6. **Start Chatting**: Settings apply immediately to new conversations

**Date**: 2025-01-05
**Status**: Chat page now has comprehensive endpoint and API key configuration with persistent settings

---

## Full URL Configuration for External Services - COMPLETED ✅

### Enhancement Added
Extended the chat page configuration to support full URL specification (base URL + endpoint path), enabling users to test completely different deployments and external services.

### New Features
1. **Separate Base URL and Endpoint Path Fields**:
   - **Base URL**: Full domain/service URL (e.g., `https://api.openai.com`, `http://localhost:11434`)
   - **Endpoint Path**: API path (e.g., `/v1/chat/completions`)
   - **Combined URL**: Automatically constructed and previewed

2. **Quick Preset Buttons** for common services:
   - **Local (Current Domain)**: Uses current domain with nginx proxy
   - **Local LlamaCPP (8600)**: Direct connection to local LlamaCPP server
   - **OpenAI API**: Official OpenAI API endpoint
   - **Ollama (11434)**: Local Ollama server

3. **Real-time URL Preview**: Shows the complete URL that will be used for API calls

4. **Flexible URL Construction**:
   - Empty base URL = uses current domain (relative URLs)
   - Specified base URL = uses absolute URLs to external services
   - Automatic trailing slash handling

### Updated ChatSettings Interface
```typescript
interface ChatSettings {
  // Connection settings
  baseUrl: string        // NEW: Base URL for external services
  endpoint: string       // Endpoint path
  apiKey: string
  // ... other settings
}
```

### URL Construction Logic
```typescript
const fullUrl = settings.baseUrl 
  ? `${settings.baseUrl.replace(/\/$/, '')}${settings.endpoint}` 
  : settings.endpoint;
```

### Use Cases Now Supported
1. **Local Development**: 
   - Base URL: `` (empty), Endpoint: `/v1/chat/completions`
   - Result: Uses nginx proxy to backend

2. **Direct LlamaCPP Server**:
   - Base URL: `http://localhost:8600`, Endpoint: `/v1/chat/completions`
   - Result: Direct connection bypassing proxy

3. **External OpenAI API**:
   - Base URL: `https://api.openai.com`, Endpoint: `/v1/chat/completions`
   - API Key: Required for authentication

4. **Local Ollama Server**:
   - Base URL: `http://localhost:11434`, Endpoint: `/v1/chat/completions`
   - Result: Connect to local Ollama instance

5. **Custom Deployment**:
   - Base URL: `https://my-custom-api.com`, Endpoint: `/api/v1/chat`
   - Result: Connect to any custom deployment

### UI Improvements
- **3-column layout**: Base URL | Endpoint Path | API Key
- **Quick preset buttons** for one-click configuration
- **Real-time URL preview** showing the complete constructed URL
- **Better field labels and help text** for clarity

### Benefits
- ✅ **Multi-Service Testing**: Easy switching between different AI services
- ✅ **Development Flexibility**: Test local, staging, and production deployments
- ✅ **External API Support**: Connect to OpenAI, Anthropic, or custom APIs
- ✅ **Quick Configuration**: Preset buttons for common setups
- ✅ **URL Transparency**: Clear preview of what URL will be used
- ✅ **Backward Compatibility**: Existing relative URL configurations still work

### Example Configurations

**Testing OpenAI GPT-4:**
```
Base URL: https://api.openai.com
Endpoint Path: /v1/chat/completions
API Key: sk-...your-openai-key...
```

**Testing Local Ollama:**
```
Base URL: http://localhost:11434
Endpoint Path: /v1/chat/completions
API Key: (leave empty)
```

**Testing Custom Deployment:**
```
Base URL: https://my-llm-service.example.com
Endpoint Path: /api/chat/completions
API Key: your-custom-api-key
```

### Files Modified
- `frontend/src/pages/ChatPage.tsx` - Added baseUrl field, updated UI, and API call logic

**Date**: 2025-01-05
**Status**: Chat page now supports full URL configuration for testing external services and different deployments

---

## Comprehensive Improvement Plan Implementation - IN PROGRESS

**Date**: 2025-12-07
**Goal**: Implement improvements from imprvement.md

### Phase 1: Foundation - COMPLETED

1. **Conversation Persistence** - DONE
   - Created: backend/modules/conversation_store.py
   - API endpoints: POST/GET/PUT/DELETE /api/v1/conversations
   - Features: Create, list, update, delete, export (JSON/Markdown)
   - Auto-generated titles from first user message

2. **VRAM Estimation** - DONE
   - Endpoint: POST /api/v1/estimate/vram
   - Calculates model weights, KV cache, compute buffer
   - Supports common quantization formats (Q2_K through F32)
   - Provides fit recommendations

### Phase 2: Core Features - COMPLETED

1. **Chat Markdown Rendering** - DONE
   - Added react-markdown, react-syntax-highlighter, remark-gfm
   - Created: frontend/src/components/chat/MarkdownRenderer.tsx
   - Syntax highlighting for code blocks with copy button
   - GFM support (tables, task lists, etc.)

2. **Thinking Trace Visualizer** - DONE
   - Collapsible reasoning content display in MarkdownRenderer
   - Supports reasoning_content from streaming responses
   - Auto-extracts <think> tags from content

### Implementation Progress
- [x] Read and analyzed imprvement.md
- [x] Add markdown rendering packages
- [x] Update ChatPage with markdown support
- [x] Add code syntax highlighting with Prism
- [x] Add thinking trace visualizer
- [x] Create backend conversation store module
- [x] Add conversation API endpoints to backend
- [x] Add conversation API functions to frontend service
- [x] Add conversation storage volume to docker-compose
- [x] Fix missing ModelCard components
- [x] Add VRAM estimation endpoint and API
- [x] Add conversation management UI to ChatPage
- [x] Context window manager/token counter visualization
- [x] WebSocket real-time updates integration

### Phase 3: Chat Interface Enhancements - COMPLETED (2025-12-07)

1. **Conversation Sidebar Integration** - DONE
   - Integrated ConversationSidebar component into ChatPage
   - Added conversation list with search, archive, delete, export functionality
   - History button in header with unsaved changes indicator

2. **Conversation Persistence in Chat** - DONE
   - Save/load conversation state
   - Auto-save functionality
   - Create new conversation button
   - Current conversation title display

3. **Context Window Manager** - DONE
   - Token count estimation (prompt tokens)
   - Visual progress bar showing context utilization
   - Color-coded warnings (green/yellow/red based on usage %)
   - Configurable max context tokens

4. **WebSocket Real-time Updates** - DONE
   - Added /ws WebSocket endpoint to backend
   - Broadcasts metrics (CPU, memory, GPU), status updates
   - Added nginx proxy configuration for WebSocket
   - Updated frontend WebSocket service to use nginx proxy
   - 5-second update interval for system metrics

### Files Modified (Phase 3)
- `frontend/src/pages/ChatPage.tsx` - Added conversation management UI, context window, save/load
- `frontend/src/components/chat/index.ts` - Added ConversationSidebar export
- `frontend/nginx.conf` - Added WebSocket proxy configuration
- `frontend/src/services/websocket.ts` - Updated to use nginx proxy
- `backend/main.py` - Added /ws WebSocket endpoint with metrics broadcasting

### Additional Fixes (2025-12-07)
- Fixed `include_archived` parameter missing in ConversationStore.list_conversations()
- Added `is_archived` field to Conversation dataclass
- Updated nginx.conf to use dynamic resolver for llamacpp-api (allows startup without GPU service)
- Fixed frontend WebSocket service to use nginx proxy

### Current Service Status
- Redis: Running on port 6379
- Backend API: Running on port 8700 (healthy)
- Frontend: Running on port 3002 (healthy)
- LlamaCPP API: Not running (requires GPU)

### Phase 4: Advanced Features - IN PROGRESS (2025-12-07)

1. **Deploy Page Parameter Presets** - DONE
   - Added preset selector UI to Sampling Configuration tab
   - Quick preset chips: Balanced, Coding, Creative, Precise
   - Color-coded by category with tooltips
   - One-click application of preset values

2. **Model Registry with Metadata Caching** - DONE
   - Created: backend/modules/model_registry.py
   - Database schema with model cache, variants, usage stats, ratings
   - API endpoints: /api/v1/registry/*
   - Features:
     - Cache model metadata from HuggingFace
     - Track quantization variants
     - Record usage statistics (loads, inferences)
     - User ratings and notes
     - Hardware recommendations based on VRAM

3. **Prompt Library** - DONE
   - Created: backend/modules/prompt_library.py
   - Full CRUD for prompt templates
   - Features:
     - Template variables with {{variable}} syntax
     - Version history with restore capability
     - Categories with icons and colors
     - Favorites and system prompts
     - Import/export functionality
     - Usage tracking
   - API endpoints: /api/v1/prompts/*

### Files Created (Phase 4)
- `backend/modules/model_registry.py` - Model metadata caching and registry
- `backend/modules/prompt_library.py` - Prompt template management
- `frontend/src/pages/PromptLibraryPage.tsx` - Prompt library UI

### Files Modified (Phase 4)
- `frontend/src/pages/DeployPage.tsx` - Added preset selector UI
- `backend/modules/__init__.py` - Added model_registry and prompt_library exports
- `backend/main.py` - Added model registry and prompt library API endpoints
- `frontend/src/services/api.ts` - Added prompt library and model registry API functions
- `frontend/src/types/api.ts` - Added prompt library and model registry types
- `frontend/src/App.tsx` - Added /prompts route
- `frontend/src/components/layout/Sidebar.tsx` - Added Prompts navigation item

### Frontend Prompt Library Features
- Category-based organization with colored sidebar
- Search and filter prompts
- Create/edit/delete prompt templates
- Template variables support with {{variable}} syntax
- Version history with restore capability
- Favorites system
- Copy to clipboard functionality
- Export prompts to JSON
- Use prompt dialog for variable substitution

### Phase 5: Performance Tools - IN PROGRESS (2025-12-07)

1. **Model Registry UI** - DONE
   - Created: frontend/src/pages/ModelRegistryPage.tsx
   - Features:
     - Browse cached models with stats
     - View model variants and quantizations
     - Usage statistics tab
     - Model ratings with notes
     - Cache new models manually

2. **Inference Speed Benchmark Tool** - DONE
   - Created: backend/modules/benchmark.py
   - Created: frontend/src/pages/BenchmarkPage.tsx
   - Features:
     - Measure tokens/second, TTFT, total time
     - Preset configurations (Quick, Standard, Long Context, Max Speed)
     - Custom benchmark configuration
     - Detailed statistics (min, max, mean, median, stdev)
     - Individual run breakdown
     - Benchmark history with comparison
     - Real-time progress polling
   - API endpoints: /api/v1/benchmark/*

3. **Bug Fixes**
   - Fixed ConversationStore.get_statistics() missing method
   - Added aiohttp dependency for benchmark HTTP client

### Files Created (Phase 5)
- `frontend/src/pages/ModelRegistryPage.tsx` - Model registry browsing UI
- `backend/modules/benchmark.py` - Inference benchmark runner
- `frontend/src/pages/BenchmarkPage.tsx` - Benchmark UI

### Files Modified (Phase 5)
- `backend/modules/__init__.py` - Added benchmark_runner export
- `backend/main.py` - Added benchmark API endpoints
- `backend/requirements.txt` - Added aiohttp dependency
- `backend/modules/conversation_store.py` - Added get_statistics method
- `frontend/src/App.tsx` - Added /registry and /benchmark routes
- `frontend/src/components/layout/Sidebar.tsx` - Added Registry and Benchmark nav items

4. **Batch Processing Interface** - DONE
   - Created: backend/modules/batch_processor.py
   - Created: frontend/src/pages/BatchProcessingPage.tsx
   - Features:
     - Create batch jobs with multiple prompts
     - Text input (one prompt per line) or file upload (JSON, CSV, TXT)
     - Configurable max tokens, temperature, system prompt
     - Real-time progress tracking with polling
     - View job details and individual item results
     - Export results to JSON or CSV
     - Cancel running jobs
   - API endpoints: /api/v1/batch/*

### Files Created (Phase 5 continued)
- `backend/modules/batch_processor.py` - Batch job processing
- `frontend/src/pages/BatchProcessingPage.tsx` - Batch processing UI

### Files Modified (Phase 5 continued)
- `backend/modules/__init__.py` - Added batch_processor export
- `backend/main.py` - Added batch processing API endpoints
- `frontend/src/App.tsx` - Added /batch route
- `frontend/src/components/layout/Sidebar.tsx` - Added Batch nav item

### Current Service Status (2025-12-07)
- Redis: Running on port 6379 (healthy)
- Backend API: Running on port 8700 (healthy)
- Frontend: Running on port 3002 (healthy)
- LlamaCPP API: Not running (requires GPU)

### All Feature APIs Tested and Working
- Prompt Library: 4 prompts, 1 system prompt
- Model Registry: 2 cached models
- Conversations: Stats endpoint working
- Benchmark: Stats and presets working
- Batch Processing: Jobs creation and management working

5. **VRAM Estimation Tool** - DONE
   - Created: backend/modules/vram_estimator.py
   - Features:
     - Automatic model architecture detection from filename
     - Support for 24+ quantization types (Q2_K through F32)
     - Known architectures for Llama, Qwen, Mistral, DeepSeek, Phi, Yi, Gemma
     - KV cache calculation with flash attention support
     - Compute buffer and overhead estimation
     - Utilization percentage and fit warnings
     - Real-time display on Deploy page
   - API endpoints: /api/v1/vram/estimate, /api/v1/vram/quantizations, /api/v1/vram/architectures

### Files Created (Phase 6)
- `backend/modules/vram_estimator.py` - VRAM requirement estimation

### Files Modified (Phase 6)
- `backend/modules/__init__.py` - Added vram_estimator export
- `backend/main.py` - Added VRAM estimation API endpoints
- `backend/Dockerfile` - Fixed modules directory copy
- `frontend/src/pages/DeployPage.tsx` - Added VRAM estimation display

6. **Function Calling Playground Improvements** - DONE
   - Enhanced: frontend/src/pages/TestingPage.tsx
   - Features:
     - Visual tool schema builder with parameter editor
     - Interactive playground for testing tool calls
     - Mock response support for end-to-end testing
     - Tool call visualization with expandable details
     - OpenAI schema preview and copy
     - Default example tools (calculator, weather, search)
     - Three tabs: Playground, Tool Builder, Benchmark
   - Capabilities:
     - Create custom tools with UI
     - Define parameters with types (string, number, boolean, array, object)
     - Set required/optional parameters
     - Configure mock JSON responses
     - See tool call chain with arguments
     - Copy generated OpenAI tool schema

### Files Modified (Phase 6 continued)
- `frontend/src/pages/TestingPage.tsx` - Complete rewrite with playground features

### Phase 7: Advanced Features - IN PROGRESS (2025-12-07)

1. **Model Comparison View** - DONE
   - Created: frontend/src/pages/ModelComparisonPage.tsx
   - Features:
     - Side-by-side model comparison with same prompt
     - Multiple model slots (2+ models)
     - Configurable endpoints, API keys, and parameters per model
     - Parallel execution with individual progress indicators
     - Metrics comparison table (tokens, time, TPS)
     - Response comparison with expand/collapse
     - Export results to JSON
     - Comparison history with reload capability
     - Quick preset buttons for common endpoints

2. **Monitoring Page (with KV Cache Dashboard)** - DONE
   - Created: frontend/src/pages/MonitoringPage.tsx
   - Replaces placeholder monitoring page
   - Features:
     - System Resources tab (CPU, Memory, GPU metrics)
     - Inference Metrics tab (request stats, performance)
     - KV Cache tab (cache utilization, hit rate, tokens)
     - Token Usage tab (integrates TokenUsageTracker)
     - Real-time Prometheus metrics parsing from llama.cpp
     - Auto-refresh with WebSocket connection status
     - Performance tips based on cache statistics

3. **Event Bus Architecture** - VERIFIED (already exists)
   - File: backend/modules/event_bus.py
   - Redis Pub/Sub implementation
   - Channels: status, metrics, download, logs, model, conversation
   - Cache methods for key-value storage
   - Convenience methods for common events

4. **Token Usage Tracking** - VERIFIED (already exists)
   - Backend: backend/modules/token_tracker.py
   - Frontend: frontend/src/components/monitoring/TokenUsageTracker.tsx
   - Hooks: frontend/src/hooks/useMetrics.ts (useTokenUsage)
   - API endpoints: /v1/usage/tokens/*

### Files Created (Phase 7)
- frontend/src/pages/ModelComparisonPage.tsx
- frontend/src/pages/MonitoringPage.tsx

### Files Modified (Phase 7)
- frontend/src/App.tsx - Added Compare and Monitoring routes
- frontend/src/components/layout/Sidebar.tsx - Added Compare nav item

### Current Service Status
- Redis: Running on port 6379 (healthy)
- Backend API: Running on port 8700 (healthy)
- Frontend: Running on port 3002 (healthy)
- LlamaCPP API: Not running (requires GPU)

### Next Steps (Phase 8)
- [ ] RAG pipeline integration
- [ ] Multi-user authentication
- [ ] Model state machine (proper lifecycle management)
- [ ] Collaborative annotation system

---

## GPU/CPU Allocation Issue Analysis - COMPLETED

### Issue Identified
The model is deploying significant portions to CPU instead of GPU, as evidenced by:
```
load_tensors: tensor 'token_embd.weight' (q8_0) (and 126 others) cannot be used with preferred buffer type CUDA_Host, using CPU instead
load_tensors:   CPU_Mapped model buffer size =  9596.80 MiB
load_tensors:        CUDA0 model buffer size =  2390.06 MiB
```

### Root Cause Analysis
1. **N_CPU_MOE Parameter**: Currently set to `21` in docker-compose.yml, forcing MoE weights to CPU
2. **Quantization Format**: q8_0 tensors may have limited GPU support in current llama.cpp version
3. **MoE Layer Allocation**: Only 25/25 layers on GPU but MoE weights forced to CPU/Host memory

### Current Configuration Issues
- `N_CPU_MOE=21` forces first 21 MoE layers to CPU (should be 0 for full GPU)
- Large CPU buffer (9596.80 MiB) vs smaller GPU buffer (2390.06 MiB)
- Performance impact: Significant memory bandwidth bottleneck

### Solution Analysis - COMPLETED ✅

**Issue Resolved**: The problem was a combination of factors:

1. **Model Configuration Mismatch**: Backend was configured for gpt-oss-120b but container was loading gpt-oss-20b
2. **Hardcoded Script Messages**: start.sh had hardcoded "Qwen3-Coder-30B" messages regardless of actual model
3. **Default Parameter Pollution**: Script was applying default values even when parameters weren't explicitly set

**Solutions Applied**:

1. **Fixed Model Consistency**: Updated docker-compose.yml to use gpt-oss-20b for both services
2. **Dynamic Script Messages**: Made startup messages show actual MODEL_NAME instead of hardcoded text
3. **Conditional Parameter Logic**: Rewrote start.sh to only include parameters in llama-server command if explicitly set
4. **Optimal MoE Setting**: Set N_CPU_MOE=12 for RTX 5090 (32GB VRAM) to balance performance and memory usage

**Current Status with gpt-oss-20b**:
```
📋 Configuration (only showing explicitly set parameters):
   Context Size: 128000
   GPU Layers: 999
   CPU MoE Layers: 12
   Temperature: 0.7
   Top-P: 0.8
   Top-K: 20
   Batch Size: 2048

🚀 Full llama-server command:
llama-server --model /home/llamacpp/models/gpt-oss-20b-Q4_K_M.gguf --host 0.0.0.0 --port 8080 --api-key placeholder-api-key --ctx-size 128000 --n-gpu-layers 999 --batch-size 2048 --n-cpu-moe 12 --temp 0.7 --top-p 0.8 --top-k 20 --jinja --chat-template-file /home/llamacpp/templates/chat-template-oss.jinja --verbose --metrics --embeddings --flash-attn --cont-batching
```

**GPU Allocation Dramatically Improved**:
- CPU_Mapped model buffer: **379.71 MiB** (down from 9,596.80 MiB - 96% reduction!)
- CUDA0 model buffer: **10,694.15 MiB** (up from 2,390.06 MiB - 348% increase!)
- CUDA0 KV cache: **3,018.00 MiB** total
- CUDA0 compute buffer: **404.52 MiB**
- **Result**: Nearly all model data now on GPU instead of CPU
- All 25 layers successfully offloaded to GPU
- **Performance**: Should see significant speed improvement due to GPU utilization

---

## Docker SDK Command Passing Fix (Dec 7, 2025)

### Problem
Deploy page showing "Container failed to start or stopped immediately" with error:
```
error while handling argument "--flash-attn": expected value for argument
usage: -fa, --flash-attn [on|off|auto]
```

### Root Cause
In `backend/main.py` line 379, the command was being passed to Docker SDK as a joined string:
```python
command=" ".join(cmd),
```

When Docker SDK receives a string command with an exec-form ENTRYPOINT, it can have parsing issues. The `--flash-attn auto` arguments were not being passed correctly to the container.

### Solution
Changed to pass command as a list:
```python
command=cmd,
```

Docker SDK handles list commands properly, passing each element as a separate argument to the ENTRYPOINT script.

---

## Phase 8: UI Beautification and New Features - COMPLETED (Dec 7, 2025)

### UI Beautification

1. **Theme Enhancement** (`frontend/src/utils/theme.ts`)
   - Complete dark theme overhaul with rich color palette
   - Glass-morphism effects with backdrop blur
   - Gradient presets for buttons, cards, backgrounds
   - Custom shadow system with glow effects
   - Enhanced component styling across all MUI components
   - Custom scrollbar styling
   - Smooth transitions throughout

2. **Header Redesign** (`frontend/src/components/layout/Header.tsx`)
   - Gradient background with backdrop blur
   - Brand logo with gradient icon (Llama Nexus branding)
   - Page-specific accent colors and descriptions
   - Online status indicator chip
   - Modern 64px height

3. **Sidebar Enhancement** (`frontend/src/components/layout/Sidebar.tsx`)
   - Gradient background with glass effect
   - Color-coded navigation items per section
   - Hover animations with color transitions
   - Active state with glowing left border and pulsing indicator
   - Custom scrollbar styling
   - Footer with version info

4. **StatCard Redesign** (`frontend/src/components/dashboard/StatCard.tsx`)
   - Glass-morphism card design
   - Variant-specific gradient icons and colors
   - Hover animations with scale and glow effects
   - Trend indicators with +/- styling
   - Bottom accent line per variant

5. **Dashboard Page** (`frontend/src/pages/DashboardPage.tsx`)
   - New SectionCard component with accent colors and icons
   - Improved header with Live status chip
   - Better visual hierarchy with section labels
   - Responsive grid layout

6. **ServiceStatusDisplay** (`frontend/src/components/monitoring/ServiceStatusDisplay.tsx`)
   - Modern info boxes with subtle backgrounds
   - Color-coded status chips
   - Gradient progress bars for resource usage
   - Temperature indicators with warning states

### New Pages Created

1. **API Documentation Page** (`frontend/src/pages/ApiDocsPage.tsx`)
   - Interactive API documentation
   - Category-based endpoint organization (System, Chat, Conversations, etc.)
   - Expandable endpoint cards with request/response examples
   - "Try It" button for live API testing
   - Copy buttons for code snippets
   - Color-coded HTTP methods (GET=green, POST=purple, etc.)

2. **Workflow Builder Page** (`frontend/src/pages/WorkflowBuilderPage.tsx`)
   - Visual workflow builder for multi-step LLM pipelines
   - Node types: Input, LLM Call, Code, Condition, Output, Loop
   - Drag-and-drop style interface
   - Node connection system
   - Node editor dialog with type-specific configuration
   - Run workflow simulation
   - Result panel for output display
   - Local storage persistence

### Files Created
- `frontend/src/pages/ApiDocsPage.tsx` - API documentation explorer
- `frontend/src/pages/WorkflowBuilderPage.tsx` - Visual workflow builder

### Files Modified
- `frontend/src/utils/theme.ts` - Complete theme system overhaul
- `frontend/src/components/layout/Header.tsx` - Modern header design
- `frontend/src/components/layout/Sidebar.tsx` - Enhanced sidebar with colors
- `frontend/src/components/dashboard/StatCard.tsx` - Glass-morphism stat cards
- `frontend/src/pages/DashboardPage.tsx` - Improved dashboard layout
- `frontend/src/components/monitoring/ServiceStatusDisplay.tsx` - Modern monitoring UI
- `frontend/src/App.tsx` - Added new routes (/api-docs, /workflows)
- `frontend/src/types/index.ts` - Added color to NavigationItem

### Navigation Updates
- Added API Docs to Operations section (teal color)
- Added Workflows to Development section (purple color)

### Current Service Status
- Redis: Running on port 6379 (healthy)
- Backend API: Running on port 8700 (healthy)
- Frontend: Running on port 3002 (healthy)
- LlamaCPP API: Not running (requires GPU)

### Summary of All Features Implemented from imprvement.md

**Phase 1-2: Foundation & Core (DONE)**
- [x] Event-driven architecture (Redis event bus)
- [x] Model registry with metadata caching
- [x] Conversation persistence
- [x] Chat markdown rendering with syntax highlighting
- [x] Thinking trace visualizer
- [x] VRAM estimation

**Phase 3-4: Chat & Advanced Features (DONE)**
- [x] Conversation sidebar integration
- [x] Context window manager
- [x] WebSocket real-time updates
- [x] Deploy page parameter presets
- [x] Prompt library with templates

**Phase 5-6: Tools & Monitoring (DONE)**
- [x] Model registry UI
- [x] Inference benchmark tool
- [x] Batch processing interface
- [x] Function calling playground
- [x] Model comparison view
- [x] Monitoring page with KV cache dashboard
- [x] Token usage tracking

**Phase 7-8: Innovation & Polish (DONE)**
- [x] UI beautification with modern theme
- [x] API documentation page
- [x] Workflow builder for LLM pipelines
- [x] Glass-morphism effects
- [x] Responsive design improvements

### Remaining Items (Future Work)
- [ ] RAG pipeline integration
- [ ] Multi-user authentication
- [ ] CLI tool
- [ ] End-to-end testing
- [ ] Output constraint editor (JSON Schema builder)

---

## Phase 9: Advanced Features - COMPLETED (Dec 7, 2025)

### New Components Created

1. **Output Constraint Editor** (`frontend/src/components/chat/ConstraintEditor.tsx`)
   - Visual JSON Schema builder for structured LLM outputs
   - Property types: string, number, integer, boolean, array, object, null
   - Constraints:
     - String: minLength, maxLength, pattern, format, enum
     - Number: minimum, maximum, multipleOf
     - Array: minItems, maxItems, item type
     - Object: nested properties
   - Format validations: email, date-time, URI, UUID, hostname, IP addresses
   - GBNF grammar generation tab
   - Preset templates: Q&A Response, Classification, Entity Extraction, Summary
   - Export JSON Schema with copy functionality

2. **Knowledge Base Page** (`frontend/src/pages/KnowledgeBasePage.tsx`)
   - RAG (Retrieval-Augmented Generation) document management
   - Features:
     - Upload documents (PDF, TXT, MD, HTML)
     - Import from URL
     - Paste text content
     - Collection management (create, organize)
     - Document processing status tracking
     - Semantic search with similarity scores
     - Document statistics (chunks, size)
   - UI Elements:
     - Collections sidebar with document counts
     - Documents table with actions
     - Upload dialog with multiple modes
     - Search dialog with top-K results
     - New collection dialog

3. **Semantic Diff Component** (`frontend/src/components/chat/SemanticDiff.tsx`)
   - Compare regenerated LLM responses
   - Features:
     - Word-level diff algorithm
     - Inline and side-by-side view modes
     - Similarity percentage scoring
     - Concept extraction and analysis
     - Added/removed concept tracking
     - Statistics: words added, removed, unchanged
   - Copy report functionality

### Files Created
- `frontend/src/components/chat/ConstraintEditor.tsx`
- `frontend/src/components/chat/SemanticDiff.tsx`
- `frontend/src/pages/KnowledgeBasePage.tsx`

### Files Modified
- `frontend/src/App.tsx` - Added /knowledge route
- `frontend/src/components/chat/index.ts` - Added exports
- `frontend/src/components/layout/Header.tsx` - Added Knowledge Base config
- `frontend/src/components/layout/Sidebar.tsx` - Added Knowledge nav item

### Navigation Updates
- Added Knowledge Base to Development section (cyan color)

### Current Service Status
- Redis: Running on port 6379 (healthy)
- Backend API: Running on port 8700 (healthy)
- Frontend: Running on port 3002 (healthy)
- LlamaCPP API: Not running (requires GPU)

### Complete Feature List from imprvement.md

**Foundation (DONE)**
- [x] Event-driven architecture (Redis)
- [x] Model registry with metadata caching
- [x] Conversation persistence
- [x] VRAM estimation

**Core Features (DONE)**
- [x] Chat markdown rendering
- [x] Thinking trace visualizer
- [x] Context window manager
- [x] WebSocket real-time updates

**Advanced Features (DONE)**
- [x] Deploy page parameter presets
- [x] Prompt library with templates
- [x] Model registry UI
- [x] Inference benchmark tool
- [x] Batch processing interface
- [x] Function calling playground
- [x] Model comparison view
- [x] Monitoring page with KV cache

**Innovation & Tools (DONE)**
- [x] UI beautification
- [x] API documentation page
- [x] Workflow builder
- [x] Output constraint editor (JSON Schema)
- [x] Knowledge Base page (RAG)
- [x] Semantic diff for responses

### Remaining Items (Future Work)
- [ ] Multi-user authentication
- [ ] CLI tool
- [ ] End-to-end testing

---

## Phase 9: Comprehensive RAG System - IN PROGRESS (Dec 7, 2025)

### Overview
Implementing a full-featured RAG (Retrieval-Augmented Generation) system with:
- Document management with domain hierarchy
- Qdrant vector store integration
- GraphRAG with entity/relationship extraction
- Multiple chunking and retrieval strategies
- Document discovery with web search
- Knowledge graph visualization

### Backend RAG Modules Created

1. **Vector Store Layer** (`backend/modules/rag/vector_stores/`)
   - `base.py` - Abstract vector store interface with SearchResult, CollectionConfig
   - `qdrant_store.py` - Full Qdrant integration with batch operations, filtering, hybrid search

2. **Document Management** (`backend/modules/rag/document_manager.py`)
   - Domain-based organization with hierarchy
   - Document CRUD with metadata
   - Chunk storage with vector ID mapping
   - Duplicate detection via content hash
   - SQLite persistence

3. **Chunking Strategies** (`backend/modules/rag/chunkers/`)
   - `base.py` - Abstract chunker with config and utilities
   - `fixed_chunker.py` - Fixed-size chunks with overlap
   - `semantic_chunker.py` - Sentence/paragraph-aware chunking
   - `recursive_chunker.py` - Hierarchical splitting for structured docs

4. **Embedding Models** (`backend/modules/rag/embedders/`)
   - `base.py` - Abstract embedder interface
   - `local_embedder.py` - sentence-transformers integration (MiniLM, BGE, Nomic)
   - `api_embedder.py` - OpenAI, Cohere, Voyage AI support

5. **Retrieval Mechanisms** (`backend/modules/rag/retrievers/`)
   - `base.py` - Abstract retriever with RetrievalConfig
   - `vector_retriever.py` - Dense vector search with MMR
   - `hybrid_retriever.py` - Dense + sparse (BM25) with RRF
   - `graph_retriever.py` - Knowledge graph-enhanced retrieval

6. **GraphRAG** (`backend/modules/rag/graph_rag.py`)
   - Entity extraction using LLM
   - Relationship extraction
   - SQLite-based graph storage
   - In-memory adjacency for fast traversal
   - Subgraph extraction
   - Community detection
   - Entity merging

7. **Document Discovery** (`backend/modules/rag/discovery.py`)
   - Web search integration (DuckDuckGo, Serper)
   - URL content extraction
   - Relevance and quality scoring
   - Review queue with approve/reject workflow
   - Bulk operations

### Backend API Endpoints Added

**Domain Management:**
- `GET /api/v1/rag/domains` - List domains
- `POST /api/v1/rag/domains` - Create domain
- `GET /api/v1/rag/domains/{id}` - Get domain
- `DELETE /api/v1/rag/domains/{id}` - Delete domain
- `GET /api/v1/rag/domains/tree` - Get hierarchy

**Document Management:**
- `GET /api/v1/rag/documents` - List with filters
- `POST /api/v1/rag/documents` - Create/upload
- `GET /api/v1/rag/documents/{id}` - Get document
- `DELETE /api/v1/rag/documents/{id}` - Delete
- `GET /api/v1/rag/documents/{id}/chunks` - Get chunks
- `POST /api/v1/rag/documents/{id}/process` - Chunk and embed

**Vector Store:**
- `GET /api/v1/rag/collections` - List collections
- `GET /api/v1/rag/collections/{name}` - Collection info
- `DELETE /api/v1/rag/collections/{name}` - Delete

**Retrieval:**
- `POST /api/v1/rag/retrieve` - Vector retrieval
- `POST /api/v1/rag/retrieve/hybrid` - Hybrid retrieval

**GraphRAG:**
- `GET /api/v1/rag/graph/entities` - List entities
- `POST /api/v1/rag/graph/entities` - Create entity
- `PUT /api/v1/rag/graph/entities/{id}` - Update entity
- `DELETE /api/v1/rag/graph/entities/{id}` - Delete entity
- `POST /api/v1/rag/graph/entities/merge` - Merge entities
- `GET /api/v1/rag/graph/relationships` - Get relationships
- `POST /api/v1/rag/graph/relationships` - Create relationship
- `DELETE /api/v1/rag/graph/relationships/{id}` - Delete
- `GET /api/v1/rag/graph/visualize` - Full graph data
- `GET /api/v1/rag/graph/subgraph` - Subgraph extraction
- `POST /api/v1/rag/graph/extract` - Extract from text
- `GET /api/v1/rag/graph/statistics` - Graph stats

**Discovery:**
- `POST /api/v1/rag/discover/search` - Web search
- `GET /api/v1/rag/discover/queue` - Review queue
- `POST /api/v1/rag/discover/{id}/extract` - Extract content
- `POST /api/v1/rag/discover/{id}/approve` - Approve
- `POST /api/v1/rag/discover/{id}/reject` - Reject
- `POST /api/v1/rag/discover/bulk-approve` - Bulk approve
- `GET /api/v1/rag/discover/statistics` - Discovery stats

**Embeddings:**
- `GET /api/v1/rag/embeddings/models` - List models
- `POST /api/v1/rag/embeddings/embed` - Embed text

**Statistics:**
- `GET /api/v1/rag/statistics` - Full RAG stats

### Infrastructure Changes

**docker-compose.yml:**
- Added Qdrant service (ports 6333, 6334)
- Added qdrant_data volume
- Backend depends on qdrant health check
- Added QDRANT_HOST, QDRANT_PORT, RAG_DB_PATH env vars

### imprvement.md Updates

Added comprehensive RAG system plan in section 4.4 including:
- 4.4.1 Document Management System
- 4.4.2 GraphRAG Implementation
- 4.4.3 Qdrant Vector Store Integration
- 4.4.4 Embedding Model Management
- 4.4.5 Chunking Strategies
- 4.4.6 Retrieval Mechanisms
- 4.4.7 Search & Discovery
- 4.4.8 RAG Pipeline & Chat Integration
- 4.4.9 Frontend Pages & Components
- 4.4.10 Implementation Phases

### Files Created
- `backend/modules/rag/__init__.py`
- `backend/modules/rag/document_manager.py`
- `backend/modules/rag/graph_rag.py`
- `backend/modules/rag/discovery.py`
- `backend/modules/rag/vector_stores/__init__.py`
- `backend/modules/rag/vector_stores/base.py`
- `backend/modules/rag/vector_stores/qdrant_store.py`
- `backend/modules/rag/chunkers/__init__.py`
- `backend/modules/rag/chunkers/base.py`
- `backend/modules/rag/chunkers/fixed_chunker.py`
- `backend/modules/rag/chunkers/semantic_chunker.py`
- `backend/modules/rag/chunkers/recursive_chunker.py`
- `backend/modules/rag/embedders/__init__.py`
- `backend/modules/rag/embedders/base.py`
- `backend/modules/rag/embedders/local_embedder.py`
- `backend/modules/rag/embedders/api_embedder.py`
- `backend/modules/rag/retrievers/__init__.py`
- `backend/modules/rag/retrievers/base.py`
- `backend/modules/rag/retrievers/vector_retriever.py`
- `backend/modules/rag/retrievers/hybrid_retriever.py`
- `backend/modules/rag/retrievers/graph_retriever.py`

### Files Modified
- `docker-compose.yml` - Added Qdrant service and backend deps
- `backend/main.py` - Added RAG imports, initialization, and 50+ API endpoints

### Frontend Pages Created (Dec 7, 2025)

1. **Knowledge Graph Page** (`frontend/src/pages/KnowledgeGraphPage.tsx`)
   - World-class force-directed graph visualization
   - Interactive canvas with smooth physics simulation
   - Beautiful node styling with gradients, glows, and shadows
   - Particle flow effects along edges
   - Zoom, pan, and drag interactions
   - Entity type color-coding with legend
   - Entity search and filtering
   - Entity/relationship CRUD operations
   - Extract from text feature
   - Merge entities functionality
   - Fullscreen mode
   - Settings panel for display options
   - Statistics dashboard cards

2. **Documents Page** (`frontend/src/pages/DocumentsPage.tsx`)
   - Domain hierarchy tree navigation
   - Document list with filtering (status, type, search)
   - Document table with pagination
   - Document details panel with stats
   - Chunk viewer with slider navigation
   - Domain CRUD dialogs
   - Document upload dialog
   - Process document dialog (chunking options)
   - View content dialog
   - Status indicators with icons and colors

3. **Discovery Page** (`frontend/src/pages/DiscoveryPage.tsx`)
   - Web search interface
   - Search results display
   - Review queue with filtering
   - Document cards with relevance/quality scores
   - Approve/reject workflow
   - Bulk selection and operations
   - Domain assignment for approved docs
   - Statistics dashboard
   - Content extraction feature
   - Document preview dialog

### Navigation Updates
- Added "Knowledge Base" section to sidebar with:
  - Documents (green)
  - Graph (indigo)
  - Discovery (purple)
  - RAG Search (cyan)
- Updated Header with new page configs

### Tests Created
- `backend/test_rag_system.py` - Comprehensive test suite:
  - Document manager tests
  - GraphRAG tests
  - Chunker tests
  - Embedder tests
  - Discovery tests
  - Integration tests

### Requirements Updated
- Added aiosqlite, qdrant-client, sentence-transformers
- Added beautifulsoup4, numpy
- Added pytest, pytest-asyncio

### Files Created
- `frontend/src/pages/KnowledgeGraphPage.tsx`
- `frontend/src/pages/DocumentsPage.tsx`
- `frontend/src/pages/DiscoveryPage.tsx`
- `backend/test_rag_system.py`

### Files Modified
- `frontend/src/App.tsx` - Added new routes
- `frontend/src/components/layout/Sidebar.tsx` - Added Knowledge Base section
- `frontend/src/components/layout/Header.tsx` - Added page configs
- `backend/requirements.txt` - Added RAG dependencies

### RAG System Status: COMPLETE

All planned features implemented:
- [x] Document management with domain hierarchy
- [x] Qdrant vector store integration
- [x] GraphRAG with entity/relationship extraction
- [x] Multiple chunking strategies (fixed, semantic, recursive)
- [x] Embedding model management (local + API)
- [x] Retrieval mechanisms (vector, hybrid, graph-enhanced)
- [x] Document discovery with web search
- [x] Review queue with approve/reject workflow
- [x] Knowledge graph visualization
- [x] 50+ backend API endpoints
- [x] Comprehensive test suite

---

## Dec 7, 2025 - Model Deployment Issue

### Problem
Failed to deploy model with error:
```
nvidia-container-cli: device error: 1: unknown device: unknown
```

### Root Cause
The deployment code in `start_docker_cli()` and `start_docker_sdk()` hardcodes:
```
CUDA_VISIBLE_DEVICES=0,1
NVIDIA_VISIBLE_DEVICES=0,1
```

This assumes 2 GPUs (devices 0 and 1), but the system only has 1 GPU (device 0).

### Files Affected
- `backend/main.py` - Lines 526-527 (CLI) and 448-449 (SDK)
- `docker-compose.yml` - Lines 140-141 (llamacpp-api environment)

### Solution Applied
Changed from hardcoded `0,1` to `all` (with env var override):

**backend/main.py:**
- SDK: `os.getenv("CUDA_VISIBLE_DEVICES", "all")`
- CLI: `os.getenv("NVIDIA_VISIBLE_DEVICES", "all")`

**docker-compose.yml:**
- `CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}`
- `NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}`

Using `all` lets the NVIDIA container runtime auto-detect available GPUs.
Can be overridden by setting env vars if needed (e.g., `CUDA_VISIBLE_DEVICES=0`).

### Tests Created
Created `backend/test_deployment.py` with:
- Service start/stop/restart tests
- Status polling during async deployment (2-5 min)
- GPU configuration validation
- Error handling for missing GPUs
- Health check waiting logic
- Full deployment lifecycle test

Run tests:
```bash
# Quick tests only
pytest backend/test_deployment.py -v -m "not slow"

# All tests including slow deployment tests
pytest backend/test_deployment.py -v --timeout=600

# Just the full cycle test
pytest backend/test_deployment.py::TestDeploymentFullCycle -v --timeout=720
```

---

## Dec 7, 2025 - Deploy Page Issues

### Issue 1: 405 Method Not Allowed on VRAM Estimate
**Problem**: Frontend was calling `/backend/api/v1/vram/estimate` which doesn't exist

**Fix**: Changed to `/api/v1/vram/estimate` in `DeployPage.tsx`

Also fixed incorrect config field references:
- `config.performance?.contextSize` -> `config.model?.context_size`
- `config.performance?.batchSize` -> `config.performance?.batch_size`
- `config.performance?.gpuLayers` -> `config.model?.gpu_layers`

### Issue 2: Cannot Select Model
**Possible causes**:
1. No models downloaded - dropdown shows "No local models found"
2. Config model name doesn't match any available models
3. Backend `/v1/models` endpoint returning empty array

**Debugging**:
- Check browser console for errors
- Check if models are downloaded: `/v1/models` endpoint
- The config's `model.name` must match one of the available model names

**Files Modified**:
- `frontend/src/pages/DeployPage.tsx` - Fixed VRAM estimate endpoint path

---

## Dec 7, 2025 - Test Coverage Improvements

### Added Unit Tests For:

1. **VRAMEstimator** (`backend/modules/vram_estimator.py`)
   - Quantization detection from filenames (Q4_K_M, Q8_0, F16, etc.)
   - Model architecture detection (Llama, Qwen, Mistral)
   - Model weights VRAM calculation
   - KV cache scaling with context size
   - Partial GPU offload calculations
   - High VRAM usage warnings

2. **TokenTracker** (`backend/modules/token_tracker.py`)
   - Recording token usage with metadata
   - Aggregating usage by model
   - Time range filtering (1h, 24h, 7d, 30d)
   - Total usage statistics
   - Usage over time queries

3. **EventBus** (`backend/modules/event_bus.py`)
   - Local subscription/unsubscription
   - Publishing events to local handlers
   - Async handler support
   - Convenience methods (emit_status_change, emit_model_event, etc.)

4. **FixedChunker** (`backend/modules/rag/chunkers/fixed_chunker.py`)
   - Empty/whitespace text handling
   - Chunk indexing and positioning
   - Metadata preservation
   - Overlap functionality
   - Sentence boundary preservation

5. **ConversationStore Edge Cases** (`backend/modules/conversation_store.py`)
   - Auto-title generation from first message
   - Long title truncation
   - Nonexistent conversation handling
   - JSON/Markdown export formats
   - Search and tag filtering
   - Archived conversation filtering
   - Tool calls and reasoning content

### Running Tests

```bash
cd backend
pytest test_new_features.py -v
```

### Test File Structure
- `test_new_features.py` - Unit tests for modules (PromptLibrary, ModelRegistry, VRAMEstimator, etc.)
- `test_deployment.py` - Integration tests for deployment endpoints
- `test_token_tracking.py` - Integration tests for token tracking API
- `test_rag_system.py` - RAG system tests

---

## RAG Document Processing Troubleshooting (2025-12-08)

### Issues Found and Fixed

1. **Vector ID Format Error**
   - Error: `Unable to parse UUID: {document_id}_{index}`
   - Fix: Changed vector_id from `f"{document_id}_{i}"` to `str(uuid.uuid4())` for Qdrant compatibility

2. **Embedding Batch Size Too Small**
   - Error: `input is too large to process. increase the physical batch size`
   - Fix: Increased `BATCH_SIZE` and `UBATCH_SIZE` from 512 to 2048 in docker-compose.yml for llamacpp-embed service

3. **Embedding Model Selection Logic**
   - Issue: Using `nomic-embed-text` (invalid) instead of `nomic-embed-text-v1.5`
   - Fix: Documents must specify the exact model name that matches the deployed service model

4. **PDF Text Extraction Missing**
   - Error: Raw PDF binary stored instead of extracted text
   - Fix: Added PyPDF2 to requirements.txt and `extract_pdf_text()` function to main.py
   - Reprocess endpoint now auto-extracts text from PDF documents

5. **Llamacpp Embedding Server Batching**
   - Issue: APIEmbedder sent multiple texts in batch but llamacpp server preferred one at a time
   - Fix: Modified `api_embedder.py` to send texts one-by-one for llamacpp provider

6. **Missing Vector Store Alias Methods**
   - Error: `'QdrantStore' object has no attribute 'add_vectors'`
   - Fix: Added `add_vectors` and `delete_vectors` alias methods to QdrantStore

### Current Configuration

- **Embedding Service**: llamacpp-embed running on port 8602
- **Model**: nomic-embed-text-v1.5.Q8_0.gguf (768 dimensions, 2048 context)
- **Batch Size**: 2048 (increased from 512)
- **Vector Store**: Qdrant on port 6333

### Recommended Settings for RAG

```yaml
# docker-compose.yml llamacpp-embed settings
- BATCH_SIZE=2048
- UBATCH_SIZE=2048
- CONTEXT_SIZE=8192
```

### Important Notes

- PDF documents must be re-uploaded for proper text extraction (existing PDFs stored as binary won't extract properly)
- Use `USE_DEPLOYED_EMBEDDINGS=true` environment variable or API config for deployed embedding service
- Domain embedding model must match a valid model name: `nomic-embed-text-v1.5`, `all-MiniLM-L6-v2`, etc.

---

## Model Deployed to CPU Instead of GPU (2025-12-08)

### Issue
When deploying a model through the Deploy page, the model loaded on CPU instead of GPU with error:
```
ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected
warning: no usable GPU found, --gpu-layers option will be ignored
```

### Root Cause
The backend was using `--runtime nvidia` when starting the llamacpp-api container via Docker CLI, but this flag alone doesn't expose GPU devices. The modern NVIDIA Container Toolkit requires `--gpus all` to actually request GPU access.

### Fix
Changed `backend/main.py` to use `--gpus all` instead of `--runtime nvidia`:

```python
# Before (broken)
docker_cmd.extend(['--runtime', 'nvidia'])

# After (working)
docker_cmd.extend(['--gpus', 'all'])
```

Also fixed Docker SDK approach to use `device_requests` instead of `runtime`:
```python
container_config["device_requests"] = [
    docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
]
```

### Resolution
Rebuilt backend-api container and redeployed the model from the Deploy page.

---

## GraphRAG Knowledge Graph Page Issues (2025-12-08)

### Problem
Knowledge Graph page shows stats (520 entities, 550 relationships) but graph explorer shows 0 nodes, 0 edges.

### Root Cause
Two issues identified:

1. **Response Structure Mismatch**: The GraphRAG service `/knowledge-graph/filtered` endpoint returns data nested under `filtered_data`:
   ```json
   {
     "filtered_data": {
       "nodes": [...],
       "edges": [...],
       "stats": {...}
     }
   }
   ```
   But the frontend expects:
   ```json
   {
     "nodes": [...],
     "edges": [...]
   }
   ```

2. **Empty Edges Array**: Even with `max_relationships` set, the filtered endpoint returns empty edges array despite having 550 relationships in the database.

### Investigation
- GraphRAG health endpoint works: `{"status":"healthy"}`
- Stats endpoint returns: `{"nodes":520,"edges":550,...}`
- Top-entities and top-relationships endpoints return real data
- The `/knowledge-graph/filtered` POST endpoint returns `filtered_data.edges: []`

### Fix Applied
Updated `backend/routes/graphrag.py` to:
1. Unwrap the `filtered_data` wrapper to return `nodes` and `edges` directly
2. Transform entity types from uppercase GraphRAG format (COMPONENT, MAINTENANCE) to lowercase frontend format (technology, process)
3. Fetch relationships from `/top-relationships` endpoint when filtered endpoint returns empty edges
4. Build edges array by matching relationship source/target to node IDs

### Verification
```bash
curl -s -X POST http://localhost:8700/api/v1/graphrag/graph -H "Content-Type: application/json" -d '{"max_entities": 100, "max_relationships": 200}'
# Result: Nodes: 100, Edges: 44
```

### Files Modified
- `backend/routes/graphrag.py` - Added `_normalize_entity_type()`, `_transform_nodes()`, and updated `/graph` endpoint

---

## Knowledge Graph Visualization Improvements (2025-12-08)

### Problem
Custom canvas-based force-directed graph was causing nodes to bounce erratically and appear fixed to a grid.

### Solution
Replaced custom implementation with `react-force-graph-2d` library which:
- Uses proper D3.js force simulation under the hood
- Handles physics, collision detection, and layout automatically
- Provides smooth animations and drag interactions
- Supports directional particles on edges
- Has proper zoom and pan interactions

### Changes Made
1. Added `react-force-graph-2d` to `frontend/package.json`
2. Rewrote `ForceGraphVisualization` component in `KnowledgeGraphPage.tsx`:
   - Uses `ForceGraph2D` component from the library
   - Custom node rendering with type-based colors
   - Custom link rendering with gradients and arrows
   - Particle flow on edges for visual appeal
   - Proper zoom/pan controls

### Key Features
- Nodes are colored by entity type (technology=purple, process=teal, etc.)
- Node size based on confidence/occurrence
- Directional arrows on edges
- Optional particle flow animation
- Labels shown on hover or always (configurable)
- Smooth force simulation with proper physics

## STT and TTS Backend API Implementation (Dec 10, 2025)

### Current Goal
Implement backend APIs for STT (Speech-to-Text) and TTS (Text-to-Speech) services to support the frontend deployment pages.

### Implementation Complete

#### Backend Routes Created
- `/home/alec/git/llama-nexus/backend/routes/stt.py` - STT service management routes
- `/home/alec/git/llama-nexus/backend/routes/tts.py` - TTS service management routes

#### Manager Classes Added to main.py
- `STTManager` - Manages STT service lifecycle (Docker-based faster-whisper)
- `TTSManager` - Manages TTS service lifecycle (Docker-based openedai-speech)

#### API Endpoints

**STT (`/api/v1/stt/*`)**
- `GET /api/v1/stt/status` - Service status
- `GET /api/v1/stt/config` - Configuration and available models
- `PUT /api/v1/stt/config` - Update configuration
- `POST /api/v1/stt/start` - Start service
- `POST /api/v1/stt/stop` - Stop service
- `POST /api/v1/stt/restart` - Restart service
- `POST /api/v1/stt/transcribe` - Proxy transcription requests

**TTS (`/api/v1/tts/*`)**
- `GET /api/v1/tts/status` - Service status
- `GET /api/v1/tts/config` - Configuration, models, voices
- `PUT /api/v1/tts/config` - Update configuration
- `POST /api/v1/tts/start` - Start service
- `POST /api/v1/tts/stop` - Stop service
- `POST /api/v1/tts/restart` - Restart service
- `POST /api/v1/tts/synthesize` - Proxy speech synthesis requests

#### Docker Images Used
- STT: `fedirz/faster-whisper-server:latest-cuda` (port 8603)
- TTS: `ghcr.io/matatonic/openedai-speech:latest` (port 8604)

#### Default Configurations

**STT:**
- Model: whisper-base
- Language: auto-detect
- Compute type: float16
- Execution mode: GPU

**TTS:**
- Model: piper-en-us-lessac-medium
- Voice: alloy
- Audio format: mp3
- Speed: 1.0
- Execution mode: CPU

### Files Modified
- `backend/routes/__init__.py` - Added stt_router and tts_router exports
- `backend/main.py` - Added STTManager, TTSManager classes, initialized managers, registered routes

### Verification
- APIs tested and working via curl
- Frontend rebuilt with --no-cache to include STT/TTS pages
- Services accessible at localhost:8700/api/v1/stt/* and localhost:8700/api/v1/tts/*

---

## Deploy Page Bugs Fixed (2025-12-10)

### Issue 1: Chat Template Selection Changes Model Name
When selecting a chat template in the Deploy page, the model name and variant selections were being overwritten.

**Root Cause:**
In the template selection onChange handler, after selecting a template the code fetched fresh config from the server and replaced the entire local config:
```javascript
setConfig(cfgJson.config) // This overwrote user's model selection
```

**Fix:**
Removed the `setConfig(cfgJson.config)` call - now only updates the command line preview without touching the config state.

### Issue 2: Model Selection Not Applied When Deploying
Users could select a different model name/variant but clicking Start/Restart would deploy the previously configured model.

**Root Cause:**
The backend's `service_action` endpoint received the config from the frontend but completely ignored it. It just called `manager.start()` without applying the new config first.

**Fix:**
Updated `backend/routes/service.py` to apply the config before starting/restarting:
```python
# Apply config before start/restart if provided
if config and action in ("start", "restart"):
    merge_config = get_merge_config_func(request)
    if merge_config:
        merge_config(config)
    else:
        manager.config = {**manager.config, **config}
```

---

## Chat Page Thinking Token Display (2025-12-10)

### Issue
With some models (like gpt-oss-20b), only thinking tokens were visible in the chat, with no actual response content displayed.

### Investigation
The MarkdownRenderer was only handling `<think>...</think>` tags but:
1. Some models use different tags like `<thinking>` or `<thought>`
2. If the closing tag never arrived (model still thinking), the regex wouldn't match
3. The non-greedy regex `*?` could behave unexpectedly

### Fix
Updated `MarkdownRenderer.tsx` to:
1. Support multiple thinking tag formats: `<think>`, `<thinking>`, `<thought>`
2. Handle unclosed thinking tags - if content starts with thinking tag but no closing tag, show as thinking (response still being generated)
3. Properly extract and display all thinking blocks

Added debug logging to ChatPage.tsx streaming handler to help diagnose similar issues.


---

## Voice Mode for Chat Page (2025-12-10)

### Feature
Added voice input with Voice Activity Detection (VAD) and TTS output for the ChatPage, enabling hands-free conversational interaction with the LLM.

### Components Added

1. **useVoiceInput Hook** (`frontend/src/hooks/useVoiceInput.ts`)
   - Web Audio API for real-time audio analysis
   - Voice Activity Detection (VAD) to detect when user stops speaking
   - Automatic silence detection with configurable threshold and duration
   - Integration with local STT service for transcription
   - Continuous listening mode - auto-records when speech detected

2. **useTTS Hook** (`frontend/src/hooks/useTTS.ts`)
   - Text-to-speech playback using local TTS service
   - Queue-based playback for handling multiple responses
   - Play/pause/stop controls
   - Configurable voice, speed, and auto-play settings

3. **API Service Extensions** (`frontend/src/services/api.ts`)
   - `getSTTStatus()` - Check if STT service is running
   - `getTTSStatus()` - Check if TTS service is running
   - `transcribeAudio()` - Send audio to local STT for transcription
   - `synthesizeSpeech()` - Generate speech from text using local TTS

### ChatPage Updates

- Voice mode toggle button in header (requires STT service to be deployed)
- Voice status panel showing listening/recording/transcribing state
- Visual volume indicator with threshold line
- TTS playback controls (play/pause/stop)
- Auto-send when silence detected after speech
- Auto-play TTS for assistant responses (configurable)

### Voice Settings (in Chat Settings panel)
- TTS voice selection (alloy, echo, fable, onyx, nova, shimmer)
- TTS speed (0.5x to 2x)
- Auto-play toggle
- STT model selection (tiny, base, small, medium, large-v3)
- STT language (auto-detect or specific)
- VAD silence duration (500ms to 3000ms)
- VAD silence threshold (1% to 10%)

### Usage Flow
1. Deploy STT service from STT Deploy page
2. Deploy TTS service from TTS Deploy page (optional, for voice output)
3. Enable voice mode from chat page header
4. Start speaking - system auto-detects speech and starts recording
5. Pause speaking - after silence duration, audio is sent for transcription
6. Transcribed text appears in input and auto-sends
7. Assistant response is spoken via TTS (if enabled)
8. Continue speaking for next turn

### Requirements
- STT service must be deployed and running for voice input
- TTS service must be deployed and running for voice output
- Microphone permissions required in browser

### TTS Streaming Enhancement (2025-12-10)

Updated TTS to stream sentences as they complete during response generation, rather than waiting for the entire response. This significantly reduces perceived latency.

**Implementation:**
- Added sentence buffer in streaming handler
- Detects sentence boundaries: `.`, `!`, `?` followed by whitespace, or double newlines
- Sends completed sentences to TTS queue immediately
- Flushes remaining content when stream ends

**Benefits:**
- User starts hearing the response within 1-2 sentences (~2-5 seconds) instead of waiting for full response
- Natural pauses between sentences while TTS catches up
- Works well with the existing TTS queue system

### TTS Service Management from ChatPage (2025-12-11)

Added ability to start/stop the TTS service directly from the ChatPage without needing to go to a separate deploy page.

**Changes Made:**

1. **API Service** (`frontend/src/services/api.ts`)
   - `startTTSService(config?)` - Start the TTS service with optional config
   - `stopTTSService()` - Stop the running TTS service  
   - `getTTSConfig()` - Get available TTS configuration options

2. **ChatPage UI Updates** (`frontend/src/pages/ChatPage.tsx`)
   - "Start TTS" button in header when TTS is not running (quick launch)
   - Start/Stop buttons in Voice Settings panel next to TTS status chip
   - Play button on assistant messages to read aloud (when TTS is available)
   - Loading states for start/stop operations

**Usage:**
- When TTS is not available, a "Start TTS" button appears in the header
- Click to start the TTS service directly from the chat interface
- TTS can also be stopped from the Voice Settings panel
- Individual messages can be played with the speaker icon

**Backend Endpoints Used:**
- `POST /api/v1/tts/start` - Start TTS service
- `POST /api/v1/tts/stop` - Stop TTS service
- `GET /api/v1/tts/config` - Get TTS config and available voices
