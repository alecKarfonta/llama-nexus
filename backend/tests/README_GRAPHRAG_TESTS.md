# GraphRAG Integration Tests

This directory contains comprehensive tests for the GraphRAG integration in llama-nexus.

## Test Files

### 1. `test_graphrag_integration.py`
Unit tests for GraphRAG proxy endpoints and workflow executors using mocks.

**Test Coverage:**
- GraphRAG proxy endpoints (health, stats, search, extract, etc.)
- Workflow executors (search, entity extraction, reasoning)
- Error handling (connection errors, timeouts, invalid inputs)
- Workflow templates registration
- Integration scenarios

**Run:**
```bash
cd /home/alec/git/llama-nexus/backend
pytest tests/test_graphrag_integration.py -v
```

**With coverage:**
```bash
pytest tests/test_graphrag_integration.py -v --cov=routes.graphrag --cov=modules.workflow.executors.graphrag_executors
```

### 2. `test_graphrag_executors.py`
Focused tests for each GraphRAG workflow executor.

**Test Coverage:**
- GraphRAGSearchExecutor (all search types)
- EntityExtractionExecutor
- MultiHopReasoningExecutor
- CausalReasoningExecutor
- ComparativeReasoningExecutor
- EntityLinkingExecutor
- CodeDetectionExecutor
- CodeSearchExecutor
- Configuration handling
- Data flow between executors

**Run:**
```bash
pytest tests/test_graphrag_executors.py -v
```

### 3. `test_graphrag_e2e.py`
End-to-end tests against live GraphRAG service.

**Prerequisites:**
- GraphRAG service must be running at `http://localhost:18000`
- Backend service must be running at `http://localhost:8700`

**Test Coverage:**
- Service connectivity
- Document upload (single and batch)
- Intelligent search (all methods: auto, hybrid, vector, graph, keyword)
- Entity operations
- Reasoning capabilities
- Knowledge graph operations
- Performance characteristics
- Edge cases (long queries, special characters, unicode)

**Run:**
```bash
# First, ensure GraphRAG is running
cd ~/git/graphrag && docker compose up -d

# Then run E2E tests
cd /home/alec/git/llama-nexus/backend
pytest tests/test_graphrag_e2e.py -v --graphrag-live
```

## Quick Integration Verification

### Option 1: Quick Test Script
Run the standalone test script from the project root:

```bash
cd /home/alec/git/llama-nexus
python test_graphrag_integration.py
```

This will:
1. Check GraphRAG service health
2. Verify backend proxy connections
3. Test document upload
4. Test search functionality
5. Test entity extraction
6. Verify workflow nodes are registered
7. Verify workflow templates exist

### Option 2: Manual API Testing

**Test GraphRAG directly:**
```bash
# Health check
curl http://localhost:18000/health

# Get stats
curl http://localhost:18000/knowledge-graph/stats

# Upload document
curl -X POST http://localhost:18000/process-document \
  -F "file=@test.txt" \
  -F "use_semantic_chunking=true"
```

**Test through backend proxy:**
```bash
# Health check
curl http://localhost:8700/api/v1/graphrag/health

# Get stats
curl http://localhost:8700/api/v1/graphrag/stats

# Intelligent search
curl -X POST http://localhost:8700/api/v1/graphrag/search/intelligent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "search_type": "auto", "top_k": 5}'
```

## Test Data Setup

To get meaningful test results, upload some documents first:

```bash
# Upload test documents via backend
curl -X POST http://localhost:8700/api/v1/graphrag/ingest/upload \
  -F "file=@sample_document.pdf" \
  -F "domain=general" \
  -F "use_semantic_chunking=true" \
  -F "build_knowledge_graph=true"
```

Or use the UI:
1. Go to Documents page
2. Click "Add Document"
3. Select "GraphRAG (Neo4j + Knowledge Graph)"
4. Upload files

## Expected Test Results

### Unit Tests
All unit tests should pass with proper mocking:
- `test_graphrag_integration.py`: ~15 tests
- `test_graphrag_executors.py`: ~20 tests

### E2E Tests
E2E tests require live services and may have variable results:
- Connection tests should always pass
- Search/extraction may return empty results if no documents uploaded
- Performance tests verify response times are reasonable

## Troubleshooting

### GraphRAG Service Not Available
```
Error: Cannot connect to GraphRAG service
```
**Solution:**
```bash
cd ~/git/graphrag
docker compose up -d
# Wait 30 seconds for services to start
docker compose ps  # Verify all services are "Up"
```

### Backend Not Running
```
Error: Cannot connect to backend
```
**Solution:**
```bash
cd /home/alec/git/llama-nexus
docker compose up -d backend
```

### Empty Search Results
```
Warning: Search returned 0 results
```
**Solution:** Upload documents first using the Documents page or API

### Import Errors
```
ImportError: No module named 'modules.workflow.executors.graphrag_executors'
```
**Solution:** Ensure you're running tests from the correct directory:
```bash
cd /home/alec/git/llama-nexus/backend
export PYTHONPATH=/home/alec/git/llama-nexus/backend:$PYTHONPATH
pytest tests/test_graphrag_integration.py -v
```

## Continuous Integration

To run all GraphRAG tests in CI:

```bash
# Unit tests (always run)
pytest tests/test_graphrag_integration.py tests/test_graphrag_executors.py -v

# E2E tests (only if GraphRAG service is available)
pytest tests/test_graphrag_e2e.py -v --graphrag-live
```

## Test Coverage Goals

- Backend proxy endpoints: 100%
- Workflow executors: >90%
- Error handling: >85%
- Integration scenarios: >80%

## Adding New Tests

When adding new GraphRAG features, add tests to:
1. `test_graphrag_integration.py` - Unit tests with mocks
2. `test_graphrag_executors.py` - Executor-specific tests
3. `test_graphrag_e2e.py` - Live service tests

Follow the existing patterns for consistency.

