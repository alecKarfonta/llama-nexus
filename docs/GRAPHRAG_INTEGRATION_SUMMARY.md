# GraphRAG Integration Summary

## Current Implementation Snapshot

The llama-nexus GraphRAG integration now exposes the main GraphRAG service
surface through typed adapters, workflow nodes, frontend helpers, and an
allowlisted upstream bridge for advanced or newly added endpoints.

## Local UI Smoke Path

The full frontend still builds through Vite, but the document-chat surface also
has a focused static build for local GraphRAG testing:

```bash
node scripts/build-graphrag-ui-esbuild.mjs
GRAPHRAG_UI_DIST_DIR=/home/alec/git/llama-nexus/frontend/dist-graphrag \
  scripts/start-graphrag-ui-static-proxy.sh
```

The static proxy serves `/graphrag` and `/chat`, proxies `/api/*` and `/v1/*`
to the configured backend, and is useful when you want to test upload,
reference expansion, PDF highlighting, and chat handoff without waiting on the
entire app shell.

For repeatable visual workflow checks, run the Playwright smoke harness against
the focused UI:

```bash
python3 scripts/graphrag_ui_e2e_smoke.py \
  --base-url http://127.0.0.1:55846 \
  --backend-url http://127.0.0.1:8799 \
  --browser firefox \
  --skip-browser-close \
  --question "How does Product Revenue connect to Regional Demand?" \
  --expect-answer-text "Product Revenue" \
  --expect-answer-text "Regional Demand" \
  --expect-graph-path "Product Revenue -> Regional Demand -> Risk Map" \
  --expect-pdf-render
```

The harness writes `summary.json` plus screenshots for the workbench, chat
answer, references drawer, PDF render, and selected graph-path evidence.

## What Was Added

### Backend: GraphRAG Proxy Route Package
**Files**: `backend/routes/graphrag/*.py`

The GraphRAG integration is split into focused route modules for health,
graph queries, search/reasoning, ingestion, chat context, feedback analytics,
and an allowlisted upstream bridge. All route modules share the pooled HTTP
client and unified `GraphRAGConfig`.

**Document Ingestion:**
- `POST /api/v1/graphrag/ingest` - Ingest JSON text/content by materializing it as an upstream document
- `POST /api/v1/graphrag/ingest/upload` - Upload single document
- `POST /api/v1/graphrag/ingest/batch` - Batch upload using GraphRAG's multi-file ingestion
- `POST /api/v1/graphrag/ingest/pipeline/preview` - Locally preview structure extraction, chunking, graph candidates, balanced relationship highlights, embedding inputs, interaction handoff lanes, and retrieval rehearsals without upstream ingestion
- `POST /api/v1/graphrag/ingest/pipeline/upload` - Upload with a tracked pipeline trace, per-stage sample outputs, chat-ready graph anchors/starter prompts, retrieval rehearsals, and adaptive remote ingest fallback across compatible GraphRAG endpoints
- `POST /api/v1/graphrag/ingest/pipeline/batch` - Upload multiple files with bounded concurrency, a shared upstream preflight, remembered per-file pipeline traces, interaction mode/rehearsal rollups, and an `upload_manifest` for file readiness and next actions
- `GET /api/v1/graphrag/ingest/pipeline/{run_id}` - Fetch the remembered pipeline trace for an upload or preview run
- `GET /api/v1/graphrag/documents` - List indexed GraphRAG documents
- `DELETE /api/v1/graphrag/documents/{document_name}` - Delete a named GraphRAG document
- `POST /api/v1/graphrag/rebuild` - Rebuild knowledge graph
- `POST /api/v1/graphrag/export` - Export graph data
- `POST /api/v1/graphrag/hybrid/process` - Route code/documents through hybrid processing

Pipeline preview supports text, Markdown, HTML, CSV, JSON, PDF, DOCX/DOCM/DOTX/DOTM, PPTX/PPTM/POTX/POTM/PPSX/PPSM, XLSX/XLSM/XLTX/XLTM, legacy DOC/PPT/XLS, EML email messages, RTF, EPUB, ODT, ODS, ODP, ZIP bundles, and common image formats with best-effort extraction. Complex Office and OpenDocument formats use XML fallbacks so slide text, slide visibility/source-part provenance, hidden-slide graph edges, source-anchored hyperlinks, PowerPoint picture/layout action links with clicked-image provenance, PowerPoint and OpenDocument shape titles/descriptions/ids, object geometry with readable slide bboxes/zones, PowerPoint comments with slide-position provenance, connector arrows with source/target labels and relation text, speaker notes with slide/source-part provenance, inherited slide layout/master text and images, PowerPoint/Word/OpenDocument table cell coordinates/spans/merge state/styling, PowerPoint table-cell fills/borders/margins/text anchors, OpenDocument presentation slide-local tables/links/images, embedded ODF chart objects with axes/legends/source tables, ODF form controls with labels/selected values/options, ODF text-range and table-cell comments with anchor text/sheet/cell/header/value provenance, ODF tracked changes with inserted/deleted text, authors, dates, and revision graph edges, ODF styled spans/paragraph styles/bullet levels as emphasis graph evidence, Word drawing/VML text boxes, Word SmartArt/diagram nodes and edges, anchored DOCX/DOCM comments, anchored footnotes/endnotes, tables, workbook sheet visibility/source-part provenance, merged spreadsheet labels, sheet-local spreadsheet chart/link/image anchors, Excel table definitions, pivot-table source and row/column/value field edges, named ranges, spreadsheet data-validation rules, spreadsheet formula dependency edges, spreadsheet conditional-formatting rule edges, anchored classic/threaded spreadsheet comments, images, and chart title/axis/legend/data-label/series/category/value edges plus series/point visual cues and trendlines can still produce useful structured samples even when optional document libraries are not installed. Legacy binary Office files use best-effort OLE stream inventory and printable text extraction with stream path/checksum provenance and a conversion warning for full OOXML fidelity. EML messages surface subject/sender/recipient/date/message-id metadata, body text, body links, attachment objects, and parsed child evidence from attached CSV/Office/PDF/OpenDocument/EPUB/image/text files. PDFs can surface metadata, outlines, links, form fields, annotations, ruled tables, scanned-page OCR, and small embedded attachments, with page-relative readable bboxes and zones on links, forms, images, annotations, detected tables, and page OCR. ZIP entries, email attachments, and PDF attachments can themselves be parsed as Office/PDF/OpenDocument/EPUB/image/text artifacts, with child evidence carried into chunks, provenance, evidence profiles, interaction handoffs, and graph relationships.

**Search & Query:**
- `GET /api/v1/graphrag/query/intent` - Analyze query intent
- `POST /api/v1/graphrag/query/complexity` - Analyze reasoning complexity and routing requirements
- `POST /api/v1/graphrag/search/intelligent` - Intelligent search with auto method selection
- `POST /api/v1/graphrag/search` - Form-compatible upstream hybrid search adapter with local pipeline graph/chunk/path fallback when remote search is unavailable or a recent `pipeline_run_id` is supplied
- `POST /api/v1/graphrag/search/advanced` - Advanced search with explicit strategy

**Chat Context Fusion:**
- `POST /api/v1/graphrag/chat-context` - Build graph-enhanced context for chat queries
- `POST /api/v1/graphrag/chat-context-v2` - Build unified RAG plus graph context with cross references and quality signals

`chat-context-v2` now adds a `graph_reasoning` payload for relationship/path-style questions. It locally searches the fetched knowledge graph for direct relationships and multi-hop paths between query entities, includes those paths in the LLM context block, and reports path counts in `quality_signals`. Path scoring uses relation labels plus document provenance such as pages, slides, slide visibility/source parts, slide layouts, slide masters, PowerPoint shape/connector metadata, SmartArt diagram parts/node IDs/hierarchy/edge types, chart series/category/value/visual/trendline graph edges, chart source parts/extraction methods, tables, OCR, attachments, and archive entries, so source-location questions prefer evidence-grounded layout paths over generic semantic edges. Recent pipeline runs also contribute compact local graph and context artifacts plus an `interaction_handoff` with graph anchors, starter prompts, retrieval rehearsals, citation scopes, and Trace/Graph/Paths/Retrieval/Chat lanes. Chat consumes those handoffs as graph-reasoning sources, graph-augmented retrieval terms, evidence checklist cards, quality signals, and a `Pipeline Interaction Handoff` context section, so it can retrieve chunks and answer path/layout/chart/diagram/slide-visibility questions from a just-previewed or local-only upload even when `GRAPHRAG_URL` is not configured yet. The dedicated `/api/v1/graphrag/search` and `/api/v1/graphrag/reasoning/*` endpoints now use the same remembered traces as a fallback or explicit scope, returning local pipeline chunks, graph edges, inferred graph paths, ordered reasoning hops, evidence lists, and fallback warnings when remote GraphRAG is unavailable or `pipeline_run_ids`/`use_local_pipeline_context` are supplied. Retrieval rehearsals make the upload-to-chat handoff inspectable by showing which starter prompts matched graph anchors, run-scoped chunks, and provenance terms before a user ever asks the chatbot. After graph reasoning, it performs bounded graph-augmented retrieval using path nodes, relationship terms, provenance terms, handoff anchors, rehearsed prompt/chunk/anchor terms, and recent pipeline chunks, then deduplicates those chunks into the normal RAG context with `retrieval_reasons`, `retrieval_queries`, `matched_graph_path_terms`, `local_pipeline_*`, and `graph_augmented_*` quality signals.

**Advanced Reasoning:**
- `POST /api/v1/graphrag/reasoning/multi-hop` - Multi-hop reasoning
- `POST /api/v1/graphrag/reasoning/causal` - Causal reasoning
- `POST /api/v1/graphrag/reasoning/comparative` - Comparative analysis
- `POST /api/v1/graphrag/reasoning/advanced` - Advanced reasoning
- `POST /api/v1/graphrag/reasoning/explain` - Explain relationships

**Entity Operations:**
- `POST /api/v1/graphrag/entity/link` - Link entities to KG
- `POST /api/v1/graphrag/entity/disambiguate` - Disambiguate entities

**Knowledge Graph Exploration:**
- `GET /api/v1/graphrag/stats?domain=...` - Domain-aware graph statistics
- `POST /api/v1/graphrag/graph` - Filtered graph with entity/relationship type filters
- `GET /api/v1/graphrag/top-entities?entity_type=...` - Top entities with type/domain filters
- `GET /api/v1/graphrag/top-relationships?relationship_type=...` - Top relationships with type/domain filters
- `POST /api/v1/graphrag/entity-neighborhood` - One-hop neighborhood around a graph entity
- `GET /api/v1/graphrag/communities?domain=...` - Community/domain summaries from the knowledge graph
- `POST /api/v1/graphrag/export` - Filtered graph export

**NER, Relationship Extraction & Enhanced Query:**
- `POST /api/v1/graphrag/ner/extract` - Extract entities from text with GraphRAG GLiNER
- `POST /api/v1/graphrag/ner/extract-batch` - Batch entity extraction
- `GET /api/v1/graphrag/ner/model-info` - Inspect the configured NER model
- `GET /api/v1/graphrag/ner/entities` - List NER entity labels
- `GET /api/v1/graphrag/rel/status` - Relationship extraction model availability
- `GET /api/v1/graphrag/rel/model-info` - Inspect the configured relationship model
- `GET /api/v1/graphrag/rel/capabilities` - List relationship extraction capabilities and default schemas
- `POST /api/v1/graphrag/rel/extract` - Extract relationships from text with optional relation schema
- `POST /api/v1/graphrag/rel/extract-batch` - Batch relationship extraction with direct REL-service support
- `POST /api/v1/graphrag/rel/extract-entities` - Extract entities through the REL GLiNER service
- `POST /api/v1/graphrag/extract/enhanced` - Enhanced multi-method extraction with SpanBERT/dependency/linking toggles
- `POST /api/v1/graphrag/query/enhanced` - Enhanced graph-aware query processing
- `POST /api/v1/graphrag/query/enhanced-answer` - Enhanced answer generation with reasoning metadata
- `GET /api/v1/graphrag/query/statistics` - Enhanced-query processor statistics
- `GET /api/v1/graphrag/reasoning/graph-statistics` - Graph-reasoner statistics, centrality, and clusters

**Code Intelligence:**
- `POST /api/v1/graphrag/code/detect` - Detect code from text or files
- `POST /api/v1/graphrag/code/search` - Search code through Code-RAG when configured
- `POST /api/v1/graphrag/code/route` - Route code files to Code-RAG processing
- `POST /api/v1/graphrag/code/ingest` - Ingest repository file paths into Code-RAG
- `POST /api/v1/graphrag/code/upload` - Upload code files into Code-RAG
- `GET /api/v1/graphrag/code/ingest/{task_id}` - Poll Code-RAG ingestion progress
- `POST /api/v1/graphrag/code/analyze` - Analyze a single code file with Code-RAG parsers
- `GET /api/v1/graphrag/code/statistics` - Inspect Code-RAG index/task statistics
- `DELETE /api/v1/graphrag/code/index?confirm=true` - Clear the Code-RAG index with explicit confirmation

**Reddit Crawler:**
- `GET /api/v1/graphrag/reddit/health` - Check GraphRAG Reddit crawler service health
- `POST /api/v1/graphrag/reddit/crawl` - Start a Reddit crawl with optional GraphRAG auto-ingestion
- `GET /api/v1/graphrag/reddit/crawl/{crawl_id}/status` - Poll crawl progress
- `GET /api/v1/graphrag/reddit/crawl/{crawl_id}/results` - Fetch completed crawl results and ingestion status
- `DELETE /api/v1/graphrag/reddit/crawl/{crawl_id}` - Cancel an active crawl
- `GET /api/v1/graphrag/reddit/crawls` - List active and completed crawls

**Operational Controls:**
- `POST /api/v1/graphrag/process-document` - Preview document chunking without ingesting
- `POST /api/v1/graphrag/process-documents-batch` - Batch preview document chunking
- `POST /api/v1/graphrag/process-document-with-ner` - Preview document chunks with per-chunk entities
- `POST /api/v1/graphrag/vector-store/add` - Add documents to the vector store without graph building
- `GET /api/v1/graphrag/domain-stats` - Get per-domain graph statistics
- `POST /api/v1/graphrag/extraction/test-enhanced` - Run enhanced extraction diagnostics
- `DELETE /api/v1/graphrag/admin/clear-all?confirm=true` - Clear GraphRAG data with explicit confirmation

**Social Memory:**
- `GET /api/v1/graphrag/social-memory/metrics` - Prometheus metrics passthrough
- `GET /api/v1/graphrag/social-memory/health` - Social-memory store health
- `POST /api/v1/graphrag/social-memory/ingest` - Ingest conversational messages
- `GET /api/v1/graphrag/social-memory/context` - Retrieve personalized memory context
- `GET /api/v1/graphrag/social-memory/person/{person_id}/facts` - List known facts for a person
- `GET /api/v1/graphrag/social-memory/explain` - Explain retrieved memory facts
- `POST /api/v1/graphrag/social-memory/identities/merge` - Merge duplicate identities with admin auth
- `POST /api/v1/graphrag/social-memory/facts/correct` - Correct stored facts with admin auth
- `POST /api/v1/graphrag/social-memory/admin/migrate` - Run social-memory migrations
- `POST /api/v1/graphrag/social-memory/admin/replay` - Replay transcripts into a sandbox
- `DELETE /api/v1/graphrag/social-memory/admin/replay/{sandbox_tag}` - Purge replay sandboxes
- `POST /api/v1/graphrag/social-memory/admin/eval` - Run LoCoMo/vTuber memory eval suites
- `GET /api/v1/graphrag/social-memory/admin/community-report` - Generate community reports

**Capability Bridge:**
- `GET /api/v1/graphrag/capabilities` - GraphRAG capability catalog and service availability
- `ANY /api/v1/graphrag/upstream/{path}` - Allowlisted proxy for advanced GraphRAG endpoints such as `/evaluate/*`, `/test/*`, `/extract-entities-relations-enhanced`, `/api/query-complexity-analysis`, and `/social-memory/*`

The NER and REL extraction adapters prefer dedicated `NER_API_URL` and
`REL_API_URL` services when configured, then fall back to the main `GRAPHRAG_URL`
service while preserving a stable llama-nexus response shape.

**Feedback & Analytics:**
- `POST /api/v1/graphrag/feedback` - Record retrieval quality ratings and comments with optional quality signals
- `GET /api/v1/graphrag/analytics?days=30&domain=...` - Inspect satisfaction trends, domain breakdowns, and low-rated queries

**Evaluation & Test Suite:**
- `GET /api/v1/graphrag/evaluation/status` - Retrieval and chunking evaluation readiness
- `POST /api/v1/graphrag/evaluation/entity-extraction` - Evaluate extraction against ground truth
- `POST /api/v1/graphrag/evaluation/query-responses` - Evaluate answer quality against expected responses
- `GET /api/v1/graphrag/evaluation/graph-completeness` - Evaluate knowledge graph completeness
- `POST /api/v1/graphrag/evaluation/retrieval-relevance` - Evaluate retrieval relevance
- `POST /api/v1/graphrag/evaluation/comprehensive` - Run GraphRAG comprehensive evaluation
- `POST /api/v1/graphrag/evaluation/chunking` - Run WikiSection chunking evaluation
- `POST /api/v1/graphrag/evaluation/retrieval` - Run GutenQA retrieval evaluation
- `POST /api/v1/graphrag/evaluation/test-suite/{suite}` - Run unit, integration, performance, quality, or comprehensive GraphRAG tests

### Backend: 40 Workflow Executors
**File**: `backend/modules/workflow/executors/graphrag_executors.py`

1. **GraphRAGSearchExecutor** - Hybrid search with LLM answers
2. **EntityExtractionExecutor** - Extract entities using GLiNER
3. **MultiHopReasoningExecutor** - Find paths between entities
4. **CausalReasoningExecutor** - Identify cause-effect relationships
5. **ComparativeReasoningExecutor** - Compare entities
6. **EntityLinkingExecutor** - Link entities to knowledge graph
7. **GraphRAGDocumentIngestExecutor** - Ingest workflow-generated text into GraphRAG
8. **GraphRAGDocumentListExecutor** - List indexed GraphRAG documents from workflows
9. **GraphRAGDocumentDeleteExecutor** - Delete a named GraphRAG document from workflows
10. **GraphRAGKnowledgeGraphRebuildExecutor** - Rebuild the knowledge graph from indexed documents
11. **CodeDetectionExecutor** - Detect code in documents
12. **CodeSearchExecutor** - Search for code examples
13. **GraphRAGCodeIngestExecutor** - Start Code-RAG indexing for file paths
14. **GraphRAGCodeIngestionStatusExecutor** - Poll Code-RAG indexing progress
15. **GraphRAGCodeAnalysisExecutor** - Analyze a code file into entities and relationships
16. **GraphRAGCodeStatsExecutor** - Collect Code-RAG index/task statistics
17. **GraphRAGCapabilitiesExecutor** - Inspect exposed service and bridge capabilities
18. **GraphRAGQueryComplexityExecutor** - Classify reasoning requirements before routing
19. **EnhancedEntityExtractionExecutor** - Run enhanced extraction with SpanBERT, dependency, and linking toggles
20. **GraphRAGNERExtractionExecutor** - Run lower-level GLiNER NER over one or many texts
21. **GraphRAGRelationExtractionExecutor** - Extract typed relationships from one or many texts with optional schemas
22. **GraphRAGEnhancedQueryExecutor** - Use graph-aware enhanced query processing in workflows
23. **GraphRAGChatContextExecutor** - Build graph-enhanced or unified RAG plus graph context with quality signals
24. **GraphRAGInstrumentationExecutor** - Collect query stats, graph stats, centrality, and clusters
25. **GraphRAGEvaluationRunExecutor** - Run GraphRAG quality, retrieval, chunking, and comprehensive evaluations
26. **GraphRAGTestSuiteExecutor** - Run GraphRAG unit, integration, performance, quality, and comprehensive tests
27. **GraphRAGDomainStatsExecutor** - Collect per-domain graph statistics
28. **GraphRAGGraphExplorerExecutor** - Explore filtered graphs, entity neighborhoods, communities, top entities, top relationships, and exports
29. **GraphRAGEnhancedExtractionDiagnosticExecutor** - Run enhanced extraction diagnostics from workflows
30. **GraphRAGEvaluationStatusExecutor** - Check retrieval and chunking evaluation readiness
31. **GraphRAGFeedbackSubmitExecutor** - Record retrieval quality feedback from workflows
32. **GraphRAGFeedbackAnalyticsExecutor** - Fetch satisfaction trends, domain breakdowns, and problem queries
33. **GraphRAGSocialMemoryIngestExecutor** - Store conversation turns in GraphRAG social memory
34. **GraphRAGSocialMemoryContextExecutor** - Retrieve or explain personalized social-memory context
35. **GraphRAGSocialMemoryFactsExecutor** - List known facts for a resolved social-memory person
36. **GraphRAGSocialMemoryAdminExecutor** - Run migration, merge, correction, replay, eval, and community-report operations
37. **GraphRAGRedditCrawlExecutor** - Start Reddit crawls with optional GraphRAG auto-ingestion
38. **GraphRAGRedditCrawlStatusExecutor** - Poll Reddit crawl progress from workflows
39. **GraphRAGRedditCrawlResultsExecutor** - Fetch completed Reddit crawl results and ingestion status
40. **GraphRAGUpstreamExecutor** - Call allowlisted upstream GraphRAG endpoints from workflows

All registered in `backend/modules/workflow/executors/__init__.py`

### Frontend: 5 GraphRAG-Aware Pages

1. **IntelligentSearchPage** (`/intelligent-search`)
   - Unified search interface
   - Auto/Hybrid/Vector/Graph/Keyword search methods
   - LLM answer generation
   - Query intent analysis
   - Source attribution

2. **ReasoningPlaygroundPage** (`/reasoning`)
   - Multi-hop reasoning between entities
   - Causal analysis (causes and effects)
   - Comparative analysis (similarities/differences)
   - Relationship explanation

3. **GraphRAGWorkbenchPage** (`/graphrag`)
   - Capability, health, graph, Code-RAG, evaluation, and social-memory status
   - Query-complexity analysis plus enhanced graph-aware query execution and fused chat context
   - NER label discovery, relationship capabilities, enhanced extraction, batch REL, and extraction diagnostic runs
   - Knowledge-graph filtered graph, entity-neighborhood, top-entity, top-relationship, and community exploration
   - Corpus document listing, named document deletion, and knowledge-graph rebuild controls
   - GraphRAG quality suites, graph completeness, answer evaluation, feedback submission, and feedback analytics
   - Code-RAG ingestion/status/analysis and social-memory ingest/context/explain/facts/eval/report/migrate tools
   - GraphRAG Reddit crawler start/status/results/cancel/list controls
   - Allowlisted upstream bridge caller for advanced GraphRAG endpoints

4. **EntityManagerPage** (`/entities`)
   - Browse all entities
   - Filter by type and occurrence
   - View entity details
   - Confidence scores

5. **Enhanced DocumentsPage** (`/documents`)
   - Toggle between Local RAG and GraphRAG
   - Upload to GraphRAG with entity extraction
   - Real-time processing status

### Frontend: 37 New Node Types
**File**: `frontend/src/types/workflow.ts`

All GraphRAG nodes added with proper schemas, inputs, and outputs.

### Workflow Templates: 12 GraphRAG Templates
**File**: `backend/modules/workflow/templates.py`

1. **Intelligent Q&A with GraphRAG** - Simple search workflow
2. **Entity Extraction Pipeline** - Document → Extract → Link → Output
3. **Multi-Hop Investigation** - Find connections between concepts
4. **Causal Analysis Workflow** - Analyze cause-effect relationships
5. **Capability-Aware GraphRAG Query** - Inspect GraphRAG coverage, query complexity, and fused chat context before routing
6. **GraphRAG Observability & Quality** - Collect capabilities, domain stats, instrumentation, feedback analytics, evaluation readiness, and quality-test signals
7. **GraphRAG Extraction Diagnostics** - Compare NER, relationship extraction, enhanced extraction, and diagnostic extraction
8. **GraphRAG Social Memory Context** - Ingest a conversation turn, retrieve personalized context, and list resolved person facts
9. **GraphRAG Knowledge Graph Explorer** - Collect filtered graph data, entity neighborhoods, communities, top entities, and top relationships with graph filters
10. **GraphRAG Code-RAG Indexing** - Index code paths, poll progress, analyze code, and collect Code-RAG stats
11. **GraphRAG Reddit Research Ingestion** - Crawl Reddit discussions, auto-ingest into GraphRAG, and inspect crawl results
12. **GraphRAG Corpus Maintenance** - List indexed documents, rebuild the knowledge graph, and collect domain stats

### Frontend API
**File**: `frontend/src/services/api.ts`

- `uploadToGraphRAG()`
- `uploadBatchToGraphRAG()`
- `ingestGraphRAGText()`
- `previewGraphRAGIngestionPipeline()`
- `uploadGraphRAGDocumentWithPipeline()`
- `getGraphRAGIngestionPipelineRun()`
- `listGraphRAGDocuments()`
- `deleteGraphRAGDocument()`
- `rebuildGraphRAGKnowledgeGraph()`
- `exportGraphRAGGraph()`
- `analyzeQueryIntent()`
- `intelligentSearch()`
- `explainRelationship()`
- `getGraphRAGCapabilities()`
- `callGraphRAGUpstream()`
- `getGraphRAGEntityNeighborhood()`
- `getGraphRAGCommunities()`
- `extractGraphRAGNER()`
- `extractGraphRAGNERBatch()`
- `getGraphRAGNERModelInfo()`
- `getGraphRAGNEREntities()`
- `getGraphRAGRELStatus()`
- `getGraphRAGRELModelInfo()`
- `getGraphRAGRELCapabilities()`
- `extractGraphRAGRelations()`
- `extractGraphRAGRelationsBatch()`
- `enhancedGraphRAGQuery()`
- `getGraphRAGChatContext()`
- `enhancedGraphRAGAnswer()`
- `getGraphRAGQueryStatistics()`
- `getGraphRAGReasoningGraphStatistics()`
- `submitGraphRAGFeedback()`
- `getGraphRAGFeedbackAnalytics()`
- `processGraphRAGDocument()`
- `processGraphRAGDocumentsBatch()`
- `processGraphRAGDocumentWithNER()`
- `addDocumentsToGraphRAGVectorStore()`
- `getGraphRAGDomainStats()`
- `routeGraphRAGCodeFile()`
- `ingestGraphRAGCodePaths()`
- `uploadGraphRAGCodeFiles()`
- `getGraphRAGCodeIngestionStatus()`
- `analyzeGraphRAGCodeFile()`
- `getGraphRAGCodeStatistics()`
- `clearGraphRAGCodeIndex()`
- `getGraphRAGRedditCrawlerHealth()`
- `startGraphRAGRedditCrawl()`
- `getGraphRAGRedditCrawlStatus()`
- `getGraphRAGRedditCrawlResults()`
- `cancelGraphRAGRedditCrawl()`
- `listGraphRAGRedditCrawls()`
- `testGraphRAGEnhancedExtraction()`
- `clearGraphRAGData()`
- `analyzeGraphRAGQueryComplexity()`
- `enhancedGraphRAGExtraction()`
- `getGraphRAGEvaluationStatus()`
- `getGraphRAGRetrievalEvaluationStatus()`
- `getGraphRAGChunkingEvaluationStatus()`
- `evaluateGraphRAGEntityExtraction()`
- `evaluateGraphRAGQueryResponses()`
- `evaluateGraphRAGGraphCompleteness()`
- `evaluateGraphRAGRetrievalRelevance()`
- `evaluateGraphRAGComprehensive()`
- `evaluateGraphRAGChunking()`
- `evaluateGraphRAGChunkingComparative()`
- `evaluateGraphRAGRetrieval()`
- `evaluateGraphRAGRetrievalComparative()`
- `runGraphRAGTestSuite()`
- `getGraphRAGTestReports()`
- `getGraphRAGSocialMemoryMetrics()`
- `getGraphRAGSocialMemoryHealth()`
- `migrateGraphRAGSocialMemory()`
- `ingestGraphRAGSocialMemoryMessage()`
- `getGraphRAGSocialMemoryContext()`
- `getGraphRAGSocialMemoryPersonFacts()`
- `explainGraphRAGSocialMemoryRetrieval()`
- `mergeGraphRAGSocialMemoryIdentities()`
- `correctGraphRAGSocialMemoryFact()`
- `replayGraphRAGSocialMemoryTranscript()`
- `purgeGraphRAGSocialMemoryReplay()`
- `runGraphRAGSocialMemoryEval()`
- `getGraphRAGSocialMemoryCommunityReport()`
- Plus the graph, entity, reasoning, hybrid processing, and service-status methods used by the UI

### Tests

1. **test_graphrag_integration.py** - Unit tests with mocks (~15 tests)
2. **test_graphrag_executors.py** - Executor-specific tests (~20 tests)
3. **test_graphrag_e2e.py** - End-to-end tests against live service
4. **test_graphrag_integration.py** (root) - Quick verification script
5. **test_graphrag_helpers.py** - Response normalization helpers
6. **test_graphrag_proxy_adapters.py** - Upstream request-shape adapters and bridge allowlist
7. **test_graphrag_advanced_workflow_executors.py** - Capability, complexity, enhanced extraction, evaluation, and upstream workflow executors
8. **test_graphrag_evaluation_adapters.py** - Typed evaluation/test-suite route adapters
9. **test_graphrag_extraction_adapters.py** - NER, REL, enhanced-query, and instrumentation adapters
10. **test_graphrag_operations_adapters.py** - Document processing, vector-store, domain stats, Code-RAG routing, diagnostics, and guarded clear-all adapters
11. **test_graphrag_graph_adapters.py** - Graph stats, entity/relationship filter, and export adapters
12. **test_graphrag_code_rag_adapters.py** - Code-RAG ingestion, upload, status, analysis, statistics, and guarded index adapters
13. **test_graphrag_social_memory_adapters.py** - Typed social-memory route adapters and admin header forwarding
14. **test_graphrag_reddit_crawler_adapters.py** - Typed Reddit crawler route adapters
15. **test_workflow_templates.py** - Template registry, node-type, and connection-port validation

---

## How to Use

### 1. Restart Backend to Register New Routes
```bash
cd /home/alec/git/llama-nexus
docker compose restart backend-api
```

### 2. Start the Bundled Helper Stack
```bash
# Neo4j + NER + REL + Code-RAG helpers
GRAPHRAG_ENABLED=true docker compose --profile graphrag up -d

# Target individual helper profiles when needed
GRAPHRAG_ENABLED=true docker compose --profile graphrag-ner up -d
GRAPHRAG_ENABLED=true docker compose --profile graphrag-rel up -d
GRAPHRAG_ENABLED=true docker compose --profile graphrag-code up -d
```

When the backend runs inside Docker with `GRAPHRAG_ENABLED=true`, blank helper URLs default to Docker service names:

- `NER_API_URL=http://graphrag-ner:8001`
- `REL_API_URL=http://graphrag-rel:8002`
- `CODE_RAG_URL=http://graphrag-code:8000`

Set `GRAPHRAG_URL` when graph search, document ingestion, graph exploration, reasoning, evaluation, or social-memory routes should call an external GraphRAG API.

`/api/v1/graphrag/services/status` includes live service checks plus workflow readiness for local pipeline preview, remote GraphRAG ingest, graph search/reasoning, extraction, Code-RAG, and Reddit crawling. It also returns endpoint source provenance so operators can tell whether each URL came from an explicit env var or a Docker compose default, plus an `interaction_contract` with the active Document Pipeline, Upload, Chat, and Search behavior. The document-pipeline contract reports supported formats, evidence/stage outputs, the preview sample cap, representative relationship sample mode, OCR readiness, and a machine-readable evidence capability matrix for presentation structure, tables, charts, diagrams, shapes/connectors, link/action evidence, email message evidence, legacy Office OLE stream evidence, forms, styled text, spreadsheet lineage, layout/OCR, and source artifacts; the upload contract includes adaptive ingest endpoint order, adaptive health preflight order (`/health`, `/api/health`, `/api/v1/health`), the selected health preflight URL, endpoint-verification mode, endpoint source, smoke-test probes, and the trace fields that record attempted URLs, endpoint attempts, selected endpoints, selected health endpoints, handoff graph anchors, citation scopes, and link relationship counts when `GRAPHRAG_URL` is configured or local-only traces are used. The `operator_checklist` turns the current mode into ordered setup, upload trace, chat handoff, remote ingest, graph search, and helper-extraction actions, while `deployment_quickstart` returns command-ready runbook cards for local trace/chat, helper extraction, remote GraphRAG ingest/search, and OCR runtime setup. The Workbench uses this to show which workflows are ready, partial, local-only, or blocked during staged deployments, with quickstart commands, env/profile chips, evidence-family cards for document KG capabilities such as Links & Actions before a file is uploaded, and adaptive health-path/attempt probes so remote GraphRAG endpoint mismatches can be diagnosed from the readiness panel.

Pipeline preview/upload traces are remembered through `GRAPHRAG_PIPELINE_RUNS_FILE`. In Docker this defaults to `/data/graphrag/pipeline_runs.json`, so recent extraction, chunking, graph-candidate, embedding-plan, preflight, evidence-profile, and upstream upload samples survive backend restarts. Each run now includes `pipeline_performance` for stage timings, slowest-stage detection, local/remote duration splits, and throughput, plus `deployment_outcome` for health preflight, compatible ingest attempts, selected endpoint, fallback use, operator action, and service-smoke links. Each remembered run also exposes an artifact manifest at `/api/v1/graphrag/ingest/pipeline/{run_id}/artifacts`, with export endpoints for structured ingest Markdown, local graph, local context, graph reasoning, interaction handoff, stages, samples, profile, summary, and full run JSON. Recent-run summaries include a compact `resume_handoff` with the scoped `chat-context-v2` request, Chat route state, starter prompts, citation scopes, available artifact links, and ready actions, so the Workbench can jump directly from a remembered run into graph-scoped chat without loading the full trace first. The Workbench recent-run cards now surface deployment outcome, endpoint retry/fallback state, total trace time, slowest stage, chat handoff readiness, recommended interaction mode, graph-anchor counts, retrieval-rehearsal counts, rehearsed chunk/anchor hits, citation scopes, ready actions, and ready Trace/Graph/Paths/Retrieval/Chat lanes before a user opens the full JSON. The evidence profile summarizes archive entries, email message body parts, embedded attachments, OCR blocks, distinct tables, form fields, layout-bearing elements, workbook sheets and hidden sheets, presentation slides and hidden slides, pivot tables, PowerPoint/OpenDocument shapes and connectors, chart series/category/value graph edges, table-derived facts with table-cell visual cues, and layout/path relationships for quick inspection in the GraphRAG Workbench. Representative chunk and embedding cards now include multi-label evidence-family arrays plus embedding evidence-family counts, so one mixed chunk can prove coverage for message, table, styled text, diagram, image, comment, revision, media, OCR, link, and chart evidence even when a five-sample preview has to stay compact. Layout-aware traces expose `layout` as a detected capability, capability card, starter prompt, and citation scope for bbox/zone review; EML traces expose `messages` as a detected capability, Message evidence card, `Trace message` starter prompt, sender/recipient/body/link/attachment graph relationships, and a `message` citation scope; PPTX and ODP deck traces also expose `presentation_structure` as a detected capability, capability card, starter prompt, and citation scope for slide visibility/source-part review, shapes/connectors expose evidence cards and starter prompts for diagram audit, link/action evidence exposes link graph anchors, a Link evidence capability card, `Trace links` starter prompt, URI/image relationship provenance, and a `link` citation scope, styled text exposes emphasized-term graph anchors and term-specific starter prompts, and any extracted forms expose a form evidence card, form-field starter prompt, and `form field` citation scope. The trace Ask Chat action carries a structured route handoff into the chatbot with run ids, starter prompts, recommended reasoning mode, answer guidance, citation scopes, capability cards, graph-anchor counts, retrieval-rehearsal counts, and the backend-prepared `chat_context_request`; graph-path starter chips preserve `source`, `target`, `bridge`, and `relation` hints from the pipeline trace through Workbench route state and Chat payload construction into `chat-context-v2`, where those hints bias local/remote path scoring and direct-relationship retrieval. The Query tab also echoes its `chat_context_request` and explicitly sends `use_local_pipeline_context`/`include_local_pipeline` when recent traces are selected, making local-context builds reproducible from the JSON result. Chat renders those as compact trace controls, preserves the user's live question, forwards the scoped request fields to the handoff's known GraphRAG context endpoint, and falls back from `chat-context-v2` to `chat-context` when unified context returns no usable text. Batch upload summaries roll those evidence totals up across files and include an `upload_manifest` with per-file trace/chat/remote readiness, deployment status/severity, endpoint attempt counts, fallback flags, layout-region counts, payload kinds, byte totals, Upload/Trace/Chat/Remote Graph lanes, and the recommended next action.

Each remembered pipeline run stores capped `local_graph` and `local_context` artifacts built from graph candidates and chunk text. Chat graph reasoning merges recent local nodes and edges with remote GraphRAG data when available, falls back to them when the graph API is missing, and lexically retrieves recent local chunks using query and graph-derived terms. Scoped Ask Chat prompts can also promote query-matched local graph edges directly into `graph_reasoning.direct_relationships` without remote NER, so broad trace actions such as link/action review still produce citation-ready URI, image-relationship, and layout evidence instead of only a graph overview. Local graph path candidates now include evidence-family labels, provenance summaries, explanations, and answer hints, and those rationales are copied into local context/embedding text so graph-path answers can explain why two entities connect rather than only listing a bridge node. `chat-context-v2` preserves the same explainability contract when it recomputes remote or local graph paths, adding evidence families, path explanations, answer hints, and hop rationales to `graph_reasoning.paths`, `answer_plan.primary_path`, answer-plan context text, evidence cards, and graph-augmented retrieval terms. Chat renders that answer plan as a compact Graph Answer Plan panel with primary-path explanations, evidence-family chips, hop rationales, and citation labels next to the retrieved chunks. The answer plan includes an evidence checklist of citation-ready graph hop cards with provenance such as file, slide/page/sheet, slide visibility, slide source part, sheet visibility, workbook source part, chart source part/extraction method, table, shape, connector, SmartArt/OpenDocument diagram node/edge metadata, styled-text emphasis/style summaries, clicked-image link relationship IDs, message subject/sender/recipient/body-part/attachment headers, attachment, archive entry, OCR source, and evidence text; hidden-slide and hidden-sheet graph paths keep visibility, package source part, and relationship ID in both location metadata and citation labels. Spreadsheet lineage paths also carry formula cells, conditional-format ranges, named-range refs, validation ranges, pivot-table ranges, pivot sources, grouping fields, and aggregate metadata into answer-plan locations and retrieval terms. Diagram paths carry diagram package parts, node IDs/types/order/level, parent labels, edge endpoints, and connection types into the same answer-plan locations, citation labels, frontend reasoning location text, and graph-augmented retrieval terms. Message paths carry subject, sender, recipients, copied recipients, message id, sent date, body part kind/content type, and email attachment filenames into answer-plan locations, citation labels, evidence families, and graph-augmented retrieval terms. The response marks these sources through `graph_reasoning.graph_sources`, `graph_reasoning.answer_plan.evidence_cards`, `quality_signals.local_pipeline_graph_*`, and `quality_signals.local_pipeline_context_*`.

### 3. Verify Integration
```bash
# Quick test
python3 test_graphrag_integration.py

# Or run full test suite
cd backend
pytest tests/test_graphrag_integration.py tests/test_graphrag_executors.py -v
```

### 4. Upload Documents to GraphRAG
- Go to **Documents** page
- Click "Add Document"
- Select "GraphRAG (Neo4j + Knowledge Graph)" destination
- Upload file → Entities automatically extracted

### 5. Use Intelligent Search
- Go to **Intelligent Search** page
- Enter natural language query
- Select search method (Auto recommended)
- Get LLM-generated answer with sources

### 6. Explore Reasoning
- Go to **Reasoning Playground** page
- Try Multi-Hop: Find connections between entities
- Try Causal: Understand cause-effect relationships
- Try Comparative: Compare two entities

### 7. Operate GraphRAG Services
- Go to **GraphRAG Workbench** (`/graphrag`)
- Inspect GraphRAG, NER, REL, Code-RAG, Reddit crawler, evaluation, and social-memory readiness
- Run enhanced query, extraction, quality, Code-RAG, Reddit crawl, and memory operations

### 8. Build GraphRAG Workflows
- Go to **Workflow Builder**
- Drag GraphRAG search, corpus maintenance, capability, complexity, enhanced extraction, Code-RAG, Reddit crawler, evaluation, or upstream bridge nodes
- Configure search, corpus, extraction, crawling, evaluation, or bridge parameters
- Connect to inputs/outputs
- Run automated knowledge workflows

### 9. Manage Entities
- Go to **Entity Manager** page
- Browse extracted entities
- Filter by type and occurrence
- View entity relationships

---

## API Endpoints Quick Reference

### Document Management
```bash
# Upload to GraphRAG
POST /api/v1/graphrag/ingest/upload
  - file: document file
  - domain: string (default: "general")
  - use_semantic_chunking: bool (default: true)
  - build_knowledge_graph: bool (default: true)

# Batch upload
POST /api/v1/graphrag/ingest/batch
  - files: array of files
  - domain: string

# Rebuild KG
POST /api/v1/graphrag/rebuild
  - domain: optional string
```

### Search
```bash
# Intelligent search
POST /api/v1/graphrag/search/intelligent
  {
    "query": "string",
    "search_type": "auto|hybrid|vector|graph|keyword",
    "top_k": 10,
    "domain": "optional"
  }

# Analyze query
GET /api/v1/graphrag/query/intent?query=your+query
```

### Reasoning
```bash
# Multi-hop
POST /api/v1/graphrag/reasoning/multi-hop
  {"query": "string", "source": "optional", "target": "optional", "max_hops": 3, "pipeline_run_ids": ["optional"], "include_local_pipeline": true}

# Causal
POST /api/v1/graphrag/reasoning/causal
  {"query": "string", "source": "optional", "target": "optional", "pipeline_run_ids": ["optional"], "include_local_pipeline": true}

# Comparative
POST /api/v1/graphrag/reasoning/comparative
  {"query": "Compare X and Y", "source": "X", "target": "Y", "pipeline_run_ids": ["optional"], "include_local_pipeline": true}

# Advanced
POST /api/v1/graphrag/reasoning/advanced
  {"query": "string", "source": "optional", "target": "optional", "pipeline_run_ids": ["optional"], "include_local_pipeline": true}

# Explain relationship
POST /api/v1/graphrag/reasoning/explain
  {"source": "entity1", "target": "entity2", "pipeline_run_ids": ["optional"], "include_local_pipeline": true}
```

### Entity Operations
```bash
# Extract entities
POST /api/v1/graphrag/extract
  {"text": "string", "domain": "string"}

# Link entities
POST /api/v1/graphrag/entity/link
  {"entities": ["array"], "context": "string"}

# Disambiguate
POST /api/v1/graphrag/entity/disambiguate
  {"entity_name": "string", "context": "string"}
```

### NER, Relationship Extraction & Enhanced Query
```bash
# Raw GLiNER entity extraction
POST /api/v1/graphrag/ner/extract
  {"text": "string", "return_offsets": true, "return_scores": true}

# Batch GLiNER entity extraction
POST /api/v1/graphrag/ner/extract-batch
  {"texts": ["string"], "return_offsets": true, "return_scores": true}

# NER model metadata and labels
GET /api/v1/graphrag/ner/model-info
GET /api/v1/graphrag/ner/entities

# Relationship extraction
GET /api/v1/graphrag/rel/status
GET /api/v1/graphrag/rel/model-info
GET /api/v1/graphrag/rel/capabilities
POST /api/v1/graphrag/rel/extract
  {"text": "string", "relations": [], "entity_labels": [], "threshold": 0.5}
POST /api/v1/graphrag/rel/extract-batch
  {"texts": ["string"], "relations": [], "entity_labels": [], "threshold": 0.5}
POST /api/v1/graphrag/rel/extract-entities
  {"text": "string", "labels": [], "threshold": 0.5}

# Enhanced query processing
POST /api/v1/graphrag/query/enhanced
  {"query": "string", "search_type": "hybrid", "top_k": 10}
POST /api/v1/graphrag/query/enhanced-answer
  {"query": "string", "context": {}}

# Query and graph instrumentation
GET /api/v1/graphrag/query/statistics
GET /api/v1/graphrag/reasoning/graph-statistics
```

### Reddit Crawler
```bash
# Requires REDDIT_CRAWLER_URL or GRAPHRAG_REDDIT_CRAWLER_URL
GET /api/v1/graphrag/reddit/health

# Start a crawl; auto_ingest lets the GraphRAG crawler ingest results itself
POST /api/v1/graphrag/reddit/crawl
  {
    "query": "local LLM retrieval quality",
    "subreddits": ["LocalLLaMA", "MachineLearning"],
    "max_posts": 25,
    "auto_ingest": true
  }

# Crawl lifecycle
GET /api/v1/graphrag/reddit/crawl/{crawl_id}/status
GET /api/v1/graphrag/reddit/crawl/{crawl_id}/results
DELETE /api/v1/graphrag/reddit/crawl/{crawl_id}
GET /api/v1/graphrag/reddit/crawls
```

### Operational Controls
```bash
# Preview document processing without ingesting
POST /api/v1/graphrag/process-document
  - multipart/form-data: file, use_semantic_chunking

# Preview batch document processing
POST /api/v1/graphrag/process-documents-batch
  - multipart/form-data: files[], use_semantic_chunking

# Preview document processing with GLiNER entities attached to chunks
POST /api/v1/graphrag/process-document-with-ner
  - multipart/form-data: file, use_semantic_chunking, extract_entities

# Add to vector store only
POST /api/v1/graphrag/vector-store/add
  - multipart/form-data: files[]

# Domain stats and diagnostics
GET /api/v1/graphrag/domain-stats
POST /api/v1/graphrag/code/route
POST /api/v1/graphrag/extraction/test-enhanced

# Guarded destructive operation
DELETE /api/v1/graphrag/admin/clear-all?confirm=true
```

### Capability Bridge
```bash
# Capability catalog
GET /api/v1/graphrag/capabilities

# Enhanced extraction via upstream bridge
POST /api/v1/graphrag/upstream/extract-entities-relations-enhanced
  - multipart/form-data: text, domain, use_spanbert, use_dependency, use_entity_linking

# Query complexity analysis
POST /api/v1/graphrag/upstream/api/query-complexity-analysis?query=your+query

# Evaluation readiness
GET /api/v1/graphrag/upstream/evaluate/retrieval/status
GET /api/v1/graphrag/upstream/evaluate/chunking/status
```

### Evaluation & Test Suite
```bash
# Combined readiness
GET /api/v1/graphrag/evaluation/status

# Classic evaluation framework
POST /api/v1/graphrag/evaluation/entity-extraction
POST /api/v1/graphrag/evaluation/query-responses
GET  /api/v1/graphrag/evaluation/graph-completeness
POST /api/v1/graphrag/evaluation/retrieval-relevance
POST /api/v1/graphrag/evaluation/comprehensive

# Benchmark datasets
POST /api/v1/graphrag/evaluation/chunking
POST /api/v1/graphrag/evaluation/chunking/comparative
POST /api/v1/graphrag/evaluation/retrieval
POST /api/v1/graphrag/evaluation/retrieval/comparative

# GraphRAG internal test suite
POST /api/v1/graphrag/evaluation/test-suite/unit
POST /api/v1/graphrag/evaluation/test-suite/integration
POST /api/v1/graphrag/evaluation/test-suite/performance
POST /api/v1/graphrag/evaluation/test-suite/quality
POST /api/v1/graphrag/evaluation/test-suite/comprehensive
GET  /api/v1/graphrag/evaluation/test-suite/reports
```

---

## Architecture

```
llama-nexus (localhost:8700)
  ├── Backend GraphRAG Route Package (/api/v1/graphrag/*)
  │   ├── Typed adapters for stable UI/workflow calls
  │   ├── NER, relationship extraction, enhanced-query, and instrumentation adapters
  │   ├── Document processing, vector-store, diagnostic, and guarded admin adapters
  │   ├── Code-RAG, Reddit crawler, and social-memory direct-service adapters
  │   ├── Evaluation and GraphRAG test-suite adapters
  │   └── Allowlisted upstream bridge for advanced GraphRAG features
  ├── Workflow Executors (37 GraphRAG nodes)
  │   └── Call backend proxy routes
  ├── Frontend Pages (5 GraphRAG-aware pages)
  │   └── Call backend API
  └── Workflow Templates (12 GraphRAG templates)

GraphRAG Service (localhost:18000)
  ├── Neo4j (localhost:7687) - Knowledge graph storage
  ├── Qdrant (localhost:16333) - Vector embeddings
  ├── NER API - GLiNER entity extraction
  ├── Relationship API - Relationship extraction
  ├── Reddit Crawler - Reddit crawl and optional auto-ingestion
  └── Hybrid Retrieval - Vector + Graph + Keyword
```

---

## Testing

### Quick Verification
```bash
python3 test_graphrag_integration.py
```

### Unit Tests (with mocks)
```bash
cd backend
pytest tests/test_graphrag_integration.py -v
pytest tests/test_graphrag_executors.py -v
pytest tests/test_graphrag_advanced_workflow_executors.py -v
pytest tests/test_graphrag_evaluation_adapters.py -v
pytest tests/test_graphrag_extraction_adapters.py -v
pytest tests/test_graphrag_operations_adapters.py -v
pytest tests/test_workflow_templates.py -v
```

### E2E Tests (requires live GraphRAG)
```bash
cd backend
pytest tests/test_graphrag_e2e.py -v --graphrag-live
```

### Manual Pipeline Smoke (live llama-nexus stack)
Use the helper script for the common path:
```bash
scripts/graphrag_pipeline_smoke.sh --domain manual-smoke /path/to/document.pptx
```
For small/simple files where a multi-hop path may not exist yet, set
`--allow-no-paths` or `GRAPHRAG_REQUIRE_PATHS=false` (or `0`/`no`/`off`) to
validate upload, artifacts, and chat context without requiring graph paths.
For staged local-only checks before remote ingest is configured, also set
`--local-only` or `GRAPHRAG_UPLOAD_TO_REMOTE=false` (or `0`/`no`/`off`).
For interactive chat after upload, scoped `pipeline_run_ids` are still the
strongest handoff, but graph/path/relationship questions without run IDs now
probe recent matching-domain pipeline traces so domain-level testing stays
smooth.

Manual Workbench path:
1. Open `http://localhost:3002/graphrag` and confirm **Service Smoke Check** reports ready workflows with no warnings.
2. In **Corpus**, choose a document and use **Preview Trace** for local stage samples or **Upload with Trace** for the full remote-ingest path.
3. Inspect the remembered run for stage samples, artifacts, graph anchors, retrieval rehearsals, and citation scopes.
4. Use **Ask Chat** from the run, then ask a follow-up graph/path question in the same domain to verify recent pipeline traces are reused.

Or run the individual requests:
```bash
# 1. Confirm the browser-facing GraphRAG contract is ready.
curl -fsS http://localhost:3002/api/v1/graphrag/services/status \
  | jq -r '._meta | "ready=\(.ready) operational_ready=\(.operational_ready) warnings=\((.warnings // []) | length)"'

# 2. Upload a document through the same proxy path used by the Workbench.
RUN_ID=$(
  curl -fsS http://localhost:3002/api/v1/graphrag/ingest/pipeline/upload \
    -F "file=@/path/to/document.pptx" \
    -F "domain=manual-smoke" \
    -F "emit_samples=true" \
    -F "submit_structured_text=true" \
  | jq -r '.run_id'
)

# 3. Inspect remembered artifacts and the chat handoff.
curl -fsS "http://localhost:3002/api/v1/graphrag/ingest/pipeline/${RUN_ID}" \
  | jq '{run_id, status, artifacts: .artifacts.artifact_count, resume_ready: .resume_handoff.ready, chunks: .summary.chunk_count, relationships: .summary.relationship_candidate_count}'

# 4. Ask a graph/path-style question scoped to the new run.
curl -fsS http://localhost:3002/api/v1/graphrag/chat-context-v2 \
  -H 'Content-Type: application/json' \
  --data "{\"query\":\"Which graph path connects two important entities in this document, and what provenance should I cite?\",\"pipeline_run_ids\":[\"${RUN_ID}\"],\"domain\":\"manual-smoke\",\"include_graph\":true,\"use_local_pipeline_context\":true}" \
  | jq '{chunks: (.chunks | length), graph_status: .graph_reasoning.status, graph_paths: (.graph_reasoning.paths | length), local_graph: .services_used.local_pipeline_graph, errors}'
```

### Test a Workflow Executor
```python
from modules.workflow.executors.graphrag_executors import GraphRAGSearchExecutor

executor = GraphRAGSearchExecutor(node_id="test", config={"searchType": "auto"})
result = await executor.execute({"query": "test"}, context)
```

---

## Next Steps

1. **Restart Backend**: `docker compose restart backend-api`
2. **Upload Test Documents**: Use Documents page → GraphRAG destination
3. **Try Intelligent Search**: Query your documents with natural language
4. **Open GraphRAG Workbench**: Use `/graphrag` for extraction, quality, Code-RAG, and memory operations
5. **Build Workflows**: Use new GraphRAG nodes in Workflow Builder
6. **Explore Reasoning**: Use Reasoning Playground to find insights

---

## Troubleshooting

### New Routes Return 404
**Problem**: New endpoints not found
**Solution**: Restart backend service to register new routes

### GraphRAG Service Unavailable
**Problem**: Cannot connect to GraphRAG
**Solution**: Start the bundled helpers, and set `GRAPHRAG_URL` if you need graph/search/ingest/reasoning routes.
```bash
cd /home/alec/git/llama-nexus
GRAPHRAG_ENABLED=true docker compose --profile graphrag up -d

# Optional external graph API
GRAPHRAG_ENABLED=true GRAPHRAG_URL=http://host.docker.internal:18000 docker compose --profile graphrag up -d
```

`/api/v1/graphrag/health` may report `partial` when NER, REL, or Code-RAG is reachable but `GRAPHRAG_URL` is not. In that state extraction/code features can work, while graph exploration and graph search remain unavailable.

### Empty Search Results
**Problem**: Search returns no results
**Solution**: Upload documents first through Documents page

### Entity Extraction Returns Empty
**Problem**: No entities found
**Solution**: Ensure NER service is running (check with `/api/v1/graphrag/ner/status`)

---

## Performance Expectations

- **Document Upload**: 2-10 seconds per page
- **Entity Extraction**: 3-15 seconds depending on text length
- **Search**: < 2 seconds for most queries
- **Multi-Hop Reasoning**: 3-10 seconds depending on depth
- **Causal/Comparative**: 5-20 seconds (uses LLM)

---

## Files Created/Modified

### Backend
- `backend/routes/graphrag/` - split proxy, adapter, context, feedback, and bridge route modules
- `backend/routes/graphrag/extraction.py` - typed NER, REL, enhanced query, and query/graph statistics adapters
- `backend/routes/graphrag/evaluation.py` - typed evaluation and GraphRAG test-suite adapters
- `backend/routes/graphrag/operations.py` - typed processing previews, vector-store, diagnostics, Code-RAG routing, domain stats, and guarded admin adapters
- `backend/routes/graphrag/reddit.py` - typed GraphRAG Reddit crawler adapters
- `backend/modules/graphrag/config.py` - unified GraphRAG service configuration including Reddit crawler URL
- `backend/modules/workflow/executors/graphrag_executors.py` - 37 GraphRAG workflow executors
- `backend/modules/workflow/executors/__init__.py` - Registration
- `backend/modules/workflow/templates.py` - 12 GraphRAG workflow templates
- `backend/tests/test_graphrag_integration.py` - Unit tests
- `backend/tests/test_graphrag_executors.py` - Executor tests
- `backend/tests/test_graphrag_e2e.py` - E2E tests
- `backend/tests/test_graphrag_helpers.py` - Normalization helper tests
- `backend/tests/test_graphrag_proxy_adapters.py` - Adapter and upstream bridge tests
- `backend/tests/test_graphrag_advanced_workflow_executors.py` - Advanced workflow executor tests
- `backend/tests/test_graphrag_evaluation_adapters.py` - Evaluation adapter tests
- `backend/tests/test_graphrag_extraction_adapters.py` - Extraction and enhanced-query adapter tests
- `backend/tests/test_graphrag_operations_adapters.py` - Operations adapter tests
- `backend/tests/test_graphrag_graph_adapters.py` - Graph adapter tests
- `backend/tests/test_graphrag_code_rag_adapters.py` - Code-RAG adapter tests
- `backend/tests/test_graphrag_social_memory_adapters.py` - Social-memory adapter tests
- `backend/tests/test_graphrag_reddit_crawler_adapters.py` - Reddit crawler adapter tests
- `backend/tests/test_workflow_templates.py` - Workflow template validity tests
- `backend/tests/conftest.py` - Test fixtures
- `backend/tests/README_GRAPHRAG_TESTS.md` - Test documentation

### Frontend
- `frontend/src/pages/IntelligentSearchPage.tsx` - New page
- `frontend/src/pages/ReasoningPlaygroundPage.tsx` - New page
- `frontend/src/pages/GraphRAGWorkbenchPage.tsx` - GraphRAG operations workbench
- `frontend/src/pages/EntityManagerPage.tsx` - New page
- `frontend/src/pages/DocumentsPage.tsx` - Enhanced
- `frontend/src/services/api.ts` - GraphRAG API, capability, operations, extraction, enhanced-query, bridge, and evaluation helpers
- `frontend/src/types/workflow.ts` - 37 GraphRAG node types
- `frontend/src/config/navigation.tsx` - GraphRAG page route and navigation metadata

### Tests & Docs
- `test_graphrag_integration.py` - Quick test script
- `GRAPHRAG_INTEGRATION_SUMMARY.md` - This file

This list covers the main GraphRAG integration files; the repository may contain additional unrelated local changes.

---

## Success!

llama-nexus is now a full-featured UI client for GraphRAG with:
- Document upload and processing
- Intelligent hybrid search
- Advanced reasoning capabilities
- Entity management
- Workflow automation
- Executable GraphRAG workflow templates for QA, extraction, reasoning, observability, quality, and diagnostics
- First-class NER, relationship extraction, enhanced-query, and graph instrumentation workflow nodes
- Workflow-native GraphRAG evaluation runs, test suites, domain stats, and extraction diagnostics
- Typed document-processing previews, vector-store-only ingestion, Code-RAG routing, diagnostics, and guarded data clearing
- Capability discovery, enhanced extraction, evaluation status, and an allowlisted upstream workflow bridge
- Comprehensive testing

**Restart the backend and start exploring!**
