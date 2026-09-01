"""
GraphRAG Routes Package

Splits the monolithic graphrag.py into focused modules:
- health.py  — Health checks, status, stats (9 endpoints)
- graph.py   — Knowledge graph queries (6 endpoints)
- search.py  — Search & reasoning (11 endpoints)
- ingest.py  — Document upload & entity management (10 endpoints)
- code.py — Code-RAG ingestion, upload, task status, analysis, statistics, and index controls
- extraction.py — NER, relationship extraction, enhanced query, and query instrumentation
- evaluation.py — Evaluation framework and GraphRAG test-suite adapters
- operations.py — Document processing previews, vector-store operations, and guarded admin tools
- social_memory.py — Conversational social memory ingest, context, explain, correction, eval, and reports
- reddit.py — GraphRAG Reddit crawler lifecycle and auto-ingestion adapters
- context.py — Chat context generation (2 endpoints: v1, v2)
- feedback.py — Retrieval quality feedback & analytics
- upstream.py — Capability catalog and allowlisted upstream bridge
- helpers.py — Shared config access, HTTP client, utilities

All configuration is sourced from the unified GraphRAGConfig singleton
in modules/graphrag/config.py.  All HTTP calls share a connection-pooled
httpx.AsyncClient managed by helpers.get_http_client().
"""

from fastapi import APIRouter

from .health import router as health_router
from .graph import router as graph_router
from .search import router as search_router
from .ingest import router as ingest_router
from .code import router as code_router
from .extraction import router as extraction_router
from .evaluation import router as evaluation_router
from .operations import router as operations_router
from .social_memory import router as social_memory_router
from .reddit import router as reddit_router
from .context import router as context_router
from .feedback import router as feedback_router
from .upstream import router as upstream_router

# Single router that merges all sub-modules
# This preserves the existing import: from .graphrag import router as graphrag_router
router = APIRouter(prefix="/api/v1/graphrag", tags=["graphrag"])

router.include_router(health_router)
router.include_router(graph_router)
router.include_router(search_router)
router.include_router(ingest_router)
router.include_router(code_router)
router.include_router(extraction_router)
router.include_router(evaluation_router)
router.include_router(operations_router)
router.include_router(social_memory_router)
router.include_router(reddit_router)
router.include_router(context_router)
router.include_router(feedback_router)
router.include_router(upstream_router)

# ── Backward-compatible re-exports ─────────────────────────────────────
from .helpers import (  # noqa: F401, E402
    check_graphrag_enabled,
    get_http_client,
    graphrag_url,
    _normalize_entity_type, _transform_nodes,
    _fetch_graph_data, _find_entity_neighbors,
)
from .health import (  # noqa: F401, E402
    graphrag_health, graphrag_services_status,
    graphrag_get_stats, graphrag_get_domains,
)
from .graph import (  # noqa: F401, E402
    graphrag_get_graph, graphrag_top_entities,
    graphrag_top_relationships, graphrag_entity_neighborhood,
)
from .search import intelligent_search  # noqa: F401, E402
from .ingest import (  # noqa: F401, E402
    graphrag_extract_entities, graphrag_upload_document,
)
from .context import graphrag_chat_context  # noqa: F401, E402
