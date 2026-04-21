"""
GraphRAG Configuration — Single Source of Truth

All GraphRAG-related settings are read from the environment once and cached
in a module-level singleton.  Every layer (routes, modules, wrapper,
executors) should import ``get_graphrag_config()`` instead of calling
``os.getenv()`` directly.

Environment variables
---------------------
GRAPHRAG_ENABLED          – master toggle (default: false)
GRAPHRAG_URL              – legacy external GraphRAG service URL
NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
NER_API_URL / REL_API_URL / CODE_RAG_URL
QDRANT_URL / GRAPHRAG_COLLECTION
LLM_API_URL / LLM_API_KEY
EMBEDDING_MODEL_NAME      – embedding model (default: nomic-embed-text-v1.5)
BACKEND_INTERNAL_URL      – internal self-call URL (default: http://localhost:8700)
GRAPHRAG_HTTP_TIMEOUT      – default httpx timeout in seconds (default: 30)
GRAPHRAG_HTTP_POOL_SIZE    – httpx connection pool size (default: 20)
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphRAGConfig:
    """Immutable configuration snapshot read from the environment."""

    # ── Master toggle ──────────────────────────────────────────────────
    enabled: bool = False

    # ── Neo4j ──────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # ── Vector store (shared Qdrant) ───────────────────────────────────
    qdrant_url: str = "http://qdrant:6333"
    graphrag_collection: str = "graphrag_documents"

    # ── External GPU services (opt-in) ─────────────────────────────────
    ner_api_url: Optional[str] = None
    rel_api_url: Optional[str] = None
    code_rag_url: Optional[str] = None

    # ── Legacy external service ────────────────────────────────────────
    graphrag_url: Optional[str] = None

    # ── LLM for reasoning ─────────────────────────────────────────────
    llm_api_url: Optional[str] = None
    llm_api_key: Optional[str] = None

    # ── Embeddings ────────────────────────────────────────────────────
    embedding_model: str = "nomic-embed-text-v1.5"

    # ── Internal networking ────────────────────────────────────────────
    backend_internal_url: str = "http://localhost:8700"

    # ── HTTP client tuning ─────────────────────────────────────────────
    http_timeout: float = 30.0
    http_pool_size: int = 20

    # ── Relationship extraction defaults ───────────────────────────────
    default_relation_types: tuple = (
        "works_at",
        "developed_by",
        "part_of",
        "located_in",
        "collaborates_with",
        "causes",
        "related_to",
        "uses",
        "depends_on",
        "created",
        "contains",
        "member_of",
        "operates_in",
        "competes_with",
        "precedes",
        "follows",
    )

    # ── Convenience properties ─────────────────────────────────────────
    @property
    def ner_enabled(self) -> bool:
        return self.ner_api_url is not None

    @property
    def rel_enabled(self) -> bool:
        return self.rel_api_url is not None

    @property
    def code_rag_enabled(self) -> bool:
        return self.code_rag_url is not None

    @property
    def has_graph_service(self) -> bool:
        """True when the external GraphRAG service URL is configured."""
        return self.graphrag_url is not None


# ── Singleton ──────────────────────────────────────────────────────────

_config: Optional[GraphRAGConfig] = None


def _str_or_none(val: Optional[str]) -> Optional[str]:
    """Return None for empty strings."""
    return val if val else None


def get_graphrag_config() -> GraphRAGConfig:
    """
    Return a cached, immutable configuration built from environment
    variables.  Safe to call from any thread after import-time.
    """
    global _config

    if _config is None:
        enabled = os.getenv("GRAPHRAG_ENABLED", "false").lower() == "true"

        ner_url = _str_or_none(os.getenv("NER_API_URL"))
        rel_url = _str_or_none(os.getenv("REL_API_URL"))
        code_rag_url = _str_or_none(os.getenv("CODE_RAG_URL"))

        # Auto-detect fallback env vars
        if enabled and not ner_url:
            ner_url = _str_or_none(os.getenv("GRAPHRAG_NER_URL"))
        if enabled and not rel_url:
            rel_url = _str_or_none(os.getenv("GRAPHRAG_REL_URL"))
        if enabled and not code_rag_url:
            code_rag_url = _str_or_none(os.getenv("GRAPHRAG_CODE_RAG_URL"))

        _config = GraphRAGConfig(
            enabled=enabled,
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            qdrant_url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
            graphrag_collection=os.getenv("GRAPHRAG_COLLECTION", "graphrag_documents"),
            ner_api_url=ner_url,
            rel_api_url=rel_url,
            code_rag_url=code_rag_url,
            graphrag_url=_str_or_none(os.getenv("GRAPHRAG_URL")),
            llm_api_url=_str_or_none(os.getenv("LLM_API_URL")),
            llm_api_key=_str_or_none(os.getenv("LLM_API_KEY")),
            embedding_model=os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text-v1.5"),
            backend_internal_url=os.getenv("BACKEND_INTERNAL_URL", "http://localhost:8700"),
            http_timeout=float(os.getenv("GRAPHRAG_HTTP_TIMEOUT", "30")),
            http_pool_size=int(os.getenv("GRAPHRAG_HTTP_POOL_SIZE", "20")),
        )

        if enabled:
            logger.info("GraphRAG enabled")
            logger.info(f"  Neo4j: {_config.neo4j_uri}")
            logger.info(f"  NER:   {'✓ ' + ner_url if ner_url else '✗ not configured'}")
            logger.info(f"  REL:   {'✓ ' + rel_url if rel_url else '✗ not configured'}")
            logger.info(f"  Code:  {'✓ ' + code_rag_url if code_rag_url else '✗ not configured'}")
            logger.info(f"  Embed: {_config.embedding_model}")
            if _config.graphrag_url:
                logger.info(f"  Legacy GraphRAG URL: {_config.graphrag_url}")
        else:
            logger.info("GraphRAG disabled")

    return _config


def reset_config():
    """Reset the cached configuration (for testing)."""
    global _config
    _config = None
