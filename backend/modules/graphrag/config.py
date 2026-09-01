"""
GraphRAG Configuration — Single Source of Truth

All GraphRAG-related settings are read from the environment once and cached
in a module-level singleton.  Every layer (routes, modules, wrapper,
executors) should import ``get_graphrag_config()`` instead of calling
``os.getenv()`` directly.

Environment variables
---------------------
GRAPHRAG_ENABLED          – master toggle (default: false)
GRAPHRAG_URL              – external GraphRAG service URL
NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
NER_API_URL / REL_API_URL / CODE_RAG_URL / REDDIT_CRAWLER_URL
QDRANT_URL / GRAPHRAG_COLLECTION
LLM_API_URL / LLM_API_KEY
EMBEDDING_MODEL_NAME      – embedding model (default: nomic-embed-text-v1.5)
BACKEND_INTERNAL_URL      – internal self-call URL (default: http://localhost:8700)
GRAPHRAG_HTTP_TIMEOUT      – default httpx timeout in seconds (default: 30)
GRAPHRAG_HTTP_POOL_SIZE    – httpx connection pool size (default: 20)
GRAPHRAG_PIPELINE_RUNS_FILE – optional JSON file for durable ingest traces
GRAPHRAG_ENABLE_IMAGE_OCR   – enable optional Pillow/pytesseract image OCR
"""

import os
from dataclasses import dataclass
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
    reddit_crawler_url: Optional[str] = None

    # ── External GraphRAG API ──────────────────────────────────────────
    graphrag_url: Optional[str] = None

    # ── URL provenance ─────────────────────────────────────────────────
    graphrag_url_source: str = "unset"
    ner_api_url_source: str = "unset"
    rel_api_url_source: str = "unset"
    code_rag_url_source: str = "unset"
    reddit_crawler_url_source: str = "unset"

    # ── LLM for reasoning ─────────────────────────────────────────────
    llm_api_url: Optional[str] = None
    llm_api_key: Optional[str] = None

    # ── Embeddings ────────────────────────────────────────────────────
    embedding_model: str = "nomic-embed-text-v1.5"

    # ── Local document pipeline extras ────────────────────────────────
    image_ocr_enabled: bool = False

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
    def reddit_crawler_enabled(self) -> bool:
        return self.reddit_crawler_url is not None

    @property
    def has_graph_service(self) -> bool:
        """True when the external GraphRAG service URL is configured."""
        return self.graphrag_url is not None

    @property
    def configured_service_count(self) -> int:
        """Number of GraphRAG-related service URLs configured."""
        return sum(
            bool(value)
            for value in (
                self.graphrag_url,
                self.ner_api_url,
                self.rel_api_url,
                self.code_rag_url,
                self.reddit_crawler_url,
            )
        )

    @property
    def service_url_sources(self) -> dict[str, str]:
        """Where each GraphRAG service URL came from."""
        return {
            "graphrag": self.graphrag_url_source,
            "ner": self.ner_api_url_source,
            "rel": self.rel_api_url_source,
            "code_rag": self.code_rag_url_source,
            "reddit_crawler": self.reddit_crawler_url_source,
        }

    @property
    def auto_discovered_service_count(self) -> int:
        """Number of service URLs inferred from Docker compose defaults."""
        return sum(source == "docker_default" for source in self.service_url_sources.values())


# ── Singleton ──────────────────────────────────────────────────────────

_config: Optional[GraphRAGConfig] = None


def _str_or_none(val: Optional[str]) -> Optional[str]:
    """Return None for empty strings."""
    return val.strip().rstrip("/") if val and val.strip() else None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _docker_default(url: str) -> Optional[str]:
    """Use Docker-network defaults only inside the compose backend container."""
    if _env_flag("RUNNING_IN_DOCKER") or _env_flag("USE_DOCKER"):
        return url
    return None


def _env_url(*names: str) -> tuple[Optional[str], str]:
    """Resolve the first non-empty URL and remember which env var supplied it."""
    for name in names:
        value = _str_or_none(os.getenv(name))
        if value:
            return value, name
    return None, "unset"


def _docker_default_url(url: str) -> tuple[Optional[str], str]:
    value = _docker_default(url)
    return (value, "docker_default") if value else (None, "unset")


def get_graphrag_config() -> GraphRAGConfig:
    """
    Return a cached, immutable configuration built from environment
    variables.  Safe to call from any thread after import-time.
    """
    global _config

    if _config is None:
        enabled = _env_flag("GRAPHRAG_ENABLED")

        ner_url, ner_source = _env_url("NER_API_URL")
        rel_url, rel_source = _env_url("REL_API_URL")
        code_rag_url, code_rag_source = _env_url("CODE_RAG_URL")
        reddit_crawler_url, reddit_crawler_source = _env_url("REDDIT_CRAWLER_URL")
        graphrag_url, graphrag_source = _env_url("GRAPHRAG_URL", "GRAPHRAG_SERVICE_URL")

        # Auto-detect fallback env vars and compose service names. This keeps
        # `GRAPHRAG_ENABLED=true docker compose --profile graphrag up` usable
        # without requiring every helper URL to be exported manually.
        if enabled and not ner_url:
            ner_url, ner_source = _env_url("GRAPHRAG_NER_URL")
            if not ner_url:
                ner_url, ner_source = _docker_default_url("http://graphrag-ner:8001")
        if enabled and not rel_url:
            rel_url, rel_source = _env_url("GRAPHRAG_REL_URL")
            if not rel_url:
                rel_url, rel_source = _docker_default_url("http://graphrag-rel:8002")
        if enabled and not code_rag_url:
            code_rag_url, code_rag_source = _env_url("GRAPHRAG_CODE_RAG_URL")
            if not code_rag_url:
                code_rag_url, code_rag_source = _docker_default_url("http://graphrag-code:8000")
        if enabled and not reddit_crawler_url:
            reddit_crawler_url, reddit_crawler_source = _env_url("GRAPHRAG_REDDIT_CRAWLER_URL")

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
            reddit_crawler_url=reddit_crawler_url,
            graphrag_url=graphrag_url,
            graphrag_url_source=graphrag_source,
            ner_api_url_source=ner_source,
            rel_api_url_source=rel_source,
            code_rag_url_source=code_rag_source,
            reddit_crawler_url_source=reddit_crawler_source,
            llm_api_url=_str_or_none(os.getenv("LLM_API_URL")),
            llm_api_key=_str_or_none(os.getenv("LLM_API_KEY")),
            embedding_model=os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text-v1.5"),
            image_ocr_enabled=_env_flag("GRAPHRAG_ENABLE_IMAGE_OCR"),
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
            logger.info(f"  Reddit:{'✓ ' + reddit_crawler_url if reddit_crawler_url else '✗ not configured'}")
            logger.info(f"  Embed: {_config.embedding_model}")
            if _config.graphrag_url:
                logger.info(f"  GraphRAG URL: {_config.graphrag_url}")
        else:
            logger.info("GraphRAG disabled")

    return _config


def reset_config():
    """Reset the cached configuration (for testing)."""
    global _config
    _config = None
