"""
GraphRAG Upstream Bridge

Allowlisted passthrough for advanced GraphRAG service endpoints that do not
yet have first-class llama-nexus wrappers.
"""

from fastapi import APIRouter, HTTPException, Request, Response
import httpx
import logging
import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .helpers import check_graphrag_enabled, get_http_client, graphrag_url as _graphrag_url
from modules.graphrag.config import get_graphrag_config

logger = logging.getLogger(__name__)

router = APIRouter()


CAPABILITY_CATALOG = [
    {
        "id": "health",
        "label": "System health and service status",
        "llama_nexus_path": "/api/v1/graphrag/health",
        "upstream_paths": ["/health", "/hybrid/status", "/ner/status", "/rel/status"],
    },
    {
        "id": "search",
        "label": "Hybrid, advanced, and intelligent search",
        "llama_nexus_path": "/api/v1/graphrag/search/intelligent",
        "upstream_paths": ["/search", "/search-advanced", "/api/enhanced-query"],
    },
    {
        "id": "knowledge_graph",
        "label": "Knowledge graph filtering, export, domains, and top relationships",
        "llama_nexus_path": "/api/v1/graphrag/graph",
        "upstream_paths": ["/knowledge-graph/*"],
    },
    {
        "id": "reasoning",
        "label": "Multi-hop, causal, comparative, advanced, and relationship reasoning",
        "llama_nexus_path": "/api/v1/graphrag/reasoning/*, /api/v1/graphrag/query/complexity",
        "upstream_paths": ["/reasoning/*", "/api/*reasoning", "/api/query-complexity-analysis", "/query/*"],
    },
    {
        "id": "extraction",
        "label": "Entity, relationship, enhanced extraction, direct NER, and direct REL",
        "llama_nexus_path": "/api/v1/graphrag/ner/*, /api/v1/graphrag/rel/*, /api/v1/graphrag/extract, /api/v1/graphrag/extract/enhanced",
        "upstream_paths": [
            "/extract-entities-relations",
            "/extract-entities-relations-enhanced",
            "/ner/*",
            "/rel/*",
            "NER_API_URL /ner",
            "NER_API_URL /ner/batch",
            "NER_API_URL /ner/entities",
            "REL_API_URL /extract-relations",
            "REL_API_URL /extract-relations/batch",
            "REL_API_URL /extract-entities",
            "REL_API_URL /capabilities",
        ],
    },
    {
        "id": "ingestion",
        "label": "Document processing, ingestion, rebuild, deletion, and hybrid routing",
        "llama_nexus_path": "/api/v1/graphrag/ingest/upload, /api/v1/graphrag/ingest/pipeline/preview, /api/v1/graphrag/ingest/pipeline/upload",
        "upstream_paths": [
            "/process-document",
            "/process-documents-batch",
            "/process-document-with-ner",
            "/ingest-documents",
            "/add-to-vector-store",
            "/rebuild-knowledge-graph",
            "/documents/*",
        ],
    },
    {
        "id": "code_rag",
        "label": "Code detection, routing, search, ingestion, analysis, statistics, and index controls",
        "llama_nexus_path": "/api/v1/graphrag/code/*",
        "upstream_paths": [
            "/code-detection/*",
            "/hybrid/*",
            "/search/code",
            "CODE_RAG_URL /search",
            "CODE_RAG_URL /ingest",
            "CODE_RAG_URL /upload",
            "CODE_RAG_URL /analyze",
            "CODE_RAG_URL /statistics",
            "CODE_RAG_URL /index",
        ],
    },
    {
        "id": "reddit_crawler",
        "label": "Reddit crawling with optional GraphRAG auto-ingestion",
        "llama_nexus_path": "/api/v1/graphrag/reddit/*",
        "upstream_paths": [
            "REDDIT_CRAWLER_URL /health",
            "REDDIT_CRAWLER_URL /crawl",
            "REDDIT_CRAWLER_URL /crawl/{crawl_id}/status",
            "REDDIT_CRAWLER_URL /crawl/{crawl_id}/results",
            "REDDIT_CRAWLER_URL /crawls",
        ],
    },
    {
        "id": "operations",
        "label": "Domain statistics, document previews, vector store updates, diagnostics, and guarded data clearing",
        "llama_nexus_path": "/api/v1/graphrag/process-document, /api/v1/graphrag/vector-store/add, /api/v1/graphrag/admin/clear-all",
        "upstream_paths": [
            "/knowledge-graph/domain-stats",
            "/process-document",
            "/process-documents-batch",
            "/process-document-with-ner",
            "/add-to-vector-store",
            "/code-detection/route",
            "/test-enhanced-extraction",
        ],
    },
    {
        "id": "evaluation",
        "label": "GraphRAG quality, chunking, retrieval, and test suite evaluation",
        "llama_nexus_path": "/api/v1/graphrag/evaluation/*",
        "upstream_paths": ["/evaluate/*", "/test/*"],
    },
    {
        "id": "social_memory",
        "label": "Social memory ingestion, context, explanation, identity, and admin tools",
        "llama_nexus_path": "/api/v1/graphrag/social-memory/*",
        "upstream_paths": ["/social-memory/*"],
    },
]

ALLOWED_UPSTREAM_PREFIXES = (
    "health",
    "supported-formats",
    "process-document",
    "process-documents-batch",
    "process-document-with-ner",
    "search",
    "search-advanced",
    "search/code",
    "ingest-documents",
    "add-to-vector-store",
    "rebuild-knowledge-graph",
    "documents/",
    "knowledge-graph/",
    "evaluate/",
    "test/",
    "ner/",
    "rel/",
    "query/",
    "reasoning/",
    "api/",
    "extract-entities-relations",
    "extract-entities-relations-enhanced",
    "extraction-stats",
    "test-enhanced-extraction",
    "code-detection/",
    "hybrid/",
    "social-memory/",
)


def _env_flag(name: str) -> bool:
    value = os.getenv(name)
    return bool(value and value.strip().lower() in {"1", "true", "yes", "on"})


def _public_url(url: str | None) -> str | None:
    if not url:
        return None

    try:
        parsed = urlsplit(url)
    except ValueError:
        return url

    if not parsed.hostname:
        return url

    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))


def _is_container_loopback_url(url: str | None) -> bool:
    if not url:
        return False

    try:
        host = urlsplit(url).hostname
    except ValueError:
        return False

    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _url_source_label(source: str | None) -> str:
    if source == "docker_default":
        return "Docker service default"
    if not source or source == "unset":
        return "Not configured"
    return source


def _url_source_payload(source: str | None) -> dict[str, Any]:
    source = source or "unset"
    label = _url_source_label(source)
    return {
        "source": source,
        "url_source": source,
        "label": label,
        "url_source_label": label,
        "auto_discovered": source == "docker_default",
        "explicit": source not in {"unset", "docker_default"},
    }


def _feature(
    *,
    feature_id: str,
    label: str,
    configured: bool,
    enabled: bool,
    required_env: list[str],
    message: str,
    next_step: str,
    url: str | None = None,
    urls: dict[str, str | None] | None = None,
    url_source: str | None = None,
    url_sources: dict[str, str | None] | None = None,
    command: str | None = None,
    optional: bool = False,
) -> dict[str, Any]:
    if not enabled:
        status = "disabled"
    elif configured:
        status = "ready"
    elif optional:
        status = "optional"
    else:
        status = "missing"

    result: dict[str, Any] = {
        "id": feature_id,
        "label": label,
        "configured": configured,
        "ready": enabled and configured,
        "status": status,
        "required_env": required_env,
        "message": message,
        "next_step": next_step,
    }
    if url:
        result["url"] = _public_url(url)
    if urls:
        result["urls"] = {key: _public_url(value) for key, value in urls.items()}
    if url_source:
        source_payload = _url_source_payload(url_source)
        result["url_source"] = source_payload["source"]
        result["url_source_label"] = source_payload["label"]
        result["auto_discovered"] = source_payload["auto_discovered"]
        result["explicit"] = source_payload["explicit"]
    if url_sources:
        result["url_sources"] = {
            key: _url_source_payload(value)
            for key, value in url_sources.items()
        }
    if command:
        result["command"] = command

    return result


def _build_deployment_diagnostics() -> dict[str, Any]:
    cfg = get_graphrag_config()
    docker_runtime = _env_flag("RUNNING_IN_DOCKER") or _env_flag("USE_DOCKER")
    graph_configured = bool(cfg.graphrag_url)
    extraction_configured = bool(cfg.ner_api_url or cfg.rel_api_url)
    helper_configured = bool(cfg.ner_api_url or cfg.rel_api_url or cfg.code_rag_url or cfg.reddit_crawler_url)
    url_sources = cfg.service_url_sources
    auto_discovered_services = [
        name for name, source in url_sources.items()
        if source == "docker_default"
    ]
    explicit_url_services = [
        name for name, source in url_sources.items()
        if source not in {"unset", "docker_default"}
    ]

    commands = {
        "full_helper_stack": "GRAPHRAG_ENABLED=true docker compose --profile graphrag up -d",
        "ner": "GRAPHRAG_ENABLED=true docker compose --profile graphrag-ner up -d",
        "rel": "GRAPHRAG_ENABLED=true docker compose --profile graphrag-rel up -d",
        "code_rag": "GRAPHRAG_ENABLED=true docker compose --profile graphrag-code up -d",
        "reddit_crawler": "GRAPHRAG_ENABLED=true docker compose --profile reddit-crawler up -d",
        "health_check": "curl http://localhost:8700/api/v1/graphrag/health",
    }

    features = {
        "local_pipeline": _feature(
            feature_id="local_pipeline",
            label="Local pipeline and run-scoped chat",
            configured=True,
            enabled=cfg.enabled,
            required_env=["GRAPHRAG_ENABLED"],
            message=(
                "Local extraction traces, sample outputs, graph candidates, cached chunks, and run-scoped chat handoff are enabled."
                if cfg.enabled
                else "Enable GraphRAG routes to use local pipeline traces and run-scoped chat handoff."
            ),
            next_step="Set GRAPHRAG_ENABLED=true and restart the backend.",
        ),
        "graph_api": _feature(
            feature_id="graph_api",
            label="Graph API",
            configured=graph_configured,
            enabled=cfg.enabled,
            required_env=["GRAPHRAG_URL", "GRAPHRAG_SERVICE_URL"],
            url=cfg.graphrag_url,
            url_source=url_sources.get("graphrag"),
            message=(
                "Graph/search/ingest/reasoning routes can use the configured GraphRAG API."
                if graph_configured
                else "Graph/search/ingest/reasoning routes need an external GraphRAG API URL."
            ),
            next_step="Set GRAPHRAG_URL to the GraphRAG API base URL.",
        ),
        "extraction": _feature(
            feature_id="extraction",
            label="Extraction",
            configured=extraction_configured,
            enabled=cfg.enabled,
            required_env=["NER_API_URL", "REL_API_URL"],
            urls={"ner": cfg.ner_api_url, "rel": cfg.rel_api_url},
            url_sources={
                "ner": url_sources.get("ner"),
                "rel": url_sources.get("rel"),
            },
            message=(
                "Dedicated entity and/or relationship extraction services are configured."
                if extraction_configured
                else "Extraction can fall back to the GraphRAG API when GRAPHRAG_URL is configured."
            ),
            next_step="Start the NER and REL profiles, or set their service URLs.",
            command=commands["full_helper_stack"],
            optional=True,
        ),
        "code_rag": _feature(
            feature_id="code_rag",
            label="Code-RAG",
            configured=bool(cfg.code_rag_url),
            enabled=cfg.enabled,
            required_env=["CODE_RAG_URL", "GRAPHRAG_CODE_RAG_URL"],
            url=cfg.code_rag_url,
            url_source=url_sources.get("code_rag"),
            message=(
                "Code search, code ingestion, and code analysis routes can use Code-RAG."
                if cfg.code_rag_url
                else "Code-RAG is optional and powers code-specific search/analysis routes."
            ),
            next_step="Start the Code-RAG profile or set CODE_RAG_URL.",
            command=commands["code_rag"],
            optional=True,
        ),
        "reddit_crawler": _feature(
            feature_id="reddit_crawler",
            label="Reddit Crawler",
            configured=bool(cfg.reddit_crawler_url),
            enabled=cfg.enabled,
            required_env=["REDDIT_CRAWLER_URL", "GRAPHRAG_REDDIT_CRAWLER_URL"],
            url=cfg.reddit_crawler_url,
            url_source=url_sources.get("reddit_crawler"),
            message=(
                "Reddit crawl lifecycle routes can use the crawler service."
                if cfg.reddit_crawler_url
                else "The Reddit crawler is optional and can be added for social corpus ingestion."
            ),
            next_step="Start the Reddit crawler profile or set REDDIT_CRAWLER_URL.",
            command=commands["reddit_crawler"],
            optional=True,
        ),
        "storage": _feature(
            feature_id="storage",
            label="Storage",
            configured=bool(cfg.neo4j_uri and cfg.qdrant_url),
            enabled=cfg.enabled,
            required_env=["NEO4J_URI", "QDRANT_URL", "GRAPHRAG_COLLECTION"],
            urls={"neo4j": cfg.neo4j_uri, "qdrant": cfg.qdrant_url},
            message="Neo4j and Qdrant settings are present for GraphRAG storage.",
            next_step="Verify Neo4j and Qdrant are reachable from the GraphRAG services.",
        ),
    }

    warnings: list[str] = []
    if not cfg.enabled and cfg.configured_service_count:
        warnings.append("GraphRAG service URLs are configured, but GRAPHRAG_ENABLED is false.")
    if cfg.enabled and not graph_configured:
        warnings.append("GRAPHRAG_URL is not configured; remote graph/search/ingest/reasoning/evaluation routes need a graph API, but local pipeline traces and run-scoped chat fallback can still operate.")
    if cfg.enabled and not helper_configured and not graph_configured:
        warnings.append("GraphRAG is enabled but no GraphRAG service URLs are configured.")
    if docker_runtime:
        for name, url in {
            "GRAPHRAG_URL": cfg.graphrag_url,
            "NER_API_URL": cfg.ner_api_url,
            "REL_API_URL": cfg.rel_api_url,
            "CODE_RAG_URL": cfg.code_rag_url,
            "REDDIT_CRAWLER_URL": cfg.reddit_crawler_url,
        }.items():
            if _is_container_loopback_url(url):
                warnings.append(
                    f"{name} points at localhost inside Docker; use a compose service name or host.docker.internal instead."
                )

    if not cfg.enabled:
        status = "disabled"
        summary = "GraphRAG is disabled. Set GRAPHRAG_ENABLED=true to activate the bridge."
    elif graph_configured:
        status = "ready"
        summary = "GraphRAG is enabled and the graph API is configured."
    elif helper_configured:
        status = "partial"
        summary = "GraphRAG helper services are configured, but the graph API URL is missing."
    else:
        status = "local"
        summary = "GraphRAG is enabled in local mode: pipeline traces, sample outputs, local graph/context artifacts, and run-scoped chat fallback are available."

    if auto_discovered_services and explicit_url_services:
        source_summary = (
            f"{len(explicit_url_services)} explicit service URL(s), "
            f"{len(auto_discovered_services)} Docker default helper URL(s)."
        )
    elif auto_discovered_services:
        source_summary = f"{len(auto_discovered_services)} Docker default helper URL(s)."
    elif explicit_url_services:
        source_summary = f"{len(explicit_url_services)} explicit service URL(s)."
    else:
        source_summary = "No GraphRAG service URLs are configured."

    return {
        "enabled": cfg.enabled,
        "status": status,
        "summary": summary,
        "source_summary": source_summary,
        "configured_service_count": cfg.configured_service_count,
        "explicit_url_count": len(explicit_url_services),
        "auto_discovered_count": len(auto_discovered_services),
        "auto_discovered_services": auto_discovered_services,
        "url_sources": {
            name: _url_source_payload(source)
            for name, source in url_sources.items()
        },
        "docker_runtime": docker_runtime,
        "features": features,
        "warnings": warnings,
        "commands": commands,
        "environment": {
            "GRAPHRAG_ENABLED": cfg.enabled,
            "GRAPHRAG_URL": _public_url(cfg.graphrag_url),
            "NER_API_URL": _public_url(cfg.ner_api_url),
            "REL_API_URL": _public_url(cfg.rel_api_url),
            "CODE_RAG_URL": _public_url(cfg.code_rag_url),
            "REDDIT_CRAWLER_URL": _public_url(cfg.reddit_crawler_url),
            "NEO4J_URI": _public_url(cfg.neo4j_uri),
            "QDRANT_URL": _public_url(cfg.qdrant_url),
            "GRAPHRAG_COLLECTION": cfg.graphrag_collection,
            "RUNNING_IN_DOCKER": docker_runtime,
        },
        "environment_sources": {
            "GRAPHRAG_URL": _url_source_payload(url_sources.get("graphrag")),
            "NER_API_URL": _url_source_payload(url_sources.get("ner")),
            "REL_API_URL": _url_source_payload(url_sources.get("rel")),
            "CODE_RAG_URL": _url_source_payload(url_sources.get("code_rag")),
            "REDDIT_CRAWLER_URL": _url_source_payload(url_sources.get("reddit_crawler")),
        },
    }


def _normalize_upstream_path(path: str) -> str:
    normalized = path.strip().lstrip("/")
    if (
        not normalized
        or ".." in normalized
        or "\\" in normalized
        or normalized.startswith(("http://", "https://"))
        or "//" in normalized
    ):
        raise HTTPException(status_code=400, detail="invalid upstream path")

    if not any(normalized == prefix or normalized.startswith(prefix) for prefix in ALLOWED_UPSTREAM_PREFIXES):
        raise HTTPException(status_code=403, detail=f"GraphRAG upstream path is not allowlisted: {normalized}")

    return normalized


@router.get("/capabilities")
async def graphrag_capabilities():
    """Describe exposed GraphRAG capabilities and bridge availability."""
    cfg = get_graphrag_config()
    diagnostics = _build_deployment_diagnostics()
    return {
        "enabled": cfg.enabled,
        "configured_service_count": cfg.configured_service_count,
        "graphrag_url_configured": bool(cfg.graphrag_url),
        "services": {
            "graphrag": {
                "enabled": cfg.enabled and bool(cfg.graphrag_url),
                "configured": bool(cfg.graphrag_url),
                "url": cfg.graphrag_url,
                **_url_source_payload(cfg.service_url_sources.get("graphrag")),
                "role": "Graph search, document ingestion, graph exploration, reasoning, evaluation, social memory",
            },
            "ner": {
                "enabled": cfg.enabled and cfg.ner_enabled,
                "configured": cfg.ner_enabled,
                "url": cfg.ner_api_url,
                **_url_source_payload(cfg.service_url_sources.get("ner")),
                "role": "Dedicated named entity extraction",
            },
            "rel": {
                "enabled": cfg.enabled and cfg.rel_enabled,
                "configured": cfg.rel_enabled,
                "url": cfg.rel_api_url,
                **_url_source_payload(cfg.service_url_sources.get("rel")),
                "role": "Dedicated relationship extraction",
            },
            "code_rag": {
                "enabled": cfg.enabled and cfg.code_rag_enabled,
                "configured": cfg.code_rag_enabled,
                "url": cfg.code_rag_url,
                **_url_source_payload(cfg.service_url_sources.get("code_rag")),
                "role": "Code search, code ingestion, code analysis",
            },
            "reddit_crawler": {
                "enabled": cfg.enabled and cfg.reddit_crawler_enabled,
                "configured": cfg.reddit_crawler_enabled,
                "url": cfg.reddit_crawler_url,
                **_url_source_payload(cfg.service_url_sources.get("reddit_crawler")),
                "role": "Reddit crawling and optional GraphRAG auto-ingestion",
            },
        },
        "deployment": {
            "backend_toggle": "GRAPHRAG_ENABLED=true",
            "core_profile": "docker compose --profile graphrag up -d",
            "relationship_profile": "docker compose --profile graphrag-rel up -d",
            "external_graph_api": "Set GRAPHRAG_URL to the external GraphRAG API for graph/search/ingest/reasoning operations",
            "diagnostics_path": "/api/v1/graphrag/deployment/diagnostics",
            "status": diagnostics["status"],
            "warnings": diagnostics["warnings"],
            "source_summary": diagnostics["source_summary"],
            "url_sources": diagnostics["url_sources"],
            "auto_discovered_count": diagnostics["auto_discovered_count"],
        },
        "capabilities": CAPABILITY_CATALOG,
        "upstream_bridge": {
            "available": cfg.enabled and bool(cfg.graphrag_url),
            "base_path": "/api/v1/graphrag/upstream/{path}",
            "allowed_prefixes": ALLOWED_UPSTREAM_PREFIXES,
        },
    }


@router.get("/deployment/diagnostics")
async def graphrag_deployment_diagnostics():
    """Return read-only deployment diagnostics and next-step hints."""
    return _build_deployment_diagnostics()


@router.api_route("/upstream/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def graphrag_upstream_proxy(path: str, request: Request):
    """Forward an allowlisted request to the configured GraphRAG service."""
    check_graphrag_enabled()
    upstream_path = _normalize_upstream_path(path)

    headers = {}
    for header_name in ("accept", "content-type"):
        if header_name in request.headers:
            headers[header_name] = request.headers[header_name]

    try:
        client = get_http_client()
        response = await client.request(
            request.method,
            f"{_graphrag_url()}/{upstream_path}",
            params=list(request.query_params.multi_items()),
            content=await request.body(),
            headers=headers,
            timeout=300.0,
        )
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG upstream bridge error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")

    response_headers = {}
    if response.headers.get("content-type"):
        response_headers["content-type"] = response.headers["content-type"]

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=response_headers,
    )
