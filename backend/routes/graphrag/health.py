"""
GraphRAG Health & Status Routes

Service health checks, status, stats, and capability queries.
All endpoints use the shared connection-pooled HTTP client.
"""

from fastapi import APIRouter, HTTPException
import httpx
import logging

from .helpers import check_graphrag_enabled, get_http_client
from modules.graphrag.config import get_graphrag_config

logger = logging.getLogger(__name__)

router = APIRouter()


async def _check_service(url: str | None, name: str) -> dict:
    """Check if a service is available."""
    if not url:
        return {"status": "not_configured", "message": f"Set {name.upper()}_API_URL to enable"}
    try:
        client = get_http_client()
        response = await client.get(f"{url}/health", timeout=5.0)
        if response.status_code == 200:
            return {"status": "healthy", "url": url}
        return {"status": "unhealthy", "url": url, "code": response.status_code}
    except Exception as e:
        return {"status": "unavailable", "url": url, "error": str(e)}


@router.get("/health")
async def graphrag_health():
    """Check GraphRAG system health and service availability."""
    cfg = get_graphrag_config()

    result = {
        "graphrag_enabled": cfg.enabled,
        "status": "healthy" if cfg.enabled else "disabled",
        "services": {},
    }

    if cfg.enabled:
        # Check optional GPU services
        result["services"]["ner"] = await _check_service(cfg.ner_api_url, "ner")
        result["services"]["rel"] = await _check_service(cfg.rel_api_url, "rel")
        result["services"]["code_rag"] = await _check_service(cfg.code_rag_url, "code_rag")

        # Check legacy external service
        if cfg.graphrag_url:
            result["services"]["graphrag"] = await _check_service(cfg.graphrag_url, "graphrag")

        # Check Neo4j connectivity
        result["services"]["neo4j"] = {
            "configured": cfg.neo4j_uri is not None,
            "uri": cfg.neo4j_uri,
        }

        result["deployment_hints"] = {
            "ner": "docker compose --profile graphrag-ner up -d",
            "rel": "docker compose --profile graphrag-rel up -d",
            "code_rag": "docker compose --profile graphrag-code up -d",
        }

    return result


@router.get("/services/status")
async def graphrag_services_status():
    """Get detailed status of all GraphRAG services (for UI display)."""
    cfg = get_graphrag_config()
    return {
        "ner": {
            "enabled": cfg.ner_enabled,
            "url": cfg.ner_api_url,
            "deploy_command": "docker compose --profile graphrag-ner up -d",
            "description": "GLiNER-based entity extraction (GPU required)",
        },
        "rel": {
            "enabled": cfg.rel_enabled,
            "url": cfg.rel_api_url,
            "deploy_command": "docker compose --profile graphrag-rel up -d",
            "description": "Relationship extraction (GPU required)",
        },
        "code_rag": {
            "enabled": cfg.code_rag_enabled,
            "url": cfg.code_rag_url,
            "deploy_command": "docker compose --profile graphrag-code up -d",
            "description": "Code detection and search",
        },
    }


@router.get("/stats")
async def graphrag_get_stats():
    """Get knowledge graph statistics from graphrag service."""
    check_graphrag_enabled()
    cfg = get_graphrag_config()

    if not cfg.graphrag_url:
        return {"error": "GRAPHRAG_URL not configured", "nodes": 0, "edges": 0}

    try:
        client = get_http_client()
        response = await client.get(f"{cfg.graphrag_url}/knowledge-graph/stats")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/domains")
async def graphrag_get_domains():
    """Get available domains from graphrag service."""
    check_graphrag_enabled()
    cfg = get_graphrag_config()

    if not cfg.graphrag_url:
        return {"domains": []}

    try:
        client = get_http_client()
        response = await client.get(f"{cfg.graphrag_url}/knowledge-graph/domains")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/ner/status")
async def graphrag_ner_status():
    """Get NER (Named Entity Recognition) service status."""
    check_graphrag_enabled()
    cfg = get_graphrag_config()

    # Check NER directly if available
    if cfg.ner_api_url:
        return await _check_service(cfg.ner_api_url, "ner")

    # Fallback to legacy service
    if cfg.graphrag_url:
        try:
            client = get_http_client()
            response = await client.get(f"{cfg.graphrag_url}/ner/status")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")

    return {"status": "not_configured", "message": "Set NER_API_URL or GRAPHRAG_URL"}


@router.get("/code/health")
async def graphrag_code_health():
    """Check code search service health."""
    check_graphrag_enabled()
    cfg = get_graphrag_config()

    # Check Code-RAG directly if available
    if cfg.code_rag_url:
        return await _check_service(cfg.code_rag_url, "code_rag")

    # Fallback to legacy service
    if cfg.graphrag_url:
        try:
            client = get_http_client()
            response = await client.get(f"{cfg.graphrag_url}/code-detection/health")
            return response.json()
        except Exception as e:
            return {"available": False, "error": str(e)}

    return {"available": False, "message": "Set CODE_RAG_URL or GRAPHRAG_URL"}


@router.get("/hybrid/status")
async def graphrag_hybrid_status():
    """Get status of GraphRAG, Code RAG, and hybrid availability."""
    check_graphrag_enabled()
    cfg = get_graphrag_config()

    if cfg.graphrag_url:
        try:
            client = get_http_client()
            response = await client.get(f"{cfg.graphrag_url}/hybrid/status", timeout=10.0)
            return response.json()
        except Exception as e:
            pass

    return {
        "graphrag_healthy": cfg.graphrag_url is not None,
        "code_rag_healthy": cfg.code_rag_url is not None,
        "hybrid_available": cfg.graphrag_url is not None and cfg.code_rag_url is not None,
        "ner_available": cfg.ner_api_url is not None,
        "rel_available": cfg.rel_api_url is not None,
    }


@router.get("/extraction-stats")
async def graphrag_extraction_stats():
    """Get entity extraction method statistics."""
    check_graphrag_enabled()
    cfg = get_graphrag_config()

    if not cfg.graphrag_url:
        return {"message": "GRAPHRAG_URL not configured", "methods": []}

    try:
        client = get_http_client()
        response = await client.get(f"{cfg.graphrag_url}/extraction-stats", timeout=10.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Extraction stats error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/supported-formats")
async def graphrag_supported_formats():
    """Get supported file formats and features."""
    check_graphrag_enabled()
    cfg = get_graphrag_config()

    if not cfg.graphrag_url:
        return {
            "formats": ["txt", "pdf", "docx", "md", "html", "epub"],
            "features": {
                "semantic_chunking": True,
                "entity_extraction": cfg.ner_enabled,
                "relationship_extraction": cfg.rel_enabled,
                "code_detection": cfg.code_rag_enabled,
            },
        }

    try:
        client = get_http_client()
        response = await client.get(f"{cfg.graphrag_url}/supported-formats", timeout=10.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Supported formats error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")
