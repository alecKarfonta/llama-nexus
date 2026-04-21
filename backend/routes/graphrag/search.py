"""
GraphRAG Search & Reasoning Routes

Search, advanced search, intelligent search, multi-hop reasoning,
relationship explanation, causal/comparative reasoning, code search.

All endpoints use the shared connection-pooled HTTP client.
"""

from fastapi import APIRouter, HTTPException, Request
import httpx
import logging

from .helpers import check_graphrag_enabled, get_http_client, graphrag_url as _graphrag_url
from modules.graphrag.config import get_graphrag_config

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/search")
async def graphrag_search(request: Request):
    """Hybrid search using graphrag service (vector + graph + keyword)."""
    check_graphrag_enabled()

    data = await request.json()
    if not data.get("query"):
        raise HTTPException(status_code=400, detail="query is required")

    try:
        client = get_http_client()
        response = await client.post(f"{_graphrag_url()}/search", json=data)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/search/advanced")
async def graphrag_advanced_search(request: Request):
    """Advanced search with type selection using graphrag service."""
    check_graphrag_enabled()

    data = await request.json()
    if not data.get("query"):
        raise HTTPException(status_code=400, detail="query is required")

    try:
        client = get_http_client()
        response = await client.post(f"{_graphrag_url()}/search-advanced", json=data)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/query/intent")
async def analyze_query_intent(query: str):
    """Analyze query to determine intent, entities, and complexity."""
    check_graphrag_enabled()

    if not query:
        raise HTTPException(status_code=400, detail="query parameter is required")

    try:
        client = get_http_client()
        response = await client.get(
            f"{_graphrag_url()}/api/analyze-query-intent",
            params={"query": query},
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPError as e:
        logger.error(f"Query intent analysis error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/search/intelligent")
async def intelligent_search(request: Request):
    """Intelligent search with automatic method selection and LLM answer generation."""
    check_graphrag_enabled()

    data = await request.json()
    query = data.get("query")

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    try:
        client = get_http_client()
        search_type = data.get("search_type", "auto")

        if search_type == "auto":
            search_type = "hybrid"

        search_payload = {
            "query": query,
            "search_type": search_type,
            "top_k": data.get("top_k", 10),
            "domain": data.get("domain"),
            "filters": data.get("filters"),
        }

        response = await client.post(
            f"{_graphrag_url()}/search-advanced",
            json=search_payload,
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()

        result["search_method_used"] = search_type

        return result

    except httpx.HTTPError as e:
        logger.error(f"Intelligent search error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")


@router.post("/reasoning/multi-hop")
async def graphrag_multi_hop_reasoning(request: Request):
    """Perform multi-hop reasoning query using graphrag service."""
    check_graphrag_enabled()

    data = await request.json()
    if not data.get("query"):
        raise HTTPException(status_code=400, detail="query is required")

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/reasoning/multi-hop",
            json=data,
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/reasoning/explain")
async def explain_relationship(request: Request):
    """Explain the relationship between two entities."""
    check_graphrag_enabled()

    data = await request.json()
    source = data.get("source")
    target = data.get("target")

    if not source or not target:
        raise HTTPException(status_code=400, detail="source and target are required")

    try:
        client = get_http_client()
        payload = {
            "source": source,
            "target": target,
            "entities": data.get("entities", []),
            "relationships": data.get("relationships", []),
        }

        response = await client.post(
            f"{_graphrag_url()}/reasoning/explain-relationship",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPError as e:
        logger.error(f"Relationship explanation error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/reasoning/causal")
async def graphrag_causal_reasoning(request: Request):
    """Causal reasoning to find cause-effect relationships."""
    check_graphrag_enabled()

    data = await request.json()
    query = data.get("query")

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/api/causal-reasoning",
            json={"query": query},
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPError as e:
        logger.error(f"Causal reasoning error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/reasoning/comparative")
async def graphrag_comparative_reasoning(request: Request):
    """Compare entities or concepts."""
    check_graphrag_enabled()

    data = await request.json()
    query = data.get("query")

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/api/comparative-reasoning",
            json={"query": query},
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPError as e:
        logger.error(f"Comparative reasoning error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/reasoning/advanced")
async def graphrag_advanced_reasoning(request: Request):
    """Advanced reasoning with query analysis."""
    check_graphrag_enabled()

    data = await request.json()
    query = data.get("query")

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/api/advanced-reasoning",
            json={"query": query},
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPError as e:
        logger.error(f"Advanced reasoning error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/code/detect")
async def graphrag_detect_code(request: Request):
    """Detect code in text."""
    check_graphrag_enabled()

    data = await request.json()
    text = data.get("text")

    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/code-detection/detect",
            json={"text": text},
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPError as e:
        logger.error(f"Code detection error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/code/search")
async def graphrag_search_code(request: Request):
    """Search for code examples."""
    check_graphrag_enabled()

    data = await request.json()
    query = data.get("query")

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/search/code",
            json=data,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPError as e:
        logger.error(f"Code search error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")
