"""
GraphRAG External Service Proxy Routes
These endpoints proxy requests to the external graphrag microservice
which provides advanced knowledge graph features using Neo4j and GLiNER.
"""

from fastapi import APIRouter, HTTPException, Request
from typing import Optional, Any
import httpx
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/graphrag", tags=["graphrag"])

# GraphRAG Service Configuration (external graphrag microservice)
GRAPHRAG_URL = os.getenv("GRAPHRAG_URL", "http://graphrag-api-1:8000")
GRAPHRAG_ENABLED = os.getenv("GRAPHRAG_ENABLED", "true").lower() == "true"


def check_graphrag_enabled():
    """Check if GraphRAG is enabled and raise exception if not."""
    if not GRAPHRAG_ENABLED:
        raise HTTPException(status_code=503, detail="GraphRAG service is disabled")


@router.get("/health")
async def graphrag_health():
    """Check if external graphrag service is available."""
    if not GRAPHRAG_ENABLED:
        return {"status": "disabled", "message": "GraphRAG service is disabled"}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{GRAPHRAG_URL}/health")
            return response.json()
    except Exception as e:
        return {"status": "unavailable", "error": str(e), "url": GRAPHRAG_URL}


@router.get("/stats")
async def graphrag_get_stats():
    """Get knowledge graph statistics from graphrag service."""
    check_graphrag_enabled()
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{GRAPHRAG_URL}/knowledge-graph/stats")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/domains")
async def graphrag_get_domains():
    """Get available domains from graphrag service."""
    check_graphrag_enabled()
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{GRAPHRAG_URL}/knowledge-graph/domains")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


def _normalize_entity_type(raw_type: str) -> str:
    """Normalize entity type to match frontend expected values.
    
    GraphRAG returns uppercase types like COMPONENT, MAINTENANCE, ORGANIZATION.
    Frontend expects lowercase: person, organization, location, technology, etc.
    """
    if not raw_type:
        return "concept"
    
    type_lower = raw_type.lower()
    
    # Map common GraphRAG types to frontend types
    type_mapping = {
        "component": "technology",
        "maintenance": "process",
        "organization": "organization",
        "person": "person",
        "location": "location",
        "date": "date",
        "event": "event",
        "product": "product",
        "technology": "technology",
        "concept": "concept",
        "process": "process",
        "entity": "concept",  # Generic fallback
    }
    
    return type_mapping.get(type_lower, "custom")


def _transform_nodes(raw_nodes: list) -> list:
    """Transform GraphRAG nodes to frontend format.
    
    GraphRAG returns: {id, label, type: "Entity", properties: {type: "COMPONENT", ...}}
    Frontend expects: {id, label, type: "technology", properties: {...}}
    """
    transformed = []
    for node in raw_nodes:
        props = node.get("properties", {})
        
        # Get the actual entity type from properties
        entity_type = props.get("type") or node.get("type", "concept")
        normalized_type = _normalize_entity_type(entity_type)
        
        transformed.append({
            "id": node.get("id") or props.get("name", ""),
            "label": node.get("label") or props.get("name", ""),
            "type": normalized_type,
            "occurrence": node.get("occurrence") or props.get("occurrence", 1),
            "properties": {
                "description": props.get("description", ""),
                "confidence": props.get("confidence", 1.0),
                "domain": props.get("domain", "general"),
            }
        })
    
    return transformed


@router.post("/graph")
async def graphrag_get_graph(request: Request):
    """Get filtered knowledge graph data from graphrag service.
    
    Transforms the GraphRAG response to match the expected frontend format.
    GraphRAG returns: {"filtered_data": {"nodes": [...], "edges": [...], ...}}
    Frontend expects: {"nodes": [...], "edges": [...]}
    """
    check_graphrag_enabled()
    
    try:
        data = await request.json()
    except:
        data = {}
    
    max_entities = data.get("max_entities", 500)
    max_relationships = data.get("max_relationships", 500)
    domain = data.get("domain")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Get filtered graph data
            response = await client.post(
                f"{GRAPHRAG_URL}/knowledge-graph/filtered",
                json=data
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract data from nested structure
            filtered_data = result.get("filtered_data", result)
            raw_nodes = filtered_data.get("nodes", [])
            edges = filtered_data.get("edges", [])
            
            # Transform nodes to frontend format
            nodes = _transform_nodes(raw_nodes)
            
            # If we got nodes but no edges, fetch top relationships and build edges
            # This handles a bug in the filtered endpoint that returns empty edges
            if nodes and not edges:
                logger.info(f"Filtered endpoint returned {len(nodes)} nodes but 0 edges, fetching relationships separately")
                
                # Get entity names/ids for matching
                node_ids = {n.get("id") for n in nodes}
                node_labels = {n.get("label") for n in nodes}
                all_identifiers = node_ids | node_labels
                
                # Fetch top relationships
                params = {"limit": max_relationships}
                if domain:
                    params["domain"] = domain
                    
                rel_response = await client.get(
                    f"{GRAPHRAG_URL}/knowledge-graph/top-relationships",
                    params=params
                )
                
                if rel_response.status_code == 200:
                    rel_data = rel_response.json()
                    top_rels = rel_data.get("top_relationships", [])
                    
                    # Filter to only relationships where both source and target are in our nodes
                    edge_id = 0
                    for rel in top_rels:
                        source = rel.get("source", "")
                        target = rel.get("target", "")
                        
                        # Check if both endpoints exist in our node set
                        if source in all_identifiers and target in all_identifiers:
                            edges.append({
                                "id": f"edge_{edge_id}",
                                "source": source,
                                "target": target,
                                "label": rel.get("type", "related_to"),
                                "type": rel.get("type", "related_to"),
                                "weight": rel.get("weight", 1)
                            })
                            edge_id += 1
                    
                    logger.info(f"Added {len(edges)} edges from top-relationships matching {len(nodes)} nodes")
            
            # Return transformed response
            return {
                "nodes": nodes,
                "edges": edges,
                "stats": filtered_data.get("stats", {
                    "nodes": len(nodes),
                    "edges": len(edges)
                })
            }
            
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/top-entities")
async def graphrag_top_entities(
    domain: Optional[str] = None,
    limit: int = 50,
    min_occurrence: int = 1
):
    """Get top entities by occurrence from graphrag service."""
    check_graphrag_enabled()
    
    params = {"limit": limit, "min_occurrence": min_occurrence}
    if domain:
        params["domain"] = domain
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{GRAPHRAG_URL}/knowledge-graph/top-entities",
                params=params
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/top-relationships")
async def graphrag_top_relationships(
    domain: Optional[str] = None,
    limit: int = 50,
    min_weight: int = 1
):
    """Get top relationships by weight from graphrag service."""
    check_graphrag_enabled()
    
    params = {"limit": limit, "min_weight": min_weight}
    if domain:
        params["domain"] = domain
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{GRAPHRAG_URL}/knowledge-graph/top-relationships",
                params=params
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/extract")
async def graphrag_extract_entities(request: Request):
    """Extract entities and relationships from text using graphrag GLiNER."""
    check_graphrag_enabled()
    
    data = await request.json()
    if not data.get("text"):
        raise HTTPException(status_code=400, detail="text is required")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{GRAPHRAG_URL}/extract-entities-relations",
                json=data
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/search")
async def graphrag_search(request: Request):
    """Hybrid search using graphrag service (vector + graph + keyword)."""
    check_graphrag_enabled()
    
    data = await request.json()
    if not data.get("query"):
        raise HTTPException(status_code=400, detail="query is required")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{GRAPHRAG_URL}/search",
                json=data
            )
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
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{GRAPHRAG_URL}/search-advanced",
                json=data
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/ingest")
async def graphrag_ingest_document(
    file: Any = None,
    domain: str = "general",
    use_semantic_chunking: bool = True,
    build_knowledge_graph: bool = True
):
    """
    Ingest a document through graphrag service.
    Note: For file uploads, use /api/v1/graphrag/ingest/upload endpoint.
    This endpoint is for JSON-based document submission.
    """
    check_graphrag_enabled()
    
    # This is a placeholder - file upload needs special handling
    raise HTTPException(
        status_code=400,
        detail="Use /api/v1/graphrag/ingest/upload for file uploads"
    )


@router.get("/documents")
async def graphrag_list_documents():
    """List documents in graphrag service."""
    check_graphrag_enabled()
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{GRAPHRAG_URL}/documents/list")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.delete("/documents/{document_name}")
async def graphrag_delete_document(document_name: str):
    """Delete a document from graphrag service."""
    check_graphrag_enabled()
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(f"{GRAPHRAG_URL}/documents/{document_name}")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/ner/status")
async def graphrag_ner_status():
    """Get NER (Named Entity Recognition) service status."""
    check_graphrag_enabled()
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{GRAPHRAG_URL}/ner/status")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/reasoning/multi-hop")
async def graphrag_multi_hop_reasoning(request: Request):
    """Perform multi-hop reasoning query using graphrag service."""
    check_graphrag_enabled()
    
    data = await request.json()
    if not data.get("query"):
        raise HTTPException(status_code=400, detail="query is required")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{GRAPHRAG_URL}/reasoning/multi-hop",
                json=data
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/communities")
async def graphrag_get_communities(domain: Optional[str] = None):
    """Get detected communities from graphrag service."""
    check_graphrag_enabled()
    
    params = {}
    if domain:
        params["domain"] = domain
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{GRAPHRAG_URL}/knowledge-graph/domain-stats",
                params=params
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")
