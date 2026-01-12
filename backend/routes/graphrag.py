"""
GraphRAG External Service Proxy Routes
These endpoints proxy requests to the external graphrag microservice
which provides advanced knowledge graph features using Neo4j and GLiNER.
"""

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from typing import Optional, Any, List
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


@router.post("/ingest/upload")
async def graphrag_upload_document(
    file: UploadFile = File(...),
    domain: str = Form("general"),
    use_semantic_chunking: bool = Form(True),
    build_knowledge_graph: bool = Form(True)
):
    """Upload and process a document through GraphRAG pipeline."""
    check_graphrag_enabled()
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    try:
        # Read file content
        content = await file.read()
        
        # Create multipart form data for GraphRAG
        files = {"file": (file.filename, content, file.content_type or "application/octet-stream")}
        data = {
            "use_semantic_chunking": str(use_semantic_chunking).lower()
        }
        
        async with httpx.AsyncClient(timeout=180.0) as client:
            # Upload and process document
            response = await client.post(
                f"{GRAPHRAG_URL}/process-document",
                files=files,
                data=data
            )
            response.raise_for_status()
            result = response.json()
            
            # If build_knowledge_graph is enabled, trigger ingestion
            if build_knowledge_graph:
                try:
                    ingest_response = await client.post(
                        f"{GRAPHRAG_URL}/ingest-documents",
                        json={
                            "documents": [file.filename],
                            "domain": domain,
                            "extract_entities": True,
                            "extract_relationships": True,
                            "build_knowledge_graph": True
                        }
                    )
                    if ingest_response.status_code == 200:
                        ingest_data = ingest_response.json()
                        result["knowledge_graph"] = {
                            "entities": ingest_data.get("entities_extracted", 0),
                            "relationships": ingest_data.get("relationships_extracted", 0)
                        }
                except Exception as e:
                    logger.warning(f"Knowledge graph building failed: {e}")
                    result["knowledge_graph_error"] = str(e)
            
            return result
    
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG upload error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")


@router.post("/ingest/batch")
async def graphrag_upload_batch(
    files: List[UploadFile] = File(...),
    domain: str = Form("general"),
    use_semantic_chunking: bool = Form(True)
):
    """Batch upload documents to GraphRAG."""
    check_graphrag_enabled()
    
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    
    try:
        # Prepare files for GraphRAG
        files_data = []
        for file in files:
            content = await file.read()
            files_data.append(
                ("files", (file.filename, content, file.content_type or "application/octet-stream"))
            )
        
        data = {
            "use_semantic_chunking": str(use_semantic_chunking).lower()
        }
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{GRAPHRAG_URL}/process-documents-batch",
                files=files_data,
                data=data
            )
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG batch upload error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")
    except Exception as e:
        logger.error(f"Batch upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading documents: {str(e)}")


@router.post("/rebuild")
async def graphrag_rebuild_kg(request: Request):
    """Rebuild knowledge graph from existing documents."""
    check_graphrag_enabled()
    
    try:
        data = await request.json()
    except:
        data = {}
    
    domain = data.get("domain")
    
    try:
        payload = {}
        if domain:
            payload["domain"] = domain
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{GRAPHRAG_URL}/rebuild-knowledge-graph",
                json=payload
            )
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG rebuild error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")
    except Exception as e:
        logger.error(f"Rebuild error: {e}")
        raise HTTPException(status_code=500, detail=f"Error rebuilding knowledge graph: {str(e)}")


@router.post("/export")
async def graphrag_export_graph(request: Request):
    """Export knowledge graph data."""
    check_graphrag_enabled()
    
    try:
        data = await request.json()
    except:
        data = {}
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                f"{GRAPHRAG_URL}/knowledge-graph/export",
                params=data
            )
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
        async with httpx.AsyncClient(timeout=30.0) as client:
            # GraphRAG expects query as a query parameter
            response = await client.get(
                f"{GRAPHRAG_URL}/api/analyze-query-intent",
                params={"query": query}
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
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Step 1: Analyze query intent if not already provided
            search_type = data.get("search_type", "auto")
            
            if search_type == "auto":
                # Let GraphRAG determine the best search method
                search_type = "hybrid"  # Default to hybrid for now
            
            # Step 2: Perform advanced search with LLM answer generation
            search_payload = {
                "query": query,
                "search_type": search_type,
                "top_k": data.get("top_k", 10),
                "domain": data.get("domain"),
                "filters": data.get("filters")
            }
            
            response = await client.post(
                f"{GRAPHRAG_URL}/search-advanced",
                json=search_payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Add search method used
            result["search_method_used"] = search_type
            
            return result
    
    except httpx.HTTPError as e:
        logger.error(f"Intelligent search error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")


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
        async with httpx.AsyncClient(timeout=60.0) as client:
            # GraphRAG expects source, target, entities, relationships
            # We need to fetch entities and relationships first
            payload = {
                "source": source,
                "target": target,
                "entities": data.get("entities", []),
                "relationships": data.get("relationships", [])
            }
            
            response = await client.post(
                f"{GRAPHRAG_URL}/reasoning/explain-relationship",
                json=payload
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
        async with httpx.AsyncClient(timeout=120.0) as client:
            # GraphRAG endpoint expects query as form data
            response = await client.post(
                f"{GRAPHRAG_URL}/api/causal-reasoning",
                data={"query": query}
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
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{GRAPHRAG_URL}/api/comparative-reasoning",
                data={"query": query}
            )
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPError as e:
        logger.error(f"Comparative reasoning error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/entity/link")
async def graphrag_link_entities(request: Request):
    """Link entities to knowledge graph."""
    check_graphrag_enabled()
    
    data = await request.json()
    entities = data.get("entities", [])
    context = data.get("context", "")
    
    if not entities:
        raise HTTPException(status_code=400, detail="entities array is required")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{GRAPHRAG_URL}/entity/link",
                json={
                    "entities": entities,
                    "context": context
                }
            )
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPError as e:
        logger.error(f"Entity linking error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/entity/disambiguate")
async def graphrag_disambiguate_entities(request: Request):
    """Disambiguate entity mentions."""
    check_graphrag_enabled()
    
    data = await request.json()
    entity_name = data.get("entity_name")
    candidates = data.get("candidates", [])
    context = data.get("context", "")
    
    if not entity_name:
        raise HTTPException(status_code=400, detail="entity_name is required")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{GRAPHRAG_URL}/entity/disambiguate",
                json={
                    "entity_name": entity_name,
                    "candidates": candidates,
                    "context": context
                }
            )
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPError as e:
        logger.error(f"Entity disambiguation error: {e}")
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
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{GRAPHRAG_URL}/api/advanced-reasoning",
                data={"query": query}
            )
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPError as e:
        logger.error(f"Advanced reasoning error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/code/health")
async def graphrag_code_health():
    """Check code search service health."""
    check_graphrag_enabled()
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{GRAPHRAG_URL}/code-detection/health")
            return response.json()
    except Exception as e:
        return {"available": False, "error": str(e)}


@router.post("/code/detect")
async def graphrag_detect_code(request: Request):
    """Detect code in text."""
    check_graphrag_enabled()
    
    data = await request.json()
    text = data.get("text")
    
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{GRAPHRAG_URL}/code-detection/detect",
                json={"text": text}
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
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{GRAPHRAG_URL}/search/code",
                json=data
            )
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPError as e:
        logger.error(f"Code search error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


# ============== Hybrid Processing Endpoints ==============

@router.get("/hybrid/status")
async def graphrag_hybrid_status():
    """Get status of GraphRAG, Code RAG, and hybrid availability."""
    check_graphrag_enabled()
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{GRAPHRAG_URL}/hybrid/status")
            return response.json()
    except Exception as e:
        return {
            "graphrag_healthy": False,
            "code_rag_healthy": False,
            "hybrid_available": False,
            "error": str(e)
        }


@router.post("/hybrid/process")
async def graphrag_hybrid_process(
    file: UploadFile = File(...),
    domain: str = Form("general")
):
    """Process file with hybrid routing (code to Code RAG, docs to GraphRAG)."""
    check_graphrag_enabled()
    
    try:
        content = await file.read()
        files = {"file": (file.filename, content, file.content_type or "application/octet-stream")}
        data = {"domain": domain}
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{GRAPHRAG_URL}/hybrid/process",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPError as e:
        logger.error(f"Hybrid processing error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")
    except Exception as e:
        logger.error(f"Hybrid processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/extraction-stats")
async def graphrag_extraction_stats():
    """Get entity extraction method statistics."""
    check_graphrag_enabled()
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{GRAPHRAG_URL}/extraction-stats")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Extraction stats error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/supported-formats")
async def graphrag_supported_formats():
    """Get supported file formats and features."""
    check_graphrag_enabled()
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{GRAPHRAG_URL}/supported-formats")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Supported formats error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")
