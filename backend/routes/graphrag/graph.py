"""
GraphRAG Graph Routes

Knowledge graph queries: filtered graph, entity neighborhoods,
top entities/relationships, communities, export.

All endpoints use the shared connection-pooled HTTP client.
"""

from fastapi import APIRouter, HTTPException, Request
from typing import Optional
import httpx
import logging

from .helpers import (
    check_graphrag_enabled, get_http_client, graphrag_url as _graphrag_url,
    _transform_nodes, _fetch_graph_data, _find_entity_neighbors,
)
from modules.graphrag.config import get_graphrag_config

logger = logging.getLogger(__name__)

router = APIRouter()


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
    except Exception:
        data = {}

    max_entities = data.get("max_entities", 500)
    max_relationships = data.get("max_relationships", 500)
    domain = data.get("domain")

    try:
        client = get_http_client()
        # Get filtered graph data
        response = await client.post(
            f"{_graphrag_url()}/knowledge-graph/filtered",
            json=data,
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
        if nodes and not edges:
            logger.info(
                f"Filtered endpoint returned {len(nodes)} nodes but 0 edges, "
                "fetching relationships separately"
            )

            node_ids = {n.get("id") for n in nodes}
            node_labels = {n.get("label") for n in nodes}
            all_identifiers = node_ids | node_labels

            params = {"limit": max_relationships}
            if domain:
                params["domain"] = domain

            rel_response = await client.get(
                f"{_graphrag_url()}/knowledge-graph/top-relationships",
                params=params,
            )

            if rel_response.status_code == 200:
                rel_data = rel_response.json()
                top_rels = rel_data.get("top_relationships", [])

                edge_id = 0
                for rel in top_rels:
                    source = rel.get("source", "")
                    target = rel.get("target", "")

                    if source in all_identifiers and target in all_identifiers:
                        edges.append({
                            "id": f"edge_{edge_id}",
                            "source": source,
                            "target": target,
                            "label": rel.get("type", "related_to"),
                            "type": rel.get("type", "related_to"),
                            "weight": rel.get("weight", 1),
                        })
                        edge_id += 1

                logger.info(f"Added {len(edges)} edges from top-relationships matching {len(nodes)} nodes")

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": filtered_data.get("stats", {
                "nodes": len(nodes),
                "edges": len(edges),
            }),
        }

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/top-entities")
async def graphrag_top_entities(
    domain: Optional[str] = None,
    limit: int = 50,
    min_occurrence: int = 1,
):
    """Get top entities by occurrence from graphrag service."""
    check_graphrag_enabled()

    params = {"limit": limit, "min_occurrence": min_occurrence}
    if domain:
        params["domain"] = domain

    try:
        client = get_http_client()
        response = await client.get(
            f"{_graphrag_url()}/knowledge-graph/top-entities",
            params=params,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/top-relationships")
async def graphrag_top_relationships(
    domain: Optional[str] = None,
    limit: int = 50,
    min_weight: int = 1,
):
    """Get top relationships by weight from graphrag service."""
    check_graphrag_enabled()

    params = {"limit": limit, "min_weight": min_weight}
    if domain:
        params["domain"] = domain

    try:
        client = get_http_client()
        response = await client.get(
            f"{_graphrag_url()}/knowledge-graph/top-relationships",
            params=params,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/entity-neighborhood")
async def graphrag_entity_neighborhood(request: Request):
    """Get 1-hop neighborhood for an entity."""
    check_graphrag_enabled()

    data = await request.json()
    entity_name = data.get("entity_name", "").strip()
    if not entity_name:
        raise HTTPException(status_code=400, detail="entity_name is required")

    limit = data.get("limit", 20)
    domain = data.get("domain")

    try:
        client = get_http_client()
        node_map, raw_edges = await _fetch_graph_data(client, domain)
        return _find_entity_neighbors(entity_name, node_map, raw_edges, limit)
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
        client = get_http_client()
        response = await client.get(
            f"{_graphrag_url()}/knowledge-graph/domain-stats",
            params=params,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/export")
async def graphrag_export_graph(request: Request):
    """Export knowledge graph data."""
    check_graphrag_enabled()

    try:
        data = await request.json()
    except Exception:
        data = {}

    try:
        client = get_http_client()
        response = await client.get(
            f"{_graphrag_url()}/knowledge-graph/export",
            params=data,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")
