"""
GraphRAG Helpers — Shared constants, utilities, and graph traversal functions.

All configuration is sourced from the unified GraphRAGConfig singleton.
An httpx connection pool is shared across all route modules.
"""

from fastapi.exceptions import HTTPException
import httpx
import logging
from functools import lru_cache

from modules.graphrag.config import get_graphrag_config

logger = logging.getLogger(__name__)

CHAT_CONTEXT_OPTIONAL_TIMEOUT_SECONDS = 2.0

GRAPHRAG_COMPATIBLE_INGEST_PATHS = (
    "ingest-documents",
    "process-document",
    "add-to-vector-store",
)

GRAPHRAG_COMPATIBLE_HEALTH_PATHS = (
    "health",
    "api/health",
    "api/v1/health",
)


# ── Config shorthand ───────────────────────────────────────────────────

def _cfg():
    """Shorthand for the singleton config."""
    return get_graphrag_config()


def graphrag_url() -> str:
    """Return the GraphRAG service URL, or raise 503 if not configured.

    Shared by search, ingest, and graph route modules.
    """
    url = _cfg().graphrag_url
    if not url:
        raise HTTPException(
            status_code=503,
            detail="GRAPHRAG_URL not configured. Set GRAPHRAG_URL to the external GraphRAG service.",
        )
    return url


def compatible_ingest_urls(service_url: str | None) -> list[str]:
    """Return GraphRAG ingest endpoints in the compatibility fallback order."""
    if not service_url:
        return []
    base_url = service_url.rstrip("/")
    return [f"{base_url}/{path}" for path in GRAPHRAG_COMPATIBLE_INGEST_PATHS]


def compatible_health_urls(service_url: str | None) -> list[str]:
    """Return GraphRAG health endpoints in the compatibility fallback order."""
    if not service_url:
        return []
    base_url = service_url.rstrip("/")
    return [f"{base_url}/{path}" for path in GRAPHRAG_COMPATIBLE_HEALTH_PATHS]


# ── Shared HTTP client ─────────────────────────────────────────────────

_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """
    Return a module-level httpx.AsyncClient with connection pooling.

    Connection limits are configured from GraphRAGConfig.
    The client is lazily created on first use and reused for all
    subsequent requests — avoiding per-request TCP/TLS overhead.
    """
    global _http_client
    if _http_client is None or _http_client.is_closed:
        cfg = _cfg()
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.http_timeout, connect=10.0),
            limits=httpx.Limits(
                max_connections=cfg.http_pool_size,
                max_keepalive_connections=cfg.http_pool_size,
            ),
            follow_redirects=True,
        )
    return _http_client


def optional_chat_context_timeout() -> float:
    """Short timeout for optional enrichment calls on interactive chat paths."""
    cfg = _cfg()
    try:
        configured = float(cfg.http_timeout)
    except (TypeError, ValueError):
        configured = CHAT_CONTEXT_OPTIONAL_TIMEOUT_SECONDS
    if configured <= 0:
        return CHAT_CONTEXT_OPTIONAL_TIMEOUT_SECONDS
    return max(0.25, min(configured, CHAT_CONTEXT_OPTIONAL_TIMEOUT_SECONDS))


async def close_http_client():
    """Close the shared client (call from FastAPI shutdown handler)."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


# ── Guard ──────────────────────────────────────────────────────────────

def check_graphrag_enabled():
    """Check if GraphRAG is enabled and raise exception if not."""
    if not _cfg().enabled:
        raise HTTPException(
            status_code=503,
            detail="GraphRAG service is disabled. Set GRAPHRAG_ENABLED=true",
        )


# ── Entity type normalization ──────────────────────────────────────────

_TYPE_MAP = {
    "component": "technology",
    "maintenance": "process",
    "organisation": "organization",
    "organization": "organization",
    "org": "organization",
    "company": "organization",
    "person": "person",
    "location": "location",
    "date": "date",
    "time": "date",
    "duration": "date",
    "event": "event",
    "product": "product",
    "solution": "product",
    "system": "service",
    "service": "service",
    "api": "api",
    "endpoint": "api",
    "specification": "concept",
    "position": "concept",
    "format": "concept",
    "model": "model",
    "dataset": "data",
    "document": "data",
    "database": "data",
    "requirement": "config",
    "safety": "security",
    "metric": "metric",
    "benchmark": "evaluation",
    "workflow": "process",
    "pipeline": "process",
    "module": "code",
    "file": "code",
    "technology": "technology",
    "concept": "concept",
    "process": "process",
    "entity": "concept",
}


def _normalize_entity_type(raw_type: str) -> str:
    """Normalize entity type to match frontend expected values.

    GraphRAG returns uppercase types like COMPONENT, MAINTENANCE, ORGANIZATION.
    Frontend expects lowercase: person, organization, location, technology, etc.
    """
    if not raw_type:
        return "concept"
    return _TYPE_MAP.get(str(raw_type).strip().lower(), "custom")


def _transform_nodes(raw_nodes: list) -> list:
    """Transform GraphRAG nodes to frontend format.

    GraphRAG returns: {id, label, type: "Entity", properties: {type: "COMPONENT", ...}}
    Frontend expects: {id, label, type: "technology", properties: {...}}
    """
    transformed = []
    for node in raw_nodes:
        props = node.get("properties", {})
        entity_type = props.get("type") or node.get("type", "concept")
        normalized_type = _normalize_entity_type(entity_type)

        transformed.append({
            "id": node.get("id") or props.get("name", ""),
            "label": node.get("label") or props.get("name", ""),
            "type": normalized_type,
            "occurrence": node.get("occurrence") or props.get("occurrence", 1),
            "properties": {
                **props,
                "description": props.get("description", ""),
                "confidence": props.get("confidence", 1.0),
                "domain": props.get("domain", "general"),
                "entity_type": entity_type,
                "original_type": entity_type,
            },
        })

    return transformed


def _edge_endpoint_id(value) -> str:
    if isinstance(value, dict):
        return value.get("id") or value.get("label") or value.get("name") or ""
    return value or ""


def _edge_relation_count(edge: dict, properties: dict) -> int:
    for key in ("relationCount", "relation_count", "count"):
        value = edge.get(key, properties.get(key))
        if value is not None:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                continue

    relation_types = properties.get("relationTypes") or properties.get("relation_types")
    if isinstance(relation_types, list):
        count = 0
        for item in relation_types:
            if not isinstance(item, dict):
                continue
            try:
                count += max(1, int(item.get("count", 1)))
            except (TypeError, ValueError):
                count += 1
        if count > 0:
            return count

    return 1


def _transform_edge(edge: dict, edge_id: int = 0) -> dict:
    """Transform GraphRAG relationship payloads to the frontend edge contract."""
    raw_properties = edge.get("properties", {})
    properties = dict(raw_properties) if isinstance(raw_properties, dict) else {}
    relation_type = (
        edge.get("type")
        or edge.get("relationship_type")
        or edge.get("label")
        or edge.get("relation")
        or properties.get("type")
        or "related_to"
    )
    weight = edge.get("weight", properties.get("weight", 1))
    relation_count = _edge_relation_count(edge, properties)

    for key, value in edge.items():
        if key not in {
            "id",
            "source",
            "target",
            "label",
            "type",
            "relationship_type",
            "relation",
            "weight",
            "relationCount",
            "relation_count",
            "count",
            "properties",
        } and value is not None:
            properties.setdefault(key, value)

    properties["type"] = relation_type
    properties["weight"] = weight
    properties["relationCount"] = relation_count
    properties.setdefault("relationTypes", [{"type": relation_type, "count": relation_count}])

    return {
        "id": edge.get("id") or f"edge_{edge_id}",
        "source": _edge_endpoint_id(edge.get("source")),
        "target": _edge_endpoint_id(edge.get("target")),
        "label": edge.get("label") or relation_type,
        "type": relation_type,
        "weight": weight,
        "relationCount": relation_count,
        "properties": properties,
    }


# ── Graph data fetching ────────────────────────────────────────────────

async def _fetch_graph_data(
    client: httpx.AsyncClient,
    domain: str = None,
    timeout: float | None = None,
) -> tuple:
    """Fetch graph nodes and edges from GraphRAG service.

    Returns (node_map, raw_edges) where node_map is {lowercase_name: node_info}.
    Shared by entity-neighborhood and chat-context expansion.
    """
    cfg = _cfg()
    if not cfg.graphrag_url:
        return {}, []

    filter_payload = {"max_entities": 200, "max_relationships": 500}
    if domain:
        filter_payload["domain"] = domain

    response = await client.post(
        f"{cfg.graphrag_url}/knowledge-graph/filtered",
        json=filter_payload,
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()

    filtered_data = result.get("filtered_data", result)
    raw_nodes = filtered_data.get("nodes", [])
    raw_edges = filtered_data.get("edges", [])

    # Fetch top relationships if edges empty
    if not raw_edges:
        rel_params = {"limit": 500}
        if domain:
            rel_params["domain"] = domain
        rel_response = await client.get(
            f"{cfg.graphrag_url}/knowledge-graph/top-relationships",
            params=rel_params,
            timeout=timeout,
        )
        if rel_response.status_code == 200:
            rel_data = rel_response.json()
            for rel in rel_data.get("top_relationships", []):
                raw_edges.append({
                    "source": rel.get("source", ""),
                    "target": rel.get("target", ""),
                    "label": rel.get("type", "related_to"),
                    "type": rel.get("type", "related_to"),
                    "weight": rel.get("weight", 1),
                })

    # Build node lookup map
    node_map = {}
    for node in raw_nodes:
        props = node.get("properties", {})
        label = node.get("label") or props.get("name", "")
        entity_type = props.get("type") or node.get("type", "concept")
        node_id = node.get("id") or label
        node_map[label.lower()] = {
            "name": label,
            "type": _normalize_entity_type(entity_type),
            "id": node_id,
            "occurrence": node.get("occurrence") or props.get("occurrence", 1),
            "description": props.get("description", ""),
        }

    return node_map, raw_edges


def _find_entity_neighbors(entity_name: str, node_map: dict, raw_edges: list, limit: int = 20) -> dict:
    """Find 1-hop neighborhood for an entity in pre-fetched graph data.

    Returns dict with center, neighbors, relationships, found_in_graph.
    """
    # Fuzzy match center entity
    center = None
    search_lower = entity_name.lower()
    for key, node_info in node_map.items():
        if key == search_lower or search_lower in key or key in search_lower:
            center = node_info
            break

    if not center:
        return {
            "center": {"name": entity_name, "type": "unknown", "description": ""},
            "neighbors": [],
            "relationships": [],
            "found_in_graph": False,
        }

    # Find connected edges and neighbors
    neighbors = {}
    relationships = []
    center_name_lower = center["name"].lower()

    for edge in raw_edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        rel_type = edge.get("label") or edge.get("type", "related_to")
        weight = edge.get("weight", 1)

        source_lower = source.lower()
        target_lower = target.lower()

        if source_lower == center_name_lower:
            neighbor_name = target
            direction = "outgoing"
        elif target_lower == center_name_lower:
            neighbor_name = source
            direction = "incoming"
        else:
            continue

        neighbor_lower = neighbor_name.lower()
        if neighbor_lower not in neighbors:
            neighbor_info = node_map.get(neighbor_lower, {
                "name": neighbor_name,
                "type": "concept",
                "occurrence": 1,
                "description": "",
            })
            neighbors[neighbor_lower] = neighbor_info

        relationships.append({
            "source": source,
            "target": target,
            "relation": rel_type,
            "direction": direction,
            "weight": weight,
        })

    # Sort neighbors by occurrence and limit
    neighbor_list = sorted(
        neighbors.values(),
        key=lambda n: n.get("occurrence", 1),
        reverse=True,
    )[:limit]

    return {
        "center": center,
        "neighbors": neighbor_list,
        "relationships": relationships[: limit * 2],
        "found_in_graph": True,
    }


# ── Shared extraction pipeline helpers ─────────────────────────────────
# Used by both /chat-context and /chat-context-v2 in context.py


async def _extract_entities_ner(
    client: httpx.AsyncClient, query: str, errors: list
) -> list[dict]:
    """Extract entities from query text using the NER service.

    Handles subword token merging (e.g. '##F', '##old' → 'Fold').
    Returns list of {name, type, score} dicts.
    """
    cfg = _cfg()
    if not cfg.ner_api_url:
        return []

    entities = []
    try:
        ner_response = await client.post(
            f"{cfg.ner_api_url}/ner",
            json={"text": query},
            timeout=optional_chat_context_timeout(),
        )
        if ner_response.status_code == 200:
            ner_data = ner_response.json()
            for ent in ner_data.get("entities", []):
                word = ent.get("word", "")
                if word.startswith("##"):
                    if entities:
                        entities[-1]["name"] += word[2:]
                    continue
                entities.append({
                    "name": word,
                    "type": ent.get("entity", "UNKNOWN").replace("I-", "").replace("B-", ""),
                    "score": round(ent.get("score", 0), 3),
                })
    except Exception as e:
        logger.warning(f"NER extraction failed: {e}")
        errors.append(f"NER: {str(e)}")

    return entities


async def _extract_relationships_rel(
    client: httpx.AsyncClient, query: str, errors: list
) -> list[dict]:
    """Extract relationships from query text using the REL service.

    Relation types are sourced from GraphRAGConfig.default_relation_types.
    Returns list of {source, target, relation, score} dicts.
    """
    cfg = _cfg()
    if not cfg.rel_api_url:
        return []

    relationships = []
    try:
        rel_response = await client.post(
            f"{cfg.rel_api_url}/extract-relations",
            json={
                "text": query,
                "relations": [{"relation": r} for r in cfg.default_relation_types],
                "threshold": 0.5,
            },
            timeout=optional_chat_context_timeout(),
        )
        if rel_response.status_code == 200:
            rel_data = rel_response.json()
            for rel in rel_data.get("relations", []):
                relationships.append({
                    "source": rel.get("source", ""),
                    "target": rel.get("target", ""),
                    "relation": rel.get("label", rel.get("relation", "")),
                    "score": round(rel.get("score", 0), 3),
                })
    except Exception as e:
        logger.warning(f"REL extraction failed: {e}")
        errors.append(f"REL: {str(e)}")

    return relationships


async def _expand_graph_neighborhood(
    client: httpx.AsyncClient,
    entities: list[dict],
    domain: str,
    errors: list,
    max_entities: int = 3,
    neighbor_limit: int = 5,
    timeout: float | None = None,
) -> tuple[list[dict], list[str]]:
    """Expand top entities via 1-hop graph neighborhood.

    Returns (expanded_entities, graph_connections).
    """
    expanded_entities = []
    graph_connections = []

    cfg = _cfg()
    if not cfg.graphrag_url or not entities:
        return expanded_entities, graph_connections

    try:
        node_map, raw_edges = await _fetch_graph_data(client, domain, timeout=timeout)

        top_entities = sorted(entities, key=lambda e: e.get("score", 0), reverse=True)[:max_entities]
        seen_entity_names = {e["name"].lower() for e in entities}

        for ent in top_entities:
            neighborhood = _find_entity_neighbors(ent["name"], node_map, raw_edges, limit=neighbor_limit)

            if neighborhood.get("found_in_graph"):
                for rel in neighborhood.get("relationships", [])[:neighbor_limit]:
                    conn_str = f"{rel['source']} --[{rel['relation']}]--> {rel['target']}"
                    if conn_str not in graph_connections:
                        graph_connections.append(conn_str)

                for neighbor in neighborhood.get("neighbors", [])[:3]:
                    if neighbor["name"].lower() not in seen_entity_names:
                        seen_entity_names.add(neighbor["name"].lower())
                        expanded_entities.append({
                            "name": neighbor["name"],
                            "type": neighbor.get("type", "ENTITY"),
                            "score": 0.7,
                            "source": "graph_expansion",
                        })

        if expanded_entities:
            logger.info(
                f"Graph expansion discovered {len(expanded_entities)} new entities, "
                f"{len(graph_connections)} connections"
            )
    except Exception as e:
        logger.warning(f"Graph expansion failed: {e}")
        errors.append(f"GraphExpansion: {str(e)}")

    return expanded_entities, graph_connections


def _deduplicate_entities(entities: list[dict], expanded: list[dict]) -> list[dict]:
    """Deduplicate entities by name (case-insensitive), filtering short names."""
    seen = set()
    unique = []
    for ent in entities + expanded:
        key = ent["name"].lower().strip()
        if key and len(key) >= 2 and key not in seen:
            seen.add(key)
            unique.append(ent)
    return unique


def _deduplicate_relationships(relationships: list[dict]) -> list[dict]:
    """Deduplicate relationships by (source, relation, target) key."""
    seen = set()
    unique = []
    for rel in relationships:
        key = f"{rel['source'].lower()}|{rel['relation'].lower()}|{rel['target'].lower()}"
        if key not in seen:
            seen.add(key)
            unique.append(rel)
    return unique
