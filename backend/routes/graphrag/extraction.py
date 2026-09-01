"""
GraphRAG Extraction & Enhanced Query Routes

Typed adapters for GraphRAG's lower-level NER, relationship extraction,
enhanced query, and query/graph instrumentation endpoints.
"""

from typing import Any, Dict
import logging

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .helpers import check_graphrag_enabled, get_http_client, graphrag_url as _graphrag_url
from modules.graphrag.config import get_graphrag_config

logger = logging.getLogger(__name__)

router = APIRouter()


class NERExtractionRequest(BaseModel):
    text: str
    return_offsets: bool = True
    return_scores: bool = True


class NERBatchExtractionRequest(BaseModel):
    texts: list[str]
    return_offsets: bool = True
    return_scores: bool = True


class RelationExtractionRequest(BaseModel):
    text: str
    relations: list[Dict[str, Any]] = Field(default_factory=list)
    entity_labels: list[str] = Field(default_factory=list)
    threshold: float = 0.5


class RelationBatchExtractionRequest(BaseModel):
    texts: list[str]
    relations: list[Dict[str, Any]] = Field(default_factory=list)
    entity_labels: list[str] = Field(default_factory=list)
    threshold: float = 0.5


class RelationEntityExtractionRequest(BaseModel):
    text: str
    labels: list[str] = Field(default_factory=list)
    threshold: float = 0.5


class EnhancedExtractionRequest(BaseModel):
    text: str
    domain: str = "general"
    use_spanbert: bool = True
    use_dependency: bool = True
    use_entity_linking: bool = True


class EnhancedQueryRequest(BaseModel):
    query: str
    search_type: str = "hybrid"
    top_k: int = Field(default=10, ge=1)
    domain: str | None = None
    filters: Dict[str, Any] | None = None


class EnhancedAnswerRequest(BaseModel):
    query: str
    context: Dict[str, Any] | None = None


def _json_or_text(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def _with_normalized_entity(entity: Any) -> Any:
    if not isinstance(entity, dict):
        return entity

    normalized = dict(entity)
    source_name = entity.get("name") or entity.get("text") or entity.get("word") or entity.get("span")
    source_type = entity.get("type") or entity.get("entity_type") or entity.get("label")

    # Many NER services return spaCy-style {word, entity, score}; keep those
    # fields while adding the frontend-friendly names used by llama-nexus.
    if not source_name and isinstance(entity.get("entity"), str):
        source_name = entity.get("entity")
    elif not source_type and isinstance(entity.get("entity"), str):
        source_type = entity.get("entity")

    if source_name is not None and "name" not in normalized:
        normalized["name"] = source_name
    if source_type is not None and "type" not in normalized:
        normalized["type"] = source_type
    if "confidence" not in normalized and entity.get("score") is not None:
        normalized["confidence"] = entity.get("score")

    return normalized


def _with_entity_count(payload: Any) -> Any:
    if isinstance(payload, dict):
        entities = payload.get("entities")
        if isinstance(entities, list):
            next_payload = {**payload, "entities": [_with_normalized_entity(entity) for entity in entities]}
            next_payload.setdefault("entity_count", len(entities))
            return next_payload
    return payload


def _with_relation_count(payload: Any) -> Any:
    if isinstance(payload, dict):
        relations = payload.get("relations")
        if isinstance(relations, list):
            next_payload = dict(payload)
            next_payload.setdefault("relation_count", len(relations))
            next_payload.setdefault("relationship_count", len(relations))
            return next_payload
    return payload


def _with_batch_totals(payload: Any, total_key: str, item_count_key: str, item_list_key: str) -> Any:
    if not isinstance(payload, dict):
        return payload

    next_payload = dict(payload)
    results = next_payload.get("results")

    if isinstance(results, list):
        next_payload.setdefault(total_key, len(results))
        next_results = []
        for item in results:
            if isinstance(item, dict):
                next_item: Any = item
                if item_list_key == "entities":
                    next_item = _with_entity_count(item)
                elif item_list_key == "relations":
                    next_item = _with_relation_count(item)

                items = next_item.get(item_list_key) if isinstance(next_item, dict) else None
                if isinstance(items, list) and item_count_key not in next_item:
                    next_results.append({**next_item, item_count_key: len(items)})
                else:
                    next_results.append(next_item)
            else:
                next_results.append(item)
        next_payload["results"] = next_results

    return next_payload


async def _call_graphrag(
    method: str,
    path: str,
    *,
    params: Dict[str, Any] | None = None,
    json: Any = None,
    data: Dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> Any:
    check_graphrag_enabled()

    try:
        client = get_http_client()
        response = await client.request(
            method,
            f"{_graphrag_url()}/{path}",
            params=params,
            json=json,
            data=data,
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG extraction/query request failed: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")

    payload = _json_or_text(response)
    if response.status_code >= 400:
        status_code = response.status_code if response.status_code < 500 else 502
        raise HTTPException(status_code=status_code, detail=payload)

    return payload


async def _call_service_url(
    method: str,
    base_url: str,
    path: str,
    *,
    params: Dict[str, Any] | None = None,
    json: Any = None,
    timeout: float = 120.0,
) -> Any:
    check_graphrag_enabled()

    try:
        client = get_http_client()
        response = await client.request(
            method,
            f"{base_url.rstrip('/')}/{path.lstrip('/')}",
            params=params,
            json=json,
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG configured service request failed: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")

    payload = _json_or_text(response)
    if response.status_code >= 400:
        status_code = response.status_code if response.status_code < 500 else 502
        raise HTTPException(status_code=status_code, detail=payload)

    return payload


@router.post("/ner/extract")
async def graphrag_ner_extract(request: NERExtractionRequest):
    """Extract entities from text using GraphRAG's GLiNER endpoint."""
    cfg = get_graphrag_config()

    if cfg.ner_api_url:
        payload = await _call_service_url(
            "POST",
            cfg.ner_api_url,
            "ner",
            json=request.model_dump(),
        )
        return _with_entity_count(payload)

    payload = await _call_graphrag(
        "POST",
        "ner/extract",
        json=request.text,
    )
    return _with_entity_count(payload)


@router.post("/ner/extract-batch")
async def graphrag_ner_extract_batch(request: NERBatchExtractionRequest):
    """Extract entities from multiple texts using GraphRAG's GLiNER endpoint."""
    cfg = get_graphrag_config()

    if cfg.ner_api_url:
        payload = await _call_service_url(
            "POST",
            cfg.ner_api_url,
            "ner/batch",
            json=request.model_dump(),
            timeout=180.0,
        )
        return _with_batch_totals(payload, "total_texts", "entity_count", "entities")

    payload = await _call_graphrag(
        "POST",
        "ner/extract-batch",
        json=request.texts,
        timeout=180.0,
    )
    return _with_batch_totals(payload, "total_texts", "entity_count", "entities")


@router.get("/ner/model-info")
async def graphrag_ner_model_info():
    """Get model metadata for the configured NER service."""
    cfg = get_graphrag_config()
    if not cfg.enabled:
        return {"available": False, "status": "disabled", "model_info": {}}
    if cfg.ner_api_url:
        return await _call_service_url("GET", cfg.ner_api_url, "model-info")
    if not cfg.graphrag_url:
        return {"available": False, "status": "not_configured", "message": "Set NER_API_URL or GRAPHRAG_URL", "model_info": {}}

    return await _call_graphrag("GET", "ner/status")


@router.get("/ner/entities")
async def graphrag_ner_entities():
    """List entity labels recognized by the configured NER service."""
    cfg = get_graphrag_config()
    if not cfg.enabled:
        return {"available": False, "status": "disabled", "entity_types": []}
    if cfg.ner_api_url:
        return await _call_service_url("GET", cfg.ner_api_url, "ner/entities")
    if not cfg.graphrag_url:
        return {"available": False, "status": "not_configured", "message": "Set NER_API_URL or GRAPHRAG_URL", "entity_types": []}

    status = await _call_graphrag("GET", "ner/status")
    model_info = status.get("model_info", {}) if isinstance(status, dict) else {}
    entity_types = (
        model_info.get("default_entity_labels")
        or model_info.get("entity_types")
        or []
    )
    return {
        "entity_types": entity_types,
        "source": "graphrag_status",
        "model_info": model_info,
    }


@router.get("/rel/status")
async def graphrag_rel_status():
    """Check GraphRAG relationship extraction availability."""
    cfg = get_graphrag_config()
    if not cfg.enabled:
        return {"rel_available": False, "status": "disabled", "model_info": {}}
    if cfg.rel_api_url:
        try:
            health = await _call_service_url("GET", cfg.rel_api_url, "health", timeout=60.0)
        except HTTPException as e:
            return {
                "rel_available": False,
                "status": "unavailable",
                "service": {"available": False, "url": cfg.rel_api_url, "error": e.detail},
                "model_info": {},
                "extraction_method": "dedicated_rel_service",
            }

        model_payload = {}
        if isinstance(health, dict) and health.get("status") == "healthy":
            try:
                model_payload = await _call_service_url("GET", cfg.rel_api_url, "model-info", timeout=60.0)
            except HTTPException:
                model_payload = {}

        model_info = model_payload.get("model_info", model_payload) if isinstance(model_payload, dict) else {}
        return {
            "rel_available": isinstance(health, dict) and health.get("status") == "healthy",
            "status": health.get("status", "unavailable") if isinstance(health, dict) else "unavailable",
            "service": health,
            "model_info": model_info if isinstance(model_info, dict) else {},
            "extraction_method": "dedicated_rel_service",
        }

    if not cfg.graphrag_url:
        return {
            "rel_available": False,
            "status": "not_configured",
            "message": "Set REL_API_URL or GRAPHRAG_URL",
            "model_info": {},
        }

    return await _call_graphrag("GET", "rel/status")


@router.get("/rel/model-info")
async def graphrag_rel_model_info():
    """Get model metadata for the configured relationship extraction service."""
    cfg = get_graphrag_config()
    if not cfg.enabled:
        return {"available": False, "status": "disabled", "model_info": {}}
    if cfg.rel_api_url:
        return await _call_service_url("GET", cfg.rel_api_url, "model-info")
    if not cfg.graphrag_url:
        return {"available": False, "status": "not_configured", "message": "Set REL_API_URL or GRAPHRAG_URL", "model_info": {}}

    status = await _call_graphrag("GET", "rel/status")
    return {
        "model_info": status.get("model_info") if isinstance(status, dict) else None,
        "rel_status": status,
    }


@router.get("/rel/capabilities")
async def graphrag_rel_capabilities():
    """List relationship extraction capabilities and default schemas."""
    cfg = get_graphrag_config()
    if not cfg.enabled:
        return {"available": False, "status": "disabled", "capabilities": []}
    if cfg.rel_api_url:
        return await _call_service_url("GET", cfg.rel_api_url, "capabilities")
    if not cfg.graphrag_url:
        return {"available": False, "status": "not_configured", "message": "Set REL_API_URL or GRAPHRAG_URL", "capabilities": []}

    status = await _call_graphrag("GET", "rel/status")
    return {
        "capabilities": status.get("capabilities") if isinstance(status, dict) else None,
        "model_info": status.get("model_info") if isinstance(status, dict) else None,
        "source": "graphrag_status",
    }


@router.post("/rel/extract")
async def graphrag_rel_extract(request: RelationExtractionRequest):
    """Extract relationships from text using GraphRAG's relationship extraction API."""
    cfg = get_graphrag_config()
    body = request.model_dump(include={"text", "relations", "entity_labels"})

    if cfg.rel_api_url:
        payload = await _call_service_url(
            "POST",
            cfg.rel_api_url,
            "extract-relations",
            json=request.model_dump(),
        )
        return _with_relation_count(payload)

    payload = await _call_graphrag(
        "POST",
        "rel/extract",
        params={"threshold": request.threshold},
        json=body,
    )
    return _with_relation_count(payload)


@router.post("/rel/extract-batch")
async def graphrag_rel_extract_batch(request: RelationBatchExtractionRequest):
    """Extract relationships from multiple texts using the REL service when available."""
    cfg = get_graphrag_config()
    body = request.model_dump()

    if cfg.rel_api_url:
        payload = await _call_service_url(
            "POST",
            cfg.rel_api_url,
            "extract-relations/batch",
            json=body,
            timeout=240.0,
        )
        return _with_batch_totals(payload, "total_texts", "relation_count", "relations")

    results = []
    for text in request.texts:
        item = await _call_graphrag(
            "POST",
            "rel/extract",
            params={"threshold": request.threshold},
            json={
                "text": text,
                "relations": request.relations,
                "entity_labels": request.entity_labels,
            },
        )
        if isinstance(item, dict):
            results.append(_with_relation_count(item))
        else:
            results.append({"text": text, "relations": [], "relation_count": 0, "response": item})

    return {
        "results": results,
        "total_texts": len(request.texts),
        "batch_mode": "llama_nexus_fallback",
    }


@router.post("/rel/extract-entities")
async def graphrag_rel_extract_entities(request: RelationEntityExtractionRequest):
    """Extract entities through the relationship extraction GLiNER service."""
    cfg = get_graphrag_config()
    if cfg.rel_api_url:
        payload = await _call_service_url(
            "POST",
            cfg.rel_api_url,
            "extract-entities",
            json=request.model_dump(),
        )
        return _with_entity_count(payload)

    payload = await _call_graphrag("POST", "ner/extract", json=request.text)
    return _with_entity_count(payload)


@router.post("/extract/enhanced")
async def graphrag_enhanced_extraction(request: EnhancedExtractionRequest):
    """Run GraphRAG's enhanced multi-method entity and relationship extraction."""
    return await _call_graphrag(
        "POST",
        "extract-entities-relations-enhanced",
        data={
            "text": request.text,
            "domain": request.domain,
            "use_spanbert": str(request.use_spanbert).lower(),
            "use_dependency": str(request.use_dependency).lower(),
            "use_entity_linking": str(request.use_entity_linking).lower(),
        },
        timeout=120.0,
    )


@router.post("/query/enhanced")
async def graphrag_enhanced_query(request: EnhancedQueryRequest):
    """Process a query with graph-based reasoning and entity linking."""
    return await _call_graphrag(
        "POST",
        "query/enhanced",
        json=request.model_dump(),
    )


@router.post("/query/enhanced-answer")
async def graphrag_enhanced_answer(request: EnhancedAnswerRequest):
    """Run GraphRAG's enhanced answer generation with intent and reasoning metadata."""
    return await _call_graphrag(
        "POST",
        "api/enhanced-query",
        params={"query": request.query},
        json=request.context,
    )


@router.get("/query/statistics")
async def graphrag_query_statistics():
    """Get GraphRAG enhanced-query processor statistics."""
    return await _call_graphrag("GET", "query/statistics")


@router.get("/reasoning/graph-statistics")
async def graphrag_reasoning_graph_statistics():
    """Get graph-reasoner statistics, centrality, and clusters."""
    return await _call_graphrag("GET", "reasoning/graph-statistics")
