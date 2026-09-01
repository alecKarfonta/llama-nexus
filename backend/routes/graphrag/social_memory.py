"""
GraphRAG Social Memory Routes

Typed adapters for GraphRAG's social-memory service: conversational ingest,
personalized context retrieval, retrieval explanation, identity correction,
replay/eval administration, and community reports.
"""

from datetime import datetime
from typing import Any, Dict, Literal
import logging

import httpx
from fastapi import APIRouter, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field

from .helpers import check_graphrag_enabled, get_http_client, graphrag_url as _graphrag_url
from modules.graphrag.config import get_graphrag_config

logger = logging.getLogger(__name__)

router = APIRouter()

SocialMemoryPlatform = Literal["twitch", "youtube", "kick", "rumble"]
SocialMemoryEvalSuite = Literal["all", "locomo", "vtuber"]
SocialMemoryFactPredicate = Literal[
    "OWNS",
    "LIKES",
    "DISLIKES",
    "WORKS_AT",
    "LIVES_IN",
    "PLAYS",
    "WATCHES",
    "KNOWS",
    "IS_A",
    "RELATES_TO",
]


class SocialMemoryIngestMessage(BaseModel):
    platform: SocialMemoryPlatform
    platform_user_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    text: str = Field(min_length=1)
    timestamp: datetime
    message_id: str = Field(min_length=1)
    stream_id: str | None = None
    session_id: str | None = None
    previous_messages: list[str] = Field(default_factory=list)
    priority: int = Field(default=0, ge=0, le=10)


class SocialMemoryMergeIdentitiesRequest(BaseModel):
    primary_person_id: str = Field(min_length=1)
    secondary_person_id: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str | None = None


class SocialMemoryCorrectFactRequest(BaseModel):
    fact_id: str = Field(min_length=1)
    new_fact_text: str = Field(min_length=1)
    corrected_by: str = Field(min_length=1)
    predicate: SocialMemoryFactPredicate | None = None
    entity_name: str | None = None
    entity_type: str | None = None


class SocialMemoryReplayRequest(BaseModel):
    transcript_path: str = Field(min_length=1)
    sandbox_tag: str | None = None
    sync: bool = True


def _json_or_text(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def _compact_params(params: Dict[str, Any]) -> Dict[str, Any]:
    compacted: Dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, datetime):
            compacted[key] = value.isoformat()
        else:
            compacted[key] = value
    return compacted


def _admin_headers(x_api_key: str | None) -> Dict[str, str] | None:
    if not x_api_key:
        return None
    return {"X-API-Key": x_api_key}


async def _request_graphrag(
    method: str,
    path: str,
    *,
    params: Dict[str, Any] | None = None,
    json: Any = None,
    headers: Dict[str, str] | None = None,
    timeout: float = 120.0,
) -> httpx.Response:
    check_graphrag_enabled()

    try:
        client = get_http_client()
        return await client.request(
            method,
            f"{_graphrag_url()}/{path}",
            params=_compact_params(params or {}),
            json=json,
            headers=headers,
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG social-memory request failed: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


async def _call_graphrag(
    method: str,
    path: str,
    *,
    params: Dict[str, Any] | None = None,
    json: Any = None,
    headers: Dict[str, str] | None = None,
    timeout: float = 120.0,
) -> Any:
    response = await _request_graphrag(
        method,
        path,
        params=params,
        json=json,
        headers=headers,
        timeout=timeout,
    )
    payload = _json_or_text(response)
    if response.status_code >= 400:
        status_code = response.status_code if response.status_code < 500 else 502
        raise HTTPException(status_code=status_code, detail=payload)
    return payload


@router.get("/social-memory/metrics")
async def graphrag_social_memory_metrics():
    """Return Prometheus metrics from GraphRAG social memory."""
    response = await _request_graphrag("GET", "social-memory/metrics", timeout=60.0)
    if response.status_code >= 400:
        payload = _json_or_text(response)
        status_code = response.status_code if response.status_code < 500 else 502
        raise HTTPException(status_code=status_code, detail=payload)

    content_type = response.headers.get("content-type", "text/plain; version=0.0.4")
    return Response(content=response.content, media_type=content_type)


@router.get("/social-memory/health")
async def graphrag_social_memory_health():
    """Check GraphRAG social-memory backing stores."""
    cfg = get_graphrag_config()
    if not cfg.enabled:
        return {"available": False, "status": "disabled", "message": "GraphRAG service is disabled. Set GRAPHRAG_ENABLED=true"}
    if not cfg.graphrag_url:
        return {"available": False, "status": "not_configured", "message": "Set GRAPHRAG_URL to enable social memory"}

    return await _call_graphrag("GET", "social-memory/health", timeout=60.0)


@router.post("/social-memory/admin/migrate")
async def graphrag_social_memory_migrate(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Run upstream social-memory graph migrations."""
    return await _call_graphrag(
        "POST",
        "social-memory/admin/migrate",
        headers=_admin_headers(x_api_key),
        timeout=300.0,
    )


@router.post("/social-memory/ingest")
async def graphrag_social_memory_ingest(
    message: SocialMemoryIngestMessage,
    sync: bool = Query(default=False),
):
    """Ingest a conversational message into GraphRAG social memory."""
    return await _call_graphrag(
        "POST",
        "social-memory/ingest",
        params={"sync": sync},
        json=message.model_dump(mode="json"),
        timeout=300.0 if sync else 120.0,
    )


@router.get("/social-memory/context")
async def graphrag_social_memory_context(
    platform: SocialMemoryPlatform = Query(...),
    platform_user_id: str = Query(...),
    query: str | None = Query(default=None),
    max_facts: int = Query(default=25, ge=1, le=100),
    include_recent_episodes: int = Query(default=3, ge=0, le=15),
    as_of: datetime | None = Query(default=None),
):
    """Retrieve personalized social-memory context for a user."""
    return await _call_graphrag(
        "GET",
        "social-memory/context",
        params={
            "platform": platform,
            "platform_user_id": platform_user_id,
            "query": query,
            "max_facts": max_facts,
            "include_recent_episodes": include_recent_episodes,
            "as_of": as_of,
        },
    )


@router.get("/social-memory/person/{person_id}/facts")
async def graphrag_social_memory_person_facts(
    person_id: str,
    as_of: datetime | None = Query(default=None),
):
    """List social-memory facts known for a resolved person."""
    return await _call_graphrag(
        "GET",
        f"social-memory/person/{person_id}/facts",
        params={"as_of": as_of},
    )


@router.get("/social-memory/explain")
async def graphrag_social_memory_explain(
    platform: SocialMemoryPlatform = Query(...),
    platform_user_id: str = Query(...),
    query: str = Query(...),
    max_facts: int = Query(default=25, ge=1, le=100),
):
    """Explain which social-memory facts were retrieved for a query."""
    return await _call_graphrag(
        "GET",
        "social-memory/explain",
        params={
            "platform": platform,
            "platform_user_id": platform_user_id,
            "query": query,
            "max_facts": max_facts,
        },
    )


@router.post("/social-memory/identities/merge")
async def graphrag_social_memory_merge_identities(
    request: SocialMemoryMergeIdentitiesRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Merge duplicate social-memory identities."""
    return await _call_graphrag(
        "POST",
        "social-memory/identities/merge",
        json=request.model_dump(mode="json"),
        headers=_admin_headers(x_api_key),
    )


@router.post("/social-memory/facts/correct")
async def graphrag_social_memory_correct_fact(
    request: SocialMemoryCorrectFactRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Correct a stored social-memory fact while preserving provenance."""
    return await _call_graphrag(
        "POST",
        "social-memory/facts/correct",
        json=request.model_dump(mode="json", exclude_none=True),
        headers=_admin_headers(x_api_key),
    )


@router.post("/social-memory/admin/replay")
async def graphrag_social_memory_replay(
    request: SocialMemoryReplayRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Replay a transcript into a social-memory sandbox."""
    return await _call_graphrag(
        "POST",
        "social-memory/admin/replay",
        json=request.model_dump(mode="json", exclude_none=True),
        headers=_admin_headers(x_api_key),
        timeout=600.0,
    )


@router.delete("/social-memory/admin/replay/{sandbox_tag}")
async def graphrag_social_memory_purge_replay(
    sandbox_tag: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Purge a social-memory replay sandbox."""
    return await _call_graphrag(
        "DELETE",
        f"social-memory/admin/replay/{sandbox_tag}",
        headers=_admin_headers(x_api_key),
        timeout=300.0,
    )


@router.post("/social-memory/admin/eval")
async def graphrag_social_memory_eval(
    suite: SocialMemoryEvalSuite = Query(default="all"),
    baseline_path: str | None = Query(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Run upstream LoCoMo/vTuber social-memory evaluation suites."""
    return await _call_graphrag(
        "POST",
        "social-memory/admin/eval",
        params={"suite": suite, "baseline_path": baseline_path},
        headers=_admin_headers(x_api_key),
        timeout=600.0,
    )


@router.get("/social-memory/admin/community-report")
async def graphrag_social_memory_community_report(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Generate a social-memory community report."""
    return await _call_graphrag(
        "GET",
        "social-memory/admin/community-report",
        headers=_admin_headers(x_api_key),
        timeout=300.0,
    )
