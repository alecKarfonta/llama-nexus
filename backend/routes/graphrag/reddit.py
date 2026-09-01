"""
GraphRAG Reddit Crawler Routes

Typed adapters for GraphRAG's standalone Reddit crawler service, including
health, crawl lifecycle, results, cancellation, and crawl listing.
"""

from typing import Any
import logging

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .helpers import check_graphrag_enabled, get_http_client
from modules.graphrag.config import get_graphrag_config

logger = logging.getLogger(__name__)

router = APIRouter()


class RedditCrawlRequest(BaseModel):
    query: str = Field(min_length=1)
    subreddits: list[str] | None = None
    sort_by: str = "relevance"
    time_filter: str = "month"
    max_posts: int = Field(default=50, ge=1)
    max_comments_per_post: int = Field(default=20, ge=0)
    comment_depth: int = Field(default=2, ge=1, le=3)
    include_replies: bool = True
    auto_ingest: bool = True
    filter_nsfw: bool = True


def _reddit_crawler_url() -> str:
    check_graphrag_enabled()
    cfg = get_graphrag_config()
    if not cfg.reddit_crawler_url:
        raise HTTPException(status_code=503, detail="REDDIT_CRAWLER_URL is not configured")
    return cfg.reddit_crawler_url.rstrip("/")


def _json_or_text(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


async def _response_payload(response: httpx.Response) -> Any:
    payload = _json_or_text(response)
    if response.status_code >= 400:
        status_code = response.status_code if response.status_code < 500 else 502
        raise HTTPException(status_code=status_code, detail=payload)
    return payload


@router.get("/reddit/health")
async def graphrag_reddit_health():
    """Check the configured GraphRAG Reddit crawler service."""
    cfg = get_graphrag_config()
    if not cfg.enabled:
        return {"available": False, "status": "disabled", "message": "GraphRAG service is disabled. Set GRAPHRAG_ENABLED=true"}
    if not cfg.reddit_crawler_url:
        return {"available": False, "status": "not_configured", "message": "Set REDDIT_CRAWLER_URL to enable Reddit crawling"}

    try:
        client = get_http_client()
        response = await client.get(f"{cfg.reddit_crawler_url.rstrip('/')}/health", timeout=30.0)
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG Reddit crawler health error: {e}")
        raise HTTPException(status_code=502, detail=f"Reddit crawler service error: {str(e)}")


@router.post("/reddit/crawl")
async def graphrag_reddit_start_crawl(request: RedditCrawlRequest):
    """Start a Reddit crawl with optional GraphRAG auto-ingestion."""
    try:
        client = get_http_client()
        response = await client.post(
            f"{_reddit_crawler_url()}/crawl",
            json=request.model_dump(mode="json", exclude_none=True),
            timeout=60.0,
        )
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG Reddit crawl start error: {e}")
        raise HTTPException(status_code=502, detail=f"Reddit crawler service error: {str(e)}")


@router.get("/reddit/crawl/{crawl_id}/status")
async def graphrag_reddit_crawl_status(crawl_id: str):
    """Get progress for a GraphRAG Reddit crawl."""
    try:
        client = get_http_client()
        response = await client.get(f"{_reddit_crawler_url()}/crawl/{crawl_id}/status", timeout=30.0)
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG Reddit crawl status error: {e}")
        raise HTTPException(status_code=502, detail=f"Reddit crawler service error: {str(e)}")


@router.get("/reddit/crawl/{crawl_id}/results")
async def graphrag_reddit_crawl_results(crawl_id: str):
    """Get results for a completed GraphRAG Reddit crawl."""
    try:
        client = get_http_client()
        response = await client.get(f"{_reddit_crawler_url()}/crawl/{crawl_id}/results", timeout=60.0)
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG Reddit crawl results error: {e}")
        raise HTTPException(status_code=502, detail=f"Reddit crawler service error: {str(e)}")


@router.delete("/reddit/crawl/{crawl_id}")
async def graphrag_reddit_cancel_crawl(crawl_id: str):
    """Cancel an active GraphRAG Reddit crawl."""
    try:
        client = get_http_client()
        response = await client.delete(f"{_reddit_crawler_url()}/crawl/{crawl_id}", timeout=30.0)
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG Reddit crawl cancel error: {e}")
        raise HTTPException(status_code=502, detail=f"Reddit crawler service error: {str(e)}")


@router.get("/reddit/crawls")
async def graphrag_reddit_list_crawls():
    """List active and completed GraphRAG Reddit crawls."""
    try:
        client = get_http_client()
        response = await client.get(f"{_reddit_crawler_url()}/crawls", timeout=30.0)
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG Reddit crawl list error: {e}")
        raise HTTPException(status_code=502, detail=f"Reddit crawler service error: {str(e)}")
