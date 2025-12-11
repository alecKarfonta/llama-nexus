"""Token usage tracking routes."""
from fastapi import APIRouter, HTTPException, Request
from typing import Optional
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/usage/tokens", tags=["tokens"])


class TimeRange(str, Enum):
    ONE_HOUR = "1h"
    ONE_DAY = "24h"
    ONE_WEEK = "7d"
    ONE_MONTH = "30d"
    ALL_TIME = "all"


def get_token_tracker(request: Request):
    """Get the token tracker from app state."""
    return getattr(request.app.state, 'token_tracker', None)


@router.get("")
async def get_token_usage(request: Request, timeRange: TimeRange = TimeRange.ONE_DAY):
    """Get token usage statistics for the specified time range."""
    token_tracker = get_token_tracker(request)
    if not token_tracker:
        raise HTTPException(status_code=503, detail="Token tracker not available")
    try:
        usage_data = token_tracker.get_token_usage(timeRange)
        return {"success": True, "data": usage_data, "timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting token usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeline")
async def get_token_timeline(
    request: Request,
    timeRange: TimeRange = TimeRange.ONE_DAY,
    granularity: str = "hour"
):
    """Get token usage timeline data for charts."""
    token_tracker = get_token_tracker(request)
    if not token_tracker:
        raise HTTPException(status_code=503, detail="Token tracker not available")
    try:
        timeline_data = token_tracker.get_token_timeline(timeRange, granularity)
        return {"success": True, "data": timeline_data, "timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting token timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_token_summary(request: Request):
    """Get a summary of token usage across all time."""
    token_tracker = get_token_tracker(request)
    if not token_tracker:
        raise HTTPException(status_code=503, detail="Token tracker not available")
    try:
        summary = token_tracker.get_summary()
        return {"success": True, "data": summary, "timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting token summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/record")
async def record_token_usage(request: Request):
    """Manually record token usage (for external integrations)."""
    token_tracker = get_token_tracker(request)
    if not token_tracker:
        raise HTTPException(status_code=503, detail="Token tracker not available")
    try:
        data = await request.json()
        token_tracker.record_usage(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            model=data.get("model"),
            endpoint=data.get("endpoint"),
        )
        return {"success": True, "message": "Usage recorded", "timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording token usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))
