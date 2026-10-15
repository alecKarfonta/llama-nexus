"""Qdrant vector database management routes.

[2026-10-15] | INFO | ROUTES.QDRANT | Vector-store lifecycle under the same
service-control API as LLM/STT/TTS/embed so the UI and installers can bring up
the full RAG stack without manual compose commands.
"""
from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["qdrant"])


def get_qdrant_manager(request: Request):
    """Get the Qdrant manager from app state."""
    return getattr(request.app.state, 'qdrant_manager', None)


@router.post("/api/v1/qdrant/start")
async def start_qdrant(request: Request):
    """Start the Qdrant vector database container."""
    qdrant_manager = get_qdrant_manager(request)
    if qdrant_manager is None:
        raise HTTPException(status_code=503, detail="Qdrant manager not available")
    try:
        logger.info("Qdrant start request")
        success = await qdrant_manager.start()
        status = await qdrant_manager.get_status()
        return {
            "success": success,
            "status": status,
            "message": "Qdrant started successfully" if success else "Qdrant failed to start"
        }
    except Exception as e:
        logger.error(f"Qdrant start error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start Qdrant: {str(e)}")


@router.post("/api/v1/qdrant/stop")
async def stop_qdrant(request: Request):
    """Stop the Qdrant vector database container."""
    qdrant_manager = get_qdrant_manager(request)
    if qdrant_manager is None:
        raise HTTPException(status_code=503, detail="Qdrant manager not available")
    try:
        success = await qdrant_manager.stop()
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop Qdrant: {str(e)}")


@router.post("/api/v1/qdrant/restart")
async def restart_qdrant(request: Request):
    """Restart the Qdrant vector database container."""
    qdrant_manager = get_qdrant_manager(request)
    if qdrant_manager is None:
        raise HTTPException(status_code=503, detail="Qdrant manager not available")
    try:
        success = await qdrant_manager.restart()
        status = await qdrant_manager.get_status()
        return {"success": success, "status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart Qdrant: {str(e)}")


@router.get("/api/v1/qdrant/status")
async def get_qdrant_status(request: Request):
    """Get current Qdrant status (container + health probe)."""
    qdrant_manager = get_qdrant_manager(request)
    if qdrant_manager is None:
        raise HTTPException(status_code=503, detail="Qdrant manager not available")
    try:
        return await qdrant_manager.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Qdrant status: {str(e)}")


@router.get("/api/v1/qdrant/config")
async def get_qdrant_config(request: Request):
    """Get current Qdrant configuration."""
    qdrant_manager = get_qdrant_manager(request)
    if qdrant_manager is None:
        raise HTTPException(status_code=503, detail="Qdrant manager not available")
    return qdrant_manager.config
