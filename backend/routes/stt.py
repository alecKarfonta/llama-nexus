"""STT (Speech-to-Text) service management routes."""
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stt"])


def get_stt_manager(request: Request):
    """Get the STT manager from app state."""
    return getattr(request.app.state, 'stt_manager', None)


# =============================================================================
# STT Service Endpoints
# =============================================================================

@router.post("/api/v1/stt/start")
async def start_stt_service(request: Request, config: Optional[Dict[str, Any]] = None):
    """Start the STT service."""
    stt_manager = get_stt_manager(request)
    if stt_manager is None:
        raise HTTPException(status_code=503, detail="STT manager not available")
    try:
        # Update config if provided
        if config:
            stt_manager.update_config(config)
        
        logger.info("STT service start request")
        success = await stt_manager.start()
        status = stt_manager.get_status()
        return {
            "success": success,
            "status": status,
            "message": "STT service started successfully" if success else "STT service failed to start"
        }
    except Exception as e:
        logger.error(f"STT service start error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start STT service: {str(e)}")


@router.post("/api/v1/stt/stop")
async def stop_stt_service(request: Request):
    """Stop the STT service."""
    stt_manager = get_stt_manager(request)
    if stt_manager is None:
        raise HTTPException(status_code=503, detail="STT manager not available")
    try:
        success = await stt_manager.stop()
        return {"success": success, "status": stt_manager.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop STT service: {str(e)}")


@router.post("/api/v1/stt/restart")
async def restart_stt_service(request: Request, config: Optional[Dict[str, Any]] = None):
    """Restart the STT service."""
    stt_manager = get_stt_manager(request)
    if stt_manager is None:
        raise HTTPException(status_code=503, detail="STT manager not available")
    try:
        # Update config if provided
        if config:
            stt_manager.update_config(config)
        
        success = await stt_manager.restart()
        return {"success": success, "status": stt_manager.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart STT service: {str(e)}")


@router.get("/api/v1/stt/status")
async def get_stt_status(request: Request):
    """Get current STT service status."""
    stt_manager = get_stt_manager(request)
    if stt_manager is None:
        raise HTTPException(status_code=503, detail="STT manager not available")
    return stt_manager.get_status()


@router.get("/api/v1/stt/config")
async def get_stt_config(request: Request):
    """Get current STT configuration."""
    stt_manager = get_stt_manager(request)
    if stt_manager is None:
        raise HTTPException(status_code=503, detail="STT manager not available")
    return {
        "config": stt_manager.config,
        "available_models": [
            {
                "name": "tiny",
                "size": "tiny",
                "params": "39M",
                "vram": "~1GB",
                "description": "Fastest, lower accuracy"
            },
            {
                "name": "base",
                "size": "base",
                "params": "74M",
                "vram": "~1GB",
                "description": "Fast, good accuracy"
            },
            {
                "name": "small",
                "size": "small",
                "params": "244M",
                "vram": "~2GB",
                "description": "Balanced speed and accuracy"
            },
            {
                "name": "medium",
                "size": "medium",
                "params": "769M",
                "vram": "~5GB",
                "description": "High accuracy, slower"
            },
            {
                "name": "large-v3",
                "size": "large-v3",
                "params": "1.5B",
                "vram": "~10GB",
                "description": "Best accuracy, slowest"
            }
        ],
        "available_languages": [
            {"code": "auto", "name": "Auto-detect"},
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ru", "name": "Russian"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "zh", "name": "Chinese"},
        ]
    }


@router.put("/api/v1/stt/config")
async def update_stt_config(request: Request, config: Dict[str, Any]):
    """Update STT configuration (requires restart to apply)."""
    stt_manager = get_stt_manager(request)
    if stt_manager is None:
        raise HTTPException(status_code=503, detail="STT manager not available")
    try:
        updated_config = stt_manager.update_config(config)
        return {
            "success": True,
            "config": updated_config,
            "message": "Configuration updated. Restart service to apply changes."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


# Proxy endpoint for transcription (forwards to the actual STT service)
@router.post("/api/v1/stt/transcribe")
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(default="base"),
    language: str = Form(default="auto"),
    response_format: str = Form(default="json")
):
    """Transcribe audio file using the STT service."""
    stt_manager = get_stt_manager(request)
    if stt_manager is None:
        raise HTTPException(status_code=503, detail="STT manager not available")
    
    if not stt_manager.is_running():
        raise HTTPException(status_code=503, detail="STT service is not running")
    
    try:
        import httpx
        import os
        
        # Forward to the actual STT service
        # Use container name when in Docker, localhost otherwise
        container_name = stt_manager.container_name
        internal_port = 8000  # Internal container port
        if os.getenv("RUNNING_IN_DOCKER", "false").lower() == "true":
            endpoint = f"http://{container_name}:{internal_port}/v1/audio/transcriptions"
        else:
            endpoint = f"http://localhost:{stt_manager.config['server']['port']}/v1/audio/transcriptions"
        
        # Read file content
        content = await file.read()
        
        if not content:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Build the files dict for multipart upload
            files = {"file": (file.filename or "audio.wav", content, file.content_type or "audio/wav")}
            
            # Build form data - only include non-empty values
            data = {}
            if model:
                data["model"] = model
            if language and language != "auto":
                data["language"] = language
            if response_format:
                data["response_format"] = response_format
            
            logger.info(f"Sending transcription request to {endpoint} with file size {len(content)} bytes")
            
            response = await client.post(
                endpoint,
                files=files,
                data=data,
            )
            
            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"Transcription service returned {response.status_code}: {error_detail}")
                raise HTTPException(status_code=response.status_code, detail=f"Transcription service error: {error_detail}")
            
            return response.json()
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Transcription timed out")
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

