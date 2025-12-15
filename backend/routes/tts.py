"""TTS (Text-to-Speech) service management routes."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tts"])


def get_tts_manager(request: Request):
    """Get the TTS manager from app state."""
    return getattr(request.app.state, 'tts_manager', None)


# =============================================================================
# TTS Service Endpoints
# =============================================================================

@router.post("/api/v1/tts/start")
async def start_tts_service(request: Request, config: Optional[Dict[str, Any]] = None):
    """Start the TTS service."""
    tts_manager = get_tts_manager(request)
    if tts_manager is None:
        raise HTTPException(status_code=503, detail="TTS manager not available")
    try:
        # Update config if provided
        if config:
            tts_manager.update_config(config)
        
        logger.info("TTS service start request")
        success = await tts_manager.start()
        status = tts_manager.get_status()
        return {
            "success": success,
            "status": status,
            "message": "TTS service started successfully" if success else "TTS service failed to start"
        }
    except Exception as e:
        logger.error(f"TTS service start error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start TTS service: {str(e)}")


@router.post("/api/v1/tts/stop")
async def stop_tts_service(request: Request):
    """Stop the TTS service."""
    tts_manager = get_tts_manager(request)
    if tts_manager is None:
        raise HTTPException(status_code=503, detail="TTS manager not available")
    try:
        success = await tts_manager.stop()
        return {"success": success, "status": tts_manager.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop TTS service: {str(e)}")


@router.post("/api/v1/tts/restart")
async def restart_tts_service(request: Request, config: Optional[Dict[str, Any]] = None):
    """Restart the TTS service."""
    tts_manager = get_tts_manager(request)
    if tts_manager is None:
        raise HTTPException(status_code=503, detail="TTS manager not available")
    try:
        # Update config if provided
        if config:
            tts_manager.update_config(config)
        
        success = await tts_manager.restart()
        return {"success": success, "status": tts_manager.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart TTS service: {str(e)}")


@router.get("/api/v1/tts/status")
async def get_tts_status(request: Request):
    """Get current TTS service status."""
    tts_manager = get_tts_manager(request)
    if tts_manager is None:
        raise HTTPException(status_code=503, detail="TTS manager not available")
    return tts_manager.get_status()


@router.get("/api/v1/tts/config")
async def get_tts_config(request: Request):
    """Get current TTS configuration."""
    tts_manager = get_tts_manager(request)
    if tts_manager is None:
        raise HTTPException(status_code=503, detail="TTS manager not available")
    return {
        "config": tts_manager.config,
        "available_models": [
            {
                "name": "tts-1",
                "provider": "Piper",
                "description": "Standard quality, fast generation",
                "quality": "Standard",
                "speed": "Fast"
            },
            {
                "name": "tts-1-hd",
                "provider": "Piper",
                "description": "High definition audio quality",
                "quality": "HD",
                "speed": "Slower"
            }
        ],
        "available_voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        "audio_formats": ["mp3", "wav", "opus", "flac"]
    }


@router.put("/api/v1/tts/config")
async def update_tts_config(request: Request, config: Dict[str, Any]):
    """Update TTS configuration (requires restart to apply)."""
    tts_manager = get_tts_manager(request)
    if tts_manager is None:
        raise HTTPException(status_code=503, detail="TTS manager not available")
    try:
        updated_config = tts_manager.update_config(config)
        return {
            "success": True,
            "config": updated_config,
            "message": "Configuration updated. Restart service to apply changes."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


# Proxy endpoint for speech synthesis (forwards to the actual TTS service)
@router.post("/api/v1/tts/synthesize")
async def synthesize_speech(request: Request, payload: Dict[str, Any]):
    """Synthesize speech from text using the TTS service."""
    tts_manager = get_tts_manager(request)
    if tts_manager is None:
        raise HTTPException(status_code=503, detail="TTS manager not available")
    
    if not tts_manager.is_running():
        raise HTTPException(status_code=503, detail="TTS service is not running")
    
    try:
        import httpx
        import os
        
        # Forward to the actual TTS service
        # Use container name when in Docker, localhost otherwise
        container_name = tts_manager.container_name
        internal_port = 8000  # Internal container port
        if os.getenv("RUNNING_IN_DOCKER", "false").lower() == "true":
            endpoint = f"http://{container_name}:{internal_port}/v1/audio/speech"
        else:
            endpoint = f"http://localhost:{tts_manager.config['server']['port']}/v1/audio/speech"
        
        # Build request payload
        tts_payload = {
            "model": payload.get("model", tts_manager.config["model"]["name"]),
            "input": payload.get("input", payload.get("text", "")),
            "voice": payload.get("voice", tts_manager.config["model"]["voice"]),
            "speed": payload.get("speed", tts_manager.config["audio"]["speed"]),
            "response_format": payload.get("response_format", tts_manager.config["audio"]["format"])
        }
        
        if not tts_payload["input"]:
            raise HTTPException(status_code=400, detail="No text provided for synthesis")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                endpoint,
                json=tts_payload,
                headers={
                    "Authorization": f"Bearer {tts_manager.config['server']['api_key']}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            # Return audio as streaming response
            content_type = {
                "mp3": "audio/mpeg",
                "wav": "audio/wav",
                "opus": "audio/opus",
                "flac": "audio/flac"
            }.get(tts_payload["response_format"], "audio/mpeg")
            
            return StreamingResponse(
                iter([response.content]),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{tts_payload['response_format']}"
                }
            )
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Speech synthesis timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")



