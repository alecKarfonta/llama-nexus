"""STT (Speech-to-Text) service management routes."""
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form, Query
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
            },
            {
                "name": "distil-large-v3.5-ct2",
                "size": "distil-large-v3.5-ct2",
                "params": "580M",
                "vram": "~8GB",
                "description": "Distilled large-v3.5 (CTranslate2) for faster inference"
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
    response_format: str = Form(default="json"),
    word_timestamps: str = Query(default="false")  # Try Query instead of Form
):
    """Transcribe audio file using the STT service.

    When word_timestamps=True, response_format is automatically set to 'verbose_json'
    and word-level timestamps will be included in the response.
    """
    # Read file content first (needed for both paths)
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Map model names to HuggingFace repositories for models not in faster-whisper's default list
    model_mappings = {
        "distil-large-v3.5-ct2": "distil-whisper/distil-large-v3.5-ct2",
        # Add more mappings here as needed for other custom models
    }
    mapped_model = model_mappings.get(model, model)

    # Parse word_timestamps
    word_timestamps_bool = word_timestamps.lower() in ('true', '1', 'yes', 'on')

    # For word timestamps, use curl to call the STT server directly
    if word_timestamps_bool:
        import subprocess
        import tempfile
        import os

        # Save the file temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Call curl directly with the correct parameters
            cmd = [
                'curl', '-s', '-X', 'POST',
                '-F', f'file=@{temp_path}',
                '-F', f'model={mapped_model}',
                '-F', 'response_format=verbose_json',
                '-F', 'timestamp_granularities[]=word',
                'http://whisper-stt:8000/v1/audio/transcriptions'
            ]

            print(f"DEBUG: Running curl command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(f"DEBUG: Curl exit code: {result.returncode}")

            if result.returncode == 0:
                import json
                response_data = json.loads(result.stdout)
                return response_data
            else:
                raise HTTPException(status_code=500, detail=f"Word timestamps failed: {result.stderr}")
        finally:
            os.unlink(temp_path)
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
            base_url = f"http://{container_name}:{internal_port}"
        else:
            base_url = f"http://localhost:{stt_manager.config['server']['port']}"

        # Always use OpenAI-compatible endpoint
        endpoint = f"{base_url}/v1/audio/transcriptions"
        
        # Map model names to HuggingFace repositories for models not in faster-whisper's default list
        model_mappings = {
            "distil-large-v3.5-ct2": "distil-whisper/distil-large-v3.5-ct2",
            # Add more mappings here as needed for other custom models
        }
        
        # Use HuggingFace repo path if model needs mapping, otherwise use the model name directly
        mapped_model = model_mappings.get(model, model)
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Build the files dict for multipart upload
            files = {"file": (file.filename or "audio.wav", content, file.content_type or "audio/wav")}

            # Build form data - OpenAI-compatible parameters
            data = {}
            if mapped_model:
                data["model"] = mapped_model
            if language and language != "auto":
                data["language"] = language

            # Handle word-level timestamps
            if word_timestamps_bool:
                logger.info("Word timestamps requested - using curl fallback")
                # For testing: use subprocess to call curl directly
                import subprocess
                import tempfile
                import os

                # Save the file temporarily
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(content)
                    temp_path = temp_file.name

                try:
                    # Call curl directly
                    cmd = [
                        'curl', '-s', '-X', 'POST',
                        '-F', f'file=@{temp_path}',
                        '-F', f'model={mapped_model}',
                        '-F', 'response_format=verbose_json',
                        '-F', 'timestamp_granularities[]=word',
                        'http://localhost:8603/v1/audio/transcriptions'
                    ]

                    logger.info(f"Running curl command: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    logger.info(f"Curl result: {result.returncode}, stdout length: {len(result.stdout)}, stderr: {result.stderr}")
                    if result.returncode == 0:
                        import json
                        response_data = json.loads(result.stdout)
                        logger.info(f"Curl response keys: {list(response_data.keys())}")
                        # Convert back to the expected format
                        return response_data
                    else:
                        logger.error(f"Curl failed: {result.stderr}")
                        raise HTTPException(status_code=500, detail=f"Curl failed: {result.stderr}")
                finally:
                    os.unlink(temp_path)
            else:
                logger.info("Standard transcription requested")
                if response_format:
                    data["response_format"] = response_format

            # Debug logging
            logger.info(f"Sending data parameters: {data}")
            logger.info(f"Files: {files}")

            response = await client.post(
                endpoint,
                files=files,
                data=data,
            )
            
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



