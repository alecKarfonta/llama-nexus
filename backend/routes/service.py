"""Service management routes for LlamaCPP and Embedding services."""
from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["service"])


def get_manager(request: Request):
    """Get the LlamaCPP manager from app state."""
    return getattr(request.app.state, 'manager', None)


def get_embedding_manager(request: Request):
    """Get the embedding manager from app state."""
    return getattr(request.app.state, 'embedding_manager', None)


def get_merge_config_func(request: Request):
    """Get the config merge function from app state."""
    return getattr(request.app.state, 'merge_and_persist_config', None)


# =============================================================================
# LlamaCPP Service Endpoints
# =============================================================================

@router.post("/api/v1/service/start")
async def start_service(request: Request):
    """Start the llamacpp service."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    try:
        logger.info("Service start request")
        success = await manager.start()
        status = manager.get_status()
        return {
            "success": success, 
            "status": status,
            "message": "Service started successfully" if success else "Service failed to start"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Service start error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/api/v1/service/stop")
async def stop_service(request: Request):
    """Stop the llamacpp service."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    success = await manager.stop()
    return {"success": success, "status": manager.get_status()}


@router.post("/api/v1/service/restart")
async def restart_service(request: Request):
    """Restart the llamacpp service."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    success = await manager.restart()
    return {"success": success, "status": manager.get_status()}


@router.get("/api/v1/service/status")
async def get_service_status(request: Request):
    """Get current service status including resources."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    status = manager.get_status()
    
    if status.get("running"):
        health = await manager.get_llamacpp_health()
        status["llamacpp_health"] = health
    
    return status


@router.get("/api/v1/service/config")
async def get_config(request: Request):
    """Get current configuration."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    return {
        "config": manager.config,
        "command": " ".join(manager.build_command()),
        "editable_fields": {
            "model": ["context_size", "gpu_layers"],
            "sampling": ["temperature", "top_p", "top_k", "min_p", "repeat_penalty", 
                        "frequency_penalty", "presence_penalty", "dry_multiplier"],
            "performance": ["threads", "batch_size", "ubatch_size", "num_predict"],
            "server": ["api_key"],
            "template": ["directory", "selected"]
        }
    }


@router.put("/api/v1/service/config")
async def update_config(request: Request, config: Dict[str, Any]):
    """Update configuration (requires restart to apply)."""
    manager = get_manager(request)
    merge_config = get_merge_config_func(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    if merge_config is None:
        raise HTTPException(status_code=503, detail="Config persistence not available")
    try:
        updated_config = merge_config(config)
        return {
            "success": True,
            "config": updated_config,
            "command": " ".join(manager.build_command()),
            "message": "Configuration updated. Restart service to apply changes."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


@router.post("/api/v1/service/config/preview")
async def preview_config(request: Request, payload: Dict[str, Any]):
    """Preview the command line that would be generated for a configuration without applying it."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    
    config = payload.get("config")
    if not config:
        raise HTTPException(status_code=400, detail="Config is required")
    
    try:
        # Create a temporary copy of the manager with the new config
        # We'll temporarily update the manager's config to generate the command
        original_config = manager.config.copy()
        
        # Merge the new config with the existing one
        temp_config = {**original_config, **config}
        manager.config = temp_config
        
        # Generate the command
        command = " ".join(manager.build_command())
        
        # Restore the original config
        manager.config = original_config
        
        return {
            "command": command,
            "config": temp_config
        }
    except Exception as e:
        # Make sure to restore original config even if there's an error
        manager.config = original_config
        raise HTTPException(status_code=500, detail=f"Failed to preview config: {str(e)}")


@router.post("/api/v1/service/config/validate")
async def validate_config(request: Request, config: Dict[str, Any]):
    """Validate a configuration without applying it."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    
    errors = []
    warnings = []
    
    # Validate model settings
    if "model" in config:
        model = config["model"]
        if "context_size" in model:
            ctx = model["context_size"]
            if ctx < 512:
                errors.append("context_size must be at least 512")
            elif ctx > 131072:
                warnings.append("context_size > 131072 may cause memory issues")
        if "gpu_layers" in model:
            layers = model["gpu_layers"]
            if layers < -1:
                errors.append("gpu_layers must be >= -1")
    
    # Validate sampling settings
    if "sampling" in config:
        sampling = config["sampling"]
        if "temperature" in sampling:
            temp = sampling["temperature"]
            if temp < 0:
                errors.append("temperature must be >= 0")
            elif temp > 2:
                warnings.append("temperature > 2 may produce incoherent output")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


@router.post("/api/v1/service/action")
async def service_action(request: Request, payload: Dict[str, Any]):
    """Perform a service action (start, stop, restart)."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    
    action = payload.get("action")
    config = payload.get("config")
    
    # Apply config before start/restart if provided
    if config and action in ("start", "restart"):
        merge_config = get_merge_config_func(request)
        if merge_config:
            logger.info(f"Applying config before {action}: model={config.get('model', {}).get('name')}/{config.get('model', {}).get('variant')}")
            merge_config(config)
        else:
            # Fallback: directly update manager config
            logger.info(f"Directly updating manager config before {action}")
            manager.config = {**manager.config, **config}
    
    if action == "start":
        success = await manager.start()
    elif action == "stop":
        success = await manager.stop()
    elif action == "restart":
        success = await manager.restart()
    else:
        raise HTTPException(status_code=400, detail="Unsupported action. Use 'start', 'stop', or 'restart'.")
    return {"success": success, "status": manager.get_status()}


# Alias endpoints for frontend compatibility
@router.get("/v1/service/config")
async def get_config_alias(request: Request):
    """Get config (frontend compatibility)."""
    return await get_config(request)


@router.put("/v1/service/config")
async def update_config_alias(request: Request, config: Dict[str, Any]):
    """Update config (frontend compatibility)."""
    return await update_config(request, config)


@router.post("/v1/service/config/preview")
async def preview_config_alias(request: Request, payload: Dict[str, Any]):
    """Preview config (frontend compatibility)."""
    return await preview_config(request, payload)


@router.post("/v1/service/config/validate")
async def validate_config_alias(request: Request, config: Dict[str, Any]):
    """Validate config (frontend compatibility)."""
    return await validate_config(request, config)


@router.get("/v1/service/status")
async def get_status_alias(request: Request):
    """Get status (frontend compatibility)."""
    return await get_service_status(request)


@router.post("/v1/service/action")
async def service_action_alias(request: Request, payload: Dict[str, Any]):
    """Service action (frontend compatibility)."""
    return await service_action(request, payload)


# =============================================================================
# Embedding Service Endpoints
# =============================================================================

@router.post("/api/v1/embedding/start")
async def start_embedding_service(request: Request):
    """Start the embedding service."""
    embedding_manager = get_embedding_manager(request)
    if embedding_manager is None:
        raise HTTPException(status_code=503, detail="Embedding manager not available")
    try:
        logger.info("Embedding service start request")
        success = await embedding_manager.start()
        status = embedding_manager.get_status()
        return {
            "success": success,
            "status": status,
            "message": "Embedding service started successfully" if success else "Embedding service failed to start"
        }
    except Exception as e:
        logger.error(f"Embedding service start error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start embedding service: {str(e)}")


@router.post("/api/v1/embedding/stop")
async def stop_embedding_service(request: Request):
    """Stop the embedding service."""
    embedding_manager = get_embedding_manager(request)
    if embedding_manager is None:
        raise HTTPException(status_code=503, detail="Embedding manager not available")
    try:
        success = await embedding_manager.stop()
        return {"success": success, "status": embedding_manager.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop embedding service: {str(e)}")


@router.post("/api/v1/embedding/restart")
async def restart_embedding_service(request: Request):
    """Restart the embedding service."""
    embedding_manager = get_embedding_manager(request)
    if embedding_manager is None:
        raise HTTPException(status_code=503, detail="Embedding manager not available")
    try:
        success = await embedding_manager.restart()
        return {"success": success, "status": embedding_manager.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart embedding service: {str(e)}")


@router.get("/api/v1/embedding/status")
async def get_embedding_status(request: Request):
    """Get current embedding service status."""
    embedding_manager = get_embedding_manager(request)
    if embedding_manager is None:
        raise HTTPException(status_code=503, detail="Embedding manager not available")
    return embedding_manager.get_status()


@router.get("/api/v1/embedding/config")
async def get_embedding_config(request: Request):
    """Get current embedding configuration."""
    embedding_manager = get_embedding_manager(request)
    if embedding_manager is None:
        raise HTTPException(status_code=503, detail="Embedding manager not available")
    return {
        "config": embedding_manager.config,
        "command": " ".join(embedding_manager.build_command()),
        "available_models": [
            {
                "name": "nomic-embed-text-v1.5",
                "dimensions": 768,
                "max_tokens": 8192,
                "description": "Nomic AI's long context embedding model (recommended)"
            },
            {
                "name": "e5-mistral-7b",
                "dimensions": 4096,
                "max_tokens": 32768,
                "description": "E5 Mistral 7B instruct model for embeddings"
            },
            {
                "name": "bge-m3",
                "dimensions": 1024,
                "max_tokens": 8192,
                "description": "BAAI BGE-M3 multilingual embedding model"
            },
            {
                "name": "gte-Qwen2-1.5B",
                "dimensions": 1536,
                "max_tokens": 32768,
                "description": "Alibaba GTE Qwen2 1.5B instruct model"
            }
        ]
    }


@router.put("/api/v1/embedding/config")
async def update_embedding_config(request: Request, config: Dict[str, Any]):
    """Update embedding configuration (requires restart to apply)."""
    embedding_manager = get_embedding_manager(request)
    if embedding_manager is None:
        raise HTTPException(status_code=503, detail="Embedding manager not available")
    try:
        updated_config = embedding_manager.update_config(config)
        return {
            "success": True,
            "config": updated_config,
            "message": "Configuration updated. Restart service to apply changes."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


@router.post("/api/v1/embedding/test")
async def test_embedding(request: Request):
    """Test the embedding service by generating embeddings for sample text."""
    import aiohttp
    
    embedding_manager = get_embedding_manager(request)
    if embedding_manager is None:
        raise HTTPException(status_code=503, detail="Embedding manager not available")
    
    status = embedding_manager.get_status()
    if not status.get("running"):
        raise HTTPException(status_code=503, detail="Embedding service is not running")
    
    try:
        data = await request.json()
        text = data.get("text", "The quick brown fox jumps over the lazy dog.")
        
        # Get the service configuration
        config = embedding_manager.config
        api_key = config.get("server", {}).get("api_key", "llamacpp-embed")
        model_name = config.get("model", {}).get("name", "nomic-embed-text-v1.5")
        port = config.get("server", {}).get("port", 8602)
        
        # Connect to the embedding service via Docker network
        # Use the container hostname for inter-container communication
        service_url = f"http://llamacpp-embed:{port}/v1/embeddings"
        
        import time
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                service_url,
                json={"input": text, "model": model_name},
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"Embedding service error: {error_text}")
                
                result = await response.json()
        
        end_time = time.time()
        time_taken = int((end_time - start_time) * 1000)
        
        return {
            "success": True,
            "model": result.get("model"),
            "usage": result.get("usage"),
            "vectorLength": len(result.get("data", [{}])[0].get("embedding", [])),
            "timeTaken": time_taken,
            "data": result.get("data")
        }
        
    except aiohttp.ClientError as e:
        raise HTTPException(status_code=503, detail=f"Failed to connect to embedding service: {str(e)}")
    except Exception as e:
        logger.error(f"Embedding test error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding test failed: {str(e)}")
