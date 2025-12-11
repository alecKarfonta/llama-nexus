"""Model registry routes."""
from fastapi import APIRouter, HTTPException, Request
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/registry", tags=["registry"])


def get_model_registry(request: Request):
    """Get the model registry from app state."""
    registry = getattr(request.app.state, 'model_registry', None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")
    return registry


@router.get("/stats")
async def get_registry_stats(request: Request):
    """Get model registry statistics."""
    registry = get_model_registry(request)
    return registry.get_registry_stats()


@router.get("/models")
async def list_cached_models(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    search: str = None,
    model_type: str = None,
):
    """List all cached models with optional filtering."""
    registry = get_model_registry(request)
    return registry.list_cached_models(
        limit=limit,
        offset=offset,
        search=search,
        model_type=model_type,
    )


@router.post("/models")
async def cache_model(request: Request):
    """Cache model metadata from HuggingFace."""
    registry = get_model_registry(request)
    data = await request.json()
    model_id = registry.cache_model(
        repo_id=data.get('repo_id'),
        name=data.get('name'),
        description=data.get('description'),
        author=data.get('author'),
        downloads=data.get('downloads', 0),
        likes=data.get('likes', 0),
        tags=data.get('tags', []),
        model_type=data.get('model_type'),
        license=data.get('license'),
        last_modified=data.get('last_modified'),
        metadata=data.get('metadata', {}),
    )
    return {"status": "cached", "model_id": model_id}


@router.get("/models/{repo_id:path}")
async def get_cached_model(request: Request, repo_id: str):
    """Get cached model metadata."""
    registry = get_model_registry(request)
    model = registry.get_cached_model(repo_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found in cache")
    return model


@router.delete("/models/{repo_id:path}")
async def delete_cached_model(request: Request, repo_id: str):
    """Delete a model from the cache."""
    registry = get_model_registry(request)
    deleted = registry.delete_model_cache(repo_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Model not found in cache")
    return {"status": "deleted", "repo_id": repo_id}


@router.post("/models/{repo_id:path}/variants")
async def add_model_variant(request: Request, repo_id: str):
    """Add a quantization variant for a model."""
    registry = get_model_registry(request)
    data = await request.json()
    registry.add_variant(
        repo_id=repo_id,
        filename=data.get('filename'),
        quantization=data.get('quantization'),
        size_bytes=data.get('size_bytes'),
        vram_required_mb=data.get('vram_required_mb'),
        quality_score=data.get('quality_score'),
        speed_score=data.get('speed_score'),
    )
    return {"status": "added"}


@router.get("/models/{repo_id:path}/variants")
async def get_model_variants(request: Request, repo_id: str):
    """Get all quantization variants for a model."""
    registry = get_model_registry(request)
    return {"variants": registry.get_variants(repo_id)}


@router.post("/models/{repo_id:path}/usage/load")
async def record_model_load(request: Request, repo_id: str, variant: str = None):
    """Record that a model was loaded."""
    registry = get_model_registry(request)
    registry.record_model_load(repo_id, variant)
    return {"status": "recorded"}


@router.post("/models/{repo_id:path}/usage/inference")
async def record_inference(request: Request, repo_id: str):
    """Record inference statistics."""
    registry = get_model_registry(request)
    data = await request.json()
    registry.record_inference(
        repo_id=repo_id,
        variant=data.get('variant'),
        tokens_generated=data.get('tokens_generated', 0),
        inference_time_ms=data.get('inference_time_ms', 0),
    )
    return {"status": "recorded"}


@router.get("/usage")
async def get_usage_stats(request: Request, repo_id: str = None):
    """Get usage statistics for models."""
    registry = get_model_registry(request)
    return {"usage": registry.get_usage_stats(repo_id)}


@router.get("/most-used")
async def get_most_used_models(request: Request, limit: int = 10):
    """Get the most frequently used models."""
    registry = get_model_registry(request)
    return {"models": registry.get_most_used_models(limit)}


@router.post("/models/{repo_id:path}/rating")
async def set_model_rating(request: Request, repo_id: str):
    """Set user rating and notes for a model."""
    registry = get_model_registry(request)
    data = await request.json()
    try:
        registry.set_rating(
            repo_id=repo_id,
            rating=data.get('rating'),
            variant=data.get('variant'),
            notes=data.get('notes'),
            tags=data.get('tags'),
        )
        return {"status": "saved"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/models/{repo_id:path}/rating")
async def get_model_rating(request: Request, repo_id: str, variant: str = None):
    """Get user rating for a model."""
    registry = get_model_registry(request)
    rating = registry.get_rating(repo_id, variant)
    if rating is None:
        return {"rating": None}
    return rating


@router.post("/models/{repo_id:path}/hardware")
async def set_hardware_recommendation(request: Request, repo_id: str):
    """Set hardware recommendations for a model variant."""
    registry = get_model_registry(request)
    data = await request.json()
    registry.set_hardware_recommendation(
        repo_id=repo_id,
        variant=data.get('variant'),
        min_vram_gb=data.get('min_vram_gb'),
        recommended_vram_gb=data.get('recommended_vram_gb'),
        min_ram_gb=data.get('min_ram_gb'),
        recommended_context_size=data.get('recommended_context_size'),
        gpu_layers_recommendation=data.get('gpu_layers_recommendation'),
        notes=data.get('notes'),
    )
    return {"status": "saved"}


@router.get("/recommendations")
async def get_recommendations_for_hardware(
    request: Request,
    vram_gb: float,
    ram_gb: float = None,
):
    """Get model recommendations based on available hardware."""
    registry = get_model_registry(request)
    return {
        "recommendations": registry.get_recommendations_for_hardware(
            available_vram_gb=vram_gb,
            available_ram_gb=ram_gb,
        )
    }
