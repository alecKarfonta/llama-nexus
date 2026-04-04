"""Deploy profile management routes."""
from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/profiles", tags=["profiles"])


def get_profile_store(request: Request):
    """Get the profile store from app state."""
    store = getattr(request.app.state, 'profile_store', None)
    if store is None:
        raise HTTPException(status_code=503, detail="Profile storage not available")
    return store


def get_manager(request: Request):
    """Get the LlamaCPP manager from app state."""
    return getattr(request.app.state, 'manager', None)


def get_merge_config_func(request: Request):
    """Get the config merge function from app state."""
    return getattr(request.app.state, 'merge_and_persist_config', None)


@router.get("")
async def list_profiles(request: Request):
    """List all saved deploy profiles."""
    store = get_profile_store(request)
    profiles = store.list_profiles()
    return {"profiles": profiles, "count": len(profiles)}


@router.get("/{profile_id}")
async def get_profile(request: Request, profile_id: str):
    """Get a specific profile."""
    store = get_profile_store(request)
    profile = store.get_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@router.post("")
async def save_profile(request: Request):
    """Save current config as a named profile, or save a custom config."""
    store = get_profile_store(request)
    manager = get_manager(request)
    
    data = await request.json()
    name = data.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Profile name is required")
    
    # Use provided config or snapshot the current manager config
    config = data.get("config", manager.config if manager else {})
    tags = data.get("tags", [])
    description = data.get("description", "")
    
    try:
        profile = store.save_profile(
            name=name,
            config=config,
            tags=tags,
            description=description,
        )
        return {"success": True, "profile": profile}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.put("/{profile_id}")
async def update_profile(request: Request, profile_id: str):
    """Update an existing profile."""
    store = get_profile_store(request)
    
    # Verify profile exists
    existing = store.get_profile(profile_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    data = await request.json()
    name = data.get("name", existing["name"])
    config = data.get("config", existing["config"])
    tags = data.get("tags", existing["tags"])
    description = data.get("description", existing["description"])
    
    try:
        profile = store.save_profile(
            name=name,
            config=config,
            tags=tags,
            description=description,
            profile_id=profile_id,
        )
        return {"success": True, "profile": profile}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/{profile_id}")
async def delete_profile(request: Request, profile_id: str):
    """Delete a profile."""
    store = get_profile_store(request)
    success = store.delete_profile(profile_id)
    if not success:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"success": True, "deleted": profile_id}


@router.post("/{profile_id}/load")
async def load_profile(request: Request, profile_id: str):
    """Load a profile's config into the active manager config (does not start service)."""
    store = get_profile_store(request)
    manager = get_manager(request)
    merge_config = get_merge_config_func(request)
    
    profile = store.get_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    if manager is None:
        raise HTTPException(status_code=503, detail="Service manager not available")
    
    # Apply the profile's config
    if merge_config:
        merge_config(profile["config"])
    else:
        manager.config = profile["config"]
    
    # Mark this profile as active
    store.set_active(profile_id)
    
    return {
        "success": True,
        "message": f"Profile '{profile['name']}' loaded. Restart service to apply.",
        "profile": profile,
    }


@router.post("/{profile_id}/deploy")
async def deploy_profile(request: Request, profile_id: str):
    """Load a profile and immediately start/restart the service."""
    store = get_profile_store(request)
    manager = get_manager(request)
    merge_config = get_merge_config_func(request)
    
    profile = store.get_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    if manager is None:
        raise HTTPException(status_code=503, detail="Service manager not available")
    
    # Apply profile config
    if merge_config:
        merge_config(profile["config"])
    else:
        manager.config = profile["config"]
    
    store.set_active(profile_id)
    
    # Start or restart service
    try:
        status = manager.get_status()
        if status["running"]:
            await manager.restart()
            action = "restarted"
        else:
            await manager.start()
            action = "started"
        
        return {
            "success": True,
            "message": f"Profile '{profile['name']}' deployed (service {action})",
            "profile": profile,
        }
    except Exception as e:
        logger.error(f"Failed to deploy profile '{profile['name']}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/from-current")
async def save_current_as_profile(request: Request):
    """Convenience: save the current running config as a new profile."""
    store = get_profile_store(request)
    manager = get_manager(request)
    
    if manager is None:
        raise HTTPException(status_code=503, detail="Service manager not available")
    
    data = await request.json()
    name = data.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Profile name is required")
    
    # Build a descriptive auto-name if not provided
    model_name = manager.config.get("model", {}).get("name", "unknown")
    variant = manager.config.get("model", {}).get("variant", "")
    
    try:
        profile = store.save_profile(
            name=name,
            config=manager.config,
            tags=data.get("tags", [model_name]),
            description=data.get("description", f"Saved from active config: {model_name} {variant}"),
        )
        return {"success": True, "profile": profile}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
