"""Chat template management routes."""
from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/templates", tags=["templates"])


def get_manager(request: Request):
    """Get the LlamaCPP manager from app state."""
    return getattr(request.app.state, 'manager', None)


def get_merge_config_func(request: Request):
    """Get the config merge function from app state."""
    return getattr(request.app.state, 'merge_and_persist_config', None)


@router.get("")
async def list_templates(request: Request):
    """List available chat templates."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    try:
        templates_dir = Path(manager.config["template"]["directory"])
        selected = manager.config["template"].get("selected", "")
        if not templates_dir.exists():
            return {"success": True, "data": {"directory": str(templates_dir), "files": [], "selected": selected}}
        files = [p.name for p in templates_dir.glob("*.jinja") if p.is_file()]
        return {"success": True, "data": {"directory": str(templates_dir), "files": files, "selected": selected}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{filename}")
async def get_template(request: Request, filename: str):
    """Get a specific template content."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    try:
        templates_dir = Path(manager.config["template"]["directory"])
        path = templates_dir / filename
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="Template not found")
        return {"success": True, "data": {"filename": filename, "content": path.read_text()}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{filename}")
async def update_template(request: Request, filename: str, payload: Dict[str, Any]):
    """Update a template's content."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    try:
        content = payload.get("content")
        if not isinstance(content, str):
            raise HTTPException(status_code=400, detail="'content' must be a string")
        templates_dir = Path(manager.config["template"]["directory"]) 
        templates_dir.mkdir(parents=True, exist_ok=True)
        path = templates_dir / filename
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(content)
        tmp_path.replace(path)
        return {"success": True, "data": {"filename": filename}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select")
async def select_template(request: Request, payload: Dict[str, Any]):
    """Select a template as the active chat template."""
    manager = get_manager(request)
    merge_config = get_merge_config_func(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    if merge_config is None:
        raise HTTPException(status_code=503, detail="Config persistence not available")
    try:
        filename = payload.get("filename")
        if filename is None or not isinstance(filename, str):
            raise HTTPException(status_code=400, detail="'filename' is required")
        
        templates_dir = Path(manager.config["template"]["directory"]) 
        
        if filename == "":
            updated = merge_config({"template": {"directory": str(templates_dir), "selected": ""}})
            return {"success": True, "data": {"selected": updated["template"].get("selected", "")}}
        
        path = templates_dir / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="Template not found")
        
        updated = merge_config({"template": {"directory": str(templates_dir), "selected": filename}})
        return {"success": True, "data": {"selected": updated["template"].get("selected", filename)}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
async def create_template(request: Request, payload: Dict[str, Any]):
    """Create a new chat template."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    try:
        filename = payload.get("filename")
        content = payload.get("content", "")
        
        if not filename or not isinstance(filename, str):
            raise HTTPException(status_code=400, detail="'filename' is required")
        
        if not isinstance(content, str):
            raise HTTPException(status_code=400, detail="'content' must be a string")
        
        if not filename.endswith('.jinja'):
            filename += '.jinja'
        
        templates_dir = Path(manager.config["template"]["directory"])
        templates_dir.mkdir(parents=True, exist_ok=True)
        path = templates_dir / filename
        
        if path.exists():
            raise HTTPException(status_code=409, detail="Template already exists")
        
        path.write_text(content)
        return {"success": True, "data": {"filename": filename}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
