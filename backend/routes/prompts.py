"""Prompt library routes."""
from fastapi import APIRouter, HTTPException, Request
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/prompts", tags=["prompts"])


def get_prompt_library(request: Request):
    """Get the prompt library from app state."""
    library = getattr(request.app.state, 'prompt_library', None)
    if library is None:
        raise HTTPException(status_code=503, detail="Prompt library not available")
    return library


@router.get("/stats")
async def get_prompt_stats(request: Request):
    """Get prompt library statistics."""
    library = get_prompt_library(request)
    return library.get_stats()


@router.get("/categories")
async def list_prompt_categories(request: Request):
    """List all prompt categories."""
    library = get_prompt_library(request)
    return {"categories": library.list_categories()}


@router.post("/categories")
async def create_prompt_category(request: Request):
    """Create a new prompt category."""
    library = get_prompt_library(request)
    data = await request.json()
    category = library.create_category(
        name=data.get('name'),
        description=data.get('description'),
        color=data.get('color', '#6B7280'),
        icon=data.get('icon', 'folder'),
        parent_id=data.get('parent_id'),
    )
    return category


@router.get("")
async def list_prompts(
    request: Request,
    category: str = None,
    search: str = None,
    is_system_prompt: bool = None,
    is_favorite: bool = None,
    limit: int = 50,
    offset: int = 0,
    order_by: str = 'updated_at',
    order_dir: str = 'DESC',
):
    """List prompts with optional filtering."""
    library = get_prompt_library(request)
    return library.list_prompts(
        category=category,
        search=search,
        is_system_prompt=is_system_prompt,
        is_favorite=is_favorite,
        limit=limit,
        offset=offset,
        order_by=order_by,
        order_dir=order_dir,
    )


@router.post("")
async def create_prompt(request: Request):
    """Create a new prompt template."""
    library = get_prompt_library(request)
    data = await request.json()
    prompt = library.create_prompt(
        name=data.get('name'),
        content=data.get('content'),
        description=data.get('description'),
        category=data.get('category', 'general'),
        tags=data.get('tags', []),
        is_system_prompt=data.get('is_system_prompt', False),
        created_by=data.get('created_by'),
        metadata=data.get('metadata', {}),
    )
    return prompt


@router.get("/{prompt_id}")
async def get_prompt(request: Request, prompt_id: str):
    """Get a prompt by ID."""
    library = get_prompt_library(request)
    prompt = library.get_prompt(prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt


@router.put("/{prompt_id}")
async def update_prompt(request: Request, prompt_id: str):
    """Update a prompt."""
    library = get_prompt_library(request)
    data = await request.json()
    prompt = library.update_prompt(
        prompt_id=prompt_id,
        name=data.get('name'),
        content=data.get('content'),
        description=data.get('description'),
        category=data.get('category'),
        tags=data.get('tags'),
        is_system_prompt=data.get('is_system_prompt'),
        is_favorite=data.get('is_favorite'),
        metadata=data.get('metadata'),
        change_note=data.get('change_note'),
    )
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt


@router.delete("/{prompt_id}")
async def delete_prompt(request: Request, prompt_id: str):
    """Delete a prompt."""
    library = get_prompt_library(request)
    deleted = library.delete_prompt(prompt_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"status": "deleted", "prompt_id": prompt_id}


@router.get("/{prompt_id}/versions")
async def get_prompt_versions(request: Request, prompt_id: str):
    """Get all versions of a prompt."""
    library = get_prompt_library(request)
    return {"versions": library.get_versions(prompt_id)}


@router.post("/{prompt_id}/restore/{version}")
async def restore_prompt_version(request: Request, prompt_id: str, version: int):
    """Restore a prompt to a specific version."""
    library = get_prompt_library(request)
    prompt = library.restore_version(prompt_id, version)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt or version not found")
    return prompt


@router.post("/{prompt_id}/render")
async def render_prompt(request: Request, prompt_id: str):
    """Render a prompt template with given variables."""
    library = get_prompt_library(request)
    data = await request.json()
    try:
        rendered = library.render_prompt(
            prompt_id=prompt_id,
            variables=data.get('variables', {}),
        )
        return {"rendered": rendered}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/export")
async def export_prompts(request: Request):
    """Export prompts to JSON."""
    library = get_prompt_library(request)
    data = await request.json()
    prompt_ids = data.get('prompt_ids')  # None exports all
    exported = library.export_prompts(prompt_ids)
    return {"data": exported}


@router.post("/import")
async def import_prompts(request: Request):
    """Import prompts from JSON."""
    library = get_prompt_library(request)
    data = await request.json()
    result = library.import_prompts(
        json_data=data.get('data'),
        overwrite=data.get('overwrite', False),
    )
    return result
