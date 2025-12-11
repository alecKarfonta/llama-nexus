"""Conversation storage routes."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


def get_conversation_store(request: Request):
    """Get the conversation store from app state."""
    store = getattr(request.app.state, 'conversation_store', None)
    if store is None:
        raise HTTPException(status_code=503, detail="Conversation storage not available")
    return store


@router.get("")
async def list_conversations(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    search: Optional[str] = None,
    tags: Optional[str] = None,
):
    """List all conversations with optional filtering."""
    store = get_conversation_store(request)
    tag_list = tags.split(',') if tags else None
    result = store.list_conversations(
        limit=limit,
        offset=offset,
        search=search,
        tags=tag_list,
    )
    return result


@router.post("")
async def create_conversation(request: Request):
    """Create a new conversation."""
    store = get_conversation_store(request)
    try:
        data = await request.json()
        conversation = store.create_conversation(
            title=data.get('title'),
            messages=data.get('messages', []),
            model=data.get('model'),
            settings=data.get('settings'),
            tags=data.get('tags', []),
        )
        return conversation.to_dict()
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_conversation_statistics(request: Request):
    """Get conversation statistics."""
    store = get_conversation_store(request)
    try:
        return store.get_statistics()
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_conversations(request: Request, query: str):
    """Search conversations by content."""
    store = get_conversation_store(request)
    results = store.search_conversations(query)
    return {"results": results}


@router.get("/{conversation_id}")
async def get_conversation(request: Request, conversation_id: str):
    """Get a specific conversation by ID."""
    store = get_conversation_store(request)
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation.to_dict()


@router.put("/{conversation_id}")
async def update_conversation(request: Request, conversation_id: str):
    """Update an existing conversation."""
    store = get_conversation_store(request)
    try:
        data = await request.json()
        conversation = store.update_conversation(
            conversation_id=conversation_id,
            messages=data.get('messages'),
            title=data.get('title'),
            model=data.get('model'),
            settings=data.get('settings'),
            tags=data.get('tags'),
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{conversation_id}/messages")
async def add_message_to_conversation(request: Request, conversation_id: str):
    """Add a message to an existing conversation."""
    store = get_conversation_store(request)
    try:
        data = await request.json()
        conversation = store.add_message(
            conversation_id=conversation_id,
            message=data,
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conversation_id}")
async def delete_conversation(request: Request, conversation_id: str):
    """Delete a conversation."""
    store = get_conversation_store(request)
    success = store.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "success", "deleted": conversation_id}


@router.get("/{conversation_id}/export")
async def export_conversation(
    request: Request,
    conversation_id: str,
    format: str = "json",
):
    """Export a conversation in the specified format (json or markdown)."""
    store = get_conversation_store(request)
    if format not in ['json', 'markdown']:
        raise HTTPException(status_code=400, detail="Format must be 'json' or 'markdown'")
    
    exported = store.export_conversation(conversation_id, format)
    if exported is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    media_type = 'application/json' if format == 'json' else 'text/markdown'
    filename = f"conversation-{conversation_id}.{'json' if format == 'json' else 'md'}"
    
    return Response(
        content=exported,
        media_type=media_type,
        headers={'Content-Disposition': f'attachment; filename="{filename}"'}
    )
