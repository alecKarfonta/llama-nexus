"""
Conversation Store Module
Provides server-side storage for chat conversations with save/load/export functionality.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class Conversation:
    """Represents a complete conversation with metadata."""
    id: str
    title: str
    messages: List[Message]
    created_at: str
    updated_at: str
    model: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    is_archived: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary for serialization."""
        return {
            'id': self.id,
            'title': self.title,
            'messages': [asdict(m) for m in self.messages],
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'model': self.model,
            'settings': self.settings,
            'tags': self.tags,
            'is_archived': self.is_archived,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create conversation from dictionary."""
        messages = [Message(**m) for m in data.get('messages', [])]
        return cls(
            id=data['id'],
            title=data['title'],
            messages=messages,
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            model=data.get('model'),
            settings=data.get('settings'),
            tags=data.get('tags', []),
            is_archived=data.get('is_archived', False),
        )


class ConversationStore:
    """
    Manages conversation storage on the server.
    Stores conversations as JSON files in a designated directory.
    """
    
    def __init__(self, storage_dir: str = None):
        """Initialize the conversation store."""
        if storage_dir is None:
            storage_dir = os.getenv('CONVERSATION_STORAGE_DIR', '/data/conversations')
        
        self.storage_dir = Path(storage_dir)
        self._ensure_storage_dir()
        
    def _ensure_storage_dir(self):
        """Create storage directory if it doesn't exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Conversation storage initialized at: {self.storage_dir}")
        
    def _get_conversation_path(self, conversation_id: str) -> Path:
        """Get the file path for a conversation."""
        return self.storage_dir / f"{conversation_id}.json"
    
    def create_conversation(
        self,
        title: str = None,
        messages: List[Dict[str, Any]] = None,
        model: str = None,
        settings: Dict[str, Any] = None,
        tags: List[str] = None,
    ) -> Conversation:
        """Create a new conversation."""
        conversation_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        # Generate title from first user message if not provided
        if title is None and messages:
            for msg in messages:
                if msg.get('role') == 'user':
                    title = msg.get('content', '')[:50] + ('...' if len(msg.get('content', '')) > 50 else '')
                    break
        
        if title is None:
            title = f"Conversation {now[:10]}"
        
        # Convert message dicts to Message objects
        message_objects = []
        if messages:
            for m in messages:
                message_objects.append(Message(
                    role=m.get('role', 'user'),
                    content=m.get('content', ''),
                    timestamp=m.get('timestamp', now),
                    reasoning_content=m.get('reasoning_content'),
                    tool_calls=m.get('tool_calls'),
                    tool_call_id=m.get('tool_call_id'),
                    name=m.get('name'),
                ))
        
        conversation = Conversation(
            id=conversation_id,
            title=title,
            messages=message_objects,
            created_at=now,
            updated_at=now,
            model=model,
            settings=settings,
            tags=tags or [],
        )
        
        self._save_conversation(conversation)
        logger.info(f"Created conversation: {conversation_id}")
        return conversation
    
    def _save_conversation(self, conversation: Conversation):
        """Save a conversation to disk."""
        path = self._get_conversation_path(conversation.id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(conversation.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Load a conversation by ID."""
        path = self._get_conversation_path(conversation_id)
        
        if not path.exists():
            logger.warning(f"Conversation not found: {conversation_id}")
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Conversation.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading conversation {conversation_id}: {e}")
            return None
    
    def update_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]] = None,
        title: str = None,
        model: str = None,
        settings: Dict[str, Any] = None,
        tags: List[str] = None,
        is_archived: bool = None,
    ) -> Optional[Conversation]:
        """Update an existing conversation."""
        conversation = self.get_conversation(conversation_id)
        
        if conversation is None:
            return None
        
        if title is not None:
            conversation.title = title
        
        if model is not None:
            conversation.model = model
            
        if settings is not None:
            conversation.settings = settings
            
        if tags is not None:
            conversation.tags = tags
            
        if is_archived is not None:
            conversation.is_archived = is_archived
        
        if messages is not None:
            now = datetime.utcnow().isoformat()
            conversation.messages = [
                Message(
                    role=m.get('role', 'user'),
                    content=m.get('content', ''),
                    timestamp=m.get('timestamp', now),
                    reasoning_content=m.get('reasoning_content'),
                    tool_calls=m.get('tool_calls'),
                    tool_call_id=m.get('tool_call_id'),
                    name=m.get('name'),
                )
                for m in messages
            ]
        
        conversation.updated_at = datetime.utcnow().isoformat()
        self._save_conversation(conversation)
        logger.info(f"Updated conversation: {conversation_id}")
        return conversation
    
    def add_message(
        self,
        conversation_id: str,
        message: Dict[str, Any],
    ) -> Optional[Conversation]:
        """Add a message to an existing conversation."""
        conversation = self.get_conversation(conversation_id)
        
        if conversation is None:
            return None
        
        now = datetime.utcnow().isoformat()
        new_message = Message(
            role=message.get('role', 'user'),
            content=message.get('content', ''),
            timestamp=message.get('timestamp', now),
            reasoning_content=message.get('reasoning_content'),
            tool_calls=message.get('tool_calls'),
            tool_call_id=message.get('tool_call_id'),
            name=message.get('name'),
        )
        
        conversation.messages.append(new_message)
        conversation.updated_at = now
        self._save_conversation(conversation)
        return conversation
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        path = self._get_conversation_path(conversation_id)
        
        if not path.exists():
            return False
        
        try:
            path.unlink()
            logger.info(f"Deleted conversation: {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}")
            return False
    
    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        search: str = None,
        tags: List[str] = None,
        include_archived: bool = False,
    ) -> Dict[str, Any]:
        """List all conversations with optional filtering."""
        conversations = []
        
        for path in self.storage_dir.glob('*.json'):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Apply search filter
                if search:
                    search_lower = search.lower()
                    title_match = search_lower in data.get('title', '').lower()
                    content_match = any(
                        search_lower in m.get('content', '').lower()
                        for m in data.get('messages', [])
                    )
                    if not (title_match or content_match):
                        continue
                
                # Apply tags filter
                if tags:
                    if not any(tag in data.get('tags', []) for tag in tags):
                        continue
                
                # Apply archived filter
                is_archived = data.get('is_archived', False)
                if not include_archived and is_archived:
                    continue
                
                # Return summary info for list
                conversations.append({
                    'id': data['id'],
                    'title': data['title'],
                    'created_at': data['created_at'],
                    'updated_at': data['updated_at'],
                    'model': data.get('model'),
                    'tags': data.get('tags', []),
                    'message_count': len(data.get('messages', [])),
                    'is_archived': data.get('is_archived', False),
                })
            except Exception as e:
                logger.error(f"Error reading conversation file {path}: {e}")
                continue
        
        # Sort by updated_at descending
        conversations.sort(key=lambda x: x['updated_at'], reverse=True)
        
        total = len(conversations)
        conversations = conversations[offset:offset + limit]
        
        return {
            'conversations': conversations,
            'total': total,
            'limit': limit,
            'offset': offset,
            'has_more': offset + limit < total,
        }
    
    def export_conversation(
        self,
        conversation_id: str,
        format: str = 'json',
    ) -> Optional[str]:
        """Export a conversation in the specified format."""
        conversation = self.get_conversation(conversation_id)
        
        if conversation is None:
            return None
        
        if format == 'json':
            return json.dumps(conversation.to_dict(), indent=2, ensure_ascii=False)
        
        elif format == 'markdown':
            lines = [
                f"# {conversation.title}",
                f"",
                f"**Created:** {conversation.created_at}",
                f"**Updated:** {conversation.updated_at}",
                f"**Model:** {conversation.model or 'Unknown'}",
                f"",
                "---",
                "",
            ]
            
            for msg in conversation.messages:
                role_label = {
                    'system': 'System',
                    'user': 'User',
                    'assistant': 'Assistant',
                    'tool': 'Tool Result',
                }.get(msg.role, msg.role.title())
                
                lines.append(f"### {role_label}")
                lines.append("")
                
                if msg.reasoning_content:
                    lines.append("<details>")
                    lines.append("<summary>Thinking Process</summary>")
                    lines.append("")
                    lines.append(msg.reasoning_content)
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")
                
                lines.append(msg.content)
                lines.append("")
                lines.append("---")
                lines.append("")
            
            return '\n'.join(lines)
        
        else:
            logger.warning(f"Unsupported export format: {format}")
            return None
    
    def search_conversations(self, query: str) -> List[Dict[str, Any]]:
        """Search conversations by content."""
        return self.list_conversations(search=query, limit=100)['conversations']
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        all_convos = self.list_conversations(limit=10000, include_archived=True)
        conversations = all_convos.get('conversations', [])
        
        active_count = 0
        archived_count = 0
        total_messages = 0
        models_used: Dict[str, int] = {}
        tags_used: Dict[str, int] = {}
        first_conversation = None
        recent_activity = 0
        
        # Calculate week ago timestamp
        from datetime import timedelta
        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        
        for conv in conversations:
            if conv.get('is_archived', False):
                archived_count += 1
            else:
                active_count += 1
            
            total_messages += conv.get('message_count', 0)
            
            model = conv.get('model')
            if model:
                models_used[model] = models_used.get(model, 0) + 1
            
            # Track tags
            for tag in conv.get('tags', []):
                tags_used[tag] = tags_used.get(tag, 0) + 1
            
            # Track oldest conversation
            created = conv.get('created_at')
            if created:
                if first_conversation is None or created < first_conversation:
                    first_conversation = created
            
            # Track recent activity (updated in last week)
            updated = conv.get('updated_at')
            if updated and updated >= week_ago:
                recent_activity += 1
        
        return {
            'active_conversations': active_count,
            'archived_conversations': archived_count,
            'total_conversations': len(conversations),
            'total_messages': total_messages,
            'recent_activity': recent_activity,
            'models_used': [{'model': m, 'count': c} for m, c in sorted(models_used.items(), key=lambda x: -x[1])],
            'tags_used': [tag for tag, _ in sorted(tags_used.items(), key=lambda x: -x[1])],
            'first_conversation': first_conversation,
        }


# Global instance
conversation_store = ConversationStore()
