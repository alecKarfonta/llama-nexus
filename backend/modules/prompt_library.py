"""
Prompt Library Module
Provides storage and management for prompt templates with versioning and organization.
"""

import os
import json
import sqlite3
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import uuid

logger = logging.getLogger(__name__)


class PromptLibrary:
    """
    Manages prompt templates with categorization, versioning, and variable support.
    """
    
    def __init__(self, db_path: str = None):
        """Initialize the prompt library."""
        if db_path is None:
            db_path = os.getenv('PROMPT_LIBRARY_DB_PATH', '/data/prompt_library.db')
        
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_database()
        logger.info(f"Prompt library initialized with database at: {self.db_path}")
    
    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Prompt templates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prompts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    content TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    tags TEXT,  -- JSON array
                    variables TEXT,  -- JSON array of variable names extracted from content
                    is_system_prompt BOOLEAN DEFAULT FALSE,
                    is_favorite BOOLEAN DEFAULT FALSE,
                    use_count INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT,
                    created_by TEXT,
                    metadata TEXT  -- JSON object for additional data
                )
            ''')
            
            # Prompt versions table for history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    change_note TEXT,
                    created_at TEXT,
                    FOREIGN KEY (prompt_id) REFERENCES prompts(id),
                    UNIQUE (prompt_id, version)
                )
            ''')
            
            # Categories table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prompt_categories (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    color TEXT,
                    icon TEXT,
                    parent_id TEXT,
                    sort_order INTEGER DEFAULT 0,
                    FOREIGN KEY (parent_id) REFERENCES prompt_categories(id)
                )
            ''')
            
            # Insert default categories
            default_categories = [
                ('general', 'General', 'General purpose prompts', '#6B7280', 'folder'),
                ('coding', 'Coding', 'Programming and code-related prompts', '#3B82F6', 'code'),
                ('writing', 'Writing', 'Creative writing and content creation', '#8B5CF6', 'edit'),
                ('analysis', 'Analysis', 'Data analysis and research prompts', '#10B981', 'chart'),
                ('chat', 'Chat', 'Conversational and chatbot prompts', '#F59E0B', 'chat'),
                ('system', 'System', 'System prompts for model behavior', '#EF4444', 'settings'),
            ]
            
            for cat_id, name, desc, color, icon in default_categories:
                cursor.execute('''
                    INSERT OR IGNORE INTO prompt_categories (id, name, description, color, icon)
                    VALUES (?, ?, ?, ?, ?)
                ''', (cat_id, name, desc, color, icon))
            
            # Create indices
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_prompts_category ON prompts(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_prompts_name ON prompts(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_prompt_versions_prompt ON prompt_versions(prompt_id)')
            
            conn.commit()
    
    def _extract_variables(self, content: str) -> List[str]:
        """Extract template variables from content. Supports {{variable}} syntax."""
        pattern = r'\{\{(\w+)\}\}'
        variables = re.findall(pattern, content)
        return list(set(variables))  # Remove duplicates
    
    def _generate_id(self) -> str:
        """Generate a unique prompt ID."""
        return str(uuid.uuid4())[:12]
    
    # Prompt CRUD Operations
    
    def create_prompt(
        self,
        name: str,
        content: str,
        description: str = None,
        category: str = 'general',
        tags: List[str] = None,
        is_system_prompt: bool = False,
        created_by: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Create a new prompt template."""
        prompt_id = self._generate_id()
        now = datetime.utcnow().isoformat()
        variables = self._extract_variables(content)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO prompts 
                (id, name, description, content, category, tags, variables,
                 is_system_prompt, created_at, updated_at, created_by, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prompt_id, name, description, content, category,
                json.dumps(tags or []), json.dumps(variables),
                is_system_prompt, now, now, created_by, json.dumps(metadata or {})
            ))
            
            # Create initial version
            cursor.execute('''
                INSERT INTO prompt_versions (prompt_id, version, content, change_note, created_at)
                VALUES (?, 1, ?, 'Initial version', ?)
            ''', (prompt_id, content, now))
            
            conn.commit()
        
        logger.info(f"Created prompt: {name} (id: {prompt_id})")
        return self.get_prompt(prompt_id)
    
    def get_prompt(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get a prompt by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM prompts WHERE id = ?', (prompt_id,))
            row = cursor.fetchone()
            
            if row:
                data = dict(row)
                data['tags'] = json.loads(data.get('tags') or '[]')
                data['variables'] = json.loads(data.get('variables') or '[]')
                data['metadata'] = json.loads(data.get('metadata') or '{}')
                return data
        
        return None
    
    def update_prompt(
        self,
        prompt_id: str,
        name: str = None,
        content: str = None,
        description: str = None,
        category: str = None,
        tags: List[str] = None,
        is_system_prompt: bool = None,
        is_favorite: bool = None,
        metadata: Dict[str, Any] = None,
        change_note: str = None,
    ) -> Optional[Dict[str, Any]]:
        """Update an existing prompt."""
        prompt = self.get_prompt(prompt_id)
        if prompt is None:
            return None
        
        now = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if name is not None:
                updates.append('name = ?')
                params.append(name)
            
            if content is not None:
                updates.append('content = ?')
                params.append(content)
                updates.append('variables = ?')
                params.append(json.dumps(self._extract_variables(content)))
                
                # Create new version if content changed
                cursor.execute('''
                    SELECT MAX(version) FROM prompt_versions WHERE prompt_id = ?
                ''', (prompt_id,))
                max_version = cursor.fetchone()[0] or 0
                
                cursor.execute('''
                    INSERT INTO prompt_versions (prompt_id, version, content, change_note, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (prompt_id, max_version + 1, content, change_note or 'Updated', now))
            
            if description is not None:
                updates.append('description = ?')
                params.append(description)
            
            if category is not None:
                updates.append('category = ?')
                params.append(category)
            
            if tags is not None:
                updates.append('tags = ?')
                params.append(json.dumps(tags))
            
            if is_system_prompt is not None:
                updates.append('is_system_prompt = ?')
                params.append(is_system_prompt)
            
            if is_favorite is not None:
                updates.append('is_favorite = ?')
                params.append(is_favorite)
            
            if metadata is not None:
                updates.append('metadata = ?')
                params.append(json.dumps(metadata))
            
            if updates:
                updates.append('updated_at = ?')
                params.append(now)
                params.append(prompt_id)
                
                cursor.execute(f'''
                    UPDATE prompts SET {', '.join(updates)} WHERE id = ?
                ''', params)
                conn.commit()
        
        return self.get_prompt(prompt_id)
    
    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM prompt_versions WHERE prompt_id = ?', (prompt_id,))
            cursor.execute('DELETE FROM prompts WHERE id = ?', (prompt_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def list_prompts(
        self,
        category: str = None,
        tags: List[str] = None,
        search: str = None,
        is_system_prompt: bool = None,
        is_favorite: bool = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = 'updated_at',
        order_dir: str = 'DESC',
    ) -> Dict[str, Any]:
        """List prompts with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = 'SELECT * FROM prompts WHERE 1=1'
            params = []
            
            if category:
                query += ' AND category = ?'
                params.append(category)
            
            if search:
                query += ' AND (name LIKE ? OR description LIKE ? OR content LIKE ?)'
                search_pattern = f'%{search}%'
                params.extend([search_pattern, search_pattern, search_pattern])
            
            if is_system_prompt is not None:
                query += ' AND is_system_prompt = ?'
                params.append(is_system_prompt)
            
            if is_favorite is not None:
                query += ' AND is_favorite = ?'
                params.append(is_favorite)
            
            if tags:
                # Filter by any matching tag
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append("tags LIKE ?")
                    params.append(f'%"{tag}"%')
                query += f' AND ({" OR ".join(tag_conditions)})'
            
            # Count total
            count_query = query.replace('SELECT *', 'SELECT COUNT(*)')
            cursor.execute(count_query, params)
            total = cursor.fetchone()[0]
            
            # Add ordering and pagination
            allowed_order = ['name', 'updated_at', 'created_at', 'use_count', 'category']
            if order_by not in allowed_order:
                order_by = 'updated_at'
            order_dir = 'DESC' if order_dir.upper() == 'DESC' else 'ASC'
            
            query += f' ORDER BY {order_by} {order_dir} LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            
            prompts = []
            for row in cursor.fetchall():
                data = dict(row)
                data['tags'] = json.loads(data.get('tags') or '[]')
                data['variables'] = json.loads(data.get('variables') or '[]')
                data['metadata'] = json.loads(data.get('metadata') or '{}')
                prompts.append(data)
            
            return {
                'prompts': prompts,
                'total': total,
                'limit': limit,
                'offset': offset,
                'has_more': offset + limit < total,
            }
    
    def increment_use_count(self, prompt_id: str) -> None:
        """Increment the use count for a prompt."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE prompts SET use_count = use_count + 1, updated_at = ? WHERE id = ?
            ''', (datetime.utcnow().isoformat(), prompt_id))
            conn.commit()
    
    # Version Operations
    
    def get_versions(self, prompt_id: str) -> List[Dict[str, Any]]:
        """Get all versions of a prompt."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM prompt_versions 
                WHERE prompt_id = ? 
                ORDER BY version DESC
            ''', (prompt_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_version(self, prompt_id: str, version: int) -> Optional[Dict[str, Any]]:
        """Get a specific version of a prompt."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM prompt_versions 
                WHERE prompt_id = ? AND version = ?
            ''', (prompt_id, version))
            row = cursor.fetchone()
            
            return dict(row) if row else None
    
    def restore_version(self, prompt_id: str, version: int) -> Optional[Dict[str, Any]]:
        """Restore a prompt to a specific version."""
        version_data = self.get_version(prompt_id, version)
        if version_data is None:
            return None
        
        return self.update_prompt(
            prompt_id,
            content=version_data['content'],
            change_note=f'Restored from version {version}'
        )
    
    # Category Operations
    
    def list_categories(self) -> List[Dict[str, Any]]:
        """List all prompt categories."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT c.*, 
                    (SELECT COUNT(*) FROM prompts WHERE category = c.id) as prompt_count
                FROM prompt_categories c
                ORDER BY c.sort_order, c.name
            ''')
            
            return [dict(row) for row in cursor.fetchall()]
    
    def create_category(
        self,
        name: str,
        description: str = None,
        color: str = '#6B7280',
        icon: str = 'folder',
        parent_id: str = None,
    ) -> Dict[str, Any]:
        """Create a new category."""
        cat_id = name.lower().replace(' ', '_')[:20]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get max sort order
            cursor.execute('SELECT MAX(sort_order) FROM prompt_categories')
            max_order = cursor.fetchone()[0] or 0
            
            cursor.execute('''
                INSERT INTO prompt_categories (id, name, description, color, icon, parent_id, sort_order)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (cat_id, name, description, color, icon, parent_id, max_order + 1))
            conn.commit()
        
        return {'id': cat_id, 'name': name, 'description': description, 
                'color': color, 'icon': icon, 'parent_id': parent_id}
    
    # Template Rendering
    
    def render_prompt(self, prompt_id: str, variables: Dict[str, str] = None) -> str:
        """Render a prompt template with given variables."""
        prompt = self.get_prompt(prompt_id)
        if prompt is None:
            raise ValueError(f"Prompt {prompt_id} not found")
        
        content = prompt['content']
        
        if variables:
            for var_name, var_value in variables.items():
                content = content.replace(f'{{{{{var_name}}}}}', str(var_value))
        
        # Increment use count
        self.increment_use_count(prompt_id)
        
        return content
    
    # Import/Export
    
    def export_prompts(self, prompt_ids: List[str] = None) -> str:
        """Export prompts to JSON."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if prompt_ids:
                placeholders = ','.join('?' * len(prompt_ids))
                cursor.execute(f'SELECT * FROM prompts WHERE id IN ({placeholders})', prompt_ids)
            else:
                cursor.execute('SELECT * FROM prompts')
            
            prompts = []
            for row in cursor.fetchall():
                data = dict(row)
                data['tags'] = json.loads(data.get('tags') or '[]')
                data['variables'] = json.loads(data.get('variables') or '[]')
                data['metadata'] = json.loads(data.get('metadata') or '{}')
                prompts.append(data)
            
            return json.dumps({
                'version': '1.0',
                'exported_at': datetime.utcnow().isoformat(),
                'prompts': prompts,
            }, indent=2)
    
    def import_prompts(self, json_data: str, overwrite: bool = False) -> Dict[str, Any]:
        """Import prompts from JSON."""
        data = json.loads(json_data)
        prompts = data.get('prompts', [])
        
        imported = 0
        skipped = 0
        errors = []
        
        for prompt in prompts:
            try:
                existing = self.get_prompt(prompt['id']) if 'id' in prompt else None
                
                if existing and not overwrite:
                    skipped += 1
                    continue
                
                if existing and overwrite:
                    self.delete_prompt(prompt['id'])
                
                self.create_prompt(
                    name=prompt['name'],
                    content=prompt['content'],
                    description=prompt.get('description'),
                    category=prompt.get('category', 'general'),
                    tags=prompt.get('tags', []),
                    is_system_prompt=prompt.get('is_system_prompt', False),
                    metadata=prompt.get('metadata', {}),
                )
                imported += 1
                
            except Exception as e:
                errors.append(f"Error importing '{prompt.get('name', 'unknown')}': {str(e)}")
        
        return {
            'imported': imported,
            'skipped': skipped,
            'errors': errors,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get prompt library statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM prompts')
            total_prompts = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM prompts WHERE is_system_prompt = 1')
            system_prompts = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM prompts WHERE is_favorite = 1')
            favorites = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(use_count) FROM prompts')
            total_uses = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT COUNT(DISTINCT category) FROM prompts')
            categories_used = cursor.fetchone()[0]
            
            return {
                'total_prompts': total_prompts,
                'system_prompts': system_prompts,
                'favorites': favorites,
                'total_uses': total_uses,
                'categories_used': categories_used,
            }


# Global instance
prompt_library = PromptLibrary()
