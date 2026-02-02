"""
Document Manager for RAG System

Handles document CRUD operations, domain organization, and processing pipeline.
Features:
- Hierarchical domain structure
- Multiple document types (PDF, DOCX, TXT, MD, HTML, URL)
- Automatic text extraction and chunking
- Metadata management
- Search and filtering
"""

import os
import json
import hashlib
import logging
import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pathlib import Path
import uuid
import sqlite3
import aiosqlite

logger = logging.getLogger(__name__)


class DocumentStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"
    ARCHIVED = "archived"


class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "md"
    HTML = "html"
    CSV = "csv"
    JSON = "json"
    URL = "url"
    EPUB = "epub"


@dataclass
class Domain:
    """Document domain/collection"""
    id: str
    name: str
    description: str = ""
    parent_id: Optional[str] = None
    # Domain-specific settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "nomic-embed-text-v1.5"
    # Metadata
    document_count: int = 0
    total_chunks: int = 0
    created_at: str = ""
    updated_at: str = ""
    # Custom collection name (if set, use this instead of domain_{id})
    custom_collection: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Domain':
        # Filter to only valid fields for this dataclass
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class DocumentChunk:
    """A chunk of a document"""
    id: str
    document_id: str
    content: str
    chunk_index: int
    total_chunks: int
    # Position in source
    start_char: int = 0
    end_char: int = 0
    # Token count for this chunk
    token_count: int = 0
    # Metadata
    page_number: Optional[int] = None
    section_header: Optional[str] = None
    # Chunk type: "text" for regular text chunks, "visual" for image descriptions
    chunk_type: str = "text"
    # Path to image file (only for visual chunks)
    image_path: Optional[str] = None
    # Embedding
    embedding: Optional[List[float]] = None
    vector_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Don't include embedding in serialization (too large)
        data.pop('embedding', None)
        return data


@dataclass
class Document:
    """A document in the RAG system"""
    id: str
    domain_id: str
    name: str
    doc_type: DocumentType
    status: DocumentStatus = DocumentStatus.PENDING
    # Content
    content: str = ""
    content_hash: str = ""
    # Source
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    # Processing
    chunk_count: int = 0
    token_count: int = 0
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Timestamps
    created_at: str = ""
    updated_at: str = ""
    processed_at: Optional[str] = None
    # Error info
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['doc_type'] = self.doc_type.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        data['doc_type'] = DocumentType(data['doc_type'])
        data['status'] = DocumentStatus(data['status'])
        if isinstance(data.get('metadata'), str):
            data['metadata'] = json.loads(data['metadata'])
        return cls(**data)


class DocumentManager:
    """
    Manages documents and domains for the RAG system.
    
    Features:
    - SQLite-based persistence
    - Domain hierarchy management
    - Document CRUD operations
    - Full-text search on metadata
    - Duplicate detection
    """
    
    def __init__(self, db_path: str = "data/rag/documents.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._initialized = False
    
    async def initialize(self):
        """Initialize database schema"""
        if self._initialized:
            return
        
        async with aiosqlite.connect(self.db_path) as db:
            # Domains table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS domains (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    parent_id TEXT,
                    chunk_size INTEGER DEFAULT 512,
                    chunk_overlap INTEGER DEFAULT 50,
                    embedding_model TEXT DEFAULT 'nomic-embed-text',
                    document_count INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (parent_id) REFERENCES domains(id)
                )
            """)
            
            # Documents table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    domain_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    content TEXT DEFAULT '',
                    content_hash TEXT DEFAULT '',
                    source_path TEXT,
                    source_url TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    token_count INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    processed_at TEXT,
                    error_message TEXT,
                    FOREIGN KEY (domain_id) REFERENCES domains(id)
                )
            """)
            
            # Chunks table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    total_chunks INTEGER NOT NULL,
                    start_char INTEGER DEFAULT 0,
                    end_char INTEGER DEFAULT 0,
                    token_count INTEGER DEFAULT 0,
                    page_number INTEGER,
                    section_header TEXT,
                    chunk_type TEXT DEFAULT 'text',
                    image_path TEXT,
                    vector_id TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)
            
            # Add new columns to existing chunks table if they don't exist
            try:
                await db.execute("ALTER TABLE chunks ADD COLUMN chunk_type TEXT DEFAULT 'text'")
            except:
                pass  # Column already exists
            try:
                await db.execute("ALTER TABLE chunks ADD COLUMN image_path TEXT")
            except:
                pass  # Column already exists
            
            # Add token_count column to chunks table if it doesn't exist
            try:
                await db.execute("ALTER TABLE chunks ADD COLUMN token_count INTEGER DEFAULT 0")
            except:
                pass  # Column already exists
            
            # Add custom_collection column to domains table if it doesn't exist
            try:
                await db.execute("ALTER TABLE domains ADD COLUMN custom_collection TEXT")
            except:
                pass  # Column already exists
            
            # Indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_chunks_vector ON chunks(vector_id)")
            
            await db.commit()
            
            # Create default General domain if none exists
            # Use UUID5 for stable, predictable ID across restarts
            GENERAL_DOMAIN_NAMESPACE = uuid.UUID('d8f5c6a3-9b2e-4f1a-8c7d-3e5f9a1b2c4d')
            GENERAL_DOMAIN_ID = str(uuid.uuid5(GENERAL_DOMAIN_NAMESPACE, "general"))
            
            cursor = await db.execute(
                "SELECT id FROM domains WHERE id = ? OR name = 'General'",
                (GENERAL_DOMAIN_ID,)
            )
            existing = await cursor.fetchone()
            if not existing:
                default_domain = Domain(
                    id=GENERAL_DOMAIN_ID,
                    name="General",
                    description="Default document domain",
                    created_at=datetime.utcnow().isoformat(),
                    updated_at=datetime.utcnow().isoformat()
                )
                await self.create_domain(default_domain)
                logger.info(f"Created General domain with stable ID: {GENERAL_DOMAIN_ID}")
        
        self._initialized = True
        logger.info(f"DocumentManager initialized with database: {self.db_path}")
    
    async def get_general_domain(self) -> Optional[Domain]:
        """Get the General domain (default domain for documents).
        
        Returns the General domain which uses a stable UUID5-based ID.
        """
        GENERAL_DOMAIN_NAMESPACE = uuid.UUID('d8f5c6a3-9b2e-4f1a-8c7d-3e5f9a1b2c4d')
        GENERAL_DOMAIN_ID = str(uuid.uuid5(GENERAL_DOMAIN_NAMESPACE, "general"))
        
        domain = await self.get_domain(GENERAL_DOMAIN_ID)
        if domain:
            return domain
        
        # Fallback: find by name if UUID doesn't match (legacy data)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM domains WHERE name = 'General' ORDER BY created_at LIMIT 1"
            )
            row = await cursor.fetchone()
            if row:
                return Domain.from_dict(dict(row))
        
        return None
    
    # Domain Operations
    
    async def create_domain(self, domain: Domain) -> Domain:
        """Create a new domain"""
        now = datetime.utcnow().isoformat()
        domain.created_at = domain.created_at or now
        domain.updated_at = now
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO domains (id, name, description, parent_id, chunk_size,
                    chunk_overlap, embedding_model, document_count, total_chunks,
                    created_at, updated_at, custom_collection)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                domain.id, domain.name, domain.description, domain.parent_id,
                domain.chunk_size, domain.chunk_overlap, domain.embedding_model,
                domain.document_count, domain.total_chunks,
                domain.created_at, domain.updated_at, domain.custom_collection
            ))
            await db.commit()
        
        logger.info(f"Created domain: {domain.name} ({domain.id})")
        return domain
    
    async def get_domain(self, domain_id: str) -> Optional[Domain]:
        """Get domain by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM domains WHERE id = ?",
                (domain_id,)
            )
            row = await cursor.fetchone()
            if row:
                return Domain.from_dict(dict(row))
        return None
    
    async def list_domains(self, parent_id: Optional[str] = None) -> List[Domain]:
        """List domains, optionally filtered by parent"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if parent_id:
                cursor = await db.execute(
                    "SELECT * FROM domains WHERE parent_id = ? ORDER BY name",
                    (parent_id,)
                )
            else:
                cursor = await db.execute(
                    "SELECT * FROM domains ORDER BY name"
                )
            rows = await cursor.fetchall()
            return [Domain.from_dict(dict(row)) for row in rows]
    
    async def update_domain(self, domain: Domain) -> Domain:
        """Update domain"""
        domain.updated_at = datetime.utcnow().isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE domains SET
                    name = ?, description = ?, parent_id = ?,
                    chunk_size = ?, chunk_overlap = ?, embedding_model = ?,
                    document_count = ?, total_chunks = ?, updated_at = ?,
                    custom_collection = ?
                WHERE id = ?
            """, (
                domain.name, domain.description, domain.parent_id,
                domain.chunk_size, domain.chunk_overlap, domain.embedding_model,
                domain.document_count, domain.total_chunks, domain.updated_at,
                domain.custom_collection, domain.id
            ))
            await db.commit()
        
        return domain
    
    async def delete_domain(self, domain_id: str, cascade: bool = False) -> bool:
        """Delete domain. If cascade=True, delete all documents too."""
        async with aiosqlite.connect(self.db_path) as db:
            # Check for documents
            cursor = await db.execute(
                "SELECT COUNT(*) FROM documents WHERE domain_id = ?",
                (domain_id,)
            )
            count = (await cursor.fetchone())[0]
            
            if count > 0 and not cascade:
                logger.warning(f"Cannot delete domain {domain_id}: has {count} documents")
                return False
            
            if cascade:
                # Delete all documents and their chunks
                await db.execute(
                    "DELETE FROM chunks WHERE document_id IN (SELECT id FROM documents WHERE domain_id = ?)",
                    (domain_id,)
                )
                await db.execute(
                    "DELETE FROM documents WHERE domain_id = ?",
                    (domain_id,)
                )
            
            await db.execute("DELETE FROM domains WHERE id = ?", (domain_id,))
            await db.commit()
        
        logger.info(f"Deleted domain: {domain_id}")
        return True
    
    async def get_domain_tree(self) -> List[Dict[str, Any]]:
        """Get hierarchical domain tree"""
        domains = await self.list_domains()
        
        # Build tree
        domain_map = {d.id: {**d.to_dict(), "children": []} for d in domains}
        root = []
        
        for domain in domains:
            if domain.parent_id and domain.parent_id in domain_map:
                domain_map[domain.parent_id]["children"].append(domain_map[domain.id])
            else:
                root.append(domain_map[domain.id])
        
        return root
    
    # Document Operations
    
    async def create_document(self, document: Document) -> Document:
        """Create a new document"""
        now = datetime.utcnow().isoformat()
        document.created_at = document.created_at or now
        document.updated_at = now
        
        # Compute content hash for deduplication
        if document.content:
            document.content_hash = hashlib.md5(document.content.encode()).hexdigest()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO documents (id, domain_id, name, doc_type, status, content,
                    content_hash, source_path, source_url, chunk_count, token_count,
                    metadata, created_at, updated_at, processed_at, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                document.id, document.domain_id, document.name, document.doc_type.value,
                document.status.value, document.content, document.content_hash,
                document.source_path, document.source_url, document.chunk_count,
                document.token_count, json.dumps(document.metadata),
                document.created_at, document.updated_at, document.processed_at,
                document.error_message
            ))
            
            # Update domain document count
            await db.execute("""
                UPDATE domains SET 
                    document_count = document_count + 1,
                    updated_at = ?
                WHERE id = ?
            """, (now, document.domain_id))
            
            await db.commit()
        
        logger.info(f"Created document: {document.name} ({document.id})")
        return document
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM documents WHERE id = ?",
                (document_id,)
            )
            row = await cursor.fetchone()
            if row:
                return Document.from_dict(dict(row))
        return None
    
    async def list_documents(
        self,
        domain_id: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
        doc_type: Optional[DocumentType] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Document], int]:
        """List documents with filtering"""
        conditions = []
        params = []
        
        if domain_id:
            conditions.append("domain_id = ?")
            params.append(domain_id)
        if status:
            conditions.append("status = ?")
            params.append(status.value)
        if doc_type:
            conditions.append("doc_type = ?")
            params.append(doc_type.value)
        if search:
            conditions.append("(name LIKE ? OR metadata LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Get total count
            cursor = await db.execute(
                f"SELECT COUNT(*) FROM documents {where_clause}",
                params
            )
            total = (await cursor.fetchone())[0]
            
            # Get paginated results
            cursor = await db.execute(
                f"""SELECT * FROM documents {where_clause}
                    ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                params + [limit, offset]
            )
            rows = await cursor.fetchall()
            documents = [Document.from_dict(dict(row)) for row in rows]
        
        return documents, total
    
    async def update_document(self, document: Document) -> Document:
        """Update document"""
        document.updated_at = datetime.utcnow().isoformat()
        
        if document.content:
            document.content_hash = hashlib.md5(document.content.encode()).hexdigest()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE documents SET
                    name = ?, domain_id = ?, doc_type = ?, status = ?,
                    content = ?, content_hash = ?, source_path = ?, source_url = ?,
                    chunk_count = ?, token_count = ?, metadata = ?,
                    updated_at = ?, processed_at = ?, error_message = ?
                WHERE id = ?
            """, (
                document.name, document.domain_id, document.doc_type.value,
                document.status.value, document.content, document.content_hash,
                document.source_path, document.source_url, document.chunk_count,
                document.token_count, json.dumps(document.metadata),
                document.updated_at, document.processed_at, document.error_message,
                document.id
            ))
            await db.commit()
        
        return document
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document and its chunks"""
        async with aiosqlite.connect(self.db_path) as db:
            # Get domain_id for count update
            cursor = await db.execute(
                "SELECT domain_id FROM documents WHERE id = ?",
                (document_id,)
            )
            row = await cursor.fetchone()
            if not row:
                return False
            
            domain_id = row[0]
            
            # Delete chunks
            await db.execute(
                "DELETE FROM chunks WHERE document_id = ?",
                (document_id,)
            )
            
            # Delete document
            await db.execute(
                "DELETE FROM documents WHERE id = ?",
                (document_id,)
            )
            
            # Update domain count
            await db.execute("""
                UPDATE domains SET 
                    document_count = document_count - 1,
                    updated_at = ?
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), domain_id))
            
            await db.commit()
        
        logger.info(f"Deleted document: {document_id}")
        return True
    
    async def check_duplicate(self, content: str, domain_id: Optional[str] = None) -> Optional[str]:
        """Check if content already exists, return existing document ID if found.
        
        Args:
            content: The content to check for duplicates
            domain_id: If provided, only check within this domain. Otherwise check globally.
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        async with aiosqlite.connect(self.db_path) as db:
            if domain_id:
                # Domain-scoped duplicate check
                cursor = await db.execute(
                    "SELECT id FROM documents WHERE content_hash = ? AND domain_id = ?",
                    (content_hash, domain_id)
                )
            else:
                # Global duplicate check (legacy behavior)
                cursor = await db.execute(
                    "SELECT id FROM documents WHERE content_hash = ?",
                    (content_hash,)
                )
            row = await cursor.fetchone()
            if row:
                return row[0]
        return None
    
    # Chunk Operations
    
    async def save_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Save document chunks"""
        if not chunks:
            return 0
        
        now = datetime.utcnow().isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany("""
                INSERT OR REPLACE INTO chunks (id, document_id, content, chunk_index,
                    total_chunks, start_char, end_char, token_count, page_number, section_header,
                    chunk_type, image_path, vector_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    chunk.id, chunk.document_id, chunk.content, chunk.chunk_index,
                    chunk.total_chunks, chunk.start_char, chunk.end_char, chunk.token_count,
                    chunk.page_number, chunk.section_header, chunk.chunk_type,
                    chunk.image_path, chunk.vector_id, now
                )
                for chunk in chunks
            ])
            
            # Update document chunk count
            if chunks:
                await db.execute("""
                    UPDATE documents SET chunk_count = ?, updated_at = ?
                    WHERE id = ?
                """, (len(chunks), now, chunks[0].document_id))
            
            await db.commit()
        
        return len(chunks)
    
    async def get_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (document_id,)
            )
            rows = await cursor.fetchall()
            return [
                DocumentChunk(
                    id=row['id'],
                    document_id=row['document_id'],
                    content=row['content'],
                    chunk_index=row['chunk_index'],
                    total_chunks=row['total_chunks'],
                    start_char=row['start_char'],
                    end_char=row['end_char'],
                    token_count=row['token_count'] if 'token_count' in row.keys() else 0,
                    page_number=row['page_number'],
                    section_header=row['section_header'],
                    chunk_type=row['chunk_type'] if 'chunk_type' in row.keys() else 'text',
                    image_path=row['image_path'] if 'image_path' in row.keys() else None,
                    vector_id=row['vector_id']
                )
                for row in rows
            ]
    
    async def get_chunk_by_vector_id(self, vector_id: str) -> Optional[DocumentChunk]:
        """Get chunk by vector ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM chunks WHERE vector_id = ?",
                (vector_id,)
            )
            row = await cursor.fetchone()
            if row:
                return DocumentChunk(
                    id=row['id'],
                    document_id=row['document_id'],
                    content=row['content'],
                    chunk_index=row['chunk_index'],
                    total_chunks=row['total_chunks'],
                    start_char=row['start_char'],
                    end_char=row['end_char'],
                    token_count=row['token_count'] if 'token_count' in row.keys() else 0,
                    page_number=row['page_number'],
                    section_header=row['section_header'],
                    chunk_type=row['chunk_type'] if 'chunk_type' in row.keys() else 'text',
                    image_path=row['image_path'] if 'image_path' in row.keys() else None,
                    vector_id=row['vector_id']
                )
        return None
    
    async def get_chunks_paginated(
        self,
        document_id: str,
        offset: int = 0,
        limit: Optional[int] = None,
        max_limit: int = 500
    ) -> Tuple[List[DocumentChunk], int]:
        """Get paginated chunks for a document, ordered by chunk_index.
        
        Args:
            document_id: Document ID to get chunks for
            offset: Starting chunk index for pagination (default: 0)
            limit: Max chunks to return (None = all, capped at max_limit)
            max_limit: Maximum allowed limit (default: 500)
            
        Returns:
            Tuple of (chunks list, total chunk count)
        """
        # Cap limit to max_limit
        if limit is not None:
            limit = min(limit, max_limit)
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Get total count
            cursor = await db.execute(
                "SELECT COUNT(*) FROM chunks WHERE document_id = ?",
                (document_id,)
            )
            total = (await cursor.fetchone())[0]
            
            # Build query with pagination
            if limit is not None:
                cursor = await db.execute(
                    "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index LIMIT ? OFFSET ?",
                    (document_id, limit, offset)
                )
            else:
                cursor = await db.execute(
                    "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index LIMIT ? OFFSET ?",
                    (document_id, max_limit, offset)
                )
            
            rows = await cursor.fetchall()
            chunks = [
                DocumentChunk(
                    id=row['id'],
                    document_id=row['document_id'],
                    content=row['content'],
                    chunk_index=row['chunk_index'],
                    total_chunks=row['total_chunks'],
                    start_char=row['start_char'],
                    end_char=row['end_char'],
                    token_count=row['token_count'] if 'token_count' in row.keys() else 0,
                    page_number=row['page_number'],
                    section_header=row['section_header'],
                    chunk_type=row['chunk_type'] if 'chunk_type' in row.keys() else 'text',
                    image_path=row['image_path'] if 'image_path' in row.keys() else None,
                    vector_id=row['vector_id']
                )
                for row in rows
            ]
            
        return chunks, total
    
    async def get_chunks_by_range(
        self,
        document_id: str,
        start_index: int,
        end_index: int
    ) -> List[DocumentChunk]:
        """Get chunks within index range [start, end] for a document.
        
        Used for fetching neighboring chunks around a similarity match.
        
        Args:
            document_id: Document ID
            start_index: Starting chunk index (inclusive, clamped to 0)
            end_index: Ending chunk index (inclusive)
            
        Returns:
            List of chunks in the range, ordered by chunk_index
        """
        # Clamp start to 0
        start_index = max(0, start_index)
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT * FROM chunks 
                   WHERE document_id = ? AND chunk_index >= ? AND chunk_index <= ?
                   ORDER BY chunk_index""",
                (document_id, start_index, end_index)
            )
            rows = await cursor.fetchall()
            return [
                DocumentChunk(
                    id=row['id'],
                    document_id=row['document_id'],
                    content=row['content'],
                    chunk_index=row['chunk_index'],
                    total_chunks=row['total_chunks'],
                    start_char=row['start_char'],
                    end_char=row['end_char'],
                    token_count=row['token_count'] if 'token_count' in row.keys() else 0,
                    page_number=row['page_number'],
                    section_header=row['section_header'],
                    chunk_type=row['chunk_type'] if 'chunk_type' in row.keys() else 'text',
                    image_path=row['image_path'] if 'image_path' in row.keys() else None,
                    vector_id=row['vector_id']
                )
                for row in rows
            ]
    
    async def delete_chunks(self, document_id: str) -> int:
        """Delete all chunks for a document"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM chunks WHERE document_id = ?",
                (document_id,)
            )
            await db.commit()
            return cursor.rowcount
    
    # Statistics
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics"""
        async with aiosqlite.connect(self.db_path) as db:
            stats = {}
            
            # Domain count
            cursor = await db.execute("SELECT COUNT(*) FROM domains")
            stats['domain_count'] = (await cursor.fetchone())[0]
            
            # Document counts by status
            cursor = await db.execute("""
                SELECT status, COUNT(*) FROM documents GROUP BY status
            """)
            status_counts = dict(await cursor.fetchall())
            stats['documents'] = {
                'total': sum(status_counts.values()),
                'by_status': status_counts
            }
            
            # Chunk count
            cursor = await db.execute("SELECT COUNT(*) FROM chunks")
            stats['chunk_count'] = (await cursor.fetchone())[0]
            
            # Document types
            cursor = await db.execute("""
                SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type
            """)
            stats['documents']['by_type'] = dict(await cursor.fetchall())
        
        return stats
