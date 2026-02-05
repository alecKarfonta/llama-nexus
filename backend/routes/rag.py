"""
RAG System Routes
Provides endpoints for document management, vector stores, retrieval, and knowledge graphs.
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from typing import Optional
from datetime import datetime
import uuid
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])

# Configuration - these will be set from main.py
RAG_AVAILABLE = False
USE_DEPLOYED_EMBEDDINGS = False
EMBEDDING_SERVICE_URL = "http://llamacpp-embed:8080/v1"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text-v1.5"
GRAPHRAG_URL = "http://graphrag-api-1:8000"
GRAPHRAG_ENABLED = True

# Context window expansion - include neighboring chunks in retrieval
RAG_CONTEXT_NEIGHBORS_BEFORE = int(os.getenv("RAG_CONTEXT_NEIGHBORS_BEFORE", "1"))
RAG_CONTEXT_NEIGHBORS_AFTER = int(os.getenv("RAG_CONTEXT_NEIGHBORS_AFTER", "1"))


def init_rag_config(
    rag_available: bool,
    use_deployed: bool,
    embedding_url: str,
    default_model: str,
    graphrag_url: str,
    graphrag_enabled: bool
):
    """Initialize RAG configuration from main app."""
    global RAG_AVAILABLE, USE_DEPLOYED_EMBEDDINGS, EMBEDDING_SERVICE_URL
    global DEFAULT_EMBEDDING_MODEL, GRAPHRAG_URL, GRAPHRAG_ENABLED
    RAG_AVAILABLE = rag_available
    USE_DEPLOYED_EMBEDDINGS = use_deployed
    EMBEDDING_SERVICE_URL = embedding_url
    DEFAULT_EMBEDDING_MODEL = default_model
    GRAPHRAG_URL = graphrag_url
    GRAPHRAG_ENABLED = graphrag_enabled


def get_rag_components(request: Request):
    """Helper to get RAG components from app state."""
    if not getattr(request.app.state, 'rag_available', False):
        raise HTTPException(status_code=503, detail="RAG system not available")
    return {
        'document_manager': getattr(request.app.state, 'document_manager', None),
        'vector_store': getattr(request.app.state, 'vector_store', None),
        'graph_rag': getattr(request.app.state, 'graph_rag', None),
        'document_discovery': getattr(request.app.state, 'document_discovery', None),
    }

# Model name aliases for backward compatibility
EMBEDDING_MODEL_ALIASES = {
    "nomic-embed-text": "nomic-embed-text-v1.5",
    "nomic-embed-text-v1.5": "nomic-embed-text-v1.5",  # Direct mapping
    "bge-large": "bge-large-en-v1.5",
}

def resolve_embedding_model_name(model_name: Optional[str]) -> str:
    """Resolve model aliases to full model names for backward compatibility."""
    if not model_name:
        return "nomic-embed-text-v1.5"
    return EMBEDDING_MODEL_ALIASES.get(model_name, model_name)


def create_embedder(request: Request, model_name: Optional[str] = None):
    """Create an embedder instance using app state factory."""
    embedder_factory = getattr(request.app.state, 'create_embedder', None)
    if not embedder_factory:
        raise HTTPException(status_code=503, detail="Embedder factory not available")
    # Resolve model aliases
    resolved_model = resolve_embedding_model_name(model_name)
    return embedder_factory(model_name=resolved_model)



def get_collection_name(domain) -> str:
    """Get Qdrant collection name for a domain.
    
    Uses custom_collection if set, otherwise defaults to domain_{id}.
    """
    if domain.custom_collection:
        return domain.custom_collection
    return f"domain_{domain.id}"


async def sync_memory_vectors_to_documents(
    domain,
    vector_store,
    document_manager
) -> int:
    """Sync vectors from a memory collection to SQLite documents.
    
    For domains with custom_collection set, this syncs Qdrant vectors
    to SQLite Document records on first access / refresh.
    
    Returns: Number of documents synced
    """
    from modules.rag.document_manager import Document, DocumentStatus, DocumentType
    
    if not domain.custom_collection:
        return 0
    
    collection = domain.custom_collection
    
    # Check if collection exists
    if not await vector_store.collection_exists(collection):
        return 0
    
    # Scroll through all vectors in the collection
    try:
        # Use scroll to get all points
        from qdrant_client.models import ScrollRequest
        points = []
        offset = None
        
        while True:
            result = await vector_store._client.scroll(
                collection_name=collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            batch, offset = result
            if not batch:
                break
            points.extend(batch)
            if offset is None:
                break
        
        # Get existing document IDs for this domain
        existing_docs, _ = await document_manager.list_documents(
            domain_id=domain.id,
            limit=10000
        )
        existing_ids = {d.id for d in existing_docs}
        
        synced = 0
        for point in points:
            # Use original_id from payload if available
            original_id = point.payload.get('original_id', str(point.id))
            doc_id = f"mem_{domain.custom_collection}_{original_id}"
            
            if doc_id in existing_ids:
                continue  # Already synced
            
            # Create document from vector payload
            content = point.payload.get('content', '')
            metadata = {k: v for k, v in point.payload.items() if k not in ('content', 'original_id')}
            
            doc = Document(
                id=doc_id,
                domain_id=domain.id,
                name=f"Memory: {original_id[:50]}..." if len(original_id) > 50 else f"Memory: {original_id}",
                doc_type=DocumentType.TXT,
                status=DocumentStatus.READY,
                content=content,
                chunk_count=1,
                metadata=metadata,
            )
            
            try:
                await document_manager.create_document(doc)
                synced += 1
            except Exception as e:
                logger.warning(f"Failed to sync memory {original_id}: {e}")
        
        logger.info(f"Synced {synced} memory vectors to domain {domain.id}")
        return synced
        
    except Exception as e:
        logger.error(f"Failed to sync memory vectors: {e}")
        return 0


def extract_pdf_text(content: str) -> str:
    """Extract text from PDF content (base64 encoded or raw bytes)."""
    import io
    import base64
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        logger.warning("PyPDF2 not installed, cannot extract PDF text")
        return content
    
    try:
        pdf_bytes = None
        
        # Try to decode as base64 first
        try:
            pdf_bytes = base64.b64decode(content)
            if pdf_bytes[:4] != b'%PDF':
                pdf_bytes = None
        except:
            pass
        
        # If base64 didn't work, try different encodings
        if pdf_bytes is None:
            for encoding in ['latin-1', 'utf-8', 'cp1252', 'iso-8859-1']:
                try:
                    pdf_bytes = content.encode(encoding)
                    if pdf_bytes[:4] == b'%PDF':
                        break
                    pdf_bytes = None
                except:
                    continue
        
        # Last resort
        if pdf_bytes is None and content.startswith('%PDF'):
            try:
                pdf_bytes = bytes([ord(c) & 0xFF for c in content])
            except:
                pass
        
        if pdf_bytes is None or pdf_bytes[:4] != b'%PDF':
            logger.warning("Content does not appear to be a valid PDF")
            return content
        
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        extracted_text = '\n\n'.join(text_parts)
        logger.info(f"Extracted {len(extracted_text)} chars from {len(reader.pages)} PDF pages")
        return extracted_text
    except Exception as e:
        logger.error(f"Failed to extract PDF text: {e}")
        return content


def extract_epub_text(content: str) -> str:
    """Extract text from EPUB content (base64 encoded)."""
    import io
    import base64
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("ebooklib not installed, cannot extract EPUB text")
        return content
    
    try:
        # Decode base64 content
        epub_bytes = base64.b64decode(content)
        
        # Read EPUB from bytes
        book = epub.read_epub(io.BytesIO(epub_bytes))
        
        # Extract text from all document items
        text_parts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            html_content = item.get_content()
            soup = BeautifulSoup(html_content, 'html.parser')
            # Get text from paragraphs and other text elements
            text = soup.get_text(separator='\n', strip=True)
            if text:
                text_parts.append(text)
        
        extracted_text = '\n\n'.join(text_parts)
        logger.info(f"Extracted {len(extracted_text)} chars from EPUB")
        return extracted_text
    except Exception as e:
        logger.error(f"Failed to extract EPUB text: {e}")
        return content


def extract_pdf_images(
    content: str,
    document_id: str,
    output_dir: str = "data/rag/images"
) -> list:
    """
    Extract images from PDF content and save to disk.
    
    Args:
        content: PDF content (base64 encoded or raw)
        document_id: Document ID for organizing images
        output_dir: Base directory for saving images
        
    Returns:
        List of tuples: (page_number, image_path, image_type)
    """
    import io
    import base64
    import os
    
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed, cannot extract PDF images")
        return []
    
    extracted_images = []
    
    try:
        # Decode PDF content to bytes (same logic as extract_pdf_text)
        pdf_bytes = None
        
        try:
            pdf_bytes = base64.b64decode(content)
            if pdf_bytes[:4] != b'%PDF':
                pdf_bytes = None
        except:
            pass
        
        if pdf_bytes is None:
            for encoding in ['latin-1', 'utf-8', 'cp1252', 'iso-8859-1']:
                try:
                    pdf_bytes = content.encode(encoding)
                    if pdf_bytes[:4] == b'%PDF':
                        break
                    pdf_bytes = None
                except:
                    continue
        
        if pdf_bytes is None and content.startswith('%PDF'):
            try:
                pdf_bytes = bytes([ord(c) & 0xFF for c in content])
            except:
                pass
        
        if pdf_bytes is None or pdf_bytes[:4] != b'%PDF':
            logger.warning("Cannot extract images: content is not valid PDF")
            return []
        
        # Create output directory for this document
        doc_image_dir = os.path.join(output_dir, document_id)
        os.makedirs(doc_image_dir, exist_ok=True)
        
        # Open PDF with PyMuPDF
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        image_count = 0
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]  # Image reference number
                
                try:
                    # Extract image
                    base_image = pdf_doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    
                    # Filter out tiny images (likely icons/bullets) and very small graphics
                    if width < 100 or height < 100:
                        continue
                    
                    # Filter out very thin images (likely lines/borders)
                    aspect_ratio = max(width, height) / max(min(width, height), 1)
                    if aspect_ratio > 10:
                        continue
                    
                    # Save image
                    image_name = f"page{page_num + 1}_img{img_index + 1}.{image_ext}"
                    image_path = os.path.join(doc_image_dir, image_name)
                    
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    extracted_images.append((page_num + 1, image_path, image_ext))
                    image_count += 1
                    
                    logger.debug(f"Extracted image: {image_path} ({width}x{height})")
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image {xref} from page {page_num + 1}: {e}")
                    continue
        
        pdf_doc.close()
        
        if image_count > 0:
            logger.info(f"Extracted {image_count} images from PDF document {document_id}")
        
        return extracted_images
        
    except Exception as e:
        logger.error(f"Failed to extract PDF images: {e}")
        return []


# Domain Management
@router.get("/domains")
async def list_rag_domains(request: Request):
    """List all document domains."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    domains = await rag['document_manager'].list_domains()
    return {"domains": [d.to_dict() for d in domains]}


@router.post("/domains")
async def create_rag_domain(request: Request):
    """Create a new domain."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    data = await request.json()
    from modules.rag.document_manager import Domain
    
    domain = Domain(
        id=str(uuid.uuid4()),
        name=data.get('name', 'New Domain'),
        description=data.get('description', ''),
        parent_id=data.get('parent_id'),
        chunk_size=data.get('chunk_size', 512),
        chunk_overlap=data.get('chunk_overlap', 50),
        embedding_model=data.get('embedding_model', 'nomic-embed-text-v1.5')
    )
    
    result = await rag['document_manager'].create_domain(domain)
    return {"domain": result.to_dict()}


@router.get("/domains/tree")
async def get_domain_tree(request: Request):
    """Get hierarchical domain tree."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    tree = await rag['document_manager'].get_domain_tree()
    return {"tree": tree}


@router.get("/domains/{domain_id}")
async def get_rag_domain(request: Request, domain_id: str):
    """Get domain details."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    domain = await rag['document_manager'].get_domain(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail="Domain not found")
    return {"domain": domain.to_dict()}


@router.put("/domains/{domain_id}")
async def update_rag_domain(request: Request, domain_id: str):
    """Update domain settings."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    domain = await rag['document_manager'].get_domain(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail="Domain not found")
    
    data = await request.json()
    
    if 'name' in data:
        domain.name = data['name']
    if 'description' in data:
        domain.description = data['description']
    if 'embedding_model' in data:
        domain.embedding_model = data['embedding_model']
    if 'chunk_size' in data:
        domain.chunk_size = data['chunk_size']
    if 'chunk_overlap' in data:
        domain.chunk_overlap = data['chunk_overlap']
    
    await rag['document_manager'].update_domain(domain)
    return {"domain": domain.to_dict()}


@router.delete("/domains/{domain_id}")
async def delete_rag_domain(request: Request, domain_id: str, cascade: bool = False):
    """Delete a domain."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")

    deleted = await rag['document_manager'].delete_domain(domain_id, cascade)
    if not deleted:
        raise HTTPException(status_code=400, detail="Cannot delete domain with documents")
    return {"status": "deleted"}


@router.post("/domains/{domain_id}/reindex")
async def reindex_domain(request: Request, domain_id: str, background_tasks: BackgroundTasks):
    """Reindex all documents in a domain."""
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    domain = await rag['document_manager'].get_domain(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail="Domain not found")
    
    data = await request.json()
    
    from modules.rag.document_manager import DocumentStatus
    docs, total = await rag['document_manager'].list_documents(
        domain_id=domain_id,
        limit=10000
    )
    
    if not docs:
        return {
            "status": "complete",
            "document_count": 0,
            "message": "No documents found in domain"
        }
    
    if data.get('recreate_collection', False):
        domain = await rag['document_manager'].get_domain(domain_id)
        collection_name = get_collection_name(domain) if domain else f"domain_{domain_id}"
        if await rag['vector_store'].collection_exists(collection_name):
            await rag['vector_store'].delete_collection(collection_name)
            logger.info(f"Deleted collection {collection_name} for reindexing")
    
    process_func = getattr(request.app.state, 'process_document_background', None)
    if process_func:
        for doc in docs:
            if doc.status != DocumentStatus.PENDING:
                background_tasks.add_task(
                    process_func,
                    doc.id,
                    data.get('chunking_strategy', 'semantic'),
                    data.get('chunk_size'),
                    data.get('chunk_overlap'),
                    data.get('embedding_model')
                )
    
    return {
        "status": "queued",
        "domain_id": domain_id,
        "document_count": len([d for d in docs if d.status != DocumentStatus.PENDING]),
        "message": f"Domain reindex started for {len(docs)} documents"
    }


@router.delete("/domains/{domain_id}/collection")
async def clear_domain_collection(request: Request, domain_id: str):
    """Clear all vectors from a domain's collection.
    
    This deletes the entire vector store collection for the domain,
    but keeps the documents in SQLite. Useful for completely resetting
    a domain's vector index.
    """
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    domain = await rag['document_manager'].get_domain(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail="Domain not found")
    
    collection_name = get_collection_name(domain) if domain else f"domain_{domain_id}"
    
    vectors_deleted = 0
    if await rag['vector_store'].collection_exists(collection_name):
        # Get collection info before deleting to report count
        try:
            info = await rag['vector_store'].get_collection_info(collection_name)
            vectors_deleted = info.get('vectors_count', 0) if info else 0
        except Exception:
            pass
        
        await rag['vector_store'].delete_collection(collection_name)
        logger.info(f"Cleared collection {collection_name} for domain {domain_id}")
    
    return {
        "status": "cleared",
        "domain_id": domain_id,
        "collection_name": collection_name,
        "vectors_deleted": vectors_deleted,
        "message": f"Collection '{collection_name}' has been cleared"
    }


# Document Management
@router.patch("/chunks/{chunk_id}")
async def update_chunk_metadata(request: Request, chunk_id: str):
    """
    Update chunk metadata.
    
    Updates metadata in both:
    1. SQLite chunks table
    2. Vector store payload (if vector_id exists)
    """
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    data = await request.json()
    metadata_update = data.get('metadata')
    
    if metadata_update is None:
        raise HTTPException(status_code=400, detail="No metadata provided")
    
    import aiosqlite
    import json
    from modules.rag.document_manager import DocumentChunk
    
    # 1. Fetch chunk from SQLite
    db_path = rag['document_manager'].db_path
    chunk = None
    
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        row = await cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Chunk not found")
            
        chunk_data = dict(row)
        # Parse existing metadata if it exists
        current_metadata = {}
        if chunk_data.get('metadata'):
            try:
                current_metadata = json.loads(chunk_data['metadata'])
            except:
                pass
        
        # Merge metadata
        updated_metadata = {**current_metadata, **metadata_update}
        
        # Update SQLite
        await db.execute(
            "UPDATE chunks SET metadata = ? WHERE id = ?",
            (json.dumps(updated_metadata), chunk_id)
        )
        await db.commit()
        
        vector_id = chunk_data.get('vector_id')
        document_id = chunk_data.get('document_id')
        
        # Get domain_id from document to know which collection update
        cursor = await db.execute("SELECT domain_id FROM documents WHERE id = ?", (document_id,))
        doc_row = await cursor.fetchone()
        domain_id = doc_row['domain_id'] if doc_row else None

    # 2. Update Vector Store Payload
    if vector_id and domain_id and rag['vector_store']:
        collection_name = f"domain_{domain_id}"
        
        # Check if custom collection
        domain = await rag['document_manager'].get_domain(domain_id)
        if domain:
            collection_name = get_collection_name(domain)
            
        try:
            # We need to fetch current payload first to merge? 
            # Or Qdrant set_payload merges? It merges by default.
            # We can use vector_store._client.set_payload directly if available,
            # but our VectorStore wrapper doesn't expose set_payload explicitly.
            # Let's check if we can access the underlying client properly.
            
            if hasattr(rag['vector_store'], '_client'):
                from qdrant_client.http import models
                
                # Check if set_payload exists on client (it should on AsyncQdrantClient)
                client = rag['vector_store']._client
                
                await client.set_payload(
                    collection_name=collection_name,
                    payload=metadata_update, # This merges with existing payload
                    points=[vector_id],
                    wait=True
                )
                logger.info(f"Updated payload for vector {vector_id} in {collection_name}")
            else:
                logger.warning("Vector store client not accessible, skipping payload update")
                
        except Exception as e:
            logger.error(f"Failed to update vector store payload: {e}")
            # We don't fail the request if vector store update fails, 
            # but we should warn.
            
    return {
        "id": chunk_id,
        "status": "updated",
        "metadata": updated_metadata
    }


@router.post("/chunks/batch-update")
async def batch_update_chunk_metadata(request: Request):
    """
    Batch update chunk metadata.
    
    Request body:
    {
        "updates": [
            {"chunk_id": "id1", "metadata": {"key": "value"}},
            {"chunk_id": "id2", "metadata": {"key": "value"}}
        ]
    }
    
    Or apply same metadata to multiple chunks:
    {
        "chunk_ids": ["id1", "id2", "id3"],
        "metadata": {"key": "value"}
    }
    """
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    data = await request.json()
    
    import aiosqlite
    import json
    
    db_path = rag['document_manager'].db_path
    results = []
    errors = []
    
    # Handle both formats
    if 'updates' in data:
        # Individual updates per chunk
        updates = data['updates']
    elif 'chunk_ids' in data and 'metadata' in data:
        # Same metadata for multiple chunks
        updates = [{"chunk_id": cid, "metadata": data['metadata']} for cid in data['chunk_ids']]
    else:
        raise HTTPException(status_code=400, detail="Provide either 'updates' array or 'chunk_ids' + 'metadata'")
    
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        
        for update in updates:
            chunk_id = update.get('chunk_id')
            metadata_update = update.get('metadata', {})
            
            if not chunk_id:
                errors.append({"error": "Missing chunk_id in update"})
                continue
                
            try:
                # Fetch current chunk
                cursor = await db.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
                row = await cursor.fetchone()
                
                if not row:
                    errors.append({"chunk_id": chunk_id, "error": "Chunk not found"})
                    continue
                
                chunk_data = dict(row)
                current_metadata = {}
                if chunk_data.get('metadata'):
                    try:
                        current_metadata = json.loads(chunk_data['metadata'])
                    except:
                        pass
                
                # Merge metadata
                updated_metadata = {**current_metadata, **metadata_update}
                
                # Update SQLite
                await db.execute(
                    "UPDATE chunks SET metadata = ? WHERE id = ?",
                    (json.dumps(updated_metadata), chunk_id)
                )
                
                results.append({
                    "chunk_id": chunk_id,
                    "status": "updated",
                    "metadata": updated_metadata
                })
                
                # Update vector store if available
                vector_id = chunk_data.get('vector_id')
                document_id = chunk_data.get('document_id')
                
                if vector_id and document_id and rag['vector_store']:
                    cursor = await db.execute("SELECT domain_id FROM documents WHERE id = ?", (document_id,))
                    doc_row = await cursor.fetchone()
                    if doc_row:
                        domain_id = doc_row['domain_id']
                        domain = await rag['document_manager'].get_domain(domain_id)
                        if domain:
                            collection_name = get_collection_name(domain)
                            try:
                                if hasattr(rag['vector_store'], '_client'):
                                    client = rag['vector_store']._client
                                    await client.set_payload(
                                        collection_name=collection_name,
                                        payload=metadata_update,
                                        points=[vector_id],
                                        wait=True
                                    )
                            except Exception as e:
                                logger.warning(f"Failed to update vector payload for {chunk_id}: {e}")
                                
            except Exception as e:
                errors.append({"chunk_id": chunk_id, "error": str(e)})
        
        await db.commit()
    
    return {
        "status": "completed",
        "updated": len(results),
        "errors": len(errors),
        "results": results,
        "error_details": errors if errors else None
    }

@router.get("/documents")
async def list_rag_documents(
    request: Request,
    domain_id: Optional[str] = None,
    status: Optional[str] = None,
    doc_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List documents with filtering."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    from modules.rag.document_manager import DocumentStatus, DocumentType
    
    # For memory domains (custom_collection), sync vectors to SQLite first
    if domain_id and rag['vector_store']:
        domain = await rag['document_manager'].get_domain(domain_id)
        if domain and domain.custom_collection:
            synced = await sync_memory_vectors_to_documents(
                domain, rag['vector_store'], rag['document_manager']
            )
            if synced > 0:
                logger.info(f"Synced {synced} memory vectors for domain {domain_id}")
    
    status_enum = DocumentStatus(status) if status else None
    type_enum = DocumentType(doc_type) if doc_type else None
    
    documents, total = await rag['document_manager'].list_documents(
        domain_id=domain_id,
        status=status_enum,
        doc_type=type_enum,
        search=search,
        limit=limit,
        offset=offset
    )
    
    return {
        "documents": [d.to_dict() for d in documents],
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.post("/documents")
async def create_rag_document(request: Request, background_tasks: BackgroundTasks):
    """Create/upload a new document."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")

    data = await request.json()
    from modules.rag.document_manager import Document, DocumentType, DocumentStatus

    content = data.get('content', '')
    doc_type = data.get('doc_type', 'txt')
    original_pdf_content = None
    
    if content and doc_type == 'pdf':
        original_pdf_content = content  # Preserve for VLM image extraction
        content = extract_pdf_text(content)
        if not content or content.startswith('%PDF'):
            raise HTTPException(
                status_code=400, 
                detail="Failed to extract text from PDF. This may be a scanned document."
            )
    elif content and doc_type == 'epub':
        original_content = content
        content = extract_epub_text(content)
        if not content or content == original_content:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract text from EPUB."
            )
    
    # Check for duplicates within the target domain only
    domain_id = data.get('domain_id')
    if content and domain_id:
        existing = await rag['document_manager'].check_duplicate(content, domain_id)
        if existing:
            raise HTTPException(status_code=409, detail=f"Duplicate content in this domain, existing document: {existing}")

    # Prepare metadata
    # NOTE: Storing original PDF content in vector store payload is disabled by default
    # to avoid massive payload bloat (can be 9MB+ per chunk). Enable with store_original_content=true
    # if VLM image extraction from original PDF is needed.
    metadata = data.get('metadata', {})
    store_original_content = data.get('store_original_content', False)
    if store_original_content and original_pdf_content:
        metadata['original_pdf_content'] = original_pdf_content
    
    # GraphRAG options (optional - only used if submodule available)
    graphrag_options = {
        'use_semantic_chunking': data.get('use_semantic_chunking', False),
        'build_knowledge_graph': data.get('build_knowledge_graph', False)
    }
    if graphrag_options['use_semantic_chunking'] or graphrag_options['build_knowledge_graph']:
        metadata['graphrag_options'] = graphrag_options

    doc = Document(
        id=str(uuid.uuid4()),
        domain_id=data.get('domain_id'),
        name=data.get('name', 'Untitled'),
        doc_type=DocumentType(data.get('doc_type', 'txt')),
        content=content,
        source_url=data.get('source_url'),
        metadata=metadata
    )

    result = await rag['document_manager'].create_document(doc)
    
    auto_process = data.get('auto_process', True)
    if auto_process and content:
        process_func = getattr(request.app.state, 'process_document_background', None)
        if process_func:
            background_tasks.add_task(
                process_func,
                result.id,
                data.get('chunking_strategy', 'semantic'),
                data.get('chunk_size'),
                data.get('chunk_overlap'),
                data.get('embedding_model')
            )
            logger.info(f"Queued document {result.id} for background processing")
    
    return {"document": result.to_dict(), "auto_processing": auto_process}


@router.post("/documents/upload")
async def upload_rag_document(request: Request, background_tasks: BackgroundTasks):
    """Upload a document file (FormData-based upload for batch ingestion)."""
    from fastapi import UploadFile
    import base64
    
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    # Parse multipart form data
    form = await request.form()
    file = form.get('file')
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    chunk_size = int(form.get('chunk_size', 512))
    chunk_overlap = int(form.get('chunk_overlap', 50))
    domain_id = form.get('domain_id')
    
    # If no domain_id provided, get default/first domain
    if not domain_id:
        general_domain = await rag['document_manager'].get_general_domain()
        if general_domain:
            domain_id = general_domain.id
        else:
            raise HTTPException(status_code=400, detail="No domain found. Please create a domain first.")
    
    from modules.rag.document_manager import Document, DocumentType, DocumentStatus
    
    # Read file content
    file_content = await file.read()
    filename = file.filename
    
    # Detect document type from extension
    ext = filename.split('.')[-1].lower() if '.' in filename else 'txt'
    doc_type_map = {
        'pdf': 'pdf', 'docx': 'docx', 'doc': 'docx', 'txt': 'txt',
        'md': 'md', 'html': 'html', 'htm': 'html', 'json': 'json',
        'csv': 'csv', 'xml': 'html', 'epub': 'epub'
    }
    doc_type = doc_type_map.get(ext, 'txt')
    
    # Process binary files (PDF, DOCX, EPUB)
    if doc_type in ['pdf', 'docx', 'epub']:
        content_b64 = base64.b64encode(file_content).decode('utf-8')
        
        if doc_type == 'pdf':
            content = extract_pdf_text(content_b64)
            if not content or content.startswith('%PDF'):
                raise HTTPException(status_code=400, detail="Failed to extract text from PDF")
        elif doc_type == 'epub':
            content = extract_epub_text(content_b64)
            if not content or content == content_b64:
                raise HTTPException(status_code=400, detail="Failed to extract text from EPUB")
        else:
            # For DOCX, store as base64 for now
            content = content_b64
    else:
        # Text-based files
        try:
            content = file_content.decode('utf-8')
        except UnicodeDecodeError:
            content = file_content.decode('latin-1')
    
    if not content:
        raise HTTPException(status_code=400, detail="Empty file content")
    
    # Check for duplicates within the target domain only
    existing = await rag['document_manager'].check_duplicate(content, domain_id)
    if existing:
        raise HTTPException(status_code=409, detail=f"Duplicate content in this domain, existing document: {existing}")
    
    # Create document
    doc = Document(
        id=str(uuid.uuid4()),
        domain_id=domain_id,
        name=filename,
        doc_type=DocumentType(doc_type),
        content=content,
        source_url=None,
        status=DocumentStatus.PENDING,
        metadata={'upload_method': 'file_upload', 'original_filename': filename}
    )
    
    result = await rag['document_manager'].create_document(doc)
    
    # Auto-process
    process_func = getattr(request.app.state, 'process_document_background', None)
    if process_func:
        background_tasks.add_task(
            process_func,
            result.id,
            'semantic',
            chunk_size,
            chunk_overlap,
            None
        )
        logger.info(f"Queued uploaded document {result.id} for background processing")
    
    return {"success": True, "document": result.to_dict(), "message": f"Uploaded {filename}"}


@router.get("/documents/{document_id}")
async def get_rag_document(request: Request, document_id: str):
    """Get document details."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    doc = await rag['document_manager'].get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"document": doc.to_dict()}


@router.delete("/documents/{document_id}")
async def delete_rag_document(request: Request, document_id: str):
    """Delete a document and its vectors from the vector store."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    # Get document first to access domain info
    doc = await rag['document_manager'].get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    vectors_deleted = 0
    
    # Delete vectors from vector store if available
    if rag['vector_store']:
        domain = await rag['document_manager'].get_domain(doc.domain_id)
        collection_name = get_collection_name(domain) if domain else f"domain_{doc.domain_id}"
        
        if await rag['vector_store'].collection_exists(collection_name):
            chunks = await rag['document_manager'].get_chunks(document_id)
            vector_ids = [chunk.vector_id for chunk in chunks if chunk.vector_id]
            
            if vector_ids:
                try:
                    await rag['vector_store'].delete_vectors(collection_name, vector_ids)
                    vectors_deleted = len(vector_ids)
                    logger.info(f"Deleted {vectors_deleted} vectors for document {document_id}")
                except Exception as e:
                    logger.warning(f"Error deleting vectors for document {document_id}: {e}")
    
    # Delete from SQLite (document + chunks)
    deleted = await rag['document_manager'].delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete document from database")
    
    return {"status": "deleted", "vectors_deleted": vectors_deleted}


@router.get("/documents/{document_id}/chunks")
async def get_document_chunks(
    request: Request,
    document_id: str,
    offset: int = 0,
    limit: Optional[int] = None,
    include_embeddings: bool = False
):
    """Get document chunks with pagination.
    
    Retrieve all chunks for a specific document, ordered by chunk_index.
    Supports pagination for sequential reading of large documents.
    
    Args:
        document_id: UUID of the document
        offset: Starting chunk index for pagination (default: 0)
        limit: Max chunks to return (default: all, max: 500)
        include_embeddings: Include vector embeddings in response (default: false)
    """
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    # Get document for metadata
    doc = await rag['document_manager'].get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get paginated chunks
    chunks, total_chunks = await rag['document_manager'].get_chunks_paginated(
        document_id,
        offset=offset,
        limit=limit
    )
    
    # Build chunk response
    chunk_data = []
    for c in chunks:
        chunk_dict = {
            "chunk_id": c.id,
            "chunk_index": c.chunk_index,
            "content": c.content,
            "token_count": c.token_count,
            "section_header": c.section_header,
            "page_number": c.page_number,
            "chunk_type": c.chunk_type,
            "metadata": {}
        }
        if include_embeddings and c.embedding:
            chunk_dict["embedding"] = c.embedding
        chunk_data.append(chunk_dict)
    
    return {
        "document_id": document_id,
        "document_name": doc.name,
        "domain_id": doc.domain_id,
        "total_chunks": total_chunks,
        "offset": offset,
        "limit": limit if limit else total_chunks,
        "chunks": chunk_data
    }



@router.post("/documents/{document_id}/process")
async def process_document(request: Request, document_id: str, background_tasks: BackgroundTasks):
    """Process document (chunk and embed)."""
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    data = await request.json()
    
    doc = await rag['document_manager'].get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    run_async = data.get('async', True)
    
    if run_async:
        process_func = getattr(request.app.state, 'process_document_background', None)
        if process_func:
            background_tasks.add_task(
                process_func,
                document_id,
                data.get('chunking_strategy', 'semantic'),
                data.get('chunk_size'),
                data.get('chunk_overlap'),
                data.get('embedding_model')
            )
        return {
            "status": "queued",
            "document_id": document_id,
            "message": "Document queued for processing"
        }
    
    # Synchronous processing - call the app's sync processing function
    sync_process_func = getattr(request.app.state, 'process_document_sync', None)
    if sync_process_func:
        result = await sync_process_func(request, document_id, data)
        return result
    
    raise HTTPException(status_code=503, detail="Processing functions not available")


@router.post("/documents/{document_id}/reprocess")
async def reprocess_document(request: Request, document_id: str, background_tasks: BackgroundTasks):
    """Reprocess an existing document."""
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    doc = await rag['document_manager'].get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    data = await request.json()
    
    # Delete existing vectors
    domain = await rag['document_manager'].get_domain(doc.domain_id)
    collection_name = get_collection_name(domain) if domain else f"domain_{doc.domain_id}"
    if await rag['vector_store'].collection_exists(collection_name):
        chunks = await rag['document_manager'].get_chunks(document_id)
        vector_ids = [chunk.vector_id for chunk in chunks if chunk.vector_id]
        if vector_ids:
            try:
                await rag['vector_store'].delete_vectors(collection_name, vector_ids)
                logger.info(f"Deleted {len(vector_ids)} vectors for document {document_id}")
            except Exception as e:
                logger.warning(f"Error deleting vectors: {e}")
    
    await rag['document_manager'].delete_chunks(document_id)
    
    from modules.rag.document_manager import DocumentStatus, DocumentType
    if doc.doc_type == DocumentType.PDF and doc.content.startswith('%PDF'):
        logger.info(f"Re-extracting text from PDF document {document_id}")
        extracted_text = extract_pdf_text(doc.content)
        if extracted_text and not extracted_text.startswith('%PDF'):
            doc.content = extracted_text
    
    doc.status = DocumentStatus.PENDING
    doc.chunk_count = 0
    doc.processed_at = None
    doc.error_message = None
    await rag['document_manager'].update_document(doc)
    
    process_func = getattr(request.app.state, 'process_document_background', None)
    if process_func:
        background_tasks.add_task(
            process_func,
            document_id,
            data.get('chunking_strategy', 'semantic'),
            data.get('chunk_size'),
            data.get('chunk_overlap'),
            data.get('embedding_model')
        )
    
    return {
        "status": "queued",
        "document_id": document_id,
        "message": "Document queued for reprocessing"
    }


@router.delete("/documents/{document_id}/vectors")
async def remove_document_from_vector_store(request: Request, document_id: str):
    """Remove a document's vectors from the vector store."""
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    doc = await rag['document_manager'].get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    domain = await rag['document_manager'].get_domain(doc.domain_id)
    collection_name = get_collection_name(domain) if domain else f"domain_{doc.domain_id}"
    chunks = await rag['document_manager'].get_chunks(document_id)
    vector_ids = [chunk.vector_id for chunk in chunks if chunk.vector_id]
    
    deleted_count = 0
    if vector_ids and await rag['vector_store'].collection_exists(collection_name):
        try:
            await rag['vector_store'].delete_vectors(collection_name, vector_ids)
            deleted_count = len(vector_ids)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting vectors: {str(e)}")
    
    return {
        "status": "deleted",
        "document_id": document_id,
        "vectors_deleted": deleted_count
    }


@router.post("/documents/{document_id}/add-to-vector-store")
async def add_document_to_vector_store(request: Request, document_id: str):
    """Add a document's chunks to vector store."""
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    doc = await rag['document_manager'].get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    chunks = await rag['document_manager'].get_chunks(document_id)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document has no chunks. Process document first.")
    
    data = await request.json()
    domain = await rag['document_manager'].get_domain(doc.domain_id)
    
    embedding_model = data.get('embedding_model', domain.embedding_model if domain else 'nomic-embed-text-v1.5')
    embedder = create_embedder(request, model_name=embedding_model)
    
    texts = [chunk.content for chunk in chunks]
    embed_result = await embedder.embed(texts)
    
    collection_name = get_collection_name(domain) if domain else f"domain_{doc.domain_id}"
    if not await rag['vector_store'].collection_exists(collection_name):
        from modules.rag.vector_stores.base import CollectionConfig
        collection_config = CollectionConfig(
            name=collection_name,
            vector_size=embed_result.dimensions
        )
        await rag['vector_store'].create_collection(collection_config)
    
    from modules.rag.vector_stores.base import VectorRecord
    records = []
    for chunk, embedding in zip(chunks, embed_result.embeddings):
        records.append(VectorRecord(
            id=chunk.vector_id or str(uuid.uuid4()),
            vector=embedding,
            payload={
                'document_id': document_id,
                'domain_id': doc.domain_id,
                'content': chunk.content,
                'chunk_index': chunk.chunk_index,
                'chunk_id': chunk.id,
                'metadata': doc.metadata
            }
        ))
    
    await rag['vector_store'].add_vectors(collection_name, records)
    
    return {
        "status": "added",
        "document_id": document_id,
        "vectors_added": len(records),
        "collection": collection_name
    }


@router.post("/documents/{document_id}/extract-knowledge")
async def extract_document_knowledge(request: Request, document_id: str):
    """Extract entities and relationships from a document to the knowledge graph."""
    import httpx
    
    if not GRAPHRAG_ENABLED:
        raise HTTPException(status_code=503, detail="GraphRAG integration not enabled")
    
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    doc = await rag['document_manager'].get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if not doc.content or len(doc.content.strip()) < 10:
        raise HTTPException(status_code=400, detail="Document has no content to extract from")
    
    try:
        body = await request.json()
    except:
        body = {}
    
    domain = body.get('domain', 'general')
    
    try:
        import tempfile
        import os as temp_os
        
        # Create a temporary file with the document content
        # Always use .txt since content is already extracted text
        # (PDF/DOCX content has already been extracted to text on upload)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(doc.content)
            tmp_path = tmp.name
        
        # Use .txt extension for the uploaded filename since content is text
        upload_name = doc.name
        if not upload_name.endswith('.txt'):
            upload_name = upload_name.rsplit('.', 1)[0] + '.txt'
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                # Upload the file to GraphRAG for ingestion (extracts AND stores in Neo4j)
                with open(tmp_path, 'rb') as f:
                    # Note: endpoint expects 'files' (plural) parameter
                    files = {'files': (upload_name, f, 'text/plain')}
                    response = await client.post(
                        f"{GRAPHRAG_URL}/ingest-documents",
                        files=files,
                        timeout=300.0
                    )
                response.raise_for_status()
                result = response.json()
                
                # Extract results from the response structure (use uploaded name)
                doc_result = result.get("results", {}).get(upload_name, {})
                
                return {
                    "status": "extracted",
                    "document_id": document_id,
                    "document_name": doc.name,
                    "entity_count": doc_result.get("entities", 0),
                    "relationship_count": doc_result.get("relationships", 0),
                    "chunks_created": doc_result.get("total_chunks", 0),
                    "extraction_method": "graphrag_ingest",
                    "message": "Document ingested and entities stored in knowledge graph"
                }
        finally:
            # Clean up temp file
            temp_os.unlink(tmp_path)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG extraction failed: {str(e)}")


@router.post("/documents/batch-process")
async def batch_process_documents(request: Request, background_tasks: BackgroundTasks):
    """Process multiple documents in batch."""
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    data = await request.json()
    document_ids = data.get('document_ids', [])
    
    if not document_ids:
        raise HTTPException(status_code=400, detail="No document IDs provided")
    
    missing = []
    for doc_id in document_ids:
        doc = await rag['document_manager'].get_document(doc_id)
        if not doc:
            missing.append(doc_id)
    
    if missing:
        raise HTTPException(status_code=404, detail=f"Documents not found: {', '.join(missing)}")
    
    process_func = getattr(request.app.state, 'process_document_background', None)
    if process_func:
        for doc_id in document_ids:
            background_tasks.add_task(
                process_func,
                doc_id,
                data.get('chunking_strategy', 'semantic'),
                data.get('chunk_size'),
                data.get('chunk_overlap'),
                data.get('embedding_model')
            )
    
    return {
        "status": "queued",
        "document_count": len(document_ids),
        "document_ids": document_ids,
        "message": f"{len(document_ids)} documents queued for processing"
    }


@router.post("/documents/process-all-pending")
async def process_all_pending(request: Request, background_tasks: BackgroundTasks):
    """Process all pending documents in a domain or globally."""
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    data = await request.json()
    domain_id = data.get('domain_id')
    
    from modules.rag.document_manager import DocumentStatus
    
    docs, total = await rag['document_manager'].list_documents(
        domain_id=domain_id,
        status=DocumentStatus.PENDING,
        limit=1000
    )
    
    if not docs:
        return {
            "status": "complete",
            "document_count": 0,
            "message": "No pending documents found"
        }
    
    process_func = getattr(request.app.state, 'process_document_background', None)
    if process_func:
        for doc in docs:
            background_tasks.add_task(
                process_func,
                doc.id,
                data.get('chunking_strategy', 'semantic'),
                data.get('chunk_size'),
                data.get('chunk_overlap'),
                data.get('embedding_model')
            )
    
    return {
        "status": "queued",
        "document_count": len(docs),
        "message": f"{len(docs)} pending documents queued for processing"
    }


@router.post("/reindex-all")
async def reindex_all_documents(request: Request, background_tasks: BackgroundTasks):
    """
    Clear existing vectors and reprocess all documents with a new embedding model.
    
    This is used when changing embedding models/dimensions and needing to rebuild
    the entire vector store.
    
    Body params:
        - domain_id: Optional domain to reindex (if not provided, reindexes all domains)
        - embedding_model: The new embedding model to use
        - chunking_strategy: Chunking strategy (default: semantic)
        - clear_vectors: Whether to delete existing vectors first (default: true)
    """
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    data = await request.json()
    domain_id = data.get('domain_id')
    embedding_model = data.get('embedding_model')
    chunking_strategy = data.get('chunking_strategy', 'semantic')
    clear_vectors = data.get('clear_vectors', True)
    
    if not embedding_model:
        raise HTTPException(status_code=400, detail="embedding_model is required")
    
    from modules.rag.document_manager import DocumentStatus
    
    # Get all documents to reprocess
    if domain_id:
        # Single domain reindex
        docs, total = await rag['document_manager'].list_documents(
            domain_id=domain_id,
            limit=10000
        )
        domains_to_clear = [domain_id]
    else:
        # Global reindex - get all documents
        docs, total = await rag['document_manager'].list_documents(limit=10000)
        # Get unique domain IDs
        domains_to_clear = list(set(doc.domain_id for doc in docs if doc.domain_id))
    
    if not docs:
        return {
            "status": "complete",
            "document_count": 0,
            "vectors_cleared": 0,
            "message": "No documents found to reindex"
        }
    
    vectors_cleared = 0
    
    # Clear existing vectors if requested
    if clear_vectors:
        for did in domains_to_clear:
            try:
                # Get domain to find collection name
                domain = await rag['document_manager'].get_domain(did)
                if domain:
                    collection_name = get_collection_name(domain)
                    # Delete the collection entirely
                    if await rag['vector_store'].collection_exists(collection_name):
                        # Get count before deleting
                        info = await rag['vector_store'].get_collection_info(collection_name)
                        if info:
                            vectors_cleared += info.get('vectors_count', 0)
                        await rag['vector_store'].delete_collection(collection_name)
                        logger.info(f"Cleared collection {collection_name} for reindex")
            except Exception as e:
                logger.warning(f"Failed to clear vectors for domain {did}: {e}")
    
    # Reset document status to pending for reprocessing
    docs_reset = 0
    for doc in docs:
        try:
            await rag['document_manager'].update_document(
                doc.id,
                status=DocumentStatus.PENDING,
                chunk_count=0,
                error_message=None
            )
            docs_reset += 1
        except Exception as e:
            logger.warning(f"Failed to reset document {doc.id}: {e}")
    
    # Queue documents for reprocessing
    process_func = getattr(request.app.state, 'process_document_background', None)
    if process_func:
        for doc in docs:
            background_tasks.add_task(
                process_func,
                doc.id,
                chunking_strategy,
                None,  # chunk_size
                None,  # chunk_overlap  
                embedding_model
            )
    
    return {
        "status": "queued",
        "document_count": len(docs),
        "documents_reset": docs_reset,
        "vectors_cleared": vectors_cleared,
        "embedding_model": embedding_model,
        "message": f"Cleared {vectors_cleared} vectors and queued {len(docs)} documents for reindexing with {embedding_model}"
    }

@router.post("/documents/batch-extract-knowledge")
async def batch_extract_document_knowledge(request: Request, background_tasks: BackgroundTasks):
    """Extract entities from multiple documents to the knowledge graph."""
    import httpx
    
    if not GRAPHRAG_ENABLED:
        raise HTTPException(status_code=503, detail="GraphRAG integration not enabled")
    
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    data = await request.json()
    document_ids = data.get('document_ids', [])
    domain = data.get('domain', 'general')
    
    if not document_ids:
        raise HTTPException(status_code=400, detail="No document IDs provided")
    
    valid_docs = []
    for doc_id in document_ids:
        doc = await rag['document_manager'].get_document(doc_id)
        if doc and doc.content and len(doc.content.strip()) >= 10:
            valid_docs.append({"id": doc.id, "name": doc.name})
    
    if not valid_docs:
        raise HTTPException(status_code=400, detail="No valid documents with content found")
    
    async def extract_documents_background(docs: list, domain: str):
        import tempfile
        import os as temp_os
        
        results = []
        async with httpx.AsyncClient(timeout=300.0) as client:
            for doc_info in docs:
                try:
                    doc = await rag['document_manager'].get_document(doc_info['id'])
                    if doc:
                        # Create temp file - always use .txt since content is already text
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
                            tmp.write(doc.content)
                            tmp_path = tmp.name
                        
                        # Use .txt extension for filename since content is text
                        upload_name = doc.name
                        if not upload_name.endswith('.txt'):
                            upload_name = upload_name.rsplit('.', 1)[0] + '.txt'
                        
                        try:
                            with open(tmp_path, 'rb') as f:
                                files = {'files': (upload_name, f, 'text/plain')}
                                response = await client.post(
                                    f"{GRAPHRAG_URL}/ingest-documents",
                                    files=files,
                                    timeout=300.0
                                )
                            response.raise_for_status()
                            result = response.json()
                            doc_result = result.get("results", {}).get(upload_name, {})
                            results.append({
                                "document_id": doc.id,
                                "document_name": doc.name,
                                "entity_count": doc_result.get("entities", 0),
                                "relationship_count": doc_result.get("relationships", 0),
                                "chunks_created": doc_result.get("total_chunks", 0),
                                "status": "success"
                            })
                            logger.info(f"Ingested {doc.name}: {doc_result.get('entities', 0)} entities extracted")
                        finally:
                            temp_os.unlink(tmp_path)
                except Exception as e:
                    results.append({
                        "document_id": doc_info['id'],
                        "document_name": doc_info['name'],
                        "status": "error",
                        "error": str(e)
                    })
                    logger.error(f"Failed to process {doc_info['name']}: {e}")
        return results
    
    background_tasks.add_task(extract_documents_background, valid_docs, domain)
    
    return {
        "status": "queued",
        "documents_queued": len(valid_docs),
        "document_ids": [d['id'] for d in valid_docs],
        "domain": domain
    }


# Collections
@router.get("/collections")
async def list_collections(request: Request):
    """List vector collections."""
    rag = get_rag_components(request)
    if not rag['vector_store']:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    collections = await rag['vector_store'].list_collections()
    return {"collections": collections}


@router.get("/collections/{name}")
async def get_collection_info(request: Request, name: str):
    """Get collection information."""
    rag = get_rag_components(request)
    if not rag['vector_store']:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    info = await rag['vector_store'].get_collection_info(name)
    if not info:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    return {
        "name": info.name,
        "vector_size": info.vector_size,
        "distance": info.distance.value,
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
        "status": info.status
    }


@router.delete("/collections/{name}")
async def delete_collection(request: Request, name: str):
    """Delete a collection."""
    rag = get_rag_components(request)
    if not rag['vector_store']:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    deleted = await rag['vector_store'].delete_collection(name)
    return {"status": "deleted" if deleted else "failed"}


# Retrieval
@router.post("/retrieve")
async def retrieve_documents(request: Request):
    """Retrieve relevant documents for a query.
    
    If domain_id is specified, search only that domain's collection.
    If domain_id is not specified:
      - If search_all_domains=True, search across all domain collections
      - Otherwise, use the General domain as default
    """
    from modules.rag.retrievers import VectorRetriever, RetrievalConfig
    
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    data = await request.json()
    query = data.get('query', '')
    search_all_domains = data.get('search_all_domains', False)
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    domain_id = data.get('domain_id')
    
    # Get the domain's embedding model and collection name
    default_embedding_model = 'nomic-embed-text-v1.5'
    domain = None
    
    if domain_id:
        domain = await rag['document_manager'].get_domain(domain_id)
        if domain and domain.embedding_model:
            default_embedding_model = domain.embedding_model
    elif not search_all_domains:
        # No domain_id specified - use the General domain as default
        domain = await rag['document_manager'].get_general_domain()
        if domain:
            domain_id = domain.id
            if domain.embedding_model:
                default_embedding_model = domain.embedding_model
            logger.info(f"No domain_id specified, using General domain: {domain.id}")
        else:
            logger.warning("No General domain found and no domain_id specified")
    
    embedding_model = data.get('embedding_model', default_embedding_model)
    embedder = create_embedder(request, model_name=embedding_model)
    
    # Cross-domain search: search all domain collections
    if search_all_domains:
        all_results = []
        domains = await rag['document_manager'].list_domains()
        
        for d in domains:
            coll_name = get_collection_name(d)
            if not await rag['vector_store'].collection_exists(coll_name):
                continue
            
            retriever = VectorRetriever(
                rag['vector_store'],
                embedder,
                rag['document_manager'],
                coll_name
            )
            
            config = RetrievalConfig(
                top_k=data.get('top_k', 10),
                score_threshold=data.get('score_threshold'),
                domain_ids=[d.id]
            )
            
            results = await retriever.retrieve(query, config)
            all_results.extend(results)
        
        # Sort by score descending and take top_k
        all_results.sort(key=lambda r: r.score if hasattr(r, 'score') else 0, reverse=True)
        all_results = all_results[:data.get('top_k', 10)]
        
        return {
            "query": query,
            "search_mode": "all_domains",
            "domains_searched": len(domains),
            "results": [r.to_dict() for r in all_results]
        }
    
    # Single domain search
    collection_name = get_collection_name(domain) if domain else None
    
    if not collection_name:
        raise HTTPException(
            status_code=400, 
            detail="No domain_id specified and no General domain available. Please specify a domain_id or create a General domain."
        )
    
    # Check if collection exists
    if not await rag['vector_store'].collection_exists(collection_name):
        return {
            "query": query,
            "collection": collection_name,
            "results": [],
            "message": f"Collection '{collection_name}' does not exist yet. Add documents to this domain first."
        }
    
    retriever = VectorRetriever(
        rag['vector_store'],
        embedder,
        rag['document_manager'],
        collection_name
    )
    
    config = RetrievalConfig(
        top_k=data.get('top_k', 10),
        score_threshold=data.get('score_threshold'),
        domain_ids=[domain_id] if domain_id else None
    )
    
    results = await retriever.retrieve(query, config)
    
    # Context window expansion - fetch neighboring chunks
    neighbors_before = data.get('context_neighbors_before', RAG_CONTEXT_NEIGHBORS_BEFORE)
    neighbors_after = data.get('context_neighbors_after', RAG_CONTEXT_NEIGHBORS_AFTER)
    
    enhanced_results = []
    for r in results:
        result_dict = r.to_dict()
        
        # Only expand context if we have chunk_index and document_id
        if neighbors_before > 0 or neighbors_after > 0:
            chunk_index = r.chunk_index
            if chunk_index is not None and r.document_id:
                start_idx = chunk_index - neighbors_before
                end_idx = chunk_index + neighbors_after
                
                # Fetch neighboring chunks
                neighbor_chunks = await rag['document_manager'].get_chunks_by_range(
                    r.document_id, start_idx, end_idx
                )
                
                # Build context object
                before_content = []
                after_content = []
                all_content = []
                
                for chunk in neighbor_chunks:
                    if chunk.chunk_index < chunk_index:
                        before_content.append(chunk.content)
                    elif chunk.chunk_index > chunk_index:
                        after_content.append(chunk.content)
                    all_content.append(chunk.content)
                
                result_dict['context'] = {
                    'before': before_content,
                    'after': after_content,
                    'combined': '\n\n'.join(all_content),
                    'neighbors_before': neighbors_before,
                    'neighbors_after': neighbors_after
                }
        
        enhanced_results.append(result_dict)
    
    return {
        "query": query,
        "collection": collection_name,
        "domain_id": domain_id,
        "context_expansion": {
            "neighbors_before": neighbors_before,
            "neighbors_after": neighbors_after
        },
        "results": enhanced_results
    }


@router.post("/retrieve/hybrid")
async def hybrid_retrieve(request: Request):
    """Hybrid retrieval combining dense and sparse search.
    
    Uses the General domain if no domain_id is specified.
    """
    from modules.rag.retrievers import HybridRetriever, RetrievalConfig
    
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    data = await request.json()
    query = data.get('query', '')
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    domain_id = data.get('domain_id')
    default_embedding_model = 'nomic-embed-text-v1.5'
    domain = None
    
    if domain_id:
        domain = await rag['document_manager'].get_domain(domain_id)
        if domain and domain.embedding_model:
            default_embedding_model = domain.embedding_model
    else:
        # No domain_id specified - use the General domain as default
        domain = await rag['document_manager'].get_general_domain()
        if domain:
            domain_id = domain.id
            if domain.embedding_model:
                default_embedding_model = domain.embedding_model
    
    embedding_model = data.get('embedding_model', default_embedding_model)
    embedder = create_embedder(request, model_name=embedding_model)
    
    collection_name = get_collection_name(domain) if domain else None
    
    if not collection_name:
        raise HTTPException(
            status_code=400, 
            detail="No domain_id specified and no General domain available."
        )
    
    # Check if collection exists
    if not await rag['vector_store'].collection_exists(collection_name):
        return {
            "query": query,
            "collection": collection_name,
            "retrieval_method": "hybrid",
            "results": [],
            "message": f"Collection '{collection_name}' does not exist yet."
        }

    retriever = HybridRetriever(
        rag['vector_store'],
        embedder,
        rag['document_manager'],
        collection_name
    )
    
    config = RetrievalConfig(
        top_k=data.get('top_k', 10),
        alpha=data.get('alpha', 0.5),
        domain_ids=[domain_id] if domain_id else None
    )
    
    results = await retriever.retrieve(query, config)
    
    # Context window expansion - fetch neighboring chunks
    neighbors_before = data.get('context_neighbors_before', RAG_CONTEXT_NEIGHBORS_BEFORE)
    neighbors_after = data.get('context_neighbors_after', RAG_CONTEXT_NEIGHBORS_AFTER)
    
    enhanced_results = []
    for r in results:
        result_dict = r.to_dict()
        
        # Only expand context if we have chunk_index and document_id
        if neighbors_before > 0 or neighbors_after > 0:
            chunk_index = r.chunk_index
            if chunk_index is not None and r.document_id:
                start_idx = chunk_index - neighbors_before
                end_idx = chunk_index + neighbors_after
                
                # Fetch neighboring chunks
                neighbor_chunks = await rag['document_manager'].get_chunks_by_range(
                    r.document_id, start_idx, end_idx
                )
                
                # Build context object
                before_content = []
                after_content = []
                all_content = []
                
                for chunk in neighbor_chunks:
                    if chunk.chunk_index < chunk_index:
                        before_content.append(chunk.content)
                    elif chunk.chunk_index > chunk_index:
                        after_content.append(chunk.content)
                    all_content.append(chunk.content)
                
                result_dict['context'] = {
                    'before': before_content,
                    'after': after_content,
                    'combined': '\n\n'.join(all_content),
                    'neighbors_before': neighbors_before,
                    'neighbors_after': neighbors_after
                }
        
        enhanced_results.append(result_dict)
    
    return {
        "query": query,
        "collection": collection_name,
        "retrieval_method": "hybrid",
        "alpha": config.alpha,
        "context_expansion": {
            "neighbors_before": neighbors_before,
            "neighbors_after": neighbors_after
        },
        "results": enhanced_results
    }


# Graph (local GraphRAG)
@router.get("/graph/entities")
async def list_entities(
    request: Request,
    entity_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List knowledge graph entities."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    from modules.rag.graph_rag import EntityType
    type_enum = EntityType(entity_type) if entity_type else None
    
    entities, total = await rag['graph_rag'].list_entities(
        entity_type=type_enum,
        search=search,
        limit=limit,
        offset=offset
    )
    
    return {
        "entities": [e.to_dict() for e in entities],
        "total": total
    }


@router.post("/graph/entities")
async def create_entity(request: Request):
    """Create a new entity."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    data = await request.json()
    from modules.rag.graph_rag import Entity, EntityType
    
    entity = Entity(
        id=str(uuid.uuid4()),
        name=data.get('name', ''),
        entity_type=EntityType(data.get('entity_type', 'concept')),
        description=data.get('description', ''),
        aliases=data.get('aliases', []),
        properties=data.get('properties', {})
    )
    
    result = await rag['graph_rag'].create_entity(entity)
    return {"entity": result.to_dict()}


@router.put("/graph/entities/{entity_id}")
async def update_entity(request: Request, entity_id: str):
    """Update an entity."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    entity = await rag['graph_rag'].get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    data = await request.json()
    from modules.rag.graph_rag import EntityType
    
    entity.name = data.get('name', entity.name)
    entity.description = data.get('description', entity.description)
    entity.aliases = data.get('aliases', entity.aliases)
    entity.properties = data.get('properties', entity.properties)
    if 'entity_type' in data:
        entity.entity_type = EntityType(data['entity_type'])
    
    result = await rag['graph_rag'].update_entity(entity)
    return {"entity": result.to_dict()}


@router.delete("/graph/entities/{entity_id}")
async def delete_entity(request: Request, entity_id: str):
    """Delete an entity."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    await rag['graph_rag'].delete_entity(entity_id)
    return {"status": "deleted"}


@router.post("/graph/entities/merge")
async def merge_entities(request: Request):
    """Merge multiple entities into one."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    data = await request.json()
    entity_ids = data.get('entity_ids', [])
    merged_name = data.get('merged_name', '')
    
    if len(entity_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 entities to merge")
    
    result = await rag['graph_rag'].merge_entities(entity_ids, merged_name)
    if not result:
        raise HTTPException(status_code=400, detail="Merge failed")
    
    return {"entity": result.to_dict()}


@router.get("/graph/relationships")
async def get_entity_relationships(request: Request, entity_id: str, direction: str = "both"):
    """Get relationships for an entity."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    relationships = await rag['graph_rag'].get_relationships(entity_id, direction)
    return {"relationships": [r.to_dict() for r in relationships]}


@router.post("/graph/relationships")
async def create_relationship(request: Request):
    """Create a new relationship."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    data = await request.json()
    from modules.rag.graph_rag import Relationship, RelationshipType
    
    rel = Relationship(
        id=str(uuid.uuid4()),
        source_id=data.get('source_id'),
        target_id=data.get('target_id'),
        relationship_type=RelationshipType(data.get('relationship_type', 'related_to')),
        description=data.get('description', ''),
        bidirectional=data.get('bidirectional', False)
    )
    
    result = await rag['graph_rag'].create_relationship(rel)
    return {"relationship": result.to_dict()}


@router.delete("/graph/relationships/{relationship_id}")
async def delete_relationship(request: Request, relationship_id: str):
    """Delete a relationship."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    await rag['graph_rag'].delete_relationship(relationship_id)
    return {"status": "deleted"}


@router.get("/graph/visualize")
async def get_graph_visualization(request: Request, limit: int = 500):
    """Get graph data for visualization."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    data = await rag['graph_rag'].get_visualization_data(limit)
    return data


@router.get("/graph/subgraph")
async def get_subgraph(request: Request, entity_ids: str, depth: int = 2):
    """Get subgraph around specified entities."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    ids = entity_ids.split(',')
    nodes, edges = await rag['graph_rag'].get_subgraph(ids, depth)
    
    return {
        "nodes": [{"id": n.id, "label": n.label, "type": n.type} for n in nodes],
        "edges": [{"source": e.source, "target": e.target, "label": e.label} for e in edges]
    }


@router.post("/graph/extract")
async def extract_entities_from_text(request: Request):
    """Extract entities and relationships from text."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    data = await request.json()
    text = data.get('text', '')
    save = data.get('save', False)
    
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    entities = await rag['graph_rag'].extract_entities_from_text(text)
    relationships = await rag['graph_rag'].extract_relationships_from_text(text, entities)
    
    if save:
        for entity in entities:
            existing = await rag['graph_rag'].find_entity_by_name(entity.name)
            if not existing:
                await rag['graph_rag'].create_entity(entity)
        
        for rel in relationships:
            await rag['graph_rag'].create_relationship(rel)
    
    return {
        "entities": [e.to_dict() for e in entities],
        "relationships": [r.to_dict() for r in relationships],
        "saved": save
    }


@router.get("/graph/statistics")
async def get_graph_statistics(request: Request):
    """Get graph statistics."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    stats = await rag['graph_rag'].get_statistics()
    return stats


# Discovery
@router.post("/discover/search")
async def discover_documents(request: Request):
    """Search web for documents."""
    rag = get_rag_components(request)
    if not rag['document_discovery']:
        raise HTTPException(status_code=503, detail="Document discovery not initialized")
    
    data = await request.json()
    query = data.get('query', '')
    max_results = data.get('max_results', 10)
    provider = data.get('provider', 'duckduckgo')
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    results = await rag['document_discovery'].search_web(query, max_results, provider)
    return {
        "query": query,
        "results": [r.to_dict() for r in results]
    }


@router.get("/discover/queue")
async def get_discovery_queue(
    request: Request,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get document review queue."""
    rag = get_rag_components(request)
    if not rag['document_discovery']:
        raise HTTPException(status_code=503, detail="Document discovery not initialized")
    
    from modules.rag.discovery import DiscoveryStatus
    status_enum = DiscoveryStatus(status) if status else None
    
    docs, total = await rag['document_discovery'].get_review_queue(
        status=status_enum,
        limit=limit,
        offset=offset
    )
    
    return {
        "documents": [d.to_dict() for d in docs],
        "total": total
    }


@router.post("/discover/{doc_id}/extract")
async def extract_discovered_content(request: Request, doc_id: str):
    """Extract content from discovered document URL."""
    rag = get_rag_components(request)
    if not rag['document_discovery']:
        raise HTTPException(status_code=503, detail="Document discovery not initialized")
    
    result = await rag['document_discovery'].extract_content(doc_id)
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"document": result.to_dict()}


@router.post("/discover/{doc_id}/approve")
async def approve_discovered_document(request: Request, doc_id: str):
    """Approve a discovered document."""
    rag = get_rag_components(request)
    if not rag['document_discovery']:
        raise HTTPException(status_code=503, detail="Document discovery not initialized")
    
    data = await request.json()
    domain_id = data.get('domain_id')
    
    result = await rag['document_discovery'].approve_document(doc_id, domain_id)
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"document": result.to_dict()}


@router.post("/discover/{doc_id}/reject")
async def reject_discovered_document(request: Request, doc_id: str):
    """Reject a discovered document."""
    rag = get_rag_components(request)
    if not rag['document_discovery']:
        raise HTTPException(status_code=503, detail="Document discovery not initialized")
    
    result = await rag['document_discovery'].reject_document(doc_id)
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"document": result.to_dict()}


@router.post("/discover/bulk-approve")
async def bulk_approve_documents(request: Request):
    """Bulk approve documents."""
    rag = get_rag_components(request)
    if not rag['document_discovery']:
        raise HTTPException(status_code=503, detail="Document discovery not initialized")
    
    data = await request.json()
    doc_ids = data.get('doc_ids', [])
    domain_id = data.get('domain_id')
    
    count = await rag['document_discovery'].bulk_approve(doc_ids, domain_id)
    return {"approved_count": count}


@router.get("/discover/statistics")
async def get_discovery_statistics(request: Request):
    """Get discovery statistics."""
    rag = get_rag_components(request)
    if not rag['document_discovery']:
        raise HTTPException(status_code=503, detail="Document discovery not initialized")
    
    stats = await rag['document_discovery'].get_statistics()
    return stats


# Vector Stores Management
@router.get("/vector-stores/collections")
async def list_vector_collections(request: Request):
    """List all vector store collections."""
    rag = get_rag_components(request)
    if not rag['vector_store']:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    collections = await rag['vector_store'].list_collections()
    return {"collections": collections}


@router.get("/vector-stores/collections/{collection_name}/stats")
async def get_vector_collection_stats(request: Request, collection_name: str):
    """Get statistics for a vector store collection."""
    rag = get_rag_components(request)
    if not rag['vector_store']:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    if not await rag['vector_store'].collection_exists(collection_name):
        raise HTTPException(status_code=404, detail="Collection not found")
    
    stats = await rag['vector_store'].get_collection_stats(collection_name)
    return {"collection": collection_name, "stats": stats}


@router.delete("/vector-stores/collections/{collection_name}")
async def delete_vector_collection(request: Request, collection_name: str):
    """Delete an entire vector store collection."""
    rag = get_rag_components(request)
    if not rag['vector_store']:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    if not await rag['vector_store'].collection_exists(collection_name):
        raise HTTPException(status_code=404, detail="Collection not found")
    
    await rag['vector_store'].delete_collection(collection_name)
    return {"status": "deleted", "collection": collection_name}


# Embeddings
@router.get("/embeddings/models")
async def list_embedding_models(request: Request):
    """List available embedding models."""
    from modules.rag.embedders import LocalEmbedder, APIEmbedder

    local_models = LocalEmbedder.list_available_models()
    api_models = APIEmbedder.list_available_models()

    return {
        "local_models": [{"name": m.name, "dimensions": m.dimensions, "description": m.description} for m in local_models],
        "api_models": [{"name": m.name, "dimensions": m.dimensions, "provider": m.provider, "description": m.description} for m in api_models]
    }


@router.get("/embeddings/config")
async def get_embedding_config(request: Request):
    """Get current embedding configuration for RAG."""
    embedding_manager = getattr(request.app.state, 'embedding_manager', None)
    
    return {
        "use_deployed_service": USE_DEPLOYED_EMBEDDINGS,
        "service_url": EMBEDDING_SERVICE_URL,
        "default_model": DEFAULT_EMBEDDING_MODEL,
        "service_running": embedding_manager.is_running() if embedding_manager else False,
        "service_status": embedding_manager.get_status() if embedding_manager and embedding_manager.is_running() else None
    }


@router.put("/embeddings/config")
async def update_embedding_rag_config(request: Request):
    """Update embedding configuration for RAG."""
    global USE_DEPLOYED_EMBEDDINGS, EMBEDDING_SERVICE_URL, DEFAULT_EMBEDDING_MODEL
    
    data = await request.json()
    embedding_manager = getattr(request.app.state, 'embedding_manager', None)
    
    if 'use_deployed_service' in data:
        USE_DEPLOYED_EMBEDDINGS = data['use_deployed_service']
        logger.info(f"Updated USE_DEPLOYED_EMBEDDINGS to: {USE_DEPLOYED_EMBEDDINGS}")
    
    if 'service_url' in data:
        EMBEDDING_SERVICE_URL = data['service_url']
        logger.info(f"Updated EMBEDDING_SERVICE_URL to: {EMBEDDING_SERVICE_URL}")
    
    if 'default_model' in data:
        DEFAULT_EMBEDDING_MODEL = data['default_model']
        logger.info(f"Updated DEFAULT_EMBEDDING_MODEL to: {DEFAULT_EMBEDDING_MODEL}")
    
    return {
        "success": True,
        "config": {
            "use_deployed_service": USE_DEPLOYED_EMBEDDINGS,
            "service_url": EMBEDDING_SERVICE_URL,
            "default_model": DEFAULT_EMBEDDING_MODEL,
            "service_running": embedding_manager.is_running() if embedding_manager else False
        }
    }


@router.post("/embeddings/embed")
async def embed_text(request: Request):
    """Embed text using specified model."""
    data = await request.json()
    texts = data.get('texts', [])
    model = data.get('model', 'nomic-embed-text-v1.5')
    
    if not texts:
        raise HTTPException(status_code=400, detail="Texts are required")

    embedder = create_embedder(request, model_name=model)
    result = await embedder.embed(texts)
    
    return {
        "model": result.model,
        "dimensions": result.dimensions,
        "embeddings_count": len(result.embeddings),
        "processing_time_ms": result.processing_time_ms
    }


# Processing Queue
@router.get("/processing/queue")
async def get_processing_queue(request: Request):
    """Get documents currently being processed or queued.
    
    Returns lightweight document summaries (without full content) to avoid
    overwhelming the browser with large responses during polling.
    """
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    from modules.rag.document_manager import DocumentStatus
    
    def doc_summary(doc):
        """Create a lightweight summary of a document (without content)."""
        data = doc.to_dict()
        # Remove potentially large fields for queue display
        data.pop('content', None)
        # Truncate error messages if too long
        if data.get('error_message') and len(data['error_message']) > 500:
            data['error_message'] = data['error_message'][:500] + '...'
        return data
    
    processing_docs, proc_total = await rag['document_manager'].list_documents(
        status=DocumentStatus.PROCESSING,
        limit=100
    )
    
    pending_docs, pend_total = await rag['document_manager'].list_documents(
        status=DocumentStatus.PENDING,
        limit=100
    )
    
    error_docs, err_total = await rag['document_manager'].list_documents(
        status=DocumentStatus.ERROR,
        limit=100
    )
    
    return {
        "processing": {
            "documents": [doc_summary(d) for d in processing_docs],
            "count": proc_total
        },
        "pending": {
            "documents": [doc_summary(d) for d in pending_docs],
            "count": pend_total
        },
        "errors": {
            "documents": [doc_summary(d) for d in error_docs],
            "count": err_total
        }
    }


# Statistics
@router.get("/statistics")
async def get_rag_statistics(request: Request):
    """Get overall RAG system statistics."""
    rag = get_rag_components(request)
    embedding_manager = getattr(request.app.state, 'embedding_manager', None)

    stats = {"available": getattr(request.app.state, 'rag_available', False)}
    
    if rag['document_manager']:
        stats['documents'] = await rag['document_manager'].get_statistics()
    
    if rag['vector_store']:
        collections = await rag['vector_store'].list_collections()
        collection_stats = []
        for coll in collections:
            try:
                coll_stats = await rag['vector_store'].get_collection_stats(coll)
                collection_stats.append({
                    "name": coll,
                    "stats": coll_stats
                })
            except:
                pass
        
        stats['vector_store'] = {
            "connected": True,
            "collections": len(collections),
            "collection_details": collection_stats
        }
    else:
        stats['vector_store'] = {"connected": False}
    
    if rag['graph_rag']:
        stats['graph'] = await rag['graph_rag'].get_statistics()
    
    if rag['document_discovery']:
        stats['discovery'] = await rag['document_discovery'].get_statistics()
    
    stats['embedding_service'] = {
        "running": embedding_manager.is_running() if embedding_manager else False,
        "config": embedding_manager.config if embedding_manager and embedding_manager.is_running() else None
    }

    return stats


# =============================================================================
# Memory API - External memory management for services like Chatter
# =============================================================================

# Default memory collections (for backwards compatibility, but any collection name is allowed)
DEFAULT_MEMORY_COLLECTIONS = ["chatter_stm", "chatter_core", "chatter_user", "chatter_chat"]

# Memory domain metadata
MEMORY_DOMAIN_INFO = {
    "chatter_stm": {"name": "Chatter: Short-Term Memory", "description": "Active conversation context and recent interactions"},
    "chatter_core": {"name": "Chatter: Core Memories", "description": "Pivotal moments, key insights, and important facts"},
    "chatter_user": {"name": "Chatter: User Profiles", "description": "User preferences, interests, and history"},
}

# UUID namespace for memory IDs (deterministic ID generation)
MEMORY_UUID_NAMESPACE = uuid.UUID('a1b2c3d4-e5f6-7890-abcd-ef1234567890')


def memory_id_to_uuid(memory_id: str) -> str:
    """Convert arbitrary memory ID string to a valid Qdrant UUID."""
    return str(uuid.uuid5(MEMORY_UUID_NAMESPACE, memory_id))


async def ensure_memory_domain(collection: str, document_manager) -> None:
    """Ensure a domain exists for a memory collection.
    
    Creates the domain if it doesn't exist, with custom_collection set.
    """
    from modules.rag.document_manager import Domain
    
    domain_id = f"memory_{collection}"
    
    # Check if domain already exists
    existing = await document_manager.get_domain(domain_id)
    if existing:
        return
    
    # Create domain with custom_collection
    info = MEMORY_DOMAIN_INFO.get(collection, {"name": f"Memory: {collection}", "description": ""})
    domain = Domain(
        id=domain_id,
        name=info["name"],
        description=info["description"],
        custom_collection=collection,
        embedding_model="nomic-embed-text-v1.5"
    )
    
    await document_manager.create_domain(domain)
    logger.info(f"Created memory domain: {domain.name} ({domain_id}) -> {collection}")


@router.post("/memory/upsert")
async def memory_upsert(request: Request):
    """
    Embed text and store in a Qdrant memory collection.
    Auto-creates collection if it doesn't exist.
    """
    rag = get_rag_components(request)
    if not rag['vector_store']:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    data = await request.json()
    
    collection = data.get('collection')
    memory_id = data.get('id')
    content = data.get('content')
    metadata = data.get('metadata', {})
    embedding_model = data.get('embedding_model', 'nomic-embed-text-v1.5')
    
    # Validate required fields
    if not collection:
        raise HTTPException(status_code=400, detail="collection is required")
    if not memory_id:
        raise HTTPException(status_code=400, detail="id is required")
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    
    # Validate collection name format (alphanumeric, underscores, hyphens only)
    import re
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', collection):
        raise HTTPException(
            status_code=400, 
            detail="Collection name must start with a letter and contain only alphanumeric characters, underscores, or hyphens"
        )
    
    # Generate embedding
    embedder = create_embedder(request, model_name=embedding_model)
    embed_result = await embedder.embed([content])
    embedding = embed_result.embeddings[0]
    
    # Auto-create collection if it doesn't exist
    if not await rag['vector_store'].collection_exists(collection):
        from modules.rag.vector_stores.base import CollectionConfig, DistanceMetric
        config = CollectionConfig(
            name=collection,
            vector_size=embed_result.dimensions,
            distance=DistanceMetric.COSINE
        )
        await rag['vector_store'].create_collection(config)
        logger.info(f"Created memory collection: {collection}")
        
        # Also create the domain for UI browsing
        if rag['document_manager']:
            await ensure_memory_domain(collection, rag['document_manager'])
    
    # Build payload with content for retrieval (include original ID)
    payload = {
        'content': content,
        'original_id': memory_id,  # Store original ID for API responses
        **metadata
    }
    
    # Convert user ID to valid UUID for Qdrant
    qdrant_id = memory_id_to_uuid(memory_id)
    
    # Upsert vector
    from modules.rag.vector_stores.base import VectorRecord
    record = VectorRecord(
        id=qdrant_id,
        vector=embedding,
        payload=payload
    )
    upserted = await rag['vector_store'].upsert(collection, [record])
    
    return {
        "success": True,
        "collection": collection,
        "id": memory_id,
        "vector_dimensions": embed_result.dimensions,
        "upserted_count": upserted
    }


@router.post("/memory/search")
async def memory_search(request: Request):
    """
    Search across one or more memory collections.
    Returns merged results sorted by score.
    """
    rag = get_rag_components(request)
    if not rag['vector_store']:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    data = await request.json()
    
    query = data.get('query')
    collections = data.get('collections', DEFAULT_MEMORY_COLLECTIONS)
    top_k = data.get('top_k', 5)
    min_score = data.get('min_score', 0.0)
    embedding_model = data.get('embedding_model', 'nomic-embed-text-v1.5')
    
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    
    # Validate collection name formats
    import re
    invalid_collections = [c for c in collections if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', c)]
    if invalid_collections:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid collection names: {invalid_collections}. Names must start with a letter and contain only alphanumeric characters, underscores, or hyphens"
        )
    
    # Generate query embedding
    embedder = create_embedder(request, model_name=embedding_model)
    query_embedding = await embedder.embed_query(query)
    
    # Filter to only collections that exist
    existing_collections = []
    for coll in collections:
        if await rag['vector_store'].collection_exists(coll):
            existing_collections.append(coll)
    
    if not existing_collections:
        return {
            "query": query,
            "results": [],
            "collections_searched": [],
            "total_results": 0
        }
    
    # Search each collection
    all_results = []
    for coll in existing_collections:
        results = await rag['vector_store'].search(
            collection=coll,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=min_score
        )
        for r in results:
            # Use original_id from payload if available, otherwise use Qdrant ID
            original_id = r.payload.get('original_id', r.id)
            all_results.append({
                "collection": coll,
                "id": original_id,
                "score": r.score,
                "content": r.payload.get('content', ''),
                "metadata": {k: v for k, v in r.payload.items() if k not in ('content', 'original_id')}
            })
    
    # Sort by score descending
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Limit total results
    max_results = top_k * len(existing_collections)
    all_results = all_results[:max_results]
    
    return {
        "query": query,
        "results": all_results,
        "collections_searched": existing_collections,
        "total_results": len(all_results)
    }


@router.delete("/memory/{collection}/{memory_id}")
async def memory_delete(request: Request, collection: str, memory_id: str):
    """
    Delete a vector from a memory collection.
    """
    rag = get_rag_components(request)
    if not rag['vector_store']:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    # Validate collection name format
    import re
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', collection):
        raise HTTPException(
            status_code=400,
            detail="Collection name must start with a letter and contain only alphanumeric characters, underscores, or hyphens"
        )
    
    # Check if collection exists
    if not await rag['vector_store'].collection_exists(collection):
        raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")
    
    # Convert user ID to Qdrant UUID
    qdrant_id = memory_id_to_uuid(memory_id)
    
    # Delete vector
    deleted = await rag['vector_store'].delete(collection, [qdrant_id])
    
    return {
        "success": deleted > 0,
        "collection": collection,
        "id": memory_id,
        "deleted_count": deleted
    }


@router.get("/memory/collections")
async def memory_list_collections(request: Request):
    """
    List memory collections and their stats.
    """
    rag = get_rag_components(request)
    if not rag['vector_store']:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    collections_info = []
    
    for coll in DEFAULT_MEMORY_COLLECTIONS:
        exists = await rag['vector_store'].collection_exists(coll)
        info = {"name": coll, "exists": exists}
        
        if exists:
            coll_info = await rag['vector_store'].get_collection_info(coll)
            if coll_info:
                info["stats"] = {
                    "vectors_count": coll_info.vectors_count,
                    "points_count": coll_info.points_count
                }
        
        collections_info.append(info)
    
    return {"collections": collections_info}

