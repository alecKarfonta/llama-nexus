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
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GRAPHRAG_URL = "http://graphrag-api-1:8000"
GRAPHRAG_ENABLED = True


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


def create_embedder(request: Request, model_name: Optional[str] = None):
    """Create an embedder instance using app state factory."""
    embedder_factory = getattr(request.app.state, 'create_embedder', None)
    if not embedder_factory:
        raise HTTPException(status_code=503, detail="Embedder factory not available")
    return embedder_factory(model_name=model_name)


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
        embedding_model=data.get('embedding_model', 'all-MiniLM-L6-v2')
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
        collection_name = f"domain_{domain_id}"
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


# Document Management
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
    
    if content and doc_type == 'pdf':
        original_content = content
        content = extract_pdf_text(content)
        if not content or content.startswith('%PDF'):
            raise HTTPException(
                status_code=400, 
                detail="Failed to extract text from PDF. This may be a scanned document."
            )
    
    if content:
        existing = await rag['document_manager'].check_duplicate(content)
        if existing:
            raise HTTPException(status_code=409, detail=f"Duplicate content, existing document: {existing}")

    doc = Document(
        id=str(uuid.uuid4()),
        domain_id=data.get('domain_id'),
        name=data.get('name', 'Untitled'),
        doc_type=DocumentType(data.get('doc_type', 'txt')),
        content=content,
        source_url=data.get('source_url'),
        metadata=data.get('metadata', {})
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
    """Delete a document."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    deleted = await rag['document_manager'].delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted"}


@router.get("/documents/{document_id}/chunks")
async def get_document_chunks(request: Request, document_id: str):
    """Get document chunks."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    chunks = await rag['document_manager'].get_chunks(document_id)
    return {"chunks": [c.to_dict() for c in chunks]}


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
    collection_name = f"domain_{doc.domain_id}"
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
    
    collection_name = f"domain_{doc.domain_id}"
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
    
    embedding_model = data.get('embedding_model', domain.embedding_model if domain else 'all-MiniLM-L6-v2')
    embedder = create_embedder(request, model_name=embedding_model)
    
    texts = [chunk.content for chunk in chunks]
    embed_result = await embedder.embed(texts)
    
    collection_name = f"domain_{doc.domain_id}"
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
    """Retrieve relevant documents for a query."""
    from modules.rag.retrievers import VectorRetriever, RetrievalConfig
    
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    data = await request.json()
    query = data.get('query', '')
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    embedding_model = data.get('embedding_model', 'all-MiniLM-L6-v2')
    embedder = create_embedder(request, model_name=embedding_model)
    
    domain_id = data.get('domain_id')
    collection_name = f"domain_{domain_id}" if domain_id else "documents"
    
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
    
    return {
        "query": query,
        "results": [r.to_dict() for r in results]
    }


@router.post("/retrieve/hybrid")
async def hybrid_retrieve(request: Request):
    """Hybrid retrieval combining dense and sparse search."""
    from modules.rag.retrievers import HybridRetriever, RetrievalConfig
    
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    data = await request.json()
    query = data.get('query', '')
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    embedding_model = data.get('embedding_model', 'all-MiniLM-L6-v2')
    embedder = create_embedder(request, model_name=embedding_model)

    domain_id = data.get('domain_id')
    collection_name = f"domain_{domain_id}" if domain_id else "documents"

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
    
    return {
        "query": query,
        "retrieval_method": "hybrid",
        "alpha": config.alpha,
        "results": [r.to_dict() for r in results]
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
    model = data.get('model', 'all-MiniLM-L6-v2')
    
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
    """Get documents currently being processed or queued."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    from modules.rag.document_manager import DocumentStatus
    
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
            "documents": [d.to_dict() for d in processing_docs],
            "count": proc_total
        },
        "pending": {
            "documents": [d.to_dict() for d in pending_docs],
            "count": pend_total
        },
        "errors": {
            "documents": [d.to_dict() for d in error_docs],
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
