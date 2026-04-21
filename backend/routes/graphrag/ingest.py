"""
GraphRAG Ingest Routes

Document upload, batch upload, rebuild KG, entity extraction,
entity linking, disambiguation, hybrid processing.

All endpoints use the shared connection-pooled HTTP client.
"""

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from typing import Any, List
import httpx
import logging

from .helpers import check_graphrag_enabled, get_http_client, graphrag_url as _graphrag_url
from modules.graphrag.config import get_graphrag_config

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/extract")
async def graphrag_extract_entities(request: Request):
    """Extract entities and relationships from text using graphrag GLiNER."""
    check_graphrag_enabled()

    data = await request.json()
    text = data.get("text")

    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/extract-entities-relations",
            json=data,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/ingest")
async def graphrag_ingest_document(
    file: Any = None,
    domain: str = "general",
    use_semantic_chunking: bool = True,
    build_knowledge_graph: bool = True,
):
    """
    Ingest a document through graphrag service.
    Note: For file uploads, use /api/v1/graphrag/ingest/upload endpoint.
    This endpoint is for JSON-based document submission.
    """
    check_graphrag_enabled()

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/ingest-documents",
            json={
                "domain": domain,
                "use_semantic_chunking": use_semantic_chunking,
                "build_knowledge_graph": build_knowledge_graph,
            },
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/documents")
async def graphrag_list_documents():
    """List documents in graphrag service."""
    check_graphrag_enabled()

    try:
        client = get_http_client()
        response = await client.get(f"{_graphrag_url()}/documents/list")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.delete("/documents/{document_name}")
async def graphrag_delete_document(document_name: str):
    """Delete a document from graphrag service."""
    check_graphrag_enabled()

    try:
        client = get_http_client()
        response = await client.delete(f"{_graphrag_url()}/documents/{document_name}")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/ingest/upload")
async def graphrag_upload_document(
    file: UploadFile = File(...),
    domain: str = Form("general"),
    use_semantic_chunking: bool = Form(True),
    build_knowledge_graph: bool = Form(True),
):
    """Upload and process a document through GraphRAG pipeline."""
    check_graphrag_enabled()

    try:
        file_content = await file.read()

        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/ingest-documents",
            files={"file": (file.filename, file_content, file.content_type or "application/octet-stream")},
            data={
                "domain": domain,
                "use_semantic_chunking": str(use_semantic_chunking).lower(),
                "build_knowledge_graph": str(build_knowledge_graph).lower(),
            },
            timeout=300.0,
        )

        if response.status_code == 200:
            result = response.json()
            return {
                "status": "success",
                "filename": file.filename,
                **result,
            }
        else:
            error_detail = response.text
            try:
                error_detail = response.json().get("detail", error_detail)
            except Exception:
                pass
            raise HTTPException(
                status_code=response.status_code,
                detail=f"GraphRAG processing failed: {error_detail}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@router.post("/ingest/batch")
async def graphrag_upload_batch(
    files: List[UploadFile] = File(...),
    domain: str = Form("general"),
    use_semantic_chunking: bool = Form(True),
):
    """Batch upload documents to GraphRAG."""
    check_graphrag_enabled()

    results = []
    errors = []

    client = get_http_client()

    for file in files:
        try:
            file_content = await file.read()

            response = await client.post(
                f"{_graphrag_url()}/ingest-documents",
                files={"file": (file.filename, file_content, file.content_type or "application/octet-stream")},
                data={
                    "domain": domain,
                    "use_semantic_chunking": str(use_semantic_chunking).lower(),
                },
                timeout=300.0,
            )

            if response.status_code == 200:
                result = response.json()
                results.append({"filename": file.filename, "status": "success", **result})
            else:
                errors.append({"filename": file.filename, "error": response.text})
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})

    return {
        "processed": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }


@router.post("/rebuild")
async def graphrag_rebuild_kg(request: Request):
    """Rebuild knowledge graph from existing documents."""
    check_graphrag_enabled()

    try:
        data = await request.json()
    except Exception:
        data = {}

    domain = data.get("domain")

    try:
        client = get_http_client()
        payload = {}
        if domain:
            payload["domain"] = domain

        response = await client.post(
            f"{_graphrag_url()}/rebuild-knowledge-graph",
            json=payload,
            timeout=600.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/entity/link")
async def graphrag_link_entities(request: Request):
    """Link entities to knowledge graph."""
    check_graphrag_enabled()

    data = await request.json()
    entities = data.get("entities", [])

    if not entities:
        raise HTTPException(status_code=400, detail="entities list is required")

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/entity/link",
            json=data,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/entity/disambiguate")
async def graphrag_disambiguate_entities(request: Request):
    """Disambiguate entity mentions."""
    check_graphrag_enabled()

    data = await request.json()
    entity_name = data.get("entity_name")

    if not entity_name:
        raise HTTPException(status_code=400, detail="entity_name is required")

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/entity/disambiguate",
            json=data,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/ingest/hybrid")
async def graphrag_hybrid_process(
    file: UploadFile = File(...),
    domain: str = Form("general"),
):
    """Process file with hybrid routing (code to Code RAG, docs to GraphRAG)."""
    check_graphrag_enabled()

    try:
        file_content = await file.read()

        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/hybrid/process",
            files={"file": (file.filename, file_content, file.content_type or "application/octet-stream")},
            data={"domain": domain},
            timeout=300.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")
    except Exception as e:
        logger.error(f"Hybrid processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
