"""
GraphRAG Operations Routes

Typed adapters for GraphRAG operational endpoints: document processing
previews, vector-store-only ingestion, domain statistics, Code-RAG routing,
enhanced-extraction diagnostics, and guarded administrative clearing.
"""

from typing import Any
import logging

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from .helpers import check_graphrag_enabled, get_http_client, graphrag_url as _graphrag_url
from modules.graphrag.config import get_graphrag_config

logger = logging.getLogger(__name__)

router = APIRouter()


def _json_or_text(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


async def _response_payload(response: httpx.Response) -> Any:
    payload = _json_or_text(response)
    if response.status_code >= 400:
        status_code = response.status_code if response.status_code < 500 else 502
        raise HTTPException(status_code=status_code, detail=payload)
    return payload


def _file_tuple(filename: str | None, content: bytes, content_type: str | None):
    return (
        filename or "document",
        content,
        content_type or "application/octet-stream",
    )


@router.post("/process-document")
async def graphrag_process_document(
    file: UploadFile = File(...),
    use_semantic_chunking: bool = Form(True),
):
    """Process one document into chunks without ingesting it into the graph."""
    check_graphrag_enabled()

    try:
        file_content = await file.read()
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/process-document",
            files={"file": _file_tuple(file.filename, file_content, file.content_type)},
            data={"use_semantic_chunking": str(use_semantic_chunking).lower()},
            timeout=300.0,
        )
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG document processing error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/process-documents-batch")
async def graphrag_process_documents_batch(
    files: list[UploadFile] = File(...),
    use_semantic_chunking: bool = Form(True),
):
    """Process multiple documents into chunks without ingesting them into the graph."""
    check_graphrag_enabled()

    try:
        upstream_files = []
        for file in files:
            upstream_files.append((
                "files",
                _file_tuple(file.filename, await file.read(), file.content_type),
            ))

        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/process-documents-batch",
            files=upstream_files,
            data={"use_semantic_chunking": str(use_semantic_chunking).lower()},
            timeout=300.0,
        )
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG batch document processing error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/process-document-with-ner")
async def graphrag_process_document_with_ner(
    file: UploadFile = File(...),
    use_semantic_chunking: bool = Form(True),
    extract_entities: bool = Form(True),
):
    """Process a document and attach per-chunk GLiNER entities."""
    check_graphrag_enabled()

    try:
        file_content = await file.read()
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/process-document-with-ner",
            files={"file": _file_tuple(file.filename, file_content, file.content_type)},
            data={
                "use_semantic_chunking": str(use_semantic_chunking).lower(),
                "extract_entities": str(extract_entities).lower(),
            },
            timeout=300.0,
        )
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG NER document processing error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/vector-store/add")
async def graphrag_add_to_vector_store(files: list[UploadFile] = File(...)):
    """Add documents to GraphRAG's vector store without knowledge-graph building."""
    check_graphrag_enabled()

    try:
        upstream_files = []
        for file in files:
            upstream_files.append((
                "files",
                _file_tuple(file.filename, await file.read(), file.content_type),
            ))

        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/add-to-vector-store",
            files=upstream_files,
            timeout=300.0,
        )
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG vector-store add error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/domain-stats")
@router.get("/domains/stats")
async def graphrag_domain_statistics():
    """Get per-domain knowledge graph statistics."""
    check_graphrag_enabled()
    cfg = get_graphrag_config()
    if not cfg.graphrag_url:
        return {
            "available": False,
            "status": "not_configured",
            "message": "Set GRAPHRAG_URL to enable domain statistics",
            "domains": [],
        }

    try:
        client = get_http_client()
        response = await client.get(f"{cfg.graphrag_url}/knowledge-graph/domain-stats", timeout=60.0)
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG domain statistics error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/code/route")
async def graphrag_route_code_file(
    file: UploadFile = File(...),
    project_name: str | None = Form(None),
):
    """Route a code file to Code-RAG for specialized processing."""
    check_graphrag_enabled()

    try:
        file_content = await file.read()
        data = {}
        if project_name:
            data["project_name"] = project_name

        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/code-detection/route",
            files={"file": _file_tuple(file.filename, file_content, file.content_type)},
            data=data,
            timeout=300.0,
        )
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG Code-RAG route error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/extraction/test-enhanced")
async def graphrag_test_enhanced_extraction(
    text: str = Form(...),
    domain: str = Form("general"),
):
    """Run GraphRAG's enhanced-extraction diagnostic endpoint."""
    check_graphrag_enabled()

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/test-enhanced-extraction",
            data={"text": text, "domain": domain},
            timeout=120.0,
        )
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG enhanced extraction diagnostic error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.delete("/admin/clear-all")
async def graphrag_clear_all(confirm: bool = Query(False)):
    """Clear GraphRAG vector-store and knowledge-graph data after explicit confirmation."""
    if not confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to clear all GraphRAG data")

    check_graphrag_enabled()

    try:
        client = get_http_client()
        response = await client.delete(f"{_graphrag_url()}/clear-all", timeout=300.0)
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG clear-all error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")
