"""
GraphRAG Code-RAG Routes

Typed adapters for GraphRAG's standalone Code-RAG service: codebase ingestion,
upload ingestion, ingestion status, file analysis, statistics, and guarded
index clearing.
"""

from typing import Any
import logging

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from .helpers import check_graphrag_enabled, get_http_client
from modules.graphrag.config import get_graphrag_config

logger = logging.getLogger(__name__)

router = APIRouter()


class CodeRAGIngestionRequest(BaseModel):
    file_paths: list[str] = Field(min_length=1)
    project_name: str = Field(min_length=1)
    language: str | None = None


class CodeRAGAnalysisRequest(BaseModel):
    file_path: str = Field(min_length=1)


def _code_rag_url() -> str:
    check_graphrag_enabled()
    cfg = get_graphrag_config()
    if not cfg.code_rag_url:
        raise HTTPException(status_code=503, detail="CODE_RAG_URL is not configured")
    return cfg.code_rag_url.rstrip("/")


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
        filename or "code",
        content,
        content_type or "application/octet-stream",
    )


@router.post("/code/ingest")
async def graphrag_code_ingest(request: CodeRAGIngestionRequest):
    """Ingest code files by path into the configured Code-RAG index."""
    try:
        client = get_http_client()
        response = await client.post(
            f"{_code_rag_url()}/ingest",
            json=request.model_dump(mode="json"),
            timeout=300.0,
        )
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"Code-RAG ingestion error: {e}")
        raise HTTPException(status_code=502, detail=f"Code-RAG service error: {str(e)}")


@router.post("/code/upload")
async def graphrag_code_upload(
    files: list[UploadFile] = File(...),
    project_name: str = Form("uploaded_project"),
):
    """Upload code files and ingest them into the configured Code-RAG index."""
    try:
        upstream_files = []
        for file in files:
            upstream_files.append((
                "files",
                _file_tuple(file.filename, await file.read(), file.content_type),
            ))

        client = get_http_client()
        response = await client.post(
            f"{_code_rag_url()}/upload",
            params={"project_name": project_name},
            files=upstream_files,
            timeout=300.0,
        )
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"Code-RAG upload error: {e}")
        raise HTTPException(status_code=502, detail=f"Code-RAG service error: {str(e)}")


@router.get("/code/ingest/{task_id}")
async def graphrag_code_ingestion_status(task_id: str):
    """Get status for a Code-RAG ingestion task."""
    try:
        client = get_http_client()
        response = await client.get(f"{_code_rag_url()}/ingest/{task_id}", timeout=60.0)
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"Code-RAG ingestion status error: {e}")
        raise HTTPException(status_code=502, detail=f"Code-RAG service error: {str(e)}")


@router.post("/code/analyze")
async def graphrag_code_analyze(request: CodeRAGAnalysisRequest):
    """Analyze a single code file path with Code-RAG parsers."""
    try:
        client = get_http_client()
        response = await client.post(
            f"{_code_rag_url()}/analyze",
            params={"file_path": request.file_path},
            timeout=120.0,
        )
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"Code-RAG analysis error: {e}")
        raise HTTPException(status_code=502, detail=f"Code-RAG service error: {str(e)}")


@router.get("/code/statistics")
async def graphrag_code_statistics():
    """Get Code-RAG search/index statistics."""
    cfg = get_graphrag_config()
    if not cfg.enabled:
        return {"available": False, "status": "disabled", "message": "GraphRAG service is disabled. Set GRAPHRAG_ENABLED=true"}
    if not cfg.code_rag_url:
        return {"available": False, "status": "not_configured", "message": "Set CODE_RAG_URL to enable Code-RAG statistics"}

    try:
        client = get_http_client()
        response = await client.get(f"{cfg.code_rag_url.rstrip('/')}/statistics", timeout=60.0)
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"Code-RAG statistics error: {e}")
        raise HTTPException(status_code=502, detail=f"Code-RAG service error: {str(e)}")


@router.delete("/code/index")
async def graphrag_code_clear_index(confirm: bool = Query(False)):
    """Clear the Code-RAG index after explicit confirmation."""
    if not confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to clear the Code-RAG index")

    try:
        client = get_http_client()
        response = await client.delete(f"{_code_rag_url()}/index", timeout=300.0)
        return await _response_payload(response)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"Code-RAG clear index error: {e}")
        raise HTTPException(status_code=502, detail=f"Code-RAG service error: {str(e)}")
