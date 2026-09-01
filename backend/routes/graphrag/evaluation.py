"""
GraphRAG Evaluation & Test Suite Routes

Typed llama-nexus adapters for GraphRAG's evaluation framework. These routes
preserve stable llama-nexus API shapes while forwarding work to the configured
GraphRAG service.
"""

from typing import Any, Dict
import logging

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .helpers import check_graphrag_enabled, get_http_client, graphrag_url as _graphrag_url
from modules.graphrag.config import get_graphrag_config

logger = logging.getLogger(__name__)

router = APIRouter()


class EntityExtractionEvaluationRequest(BaseModel):
    test_documents: list[str]
    ground_truth: Dict[str, list[Dict[str, Any]]]


class QueryResponseEvaluationRequest(BaseModel):
    test_queries: list[str]
    expected_answers: list[str]


class RetrievalRelevanceEvaluationRequest(BaseModel):
    test_queries: list[str]
    relevance_scores: list[float]


class ChunkingEvaluationRequest(BaseModel):
    dataset: str = "wikisection"
    subset: str = "en_disease"
    sample_size: int = Field(default=50, ge=1)
    force_download: bool = False


class ChunkingComparativeEvaluationRequest(BaseModel):
    subsets: list[str] = Field(default_factory=lambda: ["en_disease", "en_city"])
    sample_size: int = Field(default=50, ge=1)
    include_baseline: bool = True
    baseline_chunk_size: int = Field(default=500, ge=1)


class RetrievalEvaluationRequest(BaseModel):
    book_name: str = "A_Christmas_Carol_-_Charles_Dickens"
    max_questions: int = Field(default=30, ge=1)
    method: str = "both"
    force_download: bool = False


class RetrievalComparativeEvaluationRequest(BaseModel):
    book_names: list[str] = Field(
        default_factory=lambda: [
            "A_Christmas_Carol_-_Charles_Dickens",
            "Pride_and_Prejudice_-_Jane_Austen",
        ]
    )
    max_questions_per_book: int = Field(default=20, ge=1)
    include_baseline: bool = True


TEST_SUITE_PATHS = {
    "unit": "test/run-unit-tests",
    "integration": "test/run-integration-tests",
    "performance": "test/run-performance-tests",
    "quality": "test/run-quality-tests",
    "comprehensive": "test/run-comprehensive-tests",
}


def _json_or_text(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


async def _call_graphrag(
    method: str,
    path: str,
    *,
    params: Dict[str, Any] | None = None,
    json: Any = None,
    timeout: float = 300.0,
) -> Any:
    check_graphrag_enabled()

    try:
        client = get_http_client()
        response = await client.request(
            method,
            f"{_graphrag_url()}/{path}",
            params=params,
            json=json,
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        logger.error(f"GraphRAG evaluation request failed: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")

    payload = _json_or_text(response)
    if response.status_code >= 400:
        status_code = response.status_code if response.status_code < 500 else 502
        raise HTTPException(status_code=status_code, detail=payload)

    return payload


@router.get("/evaluation/status")
async def graphrag_evaluation_status():
    """Get retrieval and chunking evaluation readiness in one response."""
    cfg = get_graphrag_config()
    if not cfg.enabled:
        return {"available": False, "status": "disabled", "message": "GraphRAG service is disabled. Set GRAPHRAG_ENABLED=true"}
    if not cfg.graphrag_url:
        return {"available": False, "status": "not_configured", "message": "Set GRAPHRAG_URL to enable GraphRAG evaluation"}

    retrieval = await _call_graphrag("GET", "evaluate/retrieval/status", timeout=60.0)
    chunking = await _call_graphrag("GET", "evaluate/chunking/status", timeout=60.0)

    return {
        "retrieval": retrieval,
        "chunking": chunking,
        "available": not bool(retrieval.get("error") or chunking.get("error"))
        if isinstance(retrieval, dict) and isinstance(chunking, dict)
        else True,
    }


@router.get("/evaluation/retrieval/status")
async def graphrag_retrieval_evaluation_status():
    """Get GutenQA retrieval evaluation readiness."""
    return await _call_graphrag("GET", "evaluate/retrieval/status", timeout=60.0)


@router.get("/evaluation/chunking/status")
async def graphrag_chunking_evaluation_status():
    """Get WikiSection chunking evaluation readiness."""
    return await _call_graphrag("GET", "evaluate/chunking/status", timeout=60.0)


@router.post("/evaluation/entity-extraction")
async def graphrag_evaluate_entity_extraction(request: EntityExtractionEvaluationRequest):
    """Evaluate entity extraction against supplied ground truth."""
    return await _call_graphrag(
        "POST",
        "evaluate/entity-extraction",
        json=request.model_dump(),
    )


@router.post("/evaluation/query-responses")
async def graphrag_evaluate_query_responses(request: QueryResponseEvaluationRequest):
    """Evaluate query response quality against expected answers."""
    return await _call_graphrag(
        "POST",
        "evaluate/query-responses",
        json=request.model_dump(),
    )


@router.get("/evaluation/graph-completeness")
async def graphrag_evaluate_graph_completeness():
    """Evaluate knowledge graph completeness."""
    return await _call_graphrag("GET", "evaluate/graph-completeness")


@router.post("/evaluation/retrieval-relevance")
async def graphrag_evaluate_retrieval_relevance(request: RetrievalRelevanceEvaluationRequest):
    """Evaluate retrieval relevance for supplied test queries."""
    return await _call_graphrag(
        "POST",
        "evaluate/retrieval-relevance",
        json=request.model_dump(),
    )


@router.post("/evaluation/comprehensive")
async def graphrag_evaluate_comprehensive(test_data: Dict[str, Any]):
    """Run GraphRAG's comprehensive evaluation over supplied test data."""
    return await _call_graphrag(
        "POST",
        "evaluate/comprehensive",
        json=test_data,
    )


@router.post("/evaluation/chunking")
async def graphrag_evaluate_chunking(request: ChunkingEvaluationRequest):
    """Run WikiSection chunking evaluation."""
    return await _call_graphrag(
        "POST",
        "evaluate/chunking",
        params=request.model_dump(),
        timeout=600.0,
    )


@router.post("/evaluation/chunking/comparative")
async def graphrag_evaluate_chunking_comparative(request: ChunkingComparativeEvaluationRequest):
    """Run comparative WikiSection chunking evaluation."""
    return await _call_graphrag(
        "POST",
        "evaluate/chunking/comparative",
        params=request.model_dump(),
        timeout=600.0,
    )


@router.post("/evaluation/retrieval")
async def graphrag_evaluate_retrieval(request: RetrievalEvaluationRequest):
    """Run GutenQA retrieval evaluation."""
    return await _call_graphrag(
        "POST",
        "evaluate/retrieval",
        params=request.model_dump(),
        timeout=600.0,
    )


@router.post("/evaluation/retrieval/comparative")
async def graphrag_evaluate_retrieval_comparative(request: RetrievalComparativeEvaluationRequest):
    """Run comparative GutenQA retrieval evaluation."""
    return await _call_graphrag(
        "POST",
        "evaluate/retrieval/comparative",
        params=request.model_dump(),
        timeout=600.0,
    )


@router.get("/evaluation/test-suite/reports")
async def graphrag_test_suite_reports():
    """Get generated GraphRAG test suite reports."""
    return await _call_graphrag("GET", "test/reports")


@router.post("/evaluation/test-suite/{suite}")
async def graphrag_run_test_suite(suite: str):
    """Run a GraphRAG test suite by name."""
    suite_path = TEST_SUITE_PATHS.get(suite)
    if not suite_path:
        allowed = ", ".join(sorted(TEST_SUITE_PATHS))
        raise HTTPException(status_code=400, detail=f"Unknown GraphRAG test suite '{suite}'. Use one of: {allowed}")

    return await _call_graphrag("POST", suite_path, timeout=600.0)
