import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.exceptions import HTTPException


GRAPHRAG_ROUTES_DIR = Path(__file__).resolve().parents[1] / "routes" / "graphrag"
ROUTES_DIR = GRAPHRAG_ROUTES_DIR.parent


def load_graphrag_route_module(module_name: str):
    routes_pkg = sys.modules.get("routes")
    if routes_pkg is None:
        routes_pkg = ModuleType("routes")
        routes_pkg.__path__ = [str(ROUTES_DIR)]
        sys.modules["routes"] = routes_pkg

    graphrag_pkg = sys.modules.get("routes.graphrag")
    if graphrag_pkg is None:
        graphrag_pkg = ModuleType("routes.graphrag")
        graphrag_pkg.__path__ = [str(GRAPHRAG_ROUTES_DIR)]
        sys.modules["routes.graphrag"] = graphrag_pkg

    qualified_name = f"routes.graphrag.{module_name}"
    spec = importlib.util.spec_from_file_location(
        qualified_name,
        GRAPHRAG_ROUTES_DIR / f"{module_name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def graphrag_enabled(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_ENABLED", "true")
    monkeypatch.setenv("GRAPHRAG_URL", "http://graphrag.test")
    from modules.graphrag.config import reset_config

    reset_config()
    yield
    reset_config()


def _mock_response(payload, status_code=200):
    response = Mock()
    response.json.return_value = payload
    response.text = str(payload)
    response.status_code = status_code
    response.raise_for_status = Mock()
    return response


@pytest.mark.asyncio
async def test_evaluation_status_combines_retrieval_and_chunking_status():
    evaluation = load_graphrag_route_module("evaluation")
    request = AsyncMock(side_effect=[
        _mock_response({"evaluation_system": "GutenQA"}),
        _mock_response({"evaluation_system": "WikiSection"}),
    ])
    mock_client = Mock()
    mock_client.request = request

    with (
        patch.object(evaluation, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(evaluation, "get_http_client", return_value=mock_client),
    ):
        result = await evaluation.graphrag_evaluation_status()

    assert request.await_count == 2
    assert request.await_args_list[0].args[:2] == (
        "GET",
        "http://graphrag.test/evaluate/retrieval/status",
    )
    assert request.await_args_list[1].args[:2] == (
        "GET",
        "http://graphrag.test/evaluate/chunking/status",
    )
    assert result["available"] is True
    assert result["retrieval"]["evaluation_system"] == "GutenQA"
    assert result["chunking"]["evaluation_system"] == "WikiSection"


@pytest.mark.asyncio
async def test_entity_extraction_evaluation_forwards_json_body():
    evaluation = load_graphrag_route_module("evaluation")
    request = AsyncMock(return_value=_mock_response({"precision": 0.91}))
    mock_client = Mock()
    mock_client.request = request

    body = evaluation.EntityExtractionEvaluationRequest(
        test_documents=["OpenAI built GPT."],
        ground_truth={"OpenAI built GPT.": [{"name": "OpenAI", "type": "organization"}]},
    )

    with (
        patch.object(evaluation, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(evaluation, "get_http_client", return_value=mock_client),
    ):
        result = await evaluation.graphrag_evaluate_entity_extraction(body)

    assert result["precision"] == 0.91
    request.assert_awaited_once()
    method, url = request.await_args.args[:2]
    assert method == "POST"
    assert url == "http://graphrag.test/evaluate/entity-extraction"
    assert request.await_args.kwargs["json"] == {
        "test_documents": ["OpenAI built GPT."],
        "ground_truth": {"OpenAI built GPT.": [{"name": "OpenAI", "type": "organization"}]},
    }


@pytest.mark.asyncio
async def test_chunking_evaluation_forwards_controls_as_params():
    evaluation = load_graphrag_route_module("evaluation")
    request = AsyncMock(return_value=_mock_response({"evaluation_type": "WikiSection"}))
    mock_client = Mock()
    mock_client.request = request

    body = evaluation.ChunkingEvaluationRequest(
        subset="en_city",
        sample_size=12,
        force_download=True,
    )

    with (
        patch.object(evaluation, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(evaluation, "get_http_client", return_value=mock_client),
    ):
        await evaluation.graphrag_evaluate_chunking(body)

    request.assert_awaited_once()
    assert request.await_args.args[:2] == (
        "POST",
        "http://graphrag.test/evaluate/chunking",
    )
    assert request.await_args.kwargs["params"] == {
        "dataset": "wikisection",
        "subset": "en_city",
        "sample_size": 12,
        "force_download": True,
    }
    assert request.await_args.kwargs["json"] is None


@pytest.mark.asyncio
async def test_test_suite_adapter_validates_suite_names_before_proxying():
    evaluation = load_graphrag_route_module("evaluation")

    with pytest.raises(HTTPException) as exc_info:
        await evaluation.graphrag_run_test_suite("smoke")

    assert exc_info.value.status_code == 400
    assert "unit" in exc_info.value.detail


@pytest.mark.asyncio
async def test_test_suite_adapter_forwards_allowlisted_suite():
    evaluation = load_graphrag_route_module("evaluation")
    request = AsyncMock(return_value=_mock_response({"passed": True}))
    mock_client = Mock()
    mock_client.request = request

    with (
        patch.object(evaluation, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(evaluation, "get_http_client", return_value=mock_client),
    ):
        result = await evaluation.graphrag_run_test_suite("quality")

    assert result["passed"] is True
    request.assert_awaited_once()
    assert request.await_args.args[:2] == (
        "POST",
        "http://graphrag.test/test/run-quality-tests",
    )
