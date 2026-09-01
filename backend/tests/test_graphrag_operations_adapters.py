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
    return response


def _upload(filename="notes.txt", content=b"hello graph", content_type="text/plain"):
    upload = Mock()
    upload.filename = filename
    upload.content_type = content_type
    upload.read = AsyncMock(return_value=content)
    return upload


@pytest.mark.asyncio
async def test_process_document_uses_upstream_file_field_and_form_bool():
    operations = load_graphrag_route_module("operations")
    post = AsyncMock(return_value=_mock_response({"total_chunks": 2}))
    mock_client = Mock()
    mock_client.post = post

    with (
        patch.object(operations, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(operations, "get_http_client", return_value=mock_client),
    ):
        result = await operations.graphrag_process_document(
            _upload(),
            use_semantic_chunking=False,
        )

    assert result["total_chunks"] == 2
    post.assert_awaited_once()
    assert post.await_args.args[0] == "http://graphrag.test/process-document"
    assert post.await_args.kwargs["files"]["file"][0] == "notes.txt"
    assert post.await_args.kwargs["data"] == {"use_semantic_chunking": "false"}


@pytest.mark.asyncio
async def test_process_document_with_ner_forwards_extract_entities_flag():
    operations = load_graphrag_route_module("operations")
    post = AsyncMock(return_value=_mock_response({"entities_extracted": True}))
    mock_client = Mock()
    mock_client.post = post

    with (
        patch.object(operations, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(operations, "get_http_client", return_value=mock_client),
    ):
        await operations.graphrag_process_document_with_ner(
            _upload("paper.pdf", b"%PDF", "application/pdf"),
            use_semantic_chunking=True,
            extract_entities=False,
        )

    assert post.await_args.args[0] == "http://graphrag.test/process-document-with-ner"
    assert post.await_args.kwargs["files"]["file"][0] == "paper.pdf"
    assert post.await_args.kwargs["data"] == {
        "use_semantic_chunking": "true",
        "extract_entities": "false",
    }


@pytest.mark.asyncio
async def test_vector_store_add_uses_repeated_files_field():
    operations = load_graphrag_route_module("operations")
    post = AsyncMock(return_value=_mock_response({"processed": 2}))
    mock_client = Mock()
    mock_client.post = post

    with (
        patch.object(operations, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(operations, "get_http_client", return_value=mock_client),
    ):
        result = await operations.graphrag_add_to_vector_store([
            _upload("a.txt", b"a"),
            _upload("b.txt", b"b"),
        ])

    assert result["processed"] == 2
    assert post.await_args.args[0] == "http://graphrag.test/add-to-vector-store"
    assert [field for field, _ in post.await_args.kwargs["files"]] == ["files", "files"]
    assert post.await_args.kwargs["files"][0][1][0] == "a.txt"
    assert post.await_args.kwargs["files"][1][1][0] == "b.txt"


@pytest.mark.asyncio
async def test_domain_statistics_proxies_upstream_domain_stats():
    operations = load_graphrag_route_module("operations")
    get = AsyncMock(return_value=_mock_response({"general": {"entities": 4}}))
    mock_client = Mock()
    mock_client.get = get

    with (
        patch.object(operations, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(operations, "get_http_client", return_value=mock_client),
    ):
        result = await operations.graphrag_domain_statistics()

    assert result["general"]["entities"] == 4
    get.assert_awaited_once_with("http://graphrag.test/knowledge-graph/domain-stats")


@pytest.mark.asyncio
async def test_code_route_forwards_project_name_form_field():
    operations = load_graphrag_route_module("operations")
    post = AsyncMock(return_value=_mock_response({"routed": True}))
    mock_client = Mock()
    mock_client.post = post

    with (
        patch.object(operations, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(operations, "get_http_client", return_value=mock_client),
    ):
        result = await operations.graphrag_route_code_file(
            _upload("main.py", b"def main(): pass", "text/x-python"),
            project_name="llama-nexus",
        )

    assert result["routed"] is True
    assert post.await_args.args[0] == "http://graphrag.test/code-detection/route"
    assert post.await_args.kwargs["files"]["file"][0] == "main.py"
    assert post.await_args.kwargs["data"] == {"project_name": "llama-nexus"}


@pytest.mark.asyncio
async def test_enhanced_extraction_diagnostic_posts_form_data():
    operations = load_graphrag_route_module("operations")
    post = AsyncMock(return_value=_mock_response({"success": True}))
    mock_client = Mock()
    mock_client.post = post

    with (
        patch.object(operations, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(operations, "get_http_client", return_value=mock_client),
    ):
        result = await operations.graphrag_test_enhanced_extraction(
            text="GraphRAG augments retrieval.",
            domain="software",
        )

    assert result["success"] is True
    assert post.await_args.args[0] == "http://graphrag.test/test-enhanced-extraction"
    assert post.await_args.kwargs["data"] == {
        "text": "GraphRAG augments retrieval.",
        "domain": "software",
    }


@pytest.mark.asyncio
async def test_clear_all_requires_explicit_confirmation_before_proxying():
    operations = load_graphrag_route_module("operations")

    with pytest.raises(HTTPException) as exc_info:
        await operations.graphrag_clear_all(confirm=False)

    assert exc_info.value.status_code == 400
    assert "confirm=true" in exc_info.value.detail


@pytest.mark.asyncio
async def test_clear_all_forwards_only_when_confirmed():
    operations = load_graphrag_route_module("operations")
    delete = AsyncMock(return_value=_mock_response({"message": "cleared"}))
    mock_client = Mock()
    mock_client.delete = delete

    with (
        patch.object(operations, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(operations, "get_http_client", return_value=mock_client),
    ):
        result = await operations.graphrag_clear_all(confirm=True)

    assert result["message"] == "cleared"
    delete.assert_awaited_once_with("http://graphrag.test/clear-all", timeout=300.0)
