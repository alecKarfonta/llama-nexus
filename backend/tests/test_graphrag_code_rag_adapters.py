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
    monkeypatch.setenv("CODE_RAG_URL", "http://code-rag.test")
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


def _upload(filename="main.py", content=b"def main(): pass", content_type="text/x-python"):
    upload = Mock()
    upload.filename = filename
    upload.content_type = content_type
    upload.read = AsyncMock(return_value=content)
    return upload


@pytest.mark.asyncio
async def test_code_rag_ingest_forwards_file_paths_to_direct_service():
    code = load_graphrag_route_module("code")
    post = AsyncMock(return_value=_mock_response({"task_id": "task-1", "status": "started"}))
    mock_client = Mock()
    mock_client.post = post

    with patch.object(code, "get_http_client", return_value=mock_client):
        result = await code.graphrag_code_ingest(
            code.CodeRAGIngestionRequest(
                file_paths=["/repo/a.py", "/repo/b.py"],
                project_name="llama-nexus",
                language="python",
            )
        )

    assert result["task_id"] == "task-1"
    post.assert_awaited_once_with(
        "http://code-rag.test/ingest",
        json={
            "file_paths": ["/repo/a.py", "/repo/b.py"],
            "project_name": "llama-nexus",
            "language": "python",
        },
        timeout=300.0,
    )


@pytest.mark.asyncio
async def test_code_rag_upload_uses_repeated_files_and_project_query_param():
    code = load_graphrag_route_module("code")
    post = AsyncMock(return_value=_mock_response({"task_id": "task-2"}))
    mock_client = Mock()
    mock_client.post = post

    with patch.object(code, "get_http_client", return_value=mock_client):
        result = await code.graphrag_code_upload(
            [_upload("a.py", b"a"), _upload("b.py", b"b")],
            project_name="demo",
        )

    assert result["task_id"] == "task-2"
    assert post.await_args.args[0] == "http://code-rag.test/upload"
    assert post.await_args.kwargs["params"] == {"project_name": "demo"}
    assert [field for field, _ in post.await_args.kwargs["files"]] == ["files", "files"]
    assert post.await_args.kwargs["files"][0][1][0] == "a.py"
    assert post.await_args.kwargs["files"][1][1][0] == "b.py"


@pytest.mark.asyncio
async def test_code_rag_ingestion_status_forwards_task_id():
    code = load_graphrag_route_module("code")
    get = AsyncMock(return_value=_mock_response({"task_id": "task-1", "progress": 50}))
    mock_client = Mock()
    mock_client.get = get

    with patch.object(code, "get_http_client", return_value=mock_client):
        result = await code.graphrag_code_ingestion_status("task-1")

    assert result["progress"] == 50
    get.assert_awaited_once_with("http://code-rag.test/ingest/task-1", timeout=60.0)


@pytest.mark.asyncio
async def test_code_rag_analyze_uses_file_path_query_param():
    code = load_graphrag_route_module("code")
    post = AsyncMock(return_value=_mock_response({"success": True, "entities": []}))
    mock_client = Mock()
    mock_client.post = post

    with patch.object(code, "get_http_client", return_value=mock_client):
        result = await code.graphrag_code_analyze(
            code.CodeRAGAnalysisRequest(file_path="/repo/main.py")
        )

    assert result["success"] is True
    post.assert_awaited_once_with(
        "http://code-rag.test/analyze",
        params={"file_path": "/repo/main.py"},
        timeout=120.0,
    )


@pytest.mark.asyncio
async def test_code_rag_statistics_forwards_to_direct_service():
    code = load_graphrag_route_module("code")
    get = AsyncMock(return_value=_mock_response({"active_tasks": 1}))
    mock_client = Mock()
    mock_client.get = get

    with patch.object(code, "get_http_client", return_value=mock_client):
        result = await code.graphrag_code_statistics()

    assert result["active_tasks"] == 1
    get.assert_awaited_once_with("http://code-rag.test/statistics", timeout=60.0)


@pytest.mark.asyncio
async def test_code_rag_clear_index_requires_confirmation_before_proxying():
    code = load_graphrag_route_module("code")

    with pytest.raises(HTTPException) as exc_info:
        await code.graphrag_code_clear_index(confirm=False)

    assert exc_info.value.status_code == 400
    assert "confirm=true" in exc_info.value.detail


@pytest.mark.asyncio
async def test_code_rag_clear_index_forwards_only_when_confirmed():
    code = load_graphrag_route_module("code")
    delete = AsyncMock(return_value=_mock_response({"message": "Index cleared successfully"}))
    mock_client = Mock()
    mock_client.delete = delete

    with patch.object(code, "get_http_client", return_value=mock_client):
        result = await code.graphrag_code_clear_index(confirm=True)

    assert result["message"] == "Index cleared successfully"
    delete.assert_awaited_once_with("http://code-rag.test/index", timeout=300.0)


@pytest.mark.asyncio
async def test_code_rag_statistics_reports_missing_direct_code_rag_url(monkeypatch):
    from modules.graphrag.config import reset_config

    code = load_graphrag_route_module("code")
    monkeypatch.delenv("CODE_RAG_URL", raising=False)
    monkeypatch.delenv("RUNNING_IN_DOCKER", raising=False)
    monkeypatch.delenv("USE_DOCKER", raising=False)
    reset_config()

    result = await code.graphrag_code_statistics()

    assert result["available"] is False
    assert result["status"] == "not_configured"
    assert "CODE_RAG_URL" in result["message"]
