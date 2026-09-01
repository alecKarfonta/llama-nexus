import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import Request


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
    response.status_code = status_code
    response.raise_for_status = Mock()
    return response


@pytest.mark.asyncio
async def test_supported_formats_fallback_matches_local_pipeline_formats(monkeypatch):
    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    from modules.graphrag.config import reset_config

    reset_config()
    health = load_graphrag_route_module("health")

    result = await health.graphrag_supported_formats()

    assert result["available"] is False
    assert set(result["supported_formats"]) >= {
        "txt",
        "md",
        "html",
        "csv",
        "json",
        "pdf",
        "docx",
        "pptx",
        "xlsx",
        "rtf",
        "epub",
        "odt",
        "ods",
        "odp",
        "zip",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "bmp",
        "tif",
        "tiff",
        "webp",
        "svg",
    }
    assert result["features"]["structure_preservation"] is True
    assert result["features"]["archive_extraction"] is True
    assert result["features"]["embedded_attachment_extraction"] is True
    assert result["features"]["nested_document_extraction"] is True
    assert result["features"]["image_metadata_extraction"] is True
    assert result["features"]["ocr_extraction"] is True
    assert result["features"]["ocr_extraction_enabled"] is False
    assert result["features"]["ocr_extraction_ready"] is False
    assert "document_ocr" in result
    assert result["document_ocr"]["enabled"] is False
    assert result["document_ocr"]["status"] == "disabled"
    assert result["document_ocr"]["required_env"] == ["GRAPHRAG_ENABLE_IMAGE_OCR"]
    assert result["features"]["pipeline_evidence_profile"] is True


@pytest.mark.asyncio
async def test_graph_stats_forwards_optional_domain_filter():
    health = load_graphrag_route_module("health")
    get = AsyncMock(return_value=_mock_response({"nodes": 4, "edges": 3, "domain": "ops"}))
    mock_client = Mock()
    mock_client.get = get

    with patch.object(health, "get_http_client", return_value=mock_client):
        result = await health.graphrag_get_stats(domain="ops")

    assert result["domain"] == "ops"
    get.assert_awaited_once_with(
        "http://graphrag.test/knowledge-graph/stats",
        params={"domain": "ops"},
    )


@pytest.mark.asyncio
async def test_top_entities_forwards_entity_type_filter():
    graph = load_graphrag_route_module("graph")
    get = AsyncMock(return_value=_mock_response({"top_entities": [{"name": "OpenAI"}]}))
    mock_client = Mock()
    mock_client.get = get

    with patch.object(graph, "get_http_client", return_value=mock_client):
        result = await graph.graphrag_top_entities(
            domain="research",
            limit=10,
            min_occurrence=2,
            entity_type="organisation",
        )

    assert result["top_entities"][0]["name"] == "OpenAI"
    get.assert_awaited_once_with(
        "http://graphrag.test/knowledge-graph/top-entities",
        params={
            "limit": 10,
            "min_occurrence": 2,
            "domain": "research",
            "entity_type": "organisation",
        },
    )


@pytest.mark.asyncio
async def test_top_relationships_forwards_relationship_type_filter():
    graph = load_graphrag_route_module("graph")
    get = AsyncMock(return_value=_mock_response({"top_relationships": [{"type": "uses"}]}))
    mock_client = Mock()
    mock_client.get = get

    with patch.object(graph, "get_http_client", return_value=mock_client):
        result = await graph.graphrag_top_relationships(
            domain="software",
            limit=7,
            min_weight=3,
            relationship_type="uses",
        )

    assert result["top_relationships"][0]["type"] == "uses"
    get.assert_awaited_once_with(
        "http://graphrag.test/knowledge-graph/top-relationships",
        params={
            "limit": 7,
            "min_weight": 3,
            "domain": "software",
            "relationship_type": "uses",
        },
    )


@pytest.mark.asyncio
async def test_entity_neighborhood_uses_graph_helper_with_domain_filter():
    graph = load_graphrag_route_module("graph")
    request = Mock(spec=Request)
    request.json = AsyncMock(return_value={
        "entity_name": "GraphRAG",
        "domain": "research",
        "limit": 12,
    })
    mock_client = Mock()
    fetch_graph_data = AsyncMock(return_value=({"GraphRAG": {"id": "GraphRAG"}}, []))
    find_neighbors = Mock(return_value={
        "entity": "GraphRAG",
        "nodes": [{"id": "GraphRAG"}],
        "edges": [],
    })

    with (
        patch.object(graph, "get_http_client", return_value=mock_client),
        patch.object(graph, "_fetch_graph_data", fetch_graph_data),
        patch.object(graph, "_find_entity_neighbors", find_neighbors),
    ):
        result = await graph.graphrag_entity_neighborhood(request)

    assert result["entity"] == "GraphRAG"
    fetch_graph_data.assert_awaited_once_with(mock_client, "research")
    find_neighbors.assert_called_once_with("GraphRAG", {"GraphRAG": {"id": "GraphRAG"}}, [], 12)


@pytest.mark.asyncio
async def test_communities_forwards_domain_filter_to_domain_stats():
    graph = load_graphrag_route_module("graph")
    get = AsyncMock(return_value=_mock_response({
        "domain_stats": {"research": {"communities": 3}},
        "communities": [{"id": "community-1"}],
    }))
    mock_client = Mock()
    mock_client.get = get

    with (
        patch.object(graph, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(graph, "get_http_client", return_value=mock_client),
    ):
        result = await graph.graphrag_get_communities(domain="research")

    assert result["communities"][0]["id"] == "community-1"
    get.assert_awaited_once_with(
        "http://graphrag.test/knowledge-graph/domain-stats",
        params={"domain": "research"},
    )


@pytest.mark.asyncio
async def test_graph_export_forwards_full_filter_params():
    graph = load_graphrag_route_module("graph")
    get = AsyncMock(return_value=_mock_response({"entities": [], "relationships": []}))
    mock_client = Mock()
    mock_client.get = get
    request = Mock(spec=Request)
    request.json = AsyncMock(return_value={
        "format": "json",
        "domain": "docs",
        "max_entities": 25,
        "max_relationships": 40,
        "min_occurrence": 2,
    })

    with (
        patch.object(graph, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(graph, "get_http_client", return_value=mock_client),
    ):
        await graph.graphrag_export_graph(request)

    get.assert_awaited_once_with(
        "http://graphrag.test/knowledge-graph/export",
        params={
            "format": "json",
            "domain": "docs",
            "max_entities": 25,
            "max_relationships": 40,
            "min_occurrence": 2,
        },
    )
