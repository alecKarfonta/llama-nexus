import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, Mock, patch

import pytest


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


@pytest.mark.asyncio
async def test_ner_extract_forwards_raw_text_body():
    extraction = load_graphrag_route_module("extraction")
    request = AsyncMock(return_value=_mock_response({"entity_count": 1}))
    mock_client = Mock()
    mock_client.request = request

    with (
        patch.object(extraction, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(extraction, "get_http_client", return_value=mock_client),
    ):
        result = await extraction.graphrag_ner_extract(
            extraction.NERExtractionRequest(text="OpenAI built GPT.")
        )

    assert result["entity_count"] == 1
    request.assert_awaited_once()
    assert request.await_args.args[:2] == (
        "POST",
        "http://graphrag.test/ner/extract",
    )
    assert request.await_args.kwargs["json"] == "OpenAI built GPT."


@pytest.mark.asyncio
async def test_ner_extract_prefers_configured_ner_service(monkeypatch):
    extraction = load_graphrag_route_module("extraction")
    monkeypatch.setenv("NER_API_URL", "http://ner.test")
    from modules.graphrag.config import reset_config

    reset_config()
    request = AsyncMock(return_value=_mock_response({
        "entities": [{"word": "OpenAI", "entity": "ORG", "score": 0.97}],
        "text": "OpenAI built GPT.",
    }))
    mock_client = Mock()
    mock_client.request = request

    with patch.object(extraction, "get_http_client", return_value=mock_client):
        result = await extraction.graphrag_ner_extract(
            extraction.NERExtractionRequest(text="OpenAI built GPT.")
        )

    assert result["entity_count"] == 1
    assert result["entities"][0]["name"] == "OpenAI"
    assert result["entities"][0]["type"] == "ORG"
    assert result["entities"][0]["confidence"] == 0.97
    assert request.await_args.args[:2] == (
        "POST",
        "http://ner.test/ner",
    )
    assert request.await_args.kwargs["json"] == {
        "text": "OpenAI built GPT.",
        "return_offsets": True,
        "return_scores": True,
    }


@pytest.mark.asyncio
async def test_ner_batch_forwards_raw_text_list_body():
    extraction = load_graphrag_route_module("extraction")
    request = AsyncMock(return_value=_mock_response({"total_texts": 2}))
    mock_client = Mock()
    mock_client.request = request

    with (
        patch.object(extraction, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(extraction, "get_http_client", return_value=mock_client),
    ):
        result = await extraction.graphrag_ner_extract_batch(
            extraction.NERBatchExtractionRequest(texts=["first", "second"])
        )

    assert result["total_texts"] == 2
    assert request.await_args.args[:2] == (
        "POST",
        "http://graphrag.test/ner/extract-batch",
    )
    assert request.await_args.kwargs["json"] == ["first", "second"]


@pytest.mark.asyncio
async def test_ner_batch_prefers_configured_ner_service(monkeypatch):
    extraction = load_graphrag_route_module("extraction")
    monkeypatch.setenv("NER_API_URL", "http://ner.test")
    from modules.graphrag.config import reset_config

    reset_config()
    request = AsyncMock(return_value=_mock_response({
        "results": [
            {"text": "first", "entities": [{"word": "Ada", "entity": "PERSON", "score": 0.9}]},
            {"text": "second", "entities": []},
        ],
    }))
    mock_client = Mock()
    mock_client.request = request

    with patch.object(extraction, "get_http_client", return_value=mock_client):
        result = await extraction.graphrag_ner_extract_batch(
            extraction.NERBatchExtractionRequest(texts=["first", "second"])
        )

    assert result["total_texts"] == 2
    assert result["results"][0]["entity_count"] == 1
    assert result["results"][0]["entities"][0]["name"] == "Ada"
    assert result["results"][0]["entities"][0]["type"] == "PERSON"
    assert result["results"][0]["entities"][0]["confidence"] == 0.9
    assert result["results"][1]["entity_count"] == 0
    assert request.await_args.args[:2] == (
        "POST",
        "http://ner.test/ner/batch",
    )
    assert request.await_args.kwargs["json"] == {
        "texts": ["first", "second"],
        "return_offsets": True,
        "return_scores": True,
    }


@pytest.mark.asyncio
async def test_ner_entities_prefers_configured_ner_service(monkeypatch):
    extraction = load_graphrag_route_module("extraction")
    monkeypatch.setenv("NER_API_URL", "http://ner.test")
    from modules.graphrag.config import reset_config

    reset_config()
    request = AsyncMock(return_value=_mock_response({"entity_types": ["PER", "ORG"]}))
    mock_client = Mock()
    mock_client.request = request

    with patch.object(extraction, "get_http_client", return_value=mock_client):
        result = await extraction.graphrag_ner_entities()

    assert result["entity_types"] == ["PER", "ORG"]
    assert request.await_args.args[:2] == (
        "GET",
        "http://ner.test/ner/entities",
    )


@pytest.mark.asyncio
async def test_rel_extract_splits_body_and_threshold_param():
    extraction = load_graphrag_route_module("extraction")
    request = AsyncMock(return_value=_mock_response({"relation_count": 1}))
    mock_client = Mock()
    mock_client.request = request

    body = extraction.RelationExtractionRequest(
        text="Ada worked for Example Labs.",
        relations=[{"relation": "works for"}],
        entity_labels=["person", "organisation"],
        threshold=0.72,
    )

    with (
        patch.object(extraction, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(extraction, "get_http_client", return_value=mock_client),
    ):
        result = await extraction.graphrag_rel_extract(body)

    assert result["relation_count"] == 1
    assert request.await_args.args[:2] == (
        "POST",
        "http://graphrag.test/rel/extract",
    )
    assert request.await_args.kwargs["params"] == {"threshold": 0.72}
    assert request.await_args.kwargs["json"] == {
        "text": "Ada worked for Example Labs.",
        "relations": [{"relation": "works for"}],
        "entity_labels": ["person", "organisation"],
    }


@pytest.mark.asyncio
async def test_rel_extract_prefers_configured_rel_service(monkeypatch):
    extraction = load_graphrag_route_module("extraction")
    monkeypatch.setenv("REL_API_URL", "http://rel.test")
    from modules.graphrag.config import reset_config

    reset_config()
    request = AsyncMock(return_value=_mock_response({"relations": [{"source": "Ada"}], "text": "Ada worked for Labs."}))
    mock_client = Mock()
    mock_client.request = request

    body = extraction.RelationExtractionRequest(
        text="Ada worked for Labs.",
        relations=[{"relation": "works for"}],
        entity_labels=["person", "organisation"],
        threshold=0.72,
    )

    with patch.object(extraction, "get_http_client", return_value=mock_client):
        result = await extraction.graphrag_rel_extract(body)

    assert result["relation_count"] == 1
    assert result["relationship_count"] == 1
    assert request.await_args.args[:2] == (
        "POST",
        "http://rel.test/extract-relations",
    )
    assert request.await_args.kwargs["json"] == {
        "text": "Ada worked for Labs.",
        "relations": [{"relation": "works for"}],
        "entity_labels": ["person", "organisation"],
        "threshold": 0.72,
    }


@pytest.mark.asyncio
async def test_rel_extract_batch_prefers_configured_rel_service(monkeypatch):
    extraction = load_graphrag_route_module("extraction")
    monkeypatch.setenv("REL_API_URL", "http://rel.test")
    from modules.graphrag.config import reset_config

    reset_config()
    request = AsyncMock(return_value=_mock_response({"results": [{"relation_count": 1}]}))
    mock_client = Mock()
    mock_client.request = request

    body = extraction.RelationBatchExtractionRequest(
        texts=["Ada worked for Labs.", "Grace built compilers."],
        relations=[{"relation": "works for"}],
        entity_labels=["person", "organisation"],
        threshold=0.72,
    )

    with patch.object(extraction, "get_http_client", return_value=mock_client):
        result = await extraction.graphrag_rel_extract_batch(body)

    assert result["results"][0]["relation_count"] == 1
    assert result["total_texts"] == 1
    assert request.await_args.args[:2] == (
        "POST",
        "http://rel.test/extract-relations/batch",
    )
    assert request.await_args.kwargs["json"] == {
        "texts": ["Ada worked for Labs.", "Grace built compilers."],
        "relations": [{"relation": "works for"}],
        "entity_labels": ["person", "organisation"],
        "threshold": 0.72,
    }


@pytest.mark.asyncio
async def test_rel_extract_batch_falls_back_to_main_service_loop():
    extraction = load_graphrag_route_module("extraction")
    request = AsyncMock(side_effect=[
        _mock_response({"text": "Ada worked for Labs.", "relations": [{"source": "Ada"}], "relation_count": 1}),
        _mock_response({"text": "Grace built compilers.", "relations": [], "relation_count": 0}),
    ])
    mock_client = Mock()
    mock_client.request = request

    body = extraction.RelationBatchExtractionRequest(
        texts=["Ada worked for Labs.", "Grace built compilers."],
        relations=[{"relation": "works for"}],
        entity_labels=["person", "organisation"],
        threshold=0.72,
    )

    with (
        patch.object(extraction, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(extraction, "get_http_client", return_value=mock_client),
    ):
        result = await extraction.graphrag_rel_extract_batch(body)

    assert result["total_texts"] == 2
    assert result["batch_mode"] == "llama_nexus_fallback"
    assert request.await_count == 2
    assert request.await_args_list[0].args[:2] == (
        "POST",
        "http://graphrag.test/rel/extract",
    )
    assert request.await_args_list[0].kwargs["params"] == {"threshold": 0.72}
    assert request.await_args_list[0].kwargs["json"] == {
        "text": "Ada worked for Labs.",
        "relations": [{"relation": "works for"}],
        "entity_labels": ["person", "organisation"],
    }


@pytest.mark.asyncio
async def test_enhanced_extraction_posts_form_toggles_to_upstream():
    extraction = load_graphrag_route_module("extraction")
    request = AsyncMock(return_value=_mock_response({"entity_count": 2, "relationship_count": 1}))
    mock_client = Mock()
    mock_client.request = request

    body = extraction.EnhancedExtractionRequest(
        text="GraphRAG augments retrieval.",
        domain="software",
        use_spanbert=False,
        use_dependency=True,
        use_entity_linking=False,
    )

    with (
        patch.object(extraction, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(extraction, "get_http_client", return_value=mock_client),
    ):
        result = await extraction.graphrag_enhanced_extraction(body)

    assert result["entity_count"] == 2
    assert request.await_args.args[:2] == (
        "POST",
        "http://graphrag.test/extract-entities-relations-enhanced",
    )
    assert request.await_args.kwargs["data"] == {
        "text": "GraphRAG augments retrieval.",
        "domain": "software",
        "use_spanbert": "false",
        "use_dependency": "true",
        "use_entity_linking": "false",
    }


@pytest.mark.asyncio
async def test_enhanced_query_forwards_advanced_search_shape():
    extraction = load_graphrag_route_module("extraction")
    request = AsyncMock(return_value=_mock_response({"answer": "reasoned"}))
    mock_client = Mock()
    mock_client.request = request

    body = extraction.EnhancedQueryRequest(
        query="Find causes for outages",
        search_type="hybrid",
        top_k=5,
        domain="ops",
        filters={"severity": "high"},
    )

    with (
        patch.object(extraction, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(extraction, "get_http_client", return_value=mock_client),
    ):
        result = await extraction.graphrag_enhanced_query(body)

    assert result["answer"] == "reasoned"
    assert request.await_args.args[:2] == (
        "POST",
        "http://graphrag.test/query/enhanced",
    )
    assert request.await_args.kwargs["json"] == {
        "query": "Find causes for outages",
        "search_type": "hybrid",
        "top_k": 5,
        "domain": "ops",
        "filters": {"severity": "high"},
    }


@pytest.mark.asyncio
async def test_enhanced_answer_forwards_query_param_and_context_body():
    extraction = load_graphrag_route_module("extraction")
    request = AsyncMock(return_value=_mock_response({"follow_up_suggestions": []}))
    mock_client = Mock()
    mock_client.request = request

    with (
        patch.object(extraction, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(extraction, "get_http_client", return_value=mock_client),
    ):
        await extraction.graphrag_enhanced_answer(
            extraction.EnhancedAnswerRequest(
                query="Summarize the graph",
                context={"domain": "docs"},
            )
        )

    assert request.await_args.args[:2] == (
        "POST",
        "http://graphrag.test/api/enhanced-query",
    )
    assert request.await_args.kwargs["params"] == {"query": "Summarize the graph"}
    assert request.await_args.kwargs["json"] == {"domain": "docs"}


@pytest.mark.asyncio
async def test_statistics_routes_proxy_to_graphrag():
    extraction = load_graphrag_route_module("extraction")
    request = AsyncMock(side_effect=[
        _mock_response({"query_statistics": {"total": 3}}),
        _mock_response({"graph_statistics": {"nodes": 4}}),
    ])
    mock_client = Mock()
    mock_client.request = request

    with (
        patch.object(extraction, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(extraction, "get_http_client", return_value=mock_client),
    ):
        query_stats = await extraction.graphrag_query_statistics()
        graph_stats = await extraction.graphrag_reasoning_graph_statistics()

    assert query_stats["query_statistics"]["total"] == 3
    assert graph_stats["graph_statistics"]["nodes"] == 4
    assert request.await_args_list[0].args[:2] == (
        "GET",
        "http://graphrag.test/query/statistics",
    )
    assert request.await_args_list[1].args[:2] == (
        "GET",
        "http://graphrag.test/reasoning/graph-statistics",
    )
