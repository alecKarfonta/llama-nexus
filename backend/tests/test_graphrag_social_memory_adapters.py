import importlib.util
import sys
from datetime import datetime, timezone
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
    response.content = b'{"ok":true}'
    response.status_code = status_code
    response.headers = {"content-type": "application/json"}
    return response


@pytest.mark.asyncio
async def test_social_memory_ingest_forwards_json_message_and_sync_param():
    social_memory = load_graphrag_route_module("social_memory")
    request = AsyncMock(return_value=_mock_response({"queued": False, "facts_written": 2}))
    mock_client = Mock()
    mock_client.request = request

    timestamp = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)

    with (
        patch.object(social_memory, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(social_memory, "get_http_client", return_value=mock_client),
    ):
        result = await social_memory.graphrag_social_memory_ingest(
            social_memory.SocialMemoryIngestMessage(
                platform="twitch",
                platform_user_id="user-1",
                display_name="Ada",
                text="I like graph search.",
                timestamp=timestamp,
                message_id="msg-1",
                previous_messages=["hello"],
                priority=4,
            ),
            sync=True,
        )

    assert result["facts_written"] == 2
    assert request.await_args.args[:2] == (
        "POST",
        "http://graphrag.test/social-memory/ingest",
    )
    assert request.await_args.kwargs["params"] == {"sync": True}
    assert request.await_args.kwargs["json"]["timestamp"] == "2026-07-10T12:30:00Z"
    assert request.await_args.kwargs["json"]["platform"] == "twitch"
    assert request.await_args.kwargs["json"]["previous_messages"] == ["hello"]


@pytest.mark.asyncio
async def test_social_memory_context_compacts_query_params_and_serializes_as_of():
    social_memory = load_graphrag_route_module("social_memory")
    request = AsyncMock(return_value=_mock_response({"person_id": "person-1", "formatted": "Memory"}))
    mock_client = Mock()
    mock_client.request = request

    as_of = datetime(2026, 7, 10, 13, 0, tzinfo=timezone.utc)

    with (
        patch.object(social_memory, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(social_memory, "get_http_client", return_value=mock_client),
    ):
        result = await social_memory.graphrag_social_memory_context(
            platform="youtube",
            platform_user_id="channel-user",
            query=None,
            max_facts=10,
            include_recent_episodes=2,
            as_of=as_of,
        )

    assert result["person_id"] == "person-1"
    assert request.await_args.args[:2] == (
        "GET",
        "http://graphrag.test/social-memory/context",
    )
    assert request.await_args.kwargs["params"] == {
        "platform": "youtube",
        "platform_user_id": "channel-user",
        "max_facts": 10,
        "include_recent_episodes": 2,
        "as_of": "2026-07-10T13:00:00+00:00",
    }


@pytest.mark.asyncio
async def test_social_memory_explain_uses_expected_query_shape():
    social_memory = load_graphrag_route_module("social_memory")
    request = AsyncMock(return_value=_mock_response({"fact_ids": ["fact-1"], "cache_hit": True}))
    mock_client = Mock()
    mock_client.request = request

    with (
        patch.object(social_memory, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(social_memory, "get_http_client", return_value=mock_client),
    ):
        result = await social_memory.graphrag_social_memory_explain(
            platform="kick",
            platform_user_id="viewer-7",
            query="What does Ada like?",
            max_facts=5,
        )

    assert result["fact_ids"] == ["fact-1"]
    assert request.await_args.args[:2] == (
        "GET",
        "http://graphrag.test/social-memory/explain",
    )
    assert request.await_args.kwargs["params"] == {
        "platform": "kick",
        "platform_user_id": "viewer-7",
        "query": "What does Ada like?",
        "max_facts": 5,
    }


@pytest.mark.asyncio
async def test_social_memory_admin_routes_forward_api_key_header():
    social_memory = load_graphrag_route_module("social_memory")
    request = AsyncMock(side_effect=[
        _mock_response({"merged": True}),
        _mock_response({"new_fact_id": "fact-2"}),
        _mock_response([{"suite": "locomo", "accuracy": 0.9}]),
    ])
    mock_client = Mock()
    mock_client.request = request

    with (
        patch.object(social_memory, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(social_memory, "get_http_client", return_value=mock_client),
    ):
        await social_memory.graphrag_social_memory_merge_identities(
            social_memory.SocialMemoryMergeIdentitiesRequest(
                primary_person_id="person-a",
                secondary_person_id="person-b",
                confidence=0.95,
                reason="same chatter",
            ),
            x_api_key="secret-key",
        )
        await social_memory.graphrag_social_memory_correct_fact(
            social_memory.SocialMemoryCorrectFactRequest(
                fact_id="fact-1",
                new_fact_text="Ada likes graph search.",
                corrected_by="operator",
            ),
            x_api_key="secret-key",
        )
        await social_memory.graphrag_social_memory_eval(
            suite="locomo",
            baseline_path="/tmp/baseline.json",
            x_api_key="secret-key",
        )

    assert request.await_count == 3
    for call in request.await_args_list:
        assert call.kwargs["headers"] == {"X-API-Key": "secret-key"}

    assert request.await_args_list[0].args[:2] == (
        "POST",
        "http://graphrag.test/social-memory/identities/merge",
    )
    assert request.await_args_list[1].args[:2] == (
        "POST",
        "http://graphrag.test/social-memory/facts/correct",
    )
    assert request.await_args_list[2].args[:2] == (
        "POST",
        "http://graphrag.test/social-memory/admin/eval",
    )
    assert request.await_args_list[2].kwargs["params"] == {
        "suite": "locomo",
        "baseline_path": "/tmp/baseline.json",
    }


@pytest.mark.asyncio
async def test_social_memory_metrics_preserves_upstream_content_type():
    social_memory = load_graphrag_route_module("social_memory")
    upstream_response = _mock_response({"ignored": True})
    upstream_response.content = b"memory_ingest_total 3\n"
    upstream_response.headers = {"content-type": "text/plain; version=0.0.4"}
    request = AsyncMock(return_value=upstream_response)
    mock_client = Mock()
    mock_client.request = request

    with (
        patch.object(social_memory, "_graphrag_url", return_value="http://graphrag.test"),
        patch.object(social_memory, "get_http_client", return_value=mock_client),
    ):
        result = await social_memory.graphrag_social_memory_metrics()

    assert result.body == b"memory_ingest_total 3\n"
    assert result.media_type == "text/plain; version=0.0.4"
    assert request.await_args.args[:2] == (
        "GET",
        "http://graphrag.test/social-memory/metrics",
    )
