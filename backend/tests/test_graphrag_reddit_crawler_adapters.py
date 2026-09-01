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
    monkeypatch.setenv("REDDIT_CRAWLER_URL", "http://reddit-crawler.test")
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
async def test_reddit_health_forwards_to_direct_service():
    reddit = load_graphrag_route_module("reddit")
    get = AsyncMock(return_value=_mock_response({"status": "healthy"}))
    mock_client = Mock()
    mock_client.get = get

    with patch.object(reddit, "get_http_client", return_value=mock_client):
        result = await reddit.graphrag_reddit_health()

    assert result["status"] == "healthy"
    get.assert_awaited_once_with("http://reddit-crawler.test/health", timeout=30.0)


@pytest.mark.asyncio
async def test_reddit_start_crawl_forwards_crawl_request():
    reddit = load_graphrag_route_module("reddit")
    post = AsyncMock(return_value=_mock_response({"crawl_id": "crawl-1", "status": "started"}))
    mock_client = Mock()
    mock_client.post = post

    body = reddit.RedditCrawlRequest(
        query="local llm retrieval",
        subreddits=["LocalLLaMA", "MachineLearning"],
        max_posts=25,
        auto_ingest=True,
    )

    with patch.object(reddit, "get_http_client", return_value=mock_client):
        result = await reddit.graphrag_reddit_start_crawl(body)

    assert result["crawl_id"] == "crawl-1"
    post.assert_awaited_once_with(
        "http://reddit-crawler.test/crawl",
        json={
            "query": "local llm retrieval",
            "subreddits": ["LocalLLaMA", "MachineLearning"],
            "sort_by": "relevance",
            "time_filter": "month",
            "max_posts": 25,
            "max_comments_per_post": 20,
            "comment_depth": 2,
            "include_replies": True,
            "auto_ingest": True,
            "filter_nsfw": True,
        },
        timeout=60.0,
    )


@pytest.mark.asyncio
async def test_reddit_status_results_cancel_and_list_forward_crawl_id():
    reddit = load_graphrag_route_module("reddit")
    get = AsyncMock(side_effect=[
        _mock_response({"crawl_id": "crawl-1", "progress": 50}),
        _mock_response({"crawl_id": "crawl-1", "results": []}),
        _mock_response({"crawls": []}),
    ])
    delete = AsyncMock(return_value=_mock_response({"message": "cancelled"}))
    mock_client = Mock()
    mock_client.get = get
    mock_client.delete = delete

    with patch.object(reddit, "get_http_client", return_value=mock_client):
        status = await reddit.graphrag_reddit_crawl_status("crawl-1")
        results = await reddit.graphrag_reddit_crawl_results("crawl-1")
        cancelled = await reddit.graphrag_reddit_cancel_crawl("crawl-1")
        crawls = await reddit.graphrag_reddit_list_crawls()

    assert status["progress"] == 50
    assert results["results"] == []
    assert cancelled["message"] == "cancelled"
    assert crawls["crawls"] == []
    assert get.await_args_list[0].args[0] == "http://reddit-crawler.test/crawl/crawl-1/status"
    assert get.await_args_list[1].args[0] == "http://reddit-crawler.test/crawl/crawl-1/results"
    delete.assert_awaited_once_with("http://reddit-crawler.test/crawl/crawl-1", timeout=30.0)
    assert get.await_args_list[2].args[0] == "http://reddit-crawler.test/crawls"


@pytest.mark.asyncio
async def test_reddit_health_reports_missing_direct_crawler_url(monkeypatch):
    from modules.graphrag.config import reset_config

    reddit = load_graphrag_route_module("reddit")
    monkeypatch.delenv("REDDIT_CRAWLER_URL", raising=False)
    monkeypatch.delenv("RUNNING_IN_DOCKER", raising=False)
    monkeypatch.delenv("USE_DOCKER", raising=False)
    reset_config()

    result = await reddit.graphrag_reddit_health()

    assert result["available"] is False
    assert result["status"] == "not_configured"
    assert "REDDIT_CRAWLER_URL" in result["message"]
