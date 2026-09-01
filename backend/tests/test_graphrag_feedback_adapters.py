import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, Mock

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


@pytest.mark.asyncio
async def test_feedback_records_quality_signals_in_temp_database(tmp_path):
    feedback = load_graphrag_route_module("feedback")
    feedback.DB_PATH = str(tmp_path / "retrieval_feedback.db")
    feedback._db_initialized = False

    request = Mock(spec=Request)
    request.json = AsyncMock(return_value={
        "query": "How useful was graph context?",
        "rating": 1,
        "comment": "Helpful",
        "quality_signals": {
            "total_chunks": 4,
            "total_entities": 3,
            "avg_chunk_relevance": 0.82,
            "cross_reference_count": 2,
            "graph_expansion_count": 1,
        },
        "source": "workbench",
        "domain": "research",
    })

    result = await feedback.submit_feedback(request)
    analytics = await feedback.get_analytics(days=7, domain="research")

    assert result == {"status": "ok", "message": "Feedback recorded"}
    assert analytics["overall"]["total_feedback"] == 1
    assert analytics["overall"]["positive"] == 1
    assert analytics["overall"]["satisfaction_rate"] == 1.0
    assert analytics["domain_stats"][0]["domain"] == "research"


@pytest.mark.asyncio
async def test_feedback_analytics_reports_low_rated_problem_queries(tmp_path):
    feedback = load_graphrag_route_module("feedback")
    feedback.DB_PATH = str(tmp_path / "retrieval_feedback.db")
    feedback._db_initialized = False

    request = Mock(spec=Request)
    request.json = AsyncMock(return_value={
        "query": "Why did retrieval miss the source?",
        "rating": -1,
        "comment": "No source matched",
        "domain": "support",
    })

    await feedback.submit_feedback(request)
    analytics = await feedback.get_analytics(days=7, domain="support")

    assert analytics["overall"]["negative"] == 1
    assert analytics["problem_queries"][0]["query"] == "Why did retrieval miss the source?"
    assert analytics["problem_queries"][0]["comment"] == "No source matched"
