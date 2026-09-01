import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


HELPERS_PATH = Path(__file__).resolve().parents[1] / "routes" / "graphrag" / "helpers.py"
spec = importlib.util.spec_from_file_location("graphrag_helpers", HELPERS_PATH)
graphrag_helpers = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(graphrag_helpers)

_normalize_entity_type = graphrag_helpers._normalize_entity_type
_extract_entities_ner = graphrag_helpers._extract_entities_ner
_extract_relationships_rel = graphrag_helpers._extract_relationships_rel
optional_chat_context_timeout = graphrag_helpers.optional_chat_context_timeout
_transform_edge = graphrag_helpers._transform_edge
_transform_nodes = graphrag_helpers._transform_nodes


def test_normalize_entity_type_preserves_graphrag_variants():
    assert _normalize_entity_type("ORGANISATION") == "organization"
    assert _normalize_entity_type("SYSTEM") == "service"
    assert _normalize_entity_type("TIME") == "date"
    assert _normalize_entity_type("SPECIFICATION") == "concept"
    assert _normalize_entity_type("SOLUTION") == "product"
    assert _normalize_entity_type("POSITION") == "concept"
    assert _normalize_entity_type("REQUIREMENT") == "config"
    assert _normalize_entity_type("SAFETY") == "security"


def test_transform_nodes_preserves_raw_graphrag_metadata():
    nodes = _transform_nodes([
        {
            "id": "Llama Nexus",
            "label": "Llama Nexus",
            "properties": {
                "type": "ORGANISATION",
                "domain": "llama-nexus-real-demo",
                "confidence": 0.97,
                "source": "demo.md",
            },
        }
    ])

    assert nodes == [
        {
            "id": "Llama Nexus",
            "label": "Llama Nexus",
            "type": "organization",
            "occurrence": 1,
            "properties": {
                "type": "ORGANISATION",
                "domain": "llama-nexus-real-demo",
                "confidence": 0.97,
                "source": "demo.md",
                "description": "",
                "entity_type": "ORGANISATION",
                "original_type": "ORGANISATION",
            },
        }
    ]


def test_transform_edge_sets_relation_count_and_metadata():
    edge = _transform_edge({
        "id": "rel-1",
        "source": {"id": "GraphRAG"},
        "target": {"label": "Frontend"},
        "type": "supports",
        "weight": 4,
        "context": "GraphRAG supports the frontend graph view.",
        "properties": {"domain": "llama-nexus-real-demo"},
    })

    assert edge == {
        "id": "rel-1",
        "source": "GraphRAG",
        "target": "Frontend",
        "label": "supports",
        "type": "supports",
        "weight": 4,
        "relationCount": 1,
        "properties": {
            "domain": "llama-nexus-real-demo",
            "context": "GraphRAG supports the frontend graph view.",
            "type": "supports",
            "weight": 4,
            "relationCount": 1,
            "relationTypes": [{"type": "supports", "count": 1}],
        },
    }


def test_transform_edge_sums_relation_type_counts():
    edge = _transform_edge({
        "source": "Model Registry",
        "target": "Deploy",
        "label": "mixed",
        "properties": {
            "relationTypes": [
                {"type": "supports", "count": 2},
                {"type": "controls", "count": 3},
            ],
        },
    }, edge_id=7)

    assert edge["id"] == "edge_7"
    assert edge["relationCount"] == 5
    assert edge["properties"]["relationCount"] == 5
    assert edge["properties"]["relationTypes"] == [
        {"type": "supports", "count": 2},
        {"type": "controls", "count": 3},
    ]


@pytest.mark.asyncio
async def test_optional_chat_extractors_use_short_timeouts(monkeypatch):
    from modules.graphrag.config import reset_config

    monkeypatch.setenv("NER_API_URL", "http://ner.test")
    monkeypatch.setenv("REL_API_URL", "http://rel.test")
    monkeypatch.setenv("GRAPHRAG_HTTP_TIMEOUT", "30")
    reset_config()

    client = type("Client", (), {})()
    response = type("Response", (), {"status_code": 200})()
    response.json = lambda: {"entities": [], "relations": []}
    client.post = AsyncMock(return_value=response)
    errors = []

    await _extract_entities_ner(client, "What connects revenue and risk?", errors)
    await _extract_relationships_rel(client, "What connects revenue and risk?", errors)

    assert optional_chat_context_timeout() == 2.0
    assert client.post.await_args_list[0].kwargs["timeout"] == 2.0
    assert client.post.await_args_list[1].kwargs["timeout"] == 2.0

    reset_config()
