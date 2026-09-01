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
def graphrag_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHRAG_ENABLED", "true")
    monkeypatch.setenv("GRAPHRAG_URL", "http://graphrag.test")
    monkeypatch.setenv("GRAPHRAG_PIPELINE_RUNS_FILE", str(tmp_path / "pipeline_runs.json"))
    monkeypatch.delenv("NER_API_URL", raising=False)
    monkeypatch.delenv("REL_API_URL", raising=False)
    from modules.graphrag.config import reset_config
    from modules.graphrag.pipeline_runs import reset_pipeline_run_store

    reset_config()
    reset_pipeline_run_store()
    yield
    reset_config()
    reset_pipeline_run_store()


def _mock_response(payload, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


@pytest.mark.asyncio
async def test_unified_chat_context_adds_graph_path_reasoning():
    context = load_graphrag_route_module("context")
    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "How is Ada Lovelace connected to the Analytical Engine?",
        "rag_enabled": False,
        "domain": "history",
        "source": "Ada Lovelace",
        "target": "Analytical Engine",
        "bridge": "Charles Babbage",
        "relation": "designed",
    })
    mock_client = Mock()
    mock_client.post = AsyncMock(return_value=_mock_response({
        "primary_reasoning": "multi_hop",
        "requires_multi_hop": True,
    }))

    node_map = {
        "ada lovelace": {"name": "Ada Lovelace", "id": "n1", "type": "person"},
        "charles babbage": {"name": "Charles Babbage", "id": "n2", "type": "person"},
        "analytical engine": {"name": "Analytical Engine", "id": "n3", "type": "technology"},
    }
    raw_edges = [
        {
            "source": "Ada Lovelace",
            "target": "Charles Babbage",
            "label": "collaborated_with",
            "weight": 0.9,
            "evidence": "Ada Lovelace collaborated with Charles Babbage on early computing notes.",
            "metadata": {
                "local_graph_filename": "research.pptx",
                "slide": 2,
                "slide_title": "Computing Pioneers",
                "shape_title": "Ada and Babbage collaboration",
                "element_type": "connector",
            },
        },
        {
            "source": "Charles Babbage",
            "target": "Analytical Engine",
            "label": "designed",
            "weight": 0.8,
            "evidence": "The deck states Babbage designed the Analytical Engine.",
            "metadata": {
                "local_graph_filename": "research.pptx",
                "slide": 3,
                "slide_title": "Analytical Engine",
                "table": 1,
                "row": 2,
                "element_type": "table_fact",
            },
        },
    ]

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[
            {"name": "Ada Lovelace", "type": "PERSON", "score": 0.98},
            {"name": "Analytical Engine", "type": "TECHNOLOGY", "score": 0.95},
        ])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_search_graphrag", AsyncMock(return_value=([], []))),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
        patch.object(context, "_fetch_graph_data", AsyncMock(return_value=(node_map, raw_edges))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    graph_reasoning = result["graph_reasoning"]
    assert graph_reasoning["status"] == "complete"
    assert graph_reasoning["intent_hints"] == {
        "source": "Ada Lovelace",
        "target": "Analytical Engine",
        "bridge": "Charles Babbage",
        "relation": "designed",
    }
    assert graph_reasoning["query_complexity"]["requires_multi_hop"] is True
    assert graph_reasoning["paths"][0]["hops"] == 2
    assert graph_reasoning["paths"][0]["nodes"] == ["Ada Lovelace", "Charles Babbage", "Analytical Engine"]
    assert "designed" in graph_reasoning["paths"][0]["matched_query_terms"]
    assert {"connector", "semantic", "table"}.issubset(set(graph_reasoning["paths"][0]["evidence_families"]))
    assert "Charles Babbage connects Ada Lovelace and Analytical Engine" in graph_reasoning["paths"][0]["explanation"]
    assert graph_reasoning["paths"][0]["answer_hint"].startswith("Use this path")
    assert graph_reasoning["bridge_entities"][0]["name"] == "Charles Babbage"
    assert graph_reasoning["bridge_entities"][0]["path_count"] == 1
    assert "Ada Lovelace to Analytical Engine" in graph_reasoning["bridge_entities"][0]["text"]
    assert graph_reasoning["summary"]["mode"] == "multi_hop_path"
    assert graph_reasoning["summary"]["bridge_entities"] == ["Charles Babbage"]
    assert "Strongest path:" in graph_reasoning["summary"]["evidence"][0]
    assert "Use the strongest path first" in graph_reasoning["summary"]["answer_guidance"]
    assert graph_reasoning["answer_plan"]["strategy"] == "walk_strongest_path"
    assert graph_reasoning["answer_plan"]["primary_path"]["nodes"] == ["Ada Lovelace", "Charles Babbage", "Analytical Engine"]
    assert graph_reasoning["answer_plan"]["primary_path"]["explanation"] == graph_reasoning["paths"][0]["explanation"]
    assert {"connector", "semantic", "table"}.issubset(set(graph_reasoning["answer_plan"]["primary_path"]["evidence_families"]))
    assert graph_reasoning["answer_plan"]["hops"][0]["relation"] == "collaborated_with"
    assert graph_reasoning["answer_plan"]["hops"][1]["relation"] == "designed"
    assert "connector" in graph_reasoning["answer_plan"]["hops"][0]["evidence_families"]
    assert graph_reasoning["answer_plan"]["hops"][0]["rationale"]
    assert graph_reasoning["answer_plan"]["bridge_entities"] == ["Charles Babbage"]
    assert graph_reasoning["answer_plan"]["evidence_cards"][0]["role"] == "primary_path_hop"
    assert graph_reasoning["answer_plan"]["evidence_cards"][0]["citation_label"].startswith("local pipeline research.pptx")
    assert graph_reasoning["answer_plan"]["evidence_cards"][0]["location"]["filename"] == "research.pptx"
    assert graph_reasoning["answer_plan"]["evidence_cards"][0]["location"]["slide"] == 2
    assert graph_reasoning["answer_plan"]["evidence_cards"][0]["location"]["shape"] == "Ada and Babbage collaboration"
    assert graph_reasoning["answer_plan"]["evidence_cards"][1]["location"]["table"] == 1
    assert "=== Graph Answer Plan ===" in result["context_text"]
    assert "Strategy: walk_strongest_path" in result["context_text"]
    assert "Path explanation: Charles Babbage connects Ada Lovelace and Analytical Engine" in result["context_text"]
    assert "Evidence families:" in result["context_text"]
    assert all(term in result["context_text"] for term in ("connector", "semantic", "table"))
    assert "Ordered hops:" in result["context_text"]
    assert "Evidence checklist:" in result["context_text"]
    assert "Hop 1: Ada Lovelace --[collaborated_with]--> Charles Babbage" in result["context_text"]
    assert "source: research.pptx, slide 2: Computing Pioneers" in result["context_text"]
    assert "cite: local pipeline research.pptx" not in result["context_text"]
    assert "=== Graph Reasoning Summary ===" in result["context_text"]
    assert "Ada Lovelace --[collaborated_with]-- Charles Babbage" in result["context_text"]
    assert "=== Bridge Entities ===" in result["context_text"]
    assert "Charles Babbage bridges Ada Lovelace to Analytical Engine" in result["context_text"]
    assert result["quality_signals"]["graph_path_count"] == 1
    assert result["quality_signals"]["graph_layout_path_count"] == 0
    assert result["quality_signals"]["graph_semantic_path_count"] == 1
    assert result["quality_signals"]["graph_answer_strategy"] == "walk_strongest_path"
    assert result["quality_signals"]["graph_answer_evidence_card_count"] == 2
    assert result["quality_signals"]["graph_answer_provenance_anchor_count"] == 2
    assert {"ada", "lovelace", "analytical", "engine"} & set(result["quality_signals"]["graph_matched_query_terms"])
    assert result["quality_signals"]["bridge_entity_count"] == 1
    assert result["services_used"]["graph_reasoning"] is True


def test_graph_answer_plan_context_labels_document_path_nodes_as_source_evidence():
    context = load_graphrag_route_module("context")
    source_doc = "graphrag-complex-office-source.pptx"
    path = {
        "source": "Product Revenue",
        "target": "Forecast Risk",
        "nodes": ["Product Revenue", source_doc, "Forecast Risk"],
        "hops": 2,
        "score": 0.81,
        "path_type": "semantic",
        "relationships": [
            {
                "source": "Product Revenue",
                "relation": "supported_by",
                "target": source_doc,
                "evidence": "Product Revenue is documented in the deck.",
                "metadata": {
                    "local_graph_filename": source_doc,
                    "slide": 1,
                    "slide_title": "Revenue Architecture",
                },
            },
            {
                "source": source_doc,
                "relation": "mentions",
                "target": "Forecast Risk",
                "evidence": "The deck mentions Forecast Risk near the revenue discussion.",
                "metadata": {
                    "local_graph_filename": source_doc,
                    "slide": 3,
                    "slide_title": "Forecast Risk",
                },
            },
        ],
    }
    path.update(context._graph_path_explanation(path))

    graph_reasoning = {
        "paths": [path],
        "direct_relationships": [],
        "bridge_entities": [
            {
                "name": source_doc,
                "connects": [{"source": "Product Revenue", "target": "Forecast Risk"}],
            }
        ],
        "interaction_handoffs": [],
    }

    plan = context._graph_answer_plan(graph_reasoning)
    context_text = context._graph_answer_plan_context(plan)

    assert plan["primary_path"]["display_nodes"] == ["Product Revenue", "source evidence", "Forecast Risk"]
    assert plan["primary_path"]["contains_source_evidence_bridge"] is True
    assert "Product Revenue --[supported_by]-- source evidence --[mentions]-- Forecast Risk" in context_text
    assert "Path note: source evidence is a citation bridge" in context_text
    assert "Bridge entities: source evidence" in context_text
    assert f"Product Revenue --[supported_by]-- {source_doc}" not in context_text
    assert "source: graphrag-complex-office-source.pptx" in context_text
    assert "cite: local pipeline graphrag-complex-office-source.pptx" not in context_text


def test_relationship_focused_query_prefers_direct_domain_edge_over_source_path():
    context = load_graphrag_route_module("context")
    direct_relationship = {
        "source": "Risk Map",
        "relation": "gates",
        "target": "Launch Priority when compliance exposure is high",
        "relationship_type": "semantic",
        "score": 0.92,
        "weight": 0.95,
        "evidence": "Risk Map gates Launch Priority when compliance exposure is high.",
        "metadata": {
            "local_graph_filename": "graphrag-pdf-strategy.pdf",
            "page": 1,
            "location": "page 1 block 5",
            "element_type": "text",
        },
    }
    provenance_path = {
        "source": "Executive relationships",
        "target": "source document",
        "nodes": [
            "Executive relationships",
            "source evidence",
            "https://example.com/regional-risk-map",
            "source document",
        ],
        "hops": 3,
        "score": 0.99,
        "path_type": "semantic",
        "relationships": [
            {
                "source": "Executive relationships",
                "relation": "contains",
                "target": "source evidence",
                "relationship_type": "layout",
                "weight": 0.99,
            },
            {
                "source": "source evidence",
                "relation": "links_to",
                "target": "https://example.com/regional-risk-map",
                "relationship_type": "semantic",
                "weight": 0.99,
            },
            {
                "source": "https://example.com/regional-risk-map",
                "relation": "contains_link",
                "target": "source document",
                "relationship_type": "layout",
                "weight": 0.99,
            },
        ],
    }
    provenance_path.update(context._graph_path_explanation(provenance_path))

    graph_reasoning = context._finalize_graph_reasoning_result({
        "query": "What is the strongest relationship in this PDF?",
        "reasoning_needed": True,
        "paths": [provenance_path],
        "direct_relationships": [direct_relationship],
        "bridge_entities": [],
        "graph_overview": {},
        "interaction_handoffs": [],
        "pipeline_evidence": {},
    })

    plan = graph_reasoning["answer_plan"]
    context_text = context._graph_answer_plan_context(plan)

    assert plan["strategy"] == "cite_direct_relationships"
    assert plan["primary_path"] is None
    assert plan["supporting_relationships"][0]["source"] == "Risk Map"
    assert plan["supporting_relationships"][0]["relation"] == "gates"
    assert "Treat URL, file, source-document, and source-evidence paths as source context" in plan["instruction"]
    assert graph_reasoning["summary"]["mode"] == "direct_relationship"
    assert graph_reasoning["summary"]["evidence"][0].startswith("Direct relationships:")
    assert "Risk Map --[gates]--> Launch Priority when compliance exposure is high" in context_text
    assert "Primary path:" not in context_text


def test_graph_mention_pairs_ignore_document_artifact_terms():
    context = load_graphrag_route_module("context")
    node_map = {
        "recent": {"name": "recent", "type": "concept"},
        "graphrag": {"name": "GraphRAG", "type": "concept"},
        "graphrag-strategy-rich-fixture.pptx": {
            "name": "graphrag-strategy-rich-fixture.pptx",
            "type": "document",
        },
        "product revenue": {"name": "Product Revenue", "type": "metric"},
        "regional demand": {"name": "Regional Demand", "type": "metric"},
    }
    raw_edges = [
        {
            "source": "GraphRAG",
            "target": "graphrag-strategy-rich-fixture.pptx",
            "label": "processed_document",
        },
        {
            "source": "Product Revenue",
            "target": "Regional Demand",
            "label": "drives",
        },
    ]

    broad_pairs = context._entity_pairs_from_graph_mentions(
        "Use the recent GraphRAG pipeline trace for graphrag-strategy-rich-fixture.pptx to summarize key evidence and meaningful graph paths.",
        node_map,
        raw_edges,
    )
    specific_pairs = context._entity_pairs_from_graph_mentions(
        "Trace Product Revenue to Regional Demand.",
        node_map,
        raw_edges,
    )

    assert broad_pairs == []
    assert specific_pairs == [("Product Revenue", "Regional Demand")]


@pytest.mark.asyncio
async def test_pipeline_scoped_broad_chat_skips_graph_inferred_path_search():
    context = load_graphrag_route_module("context")
    client = Mock()
    errors = []
    local_node_map = {
        "rag pipeline": {"name": "RAG Pipeline", "type": "process"},
        "graphrag service": {"name": "GraphRAG service", "type": "service"},
        "graph anchors": {"name": "graph anchors", "type": "concept"},
    }
    local_edges = [
        {
            "source": "RAG Pipeline",
            "target": "GraphRAG service",
            "label": "uses",
            "weight": 1,
            "evidence": "The pipeline uses the GraphRAG service.",
        },
        {
            "source": "RAG Pipeline",
            "target": "graph anchors",
            "label": "produces",
            "weight": 1,
            "evidence": "The pipeline produces graph anchors.",
        },
    ]

    fetch_graph_data = AsyncMock(side_effect=AssertionError("remote graph fetch should be skipped"))
    with (
        patch.object(context, "_optional_query_complexity", AsyncMock(return_value={})),
        patch.object(context, "_fetch_graph_data", fetch_graph_data),
        patch.object(
            context,
            "_local_pipeline_graph_data",
            return_value=(
                local_node_map,
                local_edges,
                [{"run_id": "run-1", "filename": "strategy.pptx", "edge_count": 2}],
            ),
        ),
        patch.object(context, "_local_pipeline_interaction_handoffs", return_value=[]),
    ):
        result = await context._build_graph_reasoning_context(
            client,
            "Use the recent GraphRAG pipeline trace to summarize key evidence and meaningful graph paths.",
            [],
            "general",
            errors,
            pipeline_run_ids=["run-1"],
            prefer_local_pipeline_only=True,
        )

    assert result["status"] == "complete"
    assert fetch_graph_data.await_count == 0
    assert result["paths"] == []
    assert result["entity_pairs"] == []
    assert result["graph_sources"]["local_pipeline_runs"][0]["run_id"] == "run-1"
    assert result["direct_relationships"]
    assert result["graph_overview"]["relationship_count"] == 2


@pytest.mark.asyncio
async def test_unified_chat_context_surfaces_pipeline_evidence_handoff_without_inventing_paths():
    context = load_graphrag_route_module("context")
    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "Summarize the implementation and flow evidence in this batch.",
        "rag_enabled": False,
        "domain": "general",
        "pipeline_evidence_totals": {
            "code_blocks": 3,
            "diagram_relationships": 5,
            "links": 2,
            "forms": 1,
        },
        "pipeline_evidence_families": [
            "code_blocks",
            "diagram_relationships",
            "links",
            "forms",
        ],
        "graph_overview_counts": {
            "relationships": 7,
            "cross_document_paths": 2,
        },
    })
    mock_client = Mock()
    mock_client.post = AsyncMock(return_value=_mock_response({}))

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_search_graphrag", AsyncMock(return_value=([], []))),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
        patch.object(context, "_local_pipeline_graph_data", return_value=({}, [], [])),
        patch.object(context, "_local_pipeline_interaction_handoffs", return_value=[]),
        patch.object(context, "_local_pipeline_artifact_context", return_value={"runs": [], "run_count": 0, "artifact_count": 0}),
    ):
        result = await context.graphrag_chat_context_v2(request)

    pipeline_evidence = result["pipeline_evidence"]
    assert pipeline_evidence["counts"]["code_blocks"] == 3
    assert pipeline_evidence["counts"]["diagram_relationships"] == 5
    assert pipeline_evidence["graph_overview_counts"]["cross_document_paths"] == 2
    assert result["graph_reasoning"]["status"] == "pipeline_evidence"
    assert result["graph_reasoning"]["summary"]["mode"] == "pipeline_evidence"
    assert result["graph_reasoning"]["answer_plan"]["strategy"] == "graph_not_required"
    assert result["graph_reasoning"]["answer_plan"]["pipeline_evidence_coverage"]["counts"]["forms"] == 1
    evidence_cards = result["graph_reasoning"]["answer_plan"]["evidence_cards"]
    assert evidence_cards[0]["role"] == "pipeline_evidence_family"
    assert evidence_cards[0]["kind"] == "diagram_relationships"
    assert evidence_cards[0]["citation_label"] == "pipeline evidence: 5 diagram edges"
    assert evidence_cards[0]["location"] == {
        "evidence_family": "diagram_relationships",
        "evidence_count": 5,
    }
    assert result["graph_reasoning"]["answer_plan"]["provenance_anchors"][0] == "pipeline evidence: 5 diagram edges"
    assert result["quality_signals"]["pipeline_evidence_family_count"] == 4
    assert result["quality_signals"]["pipeline_evidence_total_count"] == 11
    assert result["quality_signals"]["graph_answer_evidence_card_count"] >= 4
    assert result["quality_signals"]["pipeline_evidence_top_families"][:2] == ["diagram_relationships", "code_blocks"]
    assert result["services_used"]["pipeline_evidence_handoff"] is True
    assert "Pipeline evidence coverage:" in result["context_text"]
    assert "Evidence Checklist" not in result["context_text"]
    assert "Evidence checklist:" in result["context_text"]
    assert "Pipeline Evidence Family 1: 5 diagram edges | source: pipeline evidence: 5 diagram edges" in result["context_text"]
    assert "3 code blocks" in result["context_text"]
    assert "5 diagram edges" in result["context_text"]
    assert "Graph overview hints:" in result["context_text"]
    assert "Use code block evidence for implementation details" in result["context_text"]
    assert "Use diagram, connector, and layout evidence" in result["context_text"]
    assert result["graph_reasoning"]["paths"] == []
    assert result["services_used"]["graph_reasoning"] is False


@pytest.mark.asyncio
async def test_unified_chat_context_retrieves_local_chunks_from_pipeline_evidence_terms():
    from modules.graphrag.pipeline_runs import get_pipeline_run_store

    context = load_graphrag_route_module("context")
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_ingest_evidence_terms_001",
        "status": "preview_only",
        "updated_at": "2026-01-15T13:00:00Z",
        "document": {"filename": "batch.md", "domain": "general"},
        "local_context": {
            "filename": "batch.md",
            "domain": "general",
            "chunk_count": 2,
            "cached_chunk_count": 2,
            "chunks": [
                {
                    "id": "chunk_evidence",
                    "chunk_index": 0,
                    "content": (
                        "The code block is a Python snippet that implements the upload handoff. "
                        "A diagram edge shows Upload -> Extract -> Graph as the pipeline flow."
                    ),
                    "metadata": {
                        "provenance": ["section=Implementation Evidence; lines=4-9"],
                    },
                },
                {
                    "id": "chunk_noise",
                    "chunk_index": 1,
                    "content": "Budget narrative and meeting logistics only.",
                    "metadata": {
                        "provenance": ["section=Meeting Notes"],
                    },
                },
            ],
        },
    })
    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "Summarize implementation evidence from this batch.",
        "rag_enabled": True,
        "domain": "general",
        "pipeline_run_ids": ["graphrag_ingest_evidence_terms_001"],
        "pipeline_evidence_totals": {
            "code_blocks": 1,
            "diagram_relationships": 1,
        },
        "pipeline_evidence_families": ["code_blocks", "diagram_relationships"],
    })
    mock_client = Mock()

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_retrieve_rag_chunks_for_query", AsyncMock(return_value=[])),
        patch.object(context, "_search_graphrag", AsyncMock(return_value=([], []))),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    assert result["chunks"]
    chunk = result["chunks"][0]
    assert chunk["id"] == "pipeline:graphrag_ingest_evidence_terms_001:chunk_evidence"
    assert "local_pipeline_evidence_context" in chunk["retrieval_reasons"]
    assert "local_pipeline_graph_augmented" in chunk["retrieval_reasons"]
    evidence_terms = chunk["metadata"]["matched_local_pipeline_evidence_terms"]
    assert {"code", "block", "implementation"} & set(evidence_terms)
    assert any(term in evidence_terms for term in ("code block", "diagram edge"))
    assert "code block" in result["quality_signals"]["graph_augmented_retrieval_terms"]
    assert result["quality_signals"]["local_pipeline_evidence_context_chunk_count"] == 1
    assert result["services_used"]["local_pipeline_context"] is True
    assert "recent pipeline trace" in result["context_text"]
    assert "code block is a Python snippet" in result["context_text"]


@pytest.mark.asyncio
async def test_unified_chat_context_retrieves_local_chunks_from_chunk_evidence_metadata():
    from modules.graphrag.pipeline_runs import get_pipeline_run_store

    context = load_graphrag_route_module("context")
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_ingest_chunk_evidence_001",
        "status": "preview_only",
        "updated_at": "2026-01-15T13:10:00Z",
        "document": {"filename": "approval-deck.pptx", "domain": "slides"},
        "local_context": {
            "filename": "approval-deck.pptx",
            "domain": "slides",
            "chunk_count": 2,
            "cached_chunk_count": 2,
            "chunks": [
                {
                    "id": "chunk_form",
                    "chunk_index": 0,
                    "content": "The required box is enabled for the launch workflow.",
                    "metadata": {
                        "evidence_families": ["form_field", "presentation_structure"],
                        "primary_evidence_family": "form_field",
                        "location_hints": [
                            "slide 3: Revenue Plan",
                            "form Approval required",
                            "visibility visible",
                        ],
                        "primary_location_hint": "slide 3: Revenue Plan",
                    },
                },
                {
                    "id": "chunk_noise",
                    "chunk_index": 1,
                    "content": "Budget narrative and meeting logistics only.",
                    "metadata": {
                        "evidence_families": ["text"],
                        "location_hints": ["slide 9: Appendix"],
                    },
                },
            ],
        },
    })
    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "Which approval form field is checked on slide 3?",
        "rag_enabled": True,
        "domain": "slides",
        "pipeline_run_ids": ["graphrag_ingest_chunk_evidence_001"],
    })
    mock_client = Mock()

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_retrieve_rag_chunks_for_query", AsyncMock(return_value=[])),
        patch.object(context, "_search_graphrag", AsyncMock(return_value=([], []))),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    assert result["chunks"]
    chunk = result["chunks"][0]
    assert chunk["id"] == "pipeline:graphrag_ingest_chunk_evidence_001:chunk_form"
    assert "local_pipeline_chunk_evidence" in chunk["retrieval_reasons"]
    matched_terms = set(chunk["metadata"]["matched_local_chunk_evidence_terms"])
    assert {"form", "field", "approval", "slide"} & matched_terms
    assert result["quality_signals"]["local_pipeline_chunk_evidence_chunk_count"] == 1
    assert "document evidence:" in result["context_text"]
    assert "required box is enabled" in result["context_text"]


@pytest.mark.asyncio
async def test_unified_chat_context_probes_recent_domain_graph_for_path_question_without_run_ids():
    from modules.graphrag.pipeline_runs import get_pipeline_run_store

    context = load_graphrag_route_module("context")
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_ingest_recent_domain_graph_001",
        "status": "complete",
        "updated_at": "2026-01-15T14:00:00Z",
        "document": {"filename": "strategy.pptx", "domain": "slides"},
        "local_graph": {
            "filename": "strategy.pptx",
            "domain": "slides",
            "node_count": 2,
            "edge_count": 1,
            "relationship_count": 1,
            "truncated": False,
            "nodes": [
                {"id": "plan", "name": "Knowledge graph launch plan", "type": "section"},
                {"id": "note", "name": "Speaker note: graph launch path", "type": "speaker_notes"},
            ],
            "edges": [
                {
                    "source": "Knowledge graph launch plan",
                    "target": "Speaker note: graph launch path",
                    "label": "contains_speaker_notes",
                    "type": "contains_speaker_notes",
                    "weight": 0.84,
                    "evidence": "Speaker note: emphasize graph-grounded launch path.",
                    "metadata": {
                        "slide": 1,
                        "slide_title": "Revenue Architecture",
                        "notes_part": "ppt/notesSlides/notesSlide1.xml",
                        "element_type": "speaker_notes",
                        "location": "slide 1 speaker notes",
                    },
                },
            ],
        },
    })
    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "Which graph path connects two important entities in this document, and what provenance should I cite?",
        "rag_enabled": False,
        "domain": "slides",
    })
    mock_client = Mock()
    fetch_graph = AsyncMock(return_value=({}, []))

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_search_graphrag", AsyncMock(return_value=([], []))),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
        patch.object(context, "_fetch_graph_data", fetch_graph),
    ):
        result = await context.graphrag_chat_context_v2(request)

    graph_reasoning = result["graph_reasoning"]
    assert fetch_graph.await_count == 0
    assert graph_reasoning["status"] == "complete"
    assert graph_reasoning["reasoning_needed"] is True
    assert graph_reasoning["graph_inferred_entity_pairs"] == [
        {
            "source": "Knowledge graph launch plan",
            "target": "Speaker note: graph launch path",
        }
    ]
    assert graph_reasoning["paths"][0]["nodes"] == [
        "Knowledge graph launch plan",
        "Speaker note: graph launch path",
    ]
    assert graph_reasoning["paths"][0]["relationships"][0]["provenance"].startswith("local pipeline strategy.pptx")
    assert graph_reasoning["graph_sources"]["local_pipeline_runs"][0]["run_id"] == "graphrag_ingest_recent_domain_graph_001"
    assert result["services_used"]["local_pipeline_graph"] is True
    assert result["services_used"]["graph_reasoning"] is True
    assert result["quality_signals"]["graph_path_count"] == 1
    assert result["quality_signals"]["local_pipeline_graph_run_count"] == 1
    assert "=== Graph Paths ===" in result["context_text"]
    assert "source: strategy.pptx" in result["context_text"]
    assert "local pipeline strategy.pptx" not in result["context_text"]


@pytest.mark.asyncio
async def test_unified_chat_context_infers_entity_pair_from_connection_question():
    context = load_graphrag_route_module("context")
    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "How is Ada Lovelace connected to the Analytical Engine?",
        "rag_enabled": False,
        "domain": "history",
    })
    mock_client = Mock()
    mock_client.post = AsyncMock(return_value=_mock_response({
        "primary_reasoning": "multi_hop",
        "requires_multi_hop": True,
    }))

    node_map = {
        "ada lovelace": {"name": "Ada Lovelace", "id": "n1", "type": "person"},
        "charles babbage": {"name": "Charles Babbage", "id": "n2", "type": "person"},
        "analytical engine": {"name": "Analytical Engine", "id": "n3", "type": "technology"},
    }
    raw_edges = [
        {"source": "Ada Lovelace", "target": "Charles Babbage", "label": "collaborated_with", "weight": 0.9},
        {"source": "Charles Babbage", "target": "Analytical Engine", "label": "designed", "weight": 0.8},
    ]

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_search_graphrag", AsyncMock(return_value=([], []))),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
        patch.object(context, "_fetch_graph_data", AsyncMock(return_value=(node_map, raw_edges))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    graph_reasoning = result["graph_reasoning"]
    assert graph_reasoning["status"] == "complete"
    assert graph_reasoning["inferred_entity_pairs"] == [
        {"source": "Ada Lovelace", "target": "Analytical Engine"}
    ]
    assert graph_reasoning["paths"][0]["nodes"] == ["Ada Lovelace", "Charles Babbage", "Analytical Engine"]
    assert graph_reasoning["paths"][0]["score"] > 0
    assert "score:" in result["context_text"]


@pytest.mark.asyncio
async def test_unified_chat_context_infers_entity_pair_from_trace_arrow_path_prompt():
    context = load_graphrag_route_module("context")
    request = Mock()
    request.json = AsyncMock(return_value={
        "query": (
            'Using the GraphRAG pipeline trace for research.md, explain the two-hop graph path '
            '"Ada Lovelace" -> "Charles Babbage" -> "Analytical Engine".'
        ),
        "rag_enabled": False,
        "domain": "history",
    })
    mock_client = Mock()
    mock_client.post = AsyncMock(return_value=_mock_response({
        "primary_reasoning": "multi_hop",
        "requires_multi_hop": True,
    }))

    node_map = {
        "ada lovelace": {"name": "Ada Lovelace", "id": "n1", "type": "person"},
        "charles babbage": {"name": "Charles Babbage", "id": "n2", "type": "person"},
        "analytical engine": {"name": "Analytical Engine", "id": "n3", "type": "technology"},
    }
    raw_edges = [
        {"source": "Ada Lovelace", "target": "Charles Babbage", "label": "collaborated_with", "weight": 0.9},
        {"source": "Charles Babbage", "target": "Analytical Engine", "label": "designed", "weight": 0.8},
    ]

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_search_graphrag", AsyncMock(return_value=([], []))),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
        patch.object(context, "_fetch_graph_data", AsyncMock(return_value=(node_map, raw_edges))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    graph_reasoning = result["graph_reasoning"]
    assert graph_reasoning["status"] == "complete"
    assert graph_reasoning["inferred_entity_pairs"] == [
        {"source": "Ada Lovelace", "target": "Analytical Engine"}
    ]
    assert graph_reasoning["paths"][0]["nodes"] == ["Ada Lovelace", "Charles Babbage", "Analytical Engine"]
    assert graph_reasoning["bridge_entities"][0]["name"] == "Charles Babbage"


@pytest.mark.asyncio
async def test_unified_chat_context_prioritizes_explicit_through_bridge_over_direct_edge():
    context = load_graphrag_route_module("context")
    request = Mock()
    request.json = AsyncMock(return_value={
        "query": (
            "Trace the path from Approval required before launch workflow to Launch workflow "
            "through Approval Dependency Cluster. Show each hop label and cite the slide evidence."
        ),
        "rag_enabled": False,
        "domain": "slides",
    })
    mock_client = Mock()
    mock_client.post = AsyncMock(return_value=_mock_response({
        "primary_reasoning": "multi_hop",
        "requires_multi_hop": True,
    }))

    node_map = {
        "approval required before launch workflow": {
            "name": "Approval required before launch workflow",
            "id": "n1",
            "type": "text",
        },
        "launch workflow": {"name": "Launch workflow", "id": "n2", "type": "concept"},
        "approval dependency cluster": {
            "name": "Approval Dependency Cluster",
            "id": "n3",
            "type": "shape_group",
        },
    }
    raw_edges = [
        {
            "source": "Approval required before launch workflow",
            "target": "Launch workflow",
            "label": "gates",
            "weight": 0.99,
            "evidence": "Approval required before launch workflow gates Launch workflow.",
            "metadata": {"local_graph_filename": "approval.pptx", "slide_title": "Launch Approval Packet"},
        },
        {
            "source": "Approval required before launch workflow",
            "target": "Approval Dependency Cluster",
            "label": "groups_visual",
            "weight": 0.8,
            "evidence": "Approval Dependency Cluster groups Approval required before launch workflow.",
            "metadata": {"local_graph_filename": "approval.pptx", "slide_title": "Launch Approval Packet"},
        },
        {
            "source": "Launch workflow",
            "target": "Approval Dependency Cluster",
            "label": "groups_visual",
            "weight": 0.8,
            "evidence": "Approval Dependency Cluster groups Launch workflow.",
            "metadata": {"local_graph_filename": "approval.pptx", "slide_title": "Launch Approval Packet"},
        },
    ]

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_search_graphrag", AsyncMock(return_value=([], []))),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
        patch.object(context, "_fetch_graph_data", AsyncMock(return_value=(node_map, raw_edges))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    graph_reasoning = result["graph_reasoning"]
    primary_path = graph_reasoning["answer_plan"]["primary_path"]
    assert graph_reasoning["intent_hints"]["bridge"] == "Approval Dependency Cluster"
    assert graph_reasoning["inferred_entity_pairs"] == [
        {"source": "Approval required before launch workflow", "target": "Launch workflow"}
    ]
    assert graph_reasoning["answer_plan"]["strategy"] == "walk_strongest_path"
    assert primary_path["nodes"] == [
        "Approval required before launch workflow",
        "Approval Dependency Cluster",
        "Launch workflow",
    ]
    assert [hop["relation"] for hop in graph_reasoning["answer_plan"]["hops"]] == [
        "groups_visual",
        "groups_visual",
    ]


@pytest.mark.asyncio
async def test_unified_chat_context_infers_entity_pair_from_unquoted_edge_prompt():
    context = load_graphrag_route_module("context")
    request = Mock()
    request.json = AsyncMock(return_value={
        "query": (
            "Explain the evidence behind Approval required -> gates in "
            "graphrag-approval-fixture.md. Cite the relevant relationship evidence."
        ),
        "rag_enabled": False,
        "domain": "slides",
    })
    mock_client = Mock()
    mock_client.post = AsyncMock(return_value=_mock_response({
        "primary_reasoning": "relationship_evidence",
        "requires_multi_hop": False,
    }))

    node_map = {
        "approval required": {"name": "Approval required", "id": "n1", "type": "form_field"},
        "gates": {"name": "gates", "id": "n2", "type": "table_column"},
    }
    raw_edges = [
        {
            "source": "Approval required",
            "target": "gates",
            "label": "has_column_2",
            "weight": 0.93,
            "evidence": "Approval required | gates | Launch workflow",
            "metadata": {
                "local_graph_filename": "graphrag-approval-fixture.md",
                "table": 1,
                "table_name": "Relationship table",
                "form_kind": "pptx_control",
            },
        },
    ]

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_search_graphrag", AsyncMock(return_value=([], []))),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
        patch.object(context, "_fetch_graph_data", AsyncMock(return_value=(node_map, raw_edges))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    graph_reasoning = result["graph_reasoning"]
    assert graph_reasoning["status"] == "complete"
    assert graph_reasoning["reasoning_needed"] is True
    assert graph_reasoning["inferred_entity_pairs"] == [
        {"source": "Approval required", "target": "gates"}
    ]
    assert graph_reasoning["paths"][0]["nodes"] == ["Approval required", "gates"]
    assert graph_reasoning["paths"][0]["hops"] == 1
    assert {"table", "form_field"}.issubset(set(graph_reasoning["paths"][0]["evidence_families"]))
    assert graph_reasoning["answer_plan"]["strategy"] == "walk_strongest_path"
    assert "Approval required --[has_column_2]-- gates" in result["context_text"]


@pytest.mark.asyncio
async def test_unified_chat_context_uses_direct_relationships_for_single_entity_question():
    context = load_graphrag_route_module("context")
    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "What does GraphRAG connect to?",
        "rag_enabled": False,
        "domain": "docs",
    })
    mock_client = Mock()
    mock_client.post = AsyncMock(return_value=_mock_response({
        "primary_reasoning": "entity_exploration",
        "requires_multi_hop": False,
    }))

    node_map = {
        "graphrag": {"name": "GraphRAG", "id": "n1", "type": "system"},
        "neo4j": {"name": "Neo4j", "id": "n2", "type": "database"},
        "qdrant": {"name": "Qdrant", "id": "n3", "type": "database"},
    }
    raw_edges = [
        {"source": "GraphRAG", "target": "Neo4j", "label": "stores_graph_in", "weight": 0.91},
        {"source": "GraphRAG", "target": "Qdrant", "label": "indexes_vectors_in", "weight": 0.88},
    ]

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[
            {"name": "GraphRAG", "type": "SYSTEM", "score": 0.97},
        ])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_search_graphrag", AsyncMock(return_value=([], []))),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
        patch.object(context, "_fetch_graph_data", AsyncMock(return_value=(node_map, raw_edges))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    graph_reasoning = result["graph_reasoning"]
    assert graph_reasoning["status"] == "complete"
    assert graph_reasoning["paths"] == []
    assert len(graph_reasoning["direct_relationships"]) == 2
    assert graph_reasoning["summary"]["mode"] == "direct_relationship"
    assert graph_reasoning["summary"]["direct_relationship_count"] == 2
    assert "Direct relationships:" in graph_reasoning["summary"]["evidence"][0]
    assert graph_reasoning["answer_plan"]["strategy"] == "cite_direct_relationships"
    assert graph_reasoning["answer_plan"]["supporting_relationships"][0]["relation"] == "stores_graph_in"
    assert graph_reasoning["direct_relationships"][0]["target"] == "Neo4j"
    assert result["quality_signals"]["direct_graph_relationship_count"] == 2
    assert result["quality_signals"]["graph_answer_strategy"] == "cite_direct_relationships"
    assert "Strategy: cite_direct_relationships" in result["context_text"]
    assert "GraphRAG --[stores_graph_in]--> Neo4j" in result["context_text"]
    assert result["services_used"]["graph_reasoning"] is True


def test_graph_reasoning_ranks_stronger_paths_over_traversal_order():
    context = load_graphrag_route_module("context")
    node_map = {
        "alpha": {"name": "Alpha", "id": "a", "type": "concept"},
        "weak bridge": {"name": "Weak Bridge", "id": "b", "type": "concept"},
        "reliable bridge": {"name": "Reliable Bridge", "id": "c", "type": "concept"},
        "omega": {"name": "Omega", "id": "d", "type": "concept"},
    }
    raw_edges = [
        {"source": "Alpha", "target": "Weak Bridge", "label": "related_to", "weight": 0.1},
        {"source": "Weak Bridge", "target": "Omega", "label": "related_to", "weight": 0.2},
        {"source": "Alpha", "target": "Reliable Bridge", "label": "supports", "weight": 0.95},
        {"source": "Reliable Bridge", "target": "Omega", "label": "supports", "weight": 0.9},
    ]

    paths = context._find_graph_paths("Alpha", "Omega", node_map, raw_edges, query="How does Alpha support Omega?")

    assert paths[0]["nodes"] == ["Alpha", "Reliable Bridge", "Omega"]
    assert paths[0]["score"] > paths[1]["score"]


def test_graph_reasoning_prefers_semantic_chain_for_concept_connection_questions():
    context = load_graphrag_route_module("context")
    query = (
        "How does Product Revenue connect to Regional Demand and Risk Map in this deck? "
        "Cite the slide, chart, or table evidence."
    )
    node_map = {
        "deck": {"name": "revenue-architecture.pptx", "id": "deck", "type": "document"},
        "slide": {"name": "Slide 1: Revenue Architecture", "id": "slide", "type": "slide"},
        "section": {"name": "Revenue Architecture", "id": "section", "type": "section"},
        "product": {"name": "Product Revenue", "id": "product", "type": "metric"},
        "demand": {"name": "Regional Demand", "id": "demand", "type": "market_signal"},
        "risk": {"name": "Risk Map", "id": "risk", "type": "visual"},
    }
    raw_edges = [
        {"source": "Slide 1: Revenue Architecture", "target": "Product Revenue", "label": "contains_text", "weight": 0.68},
        {"source": "Slide 1: Revenue Architecture", "target": "Risk Map", "label": "contains_link", "weight": 0.68},
        {"source": "Revenue Architecture", "target": "Product Revenue", "label": "contains_shape", "weight": 0.64},
        {"source": "Revenue Architecture", "target": "Risk Map", "label": "contains_image", "weight": 0.64},
        {
            "source": "Product Revenue",
            "target": "Regional Demand",
            "label": "drives",
            "weight": 0.72,
            "evidence": "Product Revenue -> Regional Demand: drives",
            "metadata": {"slide": 1, "slide_title": "Revenue Architecture", "element_type": "shape_connector"},
        },
        {
            "source": "Regional Demand",
            "target": "Risk Map",
            "label": "references",
            "weight": 0.72,
            "evidence": "Regional Demand -> Risk Map: references",
            "metadata": {"slide": 1, "slide_title": "Revenue Architecture", "element_type": "shape_connector"},
        },
    ]

    pairs = context._entity_pairs_from_query(query)
    paths = []
    for source, target in pairs:
        paths.extend(context._find_graph_paths(source, target, node_map, raw_edges, query=query))
    paths = sorted(paths, key=lambda path: context._graph_path_rank_key(path, query), reverse=True)

    assert ("Product Revenue", "Regional Demand") in pairs
    assert ("Product Revenue", "Risk Map") in pairs
    assert paths[0]["nodes"] == ["Product Revenue", "Regional Demand", "Risk Map"]
    assert [edge["relation"] for edge in paths[0]["relationships"]] == ["drives", "references"]
    assert paths[0]["path_type"] == "semantic"
    assert paths[0]["relationships"][1]["source"] == "Regional Demand"
    assert paths[0]["relationships"][1]["target"] == "Risk Map"


def test_graph_reasoning_infers_entity_pairs_from_domain_relation_verbs():
    context = load_graphrag_route_module("context")

    pairs = context._entity_pairs_from_query("How does Product Revenue drive Regional Demand?")

    assert pairs[0] == ("Product Revenue", "Regional Demand")


def test_graph_mention_pair_prefers_full_node_phrase_over_prefix_match():
    context = load_graphrag_route_module("context")
    node_map = {
        "product revenue": {"name": "Product Revenue", "id": "product", "type": "metric"},
        "regional demand": {"name": "Regional Demand", "id": "demand", "type": "market_signal"},
        "region": {"name": "Region", "id": "region", "type": "column"},
    }
    raw_edges = [
        {
            "source": "Product Revenue",
            "target": "Regional Demand",
            "label": "drives",
            "weight": 0.91,
        },
        {
            "source": "Product Revenue",
            "target": "Region",
            "label": "has_column",
            "weight": 0.52,
        },
    ]

    pairs = context._entity_pairs_from_graph_mentions(
        "Which relationship connects Product Revenue to Regional Demand?",
        node_map,
        raw_edges,
    )

    assert pairs[0] == ("Product Revenue", "Regional Demand")


def test_graph_mentions_suppress_partial_entity_words_inside_longer_mentions():
    context = load_graphrag_route_module("context")
    node_map = {
        "product": {"name": "Product", "id": "product-word", "type": "concept"},
        "revenue": {"name": "Revenue", "id": "revenue-word", "type": "metric"},
        "risk": {"name": "Risk", "id": "risk-word", "type": "concept"},
        "product revenue": {"name": "Product Revenue", "id": "product-revenue", "type": "metric"},
        "regional demand": {"name": "Regional Demand", "id": "regional-demand", "type": "signal"},
        "risk map": {"name": "Risk Map", "id": "risk-map", "type": "visual"},
    }
    raw_edges = [
        {"source": "Product", "target": "Revenue", "label": "co_occurs_with"},
        {"source": "Regional Demand", "target": "Risk", "label": "co_occurs_with"},
        {"source": "Product Revenue", "target": "Regional Demand", "label": "drives"},
        {"source": "Regional Demand", "target": "Risk Map", "label": "references"},
    ]

    pairs = context._entity_pairs_from_graph_mentions(
        "How does Product Revenue connect to Regional Demand and Risk Map in this deck?",
        node_map,
        raw_edges,
        limit=6,
    )

    assert pairs == [
        ("Product Revenue", "Regional Demand"),
        ("Regional Demand", "Risk Map"),
        ("Product Revenue", "Risk Map"),
    ]


def test_graph_reasoning_entity_names_suppress_nested_query_mentions():
    context = load_graphrag_route_module("context")

    names = context._clean_graph_reasoning_entity_names(
        "How does Product Revenue connect to Regional Demand and Risk Map in this deck?",
        [
            "Regional",
            "Product Revenue",
            "Regional Demand",
            "Risk",
            "Risk Map",
        ],
    )

    assert names == ["Product Revenue", "Regional Demand", "Risk Map"]


def test_graph_reasoning_entity_names_strip_uploaded_document_context_clause():
    context = load_graphrag_route_module("context")

    names = context._clean_graph_reasoning_entity_names(
        "How does Product Revenue connect to Regional Demand and Risk Map in this uploaded document?",
        [
            "Product Revenue",
            "Regional Demand",
            "Risk Map in this uploaded document",
        ],
    )

    assert names == ["Product Revenue", "Regional Demand", "Risk Map"]


def test_graph_answer_plan_promotes_semantic_relationships_over_weak_mentions():
    context = load_graphrag_route_module("context")
    result = context._finalize_graph_reasoning_result({
        "paths": [],
        "direct_relationships": [
            {
                "source": "Revenue",
                "relation": "co_occurs_with",
                "target": "Regional Demand",
                "relationship_type": "semantic",
                "score": 0.98,
                "weight": 0.98,
                "matched_query_terms": ["revenue", "regional", "demand"],
            },
            {
                "source": "Regional Demand",
                "relation": "references",
                "target": "Risk Map",
                "relationship_type": "semantic",
                "score": 0.62,
                "weight": 0.62,
                "matched_query_terms": ["regional", "demand", "references", "risk", "map"],
                "specific_matched_query_terms": ["regional", "demand", "references", "risk", "map"],
                "evidence": "Regional Demand -> Risk Map: references",
                "metadata": {
                    "local_graph_filename": "revenue-architecture.pptx",
                    "slide": 1,
                    "slide_title": "Revenue Architecture",
                    "element_type": "connector",
                },
            },
        ],
        "bridge_entities": [],
        "reasoning_needed": True,
    })

    plan = result["answer_plan"]
    supporting_cards = [
        card
        for card in plan["evidence_cards"]
        if card.get("role") == "direct_relationship"
    ]

    assert result["direct_relationships"][0]["relation"] == "references"
    assert plan["supporting_relationships"][0]["relation"] == "references"
    assert supporting_cards[0]["relation"] == "references"
    assert supporting_cards[0]["target"] == "Risk Map"


def test_graph_answer_plan_prefers_path_covering_query_nodes_over_short_direct_hop():
    context = load_graphrag_route_module("context")
    result = context._finalize_graph_reasoning_result({
        "query": "How does Approval connect to Finance Review Board and Risk Map in this PDF?",
        "paths": [
            {
                "source": "Finance Review Board",
                "target": "Risk Map",
                "nodes": ["Finance Review Board", "Risk Map"],
                "score": 0.95,
                "relationships": [
                    {
                        "source": "Finance Review Board",
                        "relation": "co_occurs_with",
                        "target": "Risk Map",
                        "score": 0.95,
                        "evidence": "Risk Map highlights regional blockers before release.",
                        "metadata": {"filename": "llama-nexus-rich.pdf", "page": 1},
                    },
                ],
            },
            {
                "source": "Approval",
                "target": "Risk Map",
                "nodes": ["Approval", "Finance Review Board", "Risk Map"],
                "score": 0.63,
                "relationships": [
                    {
                        "source": "Approval",
                        "relation": "co_occurs_with",
                        "target": "Finance Review Board",
                        "score": 0.64,
                        "evidence": "Finance Review Board owns approval evidence for the launch.",
                        "metadata": {"filename": "llama-nexus-rich.pdf", "page": 1},
                    },
                    {
                        "source": "Finance Review Board",
                        "relation": "co_occurs_with",
                        "target": "Risk Map",
                        "score": 0.64,
                        "evidence": "Risk Map highlights regional blockers before release.",
                        "metadata": {"filename": "llama-nexus-rich.pdf", "page": 1},
                    },
                ],
            },
        ],
        "direct_relationships": [],
        "bridge_entities": [],
        "reasoning_needed": True,
    })

    plan = result["answer_plan"]
    assert plan["strategy"] == "walk_strongest_path"
    assert plan["primary_path"]["nodes"] == ["Approval", "Finance Review Board", "Risk Map"]
    assert [hop["source"] for hop in plan["hops"]] == ["Approval", "Finance Review Board"]


def test_graph_answer_plan_prioritizes_subject_facts_over_document_artifacts():
    context = load_graphrag_route_module("context")
    result = context._finalize_graph_reasoning_result({
        "paths": [],
        "direct_relationships": [
            {
                "source": "graphrag-complex-office-source.pptx",
                "relation": "contains_chart",
                "target": "Regional Revenue Forecast",
                "relationship_type": "layout",
                "score": 0.99,
                "weight": 0.99,
                "matched_query_terms": ["regional", "revenue", "forecast"],
                "specific_matched_query_terms": ["regional", "revenue", "forecast"],
                "evidence": "Document contains Regional Revenue Forecast.",
                "metadata": {
                    "local_graph_filename": "graphrag-complex-office-source.pptx",
                    "slide": 1,
                    "slide_title": "Regional Revenue Forecast",
                    "element_type": "chart",
                },
            },
            {
                "source": "Revenue",
                "relation": "has_trendline",
                "target": "Revenue linear trendline",
                "relationship_type": "layout",
                "score": 0.97,
                "weight": 0.97,
                "matched_query_terms": ["revenue"],
                "specific_matched_query_terms": ["revenue"],
                "evidence": "Revenue has a linear trendline.",
                "metadata": {
                    "local_graph_filename": "graphrag-complex-office-source.pptx",
                    "slide": 1,
                    "slide_title": "Regional Revenue Forecast",
                    "element_type": "chart_trendline",
                    "chart_trendline_type": "linear",
                },
            },
            {
                "source": "Revenue",
                "relation": "references",
                "target": "ppt/embeddings/chartWorkbook1.xlsx",
                "relationship_type": "semantic",
                "score": 0.96,
                "weight": 0.96,
                "matched_query_terms": ["revenue", "forecast"],
                "specific_matched_query_terms": ["revenue", "forecast"],
                "evidence": "Revenue series uses the embedded chart workbook.",
                "metadata": {
                    "local_graph_filename": "graphrag-complex-office-source.pptx",
                    "chart_source_part": "ppt/embeddings/chartWorkbook1.xlsx",
                    "element_type": "chart_series",
                },
            },
            {
                "source": "Llama Nexus Logo",
                "relation": "links_to",
                "target": "https://example.com/brand-kit",
                "relationship_type": "semantic",
                "score": 0.98,
                "weight": 0.98,
                "matched_query_terms": ["revenue", "regional", "forecast"],
                "specific_matched_query_terms": ["revenue", "regional", "forecast"],
                "evidence": "Logo links to the external brand kit.",
                "metadata": {
                    "local_graph_filename": "graphrag-complex-office-source.pptx",
                    "uri": "https://example.com/brand-kit",
                    "element_type": "link",
                },
            },
            {
                "source": "Product Revenue",
                "relation": "drives",
                "target": "Regional Demand",
                "relationship_type": "semantic",
                "score": 0.7,
                "weight": 0.7,
                "matched_query_terms": ["product", "revenue", "regional", "demand"],
                "specific_matched_query_terms": ["product", "revenue", "regional", "demand"],
                "evidence": "Connector states Product Revenue drives Regional Demand.",
                "metadata": {
                    "local_graph_filename": "graphrag-complex-office-source.pptx",
                    "slide": 1,
                    "slide_title": "Regional Revenue Forecast",
                    "element_type": "connector",
                },
            },
        ],
        "bridge_entities": [],
        "reasoning_needed": True,
    })

    plan = result["answer_plan"]
    direct_cards = [
        card
        for card in plan["evidence_cards"]
        if card.get("role") == "direct_relationship"
    ]
    later_relationships = {
        (relationship.get("source"), relationship.get("relation"), relationship.get("target"))
        for relationship in result["direct_relationships"][1:]
    }

    assert result["direct_relationships"][0]["source"] == "Product Revenue"
    assert result["direct_relationships"][0]["relation"] == "drives"
    assert result["direct_relationships"][0]["target"] == "Regional Demand"
    assert plan["supporting_relationships"][0]["source"] == "Product Revenue"
    assert plan["supporting_relationships"][0]["target"] == "Regional Demand"
    assert direct_cards[0]["source"] == "Product Revenue"
    assert direct_cards[0]["target"] == "Regional Demand"
    assert ("graphrag-complex-office-source.pptx", "contains_chart", "Regional Revenue Forecast") in later_relationships
    assert ("Revenue", "has_trendline", "Revenue linear trendline") in later_relationships
    assert ("Revenue", "references", "ppt/embeddings/chartWorkbook1.xlsx") in later_relationships
    assert ("Llama Nexus Logo", "links_to", "https://example.com/brand-kit") in later_relationships


def test_graph_answer_plan_uses_exact_direct_relationship_before_paths():
    context = load_graphrag_route_module("context")
    result = context._finalize_graph_reasoning_result({
        "entity_pairs": [
            {"source": "Product Revenue", "target": "Regional Demand"},
        ],
        "paths": [
            {
                "source": "Product Revenue",
                "target": "Regional Demand",
                "nodes": ["Product Revenue", "Slide 1: Revenue Architecture", "Regional Demand"],
                "hops": 2,
                "score": 0.9,
                "path_type": "layout",
                "relationships": [
                    {
                        "source": "Product Revenue",
                        "relation": "contains_text",
                        "target": "Slide 1: Revenue Architecture",
                        "relationship_type": "layout",
                        "provenance": "slide 1 shape Product Revenue",
                    },
                    {
                        "source": "Slide 1: Revenue Architecture",
                        "relation": "contains_text",
                        "target": "Regional Demand",
                        "relationship_type": "layout",
                        "provenance": "slide 1 shape Regional Demand",
                    },
                ],
            },
        ],
        "direct_relationships": [
            {
                "source": "Regional Demand",
                "relation": "references",
                "target": "Risk Map",
                "relationship_type": "semantic",
                "score": 0.95,
                "evidence": "Regional Demand references the Risk Map.",
            },
            {
                "source": "Product Revenue",
                "relation": "drives",
                "target": "Regional Demand",
                "relationship_type": "semantic",
                "score": 0.7,
                "evidence": "Connector states Product Revenue drives Regional Demand.",
                "metadata": {
                    "local_graph_filename": "strategy.pptx",
                    "slide": 1,
                    "slide_title": "Revenue Architecture",
                    "element_type": "shape_connector",
                    "connector_title": "drives",
                },
            },
        ],
        "bridge_entities": [],
        "reasoning_needed": True,
    })

    plan = result["answer_plan"]
    context_text = context._graph_answer_plan_context(plan)

    assert plan["strategy"] == "cite_direct_relationships"
    assert plan["primary_path"] is None
    assert plan["supporting_relationships"][0]["source"] == "Product Revenue"
    assert plan["supporting_relationships"][0]["relation"] == "drives"
    assert plan["supporting_relationships"][0]["target"] == "Regional Demand"
    assert len(plan["supporting_relationships"]) == 1
    assert plan["evidence_cards"][0]["role"] == "direct_relationship"
    assert plan["evidence_cards"][0]["source"] == "Product Revenue"
    assert plan["evidence_cards"][0]["target"] == "Regional Demand"
    assert not any(card.get("target") == "Risk Map" for card in plan["evidence_cards"])
    assert "direct source-target graph relationship first" in context_text
    assert "Product Revenue --[drives]--> Regional Demand" in context_text
    assert "Regional Demand --[references]--> Risk Map" not in context_text
    assert "Primary path:" not in context_text


def test_graph_answer_plan_context_includes_table_row_location():
    context = load_graphrag_route_module("context")
    result = context._finalize_graph_reasoning_result({
        "entity_pairs": [
            {"source": "Approval Gate", "target": "Launch Workflow"},
        ],
        "paths": [],
        "direct_relationships": [
            {
                "source": "Approval Gate",
                "relation": "gates",
                "target": "Launch Workflow",
                "relationship_type": "semantic",
                "score": 0.95,
                "evidence": "Launch requires approval before release",
                "provenance": "local pipeline launch.csv; table source semantic_table_row; element table_row; row 2",
                "metadata": {
                    "local_graph_filename": "launch.csv",
                    "element_type": "table_row",
                    "row": 2,
                },
            },
        ],
        "bridge_entities": [],
        "reasoning_needed": True,
    })

    context_text = context._graph_answer_plan_context(result["answer_plan"])

    assert "Approval Gate --[gates]--> Launch Workflow" in context_text
    assert "source: launch.csv, row 2" in context_text


def test_graph_answer_plan_keeps_path_evidence_focused_on_path_nodes():
    context = load_graphrag_route_module("context")
    primary_hop = {
        "source": "Revenue",
        "relation": "has_trendline",
        "target": "Revenue linear trendline",
        "relationship_type": "layout",
        "weight": 0.68,
        "evidence": "Revenue trend trendline; shows equation; shows R-squared.",
        "metadata": {
            "local_graph_run_id": "run-revenue",
            "local_graph_filename": "strategy.pptx",
            "slide": 1,
            "slide_title": "Revenue Architecture",
            "chart_title": "Regional Revenue Forecast",
            "chart_part": "ppt/charts/chart1.xml",
            "element_type": "chart_trendline",
        },
    }
    path = {
        "source": "Revenue",
        "target": "Revenue linear trendline",
        "nodes": ["Revenue", "Revenue linear trendline"],
        "hops": 1,
        "score": 0.78,
        "path_type": "layout",
        "relationships": [primary_hop],
    }
    path.update(context._graph_path_explanation(path))

    result = context._finalize_graph_reasoning_result({
        "paths": [path],
        "direct_relationships": [
            {
                "source": "Revenue",
                "relation": "q1_value",
                "target": "12",
                "relationship_type": "semantic",
                "score": 0.77,
                "evidence": "Revenue Q1: 12",
                "metadata": {
                    "local_graph_run_id": "run-revenue",
                    "local_graph_filename": "strategy.pptx",
                    "slide": 1,
                    "slide_title": "Revenue Architecture",
                    "chart_title": "Regional Revenue Forecast",
                    "element_type": "chart_point",
                },
            },
            {
                "source": "Forecast Risk",
                "relation": "source_owner",
                "target": "Ada Lovelace",
                "relationship_type": "semantic",
                "score": 0.9,
                "evidence": "Forecast Risk owner: Ada Lovelace",
                "metadata": {
                    "local_graph_run_id": "run-revenue",
                    "local_graph_filename": "strategy.pptx",
                    "slide": 1,
                    "slide_title": "Revenue Architecture",
                    "element_type": "table_row",
                },
            },
        ],
        "bridge_entities": [],
        "interaction_handoffs": [
            {
                "run_id": "run-revenue",
                "filename": "strategy.pptx",
                "graph_anchors": [
                    {
                        "kind": "path",
                        "title": "Revenue -> Unrelated dashboard",
                        "nodes": ["Revenue", "Unrelated dashboard"],
                        "score": 0.99,
                    },
                    {
                        "kind": "path",
                        "title": "Revenue -> Regional Revenue Forecast -> Revenue linear trendline",
                        "source": "Revenue",
                        "target": "Revenue linear trendline",
                        "nodes": ["Revenue", "Regional Revenue Forecast", "Revenue linear trendline"],
                        "score": 0.5,
                    },
                ],
            },
        ],
        "reasoning_needed": True,
    })
    plan = result["answer_plan"]
    context_text = context._graph_answer_plan_context(plan)

    assert plan["strategy"] == "walk_strongest_path"
    assert plan["supporting_relationships"][0]["source"] == "Revenue"
    assert plan["supporting_relationships"][0]["relation"] == "q1_value"
    assert not any(
        relationship.get("source") == "Forecast Risk"
        for relationship in plan["supporting_relationships"]
    )
    assert plan["evidence_cards"][0]["role"] == "primary_path_hop"
    assert plan["evidence_cards"][1]["role"] == "supporting_relationship"
    handoff_cards = [
        card
        for card in plan["evidence_cards"]
        if card.get("role") == "interaction_handoff_anchor"
    ]
    assert handoff_cards[0]["text"] == "Revenue -> Regional Revenue Forecast -> Revenue linear trendline"
    assert "Primary path nodes: Revenue -> Revenue linear trendline" in context_text
    assert "one-hop graph path/direct edge" in context_text
    assert "Forecast Risk" not in context_text


def test_graph_overview_prioritizes_subject_facts_for_broad_document_summaries():
    context = load_graphrag_route_module("context")
    node_map = {
        "deck": {"name": "revenue-architecture.pptx", "id": "deck", "type": "document"},
        "chart": {"name": "Regional Revenue Forecast", "id": "chart", "type": "chart"},
        "trendline": {"name": "Revenue linear trendline", "id": "trendline", "type": "chart_trendline"},
        "logo": {"name": "Llama Nexus Logo", "id": "logo", "type": "image"},
        "brand": {"name": "https://example.com/brand-kit", "id": "brand", "type": "uri"},
        "product": {"name": "Product Revenue", "id": "product", "type": "metric"},
        "demand": {"name": "Regional Demand", "id": "demand", "type": "market_signal"},
    }
    raw_edges = [
        {
            "source": "revenue-architecture.pptx",
            "target": "Regional Revenue Forecast",
            "label": "contains_chart",
            "weight": 0.99,
            "evidence": "The deck contains the Regional Revenue Forecast chart.",
            "metadata": {
                "local_graph_run_id": "run-revenue",
                "local_graph_filename": "revenue-architecture.pptx",
                "slide": 1,
                "slide_title": "Revenue Architecture",
                "element_type": "chart",
            },
        },
        {
            "source": "Revenue",
            "target": "Revenue linear trendline",
            "label": "has_trendline",
            "weight": 0.97,
            "evidence": "Revenue has a linear trendline.",
            "metadata": {
                "local_graph_run_id": "run-revenue",
                "local_graph_filename": "revenue-architecture.pptx",
                "slide": 1,
                "slide_title": "Revenue Architecture",
                "element_type": "chart_trendline",
            },
        },
        {
            "source": "Llama Nexus Logo",
            "target": "https://example.com/brand-kit",
            "label": "links_to",
            "weight": 0.98,
            "evidence": "Logo links to the external brand kit.",
            "metadata": {
                "local_graph_run_id": "run-revenue",
                "local_graph_filename": "revenue-architecture.pptx",
                "uri": "https://example.com/brand-kit",
                "element_type": "link",
            },
        },
        {
            "source": "Product Revenue",
            "target": "Regional Demand",
            "label": "drives",
            "weight": 0.7,
            "evidence": "Connector states Product Revenue drives Regional Demand.",
            "metadata": {
                "local_graph_run_id": "run-revenue",
                "local_graph_filename": "revenue-architecture.pptx",
                "slide": 1,
                "slide_title": "Revenue Architecture",
                "element_type": "connector",
            },
        },
    ]

    overview = context._pipeline_graph_overview(
        node_map,
        raw_edges,
        sources=[{"run_id": "run-revenue", "filename": "revenue-architecture.pptx"}],
        query="Summarize the key evidence in this upload.",
    )
    graph_reasoning = context._finalize_graph_reasoning_result({
        "paths": [],
        "direct_relationships": [],
        "bridge_entities": [],
        "graph_overview": overview,
        "reasoning_needed": False,
    })
    plan = graph_reasoning["answer_plan"]

    assert overview["top_relationships"][0]["source"] == "Product Revenue"
    assert overview["top_relationships"][0]["relation"] == "drives"
    assert overview["top_relationships"][0]["target"] == "Regional Demand"
    assert plan["strategy"] == "compare_graph_overview"
    assert plan["supporting_relationships"][0]["source"] == "Product Revenue"
    assert plan["supporting_relationships"][0]["target"] == "Regional Demand"
    assert plan["evidence_cards"][0]["source"] == "Product Revenue"
    assert plan["evidence_cards"][0]["target"] == "Regional Demand"


def test_graph_reasoning_prefers_layout_paths_for_location_questions():
    context = load_graphrag_route_module("context")
    node_map = {
        "deck": {"name": "Strategy Deck", "id": "deck", "type": "document"},
        "slide": {"name": "Slide 1: Revenue Architecture", "id": "slide", "type": "slide"},
        "risk-map": {"name": "Risk Map", "id": "risk-map", "type": "image"},
    }
    raw_edges = [
        {"source": "Strategy Deck", "target": "Risk Map", "label": "related_to", "weight": 0.75},
        {"source": "Strategy Deck", "target": "Slide 1: Revenue Architecture", "label": "has_slide", "weight": 0.65},
        {"source": "Slide 1: Revenue Architecture", "target": "Risk Map", "label": "contains_image", "weight": 0.65},
    ]

    paths = context._find_graph_paths(
        "Strategy Deck",
        "Risk Map",
        node_map,
        raw_edges,
        query="Which slide contains the Risk Map?",
    )

    assert paths[0]["nodes"] == ["Strategy Deck", "Slide 1: Revenue Architecture", "Risk Map"]
    assert paths[0]["path_type"] == "layout"
    assert {"slide", "contains"} & set(paths[0]["matched_query_terms"])
    assert paths[0]["score"] > paths[1]["score"]


def test_graph_reasoning_prefers_shape_metadata_paths_for_accessibility_questions():
    context = load_graphrag_route_module("context")
    node_map = {
        "deck": {"name": "Strategy Deck", "id": "deck", "type": "document"},
        "slide": {"name": "Slide 1: Revenue Architecture", "id": "slide", "type": "slide"},
        "shape": {"name": "Risk Narrative", "id": "shape", "type": "shape"},
    }
    raw_edges = [
        {"source": "Strategy Deck", "target": "Risk Narrative", "label": "related_to", "weight": 0.8},
        {"source": "Strategy Deck", "target": "Slide 1: Revenue Architecture", "label": "has_slide", "weight": 0.66},
        {
            "source": "Slide 1: Revenue Architecture",
            "target": "Risk Narrative",
            "label": "contains_shape",
            "weight": 0.67,
            "properties": {
                "shape_title": "Risk Narrative",
                "shape_description": "Callout shape describing regional launch blockers",
                "element_type": "shape_metadata",
                "location": "slide 1 shape 4",
            },
        },
    ]

    paths = context._find_graph_paths(
        "Strategy Deck",
        "Risk Narrative",
        node_map,
        raw_edges,
        query="Which slide contains the shape description about regional launch blockers?",
    )
    graph_reasoning = {"paths": paths, "direct_relationships": []}
    terms = context._graph_augmented_retrieval_terms(graph_reasoning, limit=20)

    assert paths[0]["nodes"] == ["Strategy Deck", "Slide 1: Revenue Architecture", "Risk Narrative"]
    assert paths[0]["path_type"] == "layout"
    assert {"shape", "description", "regional", "launch", "blockers"} & set(paths[0]["matched_query_terms"])
    assert paths[0]["score"] > paths[1]["score"]
    assert any("regional launch blockers" in term for term in terms)


def test_graph_reasoning_prefers_connector_paths_for_arrow_questions():
    context = load_graphrag_route_module("context")
    connector_label = "Product Revenue -> Regional Demand: drives"
    node_map = {
        "deck": {"name": "Strategy Deck", "id": "deck", "type": "document"},
        "slide": {"name": "Slide 1: Revenue Architecture", "id": "slide", "type": "slide"},
        "connector": {"name": connector_label, "id": "connector", "type": "connector"},
    }
    raw_edges = [
        {"source": "Strategy Deck", "target": connector_label, "label": "related_to", "weight": 0.8},
        {"source": "Strategy Deck", "target": "Slide 1: Revenue Architecture", "label": "has_slide", "weight": 0.66},
        {
            "source": "Slide 1: Revenue Architecture",
            "target": connector_label,
            "label": "contains_connector",
            "weight": 0.67,
            "properties": {
                "connector_title": "drives",
                "connector_description": "Product revenue drives regional demand",
                "source_label": "Product Revenue",
                "target_label": "Regional Demand",
                "element_type": "shape_connector",
                "location": "slide 1 connector 1",
            },
        },
    ]

    paths = context._find_graph_paths(
        "Strategy Deck",
        connector_label,
        node_map,
        raw_edges,
        query="Which slide contains the arrow connecting Product Revenue to Regional Demand?",
    )
    graph_reasoning = {"paths": paths, "direct_relationships": []}
    terms = context._graph_augmented_retrieval_terms(graph_reasoning, limit=20)

    assert paths[0]["nodes"] == ["Strategy Deck", "Slide 1: Revenue Architecture", connector_label]
    assert paths[0]["path_type"] == "layout"
    assert {"slide", "arrow", "connecting", "product", "revenue", "regional", "demand"} & set(paths[0]["matched_query_terms"])
    assert paths[0]["score"] > paths[1]["score"]
    assert any("Product Revenue" in term for term in terms)
    assert any("Regional Demand" in term for term in terms)


def test_graph_reasoning_finds_chart_value_paths_from_pipeline_edges():
    context = load_graphrag_route_module("context")
    node_map = {
        "revenue architecture chart 1": {"name": "Revenue Architecture chart 1", "id": "chart-1", "type": "chart"},
        "revenue q2": {"name": "Revenue Q2", "id": "point-q2", "type": "chart_point"},
        "18": {"name": "18", "id": "value-18", "type": "metric_value"},
    }
    raw_edges = [
        {
            "source": "Revenue Architecture chart 1",
            "target": "Revenue Q2",
            "label": "contains_chart_point",
            "weight": 0.72,
            "evidence": "Revenue; Q2: 18",
            "metadata": {
                "local_graph_filename": "strategy.pptx",
                "slide": 1,
                "slide_title": "Revenue Architecture",
                "chart_part": "ppt/charts/chart1.xml",
                "chart_label": "Revenue Architecture chart 1",
                "chart_point_label": "Revenue Q2",
                "category": "Q2",
                "value": "18",
                "element_type": "chart_point",
            },
        },
        {
            "source": "Revenue Q2",
            "target": "18",
            "label": "has_value",
            "weight": 0.72,
            "evidence": "Revenue; Q2: 18",
            "metadata": {
                "local_graph_filename": "strategy.pptx",
                "slide": 1,
                "slide_title": "Revenue Architecture",
                "chart_part": "ppt/charts/chart1.xml",
                "chart_label": "Revenue Architecture chart 1",
                "chart_point_label": "Revenue Q2",
                "category": "Q2",
                "value": "18",
                "element_type": "chart_point",
            },
        },
    ]

    paths = context._find_graph_paths(
        "Revenue Architecture chart 1",
        "18",
        node_map,
        raw_edges,
        query="What value does the Revenue Architecture chart show for Q2?",
    )
    graph_reasoning = {
        "paths": paths,
        "direct_relationships": [],
        "bridge_entities": context._bridge_entities_from_paths(paths),
        "reasoning_needed": True,
    }
    finalized = context._finalize_graph_reasoning_result(graph_reasoning)

    assert paths[0]["nodes"] == ["Revenue Architecture chart 1", "Revenue Q2", "18"]
    assert paths[0]["path_type"] == "layout"
    assert {"chart", "q2", "value"} & set(paths[0]["matched_query_terms"])
    assert finalized["answer_plan"]["strategy"] == "walk_strongest_path"
    assert finalized["answer_plan"]["evidence_cards"][0]["location"]["chart_label"] == "Revenue Architecture chart 1"
    assert finalized["answer_plan"]["evidence_cards"][0]["location"]["chart_point_label"] == "Revenue Q2"
    assert "slide 1: Revenue Architecture" in finalized["answer_plan"]["evidence_cards"][0]["citation_label"]


def test_graph_reasoning_cites_powerpoint_diagram_provenance():
    context = load_graphrag_route_module("context")
    node_map = {
        "strategy.pptx": {"name": "strategy.pptx", "id": "deck", "type": "document"},
        "launch plan": {"name": "Launch Plan", "id": "diagram-root", "type": "diagram_node"},
        "risk review": {"name": "Risk Review", "id": "diagram-risk", "type": "diagram_node"},
    }
    raw_edges = [
        {
            "source": "strategy.pptx",
            "target": "Launch Plan",
            "label": "contains_diagram_node",
            "weight": 0.76,
            "evidence": "SmartArt node Launch Plan in ppt/diagrams/data1.xml.",
            "metadata": {
                "local_graph_filename": "strategy.pptx",
                "slide": 1,
                "slide_title": "Revenue Architecture",
                "diagram_part": "ppt/diagrams/data1.xml",
                "diagram_node_id": "root",
                "diagram_node_type": "doc",
                "diagram_order": 1,
                "diagram_level": 0,
                "element_type": "diagram_node",
                "location": "slide 1 diagram data1 node 1",
            },
        },
        {
            "source": "Launch Plan",
            "target": "Risk Review",
            "label": "contains_diagram_relationship",
            "weight": 0.78,
            "evidence": "Launch Plan contains Risk Review in SmartArt.",
            "metadata": {
                "local_graph_filename": "strategy.pptx",
                "slide": 1,
                "slide_title": "Revenue Architecture",
                "diagram_part": "ppt/diagrams/data1.xml",
                "diagram_connection_type": "parOf",
                "source_diagram_node_id": "root",
                "target_diagram_node_id": "risk",
                "source_label": "Launch Plan",
                "target_label": "Risk Review",
                "element_type": "diagram_relationship",
                "location": "slide 1 diagram data1 connection 1",
            },
        },
    ]

    paths = context._find_graph_paths(
        "strategy.pptx",
        "Risk Review",
        node_map,
        raw_edges,
        query="Which SmartArt diagram connection links Launch Plan to Risk Review?",
    )
    graph_reasoning = {
        "paths": paths,
        "direct_relationships": [],
        "bridge_entities": context._bridge_entities_from_paths(paths),
        "reasoning_needed": True,
    }
    finalized = context._finalize_graph_reasoning_result(graph_reasoning)
    terms = context._graph_augmented_retrieval_terms(finalized, limit=40)

    assert paths[0]["nodes"] == ["strategy.pptx", "Launch Plan", "Risk Review"]
    assert paths[0]["path_type"] == "layout"
    assert {"smartart", "diagram", "connection", "launch", "risk"} & set(paths[0]["matched_query_terms"])
    assert finalized["answer_plan"]["strategy"] == "walk_strongest_path"

    node_card = finalized["answer_plan"]["evidence_cards"][0]
    edge_card = finalized["answer_plan"]["evidence_cards"][1]
    assert node_card["location"]["diagram_node_id"] == "root"
    assert node_card["location"]["diagram_level"] == 0
    assert edge_card["location"]["diagram_part"] == "ppt/diagrams/data1.xml"
    assert edge_card["location"]["diagram_connection_type"] == "parOf"
    assert edge_card["location"]["diagram_source"] == "Launch Plan"
    assert edge_card["location"]["diagram_target"] == "Risk Review"
    assert edge_card["location"]["source_diagram_node_id"] == "root"
    assert edge_card["location"]["target_diagram_node_id"] == "risk"
    assert "diagram edge Launch Plan to Risk Review type parOf part ppt/diagrams/data1.xml" in edge_card["citation_label"]
    assert "diagram node root node type doc order 1 level 0 part ppt/diagrams/data1.xml" in node_card["citation_label"]
    assert any("ppt/diagrams/data1.xml" in term for term in terms)
    assert any("parOf" in term for term in terms)


def test_grouped_visual_edges_rank_for_visual_support_queries():
    context = load_graphrag_route_module("context")
    query = (
        "What is the approval gate relationship in this document, "
        "and which grouped visual supports it? Cite the source evidence."
    )
    graph_context = [
        {
            "source": "Approval",
            "relation": "q3_value",
            "target": "31",
            "weight": 0.88,
            "evidence": "Approval; Q3: 31",
            "metadata": {
                "element_type": "chart_point",
                "slide": 1,
                "slide_title": "Launch Approval Packet",
                "chart_label": "Revenue Plan",
            },
        },
        {
            "source": "Approval required",
            "relation": "relation",
            "target": "gates Launch workflow",
            "weight": 0.83,
            "evidence": "Source: Approval required; Relation: gates Launch workflow",
            "metadata": {
                "element_type": "table_row",
                "slide": 1,
                "slide_title": "Launch Approval Packet",
                "table_index": 1,
            },
        },
        {
            "source": "Approval Dependency Cluster",
            "relation": "groups_visual",
            "target": "Approval required before launch workflow.",
            "weight": 0.68,
            "evidence": "Approval Dependency Cluster groups Approval required before launch workflow., Launch workflow, gates",
            "metadata": {
                "element_type": "shape_group",
                "slide": 1,
                "slide_title": "Launch Approval Packet",
                "group_label": "Approval Dependency Cluster",
                "group_index": 1,
                "location": "slide 1 group 1",
            },
        },
        {
            "source": "Approval required before launch workflow.",
            "relation": "gates",
            "target": "Launch workflow",
            "weight": 0.72,
            "evidence": "Approval required before launch workflow. -> Launch workflow: gates",
            "metadata": {
                "element_type": "shape_connector",
                "slide": 1,
                "slide_title": "Launch Approval Packet",
                "parent_group_label": "Approval Dependency Cluster",
                "location": "slide 1 connector 1",
            },
        },
    ]

    ranked = context._rank_chunk_graph_context_for_query(graph_context, query, limit=4)

    assert ranked[0]["relation"] == "groups_visual"
    assert ranked[0]["source"] == "Approval Dependency Cluster"
    assert ranked[1]["relation"] == "gates"
    assert ranked[0]["query_relevance_score"] > ranked[2]["query_relevance_score"]


def test_local_pipeline_context_prefers_grouped_visual_chunks():
    context = load_graphrag_route_module("context")
    from modules.graphrag.pipeline_runs import get_pipeline_run_store

    query = (
        "What is the approval gate relationship in this document, "
        "and which grouped visual supports it? Cite the source evidence."
    )
    store = get_pipeline_run_store()
    store.upsert({
        "run_id": "run-grouped-visual",
        "document": {
            "filename": "launch-approval.pptx",
            "domain": "general",
        },
        "local_context": {
            "filename": "launch-approval.pptx",
            "domain": "general",
            "chunks": [
                {
                    "chunk_index": 0,
                    "content": "Revenue Plan shows Approval Q3 value 31 from approval.csv.",
                    "metadata": {
                        "provenance": ["slide 1: Launch Approval Packet", "chart Revenue Plan"],
                    },
                    "graph_context": [
                        {
                            "source": "Approval",
                            "relation": "q3_value",
                            "target": "31",
                            "weight": 0.88,
                            "evidence": "Approval; Q3: 31",
                            "metadata": {
                                "element_type": "chart_point",
                                "slide": 1,
                                "slide_title": "Launch Approval Packet",
                            },
                        },
                    ],
                },
                {
                    "chunk_index": 1,
                    "content": "Approval Dependency Cluster groups Approval required before launch workflow, Launch workflow, and gates.",
                    "metadata": {
                        "provenance": ["slide 1: Launch Approval Packet", "group Approval Dependency Cluster"],
                    },
                    "graph_context": [
                        {
                            "source": "Approval Dependency Cluster",
                            "relation": "groups_visual",
                            "target": "Approval required before launch workflow.",
                            "weight": 0.68,
                            "evidence": "Approval Dependency Cluster groups Approval required before launch workflow., Launch workflow, gates",
                            "metadata": {
                                "element_type": "shape_group",
                                "slide": 1,
                                "slide_title": "Launch Approval Packet",
                                "group_label": "Approval Dependency Cluster",
                                "location": "slide 1 group 1",
                            },
                        },
                        {
                            "source": "Approval required before launch workflow.",
                            "relation": "gates",
                            "target": "Launch workflow",
                            "weight": 0.72,
                            "evidence": "Approval required before launch workflow. -> Launch workflow: gates",
                            "metadata": {
                                "element_type": "shape_connector",
                                "slide": 1,
                                "slide_title": "Launch Approval Packet",
                                "parent_group_label": "Approval Dependency Cluster",
                                "location": "slide 1 connector 1",
                            },
                        },
                    ],
                },
            ],
        },
    })

    chunks = context._local_pipeline_context_chunks(
        query,
        "general",
        2,
        pipeline_run_ids=["run-grouped-visual"],
        user_query=query,
    )

    assert chunks[0]["chunk_index"] == 1
    assert chunks[0]["graph_context"][0]["relation"] == "groups_visual"
    assert "local_pipeline_grouped_visual_context" in chunks[0]["retrieval_reasons"]
    assert chunks[0]["metadata"]["grouped_visual_context_count"] == 2


def test_graph_reasoning_cites_markdown_code_block_provenance():
    context = load_graphrag_route_module("context")
    code_label = "python code block 2: def build_kg():"
    node_map = {
        "graphrag strategy": {"name": "GraphRAG Strategy", "id": "section", "type": "section"},
        code_label.lower(): {"name": code_label, "id": "code", "type": "code_block"},
        "python": {"name": "python", "id": "language", "type": "programming_language"},
    }
    code_metadata = {
        "local_graph_filename": "strategy.md",
        "section": "GraphRAG Strategy",
        "section_label": "GraphRAG Strategy",
        "markdown_code_language": "python",
        "markdown_code_block_index": 2,
        "markdown_start_line": 18,
        "markdown_end_line": 21,
        "element_type": "code_block",
        "location": "lines 18-21",
    }
    raw_edges = [
        {
            "source": "GraphRAG Strategy",
            "target": code_label,
            "label": "contains_code_block",
            "weight": 0.78,
            "evidence": 'def build_kg(): return "relationships"',
            "metadata": code_metadata,
        },
        {
            "source": code_label,
            "target": "python",
            "label": "uses_language",
            "weight": 0.74,
            "evidence": 'def build_kg(): return "relationships"',
            "metadata": code_metadata,
        },
    ]

    paths = context._find_graph_paths(
        "GraphRAG Strategy",
        "python",
        node_map,
        raw_edges,
        query="Where is the Python code block that builds the knowledge graph?",
    )
    graph_reasoning = {
        "paths": paths,
        "direct_relationships": [],
        "bridge_entities": context._bridge_entities_from_paths(paths),
        "reasoning_needed": True,
    }
    finalized = context._finalize_graph_reasoning_result(graph_reasoning)
    terms = context._graph_augmented_retrieval_terms(finalized, limit=40)

    assert paths[0]["nodes"] == ["GraphRAG Strategy", code_label, "python"]
    assert paths[0]["path_type"] == "layout"
    assert "code_block" in paths[0]["evidence_families"]
    assert {"python", "code", "block"} & set(paths[0]["matched_query_terms"])
    assert finalized["answer_plan"]["strategy"] == "walk_strongest_path"
    assert "code_block" in finalized["answer_plan"]["primary_path"]["evidence_families"]

    code_card = finalized["answer_plan"]["evidence_cards"][0]
    language_card = finalized["answer_plan"]["evidence_cards"][1]
    assert code_card["location"]["section"] == "GraphRAG Strategy"
    assert code_card["location"]["markdown_code_language"] == "python"
    assert code_card["location"]["markdown_code_block_index"] == 2
    assert code_card["location"]["markdown_start_line"] == 18
    assert code_card["location"]["markdown_end_line"] == 21
    assert "code block python #2 lines 18-21" in code_card["citation_label"]
    assert "section GraphRAG Strategy" in code_card["citation_label"]
    assert "code_block" in code_card["evidence_families"]
    assert "code_block" in language_card["evidence_families"]
    assert any("python" == term for term in terms)
    assert any("GraphRAG Strategy" == term for term in terms)
    assert any("build kg" in term for term in terms)


def test_graph_reasoning_walks_cross_document_overview_paths_with_citations():
    context = load_graphrag_route_module("context")
    node_map = {
        "alpha plan": {"name": "Alpha Plan", "id": "alpha", "type": "document"},
        "shared metric": {"name": "Shared Metric", "id": "metric", "type": "metric"},
        "beta forecast": {"name": "Beta Forecast", "id": "beta", "type": "document"},
    }
    raw_edges = [
        {
            "source": "Alpha Plan",
            "target": "Shared Metric",
            "label": "supports",
            "weight": 0.91,
            "evidence": "Alpha plan supports the shared metric.",
            "metadata": {
                "local_graph_run_id": "run-alpha",
                "local_graph_filename": "alpha.md",
                "section": "Alpha Plan",
                "element_type": "paragraph",
                "location": "line 3",
            },
        },
        {
            "source": "Beta Forecast",
            "target": "Shared Metric",
            "label": "explains",
            "weight": 0.83,
            "evidence": "Beta forecast explains the shared metric.",
            "metadata": {
                "local_graph_run_id": "run-beta",
                "local_graph_filename": "beta.md",
                "section": "Beta Forecast",
                "element_type": "paragraph",
                "location": "line 7",
            },
        },
    ]

    overview = context._pipeline_graph_overview(
        node_map,
        raw_edges,
        sources=[
            {"run_id": "run-alpha", "filename": "alpha.md"},
            {"run_id": "run-beta", "filename": "beta.md"},
        ],
        query="How do Alpha Plan and Beta Forecast share the metric?",
    )
    graph_reasoning = context._finalize_graph_reasoning_result({
        "paths": [],
        "direct_relationships": [],
        "bridge_entities": [],
        "graph_overview": overview,
        "reasoning_needed": True,
    })
    context_text = context._graph_answer_plan_context(graph_reasoning["answer_plan"])

    path = overview["cross_document_paths"][0]
    assert path["nodes"] == ["Alpha Plan", "Shared Metric", "Beta Forecast"]
    assert path["path_type"] == "semantic"
    assert path["evidence_families"] == ["semantic"]
    assert "Shared Metric connects Alpha Plan and Beta Forecast" in path["explanation"]
    assert set(path["filenames"]) == {"alpha.md", "beta.md"}
    assert path["relationships"][0]["evidence_families"] == ["semantic"]
    assert path["relationships"][0]["provenance_summary"] == "file=alpha.md"
    assert path["relationships"][1]["provenance_summary"] == "file=beta.md"

    answer_plan = graph_reasoning["answer_plan"]
    assert answer_plan["strategy"] == "compare_graph_overview"
    assert answer_plan["primary_path"]["nodes"] == ["Alpha Plan", "Shared Metric", "Beta Forecast"]
    assert answer_plan["primary_path"]["explanation"] == path["explanation"]
    assert [hop["relation"] for hop in answer_plan["hops"]] == ["supports", "explains"]
    assert answer_plan["hops"][0]["provenance"] == "local pipeline alpha.md; section Alpha Plan; element paragraph; line 3"
    assert answer_plan["hops"][1]["provenance"] == "local pipeline beta.md; section Beta Forecast; element paragraph; line 7"
    assert answer_plan["evidence_cards"][0]["role"] == "overview_path_hop"
    assert answer_plan["evidence_cards"][0]["citation_label"].startswith("local pipeline alpha.md")
    assert answer_plan["evidence_cards"][1]["citation_label"].startswith("local pipeline beta.md")
    assert "Ordered hops:" in context_text
    assert "Evidence checklist:" in context_text
    assert "Alpha Plan --[supports]--> Shared Metric" in context_text
    assert "Beta Forecast --[explains]--> Shared Metric" in context_text


def test_graph_reasoning_cites_notes_markup_styled_text_and_media_provenance():
    context = load_graphrag_route_module("context")
    node_map = {
        "slide": {"name": "Slide 1: Revenue Architecture", "id": "slide", "type": "slide"},
        "notes": {"name": "Budget risk is unresolved", "id": "notes", "type": "speaker_notes"},
        "comment": {"name": "Review launch risk", "id": "comment", "type": "comment"},
        "term": {"name": "Launch Risk", "id": "term", "type": "emphasis"},
        "media": {"name": "narration.mp4", "id": "media", "type": "media"},
    }
    raw_edges = [
        {
            "source": "Slide 1: Revenue Architecture",
            "target": "Budget risk is unresolved",
            "label": "contains_speaker_notes",
            "weight": 0.77,
            "evidence": "Budget risk is unresolved; ask Finance for final approval.",
            "metadata": {
                "local_graph_filename": "strategy.pptx",
                "slide": 1,
                "slide_title": "Revenue Architecture",
                "notes_part": "ppt/notesSlides/notesSlide1.xml",
                "notes_relationship_id": "rIdNotes1",
                "element_type": "speaker_notes",
                "location": "slide 1 speaker notes",
            },
        },
        {
            "source": "Slide 1: Revenue Architecture",
            "target": "Review launch risk",
            "label": "contains_comment",
            "weight": 0.74,
            "evidence": "Ada Reviewer: Review launch risk with Finance.",
            "metadata": {
                "local_graph_filename": "strategy.pptx",
                "slide": 1,
                "slide_title": "Revenue Architecture",
                "comment_id": "cm-7",
                "comment_kind": "threaded",
                "comment_part": "ppt/comments/comment1.xml",
                "author": "Ada Reviewer",
                "date": "2026-01-12T09:30:00Z",
                "comment_anchor_text": "Launch Risk",
                "element_type": "comment",
                "location": "slide 1 comment 1",
            },
        },
        {
            "source": "Slide 1: Revenue Architecture",
            "target": "Launch Risk",
            "label": "emphasizes",
            "weight": 0.69,
            "evidence": "Launch Risk appears in bold underlined text.",
            "metadata": {
                "local_graph_filename": "strategy.pptx",
                "slide": 1,
                "slide_title": "Revenue Architecture",
                "styled_text": True,
                "emphasis_term": "Launch Risk",
                "emphasis_styles": ["bold", "underline"],
                "style_summary": "bold; underline; font 18pt",
                "element_type": "slide_text",
                "location": "slide 1 text 3",
            },
        },
        {
            "source": "Slide 1: Revenue Architecture",
            "target": "narration.mp4",
            "label": "contains_media",
            "weight": 0.68,
            "evidence": "Video narration attached to the launch slide.",
            "metadata": {
                "local_graph_filename": "strategy.pptx",
                "slide": 1,
                "slide_title": "Revenue Architecture",
                "media_part": "ppt/media/media1.mp4",
                "media_kind": "video",
                "media_mime": "video/mp4",
                "media_size_bytes": 12048,
                "relationship_id": "rIdMedia1",
                "element_type": "media",
                "location": "slide 1 video",
            },
        },
    ]
    direct_relationships = [
        context._normalize_graph_edge(edge, node_map)
        for edge in raw_edges
    ]
    graph_reasoning = {
        "paths": [],
        "direct_relationships": direct_relationships,
        "bridge_entities": [],
        "reasoning_needed": False,
    }
    finalized = context._finalize_graph_reasoning_result(graph_reasoning)
    terms = context._graph_augmented_retrieval_terms(finalized, limit=60)
    cards = {
        card["relation"]: card
        for card in finalized["answer_plan"]["evidence_cards"]
    }

    assert finalized["answer_plan"]["strategy"] == "cite_direct_relationships"
    assert cards["contains_speaker_notes"]["location"]["notes_part"] == "ppt/notesSlides/notesSlide1.xml"
    assert cards["contains_speaker_notes"]["location"]["notes_relationship_id"] == "rIdNotes1"
    assert "speaker notes part ppt/notesSlides/notesSlide1.xml relationship rIdNotes1" in cards["contains_speaker_notes"]["citation_label"]

    assert cards["contains_comment"]["location"]["comment_id"] == "cm-7"
    assert cards["contains_comment"]["location"]["comment_author"] == "Ada Reviewer"
    assert cards["contains_comment"]["location"]["comment_anchor"] == "Launch Risk"
    assert "comment cm-7 by Ada Reviewer at 2026-01-12T09:30:00Z anchor Launch Risk part ppt/comments/comment1.xml" in cards["contains_comment"]["citation_label"]

    assert cards["emphasizes"]["location"]["emphasis_term"] == "Launch Risk"
    assert cards["emphasizes"]["location"]["emphasis_styles"] == ["bold", "underline"]
    assert "styled text emphasis Launch Risk styles bold, underline summary bold; underline; font 18pt" in cards["emphasizes"]["citation_label"]

    assert cards["contains_media"]["location"]["media_part"] == "ppt/media/media1.mp4"
    assert cards["contains_media"]["location"]["media_kind"] == "video"
    assert cards["contains_media"]["location"]["media_mime"] == "video/mp4"
    assert "media video part ppt/media/media1.mp4 video/mp4 12048 bytes" in cards["contains_media"]["citation_label"]

    assert any("Ada Reviewer" in term for term in terms)
    assert any("ppt/notesSlides/notesSlide1.xml" in term for term in terms)
    assert any("bold; underline; font 18pt" in term for term in terms)
    assert any("ppt/media/media1.mp4" in term for term in terms)


def test_graph_reasoning_cites_email_message_provenance():
    context = load_graphrag_route_module("context")
    node_map = {
        "launch.eml": {"name": "launch.eml", "id": "message-file", "type": "document"},
        "body": {"name": "Launch evidence packet is approved", "id": "body", "type": "email_body"},
        "ada": {"name": "Ada Lovelace <ada@example.com>", "id": "ada", "type": "person"},
        "grace": {"name": "Grace Hopper <grace@example.com>", "id": "grace", "type": "person"},
        "finance": {"name": "Finance Review <finance@example.com>", "id": "finance", "type": "mailbox"},
        "attachment": {"name": "launch-evidence.csv", "id": "attachment", "type": "embedded_object"},
    }
    base_metadata = {
        "local_graph_filename": "launch.eml",
        "email_subject": "GraphRAG Launch Evidence",
        "email_message_id": "<launch-evidence@example.com>",
        "email_from": "Ada Lovelace <ada@example.com>",
        "email_to": "Grace Hopper <grace@example.com>",
        "email_cc": "Finance Review <finance@example.com>",
        "email_date": "Tue, 14 Jul 2026 09:30:00 -0400",
        "email_date_iso": "2026-07-14T09:30:00-04:00",
        "element_type": "email_body",
    }
    raw_edges = [
        {
            "source": "launch.eml",
            "target": "Ada Lovelace <ada@example.com>",
            "label": "sent_by",
            "weight": 0.82,
            "evidence": "From: Ada Lovelace <ada@example.com>",
            "metadata": {
                **base_metadata,
                "email_header": "From",
                "metadata_field": "email_from",
                "location": "email header from",
            },
        },
        {
            "source": "GraphRAG Launch Evidence",
            "target": "Grace Hopper <grace@example.com>",
            "label": "sent_to",
            "weight": 0.78,
            "evidence": "To: Grace Hopper <grace@example.com>",
            "metadata": {
                **base_metadata,
                "email_header": "To",
                "metadata_field": "email_to",
                "location": "email header to",
            },
        },
        {
            "source": "GraphRAG Launch Evidence",
            "target": "Finance Review <finance@example.com>",
            "label": "cc_to",
            "weight": 0.72,
            "evidence": "Cc: Finance Review <finance@example.com>",
            "metadata": {
                **base_metadata,
                "email_header": "Cc",
                "metadata_field": "email_cc",
                "location": "email header cc",
            },
        },
        {
            "source": "launch.eml",
            "target": "Launch evidence packet is approved",
            "label": "contains_email_body",
            "weight": 0.76,
            "evidence": "Launch evidence packet is approved for rollout.",
            "metadata": {
                **base_metadata,
                "email_part_index": 1,
                "email_part_kind": "plain",
                "email_content_type": "text/plain",
                "location": "email body part 1",
            },
        },
        {
            "source": "GraphRAG Launch Evidence",
            "target": "launch-evidence.csv",
            "label": "has_attachment",
            "weight": 0.74,
            "evidence": "Attached launch-evidence.csv with final approvals.",
            "metadata": {
                **base_metadata,
                "element_type": "embedded_object",
                "email_attachment_index": 1,
                "email_attachment_filename": "launch-evidence.csv",
                "email_attachment_mime": "text/csv",
                "embedded_file_extraction_method": "csv_table",
                "location": "email attachment 1",
            },
        },
    ]
    direct_relationships = [
        context._normalize_graph_edge(edge, node_map)
        for edge in raw_edges
    ]
    graph_reasoning = {
        "paths": [],
        "direct_relationships": direct_relationships,
        "bridge_entities": [],
        "reasoning_needed": False,
    }
    finalized = context._finalize_graph_reasoning_result(graph_reasoning)
    terms = context._graph_augmented_retrieval_terms(finalized, limit=80)
    cards = {
        card["relation"]: card
        for card in finalized["answer_plan"]["evidence_cards"]
    }

    assert finalized["answer_plan"]["strategy"] == "cite_direct_relationships"
    assert cards["sent_by"]["location"]["email_from"] == "Ada Lovelace <ada@example.com>"
    assert cards["sent_by"]["location"]["email_subject"] == "GraphRAG Launch Evidence"
    assert cards["sent_by"]["location"]["email_message_id"] == "<launch-evidence@example.com>"
    assert "message GraphRAG Launch Evidence from Ada Lovelace <ada@example.com>" in cards["sent_by"]["citation_label"]
    assert "to Grace Hopper <grace@example.com>" in cards["sent_by"]["citation_label"]
    assert "id <launch-evidence@example.com>" in cards["sent_by"]["citation_label"]

    assert "message" in cards["contains_email_body"]["evidence_families"]
    assert cards["contains_email_body"]["location"]["email_part_kind"] == "plain"
    assert cards["contains_email_body"]["location"]["email_content_type"] == "text/plain"
    assert "message body part 1 plain text/plain" in cards["contains_email_body"]["citation_label"]

    assert cards["cc_to"]["location"]["email_cc"] == "Finance Review <finance@example.com>"
    assert "cc Finance Review <finance@example.com>" in cards["cc_to"]["citation_label"]

    assert cards["has_attachment"]["location"]["email_attachment_filename"] == "launch-evidence.csv"
    assert cards["has_attachment"]["location"]["email_attachment_mime"] == "text/csv"
    assert "email attachment launch-evidence.csv text/csv index 1" in cards["has_attachment"]["citation_label"]

    assert any("Ada Lovelace" in term for term in terms)
    assert any("Grace Hopper" in term for term in terms)
    assert any("Finance Review" in term for term in terms)
    assert any("launch-evidence.csv" in term for term in terms)
    assert any("<launch-evidence@example.com>" in term for term in terms)


def test_graph_reasoning_cites_hidden_workbook_sheet_provenance():
    context = load_graphrag_route_module("context")
    node_map = {
        "scorecard.xlsx": {"name": "scorecard.xlsx", "id": "workbook", "type": "document"},
        "rawdata": {"name": "RawData", "id": "sheet-raw", "type": "sheet"},
        "hidden": {"name": "hidden", "id": "visibility-hidden", "type": "sheet_visibility"},
    }
    raw_edges = [
        {
            "source": "scorecard.xlsx",
            "target": "RawData",
            "label": "has_hidden_sheet",
            "weight": 0.78,
            "evidence": "Sheet RawData; visibility: hidden; part: xl/worksheets/sheet2.xml",
            "metadata": {
                "local_graph_filename": "scorecard.xlsx",
                "sheet": 2,
                "sheet_name": "RawData",
                "sheet_visibility": "hidden",
                "sheet_part": "xl/worksheets/sheet2.xml",
                "sheet_relationship_id": "rId2",
                "element_type": "sheet_metadata",
            },
        },
        {
            "source": "RawData",
            "target": "hidden",
            "label": "has_sheet_visibility",
            "weight": 0.78,
            "evidence": "Sheet RawData visibility: hidden",
            "metadata": {
                "local_graph_filename": "scorecard.xlsx",
                "sheet": 2,
                "sheet_name": "RawData",
                "sheet_visibility": "hidden",
                "sheet_part": "xl/worksheets/sheet2.xml",
                "sheet_relationship_id": "rId2",
                "element_type": "sheet_metadata",
            },
        },
    ]

    paths = context._find_graph_paths(
        "scorecard.xlsx",
        "hidden",
        node_map,
        raw_edges,
        query="Which hidden workbook sheet stores the source assumptions?",
    )
    graph_reasoning = {
        "paths": paths,
        "direct_relationships": [],
        "bridge_entities": context._bridge_entities_from_paths(paths),
        "reasoning_needed": True,
    }
    finalized = context._finalize_graph_reasoning_result(graph_reasoning)
    terms = context._graph_augmented_retrieval_terms(finalized, limit=20)

    assert paths[0]["nodes"] == ["scorecard.xlsx", "RawData", "hidden"]
    assert paths[0]["path_type"] == "layout"
    assert {"hidden", "sheet", "workbook"} & set(paths[0]["matched_query_terms"])
    assert finalized["answer_plan"]["strategy"] == "walk_strongest_path"
    assert finalized["answer_plan"]["evidence_cards"][0]["location"]["sheet"] == "RawData"
    assert finalized["answer_plan"]["evidence_cards"][0]["location"]["sheet_visibility"] == "hidden"
    assert finalized["answer_plan"]["evidence_cards"][0]["location"]["sheet_part"] == "xl/worksheets/sheet2.xml"
    assert "RawData visibility hidden" in finalized["answer_plan"]["evidence_cards"][0]["citation_label"]
    assert "sheet part xl/worksheets/sheet2.xml" in finalized["answer_plan"]["evidence_cards"][0]["citation_label"]
    assert any("xl/worksheets/sheet2.xml" in term for term in terms)


def test_graph_reasoning_cites_hidden_powerpoint_slide_provenance():
    context = load_graphrag_route_module("context")
    node_map = {
        "strategy.pptx": {"name": "strategy.pptx", "id": "deck", "type": "document"},
        "slide 2: hidden assumptions": {"name": "Slide 2: Hidden Assumptions", "id": "slide-hidden", "type": "slide"},
        "hidden": {"name": "hidden", "id": "visibility-hidden", "type": "slide_visibility"},
    }
    raw_edges = [
        {
            "source": "strategy.pptx",
            "target": "Slide 2: Hidden Assumptions",
            "label": "has_hidden_slide",
            "weight": 0.78,
            "evidence": "Slide 2: Hidden Assumptions; visibility: hidden; part: ppt/slides/slide2.xml",
            "metadata": {
                "local_graph_filename": "strategy.pptx",
                "slide": 2,
                "slide_title": "Hidden Assumptions",
                "slide_visibility": "hidden",
                "slide_part": "ppt/slides/slide2.xml",
                "slide_relationship_id": "rIdSlide2",
                "element_type": "slide_metadata",
            },
        },
        {
            "source": "Slide 2: Hidden Assumptions",
            "target": "hidden",
            "label": "has_slide_visibility",
            "weight": 0.78,
            "evidence": "Slide 2 Hidden Assumptions visibility: hidden",
            "metadata": {
                "local_graph_filename": "strategy.pptx",
                "slide": 2,
                "slide_title": "Hidden Assumptions",
                "slide_visibility": "hidden",
                "slide_part": "ppt/slides/slide2.xml",
                "slide_relationship_id": "rIdSlide2",
                "element_type": "slide_metadata",
            },
        },
    ]

    paths = context._find_graph_paths(
        "strategy.pptx",
        "hidden",
        node_map,
        raw_edges,
        query="Which hidden PowerPoint slide stores confidential assumptions?",
    )
    graph_reasoning = {
        "paths": paths,
        "direct_relationships": [],
        "bridge_entities": context._bridge_entities_from_paths(paths),
        "reasoning_needed": True,
    }
    finalized = context._finalize_graph_reasoning_result(graph_reasoning)
    terms = context._graph_augmented_retrieval_terms(finalized, limit=20)

    assert paths[0]["nodes"] == ["strategy.pptx", "Slide 2: Hidden Assumptions", "hidden"]
    assert paths[0]["path_type"] == "layout"
    assert {"hidden", "slide", "powerpoint"} & set(paths[0]["matched_query_terms"])
    assert finalized["answer_plan"]["strategy"] == "walk_strongest_path"
    assert finalized["answer_plan"]["evidence_cards"][0]["location"]["slide"] == 2
    assert finalized["answer_plan"]["evidence_cards"][0]["location"]["slide_visibility"] == "hidden"
    assert finalized["answer_plan"]["evidence_cards"][0]["location"]["slide_part"] == "ppt/slides/slide2.xml"
    assert "slide 2: Hidden Assumptions visibility hidden" in finalized["answer_plan"]["evidence_cards"][0]["citation_label"]
    assert "slide part ppt/slides/slide2.xml" in finalized["answer_plan"]["evidence_cards"][0]["citation_label"]
    assert any("ppt/slides/slide2.xml" in term for term in terms)


def test_graph_reasoning_cites_spreadsheet_pivot_lineage():
    context = load_graphrag_route_module("context")
    node_map = {
        "scorecard": {"name": "Scorecard", "id": "sheet-scorecard", "type": "sheet"},
        "qualitypivot": {"name": "QualityPivot", "id": "pivot-quality", "type": "pivot_table"},
        "averagescore": {"name": "Average Score", "id": "avg-score", "type": "measure"},
    }
    pivot_metadata = {
        "local_graph_filename": "scorecard.xlsx",
        "sheet": 1,
        "sheet_name": "Scorecard",
        "pivot_table_name": "QualityPivot",
        "pivot_table_range": "H1:J6",
        "pivot_table_source_name": "OwnerScorecard",
        "pivot_table_source_ref": "A1:C2",
        "pivot_table_source_sheet": "Scorecard",
        "pivot_table_row_fields": ["Metric"],
        "pivot_table_column_fields": ["Owner"],
        "pivot_table_data_fields": [{"name": "Average Score", "source_field": "Score", "subtotal": "average"}],
        "pivot_axis": "data",
        "pivot_subtotal": "average",
        "element_type": "pivot_table",
    }
    raw_edges = [
        {
            "source": "Scorecard",
            "target": "QualityPivot",
            "label": "contains_pivot_table",
            "weight": 0.8,
            "evidence": "Pivot table QualityPivot summarizes OwnerScorecard.",
            "metadata": pivot_metadata,
        },
        {
            "source": "QualityPivot",
            "target": "Average Score",
            "label": "aggregates",
            "weight": 0.82,
            "evidence": "QualityPivot aggregates Average Score from Score using average.",
            "metadata": pivot_metadata,
        },
    ]

    paths = context._find_graph_paths(
        "Scorecard",
        "Average Score",
        node_map,
        raw_edges,
        query="Which pivot table aggregates Average Score from OwnerScorecard?",
    )
    graph_reasoning = {
        "paths": paths,
        "direct_relationships": [],
        "bridge_entities": context._bridge_entities_from_paths(paths),
        "reasoning_needed": True,
    }
    finalized = context._finalize_graph_reasoning_result(graph_reasoning)
    terms = context._graph_augmented_retrieval_terms(finalized, limit=24)

    assert paths[0]["nodes"] == ["Scorecard", "QualityPivot", "Average Score"]
    assert paths[0]["path_type"] == "layout"
    assert {"pivot", "table", "aggregates"} & set(paths[0]["matched_query_terms"])
    assert finalized["answer_plan"]["strategy"] == "walk_strongest_path"
    assert finalized["answer_plan"]["bridge_entities"] == ["QualityPivot"]
    first_card = finalized["answer_plan"]["evidence_cards"][0]
    aggregate_card = finalized["answer_plan"]["evidence_cards"][1]
    assert first_card["location"]["pivot_table"] == "QualityPivot"
    assert first_card["location"]["pivot_table_range"] == "H1:J6"
    assert first_card["location"]["pivot_table_source"] == "OwnerScorecard"
    assert aggregate_card["location"]["pivot_subtotal"] == "average"
    assert "pivot table QualityPivot range H1:J6 source OwnerScorecard" in first_card["citation_label"]
    assert any("OwnerScorecard" in term for term in terms)
    assert any("Average Score" in term for term in terms)


@pytest.mark.asyncio
async def test_unified_chat_context_uses_recent_pipeline_graph_without_remote_url(monkeypatch):
    from modules.graphrag.config import reset_config
    from modules.graphrag.pipeline_runs import get_pipeline_run_store, reset_pipeline_run_store

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    reset_config()
    reset_pipeline_run_store()

    context = load_graphrag_route_module("context")
    connector_label = "Product Revenue -> Regional Demand: drives"
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_ingest_local_001",
        "status": "preview_only",
        "updated_at": "2026-01-15T10:00:00Z",
        "document": {"filename": "strategy.pptx", "domain": "slides"},
        "summary": {
            "element_count": 6,
            "chunk_count": 1,
            "entity_candidate_count": 3,
            "relationship_candidate_count": 2,
        },
        "structured_ingest_artifact": {
            "filename": "strategy.structured.md",
            "content": "# Structured GraphRAG Ingest: strategy.pptx\n\n[shape_connector] slide 1 connector 1: Product Revenue -> Regional Demand: drives\n[slide_text] slide 1: Regional demand depends on product revenue and the risk map.",
            "char_count": 196,
            "sha256": "abc123def4567890",
            "truncated": False,
        },
        "document_profile": {
            "evidence_profile": {
                "source_artifacts": [
                    {
                        "kind": "attachment",
                        "filename": "revenue-source.csv",
                        "mime_type": "text/csv",
                        "checksum": "csv123",
                        "parsed_child_elements": 3,
                        "parent_provenance": "slide 1 embedded data",
                    },
                ],
            },
        },
        "local_graph": {
            "filename": "strategy.pptx",
            "domain": "slides",
            "node_count": 3,
            "edge_count": 2,
            "relationship_count": 2,
            "truncated": False,
            "nodes": [
                {"id": "deck", "name": "Strategy Deck", "type": "document"},
                {"id": "slide", "name": "Slide 1: Revenue Architecture", "type": "slide"},
                {"id": "connector", "name": connector_label, "type": "connector"},
            ],
            "edges": [
                {
                    "source": "Strategy Deck",
                    "target": "Slide 1: Revenue Architecture",
                    "label": "has_slide",
                    "type": "has_slide",
                    "weight": 0.66,
                    "evidence": "Revenue Architecture",
                    "metadata": {"slide": 1, "slide_title": "Revenue Architecture", "element_type": "slide_title"},
                },
                {
                    "source": "Slide 1: Revenue Architecture",
                    "target": connector_label,
                    "label": "contains_connector",
                    "type": "contains_connector",
                    "weight": 0.67,
                    "evidence": connector_label,
                    "metadata": {
                        "slide": 1,
                        "slide_title": "Revenue Architecture",
                        "connector_title": "drives",
                        "connector_description": "Product revenue drives regional demand",
                        "source_label": "Product Revenue",
                        "target_label": "Regional Demand",
                        "element_type": "shape_connector",
                    },
                },
            ],
        },
        "local_context": {
            "filename": "strategy.pptx",
            "domain": "slides",
            "chunk_count": 1,
            "cached_chunk_count": 1,
            "truncated": False,
            "chunks": [
                {
                    "id": "chunk_0000",
                    "chunk_index": 0,
                    "content": "[shape_connector] slide 1 connector 1: Product Revenue -> Regional Demand: drives\n[slide_text] slide 1: Regional demand depends on product revenue and the risk map.",
                    "token_estimate": 32,
                    "source_elements": ["shape_connector:8", "slide_text:3"],
                    "metadata": {
                        "provenance": [
                            "slide=1; slide_title=Revenue Architecture; connector_title=drives; source_label=Product Revenue; target_label=Regional Demand"
                        ],
                    },
                },
            ],
        },
    })
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_ingest_noise_002",
        "status": "preview_only",
        "updated_at": "2026-01-15T10:05:00Z",
        "document": {"filename": "noise.pptx", "domain": "slides"},
        "local_graph": {
            "filename": "noise.pptx",
            "domain": "slides",
            "node_count": 3,
            "edge_count": 2,
            "relationship_count": 2,
            "truncated": False,
            "nodes": [
                {"id": "noise_deck", "name": "Strategy Deck", "type": "document"},
                {"id": "noise_slide", "name": "Slide 9: Noise Architecture", "type": "slide"},
                {"id": "noise_connector", "name": connector_label, "type": "connector"},
            ],
            "edges": [
                {
                    "source": "Strategy Deck",
                    "target": "Slide 9: Noise Architecture",
                    "label": "has_slide",
                    "type": "has_slide",
                    "weight": 0.66,
                    "evidence": "Noise Architecture",
                    "metadata": {"slide": 9, "slide_title": "Noise Architecture", "element_type": "slide_title"},
                },
                {
                    "source": "Slide 9: Noise Architecture",
                    "target": connector_label,
                    "label": "contains_connector",
                    "type": "contains_connector",
                    "weight": 0.67,
                    "evidence": connector_label,
                    "metadata": {
                        "slide": 9,
                        "slide_title": "Noise Architecture",
                        "connector_title": "drives",
                        "connector_description": "Product revenue drives regional demand",
                        "source_label": "Product Revenue",
                        "target_label": "Regional Demand",
                        "element_type": "shape_connector",
                    },
                },
            ],
        },
        "local_context": {
            "filename": "noise.pptx",
            "domain": "slides",
            "chunk_count": 1,
            "cached_chunk_count": 1,
            "truncated": False,
            "chunks": [
                {
                    "id": "chunk_0000",
                    "chunk_index": 0,
                    "content": "[shape_connector] slide 9 connector 1: Product Revenue -> Regional Demand: drives\n[slide_text] slide 9: Noise deck regional demand depends on product revenue.",
                    "token_estimate": 30,
                    "source_elements": ["shape_connector:8", "slide_text:3"],
                    "metadata": {
                        "provenance": [
                            "slide=9; slide_title=Noise Architecture; connector_title=drives; source_label=Product Revenue; target_label=Regional Demand"
                        ],
                    },
                },
            ],
        },
    })

    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "Which slide contains the arrow connecting Product Revenue to Regional Demand?",
        "rag_enabled": True,
        "domain": "slides",
        "pipeline_run_ids": ["graphrag_ingest_local_001"],
        "source": "Strategy Deck",
        "target": connector_label,
    })
    mock_client = Mock()

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_retrieve_rag_chunks_for_query", AsyncMock(return_value=[])),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    graph_reasoning = result["graph_reasoning"]
    assert graph_reasoning["status"] == "complete"
    assert graph_reasoning["graph_sources"]["remote_graph"] is False
    assert [source["run_id"] for source in graph_reasoning["graph_sources"]["local_pipeline_runs"]] == [
        "graphrag_ingest_local_001",
    ]
    assert graph_reasoning["graph_sources"]["local_pipeline_runs"][0]["run_id"] == "graphrag_ingest_local_001"
    assert graph_reasoning["paths"][0]["nodes"] == ["Strategy Deck", "Slide 1: Revenue Architecture", connector_label]
    assert graph_reasoning["paths"][0]["path_type"] == "layout"
    assert {"connector", "layout", "presentation_structure"}.issubset(set(graph_reasoning["paths"][0]["evidence_families"]))
    assert "Slide 1: Revenue Architecture connects Strategy Deck" in graph_reasoning["paths"][0]["explanation"]
    assert "contains_connector" in graph_reasoning["paths"][0]["explanation"]
    assert "local pipeline strategy.pptx" in graph_reasoning["paths"][0]["relationships"][0]["provenance"]
    assert graph_reasoning["answer_plan"]["strategy"] == "walk_strongest_path"
    assert graph_reasoning["answer_plan"]["primary_path"]["answer_hint"] == graph_reasoning["paths"][0]["answer_hint"]
    assert any("local pipeline strategy.pptx" in anchor for anchor in graph_reasoning["answer_plan"]["provenance_anchors"])
    assert "Noise Architecture" not in result["context_text"]
    assert all(chunk["document_id"] == "graphrag_ingest_local_001" for chunk in result["chunks"])
    assert result["chunks"][0]["document_id"] == "graphrag_ingest_local_001"
    assert "Product Revenue -> Regional Demand: drives" in result["chunks"][0]["content"]
    assert "local_pipeline" in result["chunks"][0]["retrieval_reasons"]
    assert "=== Knowledge Context ===" in result["context_text"]
    assert "Path explanation: Slide 1: Revenue Architecture connects Strategy Deck" in result["context_text"]
    assert "Evidence families: presentation_structure, layout, connector" in result["context_text"]
    assert "recent pipeline trace" in result["context_text"]
    assert "Source locations: strategy.pptx" in result["context_text"]
    assert "Provenance anchors: local pipeline strategy.pptx" not in result["context_text"]
    assert "=== Pipeline Artifact Context ===" in result["context_text"]
    assert "Structured ingest: strategy.structured.md" in result["context_text"]
    assert "Structured source excerpt:" in result["context_text"]
    assert "Product Revenue -> Regional Demand: drives" in result["context_text"]
    assert "Local graph: 3 nodes, 2 edges" in result["context_text"]
    assert "revenue-source.csv" in result["context_text"]
    assert "=== Graph Paths ===" in result["context_text"]
    assert result["pipeline_artifacts"]["run_count"] == 1
    assert result["pipeline_artifacts"]["structured_artifact_count"] == 1
    assert result["pipeline_artifacts"]["source_artifact_count"] == 1
    assert result["pipeline_artifacts"]["runs"][0]["run_id"] == "graphrag_ingest_local_001"
    assert result["pipeline_artifacts"]["runs"][0]["structured_ingest"]["filename"] == "strategy.structured.md"
    assert result["pipeline_artifacts"]["runs"][0]["manifest_endpoint"].endswith("/artifacts/manifest")
    assert result["services_used"]["graph_reasoning"] is True
    assert result["services_used"]["local_pipeline_graph"] is True
    assert result["services_used"]["local_pipeline_context"] is True
    assert result["services_used"]["local_pipeline_artifacts"] is True
    assert result["quality_signals"]["remote_graph_used"] is False
    assert result["quality_signals"]["local_pipeline_graph_run_count"] == 1
    assert result["quality_signals"]["local_pipeline_graph_edge_count"] == 2
    assert result["quality_signals"]["local_pipeline_context_chunk_count"] == 1
    assert result["quality_signals"]["local_pipeline_context_run_count"] == 1
    assert result["quality_signals"]["local_pipeline_artifact_run_count"] == 1
    assert result["quality_signals"]["local_pipeline_structured_artifact_count"] == 1
    assert result["quality_signals"]["local_pipeline_source_artifact_count"] == 1


@pytest.mark.asyncio
async def test_unified_chat_context_infers_pair_from_local_graph_node_mentions(monkeypatch):
    from modules.graphrag.config import reset_config
    from modules.graphrag.pipeline_runs import get_pipeline_run_store, reset_pipeline_run_store

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    reset_config()
    reset_pipeline_run_store()

    context = load_graphrag_route_module("context")
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_ingest_graph_mentions_001",
        "status": "preview_only",
        "updated_at": "2026-01-15T11:00:00Z",
        "document": {"filename": "strategy.pptx", "domain": "slides"},
        "local_graph": {
            "filename": "strategy.pptx",
            "domain": "slides",
            "node_count": 3,
            "edge_count": 2,
            "relationship_count": 2,
            "truncated": False,
            "nodes": [
                {"id": "product-revenue", "name": "Product Revenue", "type": "metric"},
                {"id": "demand-driver", "name": "Demand Driver", "type": "concept"},
                {"id": "regional-demand", "name": "Regional Demand", "type": "metric"},
            ],
            "edges": [
                {
                    "source": "Product Revenue",
                    "target": "Demand Driver",
                    "label": "drives",
                    "type": "drives",
                    "weight": 0.82,
                    "evidence": "Product Revenue drives the regional demand driver.",
                    "metadata": {
                        "slide": 1,
                        "slide_title": "Revenue Architecture",
                        "connector_title": "drives",
                        "source_label": "Product Revenue",
                        "target_label": "Demand Driver",
                        "element_type": "shape_connector",
                    },
                },
                {
                    "source": "Demand Driver",
                    "target": "Regional Demand",
                    "label": "supports",
                    "type": "supports",
                    "weight": 0.79,
                    "evidence": "Demand Driver supports Regional Demand planning.",
                    "metadata": {
                        "slide": 1,
                        "slide_title": "Revenue Architecture",
                        "table": 1,
                        "row": 2,
                        "element_type": "table_fact",
                    },
                },
            ],
        },
        "local_context": {
            "filename": "strategy.pptx",
            "domain": "slides",
            "cached_chunk_count": 1,
            "chunks": [
                {
                    "id": "chunk_0000",
                    "chunk_index": 0,
                    "content": "Product Revenue drives Demand Driver, which supports Regional Demand.",
                    "metadata": {"provenance": ["slide=1; slide_title=Revenue Architecture"]},
                },
            ],
        },
    })

    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "Map Product Revenue and Regional Demand using the recent pipeline trace.",
        "rag_enabled": False,
        "domain": "slides",
        "pipeline_run_ids": ["graphrag_ingest_graph_mentions_001"],
    })
    mock_client = Mock()

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    graph_reasoning = result["graph_reasoning"]
    assert graph_reasoning["status"] == "complete"
    assert graph_reasoning["inferred_entity_pairs"] == []
    assert graph_reasoning["graph_inferred_entity_pairs"][0] == {
        "source": "Product Revenue",
        "target": "Regional Demand",
    }
    assert {"source": "Product Revenue", "target": "Regional Demand"} in graph_reasoning["graph_inferred_entity_pairs"]
    assert graph_reasoning["entity_pairs"][0] == {"source": "Product Revenue", "target": "Regional Demand"}
    assert graph_reasoning["paths"][0]["nodes"] == ["Product Revenue", "Demand Driver", "Regional Demand"]
    assert graph_reasoning["bridge_entities"][0]["name"] == "Demand Driver"
    assert graph_reasoning["answer_plan"]["strategy"] == "walk_strongest_path"
    assert "Demand Driver connects Product Revenue and Regional Demand" in graph_reasoning["paths"][0]["explanation"]
    assert "Strategy: walk_strongest_path" in result["context_text"]
    assert result["chunks"][0]["document_id"] == "graphrag_ingest_graph_mentions_001"
    assert result["chunks"][0]["document_title"] == "strategy.pptx"
    assert "Product Revenue drives Demand Driver" in result["chunks"][0]["content"]
    assert "local_pipeline" in result["chunks"][0]["retrieval_reasons"]
    assert "=== Knowledge Context ===" in result["context_text"]
    assert result["quality_signals"]["local_pipeline_context_chunk_count"] == 1
    assert result["quality_signals"]["graph_path_count"] == 1
    assert result["services_used"]["local_pipeline_graph"] is True
    assert result["services_used"]["local_pipeline_context"] is True
    assert result["services_used"]["rag"] is False


def test_graph_reasoning_cites_legacy_office_stream_provenance():
    context = load_graphrag_route_module("context")
    node_map = {
        "legacy-plan.doc": {"name": "legacy-plan.doc", "id": "legacy-doc", "type": "document"},
        "launch-plan": {"name": "Northwind Legacy Launch Plan", "id": "launch-plan", "type": "concept"},
    }
    raw_edge = {
        "source": "legacy-plan.doc",
        "target": "Northwind Legacy Launch Plan",
        "label": "contains_legacy_office_text",
        "weight": 0.7,
        "evidence": "Northwind Legacy Launch Plan; Market Expansion depends on Partner Enablement.",
        "metadata": {
            "local_graph_filename": "legacy-plan.doc",
            "legacy_office_format": "word",
            "legacy_office_stream_index": 1,
            "legacy_office_stream_path": "WordDocument",
            "legacy_office_stream_size_bytes": 256,
            "legacy_office_stream_sha256": "abc123def4567890",
            "legacy_office_extraction_method": "ole_stream_printable_strings",
            "legacy_office_string_count": 3,
            "element_type": "legacy_office_text",
            "location": "legacy office stream 1 text",
        },
    }
    direct_relationships = [context._normalize_graph_edge(raw_edge, node_map)]
    finalized = context._finalize_graph_reasoning_result({
        "paths": [],
        "direct_relationships": direct_relationships,
        "bridge_entities": [],
        "reasoning_needed": False,
    })
    card = finalized["answer_plan"]["evidence_cards"][0]
    terms = context._graph_augmented_retrieval_terms(finalized, limit=40)

    assert finalized["answer_plan"]["strategy"] == "cite_direct_relationships"
    assert "legacy_office" in card["evidence_families"]
    assert card["location"]["legacy_office_format"] == "word"
    assert card["location"]["legacy_office_stream_path"] == "WordDocument"
    assert card["location"]["legacy_office_stream_sha256"] == "abc123def4567890"
    assert "legacy Office word stream WordDocument" in card["citation_label"]
    assert "via ole_stream_printable_strings" in card["citation_label"]
    assert "sha256 abc123def4567890" in card["citation_label"]
    assert any("WordDocument" in term for term in terms)
    assert any("ole stream printable strings" in term for term in terms)


@pytest.mark.asyncio
async def test_chat_context_v1_includes_scoped_pipeline_artifact_context(monkeypatch):
    from modules.graphrag.config import reset_config
    from modules.graphrag.pipeline_runs import get_pipeline_run_store, reset_pipeline_run_store

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    reset_config()
    reset_pipeline_run_store()

    context = load_graphrag_route_module("context")
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_ingest_artifact_001",
        "status": "preview_only",
        "updated_at": "2026-01-15T10:20:00Z",
        "document": {"filename": "rollout.md", "domain": "docs"},
        "summary": {"chunk_count": 1, "relationship_candidate_count": 1},
        "structured_ingest_artifact": {
            "filename": "rollout.structured.md",
            "content": "# Structured GraphRAG Ingest: rollout.md\n\nAlpha rollout depends on Beta readiness and owner approval.",
            "char_count": 96,
            "sha256": "artifactsha123456",
            "truncated": False,
        },
        "source_document_artifact": {
            "filename": "rollout.pdf",
            "content_type": "application/pdf",
            "content_base64": "JVBERi0xLjQ=",
            "size_bytes": 8,
            "sha256": "sourcesha123456",
        },
        "local_context": {
            "filename": "rollout.md",
            "domain": "docs",
            "cached_chunk_count": 1,
            "graph_context_relationship_count": 1,
            "graph_path_context_count": 0,
        },
    })

    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "Summarize Alpha rollout artifact evidence",
        "domain": "docs",
        "pipeline_run_ids": ["graphrag_ingest_artifact_001"],
    })
    mock_client = Mock()

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
    ):
        result = await context.graphrag_chat_context(request)

    assert "=== Pipeline Artifact Context ===" in result["context_text"]
    assert "rollout.structured.md" in result["context_text"]
    assert "Alpha rollout depends on Beta readiness" in result["context_text"]
    assert result["pipeline_artifacts"]["run_count"] == 1
    artifacts = {
        artifact["kind"]: artifact
        for artifact in result["pipeline_artifacts"]["runs"][0]["artifacts"]
    }
    assert artifacts["source_document"]["content_type"] == "application/pdf"
    assert artifacts["source_document"]["endpoint"].endswith("/artifacts/source_document")
    assert result["pipeline_artifacts"]["structured_artifact_count"] == 1
    assert result["pipeline_artifacts"]["source_artifact_count"] == 1
    assert result["pipeline_artifacts"]["runs"][0]["source_artifacts"][0]["kind"] == "source_document"
    assert result["pipeline_artifacts"]["runs"][0]["source_artifacts"][0]["path"].endswith("/artifacts/source_document")
    assert result["services_used"]["local_pipeline_artifacts"] is True


@pytest.mark.asyncio
async def test_unified_chat_context_promotes_query_matched_local_graph_relationships_without_remote_ner(monkeypatch):
    from modules.graphrag.config import reset_config
    from modules.graphrag.pipeline_runs import get_pipeline_run_store, reset_pipeline_run_store

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    reset_config()
    reset_pipeline_run_store()

    context = load_graphrag_route_module("context")
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_ingest_links_001",
        "status": "preview_only",
        "updated_at": "2026-01-15T11:10:00Z",
        "document": {"filename": "links-deck.pptx", "domain": "slides"},
        "local_graph": {
            "filename": "links-deck.pptx",
            "domain": "slides",
            "node_count": 4,
            "edge_count": 2,
            "relationship_count": 2,
            "truncated": False,
            "nodes": [
                {"id": "risk_map", "name": "Risk Map", "type": "image"},
                {"id": "risk_register", "name": "https://example.com/risk-register", "type": "uri"},
                {"id": "revenue_chart", "name": "Revenue Chart", "type": "chart"},
                {"id": "q2", "name": "Revenue Q2", "type": "metric"},
            ],
            "edges": [
                {
                    "source": "Risk Map",
                    "target": "https://example.com/risk-register",
                    "label": "links_to",
                    "type": "links_to",
                    "weight": 0.86,
                    "evidence": "Risk Map image action opens the risk register.",
                    "metadata": {
                        "slide": 1,
                        "slide_title": "Revenue Architecture",
                        "uri": "https://example.com/risk-register",
                        "link_kind": "click",
                        "image_part": "ppt/media/risk-map.png",
                        "image_relationship_id": "rId9",
                        "layout_bbox": "x=7.00in y=1.00in w=2.00in h=1.50in",
                        "layout_zone": "top_right",
                        "element_type": "image_link",
                    },
                },
                {
                    "source": "Revenue Chart",
                    "target": "Revenue Q2",
                    "label": "contains_chart",
                    "type": "contains_chart",
                    "weight": 0.75,
                    "evidence": "Revenue Chart contains Revenue Q2.",
                    "metadata": {
                        "slide": 1,
                        "slide_title": "Revenue Architecture",
                        "chart_label": "Revenue Architecture chart 1",
                        "element_type": "chart",
                    },
                },
            ],
        },
    })

    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "Trace links and clicked image actions in this deck",
        "rag_enabled": True,
        "domain": "slides",
        "pipeline_run_ids": ["graphrag_ingest_links_001"],
    })
    mock_client = Mock()

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_retrieve_rag_chunks_for_query", AsyncMock(return_value=[])),
        patch.object(context, "_search_graphrag", AsyncMock(return_value=([], []))),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    graph_reasoning = result["graph_reasoning"]
    direct_relationships = graph_reasoning["direct_relationships"]
    link_relationship = next(
        relationship
        for relationship in direct_relationships
        if relationship.get("relation") == "links_to"
    )
    link_card = next(
        card
        for card in graph_reasoning["answer_plan"]["evidence_cards"]
        if card.get("relation") == "links_to"
    )

    assert graph_reasoning["status"] == "complete"
    assert graph_reasoning["reasoning_needed"] is False
    assert graph_reasoning["graph_sources"]["remote_graph"] is False
    assert graph_reasoning["summary"]["mode"] == "direct_relationship"
    assert graph_reasoning["answer_plan"]["strategy"] == "cite_direct_relationships"
    assert link_relationship["target"] == "https://example.com/risk-register"
    assert {"action", "clicked", "image", "link", "links"} & set(link_relationship["matched_query_terms"])
    assert link_card["location"]["uri"] == "https://example.com/risk-register"
    assert link_card["location"]["image_relationship_id"] == "rId9"
    assert link_card["location"]["layout_zone"] == "top_right"
    assert "uri https://example.com/risk-register" in link_card["citation_label"]
    assert "image relationship rId9" in link_card["citation_label"]
    assert "=== Relevant Graph Relationships ===" in result["context_text"]
    assert "Risk Map --[links_to]--> https://example.com/risk-register" in result["context_text"]
    assert "link https://example.com/risk-register" in result["context_text"]
    assert "image relationship rId9" not in result["context_text"]
    assert result["quality_signals"]["direct_graph_relationship_count"] >= 1
    assert result["quality_signals"]["graph_answer_strategy"] == "cite_direct_relationships"
    assert result["quality_signals"]["local_pipeline_graph_run_count"] == 1
    assert result["quality_signals"]["remote_graph_used"] is False
    assert result["services_used"]["graph_reasoning"] is True
    assert result["services_used"]["local_pipeline_graph"] is True


@pytest.mark.asyncio
async def test_unified_chat_context_uses_recent_upload_graph_for_natural_relationship_question(monkeypatch):
    from modules.graphrag.config import reset_config
    from modules.graphrag.pipeline_runs import get_pipeline_run_store, reset_pipeline_run_store

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    reset_config()
    reset_pipeline_run_store()

    context = load_graphrag_route_module("context")
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_ingest_recent_revenue_001",
        "status": "complete",
        "updated_at": "2026-01-15T12:00:00Z",
        "document": {"filename": "revenue-architecture.pptx", "domain": "general"},
        "local_graph": {
            "filename": "revenue-architecture.pptx",
            "domain": "general",
            "node_count": 4,
            "edge_count": 3,
            "relationship_count": 3,
            "truncated": False,
            "nodes": [
                {"id": "product_revenue", "name": "Product Revenue", "type": "metric"},
                {"id": "regional_demand", "name": "Regional Demand", "type": "market_signal"},
                {"id": "deck", "name": "revenue-architecture.pptx", "type": "document"},
                {"id": "chart", "name": "Regional Revenue Forecast", "type": "chart"},
            ],
            "edges": [
                {
                    "source": "revenue-architecture.pptx",
                    "target": "Regional Revenue Forecast",
                    "label": "contains_chart",
                    "type": "contains_chart",
                    "weight": 0.97,
                    "evidence": "The deck contains the Regional Revenue Forecast chart.",
                    "metadata": {
                        "local_graph_filename": "revenue-architecture.pptx",
                        "slide": 1,
                        "slide_title": "Revenue Architecture",
                        "element_type": "chart",
                    },
                },
                {
                    "source": "Product Revenue",
                    "target": "Regional Demand",
                    "label": "drives",
                    "type": "drives",
                    "weight": 0.72,
                    "evidence": "Connector states Product Revenue drives Regional Demand.",
                    "metadata": {
                        "local_graph_filename": "revenue-architecture.pptx",
                        "slide": 1,
                        "slide_title": "Revenue Architecture",
                        "element_type": "connector",
                    },
                },
                {
                    "source": "Revenue",
                    "target": "Revenue linear trendline",
                    "label": "has_trendline",
                    "type": "has_trendline",
                    "weight": 0.95,
                    "evidence": "Revenue has a linear trendline.",
                    "metadata": {
                        "local_graph_filename": "revenue-architecture.pptx",
                        "slide": 1,
                        "slide_title": "Revenue Architecture",
                        "element_type": "chart_trendline",
                    },
                },
            ],
        },
    })

    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "What does Product Revenue drive in the recent upload?",
        "rag_enabled": False,
        "domain": "general",
    })
    mock_client = Mock()

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_search_graphrag", AsyncMock(return_value=([], []))),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    graph_reasoning = result["graph_reasoning"]
    first_relationship = graph_reasoning["direct_relationships"][0]
    first_card = graph_reasoning["answer_plan"]["evidence_cards"][0]

    assert graph_reasoning["status"] == "complete"
    assert graph_reasoning["graph_sources"]["remote_graph"] is False
    assert graph_reasoning["answer_plan"]["strategy"] == "cite_direct_relationships"
    assert first_relationship["source"] == "Product Revenue"
    assert first_relationship["relation"] == "drives"
    assert first_relationship["target"] == "Regional Demand"
    assert first_card["source"] == "Product Revenue"
    assert first_card["target"] == "Regional Demand"
    assert "Product Revenue --[drives]--> Regional Demand" in result["context_text"]
    assert result["quality_signals"]["local_pipeline_graph_run_count"] == 1
    assert result["services_used"]["graph_reasoning"] is True
    assert result["services_used"]["local_pipeline_graph"] is True


@pytest.mark.asyncio
async def test_unified_chat_context_uses_pipeline_interaction_handoff_without_graph_edges(monkeypatch):
    from modules.graphrag.config import reset_config
    from modules.graphrag.pipeline_runs import get_pipeline_run_store, reset_pipeline_run_store

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    reset_config()
    reset_pipeline_run_store()

    context = load_graphrag_route_module("context")
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_ingest_handoff_001",
        "status": "preview_only",
        "updated_at": "2026-01-15T11:00:00Z",
        "document": {"filename": "chart-deck.pptx", "domain": "slides"},
        "interaction_handoff": {
            "ready": True,
            "recommended_mode": "graph_path_reasoning",
            "answer_guidance": "Use graph anchors first, then support each hop with chart provenance.",
            "citation_scopes": ["slide", "chart", "table", "link"],
            "lanes": [
                {"id": "trace", "label": "Trace", "ready": True, "count": 3},
                {"id": "chat", "label": "Chat", "ready": True, "count": 2},
            ],
            "capability_cards": [
                {
                    "id": "charts",
                    "kind": "evidence",
                    "label": "Chart evidence",
                    "count": 2,
                    "detail": "Chart series and values are available as graph evidence.",
                },
            ],
            "graph_anchors": [
                {
                    "id": "handoff-chart-q2",
                    "kind": "relationship",
                    "title": "Revenue Q2 has value 18",
                    "source": "Revenue Q2",
                    "relation": "has_value",
                    "target": "18",
                    "score": 0.91,
                    "evidence": "Revenue Q2 = 18 in the Revenue Architecture chart.",
                    "metadata": {
                        "slide": 1,
                        "slide_title": "Revenue Architecture",
                        "chart_label": "Revenue Architecture chart 1",
                        "chart_point_label": "Revenue Q2",
                        "category": "Q2",
                        "value": "18",
                    },
                },
                {
                    "id": "handoff-style-launch-risk",
                    "kind": "styled_text",
                    "title": "Slide 1: Revenue Architecture --[emphasizes]--> Launch Risk",
                    "source": "Slide 1: Revenue Architecture",
                    "relation": "emphasizes",
                    "target": "Launch Risk",
                    "score": 0.87,
                    "evidence": "Launch Risk appears in bold underlined text.",
                    "metadata": {
                        "slide": 1,
                        "slide_title": "Revenue Architecture",
                        "styled_text": True,
                        "emphasis_term": "Launch Risk",
                        "emphasis_styles": ["bold", "underline"],
                        "style_summary": "bold; underline",
                        "element_type": "slide_text",
                    },
                },
                {
                    "id": "handoff-link-risk-register",
                    "kind": "link",
                    "title": "Risk Map links to risk register",
                    "source": "Risk Map",
                    "relation": "links_to",
                    "target": "https://example.com/risk-register",
                    "uri": "https://example.com/risk-register",
                    "score": 0.86,
                    "evidence": "Risk Map image action opens the risk register.",
                    "metadata": {
                        "slide": 1,
                        "slide_title": "Revenue Architecture",
                        "uri": "https://example.com/risk-register",
                        "link_kind": "click",
                        "image_part": "ppt/media/risk-map.png",
                        "image_relationship_id": "rId9",
                        "layout_bbox": "x=7.00in y=1.00in w=2.00in h=1.50in",
                        "layout_zone": "top_right",
                    },
                },
            ],
            "starter_prompts": [
                {
                    "id": "explain-chart-q2",
                    "kind": "evidence",
                    "label": "Explain Q2 chart",
                    "prompt": "Using chart-deck.pptx, explain Revenue Q2 and cite chart provenance.",
                },
            ],
            "retrieval_rehearsals": [
                {
                    "id": "rehearsal_explain_chart_q2",
                    "prompt_id": "explain-chart-q2",
                    "kind": "evidence",
                    "label": "Explain Q2 chart",
                    "prompt": "Using chart-deck.pptx, explain Revenue Q2 and cite chart provenance.",
                    "status": "ready",
                    "score": 0.88,
                    "matched_terms": ["revenue", "q2", "chart"],
                    "matched_anchor_count": 1,
                    "matched_chunk_count": 1,
                    "top_anchors": [
                        {
                            "id": "handoff-chart-q2",
                            "kind": "relationship",
                            "title": "Revenue Q2 has value 18",
                            "source": "Revenue Q2",
                            "relation": "has_value",
                            "target": "18",
                            "score": 0.91,
                            "matched_terms": ["revenue", "q2"],
                            "evidence": "Revenue Q2 = 18 in the Revenue Architecture chart.",
                            "metadata": {"chart_label": "Revenue Architecture chart 1"},
                        },
                    ],
                    "top_chunks": [
                        {
                            "chunk_id": "chunk_0000",
                            "chunk_index": 0,
                            "score": 0.83,
                            "matched_terms": ["revenue", "q2", "chart"],
                            "evidence_preview": "Slide 1 Revenue Architecture chart: Revenue Q2 has value 18.",
                            "provenance": [
                                "slide=1; slide_title=Revenue Architecture; chart_label=Revenue Architecture chart 1"
                            ],
                        },
                    ],
                },
            ],
            "counts": {
                "graph_anchors": 3,
                "capability_cards": 1,
                "starter_prompts": 1,
                "retrieval_rehearsals": 1,
                "rehearsed_chunk_matches": 1,
                "rehearsed_anchor_matches": 1,
            },
        },
        "local_context": {
            "filename": "chart-deck.pptx",
            "domain": "slides",
            "chunk_count": 1,
            "cached_chunk_count": 1,
            "truncated": False,
            "chunks": [
                {
                    "id": "chunk_0000",
                    "chunk_index": 0,
                    "content": "Slide 1 Revenue Architecture chart: Revenue Q2 has value 18.",
                    "metadata": {
                        "provenance": [
                            "slide=1; slide_title=Revenue Architecture; chart_label=Revenue Architecture chart 1"
                        ],
                    },
                },
            ],
        },
    })

    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "What should I know about the chart handoff?",
        "rag_enabled": True,
        "domain": "slides",
        "pipeline_run_ids": ["graphrag_ingest_handoff_001"],
    })
    mock_client = Mock()

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_retrieve_rag_chunks_for_query", AsyncMock(return_value=[])),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    graph_reasoning = result["graph_reasoning"]
    assert graph_reasoning["status"] == "complete"
    assert graph_reasoning["summary"]["mode"] == "interaction_handoff"
    assert graph_reasoning["interaction_handoffs"][0]["recommended_mode"] == "graph_path_reasoning"
    assert graph_reasoning["interaction_handoffs"][0]["retrieval_rehearsals"][0]["status"] == "ready"
    assert graph_reasoning["answer_plan"]["strategy"] == "use_pipeline_interaction_handoff"
    assert graph_reasoning["answer_plan"]["evidence_cards"][0]["role"] == "interaction_handoff_anchor"
    assert graph_reasoning["answer_plan"]["evidence_cards"][0]["location"]["chart_label"] == "Revenue Architecture chart 1"
    styled_card = next(
        card
        for card in graph_reasoning["answer_plan"]["evidence_cards"]
        if card.get("kind") == "styled_text"
    )
    assert styled_card["location"]["emphasis_term"] == "Launch Risk"
    assert styled_card["location"]["emphasis_styles"] == ["bold", "underline"]
    assert "styled text emphasis Launch Risk styles bold, underline summary bold; underline" in styled_card["citation_label"]
    link_card = next(
        card
        for card in graph_reasoning["answer_plan"]["evidence_cards"]
        if card.get("kind") == "link"
    )
    assert link_card["location"]["uri"] == "https://example.com/risk-register"
    assert link_card["location"]["image_relationship_id"] == "rId9"
    assert link_card["location"]["layout_zone"] == "top_right"
    assert "uri https://example.com/risk-register" in link_card["citation_label"]
    assert "=== Pipeline Interaction Handoff ===" in result["context_text"]
    assert "Retrieval rehearsals:" in result["context_text"]
    assert "Explain Q2 chart | ready | anchors: 1 | chunks: 1" in result["context_text"]
    assert "Revenue Q2 has value 18" in result["context_text"]
    assert "styled_text: Slide 1: Revenue Architecture --[emphasizes]--> Launch Risk" in result["context_text"]
    assert "styled text emphasis Launch Risk styles bold, underline summary bold; underline" in result["context_text"]
    assert "link: Risk Map links to risk register" in result["context_text"]
    assert "uri https://example.com/risk-register" in result["context_text"]
    assert "image relationship rId9" in result["context_text"]
    assert result["chunks"][0]["document_id"] == "graphrag_ingest_handoff_001"
    assert "local_pipeline" in result["chunks"][0]["retrieval_reasons"]
    assert result["quality_signals"]["local_pipeline_interaction_handoff_count"] == 1
    assert result["quality_signals"]["local_pipeline_handoff_graph_anchor_count"] == 3
    assert result["quality_signals"]["local_pipeline_handoff_starter_prompt_count"] == 1
    assert result["quality_signals"]["local_pipeline_handoff_retrieval_rehearsal_count"] == 1
    assert result["quality_signals"]["local_pipeline_handoff_rehearsed_chunk_match_count"] == 1
    assert result["quality_signals"]["local_pipeline_handoff_rehearsed_anchor_match_count"] == 1
    assert result["quality_signals"]["local_pipeline_handoff_modes"] == ["graph path reasoning"]
    assert "chart" in result["quality_signals"]["local_pipeline_handoff_citation_scopes"]
    assert "link" in result["quality_signals"]["local_pipeline_handoff_citation_scopes"]
    assert result["services_used"]["local_pipeline_interaction_handoff"] is True


@pytest.mark.asyncio
async def test_unified_chat_context_uses_structured_chunk_graph_context(monkeypatch):
    from modules.graphrag.config import reset_config
    from modules.graphrag.pipeline_runs import get_pipeline_run_store, reset_pipeline_run_store

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    reset_config()
    reset_pipeline_run_store()

    context = load_graphrag_route_module("context")
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_ingest_graph_context_001",
        "status": "preview_only",
        "updated_at": "2026-01-15T11:00:00Z",
        "document": {"filename": "strategy.pptx", "domain": "slides"},
        "local_graph": {
            "filename": "strategy.pptx",
            "domain": "slides",
            "node_count": 2,
            "edge_count": 1,
            "relationship_count": 1,
            "truncated": False,
            "nodes": [
                {"id": "revenue", "name": "Product Revenue", "type": "metric"},
                {"id": "demand", "name": "Regional Demand", "type": "market_signal"},
            ],
            "edges": [
                {
                    "source": "Product Revenue",
                    "target": "Regional Demand",
                    "label": "drives",
                    "type": "drives",
                    "weight": 0.72,
                    "evidence": "Product revenue drives regional demand",
                    "metadata": {
                        "slide": 1,
                        "slide_title": "Revenue Architecture",
                        "connector_title": "drives",
                        "source_label": "Product Revenue",
                        "target_label": "Regional Demand",
                        "element_type": "shape_connector",
                    },
                },
            ],
        },
        "local_context": {
            "filename": "strategy.pptx",
            "domain": "slides",
            "chunk_count": 1,
            "cached_chunk_count": 1,
            "graph_context_relationship_count": 1,
            "truncated": False,
            "chunks": [
                {
                    "id": "chunk_sparse",
                    "chunk_index": 0,
                    "content": "Slide 1 contains the revenue architecture summary.",
                    "token_estimate": 12,
                    "source_elements": ["slide_text:1"],
                    "graph_context": [
                        {
                            "source": "Product Revenue",
                            "relation": "drives",
                            "target": "Regional Demand",
                            "confidence": 0.72,
                            "chunk_match_score": 9,
                            "chunk_match_reasons": ["source_text", "target_text", "relation_text"],
                            "evidence": "Product revenue drives regional demand",
                            "provenance": {
                                "slide": 1,
                                "slide_title": "Revenue Architecture",
                                "connector_title": "drives",
                                "element_type": "shape_connector",
                            },
                            "text": "Product Revenue --[drives]--> Regional Demand; evidence=Product revenue drives regional demand",
                        },
                    ],
                    "metadata": {
                        "provenance": ["slide=1; slide_title=Revenue Architecture"],
                        "graph_context_relationship_count": 1,
                    },
                },
            ],
        },
    })

    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "How does Product Revenue drive Regional Demand?",
        "rag_enabled": True,
        "domain": "slides",
        "pipeline_run_ids": ["graphrag_ingest_graph_context_001"],
    })
    mock_client = Mock()

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_retrieve_rag_chunks_for_query", AsyncMock(return_value=[])),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    assert result["chunks"]
    chunk = result["chunks"][0]
    assert chunk["id"] == "pipeline:graphrag_ingest_graph_context_001:chunk_sparse"
    assert "local_pipeline_graph_context" in chunk["retrieval_reasons"]
    assert chunk["graph_context"][0]["source"] == "Product Revenue"
    assert chunk["graph_reasoning_boost"] > 0
    assert {"product", "revenue", "regional", "demand"} & set(chunk["matched_graph_path_terms"])
    assert "Graph relationships:" in result["context_text"]
    assert "Product Revenue --[drives]--> Regional Demand" in result["context_text"]
    assert "chunk match score: 9" in result["context_text"]
    assert "source_text" not in result["context_text"]
    assert result["quality_signals"]["local_pipeline_graph_context_chunk_count"] >= 1
    assert result["quality_signals"]["local_pipeline_graph_context_relationship_count"] >= 1
    assert result["quality_signals"]["local_pipeline_graph_context_matched_chunk_count"] == 1
    assert any(
        reference["graph_context_relationship_count"] == 1
        and reference["matched_chunk_graph_context_terms"]
        for reference in result["cross_references"]
    )


def test_requested_evidence_family_detection_ignores_generic_path_citation_language():
    context = load_graphrag_route_module("context")
    available_families = [
        "attachment",
        "chart",
        "connector",
        "form_field",
        "speaker_notes",
        "styled_text",
        "table",
    ]

    assert context._requested_evidence_family_keys(
        (
            "Trace the path from Approval required before launch workflow to Launch workflow "
            "through Approval Dependency Cluster. Show each hop label and cite the slide evidence."
        ),
        available_families=available_families,
    ) == []

    assert context._requested_evidence_family_keys(
        (
            "What connects Approval required, Finance Review Board, Risk Map, "
            "launch workflow, revenue impact, and approval gates in this PDF? Cite the source."
        ),
        available_families=["comment", "semantic_relationship"],
    ) == []

    assert set(context._requested_evidence_family_keys(
        "List every extracted evidence family type available in this deck.",
        available_families=available_families,
    )) >= {"attachments", "charts", "connectors", "form_fields", "speaker_notes", "styled_text", "tables"}


def test_requested_evidence_family_detection_handles_pdf_links_and_attachments():
    context = load_graphrag_route_module("context")

    assert context._requested_evidence_family_keys(
        (
            "What PDF evidence connects GraphRAG evidence, Ada Lovelace, "
            "the approval form, links, and the source data attachment?"
        ),
        available_families=["attachment", "form_field", "link"],
    ) == ["form_fields", "attachments", "links"]

    cards = context._requested_evidence_coverage_cards(
        (
            "What PDF evidence connects GraphRAG evidence, Ada Lovelace, "
            "the approval form, links, and the source data attachment?"
        ),
        [
            {
                "id": "pdf-link",
                "content": "Link refs: uri=https://example.com/graphrag",
                "metadata": {
                    "primary_evidence_family": "link",
                    "evidence_families": ["link"],
                    "filename": "annual-strategy.pdf",
                    "page": 1,
                    "semantic_text": "[link] page 1 link 1: https://example.com/graphrag",
                },
                "graph_context": [
                    {
                        "source": "Annual Strategy",
                        "relation": "contains_link",
                        "target": "https://example.com/graphrag",
                        "evidence": "https://example.com/graphrag",
                        "provenance": {
                            "page": 1,
                            "location": "page 1 link 1",
                            "uri": "https://example.com/graphrag",
                            "element_type": "link",
                        },
                    },
                ],
            },
        ],
    )

    link_cards = [card for card in cards if card.get("kind") == "links"]
    assert link_cards
    assert "Annual Strategy links to https://example.com/graphrag" in link_cards[0]["evidence"]
    assert link_cards[0]["location"]["uri"] == "https://example.com/graphrag"


@pytest.mark.asyncio
async def test_unified_chat_context_frontloads_requested_evidence_family_coverage(monkeypatch):
    from modules.graphrag.config import reset_config
    from modules.graphrag.pipeline_runs import get_pipeline_run_store, reset_pipeline_run_store

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    reset_config()
    reset_pipeline_run_store()

    context = load_graphrag_route_module("context")
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_ingest_requested_evidence_001",
        "status": "complete",
        "updated_at": "2026-01-15T11:10:00Z",
        "document": {"filename": "launch-approval.pptx", "domain": "slides"},
        "local_graph": {
            "filename": "launch-approval.pptx",
            "domain": "slides",
            "edge_count": 1,
            "edges": [
                {
                    "source": "Regional Revenue Forecast",
                    "relation": "visually_near",
                    "target": "Slide 1 table 1",
                    "evidence": "Slide 1 table 1 is below Regional Revenue Forecast.",
                    "metadata": {
                        "slide": 1,
                        "slide_title": "Launch Approval Packet",
                        "layout_relation": "below",
                        "related_label": "Slide 1 table 1",
                        "related_element_type": "table_row",
                        "visual_distance_pct": 21.3,
                        "layout_zone": "center",
                        "target_layout_zone": "bottom_center",
                        "layout_bbox": "x=1.00in y=1.00in w=4.00in h=2.00in",
                        "target_layout_bbox": "x=1.00in y=3.20in w=4.00in h=1.00in",
                        "location": "slide 1 chart 1",
                    },
                },
            ],
        },
        "local_context": {
            "filename": "launch-approval.pptx",
            "domain": "slides",
            "chunk_count": 6,
            "cached_chunk_count": 6,
            "graph_context_relationship_count": 7,
            "chunks": [
                {
                    "id": "chart",
                    "chunk_index": 0,
                    "content": "Revenue Plan chart shows Q2 value 18 and Q3 value 31.",
                    "graph_context": [
                        {
                            "source": "Launch Approval Packet",
                            "relation": "contains_chart",
                            "target": "Revenue Plan",
                            "evidence": "Revenue Plan chart with Q2 18 and Q3 31",
                            "metadata": {"slide": 1, "slide_title": "Launch Approval Packet"},
                        },
                    ],
                    "metadata": {
                        "primary_evidence_family": "chart",
                        "evidence_families": ["chart"],
                        "chart_title": "Revenue Plan",
                        "series_name": "Approval",
                        "value": "Q2 18; Q3 31",
                        "slide_title": "Launch Approval Packet",
                        "semantic_text": "Revenue Plan chart shows Q2 18 and Q3 31.",
                    },
                },
                {
                    "id": "table",
                    "chunk_index": 1,
                    "content": "approval.csv table row: Owner Finance Review Board; Decision Approved.",
                    "graph_context": [
                        {
                            "source": "Finance Review Board",
                            "relation": "decision",
                            "target": "Approved",
                            "evidence": "Finance Review Board decision Approved",
                            "metadata": {"slide": 1, "table_name": "approval.csv"},
                        },
                    ],
                    "metadata": {
                        "primary_evidence_family": "table",
                        "evidence_families": ["table"],
                        "table_name": "approval.csv",
                        "table_cell_row_header": "Owner",
                        "table_cell_column_header": "Decision",
                        "slide_title": "Launch Approval Packet",
                        "semantic_text": "Owner: Finance Review Board; Decision: Approved",
                    },
                },
                {
                    "id": "form",
                    "chunk_index": 2,
                    "content": "Approval required form field is a checked Forms.CheckBox.1 control.",
                    "graph_context": [
                        {
                            "source": "Launch Approval Packet",
                            "relation": "contains_form_field",
                            "target": "Approval required",
                            "evidence": "Approval required: checked",
                            "metadata": {"slide": 1, "field_type": "Forms.CheckBox.1"},
                        },
                    ],
                    "metadata": {
                        "primary_evidence_family": "form_field",
                        "evidence_families": ["form_field"],
                        "label": "Approval required",
                        "field_type": "Forms.CheckBox.1",
                        "form_field_checked": True,
                        "value": "1",
                        "slide_title": "Launch Approval Packet",
                        "semantic_text": "Approval required: checked",
                    },
                },
                {
                    "id": "style",
                    "chunk_index": 3,
                    "content": "Launch Approval Packet title uses bold styled text and highlight.",
                    "graph_context": [
                        {
                            "source": "Slide 1: Launch Approval Packet",
                            "relation": "emphasizes",
                            "target": "Launch Approval Packet",
                            "evidence": "Launch Approval Packet | style: bold; highlight",
                            "metadata": {"slide": 1, "style_summary": "bold; highlight"},
                        },
                    ],
                    "metadata": {
                        "primary_evidence_family": "styled_text",
                        "evidence_families": ["styled_text"],
                        "style_summary": "bold; highlight",
                        "emphasis_terms": ["Launch Approval Packet"],
                        "slide_title": "Launch Approval Packet",
                        "semantic_text": "Launch Approval Packet title uses bold styled text and highlight.",
                    },
                },
                {
                    "id": "connector",
                    "chunk_index": 4,
                    "content": "Approval Dependency arrow connects Finance Review Board to Launch Approval.",
                    "graph_context": [
                        {
                            "source": "Launch Approval Packet",
                            "relation": "contains_connector",
                            "target": "Approval Dependency arrow",
                            "evidence": "Approval Dependency arrow connects Finance Review Board to Launch Approval.",
                            "metadata": {
                                "slide": 1,
                                "connector_title": "Approval Dependency arrow",
                                "connector_description": "Finance Review Board -> Launch Approval",
                                "source_label": "Finance Review Board",
                                "target_label": "Launch Approval",
                            },
                        },
                    ],
                    "metadata": {
                        "primary_evidence_family": "connector",
                        "evidence_families": ["connector"],
                        "connector_title": "Approval Dependency arrow",
                        "connector_description": "Finance Review Board -> Launch Approval",
                        "source_label": "Finance Review Board",
                        "target_label": "Launch Approval",
                        "layout_bbox": "x=1.00in y=2.00in w=4.00in h=0.25in",
                        "slide_title": "Launch Approval Packet",
                        "semantic_text": "Approval Dependency arrow connects Finance Review Board to Launch Approval.",
                    },
                },
                {
                    "id": "review",
                    "chunk_index": 5,
                    "content": "Maya Reviewer comment and speaker note cite Finance Review Board approval evidence.",
                    "graph_context": [
                        {
                            "source": "Launch Approval Packet",
                            "relation": "contains_comment",
                            "target": "Maya Reviewer: Finance needs approval evidence before launch.",
                            "evidence": "Maya Reviewer: Finance needs approval evidence before launch.",
                            "metadata": {"slide": 1, "comment_author": "Maya Reviewer"},
                        },
                        {
                            "source": "Launch Approval Packet",
                            "relation": "contains_speaker_notes",
                            "target": "Speaker note: cite the Finance Review Board approval evidence.",
                            "evidence": "Speaker note: cite the Finance Review Board approval evidence.",
                            "metadata": {"slide": 1, "notes_part": "notesSlide1"},
                        },
                    ],
                    "metadata": {
                        "primary_evidence_family": "comment",
                        "evidence_families": ["comment", "speaker_notes"],
                        "comment_author": "Maya Reviewer",
                        "comment_text": "Finance needs approval evidence before launch.",
                        "note_text": "cite the Finance Review Board approval evidence.",
                        "slide_title": "Launch Approval Packet",
                        "semantic_text": "Maya Reviewer: Finance needs approval evidence before launch. Speaker note: cite the Finance Review Board approval evidence.",
                    },
                },
                {
                    "id": "layout",
                    "chunk_index": 6,
                    "content": "Regional Revenue Forecast sits visually near the slide evidence table.",
                    "metadata": {
                        "primary_evidence_family": "layout",
                        "evidence_families": ["layout"],
                        "local_graph_filename": "launch-approval.pptx",
                        "slide": 1,
                        "slide_title": "Launch Approval Packet",
                        "semantic_text": "Regional Revenue Forecast sits visually near the slide evidence table.",
                    },
                },
            ],
        },
    })

    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "List every extracted evidence family in this presentation: charts, tables, form fields, styled text, connectors, comments, speaker notes, and layout relationships. For each, name the specific evidence found.",
        "rag_enabled": True,
        "domain": "slides",
        "top_k": 3,
        "pipeline_run_ids": ["graphrag_ingest_requested_evidence_001"],
        "response_mode": "chat",
        "response_evidence_card_limit": 3,
    })
    mock_client = Mock()

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_retrieve_rag_chunks_for_query", AsyncMock(return_value=[])),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    context_text = result["context_text"]
    assert "=== Requested Evidence Coverage ===" in context_text
    assert "Form Fields:" in context_text
    assert "Approval required" in context_text
    assert "Forms.CheckBox.1" in context_text
    assert "Styled Text:" in context_text
    assert "bold" in context_text
    assert "Connectors:" in context_text
    assert "Approval Dependency arrow" in context_text
    assert "Finance Review Board -> Launch Approval" in context_text
    assert "Comments:" in context_text
    assert "Maya Reviewer" in context_text
    assert "Speaker Notes:" in context_text
    assert "Finance Review Board approval evidence" in context_text
    assert "Layout Relationships:" in context_text
    assert "Regional Revenue Forecast is visually near Slide 1 table 1" in context_text
    assert "distance 21.3%" in context_text
    assert "zones center -> bottom center" in context_text
    assert "Charts:" in context_text
    assert "Revenue Plan" in context_text
    assert "Tables:" in context_text
    assert "approval.csv" in context_text
    assert result["quality_signals"]["local_pipeline_requested_evidence_chunk_count"] >= 7
    assert set(result["quality_signals"]["requested_evidence_families"]) >= {
        "charts",
        "tables",
        "form_fields",
        "styled_text",
        "connectors",
        "comments",
        "speaker_notes",
        "layout",
    }
    assert result["quality_signals"]["requested_evidence_card_count"] >= 8

    answer_cards = result["graph_reasoning"]["answer_plan"]["evidence_cards"]
    requested_cards = [
        card
        for card in answer_cards
        if card.get("role") == "requested_evidence_family"
    ]
    assert [card["kind"] for card in requested_cards[:8]] == [
        "charts",
        "tables",
        "form_fields",
        "styled_text",
        "connectors",
        "comments",
        "speaker_notes",
        "layout",
    ]
    assert len(requested_cards) >= 8
    assert any(
        card.get("kind") == "form_fields"
        and "Approval required" in card.get("evidence", "")
        and "Forms.CheckBox.1" in card.get("evidence", "")
        for card in requested_cards
    )
    assert any(
        card.get("kind") == "styled_text"
        and "bold" in card.get("evidence", "")
        for card in requested_cards
    )
    assert any(
        card.get("kind") == "connectors"
        and "Approval Dependency arrow" in card.get("evidence", "")
        and "Finance Review Board" in card.get("evidence", "")
        for card in requested_cards
    )
    assert any(
        card.get("kind") == "comments"
        and "Maya Reviewer" in card.get("evidence", "")
        for card in requested_cards
    )
    assert any(
        card.get("kind") == "speaker_notes"
        and "Finance Review Board approval evidence" in card.get("evidence", "")
        for card in requested_cards
    )
    layout_cards = [
        card
        for card in requested_cards
        if card.get("kind") == "layout"
    ]
    assert layout_cards
    layout_card = layout_cards[0]
    assert "Regional Revenue Forecast is visually near Slide 1 table 1" in layout_card.get("evidence", "")
    assert "distance 21.3%" in layout_card.get("evidence", "")
    assert "zones center -> bottom center" in layout_card.get("evidence", "")
    assert layout_card.get("location", {}).get("layout_relation") == "below"
    assert layout_card.get("location", {}).get("related_label") == "Slide 1 table 1"
    assert context._is_structural_relation("visually_near")
    assert "layout" in context._graph_edge_evidence_families({
        "relation": "visually_near",
        "metadata": {"layout_relation": "below"},
    })
    assert len(result["requested_evidence_cards"]) >= 8


@pytest.mark.asyncio
async def test_unified_chat_context_prioritizes_semantic_graph_context_for_relationship_questions(monkeypatch):
    from modules.graphrag.config import reset_config
    from modules.graphrag.pipeline_runs import get_pipeline_run_store, reset_pipeline_run_store

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    reset_config()
    reset_pipeline_run_store()

    context = load_graphrag_route_module("context")
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_ingest_graph_context_rank_001",
        "status": "submitted",
        "updated_at": "2026-01-15T11:00:00Z",
        "document": {"filename": "strategy.md", "domain": "general"},
        "local_context": {
            "filename": "strategy.md",
            "domain": "general",
            "chunk_count": 1,
            "cached_chunk_count": 1,
            "graph_context_relationship_count": 5,
            "chunks": [
                {
                    "id": "chunk_relationships",
                    "chunk_index": 0,
                    "content": "Product Revenue drives Regional Demand. Regional Demand references Risk Map.",
                    "graph_context": [
                        {
                            "source": "GraphRAG UX Flow Semantic Clean",
                            "relation": "contains_table",
                            "target": "Product Revenue | 12 | 18",
                            "confidence": 0.64,
                            "evidence": "Product Revenue | 12 | 18",
                        },
                        {
                            "source": "GraphRAG UX Flow Semantic Clean",
                            "relation": "contains_text",
                            "target": "Product Revenue drives Regional Demand. Regional Demand references Risk Map.",
                            "confidence": 0.64,
                            "evidence": "Product Revenue drives Regional Demand. Regional Demand references Risk Map.",
                        },
                        {
                            "source": "Regional Demand",
                            "relation": "co_occurs_with",
                            "target": "Product Revenue",
                            "confidence": 0.54,
                            "evidence": "Product Revenue drives Regional Demand. Regional Demand references Risk Map.",
                        },
                        {
                            "source": "Slide 1: Revenue Architecture",
                            "relation": "emphasizes",
                            "target": "Product",
                            "confidence": 0.82,
                            "evidence": "Product | style: bold",
                            "provenance": {"slide": 1, "style_summary": "bold"},
                        },
                        {
                            "source": "Product Revenue",
                            "relation": "drives",
                            "target": "Regional Demand",
                            "confidence": 0.76,
                            "evidence": "Product Revenue drives Regional Demand.",
                        },
                        {
                            "source": "Revenue Architecture",
                            "relation": "contains_connector",
                            "target": "Regional Demand -> Risk Map: references",
                            "confidence": 0.64,
                            "evidence": "Regional Demand -> Risk Map: references",
                            "provenance": {
                                "source_label": "Regional Demand",
                                "target_label": "Risk Map",
                                "connector_title": "references",
                                "slide": 1,
                            },
                        },
                    ],
                    "metadata": {"graph_context_relationship_count": 4},
                },
            ],
        },
    })

    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "How does Product Revenue connect to Regional Demand, and what does Regional Demand reference?",
        "rag_enabled": True,
        "domain": "general",
        "pipeline_run_ids": ["graphrag_ingest_graph_context_rank_001"],
    })
    mock_client = Mock()

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_retrieve_rag_chunks_for_query", AsyncMock(return_value=[])),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    relations = [
        (edge["source"], edge["relation"], edge["target"])
        for edge in result["chunks"][0]["graph_context"]
    ]
    assert result["chunks"][0]["graph_context_answer_edge_count"] == 2
    assert relations[:2] == [
        ("Product Revenue", "drives", "Regional Demand"),
        ("Regional Demand", "references", "Risk Map"),
    ]
    assert ("Slide 1: Revenue Architecture", "emphasizes", "Product") not in relations[:2]
    context_text = result["context_text"]
    graph_lines = context_text.split("Graph relationships:", 1)[1].splitlines()[1:3]
    assert "Product Revenue --[drives]--> Regional Demand" in graph_lines[0]
    assert "Regional Demand --[references]--> Risk Map" in graph_lines[1]


@pytest.mark.asyncio
async def test_unified_chat_context_uses_structured_chunk_graph_paths(monkeypatch):
    from modules.graphrag.config import reset_config
    from modules.graphrag.pipeline_runs import get_pipeline_run_store, reset_pipeline_run_store

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    reset_config()
    reset_pipeline_run_store()

    context = load_graphrag_route_module("context")
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_ingest_graph_path_001",
        "status": "preview_only",
        "updated_at": "2026-01-15T11:30:00Z",
        "document": {"filename": "strategy.pptx", "domain": "slides"},
        "local_context": {
            "filename": "strategy.pptx",
            "domain": "slides",
            "chunk_count": 1,
            "cached_chunk_count": 1,
            "graph_context_relationship_count": 0,
            "graph_path_context_count": 1,
            "truncated": False,
            "chunks": [
                {
                    "id": "chunk_path",
                    "chunk_index": 0,
                    "content": "Slide 1 summarizes the revenue architecture path.",
                    "token_estimate": 10,
                    "source_elements": ["slide_text:1"],
                    "graph_paths": [
                        {
                            "source": "Product Revenue",
                            "bridge": "Regional Demand",
                            "target": "Risk Map",
                            "nodes": ["Product Revenue", "Regional Demand", "Risk Map"],
                            "hops": 2,
                            "score": 0.91,
                            "chunk_match_score": 11,
                            "chunk_match_reasons": ["source_text", "bridge_text", "target_text"],
                            "text": "Product Revenue --[drives]-- Regional Demand --[references]-- Risk Map",
                            "relationships": [
                                {"source": "Product Revenue", "relation": "drives", "target": "Regional Demand"},
                                {"source": "Regional Demand", "relation": "references", "target": "Risk Map"},
                            ],
                        },
                    ],
                    "metadata": {
                        "provenance": ["slide=1; slide_title=Revenue Architecture"],
                        "graph_path_context_count": 1,
                    },
                },
            ],
        },
    })

    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "How does Product Revenue connect to the Risk Map?",
        "rag_enabled": True,
        "domain": "slides",
        "pipeline_run_ids": ["graphrag_ingest_graph_path_001"],
    })
    mock_client = Mock()

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_retrieve_rag_chunks_for_query", AsyncMock(return_value=[])),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    assert result["chunks"]
    chunk = result["chunks"][0]
    assert chunk["id"] == "pipeline:graphrag_ingest_graph_path_001:chunk_path"
    assert "local_pipeline_graph_path_context" in chunk["retrieval_reasons"]
    assert chunk["graph_paths"][0]["bridge"] == "Regional Demand"
    assert {"product", "revenue", "risk", "map"} & set(chunk["matched_graph_path_terms"])
    assert result["context_text"].index("=== Graph Answer Plan ===") < result["context_text"].index("=== Knowledge Context ===")
    assert "Graph paths:" in result["context_text"]
    assert "Product Revenue --[drives]-- Regional Demand --[references]-- Risk Map" in result["context_text"]
    assert "chunk match score: 11" in result["context_text"]
    assert result["quality_signals"]["local_pipeline_graph_path_context_chunk_count"] >= 1
    assert result["quality_signals"]["local_pipeline_graph_path_context_count"] >= 1
    assert result["quality_signals"]["local_pipeline_graph_path_context_matched_chunk_count"] == 1
    assert any(
        reference["graph_path_context_count"] == 1
        and reference["matched_chunk_graph_path_terms"]
        for reference in result["cross_references"]
    )


@pytest.mark.asyncio
async def test_unified_chat_context_compacts_chat_response_without_losing_reference_drilldown(monkeypatch):
    from modules.graphrag.config import reset_config
    from modules.graphrag.pipeline_runs import get_pipeline_run_store, reset_pipeline_run_store

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    reset_config()
    reset_pipeline_run_store()

    context = load_graphrag_route_module("context")
    long_source_text = "Product Revenue drives Regional Demand with slide-level evidence. " * 80
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_ingest_compact_chat_001",
        "status": "preview_only",
        "updated_at": "2026-01-15T12:30:00Z",
        "document": {
            "filename": "compact-strategy.pptx",
            "domain": "slides",
            "content_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        },
        "structured_ingest_artifact": {
            "filename": "compact-strategy.structured.md",
            "content": "# Compact Strategy\n\n" + long_source_text,
            "char_count": len(long_source_text),
            "sha256": "structuredsha123",
            "truncated": False,
        },
        "source_document_artifact": {
            "filename": "compact-strategy.pptx",
            "content_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "content_base64": "A" * 4096,
            "size_bytes": 4096,
            "sha256": "sourcesha123",
        },
        "local_graph": {
            "filename": "compact-strategy.pptx",
            "domain": "slides",
            "node_count": 2,
            "edge_count": 1,
            "relationship_count": 1,
            "nodes": [
                {"id": "revenue", "name": "Product Revenue", "type": "metric"},
                {"id": "demand", "name": "Regional Demand", "type": "market_signal"},
            ],
            "edges": [
                {
                    "source": "Product Revenue",
                    "target": "Regional Demand",
                    "label": "drives",
                    "type": "drives",
                    "weight": 0.91,
                    "evidence": long_source_text,
                    "metadata": {
                        "slide": 1,
                        "slide_title": "Revenue Architecture",
                        "connector_title": "drives",
                        "source_label": "Product Revenue",
                        "target_label": "Regional Demand",
                        "element_type": "shape_connector",
                        "source_text": long_source_text,
                    },
                },
            ],
        },
        "local_context": {
            "filename": "compact-strategy.pptx",
            "domain": "slides",
            "chunk_count": 1,
            "cached_chunk_count": 1,
            "graph_context_relationship_count": 2,
            "graph_path_context_count": 2,
            "chunks": [
                {
                    "id": "chunk_compact",
                    "chunk_index": 0,
                    "content": long_source_text,
                    "graph_context": [
                        {
                            "source": "Product Revenue",
                            "relation": "drives",
                            "target": "Regional Demand",
                            "confidence": 0.91,
                            "evidence": long_source_text,
                            "provenance": {
                                "slide": 1,
                                "slide_title": "Revenue Architecture",
                                "connector_title": "drives",
                            },
                            "text": "Product Revenue --[drives]--> Regional Demand",
                        },
                        {
                            "source": "Regional Demand",
                            "relation": "references",
                            "target": "Risk Map",
                            "confidence": 0.62,
                            "evidence": "Regional Demand references Risk Map.",
                        },
                    ],
                    "graph_paths": [
                        {
                            "source": "Product Revenue",
                            "bridge": "Regional Demand",
                            "target": "Risk Map",
                            "nodes": ["Product Revenue", "Regional Demand", "Risk Map"],
                            "hops": 2,
                            "score": 0.88,
                            "text": "Product Revenue --[drives]-- Regional Demand --[references]-- Risk Map",
                            "relationships": [
                                {"source": "Product Revenue", "relation": "drives", "target": "Regional Demand"},
                                {"source": "Regional Demand", "relation": "references", "target": "Risk Map"},
                            ],
                        },
                        {
                            "source": "Product Revenue",
                            "bridge": "Launch Priority",
                            "target": "Risk Map",
                            "nodes": ["Product Revenue", "Launch Priority", "Risk Map"],
                            "hops": 2,
                            "score": 0.4,
                            "text": "Product Revenue --[supports]-- Launch Priority --[references]-- Risk Map",
                        },
                    ],
                    "metadata": {
                        "provenance": ["slide=1; slide_title=Revenue Architecture"],
                        "slide": 1,
                        "slide_title": "Revenue Architecture",
                        "source_text": long_source_text,
                        "graph_context_relationship_count": 2,
                        "graph_path_context_count": 2,
                    },
                },
            ],
            "evidence_cards": [
                {
                    "id": "card-1",
                    "role": "primary_relationship",
                    "kind": "relationship",
                    "citation_label": "compact-strategy.pptx slide 1",
                    "evidence": long_source_text,
                    "location": {"filename": "compact-strategy.pptx", "slide": 1},
                },
                {"id": "card-2", "role": "secondary", "evidence": "Extra card"},
            ],
        },
    })

    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "Which relationship connects Product Revenue to Regional Demand, and what evidence supports it?",
        "rag_enabled": True,
        "domain": "slides",
        "pipeline_run_ids": ["graphrag_ingest_compact_chat_001"],
        "source": "Product Revenue",
        "target": "Regional Demand",
        "relation": "drives",
        "response_mode": "chat",
        "context_text_max_chars": 4000,
        "response_chunk_content_chars": 420,
        "response_graph_context_limit": 1,
        "response_graph_path_limit": 1,
        "response_evidence_card_limit": 1,
    })
    mock_client = Mock()

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_retrieve_rag_chunks_for_query", AsyncMock(return_value=[])),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    assert result["quality_signals"]["response_compacted"] is True
    assert result["quality_signals"]["response_mode"] == "chat"
    assert len(result["context_text"]) <= 4000
    assert "Product Revenue --[drives]--> Regional Demand" in result["context_text"]

    chunk = result["chunks"][0]
    assert len(chunk["content"]) <= 430
    assert "Product Revenue drives Regional Demand" in chunk["content"]
    assert len(chunk["graph_context"]) == 1
    assert len(chunk["graph_paths"]) == 1
    assert "source_text" not in chunk["metadata"]
    assert chunk["metadata"]["slide"] == 1

    relationship = result["graph_reasoning"]["direct_relationships"][0]
    assert relationship["source"] == "Product Revenue"
    assert relationship["target"] == "Regional Demand"
    assert relationship["metadata"]["slide"] == 1
    assert "source_text" not in relationship["metadata"]

    artifact_run = result["pipeline_artifacts"]["runs"][0]
    artifacts = {artifact["kind"]: artifact for artifact in artifact_run["artifacts"]}
    assert artifacts["source_document"]["endpoint"].endswith("/artifacts/source_document")
    assert artifact_run["source_artifacts"][0]["path"].endswith("/artifacts/source_document")
    assert "content_base64" not in artifacts["source_document"]
    assert len(artifact_run["evidence_cards"]) == 1


@pytest.mark.asyncio
async def test_unified_chat_context_builds_pinned_batch_graph_overview(monkeypatch):
    from modules.graphrag.config import reset_config
    from modules.graphrag.pipeline_runs import get_pipeline_run_store, reset_pipeline_run_store

    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("GRAPHRAG_SERVICE_URL", raising=False)
    reset_config()
    reset_pipeline_run_store()

    context = load_graphrag_route_module("context")
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_batch_strategy",
        "status": "preview_only",
        "updated_at": "2026-01-15T10:00:00Z",
        "document": {"filename": "strategy.pptx", "domain": "batch"},
        "local_graph": {
            "filename": "strategy.pptx",
            "domain": "batch",
            "node_count": 3,
            "edge_count": 2,
            "relationship_count": 2,
            "truncated": False,
            "nodes": [
                {"id": "deck", "name": "Strategy Deck", "type": "document"},
                {"id": "forecast", "name": "Revenue Forecast", "type": "metric"},
                {"id": "demand", "name": "Regional Demand", "type": "market_signal"},
            ],
            "edges": [
                {
                    "source": "Strategy Deck",
                    "target": "Revenue Forecast",
                    "label": "contains_chart",
                    "weight": 0.71,
                    "evidence": "Revenue forecast chart",
                    "metadata": {"slide": 1, "slide_title": "Revenue Architecture", "element_type": "chart_series"},
                },
                {
                    "source": "Revenue Forecast",
                    "target": "Regional Demand",
                    "label": "drives",
                    "weight": 0.88,
                    "evidence": "Forecast drives regional demand",
                    "metadata": {"slide": 1, "slide_title": "Revenue Architecture", "element_type": "shape_connector"},
                },
            ],
        },
        "local_context": {
            "filename": "strategy.pptx",
            "domain": "batch",
            "chunks": [
                {
                    "id": "strategy_chunk",
                    "chunk_index": 0,
                    "content": "strategy.pptx: Revenue Forecast drives Regional Demand on slide 1.",
                    "metadata": {"provenance": ["slide=1; slide_title=Revenue Architecture"]},
                },
            ],
        },
    })
    get_pipeline_run_store().upsert({
        "run_id": "graphrag_batch_scorecard",
        "status": "preview_only",
        "updated_at": "2026-01-15T10:01:00Z",
        "document": {"filename": "scorecard.xlsx", "domain": "batch"},
        "local_graph": {
            "filename": "scorecard.xlsx",
            "domain": "batch",
            "node_count": 3,
            "edge_count": 2,
            "relationship_count": 2,
            "truncated": False,
            "nodes": [
                {"id": "scorecard", "name": "Quality Scorecard", "type": "document"},
                {"id": "forecast", "name": "Revenue Forecast", "type": "metric"},
                {"id": "risk", "name": "Launch Risk", "type": "risk"},
            ],
            "edges": [
                {
                    "source": "Quality Scorecard",
                    "target": "Revenue Forecast",
                    "label": "tracks",
                    "weight": 0.83,
                    "evidence": "Scorecard tracks revenue forecast",
                    "metadata": {"sheet_name": "Q1 Scorecard", "table": 1, "element_type": "table_row"},
                },
                {
                    "source": "Revenue Forecast",
                    "target": "Launch Risk",
                    "label": "influences",
                    "weight": 0.79,
                    "evidence": "Revenue forecast influences launch risk",
                    "metadata": {"sheet_name": "Q1 Scorecard", "table": 1, "element_type": "table_row"},
                },
            ],
        },
        "local_context": {
            "filename": "scorecard.xlsx",
            "domain": "batch",
            "chunks": [
                {
                    "id": "scorecard_chunk",
                    "chunk_index": 0,
                    "content": "scorecard.xlsx: Revenue Forecast is tracked in the scorecard and influences Launch Risk.",
                    "metadata": {"provenance": ["sheet_name=Q1 Scorecard; table=1"]},
                },
            ],
        },
    })

    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "Compare these uploaded files. Find shared entities, cross-document relationships, graph paths, and provenance.",
        "rag_enabled": True,
        "domain": "batch",
        "pipeline_run_ids": ["graphrag_batch_strategy", "graphrag_batch_scorecard"],
    })
    mock_client = Mock()

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_retrieve_rag_chunks_for_query", AsyncMock(return_value=[])),
        patch.object(context, "_search_graphrag", AsyncMock(return_value=([], []))),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    graph_reasoning = result["graph_reasoning"]
    overview = graph_reasoning["graph_overview"]
    assert graph_reasoning["status"] == "complete"
    assert graph_reasoning["summary"]["mode"] == "graph_overview"
    assert graph_reasoning["graph_sources"]["remote_graph"] is False
    assert {source["run_id"] for source in graph_reasoning["graph_sources"]["local_pipeline_runs"]} == {
        "graphrag_batch_strategy",
        "graphrag_batch_scorecard",
    }
    assert overview["relationship_count"] == 4
    assert any(entity["name"] == "Revenue Forecast" for entity in overview["shared_entities"])
    assert any(relationship["source"] == "Revenue Forecast" for relationship in overview["top_relationships"])
    assert overview["bridge_entity_count"] >= 1
    assert any(bridge["name"] == "Revenue Forecast" for bridge in overview["bridge_entities"])
    assert overview["cross_document_path_count"] >= 1
    assert any(path["bridge"] == "Revenue Forecast" for path in overview["cross_document_paths"])
    assert "=== Local Pipeline Graph Overview ===" in result["context_text"]
    assert "Shared entities: Revenue Forecast" in result["context_text"]
    assert "Bridge entities: Revenue Forecast" in result["context_text"]
    assert "Cross-document paths:" in result["context_text"]
    assert "strategy.pptx" in result["context_text"]
    assert "scorecard.xlsx" in result["context_text"]
    assert "local pipeline strategy.pptx" not in result["context_text"]
    assert "local pipeline scorecard.xlsx" not in result["context_text"]
    assert {chunk["document_id"] for chunk in result["chunks"]} == {
        "graphrag_batch_strategy",
        "graphrag_batch_scorecard",
    }
    assert result["quality_signals"]["graph_overview_relationship_count"] == 4
    assert result["quality_signals"]["graph_overview_shared_entity_count"] >= 1
    assert result["quality_signals"]["graph_overview_bridge_entity_count"] >= 1
    assert result["quality_signals"]["graph_overview_cross_document_path_count"] >= 1
    assert result["quality_signals"]["local_pipeline_graph_run_count"] == 2
    assert result["quality_signals"]["local_pipeline_context_run_count"] == 2
    assert result["services_used"]["graph_reasoning"] is True


def test_graph_reasoning_prefers_attachment_paths_for_source_file_questions():
    context = load_graphrag_route_module("context")
    node_map = {
        "doc": {"name": "Annual Strategy", "id": "doc", "type": "document"},
        "attachment": {"name": "annual-strategy.pdf attachment source-data.csv", "id": "attachment", "type": "attachment"},
        "source": {"name": "source-data.csv", "id": "source", "type": "file"},
    }
    raw_edges = [
        {"source": "Annual Strategy", "target": "source-data.csv", "label": "related_to", "weight": 0.78},
        {"source": "Annual Strategy", "target": "annual-strategy.pdf attachment source-data.csv", "label": "has_attachment", "weight": 0.67},
        {"source": "annual-strategy.pdf attachment source-data.csv", "target": "source-data.csv", "label": "contains_embedded_object", "weight": 0.68},
    ]

    paths = context._find_graph_paths(
        "Annual Strategy",
        "source-data.csv",
        node_map,
        raw_edges,
        query="Which attachment has the source file?",
    )

    assert paths[0]["nodes"] == ["Annual Strategy", "annual-strategy.pdf attachment source-data.csv", "source-data.csv"]
    assert paths[0]["path_type"] == "layout"
    assert {"attachment", "source", "file"} & set(paths[0]["matched_query_terms"])
    assert paths[0]["score"] > paths[1]["score"]


def test_graph_reasoning_prioritizes_explicit_direct_relationship_over_paths():
    context = load_graphrag_route_module("context")
    result = context._finalize_graph_reasoning_result({
        "status": "complete",
        "reasoning_needed": True,
        "entity_pairs": [{
            "source": "Approval Dependency Cluster",
            "target": "Approval required before launch workflow.",
        }],
        "direct_relationships": [
            {
                "source": "Approval Dependency Cluster",
                "relation": "groups_visual",
                "target": "Launch workflow",
                "relationship_type": "semantic",
                "weight": 0.9,
                "score": 0.91,
                "evidence": "Approval Dependency Cluster groups Launch workflow.",
                "metadata": {
                    "local_graph_filename": "graphrag-grouped-visuals.pptx",
                    "slide": 1,
                    "slide_title": "Launch Approval Packet",
                    "element_type": "shape_group",
                    "location": "slide 1 group 1",
                },
            },
            {
                "source": "Approval Dependency Cluster",
                "relation": "groups_visual",
                "target": "Approval required before launch workflow.",
                "relationship_type": "semantic",
                "weight": 0.68,
                "score": 0.7,
                "evidence": "Approval Dependency Cluster groups Approval required before launch workflow., Launch workflow, gates.",
                "metadata": {
                    "local_graph_filename": "graphrag-grouped-visuals.pptx",
                    "slide": 1,
                    "slide_title": "Launch Approval Packet",
                    "element_type": "shape_group",
                    "location": "slide 1 group 1",
                },
            },
        ],
        "paths": [{
            "source": "Approval Dependency Cluster",
            "target": "Approval required before launch workflow.",
            "nodes": [
                "Approval Dependency Cluster",
                "Launch workflow",
                "Approval required before launch workflow.",
            ],
            "relationships": [],
            "hops": 2,
            "score": 0.95,
            "text": "Approval Dependency Cluster --[groups_visual]-- Launch workflow --[gates]-- Approval required before launch workflow.",
            "explanation": "Launch workflow connects Approval Dependency Cluster and Approval required before launch workflow.",
        }],
        "bridge_entities": [{"name": "Launch workflow", "path_count": 1}],
        "graph_overview": {},
        "interaction_handoffs": [],
        "pipeline_evidence": {},
    })

    first_relationship = result["direct_relationships"][0]
    assert first_relationship["source"] == "Approval Dependency Cluster"
    assert first_relationship["relation"] == "groups_visual"
    assert first_relationship["target"] == "Approval required before launch workflow."
    assert result["answer_plan"]["strategy"] == "cite_direct_relationships"
    assert result["answer_plan"]["supporting_relationships"][0]["target"] == "Approval required before launch workflow."
    assert result["summary"]["mode"] == "direct_relationship"
    assert result["summary"]["evidence"][0].startswith("Direct relationships:")
    assert "Approval Dependency Cluster --[groups_visual]--> Approval required before launch workflow." in result["summary"]["evidence"][0]
    assert "Related graph path:" in result["summary"]["evidence"][1]


def test_graph_reasoning_prefers_provenance_paths_for_nested_source_questions():
    context = load_graphrag_route_module("context")
    node_map = {
        "bundle": {"name": "Evidence Bundle", "id": "bundle", "type": "document"},
        "docx-entry": {"name": "office-evidence.zip archive docs/implementation.docx", "id": "docx-entry", "type": "archive_entry"},
        "checklist": {"name": "Launch Checklist", "id": "checklist", "type": "revision"},
    }
    raw_edges = [
        {"source": "Evidence Bundle", "target": "Launch Checklist", "label": "related_to", "weight": 0.82},
        {
            "source": "Evidence Bundle",
            "target": "office-evidence.zip archive docs/implementation.docx",
            "label": "has_archive_entry",
            "weight": 0.68,
            "properties": {
                "archive_entry_path": "docs/implementation.docx",
                "archive_entry_extraction_method": "docx_xml",
                "element_type": "archive_entry",
            },
        },
        {
            "source": "office-evidence.zip archive docs/implementation.docx",
            "target": "Launch Checklist",
            "label": "contains_revision",
            "weight": 0.68,
            "properties": {
                "archive_entry_path": "docs/implementation.docx",
                "archive_entry_extraction_method": "docx_xml",
                "element_type": "revision",
                "location": "archive entry 1 revision 1",
            },
        },
    ]

    paths = context._find_graph_paths(
        "Evidence Bundle",
        "Launch Checklist",
        node_map,
        raw_edges,
        query="Which DOCX source file contains the launch checklist?",
    )
    graph_reasoning = {"paths": paths, "direct_relationships": []}
    terms = context._graph_augmented_retrieval_terms(graph_reasoning, limit=12)

    assert paths[0]["nodes"] == [
        "Evidence Bundle",
        "office-evidence.zip archive docs/implementation.docx",
        "Launch Checklist",
    ]
    assert paths[0]["path_type"] == "layout"
    assert {"docx", "source", "file", "contains"} & set(paths[0]["matched_query_terms"])
    assert paths[0]["score"] > paths[1]["score"]
    assert "docs/implementation.docx" in terms
    assert "docx xml" in terms


@pytest.mark.asyncio
async def test_unified_chat_context_uses_structural_relationships_for_layout_question():
    context = load_graphrag_route_module("context")
    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "Which slide contains the Risk Map?",
        "rag_enabled": False,
        "domain": "slides",
    })
    mock_client = Mock()
    mock_client.post = AsyncMock(return_value=_mock_response({
        "primary_reasoning": "entity_location",
        "requires_multi_hop": False,
    }))

    node_map = {
        "strategy": {"name": "Strategy Deck", "id": "strategy", "type": "document"},
        "slide": {"name": "Slide 1: Revenue Architecture", "id": "slide", "type": "slide"},
        "risk": {"name": "Risk Map", "id": "risk", "type": "image"},
    }
    raw_edges = [
        {"source": "Strategy Deck", "target": "Slide 1: Revenue Architecture", "label": "has_slide", "weight": 0.72},
        {"source": "Slide 1: Revenue Architecture", "target": "Risk Map", "label": "contains_image", "weight": 0.71},
        {"source": "Risk Map", "target": "Regional Risk", "label": "related_to", "weight": 0.95},
    ]

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[
            {"name": "Risk Map", "type": "IMAGE", "score": 0.97},
        ])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_search_graphrag", AsyncMock(return_value=([], []))),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
        patch.object(context, "_fetch_graph_data", AsyncMock(return_value=(node_map, raw_edges))),
    ):
        result = await context.graphrag_chat_context_v2(request)

    graph_reasoning = result["graph_reasoning"]
    assert graph_reasoning["status"] == "complete"
    assert graph_reasoning["reasoning_needed"] is True
    assert graph_reasoning["direct_relationships"][0]["source"] == "Slide 1: Revenue Architecture"
    assert graph_reasoning["direct_relationships"][0]["relation"] == "contains_image"
    assert graph_reasoning["direct_relationships"][0]["relationship_type"] == "layout"
    assert "Slide 1: Revenue Architecture --[contains_image]--> Risk Map" in result["context_text"]
    assert result["quality_signals"]["direct_graph_relationship_count"] >= 1
    assert {"slide", "contains"} & set(result["quality_signals"]["graph_matched_query_terms"])


def test_graph_reasoning_handles_export_aliases_and_nested_edge_endpoints():
    context = load_graphrag_route_module("context")
    node_map = {
        "node-1": {
            "id": "node-1",
            "label": "Ada Lovelace",
            "properties": {"name": "Ada Lovelace", "aliases": ["Countess Lovelace"]},
        },
        "node-2": {
            "id": "node-2",
            "label": "Charles Babbage",
            "properties": {"name": "Charles Babbage"},
        },
        "node-3": {
            "id": "node-3",
            "label": "Analytical Engine",
            "properties": {"name": "Analytical Engine"},
        },
    }
    raw_edges = [
        {
            "startNode": {"id": "node-1"},
            "endNode": {"id": "node-2"},
            "properties": {"relationship_type": "collaborated_with", "confidence": 0.91},
        },
        {
            "source_node": "node-2",
            "targetNode": {"properties": {"name": "Analytical Engine"}},
            "properties": {"type": "designed", "score": 0.87},
        },
    ]

    paths = context._find_graph_paths(
        "Countess Lovelace",
        "Analytical Engine",
        node_map,
        raw_edges,
        query="How is Countess Lovelace connected to the Analytical Engine?",
    )
    relationships = context._mentioned_graph_relationships(["Ada Lovelace"], node_map, raw_edges)

    assert paths[0]["nodes"] == ["Ada Lovelace", "Charles Babbage", "Analytical Engine"]
    assert paths[0]["relationships"][0]["relation"] == "collaborated_with"
    assert paths[0]["relationships"][0]["weight"] == 0.91
    assert paths[0]["relationships"][1]["relation"] == "designed"
    assert paths[0]["relationships"][1]["weight"] == 0.87
    assert "Ada Lovelace --[collaborated_with]-- Charles Babbage" in paths[0]["text"]
    assert relationships[0]["source"] == "Ada Lovelace"
    assert relationships[0]["target"] == "Charles Babbage"


def test_graph_relationship_provenance_formats_document_metadata():
    context = load_graphrag_route_module("context")
    node_map = {
        "page-1": {"id": "page-1", "label": "scanned-invoice.pdf page 1"},
        "ocr-text": {"id": "ocr-text", "label": "ACME Corp Total Due 42"},
    }
    edge = {
        "source": "page-1",
        "target": "ocr-text",
        "label": "contains_text",
        "properties": {
            "page": 1,
            "ocr_source": "pdf_page_render",
            "ocr_engine": "pytesseract",
            "element_type": "page_ocr_text",
            "location": "page 1 OCR",
        },
    }

    normalized = context._normalize_graph_edge(edge, node_map)
    formatted = context._format_graph_relationship(normalized)

    assert normalized["provenance"]
    assert "page 1" in normalized["provenance"]
    assert "OCR pdf_page_render via pytesseract" in normalized["provenance"]
    assert "element page_ocr_text" in normalized["provenance"]
    assert "scanned-invoice.pdf page 1 --[contains_text]--> ACME Corp Total Due 42" in formatted
    assert "(page 1;" in formatted

    attachment_edge = {
        "source": "doc-1",
        "target": "attachment-1",
        "label": "contains_embedded_object",
        "properties": {
            "embedded_file_filename": "source-data.csv",
            "embedded_file_mime": "text/csv",
            "embedded_file_size_bytes": 32,
            "element_type": "embedded_object",
            "location": "embedded file 1",
        },
    }
    attachment_nodes = {
        "doc-1": {"id": "doc-1", "label": "annual-strategy.pdf"},
        "attachment-1": {"id": "attachment-1", "label": "source-data.csv"},
    }
    attachment = context._normalize_graph_edge(attachment_edge, attachment_nodes)
    attachment_text = context._format_graph_relationship(attachment)

    assert "attachment source-data.csv text/csv 32 bytes" in attachment["provenance"]
    assert "element embedded_object" in attachment["provenance"]
    assert "annual-strategy.pdf --[contains_embedded_object]--> source-data.csv" in attachment_text

    archive_edge = {
        "source": "bundle",
        "target": "owners",
        "label": "contains_archive_entry",
        "properties": {
            "archive_entry_path": "data/owners.csv",
            "archive_entry_mime": "text/csv",
            "archive_entry_size_bytes": 43,
            "element_type": "archive_entry",
            "location": "archive entry 2",
        },
    }
    archive_nodes = {
        "bundle": {"id": "bundle", "label": "evidence-bundle.zip"},
        "owners": {"id": "owners", "label": "data/owners.csv"},
    }
    archive = context._normalize_graph_edge(archive_edge, archive_nodes)
    archive_text = context._format_graph_relationship(archive)

    assert "archive entry data/owners.csv text/csv 43 bytes" in archive["provenance"]
    assert "element archive_entry" in archive["provenance"]
    assert "evidence-bundle.zip --[contains_archive_entry]--> data/owners.csv" in archive_text


def test_graph_reasoning_boosts_chunks_that_match_path_evidence():
    context = load_graphrag_route_module("context")
    chunks = [
        {
            "id": "entity-only",
            "content": "Ada Lovelace wrote notes about early computing.",
            "score": 0.32,
        },
        {
            "id": "path-evidence",
            "content": "Ada Lovelace collaborated with Charles Babbage, who designed the Analytical Engine.",
            "score": 0.25,
        },
    ]
    graph_reasoning = {
        "entity_names": ["Ada Lovelace", "Analytical Engine"],
        "entity_pairs": [{"source": "Ada Lovelace", "target": "Analytical Engine"}],
        "paths": [
            {
                "nodes": ["Ada Lovelace", "Charles Babbage", "Analytical Engine"],
                "relationships": [
                    {"source": "Ada Lovelace", "relation": "collaborated_with", "target": "Charles Babbage"},
                    {"source": "Charles Babbage", "relation": "designed", "target": "Analytical Engine"},
                ],
            }
        ],
        "direct_relationships": [],
    }

    cross_references = context._score_chunks_with_graph_context(
        chunks,
        [{"name": "Ada Lovelace"}],
        graph_reasoning,
    )

    assert chunks[0]["id"] == "path-evidence"
    assert chunks[0]["graph_reasoning_boost"] > 0
    assert chunks[0]["entity_boost"] > 0
    assert "Charles Babbage" in chunks[0]["matched_graph_path_terms"]
    assert "designed" in chunks[0]["matched_graph_path_terms"]
    assert cross_references[0]["chunk_id"] == "entity-only"
    assert any(ref["chunk_id"] == "path-evidence" and ref["graph_path_overlap_score"] > 0 for ref in cross_references)


def test_graph_augmented_retrieval_query_uses_bridge_terms():
    context = load_graphrag_route_module("context")
    graph_reasoning = {
        "bridge_entities": [
            {
                "name": "Charles Babbage",
                "text": "Charles Babbage bridges Ada Lovelace to Analytical Engine",
                "relationships": [
                    "Ada Lovelace --[collaborated_with]-- Charles Babbage",
                    "Charles Babbage --[designed]-- Analytical Engine",
                ],
            }
        ],
        "paths": [
            {
                "nodes": ["Ada Lovelace", "Charles Babbage", "Analytical Engine"],
                "relationships": [
                    {"source": "Ada Lovelace", "relation": "collaborated_with", "target": "Charles Babbage"},
                    {"source": "Charles Babbage", "relation": "designed", "target": "Analytical Engine"},
                ],
                "score": 0.91,
            }
        ],
        "direct_relationships": [],
    }

    terms = context._graph_augmented_retrieval_terms(graph_reasoning, limit=6)
    query = context._graph_augmented_retrieval_query(
        "How is Ada Lovelace connected to the Analytical Engine?",
        graph_reasoning,
    )

    assert "Charles Babbage" in terms
    assert "collaborated with" in terms
    assert "designed" in terms
    assert query.startswith("How is Ada Lovelace connected")
    assert "Graph path evidence terms:" in query
    assert "Charles Babbage" in query


def test_bridge_entities_summarize_intermediate_nodes_across_paths():
    context = load_graphrag_route_module("context")
    bridges = context._bridge_entities_from_paths([
        {
            "source": "Ada Lovelace",
            "target": "Analytical Engine",
            "nodes": ["Ada Lovelace", "Charles Babbage", "Analytical Engine"],
            "relationships": [
                {"source": "Ada Lovelace", "relation": "collaborated_with", "target": "Charles Babbage"},
                {"source": "Charles Babbage", "relation": "designed", "target": "Analytical Engine"},
            ],
            "score": 0.91,
            "path_type": "semantic",
            "matched_query_terms": ["designed"],
        },
        {
            "source": "Ada Lovelace",
            "target": "Difference Engine",
            "nodes": ["Ada Lovelace", "Charles Babbage", "Difference Engine"],
            "relationships": [
                {"source": "Ada Lovelace", "relation": "collaborated_with", "target": "Charles Babbage"},
                {"source": "Charles Babbage", "relation": "designed", "target": "Difference Engine"},
            ],
            "score": 0.84,
            "path_type": "semantic",
            "matched_query_terms": ["designed"],
        },
    ])

    assert bridges[0]["name"] == "Charles Babbage"
    assert bridges[0]["path_count"] == 2
    assert bridges[0]["score"] > 0.91
    assert "Ada Lovelace --[collaborated_with]-- Charles Babbage" in bridges[0]["relationships"]
    assert bridges[0]["connects"] == [
        {"source": "Ada Lovelace", "target": "Analytical Engine"},
        {"source": "Ada Lovelace", "target": "Difference Engine"},
    ]


def test_merge_rag_chunks_deduplicates_and_merges_retrieval_provenance():
    context = load_graphrag_route_module("context")

    merged = context._merge_rag_chunks(
        [
            {
                "id": "bridge",
                "content": "Ada Lovelace collaborated with Charles Babbage.",
                "score": 0.3,
                "retrieval_reasons": ["query"],
                "retrieval_queries": ["Ada Lovelace Analytical Engine"],
            }
        ],
        [
            {
                "id": "bridge",
                "content": "Ada Lovelace collaborated with Charles Babbage.",
                "score": 0.42,
                "retrieval_reasons": ["graph_augmented"],
                "retrieval_queries": ["Ada Lovelace Analytical Engine Charles Babbage designed"],
            },
            {
                "id": "engine",
                "content": "Charles Babbage designed the Analytical Engine.",
                "score": 0.21,
                "retrieval_reasons": ["graph_augmented"],
                "retrieval_queries": ["Ada Lovelace Analytical Engine Charles Babbage designed"],
            },
        ],
    )

    assert len(merged) == 2
    assert merged[0]["score"] == 0.42
    assert merged[0]["retrieval_reasons"] == ["query", "graph_augmented"]
    assert len(merged[0]["retrieval_queries"]) == 2
    assert merged[1]["id"] == "engine"


@pytest.mark.asyncio
async def test_unified_chat_context_runs_graph_augmented_retrieval_after_reasoning():
    context = load_graphrag_route_module("context")
    request = Mock()
    request.json = AsyncMock(return_value={
        "query": "How is Ada Lovelace connected to the Analytical Engine?",
        "rag_enabled": True,
        "domain": "history",
        "source": "Ada Lovelace",
        "target": "Analytical Engine",
        "top_k": 3,
    })
    mock_client = Mock()
    mock_client.post = AsyncMock(return_value=_mock_response({
        "primary_reasoning": "multi_hop",
        "requires_multi_hop": True,
    }))
    retrieval_calls = []

    async def fake_retrieve(
        request_arg,
        retrieval_query,
        retrieval_top_k,
        rag_domains,
        rag_search_mode,
        errors,
        *,
        retrieval_reason="query",
    ):
        retrieval_calls.append({
            "query": retrieval_query,
            "top_k": retrieval_top_k,
            "reason": retrieval_reason,
        })
        if retrieval_reason == "graph_augmented":
            return [
                {
                    "id": "bridge",
                    "content": "Charles Babbage designed the Analytical Engine after collaborating with Ada Lovelace.",
                    "score": 0.11,
                    "document_title": "Engine notes",
                    "retrieval_reasons": ["graph_augmented"],
                    "retrieval_queries": [retrieval_query],
                }
            ]
        return [
            {
                "id": "seed",
                "content": "Ada Lovelace wrote notes about early computing.",
                "score": 0.12,
                "document_title": "Ada notes",
                "retrieval_reasons": ["query"],
                "retrieval_queries": [retrieval_query],
            }
        ]

    node_map = {
        "ada lovelace": {"name": "Ada Lovelace", "id": "n1", "type": "person"},
        "charles babbage": {"name": "Charles Babbage", "id": "n2", "type": "person"},
        "analytical engine": {"name": "Analytical Engine", "id": "n3", "type": "technology"},
    }
    raw_edges = [
        {"source": "Ada Lovelace", "target": "Charles Babbage", "label": "collaborated_with", "weight": 0.9},
        {"source": "Charles Babbage", "target": "Analytical Engine", "label": "designed", "weight": 0.8},
    ]

    with (
        patch.object(context, "get_http_client", return_value=mock_client),
        patch.object(context, "_extract_entities_ner", AsyncMock(return_value=[
            {"name": "Ada Lovelace", "type": "PERSON", "score": 0.98},
            {"name": "Analytical Engine", "type": "TECHNOLOGY", "score": 0.95},
        ])),
        patch.object(context, "_extract_relationships_rel", AsyncMock(return_value=[])),
        patch.object(context, "_search_graphrag", AsyncMock(return_value=([], []))),
        patch.object(context, "_expand_graph_neighborhood", AsyncMock(return_value=([], []))),
        patch.object(context, "_fetch_graph_data", AsyncMock(return_value=(node_map, raw_edges))),
        patch.object(context, "_retrieve_rag_chunks_for_query", AsyncMock(side_effect=fake_retrieve)),
    ):
        result = await context.graphrag_chat_context_v2(request)

    bridge_chunk = next(chunk for chunk in result["chunks"] if chunk["id"] == "bridge")
    quality = result["quality_signals"]

    assert [call["reason"] for call in retrieval_calls] == ["query", "graph_augmented"]
    assert "Charles Babbage" in retrieval_calls[1]["query"]
    assert "designed" in retrieval_calls[1]["query"]
    assert bridge_chunk["graph_reasoning_boost"] > 0
    assert "graph_augmented" in bridge_chunk["retrieval_reasons"]
    assert quality["graph_augmented_query_used"] is True
    assert quality["supplemental_chunk_count"] == 1
    assert quality["graph_augmented_chunk_count"] == 1
    assert "Charles Babbage" in quality["graph_augmented_retrieval_terms"]
    assert "(retrieval: graph-expanded)" in result["context_text"]
