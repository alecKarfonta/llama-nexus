from modules.workflow.executors import NODE_EXECUTORS
from modules.workflow.templates import WORKFLOW_TEMPLATES, get_workflow_template


def test_all_workflow_templates_use_registered_node_types():
    invalid_nodes = []

    for template in WORKFLOW_TEMPLATES:
        for node in template["nodes"]:
            if node["type"] not in NODE_EXECUTORS:
                invalid_nodes.append((template["id"], node["id"], node["type"]))
            if node["data"].get("nodeType") != node["type"]:
                invalid_nodes.append((template["id"], node["id"], "nodeType mismatch"))

    assert invalid_nodes == []


def test_all_workflow_template_connections_reference_declared_ports():
    invalid_connections = []

    for template in WORKFLOW_TEMPLATES:
        nodes_by_id = {node["id"]: node for node in template["nodes"]}

        for connection in template.get("connections", []):
            source = nodes_by_id.get(connection["source"])
            target = nodes_by_id.get(connection["target"])
            if not source or not target:
                invalid_connections.append((template["id"], connection["id"], "missing node"))
                continue

            source_outputs = {
                port["id"]
                for port in source["data"].get("outputs", [])
            }
            target_inputs = {
                port["id"]
                for port in target["data"].get("inputs", [])
            }

            if connection["sourceHandle"] not in source_outputs:
                invalid_connections.append((
                    template["id"],
                    connection["id"],
                    f"missing source output {connection['sourceHandle']}",
                ))
            if connection["targetHandle"] not in target_inputs:
                invalid_connections.append((
                    template["id"],
                    connection["id"],
                    f"missing target input {connection['targetHandle']}",
                ))

    assert invalid_connections == []


def test_graphrag_templates_cover_quality_and_diagnostics():
    template_ids = {template["id"] for template in WORKFLOW_TEMPLATES}

    assert "graphrag-batch-processing" not in template_ids
    assert "graphrag-observability-quality" in template_ids
    assert "graphrag-extraction-diagnostics" in template_ids
    assert "graphrag-social-memory-context" in template_ids
    assert "graphrag-knowledge-graph-explorer" in template_ids
    assert "graphrag-code-rag-indexing" in template_ids
    assert "graphrag-reddit-research-ingestion" in template_ids
    assert "graphrag-corpus-maintenance" in template_ids

    capability_template = get_workflow_template("graphrag-capability-aware-query")
    quality_template = get_workflow_template("graphrag-observability-quality")
    diagnostic_template = get_workflow_template("graphrag-extraction-diagnostics")
    social_memory_template = get_workflow_template("graphrag-social-memory-context")
    graph_template = get_workflow_template("graphrag-knowledge-graph-explorer")
    code_rag_template = get_workflow_template("graphrag-code-rag-indexing")
    reddit_template = get_workflow_template("graphrag-reddit-research-ingestion")
    corpus_template = get_workflow_template("graphrag-corpus-maintenance")

    capability_node_types = {node["type"] for node in capability_template["nodes"]}
    quality_node_types = {node["type"] for node in quality_template["nodes"]}
    diagnostic_node_types = {node["type"] for node in diagnostic_template["nodes"]}
    social_memory_node_types = {node["type"] for node in social_memory_template["nodes"]}
    graph_node_types = {node["type"] for node in graph_template["nodes"]}
    code_rag_node_types = {node["type"] for node in code_rag_template["nodes"]}
    reddit_node_types = {node["type"] for node in reddit_template["nodes"]}
    corpus_node_types = {node["type"] for node in corpus_template["nodes"]}

    assert {
        "graphrag_capabilities",
        "graphrag_query_complexity",
        "graphrag_chat_context",
    }.issubset(capability_node_types)

    assert {
        "graphrag_capabilities",
        "graphrag_domain_stats",
        "graphrag_instrumentation",
        "graphrag_evaluation_status",
        "graphrag_test_suite",
        "graphrag_feedback_analytics",
    }.issubset(quality_node_types)

    assert {
        "graphrag_ner_extraction",
        "graphrag_relation_extraction",
        "enhanced_entity_extraction",
        "graphrag_enhanced_extraction_test",
    }.issubset(diagnostic_node_types)

    assert {
        "graphrag_social_memory_ingest",
        "graphrag_social_memory_context",
        "graphrag_social_memory_facts",
    }.issubset(social_memory_node_types)

    assert "graphrag_graph_explorer" in graph_node_types
    graph_explorer_modes = {
        node["data"].get("config", {}).get("mode")
        for node in graph_template["nodes"]
        if node["type"] == "graphrag_graph_explorer"
    }
    assert {
        "filtered_graph",
        "top_entities",
        "top_relationships",
        "entity_neighborhood",
        "communities",
    }.issubset(graph_explorer_modes)

    assert {
        "graphrag_code_ingest",
        "graphrag_code_ingestion_status",
        "graphrag_code_analysis",
        "graphrag_code_stats",
    }.issubset(code_rag_node_types)

    assert {
        "graphrag_reddit_crawl",
        "graphrag_reddit_crawl_status",
        "graphrag_reddit_crawl_results",
    }.issubset(reddit_node_types)

    assert {
        "graphrag_document_list",
        "graphrag_kg_rebuild",
        "graphrag_domain_stats",
    }.issubset(corpus_node_types)
