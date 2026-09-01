"""
GraphRAG Context Routes

Chat context generation with NER, REL, vector search,
and graph neighborhood expansion.

All endpoints use the shared connection-pooled HTTP client
and the unified GraphRAGConfig singleton.
"""

from fastapi import APIRouter, HTTPException, Request
import asyncio
from collections import Counter, defaultdict, deque
from itertools import combinations
import logging
import re

from .helpers import (
    check_graphrag_enabled, get_http_client,
    _fetch_graph_data, _find_entity_neighbors,
    _extract_entities_ner, _extract_relationships_rel,
    _expand_graph_neighborhood,
    _deduplicate_entities, _deduplicate_relationships,
    optional_chat_context_timeout,
)
from modules.graphrag.config import get_graphrag_config
from modules.graphrag.pipeline_runs import get_pipeline_run_store

logger = logging.getLogger(__name__)

router = APIRouter()


EMAIL_GRAPH_METADATA_KEYS = (
    "email_header",
    "email_message_id",
    "email_subject",
    "email_from",
    "email_sender",
    "email_to",
    "email_cc",
    "email_bcc",
    "email_reply_to",
    "email_date",
    "email_date_iso",
    "email_in_reply_to",
    "email_references",
    "email_part_index",
    "email_part_kind",
    "email_content_type",
    "email_disposition",
    "email_attachment_index",
    "email_attachment_filename",
    "email_attachment_mime",
)

EMAIL_GRAPH_RELATIONS = {
    "bcc_to",
    "cc_to",
    "contains_email_body",
    "has_message_id",
    "has_reply_to",
    "has_subject",
    "in_reply_to",
    "references_message",
    "sent_at",
    "sent_by",
    "sent_message",
    "sent_to",
}


# ── Shared: GraphRAG vector search ─────────────────────────────────────

async def _search_graphrag(
    client, query: str, top_k: int, domain: str, errors: list
) -> tuple[list[dict], list[dict]]:
    """Search GraphRAG for relevant passages and extract entities from results.

    Returns (search_results, additional_entities).
    """
    cfg = get_graphrag_config()
    search_results = []
    extra_entities = []

    if not cfg.graphrag_url:
        return search_results, extra_entities

    try:
        search_response = await client.post(
            f"{cfg.graphrag_url}/search-advanced",
            json={
                "query": query,
                "top_k": top_k,
                "domain": domain,
                "search_type": "hybrid",
            },
            timeout=optional_chat_context_timeout(),
        )
        if search_response.status_code == 200:
            search_data = search_response.json()
            for result in search_data.get("results", [])[:top_k]:
                search_results.append({
                    "content": result.get("content", "")[:500],
                    "score": round(result.get("score", result.get("similarity", 0)), 3),
                    "source": result.get("source_file", result.get("metadata", {}).get("source", "")),
                })
            # Grab entities from search response
            for ent in search_data.get("entities", []):
                if isinstance(ent, dict):
                    extra_entities.append({
                        "name": ent.get("name", ent.get("text", "")),
                        "type": ent.get("type", ent.get("label", "ENTITY")),
                        "score": round(ent.get("score", ent.get("confidence", 0.8)), 3),
                    })
                elif isinstance(ent, str):
                    extra_entities.append({"name": ent, "type": "ENTITY", "score": 0.8})
    except Exception as e:
        logger.warning(f"GraphRAG search failed: {e}")
        errors.append(f"Search: {str(e)}")

    return search_results, extra_entities


# ── v1: /chat-context ──────────────────────────────────────────────────

@router.post("/chat-context")
async def graphrag_chat_context(request: Request):
    """
    Get graph-enhanced context for chat queries.

    Queries NER/REL services for entity & relationship extraction,
    and optionally GraphRAG vector search for relevant passages.
    Returns structured context text for LLM injection.
    """
    check_graphrag_enabled()
    cfg = get_graphrag_config()

    data = await request.json()
    query = data.get("query", "")
    top_k = data.get("top_k", 5)
    domain = data.get("domain", "general")
    pipeline_run_ids = _string_list(data.get("pipeline_run_ids") or data.get("pipeline_run_id"))
    pipeline_evidence = _pipeline_evidence_hints_from_request(data if isinstance(data, dict) else {})

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    errors = []

    client = get_http_client()

    # 1. NER & REL extraction (parallel — independent services)
    entities, relationships = await asyncio.gather(
        _extract_entities_ner(client, query, errors),
        _extract_relationships_rel(client, query, errors),
    )

    # 2. GraphRAG vector search
    search_results, extra_entities = await _search_graphrag(
        client, query, top_k, domain, errors
    )
    entities.extend(extra_entities)

    # 3. Graph neighborhood expansion (shared helper)
    expanded_entities, graph_connections = await _expand_graph_neighborhood(
        client,
        entities,
        domain,
        errors,
        timeout=optional_chat_context_timeout(),
    )

    # Deduplicate (shared helpers)
    entities = _deduplicate_entities(entities, expanded_entities)
    relationships = _deduplicate_relationships(relationships)

    graph_reasoning = await _build_graph_reasoning_context(
        client,
        query,
        entities,
        domain,
        errors,
        explicit_source=data.get("source"),
        explicit_target=data.get("target"),
        explicit_bridge=data.get("bridge"),
        explicit_relation=data.get("relation"),
        pipeline_run_ids=pipeline_run_ids,
        pipeline_evidence=pipeline_evidence,
    )
    pipeline_artifacts = _local_pipeline_artifact_context(
        query,
        domain,
        pipeline_run_ids=pipeline_run_ids,
    )

    # Build structured context text for LLM injection
    context_parts = []

    if entities:
        extracted = [e for e in entities if e.get("source") != "graph_expansion"]
        discovered = [e for e in entities if e.get("source") == "graph_expansion"]

        entity_strs = [f"{e['name']} ({e['type']})" for e in extracted[:15]]
        context_parts.append(f"Entities: {', '.join(entity_strs)}")

        if discovered:
            disc_strs = [f"{e['name']} ({e['type']})" for e in discovered[:10]]
            context_parts.append(f"Related entities (from knowledge graph): {', '.join(disc_strs)}")

    if graph_connections:
        context_parts.append(
            "Knowledge graph connections:\n"
            + "\n".join(f"  • {c}" for c in graph_connections[:15])
        )

    answer_plan = graph_reasoning.get("answer_plan") if isinstance(graph_reasoning.get("answer_plan"), dict) else {}
    primary_answer_path = answer_plan.get("primary_path") if isinstance(answer_plan.get("primary_path"), dict) else {}
    path_answer_active = answer_plan.get("strategy") == "walk_strongest_path" and bool(primary_answer_path)
    graph_answer_plan = _graph_answer_plan_context(answer_plan)
    if graph_answer_plan:
        context_parts.append(graph_answer_plan)

    graph_handoff_context = _graph_interaction_handoff_context(
        graph_reasoning.get("interaction_handoffs", [])
    )
    if graph_handoff_context:
        context_parts.append(graph_handoff_context)

    pipeline_artifact_context = _local_pipeline_artifact_context_text(pipeline_artifacts)
    if pipeline_artifact_context:
        context_parts.append(pipeline_artifact_context)

    graph_summary = graph_reasoning.get("summary", {}) if isinstance(graph_reasoning.get("summary"), dict) else {}
    if graph_summary.get("evidence"):
        context_parts.append(
            "Graph reasoning summary:\n"
            + "\n".join(f"  • {item}" for item in graph_summary.get("evidence", [])[:4])
            + f"\nGuidance: {graph_summary.get('answer_guidance', '')}"
        )

    if graph_reasoning.get("paths"):
        path_strs = [
            f"{_format_graph_path_for_answer(path) or path['text']} (score: {path.get('score', 0)})"
            for path in graph_reasoning["paths"][:5]
        ]
        context_parts.append("Knowledge graph paths:\n" + "\n".join(f"  • {path}" for path in path_strs))

    if graph_reasoning.get("bridge_entities"):
        bridge_strs = [
            f"{_format_bridge_entity_for_answer(bridge) or bridge['text']} (score: {bridge.get('score', 0)}; paths: {bridge.get('path_count', 0)})"
            for bridge in graph_reasoning["bridge_entities"][:5]
        ]
        context_parts.append("Bridge entities:\n" + "\n".join(f"  • {bridge}" for bridge in bridge_strs))

    direct_relationships_for_context = [
        relationship
        for relationship in graph_reasoning.get("direct_relationships", []) or []
        if isinstance(relationship, dict)
    ]
    if path_answer_active:
        path_relationships_for_context = _path_supporting_relationships(
            direct_relationships_for_context,
            primary_answer_path,
            limit=8,
        )
        direct_relationships_for_context = _merge_graph_relationships(
            path_relationships_for_context,
            _query_matched_relationships_for_context(direct_relationships_for_context, limit=8),
            limit=8,
        )
    if direct_relationships_for_context:
        direct_rels = [
            _format_graph_relationship_for_answer(rel, include_provenance=False)
            for rel in direct_relationships_for_context[:8]
        ]
        context_parts.append("Relevant graph relationships:\n" + "\n".join(f"  • {rel}" for rel in direct_rels))

    if relationships:
        rel_strs = [f"{r['source']} --[{r['relation']}]--> {r['target']}" for r in relationships[:10]]
        context_parts.append("Extracted relationships:\n" + "\n".join(f"  • {r}" for r in rel_strs))

    if search_results:
        passage_strs = [f"[{i+1}] {r['content']}" for i, r in enumerate(search_results[:top_k])]
        context_parts.append("Relevant passages:\n" + "\n\n".join(passage_strs))

    context_text = ""
    if context_parts:
        context_text = _sanitize_llm_context_text(
            "\n\n--- Knowledge Graph Context ---\n"
            + "\n\n".join(context_parts)
            + "\n--- End Knowledge Graph Context ---\n\n"
            "Use the above knowledge graph context to enrich your answer. "
            "Reference specific entities, relationships, and graph connections when relevant."
        )

    return {
        "entities": entities,
        "relationships": relationships,
        "search_results": search_results,
        "context_text": context_text,
        "graph_connections": graph_connections,
        "graph_reasoning": graph_reasoning,
        "pipeline_artifacts": pipeline_artifacts,
        "pipeline_evidence": graph_reasoning.get("pipeline_evidence") if isinstance(graph_reasoning.get("pipeline_evidence"), dict) else {},
        "expanded_entities": expanded_entities,
        "source": "graphrag",
        "services_used": {
            "ner": cfg.ner_enabled,
            "rel": cfg.rel_enabled,
            "vector_search": cfg.has_graph_service,
            "graph_expansion": cfg.has_graph_service and len(expanded_entities) > 0,
            "graph_reasoning": graph_reasoning.get("status") == "complete",
            "local_pipeline_graph": bool((graph_reasoning.get("graph_sources") or {}).get("local_pipeline_runs")),
            "local_pipeline_interaction_handoff": bool(graph_reasoning.get("interaction_handoffs")),
            "local_pipeline_artifacts": bool(pipeline_artifacts.get("run_count")),
            "pipeline_evidence_handoff": bool(graph_reasoning.get("pipeline_evidence")),
        },
        "errors": errors if errors else None,
    }


# ── Unified Context Fusion ─────────────────────────────────────────────

def _entity_overlap_score(text: str, entity_names: list[str]) -> tuple[float, list[str]]:
    """Score how many entity names appear in a text chunk.

    Returns (overlap_fraction, matched_entity_names).
    """
    if not entity_names or not text:
        return 0.0, []

    text_lower = text.lower()
    matched = [name for name in entity_names if name.lower() in text_lower]
    return len(matched) / len(entity_names), matched


def _unique_reasoning_terms(values: list[str], *, min_length: int = 2) -> list[str]:
    terms = []
    seen = set()
    for value in values:
        term = re.sub(r"\s+", " ", str(value or "").replace("_", " ")).strip()
        key = term.lower()
        if len(term) >= min_length and key not in seen:
            seen.add(key)
            terms.append(term)
    return terms


def _graph_reasoning_node_terms(graph_reasoning: dict, *, include_seed_entities: bool = True) -> list[str]:
    values: list[str] = []
    if include_seed_entities:
        values.extend(graph_reasoning.get("entity_names", []) or [])

        for pair in graph_reasoning.get("entity_pairs", []) or []:
            if isinstance(pair, dict):
                values.extend([pair.get("source", ""), pair.get("target", "")])

    for path in graph_reasoning.get("paths", []) or []:
        if isinstance(path, dict):
            values.extend(path.get("nodes", []) or [])
            for relationship in path.get("relationships", []) or []:
                if isinstance(relationship, dict):
                    values.extend([relationship.get("source", ""), relationship.get("target", "")])

    for relationship in graph_reasoning.get("direct_relationships", []) or []:
        if isinstance(relationship, dict):
            values.extend([relationship.get("source", ""), relationship.get("target", "")])

    graph_overview = graph_reasoning.get("graph_overview") if isinstance(graph_reasoning.get("graph_overview"), dict) else {}
    for relationship in graph_overview.get("top_relationships", []) or []:
        if isinstance(relationship, dict):
            values.extend([relationship.get("source", ""), relationship.get("target", "")])
    for entity in graph_overview.get("shared_entities", []) or []:
        if isinstance(entity, dict):
            values.append(entity.get("name", ""))
    for bridge in graph_overview.get("bridge_entities", []) or []:
        if isinstance(bridge, dict):
            values.append(bridge.get("name", ""))
    for path in graph_overview.get("cross_document_paths", []) or []:
        if isinstance(path, dict):
            values.extend([
                path.get("source", ""),
                path.get("bridge", ""),
                path.get("target", ""),
                *(path.get("nodes", []) or []),
            ])

    return _unique_reasoning_terms(values)


def _graph_reasoning_terms(graph_reasoning: dict) -> list[str]:
    values = _graph_reasoning_node_terms(graph_reasoning, include_seed_entities=False)
    for path in graph_reasoning.get("paths", []) or []:
        if isinstance(path, dict):
            values.extend([
                path.get("explanation", ""),
                path.get("answer_hint", ""),
                *(path.get("evidence_families", []) or []),
                *(path.get("provenance_summary", []) or []),
            ])
            for relationship in path.get("relationships", []) or []:
                if isinstance(relationship, dict):
                    values.extend([
                        relationship.get("relation", ""),
                        relationship.get("provenance", ""),
                        relationship.get("evidence", ""),
                        relationship.get("rationale", ""),
                        relationship.get("provenance_summary", ""),
                        *(relationship.get("evidence_families", []) or []),
                    ])
                    metadata = relationship.get("metadata")
                    if isinstance(metadata, dict):
                        values.extend(str(metadata.get(key, "")) for key in GRAPH_PROVENANCE_METADATA_KEYS)
    for relationship in graph_reasoning.get("direct_relationships", []) or []:
        if isinstance(relationship, dict):
            values.extend([
                relationship.get("relation", ""),
                relationship.get("provenance", ""),
                relationship.get("evidence", ""),
            ])
            metadata = relationship.get("metadata")
            if isinstance(metadata, dict):
                values.extend(str(metadata.get(key, "")) for key in GRAPH_PROVENANCE_METADATA_KEYS)
    graph_overview = graph_reasoning.get("graph_overview") if isinstance(graph_reasoning.get("graph_overview"), dict) else {}
    for relationship in graph_overview.get("top_relationships", []) or []:
        if isinstance(relationship, dict):
            values.extend([
                relationship.get("source", ""),
                relationship.get("target", ""),
                relationship.get("relation", ""),
                relationship.get("provenance", ""),
                relationship.get("evidence", ""),
            ])
            metadata = relationship.get("metadata")
            if isinstance(metadata, dict):
                values.extend(str(metadata.get(key, "")) for key in GRAPH_PROVENANCE_METADATA_KEYS)
    for entity in graph_overview.get("shared_entities", []) or []:
        if isinstance(entity, dict):
            values.extend([
                entity.get("name", ""),
                *(entity.get("relationships", []) or []),
                *(entity.get("filenames", []) or []),
            ])
    for bridge in graph_overview.get("bridge_entities", []) or []:
        if isinstance(bridge, dict):
            values.extend([
                bridge.get("name", ""),
                *(bridge.get("relationships", []) or []),
                *(bridge.get("filenames", []) or []),
            ])
    for path in graph_overview.get("cross_document_paths", []) or []:
        if isinstance(path, dict):
            values.extend([
                path.get("source", ""),
                path.get("bridge", ""),
                path.get("target", ""),
                path.get("text", ""),
                *(path.get("filenames", []) or []),
            ])
            for relationship in path.get("relationships", []) or []:
                if isinstance(relationship, dict):
                    values.extend([
                        relationship.get("source", ""),
                        relationship.get("target", ""),
                        relationship.get("relation", ""),
                        relationship.get("evidence", ""),
                    ])
    return _unique_reasoning_terms(values, min_length=3)


def _graph_path_overlap_score(text: str, graph_terms: list[str]) -> tuple[float, list[str]]:
    if not text or not graph_terms:
        return 0.0, []
    text_lower = text.lower()
    matched = [term for term in graph_terms if term.lower() in text_lower]
    denominator = max(1, min(len(graph_terms), 8))
    return min(1.0, len(matched) / denominator), matched


def _chunk_graph_context_items(chunk: dict) -> list[dict]:
    graph_context = chunk.get("graph_context")
    if not isinstance(graph_context, list):
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        graph_context = metadata.get("graph_context")
    items = [item for item in (graph_context or []) if isinstance(item, dict)]
    expanded: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def add(edge: dict) -> None:
        key = (
            str(edge.get("source") or "").strip().lower(),
            str(edge.get("relation") or "").strip().lower(),
            str(edge.get("target") or "").strip().lower(),
        )
        if not all(key) or key in seen:
            return
        seen.add(key)
        expanded.append(edge)

    for item in items:
        direct = _direct_graph_context_edge_from_connector_wrapper(item)
        if direct:
            add(direct)
        add(item)
    return expanded


CONNECTOR_WRAPPER_PATTERN = re.compile(
    r"^\s*(?P<source>.+?)\s*(?:--?>|→)\s*(?P<target>.+?)(?:\s*:\s*(?P<relation>[A-Za-z][\w -]{1,48}))?\s*$"
)


def _direct_graph_context_edge_from_connector_wrapper(edge: dict) -> dict | None:
    relation_key = _graph_context_relation_key(edge)
    if relation_key != "contains_connector":
        return None

    provenance = edge.get("provenance") if isinstance(edge.get("provenance"), dict) else {}
    source = str(provenance.get("source_label") or "").strip()
    target = str(provenance.get("target_label") or "").strip()
    relation = str(
        provenance.get("connector_title")
        or provenance.get("relation")
        or provenance.get("connector_description")
        or ""
    ).strip()

    if not (source and target and relation):
        for value in (edge.get("target"), edge.get("evidence"), edge.get("text")):
            match = CONNECTOR_WRAPPER_PATTERN.match(str(value or ""))
            if not match:
                continue
            source = source or match.group("source").strip()
            target = target or match.group("target").strip()
            relation = relation or (match.group("relation") or "").strip()
            break

    relation_key = re.sub(r"[^a-z0-9]+", "_", relation.lower()).strip("_")
    if not (source and target and relation_key) or source.lower() == target.lower():
        return None
    if relation_key in {"connector", "connectors", "arrow", "line", "contains_connector"}:
        relation_key = "connects_to"

    direct = {key: value for key, value in edge.items() if key != "text"}
    direct.update({
        "source": source,
        "relation": relation_key,
        "target": target,
        "rank": edge.get("rank"),
        "legacy_connector_wrapper": True,
    })
    return direct


def _chunk_graph_path_items(chunk: dict) -> list[dict]:
    graph_paths = chunk.get("graph_paths")
    if not isinstance(graph_paths, list):
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        graph_paths = metadata.get("graph_paths")
    return [item for item in (graph_paths or []) if isinstance(item, dict)]


def _chunk_graph_context_text(graph_context: list[dict]) -> str:
    values: list[str] = []
    for edge in graph_context:
        values.extend([
            edge.get("source", ""),
            edge.get("relation", ""),
            edge.get("target", ""),
            edge.get("evidence", ""),
            edge.get("text", ""),
        ])
        provenance = edge.get("provenance")
        if isinstance(provenance, dict):
            values.extend(str(value) for value in provenance.values() if value not in (None, ""))
    return "\n".join(str(value) for value in values if value)


def _chunk_graph_path_text(graph_paths: list[dict]) -> str:
    values: list[str] = []
    for path in graph_paths:
        values.extend([
            path.get("source", ""),
            path.get("bridge", ""),
            path.get("target", ""),
            path.get("text", ""),
        ])
        values.extend(str(node) for node in path.get("nodes", []) or [] if node)
        for relationship in path.get("relationships", []) or []:
            if not isinstance(relationship, dict):
                continue
            values.extend([
                relationship.get("source", ""),
                relationship.get("relation", ""),
                relationship.get("target", ""),
                relationship.get("evidence", ""),
                relationship.get("text", ""),
            ])
            metadata = relationship.get("metadata")
            if isinstance(metadata, dict):
                values.extend(str(value) for value in metadata.values() if value not in (None, ""))
    return "\n".join(str(value) for value in values if value)


def _chunk_graph_context_query_terms(graph_context: list[dict], query: str) -> list[str]:
    query_tokens = {
        token
        for token in _query_tokens(query)
        if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
    }
    if not query_tokens or not graph_context:
        return []

    values = []
    for edge in graph_context:
        values.extend([
            edge.get("source", ""),
            edge.get("relation", ""),
            str(edge.get("relation", "")).replace("_", " "),
            edge.get("target", ""),
            edge.get("evidence", ""),
            edge.get("text", ""),
        ])
        provenance = edge.get("provenance")
        if isinstance(provenance, dict):
            values.extend(str(value) for value in provenance.values() if value not in (None, ""))
    edge_terms = {
        token
        for token in _query_tokens(" ".join(str(value) for value in values if value))
        if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
    }
    return sorted(query_tokens & edge_terms)


GENERIC_GRAPH_CONTEXT_RELATIONS = {
    "co_occurs",
    "co_occurs_with",
    "cooccurs",
    "mentioned_with",
    "mentions",
    "related_to",
}
DIRECT_GRAPH_CONTEXT_RELATIONS = {
    "blocks",
    "causes",
    "connects_to",
    "drives",
    "groups_visual",
    "gates",
    "influences",
    "links_to",
    "maps_to",
    "owns",
    "references",
    "supports",
    "unblocks",
    "uses",
}
STYLE_GRAPH_CONTEXT_RELATIONS = {"emphasizes", "uses_bullet_structure"}
GROUPED_VISUAL_GRAPH_CONTEXT_RELATIONS = {"groups_visual", "contains_shape_group"}
GROUPED_VISUAL_QUERY_TERMS = {
    "cluster",
    "evidence",
    "figure",
    "group",
    "grouped",
    "groups",
    "shape",
    "shapes",
    "support",
    "supported",
    "supporting",
    "supports",
    "visual",
    "visuals",
}


def _graph_context_relation_key(edge: dict) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(edge.get("relation") or "").lower()).strip("_")


def _is_answer_grade_graph_context_edge(edge: dict) -> bool:
    relation_key = _graph_context_relation_key(edge)
    if relation_key in GENERIC_GRAPH_CONTEXT_RELATIONS:
        return False
    if _is_structural_relation(relation_key):
        return False
    return float(edge.get("query_relevance_score") or 0) >= 0.9


def _graph_context_edge_sort_score(edge: dict, query: str) -> tuple[float, bool, float]:
    query_tokens = _token_variants({
        token
        for token in _query_tokens(query)
        if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
    })
    if not query_tokens:
        return (0.0, False, _safe_weight(edge.get("confidence", edge.get("weight", edge.get("score", 0)))))

    source = str(edge.get("source") or "")
    relation = str(edge.get("relation") or "")
    relation_key = _graph_context_relation_key(edge)
    target = str(edge.get("target") or "")
    evidence = str(edge.get("evidence") or edge.get("text") or "")
    source_tokens = _token_variants(_query_tokens(source))
    target_tokens = _token_variants(_query_tokens(target))
    endpoint_tokens = source_tokens | target_tokens
    relation_tokens = _token_variants(_query_tokens(relation.replace("_", " ")))
    evidence_tokens = _token_variants(_query_tokens(evidence))
    metadata = edge.get("metadata") if isinstance(edge.get("metadata"), dict) else {}
    metadata_tokens = _token_variants(_metadata_query_tokens(metadata))
    endpoint_hits = query_tokens & endpoint_tokens
    source_hits = query_tokens & source_tokens
    target_hits = query_tokens & target_tokens
    relation_hits = query_tokens & relation_tokens
    evidence_hits = query_tokens & evidence_tokens
    metadata_hits = query_tokens & metadata_tokens
    is_generic_relation = relation_key in GENERIC_GRAPH_CONTEXT_RELATIONS
    is_direct_relation = relation_key in DIRECT_GRAPH_CONTEXT_RELATIONS
    style_alias_hits = query_tokens & _token_variants(RELATION_QUERY_ALIASES.get(relation_key, set()))
    grouped_visual_requested = bool(query_tokens & GROUPED_VISUAL_QUERY_TERMS)
    is_grouped_visual_relation = relation_key in GROUPED_VISUAL_GRAPH_CONTEXT_RELATIONS or (
        grouped_visual_requested
        and any(metadata.get(key) for key in ("group_label", "parent_group_label", "group_title"))
    )
    is_style_relation = relation_key in STYLE_GRAPH_CONTEXT_RELATIONS
    style_relation_requested = bool(relation_hits or style_alias_hits)
    is_semantic = not _is_structural_relation(relation) and (
        not is_style_relation or style_relation_requested
    )
    confidence = _safe_weight(edge.get("confidence", edge.get("weight", edge.get("score", 0.6))))

    score = 0.0
    score += min(1.0, len(endpoint_hits) / max(1, min(len(query_tokens), 6))) * 0.42
    score += min(1.0, len(evidence_hits) / max(1, min(len(query_tokens), 6))) * 0.18
    score += min(1.0, len(relation_hits) / max(1, min(len(query_tokens), 3))) * 0.16
    score += min(1.0, len(metadata_hits) / max(1, min(len(query_tokens), 6))) * 0.12
    score += confidence * 0.14
    if source_hits and target_hits:
        score += 0.28
    if is_direct_relation and source_hits and target_hits:
        score += 0.34
    elif is_direct_relation and (endpoint_hits or relation_hits):
        score += 0.16
    if is_semantic and not is_generic_relation:
        score += 0.22
    elif is_generic_relation:
        score -= 0.18
    else:
        score -= 0.1
    if is_semantic and not is_generic_relation and len(endpoint_hits) >= 2:
        score += 0.16
    if is_style_relation and not style_relation_requested:
        score -= 0.26
    if is_grouped_visual_relation and grouped_visual_requested:
        score += 0.32
        if relation_key == "groups_visual":
            score += 0.16
        elif relation_key == "contains_shape_group":
            score += 0.08
    if any(len(value) > 90 or ". " in value or "|" in value for value in (source, target)):
        score -= 0.14
    return (round(score, 3), is_semantic and not is_generic_relation, confidence)


def _rank_chunk_graph_context_for_query(
    graph_context: list[dict],
    query: str,
    *,
    limit: int | None = None,
) -> list[dict]:
    if not graph_context:
        return []
    if not query:
        return graph_context[:limit] if limit is not None else list(graph_context)

    ranked: list[tuple[float, bool, float, int, dict]] = []
    for index, edge in enumerate(graph_context):
        if not isinstance(edge, dict):
            continue
        score, is_semantic, confidence = _graph_context_edge_sort_score(edge, query)
        ranked.append((score, is_semantic, confidence, -index, edge))
    ranked.sort(reverse=True)
    ordered = [
        {**edge, "query_relevance_score": score}
        for score, _, _, _, edge in ranked
    ]
    return ordered[:limit] if limit is not None else ordered


def _chunk_graph_path_query_terms(graph_paths: list[dict], query: str) -> list[str]:
    query_tokens = {
        token
        for token in _query_tokens(query)
        if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
    }
    if not query_tokens or not graph_paths:
        return []

    path_terms = {
        token
        for token in _query_tokens(_chunk_graph_path_text(graph_paths))
        if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
    }
    return sorted(query_tokens & path_terms)


def _chunk_retrieval_query(chunk: dict) -> str:
    retrieval_queries = chunk.get("retrieval_queries")
    if isinstance(retrieval_queries, list) and retrieval_queries:
        return str(retrieval_queries[0] or "")
    if isinstance(retrieval_queries, str):
        return retrieval_queries
    return ""


def _format_chunk_graph_context_relationship(edge: dict) -> str:
    if edge.get("text"):
        text = str(edge["text"])
    else:
        source = str(edge.get("source") or "")
        relation = str(edge.get("relation") or "related_to")
        target = str(edge.get("target") or "")
        text = f"{source} --[{relation}]--> {target}".strip()
        evidence = str(edge.get("evidence") or "")
        if evidence:
            text += f" (evidence: {evidence})"
    match_score = edge.get("chunk_match_score")
    match_reasons = edge.get("chunk_match_reasons")
    reason_values = [str(reason) for reason in match_reasons[:4] if reason] if isinstance(match_reasons, list) else []
    if match_score not in (None, "", 0) or reason_values:
        details = []
        if match_score not in (None, ""):
            details.append(f"chunk match score: {match_score}")
        if reason_values:
            details.append("reasons: " + ", ".join(reason_values))
        text += " [" + "; ".join(details) + "]"
    return text


def _format_chunk_graph_path(path: dict) -> str:
    if path.get("text"):
        text = str(path["text"])
    else:
        source = str(path.get("source") or "")
        bridge = str(path.get("bridge") or "")
        target = str(path.get("target") or "")
        text = f"{source} -> {bridge} -> {target}".strip()

    score = path.get("score")
    match_score = path.get("chunk_match_score")
    match_reasons = path.get("chunk_match_reasons")
    reason_values = [str(reason) for reason in match_reasons[:4] if reason] if isinstance(match_reasons, list) else []
    details = []
    if score not in (None, ""):
        details.append(f"path score: {score}")
    if match_score not in (None, "", 0):
        details.append(f"chunk match score: {match_score}")
    if reason_values:
        details.append("reasons: " + ", ".join(reason_values))
    if details:
        text += " [" + "; ".join(details) + "]"
    return text


def _score_chunks_with_graph_context(rag_chunks: list[dict], entities: list[dict], graph_reasoning: dict) -> list[dict]:
    graph_node_terms = _graph_reasoning_node_terms(graph_reasoning)
    graph_terms = _graph_reasoning_terms(graph_reasoning)
    all_entity_names = _unique_reasoning_terms([
        *(str(e.get("name", "")) for e in entities),
        *graph_node_terms,
    ])
    cross_references = []

    for i, chunk in enumerate(rag_chunks):
        retrieval_query = _chunk_retrieval_query(chunk)
        graph_context = _rank_chunk_graph_context_for_query(
            _chunk_graph_context_items(chunk),
            retrieval_query,
        )
        graph_paths = _chunk_graph_path_items(chunk)
        graph_context_text = _chunk_graph_context_text(graph_context)
        graph_path_text = _chunk_graph_path_text(graph_paths)
        searchable_content = "\n".join(part for part in [str(chunk.get("content", "")), graph_context_text, graph_path_text] if part)
        overlap, matched = _entity_overlap_score(searchable_content, all_entity_names)
        graph_overlap, matched_graph_terms = _graph_path_overlap_score(searchable_content, graph_terms)
        semantic_graph_context_scores = [
            float(edge.get("query_relevance_score") or 0)
            for edge in graph_context
            if isinstance(edge, dict) and _is_answer_grade_graph_context_edge(edge)
        ]
        graph_context_focus_score = round(max(semantic_graph_context_scores, default=0.0), 3)
        graph_context_answer_edge_count = sum(
            1 for edge in graph_context
            if isinstance(edge, dict) and _is_answer_grade_graph_context_edge(edge)
        )
        graph_context_query_terms = _chunk_graph_context_query_terms(graph_context, retrieval_query)
        graph_path_query_terms = _chunk_graph_path_query_terms(graph_paths, retrieval_query)
        if graph_context_query_terms or graph_path_query_terms:
            matched_graph_terms = _unique_reasoning_terms(
                [*matched_graph_terms, *graph_context_query_terms, *graph_path_query_terms],
                min_length=2,
            )
            context_term_count = len(set([*graph_context_query_terms, *graph_path_query_terms]))
            graph_overlap = max(graph_overlap, min(1.0, context_term_count / max(1, min(context_term_count + 2, 8))))
        entity_boost = round(overlap * 0.2, 3) if matched else 0
        graph_boost = round(graph_overlap * (0.32 if graph_context_query_terms or graph_path_query_terms else 0.25), 3) if matched_graph_terms else 0
        chunk["entity_boost"] = entity_boost
        chunk["graph_reasoning_boost"] = graph_boost
        chunk["graph_context_focus_score"] = graph_context_focus_score
        chunk["graph_context_answer_edge_count"] = graph_context_answer_edge_count
        chunk["boosted_score"] = round(
            chunk.get("score", 0)
            + entity_boost
            + graph_boost
            + min(0.12, graph_context_focus_score * 0.08),
            3,
        )
        chunk["matched_entities"] = matched
        chunk["matched_graph_path_terms"] = matched_graph_terms
        if graph_context:
            chunk["graph_context"] = graph_context
        if graph_paths:
            chunk["graph_paths"] = graph_paths

        if matched or matched_graph_terms:
            cross_references.append({
                "chunk_index": i,
                "chunk_id": chunk.get("id", ""),
                "matched_entities": matched,
                "matched_graph_path_terms": matched_graph_terms,
                "matched_chunk_graph_context_terms": graph_context_query_terms,
                "matched_chunk_graph_path_terms": graph_path_query_terms,
                "graph_context_relationship_count": len(graph_context),
                "graph_path_context_count": len(graph_paths),
                "overlap_score": round(overlap, 3),
                "graph_path_overlap_score": round(graph_overlap, 3),
            })

    rag_chunks.sort(
        key=lambda c: (
            c.get("graph_context_answer_edge_count", 0),
            c.get("boosted_score", c.get("score", 0)),
            c.get("graph_context_focus_score", 0),
            c.get("score", 0),
        ),
        reverse=True,
    )
    return cross_references


def _chunk_path_as_reasoning_path(path: dict, chunk: dict, query: str) -> dict | None:
    if not isinstance(path, dict):
        return None
    nodes = [str(node).strip() for node in path.get("nodes", []) or [] if str(node or "").strip()]
    if len(nodes) < 2:
        source = str(path.get("source") or "").strip()
        bridge = str(path.get("bridge") or "").strip()
        target = str(path.get("target") or "").strip()
        nodes = [node for node in (source, bridge, target) if node]
    if len(nodes) < 2:
        return None

    chunk_metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    chunk_provenance = chunk_metadata.get("provenance")
    provenance_values = [
        str(item)
        for item in chunk_provenance
        if item
    ] if isinstance(chunk_provenance, list) else ([str(chunk_provenance)] if chunk_provenance else [])
    filename = str(chunk.get("document_title") or chunk_metadata.get("filename") or "")
    relationships: list[dict] = []
    for relationship in path.get("relationships", []) or []:
        if not isinstance(relationship, dict):
            continue
        metadata = relationship.get("metadata") if isinstance(relationship.get("metadata"), dict) else {}
        normalized = {
            **relationship,
            "source": relationship.get("source") or relationship.get("traversal_source"),
            "relation": relationship.get("relation") or relationship.get("relationship_type") or "related_to",
            "target": relationship.get("target") or relationship.get("traversal_target"),
            "metadata": {
                **chunk_metadata,
                **metadata,
            },
        }
        if filename and not normalized["metadata"].get("local_graph_filename"):
            normalized["metadata"]["local_graph_filename"] = filename
        if not normalized.get("evidence"):
            normalized["evidence"] = chunk.get("content") or path.get("text") or "Graph path evidence"
        if provenance_values and not normalized.get("provenance"):
            normalized["provenance"] = "; ".join(provenance_values[:3])
        normalized["evidence_families"] = relationship.get("evidence_families") or _graph_edge_evidence_families(normalized)
        normalized["provenance_summary"] = relationship.get("provenance_summary") or _graph_edge_provenance_summary(normalized)
        normalized["rationale"] = relationship.get("rationale") or _graph_edge_rationale(normalized)
        relationships.append(normalized)

    if not relationships and len(nodes) >= 2:
        relationships = [
            {
                "source": nodes[index],
                "relation": "connects_to",
                "target": nodes[index + 1],
                "evidence": chunk.get("content") or path.get("text") or "Graph path evidence",
                "metadata": {
                    **chunk_metadata,
                    "local_graph_filename": filename or chunk_metadata.get("local_graph_filename"),
                },
                "provenance": "; ".join(provenance_values[:3]) if provenance_values else "",
            }
            for index in range(len(nodes) - 1)
        ]
        for relationship in relationships:
            relationship["evidence_families"] = _graph_edge_evidence_families(relationship)
            relationship["provenance_summary"] = _graph_edge_provenance_summary(relationship)
            relationship["rationale"] = _graph_edge_rationale(relationship)

    try:
        score = float(path.get("score") or chunk.get("boosted_score") or chunk.get("score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    promoted = {
        **path,
        "source": path.get("source") or nodes[0],
        "target": path.get("target") or nodes[-1],
        "nodes": nodes,
        "relationships": relationships,
        "hops": int(path.get("hops") or len(relationships) or max(1, len(nodes) - 1)),
        "score": score,
        "path_type": path.get("path_type") or "semantic",
        "matched_query_terms": _unique_reasoning_terms([
            *(path.get("matched_query_terms") or []),
            *(chunk.get("matched_graph_path_terms") or []),
            *_chunk_graph_path_query_terms([path], query),
        ], min_length=2)[:10],
        "provenance_summary": path.get("provenance_summary") or provenance_values[:4],
        "source_chunk_id": chunk.get("id"),
        "source_document": filename or None,
    }
    if not promoted.get("text"):
        promoted["text"] = " -> ".join(
            f"{relationship.get('source')} --[{relationship.get('relation')}]-- {relationship.get('target')}"
            for relationship in relationships
            if relationship.get("source") and relationship.get("target")
        ) or " -> ".join(nodes)
    if not promoted.get("traversal_text"):
        promoted["traversal_text"] = " -> ".join(nodes)
    promoted.update(_graph_path_explanation(promoted))
    return promoted


def _promote_chunk_graph_paths_to_reasoning(graph_reasoning: dict, rag_chunks: list[dict], query: str) -> dict:
    if not isinstance(graph_reasoning, dict) or not rag_chunks:
        return graph_reasoning

    promoted_paths: list[dict] = []
    for chunk in rag_chunks:
        if not isinstance(chunk, dict):
            continue
        graph_paths = _chunk_graph_path_items(chunk)
        if not graph_paths:
            continue
        retrieval_reasons = chunk.get("retrieval_reasons") if isinstance(chunk.get("retrieval_reasons"), list) else []
        if (
            "local_pipeline_graph_path_context" not in retrieval_reasons
            and not chunk.get("matched_graph_path_terms")
            and not _chunk_graph_path_query_terms(graph_paths, query)
        ):
            continue
        for path in graph_paths:
            promoted_path = _chunk_path_as_reasoning_path(path, chunk, query)
            if promoted_path:
                promoted_paths.append(promoted_path)

    if not promoted_paths:
        return graph_reasoning

    merged_paths: dict[tuple, dict] = {}
    for path in [
        *[path for path in graph_reasoning.get("paths", []) or [] if isinstance(path, dict)],
        *promoted_paths,
    ]:
        nodes = tuple(str(node).lower() for node in path.get("nodes", []) or [])
        relationships = tuple(
            (
                str(relationship.get("source") or "").lower(),
                str(relationship.get("relation") or "").lower(),
                str(relationship.get("target") or "").lower(),
            )
            for relationship in path.get("relationships", []) or []
            if isinstance(relationship, dict)
        )
        key = (nodes, relationships)
        existing = merged_paths.get(key)
        if existing is None or _graph_path_rank_key(path, query) > _graph_path_rank_key(existing, query):
            merged_paths[key] = path

    merged_path_values = sorted(
        merged_paths.values(),
        key=lambda item: _graph_path_rank_key(item, query),
        reverse=True,
    )
    graph_reasoning["paths"] = merged_path_values[:12]
    path_relationships = [
        relationship
        for path in graph_reasoning["paths"]
        for relationship in path.get("relationships", []) or []
        if isinstance(relationship, dict)
    ]
    graph_reasoning["direct_relationships"] = _merge_graph_relationships(
        graph_reasoning.get("direct_relationships", []) or [],
        path_relationships,
        limit=16,
    )
    graph_reasoning["bridge_entities"] = _bridge_entities_from_paths(graph_reasoning["paths"])
    graph_reasoning["status"] = "complete"
    graph_reasoning["reasoning_needed"] = True
    graph_reasoning = _finalize_graph_reasoning_result(graph_reasoning)
    graph_reasoning["chunk_graph_path_reasoning_promoted"] = True
    return graph_reasoning


def _graph_augmented_retrieval_terms(graph_reasoning: dict, *, limit: int = 10) -> list[str]:
    values: list[str] = []

    def add_relationship_values(relationship: dict) -> None:
        values.extend([
            relationship.get("relation", ""),
            relationship.get("source", ""),
            relationship.get("target", ""),
            relationship.get("evidence", ""),
            relationship.get("provenance", ""),
            relationship.get("rationale", ""),
            relationship.get("provenance_summary", ""),
            *(relationship.get("evidence_families", []) or []),
        ])
        metadata = relationship.get("metadata") if isinstance(relationship.get("metadata"), dict) else {}
        values.extend(str(metadata.get(key, "")) for key in GRAPH_PROVENANCE_METADATA_KEYS)

    for bridge in graph_reasoning.get("bridge_entities", []) or []:
        if isinstance(bridge, dict):
            values.extend([
                bridge.get("name", ""),
                *(bridge.get("relationships", []) or []),
            ])

    for path in sorted(graph_reasoning.get("paths", []) or [], key=lambda item: item.get("score", 0), reverse=True):
        if not isinstance(path, dict):
            continue
        nodes = path.get("nodes", []) or []
        values.extend(nodes[1:-1] if len(nodes) > 2 else nodes)
        values.extend([
            path.get("explanation", ""),
            path.get("answer_hint", ""),
            *(path.get("evidence_families", []) or []),
            *(path.get("provenance_summary", []) or []),
        ])
        for relationship in path.get("relationships", []) or []:
            if isinstance(relationship, dict):
                add_relationship_values(relationship)

    for relationship in graph_reasoning.get("direct_relationships", []) or []:
        if isinstance(relationship, dict):
            add_relationship_values(relationship)

    graph_overview = graph_reasoning.get("graph_overview") if isinstance(graph_reasoning.get("graph_overview"), dict) else {}
    for relationship in graph_overview.get("top_relationships", []) or []:
        if isinstance(relationship, dict):
            add_relationship_values(relationship)
    for entity in graph_overview.get("shared_entities", []) or []:
        if isinstance(entity, dict):
            values.extend([
                entity.get("name", ""),
                *(entity.get("relationships", []) or []),
                *(entity.get("filenames", []) or []),
            ])

    for handoff in graph_reasoning.get("interaction_handoffs", []) or []:
        if not isinstance(handoff, dict):
            continue
        values.extend([
            handoff.get("filename", ""),
            handoff.get("recommended_mode", ""),
            handoff.get("answer_guidance", ""),
            *(handoff.get("citation_scopes", []) or []),
        ])
        for card in handoff.get("capability_cards", []) or []:
            if isinstance(card, dict):
                values.extend([
                    card.get("label", ""),
                    card.get("detail", ""),
                    card.get("kind", ""),
                ])
        for anchor in handoff.get("graph_anchors", []) or []:
            if not isinstance(anchor, dict):
                continue
            values.extend([
                anchor.get("title", ""),
                anchor.get("label", ""),
                anchor.get("source", ""),
                anchor.get("relation", ""),
                anchor.get("target", ""),
                anchor.get("evidence", ""),
                *(anchor.get("nodes", []) or []),
                *(anchor.get("neighbours", []) or []),
            ])
            metadata = anchor.get("metadata") if isinstance(anchor.get("metadata"), dict) else {}
            values.extend(str(value) for value in metadata.values() if value not in (None, ""))
        for prompt in handoff.get("starter_prompts", []) or []:
            if isinstance(prompt, dict):
                values.extend([
                    prompt.get("label", ""),
                    prompt.get("kind", ""),
                    prompt.get("prompt", ""),
                ])
        for rehearsal in handoff.get("retrieval_rehearsals", []) or []:
            if not isinstance(rehearsal, dict):
                continue
            values.extend([
                rehearsal.get("label", ""),
                rehearsal.get("kind", ""),
                rehearsal.get("prompt", ""),
                *(rehearsal.get("matched_terms", []) or []),
            ])
            for anchor in rehearsal.get("top_anchors", []) or []:
                if not isinstance(anchor, dict):
                    continue
                values.extend([
                    anchor.get("title", ""),
                    anchor.get("source", ""),
                    anchor.get("relation", ""),
                    anchor.get("target", ""),
                    anchor.get("evidence", ""),
                    *(anchor.get("matched_terms", []) or []),
                ])
                metadata = anchor.get("metadata") if isinstance(anchor.get("metadata"), dict) else {}
                values.extend(str(value) for value in metadata.values() if value not in (None, ""))
            for chunk in rehearsal.get("top_chunks", []) or []:
                if not isinstance(chunk, dict):
                    continue
                values.extend([
                    chunk.get("chunk_id", ""),
                    chunk.get("evidence_preview", ""),
                    *(chunk.get("matched_terms", []) or []),
                    *(chunk.get("source_elements", []) or []),
                ])
                provenance = chunk.get("provenance")
                if isinstance(provenance, list):
                    values.extend(str(value) for value in provenance if value)
                elif provenance:
                    values.append(str(provenance))

    pipeline_evidence = graph_reasoning.get("pipeline_evidence") if isinstance(graph_reasoning.get("pipeline_evidence"), dict) else {}
    values.extend(_pipeline_evidence_retrieval_terms(pipeline_evidence))
    values.extend(_graph_reasoning_terms(graph_reasoning))
    return _unique_reasoning_terms(values, min_length=3)[:limit]


def _graph_augmented_retrieval_query(query: str, graph_reasoning: dict, *, limit: int = 10) -> str:
    terms = _graph_augmented_retrieval_terms(graph_reasoning, limit=limit)
    if not terms:
        return query
    return f"{query}\nGraph path evidence terms: {', '.join(terms)}"


def _graph_reasoning_summary(graph_reasoning: dict) -> dict:
    paths = [path for path in graph_reasoning.get("paths", []) or [] if isinstance(path, dict)]
    direct_relationships = [
        relationship
        for relationship in graph_reasoning.get("direct_relationships", []) or []
        if isinstance(relationship, dict)
    ]
    graph_overview = graph_reasoning.get("graph_overview") if isinstance(graph_reasoning.get("graph_overview"), dict) else {}
    overview_relationships = [
        relationship
        for relationship in graph_overview.get("top_relationships", []) or []
        if isinstance(relationship, dict)
    ]
    shared_entities = [
        entity
        for entity in graph_overview.get("shared_entities", []) or []
        if isinstance(entity, dict)
    ]
    overview_bridges = [
        bridge
        for bridge in graph_overview.get("bridge_entities", []) or []
        if isinstance(bridge, dict)
    ]
    overview_paths = [
        path
        for path in graph_overview.get("cross_document_paths", []) or []
        if isinstance(path, dict)
    ]
    bridges = [bridge for bridge in graph_reasoning.get("bridge_entities", []) or [] if isinstance(bridge, dict)]
    interaction_handoffs = [
        handoff
        for handoff in graph_reasoning.get("interaction_handoffs", []) or []
        if isinstance(handoff, dict)
    ]
    reasoning_query = str(graph_reasoning.get("query") or graph_reasoning.get("reasoning_query") or "")
    relationship_focus = _query_prefers_direct_relationships(reasoning_query)
    strongest_path = paths[0] if paths else None
    pair_direct_relationships = _entity_pair_direct_relationships(
        direct_relationships,
        graph_reasoning,
    )
    domain_direct_relationships = _domain_answer_relationships(
        _merge_graph_relationships(direct_relationships, overview_relationships, limit=24),
        limit=3,
    )
    top_direct = pair_direct_relationships[:3] if pair_direct_relationships else (
        domain_direct_relationships[:3]
        if relationship_focus and domain_direct_relationships
        else (
        _path_supporting_relationships(direct_relationships, strongest_path, limit=3)
        if strongest_path
        else direct_relationships[:3]
        )
    )
    top_bridges = bridges[:3]
    pipeline_evidence = graph_reasoning.get("pipeline_evidence") if isinstance(graph_reasoning.get("pipeline_evidence"), dict) else {}
    evidence: list[str] = []

    if pair_direct_relationships:
        evidence.append(
            "Direct relationships: "
            + "; ".join(
                _format_graph_relationship_for_answer(rel, include_provenance=False)
                for rel in top_direct
            )
        )
        if strongest_path:
            evidence.append(
                f"Related graph path: {_format_graph_path_for_answer(strongest_path) or strongest_path.get('text', '')} "
                f"(score: {strongest_path.get('score', 0)}; hops: {strongest_path.get('hops', 0)})"
            )
            if strongest_path.get("explanation"):
                evidence.append(f"Path explanation: {strongest_path.get('explanation')}")
    elif relationship_focus and domain_direct_relationships:
        evidence.append(
            "Direct relationships: "
            + "; ".join(
                _format_graph_relationship_for_answer(rel, include_provenance=False)
                for rel in top_direct
            )
        )
        if strongest_path:
            evidence.append(
                f"Related graph path: {_format_graph_path_for_answer(strongest_path) or strongest_path.get('text', '')} "
                f"(score: {strongest_path.get('score', 0)}; hops: {strongest_path.get('hops', 0)})"
            )
    elif strongest_path:
        evidence.append(
            f"Strongest path: {_format_graph_path_for_answer(strongest_path) or strongest_path.get('text', '')} "
            f"(score: {strongest_path.get('score', 0)}; hops: {strongest_path.get('hops', 0)})"
        )
        if strongest_path.get("explanation"):
            evidence.append(f"Path explanation: {strongest_path.get('explanation')}")
    if top_bridges:
        evidence.append(
            "Bridge entities: "
            + "; ".join(
                f"{bridge.get('name')} links {bridge.get('path_count', 0)} path(s)"
                for bridge in top_bridges
            )
        )
    if top_direct and not pair_direct_relationships:
        evidence.append(
            "Direct relationships: "
            + "; ".join(
                _format_graph_relationship_for_answer(rel, include_provenance=False)
                for rel in top_direct
            )
        )
    if overview_relationships:
        overview_bits = [
            f"{graph_overview.get('relationship_count', len(overview_relationships))} relationship(s)",
            f"{graph_overview.get('source_count', 0)} pipeline trace(s)",
        ]
        if shared_entities:
            overview_bits.append(
                "shared entities: "
                + ", ".join(str(entity.get("name")) for entity in shared_entities[:5] if entity.get("name"))
            )
        if overview_bridges:
            overview_bits.append(
                "bridge entities: "
                + ", ".join(str(bridge.get("name")) for bridge in overview_bridges[:5] if bridge.get("name"))
            )
        if overview_paths:
            overview_bits.append(f"{len(overview_paths)} cross-document path candidate(s)")
        evidence.append("Local pipeline graph overview: " + "; ".join(overview_bits))
    if interaction_handoffs:
        handoff_bits = []
        for handoff in interaction_handoffs[:4]:
            counts = handoff.get("counts") if isinstance(handoff.get("counts"), dict) else {}
            handoff_bits.append(
                f"{handoff.get('filename') or handoff.get('run_id')} "
                f"({handoff.get('recommended_mode', 'handoff')}; "
                f"{counts.get('graph_anchors', len(handoff.get('graph_anchors', []) or []))} anchor(s))"
            )
        evidence.append("Pipeline interaction handoff: " + "; ".join(handoff_bits))
    pipeline_evidence_summary = _pipeline_evidence_summary_text(pipeline_evidence)
    if pipeline_evidence_summary:
        evidence.append("Pipeline evidence coverage: " + pipeline_evidence_summary)

    if pair_direct_relationships:
        mode = "direct_relationship"
        answer_guidance = "Answer from the exact direct source-target graph relationship first; use graph paths only as secondary context."
    elif relationship_focus and domain_direct_relationships:
        mode = "direct_relationship"
        answer_guidance = "Answer from the strongest direct domain relationship first; mention graph paths only as secondary provenance or context."
    elif paths:
        mode = "multi_hop_path"
        answer_guidance = "Use the strongest path first, then cite bridge entities and supporting relationships."
    elif direct_relationships:
        mode = "direct_relationship"
        answer_guidance = "Answer from the direct graph relationships and cite the strongest relationship evidence."
    elif overview_relationships:
        mode = "graph_overview"
        answer_guidance = "Compare the local pipeline graph overview, cite top relationships, shared entities, and file-level provenance."
    elif interaction_handoffs:
        mode = "interaction_handoff"
        handoff_guidance = next(
            (
                str(handoff.get("answer_guidance"))
                for handoff in interaction_handoffs
                if handoff.get("answer_guidance")
            ),
            "",
        )
        answer_guidance = handoff_guidance or "Use the pipeline handoff graph anchors, starter prompts, and citation scopes as the answer plan."
    elif pipeline_evidence:
        mode = "pipeline_evidence"
        answer_guidance = "Use the pipeline evidence coverage to choose citations, but do not invent graph edges that are not present."
    elif graph_reasoning.get("reasoning_needed"):
        mode = "no_graph_evidence"
        answer_guidance = "Say that no connecting graph path was found and fall back to retrieved passages or extracted entities."
    else:
        mode = "not_requested"
        answer_guidance = "Graph path reasoning was not needed for this query."

    summary_text = evidence[0] if evidence else answer_guidance
    return {
        "mode": mode,
        "answer_guidance": answer_guidance,
        "summary_text": summary_text,
        "evidence": evidence,
        "key_entities": _graph_reasoning_node_terms(graph_reasoning)[:10],
        "evidence_terms": _graph_augmented_retrieval_terms(graph_reasoning, limit=10),
        "bridge_entities": [str(bridge.get("name")) for bridge in top_bridges if bridge.get("name")],
        "path_count": len(paths),
        "direct_relationship_count": len(direct_relationships),
        "overview_relationship_count": len(overview_relationships),
        "shared_entity_count": len(shared_entities),
        "overview_bridge_entity_count": len(overview_bridges),
        "overview_cross_document_path_count": len(overview_paths),
        "bridge_entity_count": len(bridges),
        "interaction_handoff_count": len(interaction_handoffs),
        "pipeline_evidence_family_count": len(pipeline_evidence.get("families", []) or []),
    }


def _compact_graph_evidence(value: str | None, limit: int = 320) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return f"{text[:max(1, limit - 3)].strip()}..."


DOCUMENT_LIKE_GRAPH_LABEL_RE = re.compile(
    r"\.(?:pdf|pptx?|docx?|xlsx?|csv|tsv|md|markdown|txt|html?|json|xml|eml|msg|zip)(?:$|[?#\s])",
    re.IGNORECASE,
)


def _reference_basename_for_display(value: str) -> str:
    parts = [part for part in re.split(r"[\\/]+", str(value or "")) if part]
    return parts[-1] if parts else str(value or "")


def _is_document_like_graph_label(value: str | None) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return bool(text and DOCUMENT_LIKE_GRAPH_LABEL_RE.search(_reference_basename_for_display(text)))


def _clean_graph_reference_label(value: str | None, fallback: str = "Source") -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return fallback
    return _reference_basename_for_display(text) if _is_document_like_graph_label(text) else text


def _display_graph_path_nodes(nodes: list[str]) -> list[str]:
    cleaned_nodes: list[str] = []
    seen: set[str] = set()
    for node in nodes or []:
        cleaned = _clean_graph_reference_label(str(node or ""), "")
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        cleaned_nodes.append(cleaned)
    if len(cleaned_nodes) <= 2:
        return cleaned_nodes

    display_nodes: list[str] = []
    for index, node in enumerate(cleaned_nodes):
        if _is_document_like_graph_label(node):
            label = "source evidence" if 0 < index < len(cleaned_nodes) - 1 else "source document"
        else:
            label = node
        if not display_nodes or display_nodes[-1] != label:
            display_nodes.append(label)
    return display_nodes


def _display_graph_endpoint(value: str | None, *, bridge: bool = False) -> str:
    cleaned = _clean_graph_reference_label(value, "Source")
    if _is_document_like_graph_label(cleaned):
        return "source evidence" if bridge else "source document"
    return cleaned


def _relationship_location_metadata(metadata: dict | None) -> dict[str, object]:
    if not isinstance(metadata, dict):
        return {}

    is_connector = any(
        metadata.get(key) not in (None, "", [], {})
        for key in ("connector_title", "connector_name", "connector_description", "connector_id", "connector_index")
    )
    is_diagram = any(
        metadata.get(key) not in (None, "", [], {})
        for key in (
            "diagram_part",
            "diagram_node_id",
            "diagram_node_type",
            "diagram_order",
            "diagram_level",
            "parent_diagram_node_id",
            "parent_label",
            "diagram_connection_type",
            "source_diagram_node_id",
            "target_diagram_node_id",
        )
    )
    fields = {
        "filename": metadata.get("local_graph_filename") or metadata.get("filename") or metadata.get("source_file"),
        "run_id": metadata.get("local_graph_run_id"),
        "domain": metadata.get("local_graph_domain"),
        "slide": metadata.get("slide"),
        "slide_title": metadata.get("slide_title"),
        "slide_visibility": metadata.get("slide_visibility") or metadata.get("slide_state"),
        "slide_part": metadata.get("slide_part"),
        "slide_relationship_id": metadata.get("slide_relationship_id"),
        "slide_id": metadata.get("slide_id"),
        "page": metadata.get("page"),
        "sheet": metadata.get("sheet_name") or metadata.get("sheet"),
        "sheet_visibility": metadata.get("sheet_visibility") or metadata.get("sheet_state"),
        "sheet_part": metadata.get("sheet_part"),
        "sheet_relationship_id": metadata.get("sheet_relationship_id"),
        "sheet_id": metadata.get("sheet_id"),
        "section": metadata.get("section_label") or metadata.get("section"),
        "table": metadata.get("table"),
        "table_name": metadata.get("table_display_name") or metadata.get("table_name"),
        "table_range": metadata.get("table_range"),
        "table_part": metadata.get("table_part"),
        "row": metadata.get("row"),
        "table_source": metadata.get("table_source"),
        "table_cell_fill_color": metadata.get("table_cell_fill_color"),
        "table_cell_style_name": metadata.get("table_cell_style_name"),
        "table_cell_text_anchor": metadata.get("table_cell_text_anchor"),
        "table_cell_vertical_align": metadata.get("table_cell_vertical_align"),
        "table_cell_text_direction": metadata.get("table_cell_text_direction"),
        "table_cell_width_value": metadata.get("table_cell_width_value"),
        "table_cell_width_inches": metadata.get("table_cell_width_inches"),
        "table_cell_width_type": metadata.get("table_cell_width_type"),
        "table_cell_shading_pattern": metadata.get("table_cell_shading_pattern"),
        "table_cell_visual_summary": metadata.get("table_cell_visual_summary"),
        "formula_cell": metadata.get("formula_cell"),
        "formula_expression": metadata.get("formula_expression"),
        "formula_result": metadata.get("formula_result"),
        "formula_references": metadata.get("formula_references"),
        "conditional_range": metadata.get("conditional_range_label") or metadata.get("conditional_range"),
        "conditional_type": metadata.get("conditional_type"),
        "conditional_formula": metadata.get("conditional_formula1"),
        "pivot_table": metadata.get("pivot_table_name"),
        "pivot_table_range": metadata.get("pivot_table_range"),
        "pivot_table_source": metadata.get("pivot_table_source_name"),
        "pivot_table_source_ref": metadata.get("pivot_table_source_ref"),
        "pivot_table_source_sheet": metadata.get("pivot_table_source_sheet"),
        "pivot_axis": metadata.get("pivot_axis"),
        "pivot_subtotal": metadata.get("pivot_subtotal"),
        "named_range": metadata.get("named_range_name"),
        "named_range_ref": metadata.get("named_range_ref"),
        "named_range_scope": metadata.get("named_range_scope"),
        "form_kind": metadata.get("form_kind"),
        "field_type": metadata.get("field_type"),
        "label": metadata.get("label"),
        "form_field_alias": metadata.get("form_field_alias"),
        "form_field_tag": metadata.get("form_field_tag"),
        "form_field_id": metadata.get("form_field_id"),
        "form_field_placeholder": metadata.get("form_field_placeholder"),
        "form_field_checked": metadata.get("form_field_checked"),
        "form_field_options": metadata.get("form_field_options"),
        "form_field_option_values": metadata.get("form_field_option_values"),
        "form_field_selected_option": metadata.get("form_field_selected_option"),
        "form_field_date_format": metadata.get("form_field_date_format"),
        "form_field_date_full_value": metadata.get("form_field_date_full_value"),
        "form_field_calendar": metadata.get("form_field_calendar"),
        "form_field_lock": metadata.get("form_field_lock"),
        "control_id": metadata.get("control_id"),
        "control_name": metadata.get("control_name"),
        "control_part": metadata.get("control_part"),
        "validation_range": metadata.get("validation_range"),
        "validation_type": metadata.get("validation_type"),
        "chart_title": metadata.get("chart_title"),
        "chart_label": metadata.get("chart_label"),
        "chart_point_label": metadata.get("chart_point_label"),
        "series_name": metadata.get("series_name"),
        "category": metadata.get("category"),
        "value": metadata.get("value"),
        "chart_part": metadata.get("chart_part"),
        "chart_series_visual_summary": metadata.get("chart_series_visual_summary"),
        "chart_series_fill_color": metadata.get("chart_series_fill_color"),
        "chart_series_line_color": metadata.get("chart_series_line_color"),
        "chart_point_visual_summary": metadata.get("chart_point_visual_summary"),
        "chart_point_fill_color": metadata.get("chart_point_fill_color"),
        "chart_marker_symbol": metadata.get("chart_marker_symbol"),
        "chart_trendline_type": metadata.get("chart_trendline_type"),
        "chart_trendline_line_color": metadata.get("chart_trendline_line_color"),
        "chart_trendline_line_dash": metadata.get("chart_trendline_line_dash"),
        "chart_trendline_visual_summary": metadata.get("chart_trendline_visual_summary"),
        "shape": metadata.get("shape_title") or metadata.get("shape_name"),
        "shape_description": metadata.get("shape_description"),
        "connector": metadata.get("connector_title") or metadata.get("connector_name"),
        "connector_description": metadata.get("connector_description"),
        "connector_source": metadata.get("source_label") if is_connector else None,
        "connector_target": metadata.get("target_label") if is_connector else None,
        "notes_part": metadata.get("notes_part") or metadata.get("note_part"),
        "notes_relationship_id": metadata.get("notes_relationship_id"),
        "note_id": metadata.get("note_id"),
        "comment_id": metadata.get("comment_id"),
        "comment_kind": metadata.get("comment_kind"),
        "comment_part": metadata.get("comment_part"),
        "comment_author": metadata.get("author") or metadata.get("comment_author") or metadata.get("annotation_author"),
        "comment_datetime": metadata.get("comment_datetime") or metadata.get("date"),
        "comment_anchor": (
            metadata.get("comment_anchor_text")
            or metadata.get("comment_anchor_cell")
            or metadata.get("comment_anchor_section")
            or metadata.get("note_anchor_text")
            or metadata.get("note_anchor_section")
        ),
        "threaded_comment_id": metadata.get("threaded_comment_id"),
        "threaded_parent_id": metadata.get("threaded_parent_id"),
        "revision_type": metadata.get("revision_type"),
        "revision_author": metadata.get("revision_author"),
        "revision_date": metadata.get("revision_date"),
        "revision_id": metadata.get("revision_id"),
        "annotation_type": metadata.get("annotation_type"),
        "annotation_author": metadata.get("annotation_author"),
        "annotation_subject": metadata.get("annotation_subject"),
        "annotation_name": metadata.get("annotation_name"),
        "style_summary": metadata.get("style_summary"),
        "styled_text": metadata.get("styled_text"),
        "emphasis_term": metadata.get("emphasis_term"),
        "emphasis_terms": metadata.get("emphasis_terms"),
        "emphasis_styles": metadata.get("emphasis_styles"),
        "bullet_levels": metadata.get("bullet_levels"),
        "media_part": metadata.get("media_part"),
        "media_kind": metadata.get("media_kind"),
        "media_mime": metadata.get("media_mime"),
        "media_size_bytes": metadata.get("media_size_bytes"),
        "email_header": metadata.get("email_header"),
        "email_message_id": metadata.get("email_message_id"),
        "email_subject": metadata.get("email_subject"),
        "email_from": metadata.get("email_from"),
        "email_sender": metadata.get("email_sender"),
        "email_to": metadata.get("email_to"),
        "email_cc": metadata.get("email_cc"),
        "email_bcc": metadata.get("email_bcc"),
        "email_reply_to": metadata.get("email_reply_to"),
        "email_date": metadata.get("email_date"),
        "email_date_iso": metadata.get("email_date_iso"),
        "email_in_reply_to": metadata.get("email_in_reply_to"),
        "email_references": metadata.get("email_references"),
        "email_part_index": metadata.get("email_part_index"),
        "email_part_kind": metadata.get("email_part_kind"),
        "email_content_type": metadata.get("email_content_type"),
        "email_disposition": metadata.get("email_disposition"),
        "email_attachment_index": metadata.get("email_attachment_index"),
        "email_attachment_filename": metadata.get("email_attachment_filename"),
        "email_attachment_mime": metadata.get("email_attachment_mime"),
        "uri": metadata.get("uri"),
        "link_kind": metadata.get("link_kind"),
        "markdown_code_language": metadata.get("markdown_code_language"),
        "markdown_code_block_index": metadata.get("markdown_code_block_index"),
        "markdown_start_line": metadata.get("markdown_start_line"),
        "markdown_end_line": metadata.get("markdown_end_line"),
        "markdown_frontmatter_index": metadata.get("markdown_frontmatter_index"),
        "image_part": metadata.get("image_part"),
        "image_relationship_id": metadata.get("image_relationship_id"),
        "layout_bbox": metadata.get("layout_bbox"),
        "layout_zone": metadata.get("layout_zone"),
        "diagram_part": metadata.get("diagram_part"),
        "diagram_node_id": metadata.get("diagram_node_id"),
        "diagram_node_type": metadata.get("diagram_node_type"),
        "diagram_order": metadata.get("diagram_order"),
        "diagram_level": metadata.get("diagram_level"),
        "parent_diagram_node_id": metadata.get("parent_diagram_node_id"),
        "parent_label": metadata.get("parent_label"),
        "diagram_connection_type": metadata.get("diagram_connection_type"),
        "source_diagram_node_id": metadata.get("source_diagram_node_id"),
        "target_diagram_node_id": metadata.get("target_diagram_node_id"),
        "diagram_source": metadata.get("source_label") if is_diagram else None,
        "diagram_target": metadata.get("target_label") if is_diagram else None,
        "attachment": metadata.get("embedded_file_filename") or metadata.get("embedded_file_name"),
        "archive_entry": metadata.get("archive_entry_path") or metadata.get("archive_entry_filename"),
        "legacy_office_format": metadata.get("legacy_office_format"),
        "legacy_office_stream_path": metadata.get("legacy_office_stream_path"),
        "legacy_office_stream_index": metadata.get("legacy_office_stream_index"),
        "legacy_office_extraction_method": metadata.get("legacy_office_extraction_method"),
        "legacy_office_stream_size_bytes": metadata.get("legacy_office_stream_size_bytes"),
        "legacy_office_stream_sha256": metadata.get("legacy_office_stream_sha256"),
        "legacy_office_string_count": metadata.get("legacy_office_string_count"),
        "ocr_source": metadata.get("ocr_source"),
        "element_type": metadata.get("element_type"),
        "location": metadata.get("location"),
    }
    return {
        key: value
        for key, value in fields.items()
        if value not in (None, "", [], {})
    }


def _relationship_evidence_card(
    relationship: dict,
    *,
    index: int,
    role: str,
    path: dict | None = None,
) -> dict:
    metadata = relationship.get("metadata") if isinstance(relationship.get("metadata"), dict) else {}
    provenance = relationship.get("provenance") or _relationship_provenance(metadata)
    evidence = _compact_graph_evidence(relationship.get("evidence"))
    relationship_type = relationship.get("relationship_type") or (
        "layout" if _is_structural_relation(str(relationship.get("relation") or "")) else "semantic"
    )
    evidence_families = relationship.get("evidence_families") or _graph_edge_evidence_families(relationship)
    location = _relationship_location_metadata(metadata)
    query_terms = [
        str(term)
        for term in relationship.get("matched_query_terms", []) or []
        if term
    ]
    if not query_terms and path:
        query_terms = [
            str(term)
            for term in path.get("matched_query_terms", []) or []
            if term
        ]
    location_bits = [
        str(location.get(key))
        for key in (
            "filename",
            "slide_title",
            "slide",
            "page",
            "sheet",
            "table",
            "shape",
            "connector",
            "section",
            "markdown_code_language",
            "markdown_code_block_index",
            "markdown_start_line",
            "markdown_end_line",
            "notes_part",
            "comment_id",
            "comment_author",
            "comment_anchor",
            "revision_type",
            "annotation_type",
            "emphasis_term",
            "style_summary",
            "media_part",
            "media_kind",
            "email_subject",
            "email_from",
            "email_to",
            "email_cc",
            "email_date",
            "email_message_id",
            "email_part_kind",
            "email_attachment_filename",
            "diagram_part",
            "diagram_node_id",
            "diagram_source",
            "diagram_target",
            "diagram_connection_type",
            "attachment",
            "archive_entry",
            "legacy_office_stream_path",
            "legacy_office_format",
        )
        if location.get(key) not in (None, "")
    ]
    citation_label = provenance or " | ".join(location_bits)
    if not citation_label:
        citation_label = _format_graph_relationship(relationship, path_style=True)

    card = {
        "id": f"{role}_{index}",
        "index": index,
        "role": role,
        "source": relationship.get("source"),
        "relation": relationship.get("relation"),
        "target": relationship.get("target"),
        "direction": relationship.get("direction"),
        "relationship_type": relationship_type,
        "weight": relationship.get("weight"),
        "score": relationship.get("score"),
        "path_score": path.get("score") if isinstance(path, dict) else None,
        "path_type": path.get("path_type") if isinstance(path, dict) else None,
        "evidence_families": evidence_families,
        "rationale": relationship.get("rationale") or _graph_edge_rationale(relationship),
        "provenance_summary": relationship.get("provenance_summary") or _graph_edge_provenance_summary(relationship),
        "citation_label": citation_label,
        "provenance": provenance,
        "evidence": evidence,
        "location": location,
        "matched_query_terms": query_terms[:8],
        "text": _format_graph_relationship_for_answer(
            relationship,
            include_provenance=False,
        ),
    }
    return {
        key: value
        for key, value in card.items()
        if value not in (None, "", [], {})
    }


def _handoff_anchor_enriched_metadata(anchor: dict, handoff: dict) -> dict:
    metadata = anchor.get("metadata") if isinstance(anchor.get("metadata"), dict) else {}
    filename = handoff.get("filename")
    run_id = handoff.get("run_id")
    return {
        **metadata,
        "filename": metadata.get("filename") or filename,
        "local_graph_run_id": metadata.get("local_graph_run_id") or run_id,
        "local_graph_filename": metadata.get("local_graph_filename") or filename,
        "local_graph_domain": metadata.get("local_graph_domain") or handoff.get("domain"),
    }


def _handoff_anchor_context_line(anchor: dict, handoff: dict) -> str:
    kind = str(anchor.get("kind") or "anchor")
    title = anchor.get("title") or anchor.get("label") or anchor.get("id") or "anchor"
    source = anchor.get("source")
    relation = anchor.get("relation")
    target = anchor.get("target")
    evidence = anchor.get("evidence")
    metadata = _handoff_anchor_enriched_metadata(anchor, handoff)
    provenance = _relationship_provenance(metadata)
    line = f"    - {kind}: {title}"
    if source and (relation or target):
        line += f" | edge: {source} --[{relation or 'related_to'}]--> {target or 'target'}"
    if provenance:
        line += f" | provenance: {provenance}"
    if evidence:
        line += f" | evidence: {evidence}"
    return line


def _handoff_anchor_evidence_card(anchor: dict, handoff: dict, *, index: int) -> dict:
    nodes = [str(node) for node in anchor.get("nodes", []) or [] if node]
    kind = str(anchor.get("kind") or "anchor")
    source = anchor.get("source") or (nodes[0] if nodes else anchor.get("title") or anchor.get("label"))
    target = anchor.get("target") or (nodes[-1] if len(nodes) > 1 else anchor.get("label") or "pipeline handoff")
    relation = anchor.get("relation") or (
        "handoff_path" if kind == "path" else "bridge_entity" if kind == "bridge" else "handoff_anchor"
    )
    metadata = _handoff_anchor_enriched_metadata(anchor, handoff)
    filename = handoff.get("filename")
    run_id = handoff.get("run_id")
    location = _relationship_location_metadata(metadata)
    title = anchor.get("title") or anchor.get("label") or _format_graph_relationship({
        "source": source,
        "relation": relation,
        "target": target,
    })
    provenance = _relationship_provenance(metadata)
    citation_bits = [
        f"pipeline handoff {filename}" if filename else "pipeline handoff",
        provenance,
        str(title) if title else "",
    ]
    if run_id:
        citation_bits.append(f"run {run_id}")
    evidence = _compact_graph_evidence(anchor.get("evidence") or title)
    return {
        "id": f"handoff_anchor_{index}",
        "index": index,
        "role": "interaction_handoff_anchor",
        "kind": kind,
        "source": source,
        "relation": relation,
        "target": target,
        "relationship_type": "layout" if kind in {"path", "bridge"} else "semantic",
        "score": anchor.get("score"),
        "citation_label": " | ".join(bit for bit in citation_bits if bit),
        "provenance": provenance or (f"pipeline handoff {filename}" if filename else "pipeline handoff"),
        "evidence": evidence,
        "location": location,
        "nodes": nodes,
        "text": str(title or ""),
    }


def _handoff_evidence_cards(handoffs: list[dict], *, limit: int = 6) -> list[dict]:
    cards: list[dict] = []
    for handoff in handoffs:
        if not isinstance(handoff, dict):
            continue
        for anchor in handoff.get("graph_anchors", []) or []:
            if not isinstance(anchor, dict):
                continue
            cards.append(_handoff_anchor_evidence_card(anchor, handoff, index=len(cards) + 1))
            if len(cards) >= limit:
                return cards
    return cards


def _pipeline_evidence_cards(evidence: dict | None, *, limit: int = 6) -> list[dict]:
    if not isinstance(evidence, dict) or not evidence:
        return []

    top_families = [
        family
        for family in evidence.get("top_families", []) or []
        if isinstance(family, dict)
    ]
    counts = evidence.get("counts") if isinstance(evidence.get("counts"), dict) else {}
    if not top_families and counts:
        top_families = [
            {
                "family": key,
                "label": _pipeline_evidence_label(key),
                "count": count,
            }
            for key, count in sorted(counts.items(), key=lambda item: (-_positive_int(item[1]), item[0]))[:limit]
        ]

    guidance = [str(item) for item in evidence.get("guidance", []) or [] if item]
    cards: list[dict] = []
    for family in top_families[:limit]:
        family_key = str(family.get("family") or "").strip()
        label = str(family.get("label") or _pipeline_evidence_label(family_key)).strip()
        count = _positive_int(family.get("count"))
        count_label = f"{count} {label}" if count else label
        cards.append({
            "id": f"pipeline_evidence_{family_key or len(cards) + 1}",
            "index": len(cards) + 1,
            "role": "pipeline_evidence_family",
            "kind": family_key or "pipeline_evidence",
            "relationship_type": "evidence_coverage",
            "citation_label": f"pipeline evidence: {count_label}",
            "provenance": "pipeline evidence handoff",
            "evidence": (
                f"{count_label} available from the document processing handoff."
                if count
                else f"{label} available from the document processing handoff."
            ),
            "evidence_families": [family_key] if family_key else [],
            "location": {
                "evidence_family": family_key or label,
                "evidence_count": count,
            },
            "text": count_label,
        })

    overview_counts = evidence.get("graph_overview_counts") if isinstance(evidence.get("graph_overview_counts"), dict) else {}
    overview_bits = [
        f"{_pipeline_evidence_label(key)}={_positive_int(value)}"
        for key, value in sorted(overview_counts.items())
        if _positive_int(value) > 0
    ]
    if overview_bits and len(cards) < limit:
        cards.append({
            "id": "pipeline_evidence_graph_overview",
            "index": len(cards) + 1,
            "role": "pipeline_graph_overview",
            "kind": "graph_overview",
            "relationship_type": "evidence_coverage",
            "citation_label": "pipeline graph overview: " + "; ".join(overview_bits[:4]),
            "provenance": "pipeline evidence handoff",
            "evidence": "Graph overview hints: " + "; ".join(overview_bits[:6]),
            "evidence_families": ["graph_overview"],
            "location": {"evidence_family": "graph_overview"},
            "text": "Graph overview hints",
        })

    if guidance and len(cards) < limit:
        cards.append({
            "id": "pipeline_evidence_guidance",
            "index": len(cards) + 1,
            "role": "pipeline_evidence_guidance",
            "kind": "pipeline_guidance",
            "relationship_type": "evidence_coverage",
            "citation_label": "pipeline evidence guidance",
            "provenance": "pipeline evidence handoff",
            "evidence": " ".join(guidance[:3]),
            "evidence_families": ["pipeline_guidance"],
            "location": {"evidence_family": "pipeline_guidance"},
            "text": "Evidence guidance",
        })

    return cards[:limit]


def _dedupe_evidence_cards(cards: list[dict], *, limit: int = 10) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    for card in cards:
        if not isinstance(card, dict):
            continue
        key = (
            str(card.get("source") or "").lower(),
            str(card.get("relation") or "").lower(),
            str(card.get("target") or "").lower(),
            str(card.get("citation_label") or card.get("provenance") or "").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(card)
        if len(deduped) >= limit:
            break
    return deduped


def _graph_path_node_keys(path: dict) -> set[str]:
    nodes = path.get("display_nodes") or path.get("nodes") or []
    return {
        _normalized_lookup_key(str(node))
        for node in nodes
        if _normalized_lookup_key(str(node))
    }


def _relationship_touches_path(relationship: dict, path_node_keys: set[str]) -> bool:
    if not path_node_keys:
        return False
    source_key = _normalized_lookup_key(str(relationship.get("source") or ""))
    target_key = _normalized_lookup_key(str(relationship.get("target") or ""))
    return source_key in path_node_keys or target_key in path_node_keys


def _path_supporting_relationships(
    relationships: list[dict],
    path: dict,
    *,
    limit: int = 4,
) -> list[dict]:
    path_node_keys = _graph_path_node_keys(path)
    primary_hop_keys = {
        _graph_relationship_key(relationship)
        for relationship in path.get("relationships", []) or []
        if isinstance(relationship, dict)
    }
    supporting: list[dict] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue
        key = _graph_relationship_key(relationship)
        if key in seen or key in primary_hop_keys:
            continue
        if not _relationship_touches_path(relationship, path_node_keys):
            continue
        seen.add(key)
        supporting.append(relationship)
        if len(supporting) >= limit:
            break
    return supporting


def _graph_reasoning_entity_pair_keys(graph_reasoning: dict) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for pair in graph_reasoning.get("entity_pairs", []) or []:
        if not isinstance(pair, dict):
            continue
        source_key = _normalized_lookup_key(str(pair.get("source") or ""))
        target_key = _normalized_lookup_key(str(pair.get("target") or ""))
        if source_key and target_key:
            keys.append((source_key, target_key))
    return keys


def _relationship_matches_entity_pair(
    relationship: dict,
    entity_pair_keys: list[tuple[str, str]],
) -> bool:
    if not entity_pair_keys:
        return False
    source_key = _normalized_lookup_key(str(relationship.get("source") or ""))
    target_key = _normalized_lookup_key(str(relationship.get("target") or ""))
    return any(
        (source_key == pair_source and target_key == pair_target)
        or (source_key == pair_target and target_key == pair_source)
        for pair_source, pair_target in entity_pair_keys
    )


def _relationship_is_direct_answer_fact(relationship: dict) -> bool:
    relation_key = _graph_context_relation_key(relationship)
    structural = _is_structural_relation(relation_key)
    generic = relation_key in GENERIC_GRAPH_CONTEXT_RELATIONS
    style = relation_key in STYLE_GRAPH_CONTEXT_RELATIONS
    relationship_type = str(relationship.get("relationship_type") or "").lower()
    semantic = relationship_type == "semantic" or not structural
    return relation_key in DIRECT_GRAPH_CONTEXT_RELATIONS or (
        semantic and not structural and not generic and not style
    )


def _relationship_is_domain_answer_fact(relationship: dict) -> bool:
    if not _relationship_is_direct_answer_fact(relationship):
        return False
    if _relationship_has_document_endpoint(relationship):
        return False
    if _relationship_has_internal_artifact_endpoint(relationship):
        return False
    if _relationship_has_external_url_endpoint(relationship):
        return False
    return _graph_relationship_evidence_quality(relationship) >= 4


def _domain_answer_relationships(
    relationships: list[dict],
    *,
    limit: int = 12,
) -> list[dict]:
    candidates = [
        relationship
        for relationship in relationships or []
        if isinstance(relationship, dict)
        and _relationship_is_domain_answer_fact(relationship)
    ]
    return sorted(candidates, key=_graph_relationship_rank_key, reverse=True)[:limit]


def _entity_pair_direct_relationships(
    relationships: list[dict],
    graph_reasoning: dict,
) -> list[dict]:
    entity_pair_keys = _graph_reasoning_entity_pair_keys(graph_reasoning)
    if not entity_pair_keys:
        return []
    return [
        relationship
        for relationship in relationships
        if isinstance(relationship, dict)
        and _relationship_matches_entity_pair(relationship, entity_pair_keys)
        and _relationship_is_direct_answer_fact(relationship)
    ]


def _prioritize_entity_pair_direct_relationships(
    relationships: list[dict],
    graph_reasoning: dict,
    *,
    limit: int = 12,
) -> list[dict]:
    ranked_relationships = sorted(
        [
            relationship
            for relationship in relationships or []
            if isinstance(relationship, dict)
        ],
        key=_graph_relationship_rank_key,
        reverse=True,
    )
    pair_direct_relationships = _entity_pair_direct_relationships(
        ranked_relationships,
        graph_reasoning,
    )
    ordered: list[dict] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for relationship in [*pair_direct_relationships, *ranked_relationships]:
        key = _graph_relationship_key(relationship)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(relationship)
        if len(ordered) >= limit:
            break
    return ordered


def _card_path_overlap(card: dict, path: dict) -> tuple[int, int, float]:
    path_node_keys = _graph_path_node_keys(path)
    if not path_node_keys:
        return (0, 0, 0.0)
    source_key = _normalized_lookup_key(str(path.get("source") or ""))
    target_key = _normalized_lookup_key(str(path.get("target") or ""))
    endpoint_keys = {key for key in (source_key, target_key) if key}
    card_values = [
        card.get("source"),
        card.get("target"),
        card.get("text"),
        *(card.get("nodes", []) or []),
    ]
    card_node_keys = {
        _normalized_lookup_key(str(value))
        for value in card_values
        if _normalized_lookup_key(str(value))
    }
    overlap = len(path_node_keys & card_node_keys)
    endpoint_overlap = len(endpoint_keys & card_node_keys)
    return (
        1 if endpoint_keys and endpoint_overlap == len(endpoint_keys) else 0,
        overlap,
        float(card.get("score") or 0),
    )


def _path_handoff_cards(cards: list[dict], path: dict, *, limit: int = 3) -> list[dict]:
    ranked = [
        card
        for card in cards
        if isinstance(card, dict) and _card_path_overlap(card, path)[1] > 0
    ]
    ranked.sort(key=lambda card: _card_path_overlap(card, path), reverse=True)
    return ranked[:limit]


def _graph_answer_plan(graph_reasoning: dict) -> dict:
    paths = [path for path in graph_reasoning.get("paths", []) or [] if isinstance(path, dict)]
    direct_relationships = [
        relationship
        for relationship in graph_reasoning.get("direct_relationships", []) or []
        if isinstance(relationship, dict)
    ]
    graph_overview = graph_reasoning.get("graph_overview") if isinstance(graph_reasoning.get("graph_overview"), dict) else {}
    overview_relationships = [
        relationship
        for relationship in graph_overview.get("top_relationships", []) or []
        if isinstance(relationship, dict)
    ]
    overview_paths = [
        path
        for path in graph_overview.get("cross_document_paths", []) or []
        if isinstance(path, dict)
    ]
    bridge_entities = [
        _display_graph_endpoint(bridge.get("name"), bridge=True)
        for bridge in graph_reasoning.get("bridge_entities", []) or []
        if isinstance(bridge, dict) and bridge.get("name")
    ]
    interaction_handoffs = [
        handoff
        for handoff in graph_reasoning.get("interaction_handoffs", []) or []
        if isinstance(handoff, dict)
    ]
    handoff_cards = _handoff_evidence_cards(interaction_handoffs)
    handoff_refs = [
        {
            "run_id": handoff.get("run_id"),
            "filename": handoff.get("filename"),
            "recommended_mode": handoff.get("recommended_mode"),
            "citation_scopes": handoff.get("citation_scopes", []),
            "graph_anchor_count": len(handoff.get("graph_anchors", []) or []),
            "starter_prompt_count": len(handoff.get("starter_prompts", []) or []),
        }
        for handoff in interaction_handoffs[:6]
    ]

    def relationship_step(relationship: dict, index: int) -> dict:
        provenance = relationship.get("provenance") or _relationship_provenance(relationship.get("metadata"))
        evidence = str(relationship.get("evidence") or "")
        evidence_families = relationship.get("evidence_families") or _graph_edge_evidence_families(relationship)
        return {
            "index": index,
            "source": relationship.get("source"),
            "relation": relationship.get("relation"),
            "target": relationship.get("target"),
            "direction": relationship.get("direction"),
            "relationship_type": relationship.get("relationship_type") or ("layout" if _is_structural_relation(str(relationship.get("relation") or "")) else "semantic"),
            "provenance": provenance,
            "provenance_summary": relationship.get("provenance_summary") or _graph_edge_provenance_summary(relationship),
            "evidence_families": evidence_families,
            "rationale": relationship.get("rationale") or _graph_edge_rationale(relationship),
            "evidence": evidence[:280] if evidence else None,
            "text": _format_graph_relationship_for_answer(
                relationship,
                include_provenance=False,
            ),
        }

    provenance_anchors: list[str] = []
    def add_anchor(value: str | None) -> None:
        cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
        if cleaned and cleaned not in provenance_anchors:
            provenance_anchors.append(cleaned)

    pair_direct_relationships = _entity_pair_direct_relationships(
        direct_relationships,
        graph_reasoning,
    )
    reasoning_query = str(graph_reasoning.get("query") or graph_reasoning.get("reasoning_query") or "")
    relationship_focus = _query_prefers_direct_relationships(reasoning_query)
    bridge_focus = bool(_graph_bridge_hint_from_query(reasoning_query))
    domain_direct_relationships = _domain_answer_relationships(
        _merge_graph_relationships(direct_relationships, overview_relationships, limit=24),
        limit=6,
    )
    if pair_direct_relationships:
        seen_relationships: set[tuple[str, str, str, str, str]] = set()
        prioritized_direct_relationships: list[dict] = []
        for relationship in [*pair_direct_relationships, *direct_relationships]:
            key = _graph_relationship_key(relationship)
            if key in seen_relationships:
                continue
            seen_relationships.add(key)
            prioritized_direct_relationships.append(relationship)
        direct_relationships = prioritized_direct_relationships

    if pair_direct_relationships and not bridge_focus:
        relationships = [
            relationship_step(relationship, index)
            for index, relationship in enumerate(pair_direct_relationships[:6], start=1)
        ]
        for relationship in relationships:
            add_anchor(relationship.get("provenance"))
        return {
            "strategy": "cite_direct_relationships",
            "instruction": "Answer from the direct source-target graph relationship first, naming the relationship label and friendly source location before any indirect paths. Do not describe an indirect path as stronger than an exact direct relationship.",
            "primary_path": None,
            "hops": [],
            "bridge_entities": [],
            "interaction_handoffs": handoff_refs,
            "supporting_relationships": relationships,
            "evidence_cards": _dedupe_evidence_cards([
                _relationship_evidence_card(relationship, index=index, role="direct_relationship")
                for index, relationship in enumerate(pair_direct_relationships[:6], start=1)
            ]),
            "provenance_anchors": provenance_anchors[:8],
            "fallback": "If the direct relationship is contradicted, say that before using retrieved passages or indirect paths.",
        }

    if relationship_focus and domain_direct_relationships:
        relationships = [
            relationship_step(relationship, index)
            for index, relationship in enumerate(domain_direct_relationships[:6], start=1)
        ]
        for relationship in relationships:
            add_anchor(relationship.get("provenance"))
        return {
            "strategy": "cite_direct_relationships",
            "instruction": "Answer from the strongest direct domain relationship first, naming the relationship label and friendly source location. Treat URL, file, source-document, and source-evidence paths as source context unless the user explicitly asks for paths or source lineage.",
            "primary_path": None,
            "hops": [],
            "bridge_entities": [],
            "interaction_handoffs": handoff_refs,
            "supporting_relationships": relationships,
            "evidence_cards": _dedupe_evidence_cards([
                _relationship_evidence_card(relationship, index=index, role="direct_relationship")
                for index, relationship in enumerate(domain_direct_relationships[:6], start=1)
            ]),
            "provenance_anchors": provenance_anchors[:8],
            "fallback": "If no direct domain relationship answers the question, then describe graph paths as indirect evidence chains.",
        }

    if paths:
        strongest_path = sorted(
            paths,
            key=lambda path: _graph_path_rank_key(path, reasoning_query),
            reverse=True,
        )[0]
        hops = [
            relationship_step(relationship, index)
            for index, relationship in enumerate(strongest_path.get("relationships", []) or [], start=1)
            if isinstance(relationship, dict)
        ]
        for hop in hops:
            add_anchor(hop.get("provenance"))
        path_supporting_relationships = _path_supporting_relationships(direct_relationships, strongest_path)
        supporting_relationships = [
            relationship_step(relationship, index)
            for index, relationship in enumerate(path_supporting_relationships, start=1)
        ]
        evidence_cards = [
            _relationship_evidence_card(relationship, index=index, role="primary_path_hop", path=strongest_path)
            for index, relationship in enumerate(strongest_path.get("relationships", []) or [], start=1)
            if isinstance(relationship, dict)
        ]
        evidence_cards.extend(
            _relationship_evidence_card(relationship, index=index, role="supporting_relationship")
            for index, relationship in enumerate(path_supporting_relationships, start=1)
            if isinstance(relationship, dict)
        )
        relevant_handoff_cards = _path_handoff_cards(handoff_cards, strongest_path)
        evidence_cards.extend(relevant_handoff_cards)
        for card in relevant_handoff_cards:
            add_anchor(card.get("citation_label"))
        return {
            "strategy": "walk_strongest_path",
            "instruction": "Answer by naming the Primary path exactly, then walking its ordered hops before adding supporting chunks or relationships. If the path has one hop, call it a one-hop graph path/direct edge. If it has multiple hops, enumerate every bridge node and do not collapse it into a direct relationship or substitute a pipeline handoff anchor.",
            "primary_path": {
                "source": strongest_path.get("source"),
                "target": strongest_path.get("target"),
                "nodes": strongest_path.get("nodes", []),
                "display_nodes": strongest_path.get("display_nodes") or _display_graph_path_nodes([
                    str(node)
                    for node in strongest_path.get("nodes", []) or []
                    if node
                ]),
                "hops": strongest_path.get("hops", len(hops)),
                "score": strongest_path.get("score"),
                "path_type": strongest_path.get("path_type"),
                "text": strongest_path.get("text"),
                "display_text": strongest_path.get("display_text") or _format_graph_path_for_answer(strongest_path),
                "evidence_families": strongest_path.get("evidence_families", []),
                "provenance_summary": strongest_path.get("provenance_summary", []),
                "explanation": strongest_path.get("explanation"),
                "answer_hint": strongest_path.get("answer_hint"),
                "contains_source_evidence_bridge": strongest_path.get("contains_source_evidence_bridge", False),
            },
            "hops": hops,
            "bridge_entities": bridge_entities,
            "interaction_handoffs": handoff_refs,
            "supporting_relationships": supporting_relationships,
            "evidence_cards": _dedupe_evidence_cards(evidence_cards),
            "provenance_anchors": provenance_anchors[:8],
            "fallback": "If a hop is weak or contradicted, say so and use retrieved passages only as supporting evidence.",
        }

    if direct_relationships:
        relationships = [
            relationship_step(relationship, index)
            for index, relationship in enumerate(direct_relationships[:6], start=1)
        ]
        for relationship in relationships:
            add_anchor(relationship.get("provenance"))
        for card in handoff_cards:
            add_anchor(card.get("citation_label"))
        return {
            "strategy": "cite_direct_relationships",
            "instruction": "Answer from the direct graph relationships first, naming relationship labels and friendly source locations.",
            "primary_path": None,
            "hops": [],
            "bridge_entities": [],
            "interaction_handoffs": handoff_refs,
            "supporting_relationships": relationships,
            "evidence_cards": _dedupe_evidence_cards([
                _relationship_evidence_card(relationship, index=index, role="direct_relationship")
                for index, relationship in enumerate(direct_relationships[:6], start=1)
            ] + handoff_cards),
            "provenance_anchors": provenance_anchors[:8],
            "fallback": "If direct graph relationships do not answer the question, say that and fall back to retrieved passages.",
        }

    if overview_relationships or overview_paths:
        primary_overview_path = overview_paths[0] if overview_paths else None
        overview_hops = [
            relationship_step(relationship, index)
            for index, relationship in enumerate((primary_overview_path or {}).get("relationships", []) or [], start=1)
            if isinstance(relationship, dict)
        ]
        overview_path_cards = [
            _relationship_evidence_card(relationship, index=index, role="overview_path_hop", path=primary_overview_path)
            for index, relationship in enumerate((primary_overview_path or {}).get("relationships", []) or [], start=1)
            if isinstance(relationship, dict)
        ]
        for hop in overview_hops:
            add_anchor(hop.get("provenance"))
        for relationship in overview_relationships[:8]:
            add_anchor(relationship.get("provenance") or _relationship_provenance(relationship.get("metadata")))
        for path in overview_paths[:4]:
            for filename in path.get("filenames", []) or []:
                add_anchor(str(filename))
        for card in handoff_cards:
            add_anchor(card.get("citation_label"))
        return {
            "strategy": "compare_graph_overview",
            "instruction": "Compare the local pipeline graph overview, then cite top relationships, bridge/shared entities, and file provenance.",
            "primary_path": {
                **primary_overview_path,
                "display_nodes": _display_graph_path_nodes([
                    str(node)
                    for node in primary_overview_path.get("nodes", []) or []
                    if node
                ]),
                "display_text": _format_graph_path_for_answer(primary_overview_path),
            } if isinstance(primary_overview_path, dict) else None,
            "hops": overview_hops,
            "bridge_entities": [
                str(bridge.get("name"))
                for bridge in graph_overview.get("bridge_entities", []) or []
                if isinstance(bridge, dict) and bridge.get("name")
            ][:6],
            "interaction_handoffs": handoff_refs,
            "supporting_relationships": [
                relationship_step(relationship, index)
                for index, relationship in enumerate(overview_relationships[:6], start=1)
            ],
            "evidence_cards": _dedupe_evidence_cards(overview_path_cards + [
                _relationship_evidence_card(relationship, index=index, role="overview_relationship")
                for index, relationship in enumerate(overview_relationships[:6], start=1)
            ] + handoff_cards),
            "provenance_anchors": provenance_anchors[:8],
            "fallback": "If the overview is broad, separate confirmed graph relationships from tentative cross-document paths.",
        }

    if interaction_handoffs:
        for card in handoff_cards:
            add_anchor(card.get("citation_label"))
        return {
            "strategy": "use_pipeline_interaction_handoff",
            "instruction": "Use the pipeline interaction handoff as the graph answer plan, citing graph anchors and trace provenance before retrieved passages.",
            "primary_path": None,
            "hops": [],
            "bridge_entities": [],
            "interaction_handoffs": handoff_refs,
            "supporting_relationships": [],
            "evidence_cards": _dedupe_evidence_cards(handoff_cards),
            "provenance_anchors": provenance_anchors[:8],
            "fallback": "If a handoff anchor is not enough, say what is missing and use retrieved chunks as support.",
        }

    if graph_reasoning.get("reasoning_needed"):
        return {
            "strategy": "state_no_graph_path",
            "instruction": "State that no connecting graph path was found, then use retrieved passages or extracted entities as fallback evidence.",
            "primary_path": None,
            "hops": [],
            "bridge_entities": [],
            "interaction_handoffs": [],
            "supporting_relationships": [],
            "evidence_cards": [],
            "provenance_anchors": [],
            "fallback": "Do not invent graph edges.",
        }

    return {
        "strategy": "graph_not_required",
        "instruction": "Use graph evidence only if it directly supports the answer.",
        "primary_path": None,
        "hops": [],
        "bridge_entities": [],
        "interaction_handoffs": [],
        "supporting_relationships": [],
        "evidence_cards": [],
        "provenance_anchors": [],
        "fallback": None,
    }


def _graph_source_basename(value: str | None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return re.split(r"[\\/]", text)[-1] if text else ""


def _friendly_graph_source_location(item: dict | None) -> str:
    if not isinstance(item, dict):
        return ""

    metadata = item.get("location") if isinstance(item.get("location"), dict) else {}
    if not metadata:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}

    raw_values = [
        item.get("provenance"),
        item.get("citation_label"),
        item.get("provenance_summary"),
    ]
    raw_text = "; ".join(str(value) for value in raw_values if value)
    parsed: dict[str, str] = {}

    for part in re.split(r"\s*;\s*", raw_text):
        text = re.sub(r"\s+", " ", str(part or "")).strip()
        if not text:
            continue
        if text.lower().startswith("local pipeline "):
            parsed.setdefault("filename", text[len("local pipeline "):].strip())
            continue
        key_value = re.match(r"^([A-Za-z_ -]+)=\s*(.+)$", text)
        if key_value:
            parsed.setdefault(key_value.group(1).strip().lower().replace(" ", "_"), key_value.group(2).strip())
            continue
        slide = re.match(r"^(slide\s+\d+(?::\s*.+)?)$", text, flags=re.IGNORECASE)
        if slide:
            parsed.setdefault("slide_text", slide.group(1))
            continue
        page = re.match(r"^(page\s+\d+)$", text, flags=re.IGNORECASE)
        if page:
            parsed.setdefault("page_text", page.group(1))
            continue
        row = re.match(r"^(row\s+\d+)$", text, flags=re.IGNORECASE)
        if row:
            parsed.setdefault("row_text", row.group(1))

    def first_value(*keys: str) -> str:
        for key in keys:
            value = metadata.get(key)
            if value not in (None, "", [], {}):
                return re.sub(r"\s+", " ", str(value)).strip()
            parsed_value = parsed.get(key)
            if parsed_value:
                return re.sub(r"\s+", " ", str(parsed_value)).strip()
        return ""

    bits: list[str] = []
    filename = first_value("filename", "local_graph_filename", "document_id") or parsed.get("file", "")
    if filename:
        bits.append(_graph_source_basename(filename))

    slide_title = first_value("slide_title")
    slide = first_value("slide")
    slide_text = first_value("slide_text")
    if slide_text:
        bits.append(slide_text)
    elif slide and slide_title:
        bits.append(f"slide {slide}: {slide_title}")
    elif slide_title:
        bits.append(slide_title)
    elif slide:
        bits.append(f"slide {slide}")

    page = first_value("page")
    page_text = first_value("page_text")
    if page_text:
        bits.append(page_text)
    elif page:
        bits.append(f"page {page}")

    row = first_value("row")
    row_text = first_value("row_text")
    if row_text:
        bits.append(row_text)
    elif row:
        bits.append(f"row {row}")

    for label, keys in (
        ("sheet", ("sheet_name", "sheet")),
        ("table", ("table_name", "table")),
        ("chart", ("chart_title", "chart_part", "chart")),
        ("section", ("section_label", "section")),
        ("code block", ("markdown_code_language",)),
        ("link", ("uri",)),
    ):
        value = first_value(*keys)
        if value:
            bits.append(f"{label} {value}")

    seen = set()
    friendly: list[str] = []
    for bit in bits:
        cleaned = re.sub(r"\s+", " ", str(bit or "")).strip()
        if not cleaned:
            continue
        lower = cleaned.lower()
        if (
            lower.startswith("layout")
            or " x=" in lower
            or " y=" in lower
            or lower.startswith("run ")
            or lower.startswith("sha")
        ):
            continue
        if lower not in seen:
            seen.add(lower)
            friendly.append(cleaned)

    if friendly:
        return ", ".join(friendly[:4])

    for value in raw_values:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if text.lower().startswith("pipeline evidence:"):
            return text

    return ""


def _sanitize_llm_context_text(text: str) -> str:
    sanitized_lines: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if re.match(r"^(?:Key refs|Source refs):", stripped, flags=re.IGNORECASE):
            continue

        cleaned = line.replace("local pipeline ", "")
        cleaned = cleaned.replace(" | provenance: ", " | source: ")
        cleaned = re.sub(r"\bprovenance=", "source=", cleaned)
        cleaned = re.sub(r";\s*why=.*$", "", cleaned)
        cleaned = re.sub(
            r"\s*[|;]?\s*(?:layout(?:_bbox)?(?:=|\s+[a-z0-9_-]+\s+)?)?x=-?\d+(?:\.\d+)?in\s+y=-?\d+(?:\.\d+)?in\s+w=-?\d+(?:\.\d+)?in\s+h=-?\d+(?:\.\d+)?in",
            "",
            cleaned,
        )
        cleaned = re.sub(r"\s*[|;]\s*layout_bbox=x=[^|;\]\n]+", "", cleaned)
        cleaned = re.sub(r"\s*[|;]\s*layout_zone=[^|;\]\n]+", "", cleaned)
        cleaned = re.sub(r"\s*[|;]\s*layout_[a-z_]+=[^|;\]\n]+", "", cleaned)
        cleaned = re.sub(r"\s*[|;]\s*bbox=\[[^\]]+\]", "", cleaned)
        cleaned = re.sub(r"\s*[|;]\s*layout(?:\s+[a-z0-9_-]+)?\s+x=[^|;\.\]\n]+", "", cleaned)
        cleaned = re.sub(r"\s+Cite (page)=?(\d+);?\s*layout[^.]*\.", r" Cite \1 \2.", cleaned)
        cleaned = re.sub(r";\s*reasons:\s*[^\]]+", "", cleaned)
        cleaned = re.sub(r"\[\s*chunk match score:\s*([^;\]]+)\s*\]", r"[chunk match score: \1]", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).rstrip()
        if cleaned.strip():
            sanitized_lines.append(cleaned)
    return "\n".join(sanitized_lines)


def _graph_answer_plan_context(plan: dict) -> str:
    if not isinstance(plan, dict) or not plan.get("strategy"):
        return ""
    lines = [
        f"Strategy: {plan.get('strategy')}",
        f"Instruction: {plan.get('instruction')}",
        "Answer focus: name direct graph relationships when present; otherwise describe graph paths as indirect evidence chains. Treat source document/source evidence nodes as provenance, not domain concepts.",
        "Citation style: cite friendly source locations such as filename plus page, slide, sheet, table, or chart. Do not include raw local pipeline labels, run IDs, XML parts, hashes, or layout coordinates unless the user asks for diagnostics.",
    ]
    pipeline_evidence_lines = _pipeline_evidence_context_lines(
        plan.get("pipeline_evidence_coverage") if isinstance(plan.get("pipeline_evidence_coverage"), dict) else {}
    )
    if pipeline_evidence_lines:
        lines.append("Pipeline evidence coverage:")
        lines.extend(f"  {line}" if not line.startswith("  ") else line for line in pipeline_evidence_lines)
    primary_path = plan.get("primary_path") if isinstance(plan.get("primary_path"), dict) else {}
    if primary_path:
        primary_path_text = str(
            primary_path.get("display_text")
            or primary_path.get("text")
            or " -> ".join(str(node) for node in primary_path.get("display_nodes") or primary_path.get("nodes", []) if node)
        )
        primary_path_nodes = [
            str(node)
            for node in primary_path.get("display_nodes") or primary_path.get("nodes", []) or []
            if node
        ]
        lines.append(
            "Primary path: "
            + primary_path_text
            + f" (score: {primary_path.get('score', 0)}; type: {primary_path.get('path_type', 'unknown')})"
        )
        if primary_path_nodes:
            lines.append("Primary path nodes: " + " -> ".join(primary_path_nodes))
            lines.append(
                "Path requirement: start with these nodes in this order; call a one-hop path a direct edge, and never call a multi-hop path direct."
            )
        if primary_path.get("contains_source_evidence_bridge"):
            lines.append("Path note: source evidence is a citation bridge, not a claimed domain entity.")
        if primary_path.get("explanation"):
            lines.append(f"Path explanation: {primary_path.get('explanation')}")
        if primary_path.get("answer_hint"):
            lines.append(f"Answer hint: {primary_path.get('answer_hint')}")
        if primary_path.get("evidence_families"):
            lines.append("Evidence families: " + ", ".join(str(family) for family in primary_path.get("evidence_families", [])[:8]))
    hops = [hop for hop in plan.get("hops", []) or [] if isinstance(hop, dict)]
    if hops:
        lines.append("Ordered hops:")
        lines.extend(
            f"  {hop.get('index')}. {hop.get('text') or _format_graph_relationship_for_answer(hop, path_style=True, include_provenance=False)}"
            + (f" | source: {_friendly_graph_source_location(hop)}" if _friendly_graph_source_location(hop) else "")
            for hop in hops[:6]
        )
    bridge_entities = [str(bridge) for bridge in plan.get("bridge_entities", []) or [] if bridge]
    if bridge_entities:
        lines.append("Bridge entities: " + ", ".join(bridge_entities[:6]))
    interaction_handoffs = [
        handoff
        for handoff in plan.get("interaction_handoffs", []) or []
        if isinstance(handoff, dict)
    ]
    if interaction_handoffs:
        lines.append("Pipeline handoffs:")
        lines.extend(
            "  - "
            + str(handoff.get("filename") or handoff.get("run_id") or "pipeline run")
            + f" | mode: {handoff.get('recommended_mode', 'handoff')}"
            + f" | anchors: {handoff.get('graph_anchor_count', 0)}"
            + (
                " | scopes: " + ", ".join(str(scope) for scope in handoff.get("citation_scopes", [])[:5])
                if handoff.get("citation_scopes")
                else ""
            )
            for handoff in interaction_handoffs[:6]
        )
    supporting_relationships = [
        relationship
        for relationship in plan.get("supporting_relationships", []) or []
        if isinstance(relationship, dict)
    ]
    if supporting_relationships:
        lines.append("Supporting relationships:")
        lines.extend(
            f"  - {relationship.get('text') or _format_graph_relationship_for_answer(relationship, include_provenance=False)}"
            + (f" | source: {_friendly_graph_source_location(relationship)}" if _friendly_graph_source_location(relationship) else "")
            for relationship in supporting_relationships[:6]
        )
    evidence_cards = [
        card
        for card in plan.get("evidence_cards", []) or []
        if isinstance(card, dict)
    ]
    if evidence_cards:
        lines.append("Evidence checklist:")
        for card in evidence_cards[:8]:
            role = str(card.get("role") or "evidence").replace("_", " ")
            role_label = "Hop" if card.get("role") == "primary_path_hop" else role.title()
            citation = _friendly_graph_source_location(card)
            evidence = card.get("evidence")
            if card.get("source") and card.get("relation") and card.get("target"):
                card_text = str(card.get("text") or _format_graph_relationship_for_answer(card, include_provenance=False))
            else:
                card_text = str(card.get("text") or card.get("kind") or "pipeline handoff anchor")
            line = f"  - {role_label} {card.get('index')}: {card_text}"
            if citation:
                line += f" | source: {citation}"
            if evidence:
                line += f" | evidence: {evidence}"
            lines.append(line)
    provenance_anchors = [str(anchor) for anchor in plan.get("provenance_anchors", []) or [] if anchor]
    if provenance_anchors:
        friendly_anchors = []
        seen_anchors = set()
        for anchor in provenance_anchors:
            friendly = _friendly_graph_source_location({"provenance": anchor})
            key = friendly.lower()
            if friendly and key not in seen_anchors:
                seen_anchors.add(key)
                friendly_anchors.append(friendly)
        if friendly_anchors:
            lines.append("Source locations: " + "; ".join(friendly_anchors[:6]))
    if plan.get("fallback"):
        lines.append(f"Fallback: {plan.get('fallback')}")
    return "=== Graph Answer Plan ===\n" + "\n".join(lines)


def _graph_interaction_handoff_context(handoffs: list[dict], *, limit: int = 4) -> str:
    if not isinstance(handoffs, list):
        return ""
    valid_handoffs = [handoff for handoff in handoffs if isinstance(handoff, dict)]
    if not valid_handoffs:
        return ""

    lines = []
    for handoff in valid_handoffs[:limit]:
        filename = handoff.get("filename") or handoff.get("run_id") or "pipeline run"
        lines.append(
            f"Run: {filename} | mode: {handoff.get('recommended_mode', 'handoff')} | ready: {bool(handoff.get('ready'))}"
        )
        if handoff.get("answer_guidance"):
            lines.append(f"  Guidance: {handoff.get('answer_guidance')}")
        citation_scopes = [str(scope) for scope in handoff.get("citation_scopes", []) or [] if scope]
        if citation_scopes:
            lines.append("  Citation scopes: " + ", ".join(citation_scopes[:8]))
        lanes = [
            lane for lane in handoff.get("lanes", []) or []
            if isinstance(lane, dict)
        ]
        if lanes:
            lines.append(
                "  Lanes: "
                + "; ".join(
                    f"{lane.get('label') or lane.get('id')}={lane.get('count', 0)} ({'ready' if lane.get('ready') else 'waiting'})"
                    for lane in lanes[:6]
                )
            )
        cards = [
            card for card in handoff.get("capability_cards", []) or []
            if isinstance(card, dict)
        ]
        if cards:
            lines.append(
                "  Capabilities: "
                + "; ".join(
                    f"{card.get('label')}: {card.get('count', 0)}"
                    for card in cards[:6]
                )
            )
        anchors = [
            anchor for anchor in handoff.get("graph_anchors", []) or []
            if isinstance(anchor, dict)
        ]
        if anchors:
            lines.append("  Graph anchors:")
            for anchor in anchors[:6]:
                lines.append(_handoff_anchor_context_line(anchor, handoff))
        prompts = [
            prompt for prompt in handoff.get("starter_prompts", []) or []
            if isinstance(prompt, dict)
        ]
        if prompts:
            lines.append(
                "  Starter prompts: "
                + "; ".join(
                    str(prompt.get("label") or prompt.get("id") or "prompt")
                    for prompt in prompts[:4]
                )
            )
        rehearsals = [
            rehearsal for rehearsal in handoff.get("retrieval_rehearsals", []) or []
            if isinstance(rehearsal, dict)
        ]
        if rehearsals:
            lines.append("  Retrieval rehearsals:")
            for rehearsal in rehearsals[:4]:
                label = rehearsal.get("label") or rehearsal.get("prompt_id") or rehearsal.get("id") or "prompt"
                status = rehearsal.get("status") or "unknown"
                matched_terms = [
                    str(term) for term in rehearsal.get("matched_terms", []) or []
                    if term
                ]
                lines.append(
                    "    - "
                    + f"{label} | {status} | anchors: {rehearsal.get('matched_anchor_count', 0)}"
                    + f" | chunks: {rehearsal.get('matched_chunk_count', 0)}"
                    + (f" | matched: {', '.join(matched_terms[:8])}" if matched_terms else "")
                )
                top_anchors = [
                    anchor for anchor in rehearsal.get("top_anchors", []) or []
                    if isinstance(anchor, dict)
                ]
                for anchor in top_anchors[:2]:
                    title = anchor.get("title") or anchor.get("source") or anchor.get("id") or "anchor"
                    relation = anchor.get("relation")
                    target = anchor.get("target")
                    metadata = _handoff_anchor_enriched_metadata(anchor, handoff)
                    provenance = _relationship_provenance(metadata)
                    detail = f"{title}"
                    if relation or target:
                        detail += f" ({relation or 'related_to'} -> {target or 'target'})"
                    if provenance:
                        detail += f" | {provenance}"
                    lines.append(f"      anchor: {detail}")
                top_chunks = [
                    chunk for chunk in rehearsal.get("top_chunks", []) or []
                    if isinstance(chunk, dict)
                ]
                for chunk in top_chunks[:2]:
                    chunk_label = chunk.get("chunk_id") or f"chunk {chunk.get('chunk_index', 0)}"
                    preview = _compact_graph_evidence(chunk.get("evidence_preview"), limit=180)
                    lines.append(
                        f"      chunk: {chunk_label}"
                        + (f" | {preview}" if preview else "")
                    )

    return "=== Pipeline Interaction Handoff ===\n" + "\n".join(lines)


def _finalize_graph_reasoning_result(result: dict) -> dict:
    if isinstance(result.get("direct_relationships"), list):
        result["direct_relationships"] = _prioritize_entity_pair_direct_relationships(
            result.get("direct_relationships", []) or [],
            result,
            limit=12,
        )
    result["summary"] = _graph_reasoning_summary(result)
    answer_plan = _graph_answer_plan(result)
    pipeline_evidence = result.get("pipeline_evidence") if isinstance(result.get("pipeline_evidence"), dict) else {}
    if pipeline_evidence:
        answer_plan["pipeline_evidence_coverage"] = pipeline_evidence
        answer_plan["pipeline_evidence_guidance"] = pipeline_evidence.get("guidance", [])
        existing_cards = answer_plan.get("evidence_cards") if isinstance(answer_plan.get("evidence_cards"), list) else []
        pipeline_cards = _pipeline_evidence_cards(pipeline_evidence)
        if pipeline_cards and not existing_cards:
            answer_plan["evidence_cards"] = _dedupe_evidence_cards(pipeline_cards)
            answer_plan["provenance_anchors"] = [
                str(card.get("citation_label"))
                for card in answer_plan["evidence_cards"][:8]
                if card.get("citation_label")
            ]
        elif pipeline_cards:
            answer_plan["evidence_cards"] = _dedupe_evidence_cards([*existing_cards, *pipeline_cards])
    result["answer_plan"] = answer_plan
    return result


def _chunk_merge_key(chunk: dict) -> str:
    if chunk.get("id"):
        return f"id:{chunk['id']}"
    if chunk.get("document_id") and chunk.get("chunk_index") is not None:
        return f"doc:{chunk['document_id']}:{chunk['chunk_index']}"
    content = re.sub(r"\s+", " ", chunk.get("content", "")).strip().lower()
    return f"text:{content[:1000]}"


def _merge_unique_list_values(*values: list | None) -> list:
    merged = []
    seen = set()
    for value_list in values:
        for value in value_list or []:
            key = str(value).lower()
            if key not in seen:
                seen.add(key)
                merged.append(value)
    return merged


def _merge_rag_chunks(primary_chunks: list[dict], supplemental_chunks: list[dict]) -> list[dict]:
    merged_by_key: dict[str, dict] = {}

    for chunk in [*primary_chunks, *supplemental_chunks]:
        key = _chunk_merge_key(chunk)
        if key not in merged_by_key:
            merged_by_key[key] = dict(chunk)
            continue

        existing = merged_by_key[key]
        existing["retrieval_reasons"] = _merge_unique_list_values(
            existing.get("retrieval_reasons"),
            chunk.get("retrieval_reasons"),
        )
        existing["retrieval_queries"] = _merge_unique_list_values(
            existing.get("retrieval_queries"),
            chunk.get("retrieval_queries"),
        )
        existing["graph_context"] = _merge_unique_list_values(
            existing.get("graph_context") if isinstance(existing.get("graph_context"), list) else [],
            chunk.get("graph_context") if isinstance(chunk.get("graph_context"), list) else [],
        )
        existing["graph_paths"] = _merge_unique_list_values(
            existing.get("graph_paths") if isinstance(existing.get("graph_paths"), list) else [],
            chunk.get("graph_paths") if isinstance(chunk.get("graph_paths"), list) else [],
        )
        if chunk.get("score", 0) > existing.get("score", 0):
            preserved_graph_context = existing.get("graph_context", [])
            preserved_graph_paths = existing.get("graph_paths", [])
            preserved_reasons = existing.get("retrieval_reasons", [])
            preserved_queries = existing.get("retrieval_queries", [])
            existing.update({k: v for k, v in chunk.items() if k not in {"retrieval_reasons", "retrieval_queries"}})
            existing["retrieval_reasons"] = preserved_reasons
            existing["retrieval_queries"] = preserved_queries
            existing["graph_context"] = preserved_graph_context
            existing["graph_paths"] = preserved_graph_paths

    return list(merged_by_key.values())


def _bounded_retrieval_top_k(value, *, default: int = 5, minimum: int = 3, maximum: int = 8) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = default
    return max(minimum, min(count, maximum))


def _string_list(value) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    seen = set()
    result = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


PIPELINE_EVIDENCE_LABELS = {
    "archive_entries": "archive entries",
    "archive_relationships": "archive edges",
    "attachments": "attachments",
    "attachment_relationships": "attachment edges",
    "chart_relationships": "chart edges",
    "charts": "chart facts",
    "code_block_relationships": "code block edges",
    "code_blocks": "code blocks",
    "comments": "comments",
    "connectors": "connectors",
    "cross_document_paths": "cross-document paths",
    "diagram_relationships": "diagram edges",
    "diagrams": "diagram items",
    "distinct_tables": "tables",
    "forms": "form fields",
    "relationships": "relationships",
    "graph_relationships": "graph edges",
    "bridge_entities": "bridge entities",
    "shared_entities": "shared entities",
    "hidden_sheets": "hidden sheets",
    "hidden_slides": "hidden slides",
    "images": "images",
    "layout_elements": "layout regions",
    "layout_relationships": "layout edges",
    "legacy_office_streams": "legacy Office streams",
    "legacy_office_text_blocks": "legacy Office text blocks",
    "link_relationships": "link edges",
    "links": "links",
    "media": "media items",
    "message_body_parts": "message body parts",
    "ocr_blocks": "OCR blocks",
    "ocr_relationships": "OCR edges",
    "pivot_tables": "pivot tables",
    "revisions": "revisions",
    "semantic_relationships": "semantic edges",
    "shapes": "shapes",
    "speaker_notes": "speaker notes",
    "speaker_notes_relationships": "speaker note edges",
    "structural_relationships": "structural edges",
    "styled_text_blocks": "styled text blocks",
    "styled_text_relationships": "style edges",
    "table_elements": "table elements",
    "table_facts": "table facts",
}

PIPELINE_EVIDENCE_RETRIEVAL_ALIASES = {
    "archive_entries": ("archive", "zip entry", "nested file", "packaged document"),
    "archive_relationships": ("archive edge", "nested file", "archive relationship"),
    "attachments": ("attachment", "embedded file", "attached document"),
    "attachment_relationships": ("attachment edge", "embedded object", "attached document"),
    "chart_relationships": ("chart edge", "chart value", "series", "axis", "legend", "trendline"),
    "charts": ("chart", "series", "axis", "legend", "data label", "trendline", "metric"),
    "code_block_relationships": ("code block edge", "code block", "language", "implementation", "snippet"),
    "code_blocks": ("code block", "code", "language", "implementation", "snippet", "function"),
    "comments": ("comment", "annotation", "reviewer", "markup"),
    "connectors": ("connector", "arrow", "flow", "relationship line", "dependency"),
    "diagram_relationships": ("diagram edge", "diagram relationship", "flow", "connector", "node"),
    "diagrams": ("diagram", "diagram node", "flow", "process", "dependency"),
    "distinct_tables": ("table", "row", "column", "cell", "grid"),
    "forms": ("form field", "checkbox", "selected", "input", "option"),
    "hidden_sheets": ("hidden sheet", "sheet visibility", "workbook"),
    "hidden_slides": ("hidden slide", "slide visibility", "presentation"),
    "images": ("image", "picture", "screenshot", "figure"),
    "layout_elements": ("layout", "region", "bounding box", "position", "zone"),
    "layout_relationships": ("layout edge", "contains", "position", "region", "zone"),
    "legacy_office_streams": ("legacy office", "ole stream", "compound file"),
    "legacy_office_text_blocks": ("legacy office text", "extracted stream text"),
    "link_relationships": ("link edge", "hyperlink", "uri", "external reference"),
    "links": ("link", "hyperlink", "uri", "url", "external reference"),
    "media": ("media", "audio", "video", "embedded media"),
    "message_body_parts": ("message body", "email body", "plain text", "html body"),
    "ocr_blocks": ("ocr", "scanned text", "image text", "recognized text"),
    "ocr_relationships": ("ocr edge", "image text", "recognized text"),
    "pivot_tables": ("pivot table", "pivot", "aggregation", "lineage"),
    "revisions": ("revision", "tracked change", "inserted", "deleted", "author"),
    "semantic_relationships": ("semantic edge", "relationship", "meaning"),
    "shapes": ("shape", "text box", "presentation shape"),
    "speaker_notes": ("speaker notes", "notes slide", "presenter notes"),
    "speaker_notes_relationships": ("speaker note edge", "presenter notes", "notes slide"),
    "structural_relationships": ("structural edge", "contains", "hierarchy"),
    "styled_text_blocks": ("styled text", "emphasis", "bold", "italic", "underline", "highlight"),
    "styled_text_relationships": ("style edge", "emphasis", "bold", "underline", "highlight"),
    "table_elements": ("table element", "table row", "table cell", "header"),
    "table_facts": ("table fact", "table row", "table cell", "metric"),
}

CHUNK_EVIDENCE_FAMILY_ALIASES = {
    "archive_entry": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["archive_entries"],
    "attachment": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["attachments"],
    "chart": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["charts"],
    "code_block": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["code_blocks"],
    "comment": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["comments"],
    "conditional_format": ("conditional format", "spreadsheet rule", "threshold", "highlight"),
    "connector": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["connectors"],
    "diagram": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["diagrams"],
    "form_field": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["forms"],
    "formula": ("formula", "spreadsheet formula", "cell reference", "calculation"),
    "image": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["images"],
    "layout": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["layout_elements"],
    "link": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["links"],
    "media": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["media"],
    "ocr": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["ocr_blocks"],
    "pivot_table": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["pivot_tables"],
    "presentation_structure": ("presentation", "slide", "slide visibility", "powerpoint"),
    "revision": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["revisions"],
    "speaker_notes": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["speaker_notes"],
    "styled_text": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["styled_text_blocks"],
    "table": PIPELINE_EVIDENCE_RETRIEVAL_ALIASES["table_elements"],
    "workbook_structure": ("workbook", "sheet", "sheet visibility", "spreadsheet"),
}

REQUESTED_EVIDENCE_FAMILY_DEFINITIONS = {
    "charts": {
        "label": "Charts",
        "aliases": ("chart", "charts", "plot", "plots", "series", "axis", "legend", "trendline"),
        "chunk_families": {"chart"},
        "relations": {
            "contains_chart",
            "contains_chart_series",
            "contains_chart_point",
            "has_axis",
            "has_legend",
            "has_data_labels",
            "has_trendline",
            "has_chart_value",
            "shows_category_value",
            "uses_chart_source",
            "q2_value",
            "q3_value",
        },
        "metadata_keys": (
            "chart_title",
            "chart_label",
            "series_name",
            "chart_point_label",
            "category",
            "value",
            "chart_axis_title",
            "slide_title",
        ),
    },
    "tables": {
        "label": "Tables",
        "aliases": ("table", "tables", "row", "rows", "column", "columns", "cell", "cells"),
        "chunk_families": {"table", "pivot_table"},
        "relations": {"contains_table", "contains_pivot_table", "table_fact", "decision", "relation"},
        "metadata_keys": (
            "table_name",
            "table",
            "row",
            "column",
            "table_cell_reference",
            "table_cell_row_header",
            "table_cell_column_header",
            "table_cell_context",
            "slide_title",
        ),
    },
    "form_fields": {
        "label": "Form Fields",
        "aliases": ("form field", "form fields", "form", "forms", "field", "fields", "checkbox", "input", "selected option"),
        "chunk_families": {"form_field"},
        "relations": {"contains_form_field", "has_field", "has_value"},
        "metadata_keys": (
            "label",
            "field_type",
            "form_field_checked",
            "value",
            "form_kind",
            "form_field_tag",
            "form_field_alias",
            "slide_title",
        ),
    },
    "styled_text": {
        "label": "Styled Text",
        "aliases": ("styled text", "style", "styled", "emphasis", "emphasized", "bold", "italic", "underline", "highlight"),
        "chunk_families": {"styled_text"},
        "relations": {"emphasizes", "uses_bullet_structure"},
        "metadata_keys": (
            "style_summary",
            "emphasis_terms",
            "font_size_points",
            "font_family",
            "font_color",
            "highlight_color",
            "slide_title",
        ),
    },
    "connectors": {
        "label": "Connectors",
        "aliases": ("connector", "connectors", "arrow", "arrows", "flow", "flows", "relationship line"),
        "chunk_families": {"connector"},
        "relations": {"contains_connector", "connects_to"},
        "metadata_keys": (
            "connector_title",
            "connector_name",
            "connector_description",
            "source_label",
            "target_label",
        ),
    },
    "comments": {
        "label": "Comments",
        "aliases": ("comment", "comments", "annotation", "annotations", "reviewer"),
        "chunk_families": {"comment"},
        "relations": {"contains_comment", "contains_annotation", "has_comment", "commented_on"},
        "metadata_keys": (
            "comment_author",
            "comment_text",
            "comment_anchor",
            "comment_anchor_text",
            "slide_title",
        ),
    },
    "speaker_notes": {
        "label": "Speaker Notes",
        "aliases": ("speaker notes", "speaker note", "presenter notes", "presenter note", "notes slide"),
        "chunk_families": {"speaker_notes"},
        "relations": {"contains_speaker_notes", "has_speaker_notes"},
        "metadata_keys": (
            "speaker_notes_text",
            "speaker_note_text",
            "note_text",
            "note_anchor_text",
            "notes_part",
            "slide_title",
        ),
    },
    "images": {
        "label": "Images",
        "aliases": ("image", "images", "picture", "pictures", "figure", "figures", "screenshot", "screenshots"),
        "chunk_families": {"image", "ocr"},
        "relations": {"contains_image", "contains_diagram_node", "contains_diagram_relationship"},
        "metadata_keys": ("image_alt_text", "image_description", "image_part", "ocr_text", "slide_title"),
    },
    "media": {
        "label": "Media",
        "aliases": ("media", "audio", "video", "recording", "embedded media"),
        "chunk_families": {"media", "image"},
        "relations": {"contains_media", "contains_image"},
        "metadata_keys": (
            "media_kind",
            "media_mime",
            "media_part",
            "image_part",
            "image_mime",
            "image_alt_text",
            "image_description",
            "slide_title",
        ),
    },
    "attachments": {
        "label": "Attachments",
        "aliases": ("attachment", "attachments", "embedded file", "embedded files", "embedded object", "embedded objects"),
        "chunk_families": {"attachment", "archive_entry"},
        "relations": {"contains_embedded_object", "has_attachment", "has_archive_entry"},
        "metadata_keys": (
            "embedded_file_filename",
            "embedded_file_name",
            "embedded_file_mime",
            "archive_entry_filename",
            "slide_title",
        ),
    },
    "links": {
        "label": "Links",
        "aliases": ("link", "links", "hyperlink", "hyperlinks", "uri", "uris", "url", "urls", "external reference", "external references"),
        "chunk_families": {"link"},
        "relations": {"contains_link", "links_to"},
        "metadata_keys": (
            "uri",
            "link_uri",
            "link_kind",
            "link_target",
            "source_label",
            "target_label",
            "page",
            "slide_title",
        ),
    },
    "layout": {
        "label": "Layout Relationships",
        "aliases": (
            "layout",
            "layout relationship",
            "layout relationships",
            "visual layout",
            "visual relationship",
            "visual relationships",
            "nearby",
            "near",
            "position",
            "positions",
            "spatial",
            "visually near",
        ),
        "chunk_families": {"layout"},
        "relations": {"visually_near"},
        "metadata_keys": (
            "layout_relation",
            "related_label",
            "related_element_type",
            "visual_distance_pct",
            "layout_zone",
            "target_layout_zone",
            "layout_bbox",
            "target_layout_bbox",
            "slide_title",
        ),
    },
}

REQUESTED_EVIDENCE_PRIORITY_METADATA_KEYS = {
    "charts": {"chart_title", "chart_label", "series_name", "chart_point_label", "value"},
    "tables": {
        "table_name",
        "table",
        "table_cell_reference",
        "table_cell_row_header",
        "table_cell_column_header",
        "table_cell_context",
    },
    "form_fields": {"label", "field_type", "form_field_checked", "value"},
    "styled_text": {"style_summary", "emphasis_terms"},
    "connectors": {"connector_title", "connector_name", "connector_description", "source_label", "target_label"},
    "comments": {"comment_author", "comment_text", "comment_anchor_text"},
    "speaker_notes": {"speaker_notes_text", "speaker_note_text", "note_text"},
    "images": {"image_alt_text", "image_description", "image_part", "ocr_text"},
    "media": {
        "media_kind",
        "media_mime",
        "media_part",
        "image_part",
        "image_mime",
        "image_alt_text",
        "image_description",
    },
    "attachments": {"embedded_file_filename", "embedded_file_name", "embedded_file_mime", "archive_entry_filename"},
    "links": {"uri", "link_uri", "link_kind", "link_target", "source_label", "target_label"},
    "layout": {"layout_relation", "related_label", "visual_distance_pct", "layout_zone", "target_layout_zone"},
}

EVIDENCE_FAMILY_BROAD_QUERY_TERMS = {
    "all",
    "available",
    "each",
    "every",
    "extracted",
    "family",
    "families",
    "kinds",
    "types",
}

GENERIC_CHUNK_LOCATION_TOKENS = {"document", "element", "file", "page", "section", "sheet", "slide", "source", "visibility"}
GENERIC_CHUNK_EVIDENCE_FAMILIES = {"text"}

PIPELINE_EVIDENCE_GUIDANCE = (
    (
        {"code_blocks", "code_block_relationships"},
        "Use code block evidence for implementation details, language choices, and reproducible snippets.",
    ),
    (
        {"diagrams", "diagram_relationships", "connectors", "layout_relationships"},
        "Use diagram, connector, and layout evidence for flows, dependencies, and path-style explanations.",
    ),
    (
        {"charts", "chart_relationships", "table_facts", "distinct_tables", "table_elements", "pivot_tables"},
        "Use chart, table, and pivot evidence for numeric claims, comparisons, and lineage.",
    ),
    (
        {"links", "link_relationships", "attachments", "attachment_relationships", "archive_entries", "archive_relationships"},
        "Use links, attachments, and archives to trace referenced sources and nested documents.",
    ),
    (
        {"forms"},
        "Use form field evidence for required inputs, selected options, and workflow state.",
    ),
    (
        {"comments", "revisions", "styled_text_blocks", "styled_text_relationships", "speaker_notes", "speaker_notes_relationships"},
        "Use comments, revisions, styled text, and speaker notes to explain reviewer intent, emphasis, risks, and decisions.",
    ),
    (
        {"images", "ocr_blocks", "ocr_relationships", "media"},
        "Use image, OCR, and media evidence when text was embedded in screenshots, scans, or rich media.",
    ),
    (
        {"hidden_slides", "hidden_sheets", "legacy_office_streams", "legacy_office_text_blocks"},
        "Mention hidden or legacy Office provenance when it affects confidence or source traceability.",
    ),
)


def _positive_int(value) -> int:
    if isinstance(value, bool):
        return 0
    try:
        count = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(count, 0)


def _pipeline_evidence_label(key: str) -> str:
    normalized = str(key or "").strip()
    return PIPELINE_EVIDENCE_LABELS.get(normalized, normalized.replace("_", " "))


def _pipeline_evidence_count_map(value) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, raw_count in value.items():
        normalized = str(key or "").strip()
        count = _positive_int(raw_count)
        if normalized and count > 0:
            counts[normalized] = count
    return counts


def _pipeline_evidence_hints_from_request(data: dict) -> dict:
    counts = _pipeline_evidence_count_map(data.get("pipeline_evidence_totals"))
    families = _string_list(data.get("pipeline_evidence_families"))
    overview_counts = _pipeline_evidence_count_map(data.get("graph_overview_counts"))
    family_order: list[str] = []
    for key, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if key not in family_order:
            family_order.append(key)
    for family in families:
        if family not in family_order:
            family_order.append(family)

    if not counts and not family_order and not overview_counts:
        return {}

    guidance = []
    family_set = set(family_order) | set(counts)
    for keys, text in PIPELINE_EVIDENCE_GUIDANCE:
        if keys & family_set:
            guidance.append(text)

    return {
        "counts": counts,
        "families": family_order,
        "top_families": [
            {
                "family": family,
                "label": _pipeline_evidence_label(family),
                "count": counts.get(family, 0),
            }
            for family in family_order[:12]
        ],
        "graph_overview_counts": overview_counts,
        "guidance": guidance[:6],
    }


def _pipeline_evidence_context_lines(evidence: dict | None) -> list[str]:
    if not isinstance(evidence, dict) or not evidence:
        return []
    top_families = [
        family
        for family in evidence.get("top_families", []) or []
        if isinstance(family, dict)
    ]
    counts = evidence.get("counts") if isinstance(evidence.get("counts"), dict) else {}
    if not top_families and counts:
        top_families = [
            {
                "family": key,
                "label": _pipeline_evidence_label(key),
                "count": count,
            }
            for key, count in sorted(counts.items(), key=lambda item: (-_positive_int(item[1]), item[0]))[:12]
        ]
    lines: list[str] = []
    if top_families:
        rendered = []
        for family in top_families[:10]:
            label = str(family.get("label") or _pipeline_evidence_label(str(family.get("family") or "")))
            count = _positive_int(family.get("count"))
            rendered.append(f"{count} {label}" if count else label)
        lines.append("Available evidence: " + "; ".join(rendered))

    overview_counts = evidence.get("graph_overview_counts") if isinstance(evidence.get("graph_overview_counts"), dict) else {}
    if overview_counts:
        lines.append(
            "Graph overview hints: "
            + "; ".join(
                f"{_pipeline_evidence_label(key)}={_positive_int(value)}"
                for key, value in sorted(overview_counts.items())
                if _positive_int(value) > 0
            )
        )

    guidance = [str(item) for item in evidence.get("guidance", []) or [] if item]
    if guidance:
        lines.append("Evidence guidance:")
        lines.extend(f"  - {item}" for item in guidance[:6])
    return lines


def _pipeline_evidence_summary_text(evidence: dict | None, *, limit: int = 6) -> str:
    if not isinstance(evidence, dict) or not evidence:
        return ""
    top_families = [
        family
        for family in evidence.get("top_families", []) or []
        if isinstance(family, dict)
    ][:limit]
    if not top_families:
        return ""
    return "; ".join(
        f"{_positive_int(family.get('count'))} {family.get('label') or _pipeline_evidence_label(str(family.get('family') or ''))}"
        if _positive_int(family.get("count"))
        else str(family.get("label") or _pipeline_evidence_label(str(family.get("family") or "")))
        for family in top_families
    )


def _pipeline_evidence_retrieval_terms(evidence: dict | None, *, limit: int = 32) -> list[str]:
    if not isinstance(evidence, dict) or not evidence:
        return []
    counts = evidence.get("counts") if isinstance(evidence.get("counts"), dict) else {}
    family_order: list[str] = []
    for family in evidence.get("families", []) or []:
        text = str(family or "").strip()
        if text and text not in family_order:
            family_order.append(text)
    for family, _ in sorted(counts.items(), key=lambda item: (-_positive_int(item[1]), item[0])):
        text = str(family or "").strip()
        if text and text not in family_order:
            family_order.append(text)

    values: list[str] = []
    for family in family_order[:12]:
        label = _pipeline_evidence_label(family)
        values.extend([family, label])
        values.extend(PIPELINE_EVIDENCE_RETRIEVAL_ALIASES.get(family, ()))
        count = _positive_int(counts.get(family))
        if count:
            values.append(f"{count} {label}")

    overview_counts = evidence.get("graph_overview_counts") if isinstance(evidence.get("graph_overview_counts"), dict) else {}
    for key, value in sorted(overview_counts.items(), key=lambda item: (-_positive_int(item[1]), item[0]))[:6]:
        if _positive_int(value) <= 0:
            continue
        label = _pipeline_evidence_label(str(key))
        values.extend([str(key), label, f"{_positive_int(value)} {label}"])

    return _unique_reasoning_terms(values, min_length=3)[:limit]


def _chunk_metadata_retrieval_terms(metadata: dict | None, *, limit: int = 32) -> list[str]:
    if not isinstance(metadata, dict):
        return []

    values: list[str] = []
    families = _string_list(metadata.get("evidence_families"))
    primary_family = str(metadata.get("primary_evidence_family") or "").strip()
    if primary_family and primary_family not in families:
        families.insert(0, primary_family)

    for family in families[:12]:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(family or "").lower()).strip("_")
        if not normalized or normalized in GENERIC_CHUNK_EVIDENCE_FAMILIES:
            continue
        label = normalized.replace("_", " ")
        values.extend([family, label])
        values.extend(CHUNK_EVIDENCE_FAMILY_ALIASES.get(normalized, ()))

    values.extend(_string_list(metadata.get("location_hints"))[:12])
    primary_location = str(metadata.get("primary_location_hint") or "").strip()
    if primary_location:
        values.append(primary_location)

    return _unique_reasoning_terms(values, min_length=3)[:limit]


def _normalized_evidence_family(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _requested_evidence_family_keys(query: str, *, available_families: list[str] | None = None) -> list[str]:
    query_text = re.sub(r"\s+", " ", str(query or "").lower()).strip()
    if not query_text:
        return []

    tokens = _query_tokens(query_text)
    requested: list[str] = []
    for family_key, definition in REQUESTED_EVIDENCE_FAMILY_DEFINITIONS.items():
        aliases = [str(alias).lower() for alias in definition.get("aliases", ()) if alias]
        alias_tokens = {alias for alias in aliases if " " not in alias}
        phrase_hit = any(" " in alias and alias in query_text for alias in aliases)
        token_hit = bool(tokens & alias_tokens)
        if phrase_hit or token_hit:
            requested.append(family_key)

    broad_family_query = bool(
        ({"family", "families", "type", "types", "kind", "kinds"} & tokens)
        and (
            {"evidence", "extracted", "available"} & tokens
            or tokens & EVIDENCE_FAMILY_BROAD_QUERY_TERMS
        )
    )
    if (
        not requested
        and available_families
        and broad_family_query
    ):
        family_lookup: dict[str, str] = {}
        for key, definition in REQUESTED_EVIDENCE_FAMILY_DEFINITIONS.items():
            for family in definition.get("chunk_families", set()):
                family_lookup[_normalized_evidence_family(str(family))] = key
        for family in available_families:
            mapped = family_lookup.get(_normalized_evidence_family(family))
            if mapped and mapped not in requested:
                requested.append(mapped)

    return requested


def _chunk_evidence_families(chunk: dict) -> set[str]:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    families = set()
    for family in _string_list(metadata.get("evidence_families")):
        normalized = _normalized_evidence_family(family)
        if normalized:
            families.add(normalized)
    primary_family = _normalized_evidence_family(str(metadata.get("primary_evidence_family") or ""))
    if primary_family:
        families.add(primary_family)
    return families


def _chunk_matches_requested_evidence_family(chunk: dict, family_key: str) -> bool:
    definition = REQUESTED_EVIDENCE_FAMILY_DEFINITIONS.get(family_key, {})
    chunk_families = {_normalized_evidence_family(str(family)) for family in definition.get("chunk_families", set())}
    if _chunk_evidence_families(chunk) & chunk_families:
        return True

    relation_keys = {_normalized_evidence_family(str(relation)) for relation in definition.get("relations", set())}
    for edge in _chunk_graph_context_items(chunk):
        if _graph_context_relation_key(edge) in relation_keys:
            return True
    return False


def _mark_requested_evidence_chunks(chunks: list[dict], requested_families: list[str]) -> None:
    if not chunks or not requested_families:
        return
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        matched_families = [
            family
            for family in requested_families
            if _chunk_matches_requested_evidence_family(chunk, family)
        ]
        if not matched_families:
            continue
        chunk["retrieval_reasons"] = _merge_unique_list_values(
            chunk.get("retrieval_reasons"),
            ["local_pipeline_requested_evidence"],
        )
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        chunk["metadata"] = {
            **metadata,
            "matched_requested_evidence_families": _merge_unique_list_values(
                metadata.get("matched_requested_evidence_families"),
                matched_families,
            ),
        }


def _select_local_pipeline_context_chunks(
    candidates: list[dict],
    *,
    top_k: int,
    requested_families: list[str],
) -> list[dict]:
    if not candidates:
        return []

    ranked = sorted(candidates, key=lambda chunk: chunk.get("score", 0), reverse=True)
    base_limit = _bounded_retrieval_top_k(top_k)
    coverage_limit = max(base_limit, min(10, len(requested_families) + 2))
    selected: list[dict] = []
    selected_keys: set[str] = set()

    def add_selected(chunk: dict) -> None:
        key = _chunk_merge_key(chunk)
        if key in selected_keys:
            return
        selected.append(chunk)
        selected_keys.add(key)

    for family in requested_families:
        family_chunks = [
            chunk
            for chunk in ranked
            if _chunk_matches_requested_evidence_family(chunk, family)
        ]
        if not family_chunks:
            continue
        add_selected(max(
            family_chunks,
            key=lambda chunk: _rank_requested_evidence_chunk(family, chunk),
        ))

    for chunk in ranked[:coverage_limit]:
        add_selected(chunk)

    for family in requested_families:
        if any(_chunk_matches_requested_evidence_family(chunk, family) for chunk in selected):
            continue
        for chunk in ranked:
            if not _chunk_matches_requested_evidence_family(chunk, family):
                continue
            add_selected(chunk)
            break

    _mark_requested_evidence_chunks(selected, requested_families)
    return selected[:max(coverage_limit, min(12, len(selected)))]


def _requested_graph_edge_family_candidates_for_run(
    run: dict,
    *,
    requested_families: list[str],
    run_id: str,
    filename: str,
    run_domain: str,
    updated_at,
    ranking_query: str,
    limit_per_family: int = 3,
) -> list[dict]:
    if not requested_families:
        return []

    relation_to_families: dict[str, list[str]] = {}
    for family in requested_families:
        if family != "layout":
            continue
        definition = REQUESTED_EVIDENCE_FAMILY_DEFINITIONS.get(family, {})
        for relation in definition.get("relations", set()):
            relation_key = _normalized_evidence_family(str(relation))
            if relation_key:
                relation_to_families.setdefault(relation_key, []).append(family)
    if not relation_to_families:
        return []

    graph = run.get("local_graph") if isinstance(run.get("local_graph"), dict) else {}
    graph_edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
    if not graph_edges:
        return []

    candidates: list[dict] = []
    family_counts: dict[str, int] = {}
    for edge_index, edge in enumerate(graph_edges):
        if not isinstance(edge, dict):
            continue
        relation = str(edge.get("relation") or edge.get("label") or edge.get("type") or "")
        matching_families = relation_to_families.get(_normalized_evidence_family(relation), [])
        if not matching_families:
            continue

        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        if not source or not target:
            continue

        metadata = _flatten_edge_metadata(edge.get("metadata"), edge.get("properties"))
        metadata.update({
            "local_graph_run_id": run_id,
            "local_graph_filename": filename,
            "local_graph_domain": run_domain or None,
            "run_id": run_id,
            "filename": filename,
            "domain": run_domain or None,
            "updated_at": updated_at,
        })
        normalized_edge = {
            **edge,
            "source": source,
            "target": target,
            "relation": relation,
            "label": relation,
            "type": relation,
            "metadata": metadata,
        }

        for family in matching_families:
            if family_counts.get(family, 0) >= limit_per_family:
                continue
            summary = _requested_family_edge_summary(family, normalized_edge)
            if not summary:
                continue
            definition = REQUESTED_EVIDENCE_FAMILY_DEFINITIONS.get(family, {})
            label = str(definition.get("label") or family.replace("_", " ").title())
            candidate_metadata = {
                **metadata,
                "primary_evidence_family": family,
                "evidence_families": [family],
                "source_type": "local_pipeline_graph_edge",
                "graph_context_relationship_count": 1,
                "matched_local_graph_context_terms": _chunk_graph_context_query_terms(
                    [normalized_edge],
                    ranking_query,
                )[:12],
            }
            candidates.append({
                "id": f"pipeline:{run_id}:graph-edge:{family}:{edge_index}",
                "content": f"{label}: {summary}",
                "score": 1.0,
                "document_id": run_id,
                "document_title": filename,
                "chunk_index": edge_index,
                "graph_context": [normalized_edge],
                "graph_paths": [],
                "metadata": candidate_metadata,
                "retrieval_reasons": [
                    "local_pipeline",
                    "local_pipeline_graph_context",
                    "local_pipeline_requested_evidence",
                ],
                "retrieval_queries": _merge_unique_list_values([ranking_query], []),
            })
            family_counts[family] = family_counts.get(family, 0) + 1

    return candidates


def _clean_requested_evidence_text(value: str | None, *, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    url_placeholders: dict[str, str] = {}

    def protect_url(match: re.Match) -> str:
        key = f"__GRAPHRAG_URL_{len(url_placeholders)}__"
        url_placeholders[key] = match.group(0)
        return key

    text = re.sub(r"https?://[^\s;,)]+", protect_url, text)
    def readable_office_package_part(match: re.Match) -> str:
        package_part = match.group(0).replace("\\", "/")
        basename = package_part.rstrip("/").split("/")[-1].strip()
        if basename and basename not in {".rels", "_rels"}:
            return basename
        return "source evidence"

    text = re.sub(
        r"--\[([a-z0-9_ -]+)\]-->",
        lambda match: f"--{match.group(1).replace('_', ' ')}-->",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"--\[([a-z0-9_ -]+)\]--",
        lambda match: f"--{match.group(1).replace('_', ' ')}--",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s*\[chunk match score:[^\]]+\]", "", text, flags=re.I)
    text = re.sub(r"\[([a-z0-9_ -]+)\]", lambda match: f"{match.group(1).replace('_', ' ')}:", text)
    text = re.sub(r"\b(?:ppt|xl|word|docProps|_rels)/[A-Za-z0-9_./-]+", readable_office_package_part, text)
    for key, url in url_placeholders.items():
        text = text.replace(key, url)
    text = text.replace("form_field", "form field").replace("styled_text", "styled text")
    return _compact_graph_evidence(text, limit) or ""


def _metadata_value_for_evidence(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value if item not in (None, ""))
    return str(value)


def _metadata_label_for_evidence(key: str) -> str:
    labels = {
        "chart_title": "chart",
        "chart_label": "chart",
        "chart_point_label": "point",
        "chart_axis_title": "axis",
        "table_name": "table",
        "table_cell_reference": "cell",
        "table_cell_row_header": "row header",
        "table_cell_column_header": "column header",
        "table_cell_context": "cell context",
        "field_type": "field type",
        "form_field_checked": "checked",
        "form_kind": "form kind",
        "form_field_tag": "tag",
        "form_field_alias": "alias",
        "style_summary": "style",
        "emphasis_terms": "emphasis",
        "font_size_points": "font size",
        "font_family": "font",
        "font_color": "color",
        "highlight_color": "highlight",
        "comment_author": "author",
        "comment_text": "comment",
        "comment_anchor": "anchor",
        "comment_anchor_text": "anchor",
        "connector_title": "connector",
        "connector_name": "connector",
        "connector_description": "description",
        "source_label": "source",
        "target_label": "target",
        "speaker_notes_text": "speaker notes",
        "speaker_note_text": "speaker note",
        "note_text": "note",
        "note_anchor_text": "note anchor",
        "notes_part": "notes source",
        "slide_title": "slide",
    }
    return labels.get(key, key.replace("_", " "))


def _requested_family_chunk_summary(family_key: str, chunk: dict) -> str:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    definition = REQUESTED_EVIDENCE_FAMILY_DEFINITIONS.get(family_key, {})
    chunk_families = {_normalized_evidence_family(str(family)) for family in definition.get("chunk_families", set())}
    if not (_chunk_evidence_families(chunk) & chunk_families):
        return ""
    parts: list[str] = []
    for key in definition.get("metadata_keys", ())[:10]:
        value = metadata.get(key)
        if value in (None, "", [], {}):
            continue
        rendered = _clean_requested_evidence_text(_metadata_value_for_evidence(value), limit=140)
        if rendered:
            parts.append(f"{_metadata_label_for_evidence(key)}: {rendered}")

    semantic_text = _clean_requested_evidence_text(metadata.get("semantic_text"), limit=300)
    content_text = _clean_requested_evidence_text(chunk.get("content"), limit=300)
    if semantic_text and (family_key in {"comments", "speaker_notes", "styled_text", "media"} or not parts):
        parts.append(f"evidence text: {semantic_text}")
    elif content_text and not parts:
        parts.append(content_text)

    if not parts:
        return ""
    return _compact_graph_evidence("; ".join(parts), 520) or ""


def _requested_family_edge_summary(family_key: str, edge: dict) -> str:
    edge_metadata = edge.get("metadata") if isinstance(edge.get("metadata"), dict) else {}
    edge_provenance = edge.get("provenance") if isinstance(edge.get("provenance"), dict) else {}
    metadata = _flatten_edge_metadata(edge_provenance, edge_metadata)
    if family_key == "links":
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        uri = (
            str(metadata.get("uri") or "").strip()
            or str(metadata.get("link_uri") or "").strip()
            or str(edge.get("evidence") or "").strip()
            or target
        )
        relation = _graph_context_relation_key(edge).replace("_", " ")
        location = _friendly_graph_source_location({
            "location": _relationship_location_metadata(metadata),
        })
        if source and target:
            if relation in {"contains link", "links to"} or (source.startswith("http") and target.startswith("http")):
                summary = f"{source} links to {target}"
            else:
                summary = f"{source} {relation} {target}"
        elif uri:
            summary = f"Link target {uri}"
        else:
            summary = _format_chunk_graph_context_relationship(edge)
        if uri and uri not in summary:
            summary = f"{summary}; uri {uri}"
        if location:
            summary = f"{summary}; {location}"
        return _clean_requested_evidence_text(summary, limit=360)

    if family_key == "form_fields":
        label = (
            metadata.get("label")
            or metadata.get("form_field_alias")
            or metadata.get("form")
            or edge.get("target")
        )
        value = metadata.get("value")
        field_type = metadata.get("field_type")
        checked = metadata.get("form_field_checked")
        location = _friendly_graph_source_location({
            "location": _relationship_location_metadata(metadata),
        })
        parts: list[str] = []
        if label:
            if checked is True:
                parts.append(f"{label}: checked")
            elif checked is False:
                parts.append(f"{label}: not checked")
            elif value not in (None, ""):
                parts.append(f"{label}: {value}")
            else:
                parts.append(str(label))
        else:
            parts.append(_format_chunk_graph_context_relationship(edge))
        if field_type:
            parts.append(f"type {field_type}")
        if location:
            parts.append(location)
        return _clean_requested_evidence_text("; ".join(str(part) for part in parts if part), limit=360)

    if family_key == "connectors":
        source = (
            metadata.get("source_label")
            or metadata.get("connector_source")
            or edge.get("source")
        )
        target = (
            metadata.get("target_label")
            or metadata.get("connector_target")
            or edge.get("target")
        )
        connector = (
            metadata.get("connector_title")
            or metadata.get("connector_name")
            or metadata.get("connector_description")
            or edge.get("relation")
        )
        description = metadata.get("connector_description")
        location = _friendly_graph_source_location({
            "location": _relationship_location_metadata(metadata),
        })
        parts: list[str] = []
        if source and target:
            relation = _clean_requested_evidence_text(_metadata_value_for_evidence(connector), limit=80)
            if relation:
                parts.append(f"{source} -> {target}: {relation}")
            else:
                parts.append(f"{source} -> {target}")
        else:
            parts.append(_format_chunk_graph_context_relationship(edge))
        if (
            description
            and str(description).strip().lower()
            not in " ".join(parts).strip().lower()
        ):
            parts.append(f"description: {description}")
        if location:
            parts.append(location)
        return _clean_requested_evidence_text("; ".join(str(part) for part in parts if part), limit=360)

    if family_key == "layout":
        source = edge.get("source")
        target = metadata.get("related_label") or edge.get("target")
        relation = str(metadata.get("layout_relation") or "").replace("_", " ").strip()
        distance = metadata.get("visual_distance_pct")
        parts: list[str] = []
        if source and target:
            if relation:
                parts.append(f"{source} is visually near {target} ({target} is {relation} {source})")
            else:
                parts.append(f"{source} is visually near {target}")
        else:
            parts.append(_format_chunk_graph_context_relationship(edge))
        if distance not in (None, ""):
            parts.append(f"distance {distance}%")
        if metadata.get("layout_zone") or metadata.get("target_layout_zone"):
            zones = " -> ".join(
                str(value)
                for value in (metadata.get("layout_zone"), metadata.get("target_layout_zone"))
                if value not in (None, "")
            )
            if zones:
                parts.append(f"zones {zones}")
        return _clean_requested_evidence_text("; ".join(str(part) for part in parts if part), limit=360)

    return _clean_requested_evidence_text(
        _format_chunk_graph_context_relationship(edge),
        limit=360,
    )


def _requested_family_edge_summaries(family_key: str, chunk: dict, *, limit: int = 2) -> list[str]:
    definition = REQUESTED_EVIDENCE_FAMILY_DEFINITIONS.get(family_key, {})
    relation_keys = {_normalized_evidence_family(str(relation)) for relation in definition.get("relations", set())}
    summaries: list[str] = []
    for edge in _chunk_graph_context_items(chunk):
        if _graph_context_relation_key(edge) not in relation_keys:
            continue
        relationship = _requested_family_edge_summary(family_key, edge)
        if relationship and relationship not in summaries:
            summaries.append(relationship)
        if len(summaries) >= limit:
            break
    return summaries


def _requested_evidence_coverage_context(query: str, chunks: list[dict]) -> str:
    available_families = [
        family
        for chunk in chunks
        if isinstance(chunk, dict)
        for family in _chunk_evidence_families(chunk)
    ]
    requested_families = _requested_evidence_family_keys(query, available_families=available_families)
    if not requested_families:
        return ""

    lines = [
        "=== Requested Evidence Coverage ===",
        "The user explicitly asked about: "
        + "; ".join(REQUESTED_EVIDENCE_FAMILY_DEFINITIONS[family]["label"] for family in requested_families)
        + ".",
        "Answer each requested family below. If evidence is listed here, do not report that family as missing.",
    ]

    for family in requested_families:
        definition = REQUESTED_EVIDENCE_FAMILY_DEFINITIONS.get(family, {})
        label = str(definition.get("label") or family.replace("_", " ").title())
        family_chunks = [
            chunk
            for chunk in chunks
            if isinstance(chunk, dict) and _chunk_matches_requested_evidence_family(chunk, family)
        ]
        family_chunks.sort(
            key=lambda chunk: _rank_requested_evidence_chunk(family, chunk),
            reverse=True,
        )
        snippets: list[str] = []
        for chunk in family_chunks:
            chunk_summary = _requested_family_chunk_summary(family, chunk)
            edge_summaries = _requested_family_edge_summaries(family, chunk, limit=2)
            if family in {"connectors", "layout", "links", "form_fields"}:
                snippets.extend(edge_summaries)
                if chunk_summary and not edge_summaries:
                    snippets.append(chunk_summary)
            else:
                if chunk_summary:
                    snippets.append(chunk_summary)
                snippets.extend(edge_summaries)
            snippets = _unique_reasoning_terms(snippets, min_length=3)
            if len(snippets) >= 3:
                break
        if snippets:
            lines.append(f"- {label}: " + "; ".join(snippets[:3]))
        else:
            lines.append(f"- {label}: no run-scoped evidence was selected for this family.")

    return "\n".join(lines)


def _requested_evidence_family_card_snippets(family_key: str, chunk: dict) -> list[str]:
    snippets: list[str] = []
    chunk_summary = _requested_family_chunk_summary(family_key, chunk)
    edge_summaries = _requested_family_edge_summaries(family_key, chunk, limit=2)
    if family_key in {"connectors", "layout", "links", "form_fields"} and edge_summaries:
        snippets.extend(edge_summaries)
    elif family_key in {"comments", "speaker_notes", "styled_text"} and edge_summaries:
        snippets.extend(edge_summaries)
        if chunk_summary:
            snippets.append(chunk_summary)
    else:
        if chunk_summary:
            snippets.append(chunk_summary)
        snippets.extend(edge_summaries)
    return _unique_reasoning_terms(snippets, min_length=3)[:3]


def _requested_evidence_card_location(metadata: dict, chunk: dict, family_key: str) -> dict:
    location: dict = {}
    definition = REQUESTED_EVIDENCE_FAMILY_DEFINITIONS.get(family_key, {})
    relation_keys = {_normalized_evidence_family(str(relation)) for relation in definition.get("relations", set())}
    edge_metadata: dict = {}
    for edge in _chunk_graph_context_items(chunk):
        if _graph_context_relation_key(edge) not in relation_keys:
            continue
        candidate_metadata = edge.get("metadata") if isinstance(edge.get("metadata"), dict) else {}
        if not candidate_metadata and isinstance(edge.get("provenance"), dict):
            candidate_metadata = edge.get("provenance") or {}
        if candidate_metadata:
            edge_metadata = candidate_metadata
            location = _relationship_location_metadata(edge_metadata)
            break
    if not location:
        location = _relationship_location_metadata(metadata)
    location.update({
        "evidence_family": family_key,
        "chunk_id": chunk.get("id"),
        "chunk_index": chunk.get("chunk_index"),
    })
    for key in REQUESTED_EVIDENCE_FAMILY_DEFINITIONS.get(family_key, {}).get("metadata_keys", ())[:8]:
        value = edge_metadata.get(key, metadata.get(key))
        if value not in (None, "", [], {}):
            location[key] = value
    return {
        key: value
        for key, value in location.items()
        if value not in (None, "", [], {})
    }


def _requested_family_specific_priority(family_key: str, chunk: dict) -> int:
    if family_key != "media":
        return 0

    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    families = _chunk_evidence_families(chunk)
    media_kind = _normalized_evidence_family(str(metadata.get("media_kind") or ""))
    has_image = bool(
        "image" in families
        or metadata.get("image_part")
        or metadata.get("image_mime")
        or metadata.get("image_alt_text")
        or metadata.get("image_description")
    )
    if has_image:
        return 5
    if media_kind in {"audio", "video", "recording"}:
        return 4
    if metadata.get("media_part") and media_kind != "embedded_package":
        return 3
    if "attachment" in families or media_kind == "embedded_package":
        return -2
    return 0


def _rank_requested_evidence_chunk(family_key: str, chunk: dict) -> tuple[int, int, int, int, int, int, float]:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    definition = REQUESTED_EVIDENCE_FAMILY_DEFINITIONS.get(family_key, {})
    family_set = {
        _normalized_evidence_family(str(family))
        for family in definition.get("chunk_families", set())
    }
    primary_family = _normalized_evidence_family(str(metadata.get("primary_evidence_family") or ""))
    has_primary_family = 1 if primary_family in family_set else 0
    has_chunk_family = 1 if _chunk_evidence_families(chunk) & family_set else 0
    priority_keys = REQUESTED_EVIDENCE_PRIORITY_METADATA_KEYS.get(family_key, set())
    specific_field_score = sum(
        4 if key in priority_keys else 1
        for key in definition.get("metadata_keys", ())
        if metadata.get(key) not in (None, "", [], {})
    )
    edge_count = len(_requested_family_edge_summaries(family_key, chunk, limit=3))
    has_summary = 1 if _requested_evidence_family_card_snippets(family_key, chunk) else 0
    try:
        score = float(chunk.get("score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    family_priority = _requested_family_specific_priority(family_key, chunk)
    return (family_priority, specific_field_score, has_primary_family, has_chunk_family, edge_count, has_summary, score)


def _requested_evidence_coverage_cards(query: str, chunks: list[dict], *, limit: int = 12) -> list[dict]:
    available_families = [
        family
        for chunk in chunks
        if isinstance(chunk, dict)
        for family in _chunk_evidence_families(chunk)
    ]
    requested_families = _requested_evidence_family_keys(
        query,
        available_families=available_families,
    )
    if not requested_families:
        return []

    cards: list[dict] = []
    for family in requested_families:
        definition = REQUESTED_EVIDENCE_FAMILY_DEFINITIONS.get(family, {})
        label = str(definition.get("label") or family.replace("_", " ").title())
        family_chunks = [
            chunk
            for chunk in chunks
            if isinstance(chunk, dict) and _chunk_matches_requested_evidence_family(chunk, family)
        ]
        if not family_chunks:
            continue
        family_chunks.sort(
            key=lambda chunk: _rank_requested_evidence_chunk(family, chunk),
            reverse=True,
        )

        selected_chunk = family_chunks[0]
        snippets = _requested_evidence_family_card_snippets(family, selected_chunk)
        for chunk in family_chunks[1:]:
            if snippets:
                break
            candidate_snippets = _requested_evidence_family_card_snippets(family, chunk)
            if candidate_snippets:
                selected_chunk = chunk
                snippets = candidate_snippets
                break

        if not snippets:
            fallback = _clean_requested_evidence_text(selected_chunk.get("content"), limit=420)
            if fallback:
                snippets = [fallback]
        if family == "media":
            for chunk in family_chunks:
                if "image" not in _chunk_evidence_families(chunk):
                    continue
                candidate_snippets = _requested_evidence_family_card_snippets(family, chunk)
                for snippet in candidate_snippets:
                    if snippet and snippet not in snippets:
                        snippets.append(snippet)
                        break
                break
        if not snippets:
            continue

        metadata = selected_chunk.get("metadata") if isinstance(selected_chunk.get("metadata"), dict) else {}
        evidence = _compact_graph_evidence("; ".join(snippets[:3]), 620) or ""
        citation_detail = _compact_graph_evidence(snippets[0], 180) or evidence
        source_label = (
            metadata.get("local_graph_filename")
            or metadata.get("filename")
            or selected_chunk.get("document_title")
            or selected_chunk.get("document_id")
        )
        location = _requested_evidence_card_location(metadata, selected_chunk, family)
        provenance = _relationship_provenance(metadata) or str(source_label or "")
        location_hint = _friendly_graph_source_location({"location": location})
        citation_bits = [label, citation_detail]
        if location_hint:
            citation_bits.append(location_hint)
        card = {
            "id": f"requested_evidence_{family}",
            "index": len(cards) + 1,
            "role": "requested_evidence_family",
            "kind": family,
            "title": label,
            "relationship_type": "requested_evidence_family",
            "citation_label": " | ".join(bit for bit in citation_bits if bit),
            "provenance": provenance,
            "provenance_summary": location_hint,
            "evidence": evidence,
            "text": f"{label}: {evidence}",
            "reason": "Requested evidence family coverage",
            "evidence_families": _merge_unique_list_values(
                [family],
                metadata.get("evidence_families"),
            ),
            "categories": [family, label.lower()],
            "location_hints": _merge_unique_list_values(
                [location_hint],
                metadata.get("location_hints"),
            ),
            "location": location,
            "metadata": {
                key: value
                for key, value in metadata.items()
                if key in set(definition.get("metadata_keys", ()))
                or key
                in {
                    "primary_evidence_family",
                    "evidence_families",
                    "local_graph_filename",
                    "local_graph_run_id",
                    "filename",
                    "run_id",
                    "slide",
                    "slide_title",
                    "page",
                    "sheet_name",
                    "table_name",
                    "semantic_text",
                }
            },
            "score": selected_chunk.get("score") if isinstance(selected_chunk.get("score"), (int, float)) else 1.0,
        }
        cards.append({
            key: value
            for key, value in card.items()
            if value not in (None, "", [], {})
        })
        if len(cards) >= limit:
            break

    return _dedupe_evidence_cards(cards, limit=limit)


async def _retrieve_rag_chunks_for_query(
    request: Request,
    query: str,
    top_k: int,
    rag_domains: list,
    rag_search_mode: str,
    errors: list,
    *,
    retrieval_reason: str = "query",
) -> list[dict]:
    rag_chunks = []
    try:
        from modules.rag.retrievers import VectorRetriever, RetrievalConfig
        from routes.rag import get_rag_components, create_embedder, get_collection_name

        rag = get_rag_components(request)
        if not (rag.get("document_manager") and rag.get("vector_store")):
            logger.warning("RAG components not initialized, skipping RAG retrieval")
            return rag_chunks

        rag_domain = None
        domain_id = rag_domains[0] if rag_domains else None

        if domain_id:
            rag_domain = await rag["document_manager"].get_domain(domain_id)
        elif rag_search_mode != "global":
            rag_domain = await rag["document_manager"].get_general_domain()
            if rag_domain:
                domain_id = rag_domain.id

        embedding_model = "nomic-embed-text-v1.5"
        if rag_domain and rag_domain.embedding_model:
            embedding_model = rag_domain.embedding_model

        embedder = create_embedder(request, model_name=embedding_model)

        if rag_search_mode == "global":
            all_domains = await rag["document_manager"].list_domains()
            coll_names = [get_collection_name(d) for d in all_domains]
            existence = await asyncio.gather(
                *[rag["vector_store"].collection_exists(cn) for cn in coll_names]
            )
            valid = [(d, cn) for d, cn, ex in zip(all_domains, coll_names, existence) if ex]

            async def _search_domain(d, cn):
                retriever = VectorRetriever(rag["vector_store"], embedder, rag["document_manager"], cn)
                return await retriever.retrieve(query, RetrievalConfig(top_k=top_k, domain_ids=[d.id]))

            domain_results = await asyncio.gather(*[_search_domain(d, cn) for d, cn in valid])
            all_results = sorted(
                [result for batch in domain_results for result in batch],
                key=lambda result: result.score if hasattr(result, "score") else 0,
                reverse=True,
            )[:top_k]
        else:
            collection_name = get_collection_name(rag_domain) if rag_domain else None
            if collection_name and await rag["vector_store"].collection_exists(collection_name):
                retriever = VectorRetriever(
                    rag["vector_store"], embedder, rag["document_manager"], collection_name
                )
                all_results = await retriever.retrieve(
                    query, RetrievalConfig(top_k=top_k, domain_ids=[domain_id] if domain_id else None)
                )
            else:
                all_results = []

        for result in all_results:
            rd = result.to_dict()
            rag_chunks.append({
                "id": rd.get("id") or rd.get("chunk_id", ""),
                "content": rd.get("content") or rd.get("text", ""),
                "score": round(rd.get("score") or rd.get("similarity", 0), 3),
                "document_id": rd.get("document_id", ""),
                "document_title": rd.get("document_title") or rd.get("metadata", {}).get("title", ""),
                "chunk_index": rd.get("chunk_index", 0),
                "metadata": rd.get("metadata", {}),
                "retrieval_reasons": [retrieval_reason],
                "retrieval_queries": [query],
            })
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")
        error_label = "RAG" if retrieval_reason == "query" else f"RAG {retrieval_reason}"
        errors.append(f"{error_label}: {str(e)}")

    return rag_chunks


def _structured_assignment_value(value: str):
    text = str(value or "").strip().strip('"')
    if re.fullmatch(r"-?\d+", text):
        try:
            return int(text)
        except ValueError:
            return text
    if re.fullmatch(r"-?\d+\.\d+", text):
        try:
            return float(text)
        except ValueError:
            return text
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    return text


def _structured_assignment_metadata(value: str) -> dict:
    metadata: dict = {}
    for segment in re.split(r"\s*[|;]\s*", str(value or "")):
        if "=" not in segment:
            continue
        key, raw_value = segment.split("=", 1)
        normalized_key = re.sub(r"[^a-zA-Z0-9_]+", "_", key.strip()).strip("_").lower()
        if not normalized_key:
            continue
        metadata[normalized_key] = _structured_assignment_value(raw_value)
    return metadata


def _structured_artifact_requested_evidence_candidates(
    run: dict,
    *,
    requested_families: list[str],
    run_id: str,
    filename: str,
    run_domain: str,
    updated_at,
    ranking_query: str,
    limit_per_family: int = 4,
) -> list[dict]:
    if not requested_families:
        return []

    structured_artifact = (
        run.get("structured_ingest_artifact")
        if isinstance(run.get("structured_ingest_artifact"), dict)
        else {}
    )
    structured_content = _load_structured_artifact_content(structured_artifact)
    if not structured_content:
        return []

    candidates: list[dict] = []
    family_counts: dict[str, int] = {}
    blocks = re.split(r"(?m)(?=^Scope:\s+)", structured_content)
    for block_index, block in enumerate(blocks):
        block_text = block.strip()
        if not block_text.startswith("Scope:"):
            continue

        evidence_match = re.search(r"(?m)^Evidence types:\s*(.+)$", block_text)
        if not evidence_match:
            continue
        evidence_families = [
            _normalized_evidence_family(family)
            for family in re.split(r"\s*,\s*", evidence_match.group(1).strip())
            if family.strip()
        ]
        metadata: dict = {
            "source_type": "local_pipeline_structured_artifact",
            "run_id": run_id,
            "filename": filename,
            "domain": run_domain or None,
            "updated_at": updated_at,
            "primary_evidence_family": evidence_families[0] if evidence_families else None,
            "evidence_families": evidence_families,
        }
        for label in ("Provenance", "Link refs", "Key refs", "Source refs"):
            label_match = re.search(rf"(?m)^{re.escape(label)}:\s*(.+)$", block_text)
            if label_match:
                metadata.update(_structured_assignment_metadata(label_match.group(1)))

        focus_lines = [
            match.group(1).strip()
            for match in re.finditer(r"(?m)^Evidence focus:\s*(.+)$", block_text)
            if match.group(1).strip()
        ]
        focus_lines.extend(
            line.strip()
            for line in block_text.splitlines()
            if re.match(r"^\[[a-zA-Z0-9_ -]+\]\s+", line.strip())
        )
        semantic_text = _clean_requested_evidence_text(" ".join(focus_lines), limit=700)
        if semantic_text:
            metadata["semantic_text"] = semantic_text

        scope_match = re.search(r"(?m)^Scope:\s*(.+)$", block_text)
        content = _clean_requested_evidence_text(
            " ".join(
                line.strip()
                for line in block_text.splitlines()
                if line.strip()
            ),
            limit=1200,
        )
        matched_families = [
            family
            for family in requested_families
            if _chunk_matches_requested_evidence_family({"metadata": metadata}, family)
        ]
        if not matched_families:
            continue
        if all(family_counts.get(family, 0) >= limit_per_family for family in matched_families):
            continue

        metadata["matched_requested_evidence_families"] = matched_families
        try:
            chunk_index = 10_000 + block_index
        except (TypeError, ValueError):
            chunk_index = len(candidates)
        candidates.append({
            "id": f"pipeline:{run_id}:structured:{block_index}",
            "content": content or semantic_text or block_text[:1200],
            "score": 0.97,
            "document_id": run_id,
            "document_title": filename,
            "chunk_index": chunk_index,
            "graph_context": [],
            "graph_paths": [],
            "metadata": metadata,
            "retrieval_reasons": [
                "local_pipeline",
                "local_pipeline_requested_evidence",
                "local_pipeline_structured_artifact",
            ],
            "retrieval_queries": _merge_unique_list_values([ranking_query], []),
        })
        for family in matched_families:
            family_counts[family] = family_counts.get(family, 0) + 1
        if scope_match and len(candidates) >= max(12, len(requested_families) * limit_per_family):
            break

    return candidates


def _local_pipeline_context_chunks(
    query: str,
    domain: str | None,
    top_k: int,
    graph_reasoning: dict | None = None,
    *,
    pipeline_run_ids: list[str] | None = None,
    user_query: str | None = None,
    max_runs: int = 12,
) -> list[dict]:
    ranking_query = user_query or query
    requested_evidence_families = _requested_evidence_family_keys(ranking_query)
    query_terms = {
        token
        for token in _query_tokens(query)
        if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
    }
    grouped_visual_requested = bool(_token_variants(query_terms) & GROUPED_VISUAL_QUERY_TERMS)
    query_lower = str(query or "").lower()
    graph_terms = _graph_augmented_retrieval_terms(graph_reasoning or {}, limit=16)
    pipeline_evidence = (graph_reasoning or {}).get("pipeline_evidence") if isinstance(graph_reasoning, dict) else {}
    evidence_terms = _pipeline_evidence_retrieval_terms(
        pipeline_evidence if isinstance(pipeline_evidence, dict) else {},
        limit=18,
    )
    graph_tokens = {
        token
        for token in _query_tokens(" ".join(graph_terms))
        if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
    }
    evidence_tokens = {
        token
        for token in _query_tokens(" ".join(evidence_terms))
        if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
    }
    if not query_terms and not graph_tokens and not evidence_tokens:
        return []

    try:
        runs = get_pipeline_run_store().list_runs()
    except Exception as exc:
        logger.debug(f"Local GraphRAG pipeline context fallback unavailable: {exc}")
        return []

    wanted_domain = str(domain or "").lower()
    wanted_run_ids = set(pipeline_run_ids or [])
    candidates: list[dict] = []
    for run_rank, run in enumerate(reversed(runs)):
        if not wanted_run_ids and run_rank >= max_runs:
            break
        if not isinstance(run, dict):
            continue
        run_id = str(run.get("run_id") or "")
        if wanted_run_ids and run_id not in wanted_run_ids:
            continue
        document = run.get("document") if isinstance(run.get("document"), dict) else {}
        local_context = run.get("local_context") if isinstance(run.get("local_context"), dict) else {}
        run_domain = str(local_context.get("domain") or document.get("domain") or "").lower()
        if wanted_domain and wanted_domain != "general" and run_domain and run_domain != wanted_domain:
            continue

        chunks = local_context.get("chunks") if isinstance(local_context.get("chunks"), list) else []
        if not chunks:
            samples = run.get("samples") if isinstance(run.get("samples"), dict) else {}
            chunks = samples.get("chunks") if isinstance(samples.get("chunks"), list) else []
        if not chunks:
            continue

        filename = str(local_context.get("filename") or document.get("filename") or "Pipeline Upload")
        updated_at = run.get("updated_at") or run.get("created_at")
        recent_bonus = max(0.0, 0.08 - (run_rank * 0.01))
        candidates.extend(_requested_graph_edge_family_candidates_for_run(
            run,
            requested_families=requested_evidence_families,
            run_id=run_id,
            filename=filename,
            run_domain=run_domain,
            updated_at=updated_at,
            ranking_query=ranking_query,
        ))
        candidates.extend(_structured_artifact_requested_evidence_candidates(
            run,
            requested_families=requested_evidence_families,
            run_id=run_id,
            filename=filename,
            run_domain=run_domain,
            updated_at=updated_at,
            ranking_query=ranking_query,
        ))

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            content = str(chunk.get("content") or chunk.get("text") or "")
            if not content:
                continue
            metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            matched_requested_families = [
                family
                for family in requested_evidence_families
                if _chunk_matches_requested_evidence_family(chunk, family)
            ]
            graph_context = _rank_chunk_graph_context_for_query(
                _chunk_graph_context_items(chunk),
                ranking_query,
            )
            grouped_visual_context_edges = [
                edge
                for edge in graph_context
                if isinstance(edge, dict)
                and (
                    _graph_context_relation_key(edge) in GROUPED_VISUAL_GRAPH_CONTEXT_RELATIONS
                    or any(
                        (edge.get("metadata") if isinstance(edge.get("metadata"), dict) else {}).get(key)
                        for key in ("group_label", "parent_group_label", "group_title")
                    )
                )
            ]
            has_grouped_visual_context = grouped_visual_requested and bool(grouped_visual_context_edges)
            graph_paths = _chunk_graph_path_items(chunk)
            graph_context_text = _chunk_graph_context_text(graph_context)
            graph_path_text = _chunk_graph_path_text(graph_paths)
            provenance_text = " ".join(str(value) for value in metadata.get("provenance", []) if value) if isinstance(metadata.get("provenance"), list) else str(metadata.get("provenance") or "")
            chunk_evidence_terms = _chunk_metadata_retrieval_terms(metadata, limit=24)
            chunk_evidence_text = " ".join(chunk_evidence_terms)
            searchable = f"{content}\n{provenance_text}\n{graph_context_text}\n{graph_path_text}\n{filename}"
            searchable_lower = searchable.lower()
            text_tokens = {
                token
                for token in _query_tokens(searchable)
                if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
            }
            query_hits = sorted(query_terms & text_tokens)
            graph_token_hits = sorted(graph_tokens & text_tokens)
            evidence_token_hits = sorted(evidence_tokens & text_tokens)
            chunk_evidence_tokens = {
                token
                for token in _query_tokens(chunk_evidence_text)
                if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
            }
            chunk_evidence_token_hits = sorted(query_terms & chunk_evidence_tokens)
            graph_phrase_hits = [
                term for term in graph_terms
                if len(str(term)) >= 3 and str(term).lower() in searchable_lower
            ]
            evidence_phrase_hits = [
                term for term in evidence_terms
                if len(str(term)) >= 3 and str(term).lower() in searchable_lower
            ]
            chunk_evidence_phrase_hits = [
                term for term in chunk_evidence_terms
                if len(str(term)) >= 3 and str(term).lower() in query_lower
            ]
            has_strong_chunk_evidence_match = bool(
                chunk_evidence_phrase_hits
                or len(chunk_evidence_token_hits) >= 2
                or any(term not in GENERIC_CHUNK_LOCATION_TOKENS for term in chunk_evidence_token_hits)
            )
            matched_chunk_evidence_terms = (
                _unique_reasoning_terms([*chunk_evidence_token_hits, *chunk_evidence_phrase_hits], min_length=2)[:12]
                if has_strong_chunk_evidence_match
                else []
            )
            graph_context_query_hits = _chunk_graph_context_query_terms(graph_context, ranking_query)
            graph_path_query_hits = _chunk_graph_path_query_terms(graph_paths, ranking_query)
            if (
                not query_hits
                and not graph_token_hits
                and not graph_phrase_hits
                and not evidence_token_hits
                and not evidence_phrase_hits
                and not has_strong_chunk_evidence_match
                and not graph_context_query_hits
                and not graph_path_query_hits
                and not matched_requested_families
            ):
                continue

            query_score = len(query_hits) / max(1, min(len(query_terms), 8))
            graph_context_score = len(graph_context_query_hits) / max(1, min(len(query_terms), 8)) if graph_context_query_hits else 0
            graph_path_score = len(graph_path_query_hits) / max(1, min(len(query_terms), 8)) if graph_path_query_hits else 0
            evidence_score = min(
                1.0,
                (len(evidence_token_hits) / max(1, min(len(evidence_tokens), 8)))
                + (len(evidence_phrase_hits[:4]) * 0.12),
            )
            chunk_evidence_score = min(
                1.0,
                (len(chunk_evidence_token_hits) / max(1, min(len(query_terms), 8)))
                + (len(chunk_evidence_phrase_hits[:4]) * 0.14),
            ) if has_strong_chunk_evidence_match else 0.0
            graph_score = min(
                1.0,
                (len(graph_token_hits) / max(1, min(len(graph_tokens), 8)))
                + (len(graph_phrase_hits[:4]) * 0.12)
                + (evidence_score * 0.4)
                + (graph_context_score * 0.35)
                + (graph_path_score * 0.42),
            )
            grouped_visual_score = min(
                0.24,
                0.12 + (min(len(grouped_visual_context_edges), 2) * 0.06),
            ) if has_grouped_visual_context else 0.0
            score = round(min(1.0, (query_score * 0.50) + (graph_score * 0.30) + (chunk_evidence_score * 0.16) + grouped_visual_score + recent_bonus), 3)
            if matched_requested_families:
                score = max(score, 0.94)
            retrieval_reasons = ["local_pipeline"]
            if graph_token_hits or graph_phrase_hits:
                retrieval_reasons.append("local_pipeline_graph_augmented")
            if evidence_token_hits or evidence_phrase_hits:
                retrieval_reasons.append("local_pipeline_evidence_context")
            if has_strong_chunk_evidence_match:
                retrieval_reasons.append("local_pipeline_chunk_evidence")
            if graph_context_query_hits:
                retrieval_reasons.append("local_pipeline_graph_context")
            if graph_path_query_hits:
                retrieval_reasons.append("local_pipeline_graph_path_context")
            if has_grouped_visual_context:
                retrieval_reasons.append("local_pipeline_grouped_visual_context")
            if matched_requested_families:
                retrieval_reasons.append("local_pipeline_requested_evidence")
            try:
                chunk_index = int(chunk.get("chunk_index", 0) or 0)
            except (TypeError, ValueError):
                chunk_index = 0

            candidates.append({
                "id": f"pipeline:{run_id}:{chunk.get('id') or chunk.get('chunk_id') or chunk.get('chunk_index', len(candidates))}",
                "content": content,
                "score": score,
                "document_id": run_id,
                "document_title": filename,
                "chunk_index": chunk_index,
                "graph_context": graph_context,
                "graph_paths": graph_paths,
                "metadata": {
                    **metadata,
                    "source_type": "local_pipeline_run",
                    "run_id": run_id,
                    "filename": filename,
                    "domain": run_domain or None,
                    "updated_at": updated_at,
                    "matched_local_query_terms": query_hits[:12],
                    "matched_local_graph_terms": _unique_reasoning_terms([*graph_token_hits, *graph_phrase_hits], min_length=2)[:12],
                    "matched_local_pipeline_evidence_terms": _unique_reasoning_terms([*evidence_token_hits, *evidence_phrase_hits], min_length=2)[:12],
                    "matched_local_chunk_evidence_terms": matched_chunk_evidence_terms,
                    "local_chunk_evidence_terms": chunk_evidence_terms[:16],
                    "matched_local_graph_context_terms": graph_context_query_hits[:12],
                    "matched_local_graph_path_terms": graph_path_query_hits[:12],
                    "graph_context_relationship_count": len(graph_context),
                    "graph_path_context_count": len(graph_paths),
                    "grouped_visual_context_count": len(grouped_visual_context_edges),
                    "matched_requested_evidence_families": _merge_unique_list_values(
                        metadata.get("matched_requested_evidence_families"),
                        matched_requested_families,
                    ),
                },
                "retrieval_reasons": retrieval_reasons,
                "retrieval_queries": _merge_unique_list_values([ranking_query], [query]),
            })

    available_families = [
        family
        for candidate in candidates
        for family in _chunk_evidence_families(candidate)
    ]
    requested_evidence_families = _requested_evidence_family_keys(
        ranking_query,
        available_families=available_families,
    ) or requested_evidence_families
    return _select_local_pipeline_context_chunks(
        candidates,
        top_k=top_k,
        requested_families=requested_evidence_families,
    )


def _value_available(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple, set, str, bytes)):
        return bool(value)
    return True


def _pipeline_artifact_endpoint(run_id: str, kind: str) -> str | None:
    if not run_id:
        return None
    return f"/api/v1/graphrag/ingest/pipeline/{run_id}/artifacts/{kind}"


def _load_structured_artifact_content(structured_artifact: dict, *, max_chars: int = 500_000) -> str:
    content = structured_artifact.get("content")
    if isinstance(content, str) and content:
        return content[:max_chars]

    storage_path = str(structured_artifact.get("content_storage_path") or "").strip()
    if not storage_path:
        return ""
    try:
        with open(storage_path, "r", encoding="utf-8") as handle:
            return handle.read(max_chars)
    except OSError as exc:
        logger.debug(f"Unable to read structured GraphRAG artifact: {exc}")
        return ""


def _structured_artifact_excerpt(content: str, query: str, *, char_limit: int = 900) -> str:
    text = re.sub(r"\n{3,}", "\n\n", str(content or "")).strip()
    if not text:
        return ""

    query_terms = {
        token
        for token in _query_tokens(query)
        if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 3
    }
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    selected: list[str] = []
    if query_terms:
        for line in lines:
            line_tokens = _query_tokens(line)
            if query_terms & line_tokens:
                selected.append(line)
            if len("\n".join(selected)) >= char_limit:
                break
    if not selected:
        selected = lines[:8]

    excerpt = "\n".join(selected).strip()
    if len(excerpt) > char_limit:
        excerpt = excerpt[:char_limit].rsplit(" ", 1)[0].strip() + "..."
    return excerpt


def _source_artifact_records(document_profile: dict, *, limit: int = 5) -> list[dict]:
    evidence_profile = document_profile.get("evidence_profile") if isinstance(document_profile.get("evidence_profile"), dict) else {}
    source_artifacts = evidence_profile.get("source_artifacts") if isinstance(evidence_profile.get("source_artifacts"), list) else []
    records: list[dict] = []
    for artifact in source_artifacts:
        if not isinstance(artifact, dict):
            continue
        records.append({
            "kind": artifact.get("kind"),
            "filename": artifact.get("filename") or artifact.get("path"),
            "mime_type": artifact.get("mime_type") or artifact.get("content_type"),
            "format": artifact.get("format"),
            "path": artifact.get("path"),
            "extraction_method": artifact.get("extraction_method"),
            "checksum": artifact.get("checksum") or artifact.get("sha256"),
            "parsed_child_elements": artifact.get("parsed_child_elements", 0),
            "parent_provenance": artifact.get("parent_provenance"),
        })
        if len(records) >= limit:
            break
    return records


def _local_pipeline_artifact_context(
    query: str,
    domain: str | None,
    *,
    pipeline_run_ids: list[str] | None = None,
    max_runs: int = 12,
) -> dict:
    try:
        runs = get_pipeline_run_store().list_runs()
    except Exception as exc:
        logger.debug(f"Local GraphRAG pipeline artifact context unavailable: {exc}")
        return {"runs": [], "run_count": 0, "artifact_count": 0}

    wanted_domain = str(domain or "").lower()
    wanted_run_ids = set(pipeline_run_ids or [])
    query_terms = {
        token
        for token in _query_tokens(query)
        if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
    }
    records: list[dict] = []

    for run_rank, run in enumerate(reversed(runs)):
        if not wanted_run_ids and run_rank >= max_runs:
            break
        if not isinstance(run, dict):
            continue
        run_id = str(run.get("run_id") or "")
        if wanted_run_ids and run_id not in wanted_run_ids:
            continue

        document = run.get("document") if isinstance(run.get("document"), dict) else {}
        local_graph = run.get("local_graph") if isinstance(run.get("local_graph"), dict) else {}
        local_context = run.get("local_context") if isinstance(run.get("local_context"), dict) else {}
        local_reasoning = run.get("local_graph_reasoning") if isinstance(run.get("local_graph_reasoning"), dict) else {}
        interaction_handoff = run.get("interaction_handoff") if isinstance(run.get("interaction_handoff"), dict) else {}
        document_profile = run.get("document_profile") if isinstance(run.get("document_profile"), dict) else {}
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        samples = run.get("samples") if isinstance(run.get("samples"), dict) else {}
        source_document_artifact = run.get("source_document_artifact") if isinstance(run.get("source_document_artifact"), dict) else {}
        stages = run.get("stages") if isinstance(run.get("stages"), list) else []
        structured_artifact = run.get("structured_ingest_artifact") if isinstance(run.get("structured_ingest_artifact"), dict) else {}
        structured_summary = run.get("structured_ingest") if isinstance(run.get("structured_ingest"), dict) else {}

        run_domain = str(
            local_context.get("domain")
            or local_graph.get("domain")
            or document.get("domain")
            or ""
        ).lower()
        if wanted_domain and wanted_domain != "general" and run_domain and run_domain != wanted_domain:
            continue

        filename = str(
            local_context.get("filename")
            or local_graph.get("filename")
            or document.get("filename")
            or "Pipeline Upload"
        )

        artifacts: list[dict] = []

        def add_artifact(kind: str, label: str, payload, content_type: str = "application/json", filename_value: str | None = None) -> None:
            if not _value_available(payload):
                return
            artifacts.append({
                "kind": kind,
                "label": label,
                "content_type": content_type,
                "filename": filename_value,
                "endpoint": _pipeline_artifact_endpoint(run_id, kind),
            })

        structured_content = _load_structured_artifact_content(structured_artifact)
        add_artifact(
            "source_document",
            "Original source document",
            source_document_artifact,
            str(source_document_artifact.get("content_type") or document.get("content_type") or "application/octet-stream"),
            str(source_document_artifact.get("filename") or document.get("filename") or "") or None,
        )
        add_artifact(
            "structured_ingest",
            "Structured ingest Markdown",
            structured_content,
            "text/markdown",
            str(structured_artifact.get("filename") or structured_summary.get("filename") or "") or None,
        )
        add_artifact("local_graph", "Local graph artifact", local_graph)
        add_artifact("local_context", "Local context cache", local_context)
        add_artifact("local_graph_reasoning", "Local graph reasoning index", local_reasoning)
        add_artifact("interaction_handoff", "Interaction handoff", interaction_handoff)
        add_artifact("document_profile", "Document profile", document_profile)
        add_artifact("summary", "Run summary", summary)
        add_artifact("samples", "Pipeline samples", samples)
        add_artifact("stages", "Pipeline stages", stages)
        if not artifacts:
            continue

        source_artifacts = []
        if _value_available(source_document_artifact):
            source_artifacts.append({
                "kind": "source_document",
                "filename": source_document_artifact.get("filename") or document.get("filename"),
                "mime_type": source_document_artifact.get("content_type") or document.get("content_type"),
                "checksum": source_document_artifact.get("sha256"),
                "size_bytes": source_document_artifact.get("size_bytes") or document.get("size_bytes"),
                "path": _pipeline_artifact_endpoint(run_id, "source_document"),
                "extraction_method": "original_upload",
                "parsed_child_elements": 0,
            })
        source_artifacts.extend(_source_artifact_records(document_profile))
        evidence_cards = [
            card
            for card in local_context.get("evidence_cards", [])
            if isinstance(card, dict)
        ] if isinstance(local_context.get("evidence_cards"), list) else []
        excerpt = _structured_artifact_excerpt(structured_content, query)
        searchable = "\n".join([
            filename,
            structured_content[:3000],
            " ".join(str(node.get("name") or node.get("label") or node.get("id") or "") for node in local_graph.get("nodes", []) if isinstance(node, dict)),
            " ".join(str(edge.get("evidence") or edge.get("label") or edge.get("type") or "") for edge in local_graph.get("edges", []) if isinstance(edge, dict)),
            " ".join(str(artifact.get("filename") or artifact.get("kind") or "") for artifact in source_artifacts),
        ])
        if not wanted_run_ids and query_terms:
            searchable_tokens = _query_tokens(searchable)
            if not (query_terms & searchable_tokens):
                continue

        record = {
            "run_id": run_id,
            "filename": filename,
            "domain": run_domain or None,
            "status": run.get("status"),
            "updated_at": run.get("updated_at") or run.get("created_at"),
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "manifest_endpoint": _pipeline_artifact_endpoint(run_id, "manifest"),
            "structured_ingest": {
                "filename": structured_artifact.get("filename") or structured_summary.get("filename"),
                "char_count": structured_artifact.get("char_count") or structured_summary.get("char_count") or len(structured_content),
                "sha256": structured_artifact.get("sha256") or structured_summary.get("artifact_sha256"),
                "truncated": structured_artifact.get("truncated", structured_summary.get("truncated", False)),
                "excerpt": excerpt,
            } if structured_content else None,
            "local_graph": {
                "node_count": local_graph.get("node_count", len(local_graph.get("nodes", []) if isinstance(local_graph.get("nodes"), list) else [])),
                "edge_count": local_graph.get("edge_count", len(local_graph.get("edges", []) if isinstance(local_graph.get("edges"), list) else [])),
            } if local_graph else None,
            "local_context": {
                "chunk_count": local_context.get("cached_chunk_count") or local_context.get("chunk_count") or len(local_context.get("chunks", []) if isinstance(local_context.get("chunks"), list) else []),
                "graph_context_relationship_count": local_context.get("graph_context_relationship_count", 0),
                "graph_path_context_count": local_context.get("graph_path_context_count", 0),
                "evidence_card_count": len(evidence_cards),
            } if local_context else None,
            "local_graph_reasoning": {
                "bridge_entity_count": local_reasoning.get("bridge_entity_count", len(local_reasoning.get("bridge_entities", []) if isinstance(local_reasoning.get("bridge_entities"), list) else [])),
                "path_candidate_count": local_reasoning.get("path_candidate_count", len(local_reasoning.get("path_candidates", []) if isinstance(local_reasoning.get("path_candidates"), list) else [])),
            } if local_reasoning else None,
            "source_artifacts": source_artifacts,
            "evidence_cards": evidence_cards[:8],
        }
        records.append(record)

    return {
        "runs": records,
        "run_count": len(records),
        "artifact_count": sum(int(record.get("artifact_count", 0) or 0) for record in records),
        "structured_artifact_count": len([record for record in records if record.get("structured_ingest")]),
        "source_artifact_count": sum(len(record.get("source_artifacts", []) or []) for record in records),
    }


def _local_pipeline_artifact_context_text(artifact_context: dict) -> str:
    runs = artifact_context.get("runs") if isinstance(artifact_context.get("runs"), list) else []
    if not runs:
        return ""

    lines = [
        "=== Pipeline Artifact Context ===",
        "Use these run artifacts as the source package for structured text, local graph evidence, retrieval chunks, and graph reasoning.",
    ]
    for run in runs[:4]:
        if not isinstance(run, dict):
            continue
        lines.append(f"Run {run.get('run_id')}: {run.get('filename')} ({run.get('status') or 'unknown'})")
        artifacts = [artifact for artifact in run.get("artifacts", []) or [] if isinstance(artifact, dict)]
        if artifacts:
            lines.append(
                "Artifacts: "
                + "; ".join(
                    f"{artifact.get('label')} -> {artifact.get('endpoint')}"
                    for artifact in artifacts[:8]
                    if artifact.get("label")
                )
            )
        if run.get("manifest_endpoint"):
            lines.append(f"Artifact manifest: {run.get('manifest_endpoint')}")
        structured = run.get("structured_ingest") if isinstance(run.get("structured_ingest"), dict) else {}
        if structured:
            sha = str(structured.get("sha256") or "")
            sha_text = f"; sha256 {sha[:12]}" if sha else ""
            lines.append(
                f"Structured ingest: {structured.get('filename') or 'structured markdown'} "
                f"({structured.get('char_count', 0)} chars{sha_text})"
            )
            if structured.get("excerpt"):
                lines.append("Structured source excerpt:")
                lines.extend(f"  {line}" for line in str(structured["excerpt"]).splitlines()[:10] if line.strip())
        local_graph = run.get("local_graph") if isinstance(run.get("local_graph"), dict) else {}
        if local_graph:
            lines.append(f"Local graph: {local_graph.get('node_count', 0)} nodes, {local_graph.get('edge_count', 0)} edges")
        local_context = run.get("local_context") if isinstance(run.get("local_context"), dict) else {}
        if local_context:
            lines.append(
                f"Local context: {local_context.get('chunk_count', 0)} chunks, "
                f"{local_context.get('graph_context_relationship_count', 0)} graph relationships, "
                f"{local_context.get('graph_path_context_count', 0)} graph paths"
            )
        local_reasoning = run.get("local_graph_reasoning") if isinstance(run.get("local_graph_reasoning"), dict) else {}
        if local_reasoning:
            lines.append(
                f"Graph reasoning artifact: {local_reasoning.get('bridge_entity_count', 0)} bridge entities, "
                f"{local_reasoning.get('path_candidate_count', 0)} path candidates"
            )
        source_artifacts = [artifact for artifact in run.get("source_artifacts", []) or [] if isinstance(artifact, dict)]
        if source_artifacts:
            lines.append("Source artifacts:")
            for artifact in source_artifacts[:5]:
                descriptor = artifact.get("mime_type") or artifact.get("format") or "unknown MIME"
                if artifact.get("extraction_method"):
                    descriptor = f"{descriptor}; {artifact.get('extraction_method')}"
                lines.append(
                    f"  {artifact.get('kind') or 'artifact'} {artifact.get('filename') or ''} "
                    f"({descriptor}; parsed elements: {artifact.get('parsed_child_elements', 0)})".strip()
                )

    return "\n".join(line for line in lines if line)


def _clean_entity_candidate(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip(" .,:;?!\"'")
    value = re.sub(
        r"^(?:the|a|an|how\s+(?:is|are|does)\s+|what\s+(?:is|are)\s+|"
        r"relationship between|connection between|path between|link between)\s+",
        "",
        value,
        flags=re.I,
    )
    return value[:120].strip()


def _strip_graph_bridge_clause(value: str) -> str:
    return re.split(r"\s+(?:through|via)\s+.+$", str(value or ""), maxsplit=1, flags=re.I)[0]


def _strip_graph_entity_context_clause(value: str) -> str:
    text = _strip_graph_bridge_clause(value)
    text = re.split(
        r"\s+(?:in|from|within)\s+(?:this|the|a|an)?\s*"
        r"(?:(?:uploaded|selected|current|active)\s+)?"
        r"(?:deck|document|file|source|upload|presentation|spreadsheet|workbook)\b.*$",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]
    text = re.split(
        r"\s+(?:cite|explain|include|including|show|using|with)\b.*$",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]
    return text


def _query_target_entity_candidates(value: str) -> list[str]:
    cleaned = _clean_entity_candidate(_strip_graph_entity_context_clause(value))
    if not cleaned:
        return []

    pieces = [
        _clean_entity_candidate(piece)
        for piece in re.split(r"\s*(?:,|\band\b|\bor\b)\s*", cleaned, flags=re.I)
    ]
    useful_pieces = [
        piece
        for piece in pieces
        if len(piece) >= 2
        and {
            token
            for token in _query_tokens(piece)
            if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
        }
        - GRAPH_REASONING_GENERIC_NODE_TERMS
    ]
    if len(useful_pieces) >= 2:
        return useful_pieces[:4]
    return [cleaned]


def _graph_bridge_hint_from_query(query: str) -> str:
    query_text = re.sub(r"\s+", " ", str(query or "")).strip()
    if not query_text:
        return ""

    metadata_match = re.search(r"(?:^|[\n;])\s*bridge\s*:\s*([^;\n]+)", query_text, flags=re.I)
    if metadata_match:
        return _clean_entity_candidate(metadata_match.group(1))

    if not re.search(r"\b(?:bridge|bridges|connect|connection|hop|hops|path|paths|trace|traces)\b", query_text, flags=re.I):
        return ""

    for match in re.finditer(
        r"\b(?:through|via)\s+(.+?)(?=(?:[?.;,]|$|\s+(?:and|before|cite|explain|include|including|show|using|with)\b))",
        query_text,
        flags=re.I,
    ):
        candidate = _clean_entity_candidate(match.group(1))
        if len(candidate) < 3:
            continue
        tokens = {token for token in _query_tokens(candidate) if len(token) >= 2}
        if not tokens or tokens <= GRAPH_REASONING_GENERIC_NODE_TERMS:
            continue
        if re.match(r"^(?:this|that|the)\s+(?:deck|document|file|page|slide|source)\b", candidate, flags=re.I):
            continue
        return candidate
    return ""


def _clean_arrow_entity_candidate(value: str, *, side: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" .,:;?!\"'")
    if side == "source":
        text = re.sub(
            r"^.*?\b(?:behind|between|from|about|for|edge|relationship|connection|path|link)\s+",
            "",
            text,
            flags=re.I,
        )
        if ":" in text:
            text = text.rsplit(":", 1)[-1]
    else:
        text = re.split(
            r"\s+(?:in|from|for|within|using)\s+(?:the\s+)?"
            r"(?:document|file|trace|pipeline|run|batch|source)?\s*[\w./:-]+",
            text,
            maxsplit=1,
            flags=re.I,
        )[0]
        text = re.split(
            r"\s+(?:and then|then|while|cite|explain|show|summarize)\b",
            text,
            maxsplit=1,
            flags=re.I,
        )[0]
    return _clean_entity_candidate(text)


def _compact_graph_intent_hints(
    *,
    source: str | None = None,
    target: str | None = None,
    bridge: str | None = None,
    relation: str | None = None,
) -> dict[str, str]:
    hints = {
        "source": _clean_entity_candidate(str(source or "")),
        "target": _clean_entity_candidate(str(target or "")),
        "bridge": _clean_entity_candidate(str(bridge or "")),
        "relation": str(relation or "").replace("_", " ").strip(),
    }
    return {key: value for key, value in hints.items() if value}


def _graph_intent_query(
    query: str,
    *,
    source: str | None = None,
    target: str | None = None,
    bridge: str | None = None,
    relation: str | None = None,
) -> str:
    hints = _compact_graph_intent_hints(
        source=source,
        target=target,
        bridge=bridge,
        relation=relation,
    )
    if not hints:
        return query

    hint_text = "; ".join(f"{label}: {value}" for label, value in hints.items())
    return f"{query}\nGraph intent hints: {hint_text}".strip()


GRAPH_RELATION_QUERY_TRIGGERS = (
    "affect",
    "affects",
    "block",
    "blocks",
    "cause",
    "causes",
    "depend on",
    "depends on",
    "drive",
    "drives",
    "enable",
    "enables",
    "gate",
    "gates",
    "influence",
    "influences",
    "map to",
    "maps to",
    "own",
    "owns",
    "reference",
    "references",
    "support",
    "supports",
    "unblock",
    "unblocks",
    "use",
    "uses",
)


def _entity_pairs_from_query(query: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen = set()

    def add_pair(source: str, target: str) -> None:
        source = _clean_entity_candidate(source)
        target = _clean_entity_candidate(target)
        if len(source) < 2 or len(target) < 2:
            return
        key = (source.lower(), target.lower())
        if key not in seen and source.lower() != target.lower():
            seen.add(key)
            pairs.append((source, target))

    arrow_path_spans: list[tuple[int, int]] = []
    arrow_path_patterns = (
        r'"([^"]{2,160})"\s*(?:[-=]+>|→|➜)\s*"([^"]{2,160})"\s*(?:[-=]+>|→|➜)\s*"([^"]{2,160})"',
        r"'([^']{2,160})'\s*(?:[-=]+>|→|➜)\s*'([^']{2,160})'\s*(?:[-=]+>|→|➜)\s*'([^']{2,160})'",
    )
    for pattern in arrow_path_patterns:
        for match in re.finditer(pattern, query):
            source = _clean_entity_candidate(match.group(1))
            target = _clean_entity_candidate(match.group(3))
            before = len(pairs)
            add_pair(source, target)
            if len(pairs) > before:
                arrow_path_spans.append(match.span())

    arrow_edge_patterns = (
        r'"([^"]{2,160})"\s*(?:[-=]+>|→|➜)\s*"([^"]{2,160})"',
        r"'([^']{2,160})'\s*(?:[-=]+>|→|➜)\s*'([^']{2,160})'",
        r"(?:behind|between|from|about|for|edge|relationship|connection|path|link)\s+([^.\n;:]{2,160}?)\s*(?:[-=]+>|→|➜)\s*([^.\n;]{2,160}?)(?:[?.;,]|$)",
        r"(?:^|[\n([{])\s*([^.\n;:]{2,160}?)\s*(?:[-=]+>|→|➜)\s*([^.\n;]{2,160}?)(?:[?.;,)\]}]|$)",
    )
    for pattern in arrow_edge_patterns:
        for match in re.finditer(pattern, query, flags=re.I):
            start, end = match.span()
            if any(start < path_end and end > path_start for path_start, path_end in arrow_path_spans):
                continue
            source = _clean_arrow_entity_candidate(match.group(1), side="source")
            target = _clean_arrow_entity_candidate(match.group(2), side="target")
            add_pair(source, target)

    patterns = (
        r"(?:relationship|connection|path|link|bridge)s?\s+(?:between|from)\s+(.+?)\s+(?:and|to|with)\s+(.+?)(?:[?.]|$)",
        r"(?:how\s+(?:is|are)\s+)?(.+?)\s+(?:connected|linked|related)\s+(?:to|with)\s+(.+?)(?:[?.]|$)",
        r"(?:what|which)\s+(?:links?|connects?|bridges?)\s+(.+?)\s+(?:and|to|with)\s+(.+?)(?:[?.]|$)",
        r"(?:how\s+does|why\s+does|does)\s+(.+?)\s+(?:influence|influences|affect|affects|cause|causes|drive|drives|enable|enables|support|supports|block|blocks|gate|gates|reference|references|depend\s+on|depends\s+on|connect\s+to|maps?\s+to|uses?)\s+(.+?)(?:[?.]|$)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, query, flags=re.I):
            source = _clean_entity_candidate(match.group(1))
            for target in _query_target_entity_candidates(match.group(2)):
                add_pair(source, target)
    return pairs[:4]


def _entity_names_for_reasoning(query: str, entities: list[dict]) -> list[str]:
    names = []
    seen = set()

    for entity in sorted(entities, key=lambda item: item.get("score", 0), reverse=True):
        name = _clean_entity_candidate(entity.get("name", ""))
        key = name.lower()
        if len(name) >= 2 and key not in seen:
            seen.add(key)
            names.append(name)

    for pair in _entity_pairs_from_query(query):
        for group in pair:
            name = _clean_entity_candidate(group)
            key = name.lower()
            if len(name) >= 2 and key not in seen:
                seen.add(key)
                names.append(name)

    return names[:8]


def _query_phrase_position(query: str, phrase: str) -> int:
    spans = _query_phrase_spans(query, phrase)
    return spans[0][0] if spans else -1


def _query_phrase_spans(query: str, phrase: str) -> list[tuple[int, int]]:
    phrase_text = str(phrase or "").strip().lower()
    if not phrase_text:
        return []
    return [
        match.span()
        for match in re.finditer(
            rf"(?<![a-z0-9]){re.escape(phrase_text)}(?![a-z0-9])",
            str(query or "").lower(),
        )
    ]


def _clean_graph_reasoning_entity_names(query: str, names: list[str], *, limit: int = 8) -> list[str]:
    records: list[dict] = []
    seen: set[str] = set()
    for raw_name in names:
        name = _clean_entity_candidate(_strip_graph_entity_context_clause(str(raw_name or "")))
        key = name.lower()
        if len(name) < 2 or key in seen:
            continue
        seen.add(key)
        records.append({
            "name": name,
            "lookup_key": _normalized_lookup_key(name),
            "spans": _query_phrase_spans(query, name),
            "tokens": {
                token
                for token in _query_tokens(name)
                if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
            },
        })

    cleaned: list[str] = []
    for record in records:
        spans = record["spans"]
        tokens = record["tokens"]
        lookup_key = record["lookup_key"]
        nested_in_longer_mention = False
        if spans and tokens and lookup_key:
            for other in records:
                if other is record:
                    continue
                if len(other["lookup_key"]) <= len(lookup_key):
                    continue
                if not tokens <= other["tokens"]:
                    continue
                other_spans = other["spans"]
                if other_spans and all(
                    any(
                        int(other_start) <= int(start) and int(end) <= int(other_end)
                        for other_start, other_end in other_spans
                    )
                    for start, end in spans
                ):
                    nested_in_longer_mention = True
                    break
        if not nested_in_longer_mention:
            cleaned.append(record["name"])
        if len(cleaned) >= limit:
            break

    return cleaned


def _graph_node_query_candidates(
    query: str,
    node_map: dict,
    raw_edges: list,
    *,
    limit: int = 12,
    normalized_edges: list[dict] | None = None,
) -> list[dict]:
    query_text = str(query or "")
    query_lower = query_text.lower()
    query_tokens = _token_variants({
        token
        for token in _query_tokens(query_text)
        if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
    })
    if not query_tokens or not (node_map or raw_edges):
        return []

    candidates: dict[str, dict] = {}

    def add_candidate(name: str, node: dict | None = None) -> None:
        cleaned = _clean_entity_candidate(name)
        if len(cleaned) < 2:
            return
        canonical_name = _clean_entity_candidate(_node_display_name(node)) if isinstance(node, dict) else cleaned
        if len(canonical_name) < 2:
            canonical_name = cleaned
        key = canonical_name.lower()
        record = candidates.setdefault(key, {
            "name": canonical_name,
            "node": node or {},
            "aliases": set(),
            "edge_count": 0,
            "best_position": len(query_text) + 1,
        })
        record["aliases"].add(cleaned)
        if node and not record.get("node"):
            record["node"] = node

    for node in node_map.values():
        if not isinstance(node, dict):
            continue
        display_name = _node_display_name(node)
        if display_name:
            add_candidate(display_name, node)
        for alias in _node_alias_values(node):
            add_candidate(alias, node)

    graph_edges = normalized_edges if normalized_edges is not None else _normalize_graph_edges(raw_edges, node_map)
    for normalized in graph_edges:
        for endpoint in (normalized.get("source"), normalized.get("target")):
            add_candidate(str(endpoint or ""))
            key = _clean_entity_candidate(str(endpoint or "")).lower()
            if key in candidates:
                candidates[key]["edge_count"] += 1

    scored: list[dict] = []
    for record in candidates.values():
        aliases = [str(alias) for alias in record.get("aliases", set()) if alias]
        alias_scores = []
        positions = []
        match_spans: list[tuple[int, int]] = []
        for alias in aliases:
            alias_lower = alias.lower()
            alias_tokens = {
                token
                for token in _query_tokens(alias)
                if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
            }
            if not alias_tokens:
                continue
            phrase_position = _query_phrase_position(query_lower, alias_lower)
            normalized_alias = _normalized_lookup_key(alias)
            normalized_query = _normalized_lookup_key(query_text)
            exact_phrase = phrase_position >= 0
            normalized_phrase = bool(normalized_alias and normalized_alias in normalized_query)
            overlap = len(query_tokens & _token_variants(alias_tokens))
            coverage = overlap / max(1, min(len(alias_tokens), 4))
            if not exact_phrase and not normalized_phrase and coverage <= 0:
                continue
            if exact_phrase:
                positions.append(phrase_position)
                match_spans.extend(_query_phrase_spans(query_lower, alias_lower))
            alias_scores.append(
                (0.62 if exact_phrase else 0.0)
                + (0.2 if normalized_phrase else 0.0)
                + (coverage * 0.34)
            )
        if not alias_scores:
            continue
        name = str(record["name"])
        node_type = str((record.get("node") or {}).get("type") or "").lower()
        generic_penalty = 0.18 if re.match(r"^(?:slide|page|sheet)\s+\d+\b", name, flags=re.I) else 0.0
        document_penalty = 0.08 if node_type in {"document", "file"} and "." in name else 0.0
        edge_bonus = min(0.12, float(record.get("edge_count", 0) or 0) * 0.015)
        score = round(min(1.0, max(alias_scores) + edge_bonus - generic_penalty - document_penalty), 3)
        if score <= 0:
            continue
        scored.append({
            "name": name,
            "score": score,
            "position": min(positions) if positions else len(query_text) + len(scored),
            "match_spans": sorted(set(match_spans)),
            "edge_count": record.get("edge_count", 0),
            "type": node_type or "concept",
        })

    return sorted(scored, key=lambda item: (item.get("score", 0), -item.get("position", 0)), reverse=True)[:limit]


def _entity_pairs_from_graph_mentions(
    query: str,
    node_map: dict,
    raw_edges: list,
    *,
    limit: int = 4,
    normalized_edges: list[dict] | None = None,
) -> list[tuple[str, str]]:
    def useful_candidate(candidate: dict) -> bool:
        name = _clean_entity_candidate(str(candidate.get("name") or ""))
        if len(name) < 2:
            return False
        normalized_name = name.lower()
        node_type = str(candidate.get("type") or "").lower()
        if normalized_name.startswith(("graphrag ", "rag ")):
            return False
        if GRAPH_REASONING_DOCUMENT_NAME_PATTERN.search(normalized_name):
            return False
        if node_type in {"document", "file"} and "." in normalized_name:
            return False
        tokens = {
            token
            for token in _query_tokens(name)
            if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
        }
        if not tokens:
            return False
        if tokens <= GRAPH_REASONING_GENERIC_NODE_TERMS:
            return False
        return True

    candidates = [
        candidate
        for candidate in _graph_node_query_candidates(
            query,
            node_map,
            raw_edges,
            limit=10,
            normalized_edges=normalized_edges,
        )
        if useful_candidate(candidate)
    ]
    if candidates:
        filtered_candidates: list[dict] = []
        for candidate in candidates:
            spans = candidate.get("match_spans") if isinstance(candidate.get("match_spans"), list) else []
            name = _clean_entity_candidate(str(candidate.get("name") or ""))
            name_key = _normalized_lookup_key(name)
            name_len = len(name_key)
            covered_by_longer = False
            for other in candidates:
                if other is candidate:
                    continue
                other_name = _clean_entity_candidate(str(other.get("name") or ""))
                other_key = _normalized_lookup_key(other_name)
                if len(other_key) <= name_len:
                    continue
                other_spans = other.get("match_spans") if isinstance(other.get("match_spans"), list) else []
                if not spans or not other_spans:
                    continue
                if not (
                    name_key
                    and (
                        f" {other_key} ".find(f" {name_key} ") >= 0
                        or _query_tokens(name) <= _query_tokens(other_name)
                    )
                ):
                    continue
                if all(
                    any(
                        int(other_start) <= int(start) and int(end) <= int(other_end)
                        for other_start, other_end in other_spans
                    )
                    for start, end in spans
                ):
                    covered_by_longer = True
                    break
            if not covered_by_longer:
                filtered_candidates.append(candidate)
        candidates = filtered_candidates
    if len(candidates) < 2:
        return []

    ordered = sorted(candidates, key=lambda item: item.get("position", 0))
    exact_ordered = [candidate for candidate in ordered if int(candidate.get("position", 0) or 0) <= len(query)]
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def push(source: str, target: str) -> None:
        source = _clean_entity_candidate(source)
        target = _clean_entity_candidate(target)
        key = tuple(sorted((source.lower(), target.lower())))
        if len(source) < 2 or len(target) < 2 or source.lower() == target.lower() or key in seen:
            return
        seen.add(key)
        pairs.append((source, target))

    primary_pool = exact_ordered if len(exact_ordered) >= 2 else ordered
    if len(primary_pool) >= 3:
        for left, right in zip(primary_pool, primary_pool[1:]):
            push(str(left.get("name") or ""), str(right.get("name") or ""))
            if len(pairs) >= limit:
                break
        if len(pairs) < limit:
            push(primary_pool[0]["name"], primary_pool[-1]["name"])
    else:
        push(primary_pool[0]["name"], primary_pool[-1]["name"])
    if len(exact_ordered) >= 3:
        return pairs[:limit]
    if len(exact_ordered) == 2:
        return pairs[:limit]

    ranked = sorted(candidates, key=lambda item: (item.get("score", 0), item.get("edge_count", 0)), reverse=True)
    for left, right in combinations(ranked[:5], 2):
        push(str(left.get("name") or ""), str(right.get("name") or ""))
        if len(pairs) >= limit:
            break

    return pairs[:limit]


def _graph_reasoning_needed(query: str, entity_names: list[str], explicit_pair: tuple[str, str] | None = None) -> bool:
    if explicit_pair:
        return True
    query_lower = query.lower()
    relationship_question = any(
        token in query.lower()
        for token in (
            "connect", "connection", "relationship", "related", "path", "between",
            "link", "bridge", "influence", "cause", "why", "how does", "compare",
            "map", "trace",
            "where", "located", "location", "which slide", "which page",
            "section", "heading", "chapter", "sheet", "table", "chart", "image",
            "figure", "diagram", "contains", "include", "includes",
            *GRAPH_RELATION_QUERY_TRIGGERS,
        )
    )
    if len(entity_names) >= 2:
        return relationship_question
    if len(entity_names) >= 1:
        return relationship_question and any(
            token in query_lower
            for token in (
                "connect", "connection", "relationship", "relationships", "related",
                "linked", "neighbors", "neighborhood", "around", "depends on",
                "what does", "show", "explore",
                "where", "located", "location", "which slide", "which page",
                "section", "heading", "chapter", "sheet", "table", "chart",
                "image", "figure", "diagram", "contains", "include", "includes",
                *GRAPH_RELATION_QUERY_TRIGGERS,
            )
        )
    return False


def _local_pipeline_graph_probe_needed(query: str) -> bool:
    query_lower = str(query or "").lower()
    return any(
        token in query_lower
        for token in (
            "knowledge graph",
            "graph",
            "relationship",
            "relationships",
            "connect",
            "connection",
            "connected",
            "related",
            "path",
            "between",
            "link",
            "links",
            "bridge",
            "trace",
            "provenance",
            "cite",
            "citation",
            "which slide",
            "which page",
            "where",
            "chart",
            "table",
            "diagram",
            "contains",
            *GRAPH_RELATION_QUERY_TRIGGERS,
        )
    )


def _normalized_lookup_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _node_display_name(node: dict) -> str:
    properties = node.get("properties", {}) if isinstance(node.get("properties"), dict) else {}
    return str(
        node.get("name")
        or node.get("label")
        or properties.get("name")
        or properties.get("label")
        or node.get("id")
        or ""
    )


def _node_alias_values(node: dict) -> list[str]:
    properties = node.get("properties", {}) if isinstance(node.get("properties"), dict) else {}
    values = [
        node.get("name"),
        node.get("label"),
        node.get("id"),
        properties.get("name"),
        properties.get("label"),
        properties.get("canonical_name"),
    ]
    for key in ("aliases", "alias", "alternate_names", "synonyms"):
        aliases = node.get(key, properties.get(key))
        if isinstance(aliases, list):
            values.extend(aliases)
        elif aliases:
            values.append(aliases)
    return [str(value) for value in values if value]


def _node_aliases(node_map: dict) -> dict[str, dict]:
    aliases = {}
    for node in node_map.values():
        for value in _node_alias_values(node):
            aliases[value.lower()] = node
            normalized = _normalized_lookup_key(value)
            if normalized:
                aliases[normalized] = node
    return aliases


def _match_graph_node(entity_name: str, node_map: dict) -> dict | None:
    aliases = _node_aliases(node_map)
    search = entity_name.lower()
    if search in aliases:
        return aliases[search]
    normalized_search = _normalized_lookup_key(entity_name)
    if normalized_search in aliases:
        return aliases[normalized_search]

    for alias, node in aliases.items():
        if (search and (search in alias or alias in search)) or (
            normalized_search and (normalized_search in alias or alias in normalized_search)
        ):
            return node
    return None


def _edge_endpoint_name(value, node_map: dict) -> str:
    raw = value
    if isinstance(value, dict):
        properties = value.get("properties", {}) if isinstance(value.get("properties"), dict) else {}
        raw = (
            value.get("label")
            or value.get("name")
            or properties.get("name")
            or properties.get("label")
            or value.get("id")
            or ""
        )
    raw_text = str(raw or "")
    aliases = _node_aliases(node_map)
    node = aliases.get(raw_text.lower()) or aliases.get(_normalized_lookup_key(raw_text))
    return _node_display_name(node) if node else raw_text


def _edge_endpoint_value(edge: dict, side: str):
    properties = edge.get("properties", {}) if isinstance(edge.get("properties"), dict) else {}
    candidates = (
        side,
        f"{side}_id",
        f"{side}Id",
        f"{side}_node",
        f"{side}Node",
        f"{side}_name",
        f"{side}Name",
    )
    for key in candidates:
        value = edge.get(key, properties.get(key))
        if value:
            return value
    if side == "source":
        for key in ("from", "start", "start_node", "startNode", "start_id", "startId", "from_id", "fromId"):
            value = edge.get(key, properties.get(key))
            if value:
                return value
        return None
    for key in ("to", "end", "end_node", "endNode", "end_id", "endId", "to_id", "toId"):
        value = edge.get(key, properties.get(key))
        if value:
            return value
    return None


def _safe_weight(value) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 1.0
    if weight < 0:
        return 0.0
    if weight > 1:
        return 1.0
    return weight


def _flatten_edge_metadata(*metadata_sources: dict | None) -> dict:
    flattened: dict = {}
    for metadata in metadata_sources:
        if not isinstance(metadata, dict):
            continue
        nested = metadata.get("metadata")
        if isinstance(nested, dict):
            flattened.update(nested)
        flattened.update({key: value for key, value in metadata.items() if key != "metadata"})
    return flattened


def _relationship_provenance(metadata: dict | None) -> str:
    if not isinstance(metadata, dict):
        return ""

    parts: list[str] = []

    def compact_value(value, *, limit: int = 6) -> str:
        if isinstance(value, list):
            return ", ".join(str(item) for item in value[:limit] if item not in (None, ""))
        return str(value) if value not in (None, "") else ""

    if metadata.get("local_graph_filename"):
        parts.append(f"local pipeline {metadata['local_graph_filename']}")
    if any(metadata.get(key) not in (None, "", [], {}) for key in EMAIL_GRAPH_METADATA_KEYS):
        message_part = "message"
        subject = compact_value(metadata.get("email_subject"), limit=1)
        message_id = compact_value(metadata.get("email_message_id"), limit=1)
        if subject:
            message_part += f" {subject}"
        elif message_id:
            message_part += f" {message_id}"
        sender = compact_value(metadata.get("email_from") or metadata.get("email_sender"), limit=2)
        if sender:
            message_part += f" from {sender}"
        recipients = compact_value(metadata.get("email_to"), limit=3)
        if recipients:
            message_part += f" to {recipients}"
        cc = compact_value(metadata.get("email_cc"), limit=2)
        if cc:
            message_part += f" cc {cc}"
        date_value = compact_value(metadata.get("email_date_iso") or metadata.get("email_date"), limit=1)
        if date_value:
            message_part += f" date {date_value}"
        if message_id and subject:
            message_part += f" id {message_id}"
        parts.append(message_part)
    if metadata.get("email_part_index") or metadata.get("email_part_kind") or metadata.get("email_content_type"):
        body_part = "message body"
        if metadata.get("email_part_index") is not None:
            body_part += f" part {metadata['email_part_index']}"
        if metadata.get("email_part_kind"):
            body_part += f" {metadata['email_part_kind']}"
        if metadata.get("email_content_type"):
            body_part += f" {metadata['email_content_type']}"
        if metadata.get("email_disposition"):
            body_part += f" disposition {metadata['email_disposition']}"
        parts.append(body_part)
    if metadata.get("email_attachment_filename") or metadata.get("email_attachment_index") is not None:
        attachment_part = "email attachment"
        if metadata.get("email_attachment_filename"):
            attachment_part += f" {metadata['email_attachment_filename']}"
        if metadata.get("email_attachment_mime"):
            attachment_part += f" {metadata['email_attachment_mime']}"
        if metadata.get("email_attachment_index") is not None:
            attachment_part += f" index {metadata['email_attachment_index']}"
        parts.append(attachment_part)
    if metadata.get("email_reply_to"):
        parts.append(f"reply-to {compact_value(metadata.get('email_reply_to'), limit=2)}")
    if metadata.get("email_in_reply_to"):
        parts.append(f"in reply to {compact_value(metadata.get('email_in_reply_to'), limit=2)}")
    if metadata.get("email_references"):
        parts.append(f"references {compact_value(metadata.get('email_references'), limit=3)}")
    if metadata.get("slide_title"):
        slide_part = f"slide {metadata.get('slide')}: {metadata.get('slide_title')}" if metadata.get("slide") else str(metadata["slide_title"])
        if metadata.get("slide_visibility") or metadata.get("slide_state"):
            slide_part += f" visibility {metadata.get('slide_visibility') or metadata.get('slide_state')}"
        parts.append(slide_part)
    elif metadata.get("slide") is not None:
        slide_part = f"slide {metadata['slide']}"
        if metadata.get("slide_visibility") or metadata.get("slide_state"):
            slide_part += f" visibility {metadata.get('slide_visibility') or metadata.get('slide_state')}"
        parts.append(slide_part)
    if metadata.get("slide_part"):
        parts.append(f"slide part {metadata['slide_part']}")
    if metadata.get("slide_relationship_id"):
        parts.append(f"slide relationship {metadata['slide_relationship_id']}")
    if metadata.get("page") is not None:
        parts.append(f"page {metadata['page']}")
    if metadata.get("sheet_name"):
        sheet_part = str(metadata["sheet_name"])
        if metadata.get("sheet_visibility") or metadata.get("sheet_state"):
            sheet_part += f" visibility {metadata.get('sheet_visibility') or metadata.get('sheet_state')}"
        parts.append(sheet_part)
    elif metadata.get("sheet") is not None:
        sheet_part = f"sheet {metadata['sheet']}"
        if metadata.get("sheet_visibility") or metadata.get("sheet_state"):
            sheet_part += f" visibility {metadata.get('sheet_visibility') or metadata.get('sheet_state')}"
        parts.append(sheet_part)
    if metadata.get("sheet_part"):
        parts.append(f"sheet part {metadata['sheet_part']}")
    if metadata.get("sheet_relationship_id"):
        parts.append(f"sheet relationship {metadata['sheet_relationship_id']}")
    if metadata.get("section_label") or metadata.get("section"):
        parts.append(f"section {metadata.get('section_label') or metadata.get('section')}")
    if metadata.get("markdown_code_language") or metadata.get("markdown_code_block_index"):
        code_part = "code block"
        if metadata.get("markdown_code_language"):
            code_part += f" {metadata['markdown_code_language']}"
        if metadata.get("markdown_code_block_index") is not None:
            code_part += f" #{metadata['markdown_code_block_index']}"
        start_line = metadata.get("markdown_start_line")
        end_line = metadata.get("markdown_end_line")
        if start_line is not None and end_line is not None:
            code_part += f" lines {start_line}-{end_line}"
        elif start_line is not None:
            code_part += f" line {start_line}"
        parts.append(code_part)
    if metadata.get("formula_cell"):
        formula_part = f"formula {metadata['formula_cell']}"
        if metadata.get("formula_expression"):
            formula_part += f" = {metadata['formula_expression']}"
        if metadata.get("formula_result"):
            formula_part += f" result {metadata['formula_result']}"
        parts.append(formula_part)
    if metadata.get("conditional_range_label") or metadata.get("conditional_range"):
        conditional_part = f"conditional range {metadata.get('conditional_range_label') or metadata.get('conditional_range')}"
        if metadata.get("conditional_type"):
            conditional_part += f" {metadata['conditional_type']}"
        if metadata.get("conditional_formula1"):
            conditional_part += f" formula {metadata['conditional_formula1']}"
        parts.append(conditional_part)
    if metadata.get("pivot_table_name"):
        pivot_part = f"pivot table {metadata['pivot_table_name']}"
        if metadata.get("pivot_table_range"):
            pivot_part += f" range {metadata['pivot_table_range']}"
        if metadata.get("pivot_table_source_name") or metadata.get("pivot_table_source_ref"):
            pivot_part += f" source {metadata.get('pivot_table_source_name') or metadata.get('pivot_table_source_ref')}"
        parts.append(pivot_part)
    if metadata.get("named_range_name"):
        named_part = f"named range {metadata['named_range_name']}"
        if metadata.get("named_range_ref"):
            named_part += f" {metadata['named_range_ref']}"
        parts.append(named_part)
    if metadata.get("form_kind") or metadata.get("field_type") or metadata.get("label"):
        form_part = "form field"
        if metadata.get("label"):
            form_part += f" {metadata['label']}"
        if metadata.get("field_type"):
            form_part += f" {metadata['field_type']}"
        selected_value = (
            metadata.get("form_field_selected_option")
            or ("checked" if metadata.get("form_field_checked") is True else "unchecked" if metadata.get("form_field_checked") is False else None)
            or metadata.get("value")
        )
        if selected_value:
            form_part += f" value {selected_value}"
        if metadata.get("form_field_options"):
            options = metadata.get("form_field_options")
            if isinstance(options, list):
                form_part += " options " + ", ".join(str(option) for option in options[:6])
            else:
                form_part += f" options {options}"
        if metadata.get("form_field_date_format"):
            form_part += f" date format {metadata['form_field_date_format']}"
        parts.append(form_part)
    if metadata.get("validation_range"):
        validation_part = f"validation range {metadata['validation_range']}"
        if metadata.get("validation_type"):
            validation_part += f" {metadata['validation_type']}"
        parts.append(validation_part)
    if metadata.get("notes_part") or metadata.get("note_part") or metadata.get("note_id"):
        notes_part = f"speaker notes part {metadata.get('notes_part') or metadata.get('note_part') or 'notes'}"
        if metadata.get("notes_relationship_id"):
            notes_part += f" relationship {metadata['notes_relationship_id']}"
        if metadata.get("note_id"):
            notes_part += f" note {metadata['note_id']}"
        parts.append(notes_part)
    if (
        metadata.get("comment_id")
        or metadata.get("comment_kind")
        or metadata.get("comment_part")
        or metadata.get("comment_anchor_text")
        or metadata.get("comment_anchor_cell")
        or metadata.get("author")
    ):
        comment_part = f"comment {metadata.get('comment_id') or metadata.get('comment_kind') or 'markup'}"
        if metadata.get("author"):
            comment_part += f" by {metadata['author']}"
        if metadata.get("date") or metadata.get("comment_datetime"):
            comment_part += f" at {metadata.get('date') or metadata.get('comment_datetime')}"
        anchor = (
            metadata.get("comment_anchor_text")
            or metadata.get("comment_anchor_cell")
            or metadata.get("comment_anchor_section")
            or metadata.get("comment_anchor_paragraph")
        )
        if anchor:
            comment_part += f" anchor {anchor}"
        if metadata.get("comment_part"):
            comment_part += f" part {metadata['comment_part']}"
        parts.append(comment_part)
    if metadata.get("annotation_type") or metadata.get("annotation_subject") or metadata.get("annotation_name"):
        annotation_part = f"annotation {metadata.get('annotation_type') or metadata.get('annotation_name') or 'markup'}"
        if metadata.get("annotation_author"):
            annotation_part += f" by {metadata['annotation_author']}"
        if metadata.get("annotation_subject"):
            annotation_part += f" subject {metadata['annotation_subject']}"
        if metadata.get("annotation_name"):
            annotation_part += f" name {metadata['annotation_name']}"
        parts.append(annotation_part)
    if metadata.get("revision_type") or metadata.get("revision_id"):
        revision_part = f"revision {metadata.get('revision_type') or metadata.get('revision_id')}"
        if metadata.get("revision_author"):
            revision_part += f" by {metadata['revision_author']}"
        if metadata.get("revision_date"):
            revision_part += f" at {metadata['revision_date']}"
        if metadata.get("revision_id"):
            revision_part += f" id {metadata['revision_id']}"
        parts.append(revision_part)
    if metadata.get("styled_text") or metadata.get("style_summary") or metadata.get("emphasis_term") or metadata.get("bullet_levels"):
        style_part = "styled text"
        if metadata.get("emphasis_term"):
            style_part += f" emphasis {metadata['emphasis_term']}"
        elif metadata.get("emphasis_terms"):
            terms = compact_value(metadata.get("emphasis_terms"), limit=4)
            if terms:
                style_part += f" emphasis {terms}"
        styles = compact_value(metadata.get("emphasis_styles"), limit=4)
        if styles:
            style_part += f" styles {styles}"
        bullets = compact_value(metadata.get("bullet_levels"), limit=6)
        if bullets:
            style_part += f" bullet levels {bullets}"
        if metadata.get("style_summary"):
            style_part += f" summary {metadata['style_summary']}"
        parts.append(style_part)
    if metadata.get("shape_title") or metadata.get("shape_description") or metadata.get("shape_name"):
        shape_label = metadata.get("shape_title") or metadata.get("shape_name")
        shape_part = f"shape {shape_label}" if shape_label else "shape"
        if metadata.get("shape_description"):
            shape_part += f" description {metadata['shape_description']}"
        parts.append(shape_part)
    if metadata.get("uri") or metadata.get("link_kind"):
        link_part = "link"
        if metadata.get("uri"):
            link_part += f" uri {metadata['uri']}"
        if metadata.get("link_kind"):
            link_part += f" kind {metadata['link_kind']}"
        if metadata.get("relationship_id"):
            link_part += f" relationship {metadata['relationship_id']}"
        if metadata.get("image_relationship_id"):
            link_part += f" image relationship {metadata['image_relationship_id']}"
        if metadata.get("image_part"):
            link_part += f" image {metadata['image_part']}"
        parts.append(link_part)
    if metadata.get("connector_title") or metadata.get("connector_description") or metadata.get("connector_name"):
        connector_label = metadata.get("connector_title") or metadata.get("connector_name")
        connector_part = f"connector {connector_label}" if connector_label else "connector"
        if metadata.get("source_label") and metadata.get("target_label"):
            connector_part += f" {metadata['source_label']} to {metadata['target_label']}"
        if metadata.get("connector_description"):
            connector_part += f" description {metadata['connector_description']}"
        parts.append(connector_part)
    if (
        metadata.get("diagram_part")
        or metadata.get("diagram_node_id")
        or metadata.get("diagram_connection_type")
        or metadata.get("source_diagram_node_id")
        or metadata.get("target_diagram_node_id")
    ):
        diagram_part = "diagram"
        if metadata.get("diagram_node_id"):
            diagram_part += f" node {metadata['diagram_node_id']}"
        if metadata.get("diagram_node_type"):
            diagram_part += f" node type {metadata['diagram_node_type']}"
        if metadata.get("diagram_order") is not None:
            diagram_part += f" order {metadata['diagram_order']}"
        if metadata.get("diagram_level") is not None:
            diagram_part += f" level {metadata['diagram_level']}"
        parent_label = metadata.get("parent_label") or metadata.get("parent_diagram_node_id")
        if parent_label:
            diagram_part += f" parent {parent_label}"
        source_label = metadata.get("source_label") or metadata.get("source_diagram_node_id")
        target_label = metadata.get("target_label") or metadata.get("target_diagram_node_id")
        if source_label and target_label:
            diagram_part += f" edge {source_label} to {target_label}"
        elif source_label:
            diagram_part += f" source {source_label}"
        elif target_label:
            diagram_part += f" target {target_label}"
        if metadata.get("diagram_connection_type"):
            diagram_part += f" type {metadata['diagram_connection_type']}"
        if metadata.get("diagram_part"):
            diagram_part += f" part {metadata['diagram_part']}"
        parts.append(diagram_part)
    if metadata.get("layout_bbox") or metadata.get("layout_zone") or metadata.get("layout_relation"):
        layout_part = "layout"
        if metadata.get("layout_relation"):
            layout_part += f" {metadata['layout_relation']}"
        if metadata.get("related_label"):
            layout_part += f" near {metadata['related_label']}"
        if metadata.get("visual_distance_pct") not in (None, ""):
            layout_part += f" distance {metadata['visual_distance_pct']}%"
        if metadata.get("layout_zone"):
            layout_part += f" {metadata['layout_zone']}"
        if metadata.get("target_layout_zone"):
            layout_part += f" target {metadata['target_layout_zone']}"
        if metadata.get("layout_bbox"):
            layout_part += f" {metadata['layout_bbox']}"
        parts.append(layout_part)
    if metadata.get("table") is not None:
        table_part = f"table {metadata['table']}"
        if metadata.get("row") is not None:
            table_part += f" row {metadata['row']}"
        if metadata.get("column") is not None:
            table_part += f" column {metadata['column']}"
        if metadata.get("table_cell_reference"):
            table_part += f" cell {metadata['table_cell_reference']}"
        elif metadata.get("cell"):
            table_part += f" cell {metadata['cell']}"
        span_parts = []
        if metadata.get("table_cell_grid_span"):
            span_parts.append(f"grid span {metadata['table_cell_grid_span']}")
        if metadata.get("table_cell_row_span"):
            span_parts.append(f"row span {metadata['table_cell_row_span']}")
        if span_parts:
            table_part += f" ({', '.join(span_parts)})"
        if metadata.get("table_cell_style_name"):
            table_part += f" style {metadata['table_cell_style_name']}"
        if metadata.get("table_cell_fill_color"):
            table_part += f" fill {metadata['table_cell_fill_color']}"
        if metadata.get("table_cell_visual_summary"):
            table_part += f" visual {metadata['table_cell_visual_summary']}"
        if metadata.get("cell_value_type"):
            table_part += f" type {metadata['cell_value_type']}"
        parts.append(table_part)
    if metadata.get("table_display_name") or metadata.get("table_name") or metadata.get("table_range"):
        table_label = metadata.get("table_display_name") or metadata.get("table_name") or "table"
        table_part = f"table {table_label}"
        if metadata.get("table_range"):
            table_part += f" range {metadata['table_range']}"
        parts.append(table_part)
    if metadata.get("table_source"):
        parts.append(f"table source {metadata['table_source']}")
    if metadata.get("chart_label") or metadata.get("chart_title") or metadata.get("chart_part"):
        chart_part = str(metadata.get("chart_label") or metadata.get("chart_title") or metadata.get("chart_part"))
        if metadata.get("chart_point_label"):
            chart_part += f" point {metadata['chart_point_label']}"
        elif metadata.get("series_name"):
            chart_part += f" series {metadata['series_name']}"
        if metadata.get("chart_series_visual_summary"):
            chart_part += f" visual {metadata['chart_series_visual_summary']}"
        if metadata.get("chart_point_visual_summary"):
            chart_part += f" point visual {metadata['chart_point_visual_summary']}"
        if metadata.get("chart_trendline_type"):
            chart_part += f" trendline {metadata['chart_trendline_type']}"
        if metadata.get("chart_trendline_visual_summary"):
            chart_part += f" trendline visual {metadata['chart_trendline_visual_summary']}"
        parts.append(f"chart {chart_part}")
    if metadata.get("media_part") or metadata.get("media_kind"):
        media_part = f"media {metadata.get('media_kind') or metadata.get('media_part')}"
        if metadata.get("media_part"):
            media_part += f" part {metadata['media_part']}"
        if metadata.get("media_mime"):
            media_part += f" {metadata['media_mime']}"
        if metadata.get("media_size_bytes") is not None:
            media_part += f" {metadata['media_size_bytes']} bytes"
        parts.append(media_part)
    if metadata.get("embedded_file_filename") or metadata.get("embedded_file_name"):
        attachment = metadata.get("embedded_file_filename") or metadata.get("embedded_file_name")
        attachment_part = f"attachment {attachment}"
        if metadata.get("embedded_file_mime"):
            attachment_part += f" {metadata['embedded_file_mime']}"
        if metadata.get("embedded_file_size_bytes") is not None:
            attachment_part += f" {metadata['embedded_file_size_bytes']} bytes"
        parts.append(attachment_part)
    if metadata.get("archive_entry_path") or metadata.get("archive_entry_filename"):
        entry = metadata.get("archive_entry_path") or metadata.get("archive_entry_filename")
        entry_part = f"archive entry {entry}"
        if metadata.get("archive_entry_mime"):
            entry_part += f" {metadata['archive_entry_mime']}"
        if metadata.get("archive_entry_size_bytes") is not None:
            entry_part += f" {metadata['archive_entry_size_bytes']} bytes"
        parts.append(entry_part)
    if metadata.get("legacy_office_stream_path") or metadata.get("legacy_office_format"):
        legacy_part = f"legacy Office {metadata.get('legacy_office_format') or 'document'}"
        if metadata.get("legacy_office_stream_path"):
            legacy_part += f" stream {metadata['legacy_office_stream_path']}"
        if metadata.get("legacy_office_stream_index") is not None:
            legacy_part += f" index {metadata['legacy_office_stream_index']}"
        if metadata.get("legacy_office_extraction_method"):
            legacy_part += f" via {metadata['legacy_office_extraction_method']}"
        if metadata.get("legacy_office_string_count") is not None:
            legacy_part += f" strings {metadata['legacy_office_string_count']}"
        if metadata.get("legacy_office_stream_size_bytes") is not None:
            legacy_part += f" {metadata['legacy_office_stream_size_bytes']} bytes"
        if metadata.get("legacy_office_stream_sha256"):
            legacy_part += f" sha256 {metadata['legacy_office_stream_sha256']}"
        parts.append(legacy_part)
    if metadata.get("ocr_source"):
        engine = metadata.get("ocr_engine")
        parts.append(f"OCR {metadata['ocr_source']}" + (f" via {engine}" if engine else ""))
    if metadata.get("element_type"):
        parts.append(f"element {metadata['element_type']}")
    if metadata.get("location"):
        parts.append(str(metadata["location"]))

    unique: list[str] = []
    seen = set()
    for part in parts:
        cleaned = re.sub(r"\s+", " ", str(part or "")).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return "; ".join(unique[:5])


def _format_graph_relationship(rel: dict, *, path_style: bool = False) -> str:
    connector = "--" if path_style else "-->"
    text = f"{rel.get('source')} --[{rel.get('relation')}]{connector} {rel.get('target')}"
    provenance = rel.get("provenance") or _relationship_provenance(rel.get("metadata"))
    return f"{text} ({provenance})" if provenance else text


def _format_graph_relationship_for_answer(
    rel: dict,
    *,
    path_style: bool = False,
    include_provenance: bool = True,
) -> str:
    connector = "--" if path_style else "-->"
    source = _display_graph_endpoint(rel.get("source"), bridge=False)
    target = _display_graph_endpoint(rel.get("target"), bridge=False)
    relation = rel.get("relation") or "related_to"
    text = f"{source} --[{relation}]{connector} {target}"
    provenance = rel.get("provenance") or _relationship_provenance(rel.get("metadata"))
    return f"{text} ({provenance})" if include_provenance and provenance else text


def _format_graph_path_for_answer(path: dict) -> str:
    if not isinstance(path, dict):
        return ""
    nodes = [str(node) for node in path.get("nodes", []) or [] if node]
    relationships = [rel for rel in path.get("relationships", []) or [] if isinstance(rel, dict)]
    if nodes and len(relationships) == len(nodes) - 1:
        display_nodes = _display_graph_path_nodes(nodes)
        if len(display_nodes) == len(nodes):
            pieces = [display_nodes[0]]
            for index, relationship in enumerate(relationships):
                relation = relationship.get("relation") or "related_to"
                pieces.append(f"--[{relation}]-- {display_nodes[index + 1]}")
            return " ".join(pieces)
    if nodes:
        return " -> ".join(_display_graph_path_nodes(nodes))
    text = str(path.get("text") or "").strip()
    return text


def _format_bridge_entity_for_answer(bridge: dict) -> str:
    if not isinstance(bridge, dict):
        return ""
    bridge_name = _display_graph_endpoint(bridge.get("name"), bridge=True)
    connects = bridge.get("connects") if isinstance(bridge.get("connects"), list) else []
    connect_texts = []
    for item in connects[:4]:
        if not isinstance(item, dict):
            continue
        source = _display_graph_endpoint(item.get("source"), bridge=False)
        target = _display_graph_endpoint(item.get("target"), bridge=False)
        if source and target:
            connect_texts.append(f"{source} to {target}")
    if connect_texts:
        return f"{bridge_name} bridges " + "; ".join(connect_texts)
    return str(bridge.get("text") or bridge_name)


def _graph_edge_evidence_families(edge: dict) -> list[str]:
    relation = str(edge.get("relation") or edge.get("label") or edge.get("type") or "").lower()
    metadata = edge.get("metadata") if isinstance(edge.get("metadata"), dict) else {}
    searchable = " ".join(
        re.sub(r"\s+", " ", str(value or "")).strip()
        for value in (
            edge.get("source"),
            edge.get("target"),
            relation,
            edge.get("evidence"),
            edge.get("provenance"),
            *metadata.values(),
        )
        if value not in (None, "")
    ).lower()
    families: list[str] = []

    def add(family: str, condition: bool) -> None:
        if condition and family not in families:
            families.append(family)

    add("connector", "connector" in searchable or "shape_connector" in searchable or "arrow" in searchable)
    add("diagram", bool(metadata.get("diagram_part") or metadata.get("diagram_node_id") or "diagram" in searchable or "smartart" in searchable))
    add("chart", bool(metadata.get("chart_part") or metadata.get("chart_source_part") or "chart" in searchable))
    add("table", bool(metadata.get("table_name") or metadata.get("table") or re.search(r"\btable\b", searchable)))
    add("code_block", bool(
        metadata.get("markdown_code_language")
        or metadata.get("markdown_code_block_index") is not None
        or metadata.get("element_type") == "code_block"
        or relation in {"contains_code_block", "uses_language"}
    ))
    add("styled_text", bool(metadata.get("styled_text") or metadata.get("style_summary") or metadata.get("emphasis_term") or relation in {"emphasizes", "uses_bullet_structure"}))
    add("link", bool(metadata.get("uri") or metadata.get("link_kind") or relation in {"links_to", "contains_link"}))
    add("image", bool(metadata.get("image_part") or metadata.get("image_relationship_id") or "image" in searchable or "picture" in searchable))
    add("form_field", bool(metadata.get("form_kind") or metadata.get("field_type") or metadata.get("control_part") or "form_field" in searchable))
    add("speaker_notes", bool(metadata.get("notes_part") or "speaker_notes" in searchable or "speaker note" in searchable))
    add("comment", bool(metadata.get("comment_id") or metadata.get("comment_anchor_text") or "comment" in searchable))
    add("revision", bool(metadata.get("revision_id") or metadata.get("revision_type") or "revision" in searchable))
    add("ocr", bool(metadata.get("ocr_source") or metadata.get("ocr_engine") or "ocr" in searchable))
    add("media", bool(metadata.get("media_part") or metadata.get("media_kind") or "media" in searchable))
    add("message", bool(
        any(metadata.get(key) not in (None, "", [], {}) for key in EMAIL_GRAPH_METADATA_KEYS)
        or relation in EMAIL_GRAPH_RELATIONS
        or "email_" in searchable
        or "message/rfc822" in searchable
    ))
    add("attachment", bool(metadata.get("embedded_file_name") or metadata.get("embedded_file_filename") or "attachment" in searchable or "embedded" in searchable))
    add("archive_entry", bool(metadata.get("archive_entry") or metadata.get("archive_entry_path") or "archive_entry" in searchable))
    add("legacy_office", bool(metadata.get("legacy_office_stream_path") or "legacy_office_" in searchable))
    add("formula", bool(metadata.get("formula_cell") or metadata.get("formula_expression") or "formula" in searchable))
    add("pivot_table", bool(metadata.get("pivot_table_name") or metadata.get("pivot_table_range") or "pivot_table" in searchable or "pivot table" in searchable))
    add("conditional_format", bool(metadata.get("conditional_range") or metadata.get("conditional_type") or "conditional_format" in searchable))
    add("presentation_structure", bool(metadata.get("slide_part") or metadata.get("slide_visibility") or metadata.get("slide_state") or metadata.get("slide_title") or metadata.get("slide")))
    add("workbook_structure", bool(metadata.get("sheet_part") or metadata.get("sheet_visibility") or metadata.get("sheet_state") or metadata.get("sheet_name") or metadata.get("sheet")))
    add(
        "layout",
        bool(
            metadata.get("layout_relation")
            or relation == "visually_near"
            or metadata.get("layout_bbox")
            or metadata.get("layout_zone")
            or (
                relation in {"has_slide", "contains_connector"}
                and (metadata.get("slide") or metadata.get("slide_title"))
            )
        ),
    )
    add("semantic", not _is_structural_relation(relation) or not families)
    return families or ["semantic"]


def _graph_edge_provenance_summary(edge: dict) -> str:
    metadata = edge.get("metadata") if isinstance(edge.get("metadata"), dict) else {}
    hints: list[str] = []

    def add(label: str, value) -> None:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        rendered = f"{label}={text}"
        if text and rendered not in hints:
            hints.append(rendered)

    add("file", metadata.get("local_graph_filename"))
    add("slide", metadata.get("slide_title") or metadata.get("slide"))
    add("page", metadata.get("page"))
    add("sheet", metadata.get("sheet_name") or metadata.get("sheet"))
    add("table", metadata.get("table_name") or metadata.get("table"))
    add("cell", metadata.get("cell") or metadata.get("formula_cell"))
    add("code", metadata.get("markdown_code_language"))
    add("code_block", metadata.get("markdown_code_block_index"))
    add("chart", metadata.get("chart_title") or metadata.get("chart_part"))
    add("diagram", metadata.get("diagram_part"))
    add("image", metadata.get("image_part"))
    add("message", metadata.get("email_subject") or metadata.get("email_message_id"))
    add("from", metadata.get("email_from") or metadata.get("email_sender"))
    add("to", metadata.get("email_to"))
    add("email_attachment", metadata.get("email_attachment_filename"))
    add("legacy_stream", metadata.get("legacy_office_stream_path"))
    add("source", metadata.get("chart_source_part") or metadata.get("embedded_file_name") or metadata.get("embedded_file_filename"))
    add("uri", metadata.get("uri"))
    add("layout", metadata.get("layout_bbox") or metadata.get("layout_zone"))
    add("visual", metadata.get("layout_relation"))
    add("near", metadata.get("related_label"))
    add("distance", metadata.get("visual_distance_pct"))
    add("visibility", metadata.get("slide_visibility") or metadata.get("sheet_visibility") or metadata.get("slide_state") or metadata.get("sheet_state"))
    return "; ".join(hints[:6])


def _graph_edge_rationale(edge: dict) -> str:
    relation = str(edge.get("relation") or "related_to")
    families = _graph_edge_evidence_families(edge)
    family_text = ", ".join(family.replace("_", " ") for family in families[:4])
    provenance_summary = _graph_edge_provenance_summary(edge)
    text = f"{relation.replace('_', ' ')} evidence"
    if family_text:
        text += f" from {family_text}"
    if provenance_summary:
        text += f" at {provenance_summary}"
    return _compact_graph_evidence(text, 240) or text


def _graph_path_explanation(path: dict) -> dict:
    relationships = [relationship for relationship in path.get("relationships", []) or [] if isinstance(relationship, dict)]
    raw_nodes = [str(node) for node in path.get("nodes", []) or [] if node]
    nodes = _display_graph_path_nodes(raw_nodes)
    source = _display_graph_endpoint(path.get("source") or (raw_nodes or [""])[0], bridge=False)
    target = _display_graph_endpoint(path.get("target") or (raw_nodes or [""])[-1], bridge=False)
    bridge = nodes[1] if len(nodes) > 2 else ""
    evidence_families: list[str] = []
    provenance_summary: list[str] = []
    for relationship in relationships:
        for family in relationship.get("evidence_families") or _graph_edge_evidence_families(relationship):
            if family and family not in evidence_families:
                evidence_families.append(str(family))
        summary = relationship.get("provenance_summary") or _graph_edge_provenance_summary(relationship)
        if summary and summary not in provenance_summary:
            provenance_summary.append(str(summary))
    relation_text = _format_graph_path_for_answer(path) or "; ".join(
        _format_graph_relationship_for_answer(relationship, path_style=True, include_provenance=False)
        for relationship in relationships
        if relationship.get("source") and relationship.get("target")
    )
    family_text = ", ".join(family.replace("_", " ") for family in evidence_families[:5]) or "graph"
    connector = bridge or " -> ".join(nodes[1:-1])
    if connector:
        explanation = f"{connector} connects {source} and {target} through {family_text} evidence"
    else:
        explanation = f"{source} connects to {target} through {family_text} evidence"
    if relation_text:
        explanation += f": {relation_text}"
    if provenance_summary:
        friendly_source = _friendly_graph_source_location({"provenance_summary": provenance_summary[0]})
        if friendly_source:
            explanation += f" Cite {friendly_source}."
    answer_hint = f"Use this path to explain how {source} and {target} are related, then cite the supporting {family_text} evidence."
    return {
        "evidence_families": evidence_families[:8],
        "provenance_summary": provenance_summary[:6],
        "display_nodes": nodes,
        "display_text": relation_text,
        "contains_source_evidence_bridge": "source evidence" in {node.lower() for node in nodes},
        "explanation": _compact_graph_evidence(explanation, 620),
        "answer_hint": _compact_graph_evidence(answer_hint, 360),
    }


def _normalize_graph_edge(edge: dict, node_map: dict) -> dict:
    properties = edge.get("properties", {}) if isinstance(edge.get("properties"), dict) else {}
    metadata = _flatten_edge_metadata(edge.get("metadata"), properties)
    relation = (
        edge.get("label")
        or edge.get("type")
        or edge.get("relationship_type")
        or edge.get("relation")
        or properties.get("type")
        or properties.get("relationship_type")
        or properties.get("relation")
        or properties.get("label")
    ) or "related_to"
    return {
        "source": _edge_endpoint_name(_edge_endpoint_value(edge, "source"), node_map),
        "target": _edge_endpoint_name(_edge_endpoint_value(edge, "target"), node_map),
        "relation": relation,
        "weight": _safe_weight(edge.get("weight", properties.get("weight", properties.get("confidence", properties.get("score", 1))))),
        "evidence": edge.get("evidence") or properties.get("evidence") or properties.get("description") or properties.get("source_text"),
        "metadata": metadata,
        "provenance": _relationship_provenance(metadata),
    }


def _normalize_graph_edges(raw_edges: list, node_map: dict) -> list[dict]:
    return [
        _normalize_graph_edge(edge, node_map)
        for edge in raw_edges
        if isinstance(edge, dict)
    ]


def _merge_node_map(target: dict, source: dict) -> None:
    for key, node in source.items():
        if not key or not isinstance(node, dict):
            continue
        target.setdefault(str(key).lower(), node)


def _local_pipeline_graph_data(
    domain: str | None = None,
    *,
    pipeline_run_ids: list[str] | None = None,
    max_runs: int = 12,
    max_edges: int = 500,
) -> tuple[dict, list[dict], list[dict]]:
    try:
        runs = get_pipeline_run_store().list_runs()
    except Exception as exc:
        logger.debug(f"Local GraphRAG pipeline graph fallback unavailable: {exc}")
        return {}, [], []

    node_map: dict[str, dict] = {}
    raw_edges: list[dict] = []
    sources: list[dict] = []
    seen_edges: set[tuple[str, str, str, str]] = set()
    wanted_domain = str(domain or "").lower()
    wanted_run_ids = set(pipeline_run_ids or [])

    for run in reversed(runs):
        if len(raw_edges) >= max_edges or (not wanted_run_ids and len(sources) >= max_runs):
            break
        if not isinstance(run, dict):
            continue
        run_id = str(run.get("run_id") or "")
        if wanted_run_ids and run_id not in wanted_run_ids:
            continue
        graph = run.get("local_graph") if isinstance(run.get("local_graph"), dict) else {}
        graph_edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
        if not graph_edges:
            continue
        document = run.get("document") if isinstance(run.get("document"), dict) else {}
        graph_domain = str(graph.get("domain") or document.get("domain") or "").lower()
        if wanted_domain and wanted_domain != "general" and graph_domain and graph_domain != wanted_domain:
            continue

        filename = str(graph.get("filename") or document.get("filename") or "")
        source_record = {
            "run_id": run_id,
            "filename": filename,
            "domain": graph_domain or None,
            "status": run.get("status"),
            "edge_count": graph.get("edge_count", len(graph_edges)),
            "updated_at": run.get("updated_at") or run.get("created_at"),
        }
        sources.append(source_record)

        for node in graph.get("nodes") if isinstance(graph.get("nodes"), list) else []:
            if not isinstance(node, dict):
                continue
            name = _node_display_name(node)
            if not name:
                continue
            node_map.setdefault(name.lower(), {
                "name": name,
                "label": name,
                "id": node.get("id") or name,
                "type": node.get("type") or "concept",
                "source": "local_pipeline_run",
                "run_id": run_id,
            })

        for edge in graph_edges:
            if len(raw_edges) >= max_edges:
                break
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            relation = str(edge.get("label") or edge.get("type") or edge.get("relation") or "related_to")
            evidence = str(edge.get("evidence") or "")
            edge_key = (source.lower(), relation.lower(), target.lower(), evidence[:120].lower())
            if not source or not target or edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            node_map.setdefault(source.lower(), {"name": source, "label": source, "id": source, "type": "concept", "source": "local_pipeline_run", "run_id": run_id})
            node_map.setdefault(target.lower(), {"name": target, "label": target, "id": target, "type": "concept", "source": "local_pipeline_run", "run_id": run_id})
            metadata = _flatten_edge_metadata(edge.get("metadata"), edge.get("properties"))
            metadata.update({
                "local_graph_run_id": run_id,
                "local_graph_filename": filename,
                "local_graph_domain": graph_domain or None,
                "local_graph_status": run.get("status"),
            })
            raw_edges.append({
                **edge,
                "source": source,
                "target": target,
                "label": relation,
                "type": relation,
                "metadata": metadata,
                "properties": {
                    **(edge.get("properties") if isinstance(edge.get("properties"), dict) else {}),
                    **metadata,
                    "type": relation,
                    "relationship_type": relation,
                },
            })

    return node_map, raw_edges, sources


def _handoff_count(handoff: dict, key: str) -> int:
    counts = handoff.get("counts") if isinstance(handoff.get("counts"), dict) else {}
    try:
        return int(counts.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _handoff_rehearsal_match_count(handoff: dict, key: str, rehearsal_key: str) -> int:
    count = _handoff_count(handoff, key)
    if count:
        return count
    rehearsals = handoff.get("retrieval_rehearsals") if isinstance(handoff.get("retrieval_rehearsals"), list) else []
    total = 0
    for rehearsal in rehearsals:
        if not isinstance(rehearsal, dict):
            continue
        try:
            total += int(rehearsal.get(rehearsal_key, 0) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _local_pipeline_interaction_handoffs(
    domain: str | None = None,
    *,
    pipeline_run_ids: list[str] | None = None,
    max_runs: int = 12,
) -> list[dict]:
    try:
        runs = get_pipeline_run_store().list_runs()
    except Exception as exc:
        logger.debug(f"Local GraphRAG pipeline interaction handoff unavailable: {exc}")
        return []

    wanted_domain = str(domain or "").lower()
    wanted_run_ids = set(pipeline_run_ids or [])
    handoffs: list[dict] = []
    seen_run_ids: set[str] = set()

    for run in reversed(runs):
        if not wanted_run_ids and len(handoffs) >= max_runs:
            break
        if not isinstance(run, dict):
            continue
        run_id = str(run.get("run_id") or "")
        if wanted_run_ids and run_id not in wanted_run_ids:
            continue
        if run_id in seen_run_ids:
            continue
        handoff = run.get("interaction_handoff") if isinstance(run.get("interaction_handoff"), dict) else {}
        if not handoff:
            continue
        document = run.get("document") if isinstance(run.get("document"), dict) else {}
        local_graph = run.get("local_graph") if isinstance(run.get("local_graph"), dict) else {}
        local_context = run.get("local_context") if isinstance(run.get("local_context"), dict) else {}
        handoff_domain = str(
            handoff.get("domain")
            or local_graph.get("domain")
            or local_context.get("domain")
            or document.get("domain")
            or ""
        ).lower()
        if wanted_domain and wanted_domain != "general" and handoff_domain and handoff_domain != wanted_domain:
            continue

        filename = str(
            handoff.get("filename")
            or local_graph.get("filename")
            or local_context.get("filename")
            or document.get("filename")
            or ""
        )
        raw_lanes = handoff.get("lanes") if isinstance(handoff.get("lanes"), list) else []
        raw_cards = handoff.get("capability_cards") if isinstance(handoff.get("capability_cards"), list) else []
        raw_anchors = handoff.get("graph_anchors") if isinstance(handoff.get("graph_anchors"), list) else []
        raw_prompts = handoff.get("starter_prompts") if isinstance(handoff.get("starter_prompts"), list) else []
        raw_rehearsals = handoff.get("retrieval_rehearsals") if isinstance(handoff.get("retrieval_rehearsals"), list) else []
        lanes = [
            {
                "id": lane.get("id"),
                "label": lane.get("label"),
                "ready": lane.get("ready", False),
                "count": lane.get("count", 0),
                "detail": lane.get("detail"),
            }
            for lane in raw_lanes[:8]
            if isinstance(lane, dict)
        ]
        capability_cards = [
            {
                "id": card.get("id"),
                "kind": card.get("kind"),
                "label": card.get("label"),
                "count": card.get("count", 0),
                "detail": card.get("detail"),
            }
            for card in raw_cards[:8]
            if isinstance(card, dict)
        ]
        graph_anchors = [
            {
                "id": anchor.get("id"),
                "kind": anchor.get("kind"),
                "title": anchor.get("title") or anchor.get("label"),
                "label": anchor.get("label") or anchor.get("title"),
                "score": anchor.get("score"),
                "source": anchor.get("source"),
                "relation": anchor.get("relation"),
                "target": anchor.get("target"),
                "nodes": anchor.get("nodes") if isinstance(anchor.get("nodes"), list) else [],
                "neighbours": anchor.get("neighbours") if isinstance(anchor.get("neighbours"), list) else [],
                "relationships": anchor.get("relationships") if isinstance(anchor.get("relationships"), list) else [],
                "evidence": _compact_graph_evidence(anchor.get("evidence"), limit=420),
                "metadata": anchor.get("metadata") if isinstance(anchor.get("metadata"), dict) else {},
            }
            for anchor in raw_anchors[:10]
            if isinstance(anchor, dict)
        ]
        starter_prompts = [
            {
                "id": prompt.get("id"),
                "kind": prompt.get("kind"),
                "label": prompt.get("label"),
                "prompt": _compact_graph_evidence(prompt.get("prompt"), limit=520),
            }
            for prompt in raw_prompts[:6]
            if isinstance(prompt, dict)
        ]
        retrieval_rehearsals = [
            {
                "id": rehearsal.get("id"),
                "prompt_id": rehearsal.get("prompt_id"),
                "kind": rehearsal.get("kind"),
                "label": rehearsal.get("label"),
                "prompt": _compact_graph_evidence(rehearsal.get("prompt"), limit=520),
                "status": rehearsal.get("status"),
                "score": rehearsal.get("score"),
                "matched_terms": rehearsal.get("matched_terms", []) if isinstance(rehearsal.get("matched_terms"), list) else [],
                "matched_anchor_count": rehearsal.get("matched_anchor_count", 0),
                "matched_chunk_count": rehearsal.get("matched_chunk_count", 0),
                "top_anchors": rehearsal.get("top_anchors", [])[:4] if isinstance(rehearsal.get("top_anchors"), list) else [],
                "top_chunks": rehearsal.get("top_chunks", [])[:4] if isinstance(rehearsal.get("top_chunks"), list) else [],
            }
            for rehearsal in raw_rehearsals[:6]
            if isinstance(rehearsal, dict)
        ]
        record = {
            "run_id": run_id,
            "filename": filename,
            "domain": handoff_domain or None,
            "ready": handoff.get("ready", False),
            "recommended_mode": handoff.get("recommended_mode"),
            "answer_guidance": handoff.get("answer_guidance"),
            "citation_scopes": handoff.get("citation_scopes", []) if isinstance(handoff.get("citation_scopes"), list) else [],
            "lanes": lanes,
            "capability_cards": capability_cards,
            "graph_anchors": graph_anchors,
            "starter_prompts": starter_prompts,
            "retrieval_rehearsals": retrieval_rehearsals,
            "counts": {
                "graph_anchors": _handoff_count(handoff, "graph_anchors") or len(graph_anchors),
                "capability_cards": _handoff_count(handoff, "capability_cards") or len(capability_cards),
                "starter_prompts": _handoff_count(handoff, "starter_prompts") or len(starter_prompts),
                "retrieval_rehearsals": _handoff_count(handoff, "retrieval_rehearsals") or len(retrieval_rehearsals),
                "rehearsed_chunk_matches": _handoff_count(handoff, "rehearsed_chunk_matches"),
                "rehearsed_anchor_matches": _handoff_count(handoff, "rehearsed_anchor_matches"),
            },
            "updated_at": run.get("updated_at") or run.get("created_at"),
        }
        handoffs.append(record)
        if run_id:
            seen_run_ids.add(run_id)

    return handoffs


STRUCTURAL_GRAPH_RELATIONS = {
    "contains",
    "has_page",
    "has_slide",
    "has_hidden_slide",
    "has_slide_visibility",
    "has_slide_id",
    "uses_slide_part",
    "has_sheet",
    "has_hidden_sheet",
    "has_sheet_visibility",
    "has_sheet_id",
    "uses_sheet_part",
    "has_chapter",
    "has_attachment",
    "has_archive_entry",
    "has_section",
    "has_subsection",
    "contains_heading",
    "contains_text",
    "contains_table",
    "contains_pivot_table",
    "contains_chart",
    "contains_shape",
    "contains_connector",
    "contains_image",
    "contains_link",
    "contains_comment",
    "contains_annotation",
    "contains_form_field",
    "contains_footnote",
    "contains_endnote",
    "contains_outline_item",
    "contains_revision",
    "contains_speaker_notes",
    "contains_media",
    "contains_embedded_object",
    "contains_archive_entry",
    "contains_code_block",
    "contains_diagram_node",
    "contains_diagram_relationship",
    "covers_range",
    "visually_near",
    "uses_source",
    "groups_by",
    "columns_by",
    "filters_by",
    "aggregates",
    "depends_on",
    "evaluated_by",
    "uses_condition",
    "applies_to",
    "uses_formula",
    "calculates",
    "labeled_by",
}

RELATION_QUERY_ALIASES = {
    "has_page": {"page", "pages", "where", "location", "located", "found", "document"},
    "has_slide": {"slide", "slides", "deck", "presentation", "powerpoint", "where", "location", "located"},
    "has_hidden_slide": {"hidden", "visibility", "slide", "slides", "deck", "presentation", "powerpoint", "source", "assumption"},
    "has_slide_visibility": {"hidden", "visible", "visibility", "state", "slide", "slides", "deck", "presentation", "powerpoint"},
    "has_slide_id": {"slide", "id", "identifier", "deck", "presentation", "powerpoint"},
    "uses_slide_part": {"slide", "part", "source", "xml", "deck", "presentation", "powerpoint", "provenance"},
    "has_sheet": {"sheet", "worksheet", "workbook", "spreadsheet", "excel", "where", "location"},
    "has_hidden_sheet": {"hidden", "visibility", "sheet", "worksheet", "workbook", "spreadsheet", "excel", "source", "assumption"},
    "has_sheet_visibility": {"hidden", "visible", "visibility", "state", "sheet", "worksheet", "workbook", "spreadsheet", "excel"},
    "has_sheet_id": {"sheet", "id", "identifier", "workbook", "spreadsheet", "excel"},
    "uses_sheet_part": {"sheet", "part", "source", "xml", "workbook", "spreadsheet", "excel", "provenance"},
    "has_chapter": {"chapter", "ebook", "section", "where", "location"},
    "has_attachment": {"attachment", "embedded", "exhibit", "source", "file", "where", "location"},
    "has_archive_entry": {"archive", "zip", "bundle", "entry", "source", "file", "where", "location"},
    "has_message_id": {"email", "message", "message id", "id", "identifier", "header", "source"},
    "has_subject": {"email", "message", "subject", "title", "topic", "header"},
    "has_reply_to": {"email", "message", "reply", "reply-to", "recipient", "header"},
    "has_section": {"section", "heading", "part", "under", "where", "location", "located"},
    "has_subsection": {"subsection", "section", "heading", "under", "nested"},
    "has_description": {"description", "describes", "alt", "accessibility", "title", "label"},
    "contains_heading": {"heading", "title", "section", "contains", "includes"},
    "contains_text": {"text", "paragraph", "body", "mentions", "contains", "includes"},
    "contains_email_body": {"email", "message", "body", "text", "claim", "content", "contains", "includes"},
    "contains_table": {"table", "row", "column", "cells", "spreadsheet", "contains", "includes"},
    "contains_pivot_table": {"pivot", "pivot table", "table", "source", "aggregate", "summary", "spreadsheet", "excel", "contains", "includes"},
    "contains_chart": {"chart", "graph", "plot", "metric", "visual", "contains", "includes"},
    "has_axis": {"chart", "axis", "measure", "dimension", "category", "value", "x axis", "y axis"},
    "has_legend": {"chart", "legend", "series", "key"},
    "has_data_labels": {"chart", "data", "labels", "values", "series", "category"},
    "contains_shape": {"shape", "object", "alt", "accessibility", "title", "description", "label", "contains", "includes"},
    "contains_shape_group": {"shape", "shapes", "group", "groups", "grouped", "visual", "visuals", "cluster", "contains", "includes", "support", "supports", "supporting", "evidence"},
    "contains_connector": {"connector", "arrow", "line", "flow", "link", "connects", "connection", "contains", "includes"},
    "contains_image": {"image", "picture", "figure", "diagram", "visual", "contains", "includes"},
    "contains_link": {"link", "links", "url", "uri", "source", "reference", "click", "clicked", "action", "actions", "contains", "includes"},
    "links_to": {"link", "links", "url", "uri", "source", "reference", "click", "clicked", "action", "actions", "opens", "target"},
    "contains_comment": {"comment", "annotation", "review", "note", "contains", "includes"},
    "contains_annotation": {"annotation", "comment", "review", "note", "contains", "includes"},
    "contains_form_field": {"form", "field", "input", "value", "contains", "includes"},
    "contains_footnote": {"footnote", "note", "citation", "contains", "includes"},
    "contains_endnote": {"endnote", "note", "citation", "contains", "includes"},
    "contains_outline_item": {"outline", "toc", "section", "heading", "contains", "includes"},
    "contains_revision": {"revision", "change", "inserted", "deleted", "edit", "contains", "includes"},
    "contains_speaker_notes": {"speaker", "notes", "presenter", "contains", "includes"},
    "contains_media": {"media", "audio", "video", "recording", "contains", "includes"},
    "contains_embedded_object": {"embedded", "attachment", "object", "ole", "contains", "includes"},
    "contains_archive_entry": {"archive", "zip", "bundle", "entry", "contains", "includes"},
    "contains_code_block": {"code", "block", "fence", "fenced", "snippet", "language", "python", "javascript", "typescript", "markdown", "contains", "includes", "where", "section"},
    "contains_diagram_node": {"diagram", "smartart", "node", "flow", "hierarchy", "parent", "contains", "includes"},
    "contains_diagram_relationship": {"diagram", "smartart", "relationship", "edge", "connection", "connects", "flow", "contains", "includes"},
    "sent_by": {"email", "message", "sender", "from", "author", "sent", "who", "header"},
    "sent_message": {"email", "message", "sender", "from", "sent", "authored", "body"},
    "sent_to": {"email", "message", "recipient", "recipients", "to", "sent", "who", "header"},
    "cc_to": {"email", "message", "cc", "copy", "recipient", "copied", "header"},
    "bcc_to": {"email", "message", "bcc", "blind", "copy", "recipient", "header"},
    "sent_at": {"email", "message", "date", "time", "sent", "when", "header"},
    "references_message": {"email", "message", "references", "thread", "prior", "conversation"},
    "in_reply_to": {"email", "message", "reply", "thread", "prior", "conversation"},
    "emphasizes": {"emphasis", "emphasized", "styled", "bold", "italic", "underline", "highlight", "important"},
    "gates": {"gate", "gates", "gated", "blocks", "requires", "prerequisite", "approval", "workflow", "launch", "relationship"},
    "groups_visual": {"group", "groups", "grouped", "visual", "visuals", "shape", "shapes", "cluster", "supports", "support", "supporting", "evidence", "source", "contains", "bundles", "organizes"},
    "uses_bullet_structure": {"bullet", "bullets", "hierarchy", "outline", "indent", "levels", "structured"},
    "covers_range": {"range", "covers", "cells", "spreadsheet", "excel", "where", "location"},
    "labeled_by": {"label", "labeled", "merged", "heading", "range", "spreadsheet"},
    "has_merged_label": {"label", "merged", "range", "heading", "spreadsheet"},
    "has_column": {"column", "field", "header", "table", "spreadsheet"},
    "uses_source": {"source", "uses", "from", "table", "range", "pivot", "lineage", "provenance"},
    "groups_by": {"group", "groups", "row", "field", "pivot", "by"},
    "columns_by": {"column", "columns", "field", "pivot", "by"},
    "filters_by": {"filter", "filters", "field", "pivot", "page"},
    "aggregates": {"aggregate", "aggregates", "summary", "value", "measure", "average", "sum", "count", "pivot"},
    "has_formula": {"formula", "calculation", "cell", "spreadsheet", "excel"},
    "uses_formula": {"formula", "expression", "calculation", "uses"},
    "calculates": {"calculate", "calculates", "result", "value", "formula"},
    "depends_on": {"depends", "dependency", "reference", "formula", "source", "input", "upstream", "lineage"},
    "has_conditional_format": {"conditional", "format", "rule", "threshold", "range", "spreadsheet"},
    "applies_to": {"applies", "range", "conditional", "format", "rule"},
    "evaluated_by": {"evaluated", "condition", "conditional", "format", "rule"},
    "uses_condition": {"condition", "conditional", "formula", "threshold", "rule"},
    "has_named_range": {"named", "range", "defined", "name", "spreadsheet", "excel"},
    "has_field": {"field", "form", "validation", "input", "label"},
    "has_value": {"value", "result", "cell", "field", "metric"},
    "visually_near": {"layout", "visual", "near", "nearby", "spatial", "position", "relationship", "relationships", "adjacent"},
    "uses_language": {"code", "language", "python", "javascript", "typescript", "bash", "sql", "mermaid", "uses"},
}


def _query_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _token_variants(tokens: set[str]) -> set[str]:
    variants = set(tokens)
    for token in list(tokens):
        if len(token) > 3 and token.endswith("s"):
            variants.add(token[:-1])
        if len(token) > 4 and token.endswith("ed"):
            variants.add(token[:-2])
        if len(token) > 2 and not token.endswith("s"):
            variants.add(f"{token}s")
    return variants


def _is_structural_relation(relation: str) -> bool:
    relation_key = re.sub(r"[^a-z0-9]+", "_", str(relation or "").lower()).strip("_")
    return relation_key in STRUCTURAL_GRAPH_RELATIONS or relation_key.startswith(("has_", "contains_"))


def _layout_query_score(query: str) -> float:
    query_lower = str(query or "").lower()
    if not query_lower:
        return 0.0
    layout_terms = {
        "where", "located", "location", "find", "found", "which", "slide", "page",
        "section", "heading", "chapter", "sheet", "hidden", "visibility", "visible",
        "workbook", "worksheet", "source", "part", "table", "pivot", "formula",
        "conditional", "range", "column", "aggregate", "field", "chart", "image",
        "figure", "diagram", "shape", "object", "connector", "arrow", "line", "flow", "alt", "description", "deck", "presentation", "spreadsheet", "contains",
        "top", "bottom", "left", "right", "center", "middle", "bbox", "position", "layout", "zone",
        "include", "includes", "under", "email", "message", "subject", "sender", "recipient", "cc", "bcc", "attachment",
    }
    tokens = _query_tokens(query_lower)
    overlap = len(tokens & layout_terms)
    phrase_bonus = 0.25 if any(
        phrase in query_lower
        for phrase in ("which slide", "which page", "what section", "where in", "where is", "where does")
    ) else 0.0
    return min(1.0, (overlap / 3) + phrase_bonus)


def _relation_query_score(relation: str, query: str) -> float:
    if not query:
        return 0.0
    query_tokens = _query_tokens(query)
    relation_key = re.sub(r"[^a-z0-9]+", "_", str(relation or "").lower()).strip("_")
    relation_tokens = _query_tokens(relation_key.replace("_", " "))
    relation_tokens.update(RELATION_QUERY_ALIASES.get(relation_key, set()))
    if not relation_tokens:
        return 0.0
    denominator = max(1, min(len(relation_tokens), 3))
    return min(1.0, len(query_tokens & relation_tokens) / denominator)


GRAPH_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "does", "for", "from", "has", "have",
    "how", "in", "is", "it", "of", "on", "or", "the", "through", "to", "via", "was",
    "what", "where", "which", "who", "why", "with",
}

GRAPH_REASONING_GENERIC_NODE_TERMS = {
    "anchor",
    "anchors",
    "answer",
    "attached",
    "chat",
    "context",
    "chart",
    "charts",
    "deck",
    "document",
    "documents",
    "evidence",
    "file",
    "files",
    "graph",
    "graphrag",
    "import",
    "imports",
    "key",
    "meaningful",
    "path",
    "paths",
    "pipeline",
    "presentation",
    "provenance",
    "rag",
    "recent",
    "relationship",
    "relationships",
    "run",
    "runs",
    "service",
    "services",
    "sheet",
    "sheets",
    "slide",
    "slides",
    "source",
    "sources",
    "summarize",
    "summary",
    "table",
    "tables",
    "template",
    "templates",
    "trace",
}

GRAPH_REASONING_DOCUMENT_NAME_PATTERN = re.compile(
    r"\.(?:csv|docx?|html?|json|md|pdf|pptx?|txt|xlsx?|zip)$",
    flags=re.I,
)


GRAPH_OVERVIEW_QUERY_TERMS = {
    "compare",
    "cross",
    "deck",
    "document",
    "documents",
    "entity",
    "entities",
    "evidence",
    "file",
    "files",
    "find",
    "graph",
    "graphs",
    "local",
    "path",
    "paths",
    "pipeline",
    "provenance",
    "relationship",
    "relationships",
    "run",
    "runs",
    "shared",
    "trace",
    "uploaded",
}


GRAPH_PROVENANCE_METADATA_KEYS = (
    "archive_entry_path",
    "archive_entry_filename",
    "archive_entry_mime",
    "archive_entry_extraction_method",
    "legacy_office_format",
    "legacy_office_stream_index",
    "legacy_office_stream_path",
    "legacy_office_stream_size_bytes",
    "legacy_office_stream_sha256",
    "legacy_office_stream_truncated",
    "legacy_office_extraction_method",
    "legacy_office_string_count",
    "embedded_file_filename",
    "embedded_file_name",
    "embedded_file_mime",
    "embedded_file_extraction_method",
    *EMAIL_GRAPH_METADATA_KEYS,
    "section",
    "section_label",
    "markdown_code_language",
    "markdown_code_block_index",
    "markdown_start_line",
    "markdown_end_line",
    "markdown_frontmatter_index",
    "table_name",
    "table_display_name",
    "table_range",
    "table_part",
    "table_columns",
    "table_cell_reference",
    "table_cell_grid_span",
    "table_cell_row_span",
    "table_cell_h_merge",
    "table_cell_v_merge",
    "table_cell_merge_flags",
    "table_cell_style_name",
    "table_cell_fill_color",
    "table_cell_text_anchor",
    "table_cell_vertical_align",
    "table_cell_text_direction",
    "table_cell_width_value",
    "table_cell_width_inches",
    "table_cell_width_type",
    "table_cell_shading_pattern",
    "table_cell_margin_left_inches",
    "table_cell_margin_right_inches",
    "table_cell_margin_top_inches",
    "table_cell_margin_bottom_inches",
    "table_cell_border_left_color",
    "table_cell_border_right_color",
    "table_cell_border_top_color",
    "table_cell_border_bottom_color",
    "table_cell_border_left_width_inches",
    "table_cell_border_right_width_inches",
    "table_cell_border_top_width_inches",
    "table_cell_border_bottom_width_inches",
    "table_cell_visual_summary",
    "table_cell_column_repeat",
    "table_cell_row_repeat",
    "table_cell_is_covered",
    "cell",
    "cell_value_type",
    "layout_bbox",
    "layout_zone",
    "layout_relation",
    "visual_distance_pct",
    "related_element_type",
    "related_label",
    "related_location",
    "target_layout_zone",
    "target_layout_bbox",
    "layout_x_inches",
    "layout_y_inches",
    "layout_width_inches",
    "layout_height_inches",
    "layout_center_x_inches",
    "layout_center_y_inches",
    "layout_left_pct",
    "layout_top_pct",
    "layout_width_pct",
    "layout_height_pct",
    "layout_center_x_pct",
    "layout_center_y_pct",
    "layout_x_points",
    "layout_y_points",
    "layout_width_points",
    "layout_height_points",
    "layout_center_x_points",
    "layout_center_y_points",
    "page_width_points",
    "page_height_points",
    "page_width_inches",
    "page_height_inches",
    "table_source",
    "row",
    "column",
    "sheet_name",
    "sheet_id",
    "sheet_part",
    "sheet_relationship_id",
    "sheet_state",
    "sheet_visibility",
    "formula_cell",
    "formula_expression",
    "formula_references",
    "formula_result",
    "conditional_range",
    "conditional_range_label",
    "conditional_type",
    "conditional_formula1",
    "conditional_formula_references",
    "pivot_table_name",
    "pivot_table_range",
    "pivot_table_source_name",
    "pivot_table_source_ref",
    "pivot_table_source_sheet",
    "pivot_table_row_fields",
    "pivot_table_column_fields",
    "pivot_table_page_fields",
    "pivot_table_data_fields",
    "pivot_axis",
    "pivot_subtotal",
    "named_range_name",
    "named_range_ref",
    "named_range_scope",
    "form_kind",
    "field_type",
    "label",
    "value",
    "form_field_alias",
    "form_field_tag",
    "form_field_id",
    "form_field_placeholder",
    "form_field_checked",
    "form_field_options",
    "form_field_option_values",
    "form_field_selected_option",
    "form_field_date_format",
    "form_field_date_full_value",
    "form_field_calendar",
    "form_field_lock",
    "control_id",
    "control_name",
    "control_part",
    "validation_range",
    "validation_type",
    "validation_formula1",
    "chart_axis_type",
    "chart_axis_title",
    "chart_axis_number_format",
    "chart_legend_position",
    "chart_label_fields",
    "chart_label_position",
    "chart_label_number_format",
    "chart_series_fill_color",
    "chart_series_line_color",
    "chart_series_line_dash",
    "chart_series_visual_summary",
    "chart_marker_symbol",
    "chart_marker_fill_color",
    "chart_marker_line_color",
    "chart_point_fill_color",
    "chart_point_visual_summary",
    "chart_trendline_type",
    "chart_trendline_line_color",
    "chart_trendline_line_dash",
    "chart_trendline_visual_summary",
    "style_summary",
    "styled_text",
    "emphasis_term",
    "emphasis_terms",
    "emphasis_styles",
    "bullet_levels",
    "media_part",
    "media_kind",
    "media_mime",
    "media_size_bytes",
    "comment_id",
    "comment_kind",
    "comment_part",
    "comment_anchor_text",
    "comment_anchor_section",
    "comment_anchor_paragraph",
    "comment_anchor_sheet",
    "comment_anchor_cell",
    "comment_anchor_header",
    "comment_anchor_value",
    "comment_datetime",
    "threaded_comment_id",
    "threaded_parent_id",
    "person_id",
    "author",
    "author_id",
    "initials",
    "date",
    "revision_type",
    "revision_author",
    "revision_date",
    "revision_id",
    "annotation_type",
    "annotation_author",
    "annotation_subject",
    "annotation_content",
    "annotation_name",
    "slide_title",
    "slide_id",
    "slide_part",
    "slide_relationship_id",
    "slide_state",
    "slide_visibility",
    "notes_part",
    "notes_relationship_id",
    "shape_id",
    "shape_index",
    "shape_name",
    "shape_title",
    "shape_description",
    "connector_id",
    "connector_index",
    "connector_name",
    "connector_title",
    "connector_description",
    "source_shape_id",
    "target_shape_id",
    "diagram_part",
    "diagram_node_id",
    "diagram_node_type",
    "diagram_order",
    "diagram_level",
    "parent_diagram_node_id",
    "parent_label",
    "diagram_connection_type",
    "source_diagram_node_id",
    "target_diagram_node_id",
    "source_label",
    "target_label",
    "local_graph_run_id",
    "local_graph_filename",
    "local_graph_domain",
    "local_graph_status",
    "section_label",
    "element_type",
    "location",
    "ocr_source",
    "ocr_engine",
    "page",
    "slide",
    "sheet",
    "table",
    "row",
    "column",
)


CHAT_CONTEXT_COMPACT_RESPONSE_MODES = {"chat", "compact", "interactive"}

CHAT_CONTEXT_METADATA_KEEP_KEYS = (
    "run_id",
    "domain",
    "filename",
    "document_id",
    "document_title",
    "artifact_id",
    "artifact_type",
    "artifact_filename",
    "artifact_endpoint",
    "manifest_endpoint",
    "source_artifact_endpoint",
    "source_type",
    "source_stage",
    "content_type",
    "mime_type",
    "updated_at",
    "created_at",
    "primary_evidence_family",
    "evidence_families",
    "location_hints",
    "source_locations",
    "matched_query_terms",
    "specific_matched_query_terms",
    "matched_entities",
    "matched_graph_path_terms",
    "matched_local_chunk_evidence_terms",
    "matched_local_graph_terms",
    "matched_requested_evidence_families",
    "retrieval_reasons",
    "semantic_text",
    "chart_title",
    "chart_label",
    "series_name",
    "chart_point_label",
    "category",
    "image_part",
    "image_mime",
    "image_alt_text",
    "image_description",
    "media_kind",
    "media_mime",
    "media_part",
    "graph_context_relationship_count",
    "graph_path_count",
    "local_graph_node_count",
    "local_graph_edge_count",
    *GRAPH_PROVENANCE_METADATA_KEYS,
)

CHAT_CONTEXT_HEAVY_KEYS = {
    "base64",
    "bytes",
    "content_base64",
    "document_text",
    "full_text",
    "raw",
    "raw_content",
    "raw_text",
    "raw_xml",
    "source_text",
    "text_content",
    "xml",
}


def _bool_from_request(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _compact_chat_context_requested(data: dict) -> bool:
    response_mode = str(data.get("response_mode") or data.get("mode") or "").strip().lower()
    return response_mode in CHAT_CONTEXT_COMPACT_RESPONSE_MODES or _bool_from_request(
        data.get("compact_response")
    )


def _bounded_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _compact_chat_context_string(value, limit: int = 320):
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    compacted = _compact_graph_evidence(text, limit=limit)
    return compacted if compacted is not None else text[: max(limit - 1, 0)].rstrip() + "…"


def _compact_chat_context_list(values, *, item_limit: int = 8, string_limit: int = 280, depth: int = 0) -> list:
    if not isinstance(values, list):
        return []
    return [
        _compact_chat_context_value(item, string_limit=string_limit, item_limit=item_limit, depth=depth + 1)
        for item in values[:item_limit]
    ]


def _compact_chat_context_value(value, *, string_limit: int = 280, item_limit: int = 8, depth: int = 0):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _compact_chat_context_string(value, string_limit)
    if isinstance(value, list):
        if depth >= 3:
            return [
                _compact_chat_context_string(item, string_limit)
                if isinstance(item, str)
                else item
                for item in value[:item_limit]
                if item is None or isinstance(item, (bool, int, float, str))
            ]
        return _compact_chat_context_list(
            value,
            item_limit=item_limit,
            string_limit=string_limit,
            depth=depth,
        )
    if isinstance(value, dict):
        if depth >= 3:
            return {
                str(key): _compact_chat_context_value(
                    item,
                    string_limit=string_limit,
                    item_limit=item_limit,
                    depth=depth + 1,
                )
                for key, item in list(value.items())[:item_limit]
                if str(key) not in CHAT_CONTEXT_HEAVY_KEYS
            }
        return {
            str(key): _compact_chat_context_value(
                item,
                string_limit=string_limit,
                item_limit=item_limit,
                depth=depth + 1,
            )
            for key, item in value.items()
            if str(key) not in CHAT_CONTEXT_HEAVY_KEYS
        }
    return _compact_chat_context_string(value, string_limit)


def _compact_chat_context_metadata(metadata: dict | None, *, string_limit: int = 320, item_limit: int = 10) -> dict:
    if not isinstance(metadata, dict):
        return {}
    compacted = {}
    for key in CHAT_CONTEXT_METADATA_KEEP_KEYS:
        if key in metadata and metadata.get(key) not in (None, "", [], {}):
            compacted[key] = _compact_chat_context_value(
                metadata.get(key),
                string_limit=string_limit,
                item_limit=item_limit,
            )
    return compacted


def _compact_chat_context_provenance(value):
    if isinstance(value, dict):
        return _compact_chat_context_metadata(value, string_limit=360, item_limit=10)
    if isinstance(value, list):
        return [
            _compact_chat_context_provenance(item)
            if isinstance(item, dict)
            else _compact_chat_context_value(item, string_limit=260, item_limit=6)
            for item in value[:8]
        ]
    return _compact_chat_context_value(value, string_limit=320, item_limit=6)


def _compact_chat_context_relationship(relationship: dict | None, *, metadata_limit: int = 320) -> dict:
    if not isinstance(relationship, dict):
        return {}
    keep_keys = (
        "id",
        "source",
        "target",
        "relation",
        "label",
        "type",
        "relationship_type",
        "scope_label",
        "citation_label",
        "score",
        "weight",
        "confidence",
        "chunk_match_score",
        "query_relevance_score",
        "chunk_match_reasons",
        "matched_query_terms",
        "specific_matched_query_terms",
        "evidence_families",
        "categories",
        "location_hints",
        "source_locations",
        "location",
        "provenance_summary",
        "provenance",
        "metadata",
        "evidence",
        "rationale",
        "answer_hint",
        "text",
    )
    compacted = {}
    for key in keep_keys:
        if key not in relationship or relationship.get(key) in (None, "", [], {}):
            continue
        value = relationship.get(key)
        if key == "metadata":
            compacted[key] = _compact_chat_context_metadata(
                value if isinstance(value, dict) else {},
                string_limit=metadata_limit,
                item_limit=10,
            )
        elif key in {"provenance", "source_locations", "location", "location_hints"}:
            compacted[key] = _compact_chat_context_provenance(value)
        elif key in {"evidence", "rationale", "answer_hint", "text"}:
            compacted[key] = _compact_chat_context_string(value, 520)
        else:
            compacted[key] = _compact_chat_context_value(value, string_limit=280, item_limit=10)
    return compacted


def _compact_chat_context_path(path: dict | None, *, relationship_limit: int = 4) -> dict:
    if not isinstance(path, dict):
        return {}
    keep_keys = (
        "id",
        "source",
        "target",
        "bridge",
        "nodes",
        "hops",
        "score",
        "path_type",
        "evidence_families",
        "matched_query_terms",
        "text",
        "explanation",
        "answer_hint",
        "provenance_summary",
        "relationships",
    )
    compacted = {}
    for key in keep_keys:
        if key not in path or path.get(key) in (None, "", [], {}):
            continue
        value = path.get(key)
        if key == "relationships" and isinstance(value, list):
            compacted[key] = [
                _compact_chat_context_relationship(item)
                for item in value[:relationship_limit]
                if isinstance(item, dict)
            ]
        elif key in {"text", "explanation", "answer_hint"}:
            compacted[key] = _compact_chat_context_string(value, 620)
        else:
            compacted[key] = _compact_chat_context_value(value, string_limit=280, item_limit=8)
    return compacted


def _compact_chat_context_evidence_card(card: dict | None, *, metadata_limit: int = 320) -> dict:
    if not isinstance(card, dict):
        return {}
    keep_keys = (
        "id",
        "index",
        "role",
        "kind",
        "title",
        "text",
        "evidence",
        "reason",
        "scope_label",
        "citation_label",
        "source",
        "target",
        "relation",
        "relationship_type",
        "provenance",
        "provenance_summary",
        "evidence_families",
        "categories",
        "location_hints",
        "location",
        "source_locations",
        "metadata",
        "score",
    )
    compacted = {}
    for key in keep_keys:
        if key not in card or card.get(key) in (None, "", [], {}):
            continue
        value = card.get(key)
        if key == "metadata":
            compacted[key] = _compact_chat_context_metadata(
                value if isinstance(value, dict) else {},
                string_limit=metadata_limit,
                item_limit=10,
            )
        elif key in {"text", "evidence", "reason"}:
            compacted[key] = _compact_chat_context_string(value, 620)
        elif key in {"provenance", "location", "source_locations", "location_hints"}:
            compacted[key] = _compact_chat_context_provenance(value)
        else:
            compacted[key] = _compact_chat_context_value(value, string_limit=280, item_limit=10)
    return compacted


def _compact_chat_context_handoff(handoff: dict | None, *, evidence_card_limit: int = 4) -> dict:
    if not isinstance(handoff, dict):
        return {}
    compacted = {}
    for key in (
        "run_id",
        "filename",
        "domain",
        "status",
        "recommended_mode",
        "answer_guidance",
        "citation_scopes",
        "counts",
        "evidence_totals",
    ):
        if key in handoff and handoff.get(key) not in (None, "", [], {}):
            compacted[key] = _compact_chat_context_value(handoff.get(key), string_limit=360, item_limit=10)

    capability_cards = handoff.get("capability_cards")
    if isinstance(capability_cards, list):
        compacted["capability_cards"] = [
            _compact_chat_context_value(card, string_limit=220, item_limit=8, depth=1)
            for card in capability_cards[:8]
            if isinstance(card, dict)
        ]

    graph_anchors = handoff.get("graph_anchors")
    if isinstance(graph_anchors, list):
        compacted["graph_anchors"] = [
            _compact_chat_context_relationship(anchor)
            for anchor in graph_anchors[:8]
            if isinstance(anchor, dict)
        ]

    evidence_cards = handoff.get("evidence_cards")
    if isinstance(evidence_cards, list):
        compacted["evidence_cards"] = [
            _compact_chat_context_evidence_card(card)
            for card in evidence_cards[:evidence_card_limit]
            if isinstance(card, dict)
        ]

    starter_prompts = handoff.get("starter_prompts")
    if isinstance(starter_prompts, list):
        compacted["starter_prompts"] = [
            _compact_chat_context_value(prompt, string_limit=360, item_limit=6)
            for prompt in starter_prompts[:4]
        ]

    rehearsals = handoff.get("retrieval_rehearsals")
    if isinstance(rehearsals, list):
        compacted_rehearsals = []
        for rehearsal in rehearsals[:3]:
            if not isinstance(rehearsal, dict):
                continue
            compacted_rehearsal = {}
            for key in ("id", "label", "prompt", "query", "intent", "score", "matched_terms"):
                if key in rehearsal and rehearsal.get(key) not in (None, "", [], {}):
                    compacted_rehearsal[key] = _compact_chat_context_value(
                        rehearsal.get(key),
                        string_limit=360,
                        item_limit=8,
                    )
            for list_key in ("top_anchors", "top_chunks", "matched_anchors", "matched_chunks"):
                values = rehearsal.get(list_key)
                if isinstance(values, list):
                    compacted_rehearsal[list_key] = [
                        _compact_chat_context_value(item, string_limit=260, item_limit=6, depth=1)
                        for item in values[:2]
                    ]
            compacted_rehearsals.append(compacted_rehearsal)
        if compacted_rehearsals:
            compacted["retrieval_rehearsals"] = compacted_rehearsals

    return compacted


def _compact_chat_context_graph_overview(overview: dict | None) -> dict:
    if not isinstance(overview, dict):
        return {}
    compacted = {}
    for key in (
        "source_count",
        "relationship_count",
        "entity_count",
        "shared_entity_count",
        "bridge_entity_count",
        "cross_document_path_count",
    ):
        if key in overview:
            compacted[key] = overview.get(key)
    list_limits = {
        "top_relationships": 10,
        "shared_entities": 8,
        "bridge_entities": 8,
        "cross_document_paths": 6,
    }
    for key, limit in list_limits.items():
        values = overview.get(key)
        if not isinstance(values, list):
            continue
        if key == "top_relationships":
            compacted[key] = [
                _compact_chat_context_relationship(value)
                for value in values[:limit]
                if isinstance(value, dict)
            ]
        elif key == "cross_document_paths":
            compacted[key] = [
                _compact_chat_context_path(value)
                for value in values[:limit]
                if isinstance(value, dict)
            ]
        else:
            compacted[key] = [
                _compact_chat_context_value(value, string_limit=260, item_limit=8, depth=1)
                for value in values[:limit]
            ]
    return compacted


def _compact_chat_context_answer_plan(answer_plan: dict | None, *, evidence_card_limit: int = 6) -> dict:
    if not isinstance(answer_plan, dict):
        return {}
    compacted = {}
    for key in (
        "strategy",
        "summary",
        "answer_guidance",
        "recommended_mode",
        "primary_path",
        "primary_relationship",
        "provenance_anchors",
    ):
        if key not in answer_plan or answer_plan.get(key) in (None, "", [], {}):
            continue
        value = answer_plan.get(key)
        if key == "primary_path":
            compacted[key] = _compact_chat_context_path(value if isinstance(value, dict) else {})
        elif key == "primary_relationship":
            compacted[key] = _compact_chat_context_relationship(value if isinstance(value, dict) else {})
        elif key == "provenance_anchors" and isinstance(value, list):
            compacted[key] = [
                _compact_chat_context_provenance(item)
                for item in value[:8]
            ]
        else:
            compacted[key] = _compact_chat_context_value(value, string_limit=520, item_limit=8)

    evidence_cards = answer_plan.get("evidence_cards")
    if isinstance(evidence_cards, list):
        requested_cards = [
            card
            for card in evidence_cards
            if isinstance(card, dict) and card.get("role") == "requested_evidence_family"
        ]
        other_cards = [
            card
            for card in evidence_cards
            if isinstance(card, dict) and card.get("role") != "requested_evidence_family"
        ]
        requested_card_limit = min(12, max(evidence_card_limit, len(requested_cards)))
        selected_cards = [
            *requested_cards[:requested_card_limit],
            *other_cards[:max(0, evidence_card_limit - min(len(requested_cards), requested_card_limit))],
        ]
        compacted["evidence_cards"] = [
            _compact_chat_context_evidence_card(card)
            for card in selected_cards
            if isinstance(card, dict)
        ]
    return compacted


def _compact_chat_context_graph_reasoning(
    graph_reasoning: dict | None,
    *,
    path_limit: int = 8,
    relationship_limit: int = 16,
    evidence_card_limit: int = 6,
) -> dict:
    if not isinstance(graph_reasoning, dict):
        return {}
    compacted = {}
    for key in (
        "status",
        "reasoning_needed",
        "query",
        "intent_hints",
        "entity_names",
        "entity_pairs",
        "inferred_entity_pairs",
        "graph_inferred_entity_pairs",
        "query_complexity",
        "summary",
        "pipeline_evidence",
        "graph_sources",
    ):
        if key in graph_reasoning and graph_reasoning.get(key) not in (None, "", [], {}):
            compacted[key] = _compact_chat_context_value(
                graph_reasoning.get(key),
                string_limit=520 if key == "summary" else 280,
                item_limit=12,
            )

    handoffs = graph_reasoning.get("interaction_handoffs")
    if isinstance(handoffs, list):
        compacted["interaction_handoffs"] = [
            _compact_chat_context_handoff(handoff, evidence_card_limit=evidence_card_limit)
            for handoff in handoffs[:4]
            if isinstance(handoff, dict)
        ]

    answer_plan = _compact_chat_context_answer_plan(
        graph_reasoning.get("answer_plan") if isinstance(graph_reasoning.get("answer_plan"), dict) else {},
        evidence_card_limit=evidence_card_limit,
    )
    if answer_plan:
        compacted["answer_plan"] = answer_plan

    paths = graph_reasoning.get("paths")
    if isinstance(paths, list):
        compacted["paths"] = [
            _compact_chat_context_path(path)
            for path in paths[:path_limit]
            if isinstance(path, dict)
        ]

    bridge_entities = graph_reasoning.get("bridge_entities")
    if isinstance(bridge_entities, list):
        compacted["bridge_entities"] = [
            _compact_chat_context_value(bridge, string_limit=360, item_limit=8, depth=1)
            for bridge in bridge_entities[:8]
            if isinstance(bridge, dict)
        ]

    direct_relationships = graph_reasoning.get("direct_relationships")
    if isinstance(direct_relationships, list):
        compacted["direct_relationships"] = [
            _compact_chat_context_relationship(relationship)
            for relationship in direct_relationships[:relationship_limit]
            if isinstance(relationship, dict)
        ]

    graph_overview = _compact_chat_context_graph_overview(
        graph_reasoning.get("graph_overview") if isinstance(graph_reasoning.get("graph_overview"), dict) else {}
    )
    if graph_overview:
        compacted["graph_overview"] = graph_overview

    return compacted


def _compact_chat_context_chunk(
    chunk: dict | None,
    *,
    content_limit: int,
    graph_context_limit: int,
    graph_path_limit: int,
) -> dict:
    if not isinstance(chunk, dict):
        return {}
    compacted = {}
    keep_keys = (
        "id",
        "document_id",
        "document_title",
        "chunk_index",
        "score",
        "source",
        "retrieval_reason",
        "retrieval_reasons",
        "retrieval_queries",
        "matched_entities",
        "matched_graph_path_terms",
        "graph_reasoning_boost",
        "graph_path_match_count",
        "graph_context_match_count",
        "content",
        "metadata",
        "graph_context",
        "graph_paths",
    )
    for key in keep_keys:
        if key not in chunk or chunk.get(key) in (None, "", [], {}):
            continue
        value = chunk.get(key)
        if key == "content":
            compacted[key] = _compact_chat_context_string(value, content_limit)
        elif key == "metadata":
            compacted[key] = _compact_chat_context_metadata(
                value if isinstance(value, dict) else {},
                string_limit=360,
                item_limit=10,
            )
        elif key == "graph_context" and isinstance(value, list):
            compacted[key] = [
                _compact_chat_context_relationship(item)
                for item in value[:graph_context_limit]
                if isinstance(item, dict)
            ]
        elif key == "graph_paths" and isinstance(value, list):
            compacted[key] = [
                _compact_chat_context_path(item)
                for item in value[:graph_path_limit]
                if isinstance(item, dict)
            ]
        else:
            compacted[key] = _compact_chat_context_value(value, string_limit=280, item_limit=10)
    return compacted


def _compact_chat_context_artifact(artifact: dict | None) -> dict:
    if not isinstance(artifact, dict):
        return {}
    keep_keys = (
        "id",
        "kind",
        "type",
        "stage",
        "filename",
        "name",
        "label",
        "endpoint",
        "download_url",
        "path",
        "mime_type",
        "content_type",
        "checksum",
        "size_bytes",
        "extraction_method",
        "parsed_child_elements",
        "created_at",
        "updated_at",
        "summary",
        "description",
        "sample",
        "sample_outputs",
        "source_locations",
        "metadata",
    )
    compacted = {}
    for key in keep_keys:
        if key not in artifact or artifact.get(key) in (None, "", [], {}):
            continue
        value = artifact.get(key)
        if key == "metadata":
            compacted[key] = _compact_chat_context_metadata(value if isinstance(value, dict) else {})
        elif key in {"summary", "description", "sample", "sample_outputs"}:
            compacted[key] = _compact_chat_context_value(value, string_limit=520, item_limit=6)
        elif key == "source_locations":
            compacted[key] = _compact_chat_context_provenance(value)
        else:
            compacted[key] = _compact_chat_context_value(value, string_limit=260, item_limit=8)
    return compacted


def _compact_chat_context_pipeline_artifacts(
    pipeline_artifacts: dict | None,
    *,
    evidence_card_limit: int = 4,
) -> dict:
    if not isinstance(pipeline_artifacts, dict):
        return {"runs": [], "run_count": 0, "artifact_count": 0}
    compacted = {}
    for key, value in pipeline_artifacts.items():
        if key == "runs" or value in (None, "", [], {}):
            continue
        compacted[key] = _compact_chat_context_value(value, string_limit=320, item_limit=10)

    compacted_runs = []
    for run in pipeline_artifacts.get("runs", []) or []:
        if not isinstance(run, dict):
            continue
        compacted_run = {}
        for key in (
            "run_id",
            "filename",
            "domain",
            "status",
            "updated_at",
            "artifact_count",
            "manifest_endpoint",
            "summary",
            "document",
            "structured_ingest",
            "local_graph",
            "local_context",
            "local_graph_reasoning",
        ):
            if key not in run or run.get(key) in (None, "", [], {}):
                continue
            if key in {"structured_ingest", "local_graph", "local_context", "local_graph_reasoning"}:
                compacted_run[key] = _compact_chat_context_value(
                    run.get(key),
                    string_limit=520,
                    item_limit=8,
                    depth=1,
                )
            else:
                compacted_run[key] = _compact_chat_context_value(
                    run.get(key),
                    string_limit=360,
                    item_limit=10,
                )

        artifacts = run.get("artifacts")
        if isinstance(artifacts, list):
            compacted_run["artifacts"] = [
                _compact_chat_context_artifact(artifact)
                for artifact in artifacts[:12]
                if isinstance(artifact, dict)
            ]

        source_artifacts = run.get("source_artifacts")
        if isinstance(source_artifacts, list):
            compacted_run["source_artifacts"] = [
                _compact_chat_context_artifact(artifact)
                for artifact in source_artifacts[:8]
                if isinstance(artifact, dict)
            ]

        evidence_cards = run.get("evidence_cards")
        if isinstance(evidence_cards, list):
            compacted_run["evidence_cards"] = [
                _compact_chat_context_evidence_card(card)
                for card in evidence_cards[:evidence_card_limit]
                if isinstance(card, dict)
            ]

        compacted_runs.append(compacted_run)
    compacted["runs"] = compacted_runs
    compacted["run_count"] = int(pipeline_artifacts.get("run_count", len(compacted_runs)) or len(compacted_runs))
    return compacted


def _compact_chat_context_cross_references(cross_references, *, item_limit: int = 12) -> list:
    if not isinstance(cross_references, list):
        return []
    compacted = []
    for ref in cross_references[:item_limit]:
        if not isinstance(ref, dict):
            continue
        compacted.append(_compact_chat_context_value(ref, string_limit=260, item_limit=8, depth=1))
    return compacted


def _compact_context_text_for_response(context_text: str, *, max_chars: int = 24000) -> str:
    if not context_text or len(context_text) <= max_chars:
        return context_text

    section_names = (
        "=== Requested Evidence Coverage ===",
        "=== Graph Answer Plan ===",
        "=== Relevant Graph Relationships ===",
        "=== Graph Reasoning Summary ===",
        "=== Graph Paths ===",
        "=== Bridge Entities ===",
        "=== Local Pipeline Graph Overview ===",
        "=== Pipeline Artifact Context ===",
        "=== Knowledge Context ===",
    )
    sections = re.split(r"\n(?==== )", context_text)
    prioritized = []
    for name in section_names:
        for section in sections:
            if name in section and section not in prioritized:
                prioritized.append(section)
                break
    if not prioritized and sections:
        prioritized = sections[:3]

    suffix = (
        f"\n\n[GraphRAG chat context compacted from {len(context_text)} characters. "
        "Full source artifacts remain available through the cited document references.]"
    )
    budget = max(1000, max_chars - len(suffix))
    output_parts = []
    used = 0
    for section in prioritized:
        remaining = budget - used
        if remaining <= 0:
            break
        if len(section) > remaining:
            section = section[:remaining].rstrip()
        output_parts.append(section)
        used += len(section)
    compacted = "\n".join(output_parts).rstrip()
    if len(compacted) > budget:
        compacted = compacted[:budget].rstrip()
    return compacted + suffix


def _compact_chat_context_response(response: dict, data: dict) -> dict:
    chunk_limit = _bounded_int(data.get("response_chunk_content_chars"), 700, 360, 6000)
    graph_context_limit = _bounded_int(data.get("response_graph_context_limit"), 2, 1, 12)
    graph_path_limit = _bounded_int(data.get("response_graph_path_limit"), 4, 1, 8)
    evidence_card_limit = _bounded_int(data.get("response_evidence_card_limit"), 3, 1, 12)
    path_limit = _bounded_int(data.get("response_path_limit"), 4, 1, 16)
    relationship_limit = _bounded_int(data.get("response_relationship_limit"), 8, 4, 32)
    context_text_limit = _bounded_int(data.get("context_text_max_chars"), 12000, 4000, 60000)

    compacted = dict(response)
    compacted["chunks"] = [
        _compact_chat_context_chunk(
            chunk,
            content_limit=chunk_limit,
            graph_context_limit=graph_context_limit,
            graph_path_limit=graph_path_limit,
        )
        for chunk in response.get("chunks", []) or []
        if isinstance(chunk, dict)
    ]
    compacted["graph_reasoning"] = _compact_chat_context_graph_reasoning(
        response.get("graph_reasoning") if isinstance(response.get("graph_reasoning"), dict) else {},
        path_limit=path_limit,
        relationship_limit=relationship_limit,
        evidence_card_limit=evidence_card_limit,
    )
    compacted["pipeline_artifacts"] = _compact_chat_context_pipeline_artifacts(
        response.get("pipeline_artifacts") if isinstance(response.get("pipeline_artifacts"), dict) else {},
        evidence_card_limit=evidence_card_limit,
    )
    if isinstance(response.get("requested_evidence_cards"), list):
        compacted["requested_evidence_cards"] = [
            _compact_chat_context_evidence_card(card)
            for card in response.get("requested_evidence_cards", [])[:12]
            if isinstance(card, dict)
        ]
    compacted["cross_references"] = _compact_chat_context_cross_references(response.get("cross_references"))
    compacted["context_text"] = _compact_context_text_for_response(
        str(response.get("context_text") or ""),
        max_chars=context_text_limit,
    )

    quality_signals = dict(response.get("quality_signals") or {})
    quality_signals.update({
        "response_mode": "chat",
        "response_compacted": True,
        "response_context_text_chars": len(compacted.get("context_text") or ""),
        "response_chunk_content_chars": chunk_limit,
        "response_graph_context_limit": graph_context_limit,
        "response_graph_path_limit": graph_path_limit,
    })
    compacted["quality_signals"] = quality_signals
    return compacted


def _metadata_query_tokens(metadata: dict | None) -> set[str]:
    if not isinstance(metadata, dict):
        return set()

    values = [_relationship_provenance(metadata)]
    for key in GRAPH_PROVENANCE_METADATA_KEYS:
        value = metadata.get(key)
        if value is not None and value != "":
            values.append(str(value))

    tokens = _query_tokens(" ".join(values))
    return {token for token in tokens if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2}


def _edge_query_terms(edge: dict) -> set[str]:
    relation = str(edge.get("relation") or "")
    relation_key = re.sub(r"[^a-z0-9]+", "_", relation.lower()).strip("_")
    terms = _query_tokens(relation_key.replace("_", " "))
    terms.update(RELATION_QUERY_ALIASES.get(relation_key, set()))
    terms.update(_metadata_query_tokens(edge.get("metadata")))
    for key in ("source", "target", "evidence", "provenance"):
        value = edge.get(key)
        if value:
            terms.update(_query_tokens(str(value)))
    return {term for term in terms if term not in GRAPH_QUERY_STOPWORDS and len(term) >= 2}


def _edge_query_score(edge: dict, query: str) -> float:
    if not query:
        return 0.0
    query_tokens = _token_variants({
        token for token in _query_tokens(query)
        if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
    })
    edge_terms = _edge_query_terms(edge)
    if not query_tokens or not edge_terms:
        return 0.0
    denominator = max(1, min(len(query_tokens), 4))
    return min(1.0, len(query_tokens & edge_terms) / denominator)


def _query_relevant_graph_relationships(
    raw_edges: list,
    node_map: dict,
    query: str,
    *,
    limit: int = 12,
    normalized_edges: list[dict] | None = None,
) -> list[dict]:
    query_tokens = _token_variants({
        token for token in _query_tokens(query)
        if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
    })
    if not query_tokens:
        return []

    relationships: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    graph_edges = normalized_edges if normalized_edges is not None else _normalize_graph_edges(raw_edges, node_map)
    for normalized_edge in graph_edges:
        normalized = dict(normalized_edge)
        edge_terms = _edge_query_terms(normalized)
        matched_terms = sorted(query_tokens & edge_terms)
        if not matched_terms:
            continue
        specific_matched_terms = [
            term for term in matched_terms
            if term not in GRAPH_OVERVIEW_QUERY_TERMS
        ]
        if not specific_matched_terms:
            continue
        key = (
            str(normalized.get("source") or "").lower(),
            str(normalized.get("relation") or "").lower(),
            str(normalized.get("target") or "").lower(),
            str(normalized.get("evidence") or "")[:160].lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        relation_score = _edge_query_score(normalized, query)
        normalized["score"] = round(
            (relation_score * 0.5)
            + (_safe_weight(normalized.get("weight", 1)) * 0.28)
            + (0.16 if not _is_structural_relation(normalized.get("relation", "")) else 0.08),
            3,
        )
        normalized["relationship_type"] = "layout" if _is_structural_relation(normalized.get("relation", "")) else "semantic"
        normalized["matched_query_terms"] = matched_terms[:10]
        normalized["specific_matched_query_terms"] = specific_matched_terms[:10]
        relationships.append(normalized)

    return sorted(relationships, key=_graph_relationship_rank_key, reverse=True)[:limit]


def _graph_relationship_key(relationship: dict) -> tuple[str, str, str, str, str]:
    return (
        str(relationship.get("source") or "").lower(),
        str(relationship.get("relation") or "").lower(),
        str(relationship.get("target") or "").lower(),
        str(relationship.get("provenance") or _relationship_provenance(relationship.get("metadata")) or "").lower(),
        str(relationship.get("evidence") or "")[:160].lower(),
    )


def _merge_graph_relationships(*relationship_groups: list[dict], limit: int = 12) -> list[dict]:
    merged: dict[tuple[str, str, str, str, str], dict] = {}
    for relationships in relationship_groups:
        for relationship in relationships or []:
            if not isinstance(relationship, dict):
                continue
            key = _graph_relationship_key(relationship)
            existing = merged.get(key)
            if existing is None or relationship.get("score", 0) > existing.get("score", 0):
                merged[key] = relationship

    return sorted(merged.values(), key=_graph_relationship_rank_key, reverse=True)[:limit]


def _query_matched_relationships_for_context(relationships: list[dict], *, limit: int = 12) -> list[dict]:
    candidates: list[dict] = []
    for relationship in relationships or []:
        if not isinstance(relationship, dict):
            continue
        specific_terms = relationship.get("specific_matched_query_terms")
        matched_terms = relationship.get("matched_query_terms")
        term_count = len(specific_terms if isinstance(specific_terms, list) else matched_terms if isinstance(matched_terms, list) else [])
        if term_count >= 2:
            candidates.append(relationship)
    return sorted(candidates, key=_graph_relationship_rank_key, reverse=True)[:limit]


DOCUMENT_NODE_PATTERN = re.compile(
    r"(?:^|[/\\])[^/\\\s]+\.(?:pdf|pptx?|docx?|xlsx?|csv|tsv|md|txt|rtf|html?|odt|ods|odp|eml|msg|zip|json|xml)$",
    flags=re.I,
)
INTERNAL_OFFICE_ARTIFACT_PATTERN = re.compile(
    r"(?:^|/)(?:ppt|xl|word|docprops)/|(?:^|/)_rels/|/embeddings/|/charts/|chartworkbook|\.rels\b|\.xml\b",
    flags=re.I,
)
EXTERNAL_URL_PATTERN = re.compile(r"^(?:https?|ftp)://|^mailto:|^www\.", flags=re.I)
CHART_ANNOTATION_TERMS = {"axis", "axes", "legend", "trendline", "label", "labels", "data_label", "data_labels"}
CHART_DETAIL_QUERY_TERMS = {"axis", "axes", "legend", "trendline", "label", "labels", "chart", "series", "plot"}
LINK_DETAIL_QUERY_TERMS = {"action", "clicked", "click", "href", "hyperlink", "link", "links", "uri", "url", "website"}
ARTIFACT_DETAIL_QUERY_TERMS = {
    "archive",
    "attachment",
    "embedded",
    "file",
    "internal",
    "ole",
    "package",
    "part",
    "path",
    "source",
    "stream",
    "workbook",
    "xml",
}
DOCUMENT_DETAIL_QUERY_TERMS = {"deck", "document", "documents", "file", "files", "page", "slide", "source", "where"}
RELATIONSHIP_FOCUS_QUERY_TERMS = {
    "edge",
    "edges",
    "important",
    "key",
    "main",
    "primary",
    "relation",
    "relations",
    "relationship",
    "relationships",
    "strong",
    "strongest",
    "top",
}
PATH_FOCUS_QUERY_TERMS = {
    "between",
    "bridge",
    "bridges",
    "connect",
    "connected",
    "connection",
    "connections",
    "hop",
    "hops",
    "path",
    "paths",
    "through",
    "trace",
    "via",
}


def _relationship_endpoint_text(relationship: dict) -> str:
    return " ".join(
        str(relationship.get(key) or "")
        for key in ("source", "target")
        if relationship.get(key) not in (None, "")
    )


def _relationship_requested_terms(relationship: dict) -> set[str]:
    values: list[str] = []
    for key in ("specific_matched_query_terms", "matched_query_terms"):
        terms = relationship.get(key)
        if isinstance(terms, list):
            values.extend(str(term) for term in terms if term)
    return {
        token
        for token in _query_tokens(" ".join(values))
        if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
    }


def _relationship_has_document_endpoint(relationship: dict) -> bool:
    endpoint_text = _relationship_endpoint_text(relationship)
    return bool(DOCUMENT_NODE_PATTERN.search(endpoint_text))


def _relationship_has_internal_artifact_endpoint(relationship: dict) -> bool:
    endpoint_text = _relationship_endpoint_text(relationship).replace("\\", "/")
    return bool(INTERNAL_OFFICE_ARTIFACT_PATTERN.search(endpoint_text))


def _relationship_has_external_url_endpoint(relationship: dict) -> bool:
    return any(
        EXTERNAL_URL_PATTERN.search(str(relationship.get(key) or "").strip())
        for key in ("source", "target")
    )


def _relationship_is_chart_annotation(relationship: dict) -> bool:
    relation_key = _graph_context_relation_key(relationship)
    metadata = relationship.get("metadata") if isinstance(relationship.get("metadata"), dict) else {}
    element_type = str(metadata.get("element_type") or "").lower()
    if element_type in {"chart_axis", "chart_data_label", "chart_legend", "chart_trendline"}:
        return True
    if relation_key in {"has_axis", "has_data_labels", "has_legend", "has_trendline"}:
        return True
    endpoint_tokens = _query_tokens(_relationship_endpoint_text(relationship).replace("_", " "))
    return bool(endpoint_tokens & CHART_ANNOTATION_TERMS)


def _relationship_is_container_detail(relationship: dict) -> bool:
    relation_key = _graph_context_relation_key(relationship)
    metadata = relationship.get("metadata") if isinstance(relationship.get("metadata"), dict) else {}
    element_type = str(metadata.get("element_type") or "").lower()
    if relation_key.startswith("contains_"):
        return True
    if _relationship_has_document_endpoint(relationship) and relation_key.startswith(("has_", "uses_")):
        return True
    return element_type in {"slide_metadata", "sheet_metadata", "document_metadata"}


def _query_prefers_direct_relationships(query: str) -> bool:
    tokens = _query_tokens(query)
    if not tokens:
        return False
    relationship_focus = bool(tokens & RELATIONSHIP_FOCUS_QUERY_TERMS)
    path_focus = bool(tokens & PATH_FOCUS_QUERY_TERMS)
    if relationship_focus and not path_focus:
        return True

    query_lower = str(query or "").lower()
    return bool(
        re.search(r"\b(?:strongest|top|main|key|primary)\s+(?:graph\s+)?relationships?\b", query_lower)
        or re.search(r"\bwhat\s+(?:is|are).{0,40}\brelationships?\b", query_lower)
    )


def _query_prefers_layout_graph_paths(query: str) -> bool:
    query_lower = str(query or "").lower()
    if not query_lower:
        return False
    tokens = _query_tokens(query_lower)
    layout_terms = {
        "archive",
        "attachment",
        "chart",
        "charts",
        "deck",
        "diagram",
        "diagrams",
        "docx",
        "file",
        "files",
        "figure",
        "hidden",
        "image",
        "page",
        "pages",
        "pdf",
        "pptx",
        "presentation",
        "sheet",
        "sheets",
        "slide",
        "slides",
        "source",
        "spreadsheet",
        "table",
        "tables",
        "workbook",
        "xml",
    }
    if re.search(
        r"\b(?:which|what|where)\b.{0,80}\b"
        r"(?:archive|attachment|chart|deck|diagram|docx|file|figure|hidden|image|page|pdf|pptx|presentation|sheet|slide|source|spreadsheet|table|workbook|xml)\b",
        query_lower,
    ):
        return True
    if re.search(r"\b(?:contains?|includes?|located|found|stored|shows?|value|visibility)\b", query_lower) and tokens & layout_terms:
        return True
    return False


def _query_prefers_semantic_graph_paths(query: str) -> bool:
    query_lower = str(query or "").lower()
    if not query_lower or _query_prefers_layout_graph_paths(query_lower):
        return False
    tokens = _query_tokens(query_lower)
    relation_focus = any(trigger in query_lower for trigger in GRAPH_RELATION_QUERY_TRIGGERS)
    return relation_focus or bool(tokens & (PATH_FOCUS_QUERY_TERMS | RELATIONSHIP_FOCUS_QUERY_TERMS))


def _graph_relationship_evidence_quality(relationship: dict) -> int:
    relation_key = _graph_context_relation_key(relationship)
    structural = _is_structural_relation(relation_key)
    generic = relation_key in GENERIC_GRAPH_CONTEXT_RELATIONS
    style = relation_key in STYLE_GRAPH_CONTEXT_RELATIONS
    relationship_type = str(relationship.get("relationship_type") or "").lower()
    semantic = relationship_type == "semantic" or not structural
    direct_subject_fact = semantic and not structural and not generic and not style
    requested_terms = _relationship_requested_terms(relationship)

    internal_artifact = _relationship_has_internal_artifact_endpoint(relationship)
    external_url = _relationship_has_external_url_endpoint(relationship)
    chart_annotation = _relationship_is_chart_annotation(relationship)
    container_detail = _relationship_is_container_detail(relationship)
    document_endpoint = _relationship_has_document_endpoint(relationship)

    if direct_subject_fact and not any((internal_artifact, external_url, chart_annotation, container_detail, document_endpoint)):
        return 5
    if external_url:
        return 4 if requested_terms & LINK_DETAIL_QUERY_TERMS else 1
    if chart_annotation:
        return 4 if requested_terms & CHART_DETAIL_QUERY_TERMS else 2
    if internal_artifact:
        return 3 if requested_terms & ARTIFACT_DETAIL_QUERY_TERMS else 1
    if container_detail or document_endpoint:
        return 3 if requested_terms & DOCUMENT_DETAIL_QUERY_TERMS else 1
    if direct_subject_fact:
        return 4
    if generic:
        return 0
    return 3 if semantic else 2


def _graph_relationship_rank_key(relationship: dict) -> tuple:
    relation_key = _graph_context_relation_key(relationship)
    structural = _is_structural_relation(relation_key)
    generic = relation_key in GENERIC_GRAPH_CONTEXT_RELATIONS
    style = relation_key in STYLE_GRAPH_CONTEXT_RELATIONS
    relationship_type = str(relationship.get("relationship_type") or "").lower()
    semantic = relationship_type == "semantic" or not structural
    direct = relation_key in DIRECT_GRAPH_CONTEXT_RELATIONS or (
        semantic and not structural and not generic and not style
    )
    specific_terms = len(relationship.get("specific_matched_query_terms", []) or [])
    matched_terms = len(relationship.get("matched_query_terms", []) or [])
    evidence_family_count = len(relationship.get("evidence_families", []) or [])

    return (
        _graph_relationship_evidence_quality(relationship),
        1 if direct else 0,
        1 if semantic and not generic and not structural else 0,
        0 if not generic else -1,
        0 if not structural else -1,
        specific_terms,
        matched_terms,
        float(relationship.get("score") or 0),
        _safe_weight(relationship.get("weight", relationship.get("confidence", 0))),
        evidence_family_count,
    )


def _query_entity_phrases(query: str) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for match in re.findall(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)+\b", str(query or "")):
        phrase = _clean_entity_candidate(match)
        key = phrase.lower()
        if len(phrase.split()) < 2 or key in seen:
            continue
        if all(term in GRAPH_REASONING_GENERIC_NODE_TERMS for term in _query_tokens(phrase)):
            continue
        seen.add(key)
        phrases.append(phrase)
    return phrases[:8]


def _path_query_entity_coverage(path: dict, query: str) -> tuple[int, int]:
    query_text = str(query or "").lower()
    phrases = _query_entity_phrases(query)
    seen_phrases = {phrase.lower() for phrase in phrases}
    path_entity_values: list[str] = []
    path_entity_values.extend(str(node or "") for node in (path.get("display_nodes") or path.get("nodes") or []))
    for relationship in path.get("relationships", []) or []:
        if not isinstance(relationship, dict):
            continue
        path_entity_values.extend([
            str(relationship.get("source") or ""),
            str(relationship.get("target") or ""),
        ])
    for value in path_entity_values:
        phrase = _clean_entity_candidate(value)
        key = phrase.lower()
        if not phrase or key in seen_phrases:
            continue
        terms = [
            term
            for term in _query_tokens(phrase)
            if term not in GRAPH_QUERY_STOPWORDS and len(term) >= 3
        ]
        if not terms:
            continue
        if all(term in GRAPH_REASONING_GENERIC_NODE_TERMS for term in terms):
            continue
        if key in query_text or all(term in query_text for term in terms):
            seen_phrases.add(key)
            phrases.append(phrase)
    if not phrases:
        return (0, 0)
    node_text = " ".join(str(node or "") for node in (path.get("display_nodes") or path.get("nodes") or [])).lower()
    relationship_text = " ".join(
        " ".join(str(relationship.get(key) or "") for key in ("source", "relation", "target", "evidence", "provenance"))
        for relationship in path.get("relationships", []) or []
        if isinstance(relationship, dict)
    ).lower()

    def matches(text: str, phrase: str) -> bool:
        phrase_key = phrase.lower()
        if phrase_key in text:
            return True
        terms = [term for term in _query_tokens(phrase) if term not in GRAPH_QUERY_STOPWORDS and len(term) >= 3]
        return bool(terms) and all(term in text for term in terms)

    node_matches = sum(1 for phrase in phrases if matches(node_text, phrase))
    evidence_matches = sum(1 for phrase in phrases if matches(relationship_text, phrase))
    return (node_matches, evidence_matches)


def _graph_path_rank_key(path: dict, query: str = "") -> tuple:
    relationships = [
        relationship
        for relationship in path.get("relationships", []) or []
        if isinstance(relationship, dict)
    ]
    qualities = [_graph_relationship_evidence_quality(relationship) for relationship in relationships]
    average_quality = sum(qualities) / len(qualities) if qualities else 0.0
    minimum_quality = min(qualities) if qualities else 0
    semantic_preferred = _query_prefers_semantic_graph_paths(query)
    layout_preferred = _query_prefers_layout_graph_paths(query)
    node_coverage, evidence_coverage = _path_query_entity_coverage(path, query)
    phrase_count = len(_query_entity_phrases(query))
    full_node_coverage = phrase_count > 0 and node_coverage >= phrase_count
    bridge_hint = _graph_bridge_hint_from_query(query)
    bridge_key = _normalized_lookup_key(bridge_hint)
    path_node_keys = [
        _normalized_lookup_key(str(node))
        for node in path.get("display_nodes") or path.get("nodes") or []
        if _normalized_lookup_key(str(node))
    ]
    bridge_match = bool(
        bridge_key
        and bridge_key in set(path_node_keys[1:-1] if len(path_node_keys) > 2 else path_node_keys)
    )
    specific_relationship_count = sum(
        1
        for relationship in relationships
        if not _is_structural_relation(_graph_context_relation_key(relationship))
        and _graph_context_relation_key(relationship) not in GENERIC_GRAPH_CONTEXT_RELATIONS
    )
    structural_relationship_count = sum(
        1
        for relationship in relationships
        if _is_structural_relation(_graph_context_relation_key(relationship))
    )
    generic_relationship_count = sum(
        1
        for relationship in relationships
        if _graph_context_relation_key(relationship) in GENERIC_GRAPH_CONTEXT_RELATIONS
    )
    semantic_path = bool(relationships) and structural_relationship_count == 0
    layout_path = path.get("path_type") == "layout" or structural_relationship_count > 0
    return (
        1 if bridge_match else 0,
        1 if semantic_preferred and semantic_path else 0,
        1 if layout_preferred and layout_path else 0,
        node_coverage,
        evidence_coverage,
        1 if full_node_coverage else 0,
        0 if not semantic_preferred else -structural_relationship_count,
        0 if not semantic_preferred else -generic_relationship_count,
        specific_relationship_count,
        minimum_quality,
        average_quality,
        1 if not layout_path else 0,
        float(path.get("score") or 0),
        _safe_weight(path.get("average_weight", 0)),
    )


def _pipeline_graph_overview(
    node_map: dict,
    raw_edges: list[dict],
    sources: list[dict] | None = None,
    *,
    query: str = "",
    limit: int = 12,
    normalized_edges: list[dict] | None = None,
) -> dict:
    overview_edges: list[dict] = []
    relation_counts: Counter[str] = Counter()
    entity_runs: dict[str, dict] = defaultdict(lambda: {
        "name": "",
        "runs": set(),
        "filenames": set(),
        "relationships": set(),
        "incident_edges": [],
    })
    query_tokens = _token_variants({
        token for token in _query_tokens(query)
        if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
    })

    graph_edges = normalized_edges if normalized_edges is not None else _normalize_graph_edges(raw_edges, node_map)
    for normalized_edge in graph_edges:
        normalized = dict(normalized_edge)
        relation = str(normalized.get("relation") or "related_to")
        metadata = normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {}
        filename = str(metadata.get("local_graph_filename") or "")
        run_id = str(metadata.get("local_graph_run_id") or "")
        relation_counts[relation] += 1
        normalized["score"] = round(
            (_safe_weight(normalized.get("weight", 1)) * 0.35)
            + (_edge_query_score(normalized, query) * 0.35)
            + (0.18 if not _is_structural_relation(relation) else 0.08),
            3,
        )
        normalized["relationship_type"] = "layout" if _is_structural_relation(relation) else "semantic"
        normalized["matched_query_terms"] = sorted(query_tokens & _edge_query_terms(normalized))
        normalized["evidence_families"] = _graph_edge_evidence_families(normalized)
        normalized["provenance_summary"] = _graph_edge_provenance_summary(normalized)
        normalized["rationale"] = _graph_edge_rationale(normalized)
        overview_edges.append(normalized)

        for endpoint in (normalized.get("source"), normalized.get("target")):
            name = str(endpoint or "").strip()
            if not name or re.match(r"^(?:slide|page|sheet)\s+\d+\b", name, flags=re.I):
                continue
            record = entity_runs[name.lower()]
            record["name"] = name
            if run_id:
                record["runs"].add(run_id)
            if filename:
                record["filenames"].add(filename)
            record["relationships"].add(relation)
            record["incident_edges"].append(normalized)

    shared_entities = []
    for record in entity_runs.values():
        if len(record["runs"]) < 2 and len(record["filenames"]) < 2:
            continue
        shared_entities.append({
            "name": record["name"],
            "run_count": len(record["runs"]),
            "filenames": sorted(record["filenames"])[:6],
            "relationships": sorted(record["relationships"])[:8],
        })

    top_relationships = sorted(
        overview_edges,
        key=_graph_relationship_rank_key,
        reverse=True,
    )[:limit]
    bridge_entities = []
    cross_document_paths = []
    for record in entity_runs.values():
        incident_edges = [edge for edge in record.get("incident_edges", []) if isinstance(edge, dict)]
        if len(record["runs"]) < 2 and len(record["filenames"]) < 2:
            continue
        if len(incident_edges) < 2:
            continue
        bridge_name = record["name"]
        bridge_entities.append({
            "name": bridge_name,
            "run_count": len(record["runs"]),
            "filenames": sorted(record["filenames"])[:6],
            "relationship_count": len(incident_edges),
            "relationships": sorted(record["relationships"])[:8],
            "score": round(
                min(1.0, 0.45 + (min(len(record["runs"]), 4) * 0.08) + (min(len(incident_edges), 6) * 0.035)),
                3,
            ),
        })
        sorted_edges = sorted(
            incident_edges,
            key=_graph_relationship_rank_key,
            reverse=True,
        )
        for left_index, left in enumerate(sorted_edges[:6]):
            left_metadata = left.get("metadata") if isinstance(left.get("metadata"), dict) else {}
            for right in sorted_edges[left_index + 1:6]:
                right_metadata = right.get("metadata") if isinstance(right.get("metadata"), dict) else {}
                left_run = str(left_metadata.get("local_graph_run_id") or "")
                right_run = str(right_metadata.get("local_graph_run_id") or "")
                left_filename = str(left_metadata.get("local_graph_filename") or "")
                right_filename = str(right_metadata.get("local_graph_filename") or "")
                if left_run == right_run and left_filename == right_filename:
                    continue
                left_other = left.get("target") if left.get("source") == bridge_name else left.get("source")
                right_other = right.get("target") if right.get("source") == bridge_name else right.get("source")
                if not left_other or not right_other or left_other == right_other:
                    continue
                score = round(min(1.0, ((left.get("score", 0) + right.get("score", 0)) / 2) + 0.08), 3)
                path_relationships = [left, right]
                average_weight = round(
                    sum(_safe_weight(edge.get("weight", 1)) for edge in path_relationships) / len(path_relationships),
                    3,
                )
                path = {
                    "bridge": bridge_name,
                    "source": left_other,
                    "target": right_other,
                    "nodes": [left_other, bridge_name, right_other],
                    "hops": 2,
                    "score": score,
                    "average_weight": average_weight,
                    "path_type": _path_type(path_relationships),
                    "matched_query_terms": _path_matched_query_terms(path_relationships, query),
                    "filenames": sorted({left_filename, right_filename} - {""}),
                    "run_ids": sorted({left_run, right_run} - {""}),
                    "relationships": path_relationships,
                    "text": (
                        f"{left_other} --[{left.get('relation')}]-- {bridge_name} "
                        f"--[{right.get('relation')}]-- {right_other}"
                    ),
                }
                path.update(_graph_path_explanation(path))
                cross_document_paths.append(path)
                break
            if len(cross_document_paths) >= limit * 2:
                break

    return {
        "source_count": len(sources or []),
        "relationship_count": len(overview_edges),
        "relationship_type_counts": dict(relation_counts.most_common(12)),
        "top_relationships": top_relationships,
        "shared_entities": sorted(
            shared_entities,
            key=lambda item: (item["run_count"], len(item["relationships"])),
            reverse=True,
        )[:limit],
        "bridge_entity_count": len(bridge_entities),
        "bridge_entities": sorted(
            bridge_entities,
            key=lambda item: (item.get("score", 0), item["run_count"], item["relationship_count"]),
            reverse=True,
        )[:limit],
        "cross_document_path_count": len(cross_document_paths),
        "cross_document_paths": sorted(
            cross_document_paths,
            key=lambda path: _graph_path_rank_key(path, query),
            reverse=True,
        )[:limit],
    }


def _score_graph_path(path_edges: list[dict], query: str) -> tuple[float, float]:
    if not path_edges:
        return 0.0, 0.0
    weights = [_safe_weight(edge.get("weight", 1)) for edge in path_edges]
    average_weight = sum(weights) / len(weights)
    relation_match = max((_edge_query_score(edge, query) for edge in path_edges), default=0.0)
    layout_match = _layout_query_score(query) if any(_is_structural_relation(edge.get("relation", "")) for edge in path_edges) else 0.0
    hop_bonus = 1 / max(len(path_edges), 1)
    score = (average_weight * 0.52) + (relation_match * 0.24) + (hop_bonus * 0.08) + (layout_match * 0.16)
    return round(score, 3), round(average_weight, 3)


def _path_type(path_edges: list[dict]) -> str:
    return "layout" if any(_is_structural_relation(edge.get("relation", "")) for edge in path_edges) else "semantic"


def _path_matched_query_terms(path_edges: list[dict], query: str) -> list[str]:
    query_tokens = {token for token in _query_tokens(query) if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2}
    matched: list[str] = []
    seen = set()
    for edge in path_edges:
        for term in sorted(query_tokens & _edge_query_terms(edge)):
            if term not in seen:
                seen.add(term)
                matched.append(term)
    return matched


def _find_graph_paths(
    source: str,
    target: str,
    node_map: dict,
    raw_edges: list,
    max_hops: int = 3,
    limit: int = 3,
    query: str = "",
    normalized_edges: list[dict] | None = None,
) -> list[dict]:
    source_node = _match_graph_node(source, node_map)
    target_node = _match_graph_node(target, node_map)
    if not source_node or not target_node:
        return []

    source_name = _node_display_name(source_node)
    target_name = _node_display_name(target_node)
    graph_edges = normalized_edges if normalized_edges is not None else _normalize_graph_edges(raw_edges, node_map)
    adjacency: dict[str, list[dict]] = {}

    for edge in graph_edges:
        src = edge["source"]
        tgt = edge["target"]
        if not src or not tgt:
            continue
        adjacency.setdefault(src.lower(), []).append({
            **edge,
            "from": src,
            "to": tgt,
            "canonical_source": src,
            "canonical_target": tgt,
            "direction": "outgoing",
        })
        adjacency.setdefault(tgt.lower(), []).append({
            **edge,
            "from": tgt,
            "to": src,
            "canonical_source": src,
            "canonical_target": tgt,
            "direction": "incoming",
        })

    paths = []
    seen_path_keys = set()
    queue = deque([(source_name, [source_name], {source_name.lower()}, [])])
    expansions = 0
    candidate_limit = max(limit * 8, 16)

    while queue and expansions < 2000 and len(paths) < candidate_limit:
        current, visited_nodes, visited_keys, path_edges = queue.popleft()
        expansions += 1
        if len(path_edges) >= max_hops:
            continue

        for edge in sorted(adjacency.get(current.lower(), []), key=lambda item: item.get("weight", 0), reverse=True):
            next_node = edge["to"]
            next_key = next_node.lower()
            if next_key in visited_keys:
                continue
            path_edge = {
                "source": edge.get("canonical_source") or edge.get("source") or edge["from"],
                "relation": edge["relation"],
                "target": edge.get("canonical_target") or edge.get("target") or next_node,
                "traversal_source": edge["from"],
                "traversal_target": next_node,
                "direction": edge["direction"],
                "weight": edge["weight"],
                "evidence": edge.get("evidence"),
                "metadata": edge.get("metadata", {}),
                "provenance": edge.get("provenance", ""),
            }
            path_edge["evidence_families"] = _graph_edge_evidence_families(path_edge)
            path_edge["provenance_summary"] = _graph_edge_provenance_summary(path_edge)
            path_edge["rationale"] = _graph_edge_rationale(path_edge)
            next_path_nodes = [*visited_nodes, next_node]
            next_path_edges = [*path_edges, path_edge]

            if next_node.lower() == target_name.lower():
                key = tuple(node.lower() for node in next_path_nodes)
                if key in seen_path_keys:
                    continue
                seen_path_keys.add(key)
                score, average_weight = _score_graph_path(next_path_edges, query)
                path = {
                    "source": source_name,
                    "target": target_name,
                    "hops": len(next_path_edges),
                    "nodes": next_path_nodes,
                    "relationships": next_path_edges,
                    "score": score,
                    "average_weight": average_weight,
                    "path_type": _path_type(next_path_edges),
                    "matched_query_terms": _path_matched_query_terms(next_path_edges, query),
                    "traversal_text": " -> ".join(next_path_nodes),
                    "text": " -> ".join(
                        f"{rel['source']} --[{rel['relation']}]-- {rel['target']}"
                        for rel in next_path_edges
                    ),
                }
                path.update(_graph_path_explanation(path))
                paths.append(path)
                continue

            queue.append((next_node, next_path_nodes, {*visited_keys, next_key}, next_path_edges))

    return sorted(paths, key=lambda path: _graph_path_rank_key(path, query), reverse=True)[:limit]


def _bridge_entities_from_paths(paths: list[dict], *, limit: int = 8) -> list[dict]:
    bridges: dict[str, dict] = {}

    for path_index, path in enumerate(sorted(paths or [], key=lambda item: item.get("score", 0), reverse=True), start=1):
        if not isinstance(path, dict):
            continue
        nodes = path.get("nodes", []) or []
        relationships = path.get("relationships", []) or []
        if len(nodes) <= 2:
            continue
        endpoints = {
            "source": path.get("source") or nodes[0],
            "target": path.get("target") or nodes[-1],
        }
        for bridge_name in nodes[1:-1]:
            key = str(bridge_name).lower()
            bridge = bridges.setdefault(key, {
                "name": bridge_name,
                "path_count": 0,
                "score": 0.0,
                "connects": [],
                "relationships": [],
                "path_types": [],
                "matched_query_terms": [],
            })
            bridge["path_count"] += 1
            bridge["score"] = max(float(bridge.get("score", 0)), float(path.get("score", 0) or 0))
            if endpoints not in bridge["connects"]:
                bridge["connects"].append(endpoints)
            path_type = path.get("path_type")
            if path_type and path_type not in bridge["path_types"]:
                bridge["path_types"].append(path_type)
            for term in path.get("matched_query_terms", []) or []:
                if term not in bridge["matched_query_terms"]:
                    bridge["matched_query_terms"].append(term)
            for rel in relationships:
                if not isinstance(rel, dict):
                    continue
                source = str(rel.get("source") or "")
                target = str(rel.get("target") or "")
                if bridge_name not in {source, target}:
                    continue
                rel_text = _format_graph_relationship(rel, path_style=True)
                if rel_text not in bridge["relationships"]:
                    bridge["relationships"].append(rel_text)

    bridge_list = []
    for bridge in bridges.values():
        score = round(min(1.0, float(bridge.get("score", 0)) + min(bridge.get("path_count", 1) - 1, 3) * 0.04), 3)
        bridge["score"] = score
        bridge["relationships"] = bridge["relationships"][:6]
        bridge["connects"] = bridge["connects"][:4]
        bridge["text"] = (
            f"{bridge['name']} bridges "
            + "; ".join(f"{item['source']} to {item['target']}" for item in bridge["connects"])
        )
        bridge_list.append(bridge)

    return sorted(bridge_list, key=lambda item: (item.get("score", 0), item.get("path_count", 0)), reverse=True)[:limit]


def _mentioned_graph_relationships(
    entity_names: list[str],
    node_map: dict,
    raw_edges: list,
    limit: int = 12,
    query: str = "",
    normalized_edges: list[dict] | None = None,
) -> list[dict]:
    matched_nodes = {
        _node_display_name(node).lower()
        for name in entity_names
        if (node := _match_graph_node(name, node_map))
    }
    relationships = []

    graph_edges = normalized_edges if normalized_edges is not None else _normalize_graph_edges(raw_edges, node_map)
    for normalized_edge in graph_edges:
        normalized = dict(normalized_edge)
        source_key = normalized["source"].lower()
        target_key = normalized["target"].lower()
        if source_key in matched_nodes or target_key in matched_nodes:
            endpoint_hits = int(source_key in matched_nodes) + int(target_key in matched_nodes)
            relation_score = _edge_query_score(normalized, query)
            normalized["score"] = round((endpoint_hits * 0.35) + (_safe_weight(normalized.get("weight", 1)) * 0.25) + (relation_score * 0.25), 3)
            normalized["relationship_type"] = "layout" if _is_structural_relation(normalized.get("relation", "")) else "semantic"
            query_tokens = _token_variants({
                token for token in _query_tokens(query)
                if token not in GRAPH_QUERY_STOPWORDS and len(token) >= 2
            })
            normalized["matched_query_terms"] = sorted(query_tokens & _edge_query_terms(normalized))
            relationships.append(normalized)

    return sorted(relationships, key=_graph_relationship_rank_key, reverse=True)[:limit]


async def _optional_query_complexity(client, query: str, errors: list) -> dict:
    cfg = get_graphrag_config()
    if not cfg.graphrag_url:
        return {}
    try:
        response = await client.post(
            f"{cfg.graphrag_url}/api/query-complexity-analysis",
            params={"query": query},
            timeout=optional_chat_context_timeout(),
        )
        if response.status_code == 200:
            payload = response.json()
            return payload if isinstance(payload, dict) else {}
    except Exception as e:
        logger.debug(f"GraphRAG query complexity analysis skipped: {e}")
    return {}


async def _build_graph_reasoning_context(
    client,
    query: str,
    entities: list[dict],
    domain: str,
    errors: list,
    explicit_source: str | None = None,
    explicit_target: str | None = None,
    explicit_bridge: str | None = None,
    explicit_relation: str | None = None,
    pipeline_run_ids: list[str] | None = None,
    pipeline_evidence: dict | None = None,
    prefer_local_pipeline_only: bool = False,
) -> dict:
    cfg = get_graphrag_config()
    explicit_bridge = explicit_bridge or _graph_bridge_hint_from_query(query)
    intent_hints = _compact_graph_intent_hints(
        source=explicit_source,
        target=explicit_target,
        bridge=explicit_bridge,
        relation=explicit_relation,
    )
    reasoning_query = _graph_intent_query(
        query,
        source=explicit_source,
        target=explicit_target,
        bridge=explicit_bridge,
        relation=explicit_relation,
    )
    entity_names = _entity_names_for_reasoning(reasoning_query, entities)
    explicit_pair = None
    inferred_pairs = _entity_pairs_from_query(query)
    if explicit_source and explicit_target:
        explicit_pair = (_clean_entity_candidate(explicit_source), _clean_entity_candidate(explicit_target))
        for name in explicit_pair:
            if name and name.lower() not in {candidate.lower() for candidate in entity_names}:
                entity_names.append(name)
    else:
        for pair in inferred_pairs:
            for name in pair:
                if name and name.lower() not in {candidate.lower() for candidate in entity_names}:
                    entity_names.append(name)
    if intent_hints.get("bridge") and intent_hints["bridge"].lower() not in {candidate.lower() for candidate in entity_names}:
        entity_names.append(intent_hints["bridge"])
    entity_names = _clean_graph_reasoning_entity_names(reasoning_query, entity_names)

    inferred_explicit_pair = inferred_pairs[0] if inferred_pairs else None
    should_reason = _graph_reasoning_needed(
        reasoning_query,
        entity_names,
        explicit_pair or inferred_explicit_pair,
    )
    result = {
        "status": "not_requested" if not should_reason else "unavailable",
        "reasoning_needed": should_reason,
        "query": reasoning_query,
        "intent_hints": intent_hints,
        "entity_names": entity_names,
        "entity_pairs": [],
        "inferred_entity_pairs": [{"source": source, "target": target} for source, target in inferred_pairs],
        "graph_inferred_entity_pairs": [],
        "paths": [],
        "bridge_entities": [],
        "direct_relationships": [],
        "graph_overview": {},
        "query_complexity": {},
        "graph_sources": {
            "remote_graph": False,
            "local_pipeline_runs": [],
            "local_pipeline_interaction_handoffs": [],
        },
        "interaction_handoffs": [],
        "pipeline_evidence": pipeline_evidence if isinstance(pipeline_evidence, dict) else {},
        "summary": {},
        "answer_plan": {},
    }
    probe_recent_pipeline_graph = _local_pipeline_graph_probe_needed(reasoning_query)
    if not should_reason and not pipeline_run_ids and not result["pipeline_evidence"] and not probe_recent_pipeline_graph:
        return _finalize_graph_reasoning_result(result)

    if should_reason and cfg.graphrag_url and not prefer_local_pipeline_only:
        result["query_complexity"] = await _optional_query_complexity(client, reasoning_query, errors)

    node_map: dict = {}
    raw_edges: list = []
    try:
        if should_reason and cfg.graphrag_url and not prefer_local_pipeline_only:
            node_map, raw_edges = await _fetch_graph_data(
                client,
                domain,
                timeout=optional_chat_context_timeout(),
            )
            result["graph_sources"]["remote_graph"] = bool(node_map or raw_edges)
    except Exception as e:
        logger.warning(f"Graph path context failed: {e}")
        errors.append(f"Graph paths: {str(e)}")

    local_node_map, local_edges, local_sources = _local_pipeline_graph_data(
        domain,
        pipeline_run_ids=pipeline_run_ids,
    )
    local_handoffs = _local_pipeline_interaction_handoffs(
        domain,
        pipeline_run_ids=pipeline_run_ids,
    )
    if local_handoffs:
        result["interaction_handoffs"] = local_handoffs
        result["graph_sources"]["local_pipeline_interaction_handoffs"] = [
            {
                "run_id": handoff.get("run_id"),
                "filename": handoff.get("filename"),
                "domain": handoff.get("domain"),
                "recommended_mode": handoff.get("recommended_mode"),
                "graph_anchor_count": len(handoff.get("graph_anchors", []) or []),
                "starter_prompt_count": len(handoff.get("starter_prompts", []) or []),
            }
            for handoff in local_handoffs
        ]
    if local_edges:
        _merge_node_map(node_map, local_node_map)
        raw_edges = [*raw_edges, *local_edges]
        result["graph_sources"]["local_pipeline_runs"] = local_sources

    normalized_edges = _normalize_graph_edges(raw_edges, node_map) if raw_edges else []

    if raw_edges:
        graph_inferred_pairs = [] if explicit_pair else _entity_pairs_from_graph_mentions(
            reasoning_query,
            node_map,
            raw_edges,
            normalized_edges=normalized_edges,
        )
        if graph_inferred_pairs:
            result["graph_inferred_entity_pairs"] = [
                {"source": source, "target": target}
                for source, target in graph_inferred_pairs
            ]
            existing_pair_keys = {(source.lower(), target.lower()) for source, target in inferred_pairs}
            for pair in graph_inferred_pairs:
                if (pair[0].lower(), pair[1].lower()) not in existing_pair_keys:
                    inferred_pairs.append(pair)
                    existing_pair_keys.add((pair[0].lower(), pair[1].lower()))
                for name in pair:
                    if name and name.lower() not in {candidate.lower() for candidate in entity_names}:
                        entity_names.append(name)
            entity_names = _clean_graph_reasoning_entity_names(reasoning_query, entity_names)
            result["entity_names"] = entity_names[:8]
            if not should_reason:
                should_reason = True
                result["reasoning_needed"] = True
                result["status"] = "unavailable"

        result["graph_overview"] = _pipeline_graph_overview(
            node_map,
            raw_edges,
            local_sources,
            query=reasoning_query,
            normalized_edges=normalized_edges,
        )
        result["direct_relationships"] = _merge_graph_relationships(
            result.get("direct_relationships", []),
            _query_relevant_graph_relationships(
                raw_edges,
                node_map,
                reasoning_query,
                normalized_edges=normalized_edges,
            ),
        )

    if not raw_edges:
        if local_handoffs:
            result["status"] = "complete"
        elif result["pipeline_evidence"]:
            result["status"] = "pipeline_evidence"
        return _finalize_graph_reasoning_result(result)

    if not should_reason:
        if result["direct_relationships"] or result["graph_overview"].get("top_relationships"):
            result["status"] = "complete"
        elif local_handoffs:
            result["status"] = "complete"
        return _finalize_graph_reasoning_result(result)

    pairs = (
        [explicit_pair]
        if explicit_pair
        else inferred_pairs
        if inferred_pairs
        else []
        if pipeline_run_ids
        else list(combinations(entity_names[:4], 2))
    )
    result["entity_pairs"] = [{"source": source, "target": target} for source, target in pairs]

    for source, target in pairs[:4]:
        result["paths"].extend(
            _find_graph_paths(
                source,
                target,
                node_map,
                raw_edges,
                query=reasoning_query,
                normalized_edges=normalized_edges,
            )
        )

    result["paths"] = sorted(
        [path for path in result["paths"] if isinstance(path, dict)],
        key=lambda path: _graph_path_rank_key(path, reasoning_query),
        reverse=True,
    )
    result["bridge_entities"] = _bridge_entities_from_paths(result["paths"])
    result["direct_relationships"] = _merge_graph_relationships(
        _mentioned_graph_relationships(
            entity_names,
            node_map,
            raw_edges,
            query=reasoning_query,
            normalized_edges=normalized_edges,
        ),
        result.get("direct_relationships", []),
    )
    if result["paths"] or result["direct_relationships"]:
        result["status"] = "complete"
    else:
        result["status"] = "no_paths_found"
    return _finalize_graph_reasoning_result(result)


@router.post("/chat-context-v2")
async def graphrag_chat_context_v2(request: Request):
    """
    Unified RAG + Graph context fusion endpoint.

    Merges vector RAG retrieval and GraphRAG (NER, REL, graph expansion)
    into a single call with cross-referencing between chunks and entities.

    Returns:
    - chunks: RAG chunks with entity annotations
    - entities: extracted + graph-expanded entities
    - relationships: extracted relationships
    - cross_references: which chunks mention which entities
    - context_text: single enriched context block for LLM
    - quality_signals: relevance and coverage metrics
    """
    check_graphrag_enabled()
    cfg = get_graphrag_config()

    data = await request.json()
    query = data.get("query", "")
    top_k = data.get("top_k", 5)
    graph_domain = data.get("domain", "general")
    rag_enabled = data.get("rag_enabled", True)
    rag_domains = data.get("rag_domains", [])
    rag_search_mode = data.get("rag_search_mode", "domain")
    pipeline_run_ids = _string_list(data.get("pipeline_run_ids") or data.get("pipeline_run_id"))
    pipeline_evidence = _pipeline_evidence_hints_from_request(data if isinstance(data, dict) else {})
    local_pipeline_requested = bool(
        pipeline_run_ids
        or data.get("use_local_pipeline_context") is True
        or data.get("include_local_pipeline") is True
    )
    local_pipeline_enabled = bool(
        (rag_enabled or local_pipeline_requested)
        and data.get("use_local_pipeline_context") is not False
        and data.get("include_local_pipeline") is not False
    )
    prefer_local_pipeline_only = bool(pipeline_run_ids and local_pipeline_enabled)

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    errors = []
    rag_chunks = []

    client = get_http_client()

    # ── Steps 1-2: NER & REL (parallel — independent services) ────
    entities, relationships = await asyncio.gather(
        _extract_entities_ner(client, query, errors),
        _extract_relationships_rel(client, query, errors),
    )

    # ── Step 3: RAG vector retrieval (direct call, no HTTP self-loop) ──
    if rag_enabled:
        rag_chunks = await _retrieve_rag_chunks_for_query(
            request,
            query,
            top_k,
            rag_domains,
            rag_search_mode,
            errors,
        )

    # ── Step 4: GraphRAG vector search ────────────────────────────
    graph_search_results = []
    if not prefer_local_pipeline_only:
        graph_search_results, extra_entities = await _search_graphrag(
            client, query, top_k, graph_domain, errors
        )
        entities.extend(extra_entities)

    # ── Step 5: Graph neighborhood expansion (shared helper) ──────
    if prefer_local_pipeline_only:
        expanded_entities, graph_connections = [], []
    else:
        expanded_entities, graph_connections = await _expand_graph_neighborhood(
            client,
            entities,
            graph_domain,
            errors,
            timeout=optional_chat_context_timeout(),
        )

    # ── Deduplication (shared helpers) ─────────────────────────────
    entities = _deduplicate_entities(entities, expanded_entities)
    relationships = _deduplicate_relationships(relationships)

    # ── Step 6: Graph path and relationship reasoning ──────────────
    graph_reasoning = await _build_graph_reasoning_context(
        client,
        query,
        entities,
        graph_domain,
        errors,
        explicit_source=data.get("source"),
        explicit_target=data.get("target"),
        explicit_bridge=data.get("bridge"),
        explicit_relation=data.get("relation"),
        pipeline_run_ids=pipeline_run_ids,
        pipeline_evidence=pipeline_evidence,
        prefer_local_pipeline_only=prefer_local_pipeline_only,
    )

    graph_augmented_terms = _graph_augmented_retrieval_terms(graph_reasoning)
    graph_augmented_query = _graph_augmented_retrieval_query(query, graph_reasoning)
    supplemental_rag_chunks = []
    local_pipeline_chunks = []
    if rag_enabled and graph_augmented_query != query:
        supplemental_top_k = _bounded_retrieval_top_k(top_k)
        supplemental_rag_chunks = await _retrieve_rag_chunks_for_query(
            request,
            graph_augmented_query,
            supplemental_top_k,
            rag_domains,
            rag_search_mode,
            errors,
            retrieval_reason="graph_augmented",
        )
        rag_chunks = _merge_rag_chunks(rag_chunks, supplemental_rag_chunks)

    if local_pipeline_enabled:
        local_pipeline_chunks = _local_pipeline_context_chunks(
            graph_augmented_query,
            graph_domain,
            top_k,
            graph_reasoning,
            pipeline_run_ids=pipeline_run_ids,
            user_query=query,
        )
        rag_chunks = _merge_rag_chunks(rag_chunks, local_pipeline_chunks)

    pipeline_artifacts = (
        _local_pipeline_artifact_context(
            graph_augmented_query,
            graph_domain,
            pipeline_run_ids=pipeline_run_ids,
        )
        if local_pipeline_enabled
        else {"runs": [], "run_count": 0, "artifact_count": 0}
    )

    # ── Step 7: Cross-referencing — entity and graph-path-aware chunk scoring ─────
    cross_references = _score_chunks_with_graph_context(rag_chunks, entities, graph_reasoning)
    graph_reasoning = _promote_chunk_graph_paths_to_reasoning(graph_reasoning, rag_chunks, query)

    # ── Step 8: Build unified context template ─────────────────────
    context_parts = []
    answer_plan = graph_reasoning.get("answer_plan") if isinstance(graph_reasoning.get("answer_plan"), dict) else {}
    primary_answer_path = answer_plan.get("primary_path") if isinstance(answer_plan.get("primary_path"), dict) else {}
    path_answer_active = answer_plan.get("strategy") == "walk_strongest_path" and bool(primary_answer_path)
    frontloaded_graph_answer_plan = False
    graph_answer_plan = _graph_answer_plan_context(answer_plan)
    if graph_answer_plan and answer_plan.get("strategy") in {"walk_strongest_path", "cite_direct_relationships"}:
        context_parts.append(graph_answer_plan)
        frontloaded_graph_answer_plan = True

    requested_evidence_context = _requested_evidence_coverage_context(query, rag_chunks)
    requested_evidence_families = _requested_evidence_family_keys(
        query,
        available_families=[
            family
            for chunk in rag_chunks
            if isinstance(chunk, dict)
            for family in _chunk_evidence_families(chunk)
        ],
    )
    if requested_evidence_context:
        context_parts.append(requested_evidence_context)

    # Interleaved chunks with entity annotations
    if rag_chunks:
        chunk_strs = []
        chunk_context_limit = top_k
        if requested_evidence_families:
            chunk_context_limit = max(top_k, min(len(rag_chunks), top_k + 3, 10))
        for i, chunk in enumerate(rag_chunks[:chunk_context_limit]):
            metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            annotation = ""
            if chunk.get("matched_entities"):
                ent_str = ", ".join(chunk["matched_entities"][:5])
                annotation = f" (entities: {ent_str})"
                relevant_conns = []
                for conn in graph_connections:
                    for ent_name in chunk["matched_entities"]:
                        if ent_name.lower() in conn.lower():
                            relevant_conns.append(conn)
                            break
                if relevant_conns:
                    annotation += f" | graph: {'; '.join(relevant_conns[:2])}"
            if chunk.get("matched_graph_path_terms"):
                graph_terms = ", ".join(chunk["matched_graph_path_terms"][:5])
                annotation += f" (graph path evidence: {graph_terms})"
            matched_chunk_evidence = (
                metadata.get("matched_local_chunk_evidence_terms")
                if isinstance(metadata.get("matched_local_chunk_evidence_terms"), list)
                else []
            )
            if matched_chunk_evidence and "local_pipeline_chunk_evidence" in (chunk.get("retrieval_reasons") or []):
                evidence_terms = ", ".join(str(term) for term in matched_chunk_evidence[:5])
                annotation += f" (document evidence: {evidence_terms})"
            if "graph_augmented" in (chunk.get("retrieval_reasons") or []):
                annotation += " (retrieval: graph-expanded)"
            if "local_pipeline" in (chunk.get("retrieval_reasons") or []):
                annotation += " (retrieval: recent pipeline trace)"
            graph_context_edges = _rank_chunk_graph_context_for_query(
                _chunk_graph_context_items(chunk),
                _chunk_retrieval_query(chunk),
            )
            if graph_context_edges:
                annotation += f" (local graph context: {len(graph_context_edges)} relationship{'s' if len(graph_context_edges) != 1 else ''})"
            graph_context_paths = _chunk_graph_path_items(chunk)
            if graph_context_paths:
                annotation += f" (local graph paths: {len(graph_context_paths)} path{'s' if len(graph_context_paths) != 1 else ''})"

            source_info = ""
            if chunk.get("document_title"):
                source_info = f" [source: {chunk['document_title']}]"

            chunk_content = str(chunk["content"])
            if graph_context_edges:
                graph_context_lines = [
                    _format_chunk_graph_context_relationship(edge)
                    for edge in graph_context_edges[:4]
                ]
                chunk_content += "\nGraph relationships:\n" + "\n".join(f"  - {line}" for line in graph_context_lines)
            if graph_context_paths:
                graph_path_lines = [
                    _format_chunk_graph_path(path)
                    for path in graph_context_paths[:4]
                ]
                chunk_content += "\nGraph paths:\n" + "\n".join(f"  - {line}" for line in graph_path_lines)

            chunk_strs.append(f"[{i+1}] {chunk_content}{annotation}{source_info}")

        context_parts.append("=== Knowledge Context ===\n" + "\n\n".join(chunk_strs))

    pipeline_artifact_context = _local_pipeline_artifact_context_text(pipeline_artifacts)
    if pipeline_artifact_context:
        context_parts.append(pipeline_artifact_context)

    # GraphRAG search results (supplementary)
    if graph_search_results:
        graph_strs = [f"[G{i+1}] {r['content']}" for i, r in enumerate(graph_search_results[:3])]
        context_parts.append("=== Graph Search Results ===\n" + "\n\n".join(graph_strs))

    # Entity graph summary
    if entities:
        extracted = [e for e in entities if e.get("source") != "graph_expansion"]
        discovered = [e for e in entities if e.get("source") == "graph_expansion"]

        entity_section = "=== Entity Graph ===\n"
        entity_strs = [f"{e['name']} ({e['type']})" for e in extracted[:15]]
        entity_section += f"Extracted: {', '.join(entity_strs)}"

        if discovered:
            disc_strs = [f"{e['name']} ({e['type']})" for e in discovered[:10]]
            entity_section += f"\nDiscovered via graph: {', '.join(disc_strs)}"

        if graph_connections:
            entity_section += "\nConnections:\n" + "\n".join(f"  • {c}" for c in graph_connections[:15])

        context_parts.append(entity_section)

    requested_evidence_cards = _requested_evidence_coverage_cards(query, rag_chunks)
    if requested_evidence_cards:
        answer_plan_for_cards = (
            graph_reasoning.get("answer_plan")
            if isinstance(graph_reasoning.get("answer_plan"), dict)
            else {}
        )
        existing_cards = (
            answer_plan_for_cards.get("evidence_cards")
            if isinstance(answer_plan_for_cards.get("evidence_cards"), list)
            else []
        )
        answer_plan_for_cards["evidence_cards"] = _dedupe_evidence_cards(
            [*requested_evidence_cards, *existing_cards],
            limit=max(10, len(requested_evidence_cards) + len(existing_cards)),
        )
        existing_anchors = (
            answer_plan_for_cards.get("provenance_anchors")
            if isinstance(answer_plan_for_cards.get("provenance_anchors"), list)
            else []
        )
        answer_plan_for_cards["provenance_anchors"] = _merge_unique_list_values(
            existing_anchors,
            [
                card.get("provenance_summary") or card.get("citation_label")
                for card in requested_evidence_cards
            ],
        )[:12]
        graph_reasoning["answer_plan"] = answer_plan_for_cards

    answer_plan = graph_reasoning.get("answer_plan") if isinstance(graph_reasoning.get("answer_plan"), dict) else {}
    primary_answer_path = answer_plan.get("primary_path") if isinstance(answer_plan.get("primary_path"), dict) else {}
    path_answer_active = answer_plan.get("strategy") == "walk_strongest_path" and bool(primary_answer_path)
    graph_answer_plan = _graph_answer_plan_context(answer_plan)
    if graph_answer_plan and not frontloaded_graph_answer_plan:
        context_parts.append(graph_answer_plan)

    graph_handoff_context = _graph_interaction_handoff_context(
        graph_reasoning.get("interaction_handoffs", [])
    )
    if graph_handoff_context:
        context_parts.append(graph_handoff_context)

    graph_summary = graph_reasoning.get("summary", {}) if isinstance(graph_reasoning.get("summary"), dict) else {}
    if graph_summary.get("evidence"):
        summary_lines = [
            f"Mode: {graph_summary.get('mode', 'unknown')}",
            f"Guidance: {graph_summary.get('answer_guidance', '')}",
            *[f"Evidence: {item}" for item in graph_summary.get("evidence", [])[:4]],
        ]
        context_parts.append("=== Graph Reasoning Summary ===\n" + "\n".join(summary_lines))

    graph_overview = graph_reasoning.get("graph_overview") if isinstance(graph_reasoning.get("graph_overview"), dict) else {}
    overview_relationships = [
        relationship
        for relationship in graph_overview.get("top_relationships", []) or []
        if isinstance(relationship, dict)
    ]
    overview_shared_entities = [
        entity
        for entity in graph_overview.get("shared_entities", []) or []
        if isinstance(entity, dict)
    ]
    overview_bridge_entities = [
        bridge
        for bridge in graph_overview.get("bridge_entities", []) or []
        if isinstance(bridge, dict)
    ]
    overview_paths = [
        path
        for path in graph_overview.get("cross_document_paths", []) or []
        if isinstance(path, dict)
    ]
    if not path_answer_active and (overview_relationships or overview_bridge_entities or overview_paths):
        overview_lines = [
            f"Pipeline traces: {graph_overview.get('source_count', 0)}",
            f"Relationships: {graph_overview.get('relationship_count', len(overview_relationships))}",
        ]
        if overview_shared_entities:
            overview_lines.append(
                "Shared entities: "
                + "; ".join(
                    f"{entity.get('name')} ({', '.join(entity.get('filenames', [])[:3])})"
                    for entity in overview_shared_entities[:6]
                    if entity.get("name")
                )
            )
        if overview_bridge_entities:
            overview_lines.append(
                "Bridge entities: "
                + "; ".join(
                    f"{bridge.get('name')} ({bridge.get('relationship_count', 0)} relationships; {', '.join(bridge.get('filenames', [])[:3])})"
                    for bridge in overview_bridge_entities[:6]
                    if bridge.get("name")
                )
            )
        if overview_paths:
            overview_lines.append("Cross-document paths:")
            overview_lines.extend(
                f"  • {_format_graph_path_for_answer(path) or path.get('text', '')} (score: {path.get('score', 0)}; files: {', '.join(path.get('filenames', [])[:3])})"
                for path in overview_paths[:6]
                if path.get("text")
            )
        if overview_relationships:
            overview_lines.append("Top relationships:")
            overview_lines.extend(
                f"  • {_format_graph_relationship_for_answer(relationship, include_provenance=False)}"
                for relationship in overview_relationships[:10]
            )
        context_parts.append("=== Local Pipeline Graph Overview ===\n" + "\n".join(overview_lines))

    direct_relationships_for_context = [
        relationship
        for relationship in graph_reasoning.get("direct_relationships", []) or []
        if isinstance(relationship, dict)
    ]
    if path_answer_active:
        path_relationships_for_context = _path_supporting_relationships(
            direct_relationships_for_context,
            primary_answer_path,
            limit=10,
        )
        direct_relationships_for_context = _merge_graph_relationships(
            path_relationships_for_context,
            _query_matched_relationships_for_context(direct_relationships_for_context, limit=10),
            limit=10,
        )
    if direct_relationships_for_context:
        direct_rels = [
            _format_graph_relationship_for_answer(rel, include_provenance=False)
            for rel in direct_relationships_for_context[:10]
        ]
        context_parts.append("=== Relevant Graph Relationships ===\n" + "\n".join(f"  • {rel}" for rel in direct_rels))

    if graph_reasoning.get("paths"):
        path_strs = [
            f"{_format_graph_path_for_answer(path) or path['text']} (score: {path.get('score', 0)})"
            for path in graph_reasoning["paths"][:5]
        ]
        context_parts.append("=== Graph Paths ===\n" + "\n".join(f"  • {path}" for path in path_strs))

    if graph_reasoning.get("bridge_entities"):
        bridge_strs = [
            f"{_format_bridge_entity_for_answer(bridge) or bridge['text']} (score: {bridge.get('score', 0)}; paths: {bridge.get('path_count', 0)})"
            for bridge in graph_reasoning["bridge_entities"][:5]
        ]
        context_parts.append("=== Bridge Entities ===\n" + "\n".join(f"  • {bridge}" for bridge in bridge_strs))

    if relationships:
        rel_strs = [f"{r['source']} --[{r['relation']}]--> {r['target']}" for r in relationships[:10]]
        context_parts.append("Extracted relationships:\n" + "\n".join(f"  • {r}" for r in rel_strs))

    context_text = ""
    if context_parts:
        context_text = _sanitize_llm_context_text(
            "\n\n--- Unified Knowledge Context ---\n"
            + "\n\n".join(context_parts)
            + "\n--- End Knowledge Context ---\n\n"
            "Use the above context to answer the user's question. "
            "Reference specific chunks, entities, relationships, and graph connections when relevant."
        )

    # ── Quality signals ────────────────────────────────────────────
    avg_chunk_relevance = 0.0
    if rag_chunks:
        avg_chunk_relevance = round(sum(c["score"] for c in rag_chunks) / len(rag_chunks), 3)

    entity_names_in_graph = sum(1 for e in entities if e.get("source") != "graph_expansion")
    entity_coverage = round(
        len([e for e in entities if e.get("source") == "graph_expansion"]) / max(entity_names_in_graph, 1), 3
    )
    graph_paths = [path for path in graph_reasoning.get("paths", []) if isinstance(path, dict)]
    direct_graph_relationships = [
        relationship
        for relationship in graph_reasoning.get("direct_relationships", [])
        if isinstance(relationship, dict)
    ]
    matched_graph_query_terms = _unique_reasoning_terms([
        *[
            term
            for path in graph_paths
            for term in (path.get("matched_query_terms") or [])
        ],
        *[
            term
            for relationship in direct_graph_relationships
            for term in (relationship.get("matched_query_terms") or [])
        ],
    ], min_length=2)[:12]
    graph_sources = graph_reasoning.get("graph_sources") if isinstance(graph_reasoning.get("graph_sources"), dict) else {}
    local_pipeline_sources = graph_sources.get("local_pipeline_runs") if isinstance(graph_sources.get("local_pipeline_runs"), list) else []
    interaction_handoffs = graph_reasoning.get("interaction_handoffs") if isinstance(graph_reasoning.get("interaction_handoffs"), list) else []
    answer_plan = graph_reasoning.get("answer_plan") if isinstance(graph_reasoning.get("answer_plan"), dict) else {}
    answer_evidence_cards = answer_plan.get("evidence_cards") if isinstance(answer_plan.get("evidence_cards"), list) else []
    graph_overview = graph_reasoning.get("graph_overview") if isinstance(graph_reasoning.get("graph_overview"), dict) else {}
    pipeline_evidence = graph_reasoning.get("pipeline_evidence") if isinstance(graph_reasoning.get("pipeline_evidence"), dict) else {}
    pipeline_evidence_counts = pipeline_evidence.get("counts") if isinstance(pipeline_evidence.get("counts"), dict) else {}
    pipeline_evidence_families = [
        str(family)
        for family in pipeline_evidence.get("families", []) or []
        if family
    ]
    overview_relationships = graph_overview.get("top_relationships") if isinstance(graph_overview.get("top_relationships"), list) else []
    overview_shared_entities = graph_overview.get("shared_entities") if isinstance(graph_overview.get("shared_entities"), list) else []
    overview_bridge_entities = graph_overview.get("bridge_entities") if isinstance(graph_overview.get("bridge_entities"), list) else []
    overview_paths = graph_overview.get("cross_document_paths") if isinstance(graph_overview.get("cross_document_paths"), list) else []

    quality_signals = {
        "avg_chunk_relevance": avg_chunk_relevance,
        "entity_coverage": entity_coverage,
        "graph_expansion_count": len(expanded_entities),
        "cross_reference_count": len(cross_references),
        "graph_path_count": len(graph_paths),
        "graph_layout_path_count": len([path for path in graph_paths if path.get("path_type") == "layout"]),
        "graph_semantic_path_count": len([path for path in graph_paths if path.get("path_type") != "layout"]),
        "bridge_entity_count": len(graph_reasoning.get("bridge_entities", [])),
        "direct_graph_relationship_count": len(direct_graph_relationships),
        "graph_overview_relationship_count": int(graph_overview.get("relationship_count", 0) or 0),
        "graph_overview_top_relationship_count": len(overview_relationships),
        "graph_overview_shared_entity_count": len(overview_shared_entities),
        "graph_overview_bridge_entity_count": len(overview_bridge_entities),
        "graph_overview_cross_document_path_count": len(overview_paths),
        "graph_answer_strategy": answer_plan.get("strategy"),
        "graph_answer_evidence_card_count": len(answer_evidence_cards),
        "graph_answer_provenance_anchor_count": len(answer_plan.get("provenance_anchors", []) or []),
        "pipeline_evidence_family_count": len(pipeline_evidence_families),
        "pipeline_evidence_total_count": sum(_positive_int(value) for value in pipeline_evidence_counts.values()),
        "pipeline_evidence_top_families": pipeline_evidence_families[:12],
        "pipeline_evidence_guidance_count": len(pipeline_evidence.get("guidance", []) or []),
        "requested_evidence_families": requested_evidence_families,
        "requested_evidence_card_count": len(requested_evidence_cards),
        "local_pipeline_requested_evidence_chunk_count": len([
            chunk for chunk in rag_chunks
            if "local_pipeline_requested_evidence" in (chunk.get("retrieval_reasons") or [])
        ]),
        "graph_matched_query_terms": matched_graph_query_terms,
        "graph_evidence_chunk_count": len([chunk for chunk in rag_chunks if chunk.get("graph_reasoning_boost", 0) > 0]),
        "graph_augmented_query_used": rag_enabled and graph_augmented_query != query,
        "graph_augmented_retrieval_terms": graph_augmented_terms,
        "supplemental_chunk_count": len(supplemental_rag_chunks),
        "graph_augmented_chunk_count": len([
            chunk for chunk in rag_chunks
            if "graph_augmented" in (chunk.get("retrieval_reasons") or [])
        ]),
        "local_pipeline_context_chunk_count": len(local_pipeline_chunks),
        "local_pipeline_evidence_context_chunk_count": len([
            chunk for chunk in rag_chunks
            if "local_pipeline_evidence_context" in (chunk.get("retrieval_reasons") or [])
        ]),
        "local_pipeline_chunk_evidence_chunk_count": len([
            chunk for chunk in rag_chunks
            if "local_pipeline_chunk_evidence" in (chunk.get("retrieval_reasons") or [])
        ]),
        "local_pipeline_graph_context_chunk_count": len([
            chunk for chunk in rag_chunks
            if _chunk_graph_context_items(chunk)
        ]),
        "local_pipeline_graph_context_relationship_count": sum(
            len(_chunk_graph_context_items(chunk))
            for chunk in rag_chunks
        ),
        "local_pipeline_graph_context_matched_chunk_count": len([
            chunk for chunk in rag_chunks
            if "local_pipeline_graph_context" in (chunk.get("retrieval_reasons") or [])
        ]),
        "local_pipeline_graph_path_context_chunk_count": len([
            chunk for chunk in rag_chunks
            if _chunk_graph_path_items(chunk)
        ]),
        "local_pipeline_graph_path_context_count": sum(
            len(_chunk_graph_path_items(chunk))
            for chunk in rag_chunks
        ),
        "local_pipeline_graph_path_context_matched_chunk_count": len([
            chunk for chunk in rag_chunks
            if "local_pipeline_graph_path_context" in (chunk.get("retrieval_reasons") or [])
        ]),
        "local_pipeline_context_run_count": len({
            str((chunk.get("metadata") or {}).get("run_id") or chunk.get("document_id") or "")
            for chunk in local_pipeline_chunks
            if isinstance(chunk, dict)
        } - {""}),
        "local_pipeline_artifact_run_count": int(pipeline_artifacts.get("run_count", 0) or 0),
        "local_pipeline_artifact_count": int(pipeline_artifacts.get("artifact_count", 0) or 0),
        "local_pipeline_structured_artifact_count": int(pipeline_artifacts.get("structured_artifact_count", 0) or 0),
        "local_pipeline_source_artifact_count": int(pipeline_artifacts.get("source_artifact_count", 0) or 0),
        "remote_graph_used": bool(graph_sources.get("remote_graph")),
        "local_pipeline_graph_run_count": len(local_pipeline_sources),
        "local_pipeline_graph_edge_count": sum(
            int(source.get("edge_count", 0) or 0)
            for source in local_pipeline_sources
            if isinstance(source, dict)
        ),
        "local_pipeline_interaction_handoff_count": len(interaction_handoffs),
        "local_pipeline_handoff_graph_anchor_count": sum(
            len(handoff.get("graph_anchors", []) or [])
            for handoff in interaction_handoffs
            if isinstance(handoff, dict)
        ),
        "local_pipeline_handoff_capability_card_count": sum(
            len(handoff.get("capability_cards", []) or [])
            for handoff in interaction_handoffs
            if isinstance(handoff, dict)
        ),
        "local_pipeline_handoff_starter_prompt_count": sum(
            len(handoff.get("starter_prompts", []) or [])
            for handoff in interaction_handoffs
            if isinstance(handoff, dict)
        ),
        "local_pipeline_handoff_retrieval_rehearsal_count": sum(
            _handoff_count(handoff, "retrieval_rehearsals") or len(handoff.get("retrieval_rehearsals", []) or [])
            for handoff in interaction_handoffs
            if isinstance(handoff, dict)
        ),
        "local_pipeline_handoff_rehearsed_chunk_match_count": sum(
            _handoff_rehearsal_match_count(handoff, "rehearsed_chunk_matches", "matched_chunk_count")
            for handoff in interaction_handoffs
            if isinstance(handoff, dict)
        ),
        "local_pipeline_handoff_rehearsed_anchor_match_count": sum(
            _handoff_rehearsal_match_count(handoff, "rehearsed_anchor_matches", "matched_anchor_count")
            for handoff in interaction_handoffs
            if isinstance(handoff, dict)
        ),
        "local_pipeline_handoff_modes": _unique_reasoning_terms([
            str(handoff.get("recommended_mode") or "")
            for handoff in interaction_handoffs
            if isinstance(handoff, dict)
        ], min_length=3),
        "local_pipeline_handoff_citation_scopes": _unique_reasoning_terms([
            str(scope)
            for handoff in interaction_handoffs
            if isinstance(handoff, dict)
            for scope in (handoff.get("citation_scopes", []) or [])
        ], min_length=2),
        "total_chunks": len(rag_chunks),
        "total_entities": len(entities),
    }

    response_payload = {
        "chunks": rag_chunks,
        "entities": entities,
        "relationships": relationships,
        "search_results": graph_search_results,
        "graph_connections": graph_connections,
        "graph_reasoning": graph_reasoning,
        "pipeline_artifacts": pipeline_artifacts,
        "pipeline_evidence": pipeline_evidence,
        "requested_evidence_cards": requested_evidence_cards,
        "expanded_entities": expanded_entities,
        "cross_references": cross_references,
        "context_text": context_text,
        "quality_signals": quality_signals,
        "source": "unified",
        "services_used": {
            "ner": cfg.ner_enabled,
            "rel": cfg.rel_enabled,
            "rag": rag_enabled and len(rag_chunks) > 0,
            "vector_search": cfg.has_graph_service,
            "graph_expansion": cfg.has_graph_service and len(expanded_entities) > 0,
            "graph_reasoning": graph_reasoning.get("status") == "complete",
            "local_pipeline_graph": bool((graph_reasoning.get("graph_sources") or {}).get("local_pipeline_runs")),
            "local_pipeline_interaction_handoff": len(interaction_handoffs) > 0,
            "local_pipeline_context": len(local_pipeline_chunks) > 0,
            "local_pipeline_artifacts": bool(pipeline_artifacts.get("run_count")),
            "pipeline_evidence_handoff": bool(pipeline_evidence),
        },
        "errors": errors if errors else None,
    }
    if _compact_chat_context_requested(data if isinstance(data, dict) else {}):
        response_payload = _compact_chat_context_response(
            response_payload,
            data if isinstance(data, dict) else {},
        )
    return response_payload
