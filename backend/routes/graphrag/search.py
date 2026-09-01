"""
GraphRAG Search & Reasoning Routes

Search, advanced search, intelligent search, multi-hop reasoning,
relationship explanation, causal/comparative reasoning, code search.

All endpoints use the shared connection-pooled HTTP client.
"""

from fastapi import APIRouter, HTTPException, Request
import httpx
import logging
from pathlib import PurePath
import re
from typing import Any

from .helpers import check_graphrag_enabled, get_http_client, graphrag_url as _graphrag_url
from modules.graphrag.config import get_graphrag_config

from .context import (
    _edge_query_score,
    _entity_pairs_from_query,
    _find_graph_paths,
    _format_graph_relationship,
    _local_pipeline_context_chunks,
    _local_pipeline_graph_data,
    _normalize_graph_edge,
    _pipeline_graph_overview,
    _relationship_provenance,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _string_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _bounded_search_limit(value: Any, *, default: int = 10, minimum: int = 1, maximum: int = 25) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _result_key(result: dict[str, Any]) -> tuple[str, str, str]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    return (
        str(result.get("result_type") or result.get("type") or "").lower(),
        str(result.get("source") or metadata.get("run_id") or metadata.get("filename") or "").lower(),
        str(result.get("content") or result.get("text") or "")[:240].lower(),
    )


def _local_pipeline_search_response(
    *,
    query: str,
    domain: str | None,
    top_k: int,
    pipeline_run_ids: list[str] | None = None,
    source: str | None = None,
    target: str | None = None,
    remote_error: str | None = None,
) -> dict[str, Any]:
    node_map, raw_edges, graph_sources = _local_pipeline_graph_data(
        domain,
        pipeline_run_ids=pipeline_run_ids,
        max_edges=max(100, top_k * 20),
    )
    path_pairs: list[tuple[str, str]] = []
    if source and target:
        path_pairs.append((str(source), str(target)))
    for pair in _entity_pairs_from_query(query):
        if pair not in path_pairs:
            path_pairs.append(pair)

    results: list[dict[str, Any]] = []
    path_records: list[dict[str, Any]] = []
    for pair_source, pair_target in path_pairs[:4]:
        for path in _find_graph_paths(
            pair_source,
            pair_target,
            node_map,
            raw_edges,
            max_hops=3,
            limit=3,
            query=query,
        ):
            path_records.append(path)
            provenance = "; ".join(
                _relationship_provenance(rel.get("metadata"))
                for rel in path.get("relationships", [])
                if isinstance(rel, dict) and _relationship_provenance(rel.get("metadata"))
            )
            results.append({
                "content": path.get("text") or " -> ".join(str(node) for node in path.get("nodes", []) if node),
                "source": ", ".join(path.get("nodes", [])[:1]) or pair_source,
                "score": path.get("score", 0),
                "result_type": "local_pipeline_graph_path",
                "metadata": {
                    "source_type": "local_pipeline_run",
                    "path_type": path.get("path_type"),
                    "hops": path.get("hops"),
                    "nodes": path.get("nodes"),
                    "matched_query_terms": path.get("matched_query_terms", []),
                    "relationships": path.get("relationships", []),
                    "provenance": provenance,
                    "pipeline_run_ids": sorted({
                        str(rel.get("metadata", {}).get("local_graph_run_id"))
                        for rel in path.get("relationships", [])
                        if isinstance(rel, dict)
                        and isinstance(rel.get("metadata"), dict)
                        and rel.get("metadata", {}).get("local_graph_run_id")
                    }),
                },
            })

    edge_hits: list[dict[str, Any]] = []
    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        normalized = _normalize_graph_edge(edge, node_map)
        score = _edge_query_score(normalized, query)
        if score <= 0:
            continue
        normalized["score"] = round(min(1.0, score + (float(normalized.get("weight", 0) or 0) * 0.18)), 3)
        edge_hits.append(normalized)
    edge_hits = sorted(edge_hits, key=lambda item: item.get("score", 0), reverse=True)[:top_k]
    for edge in edge_hits:
        metadata = edge.get("metadata") if isinstance(edge.get("metadata"), dict) else {}
        provenance = _relationship_provenance(metadata)
        content = _format_graph_relationship(edge)
        if edge.get("evidence"):
            content = f"{content}: {edge['evidence']}"
        results.append({
            "content": content,
            "source": metadata.get("local_graph_filename") or edge.get("source"),
            "score": edge.get("score", 0),
            "result_type": "local_pipeline_graph_edge",
            "metadata": {
                **metadata,
                "source_type": "local_pipeline_run",
                "source": edge.get("source"),
                "target": edge.get("target"),
                "relation": edge.get("relation"),
                "relationship_type": "layout" if str(edge.get("relation") or "").startswith(("has_", "contains_")) else "semantic",
                "provenance": provenance,
            },
        })

    graph_reasoning = {
        "paths": path_records,
        "direct_relationships": edge_hits,
        "graph_sources": {"local_pipeline_runs": graph_sources},
    }
    chunk_hits = _local_pipeline_context_chunks(
        query,
        domain,
        top_k,
        graph_reasoning=graph_reasoning,
        pipeline_run_ids=pipeline_run_ids,
    )
    for chunk in chunk_hits:
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        results.append({
            "content": chunk.get("content") or "",
            "source": metadata.get("filename") or chunk.get("document_title") or chunk.get("document_id"),
            "score": chunk.get("score", 0),
            "result_type": "local_pipeline_chunk",
            "metadata": {
                **metadata,
                "source_type": "local_pipeline_run",
                "document_id": chunk.get("document_id"),
                "chunk_index": chunk.get("chunk_index"),
                "retrieval_reasons": chunk.get("retrieval_reasons", []),
                "retrieval_queries": chunk.get("retrieval_queries", []),
            },
        })

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for result in sorted(results, key=lambda item: float(item.get("score", 0) or 0), reverse=True):
        key = _result_key(result)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
        if len(deduped) >= top_k:
            break

    graph_overview = _pipeline_graph_overview(node_map, raw_edges, graph_sources, query=query, limit=min(12, top_k))
    response = {
        "query": query,
        "results": deduped,
        "total_results": len(deduped),
        "source": "local_pipeline",
        "search_method_used": "local_pipeline_graph",
        "local_pipeline": {
            "ready": bool(graph_sources or chunk_hits),
            "run_count": len(graph_sources),
            "graph_edge_count": len(raw_edges),
            "matched_graph_edge_count": len(edge_hits),
            "matched_chunk_count": len(chunk_hits),
            "matched_path_count": len(path_records),
            "pipeline_run_ids": [source.get("run_id") for source in graph_sources if source.get("run_id")],
            "fallback_reason": remote_error,
        },
        "graph_overview": graph_overview,
    }
    if remote_error:
        response["warnings"] = [f"Remote GraphRAG search unavailable; returned local pipeline results. {remote_error}"]
    return response


def _merge_remote_and_local_search(
    remote_payload: dict[str, Any],
    local_payload: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    remote_results = remote_payload.get("results") if isinstance(remote_payload.get("results"), list) else []
    local_results = local_payload.get("results") if isinstance(local_payload.get("results"), list) else []
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for index, result in enumerate(remote_results):
        if not isinstance(result, dict):
            continue
        normalized = dict(result)
        normalized.setdefault("result_type", normalized.get("type") or "remote_graphrag")
        normalized.setdefault("score", normalized.get("score", normalized.get("similarity", 0)))
        normalized.setdefault("source", normalized.get("source") or normalized.get("document_id") or "graphrag")
        key = _result_key(normalized)
        if key not in seen:
            seen.add(key)
            merged.append(normalized)

    for result in local_results:
        key = _result_key(result)
        if key not in seen:
            seen.add(key)
            merged.append(result)

    merged = sorted(merged, key=lambda item: float(item.get("score", 0) or 0), reverse=True)[:top_k]
    return {
        **remote_payload,
        "results": merged,
        "total_results": len(merged),
        "source": "remote_plus_local_pipeline",
        "local_pipeline": local_payload.get("local_pipeline", {}),
        "local_pipeline_results": local_results,
        "graph_overview": local_payload.get("graph_overview") or remote_payload.get("graph_overview"),
    }


def _bounded_reasoning_hops(value: Any, *, default: int = 3) -> int:
    return _bounded_search_limit(value, default=default, minimum=1, maximum=6)


def _clean_match_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold())).strip()


def _text_match_score(needle: Any, *haystacks: Any) -> float:
    cleaned_needle = _clean_match_text(needle)
    if not cleaned_needle:
        return 0.0

    needle_terms = {term for term in cleaned_needle.split() if len(term) >= 2}
    best = 0.0
    for haystack in haystacks:
        cleaned_haystack = _clean_match_text(haystack)
        if not cleaned_haystack:
            continue
        if cleaned_needle == cleaned_haystack:
            best = max(best, 1.0)
            continue
        if cleaned_needle in cleaned_haystack or cleaned_haystack in cleaned_needle:
            best = max(best, 0.82)
            continue
        haystack_terms = {term for term in cleaned_haystack.split() if len(term) >= 2}
        if needle_terms and haystack_terms:
            best = max(best, len(needle_terms & haystack_terms) / max(len(needle_terms), 1))
    return round(min(1.0, best), 3)


def _edge_match_texts(edge: dict[str, Any]) -> list[Any]:
    metadata = edge.get("metadata") if isinstance(edge.get("metadata"), dict) else {}
    return [
        edge.get("source"),
        edge.get("target"),
        edge.get("relation"),
        edge.get("evidence"),
        metadata.get("slide_title"),
        metadata.get("connector_title"),
        metadata.get("connector_description"),
        metadata.get("source_label"),
        metadata.get("target_label"),
        metadata.get("shape_title"),
        metadata.get("shape_description"),
        metadata.get("chart_title"),
        metadata.get("chart_label"),
        metadata.get("series_name"),
        metadata.get("category"),
        metadata.get("table_name"),
        metadata.get("local_graph_filename"),
    ]


def _relationship_endpoint_score(edge: dict[str, Any], source: str | None, target: str | None) -> float:
    if not source or not target:
        return 0.0

    source_fields = [
        edge.get("source"),
        edge.get("evidence"),
        *_edge_match_texts(edge),
    ]
    target_fields = [
        edge.get("target"),
        edge.get("evidence"),
        *_edge_match_texts(edge),
    ]
    forward = (_text_match_score(source, *source_fields) + _text_match_score(target, *target_fields)) / 2
    reverse = (_text_match_score(source, *target_fields) + _text_match_score(target, *source_fields)) / 2
    return round(max(forward, reverse), 3)


def _reasoning_path_steps(path: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for index, relationship in enumerate(path.get("relationships", []) or [], start=1):
        if not isinstance(relationship, dict):
            continue
        metadata = relationship.get("metadata") if isinstance(relationship.get("metadata"), dict) else {}
        confidence = relationship.get("weight", path.get("average_weight", path.get("score", 0)))
        steps.append({
            "hop": index,
            "entity": relationship.get("target") or relationship.get("source") or f"Hop {index}",
            "relation": relationship.get("relation") or "related_to",
            "source": relationship.get("source"),
            "target": relationship.get("target"),
            "direction": relationship.get("direction"),
            "confidence": round(float(confidence or 0), 3),
            "evidence": relationship.get("evidence"),
            "provenance": relationship.get("provenance") or _relationship_provenance(metadata),
            "metadata": metadata,
        })
    return steps


def _bridge_entities_from_reasoning_paths(paths: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    bridges: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not isinstance(path, dict):
            continue
        nodes = [str(node) for node in path.get("nodes", []) or [] if node]
        if len(nodes) <= 2:
            continue
        for bridge_name in nodes[1:-1]:
            key = bridge_name.casefold()
            bridge = bridges.setdefault(key, {
                "name": bridge_name,
                "path_count": 0,
                "score": 0.0,
                "connects": [],
                "path_types": [],
            })
            bridge["path_count"] += 1
            bridge["score"] = max(float(bridge.get("score", 0) or 0), float(path.get("score", 0) or 0))
            connects = {"source": path.get("source") or nodes[0], "target": path.get("target") or nodes[-1]}
            if connects not in bridge["connects"]:
                bridge["connects"].append(connects)
            path_type = path.get("path_type")
            if path_type and path_type not in bridge["path_types"]:
                bridge["path_types"].append(path_type)

    return sorted(bridges.values(), key=lambda item: (item.get("score", 0), item.get("path_count", 0)), reverse=True)[:limit]


def _local_reasoning_sources(
    paths: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    graph_sources: list[dict[str, Any]],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for path in paths[:4]:
        if not isinstance(path, dict):
            continue
        key = ("path", str(path.get("text") or path.get("nodes")))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "content": path.get("text") or " -> ".join(str(node) for node in path.get("nodes", []) if node),
            "source": path.get("source"),
            "target": path.get("target"),
            "score": path.get("score", 0),
            "source_type": "local_pipeline_graph_path",
            "metadata": {
                "nodes": path.get("nodes", []),
                "relationships": path.get("relationships", []),
                "path_type": path.get("path_type"),
            },
        })
        if len(sources) >= limit:
            return sources

    for relationship in relationships[:6]:
        if not isinstance(relationship, dict):
            continue
        metadata = relationship.get("metadata") if isinstance(relationship.get("metadata"), dict) else {}
        content = _format_graph_relationship(relationship)
        if relationship.get("evidence"):
            content = f"{content}: {relationship['evidence']}"
        key = ("relationship", content)
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "content": content,
            "source": metadata.get("local_graph_filename") or relationship.get("source"),
            "score": relationship.get("score", relationship.get("weight", 0)),
            "source_type": "local_pipeline_graph_edge",
            "metadata": metadata,
        })
        if len(sources) >= limit:
            return sources

    for graph_source in graph_sources:
        if not isinstance(graph_source, dict):
            continue
        filename = graph_source.get("filename") or graph_source.get("run_id") or "local pipeline run"
        key = ("run", str(graph_source.get("run_id") or filename))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "content": f"Local pipeline graph trace for {filename}",
            "source": filename,
            "score": 1.0,
            "source_type": "local_pipeline_run",
            "metadata": graph_source,
        })
        if len(sources) >= limit:
            break

    return sources


def _local_pipeline_reasoning_response(
    *,
    query: str,
    domain: str | None,
    source: str | None = None,
    target: str | None = None,
    max_hops: int = 3,
    pipeline_run_ids: list[str] | None = None,
    mode: str = "multi_hop",
    remote_error: str | None = None,
) -> dict[str, Any]:
    effective_query = str(query or "").strip()
    if not effective_query and source and target:
        effective_query = f"Find connection between {source} and {target}"

    node_map, raw_edges, graph_sources = _local_pipeline_graph_data(
        domain,
        pipeline_run_ids=pipeline_run_ids,
        max_edges=max(200, max_hops * 80),
    )
    path_pairs: list[tuple[str, str]] = []
    if source and target:
        path_pairs.append((str(source), str(target)))
    for pair in _entity_pairs_from_query(effective_query):
        if pair not in path_pairs:
            path_pairs.append(pair)

    path_records: list[dict[str, Any]] = []
    seen_paths: set[tuple[str, ...]] = set()
    for pair_source, pair_target in path_pairs[:5]:
        for path in _find_graph_paths(
            pair_source,
            pair_target,
            node_map,
            raw_edges,
            max_hops=max_hops,
            limit=4,
            query=effective_query,
        ):
            key = tuple(str(node).casefold() for node in path.get("nodes", []) if node)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            path_records.append(path)

    relationship_hits: list[dict[str, Any]] = []
    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        normalized = _normalize_graph_edge(edge, node_map)
        query_score = _edge_query_score(normalized, effective_query)
        endpoint_score = _relationship_endpoint_score(normalized, source, target)
        if query_score <= 0 and endpoint_score <= 0:
            continue
        normalized["score"] = round(
            min(
                1.0,
                (float(normalized.get("weight", 0) or 0) * 0.35)
                + (query_score * 0.3)
                + (endpoint_score * 0.35),
            ),
            3,
        )
        normalized["relationship_type"] = "layout" if str(normalized.get("relation") or "").startswith(("has_", "contains_")) else "semantic"
        relationship_hits.append(normalized)
    relationship_hits = sorted(relationship_hits, key=lambda item: item.get("score", 0), reverse=True)[:12]

    path_records = sorted(path_records, key=lambda item: item.get("score", 0), reverse=True)[:8]
    strongest_path = path_records[0] if path_records else None
    reasoning_path = _reasoning_path_steps(strongest_path) if strongest_path else []
    bridge_entities = _bridge_entities_from_reasoning_paths(path_records)
    graph_overview = _pipeline_graph_overview(node_map, raw_edges, graph_sources, query=effective_query, limit=10)
    sources = _local_reasoning_sources(path_records, relationship_hits, graph_sources)

    if strongest_path:
        answer = (
            f"Found {len(path_records)} local pipeline graph path"
            f"{'' if len(path_records) == 1 else 's'}. Strongest path: {strongest_path.get('text')}"
        )
    elif relationship_hits:
        answer = (
            f"Found {len(relationship_hits)} local pipeline relationship"
            f"{'' if len(relationship_hits) == 1 else 's'} that match the request."
        )
    else:
        answer = "No local pipeline graph path was found for the requested entities."

    response: dict[str, Any] = {
        "query": effective_query,
        "source_entity": source,
        "target_entity": target,
        "reasoning_source": "local_pipeline",
        "reasoning_task": mode,
        "reasoning_method": "local_pipeline_graph_paths",
        "reasoning_path": reasoning_path,
        "reasoning_paths": path_records,
        "paths": path_records,
        "direct_relationships": relationship_hits,
        "bridge_entities": bridge_entities,
        "answer": answer,
        "sources": sources,
        "graph_overview": graph_overview,
        "local_pipeline": {
            "ready": bool(graph_sources or path_records or relationship_hits),
            "run_count": len(graph_sources),
            "graph_edge_count": len(raw_edges),
            "matched_path_count": len(path_records),
            "matched_relationship_count": len(relationship_hits),
            "pipeline_run_ids": [item.get("run_id") for item in graph_sources if item.get("run_id")],
            "fallback_reason": remote_error,
        },
    }
    if mode == "explain":
        response.update({
            "source": source,
            "target": target,
            "relationship": _format_graph_relationship(relationship_hits[0]) if relationship_hits else None,
            "explanation": answer,
            "evidence": sources,
            "confidence": (
                strongest_path.get("score")
                if strongest_path
                else relationship_hits[0].get("score", 0)
                if relationship_hits
                else 0
            ),
        })
    if remote_error:
        response["warnings"] = [f"Remote GraphRAG reasoning unavailable; returned local pipeline graph evidence. {remote_error}"]
    return response


def _attach_local_reasoning(
    remote_payload: dict[str, Any],
    local_payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(remote_payload, dict):
        return remote_payload
    merged = {
        **remote_payload,
        "local_pipeline_reasoning": local_payload,
    }
    remote_sources = remote_payload.get("sources") if isinstance(remote_payload.get("sources"), list) else []
    local_sources = local_payload.get("sources") if isinstance(local_payload.get("sources"), list) else []
    if local_sources:
        merged["sources"] = [*remote_sources, *local_sources]
    if "reasoning_path" not in merged and local_payload.get("reasoning_path"):
        merged["reasoning_path"] = local_payload["reasoning_path"]
    return merged


_LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".cs": "csharp",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
}

_LANGUAGE_PATTERNS = (
    ("python", ("def ", "class ", "import ", "from ", "if __name__")),
    ("javascript", ("function ", "const ", "let ", "var ", "console.log", "export ")),
    ("typescript", ("interface ", "type ", ": string", ": number", "implements ")),
    ("java", ("public class", "import java", "package ", "public static void main")),
    ("go", ("package main", "func ", "import \"", "fmt.")),
    ("rust", ("fn main", "use std::", "println!", "let mut ")),
    ("cpp", ("#include <iostream>", "std::", "using namespace std")),
    ("c", ("#include <stdio.h>", "int main(", "printf(")),
    ("php", ("<?php", "$", "echo ")),
    ("ruby", ("def ", "puts ", "require ", "end")),
)


def _detect_text_code(text: str, filename: str = "") -> dict:
    """Local text adapter for the UI's code detection contract."""
    extension = PurePath(filename or "").suffix.lower()
    language = _LANGUAGE_BY_EXTENSION.get(extension)
    confidence = 0.98 if language else 0.0

    if not language:
        scores = []
        for candidate, patterns in _LANGUAGE_PATTERNS:
            matches = sum(1 for pattern in patterns if pattern in text)
            if matches:
                scores.append((matches, candidate))
        if scores:
            matches, language = max(scores)
            confidence = min(0.95, 0.58 + matches * 0.12)

    has_code = language is not None
    code_type = "snippet"
    if language == "python" and "class " in text:
        code_type = "class"
    elif language and any(token in text for token in ("def ", "function ", "func ", "fn ")):
        code_type = "function"

    blocks = []
    if has_code:
        blocks.append({
            "language": language,
            "content": text,
            "type": code_type,
            "metadata": {
                "filename": filename,
                "confidence": round(confidence, 3),
                "source": "llama-nexus-adapter",
            },
        })

    by_language = {language: 1} if has_code else {}
    by_type = {code_type: 1} if has_code else {}

    return {
        "has_code": has_code,
        "is_code": has_code,
        "language": language,
        "languages": [language] if has_code else [],
        "confidence": round(confidence, 3),
        "suggested_processor": "code" if has_code else "standard",
        "code_blocks": blocks,
        "statistics": {
            "total_blocks": len(blocks),
            "by_language": by_language,
            "by_type": by_type,
        },
    }


def _normalize_detection_response(data: dict, fallback_text: str = "", filename: str = "") -> dict:
    """Normalize upstream file-level detection into the UI text-detection shape."""
    if "has_code" in data and "code_blocks" in data and "statistics" in data:
        return data

    file_info = data.get("file_info") or {}
    is_code = bool(data.get("is_code", file_info.get("is_code", False)))
    language = data.get("language") or file_info.get("language")
    confidence = data.get("confidence", 0.95 if is_code else 0.0)
    code_type = data.get("code_type") or "snippet"

    blocks = []
    if is_code:
        blocks.append({
            "language": language or "text",
            "content": fallback_text,
            "type": code_type,
            "metadata": {
                "filename": filename or data.get("filename") or file_info.get("file_name"),
                "file_info": file_info,
                "source": "graphrag",
            },
        })

    normalized = {
        **data,
        "has_code": is_code,
        "is_code": is_code,
        "language": language,
        "languages": [language] if is_code and language else [],
        "confidence": confidence,
        "suggested_processor": data.get("suggested_processor", "code" if is_code else "standard"),
        "code_blocks": blocks,
        "statistics": {
            "total_blocks": len(blocks),
            "by_language": {language: len(blocks)} if language and blocks else {},
            "by_type": {code_type: len(blocks)} if blocks else {},
        },
    }
    return normalized


def _normalize_code_search_response(data: dict) -> dict:
    """Normalize Code-RAG nested entity results for the llama-nexus code UI."""
    normalized_results = []

    for item in data.get("results", []):
        if not isinstance(item, dict):
            continue
        entity = item.get("entity") if isinstance(item.get("entity"), dict) else None
        if entity:
            code_type = entity.get("entity_type") or entity.get("type") or "other"
            content_parts = []
            if entity.get("docstring"):
                content_parts.append(str(entity["docstring"]))
            if entity.get("parameters"):
                params = ", ".join(
                    p.get("name", "") for p in entity.get("parameters", [])
                    if isinstance(p, dict)
                )
                content_parts.append(f"{entity.get('name', '')}({params})")
            if not content_parts:
                content_parts.append(entity.get("name") or item.get("explanation") or "")

            normalized_results.append({
                "file_path": entity.get("file_path", ""),
                "language": entity.get("language", item.get("language", "unknown")),
                "code_type": code_type,
                "name": entity.get("name"),
                "content": "\n".join(part for part in content_parts if part),
                "line_start": entity.get("line_start", 1),
                "line_end": entity.get("line_end", entity.get("line_start", 1)),
                "score": item.get("score", 0),
                "context": item.get("explanation", ""),
                "metadata": item.get("metadata", {}),
            })
            continue

        normalized_results.append({
            "file_path": item.get("file_path") or item.get("source") or "",
            "language": item.get("language", "unknown"),
            "code_type": item.get("code_type") or item.get("result_type") or "other",
            "name": item.get("name"),
            "content": item.get("content") or item.get("code") or "",
            "line_start": item.get("line_start", 1),
            "line_end": item.get("line_end", item.get("line_start", 1)),
            "score": item.get("score", 0),
            "context": item.get("context") or item.get("explanation", ""),
            "metadata": item.get("metadata", {}),
        })

    return {
        **data,
        "results": normalized_results,
        "total_results": data.get("total_results", len(normalized_results)),
    }


@router.post("/search")
async def graphrag_search(request: Request):
    """Hybrid search using graphrag service (vector + graph + keyword)."""
    check_graphrag_enabled()

    data = await request.json()
    query = str(data.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    top_k = _bounded_search_limit(data.get("top_k", data.get("limit", 10)))
    domain = data.get("domain")
    pipeline_run_ids = _string_list(data.get("pipeline_run_ids") or data.get("pipeline_run_id"))
    cfg = get_graphrag_config()
    use_local = bool(
        data.get("use_local_pipeline_context")
        or data.get("include_local_pipeline")
        or pipeline_run_ids
        or data.get("search_mode") == "local_pipeline"
        or data.get("local_only")
    )
    local_only = bool(data.get("search_mode") == "local_pipeline" or data.get("local_only") or not cfg.graphrag_url)

    if local_only:
        return _local_pipeline_search_response(
            query=query,
            domain=domain,
            top_k=top_k,
            pipeline_run_ids=pipeline_run_ids,
            source=data.get("source"),
            target=data.get("target"),
            remote_error=None if cfg.graphrag_url else "GRAPHRAG_URL is not configured.",
        )

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/search",
            data={
                "query": query,
                "top_k": str(top_k),
            },
        )
        response.raise_for_status()
        payload = response.json()
        if use_local and isinstance(payload, dict):
            local_payload = _local_pipeline_search_response(
                query=query,
                domain=domain,
                top_k=top_k,
                pipeline_run_ids=pipeline_run_ids,
                source=data.get("source"),
                target=data.get("target"),
            )
            if local_payload.get("results"):
                return _merge_remote_and_local_search(payload, local_payload, top_k=top_k)
        return payload
    except httpx.HTTPError as e:
        local_payload = _local_pipeline_search_response(
            query=query,
            domain=domain,
            top_k=top_k,
            pipeline_run_ids=pipeline_run_ids,
            source=data.get("source"),
            target=data.get("target"),
            remote_error=str(e),
        )
        if local_payload.get("results") or data.get("local_fallback", True):
            return local_payload
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/search/advanced")
async def graphrag_advanced_search(request: Request):
    """Advanced search with type selection using graphrag service."""
    check_graphrag_enabled()

    data = await request.json()
    query = str(data.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    top_k = _bounded_search_limit(data.get("top_k", data.get("limit", 10)))
    domain = data.get("domain")
    pipeline_run_ids = _string_list(data.get("pipeline_run_ids") or data.get("pipeline_run_id"))
    cfg = get_graphrag_config()
    use_local = bool(
        data.get("use_local_pipeline_context")
        or data.get("include_local_pipeline")
        or pipeline_run_ids
        or data.get("search_type") == "local_pipeline"
        or data.get("search_mode") == "local_pipeline"
        or data.get("local_only")
    )
    local_only = bool(
        data.get("search_type") == "local_pipeline"
        or data.get("search_mode") == "local_pipeline"
        or data.get("local_only")
        or not cfg.graphrag_url
    )
    if local_only:
        return _local_pipeline_search_response(
            query=query,
            domain=domain,
            top_k=top_k,
            pipeline_run_ids=pipeline_run_ids,
            source=data.get("source"),
            target=data.get("target"),
            remote_error=None if cfg.graphrag_url else "GRAPHRAG_URL is not configured.",
        )

    try:
        client = get_http_client()
        response = await client.post(f"{_graphrag_url()}/search-advanced", json=data)
        response.raise_for_status()
        payload = response.json()
        if use_local and isinstance(payload, dict):
            local_payload = _local_pipeline_search_response(
                query=query,
                domain=domain,
                top_k=top_k,
                pipeline_run_ids=pipeline_run_ids,
                source=data.get("source"),
                target=data.get("target"),
            )
            if local_payload.get("results"):
                return _merge_remote_and_local_search(payload, local_payload, top_k=top_k)
        return payload
    except httpx.HTTPError as e:
        local_payload = _local_pipeline_search_response(
            query=query,
            domain=domain,
            top_k=top_k,
            pipeline_run_ids=pipeline_run_ids,
            source=data.get("source"),
            target=data.get("target"),
            remote_error=str(e),
        )
        if local_payload.get("results") or data.get("local_fallback", True):
            return local_payload
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/query/intent")
async def analyze_query_intent(query: str):
    """Analyze query to determine intent, entities, and complexity."""
    check_graphrag_enabled()

    if not query:
        raise HTTPException(status_code=400, detail="query parameter is required")

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/api/analyze-query-intent",
            params={"query": query},
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPError as e:
        logger.error(f"Query intent analysis error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/query/complexity")
async def analyze_query_complexity(request: Request):
    """Analyze query complexity and required reasoning strategy."""
    check_graphrag_enabled()

    data = await request.json()
    query = data.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/api/query-complexity-analysis",
            params={"query": query},
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPError as e:
        logger.error(f"Query complexity analysis error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/search/intelligent")
async def intelligent_search(request: Request):
    """Intelligent search with automatic method selection and LLM answer generation."""
    check_graphrag_enabled()

    data = await request.json()
    query = str(data.get("query") or "").strip()

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    top_k = _bounded_search_limit(data.get("top_k", data.get("limit", 10)))
    pipeline_run_ids = _string_list(data.get("pipeline_run_ids") or data.get("pipeline_run_id"))
    cfg = get_graphrag_config()
    local_only = bool(
        data.get("search_type") == "local_pipeline"
        or data.get("search_mode") == "local_pipeline"
        or data.get("local_only")
        or not cfg.graphrag_url
    )
    if local_only:
        return _local_pipeline_search_response(
            query=query,
            domain=data.get("domain"),
            top_k=top_k,
            pipeline_run_ids=pipeline_run_ids,
            source=data.get("source"),
            target=data.get("target"),
            remote_error=None if cfg.graphrag_url else "GRAPHRAG_URL is not configured.",
        )

    try:
        client = get_http_client()
        search_type = data.get("search_type", "auto")

        if search_type == "auto":
            search_type = "hybrid"

        search_payload = {
            "query": query,
            "search_type": search_type,
            "top_k": top_k,
            "domain": data.get("domain"),
            "filters": data.get("filters"),
        }

        response = await client.post(
            f"{_graphrag_url()}/search-advanced",
            json=search_payload,
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()

        result["search_method_used"] = search_type
        if data.get("use_local_pipeline_context") or data.get("include_local_pipeline") or pipeline_run_ids:
            local_payload = _local_pipeline_search_response(
                query=query,
                domain=data.get("domain"),
                top_k=top_k,
                pipeline_run_ids=pipeline_run_ids,
                source=data.get("source"),
                target=data.get("target"),
            )
            if local_payload.get("results"):
                result = _merge_remote_and_local_search(result, local_payload, top_k=top_k)
                result["search_method_used"] = f"{search_type}+local_pipeline"

        return result

    except httpx.HTTPError as e:
        logger.error(f"Intelligent search error: {e}")
        local_payload = _local_pipeline_search_response(
            query=query,
            domain=data.get("domain"),
            top_k=top_k,
            pipeline_run_ids=pipeline_run_ids,
            source=data.get("source"),
            target=data.get("target"),
            remote_error=str(e),
        )
        if local_payload.get("results") or data.get("local_fallback", True):
            return local_payload
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")


@router.post("/reasoning/multi-hop")
async def graphrag_multi_hop_reasoning(request: Request):
    """Perform multi-hop reasoning query using graphrag service."""
    check_graphrag_enabled()

    data = await request.json()
    query = str(data.get("query") or "").strip()
    source = data.get("source")
    target = data.get("target")
    max_hops = _bounded_reasoning_hops(data.get("max_hops", data.get("maxHops", 3)))
    domain = data.get("domain")
    pipeline_run_ids = _string_list(data.get("pipeline_run_ids") or data.get("pipeline_run_id"))
    cfg = get_graphrag_config()
    use_local = bool(
        data.get("use_local_pipeline_context")
        or data.get("include_local_pipeline")
        or pipeline_run_ids
        or data.get("reasoning_mode") == "local_pipeline"
        or data.get("search_mode") == "local_pipeline"
        or data.get("local_only")
    )
    local_only = bool(
        data.get("reasoning_mode") == "local_pipeline"
        or data.get("search_mode") == "local_pipeline"
        or data.get("local_only")
        or not cfg.graphrag_url
    )

    if not query and not (source and target):
        raise HTTPException(status_code=400, detail="query or source/target is required")

    if local_only:
        return _local_pipeline_reasoning_response(
            query=query,
            domain=domain,
            source=source,
            target=target,
            max_hops=max_hops,
            pipeline_run_ids=pipeline_run_ids,
            remote_error=None if cfg.graphrag_url else "GRAPHRAG_URL is not configured.",
        )

    try:
        client = get_http_client()
        if source and target and ("entities" in data or "relationships" in data):
            response = await client.post(
                f"{_graphrag_url()}/reasoning/multi-hop",
                json={
                    "source": source,
                    "target": target,
                    "max_hops": max_hops,
                    "entities": data.get("entities", []),
                    "relationships": data.get("relationships", []),
                },
                timeout=120.0,
            )
        else:
            response = await client.post(
                f"{_graphrag_url()}/api/multi-hop-reasoning",
                params={
                    "query": query or f"Find connection between {source} and {target}",
                    "max_hops": max_hops,
                },
                timeout=120.0,
            )
        response.raise_for_status()
        result = response.json()
        if "reasoning_path" not in result and "reasoning_paths" in result:
            result["reasoning_path"] = result["reasoning_paths"]
        result.setdefault("sources", [])
        if use_local:
            local_payload = _local_pipeline_reasoning_response(
                query=query,
                domain=domain,
                source=source,
                target=target,
                max_hops=max_hops,
                pipeline_run_ids=pipeline_run_ids,
            )
            if local_payload.get("reasoning_path") or local_payload.get("direct_relationships"):
                return _attach_local_reasoning(result, local_payload)
        return result
    except httpx.HTTPError as e:
        local_payload = _local_pipeline_reasoning_response(
            query=query,
            domain=domain,
            source=source,
            target=target,
            max_hops=max_hops,
            pipeline_run_ids=pipeline_run_ids,
            remote_error=str(e),
        )
        if local_payload.get("local_pipeline", {}).get("ready") or data.get("local_fallback", True):
            return local_payload
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/reasoning/explain")
async def explain_relationship(request: Request):
    """Explain the relationship between two entities."""
    check_graphrag_enabled()

    data = await request.json()
    source = data.get("source")
    target = data.get("target")
    domain = data.get("domain")
    pipeline_run_ids = _string_list(data.get("pipeline_run_ids") or data.get("pipeline_run_id"))
    max_hops = _bounded_reasoning_hops(data.get("max_hops", data.get("maxHops", 3)))
    cfg = get_graphrag_config()
    use_local = bool(
        data.get("use_local_pipeline_context")
        or data.get("include_local_pipeline")
        or pipeline_run_ids
        or data.get("reasoning_mode") == "local_pipeline"
        or data.get("search_mode") == "local_pipeline"
        or data.get("local_only")
    )
    local_only = bool(
        data.get("reasoning_mode") == "local_pipeline"
        or data.get("search_mode") == "local_pipeline"
        or data.get("local_only")
        or not cfg.graphrag_url
    )

    if not source or not target:
        raise HTTPException(status_code=400, detail="source and target are required")

    query = str(data.get("query") or f"Explain relationship between {source} and {target}")
    if local_only:
        return _local_pipeline_reasoning_response(
            query=query,
            domain=domain,
            source=source,
            target=target,
            max_hops=max_hops,
            pipeline_run_ids=pipeline_run_ids,
            mode="explain",
            remote_error=None if cfg.graphrag_url else "GRAPHRAG_URL is not configured.",
        )

    try:
        client = get_http_client()
        payload = {
            "source": source,
            "target": target,
            "entities": data.get("entities", []),
            "relationships": data.get("relationships", []),
        }

        response = await client.post(
            f"{_graphrag_url()}/reasoning/explain-relationship",
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
        if use_local and isinstance(result, dict):
            local_payload = _local_pipeline_reasoning_response(
                query=query,
                domain=domain,
                source=source,
                target=target,
                max_hops=max_hops,
                pipeline_run_ids=pipeline_run_ids,
                mode="explain",
            )
            if local_payload.get("paths") or local_payload.get("direct_relationships"):
                return _attach_local_reasoning(result, local_payload)
        return result

    except httpx.HTTPError as e:
        logger.error(f"Relationship explanation error: {e}")
        local_payload = _local_pipeline_reasoning_response(
            query=query,
            domain=domain,
            source=source,
            target=target,
            max_hops=max_hops,
            pipeline_run_ids=pipeline_run_ids,
            mode="explain",
            remote_error=str(e),
        )
        if local_payload.get("local_pipeline", {}).get("ready") or data.get("local_fallback", True):
            return local_payload
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/reasoning/causal")
async def graphrag_causal_reasoning(request: Request):
    """Causal reasoning to find cause-effect relationships."""
    check_graphrag_enabled()

    data = await request.json()
    query = str(data.get("query") or "").strip()
    domain = data.get("domain")
    source = data.get("source")
    target = data.get("target")
    pipeline_run_ids = _string_list(data.get("pipeline_run_ids") or data.get("pipeline_run_id"))
    max_hops = _bounded_reasoning_hops(data.get("max_hops", data.get("maxHops", 3)))
    cfg = get_graphrag_config()
    use_local = bool(
        data.get("use_local_pipeline_context")
        or data.get("include_local_pipeline")
        or pipeline_run_ids
        or data.get("reasoning_mode") == "local_pipeline"
        or data.get("search_mode") == "local_pipeline"
        or data.get("local_only")
    )
    local_only = bool(
        data.get("reasoning_mode") == "local_pipeline"
        or data.get("search_mode") == "local_pipeline"
        or data.get("local_only")
        or not cfg.graphrag_url
    )

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    if local_only:
        return _local_pipeline_reasoning_response(
            query=query,
            domain=domain,
            source=source,
            target=target,
            max_hops=max_hops,
            pipeline_run_ids=pipeline_run_ids,
            mode="causal",
            remote_error=None if cfg.graphrag_url else "GRAPHRAG_URL is not configured.",
        )

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/api/causal-reasoning",
            params={"query": query},
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()
        if use_local and isinstance(result, dict):
            local_payload = _local_pipeline_reasoning_response(
                query=query,
                domain=domain,
                source=source,
                target=target,
                max_hops=max_hops,
                pipeline_run_ids=pipeline_run_ids,
                mode="causal",
            )
            if local_payload.get("reasoning_path") or local_payload.get("direct_relationships"):
                return _attach_local_reasoning(result, local_payload)
        return result

    except httpx.HTTPError as e:
        logger.error(f"Causal reasoning error: {e}")
        local_payload = _local_pipeline_reasoning_response(
            query=query,
            domain=domain,
            source=source,
            target=target,
            max_hops=max_hops,
            pipeline_run_ids=pipeline_run_ids,
            mode="causal",
            remote_error=str(e),
        )
        if local_payload.get("local_pipeline", {}).get("ready") or data.get("local_fallback", True):
            return local_payload
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/reasoning/comparative")
async def graphrag_comparative_reasoning(request: Request):
    """Compare entities or concepts."""
    check_graphrag_enabled()

    data = await request.json()
    query = str(data.get("query") or "").strip()
    domain = data.get("domain")
    source = data.get("source")
    target = data.get("target")
    pipeline_run_ids = _string_list(data.get("pipeline_run_ids") or data.get("pipeline_run_id"))
    max_hops = _bounded_reasoning_hops(data.get("max_hops", data.get("maxHops", 3)))
    cfg = get_graphrag_config()
    use_local = bool(
        data.get("use_local_pipeline_context")
        or data.get("include_local_pipeline")
        or pipeline_run_ids
        or data.get("reasoning_mode") == "local_pipeline"
        or data.get("search_mode") == "local_pipeline"
        or data.get("local_only")
    )
    local_only = bool(
        data.get("reasoning_mode") == "local_pipeline"
        or data.get("search_mode") == "local_pipeline"
        or data.get("local_only")
        or not cfg.graphrag_url
    )

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    if local_only:
        return _local_pipeline_reasoning_response(
            query=query,
            domain=domain,
            source=source,
            target=target,
            max_hops=max_hops,
            pipeline_run_ids=pipeline_run_ids,
            mode="comparative",
            remote_error=None if cfg.graphrag_url else "GRAPHRAG_URL is not configured.",
        )

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/api/comparative-reasoning",
            params={"query": query},
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()
        if use_local and isinstance(result, dict):
            local_payload = _local_pipeline_reasoning_response(
                query=query,
                domain=domain,
                source=source,
                target=target,
                max_hops=max_hops,
                pipeline_run_ids=pipeline_run_ids,
                mode="comparative",
            )
            if local_payload.get("reasoning_path") or local_payload.get("direct_relationships"):
                return _attach_local_reasoning(result, local_payload)
        return result

    except httpx.HTTPError as e:
        logger.error(f"Comparative reasoning error: {e}")
        local_payload = _local_pipeline_reasoning_response(
            query=query,
            domain=domain,
            source=source,
            target=target,
            max_hops=max_hops,
            pipeline_run_ids=pipeline_run_ids,
            mode="comparative",
            remote_error=str(e),
        )
        if local_payload.get("local_pipeline", {}).get("ready") or data.get("local_fallback", True):
            return local_payload
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/reasoning/advanced")
async def graphrag_advanced_reasoning(request: Request):
    """Advanced reasoning with query analysis."""
    check_graphrag_enabled()

    data = await request.json()
    query = str(data.get("query") or "").strip()
    domain = data.get("domain")
    source = data.get("source")
    target = data.get("target")
    pipeline_run_ids = _string_list(data.get("pipeline_run_ids") or data.get("pipeline_run_id"))
    max_hops = _bounded_reasoning_hops(data.get("max_hops", data.get("maxHops", 3)))
    cfg = get_graphrag_config()
    use_local = bool(
        data.get("use_local_pipeline_context")
        or data.get("include_local_pipeline")
        or pipeline_run_ids
        or data.get("reasoning_mode") == "local_pipeline"
        or data.get("search_mode") == "local_pipeline"
        or data.get("local_only")
    )
    local_only = bool(
        data.get("reasoning_mode") == "local_pipeline"
        or data.get("search_mode") == "local_pipeline"
        or data.get("local_only")
        or not cfg.graphrag_url
    )

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    if local_only:
        return _local_pipeline_reasoning_response(
            query=query,
            domain=domain,
            source=source,
            target=target,
            max_hops=max_hops,
            pipeline_run_ids=pipeline_run_ids,
            mode="advanced",
            remote_error=None if cfg.graphrag_url else "GRAPHRAG_URL is not configured.",
        )

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/api/advanced-reasoning",
            params={"query": query},
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()
        if use_local and isinstance(result, dict):
            local_payload = _local_pipeline_reasoning_response(
                query=query,
                domain=domain,
                source=source,
                target=target,
                max_hops=max_hops,
                pipeline_run_ids=pipeline_run_ids,
                mode="advanced",
            )
            if local_payload.get("reasoning_path") or local_payload.get("direct_relationships"):
                return _attach_local_reasoning(result, local_payload)
        return result

    except httpx.HTTPError as e:
        logger.error(f"Advanced reasoning error: {e}")
        local_payload = _local_pipeline_reasoning_response(
            query=query,
            domain=domain,
            source=source,
            target=target,
            max_hops=max_hops,
            pipeline_run_ids=pipeline_run_ids,
            mode="advanced",
            remote_error=str(e),
        )
        if local_payload.get("local_pipeline", {}).get("ready") or data.get("local_fallback", True):
            return local_payload
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/code/detect")
async def graphrag_detect_code(request: Request):
    """Detect code in text."""
    check_graphrag_enabled()

    cfg = get_graphrag_config()
    content_type = request.headers.get("content-type", "")
    filename = "snippet.txt"
    text = ""
    file_content = b""
    upload_content_type = "text/plain"

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="file is required")
        filename = getattr(upload, "filename", None) or filename
        upload_content_type = getattr(upload, "content_type", None) or upload_content_type
        file_content = await upload.read()
        text = file_content.decode("utf-8", errors="ignore")
    else:
        data = await request.json()
        text = data.get("text") or data.get("content") or ""
        filename = data.get("filename") or filename
        upload_content_type = data.get("content_type") or upload_content_type
        file_content = text.encode("utf-8")

    if not text and not file_content:
        raise HTTPException(status_code=400, detail="text or file is required")

    if not cfg.graphrag_url:
        return _detect_text_code(text, filename)

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/code-detection/detect",
            files={"file": (filename, file_content, upload_content_type)},
        )
        response.raise_for_status()
        return _normalize_detection_response(response.json(), text, filename)

    except httpx.HTTPError as e:
        logger.error(f"Code detection error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/code/search")
async def graphrag_search_code(request: Request):
    """Search for code examples."""
    check_graphrag_enabled()

    data = await request.json()
    query = data.get("query")

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    try:
        client = get_http_client()
        cfg = get_graphrag_config()
        top_k = data.get("top_k", data.get("limit", 10))
        if cfg.code_rag_url:
            languages = data.get("languages")
            selected_language = data.get("language")
            if not selected_language and isinstance(languages, list) and languages:
                selected_language = languages[0]
            elif not selected_language and isinstance(languages, str):
                selected_language = languages

            response = await client.post(
                f"{cfg.code_rag_url}/search",
                json={
                    "query": query,
                    "language": selected_language,
                    "top_k": top_k,
                    "threshold": data.get("threshold", 0.0),
                    "context": data.get("context"),
                },
            )
        else:
            response = await client.post(
                f"{_graphrag_url()}/search/code",
                json={**data, "top_k": top_k},
            )
        response.raise_for_status()
        return _normalize_code_search_response(response.json())

    except httpx.HTTPError as e:
        logger.error(f"Code search error: {e}")
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")
