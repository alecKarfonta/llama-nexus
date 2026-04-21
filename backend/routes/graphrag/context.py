"""
GraphRAG Context Routes

Chat context generation with NER, REL, vector search,
and graph neighborhood expansion.

All endpoints use the shared connection-pooled HTTP client
and the unified GraphRAGConfig singleton.
"""

from fastapi import APIRouter, HTTPException, Request
import asyncio
import logging

from .helpers import (
    check_graphrag_enabled, get_http_client,
    _fetch_graph_data, _find_entity_neighbors,
    _extract_entities_ner, _extract_relationships_rel,
    _expand_graph_neighborhood,
    _deduplicate_entities, _deduplicate_relationships,
)
from modules.graphrag.config import get_graphrag_config

logger = logging.getLogger(__name__)

router = APIRouter()


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
        client, entities, domain, errors
    )

    # Deduplicate (shared helpers)
    entities = _deduplicate_entities(entities, expanded_entities)
    relationships = _deduplicate_relationships(relationships)

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

    if relationships:
        rel_strs = [f"{r['source']} --[{r['relation']}]--> {r['target']}" for r in relationships[:10]]
        context_parts.append("Extracted relationships:\n" + "\n".join(f"  • {r}" for r in rel_strs))

    if search_results:
        passage_strs = [f"[{i+1}] {r['content']}" for i, r in enumerate(search_results[:top_k])]
        context_parts.append("Relevant passages:\n" + "\n\n".join(passage_strs))

    context_text = ""
    if context_parts:
        context_text = (
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
        "expanded_entities": expanded_entities,
        "source": "graphrag",
        "services_used": {
            "ner": cfg.ner_enabled,
            "rel": cfg.rel_enabled,
            "vector_search": cfg.has_graph_service,
            "graph_expansion": cfg.has_graph_service and len(expanded_entities) > 0,
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
    domain = data.get("domain", "general")
    rag_enabled = data.get("rag_enabled", True)
    rag_domains = data.get("rag_domains", [])
    rag_search_mode = data.get("rag_search_mode", "domain")

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
        try:
            from modules.rag.retrievers import VectorRetriever, RetrievalConfig
            from routes.rag import get_rag_components, create_embedder, get_collection_name

            rag = get_rag_components(request)
            if rag.get('document_manager') and rag.get('vector_store'):
                # Resolve domain
                domain = None
                domain_id = rag_domains[0] if rag_domains else None

                if domain_id:
                    domain = await rag['document_manager'].get_domain(domain_id)
                elif rag_search_mode != "global":
                    domain = await rag['document_manager'].get_general_domain()
                    if domain:
                        domain_id = domain.id

                embedding_model = 'nomic-embed-text-v1.5'
                if domain and domain.embedding_model:
                    embedding_model = domain.embedding_model

                embedder = create_embedder(request, model_name=embedding_model)

                if rag_search_mode == "global":
                    # Cross-domain search
                    all_domains = await rag['document_manager'].list_domains()
                    coll_names = [get_collection_name(d) for d in all_domains]
                    existence = await asyncio.gather(
                        *[rag['vector_store'].collection_exists(cn) for cn in coll_names]
                    )
                    valid = [(d, cn) for d, cn, ex in zip(all_domains, coll_names, existence) if ex]

                    async def _search_domain(d, cn):
                        r = VectorRetriever(rag['vector_store'], embedder, rag['document_manager'], cn)
                        return await r.retrieve(query, RetrievalConfig(top_k=top_k, domain_ids=[d.id]))

                    domain_results = await asyncio.gather(*[_search_domain(d, cn) for d, cn in valid])
                    all_results = sorted(
                        [r for batch in domain_results for r in batch],
                        key=lambda r: r.score if hasattr(r, 'score') else 0,
                        reverse=True
                    )[:top_k]
                else:
                    # Single domain search
                    collection_name = get_collection_name(domain) if domain else None
                    if collection_name and await rag['vector_store'].collection_exists(collection_name):
                        retriever = VectorRetriever(
                            rag['vector_store'], embedder, rag['document_manager'], collection_name
                        )
                        all_results = await retriever.retrieve(
                            query, RetrievalConfig(top_k=top_k, domain_ids=[domain_id] if domain_id else None)
                        )
                    else:
                        all_results = []

                for r in all_results:
                    rd = r.to_dict()
                    rag_chunks.append({
                        "id": rd.get("id") or rd.get("chunk_id", ""),
                        "content": rd.get("content") or rd.get("text", ""),
                        "score": round(rd.get("score") or rd.get("similarity", 0), 3),
                        "document_id": rd.get("document_id", ""),
                        "document_title": rd.get("document_title") or rd.get("metadata", {}).get("title", ""),
                        "chunk_index": rd.get("chunk_index", 0),
                        "metadata": rd.get("metadata", {}),
                    })
            else:
                logger.warning("RAG components not initialized, skipping RAG retrieval")
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            errors.append(f"RAG: {str(e)}")

    # ── Step 4: GraphRAG vector search ────────────────────────────
    graph_search_results, extra_entities = await _search_graphrag(
        client, query, top_k, domain, errors
    )
    entities.extend(extra_entities)

    # ── Step 5: Graph neighborhood expansion (shared helper) ──────
    expanded_entities, graph_connections = await _expand_graph_neighborhood(
        client, entities, domain, errors
    )

    # ── Deduplication (shared helpers) ─────────────────────────────
    entities = _deduplicate_entities(entities, expanded_entities)
    relationships = _deduplicate_relationships(relationships)

    # ── Step 6: Cross-referencing — entity-aware chunk scoring ─────
    all_entity_names = [e["name"] for e in entities]
    cross_references = []

    for i, chunk in enumerate(rag_chunks):
        overlap, matched = _entity_overlap_score(chunk["content"], all_entity_names)
        if matched:
            chunk["entity_boost"] = round(overlap * 0.2, 3)
            chunk["boosted_score"] = round(chunk["score"] + chunk["entity_boost"], 3)
            chunk["matched_entities"] = matched
            cross_references.append({
                "chunk_index": i,
                "chunk_id": chunk.get("id", ""),
                "matched_entities": matched,
                "overlap_score": round(overlap, 3),
            })
        else:
            chunk["entity_boost"] = 0
            chunk["boosted_score"] = chunk["score"]
            chunk["matched_entities"] = []

    # Resort chunks by boosted score
    rag_chunks.sort(key=lambda c: c.get("boosted_score", c["score"]), reverse=True)

    # ── Step 7: Build unified context template ─────────────────────
    context_parts = []

    # Interleaved chunks with entity annotations
    if rag_chunks:
        chunk_strs = []
        for i, chunk in enumerate(rag_chunks[:top_k]):
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

            source_info = ""
            if chunk.get("document_title"):
                source_info = f" [source: {chunk['document_title']}]"

            chunk_strs.append(f"[{i+1}] {chunk['content']}{annotation}{source_info}")

        context_parts.append("=== Knowledge Context ===\n" + "\n\n".join(chunk_strs))

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

    if relationships:
        rel_strs = [f"{r['source']} --[{r['relation']}]--> {r['target']}" for r in relationships[:10]]
        context_parts.append("Extracted relationships:\n" + "\n".join(f"  • {r}" for r in rel_strs))

    context_text = ""
    if context_parts:
        context_text = (
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

    quality_signals = {
        "avg_chunk_relevance": avg_chunk_relevance,
        "entity_coverage": entity_coverage,
        "graph_expansion_count": len(expanded_entities),
        "cross_reference_count": len(cross_references),
        "total_chunks": len(rag_chunks),
        "total_entities": len(entities),
    }

    return {
        "chunks": rag_chunks,
        "entities": entities,
        "relationships": relationships,
        "search_results": graph_search_results,
        "graph_connections": graph_connections,
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
        },
        "errors": errors if errors else None,
    }
