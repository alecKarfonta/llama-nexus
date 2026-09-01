from unittest.mock import AsyncMock, Mock, patch

import pytest


@pytest.fixture
def mock_execution_context():
    context = Mock()
    context.log = Mock()
    context.workflow_id = "workflow-1"
    context.execution_id = "execution-1"
    return context


def _mock_response(payload, status_code=200):
    response = Mock()
    response.json.return_value = payload
    response.text = str(payload)
    response.status_code = status_code
    response.headers = {"content-type": "application/json"}
    response.is_error = status_code >= 400
    response.raise_for_status = Mock()
    return response


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_graphrag_capabilities_executor(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGCapabilitiesExecutor

    payload = {
        "enabled": True,
        "services": {"graphrag": {"enabled": True}},
        "capabilities": [
            {"id": "search", "label": "Search"},
            {"id": "evaluation", "label": "Evaluation"},
        ],
        "upstream_bridge": {
            "available": True,
            "allowed_prefixes": ["search", "evaluate/"],
        },
    }
    mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=_mock_response(payload))

    result = await GraphRAGCapabilitiesExecutor(config={}).execute({}, mock_execution_context)

    assert result["enabled"] is True
    assert result["capability_ids"] == ["search", "evaluation"]
    assert result["capability_count"] == 2
    assert result["bridge_available"] is True
    assert result["allowed_prefixes"] == ["search", "evaluate/"]


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_query_complexity_executor_uses_typed_backend_route(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGQueryComplexityExecutor

    payload = {
        "query": "How did A cause B?",
        "primary_reasoning": "causal",
        "detected_patterns": ["cause_effect"],
        "complexity_level": "moderate",
        "requires_multi_hop": True,
    }
    post = AsyncMock(return_value=_mock_response(payload))
    mock_client.return_value.__aenter__.return_value.post = post

    result = await GraphRAGQueryComplexityExecutor(config={}).execute(
        {"query": "How did A cause B?"},
        mock_execution_context,
    )

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/query/complexity")
    assert post.await_args.kwargs["json"] == {"query": "How did A cause B?"}
    assert result["primary_reasoning"] == "causal"
    assert result["requires_multi_hop"] is True
    assert result["analysis"] == payload


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_enhanced_entity_extraction_executor_posts_typed_json(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import EnhancedEntityExtractionExecutor

    payload = {
        "entities": [{"name": "GraphRAG", "type": "technology"}],
        "relationships": [{"source": "GraphRAG", "target": "Llama Nexus"}],
        "claims": ["GraphRAG augments retrieval."],
        "entity_count": 1,
        "relationship_count": 1,
        "extraction_metadata": {"methods": ["dependency"]},
        "span_extractions": [],
        "dependency_extractions": [{"relation": "augments"}],
        "linked_entities": [{"name": "GraphRAG", "id": "graph-rag"}],
    }
    post = AsyncMock(return_value=_mock_response(payload))
    mock_client.return_value.__aenter__.return_value.post = post

    executor = EnhancedEntityExtractionExecutor(
        config={
            "useSpanbert": False,
            "useDependency": True,
            "useEntityLinking": False,
        }
    )
    result = await executor.execute(
        {"text": "GraphRAG augments Llama Nexus.", "domain": "software"},
        mock_execution_context,
    )

    assert post.await_args.args[0].endswith("/api/v1/graphrag/extract/enhanced")
    assert post.await_args.kwargs["json"] == {
        "text": "GraphRAG augments Llama Nexus.",
        "domain": "software",
        "use_spanbert": False,
        "use_dependency": True,
        "use_entity_linking": False,
    }
    assert result["entity_count"] == 1
    assert result["relationship_count"] == 1
    assert result["linked_entities"][0]["id"] == "graph-rag"


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_document_ingest_executor_posts_text_to_typed_backend_route(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGDocumentIngestExecutor

    post = AsyncMock(return_value=_mock_response({
        "status": "success",
        "filename": "note.md",
        "results": {
            "note.md": {
                "chunks": [{"text": "hello"}],
                "total_chunks": 1,
                "entities": 2,
                "relationships": 1,
            }
        },
        "total_files": 1,
    }))
    mock_client.return_value.__aenter__.return_value.post = post

    result = await GraphRAGDocumentIngestExecutor(
        config={"domain": "docs", "buildKnowledgeGraph": True}
    ).execute(
        {
            "text": "# GraphRAG\nContext.",
            "filename": "note.md",
            "content_type": "text/markdown",
        },
        mock_execution_context,
    )

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/ingest")
    assert post.await_args.kwargs["json"] == {
        "text": "# GraphRAG\nContext.",
        "filename": "note.md",
        "domain": "docs",
        "content_type": "text/markdown",
        "use_semantic_chunking": True,
        "build_knowledge_graph": True,
    }
    assert result["filename"] == "note.md"
    assert result["total_chunks"] == 1
    assert result["entities"] == 2
    assert result["relationships"] == 1


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_document_list_executor_fetches_indexed_documents(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGDocumentListExecutor

    get = AsyncMock(return_value=_mock_response({
        "documents": [{"name": "note.md", "chunks": 3}],
        "total_documents": 1,
        "vector_store_documents": 1,
        "knowledge_graph_documents": 1,
    }))
    mock_client.return_value.__aenter__.return_value.get = get

    result = await GraphRAGDocumentListExecutor(config={}).execute({}, mock_execution_context)

    get.assert_awaited_once()
    assert get.await_args.args[0].endswith("/api/v1/graphrag/documents")
    assert result["documents"][0]["name"] == "note.md"
    assert result["total_documents"] == 1


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_document_delete_executor_deletes_encoded_document_name(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGDocumentDeleteExecutor

    delete = AsyncMock(return_value=_mock_response({
        "status": "success",
        "message": "deleted",
        "deleted": True,
    }))
    mock_client.return_value.__aenter__.return_value.delete = delete

    result = await GraphRAGDocumentDeleteExecutor(config={}).execute(
        {"document_name": "docs/note.md"},
        mock_execution_context,
    )

    delete.assert_awaited_once()
    assert delete.await_args.args[0].endswith("/api/v1/graphrag/documents/docs%2Fnote.md")
    assert result["document_name"] == "docs/note.md"
    assert result["deleted"] is True


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_knowledge_graph_rebuild_executor_posts_domain(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGKnowledgeGraphRebuildExecutor

    post = AsyncMock(return_value=_mock_response({
        "message": "rebuilt",
        "documents_processed": 4,
        "entities_extracted": 12,
        "relationships_extracted": 9,
    }))
    mock_client.return_value.__aenter__.return_value.post = post

    result = await GraphRAGKnowledgeGraphRebuildExecutor(config={"domain": "docs"}).execute(
        {},
        mock_execution_context,
    )

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/rebuild")
    assert post.await_args.kwargs["json"] == {"domain": "docs"}
    assert result["documents_processed"] == 4
    assert result["entities_extracted"] == 12


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_code_rag_ingest_executor_posts_file_paths(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGCodeIngestExecutor

    post = AsyncMock(return_value=_mock_response({
        "task_id": "task-1",
        "status": "started",
        "message": "Started processing 2 files",
    }))
    mock_client.return_value.__aenter__.return_value.post = post

    result = await GraphRAGCodeIngestExecutor(config={"projectName": "demo", "language": "python"}).execute(
        {"file_paths": ["/repo/a.py", "/repo/b.py"]},
        mock_execution_context,
    )

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/code/ingest")
    assert post.await_args.kwargs["json"] == {
        "file_paths": ["/repo/a.py", "/repo/b.py"],
        "project_name": "demo",
        "language": "python",
    }
    assert result["task_id"] == "task-1"
    assert result["file_count"] == 2
    assert result["project_name"] == "demo"


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_code_rag_ingestion_status_executor_gets_task_progress(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGCodeIngestionStatusExecutor

    get = AsyncMock(return_value=_mock_response({
        "task_id": "task-1",
        "status": "processing",
        "progress": 40,
        "processed_files": 2,
        "total_files": 5,
        "entities_found": 9,
    }))
    mock_client.return_value.__aenter__.return_value.get = get

    result = await GraphRAGCodeIngestionStatusExecutor(config={}).execute(
        {"task_id": "task-1"},
        mock_execution_context,
    )

    get.assert_awaited_once()
    assert get.await_args.args[0].endswith("/api/v1/graphrag/code/ingest/task-1")
    assert result["status"] == "processing"
    assert result["progress"] == 40
    assert result["entities_found"] == 9


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_code_rag_analysis_executor_returns_entities_and_relationships(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGCodeAnalysisExecutor

    post = AsyncMock(return_value=_mock_response({
        "file_path": "/repo/main.py",
        "language": "python",
        "success": True,
        "entities": [{"name": "main"}],
        "relationships": [{"source": "main", "target": "helper"}],
        "parse_time_ms": 12.5,
    }))
    mock_client.return_value.__aenter__.return_value.post = post

    result = await GraphRAGCodeAnalysisExecutor(config={}).execute(
        {"file_path": "/repo/main.py"},
        mock_execution_context,
    )

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/code/analyze")
    assert post.await_args.kwargs["json"] == {"file_path": "/repo/main.py"}
    assert result["success"] is True
    assert result["entity_count"] == 1
    assert result["relationship_count"] == 1


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_code_rag_stats_executor_collects_statistics(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGCodeStatsExecutor

    get = AsyncMock(return_value=_mock_response({
        "search_engine": {"entities": 12},
        "active_tasks": 1,
        "completed_tasks": 2,
        "failed_tasks": 0,
    }))
    mock_client.return_value.__aenter__.return_value.get = get

    result = await GraphRAGCodeStatsExecutor(config={}).execute({}, mock_execution_context)

    get.assert_awaited_once()
    assert get.await_args.args[0].endswith("/api/v1/graphrag/code/statistics")
    assert result["search_engine"]["entities"] == 12
    assert result["active_tasks"] == 1


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_reddit_crawl_executor_starts_crawl(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGRedditCrawlExecutor

    post = AsyncMock(return_value=_mock_response({
        "crawl_id": "crawl-1",
        "status": "started",
        "posts_found": 12,
        "comments_found": 34,
        "total_content_length": 2048,
        "estimated_processing_time": "2m",
        "ingestion_status": "queued",
    }))
    mock_client.return_value.__aenter__.return_value.post = post

    result = await GraphRAGRedditCrawlExecutor(config={
        "subreddits": "r/LocalLLaMA, MachineLearning",
        "maxPosts": 25,
        "autoIngest": True,
    }).execute(
        {"query": "graph retrieval", "filter_nsfw": False},
        mock_execution_context,
    )

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/reddit/crawl")
    assert post.await_args.kwargs["json"] == {
        "query": "graph retrieval",
        "subreddits": ["LocalLLaMA", "MachineLearning"],
        "sort_by": "relevance",
        "time_filter": "month",
        "max_posts": 25,
        "max_comments_per_post": 20,
        "comment_depth": 2,
        "include_replies": True,
        "auto_ingest": True,
        "filter_nsfw": False,
    }
    assert result["crawl_id"] == "crawl-1"
    assert result["posts_found"] == 12
    assert result["subreddits"] == ["LocalLLaMA", "MachineLearning"]


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_reddit_crawl_status_executor_polls_status(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGRedditCrawlStatusExecutor

    get = AsyncMock(return_value=_mock_response({
        "crawl_id": "crawl-1",
        "status": "running",
        "progress": 0.5,
        "posts_processed": 6,
        "comments_processed": 18,
        "current_post": "post-6",
    }))
    mock_client.return_value.__aenter__.return_value.get = get

    result = await GraphRAGRedditCrawlStatusExecutor(config={}).execute(
        {"crawl_id": "crawl-1"},
        mock_execution_context,
    )

    get.assert_awaited_once()
    assert get.await_args.args[0].endswith("/api/v1/graphrag/reddit/crawl/crawl-1/status")
    assert result["status"] == "running"
    assert result["progress"] == 0.5


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_reddit_crawl_results_executor_fetches_results(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGRedditCrawlResultsExecutor

    get = AsyncMock(return_value=_mock_response({
        "crawl_id": "crawl-1",
        "status": "completed",
        "results": [{"id": "post-1"}, {"id": "comment-1"}],
        "statistics": {"posts": 1, "comments": 1},
        "ingestion_status": "ingested",
    }))
    mock_client.return_value.__aenter__.return_value.get = get

    result = await GraphRAGRedditCrawlResultsExecutor(config={}).execute(
        {"crawl_id": "crawl-1"},
        mock_execution_context,
    )

    get.assert_awaited_once()
    assert get.await_args.args[0].endswith("/api/v1/graphrag/reddit/crawl/crawl-1/results")
    assert result["status"] == "completed"
    assert result["result_count"] == 2
    assert result["statistics"] == {"posts": 1, "comments": 1}


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_evaluation_status_executor_fetches_retrieval_and_chunking(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGEvaluationStatusExecutor

    get = AsyncMock(return_value=_mock_response({
        "retrieval": {"evaluation_system": "GutenQA", "dataset_available": True},
        "chunking": {"evaluation_system": "WikiSection", "dataset_available": False},
        "available": True,
    }))
    mock_client.return_value.__aenter__.return_value.get = get

    result = await GraphRAGEvaluationStatusExecutor(config={}).execute({}, mock_execution_context)

    get.assert_awaited_once()
    assert get.await_args.args[0].endswith("/api/v1/graphrag/evaluation/status")
    assert result["available"] is True
    assert result["retrieval"]["evaluation_system"] == "GutenQA"
    assert result["chunking"]["evaluation_system"] == "WikiSection"


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_feedback_submit_executor_posts_quality_payload(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGFeedbackSubmitExecutor

    post = AsyncMock(return_value=_mock_response({"status": "ok", "message": "Feedback recorded"}))
    mock_client.return_value.__aenter__.return_value.post = post

    result = await GraphRAGFeedbackSubmitExecutor(config={"source": "workflow"}).execute(
        {
            "query": "How useful was graph context?",
            "rating": 1,
            "comment": "Helpful graph evidence",
            "domain": "research",
            "quality_signals": {"total_chunks": 4, "avg_chunk_relevance": 0.8},
        },
        mock_execution_context,
    )

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/feedback")
    assert post.await_args.kwargs["json"] == {
        "query": "How useful was graph context?",
        "rating": 1,
        "comment": "Helpful graph evidence",
        "quality_signals": {"total_chunks": 4, "avg_chunk_relevance": 0.8},
        "source": "workflow",
        "domain": "research",
    }
    assert result["status"] == "ok"
    assert result["rating"] == 1


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_feedback_analytics_executor_gets_domain_trends(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGFeedbackAnalyticsExecutor

    get = AsyncMock(return_value=_mock_response({
        "period_days": 14,
        "domain_filter": "research",
        "overall": {"total_feedback": 5, "satisfaction_rate": 0.8},
        "daily_trends": [{"date": "2026-07-10", "count": 5}],
        "domain_stats": [{"domain": "research", "count": 5}],
        "problem_queries": [{"query": "bad result"}],
    }))
    mock_client.return_value.__aenter__.return_value.get = get

    result = await GraphRAGFeedbackAnalyticsExecutor(config={"days": 14, "domain": "research"}).execute(
        {},
        mock_execution_context,
    )

    get.assert_awaited_once()
    assert get.await_args.args[0].endswith("/api/v1/graphrag/analytics")
    assert get.await_args.kwargs["params"] == {"days": 14, "domain": "research"}
    assert result["total_feedback"] == 5
    assert result["satisfaction_rate"] == 0.8
    assert result["problem_queries"][0]["query"] == "bad result"


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_social_memory_ingest_executor_posts_typed_message(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGSocialMemoryIngestExecutor

    post = AsyncMock(return_value=_mock_response({
        "person_id": "person-1",
        "episode_id": "episode-1",
        "facts_written": 2,
        "queued": False,
    }))
    mock_client.return_value.__aenter__.return_value.post = post

    result = await GraphRAGSocialMemoryIngestExecutor(config={"platform": "youtube", "sync": True}).execute(
        {
            "platform_user_id": "channel-user",
            "display_name": "Ada",
            "text": "I like graph search.",
            "timestamp": "2026-07-10T12:30:00Z",
            "message_id": "msg-1",
            "priority": 3,
        },
        mock_execution_context,
    )

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/social-memory/ingest")
    assert post.await_args.kwargs["params"] == {"sync": True}
    assert post.await_args.kwargs["json"] == {
        "platform": "youtube",
        "platform_user_id": "channel-user",
        "display_name": "Ada",
        "text": "I like graph search.",
        "timestamp": "2026-07-10T12:30:00Z",
        "message_id": "msg-1",
        "stream_id": None,
        "session_id": None,
        "previous_messages": [],
        "priority": 3,
    }
    assert result["person_id"] == "person-1"
    assert result["facts_written"] == 2
    assert result["queued"] is False


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_social_memory_context_executor_can_fetch_retrieval_trace(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGSocialMemoryContextExecutor

    get = AsyncMock(return_value=_mock_response({
        "person_id": "person-1",
        "query": "What does Ada like?",
        "retrieval_path": "hot",
        "cache_hit": True,
        "fact_ids": ["fact-1"],
    }))
    mock_client.return_value.__aenter__.return_value.get = get

    result = await GraphRAGSocialMemoryContextExecutor(config={"mode": "explain"}).execute(
        {
            "platform": "twitch",
            "platform_user_id": "viewer-1",
            "query": "What does Ada like?",
            "max_facts": 5,
        },
        mock_execution_context,
    )

    get.assert_awaited_once()
    assert get.await_args.args[0].endswith("/api/v1/graphrag/social-memory/explain")
    assert get.await_args.kwargs["params"] == {
        "platform": "twitch",
        "platform_user_id": "viewer-1",
        "query": "What does Ada like?",
        "max_facts": 5,
    }
    assert result["mode"] == "explain"
    assert result["fact_ids"] == ["fact-1"]
    assert result["trace"]["cache_hit"] is True


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_social_memory_facts_executor_lists_person_facts(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGSocialMemoryFactsExecutor

    get = AsyncMock(return_value=_mock_response({
        "person_id": "person-1",
        "facts": [{"id": "fact-1", "text": "Ada likes graph search."}],
    }))
    mock_client.return_value.__aenter__.return_value.get = get

    result = await GraphRAGSocialMemoryFactsExecutor(config={}).execute(
        {"person_id": "person-1", "as_of": "2026-07-10T12:30:00Z"},
        mock_execution_context,
    )

    get.assert_awaited_once()
    assert get.await_args.args[0].endswith("/api/v1/graphrag/social-memory/person/person-1/facts")
    assert get.await_args.kwargs["params"] == {"as_of": "2026-07-10T12:30:00Z"}
    assert result["fact_count"] == 1
    assert result["facts"][0]["id"] == "fact-1"


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_social_memory_admin_executor_corrects_fact(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGSocialMemoryAdminExecutor

    request = AsyncMock(return_value=_mock_response({"updated": True}))
    mock_client.return_value.__aenter__.return_value.request = request

    result = await GraphRAGSocialMemoryAdminExecutor(config={"adminApiKey": "secret"}).execute(
        {
            "mode": "correct_fact",
            "fact_id": "fact-1",
            "new_fact_text": "Ada likes knowledge graphs.",
            "corrected_by": "operator",
            "predicate": "LIKES",
        },
        mock_execution_context,
    )

    request.assert_awaited_once()
    assert request.await_args.args[0] == "POST"
    assert request.await_args.args[1].endswith("/api/v1/graphrag/social-memory/facts/correct")
    assert request.await_args.kwargs["headers"] == {"X-API-Key": "secret"}
    assert request.await_args.kwargs["json"] == {
        "fact_id": "fact-1",
        "new_fact_text": "Ada likes knowledge graphs.",
        "corrected_by": "operator",
        "predicate": "LIKES",
    }
    assert result["mode"] == "correct_fact"
    assert result["result"] == {"updated": True}


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_ner_extraction_executor_uses_typed_backend_route(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGNERExtractionExecutor

    post = AsyncMock(return_value=_mock_response({
        "text": "OpenAI built GPT.",
        "entities": [{"name": "OpenAI"}],
        "entity_count": 1,
        "extraction_method": "gliner",
    }))
    mock_client.return_value.__aenter__.return_value.post = post

    result = await GraphRAGNERExtractionExecutor(config={}).execute(
        {"text": "OpenAI built GPT."},
        mock_execution_context,
    )

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/ner/extract")
    assert post.await_args.kwargs["json"] == {"text": "OpenAI built GPT."}
    assert result["entity_count"] == 1
    assert result["entities"][0]["name"] == "OpenAI"
    assert result["total_texts"] == 1


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_relation_extraction_executor_posts_relation_payload(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGRelationExtractionExecutor

    post = AsyncMock(return_value=_mock_response({
        "relations": [{"source": "Ada", "target": "Labs"}],
        "relation_count": 1,
        "processing_time": 0.12,
    }))
    mock_client.return_value.__aenter__.return_value.post = post

    executor = GraphRAGRelationExtractionExecutor(config={"threshold": 0.7})
    result = await executor.execute(
        {
            "text": "Ada worked for Labs.",
            "relations": [{"relation": "works for"}],
            "entity_labels": ["person", "organisation"],
        },
        mock_execution_context,
    )

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/rel/extract")
    assert post.await_args.kwargs["json"] == {
        "text": "Ada worked for Labs.",
        "relations": [{"relation": "works for"}],
        "entity_labels": ["person", "organisation"],
        "threshold": 0.7,
    }
    assert result["relation_count"] == 1
    assert result["relations"][0]["source"] == "Ada"
    assert result["total_texts"] == 1


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_relation_extraction_executor_posts_batch_payload(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGRelationExtractionExecutor

    post = AsyncMock(return_value=_mock_response({
        "results": [
            {"text": "Ada worked for Labs.", "relations": [{"source": "Ada"}], "relation_count": 1},
            {"text": "Grace built compilers.", "relations": [{"source": "Grace"}], "relation_count": 1},
        ],
        "total_texts": 2,
        "processing_time": 0.2,
    }))
    mock_client.return_value.__aenter__.return_value.post = post

    executor = GraphRAGRelationExtractionExecutor(config={"threshold": 0.7})
    result = await executor.execute(
        {
            "texts": ["Ada worked for Labs.", "Grace built compilers."],
            "relations": [{"relation": "works for"}],
            "entity_labels": ["person", "organisation"],
        },
        mock_execution_context,
    )

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/rel/extract-batch")
    assert post.await_args.kwargs["json"] == {
        "texts": ["Ada worked for Labs.", "Grace built compilers."],
        "relations": [{"relation": "works for"}],
        "entity_labels": ["person", "organisation"],
        "threshold": 0.7,
    }
    assert result["relation_count"] == 2
    assert result["total_texts"] == 2
    assert len(result["relations"]) == 2


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_enhanced_query_executor_returns_reasoning_outputs(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGEnhancedQueryExecutor

    payload = {
        "answer": "A reasoned answer",
        "results": [{"content": "evidence"}],
        "reasoning_paths": [{"path": ["A", "B"]}],
        "inferred_relationships": [{"source": "A", "target": "B"}],
        "entity_clusters": [["A", "B"]],
        "confidence": 0.82,
        "explanation": "Graph path evidence",
    }
    post = AsyncMock(return_value=_mock_response(payload))
    mock_client.return_value.__aenter__.return_value.post = post

    result = await GraphRAGEnhancedQueryExecutor(config={"searchType": "graph", "topK": 4}).execute(
        {"query": "Explain A to B", "domain": "docs"},
        mock_execution_context,
    )

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/query/enhanced")
    assert post.await_args.kwargs["json"]["search_type"] == "graph"
    assert post.await_args.kwargs["json"]["top_k"] == 4
    assert result["answer"] == "A reasoned answer"
    assert result["confidence"] == 0.82
    assert result["reasoning_paths"][0]["path"] == ["A", "B"]


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_chat_context_executor_posts_unified_context_payload(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGChatContextExecutor

    payload = {
        "context_text": "Unified context",
        "chunks": [{"id": "chunk-1"}],
        "entities": [{"name": "GraphRAG"}],
        "relationships": [{"source": "GraphRAG", "target": "RAG"}],
        "graph_connections": ["GraphRAG -> RAG"],
        "cross_references": [{"chunk_id": "chunk-1"}],
        "quality_signals": {"total_chunks": 1, "total_entities": 1},
        "services_used": {"rag": True, "graph_expansion": True},
        "source": "unified",
    }
    post = AsyncMock(return_value=_mock_response(payload))
    mock_client.return_value.__aenter__.return_value.post = post

    result = await GraphRAGChatContextExecutor(config={
        "mode": "v2",
        "topK": 6,
        "domain": "docs",
        "ragEnabled": True,
        "ragDomains": ["domain-1"],
        "ragSearchMode": "domain",
        "pipelineRunIds": ["run-from-config"],
    }).execute(
        {"query": "Build context", "pipeline_run_ids": "run-1,run-2"},
        mock_execution_context,
    )

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/chat-context-v2")
    assert post.await_args.kwargs["json"] == {
        "query": "Build context",
        "top_k": 6,
        "domain": "docs",
        "rag_enabled": True,
        "rag_domains": ["domain-1"],
        "rag_search_mode": "domain",
        "pipeline_run_ids": ["run-1", "run-2"],
    }
    assert result["context_text"] == "Unified context"
    assert result["quality_signals"]["total_chunks"] == 1
    assert result["cross_references"][0]["chunk_id"] == "chunk-1"


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_instrumentation_executor_collects_query_and_graph_stats(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGInstrumentationExecutor

    get = AsyncMock(side_effect=[
        _mock_response({"query_statistics": {"total_queries": 3}}),
        _mock_response({
            "graph_statistics": {"nodes": 7},
            "entity_centrality": {"GraphRAG": 0.9},
            "entity_clusters": [["GraphRAG", "Llama Nexus"]],
            "cluster_count": 1,
        }),
    ])
    mock_client.return_value.__aenter__.return_value.get = get

    result = await GraphRAGInstrumentationExecutor(config={}).execute({}, mock_execution_context)

    assert get.await_count == 2
    assert get.await_args_list[0].args[0].endswith("/api/v1/graphrag/query/statistics")
    assert get.await_args_list[1].args[0].endswith("/api/v1/graphrag/reasoning/graph-statistics")
    assert result["query_statistics"]["total_queries"] == 3
    assert result["graph_statistics"]["nodes"] == 7
    assert result["cluster_count"] == 1


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_evaluation_run_executor_posts_selected_payload(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGEvaluationRunExecutor

    post = AsyncMock(return_value=_mock_response({"precision": 0.9}))
    mock_client.return_value.__aenter__.return_value.post = post

    result = await GraphRAGEvaluationRunExecutor(config={"evaluationType": "entity_extraction"}).execute(
        {
            "test_documents": ["OpenAI built GPT."],
            "ground_truth": {"OpenAI built GPT.": [{"name": "OpenAI"}]},
        },
        mock_execution_context,
    )

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/evaluation/entity-extraction")
    assert post.await_args.kwargs["json"] == {
        "test_documents": ["OpenAI built GPT."],
        "ground_truth": {"OpenAI built GPT.": [{"name": "OpenAI"}]},
    }
    assert result["evaluation_type"] == "entity_extraction"
    assert result["success"] is True
    assert result["result"]["precision"] == 0.9


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_evaluation_run_executor_uses_get_for_graph_completeness(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGEvaluationRunExecutor

    get = AsyncMock(return_value=_mock_response({"completeness": 0.8}))
    mock_client.return_value.__aenter__.return_value.get = get

    result = await GraphRAGEvaluationRunExecutor(config={}).execute({}, mock_execution_context)

    get.assert_awaited_once()
    assert get.await_args.args[0].endswith("/api/v1/graphrag/evaluation/graph-completeness")
    assert result["evaluation_type"] == "graph_completeness"
    assert result["result"]["completeness"] == 0.8


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_test_suite_executor_runs_allowed_suite(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGTestSuiteExecutor

    post = AsyncMock(return_value=_mock_response({"passed": True}))
    mock_client.return_value.__aenter__.return_value.post = post

    result = await GraphRAGTestSuiteExecutor(config={"suite": "performance"}).execute({}, mock_execution_context)

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/evaluation/test-suite/performance")
    assert result["suite"] == "performance"
    assert result["success"] is True


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_domain_stats_executor_collects_domains(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGDomainStatsExecutor

    get = AsyncMock(return_value=_mock_response({"general": {"entities": 4}, "docs": {"entities": 7}}))
    mock_client.return_value.__aenter__.return_value.get = get

    result = await GraphRAGDomainStatsExecutor(config={}).execute({}, mock_execution_context)

    get.assert_awaited_once()
    assert get.await_args.args[0].endswith("/api/v1/graphrag/domain-stats")
    assert result["domain_count"] == 2
    assert result["domains"] == ["general", "docs"]


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_graph_explorer_executor_posts_filtered_graph_payload(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGGraphExplorerExecutor

    post = AsyncMock(return_value=_mock_response({
        "nodes": [{"id": "OpenAI"}],
        "edges": [{"source": "OpenAI", "target": "GPT"}],
        "stats": {"nodes": 1, "edges": 1},
    }))
    mock_client.return_value.__aenter__.return_value.post = post

    result = await GraphRAGGraphExplorerExecutor(config={"mode": "filtered_graph"}).execute(
        {
            "domain": "research",
            "max_entities": 25,
            "entity_types": ["organisation"],
            "relationship_types": ["created"],
        },
        mock_execution_context,
    )

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/graph")
    assert post.await_args.kwargs["json"] == {
        "domain": "research",
        "max_entities": 25,
        "entity_types": ["organisation"],
        "relationship_types": ["created"],
    }
    assert result["mode"] == "filtered_graph"
    assert result["nodes"][0]["id"] == "OpenAI"
    assert result["stats"]["edges"] == 1


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_graph_explorer_executor_forwards_top_entity_type_filter(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGGraphExplorerExecutor

    get = AsyncMock(return_value=_mock_response({
        "top_entities": [{"name": "OpenAI", "type": "organisation"}],
    }))
    mock_client.return_value.__aenter__.return_value.get = get

    result = await GraphRAGGraphExplorerExecutor(config={"mode": "top_entities"}).execute(
        {
            "domain": "research",
            "limit": 8,
            "min_occurrence": 2,
            "entity_type": "organisation",
        },
        mock_execution_context,
    )

    get.assert_awaited_once()
    assert get.await_args.args[0].endswith("/api/v1/graphrag/top-entities")
    assert get.await_args.kwargs["params"] == {
        "domain": "research",
        "limit": 8,
        "min_occurrence": 2,
        "entity_type": "organisation",
    }
    assert result["entities"][0]["name"] == "OpenAI"


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_graph_explorer_executor_posts_entity_neighborhood_payload(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGGraphExplorerExecutor

    post = AsyncMock(return_value=_mock_response({
        "entity": "GraphRAG",
        "nodes": [{"id": "GraphRAG"}],
        "edges": [],
    }))
    mock_client.return_value.__aenter__.return_value.post = post

    result = await GraphRAGGraphExplorerExecutor(config={
        "mode": "entity_neighborhood",
        "entityName": "GraphRAG",
        "domain": "research",
        "limit": 12,
    }).execute({}, mock_execution_context)

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/entity-neighborhood")
    assert post.await_args.kwargs["json"] == {
        "entity_name": "GraphRAG",
        "domain": "research",
        "limit": 12,
    }
    assert result["mode"] == "entity_neighborhood"
    assert result["nodes"][0]["id"] == "GraphRAG"
    assert result["neighborhood"]["entity"] == "GraphRAG"


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_graph_explorer_executor_fetches_communities_with_domain_filter(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGGraphExplorerExecutor

    get = AsyncMock(return_value=_mock_response({
        "domain_stats": {"research": {"communities": 1}},
        "communities": [{"id": "community-1"}],
    }))
    mock_client.return_value.__aenter__.return_value.get = get

    result = await GraphRAGGraphExplorerExecutor(config={
        "mode": "communities",
        "domain": "research",
    }).execute({}, mock_execution_context)

    get.assert_awaited_once()
    assert get.await_args.args[0].endswith("/api/v1/graphrag/communities")
    assert get.await_args.kwargs["params"] == {"domain": "research"}
    assert result["communities"][0]["id"] == "community-1"
    assert result["community_count"] == 1


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_enhanced_extraction_diagnostic_executor_posts_form_data(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGEnhancedExtractionDiagnosticExecutor

    post = AsyncMock(return_value=_mock_response({
        "success": True,
        "message": "completed",
        "entities_found": 2,
        "relationships_found": 1,
        "extraction_methods_used": ["spanbert"],
        "sample_entities": [{"name": "GraphRAG"}],
        "sample_relationships": [{"source": "GraphRAG"}],
    }))
    mock_client.return_value.__aenter__.return_value.post = post

    result = await GraphRAGEnhancedExtractionDiagnosticExecutor(config={}).execute(
        {"text": "GraphRAG augments retrieval.", "domain": "software"},
        mock_execution_context,
    )

    post.assert_awaited_once()
    assert post.await_args.args[0].endswith("/api/v1/graphrag/extraction/test-enhanced")
    assert post.await_args.kwargs["data"] == {
        "text": "GraphRAG augments retrieval.",
        "domain": "software",
    }
    assert result["success"] is True
    assert result["entities_found"] == 2
    assert result["extraction_methods_used"] == ["spanbert"]


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_graphrag_upstream_executor_normalizes_bridge_path(mock_client, mock_execution_context):
    from modules.workflow.executors.graphrag_executors import GraphRAGUpstreamExecutor

    request = AsyncMock(return_value=_mock_response({"ok": True}))
    mock_client.return_value.__aenter__.return_value.request = request

    result = await GraphRAGUpstreamExecutor(config={"method": "POST", "timeout": 15}).execute(
        {
            "path": "/api/v1/graphrag/upstream/api/query-complexity-analysis",
            "params": {"query": "compare A and B"},
            "body": {"trace": True},
        },
        mock_execution_context,
    )

    request.assert_awaited_once()
    method, url = request.await_args.args
    assert method == "POST"
    assert url.endswith("/api/v1/graphrag/upstream/api/query-complexity-analysis")
    assert request.await_args.kwargs["params"] == {"query": "compare A and B"}
    assert request.await_args.kwargs["json"] == {"trace": True}
    assert result["response"] == {"ok": True}
    assert result["path"] == "api/query-complexity-analysis"


def test_new_graphrag_executors_are_registered():
    from modules.workflow.executors import NODE_EXECUTORS

    assert "graphrag_capabilities" in NODE_EXECUTORS
    assert "graphrag_query_complexity" in NODE_EXECUTORS
    assert "enhanced_entity_extraction" in NODE_EXECUTORS
    assert "graphrag_document_ingest" in NODE_EXECUTORS
    assert "graphrag_document_list" in NODE_EXECUTORS
    assert "graphrag_document_delete" in NODE_EXECUTORS
    assert "graphrag_kg_rebuild" in NODE_EXECUTORS
    assert "graphrag_code_ingest" in NODE_EXECUTORS
    assert "graphrag_code_ingestion_status" in NODE_EXECUTORS
    assert "graphrag_code_analysis" in NODE_EXECUTORS
    assert "graphrag_code_stats" in NODE_EXECUTORS
    assert "graphrag_ner_extraction" in NODE_EXECUTORS
    assert "graphrag_relation_extraction" in NODE_EXECUTORS
    assert "graphrag_enhanced_query" in NODE_EXECUTORS
    assert "graphrag_chat_context" in NODE_EXECUTORS
    assert "graphrag_instrumentation" in NODE_EXECUTORS
    assert "graphrag_evaluation_run" in NODE_EXECUTORS
    assert "graphrag_test_suite" in NODE_EXECUTORS
    assert "graphrag_domain_stats" in NODE_EXECUTORS
    assert "graphrag_graph_explorer" in NODE_EXECUTORS
    assert "graphrag_enhanced_extraction_test" in NODE_EXECUTORS
    assert "graphrag_evaluation_status" in NODE_EXECUTORS
    assert "graphrag_feedback_submit" in NODE_EXECUTORS
    assert "graphrag_feedback_analytics" in NODE_EXECUTORS
    assert "graphrag_social_memory_ingest" in NODE_EXECUTORS
    assert "graphrag_social_memory_context" in NODE_EXECUTORS
    assert "graphrag_social_memory_facts" in NODE_EXECUTORS
    assert "graphrag_social_memory_admin" in NODE_EXECUTORS
    assert "graphrag_reddit_crawl" in NODE_EXECUTORS
    assert "graphrag_reddit_crawl_status" in NODE_EXECUTORS
    assert "graphrag_reddit_crawl_results" in NODE_EXECUTORS
    assert "graphrag_upstream" in NODE_EXECUTORS
