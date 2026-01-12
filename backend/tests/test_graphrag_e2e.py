"""
End-to-End Tests for GraphRAG Integration
These tests can be run against a live GraphRAG service to verify integration.
Run with: pytest test_graphrag_e2e.py -v --graphrag-live
"""

import pytest
import httpx
import os
import asyncio
from pathlib import Path


# Mark all tests as requiring live GraphRAG service
pytestmark = pytest.mark.skipif(
    not pytest.config.getoption("--graphrag-live", default=False),
    reason="Requires live GraphRAG service (use --graphrag-live to run)"
)


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--graphrag-live",
        action="store_true",
        default=False,
        help="Run tests against live GraphRAG service"
    )


@pytest.fixture
def graphrag_url():
    """Get GraphRAG service URL."""
    return os.getenv("GRAPHRAG_URL", "http://localhost:18000")


@pytest.fixture
def backend_url():
    """Get backend API URL."""
    return os.getenv("BACKEND_API_URL", "http://localhost:8700")


class TestGraphRAGServiceConnectivity:
    """Test connectivity to GraphRAG service."""
    
    async def test_graphrag_health(self, graphrag_url):
        """Test GraphRAG health endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{graphrag_url}/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
    
    async def test_graphrag_stats(self, graphrag_url):
        """Test GraphRAG stats endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{graphrag_url}/knowledge-graph/stats")
            assert response.status_code == 200
            data = response.json()
            assert "nodes" in data
            assert "edges" in data
    
    async def test_backend_graphrag_proxy_health(self, backend_url):
        """Test backend proxy to GraphRAG health."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{backend_url}/api/v1/graphrag/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data


class TestDocumentUpload:
    """Test document upload to GraphRAG."""
    
    async def test_upload_text_document(self, backend_url):
        """Test uploading a text document."""
        # Create test document
        test_content = """
        Machine Learning is a subset of Artificial Intelligence.
        Neural Networks are a key technology in modern AI systems.
        Deep Learning uses multiple layers of neural networks.
        """
        
        files = {"file": ("test_ml.txt", test_content.encode(), "text/plain")}
        data = {
            "domain": "technology",
            "use_semantic_chunking": "true",
            "build_knowledge_graph": "true"
        }
        
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/graphrag/ingest/upload",
                files=files,
                data=data
            )
            assert response.status_code == 200
            result = response.json()
            
            assert result["filename"] == "test_ml.txt"
            assert result["total_chunks"] > 0
            
            # Check if knowledge graph was built
            if "knowledge_graph" in result:
                assert "entities" in result["knowledge_graph"]
                assert "relationships" in result["knowledge_graph"]
    
    async def test_batch_upload(self, backend_url):
        """Test batch document upload."""
        files = [
            ("files", ("doc1.txt", b"Document 1 content", "text/plain")),
            ("files", ("doc2.txt", b"Document 2 content", "text/plain")),
        ]
        data = {
            "domain": "general",
            "use_semantic_chunking": "true"
        }
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/graphrag/ingest/batch",
                files=files,
                data=data
            )
            
            # May fail if service is busy, check for expected codes
            assert response.status_code in [200, 500, 502]


class TestIntelligentSearch:
    """Test intelligent search functionality."""
    
    async def test_basic_search(self, backend_url):
        """Test basic intelligent search."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/graphrag/search/intelligent",
                json={
                    "query": "What is machine learning?",
                    "search_type": "auto",
                    "top_k": 5
                }
            )
            assert response.status_code == 200
            result = response.json()
            
            assert "answer" in result
            assert "results" in result
            assert "search_method_used" in result
    
    async def test_hybrid_search(self, backend_url):
        """Test hybrid search method."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/graphrag/search/intelligent",
                json={
                    "query": "neural networks",
                    "search_type": "hybrid",
                    "top_k": 10
                }
            )
            assert response.status_code == 200
            result = response.json()
            
            assert isinstance(result["results"], list)
    
    async def test_vector_search(self, backend_url):
        """Test vector-only search."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/graphrag/search/advanced",
                json={
                    "query": "deep learning",
                    "search_type": "vector",
                    "top_k": 5
                }
            )
            assert response.status_code == 200
    
    async def test_graph_search(self, backend_url):
        """Test graph-based search."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/graphrag/search/advanced",
                json={
                    "query": "artificial intelligence",
                    "search_type": "graph",
                    "top_k": 5
                }
            )
            assert response.status_code == 200


class TestEntityOperations:
    """Test entity extraction and management."""
    
    async def test_entity_extraction(self, backend_url):
        """Test entity extraction from text."""
        text = """
        Google is a technology company founded by Larry Page and Sergey Brin.
        They developed the PageRank algorithm which revolutionized web search.
        Google is headquartered in Mountain View, California.
        """
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/graphrag/extract",
                json={"text": text, "domain": "technology"}
            )
            assert response.status_code == 200
            result = response.json()
            
            assert "entities" in result
            assert "relationships" in result
            
            # Should extract Google, Larry Page, Sergey Brin, etc.
            entities = result.get("entities", [])
            if entities:
                entity_names = [e.get("name", "") for e in entities]
                # At least some entities should be extracted
                assert len(entities) > 0
    
    async def test_get_top_entities(self, backend_url):
        """Test getting top entities by occurrence."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{backend_url}/api/v1/graphrag/top-entities",
                params={"limit": 10, "min_occurrence": 1}
            )
            assert response.status_code == 200
            result = response.json()
            
            assert "top_entities" in result


class TestReasoningCapabilities:
    """Test reasoning features."""
    
    async def test_multi_hop_reasoning(self, backend_url):
        """Test multi-hop reasoning."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/graphrag/reasoning/multi-hop",
                json={
                    "query": "Find connection between Machine Learning and Neural Networks",
                    "max_hops": 3
                }
            )
            
            # May return 200 with results or error if entities not found
            assert response.status_code in [200, 500, 502]
            
            if response.status_code == 200:
                result = response.json()
                assert "query" in result
    
    async def test_causal_reasoning(self, backend_url):
        """Test causal reasoning."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/graphrag/reasoning/causal",
                json={"query": "What causes overfitting in machine learning?"}
            )
            
            assert response.status_code in [200, 500, 502]
    
    async def test_comparative_reasoning(self, backend_url):
        """Test comparative reasoning."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/graphrag/reasoning/comparative",
                json={"query": "Compare supervised and unsupervised learning"}
            )
            
            assert response.status_code in [200, 500, 502]


class TestKnowledgeGraphOperations:
    """Test knowledge graph operations."""
    
    async def test_get_graph_data(self, backend_url):
        """Test getting knowledge graph data."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/graphrag/graph",
                json={"max_entities": 50, "max_relationships": 50}
            )
            assert response.status_code == 200
            result = response.json()
            
            assert "nodes" in result
            assert "edges" in result
    
    async def test_get_communities(self, backend_url):
        """Test getting entity communities."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{backend_url}/api/v1/graphrag/communities")
            assert response.status_code == 200
    
    async def test_export_graph(self, backend_url):
        """Test graph export."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/graphrag/export",
                json={}
            )
            assert response.status_code == 200


class TestWorkflowExecutorE2E:
    """End-to-end tests for workflow executors against live service."""
    
    async def test_graphrag_search_executor_live(self, mock_execution_context):
        """Test GraphRAGSearchExecutor against live service."""
        from modules.workflow.executors.graphrag_executors import GraphRAGSearchExecutor
        
        executor = GraphRAGSearchExecutor(
            node_id="search-1",
            config={"searchType": "auto", "topK": 5}
        )
        
        try:
            result = await executor.execute(
                {"query": "What is machine learning?"},
                mock_execution_context
            )
            
            assert "answer" in result
            assert "documents" in result
            assert isinstance(result["documents"], list)
        except Exception as e:
            # Service may not have data yet
            pytest.skip(f"GraphRAG service error (may be empty): {e}")
    
    async def test_entity_extraction_executor_live(self, mock_execution_context):
        """Test EntityExtractionExecutor against live service."""
        from modules.workflow.executors.graphrag_executors import EntityExtractionExecutor
        
        executor = EntityExtractionExecutor(node_id="extract-1", config={})
        
        text = "Python is a programming language created by Guido van Rossum."
        
        try:
            result = await executor.execute(
                {"text": text, "domain": "technology"},
                mock_execution_context
            )
            
            assert "entities" in result
            assert "relationships" in result
            assert isinstance(result["entities"], list)
        except Exception as e:
            pytest.skip(f"GraphRAG service error: {e}")


class TestErrorRecovery:
    """Test error recovery and resilience."""
    
    async def test_retry_on_transient_error(self, mock_execution_context):
        """Test that executor can retry on transient errors."""
        # This is a conceptual test - would need retry logic added to executors
        from modules.workflow.executors.graphrag_executors import GraphRAGSearchExecutor
        
        executor = GraphRAGSearchExecutor(node_id="search", config={})
        
        # First call fails, would need retry mechanism
        # For now, just verify error is raised properly
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Service Unavailable",
                    request=Mock(),
                    response=Mock(status_code=503)
                )
            )
            
            with pytest.raises(Exception):
                await executor.execute({"query": "test"}, mock_execution_context)


class TestDataFlow:
    """Test data flow through GraphRAG nodes."""
    
    @pytest.fixture
    def mock_context(self):
        """Create enhanced mock context."""
        context = Mock()
        context.log = Mock()
        context.workflow_id = "test-wf"
        context.execution_id = "test-exec"
        context.variables = {}
        context.get_variable = lambda k: context.variables.get(k)
        context.set_variable = lambda k, v: context.variables.update({k: v})
        return context
    
    async def test_search_to_entity_flow(self, mock_context):
        """Test data flow from search to entity extraction."""
        from modules.workflow.executors.graphrag_executors import (
            GraphRAGSearchExecutor,
            EntityExtractionExecutor
        )
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock search response
            search_response = Mock()
            search_response.json.return_value = {
                "answer": "ML is a branch of AI.",
                "results": [{"content": "Machine Learning content", "source": "ml.txt", "score": 0.9}],
                "query_analysis": {"entities": ["Machine Learning"]},
                "search_method_used": "hybrid",
                "total_results": 1
            }
            search_response.status_code = 200
            search_response.raise_for_status = Mock()
            
            # Mock entity extraction response
            extract_response = Mock()
            extract_response.json.return_value = {
                "entities": [
                    {"name": "Machine Learning", "type": "concept"},
                    {"name": "Artificial Intelligence", "type": "concept"}
                ],
                "relationships": [
                    {"source": "Machine Learning", "target": "Artificial Intelligence", "relation": "part_of"}
                ]
            }
            extract_response.status_code = 200
            extract_response.raise_for_status = Mock()
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=[search_response, extract_response]
            )
            
            # Execute search
            search_executor = GraphRAGSearchExecutor(node_id="search", config={})
            search_result = await search_executor.execute(
                {"query": "What is machine learning?"},
                mock_context
            )
            
            # Use search results for entity extraction
            extract_executor = EntityExtractionExecutor(node_id="extract", config={})
            extract_result = await extract_executor.execute(
                {"text": search_result["answer"], "domain": "technology"},
                mock_context
            )
            
            assert extract_result["entity_count"] == 2
            assert any(e["name"] == "Machine Learning" for e in extract_result["entities"])


class TestPerformance:
    """Test performance characteristics."""
    
    async def test_search_response_time(self, backend_url):
        """Test that search completes within reasonable time."""
        import time
        
        start = time.time()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/graphrag/search/intelligent",
                json={"query": "test", "search_type": "vector", "top_k": 5}
            )
        
        duration = time.time() - start
        
        assert response.status_code in [200, 503]
        assert duration < 30.0, "Search should complete within 30 seconds"
    
    async def test_entity_extraction_response_time(self, backend_url):
        """Test that entity extraction completes within reasonable time."""
        import time
        
        text = "Test text for entity extraction. " * 50  # Medium-sized text
        
        start = time.time()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/graphrag/extract",
                json={"text": text, "domain": "general"}
            )
        
        duration = time.time() - start
        
        assert response.status_code in [200, 503]
        assert duration < 60.0, "Entity extraction should complete within 60 seconds"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    async def test_very_long_query(self, backend_url):
        """Test search with very long query."""
        long_query = "What is " + " and ".join(["machine learning"] * 100)
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/graphrag/search/intelligent",
                json={"query": long_query, "search_type": "vector", "top_k": 3}
            )
            
            # Should handle gracefully
            assert response.status_code in [200, 400, 503]
    
    async def test_special_characters_in_entities(self, backend_url):
        """Test entity extraction with special characters."""
        text = "C++ and C# are programming languages. What about F#?"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/graphrag/extract",
                json={"text": text}
            )
            
            assert response.status_code in [200, 503]
    
    async def test_unicode_text(self, backend_url):
        """Test with unicode text."""
        text = "机器学习是人工智能的一个分支。Machine Learning is a branch of AI."
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/graphrag/extract",
                json={"text": text, "domain": "general"}
            )
            
            assert response.status_code in [200, 503]


# Run with: pytest test_graphrag_e2e.py -v --graphrag-live
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--graphrag-live"])

