"""
Tests for GraphRAG Workflow Executors
Tests each executor's functionality, input validation, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import httpx


@pytest.fixture
def mock_execution_context():
    """Create a mock execution context."""
    context = Mock()
    context.log = Mock()
    context.workflow_id = "test-workflow-123"
    context.execution_id = "test-exec-456"
    context.get_variable = Mock(return_value=None)
    context.set_variable = Mock()
    return context


class TestGraphRAGSearchExecutor:
    """Test GraphRAGSearchExecutor functionality."""
    
    @patch('httpx.AsyncClient')
    async def test_successful_search(self, mock_client, mock_execution_context):
        """Test successful GraphRAG search."""
        from modules.workflow.executors.graphrag_executors import GraphRAGSearchExecutor
        
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "answer": "This is the answer to your question.",
            "results": [
                {"content": "Result 1", "source": "doc1.pdf", "score": 0.92},
                {"content": "Result 2", "source": "doc2.pdf", "score": 0.87}
            ],
            "query_analysis": {
                "intent": "factual",
                "entities": ["GraphRAG", "Neo4j"],
                "keywords": ["search", "knowledge"]
            },
            "search_method_used": "hybrid",
            "total_results": 2
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        executor = GraphRAGSearchExecutor(
            node_id="search-node",
            config={"searchType": "auto", "topK": 10}
        )
        
        result = await executor.execute(
            {"query": "What is GraphRAG?"},
            mock_execution_context
        )
        
        assert result["answer"] == "This is the answer to your question."
        assert len(result["documents"]) == 2
        assert len(result["sources"]) == 2
        assert result["sources"][0] == "doc1.pdf"
        assert result["total_results"] == 2
        assert result["search_method"] == "hybrid"
        assert len(result["entities"]) == 2
        
        mock_execution_context.log.assert_called()
    
    async def test_empty_query(self, mock_execution_context):
        """Test that empty query raises error."""
        from modules.workflow.executors.graphrag_executors import GraphRAGSearchExecutor
        
        executor = GraphRAGSearchExecutor(node_id="search-node", config={})
        
        with pytest.raises(ValueError, match="Query is required"):
            await executor.execute({"query": ""}, mock_execution_context)
    
    @patch('httpx.AsyncClient')
    async def test_service_unavailable(self, mock_client, mock_execution_context):
        """Test handling when GraphRAG service is unavailable."""
        from modules.workflow.executors.graphrag_executors import GraphRAGSearchExecutor
        
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.raise_for_status = Mock(side_effect=httpx.HTTPStatusError(
            "Service Unavailable", request=Mock(), response=mock_response
        ))
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        executor = GraphRAGSearchExecutor(node_id="search-node", config={})
        
        with pytest.raises(Exception, match="GraphRAG service is not available"):
            await executor.execute({"query": "test"}, mock_execution_context)
    
    @patch('httpx.AsyncClient')
    async def test_with_domain_filter(self, mock_client, mock_execution_context):
        """Test search with domain filter."""
        from modules.workflow.executors.graphrag_executors import GraphRAGSearchExecutor
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "answer": "Domain-specific answer",
            "results": [],
            "query_analysis": {},
            "search_method_used": "graph",
            "total_results": 0
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        executor = GraphRAGSearchExecutor(node_id="search-node", config={"searchType": "graph"})
        
        result = await executor.execute(
            {"query": "test query", "domain": "technical"},
            mock_execution_context
        )
        
        assert result["search_method"] == "graph"


class TestEntityExtractionExecutor:
    """Test EntityExtractionExecutor functionality."""
    
    @patch('httpx.AsyncClient')
    async def test_successful_extraction(self, mock_client, mock_execution_context):
        """Test successful entity extraction."""
        from modules.workflow.executors.graphrag_executors import EntityExtractionExecutor
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "entities": [
                {"name": "Python", "type": "technology", "confidence": 0.95},
                {"name": "Machine Learning", "type": "concept", "confidence": 0.90}
            ],
            "relationships": [
                {"source": "Python", "target": "Machine Learning", "relation": "used_for", "confidence": 0.85}
            ]
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        executor = EntityExtractionExecutor(node_id="extract-node", config={})
        
        result = await executor.execute(
            {"text": "Python is used for Machine Learning applications.", "domain": "tech"},
            mock_execution_context
        )
        
        assert result["entity_count"] == 2
        assert result["relationship_count"] == 1
        assert result["entities"][0]["name"] == "Python"
        assert result["relationships"][0]["relation"] == "used_for"
    
    async def test_empty_text(self, mock_execution_context):
        """Test extraction with empty text returns empty results."""
        from modules.workflow.executors.graphrag_executors import EntityExtractionExecutor
        
        executor = EntityExtractionExecutor(node_id="extract-node", config={})
        
        result = await executor.execute({"text": ""}, mock_execution_context)
        
        assert result["entities"] == []
        assert result["relationships"] == []


class TestMultiHopReasoningExecutor:
    """Test MultiHopReasoningExecutor functionality."""
    
    @patch('httpx.AsyncClient')
    async def test_successful_reasoning(self, mock_client, mock_execution_context):
        """Test successful multi-hop reasoning."""
        from modules.workflow.executors.graphrag_executors import MultiHopReasoningExecutor
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "reasoning_path": [
                {"hop": 1, "entity": "OpenAI", "relation": "develops", "confidence": 0.9},
                {"hop": 2, "entity": "GPT-4", "relation": "is_a", "confidence": 0.85},
                {"hop": 3, "entity": "Language Model", "relation": "type_of", "confidence": 0.8}
            ],
            "answer": "OpenAI develops GPT-4, which is a type of Language Model.",
            "sources": [
                {"content": "Source 1", "source": "doc1.txt", "score": 0.9}
            ]
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        executor = MultiHopReasoningExecutor(
            node_id="reasoning-node",
            config={"maxHops": 5}
        )
        
        result = await executor.execute(
            {"source": "OpenAI", "target": "Language Model"},
            mock_execution_context
        )
        
        assert result["hop_count"] == 3
        assert len(result["reasoning_path"]) == 3
        assert "OpenAI" in result["answer"]
        assert len(result["sources"]) == 1
    
    async def test_missing_entities(self, mock_execution_context):
        """Test error when entities are missing."""
        from modules.workflow.executors.graphrag_executors import MultiHopReasoningExecutor
        
        executor = MultiHopReasoningExecutor(node_id="reasoning-node", config={})
        
        with pytest.raises(ValueError, match="Both source and target"):
            await executor.execute({"source": "Entity1"}, mock_execution_context)


class TestCodeIntelligenceExecutors:
    """Test code detection and search executors."""
    
    @patch('httpx.AsyncClient')
    async def test_code_detection(self, mock_client, mock_execution_context):
        """Test code detection executor."""
        from modules.workflow.executors.graphrag_executors import CodeDetectionExecutor
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "is_code": True,
            "language": "python",
            "confidence": 0.95,
            "suggested_processor": "code"
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        executor = CodeDetectionExecutor(node_id="detect-node", config={})
        
        python_code = "def hello():\n    print('Hello, world!')"
        result = await executor.execute(
            {"content": python_code, "filename": "hello.py"},
            mock_execution_context
        )
        
        assert result["is_code"] is True
        assert result["language"] == "python"
        assert result["confidence"] == 0.95
    
    @patch('httpx.AsyncClient')
    async def test_code_search(self, mock_client, mock_execution_context):
        """Test code search executor."""
        from modules.workflow.executors.graphrag_executors import CodeSearchExecutor
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {"code": "def example(): pass", "source": "file1.py"},
                {"code": "class Example: pass", "source": "file2.py"}
            ]
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        executor = CodeSearchExecutor(node_id="search-node", config={"topK": 10})
        
        result = await executor.execute(
            {"query": "example function", "language": "python"},
            mock_execution_context
        )
        
        assert len(result["code_examples"]) == 2
        assert len(result["sources"]) == 2
        assert result["total_results"] == 2


class TestReasoningExecutors:
    """Test reasoning executors."""
    
    @patch('httpx.AsyncClient')
    async def test_causal_reasoning(self, mock_client, mock_execution_context):
        """Test causal reasoning executor."""
        from modules.workflow.executors.graphrag_executors import CausalReasoningExecutor
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "causes": ["Insufficient maintenance", "Old components"],
            "effects": ["System failure", "Downtime"],
            "reasoning": "The causes lead to the effects because...",
            "answer": "System failures are caused by..."
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        executor = CausalReasoningExecutor(node_id="causal-node", config={})
        
        result = await executor.execute(
            {"query": "What causes system failures?"},
            mock_execution_context
        )
        
        assert len(result["causes"]) == 2
        assert len(result["effects"]) == 2
        assert "reasoning" in result
        assert "answer" in result
    
    @patch('httpx.AsyncClient')
    async def test_comparative_reasoning(self, mock_client, mock_execution_context):
        """Test comparative reasoning executor."""
        from modules.workflow.executors.graphrag_executors import ComparativeReasoningExecutor
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "similarities": ["Both are databases"],
            "differences": ["MongoDB is NoSQL, PostgreSQL is SQL"],
            "analysis": "Detailed comparison analysis...",
            "answer": "MongoDB and PostgreSQL are both..."
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        executor = ComparativeReasoningExecutor(node_id="compare-node", config={})
        
        result = await executor.execute(
            {"entity1": "MongoDB", "entity2": "PostgreSQL"},
            mock_execution_context
        )
        
        assert len(result["similarities"]) >= 1
        assert len(result["differences"]) >= 1
        assert "analysis" in result
        assert "answer" in result
    
    async def test_comparative_missing_entity(self, mock_execution_context):
        """Test comparative reasoning with missing entity."""
        from modules.workflow.executors.graphrag_executors import ComparativeReasoningExecutor
        
        executor = ComparativeReasoningExecutor(node_id="compare-node", config={})
        
        with pytest.raises(ValueError, match="Both entities are required"):
            await executor.execute({"entity1": "MongoDB"}, mock_execution_context)


class TestEntityLinkingExecutor:
    """Test entity linking executor."""
    
    @patch('httpx.AsyncClient')
    async def test_successful_linking(self, mock_client, mock_execution_context):
        """Test successful entity linking."""
        from modules.workflow.executors.graphrag_executors import EntityLinkingExecutor
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "linked_entities": [
                {"mention": "OpenAI", "linked_to": "OpenAI", "confidence": 0.95},
                {"mention": "GPT", "linked_to": "GPT-4", "confidence": 0.80}
            ],
            "disambiguated": [
                {"mention": "GPT", "candidates": ["GPT-3", "GPT-4"], "selected": "GPT-4"}
            ]
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        executor = EntityLinkingExecutor(node_id="link-node", config={})
        
        entities = ["OpenAI", "GPT"]
        context = "OpenAI's GPT-4 is a powerful language model."
        
        result = await executor.execute(
            {"entities": entities, "context": context},
            mock_execution_context
        )
        
        assert len(result["linked_entities"]) == 2
        assert len(result["disambiguated"]) == 1
        assert result["link_count"] == 2
    
    async def test_empty_entities(self, mock_execution_context):
        """Test linking with empty entities list."""
        from modules.workflow.executors.graphrag_executors import EntityLinkingExecutor
        
        executor = EntityLinkingExecutor(node_id="link-node", config={})
        
        result = await executor.execute({"entities": []}, mock_execution_context)
        
        assert result["linked_entities"] == []
        assert result["disambiguated"] == []


class TestExecutorConfiguration:
    """Test executor configuration handling."""
    
    def test_get_config_value_default(self):
        """Test getting config value with default."""
        from modules.workflow.executors.graphrag_executors import GraphRAGSearchExecutor
        
        executor = GraphRAGSearchExecutor(node_id="test", config={})
        
        # Should return default when key not in config
        value = executor.get_config_value("searchType", "auto")
        assert value == "auto"
    
    def test_get_config_value_override(self):
        """Test getting config value with override."""
        from modules.workflow.executors.graphrag_executors import GraphRAGSearchExecutor
        
        executor = GraphRAGSearchExecutor(
            node_id="test",
            config={"searchType": "hybrid", "topK": 20}
        )
        
        assert executor.get_config_value("searchType", "auto") == "hybrid"
        assert executor.get_config_value("topK", 10) == 20


class TestExecutorErrorHandling:
    """Test error handling across executors."""
    
    @patch('httpx.AsyncClient')
    async def test_connection_error(self, mock_client, mock_execution_context):
        """Test handling of connection errors."""
        from modules.workflow.executors.graphrag_executors import EntityExtractionExecutor
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        
        executor = EntityExtractionExecutor(node_id="extract-node", config={})
        
        with pytest.raises(Exception, match="failed"):
            await executor.execute({"text": "test text"}, mock_execution_context)
    
    @patch('httpx.AsyncClient')
    async def test_timeout_error(self, mock_client, mock_execution_context):
        """Test handling of timeout errors."""
        from modules.workflow.executors.graphrag_executors import MultiHopReasoningExecutor
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=httpx.TimeoutException("Request timed out")
        )
        
        executor = MultiHopReasoningExecutor(node_id="reasoning-node", config={})
        
        with pytest.raises(Exception, match="failed"):
            await executor.execute(
                {"source": "Entity1", "target": "Entity2"},
                mock_execution_context
            )
    
    @patch('httpx.AsyncClient')
    async def test_invalid_response(self, mock_client, mock_execution_context):
        """Test handling of invalid response format."""
        from modules.workflow.executors.graphrag_executors import GraphRAGSearchExecutor
        
        mock_response = Mock()
        mock_response.json.return_value = {}  # Empty response
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        executor = GraphRAGSearchExecutor(node_id="search-node", config={})
        
        result = await executor.execute({"query": "test"}, mock_execution_context)
        
        # Should handle gracefully with default values
        assert "answer" in result
        assert "documents" in result
        assert result["documents"] == []


class TestExecutorIntegration:
    """Test executors working together in a workflow."""
    
    @patch('httpx.AsyncClient')
    async def test_extract_then_link_workflow(self, mock_client, mock_execution_context):
        """Test workflow: extract entities then link them."""
        from modules.workflow.executors.graphrag_executors import (
            EntityExtractionExecutor,
            EntityLinkingExecutor
        )
        
        # Mock extraction response
        extract_response = Mock()
        extract_response.json.return_value = {
            "entities": [
                {"name": "Entity1", "type": "concept"},
                {"name": "Entity2", "type": "technology"}
            ],
            "relationships": []
        }
        extract_response.status_code = 200
        extract_response.raise_for_status = Mock()
        
        # Mock linking response
        link_response = Mock()
        link_response.json.return_value = {
            "linked_entities": [
                {"mention": "Entity1", "linked_to": "Entity1"},
                {"mention": "Entity2", "linked_to": "Entity2"}
            ],
            "disambiguated": []
        }
        link_response.status_code = 200
        link_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=[extract_response, link_response]
        )
        
        # Step 1: Extract
        extractor = EntityExtractionExecutor(node_id="extract", config={})
        extract_result = await extractor.execute(
            {"text": "Test text with entities"},
            mock_execution_context
        )
        
        # Step 2: Link
        linker = EntityLinkingExecutor(node_id="link", config={})
        link_result = await linker.execute(
            {"entities": extract_result["entities"], "context": "Test text"},
            mock_execution_context
        )
        
        assert len(link_result["linked_entities"]) == 2


# Run tests with coverage
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=modules.workflow.executors.graphrag_executors", "--cov-report=term-missing"])

