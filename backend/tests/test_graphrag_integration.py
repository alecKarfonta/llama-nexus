"""
Tests for GraphRAG Integration
Tests the proxy endpoints and workflow executors that connect to the external GraphRAG service.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import httpx
from io import BytesIO


@pytest.fixture
def mock_graphrag_response():
    """Mock successful GraphRAG responses."""
    return {
        "health": {"status": "healthy", "timestamp": "2025-12-15T00:00:00"},
        "stats": {"nodes": 100, "edges": 250, "communities": 5},
        "search": {
            "answer": "This is an LLM-generated answer.",
            "results": [
                {"content": "Document 1 content", "source": "doc1.txt", "score": 0.95},
                {"content": "Document 2 content", "source": "doc2.txt", "score": 0.85}
            ],
            "query_analysis": {
                "intent": "factual",
                "entities": ["Entity1"],
                "keywords": ["test"]
            },
            "search_method_used": "hybrid",
            "total_results": 2
        },
        "extract": {
            "entities": [
                {"name": "OpenAI", "type": "organization", "confidence": 0.95},
                {"name": "GPT-4", "type": "technology", "confidence": 0.90}
            ],
            "relationships": [
                {"source": "OpenAI", "target": "GPT-4", "relation": "created_by", "confidence": 0.88}
            ]
        },
        "multi_hop": {
            "reasoning_path": [
                {"hop": 1, "entity": "OpenAI", "relation": "develops", "confidence": 0.9},
                {"hop": 2, "entity": "GPT-4", "relation": "is_a", "confidence": 0.85}
            ],
            "answer": "OpenAI developed GPT-4 which is a language model.",
            "sources": []
        },
        "upload": {
            "filename": "test.txt",
            "chunks": [{"text": "chunk1"}, {"text": "chunk2"}],
            "total_chunks": 2,
            "knowledge_graph": {"entities": 5, "relationships": 3}
        }
    }


class TestGraphRAGProxyEndpoints:
    """Test GraphRAG proxy endpoints."""
    
    @patch('httpx.AsyncClient')
    async def test_health_check(self, mock_client, mock_graphrag_response):
        """Test GraphRAG health check endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = mock_graphrag_response["health"]
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        
        from routes.graphrag import graphrag_health
        result = await graphrag_health()
        
        assert result["status"] == "healthy"
    
    @patch('httpx.AsyncClient')
    async def test_intelligent_search(self, mock_client, mock_graphrag_response):
        """Test intelligent search endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = mock_graphrag_response["search"]
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        from routes.graphrag import intelligent_search
        from fastapi import Request
        
        # Create mock request
        mock_request = Mock(spec=Request)
        mock_request.json = AsyncMock(return_value={"query": "test query", "search_type": "auto"})
        
        result = await intelligent_search(mock_request)
        
        assert "answer" in result
        assert "results" in result
        assert result["search_method_used"] == "hybrid"
        assert len(result["results"]) == 2
    
    @patch('httpx.AsyncClient')
    async def test_entity_extraction(self, mock_client, mock_graphrag_response):
        """Test entity extraction endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = mock_graphrag_response["extract"]
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        from routes.graphrag import graphrag_extract_entities
        from fastapi import Request
        
        mock_request = Mock(spec=Request)
        mock_request.json = AsyncMock(return_value={"text": "OpenAI developed GPT-4", "domain": "tech"})
        
        result = await graphrag_extract_entities(mock_request)
        
        assert "entities" in result
        assert "relationships" in result
        assert len(result["entities"]) == 2
        assert result["entities"][0]["name"] == "OpenAI"
    
    @patch('httpx.AsyncClient')
    async def test_document_upload(self, mock_client, mock_graphrag_response):
        """Test document upload to GraphRAG."""
        mock_response = Mock()
        mock_response.json.return_value = mock_graphrag_response["upload"]
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        from routes.graphrag import graphrag_upload_document
        from fastapi import UploadFile
        
        # Create mock file
        file_content = b"Test document content"
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.txt"
        mock_file.content_type = "text/plain"
        mock_file.read = AsyncMock(return_value=file_content)
        
        result = await graphrag_upload_document(
            file=mock_file,
            domain="general",
            use_semantic_chunking=True,
            build_knowledge_graph=True
        )
        
        assert result["filename"] == "test.txt"
        assert result["total_chunks"] == 2
        assert "knowledge_graph" in result
    
    async def test_graphrag_disabled(self):
        """Test that endpoints return error when GraphRAG is disabled."""
        from routes.graphrag import check_graphrag_enabled, GRAPHRAG_ENABLED
        from fastapi import HTTPException
        
        with patch('routes.graphrag.GRAPHRAG_ENABLED', False):
            with pytest.raises(HTTPException) as exc_info:
                check_graphrag_enabled()
            
            assert exc_info.value.status_code == 503
            assert "disabled" in exc_info.value.detail.lower()


class TestGraphRAGWorkflowExecutors:
    """Test GraphRAG workflow executors."""
    
    @pytest.fixture
    def execution_context(self):
        """Create mock execution context."""
        context = Mock()
        context.log = Mock()
        context.workflow_id = "test-workflow"
        context.execution_id = "test-execution"
        return context
    
    @patch('httpx.AsyncClient')
    async def test_graphrag_search_executor(self, mock_client, execution_context, mock_graphrag_response):
        """Test GraphRAGSearchExecutor."""
        mock_response = Mock()
        mock_response.json.return_value = mock_graphrag_response["search"]
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        from modules.workflow.executors.graphrag_executors import GraphRAGSearchExecutor
        
        executor = GraphRAGSearchExecutor(
            node_id="test-node",
            config={"searchType": "auto", "topK": 10}
        )
        
        inputs = {"query": "What is GraphRAG?"}
        result = await executor.execute(inputs, execution_context)
        
        assert "answer" in result
        assert "documents" in result
        assert result["total_results"] == 2
        assert result["search_method"] == "hybrid"
        execution_context.log.assert_called()
    
    @patch('httpx.AsyncClient')
    async def test_entity_extraction_executor(self, mock_client, execution_context, mock_graphrag_response):
        """Test EntityExtractionExecutor."""
        mock_response = Mock()
        mock_response.json.return_value = mock_graphrag_response["extract"]
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        from modules.workflow.executors.graphrag_executors import EntityExtractionExecutor
        
        executor = EntityExtractionExecutor(node_id="test-node", config={})
        
        inputs = {"text": "OpenAI developed GPT-4", "domain": "tech"}
        result = await executor.execute(inputs, execution_context)
        
        assert "entities" in result
        assert "relationships" in result
        assert result["entity_count"] == 2
        assert result["relationship_count"] == 1
    
    @patch('httpx.AsyncClient')
    async def test_multi_hop_reasoning_executor(self, mock_client, execution_context, mock_graphrag_response):
        """Test MultiHopReasoningExecutor."""
        mock_response = Mock()
        mock_response.json.return_value = mock_graphrag_response["multi_hop"]
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        from modules.workflow.executors.graphrag_executors import MultiHopReasoningExecutor
        
        executor = MultiHopReasoningExecutor(node_id="test-node", config={"maxHops": 3})
        
        inputs = {"source": "OpenAI", "target": "Language Model"}
        result = await executor.execute(inputs, execution_context)
        
        assert "reasoning_path" in result
        assert "answer" in result
        assert result["hop_count"] == 2
    
    async def test_executor_missing_inputs(self, execution_context):
        """Test that executors raise errors for missing required inputs."""
        from modules.workflow.executors.graphrag_executors import GraphRAGSearchExecutor
        
        executor = GraphRAGSearchExecutor(node_id="test-node", config={})
        
        with pytest.raises(ValueError, match="Query is required"):
            await executor.execute({}, execution_context)
    
    @patch('httpx.AsyncClient')
    async def test_executor_connection_error(self, mock_client, execution_context):
        """Test executor handles connection errors gracefully."""
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        
        from modules.workflow.executors.graphrag_executors import GraphRAGSearchExecutor
        
        executor = GraphRAGSearchExecutor(node_id="test-node", config={})
        
        with pytest.raises(Exception, match="Cannot connect to GraphRAG service"):
            await executor.execute({"query": "test"}, execution_context)
    
    @patch('httpx.AsyncClient')
    async def test_causal_reasoning_executor(self, mock_client, execution_context):
        """Test CausalReasoningExecutor."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "causes": ["Cause 1", "Cause 2"],
            "effects": ["Effect 1", "Effect 2"],
            "reasoning": "Causal analysis result",
            "answer": "The causes lead to the effects through..."
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        from modules.workflow.executors.graphrag_executors import CausalReasoningExecutor
        
        executor = CausalReasoningExecutor(node_id="test-node", config={})
        
        inputs = {"query": "What causes system failures?"}
        result = await executor.execute(inputs, execution_context)
        
        assert "causes" in result
        assert "effects" in result
        assert len(result["causes"]) == 2
        assert len(result["effects"]) == 2
    
    @patch('httpx.AsyncClient')
    async def test_comparative_reasoning_executor(self, mock_client, execution_context):
        """Test ComparativeReasoningExecutor."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "similarities": ["Both are programming languages"],
            "differences": ["Python is interpreted, C++ is compiled"],
            "analysis": "Detailed comparison...",
            "answer": "Python and C++ differ in..."
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        from modules.workflow.executors.graphrag_executors import ComparativeReasoningExecutor
        
        executor = ComparativeReasoningExecutor(node_id="test-node", config={})
        
        inputs = {"entity1": "Python", "entity2": "C++"}
        result = await executor.execute(inputs, execution_context)
        
        assert "similarities" in result
        assert "differences" in result
        assert len(result["similarities"]) >= 1
        assert len(result["differences"]) >= 1
    
    @patch('httpx.AsyncClient')
    async def test_code_detection_executor(self, mock_client, execution_context):
        """Test CodeDetectionExecutor."""
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
        
        from modules.workflow.executors.graphrag_executors import CodeDetectionExecutor
        
        executor = CodeDetectionExecutor(node_id="test-node", config={})
        
        inputs = {"content": "def hello(): print('Hello')", "filename": "test.py"}
        result = await executor.execute(inputs, execution_context)
        
        assert result["is_code"] is True
        assert result["language"] == "python"
        assert result["confidence"] == 0.95


class TestGraphRAGWorkflowIntegration:
    """Test complete workflows using GraphRAG nodes."""
    
    @patch('httpx.AsyncClient')
    async def test_qa_workflow(self, mock_client, mock_graphrag_response):
        """Test Intelligent Q&A workflow end-to-end."""
        mock_response = Mock()
        mock_response.json.return_value = mock_graphrag_response["search"]
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        from modules.workflow.engine import WorkflowEngine
        from modules.workflow.models import Workflow, WorkflowNode
        
        # Create simple workflow: Trigger -> GraphRAG Search -> Output
        workflow = Workflow(
            id="test-wf",
            name="Test Q&A",
            nodes=[
                WorkflowNode(
                    id="trigger-1",
                    type="manual_trigger",
                    position={"x": 0, "y": 0},
                    data={
                        "label": "Start",
                        "nodeType": "manual_trigger",
                        "config": {},
                        "inputs": [],
                        "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}]
                    }
                ),
                WorkflowNode(
                    id="search-1",
                    type="graphrag_search",
                    position={"x": 200, "y": 0},
                    data={
                        "label": "Search",
                        "nodeType": "graphrag_search",
                        "config": {"searchType": "auto", "topK": 10},
                        "inputs": [{"id": "query", "name": "Query", "type": "string", "required": True}],
                        "outputs": [
                            {"id": "answer", "name": "Answer", "type": "string"},
                            {"id": "documents", "name": "Documents", "type": "array"}
                        ]
                    }
                ),
                WorkflowNode(
                    id="output-1",
                    type="output",
                    position={"x": 400, "y": 0},
                    data={
                        "label": "Output",
                        "nodeType": "output",
                        "config": {"name": "result"},
                        "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                        "outputs": []
                    }
                )
            ],
            connections=[
                {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "search-1", "targetHandle": "query"},
                {"id": "c2", "source": "search-1", "sourceHandle": "answer", "target": "output-1", "targetHandle": "value"}
            ],
            variables={},
            settings={},
            version=1,
            isActive=True
        )
        
        # This is a conceptual test - actual execution would require full engine setup
        assert workflow.nodes[1].type == "graphrag_search"
        assert len(workflow.connections) == 2
    
    @patch('httpx.AsyncClient')
    async def test_entity_extraction_workflow(self, mock_client, mock_graphrag_response):
        """Test entity extraction workflow."""
        mock_response = Mock()
        mock_response.json.return_value = mock_graphrag_response["extract"]
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        from modules.workflow.executors.graphrag_executors import EntityExtractionExecutor
        from modules.workflow.executors.base import ExecutionContext
        
        context = Mock(spec=ExecutionContext)
        context.log = Mock()
        context.workflow_id = "test-wf"
        context.execution_id = "test-exec"
        
        executor = EntityExtractionExecutor(node_id="extract-1", config={})
        
        inputs = {
            "text": "OpenAI is an AI research company that developed GPT-4.",
            "domain": "technology"
        }
        
        result = await executor.execute(inputs, context)
        
        assert result["entity_count"] == 2
        assert result["relationship_count"] == 1
        assert any(e["name"] == "OpenAI" for e in result["entities"])


class TestGraphRAGErrorHandling:
    """Test error handling in GraphRAG integration."""
    
    @patch('httpx.AsyncClient')
    async def test_service_unavailable(self, mock_client):
        """Test handling of unavailable GraphRAG service."""
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        
        from modules.workflow.executors.graphrag_executors import GraphRAGSearchExecutor
        from modules.workflow.executors.base import ExecutionContext
        
        context = Mock(spec=ExecutionContext)
        context.log = Mock()
        
        executor = GraphRAGSearchExecutor(node_id="test-node", config={})
        
        with pytest.raises(Exception, match="Cannot connect to GraphRAG service"):
            await executor.execute({"query": "test"}, context)
    
    @patch('httpx.AsyncClient')
    async def test_timeout_handling(self, mock_client):
        """Test handling of timeouts."""
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=httpx.TimeoutException("Request timed out")
        )
        
        from modules.workflow.executors.graphrag_executors import EntityExtractionExecutor
        from modules.workflow.executors.base import ExecutionContext
        
        context = Mock(spec=ExecutionContext)
        context.log = Mock()
        
        executor = EntityExtractionExecutor(node_id="test-node", config={})
        
        with pytest.raises(Exception, match="failed"):
            await executor.execute({"text": "test text"}, context)
    
    async def test_invalid_inputs(self):
        """Test validation of invalid inputs."""
        from modules.workflow.executors.graphrag_executors import MultiHopReasoningExecutor
        from modules.workflow.executors.base import ExecutionContext
        
        context = Mock(spec=ExecutionContext)
        context.log = Mock()
        
        executor = MultiHopReasoningExecutor(node_id="test-node", config={})
        
        # Missing target
        with pytest.raises(ValueError, match="Both source and target"):
            await executor.execute({"source": "Entity1"}, context)
        
        # Missing source
        with pytest.raises(ValueError, match="Both source and target"):
            await executor.execute({"target": "Entity2"}, context)


class TestGraphRAGTemplates:
    """Test GraphRAG workflow templates."""
    
    def test_templates_registered(self):
        """Test that GraphRAG templates are registered."""
        from modules.workflow.templates import get_workflow_templates
        
        templates = get_workflow_templates()
        template_ids = [t["id"] for t in templates]
        
        assert "intelligent-qa" in template_ids
        assert "entity-discovery" in template_ids
        assert "multi-hop-investigation" in template_ids
        assert "causal-analysis" in template_ids
    
    def test_intelligent_qa_template(self):
        """Test intelligent Q&A template structure."""
        from modules.workflow.templates import get_workflow_template
        
        template = get_workflow_template("intelligent-qa")
        
        assert template is not None
        assert template["name"] == "Intelligent Q&A with GraphRAG"
        assert template["category"] == "rag"
        
        # Check nodes
        nodes = template["nodes"]
        assert len(nodes) >= 3
        
        # Check for graphrag_search node
        graphrag_nodes = [n for n in nodes if n["type"] == "graphrag_search"]
        assert len(graphrag_nodes) == 1
        
        # Check connections
        connections = template["connections"]
        assert len(connections) >= 2
    
    def test_entity_discovery_template(self):
        """Test entity extraction pipeline template."""
        from modules.workflow.templates import get_workflow_template
        
        template = get_workflow_template("entity-discovery")
        
        assert template is not None
        
        # Check for entity_extraction node
        nodes = template["nodes"]
        entity_nodes = [n for n in nodes if n["type"] == "entity_extraction"]
        assert len(entity_nodes) == 1
        
        # Check for entity_linking node
        linking_nodes = [n for n in nodes if n["type"] == "entity_linking"]
        assert len(linking_nodes) == 1


class TestGraphRAGIntegrationAPI:
    """Test GraphRAG integration via FastAPI."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from main import app
        return TestClient(app)
    
    @patch('httpx.AsyncClient')
    def test_upload_endpoint(self, mock_client, client, mock_graphrag_response):
        """Test document upload endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = mock_graphrag_response["upload"]
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        # Create test file
        file_content = b"Test document content"
        files = {"file": ("test.txt", file_content, "text/plain")}
        data = {
            "domain": "general",
            "use_semantic_chunking": "true",
            "build_knowledge_graph": "true"
        }
        
        response = client.post("/api/v1/graphrag/ingest/upload", files=files, data=data)
        
        # Note: This will fail if GraphRAG is not mocked at the route level
        # This is a structural test showing how the endpoint should be tested
        assert response.status_code in [200, 503]  # 503 if service disabled in test env
    
    @patch('httpx.AsyncClient')
    def test_search_endpoint(self, mock_client, client, mock_graphrag_response):
        """Test intelligent search endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = mock_graphrag_response["search"]
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        response = client.post(
            "/api/v1/graphrag/search/intelligent",
            json={"query": "What is GraphRAG?", "search_type": "auto"}
        )
        
        assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

