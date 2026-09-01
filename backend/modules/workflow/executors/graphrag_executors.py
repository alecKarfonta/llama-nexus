"""
GraphRAG Node Executors - Knowledge graph and advanced reasoning operations
Integrates with external GraphRAG service for hybrid search, entity extraction,
multi-hop reasoning, and more.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List
from urllib.parse import quote
from uuid import uuid4
import httpx
import os
from .base import NodeExecutor, ExecutionContext


def _backend_url() -> str:
    return os.environ.get("BACKEND_API_URL", "http://localhost:8700")


def _json_or_text(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def _iso_datetime(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if value:
        return str(value)
    return datetime.now(timezone.utc).isoformat()


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip().removeprefix("r/") for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip().removeprefix("r/") for item in value if str(item).strip()]
    return [str(value).strip().removeprefix("r/")] if str(value).strip() else []


def _compact_dict(value: Dict[str, Any]) -> Dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _admin_headers(admin_api_key: Any) -> Dict[str, str] | None:
    if not admin_api_key:
        return None
    return {"X-API-Key": str(admin_api_key)}


class GraphRAGSearchExecutor(NodeExecutor):
    """Hybrid search using GraphRAG (vector + graph + keyword)"""
    
    node_type = "graphrag_search"
    display_name = "GraphRAG Hybrid Search"
    category = "rag"
    description = "Hybrid search with LLM answer generation using GraphRAG"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        query = inputs.get("query", "")
        domain = inputs.get("domain")
        
        if not query:
            raise ValueError("Query is required for GraphRAG search")
        
        search_type = self.get_config_value("searchType", "auto")
        top_k = self.get_config_value("topK", 10)
        
        context.log(f"GraphRAG search: '{query[:100]}...' with method: {search_type}")
        
        # Call backend GraphRAG proxy
        backend_url = _backend_url()
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{backend_url}/api/v1/graphrag/search/intelligent",
                    json={
                        "query": query,
                        "search_type": search_type,
                        "top_k": top_k,
                        "domain": domain,
                    }
                )
                response.raise_for_status()
                data = response.json()
            
            answer = data.get("answer", "")
            results = data.get("results", [])
            documents = [r.get("content", "") for r in results]
            sources = [r.get("source", "") for r in results]
            
            # Extract entities from query analysis if available
            query_analysis = data.get("query_analysis", {})
            entities = query_analysis.get("entities", [])
            
            context.log(f"Found {len(documents)} documents, LLM answer generated")
            
            return {
                "answer": answer,
                "documents": documents,
                "sources": sources,
                "entities": entities,
                "total_results": len(documents),
                "search_method": data.get("search_method_used", search_type),
            }
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                raise Exception(
                    "GraphRAG service is not available. "
                    "Please ensure the GraphRAG service is running at port 18000."
                )
            raise Exception(f"GraphRAG search failed: {str(e)}")
        except httpx.ConnectError:
            raise Exception(
                "Cannot connect to GraphRAG service. "
                "Check that the service is running and accessible."
            )


class EntityExtractionExecutor(NodeExecutor):
    """Extract entities and relationships using GLiNER"""
    
    node_type = "entity_extraction"
    display_name = "Extract Entities"
    category = "rag"
    description = "Extract entities and relationships using GraphRAG GLiNER"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        text = inputs.get("text", "")
        domain = inputs.get("domain", "general")
        
        if not text:
            return {"entities": [], "relationships": []}
        
        context.log(f"Extracting entities from {len(text)} chars of text")
        
        backend_url = _backend_url()
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{backend_url}/api/v1/graphrag/extract",
                    json={
                        "text": text,
                        "domain": domain,
                    }
                )
                response.raise_for_status()
                data = response.json()
            
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
            
            context.log(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
            
            return {
                "entities": entities,
                "relationships": relationships,
                "entity_count": len(entities),
                "relationship_count": len(relationships),
            }
            
        except Exception as e:
            raise Exception(f"Entity extraction failed: {str(e)}")


class MultiHopReasoningExecutor(NodeExecutor):
    """Multi-hop reasoning across knowledge graph"""
    
    node_type = "multi_hop_reasoning"
    display_name = "Multi-Hop Reasoning"
    category = "rag"
    description = "Find paths and reasoning chains between entities"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        source = inputs.get("source", "")
        target = inputs.get("target", "")
        
        if not source or not target:
            raise ValueError("Both source and target entities are required")
        
        max_hops = self.get_config_value("maxHops", 3)
        
        context.log(f"Multi-hop reasoning: {source} -> {target} (max {max_hops} hops)")
        
        backend_url = _backend_url()
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{backend_url}/api/v1/graphrag/reasoning/multi-hop",
                    json={
                        "query": f"Find connection between {source} and {target}",
                        "max_hops": max_hops,
                    }
                )
                response.raise_for_status()
                data = response.json()
            
            reasoning_path = data.get("reasoning_path", [])
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            
            context.log(f"Found reasoning path with {len(reasoning_path)} hops")
            
            return {
                "reasoning_path": reasoning_path,
                "answer": answer,
                "sources": sources,
                "hop_count": len(reasoning_path),
            }
            
        except Exception as e:
            raise Exception(f"Multi-hop reasoning failed: {str(e)}")


class CausalReasoningExecutor(NodeExecutor):
    """Find cause-effect relationships"""
    
    node_type = "causal_reasoning"
    display_name = "Causal Reasoning"
    category = "rag"
    description = "Identify cause-effect relationships using GraphRAG"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        query = inputs.get("query", "")
        
        if not query:
            raise ValueError("Query is required for causal reasoning")
        
        context.log(f"Causal reasoning: {query[:100]}...")
        
        backend_url = _backend_url()
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{backend_url}/api/v1/graphrag/reasoning/causal",
                    json={"query": query}
                )
                response.raise_for_status()
                data = response.json()
            
            causes = data.get("causes", [])
            effects = data.get("effects", [])
            reasoning = data.get("reasoning", "")
            answer = data.get("answer", reasoning)
            
            context.log(f"Found {len(causes)} causes and {len(effects)} effects")
            
            return {
                "causes": causes,
                "effects": effects,
                "reasoning": reasoning,
                "answer": answer,
            }
            
        except Exception as e:
            raise Exception(f"Causal reasoning failed: {str(e)}")


class ComparativeReasoningExecutor(NodeExecutor):
    """Compare entities or concepts"""
    
    node_type = "comparative_reasoning"
    display_name = "Comparative Analysis"
    category = "rag"
    description = "Compare entities and find similarities/differences"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        entity1 = inputs.get("entity1", "")
        entity2 = inputs.get("entity2", "")
        
        if not entity1 or not entity2:
            raise ValueError("Both entities are required for comparison")
        
        context.log(f"Comparing: {entity1} vs {entity2}")
        
        backend_url = _backend_url()
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{backend_url}/api/v1/graphrag/reasoning/comparative",
                    json={
                        "query": f"Compare {entity1} and {entity2}"
                    }
                )
                response.raise_for_status()
                data = response.json()
            
            similarities = data.get("similarities", [])
            differences = data.get("differences", [])
            analysis = data.get("analysis", "")
            answer = data.get("answer", analysis)
            
            context.log(f"Comparison complete: {len(similarities)} similarities, {len(differences)} differences")
            
            return {
                "similarities": similarities,
                "differences": differences,
                "analysis": analysis,
                "answer": answer,
            }
            
        except Exception as e:
            raise Exception(f"Comparative reasoning failed: {str(e)}")


class EntityLinkingExecutor(NodeExecutor):
    """Link and disambiguate entities"""
    
    node_type = "entity_linking"
    display_name = "Entity Linking"
    category = "rag"
    description = "Link entity mentions to knowledge graph entities"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        entities = inputs.get("entities", [])
        text_context = inputs.get("context", "")
        
        if not entities:
            return {"linked_entities": [], "disambiguated": []}
        
        context.log(f"Linking {len(entities)} entities")
        
        backend_url = _backend_url()
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{backend_url}/api/v1/graphrag/entity/link",
                    json={
                        "entities": entities,
                        "context": text_context,
                    }
                )
                response.raise_for_status()
                data = response.json()
            
            linked = data.get("linked_entities", [])
            disambiguated = data.get("disambiguated", [])
            
            context.log(f"Linked {len(linked)} entities, disambiguated {len(disambiguated)}")
            
            return {
                "linked_entities": linked,
                "disambiguated": disambiguated,
                "link_count": len(linked),
            }
            
        except Exception as e:
            raise Exception(f"Entity linking failed: {str(e)}")


class GraphRAGDocumentIngestExecutor(NodeExecutor):
    """Ingest generated or supplied text into GraphRAG"""

    node_type = "graphrag_document_ingest"
    display_name = "GraphRAG Document Ingest"
    category = "rag"
    description = "Ingest text content into GraphRAG vector store and knowledge graph"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        text = inputs.get("text") or inputs.get("content") or inputs.get("document")
        if not text:
            raise ValueError("text, content, or document is required for GraphRAG document ingest")

        filename = inputs.get("filename") or self.get_config_value("filename", "workflow-document.txt")
        payload = {
            "text": str(text),
            "filename": filename,
            "domain": inputs.get("domain", self.get_config_value("domain", "general")),
            "content_type": inputs.get("content_type", self.get_config_value("contentType", "text/plain")),
            "use_semantic_chunking": inputs.get(
                "use_semantic_chunking",
                self.get_config_value("useSemanticChunking", True),
            ),
            "build_knowledge_graph": inputs.get(
                "build_knowledge_graph",
                self.get_config_value("buildKnowledgeGraph", True),
            ),
        }

        context.log(f"Ingesting text into GraphRAG as {filename}")

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{_backend_url()}/api/v1/graphrag/ingest",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            result_map = data.get("results", {}) if isinstance(data, dict) else {}
            document_result = result_map.get(filename, {}) if isinstance(result_map, dict) else {}

            return {
                "status": data.get("status", "success") if isinstance(data, dict) else "success",
                "filename": data.get("filename", filename) if isinstance(data, dict) else filename,
                "domain": payload["domain"],
                "total_files": data.get("total_files", 1) if isinstance(data, dict) else 1,
                "chunks": document_result.get("chunks", []),
                "total_chunks": document_result.get("total_chunks", 0),
                "entities": document_result.get("entities", 0),
                "relationships": document_result.get("relationships", 0),
                "result": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG document ingest failed: {str(e)}")


class GraphRAGDocumentListExecutor(NodeExecutor):
    """List documents currently known to GraphRAG"""

    node_type = "graphrag_document_list"
    display_name = "GraphRAG Document List"
    category = "rag"
    description = "List documents currently indexed by GraphRAG"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        context.log("Listing GraphRAG documents")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(f"{_backend_url()}/api/v1/graphrag/documents")
                response.raise_for_status()
                data = response.json()

            documents = data.get("documents", []) if isinstance(data, dict) else []
            return {
                "documents": documents,
                "total_documents": data.get("total_documents", len(documents)) if isinstance(data, dict) else len(documents),
                "vector_store_documents": data.get("vector_store_documents", 0) if isinstance(data, dict) else 0,
                "knowledge_graph_documents": data.get("knowledge_graph_documents", 0) if isinstance(data, dict) else 0,
                "response": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG document list failed: {str(e)}")


class GraphRAGDocumentDeleteExecutor(NodeExecutor):
    """Delete a document from GraphRAG"""

    node_type = "graphrag_document_delete"
    display_name = "GraphRAG Document Delete"
    category = "rag"
    description = "Delete a named document from GraphRAG"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        document_name = inputs.get("document_name") or inputs.get("filename") or self.get_config_value("documentName", "")
        if not document_name:
            raise ValueError("document_name is required for GraphRAG document delete")

        context.log(f"Deleting GraphRAG document: {document_name}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.delete(
                    f"{_backend_url()}/api/v1/graphrag/documents/{quote(str(document_name), safe='')}"
                )
                response.raise_for_status()
                data = response.json()

            return {
                "document_name": document_name,
                "status": data.get("status") if isinstance(data, dict) else None,
                "message": data.get("message") if isinstance(data, dict) else None,
                "deleted": data.get("deleted", True) if isinstance(data, dict) else True,
                "response": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG document delete failed: {str(e)}")


class GraphRAGKnowledgeGraphRebuildExecutor(NodeExecutor):
    """Rebuild GraphRAG knowledge graph from stored documents"""

    node_type = "graphrag_kg_rebuild"
    display_name = "GraphRAG KG Rebuild"
    category = "rag"
    description = "Rebuild the GraphRAG knowledge graph from indexed documents"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        domain = inputs.get("domain") or self.get_config_value("domain")
        payload = _compact_dict({"domain": domain})

        context.log(f"Rebuilding GraphRAG knowledge graph{f' for {domain}' if domain else ''}")

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{_backend_url()}/api/v1/graphrag/rebuild",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            return {
                "domain": domain,
                "message": data.get("message") if isinstance(data, dict) else None,
                "documents_processed": data.get("documents_processed", 0) if isinstance(data, dict) else 0,
                "entities_extracted": data.get("entities_extracted", 0) if isinstance(data, dict) else 0,
                "relationships_extracted": data.get("relationships_extracted", 0) if isinstance(data, dict) else 0,
                "response": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG knowledge graph rebuild failed: {str(e)}")


class CodeDetectionExecutor(NodeExecutor):
    """Detect code in documents and route appropriately"""
    
    node_type = "code_detection"
    display_name = "Code Detection"
    category = "tools"
    description = "Detect code blocks in documents using GraphRAG"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        content = inputs.get("content", "")
        filename = inputs.get("filename", "")
        
        if not content:
            return {"is_code": False, "language": None, "confidence": 0}
        
        context.log(f"Detecting code in content ({len(content)} chars)")
        
        backend_url = _backend_url()
        
        try:
            # Create form data with file
            import io
            files = {"file": (filename or "content.txt", io.BytesIO(content.encode()))}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{backend_url}/api/v1/graphrag/code/detect",
                    files=files
                )
                response.raise_for_status()
                data = response.json()
            
            is_code = data.get("is_code", False)
            language = data.get("language")
            confidence = data.get("confidence", 0)
            
            context.log(f"Code detection: is_code={is_code}, language={language}, confidence={confidence}")
            
            return {
                "is_code": is_code,
                "language": language,
                "confidence": confidence,
                "suggested_processor": data.get("suggested_processor", "standard"),
            }
            
        except Exception as e:
            raise Exception(f"Code detection failed: {str(e)}")


class CodeSearchExecutor(NodeExecutor):
    """Search specifically for code examples"""
    
    node_type = "code_search"
    display_name = "Code Search"
    category = "tools"
    description = "Search for code examples using GraphRAG code intelligence"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        query = inputs.get("query", "")
        language = inputs.get("language")
        
        if not query:
            raise ValueError("Query is required for code search")
        
        top_k = self.get_config_value("topK", 10)
        
        context.log(f"Searching for code: {query[:100]}...")
        
        backend_url = _backend_url()
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{backend_url}/api/v1/graphrag/code/search",
                    json={
                        "query": query,
                        "language": language,
                        "top_k": top_k,
                    }
                )
                response.raise_for_status()
                data = response.json()
            
            results = data.get("results", [])
            code_examples = [r.get("code") or r.get("content", "") for r in results]
            sources = [r.get("source") or r.get("file_path", "") for r in results]
            
            context.log(f"Found {len(code_examples)} code examples")
            
            return {
                "code_examples": code_examples,
                "sources": sources,
                "total_results": len(results),
            }
            
        except Exception as e:
            raise Exception(f"Code search failed: {str(e)}")


class GraphRAGCodeIngestExecutor(NodeExecutor):
    """Ingest code file paths into Code-RAG"""

    node_type = "graphrag_code_ingest"
    display_name = "Code-RAG Ingest"
    category = "tools"
    description = "Start Code-RAG indexing for one or more code file paths"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        file_paths = inputs.get("file_paths") or inputs.get("files") or []
        project_name = inputs.get("project_name") or self.get_config_value("projectName", "default_project")
        language = inputs.get("language") or self.get_config_value("language")

        if isinstance(file_paths, str):
            file_paths = [file_paths]
        if not file_paths:
            raise ValueError("file_paths is required for Code-RAG ingest")
        if not project_name:
            raise ValueError("project_name is required for Code-RAG ingest")

        payload = {
            "file_paths": file_paths,
            "project_name": project_name,
            "language": language,
        }

        context.log(f"Starting Code-RAG ingest for {len(file_paths)} files in {project_name}")

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{_backend_url()}/api/v1/graphrag/code/ingest",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            return {
                "task_id": data.get("task_id"),
                "status": data.get("status"),
                "message": data.get("message"),
                "project_name": project_name,
                "file_count": len(file_paths),
                "response": data,
            }
        except Exception as e:
            raise Exception(f"Code-RAG ingest failed: {str(e)}")


class GraphRAGCodeIngestionStatusExecutor(NodeExecutor):
    """Poll Code-RAG ingestion task status"""

    node_type = "graphrag_code_ingestion_status"
    display_name = "Code-RAG Ingest Status"
    category = "tools"
    description = "Check progress for a Code-RAG indexing task"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        task_id = inputs.get("task_id") or self.get_config_value("taskId")
        if not task_id:
            raise ValueError("task_id is required for Code-RAG ingestion status")

        context.log(f"Checking Code-RAG ingestion task {task_id}")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(f"{_backend_url()}/api/v1/graphrag/code/ingest/{task_id}")
                response.raise_for_status()
                data = response.json()

            return {
                "task_id": data.get("task_id", task_id),
                "status": data.get("status"),
                "progress": data.get("progress", 0),
                "processed_files": data.get("processed_files", 0),
                "total_files": data.get("total_files", 0),
                "entities_found": data.get("entities_found", 0),
                "error": data.get("error"),
                "elapsed_time": data.get("elapsed_time", 0),
                "response": data,
            }
        except Exception as e:
            raise Exception(f"Code-RAG ingestion status failed: {str(e)}")


class GraphRAGCodeAnalysisExecutor(NodeExecutor):
    """Analyze a code file through Code-RAG parsers"""

    node_type = "graphrag_code_analysis"
    display_name = "Code-RAG Analysis"
    category = "tools"
    description = "Parse a code file and return entities and relationships from Code-RAG"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        file_path = inputs.get("file_path") or inputs.get("path") or self.get_config_value("filePath")
        if not file_path:
            raise ValueError("file_path is required for Code-RAG analysis")

        context.log(f"Analyzing code file with Code-RAG: {file_path}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{_backend_url()}/api/v1/graphrag/code/analyze",
                    json={"file_path": file_path},
                )
                response.raise_for_status()
                data = response.json()

            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
            return {
                "file_path": data.get("file_path", file_path),
                "language": data.get("language"),
                "success": data.get("success", False),
                "entities": entities,
                "relationships": relationships,
                "entity_count": len(entities),
                "relationship_count": len(relationships),
                "parse_time_ms": data.get("parse_time_ms", 0),
                "error": data.get("error"),
                "response": data,
            }
        except Exception as e:
            raise Exception(f"Code-RAG analysis failed: {str(e)}")


class GraphRAGCodeStatsExecutor(NodeExecutor):
    """Collect Code-RAG index and task statistics"""

    node_type = "graphrag_code_stats"
    display_name = "Code-RAG Statistics"
    category = "tools"
    description = "Get Code-RAG search index and ingestion task statistics"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        context.log("Collecting Code-RAG statistics")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(f"{_backend_url()}/api/v1/graphrag/code/statistics")
                response.raise_for_status()
                data = response.json()

            return {
                "search_engine": data.get("search_engine", {}),
                "active_tasks": data.get("active_tasks", 0),
                "completed_tasks": data.get("completed_tasks", 0),
                "failed_tasks": data.get("failed_tasks", 0),
                "response": data,
            }
        except Exception as e:
            raise Exception(f"Code-RAG statistics failed: {str(e)}")


class GraphRAGCapabilitiesExecutor(NodeExecutor):
    """Discover exposed GraphRAG capabilities and bridge status"""

    node_type = "graphrag_capabilities"
    display_name = "GraphRAG Capabilities"
    category = "rag"
    description = "Inspect enabled GraphRAG services, capabilities, and upstream bridge coverage"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        context.log("Inspecting GraphRAG capabilities")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{_backend_url()}/api/v1/graphrag/capabilities")
                response.raise_for_status()
                data = response.json()

            capabilities = data.get("capabilities", [])
            bridge = data.get("upstream_bridge", {})

            context.log(f"GraphRAG exposes {len(capabilities)} capability groups")

            return {
                "enabled": data.get("enabled", False),
                "services": data.get("services", {}),
                "capabilities": capabilities,
                "capability_ids": [capability.get("id") for capability in capabilities if capability.get("id")],
                "capability_count": len(capabilities),
                "bridge_available": bridge.get("available", False),
                "allowed_prefixes": list(bridge.get("allowed_prefixes", [])),
            }
        except Exception as e:
            raise Exception(f"GraphRAG capability discovery failed: {str(e)}")


class GraphRAGQueryComplexityExecutor(NodeExecutor):
    """Analyze query complexity and routing requirements"""

    node_type = "graphrag_query_complexity"
    display_name = "GraphRAG Query Complexity"
    category = "rag"
    description = "Classify GraphRAG query complexity and required reasoning strategy"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        query = inputs.get("query", "")
        if not query:
            raise ValueError("Query is required for GraphRAG query complexity analysis")

        context.log(f"Analyzing GraphRAG query complexity: {query[:100]}...")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{_backend_url()}/api/v1/graphrag/query/complexity",
                    json={"query": query},
                )
                response.raise_for_status()
                data = response.json()

            if data.get("error"):
                raise Exception(data["error"])

            return {
                "query": data.get("query", query),
                "primary_reasoning": data.get("primary_reasoning"),
                "detected_patterns": data.get("detected_patterns", []),
                "complexity_level": data.get("complexity_level"),
                "requires_multi_hop": data.get("requires_multi_hop", False),
                "analysis": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG query complexity analysis failed: {str(e)}")


class EnhancedEntityExtractionExecutor(NodeExecutor):
    """Enhanced multi-method entity and relationship extraction"""

    node_type = "enhanced_entity_extraction"
    display_name = "Enhanced Entity Extraction"
    category = "rag"
    description = "Extract entities, relationships, claims, and linked entities using GraphRAG enhanced extraction"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        text = inputs.get("text", "")
        domain = inputs.get("domain", "general")

        if not text:
            return {
                "entities": [],
                "relationships": [],
                "entity_count": 0,
                "relationship_count": 0,
                "claims": [],
                "extraction_metadata": {},
                "span_extractions": [],
                "dependency_extractions": [],
                "linked_entities": [],
            }

        payload = {
            "text": text,
            "domain": domain,
            "use_spanbert": self.get_config_value("useSpanbert", True),
            "use_dependency": self.get_config_value("useDependency", True),
            "use_entity_linking": self.get_config_value("useEntityLinking", True),
        }

        context.log(f"Running enhanced GraphRAG extraction over {len(text)} chars")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{_backend_url()}/api/v1/graphrag/extract/enhanced",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            if data.get("error"):
                raise Exception(data["error"])

            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
            context.log(f"Enhanced extraction found {len(entities)} entities and {len(relationships)} relationships")

            return {
                "entities": entities,
                "relationships": relationships,
                "entity_count": data.get("entity_count", len(entities)),
                "relationship_count": data.get("relationship_count", len(relationships)),
                "claims": data.get("claims", []),
                "extraction_metadata": data.get("extraction_metadata", {}),
                "span_extractions": data.get("span_extractions", []),
                "dependency_extractions": data.get("dependency_extractions", []),
                "linked_entities": data.get("linked_entities", []),
            }
        except Exception as e:
            raise Exception(f"Enhanced entity extraction failed: {str(e)}")


class GraphRAGNERExtractionExecutor(NodeExecutor):
    """Lower-level GLiNER entity extraction through GraphRAG"""

    node_type = "graphrag_ner_extraction"
    display_name = "GraphRAG NER Extraction"
    category = "rag"
    description = "Extract entities from one or more texts using GraphRAG's GLiNER endpoint"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        text = inputs.get("text", "")
        texts = inputs.get("texts")

        if not text and not texts:
            return {
                "entities": [],
                "entity_count": 0,
                "results": [],
                "total_texts": 0,
            }

        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                if texts:
                    context.log(f"Running GraphRAG NER batch over {len(texts)} texts")
                    response = await client.post(
                        f"{_backend_url()}/api/v1/graphrag/ner/extract-batch",
                        json={"texts": texts},
                    )
                else:
                    context.log(f"Running GraphRAG NER over {len(text)} chars")
                    response = await client.post(
                        f"{_backend_url()}/api/v1/graphrag/ner/extract",
                        json={"text": text},
                    )
                response.raise_for_status()
                data = response.json()

            if texts:
                results = data.get("results", [])
                entities = []
                for item in results:
                    if isinstance(item, dict):
                        entities.extend(item.get("entities", []))
                entity_count = sum(
                    item.get("entity_count", len(item.get("entities", [])))
                    for item in results
                    if isinstance(item, dict)
                )
                return {
                    "entities": entities,
                    "entity_count": entity_count,
                    "results": results,
                    "total_texts": data.get("total_texts", len(texts)),
                    "extraction_method": data.get("extraction_method", "gliner"),
                }

            entities = data.get("entities", [])
            return {
                "entities": entities,
                "entity_count": data.get("entity_count", len(entities)),
                "results": [{"text": data.get("text", text), "entities": entities}],
                "total_texts": 1,
                "extraction_method": data.get("extraction_method", "gliner"),
            }
        except Exception as e:
            raise Exception(f"GraphRAG NER extraction failed: {str(e)}")


class GraphRAGRelationExtractionExecutor(NodeExecutor):
    """Relationship extraction through GraphRAG"""

    node_type = "graphrag_relation_extraction"
    display_name = "GraphRAG Relationship Extraction"
    category = "rag"
    description = "Extract typed relationships from text using GraphRAG's relationship extraction endpoint"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        text = inputs.get("text", "")
        texts = inputs.get("texts")

        if texts and not isinstance(texts, list):
            texts = [texts]

        if not text and not texts:
            return {"relations": [], "relation_count": 0, "results": [], "total_texts": 0}

        payload = {
            "relations": inputs.get("relations", self.get_config_value("relations", [])),
            "entity_labels": inputs.get("entity_labels", self.get_config_value("entityLabels", [])),
            "threshold": inputs.get("threshold", self.get_config_value("threshold", 0.5)),
        }

        if texts:
            payload["texts"] = texts
            endpoint = "rel/extract-batch"
            context.log(f"Extracting GraphRAG relationships across {len(texts)} texts")
        else:
            payload["text"] = text
            endpoint = "rel/extract"
            context.log(f"Extracting GraphRAG relationships from {len(text)} chars")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{_backend_url()}/api/v1/graphrag/{endpoint}",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            if texts:
                results = data.get("results", [])
                relations = []
                for item in results:
                    if isinstance(item, dict):
                        relations.extend(item.get("relations", []))
                relation_count = sum(
                    item.get("relation_count", len(item.get("relations", [])))
                    for item in results
                    if isinstance(item, dict)
                )
                return {
                    "relations": relations,
                    "relation_count": relation_count,
                    "results": results,
                    "total_texts": data.get("total_texts", len(texts)),
                    "processing_time": data.get("processing_time", 0),
                    "model_info": data.get("model_info", {}),
                    "response": data,
                }

            relations = data.get("relations", [])
            return {
                "relations": relations,
                "relation_count": data.get("relation_count", len(relations)),
                "results": [{"text": data.get("text", text), "relations": relations}],
                "total_texts": 1,
                "processing_time": data.get("processing_time", 0),
                "model_info": data.get("model_info", {}),
                "response": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG relationship extraction failed: {str(e)}")


class GraphRAGEnhancedQueryExecutor(NodeExecutor):
    """Enhanced GraphRAG query processing with reasoning metadata"""

    node_type = "graphrag_enhanced_query"
    display_name = "GraphRAG Enhanced Query"
    category = "rag"
    description = "Process a query with graph reasoning, entity linking, and enhanced answer metadata"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        query = inputs.get("query", "")
        if not query:
            raise ValueError("Query is required for GraphRAG enhanced query")

        payload = {
            "query": query,
            "search_type": inputs.get("search_type", self.get_config_value("searchType", "hybrid")),
            "top_k": inputs.get("top_k", self.get_config_value("topK", 10)),
            "domain": inputs.get("domain"),
            "filters": inputs.get("filters"),
        }

        context.log(f"Running enhanced GraphRAG query: {query[:100]}...")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{_backend_url()}/api/v1/graphrag/query/enhanced",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            return {
                "answer": data.get("answer", ""),
                "results": data.get("results", []),
                "reasoning_paths": data.get("reasoning_paths", []),
                "inferred_relationships": data.get("inferred_relationships", []),
                "entity_clusters": data.get("entity_clusters", []),
                "confidence": data.get("confidence", 0.0),
                "explanation": data.get("explanation", ""),
                "response": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG enhanced query failed: {str(e)}")


class GraphRAGChatContextExecutor(NodeExecutor):
    """Build GraphRAG-enriched context for chat or answer generation"""

    node_type = "graphrag_chat_context"
    display_name = "GraphRAG Chat Context"
    category = "rag"
    description = "Build graph-enhanced or unified RAG plus graph context for a query"

    _ALLOWED_MODES = {"v1", "v2", "unified"}

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        query = inputs.get("query") or self.get_config_value("query", "")
        if not query:
            raise ValueError("query is required for GraphRAG chat context")

        mode = inputs.get("mode") or self.get_config_value("mode", "v2")
        if mode not in self._ALLOWED_MODES:
            raise ValueError("mode must be v1, v2, or unified")

        rag_domains = inputs.get("rag_domains", self.get_config_value("ragDomains", []))
        if isinstance(rag_domains, str):
            rag_domains = [item.strip() for item in rag_domains.split(",") if item.strip()]
        elif not isinstance(rag_domains, list):
            rag_domains = []

        pipeline_run_ids = _string_list(
            inputs.get(
                "pipeline_run_ids",
                inputs.get("pipelineRunIds", self.get_config_value("pipelineRunIds", [])),
            )
        )

        payload = {
            "query": query,
            "top_k": inputs.get("top_k", self.get_config_value("topK", 5)),
            "domain": inputs.get("domain", self.get_config_value("domain", "general")),
        }
        if pipeline_run_ids:
            payload["pipeline_run_ids"] = pipeline_run_ids

        endpoint = "chat-context"
        if mode in {"v2", "unified"}:
            endpoint = "chat-context-v2"
            payload.update({
                "rag_enabled": inputs.get("rag_enabled", self.get_config_value("ragEnabled", True)),
                "rag_domains": rag_domains,
                "rag_search_mode": inputs.get("rag_search_mode", self.get_config_value("ragSearchMode", "domain")),
            })

        context.log(f"Building GraphRAG chat context ({mode}) for query: {str(query)[:80]}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{_backend_url()}/api/v1/graphrag/{endpoint}",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            return {
                "mode": mode,
                "context_text": data.get("context_text", "") if isinstance(data, dict) else "",
                "chunks": data.get("chunks", []) if isinstance(data, dict) else [],
                "entities": data.get("entities", []) if isinstance(data, dict) else [],
                "relationships": data.get("relationships", []) if isinstance(data, dict) else [],
                "search_results": data.get("search_results", []) if isinstance(data, dict) else [],
                "graph_connections": data.get("graph_connections", []) if isinstance(data, dict) else [],
                "expanded_entities": data.get("expanded_entities", []) if isinstance(data, dict) else [],
                "cross_references": data.get("cross_references", []) if isinstance(data, dict) else [],
                "quality_signals": data.get("quality_signals", {}) if isinstance(data, dict) else {},
                "services_used": data.get("services_used", {}) if isinstance(data, dict) else {},
                "errors": data.get("errors") if isinstance(data, dict) else None,
                "source": data.get("source") if isinstance(data, dict) else None,
                "response": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG chat context failed: {str(e)}")


class GraphRAGInstrumentationExecutor(NodeExecutor):
    """Collect GraphRAG query processor and graph reasoner instrumentation"""

    node_type = "graphrag_instrumentation"
    display_name = "GraphRAG Instrumentation"
    category = "rag"
    description = "Collect GraphRAG query statistics, graph statistics, centrality, and clusters"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        context.log("Collecting GraphRAG instrumentation")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                query_response = await client.get(f"{_backend_url()}/api/v1/graphrag/query/statistics")
                graph_response = await client.get(f"{_backend_url()}/api/v1/graphrag/reasoning/graph-statistics")
                query_response.raise_for_status()
                graph_response.raise_for_status()
                query_data = query_response.json()
                graph_data = graph_response.json()

            return {
                "query_statistics": query_data.get("query_statistics", query_data),
                "graph_statistics": graph_data.get("graph_statistics", {}),
                "entity_centrality": graph_data.get("entity_centrality", {}),
                "entity_clusters": graph_data.get("entity_clusters", []),
                "cluster_count": graph_data.get("cluster_count", 0),
                "query_response": query_data,
                "graph_response": graph_data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG instrumentation collection failed: {str(e)}")


class GraphRAGEvaluationRunExecutor(NodeExecutor):
    """Run GraphRAG evaluation jobs from workflows"""

    node_type = "graphrag_evaluation_run"
    display_name = "GraphRAG Evaluation Run"
    category = "rag"
    description = "Run GraphRAG quality, retrieval, chunking, or comprehensive evaluations"

    _EVALUATION_ROUTES = {
        "entity_extraction": ("POST", "evaluation/entity-extraction"),
        "query_responses": ("POST", "evaluation/query-responses"),
        "graph_completeness": ("GET", "evaluation/graph-completeness"),
        "retrieval_relevance": ("POST", "evaluation/retrieval-relevance"),
        "comprehensive": ("POST", "evaluation/comprehensive"),
        "chunking": ("POST", "evaluation/chunking"),
        "chunking_comparative": ("POST", "evaluation/chunking/comparative"),
        "retrieval": ("POST", "evaluation/retrieval"),
        "retrieval_comparative": ("POST", "evaluation/retrieval/comparative"),
    }

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        evaluation_type = inputs.get("evaluation_type") or self.get_config_value("evaluationType", "graph_completeness")
        route_info = self._EVALUATION_ROUTES.get(evaluation_type)
        if not route_info:
            allowed = ", ".join(sorted(self._EVALUATION_ROUTES))
            raise ValueError(f"Unknown GraphRAG evaluation type '{evaluation_type}'. Use one of: {allowed}")

        method, route = route_info
        payload = inputs.get("payload")
        if payload is None:
            payload = {
                key: value
                for key, value in inputs.items()
                if key not in {"evaluation_type", "payload"} and value is not None
            }

        context.log(f"Running GraphRAG evaluation: {evaluation_type}")

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                url = f"{_backend_url()}/api/v1/graphrag/{route}"
                if method == "GET":
                    response = await client.get(url)
                else:
                    response = await client.post(url, json=payload or {})
                response.raise_for_status()
                data = response.json()

            return {
                "evaluation_type": evaluation_type,
                "result": data,
                "success": not bool(isinstance(data, dict) and data.get("error")),
            }
        except Exception as e:
            raise Exception(f"GraphRAG evaluation run failed: {str(e)}")


class GraphRAGTestSuiteExecutor(NodeExecutor):
    """Run GraphRAG internal test suites from workflows"""

    node_type = "graphrag_test_suite"
    display_name = "GraphRAG Test Suite"
    category = "rag"
    description = "Run GraphRAG unit, integration, performance, quality, or comprehensive tests"

    _ALLOWED_SUITES = {"unit", "integration", "performance", "quality", "comprehensive"}

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        suite = inputs.get("suite") or self.get_config_value("suite", "quality")
        if suite not in self._ALLOWED_SUITES:
            allowed = ", ".join(sorted(self._ALLOWED_SUITES))
            raise ValueError(f"Unknown GraphRAG test suite '{suite}'. Use one of: {allowed}")

        context.log(f"Running GraphRAG test suite: {suite}")

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{_backend_url()}/api/v1/graphrag/evaluation/test-suite/{suite}",
                )
                response.raise_for_status()
                data = response.json()

            return {
                "suite": suite,
                "result": data,
                "success": not bool(isinstance(data, dict) and data.get("error")),
            }
        except Exception as e:
            raise Exception(f"GraphRAG test suite failed: {str(e)}")


class GraphRAGDomainStatsExecutor(NodeExecutor):
    """Collect per-domain GraphRAG statistics"""

    node_type = "graphrag_domain_stats"
    display_name = "GraphRAG Domain Stats"
    category = "rag"
    description = "Get per-domain knowledge graph statistics from GraphRAG"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        context.log("Collecting GraphRAG domain statistics")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(f"{_backend_url()}/api/v1/graphrag/domain-stats")
                response.raise_for_status()
                data = response.json()

            return {
                "domain_stats": data,
                "domains": list(data.keys()) if isinstance(data, dict) else [],
                "domain_count": len(data) if isinstance(data, dict) else 0,
            }
        except Exception as e:
            raise Exception(f"GraphRAG domain statistics failed: {str(e)}")


class GraphRAGGraphExplorerExecutor(NodeExecutor):
    """Explore and export GraphRAG knowledge graph data"""

    node_type = "graphrag_graph_explorer"
    display_name = "GraphRAG Graph Explorer"
    category = "rag"
    description = "Fetch filtered graph data, graph neighborhoods, communities, top entities, top relationships, or exports"

    _ALLOWED_MODES = {
        "filtered_graph",
        "entity_neighborhood",
        "communities",
        "top_entities",
        "top_relationships",
        "export",
    }

    def _filters(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        keys = {
            "domain",
            "entity",
            "entity_name",
            "max_entities",
            "max_relationships",
            "min_occurrence",
            "min_confidence",
            "entity_types",
            "relationship_types",
            "sort_by",
            "sort_order",
            "entity_type",
            "relationship_type",
            "limit",
            "min_weight",
            "format",
        }
        config_keys = {
            "domain": "domain",
            "entity_name": "entityName",
            "max_entities": "maxEntities",
            "max_relationships": "maxRelationships",
            "min_occurrence": "minOccurrence",
            "min_confidence": "minConfidence",
            "entity_types": "entityTypes",
            "relationship_types": "relationshipTypes",
            "sort_by": "sortBy",
            "sort_order": "sortOrder",
            "entity_type": "entityType",
            "relationship_type": "relationshipType",
            "limit": "limit",
            "min_weight": "minWeight",
            "format": "format",
        }
        filters = inputs.get("filters")
        if isinstance(filters, dict):
            merged = dict(filters)
        else:
            merged = {}
        for key in keys:
            if key in inputs and inputs[key] is not None:
                merged[key] = inputs[key]
        for key, config_key in config_keys.items():
            if merged.get(key) is None:
                config_value = self.get_config_value(config_key)
                if config_value is not None:
                    merged[key] = config_value
        return {key: value for key, value in merged.items() if value is not None}

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        mode = inputs.get("mode") or self.get_config_value("mode", "filtered_graph")
        if mode not in self._ALLOWED_MODES:
            allowed = ", ".join(sorted(self._ALLOWED_MODES))
            raise ValueError(f"Unknown GraphRAG graph explorer mode '{mode}'. Use one of: {allowed}")

        filters = self._filters(inputs)
        context.log(f"Running GraphRAG graph explorer mode: {mode}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                if mode == "filtered_graph":
                    response = await client.post(f"{_backend_url()}/api/v1/graphrag/graph", json=filters)
                elif mode == "entity_neighborhood":
                    entity_name = filters.get("entity_name") or filters.get("entity")
                    if isinstance(entity_name, str):
                        entity_name = entity_name.strip()
                    if not entity_name:
                        raise ValueError("entity_name is required for GraphRAG graph explorer entity_neighborhood mode")

                    payload = _compact_dict({
                        "entity_name": entity_name,
                        "domain": filters.get("domain"),
                        "limit": filters.get("limit"),
                    })
                    response = await client.post(f"{_backend_url()}/api/v1/graphrag/entity-neighborhood", json=payload)
                elif mode == "communities":
                    params = {
                        key: value
                        for key, value in filters.items()
                        if key in {"domain"}
                    }
                    response = await client.get(f"{_backend_url()}/api/v1/graphrag/communities", params=params)
                elif mode == "top_entities":
                    params = {
                        key: value
                        for key, value in filters.items()
                        if key in {"domain", "limit", "min_occurrence", "entity_type"}
                    }
                    response = await client.get(f"{_backend_url()}/api/v1/graphrag/top-entities", params=params)
                elif mode == "top_relationships":
                    params = {
                        key: value
                        for key, value in filters.items()
                        if key in {"domain", "limit", "min_weight", "relationship_type"}
                    }
                    response = await client.get(f"{_backend_url()}/api/v1/graphrag/top-relationships", params=params)
                else:
                    response = await client.post(f"{_backend_url()}/api/v1/graphrag/export", json=filters)

                response.raise_for_status()
                data = response.json()

            nodes = data.get("nodes", data.get("entities", [])) if isinstance(data, dict) else []
            edges = data.get("edges", data.get("relationships", [])) if isinstance(data, dict) else []
            entities = data.get("top_entities", data.get("entities", nodes)) if isinstance(data, dict) else []
            relationships = data.get("top_relationships", data.get("relationships", edges)) if isinstance(data, dict) else []
            communities = data.get("communities", []) if isinstance(data, dict) else []
            community_count = (
                data.get("community_count")
                or data.get("total_communities")
                or len(communities)
            ) if isinstance(data, dict) else 0

            return {
                "mode": mode,
                "filters": filters,
                "nodes": nodes,
                "edges": edges,
                "entities": entities,
                "relationships": relationships,
                "communities": communities,
                "community_count": community_count,
                "neighborhood": data if mode == "entity_neighborhood" else {},
                "stats": data.get("stats", {}) if isinstance(data, dict) else {},
                "result": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG graph exploration failed: {str(e)}")


class GraphRAGEnhancedExtractionDiagnosticExecutor(NodeExecutor):
    """Run enhanced extraction diagnostics from workflows"""

    node_type = "graphrag_enhanced_extraction_test"
    display_name = "Enhanced Extraction Diagnostic"
    category = "rag"
    description = "Run GraphRAG's enhanced extraction diagnostic over sample text"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        text = inputs.get("text", "")
        domain = inputs.get("domain", "general")
        if not text:
            raise ValueError("Text is required for enhanced extraction diagnostics")

        context.log(f"Testing enhanced GraphRAG extraction over {len(text)} chars")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{_backend_url()}/api/v1/graphrag/extraction/test-enhanced",
                    data={"text": text, "domain": domain},
                )
                response.raise_for_status()
                data = response.json()

            return {
                "success": data.get("success", False),
                "message": data.get("message", ""),
                "entities_found": data.get("entities_found", 0),
                "relationships_found": data.get("relationships_found", 0),
                "extraction_methods_used": data.get("extraction_methods_used", []),
                "sample_entities": data.get("sample_entities", []),
                "sample_relationships": data.get("sample_relationships", []),
                "result": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG enhanced extraction diagnostic failed: {str(e)}")


class GraphRAGEvaluationStatusExecutor(NodeExecutor):
    """Inspect GraphRAG evaluation dataset and metric availability"""

    node_type = "graphrag_evaluation_status"
    display_name = "GraphRAG Evaluation Status"
    category = "rag"
    description = "Check retrieval and chunking evaluation readiness exposed by GraphRAG"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        context.log("Checking GraphRAG evaluation status")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(f"{_backend_url()}/api/v1/graphrag/evaluation/status")
                response.raise_for_status()
                data = response.json()

            return {
                "retrieval": data.get("retrieval", {}),
                "chunking": data.get("chunking", {}),
                "available": data.get("available", False),
            }
        except Exception as e:
            raise Exception(f"GraphRAG evaluation status check failed: {str(e)}")


class GraphRAGFeedbackSubmitExecutor(NodeExecutor):
    """Submit retrieval-quality feedback to GraphRAG analytics"""

    node_type = "graphrag_feedback_submit"
    display_name = "GraphRAG Feedback Submit"
    category = "rag"
    description = "Record retrieval quality feedback for a GraphRAG query"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        query = inputs.get("query") or self.get_config_value("query", "")
        if not query:
            raise ValueError("query is required for GraphRAG feedback submission")

        try:
            rating = int(inputs.get("rating", self.get_config_value("rating", 0)))
        except (TypeError, ValueError):
            raise ValueError("rating must be -1, 0, or 1")
        if rating not in {-1, 0, 1}:
            raise ValueError("rating must be -1, 0, or 1")

        payload = {
            "query": query,
            "rating": rating,
            "comment": inputs.get("comment", self.get_config_value("comment", "")),
            "quality_signals": inputs.get("quality_signals", {}),
            "source": inputs.get("source", self.get_config_value("source", "workflow")),
            "domain": inputs.get("domain", self.get_config_value("domain", "general")),
        }

        context.log(f"Submitting GraphRAG feedback for query: {str(query)[:80]}")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{_backend_url()}/api/v1/graphrag/feedback",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            return {
                "status": data.get("status") if isinstance(data, dict) else None,
                "message": data.get("message") if isinstance(data, dict) else None,
                "query": query,
                "rating": rating,
                "response": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG feedback submission failed: {str(e)}")


class GraphRAGFeedbackAnalyticsExecutor(NodeExecutor):
    """Fetch retrieval-quality feedback analytics"""

    node_type = "graphrag_feedback_analytics"
    display_name = "GraphRAG Feedback Analytics"
    category = "rag"
    description = "Fetch retrieval feedback trends, domain breakdowns, and problem queries"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        try:
            days = int(inputs.get("days", self.get_config_value("days", 30)))
        except (TypeError, ValueError):
            raise ValueError("days must be an integer")
        domain = inputs.get("domain") or self.get_config_value("domain")
        params = _compact_dict({"days": days, "domain": domain})

        context.log(f"Fetching GraphRAG feedback analytics for {days} days")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(
                    f"{_backend_url()}/api/v1/graphrag/analytics",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

            overall = data.get("overall", {}) if isinstance(data, dict) else {}
            return {
                "period_days": data.get("period_days", days) if isinstance(data, dict) else days,
                "domain_filter": data.get("domain_filter") if isinstance(data, dict) else domain,
                "overall": overall,
                "daily_trends": data.get("daily_trends", []) if isinstance(data, dict) else [],
                "domain_stats": data.get("domain_stats", []) if isinstance(data, dict) else [],
                "problem_queries": data.get("problem_queries", []) if isinstance(data, dict) else [],
                "total_feedback": overall.get("total_feedback", 0),
                "satisfaction_rate": overall.get("satisfaction_rate", 0),
                "response": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG feedback analytics failed: {str(e)}")


class GraphRAGSocialMemoryIngestExecutor(NodeExecutor):
    """Ingest conversational events into GraphRAG social memory"""

    node_type = "graphrag_social_memory_ingest"
    display_name = "Social Memory Ingest"
    category = "rag"
    description = "Store chat or conversation messages in GraphRAG social memory"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        platform = inputs.get("platform") or self.get_config_value("platform", "twitch")
        platform_user_id = inputs.get("platform_user_id") or inputs.get("user_id")
        text = inputs.get("text") or inputs.get("message")

        if not platform_user_id:
            raise ValueError("platform_user_id is required for social memory ingest")
        if not text:
            raise ValueError("text is required for social memory ingest")

        execution_id = getattr(context, "execution_id", "workflow")
        message = {
            "platform": platform,
            "platform_user_id": platform_user_id,
            "display_name": inputs.get("display_name") or inputs.get("username") or platform_user_id,
            "text": text,
            "timestamp": _iso_datetime(inputs.get("timestamp")),
            "message_id": inputs.get("message_id") or f"{execution_id}-{uuid4()}",
            "stream_id": inputs.get("stream_id"),
            "session_id": inputs.get("session_id"),
            "previous_messages": inputs.get("previous_messages", []),
            "priority": inputs.get("priority", self.get_config_value("priority", 0)),
        }
        sync = inputs.get("sync", self.get_config_value("sync", False))

        context.log(f"Ingesting social memory message for {platform}:{platform_user_id}")

        try:
            async with httpx.AsyncClient(timeout=300.0 if sync else 120.0) as client:
                response = await client.post(
                    f"{_backend_url()}/api/v1/graphrag/social-memory/ingest",
                    params={"sync": sync},
                    json=message,
                )
                response.raise_for_status()
                data = response.json()

            context.log(
                f"Social memory ingest queued={data.get('queued', False)} "
                f"facts_written={data.get('facts_written', 0)}"
            )

            return {
                "person_id": data.get("person_id"),
                "episode_id": data.get("episode_id"),
                "facts_written": data.get("facts_written", 0),
                "queued": data.get("queued", False),
                "skipped_duplicate": data.get("skipped_duplicate", False),
                "queue_message_id": data.get("queue_message_id"),
                "message": message,
                "response": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG social memory ingest failed: {str(e)}")


class GraphRAGSocialMemoryContextExecutor(NodeExecutor):
    """Retrieve or explain personalized GraphRAG social memory context"""

    node_type = "graphrag_social_memory_context"
    display_name = "Social Memory Context"
    category = "rag"
    description = "Retrieve personalized context or retrieval traces from GraphRAG social memory"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        platform = inputs.get("platform") or self.get_config_value("platform", "twitch")
        platform_user_id = inputs.get("platform_user_id") or inputs.get("user_id")
        query = inputs.get("query")
        mode = inputs.get("mode") or self.get_config_value("mode", "context")

        if not platform_user_id:
            raise ValueError("platform_user_id is required for social memory context")
        if mode not in {"context", "explain"}:
            raise ValueError("mode must be context or explain")
        if mode == "explain" and not query:
            raise ValueError("query is required when social memory mode is explain")

        params = {
            "platform": platform,
            "platform_user_id": platform_user_id,
            "query": query,
            "max_facts": inputs.get("max_facts", self.get_config_value("maxFacts", 25)),
        }
        if mode == "context":
            params["include_recent_episodes"] = inputs.get(
                "include_recent_episodes",
                self.get_config_value("includeRecentEpisodes", 3),
            )
            params["as_of"] = _iso_datetime(inputs.get("as_of")) if inputs.get("as_of") else None
        params = {key: value for key, value in params.items() if value is not None}
        endpoint = "explain" if mode == "explain" else "context"

        context.log(f"Fetching social memory {endpoint} for {platform}:{platform_user_id}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.get(
                    f"{_backend_url()}/api/v1/graphrag/social-memory/{endpoint}",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

            facts = data.get("facts", []) if isinstance(data, dict) else []
            fact_ids = data.get("fact_ids", []) if isinstance(data, dict) else []
            context.log(f"Social memory {endpoint} returned {len(facts) or len(fact_ids)} facts")

            return {
                "person_id": data.get("person_id") if isinstance(data, dict) else None,
                "formatted": data.get("formatted", "") if isinstance(data, dict) else "",
                "facts": facts,
                "recent_episodes": data.get("recent_episodes", []) if isinstance(data, dict) else [],
                "retrieval_path": data.get("retrieval_path") if isinstance(data, dict) else None,
                "cache_hit": data.get("cache_hit", False) if isinstance(data, dict) else False,
                "fact_ids": fact_ids,
                "mode": mode,
                "trace": data if mode == "explain" else {},
                "response": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG social memory context failed: {str(e)}")


class GraphRAGSocialMemoryFactsExecutor(NodeExecutor):
    """List known facts for a resolved GraphRAG social-memory person"""

    node_type = "graphrag_social_memory_facts"
    display_name = "Social Memory Facts"
    category = "rag"
    description = "List known social-memory facts for a resolved person"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        person_id = inputs.get("person_id") or self.get_config_value("personId", "")
        if not person_id:
            raise ValueError("person_id is required for social memory facts")

        params = _compact_dict({
            "as_of": _iso_datetime(inputs.get("as_of")) if inputs.get("as_of") else None,
        })

        context.log(f"Fetching social memory facts for {person_id}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.get(
                    f"{_backend_url()}/api/v1/graphrag/social-memory/person/{person_id}/facts",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

            if isinstance(data, dict):
                facts = data.get("facts", [])
            elif isinstance(data, list):
                facts = data
            else:
                facts = []
            return {
                "person_id": data.get("person_id", person_id) if isinstance(data, dict) else person_id,
                "facts": facts,
                "fact_count": len(facts),
                "as_of": params.get("as_of"),
                "response": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG social memory facts failed: {str(e)}")


class GraphRAGSocialMemoryAdminExecutor(NodeExecutor):
    """Run advanced GraphRAG social-memory stewardship operations"""

    node_type = "graphrag_social_memory_admin"
    display_name = "Social Memory Admin"
    category = "rag"
    description = "Run migration, merge, correction, replay, evaluation, and community-report operations"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        mode = inputs.get("mode") or self.get_config_value("mode", "community_report")
        admin_api_key = inputs.get("admin_api_key") or self.get_config_value("adminApiKey")
        payload = inputs.get("payload") if isinstance(inputs.get("payload"), dict) else {}
        headers = _admin_headers(admin_api_key)

        method = "GET"
        path = "social-memory/admin/community-report"
        params: Dict[str, Any] | None = None
        json_payload: Dict[str, Any] | None = None
        timeout = 300.0

        if mode == "migrate":
            method = "POST"
            path = "social-memory/admin/migrate"
        elif mode == "merge_identities":
            method = "POST"
            path = "social-memory/identities/merge"
            json_payload = {
                "primary_person_id": inputs.get("primary_person_id") or payload.get("primary_person_id"),
                "secondary_person_id": inputs.get("secondary_person_id") or payload.get("secondary_person_id"),
                "confidence": inputs.get("confidence", payload.get("confidence")),
                "reason": inputs.get("reason", payload.get("reason")),
            }
            if not json_payload["primary_person_id"] or not json_payload["secondary_person_id"]:
                raise ValueError("primary_person_id and secondary_person_id are required for identity merge")
            if json_payload["confidence"] is None:
                raise ValueError("confidence is required for identity merge")
            json_payload = _compact_dict(json_payload)
            timeout = 120.0
        elif mode == "correct_fact":
            method = "POST"
            path = "social-memory/facts/correct"
            json_payload = {
                "fact_id": inputs.get("fact_id") or payload.get("fact_id"),
                "new_fact_text": inputs.get("new_fact_text") or payload.get("new_fact_text"),
                "corrected_by": inputs.get("corrected_by") or payload.get("corrected_by"),
                "predicate": inputs.get("predicate", payload.get("predicate")),
                "entity_name": inputs.get("entity_name", payload.get("entity_name")),
                "entity_type": inputs.get("entity_type", payload.get("entity_type")),
            }
            if not json_payload["fact_id"] or not json_payload["new_fact_text"] or not json_payload["corrected_by"]:
                raise ValueError("fact_id, new_fact_text, and corrected_by are required for fact correction")
            json_payload = _compact_dict(json_payload)
            timeout = 120.0
        elif mode == "replay_transcript":
            method = "POST"
            path = "social-memory/admin/replay"
            json_payload = {
                "transcript_path": inputs.get("transcript_path") or payload.get("transcript_path"),
                "sandbox_tag": inputs.get("sandbox_tag", payload.get("sandbox_tag")),
                "sync": inputs.get("sync", payload.get("sync", True)),
            }
            if not json_payload["transcript_path"]:
                raise ValueError("transcript_path is required for transcript replay")
            json_payload = _compact_dict(json_payload)
            timeout = 600.0
        elif mode == "purge_replay":
            method = "DELETE"
            sandbox_tag = inputs.get("sandbox_tag") or payload.get("sandbox_tag")
            if not sandbox_tag:
                raise ValueError("sandbox_tag is required for replay purge")
            path = f"social-memory/admin/replay/{sandbox_tag}"
        elif mode == "run_eval":
            method = "POST"
            path = "social-memory/admin/eval"
            params = _compact_dict({
                "suite": inputs.get("suite", payload.get("suite", self.get_config_value("suite", "all"))),
                "baseline_path": inputs.get("baseline_path", payload.get("baseline_path")),
            })
            timeout = 600.0
        elif mode == "community_report":
            method = "GET"
            path = "social-memory/admin/community-report"
        else:
            raise ValueError(
                "mode must be migrate, merge_identities, correct_fact, replay_transcript, "
                "purge_replay, run_eval, or community_report"
            )

        context.log(f"Running social memory admin mode: {mode}")

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(
                    method,
                    f"{_backend_url()}/api/v1/graphrag/{path}",
                    params=params,
                    json=json_payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = _json_or_text(response)

            return {
                "mode": mode,
                "status_code": response.status_code,
                "result": data,
                "response": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG social memory admin failed: {str(e)}")


class GraphRAGRedditCrawlExecutor(NodeExecutor):
    """Start a GraphRAG Reddit crawl with optional auto-ingestion"""

    node_type = "graphrag_reddit_crawl"
    display_name = "GraphRAG Reddit Crawl"
    category = "rag"
    description = "Start a Reddit crawl through GraphRAG's crawler service"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        query = inputs.get("query") or self.get_config_value("query", "")
        if not query:
            raise ValueError("query is required for GraphRAG Reddit crawl")

        subreddits = _string_list(inputs.get("subreddits", self.get_config_value("subreddits", [])))
        payload = {
            "query": query,
            "subreddits": subreddits or None,
            "sort_by": inputs.get("sort_by", self.get_config_value("sortBy", "relevance")),
            "time_filter": inputs.get("time_filter", self.get_config_value("timeFilter", "month")),
            "max_posts": inputs.get("max_posts", self.get_config_value("maxPosts", 50)),
            "max_comments_per_post": inputs.get(
                "max_comments_per_post",
                self.get_config_value("maxCommentsPerPost", 20),
            ),
            "comment_depth": inputs.get("comment_depth", self.get_config_value("commentDepth", 2)),
            "include_replies": inputs.get("include_replies", self.get_config_value("includeReplies", True)),
            "auto_ingest": inputs.get("auto_ingest", self.get_config_value("autoIngest", True)),
            "filter_nsfw": inputs.get("filter_nsfw", self.get_config_value("filterNsfw", True)),
        }

        context.log(f"Starting GraphRAG Reddit crawl for '{str(query)[:80]}'")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{_backend_url()}/api/v1/graphrag/reddit/crawl",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            return {
                "crawl_id": data.get("crawl_id"),
                "status": data.get("status"),
                "posts_found": data.get("posts_found", 0),
                "comments_found": data.get("comments_found", 0),
                "total_content_length": data.get("total_content_length", 0),
                "estimated_processing_time": data.get("estimated_processing_time", ""),
                "ingestion_status": data.get("ingestion_status"),
                "query": query,
                "subreddits": subreddits,
                "response": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG Reddit crawl failed: {str(e)}")


class GraphRAGRedditCrawlStatusExecutor(NodeExecutor):
    """Poll a GraphRAG Reddit crawl"""

    node_type = "graphrag_reddit_crawl_status"
    display_name = "Reddit Crawl Status"
    category = "rag"
    description = "Poll status for a GraphRAG Reddit crawl"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        crawl_id = inputs.get("crawl_id") or self.get_config_value("crawlId", "")
        if not crawl_id:
            raise ValueError("crawl_id is required for GraphRAG Reddit crawl status")

        context.log(f"Polling GraphRAG Reddit crawl status: {crawl_id}")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(
                    f"{_backend_url()}/api/v1/graphrag/reddit/crawl/{crawl_id}/status"
                )
                response.raise_for_status()
                data = response.json()

            return {
                "crawl_id": data.get("crawl_id", crawl_id),
                "status": data.get("status"),
                "progress": data.get("progress", 0),
                "posts_processed": data.get("posts_processed", 0),
                "comments_processed": data.get("comments_processed", 0),
                "current_post": data.get("current_post"),
                "error": data.get("error"),
                "response": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG Reddit crawl status failed: {str(e)}")


class GraphRAGRedditCrawlResultsExecutor(NodeExecutor):
    """Fetch completed GraphRAG Reddit crawl results"""

    node_type = "graphrag_reddit_crawl_results"
    display_name = "Reddit Crawl Results"
    category = "rag"
    description = "Fetch results and ingestion status for a completed GraphRAG Reddit crawl"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        crawl_id = inputs.get("crawl_id") or self.get_config_value("crawlId", "")
        if not crawl_id:
            raise ValueError("crawl_id is required for GraphRAG Reddit crawl results")

        context.log(f"Fetching GraphRAG Reddit crawl results: {crawl_id}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.get(
                    f"{_backend_url()}/api/v1/graphrag/reddit/crawl/{crawl_id}/results"
                )
                response.raise_for_status()
                data = response.json()

            results = data.get("results", []) if isinstance(data, dict) else []
            statistics = data.get("statistics", {}) if isinstance(data, dict) else {}
            return {
                "crawl_id": data.get("crawl_id", crawl_id) if isinstance(data, dict) else crawl_id,
                "status": data.get("status") if isinstance(data, dict) else None,
                "results": results,
                "result_count": len(results),
                "statistics": statistics,
                "ingestion_status": data.get("ingestion_status") if isinstance(data, dict) else None,
                "response": data,
            }
        except Exception as e:
            raise Exception(f"GraphRAG Reddit crawl results failed: {str(e)}")


class GraphRAGUpstreamExecutor(NodeExecutor):
    """Call an allowlisted GraphRAG upstream endpoint through llama-nexus"""

    node_type = "graphrag_upstream"
    display_name = "GraphRAG Upstream"
    category = "rag"
    description = "Call an allowlisted GraphRAG endpoint through the llama-nexus upstream bridge"

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        path = inputs.get("path") or self.get_config_value("path", "")
        method = (inputs.get("method") or self.get_config_value("method", "GET")).upper()
        params = inputs.get("params") or self.get_config_value("params", {})
        body = inputs.get("body")
        timeout = self.get_config_value("timeout", 120)

        if not path:
            raise ValueError("Path is required for GraphRAG upstream calls")
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError("Method must be GET, POST, PUT, PATCH, or DELETE")

        normalized_path = str(path).strip()
        bridge_prefixes = (
            "/api/v1/graphrag/upstream/",
            "api/v1/graphrag/upstream/",
        )
        for prefix in bridge_prefixes:
            if normalized_path.startswith(prefix):
                normalized_path = normalized_path[len(prefix):]
                break
        normalized_path = normalized_path.lstrip("/")

        context.log(f"Calling GraphRAG upstream bridge: {method} {normalized_path}")

        request_kwargs: Dict[str, Any] = {"params": params}
        if method in {"POST", "PUT", "PATCH"} and body is not None:
            if isinstance(body, (dict, list)):
                request_kwargs["json"] = body
            else:
                request_kwargs["content"] = str(body)

        try:
            async with httpx.AsyncClient(timeout=float(timeout)) as client:
                response = await client.request(
                    method,
                    f"{_backend_url()}/api/v1/graphrag/upstream/{normalized_path}",
                    **request_kwargs,
                )

            payload = _json_or_text(response)
            status_code = getattr(response, "status_code", 200)
            context.log(f"GraphRAG upstream response: {status_code}")

            return {
                "response": payload,
                "status_code": status_code,
                "path": normalized_path,
                "method": method,
            }
        except Exception as e:
            raise Exception(f"GraphRAG upstream call failed: {str(e)}")
