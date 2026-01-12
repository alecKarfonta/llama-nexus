"""
GraphRAG Node Executors - Knowledge graph and advanced reasoning operations
Integrates with external GraphRAG service for hybrid search, entity extraction,
multi-hop reasoning, and more.
"""

from typing import Dict, Any, List
import httpx
import os
from .base import NodeExecutor, ExecutionContext


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
        backend_url = os.environ.get("BACKEND_API_URL", "http://localhost:8700")
        
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
        
        backend_url = os.environ.get("BACKEND_API_URL", "http://localhost:8700")
        
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
        
        backend_url = os.environ.get("BACKEND_API_URL", "http://localhost:8700")
        
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
        
        backend_url = os.environ.get("BACKEND_API_URL", "http://localhost:8700")
        
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
        
        backend_url = os.environ.get("BACKEND_API_URL", "http://localhost:8700")
        
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
        
        backend_url = os.environ.get("BACKEND_API_URL", "http://localhost:8700")
        
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
        
        backend_url = os.environ.get("BACKEND_API_URL", "http://localhost:8700")
        
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
        
        backend_url = os.environ.get("BACKEND_API_URL", "http://localhost:8700")
        
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
            code_examples = [r.get("code", "") for r in results]
            sources = [r.get("source", "") for r in results]
            
            context.log(f"Found {len(code_examples)} code examples")
            
            return {
                "code_examples": code_examples,
                "sources": sources,
                "total_results": len(results),
            }
            
        except Exception as e:
            raise Exception(f"Code search failed: {str(e)}")

