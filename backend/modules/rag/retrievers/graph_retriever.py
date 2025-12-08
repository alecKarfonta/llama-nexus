"""
Graph-Enhanced Retriever

Uses knowledge graph for improved retrieval.
Combines entity extraction, graph traversal, and vector search.
"""

import logging
from typing import List, Optional, Dict, Any, Set
from collections import defaultdict

from .base import Retriever, RetrievalResult, RetrievalConfig
from .vector_retriever import VectorRetriever
from ..vector_stores.base import VectorStore
from ..embedders.base import Embedder
from ..document_manager import DocumentManager
from ..graph_rag import GraphRAG, Entity

logger = logging.getLogger(__name__)


class GraphRetriever(Retriever):
    """
    Knowledge graph-enhanced retrieval.
    
    Features:
    - Entity extraction from query
    - Graph traversal for related entities
    - Entity-document mapping
    - Combined graph + vector scoring
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        document_manager: DocumentManager,
        graph_rag: GraphRAG,
        collection_name: str = "documents",
        llm_endpoint: Optional[str] = None
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.document_manager = document_manager
        self.graph_rag = graph_rag
        self.collection_name = collection_name
        self.llm_endpoint = llm_endpoint or "http://localhost:8600/v1/chat/completions"
        
        # Base vector retriever
        self.vector_retriever = VectorRetriever(
            vector_store, embedder, document_manager, collection_name
        )
    
    async def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None
    ) -> List[RetrievalResult]:
        """Graph-enhanced retrieval"""
        config = config or RetrievalConfig()
        
        # Step 1: Extract entities from query
        query_entities = await self._extract_query_entities(query)
        logger.info(f"Extracted {len(query_entities)} entities from query")
        
        # Step 2: Find related entities via graph traversal
        related_entities = await self._get_related_entities(
            query_entities,
            depth=2
        )
        logger.info(f"Found {len(related_entities)} related entities")
        
        # Step 3: Get documents connected to entities
        entity_doc_ids = await self._get_entity_documents(
            query_entities + related_entities
        )
        logger.info(f"Found {len(entity_doc_ids)} documents via graph")
        
        # Step 4: Vector retrieval (boosted by graph connections)
        vector_results = await self.vector_retriever.retrieve(
            query,
            RetrievalConfig(
                top_k=config.top_k * 2,  # Get more for reranking
                domain_ids=config.domain_ids,
                metadata_filter=config.metadata_filter
            )
        )
        
        # Step 5: Combine scores
        final_results = self._combine_graph_vector_scores(
            vector_results,
            entity_doc_ids,
            query_entities
        )
        
        return final_results[:config.top_k]
    
    async def retrieve_batch(
        self,
        queries: List[str],
        config: Optional[RetrievalConfig] = None
    ) -> List[List[RetrievalResult]]:
        """Batch graph retrieval"""
        results = []
        for query in queries:
            result = await self.retrieve(query, config)
            results.append(result)
        return results
    
    async def _extract_query_entities(self, query: str) -> List[Entity]:
        """Extract entities from query text"""
        # First try to match known entities
        known_entities = []
        
        # Get all entities (cached in memory)
        all_entities, _ = await self.graph_rag.list_entities(limit=1000)
        
        # Simple matching
        query_lower = query.lower()
        for entity in all_entities:
            if entity.name.lower() in query_lower:
                known_entities.append(entity)
            else:
                for alias in entity.aliases:
                    if alias.lower() in query_lower:
                        known_entities.append(entity)
                        break
        
        # If no known entities found, extract new ones
        if not known_entities and self.llm_endpoint:
            try:
                extracted = await self.graph_rag.extract_entities_from_text(
                    query,
                    self.llm_endpoint
                )
                # Try to match extracted to known
                for ext_entity in extracted:
                    existing = await self.graph_rag.find_entity_by_name(ext_entity.name)
                    if existing:
                        known_entities.append(existing)
            except Exception as e:
                logger.warning(f"Entity extraction failed: {e}")
        
        return known_entities
    
    async def _get_related_entities(
        self,
        entities: List[Entity],
        depth: int = 2
    ) -> List[Entity]:
        """Get entities related to input entities via graph traversal"""
        if not entities:
            return []
        
        entity_ids = [e.id for e in entities]
        visited = set(entity_ids)
        related = []
        
        # BFS traversal
        frontier = set(entity_ids)
        
        for _ in range(depth):
            new_frontier = set()
            
            for entity_id in frontier:
                if entity_id not in self.graph_rag._adjacency:
                    continue
                
                for neighbor_id in self.graph_rag._adjacency[entity_id]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        new_frontier.add(neighbor_id)
                        
                        neighbor = await self.graph_rag.get_entity(neighbor_id)
                        if neighbor:
                            related.append(neighbor)
            
            frontier = new_frontier
        
        return related
    
    async def _get_entity_documents(
        self,
        entities: List[Entity]
    ) -> Dict[str, float]:
        """
        Get documents connected to entities.
        Returns doc_id -> graph_score mapping.
        """
        doc_scores: Dict[str, float] = defaultdict(float)
        
        for entity in entities:
            for doc_id in entity.source_documents:
                # Score based on number of entity connections
                doc_scores[doc_id] += 1.0
            
            for chunk_id in entity.source_chunks:
                # Map chunk to document
                chunk = await self.document_manager.get_chunk_by_vector_id(chunk_id)
                if chunk:
                    doc_scores[chunk.document_id] += 0.5
        
        # Normalize scores
        if doc_scores:
            max_score = max(doc_scores.values())
            doc_scores = {
                k: v / max_score
                for k, v in doc_scores.items()
            }
        
        return dict(doc_scores)
    
    def _combine_graph_vector_scores(
        self,
        vector_results: List[RetrievalResult],
        entity_doc_ids: Dict[str, float],
        query_entities: List[Entity],
        graph_weight: float = 0.3
    ) -> List[RetrievalResult]:
        """Combine graph and vector scores"""
        combined = []
        
        # Entity names for highlighting
        entity_names = {e.name.lower() for e in query_entities}
        
        for result in vector_results:
            # Base vector score
            vector_score = result.score
            
            # Graph boost
            graph_score = entity_doc_ids.get(result.document_id, 0)
            
            # Check if chunk contains any entities
            content_lower = result.content.lower()
            entity_mention_boost = 0
            mentioned_entities = []
            
            for entity in query_entities:
                if entity.name.lower() in content_lower:
                    entity_mention_boost += 0.1
                    mentioned_entities.append(entity.name)
            
            # Combined score
            final_score = (
                (1 - graph_weight) * vector_score +
                graph_weight * (graph_score + entity_mention_boost)
            )
            
            result.score = final_score
            result.metadata['retrieval_method'] = 'graph_enhanced'
            result.metadata['graph_score'] = graph_score
            result.metadata['vector_score'] = vector_score
            result.metadata['mentioned_entities'] = mentioned_entities
            
            combined.append(result)
        
        # Sort by combined score
        combined.sort(key=lambda x: x.score, reverse=True)
        
        return combined
    
    async def retrieve_with_graph_context(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None
    ) -> Dict[str, Any]:
        """
        Retrieve with additional graph context.
        Returns results plus relevant graph information.
        """
        config = config or RetrievalConfig()
        
        # Get retrieval results
        results = await self.retrieve(query, config)
        
        # Extract entities from query
        query_entities = await self._extract_query_entities(query)
        
        # Get subgraph around entities
        if query_entities:
            entity_ids = [e.id for e in query_entities]
            nodes, edges = await self.graph_rag.get_subgraph(entity_ids, depth=2)
        else:
            nodes, edges = [], []
        
        return {
            'results': [r.to_dict() for r in results],
            'query_entities': [e.to_dict() for e in query_entities],
            'graph': {
                'nodes': [{'id': n.id, 'label': n.label, 'type': n.type} for n in nodes],
                'edges': [{'source': e.source, 'target': e.target, 'label': e.label} for e in edges]
            }
        }
