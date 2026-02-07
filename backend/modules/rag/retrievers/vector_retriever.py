"""
Vector-Based Retriever

Dense vector similarity search with optional MMR for diversity.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
import numpy as np

from .base import Retriever, RetrievalResult, RetrievalConfig, RetrievalStrategy
from ..vector_stores.base import VectorStore, SearchResult
from ..embedders.base import Embedder
from ..document_manager import DocumentManager

logger = logging.getLogger(__name__)


class VectorRetriever(Retriever):
    """
    Vector-based document retrieval.
    
    Features:
    - Dense vector similarity search
    - Maximum Marginal Relevance (MMR) for diversity
    - Multi-query retrieval
    - HyDE (Hypothetical Document Embeddings)
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        document_manager: DocumentManager,
        collection_name: str = "documents"
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.document_manager = document_manager
        self.collection_name = collection_name
    
    async def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None
    ) -> List[RetrievalResult]:
        """Retrieve documents using vector similarity"""
        config = config or RetrievalConfig()
        
        # Get query embedding
        query_embedding = await self.embedder.embed_query(query)
        
        # Build filter
        filter_dict = self._build_filter(config)
        
        # Determine how many to retrieve
        retrieve_k = config.rerank_top_k if config.rerank else config.top_k
        
        # Search
        if config.strategy == RetrievalStrategy.MMR:
            results = await self._mmr_search(
                query_embedding,
                retrieve_k,
                config.mmr_lambda,
                filter_dict
            )
        else:
            search_results = await self.vector_store.search(
                self.collection_name,
                query_embedding,
                limit=retrieve_k,
                filter=filter_dict,
                score_threshold=config.score_threshold
            )
            results = await self._convert_results(search_results)
        
        # Rerank if enabled
        if config.rerank:
            results = await self._rerank(query, results, config.top_k)
        
        return results[:config.top_k]
    
    async def retrieve_batch(
        self,
        queries: List[str],
        config: Optional[RetrievalConfig] = None
    ) -> List[List[RetrievalResult]]:
        """Retrieve for multiple queries"""
        config = config or RetrievalConfig()
        
        # Embed all queries
        embed_result = await self.embedder.embed(queries)
        query_embeddings = embed_result.embeddings
        
        # Build filter
        filter_dict = self._build_filter(config)
        
        # Batch search
        batch_results = await self.vector_store.search_batch(
            self.collection_name,
            query_embeddings,
            limit=config.top_k,
            filter=filter_dict
        )
        
        # Convert results
        all_results = []
        for search_results in batch_results:
            results = await self._convert_results(search_results)
            all_results.append(results)
        
        return all_results
    
    async def _mmr_search(
        self,
        query_embedding: List[float],
        k: int,
        lambda_param: float,
        filter_dict: Optional[Dict[str, Any]]
    ) -> List[RetrievalResult]:
        """
        Maximum Marginal Relevance search.
        Balances relevance with diversity.
        """
        # Get more candidates than needed
        candidates_k = min(k * 3, 100)
        
        search_results = await self.vector_store.search(
            self.collection_name,
            query_embedding,
            limit=candidates_k,
            filter=filter_dict,
            with_vectors=True
        )
        
        if not search_results:
            return []
        
        # Convert to numpy for efficient computation
        query_vec = np.array(query_embedding)
        
        # Extract candidate info
        candidates = []
        for result in search_results:
            if result.vector:
                candidates.append({
                    'result': result,
                    'vector': np.array(result.vector),
                    'score': result.score
                })
        
        if not candidates:
            return await self._convert_results(search_results[:k])
        
        # MMR selection
        selected = []
        selected_vecs = []
        
        while len(selected) < k and candidates:
            best_idx = -1
            best_mmr = -float('inf')
            
            for i, cand in enumerate(candidates):
                # Relevance to query
                relevance = cand['score']
                
                # Max similarity to already selected
                if selected_vecs:
                    similarities = [
                        np.dot(cand['vector'], sel_vec) / 
                        (np.linalg.norm(cand['vector']) * np.linalg.norm(sel_vec))
                        for sel_vec in selected_vecs
                    ]
                    max_sim = max(similarities)
                else:
                    max_sim = 0
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            
            if best_idx >= 0:
                best_cand = candidates.pop(best_idx)
                selected.append(best_cand['result'])
                selected_vecs.append(best_cand['vector'])
        
        return await self._convert_results(selected)
    
    async def _convert_results(
        self,
        search_results: List[SearchResult]
    ) -> List[RetrievalResult]:
        """Convert search results to retrieval results with metadata"""
        # Fetch all document metadata in parallel
        doc_ids = [sr.payload.get('document_id') for sr in search_results]
        unique_doc_ids = list(set(did for did in doc_ids if did))
        
        if unique_doc_ids:
            docs = await asyncio.gather(
                *[self.document_manager.get_document(did) for did in unique_doc_ids]
            )
            doc_map = {did: doc for did, doc in zip(unique_doc_ids, docs) if doc}
        else:
            doc_map = {}
        
        results = []
        for sr in search_results:
            payload = sr.payload
            doc_id = payload.get('document_id')
            doc = doc_map.get(doc_id)
            doc_name = doc.name if doc else None
            domain_id = doc.domain_id if doc else None
            
            results.append(RetrievalResult(
                chunk_id=sr.id,
                document_id=payload.get('document_id', ''),
                content=payload.get('content', ''),
                score=sr.score,
                domain_id=domain_id or payload.get('domain_id'),
                document_name=doc_name or payload.get('document_name'),
                chunk_index=payload.get('chunk_index'),
                total_chunks=payload.get('total_chunks'),
                section_header=payload.get('section_header'),
                metadata={
                    k: v for k, v in payload.items()
                    if k not in ['content', 'document_id', 'domain_id', 
                                'chunk_index', 'total_chunks', 'section_header']
                }
            ))
        
        return results
    
    def _build_filter(self, config: RetrievalConfig) -> Optional[Dict[str, Any]]:
        """Build filter from config"""
        filter_dict = {}
        
        if config.domain_ids:
            if len(config.domain_ids) == 1:
                filter_dict['domain_id'] = config.domain_ids[0]
            else:
                filter_dict['domain_id'] = {'$in': config.domain_ids}
        
        if config.document_ids:
            if len(config.document_ids) == 1:
                filter_dict['document_id'] = config.document_ids[0]
            else:
                filter_dict['document_id'] = {'$in': config.document_ids}
        
        if config.metadata_filter:
            filter_dict.update(config.metadata_filter)
        
        return filter_dict if filter_dict else None
    
    async def _rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Rerank results using cross-encoder or LLM.
        Default implementation uses embedding similarity.
        """
        if not results:
            return results
        
        # Simple reranking: re-embed and compute similarity
        texts = [r.content for r in results]
        embed_result = await self.embedder.embed(texts)
        doc_embeddings = embed_result.embeddings
        
        query_embedding = await self.embedder.embed_query(query)
        query_vec = np.array(query_embedding)
        
        # Compute similarities
        scores = []
        for doc_emb in doc_embeddings:
            doc_vec = np.array(doc_emb)
            sim = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            )
            scores.append(sim)
        
        # Sort by new scores
        indexed_results = list(zip(results, scores))
        indexed_results.sort(key=lambda x: x[1], reverse=True)
        
        # Update scores
        reranked = []
        for result, score in indexed_results[:top_k]:
            result.score = float(score)
            reranked.append(result)
        
        return reranked
