"""
Hybrid Retriever

Combines dense vector search with sparse keyword search.
Uses Reciprocal Rank Fusion for score combination.
"""

import logging
import re
from typing import List, Optional, Dict, Any, Set
from collections import defaultdict
import math

from .base import Retriever, RetrievalResult, RetrievalConfig
from .vector_retriever import VectorRetriever
from ..vector_stores.base import VectorStore
from ..embedders.base import Embedder
from ..document_manager import DocumentManager

logger = logging.getLogger(__name__)


class BM25:
    """Simple BM25 implementation for keyword scoring"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_lens: Dict[str, int] = {}
        self.avg_doc_len = 0
        self.corpus_size = 0
        self.doc_terms: Dict[str, Dict[str, int]] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def add_document(self, doc_id: str, text: str):
        """Add document to index"""
        tokens = self._tokenize(text)
        self.doc_lens[doc_id] = len(tokens)
        
        # Count term frequencies
        term_counts: Dict[str, int] = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1
        
        self.doc_terms[doc_id] = dict(term_counts)
        
        # Update document frequencies
        for term in term_counts:
            self.doc_freqs[term] += 1
        
        # Update averages
        self.corpus_size = len(self.doc_lens)
        self.avg_doc_len = sum(self.doc_lens.values()) / max(1, self.corpus_size)
    
    def score(self, doc_id: str, query: str) -> float:
        """Calculate BM25 score for query against document"""
        if doc_id not in self.doc_terms:
            return 0.0
        
        query_tokens = self._tokenize(query)
        doc_len = self.doc_lens[doc_id]
        score = 0.0
        
        for term in query_tokens:
            if term not in self.doc_terms[doc_id]:
                continue
            
            tf = self.doc_terms[doc_id][term]
            df = self.doc_freqs.get(term, 0)
            
            # IDF component
            idf = math.log(
                (self.corpus_size - df + 0.5) / (df + 0.5) + 1
            )
            
            # TF component with length normalization
            tf_norm = (
                tf * (self.k1 + 1) /
                (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len))
            )
            
            score += idf * tf_norm
        
        return score
    
    def search(self, query: str, doc_ids: List[str], top_k: int = 10) -> List[tuple]:
        """Search documents and return (doc_id, score) pairs"""
        scores = []
        for doc_id in doc_ids:
            score = self.score(doc_id, query)
            if score > 0:
                scores.append((doc_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridRetriever(Retriever):
    """
    Hybrid retrieval combining dense and sparse search.
    
    Features:
    - Dense vector similarity (semantic)
    - BM25 keyword scoring (lexical)
    - Reciprocal Rank Fusion for combination
    - Configurable weighting
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
        
        # Vector retriever for dense search
        self.vector_retriever = VectorRetriever(
            vector_store, embedder, document_manager, collection_name
        )
        
        # BM25 for sparse search
        self.bm25 = BM25()
        self._bm25_indexed = False
    
    async def _ensure_bm25_index(self):
        """Build BM25 index from documents"""
        if self._bm25_indexed:
            return
        
        # Get all chunks
        offset = None
        while True:
            records, next_offset = await self.vector_store.scroll(
                self.collection_name,
                limit=100,
                offset=offset
            )
            
            for record in records:
                content = record.payload.get('content', '')
                if content:
                    self.bm25.add_document(record.id, content)
            
            if not next_offset:
                break
            offset = next_offset
        
        self._bm25_indexed = True
        logger.info(f"BM25 index built with {self.bm25.corpus_size} documents")
    
    async def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None
    ) -> List[RetrievalResult]:
        """Hybrid retrieval with RRF combination"""
        config = config or RetrievalConfig()
        
        # Get more results for fusion
        fusion_k = config.top_k * 2
        
        # Dense retrieval
        dense_config = RetrievalConfig(
            top_k=fusion_k,
            domain_ids=config.domain_ids,
            document_ids=config.document_ids,
            metadata_filter=config.metadata_filter
        )
        dense_results = await self.vector_retriever.retrieve(query, dense_config)
        
        # Sparse retrieval (BM25)
        await self._ensure_bm25_index()
        
        # Get candidate doc IDs
        candidate_ids = [r.chunk_id for r in dense_results]
        if not candidate_ids:
            # If no dense results, get all doc IDs
            records, _ = await self.vector_store.scroll(
                self.collection_name, limit=1000
            )
            candidate_ids = [r.id for r in records]
        
        sparse_results = self.bm25.search(query, candidate_ids, fusion_k)
        
        # Reciprocal Rank Fusion
        rrf_scores = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            config.alpha  # Weight for dense (1 = all dense, 0 = all sparse)
        )
        
        # Sort by RRF score and take top_k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Build final results
        result_map = {r.chunk_id: r for r in dense_results}
        final_results = []
        
        for chunk_id in sorted_ids[:config.top_k]:
            if chunk_id in result_map:
                result = result_map[chunk_id]
                result.score = rrf_scores[chunk_id]
                result.metadata['retrieval_method'] = 'hybrid'
                final_results.append(result)
        
        return final_results
    
    async def retrieve_batch(
        self,
        queries: List[str],
        config: Optional[RetrievalConfig] = None
    ) -> List[List[RetrievalResult]]:
        """Batch hybrid retrieval"""
        results = []
        for query in queries:
            result = await self.retrieve(query, config)
            results.append(result)
        return results
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[tuple],  # (doc_id, score)
        alpha: float = 0.5,
        k: int = 60  # RRF constant
    ) -> Dict[str, float]:
        """
        Combine rankings using Reciprocal Rank Fusion.
        
        RRF score = alpha * 1/(k + rank_dense) + (1-alpha) * 1/(k + rank_sparse)
        """
        scores = defaultdict(float)
        
        # Dense scores
        for rank, result in enumerate(dense_results, 1):
            scores[result.chunk_id] += alpha * (1.0 / (k + rank))
        
        # Sparse scores
        for rank, (doc_id, _) in enumerate(sparse_results, 1):
            scores[doc_id] += (1 - alpha) * (1.0 / (k + rank))
        
        return dict(scores)
    
    def reset_bm25_index(self):
        """Reset BM25 index (call after adding new documents)"""
        self.bm25 = BM25()
        self._bm25_indexed = False
