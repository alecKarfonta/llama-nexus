"""
Base Retriever Interface

Provides abstract interface for document retrieval strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies"""
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    GRAPH = "graph"
    MMR = "mmr"  # Maximum Marginal Relevance
    HYDE = "hyde"  # Hypothetical Document Embeddings


@dataclass
class RetrievalConfig:
    """Configuration for retrieval"""
    strategy: RetrievalStrategy = RetrievalStrategy.VECTOR
    top_k: int = 10
    score_threshold: Optional[float] = None
    
    # MMR settings
    mmr_lambda: float = 0.5  # Balance between relevance and diversity
    
    # Hybrid settings
    alpha: float = 0.5  # Weight between dense (1) and sparse (0)
    
    # Reranking
    rerank: bool = False
    rerank_top_k: int = 20  # Retrieve more, then rerank to top_k
    
    # Filters
    domain_ids: Optional[List[str]] = None
    document_ids: Optional[List[str]] = None
    metadata_filter: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalResult:
    """A retrieved document chunk"""
    chunk_id: str
    document_id: str
    content: str
    score: float
    # Metadata
    domain_id: Optional[str] = None
    document_name: Optional[str] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    section_header: Optional[str] = None
    # Additional info
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'content': self.content,
            'score': self.score,
            'domain_id': self.domain_id,
            'document_name': self.document_name,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,
            'section_header': self.section_header,
            'metadata': self.metadata
        }


class Retriever(ABC):
    """
    Abstract base class for document retrieval.
    
    Implementations:
    - VectorRetriever: Dense vector similarity search
    - KeywordRetriever: BM25/TF-IDF keyword search
    - HybridRetriever: Combined dense + sparse
    - GraphRetriever: Knowledge graph-enhanced
    """
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            config: Retrieval configuration
            
        Returns:
            List of retrieval results sorted by relevance
        """
        pass
    
    @abstractmethod
    async def retrieve_batch(
        self,
        queries: List[str],
        config: Optional[RetrievalConfig] = None
    ) -> List[List[RetrievalResult]]:
        """Retrieve for multiple queries"""
        pass
    
    async def retrieve_with_context(
        self,
        query: str,
        context_window: int = 1,
        config: Optional[RetrievalConfig] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve documents with surrounding context.
        Returns retrieved chunks plus neighboring chunks.
        """
        # Default implementation - override for better performance
        results = await self.retrieve(query, config)
        return results
    
    def _deduplicate_results(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Remove duplicate chunks, keeping highest score"""
        seen = {}
        for result in results:
            if result.chunk_id not in seen:
                seen[result.chunk_id] = result
            elif result.score > seen[result.chunk_id].score:
                seen[result.chunk_id] = result
        return list(seen.values())
    
    def _filter_by_threshold(
        self,
        results: List[RetrievalResult],
        threshold: float
    ) -> List[RetrievalResult]:
        """Filter results by score threshold"""
        return [r for r in results if r.score >= threshold]
