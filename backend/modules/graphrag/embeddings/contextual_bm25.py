"""
Contextual BM25 implementation for enhanced lexical + semantic matching.

This module provides BM25 with contextual enhancements, allowing for better
lexical search that considers document context and semantic relationships.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class DocumentContext:
    """Context information for a document chunk."""
    source_file: str
    section: Optional[str] = None
    preceding_context: Optional[str] = None
    following_context: Optional[str] = None
    entities: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BM25Result:
    """Result from BM25 search with context."""
    content: str
    score: float
    source: str
    context: DocumentContext
    highlighted_terms: List[str]
    metadata: Dict[str, Any]


class ContextualBM25:
    """
    Enhanced BM25 implementation with contextual information.
    
    This class provides BM25 search enhanced with:
    - Document context (preceding/following text)
    - Entity-aware search
    - Semantic term expansion
    - Contextual scoring adjustments
    """
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize the contextual BM25 system.
        
        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (document length normalization)
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.document_contexts = []
        self.tokenized_docs = []
        self.term_frequencies = defaultdict(int)
        self.entity_mapping = defaultdict(list)  # entity -> document indices
        
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 tokenization.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            List of preprocessed tokens
        """
        # Convert to lowercase and remove extra whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Tokenize (simple word boundaries)
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove very short tokens and stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        tokens = [token for token in tokens if len(token) > 2 and token not in stop_words]
        
        return tokens
    
    def _create_contextual_document(self, content: str, context: DocumentContext) -> str:
        """
        Create an enhanced document by adding contextual information.
        
        Args:
            content: Original document content
            context: Document context information
            
        Returns:
            Enhanced document text with context
        """
        enhanced_content = content
        
        # Add preceding context with lower weight
        if context.preceding_context:
            enhanced_content = f"{context.preceding_context} {enhanced_content}"
        
        # Add following context with lower weight
        if context.following_context:
            enhanced_content = f"{enhanced_content} {context.following_context}"
        
        # Add entities for better matching
        if context.entities:
            entity_text = " ".join(context.entities)
            enhanced_content = f"{enhanced_content} {entity_text}"
        
        # Add section information
        if context.section:
            enhanced_content = f"{context.section} {enhanced_content}"
        
        return enhanced_content
    
    def add_documents(self, documents: List[str], contexts: List[DocumentContext]):
        """
        Add documents to the BM25 index with their contexts.
        
        Args:
            documents: List of document texts
            contexts: List of document contexts
        """
        if len(documents) != len(contexts):
            raise ValueError("Number of documents must match number of contexts")
        
        self.documents.extend(documents)
        self.document_contexts.extend(contexts)
        
        # Create contextual documents and tokenize
        for doc, context in zip(documents, contexts):
            contextual_doc = self._create_contextual_document(doc, context)
            tokens = self._preprocess_text(contextual_doc)
            self.tokenized_docs.append(tokens)
            
            # Update term frequencies
            for token in tokens:
                self.term_frequencies[token] += 1
            
            # Update entity mapping
            if context.entities:
                for entity in context.entities:
                    self.entity_mapping[entity.lower()].append(len(self.documents) - 1)
        
        # Rebuild BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)
        
        logger.info(f"Added {len(documents)} documents to BM25 index. Total: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = 10, context_boost: float = 1.2) -> List[BM25Result]:
        """
        Search documents using contextual BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            context_boost: Boost factor for contextual matches
            
        Returns:
            List of BM25 search results
        """
        if not self.bm25:
            logger.warning("BM25 index not initialized. No documents to search.")
            return []
        
        # Preprocess query
        query_tokens = self._preprocess_text(query)
        
        if not query_tokens:
            logger.warning("Query produced no valid tokens after preprocessing.")
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Apply contextual boosts
        boosted_scores = self._apply_contextual_boosts(query_tokens, scores, context_boost)
        
        # Get top results
        top_indices = np.argsort(boosted_scores)[-top_k:][::-1]
        
        # Create results
        results = []
        for idx in top_indices:
            # Include all results, BM25 can return negative scores for small datasets
            highlighted_terms = self._get_highlighted_terms(query_tokens, self.tokenized_docs[idx])
            
            result = BM25Result(
                content=self.documents[idx],
                score=float(boosted_scores[idx]),
                source=self.document_contexts[idx].source_file,
                context=self.document_contexts[idx],
                highlighted_terms=highlighted_terms,
                metadata=self.document_contexts[idx].metadata
            )
            results.append(result)
        
        return results
    
    def _apply_contextual_boosts(self, query_tokens: List[str], scores: np.ndarray, context_boost: float) -> np.ndarray:
        """
        Apply contextual boosts to BM25 scores.
        
        Args:
            query_tokens: Preprocessed query tokens
            scores: Original BM25 scores
            context_boost: Boost factor for contextual matches
            
        Returns:
            Boosted scores
        """
        boosted_scores = scores.copy()
        
        for i, context in enumerate(self.document_contexts):
            boost_factor = 1.0
            
            # Boost for entity matches
            for token in query_tokens:
                if token in [entity.lower() for entity in context.entities]:
                    boost_factor *= context_boost
            
            # Boost for section matches
            if context.section:
                section_tokens = self._preprocess_text(context.section)
                if any(token in section_tokens for token in query_tokens):
                    boost_factor *= (context_boost * 0.8)  # Slightly lower boost for section matches
            
            # Apply boost
            boosted_scores[i] *= boost_factor
        
        return boosted_scores
    
    def _get_highlighted_terms(self, query_tokens: List[str], doc_tokens: List[str]) -> List[str]:
        """
        Get terms that should be highlighted in the result.
        
        Args:
            query_tokens: Query tokens
            doc_tokens: Document tokens
            
        Returns:
            List of terms to highlight
        """
        highlighted = []
        doc_tokens_set = set(doc_tokens)
        
        for token in query_tokens:
            if token in doc_tokens_set:
                highlighted.append(token)
        
        return highlighted
    
    def get_term_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed terms.
        
        Returns:
            Dictionary with term statistics
        """
        if not self.bm25:
            return {}
        
        total_terms = sum(self.term_frequencies.values())
        unique_terms = len(self.term_frequencies)
        
        # Get top terms
        top_terms = sorted(self.term_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            'total_documents': len(self.documents),
            'total_terms': total_terms,
            'unique_terms': unique_terms,
            'avg_doc_length': total_terms / len(self.documents) if self.documents else 0,
            'top_terms': top_terms,
            'entities_mapped': len(self.entity_mapping)
        }
    
    def clear(self):
        """Clear all indexed documents and reset the BM25 system."""
        self.documents = []
        self.document_contexts = []
        self.tokenized_docs = []
        self.term_frequencies = defaultdict(int)
        self.entity_mapping = defaultdict(list)
        self.bm25 = None
        
        logger.info("BM25 index cleared") 