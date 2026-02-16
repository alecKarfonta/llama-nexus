"""
Two-Stage Filtering Mechanism for GraphRAG-FI (Filtering and Integration)

This module implements advanced filtering techniques to refine retrieved information
and reduce over-reliance on external knowledge while maintaining accuracy.

Based on research: "Empowering GraphRAG with Knowledge Filtering and Integration"
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Handle optional dependencies gracefully
try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None
    
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    
try:
    from transformers import AutoTokenizer, AutoModel  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModel = None

logger = logging.getLogger(__name__)


class FilterStage(Enum):
    """Enumeration of filtering stages"""
    RELEVANCE = "relevance"
    QUALITY = "quality"
    CONFIDENCE = "confidence"


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with metadata"""
    content: str
    score: float
    source: str
    entity_matches: List[str]
    graph_distance: Optional[int] = None
    vector_similarity: Optional[float] = None
    chunk_id: Optional[str] = None


@dataclass
class FilteringResult:
    """Result of filtering process"""
    filtered_chunks: List[RetrievedChunk]
    relevance_scores: List[float]
    quality_scores: List[float]
    confidence_scores: List[float]
    filtering_metadata: Dict[str, Any]


class TwoStageFilter:
    """
    Two-stage filtering mechanism for GraphRAG retrieval refinement.
    
    Stage 1: Relevance Filtering - Remove irrelevant chunks
    Stage 2: Quality Assessment - Evaluate coherence and informativeness
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        relevance_threshold: float = 0.1,
        quality_threshold: float = 0.2,
        confidence_threshold: float = 0.3,
        max_chunks: int = 20,
        use_gpu: bool = True
    ):
        """
        Initialize the two-stage filter.
        
        Args:
            embedding_model: Model for semantic similarity computation
            relevance_threshold: Minimum relevance score to pass stage 1
            quality_threshold: Minimum quality score to pass stage 2
            confidence_threshold: Minimum confidence for final selection
            max_chunks: Maximum number of chunks to return
            use_gpu: Whether to use GPU acceleration
        """
        # Check dependencies
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for TwoStageFilter. Install with: pip install sentence-transformers")
        
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for TwoStageFilter. Install with: pip install torch")
        
        self.relevance_threshold = relevance_threshold
        self.quality_threshold = quality_threshold
        self.confidence_threshold = confidence_threshold
        self.max_chunks = max_chunks
        
        # Initialize models
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"  # type: ignore
        self.device = device
        
        # Embedding model for semantic similarity
        self.embedding_model = SentenceTransformer(embedding_model, device=device)  # type: ignore
        
        # Initialize quality assessment components
        self._init_quality_models()
        
        logger.info(f"TwoStageFilter initialized with device: {device}")
    
    def _init_quality_models(self):
        """Initialize models for quality assessment"""
        # Simple quality indicators - can be replaced with more sophisticated models
        self.quality_keywords = {
            'high_quality': ['specifically', 'detailed', 'comprehensive', 'analysis', 'evidence'],
            'low_quality': ['maybe', 'probably', 'unclear', 'vague', 'uncertain']
        }
    
    def filter_chunks(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        entity_context: Optional[List[str]] = None
    ) -> FilteringResult:
        """
        Apply two-stage filtering to retrieved chunks.
        
        Args:
            query: Original user query
            retrieved_chunks: List of chunks from retrieval system
            entity_context: Optional list of relevant entities for context
            
        Returns:
            FilteringResult with filtered chunks and metadata
        """
        logger.info(f"Starting two-stage filtering for {len(retrieved_chunks)} chunks")
        
        # Stage 1: Relevance Filtering
        stage1_chunks, relevance_scores = self._stage1_relevance_filtering(
            query, retrieved_chunks, entity_context
        )
        
        logger.info(f"Stage 1 filtering: {len(stage1_chunks)}/{len(retrieved_chunks)} chunks passed")
        
        # Stage 2: Quality Assessment
        stage2_chunks, quality_scores = self._stage2_quality_assessment(
            query, stage1_chunks
        )
        
        logger.info(f"Stage 2 filtering: {len(stage2_chunks)}/{len(stage1_chunks)} chunks passed")
        
        # Confidence scoring and final selection
        final_chunks, confidence_scores = self._confidence_scoring_and_selection(
            query, stage2_chunks, relevance_scores, quality_scores
        )
        
        logger.info(f"Final selection: {len(final_chunks)} chunks selected")
        
        # Fallback logic: if filtering removed too many chunks, use less aggressive filtering
        min_chunks_threshold = max(1, min(3, len(retrieved_chunks) // 3))  # Keep at least 1-3 chunks
        
        if len(final_chunks) < min_chunks_threshold:
            logger.warning(f"Filtering too aggressive ({len(final_chunks)} < {min_chunks_threshold}), applying fallback")
            
            # Fallback 1: Use stage 1 results if available
            if len(stage1_chunks) >= min_chunks_threshold:
                final_chunks = stage1_chunks[:self.max_chunks]
                confidence_scores = relevance_scores[:len(final_chunks)]
                logger.info(f"Using stage 1 fallback: {len(final_chunks)} chunks")
            
            # Fallback 2: Use original results with basic filtering
            elif len(retrieved_chunks) > 0:
                # Keep top chunks by original score
                sorted_chunks = sorted(retrieved_chunks, key=lambda x: x.score, reverse=True)
                final_chunks = sorted_chunks[:min(self.max_chunks, len(sorted_chunks))]
                confidence_scores = [chunk.score for chunk in final_chunks]
                logger.info(f"Using original results fallback: {len(final_chunks)} chunks")
        
        # Prepare metadata
        filtering_metadata = {
            'original_count': len(retrieved_chunks),
            'stage1_count': len(stage1_chunks),
            'stage2_count': len(stage2_chunks),
            'final_count': len(final_chunks),
            'avg_relevance': np.mean(relevance_scores) if relevance_scores else 0,
            'avg_quality': np.mean(quality_scores) if quality_scores else 0,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'fallback_used': len(final_chunks) < len(stage2_chunks) if stage2_chunks else False
        }
        
        return FilteringResult(
            filtered_chunks=final_chunks,
            relevance_scores=relevance_scores,
            quality_scores=quality_scores,
            confidence_scores=confidence_scores,
            filtering_metadata=filtering_metadata
        )
    
    def _stage1_relevance_filtering(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        entity_context: Optional[List[str]] = None
    ) -> Tuple[List[RetrievedChunk], List[float]]:
        """
        Stage 1: Filter chunks based on relevance to query and entities.
        
        Args:
            query: User query
            chunks: Retrieved chunks
            entity_context: Relevant entities for context
            
        Returns:
            Tuple of (filtered_chunks, relevance_scores)
        """
        if not chunks:
            return [], []
        
        # Compute semantic similarity between query and chunks
        query_embedding = self.embedding_model.encode([query])
        chunk_texts = [chunk.content for chunk in chunks]
        chunk_embeddings = self.embedding_model.encode(chunk_texts)
        
        # Calculate semantic similarities
        if TORCH_AVAILABLE and torch is not None and F is not None:
            semantic_similarities = F.cosine_similarity(
                torch.tensor(query_embedding),
                torch.tensor(chunk_embeddings),
                dim=1
            ).numpy()
        else:
            # Fallback to numpy cosine similarity
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                semantic_similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            except ImportError:
                # Final fallback to manual cosine similarity
                semantic_similarities = []
                for chunk_emb in chunk_embeddings:
                    dot_product = np.dot(query_embedding[0], chunk_emb)
                    norm_query = np.linalg.norm(query_embedding[0])
                    norm_chunk = np.linalg.norm(chunk_emb)
                    similarity = dot_product / (norm_query * norm_chunk)
                    semantic_similarities.append(similarity)
                semantic_similarities = np.array(semantic_similarities)
        
        # Entity-based relevance scoring
        entity_scores = self._calculate_entity_relevance(chunks, entity_context)
        
        # Combine semantic and entity-based scores
        relevance_scores = []
        for i, chunk in enumerate(chunks):
            semantic_score = semantic_similarities[i]
            entity_score = entity_scores[i] if i < len(entity_scores) else 0
            
            # Weighted combination (can be tuned)
            relevance_score = 0.7 * semantic_score + 0.3 * entity_score
            relevance_scores.append(float(relevance_score))
        
        # Filter chunks that meet relevance threshold
        filtered_chunks = []
        filtered_scores = []
        
        for chunk, score in zip(chunks, relevance_scores):
            if score >= self.relevance_threshold:
                filtered_chunks.append(chunk)
                filtered_scores.append(score)
        
        return filtered_chunks, filtered_scores
    
    def _calculate_entity_relevance(
        self,
        chunks: List[RetrievedChunk],
        entity_context: Optional[List[str]]
    ) -> List[float]:
        """Calculate entity-based relevance scores"""
        if not entity_context:
            return [0.0] * len(chunks)
        
        entity_scores = []
        for chunk in chunks:
            # Count entity matches in chunk
            entity_matches = 0
            chunk_lower = chunk.content.lower()
            
            for entity in entity_context:
                if entity.lower() in chunk_lower:
                    entity_matches += 1
            
            # Normalize by number of entities
            entity_score = entity_matches / len(entity_context) if entity_context else 0
            entity_scores.append(entity_score)
        
        return entity_scores
    
    def _stage2_quality_assessment(
        self,
        query: str,
        chunks: List[RetrievedChunk]
    ) -> Tuple[List[RetrievedChunk], List[float]]:
        """
        Stage 2: Assess quality and informativeness of chunks.
        
        Args:
            query: User query
            chunks: Chunks that passed stage 1
            
        Returns:
            Tuple of (high_quality_chunks, quality_scores)
        """
        if not chunks:
            return [], []
        
        quality_scores = []
        
        for chunk in chunks:
            quality_score = self._assess_chunk_quality(chunk, query)
            quality_scores.append(quality_score)
        
        # Filter chunks that meet quality threshold
        filtered_chunks = []
        filtered_scores = []
        
        for chunk, score in zip(chunks, quality_scores):
            if score >= self.quality_threshold:
                filtered_chunks.append(chunk)
                filtered_scores.append(score)
        
        return filtered_chunks, filtered_scores
    
    def _assess_chunk_quality(self, chunk: RetrievedChunk, query: str) -> float:
        """
        Assess the quality of a chunk based on simplified factors.
        
        Args:
            chunk: Chunk to assess
            query: Original query for context
            
        Returns:
            Quality score between 0 and 1
        """
        content = chunk.content
        
        # Start with a baseline quality score
        quality_score = 0.6  # More lenient baseline
        
        # 1. Basic length check (very lenient)
        word_count = len(content.split())
        if word_count < 5:  # Too short
            quality_score -= 0.3
        elif word_count > 1000:  # Too long
            quality_score -= 0.1
        
        # 2. Basic completeness check
        if not content.strip():
            quality_score = 0.0
        elif len(content.strip()) < 10:
            quality_score -= 0.2
        
        # 3. Check for obviously broken text
        if content.count('�') > 0:  # Encoding issues
            quality_score -= 0.3
        
        # 4. Boost for entity matches
        if chunk.entity_matches and len(chunk.entity_matches) > 0:
            quality_score += 0.2
        
        # 5. Boost for higher original retrieval scores
        if chunk.score > 0.7:
            quality_score += 0.1
        
        return min(1.0, max(0.0, quality_score))
    
    def _assess_length_quality(self, content: str) -> float:
        """Assess if content length is appropriate"""
        word_count = len(content.split())
        
        if word_count < 10:  # Too short
            return 0.3
        elif word_count > 500:  # Too long
            return 0.6
        elif 20 <= word_count <= 200:  # Optimal range
            return 1.0
        else:
            return 0.8
    
    def _assess_information_density(self, content: str) -> float:
        """Assess information density of content"""
        # Count informative words (nouns, adjectives, verbs)
        words = content.lower().split()
        
        # Simple heuristic: avoid too many common words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        informative_ratio = 1 - (sum(1 for word in words if word in common_words) / len(words))
        
        return min(1.0, informative_ratio * 1.5)  # Boost slightly
    
    def _assess_coherence(self, content: str) -> float:
        """Assess basic coherence of content"""
        sentences = content.split('.')
        
        if len(sentences) < 2:
            return 0.5
        
        # Check for proper sentence structure
        coherence_score = 1.0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Basic checks
                if not sentence[0].isupper():  # Should start with capital
                    coherence_score -= 0.1
                if len(sentence.split()) < 3:  # Very short sentences
                    coherence_score -= 0.1
        
        return max(0.0, coherence_score)
    
    def _assess_keyword_quality(self, content: str) -> float:
        """Assess quality based on keywords"""
        content_lower = content.lower()
        
        high_quality_count = sum(1 for kw in self.quality_keywords['high_quality'] if kw in content_lower)
        low_quality_count = sum(1 for kw in self.quality_keywords['low_quality'] if kw in content_lower)
        
        # Score based on quality indicators
        if high_quality_count > low_quality_count:
            return 0.8 + 0.2 * (high_quality_count / (high_quality_count + low_quality_count))
        elif low_quality_count > high_quality_count:
            return 0.4 - 0.2 * (low_quality_count / (high_quality_count + low_quality_count))
        else:
            return 0.6  # Neutral
    
    def _assess_completeness(self, content: str) -> float:
        """Assess if content appears complete"""
        # Check for proper sentence endings
        sentences = content.split('.')
        complete_sentences = sum(1 for s in sentences if s.strip() and len(s.strip()) > 5)
        
        if len(sentences) <= 1:
            return 0.5
        
        completeness = complete_sentences / len(sentences)
        return completeness
    
    def _confidence_scoring_and_selection(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        relevance_scores: List[float],
        quality_scores: List[float]
    ) -> Tuple[List[RetrievedChunk], List[float]]:
        """
        Final confidence scoring and chunk selection.
        
        Args:
            query: Original query
            chunks: High-quality chunks from stage 2
            relevance_scores: Relevance scores from stage 1
            quality_scores: Quality scores from stage 2
            
        Returns:
            Tuple of (final_chunks, confidence_scores)
        """
        if not chunks:
            return [], []
        
        confidence_scores = []
        
        # Get relevance and quality scores for remaining chunks
        for i, chunk in enumerate(chunks):
            relevance = relevance_scores[i] if i < len(relevance_scores) else 0.5
            quality = quality_scores[i] if i < len(quality_scores) else 0.5
            
            # Additional confidence factors
            diversity_bonus = self._calculate_diversity_bonus(chunk, chunks[:i])
            source_confidence = self._calculate_source_confidence(chunk)
            
            # Combine confidence factors
            confidence = (
                0.4 * relevance +
                0.4 * quality +
                0.1 * diversity_bonus +
                0.1 * source_confidence
            )
            
            confidence_scores.append(confidence)
        
        # Select chunks that meet confidence threshold
        chunk_confidence_pairs = list(zip(chunks, confidence_scores))
        
        # Filter by confidence threshold
        filtered_pairs = [
            (chunk, conf) for chunk, conf in chunk_confidence_pairs
            if conf >= self.confidence_threshold
        ]
        
        # Sort by confidence (descending) and limit to max_chunks
        filtered_pairs.sort(key=lambda x: x[1], reverse=True)
        filtered_pairs = filtered_pairs[:self.max_chunks]
        
        if filtered_pairs:
            final_chunks, final_confidences = zip(*filtered_pairs)
            return list(final_chunks), list(final_confidences)
        else:
            return [], []
    
    def _calculate_diversity_bonus(self, chunk: RetrievedChunk, previous_chunks: List[RetrievedChunk]) -> float:
        """Calculate diversity bonus to avoid redundant chunks"""
        if not previous_chunks:
            return 1.0
        
        # Check similarity with previous chunks
        chunk_embedding = self.embedding_model.encode([chunk.content])
        previous_texts = [c.content for c in previous_chunks]
        previous_embeddings = self.embedding_model.encode(previous_texts)
        
        if len(previous_embeddings) > 0:
            if TORCH_AVAILABLE and torch is not None and F is not None:
                similarities = F.cosine_similarity(
                    torch.tensor(chunk_embedding),
                    torch.tensor(previous_embeddings),
                    dim=1
                ).numpy()
            else:
                # Fallback to manual cosine similarity
                similarities = []
                for prev_emb in previous_embeddings:
                    dot_product = np.dot(chunk_embedding[0], prev_emb)
                    norm_chunk = np.linalg.norm(chunk_embedding[0])
                    norm_prev = np.linalg.norm(prev_emb)
                    similarity = dot_product / (norm_chunk * norm_prev)
                    similarities.append(similarity)
                similarities = np.array(similarities)
            
            max_similarity = np.max(similarities)
            diversity_bonus = 1.0 - max_similarity  # Higher bonus for more diverse chunks
        else:
            diversity_bonus = 1.0
        
        return max(0.0, diversity_bonus)
    
    def _calculate_source_confidence(self, chunk: RetrievedChunk) -> float:
        """Calculate confidence based on source characteristics"""
        # Simple heuristics - can be enhanced with source reputation, etc.
        confidence = 0.5  # Default confidence
        
        # Boost confidence for graph-retrieved chunks
        if chunk.graph_distance is not None and chunk.graph_distance <= 2:
            confidence += 0.3
        
        # Boost confidence for high vector similarity
        if chunk.vector_similarity is not None and chunk.vector_similarity > 0.8:
            confidence += 0.2
        
        # Boost confidence for entity matches
        if chunk.entity_matches and len(chunk.entity_matches) > 0:
            confidence += 0.1 * min(len(chunk.entity_matches), 3)
        
        return min(1.0, confidence)


class FilteringIntegrator:
    """
    Integrates filtering results with LLM generation using logits-based selection.
    
    This component implements the integration part of GraphRAG-FI to balance
    external knowledge with LLM intrinsic reasoning.
    """
    
    def __init__(
        self,
        integration_threshold: float = 0.7,
        balance_factor: float = 0.6
    ):
        """
        Initialize the filtering integrator.
        
        Args:
            integration_threshold: Threshold for integrating external knowledge
            balance_factor: Factor for balancing external vs intrinsic knowledge
        """
        self.integration_threshold = integration_threshold
        self.balance_factor = balance_factor
    
    def integrate_filtered_knowledge(
        self,
        filtering_result: FilteringResult,
        llm_intrinsic_confidence: float,
        query_complexity: float
    ) -> Dict[str, Any]:
        """
        Integrate filtered knowledge with LLM intrinsic reasoning.
        
        Args:
            filtering_result: Result from two-stage filtering
            llm_intrinsic_confidence: LLM's confidence in its intrinsic knowledge
            query_complexity: Complexity score of the query (0-1)
            
        Returns:
            Integration decision and metadata
        """
        # Calculate integration weights
        external_confidence = np.mean(filtering_result.confidence_scores) if filtering_result.confidence_scores else 0
        
        # Adaptive balancing based on confidence levels and query complexity
        if external_confidence > self.integration_threshold and query_complexity > 0.5:
            # Use external knowledge heavily for complex queries with high-confidence retrieval
            external_weight = self.balance_factor + 0.2
        elif llm_intrinsic_confidence > 0.8 and external_confidence < 0.5:
            # Rely more on intrinsic knowledge when external is low quality
            external_weight = self.balance_factor - 0.3
        else:
            # Default balanced approach
            external_weight = self.balance_factor
        
        external_weight = max(0.1, min(0.9, external_weight))  # Clamp between 0.1 and 0.9
        intrinsic_weight = 1.0 - external_weight
        
        integration_metadata = {
            'external_weight': external_weight,
            'intrinsic_weight': intrinsic_weight,
            'external_confidence': external_confidence,
            'llm_intrinsic_confidence': llm_intrinsic_confidence,
            'query_complexity': query_complexity,
            'integration_strategy': self._determine_strategy(external_weight),
            'chunks_used': len(filtering_result.filtered_chunks)
        }
        
        return {
            'filtered_chunks': filtering_result.filtered_chunks,
            'integration_weights': {
                'external': external_weight,
                'intrinsic': intrinsic_weight
            },
            'metadata': integration_metadata
        }
    
    def _determine_strategy(self, external_weight: float) -> str:
        """Determine integration strategy based on weights"""
        if external_weight > 0.7:
            return "external_heavy"
        elif external_weight < 0.4:
            return "intrinsic_heavy"
        else:
            return "balanced"


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    filter_system = TwoStageFilter(
        relevance_threshold=0.3,
        quality_threshold=0.5,
        confidence_threshold=0.6
    )
    
    # Mock retrieved chunks
    sample_chunks = [
        RetrievedChunk(
            content="The character development in this novel showcases the author's masterful storytelling.",
            score=0.8,
            source="chapter_analysis.txt",
            entity_matches=["character", "novel", "author"]
        ),
        RetrievedChunk(
            content="Maybe the book was written in 1850 or something like that.",
            score=0.4,
            source="uncertain_facts.txt",
            entity_matches=["book"]
        )
    ]
    
    query = "How does character development contribute to the narrative?"
    entities = ["character", "development", "narrative"]
    
    result = filter_system.filter_chunks(query, sample_chunks, entities)
    
    print(f"Filtered {len(result.filtered_chunks)} chunks from {len(sample_chunks)} original chunks")
    print(f"Average confidence: {np.mean(result.confidence_scores):.3f}")
    print(f"Filtering metadata: {result.filtering_metadata}") 