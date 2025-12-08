"""
Base Embedder Interface

Provides abstract interface for text embedding models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModel:
    """Information about an embedding model"""
    name: str
    dimensions: int
    max_tokens: int = 512
    description: str = ""
    provider: str = "local"  # local, openai, cohere, etc.
    is_available: bool = True
    
    # Performance metrics
    tokens_per_second: Optional[float] = None
    memory_usage_mb: Optional[float] = None


@dataclass
class EmbeddingResult:
    """Result of embedding operation"""
    embeddings: List[List[float]]
    model: str
    dimensions: int
    token_count: int
    # Processing info
    processing_time_ms: float = 0
    batch_size: int = 1


class Embedder(ABC):
    """
    Abstract base class for text embedding models.
    
    Implementations:
    - LocalEmbedder: Sentence transformers, local models
    - APIEmbedder: OpenAI, Cohere, Voyage AI
    """
    
    @abstractmethod
    async def embed(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> EmbeddingResult:
        """
        Embed a list of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        pass
    
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        May use different handling than document embedding.
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> EmbeddingModel:
        """Get information about the current model"""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the embedder is available"""
        pass
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding"""
        # Basic cleaning
        text = text.strip()
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens (rough estimate)"""
        # Rough estimate: 4 chars per token
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            return text[:max_chars]
        return text
