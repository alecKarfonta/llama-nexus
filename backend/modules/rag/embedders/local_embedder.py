"""
Local Embedding Models

Uses sentence-transformers for local embedding computation.
Supports various models including nomic-embed-text, all-MiniLM, bge-large, etc.
"""

import logging
import time
import asyncio
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from .base import Embedder, EmbeddingResult, EmbeddingModel

logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "all-MiniLM-L6-v2": {
        "dimensions": 384,
        "max_tokens": 256,
        "description": "Fast, lightweight model for semantic similarity"
    },
    "all-mpnet-base-v2": {
        "dimensions": 768,
        "max_tokens": 384,
        "description": "Higher quality, slower than MiniLM"
    },
    "nomic-embed-text-v1": {
        "dimensions": 768,
        "max_tokens": 8192,
        "description": "Long context embedding model by Nomic AI"
    },
    "nomic-embed-text-v1.5": {
        "dimensions": 768,
        "max_tokens": 8192,
        "description": "Updated Nomic embed with matryoshka support"
    },
    "BAAI/bge-large-en-v1.5": {
        "dimensions": 1024,
        "max_tokens": 512,
        "description": "High quality BGE model by BAAI"
    },
    "BAAI/bge-small-en-v1.5": {
        "dimensions": 384,
        "max_tokens": 512,
        "description": "Smaller BGE model, faster inference"
    },
    "thenlper/gte-large": {
        "dimensions": 1024,
        "max_tokens": 512,
        "description": "General Text Embeddings large model"
    },
    "intfloat/e5-large-v2": {
        "dimensions": 1024,
        "max_tokens": 512,
        "description": "E5 embeddings with instruction prefix support"
    }
}


class LocalEmbedder(Embedder):
    """
    Local embedding using sentence-transformers.
    
    Features:
    - Multiple model support
    - GPU acceleration
    - Batch processing
    - Model caching
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,  # "cuda", "cpu", or None for auto
        normalize: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        
        # Get model config
        self._config = MODEL_CONFIGS.get(model_name, {
            "dimensions": 384,
            "max_tokens": 512,
            "description": "Custom model"
        })
    
    def _load_model(self):
        """Load the model (lazy loading)"""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            logger.info(f"Model loaded on device: {self._model.device}")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Run: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def _embed_sync(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool
    ) -> EmbeddingResult:
        """Synchronous embedding (runs in thread pool)"""
        self._load_model()
        
        start_time = time.time()
        
        # Preprocess texts
        processed_texts = [
            self._truncate_text(
                self._preprocess_text(t),
                self._config["max_tokens"]
            )
            for t in texts
        ]
        
        # Embed
        embeddings = self._model.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Estimate token count
        total_chars = sum(len(t) for t in processed_texts)
        token_count = total_chars // 4
        
        return EmbeddingResult(
            embeddings=embeddings.tolist(),
            model=self.model_name,
            dimensions=self._config["dimensions"],
            token_count=token_count,
            processing_time_ms=elapsed_ms,
            batch_size=len(texts)
        )
    
    async def embed(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> EmbeddingResult:
        """Embed texts asynchronously"""
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model=self.model_name,
                dimensions=self._config["dimensions"],
                token_count=0
            )
        
        # Run in thread pool to not block
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            self._embed_sync,
            texts,
            batch_size,
            show_progress
        )
        
        return result
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        result = await self.embed([text])
        return result.embeddings[0] if result.embeddings else []
    
    def get_model_info(self) -> EmbeddingModel:
        """Get model information"""
        return EmbeddingModel(
            name=self.model_name,
            dimensions=self._config["dimensions"],
            max_tokens=self._config["max_tokens"],
            description=self._config["description"],
            provider="local",
            is_available=True
        )
    
    async def is_available(self) -> bool:
        """Check if model is available"""
        try:
            self._load_model()
            return True
        except Exception:
            return False
    
    @staticmethod
    def list_available_models() -> List[EmbeddingModel]:
        """List all available local models"""
        return [
            EmbeddingModel(
                name=name,
                dimensions=config["dimensions"],
                max_tokens=config["max_tokens"],
                description=config["description"],
                provider="local"
            )
            for name, config in MODEL_CONFIGS.items()
        ]
