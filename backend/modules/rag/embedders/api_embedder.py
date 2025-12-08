"""
API-Based Embedding Models

Supports OpenAI, Cohere, Voyage AI, and custom endpoints.
"""

import logging
import time
import asyncio
from typing import List, Optional, Dict, Any
import aiohttp

from .base import Embedder, EmbeddingResult, EmbeddingModel

logger = logging.getLogger(__name__)


# API model configurations
API_MODEL_CONFIGS = {
    # Local llama.cpp embedding service
    "nomic-embed-text-v1.5": {
        "provider": "llamacpp",
        "dimensions": 768,
        "max_tokens": 8192,
        "description": "Nomic AI long context embedding (local)"
    },
    "e5-mistral-7b": {
        "provider": "llamacpp",
        "dimensions": 4096,
        "max_tokens": 32768,
        "description": "E5 Mistral 7B embedding (local)"
    },
    "bge-m3": {
        "provider": "llamacpp",
        "dimensions": 1024,
        "max_tokens": 8192,
        "description": "BAAI BGE-M3 multilingual (local)"
    },
    "gte-Qwen2-1.5B": {
        "provider": "llamacpp",
        "dimensions": 1536,
        "max_tokens": 32768,
        "description": "Alibaba GTE Qwen2 1.5B (local)"
    },
    # OpenAI
    "text-embedding-3-small": {
        "provider": "openai",
        "dimensions": 1536,
        "max_tokens": 8191,
        "description": "OpenAI small embedding model, cost-effective"
    },
    "text-embedding-3-large": {
        "provider": "openai",
        "dimensions": 3072,
        "max_tokens": 8191,
        "description": "OpenAI large embedding model, highest quality"
    },
    "text-embedding-ada-002": {
        "provider": "openai",
        "dimensions": 1536,
        "max_tokens": 8191,
        "description": "OpenAI Ada v2 (legacy)"
    },
    # Cohere
    "embed-english-v3.0": {
        "provider": "cohere",
        "dimensions": 1024,
        "max_tokens": 512,
        "description": "Cohere English embedding model v3"
    },
    "embed-multilingual-v3.0": {
        "provider": "cohere",
        "dimensions": 1024,
        "max_tokens": 512,
        "description": "Cohere multilingual embedding model"
    },
    # Voyage AI
    "voyage-large-2": {
        "provider": "voyage",
        "dimensions": 1536,
        "max_tokens": 16000,
        "description": "Voyage AI large model with long context"
    },
    "voyage-code-2": {
        "provider": "voyage",
        "dimensions": 1536,
        "max_tokens": 16000,
        "description": "Voyage AI code-optimized embeddings"
    }
}


class APIEmbedder(Embedder):
    """
    API-based embedding using various providers.
    
    Supports:
    - OpenAI (text-embedding-3-small/large)
    - Cohere (embed-english-v3.0)
    - Voyage AI (voyage-large-2)
    - Custom endpoints
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        
        # Get config
        self._config = API_MODEL_CONFIGS.get(model_name, {
            "provider": "custom",
            "dimensions": 1536,
            "max_tokens": 8191,
            "description": "Custom API model"
        })
        
        self._provider = self._config.get("provider", "custom")
        
        # Set default base URLs
        if not base_url:
            if self._provider == "llamacpp":
                # Local embedding service
                self.base_url = "http://localhost:8602/v1"
            elif self._provider == "openai":
                self.base_url = "https://api.openai.com/v1"
            elif self._provider == "cohere":
                self.base_url = "https://api.cohere.ai/v1"
            elif self._provider == "voyage":
                self.base_url = "https://api.voyageai.com/v1"
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {"Content-Type": "application/json"}
        
        if self.api_key:
            if self._provider == "cohere":
                headers["Authorization"] = f"Bearer {self.api_key}"
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    async def _embed_openai(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """Embed using OpenAI API"""
        url = f"{self.base_url}/embeddings"
        
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"OpenAI API error: {error}")
                
                result = await response.json()
                
                # Sort by index to ensure correct order
                embeddings_data = sorted(
                    result["data"],
                    key=lambda x: x["index"]
                )
                
                return [item["embedding"] for item in embeddings_data]
    
    async def _embed_cohere(
        self,
        texts: List[str],
        input_type: str = "search_document"
    ) -> List[List[float]]:
        """Embed using Cohere API"""
        url = f"{self.base_url}/embed"
        
        payload = {
            "model": self.model_name,
            "texts": texts,
            "input_type": input_type,
            "truncate": "END"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"Cohere API error: {error}")
                
                result = await response.json()
                return result["embeddings"]
    
    async def _embed_voyage(
        self,
        texts: List[str],
        input_type: str = "document"
    ) -> List[List[float]]:
        """Embed using Voyage AI API"""
        url = f"{self.base_url}/embeddings"
        
        payload = {
            "model": self.model_name,
            "input": texts,
            "input_type": input_type
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"Voyage API error: {error}")
                
                result = await response.json()
                return [item["embedding"] for item in result["data"]]
    
    async def embed(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = False
    ) -> EmbeddingResult:
        """Embed texts using API"""
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model=self.model_name,
                dimensions=self._config["dimensions"],
                token_count=0
            )
        
        start_time = time.time()
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Preprocess
            processed = [
                self._truncate_text(
                    self._preprocess_text(t),
                    self._config["max_tokens"]
                )
                for t in batch
            ]
            
            # Call appropriate API
            if self._provider == "openai" or self._provider == "llamacpp":
                # Both OpenAI and llama.cpp use the same API format
                embeddings = await self._embed_openai(processed)
            elif self._provider == "cohere":
                embeddings = await self._embed_cohere(processed)
            elif self._provider == "voyage":
                embeddings = await self._embed_voyage(processed)
            else:
                # Custom endpoint - assume OpenAI-compatible
                embeddings = await self._embed_openai(processed)
            
            all_embeddings.extend(embeddings)
        
        elapsed_ms = (time.time() - start_time) * 1000
        total_chars = sum(len(t) for t in texts)
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model=self.model_name,
            dimensions=self._config["dimensions"],
            token_count=total_chars // 4,
            processing_time_ms=elapsed_ms,
            batch_size=len(texts)
        )
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        # Some providers use different input types for queries vs documents
        processed = self._truncate_text(
            self._preprocess_text(text),
            self._config["max_tokens"]
        )
        
        if self._provider == "cohere":
            embeddings = await self._embed_cohere([processed], "search_query")
        elif self._provider == "voyage":
            embeddings = await self._embed_voyage([processed], "query")
        else:
            # OpenAI, llama.cpp, and custom endpoints
            embeddings = await self._embed_openai([processed])
        
        return embeddings[0] if embeddings else []
    
    def get_model_info(self) -> EmbeddingModel:
        """Get model information"""
        return EmbeddingModel(
            name=self.model_name,
            dimensions=self._config["dimensions"],
            max_tokens=self._config["max_tokens"],
            description=self._config["description"],
            provider=self._provider,
            is_available=bool(self.api_key)
        )
    
    async def is_available(self) -> bool:
        """Check if API is available"""
        if not self.api_key:
            return False
        
        try:
            # Test with minimal request
            await self.embed_query("test")
            return True
        except Exception:
            return False
    
    @staticmethod
    def list_available_models() -> List[EmbeddingModel]:
        """List all available API models"""
        return [
            EmbeddingModel(
                name=name,
                dimensions=config["dimensions"],
                max_tokens=config["max_tokens"],
                description=config["description"],
                provider=config["provider"]
            )
            for name, config in API_MODEL_CONFIGS.items()
        ]
