"""
Central application state and shared instances.
Extracted from main.py to prevent circular dependencies.
"""
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

from enhanced_logger import enhanced_logger as logger
from modules.managers import (
    LlamaCPPManager,
    ModelDownloadManager,
    EmbeddingManager,
    STTManager,
    StreamingSTTManager,
    TTSManager
)

# Initialize service managers
manager = LlamaCPPManager()
download_manager = ModelDownloadManager()
embedding_manager = EmbeddingManager()
stt_manager = STTManager()
streaming_stt_manager = StreamingSTTManager()
tts_manager = TTSManager()

# Wire up download manager reference
manager.download_manager = download_manager

def _merge_and_persist_config(new_config: Dict[str, Any]):
    """
    Deep merge the new configuration into the existing configuration
    and persist it to disk. 
    """
    import copy
    
    # Deep merge helper
    def deep_merge(d1, d2):
        for k, v in d2.items():
            if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
                deep_merge(d1[k], v)
            else:
                d1[k] = copy.deepcopy(v)
        return d1

    # Merge new config into existing
    deep_merge(manager.config, new_config)

    # Persist the merged configuration
    try:
        config_file = Path("/tmp/llamacpp_config.json")
        with open(config_file, "w") as f:
            json.dump(manager.config, f, indent=2)
        logger.info(f"Persisted configuration to {config_file}")
    except Exception as e:
        logger.error(f"Failed to persist configuration: {e}")

# RAG Embedding Configuration
USE_DEPLOYED_EMBEDDINGS = os.getenv("USE_DEPLOYED_EMBEDDINGS", "false").lower() == "true"
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://llamacpp-embed:8080/v1")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "nomic-embed-text-v1.5")
GRAPHRAG_ENABLED = os.getenv("GRAPHRAG_ENABLED", "true").lower() == "true"

# Embedder cache
_embedder_cache: Dict[str, Any] = {}

def create_embedder(model_name: Optional[str] = None, use_deployed: Optional[bool] = None):
    """
    Factory function to create or retrieve a cached embedder instance.
    """
    from modules.rag.embeddings import LocalEmbedder, APIEmbedder
    
    is_deployed = use_deployed if use_deployed is not None else USE_DEPLOYED_EMBEDDINGS
    target_model = model_name or DEFAULT_EMBEDDING_MODEL
    
    # Create cache key
    cache_key = f"{'api' if is_deployed else 'local'}:{target_model}"
    
    if cache_key in _embedder_cache:
        return _embedder_cache[cache_key]
        
    try:
        if is_deployed:
            # Add protocol prefix if missing
            url = EMBEDDING_SERVICE_URL
            if not url.startswith("http://") and not url.startswith("https://"):
                url = f"http://{url}"
                
            logger.info(f"Creating new API Embedder caching instance: {url} (model: {target_model})")
            embedder = APIEmbedder(url=url, model=target_model)
        else:
            logger.info(f"Creating new Local Embedder caching instance for model: {target_model}")
            embedder = LocalEmbedder(model_name=target_model)
            
        _embedder_cache[cache_key] = embedder
        return embedder
    except Exception as e:
        logger.error(f"Failed to create embedder. mode=api={is_deployed}, model={target_model}, error={e}")
        # Only fallback if trying to use API, if local fails we have a problem
        if is_deployed:
            logger.warning("API embedding failed, falling back to local embedding")
            # Try recursive fallback to local
            return create_embedder(model_name=target_model, use_deployed=False)
        else:
            raise e
