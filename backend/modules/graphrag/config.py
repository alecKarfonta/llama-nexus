"""
GraphRAG Configuration Management

Centralized configuration for all GraphRAG components.
"""

import os
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGConfig:
    """Configuration for GraphRAG services."""
    
    # Feature toggle
    enabled: bool = False
    
    # Neo4j settings
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # Qdrant settings (use existing llama-nexus Qdrant)
    qdrant_url: str = "http://qdrant:6333"
    graphrag_collection: str = "graphrag_documents"
    
    # External services (GPU-based, opt-in)
    ner_api_url: Optional[str] = None
    rel_api_url: Optional[str] = None
    code_rag_url: Optional[str] = None
    
    # LLM settings for reasoning
    llm_api_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    
    # Anthropic for contextual embeddings (optional)
    anthropic_api_key: Optional[str] = None
    
    @property
    def ner_enabled(self) -> bool:
        """Check if NER service is configured."""
        return self.ner_api_url is not None
    
    @property
    def rel_enabled(self) -> bool:
        """Check if Relationship Extraction service is configured."""
        return self.rel_api_url is not None
    
    @property
    def code_rag_enabled(self) -> bool:
        """Check if Code-RAG service is configured."""
        return self.code_rag_url is not None


_config: Optional[GraphRAGConfig] = None


def get_graphrag_config() -> GraphRAGConfig:
    """Get or create GraphRAG configuration from environment."""
    global _config
    
    if _config is None:
        enabled = os.getenv("GRAPHRAG_ENABLED", "false").lower() == "true"
        
        # Check for NER/Rel services - only set if explicitly configured
        ner_url = os.getenv("NER_API_URL")
        rel_url = os.getenv("REL_API_URL")
        code_rag_url = os.getenv("CODE_RAG_URL")
        
        # Auto-detect services if running under graphrag profile
        if enabled and not ner_url:
            # Check if service is reachable (will be validated at runtime)
            ner_url = os.getenv("GRAPHRAG_NER_URL", None)
        if enabled and not rel_url:
            rel_url = os.getenv("GRAPHRAG_REL_URL", None)
        if enabled and not code_rag_url:
            code_rag_url = os.getenv("GRAPHRAG_CODE_RAG_URL", None)
        
        _config = GraphRAGConfig(
            enabled=enabled,
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            qdrant_url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
            graphrag_collection=os.getenv("GRAPHRAG_COLLECTION", "graphrag_documents"),
            ner_api_url=ner_url,
            rel_api_url=rel_url,
            code_rag_url=code_rag_url,
            llm_api_url=os.getenv("LLM_API_URL"),
            llm_api_key=os.getenv("LLM_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        
        if enabled:
            logger.info("GraphRAG enabled")
            logger.info(f"  Neo4j: {_config.neo4j_uri}")
            logger.info(f"  NER service: {'✓ ' + ner_url if ner_url else '✗ Not configured'}")
            logger.info(f"  Rel service: {'✓ ' + rel_url if rel_url else '✗ Not configured'}")
            logger.info(f"  Code-RAG: {'✓ ' + code_rag_url if code_rag_url else '✗ Not configured'}")
        else:
            logger.info("GraphRAG disabled")
    
    return _config


def reset_config():
    """Reset configuration (for testing)."""
    global _config
    _config = None
