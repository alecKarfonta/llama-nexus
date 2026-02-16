"""
GraphRAG Module - Knowledge Graph-Enhanced Retrieval

This module provides advanced RAG capabilities using knowledge graphs:
- Semantic chunking of documents
- Entity extraction via GLiNER (requires NER service)
- Relationship extraction (requires Rel service)
- Knowledge graph construction with Neo4j
- Hybrid retrieval (vector + graph + keyword)
- Advanced reasoning (multi-hop, causal, comparative)

Services:
- Core functionality works without GPU services
- NER service (graphrag-ner) - Entity extraction, opt-in
- Rel service (graphrag-rel) - Relationship extraction, opt-in
"""

from .config import GraphRAGConfig, get_graphrag_config

__all__ = [
    'GraphRAGConfig',
    'get_graphrag_config',
]
