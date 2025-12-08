"""
Comprehensive RAG (Retrieval-Augmented Generation) System

This module provides:
- Document management with domain hierarchy
- Multiple vector store backends (Qdrant, ChromaDB)
- GraphRAG with entity/relationship extraction
- Multiple chunking strategies
- Embedding model management
- Advanced retrieval mechanisms
"""

from .document_manager import DocumentManager, Document, Domain
from .vector_stores.base import VectorStore
from .vector_stores.qdrant_store import QdrantStore
from .chunkers.base import Chunker
from .embedders.base import Embedder
from .graph_rag import GraphRAG, Entity, Relationship

__all__ = [
    'DocumentManager',
    'Document', 
    'Domain',
    'VectorStore',
    'QdrantStore',
    'Chunker',
    'Embedder',
    'GraphRAG',
    'Entity',
    'Relationship',
]
