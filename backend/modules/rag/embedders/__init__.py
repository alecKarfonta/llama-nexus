"""Embedding model implementations"""
from .base import Embedder, EmbeddingResult
from .local_embedder import LocalEmbedder
from .api_embedder import APIEmbedder

__all__ = ['Embedder', 'EmbeddingResult', 'LocalEmbedder', 'APIEmbedder']
