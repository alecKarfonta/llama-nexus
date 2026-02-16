"""External service clients (NER, Relationship Extraction).

These services are optional and GPU-based. Start with:
  docker compose --profile graphrag-ner up -d
  docker compose --profile graphrag-rel up -d
"""
from .ner_client import NERClient
from .rel_extractor import RelationshipExtractor

__all__ = ['NERClient', 'RelationshipExtractor']
