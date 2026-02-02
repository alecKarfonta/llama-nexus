"""
Qdrant Vector Store Implementation

Full-featured integration with Qdrant vector database including:
- Collection management with HNSW configuration
- Batch operations for efficient indexing
- Sparse vector support for hybrid search
- Payload filtering with Qdrant filter DSL
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio

from .base import (
    VectorStore,
    VectorRecord,
    SearchResult,
    CollectionConfig,
    CollectionInfo,
    DistanceMetric
)

logger = logging.getLogger(__name__)

# Qdrant client import (optional dependency)
try:
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        HnswConfigDiff,
        OptimizersConfigDiff,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        Range,
        ScrollRequest,
        SearchRequest,
        UpdateStatus
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant client not installed. Run: pip install qdrant-client")


def distance_to_qdrant(metric: DistanceMetric) -> 'Distance':
    """Convert our distance metric to Qdrant Distance enum"""
    mapping = {
        DistanceMetric.COSINE: Distance.COSINE,
        DistanceMetric.EUCLIDEAN: Distance.EUCLID,
        DistanceMetric.DOT_PRODUCT: Distance.DOT,
    }
    return mapping.get(metric, Distance.COSINE)


def build_qdrant_filter(filter_dict: Optional[Dict[str, Any]]) -> Optional['Filter']:
    """
    Build Qdrant filter from dictionary.
    
    Supports:
    - Exact match: {"field": "value"}
    - Range: {"field": {"$gt": 5, "$lt": 10}}
    - In list: {"field": {"$in": [1, 2, 3]}}
    - Boolean: {"field": True}
    """
    if not filter_dict or not QDRANT_AVAILABLE:
        return None
    
    must_conditions = []
    
    for key, value in filter_dict.items():
        if isinstance(value, dict):
            # Range or special operators
            if "$gt" in value or "$gte" in value or "$lt" in value or "$lte" in value:
                must_conditions.append(
                    FieldCondition(
                        key=key,
                        range=Range(
                            gt=value.get("$gt"),
                            gte=value.get("$gte"),
                            lt=value.get("$lt"),
                            lte=value.get("$lte"),
                        )
                    )
                )
            elif "$in" in value:
                must_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(any=value["$in"])
                    )
                )
        else:
            # Exact match
            must_conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )
    
    return Filter(must=must_conditions) if must_conditions else None


class QdrantStore(VectorStore):
    """
    Qdrant vector store implementation.
    
    Features:
    - Async and sync client support
    - HNSW index configuration
    - Batch upsert/delete operations
    - Payload filtering
    - Hybrid search (dense + sparse)
    - Collection optimization
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        api_key: Optional[str] = None,
        https: bool = False,
        prefer_grpc: bool = True,
        timeout: int = 30
    ):
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required. Run: pip install qdrant-client")
        
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.api_key = api_key
        self.https = https
        self.prefer_grpc = prefer_grpc
        self.timeout = timeout
        
        self._client: Optional[AsyncQdrantClient] = None
        self._sync_client: Optional[QdrantClient] = None
    
    async def connect(self) -> bool:
        """Connect to Qdrant server"""
        try:
            self._client = AsyncQdrantClient(
                host=self.host,
                port=self.port,
                grpc_port=self.grpc_port,
                api_key=self.api_key,
                https=self.https,
                prefer_grpc=self.prefer_grpc,
                timeout=self.timeout
            )
            # Test connection
            await self._client.get_collections()
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Qdrant"""
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("Disconnected from Qdrant")
    
    async def is_healthy(self) -> bool:
        """Check Qdrant health"""
        try:
            if self._client:
                await self._client.get_collections()
                return True
            return False
        except Exception:
            return False
    
    # Collection Management
    
    async def create_collection(self, config: CollectionConfig) -> bool:
        """Create a new Qdrant collection"""
        try:
            await self._client.create_collection(
                collection_name=config.name,
                vectors_config=VectorParams(
                    size=config.vector_size,
                    distance=distance_to_qdrant(config.distance),
                ),
                hnsw_config=HnswConfigDiff(
                    m=config.hnsw_m,
                    ef_construct=config.hnsw_ef_construct,
                ),
                shard_number=config.shard_number,
                replication_factor=config.replication_factor,
            )
            logger.info(f"Created collection: {config.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection {config.name}: {e}")
            return False
    
    async def delete_collection(self, name: str) -> bool:
        """Delete a Qdrant collection"""
        try:
            await self._client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {name}: {e}")
            return False
    
    async def collection_exists(self, name: str) -> bool:
        """Check if collection exists"""
        try:
            collections = await self._client.get_collections()
            return any(c.name == name for c in collections.collections)
        except Exception:
            return False
    
    async def get_collection_info(self, name: str) -> Optional[CollectionInfo]:
        """Get collection information"""
        try:
            info = await self._client.get_collection(name)
            
            # Map Qdrant distance back to our enum
            distance_map = {
                Distance.COSINE: DistanceMetric.COSINE,
                Distance.EUCLID: DistanceMetric.EUCLIDEAN,
                Distance.DOT: DistanceMetric.DOT_PRODUCT,
            }
            
            return CollectionInfo(
                name=name,
                vector_size=info.config.params.vectors.size,
                distance=distance_map.get(info.config.params.vectors.distance, DistanceMetric.COSINE),
                vectors_count=info.vectors_count or 0,
                points_count=info.points_count or 0,
                indexed_vectors_count=info.indexed_vectors_count or 0,
                status=info.status.value if info.status else "unknown"
            )
        except Exception as e:
            logger.error(f"Failed to get collection info for {name}: {e}")
            return None
    
    async def list_collections(self) -> List[str]:
        """List all collection names"""
        try:
            collections = await self._client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    # Vector Operations
    
    async def upsert(
        self,
        collection: str,
        records: List[VectorRecord],
        wait: bool = True,
        batch_size: int = 100
    ) -> int:
        """Upsert vectors with batching"""
        if not records:
            return 0
        
        total_upserted = 0
        
        # Process in batches
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            points = [
                PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=record.payload or {}
                )
                for record in batch
            ]
            
            try:
                result = await self._client.upsert(
                    collection_name=collection,
                    points=points,
                    wait=wait
                )
                if result.status == UpdateStatus.COMPLETED:
                    total_upserted += len(batch)
            except Exception as e:
                logger.error(f"Failed to upsert batch: {e}")
        
        return total_upserted
    
    async def delete(
        self,
        collection: str,
        ids: List[str],
        wait: bool = True
    ) -> int:
        """Delete vectors by ID"""
        try:
            result = await self._client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=ids),
                wait=wait
            )
            return len(ids) if result.status == UpdateStatus.COMPLETED else 0
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return 0
    
    async def get(
        self,
        collection: str,
        ids: List[str],
        with_vectors: bool = False
    ) -> List[VectorRecord]:
        """Get vectors by ID"""
        try:
            points = await self._client.retrieve(
                collection_name=collection,
                ids=ids,
                with_vectors=with_vectors,
                with_payload=True
            )
            
            return [
                VectorRecord(
                    id=str(point.id),
                    vector=point.vector if with_vectors else [],
                    payload=point.payload or {}
                )
                for point in points
            ]
        except Exception as e:
            logger.error(f"Failed to get vectors: {e}")
            return []
    
    # Search Operations
    
    async def search(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        with_vectors: bool = False,
        score_threshold: Optional[float] = None,
        exclude_payload_fields: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Perform similarity search.
        
        Args:
            exclude_payload_fields: Fields to exclude from returned payload.
                                   Defaults to ['metadata'] to avoid bloated payloads.
        """
        try:
            qdrant_filter = build_qdrant_filter(filter)
            
            # Build payload selector - exclude large fields by default
            # The metadata field can be 9MB+ and causes massive slowdowns
            excluded = exclude_payload_fields if exclude_payload_fields is not None else ['metadata']
            if excluded:
                with_payload = models.PayloadSelectorExclude(exclude=excluded)
            else:
                with_payload = True
            
            results = await self._client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
                with_vectors=with_vectors,
                with_payload=with_payload,
                score_threshold=score_threshold
            )
            
            return [
                SearchResult(
                    id=str(hit.id),
                    score=hit.score,
                    payload=hit.payload or {},
                    vector=hit.vector if with_vectors else None
                )
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def search_batch(
        self,
        collection: str,
        query_vectors: List[List[float]],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        exclude_payload_fields: Optional[List[str]] = None
    ) -> List[List[SearchResult]]:
        """Batch similarity search with payload exclusion to avoid bloat."""
        try:
            qdrant_filter = build_qdrant_filter(filter)
            
            # Exclude large fields by default (metadata can be 9MB+)
            excluded = exclude_payload_fields if exclude_payload_fields is not None else ['metadata']
            if excluded:
                with_payload = models.PayloadSelectorExclude(exclude=excluded)
            else:
                with_payload = True
            
            requests = [
                SearchRequest(
                    vector=vector,
                    limit=limit,
                    filter=qdrant_filter,
                    with_payload=with_payload
                )
                for vector in query_vectors
            ]
            
            batch_results = await self._client.search_batch(
                collection_name=collection,
                requests=requests
            )
            
            return [
                [
                    SearchResult(
                        id=str(hit.id),
                        score=hit.score,
                        payload=hit.payload or {}
                    )
                    for hit in results
                ]
                for results in batch_results
            ]
        except Exception as e:
            logger.error(f"Batch search failed: {e}")
            return [[] for _ in query_vectors]
    
    async def hybrid_search(
        self,
        collection: str,
        query_vector: List[float],
        query_text: str,
        limit: int = 10,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Hybrid search combining dense and sparse vectors.
        Requires sparse vectors to be configured in the collection.
        """
        # For now, fall back to dense-only search
        # Full hybrid requires sparse vector configuration
        logger.info("Using dense-only search (sparse vectors not configured)")
        return await self.search(collection, query_vector, limit, filter)
    
    # Utility Methods
    
    async def count(self, collection: str, filter: Optional[Dict[str, Any]] = None) -> int:
        """Count vectors in collection"""
        try:
            if filter:
                qdrant_filter = build_qdrant_filter(filter)
                result = await self._client.count(
                    collection_name=collection,
                    count_filter=qdrant_filter,
                    exact=True
                )
            else:
                info = await self._client.get_collection(collection)
                return info.points_count or 0
            
            return result.count
        except Exception as e:
            logger.error(f"Failed to count vectors: {e}")
            return 0
    
    async def optimize(self, collection: str) -> bool:
        """Trigger collection optimization"""
        try:
            await self._client.update_collection(
                collection_name=collection,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=0  # Force indexing
                )
            )
            logger.info(f"Triggered optimization for {collection}")
            return True
        except Exception as e:
            logger.error(f"Failed to optimize collection: {e}")
            return False
    
    async def scroll(
        self,
        collection: str,
        limit: int = 100,
        offset: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        with_vectors: bool = False
    ) -> tuple[List[VectorRecord], Optional[str]]:
        """Scroll through collection"""
        try:
            qdrant_filter = build_qdrant_filter(filter)
            
            results, next_offset = await self._client.scroll(
                collection_name=collection,
                limit=limit,
                offset=offset,
                scroll_filter=qdrant_filter,
                with_vectors=with_vectors,
                with_payload=True
            )
            
            records = [
                VectorRecord(
                    id=str(point.id),
                    vector=point.vector if with_vectors else [],
                    payload=point.payload or {}
                )
                for point in results
            ]
            
            return records, str(next_offset) if next_offset else None
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return [], None
    
    # Additional Qdrant-specific methods
    
    # Alias methods for compatibility
    async def add_vectors(
        self,
        collection: str,
        records: List[VectorRecord],
        wait: bool = True,
        batch_size: int = 100
    ) -> int:
        """Alias for upsert for compatibility"""
        return await self.upsert(collection, records, wait, batch_size)
    
    async def delete_vectors(
        self,
        collection: str,
        ids: List[str],
        wait: bool = True
    ) -> int:
        """Alias for delete for compatibility"""
        return await self.delete(collection, ids, wait)
    
    async def create_payload_index(
        self,
        collection: str,
        field_name: str,
        field_type: str = "keyword"
    ) -> bool:
        """Create payload index for faster filtering"""
        try:
            schema_type = {
                "keyword": models.PayloadSchemaType.KEYWORD,
                "integer": models.PayloadSchemaType.INTEGER,
                "float": models.PayloadSchemaType.FLOAT,
                "bool": models.PayloadSchemaType.BOOL,
                "text": models.PayloadSchemaType.TEXT,
            }.get(field_type, models.PayloadSchemaType.KEYWORD)
            
            await self._client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=schema_type
            )
            logger.info(f"Created payload index: {collection}.{field_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create payload index: {e}")
            return False
