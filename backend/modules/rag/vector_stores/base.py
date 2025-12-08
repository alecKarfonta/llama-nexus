"""
Base Vector Store Interface

Provides abstract interface for vector database operations.
Implementations: Qdrant, ChromaDB, Milvus, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DistanceMetric(str, Enum):
    """Supported distance metrics for vector similarity"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot"


@dataclass
class VectorRecord:
    """A vector record with ID, vector, and metadata"""
    id: str
    vector: List[float]
    payload: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None


@dataclass
class SearchResult:
    """Search result with score and metadata"""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None


@dataclass
class CollectionConfig:
    """Configuration for a vector collection"""
    name: str
    vector_size: int
    distance: DistanceMetric = DistanceMetric.COSINE
    # HNSW index parameters
    hnsw_m: int = 16  # Number of edges per node
    hnsw_ef_construct: int = 100  # Size of dynamic candidate list
    # Quantization
    quantization_enabled: bool = False
    quantization_type: str = "scalar"  # scalar, product
    # Sharding
    shard_number: int = 1
    replication_factor: int = 1


@dataclass
class CollectionInfo:
    """Information about a collection"""
    name: str
    vector_size: int
    distance: DistanceMetric
    vectors_count: int
    points_count: int
    indexed_vectors_count: int
    status: str


class VectorStore(ABC):
    """
    Abstract base class for vector store implementations.
    
    Provides unified interface for:
    - Collection management
    - Vector CRUD operations
    - Similarity search
    - Filtering and hybrid search
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the vector store"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the vector store"""
        pass
    
    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if the vector store is healthy"""
        pass
    
    # Collection Management
    
    @abstractmethod
    async def create_collection(self, config: CollectionConfig) -> bool:
        """Create a new collection"""
        pass
    
    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection"""
        pass
    
    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists"""
        pass
    
    @abstractmethod
    async def get_collection_info(self, name: str) -> Optional[CollectionInfo]:
        """Get collection information"""
        pass
    
    @abstractmethod
    async def list_collections(self) -> List[str]:
        """List all collection names"""
        pass
    
    # Vector Operations
    
    @abstractmethod
    async def upsert(
        self,
        collection: str,
        records: List[VectorRecord],
        wait: bool = True
    ) -> int:
        """
        Upsert vectors into collection.
        Returns number of vectors upserted.
        """
        pass
    
    @abstractmethod
    async def delete(
        self,
        collection: str,
        ids: List[str],
        wait: bool = True
    ) -> int:
        """
        Delete vectors by ID.
        Returns number of vectors deleted.
        """
        pass
    
    @abstractmethod
    async def get(
        self,
        collection: str,
        ids: List[str],
        with_vectors: bool = False
    ) -> List[VectorRecord]:
        """Get vectors by ID"""
        pass
    
    # Search Operations
    
    @abstractmethod
    async def search(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        with_vectors: bool = False,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Perform similarity search.
        
        Args:
            collection: Collection name
            query_vector: Query embedding
            limit: Maximum results
            filter: Payload filter conditions
            with_vectors: Include vectors in results
            score_threshold: Minimum score threshold
            
        Returns:
            List of search results sorted by score
        """
        pass
    
    @abstractmethod
    async def search_batch(
        self,
        collection: str,
        query_vectors: List[List[float]],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[List[SearchResult]]:
        """Batch similarity search"""
        pass
    
    # Hybrid Search (if supported)
    
    async def hybrid_search(
        self,
        collection: str,
        query_vector: List[float],
        query_text: str,
        limit: int = 10,
        alpha: float = 0.5,  # Weight between dense (1) and sparse (0)
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Hybrid search combining dense and sparse vectors.
        Default implementation falls back to dense-only search.
        """
        logger.warning("Hybrid search not implemented, falling back to dense search")
        return await self.search(collection, query_vector, limit, filter)
    
    # Utility Methods
    
    @abstractmethod
    async def count(self, collection: str, filter: Optional[Dict[str, Any]] = None) -> int:
        """Count vectors in collection, optionally filtered"""
        pass
    
    @abstractmethod
    async def optimize(self, collection: str) -> bool:
        """Optimize collection index"""
        pass
    
    async def scroll(
        self,
        collection: str,
        limit: int = 100,
        offset: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        with_vectors: bool = False
    ) -> tuple[List[VectorRecord], Optional[str]]:
        """
        Scroll through collection.
        Returns (records, next_offset) where next_offset is None when done.
        """
        raise NotImplementedError("Scroll not implemented")
