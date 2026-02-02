"""
GraphRAG Integration Wrapper

Provides a unified interface to GraphRAG components with graceful degradation
when the optional graphrag submodule is not initialized.

Usage:
    from modules.graphrag_wrapper import is_graphrag_available, GraphRAGWrapper
    
    if is_graphrag_available():
        wrapper = GraphRAGWrapper(neo4j_uri="bolt://neo4j:7687")
        chunks = wrapper.semantic_chunk(text)
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Check if graphrag submodule exists
# In Docker: project is mounted at /home/alec/git/llama-nexus (read-only)
# Fallback to relative path for local development
GRAPHRAG_BACKEND_PATH = os.getenv(
    'GRAPHRAG_BACKEND_PATH',
    '/home/alec/git/llama-nexus/graphrag/backend'
)
# Fallback: try relative path if absolute doesn't exist
if not os.path.exists(GRAPHRAG_BACKEND_PATH):
    GRAPHRAG_BACKEND_PATH = os.path.join(
        os.path.dirname(__file__), '../../graphrag/backend'
    )
GRAPHRAG_AVAILABLE = os.path.exists(GRAPHRAG_BACKEND_PATH)

# Track which components are available
SEMANTIC_CHUNKER_AVAILABLE = False
ENTITY_EXTRACTOR_AVAILABLE = False
KG_BUILDER_AVAILABLE = False
HYBRID_RETRIEVER_AVAILABLE = False

if GRAPHRAG_AVAILABLE:
    sys.path.insert(0, GRAPHRAG_BACKEND_PATH)
    
    # Try importing each component independently
    try:
        from semantic_chunker import SemanticChunker
        SEMANTIC_CHUNKER_AVAILABLE = True
        logger.info("GraphRAG SemanticChunker loaded")
    except ImportError as e:
        logger.warning(f"SemanticChunker not available: {e}")
        SemanticChunker = None
    
    try:
        from entity_extractor import EntityExtractor, Entity, Relationship
        ENTITY_EXTRACTOR_AVAILABLE = True
        logger.info("GraphRAG EntityExtractor loaded")
    except ImportError as e:
        logger.warning(f"EntityExtractor not available: {e}")
        EntityExtractor = None
    
    try:
        from knowledge_graph_builder import KnowledgeGraphBuilder
        KG_BUILDER_AVAILABLE = True
        logger.info("GraphRAG KnowledgeGraphBuilder loaded")
    except ImportError as e:
        logger.warning(f"KnowledgeGraphBuilder not available (neo4j needed): {e}")
        KnowledgeGraphBuilder = None
    
    try:
        from hybrid_retriever import HybridRetriever
        HYBRID_RETRIEVER_AVAILABLE = True
        logger.info("GraphRAG HybridRetriever loaded")
    except ImportError as e:
        logger.warning(f"HybridRetriever not available: {e}")
        HybridRetriever = None
    
    # Consider GraphRAG available if at least SemanticChunker works
    GRAPHRAG_AVAILABLE = SEMANTIC_CHUNKER_AVAILABLE


def is_graphrag_available() -> bool:
    """Check if GraphRAG submodule is available and importable."""
    return GRAPHRAG_AVAILABLE


@dataclass
class ExtractionResult:
    """Result from entity/relationship extraction."""
    entities: List[Any]
    relationships: List[Dict[str, Any]]
    claims: List[str]


class GraphRAGWrapper:
    """
    Unified interface to GraphRAG components.
    
    Provides access to:
    - SemanticChunker: Embedding-based document chunking
    - EntityExtractor: GLiNER-based NER and relationship extraction
    - KnowledgeGraphBuilder: Neo4j knowledge graph construction
    - HybridRetriever: Combined vector + graph + keyword search
    """
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        qdrant_url: str = "http://qdrant:6333",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize GraphRAG wrapper.
        
        Args:
            neo4j_uri: Neo4j bolt URI (optional - KG features disabled if not provided)
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            qdrant_url: Qdrant vector store URL
            embedding_model: Model for semantic chunking embeddings
        """
        if not GRAPHRAG_AVAILABLE:
            raise ImportError(
                "GraphRAG submodule not available. "
                "Run 'git submodule update --init --recursive' to initialize."
            )
        
        self._neo4j_uri = neo4j_uri
        self._qdrant_url = qdrant_url
        
        # Lazy initialization - components created on first use
        self._chunker: Optional[SemanticChunker] = None
        self._extractor: Optional[EntityExtractor] = None
        self._kg_builder: Optional[KnowledgeGraphBuilder] = None
        self._retriever: Optional[HybridRetriever] = None
        
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._embedding_model = embedding_model
    
    @property
    def chunker(self) -> "SemanticChunker":
        """Lazy-load semantic chunker."""
        if self._chunker is None:
            self._chunker = SemanticChunker(model_name=self._embedding_model)
        return self._chunker
    
    @property
    def extractor(self) -> "EntityExtractor":
        """Lazy-load entity extractor."""
        if self._extractor is None:
            self._extractor = EntityExtractor()
        return self._extractor
    
    @property
    def kg_builder(self) -> Optional["KnowledgeGraphBuilder"]:
        """Lazy-load knowledge graph builder (requires Neo4j)."""
        if self._kg_builder is None and self._neo4j_uri:
            self._kg_builder = KnowledgeGraphBuilder(
                neo4j_uri=self._neo4j_uri,
                neo4j_user=self._neo4j_user,
                neo4j_password=self._neo4j_password
            )
        return self._kg_builder
    
    @property
    def retriever(self) -> "HybridRetriever":
        """Lazy-load hybrid retriever."""
        if self._retriever is None:
            self._retriever = HybridRetriever(qdrant_url=self._qdrant_url)
        return self._retriever
    
    def semantic_chunk(self, text: str) -> List[str]:
        """
        Create semantic chunks from text using embedding-based clustering.
        
        Args:
            text: Document text to chunk
            
        Returns:
            List of text chunks grouped by semantic similarity
        """
        return self.chunker.create_semantic_chunks(text)
    
    def extract_entities(
        self, 
        text: str, 
        domain: str = "general"
    ) -> ExtractionResult:
        """
        Extract entities and relationships from text using GLiNER.
        
        Args:
            text: Text to extract from
            domain: Domain/context for extraction
            
        Returns:
            ExtractionResult with entities, relationships, and claims
        """
        result = self.extractor.extract_entities_and_relations(text, domain)
        return ExtractionResult(
            entities=result.entities if hasattr(result, 'entities') else result.get('entities', []),
            relationships=result.relationships if hasattr(result, 'relationships') else result.get('relationships', []),
            claims=result.claims if hasattr(result, 'claims') else result.get('claims', [])
        )
    
    def build_knowledge_graph(
        self,
        entities: List[Any],
        relationships: List[Dict[str, Any]],
        domain: str = "general"
    ) -> bool:
        """
        Add entities and relationships to the Neo4j knowledge graph.
        
        Args:
            entities: List of extracted entities
            relationships: List of relationships between entities
            domain: Domain to tag the entities with
            
        Returns:
            True if successful, raises if Neo4j not configured
        """
        if not self.kg_builder:
            raise RuntimeError(
                "Knowledge graph builder not available. "
                "Configure NEO4J_URI to enable KG features."
            )
        self.kg_builder.add_entities_and_relationships(entities, relationships, domain)
        return True
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector, graph, and keyword search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with content, source, and scores
        """
        results = self.retriever.retrieve(query, top_k)
        return [
            {
                "content": r.content,
                "source": r.source,
                "score": r.score,
                "result_type": r.result_type,
                "metadata": r.metadata
            }
            for r in results
        ]
    
    def close(self):
        """Clean up connections."""
        if self._kg_builder:
            self._kg_builder.close()
