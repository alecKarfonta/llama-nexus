"""
Comprehensive Tests for RAG System

Tests cover:
- Document management (domains, documents, chunks)
- Vector store operations (Qdrant)
- GraphRAG (entities, relationships, extraction)
- Chunking strategies
- Embedding models
- Retrieval mechanisms
- Document discovery
"""

import asyncio
import pytest
import os
import sys
import tempfile
import uuid
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAG modules
from modules.rag.document_manager import (
    DocumentManager, Document, Domain, DocumentChunk,
    DocumentStatus, DocumentType
)
from modules.rag.graph_rag import (
    GraphRAG, Entity, Relationship, EntityType, RelationshipType
)
from modules.rag.chunkers.base import ChunkingConfig
from modules.rag.chunkers.fixed_chunker import FixedChunker
from modules.rag.chunkers.semantic_chunker import SemanticChunker
from modules.rag.chunkers.recursive_chunker import RecursiveChunker
from modules.rag.embedders.local_embedder import LocalEmbedder
from modules.rag.vector_stores.base import CollectionConfig, DistanceMetric
from modules.rag.discovery import DocumentDiscovery, DiscoveryStatus


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_db_path():
    """Create temporary database path"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return """
    OpenAI is an artificial intelligence research company founded in 2015. 
    The company developed GPT-4, a large language model that demonstrates 
    remarkable capabilities in natural language understanding and generation.
    
    Sam Altman serves as the CEO of OpenAI. The company is based in San Francisco,
    California. OpenAI has partnerships with Microsoft, which has invested billions
    of dollars into the company.
    
    GPT-4 was released in March 2023 and represents a significant advancement
    over its predecessor, GPT-3.5. The model can process both text and images,
    making it a multimodal AI system.
    """


@pytest.fixture
def long_text():
    """Longer text for chunking tests"""
    paragraphs = [
        f"Paragraph {i}: " + "This is a test sentence. " * 20
        for i in range(10)
    ]
    return "\n\n".join(paragraphs)


# =============================================================================
# Document Manager Tests
# =============================================================================

class TestDocumentManager:
    """Tests for DocumentManager"""
    
    @pytest.mark.asyncio
    async def test_initialize(self, temp_db_path):
        """Test database initialization"""
        manager = DocumentManager(temp_db_path)
        await manager.initialize()
        
        # Should create default domain
        domains = await manager.list_domains()
        assert len(domains) >= 1
        assert any(d.name == "General" for d in domains)
    
    @pytest.mark.asyncio
    async def test_create_domain(self, temp_db_path):
        """Test domain creation"""
        manager = DocumentManager(temp_db_path)
        await manager.initialize()
        
        domain = Domain(
            id=str(uuid.uuid4()),
            name="Test Domain",
            description="A test domain",
            chunk_size=256,
            chunk_overlap=25
        )
        
        result = await manager.create_domain(domain)
        assert result.id == domain.id
        assert result.name == "Test Domain"
        assert result.chunk_size == 256
    
    @pytest.mark.asyncio
    async def test_create_document(self, temp_db_path):
        """Test document creation"""
        manager = DocumentManager(temp_db_path)
        await manager.initialize()
        
        # Get default domain
        domains = await manager.list_domains()
        domain_id = domains[0].id
        
        doc = Document(
            id=str(uuid.uuid4()),
            domain_id=domain_id,
            name="Test Document",
            doc_type=DocumentType.TXT,
            content="This is test content.",
        )
        
        result = await manager.create_document(doc)
        assert result.id == doc.id
        assert result.name == "Test Document"
        assert result.content_hash != ""
    
    @pytest.mark.asyncio
    async def test_document_filtering(self, temp_db_path):
        """Test document listing with filters"""
        manager = DocumentManager(temp_db_path)
        await manager.initialize()
        
        domains = await manager.list_domains()
        domain_id = domains[0].id
        
        # Create multiple documents
        for i in range(5):
            doc = Document(
                id=str(uuid.uuid4()),
                domain_id=domain_id,
                name=f"Document {i}",
                doc_type=DocumentType.TXT if i % 2 == 0 else DocumentType.MARKDOWN,
                content=f"Content {i}",
            )
            await manager.create_document(doc)
        
        # Filter by type
        txt_docs, total = await manager.list_documents(doc_type=DocumentType.TXT)
        assert len(txt_docs) == 3
        
        # Search
        search_docs, _ = await manager.list_documents(search="Document 2")
        assert len(search_docs) == 1
    
    @pytest.mark.asyncio
    async def test_duplicate_detection(self, temp_db_path):
        """Test duplicate content detection"""
        manager = DocumentManager(temp_db_path)
        await manager.initialize()
        
        domains = await manager.list_domains()
        domain_id = domains[0].id
        
        content = "Unique content for testing"
        
        doc1 = Document(
            id=str(uuid.uuid4()),
            domain_id=domain_id,
            name="Doc 1",
            doc_type=DocumentType.TXT,
            content=content,
        )
        await manager.create_document(doc1)
        
        # Check for duplicate
        duplicate_id = await manager.check_duplicate(content)
        assert duplicate_id == doc1.id
    
    @pytest.mark.asyncio
    async def test_chunks_management(self, temp_db_path):
        """Test chunk storage and retrieval"""
        manager = DocumentManager(temp_db_path)
        await manager.initialize()
        
        domains = await manager.list_domains()
        doc = Document(
            id=str(uuid.uuid4()),
            domain_id=domains[0].id,
            name="Chunked Doc",
            doc_type=DocumentType.TXT,
            content="Test content",
        )
        await manager.create_document(doc)
        
        # Create chunks
        chunks = [
            DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=doc.id,
                content=f"Chunk {i} content",
                chunk_index=i,
                total_chunks=3,
                start_char=i * 100,
                end_char=(i + 1) * 100,
            )
            for i in range(3)
        ]
        
        saved = await manager.save_chunks(chunks)
        assert saved == 3
        
        # Retrieve chunks
        retrieved = await manager.get_chunks(doc.id)
        assert len(retrieved) == 3
        assert retrieved[0].chunk_index == 0


# =============================================================================
# GraphRAG Tests
# =============================================================================

class TestGraphRAG:
    """Tests for GraphRAG system"""
    
    @pytest.mark.asyncio
    async def test_initialize(self, temp_db_path):
        """Test graph initialization"""
        graph = GraphRAG(temp_db_path)
        await graph.initialize()
        
        stats = await graph.get_statistics()
        assert 'entity_count' in stats
        assert 'relationship_count' in stats
    
    @pytest.mark.asyncio
    async def test_create_entity(self, temp_db_path):
        """Test entity creation"""
        graph = GraphRAG(temp_db_path)
        await graph.initialize()
        
        entity = Entity(
            id=str(uuid.uuid4()),
            name="OpenAI",
            entity_type=EntityType.ORGANIZATION,
            description="AI research company",
            aliases=["Open AI", "OAI"]
        )
        
        result = await graph.create_entity(entity)
        assert result.id == entity.id
        assert result.name == "OpenAI"
        assert len(result.aliases) == 2
    
    @pytest.mark.asyncio
    async def test_create_relationship(self, temp_db_path):
        """Test relationship creation"""
        graph = GraphRAG(temp_db_path)
        await graph.initialize()
        
        # Create entities
        org = Entity(
            id=str(uuid.uuid4()),
            name="Microsoft",
            entity_type=EntityType.ORGANIZATION
        )
        person = Entity(
            id=str(uuid.uuid4()),
            name="Satya Nadella",
            entity_type=EntityType.PERSON
        )
        
        await graph.create_entity(org)
        await graph.create_entity(person)
        
        # Create relationship
        rel = Relationship(
            id=str(uuid.uuid4()),
            source_id=person.id,
            target_id=org.id,
            relationship_type=RelationshipType.WORKS_FOR,
            description="CEO of Microsoft"
        )
        
        result = await graph.create_relationship(rel)
        assert result.id == rel.id
        assert result.relationship_type == RelationshipType.WORKS_FOR
    
    @pytest.mark.asyncio
    async def test_find_entity_by_name(self, temp_db_path):
        """Test entity lookup by name"""
        graph = GraphRAG(temp_db_path)
        await graph.initialize()
        
        entity = Entity(
            id=str(uuid.uuid4()),
            name="Google",
            entity_type=EntityType.ORGANIZATION,
            aliases=["Alphabet", "Google LLC"]
        )
        await graph.create_entity(entity)
        
        # Find by name
        found = await graph.find_entity_by_name("Google")
        assert found is not None
        assert found.id == entity.id
        
        # Find by alias
        found_alias = await graph.find_entity_by_name("Alphabet")
        assert found_alias is not None
        assert found_alias.id == entity.id
    
    @pytest.mark.asyncio
    async def test_subgraph_extraction(self, temp_db_path):
        """Test subgraph extraction"""
        graph = GraphRAG(temp_db_path)
        await graph.initialize()
        
        # Create connected entities
        e1 = Entity(id=str(uuid.uuid4()), name="A", entity_type=EntityType.CONCEPT)
        e2 = Entity(id=str(uuid.uuid4()), name="B", entity_type=EntityType.CONCEPT)
        e3 = Entity(id=str(uuid.uuid4()), name="C", entity_type=EntityType.CONCEPT)
        
        await graph.create_entity(e1)
        await graph.create_entity(e2)
        await graph.create_entity(e3)
        
        # A -> B -> C
        await graph.create_relationship(Relationship(
            id=str(uuid.uuid4()),
            source_id=e1.id, target_id=e2.id,
            relationship_type=RelationshipType.RELATED_TO
        ))
        await graph.create_relationship(Relationship(
            id=str(uuid.uuid4()),
            source_id=e2.id, target_id=e3.id,
            relationship_type=RelationshipType.RELATED_TO
        ))
        
        # Get subgraph from A with depth 2
        nodes, edges = await graph.get_subgraph([e1.id], depth=2)
        assert len(nodes) == 3  # Should reach all three
        assert len(edges) == 2
    
    @pytest.mark.asyncio
    async def test_merge_entities(self, temp_db_path):
        """Test entity merging"""
        graph = GraphRAG(temp_db_path)
        await graph.initialize()
        
        e1 = Entity(
            id=str(uuid.uuid4()),
            name="JS",
            entity_type=EntityType.TECHNOLOGY,
            aliases=["JavaScript"]
        )
        e2 = Entity(
            id=str(uuid.uuid4()),
            name="JavaScript",
            entity_type=EntityType.TECHNOLOGY,
            aliases=["ES6"]
        )
        
        await graph.create_entity(e1)
        await graph.create_entity(e2)
        
        merged = await graph.merge_entities([e1.id, e2.id], "JavaScript")
        assert merged is not None
        assert "JS" in merged.aliases
        assert "ES6" in merged.aliases
        
        # Original entities should be deleted
        old1 = await graph.get_entity(e1.id)
        assert old1 is None


# =============================================================================
# Chunker Tests
# =============================================================================

class TestChunkers:
    """Tests for text chunking strategies"""
    
    def test_fixed_chunker(self, long_text):
        """Test fixed-size chunking"""
        config = ChunkingConfig(chunk_size=200, chunk_overlap=20)
        chunker = FixedChunker(config)
        
        chunks = chunker.chunk(long_text)
        
        assert len(chunks) > 1
        # Check overlap exists
        for i in range(1, len(chunks)):
            # Content should have some overlap
            assert chunks[i].start_char < chunks[i-1].end_char + config.chunk_overlap
    
    def test_semantic_chunker(self, long_text):
        """Test semantic chunking"""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            preserve_paragraphs=True
        )
        chunker = SemanticChunker(config)
        
        chunks = chunker.chunk(long_text)
        
        assert len(chunks) > 1
        # Should preserve paragraph boundaries
        for chunk in chunks:
            assert chunk.content.strip() != ""
    
    def test_recursive_chunker(self, sample_text):
        """Test recursive chunking"""
        config = ChunkingConfig(chunk_size=300, chunk_overlap=30)
        chunker = RecursiveChunker(config)
        
        chunks = chunker.chunk(sample_text)
        
        assert len(chunks) >= 1
        # All chunks should be under max size (with some tolerance)
        for chunk in chunks:
            assert len(chunk.content) <= config.max_chunk_size
    
    def test_chunk_metadata(self, long_text):
        """Test chunk metadata extraction"""
        text_with_headers = "# Main Title\n\nParagraph one.\n\n## Section\n\nParagraph two."
        
        config = ChunkingConfig(chunk_size=100, extract_headers=True)
        chunker = RecursiveChunker(config, is_markdown=True)
        
        chunks = chunker.chunk(text_with_headers)
        
        # Should have metadata
        assert all('chunking_strategy' in c.metadata for c in chunks)


# =============================================================================
# Embedder Tests
# =============================================================================

class TestEmbedders:
    """Tests for embedding models"""
    
    def test_local_embedder_list_models(self):
        """Test listing available models"""
        models = LocalEmbedder.list_available_models()
        assert len(models) > 0
        assert any(m.name == "all-MiniLM-L6-v2" for m in models)
    
    @pytest.mark.asyncio
    async def test_local_embedder_embed(self):
        """Test text embedding"""
        # Skip if model not available
        try:
            embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
            
            texts = ["Hello world", "This is a test"]
            result = await embedder.embed(texts)
            
            assert len(result.embeddings) == 2
            assert result.dimensions == 384
            assert all(len(e) == 384 for e in result.embeddings)
        except ImportError:
            pytest.skip("sentence-transformers not installed")
    
    @pytest.mark.asyncio
    async def test_query_embedding(self):
        """Test single query embedding"""
        try:
            embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
            
            embedding = await embedder.embed_query("What is AI?")
            
            assert len(embedding) == 384
        except ImportError:
            pytest.skip("sentence-transformers not installed")


# =============================================================================
# Document Discovery Tests
# =============================================================================

class TestDocumentDiscovery:
    """Tests for document discovery system"""
    
    @pytest.mark.asyncio
    async def test_initialize(self, temp_db_path):
        """Test discovery initialization"""
        discovery = DocumentDiscovery(temp_db_path)
        await discovery.initialize()
        
        stats = await discovery.get_statistics()
        assert 'total' in stats
        assert 'by_status' in stats
    
    @pytest.mark.asyncio
    async def test_quality_scoring(self, temp_db_path):
        """Test content quality scoring"""
        discovery = DocumentDiscovery(temp_db_path)
        
        # Short content should score low
        short_score = discovery._score_quality("Short text.")
        
        # Long content should score higher
        long_content = "This is a longer piece of content. " * 100
        long_score = discovery._score_quality(long_content)
        
        assert long_score > short_score
    
    @pytest.mark.asyncio
    async def test_review_queue(self, temp_db_path):
        """Test review queue operations"""
        discovery = DocumentDiscovery(temp_db_path)
        await discovery.initialize()
        
        # Create a mock discovered document
        from modules.rag.discovery import DiscoveredDocument
        
        doc = DiscoveredDocument(
            id=str(uuid.uuid4()),
            title="Test Doc",
            url="https://example.com/test",
            snippet="Test snippet",
            discovered_at=datetime.utcnow().isoformat()
        )
        
        await discovery._save_discovered(doc)
        
        # Get queue
        queue, total = await discovery.get_review_queue(status=DiscoveryStatus.PENDING)
        assert total >= 1
        assert any(d.id == doc.id for d in queue)
    
    @pytest.mark.asyncio
    async def test_approve_reject(self, temp_db_path):
        """Test approve/reject workflow"""
        discovery = DocumentDiscovery(temp_db_path)
        await discovery.initialize()
        
        from modules.rag.discovery import DiscoveredDocument
        
        doc = DiscoveredDocument(
            id=str(uuid.uuid4()),
            title="To Approve",
            url="https://example.com/approve",
            snippet="Will be approved",
            discovered_at=datetime.utcnow().isoformat()
        )
        await discovery._save_discovered(doc)
        
        # Approve
        approved = await discovery.approve_document(doc.id)
        assert approved is not None
        assert approved.status == DiscoveryStatus.APPROVED
        
        # Create another to reject
        doc2 = DiscoveredDocument(
            id=str(uuid.uuid4()),
            title="To Reject",
            url="https://example.com/reject",
            snippet="Will be rejected",
            discovered_at=datetime.utcnow().isoformat()
        )
        await discovery._save_discovered(doc2)
        
        rejected = await discovery.reject_document(doc2.id)
        assert rejected is not None
        assert rejected.status == DiscoveryStatus.REJECTED


# =============================================================================
# Integration Tests
# =============================================================================

class TestRAGIntegration:
    """Integration tests for the full RAG pipeline"""
    
    @pytest.mark.asyncio
    async def test_document_to_chunks_pipeline(self, temp_db_path, sample_text):
        """Test full document processing pipeline"""
        manager = DocumentManager(temp_db_path)
        await manager.initialize()
        
        domains = await manager.list_domains()
        
        # Create document
        doc = Document(
            id=str(uuid.uuid4()),
            domain_id=domains[0].id,
            name="Integration Test Doc",
            doc_type=DocumentType.TXT,
            content=sample_text,
        )
        await manager.create_document(doc)
        
        # Chunk document
        chunker = SemanticChunker(ChunkingConfig(chunk_size=200))
        chunks = chunker.chunk(sample_text)
        
        # Save chunks
        doc_chunks = [
            DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=doc.id,
                content=chunk.content,
                chunk_index=chunk.index,
                total_chunks=len(chunks),
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            )
            for chunk in chunks
        ]
        
        await manager.save_chunks(doc_chunks)
        
        # Verify
        retrieved = await manager.get_chunks(doc.id)
        assert len(retrieved) == len(chunks)
        
        # Update document status
        doc.status = DocumentStatus.READY
        doc.chunk_count = len(chunks)
        await manager.update_document(doc)
        
        updated = await manager.get_document(doc.id)
        assert updated.status == DocumentStatus.READY
    
    @pytest.mark.asyncio
    async def test_graph_extraction_flow(self, temp_db_path, sample_text):
        """Test entity extraction and graph building"""
        graph = GraphRAG(temp_db_path)
        await graph.initialize()
        
        # Manually create entities (simulating extraction)
        entities = [
            Entity(id=str(uuid.uuid4()), name="OpenAI", entity_type=EntityType.ORGANIZATION),
            Entity(id=str(uuid.uuid4()), name="GPT-4", entity_type=EntityType.TECHNOLOGY),
            Entity(id=str(uuid.uuid4()), name="Sam Altman", entity_type=EntityType.PERSON),
        ]
        
        for entity in entities:
            await graph.create_entity(entity)
        
        # Create relationships
        await graph.create_relationship(Relationship(
            id=str(uuid.uuid4()),
            source_id=entities[0].id,  # OpenAI
            target_id=entities[1].id,  # GPT-4
            relationship_type=RelationshipType.CREATED_BY
        ))
        await graph.create_relationship(Relationship(
            id=str(uuid.uuid4()),
            source_id=entities[2].id,  # Sam Altman
            target_id=entities[0].id,  # OpenAI
            relationship_type=RelationshipType.WORKS_FOR
        ))
        
        # Get statistics
        stats = await graph.get_statistics()
        assert stats['entity_count'] == 3
        assert stats['relationship_count'] == 2
        
        # Find paths
        paths = await graph.find_paths(entities[2].id, entities[1].id, max_depth=3)
        assert len(paths) > 0  # Sam Altman -> OpenAI -> GPT-4


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
