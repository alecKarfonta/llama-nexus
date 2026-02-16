"""
Contextual Embeddings with Document-Specific Context

This module implements advanced contextual embeddings that incorporate
document-specific context to improve retrieval accuracy and relevance.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import hashlib
from datetime import datetime
import json

@dataclass
class DocumentContext:
    """Represents document-specific context information."""
    document_id: str
    document_type: str
    document_title: Optional[str] = None
    document_summary: Optional[str] = None
    main_topics: List[str] = None
    technical_level: str = "general"
    domain: Optional[str] = None
    language: str = "en"
    creation_date: Optional[str] = None
    author: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.main_topics is None:
            self.main_topics = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ChunkContext:
    """Represents chunk-specific context information."""
    chunk_id: str
    chunk_index: int
    section_header: Optional[str] = None
    content_type: str = "general"
    key_entities: List[str] = None
    preceding_context: Optional[str] = None
    following_context: Optional[str] = None
    chunk_position: str = "middle"  # "beginning", "middle", "ending"
    importance_score: float = 0.5
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.key_entities is None:
            self.key_entities = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ContextualEmbedding:
    """Represents a contextual embedding with metadata."""
    embedding: np.ndarray
    original_text: str
    enhanced_text: str
    document_context: DocumentContext
    chunk_context: ChunkContext
    embedding_metadata: Dict[str, Any]

class ContextualEmbedder:
    """Advanced embedder that generates contextual embeddings with document-specific context."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 context_window_size: int = 200,
                 max_context_length: int = 512):
        """
        Initialize the contextual embedder.
        
        Args:
            model_name: Sentence transformer model to use
            context_window_size: Size of context window for surrounding text
            max_context_length: Maximum length of enhanced text for embedding
        """
        self.model = SentenceTransformer(model_name)
        self.context_window_size = context_window_size
        self.max_context_length = max_context_length
        
        # Cache for embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        # Document context cache
        self._document_context_cache: Dict[str, DocumentContext] = {}
        
    def create_document_context(self, 
                               document_id: str,
                               chunks: List[Dict[str, Any]],
                               document_metadata: Optional[Dict[str, Any]] = None) -> DocumentContext:
        """
        Create document-specific context from chunks and metadata.
        
        Args:
            document_id: Unique identifier for the document
            chunks: List of document chunks
            document_metadata: Optional document-level metadata
            
        Returns:
            DocumentContext object with extracted information
        """
        if not chunks:
            return DocumentContext(
                document_id=document_id,
                document_type="unknown",
                metadata=document_metadata or {}
            )
        
        # Analyze document content
        combined_text = ' '.join([chunk.get('text', '') for chunk in chunks[:5]])
        
        # Extract document information
        doc_type = self._classify_document_type(combined_text)
        main_topics = self._extract_main_topics(chunks)
        technical_level = self._assess_technical_level(combined_text)
        domain = self._identify_domain(combined_text, main_topics)
        summary = self._generate_document_summary(chunks)
        
        # Create document context
        doc_context = DocumentContext(
            document_id=document_id,
            document_type=doc_type,
            document_title=document_metadata.get('title') if document_metadata else None,
            document_summary=summary,
            main_topics=main_topics,
            technical_level=technical_level,
            domain=domain,
            language=document_metadata.get('language', 'en') if document_metadata else 'en',
            creation_date=document_metadata.get('creation_date') if document_metadata else None,
            author=document_metadata.get('author') if document_metadata else None,
            metadata=document_metadata or {}
        )
        
        # Cache the context
        self._document_context_cache[document_id] = doc_context
        
        return doc_context
    
    def create_chunk_context(self,
                           chunk: Dict[str, Any],
                           chunk_index: int,
                           all_chunks: List[Dict[str, Any]],
                           document_context: DocumentContext) -> ChunkContext:
        """
        Create chunk-specific context information.
        
        Args:
            chunk: The chunk to create context for
            chunk_index: Index of the chunk in the document
            all_chunks: All chunks in the document
            document_context: Document-level context
            
        Returns:
            ChunkContext object with extracted information
        """
        chunk_text = chunk.get('text', '')
        
        # Determine chunk position
        total_chunks = len(all_chunks)
        if chunk_index == 0:
            position = "beginning"
        elif chunk_index == total_chunks - 1:
            position = "ending"
        else:
            position = "middle"
        
        # Extract chunk-specific information
        section_header = self._extract_section_header(chunk_text)
        content_type = self._classify_content_type(chunk_text, document_context)
        key_entities = self._extract_chunk_entities(chunk_text)
        importance_score = self._calculate_importance_score(chunk_text, chunk_index, total_chunks)
        
        # Get surrounding context
        preceding_context = self._get_preceding_context(chunk_index, all_chunks)
        following_context = self._get_following_context(chunk_index, all_chunks)
        
        return ChunkContext(
            chunk_id=chunk.get('chunk_id', f"chunk_{chunk_index}"),
            chunk_index=chunk_index,
            section_header=section_header,
            content_type=content_type,
            key_entities=key_entities,
            preceding_context=preceding_context,
            following_context=following_context,
            chunk_position=position,
            importance_score=importance_score,
            metadata=chunk.get('metadata', {})
        )
    
    def generate_contextual_embedding(self,
                                    chunk: Dict[str, Any],
                                    document_context: DocumentContext,
                                    chunk_context: ChunkContext) -> ContextualEmbedding:
        """
        Generate a contextual embedding with document-specific context.
        
        Args:
            chunk: The chunk to embed
            document_context: Document-level context
            chunk_context: Chunk-specific context
            
        Returns:
            ContextualEmbedding object with enhanced embedding
        """
        original_text = chunk.get('text', '')
        
        # Create enhanced text with context
        enhanced_text = self._create_enhanced_text(
            original_text, document_context, chunk_context
        )
        
        # Truncate if necessary
        if len(enhanced_text) > self.max_context_length:
            enhanced_text = enhanced_text[:self.max_context_length]
        
        # Generate embedding
        embedding = self.model.encode(enhanced_text)
        
        # Create embedding metadata
        embedding_metadata = {
            'enhancement_length': len(enhanced_text) - len(original_text),
            'context_parts': self._count_context_parts(document_context, chunk_context),
            'embedding_model': self.model.get_sentence_embedding_dimension(),
            'generated_at': datetime.now().isoformat(),
            'cache_key': self._create_cache_key(original_text, document_context, chunk_context)
        }
        
        return ContextualEmbedding(
            embedding=embedding,
            original_text=original_text,
            enhanced_text=enhanced_text,
            document_context=document_context,
            chunk_context=chunk_context,
            embedding_metadata=embedding_metadata
        )
    
    def embed_document_chunks(self,
                            chunks: List[Dict[str, Any]],
                            document_id: str,
                            document_metadata: Optional[Dict[str, Any]] = None) -> List[ContextualEmbedding]:
        """
        Generate contextual embeddings for all chunks in a document.
        
        Args:
            chunks: List of document chunks
            document_id: Unique identifier for the document
            document_metadata: Optional document-level metadata
            
        Returns:
            List of ContextualEmbedding objects
        """
        if not chunks:
            return []
        
        # Create document context
        document_context = self.create_document_context(
            document_id, chunks, document_metadata
        )
        
        # Sort chunks by index for proper ordering
        sorted_chunks = sorted(chunks, key=lambda x: x.get('chunk_index', 0))
        
        # Generate embeddings for each chunk
        contextual_embeddings = []
        
        for i, chunk in enumerate(sorted_chunks):
            # Create chunk context
            chunk_context = self.create_chunk_context(
                chunk, i, sorted_chunks, document_context
            )
            
            # Generate contextual embedding
            contextual_embedding = self.generate_contextual_embedding(
                chunk, document_context, chunk_context
            )
            
            contextual_embeddings.append(contextual_embedding)
        
        return contextual_embeddings
    
    def _create_enhanced_text(self,
                             original_text: str,
                             document_context: DocumentContext,
                             chunk_context: ChunkContext) -> str:
        """
        Create enhanced text with document-specific context.
        
        Args:
            original_text: Original chunk text
            document_context: Document-level context
            chunk_context: Chunk-specific context
            
        Returns:
            Enhanced text with context
        """
        context_parts = []
        
        # 1. Document-level context
        doc_context = self._build_document_context_string(document_context)
        if doc_context:
            context_parts.append(doc_context)
        
        # 2. Chunk position and type context
        chunk_context_str = self._build_chunk_context_string(chunk_context)
        if chunk_context_str:
            context_parts.append(chunk_context_str)
        
        # 3. Surrounding context
        surrounding_context = self._build_surrounding_context(chunk_context)
        if surrounding_context:
            context_parts.append(surrounding_context)
        
        # 4. Domain-specific context
        domain_context = self._build_domain_context(document_context, chunk_context)
        if domain_context:
            context_parts.append(domain_context)
        
        # Combine contexts with original text
        if context_parts:
            enhanced_text = f"{' '.join(context_parts)}\n\n{original_text}"
        else:
            enhanced_text = original_text
        
        return enhanced_text
    
    def _build_document_context_string(self, doc_context: DocumentContext) -> Optional[str]:
        """Build document-level context string."""
        parts = []
        
        # Document type and domain
        if doc_context.document_type != "unknown":
            parts.append(f"Document type: {doc_context.document_type}")
        
        if doc_context.domain:
            parts.append(f"Domain: {doc_context.domain}")
        
        # Technical level
        if doc_context.technical_level != "general":
            parts.append(f"Technical level: {doc_context.technical_level}")
        
        # Main topics
        if doc_context.main_topics:
            topics_str = ', '.join(doc_context.main_topics[:3])
            parts.append(f"Main topics: {topics_str}")
        
        # Document summary
        if doc_context.document_summary:
            summary = doc_context.document_summary[:100] + "..." if len(doc_context.document_summary) > 100 else doc_context.document_summary
            parts.append(f"Document summary: {summary}")
        
        return '; '.join(parts) if parts else None
    
    def _build_chunk_context_string(self, chunk_context: ChunkContext) -> Optional[str]:
        """Build chunk-specific context string."""
        parts = []
        
        # Section header
        if chunk_context.section_header:
            parts.append(f"Section: {chunk_context.section_header}")
        
        # Content type
        if chunk_context.content_type != "general":
            parts.append(f"Content type: {chunk_context.content_type}")
        
        # Chunk position
        if chunk_context.chunk_position != "middle":
            parts.append(f"Position: {chunk_context.chunk_position}")
        
        # Key entities
        if chunk_context.key_entities:
            entities_str = ', '.join(chunk_context.key_entities[:3])
            parts.append(f"Key entities: {entities_str}")
        
        return '; '.join(parts) if parts else None
    
    def _build_surrounding_context(self, chunk_context: ChunkContext) -> Optional[str]:
        """Build surrounding context string."""
        parts = []
        
        # Preceding context
        if chunk_context.preceding_context:
            parts.append(f"Preceding: {chunk_context.preceding_context}")
        
        # Following context
        if chunk_context.following_context:
            parts.append(f"Following: {chunk_context.following_context}")
        
        return '; '.join(parts) if parts else None
    
    def _build_domain_context(self, doc_context: DocumentContext, chunk_context: ChunkContext) -> Optional[str]:
        """Build domain-specific context string."""
        if not doc_context.domain:
            return None
        
        # Domain-specific enhancements
        domain_enhancements = {
            "technical": "Technical documentation with procedures and specifications",
            "literature": "Literary work with narrative and character development",
            "academic": "Academic paper with research methodology and findings",
            "legal": "Legal document with regulations and compliance requirements",
            "medical": "Medical documentation with clinical procedures and terminology"
        }
        
        return domain_enhancements.get(doc_context.domain.lower())
    
    def _classify_document_type(self, text: str) -> str:
        """Classify document type based on content."""
        text_lower = text.lower()
        
        # Classification patterns
        patterns = {
            'technical_manual': [
                'procedure', 'installation', 'configuration', 'specification',
                'technical', 'manual', 'guide', 'instruction', 'maintenance'
            ],
            'research_paper': [
                'abstract', 'introduction', 'methodology', 'results', 'conclusion',
                'research', 'study', 'analysis', 'findings', 'hypothesis'
            ],
            'narrative': [
                'story', 'character', 'plot', 'chapter', 'narrative',
                'once upon', 'beginning', 'ending', 'dialogue'
            ],
            'reference': [
                'definition', 'glossary', 'index', 'reference', 'dictionary',
                'encyclopedia', 'lookup', 'terminology'
            ]
        }
        
        # Score each type
        type_scores = {}
        for doc_type, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            type_scores[doc_type] = score
        
        # Return highest scoring type
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return 'general'
    
    def _extract_main_topics(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract main topics from document chunks."""
        if not chunks:
            return []
        
        # Simple topic extraction using keyword frequency
        word_freq = {}
        
        for chunk in chunks[:3]:  # First 3 chunks usually contain main topics
            text = chunk.get('text', '')
            words = re.findall(r'\b[A-Z][a-z]+\b', text)  # Capitalized words
            
            for word in words:
                if len(word) > 3:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top topics
        sorted_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [topic[0] for topic in sorted_topics[:5]]
    
    def _assess_technical_level(self, text: str) -> str:
        """Assess technical complexity level."""
        text_lower = text.lower()
        
        # Technical indicators
        high_tech_indicators = [
            'algorithm', 'implementation', 'architecture', 'optimization',
            'configuration', 'specification', 'parameter', 'protocol'
        ]
        
        medium_tech_indicators = [
            'system', 'process', 'method', 'procedure', 'function',
            'component', 'interface', 'operation', 'maintenance'
        ]
        
        basic_tech_indicators = [
            'guide', 'instruction', 'step', 'overview', 'introduction',
            'basic', 'simple', 'easy', 'beginner'
        ]
        
        # Count indicators
        high_count = sum(1 for indicator in high_tech_indicators if indicator in text_lower)
        medium_count = sum(1 for indicator in medium_tech_indicators if indicator in text_lower)
        basic_count = sum(1 for indicator in basic_tech_indicators if indicator in text_lower)
        
        # Determine level
        if high_count > medium_count and high_count > basic_count:
            return 'advanced'
        elif medium_count > basic_count:
            return 'intermediate'
        elif basic_count > 0:
            return 'basic'
        else:
            return 'general'
    
    def _identify_domain(self, text: str, main_topics: List[str]) -> Optional[str]:
        """Identify the domain of the document."""
        text_lower = text.lower()
        
        # Domain patterns
        domain_patterns = {
            'technical': ['technical', 'engineering', 'software', 'hardware', 'system'],
            'literature': ['story', 'novel', 'poem', 'literature', 'fiction', 'narrative'],
            'academic': ['research', 'study', 'analysis', 'methodology', 'hypothesis'],
            'legal': ['legal', 'law', 'regulation', 'compliance', 'contract'],
            'medical': ['medical', 'clinical', 'patient', 'diagnosis', 'treatment']
        }
        
        # Score domains
        domain_scores = {}
        for domain, keywords in domain_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        # Return highest scoring domain
        if domain_scores:
            max_score = max(domain_scores.values())
            if max_score > 0:
                return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _generate_document_summary(self, chunks: List[Dict[str, Any]]) -> str:
        """Generate document summary from chunks."""
        if not chunks:
            return ''
        
        # Use first chunk as summary base
        first_chunk = chunks[0].get('text', '')
        sentences = re.split(r'[.!?]+', first_chunk)
        
        if sentences:
            # Get first meaningful sentence
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 20:
                return first_sentence[:150] + '...' if len(first_sentence) > 150 else first_sentence
        
        return ''
    
    def _extract_section_header(self, text: str) -> Optional[str]:
        """Extract section header from chunk text."""
        # Look for common header patterns
        header_patterns = [
            r'^(#+)\s+(.+)$',  # Markdown headers
            r'^([A-Z][A-Z\s]+)$',  # ALL CAPS headers
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)$',  # Title Case headers
        ]
        
        lines = text.split('\n')
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            for pattern in header_patterns:
                match = re.match(pattern, line)
                if match:
                    return match.group(2) if len(match.groups()) > 1 else match.group(1)
        
        return None
    
    def _classify_content_type(self, text: str, doc_context: DocumentContext) -> str:
        """Classify the content type of a chunk."""
        text_lower = text.lower()
        
        # Content type patterns
        if 'procedure' in text_lower or 'step' in text_lower:
            return 'procedural'
        elif 'definition' in text_lower or 'meaning' in text_lower:
            return 'definitional'
        elif '?' in text or 'example' in text_lower:
            return 'explanatory'
        elif doc_context.document_type == 'technical_manual':
            return 'technical'
        elif 'character' in text_lower or 'dialogue' in text_lower:
            return 'narrative'
        else:
            return 'general'
    
    def _extract_chunk_entities(self, text: str) -> List[str]:
        """Extract key entities from chunk text."""
        entities = set()
        
        # Extract capitalized words/phrases
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        for word in words:
            if len(word) > 3 and word not in ['The', 'This', 'That', 'There']:
                entities.add(word)
        
        return list(entities)[:5]  # Top 5 entities
    
    def _calculate_importance_score(self, text: str, chunk_index: int, total_chunks: int) -> float:
        """Calculate importance score for a chunk."""
        # Base score
        score = 0.5
        
        # Position-based scoring (beginning and ending chunks are more important)
        if chunk_index == 0:
            score += 0.2  # Beginning
        elif chunk_index == total_chunks - 1:
            score += 0.2  # Ending
        
        # Content-based scoring
        text_lower = text.lower()
        
        # Important content indicators
        important_indicators = [
            'important', 'key', 'main', 'primary', 'essential',
            'conclusion', 'summary', 'overview', 'introduction'
        ]
        
        for indicator in important_indicators:
            if indicator in text_lower:
                score += 0.1
        
        # Length-based scoring (longer chunks might be more important)
        if len(text) > 200:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _get_preceding_context(self, chunk_index: int, all_chunks: List[Dict[str, Any]]) -> Optional[str]:
        """Get context from preceding chunk."""
        if chunk_index == 0:
            return None
        
        prev_chunk = all_chunks[chunk_index - 1]
        prev_text = prev_chunk.get('text', '')
        
        # Extract last sentence for context
        sentences = re.split(r'[.!?]+', prev_text)
        if len(sentences) > 1:
            last_sentence = sentences[-2].strip()
            if last_sentence and len(last_sentence) > 20:
                return last_sentence[:100] + "..." if len(last_sentence) > 100 else last_sentence
        
        return None
    
    def _get_following_context(self, chunk_index: int, all_chunks: List[Dict[str, Any]]) -> Optional[str]:
        """Get context from following chunk."""
        if chunk_index >= len(all_chunks) - 1:
            return None
        
        next_chunk = all_chunks[chunk_index + 1]
        next_text = next_chunk.get('text', '')
        
        # Extract first sentence for context
        sentences = re.split(r'[.!?]+', next_text)
        if sentences:
            first_sentence = sentences[0].strip()
            if first_sentence and len(first_sentence) > 20:
                return first_sentence[:100] + "..." if len(first_sentence) > 100 else first_sentence
        
        return None
    
    def _count_context_parts(self, doc_context: DocumentContext, chunk_context: ChunkContext) -> int:
        """Count the number of context parts used."""
        count = 0
        
        # Document context parts
        if doc_context.document_type != "unknown":
            count += 1
        if doc_context.domain:
            count += 1
        if doc_context.main_topics:
            count += 1
        if doc_context.document_summary:
            count += 1
        
        # Chunk context parts
        if chunk_context.section_header:
            count += 1
        if chunk_context.content_type != "general":
            count += 1
        if chunk_context.key_entities:
            count += 1
        if chunk_context.preceding_context:
            count += 1
        if chunk_context.following_context:
            count += 1
        
        return count
    
    def _create_cache_key(self, text: str, doc_context: DocumentContext, chunk_context: ChunkContext) -> str:
        """Create a cache key for the embedding."""
        content = f"{text[:100]}:{doc_context.document_id}:{chunk_context.chunk_id}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self._document_context_cache.clear()
    
    def get_cache_size(self) -> Tuple[int, int]:
        """Get the size of both caches."""
        return len(self._embedding_cache), len(self._document_context_cache) 