from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from datetime import datetime

@dataclass
class ContextualChunk:
    """Represents a chunk with its contextual enhancement."""
    original_text: str
    enhanced_text: str
    context_type: str
    enhancement_metadata: Dict[str, Any]

class ContextualEnhancer:
    """Adds explanatory context to chunks before embedding for improved retrieval."""
    
    def __init__(self, context_window_size: int = 200):
        """Initialize the contextual enhancer."""
        self.context_window_size = context_window_size
        self.document_cache: Dict[str, List[Dict[str, Any]]] = {}
        
    def enhance_chunks_for_embedding(self, chunks: List[Dict[str, Any]], 
                                   document_metadata: Optional[Dict[str, Any]] = None) -> List[ContextualChunk]:
        """
        Enhance chunks with contextual information before embedding.
        
        Args:
            chunks: List of document chunks to enhance
            document_metadata: Optional document-level metadata
            
        Returns:
            List of contextually enhanced chunks
        """
        if not chunks:
            return []
            
        # Group chunks by document for context analysis
        document_groups = self._group_chunks_by_document(chunks)
        
        enhanced_chunks = []
        
        for doc_name, doc_chunks in document_groups.items():
            # Sort chunks by index for proper ordering
            sorted_chunks = sorted(doc_chunks, key=lambda x: x.get('chunk_index', 0))
            
            # Extract document-level context
            doc_context = self._extract_document_context(sorted_chunks, document_metadata)
            
            # Enhance each chunk with context
            for i, chunk in enumerate(sorted_chunks):
                enhanced_chunk = self._enhance_single_chunk(
                    chunk, sorted_chunks, i, doc_context
                )
                enhanced_chunks.append(enhanced_chunk)
                
        return enhanced_chunks
    
    def _group_chunks_by_document(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group chunks by their source document."""
        document_groups = {}
        
        for chunk in chunks:
            doc_name = chunk.get('source_file', 'unknown')
            if doc_name not in document_groups:
                document_groups[doc_name] = []
            document_groups[doc_name].append(chunk)
            
        return document_groups
    
    def _extract_document_context(self, chunks: List[Dict[str, Any]], 
                                 document_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract document-level contextual information."""
        if not chunks:
            return {}
            
        # Analyze document structure and content
        doc_context = {
            'total_chunks': len(chunks),
            'document_type': self._classify_document_type(chunks),
            'main_topics': self._extract_main_topics(chunks),
            'document_summary': self._generate_document_summary(chunks),
            'entities': self._extract_document_entities(chunks),
            'technical_level': self._assess_technical_level(chunks)
        }
        
        # Add metadata if available
        if document_metadata:
            doc_context.update(document_metadata)
            
        return doc_context
    
    def _enhance_single_chunk(self, chunk: Dict[str, Any], all_chunks: List[Dict[str, Any]], 
                             chunk_index: int, doc_context: Dict[str, Any]) -> ContextualChunk:
        """Enhance a single chunk with contextual information."""
        original_text = chunk.get('text', '')
        
        # Build contextual enhancement
        context_parts = []
        
        # 1. Document context
        document_context = self._build_document_context(chunk, doc_context)
        if document_context:
            context_parts.append(document_context)
        
        # 2. Preceding context
        preceding_context = self._build_preceding_context(chunk, all_chunks, chunk_index)
        if preceding_context:
            context_parts.append(preceding_context)
        
        # 3. Following context
        following_context = self._build_following_context(chunk, all_chunks, chunk_index)
        if following_context:
            context_parts.append(following_context)
        
        # 4. Chunk-specific context
        chunk_context = self._build_chunk_context(chunk, doc_context)
        if chunk_context:
            context_parts.append(chunk_context)
        
        # Combine contexts
        if context_parts:
            enhanced_text = f"{' '.join(context_parts)}\n\n{original_text}"
        else:
            enhanced_text = original_text
            
        return ContextualChunk(
            original_text=original_text,
            enhanced_text=enhanced_text,
            context_type=self._determine_context_type(chunk, doc_context),
            enhancement_metadata={
                'context_parts': len(context_parts),
                'enhancement_length': len(enhanced_text) - len(original_text),
                'document_type': doc_context.get('document_type', 'unknown'),
                'chunk_position': f"{chunk_index + 1}/{len(all_chunks)}"
            }
        )
    
    def _build_document_context(self, chunk: Dict[str, Any], doc_context: Dict[str, Any]) -> Optional[str]:
        """Build document-level context for the chunk."""
        context_parts = []
        
        # Document type and summary
        doc_type = doc_context.get('document_type', 'unknown')
        if doc_type != 'unknown':
            context_parts.append(f"Document type: {doc_type}")
        
        # Document summary
        doc_summary = doc_context.get('document_summary', '')
        if doc_summary:
            context_parts.append(f"Document summary: {doc_summary}")
        
        # Main topics
        main_topics = doc_context.get('main_topics', [])
        if main_topics:
            context_parts.append(f"Main topics: {', '.join(main_topics[:3])}")
        
        # Technical level
        tech_level = doc_context.get('technical_level', 'unknown')
        if tech_level != 'unknown':
            context_parts.append(f"Technical level: {tech_level}")
            
        return '; '.join(context_parts) if context_parts else None
    
    def _build_preceding_context(self, chunk: Dict[str, Any], all_chunks: List[Dict[str, Any]], 
                                chunk_index: int) -> Optional[str]:
        """Build context from preceding chunks."""
        if chunk_index == 0:
            return None
            
        # Get preceding chunk
        prev_chunk = all_chunks[chunk_index - 1]
        prev_text = prev_chunk.get('text', '')
        
        # Extract last part of previous chunk for context
        prev_sentences = re.split(r'[.!?]+', prev_text)
        if len(prev_sentences) > 1:
            # Get last meaningful sentence
            last_sentence = prev_sentences[-2].strip()
            if last_sentence and len(last_sentence) > 20:
                return f"Preceding context: {last_sentence}"
        
        return None
    
    def _build_following_context(self, chunk: Dict[str, Any], all_chunks: List[Dict[str, Any]], 
                                chunk_index: int) -> Optional[str]:
        """Build context from following chunks."""
        if chunk_index >= len(all_chunks) - 1:
            return None
            
        # Get following chunk
        next_chunk = all_chunks[chunk_index + 1]
        next_text = next_chunk.get('text', '')
        
        # Extract first part of next chunk for context
        next_sentences = re.split(r'[.!?]+', next_text)
        if next_sentences:
            # Get first meaningful sentence
            first_sentence = next_sentences[0].strip()
            if first_sentence and len(first_sentence) > 20:
                return f"Following context: {first_sentence}"
        
        return None
    
    def _build_chunk_context(self, chunk: Dict[str, Any], doc_context: Dict[str, Any]) -> Optional[str]:
        """Build chunk-specific contextual information."""
        context_parts = []
        
        # Chunk metadata
        metadata = chunk.get('metadata', {})
        
        # Section information
        section_header = chunk.get('section_header')
        if section_header:
            context_parts.append(f"Section: {section_header}")
        
        # Content type
        content_type = metadata.get('content_type', 'unknown')
        if content_type != 'unknown':
            context_parts.append(f"Content type: {content_type}")
        
        # Entities in chunk
        chunk_entities = self._extract_chunk_entities(chunk.get('text', ''))
        if chunk_entities:
            context_parts.append(f"Key entities: {', '.join(chunk_entities[:3])}")
        
        return '; '.join(context_parts) if context_parts else None
    
    def _classify_document_type(self, chunks: List[Dict[str, Any]]) -> str:
        """Classify the document type based on content analysis."""
        if not chunks:
            return 'unknown'
            
        # Combine text from all chunks for analysis
        combined_text = ' '.join([chunk.get('text', '') for chunk in chunks[:5]])  # First 5 chunks
        text_lower = combined_text.lower()
        
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
    
    def _generate_document_summary(self, chunks: List[Dict[str, Any]]) -> str:
        """Generate a brief document summary."""
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
    
    def _extract_document_entities(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract key entities from document."""
        entities = set()
        
        for chunk in chunks[:3]:  # First 3 chunks
            text = chunk.get('text', '')
            # Simple entity extraction - capitalized words
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            
            for word in words:
                if len(word) > 3 and word not in ['The', 'This', 'That', 'There']:
                    entities.add(word)
        
        return list(entities)[:10]  # Top 10 entities
    
    def _assess_technical_level(self, chunks: List[Dict[str, Any]]) -> str:
        """Assess the technical complexity level of the document."""
        if not chunks:
            return 'unknown'
            
        # Combine text from chunks
        combined_text = ' '.join([chunk.get('text', '') for chunk in chunks[:3]])
        text_lower = combined_text.lower()
        
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
    
    def _extract_chunk_entities(self, text: str) -> List[str]:
        """Extract entities from a single chunk."""
        entities = set()
        
        # Extract capitalized words/phrases
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        for word in words:
            if len(word) > 3 and word not in ['The', 'This', 'That', 'There']:
                entities.add(word)
        
        return list(entities)[:5]  # Top 5 entities per chunk
    
    def _determine_context_type(self, chunk: Dict[str, Any], doc_context: Dict[str, Any]) -> str:
        """Determine the type of contextual enhancement applied."""
        text = chunk.get('text', '')
        
        # Analyze chunk characteristics
        if 'procedure' in text.lower() or 'step' in text.lower():
            return 'procedural'
        elif 'definition' in text.lower() or 'meaning' in text.lower():
            return 'definitional'
        elif '?' in text or 'example' in text.lower():
            return 'explanatory'
        elif doc_context.get('document_type') == 'technical_manual':
            return 'technical'
        else:
            return 'general' 