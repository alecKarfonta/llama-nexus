"""
Semantic Chunker

Splits text based on semantic boundaries (sentences, paragraphs).
Better preserves meaning compared to fixed-size chunking.
"""

from typing import List, Optional, Dict, Any
import re
from .base import Chunker, Chunk, ChunkingConfig


class SemanticChunker(Chunker):
    """
    Semantic-aware text chunking.
    
    Features:
    - Sentence and paragraph boundary detection
    - Combines small sentences into larger chunks
    - Respects semantic units
    - Better for natural language text
    """
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text at semantic boundaries"""
        text = self._clean_text(text)
        if not text:
            return []
        
        chunks = []
        headers = self._extract_headers(text) if self.config.extract_headers else []
        
        # First split into paragraphs
        if self.config.preserve_paragraphs:
            units = self._split_paragraphs(text)
        else:
            units = self._split_sentences(text)
        
        current_chunk = []
        current_length = 0
        current_start = 0
        position = 0
        index = 0
        
        for unit in units:
            unit_length = len(unit)
            
            # Check if adding this unit would exceed chunk size
            if current_length + unit_length > self.config.chunk_size and current_chunk:
                # Save current chunk
                chunk_content = ' '.join(current_chunk)
                
                if len(chunk_content) >= self.config.min_chunk_size:
                    chunk_metadata = {
                        **(metadata or {}),
                        'chunking_strategy': 'semantic',
                        'unit_type': 'paragraph' if self.config.preserve_paragraphs else 'sentence'
                    }
                    
                    if self.config.extract_headers:
                        header = self._find_nearest_header(current_start, headers)
                        if header:
                            chunk_metadata['section_header'] = header
                    
                    chunks.append(Chunk(
                        content=chunk_content,
                        index=index,
                        start_char=current_start,
                        end_char=position,
                        metadata=chunk_metadata
                    ))
                    index += 1
                
                # Start new chunk with overlap
                if self.config.chunk_overlap > 0 and current_chunk:
                    # Keep some units for overlap
                    overlap_content = current_chunk[-1] if len(current_chunk) > 0 else ""
                    current_chunk = [overlap_content] if overlap_content else []
                    current_length = len(overlap_content)
                    current_start = position - len(overlap_content)
                else:
                    current_chunk = []
                    current_length = 0
                    current_start = position
            
            # Handle units larger than max chunk size
            if unit_length > self.config.max_chunk_size:
                # Split large unit into smaller pieces
                sub_chunks = self._split_large_unit(unit, position, index, metadata, headers)
                chunks.extend(sub_chunks)
                index += len(sub_chunks)
                position += unit_length
                continue
            
            current_chunk.append(unit)
            current_length += unit_length + 1  # +1 for space
            position += unit_length + 2  # +2 for paragraph break
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            if len(chunk_content) >= self.config.min_chunk_size:
                chunk_metadata = {
                    **(metadata or {}),
                    'chunking_strategy': 'semantic',
                    'unit_type': 'paragraph' if self.config.preserve_paragraphs else 'sentence'
                }
                
                if self.config.extract_headers:
                    header = self._find_nearest_header(current_start, headers)
                    if header:
                        chunk_metadata['section_header'] = header
                
                chunks.append(Chunk(
                    content=chunk_content,
                    index=index,
                    start_char=current_start,
                    end_char=len(text),
                    metadata=chunk_metadata
                ))
        
        # Update total chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.metadata['total_chunks'] = total
        
        return chunks
    
    def _split_large_unit(
        self,
        unit: str,
        start_position: int,
        start_index: int,
        metadata: Optional[Dict[str, Any]],
        headers: List[Dict[str, Any]]
    ) -> List[Chunk]:
        """Split a unit that's too large into smaller chunks"""
        chunks = []
        
        # Fall back to sentence splitting
        sentences = self._split_sentences(unit)
        
        current_chunk = []
        current_length = 0
        current_start = start_position
        position = start_position
        index = start_index
        
        for sentence in sentences:
            if current_length + len(sentence) > self.config.chunk_size and current_chunk:
                chunk_content = ' '.join(current_chunk)
                
                chunk_metadata = {
                    **(metadata or {}),
                    'chunking_strategy': 'semantic_split',
                }
                
                if self.config.extract_headers:
                    header = self._find_nearest_header(current_start, headers)
                    if header:
                        chunk_metadata['section_header'] = header
                
                chunks.append(Chunk(
                    content=chunk_content,
                    index=index,
                    start_char=current_start,
                    end_char=position,
                    metadata=chunk_metadata
                ))
                index += 1
                
                current_chunk = []
                current_length = 0
                current_start = position
            
            current_chunk.append(sentence)
            current_length += len(sentence) + 1
            position += len(sentence) + 1
        
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunk_metadata = {
                **(metadata or {}),
                'chunking_strategy': 'semantic_split',
            }
            
            chunks.append(Chunk(
                content=chunk_content,
                index=index,
                start_char=current_start,
                end_char=position,
                metadata=chunk_metadata
            ))
        
        return chunks
