"""
Fixed-Size Chunker

Splits text into fixed-size chunks with configurable overlap.
Simple but effective for uniform document processing.
"""

from typing import List, Optional, Dict, Any
from .base import Chunker, Chunk, ChunkingConfig


class FixedChunker(Chunker):
    """
    Fixed-size text chunking with overlap.
    
    Features:
    - Configurable chunk size and overlap
    - Character or token-based sizing
    - Sentence boundary awareness (optional)
    """
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into fixed-size chunks"""
        text = self._clean_text(text)
        if not text:
            return []
        
        chunks = []
        headers = self._extract_headers(text) if self.config.extract_headers else []
        
        # Determine step size
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        step = chunk_size - overlap
        
        position = 0
        index = 0
        
        while position < len(text):
            # Calculate end position
            end = min(position + chunk_size, len(text))
            
            # Try to find a good break point
            if self.config.preserve_sentences and end < len(text):
                # Look for sentence boundary
                chunk_text = text[position:end]
                
                # Find last sentence boundary
                for boundary in ['. ', '! ', '? ', '\n']:
                    last_boundary = chunk_text.rfind(boundary)
                    if last_boundary > chunk_size // 2:  # Don't cut too short
                        end = position + last_boundary + len(boundary)
                        break
            
            # Extract chunk
            chunk_content = text[position:end].strip()
            
            if len(chunk_content) >= self.config.min_chunk_size:
                chunk_metadata = {
                    **(metadata or {}),
                    'chunking_strategy': 'fixed',
                    'chunk_size_target': chunk_size,
                    'overlap': overlap
                }
                
                if self.config.extract_headers:
                    header = self._find_nearest_header(position, headers)
                    if header:
                        chunk_metadata['section_header'] = header
                
                chunks.append(Chunk(
                    content=chunk_content,
                    index=index,
                    start_char=position,
                    end_char=end,
                    metadata=chunk_metadata
                ))
                index += 1
            
            position += step
            
            # Prevent infinite loop
            if step <= 0:
                break
        
        # Update total_chunks in metadata
        total = len(chunks)
        for chunk in chunks:
            chunk.metadata['total_chunks'] = total
        
        return chunks
