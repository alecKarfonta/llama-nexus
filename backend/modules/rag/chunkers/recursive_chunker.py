"""
Recursive Character Chunker

Hierarchically splits text using multiple separators.
Best for structured documents with headers, code, etc.
"""

from typing import List, Optional, Dict, Any
import re
from .base import Chunker, Chunk, ChunkingConfig


class RecursiveChunker(Chunker):
    """
    Recursive text chunking with hierarchical separators.
    
    Features:
    - Multiple separator levels (headers -> paragraphs -> sentences -> words)
    - Markdown-aware chunking
    - Code-aware chunking
    - Best for preserving document structure
    """
    
    # Default separators in order of priority
    DEFAULT_SEPARATORS = [
        "\n\n\n",      # Multiple blank lines (major sections)
        "\n\n",        # Paragraph breaks
        "\n",          # Line breaks
        ". ",          # Sentences
        ", ",          # Clauses
        " ",           # Words
        ""             # Characters (fallback)
    ]
    
    # Markdown-specific separators
    MARKDOWN_SEPARATORS = [
        "\n# ",        # H1
        "\n## ",       # H2
        "\n### ",      # H3
        "\n#### ",     # H4
        "\n##### ",    # H5
        "\n###### ",   # H6
        "\n```",       # Code blocks
        "\n\n",        # Paragraphs
        "\n",          # Lines
        ". ",          # Sentences
        " ",           # Words
    ]
    
    # Code-specific separators
    CODE_SEPARATORS = [
        "\nclass ",    # Class definitions
        "\ndef ",      # Function definitions (Python)
        "\nfunction ", # Function definitions (JS)
        "\n\n",        # Blank lines
        "\n",          # Lines
        "; ",          # Statements
        " ",           # Tokens
    ]
    
    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        separators: Optional[List[str]] = None,
        is_markdown: bool = False,
        is_code: bool = False
    ):
        super().__init__(config)
        
        if separators:
            self.separators = separators
        elif is_markdown:
            self.separators = self.MARKDOWN_SEPARATORS
        elif is_code:
            self.separators = self.CODE_SEPARATORS
        else:
            self.separators = self.DEFAULT_SEPARATORS
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Recursively split text using hierarchical separators"""
        text = self._clean_text(text)
        if not text:
            return []
        
        headers = self._extract_headers(text) if self.config.extract_headers else []
        
        # Recursively split
        raw_chunks = self._split_recursive(text, self.separators)
        
        # Convert to Chunk objects
        chunks = []
        position = 0
        
        for index, content in enumerate(raw_chunks):
            content = content.strip()
            if len(content) < self.config.min_chunk_size:
                continue
            
            # Find position in original text
            start = text.find(content, position)
            if start == -1:
                start = position
            end = start + len(content)
            
            chunk_metadata = {
                **(metadata or {}),
                'chunking_strategy': 'recursive',
                'separators_used': len(self.separators)
            }
            
            if self.config.extract_headers:
                header = self._find_nearest_header(start, headers)
                if header:
                    chunk_metadata['section_header'] = header
            
            chunks.append(Chunk(
                content=content,
                index=len(chunks),
                start_char=start,
                end_char=end,
                metadata=chunk_metadata
            ))
            
            position = end
        
        # Add overlap between chunks
        if self.config.chunk_overlap > 0:
            chunks = self._add_overlap(chunks, text)
        
        # Update total chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.metadata['total_chunks'] = total
        
        return chunks
    
    def _split_recursive(
        self,
        text: str,
        separators: List[str],
        depth: int = 0
    ) -> List[str]:
        """Recursively split text using separators"""
        if not text or not separators:
            return [text] if text else []
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Handle empty separator (character-level split)
        if separator == "":
            # Split into chunks of target size
            result = []
            for i in range(0, len(text), self.config.chunk_size):
                result.append(text[i:i + self.config.chunk_size])
            return result
        
        # Split by current separator
        splits = text.split(separator)
        
        # If no splits or only one part, try next separator
        if len(splits) == 1:
            if remaining_separators:
                return self._split_recursive(text, remaining_separators, depth + 1)
            return [text]
        
        # Process each split
        result = []
        current_chunk = ""
        
        for i, split in enumerate(splits):
            # Add separator back (except for first split)
            if i > 0 and separator.strip():
                potential_chunk = current_chunk + separator + split
            else:
                potential_chunk = current_chunk + split if current_chunk else split
            
            # Check if chunk is too large
            if len(potential_chunk) > self.config.chunk_size:
                # Save current chunk if it exists
                if current_chunk:
                    result.append(current_chunk)
                
                # Check if split itself is too large
                if len(split) > self.config.chunk_size:
                    # Recursively split the large piece
                    if remaining_separators:
                        sub_chunks = self._split_recursive(
                            split, remaining_separators, depth + 1
                        )
                        result.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        # Force split at max size
                        for j in range(0, len(split), self.config.chunk_size):
                            result.append(split[j:j + self.config.chunk_size])
                        current_chunk = ""
                else:
                    current_chunk = split
            else:
                current_chunk = potential_chunk
        
        # Don't forget last chunk
        if current_chunk:
            result.append(current_chunk)
        
        return result
    
    def _add_overlap(self, chunks: List[Chunk], original_text: str) -> List[Chunk]:
        """Add overlap between chunks"""
        if len(chunks) <= 1:
            return chunks
        
        overlap_size = self.config.chunk_overlap
        result = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - add suffix overlap from next chunk
                next_chunk = chunks[1]
                suffix = next_chunk.content[:overlap_size]
                new_content = chunk.content
                if suffix and not chunk.content.endswith(suffix):
                    new_content = chunk.content + " " + suffix
                    
                result.append(Chunk(
                    content=new_content,
                    index=chunk.index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata={**chunk.metadata, 'has_overlap': True}
                ))
            elif i == len(chunks) - 1:
                # Last chunk - add prefix overlap from previous chunk
                prev_chunk = chunks[i - 1]
                prefix = prev_chunk.content[-overlap_size:]
                new_content = chunk.content
                if prefix and not chunk.content.startswith(prefix):
                    new_content = prefix + " " + chunk.content
                    
                result.append(Chunk(
                    content=new_content,
                    index=chunk.index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata={**chunk.metadata, 'has_overlap': True}
                ))
            else:
                # Middle chunk - add both prefix and suffix
                prev_chunk = chunks[i - 1]
                next_chunk = chunks[i + 1]
                prefix = prev_chunk.content[-overlap_size//2:]
                suffix = next_chunk.content[:overlap_size//2]
                
                new_content = chunk.content
                if prefix and not chunk.content.startswith(prefix):
                    new_content = prefix + " " + new_content
                if suffix and not new_content.endswith(suffix):
                    new_content = new_content + " " + suffix
                    
                result.append(Chunk(
                    content=new_content,
                    index=chunk.index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata={**chunk.metadata, 'has_overlap': True}
                ))
        
        return result
