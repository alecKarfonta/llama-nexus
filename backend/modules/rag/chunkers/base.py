"""
Base Chunker Interface

Provides abstract interface for text chunking strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re


@dataclass
class ChunkingConfig:
    """Configuration for chunking"""
    chunk_size: int = 512  # Target chunk size in tokens/characters
    chunk_overlap: int = 50  # Overlap between chunks
    min_chunk_size: int = 5  # Minimum chunk size
    max_chunk_size: int = 2048  # Maximum chunk size
    
    # Strategy-specific options
    use_token_count: bool = False  # Use token count vs character count
    preserve_sentences: bool = True  # Try to keep sentences together
    preserve_paragraphs: bool = True  # Try to keep paragraphs together
    
    # Metadata extraction
    extract_headers: bool = True  # Extract section headers
    include_metadata: bool = True  # Include chunk metadata


@dataclass
class Chunk:
    """A text chunk with metadata"""
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def length(self) -> int:
        return len(self.content)


class Chunker(ABC):
    """
    Abstract base class for text chunking strategies.
    
    Implementations:
    - FixedChunker: Fixed-size chunks with overlap
    - SemanticChunker: Sentence/paragraph-aware chunking
    - RecursiveChunker: Hierarchical recursive chunking
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
    
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of Chunk objects
        """
        pass
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters except newlines
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text.strip()
    
    def _extract_headers(self, text: str) -> List[Dict[str, Any]]:
        """Extract headers/sections from text"""
        headers = []
        
        # Markdown headers
        for match in re.finditer(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE):
            headers.append({
                'level': len(match.group(1)),
                'text': match.group(2).strip(),
                'position': match.start()
            })
        
        return headers
    
    def _find_nearest_header(
        self,
        position: int,
        headers: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Find the nearest header before a position"""
        nearest = None
        for header in headers:
            if header['position'] <= position:
                nearest = header['text']
            else:
                break
        return nearest
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimation (4 chars per token on average)"""
        return len(text) // 4
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
