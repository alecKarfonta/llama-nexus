from typing import List, Dict, Any, Optional
from document_processor import DocumentProcessor, DocumentChunk, DocumentMetadata
from semantic_chunker import SemanticChunker
import spacy
import re
from datetime import datetime

class EnhancedDocumentProcessor:
    """Enhanced document processor with semantic chunking and metadata extraction."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.semantic_chunker = SemanticChunker()
        # Load spaCy model for NLP tasks - no fallback, let it fail if not installed
        self.nlp = spacy.load("en_core_web_sm")
    
    def process_document_enhanced(self, file_path: str, use_semantic_chunking: bool = True) -> List[DocumentChunk]:
        """Process document with enhanced features."""
        # Always use semantic chunking - no fallback to basic processing
        
        # Extract metadata
        metadata = self.document_processor.extract_metadata(file_path)
        
        # Get document content based on file type - let errors bubble up
        content = self._extract_document_content(file_path)
        
        if not content.strip():
            raise ValueError(f"No content extracted from document: {file_path}")
        
        # Determine content type for the whole document
        content_type = self._classify_content_type(content)
        
        # Apply semantic chunking to the entire document content
        semantic_chunks = self.semantic_chunker.create_adaptive_chunks(
            content, content_type
        )
        
        # Create enhanced chunks with semantic boundaries
        enhanced_chunks = []
        for i, semantic_chunk in enumerate(semantic_chunks):
            enhanced_chunk = DocumentChunk(
                text=semantic_chunk,
                chunk_id=f"{file_path}_semantic_{i}",
                source_file=file_path,
                page_number=None,  # We don't have page info for whole-document chunking
                section_header=None,
                chunk_index=i,
                metadata=self._enhance_metadata(metadata.__dict__ if metadata else {}, semantic_chunk)
            )
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def _extract_document_content(self, file_path: str) -> str:
        """Extract text content from document based on file type."""
        import os
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            # For PDFs, extract text using PyMuPDF - no fallback
            import fitz
            doc = fitz.open(file_path)
            content = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                content += page.get_text() + "\n"
            doc.close()
            return content
        
        elif file_ext in ['.docx']:
            # For DOCX, extract text using python-docx - no fallback
            from docx import Document as DocxDocument
            doc = DocxDocument(file_path)
            content = ""
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            return content
        
        elif file_ext in ['.html', '.htm']:
            # For HTML, extract text using BeautifulSoup - no fallback
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                return soup.get_text(separator='\n', strip=True)
        
        else:
            # For text-based files (txt, md, etc.) - no fallback
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def extract_enhanced_metadata(self, file_path: str) -> DocumentMetadata:
        """Extract enhanced metadata including content analysis."""
        metadata = self.document_processor.extract_metadata(file_path)
        
        # Read file content for analysis - no fallback
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract additional metadata
        metadata.sections = self._extract_sections(content)
        metadata.creation_date = self._extract_creation_date(content)
        metadata.author = self._extract_author(content)
        
        return metadata
    
    def _classify_content_type(self, text: str) -> str:
        """Classify content type for adaptive chunking."""
        if not text:
            return "general"
        
        # Simple heuristics for content classification
        text_lower = text.lower()
        
        # Technical content indicators
        technical_indicators = [
            'specification', 'technical', 'procedure', 'installation',
            'configuration', 'api', 'function', 'parameter', 'error',
            'warning', 'debug', 'log', 'system', 'component'
        ]
        
        # Narrative content indicators
        narrative_indicators = [
            'story', 'narrative', 'description', 'background',
            'history', 'overview', 'introduction', 'conclusion'
        ]
        
        # Structured content indicators
        structured_indicators = [
            'table', 'list', 'item', 'step', 'instruction',
            'checklist', 'form', 'data', 'record'
        ]
        
        # Count matches
        technical_count = sum(1 for indicator in technical_indicators if indicator in text_lower)
        narrative_count = sum(1 for indicator in narrative_indicators if indicator in text_lower)
        structured_count = sum(1 for indicator in structured_indicators if indicator in text_lower)
        
        # Determine content type
        if technical_count > narrative_count and technical_count > structured_count:
            return "technical"
        elif narrative_count > technical_count and narrative_count > structured_count:
            return "narrative"
        elif structured_count > technical_count and structured_count > narrative_count:
            return "structured"
        else:
            return "general"
    
    def _enhance_metadata(self, base_metadata: Dict[str, Any], chunk_text: str) -> Dict[str, Any]:
        """Enhance metadata with chunk-specific information."""
        enhanced_metadata = base_metadata.copy() if base_metadata else {}
        
        # Add chunk-specific metadata
        enhanced_metadata.update({
            'word_count': len(chunk_text.split()),
            'character_count': len(chunk_text),
            'content_type': self._classify_content_type(chunk_text),
            'has_numbers': bool(re.search(r'\d', chunk_text)),
            'has_urls': bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', chunk_text)),
            'sentence_count': len(re.split(r'[.!?]+', chunk_text))
        })
        
        return enhanced_metadata
    
    def _extract_sections(self, content: str) -> List[str]:
        """Extract section headers from content."""
        sections = []
        
        # Look for common section patterns
        section_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^[A-Z][A-Z\s]+\n[-=]+\n',  # Underlined headers
            r'^\d+\.\s+(.+)$',  # Numbered sections
            r'^[A-Z][^.!?]*[.!?]?\n',  # Capitalized lines
        ]
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section patterns
            for pattern in section_patterns:
                if re.match(pattern, line, re.MULTILINE):
                    # Clean up the section title
                    title = re.sub(r'^#+\s+', '', line)
                    title = re.sub(r'^\d+\.\s+', '', title)
                    if title and len(title) < 100:  # Reasonable length
                        sections.append(title)
                    break
        
        return sections[:10]  # Limit to first 10 sections
    
    def _extract_creation_date(self, content: str) -> Optional[str]:
        """Extract creation date from content."""
        # Look for date patterns
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_author(self, content: str) -> Optional[str]:
        """Extract author information from content."""
        # Look for author patterns
        author_patterns = [
            r'Author[:\s]+([^\n]+)',
            r'By[:\s]+([^\n]+)',
            r'Written by[:\s]+([^\n]+)',
            r'©\s*\d{4}\s+([^\n]+)',
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                author = match.group(1).strip()
                if len(author) < 100:  # Reasonable length
                    return author
        
        return None
    
    def process_batch(self, file_paths: List[str], use_semantic_chunking: bool = True) -> Dict[str, List[DocumentChunk]]:
        """Process multiple documents in batch."""
        results = {}
        
        for file_path in file_paths:
            # No exception handling - let errors bubble up
            chunks = self.process_document_enhanced(file_path, use_semantic_chunking)
            results[file_path] = chunks
        
        return results 