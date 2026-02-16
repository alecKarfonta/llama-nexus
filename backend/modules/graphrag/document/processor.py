import os
from typing import Dict, List, Any
from dataclasses import dataclass
import fitz  # PyMuPDF for PDF processing
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
import csv
import json
import markdown

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with metadata."""
    text: str
    chunk_id: str
    source_file: str
    page_number: int | None = None
    section_header: str | None = None
    chunk_index: int = 0
    metadata: Dict[str, Any] | None = None

@dataclass
class DocumentMetadata:
    """Metadata extracted from a document."""
    title: str | None = None
    author: str | None = None
    creation_date: str | None = None
    file_type: str | None = None
    file_size: int | None = None
    page_count: int | None = None
    sections: List[str] | None = None

class DocumentProcessor:
    """Handles multi-format document ingestion and processing."""
    
    def __init__(self):
        self.processors = {
            '.pdf': self.process_pdf,
            '.docx': self.process_docx,
            '.txt': self.process_text,
            '.md': self.process_markdown,
            '.html': self.process_html,
            '.csv': self.process_csv,
            '.json': self.process_json
        }
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Process a document and return chunks with metadata."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.processors:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Extract metadata
        metadata = self.extract_metadata(file_path)
        
        # Process document based on type
        chunks = self.processors[file_ext](file_path, metadata)
        
        return chunks
    
    def process_pdf(self, file_path: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Process PDF documents using PyMuPDF."""
        chunks = []
        doc = fitz.open(file_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Split text into chunks (basic implementation)
            text_chunks = self.create_text_chunks(text, max_chunk_size=1000)
            
            for chunk_idx, chunk_text in enumerate(text_chunks):
                chunk = DocumentChunk(
                    text=chunk_text,
                    chunk_id=f"{os.path.basename(file_path)}_p{page_num}_c{chunk_idx}",
                    source_file=file_path,
                    page_number=page_num + 1,
                    chunk_index=chunk_idx,
                    metadata=metadata.__dict__
                )
                chunks.append(chunk)
        
        doc.close()
        return chunks
    
    def process_docx(self, file_path: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Process DOCX documents using python-docx."""
        chunks = []
        doc = DocxDocument(file_path)
        
        current_section = ""
        chunk_text = ""
        chunk_idx = 0
        
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            
            # Check if this is a header (simple heuristic)
            if paragraph.style.name.startswith('Heading'):
                current_section = text
                # Start new chunk for header
                if chunk_text:
                    chunk = DocumentChunk(
                        text=chunk_text,
                        chunk_id=f"{os.path.basename(file_path)}_c{chunk_idx}",
                        source_file=file_path,
                        section_header=current_section,
                        chunk_index=chunk_idx,
                        metadata=metadata.__dict__
                    )
                    chunks.append(chunk)
                    chunk_text = ""
                    chunk_idx += 1
                
                chunk_text = text
            else:
                chunk_text += "\n" + text if chunk_text else text
                
                # Create new chunk if text is getting long
                if len(chunk_text) > 1000:
                    chunk = DocumentChunk(
                        text=chunk_text,
                        chunk_id=f"{os.path.basename(file_path)}_c{chunk_idx}",
                        source_file=file_path,
                        section_header=current_section,
                        chunk_index=chunk_idx,
                        metadata=metadata.__dict__
                    )
                    chunks.append(chunk)
                    chunk_text = ""
                    chunk_idx += 1
        
        # Add remaining text as final chunk
        if chunk_text:
            chunk = DocumentChunk(
                text=chunk_text,
                chunk_id=f"{os.path.basename(file_path)}_c{chunk_idx}",
                source_file=file_path,
                section_header=current_section,
                chunk_index=chunk_idx,
                metadata=metadata.__dict__
            )
            chunks.append(chunk)
        
        return chunks
    
    def process_text(self, file_path: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Process plain text files."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        text_chunks = self.create_text_chunks(text, max_chunk_size=1000)
        chunks = []
        
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk = DocumentChunk(
                text=chunk_text,
                chunk_id=f"{os.path.basename(file_path)}_c{chunk_idx}",
                source_file=file_path,
                chunk_index=chunk_idx,
                metadata=metadata.__dict__
            )
            chunks.append(chunk)
        
        return chunks
    
    def process_markdown(self, file_path: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Process markdown files with section-aware chunking."""
        with open(file_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
        
        # Parse markdown to extract structure
        md = markdown.Markdown(extensions=['toc', 'tables', 'fenced_code'])
        html_content = md.convert(md_text)
        
        # Use BeautifulSoup to extract text while preserving structure
        soup = BeautifulSoup(html_content, 'html.parser')
        
        chunks = []
        current_section = ""
        chunk_text = ""
        chunk_idx = 0
        
        # Process each element to maintain section structure
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'code', 'pre']):
            text = element.get_text(strip=True)
            if not text:
                continue
            
            # Check if this is a header
            if element.name.startswith('h'):
                # Start new section
                current_section = text
                # Add header to new chunk
                if chunk_text:
                    chunk = DocumentChunk(
                        text=chunk_text,
                        chunk_id=f"{os.path.basename(file_path)}_c{chunk_idx}",
                        source_file=file_path,
                        section_header=current_section,
                        chunk_index=chunk_idx,
                        metadata=metadata.__dict__
                    )
                    chunks.append(chunk)
                    chunk_text = ""
                    chunk_idx += 1
                
                chunk_text = f"# {text}\n"
            else:
                # Add content to current chunk
                if chunk_text:
                    chunk_text += "\n" + text
                else:
                    chunk_text = text
                
                # Create new chunk if text is getting long
                if len(chunk_text) > 1000:
                    chunk = DocumentChunk(
                        text=chunk_text,
                        chunk_id=f"{os.path.basename(file_path)}_c{chunk_idx}",
                        source_file=file_path,
                        section_header=current_section,
                        chunk_index=chunk_idx,
                        metadata=metadata.__dict__
                    )
                    chunks.append(chunk)
                    chunk_text = ""
                    chunk_idx += 1
        
        # Add remaining text as final chunk
        if chunk_text:
            chunk = DocumentChunk(
                text=chunk_text,
                chunk_id=f"{os.path.basename(file_path)}_c{chunk_idx}",
                source_file=file_path,
                section_header=current_section,
                chunk_index=chunk_idx,
                metadata=metadata.__dict__
            )
            chunks.append(chunk)
        
        return chunks
    
    def process_html(self, file_path: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Process HTML files using BeautifulSoup."""
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # Extract text content
        text = soup.get_text(separator='\n', strip=True)
        
        text_chunks = self.create_text_chunks(text, max_chunk_size=1000)
        chunks = []
        
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk = DocumentChunk(
                text=chunk_text,
                chunk_id=f"{os.path.basename(file_path)}_c{chunk_idx}",
                source_file=file_path,
                chunk_index=chunk_idx,
                metadata=metadata.__dict__
            )
            chunks.append(chunk)
        
        return chunks
    
    def process_csv(self, file_path: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Process CSV files."""
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Convert to structured text
        if rows:
            headers = list(rows[0].keys())
            text = f"Headers: {', '.join(headers)}\n\n"
            
            for i, row in enumerate(rows):
                text += f"Row {i+1}: {', '.join([f'{k}: {v}' for k, v in row.items()])}\n"
                
                # Create new chunk every 50 rows
                if (i + 1) % 50 == 0:
                    chunk = DocumentChunk(
                        text=text,
                        chunk_id=f"{os.path.basename(file_path)}_c{i//50}",
                        source_file=file_path,
                        chunk_index=i//50,
                        metadata=metadata.__dict__
                    )
                    chunks.append(chunk)
                    text = ""
            
            # Add remaining text
            if text:
                chunk = DocumentChunk(
                    text=text,
                    chunk_id=f"{os.path.basename(file_path)}_c{len(chunks)}",
                    source_file=file_path,
                    chunk_index=len(chunks),
                    metadata=metadata.__dict__
                )
                chunks.append(chunk)
        
        return chunks
    
    def process_json(self, file_path: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Process JSON files."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert JSON to structured text
        text = json.dumps(data, indent=2)
        
        text_chunks = self.create_text_chunks(text, max_chunk_size=1000)
        chunks = []
        
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk = DocumentChunk(
                text=chunk_text,
                chunk_id=f"{os.path.basename(file_path)}_c{chunk_idx}",
                source_file=file_path,
                chunk_index=chunk_idx,
                metadata=metadata.__dict__
            )
            chunks.append(chunk)
        
        return chunks
    
    def create_text_chunks(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Create overlapping text chunks for better context retention."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start, end - 200), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Overlap by 100 characters for context retention
            start = end - 100
            if start >= len(text):
                break
        
        return chunks
    
    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """Extract metadata from document."""
        file_stat = os.stat(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        metadata = DocumentMetadata(
            title=os.path.basename(file_path),
            file_type=file_ext,
            file_size=file_stat.st_size
        )
        
        # Extract additional metadata based on file type
        if file_ext == '.pdf':
            try:
                doc = fitz.open(file_path)
                metadata.page_count = len(doc)
                # Try to extract title from PDF metadata
                if doc.metadata.get('title'):
                    metadata.title = doc.metadata['title']
                doc.close()
            except:
                pass
        
        return metadata 