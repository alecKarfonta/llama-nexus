from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import DBSCAN
import re

class SemanticChunker:
    """Intelligent text chunking using semantic similarity."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.min_chunk_size = 100
        self.max_chunk_size = 1200
        self.overlap_size = 100
        
        # Procedure detection patterns
        self.procedure_patterns = [
            r'(step\s+\d+|^\d+\.|^[a-z]\)|first|second|third|next|then|finally|WARNING:|CAUTION:|NOTE:)',
            r'(procedure|inspection|maintenance|installation|removal|replacement)',
            r'(before\s+\w+ing|after\s+\w+ing|while\s+\w+ing)',
            r'(turn\s+\w+|remove\s+\w+|install\s+\w+|disconnect\s+\w+|connect\s+\w+)',
            r'(check\s+\w+|verify\s+\w+|ensure\s+\w+|make\s+sure)'
        ]
        
        # Technical document markers
        self.technical_markers = [
            r'(specification|requirement|tolerance|parameter)',
            r'(component|system|assembly|part\s+number)',
            r'(pressure|temperature|voltage|current|resistance)',
            r'(manual|guide|instruction|documentation)'
        ]
    
    def create_semantic_chunks(self, text: str) -> List[str]:
        """Create semantic chunks based on content similarity."""
        # Detect document type first
        doc_type = self._detect_document_type(text)
        
        # Use specialized chunking for procedures
        if doc_type == "procedure":
            return self._chunk_procedure(text)
        elif doc_type == "technical":
            return self._chunk_technical(text)
        else:
            return self._chunk_general(text)
    
    def _detect_document_type(self, text: str) -> str:
        """Detect the type of document to apply appropriate chunking strategy."""
        text_lower = text.lower()
        
        # Count procedure indicators
        procedure_count = 0
        for pattern in self.procedure_patterns:
            procedure_count += len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        # Count technical indicators
        technical_count = 0
        for pattern in self.technical_markers:
            technical_count += len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        # Determine document type
        if procedure_count > 5 or any(word in text_lower for word in ['warning:', 'caution:', 'procedure', 'step']):
            return "procedure"
        elif technical_count > 3 or any(word in text_lower for word in ['specification', 'manual', 'guide']):
            return "technical"
        else:
            return "general"
    
    def _chunk_procedure(self, text: str) -> List[str]:
        """Fixed specialized chunking for procedural documents."""
        lines = text.split('\n')
        chunks = []
        current_chunk = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a major section boundary
            if self._is_major_section_boundary(line):
                # Major sections always start new chunks
                if current_chunk and len(current_chunk) > self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            elif self._is_minor_boundary(line):
                # Minor boundaries (like steps) continue in the same chunk unless it's getting too large
                if len(current_chunk) > self.max_chunk_size:
                    # Split here if chunk is getting too large
                    chunks.append(current_chunk.strip())
                    current_chunk = line + "\n"
                else:
                    # Continue in same chunk
                    current_chunk += line + "\n"
            else:
                # Regular content
                current_chunk += line + "\n"
                
                # Split if chunk gets too large
                if len(current_chunk) > self.max_chunk_size:
                    # Try to find a good break point
                    break_point = self._find_procedure_break_point(current_chunk)
                    if break_point:
                        chunks.append(current_chunk[:break_point].strip())
                        current_chunk = current_chunk[break_point:].strip() + "\n"
                    else:
                        # Fallback to size-based splitting
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If no meaningful chunks found, fall back to general chunking
        if not chunks:
            return self._chunk_general(text)
        
        return chunks
    
    def _is_procedure_boundary(self, line: str) -> bool:
        """Check if a line marks the start of a new procedure section."""
        line_lower = line.lower()
        line_original = line.strip()
        
        # Strong indicators of procedure boundaries
        strong_indicators = [
            r'^step\s+\d+:',  # "Step 1:", "Step 2:", etc.
            r'^\d+\.',  # "1.", "2.", etc.
            r'^[a-z]\)',  # "a)", "b)", etc.
            r'^section\s+\d+:',  # "SECTION 1:", "SECTION 2:", etc.
            r'^warning:',  # "WARNING:"
            r'^caution:',  # "CAUTION:"
            r'^note:',  # "NOTE:"
            r'^procedure:',  # "PROCEDURE:"
            r'^(inspection|maintenance|installation|removal):',
            r'^(fuel\s+system|engine|transmission|brake|electrical):',
            r'^problem:',  # "Problem:"
            r'^every\s+\d+',  # "Every 5,000 miles:"
            r'^oil\s+change\s+procedure:',  # "Oil Change Procedure:"
            r'^[A-Z]{2,}[0-9A-Z]{3,}\s+[A-Z0-9]{5,}'  # Technical codes like "SEF710G SMA804A"
        ]
        
        # Check against patterns
        for pattern in strong_indicators:
            if re.match(pattern, line_lower, re.IGNORECASE):
                return True
        
        # Check for section headers that might be in uppercase
        if re.match(r'^SECTION\s+\d+:', line_original):
            return True
        
        # Check for maintenance schedule patterns
        if re.match(r'^Every\s+\d+', line_original):
            return True
        
        return False
    
    def _is_major_section_boundary(self, line: str) -> bool:
        """Check if a line marks a major section boundary."""
        line_lower = line.lower()
        line_original = line.strip()
        
        # Major section indicators
        major_indicators = [
            r'^section\s+\d+:',  # "SECTION 1:", "SECTION 2:", etc.
            r'^(procedure|inspection|maintenance|installation|removal):',
            r'^oil\s+change\s+procedure:',  # "Oil Change Procedure:"
            r'^(fuel\s+system|engine|transmission|brake|electrical):',
        ]
        
        # Check against patterns
        for pattern in major_indicators:
            if re.match(pattern, line_lower, re.IGNORECASE):
                return True
        
        # Check for section headers that might be in uppercase
        if re.match(r'^SECTION\s+\d+:', line_original):
            return True
        
        return False
    
    def _is_minor_boundary(self, line: str) -> bool:
        """Check if a line marks a minor boundary (like procedure steps)."""
        line_lower = line.lower()
        line_original = line.strip()
        
        # Minor boundary indicators  
        minor_indicators = [
            r'^step\s+\d+:',  # "Step 1:", "Step 2:", etc.
            r'^\d+\.',  # "1.", "2.", etc.
            r'^[a-z]\)',  # "a)", "b)", etc.
            r'^warning:',  # "WARNING:"
            r'^caution:',  # "CAUTION:"
            r'^note:',  # "NOTE:"
            r'^problem:',  # "Problem:"
            r'^every\s+\d+',  # "Every 5,000 miles:"
        ]
        
        # Check against patterns
        for pattern in minor_indicators:
            if re.match(pattern, line_lower, re.IGNORECASE):
                return True
        
        # Check for maintenance schedule patterns
        if re.match(r'^Every\s+\d+', line_original):
            return True
        
        return False
    
    def _find_procedure_break_point(self, chunk: str) -> int:
        """Find the best break point within a procedure chunk."""
        lines = chunk.split('\n')
        best_break = None
        
        for i, line in enumerate(lines):
            line_pos = chunk.find(line)
            
            # Good break points in procedures
            if any(pattern in line.lower() for pattern in ['turn ignition', 'remove fuse', 'start engine', 'disconnect']):
                if line_pos > len(chunk) * 0.3:  # Not too early
                    best_break = line_pos
                    break
        
        return best_break or 0
    
    def _chunk_technical(self, text: str) -> List[str]:
        """Specialized chunking for technical documents."""
        # Use more conservative parameters for technical documents
        sentences = self._split_into_sentences_advanced(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # Get embeddings for all sentences
        embeddings = self.model.encode(sentences)
        
        # Use more conservative DBSCAN parameters for technical content
        clustering = DBSCAN(eps=0.4, min_samples=3).fit(embeddings)
        
        # Group sentences by cluster
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(sentences[i])
        
        # Create chunks from clusters
        chunks = []
        for cluster_sentences in clusters.values():
            chunk_text = " ".join(cluster_sentences)
            
            # Split large clusters into smaller chunks
            if len(chunk_text) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(chunk_text)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)
        
        # If no meaningful clusters found, fall back to size-based chunking
        if not chunks:
            chunks = self._fallback_chunking(text)
        
        return chunks
    
    def _chunk_general(self, text: str) -> List[str]:
        """General semantic chunking for regular documents."""
        # Split text into sentences
        sentences = self._split_into_sentences_advanced(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # Get embeddings for all sentences
        embeddings = self.model.encode(sentences)
        
        # Use original DBSCAN parameters for general content
        clustering = DBSCAN(eps=0.3, min_samples=2).fit(embeddings)
        
        # Group sentences by cluster
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(sentences[i])
        
        # Create chunks from clusters
        chunks = []
        for cluster_sentences in clusters.values():
            chunk_text = " ".join(cluster_sentences)
            
            # Split large clusters into smaller chunks
            if len(chunk_text) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(chunk_text)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)
        
        # If no meaningful clusters found, fall back to size-based chunking
        if not chunks:
            chunks = self._fallback_chunking(text)
        
        return chunks
    
    def _split_into_sentences_advanced(self, text: str) -> List[str]:
        """Advanced sentence splitting that preserves procedural sequences."""
        # First pass: basic sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Second pass: merge sentences that are part of procedures
        merged_sentences = []
        i = 0
        while i < len(sentences):
            current_sentence = sentences[i]
            
            # Check if next sentence is a continuation of a procedure
            if i + 1 < len(sentences):
                next_sentence = sentences[i + 1]
                
                # Merge if current sentence ends with procedure indicators
                if any(current_sentence.lower().endswith(word) for word in 
                       ['remove', 'install', 'disconnect', 'connect', 'turn', 'start', 'stop']):
                    # And next sentence starts with continuation words
                    if any(next_sentence.lower().startswith(word) for word in 
                           ['after', 'before', 'then', 'next', 'while', 'until']):
                        current_sentence += ". " + next_sentence
                        i += 1  # Skip the next sentence as it's been merged
            
            merged_sentences.append(current_sentence)
            i += 1
        
        return merged_sentences
    
    def _split_large_chunk(self, chunk_text: str) -> List[str]:
        """Split large chunks into smaller ones while preserving context."""
        chunks = []
        start = 0
        
        while start < len(chunk_text):
            end = start + self.max_chunk_size
            
            # Try to break at sentence boundary
            if end < len(chunk_text):
                # Look for sentence endings
                for i in range(end, max(start, end - 200), -1):
                    if chunk_text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = chunk_text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Overlap for context retention
            start = end - self.overlap_size
            if start >= len(chunk_text):
                break
        
        return chunks
    
    def _fallback_chunking(self, text: str) -> List[str]:
        """Fallback to size-based chunking when semantic clustering fails."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                for i in range(end, max(start, end - 200), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            # Overlap for context retention
            start = end - self.overlap_size
            if start >= len(text):
                break
        
        return chunks
    
    def create_adaptive_chunks(self, text: str, content_type: str = "general") -> List[str]:
        """Create chunks with size adapted to content type."""
        # Adjust chunk size based on content type
        if content_type == "technical":
            self.max_chunk_size = 800
            self.min_chunk_size = 150
        elif content_type == "narrative":
            self.max_chunk_size = 1200
            self.min_chunk_size = 80
        elif content_type == "structured":
            self.max_chunk_size = 600
            self.min_chunk_size = 200
        elif content_type == "procedure":
            self.max_chunk_size = 1000
            self.min_chunk_size = 150
        else:
            self.max_chunk_size = 1000
            self.min_chunk_size = 100
        
        return self.create_semantic_chunks(text)
    
    def preserve_structure(self, text: str, structure_markers: Dict[str, str]) -> List[Dict[str, Any]]:
        """Create chunks while preserving document structure."""
        chunks = []
        current_section = ""
        current_subsection = ""
        
        # Split by structure markers
        lines = text.split('\n')
        current_chunk = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            if any(marker in line for marker in structure_markers.get('section', [])):
                # Save current chunk
                if current_chunk:
                    chunks.append({
                        'text': current_chunk,
                        'section': current_section,
                        'subsection': current_subsection
                    })
                    current_chunk = ""
                
                current_section = line
                current_subsection = ""
            
            # Check for subsection headers
            elif any(marker in line for marker in structure_markers.get('subsection', [])):
                if current_chunk:
                    chunks.append({
                        'text': current_chunk,
                        'section': current_section,
                        'subsection': current_subsection
                    })
                    current_chunk = ""
                
                current_subsection = line
            
            else:
                current_chunk += line + "\n"
                
                # Create new chunk if getting too large
                if len(current_chunk) > self.max_chunk_size:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'section': current_section,
                        'subsection': current_subsection
                    })
                    current_chunk = ""
        
        # Add remaining chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'section': current_section,
                'subsection': current_subsection
            })
        
        return chunks 