"""
Book Dataset Generator

Generates training datasets from document chunks by creating prompt/completion pairs
through sliding window subsequencing. Configurable context windows and domain filtering.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BookGenerationMode(str, Enum):
    """Mode for generating training data from book chunks."""
    CONTINUATION = "continuation"      # Context → next chunk (original mode)
    QA_GENERATION = "qa_generation"    # Generate Q&A pairs from chunks
    SUMMARIZATION = "summarization"    # Generate summaries of chunks
    EXPLANATION = "explanation"        # Explain concepts in chunks
    MIXED = "mixed"                    # Combination of above


@dataclass
class BookDatasetConfig:
    """Configuration for book dataset generation."""
    
    # Source configuration
    domain_id: str                              # Source domain containing the book(s)
    document_ids: Optional[List[str]] = None    # Specific docs, or None for all in domain
    
    # Window configuration
    context_chunks: int = 3                     # Number of past chunks in prompt
    output_chunks: int = 1                      # Number of chunks to predict
    stride: int = 1                             # Sliding window stride
    
    # Optional settings
    include_metadata: bool = False              # Include chapter/page info
    system_prompt: str = ""                     # Optional system prompt prefix
    max_examples: Optional[int] = None          # Limit total examples
    
    # Output configuration
    name: str = ""
    description: str = ""


@dataclass
class GeneratedExample:
    """A single training example."""
    prompt: str
    completion: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            **({"metadata": self.metadata} if self.metadata else {})
        }


class BookDatasetGenerator:
    """Generates training datasets from document chunks."""
    
    def __init__(self, document_manager):
        """
        Initialize the generator.
        
        Args:
            document_manager: DocumentManager instance for chunk retrieval
        """
        self.document_manager = document_manager
    
    async def preview(
        self, 
        config: BookDatasetConfig, 
        num_examples: int = 3
    ) -> List[GeneratedExample]:
        """
        Generate a preview of training examples.
        
        Args:
            config: Dataset generation configuration
            num_examples: Number of examples to preview
            
        Returns:
            List of generated example previews
        """
        examples = []
        
        # Get documents from domain
        documents = await self._get_documents(config)
        if not documents:
            return examples
        
        # Generate from first document only for preview
        doc = documents[0]
        chunks, total = await self.document_manager.get_chunks_paginated(
            doc.id, offset=0, limit=config.context_chunks + config.output_chunks + num_examples
        )
        
        if len(chunks) < config.context_chunks + config.output_chunks:
            return examples
        
        # Generate preview examples
        for i in range(min(num_examples, len(chunks) - config.context_chunks - config.output_chunks + 1)):
            start_idx = i * config.stride
            if start_idx + config.context_chunks + config.output_chunks > len(chunks):
                break
                
            example = self._create_example(
                chunks[start_idx:start_idx + config.context_chunks],
                chunks[start_idx + config.context_chunks:start_idx + config.context_chunks + config.output_chunks],
                config,
                doc.name
            )
            examples.append(example)
        
        return examples
    
    async def generate(
        self, 
        config: BookDatasetConfig
    ) -> Tuple[List[GeneratedExample], Dict[str, Any]]:
        """
        Generate full training dataset.
        
        Args:
            config: Dataset generation configuration
            
        Returns:
            Tuple of (examples list, statistics dict)
        """
        examples = []
        stats = {
            "documents_processed": 0,
            "total_chunks": 0,
            "examples_generated": 0,
            "skipped_short_docs": 0,
        }
        
        documents = await self._get_documents(config)
        
        for doc in documents:
            # Get all chunks for this document
            chunks, total = await self.document_manager.get_chunks_paginated(
                doc.id, offset=0, limit=10000  # Get all chunks
            )
            
            stats["total_chunks"] += len(chunks)
            
            # Skip if document is too short
            min_chunks = config.context_chunks + config.output_chunks
            if len(chunks) < min_chunks:
                stats["skipped_short_docs"] += 1
                logger.info(f"Skipping document {doc.name}: only {len(chunks)} chunks (need {min_chunks})")
                continue
            
            stats["documents_processed"] += 1
            
            # Sliding window through document
            i = 0
            while i + config.context_chunks + config.output_chunks <= len(chunks):
                context_chunks = chunks[i:i + config.context_chunks]
                output_chunks = chunks[i + config.context_chunks:i + config.context_chunks + config.output_chunks]
                
                example = self._create_example(context_chunks, output_chunks, config, doc.name)
                examples.append(example)
                
                # Check max examples limit
                if config.max_examples and len(examples) >= config.max_examples:
                    stats["examples_generated"] = len(examples)
                    return examples, stats
                
                i += config.stride
        
        stats["examples_generated"] = len(examples)
        return examples, stats
    
    async def generate_and_save(
        self,
        config: BookDatasetConfig,
        output_path: Path
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Generate dataset and save to JSONL file.
        
        Args:
            config: Dataset generation configuration
            output_path: Path to save the JSONL file
            
        Returns:
            Tuple of (file path, statistics)
        """
        examples, stats = await self.generate(config)
        
        # Write to JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example.to_dict(), ensure_ascii=False) + '\n')
        
        stats["file_path"] = str(output_path)
        stats["file_size_bytes"] = output_path.stat().st_size
        
        logger.info(f"Generated book dataset: {len(examples)} examples -> {output_path}")
        return output_path, stats
    
    async def _get_documents(self, config: BookDatasetConfig) -> List:
        """Get documents based on config."""
        if config.document_ids:
            # Get specific documents
            documents = []
            for doc_id in config.document_ids:
                doc = await self.document_manager.get_document(doc_id)
                if doc:
                    documents.append(doc)
            return documents
        else:
            # Get all documents in domain
            documents, _ = await self.document_manager.list_documents(
                domain_id=config.domain_id,
                limit=1000
            )
            return documents
    
    def _create_example(
        self,
        context_chunks: List,
        output_chunks: List,
        config: BookDatasetConfig,
        document_name: str
    ) -> GeneratedExample:
        """Create a single training example from chunks."""
        
        # Build prompt
        context_text = "\n\n".join(chunk.content for chunk in context_chunks)
        
        if config.system_prompt:
            prompt = f"{config.system_prompt}\n\nContinue the following passage:\n\n{context_text}"
        else:
            prompt = f"Continue the following passage:\n\n{context_text}"
        
        # Build completion
        completion = "\n\n".join(chunk.content for chunk in output_chunks)
        
        # Build metadata
        metadata = {}
        if config.include_metadata:
            metadata = {
                "document": document_name,
                "context_start_idx": context_chunks[0].chunk_index if context_chunks else 0,
                "output_start_idx": output_chunks[0].chunk_index if output_chunks else 0,
            }
            # Include page numbers if available
            if context_chunks and context_chunks[0].page_number is not None:
                metadata["context_pages"] = [c.page_number for c in context_chunks if c.page_number]
            if output_chunks and output_chunks[0].page_number is not None:
                metadata["output_pages"] = [c.page_number for c in output_chunks if c.page_number]
        
        return GeneratedExample(
            prompt=prompt,
            completion=completion,
            metadata=metadata
        )


async def estimate_dataset_size(
    document_manager,
    config: BookDatasetConfig
) -> Dict[str, Any]:
    """
    Estimate the size of a book dataset without generating it.
    
    Returns estimated example count and token counts.
    """
    generator = BookDatasetGenerator(document_manager)
    documents = await generator._get_documents(config)
    
    total_chunks = 0
    estimated_examples = 0
    valid_docs = 0
    
    for doc in documents:
        _, count = await document_manager.get_chunks_paginated(doc.id, limit=1)
        total_chunks += count
        
        min_chunks = config.context_chunks + config.output_chunks
        if count >= min_chunks:
            valid_docs += 1
            # Calculate examples from this document
            doc_examples = (count - min_chunks) // config.stride + 1
            estimated_examples += doc_examples
    
    if config.max_examples:
        estimated_examples = min(estimated_examples, config.max_examples)
    
    return {
        "total_documents": len(documents),
        "valid_documents": valid_docs,
        "total_chunks": total_chunks,
        "estimated_examples": estimated_examples,
    }
