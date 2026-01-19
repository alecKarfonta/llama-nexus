"""
Book Dataset Distillation Module

Generates high-quality training data from book chunks using LLM distillation.
Transforms raw text into Q&A pairs, summaries, and explanations using teacher models.
"""

import asyncio
import json
import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

from pydantic import BaseModel, Field

from .book_dataset import BookGenerationMode, GeneratedExample
from .distillation import (
    TeacherProvider,
    TeacherClient,
    OpenAITeacher,
    AnthropicTeacher,
    GoogleTeacher,
    LocalTeacher,
    GenerationResult,
    QualityConfig,
    QualityScore,
)
from .config import Dataset, DatasetFormat, DatasetStatus
from .storage import DatasetStore

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class BookDistillationConfig(BaseModel):
    """Configuration for book dataset distillation."""
    
    # Source configuration
    domain_id: str
    document_ids: Optional[List[str]] = None
    
    # Generation mode
    generation_mode: BookGenerationMode = BookGenerationMode.QA_GENERATION
    
    # Teacher model configuration
    teacher_provider: TeacherProvider = TeacherProvider.OPENAI
    teacher_model: str = "gpt-4o-mini"
    teacher_api_key: Optional[str] = None
    teacher_base_url: Optional[str] = None
    
    # Generation settings
    questions_per_chunk: int = Field(3, ge=1, le=10)
    include_source_context: bool = True
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=100)
    
    # Quality settings
    quality_enabled: bool = True
    quality_threshold: float = Field(0.7, ge=0.0, le=1.0)
    
    # Processing settings
    batch_size: int = Field(5, ge=1, le=20)
    max_chunks: Optional[int] = None  # Limit chunks to process
    
    # Output configuration
    name: str = ""
    description: str = ""


class DistillationJobStatus(str, Enum):
    """Status of a book distillation job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DistillationProgress:
    """Progress tracking for distillation jobs."""
    total_chunks: int = 0
    processed_chunks: int = 0
    generated_examples: int = 0
    failed_chunks: int = 0
    current_document: str = ""
    
    @property
    def progress_percent(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return (self.processed_chunks / self.total_chunks) * 100


@dataclass
class BookDistillationJob:
    """A book distillation job."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config: BookDistillationConfig = None
    status: DistillationJobStatus = DistillationJobStatus.PENDING
    progress: DistillationProgress = field(default_factory=DistillationProgress)
    output_dataset_id: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "config": self.config.dict() if self.config else None,
            "status": self.status.value,
            "progress": {
                "total_chunks": self.progress.total_chunks,
                "processed_chunks": self.progress.processed_chunks,
                "generated_examples": self.progress.generated_examples,
                "failed_chunks": self.progress.failed_chunks,
                "current_document": self.progress.current_document,
                "progress_percent": self.progress.progress_percent,
            },
            "output_dataset_id": self.output_dataset_id,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# =============================================================================
# Prompt Templates
# =============================================================================

QA_GENERATION_PROMPT = """You are an expert educator creating study questions. Given the following text passage, generate {n} high-quality question-answer pairs.

Requirements:
- Questions should test understanding of key concepts from the passage
- Include a mix of factual recall and analytical questions
- Answers should be comprehensive but concise
- All answers must be derivable from the passage

Text:
{content}

Generate exactly {n} question-answer pairs in this JSON format:
[
  {{"question": "...", "answer": "..."}},
  ...
]

Response (JSON only):"""

SUMMARIZATION_PROMPT = """You are an expert at distilling information. Summarize the following passage into a clear, structured format.

Text:
{content}

Generate an instruction-response pair where:
- The instruction asks to summarize or explain the key points
- The response provides a comprehensive summary

Response in JSON format:
{{"instruction": "...", "response": "..."}}

Response (JSON only):"""

EXPLANATION_PROMPT = """You are an expert tutor. Based on the following passage, identify any complex concepts and explain them clearly.

Text:
{content}

Generate instruction-response pairs for each concept worth explaining:
- The instruction should ask about a specific concept
- The response should provide a clear, educational explanation

Response in JSON format (list of pairs):
[
  {{"instruction": "...", "response": "..."}},
  ...
]

Response (JSON only):"""


# =============================================================================
# Distillation Generator
# =============================================================================

class BookDistillationGenerator:
    """Generates training data from book chunks using LLM distillation."""
    
    def __init__(self, document_manager, dataset_store: DatasetStore):
        """
        Initialize the distillation generator.
        
        Args:
            document_manager: DocumentManager for accessing chunks
            dataset_store: DatasetStore for saving generated datasets
        """
        self.document_manager = document_manager
        self.dataset_store = dataset_store
        self._teacher_client: Optional[TeacherClient] = None
        self._cancel_requested = False
    
    def _get_teacher_client(self, config: BookDistillationConfig) -> TeacherClient:
        """Create teacher client based on config."""
        api_key = config.teacher_api_key or self._get_env_api_key(config.teacher_provider)
        
        if config.teacher_provider == TeacherProvider.OPENAI:
            return OpenAITeacher(api_key=api_key, model=config.teacher_model)
        elif config.teacher_provider == TeacherProvider.ANTHROPIC:
            return AnthropicTeacher(api_key=api_key, model=config.teacher_model)
        elif config.teacher_provider == TeacherProvider.GOOGLE:
            return GoogleTeacher(api_key=api_key, model=config.teacher_model)
        elif config.teacher_provider == TeacherProvider.LOCAL:
            return LocalTeacher(
                base_url=config.teacher_base_url or "http://localhost:8600/v1",
                model=config.teacher_model
            )
        else:
            raise ValueError(f"Unknown teacher provider: {config.teacher_provider}")
    
    def _get_env_api_key(self, provider: TeacherProvider) -> str:
        """Get API key from environment."""
        import os
        env_keys = {
            TeacherProvider.OPENAI: "OPENAI_API_KEY",
            TeacherProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            TeacherProvider.GOOGLE: "GOOGLE_API_KEY",
        }
        key = os.environ.get(env_keys.get(provider, ""), "")
        if not key and provider != TeacherProvider.LOCAL:
            raise ValueError(f"No API key found for {provider.value}. Set {env_keys.get(provider)} or provide teacher_api_key.")
        return key
    
    async def estimate(
        self,
        config: BookDistillationConfig
    ) -> Dict[str, Any]:
        """
        Estimate the distillation job size and cost.
        
        Returns estimated chunk count, example count, and API cost.
        """
        documents = await self._get_documents(config)
        
        total_chunks = 0
        for doc in documents:
            _, count = await self.document_manager.get_chunks_paginated(doc.id, limit=1)
            total_chunks += count
        
        if config.max_chunks:
            total_chunks = min(total_chunks, config.max_chunks)
        
        # Estimate examples based on mode
        if config.generation_mode == BookGenerationMode.QA_GENERATION:
            estimated_examples = total_chunks * config.questions_per_chunk
        elif config.generation_mode == BookGenerationMode.MIXED:
            estimated_examples = total_chunks * (config.questions_per_chunk + 2)  # Q&A + summary + explanation
        else:
            estimated_examples = total_chunks
        
        # Rough cost estimate (tokens per chunk ~500, response ~300)
        tokens_per_chunk = 800
        cost_per_1k_tokens = 0.002 if "mini" in config.teacher_model else 0.01
        estimated_cost = (total_chunks * tokens_per_chunk / 1000) * cost_per_1k_tokens
        
        return {
            "total_documents": len(documents),
            "total_chunks": total_chunks,
            "estimated_examples": estimated_examples,
            "estimated_cost_usd": round(estimated_cost, 2),
            "model": config.teacher_model,
        }
    
    async def preview(
        self,
        config: BookDistillationConfig,
        num_chunks: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate a preview of distilled examples from a few chunks.
        """
        self._teacher_client = self._get_teacher_client(config)
        documents = await self._get_documents(config)
        
        if not documents:
            return []
        
        # Get first chunk from first document
        doc = documents[0]
        chunks, _ = await self.document_manager.get_chunks_paginated(
            doc.id, offset=0, limit=num_chunks
        )
        
        examples = []
        for chunk in chunks:
            chunk_examples = await self._distill_chunk(chunk, config)
            examples.extend(chunk_examples)
        
        return examples
    
    async def run(
        self,
        job: BookDistillationJob,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run a full distillation job.
        
        Args:
            job: The distillation job to run
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (examples list, statistics dict)
        """
        config = job.config
        job.status = DistillationJobStatus.RUNNING
        job.started_at = datetime.utcnow()
        
        self._teacher_client = self._get_teacher_client(config)
        self._cancel_requested = False
        
        examples = []
        stats = {
            "documents_processed": 0,
            "chunks_processed": 0,
            "chunks_failed": 0,
            "examples_generated": 0,
            "total_time_seconds": 0,
        }
        
        start_time = time.time()
        
        try:
            documents = await self._get_documents(config)
            
            # Count total chunks
            total_chunks = 0
            for doc in documents:
                _, count = await self.document_manager.get_chunks_paginated(doc.id, limit=1)
                total_chunks += count
            
            if config.max_chunks:
                total_chunks = min(total_chunks, config.max_chunks)
            
            job.progress.total_chunks = total_chunks
            chunks_processed = 0
            
            for doc in documents:
                if self._cancel_requested:
                    job.status = DistillationJobStatus.CANCELLED
                    break
                
                job.progress.current_document = doc.name
                
                # Get all chunks for document
                all_chunks, _ = await self.document_manager.get_chunks_paginated(
                    doc.id, offset=0, limit=10000
                )
                
                # Process in batches
                for i in range(0, len(all_chunks), config.batch_size):
                    if self._cancel_requested:
                        break
                    
                    if config.max_chunks and chunks_processed >= config.max_chunks:
                        break
                    
                    batch = all_chunks[i:i + config.batch_size]
                    
                    for chunk in batch:
                        if config.max_chunks and chunks_processed >= config.max_chunks:
                            break
                        
                        try:
                            chunk_examples = await self._distill_chunk(chunk, config)
                            examples.extend(chunk_examples)
                            job.progress.generated_examples += len(chunk_examples)
                        except Exception as e:
                            logger.warning(f"Failed to distill chunk {chunk.id}: {e}")
                            job.progress.failed_chunks += 1
                            stats["chunks_failed"] += 1
                        
                        chunks_processed += 1
                        job.progress.processed_chunks = chunks_processed
                        
                        if progress_callback:
                            await progress_callback(job)
                    
                    # Small delay to respect rate limits
                    await asyncio.sleep(0.1)
                
                stats["documents_processed"] += 1
            
            stats["chunks_processed"] = chunks_processed
            stats["examples_generated"] = len(examples)
            stats["total_time_seconds"] = time.time() - start_time
            
            if not self._cancel_requested:
                job.status = DistillationJobStatus.COMPLETED
            
            job.completed_at = datetime.utcnow()
            
        except Exception as e:
            job.status = DistillationJobStatus.FAILED
            job.error = str(e)
            logger.error(f"Distillation job failed: {e}")
            raise
        
        return examples, stats
    
    async def run_and_save(
        self,
        job: BookDistillationJob,
        progress_callback: Optional[callable] = None
    ) -> BookDistillationJob:
        """
        Run distillation job and save results as a dataset.
        """
        examples, stats = await self.run(job, progress_callback)
        
        if not examples:
            job.error = "No examples generated"
            return job
        
        # Create dataset
        config = job.config
        dataset_id = str(uuid.uuid4())
        dataset_name = config.name or f"distilled_{job.id[:8]}"
        
        # Determine format based on generation mode
        if config.generation_mode == BookGenerationMode.QA_GENERATION:
            format_type = DatasetFormat.INSTRUCTION
        else:
            format_type = DatasetFormat.ALPACA
        
        # Save to file
        target_path = self.dataset_store.dataset_path(dataset_id, f"{dataset_name}.jsonl")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Register dataset
        dataset = Dataset(
            id=dataset_id,
            name=dataset_name,
            description=f"Distilled from domain {config.domain_id}. {config.description}".strip(),
            format=format_type,
            status=DatasetStatus.READY,
            num_examples=len(examples),
            total_tokens=0,
            file_path=str(target_path),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            statistics={
                **stats,
                "source_domain": config.domain_id,
                "generation_mode": config.generation_mode.value,
                "teacher_model": config.teacher_model,
            },
        )
        self.dataset_store.upsert(dataset)
        
        job.output_dataset_id = dataset_id
        return job
    
    def cancel(self):
        """Request cancellation of running job."""
        self._cancel_requested = True
    
    async def _get_documents(self, config: BookDistillationConfig) -> List:
        """Get documents based on config."""
        if config.document_ids:
            documents = []
            for doc_id in config.document_ids:
                doc = await self.document_manager.get_document(doc_id)
                if doc:
                    documents.append(doc)
            return documents
        else:
            documents, _ = await self.document_manager.list_documents(
                domain_id=config.domain_id,
                limit=1000
            )
            return documents
    
    async def _distill_chunk(
        self,
        chunk,
        config: BookDistillationConfig
    ) -> List[Dict[str, Any]]:
        """
        Distill a single chunk into training examples.
        """
        mode = config.generation_mode
        examples = []
        
        if mode == BookGenerationMode.QA_GENERATION:
            examples = await self._generate_qa(chunk, config)
        elif mode == BookGenerationMode.SUMMARIZATION:
            examples = await self._generate_summary(chunk, config)
        elif mode == BookGenerationMode.EXPLANATION:
            examples = await self._generate_explanation(chunk, config)
        elif mode == BookGenerationMode.MIXED:
            qa = await self._generate_qa(chunk, config)
            summary = await self._generate_summary(chunk, config)
            explanation = await self._generate_explanation(chunk, config)
            examples = qa + summary + explanation
        
        return examples
    
    async def _generate_qa(
        self,
        chunk,
        config: BookDistillationConfig
    ) -> List[Dict[str, Any]]:
        """Generate Q&A pairs from a chunk."""
        prompt = QA_GENERATION_PROMPT.format(
            n=config.questions_per_chunk,
            content=chunk.content
        )
        
        try:
            result = await self._teacher_client.generate(
                prompt=prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            
            # Parse JSON response
            qa_pairs = self._parse_json_response(result.content)
            
            examples = []
            for qa in qa_pairs:
                if isinstance(qa, dict) and "question" in qa and "answer" in qa:
                    example = {
                        "instruction": qa["question"],
                        "input": "",
                        "output": qa["answer"],
                    }
                    if config.include_source_context:
                        example["input"] = f"Based on the following context:\n\n{chunk.content[:500]}..."
                    examples.append(example)
            
            return examples
            
        except Exception as e:
            logger.warning(f"Q&A generation failed: {e}")
            return []
    
    async def _generate_summary(
        self,
        chunk,
        config: BookDistillationConfig
    ) -> List[Dict[str, Any]]:
        """Generate summary from a chunk."""
        prompt = SUMMARIZATION_PROMPT.format(content=chunk.content)
        
        try:
            result = await self._teacher_client.generate(
                prompt=prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            
            parsed = self._parse_json_response(result.content)
            
            if isinstance(parsed, dict) and "instruction" in parsed and "response" in parsed:
                return [{
                    "instruction": parsed["instruction"],
                    "input": chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content,
                    "output": parsed["response"],
                }]
            
            return []
            
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return []
    
    async def _generate_explanation(
        self,
        chunk,
        config: BookDistillationConfig
    ) -> List[Dict[str, Any]]:
        """Generate explanations from a chunk."""
        prompt = EXPLANATION_PROMPT.format(content=chunk.content)
        
        try:
            result = await self._teacher_client.generate(
                prompt=prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            
            parsed = self._parse_json_response(result.content)
            
            examples = []
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "instruction" in item and "response" in item:
                        examples.append({
                            "instruction": item["instruction"],
                            "input": "",
                            "output": item["response"],
                        })
            elif isinstance(parsed, dict) and "instruction" in parsed:
                examples.append({
                    "instruction": parsed["instruction"],
                    "input": "",
                    "output": parsed.get("response", ""),
                })
            
            return examples
            
        except Exception as e:
            logger.warning(f"Explanation generation failed: {e}")
            return []
    
    def _parse_json_response(self, content: str) -> Any:
        """Parse JSON from LLM response, handling common issues."""
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON array or object
        for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
            match = re.search(pattern, content)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        
        logger.warning(f"Failed to parse JSON from response: {content[:200]}...")
        return []


# =============================================================================
# Job Manager
# =============================================================================

class BookDistillationManager:
    """Manages book distillation jobs."""
    
    def __init__(self, document_manager, dataset_store: DatasetStore):
        self.document_manager = document_manager
        self.dataset_store = dataset_store
        self._jobs: Dict[str, BookDistillationJob] = {}
        self._generators: Dict[str, BookDistillationGenerator] = {}
    
    def create_job(self, config: BookDistillationConfig) -> BookDistillationJob:
        """Create a new distillation job."""
        job = BookDistillationJob(config=config)
        self._jobs[job.id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[BookDistillationJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)
    
    def list_jobs(self) -> List[BookDistillationJob]:
        """List all jobs."""
        return list(self._jobs.values())
    
    async def start_job(
        self,
        job_id: str,
        progress_callback: Optional[callable] = None
    ) -> BookDistillationJob:
        """Start a distillation job."""
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        generator = BookDistillationGenerator(self.document_manager, self.dataset_store)
        self._generators[job_id] = generator
        
        try:
            await generator.run_and_save(job, progress_callback)
        finally:
            del self._generators[job_id]
        
        return job
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        generator = self._generators.get(job_id)
        if generator:
            generator.cancel()
            return True
        return False
    
    async def estimate_job(self, config: BookDistillationConfig) -> Dict[str, Any]:
        """Estimate job size and cost."""
        generator = BookDistillationGenerator(self.document_manager, self.dataset_store)
        return await generator.estimate(config)
    
    async def preview_job(
        self,
        config: BookDistillationConfig,
        num_chunks: int = 1
    ) -> List[Dict[str, Any]]:
        """Preview distillation results."""
        generator = BookDistillationGenerator(self.document_manager, self.dataset_store)
        return await generator.preview(config, num_chunks)
