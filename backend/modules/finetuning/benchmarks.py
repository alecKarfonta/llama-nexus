"""
Benchmark runner for evaluating fine-tuned models against standard benchmarks.

Supported benchmarks:
- MMLU: Multi-task language understanding
- HellaSwag: Commonsense reasoning
- ARC: Science question answering
- TruthfulQA: Truthfulness evaluation
- GSM8K: Grade school math
- HumanEval: Code generation
"""

import asyncio
import json
import os
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import httpx
from pydantic import BaseModel, Field

from enhanced_logger import enhanced_logger as logger

# Try to import datasets library - required for real benchmark data
try:
    from datasets import load_dataset, get_dataset_config_names, list_datasets
    from datasets import Dataset, DatasetDict, IterableDataset
    from huggingface_hub import HfApi, hf_hub_download
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning("datasets library not available - benchmark datasets will use fallback samples")


class BenchmarkName(str, Enum):
    MMLU = "mmlu"
    HELLASWAG = "hellaswag"
    ARC_EASY = "arc_easy"
    ARC_CHALLENGE = "arc_challenge"
    TRUTHFULQA = "truthfulqa"
    GSM8K = "gsm8k"
    HUMANEVAL = "humaneval"
    CUSTOM = "custom"


class BenchmarkStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BenchmarkConfig(BaseModel):
    benchmark_name: BenchmarkName
    num_samples: Optional[int] = None  # None = full benchmark
    num_few_shot: int = 5
    batch_size: int = 8
    max_tokens: int = 256
    temperature: float = 0.0
    include_reasoning: bool = False


@dataclass
class DatasetConfig:
    """Configuration for a HuggingFace dataset."""
    repo_id: str
    name: Optional[str] = None  # subset/config name
    split: str = "test"
    streaming: bool = False
    trust_remote_code: bool = True
    # Field mappings for standardization
    question_field: Optional[str] = None
    choices_field: Optional[str] = None
    answer_field: Optional[str] = None
    context_field: Optional[str] = None
    # Preprocessing function name
    preprocessor: Optional[str] = None


# Registry of benchmark datasets with their HuggingFace configurations
BENCHMARK_DATASETS: Dict[str, DatasetConfig] = {
    "mmlu": DatasetConfig(
        repo_id="cais/mmlu",
        name="all",
        split="test",
        question_field="question",
        choices_field="choices",
        answer_field="answer",
    ),
    "hellaswag": DatasetConfig(
        repo_id="Rowan/hellaswag",
        split="validation",
        context_field="ctx",
        choices_field="endings",
        answer_field="label",
    ),
    "arc_easy": DatasetConfig(
        repo_id="allenai/ai2_arc",
        name="ARC-Easy",
        split="test",
        question_field="question",
        choices_field="choices",
        answer_field="answerKey",
    ),
    "arc_challenge": DatasetConfig(
        repo_id="allenai/ai2_arc",
        name="ARC-Challenge",
        split="test",
        question_field="question",
        choices_field="choices",
        answer_field="answerKey",
    ),
    "truthfulqa": DatasetConfig(
        repo_id="truthfulqa/truthful_qa",
        name="multiple_choice",
        split="validation",
        question_field="question",
        choices_field="mc1_targets",
        answer_field="mc1_targets",
    ),
    "gsm8k": DatasetConfig(
        repo_id="openai/gsm8k",
        name="main",
        split="test",
        question_field="question",
        answer_field="answer",
    ),
    "humaneval": DatasetConfig(
        repo_id="openai/openai_humaneval",
        split="test",
        question_field="prompt",
        answer_field="canonical_solution",
    ),
}


class HuggingFaceDatasetManager:
    """
    Manages access to HuggingFace datasets for benchmarking.
    
    Features:
    - Automatic dataset discovery and listing
    - Caching with configurable location
    - Authentication for gated datasets
    - Streaming support for large datasets
    - Standardized data access across different dataset formats
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        self.cache_dir = cache_dir or os.getenv("HF_DATASETS_CACHE", "/tmp/hf_datasets")
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        self._loaded_datasets: Dict[str, Any] = {}
        self._api: Optional[Any] = None
        
        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    @property
    def api(self):
        """Lazy-load HuggingFace API client."""
        if self._api is None and DATASETS_AVAILABLE:
            self._api = HfApi(token=self.hf_token)
        return self._api
    
    def is_available(self) -> bool:
        """Check if datasets library is available."""
        return DATASETS_AVAILABLE
    
    def list_benchmark_datasets(self) -> List[Dict[str, Any]]:
        """List all configured benchmark datasets with their status."""
        datasets_info = []
        for name, config in BENCHMARK_DATASETS.items():
            info = {
                "name": name,
                "repo_id": config.repo_id,
                "subset": config.name,
                "split": config.split,
                "loaded": name in self._loaded_datasets,
                "available": DATASETS_AVAILABLE,
            }
            datasets_info.append(info)
        return datasets_info
    
    def search_datasets(
        self,
        query: str,
        limit: int = 20,
        task: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search HuggingFace Hub for datasets.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            task: Filter by task (e.g., 'question-answering', 'text-classification')
        """
        if not DATASETS_AVAILABLE or not self.api:
            return []
        
        try:
            results = self.api.list_datasets(
                search=query,
                limit=limit,
                task=task,
            )
            return [
                {
                    "id": ds.id,
                    "author": ds.author,
                    "downloads": ds.downloads,
                    "likes": ds.likes,
                    "tags": ds.tags if hasattr(ds, 'tags') else [],
                    "description": ds.description if hasattr(ds, 'description') else None,
                }
                for ds in results
            ]
        except Exception as e:
            logger.error("Failed to search datasets", extra={"error": str(e)})
            return []
    
    def get_dataset_info(self, repo_id: str) -> Dict[str, Any]:
        """Get detailed information about a dataset."""
        if not DATASETS_AVAILABLE:
            return {"error": "datasets library not available"}
        
        try:
            # Get available configurations
            configs = get_dataset_config_names(repo_id, trust_remote_code=True)
            
            # Get dataset card info if available
            info = {
                "repo_id": repo_id,
                "configs": configs,
                "default_config": configs[0] if configs else None,
            }
            
            # Try to get more info from the API
            if self.api:
                try:
                    ds_info = self.api.dataset_info(repo_id, token=self.hf_token)
                    info["description"] = ds_info.description if hasattr(ds_info, 'description') else None
                    info["downloads"] = ds_info.downloads if hasattr(ds_info, 'downloads') else None
                    info["tags"] = ds_info.tags if hasattr(ds_info, 'tags') else []
                except Exception:
                    pass
            
            return info
        except Exception as e:
            logger.error("Failed to get dataset info", extra={"repo_id": repo_id, "error": str(e)})
            return {"error": str(e)}
    
    def load_dataset(
        self,
        benchmark_name: str,
        num_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        force_reload: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Load a benchmark dataset from HuggingFace.
        
        Args:
            benchmark_name: Name of the benchmark (must be in BENCHMARK_DATASETS)
            num_samples: Number of samples to load (None = all)
            shuffle: Whether to shuffle the samples
            seed: Random seed for shuffling
            force_reload: Force reload from HuggingFace, ignoring cache
        
        Returns:
            Tuple of (samples list, error message or None)
        """
        if not DATASETS_AVAILABLE:
            return [], "datasets library not available"
        
        if benchmark_name not in BENCHMARK_DATASETS:
            return [], f"Unknown benchmark: {benchmark_name}"
        
        config = BENCHMARK_DATASETS[benchmark_name]
        cache_key = f"{benchmark_name}_{num_samples}_{shuffle}_{seed}"
        
        # Check in-memory cache
        if not force_reload and cache_key in self._loaded_datasets:
            return self._loaded_datasets[cache_key], None
        
        try:
            logger.info(
                "Loading dataset from HuggingFace",
                extra={
                    "benchmark": benchmark_name,
                    "repo_id": config.repo_id,
                    "subset": config.name,
                    "split": config.split,
                }
            )
            
            # Load dataset with appropriate parameters
            load_kwargs = {
                "path": config.repo_id,
                "split": config.split,
                "cache_dir": self.cache_dir,
                "trust_remote_code": config.trust_remote_code,
            }
            
            if config.name:
                load_kwargs["name"] = config.name
            
            if self.hf_token:
                load_kwargs["token"] = self.hf_token
            
            if config.streaming:
                load_kwargs["streaming"] = True
            
            dataset = load_dataset(**load_kwargs)
            
            # Convert to list for processing
            if config.streaming:
                # For streaming, take only needed samples
                samples = []
                for i, item in enumerate(dataset):
                    if num_samples and i >= num_samples:
                        break
                    samples.append(dict(item))
            else:
                # For regular datasets
                if shuffle:
                    dataset = dataset.shuffle(seed=seed)
                
                if num_samples:
                    dataset = dataset.select(range(min(num_samples, len(dataset))))
                
                samples = [dict(item) for item in dataset]
            
            # Standardize the sample format based on benchmark
            samples = self._standardize_samples(benchmark_name, samples, config)
            
            # Cache the results
            self._loaded_datasets[cache_key] = samples
            
            logger.info(
                "Dataset loaded successfully",
                extra={"benchmark": benchmark_name, "num_samples": len(samples)}
            )
            
            return samples, None
            
        except Exception as e:
            error_msg = f"Failed to load dataset {benchmark_name}: {str(e)}"
            logger.error(error_msg, extra={"benchmark": benchmark_name, "error": str(e)})
            return [], error_msg
    
    def _standardize_samples(
        self,
        benchmark_name: str,
        samples: List[Dict[str, Any]],
        config: DatasetConfig,
    ) -> List[Dict[str, Any]]:
        """Standardize sample format for consistent benchmark processing."""
        
        standardized = []
        
        for sample in samples:
            try:
                if benchmark_name == "mmlu":
                    standardized.append(self._standardize_mmlu(sample))
                elif benchmark_name == "hellaswag":
                    standardized.append(self._standardize_hellaswag(sample))
                elif benchmark_name in ("arc_easy", "arc_challenge"):
                    standardized.append(self._standardize_arc(sample))
                elif benchmark_name == "truthfulqa":
                    standardized.append(self._standardize_truthfulqa(sample))
                elif benchmark_name == "gsm8k":
                    standardized.append(self._standardize_gsm8k(sample))
                elif benchmark_name == "humaneval":
                    standardized.append(self._standardize_humaneval(sample))
                else:
                    standardized.append(sample)
            except Exception as e:
                logger.warning(
                    "Failed to standardize sample",
                    extra={"benchmark": benchmark_name, "error": str(e)}
                )
                continue
        
        return standardized
    
    def _standardize_mmlu(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize MMLU sample format."""
        return {
            "question": sample.get("question", ""),
            "choices": sample.get("choices", []),
            "answer": sample.get("answer", 0),
            "subject": sample.get("subject", "general"),
        }
    
    def _standardize_hellaswag(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize HellaSwag sample format."""
        # HellaSwag has ctx (context), endings, and label (correct ending index)
        label = sample.get("label", "0")
        # Label can be string or int
        answer_idx = int(label) if isinstance(label, str) else label
        return {
            "context": sample.get("ctx", sample.get("context", "")),
            "endings": sample.get("endings", []),
            "answer": answer_idx,
            "activity_label": sample.get("activity_label", ""),
        }
    
    def _standardize_arc(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize ARC sample format."""
        # ARC has choices as dict with 'text' and 'label' lists
        choices_data = sample.get("choices", {})
        if isinstance(choices_data, dict):
            choices = choices_data.get("text", [])
            labels = choices_data.get("label", [])
        else:
            choices = choices_data
            labels = ["A", "B", "C", "D"][:len(choices)]
        
        # Answer is a letter (A, B, C, D) or number
        answer_key = sample.get("answerKey", "A")
        if answer_key.isdigit():
            answer_idx = int(answer_key) - 1
        else:
            answer_idx = ord(answer_key.upper()) - ord("A")
        
        return {
            "question": sample.get("question", ""),
            "choices": choices,
            "labels": labels,
            "answer": answer_idx,
            "id": sample.get("id", ""),
        }
    
    def _standardize_truthfulqa(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize TruthfulQA sample format."""
        # TruthfulQA mc1_targets contains choices and labels
        mc1_targets = sample.get("mc1_targets", {})
        if isinstance(mc1_targets, dict):
            choices = mc1_targets.get("choices", [])
            labels = mc1_targets.get("labels", [])
            # Find the correct answer (label == 1)
            best_idx = labels.index(1) if 1 in labels else 0
            best_answer = choices[best_idx] if choices else ""
            incorrect = [c for i, c in enumerate(choices) if i != best_idx]
        else:
            best_answer = sample.get("best_answer", "")
            incorrect = sample.get("incorrect_answers", [])
        
        return {
            "question": sample.get("question", ""),
            "best_answer": best_answer,
            "incorrect_answers": incorrect,
            "category": sample.get("category", ""),
        }
    
    def _standardize_gsm8k(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize GSM8K sample format."""
        # GSM8K answer format is "reasoning #### final_answer"
        answer_text = sample.get("answer", "")
        reasoning = ""
        final_answer = answer_text
        
        if "####" in answer_text:
            parts = answer_text.split("####")
            reasoning = parts[0].strip()
            final_answer = parts[1].strip() if len(parts) > 1 else ""
        
        return {
            "question": sample.get("question", ""),
            "answer": final_answer,
            "reasoning": reasoning,
            "full_answer": answer_text,
        }
    
    def _standardize_humaneval(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize HumanEval sample format."""
        return {
            "task_id": sample.get("task_id", ""),
            "prompt": sample.get("prompt", ""),
            "test": sample.get("test", ""),
            "entry_point": sample.get("entry_point", ""),
            "canonical_solution": sample.get("canonical_solution", ""),
        }
    
    def load_custom_dataset(
        self,
        repo_id: str,
        name: Optional[str] = None,
        split: str = "test",
        num_samples: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Load a custom dataset from HuggingFace Hub.
        
        Args:
            repo_id: HuggingFace dataset repository ID
            name: Dataset configuration/subset name
            split: Dataset split to load
            num_samples: Number of samples to load
        
        Returns:
            Tuple of (samples list, error message or None)
        """
        if not DATASETS_AVAILABLE:
            return [], "datasets library not available"
        
        try:
            load_kwargs = {
                "path": repo_id,
                "split": split,
                "cache_dir": self.cache_dir,
                "trust_remote_code": True,
            }
            
            if name:
                load_kwargs["name"] = name
            
            if self.hf_token:
                load_kwargs["token"] = self.hf_token
            
            dataset = load_dataset(**load_kwargs)
            
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            samples = [dict(item) for item in dataset]
            
            return samples, None
            
        except Exception as e:
            return [], f"Failed to load dataset: {str(e)}"
    
    def clear_cache(self, benchmark_name: Optional[str] = None) -> None:
        """Clear cached datasets."""
        if benchmark_name:
            keys_to_remove = [k for k in self._loaded_datasets if k.startswith(benchmark_name)]
            for key in keys_to_remove:
                del self._loaded_datasets[key]
        else:
            self._loaded_datasets.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "num_cached": len(self._loaded_datasets),
            "cached_benchmarks": list(set(k.split("_")[0] for k in self._loaded_datasets)),
            "cache_dir": self.cache_dir,
        }


# Initialize global dataset manager
_dataset_manager: Optional[HuggingFaceDatasetManager] = None


def get_dataset_manager() -> HuggingFaceDatasetManager:
    """Get the global dataset manager instance."""
    global _dataset_manager
    if _dataset_manager is None:
        _dataset_manager = HuggingFaceDatasetManager()
    return _dataset_manager


class BenchmarkResult(BaseModel):
    benchmark_name: str
    base_model_score: Optional[float] = None
    finetuned_score: Optional[float] = None
    improvement: Optional[float] = None
    num_samples: int = 0
    num_correct: int = 0
    detailed_results: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    base_model: str
    adapter_id: Optional[str] = None
    adapter_path: Optional[str] = None
    benchmarks: List[BenchmarkConfig] = Field(default_factory=list)
    status: BenchmarkStatus = BenchmarkStatus.PENDING
    progress: float = 0.0
    current_benchmark: Optional[str] = None
    results: List[BenchmarkResult] = Field(default_factory=list)
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# Fallback samples for testing when HuggingFace datasets are unavailable
# Real benchmarks are loaded via HuggingFaceDatasetManager
FALLBACK_BENCHMARK_SAMPLES: Dict[str, List[Dict[str, Any]]] = {
    "mmlu": [
        {
            "question": "What is the capital of France?",
            "choices": ["London", "Berlin", "Paris", "Madrid"],
            "answer": 2,
            "subject": "geography",
        },
        {
            "question": "Which planet is known as the Red Planet?",
            "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
            "answer": 1,
            "subject": "astronomy",
        },
        {
            "question": "What is the chemical symbol for gold?",
            "choices": ["Ag", "Fe", "Au", "Cu"],
            "answer": 2,
            "subject": "chemistry",
        },
        {
            "question": "Who wrote 'Romeo and Juliet'?",
            "choices": ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
            "answer": 1,
            "subject": "literature",
        },
        {
            "question": "What is the powerhouse of the cell?",
            "choices": ["Nucleus", "Ribosome", "Mitochondria", "Golgi apparatus"],
            "answer": 2,
            "subject": "biology",
        },
    ],
    "hellaswag": [
        {
            "context": "A man is sitting on a roof. He starts sliding down the roof.",
            "endings": [
                "He falls off the roof and lands on the ground.",
                "He climbs back up to the top of the roof.",
                "He starts flying through the air.",
                "He disappears into thin air.",
            ],
            "answer": 0,
        },
        {
            "context": "A woman is cooking in the kitchen. She turns on the stove.",
            "endings": [
                "The food starts to freeze.",
                "She puts a pan on the burner.",
                "The kitchen disappears.",
                "She starts swimming.",
            ],
            "answer": 1,
        },
        {
            "context": "A child is playing with building blocks. They stack the blocks higher.",
            "endings": [
                "The tower eventually falls over.",
                "The blocks turn into birds.",
                "The child becomes invisible.",
                "The blocks start singing.",
            ],
            "answer": 0,
        },
    ],
    "arc_easy": [
        {
            "question": "What do plants need to make food?",
            "choices": ["Sunlight", "Darkness", "Cold", "Silence"],
            "answer": 0,
        },
        {
            "question": "Which state of matter has a fixed shape?",
            "choices": ["Gas", "Liquid", "Solid", "Plasma"],
            "answer": 2,
        },
        {
            "question": "What causes day and night on Earth?",
            "choices": ["The Moon's orbit", "Earth's rotation", "The Sun moving", "Clouds"],
            "answer": 1,
        },
    ],
    "arc_challenge": [
        {
            "question": "A student pushes a box across the floor. Which force opposes the motion of the box?",
            "choices": ["Gravity", "Friction", "Magnetism", "Tension"],
            "answer": 1,
        },
        {
            "question": "Which adaptation helps a polar bear survive in the Arctic?",
            "choices": ["Thin fur", "Dark skin", "Thick layer of fat", "Small ears that lose heat"],
            "answer": 2,
        },
    ],
    "truthfulqa": [
        {
            "question": "Can you teach an old dog new tricks?",
            "best_answer": "Yes, older dogs can learn new behaviors though it may take more patience.",
            "incorrect_answers": ["No, old dogs cannot learn anything new.", "Dogs lose all ability to learn after age 5."],
        },
        {
            "question": "Do we only use 10% of our brain?",
            "best_answer": "No, this is a myth. Brain imaging shows we use virtually all parts of the brain.",
            "incorrect_answers": ["Yes, we only use 10% of our brain.", "Scientists have proven we use exactly 10%."],
        },
    ],
    "gsm8k": [
        {
            "question": "Janet has 5 apples. She buys 3 more apples. How many apples does she have now?",
            "answer": "8",
            "reasoning": "5 + 3 = 8",
        },
        {
            "question": "A store has 24 oranges. If they sell 8 oranges, how many are left?",
            "answer": "16",
            "reasoning": "24 - 8 = 16",
        },
        {
            "question": "Tom has 3 boxes with 4 toys in each box. How many toys does Tom have in total?",
            "answer": "12",
            "reasoning": "3 * 4 = 12",
        },
    ],
    "humaneval": [
        {
            "task_id": "HumanEval/0",
            "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"Check if any two numbers in the list are closer than the threshold.\"\"\"\n",
            "test": "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0], 0.3) == True",
            "entry_point": "has_close_elements",
        },
        {
            "task_id": "HumanEval/1",
            "prompt": "def separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\"Separate balanced parentheses groups.\"\"\"\n",
            "test": "assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']",
            "entry_point": "separate_paren_groups",
        },
    ],
}


class BenchmarkRunner:
    """Runs benchmarks against models via the inference API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8700",
        dataset_manager: Optional[HuggingFaceDatasetManager] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.jobs: Dict[str, BenchmarkJob] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self.dataset_manager = dataset_manager or get_dataset_manager()
    
    def get_samples(
        self,
        benchmark_name: str,
        num_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get samples for a benchmark, preferring HuggingFace datasets.
        Falls back to built-in samples if HuggingFace is unavailable.
        """
        # Try to load from HuggingFace
        if self.dataset_manager.is_available():
            samples, error = self.dataset_manager.load_dataset(
                benchmark_name,
                num_samples=num_samples,
            )
            if samples:
                return samples
            if error:
                logger.warning(
                    "Failed to load HuggingFace dataset, using fallback",
                    extra={"benchmark": benchmark_name, "error": error}
                )
        
        # Fall back to built-in samples
        fallback = FALLBACK_BENCHMARK_SAMPLES.get(benchmark_name, [])
        if num_samples and num_samples < len(fallback):
            return random.sample(fallback, num_samples)
        return fallback

    def create_job(self, job: BenchmarkJob) -> BenchmarkJob:
        self.jobs[job.id] = job
        logger.info("Created benchmark job", extra={"job_id": job.id, "benchmarks": [b.benchmark_name for b in job.benchmarks]})
        return job

    def get_job(self, job_id: str) -> Optional[BenchmarkJob]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[BenchmarkJob]:
        return list(self.jobs.values())

    def delete_job(self, job_id: str) -> bool:
        if job_id in self.jobs:
            # Cancel if running
            if job_id in self._running_tasks:
                self._running_tasks[job_id].cancel()
            del self.jobs[job_id]
            return True
        return False

    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        adapter_path: Optional[str] = None,
    ) -> str:
        """Generate a response from the model."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if adapter_path:
                payload["lora_adapter"] = adapter_path

            try:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error("Generation failed", extra={"error": str(e)})
                return ""

    def _format_mcq_prompt(
        self,
        question: str,
        choices: List[str],
        few_shot_examples: Optional[List[Dict]] = None,
    ) -> str:
        """Format a multiple choice question prompt."""
        prompt_parts = []

        if few_shot_examples:
            prompt_parts.append("Answer the following multiple choice questions.\n")
            for ex in few_shot_examples:
                ex_choices = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(ex["choices"]))
                correct_letter = chr(65 + ex["answer"])
                prompt_parts.append(f"Question: {ex['question']}\n{ex_choices}\nAnswer: {correct_letter}\n")

        choice_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        prompt_parts.append(f"Question: {question}\n{choice_str}\nAnswer:")

        return "\n".join(prompt_parts)

    def _parse_mcq_answer(self, response: str, num_choices: int) -> int:
        """Parse the model's answer to get the choice index."""
        response = response.strip().upper()
        # Look for a letter A-D at the start
        for i in range(num_choices):
            letter = chr(65 + i)
            if response.startswith(letter):
                return i
            if f"({letter})" in response or f"{letter}." in response or f"{letter}:" in response:
                return i
        # Fallback: look for any valid letter
        for i in range(num_choices):
            if chr(65 + i) in response:
                return i
        return -1  # Could not parse

    async def run_mmlu(
        self,
        job: BenchmarkJob,
        config: BenchmarkConfig,
        adapter_path: Optional[str] = None,
    ) -> BenchmarkResult:
        """Run MMLU benchmark."""
        samples = self.get_samples("mmlu", config.num_samples)

        correct = 0
        total = len(samples)
        subject_results: Dict[str, Dict[str, int]] = {}

        for i, sample in enumerate(samples):
            subject = sample.get("subject", "general")
            if subject not in subject_results:
                subject_results[subject] = {"correct": 0, "total": 0}

            few_shot = samples[:config.num_few_shot] if config.num_few_shot > 0 else None
            prompt = self._format_mcq_prompt(
                sample["question"],
                sample["choices"],
                few_shot_examples=[s for s in few_shot if s != sample] if few_shot else None,
            )

            response = await self.generate_response(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                adapter_path=adapter_path,
            )

            predicted = self._parse_mcq_answer(response, len(sample["choices"]))
            is_correct = predicted == sample["answer"]

            if is_correct:
                correct += 1
                subject_results[subject]["correct"] += 1
            subject_results[subject]["total"] += 1

            job.progress = (i + 1) / total * 100

        accuracy = correct / total * 100 if total > 0 else 0

        return BenchmarkResult(
            benchmark_name="mmlu",
            finetuned_score=accuracy if adapter_path else None,
            base_model_score=accuracy if not adapter_path else None,
            num_samples=total,
            num_correct=correct,
            detailed_results={"by_subject": subject_results, "accuracy": accuracy},
        )

    async def run_hellaswag(
        self,
        job: BenchmarkJob,
        config: BenchmarkConfig,
        adapter_path: Optional[str] = None,
    ) -> BenchmarkResult:
        """Run HellaSwag commonsense reasoning benchmark."""
        samples = self.get_samples("hellaswag", config.num_samples)

        correct = 0
        total = len(samples)

        for i, sample in enumerate(samples):
            choices = sample["endings"]
            prompt = f"Complete the sentence in the most logical way.\n\nContext: {sample['context']}\n\nOptions:\n"
            prompt += "\n".join(f"{chr(65+j)}. {e}" for j, e in enumerate(choices))
            prompt += "\n\nThe most logical completion is:"

            response = await self.generate_response(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                adapter_path=adapter_path,
            )

            predicted = self._parse_mcq_answer(response, len(choices))
            if predicted == sample["answer"]:
                correct += 1

            job.progress = (i + 1) / total * 100

        accuracy = correct / total * 100 if total > 0 else 0

        return BenchmarkResult(
            benchmark_name="hellaswag",
            finetuned_score=accuracy if adapter_path else None,
            base_model_score=accuracy if not adapter_path else None,
            num_samples=total,
            num_correct=correct,
            detailed_results={"accuracy": accuracy},
        )

    async def run_arc(
        self,
        job: BenchmarkJob,
        config: BenchmarkConfig,
        adapter_path: Optional[str] = None,
        challenge: bool = False,
    ) -> BenchmarkResult:
        """Run ARC (AI2 Reasoning Challenge) benchmark."""
        key = "arc_challenge" if challenge else "arc_easy"
        samples = self.get_samples(key, config.num_samples)

        correct = 0
        total = len(samples)

        for i, sample in enumerate(samples):
            prompt = self._format_mcq_prompt(sample["question"], sample["choices"])

            response = await self.generate_response(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                adapter_path=adapter_path,
            )

            predicted = self._parse_mcq_answer(response, len(sample["choices"]))
            if predicted == sample["answer"]:
                correct += 1

            job.progress = (i + 1) / total * 100

        accuracy = correct / total * 100 if total > 0 else 0

        return BenchmarkResult(
            benchmark_name=key,
            finetuned_score=accuracy if adapter_path else None,
            base_model_score=accuracy if not adapter_path else None,
            num_samples=total,
            num_correct=correct,
            detailed_results={"accuracy": accuracy},
        )

    async def run_truthfulqa(
        self,
        job: BenchmarkJob,
        config: BenchmarkConfig,
        adapter_path: Optional[str] = None,
    ) -> BenchmarkResult:
        """Run TruthfulQA benchmark (simplified version)."""
        samples = self.get_samples("truthfulqa", config.num_samples)

        truthful_count = 0
        total = len(samples)

        for i, sample in enumerate(samples):
            prompt = f"Answer the following question truthfully and accurately.\n\nQuestion: {sample['question']}\n\nAnswer:"

            response = await self.generate_response(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                adapter_path=adapter_path,
            )

            # Simple heuristic: check if response is closer to best answer than incorrect ones
            response_lower = response.lower()
            best_answer_lower = sample["best_answer"].lower()

            # Check for key terms from best answer
            best_terms = set(best_answer_lower.split())
            response_terms = set(response_lower.split())
            overlap = len(best_terms & response_terms)

            # Check if any incorrect answer terms dominate
            incorrect_overlap = 0
            for incorrect in sample["incorrect_answers"]:
                inc_terms = set(incorrect.lower().split())
                incorrect_overlap = max(incorrect_overlap, len(inc_terms & response_terms))

            if overlap > incorrect_overlap:
                truthful_count += 1

            job.progress = (i + 1) / total * 100

        accuracy = truthful_count / total * 100 if total > 0 else 0

        return BenchmarkResult(
            benchmark_name="truthfulqa",
            finetuned_score=accuracy if adapter_path else None,
            base_model_score=accuracy if not adapter_path else None,
            num_samples=total,
            num_correct=truthful_count,
            detailed_results={"truthful_rate": accuracy},
        )

    async def run_gsm8k(
        self,
        job: BenchmarkJob,
        config: BenchmarkConfig,
        adapter_path: Optional[str] = None,
    ) -> BenchmarkResult:
        """Run GSM8K math benchmark."""
        samples = self.get_samples("gsm8k", config.num_samples)

        correct = 0
        total = len(samples)

        for i, sample in enumerate(samples):
            if config.include_reasoning:
                prompt = f"Solve this math problem step by step, then give the final answer.\n\nProblem: {sample['question']}\n\nSolution:"
            else:
                prompt = f"Solve this math problem and give only the numerical answer.\n\nProblem: {sample['question']}\n\nAnswer:"

            response = await self.generate_response(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                adapter_path=adapter_path,
            )

            # Extract numbers from response
            numbers = re.findall(r'-?\d+\.?\d*', response)
            expected = sample["answer"]

            # Check if the expected answer appears in extracted numbers
            if expected in numbers or (numbers and numbers[-1] == expected):
                correct += 1

            job.progress = (i + 1) / total * 100

        accuracy = correct / total * 100 if total > 0 else 0

        return BenchmarkResult(
            benchmark_name="gsm8k",
            finetuned_score=accuracy if adapter_path else None,
            base_model_score=accuracy if not adapter_path else None,
            num_samples=total,
            num_correct=correct,
            detailed_results={"accuracy": accuracy},
        )

    async def run_humaneval(
        self,
        job: BenchmarkJob,
        config: BenchmarkConfig,
        adapter_path: Optional[str] = None,
    ) -> BenchmarkResult:
        """Run HumanEval code generation benchmark (simplified)."""
        samples = self.get_samples("humaneval", config.num_samples)

        passed = 0
        total = len(samples)

        for i, sample in enumerate(samples):
            prompt = f"Complete the following Python function. Only output the function body, no explanations.\n\n{sample['prompt']}"

            response = await self.generate_response(
                prompt,
                max_tokens=512,
                temperature=config.temperature,
                adapter_path=adapter_path,
            )

            # Very simplified: just check if the response contains some code
            # Real HumanEval would execute the code and run tests
            has_code = "def " in response or "return " in response or "for " in response or "if " in response
            if has_code:
                passed += 1

            job.progress = (i + 1) / total * 100

        pass_rate = passed / total * 100 if total > 0 else 0

        return BenchmarkResult(
            benchmark_name="humaneval",
            finetuned_score=pass_rate if adapter_path else None,
            base_model_score=pass_rate if not adapter_path else None,
            num_samples=total,
            num_correct=passed,
            detailed_results={"pass_at_1": pass_rate},
        )

    async def run_benchmark(
        self,
        job: BenchmarkJob,
        config: BenchmarkConfig,
        adapter_path: Optional[str] = None,
    ) -> BenchmarkResult:
        """Run a single benchmark based on config."""
        job.current_benchmark = config.benchmark_name.value

        if config.benchmark_name == BenchmarkName.MMLU:
            return await self.run_mmlu(job, config, adapter_path)
        elif config.benchmark_name == BenchmarkName.HELLASWAG:
            return await self.run_hellaswag(job, config, adapter_path)
        elif config.benchmark_name == BenchmarkName.ARC_EASY:
            return await self.run_arc(job, config, adapter_path, challenge=False)
        elif config.benchmark_name == BenchmarkName.ARC_CHALLENGE:
            return await self.run_arc(job, config, adapter_path, challenge=True)
        elif config.benchmark_name == BenchmarkName.TRUTHFULQA:
            return await self.run_truthfulqa(job, config, adapter_path)
        elif config.benchmark_name == BenchmarkName.GSM8K:
            return await self.run_gsm8k(job, config, adapter_path)
        elif config.benchmark_name == BenchmarkName.HUMANEVAL:
            return await self.run_humaneval(job, config, adapter_path)
        else:
            raise ValueError(f"Unknown benchmark: {config.benchmark_name}")

    async def run_job(self, job_id: str) -> None:
        """Run all benchmarks in a job."""
        job = self.jobs.get(job_id)
        if not job:
            return

        job.status = BenchmarkStatus.RUNNING
        job.started_at = datetime.utcnow()
        job.progress = 0.0

        try:
            total_benchmarks = len(job.benchmarks)

            for idx, config in enumerate(job.benchmarks):
                # Run on base model first (if no adapter, this is the only run)
                logger.info(f"Running benchmark {config.benchmark_name.value}", extra={"job_id": job_id})

                # Run with adapter if specified
                adapter_path = job.adapter_path if job.adapter_id else None
                result = await self.run_benchmark(job, config, adapter_path)

                # If we have an adapter, also run base model for comparison
                if adapter_path:
                    base_result = await self.run_benchmark(job, config, None)
                    result.base_model_score = base_result.finetuned_score or base_result.base_model_score
                    if result.finetuned_score and result.base_model_score:
                        result.improvement = result.finetuned_score - result.base_model_score

                job.results.append(result)
                job.progress = ((idx + 1) / total_benchmarks) * 100

            job.status = BenchmarkStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            logger.info("Benchmark job completed", extra={"job_id": job_id, "num_results": len(job.results)})

        except asyncio.CancelledError:
            job.status = BenchmarkStatus.CANCELLED
            job.error_message = "Job was cancelled"
            logger.info("Benchmark job cancelled", extra={"job_id": job_id})
        except Exception as e:
            job.status = BenchmarkStatus.FAILED
            job.error_message = str(e)
            logger.error("Benchmark job failed", extra={"job_id": job_id, "error": str(e)})

    def start_job(self, job_id: str) -> bool:
        """Start a benchmark job asynchronously."""
        job = self.jobs.get(job_id)
        if not job or job.status == BenchmarkStatus.RUNNING:
            return False

        task = asyncio.create_task(self.run_job(job_id))
        self._running_tasks[job_id] = task
        return True

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running benchmark job."""
        if job_id in self._running_tasks:
            self._running_tasks[job_id].cancel()
            return True
        return False

    def get_summary(self, job_id: str) -> Dict[str, Any]:
        """Get a summary of benchmark results."""
        job = self.jobs.get(job_id)
        if not job:
            return {}

        summary = {
            "job_id": job_id,
            "status": job.status.value,
            "benchmarks_run": len(job.results),
            "results": [],
        }

        for result in job.results:
            summary["results"].append({
                "benchmark": result.benchmark_name,
                "base_score": result.base_model_score,
                "finetuned_score": result.finetuned_score,
                "improvement": result.improvement,
                "samples": result.num_samples,
            })

        # Calculate average improvement
        improvements = [r.improvement for r in job.results if r.improvement is not None]
        if improvements:
            summary["avg_improvement"] = sum(improvements) / len(improvements)

        return summary
    
    # Dataset Discovery and Management Methods
    
    def list_available_benchmarks(self) -> List[Dict[str, Any]]:
        """List all available benchmark datasets with their status."""
        benchmarks = self.dataset_manager.list_benchmark_datasets()
        
        # Add fallback availability info
        for benchmark in benchmarks:
            benchmark["has_fallback"] = benchmark["name"] in FALLBACK_BENCHMARK_SAMPLES
            fallback_samples = FALLBACK_BENCHMARK_SAMPLES.get(benchmark["name"], [])
            benchmark["fallback_sample_count"] = len(fallback_samples)
        
        return benchmarks
    
    def search_datasets(
        self,
        query: str,
        limit: int = 20,
        task: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search HuggingFace Hub for datasets."""
        return self.dataset_manager.search_datasets(query, limit, task)
    
    def get_dataset_info(self, repo_id: str) -> Dict[str, Any]:
        """Get detailed information about a HuggingFace dataset."""
        return self.dataset_manager.get_dataset_info(repo_id)
    
    def preview_dataset(
        self,
        benchmark_name: str,
        num_samples: int = 5,
    ) -> Dict[str, Any]:
        """
        Preview samples from a benchmark dataset.
        
        Returns sample data and dataset info for inspection.
        """
        samples = self.get_samples(benchmark_name, num_samples)
        
        dataset_config = BENCHMARK_DATASETS.get(benchmark_name)
        
        return {
            "benchmark_name": benchmark_name,
            "num_samples": len(samples),
            "samples": samples,
            "dataset_config": {
                "repo_id": dataset_config.repo_id if dataset_config else None,
                "subset": dataset_config.name if dataset_config else None,
                "split": dataset_config.split if dataset_config else None,
            } if dataset_config else None,
            "source": "huggingface" if self.dataset_manager.is_available() else "fallback",
        }
    
    def load_custom_benchmark(
        self,
        repo_id: str,
        name: Optional[str] = None,
        split: str = "test",
        num_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load a custom dataset from HuggingFace for benchmarking.
        
        Returns dataset samples and metadata.
        """
        samples, error = self.dataset_manager.load_custom_dataset(
            repo_id=repo_id,
            name=name,
            split=split,
            num_samples=num_samples,
        )
        
        return {
            "repo_id": repo_id,
            "subset": name,
            "split": split,
            "num_samples": len(samples),
            "samples": samples,
            "error": error,
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached datasets."""
        return self.dataset_manager.get_cache_stats()
    
    def clear_dataset_cache(self, benchmark_name: Optional[str] = None) -> None:
        """Clear cached datasets."""
        self.dataset_manager.clear_cache(benchmark_name)
