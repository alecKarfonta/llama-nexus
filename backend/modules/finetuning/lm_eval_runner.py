"""
LM Evaluation Harness Runner

Wrapper for EleutherAI's lm-evaluation-harness CLI.
Provides standardized benchmark evaluation using industry-standard scoring.
"""

import asyncio
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from enhanced_logger import enhanced_logger as logger

# Try to import redis for GPU worker dispatch
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


class LMEvalStatus(str, Enum):
    """Status of an lm-eval benchmark job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class LMEvalTask:
    """Configuration for a single benchmark task."""
    name: str  # e.g., "mmlu", "hellaswag", "arc_easy"
    num_fewshot: int = 5
    limit: Optional[int] = None  # Sample limit for quick testing


@dataclass
class LMEvalJob:
    """A benchmark job using lm-evaluation-harness."""
    id: str
    tasks: List[LMEvalTask]
    model_path: Optional[str] = None  # Path to the GGUF model being evaluated
    gpu_device: Optional[int] = None  # GPU device used for evaluation
    status: LMEvalStatus = LMEvalStatus.PENDING
    progress: float = 0.0
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    log_output: str = ""


# Popular tasks supported by lm-eval
POPULAR_TASKS = {
    "mmlu": {
        "display_name": "MMLU",
        "description": "Massive Multitask Language Understanding - 57 subjects, tests knowledge",
        "default_fewshot": 5,
        "category": "knowledge",
    },
    "hellaswag": {
        "display_name": "HellaSwag",
        "description": "Commonsense reasoning about physical situations",
        "default_fewshot": 10,
        "category": "reasoning",
    },
    "arc_easy": {
        "display_name": "ARC Easy",
        "description": "AI2 Reasoning Challenge - easy science questions",
        "default_fewshot": 25,
        "category": "reasoning",
    },
    "arc_challenge": {
        "display_name": "ARC Challenge",
        "description": "AI2 Reasoning Challenge - hard science questions",
        "default_fewshot": 25,
        "category": "reasoning",
    },
    "truthfulqa_mc2": {
        "display_name": "TruthfulQA MC",
        "description": "Tests model truthfulness on tricky questions",
        "default_fewshot": 0,
        "category": "truthfulness",
    },
    "winogrande": {
        "display_name": "Winogrande",
        "description": "Commonsense reasoning with fill-in-the-blank",
        "default_fewshot": 5,
        "category": "reasoning",
    },
    "gsm8k": {
        "display_name": "GSM8K",
        "description": "Grade school math word problems",
        "default_fewshot": 5,
        "category": "math",
    },
    "boolq": {
        "display_name": "BoolQ",
        "description": "Yes/no reading comprehension questions",
        "default_fewshot": 0,
        "category": "reading",
    },
    "openbookqa": {
        "display_name": "OpenBookQA",
        "description": "Science knowledge with open-book reasoning",
        "default_fewshot": 0,
        "category": "knowledge",
    },
    "piqa": {
        "display_name": "PIQA",
        "description": "Physical intuition questions",
        "default_fewshot": 0,
        "category": "reasoning",
    },
}


class LMEvalRunner:
    """
    Wrapper for lm-evaluation-harness CLI.
    
    Uses the local-chat-completions model type to connect to our
    deployed llama.cpp server via the OpenAI-compatible API.
    """
    
    # Shared results directory with the lm-eval-worker
    SHARED_RESULTS_DIR = Path("/data/lm_eval_results")
    
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        output_dir: str = "/tmp/lm_eval_results",
    ):
        # Default to environment variables if not provided
        if endpoint_url is None:
            env_url = os.getenv("VLM_ENDPOINT_URL") or os.getenv("LLM_ENDPOINT_URL")
            if env_url:
                # Ensure we have the base URL without /v1
                endpoint_url = env_url.replace("/v1", "").rstrip("/")
            elif os.getenv("RUNNING_IN_DOCKER") == "true":
                endpoint_url = "http://llamacpp-api:8080"
            else:
                endpoint_url = "http://localhost:8700"
        
        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_key = api_key or os.getenv("VLM_API_KEY") or os.getenv("LLM_API_KEY")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.jobs: Dict[str, LMEvalJob] = {}
        self._running_processes: Dict[str, subprocess.Popen] = {}
        
        # Load historical results from shared filesystem on startup
        self._load_historical_results()
    
    def _load_historical_results(self):
        """
        Load completed benchmark results from the shared filesystem.
        
        The lm-eval-worker saves results to /data/lm_eval_results/lm_eval_{job_id}/results.json
        This allows results to persist across backend restarts.
        """
        if not self.SHARED_RESULTS_DIR.exists():
            logger.info("Shared results directory not found, skipping historical load")
            return
        
        loaded_count = 0
        for job_dir in self.SHARED_RESULTS_DIR.iterdir():
            if not job_dir.is_dir():
                continue
            
            # Extract job ID from directory name (format: lm_eval_{uuid})
            dir_name = job_dir.name
            if dir_name.startswith("lm_eval_"):
                job_id = dir_name[8:]  # Strip "lm_eval_" prefix
            else:
                continue
            
            results_file = job_dir / "results.json"
            if not results_file.exists():
                continue
            
            try:
                with open(results_file) as f:
                    raw_results = json.load(f)
                
                # Parse results into our format
                parsed = self._parse_results(raw_results)
                
                # Try to extract model path from lm-eval config.model_args
                # Format: "model_path=/path/to/model.gguf,n_ctx=2048,..."
                model_path = None
                config = raw_results.get("config", {})
                model_args = config.get("model_args", "")
                if model_args and "model_path=" in model_args:
                    for arg in model_args.split(","):
                        if arg.startswith("model_path="):
                            full_path = arg.split("=", 1)[1]
                            # Extract just the filename from the full path
                            model_path = Path(full_path).name
                            break
                
                # Fallback: try config.json file if exists
                if not model_path:
                    config_file = job_dir / "config.json"
                    if config_file.exists():
                        with open(config_file) as f:
                            config_data = json.load(f)
                            model_path = config_data.get("model_path")
                
                # Reconstruct job object
                # Get task names from results
                task_names = list(parsed.get("tasks", {}).keys())
                tasks = [LMEvalTask(name=t) for t in task_names if t]
                
                # Get created time from file mtime
                created_at = datetime.fromtimestamp(results_file.stat().st_mtime)
                
                job = LMEvalJob(
                    id=job_id,
                    tasks=tasks,
                    model_path=model_path or parsed.get("model_name") or "unknown",
                    status=LMEvalStatus.COMPLETED,
                    progress=100.0,
                    results=parsed,
                    created_at=created_at,
                    completed_at=created_at,
                )
                
                self.jobs[job_id] = job
                loaded_count += 1
                
            except Exception as e:
                logger.warning(
                    f"Failed to load historical result {job_dir.name}: {e}"
                )
                continue
        
        if loaded_count > 0:
            logger.info(
                f"Loaded {loaded_count} historical benchmark results from filesystem"
            )
    
    def get_available_tasks(self) -> List[Dict[str, Any]]:
        """Return list of available benchmark tasks."""
        return [
            {
                "name": name,
                **info,
            }
            for name, info in POPULAR_TASKS.items()
        ]
    
    def create_job(self, job: LMEvalJob) -> LMEvalJob:
        """Create a new benchmark job."""
        self.jobs[job.id] = job
        logger.info(
            "Created lm-eval benchmark job",
            extra={"job_id": job.id, "tasks": [t.name for t in job.tasks]}
        )
        return job
    
    def get_job(self, job_id: str) -> Optional[LMEvalJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)
    
    def list_jobs(self) -> List[LMEvalJob]:
        """List all jobs."""
        return list(self.jobs.values())
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job and cancel if running. Also removes persisted filesystem results."""
        deleted_memory = False
        deleted_fs = False
        
        # Delete from in-memory storage
        if job_id in self.jobs:
            if job_id in self._running_processes:
                self._running_processes[job_id].terminate()
                del self._running_processes[job_id]
            del self.jobs[job_id]
            deleted_memory = True
        
        # Delete from filesystem (persisted results)
        result_dir = self.SHARED_RESULTS_DIR / f"lm_eval_{job_id}"
        if result_dir.exists() and result_dir.is_dir():
            import shutil
            try:
                shutil.rmtree(result_dir)
                logger.info(f"Deleted benchmark results directory: {result_dir}")
                deleted_fs = True
            except Exception as e:
                logger.warning(f"Failed to delete results directory {result_dir}: {e}")
        
        return deleted_memory or deleted_fs
    
    async def run_benchmark(
        self,
        job: LMEvalJob,
        model_path: Optional[str] = None,
        gpu_device: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Run lm-eval benchmark via GPU worker and yield progress updates.
        
        Dispatches the job to the lm-eval-worker via Redis queue and
        listens for status updates via pub/sub.
        
        Args:
            job: The benchmark job configuration
            model_path: Optional path to GGUF model file (if not specified, worker uses first available)
            gpu_device: Optional GPU device index to use for evaluation
        
        Yields:
            Dict with progress updates and final results
        """
        job.status = LMEvalStatus.RUNNING
        job.progress = 0.0
        
        # Build job payload for worker
        task_names = [t.name for t in job.tasks]
        num_fewshot = job.tasks[0].num_fewshot if job.tasks else 5
        limit = job.tasks[0].limit if job.tasks else None
        
        job_payload = {
            "id": job.id,
            "tasks": task_names,
            "num_fewshot": num_fewshot,
            "limit": limit,
            "model_path": model_path,
            "gpu_device": gpu_device,
        }
        
        # Try Redis-based GPU worker dispatch
        if REDIS_AVAILABLE:
            try:
                async for update in self._run_via_redis(job, job_payload):
                    yield update
                return
            except Exception as e:
                logger.warning(
                    "Redis dispatch failed, cannot run benchmark",
                    extra={"job_id": job.id, "error": str(e)}
                )
                job.status = LMEvalStatus.FAILED
                job.error = f"GPU worker not available: {str(e)}"
                job.completed_at = datetime.now()
                yield {
                    "type": "error",
                    "job_id": job.id,
                    "error": job.error,
                }
        else:
            job.status = LMEvalStatus.FAILED
            job.error = "Redis not available for GPU worker dispatch"
            job.completed_at = datetime.now()
            yield {
                "type": "error",
                "job_id": job.id,
                "error": job.error,
            }
    
    async def _run_via_redis(
        self,
        job: LMEvalJob,
        job_payload: Dict[str, Any],
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Dispatch job to GPU worker via Redis and stream updates.
        """
        import json
        
        # Connect to Redis
        client = redis.from_url(REDIS_URL, decode_responses=True)
        pubsub = client.pubsub()
        
        # Subscribe to status and log channels for this job
        status_channel = f"lm_eval:status:{job.id}"
        log_channel = f"lm_eval:logs:{job.id}"
        pubsub.subscribe(status_channel, log_channel)
        
        logger.info(
            "Dispatching lm-eval job to GPU worker",
            extra={"job_id": job.id, "tasks": job_payload["tasks"]}
        )
        
        yield {
            "type": "status",
            "job_id": job.id,
            "status": "running",
            "message": f"Dispatching to GPU worker: {len(job_payload['tasks'])} task(s)...",
        }
        
        # Push job to the queue
        client.lpush("lm_eval:jobs", json.dumps(job_payload))
        
        # Listen for updates with timeout
        timeout = 3600  # 1 hour max
        start_time = asyncio.get_event_loop().time()
        
        try:
            while True:
                # Check timeout
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    raise TimeoutError("Benchmark timed out after 1 hour")
                
                # Get message with short timeout to allow async yields
                message = pubsub.get_message(timeout=0.5)
                
                if message and message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                    except json.JSONDecodeError:
                        continue
                    
                    channel = message["channel"]
                    
                    if channel == status_channel:
                        status = data.get("status")
                        progress = data.get("progress", 0)
                        msg = data.get("message", "")
                        
                        job.progress = progress
                        
                        if status == "completed":
                            job.status = LMEvalStatus.COMPLETED
                            job.results = data.get("results", {})
                            job.completed_at = datetime.now()
                            job.progress = 100.0
                            
                            logger.info(
                                "lm-eval benchmark completed",
                                extra={"job_id": job.id, "results": job.results}
                            )
                            
                            yield {
                                "type": "complete",
                                "job_id": job.id,
                                "results": job.results,
                            }
                            return
                        
                        elif status == "failed":
                            error = data.get("error", "Unknown error")
                            job.status = LMEvalStatus.FAILED
                            job.error = error
                            job.completed_at = datetime.now()
                            
                            logger.error(
                                "lm-eval benchmark failed",
                                extra={"job_id": job.id, "error": error}
                            )
                            
                            yield {
                                "type": "error",
                                "job_id": job.id,
                                "error": error,
                            }
                            return
                        
                        else:
                            yield {
                                "type": "progress",
                                "job_id": job.id,
                                "progress": progress,
                                "message": msg[:100],
                            }
                    
                    elif channel == log_channel:
                        msg = data.get("message", "")
                        job.log_output += msg + "\n"
                        yield {
                            "type": "log",
                            "job_id": job.id,
                            "message": msg[:200],
                        }
                
                # Small async sleep to prevent busy loop
                await asyncio.sleep(0.1)
        
        finally:
            pubsub.unsubscribe()
            pubsub.close()
    
    def _parse_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """Parse lm-eval results into a simpler format."""
        parsed = {
            "tasks": {},
            "model_name": raw_results.get("model_name", "unknown"),
            "date": raw_results.get("date", datetime.now().isoformat()),
        }
        
        results = raw_results.get("results", {})
        for task_name, task_results in results.items():
            # Extract key metrics
            metrics = {}
            for key, value in task_results.items():
                if isinstance(value, (int, float)):
                    # Strip lm-eval v0.4+ filter suffixes (e.g., "acc,none" -> "acc")
                    clean_key = key.split(",")[0]
                    # Skip stderr keys
                    if "stderr" in clean_key:
                        continue
                    # Convert to percentage if it's a ratio
                    if 0 <= value <= 1:
                        metrics[clean_key] = round(value * 100, 2)
                    else:
                        metrics[clean_key] = round(value, 4) if isinstance(value, float) else value
            
            # Get primary score (now using cleaned keys)
            primary_score = metrics.get("acc_norm") or metrics.get("acc") or metrics.get("exact_match") or 0
            
            parsed["tasks"][task_name] = {
                "score": primary_score,
                "metrics": metrics,
            }
        
        return parsed


# Singleton instance
_lm_eval_runner: Optional[LMEvalRunner] = None


def get_lm_eval_runner() -> LMEvalRunner:
    """Get or create the singleton LMEvalRunner instance."""
    global _lm_eval_runner
    if _lm_eval_runner is None:
        _lm_eval_runner = LMEvalRunner()
    return _lm_eval_runner
