"""
Terminal-Bench 2.0 runner using Harbor worker dispatch.

This mirrors the LM-Eval runner pattern:
- In-memory job tracking in backend API process
- Redis queue dispatch to a dedicated worker
- Pub/sub status and logs streaming
- Historical result loading from shared /data volume
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from enhanced_logger import enhanced_logger as logger

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


class TerminalBenchStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TerminalBenchJob:
    id: str
    name: str
    dataset: str = "terminal-bench@2.0"
    mode: str = "smoke"
    task_limit: Optional[int] = None
    n_concurrent: int = 1
    timeout_multiplier: float = 1.0
    harbor_agent: str = "local-openai-agent"
    model_name: str = "openai/gpt-4o-mini"
    endpoint_url: str = "http://llamacpp-api:8080/v1"
    api_key: str = "placeholder-api-key"
    jobs_dir: Optional[str] = None
    status: TerminalBenchStatus = TerminalBenchStatus.PENDING
    progress: float = 0.0
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    log_output: str = ""
    artifacts: Dict[str, str] = field(default_factory=dict)


class TerminalBenchRunner:
    """
    Dispatches Terminal-Bench jobs to a Redis-backed worker.
    """

    SHARED_RESULTS_DIR = Path("/data/terminal_bench_results")

    def __init__(self):
        self.jobs: Dict[str, TerminalBenchJob] = {}
        self._running_processes: Dict[str, subprocess.Popen] = {}
        self._load_historical_results()

    def _load_historical_results(self):
        if not self.SHARED_RESULTS_DIR.exists():
            logger.info("Terminal-Bench shared results directory not found")
            return

        loaded_count = 0
        for job_dir in self.SHARED_RESULTS_DIR.iterdir():
            if not job_dir.is_dir():
                continue
            if not job_dir.name.startswith("terminal_bench_"):
                continue

            job_id = job_dir.name.replace("terminal_bench_", "", 1)
            summary_file = job_dir / "terminal_bench_summary.json"
            if not summary_file.exists():
                continue

            try:
                with summary_file.open("r", encoding="utf-8") as f:
                    summary = json.load(f)

                created_at = datetime.fromtimestamp(summary_file.stat().st_mtime)
                completed_at = created_at
                if summary.get("completed_at"):
                    try:
                        completed_at = datetime.fromisoformat(summary["completed_at"])
                    except Exception:
                        completed_at = created_at

                status_value = summary.get("status", "completed")
                if status_value not in {
                    TerminalBenchStatus.PENDING.value,
                    TerminalBenchStatus.RUNNING.value,
                    TerminalBenchStatus.COMPLETED.value,
                    TerminalBenchStatus.FAILED.value,
                    TerminalBenchStatus.CANCELLED.value,
                }:
                    status_value = TerminalBenchStatus.COMPLETED.value

                job = TerminalBenchJob(
                    id=job_id,
                    name=summary.get("name", f"terminal-bench-{job_id[:8]}"),
                    dataset=summary.get("dataset", "terminal-bench@2.0"),
                    mode=summary.get("mode", "full"),
                    task_limit=summary.get("task_limit"),
                    n_concurrent=summary.get("n_concurrent", 1),
                    timeout_multiplier=float(summary.get("timeout_multiplier", 1.0)),
                    harbor_agent=summary.get("harbor_agent", "local-openai-agent"),
                    model_name=summary.get("model_name", "openai/gpt-4o-mini"),
                    endpoint_url=summary.get("endpoint_url", "http://llamacpp-api:8080/v1"),
                    api_key=summary.get("api_key", "placeholder-api-key"),
                    jobs_dir=summary.get("jobs_dir"),
                    status=TerminalBenchStatus(status_value),
                    progress=float(summary.get("progress", 100.0)),
                    results=summary.get("results", {}),
                    error=summary.get("error"),
                    created_at=created_at,
                    completed_at=completed_at,
                    artifacts=summary.get("artifacts", {}),
                )
                self.jobs[job_id] = job
                loaded_count += 1
            except Exception as exc:
                logger.warning(f"Failed to load Terminal-Bench result {job_dir.name}: {exc}")

        if loaded_count:
            logger.info(f"Loaded {loaded_count} historical Terminal-Bench jobs")

    def get_available_datasets(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "terminal-bench@2.0",
                "display_name": "Terminal-Bench 2.0",
                "description": "Harbor-based benchmark for agentic terminal tasks",
                "supports_modes": ["smoke", "full"],
                "default_smoke_limit": 5,
            }
        ]

    def create_job(self, job: TerminalBenchJob) -> TerminalBenchJob:
        self.jobs[job.id] = job
        logger.info(
            "Created Terminal-Bench job",
            extra={"job_id": job.id, "dataset": job.dataset, "mode": job.mode},
        )
        return job

    def get_job(self, job_id: str) -> Optional[TerminalBenchJob]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[TerminalBenchJob]:
        return list(self.jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job.status not in (TerminalBenchStatus.PENDING, TerminalBenchStatus.RUNNING):
            return False

        job.status = TerminalBenchStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        if REDIS_AVAILABLE:
            try:
                client = redis.from_url(REDIS_URL, decode_responses=True)
                client.set(f"terminal_bench:job:{job_id}:cancel", "1", ex=3600)
                client.publish(
                    f"terminal_bench:status:{job_id}",
                    json.dumps(
                        {
                            "job_id": job_id,
                            "status": TerminalBenchStatus.CANCELLED.value,
                            "progress": job.progress,
                            "message": "Cancellation requested",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    ),
                )
            except Exception as exc:
                logger.warning(f"Failed to publish Terminal-Bench cancellation for {job_id}: {exc}")
        return True

    def delete_job(self, job_id: str) -> bool:
        deleted_memory = False
        deleted_fs = False
        if job_id in self.jobs:
            del self.jobs[job_id]
            deleted_memory = True

        job_dir = self.SHARED_RESULTS_DIR / f"terminal_bench_{job_id}"
        if job_dir.exists() and job_dir.is_dir():
            try:
                shutil.rmtree(job_dir)
                deleted_fs = True
            except Exception as exc:
                logger.warning(f"Failed to delete Terminal-Bench result directory {job_dir}: {exc}")

        return deleted_memory or deleted_fs

    async def run_benchmark(self, job: TerminalBenchJob) -> AsyncIterator[Dict[str, Any]]:
        job.status = TerminalBenchStatus.RUNNING
        job.progress = 0.0

        task_limit = job.task_limit
        if task_limit is None and job.mode == "smoke":
            task_limit = 5

        payload: Dict[str, Any] = {
            "id": job.id,
            "name": job.name,
            "dataset": job.dataset,
            "mode": job.mode,
            "task_limit": task_limit,
            "n_concurrent": job.n_concurrent,
            "timeout_multiplier": job.timeout_multiplier,
            "harbor_agent": job.harbor_agent,
            "model_name": job.model_name,
            "endpoint_url": job.endpoint_url,
            "api_key": job.api_key,
        }
        if job.jobs_dir:
            payload["jobs_dir"] = job.jobs_dir

        if not REDIS_AVAILABLE:
            job.status = TerminalBenchStatus.FAILED
            job.error = "Redis is not available for Terminal-Bench worker dispatch"
            job.completed_at = datetime.utcnow()
            yield {"type": "error", "job_id": job.id, "error": job.error}
            return

        try:
            async for update in self._run_via_redis(job, payload):
                yield update
        except Exception as exc:
            logger.error("Terminal-Bench run failed", extra={"job_id": job.id, "error": str(exc)})
            job.status = TerminalBenchStatus.FAILED
            job.error = str(exc)
            job.completed_at = datetime.utcnow()
            yield {"type": "error", "job_id": job.id, "error": job.error}

    async def _run_via_redis(
        self, job: TerminalBenchJob, payload: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        client = redis.from_url(REDIS_URL, decode_responses=True)
        pubsub = client.pubsub()

        status_channel = f"terminal_bench:status:{job.id}"
        log_channel = f"terminal_bench:logs:{job.id}"
        pubsub.subscribe(status_channel, log_channel)

        yield {
            "type": "status",
            "job_id": job.id,
            "status": TerminalBenchStatus.RUNNING.value,
            "message": "Dispatching Terminal-Bench job to worker",
        }

        client.lpush("terminal_bench:jobs", json.dumps(payload))

        timeout_seconds = int(os.getenv("TERMINAL_BENCH_MAX_SECONDS", "86400"))
        start_time = asyncio.get_event_loop().time()
        try:
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout_seconds:
                    raise TimeoutError(f"Terminal-Bench timed out after {timeout_seconds} seconds")

                message = pubsub.get_message(timeout=0.5)
                if message and message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                    except json.JSONDecodeError:
                        continue

                    channel = message["channel"]
                    if channel == status_channel:
                        status = data.get("status")
                        progress = float(data.get("progress", 0.0))
                        message_text = data.get("message", "")
                        job.progress = progress

                        if status == TerminalBenchStatus.COMPLETED.value:
                            job.status = TerminalBenchStatus.COMPLETED
                            job.progress = 100.0
                            job.results = data.get("results", {})
                            job.artifacts = data.get("artifacts", {})
                            job.completed_at = datetime.utcnow()
                            yield {
                                "type": "complete",
                                "job_id": job.id,
                                "results": job.results,
                                "artifacts": job.artifacts,
                            }
                            return
                        if status == TerminalBenchStatus.FAILED.value:
                            job.status = TerminalBenchStatus.FAILED
                            job.error = data.get("error", "Terminal-Bench job failed")
                            job.completed_at = datetime.utcnow()
                            yield {"type": "error", "job_id": job.id, "error": job.error}
                            return
                        if status == TerminalBenchStatus.CANCELLED.value:
                            job.status = TerminalBenchStatus.CANCELLED
                            job.completed_at = datetime.utcnow()
                            yield {
                                "type": "cancelled",
                                "job_id": job.id,
                                "message": message_text or "Job cancelled",
                            }
                            return

                        yield {
                            "type": "progress",
                            "job_id": job.id,
                            "progress": progress,
                            "message": message_text[:200],
                        }
                    elif channel == log_channel:
                        log_line = data.get("message", "")
                        job.log_output += log_line + "\n"
                        yield {"type": "log", "job_id": job.id, "message": log_line[:400]}
                await asyncio.sleep(0.1)
        finally:
            pubsub.unsubscribe()
            pubsub.close()


_terminal_bench_runner: Optional[TerminalBenchRunner] = None


def get_terminal_bench_runner() -> TerminalBenchRunner:
    global _terminal_bench_runner
    if _terminal_bench_runner is None:
        _terminal_bench_runner = TerminalBenchRunner()
    return _terminal_bench_runner
