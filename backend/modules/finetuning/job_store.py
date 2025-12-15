"""
Persistent storage for fine-tuning jobs and lightweight logs.

Jobs are stored in a JSON index. Logs are stored per-job in plain text files to
avoid bloating the index. This is intentionally simple and can be swapped for a
DB later.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from enhanced_logger import enhanced_logger as logger

from .config import FineTuningJob, TrainingStatus


class JobStore:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "jobs_index.json"
        if not self.index_path.exists():
            self._write_index([])

    def _read_index(self) -> List[Dict]:
        try:
            with self.index_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            logger.warning("Job index corrupted, resetting to empty list")
            return []

    def _write_index(self, items: List[Dict]) -> None:
        tmp_path = self.index_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(items, fh, default=str, indent=2)
        tmp_path.replace(self.index_path)

    def list(self) -> List[FineTuningJob]:
        return [FineTuningJob(**item) for item in self._read_index()]

    def get(self, job_id: str) -> Optional[FineTuningJob]:
        for item in self._read_index():
            if item.get("id") == job_id:
                return FineTuningJob(**item)
        return None

    def upsert(self, job: FineTuningJob) -> FineTuningJob:
        items = self._read_index()
        replaced = False
        for idx, item in enumerate(items):
            if item.get("id") == job.id:
                items[idx] = job.dict()
                replaced = True
                break
        if not replaced:
            items.append(job.dict())
        self._write_index(items)
        return job

    def delete(self, job_id: str) -> bool:
        items = self._read_index()
        remaining = [item for item in items if item.get("id") != job_id]
        removed = len(items) != len(remaining)
        if removed:
            self._write_index(remaining)
            log_path = self.log_path(job_id)
            if log_path.exists():
                try:
                    log_path.unlink()
                except Exception as exc:
                    logger.warning(
                        "Failed to delete job log file",
                        extra={"job_id": job_id, "error": str(exc)},
                    )
        return removed

    def log_path(self, job_id: str) -> Path:
        return self.base_dir / f"{job_id}.log"

    def append_log(self, job_id: str, message: str) -> None:
        log_file = self.log_path(job_id)
        timestamp = datetime.utcnow().isoformat()
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(f"[{timestamp}] {message}\n")

    def read_logs(self, job_id: str, limit: int = 200) -> List[str]:
        log_file = self.log_path(job_id)
        if not log_file.exists():
            return []
        with log_file.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
        return lines[-limit:]

    def metrics_path(self, job_id: str) -> Path:
        return self.base_dir / f"{job_id}_metrics.jsonl"

    def append_metrics(self, job_id: str, step: int, loss: float, **kwargs) -> None:
        """Append a metrics data point for loss chart visualization."""
        metrics_file = self.metrics_path(job_id)
        entry = {
            "step": step,
            "loss": loss,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs,
        }
        with metrics_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def read_metrics(self, job_id: str) -> List[Dict]:
        """Read all metrics history for a job."""
        metrics_file = self.metrics_path(job_id)
        if not metrics_file.exists():
            return []
        metrics = []
        with metrics_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        metrics.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return metrics


def default_job_dir() -> Path:
    env_path = os.getenv("FINETUNE_JOB_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "finetune_jobs"
    )


def init_job_store() -> JobStore:
    base_dir = default_job_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Initialized job store", extra={"path": str(base_dir)})
    return JobStore(base_dir=base_dir)
