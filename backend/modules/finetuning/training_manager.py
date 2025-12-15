"""
Training job manager backed by persistent storage.

Supports two execution modes:
1. Docker container mode (recommended): Spawns training in a dedicated GPU container
2. Subprocess mode (fallback): Runs training_worker.py directly
"""

import os
import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from enhanced_logger import enhanced_logger as logger

from .config import FineTuningJob, TrainingStatus
from .job_store import JobStore, init_job_store
import redis


DEFAULT_WORKER_CMD = ["python3", "/app/training_worker.py"]
TRAINING_IMAGE = os.getenv("TRAINING_IMAGE", "llama-nexus-training:latest")
USE_DOCKER_TRAINING = os.getenv("USE_DOCKER_TRAINING", "true").lower() == "true"

# Try to import docker
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False


class TrainingManager:
    def __init__(
        self,
        store: Optional[JobStore] = None,
        dataset_dir: Optional[Path] = None,
        job_dir: Optional[Path] = None,
    ):
        self.store = store or init_job_store()
        self.dataset_dir = dataset_dir
        self.job_dir = job_dir or Path(os.getenv("FINETUNE_JOB_DIR", str(self.store.base_dir)))

    def create_job(self, job: FineTuningJob) -> FineTuningJob:
        if not job.id:
            job.id = str(uuid.uuid4())
        job.status = TrainingStatus.QUEUED
        if not job.created_at:
            job.created_at = datetime.utcnow()
        logger.info("Created fine-tuning job", extra={"job_id": job.id})
        return self.store.upsert(job)

    def list_jobs(self) -> List[FineTuningJob]:
        return self.store.list()

    def get_job(self, job_id: str) -> Optional[FineTuningJob]:
        return self.store.get(job_id)

    def update_status(
        self, job_id: str, status: TrainingStatus, error: Optional[str] = None
    ) -> FineTuningJob:
        job = self.store.get(job_id)
        if not job:
            raise KeyError(f"Job {job_id} not found")
        job.status = status
        job.error = error
        job.updated_at = datetime.utcnow()
        if status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED):
            job.completed_at = datetime.utcnow()
        self.store.upsert(job)
        logger.info(
            "Updated fine-tuning job status",
            extra={"job_id": job_id, "status": status.value, "error": error},
        )
        return job

    def update_progress(
        self, job_id: str, step: int, total_steps: int, loss: Optional[float] = None
    ) -> FineTuningJob:
        job = self.store.get(job_id)
        if not job:
            raise KeyError(f"Job {job_id} not found")
        job.current_step = step
        job.total_steps = total_steps
        job.progress = round((step / total_steps) * 100, 2) if total_steps else 0.0
        job.current_loss = loss
        job.updated_at = datetime.utcnow()
        self.store.upsert(job)
        return job

    def build_job_config(
        self,
        job: FineTuningJob,
        dataset: Dict[str, Any],
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        output_dir = str(self.job_dir / job.id / "checkpoints")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        adapter_dir = str(self.job_dir / job.id / "adapter")
        Path(adapter_dir).mkdir(parents=True, exist_ok=True)
        
        config = {
            "id": job.id,
            "name": job.name,
            "dataset_id": job.dataset_id,
            "dataset_path": dataset["file_path"],
            "dataset_format": dataset["format"],
            "base_model": job.base_model,
            "lora_config": job.lora_config.dict(),
            "training_config": job.training_config.dict(),
            "qlora_config": job.qlora_config.dict(),
            "output_dir": output_dir,
            "adapter_dir": adapter_dir,
        }
        
        # Add resume configuration if specified
        if resume_from_checkpoint:
            config["resume_from_checkpoint"] = resume_from_checkpoint
        
        return config

    def list_checkpoints(self, job_id: str) -> List[Dict[str, Any]]:
        """List available checkpoints for a job."""
        job = self.store.get(job_id)
        if not job:
            return []
        
        output_dir = self.job_dir / job_id / "checkpoints"
        if not output_dir.exists():
            return []
        
        checkpoints = []
        for item in output_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                state_file = item / "trainer_state.json"
                checkpoint_info = {
                    "path": str(item),
                    "name": item.name,
                    "step": 0,
                    "epoch": 0,
                    "loss": None,
                }
                
                if state_file.exists():
                    try:
                        with state_file.open("r") as f:
                            state = json.load(f)
                        checkpoint_info["step"] = state.get("global_step", 0)
                        checkpoint_info["epoch"] = state.get("epoch", 0)
                        # Get best metric if available
                        if state.get("best_metric"):
                            checkpoint_info["loss"] = state["best_metric"]
                        # Get log history for last loss
                        log_history = state.get("log_history", [])
                        if log_history:
                            last_log = next(
                                (l for l in reversed(log_history) if "loss" in l),
                                {}
                            )
                            if last_log:
                                checkpoint_info["loss"] = last_log.get("loss")
                    except (json.JSONDecodeError, IOError):
                        pass
                
                checkpoints.append(checkpoint_info)
        
        # Sort by step number
        checkpoints.sort(key=lambda x: x["step"], reverse=True)
        return checkpoints

    def get_latest_checkpoint(self, job_id: str) -> Optional[str]:
        """Get the path to the latest checkpoint for a job."""
        checkpoints = self.list_checkpoints(job_id)
        if checkpoints:
            return checkpoints[0]["path"]
        return None

    def start_job(
        self,
        job: FineTuningJob,
        dataset: Dict[str, Any],
        worker_cmd: Optional[List[str]] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        """
        Start a training job.
        
        Args:
            job: The fine-tuning job configuration
            dataset: Dataset information dict
            worker_cmd: Optional custom worker command
            resume_from_checkpoint: Checkpoint path to resume from, or "latest"
        """
        cfg = self.build_job_config(job, dataset, resume_from_checkpoint)
        cfg_path = self.job_dir / f"{job.id}.json"
        with cfg_path.open("w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)

        job.adapter_path = cfg["adapter_dir"]
        self.store.upsert(job)

        # Try Docker container mode first (recommended for GPU training)
        if USE_DOCKER_TRAINING and DOCKER_AVAILABLE:
            try:
                self._start_docker_training(job, cfg_path, resume_from_checkpoint)
                return
            except Exception as exc:
                logger.warning(
                    "Docker training failed, falling back to subprocess",
                    extra={"job_id": job.id, "error": str(exc)}
                )

        # Fallback to subprocess mode
        cmd = worker_cmd or os.getenv("TRAINING_WORKER_CMD")
        if isinstance(cmd, str):
            cmd = cmd.split()
        if not cmd:
            cmd = DEFAULT_WORKER_CMD
        cmd = list(cmd) + [str(cfg_path)]

        logger.info(
            "Starting training worker (subprocess mode)",
            extra={
                "job_id": job.id,
                "cmd": cmd,
                "resume_from": resume_from_checkpoint,
            }
        )
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except Exception as exc:
            logger.error("Failed to start training worker", extra={"job_id": job.id, "error": str(exc)})
            raise

    def _start_docker_training(
        self,
        job: FineTuningJob,
        cfg_path: Path,
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        """Start training in a Docker container with GPU support."""
        client = docker.from_env()
        
        # Prepare volume mounts
        volumes = {
            str(self.job_dir): {"bind": "/app/jobs", "mode": "rw"},
            str(cfg_path.parent): {"bind": "/app/config", "mode": "ro"},
        }
        
        # Add HuggingFace cache mount if available
        hf_cache = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        if Path(hf_cache).exists():
            volumes[hf_cache] = {"bind": "/root/.cache/huggingface", "mode": "rw"}
        
        # Environment variables
        environment = {
            "JOB_CONFIG_PATH": f"/app/config/{cfg_path.name}",
            "REDIS_URL": os.getenv("REDIS_URL", "redis://redis:6379/0"),
            "HF_HOME": "/root/.cache/huggingface",
        }
        
        # GPU configuration
        device_requests = [
            docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
        ]
        
        logger.info(
            "Starting training container",
            extra={
                "job_id": job.id,
                "image": TRAINING_IMAGE,
                "resume_from": resume_from_checkpoint,
            }
        )
        
        container = client.containers.run(
            TRAINING_IMAGE,
            detach=True,
            name=f"training-{job.id[:8]}",
            volumes=volumes,
            environment=environment,
            device_requests=device_requests,
            network="llama-nexus_default",  # Connect to same network as redis
            remove=True,  # Auto-remove when done
        )
        
        logger.info(
            "Training container started",
            extra={"job_id": job.id, "container_id": container.id[:12]}
        )

    def append_log(self, job_id: str, message: str) -> None:
        self.store.append_log(job_id, message)

    def read_logs(self, job_id: str, limit: int = 200) -> List[str]:
        return self.store.read_logs(job_id, limit=limit)

    def handle_status_event(self, job_id: str, payload: Dict[str, Any]) -> Optional[FineTuningJob]:
        job = self.store.get(job_id)
        if not job:
            return None
        step = int(payload.get("step", 0))
        total = int(payload.get("total_steps", job.total_steps or 0))
        loss = payload.get("loss")
        status = payload.get("status")
        adapter_path = payload.get("adapter_path")
        metrics = payload.get("metrics")
        if status:
            try:
                job.status = TrainingStatus(status)
            except ValueError:
                logger.warning("Unknown status in event", extra={"job_id": job_id, "status": status})
        job.current_step = step
        job.total_steps = total
        if total:
            job.progress = round((step / total) * 100, 2)
        if loss is not None:
            job.current_loss = float(loss)
            # Store metrics history for loss chart
            self.store.append_metrics(job_id, step, float(loss))
        if adapter_path:
            job.adapter_path = adapter_path
        if metrics:
            try:
                job.metrics.update(metrics)
            except Exception:
                job.metrics = metrics  # fallback overwrite if not a dict
        job.updated_at = datetime.utcnow()
        if job.status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED):
            job.completed_at = datetime.utcnow()
        return self.store.upsert(job)

    def read_metrics_history(self, job_id: str) -> List[Dict[str, Any]]:
        """Get metrics history for loss chart visualization."""
        return self.store.read_metrics(job_id)

    def handle_log_event(self, job_id: str, message: str) -> None:
        self.append_log(job_id, message)

    def send_command(self, job_id: str, command: str) -> None:
        url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        try:
            client = redis.from_url(url, decode_responses=True)
            client.publish(f"training:commands:{job_id}", command)
        except Exception as exc:
            logger.warning("Failed to publish command", extra={"job_id": job_id, "error": str(exc)})
