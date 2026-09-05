"""Async manager for MTP head graft jobs (isolated Docker worker)."""
from __future__ import annotations

import asyncio
import os
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from fastapi import HTTPException

from enhanced_logger import enhanced_logger as logger
from modules.gguf_metadata import scan_gguf_mtp
from modules.gguf_model_naming import (
    parse_name_variant,
    plan_graft_deploy_identity,
    plan_graft_output_filename,
    resolve_gguf_path,
)
from modules.mtp_graft import preflight_graft


@dataclass
class MtpGraftJob:
    job_id: str
    status: str  # queued | preflight | grafting | validating | completed | failed | cancelled
    target_name: str
    target_variant: str
    donor_name: Optional[str] = None
    donor_variant: Optional[str] = None
    donor_filename: Optional[str] = None
    donor_repo_id: Optional[str] = None
    output_filename: Optional[str] = None
    progress: float = 0.0
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: Optional[str] = None
    logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["logs"] = self.logs[-100:]
        return data


class MtpGraftManager:
    def __init__(self) -> None:
        self.models_dir = Path(os.getenv("LLAMACPP_MODELS_DIR", "/home/llamacpp/models"))
        self.metadata_dir = self.models_dir / ".metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.convert_script = Path(__file__).resolve().parents[2] / "scripts" / "mtp_graft_convert.py"
        self.models_volume = os.getenv("LLAMACPP_MODELS_VOLUME", "llama-nexus_gpt_oss_models")
        self._jobs: Dict[str, MtpGraftJob] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._cancel_events: Dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def list_jobs(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return [job.to_dict() for job in self._jobs.values()]

    async def get_job(self, job_id: str) -> MtpGraftJob:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Graft job not found: {job_id}")
            return job

    def _resolve_by_registry(self, name: str, variant: str) -> Path:
        path = resolve_gguf_path(self.models_dir, name, variant or "unknown")
        if path:
            return path
        raise HTTPException(status_code=404, detail=f"Model file not found for {name}:{variant}")

    def _resolve_filename(self, filename: str) -> Path:
        path = self.models_dir / filename
        if path.is_file():
            return path
        raise HTTPException(status_code=404, detail=f"GGUF file not found: {filename}")

    def _resolve_target(self, payload: Dict[str, Any]) -> tuple[Path, str, str]:
        if payload.get("filename"):
            path = self._resolve_filename(payload["filename"])
            name, variant = parse_name_variant(path.name)
            return path, name, variant or "unknown"
        name = payload.get("name") or payload.get("target_name")
        variant = payload.get("variant") or payload.get("target_variant") or "unknown"
        if not name:
            raise HTTPException(status_code=400, detail="target name or filename required")
        path = self._resolve_by_registry(name, variant)
        return path, name, variant

    def _resolve_donor(self, payload: Dict[str, Any]) -> tuple[Path, Optional[str], Optional[str], Optional[str]]:
        if payload.get("filename"):
            path = self._resolve_filename(payload["filename"])
            name, variant = parse_name_variant(path.name)
            return path, name, variant, path.name
        name = payload.get("name") or payload.get("donor_name")
        variant = payload.get("variant") or payload.get("donor_variant")
        if name and variant:
            path = self._resolve_by_registry(name, variant)
            return path, name, variant, path.name
        raise HTTPException(status_code=400, detail="donor name/variant or filename required")

    async def preflight(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        target_path, target_name, target_variant = self._resolve_target(payload.get("target") or payload)
        donor_payload = payload.get("donor") or {}
        overwrite = bool(payload.get("overwrite"))

        if donor_payload.get("repo_id") and donor_payload.get("filename"):
            donor_path = self.models_dir / donor_payload["filename"]
            if not donor_path.is_file():
                out_name = plan_graft_output_filename(target_name, target_variant)
                deploy_name, deploy_variant = plan_graft_deploy_identity(target_name, target_variant)
                return {
                    "ok": False,
                    "errors": [
                        f"Donor GGUF not on disk: {donor_payload['filename']} — download from "
                        f"{donor_payload['repo_id']} first"
                    ],
                    "warnings": [],
                    "planned_output_filename": out_name,
                    "planned_deploy_name": deploy_name,
                    "planned_deploy_variant": deploy_variant,
                    "disk_required_bytes": target_path.stat().st_size + 500_000_000,
                }

        donor_path, _, _, _ = self._resolve_donor(donor_payload)
        output_filename = payload.get("output_filename") or plan_graft_output_filename(
            target_name, target_variant
        )
        output_path = self.models_dir / output_filename
        result = preflight_graft(
            target_path,
            donor_path,
            target_name=target_name,
            target_variant=target_variant,
            output_path=output_path,
            overwrite=overwrite,
        )
        return result.to_dict()

    async def start_job(self, payload: Dict[str, Any], download_manager=None) -> MtpGraftJob:
        pre = await self.preflight(payload)
        if not pre.get("ok"):
            raise HTTPException(status_code=400, detail="; ".join(pre.get("errors") or ["Preflight failed"]))

        target_path, target_name, target_variant = self._resolve_target(payload.get("target") or payload)
        donor_path, donor_name, donor_variant, donor_filename = self._resolve_donor(payload.get("donor") or {})
        output_filename = payload.get("output_filename") or pre["planned_output_filename"]
        overwrite = bool(payload.get("overwrite"))

        job_id = str(uuid.uuid4())[:12]
        job = MtpGraftJob(
            job_id=job_id,
            status="queued",
            target_name=target_name,
            target_variant=target_variant,
            donor_name=donor_name,
            donor_variant=donor_variant,
            donor_filename=donor_filename,
            donor_repo_id=(payload.get("donor") or {}).get("repo_id"),
            output_filename=output_filename,
        )

        async with self._lock:
            self._jobs[job_id] = job
            cancel_event = asyncio.Event()
            self._cancel_events[job_id] = cancel_event
            task = asyncio.create_task(
                self._run_job(
                    job_id,
                    target_path,
                    donor_path,
                    self.models_dir / output_filename,
                    overwrite,
                    cancel_event,
                )
            )
            self._tasks[job_id] = task

        return job

    async def cancel_job(self, job_id: str) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Graft job not found: {job_id}")
            if job.status not in ("queued", "preflight", "grafting", "validating"):
                raise HTTPException(status_code=400, detail=f"Job not cancellable in status {job.status}")
            self._cancel_events[job_id].set()
            return True

    async def _append_log(self, job_id: str, line: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.logs.append(line)
                if len(job.logs) > 500:
                    job.logs = job.logs[-500:]

    async def _set_status(self, job_id: str, status: str, *, progress: Optional[float] = None, error: Optional[str] = None) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = status
            if progress is not None:
                job.progress = progress
            if error is not None:
                job.error = error
            if status in ("completed", "failed", "cancelled"):
                job.finished_at = datetime.now().isoformat()

    async def _run_job(
        self,
        job_id: str,
        target_path: Path,
        donor_path: Path,
        output_path: Path,
        overwrite: bool,
        cancel_event: asyncio.Event,
    ) -> None:
        proc: Optional[asyncio.subprocess.Process] = None
        try:
            if cancel_event.is_set():
                raise asyncio.CancelledError()

            await self._set_status(job_id, "grafting", progress=5.0)
            await self._append_log(job_id, f"Target: {target_path.name}")
            await self._append_log(job_id, f"Donor: {donor_path.name}")
            await self._append_log(job_id, f"Output: {output_path.name}")

            if output_path.is_file() and overwrite:
                output_path.unlink()

            if not self.convert_script.is_file():
                raise RuntimeError(f"Graft script missing: {self.convert_script}")

            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "--name",
                f"mtp-graft-{job_id}",
                "-e",
                f"TARGET={target_path.name}",
                "-e",
                f"DONOR={donor_path.name}",
                "-e",
                f"OUTPUT={output_path.name}",
                "-e",
                f"MODELS=/models",
                "-v",
                f"{self.models_volume}:/models",
                "-v",
                f"{self.convert_script}:/opt/mtp_graft_convert.py:ro",
                "python:3.12-slim",
                "bash",
                "-c",
                (
                    "set -euo pipefail && "
                    "export PIP_DISABLE_PIP_VERSION_CHECK=1 && "
                    "apt-get update -qq >/dev/null && "
                    "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq util-linux >/dev/null && "
                    "pip install -q 'gguf>=0.10.0' && "
                    "run_nice() { if command -v ionice >/dev/null 2>&1; then ionice -c3 nice -n19 \"$@\"; else nice -n19 \"$@\"; fi; } && "
                    "run_nice python3 /opt/mtp_graft_convert.py "
                    "\"/models/${TARGET}\" \"/models/${DONOR}\" \"/models/${OUTPUT}\""
                ),
            ]

            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            assert proc.stdout is not None
            while True:
                if cancel_event.is_set():
                    proc.terminate()
                    raise asyncio.CancelledError()
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    await self._append_log(job_id, text)
                    if "Copied" in text and "/" in text:
                        try:
                            part = text.split("Copied", 1)[1]
                            nums = part.strip().split()[0]
                            done, total = nums.split("/")
                            pct = 5.0 + (float(done) / float(total)) * 85.0
                            await self._set_status(job_id, "grafting", progress=pct)
                        except Exception:
                            pass

            rc = await proc.wait()
            if rc != 0:
                raise RuntimeError(f"Graft container exited with code {rc}")

            await self._set_status(job_id, "validating", progress=95.0)
            if not output_path.is_file():
                raise RuntimeError(f"Output file not created: {output_path.name}")

            out_info = scan_gguf_mtp(output_path)
            if not out_info.mtp_capable:
                raise RuntimeError("Output GGUF is not mtp_capable after graft")

            tgt_name, tgt_variant = parse_name_variant(target_path.name)
            deploy_name, deploy_variant = plan_graft_deploy_identity(
                tgt_name,
                tgt_variant or "unknown",
            )

            self._save_graft_metadata(
                deploy_name,
                deploy_variant,
                output_path.name,
                donor_path.name,
            )

            await self._append_log(job_id, f"OK: {output_path.name} mtp_capable nextn={out_info.mtp_nextn_layers}")
            await self._set_status(job_id, "completed", progress=100.0)

        except asyncio.CancelledError:
            await self._set_status(job_id, "cancelled", error="Cancelled by user")
            if proc and proc.returncode is None:
                proc.kill()
        except Exception as exc:
            logger.error(f"MTP graft job {job_id} failed: {exc}", exc_info=True)
            await self._append_log(job_id, f"ERROR: {exc}")
            await self._set_status(job_id, "failed", error=str(exc))
            if output_path.is_file() and overwrite:
                try:
                    output_path.unlink()
                except OSError:
                    pass
        finally:
            async with self._lock:
                self._tasks.pop(job_id, None)
                self._cancel_events.pop(job_id, None)

    def _save_graft_metadata(
        self,
        model_name: str,
        variant: str,
        filename: str,
        donor_filename: str,
    ) -> None:
        info = scan_gguf_mtp(self.models_dir / filename)
        metadata = {
            "model_name": model_name,
            "variant": variant,
            "filename": filename,
            "source": "mtp_graft",
            "mtp_grafted": True,
            "mtp_donor": donor_filename,
            "grafted_at": datetime.now().isoformat(),
            "mtp_capable": info.mtp_capable,
            "mtp_nextn_layers": info.mtp_nextn_layers,
            "gguf_architecture": info.architecture,
            "mtp_scanned_at": datetime.now().isoformat(),
        }
        metadata_file = self.metadata_dir / f"{model_name}-{variant}.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            import json

            json.dump(metadata, f, indent=2)
