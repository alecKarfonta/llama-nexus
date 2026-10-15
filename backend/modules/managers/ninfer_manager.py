"""
NInferManager - Manages a NInfer inference service running as a Docker container.

NInfer (https://github.com/Neroued/ninfer) is a single-GPU RTX 5090 (sm_120a,
Blackwell) C++/CUDA inference engine for registered Qwen checkpoints. It serves
OpenAI-compatible HTTP via `ninfer-serve <artifact.ninfer>`.

Like VLLMManager, this manager works with the `ninfer-api` container defined in
docker-compose.yml (profile `ninfer`) and manages lifecycle via docker compose.
"""

import os
import asyncio
import subprocess
import json
import httpx
from typing import Optional, Dict, Any, List
from datetime import datetime
from collections import deque

from enhanced_logger import enhanced_logger as logger

NINFER_DEPLOY_CONFIG_PATH = os.getenv("NINFER_DEPLOY_CONFIG_PATH", "/data/ninfer_deploy_config.json")

# [2026-08-24 | INFO | NinferManager.MODULE | Registered NInfer artifacts (repo, file, weights profile)
REGISTERED_ARTIFACTS: List[Dict[str, str]] = [
    {
        "artifact": "qwen3_8_27b_nvfp4.ninfer",
        "label": "Qwen3.8-27B NVFP4 (Blackwell W4A4 prefill)",
        "repo": "neroued/Qwen3.8-27B-nvfp4-NInfer",
        "model_id": "qwen3.8-27b",
        "weights": "nvfp4",
    },
    {
        "artifact": "qwen3_8_27b.ninfer",
        "label": "Qwen3.8-27B groupwise-int (Q4/Q5/Q6 mixed)",
        "repo": "neroued/Qwen3.8-27B-NInfer",
        "model_id": "qwen3.8-27b",
        "weights": "groupwise-int",
    },
    {
        "artifact": "qwen3_6_27b_nvfp4.ninfer",
        "label": "Qwen3.6-27B NVFP4 (Blackwell W4A4 prefill)",
        "repo": "neroued/Qwen3.6-27B-nvfp4-NInfer",
        "model_id": "qwen3.6-27b",
        "weights": "nvfp4",
    },
    {
        "artifact": "qwen3_6_27b.ninfer",
        "label": "Qwen3.6-27B groupwise-int",
        "repo": "neroued/Qwen3.6-27B-NInfer",
        "model_id": "qwen3.6-27b",
        "weights": "groupwise-int",
    },
    {
        "artifact": "qwen3_6_35b_a3b.ninfer",
        "label": "Qwen3.6-35B-A3B groupwise-int",
        "repo": "neroued/Qwen3.6-35B-A3B-NInfer",
        "model_id": "qwen3.6-35b-a3b",
        "weights": "groupwise-int",
    },
]

MODELS_DIR = os.getenv("NINFER_MODELS_DIR", "/home/llamacpp/models")
# Path where the models volume is mounted inside the ninfer-api container
CONTAINER_MODELS_DIR = os.getenv("NINFER_CONTAINER_MODELS_DIR", "/models")


def _deep_merge_dict(base: Dict[str, Any], incoming: Dict[str, Any]) -> None:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value


try:
    import docker
    DOCKER_AVAILABLE = True
    try:
        docker_client = docker.from_env()
        docker_client.ping()
    except Exception:
        docker_client = None
except ImportError:
    DOCKER_AVAILABLE = False
    docker_client = None


class NinferManager:
    """Manages the NInfer (Blackwell/Qwen) inference backend lifecycle and configuration."""

    def __init__(self):
        self.config: Dict[str, Any] = self.load_default_config()
        self.log_buffer: deque = deque(maxlen=2000)
        self.start_time: Optional[datetime] = None
        self.container_name = "ninfer-api"
        self.websocket_clients = []
        self._log_reader_task = None
        self.broadcast_ws_event = None  # Wired up externally
        self.last_action_error: Optional[str] = None
        self._load_persisted_config_if_any()

    def _load_persisted_config_if_any(self) -> None:
        path = NINFER_DEPLOY_CONFIG_PATH
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, dict):
                _deep_merge_dict(self.config, data)
                logger.info("Loaded persisted NInfer Deploy config from %s", path)
        except Exception as e:
            logger.warning("Could not load persisted NInfer config from %s: %s", path, e)

    def persist_deploy_config(self) -> None:
        """Write current config to disk (survives backend-api container restart)."""
        path = NINFER_DEPLOY_CONFIG_PATH
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(self.config, fp, indent=2)
        except OSError as e:
            logger.warning("Could not persist NInfer Deploy config to %s: %s", path, e)

    def artifact_path(self) -> str:
        """Host-side path (backend container view) used for existence checks."""
        model = self.config.get("model") or {}
        artifact = str(model.get("artifact") or "qwen3_8_27b_nvfp4.ninfer")
        if artifact.startswith("/"):
            return artifact
        return f"{MODELS_DIR.rstrip('/')}/{artifact}"

    def container_artifact_path(self) -> str:
        """Path passed to ninfer-serve inside the ninfer-api container."""
        model = self.config.get("model") or {}
        artifact = str(model.get("artifact") or "qwen3_8_27b_nvfp4.ninfer")
        if artifact.startswith("/"):
            return artifact
        return f"{CONTAINER_MODELS_DIR.rstrip('/')}/{artifact}"

    def server_listen_port(self) -> int:
        raw = (self.config.get("server") or {}).get("port", 8080)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 8080

    def internal_http_base(self) -> str:
        return f"http://{self.container_name}:{self.server_listen_port()}"

    def load_default_config(self) -> Dict[str, Any]:
        return {
            "backend_type": "ninfer",
            "model": {
                "artifact": os.getenv("NINFER_ARTIFACT", "qwen3_8_27b_nvfp4.ninfer"),
                "served_name": os.getenv("NINFER_SERVED_MODEL_NAME", ""),
            },
            "performance": {
                "max_context": int(os.getenv("NINFER_MAX_CONTEXT", "32768")),
                "kv_capacity": os.getenv("NINFER_KV_CAPACITY", "auto"),
                # [2026-08-30 | INFO | NinferManager.load_default_config | int8 group-64 won
                # the Phase-1 A/B (geomean 182.1 vs 179.9 bf16).
                "kv_dtype": os.getenv("NINFER_KV_DTYPE", "int8"),
                "max_concurrency": int(os.getenv("NINFER_MAX_CONCURRENCY", "2")),
            },
            "speculative": {
                "enabled": os.getenv("NINFER_SPEC_ENABLED", "true").lower() == "true",
                "method": os.getenv("NINFER_SPEC_METHOD", "mtp"),
                "draft_tokens": int(os.getenv("NINFER_DRAFT_TOKENS", "4")),
                "lm_head_draft": os.getenv("NINFER_LM_HEAD_DRAFT", "true").lower() == "true",
            },
            "vision": {
                # [2026-08-24 | INFO | NinferManager.load_default_config | Vision weights only load when the server starts with --vision
                "enabled": os.getenv("NINFER_VISION", "true").lower() == "true",
            },
            "sampling": {
                # [2026-08-30 | INFO | NinferManager.load_default_config | Campaign sampling
                # (temp 0.6 / top-k 20 / presence 1.0) beat default 0.7 on geomean (+1.4%).
                "temperature": float(os.getenv("NINFER_TEMPERATURE", "0.6")),
                "top_p": float(os.getenv("NINFER_TOP_P", "0.95")),
                "top_k": int(os.getenv("NINFER_TOP_K", "20")),
                "presence_penalty": float(os.getenv("NINFER_PRESENCE_PENALTY", "1.0")),
                "greedy": os.getenv("NINFER_GREEDY", "false").lower() == "true",
                "no_thinking": os.getenv("NINFER_NO_THINKING", "true").lower() == "true",
            },
            "execution": {
                "gpu_device": os.getenv("NINFER_GPU_DEVICE", "0"),
            },
            "server": {
                "host": os.getenv("NINFER_HOST", "0.0.0.0"),
                "port": int(os.getenv("NINFER_PORT", "8080")),
                "api_key": os.getenv("API_KEY", "placeholder-api-key"),
                "published_port": int(os.getenv("NINFER_PUBLISHED_PORT", "8603")),
            },
        }

    def registered_artifacts(self) -> List[Dict[str, str]]:
        return list(REGISTERED_ARTIFACTS)

    def artifact_exists(self) -> bool:
        try:
            return os.path.isfile(self.artifact_path())
        except OSError:
            return False

    def build_command(self) -> list:
        """Build the ninfer-serve launch command from the current config.

        Returns the command as a list of tokens (matching manager convention)
        suitable for display in the DeployPage command preview.
        """
        cfg = self.config
        perf = cfg.get("performance") or {}
        spec = cfg.get("speculative") or {}
        server = cfg.get("server") or {}

        kv_cap = str(perf.get("kv_capacity", "auto"))
        try:
            kv_cap_i = int(kv_cap)
            kv_cap = str(kv_cap_i)
        except (TypeError, ValueError):
            kv_cap = "auto"

        cmd = [
            "ninfer-serve", self.container_artifact_path(),
            "--host", str(server.get("host", "0.0.0.0")),
            "--port", str(server.get("port", 8080)),
            "--max-context", str(int(perf.get("max_context", 32768))),
            "--kv-capacity", kv_cap,
            "--max-concurrency", str(int(perf.get("max_concurrency", 2))),
        ]
        kv_dtype = str(perf.get("kv_dtype") or "int8").strip().lower()
        if kv_dtype in ("bf16", "int8", "fp8"):
            cmd.extend(["--kv-dtype", kv_dtype])

        served = str((cfg.get("model") or {}).get("served_name") or "").strip()
        if served:
            cmd.extend(["--model-id", served])

        if (cfg.get("vision") or {}).get("enabled", False):
            cmd.append("--vision")

        if spec.get("enabled", False):
            method = str(spec.get("method") or "mtp").strip() or "mtp"
            cmd.extend(["--spec", method])
            cmd.extend(["--draft-tokens", str(int(spec.get("draft_tokens", 4)))])
            if spec.get("lm_head_draft", True):
                cmd.append("--lm-head-draft")

        samp = cfg.get("sampling") or {}
        cmd.extend([
            "--temperature", str(float(samp.get("temperature", 0.6))),
            "--top-p", str(float(samp.get("top_p", 0.95))),
            "--top-k", str(int(samp.get("top_k", 20))),
            "--presence-penalty", str(float(samp.get("presence_penalty", 1.0))),
        ])
        if samp.get("greedy", False):
            cmd.append("--greedy")
        if samp.get("no_thinking", True):
            cmd.append("--no-thinking")

        api_key = str(server.get("api_key") or "").strip()
        if api_key and api_key != "placeholder-api-key":
            cmd.extend(["--api-key", api_key])

        return cmd

    def get_field_metadata(self) -> Dict[str, Any]:
        return {
            "model": {
                "artifact": {"scope": "ninfer", "type": "select", "description": "Registered NInfer artifact file (weights profile is baked into the artifact).", "ninfer_flag": "positional", "options": [a["artifact"] for a in REGISTERED_ARTIFACTS]},
                "served_name": {"scope": "ninfer", "type": "text", "description": "Optional --model-id alias; empty uses the artifact identity.", "ninfer_flag": "--model-id", "llamacpp_equivalent": "model.name"},
            },
            "performance": {
                "max_context": {"scope": "ninfer", "type": "number", "description": "Logical context ceiling per sequence (native max 262144).", "ninfer_flag": "--max-context", "llamacpp_equivalent": "model.context_size", "min": 1024, "max": 262144, "step": 1024},
                "kv_capacity": {"scope": "ninfer", "type": "text", "description": "Shared Main Text KV pool size in tokens, or 'auto' for largest usable.", "ninfer_flag": "--kv-capacity"},
                "kv_dtype": {"scope": "ninfer", "type": "select", "description": "KV cache storage (bf16 / int8 group-64 / fp8 e4m3 row-256).", "ninfer_flag": "--kv-dtype", "options": ["bf16", "int8", "fp8"]},
                "max_concurrency": {"scope": "ninfer", "type": "number", "description": "Startup-fixed active request lanes (1-8).", "ninfer_flag": "--max-concurrency", "llamacpp_equivalent": "performance.parallel_slots", "min": 1, "max": 8},
            },
            "speculative": {
                "enabled": {"scope": "ninfer", "type": "boolean", "description": "Enable speculative decoding (MTP heads ship inside the artifact).", "ninfer_flag": "--spec mtp"},
                "draft_tokens": {"scope": "ninfer", "type": "number", "description": "MTP draft window (1-5).", "ninfer_flag": "--draft-tokens", "min": 1, "max": 5},
                "lm_head_draft": {"scope": "ninfer", "type": "boolean", "description": "Use optimized LM-head draft proposal.", "ninfer_flag": "--lm-head-draft"},
            },
            "vision": {
                "enabled": {"scope": "ninfer", "type": "boolean", "description": "Load Vision weights so image/video messages are accepted (cannot be enabled later).", "ninfer_flag": "--vision"},
            },
            "sampling": {
                "temperature": {"scope": "ninfer", "type": "number", "description": "Process-level default temperature (request fields override).", "ninfer_flag": "--temperature"},
                "top_p": {"scope": "ninfer", "type": "number", "description": "Process-level default top-p.", "ninfer_flag": "--top-p"},
                "top_k": {"scope": "ninfer", "type": "number", "description": "Process-level default top-k.", "ninfer_flag": "--top-k"},
                "presence_penalty": {"scope": "ninfer", "type": "number", "description": "Process-level default presence penalty.", "ninfer_flag": "--presence-penalty"},
                "greedy": {"scope": "ninfer", "type": "boolean", "description": "Force exact argmax (max MTP agreement; structured/code only).", "ninfer_flag": "--greedy"},
                "no_thinking": {"scope": "ninfer", "type": "boolean", "description": "Process default thinking-off (single-user chat profile).", "ninfer_flag": "--no-thinking"},
            },
            "server": {
                "host": {"scope": "shared", "type": "text", "description": "IP address to listen on.", "ninfer_flag": "--host"},
                "port": {"scope": "shared", "type": "number", "description": "Port to listen on inside the container.", "ninfer_flag": "--port"},
                "published_port": {"scope": "shared", "type": "number", "description": "Host port mapped to the container (compose)."},
            },
        }

    def compose_launch_environment(self) -> Dict[str, str]:
        """Env merged into ``docker compose`` when starting/recreating ``ninfer-api``.

        Keys align with ``docker-compose.yml`` substitutions and ``scripts/start-ninfer.sh``.
        """
        cfg = self.config
        perf = cfg.get("performance") or {}
        spec = cfg.get("speculative") or {}
        server = cfg.get("server") or {}
        vision = cfg.get("vision") or {}
        execution = cfg.get("execution") or {}

        kv_cap = str(perf.get("kv_capacity", "auto"))
        try:
            kv_cap = str(int(kv_cap))
        except (TypeError, ValueError):
            kv_cap = "auto"

        return {
            "NINFER_ARTIFACT": str((cfg.get("model") or {}).get("artifact") or "qwen3_8_27b_nvfp4.ninfer"),
            "NINFER_SERVED_MODEL_NAME": str((cfg.get("model") or {}).get("served_name") or ""),
            "NINFER_MAX_CONTEXT": str(int(perf.get("max_context") or 32768)),
            "NINFER_KV_CAPACITY": kv_cap,
            "NINFER_KV_DTYPE": str(perf.get("kv_dtype") or "int8"),
            "NINFER_MAX_CONCURRENCY": str(int(perf.get("max_concurrency") or 2)),
            "NINFER_SPEC_ENABLED": "true" if spec.get("enabled", False) else "false",
            "NINFER_SPEC_METHOD": str(spec.get("method") or "mtp"),
            "NINFER_DRAFT_TOKENS": str(int(spec.get("draft_tokens") or 4)),
            "NINFER_LM_HEAD_DRAFT": "true" if spec.get("lm_head_draft", True) else "false",
            "NINFER_VISION": "true" if vision.get("enabled", False) else "false",
            "NINFER_TEMPERATURE": str(float((cfg.get("sampling") or {}).get("temperature") or 0.6)),
            "NINFER_TOP_P": str(float((cfg.get("sampling") or {}).get("top_p") or 0.95)),
            "NINFER_TOP_K": str(int((cfg.get("sampling") or {}).get("top_k") or 20)),
            "NINFER_PRESENCE_PENALTY": str(float((cfg.get("sampling") or {}).get("presence_penalty") or 1.0)),
            "NINFER_GREEDY": "true" if (cfg.get("sampling") or {}).get("greedy") else "false",
            "NINFER_NO_THINKING": "true" if (cfg.get("sampling") or {}).get("no_thinking", True) else "false",
            "NINFER_GPU_DEVICE": str(execution.get("gpu_device") or "0"),
            "HOST": str(server.get("host") or "0.0.0.0"),
            "PORT": str(self.server_listen_port()),
            "NINFER_PUBLISHED_PORT": str(int(server.get("published_port") or 8603)),
            # [2026-08-31 | INFO | NinferManager.compose_launch_environment | start-ninfer.sh maps API_KEY to --api-key
            "API_KEY": str(server.get("api_key") or os.getenv("API_KEY") or "placeholder-api-key"),
        }

    def _docker_image_name(self) -> str:
        return os.getenv("NINFER_DOCKER_IMAGE", "llama-nexus-ninfer-api:latest").strip()

    def _docker_image_exists(self) -> bool:
        """True when the local ninfer-api image is already tagged (no Hub pull)."""
        image = self._docker_image_name()
        if not image:
            return False
        try:
            proc = subprocess.run(
                ["docker", "image", "inspect", "--format", "{{.Id}}", image],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return proc.returncode == 0 and bool((proc.stdout or "").strip())
        except Exception:
            return False

    async def _run_docker(self, args: list, **kwargs) -> subprocess.CompletedProcess:
        # [2026-09-28 | INFO | NinferManager._run_docker | Keep uvicorn free during compose
        return await asyncio.to_thread(subprocess.run, args, **kwargs)

    async def start(self) -> bool:
        """Start the NInfer container via ``docker compose --profile ninfer up -d``.

        Falls back to ``docker start`` when no compose file is available.
        """
        compose_err: Optional[str] = None
        try:
            logger.info("Starting NInfer service...")
            project_dir = os.getenv("PROJECT_DIR", "/home/alec/git/llama-nexus")
            compose_path = os.path.join(project_dir, "docker-compose.yml")
            launch_env = {**os.environ, **self.compose_launch_environment()}

            if not self.artifact_exists():
                # [2026-08-24 | WARN | NinferManager.start | Artifact missing blocks start with a clear error instead of a crash-loop
                self.last_action_error = (
                    f"NInfer artifact not found at {self.artifact_path()}. "
                    "Download a registered .ninfer artifact from the Models page first "
                    "(e.g. neroued/Qwen3.8-27B-nvfp4-NInfer)."
                )
                logger.error(self.last_action_error)
                return False

            if not self._docker_image_exists():
                # [2026-09-28 | WARN | NinferManager.start | Compose up without an image
                # pulls Hub (denied) then rebuilds for minutes and times out /v1/service/action.
                image = self._docker_image_name()
                self.last_action_error = (
                    f"NInfer Docker image not found: {image}. "
                    "Build it once from the repository root, then start again: "
                    "docker compose --profile ninfer build ninfer-api"
                )
                logger.error(self.last_action_error)
                return False

            if os.path.isfile(compose_path):
                compose_cmd = [
                    "docker",
                    "compose",
                    "-f",
                    compose_path,
                    "--profile",
                    "ninfer",
                    "up",
                    "-d",
                    "--no-build",
                    "--force-recreate",
                    "ninfer-api",
                ]
                proc = await self._run_docker(
                    compose_cmd,
                    cwd=project_dir,
                    env=launch_env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if proc.returncode == 0:
                    self.start_time = datetime.now()
                    self.last_action_error = None
                    logger.info(
                        "NInfer container started via docker compose "
                        f"(NINFER_ARTIFACT={launch_env.get('NINFER_ARTIFACT')}, PORT={launch_env.get('PORT')})"
                    )
                    self._start_log_reader()
                    return True
                compose_err = (proc.stderr or proc.stdout or "").strip() or f"exit code {proc.returncode}"
                logger.warning("docker compose failed (%s), trying docker start...", compose_err[:500])

            proc = await self._run_docker(
                ["docker", "start", self.container_name],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode == 0:
                self.start_time = datetime.now()
                if compose_err:
                    self.last_action_error = (
                        "docker compose did not apply Deploy env; container started with docker start only — "
                        "settings may be stale until compose succeeds. Compose output: "
                        + compose_err[:4000]
                    )
                    logger.warning(self.last_action_error)
                else:
                    self.last_action_error = None
                logger.info("NInfer container started via docker start (compose unavailable or failed)")
                self._start_log_reader()
                return True

            tail = (proc.stderr or proc.stdout or "").strip()
            msg = tail or "docker start failed"
            if compose_err:
                msg = f"compose: {compose_err[:2000]}; docker start: {msg}"
            self.last_action_error = msg[:8000]
            logger.error(f"Failed to start NInfer: {msg}")
            return False
        except Exception as e:
            self.last_action_error = str(e)[:8000]
            logger.error(f"Failed to start NInfer: {e}")
            return False

    async def stop(self) -> bool:
        """Stop the NInfer container."""
        try:
            logger.info("Stopping NInfer service...")
            proc = subprocess.run(
                ["docker", "stop", self.container_name],
                capture_output=True, text=True, timeout=60,
            )
            self.start_time = None
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip() or "docker stop failed"
                self.last_action_error = err[:8000]
                logger.error("Failed to stop NInfer: %s", err)
                return False
            self.last_action_error = None
            return True
        except Exception as e:
            self.last_action_error = str(e)[:8000]
            logger.error(f"Failed to stop NInfer: {e}")
            return False

    async def restart(self) -> bool:
        """Restart the NInfer container."""
        await self.stop()
        await asyncio.sleep(2)
        return await self.start()

    def is_running(self) -> bool:
        """Check if the NInfer container is running."""
        try:
            if DOCKER_AVAILABLE and docker_client:
                container = docker_client.containers.get(self.container_name)
                return container.status == "running"
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}",
                 "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=5,
            )
            return self.container_name in result.stdout.strip()
        except Exception:
            return False

    async def get_health(self) -> Dict[str, Any]:
        """Check health of the NInfer service.

        Prefers GET /health (unauthenticated). /v1/models is a fallback and
        needs the deploy API key once --api-key is set.
        """
        api_key = str((self.config.get("server") or {}).get("api_key") or "").strip()
        headers = {}
        if api_key and api_key != "placeholder-api-key":
            headers["Authorization"] = f"Bearer {api_key}"
        for path in ("/health", "/v1/models"):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.internal_http_base()}{path}",
                        timeout=5.0,
                        headers=headers,
                    )
                    if response.status_code == 200:
                        return {"healthy": True, "status_code": 200, "probe": path}
            except Exception:
                continue
        return {"healthy": False, "error": "no probe succeeded"}

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the NInfer service."""
        running = self.is_running()
        model = self.config.get("model", {})
        perf = self.config.get("performance", {})
        return {
            "running": running,
            "backend": "ninfer",
            "container": self.container_name,
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time and running else 0,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_action_error": self.last_action_error,
            "artifact_exists": self.artifact_exists(),
            "config": self.config,
            "model": {
                "name": model.get("artifact"),
                "max_context": perf.get("max_context"),
                "vision": (self.config.get("vision") or {}).get("enabled"),
            },
        }

    def get_logs(self, lines: int = 100) -> list:
        """Get recent log lines from the buffer."""
        try:
            lines = int(lines)
        except (ValueError, TypeError):
            lines = 100
        return list(self.log_buffer)[-max(0, lines):]

    async def add_log_line(self, line: str):
        """Add a log line to the buffer and broadcast to websocket clients."""
        log_entry = {
            "message": line,
            "timestamp": datetime.now().isoformat(),
        }
        self.log_buffer.append(log_entry)

        disconnected = []
        for client in self.websocket_clients:
            try:
                await client.send_json({
                    "type": "log",
                    "data": line,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception:
                disconnected.append(client)
        for client in disconnected:
            self.websocket_clients.remove(client)

    def _start_log_reader(self):
        """Start the background docker log reader task."""
        if self._log_reader_task and not self._log_reader_task.done():
            self._log_reader_task.cancel()
        self._log_reader_task = asyncio.create_task(self._read_docker_logs())
        logger.info("Started NInfer docker log reader task")

    async def _read_docker_logs(self):
        """Read logs from the NInfer docker container in the background."""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "logs", "-f", "--tail", "100", self.container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            logger.info(f"Reading docker logs for {self.container_name}")

            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if line_str:
                    await self.add_log_line(line_str)
        except asyncio.CancelledError:
            logger.info("NInfer log reader task cancelled")
        except Exception as e:
            logger.error(f"Error reading NInfer docker logs: {e}")
