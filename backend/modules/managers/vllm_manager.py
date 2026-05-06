"""
VLLMManager - Manages a vLLM inference service running as a Docker container.

Unlike LlamaCPPManager which builds CLI commands and spawns containers,
VLLMManager works with the existing vllm-api container defined in docker-compose.yml.
It manages lifecycle (start/stop) and provides health/status info.
"""

import os
import asyncio
import subprocess
import httpx
from typing import Optional, Dict, Any
from datetime import datetime
from collections import deque

from enhanced_logger import enhanced_logger as logger

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


class VLLMManager:
    """Manages vLLM inference backend lifecycle and configuration."""

    def __init__(self):
        self.config: Dict[str, Any] = self.load_default_config()
        self.log_buffer: deque = deque(maxlen=2000)
        self.start_time: Optional[datetime] = None
        self.container_name = "vllm-api"
        self.websocket_clients = []
        self._log_reader_task = None
        self.broadcast_ws_event = None  # Wired up externally

    def load_default_config(self) -> Dict[str, Any]:
        return {
            "backend_type": "vllm",
            "model": {
                "name": os.getenv("VLLM_MODEL_NAME", "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"),
                "served_name": os.getenv("VLLM_SERVED_MODEL_NAME", "Nemotron-3-Nano-Omni-30B-A3B-Reasoning"),
            },
            "performance": {
                "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "16384")),
                "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.95")),
                "tensor_parallel_size": int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
                "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", "16")),
                "max_num_batched_tokens": int(os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", "8192")),
                "kv_cache_dtype": os.getenv("VLLM_KV_CACHE_DTYPE", "fp8"),
                "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "true").lower() == "true",
            },
            "reasoning": {
                "reasoning_parser": os.getenv("VLLM_REASONING_PARSER", "nemotron_v3"),
            },
            "tools": {
                "enable_auto_tool_choice": True,
                "tool_call_parser": os.getenv("VLLM_TOOL_CALL_PARSER", "qwen3_coder"),
            },
            "media": {
                "video_pruning_rate": float(os.getenv("VLLM_VIDEO_PRUNING_RATE", "0.5")),
                "video_fps": int(os.getenv("VLLM_VIDEO_FPS", "2")),
                "video_num_frames": int(os.getenv("VLLM_VIDEO_NUM_FRAMES", "256")),
            },
            "server": {
                "host": os.getenv("VLLM_HOST", "0.0.0.0"),
                "port": int(os.getenv("VLLM_PORT", "8080")),
                "api_key": os.getenv("API_KEY", "placeholder-api-key"),
            },
        }

    async def start(self) -> bool:
        """Start the vLLM container.

        Tries docker start first (for existing stopped containers),
        then falls back to docker compose if the container doesn't exist.
        """
        try:
            logger.info("Starting vLLM service...")

            # First, try docker start (works for existing containers managed by compose)
            proc = subprocess.run(
                ["docker", "start", self.container_name],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode == 0:
                self.start_time = datetime.now()
                logger.info("vLLM container started successfully")
                self._start_log_reader()
                return True

            # If container doesn't exist, try docker compose
            logger.info(f"docker start failed ({proc.stderr.strip()}), trying docker compose...")
            compose_file = os.getenv("COMPOSE_FILE", "")
            compose_cmd = ["docker", "compose"]
            if compose_file:
                compose_cmd.extend(["-f", compose_file])
            compose_cmd.extend(["--profile", "vllm", "up", "-d", "vllm-api"])

            proc = subprocess.run(
                compose_cmd,
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode == 0:
                self.start_time = datetime.now()
                logger.info("vLLM container started via docker compose")
                self._start_log_reader()
                return True
            else:
                logger.error(f"Failed to start vLLM: {proc.stderr}")
                return False
        except Exception as e:
            logger.error(f"Failed to start vLLM: {e}")
            return False

    async def stop(self) -> bool:
        """Stop the vLLM container."""
        try:
            logger.info("Stopping vLLM service...")
            proc = subprocess.run(
                ["docker", "stop", self.container_name],
                capture_output=True, text=True, timeout=60,
            )
            self.start_time = None
            return proc.returncode == 0
        except Exception as e:
            logger.error(f"Failed to stop vLLM: {e}")
            return False

    async def restart(self) -> bool:
        """Restart the vLLM container."""
        await self.stop()
        await asyncio.sleep(2)
        return await self.start()

    def is_running(self) -> bool:
        """Check if the vLLM container is running."""
        try:
            if DOCKER_AVAILABLE and docker_client:
                container = docker_client.containers.get(self.container_name)
                return container.status == "running"
            else:
                result = subprocess.run(
                    ["docker", "ps", "--filter", f"name={self.container_name}",
                     "--format", "{{.Names}}"],
                    capture_output=True, text=True, timeout=5,
                )
                return self.container_name in result.stdout.strip()
        except Exception:
            return False

    async def get_health(self) -> Dict[str, Any]:
        """Check health of the vLLM service."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://{self.container_name}:8080/health", timeout=5.0,
                )
                return {
                    "healthy": response.status_code == 200,
                    "status_code": response.status_code,
                    "response": response.text,
                }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the vLLM service."""
        running = self.is_running()
        return {
            "running": running,
            "backend": "vllm",
            "container": self.container_name,
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time and running else 0,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "config": self.config,
            "model": {
                "name": self.config.get("model", {}).get("name"),
                "served_name": self.config.get("model", {}).get("served_name"),
                "max_model_len": self.config.get("performance", {}).get("max_model_len"),
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
        logger.info("Started vLLM docker log reader task")

    async def _read_docker_logs(self):
        """Read logs from the vLLM docker container in the background."""
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
            logger.info("vLLM log reader task cancelled")
        except Exception as e:
            logger.error(f"Error reading vLLM docker logs: {e}")
