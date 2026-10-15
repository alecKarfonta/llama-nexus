"""QdrantManager - Manages the Qdrant vector database container lifecycle.

[2026-10-15] | INFO | MANAGERS.QDRANT | Qdrant previously required manual start
(`docker compose --profile embed up -d qdrant`). This manager brings it under
backend/API control like the other AI services so the all-in-one install can
guarantee RAG works out of the box.
"""

import os
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from collections import deque

import docker
import httpx

from enhanced_logger import enhanced_logger as logger
from modules.managers.base import DOCKER_AVAILABLE, docker_client


class QdrantManager:
    """Manages Qdrant vector DB lifecycle and configuration."""

    def __init__(self):
        self.docker_container = None
        self.config: Dict[str, Any] = self.load_default_config()
        self.log_buffer = deque(maxlen=1000)
        self.start_time: Optional[datetime] = None
        self.use_docker = DOCKER_AVAILABLE and os.getenv("USE_DOCKER", "false").lower() == "true"
        self.container_name = "llama-nexus-qdrant"
        self.docker_network = os.getenv("DOCKER_NETWORK", "llama-nexus_default")

    def load_default_config(self) -> Dict[str, Any]:
        """[2026-10-15] | INFO | QDRANT_MANAGER.LOAD_DEFAULT_CONFIG | Env-driven defaults."""
        return {
            "server": {
                "host": os.getenv("QDRANT_HOST", "qdrant"),
                "http_port": int(os.getenv("QDRANT_PORT", "6333")),
                "grpc_port": int(os.getenv("QDRANT_GRPC_PORT", "6334")),
                "published_http_port": int(os.getenv("QDRANT_HTTP_PORT", "6333")),
                "published_grpc_port": int(os.getenv("QDRANT_GRPC_PUBLISHED_PORT", "6334")),
                "api_key": os.getenv("QDRANT_API_KEY", ""),
            },
            "image": os.getenv("QDRANT_IMAGE", "qdrant/qdrant:latest"),
        }

    def build_environment(self) -> Dict[str, str]:
        """Container environment for qdrant/qdrant image."""
        env = {
            "QDRANT__SERVICE__HTTP_PORT": str(self.config["server"]["http_port"]),
            "QDRANT__SERVICE__GRPC_PORT": str(self.config["server"]["grpc_port"]),
        }
        api_key = self.config["server"].get("api_key", "")
        if api_key:
            env["QDRANT__SERVICE__API_KEY"] = api_key
        return env

    async def start(self) -> bool:
        """Start Qdrant container (idempotent)."""
        if self.use_docker and DOCKER_AVAILABLE and docker_client:
            return await self.start_docker_sdk()
        return await self.start_docker_cli()

    async def start_docker_sdk(self) -> bool:
        try:
            try:
                existing = docker_client.containers.get(self.container_name)
                if existing.status == "running":
                    logger.info(f"Qdrant container already running: {self.container_name}")
                    self.docker_container = existing
                    self.start_time = self.start_time or datetime.now()
                    return True
                existing.remove(force=True)
            except docker.errors.NotFound:
                pass

            s = self.config["server"]
            container_config = {
                "image": self.config["image"],
                "name": self.container_name,
                "hostname": "qdrant",
                "detach": True,
                "environment": self.build_environment(),
                "network": self.docker_network,
                "ports": {
                    f"{s['http_port']}/tcp": s["published_http_port"],
                    f"{s['grpc_port']}/tcp": s["published_grpc_port"],
                },
                "volumes": {
                    "llama-nexus_qdrant_data": {"bind": "/qdrant/storage", "mode": "rw"}
                },
                "restart_policy": {"Name": "unless-stopped"},
            }
            self.docker_container = docker_client.containers.run(**container_config)
            self.start_time = datetime.now()
            logger.info(f"Qdrant container started: {self.container_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to start Qdrant via Docker SDK: {e}")
            return False

    async def start_docker_cli(self) -> bool:
        try:
            rm = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", self.container_name,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await rm.communicate()
            await asyncio.sleep(0.5)

            s = self.config["server"]
            cmd = [
                "docker", "run", "-d",
                "--name", self.container_name,
                "--hostname", "qdrant",
                "--network", self.docker_network,
                "-p", f"{s['published_http_port']}:{s['http_port']}",
                "-p", f"{s['published_grpc_port']}:{s['grpc_port']}",
                "-v", "llama-nexus_qdrant_data:/qdrant/storage",
                "--restart", "unless-stopped",
            ]
            for key, value in self.build_environment().items():
                cmd.extend(["-e", f"{key}={value}"])
            cmd.append(self.config["image"])

            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error(f"Qdrant docker run failed: {stderr.decode()[:500]}")
                return False
            self.start_time = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Failed to start Qdrant via CLI: {e}")
            return False

    async def stop(self) -> bool:
        try:
            if self.use_docker and DOCKER_AVAILABLE and docker_client:
                try:
                    c = docker_client.containers.get(self.container_name)
                    c.stop(timeout=15)
                    logger.info("Qdrant container stopped")
                except docker.errors.NotFound:
                    pass
                return True
            proc = await asyncio.create_subprocess_exec(
                "docker", "stop", self.container_name,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return True
        except Exception as e:
            logger.error(f"Failed to stop Qdrant: {e}")
            return False

    async def restart(self) -> bool:
        await self.stop()
        await asyncio.sleep(1)
        return await self.start()

    def is_running(self) -> bool:
        """Best-effort running check (container status via Docker)."""
        if not (DOCKER_AVAILABLE and docker_client):
            return False
        try:
            c = docker_client.containers.get(self.container_name)
            return c.status == "running"
        except Exception:
            return False

    async def is_healthy(self) -> bool:
        """Probe the HTTP health endpoint (works from inside the Docker network)."""
        host = self.config["server"]["host"]
        port = self.config["server"]["http_port"]
        url = f"http://{host}:{port}/healthz"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(url)
                return resp.status_code == 200
        except Exception:
            return False

    async def get_status(self) -> Dict[str, Any]:
        running = self.is_running()
        healthy = await self.is_healthy() if running else False
        return {
            "running": running,
            "healthy": healthy,
            "container": self.container_name,
            "image": self.config["image"],
            "host": self.config["server"]["host"],
            "http_port": self.config["server"]["http_port"],
            "grpc_port": self.config["server"]["grpc_port"],
            "uptime_seconds": int((datetime.now() - self.start_time).total_seconds()) if self.start_time and running else None,
        }
