import os
import asyncio
import subprocess
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import psutil
import httpx
from collections import deque
import re

from huggingface_hub import hf_hub_url, HfApi
from enhanced_logger import enhanced_logger as logger
from modules.managers.base import DOCKER_AVAILABLE, docker_client


class StreamingSTTManager:
    """Manages Streaming STT (NVIDIA Nemotron) Service lifecycle and configuration.
    
    This manager controls the streaming-stt Docker container which uses
    NVIDIA's Nemotron-Speech-Streaming-En-0.6b model for real-time
    WebSocket-based speech-to-text transcription.
    """
    
    def __init__(self):
        self.docker_container = None
        self.config: Dict[str, Any] = self.load_default_config()
        self.log_buffer = deque(maxlen=1000)
        self.start_time: Optional[datetime] = None
        self.use_docker = DOCKER_AVAILABLE and os.getenv("USE_DOCKER", "false").lower() == "true"
        self.container_name = "streaming-stt"
        self.docker_network = os.getenv("DOCKER_NETWORK", "llama-nexus_default")
    
    def load_default_config(self) -> Dict[str, Any]:
        """Load default Streaming STT configuration"""
        return {
            "model": {
                "name": os.getenv("STREAMING_STT_MODEL", "nvidia/nemotron-speech-streaming-en-0.6b"),
            },
            "server": {
                "host": "0.0.0.0",
                "port": int(os.getenv("STREAMING_STT_PORT", "8609")),
                "internal_port": 8009,  # Container internal port
            },
            "vad": {
                "threshold": float(os.getenv("STT_VAD_THRESHOLD", "0.005")),
                "silence_threshold": float(os.getenv("STT_VAD_SILENCE_THRESHOLD", "0.005")),
                "hysteresis_ms": int(os.getenv("STT_VAD_HYSTERESIS_MS", "100")),
            },
            "processing": {
                "chunk_size_ms": int(os.getenv("STREAMING_STT_CHUNK_MS", "80")),
                "sample_rate": int(os.getenv("STREAMING_STT_SAMPLE_RATE", "16000")),
                "sentence_end_silence_ms": int(os.getenv("STT_SENTENCE_END_SILENCE_MS", "800")),
                "min_sentence_words": int(os.getenv("STT_MIN_SENTENCE_WORDS", "2")),
                "model_buffer_ms": int(os.getenv("STT_MODEL_BUFFER_MS", "896")),
            },
            "logging": {
                "level": os.getenv("STT_LOG_LEVEL", "INFO"),
            }
        }
    
    def get_internal_ws_url(self) -> str:
        """Get the WebSocket URL for connecting to the streaming-stt container."""
        # When running in Docker, use container name; otherwise use localhost
        if os.getenv("RUNNING_IN_DOCKER", "false").lower() == "true":
            return f"ws://{self.container_name}:8009/ws/stt"
        else:
            return f"ws://localhost:{self.config['server']['port']}/ws/stt"
    
    async def start_docker(self) -> bool:
        """Start Streaming STT service in Docker"""
        try:
            # Check if container already exists
            check_cmd = ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"]
            check_process = await asyncio.create_subprocess_exec(
                *check_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await check_process.communicate()
            
            if self.container_name in stdout.decode():
                # Check if it's running
                running_check = ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"]
                running_process = await asyncio.create_subprocess_exec(
                    *running_check,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                running_stdout, _ = await running_process.communicate()
                
                if self.container_name in running_stdout.decode():
                    logger.info(f"Streaming STT container already running: {self.container_name}")
                    self.start_time = datetime.now()
                    return True
                else:
                    # Remove stopped container
                    rm_cmd = ["docker", "rm", "-f", self.container_name]
                    await asyncio.create_subprocess_exec(*rm_cmd)
            
            # Start the container using docker compose with profile
            docker_cmd = [
                "docker", "compose",
                "--profile", "streaming-stt",
                "up", "-d", "streaming-stt"
            ]
            
            # Run from the llama-nexus directory
            project_dir = os.getenv("PROJECT_DIR", "/home/alec/git/llama-nexus")
            
            start_process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_dir
            )
            stdout, stderr = await start_process.communicate()
            
            if start_process.returncode != 0:
                logger.error(f"Failed to start Streaming STT container: {stderr.decode()}")
                return False
            
            self.start_time = datetime.now()
            logger.info(f"Streaming STT container started: {self.container_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Streaming STT via Docker: {e}")
            return False
    
    async def stop_docker(self) -> bool:
        """Stop Streaming STT service Docker container"""
        try:
            stop_cmd = ["docker", "stop", self.container_name]
            stop_process = await asyncio.create_subprocess_exec(
                *stop_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await stop_process.communicate()
            
            rm_cmd = ["docker", "rm", self.container_name]
            rm_process = await asyncio.create_subprocess_exec(
                *rm_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await rm_process.communicate()
            
            self.start_time = None
            logger.info(f"Streaming STT container stopped: {self.container_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop Streaming STT via Docker: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the Streaming STT service"""
        return await self.start_docker()
    
    async def stop(self) -> bool:
        """Stop the Streaming STT service"""
        return await self.stop_docker()
    
    async def restart(self) -> bool:
        """Restart the Streaming STT service"""
        await self.stop()
        await asyncio.sleep(2)
        return await self.start()
    
    def is_running(self) -> bool:
        """Check if Streaming STT service is running"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return self.container_name in result.stdout
        except Exception:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        is_running = self.is_running()
        
        return {
            "running": is_running,
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time and is_running else 0,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "config": self.config,
            "model": {
                "name": self.config["model"]["name"],
            },
            "endpoint": f"http://localhost:{self.config['server']['port']}",
            "websocket_url": f"ws://localhost:{self.config['server']['port']}/ws/stt",
            "relay_url": "/api/v1/streaming-stt/ws",
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration"""
        for category in ["model", "server", "vad", "processing", "logging"]:
            if category in new_config:
                if category not in self.config:
                    self.config[category] = {}
                filtered = {k: v for k, v in new_config[category].items() if v is not None}
                self.config[category].update(filtered)
        
        # Save config
        config_file = Path("/tmp/streaming_stt_config.json")
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2)
        
        return self.config

