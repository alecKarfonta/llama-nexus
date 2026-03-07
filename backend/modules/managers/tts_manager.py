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


class TTSManager:
    """Manages TTS (Text-to-Speech) Service lifecycle and configuration"""
    
    def __init__(self):
        self.docker_container = None
        self.config: Dict[str, Any] = self.load_default_config()
        self.log_buffer = deque(maxlen=1000)
        self.start_time: Optional[datetime] = None
        self.use_docker = DOCKER_AVAILABLE and os.getenv("USE_DOCKER", "false").lower() == "true"
        self.container_name = "openedai-tts"
        self.docker_network = os.getenv("DOCKER_NETWORK", "llama-nexus_default")
    
    def load_default_config(self) -> Dict[str, Any]:
        """Load default TTS configuration"""
        return {
            "model": {
                "name": os.getenv("TTS_MODEL_NAME", "tts-1"),
                "voice": os.getenv("TTS_VOICE", "alloy"),
                "language": os.getenv("TTS_LANGUAGE", "en"),
            },
            "server": {
                "host": "0.0.0.0",
                "port": int(os.getenv("TTS_PORT", "8604")),
                "api_key": os.getenv("TTS_API_KEY", "tts-api-key"),
            },
            "execution": {
                "mode": os.getenv("TTS_EXECUTION_MODE", "cpu"),
                "cuda_devices": os.getenv("TTS_CUDA_DEVICES", "0"),
            },
            "audio": {
                "speed": float(os.getenv("TTS_SPEED", "1.0")),
                "format": os.getenv("TTS_FORMAT", "mp3"),
                "sample_rate": int(os.getenv("TTS_SAMPLE_RATE", "22050")),
            }
        }
    
    async def start_docker(self) -> bool:
        """Start TTS service in Docker"""
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
                # Container exists, remove it
                rm_cmd = ["docker", "rm", "-f", self.container_name]
                await asyncio.create_subprocess_exec(*rm_cmd)
            
            # Build docker command for openedai-speech
            execution_mode = self.config.get("execution", {}).get("mode", "cpu")
            cuda_devices = self.config.get("execution", {}).get("cuda_devices", "0")
            
            docker_cmd = [
                "docker", "run",
                "-d",
                "--name", self.container_name,
                f"--network={self.docker_network}",
                "-p", f"{self.config['server']['port']}:8000",
                "-v", "tts_voices:/app/voices",
                "-v", "tts_config:/app/config",
                "-e", f"TTS_HOME=/app/voices",
                "-e", f"HF_HOME=/app/voices",
            ]
            
            # Add GPU runtime only for GPU mode (for XTTS models)
            if execution_mode == "gpu":
                docker_cmd.extend([
                    "--runtime", "nvidia",
                    "-e", f"NVIDIA_VISIBLE_DEVICES={cuda_devices}",
                ])
            
            # Use openedai-speech image
            docker_cmd.append("ghcr.io/matatonic/openedai-speech:latest")
            
            # Start container
            start_process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await start_process.communicate()
            
            if start_process.returncode != 0:
                logger.error(f"Failed to start TTS container: {stderr.decode()}")
                return False
            
            self.start_time = datetime.now()
            logger.info(f"TTS container started: {self.container_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start TTS via Docker: {e}")
            return False
    
    async def stop_docker(self) -> bool:
        """Stop TTS service Docker container"""
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
            logger.info(f"TTS container stopped: {self.container_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop TTS via Docker: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the TTS service"""
        return await self.start_docker()
    
    async def stop(self) -> bool:
        """Stop the TTS service"""
        return await self.stop_docker()
    
    async def restart(self) -> bool:
        """Restart the TTS service"""
        await self.stop()
        await asyncio.sleep(2)
        return await self.start()
    
    def is_running(self) -> bool:
        """Check if TTS service is running"""
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
                "voice": self.config["model"]["voice"],
            },
            "endpoint": f"http://localhost:{self.config['server']['port']}",
            "available_voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration"""
        for category in ["model", "server", "execution", "audio"]:
            if category in new_config:
                if category not in self.config:
                    self.config[category] = {}
                filtered = {k: v for k, v in new_config[category].items() if v is not None}
                self.config[category].update(filtered)
        
        # Save config
        config_file = Path("/tmp/tts_config.json")
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2)
        
        return self.config


