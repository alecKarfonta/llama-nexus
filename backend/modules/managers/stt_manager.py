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


class STTManager:
    """Manages STT (Speech-to-Text) Service lifecycle and configuration"""
    
    def __init__(self):
        self.docker_container = None
        self.config: Dict[str, Any] = self.load_default_config()
        self.log_buffer = deque(maxlen=1000)
        self.start_time: Optional[datetime] = None
        self.use_docker = DOCKER_AVAILABLE and os.getenv("USE_DOCKER", "false").lower() == "true"
        self.container_name = "whisper-stt"
        self.docker_network = os.getenv("DOCKER_NETWORK", "llama-nexus_default")
    
    def load_default_config(self) -> Dict[str, Any]:
        """Load default STT configuration"""
        return {
            "model": {
                "name": os.getenv("STT_MODEL_NAME", "base"),
                "size": os.getenv("STT_MODEL_SIZE", "base"),
                "language": os.getenv("STT_LANGUAGE", "auto"),
                "task": os.getenv("STT_TASK", "transcribe"),
            },
            "server": {
                "host": "0.0.0.0",
                "port": int(os.getenv("STT_PORT", "8603")),
                "api_key": os.getenv("STT_API_KEY", "stt-api-key"),
            },
            "execution": {
                "mode": os.getenv("STT_EXECUTION_MODE", "gpu"),
                "cuda_devices": os.getenv("STT_CUDA_DEVICES", "0"),
                "compute_type": os.getenv("STT_COMPUTE_TYPE", "float16"),
            }
        }
    
    async def start_docker(self) -> bool:
        """Start STT service in Docker"""
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
            
            # Build docker command for faster-whisper
            execution_mode = self.config.get("execution", {}).get("mode", "gpu")
            cuda_devices = self.config.get("execution", {}).get("cuda_devices", "0")
            
            # Map model names to HuggingFace repositories for models not in faster-whisper's default list
            model_size = self.config['model']['size']
            model_mappings = {
                "distil-large-v3.5-ct2": "distil-whisper/distil-large-v3.5-ct2",
                # Add more mappings here as needed for other custom models
            }
            
            # Use HuggingFace repo path if model needs mapping, otherwise use the model name directly
            whisper_model = model_mappings.get(model_size, model_size)
            
            docker_cmd = [
                "docker", "run",
                "-d",
                "--name", self.container_name,
                f"--network={self.docker_network}",
                "-p", f"{self.config['server']['port']}:8000",
                "-e", f"WHISPER_MODEL={whisper_model}",
                "-e", f"WHISPER_LANG={self.config['model']['language']}",
                "-e", f"COMPUTE_TYPE={self.config['execution']['compute_type']}",
            ]
            
            # Add GPU runtime only for GPU mode
            if execution_mode == "gpu":
                docker_cmd.extend([
                    "--runtime", "nvidia",
                    "-e", f"NVIDIA_VISIBLE_DEVICES={cuda_devices}",
                ])
            
            # Use faster-whisper-server image
            docker_cmd.append("fedirz/faster-whisper-server:latest-cuda")
            
            # Start container
            start_process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await start_process.communicate()
            
            if start_process.returncode != 0:
                logger.error(f"Failed to start STT container: {stderr.decode()}")
                return False
            
            self.start_time = datetime.now()
            logger.info(f"STT container started: {self.container_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start STT via Docker: {e}")
            return False
    
    async def stop_docker(self) -> bool:
        """Stop STT service Docker container"""
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
            logger.info(f"STT container stopped: {self.container_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop STT via Docker: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the STT service"""
        return await self.start_docker()
    
    async def stop(self) -> bool:
        """Stop the STT service"""
        return await self.stop_docker()
    
    async def restart(self) -> bool:
        """Restart the STT service"""
        await self.stop()
        await asyncio.sleep(2)
        return await self.start()
    
    def is_running(self) -> bool:
        """Check if STT service is running"""
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
                "size": self.config["model"]["size"],
                "language": self.config["model"]["language"],
            },
            "endpoint": f"http://localhost:{self.config['server']['port']}"
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration"""
        for category in ["model", "server", "execution"]:
            if category in new_config:
                if category not in self.config:
                    self.config[category] = {}
                filtered = {k: v for k, v in new_config[category].items() if v is not None}
                self.config[category].update(filtered)
        
        # Save config
        config_file = Path("/tmp/stt_config.json")
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2)
        
        return self.config

