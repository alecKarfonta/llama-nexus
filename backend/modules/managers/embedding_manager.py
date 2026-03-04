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


class EmbeddingManager:
    """Manages Embedding Service lifecycle and configuration"""
    
    def __init__(self):
        self.docker_container = None
        self.config: Dict[str, Any] = self.load_default_config()
        self.log_buffer = deque(maxlen=1000)
        self.start_time: Optional[datetime] = None
        self.use_docker = DOCKER_AVAILABLE and os.getenv("USE_DOCKER", "false").lower() == "true"
        self.container_name = "llamacpp-embed"
        self.docker_network = os.getenv("DOCKER_NETWORK", "llama-nexus_default")
    
    def load_default_config(self) -> Dict[str, Any]:
        """Load default embedding configuration"""
        return {
            "model": {
                "name": os.getenv("EMBED_MODEL_NAME", "nomic-embed-text-v1.5"),
                "variant": os.getenv("EMBED_MODEL_VARIANT", "Q8_0"),
                "context_size": int(os.getenv("EMBED_CONTEXT_SIZE", "8192")),
                "gpu_layers": int(os.getenv("EMBED_GPU_LAYERS", "999")),
            },
            "performance": {
                "threads": int(os.getenv("EMBED_THREADS", "-1")),
                "batch_size": int(os.getenv("EMBED_BATCH_SIZE", "512")),
                "ubatch_size": int(os.getenv("EMBED_UBATCH_SIZE", "512")),
                "pooling_type": os.getenv("EMBED_POOLING_TYPE", "mean"),
            },
            "execution": {
                "mode": os.getenv("EMBED_EXECUTION_MODE", "gpu"),
                "cuda_devices": os.getenv("EMBED_CUDA_DEVICES", "all"),
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8602,
                "api_key": os.getenv("EMBED_API_KEY", "llamacpp-embed"),
            }
        }
    
    def _get_model_file(self, model_name: str, variant: str) -> str:
        """Get the model filename based on name and variant"""
        # Map model names to filenames with correct separators (dots vs hyphens)
        model_mappings = {
            "nomic-embed-text-v1.5": f"nomic-embed-text-v1.5.{variant}.gguf",
            "e5-mistral-7b": f"e5-mistral-7b-instruct.{variant}.gguf",
            "bge-m3": f"bge-m3.{variant}.gguf",
            "gte-Qwen2-1.5B": f"gte-Qwen2-1.5B-instruct.{variant}.gguf",
        }
        return model_mappings.get(model_name, f"{model_name}.{variant}.gguf")

    def build_command(self) -> List[str]:
        """Build llama-server embedding command"""
        model_name = self.config['model']['name']
        variant = self.config['model']['variant']
        
        # Only map repos here, filenames are handled by _get_model_file
        model_repos = {
            "nomic-embed-text-v1.5": "nomic-ai/nomic-embed-text-v1.5-GGUF",
            "e5-mistral-7b": "intfloat/e5-mistral-7b-instruct-GGUF",
            "bge-m3": "BAAI/bge-m3-GGUF",
            "gte-Qwen2-1.5B": "Alibaba-NLP/gte-Qwen2-1.5B-instruct-GGUF",
        }
        
        model_file = self._get_model_file(model_name, variant)
        
        # This block is replaced by logic above
        
        model_path = f"/home/llamacpp/models/{model_file}"
        
        cmd = [
            "llama-server",
            "--model", model_path,
            "--host", self.config["server"]["host"],
            "--port", str(self.config["server"]["port"]),
            "--api-key", self.config["server"]["api_key"],
            "--ctx-size", str(self.config["model"]["context_size"]),
            "--batch-size", str(self.config["performance"]["batch_size"]),
            "--ubatch-size", str(self.config["performance"]["ubatch_size"]),
            "--pooling", self.config["performance"]["pooling_type"],
            "--embeddings",
            "--metrics",
            "--flash-attn", "auto",
            "--cont-batching",
        ]
        
        # Override GPU layers based on execution mode
        execution_mode = self.config.get("execution", {}).get("mode", "gpu")
        if execution_mode == "cpu":
            cmd.extend(["--n-gpu-layers", "0"])
        else:
            cmd.extend(["--n-gpu-layers", str(self.config["model"]["gpu_layers"])])
        
        # Add threads if specified
        threads = self.config["performance"].get("threads", -1)
        if threads != 0:
            cmd.extend(["--threads", str(threads)])
        
        return cmd
    
    async def start_docker(self) -> bool:
        """Start embedding service in Docker"""
        try:
            if DOCKER_AVAILABLE and docker_client:
                return await self.start_docker_sdk()
            else:
                return await self.start_docker_cli()
        except Exception as e:
            logger.error(f"Failed to start embedding Docker: {e}")
            return False
    
    async def start_docker_sdk(self) -> bool:
        """Start using Docker SDK"""
        try:
            # Check if container already exists
            try:
                existing = docker_client.containers.get(self.container_name)
                if existing.status == "running":
                    logger.info(f"Embedding container already running: {self.container_name}")
                    self.docker_container = existing
                    self.start_time = datetime.now()
                    return True
                else:
                    # Remove stopped container
                    existing.remove(force=True)
            except docker.errors.NotFound:
                pass
            
            # Build environment
            execution_mode = self.config.get("execution", {}).get("mode", "gpu")
            cuda_devices = self.config.get("execution", {}).get("cuda_devices", "all")
            
            environment = {
                "MODEL_NAME": self.config["model"]["name"],
                "MODEL_VARIANT": self.config["model"]["variant"],
                "CONTEXT_SIZE": str(self.config["model"]["context_size"]),
                "GPU_LAYERS": str(self.config["model"]["gpu_layers"]),
                "HOST": self.config["server"]["host"],
                "PORT": str(self.config["server"]["port"]),
                "API_KEY": self.config["server"]["api_key"],
                "THREADS": str(self.config["performance"]["threads"]),
                "BATCH_SIZE": str(self.config["performance"]["batch_size"]),
                "UBATCH_SIZE": str(self.config["performance"]["ubatch_size"]),
                "POOLING_TYPE": self.config["performance"]["pooling_type"],
                "CUDA_VISIBLE_DEVICES": cuda_devices,
                "CUDA_VISIBLE_DEVICES": cuda_devices,
                "NVIDIA_VISIBLE_DEVICES": cuda_devices,
                "MODEL_FILE": self._get_model_file(self.config["model"]["name"], self.config["model"]["variant"]),
            }
            
            # Create container config
            container_config = {
                "image": "llama-nexus-llamacpp-api:latest",
                "name": self.container_name,
                "hostname": "llamacpp-embed",
                "detach": True,
                "environment": environment,
                "command": " ".join(self.build_command()),
                "network": self.docker_network,
                "ports": {f"{self.config['server']['port']}/tcp": self.config['server']['port']},
                "volumes": {
                    "llama-nexus_gpt_oss_models": {"bind": "/home/llamacpp/models", "mode": "rw"}
                },
                "shm_size": "8gb",
            }
            
            # Add GPU runtime only for GPU mode
            if execution_mode == "gpu":
                # Use specific GPU(s) if cuda_devices is set to something other than "all"
                if cuda_devices and cuda_devices != "all":
                    # Use device_ids for specific GPU targeting
                    device_ids = [cuda_devices] if ',' not in cuda_devices else cuda_devices.split(',')
                    container_config["device_requests"] = [
                        docker.types.DeviceRequest(
                            driver='nvidia',
                            device_ids=device_ids,
                            capabilities=[['gpu', 'compute', 'utility']]
                        )
                    ]
                else:
                    # Use all GPUs
                    container_config["device_requests"] = [
                        docker.types.DeviceRequest(
                            driver='nvidia',
                            count=-1,
                            capabilities=[['gpu', 'compute', 'utility']]
                        )
                    ]
            
            # Create and start container
            self.docker_container = docker_client.containers.run(**container_config)
            self.start_time = datetime.now()
            logger.info(f"Embedding container started: {self.container_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start embedding via Docker SDK: {e}")
            return False
    
    async def start_docker_cli(self) -> bool:
        """Start using Docker CLI"""
        try:
            # Always try to remove any existing container first (force remove handles non-existent containers)
            rm_cmd = ["docker", "rm", "-f", self.container_name]
            rm_process = await asyncio.create_subprocess_exec(
                *rm_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await rm_process.communicate()  # Wait for removal to complete
            
            # Small delay to ensure Docker has released the name
            await asyncio.sleep(0.5)
            
            # Build environment and command
            execution_mode = self.config.get("execution", {}).get("mode", "gpu")
            cuda_devices = self.config.get("execution", {}).get("cuda_devices", "all")
            
            # Get model repository info for the environment
            model_name = self.config['model']['name']
            model_mappings = {
                "nomic-embed-text-v1.5": "nomic-ai/nomic-embed-text-v1.5-GGUF",
                "e5-mistral-7b": "intfloat/e5-mistral-7b-instruct-GGUF",
                "bge-m3": "BAAI/bge-m3-GGUF",
                "gte-Qwen2-1.5B": "Alibaba-NLP/gte-Qwen2-1.5B-instruct-GGUF",
            }
            model_repo = model_mappings.get(model_name, "")
            
            docker_cmd = [
                "docker", "run",
                "-d",
                "--name", self.container_name,
                "--hostname", "llamacpp-embed",
                f"--network={self.docker_network}",
                f"-p", f"{self.config['server']['port']}:{self.config['server']['port']}",
                "-v", "llama-nexus_gpt_oss_models:/home/llamacpp/models",
                "--shm-size", "8gb",
                "-e", f"MODEL_NAME={self.config['model']['name']}",
                "-e", f"MODEL_VARIANT={self.config['model']['variant']}",
                "-e", f"MODEL_REPO={model_repo}",
                "-e", f"CONTEXT_SIZE={self.config['model']['context_size']}",
                "-e", f"GPU_LAYERS={self.config['model']['gpu_layers']}",
                "-e", f"HOST={self.config['server']['host']}",
                "-e", f"PORT={self.config['server']['port']}",
                "-e", f"API_KEY={self.config['server']['api_key']}",
                "-e", f"NVIDIA_VISIBLE_DEVICES={cuda_devices}",
                "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                "-e", "HF_HOME=/home/llamacpp/models/.cache",
                "-e", f"MODEL_FILE={self._get_model_file(model_name, self.config['model']['variant'])}",
            ]
            
            # Add GPU runtime only for GPU mode
            if execution_mode == "gpu":
                # Use specific GPU(s) if cuda_devices is set to something other than "all"
                if cuda_devices and cuda_devices != "all":
                    # Format: --gpus "device=2" or --gpus "device=0,1"
                    docker_cmd.insert(2, f'"device={cuda_devices}"')
                else:
                    docker_cmd.insert(2, "all")
                docker_cmd.insert(2, "--gpus")
            
            # Add image and command arguments - start.sh will download model then exec our command
            docker_cmd.append("llama-nexus-llamacpp-api:latest")
            # Pass llama-server command as arguments to start.sh
            docker_cmd.extend(self.build_command())
            
            # Start container
            start_process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await start_process.communicate()
            
            if start_process.returncode != 0:
                logger.error(f"Failed to start embedding container: {stderr.decode()}")
                return False
            
            self.start_time = datetime.now()
            logger.info(f"Embedding container started via CLI: {self.container_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start embedding via Docker CLI: {e}")
            return False
    
    async def stop_docker(self) -> bool:
        """Stop embedding service Docker container"""
        try:
            if DOCKER_AVAILABLE and docker_client:
                return await self.stop_docker_sdk()
            else:
                return await self.stop_docker_cli()
        except Exception as e:
            logger.error(f"Failed to stop embedding Docker: {e}")
            return False
    
    async def stop_docker_sdk(self) -> bool:
        """Stop using Docker SDK"""
        try:
            container = docker_client.containers.get(self.container_name)
            container.stop(timeout=10)
            container.remove()
            self.docker_container = None
            self.start_time = None
            logger.info(f"Embedding container stopped: {self.container_name}")
            return True
        except docker.errors.NotFound:
            logger.warning(f"Embedding container not found: {self.container_name}")
            return False
        except Exception as e:
            logger.error(f"Failed to stop embedding via SDK: {e}")
            return False
    
    async def stop_docker_cli(self) -> bool:
        """Stop using Docker CLI"""
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
            logger.info(f"Embedding container stopped via CLI: {self.container_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop embedding via CLI: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the embedding service"""
        return await self.start_docker()
    
    async def stop(self) -> bool:
        """Stop the embedding service"""
        return await self.stop_docker()
    
    async def restart(self) -> bool:
        """Restart the embedding service"""
        await self.stop()
        await asyncio.sleep(2)
        return await self.start()
    
    def is_running(self) -> bool:
        """Check if embedding service is running"""
        try:
            if DOCKER_AVAILABLE and docker_client:
                try:
                    container = docker_client.containers.get(self.container_name)
                    return container.status == "running"
                except docker.errors.NotFound:
                    return False
            else:
                # CLI check
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
        
        status = {
            "running": is_running,
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time and is_running else 0,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "config": self.config,
            "model": {
                "name": self.config["model"]["name"],
                "variant": self.config["model"]["variant"],
                "context_size": self.config["model"]["context_size"],
                "gpu_layers": self.config["model"]["gpu_layers"],
            },
            "endpoint": f"http://{self.config['server']['host']}:{self.config['server']['port']}"
        }
        
        return status
    
    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration"""
        for category in ["model", "performance", "execution", "server"]:
            if category in new_config:
                if category not in self.config:
                    self.config[category] = {}
                filtered_category = {k: v for k, v in new_config[category].items() if v is not None}
                self.config[category].update(filtered_category)
        
        # Save config
        config_file = Path("/tmp/embedding_config.json")
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2)
        
        return self.config

