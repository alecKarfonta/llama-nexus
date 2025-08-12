"""
Enhanced FastAPI Backend for LlamaCPP Model Management
Provides APIs for managing llamacpp instances with Docker support
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
import subprocess
import json
import os
import signal
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
import psutil
import logging
from pathlib import Path
import yaml
import httpx
from contextlib import asynccontextmanager
from collections import deque
import threading
from dataclasses import dataclass, asdict
from huggingface_hub import hf_hub_url
from huggingface_hub import HfApi

# Import enhanced logger
from enhanced_logger import enhanced_logger as logger

# Try to import docker, fallback to subprocess if not available
try:
    import docker
    DOCKER_AVAILABLE = True
    # Try different approaches to initialize Docker client
    docker_client = None
    
    # Method 1: Try direct unix socket with proper path
    try:
        docker_client = docker.DockerClient(base_url='unix:///var/run/docker.sock')
        # Test the connection
        docker_client.ping()
        logger.info("Docker client initialized successfully with connection_method='unix_socket'")
    except Exception as e1:
        logger.warning(f"Docker unix socket connection failed with error='{e1}'")
        
        # Method 2: Try from_env as fallback
        try:
            import os
            os.environ['DOCKER_HOST'] = 'unix:///var/run/docker.sock'
            docker_client = docker.from_env()
            docker_client.ping()
            logger.info("Docker client initialized successfully with connection_method='from_env'")
        except Exception as e2:
            logger.warning(f"Docker from_env connection failed with error='{e2}'")
            docker_client = None
            
    if docker_client is None:
        DOCKER_AVAILABLE = False
        logger.warning("All Docker connection methods failed, fallback_mode='subprocess'")
        
except ImportError:
    DOCKER_AVAILABLE = False
    docker_client = None
    logger.warning("Docker SDK not available, fallback_mode='subprocess'")
except Exception as e:
    DOCKER_AVAILABLE = False
    docker_client = None
    logger.warning(f"Docker client initialization failed with error='{e}', fallback_mode='subprocess'")

# Get log level for startup reporting
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Log startup configuration with enhanced formatting
logger.log_startup("LlamaCPP Management API", 
                  log_level=log_level,
                  docker_available=DOCKER_AVAILABLE, 
                  use_docker_mode=os.getenv('USE_DOCKER', 'false'))

class LlamaCPPManager:
    """Manages LlamaCPP instance lifecycle and configuration"""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.docker_container = None
        self.config: Dict[str, Any] = self.load_default_config()
        self.log_buffer = deque(maxlen=2000)
        self.start_time: Optional[datetime] = None
        self.websocket_clients: List[WebSocket] = []
        self.log_reader_task = None
        # Check if Docker is available via CLI as fallback
        docker_cli_available = False
        if not DOCKER_AVAILABLE and os.getenv("USE_DOCKER", "false").lower() == "true":
            try:
                result = subprocess.run(["docker", "--version"], capture_output=True, timeout=5)
                docker_cli_available = result.returncode == 0
                if docker_cli_available:
                    logger.info("Docker CLI available, operation_mode='cli_commands'")
            except Exception as e:
                logger.warning(f"Docker CLI check failed with error='{e}'")
        
        self.use_docker = (DOCKER_AVAILABLE or docker_cli_available) and os.getenv("USE_DOCKER", "false").lower() == "true"
        self.container_name = os.getenv("LLAMACPP_CONTAINER_NAME", "llamacpp-gpt-oss-managed")
        # Docker network to attach launched containers to (defaults to compose default)
        self.docker_network = os.getenv("DOCKER_NETWORK", "llamacpp-gpt-oss-api_default")
        
        # Defer container log reading to runtime callers to avoid loop issues during import
        
    def load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from environment or defaults"""
        return {
            "model": {
                "name": os.getenv("MODEL_NAME", "Qwen3-Coder-30B-A3B-Instruct"),
                "variant": os.getenv("MODEL_VARIANT", "Q4_K_M"),
                "context_size": int(os.getenv("CONTEXT_SIZE", "128000")),
                "gpu_layers": int(os.getenv("GPU_LAYERS", "999")),
                "n_cpu_moe": int(os.getenv("N_CPU_MOE", "0")),
            },
            "sampling": {
                "temperature": float(os.getenv("TEMPERATURE", "0.7")),
                "top_p": float(os.getenv("TOP_P", "0.8")),
                "top_k": int(os.getenv("TOP_K", "20")),
                "min_p": float(os.getenv("MIN_P", "0.03")),
                "repeat_penalty": float(os.getenv("REPEAT_PENALTY", "1.05")),
                "repeat_last_n": int(os.getenv("REPEAT_LAST_N", "256")),
                "frequency_penalty": float(os.getenv("FREQUENCY_PENALTY", "0.3")),
                "presence_penalty": float(os.getenv("PRESENCE_PENALTY", "0.2")),
                "dry_multiplier": float(os.getenv("DRY_MULTIPLIER", "0.6")),
                "dry_base": float(os.getenv("DRY_BASE", "2.0")),
                "dry_allowed_length": int(os.getenv("DRY_ALLOWED_LENGTH", "1")),
                "dry_penalty_last_n": int(os.getenv("DRY_PENALTY_LAST_N", "1024")),
            },
            "performance": {
                "threads": int(os.getenv("THREADS", "-1")),
                "batch_size": int(os.getenv("BATCH_SIZE", "2048")),
                "ubatch_size": int(os.getenv("UBATCH_SIZE", "512")),
                "num_keep": int(os.getenv("NUM_KEEP", "1024")),
                "num_predict": int(os.getenv("NUM_PREDICT", "64768")),
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8080,
                "api_key": os.getenv("API_KEY", "placeholder-api-key"),
            }
        }
    
    def build_command(self) -> List[str]:
        """Build llamacpp server command from configuration"""
        # Try different filename patterns to find the actual model file
        model_name = self.config['model']['name']
        variant = self.config['model']['variant']
        
        # Try different filename patterns including merged multi-part files
        possible_paths = [
            f"/home/llamacpp/models/{model_name}-{variant}.gguf",  # ModelName-Q4_K_M.gguf (merged)
            f"/home/llamacpp/models/{model_name}.{variant}.gguf",  # ModelName.Q4_K_M.gguf
            f"/home/llamacpp/models/{model_name}{variant}.gguf",   # ModelNameQ4_K_M.gguf
            f"/home/llamacpp/models/{model_name}_{variant}.gguf",  # ModelName_Q4_K_M.gguf
            # For multi-part files that haven't been merged yet, use the first part
            f"/home/llamacpp/models/{model_name}-{variant}-00001-of-00002.gguf",  # First part of split files
            f"/home/llamacpp/models/{model_name}-{variant}-00001-of-00003.gguf",  # First part of 3-way split
            # Check for files in subdirectories (HuggingFace download style)
            f"/home/llamacpp/models/{variant}/{model_name}-{variant}.gguf",  # In variant subdirectory
            f"/home/llamacpp/models/{variant}/{model_name}-{variant}-00001-of-00002.gguf",  # Multi-part in subdirectory
            f"/home/llamacpp/models/{variant}/{model_name}-{variant}-00001-of-00003.gguf",  # 3-part in subdirectory
        ]
        
        # Find the first path that exists
        model_path = None
        for path in possible_paths:
            if Path(path).exists():
                model_path = path
                logger.info(f"Found model file at: {path}")
                break
        
        # If no path exists, use the default pattern
        if model_path is None:
            model_path = possible_paths[0]
            logger.warning(f"Model file not found, using default path: {model_path}")
        
        cmd = [
            "llama-server",
            "--model", model_path,
            "--host", self.config["server"]["host"],
            "--port", str(self.config["server"]["port"]),
            "--api-key", self.config["server"]["api_key"],
            "--ctx-size", str(self.config["model"]["context_size"]),
            "--n-gpu-layers", str(self.config["model"]["gpu_layers"]),
            "--n-predict", str(self.config["performance"]["num_predict"]),
            "--threads", str(self.config["performance"]["threads"]),
            "--batch-size", str(self.config["performance"]["batch_size"]),
            "--ubatch-size", str(self.config["performance"]["ubatch_size"]),
            "--n-cpu-moe", str(self.config["model"]["n_cpu_moe"]),
            "--temp", str(self.config["sampling"]["temperature"]),
            "--top-p", str(self.config["sampling"]["top_p"]),
            "--top-k", str(self.config["sampling"]["top_k"]),
            "--min-p", str(self.config["sampling"]["min_p"]),
            "--repeat-penalty", str(self.config["sampling"]["repeat_penalty"]),
            "--repeat-last-n", str(self.config["sampling"]["repeat_last_n"]),
            "--frequency-penalty", str(self.config["sampling"]["frequency_penalty"]),
            "--presence-penalty", str(self.config["sampling"]["presence_penalty"]),
            "--dry-multiplier", str(self.config["sampling"]["dry_multiplier"]),
            "--dry-base", str(self.config["sampling"]["dry_base"]),
            "--dry-allowed-length", str(self.config["sampling"]["dry_allowed_length"]),
            "--dry-penalty-last-n", str(self.config["sampling"]["dry_penalty_last_n"]),
            "--chat-template-file", "/home/llamacpp/chat-template.jinja",
            "--jinja",
            "--verbose",
            "--metrics",
            "--embeddings",
            "--flash-attn",
            "--cont-batching"
        ]
        
        return cmd
    
    async def start_docker(self) -> bool:
        """Start llamacpp in Docker container"""
        try:
            logger.log_operation_start("Docker container startup", container_name=self.container_name, docker_network=self.docker_network)
            
            if DOCKER_AVAILABLE and docker_client:
                return await self.start_docker_sdk()
            else:
                return await self.start_docker_cli()
                
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            error_msg = f"Unexpected error starting Docker container: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise HTTPException(status_code=500, detail=error_msg)
    
    async def start_docker_sdk(self) -> bool:
        """Start Docker container using Python SDK"""
        logger.info("Using Docker Python SDK for container_management='docker_sdk'")
        
        # Check if container exists
        try:
            container = docker_client.containers.get(self.container_name)
            if container.status == "running":
                raise HTTPException(status_code=400, detail="LlamaCPP container is already running")
                
            # Remove old container if it exists but not running
            logger.info(f"Removing existing container_name='{self.container_name}', status='not_running'")
            container.remove(force=True)
        except docker.errors.NotFound:
            logger.info(f"No existing container found with container_name='{self.container_name}'")
        
        # Build and validate command
        cmd = self.build_command()
        cmd_str = ' '.join(cmd)
        logger.info(f"Built llamacpp command with {len(cmd)} arguments: '{cmd_str[:100]}...'" if len(cmd_str) > 100 else f"Built llamacpp command: '{cmd_str}'")
        
        # Validate Docker image exists
        image_name = "llamacpp-gpt-oss-api-llamacpp-api"
        try:
            docker_client.images.get(image_name)
            logger.info(f"Docker image validation successful for image_name='{image_name}'")
        except docker.errors.ImageNotFound:
            error_msg = f"Docker image not found: {image_name}. Please build the image first."
            logger.error(f"Docker image validation failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        logger.info("Initiating Docker container creation and startup...")
        
        # Run container with the command
        self.docker_container = docker_client.containers.run(
            image=image_name,
            name=self.container_name,
            command=" ".join(cmd),
            detach=True,
            auto_remove=False,
            network=self.docker_network,
            volumes={
                "llamacpp-gpt-oss-api_gpt_oss_models": {"bind": "/home/llamacpp/models", "mode": "rw"},
                "/home/alec/git/llamacpp-gpt-oss-api/chat-template.jinja": {"bind": "/home/llamacpp/chat-template.jinja", "mode": "ro"}
            },
            environment={
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_VISIBLE_DEVICES": "0",
                "MODEL_NAME": self.config['model']['name'],
                "MODEL_VARIANT": self.config['model']['variant'],
            },
            runtime="nvidia",
            shm_size="16g",
            ports={"8080/tcp": 8600}
        )
        
        logger.log_operation_success("Docker container creation", container_id=self.docker_container.id[:12], container_name=self.container_name)
        self.start_time = datetime.now()
        
        # Start log reader
        asyncio.create_task(self.read_docker_logs())
        
        return True
    
    async def start_docker_cli(self) -> bool:
        """Start Docker container using CLI commands"""
        logger.info("Using Docker CLI for container_management='docker_cli'")
        
        # Check if container is already running
        check_process = await asyncio.create_subprocess_exec(
            'docker', 'ps', '--filter', f'name={self.container_name}', '--format', '{{.Names}}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await check_process.communicate()
        
        if check_process.returncode == 0:
            container_names = stdout.decode().strip().split('\n')
            if self.container_name in container_names:
                raise HTTPException(status_code=400, detail="LlamaCPP container is already running")
        
        # Remove existing container if it exists but not running
        logger.info(f"Removing any existing container: {self.container_name}")
        remove_process = await asyncio.create_subprocess_exec(
            'docker', 'rm', '-f', self.container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await remove_process.communicate()  # Don't care about the result
        
        # Build command
        cmd = self.build_command()
        logger.info(f"Built command: {' '.join(cmd)}")
        
        # Check if image exists
        image_name = "llamacpp-gpt-oss-api-llamacpp-api"
        image_check = await asyncio.create_subprocess_exec(
            'docker', 'images', '--format', '{{.Repository}}:{{.Tag}}', image_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await image_check.communicate()
        
        if image_check.returncode != 0 or not stdout.decode().strip():
            error_msg = f"Docker image not found: {image_name}. Please build the image first."
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        logger.info(f"Docker image found: {image_name}")
        
        # Build docker run command
        docker_cmd = [
            'docker', 'run', '-d',
            '--name', self.container_name,
            '--runtime', 'nvidia',
            '--shm-size', '16g',
            '-p', '8600:8080',
            '--network', self.docker_network,
            '-v', 'llamacpp-gpt-oss-api_gpt_oss_models:/home/llamacpp/models',
            '-v', '/home/alec/git/llamacpp-gpt-oss-api/chat-template.jinja:/home/llamacpp/chat-template.jinja:ro',
            '-e', 'CUDA_VISIBLE_DEVICES=0',
            '-e', 'NVIDIA_VISIBLE_DEVICES=0',
            '-e', f'MODEL_NAME={self.config["model"]["name"]}',
            '-e', f'MODEL_VARIANT={self.config["model"]["variant"]}',
            image_name
        ] + cmd
        
        logger.info(f"Starting container with CLI command: {' '.join(docker_cmd)}")
        
        # Start the container
        start_process = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await start_process.communicate()
        
        if start_process.returncode != 0:
            error_msg = f"Failed to start container: {stderr.decode().strip()}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        container_id = stdout.decode().strip()
        logger.info(f"Container started with ID: {container_id[:12]}")
        self.start_time = datetime.now()
        
        # Start log reader
        asyncio.create_task(self.read_docker_logs_cli())
        
        # Monitor container startup for a few seconds
        logger.info("Monitoring container startup...")
        await asyncio.sleep(3)
        
        # Check if container is still running
        check_process = await asyncio.create_subprocess_exec(
            'docker', 'ps', '--filter', f'name={self.container_name}', '--format', '{{.Names}}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await check_process.communicate()
        
        if check_process.returncode == 0:
            container_names = stdout.decode().strip().split('\n')
            if self.container_name not in container_names:
                # Container stopped, get logs for error details
                logs_process = await asyncio.create_subprocess_exec(
                    'docker', 'logs', '--tail', '50', self.container_name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                logs_stdout, logs_stderr = await logs_process.communicate()
                
                error_msg = f"Container failed to start or stopped immediately"
                if logs_stdout:
                    error_msg += f". Logs: {logs_stdout.decode()[-500:]}"
                if logs_stderr:
                    error_msg += f". Errors: {logs_stderr.decode()[-500:]}"
                
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
        
        logger.info("Container successfully started and running")
        return True
    
    async def start_subprocess(self) -> bool:
        """Start llamacpp as subprocess"""
        if self.process and self.process.poll() is None:
            raise HTTPException(status_code=400, detail="LlamaCPP is already running")
        
        try:
            # Pre-flight checks
            logger.log_operation_start("LlamaCPP subprocess startup", mode="subprocess")
            
            # Build and validate command
            cmd = self.build_command()
            cmd_str = ' '.join(cmd)
            logger.info(f"Built llamacpp subprocess command with {len(cmd)} arguments: '{cmd_str[:100]}...'" if len(cmd_str) > 100 else f"Built llamacpp subprocess command: '{cmd_str}'")
            
            # Validate model file exists
            model_path = f"/home/llamacpp/models/{self.config['model']['name']}-{self.config['model']['variant']}.gguf"
            logger.info(f"Validating model file at path='{model_path}'")
            if not os.path.exists(model_path):
                error_msg = f"Model file not found: {model_path}"
                logger.error(f"Model file validation failed: {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Validate chat template file exists
            template_path = "/home/alec/git/llamacpp-gpt-oss-api/chat-template.jinja"
            logger.info(f"Validating chat template file at path='{template_path}'")
            if not os.path.exists(template_path):
                error_msg = f"Chat template file not found: {template_path}"
                logger.error(f"Chat template validation failed: {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Check if llama-server executable exists
            try:
                llama_server_check = subprocess.run(
                    ["which", "llama-server"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if llama_server_check.returncode != 0:
                    error_msg = "llama-server executable not found in PATH"
                    logger.error(error_msg)
                    raise HTTPException(status_code=500, detail=error_msg)
                logger.info(f"Found llama-server at: {llama_server_check.stdout.strip()}")
            except Exception as e:
                error_msg = f"Failed to check llama-server executable: {e}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            
            logger.info("All pre-flight checks passed, starting process...")
            
            # Start the process with separate stderr capture
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Separate stderr for better error capture
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            logger.info(f"Process started with PID: {self.process.pid}")
            self.start_time = datetime.now()
            
            # Start log reader task
            self.log_reader_task = asyncio.create_task(self.read_subprocess_logs())
            
            # Wait and monitor startup
            logger.info("Monitoring process startup...")
            startup_timeout = 10  # seconds
            check_interval = 0.5  # seconds
            
            for i in range(int(startup_timeout / check_interval)):
                await asyncio.sleep(check_interval)
                
                # Check if process is still running
                if self.process.poll() is not None:
                    # Process terminated, get exit code and stderr
                    exit_code = self.process.returncode
                    stdout, stderr = self.process.communicate()
                    
                    logger.error(f"Process terminated unexpectedly with exit_code={exit_code}")
                    logger.error(f"Process stdout: '{stdout}'")
                    logger.error(f"Process stderr: '{stderr}'")
                    
                    error_msg = f"Process terminated immediately (exit code: {exit_code})"
                    if stderr:
                        error_msg += f". Error: {stderr.strip()}"
                    if stdout:
                        error_msg += f". Output: {stdout.strip()}"
                    
                    raise HTTPException(status_code=500, detail=error_msg)
                
                logger.debug(f"Process still running after {(i+1) * check_interval:.1f}s")
            
            logger.log_operation_success("LlamaCPP subprocess startup", duration=startup_timeout, pid=self.process.pid)
            return True
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            error_msg = f"Unexpected error starting llamacpp: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise HTTPException(status_code=500, detail=error_msg)
    
    async def start(self) -> bool:
        """Start the llamacpp service"""
        if self.use_docker:
            return await self.start_docker()
        else:
            return await self.start_subprocess()
    
    async def stop_docker(self) -> bool:
        """Stop Docker container"""
        try:
            if DOCKER_AVAILABLE and docker_client:
                return await self.stop_docker_sdk()
            else:
                return await self.stop_docker_cli()
                
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            logger.error(f"Failed to stop Docker container: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def stop_docker_sdk(self) -> bool:
        """Stop Docker container using Python SDK"""
        try:
            if not self.docker_container:
                container = docker_client.containers.get(self.container_name)
            else:
                container = self.docker_container
                
            container.stop(timeout=10)
            container.remove(force=True)
            
            self.docker_container = None
            self.start_time = None
            return True
            
        except docker.errors.NotFound:
            raise HTTPException(status_code=400, detail="LlamaCPP container not found")
    
    async def stop_docker_cli(self) -> bool:
        """Stop Docker container using CLI commands"""
        # Check if container exists and is running
        check_process = await asyncio.create_subprocess_exec(
            'docker', 'ps', '--filter', f'name={self.container_name}', '--format', '{{.Names}}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await check_process.communicate()
        
        if check_process.returncode != 0:
            raise HTTPException(status_code=500, detail="Failed to check container status")
        
        container_names = stdout.decode().strip().split('\n')
        if self.container_name not in container_names:
            raise HTTPException(status_code=400, detail="LlamaCPP container not found or not running")
        
        # Stop the container
        logger.info(f"Stopping container: {self.container_name}")
        stop_process = await asyncio.create_subprocess_exec(
            'docker', 'stop', self.container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await stop_process.communicate()
        
        if stop_process.returncode != 0:
            logger.warning(f"Failed to stop container gracefully: {stderr.decode()}")
        
        # Remove the container
        logger.info(f"Removing container: {self.container_name}")
        remove_process = await asyncio.create_subprocess_exec(
            'docker', 'rm', '-f', self.container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await remove_process.communicate()
        
        self.docker_container = None
        self.start_time = None
        return True
    
    async def stop_subprocess(self) -> bool:
        """Stop subprocess"""
        if not self.process or self.process.poll() is not None:
            raise HTTPException(status_code=400, detail="LlamaCPP is not running")
        
        try:
            # Cancel log reader task
            if self.log_reader_task:
                self.log_reader_task.cancel()
            
            # Send SIGTERM for graceful shutdown
            self.process.terminate()
            
            # Wait up to 10 seconds for graceful shutdown
            for _ in range(10):
                if self.process.poll() is not None:
                    break
                await asyncio.sleep(1)
            
            # Force kill if still running
            if self.process.poll() is None:
                self.process.kill()
                await asyncio.sleep(1)
            
            self.process = None
            self.start_time = None
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop llamacpp: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def stop(self) -> bool:
        """Stop the llamacpp service"""
        if self.use_docker:
            return await self.stop_docker()
        else:
            return await self.stop_subprocess()
    
    async def restart(self) -> bool:
        """Restart the llamacpp service"""
        try:
            await self.stop()
        except:
            pass  # Ignore stop errors during restart
        
        await asyncio.sleep(2)
        return await self.start()
    
    async def get_llamacpp_health(self) -> Dict[str, Any]:
        """Check health of the actual llamacpp service"""
        try:
            async with httpx.AsyncClient() as client:
                base_host = 'localhost' if not self.use_docker else self.container_name
                response = await client.get(f"http://{base_host}:8080/health", timeout=5.0)
                return {
                    "healthy": response.status_code == 200,
                    "status_code": response.status_code,
                    "response": response.text,
                }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def add_log_line(self, line: str):
        """Add a log line and broadcast to websocket clients"""
        # Add to buffer
        self.log_buffer.append({
            "message": line,
            "timestamp": datetime.now().isoformat()
        })
        
        # Broadcast to websocket clients
        disconnected_clients = []
        for client in self.websocket_clients:
            try:
                await client.send_json({
                    "type": "log",
                    "data": line,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception:
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.websocket_clients.remove(client)
    
    def get_logs(self, lines: int = 100) -> List[Dict[str, str]]:
        """Get recent log lines"""
        try:
            lines = int(lines)
        except Exception:
            lines = 100
        # Return a shallow copy slice to avoid exposing internal deque refs
        return list(self.log_buffer)[-max(0, lines):]
    
    async def read_docker_logs_cli(self):
        """Read logs from Docker container using CLI"""
        try:
            # Use docker logs with follow flag to stream logs
            process = await asyncio.create_subprocess_exec(
                'docker', 'logs', '-f', self.container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            logger.info(f"Started reading Docker logs for container: {self.container_name}")
            
            async for line in process.stdout:
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str:  # Only add non-empty lines
                        await self.add_log_line(line_str)
                        
        except Exception as e:
            logger.error(f"Error reading Docker logs via CLI: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of llamacpp"""
        is_running = False
        pid: Optional[int] = None
        
        if self.use_docker:
            if DOCKER_AVAILABLE and docker_client:
                try:
                    container = docker_client.containers.get(self.container_name)
                    is_running = container.status == "running"
                    if is_running:
                        pid = container.attrs.get('State', {}).get('Pid')
                except Exception:
                    pass
            else:
                try:
                    result = subprocess.run(
                        ['docker', 'ps', '--filter', f'name={self.container_name}', '--format', '{{.Names}}'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        container_names = result.stdout.strip().split('\n')
                        is_running = self.container_name in container_names and bool(result.stdout.strip())
                        if is_running:
                            pid_result = subprocess.run(
                                ['docker', 'inspect', '--format', '{{.State.Pid}}', self.container_name],
                                capture_output=True,
                                text=True,
                                timeout=5
                            )
                            if pid_result.returncode == 0:
                                try:
                                    pid = int(pid_result.stdout.strip())
                                except Exception:
                                    pass
                except Exception as e:
                    logger.warning(f"Error checking Docker container status: {e}")
        else:
            is_running = self.process is not None and self.process.poll() is None
            if is_running and self.process is not None:
                pid = self.process.pid
        
        status: Dict[str, Any] = {
            "running": is_running,
            "pid": pid,
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time and is_running else 0,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "mode": "docker" if self.use_docker else "subprocess",
            "config": self.config,
            "model": {
                "name": self.config["model"]["name"],
                "variant": self.config["model"]["variant"],
                "context_size": self.config["model"]["context_size"],
                "gpu_layers": self.config["model"]["gpu_layers"],
                "n_cpu_moe": self.config["model"]["n_cpu_moe"],
            }
        }
        
        if is_running and pid:
            try:
                process = psutil.Process(pid)
                memory_info = process.memory_info()
                status["resources"] = {
                    "cpu_percent": process.cpu_percent(),
                    "memory_mb": memory_info.rss / (1024 * 1024),
                    "memory_percent": process.memory_percent(),
                    "num_threads": process.num_threads(),
                }
            except Exception:
                pass
        
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                values = result.stdout.strip().split(", ")
                if len(values) >= 4:
                    status["gpu"] = {
                        "vram_used_mb": int(values[0]),
                        "vram_total_mb": int(values[1]),
                        "gpu_usage_percent": float(values[2]),
                        "temperature_c": float(values[3]),
                    }
        except Exception:
            pass
        
        return status
    
    # --- End of LlamaCPPManager ---


# Simple in-memory download manager for model files
@dataclass
class DownloadRecord:
    model_id: str
    repo_id: str
    filename: str
    status: str  # queued | downloading | completed | failed | cancelled
    progress: float
    total_size: int
    downloaded_size: int
    speed: float
    eta: float
    error: Optional[str] = None
    parts_info: Optional[Dict[str, Any]] = None  # For multi-part downloads


class ModelDownloadManager:
    def __init__(self):
        self._downloads: Dict[str, DownloadRecord] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._cancel_events: Dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        # Models directory shared with llamacpp
        self.models_dir = Path("/home/llamacpp/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _derive_model_id(self, filename: str) -> str:
        # Strip extension
        base = filename[:-5] if filename.endswith('.gguf') else Path(filename).stem
        return base

    async def list_local_models(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        seen_models = set()  # Track models to avoid duplicates for multi-part files
        
        # Scan both root directory and subdirectories for .gguf files
        for entry in self.models_dir.rglob('*.gguf'):
            try:
                stat = entry.stat()
                name, variant = self._parse_name_variant(entry.name)
                
                # Create a unique identifier for this model (combining name and variant)
                model_key = f"{name}:{variant or 'unknown'}"
                
                # For multi-part files, only add the model once
                if self._is_multipart_file(entry.name):
                    # Only include the first part to represent the complete model
                    if '-00001-of-' not in entry.name:
                        continue
                    
                    # Calculate total size for multi-part files
                    all_parts = self._get_multipart_files(entry.name)
                    total_size = 0
                    for part_file in all_parts:
                        part_path = entry.parent / part_file
                        if part_path.exists():
                            total_size += part_path.stat().st_size
                    
                    # Use total size and latest modification time
                    latest_mtime = stat.st_mtime
                    for part_file in all_parts:
                        part_path = entry.parent / part_file
                        if part_path.exists():
                            part_stat = part_path.stat()
                            if part_stat.st_mtime > latest_mtime:
                                latest_mtime = part_stat.st_mtime
                else:
                    total_size = stat.st_size
                    latest_mtime = stat.st_mtime
                
                # Only add if we haven't seen this model before
                if model_key not in seen_models:
                    seen_models.add(model_key)
                    items.append({
                        "name": name,
                        "variant": variant or "unknown",
                        "size": total_size,
                        "status": "available",
                        "lastModified": datetime.fromtimestamp(latest_mtime).isoformat()
                    })
                    
            except Exception as e:
                # Skip any unreadable files but log the error
                logger.debug(f"Skipping unreadable model file {entry}: {e}")
                continue
                
        # Merge in any actively downloading items
        async with self._lock:
            for rec in self._downloads.values():
                if rec.status in ("queued", "downloading"):
                    name, variant = self._parse_name_variant(rec.filename)
                    model_key = f"{name}:{variant or 'unknown'}"
                    
                    # Only add if not already present from local scan
                    if model_key not in seen_models:
                        items.append({
                            "name": name,
                            "variant": variant or "unknown",
                            "size": rec.total_size or 0,
                            "status": "downloading",
                            "downloadProgress": rec.progress
                        })
                        
        return items

    def _parse_name_variant(self, filename: str) -> (str, Optional[str]):
        # Remove extension
        stem = filename[:-5] if filename.endswith('.gguf') else Path(filename).stem
        
        # Handle multi-part files by removing the part suffix first
        import re
        multipart_pattern = r'(.+)-(\d{5}-of-\d{5})$'
        multipart_match = re.match(multipart_pattern, stem)
        if multipart_match:
            stem = multipart_match.group(1)
        
        # Try dot pattern: ModelName.Q4_K_M
        if '.' in stem:
            parts = stem.split('.')
            if parts[-1].startswith('Q'):
                return ('.'.join(parts[:-1]), parts[-1])
        
        # Try hyphen pattern: ModelName-Q4_K_M
        if '-' in stem:
            parts = stem.split('-')
            if parts[-1].startswith('Q'):
                return ('-'.join(parts[:-1]), parts[-1])
        
        # Try to extract variant from end of filename (e.g., devstralQ4_K_M)
        # Look for common quantization patterns like Q4_K_M, Q5_K_S, etc.
        pattern = r'(.*?)(Q\d+_[KS](_[MS])?)$'
        match = re.match(pattern, stem)
        if match:
            return (match.group(1), match.group(2))
        
        # If no pattern matches, check if the entire name contains a quantization pattern
        # This handles cases where the quantization is embedded in the name
        if re.search(r'Q\d+_[KS](_[MS])?', stem):
            # Extract the quantization pattern
            quant_match = re.search(r'(Q\d+_[KS](_[MS])?)', stem)
            if quant_match:
                quant = quant_match.group(1)
                # Remove the quantization from the name
                name = stem.replace(quant, '')
                # Clean up any trailing underscores or hyphens
                name = name.rstrip('_-')
                if name:
                    return (name, quant)
        
        return (stem, None)

    def _is_multipart_file(self, filename: str) -> bool:
        """Check if filename indicates a multi-part GGUF file"""
        import re
        # Pattern for multi-part files: filename-00001-of-00002.gguf
        return bool(re.search(r'-\d{5}-of-\d{5}\.gguf$', filename))
    
    def _get_multipart_files(self, filename: str) -> List[str]:
        """Get all part filenames for a multi-part download"""
        import re
        
        # Extract the base name and part info
        match = re.match(r'(.+)-(\d{5})-of-(\d{5})\.gguf$', filename)
        if not match:
            return [filename]  # Not a multi-part file
        
        base_name = match.group(1)
        total_parts = int(match.group(3))
        
        # Generate all part filenames
        part_files = []
        for i in range(1, total_parts + 1):
            part_filename = f"{base_name}-{i:05d}-of-{total_parts:05d}.gguf"
            part_files.append(part_filename)
        
        return part_files
    


    async def get_downloads(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return [asdict(rec) for rec in self._downloads.values()]

    async def start_download(self, repo_id: str, filename: str) -> DownloadRecord:
        model_id = self._derive_model_id(filename)
        dest_path = self.models_dir / filename
        async with self._lock:
            if model_id in self._downloads and self._downloads[model_id].status in ("queued", "downloading"):
                raise HTTPException(status_code=400, detail=f"Download already in progress for {model_id}")
            
            # Check if this is a multi-part file and handle accordingly
            is_multipart = self._is_multipart_file(filename)
            if is_multipart:
                logger.info(f"Detected multi-part file: {filename}")
                # For multi-part files, we need to download all parts
                part_files = self._get_multipart_files(filename)
                logger.info(f"Multi-part files to download: {part_files}")
                
                # Create a download record for the combined model
                rec = DownloadRecord(
                    model_id=model_id,
                    repo_id=repo_id,
                    filename=filename,
                    status="queued",
                    progress=0.0,
                    total_size=0,
                    downloaded_size=0,
                    speed=0.0,
                    eta=0.0,
                )
                self._downloads[model_id] = rec
                cancel_event = asyncio.Event()
                self._cancel_events[model_id] = cancel_event
                
                # Launch background task for multi-part download
                task = asyncio.create_task(self._run_multipart_download(model_id, repo_id, part_files, cancel_event))
                self._tasks[model_id] = task
                return rec
            else:
                # Initialize record for single file
                rec = DownloadRecord(
                    model_id=model_id,
                    repo_id=repo_id,
                    filename=filename,
                    status="queued",
                    progress=0.0,
                    total_size=0,
                    downloaded_size=0,
                    speed=0.0,
                    eta=0.0,
                )
                self._downloads[model_id] = rec
                cancel_event = asyncio.Event()
                self._cancel_events[model_id] = cancel_event
                # Launch background task
                task = asyncio.create_task(self._run_download(model_id, repo_id, filename, dest_path, cancel_event))
                self._tasks[model_id] = task
                return rec

    async def cancel_download(self, model_id: str) -> bool:
        async with self._lock:
            if model_id not in self._downloads:
                raise HTTPException(status_code=404, detail=f"No download found for {model_id}")
            rec = self._downloads[model_id]
            if rec.status not in ("queued", "downloading"):
                raise HTTPException(status_code=400, detail=f"Download not in progress for {model_id}")
            self._cancel_events[model_id].set()
            return True

    async def _run_download(self, model_id: str, repo_id: str, filename: str, dest_path: Path, cancel_event: asyncio.Event):
        url = hf_hub_url(repo_id=repo_id, filename=filename)
        headers = {"User-Agent": "llamacpp-gpt-oss-api/1.0"}
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        temp_path = dest_path.with_suffix('.part')
        
        # Ensure parent directories exist
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        last_report_time = start_time
        last_reported = 0

        async with self._lock:
            rec = self._downloads[model_id]
            rec.status = "downloading"

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
                # HEAD to get size
                head = await client.head(url, headers=headers)
                if head.status_code >= 400:
                    raise HTTPException(status_code=502, detail=f"Failed to fetch file headers: HTTP {head.status_code}")
                total_size = int(head.headers.get('Content-Length', '0'))
                async with self._lock:
                    rec.total_size = total_size
                # Stream GET
                async with client.stream('GET', url, headers=headers) as resp:
                    if resp.status_code >= 400:
                        raise HTTPException(status_code=502, detail=f"Failed to start download: HTTP {resp.status_code}")
                    with open(temp_path, 'wb') as f:
                        async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                            if cancel_event.is_set():
                                raise asyncio.CancelledError()
                            if not chunk:
                                continue
                            f.write(chunk)
                            async with self._lock:
                                rec.downloaded_size += len(chunk)
                                if rec.total_size > 0:
                                    rec.progress = min(100.0, (rec.downloaded_size / rec.total_size) * 100)
                            # update speed periodically
                            now = time.time()
                            if now - last_report_time >= 0.5:
                                bytes_delta = rec.downloaded_size - last_reported
                                time_delta = now - last_report_time
                                async with self._lock:
                                    rec.speed = bytes_delta / time_delta if time_delta > 0 else 0
                                    remaining = max(0, rec.total_size - rec.downloaded_size)
                                    rec.eta = (remaining / rec.speed) if rec.speed > 0 else 0
                                last_report_time = now
                                last_reported = rec.downloaded_size
                # Move temp to final
                temp_path.replace(dest_path)
                async with self._lock:
                    rec.status = "completed"
                    rec.progress = 100.0
                    rec.speed = (rec.total_size / max(1e-6, (time.time() - start_time))) if rec.total_size else 0.0
                    rec.eta = 0.0
        except asyncio.CancelledError:
            # Cleanup and mark cancelled
            try:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            async with self._lock:
                rec.status = "cancelled"
                rec.error = "Cancelled by user"
                rec.eta = 0.0
        except HTTPException as e:
            try:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            async with self._lock:
                rec.status = "failed"
                rec.error = e.detail if isinstance(e.detail, str) else str(e.detail)
        except Exception as e:
            try:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            async with self._lock:
                rec.status = "failed"
                rec.error = str(e)
        finally:
            # Cleanup task maps
            async with self._lock:
                self._tasks.pop(model_id, None)
                self._cancel_events.pop(model_id, None)

    async def _run_multipart_download(self, model_id: str, repo_id: str, part_files: List[str], cancel_event: asyncio.Event):
        """Download all parts of a multi-part GGUF file. llamacpp will automatically load all parts when given the first one."""
        start_time = time.time()
        
        async with self._lock:
            rec = self._downloads[model_id]
            rec.status = "downloading"
            # Add parts info for better progress tracking
            rec.parts_info = {"total": len(part_files), "completed": 0, "current": ""}
        
        try:
            # Get total size of all parts
            total_size = 0
            headers = {"User-Agent": "llamacpp-gpt-oss-api/1.0"}
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
            
            async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
                # Get sizes of all parts
                logger.info(f"Getting file sizes for {len(part_files)} parts...")
                for part_file in part_files:
                    url = hf_hub_url(repo_id=repo_id, filename=part_file)
                    head = await client.head(url, headers=headers)
                    if head.status_code >= 400:
                        raise HTTPException(status_code=502, detail=f"Failed to fetch headers for {part_file}: HTTP {head.status_code}")
                    part_size = int(head.headers.get('Content-Length', '0'))
                    total_size += part_size
                
                async with self._lock:
                    rec.total_size = total_size
                
                # Download all parts
                total_downloaded = 0
                
                for i, part_file in enumerate(part_files):
                    if cancel_event.is_set():
                        raise asyncio.CancelledError()
                    
                    logger.info(f"Downloading part {i+1}/{len(part_files)}: {part_file}")
                    async with self._lock:
                        rec.parts_info["current"] = f"Part {i+1}/{len(part_files)}: {part_file}"
                    
                    url = hf_hub_url(repo_id=repo_id, filename=part_file)
                    temp_path = self.models_dir / f"{part_file}.part"
                    final_path = self.models_dir / part_file
                    
                    # Ensure parent directories exist
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    async with client.stream('GET', url, headers=headers) as resp:
                        if resp.status_code >= 400:
                            raise HTTPException(status_code=502, detail=f"Failed to download {part_file}: HTTP {resp.status_code}")
                        
                        with open(temp_path, 'wb') as f:
                            async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                                if cancel_event.is_set():
                                    raise asyncio.CancelledError()
                                if not chunk:
                                    continue
                                f.write(chunk)
                                total_downloaded += len(chunk)
                                
                                async with self._lock:
                                    rec.downloaded_size = total_downloaded
                                    if rec.total_size > 0:
                                        rec.progress = min(100.0, (total_downloaded / rec.total_size) * 100)
                    
                    # Move temp to final
                    temp_path.replace(final_path)
                    
                    async with self._lock:
                        rec.parts_info["completed"] = i + 1
                    
                    logger.info(f"Completed part {i+1}/{len(part_files)}: {part_file}")
                
                logger.info(f"Successfully downloaded all {len(part_files)} parts. llamacpp will automatically load all parts when using the first part.")
                
                async with self._lock:
                    rec.status = "completed"
                    rec.progress = 100.0
                    rec.speed = (rec.total_size / max(1e-6, (time.time() - start_time))) if rec.total_size else 0.0
                    rec.eta = 0.0
                    rec.parts_info["current"] = f"Completed all {len(part_files)} parts"
                    
        except asyncio.CancelledError:
            # Cleanup any partial downloads
            for part_file in part_files:
                try:
                    temp_path = self.models_dir / f"{part_file}.part"
                    final_path = self.models_dir / part_file
                    if temp_path.exists():
                        temp_path.unlink()
                    if final_path.exists():
                        final_path.unlink()
                except Exception:
                    pass
            async with self._lock:
                rec.status = "cancelled"
                rec.error = "Cancelled by user"
                rec.eta = 0.0
        except Exception as e:
            # Cleanup any partial downloads
            for part_file in part_files:
                try:
                    temp_path = self.models_dir / f"{part_file}.part"
                    final_path = self.models_dir / part_file
                    if temp_path.exists():
                        temp_path.unlink()
                    if final_path.exists():
                        final_path.unlink()
                except Exception:
                    pass
            async with self._lock:
                rec.status = "failed"
                rec.error = str(e)
        finally:
            # Cleanup task maps
            async with self._lock:
                self._tasks.pop(model_id, None)
                self._cancel_events.pop(model_id, None)
    
    async def read_subprocess_logs(self):
        """Read logs from subprocess and broadcast to websocket clients"""
        if not self.process:
            return
        
        try:
            # Create tasks to read both stdout and stderr
            async def read_stdout():
                if self.process.stdout:
                    while self.process and self.process.poll() is None:
                        line = self.process.stdout.readline()
                        if line:
                            await self.add_log_line(f"[STDOUT] {line.strip()}")
                        else:
                            await asyncio.sleep(0.1)
            
            async def read_stderr():
                if self.process.stderr:
                    while self.process and self.process.poll() is None:
                        line = self.process.stderr.readline()
                        if line:
                            await self.add_log_line(f"[STDERR] {line.strip()}")
                        else:
                            await asyncio.sleep(0.1)
            
            # Run both readers concurrently
            await asyncio.gather(read_stdout(), read_stderr(), return_exceptions=True)
            
        except asyncio.CancelledError:
            logger.info("Log reader task was cancelled")
        except Exception as e:
            logger.error(f"Error reading subprocess logs: {e}", exc_info=True)
    
    async def read_docker_logs(self):
        """Read logs from Docker container using docker CLI as fallback"""
        try:
            if docker_client and DOCKER_AVAILABLE:
                # Try using Python Docker client first
                container = self.docker_container or docker_client.containers.get(self.container_name)
                # Stream logs
                for line in container.logs(stream=True, follow=True):
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                    await self.add_log_line(line.strip())
            else:
                # Fallback to docker CLI subprocess
                await self.read_docker_logs_cli()
                
        except Exception as e:
            logger.warning(f"Error reading Docker logs with Python client: {e}")
            # Fallback to CLI method
            try:
                await self.read_docker_logs_cli()
            except Exception as cli_error:
                logger.error(f"Error reading Docker logs with CLI: {cli_error}")
    
    async def read_docker_logs_cli(self):
        """Read logs from Docker container using CLI"""
        try:
            # Use docker logs with follow flag to stream logs
            process = await asyncio.create_subprocess_exec(
                'docker', 'logs', '-f', self.container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            logger.info(f"Started reading Docker logs for container: {self.container_name}")
            
            async for line in process.stdout:
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str:  # Only add non-empty lines
                        await self.add_log_line(line_str)
                        
        except Exception as e:
            logger.error(f"Error reading Docker logs via CLI: {e}")
    
    async def check_existing_container(self):
        """Check if target container is already running and start log reading"""
        try:
            # Wait a bit for the manager to initialize
            await asyncio.sleep(2)
            
            # Check if container is running using CLI
            process = await asyncio.create_subprocess_exec(
                'docker', 'ps', '--filter', f'name={self.container_name}', '--format', '{{.Names}}',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                container_names = stdout.decode().strip().split('\n')
                if self.container_name in container_names:
                    logger.info(f"Found existing container {self.container_name}, starting log reading")
                    self.log_reader_task = asyncio.create_task(self.read_docker_logs_cli())
                else:
                    logger.info(f"Container {self.container_name} not found or not running")
            else:
                logger.warning(f"Failed to check for existing container: {stderr.decode()}")
                
        except Exception as e:
            logger.warning(f"Error checking for existing container: {e}")

    async def add_log_line(self, line: str):
        """Add a log line and broadcast to websocket clients"""
        # Add to buffer
        self.log_buffer.append({
            "message": line,
            "timestamp": datetime.now().isoformat()
        })
        
        # Broadcast to websocket clients
        disconnected_clients = []
        for client in self.websocket_clients:
            try:
                await client.send_json({
                    "type": "log",
                    "data": line,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception:
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.websocket_clients.remove(client)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of llamacpp"""
        is_running = False
        pid = None
        
        if self.use_docker:
            if DOCKER_AVAILABLE and docker_client:
                # Use Python SDK
                try:
                    container = docker_client.containers.get(self.container_name)
                    is_running = container.status == "running"
                    if is_running:
                        pid = container.attrs['State']['Pid']
                except:
                    pass
            else:
                # Use Docker CLI
                try:
                    import subprocess
                    result = subprocess.run(
                        ['docker', 'ps', '--filter', f'name={self.container_name}', '--format', '{{.Names}}'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        container_names = result.stdout.strip().split('\n')
                        is_running = self.container_name in container_names and result.stdout.strip()
                        
                        if is_running:
                            # Get PID using docker inspect
                            pid_result = subprocess.run(
                                ['docker', 'inspect', '--format', '{{.State.Pid}}', self.container_name],
                                capture_output=True,
                                text=True,
                                timeout=5
                            )
                            if pid_result.returncode == 0:
                                try:
                                    pid = int(pid_result.stdout.strip())
                                except:
                                    pass
                except Exception as e:
                    logger.warning(f"Error checking Docker container status: {e}")
        else:
            is_running = self.process is not None and self.process.poll() is None
            if is_running:
                pid = self.process.pid
        
        status = {
            "running": is_running,
            "pid": pid,
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time and is_running else 0,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "mode": "docker" if self.use_docker else "subprocess",
            "config": self.config,
            "model": {
                "name": self.config["model"]["name"],
                "variant": self.config["model"]["variant"],
                "context_size": self.config["model"]["context_size"],
                "gpu_layers": self.config["model"]["gpu_layers"],
                "n_cpu_moe": self.config["model"]["n_cpu_moe"],
            }
        }
        
        # Add resource usage if running
        if is_running and pid:
            try:
                process = psutil.Process(pid)
                memory_info = process.memory_info()
                status["resources"] = {
                    "cpu_percent": process.cpu_percent(),
                    "memory_mb": memory_info.rss / (1024 * 1024),
                    "memory_percent": process.memory_percent(),
                    "num_threads": process.num_threads(),
                }
            except:
                pass
        
        # Add GPU info
        try:
            # First check if nvidia-smi is available
            which_result = subprocess.run(
                ["which", "nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if which_result.returncode == 0:
                # nvidia-smi is available, try to get GPU info
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu", 
                     "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    values = result.stdout.strip().split(", ")
                    if len(values) >= 4:
                        status["gpu"] = {
                            "vram_used_mb": int(values[0]),
                            "vram_total_mb": int(values[1]),
                            "gpu_usage_percent": float(values[2]),
                            "temperature_c": float(values[3])
                        }
                else:
                    logger.warning(f"nvidia-smi command failed with return code: {result.returncode}")
            else:
                logger.warning("nvidia-smi not found in PATH")
                status["gpu"] = {
                    "status": "unavailable",
                    "reason": "nvidia-smi not found"
                }
        except Exception as e:
            logger.warning(f"Could not get GPU info: {str(e)}")
            status["gpu"] = {
                "status": "error",
                "reason": str(e)
            }
        
        return status
    
    def get_logs(self, lines: int = 100) -> List[Dict[str, str]]:
        """Get recent log lines"""
        try:
            lines = int(lines)
        except Exception:
            lines = 100
        # Return a shallow copy slice to avoid exposing internal deque refs
        return list(self.log_buffer)[-max(0, lines):]
    
    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration"""
        # Deep merge configs
        for category in ["model", "sampling", "performance", "server"]:
            if category in new_config:
                if category not in self.config:
                    self.config[category] = {}
                self.config[category].update(new_config[category])
        
        # Save config to file for persistence
        config_file = Path("/tmp/llamacpp_config.json")
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2)
        
        return self.config
    
    async def get_llamacpp_health(self) -> Dict[str, Any]:
        """Check health of the actual llamacpp service"""
        try:
            async with httpx.AsyncClient() as client:
                # Try to reach llamacpp health endpoint
                response = await client.get(
                    f"http://{'localhost' if not self.use_docker else self.container_name}:8080/health",
                    timeout=5.0
                )
                return {
                    "healthy": response.status_code == 200,
                    "status_code": response.status_code,
                    "response": response.text
                }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

# Initialize manager
manager = LlamaCPPManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting LlamaCPP Management API")
    
    # Load saved config if exists
    config_file = Path("/tmp/llamacpp_config.json")
    if config_file.exists():
        with open(config_file) as f:
            saved_config = json.load(f)
            manager.config = saved_config
    
    yield
    
    # Shutdown
    logger.info("Shutting down LlamaCPP Management API")
    if not manager.use_docker and manager.process and manager.process.poll() is None:
        await manager.stop()

# Create FastAPI app
app = FastAPI(
    title="LlamaCPP Management API",
    description="API for managing LlamaCPP model deployments",
    version="2.0.0",
    lifespan=lifespan
)

# Create a logging middleware
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log the request with our enhanced logger
        # Skip logging for WebSocket connections
        if not request.url.path.endswith("/stream"):
            logger.log_api_call(
                request.method, 
                request.url.path, 
                response.status_code, 
                duration
            )
        
        return response

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Health check endpoint
@app.get("/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mode": "docker" if manager.use_docker else "subprocess"
    }

# Service control endpoints
@app.post("/api/v1/service/start")
async def start_service():
    """Start the llamacpp service"""
    try:
        logger.log_operation_start("Service start request", endpoint="/api/v1/service/start")
        success = await manager.start()
        status = manager.get_status()
        logger.log_operation_success("Service start request", success=success, mode=status.get("mode", "unknown"))
        return {
            "success": success, 
            "status": status,
            "message": "Service started successfully" if success else "Service failed to start"
        }
    except HTTPException as e:
        logger.log_operation_failure("Service start request", f"HTTP error: {e.detail}")
        raise e
    except Exception as e:
        logger.log_operation_failure("Service start request", f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/api/v1/service/stop")
async def stop_service():
    """Stop the llamacpp service"""
    success = await manager.stop()
    return {"success": success, "status": manager.get_status()}

@app.post("/api/v1/service/restart")
async def restart_service():
    """Restart the llamacpp service"""
    success = await manager.restart()
    return {"success": success, "status": manager.get_status()}

@app.get("/api/v1/service/status")
async def get_service_status():
    """Get current service status including resources"""
    status = manager.get_status()
    
    # Add llamacpp health check
    if status["running"]:
        health = await manager.get_llamacpp_health()
        status["llamacpp_health"] = health
    
    return status

# Configuration endpoints
@app.get("/api/v1/service/config")
async def get_config():
    """Get current configuration"""
    return {
        "config": manager.config,
        "command": " ".join(manager.build_command()),
        "editable_fields": {
            "model": ["context_size", "gpu_layers"],
            "sampling": ["temperature", "top_p", "top_k", "min_p", "repeat_penalty", 
                        "frequency_penalty", "presence_penalty", "dry_multiplier"],
            "performance": ["threads", "batch_size", "ubatch_size", "num_predict"],
            "server": ["api_key"]
        }
    }

def _merge_and_persist_config(new_config: Dict[str, Any]) -> Dict[str, Any]:
    # Deep merge into manager.config and persist to disk
    base = manager.config
    for category in ["model", "sampling", "performance", "server"]:
        if category in new_config:
            if category not in base:
                base[category] = {}
            base[category].update(new_config[category])
    # Persist
    config_file = Path("/tmp/llamacpp_config.json")
    with open(config_file, "w") as f:
        json.dump(base, f, indent=2)
    return base


@app.put("/api/v1/service/config")
async def update_config(config: Dict[str, Any]):
    """Update configuration (requires restart to apply)"""
    try:
        updated_config = _merge_and_persist_config(config)
        return {
            "config": updated_config,
            "command": " ".join(manager.build_command()),
            "message": "Configuration updated. Restart service to apply changes."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add aliases without /api prefix for frontend compatibility
@app.get("/v1/service/config")
async def get_config_alias():
    """Get current configuration (frontend compatibility)"""
    return await get_config()

@app.put("/v1/service/config")
async def update_config_alias(config: Dict[str, Any]):
    """Update configuration (frontend compatibility)"""
    return await update_config(config)

@app.post("/v1/service/config/validate")
async def validate_config_alias(payload: Dict[str, Any]):
    """Validate configuration (frontend compatibility)"""
    return await validate_config(payload)

@app.get("/v1/service/status")
async def get_service_status_alias():
    """Get service status (frontend compatibility)"""
    return await get_service_status()

@app.post("/v1/service/action")
async def service_action_alias(payload: Dict[str, Any]):
    """Service action (frontend compatibility)"""
    return await service_action(payload)

@app.get("/v1/resources")
async def get_resources_alias():
    """Get resources (frontend compatibility)"""
    return await get_resources()

@app.post("/api/v1/service/config/validate")
async def validate_config(payload: Dict[str, Any]):
    """Validate configuration without applying"""
    config = payload.get("config", {})
    errors = []
    warnings = []
    
    # Validate model settings
    if "model" in config:
        if "context_size" in config["model"]:
            ctx = config["model"]["context_size"]
            if ctx > 131072:
                errors.append(f"Context size {ctx} exceeds maximum (131072)")
            elif ctx > 65536:
                warnings.append(f"Context size {ctx} is very large, may impact performance")
                
        if "gpu_layers" in config["model"]:
            layers = config["model"]["gpu_layers"]
            if layers < 0 and layers != -1:
                errors.append("GPU layers must be >= 0 or -1 for all")
    
    # Validate sampling settings
    if "sampling" in config:
        if "temperature" in config["sampling"]:
            temp = config["sampling"]["temperature"]
            if not (0 <= temp <= 2):
                errors.append(f"Temperature {temp} must be between 0 and 2")
                
        if "top_p" in config["sampling"]:
            top_p = config["sampling"]["top_p"]
            if not (0 <= top_p <= 1):
                errors.append(f"Top-p {top_p} must be between 0 and 1")
                
        if "top_k" in config["sampling"]:
            top_k = config["sampling"]["top_k"]
            if top_k < 0:
                errors.append(f"Top-k {top_k} must be >= 0")
    
    # Validate performance settings
    if "performance" in config:
        if "batch_size" in config["performance"]:
            batch = config["performance"]["batch_size"]
            if batch < 1:
                errors.append(f"Batch size {batch} must be >= 1")
            elif batch > 4096:
                warnings.append(f"Batch size {batch} is very large, may impact latency")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors if errors else None,
        "warnings": warnings if warnings else None
    }

# Log endpoints
@app.get("/api/v1/logs")
async def get_logs(lines: int = 100):
    """Get recent log lines"""
    logs = manager.get_logs(lines)
    return {
        "logs": logs,
        "total_lines": len(manager.log_buffer)
    }

@app.delete("/api/v1/logs")
async def clear_logs():
    """Clear log buffer"""
    manager.log_buffer.clear()
    return {"message": "Logs cleared"}

@app.websocket("/api/v1/logs/stream")
async def websocket_logs(websocket: WebSocket):
    """Stream logs via WebSocket"""
    await websocket.accept()
    manager.websocket_clients.append(websocket)
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "timestamp": datetime.now().isoformat()
        })
        
        # Send recent logs first
        recent_logs = manager.get_logs(50)
        for log in recent_logs:
            await websocket.send_json({
                "type": "log",
                "data": log["message"],
                "timestamp": log["timestamp"]
            })
        
        # Keep connection alive with periodic pings
        while True:
            await asyncio.sleep(30)
            await websocket.send_json({
                "type": "ping",
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        manager.websocket_clients.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in manager.websocket_clients:
            manager.websocket_clients.remove(websocket)

# Resource monitoring endpoint
@app.get("/api/v1/resources")
async def get_resources():
    """Get system resource usage"""
    try:
        resources = {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            "memory": {
                "total_mb": psutil.virtual_memory().total / (1024 * 1024),
                "used_mb": psutil.virtual_memory().used / (1024 * 1024),
                "available_mb": psutil.virtual_memory().available / (1024 * 1024),
                "percent": psutil.virtual_memory().percent
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Get GPU info using nvidia-smi
        try:
            # First check if nvidia-smi is available
            which_result = subprocess.run(
                ["which", "nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if which_result.returncode == 0:
                # nvidia-smi is available, try to get GPU info
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    values = result.stdout.strip().split(", ")
                    if len(values) >= 5:
                        resources["gpu"] = {
                            "vram_used_mb": int(values[0]),
                            "vram_total_mb": int(values[1]),
                            "usage_percent": float(values[2]),
                            "temperature_c": float(values[3]),
                            "power_watts": float(values[4]) if values[4] != "N/A" else None
                        }
                else:
                    logger.warning(f"nvidia-smi command failed with return code: {result.returncode}")
                    resources["gpu"] = {
                        "status": "error",
                        "reason": f"nvidia-smi command failed with return code: {result.returncode}"
                    }
            else:
                logger.warning("nvidia-smi not found in PATH")
                resources["gpu"] = {
                    "status": "unavailable",
                    "reason": "nvidia-smi not found"
                }
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
            resources["gpu"] = {
                "status": "error",
                "reason": str(e)
            }
        
        return resources
        
    except Exception as e:
        logger.error(f"Error getting resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model info endpoint
@app.get("/api/v1/models/current")
async def get_current_model():
    """Get information about the currently configured model"""
    status = manager.get_status()
    
    return {
        "name": manager.config["model"]["name"],
        "variant": manager.config["model"]["variant"],
        "context_size": manager.config["model"]["context_size"],
        "gpu_layers": manager.config["model"]["gpu_layers"],
        "status": "loaded" if status["running"] else "not_loaded",
        "file_path": f"/home/llamacpp/models/{manager.config['model']['name']}-{manager.config['model']['variant']}.gguf",
        "estimated_vram_gb": 19 if "30B" in manager.config["model"]["name"] else 8  # Rough estimate
    }

# Add alias without /api prefix for frontend compatibility
@app.get("/v1/models/current")
async def get_current_model_alias():
    """Get information about the currently configured model (frontend compatibility)"""
    return await get_current_model()

# Configuration presets
@app.get("/api/v1/config/presets")
async def get_config_presets():
    """Get available configuration presets"""
    return [
        {
            "id": "balanced",
            "name": "Balanced",
            "description": "Good balance between quality and speed",
            "config": {
                "sampling": {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "repeat_penalty": 1.05
                }
            }
        },
        {
            "id": "creative",
            "name": "Creative",
            "description": "More creative and varied outputs",
            "config": {
                "sampling": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 40,
                    "repeat_penalty": 1.0
                }
            }
        },
        {
            "id": "precise",
            "name": "Precise",
            "description": "More deterministic and focused outputs",
            "config": {
                "sampling": {
                    "temperature": 0.3,
                    "top_p": 0.5,
                    "top_k": 10,
                    "repeat_penalty": 1.1
                }
            }
        },
        {
            "id": "fast",
            "name": "Fast Inference",
            "description": "Optimized for speed",
            "config": {
                "performance": {
                    "batch_size": 512,
                    "ubatch_size": 128,
                    "num_predict": 2048
                }
            }
        }
    ]

@app.post("/api/v1/config/presets/{preset_id}/apply")
async def apply_config_preset(preset_id: str):
    """Apply a configuration preset"""
    presets = await get_config_presets()
    preset = next((p for p in presets if p["id"] == preset_id), None)
    
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset {preset_id} not found")
    
    updated_config = manager.update_config(preset["config"])
    
    return {
        "config": updated_config,
        "preset": preset,
        "message": f"Applied preset '{preset['name']}'. Restart service to apply changes."
    }

@app.post("/api/v1/service/action")
async def service_action(payload: Dict[str, Any]):
    """Unified service action endpoint for start/stop/restart with optional config update"""
    action = payload.get("action")
    cfg = payload.get("config")
    if cfg:
        _merge_and_persist_config(cfg)
    if action == "start":
        success = await manager.start()
    elif action == "stop":
        success = await manager.stop()
    elif action == "restart":
        success = await manager.restart()
    else:
        raise HTTPException(status_code=400, detail="Unsupported action. Use 'start', 'stop', or 'restart'.")
    return {"success": success, "status": manager.get_status()}

# --- Model management endpoints ---
download_manager = ModelDownloadManager()


@app.get("/v1/models")
async def list_models():
    try:
        items = await download_manager.list_local_models()
        return {"success": True, "data": items, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models/repo-files")
async def list_repo_files(repo_id: str, revision: str = "main"):
    """List available .gguf files in a HuggingFace model repository.

    Uses the Hugging Face Hub API to enumerate files from the specified repo
    and returns only filenames ending with .gguf. This helps the frontend
    present valid choices and avoid 404s from incorrect filenames.
    """
    try:
        # Validate repo_id format
        if not repo_id or '/' not in repo_id:
            raise HTTPException(status_code=422, detail="Invalid repository ID. Must be in format 'owner/repo'")
            
        api = HfApi(token=os.getenv("HUGGINGFACE_TOKEN"))
        try:
            logger.info(f"Listing repo files for {repo_id} with token: {'*****' if os.getenv('HUGGINGFACE_TOKEN') else 'None'}")
            files = api.list_repo_files(repo_id=repo_id, revision=revision, repo_type="model")
            gguf_files = [f for f in files if f.lower().endswith(".gguf")]
            logger.info(f"Found {len(gguf_files)} GGUF files in repo {repo_id}")
            return {"success": True, "data": {"files": gguf_files}, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error listing repo files for {repo_id}: {str(e)}")
            # Surface clear error up to the client; do not suppress
            raise HTTPException(status_code=502, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in list_repo_files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/models/download")
async def download_model(payload: Dict[str, Any]):
    repo_id = payload.get("repositoryId")
    filename = payload.get("filename")
    if not repo_id or not filename:
        raise HTTPException(status_code=400, detail="'repositoryId' and 'filename' are required")
    if not str(filename).endswith('.gguf'):
        raise HTTPException(status_code=400, detail="filename must end with .gguf")
    try:
        rec = await download_manager.start_download(repo_id=repo_id, filename=filename)
        return {"success": True, "data": asdict(rec), "timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models/downloads")
async def list_downloads():
    try:
        downloads = await download_manager.get_downloads()
        return {"success": True, "data": downloads, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/models/downloads/{model_id}")
async def cancel_download(model_id: str):
    try:
        await download_manager.cancel_download(model_id)
        return {"success": True, "data": {"modelId": model_id, "status": "cancelled"}, "timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Configure custom uvicorn logging to suppress default logs
    log_config = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "minimal": {
                "format": "%(message)s"
            }
        },
        "handlers": {
            "null": {
                "class": "logging.NullHandler",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["null"], "level": "WARNING"},
            "uvicorn.error": {"handlers": ["null"], "level": "WARNING"},
            "uvicorn.access": {"handlers": ["null"], "level": "WARNING"},
        },
    }
    
    # Run with custom logging configuration
    uvicorn.run(app, host="0.0.0.0", port=8700, log_config=log_config)
