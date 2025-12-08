"""
Enhanced FastAPI Backend for LlamaCPP Model Management
Provides APIs for managing llamacpp instances with Docker support
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from enum import Enum
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
import re
import uuid

# Import enhanced logger
from enhanced_logger import enhanced_logger as logger
# Token tracking imports
try:
    from modules.token_tracker import token_tracker  # type: ignore
    from modules.token_middleware import TokenUsageMiddleware  # type: ignore
except ImportError:
    try:
        # Fallback for local development without modules prefix
        from token_tracker import token_tracker  # type: ignore
        from token_middleware import TokenUsageMiddleware  # type: ignore
    except ImportError:
        token_tracker = None  # type: ignore
        class TokenUsageMiddleware(BaseHTTPMiddleware):  # type: ignore
            async def dispatch(self, request, call_next):
                return await call_next(request)
        logger.warning("Token tracker not available")

# Conversation storage imports
try:
    from modules.conversation_store import conversation_store
except ImportError:
    try:
        from conversation_store import conversation_store  # type: ignore
    except ImportError:
        conversation_store = None  # type: ignore
        logger.warning("Conversation store not available")

# Model registry imports
try:
    from modules.model_registry import model_registry
except ImportError:
    try:
        from model_registry import model_registry  # type: ignore
    except ImportError:
        model_registry = None  # type: ignore
        logger.warning("Model registry not available")

# Prompt library imports
try:
    from modules.prompt_library import prompt_library
except ImportError:
    try:
        from prompt_library import prompt_library  # type: ignore
    except ImportError:
        prompt_library = None  # type: ignore
        logger.warning("Prompt library not available")

# Benchmark runner imports
try:
    from modules.benchmark import benchmark_runner, BenchmarkConfig
except ImportError:
    try:
        from benchmark import benchmark_runner, BenchmarkConfig  # type: ignore
    except ImportError:
        benchmark_runner = None  # type: ignore
        BenchmarkConfig = None  # type: ignore
        logger.warning("Benchmark runner not available")

# Batch processor imports
try:
    from modules.batch_processor import batch_processor
except ImportError:
    try:
        from batch_processor import batch_processor  # type: ignore
    except ImportError:
        batch_processor = None  # type: ignore
        logger.warning("Batch processor not available")

# RAG system imports
try:
    from modules.rag import DocumentManager, Document, Domain, QdrantStore, GraphRAG
    from modules.rag.chunkers import FixedChunker, SemanticChunker, RecursiveChunker, ChunkingConfig
    from modules.rag.embedders import LocalEmbedder, APIEmbedder
    from modules.rag.retrievers import VectorRetriever, HybridRetriever, GraphRetriever, RetrievalConfig
    from modules.rag.discovery import DocumentDiscovery
    RAG_AVAILABLE = True
    logger.info("RAG system loaded successfully")
except ImportError as e:
    RAG_AVAILABLE = False
    DocumentManager = None  # type: ignore
    Document = None  # type: ignore
    Domain = None  # type: ignore
    QdrantStore = None  # type: ignore
    GraphRAG = None  # type: ignore
    DocumentDiscovery = None  # type: ignore
    logger.warning(f"RAG system not available: {e}")

# RAG Embedding Configuration
USE_DEPLOYED_EMBEDDINGS = os.getenv("USE_DEPLOYED_EMBEDDINGS", "false").lower() == "true"
# Use Docker network name for inter-container communication
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://llamacpp-embed:8080/v1")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def create_embedder(model_name: Optional[str] = None, use_deployed: Optional[bool] = None):
    """
    Factory function to create an embedder instance.
    
    Args:
        model_name: Name of the embedding model to use
        use_deployed: Whether to use the deployed embedding service.
                     If None, uses the USE_DEPLOYED_EMBEDDINGS env var.
    
    Returns:
        An Embedder instance (LocalEmbedder or APIEmbedder)
    """
    if not RAG_AVAILABLE:
        raise RuntimeError("RAG system not available")
    
    model_name = model_name or DEFAULT_EMBEDDING_MODEL
    use_deployed = USE_DEPLOYED_EMBEDDINGS if use_deployed is None else use_deployed
    
    # Check if the requested model is one supported by the deployed service
    deployed_models = ["nomic-embed-text-v1.5", "e5-mistral-7b", "bge-m3", "gte-Qwen2-1.5B"]
    
    if use_deployed and model_name in deployed_models:
        logger.info(f"Using deployed embedding service for model: {model_name} at {EMBEDDING_SERVICE_URL}")
        # Use APIEmbedder with llama.cpp provider
        return APIEmbedder(
            model_name=model_name,
            api_key="llamacpp-embed",  # Match the API key from embedding service
            base_url=EMBEDDING_SERVICE_URL,
            timeout=300  # Longer timeout for large documents
        )
    else:
        logger.info(f"Using local sentence-transformers embedder for model: {model_name}")
        return LocalEmbedder(model_name=model_name)

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
        self.container_name = os.getenv("LLAMACPP_CONTAINER_NAME", "llamacpp-api")
        # Docker network to attach launched containers to (defaults to compose default)
        self.docker_network = os.getenv("DOCKER_NETWORK", "llama-nexus_default")
        
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
            "template": {
                "directory": os.getenv("TEMPLATE_DIR", "/home/llamacpp/templates"),
                "selected": os.getenv("CHAT_TEMPLATE", "chat-template-oss.jinja"),
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
            "execution": {
                "mode": os.getenv("EXECUTION_MODE", "gpu"),  # "gpu" or "cpu"
                "cuda_devices": os.getenv("CUDA_DEVICES", "all"),  # "all", "0", "0,1", etc.
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8080,
                "api_key": os.getenv("API_KEY", "placeholder-api-key"),
                "metrics": True,
                "embedding": True,
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
        
        # Helper function to add parameter if value is not None/undefined
        def add_param_if_set(cmd_list, flag, value, convert_to_str=True):
            if value is not None and value != "":
                if convert_to_str:
                    cmd_list.extend([flag, str(value)])
                else:
                    cmd_list.extend([flag, value])
        
        cmd = [
            "llama-server",
            "--model", model_path,
            "--host", self.config["server"]["host"],
            "--port", str(self.config["server"]["port"]),
            "--api-key", self.config["server"]["api_key"],
        ]
        
        # Add optional model parameters only if they are set
        add_param_if_set(cmd, "--ctx-size", self.config["model"].get("context_size"))
        
        # Override GPU layers based on execution mode
        execution_mode = self.config.get("execution", {}).get("mode", "gpu")
        if execution_mode == "cpu":
            add_param_if_set(cmd, "--n-gpu-layers", 0)
        else:
            add_param_if_set(cmd, "--n-gpu-layers", self.config["model"].get("gpu_layers"))
        
        add_param_if_set(cmd, "--n-cpu-moe", self.config["model"].get("n_cpu_moe"))
        
        # Add optional RoPE parameters
        add_param_if_set(cmd, "--rope-scaling", self.config["model"].get("rope_scaling"))
        add_param_if_set(cmd, "--rope-freq-base", self.config["model"].get("rope_freq_base"))
        add_param_if_set(cmd, "--rope-freq-scale", self.config["model"].get("rope_freq_scale"))
        
        # Add optional LoRA parameters
        add_param_if_set(cmd, "--lora", self.config["model"].get("lora"))
        add_param_if_set(cmd, "--lora-base", self.config["model"].get("lora_base"))
        add_param_if_set(cmd, "--mmproj", self.config["model"].get("mmproj"))
        
        # Add optional performance parameters
        add_param_if_set(cmd, "--n-predict", self.config["performance"].get("num_predict"))
        add_param_if_set(cmd, "--threads", self.config["performance"].get("threads"))
        add_param_if_set(cmd, "--threads-batch", self.config["performance"].get("threads_batch"))
        add_param_if_set(cmd, "--batch-size", self.config["performance"].get("batch_size"))
        add_param_if_set(cmd, "--ubatch-size", self.config["performance"].get("ubatch_size"))
        add_param_if_set(cmd, "--keep", self.config["performance"].get("num_keep"))
        add_param_if_set(cmd, "--parallel", self.config["performance"].get("parallel_slots"))
        add_param_if_set(cmd, "--split-mode", self.config["performance"].get("split_mode"))
        add_param_if_set(cmd, "--tensor-split", self.config["performance"].get("tensor_split"))
        add_param_if_set(cmd, "--main-gpu", self.config["performance"].get("main_gpu"))
        add_param_if_set(cmd, "--cache-type-k", self.config["performance"].get("cache_type_k"))
        add_param_if_set(cmd, "--cache-type-v", self.config["performance"].get("cache_type_v"))
        
        # Add optional sampling parameters
        add_param_if_set(cmd, "--temp", self.config["sampling"].get("temperature"))
        add_param_if_set(cmd, "--top-p", self.config["sampling"].get("top_p"))
        add_param_if_set(cmd, "--top-k", self.config["sampling"].get("top_k"))
        add_param_if_set(cmd, "--min-p", self.config["sampling"].get("min_p"))
        add_param_if_set(cmd, "--repeat-penalty", self.config["sampling"].get("repeat_penalty"))
        add_param_if_set(cmd, "--repeat-last-n", self.config["sampling"].get("repeat_last_n"))
        add_param_if_set(cmd, "--frequency-penalty", self.config["sampling"].get("frequency_penalty"))
        add_param_if_set(cmd, "--presence-penalty", self.config["sampling"].get("presence_penalty"))
        add_param_if_set(cmd, "--dry-multiplier", self.config["sampling"].get("dry_multiplier"))
        add_param_if_set(cmd, "--dry-base", self.config["sampling"].get("dry_base"))
        add_param_if_set(cmd, "--dry-allowed-length", self.config["sampling"].get("dry_allowed_length"))
        add_param_if_set(cmd, "--dry-penalty-last-n", self.config["sampling"].get("dry_penalty_last_n"))
        
        # Add optional server parameters
        add_param_if_set(cmd, "--timeout", self.config["server"].get("timeout"))
        add_param_if_set(cmd, "--system-prompt-file", self.config["server"].get("system_prompt_file"))
        add_param_if_set(cmd, "--log-format", self.config["server"].get("log_format"))
        
        # Add boolean flags only if they are explicitly set to True
        if self.config["server"].get("embedding"):
            cmd.append("--embeddings")
        if self.config["server"].get("metrics"):
            cmd.append("--metrics")
        if self.config["server"].get("log_disable"):
            cmd.append("--log-disable")
        if self.config["server"].get("slots_endpoint_disable"):
            cmd.append("--slots-endpoint-disable")
        if self.config["performance"].get("memory_f32"):
            cmd.append("--memory-f32")
        if self.config["performance"].get("mlock"):
            cmd.append("--mlock")
        if self.config["performance"].get("no_mmap"):
            cmd.append("--no-mmap")
        if self.config["performance"].get("continuous_batching"):
            cmd.append("--cont-batching")
        
        # Add NUMA parameter if set
        add_param_if_set(cmd, "--numa", self.config["performance"].get("numa"))
        
        # Add context extension parameters if set
        if "context_extension" in self.config:
            add_param_if_set(cmd, "--yarn-ext-factor", self.config["context_extension"].get("yarn_ext_factor"))
            add_param_if_set(cmd, "--yarn-attn-factor", self.config["context_extension"].get("yarn_attn_factor"))
            add_param_if_set(cmd, "--yarn-beta-slow", self.config["context_extension"].get("yarn_beta_slow"))
            add_param_if_set(cmd, "--yarn-beta-fast", self.config["context_extension"].get("yarn_beta_fast"))
            add_param_if_set(cmd, "--grp-attn-n", self.config["context_extension"].get("group_attn_n"))
            add_param_if_set(cmd, "--grp-attn-w", self.config["context_extension"].get("group_attn_w"))
        
        # Only add chat template file if one is selected (not empty)
        if self.config.get("template", {}).get("selected"):
            cmd.extend([
                "--jinja",
                "--chat-template-file", str(Path(self.config["template"]["directory"]) / self.config["template"]["selected"]),
            ])
        
        # Add default flags that are always enabled
        cmd.extend([
            "--verbose",
            "--flash-attn", "auto"
        ])
        
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
        image_name = "llama-nexus-llamacpp-api"
        try:
            docker_client.images.get(image_name)
            logger.info(f"Docker image validation successful for image_name='{image_name}'")
        except docker.errors.ImageNotFound:
            error_msg = f"Docker image not found: {image_name}. Please build the image first."
            logger.error(f"Docker image validation failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        logger.info("Initiating Docker container creation and startup...")
        
        # Resolve host templates dir (can be overridden via env)
        host_templates_dir = os.getenv(
            "TEMPLATES_HOST_DIR",
            str(Path(__file__).resolve().parents[1] / "chat-templates")
        )

        # Determine runtime and device visibility based on execution mode
        execution_mode = self.config.get("execution", {}).get("mode", "gpu")
        cuda_devices = self.config.get("execution", {}).get("cuda_devices", "all")
        
        runtime_config = "nvidia" if execution_mode == "gpu" else None
        env_vars = {
            "MODEL_NAME": self.config['model']['name'],
            "MODEL_VARIANT": self.config['model']['variant'],
        }
        
        if execution_mode == "gpu":
            env_vars["CUDA_VISIBLE_DEVICES"] = cuda_devices
            env_vars["NVIDIA_VISIBLE_DEVICES"] = cuda_devices
        
        # Run container with the command
        self.docker_container = docker_client.containers.run(
            image=image_name,
            name=self.container_name,
            command=cmd,
            detach=True,
            auto_remove=False,
            network=self.docker_network,
            volumes={
                "llamacpp-api_gpt_oss_models": {"bind": "/home/llamacpp/models", "mode": "rw"},
                host_templates_dir: {"bind": "/home/llamacpp/templates", "mode": "ro"}
            },
            environment=env_vars,
            runtime=runtime_config,
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
        image_name = "llama-nexus-llamacpp-api"
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
        host_templates_dir = os.getenv(
            "TEMPLATES_HOST_DIR",
            str(Path(__file__).resolve().parents[1] / "chat-templates")
        )
        
        # Determine runtime and device visibility based on execution mode
        execution_mode = self.config.get("execution", {}).get("mode", "gpu")
        cuda_devices = self.config.get("execution", {}).get("cuda_devices", "all")
        
        docker_cmd = [
            'docker', 'run', '-d',
            '--name', self.container_name,
            '--shm-size', '16g',
            '-p', '8600:8080',
            '--network', self.docker_network,
            '-v', 'llama-nexus_gpt_oss_models:/home/llamacpp/models',
            '-v', f'{host_templates_dir}:/home/llamacpp/templates:ro',
            '-e', f'MODEL_NAME={self.config["model"]["name"]}',
            '-e', f'MODEL_VARIANT={self.config["model"]["variant"]}',
        ]
        
        # Add runtime and CUDA env vars only for GPU mode
        if execution_mode == "gpu":
            docker_cmd.extend(['--runtime', 'nvidia'])
            docker_cmd.extend(['-e', f'CUDA_VISIBLE_DEVICES={cuda_devices}'])
            docker_cmd.extend(['-e', f'NVIDIA_VISIBLE_DEVICES={cuda_devices}'])
        
        docker_cmd.append(image_name)
        docker_cmd.extend(cmd)
        
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
            
            # Validate chat template file exists (only if one is selected)
            if self.config["template"]["selected"]:
                template_path = str(Path(self.config["template"]["directory"]) / self.config["template"]["selected"]) 
                logger.info(f"Validating chat template file at path='{template_path}'")
                if not os.path.exists(template_path):
                    error_msg = f"Chat template file not found: {template_path}"
                    logger.error(f"Chat template validation failed: {error_msg}")
                    raise HTTPException(status_code=400, detail=error_msg)
            else:
                logger.info("No custom chat template selected, will use tokenizer default")
            
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
                    # Get relative path for linking to local files
                    relative_path = str(entry.relative_to(self.models_dir))
                    items.append({
                        "name": name,
                        "variant": variant or "unknown",
                        "size": total_size,
                        "status": "available",
                        "lastModified": datetime.fromtimestamp(latest_mtime).isoformat(),
                        "localPath": relative_path,
                        "filename": entry.name
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
        headers = {"User-Agent": "llama-nexus1.0"}
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
            headers = {"User-Agent": "llama-nexus1.0"}
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
                            "temperature_c": float(values[3]),
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
        # Deep merge configs, filtering out None values
        for category in ["model", "sampling", "performance", "server"]:
            if category in new_config:
                if category not in self.config:
                    self.config[category] = {}
                # Filter out None values before updating
                filtered_category = {k: v for k, v in new_config[category].items() if v is not None}
                self.config[category].update(filtered_category)
        
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
    
    def build_command(self) -> List[str]:
        """Build llama-server embedding command"""
        model_name = self.config['model']['name']
        variant = self.config['model']['variant']
        
        # Map model names to HuggingFace repositories
        model_mappings = {
            "nomic-embed-text-v1.5": ("nomic-ai/nomic-embed-text-v1.5-GGUF", f"nomic-embed-text-v1.5.{variant}.gguf"),
            "e5-mistral-7b": ("intfloat/e5-mistral-7b-instruct-GGUF", f"e5-mistral-7b-instruct.{variant}.gguf"),
            "bge-m3": ("BAAI/bge-m3-GGUF", f"bge-m3.{variant}.gguf"),
            "gte-Qwen2-1.5B": ("Alibaba-NLP/gte-Qwen2-1.5B-instruct-GGUF", f"gte-Qwen2-1.5B-instruct.{variant}.gguf"),
        }
        
        if model_name in model_mappings:
            _, model_file = model_mappings[model_name]
        else:
            model_file = f"{model_name}.{variant}.gguf"
        
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
            "--verbose",
            "--flash-attn",
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
                "NVIDIA_VISIBLE_DEVICES": cuda_devices,
            }
            
            # Create container config
            container_config = {
                "image": "llamacpp-api:latest",
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
                container_config["runtime"] = "nvidia"
            
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
            # Check if container exists
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
            
            # Build environment and command
            execution_mode = self.config.get("execution", {}).get("mode", "gpu")
            cuda_devices = self.config.get("execution", {}).get("cuda_devices", "all")
            
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
                "-e", f"CONTEXT_SIZE={self.config['model']['context_size']}",
                "-e", f"GPU_LAYERS={self.config['model']['gpu_layers']}",
                "-e", f"HOST={self.config['server']['host']}",
                "-e", f"PORT={self.config['server']['port']}",
                "-e", f"API_KEY={self.config['server']['api_key']}",
                "-e", f"CUDA_VISIBLE_DEVICES={cuda_devices}",
                "-e", f"NVIDIA_VISIBLE_DEVICES={cuda_devices}",
            ]
            
            # Add GPU runtime only for GPU mode
            if execution_mode == "gpu":
                docker_cmd.insert(2, "--runtime=nvidia")
            
            docker_cmd.extend([
                "llamacpp-api:latest",
                "bash", "-c",
                " ".join(self.build_command())
            ])
            
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


# Initialize managers
manager = LlamaCPPManager()
embedding_manager = EmbeddingManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting LlamaCPP Management API")

    # Load saved config if exists and merge with defaults
    config_file = Path("/tmp/llamacpp_config.json")
    if config_file.exists():
        with open(config_file) as f:
            saved_config = json.load(f)
            # Merge saved config with default config to ensure all required fields exist
            default_config = manager.load_default_config()
            for category in default_config:
                if category in saved_config:
                    # Merge category, keeping saved values but adding missing defaults
                    for key, default_value in default_config[category].items():
                        if key not in saved_config[category] or saved_config[category][key] is None:
                            saved_config[category][key] = default_value
                else:
                    # Add missing category entirely
                    saved_config[category] = default_config[category]
            manager.config = saved_config

    # Initialize RAG system
    if RAG_AVAILABLE:
        try:
            rag_db_path = os.getenv("RAG_DB_PATH", "data/rag")
            os.makedirs(rag_db_path, exist_ok=True)
            
            # Initialize document manager
            app.state.document_manager = DocumentManager(f"{rag_db_path}/documents.db")
            await app.state.document_manager.initialize()
            
            # Initialize GraphRAG
            app.state.graph_rag = GraphRAG(f"{rag_db_path}/graph.db")
            await app.state.graph_rag.initialize()
            
            # Initialize Qdrant vector store
            qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            app.state.vector_store = QdrantStore(host=qdrant_host, port=qdrant_port)
            if await app.state.vector_store.connect():
                logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
            else:
                logger.warning("Failed to connect to Qdrant")
                app.state.vector_store = None
            
            # Initialize document discovery
            app.state.document_discovery = DocumentDiscovery(f"{rag_db_path}/discovery.db")
            await app.state.document_discovery.initialize()
            
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            app.state.document_manager = None
            app.state.graph_rag = None
            app.state.vector_store = None
            app.state.document_discovery = None

    yield

    # Shutdown
    logger.info("Shutting down LlamaCPP Management API")
    if not manager.use_docker and manager.process and manager.process.poll() is None:
        await manager.stop()
    
    # Cleanup RAG
    if RAG_AVAILABLE and hasattr(app.state, 'vector_store') and app.state.vector_store:
        await app.state.vector_store.disconnect()

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

# Add token usage tracking middleware if available
try:
    app.add_middleware(TokenUsageMiddleware)
except Exception:
    logger.warning("TokenUsageMiddleware not available; token tracking middleware not enabled")

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

# Embedding Service Endpoints
@app.post("/api/v1/embedding/start")
async def start_embedding_service():
    """Start the embedding service"""
    try:
        logger.log_operation_start("Embedding service start request", endpoint="/api/v1/embedding/start")
        success = await embedding_manager.start()
        status = embedding_manager.get_status()
        logger.log_operation_success("Embedding service start request", success=success)
        return {
            "success": success,
            "status": status,
            "message": "Embedding service started successfully" if success else "Embedding service failed to start"
        }
    except Exception as e:
        logger.log_operation_failure("Embedding service start request", f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start embedding service: {str(e)}")

@app.post("/api/v1/embedding/stop")
async def stop_embedding_service():
    """Stop the embedding service"""
    try:
        success = await embedding_manager.stop()
        return {"success": success, "status": embedding_manager.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop embedding service: {str(e)}")

@app.post("/api/v1/embedding/restart")
async def restart_embedding_service():
    """Restart the embedding service"""
    try:
        success = await embedding_manager.restart()
        return {"success": success, "status": embedding_manager.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart embedding service: {str(e)}")

@app.get("/api/v1/embedding/status")
async def get_embedding_status():
    """Get current embedding service status"""
    return embedding_manager.get_status()

@app.get("/api/v1/embedding/config")
async def get_embedding_config():
    """Get current embedding configuration"""
    return {
        "config": embedding_manager.config,
        "command": " ".join(embedding_manager.build_command()),
        "available_models": [
            {
                "name": "nomic-embed-text-v1.5",
                "dimensions": 768,
                "max_tokens": 8192,
                "description": "Nomic AI's long context embedding model (recommended)"
            },
            {
                "name": "e5-mistral-7b",
                "dimensions": 4096,
                "max_tokens": 32768,
                "description": "E5 Mistral 7B instruct model for embeddings"
            },
            {
                "name": "bge-m3",
                "dimensions": 1024,
                "max_tokens": 8192,
                "description": "BAAI BGE-M3 multilingual embedding model"
            },
            {
                "name": "gte-Qwen2-1.5B",
                "dimensions": 1536,
                "max_tokens": 32768,
                "description": "Alibaba GTE Qwen2 1.5B instruct model"
            }
        ]
    }

@app.put("/api/v1/embedding/config")
async def update_embedding_config(config: Dict[str, Any]):
    """Update embedding configuration (requires restart to apply)"""
    try:
        updated_config = embedding_manager.update_config(config)
        return {
            "success": True,
            "config": updated_config,
            "message": "Configuration updated. Restart service to apply changes."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")

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
            "server": ["api_key"],
            "template": ["directory", "selected"]
        }
    }

def _merge_and_persist_config(new_config: Dict[str, Any]) -> Dict[str, Any]:
    # Deep merge into manager.config and persist to disk
    import copy
    base = copy.deepcopy(manager.config)  # Create a deep copy to avoid modifying original
    
    # The new_config comes wrapped in a 'config' key, so extract it
    config_data = new_config.get('config', new_config)
    
    # Required fields that should never be removed
    required_fields = {
        "model": ["name", "variant"],
        "server": ["host", "port", "api_key"]
    }
    
    for category in ["model", "sampling", "performance", "context_extension", "server", "template"]:
        if category in config_data:
            if category not in base:
                base[category] = {}
            # Update with new values, removing keys that are None or undefined
            for key, value in config_data[category].items():
                if value is None or value == "":
                    # Only remove the key if it's not a required field
                    if category not in required_fields or key not in required_fields[category]:
                        base[category].pop(key, None)
                else:
                    base[category][key] = value
    
    # Update the manager's config with the modified base
    manager.config = base
    
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

@app.post("/api/v1/service/config/preview")
async def preview_config(config: Dict[str, Any]):
    """Preview command line for a configuration without saving it"""
    try:
        # Create a minimal base config with only required fields
        import copy
        temp_config = {
            "model": {
                "name": manager.config["model"]["name"],
                "variant": manager.config["model"]["variant"]
            },
            "server": {
                "host": manager.config["server"]["host"],
                "port": manager.config["server"]["port"],
                "api_key": manager.config["server"]["api_key"]
            },
            "sampling": {},
            "performance": {},
            "context_extension": {},
            "template": manager.config.get("template", {})
        }
        
        # Apply the preview config changes - only add non-null values
        config_data = config.get('config', config)
        for category in ["model", "sampling", "performance", "context_extension", "server", "template"]:
            if category in config_data:
                if category not in temp_config:
                    temp_config[category] = {}
                # Only add values that are not None or empty string
                for key, value in config_data[category].items():
                    if value is not None and value != "":
                        temp_config[category][key] = value
        
        # Temporarily replace manager's config to build the command
        original_config = manager.config
        manager.config = temp_config
        
        try:
            command = " ".join(manager.build_command())
        finally:
            # Restore original config
            manager.config = original_config
        
        return {
            "command": command
        }
    except Exception as e:
        logger.error(f"Preview config error: {str(e)}")
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

@app.get("/api/v1/logs/container")
async def get_container_logs(lines: int = 100):
    """Get recent logs from the Docker container"""
    try:
        result = subprocess.run(
            ['docker', 'logs', '--tail', str(lines), manager.container_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        logs = (result.stdout + result.stderr).strip().split('\n')
        return {
            "success": True,
            "logs": [{"timestamp": "", "level": "INFO", "message": line} for line in logs if line],
            "container": manager.container_name
        }
    except Exception as e:
        logger.error(f"Failed to get container logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/logs/container/stream")
async def stream_container_logs():
    """Stream logs from the Docker container using Server-Sent Events"""
    from fastapi.responses import StreamingResponse
    
    async def log_generator():
        try:
            process = await asyncio.create_subprocess_exec(
                'docker', 'logs', '-f', '--tail', '50', manager.container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                
                log_line = line.decode('utf-8').strip()
                if log_line:
                    # Format as SSE
                    yield f"data: {json.dumps({'message': log_line, 'timestamp': datetime.now().isoformat()})}\n\n"
                
                await asyncio.sleep(0.01)  # Small delay to prevent overwhelming the client
                
        except Exception as e:
            logger.error(f"Error streaming logs: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

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


# General WebSocket endpoint for real-time updates
_ws_clients: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    General WebSocket endpoint for real-time updates.
    Broadcasts metrics, status changes, download progress, and model events.
    """
    await websocket.accept()
    _ws_clients.append(websocket)
    
    try:
        # Send connection confirmation with current status
        await websocket.send_json({
            "type": "connected",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "service_running": manager.running,
                "model_name": manager.current_model_name if hasattr(manager, 'current_model_name') else None
            }
        })
        
        # Keep connection alive and periodically send metrics
        while True:
            await asyncio.sleep(5)  # Send updates every 5 seconds
            
            try:
                # Gather current metrics
                metrics_data = {
                    "cpu": {
                        "percent": psutil.cpu_percent(),
                    },
                    "memory": {
                        "total_mb": psutil.virtual_memory().total / (1024 * 1024),
                        "used_mb": psutil.virtual_memory().used / (1024 * 1024),
                        "percent": psutil.virtual_memory().percent
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                # Try to get GPU info
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0:
                        parts = result.stdout.strip().split(",")
                        if len(parts) >= 3:
                            metrics_data["gpu"] = {
                                "vram_used_mb": float(parts[0].strip()),
                                "vram_total_mb": float(parts[1].strip()),
                                "usage_percent": float(parts[2].strip())
                            }
                except:
                    pass  # GPU info not available
                
                await websocket.send_json({
                    "type": "metrics",
                    "timestamp": datetime.now().isoformat(),
                    "payload": metrics_data
                })
                
                # Also send status update
                await websocket.send_json({
                    "type": "status",
                    "timestamp": datetime.now().isoformat(),
                    "payload": {
                        "running": manager.running,
                        "model_name": manager.current_model_name if hasattr(manager, 'current_model_name') else None,
                        "uptime": manager.uptime if hasattr(manager, 'uptime') else 0
                    }
                })
                
            except Exception as e:
                logger.warning(f"Error sending metrics via WebSocket: {e}")
                
    except WebSocketDisconnect:
        _ws_clients.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


async def broadcast_ws_event(event_type: str, data: dict):
    """Broadcast an event to all connected WebSocket clients"""
    message = {
        "type": event_type,
        "timestamp": datetime.now().isoformat(),
        "payload": data
    }
    
    disconnected = []
    for client in _ws_clients:
        try:
            await client.send_json(message)
        except:
            disconnected.append(client)
    
    for client in disconnected:
        _ws_clients.remove(client)


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

@app.get("/api/v1/system/gpus")
async def list_gpus():
    """List all available GPUs with detailed information"""
    try:
        # Check if nvidia-smi is available
        which_result = subprocess.run(
            ["which", "nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if which_result.returncode != 0:
            return {
                "available": False,
                "gpus": [],
                "reason": "nvidia-smi not found"
            }
        
        # Get detailed GPU information
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return {
                "available": False,
                "gpus": [],
                "reason": f"nvidia-smi failed with code {result.returncode}"
            }
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            values = [v.strip() for v in line.split(',')]
            if len(values) >= 7:
                gpus.append({
                    "index": int(values[0]),
                    "name": values[1],
                    "vram_total_mb": int(values[2]),
                    "vram_used_mb": int(values[3]),
                    "vram_free_mb": int(values[4]),
                    "utilization_percent": float(values[5]) if values[5] != "N/A" else 0,
                    "temperature_c": float(values[6]) if values[6] != "N/A" else 0,
                    "power_draw_watts": float(values[7]) if len(values) > 7 and values[7] != "N/A" else None,
                    "power_limit_watts": float(values[8]) if len(values) > 8 and values[8] != "N/A" else None,
                })
        
        return {
            "available": True,
            "gpus": gpus,
            "count": len(gpus)
        }
        
    except Exception as e:
        logger.error(f"Error listing GPUs: {e}")
        return {
            "available": False,
            "gpus": [],
            "reason": str(e)
        }

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

# LlamaCPP Commit Management Endpoints
@app.get("/api/v1/llamacpp/commits")
async def get_llamacpp_commits():
    """Get available llama.cpp commits/releases"""
    try:
        async with httpx.AsyncClient() as client:
            # Get latest releases
            releases_response = await client.get(
                "https://api.github.com/repos/ggml-org/llama.cpp/releases",
                params={"per_page": 20}
            )
            releases_response.raise_for_status()
            releases = releases_response.json()
            
            # Get current commit from Dockerfile
            current_commit = get_current_llamacpp_commit()
            
            # Format releases
            formatted_releases = []
            for release in releases:
                formatted_releases.append({
                    "tag": release["tag_name"],
                    "name": release["name"] or release["tag_name"],
                    "published_at": release["published_at"],
                    "body": release["body"][:200] + "..." if len(release["body"]) > 200 else release["body"],
                    "is_current": release["tag_name"] == current_commit
                })
            
            # Also get some recent commits from master branch
            commits_response = await client.get(
                "https://api.github.com/repos/ggml-org/llama.cpp/commits",
                params={"per_page": 10}
            )
            commits_response.raise_for_status()
            commits = commits_response.json()
            
            formatted_commits = []
            for commit in commits:
                formatted_commits.append({
                    "tag": commit["sha"][:8],
                    "name": f"{commit['sha'][:8]} - {commit['commit']['message'].split(chr(10))[0][:50]}",
                    "published_at": commit["commit"]["committer"]["date"],
                    "body": commit["commit"]["message"],
                    "is_current": commit["sha"][:8] == current_commit or commit["sha"] == current_commit
                })
            
            return {
                "current_commit": current_commit,
                "releases": formatted_releases,
                "recent_commits": formatted_commits
            }
    except Exception as e:
        logger.error(f"Failed to fetch llama.cpp commits: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch commits: {str(e)}")

@app.get("/api/v1/llamacpp/commits/{commit_id}/validate")
async def validate_llamacpp_commit(commit_id: str):
    """Validate that a commit exists in the llama.cpp repository"""
    try:
        async with httpx.AsyncClient() as client:
            # Try to get commit info from GitHub API
            response = await client.get(
                f"https://api.github.com/repos/ggml-org/llama.cpp/commits/{commit_id}",
                timeout=10.0
            )
            
            if response.status_code == 404:
                return {
                    "valid": False,
                    "error": "Commit not found in repository"
                }
            elif response.status_code != 200:
                return {
                    "valid": False,
                    "error": f"GitHub API error: {response.status_code}"
                }
            
            commit_data = response.json()
            
            return {
                "valid": True,
                "commit": {
                    "sha": commit_data["sha"],
                    "short_sha": commit_data["sha"][:8],
                    "message": commit_data["commit"]["message"].split('\n')[0][:100],
                    "author": commit_data["commit"]["author"]["name"],
                    "date": commit_data["commit"]["author"]["date"],
                    "url": commit_data["html_url"]
                }
            }
    except httpx.TimeoutException:
        return {
            "valid": False,
            "error": "Timeout while validating commit"
        }
    except Exception as e:
        logger.error(f"Failed to validate commit {commit_id}: {e}")
        return {
            "valid": False,
            "error": f"Validation error: {str(e)}"
        }

@app.post("/api/v1/llamacpp/commits/{commit_id}/apply")
async def apply_llamacpp_commit(commit_id: str):
    """Update Dockerfile to use a specific llama.cpp commit"""
    try:
        # First validate the commit exists
        validation = await validate_llamacpp_commit(commit_id)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=f"Invalid commit: {validation['error']}")
        
        dockerfile_path = Path("/home/alec/git/llama-nexus/Dockerfile")
        
        if not dockerfile_path.exists():
            raise HTTPException(status_code=404, detail="Dockerfile not found")
        
        # Read current Dockerfile
        content = dockerfile_path.read_text()
        
        # Update the git checkout line
        updated_content = re.sub(
            r'git checkout [^\s&\\]+',
            f'git checkout {commit_id}',
            content
        )
        
        if updated_content == content:
            raise HTTPException(status_code=400, detail="Could not find git checkout line in Dockerfile")
        
        # Write updated Dockerfile
        dockerfile_path.write_text(updated_content)
        
        logger.info(f"Updated Dockerfile to use llama.cpp commit: {commit_id}")
        
        return {
            "message": f"Dockerfile updated to use commit {commit_id}",
            "commit": commit_id,
            "commit_info": validation.get("commit"),
            "requires_rebuild": True
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update Dockerfile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update Dockerfile: {str(e)}")

@app.post("/api/v1/llamacpp/rebuild")
async def rebuild_llamacpp():
    """Rebuild the llama.cpp containers with the current Dockerfile"""
    try:
        # Use docker compose to rebuild - specify the compose file explicitly
        compose_file = "/home/alec/git/llama-nexus/docker-compose.yml"
        
        # Check if compose file exists
        if not Path(compose_file).exists():
            raise HTTPException(status_code=500, detail=f"Docker compose file not found: {compose_file}")
        
        result = subprocess.run(
            ["docker", "compose", "-f", compose_file, "up", "-d", "--build"],
            cwd="/home/alec/git/llama-nexus",
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Docker rebuild failed: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Rebuild failed: {result.stderr}")
        
        logger.info("LlamaCPP containers rebuilt successfully")
        
        return {
            "message": "Containers rebuilt successfully",
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Rebuild timed out after 30 minutes")
    except Exception as e:
        logger.error(f"Failed to rebuild containers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild: {str(e)}")

def get_current_llamacpp_commit() -> str:
    """Get the current llama.cpp commit from Dockerfile"""
    try:
        dockerfile_path = Path("/home/alec/git/llama-nexus/Dockerfile")
        if not dockerfile_path.exists():
            return "unknown"
        
        content = dockerfile_path.read_text()
        match = re.search(r'git checkout ([^\s&\\]+)', content)
        if match:
            return match.group(1)
        return "unknown"
    except Exception:
        return "unknown"

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


@app.get("/v1/models/local-files")
async def list_local_model_files():
    """List all downloaded model files in the models directory"""
    try:
        models_dir = Path("/home/llamacpp/models")
        if not models_dir.exists():
            return {"success": True, "data": {"files": [], "total_size": 0}, "timestamp": datetime.now().isoformat()}
        
        files = []
        total_size = 0
        
        # Walk through the directory
        for item in models_dir.rglob('*'):
            if item.is_file():
                # Filter for model files
                if item.suffix.lower() in ['.gguf', '.safetensors', '.bin', '.pth', '.pt']:
                    stat = item.stat()
                    files.append({
                        "name": item.name,
                        "path": str(item.relative_to(models_dir)),
                        "full_path": str(item),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "extension": item.suffix
                    })
                    total_size += stat.st_size
        
        # Sort by modified date (newest first)
        files.sort(key=lambda x: x['modified'], reverse=True)
        
        logger.info(f"Found {len(files)} local model files, total size: {total_size / (1024**3):.2f} GB")
        return {
            "success": True, 
            "data": {
                "files": files,
                "total_size": total_size,
                "total_count": len(files)
            }, 
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing local model files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/v1/models/local-files")
async def delete_local_model_file(file_path: str):
    """Delete a model file from the local filesystem"""
    try:
        models_dir = Path("/home/llamacpp/models")
        # Resolve the full path and ensure it's within models_dir
        full_path = (models_dir / file_path).resolve()
        
        # Security check: ensure the path is within models directory
        if not str(full_path).startswith(str(models_dir.resolve())):
            raise HTTPException(status_code=403, detail="Access denied: path outside models directory")
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if not full_path.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")
        
        # Get file info before deletion
        file_size = full_path.stat().st_size
        file_name = full_path.name
        
        # Delete the file
        full_path.unlink()
        
        logger.info(f"Deleted model file: {file_name} ({file_size / (1024**3):.2f} GB)")
        return {
            "success": True,
            "data": {
                "deleted_file": file_name,
                "size_freed": file_size
            },
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models/repo-files")
async def list_repo_files(repo_id: str, revision: str = "main"):
    """List available model files in a HuggingFace model repository.

    Uses the Hugging Face Hub API to enumerate files from the specified repo
    and returns filenames ending with supported extensions (.gguf, .safetensors, .bin, .pth, .pt).
    This helps the frontend present valid choices and avoid 404s from incorrect filenames.
    """
    try:
        # Validate repo_id format
        if not repo_id or '/' not in repo_id:
            raise HTTPException(status_code=422, detail="Invalid repository ID. Must be in format 'owner/repo'")
            
        api = HfApi(token=os.getenv("HUGGINGFACE_TOKEN"))
        try:
            logger.info(f"Listing repo files for {repo_id} with token: {'*****' if os.getenv('HUGGINGFACE_TOKEN') else 'None'}")
            files = api.list_repo_files(repo_id=repo_id, revision=revision, repo_type="model")
            # Support multiple model formats, not just GGUF
            model_extensions = ('.gguf', '.safetensors', '.bin', '.pth', '.pt')
            model_files = [f for f in files if f.lower().endswith(model_extensions)]
            logger.info(f"Found {len(model_files)} model files in repo {repo_id}")
            return {"success": True, "data": {"files": model_files}, "timestamp": datetime.now().isoformat()}
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
    # Support multiple model formats
    supported_extensions = ('.gguf', '.safetensors', '.bin', '.pth', '.pt')
    if not str(filename).lower().endswith(supported_extensions):
        raise HTTPException(status_code=400, detail=f"Only files with extensions {supported_extensions} are supported")
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


@app.post("/v1/models/estimate-vram")
async def estimate_vram(payload: Dict[str, Any]):
    """
    Estimate VRAM requirements for a model configuration.
    
    Takes model parameters, quantization level, context size, and batch size
    to calculate approximate VRAM requirements.
    """
    try:
        # Extract parameters with defaults
        params_b = payload.get("parameters_b", 7.0)  # Model parameters in billions
        quant_bits = payload.get("quant_bits", 4.0)  # Bits per weight (e.g., 4 for Q4_K_M)
        context_size = payload.get("context_size", 4096)
        batch_size = payload.get("batch_size", 1)
        num_layers = payload.get("num_layers", 32)  # Number of transformer layers
        head_dim = payload.get("head_dim", 128)  # Head dimension
        num_kv_heads = payload.get("num_kv_heads", 8)  # Number of KV heads (for GQA)
        available_vram_gb = payload.get("available_vram_gb", 24.0)
        
        # Quantization bits mapping for common formats
        quant_map = {
            "F32": 32.0, "F16": 16.0, "BF16": 16.0,
            "Q8_0": 8.5, "Q6_K": 6.5, "Q5_K_M": 5.5, "Q5_K_S": 5.5,
            "Q4_K_M": 4.5, "Q4_K_S": 4.5, "Q4_0": 4.0,
            "Q3_K_M": 3.5, "Q3_K_S": 3.5, "Q2_K": 2.5,
            "IQ4_XS": 4.25, "IQ3_XXS": 3.0, "IQ2_XXS": 2.0,
            "MXFP4": 4.5  # Native MXFP4 format
        }
        
        # If quant_bits is a string (quantization name), look it up
        if isinstance(quant_bits, str):
            quant_bits = quant_map.get(quant_bits.upper(), 4.5)
        
        # Calculate model weights VRAM (in GB)
        # Formula: params_b * (bits/8) * overhead_factor
        overhead_factor = 1.1  # 10% overhead for model loading
        model_vram_gb = params_b * (quant_bits / 8) * overhead_factor
        
        # Calculate KV cache VRAM (in GB)
        # Formula: 2 * num_layers * context_size * batch_size * num_kv_heads * head_dim * 2 (bytes) / 1e9
        # The "2" at the start is for K and V caches
        kv_cache_gb = (2 * num_layers * context_size * batch_size * num_kv_heads * head_dim * 2) / 1e9
        
        # Compute buffer (temporary activations) - rough estimate
        # Typically 0.5-1GB for most models
        compute_buffer_gb = 0.5 + (params_b / 20)  # Scale with model size
        
        # Additional overhead (CUDA context, etc.)
        cuda_overhead_gb = 0.5
        
        # Total VRAM
        total_vram_gb = model_vram_gb + kv_cache_gb + compute_buffer_gb + cuda_overhead_gb
        
        # Determine if model fits
        fits_in_vram = total_vram_gb <= available_vram_gb
        
        # Generate recommendation
        if fits_in_vram:
            headroom = available_vram_gb - total_vram_gb
            if headroom > 4:
                recommendation = f"Model fits comfortably with {headroom:.1f}GB headroom. You could increase context size or use a higher quality quantization."
            elif headroom > 1:
                recommendation = f"Model fits with {headroom:.1f}GB headroom. Should work well for typical workloads."
            else:
                recommendation = f"Model fits with minimal headroom ({headroom:.1f}GB). Consider reducing context size for stability."
        else:
            overage = total_vram_gb - available_vram_gb
            suggestion_bits = quant_bits * (available_vram_gb / total_vram_gb) * 0.9
            suggestions = [q for q, b in quant_map.items() if b <= suggestion_bits]
            suggestion = suggestions[0] if suggestions else "Q2_K"
            recommendation = f"Model exceeds available VRAM by {overage:.1f}GB. Consider using {suggestion} quantization or reducing context size to {int(context_size * 0.5)}."
        
        return {
            "success": True,
            "data": {
                "model_vram_gb": round(model_vram_gb, 2),
                "kv_cache_gb": round(kv_cache_gb, 2),
                "compute_buffer_gb": round(compute_buffer_gb, 2),
                "overhead_gb": round(cuda_overhead_gb, 2),
                "total_vram_gb": round(total_vram_gb, 2),
                "available_vram_gb": available_vram_gb,
                "fits_in_vram": fits_in_vram,
                "utilization_percent": round((total_vram_gb / available_vram_gb) * 100, 1),
                "breakdown": {
                    "weights": round(model_vram_gb, 2),
                    "kv_cache": round(kv_cache_gb, 2),
                    "compute_buffer": round(compute_buffer_gb, 2),
                    "overhead": round(cuda_overhead_gb, 2)
                },
                "recommendation": recommendation,
                "input_parameters": {
                    "parameters_b": params_b,
                    "quant_bits": quant_bits,
                    "context_size": context_size,
                    "batch_size": batch_size,
                    "num_layers": num_layers,
                    "num_kv_heads": num_kv_heads
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error estimating VRAM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Chat template management endpoints ---
@app.get("/v1/templates")
async def list_templates():
    try:
        templates_dir = Path(manager.config["template"]["directory"])
        selected = manager.config["template"].get("selected", "")
        if not templates_dir.exists():
            return {"success": True, "data": {"directory": str(templates_dir), "files": [], "selected": selected}}
        files = [p.name for p in templates_dir.glob("*.jinja") if p.is_file()]
        return {"success": True, "data": {"directory": str(templates_dir), "files": files, "selected": selected}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/templates/{filename}")
async def get_template(filename: str):
    try:
        templates_dir = Path(manager.config["template"]["directory"])
        path = templates_dir / filename
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="Template not found")
        return {"success": True, "data": {"filename": filename, "content": path.read_text()}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/v1/templates/{filename}")
async def update_template(filename: str, payload: Dict[str, Any]):
    try:
        content = payload.get("content")
        if not isinstance(content, str):
            raise HTTPException(status_code=400, detail="'content' must be a string")
        templates_dir = Path(manager.config["template"]["directory"]) 
        templates_dir.mkdir(parents=True, exist_ok=True)
        path = templates_dir / filename
        # Overwrite file atomically
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(content)
        tmp_path.replace(path)
        return {"success": True, "data": {"filename": filename}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/templates/select")
async def select_template(payload: Dict[str, Any]):
    try:
        filename = payload.get("filename")
        if filename is None or not isinstance(filename, str):
            raise HTTPException(status_code=400, detail="'filename' is required")
        
        templates_dir = Path(manager.config["template"]["directory"]) 
        
        # Allow empty filename for tokenizer default
        if filename == "":
            # Update config to use no template (tokenizer default)
            updated = _merge_and_persist_config({"template": {"directory": str(templates_dir), "selected": ""}})
            return {"success": True, "data": {"selected": updated["template"].get("selected", "")}}
        
        # For non-empty filenames, validate the file exists
        path = templates_dir / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="Template not found")
        
        # Update config and persist
        updated = _merge_and_persist_config({"template": {"directory": str(templates_dir), "selected": filename}})
        return {"success": True, "data": {"selected": updated["template"].get("selected", filename)}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/templates")
async def create_template(payload: Dict[str, Any]):
    try:
        filename = payload.get("filename")
        content = payload.get("content", "")
        
        if not filename or not isinstance(filename, str):
            raise HTTPException(status_code=400, detail="'filename' is required")
        
        if not isinstance(content, str):
            raise HTTPException(status_code=400, detail="'content' must be a string")
        
        # Ensure filename has .jinja extension
        if not filename.endswith('.jinja'):
            filename += '.jinja'
        
        templates_dir = Path(manager.config["template"]["directory"])
        templates_dir.mkdir(parents=True, exist_ok=True)
        path = templates_dir / filename
        
        # Check if file already exists
        if path.exists():
            raise HTTPException(status_code=409, detail="Template already exists")
        
        # Create the new template file
        path.write_text(content)
        return {"success": True, "data": {"filename": filename}}
    except HTTPException:
        raise
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

# --- Token Usage Tracking Endpoints ---

class TimeRange(str, Enum):
    ONE_HOUR = "1h"
    ONE_DAY = "24h"
    ONE_WEEK = "7d"
    ONE_MONTH = "30d"
    ALL_TIME = "all"

@app.get("/v1/usage/tokens")
async def get_token_usage(timeRange: TimeRange = TimeRange.ONE_DAY):
    try:
        if not token_tracker:
            raise HTTPException(status_code=503, detail="Token tracker not available")
        usage_data = token_tracker.get_token_usage(timeRange)  # type: ignore
        return {"success": True, "data": usage_data, "timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting token usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/usage/tokens/timeline")
async def get_token_usage_timeline(timeRange: TimeRange = TimeRange.ONE_DAY):
    try:
        if not token_tracker:
            raise HTTPException(status_code=503, detail="Token tracker not available")
        interval = "hour"
        if timeRange == TimeRange.ONE_HOUR:
            interval = "minute"
        elif timeRange in [TimeRange.ONE_WEEK, TimeRange.ONE_MONTH]:
            interval = "day"
        usage_data = token_tracker.get_token_usage_over_time(timeRange, interval)  # type: ignore
        return {"success": True, "data": usage_data, "interval": interval, "timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting token usage timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/usage/tokens/summary")
async def get_token_usage_summary(timeRange: TimeRange = TimeRange.ALL_TIME):
    try:
        if not token_tracker:
            raise HTTPException(status_code=503, detail="Token tracker not available")
        summary = token_tracker.get_total_token_usage(timeRange)  # type: ignore
        return {"success": True, "data": summary, "timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting token usage summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/usage/tokens/record")
async def record_token_usage(request: Request):
    try:
        if not token_tracker:
            raise HTTPException(status_code=503, detail="Token tracker not available")
        body = await request.json()
        if not body.get("model_id"):
            raise HTTPException(status_code=400, detail="model_id is required")
        if "prompt_tokens" not in body:
            raise HTTPException(status_code=400, detail="prompt_tokens is required")
        if "completion_tokens" not in body:
            raise HTTPException(status_code=400, detail="completion_tokens is required")
        success = token_tracker.record_token_usage(  # type: ignore
            model_id=body.get("model_id"),
            prompt_tokens=body.get("prompt_tokens"),
            completion_tokens=body.get("completion_tokens"),
            model_name=body.get("model_name"),
            request_id=body.get("request_id"),
            user_id=body.get("user_id"),
            endpoint=body.get("endpoint"),
            metadata=body.get("metadata"),
        )
        if success:
            return {"success": True, "message": "Token usage recorded successfully", "timestamp": datetime.now().isoformat()}
        else:
            raise HTTPException(status_code=500, detail="Failed to record token usage")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording token usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Benchmark Testing Endpoints ---

# In-memory storage for benchmarks (in production, this would be a database)
benchmarks_storage: Dict[str, Dict[str, Any]] = {}

class BenchmarkStatus(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

def load_benchmark_tests():
    """Load benchmark tests from JSON file"""
    try:
        # Try to load from frontend public directory first (for development)
        test_file_path = "/app/public/tool_call_tests.json"
        if not os.path.exists(test_file_path):
            # Fallback to local file for development
            test_file_path = "tool_call_tests.json"
        
        with open(test_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading benchmark tests: {e}")
        return {
            "tool_calling": [],
            "basic": [],
            "advanced": []
        }

def create_benchmark_result(benchmark_id: str, test_category: str, max_examples: int):
    """Create a new benchmark result entry"""
    benchmarks_storage[benchmark_id] = {
        "benchmark_id": benchmark_id,
        "status": BenchmarkStatus.STARTING,
        "test_category": test_category,
        "max_examples": max_examples,
        "started_at": datetime.now().isoformat(),
        "progress": 0,
        "total": 0,
        "current_test": None,
        "results": {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "errors": []
        }
    }
    return benchmarks_storage[benchmark_id]

def update_benchmark_progress(benchmark_id: str, status: BenchmarkStatus, **kwargs):
    """Update benchmark progress"""
    if benchmark_id in benchmarks_storage:
        benchmarks_storage[benchmark_id]["status"] = status
        for key, value in kwargs.items():
            benchmarks_storage[benchmark_id][key] = value
        
        # Update accuracy if results changed
        if "results" in kwargs and "total" in kwargs["results"] and kwargs["results"]["total"] > 0:
            accuracy = (kwargs["results"]["correct"] / kwargs["results"]["total"]) * 100
            benchmarks_storage[benchmark_id]["results"]["accuracy"] = round(accuracy, 2)

async def run_single_test(test: Dict[str, Any], test_category: str) -> bool:
    """Run a single benchmark test and return success/failure"""
    try:
        if test_category == "tool_calling":
            return await run_tool_calling_test(test)
        elif test_category == "basic":
            return await run_basic_test(test)
        elif test_category == "advanced":
            return await run_advanced_test(test)
        else:
            logger.warning(f"Unknown test category: {test_category}")
            return False
    except Exception as e:
        logger.error(f"Error running test {test.get('name', 'Unknown')}: {e}")
        return False

async def run_tool_calling_test(test: Dict[str, Any]) -> bool:
    """Run a tool calling test against the LLM"""
    try:
        # Check if LLM service is running
        status = manager.get_status()
        if not status.get("running", False):
            logger.error("LLM service is not running")
            return False
        
        # Prepare the chat completion request
        messages = [
            {"role": "user", "content": test["prompt"]}
        ]
        
        tools = test.get("tools", [])
        
        # Make request to the LLM
        async with httpx.AsyncClient(timeout=30.0) as client:
            llamacpp_url = f"http://localhost:{manager.config['server']['port']}"
            
            request_data = {
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0.1,  # Low temperature for consistent results
                "max_tokens": 500
            }
            
            response = await client.post(
                f"{llamacpp_url}/v1/chat/completions",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"LLM request failed: {response.status_code} - {response.text}")
                return False
            
            result = response.json()
            
            # Check if the LLM made the expected tool call
            return validate_tool_call_response(result, test)
            
    except Exception as e:
        logger.error(f"Tool calling test failed: {e}")
        return False

async def run_basic_test(test: Dict[str, Any]) -> bool:
    """Run a basic knowledge test"""
    try:
        # Check if LLM service is running
        status = manager.get_status()
        if not status.get("running", False):
            logger.error("LLM service is not running")
            return False
        
        messages = [
            {"role": "user", "content": test["prompt"]}
        ]
        
        # Make request to the LLM
        async with httpx.AsyncClient(timeout=30.0) as client:
            llamacpp_url = f"http://localhost:{manager.config['server']['port']}"
            
            request_data = {
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 200
            }
            
            response = await client.post(
                f"{llamacpp_url}/v1/chat/completions",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"LLM request failed: {response.status_code} - {response.text}")
                return False
            
            result = response.json()
            
            # Check if response contains expected content
            return validate_basic_response(result, test)
            
    except Exception as e:
        logger.error(f"Basic test failed: {e}")
        return False

async def run_advanced_test(test: Dict[str, Any]) -> bool:
    """Run an advanced reasoning test"""
    try:
        # Check if LLM service is running
        status = manager.get_status()
        if not status.get("running", False):
            logger.error("LLM service is not running")
            return False
        
        messages = [
            {"role": "user", "content": test["prompt"]}
        ]
        
        # Make request to the LLM
        async with httpx.AsyncClient(timeout=30.0) as client:
            llamacpp_url = f"http://localhost:{manager.config['server']['port']}"
            
            request_data = {
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 400
            }
            
            response = await client.post(
                f"{llamacpp_url}/v1/chat/completions",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"LLM request failed: {response.status_code} - {response.text}")
                return False
            
            result = response.json()
            
            # Check if response contains expected reasoning
            return validate_advanced_response(result, test)
            
    except Exception as e:
        logger.error(f"Advanced test failed: {e}")
        return False

def validate_tool_call_response(response: Dict[str, Any], test: Dict[str, Any]) -> bool:
    """Validate that the LLM made the correct tool call"""
    try:
        choices = response.get("choices", [])
        if not choices:
            return False
        
        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])
        
        if not tool_calls:
            logger.info(f"No tool calls made for test: {test.get('name')}")
            return False
        
        # Check if the expected tool was called
        expected_tool = test.get("expected_tool")
        expected_args = test.get("expected_args", {})
        
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            function_name = function.get("name")
            
            if function_name == expected_tool:
                # Parse arguments
                try:
                    args_str = function.get("arguments", "{}")
                    actual_args = json.loads(args_str)
                    
                    # Check if required arguments are present
                    for key, expected_value in expected_args.items():
                        if key not in actual_args:
                            logger.info(f"Missing argument '{key}' in tool call")
                            continue
                        
                        # For location-based arguments, do fuzzy matching
                        if key == "location" and isinstance(expected_value, str) and isinstance(actual_args[key], str):
                            if expected_value.lower() in actual_args[key].lower() or actual_args[key].lower() in expected_value.lower():
                                continue
                        
                        # For exact matches
                        if actual_args[key] == expected_value:
                            continue
                        
                        # If we get here, the argument doesn't match
                        logger.info(f"Argument mismatch for '{key}': expected {expected_value}, got {actual_args[key]}")
                    
                    # If we made it here, the tool call is valid
                    logger.info(f"Successful tool call validation for test: {test.get('name')}")
                    return True
                    
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in tool call arguments: {function.get('arguments')}")
                    return False
        
        logger.info(f"Expected tool '{expected_tool}' not called")
        return False
        
    except Exception as e:
        logger.error(f"Error validating tool call response: {e}")
        return False

def validate_basic_response(response: Dict[str, Any], test: Dict[str, Any]) -> bool:
    """Validate basic test response contains expected content"""
    try:
        choices = response.get("choices", [])
        if not choices:
            return False
        
        message = choices[0].get("message", {})
        content = message.get("content", "").lower()
        
        expected_contains = test.get("expected_response_contains", [])
        
        # Check if any of the expected phrases are in the response
        for expected in expected_contains:
            if expected.lower() in content:
                logger.info(f"Basic test passed: found '{expected}' in response")
                return True
        
        logger.info(f"Basic test failed: none of {expected_contains} found in response")
        return False
        
    except Exception as e:
        logger.error(f"Error validating basic response: {e}")
        return False

def validate_advanced_response(response: Dict[str, Any], test: Dict[str, Any]) -> bool:
    """Validate advanced test response shows reasoning"""
    try:
        choices = response.get("choices", [])
        if not choices:
            return False
        
        message = choices[0].get("message", {})
        content = message.get("content", "").lower()
        
        expected_contains = test.get("expected_response_contains", [])
        
        # For advanced tests, we need more sophisticated validation
        # Check if response contains expected reasoning elements
        for expected in expected_contains:
            if expected.lower() in content:
                logger.info(f"Advanced test passed: found '{expected}' in response")
                return True
        
        # Also check for reasoning indicators
        reasoning_indicators = ["because", "therefore", "since", "thus", "so", "step", "first", "then", "calculate"]
        reasoning_found = any(indicator in content for indicator in reasoning_indicators)
        
        if reasoning_found and len(content) > 50:  # Ensure substantial response
            logger.info("Advanced test passed: reasoning indicators found")
            return True
        
        logger.info(f"Advanced test failed: insufficient reasoning in response")
        return False
        
    except Exception as e:
        logger.error(f"Error validating advanced response: {e}")
        return False

@app.post("/api/v1/benchmark/bfcl/start")
async def start_bfcl_benchmark(
    test_category: str = "tool_calling",
    max_examples: int = 10
):
    """Start a new BFCL benchmark test"""
    try:
        benchmark_id = str(uuid.uuid4())
        
        # Load test cases
        test_cases = load_benchmark_tests()
        category_tests = test_cases.get(test_category, [])
        
        # Limit tests to max_examples
        if max_examples > 0:
            category_tests = category_tests[:max_examples]
        
        # Create benchmark entry
        benchmark = create_benchmark_result(benchmark_id, test_category, len(category_tests))
        benchmark["total"] = len(category_tests)
        
        # Run the actual benchmark tests
        benchmark["status"] = BenchmarkStatus.RUNNING
        
        async def run_benchmark_tests():
            try:
                test_cases_data = load_benchmark_tests()
                tests = test_cases_data.get(test_category, [])[:max_examples]
                total_tests = len(tests)
                
                benchmarks_storage[benchmark_id]["total"] = total_tests
                benchmarks_storage[benchmark_id]["progress"] = 0
                
                # Run actual tests
                for i, test in enumerate(tests):
                    if benchmarks_storage[benchmark_id]["status"] == BenchmarkStatus.CANCELLED:
                        break
                    
                    benchmarks_storage[benchmark_id]["current_test"] = test.get("name", f"Test {i+1}")
                    benchmarks_storage[benchmark_id]["progress"] = i + 1
                    
                    # Run the actual test
                    try:
                        success = await run_single_test(test, test_category)
                        if success:
                            benchmarks_storage[benchmark_id]["results"]["correct"] += 1
                        else:
                            benchmarks_storage[benchmark_id]["results"]["errors"].append(
                                f"Failed test: {test.get('name', f'Test {i+1}')}"
                            )
                    except Exception as test_error:
                        logger.error(f"Test execution error: {test_error}")
                        benchmarks_storage[benchmark_id]["results"]["errors"].append(
                            f"Error in {test.get('name', f'Test {i+1}')}: {str(test_error)}"
                        )
                
                # Finalize results
                if benchmarks_storage[benchmark_id]["status"] != BenchmarkStatus.CANCELLED:
                    benchmarks_storage[benchmark_id]["results"]["total"] = total_tests
                    if total_tests > 0:
                        accuracy = (benchmarks_storage[benchmark_id]["results"]["correct"] / total_tests) * 100
                        benchmarks_storage[benchmark_id]["results"]["accuracy"] = round(accuracy, 2)
                    
                    benchmarks_storage[benchmark_id]["status"] = BenchmarkStatus.COMPLETED
                    benchmarks_storage[benchmark_id]["completed_at"] = datetime.now().isoformat()
                
            except Exception as e:
                logger.error(f"Benchmark execution failed: {e}")
                benchmarks_storage[benchmark_id]["status"] = BenchmarkStatus.FAILED
                benchmarks_storage[benchmark_id]["error"] = str(e)
                benchmarks_storage[benchmark_id]["completed_at"] = datetime.now().isoformat()
        
        # Start the benchmark tests in background
        asyncio.create_task(run_benchmark_tests())
        
        return {
            "benchmark_id": benchmark_id,
            "status": benchmark["status"],
            "test_category": test_category,
            "max_examples": len(category_tests)
        }
        
    except Exception as e:
        logger.error(f"Error starting benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Conversation Storage API Endpoints
# ============================================================================

@app.post("/api/v1/conversations")
async def create_conversation(
    title: Optional[str] = None,
    model: Optional[str] = None
):
    """Create a new conversation"""
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation storage not available")
    
    try:
        conversation_id = conversation_store.create_conversation(
            title=title,
            model=model
        )
        return {"id": conversation_id, "title": title or "New Conversation", "model": model}
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/conversations")
async def list_conversations(
    include_archived: bool = False,
    limit: int = 50,
    offset: int = 0,
    search: Optional[str] = None
):
    """List conversations with pagination and optional search"""
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation storage not available")
    
    try:
        result = conversation_store.list_conversations(
            include_archived=include_archived,
            limit=limit,
            offset=offset,
            search=search
        )
        return result
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/conversations/stats")
async def get_conversation_statistics():
    """Get conversation statistics"""
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation storage not available")
    
    try:
        return conversation_store.get_statistics()
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a conversation by ID with all messages"""
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation storage not available")
    
    try:
        conversation = conversation_store.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    title: Optional[str] = None,
    model: Optional[str] = None,
    is_archived: Optional[bool] = None
):
    """Update a conversation's properties"""
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation storage not available")
    
    try:
        success = conversation_store.update_conversation(
            conversation_id=conversation_id,
            title=title,
            model=model,
            is_archived=is_archived
        )
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages"""
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation storage not available")
    
    try:
        success = conversation_store.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/conversations/{conversation_id}/messages")
async def add_message(
    conversation_id: str,
    role: str,
    content: str,
    name: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    reasoning_content: Optional[str] = None
):
    """Add a message to a conversation"""
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation storage not available")
    
    if role not in ['system', 'user', 'assistant', 'tool']:
        raise HTTPException(status_code=400, detail="Invalid role. Must be: system, user, assistant, or tool")
    
    try:
        message_id = conversation_store.add_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            reasoning_content=reasoning_content
        )
        return {"id": message_id, "role": role}
    except Exception as e:
        logger.error(f"Error adding message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/conversations/{conversation_id}/export")
async def export_conversation(
    conversation_id: str,
    format: str = "json"
):
    """Export a conversation to JSON or Markdown format"""
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation storage not available")
    
    if format not in ['json', 'markdown']:
        raise HTTPException(status_code=400, detail="Invalid format. Must be: json or markdown")
    
    try:
        exported = conversation_store.export_conversation(conversation_id, format)
        if exported is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        content_type = "application/json" if format == "json" else "text/markdown"
        return {
            "content": exported,
            "content_type": content_type,
            "format": format
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# BFCL Benchmark Endpoints
# ============================================================================

@app.get("/api/v1/benchmark/bfcl")
async def list_bfcl_benchmarks():
    """List all BFCL benchmarks"""
    return benchmarks_storage

@app.get("/api/v1/benchmark/bfcl/{benchmark_id}")
async def get_bfcl_benchmark(benchmark_id: str):
    """Get specific BFCL benchmark status"""
    if benchmark_id not in benchmarks_storage:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return benchmarks_storage[benchmark_id]

@app.delete("/api/v1/benchmark/bfcl/{benchmark_id}")
async def stop_bfcl_benchmark(benchmark_id: str):
    """Stop a running BFCL benchmark"""
    if benchmark_id not in benchmarks_storage:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    if benchmarks_storage[benchmark_id]["status"] in [BenchmarkStatus.RUNNING, BenchmarkStatus.STARTING]:
        benchmarks_storage[benchmark_id]["status"] = BenchmarkStatus.CANCELLED
        benchmarks_storage[benchmark_id]["completed_at"] = datetime.now().isoformat()
    
    return {
        "status": benchmarks_storage[benchmark_id]["status"],
        "benchmark_id": benchmark_id
    }

@app.delete("/api/v1/benchmark/bfcl")
async def clear_completed_benchmarks():
    """Clear completed/failed/cancelled benchmarks"""
    completed_statuses = [BenchmarkStatus.COMPLETED, BenchmarkStatus.FAILED, BenchmarkStatus.CANCELLED]
    to_remove = [
        bid for bid, benchmark in benchmarks_storage.items() 
        if benchmark["status"] in completed_statuses
    ]
    
    for bid in to_remove:
        del benchmarks_storage[bid]
    
    return {"status": "success", "cleared": len(to_remove)}


# ============================================================================
# VRAM Estimation Endpoint
# ============================================================================

@app.post("/api/v1/estimate/vram")
async def estimate_vram_requirements(request: Request):
    """
    Estimate VRAM requirements for a model configuration.
    
    Formula based on:
    - Model weights: params_b * (bits_per_param / 8) * overhead_factor
    - KV cache: 2 * layers * head_dim * num_heads * ctx_size * batch_size * bytes_per_element
    - Activation memory: Additional overhead for computation
    """
    try:
        data = await request.json()
        
        # Model parameters (in billions)
        params_b = float(data.get('parameters', 7))  # Default to 7B
        
        # Quantization bits (approximate)
        quant_type = data.get('quantization', 'Q4_K_M')
        quant_bits = {
            'Q2_K': 2.5,
            'Q3_K_S': 3.0,
            'Q3_K_M': 3.3,
            'Q3_K_L': 3.5,
            'Q4_0': 4.0,
            'Q4_K_S': 4.3,
            'Q4_K_M': 4.5,
            'Q5_0': 5.0,
            'Q5_K_S': 5.3,
            'Q5_K_M': 5.5,
            'Q6_K': 6.5,
            'Q8_0': 8.0,
            'F16': 16.0,
            'F32': 32.0,
            'MXFP4': 4.0,  # Microsoft FP4 format
        }.get(quant_type.upper(), 4.5)
        
        # Context and batch settings
        ctx_size = int(data.get('context_size', 8192))
        batch_size = int(data.get('batch_size', 1))
        
        # Model architecture estimates (reasonable defaults for transformer models)
        num_layers = int(data.get('num_layers', 0))
        if num_layers == 0:
            # Estimate layers based on model size
            if params_b <= 3:
                num_layers = 26
            elif params_b <= 7:
                num_layers = 32
            elif params_b <= 13:
                num_layers = 40
            elif params_b <= 20:
                num_layers = 48
            elif params_b <= 34:
                num_layers = 60
            elif params_b <= 70:
                num_layers = 80
            else:
                num_layers = 96
        
        head_dim = int(data.get('head_dim', 128))
        num_kv_heads = int(data.get('num_kv_heads', 0))
        if num_kv_heads == 0:
            # Estimate KV heads (GQA models use fewer)
            if params_b <= 7:
                num_kv_heads = 8
            elif params_b <= 13:
                num_kv_heads = 8
            elif params_b <= 34:
                num_kv_heads = 8
            else:
                num_kv_heads = 8
        
        # KV cache quantization
        kv_cache_type = data.get('kv_cache_type', 'f16')
        kv_bytes = {
            'f32': 4,
            'f16': 2,
            'q8_0': 1,
            'q4_0': 0.5,
        }.get(kv_cache_type.lower(), 2)
        
        # Calculate model weights VRAM (in GB)
        model_vram_gb = params_b * (quant_bits / 8) * 1.1  # 10% overhead for tensor structures
        
        # Calculate KV cache VRAM (in GB)
        # KV cache = 2 (K and V) * layers * head_dim * num_kv_heads * ctx_size * batch_size * bytes / 1e9
        kv_cache_gb = (2 * num_layers * head_dim * num_kv_heads * ctx_size * batch_size * kv_bytes) / 1e9
        
        # Activation memory (rough estimate: ~10-20% of model weights for inference)
        activation_gb = model_vram_gb * 0.15
        
        # Total VRAM estimate
        total_vram_gb = model_vram_gb + kv_cache_gb + activation_gb
        
        # Add buffer for CUDA runtime and other overhead
        cuda_overhead_gb = 0.5
        total_with_overhead_gb = total_vram_gb + cuda_overhead_gb
        
        # Recommendation
        if total_with_overhead_gb <= 6:
            recommendation = "Should fit on 8GB GPUs"
        elif total_with_overhead_gb <= 10:
            recommendation = "Recommended: 12GB+ GPU"
        elif total_with_overhead_gb <= 14:
            recommendation = "Recommended: 16GB+ GPU"
        elif total_with_overhead_gb <= 22:
            recommendation = "Recommended: 24GB+ GPU"
        elif total_with_overhead_gb <= 44:
            recommendation = "Recommended: 48GB+ GPU or multi-GPU"
        else:
            recommendation = "Requires multi-GPU setup or model offloading"
        
        return {
            "estimate": {
                "model_weights_gb": round(model_vram_gb, 2),
                "kv_cache_gb": round(kv_cache_gb, 2),
                "activation_memory_gb": round(activation_gb, 2),
                "cuda_overhead_gb": round(cuda_overhead_gb, 2),
                "total_vram_gb": round(total_with_overhead_gb, 2),
            },
            "inputs": {
                "parameters_b": params_b,
                "quantization": quant_type,
                "quantization_bits": quant_bits,
                "context_size": ctx_size,
                "batch_size": batch_size,
                "num_layers": num_layers,
                "head_dim": head_dim,
                "num_kv_heads": num_kv_heads,
                "kv_cache_type": kv_cache_type,
            },
            "recommendation": recommendation,
        }
        
    except Exception as e:
        logger.error(f"Error estimating VRAM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Conversation Storage Endpoints
# ============================================================================

@app.get("/api/v1/conversations")
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    search: Optional[str] = None,
    tags: Optional[str] = None,
):
    """List all conversations with optional filtering."""
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation storage not available")
    
    tag_list = tags.split(',') if tags else None
    result = conversation_store.list_conversations(
        limit=limit,
        offset=offset,
        search=search,
        tags=tag_list,
    )
    return result


@app.post("/api/v1/conversations")
async def create_conversation(request: Request):
    """Create a new conversation."""
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation storage not available")
    
    try:
        data = await request.json()
        conversation = conversation_store.create_conversation(
            title=data.get('title'),
            messages=data.get('messages', []),
            model=data.get('model'),
            settings=data.get('settings'),
            tags=data.get('tags', []),
        )
        return conversation.to_dict()
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation by ID."""
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation storage not available")
    
    conversation = conversation_store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversation.to_dict()


@app.put("/api/v1/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, request: Request):
    """Update an existing conversation."""
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation storage not available")
    
    try:
        data = await request.json()
        conversation = conversation_store.update_conversation(
            conversation_id=conversation_id,
            messages=data.get('messages'),
            title=data.get('title'),
            model=data.get('model'),
            settings=data.get('settings'),
            tags=data.get('tags'),
        )
        
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return conversation.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/conversations/{conversation_id}/messages")
async def add_message_to_conversation(conversation_id: str, request: Request):
    """Add a message to an existing conversation."""
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation storage not available")
    
    try:
        data = await request.json()
        conversation = conversation_store.add_message(
            conversation_id=conversation_id,
            message=data,
        )
        
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return conversation.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation storage not available")
    
    success = conversation_store.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"status": "success", "deleted": conversation_id}


@app.get("/api/v1/conversations/{conversation_id}/export")
async def export_conversation(
    conversation_id: str,
    format: str = "json",
):
    """Export a conversation in the specified format (json or markdown)."""
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation storage not available")
    
    if format not in ['json', 'markdown']:
        raise HTTPException(status_code=400, detail="Format must be 'json' or 'markdown'")
    
    exported = conversation_store.export_conversation(conversation_id, format)
    if exported is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    media_type = 'application/json' if format == 'json' else 'text/markdown'
    filename = f"conversation-{conversation_id}.{'json' if format == 'json' else 'md'}"
    
    from fastapi.responses import Response
    return Response(
        content=exported,
        media_type=media_type,
        headers={'Content-Disposition': f'attachment; filename="{filename}"'}
    )


@app.get("/api/v1/conversations/search")
async def search_conversations(query: str):
    """Search conversations by content."""
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation storage not available")
    
    results = conversation_store.search_conversations(query)
    return {"results": results}


# ============================================================================
# Model Registry Endpoints
# ============================================================================

@app.get("/api/v1/registry/stats")
async def get_registry_stats():
    """Get model registry statistics."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    return model_registry.get_registry_stats()


@app.get("/api/v1/registry/models")
async def list_cached_models(
    limit: int = 50,
    offset: int = 0,
    search: str = None,
    model_type: str = None,
):
    """List all cached models with optional filtering."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    return model_registry.list_cached_models(
        limit=limit,
        offset=offset,
        search=search,
        model_type=model_type,
    )


@app.post("/api/v1/registry/models")
async def cache_model(request: Request):
    """Cache model metadata from HuggingFace."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    data = await request.json()
    model_id = model_registry.cache_model(
        repo_id=data.get('repo_id'),
        name=data.get('name'),
        description=data.get('description'),
        author=data.get('author'),
        downloads=data.get('downloads', 0),
        likes=data.get('likes', 0),
        tags=data.get('tags', []),
        model_type=data.get('model_type'),
        license=data.get('license'),
        last_modified=data.get('last_modified'),
        metadata=data.get('metadata', {}),
    )
    
    return {"status": "cached", "model_id": model_id}


@app.get("/api/v1/registry/models/{repo_id:path}")
async def get_cached_model(repo_id: str):
    """Get cached model metadata."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    model = model_registry.get_cached_model(repo_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found in cache")
    
    return model


@app.delete("/api/v1/registry/models/{repo_id:path}")
async def delete_cached_model(repo_id: str):
    """Delete a model from the cache."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    deleted = model_registry.delete_model_cache(repo_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Model not found in cache")
    
    return {"status": "deleted", "repo_id": repo_id}


@app.post("/api/v1/registry/models/{repo_id:path}/variants")
async def add_model_variant(repo_id: str, request: Request):
    """Add a quantization variant for a model."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    data = await request.json()
    model_registry.add_variant(
        repo_id=repo_id,
        filename=data.get('filename'),
        quantization=data.get('quantization'),
        size_bytes=data.get('size_bytes'),
        vram_required_mb=data.get('vram_required_mb'),
        quality_score=data.get('quality_score'),
        speed_score=data.get('speed_score'),
    )
    
    return {"status": "added"}


@app.get("/api/v1/registry/models/{repo_id:path}/variants")
async def get_model_variants(repo_id: str):
    """Get all quantization variants for a model."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    return {"variants": model_registry.get_variants(repo_id)}


@app.post("/api/v1/registry/models/{repo_id:path}/usage/load")
async def record_model_load(repo_id: str, variant: str = None):
    """Record that a model was loaded."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    model_registry.record_model_load(repo_id, variant)
    return {"status": "recorded"}


@app.post("/api/v1/registry/models/{repo_id:path}/usage/inference")
async def record_inference(repo_id: str, request: Request):
    """Record inference statistics."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    data = await request.json()
    model_registry.record_inference(
        repo_id=repo_id,
        variant=data.get('variant'),
        tokens_generated=data.get('tokens_generated', 0),
        inference_time_ms=data.get('inference_time_ms', 0),
    )
    
    return {"status": "recorded"}


@app.get("/api/v1/registry/usage")
async def get_usage_stats(repo_id: str = None):
    """Get usage statistics for models."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    return {"usage": model_registry.get_usage_stats(repo_id)}


@app.get("/api/v1/registry/most-used")
async def get_most_used_models(limit: int = 10):
    """Get the most frequently used models."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    return {"models": model_registry.get_most_used_models(limit)}


@app.post("/api/v1/registry/models/{repo_id:path}/rating")
async def set_model_rating(repo_id: str, request: Request):
    """Set user rating and notes for a model."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    data = await request.json()
    try:
        model_registry.set_rating(
            repo_id=repo_id,
            rating=data.get('rating'),
            variant=data.get('variant'),
            notes=data.get('notes'),
            tags=data.get('tags'),
        )
        return {"status": "saved"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/registry/models/{repo_id:path}/rating")
async def get_model_rating(repo_id: str, variant: str = None):
    """Get user rating for a model."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    rating = model_registry.get_rating(repo_id, variant)
    if rating is None:
        return {"rating": None}
    
    return rating


@app.post("/api/v1/registry/models/{repo_id:path}/hardware")
async def set_hardware_recommendation(repo_id: str, request: Request):
    """Set hardware recommendations for a model variant."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    data = await request.json()
    model_registry.set_hardware_recommendation(
        repo_id=repo_id,
        variant=data.get('variant'),
        min_vram_gb=data.get('min_vram_gb'),
        recommended_vram_gb=data.get('recommended_vram_gb'),
        min_ram_gb=data.get('min_ram_gb'),
        recommended_context_size=data.get('recommended_context_size'),
        gpu_layers_recommendation=data.get('gpu_layers_recommendation'),
        notes=data.get('notes'),
    )
    
    return {"status": "saved"}


@app.get("/api/v1/registry/recommendations")
async def get_recommendations_for_hardware(
    vram_gb: float,
    ram_gb: float = None,
):
    """Get model recommendations based on available hardware."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    return {
        "recommendations": model_registry.get_recommendations_for_hardware(
            available_vram_gb=vram_gb,
            available_ram_gb=ram_gb,
        )
    }


# ============================================================================
# Prompt Library Endpoints
# ============================================================================

@app.get("/api/v1/prompts/stats")
async def get_prompt_stats():
    """Get prompt library statistics."""
    if prompt_library is None:
        raise HTTPException(status_code=503, detail="Prompt library not available")
    
    return prompt_library.get_stats()


@app.get("/api/v1/prompts/categories")
async def list_prompt_categories():
    """List all prompt categories."""
    if prompt_library is None:
        raise HTTPException(status_code=503, detail="Prompt library not available")
    
    return {"categories": prompt_library.list_categories()}


@app.post("/api/v1/prompts/categories")
async def create_prompt_category(request: Request):
    """Create a new prompt category."""
    if prompt_library is None:
        raise HTTPException(status_code=503, detail="Prompt library not available")
    
    data = await request.json()
    category = prompt_library.create_category(
        name=data.get('name'),
        description=data.get('description'),
        color=data.get('color', '#6B7280'),
        icon=data.get('icon', 'folder'),
        parent_id=data.get('parent_id'),
    )
    return category


@app.get("/api/v1/prompts")
async def list_prompts(
    category: str = None,
    search: str = None,
    is_system_prompt: bool = None,
    is_favorite: bool = None,
    limit: int = 50,
    offset: int = 0,
    order_by: str = 'updated_at',
    order_dir: str = 'DESC',
):
    """List prompts with optional filtering."""
    if prompt_library is None:
        raise HTTPException(status_code=503, detail="Prompt library not available")
    
    return prompt_library.list_prompts(
        category=category,
        search=search,
        is_system_prompt=is_system_prompt,
        is_favorite=is_favorite,
        limit=limit,
        offset=offset,
        order_by=order_by,
        order_dir=order_dir,
    )


@app.post("/api/v1/prompts")
async def create_prompt(request: Request):
    """Create a new prompt template."""
    if prompt_library is None:
        raise HTTPException(status_code=503, detail="Prompt library not available")
    
    data = await request.json()
    prompt = prompt_library.create_prompt(
        name=data.get('name'),
        content=data.get('content'),
        description=data.get('description'),
        category=data.get('category', 'general'),
        tags=data.get('tags', []),
        is_system_prompt=data.get('is_system_prompt', False),
        created_by=data.get('created_by'),
        metadata=data.get('metadata', {}),
    )
    return prompt


@app.get("/api/v1/prompts/{prompt_id}")
async def get_prompt(prompt_id: str):
    """Get a prompt by ID."""
    if prompt_library is None:
        raise HTTPException(status_code=503, detail="Prompt library not available")
    
    prompt = prompt_library.get_prompt(prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    return prompt


@app.put("/api/v1/prompts/{prompt_id}")
async def update_prompt(prompt_id: str, request: Request):
    """Update a prompt."""
    if prompt_library is None:
        raise HTTPException(status_code=503, detail="Prompt library not available")
    
    data = await request.json()
    prompt = prompt_library.update_prompt(
        prompt_id=prompt_id,
        name=data.get('name'),
        content=data.get('content'),
        description=data.get('description'),
        category=data.get('category'),
        tags=data.get('tags'),
        is_system_prompt=data.get('is_system_prompt'),
        is_favorite=data.get('is_favorite'),
        metadata=data.get('metadata'),
        change_note=data.get('change_note'),
    )
    
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    return prompt


@app.delete("/api/v1/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str):
    """Delete a prompt."""
    if prompt_library is None:
        raise HTTPException(status_code=503, detail="Prompt library not available")
    
    deleted = prompt_library.delete_prompt(prompt_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    return {"status": "deleted", "prompt_id": prompt_id}


@app.get("/api/v1/prompts/{prompt_id}/versions")
async def get_prompt_versions(prompt_id: str):
    """Get all versions of a prompt."""
    if prompt_library is None:
        raise HTTPException(status_code=503, detail="Prompt library not available")
    
    return {"versions": prompt_library.get_versions(prompt_id)}


@app.post("/api/v1/prompts/{prompt_id}/restore/{version}")
async def restore_prompt_version(prompt_id: str, version: int):
    """Restore a prompt to a specific version."""
    if prompt_library is None:
        raise HTTPException(status_code=503, detail="Prompt library not available")
    
    prompt = prompt_library.restore_version(prompt_id, version)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt or version not found")
    
    return prompt


@app.post("/api/v1/prompts/{prompt_id}/render")
async def render_prompt(prompt_id: str, request: Request):
    """Render a prompt template with given variables."""
    if prompt_library is None:
        raise HTTPException(status_code=503, detail="Prompt library not available")
    
    data = await request.json()
    try:
        rendered = prompt_library.render_prompt(
            prompt_id=prompt_id,
            variables=data.get('variables', {}),
        )
        return {"rendered": rendered}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/v1/prompts/export")
async def export_prompts(request: Request):
    """Export prompts to JSON."""
    if prompt_library is None:
        raise HTTPException(status_code=503, detail="Prompt library not available")
    
    data = await request.json()
    prompt_ids = data.get('prompt_ids')  # None exports all
    
    exported = prompt_library.export_prompts(prompt_ids)
    return {"data": exported}


@app.post("/api/v1/prompts/import")
async def import_prompts(request: Request):
    """Import prompts from JSON."""
    if prompt_library is None:
        raise HTTPException(status_code=503, detail="Prompt library not available")
    
    data = await request.json()
    result = prompt_library.import_prompts(
        json_data=data.get('data'),
        overwrite=data.get('overwrite', False),
    )
    return result


# ============================================================================
# Benchmark Endpoints
# ============================================================================

@app.get("/api/v1/benchmark/stats")
async def get_benchmark_stats():
    """Get overall benchmark statistics."""
    if benchmark_runner is None:
        raise HTTPException(status_code=503, detail="Benchmark runner not available")
    
    return benchmark_runner.get_stats()


@app.get("/api/v1/benchmark/presets")
async def list_benchmark_presets():
    """List available benchmark presets."""
    if benchmark_runner is None:
        raise HTTPException(status_code=503, detail="Benchmark runner not available")
    
    return {"presets": benchmark_runner.list_presets()}


@app.get("/api/v1/benchmark/history")
async def list_benchmarks(
    limit: int = 50,
    offset: int = 0,
    status: str = None,
):
    """List benchmark history."""
    if benchmark_runner is None:
        raise HTTPException(status_code=503, detail="Benchmark runner not available")
    
    return benchmark_runner.list_benchmarks(limit=limit, offset=offset, status=status)


@app.post("/api/v1/benchmark/run")
async def run_benchmark(request: Request, background_tasks: BackgroundTasks):
    """Start a new benchmark run."""
    if benchmark_runner is None:
        raise HTTPException(status_code=503, detail="Benchmark runner not available")
    
    data = await request.json()
    
    # Get preset or custom config
    preset_name = data.get('preset', 'standard')
    config = benchmark_runner.get_preset(preset_name)
    
    # Override with custom values if provided
    if 'prompt_tokens' in data:
        config.prompt_tokens = data['prompt_tokens']
    if 'max_output_tokens' in data:
        config.max_output_tokens = data['max_output_tokens']
    if 'num_runs' in data:
        config.num_runs = data['num_runs']
    if 'warmup_runs' in data:
        config.warmup_runs = data['warmup_runs']
    if 'temperature' in data:
        config.temperature = data['temperature']
    
    model_name = data.get('model_name', 'unknown')
    model_variant = data.get('model_variant', 'unknown')
    api_key = data.get('api_key')
    
    # Run benchmark in background
    async def run_in_background():
        try:
            await benchmark_runner.run_benchmark(
                config=config,
                model_name=model_name,
                model_variant=model_variant,
                api_key=api_key,
            )
        except Exception as e:
            logger.error(f"Background benchmark failed: {e}")
    
    # Start the benchmark
    import asyncio
    loop = asyncio.get_event_loop()
    task = loop.create_task(run_in_background())
    
    # Wait a moment to get the benchmark ID
    await asyncio.sleep(0.1)
    
    # Find the active benchmark
    active = list(benchmark_runner._active_benchmarks.keys())
    benchmark_id = active[-1] if active else None
    
    return {
        "status": "started",
        "benchmark_id": benchmark_id,
        "config": {
            "preset": preset_name,
            "prompt_tokens": config.prompt_tokens,
            "max_output_tokens": config.max_output_tokens,
            "num_runs": config.num_runs,
        }
    }


@app.get("/api/v1/benchmark/{benchmark_id}")
async def get_benchmark(benchmark_id: str):
    """Get benchmark status and results."""
    if benchmark_runner is None:
        raise HTTPException(status_code=503, detail="Benchmark runner not available")
    
    result = benchmark_runner.get_benchmark(benchmark_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    return result


@app.delete("/api/v1/benchmark/{benchmark_id}")
async def delete_benchmark(benchmark_id: str):
    """Delete a benchmark."""
    if benchmark_runner is None:
        raise HTTPException(status_code=503, detail="Benchmark runner not available")
    
    deleted = benchmark_runner.delete_benchmark(benchmark_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    return {"status": "deleted", "benchmark_id": benchmark_id}


# ============================================================================
# Batch Processing Endpoints
# ============================================================================

@app.get("/api/v1/batch/stats")
async def get_batch_stats():
    """Get batch processing statistics."""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="Batch processor not available")
    
    return batch_processor.get_stats()


@app.get("/api/v1/batch/jobs")
async def list_batch_jobs(
    status: str = None,
    limit: int = 50,
    offset: int = 0,
):
    """List batch jobs."""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="Batch processor not available")
    
    return batch_processor.list_jobs(status=status, limit=limit, offset=offset)


@app.post("/api/v1/batch/jobs")
async def create_batch_job(request: Request):
    """Create a new batch job."""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="Batch processor not available")
    
    data = await request.json()
    
    # Parse input data
    items = []
    if 'items' in data:
        # Direct items array
        items = data['items']
    elif 'content' in data and 'file_type' in data:
        # File content to parse
        items = batch_processor.parse_input_file(data['content'], data['file_type'])
    else:
        raise HTTPException(status_code=400, detail="Must provide 'items' or 'content' with 'file_type'")
    
    if not items:
        raise HTTPException(status_code=400, detail="No items to process")
    
    config = data.get('config', {})
    name = data.get('name', f"Batch Job {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    job = batch_processor.create_batch_job(name=name, items=items, config=config)
    
    return {
        "status": "created",
        "job_id": job.id,
        "total_items": job.total_items,
    }


@app.post("/api/v1/batch/jobs/{job_id}/run")
async def run_batch_job(job_id: str, request: Request, background_tasks: BackgroundTasks):
    """Start running a batch job."""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="Batch processor not available")
    
    job = batch_processor.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job['status'] not in ['pending', 'failed']:
        raise HTTPException(status_code=400, detail=f"Job cannot be run (status: {job['status']})")
    
    data = await request.json() if request.headers.get('content-type') == 'application/json' else {}
    api_key = data.get('api_key')
    
    # Run in background
    async def run_in_background():
        try:
            await batch_processor.run_batch_job(job_id, api_key=api_key)
        except Exception as e:
            logger.error(f"Background batch job failed: {e}")
    
    import asyncio
    loop = asyncio.get_event_loop()
    loop.create_task(run_in_background())
    
    return {"status": "started", "job_id": job_id}


@app.get("/api/v1/batch/jobs/{job_id}")
async def get_batch_job(job_id: str):
    """Get batch job details."""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="Batch processor not available")
    
    job = batch_processor.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job


@app.get("/api/v1/batch/jobs/{job_id}/items")
async def get_batch_job_items(
    job_id: str,
    status: str = None,
    limit: int = 100,
    offset: int = 0,
):
    """Get items for a batch job."""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="Batch processor not available")
    
    return batch_processor.get_job_items(job_id, status=status, limit=limit, offset=offset)


@app.post("/api/v1/batch/jobs/{job_id}/cancel")
async def cancel_batch_job(job_id: str):
    """Cancel a running batch job."""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="Batch processor not available")
    
    job = batch_processor.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job['status'] != 'running':
        raise HTTPException(status_code=400, detail="Job is not running")
    
    batch_processor.cancel_job(job_id)
    return {"status": "cancelling", "job_id": job_id}


@app.get("/api/v1/batch/jobs/{job_id}/export")
async def export_batch_job(job_id: str, format: str = "json"):
    """Export batch job results."""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="Batch processor not available")
    
    try:
        content = batch_processor.export_results(job_id, format=format)
        
        if format == 'csv':
            return Response(
                content=content,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=batch_{job_id}.csv"}
            )
        else:
            return Response(
                content=content,
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=batch_{job_id}.json"}
            )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/api/v1/batch/jobs/{job_id}")
async def delete_batch_job(job_id: str):
    """Delete a batch job."""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="Batch processor not available")
    
    deleted = batch_processor.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"status": "deleted", "job_id": job_id}


# =============================================================================
# VRAM Estimation Endpoints
# =============================================================================

try:
    from modules import vram_estimator
except ImportError:
    vram_estimator = None


@app.post("/api/v1/vram/estimate")
async def estimate_vram_api(request: Request):
    """Estimate VRAM requirements for model deployment."""
    if vram_estimator is None:
        raise HTTPException(status_code=503, detail="VRAM estimator not available")
    
    data = await request.json()
    
    estimate = vram_estimator.estimate_vram(
        model_name=data.get("model_name", ""),
        params_b=data.get("params_b"),
        quantization=data.get("quantization", "Q4_K_M"),
        context_size=data.get("context_size", 4096),
        batch_size=data.get("batch_size", 1),
        gpu_layers=data.get("gpu_layers", -1),
        kv_cache_type=data.get("kv_cache_type", "f16"),
        available_vram_gb=data.get("available_vram_gb", 24.0),
        flash_attention=data.get("flash_attention", True),
    )
    
    return {
        "model_weights_mb": estimate.model_weights_mb,
        "kv_cache_mb": estimate.kv_cache_mb,
        "compute_buffer_mb": estimate.compute_buffer_mb,
        "overhead_mb": estimate.overhead_mb,
        "total_mb": estimate.total_mb,
        "total_gb": estimate.total_gb,
        "fits_in_vram": estimate.fits_in_vram,
        "available_vram_gb": estimate.available_vram_gb,
        "utilization_percent": estimate.utilization_percent,
        "warnings": estimate.warnings,
    }


@app.get("/api/v1/vram/quantizations")
async def get_quantization_options():
    """Get available quantization options with bits per weight."""
    if vram_estimator is None:
        raise HTTPException(status_code=503, detail="VRAM estimator not available")
    
    return {"quantizations": vram_estimator.get_quantization_options()}


@app.get("/api/v1/vram/architectures")
async def get_model_architectures():
    """Get known model architectures."""
    if vram_estimator is None:
        raise HTTPException(status_code=503, detail="VRAM estimator not available")
    
    return {"architectures": vram_estimator.get_model_architectures()}


# =============================================================================
# RAG System Endpoints
# =============================================================================

def get_rag_components(request: Request):
    """Helper to get RAG components from app state."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG system not available")
    return {
        'document_manager': getattr(request.app.state, 'document_manager', None),
        'vector_store': getattr(request.app.state, 'vector_store', None),
        'graph_rag': getattr(request.app.state, 'graph_rag', None),
        'document_discovery': getattr(request.app.state, 'document_discovery', None),
    }


# Domain Management
@app.get("/api/v1/rag/domains")
async def list_rag_domains(request: Request):
    """List all document domains."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    domains = await rag['document_manager'].list_domains()
    return {"domains": [d.to_dict() for d in domains]}


@app.post("/api/v1/rag/domains")
async def create_rag_domain(request: Request):
    """Create a new domain."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    data = await request.json()
    from modules.rag.document_manager import Domain
    
    domain = Domain(
        id=str(uuid.uuid4()),
        name=data.get('name', 'New Domain'),
        description=data.get('description', ''),
        parent_id=data.get('parent_id'),
        chunk_size=data.get('chunk_size', 512),
        chunk_overlap=data.get('chunk_overlap', 50),
        embedding_model=data.get('embedding_model', 'all-MiniLM-L6-v2')
    )
    
    result = await rag['document_manager'].create_domain(domain)
    return {"domain": result.to_dict()}


@app.get("/api/v1/rag/domains/{domain_id}")
async def get_rag_domain(request: Request, domain_id: str):
    """Get domain details."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    domain = await rag['document_manager'].get_domain(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail="Domain not found")
    return {"domain": domain.to_dict()}


@app.delete("/api/v1/rag/domains/{domain_id}")
async def delete_rag_domain(request: Request, domain_id: str, cascade: bool = False):
    """Delete a domain."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    deleted = await rag['document_manager'].delete_domain(domain_id, cascade)
    if not deleted:
        raise HTTPException(status_code=400, detail="Cannot delete domain with documents")
    return {"status": "deleted"}


@app.get("/api/v1/rag/domains/tree")
async def get_domain_tree(request: Request):
    """Get hierarchical domain tree."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    tree = await rag['document_manager'].get_domain_tree()
    return {"tree": tree}


# Document Management
@app.get("/api/v1/rag/documents")
async def list_rag_documents(
    request: Request,
    domain_id: Optional[str] = None,
    status: Optional[str] = None,
    doc_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List documents with filtering."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    from modules.rag.document_manager import DocumentStatus, DocumentType
    
    status_enum = DocumentStatus(status) if status else None
    type_enum = DocumentType(doc_type) if doc_type else None
    
    documents, total = await rag['document_manager'].list_documents(
        domain_id=domain_id,
        status=status_enum,
        doc_type=type_enum,
        search=search,
        limit=limit,
        offset=offset
    )
    
    return {
        "documents": [d.to_dict() for d in documents],
        "total": total,
        "limit": limit,
        "offset": offset
    }


@app.post("/api/v1/rag/documents")
async def create_rag_document(request: Request, background_tasks: BackgroundTasks):
    """Create/upload a new document."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")

    data = await request.json()
    from modules.rag.document_manager import Document, DocumentType, DocumentStatus

    # Check for duplicate
    content = data.get('content', '')
    if content:
        existing = await rag['document_manager'].check_duplicate(content)
        if existing:
            raise HTTPException(status_code=409, detail=f"Duplicate content, existing document: {existing}")

    doc = Document(
        id=str(uuid.uuid4()),
        domain_id=data.get('domain_id'),
        name=data.get('name', 'Untitled'),
        doc_type=DocumentType(data.get('doc_type', 'txt')),
        content=content,
        source_url=data.get('source_url'),
        metadata=data.get('metadata', {})
    )

    result = await rag['document_manager'].create_document(doc)
    
    # Auto-process if enabled (default: true)
    auto_process = data.get('auto_process', True)
    if auto_process and content:
        # Add background task to process document
        background_tasks.add_task(
            process_document_background,
            result.id,
            data.get('chunking_strategy', 'semantic'),
            data.get('chunk_size'),
            data.get('chunk_overlap'),
            data.get('embedding_model')
        )
        logger.info(f"Queued document {result.id} for background processing")
    
    return {"document": result.to_dict(), "auto_processing": auto_process}


@app.get("/api/v1/rag/documents/{document_id}")
async def get_rag_document(request: Request, document_id: str):
    """Get document details."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    doc = await rag['document_manager'].get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"document": doc.to_dict()}


@app.delete("/api/v1/rag/documents/{document_id}")
async def delete_rag_document(request: Request, document_id: str):
    """Delete a document."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    deleted = await rag['document_manager'].delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted"}


@app.get("/api/v1/rag/documents/{document_id}/chunks")
async def get_document_chunks(request: Request, document_id: str):
    """Get document chunks."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    chunks = await rag['document_manager'].get_chunks(document_id)
    return {"chunks": [c.to_dict() for c in chunks]}


async def process_document_background(
    document_id: str,
    chunking_strategy: str = 'semantic',
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embedding_model: Optional[str] = None
):
    """Background task to process a document."""
    try:
        # Get RAG components from app state
        rag = {
            'document_manager': app.state.document_manager,
            'vector_store': app.state.vector_store,
            'graph_rag': app.state.graph_rag,
            'document_discovery': app.state.document_discovery
        }
        
        from modules.rag.document_manager import DocumentStatus
        
        doc = await rag['document_manager'].get_document(document_id)
        if not doc:
            logger.error(f"Document {document_id} not found for processing")
            return
        
        # Update status to processing
        doc.status = DocumentStatus.PROCESSING
        await rag['document_manager'].update_document(doc)
        
        # Get domain settings
        domain = await rag['document_manager'].get_domain(doc.domain_id)
        
        # Select chunker
        config = ChunkingConfig(
            chunk_size=chunk_size or (domain.chunk_size if domain else 512),
            chunk_overlap=chunk_overlap or (domain.chunk_overlap if domain else 50)
        )
        
        if chunking_strategy == 'fixed':
            chunker = FixedChunker(config)
        elif chunking_strategy == 'recursive':
            chunker = RecursiveChunker(config)
        else:
            chunker = SemanticChunker(config)
        
        # Chunk document
        raw_chunks = chunker.chunk(doc.content)
        
        # Create embedder
        model_name = embedding_model or (domain.embedding_model if domain else 'all-MiniLM-L6-v2')
        embedder = create_embedder(model_name=model_name)
        
        # Embed chunks in batches to handle large documents
        texts = [c.content for c in raw_chunks]
        logger.info(f"Embedding {len(texts)} chunks for document {document_id}")
        
        # Process in smaller batches for large documents
        # Start with very small batch size for deployed service
        batch_size = 4 if USE_DEPLOYED_EMBEDDINGS else 32
        embed_result = await embedder.embed(texts, batch_size=batch_size, show_progress=False)
        
        # Ensure collection exists
        collection_name = f"domain_{doc.domain_id}"
        if not await rag['vector_store'].collection_exists(collection_name):
            from modules.rag.vector_stores.base import CollectionConfig
            collection_config = CollectionConfig(
                name=collection_name,
                vector_size=embed_result.dimensions,
                distance_metric='cosine'
            )
            await rag['vector_store'].create_collection(collection_config)
        
        # Store in vector store and create chunks
        from modules.rag.document_manager import DocumentChunk
        from modules.rag.vector_stores.base import VectorRecord
        
        chunks = []
        records = []
        
        for i, (chunk, embedding) in enumerate(zip(raw_chunks, embed_result.embeddings)):
            chunk_id = str(uuid.uuid4())
            vector_id = f"{document_id}_{i}"
            
            doc_chunk = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                content=chunk.content,
                chunk_index=chunk.index,
                total_chunks=len(raw_chunks),
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                vector_id=vector_id
            )
            chunks.append(doc_chunk)
            
            records.append(VectorRecord(
                id=vector_id,
                vector=embedding,
                payload={
                    'document_id': document_id,
                    'domain_id': doc.domain_id,
                    'content': chunk.content,
                    'chunk_index': chunk.index,
                    'chunk_id': chunk_id,
                    'metadata': doc.metadata
                }
            ))
        
        # Save chunks to document manager
        await rag['document_manager'].save_chunks(chunks)
        
        # Add to vector store
        await rag['vector_store'].add_vectors(collection_name, records)
        
        # Update document status
        doc.status = DocumentStatus.READY
        doc.chunk_count = len(chunks)
        doc.token_count = embed_result.token_count
        doc.processed_at = datetime.utcnow().isoformat()
        await rag['document_manager'].update_document(doc)
        
        logger.info(f"Successfully processed document {document_id}: {len(chunks)} chunks created")
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        # Update document status to error
        try:
            doc = await rag['document_manager'].get_document(document_id)
            if doc:
                doc.status = DocumentStatus.ERROR
                doc.error_message = str(e)
                await rag['document_manager'].update_document(doc)
        except:
            pass


@app.post("/api/v1/rag/documents/{document_id}/process")
async def process_document(request: Request, document_id: str, background_tasks: BackgroundTasks):
    """Process document (chunk and embed). Can run synchronously or in background."""
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    data = await request.json()
    
    # Check if document exists
    doc = await rag['document_manager'].get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Run in background or synchronously
    run_async = data.get('async', True)
    
    if run_async:
        # Queue for background processing
        background_tasks.add_task(
            process_document_background,
            document_id,
            data.get('chunking_strategy', 'semantic'),
            data.get('chunk_size'),
            data.get('chunk_overlap'),
            data.get('embedding_model')
        )
        return {
            "status": "queued",
            "document_id": document_id,
            "message": "Document queued for processing"
        }
    
    # Otherwise, process synchronously (original code follows)
    data = await request.json()
    
    doc = await rag['document_manager'].get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get domain settings
    domain = await rag['document_manager'].get_domain(doc.domain_id)
    
    # Select chunker
    strategy = data.get('chunking_strategy', 'semantic')
    config = ChunkingConfig(
        chunk_size=data.get('chunk_size', domain.chunk_size if domain else 512),
        chunk_overlap=data.get('chunk_overlap', domain.chunk_overlap if domain else 50)
    )
    
    if strategy == 'fixed':
        chunker = FixedChunker(config)
    elif strategy == 'recursive':
        chunker = RecursiveChunker(config)
    else:
        chunker = SemanticChunker(config)
    
    # Chunk document
    raw_chunks = chunker.chunk(doc.content)
    
    # Create embedder
    embedding_model = data.get('embedding_model', domain.embedding_model if domain else 'all-MiniLM-L6-v2')
    embedder = create_embedder(model_name=embedding_model)
    
    # Embed chunks
    texts = [c.content for c in raw_chunks]
    embed_result = await embedder.embed(texts)
    
    # Ensure collection exists
    collection_name = f"domain_{doc.domain_id}"
    if not await rag['vector_store'].collection_exists(collection_name):
        from modules.rag.vector_stores.base import CollectionConfig
        await rag['vector_store'].create_collection(CollectionConfig(
            name=collection_name,
            vector_size=embed_result.dimensions
        ))
    
    # Save to vector store
    from modules.rag.vector_stores.base import VectorRecord
    from modules.rag.document_manager import DocumentChunk
    
    saved_chunks = []
    records = []
    
    for i, (chunk, embedding) in enumerate(zip(raw_chunks, embed_result.embeddings)):
        chunk_id = str(uuid.uuid4())
        vector_id = f"{document_id}_{i}"
        
        doc_chunk = DocumentChunk(
            id=chunk_id,
            document_id=document_id,
            content=chunk.content,
            chunk_index=chunk.index,
            total_chunks=len(raw_chunks),
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            section_header=chunk.metadata.get('section_header'),
            vector_id=vector_id
        )
        saved_chunks.append(doc_chunk)
        
        records.append(VectorRecord(
            id=vector_id,
            vector=embedding,
            payload={
                'document_id': document_id,
                'domain_id': doc.domain_id,
                'content': chunk.content,
                'chunk_index': chunk.index,
                'total_chunks': len(raw_chunks),
                'document_name': doc.name
            }
        ))
    
    # Save chunks to database
    await rag['document_manager'].save_chunks(saved_chunks)
    
    # Upsert vectors
    upserted = await rag['vector_store'].upsert(collection_name, records)
    
    # Update document status
    from modules.rag.document_manager import DocumentStatus
    doc.status = DocumentStatus.READY
    doc.chunk_count = len(saved_chunks)
    await rag['document_manager'].update_document(doc)
    
    return {
        "status": "processed",
        "chunks_created": len(saved_chunks),
        "vectors_upserted": upserted
    }


# Vector Store / Collections
@app.get("/api/v1/rag/collections")
async def list_collections(request: Request):
    """List vector collections."""
    rag = get_rag_components(request)
    if not rag['vector_store']:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    collections = await rag['vector_store'].list_collections()
    return {"collections": collections}


@app.get("/api/v1/rag/collections/{name}")
async def get_collection_info(request: Request, name: str):
    """Get collection information."""
    rag = get_rag_components(request)
    if not rag['vector_store']:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    info = await rag['vector_store'].get_collection_info(name)
    if not info:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    return {
        "name": info.name,
        "vector_size": info.vector_size,
        "distance": info.distance.value,
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
        "status": info.status
    }


@app.delete("/api/v1/rag/collections/{name}")
async def delete_collection(request: Request, name: str):
    """Delete a collection."""
    rag = get_rag_components(request)
    if not rag['vector_store']:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    deleted = await rag['vector_store'].delete_collection(name)
    return {"status": "deleted" if deleted else "failed"}


# Retrieval
@app.post("/api/v1/rag/retrieve")
async def retrieve_documents(request: Request):
    """Retrieve relevant documents for a query."""
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    data = await request.json()
    query = data.get('query', '')
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    # Setup retriever
    embedding_model = data.get('embedding_model', 'all-MiniLM-L6-v2')
    embedder = create_embedder(model_name=embedding_model)
    
    # Determine collection
    domain_id = data.get('domain_id')
    collection_name = f"domain_{domain_id}" if domain_id else "documents"
    
    retriever = VectorRetriever(
        rag['vector_store'],
        embedder,
        rag['document_manager'],
        collection_name
    )
    
    config = RetrievalConfig(
        top_k=data.get('top_k', 10),
        score_threshold=data.get('score_threshold'),
        domain_ids=[domain_id] if domain_id else None
    )
    
    results = await retriever.retrieve(query, config)
    
    return {
        "query": query,
        "results": [r.to_dict() for r in results]
    }


@app.post("/api/v1/rag/retrieve/hybrid")
async def hybrid_retrieve(request: Request):
    """Hybrid retrieval combining dense and sparse search."""
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    data = await request.json()
    query = data.get('query', '')
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    embedding_model = data.get('embedding_model', 'all-MiniLM-L6-v2')
    embedder = create_embedder(model_name=embedding_model)

    domain_id = data.get('domain_id')
    collection_name = f"domain_{domain_id}" if domain_id else "documents"

    retriever = HybridRetriever(
        rag['vector_store'],
        embedder,
        rag['document_manager'],
        collection_name
    )
    
    config = RetrievalConfig(
        top_k=data.get('top_k', 10),
        alpha=data.get('alpha', 0.5),  # Dense vs sparse weight
        domain_ids=[domain_id] if domain_id else None
    )
    
    results = await retriever.retrieve(query, config)
    
    return {
        "query": query,
        "retrieval_method": "hybrid",
        "alpha": config.alpha,
        "results": [r.to_dict() for r in results]
    }


# GraphRAG Endpoints
@app.get("/api/v1/rag/graph/entities")
async def list_entities(
    request: Request,
    entity_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List knowledge graph entities."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    from modules.rag.graph_rag import EntityType
    type_enum = EntityType(entity_type) if entity_type else None
    
    entities, total = await rag['graph_rag'].list_entities(
        entity_type=type_enum,
        search=search,
        limit=limit,
        offset=offset
    )
    
    return {
        "entities": [e.to_dict() for e in entities],
        "total": total
    }


@app.post("/api/v1/rag/graph/entities")
async def create_entity(request: Request):
    """Create a new entity."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    data = await request.json()
    from modules.rag.graph_rag import Entity, EntityType
    
    entity = Entity(
        id=str(uuid.uuid4()),
        name=data.get('name', ''),
        entity_type=EntityType(data.get('entity_type', 'concept')),
        description=data.get('description', ''),
        aliases=data.get('aliases', []),
        properties=data.get('properties', {})
    )
    
    result = await rag['graph_rag'].create_entity(entity)
    return {"entity": result.to_dict()}


@app.put("/api/v1/rag/graph/entities/{entity_id}")
async def update_entity(request: Request, entity_id: str):
    """Update an entity."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    entity = await rag['graph_rag'].get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    data = await request.json()
    from modules.rag.graph_rag import EntityType
    
    entity.name = data.get('name', entity.name)
    entity.description = data.get('description', entity.description)
    entity.aliases = data.get('aliases', entity.aliases)
    entity.properties = data.get('properties', entity.properties)
    if 'entity_type' in data:
        entity.entity_type = EntityType(data['entity_type'])
    
    result = await rag['graph_rag'].update_entity(entity)
    return {"entity": result.to_dict()}


@app.delete("/api/v1/rag/graph/entities/{entity_id}")
async def delete_entity(request: Request, entity_id: str):
    """Delete an entity."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    await rag['graph_rag'].delete_entity(entity_id)
    return {"status": "deleted"}


@app.post("/api/v1/rag/graph/entities/merge")
async def merge_entities(request: Request):
    """Merge multiple entities into one."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    data = await request.json()
    entity_ids = data.get('entity_ids', [])
    merged_name = data.get('merged_name', '')
    
    if len(entity_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 entities to merge")
    
    result = await rag['graph_rag'].merge_entities(entity_ids, merged_name)
    if not result:
        raise HTTPException(status_code=400, detail="Merge failed")
    
    return {"entity": result.to_dict()}


@app.get("/api/v1/rag/graph/relationships")
async def get_entity_relationships(request: Request, entity_id: str, direction: str = "both"):
    """Get relationships for an entity."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    relationships = await rag['graph_rag'].get_relationships(entity_id, direction)
    return {"relationships": [r.to_dict() for r in relationships]}


@app.post("/api/v1/rag/graph/relationships")
async def create_relationship(request: Request):
    """Create a new relationship."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    data = await request.json()
    from modules.rag.graph_rag import Relationship, RelationshipType
    
    rel = Relationship(
        id=str(uuid.uuid4()),
        source_id=data.get('source_id'),
        target_id=data.get('target_id'),
        relationship_type=RelationshipType(data.get('relationship_type', 'related_to')),
        description=data.get('description', ''),
        bidirectional=data.get('bidirectional', False)
    )
    
    result = await rag['graph_rag'].create_relationship(rel)
    return {"relationship": result.to_dict()}


@app.delete("/api/v1/rag/graph/relationships/{relationship_id}")
async def delete_relationship(request: Request, relationship_id: str):
    """Delete a relationship."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    await rag['graph_rag'].delete_relationship(relationship_id)
    return {"status": "deleted"}


@app.get("/api/v1/rag/graph/visualize")
async def get_graph_visualization(request: Request, limit: int = 500):
    """Get graph data for visualization."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    data = await rag['graph_rag'].get_visualization_data(limit)
    return data


@app.get("/api/v1/rag/graph/subgraph")
async def get_subgraph(request: Request, entity_ids: str, depth: int = 2):
    """Get subgraph around specified entities."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    ids = entity_ids.split(',')
    nodes, edges = await rag['graph_rag'].get_subgraph(ids, depth)
    
    return {
        "nodes": [{"id": n.id, "label": n.label, "type": n.type} for n in nodes],
        "edges": [{"source": e.source, "target": e.target, "label": e.label} for e in edges]
    }


@app.post("/api/v1/rag/graph/extract")
async def extract_entities_from_text(request: Request):
    """Extract entities and relationships from text."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    data = await request.json()
    text = data.get('text', '')
    save = data.get('save', False)
    
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Extract entities
    entities = await rag['graph_rag'].extract_entities_from_text(text)
    
    # Extract relationships
    relationships = await rag['graph_rag'].extract_relationships_from_text(text, entities)
    
    # Save if requested
    if save:
        for entity in entities:
            existing = await rag['graph_rag'].find_entity_by_name(entity.name)
            if not existing:
                await rag['graph_rag'].create_entity(entity)
        
        for rel in relationships:
            await rag['graph_rag'].create_relationship(rel)
    
    return {
        "entities": [e.to_dict() for e in entities],
        "relationships": [r.to_dict() for r in relationships],
        "saved": save
    }


@app.get("/api/v1/rag/graph/statistics")
async def get_graph_statistics(request: Request):
    """Get graph statistics."""
    rag = get_rag_components(request)
    if not rag['graph_rag']:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    stats = await rag['graph_rag'].get_statistics()
    return stats


# Document Discovery Endpoints
@app.post("/api/v1/rag/discover/search")
async def discover_documents(request: Request):
    """Search web for documents."""
    rag = get_rag_components(request)
    if not rag['document_discovery']:
        raise HTTPException(status_code=503, detail="Document discovery not initialized")
    
    data = await request.json()
    query = data.get('query', '')
    max_results = data.get('max_results', 10)
    provider = data.get('provider', 'duckduckgo')
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    results = await rag['document_discovery'].search_web(query, max_results, provider)
    return {
        "query": query,
        "results": [r.to_dict() for r in results]
    }


@app.get("/api/v1/rag/discover/queue")
async def get_discovery_queue(
    request: Request,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get document review queue."""
    rag = get_rag_components(request)
    if not rag['document_discovery']:
        raise HTTPException(status_code=503, detail="Document discovery not initialized")
    
    from modules.rag.discovery import DiscoveryStatus
    status_enum = DiscoveryStatus(status) if status else None
    
    docs, total = await rag['document_discovery'].get_review_queue(
        status=status_enum,
        limit=limit,
        offset=offset
    )
    
    return {
        "documents": [d.to_dict() for d in docs],
        "total": total
    }


@app.post("/api/v1/rag/discover/{doc_id}/extract")
async def extract_discovered_content(request: Request, doc_id: str):
    """Extract content from discovered document URL."""
    rag = get_rag_components(request)
    if not rag['document_discovery']:
        raise HTTPException(status_code=503, detail="Document discovery not initialized")
    
    result = await rag['document_discovery'].extract_content(doc_id)
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"document": result.to_dict()}


@app.post("/api/v1/rag/discover/{doc_id}/approve")
async def approve_discovered_document(request: Request, doc_id: str):
    """Approve a discovered document."""
    rag = get_rag_components(request)
    if not rag['document_discovery']:
        raise HTTPException(status_code=503, detail="Document discovery not initialized")
    
    data = await request.json()
    domain_id = data.get('domain_id')
    
    result = await rag['document_discovery'].approve_document(doc_id, domain_id)
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"document": result.to_dict()}


@app.post("/api/v1/rag/discover/{doc_id}/reject")
async def reject_discovered_document(request: Request, doc_id: str):
    """Reject a discovered document."""
    rag = get_rag_components(request)
    if not rag['document_discovery']:
        raise HTTPException(status_code=503, detail="Document discovery not initialized")
    
    result = await rag['document_discovery'].reject_document(doc_id)
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"document": result.to_dict()}


@app.post("/api/v1/rag/discover/bulk-approve")
async def bulk_approve_documents(request: Request):
    """Bulk approve documents."""
    rag = get_rag_components(request)
    if not rag['document_discovery']:
        raise HTTPException(status_code=503, detail="Document discovery not initialized")
    
    data = await request.json()
    doc_ids = data.get('doc_ids', [])
    domain_id = data.get('domain_id')
    
    count = await rag['document_discovery'].bulk_approve(doc_ids, domain_id)
    return {"approved_count": count}


@app.get("/api/v1/rag/discover/statistics")
async def get_discovery_statistics(request: Request):
    """Get discovery statistics."""
    rag = get_rag_components(request)
    if not rag['document_discovery']:
        raise HTTPException(status_code=503, detail="Document discovery not initialized")
    
    stats = await rag['document_discovery'].get_statistics()
    return stats


# Batch Processing & Vector Store Management
@app.post("/api/v1/rag/documents/batch-process")
async def batch_process_documents(request: Request, background_tasks: BackgroundTasks):
    """Process multiple documents in batch."""
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    data = await request.json()
    document_ids = data.get('document_ids', [])
    
    if not document_ids:
        raise HTTPException(status_code=400, detail="No document IDs provided")
    
    # Validate all documents exist
    missing = []
    for doc_id in document_ids:
        doc = await rag['document_manager'].get_document(doc_id)
        if not doc:
            missing.append(doc_id)
    
    if missing:
        raise HTTPException(status_code=404, detail=f"Documents not found: {', '.join(missing)}")
    
    # Queue all documents for processing
    for doc_id in document_ids:
        background_tasks.add_task(
            process_document_background,
            doc_id,
            data.get('chunking_strategy', 'semantic'),
            data.get('chunk_size'),
            data.get('chunk_overlap'),
            data.get('embedding_model')
        )
    
    return {
        "status": "queued",
        "document_count": len(document_ids),
        "document_ids": document_ids,
        "message": f"{len(document_ids)} documents queued for processing"
    }


@app.post("/api/v1/rag/documents/process-all-pending")
async def process_all_pending(request: Request, background_tasks: BackgroundTasks):
    """Process all pending documents in a domain or globally."""
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    data = await request.json()
    domain_id = data.get('domain_id')
    
    from modules.rag.document_manager import DocumentStatus
    
    # Get all pending documents
    docs, total = await rag['document_manager'].list_documents(
        domain_id=domain_id,
        status=DocumentStatus.PENDING,
        limit=1000  # Process up to 1000 docs
    )
    
    if not docs:
        return {
            "status": "complete",
            "document_count": 0,
            "message": "No pending documents found"
        }
    
    # Queue all for processing
    for doc in docs:
        background_tasks.add_task(
            process_document_background,
            doc.id,
            data.get('chunking_strategy', 'semantic'),
            data.get('chunk_size'),
            data.get('chunk_overlap'),
            data.get('embedding_model')
        )
    
    return {
        "status": "queued",
        "document_count": len(docs),
        "message": f"{len(docs)} pending documents queued for processing"
    }


@app.post("/api/v1/rag/documents/{document_id}/reprocess")
async def reprocess_document(request: Request, document_id: str, background_tasks: BackgroundTasks):
    """Reprocess an existing document (delete old vectors and reprocess)."""
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    doc = await rag['document_manager'].get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    data = await request.json()
    
    # Delete existing vectors from vector store
    collection_name = f"domain_{doc.domain_id}"
    if await rag['vector_store'].collection_exists(collection_name):
        # Find and delete all vectors for this document
        chunks = await rag['document_manager'].get_chunks(document_id)
        vector_ids = [chunk.vector_id for chunk in chunks if chunk.vector_id]
        if vector_ids:
            try:
                await rag['vector_store'].delete_vectors(collection_name, vector_ids)
                logger.info(f"Deleted {len(vector_ids)} vectors for document {document_id}")
            except Exception as e:
                logger.warning(f"Error deleting vectors: {e}")
    
    # Delete old chunks
    await rag['document_manager'].delete_chunks(document_id)
    
    # Reset document status
    from modules.rag.document_manager import DocumentStatus
    doc.status = DocumentStatus.PENDING
    doc.chunk_count = 0
    doc.processed_at = None
    doc.error_message = None
    await rag['document_manager'].update_document(doc)
    
    # Queue for reprocessing
    background_tasks.add_task(
        process_document_background,
        document_id,
        data.get('chunking_strategy', 'semantic'),
        data.get('chunk_size'),
        data.get('chunk_overlap'),
        data.get('embedding_model')
    )
    
    return {
        "status": "queued",
        "document_id": document_id,
        "message": "Document queued for reprocessing"
    }


@app.delete("/api/v1/rag/documents/{document_id}/vectors")
async def remove_document_from_vector_store(request: Request, document_id: str):
    """Remove a document's vectors from the vector store (but keep document and chunks in DB)."""
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    doc = await rag['document_manager'].get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete vectors from vector store
    collection_name = f"domain_{doc.domain_id}"
    chunks = await rag['document_manager'].get_chunks(document_id)
    vector_ids = [chunk.vector_id for chunk in chunks if chunk.vector_id]
    
    deleted_count = 0
    if vector_ids and await rag['vector_store'].collection_exists(collection_name):
        try:
            await rag['vector_store'].delete_vectors(collection_name, vector_ids)
            deleted_count = len(vector_ids)
            logger.info(f"Deleted {deleted_count} vectors for document {document_id}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting vectors: {str(e)}")
    
    return {
        "status": "deleted",
        "document_id": document_id,
        "vectors_deleted": deleted_count
    }


@app.post("/api/v1/rag/documents/{document_id}/add-to-vector-store")
async def add_document_to_vector_store(request: Request, document_id: str):
    """Add a document's chunks to vector store (if chunks exist but not in vector store)."""
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    doc = await rag['document_manager'].get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    chunks = await rag['document_manager'].get_chunks(document_id)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document has no chunks. Process document first.")
    
    # Re-embed and add to vector store
    data = await request.json()
    domain = await rag['document_manager'].get_domain(doc.domain_id)
    
    embedding_model = data.get('embedding_model', domain.embedding_model if domain else 'all-MiniLM-L6-v2')
    embedder = create_embedder(model_name=embedding_model)
    
    # Embed all chunks
    texts = [chunk.content for chunk in chunks]
    embed_result = await embedder.embed(texts)
    
    # Ensure collection exists
    collection_name = f"domain_{doc.domain_id}"
    if not await rag['vector_store'].collection_exists(collection_name):
        from modules.rag.vector_stores.base import CollectionConfig
        collection_config = CollectionConfig(
            name=collection_name,
            dimension=embed_result.dimensions,
            distance_metric='cosine'
        )
        await rag['vector_store'].create_collection(collection_config)
    
    # Create vector records
    from modules.rag.vector_stores.base import VectorRecord
    records = []
    for chunk, embedding in zip(chunks, embed_result.embeddings):
        records.append(VectorRecord(
            id=chunk.vector_id or f"{document_id}_{chunk.chunk_index}",
            vector=embedding,
            payload={
                'document_id': document_id,
                'domain_id': doc.domain_id,
                'content': chunk.content,
                'chunk_index': chunk.chunk_index,
                'chunk_id': chunk.id,
                'metadata': doc.metadata
            }
        ))
    
    # Add to vector store
    await rag['vector_store'].add_vectors(collection_name, records)
    
    return {
        "status": "added",
        "document_id": document_id,
        "vectors_added": len(records),
        "collection": collection_name
    }


@app.get("/api/v1/rag/vector-stores/collections")
async def list_collections(request: Request):
    """List all vector store collections."""
    rag = get_rag_components(request)
    if not rag['vector_store']:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    collections = await rag['vector_store'].list_collections()
    return {"collections": collections}


@app.get("/api/v1/rag/vector-stores/collections/{collection_name}/stats")
async def get_collection_stats(request: Request, collection_name: str):
    """Get statistics for a vector store collection."""
    rag = get_rag_components(request)
    if not rag['vector_store']:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    if not await rag['vector_store'].collection_exists(collection_name):
        raise HTTPException(status_code=404, detail="Collection not found")
    
    stats = await rag['vector_store'].get_collection_stats(collection_name)
    return {"collection": collection_name, "stats": stats}


@app.delete("/api/v1/rag/vector-stores/collections/{collection_name}")
async def delete_collection(request: Request, collection_name: str):
    """Delete an entire vector store collection."""
    rag = get_rag_components(request)
    if not rag['vector_store']:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    if not await rag['vector_store'].collection_exists(collection_name):
        raise HTTPException(status_code=404, detail="Collection not found")
    
    await rag['vector_store'].delete_collection(collection_name)
    return {"status": "deleted", "collection": collection_name}


@app.post("/api/v1/rag/domains/{domain_id}/reindex")
async def reindex_domain(request: Request, domain_id: str, background_tasks: BackgroundTasks):
    """Reindex all documents in a domain."""
    rag = get_rag_components(request)
    if not rag['document_manager'] or not rag['vector_store']:
        raise HTTPException(status_code=503, detail="RAG components not initialized")
    
    domain = await rag['document_manager'].get_domain(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail="Domain not found")
    
    data = await request.json()
    
    # Get all documents in domain
    from modules.rag.document_manager import DocumentStatus
    docs, total = await rag['document_manager'].list_documents(
        domain_id=domain_id,
        limit=10000
    )
    
    if not docs:
        return {
            "status": "complete",
            "document_count": 0,
            "message": "No documents found in domain"
        }
    
    # Delete old collection if requested
    if data.get('recreate_collection', False):
        collection_name = f"domain_{domain_id}"
        if await rag['vector_store'].collection_exists(collection_name):
            await rag['vector_store'].delete_collection(collection_name)
            logger.info(f"Deleted collection {collection_name} for reindexing")
    
    # Queue all documents for reprocessing
    for doc in docs:
        if doc.status != DocumentStatus.PENDING:  # Only reprocess already processed docs
            background_tasks.add_task(
                process_document_background,
                doc.id,
                data.get('chunking_strategy', 'semantic'),
                data.get('chunk_size'),
                data.get('chunk_overlap'),
                data.get('embedding_model')
            )
    
    return {
        "status": "queued",
        "domain_id": domain_id,
        "document_count": len([d for d in docs if d.status != DocumentStatus.PENDING]),
        "message": f"Domain reindex started for {len(docs)} documents"
    }


# Embedding Models
@app.get("/api/v1/rag/embeddings/models")
async def list_embedding_models():
    """List available embedding models."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG system not available")

    local_models = LocalEmbedder.list_available_models()
    api_models = APIEmbedder.list_available_models()

    return {
        "local_models": [{"name": m.name, "dimensions": m.dimensions, "description": m.description} for m in local_models],
        "api_models": [{"name": m.name, "dimensions": m.dimensions, "provider": m.provider, "description": m.description} for m in api_models]
    }

@app.get("/api/v1/rag/embeddings/config")
async def get_embedding_config():
    """Get current embedding configuration for RAG."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    return {
        "use_deployed_service": USE_DEPLOYED_EMBEDDINGS,
        "service_url": EMBEDDING_SERVICE_URL,
        "default_model": DEFAULT_EMBEDDING_MODEL,
        "service_running": embedding_manager.is_running(),
        "service_status": embedding_manager.get_status() if embedding_manager.is_running() else None
    }

@app.put("/api/v1/rag/embeddings/config")
async def update_embedding_rag_config(request: Request):
    """Update embedding configuration for RAG."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    data = await request.json()
    
    global USE_DEPLOYED_EMBEDDINGS, EMBEDDING_SERVICE_URL, DEFAULT_EMBEDDING_MODEL
    
    if 'use_deployed_service' in data:
        USE_DEPLOYED_EMBEDDINGS = data['use_deployed_service']
        logger.info(f"Updated USE_DEPLOYED_EMBEDDINGS to: {USE_DEPLOYED_EMBEDDINGS}")
    
    if 'service_url' in data:
        EMBEDDING_SERVICE_URL = data['service_url']
        logger.info(f"Updated EMBEDDING_SERVICE_URL to: {EMBEDDING_SERVICE_URL}")
    
    if 'default_model' in data:
        DEFAULT_EMBEDDING_MODEL = data['default_model']
        logger.info(f"Updated DEFAULT_EMBEDDING_MODEL to: {DEFAULT_EMBEDDING_MODEL}")
    
    return {
        "success": True,
        "config": {
            "use_deployed_service": USE_DEPLOYED_EMBEDDINGS,
            "service_url": EMBEDDING_SERVICE_URL,
            "default_model": DEFAULT_EMBEDDING_MODEL,
            "service_running": embedding_manager.is_running()
        }
    }


@app.post("/api/v1/rag/embeddings/embed")
async def embed_text(request: Request):
    """Embed text using specified model."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    data = await request.json()
    texts = data.get('texts', [])
    model = data.get('model', 'all-MiniLM-L6-v2')
    
    if not texts:
        raise HTTPException(status_code=400, detail="Texts are required")

    embedder = create_embedder(model_name=model)
    result = await embedder.embed(texts)
    
    return {
        "model": result.model,
        "dimensions": result.dimensions,
        "embeddings_count": len(result.embeddings),
        "processing_time_ms": result.processing_time_ms
    }


# RAG Statistics
@app.get("/api/v1/rag/processing/queue")
async def get_processing_queue(request: Request):
    """Get documents currently being processed or queued."""
    rag = get_rag_components(request)
    if not rag['document_manager']:
        raise HTTPException(status_code=503, detail="Document manager not initialized")
    
    from modules.rag.document_manager import DocumentStatus
    
    # Get processing and pending documents
    processing_docs, proc_total = await rag['document_manager'].list_documents(
        status=DocumentStatus.PROCESSING,
        limit=100
    )
    
    pending_docs, pend_total = await rag['document_manager'].list_documents(
        status=DocumentStatus.PENDING,
        limit=100
    )
    
    error_docs, err_total = await rag['document_manager'].list_documents(
        status=DocumentStatus.ERROR,
        limit=100
    )
    
    return {
        "processing": {
            "documents": [d.to_dict() for d in processing_docs],
            "count": proc_total
        },
        "pending": {
            "documents": [d.to_dict() for d in pending_docs],
            "count": pend_total
        },
        "errors": {
            "documents": [d.to_dict() for d in error_docs],
            "count": err_total
        }
    }


@app.get("/api/v1/rag/statistics")
async def get_rag_statistics(request: Request):
    """Get overall RAG system statistics."""
    rag = get_rag_components(request)

    stats = {"available": RAG_AVAILABLE}
    
    if rag['document_manager']:
        stats['documents'] = await rag['document_manager'].get_statistics()
    
    if rag['vector_store']:
        collections = await rag['vector_store'].list_collections()
        collection_stats = []
        for coll in collections:
            try:
                coll_stats = await rag['vector_store'].get_collection_stats(coll)
                collection_stats.append({
                    "name": coll,
                    "stats": coll_stats
                })
            except:
                pass
        
        stats['vector_store'] = {
            "connected": True,
            "collections": len(collections),
            "collection_details": collection_stats
        }
    else:
        stats['vector_store'] = {"connected": False}
    
    if rag['graph_rag']:
        stats['graph'] = await rag['graph_rag'].get_statistics()
    
    if rag['document_discovery']:
        stats['discovery'] = await rag['document_discovery'].get_statistics()
    
    # Add embedding service status
    stats['embedding_service'] = {
        "running": embedding_manager.is_running(),
        "config": embedding_manager.config if embedding_manager.is_running() else None
    }

    return stats


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
