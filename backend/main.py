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

print("DEBUG: main.py loaded and imports completed")
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

# Workflow module imports
try:
    from modules.workflow import (
        WorkflowStorage,
        WorkflowEngine,
        Workflow,
        WorkflowNode,
        WorkflowConnection,
        WorkflowExecution,
        ExecutionStatus,
    )
    from modules.workflow.models import WorkflowCreate, WorkflowUpdate
    from modules.workflow.executors import NODE_EXECUTORS
    WORKFLOW_AVAILABLE = True
    logger.info("Workflow module loaded successfully")
except ImportError as e:
    WORKFLOW_AVAILABLE = False
    WorkflowStorage = None  # type: ignore
    WorkflowEngine = None  # type: ignore
    logger.warning(f"Workflow module not available: {e}")

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

# Route module imports
try:
    from routes import (
        rag_router, graphrag_router, workflows_router,
        conversations_router, registry_router, prompts_router,
        benchmark_router,
        batch_router,
        models_router,
        templates_router,
        tokens_router,
        service_router,
        stt_router,
        streaming_stt_router,
        tts_router,
        tools_router,
        finetuning_router,
        quantization_router,
        reddit_router,
        mcp_router,
    )
    from routes.rag import init_rag_config
    ROUTES_AVAILABLE = True
    logger.info("Route modules loaded successfully")
except ImportError as e:
    ROUTES_AVAILABLE = False
    rag_router = None  # type: ignore
    graphrag_router = None  # type: ignore
    workflows_router = None  # type: ignore
    conversations_router = None  # type: ignore
    registry_router = None  # type: ignore
    prompts_router = None  # type: ignore
    benchmark_router = None  # type: ignore
    batch_router = None  # type: ignore
    models_router = None  # type: ignore
    templates_router = None  # type: ignore
    tokens_router = None  # type: ignore
    service_router = None  # type: ignore
    stt_router = None  # type: ignore
    streaming_stt_router = None  # type: ignore
    tts_router = None  # type: ignore
    tools_router = None  # type: ignore
    finetuning_router = None  # type: ignore
    quantization_router = None  # type: ignore
    reddit_router = None  # type: ignore
    mcp_router = None  # type: ignore
    logger.warning(f"Route modules not available: {e}")

# MCP (Model Context Protocol) imports
try:
    from modules.mcp import MCPClientManager, MCPConfigStore
    MCP_AVAILABLE = True
    logger.info("MCP module loaded successfully")
except ImportError as e:
    MCP_AVAILABLE = False
    MCPClientManager = None  # type: ignore
    MCPConfigStore = None  # type: ignore
    logger.warning(f"MCP module not available: {e}")

# RAG Embedding Configuration
USE_DEPLOYED_EMBEDDINGS = os.getenv("USE_DEPLOYED_EMBEDDINGS", "false").lower() == "true"
# Use Docker network name for inter-container communication
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://llamacpp-embed:8080/v1")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "nomic-embed-text-v1.5")

# GraphRAG Service Configuration (external graphrag microservice)
# In Docker: http://graphrag-api-1:8000, on host: http://localhost:18000
GRAPHRAG_URL = os.getenv("GRAPHRAG_URL", "http://graphrag-api-1:8000")
GRAPHRAG_ENABLED = os.getenv("GRAPHRAG_ENABLED", "true").lower() == "true"

# Embedder cache - keeps models warm between requests to avoid cold start penalty
_embedder_cache: Dict[str, Any] = {}


def create_embedder(model_name: Optional[str] = None, use_deployed: Optional[bool] = None):
    """
    Factory function to create or retrieve a cached embedder instance.
    
    Caches LocalEmbedder instances to avoid 4-5s cold start on each request.
    APIEmbedder instances are also cached for consistency.
    
    Args:
        model_name: Name of the embedding model to use
        use_deployed: Whether to use the deployed embedding service.
                     If None, uses the USE_DEPLOYED_EMBEDDINGS env var.
    
    Returns:
        An Embedder instance (LocalEmbedder or APIEmbedder)
    """
    global _embedder_cache
    
    if not RAG_AVAILABLE:
        raise RuntimeError("RAG system not available")
    
    # Model name aliases for backward compatibility
    EMBEDDING_MODEL_ALIASES = {
        "nomic-embed-text": "nomic-embed-text-v1.5",
        "bge-large": "bge-large-en-v1.5",
    }
    
    model_name = model_name or DEFAULT_EMBEDDING_MODEL
    # Resolve aliases (e.g., "nomic-embed-text" -> "nomic-embed-text-v1.5")
    model_name = EMBEDDING_MODEL_ALIASES.get(model_name, model_name)
    
    use_deployed = USE_DEPLOYED_EMBEDDINGS if use_deployed is None else use_deployed
    
    # Check if the requested model is one supported by the deployed service
    deployed_models = ["nomic-embed-text-v1.5", "nomic-embed-text", "e5-mistral-7b", "bge-m3", "gte-Qwen2-1.5B"]
    
    # Create cache key based on model and deployment mode
    use_api = use_deployed and model_name in deployed_models
    cache_key = f"{'api' if use_api else 'local'}:{model_name}"
    
    # Return cached embedder if available
    if cache_key in _embedder_cache:
        logger.debug(f"Using cached embedder for {cache_key}")
        return _embedder_cache[cache_key]
    
    # Create new embedder and cache it
    if use_api:
        logger.info(f"Creating and caching API embedder for model: {model_name} at {EMBEDDING_SERVICE_URL}")
        embedder = APIEmbedder(
            model_name=model_name,
            api_key="llamacpp-embed",  # Match the API key from embedding service
            base_url=EMBEDDING_SERVICE_URL,
            timeout=300  # Longer timeout for large documents
        )
    else:
        # Auto-detect CUDA - sentence-transformers will use GPU if available
        logger.info(f"Creating and caching local embedder for model: {model_name} (device=auto)")
        embedder = LocalEmbedder(model_name=model_name, device=None)  # None = auto-detect
    
    _embedder_cache[cache_key] = embedder
    return embedder

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
        
        # Reference to download manager for metadata access
        self.download_manager: Optional[ModelDownloadManager] = None
        
        # Defer container log reading to runtime callers to avoid loop issues during import
        
    def load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from environment or defaults"""
        return {
            "model": {
                "name": os.getenv("MODEL_NAME", "Qwen3-Coder-30B-A3B-Instruct"),
                "variant": os.getenv("MODEL_VARIANT", "Q4_K_M"),
                "mmproj": os.getenv("MMPROJ_FILE"),
                "context_size": int(os.getenv("CONTEXT_SIZE", "128000")),
                "gpu_layers": int(os.getenv("GPU_LAYERS", "999")),
                "n_cpu_moe": int(os.getenv("N_CPU_MOE", "0")),
            },
            "template": {
                "directory": os.getenv("TEMPLATE_DIR", "/home/llamacpp/templates"),
                "selected": os.getenv("CHAT_TEMPLATE", ""),
            },
            "sampling": {
                # Empty by default — llama-server uses its own built-in defaults
                # (and GGUF-embedded model defaults) when not explicitly set.
                # Users can set values via the UI or presets.
            },
            "performance": {
                "threads": int(os.getenv("THREADS", "-1")),
                "batch_size": int(os.getenv("BATCH_SIZE", "2048")),
                "ubatch_size": int(os.getenv("UBATCH_SIZE", "512")),
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
                "embedding": False,  # Disabled by default - causes crashes with some models
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
        add_param_if_set(cmd, "--lora-base", self.config["model"].get("lora_base"))
        
        # Resolve mmproj path if specified
        mmproj_file = self.config["model"].get("mmproj")
        if mmproj_file:
            mmproj_path = mmproj_file
            # If it's just a filename (no separator), check in models dir
            if os.sep not in mmproj_file:
                # Add check for models directory logic similar to model_path
                possible_mmproj_paths = [
                    f"/home/llamacpp/models/{mmproj_file}",
                    f"/home/llamacpp/models/{variant}/{mmproj_file}"
                ]
                
                # Try to find existing file
                found = False
                for path in possible_mmproj_paths:
                    if Path(path).exists():
                        mmproj_path = path
                        found = True
                        break
                
                # Default to standard location if not found
                if not found:
                    mmproj_path = f"/home/llamacpp/models/{mmproj_file}"
            
            add_param_if_set(cmd, "--mmproj", mmproj_path)
        
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
        
        # Get repository info from metadata if available
        model_name = self.config['model']['name']
        variant = self.config['model']['variant']
        model_repo = None
        
        if self.download_manager:
            model_repo = self.download_manager.get_model_repository(model_name, variant)
        
        env_vars = {
            "MODEL_NAME": model_name,
            "MODEL_VARIANT": variant,
        }
        
        # Only set MODEL_REPO if we have metadata for it
        if model_repo:
            env_vars["MODEL_REPO"] = model_repo
            logger.info(f"Using repository from metadata: {model_repo} for {model_name}-{variant}")
        else:
            logger.warning(f"No repository metadata found for {model_name}-{variant}, container will use fallback logic")
        
        # Build container config
        container_config = {
            "image": image_name,
            "name": self.container_name,
            "command": cmd,
            "detach": True,
            "auto_remove": False,
            "network": self.docker_network,
            "volumes": {
                "llamacpp-api_gpt_oss_models": {"bind": "/home/llamacpp/models", "mode": "rw"},
                host_templates_dir: {"bind": "/home/llamacpp/templates", "mode": "ro"}
            },
            "environment": env_vars,
            "shm_size": "16g",
            "ports": {"8080/tcp": 8600}
        }
        
        # Add GPU device requests for GPU mode
        if execution_mode == "gpu":
            env_vars["CUDA_VISIBLE_DEVICES"] = cuda_devices
            env_vars["NVIDIA_VISIBLE_DEVICES"] = cuda_devices
            container_config["device_requests"] = [
                docker.types.DeviceRequest(
                    driver='nvidia',
                    count=-1,
                    capabilities=[['gpu', 'compute', 'utility']]
                )
            ]
        
        # Run container with the command
        self.docker_container = docker_client.containers.run(**container_config)
        
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
        
        # Get repository info from metadata if available
        model_name = self.config['model']['name']
        variant = self.config['model']['variant']
        model_repo = None
        
        if self.download_manager:
            model_repo = self.download_manager.get_model_repository(model_name, variant)
        
        docker_cmd = [
            'docker', 'run', '-d',
            '--name', self.container_name,
            '--shm-size', '16g',
            '-p', '8600:8080',
            '--network', self.docker_network,
            '-v', 'llama-nexus_gpt_oss_models:/home/llamacpp/models',
            '-v', f'{host_templates_dir}:/home/llamacpp/templates:ro',
            '-e', f'MODEL_NAME={model_name}',
            '-e', f'MODEL_VARIANT={variant}',
        ]
        
        # Only set MODEL_REPO if we have metadata for it
        if model_repo:
            docker_cmd.extend(['-e', f'MODEL_REPO={model_repo}'])
            logger.info(f"Using repository from metadata: {model_repo} for {model_name}-{variant}")
        else:
            logger.warning(f"No repository metadata found for {model_name}-{variant}, container will use fallback logic")
        
        # Add runtime and CUDA env vars only for GPU mode
        if execution_mode == "gpu":
            docker_cmd.extend(['--runtime', 'nvidia'])
            docker_cmd.extend(['-e', f'NVIDIA_VISIBLE_DEVICES={cuda_devices}'])
            docker_cmd.extend(['-e', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility'])
        
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
        # Metadata directory for storing model source information
        self.metadata_dir = self.models_dir / ".metadata"
        self.metadata_dir.mkdir(exist_ok=True)

    def _derive_model_id(self, filename: str) -> str:
        # Strip extension
        base = filename[:-5] if filename.endswith('.gguf') else Path(filename).stem
        return base

    def _save_model_metadata(self, model_name: str, variant: str, repo_id: str, filename: str):
        """Save model metadata to track repository source."""
        metadata = {
            "model_name": model_name,
            "variant": variant,
            "repo_id": repo_id,
            "filename": filename,
            "downloaded_at": datetime.now().isoformat(),
            "source": "huggingface"
        }
        
        # Use model_name-variant as the metadata filename
        metadata_file = self.metadata_dir / f"{model_name}-{variant}.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata for {model_name}-{variant} from {repo_id}")
        except Exception as e:
            logger.warning(f"Failed to save metadata for {model_name}-{variant}: {e}")

    def _load_model_metadata(self, model_name: str, variant: str) -> Optional[Dict[str, Any]]:
        """Load model metadata if it exists."""
        metadata_file = self.metadata_dir / f"{model_name}-{variant}.json"
        try:
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata for {model_name}-{variant}: {e}")
        return None

    def get_model_repository(self, model_name: str, variant: str) -> Optional[str]:
        """Get the repository ID for a model, or None if unknown."""
        metadata = self._load_model_metadata(model_name, variant)
        return metadata.get("repo_id") if metadata else None

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
        
        # Scan for Transformers-style model directories (safetensors/bin with config.json)
        for config_file in self.models_dir.rglob('config.json'):
            try:
                # Skip if in .metadata, checkpoint, or cache directories
                config_path_str = str(config_file)
                if any(skip in config_path_str for skip in ['.metadata', 'checkpoint', '.cache', '__pycache__']):
                    continue
                    
                model_dir = config_file.parent
                
                # Check if directory contains model weights
                weight_files = list(model_dir.glob('*.safetensors')) + list(model_dir.glob('model*.bin'))
                if not weight_files:
                    continue
                    
                # Parse config.json for metadata
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Skip if this looks like an adapter config (LoRA)
                if 'base_model_name_or_path' in config or 'peft_type' in config:
                    continue
                
                # Extract parameters count
                parameters = self._extract_parameter_count(config)
                context_length = config.get('max_position_embeddings', 2048)
                
                # Get architecture type
                architectures = config.get('architectures', [])
                architecture = config.get('model_type', architectures[0] if architectures else 'unknown')
                
                # Calculate total size from weight files
                total_size = sum(wf.stat().st_size for wf in weight_files)
                latest_mtime = max(wf.stat().st_mtime for wf in weight_files)
                
                # Use directory name as model name
                model_name = model_dir.name
                
                # Determine variant from weight file type
                has_safetensors = any(wf.suffix == '.safetensors' for wf in weight_files)
                variant = "safetensors" if has_safetensors else "bin"
                
                # Check for training metadata (indicates fine-tuned model)
                training_metadata = None
                training_metadata_path = model_dir / 'training_metadata.json'
                if training_metadata_path.exists():
                    try:
                        with open(training_metadata_path, 'r') as f:
                            training_metadata = json.load(f)
                    except Exception as e:
                        logger.debug(f"Could not load training metadata: {e}")
                
                model_key = f"{model_name}:transformers"
                if model_key not in seen_models:
                    seen_models.add(model_key)
                    model_info = {
                        "name": model_name,
                        "variant": variant,
                        "size": total_size,
                        "status": "available",
                        "lastModified": datetime.fromtimestamp(latest_mtime).isoformat(),
                        "localPath": str(model_dir.relative_to(self.models_dir)),
                        "filename": model_name,
                        "framework": "transformers",
                        "parameters": parameters,
                        "contextLength": context_length,
                        "architecture": architecture
                    }
                    
                    # Include training metadata if this is a fine-tuned model
                    if training_metadata:
                        model_info["source"] = training_metadata.get("source", "trained")
                        model_info["trainingJobId"] = training_metadata.get("training_job_id")
                        model_info["baseModel"] = training_metadata.get("base_model")
                        model_info["mergedAt"] = training_metadata.get("merged_at")
                        model_info["finalLoss"] = training_metadata.get("final_loss")
                        model_info["totalSteps"] = training_metadata.get("total_steps")
                        model_info["loraConfig"] = training_metadata.get("lora_config")
                    
                    items.append(model_info)
            except Exception as e:
                logger.debug(f"Skipping invalid model directory {config_file.parent}: {e}")
                continue
                
        # Merge in any actively downloading items (only queued/downloading, not completed)
        async with self._lock:
            for rec in self._downloads.values():
                # Only show actively downloading items - completed ones should appear in local scan
                if rec.status not in ("queued", "downloading"):
                    continue
                
                # For Transformers models (safetensors/bin), use repo_id as model name
                is_transformers = any(rec.filename.lower().endswith(ext) for ext in ('.safetensors', '.bin', '.pth', '.pt'))
                
                if is_transformers:
                    # Use repo_id formatted as model name (same as subdirectory)
                    name = rec.repo_id.replace('/', '_')
                    variant = "safetensors" if rec.filename.lower().endswith('.safetensors') else "bin"
                else:
                    name, variant = self._parse_name_variant(rec.filename)
                    variant = variant or "unknown"
                
                model_key = f"{name}:{variant}"
                
                # Only add if not already present from local scan
                if model_key not in seen_models:
                    seen_models.add(model_key)
                    items.append({
                        "name": name,
                        "variant": variant,
                        "size": rec.total_size or 0,
                        "status": "downloading",
                        "downloadProgress": rec.progress,
                        "framework": "transformers" if is_transformers else "gguf"
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

    def _extract_parameter_count(self, config: Dict[str, Any]) -> str:
        """Extract or estimate parameter count from Transformers config.json."""
        # Check if explicitly provided
        if 'num_parameters' in config:
            return self._format_param_count(config['num_parameters'])
        
        # Estimate from architecture parameters
        hidden_size = config.get('hidden_size', 0)
        num_hidden_layers = config.get('num_hidden_layers', 0)
        vocab_size = config.get('vocab_size', 0)
        intermediate_size = config.get('intermediate_size', hidden_size * 4 if hidden_size else 0)
        
        if hidden_size and num_hidden_layers and vocab_size:
            # Rough estimate: embeddings + transformer layers
            embedding_params = vocab_size * hidden_size * 2  # input + output embeddings
            layer_params = num_hidden_layers * (
                4 * hidden_size * hidden_size +  # attention (Q, K, V, O)
                2 * hidden_size * intermediate_size +  # FFN
                4 * hidden_size  # layer norms
            )
            total_params = embedding_params + layer_params
            return self._format_param_count(total_params)
        
        return "?B"

    def _format_param_count(self, num_params: int) -> str:
        """Format parameter count as human-readable string."""
        if num_params >= 1e12:
            return f"{num_params/1e12:.1f}T"
        elif num_params >= 1e9:
            return f"{num_params/1e9:.1f}B"
        elif num_params >= 1e6:
            return f"{num_params/1e6:.0f}M"
        else:
            return f"{num_params/1e3:.0f}K"

    def _is_multipart_file(self, filename: str) -> bool:
        """Check if filename indicates a multi-part model file (GGUF or safetensors)"""
        import re
        # Pattern for multi-part GGUF: filename-00001-of-00002.gguf
        # Pattern for multi-part safetensors: model-00001-of-00002.safetensors
        return bool(re.search(r'-\d{5}-of-\d{5}\.(gguf|safetensors)$', filename))
    
    def _get_multipart_files(self, filename: str) -> List[str]:
        """Get all part filenames for a multi-part download (GGUF or safetensors)"""
        import re
        
        # Extract the base name, part info, and extension
        match = re.match(r'(.+)-(\d{5})-of-(\d{5})\.(gguf|safetensors)$', filename)
        if not match:
            return [filename]  # Not a multi-part file
        
        base_name = match.group(1)
        total_parts = int(match.group(3))
        extension = match.group(4)
        
        # Generate all part filenames
        part_files = []
        for i in range(1, total_parts + 1):
            part_filename = f"{base_name}-{i:05d}-of-{total_parts:05d}.{extension}"
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

    async def _download_transformers_metadata(self, repo_id: str, model_dir: Path, cancel_event: asyncio.Event):
        """Download config.json and other essential metadata files for Transformers models."""
        metadata_files = ['config.json', 'tokenizer_config.json', 'tokenizer.json']
        
        headers = {"User-Agent": "llama-nexus1.0"}
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            for meta_file in metadata_files:
                if cancel_event.is_set():
                    return
                try:
                    url = hf_hub_url(repo_id=repo_id, filename=meta_file)
                    response = await client.get(url, headers=headers)
                    if response.status_code == 200:
                        (model_dir / meta_file).write_bytes(response.content)
                        logger.info(f"Downloaded {meta_file} for {repo_id}")
                except Exception as e:
                    logger.debug(f"Could not download {meta_file} for {repo_id}: {e}")

    async def _run_download(self, model_id: str, repo_id: str, filename: str, dest_path: Path, cancel_event: asyncio.Event):
        # Check if this is a Transformers model (safetensors/bin) - needs special handling
        is_transformers_model = any(filename.lower().endswith(ext) for ext in ('.safetensors', '.bin', '.pth', '.pt')) and not filename.endswith('.gguf')
        
        if is_transformers_model:
            # Create a subdirectory for the model based on repo_id
            model_subdir = repo_id.replace('/', '_')
            model_dir = self.models_dir / model_subdir
            model_dir.mkdir(parents=True, exist_ok=True)
            dest_path = model_dir / filename
            
            # Download config.json and tokenizer files for metadata
            logger.info(f"Downloading Transformers model to {model_dir}")
            await self._download_transformers_metadata(repo_id, model_dir, cancel_event)
        
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
                
                # Save metadata for the downloaded model
                try:
                    model_name, variant = self._parse_name_variant(filename)
                    if model_name and variant:
                        self._save_model_metadata(model_name, variant, repo_id, filename)
                except Exception as e:
                    logger.warning(f"Failed to save metadata for {filename}: {e}")
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
        """Download all parts of a multi-part model file (GGUF or safetensors)."""
        start_time = time.time()
        
        # Check if this is a Transformers model (safetensors) - needs subdirectory
        first_file = part_files[0] if part_files else ""
        is_transformers_model = first_file.lower().endswith('.safetensors')
        
        if is_transformers_model:
            # Create a subdirectory for the model based on repo_id
            model_subdir = repo_id.replace('/', '_')
            model_dir = self.models_dir / model_subdir
            model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading multi-part Transformers model to {model_dir}")
            await self._download_transformers_metadata(repo_id, model_dir, cancel_event)
        else:
            model_dir = self.models_dir
        
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
                    temp_path = model_dir / f"{part_file}.part"
                    final_path = model_dir / part_file
                    
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
                
                # Save metadata for the downloaded model (use first part filename for parsing)
                try:
                    first_part = part_files[0]
                    model_name, variant = self._parse_name_variant(first_part)
                    if model_name and variant:
                        self._save_model_metadata(model_name, variant, repo_id, first_part)
                except Exception as e:
                    logger.warning(f"Failed to save metadata for multipart download: {e}")
                    
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


# Initialize managers
manager = LlamaCPPManager()
download_manager = ModelDownloadManager()
embedding_manager = EmbeddingManager()
stt_manager = STTManager()
streaming_stt_manager = StreamingSTTManager()
tts_manager = TTSManager()

# Wire up download manager reference
manager.download_manager = download_manager


def _merge_and_persist_config(new_config: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge config and persist to disk."""
    import copy
    base = copy.deepcopy(manager.config)
    
    # The new_config comes wrapped in a 'config' key, so extract it
    config_data = new_config.get('config', new_config)
    
    # Required fields that should never be removed
    # These match the required fields in the startup config loading
    required_fields = {
        "model": ["name", "variant", "context_size", "gpu_layers", "n_cpu_moe"],
        "server": ["host", "port", "api_key", "metrics"],
        "template": ["directory"],
        "execution": ["mode", "cuda_devices"],
        "performance": ["threads", "batch_size", "ubatch_size"]
    }
    
    for category in ["model", "sampling", "performance", "context_extension", "server", "template", "execution"]:
        if category in config_data:
            if category not in base:
                base[category] = {}
            for key, value in config_data[category].items():
                if value is None or value == "":
                    if category not in required_fields or key not in required_fields[category]:
                        base[category].pop(key, None)
                else:
                    base[category][key] = value
    
    manager.config = base
    
    # Persist
    config_file = Path("/tmp/llamacpp_config.json")
    with open(config_file, "w") as f:
        json.dump(base, f, indent=2)
    return base


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting LlamaCPP Management API")

    # Load saved config if exists and merge with defaults
    config_file = Path("/tmp/llamacpp_config.json")
    if config_file.exists():
        with open(config_file) as f:
            saved_config = json.load(f)
            # Merge saved config with default config to ensure ONLY required fields exist
            # Optional fields (like sampling parameters) should remain None/absent if cleared by user
            default_config = manager.load_default_config()
            
            # Define required fields that must always have values
            required_fields = {
                "model": ["name", "variant", "context_size", "gpu_layers", "n_cpu_moe"],
                "server": ["host", "port", "api_key", "metrics"],
                "template": ["directory"],
                "execution": ["mode", "cuda_devices"],
                "performance": ["threads", "batch_size", "ubatch_size"]
            }
            
            for category in default_config:
                if category in saved_config:
                    # Merge category, but only restore required fields if missing/None
                    for key, default_value in default_config[category].items():
                        if key not in saved_config[category] or saved_config[category][key] is None:
                            # Only add back if it's a required field
                            if category in required_fields and key in required_fields[category]:
                                saved_config[category][key] = default_value
                            # Otherwise, leave it as None/absent so llama-server uses its defaults
                else:
                    # Add missing category entirely, but filter to required fields only
                    if category in required_fields:
                        saved_config[category] = {
                            k: v for k, v in default_config[category].items()
                            if k in required_fields[category]
                        }
                    else:
                        # For categories without required fields, add empty dict
                        saved_config[category] = {}
            manager.config = saved_config

    # Initialize RAG system
    app.state.rag_available = RAG_AVAILABLE
    app.state.create_embedder = create_embedder
    app.state.embedding_manager = embedding_manager
    app.state.stt_manager = stt_manager
    app.state.streaming_stt_manager = streaming_stt_manager
    app.state.tts_manager = tts_manager
    
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

            # Set up background processing function reference (defined later in this file)
            # This will be set after app creation since the function needs app reference
            
            # Pre-warm the default embedding model to avoid cold start on first request
            # This loads the model into memory during startup (~4-5s) instead of first query
            try:
                logger.info(f"Pre-warming embedding model: {DEFAULT_EMBEDDING_MODEL}")
                warmup_embedder = create_embedder(model_name=DEFAULT_EMBEDDING_MODEL)
                # Trigger model load by embedding a dummy query
                await warmup_embedder.embed(["warmup query"])
                logger.info(f"Embedding model pre-warmed and cached: {DEFAULT_EMBEDDING_MODEL}")
            except Exception as warmup_error:
                logger.warning(f"Failed to pre-warm embedding model: {warmup_error}")
            
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            app.state.document_manager = None
            app.state.graph_rag = None
            app.state.vector_store = None
            app.state.document_discovery = None

    # Initialize Workflow system
    app.state.workflow_available = WORKFLOW_AVAILABLE
    if WORKFLOW_AVAILABLE:
        try:
            workflow_db_path = os.getenv("WORKFLOW_DB_PATH", "data/workflows.db")
            os.makedirs(os.path.dirname(workflow_db_path), exist_ok=True)

            app.state.workflow_storage = WorkflowStorage(workflow_db_path)
            app.state.workflow_engine = WorkflowEngine(app.state.workflow_storage)

            logger.info("Workflow system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Workflow system: {e}")
            app.state.workflow_storage = None
            app.state.workflow_engine = None

    # Set module instances on app.state for route access
    app.state.conversation_store = conversation_store
    app.state.model_registry = model_registry
    app.state.prompt_library = prompt_library
    app.state.benchmark_runner = benchmark_runner
    app.state.batch_processor = batch_processor
    app.state.manager = manager
    app.state.token_tracker = token_tracker

    # Initialize Reddit crawler scheduler
    try:
        from modules.finetuning.reddit_scheduler import get_reddit_scheduler
        app.state.reddit_scheduler = get_reddit_scheduler()
        if app.state.reddit_scheduler.config.enabled:
            await app.state.reddit_scheduler.start()
            logger.info("Reddit crawler scheduler started (auto-enabled)")
        else:
            logger.info("Reddit crawler scheduler initialized (disabled)")
    except Exception as e:
        logger.warning(f"Reddit crawler scheduler not available: {e}")
        app.state.reddit_scheduler = None

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

# Include route modules
if ROUTES_AVAILABLE:
    # Initialize RAG route configuration
    init_rag_config(
        rag_available=RAG_AVAILABLE,
        use_deployed=USE_DEPLOYED_EMBEDDINGS,
        embedding_url=EMBEDDING_SERVICE_URL,
        default_model=DEFAULT_EMBEDDING_MODEL,
        graphrag_url=GRAPHRAG_URL,
        graphrag_enabled=GRAPHRAG_ENABLED
    )
    
    # Include routers
    app.include_router(rag_router)
    app.include_router(graphrag_router)
    app.include_router(workflows_router)
    app.include_router(conversations_router)
    app.include_router(registry_router)
    app.include_router(prompts_router)
    app.include_router(benchmark_router)
    app.include_router(batch_router)
    app.include_router(models_router)
    app.include_router(templates_router)
    app.include_router(tokens_router)
    app.include_router(service_router)
    app.include_router(stt_router)
    app.include_router(streaming_stt_router)
    app.include_router(tts_router)
    app.include_router(tools_router)
    app.include_router(finetuning_router)
    app.include_router(quantization_router)
    if reddit_router:
        app.include_router(reddit_router)
    if mcp_router:
        app.include_router(mcp_router)
    logger.info("Route modules included: rag, graphrag, workflows, conversations, registry, prompts, benchmark, batch, models, templates, tokens, service, stt, tts, tools, finetuning, quantization, mcp")

# Initialize MCP Client Manager
if MCP_AVAILABLE:
    try:
        mcp_config_store = MCPConfigStore()
        mcp_client_manager = MCPClientManager(config_store=mcp_config_store)
        app.state.mcp_client_manager = mcp_client_manager
        logger.info("MCP Client Manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize MCP Client Manager: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mode": "docker" if manager.use_docker else "subprocess"
    }


# Service, Embedding routes moved to routes/service.py

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


# ============================================================================
# Training WebSocket Endpoint for Real-time Metrics
# ============================================================================
_training_ws_clients: List[WebSocket] = []

@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    """
    WebSocket endpoint for real-time training metrics.
    Streams training progress, loss curves, and GPU utilization.
    """
    await websocket.accept()
    _training_ws_clients.append(websocket)
    
    try:
        # Import here to avoid circular imports
        try:
            from modules.finetuning import register_ws_broadcaster, unregister_ws_broadcaster
        except ImportError:
            from finetuning import register_ws_broadcaster, unregister_ws_broadcaster
        
        # Capture the current running event loop for thread-safe scheduling
        main_loop = asyncio.get_running_loop()
        
        # Create async callback for broadcasting
        async def send_training_event(event_type: str, data: dict):
            try:
                await websocket.send_json({
                    "type": event_type,
                    "timestamp": datetime.now().isoformat(),
                    "payload": data
                })
            except Exception as e:
                logger.warning(f"Failed to send training event: {e}")
        
        # Thread-safe wrapper for the async callback - called from Redis consumer thread
        def sync_broadcast(event_type: str, data: dict):
            try:
                # Use run_coroutine_threadsafe to schedule from background thread
                future = asyncio.run_coroutine_threadsafe(
                    send_training_event(event_type, data),
                    main_loop
                )
                # Don't wait for result - fire and forget
            except Exception as e:
                logger.warning(f"Training broadcast error: {e}")
        
        # Register the callback
        register_ws_broadcaster(sync_broadcast)
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "timestamp": datetime.now().isoformat(),
            "data": {"endpoint": "training"}
        })
        
        # Keep connection alive and send GPU metrics periodically
        while True:
            await asyncio.sleep(2)  # Send GPU metrics every 2 seconds
            
            try:
                gpu_metrics = {}
                # Get GPU metrics
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                         "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0:
                        parts = [p.strip() for p in result.stdout.strip().split(",")]
                        if len(parts) >= 5:
                            gpu_metrics = {
                                "vram_used_gb": float(parts[0]) / 1024,
                                "vram_total_gb": float(parts[1]) / 1024,
                                "gpu_utilization": float(parts[2]),
                                "temperature_c": float(parts[3]),
                                "power_w": float(parts[4]) if parts[4] != "[N/A]" else None,
                            }
                except Exception:
                    pass  # GPU metrics not available
                
                await websocket.send_json({
                    "type": "gpu_metrics",
                    "timestamp": datetime.now().isoformat(),
                    "payload": gpu_metrics
                })
                
            except Exception as e:
                logger.warning(f"Error sending GPU metrics: {e}")
                
    except WebSocketDisconnect:
        if websocket in _training_ws_clients:
            _training_ws_clients.remove(websocket)
    except Exception as e:
        logger.error(f"Training WebSocket error: {e}")
        if websocket in _training_ws_clients:
            _training_ws_clients.remove(websocket)


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
                    # Handle multi-GPU systems: nvidia-smi returns one line per GPU
                    # Take only the first GPU (primary) for this endpoint
                    first_gpu_line = result.stdout.strip().split('\n')[0]
                    values = [v.strip() for v in first_gpu_line.split(',')]
                    if len(values) >= 5:
                        resources["gpu"] = {
                            "vram_used_mb": int(float(values[0])),
                            "vram_total_mb": int(float(values[1])),
                            "usage_percent": float(values[2]) if values[2] != "[N/A]" else 0,
                            "temperature_c": float(values[3]) if values[3] != "[N/A]" else 0,
                            "power_watts": float(values[4]) if values[4] not in ["N/A", "[N/A]"] else None
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

# /v1/models/current alias handled by routes/models.py

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

# /api/v1/service/action handled by routes/service.py

# --- Model management endpoints ---
app.state.download_manager = download_manager
app.state.merge_and_persist_config = _merge_and_persist_config

# Model, Template, Token routes moved to routes/models.py, routes/templates.py, routes/tokens.py


# BFCL Benchmark routes moved to routes/benchmark.py


# Conversation, Registry, Prompts, Benchmark, Batch routes moved to routes/

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
# RAG System - Background Processing Function (routes moved to routes/rag.py)
# =============================================================================
# Note: process_document_background is kept here because it accesses app.state directly


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
        
        from modules.rag.document_manager import DocumentStatus, DocumentType
        
        doc = await rag['document_manager'].get_document(document_id)
        if not doc:
            logger.error(f"Document {document_id} not found for processing")
            return
        
        # Update status to processing
        doc.status = DocumentStatus.PROCESSING
        await rag['document_manager'].update_document(doc)
        
        # Get domain settings
        domain = await rag['document_manager'].get_domain(doc.domain_id)
        
        # Check for GraphRAG options in metadata
        graphrag_options = doc.metadata.get('graphrag_options', {})
        use_semantic_chunking = graphrag_options.get('use_semantic_chunking', False)
        build_knowledge_graph = graphrag_options.get('build_knowledge_graph', False)
        
        # Try to use GraphRAG semantic chunking if requested
        graphrag_chunking_used = False
        raw_chunks = None
        
        if use_semantic_chunking:
            try:
                from modules.graphrag_wrapper import is_graphrag_available, GraphRAGWrapper
                if is_graphrag_available():
                    logger.info(f"Using GraphRAG semantic chunking for document {document_id}")
                    wrapper = GraphRAGWrapper()
                    graphrag_chunks = wrapper.semantic_chunk(doc.content)
                    
                    # Convert GraphRAG chunks to our ChunkResult format
                    from modules.rag.chunking import ChunkResult
                    raw_chunks = []
                    start_pos = 0
                    for i, chunk_text in enumerate(graphrag_chunks):
                        end_pos = start_pos + len(chunk_text)
                        raw_chunks.append(ChunkResult(
                            content=chunk_text,
                            index=i,
                            start_char=start_pos,
                            end_char=end_pos
                        ))
                        start_pos = end_pos + 1  # +1 for separator
                    graphrag_chunking_used = True
                    logger.info(f"GraphRAG created {len(raw_chunks)} semantic chunks")
                else:
                    logger.warning("GraphRAG requested but submodule not available, falling back to standard chunking")
            except Exception as e:
                logger.warning(f"GraphRAG semantic chunking failed: {e}, falling back to standard chunking")
        
        # Fallback to standard chunking if GraphRAG not used
        if raw_chunks is None:
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
            
            raw_chunks = chunker.chunk(doc.content)
        
        # Extract entities and build knowledge graph if requested
        if build_knowledge_graph:
            try:
                from modules.graphrag_wrapper import is_graphrag_available, GraphRAGWrapper
                if is_graphrag_available():
                    neo4j_uri = os.getenv('NEO4J_URI')
                    if neo4j_uri:
                        logger.info(f"Extracting entities for document {document_id}")
                        wrapper = GraphRAGWrapper(neo4j_uri=neo4j_uri)
                        extraction = wrapper.extract_entities(doc.content, domain.name if domain else 'general')
                        
                        if extraction.entities:
                            wrapper.build_knowledge_graph(
                                extraction.entities, 
                                extraction.relationships, 
                                domain.name if domain else 'general'
                            )
                            logger.info(f"Added {len(extraction.entities)} entities to knowledge graph")
                    else:
                        logger.warning("NEO4J_URI not configured, skipping knowledge graph building")
                else:
                    logger.warning("GraphRAG submodule not available, skipping knowledge graph building")
            except Exception as e:
                logger.warning(f"Knowledge graph building failed (non-fatal): {e}")
        
        # Create embedder - use domain's configured model if not overridden
        # KEY FIX: Use domain.embedding_model when embedding_model is None
        model_name = embedding_model or (domain.embedding_model if domain else 'nomic-embed-text-v1.5')
        logger.info(f"Using embedding model: {model_name} for document {document_id}")
        embedder = create_embedder(model_name=model_name)
        
        # ==========================================
        # VLM Visual Processing for PDF Documents
        # ==========================================
        visual_descriptions = []  # List of (page_num, image_path, description) tuples
        
        if doc.doc_type == DocumentType.PDF:
            try:
                from modules.rag.vlm_client import VLMClient, VLM_ENABLED
                from routes.rag import extract_pdf_images
                import os
                
                # Extract images from original PDF content
                # Get the original content (may be base64 or raw PDF)
                original_content = doc.metadata.get('original_pdf_content', doc.content)
                if original_content.startswith('%PDF') or len(original_content) > 1000:
                    extracted_images = extract_pdf_images(
                        original_content,
                        document_id,
                        output_dir="data/rag/images"
                    )
                    
                    if extracted_images:
                        logger.info(f"Found {len(extracted_images)} images in PDF")
                        
                        # Try VLM if enabled
                        vlm_available = False
                        vlm_client = None
                        if VLM_ENABLED:
                            try:
                                vlm_client = VLMClient()
                                # Quick connectivity check could go here
                                vlm_available = True
                            except Exception as e:
                                logger.warning(f"VLM client initialization failed: {e}")
                        
                        for page_num, image_path, image_type in extracted_images:
                            description = None
                            image_filename = os.path.basename(image_path)
                            
                            # Try VLM description first
                            if vlm_available and vlm_client:
                                try:
                                    description = await vlm_client.describe_image(image_path)
                                    if description:
                                        logger.info(f"VLM described image from page {page_num}: {len(description)} chars")
                                except Exception as e:
                                    logger.warning(f"VLM failed for {image_path}: {e}")
                                    description = None
                            
                            # Fallback: create placeholder with image filename reference
                            if not description:
                                description = (
                                    f"[Visual content from page {page_num}] "
                                    f"Image file: {image_filename}. "
                                    f"Document: {doc.name}. "
                                    f"This is an embedded graphic or chart that could not be automatically described. "
                                    f"Image path: {image_path}"
                                )
                                logger.info(f"Using placeholder for image {image_filename} (page {page_num})")
                            
                            visual_descriptions.append((page_num, image_path, description))
                        
                        vlm_count = sum(1 for _, _, d in visual_descriptions if not d.startswith('[Visual content'))
                        placeholder_count = len(visual_descriptions) - vlm_count
                        logger.info(f"Processed {len(visual_descriptions)} images: {vlm_count} VLM described, {placeholder_count} placeholders")
                        
            except ImportError as e:
                logger.warning(f"VLM/image processing unavailable: {e}")
            except Exception as e:
                logger.warning(f"Visual processing failed (continuing with text only): {e}")
        
        # Combine text chunks and visual descriptions for embedding
        all_texts = [c.content for c in raw_chunks]
        visual_texts = [desc for _, _, desc in visual_descriptions]
        all_texts.extend(visual_texts)
        
        # Embed all chunks (text + visual descriptions)
        logger.info(f"Embedding {len(all_texts)} items ({len(raw_chunks)} text chunks + {len(visual_descriptions)} visuals) for document {document_id}")
        
        # Process in smaller batches for large documents
        batch_size = 4 if USE_DEPLOYED_EMBEDDINGS else 32
        embed_result = await embedder.embed(all_texts, batch_size=batch_size, show_progress=False)
        
        # Ensure collection exists
        collection_name = f"domain_{doc.domain_id}"
        if not await rag['vector_store'].collection_exists(collection_name):
            from modules.rag.vector_stores.base import CollectionConfig
            collection_config = CollectionConfig(
                name=collection_name,
                vector_size=embed_result.dimensions
            )
            await rag['vector_store'].create_collection(collection_config)
        
        # Store in vector store and create chunks
        from modules.rag.document_manager import DocumentChunk
        from modules.rag.vector_stores.base import VectorRecord
        
        chunks = []
        records = []
        
        total_chunks = len(raw_chunks) + len(visual_descriptions)
        
        # Process text chunks
        for i, (chunk, embedding) in enumerate(zip(raw_chunks, embed_result.embeddings[:len(raw_chunks)])):
            chunk_id = str(uuid.uuid4())
            vector_id = str(uuid.uuid4())
            
            doc_chunk = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                content=chunk.content,
                chunk_index=chunk.index,
                total_chunks=total_chunks,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                token_count=len(chunk.content.split()),  # Approximate token count
                chunk_type="text",
                vector_id=vector_id,
                metadata={
                    "document_name": doc.name,
                    "source_path": doc.source_path,
                    "chunk_index": chunk.index,
                    "total_chunks": total_chunks,
                    **doc.metadata
                }
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
                    'chunk_type': 'text',
                    'document_name': doc.name,
                    'source_path': doc.source_path,
                    'metadata': doc.metadata
                }
            ))
        
        # Process visual chunks
        for i, ((page_num, image_path, description), embedding) in enumerate(
            zip(visual_descriptions, embed_result.embeddings[len(raw_chunks):])
        ):
            chunk_id = str(uuid.uuid4())
            vector_id = str(uuid.uuid4())
            chunk_index = len(raw_chunks) + i
            
            doc_chunk = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                content=description,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                start_char=0,
                end_char=0,
                token_count=len(description.split()),  # Approximate token count
                page_number=page_num,
                chunk_type="visual",
                image_path=image_path,
                vector_id=vector_id,
                metadata={
                    "document_name": doc.name,
                    "source_path": doc.source_path,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "page_number": page_num,
                    **doc.metadata
                }
            )
            chunks.append(doc_chunk)
            
            records.append(VectorRecord(
                id=vector_id,
                vector=embedding,
                payload={
                    'document_id': document_id,
                    'domain_id': doc.domain_id,
                    'content': description,
                    'chunk_index': chunk_index,
                    'chunk_id': chunk_id,
                    'chunk_type': 'visual',
                    'image_path': image_path,
                    'page_number': page_num,
                    'document_name': doc.name,
                    'source_path': doc.source_path,
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
        
        logger.info(f"Successfully processed document {document_id}: {len(chunks)} chunks ({len(raw_chunks)} text + {len(visual_descriptions)} visual)")
        
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

# Set the background processing function reference for routes
app.state.process_document_background = process_document_background

# RAG routes moved to routes/rag.py
# GraphRAG routes moved to routes/graphrag.py


# Remaining RAG route code removed - see routes/rag.py


# All RAG/GraphRAG routes have been moved to routes/rag.py and routes/graphrag.py


# Workflow routes moved to routes/workflows.py


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
