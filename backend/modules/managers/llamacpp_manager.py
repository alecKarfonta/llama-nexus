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

