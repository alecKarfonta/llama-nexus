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

from fastapi import HTTPException
from huggingface_hub import hf_hub_url, HfApi
from enhanced_logger import enhanced_logger as logger
from modules.managers.base import DOCKER_AVAILABLE, docker_client

try:
    import docker
except ImportError:
    docker = None  # type: ignore[assignment]
from modules.llamacpp_build_info import (
    build_info_from_tag_and_version,
    parse_llguidance_enabled,
)
from modules.mtp_deploy import (
    apply_mtp_workload_profile,
    chat_template_kwargs_cli_value,
    mtp_env_from_config,
    mtp_enabled_in_config,
    normalize_mtp_config,
    validate_mtp_deployment,
)
from modules.mtp_log_parser import parse_mtp_stats_from_log_line


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
        self.llamacpp_docker_image = os.getenv(
            "LLAMACPP_DOCKER_IMAGE",
            "llama-nexus-llamacpp-api:latest",
        ).strip()
        mv = os.getenv("LLAMACPP_MODELS_VOLUME", "").strip()
        if mv:
            self.models_volume_name = mv
        elif self.docker_network.endswith("_default"):
            self.models_volume_name = f"{self.docker_network[:-len('_default')]}_gpt_oss_models"
        else:
            self.models_volume_name = "llama-nexus_gpt_oss_models"
        
        # Reference to download manager for metadata access
        self.download_manager: Optional[ModelDownloadManager] = None
        
        # Defer container log reading to runtime callers to avoid loop issues during import

    def _execution_uses_gpu(self) -> bool:
        """True when the managed container should receive GPU devices."""
        mode = self.config.get("execution", {}).get("mode", "gpu")
        if mode == "cpu":
            return False
        gpu_layers = self.config.get("model", {}).get("gpu_layers", 999)
        return gpu_layers != 0

    def _resolve_cuda_devices(self) -> str:
        cuda_devices = self.config.get("execution", {}).get("cuda_devices", "all")
        if cuda_devices is None or str(cuda_devices).strip() == "":
            return os.getenv("CUDA_DEVICES", "all")
        return str(cuda_devices)

    def _docker_gpu_attachment(self, cuda_devices: str) -> Dict[str, Any]:
        """
        Build Docker GPU attachment for requested host GPU indices.

        ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` aligns CUDA ordinals with ``nvidia-smi``
        indices on mixed GPU topologies (e.g. 5070 + 5060 rigs).
        """
        dev = str(cuda_devices or "all").strip()
        env = {
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
        caps = [["gpu", "compute", "utility"]]
        if dev and dev.lower() != "all":
            device_ids = [d.strip() for d in dev.split(",") if d.strip()]
            device_requests = [
                docker.types.DeviceRequest(
                    driver="nvidia",
                    device_ids=device_ids,
                    capabilities=caps,
                )
            ]
            env["CUDA_VISIBLE_DEVICES"] = dev
            env["NVIDIA_VISIBLE_DEVICES"] = dev
        else:
            device_requests = [
                docker.types.DeviceRequest(driver="nvidia", count=-1, capabilities=caps)
            ]
        # Docker CLI requires quoted form: --gpus '"device=0,1"' (bare device=0,1 errors).
        gpus_cli = "all"
        if dev and dev.lower() != "all":
            gpus_cli = f'"device={dev}"'
        return {"device_requests": device_requests, "env": env, "gpus_cli": gpus_cli}
        
    def load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from environment or defaults"""
        def optional_int(name: str) -> Optional[int]:
            value = os.getenv(name)
            return int(value) if value not in (None, "") else None

        def optional_float(name: str) -> Optional[float]:
            value = os.getenv(name)
            return float(value) if value not in (None, "") else None

        sampling_config = {
            "temperature": optional_float("TEMPERATURE"),
            "top_p": optional_float("TOP_P"),
            "top_k": optional_int("TOP_K"),
            "min_p": optional_float("MIN_P"),
            "repeat_penalty": optional_float("REPEAT_PENALTY"),
            "repeat_last_n": optional_int("REPEAT_LAST_N"),
            "frequency_penalty": optional_float("FREQUENCY_PENALTY"),
            "presence_penalty": optional_float("PRESENCE_PENALTY"),
            "dry_multiplier": optional_float("DRY_MULTIPLIER"),
            "dry_base": optional_float("DRY_BASE"),
            "dry_allowed_length": optional_int("DRY_ALLOWED_LENGTH"),
            "dry_penalty_last_n": optional_int("DRY_PENALTY_LAST_N"),
        }
        sampling_config = {k: v for k, v in sampling_config.items() if v is not None}

        return {
            "model": {
                "name": os.getenv("MODEL_NAME", "Qwen3.6-27B"),
                "variant": os.getenv("MODEL_VARIANT", "Q6_K"),
                "mmproj": os.getenv("MMPROJ_FILE"),
                "context_size": int(os.getenv("CONTEXT_SIZE", "128000")),
                "gpu_layers": int(os.getenv("GPU_LAYERS", "999")),
                "n_cpu_moe": int(os.getenv("N_CPU_MOE", "0")),
                "flash_attn": os.getenv("FLASH_ATTN", "auto"),
            },
            "template": {
                "directory": os.getenv("TEMPLATE_DIR", "/home/llamacpp/templates"),
                "selected": os.getenv("CHAT_TEMPLATE", ""),
            },
            "sampling": sampling_config,
            "performance": {
                "threads": int(os.getenv("THREADS", "-1")),
                "batch_size": int(os.getenv("BATCH_SIZE", "2048")),
                "ubatch_size": int(os.getenv("UBATCH_SIZE", "512")),
                "num_predict": optional_int("NUM_PREDICT"),
                "num_keep": optional_int("NUM_KEEP"),
                # Default to 1 slot — prevents llama.cpp auto-selecting n_parallel=4,
                # which silently reserves KV cache × 4 (wastes ~4.5 GB on 256K context).
                # Set higher only for multi-user / concurrent request workloads.
                "parallel_slots": int(os.getenv("PARALLEL_SLOTS", "1")),
                "ctx_checkpoints": optional_int("CTX_CHECKPOINTS"),
                "split_mode": os.getenv("SPLIT_MODE"),
                "tensor_split": os.getenv("TENSOR_SPLIT"),
                "main_gpu": optional_int("MAIN_GPU"),
                "cache_type_k": os.getenv("CACHE_TYPE_K"),
                "cache_type_v": os.getenv("CACHE_TYPE_V"),
            },
            "speculative": {
                # Empty by default — classic draft-model speculation (separate GGUF)
            },
            "mtp": {
                "enabled": os.getenv("MTP_ENABLED", "false").lower() in ("1", "true", "yes"),
                "workload_profile": os.getenv("MTP_WORKLOAD_PROFILE", "chat"),
                "draft_n_max": int(os.getenv("MTP_DRAFT_N_MAX", "3")),
                "draft_n_min": int(os.getenv("MTP_DRAFT_N_MIN", "0")),
                "draft_p_min": float(os.getenv("MTP_DRAFT_P_MIN", "0.75")),
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
                "reasoning_budget": 0,  # Disable thinking by default - prevents thinking models from burning entire token budget on reasoning
            }
        }
    
    def build_command(self) -> List[str]:
        """Build llamacpp server command from configuration"""
        config = apply_mtp_workload_profile(self.config)
        # Try different filename patterns to find the actual model file
        model_name = config['model']['name']
        variant = config['model']['variant']
        
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

        def add_dry_sampling_params(cmd_list, sampling: Dict[str, Any]):
            """Emit DRY flags only when DRY is enabled (non-zero multiplier)."""
            multiplier = sampling.get("dry_multiplier")
            if multiplier is None or multiplier == 0 or multiplier == 0.0:
                return
            add_param_if_set(cmd_list, "--dry-multiplier", multiplier)
            add_param_if_set(cmd_list, "--dry-base", sampling.get("dry_base"))
            add_param_if_set(cmd_list, "--dry-allowed-length", sampling.get("dry_allowed_length"))
            penalty_last_n = sampling.get("dry_penalty_last_n")
            # 0 disables DRY range checking in llama.cpp — omit (same as default off)
            if penalty_last_n is not None and penalty_last_n != 0:
                add_param_if_set(cmd_list, "--dry-penalty-last-n", penalty_last_n)
        
        cmd = [
            "llama-server",
            "--model", model_path,
            "--host", config["server"]["host"],
            "--port", str(config["server"]["port"]),
            "--api-key", config["server"]["api_key"],
        ]
        
        # Add optional model parameters only if they are set
        add_param_if_set(cmd, "--ctx-size", self.config["model"].get("context_size"))
        
        # Override GPU layers based on execution mode
        if not self._execution_uses_gpu():
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
        add_param_if_set(cmd, "--parallel", config["performance"].get("parallel_slots"))
        add_param_if_set(cmd, "--split-mode", self.config["performance"].get("split_mode"))
        add_param_if_set(cmd, "--tensor-split", self.config["performance"].get("tensor_split"))
        add_param_if_set(cmd, "--main-gpu", self.config["performance"].get("main_gpu"))
        add_param_if_set(cmd, "--cache-type-k", self.config["performance"].get("cache_type_k"))
        add_param_if_set(cmd, "--cache-type-v", self.config["performance"].get("cache_type_v"))
        add_param_if_set(cmd, "--ctx-checkpoints", self.config["performance"].get("ctx_checkpoints"))
        
        # Add optional sampling parameters
        add_param_if_set(cmd, "--temp", self.config["sampling"].get("temperature"))
        add_param_if_set(cmd, "--top-p", self.config["sampling"].get("top_p"))
        add_param_if_set(cmd, "--top-k", self.config["sampling"].get("top_k"))
        add_param_if_set(cmd, "--min-p", self.config["sampling"].get("min_p"))
        add_param_if_set(cmd, "--repeat-penalty", self.config["sampling"].get("repeat_penalty"))
        add_param_if_set(cmd, "--repeat-last-n", self.config["sampling"].get("repeat_last_n"))
        add_param_if_set(cmd, "--frequency-penalty", self.config["sampling"].get("frequency_penalty"))
        add_param_if_set(cmd, "--presence-penalty", self.config["sampling"].get("presence_penalty"))
        add_dry_sampling_params(cmd, self.config["sampling"])
        
        # Add advanced sampling parameters
        add_param_if_set(cmd, "--top-n-sigma", self.config["sampling"].get("top_n_sigma"))
        add_param_if_set(cmd, "--typical-p", self.config["sampling"].get("typical_p"))
        add_param_if_set(cmd, "--xtc-probability", self.config["sampling"].get("xtc_probability"))
        add_param_if_set(cmd, "--xtc-threshold", self.config["sampling"].get("xtc_threshold"))
        add_param_if_set(cmd, "--mirostat", self.config["sampling"].get("mirostat"))
        add_param_if_set(cmd, "--mirostat-lr", self.config["sampling"].get("mirostat_lr"))
        add_param_if_set(cmd, "--mirostat-ent", self.config["sampling"].get("mirostat_ent"))
        add_param_if_set(cmd, "--dynatemp-range", self.config["sampling"].get("dynatemp_range"))
        add_param_if_set(cmd, "--dynatemp-exp", self.config["sampling"].get("dynatemp_exp"))
        
        # Add optional server parameters
        add_param_if_set(cmd, "--timeout", self.config["server"].get("timeout"))
        add_param_if_set(cmd, "--system-prompt-file", self.config["server"].get("system_prompt_file"))
        add_param_if_set(cmd, "--log-format", self.config["server"].get("log_format"))
        add_param_if_set(cmd, "--cache-reuse", config["server"].get("cache_reuse"))
        add_param_if_set(cmd, "--reasoning-format", self.config["server"].get("reasoning_format"))
        add_param_if_set(cmd, "--reasoning-budget", self.config["server"].get("reasoning_budget"))
        add_param_if_set(cmd, "--sleep-idle-seconds", self.config["server"].get("sleep_idle_seconds"))
        
        # Add boolean flags only if they are explicitly set to True
        if self.config["server"].get("embedding"):
            cmd.append("--embeddings")
        if self.config["server"].get("metrics"):
            cmd.append("--metrics")
        if self.config["server"].get("log_disable"):
            cmd.append("--log-disable")
        if self.config["server"].get("slots_endpoint_disable"):
            cmd.append("--no-slots")
        if self.config["performance"].get("memory_f32"):
            cmd.append("--memory-f32")
        if self.config["performance"].get("mlock"):
            cmd.append("--mlock")
        if self.config["performance"].get("no_mmap"):
            cmd.append("--no-mmap")
        if self.config["performance"].get("continuous_batching"):
            cmd.append("--cont-batching")
        
        # Prompt caching (tri-state: None=default, True=enabled, False=disabled)
        cache_prompt = self.config["server"].get("cache_prompt")
        if cache_prompt is True:
            cmd.append("--cache-prompt")
        elif cache_prompt is False:
            cmd.append("--no-cache-prompt")
        
        # KV offload (tri-state)
        kv_offload = self.config["model"].get("kv_offload")
        if kv_offload is True:
            cmd.append("--kv-offload")
        elif kv_offload is False:
            cmd.append("--no-kv-offload")
        
        # Jinja templates (only add if explicitly set; template-file logic below handles --jinja for custom templates)
        jinja = config["server"].get("jinja")
        if jinja is True:
            cmd.append("--jinja")
        elif jinja is False:
            cmd.append("--no-jinja")

        chat_kwargs = chat_template_kwargs_cli_value(config.get("server") or {})
        if chat_kwargs:
            if "--jinja" not in cmd and "--no-jinja" not in cmd:
                cmd.append("--jinja")
            add_param_if_set(cmd, "--chat-template-kwargs", chat_kwargs)
        
        # Add NUMA parameter if set
        add_param_if_set(cmd, "--numa", self.config["performance"].get("numa"))
        
        # Add model-level performance parameters
        add_param_if_set(cmd, "--defrag-thold", self.config["model"].get("defrag_thold"))
        
        # Flash attention requires an explicit value in current llama-server:
        # --flash-attn [on|off|auto]
        flash_attn = self.config["model"].get("flash_attn", "auto")
        flash_attn_value = "auto"
        if isinstance(flash_attn, bool):
            flash_attn_value = "on" if flash_attn else "off"
        elif flash_attn is None:
            flash_attn_value = "auto"
        else:
            normalized = str(flash_attn).strip().lower()
            if normalized in ("", "none", "null", "."):
                flash_attn_value = "auto"
            elif normalized in ("true", "1", "yes", "on"):
                flash_attn_value = "on"
            elif normalized in ("false", "0", "no", "off"):
                flash_attn_value = "off"
            elif normalized in ("auto", "on", "off"):
                flash_attn_value = normalized
            else:
                # Keep errors visible but avoid invalid CLI syntax.
                logger.warning(
                    f"Invalid flash_attn value '{flash_attn}', defaulting to 'auto'"
                )
                flash_attn_value = "auto"
        cmd.extend(["--flash-attn", flash_attn_value])
        
        # Multi-Token Prediction (built-in draft heads in the same GGUF)
        if mtp_enabled_in_config(config):
            mtp = normalize_mtp_config(config)
            cmd.extend(["--spec-type", "draft-mtp"])
            add_param_if_set(cmd, "--spec-draft-n-max", mtp.get("draft_n_max"))
            add_param_if_set(cmd, "--spec-draft-n-min", mtp.get("draft_n_min"))
            add_param_if_set(cmd, "--spec-draft-p-min", mtp.get("draft_p_min"))

        # Classic speculative decoding (separate draft model file)
        if "speculative" in self.config:
            spec = self.config["speculative"]
            add_param_if_set(cmd, "--model-draft", spec.get("model_draft"))
            add_param_if_set(cmd, "--gpu-layers-draft", spec.get("gpu_layers_draft"))
            add_param_if_set(cmd, "--ctx-size-draft", spec.get("ctx_size_draft"))
            add_param_if_set(cmd, "--draft-max", spec.get("draft_max"))
            add_param_if_set(cmd, "--draft-min", spec.get("draft_min"))
            add_param_if_set(cmd, "--draft-p-min", spec.get("draft_p_min"))
        
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
            # Always enable jinja when using a custom template file
            if "--jinja" not in cmd and "--no-jinja" not in cmd:
                cmd.append("--jinja")
            cmd.extend([
                "--chat-template-file", str(Path(self.config["template"]["directory"]) / self.config["template"]["selected"]),
            ])
        
        return cmd

    def _model_mtp_capable(self) -> bool:
        """Whether the configured model GGUF has MTP heads (from registry metadata)."""
        model_name = self.config.get("model", {}).get("name")
        variant = self.config.get("model", {}).get("variant")
        if not model_name or not variant or not self.download_manager:
            return False
        meta = self.download_manager._load_model_metadata(model_name, variant)
        return bool(meta and meta.get("mtp_capable"))

    def validate_mtp_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate MTP settings; returns {errors, warnings}."""
        cfg = dict(self.config)
        if config:
            for section in ("model", "performance", "mtp", "sampling", "server"):
                if section in config and isinstance(config[section], dict):
                    cfg[section] = {**cfg.get(section, {}), **config[section]}
        cfg = apply_mtp_workload_profile(cfg)
        cfg["mtp"] = normalize_mtp_config(cfg)
        build_info = self.get_llamacpp_build_info()
        errors, warnings = validate_mtp_deployment(
            cfg,
            model_mtp_capable=self._model_mtp_capable_for_config(cfg),
            llamacpp_build_number=build_info.get("build_number"),
            mtp_supported_build=build_info.get("mtp_supported"),
        )
        return {"errors": errors, "warnings": warnings, "valid": len(errors) == 0}

    def _model_mtp_capable_for_config(self, config: Dict[str, Any]) -> bool:
        model_name = config.get("model", {}).get("name")
        variant = config.get("model", {}).get("variant")
        if not model_name or not variant or not self.download_manager:
            return False
        meta = self.download_manager._load_model_metadata(model_name, variant)
        return bool(meta and meta.get("mtp_capable"))

    def _ensure_mtp_valid_or_raise(self, config: Optional[Dict[str, Any]] = None) -> None:
        result = self.validate_mtp_config(config)
        if not result["valid"]:
            raise HTTPException(
                status_code=400,
                detail="; ".join(result["errors"]),
            )
    
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
        
        self._ensure_mtp_valid_or_raise()

        # Build and validate command
        cmd = self.build_command()
        cmd_str = ' '.join(cmd)
        logger.info(f"Built llamacpp command with {len(cmd)} arguments: '{cmd_str[:100]}...'" if len(cmd_str) > 100 else f"Built llamacpp command: '{cmd_str}'")
        
        # Validate Docker image exists
        image_name = self.llamacpp_docker_image
        try:
            docker_client.images.get(image_name)
            logger.info(f"Docker image validation successful for image_name='{image_name}'")
        except docker.errors.ImageNotFound:
            error_msg = (
                f"Docker image not found: {image_name}. "
                "From the repository root build the llamacpp-api image "
                "(this tags the inference image even if you do not run the optional `extra` profile): "
                "`docker compose build llamacpp-api`"
            )
            logger.error(f"Docker image validation failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        logger.info("Initiating Docker container creation and startup...")
        
        # Resolve host templates dir (can be overridden via env)
        host_templates_dir = os.getenv(
            "TEMPLATES_HOST_DIR",
            str(Path(__file__).resolve().parents[1] / "chat-templates")
        )

        cuda_devices = self._resolve_cuda_devices()
        
        # Get repository info from metadata if available
        model_name = self.config['model']['name']
        variant = self.config['model']['variant']
        model_repo = None
        
        if self.download_manager:
            model_repo = self.download_manager.get_model_repository(model_name, variant)
        
        env_vars = {
            "MODEL_NAME": model_name,
            "MODEL_VARIANT": variant,
            **mtp_env_from_config(self.config),
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
                self.models_volume_name: {"bind": "/home/llamacpp/models", "mode": "rw"},
                host_templates_dir: {"bind": "/home/llamacpp/templates", "mode": "ro"}
            },
            "environment": env_vars,
            "shm_size": "16g",
            "ports": {"8080/tcp": 8600}
        }
        
        # Attach GPU unless explicitly in CPU mode (or gpu_layers=0)
        if self._execution_uses_gpu():
            gpu_attach = self._docker_gpu_attachment(cuda_devices)
            env_vars.update(gpu_attach["env"])
            container_config["device_requests"] = gpu_attach["device_requests"]
        
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
        
        self._ensure_mtp_valid_or_raise()

        # Build command
        cmd = self.build_command()
        logger.info(f"Built command: {' '.join(cmd)}")
        
        # Check if image exists (-q avoids fragile repository-only filters when tag matters)
        image_name = self.llamacpp_docker_image
        image_check = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            image_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await image_check.communicate()

        if image_check.returncode != 0 or not stdout.decode().strip():
            error_msg = (
                f"Docker image not found: {image_name}. "
                "From the repository root build the llamacpp-api image: "
                "`docker compose build llamacpp-api`"
            )
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        logger.info(f"Docker image found: {image_name}")
        
        # Build docker run command
        host_templates_dir = os.getenv(
            "TEMPLATES_HOST_DIR",
            str(Path(__file__).resolve().parents[1] / "chat-templates")
        )
        
        cuda_devices = self._resolve_cuda_devices()
        
        # Get repository info from metadata if available
        model_name = self.config['model']['name']
        variant = self.config['model']['variant']
        model_repo = None
        
        if self.download_manager:
            model_repo = self.download_manager.get_model_repository(model_name, variant)
        
        mtp_env = mtp_env_from_config(self.config)
        docker_cmd = [
            'docker', 'run', '-d',
            '--name', self.container_name,
            '--shm-size', '16g',
            '-p', '8600:8080',
            '--network', self.docker_network,
            '-v', f'{self.models_volume_name}:/home/llamacpp/models',
            '-v', f'{host_templates_dir}:/home/llamacpp/templates:ro',
            '-e', f'MODEL_NAME={model_name}',
            '-e', f'MODEL_VARIANT={variant}',
            '-e', f'MTP_ENABLED={mtp_env["MTP_ENABLED"]}',
            '-e', f'MTP_DRAFT_N_MAX={mtp_env["MTP_DRAFT_N_MAX"]}',
            '-e', f'MTP_DRAFT_N_MIN={mtp_env["MTP_DRAFT_N_MIN"]}',
            '-e', f'MTP_DRAFT_P_MIN={mtp_env["MTP_DRAFT_P_MIN"]}',
        ]
        
        # Only set MODEL_REPO if we have metadata for it
        if model_repo:
            docker_cmd.extend(['-e', f'MODEL_REPO={model_repo}'])
            logger.info(f"Using repository from metadata: {model_repo} for {model_name}-{variant}")
        else:
            logger.warning(f"No repository metadata found for {model_name}-{variant}, container will use fallback logic")
        
        # Attach GPU unless explicitly in CPU mode (or gpu_layers=0)
        if self._execution_uses_gpu():
            gpu_attach = self._docker_gpu_attachment(cuda_devices)
            docker_cmd.extend(['--gpus', gpu_attach["gpus_cli"]])
            for key, value in gpu_attach["env"].items():
                docker_cmd.extend(['-e', f'{key}={value}'])
        
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
            
            self._ensure_mtp_valid_or_raise()

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
        timestamp = datetime.now().isoformat()
        # Add to buffer
        self.log_buffer.append({
            "message": line,
            "timestamp": timestamp,
        })

        mtp_stats = parse_mtp_stats_from_log_line(line)
        if mtp_stats is not None:
            self.log_buffer.append({
                "message": line,
                "timestamp": timestamp,
                "mtp_stats": mtp_stats,
            })
        
        # Broadcast to websocket clients
        disconnected_clients = []
        for client in self.websocket_clients:
            try:
                await client.send_json({
                    "type": "log",
                    "data": line,
                    "timestamp": timestamp,
                })
                if mtp_stats is not None:
                    await client.send_json({
                        "type": "mtp_stats",
                        "data": mtp_stats,
                        "timestamp": timestamp,
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

    def _probe_running(self) -> bool:
        """Check if the llamacpp service container or subprocess is running (no status payload)."""
        if self.use_docker:
            if DOCKER_AVAILABLE and docker_client:
                try:
                    container = docker_client.containers.get(self.container_name)
                    return container.status == "running"
                except Exception:
                    return False
            try:
                result = subprocess.run(
                    ['docker', 'ps', '--filter', f'name={self.container_name}', '--format', '{{.Names}}'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    container_names = result.stdout.strip().split('\n')
                    return self.container_name in container_names and bool(result.stdout.strip())
            except Exception as e:
                logger.warning(f"Error checking Docker container status: {e}")
            return False
        return self.process is not None and self.process.poll() is None

    def is_running(self) -> bool:
        """Check if the llamacpp service container or subprocess is running."""
        return self._probe_running()

    def _env_dict_from_docker_config(self, config: Dict[str, Any]) -> Dict[str, str]:
        env_list = config.get("Env") or []
        result: Dict[str, str] = {}
        for entry in env_list:
            if isinstance(entry, str) and "=" in entry:
                key, value = entry.split("=", 1)
                result[key] = value
        return result

    def _docker_cat_file(self, target: str, path: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["docker", "exec", target, "cat", path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _read_llamacpp_build_from_container_attrs(
        self,
        attrs: Dict[str, Any],
        *,
        exec_target: Optional[str] = None,
    ) -> Dict[str, Any]:
        config = attrs.get("Config") or {}
        labels = config.get("Labels") or {}
        env = self._env_dict_from_docker_config(config)
        tag = env.get("LLAMACPP_BUILD_TAG") or labels.get("org.opencontainers.image.version")
        cli_version: Optional[str] = None
        if exec_target:
            cli_version = self._docker_cat_file(exec_target, "/etc/llama-nexus/llama-cli-version")
            if not tag:
                tag = self._docker_cat_file(exec_target, "/etc/llama-nexus/llamacpp-build-tag")
        else:
            try:
                version_path = Path("/etc/llama-nexus/llama-cli-version")
                if version_path.is_file():
                    cli_version = version_path.read_text(encoding="utf-8").strip()
                tag_path = Path("/etc/llama-nexus/llamacpp-build-tag")
                if not tag and tag_path.is_file():
                    tag = tag_path.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        llguidance_raw: Optional[str] = None
        if exec_target:
            llguidance_raw = self._docker_cat_file(
                exec_target, "/etc/llama-nexus/llamacpp-llguidance-enabled"
            )
        else:
            try:
                llg_path = Path("/etc/llama-nexus/llamacpp-llguidance-enabled")
                if llg_path.is_file():
                    llguidance_raw = llg_path.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        return build_info_from_tag_and_version(
            tag,
            cli_version,
            llguidance_enabled=parse_llguidance_enabled(llguidance_raw),
        )

    def _read_llamacpp_build_from_docker_image(self) -> Dict[str, Any]:
        if DOCKER_AVAILABLE and docker_client:
            try:
                image = docker_client.images.get(self.llamacpp_docker_image)
                return self._read_llamacpp_build_from_container_attrs(image.attrs)
            except Exception as exc:
                logger.debug(f"Could not inspect llama.cpp image build info: {exc}")
        try:
            result = subprocess.run(
                [
                    "docker",
                    "inspect",
                    self.llamacpp_docker_image,
                    "--format",
                    "{{json .Config}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                config = json.loads(result.stdout)
                return self._read_llamacpp_build_from_container_attrs({"Config": config})
        except Exception as exc:
            logger.debug(f"docker inspect image build info failed: {exc}")
        return build_info_from_tag_and_version(None)

    def get_llamacpp_build_info(self, *, running: Optional[bool] = None) -> Dict[str, Any]:
        """Resolve pinned llama.cpp build tag/version from the inference image or container."""
        if self.use_docker:
            running = self._probe_running() if running is None else running
            exec_target = self.container_name if running else None
            if DOCKER_AVAILABLE and docker_client:
                try:
                    container = docker_client.containers.get(self.container_name)
                    return self._read_llamacpp_build_from_container_attrs(
                        container.attrs,
                        exec_target=exec_target,
                    )
                except Exception:
                    pass
            else:
                try:
                    result = subprocess.run(
                        [
                            "docker",
                            "inspect",
                            self.container_name,
                            "--format",
                            "{{json .Config}}",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        config = json.loads(result.stdout)
                        return self._read_llamacpp_build_from_container_attrs(
                            {"Config": config},
                            exec_target=exec_target,
                        )
                except Exception as exc:
                    logger.debug(f"docker inspect build info failed: {exc}")
            return self._read_llamacpp_build_from_docker_image()

        cli_version: Optional[str] = None
        try:
            result = subprocess.run(
                ["llama-cli", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                cli_version = result.stdout.strip().splitlines()[0] if result.stdout else None
        except Exception:
            pass
        return build_info_from_tag_and_version(None, cli_version)

    def get_status(self) -> Dict[str, Any]:
        """Get current status of llamacpp"""
        is_running = self._probe_running()
        pid: Optional[int] = None

        if is_running:
            if self.use_docker:
                if DOCKER_AVAILABLE and docker_client:
                    try:
                        container = docker_client.containers.get(self.container_name)
                        pid = container.attrs.get('State', {}).get('Pid')
                    except Exception:
                        pass
                else:
                    try:
                        pid_result = subprocess.run(
                            ['docker', 'inspect', '--format', '{{.State.Pid}}', self.container_name],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if pid_result.returncode == 0:
                            pid = int(pid_result.stdout.strip())
                    except Exception:
                        pass
            elif self.process is not None:
                pid = self.process.pid

        llamacpp_build = self.get_llamacpp_build_info(running=is_running)

        status: Dict[str, Any] = {
            "running": is_running,
            "pid": pid,
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time and is_running else 0,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "mode": "docker" if self.use_docker else "subprocess",
            "llamacpp_build": llamacpp_build,
            "llamacpp_build_number": llamacpp_build.get("build_number"),
            "mtp_supported": llamacpp_build.get("mtp_supported", False),
            "llguidance_enabled": llamacpp_build.get("llguidance_enabled", False),
            "grammar_gbnf_supported": llamacpp_build.get("grammar_gbnf_supported", True),
            "config": self.config,
            "model": {
                "name": self.config.get("model", {}).get("name"),
                "variant": self.config.get("model", {}).get("variant"),
                "context_size": self.config.get("model", {}).get("context_size", 4096),
                "gpu_layers": self.config.get("model", {}).get("gpu_layers", 999),
                "n_cpu_moe": self.config.get("model", {}).get("n_cpu_moe", 0),
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
    
    def get_field_metadata(self) -> Dict[str, Any]:
        """Return metadata describing which config fields exist for llama.cpp,
        their scope, and cross-framework mapping information."""
        return {
            "model": {
                "name": {"scope": "shared", "type": "text", "description": "Model file name (without extension)", "llamacpp_flag": "--model", "vllm_equivalent": "model.name"},
                "variant": {"scope": "llamacpp", "type": "text", "description": "GGUF quantization variant (e.g., Q4_K_M)", "llamacpp_flag": "(part of filename)", "vllm_equivalent": None, "note": "vLLM uses HuggingFace model names, not GGUF variants"},
                "context_size": {"scope": "llamacpp", "type": "number", "description": "Context window size in tokens.", "llamacpp_flag": "--ctx-size", "vllm_equivalent": "performance.max_model_len", "mapping_note": "llama.cpp: context_size, vLLM: max_model_len", "min": 512, "max": 1000000},
                "gpu_layers": {"scope": "llamacpp", "type": "number", "description": "Number of model layers to offload to GPU. -1 for all, 0 for CPU only.", "llamacpp_flag": "--n-gpu-layers", "vllm_equivalent": None, "note": "vLLM auto-offloads all layers; use gpu_memory_utilization to control VRAM usage", "min": -1, "max": 999},
                "n_cpu_moe": {"scope": "llamacpp", "type": "number", "description": "Run MoE experts on CPU (0=GPU, >0=number of CPU MoE layers).", "llamacpp_flag": "--n-cpu-moe", "vllm_equivalent": None},
                "flash_attn": {"scope": "llamacpp", "type": "select", "description": "Flash attention mode.", "llamacpp_flag": "--flash-attn", "options": ["auto", "on", "off"], "vllm_equivalent": None, "note": "vLLM uses flash attention by default when available"},
                "mmproj": {"scope": "llamacpp", "type": "text", "description": "Vision projection model file for multimodal models.", "llamacpp_flag": "--mmproj", "vllm_equivalent": None},
                "lora": {"scope": "llamacpp", "type": "text", "description": "Path to LoRA adapter file.", "llamacpp_flag": "--lora", "vllm_equivalent": None, "note": "vLLM handles LoRA differently via its own adapter system"},
                "lora_base": {"scope": "llamacpp", "type": "text", "description": "Base model for LoRA adapter.", "llamacpp_flag": "--lora-base", "vllm_equivalent": None},
                "kv_offload": {"scope": "llamacpp", "type": "boolean", "description": "Offload KV cache to CPU.", "llamacpp_flag": "--kv-offload / --no-kv-offload", "vllm_equivalent": None},
                "defrag_thold": {"scope": "llamacpp", "type": "number", "description": "KV cache defragmentation threshold.", "llamacpp_flag": "--defrag-thold", "vllm_equivalent": None},
            },
            "sampling": {
                "temperature": {"scope": "shared", "type": "number", "description": "Controls randomness in text generation.", "llamacpp_flag": "--temp", "vllm_flag": "per-request", "min": 0, "max": 2, "step": 0.1},
                "top_p": {"scope": "shared", "type": "number", "description": "Nucleus sampling threshold.", "llamacpp_flag": "--top-p", "vllm_flag": "per-request", "min": 0, "max": 1, "step": 0.05},
                "top_k": {"scope": "shared", "type": "number", "description": "Limits next token selection to K most probable tokens.", "llamacpp_flag": "--top-k", "vllm_flag": "per-request", "min": -1, "max": 200},
                "min_p": {"scope": "llamacpp", "type": "number", "description": "Minimum probability threshold relative to most likely token.", "llamacpp_flag": "--min-p", "vllm_equivalent": None, "min": 0, "max": 1, "step": 0.01},
                "repeat_penalty": {"scope": "shared", "type": "number", "description": "Penalizes repeated tokens.", "llamacpp_flag": "--repeat-penalty", "vllm_flag": "per-request (repetition_penalty)", "mapping_note": "llama.cpp: repeat_penalty, vLLM: repetition_penalty", "min": 0.1, "max": 2.0, "step": 0.1},
                "repeat_last_n": {"scope": "llamacpp", "type": "number", "description": "Number of recent tokens for repetition penalty.", "llamacpp_flag": "--repeat-last-n", "vllm_equivalent": None, "min": -1, "max": 2048},
                "frequency_penalty": {"scope": "shared", "type": "number", "description": "Penalizes tokens based on frequency.", "llamacpp_flag": "--frequency-penalty", "vllm_flag": "per-request", "min": -2.0, "max": 2.0, "step": 0.1},
                "presence_penalty": {"scope": "shared", "type": "number", "description": "Penalizes tokens that have already appeared.", "llamacpp_flag": "--presence-penalty", "vllm_flag": "per-request", "min": -2.0, "max": 2.0, "step": 0.1},
                "dry_multiplier": {"scope": "llamacpp", "type": "number", "description": "DRY (Don't Repeat Yourself) penalty strength.", "llamacpp_flag": "--dry-multiplier", "vllm_equivalent": None, "min": 0, "max": 5, "step": 0.1},
                "dry_base": {"scope": "llamacpp", "type": "number", "description": "DRY penalty base value.", "llamacpp_flag": "--dry-base", "vllm_equivalent": None, "min": 1.0, "max": 4.0, "step": 0.1},
                "dry_allowed_length": {"scope": "llamacpp", "type": "number", "description": "Minimum sequence length before DRY penalty.", "llamacpp_flag": "--dry-allowed-length", "vllm_equivalent": None, "min": 1, "max": 20},
                "dry_penalty_last_n": {"scope": "llamacpp", "type": "number", "description": "Number of recent tokens for DRY penalty.", "llamacpp_flag": "--dry-penalty-last-n", "vllm_equivalent": None, "min": -1, "max": 2048},
                "mirostat": {"scope": "llamacpp", "type": "select", "description": "Mirostat adaptive sampling mode.", "llamacpp_flag": "--mirostat", "options": ["", "0", "1", "2"], "vllm_equivalent": None},
                "mirostat_lr": {"scope": "llamacpp", "type": "number", "description": "Mirostat learning rate.", "llamacpp_flag": "--mirostat-lr", "vllm_equivalent": None, "min": 0, "max": 1, "step": 0.01},
                "mirostat_ent": {"scope": "llamacpp", "type": "number", "description": "Mirostat target entropy.", "llamacpp_flag": "--mirostat-ent", "vllm_equivalent": None, "min": 0, "max": 10, "step": 0.1},
                "dynatemp_range": {"scope": "llamacpp", "type": "number", "description": "Dynamic temperature range.", "llamacpp_flag": "--dynatemp-range", "vllm_equivalent": None, "min": 0, "max": 5, "step": 0.1},
                "dynatemp_exp": {"scope": "llamacpp", "type": "number", "description": "Dynamic temperature exponent.", "llamacpp_flag": "--dynatemp-exp", "vllm_equivalent": None, "min": 0, "max": 5, "step": 0.1},
                "top_n_sigma": {"scope": "llamacpp", "type": "number", "description": "Top N-Sigma filtering threshold.", "llamacpp_flag": "--top-n-sigma", "vllm_equivalent": None, "min": 0, "max": 10, "step": 0.1},
                "typical_p": {"scope": "llamacpp", "type": "number", "description": "Locally typical sampling threshold.", "llamacpp_flag": "--typical-p", "vllm_equivalent": None, "min": 0, "max": 1, "step": 0.05},
                "xtc_probability": {"scope": "llamacpp", "type": "number", "description": "XTC sampling probability.", "llamacpp_flag": "--xtc-probability", "vllm_equivalent": None, "min": 0, "max": 1, "step": 0.05},
                "xtc_threshold": {"scope": "llamacpp", "type": "number", "description": "XTC threshold for token removal.", "llamacpp_flag": "--xtc-threshold", "vllm_equivalent": None, "min": 0, "max": 1, "step": 0.05},
            },
            "performance": {
                "threads": {"scope": "llamacpp", "type": "number", "description": "Number of computation threads.", "llamacpp_flag": "--threads", "vllm_equivalent": None, "note": "vLLM manages threading internally"},
                "threads_batch": {"scope": "llamacpp", "type": "number", "description": "Threads for batch processing.", "llamacpp_flag": "--threads-batch", "vllm_equivalent": None},
                "batch_size": {"scope": "llamacpp", "type": "number", "description": "Batch size for prompt processing.", "llamacpp_flag": "--batch-size", "vllm_equivalent": "performance.max_num_batched_tokens"},
                "ubatch_size": {"scope": "llamacpp", "type": "number", "description": "Micro batch size.", "llamacpp_flag": "--ubatch-size", "vllm_equivalent": None},
                "num_predict": {"scope": "llamacpp", "type": "number", "description": "Maximum tokens to predict.", "llamacpp_flag": "--n-predict", "vllm_equivalent": None, "note": "vLLM controls this per-request via max_tokens"},
                "num_keep": {"scope": "llamacpp", "type": "number", "description": "Number of tokens to keep from initial prompt.", "llamacpp_flag": "--keep", "vllm_equivalent": None},
                "parallel_slots": {"scope": "llamacpp", "type": "number", "description": "Number of parallel slots (concurrent requests).", "llamacpp_flag": "--parallel", "vllm_equivalent": "performance.data_parallel_size", "mapping_note": "Roughly equivalent: more slots/replicas = more concurrency"},
                "cache_type_k": {"scope": "llamacpp", "type": "select", "description": "KV cache data type for K.", "llamacpp_flag": "--cache-type-k", "options": ["f16", "q8_0", "q4_0"], "vllm_equivalent": "performance.kv_cache_dtype"},
                "cache_type_v": {"scope": "llamacpp", "type": "select", "description": "KV cache data type for V.", "llamacpp_flag": "--cache-type-v", "options": ["f16", "q8_0", "q4_0"], "vllm_equivalent": "performance.kv_cache_dtype", "note": "llama.cpp has separate K and V types, vLLM uses one"},
                "split_mode": {"scope": "llamacpp", "type": "select", "description": "How to split model across GPUs.", "llamacpp_flag": "--split-mode", "options": ["none", "layer", "row"], "vllm_equivalent": "performance.tensor_parallel_size"},
                "continuous_batching": {"scope": "llamacpp", "type": "boolean", "description": "Enable continuous batching.", "llamacpp_flag": "--cont-batching", "vllm_equivalent": None, "note": "vLLM uses continuous batching by default"},
                "memory_f32": {"scope": "llamacpp", "type": "boolean", "description": "Use float32 for memory.", "llamacpp_flag": "--memory-f32", "vllm_equivalent": None},
                "mlock": {"scope": "llamacpp", "type": "boolean", "description": "Lock model in memory.", "llamacpp_flag": "--mlock", "vllm_equivalent": None},
                "no_mmap": {"scope": "llamacpp", "type": "boolean", "description": "Disable memory mapping.", "llamacpp_flag": "--no-mmap", "vllm_equivalent": None},
            },
            "mtp": {
                "enabled": {"scope": "llamacpp", "type": "boolean", "description": "Enable Multi-Token Prediction (MTP) using built-in GGUF heads.", "llamacpp_flag": "--spec-type draft-mtp", "vllm_equivalent": "speculative.method", "mapping_note": "Requires mtp_capable GGUF and llama.cpp b9193+"},
                "workload_profile": {"scope": "llamacpp", "type": "select", "description": "Workload preset for MTP and server tuning (chat, agent, throughput, custom).", "options": ["chat", "agent", "throughput", "custom"]},
                "draft_n_max": {"scope": "llamacpp", "type": "number", "description": "Max MTP draft tokens per step (--spec-draft-n-max). Chat: 2, agent/tools: 8.", "llamacpp_flag": "--spec-draft-n-max", "min": 1, "max": 16},
                "draft_n_min": {"scope": "llamacpp", "type": "number", "description": "Min MTP draft tokens (--spec-draft-n-min).", "llamacpp_flag": "--spec-draft-n-min", "min": 0, "max": 16},
                "draft_p_min": {"scope": "llamacpp", "type": "number", "description": "Min draft confidence (--spec-draft-p-min). Higher = fewer drafts, often better acceptance.", "llamacpp_flag": "--spec-draft-p-min", "min": 0, "max": 1, "step": 0.05},
            },
            "speculative": {
                "model_draft": {"scope": "llamacpp", "type": "text", "description": "Draft model for speculative decoding.", "llamacpp_flag": "--model-draft", "vllm_equivalent": "speculative.method", "mapping_note": "llama.cpp uses a draft model, vLLM uses methods like MTP"},
                "gpu_layers_draft": {"scope": "llamacpp", "type": "number", "description": "GPU layers for draft model.", "llamacpp_flag": "--gpu-layers-draft", "vllm_equivalent": None},
                "ctx_size_draft": {"scope": "llamacpp", "type": "number", "description": "Context size for draft model.", "llamacpp_flag": "--ctx-size-draft", "vllm_equivalent": None},
                "draft_max": {"scope": "llamacpp", "type": "number", "description": "Maximum draft tokens.", "llamacpp_flag": "--draft-max", "vllm_equivalent": "speculative.num_speculative_tokens"},
                "draft_min": {"scope": "llamacpp", "type": "number", "description": "Minimum draft tokens.", "llamacpp_flag": "--draft-min", "vllm_equivalent": None},
                "draft_p_min": {"scope": "llamacpp", "type": "number", "description": "Minimum draft probability.", "llamacpp_flag": "--draft-p-min", "vllm_equivalent": None},
            },
            "execution": {
                "mode": {"scope": "llamacpp", "type": "select", "description": "Execution mode (GPU or CPU).", "llamacpp_flag": "(controls --n-gpu-layers)", "options": ["gpu", "cpu"], "vllm_equivalent": None, "note": "vLLM requires GPU"},
                "cuda_devices": {"scope": "llamacpp", "type": "text", "description": "CUDA device visibility.", "llamacpp_flag": "env:CUDA_VISIBLE_DEVICES", "vllm_equivalent": None},
            },
            "context_extension": {
                "yarn_ext_factor": {"scope": "llamacpp", "type": "number", "description": "YaRN extrapolation mix factor.", "llamacpp_flag": "--yarn-ext-factor", "vllm_equivalent": None},
                "yarn_attn_factor": {"scope": "llamacpp", "type": "number", "description": "YaRN attention scaling.", "llamacpp_flag": "--yarn-attn-factor", "vllm_equivalent": None},
                "yarn_beta_slow": {"scope": "llamacpp", "type": "number", "description": "YaRN high correction dimension.", "llamacpp_flag": "--yarn-beta-slow", "vllm_equivalent": None},
                "yarn_beta_fast": {"scope": "llamacpp", "type": "number", "description": "YaRN low correction dimension.", "llamacpp_flag": "--yarn-beta-fast", "vllm_equivalent": None},
                "group_attn_n": {"scope": "llamacpp", "type": "number", "description": "Group attention factor.", "llamacpp_flag": "--grp-attn-n", "vllm_equivalent": None},
                "group_attn_w": {"scope": "llamacpp", "type": "number", "description": "Group attention width.", "llamacpp_flag": "--grp-attn-w", "vllm_equivalent": None},
            },
            "server": {
                "host": {"scope": "shared", "type": "text", "description": "IP address to listen on.", "llamacpp_flag": "--host", "vllm_flag": "--host"},
                "port": {"scope": "shared", "type": "number", "description": "Port to listen on.", "llamacpp_flag": "--port", "vllm_flag": "--port"},
                "api_key": {"scope": "shared", "type": "text", "description": "API key for authentication.", "llamacpp_flag": "--api-key", "vllm_flag": "--api-key"},
                "timeout": {"scope": "llamacpp", "type": "number", "description": "Server read/write timeout in seconds.", "llamacpp_flag": "--timeout", "vllm_equivalent": None},
                "embedding": {"scope": "llamacpp", "type": "boolean", "description": "Enable embedding endpoint.", "llamacpp_flag": "--embeddings", "vllm_equivalent": None, "note": "vLLM supports embeddings natively"},
                "metrics": {"scope": "llamacpp", "type": "boolean", "description": "Enable metrics endpoint.", "llamacpp_flag": "--metrics", "vllm_equivalent": None, "note": "vLLM exposes metrics at /metrics by default"},
                "cache_prompt": {"scope": "llamacpp", "type": "boolean", "description": "Cache processed prompts.", "llamacpp_flag": "--cache-prompt / --no-cache-prompt", "vllm_equivalent": None},
                "reasoning_format": {"scope": "llamacpp", "type": "select", "description": "Format for reasoning content extraction.", "llamacpp_flag": "--reasoning-format", "options": ["deepseek", "none"], "vllm_equivalent": "reasoning.reasoning_parser", "mapping_note": "llama.cpp: reasoning_format, vLLM: reasoning_parser"},
                "reasoning_budget": {"scope": "llamacpp", "type": "number", "description": "Maximum reasoning tokens.", "llamacpp_flag": "--reasoning-budget", "vllm_equivalent": None},
                "jinja": {"scope": "llamacpp", "type": "boolean", "description": "Enable Jinja template processing.", "llamacpp_flag": "--jinja", "vllm_equivalent": None},
                "chat_template_kwargs": {"scope": "llamacpp", "type": "text", "description": "JSON kwargs passed to the chat template (--chat-template-kwargs). Agent profile sets enable_thinking=false.", "llamacpp_flag": "--chat-template-kwargs"},
                "cache_reuse": {"scope": "llamacpp", "type": "number", "description": "Reuse prefix KV cache across requests (--cache-reuse). Agent profile uses 1024.", "llamacpp_flag": "--cache-reuse"},
            },
        }

    # --- End of LlamaCPPManager ---

