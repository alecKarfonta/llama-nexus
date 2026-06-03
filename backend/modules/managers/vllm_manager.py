"""
VLLMManager - Manages a vLLM inference service running as a Docker container.

Unlike LlamaCPPManager which builds CLI commands and spawns containers,
VLLMManager works with the existing vllm-api container defined in docker-compose.yml.
It manages lifecycle (start/stop) and provides health/status info.
"""

import os
import asyncio
import subprocess
import json
import httpx
from typing import Optional, Dict, Any
from datetime import datetime
from collections import deque

from enhanced_logger import enhanced_logger as logger

VLLM_DEPLOY_CONFIG_PATH = os.getenv("VLLM_DEPLOY_CONFIG_PATH", "/data/vllm_deploy_config.json")


def _deep_merge_dict(base: Dict[str, Any], incoming: Dict[str, Any]) -> None:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value


try:
    import docker
    DOCKER_AVAILABLE = True
    try:
        docker_client = docker.from_env()
        docker_client.ping()
    except Exception:
        docker_client = None
except ImportError:
    DOCKER_AVAILABLE = False
    docker_client = None


class VLLMManager:
    """Manages vLLM inference backend lifecycle and configuration."""

    def __init__(self):
        self.config: Dict[str, Any] = self.load_default_config()
        self.log_buffer: deque = deque(maxlen=2000)
        self.start_time: Optional[datetime] = None
        self.container_name = "vllm-api"
        self.websocket_clients = []
        self._log_reader_task = None
        self.broadcast_ws_event = None  # Wired up externally
        self.last_action_error: Optional[str] = None
        self._load_persisted_config_if_any()

    def _load_persisted_config_if_any(self) -> None:
        path = VLLM_DEPLOY_CONFIG_PATH
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, dict):
                _deep_merge_dict(self.config, data)
                logger.info("Loaded persisted vLLM Deploy config from %s", path)
        except Exception as e:
            logger.warning("Could not load persisted vLLM config from %s: %s", path, e)

    def persist_deploy_config(self) -> None:
        """Write current config to disk (survives backend-api container restart)."""
        path = VLLM_DEPLOY_CONFIG_PATH
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(self.config, fp, indent=2)
        except OSError as e:
            logger.warning("Could not persist vLLM Deploy config to %s: %s", path, e)

    def server_listen_port(self) -> int:
        """TCP port vLLM listens on inside the container (matches Deploy Server.port)."""
        raw = (self.config.get("server") or {}).get("port", 8080)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 8080

    def internal_http_base(self) -> str:
        """Base URL for sibling containers on the compose network."""
        return f"http://{self.container_name}:{self.server_listen_port()}"

    def load_default_config(self) -> Dict[str, Any]:
        return {
            "backend_type": "vllm",
            "model": {
                "name": os.getenv("VLLM_MODEL_NAME", "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"),
                "served_name": os.getenv("VLLM_SERVED_MODEL_NAME", "Nemotron-3-Nano-Omni-30B-A3B-Reasoning"),
                "dtype": os.getenv("VLLM_DTYPE", "auto"),
                "quantization": os.getenv("VLLM_QUANTIZATION", "fp4"),
            },
            "sampling": {
                "temperature": float(os.getenv("VLLM_TEMPERATURE", "0.7")),
                "top_p": float(os.getenv("VLLM_TOP_P", "0.8")),
                "top_k": int(os.getenv("VLLM_TOP_K", "-1")),
                "repetition_penalty": float(os.getenv("VLLM_REPETITION_PENALTY", "1.0")),
                "frequency_penalty": float(os.getenv("VLLM_FREQUENCY_PENALTY", "0.0")),
                "presence_penalty": float(os.getenv("VLLM_PRESENCE_PENALTY", "0.0")),
            },
            "performance": {
                "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "16384")),
                "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.95")),
                "tensor_parallel_size": int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
                "pipeline_parallel_size": int(os.getenv("VLLM_PIPELINE_PARALLEL_SIZE", "1")),
                "data_parallel_size": int(os.getenv("VLLM_DATA_PARALLEL_SIZE", "1")),
                "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", "16")),
                "max_num_batched_tokens": int(os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", "8192")),
                "kv_cache_dtype": os.getenv("VLLM_KV_CACHE_DTYPE", "fp8"),
                "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "true").lower() == "true",
                "enable_chunked_prefill": os.getenv("VLLM_ENABLE_CHUNKED_PREFILL", "true").lower() == "true",
                "async_scheduling": os.getenv("VLLM_ASYNC_SCHEDULING", "true").lower() == "true",
            },
            "moe": {
                "moe_backend": os.getenv("VLLM_MOE_BACKEND", "marlin"),
                "mamba_ssm_cache_dtype": os.getenv("VLLM_MAMBA_SSM_CACHE_DTYPE", "float16"),
            },
            "reasoning": {
                "reasoning_parser": os.getenv("VLLM_REASONING_PARSER", "nemotron_v3"),
                "reasoning_parser_plugin": os.getenv("VLLM_REASONING_PARSER_PLUGIN", ""),
            },
            "speculative": {
                "method": os.getenv("VLLM_SPECULATIVE_METHOD", ""),
                "num_speculative_tokens": int(os.getenv("VLLM_SPECULATIVE_TOKENS", "0")),
                "speculative_moe_backend": os.getenv("VLLM_SPECULATIVE_MOE_BACKEND", ""),
            },
            "tools": {
                "enable_auto_tool_choice": True,
                "tool_call_parser": os.getenv("VLLM_TOOL_CALL_PARSER", "qwen3_coder"),
            },
            "media": {
                "video_pruning_rate": float(os.getenv("VLLM_VIDEO_PRUNING_RATE", "0.5")),
                "video_fps": int(os.getenv("VLLM_VIDEO_FPS", "2")),
                "video_num_frames": int(os.getenv("VLLM_VIDEO_NUM_FRAMES", "256")),
            },
            "environment": {
                "vllm_nvfp4_gemm_backend": os.getenv("VLLM_NVFP4_GEMM_BACKEND", "marlin"),
                "vllm_allow_long_max_model_len": os.getenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1"),
                "vllm_flashinfer_allreduce_backend": os.getenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "trtllm"),
                "vllm_use_flashinfer_moe_fp4": os.getenv("VLLM_USE_FLASHINFER_MOE_FP4", "0"),
                "hf_token": os.getenv("HF_TOKEN", ""),
            },
            "server": {
                "host": os.getenv("VLLM_HOST", "0.0.0.0"),
                "port": int(os.getenv("VLLM_PORT", "8080")),
                "api_key": os.getenv("API_KEY", "placeholder-api-key"),
                "trust_remote_code": True,
            },
        }

    def build_command(self) -> list:
        """Build the vLLM launch command from the current config.

        Returns the command as a list of tokens (matching LlamaCPPManager
        convention) suitable for display in the DeployPage command preview.
        """
        cfg = self.config
        model = cfg.get("model", {})
        sampling = cfg.get("sampling", {})
        perf = cfg.get("performance", {})
        moe = cfg.get("moe", {})
        reasoning = cfg.get("reasoning", {})
        speculative = cfg.get("speculative", {})
        tools = cfg.get("tools", {})
        media = cfg.get("media", {})
        server = cfg.get("server", {})

        cmd = [
            "vllm", "serve", model.get("name", ""),
            "--served-model-name", model.get("served_name", ""),
            "--host", str(server.get("host", "0.0.0.0")),
            "--port", str(server.get("port", 8080)),
            "--api-key", server.get("api_key", ""),
            "--dtype", model.get("dtype", "auto"),
            "--max-model-len", str(perf.get("max_model_len", 16384)),
            "--gpu-memory-utilization", str(perf.get("gpu_memory_utilization", 0.95)),
            "--tensor-parallel-size", str(perf.get("tensor_parallel_size", 1)),
            "--kv-cache-dtype", perf.get("kv_cache_dtype", "fp8"),
            "--max-num-seqs", str(perf.get("max_num_seqs", 16)),
            "--max-num-batched-tokens", str(perf.get("max_num_batched_tokens", 8192)),
        ]

        if server.get("trust_remote_code", True):
            cmd.append("--trust-remote-code")

        if perf.get("enforce_eager", True):
            cmd.append("--enforce-eager")

        # Pipeline parallel
        pp = perf.get("pipeline_parallel_size", 1)
        if pp and pp > 1:
            cmd.extend(["--pipeline-parallel-size", str(pp)])

        # Data parallel
        dp = perf.get("data_parallel_size", 1)
        if dp and dp > 1:
            cmd.extend(["--data-parallel-size", str(dp)])

        # Chunked prefill
        if perf.get("enable_chunked_prefill", True):
            cmd.append("--enable-chunked-prefill")

        # Async scheduling
        if perf.get("async_scheduling", True):
            cmd.append("--async-scheduling")

        # MoE backend
        moe_backend = moe.get("moe_backend", "")
        if moe_backend:
            cmd.extend(["--moe-backend", moe_backend])

        mamba_dtype = moe.get("mamba_ssm_cache_dtype", "")
        if mamba_dtype:
            cmd.extend(["--mamba_ssm_cache_dtype", mamba_dtype])

        # Quantization
        quant = model.get("quantization", "")
        if quant and str(quant).strip().lower() not in ("", "none"):
            cmd.extend(["--quantization", quant])

        # Reasoning
        rp = reasoning.get("reasoning_parser", "")
        rp_lc = str(rp).strip().lower() if rp else ""
        if rp and rp_lc != "none":
            cmd.extend(["--reasoning-parser", rp])
        rp_plugin = reasoning.get("reasoning_parser_plugin", "")
        if rp_plugin:
            cmd.extend(["--reasoning-parser-plugin", rp_plugin])

        # Tools
        if tools.get("enable_auto_tool_choice", True):
            cmd.append("--enable-auto-tool-choice")
        tcp = tools.get("tool_call_parser", "")
        if tcp:
            cmd.extend(["--tool-call-parser", tcp])

        # Speculative decoding
        spec_method = str(speculative.get("method") or "").strip()
        if spec_method:
            spec_config = {"method": spec_method}
            spec_tokens = speculative.get("num_speculative_tokens", 0)
            if spec_tokens:
                spec_config["num_speculative_tokens"] = spec_tokens
            spec_moe = speculative.get("speculative_moe_backend", "")
            if spec_moe:
                spec_config["moe_backend"] = spec_moe
            cmd.extend(["--speculative_config", json.dumps(spec_config)])

        # Media
        if media.get("video_pruning_rate") is not None:
            cmd.extend(["--video-pruning-rate", str(media["video_pruning_rate"])])

        # Sampling parameters (applied at request time, shown for reference)
        sampling_note_parts = []
        temp = sampling.get("temperature")
        if temp is not None:
            sampling_note_parts.append(f"temperature={temp}")
        top_p = sampling.get("top_p")
        if top_p is not None:
            sampling_note_parts.append(f"top_p={top_p}")
        rep_pen = sampling.get("repetition_penalty")
        if rep_pen is not None and rep_pen != 1.0:
            sampling_note_parts.append(f"repetition_penalty={rep_pen}")
        freq_pen = sampling.get("frequency_penalty")
        if freq_pen is not None and freq_pen != 0.0:
            sampling_note_parts.append(f"frequency_penalty={freq_pen}")
        pres_pen = sampling.get("presence_penalty")
        if pres_pen is not None and pres_pen != 0.0:
            sampling_note_parts.append(f"presence_penalty={pres_pen}")

        if sampling_note_parts:
            cmd.extend(["# sampling:", ", ".join(sampling_note_parts)])

        return cmd

    def get_field_metadata(self) -> Dict[str, Any]:
        """Return metadata describing which config fields exist for vLLM,
        their scope, and cross-framework mapping information.

        Each field has:
          - scope: "shared" | "vllm" | "llamacpp"
          - type: "number" | "text" | "select" | "boolean"
          - description: human-readable description
          - vllm_flag: the CLI flag for vLLM
          - llamacpp_equivalent: what it maps to in llama.cpp (if any)
          - min/max/step: for number fields
          - options: for select fields
        """
        return {
            "model": {
                "name": {"scope": "shared", "type": "text", "description": "Model name or HuggingFace repo ID", "vllm_flag": "--model", "llamacpp_equivalent": "model.name"},
                "served_name": {"scope": "vllm", "type": "text", "description": "Name exposed to clients via API", "vllm_flag": "--served-model-name", "llamacpp_equivalent": None},
                "dtype": {"scope": "vllm", "type": "select", "description": "Data type for model weights", "vllm_flag": "--dtype", "options": ["auto", "float16", "bfloat16", "float32"], "llamacpp_equivalent": None},
                "quantization": {"scope": "vllm", "type": "select", "description": "Quantization method for the model", "vllm_flag": "--quantization", "options": ["fp4", "awq", "gptq", "bitsandbytes", "None"], "llamacpp_equivalent": None, "note": "llama.cpp uses GGUF quantization variants instead"},
            },
            "sampling": {
                "temperature": {"scope": "shared", "type": "number", "description": "Controls randomness in text generation. Higher values produce more creative output, lower values produce more focused and deterministic text.", "vllm_flag": "per-request", "llamacpp_flag": "--temp", "min": 0, "max": 2, "step": 0.1},
                "top_p": {"scope": "shared", "type": "number", "description": "Nucleus sampling: limits next token selection to a subset with cumulative probability above threshold.", "vllm_flag": "per-request", "llamacpp_flag": "--top-p", "min": 0, "max": 1, "step": 0.05},
                "top_k": {"scope": "shared", "type": "number", "description": "Limits next token selection to the K most probable tokens. -1 to disable.", "vllm_flag": "per-request", "llamacpp_flag": "--top-k", "min": -1, "max": 200},
                "repetition_penalty": {"scope": "shared", "type": "number", "description": "Penalizes repeated tokens. Values > 1.0 discourage repetition.", "vllm_flag": "per-request", "llamacpp_flag": "--repeat-penalty", "min": 0.1, "max": 2.0, "step": 0.1, "mapping_note": "llama.cpp: repeat_penalty, vLLM: repetition_penalty"},
                "frequency_penalty": {"scope": "shared", "type": "number", "description": "Penalizes tokens based on their frequency in the text so far.", "vllm_flag": "per-request", "llamacpp_flag": "--frequency-penalty", "min": -2.0, "max": 2.0, "step": 0.1},
                "presence_penalty": {"scope": "shared", "type": "number", "description": "Penalizes tokens that have already appeared, encouraging new topics.", "vllm_flag": "per-request", "llamacpp_flag": "--presence-penalty", "min": -2.0, "max": 2.0, "step": 0.1},
            },
            "performance": {
                "max_model_len": {"scope": "vllm", "type": "number", "description": "Maximum model context length (sequence length). Equivalent to context_size in llama.cpp. vLLM will allocate KV cache for this length.", "vllm_flag": "--max-model-len", "llamacpp_equivalent": "model.context_size", "mapping_note": "llama.cpp: context_size, vLLM: max_model_len"},
                "gpu_memory_utilization": {"scope": "vllm", "type": "number", "description": "Fraction of GPU memory to use (0.0-1.0). vLLM pre-allocates this fraction. llama.cpp has no direct equivalent -- it offloads layers until VRAM is full.", "vllm_flag": "--gpu-memory-utilization", "llamacpp_equivalent": None, "min": 0.0, "max": 1.0, "step": 0.05},
                "tensor_parallel_size": {"scope": "vllm", "type": "number", "description": "Number of GPUs for tensor parallelism (splitting layers across GPUs). llama.cpp uses split_mode + tensor_split instead.", "vllm_flag": "--tensor-parallel-size", "llamacpp_equivalent": "performance.split_mode", "min": 1, "max": 16},
                "pipeline_parallel_size": {"scope": "vllm", "type": "number", "description": "Number of pipeline parallel stages. Splits model across GPUs by layers.", "vllm_flag": "--pipeline-parallel-size", "llamacpp_equivalent": None, "min": 1, "max": 16},
                "data_parallel_size": {"scope": "vllm", "type": "number", "description": "Number of data parallel replicas. Replicates the model for higher throughput.", "vllm_flag": "--data-parallel-size", "llamacpp_equivalent": "performance.parallel_slots", "mapping_note": "llama.cpp: parallel_slots, vLLM: data_parallel_size"},
                "max_num_seqs": {"scope": "vllm", "type": "number", "description": "Maximum number of sequences per iteration. Controls batching.", "vllm_flag": "--max-num-seqs", "llamacpp_equivalent": "performance.parallel_slots"},
                "max_num_batched_tokens": {"scope": "vllm", "type": "number", "description": "Maximum number of tokens per batch.", "vllm_flag": "--max-num-batched-tokens", "llamacpp_equivalent": "performance.batch_size"},
                "kv_cache_dtype": {"scope": "vllm", "type": "select", "description": "Data type for KV cache. llama.cpp uses cache_type_k/cache_type_v (separate for K and V).", "vllm_flag": "--kv-cache-dtype", "options": ["auto", "fp8", "fp8_e5m2", "fp8_e4m3"], "llamacpp_equivalent": "performance.cache_type_k", "mapping_note": "llama.cpp has separate K and V cache types, vLLM uses one setting"},
                "enforce_eager": {"scope": "vllm", "type": "boolean", "description": "Disable CUDA graphs (use eager mode). Useful for debugging.", "vllm_flag": "--enforce-eager", "llamacpp_equivalent": None},
                "enable_chunked_prefill": {"scope": "vllm", "type": "boolean", "description": "Enable chunked prefill for better scheduling of long prompts.", "vllm_flag": "--enable-chunked-prefill", "llamacpp_equivalent": None},
                "async_scheduling": {"scope": "vllm", "type": "boolean", "description": "Enable async scheduling for improved throughput.", "vllm_flag": "--async-scheduling", "llamacpp_equivalent": None},
            },
            "moe": {
                "moe_backend": {"scope": "vllm", "type": "select", "description": "Backend for MoE (Mixture of Experts) computation.", "vllm_flag": "--moe-backend", "options": ["marlin", "triton"], "llamacpp_equivalent": None},
                "mamba_ssm_cache_dtype": {"scope": "vllm", "type": "select", "description": "Data type for Mamba SSM cache.", "vllm_flag": "--mamba_ssm_cache_dtype", "options": ["float16", "float32", "bfloat16"], "llamacpp_equivalent": None},
            },
            "reasoning": {
                "reasoning_parser": {"scope": "vllm", "type": "select", "description": "Parser for extracting reasoning/thinking content from model output.", "vllm_flag": "--reasoning-parser", "options": ["nemotron_v3", "deepseek_r1", "None"], "llamacpp_equivalent": "server.reasoning_format", "mapping_note": "llama.cpp: reasoning_format, vLLM: reasoning_parser"},
                "reasoning_parser_plugin": {"scope": "vllm", "type": "text", "description": "Path to custom reasoning parser plugin Python file.", "vllm_flag": "--reasoning-parser-plugin", "llamacpp_equivalent": None},
            },
            "speculative": {
                "method": {"scope": "vllm", "type": "select", "description": "Speculative decoding method.", "vllm_flag": "--speculative_config.method", "options": ["mtp", "ngram", "eagle", "medusa", ""], "llamacpp_equivalent": "speculative.model_draft", "mapping_note": "llama.cpp uses draft model, vLLM uses methods like MTP"},
                "num_speculative_tokens": {"scope": "vllm", "type": "number", "description": "Number of speculative tokens to generate.", "vllm_flag": "--speculative_config.num_speculative_tokens", "llamacpp_equivalent": "speculative.draft_max", "min": 0, "max": 16},
            },
            "tools": {
                "enable_auto_tool_choice": {"scope": "vllm", "type": "boolean", "description": "Enable automatic tool choice for function calling.", "vllm_flag": "--enable-auto-tool-choice", "llamacpp_equivalent": None},
                "tool_call_parser": {"scope": "vllm", "type": "select", "description": "Parser for extracting tool calls from model output.", "vllm_flag": "--tool-call-parser", "options": ["qwen3_coder", "hermes", "mistral", "internlm", "llama3_json", ""], "llamacpp_equivalent": None},
            },
            "environment": {
                "vllm_nvfp4_gemm_backend": {"scope": "vllm", "type": "select", "description": "Backend for NVFP4 GEMM operations.", "vllm_flag": "env:VLLM_NVFP4_GEMM_BACKEND", "options": ["marlin", "triton", "cutlass"], "llamacpp_equivalent": None},
                "vllm_allow_long_max_model_len": {"scope": "vllm", "type": "select", "description": "Allow max_model_len beyond recommended limit.", "vllm_flag": "env:VLLM_ALLOW_LONG_MAX_MODEL_LEN", "options": ["0", "1"], "llamacpp_equivalent": None},
                "vllm_flashinfer_allreduce_backend": {"scope": "vllm", "type": "select", "description": "Backend for FlashInfer all-reduce operations.", "vllm_flag": "env:VLLM_FLASHINFER_ALLREDUCE_BACKEND", "options": ["trtllm", "custom"], "llamacpp_equivalent": None},
                "vllm_use_flashinfer_moe_fp4": {"scope": "vllm", "type": "select", "description": "Use FlashInfer MoE for FP4 quantization.", "vllm_flag": "env:VLLM_USE_FLASHINFER_MOE_FP4", "options": ["0", "1"], "llamacpp_equivalent": None},
                "hf_token": {"scope": "shared", "type": "text", "description": "HuggingFace API token for gated model access.", "vllm_flag": "env:HF_TOKEN", "llamacpp_equivalent": None},
            },
            "server": {
                "host": {"scope": "shared", "type": "text", "description": "IP address to listen on.", "vllm_flag": "--host", "llamacpp_flag": "--host"},
                "port": {"scope": "shared", "type": "number", "description": "Port to listen on.", "vllm_flag": "--port", "llamacpp_flag": "--port"},
                "api_key": {"scope": "shared", "type": "text", "description": "API key for authentication.", "vllm_flag": "--api-key", "llamacpp_flag": "--api-key"},
                "trust_remote_code": {"scope": "vllm", "type": "boolean", "description": "Allow remote code execution from model repo.", "vllm_flag": "--trust-remote-code", "llamacpp_equivalent": None},
            },
        }

    def compose_launch_environment(self) -> Dict[str, str]:
        """Env merged into ``docker compose`` when starting/recreating ``vllm-api``.

        Keys align with ``docker-compose.yml`` substitutions and ``scripts/start-vllm.sh``.
        """
        cfg = self.config
        model = cfg.get("model") or {}
        perf = cfg.get("performance") or {}
        server = cfg.get("server") or {}
        moe = cfg.get("moe") or {}
        reasoning = cfg.get("reasoning") or {}
        speculative = cfg.get("speculative") or {}
        tools = cfg.get("tools") or {}
        media = cfg.get("media") or {}
        env_section = cfg.get("environment") or {}

        def tf_perf(key: str, default: bool = True) -> str:
            v = perf.get(key)
            if v is None:
                return "true" if default else "false"
            return "true" if bool(v) else "false"

        def tf_server(key: str, default: bool = True) -> str:
            v = server.get(key)
            if v is None:
                return "true" if default else "false"
            return "true" if bool(v) else "false"

        def tf_tools(key: str, default: bool = True) -> str:
            v = tools.get(key)
            if v is None:
                return "true" if default else "false"
            return "true" if bool(v) else "false"

        gpu_mu = perf.get("gpu_memory_utilization")
        if gpu_mu is None:
            gpu_mu = 0.95

        port_raw = server.get("port", 8080)
        try:
            port_int = int(port_raw)
        except (TypeError, ValueError):
            port_int = 8080

        out: Dict[str, str] = {
            "MODEL_NAME": str(model.get("name") or "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"),
            "SERVED_MODEL_NAME": str(model.get("served_name") or "Nemotron-3-Nano-Omni-30B-A3B-Reasoning"),
            "HOST": str(server.get("host") or "0.0.0.0"),
            "PORT": str(port_int),
            "API_KEY": str(server.get("api_key") or "placeholder-api-key"),
            "MAX_MODEL_LEN": str(int(perf.get("max_model_len") or 16384)),
            "GPU_MEMORY_UTILIZATION": str(gpu_mu),
            "TENSOR_PARALLEL_SIZE": str(int(perf.get("tensor_parallel_size") or 1)),
            "PIPELINE_PARALLEL_SIZE": str(int(perf.get("pipeline_parallel_size") or 1)),
            "DATA_PARALLEL_SIZE": str(int(perf.get("data_parallel_size") or 1)),
            "KV_CACHE_DTYPE": str(perf.get("kv_cache_dtype") or "fp8"),
            "MAX_NUM_SEQS": str(int(perf.get("max_num_seqs") or 16)),
            "MAX_NUM_BATCHED_TOKENS": str(int(perf.get("max_num_batched_tokens") or 8192)),
            "VLLM_DTYPE": str(model.get("dtype") or "auto"),
            "VLLM_QUANTIZATION": str(model.get("quantization") or ""),
            "REASONING_PARSER": str(reasoning.get("reasoning_parser") or ""),
            "REASONING_PARSER_PLUGIN": str(reasoning.get("reasoning_parser_plugin") or ""),
            "TOOL_CALL_PARSER": str(tools.get("tool_call_parser") or ""),
            "VIDEO_PRUNING_RATE": str(
                media["video_pruning_rate"]
                if media.get("video_pruning_rate") is not None
                else 0.5
            ),
            "VIDEO_FPS": str(int(media.get("video_fps") or 2)),
            "VIDEO_NUM_FRAMES": str(int(media.get("video_num_frames") or 256)),
            "MOE_BACKEND": str(moe.get("moe_backend") or ""),
            "MAMBA_SSM_CACHE_DTYPE": str(moe.get("mamba_ssm_cache_dtype") or ""),
            "VLLM_ENFORCE_EAGER": tf_perf("enforce_eager", True),
            "VLLM_ENABLE_CHUNKED_PREFILL": tf_perf("enable_chunked_prefill", True),
            "VLLM_ASYNC_SCHEDULING": tf_perf("async_scheduling", True),
            "TRUST_REMOTE_CODE": tf_server("trust_remote_code", True),
            "VLLM_ENABLE_AUTO_TOOL_CHOICE": tf_tools("enable_auto_tool_choice", True),
            "VLLM_NVFP4_GEMM_BACKEND": str(env_section.get("vllm_nvfp4_gemm_backend") or "marlin"),
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": str(env_section.get("vllm_allow_long_max_model_len") or "1"),
            "VLLM_FLASHINFER_ALLREDUCE_BACKEND": str(env_section.get("vllm_flashinfer_allreduce_backend") or "trtllm"),
            "VLLM_USE_FLASHINFER_MOE_FP4": str(env_section.get("vllm_use_flashinfer_moe_fp4") or "0"),
        }

        spec_method = str(speculative.get("method") or "").strip()
        if spec_method:
            spec_cfg: Dict[str, Any] = {"method": spec_method}
            ntok = speculative.get("num_speculative_tokens") or 0
            try:
                ntok_int = int(ntok)
            except (TypeError, ValueError):
                ntok_int = 0
            if ntok_int:
                spec_cfg["num_speculative_tokens"] = ntok_int
            smoe = str(speculative.get("speculative_moe_backend") or "").strip()
            if smoe:
                spec_cfg["moe_backend"] = smoe
            out["VLLM_SPECULATIVE_CONFIG"] = json.dumps(spec_cfg, separators=(",", ":"))
        else:
            out["VLLM_SPECULATIVE_CONFIG"] = ""

        hf = str(env_section.get("hf_token") or "").strip()
        if hf:
            out["HUGGINGFACE_TOKEN"] = hf
            out["HF_TOKEN"] = hf

        return out

    async def start(self) -> bool:
        """Start the vLLM container.

        Prefer ``docker compose up --force-recreate`` from the project directory
        so the merged ``compose_launch_environment()`` matches ``VLLMManager.config``.
        Falls back to ``docker start`` when no compose file is available (legacy setups).
        """
        compose_err: Optional[str] = None
        try:
            logger.info("Starting vLLM service...")
            project_dir = os.getenv("PROJECT_DIR", "/home/alec/git/llama-nexus")
            compose_path = os.path.join(project_dir, "docker-compose.yml")
            launch_env = {**os.environ, **self.compose_launch_environment()}

            if os.path.isfile(compose_path):
                compose_cmd = [
                    "docker",
                    "compose",
                    "-f",
                    compose_path,
                    "--profile",
                    "vllm",
                    "up",
                    "-d",
                    "--force-recreate",
                    "vllm-api",
                ]
                proc = subprocess.run(
                    compose_cmd,
                    cwd=project_dir,
                    env=launch_env,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if proc.returncode == 0:
                    self.start_time = datetime.now()
                    self.last_action_error = None
                    logger.info(
                        "vLLM container started via docker compose "
                        f"(MODEL_NAME={launch_env.get('MODEL_NAME')}, PORT={launch_env.get('PORT')})"
                    )
                    self._start_log_reader()
                    return True
                compose_err = (proc.stderr or proc.stdout or "").strip() or f"exit code {proc.returncode}"
                logger.warning("docker compose failed (%s), trying docker start...", compose_err[:500])

            proc = subprocess.run(
                ["docker", "start", self.container_name],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode == 0:
                self.start_time = datetime.now()
                if compose_err:
                    self.last_action_error = (
                        "docker compose did not apply Deploy env; container started with docker start only — "
                        "settings may be stale until compose succeeds. Compose output: "
                        + compose_err[:4000]
                    )
                    logger.warning(self.last_action_error)
                else:
                    self.last_action_error = None
                logger.info("vLLM container started via docker start (compose unavailable or failed)")
                self._start_log_reader()
                return True

            tail = (proc.stderr or proc.stdout or "").strip()
            msg = tail or "docker start failed"
            if compose_err:
                msg = f"compose: {compose_err[:2000]}; docker start: {msg}"
            self.last_action_error = msg[:8000]
            logger.error(f"Failed to start vLLM: {msg}")
            return False
        except Exception as e:
            self.last_action_error = str(e)[:8000]
            logger.error(f"Failed to start vLLM: {e}")
            return False

    async def stop(self) -> bool:
        """Stop the vLLM container."""
        try:
            logger.info("Stopping vLLM service...")
            proc = subprocess.run(
                ["docker", "stop", self.container_name],
                capture_output=True, text=True, timeout=60,
            )
            self.start_time = None
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip() or "docker stop failed"
                self.last_action_error = err[:8000]
                logger.error("Failed to stop vLLM: %s", err)
                return False
            self.last_action_error = None
            return True
        except Exception as e:
            self.last_action_error = str(e)[:8000]
            logger.error(f"Failed to stop vLLM: {e}")
            return False

    async def restart(self) -> bool:
        """Restart the vLLM container."""
        await self.stop()
        await asyncio.sleep(2)
        return await self.start()

    def is_running(self) -> bool:
        """Check if the vLLM container is running."""
        try:
            if DOCKER_AVAILABLE and docker_client:
                container = docker_client.containers.get(self.container_name)
                return container.status == "running"
            else:
                result = subprocess.run(
                    ["docker", "ps", "--filter", f"name={self.container_name}",
                     "--format", "{{.Names}}"],
                    capture_output=True, text=True, timeout=5,
                )
                return self.container_name in result.stdout.strip()
        except Exception:
            return False

    async def get_health(self) -> Dict[str, Any]:
        """Check health of the vLLM service."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.internal_http_base()}/health", timeout=5.0,
                )
                return {
                    "healthy": response.status_code == 200,
                    "status_code": response.status_code,
                    "response": response.text,
                }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the vLLM service."""
        running = self.is_running()
        return {
            "running": running,
            "backend": "vllm",
            "container": self.container_name,
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time and running else 0,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_action_error": self.last_action_error,
            "config": self.config,
            "model": {
                "name": self.config.get("model", {}).get("name"),
                "served_name": self.config.get("model", {}).get("served_name"),
                "max_model_len": self.config.get("performance", {}).get("max_model_len"),
            },
        }

    def get_logs(self, lines: int = 100) -> list:
        """Get recent log lines from the buffer."""
        try:
            lines = int(lines)
        except (ValueError, TypeError):
            lines = 100
        return list(self.log_buffer)[-max(0, lines):]

    async def add_log_line(self, line: str):
        """Add a log line to the buffer and broadcast to websocket clients."""
        log_entry = {
            "message": line,
            "timestamp": datetime.now().isoformat(),
        }
        self.log_buffer.append(log_entry)

        disconnected = []
        for client in self.websocket_clients:
            try:
                await client.send_json({
                    "type": "log",
                    "data": line,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception:
                disconnected.append(client)
        for client in disconnected:
            self.websocket_clients.remove(client)

    def _start_log_reader(self):
        """Start the background docker log reader task."""
        if self._log_reader_task and not self._log_reader_task.done():
            self._log_reader_task.cancel()
        self._log_reader_task = asyncio.create_task(self._read_docker_logs())
        logger.info("Started vLLM docker log reader task")

    async def _read_docker_logs(self):
        """Read logs from the vLLM docker container in the background."""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "logs", "-f", "--tail", "100", self.container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            logger.info(f"Reading docker logs for {self.container_name}")

            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if line_str:
                    await self.add_log_line(line_str)
        except asyncio.CancelledError:
            logger.info("vLLM log reader task cancelled")
        except Exception as e:
            logger.error(f"Error reading vLLM docker logs: {e}")
