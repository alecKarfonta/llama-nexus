"""
Built-in one-click llama.cpp deployment presets for conversation workloads.
Mirrors frontend/src/config/llamacppDeployPresets.ts
"""
from typing import Any, Dict, List

QWEN36_CHAT_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 1.5,
    "frequency_penalty": 0.0,
    "dry_multiplier": 0,
    "dry_base": 2.0,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": 0,
}

QWEN36_RP_SAMPLING = {
    "temperature": 0.85,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.05,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "dry_multiplier": 0.8,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": -1,
}

QWEN36_CODER_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "dry_multiplier": 0,
}


def _conversation_runtime(**overrides: Dict[str, Any]) -> Dict[str, Any]:
    model_overrides = overrides.pop("model", {})
    performance_overrides = overrides.pop("performance", {})
    server_overrides = overrides.pop("server", {})
    sampling = overrides.pop("sampling", None)

    config: Dict[str, Any] = {
        "model": {
            "context_size": 32768,
            "gpu_layers": 999,
            "n_cpu_moe": 0,
            "flash_attn": "auto",
            **model_overrides,
        },
        "template": {"selected": ""},
        "performance": {
            "threads": 22,
            "batch_size": 1024,
            "ubatch_size": 512,
            "num_keep": 1024,
            "num_predict": 1024,
            "parallel_slots": 1,
            "split_mode": "layer",
            "main_gpu": 2,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            **performance_overrides,
        },
        "server": {
            "reasoning_budget": 0,
            "cache_reuse": 256,
            "jinja": True,
            "metrics": True,
            **server_overrides,
        },
        "speculative": {},
    }
    if sampling is not None:
        config["sampling"] = sampling
    config.update(overrides)
    return config


LLAMACPP_DEPLOY_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "chat-vtuber",
        "name": "Chat · VTuber",
        "description": (
            "Qwen3.6-35B-A3B uncensored heretic Q8_0 — personality-driven chat, "
            "low latency, presence penalty anti-repeat."
        ),
        "category": "conversation",
        "config": _conversation_runtime(
            model={
                "name": "Qwen3.6-35B-A3B-uncensored-heretic",
                "variant": "Q8_0",
            },
            sampling=QWEN36_CHAT_SAMPLING,
        ),
    },
    {
        "id": "chat-assistant",
        "name": "Chat · Assistant",
        "description": (
            "Qwen3.6-35B-A3B official Q8_0 — accurate instruction-following assistant chat."
        ),
        "category": "assistant",
        "config": _conversation_runtime(
            model={"name": "Qwen3.6-35B-A3B", "variant": "Q8_0"},
            sampling=QWEN36_CHAT_SAMPLING,
        ),
    },
    {
        "id": "chat-rp",
        "name": "Chat · RP / Character",
        "description": (
            "Uncensored heretic Q8_0 with min-p + DRY sampling for long persona / RP sessions."
        ),
        "category": "rp",
        "config": _conversation_runtime(
            model={
                "name": "Qwen3.6-35B-A3B-uncensored-heretic",
                "variant": "Q8_0",
            },
            sampling=QWEN36_RP_SAMPLING,
        ),
    },
    {
        "id": "chat-multi-platform",
        "name": "Chat · Multi-Platform",
        "description": (
            "One model, 4 parallel slots (~16K ctx each) for per-platform bot personalities."
        ),
        "category": "multi",
        "config": _conversation_runtime(
            model={
                "name": "Qwen3.6-35B-A3B-uncensored-heretic",
                "variant": "Q8_0",
                "context_size": 65536,
            },
            performance={"parallel_slots": 4},
            sampling=QWEN36_CHAT_SAMPLING,
        ),
    },
    {
        "id": "chat-fast",
        "name": "Chat · Fast / Side Bot",
        "description": (
            "Qwen3-4B-Instruct Q8_0 on a single GPU — minimal VRAM for a side endpoint."
        ),
        "category": "fast",
        "config": _conversation_runtime(
            model={
                "name": "Qwen3-4B-Instruct",
                "variant": "Q8_0",
                "context_size": 16384,
            },
            performance={
                "num_predict": 512,
                "num_keep": 512,
                "split_mode": "none",
                "main_gpu": 0,
            },
            sampling=QWEN36_CHAT_SAMPLING,
        ),
    },
]

SAMPLING_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "chat-qwen36",
        "name": "Chat (Qwen3.6)",
        "description": "Qwen3.6 non-thinking conversation sampler",
        "category": "conversation",
        "config": {"sampling": QWEN36_CHAT_SAMPLING},
    },
    {
        "id": "coding-qwen3-coder",
        "name": "Coding (Qwen3-Coder)",
        "description": "Qwen3-Coder code generation sampler",
        "category": "coding",
        "config": {"sampling": QWEN36_CODER_SAMPLING},
    },
    {
        "id": "rp-dry",
        "name": "RP / Character (DRY)",
        "description": "Long character sessions with min-p + DRY",
        "category": "rp",
        "config": {"sampling": QWEN36_RP_SAMPLING},
    },
]
