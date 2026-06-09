"""MTP deployment validation and env helpers for llama.cpp."""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Literal, Optional, Tuple

from modules.llamacpp_build_info import MTP_MIN_BUILD_NUMBER, mtp_supported_for_build

MtpWorkloadProfile = Literal["chat", "agent", "throughput", "custom"]
WORKLOAD_PROFILES: Tuple[MtpWorkloadProfile, ...] = (
    "chat",
    "agent",
    "throughput",
    "custom",
)
DEFAULT_WORKLOAD_PROFILE: MtpWorkloadProfile = "chat"

# Experiment-backed defaults (fast-inference/bench).
_PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "chat": {
        "mtp": {
            "enabled": True,
            "draft_n_max": 2,
            "draft_n_min": 0,
            "draft_p_min": 0.75,
        },
        "performance": {"parallel_slots": 1},
        "server": {
            "jinja": True,
            "chat_template_kwargs": {"preserve_thinking": True},
            "cache_reuse": None,
        },
    },
    "agent": {
        "mtp": {
            "enabled": True,
            "draft_n_max": 8,
            "draft_n_min": 0,
            "draft_p_min": 0.75,
        },
        "performance": {"parallel_slots": 1},
        "server": {
            "jinja": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "cache_reuse": 1024,
        },
    },
    "throughput": {
        "mtp": {
            "enabled": False,
            "draft_n_max": 2,
            "draft_n_min": 0,
            "draft_p_min": 0.75,
        },
        "performance": {"parallel_slots": 4},
        "server": {
            "jinja": True,
            "chat_template_kwargs": None,
            "cache_reuse": None,
        },
    },
}


def _normalize_workload_profile(raw: Any) -> MtpWorkloadProfile:
    if isinstance(raw, str) and raw in WORKLOAD_PROFILES:
        return raw  # type: ignore[return-value]
    return DEFAULT_WORKLOAD_PROFILE


def mtp_enabled_in_config(config: Dict[str, Any]) -> bool:
    mtp = config.get("mtp") or {}
    return bool(mtp.get("enabled"))


def normalize_mtp_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return MTP block with defaults applied."""
    raw = config.get("mtp") or {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "workload_profile": _normalize_workload_profile(raw.get("workload_profile")),
        "draft_n_max": int(raw.get("draft_n_max", 3)),
        "draft_n_min": int(raw.get("draft_n_min", 0)),
        "draft_p_min": float(raw.get("draft_p_min", 0.75)),
    }


def apply_mtp_workload_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge experiment-backed workload profile defaults into config.
    Custom profile leaves user-controlled fields unchanged.
    """
    cfg = copy.deepcopy(config)
    mtp = normalize_mtp_config(cfg)
    profile = mtp["workload_profile"]
    cfg["mtp"] = mtp

    if profile == "custom":
        return cfg

    preset = _PROFILE_DEFAULTS.get(profile)
    if not preset:
        return cfg

    mtp_block = cfg.setdefault("mtp", {})
    for key, value in preset.get("mtp", {}).items():
        mtp_block.setdefault(key, value)

    perf_block = cfg.setdefault("performance", {})
    for key, value in preset.get("performance", {}).items():
        perf_block.setdefault(key, value)

    server_preset = preset.get("server") or {}
    server_block = cfg.setdefault("server", {})
    for key, value in server_preset.items():
        if value is None:
            continue
        server_block.setdefault(key, value)

    return cfg


def chat_template_kwargs_cli_value(server: Dict[str, Any]) -> Optional[str]:
    """Serialize server.chat_template_kwargs for --chat-template-kwargs."""
    raw = server.get("chat_template_kwargs")
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        json.loads(stripped)
        return stripped
    if isinstance(raw, dict):
        return json.dumps(raw, separators=(",", ":"))
    raise ValueError("server.chat_template_kwargs must be a JSON object or string")


def mtp_env_from_config(config: Dict[str, Any]) -> Dict[str, str]:
    """Compose-driven env mirror of the MTP config block."""
    mtp = normalize_mtp_config(apply_mtp_workload_profile(config))
    return {
        "MTP_ENABLED": "true" if mtp["enabled"] else "false",
        "MTP_DRAFT_N_MAX": str(mtp["draft_n_max"]),
        "MTP_DRAFT_N_MIN": str(mtp["draft_n_min"]),
        "MTP_DRAFT_P_MIN": str(mtp["draft_p_min"]),
    }


def validate_mtp_deployment(
    config: Dict[str, Any],
    *,
    model_mtp_capable: bool,
    llamacpp_build_number: Optional[int],
    mtp_supported_build: Optional[bool] = None,
) -> Tuple[List[str], List[str]]:
    """
    Validate MTP settings. Returns (errors, warnings).
    Errors block deployment; warnings are advisory.
    """
    errors: List[str] = []
    warnings: List[str] = []

    mtp_raw = normalize_mtp_config(config)
    parallel_raw = int((config.get("performance") or {}).get("parallel_slots") or 1)
    effective = apply_mtp_workload_profile(config)
    mtp = normalize_mtp_config(effective)

    if not mtp["enabled"]:
        return errors, warnings

    if mtp_raw["workload_profile"] == "agent" and parallel_raw > 1:
        warnings.append(
            "Agent workload profile expects parallel_slots=1 for tool-call latency. "
            f"performance.parallel_slots={parallel_raw} will be overridden to 1 at deploy."
        )

    if mtp["draft_n_max"] < 1 or mtp["draft_n_max"] > 16:
        errors.append("mtp.draft_n_max must be between 1 and 16")
    if mtp["draft_n_min"] < 0 or mtp["draft_n_min"] > mtp["draft_n_max"]:
        errors.append("mtp.draft_n_min must be >= 0 and <= mtp.draft_n_max")
    if mtp["draft_p_min"] < 0.0 or mtp["draft_p_min"] > 1.0:
        errors.append("mtp.draft_p_min must be between 0.0 and 1.0")

    if not model_mtp_capable:
        errors.append(
            "MTP is enabled but the selected GGUF has no MTP prediction heads "
            "(mtp_capable=false). Use an MTP-converted GGUF or disable MTP."
        )

    build_ok = mtp_supported_build
    if build_ok is None:
        build_ok = mtp_supported_for_build(llamacpp_build_number)
    if not build_ok:
        tag_hint = (
            f"b{llamacpp_build_number}"
            if llamacpp_build_number is not None
            else "unknown"
        )
        errors.append(
            f"MTP requires llama.cpp build b{MTP_MIN_BUILD_NUMBER}+ "
            f"(current build: {tag_hint}). Rebuild llamacpp-api from Dockerfile."
        )

    parallel = int((effective.get("performance") or {}).get("parallel_slots") or 1)
    if parallel > 1:
        warnings.append(
            f"MTP is enabled with parallel_slots={parallel}. "
            "Multi-slot deployments often reduce aggregate throughput with MTP; "
            "use parallel_slots=1 for single-stream latency workloads."
        )
        if mtp["workload_profile"] == "agent":
            warnings.append(
                "Agent workload profile expects parallel_slots=1 for tool-call latency. "
                f"Current parallel_slots={parallel} may hurt tool-path throughput."
            )

    server = effective.get("server") or {}
    if server.get("chat_template_kwargs") not in (None, ""):
        try:
            chat_template_kwargs_cli_value(server)
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"server.chat_template_kwargs is invalid JSON: {exc}")

    return errors, warnings
