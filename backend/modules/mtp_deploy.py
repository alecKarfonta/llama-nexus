"""MTP deployment validation and env helpers for llama.cpp."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from modules.llamacpp_build_info import MTP_MIN_BUILD_NUMBER, mtp_supported_for_build


def mtp_enabled_in_config(config: Dict[str, Any]) -> bool:
    mtp = config.get("mtp") or {}
    return bool(mtp.get("enabled"))


def normalize_mtp_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return MTP block with defaults applied."""
    raw = config.get("mtp") or {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "draft_n_max": int(raw.get("draft_n_max", 3)),
        "draft_n_min": int(raw.get("draft_n_min", 0)),
        "draft_p_min": float(raw.get("draft_p_min", 0.75)),
    }


def mtp_env_from_config(config: Dict[str, Any]) -> Dict[str, str]:
    """Compose-driven env mirror of the MTP config block."""
    mtp = normalize_mtp_config(config)
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

    if not mtp_enabled_in_config(config):
        return errors, warnings

    mtp = normalize_mtp_config(config)

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

    parallel = int((config.get("performance") or {}).get("parallel_slots") or 1)
    if parallel > 1:
        warnings.append(
            f"MTP is enabled with parallel_slots={parallel}. "
            "Multi-slot deployments often reduce aggregate throughput with MTP; "
            "use parallel_slots=1 for single-stream latency workloads."
        )

    return errors, warnings
