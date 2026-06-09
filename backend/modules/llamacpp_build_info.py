"""Helpers for llama.cpp container build tags (MTP gating, status API)."""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

# First upstream release with merged MTP (ggml-org/llama.cpp PR #22673).
MTP_MIN_BUILD_NUMBER = 9193


def parse_llamacpp_build_number(tag: Optional[str]) -> Optional[int]:
    """Parse numeric build from tags like b9193 or build 9193."""
    if not tag:
        return None
    text = str(tag).strip()
    match = re.search(r"\bb?(\d{4,5})\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def mtp_supported_for_build(build_number: Optional[int]) -> bool:
    return build_number is not None and build_number >= MTP_MIN_BUILD_NUMBER


def parse_llguidance_enabled(raw: Optional[str]) -> bool:
    if raw is None:
        return False
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def build_info_from_tag_and_version(
    tag: Optional[str],
    cli_version: Optional[str] = None,
    *,
    llguidance_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    """Normalized build payload for service status responses."""
    build_number = parse_llamacpp_build_number(tag)
    if build_number is None and cli_version:
        build_number = parse_llamacpp_build_number(cli_version)
    return {
        "tag": tag,
        "build_number": build_number,
        "cli_version": (cli_version or "").strip() or None,
        "mtp_supported": mtp_supported_for_build(build_number),
        "mtp_min_build_number": MTP_MIN_BUILD_NUMBER,
        "grammar_gbnf_supported": True,
        "llguidance_enabled": bool(llguidance_enabled),
    }
