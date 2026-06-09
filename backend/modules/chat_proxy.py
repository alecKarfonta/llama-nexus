"""Chat completion proxy helpers — passthrough contract for tool/grammar fields."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

# OpenAI-compatible fields that must reach llama-server unchanged for agent/grammar paths.
CHAT_PROXY_PASSTHROUGH_KEYS: Tuple[str, ...] = (
    "tools",
    "tool_choice",
    "response_format",
    "grammar",
)


def parse_chat_proxy_body(body: bytes) -> Dict[str, Any]:
    """Parse a proxied chat completion request body."""
    if not body:
        return {}
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("chat completion body must be a JSON object")
    return parsed


def chat_proxy_body_unchanged(body: bytes) -> bytes:
    """Proxy forwards the raw body without mutation."""
    return body


def missing_passthrough_keys(original: Dict[str, Any], forwarded: Dict[str, Any]) -> List[str]:
    """Return passthrough keys present in original but absent or altered in forwarded."""
    missing: List[str] = []
    for key in CHAT_PROXY_PASSTHROUGH_KEYS:
        if key not in original:
            continue
        if key not in forwarded:
            missing.append(key)
        elif forwarded[key] != original[key]:
            missing.append(key)
    return missing
