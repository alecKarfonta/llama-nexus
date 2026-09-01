"""Helpers for choosing host-reachable inference endpoints."""

from __future__ import annotations

import os
from pathlib import Path


def _clean_base_url(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text.rstrip("/")


def _env_flag(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def running_inside_container() -> bool:
    return (
        _env_flag("RUNNING_IN_DOCKER")
        or _env_flag("USE_DOCKER")
        or Path("/.dockerenv").exists()
    )


def _dedupe(urls: list[str | None]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for url in urls:
        cleaned = _clean_base_url(url)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def vllm_proxy_base_url_candidates(internal_base_url: str | None = None) -> list[str]:
    """Return vLLM base URLs in the order the backend should try them.

    In Docker, the internal service name is normally correct. In the desktop
    host runtime, that name does not resolve; the vLLM container is commonly
    published on 8601 and should be tried before the internal Docker DNS name.
    """
    explicit = [
        os.getenv("VLLM_PROXY_BASE_URL"),
        os.getenv("INFERENCE_BACKEND_URL"),
    ]
    host_local = [] if running_inside_container() else [
        "http://127.0.0.1:8601",
        "http://localhost:8601",
    ]
    return _dedupe([*explicit, *host_local, internal_base_url])


def vllm_proxy_base_url(internal_base_url: str | None = None) -> str | None:
    candidates = vllm_proxy_base_url_candidates(internal_base_url)
    return candidates[0] if candidates else None


def has_explicit_vllm_proxy_base_url() -> bool:
    return bool(
        _clean_base_url(os.getenv("VLLM_PROXY_BASE_URL"))
        or _clean_base_url(os.getenv("INFERENCE_BACKEND_URL"))
    )
