"""Workspace-safe runtime paths for local and container deployments."""

from __future__ import annotations

import os
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent


def _writable_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and os.access(path, os.W_OK | os.X_OK)


def backend_data_dir() -> Path:
    override = os.getenv("LLAMA_NEXUS_BACKEND_DATA_DIR")
    if override:
        return Path(override).expanduser()

    container_data = Path("/data")
    if _writable_dir(container_data):
        return container_data

    return BACKEND_ROOT / "data"


def backend_data_path(filename: str) -> Path:
    return backend_data_dir() / filename


def backend_data_subdir(name: str) -> Path:
    return backend_data_dir() / name


def app_or_backend_data_subdir(name: str) -> Path:
    app_root = Path("/app")
    if _writable_dir(app_root):
        return app_root / name
    return backend_data_subdir(name)


def container_or_backend_subdir(container_path: str, fallback_name: str) -> Path:
    path = Path(container_path)
    if _writable_dir(path):
        return path
    return backend_data_subdir(fallback_name)
