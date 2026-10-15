"""Workspace-safe runtime paths for local and container deployments."""

from __future__ import annotations

import os
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent


def project_root() -> Path:
    """[2026-10-15] | INFO | LOCAL_PATHS.PROJECT_ROOT | Repo root for compose/Dockerfile access.

    Resolution order: PROJECT_DIR env (set by docker-compose) -> parent of backend/
    (works bare-metal and inside the container where the repo is mounted at /workspace).
    """
    env_dir = os.getenv("PROJECT_DIR", "").strip()
    if env_dir:
        candidate = Path(env_dir)
        if (candidate / "docker-compose.yml").exists():
            return candidate
    repo_root = BACKEND_ROOT.parent
    if (repo_root / "docker-compose.yml").exists():
        return repo_root
    # Container fallback: repo is mounted read-only at /workspace by compose
    workspace = Path("/workspace")
    if (workspace / "docker-compose.yml").exists():
        return workspace
    return repo_root


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
