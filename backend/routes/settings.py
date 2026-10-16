"""Settings routes: HF token, API keys, storage paths, LAN exposure.

[2026-10-16] | INFO | SETTINGS.ROUTES | Backs the real Settings page
(Milestone 2.4). Secrets are write-only over the wire: GET returns masked
hints; PUT accepts a masked echo (unchanged), a new value, or "" (clear).
Effective values for API keys are reported from the live deployment configs
where applicable (llamacpp manager / embedding manager), since those are the
values inference actually uses until the next deploy.
"""

import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from typing import Any, Dict, Optional

from enhanced_logger import enhanced_logger as logger
from modules.app_settings import (
    AppSettingsStore,
    effective_bind_host,
    is_masked,
    lan_exposed,
    mask_secret,
    storage_snapshot,
    suggested_management_token,
    update_env_file_bind_host,
)
from local_paths import backend_data_dir, project_root

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


def get_settings_store(request: Request) -> Optional[AppSettingsStore]:
    """Settings store wired by app_lifespan (None when module unavailable)."""
    return getattr(request.app.state, "app_settings_store", None)


def _env_file_writable() -> bool:
    path = project_root() / ".env"
    return path.is_file() and os.access(path, os.W_OK)


def _effective_api_keys(request: Request) -> Dict[str, Dict[str, Any]]:
    """Masked inference/embedding keys from live configs (env fallback).

    Managers resolve via app.state (convention used by the other routes) so
    tests can inject mocks without importing the full app_state graph.
    """
    manager = getattr(request.app.state, "manager", None)
    embedding_manager = getattr(request.app.state, "embedding_manager", None)

    inference_key = ""
    try:
        inference_key = str(manager.config.get("server", {}).get("api_key") or "")
    except Exception:
        pass
    inference_key = inference_key or os.getenv("API_KEY", "")

    embed_key = ""
    try:
        embed_key = str(embedding_manager.config.get("server", {}).get("api_key") or "")
    except Exception:
        pass
    embed_key = embed_key or os.getenv("EMBEDDING_SERVICE_API_KEY", "") or os.getenv("EMBED_API_KEY", "")

    return {
        "inference": {
            "masked": mask_secret(inference_key),
            "source": "llamacpp_config" if inference_key else None,
        },
        "embedding": {
            "masked": mask_secret(embed_key),
            "source": "embedding_config" if embed_key else None,
        },
    }


def _storage_paths() -> Dict[str, str]:
    return {
        "Models (GGUF)": str(os.getenv("MODELS_DIR") or "/home/llamacpp/models"),
        "Backend data": str(backend_data_dir()),
        "Conversations": str(backend_data_dir() / "conversations"),
        "Finetune datasets": str(backend_data_dir() / "finetune_datasets"),
        "Finetune jobs": str(backend_data_dir() / "finetune_jobs"),
        "HF cache": str(os.getenv("HF_HOME") or "/root/.cache/huggingface"),
    }


# ---------------------------------------------------------------------------
# Aggregate GET
# ---------------------------------------------------------------------------

@router.get("")
async def get_settings(request: Request):
    """Full settings payload for the Settings page."""
    store = get_settings_store(request)
    if store is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Settings store not available"},
        )

    bind_host = effective_bind_host()
    effective_lan = lan_exposed(bind_host)
    desired_lan = store.get_lan_desired()
    management_auth = os.getenv("MANAGEMENT_AUTH_TOKEN", "").strip()
    hf_token = store.get_hf_token()

    return {
        "hf_token_set": bool(hf_token),
        "hf_token_masked": mask_secret(hf_token),
        "hf_token_source": "settings" if hf_token else (
            "env" if os.getenv("HUGGINGFACE_TOKEN") else None
        ),
        "api_keys": _effective_api_keys(request),
        "storage": storage_snapshot(_storage_paths()),
        "lan": {
            "effective": effective_lan,
            "desired": desired_lan,
            "bind_host": bind_host,
            "management_auth_configured": bool(management_auth),
            "env_file_writable": _env_file_writable(),
        },
        "security": {
            "management_auth_configured": bool(management_auth),
            "bind_host": bind_host,
        },
        "docker": bool(os.getenv("RUNNING_IN_DOCKER", "").strip()),
    }


# ---------------------------------------------------------------------------
# HF token
# ---------------------------------------------------------------------------

@router.put("/hf-token")
async def update_hf_token(request: Request):
    """Set / clear the HuggingFace token.

    Body: {"token": "hf_..."} — masked echo leaves it unchanged; "" clears.
    """
    store = get_settings_store(request)
    if store is None:
        return JSONResponse(status_code=503, content={"detail": "Settings store not available"})

    body = await request.json()
    token = (body.get("token") or "").strip()

    if is_masked(token):
        return {"status": "unchanged", "hf_token_masked": mask_secret(store.get_hf_token())}
    if token and not token.startswith("hf_"):
        return JSONResponse(
            status_code=400,
            content={"detail": "HuggingFace tokens start with 'hf_' — check the value from huggingface.co/settings/tokens."},
        )

    store.set_hf_token(token)
    logger.info("Settings: HuggingFace token %s", "updated" if token else "cleared")
    return {
        "status": "updated" if token else "cleared",
        "hf_token_masked": mask_secret(store.get_hf_token()),
        "hf_token_set": bool(store.get_hf_token()),
    }


# ---------------------------------------------------------------------------
# LAN exposure
# ---------------------------------------------------------------------------

@router.put("/lan")
async def update_lan_exposure(request: Request):
    """Toggle LAN exposure (BIND_HOST 0.0.0.0 vs 127.0.0.1).

    Persists desired state, rewrites .env when writable (bare-metal), and
    reports whether a host-side restart is required for it to take effect.
    Body: {"exposed": true|false}
    """
    store = get_settings_store(request)
    if store is None:
        return JSONResponse(status_code=503, content={"detail": "Settings store not available"})

    body = await request.json()
    exposed = bool(body.get("exposed"))

    store.set_lan_desired(exposed)
    env_updated = update_env_file_bind_host(exposed)
    effective_now = lan_exposed(effective_bind_host())

    # Desired-state mismatch means the stack needs a host-side compose
    # restart (ports are bound by Docker before the backend starts).
    restart_required = effective_now != exposed

    response: Dict[str, Any] = {
        "status": "updated",
        "desired": exposed,
        "effective": effective_now,
        "env_file_updated": env_updated,
        "restart_required": restart_required,
    }

    if restart_required:
        if env_updated:
            response["instructions"] = (
                ".env updated. Restart the stack on the host (docker compose up -d) "
                "to apply the new binding."
            )
        else:
            # [2026-10-16] | INFO | SETTINGS.LAN_ENV_READONLY | In-container the
            # repo is mounted read-only — the user must edit .env on the host.
            response["instructions"] = (
                f"The backend cannot write .env from inside the container. On the host, set "
                f"BIND_HOST={'0.0.0.0' if exposed else '127.0.0.1'} in the .env next to "
                f"docker-compose.yml, then restart the stack (docker compose up -d)."
            )

    if exposed and not os.getenv("MANAGEMENT_AUTH_TOKEN", "").strip():
        response["management_token_suggestion"] = suggested_management_token()
        response["warning"] = (
            "LAN exposure is on but MANAGEMENT_AUTH_TOKEN is not set — the management "
            "API will be open to your network. Add the suggested token to .env."
        )

    logger.info("Settings: LAN exposure desired=%s env_updated=%s", exposed, env_updated)
    return response


# ---------------------------------------------------------------------------
# Management token suggestion (explicit endpoint for the UI's copy action)
# ---------------------------------------------------------------------------

@router.get("/management-token/suggest")
async def suggest_management_token_endpoint():
    """Fresh random token suggestion for MANAGEMENT_AUTH_TOKEN."""
    return {"token": suggested_management_token()}
