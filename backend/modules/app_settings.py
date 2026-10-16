"""Server-side application settings (Settings page backing store).

[2026-10-16] | INFO | APP_SETTINGS.MODULE | The Settings page used to be pure
localStorage cosmetics. Real installation state — HF token, desired LAN
exposure — now lives here, persisted as JSON on the backend_data volume
(next to llamacpp_config.json, the existing config mechanism) so values
survive container rebuilds. Secrets are re-applied to os.environ on boot
because runtime env changes die with the process.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from enhanced_logger import enhanced_logger as logger
from local_paths import backend_data_path, project_root

# [2026-10-16] | INFO | APP_SETTINGS.MASKING | GET responses never return raw
# secrets — only a masked hint. The client echoes the mask back on save to
# mean "leave unchanged"; an explicit empty string clears the value.
MASK_PREFIX = "••••"
DEFAULT_BIND_HOST = "127.0.0.1"
LAN_BIND_HOST = "0.0.0.0"


def mask_secret(value: Optional[str]) -> str:
    """Return a non-reversible display hint for a secret (last 4 chars only)."""
    if not value:
        return ""
    tail = value[-4:] if len(value) > 4 else value
    return f"{MASK_PREFIX}{tail}"


def is_masked(value: Optional[str]) -> bool:
    """True when a submitted value is the masked echo we sent to the client."""
    return bool(value) and value.startswith(MASK_PREFIX)


def effective_bind_host() -> str:
    """Host address the compose stack binds UI/API ports to (env pass-through)."""
    return (os.getenv("BIND_HOST") or DEFAULT_BIND_HOST).strip() or DEFAULT_BIND_HOST


def lan_exposed(bind_host: Optional[str] = None) -> bool:
    """True when the stack binds to a wildcard address reachable from the LAN."""
    host = (bind_host if bind_host is not None else effective_bind_host()).strip().lower()
    return host in {LAN_BIND_HOST, "::", "[::]"}


def suggested_management_token() -> str:
    """Random token suggested when LAN exposure is on without management auth."""
    return f"nx-{secrets.token_hex(16)}"


def update_env_file_bind_host(exposed: bool) -> bool:
    """Rewrite BIND_HOST in the compose .env when that file is writable.

    [2026-10-16] | INFO | APP_SETTINGS.ENV_FILE | In the container the repo is
    mounted read-only at /workspace, so production installs must edit .env on
    the host — the caller surfaces restart instructions in that case. Returns
    True only when the file was actually rewritten.
    """
    target = LAN_BIND_HOST if exposed else DEFAULT_BIND_HOST
    try:
        env_path = project_root() / ".env"
        if not env_path.is_file() or not os.access(env_path, os.W_OK):
            return False
        content = env_path.read_text()
        line = f"BIND_HOST={target}"
        if re.search(r"(?m)^\s*BIND_HOST\s*=", content):
            content = re.sub(r"(?m)^\s*BIND_HOST\s*=.*$", line, content)
        else:
            content = content.rstrip("\n") + f"\n{line}\n"
        env_path.write_text(content)
        logger.info(f"AppSettings: .env BIND_HOST updated to {target}")
        return True
    except Exception as e:
        logger.warning(f"AppSettings: failed to update .env BIND_HOST: {e}")
        return False


def storage_snapshot(paths: Dict[str, str]) -> List[Dict[str, Any]]:
    """Annotate storage paths with existence + free/total space where readable."""
    rows: List[Dict[str, Any]] = []
    for label, raw in paths.items():
        path = Path(raw)
        row: Dict[str, Any] = {"label": label, "path": raw, "exists": path.exists()}
        try:
            probe = path if path.exists() else path.parent
            usage = shutil.disk_usage(probe)
            row["free_gb"] = round(usage.free / (1024 ** 3), 1)
            row["total_gb"] = round(usage.total / (1024 ** 3), 1)
        except Exception as e:
            # [2026-10-16] | DEBUG | APP_SETTINGS.STORAGE | Some paths (e.g.
            # missing mounts) cannot be probed; the UI shows them as unknown.
            row["error"] = str(e)
        rows.append(row)
    return rows


class AppSettingsStore:
    """Persisted installation settings: secrets + desired LAN state."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or backend_data_path("app_settings.json")
        self.settings: Dict[str, Any] = self._load()
        self._apply_environment()

    # -- persistence -------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            logger.error(f"Failed to load app settings from {self.config_path}: {e}")
        return {}

    def persist(self) -> None:
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist app settings to {self.config_path}: {e}")

    def _touch(self) -> None:
        self.settings["updated_at"] = datetime.now().isoformat()
        self.persist()

    def _apply_environment(self) -> None:
        """Re-apply persisted secrets to os.environ at boot (see module docstring)."""
        token = self.get_hf_token()
        if token:
            os.environ["HUGGINGFACE_TOKEN"] = token
            os.environ["HF_TOKEN"] = token
            logger.info("AppSettings: HuggingFace token restored from persisted settings")

    # -- HuggingFace token --------------------------------------------------

    def get_hf_token(self) -> Optional[str]:
        return (self.settings.get("hf_token") or "").strip() or None

    def set_hf_token(self, value: Optional[str]) -> None:
        token = (value or "").strip()
        if token:
            self.settings["hf_token"] = token
            os.environ["HUGGINGFACE_TOKEN"] = token
            os.environ["HF_TOKEN"] = token
        else:
            self.settings.pop("hf_token", None)
            os.environ.pop("HUGGINGFACE_TOKEN", None)
            os.environ.pop("HF_TOKEN", None)
        self._touch()

    # -- LAN exposure (desired state; effective state comes from BIND_HOST) --

    def get_lan_desired(self) -> Optional[bool]:
        value = self.settings.get("lan_exposed")
        return value if isinstance(value, bool) else None

    def set_lan_desired(self, exposed: bool) -> None:
        self.settings["lan_exposed"] = bool(exposed)
        self._touch()
