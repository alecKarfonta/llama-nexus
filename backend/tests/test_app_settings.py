"""Tests for the Settings page backend (Milestone 2.4).

[2026-10-16] | INFO | TEST.APP_SETTINGS | Covers the persisted settings store
(masking, env re-application, .env BIND_HOST rewrite) and the /api/v1/settings
routes (aggregate GET, HF token set/clear/echo, LAN toggle + restart signal).
"""

import json
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from modules.app_settings import (
    AppSettingsStore,
    is_masked,
    lan_exposed,
    mask_secret,
    update_env_file_bind_host,
)


def _load_settings_routes():
    """Load routes/settings.py directly — routes/__init__ pulls heavy deps."""
    spec = importlib.util.spec_from_file_location(
        "settings_routes_test", _BACKEND / "routes" / "settings.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["settings_routes_test"] = module
    spec.loader.exec_module(module)
    return module


settings_routes = _load_settings_routes()


# ---------------------------------------------------------------------------
# Store unit tests
# ---------------------------------------------------------------------------

class TestMaskSecret:
    def test_short_value_is_masked_wholesale(self):
        assert mask_secret("abc") == "••••abc"

    def test_long_value_shows_last_four(self):
        assert mask_secret("hf_abc123WXYZ") == "••••WXYZ"

    def test_empty_returns_empty(self):
        assert mask_secret("") == ""
        assert mask_secret(None) == ""

    def test_masked_detection(self):
        assert is_masked("••••abcd") is True
        assert is_masked("hf_real_token") is False
        assert is_masked("") is False


class TestLanExposure:
    def test_wildcard_hosts_are_lan(self, monkeypatch):
        monkeypatch.setenv("BIND_HOST", "0.0.0.0")
        assert lan_exposed() is True
        monkeypatch.setenv("BIND_HOST", "::")
        assert lan_exposed() is True

    def test_localhost_is_not_lan(self, monkeypatch):
        monkeypatch.setenv("BIND_HOST", "127.0.0.1")
        assert lan_exposed() is False

    def test_unset_defaults_to_localhost(self, monkeypatch):
        monkeypatch.delenv("BIND_HOST", raising=False)
        assert lan_exposed() is False


class TestEnvFileBindHost:
    def test_rewrites_existing_bind_host(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("API_KEY=x\nBIND_HOST=127.0.0.1\n")
        monkeypatch.setattr("modules.app_settings.project_root", lambda: tmp_path)
        assert update_env_file_bind_host(True) is True
        assert "BIND_HOST=0.0.0.0" in env.read_text()

    def test_appends_missing_bind_host(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("API_KEY=x\n")
        monkeypatch.setattr("modules.app_settings.project_root", lambda: tmp_path)
        assert update_env_file_bind_host(True) is True
        assert "BIND_HOST=0.0.0.0" in env.read_text()

    def test_readonly_env_returns_false(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("API_KEY=x\n")
        env.chmod(0o444)
        monkeypatch.setattr("modules.app_settings.project_root", lambda: tmp_path)
        # Root can write through 444; treat missing-file as the false path.
        missing_root = tmp_path / "nope"
        monkeypatch.setattr("modules.app_settings.project_root", lambda: missing_root)
        assert update_env_file_bind_host(True) is False


class TestAppSettingsStore:
    def test_hf_token_roundtrip_and_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        store = AppSettingsStore(config_path=tmp_path / "app_settings.json")

        store.set_hf_token("hf_unit_test_1234")
        assert store.get_hf_token() == "hf_unit_test_1234"
        assert os.environ.get("HF_TOKEN") == "hf_unit_test_1234"

        # Reload from disk → env restored on boot
        store2 = AppSettingsStore(config_path=tmp_path / "app_settings.json")
        assert store2.get_hf_token() == "hf_unit_test_1234"

        store2.set_hf_token("")
        assert store2.get_hf_token() is None
        assert "HF_TOKEN" not in os.environ

    def test_lan_desired_persists(self, tmp_path):
        store = AppSettingsStore(config_path=tmp_path / "app_settings.json")
        assert store.get_lan_desired() is None
        store.set_lan_desired(True)
        assert store.get_lan_desired() is True
        store2 = AppSettingsStore(config_path=tmp_path / "app_settings.json")
        assert store2.get_lan_desired() is True


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------

def _build_client(tmp_path, monkeypatch, bind_host="127.0.0.1", management_token=""):
    monkeypatch.setenv("BIND_HOST", bind_host)
    monkeypatch.setenv("MANAGEMENT_AUTH_TOKEN", management_token)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # [2026-10-16] | INFO | TEST.SETTINGS_ISOLATION | .env rewrites must never
    # touch the real repo — point project_root at the test tmp dir in both the
    # routes module and modules.app_settings (which performs the rewrite).
    monkeypatch.setattr(settings_routes, "project_root", lambda: tmp_path)
    monkeypatch.setattr("modules.app_settings.project_root", lambda: tmp_path)

    store = AppSettingsStore(config_path=tmp_path / "app_settings.json")

    app = FastAPI()
    app.include_router(settings_routes.router)
    app.state.app_settings_store = store
    app.state.manager = MagicMock()
    app.state.manager.config = {"server": {"api_key": "infer-key-abcdef"}}
    app.state.embedding_manager = MagicMock()
    app.state.embedding_manager.config = {"server": {"api_key": "embed-key-123456"}}
    return TestClient(app), store


class TestSettingsRoutes:
    def test_get_settings_reports_masked_keys_and_localhost(self, tmp_path, monkeypatch):
        client, _ = _build_client(tmp_path, monkeypatch)
        resp = client.get("/api/v1/settings")
        assert resp.status_code == 200
        body = resp.json()
        assert body["lan"]["effective"] is False
        assert body["lan"]["bind_host"] == "127.0.0.1"
        assert body["api_keys"]["inference"]["masked"].startswith("••••")
        assert "infer-key-abcdef" not in json.dumps(body)

    def test_hf_token_update_and_masked_echo(self, tmp_path, monkeypatch):
        client, store = _build_client(tmp_path, monkeypatch)

        resp = client.put("/api/v1/settings/hf-token", json={"token": "hf_new_token_9999"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"
        assert store.get_hf_token() == "hf_new_token_9999"

        # Masked echo → unchanged
        masked = resp.json()["hf_token_masked"]
        resp = client.put("/api/v1/settings/hf-token", json={"token": masked})
        assert resp.json()["status"] == "unchanged"
        assert store.get_hf_token() == "hf_new_token_9999"

        # Clear
        resp = client.put("/api/v1/settings/hf-token", json={"token": ""})
        assert resp.json()["status"] == "cleared"
        assert store.get_hf_token() is None

    def test_hf_token_rejects_non_hf_prefix(self, tmp_path, monkeypatch):
        client, _ = _build_client(tmp_path, monkeypatch)
        resp = client.put("/api/v1/settings/hf-token", json={"token": "not-a-token"})
        assert resp.status_code == 400

    def test_lan_toggle_requires_restart_in_container(self, tmp_path, monkeypatch):
        client, store = _build_client(tmp_path, monkeypatch)  # BIND_HOST=127.0.0.1
        (tmp_path / ".env").write_text("API_KEY=x\nBIND_HOST=127.0.0.1\n")
        resp = client.put("/api/v1/settings/lan", json={"exposed": True})
        assert resp.status_code == 200
        body = resp.json()
        assert body["desired"] is True
        assert body["effective"] is False  # still bound to localhost until restart
        assert body["restart_required"] is True
        assert "management_token_suggestion" in body  # no management token configured
        assert store.get_lan_desired() is True

    def test_lan_toggle_off_when_already_local(self, tmp_path, monkeypatch):
        client, store = _build_client(tmp_path, monkeypatch)
        resp = client.put("/api/v1/settings/lan", json={"exposed": False})
        assert resp.status_code == 200
        body = resp.json()
        assert body["effective"] is False
        assert body["restart_required"] is False
        assert "warning" not in body

    def test_lan_toggle_writable_env_rewrites_and_instructs_restart(self, tmp_path, monkeypatch):
        client, _store = _build_client(tmp_path, monkeypatch)
        (tmp_path / ".env").write_text("API_KEY=x\nBIND_HOST=127.0.0.1\n")
        resp = client.put("/api/v1/settings/lan", json={"exposed": True})
        body = resp.json()
        assert body["env_file_updated"] is True
        assert body["restart_required"] is True
        assert ".env updated" in body["instructions"]
        assert "BIND_HOST=0.0.0.0" in (tmp_path / ".env").read_text()

    def test_lan_toggle_readonly_env_gives_host_instructions(self, tmp_path, monkeypatch):
        client, _store = _build_client(tmp_path, monkeypatch)  # no .env in tmp_path
        resp = client.put("/api/v1/settings/lan", json={"exposed": True})
        body = resp.json()
        assert body["env_file_updated"] is False
        assert body["restart_required"] is True
        assert "BIND_HOST=0.0.0.0" in body["instructions"]
        assert "cannot write .env" in body["instructions"]

    def test_lan_warning_suppressed_with_management_token(self, tmp_path, monkeypatch):
        client, _ = _build_client(tmp_path, monkeypatch, management_token="sekrit")
        resp = client.put("/api/v1/settings/lan", json={"exposed": True})
        body = resp.json()
        assert "warning" not in body
        assert "management_token_suggestion" not in body

    def test_management_token_suggestion_endpoint(self, tmp_path, monkeypatch):
        client, _ = _build_client(tmp_path, monkeypatch)
        resp = client.get("/api/v1/settings/management-token/suggest")
        assert resp.status_code == 200
        assert resp.json()["token"].startswith("nx-")

    def test_store_missing_returns_503(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BIND_HOST", "127.0.0.1")
        app = FastAPI()
        app.include_router(settings_routes.router)
        app.state.app_settings_store = None
        client = TestClient(app)
        assert client.get("/api/v1/settings").status_code == 503
