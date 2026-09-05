"""Route tests for MTP graft API."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _load_routes():
    spec = importlib.util.spec_from_file_location("models_routes_test", _BACKEND / "routes" / "models.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["models_routes_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def client():
    routes = _load_routes()
    app = FastAPI()
    app.include_router(routes.router, prefix="/v1/models")

    mgr = MagicMock()
    mgr.preflight = AsyncMock(
        return_value={
            "ok": True,
            "errors": [],
            "warnings": [],
            "planned_output_filename": "demo-MTP-Q6_K.gguf",
        }
    )
    mgr.start_job = AsyncMock(
        return_value=MagicMock(
            to_dict=lambda: {"job_id": "abc123", "status": "queued", "progress": 0.0, "logs": []},
        )
    )
    mgr.list_jobs = AsyncMock(return_value=[])
    mgr.get_job = AsyncMock(
        return_value=MagicMock(
            to_dict=lambda: {"job_id": "abc123", "status": "completed", "progress": 100.0, "logs": []},
        )
    )
    mgr.cancel_job = AsyncMock(return_value=True)
    app.state.mtp_graft_manager = mgr

    return TestClient(app), mgr


def test_mtp_graft_preflight_route(client):
    test_client, mgr = client
    resp = test_client.post(
        "/v1/models/mtp-graft/preflight",
        json={"target": {"name": "demo", "variant": "Q6_K"}, "donor": {"name": "donor", "variant": "Q8_0"}},
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["ok"] is True
    mgr.preflight.assert_awaited_once()


def test_mtp_graft_donor_suggestions_route(client):
    test_client, _mgr = client
    resp = test_client.get(
        "/v1/models/mtp-graft/suggestions/donor",
        params={"target_name": "Qwen3.6-35B-A3B-test"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["family"] == "qwen3.6-moe"
    assert len(data["suggested_donors"]) >= 1


def test_mtp_graft_start_route(client):
    test_client, mgr = client
    resp = test_client.post(
        "/v1/models/mtp-graft",
        json={"target": {"name": "demo", "variant": "Q6_K"}, "donor": {"name": "donor", "variant": "Q8_0"}},
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["job_id"] == "abc123"
    mgr.start_job.assert_awaited_once()
