"""Route tests for the NInfer backend (config/action/status dispatch + chat proxy)."""
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
    spec = importlib.util.spec_from_file_location("service_routes_test", _BACKEND / "routes" / "service.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["service_routes_test"] = module
    spec.loader.exec_module(module)
    return module


def _mock_backend_manager(running: bool = False, backend_name: str = "ninfer"):
    mgr = MagicMock()
    mgr.is_running = MagicMock(return_value=running)
    mgr.get_status = MagicMock(
        return_value={
            "running": running,
            "backend": backend_name,
            "container": f"{backend_name}-api",
            "last_action_error": None,
            "config": {"server": {"api_key": "k-test"}},
        }
    )
    mgr.get_health = AsyncMock(return_value={"healthy": running})
    mgr.stop = AsyncMock(return_value=True)
    mgr.start = AsyncMock(return_value=True)
    mgr.restart = AsyncMock(return_value=True)
    mgr.persist_deploy_config = MagicMock()
    if backend_name == "ninfer":
        mgr.config = {
            "backend_type": "ninfer",
            "model": {"artifact": "qwen3_8_27b_nvfp4.ninfer", "served_name": ""},
            "server": {"host": "0.0.0.0", "port": 8080, "api_key": "k-test"},
        }
        mgr.build_command = MagicMock(
            return_value=["ninfer-serve", "/models/qwen3_8_27b_nvfp4.ninfer", "--host", "0.0.0.0", "--port", "8080"]
        )
        mgr.get_field_metadata = MagicMock(return_value={})
        mgr.registered_artifacts = MagicMock(return_value=[])
        mgr.artifact_exists = MagicMock(return_value=True)
        mgr.internal_http_base = MagicMock(return_value="http://ninfer-api:8080")
    return mgr


@pytest.fixture
def client():
    routes = _load_routes()
    app = FastAPI()
    app.include_router(routes.router)

    llamacpp_mgr = _mock_backend_manager(running=False, backend_name="llamacpp")
    llamacpp_mgr.use_docker = True
    vllm_mgr = _mock_backend_manager(running=False, backend_name="vllm")
    ninfer_mgr = _mock_backend_manager(running=False, backend_name="ninfer")

    app.state.manager = llamacpp_mgr
    app.state.vllm_manager = vllm_mgr
    app.state.ninfer_manager = ninfer_mgr
    app.state.merge_and_persist_config = MagicMock()

    return TestClient(app), llamacpp_mgr, vllm_mgr, ninfer_mgr


def test_get_config_dispatches_to_ninfer(client):
    test_client, _llama, _vllm, ninfer = client
    resp = test_client.get("/api/v1/service/config", params={"backend": "ninfer"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"] == "ninfer"
    assert body["config"]["model"]["artifact"] == "qwen3_8_27b_nvfp4.ninfer"
    assert body["command"].startswith("ninfer-serve")
    assert body["artifact_exists"] is True


def test_put_config_merges_and_persists_ninfer(client):
    test_client, _llama, _vllm, ninfer = client
    resp = test_client.put(
        "/api/v1/service/config",
        params={"backend": "ninfer"},
        json={"performance": {"max_context": 65536}},
    )
    assert resp.status_code == 200
    assert resp.json()["backend"] == "ninfer"
    assert ninfer.config["performance"]["max_context"] == 65536
    ninfer.persist_deploy_config.assert_called_once()


def test_backends_endpoint_includes_ninfer(client):
    test_client, _llama, _vllm, ninfer = client
    resp = test_client.get("/api/v1/service/backends")
    assert resp.status_code == 200
    body = resp.json()
    assert "ninfer" in body["backends"]
    assert body["backends"]["ninfer"]["backend"] == "ninfer"


def test_ninfer_status_endpoint(client):
    test_client, _llama, _vllm, ninfer = client
    ninfer.get_status.return_value["running"] = True
    resp = test_client.get("/api/v1/service/ninfer/status")
    assert resp.status_code == 200
    assert resp.json()["backend"] == "ninfer"
    ninfer.get_health.assert_awaited_once()


def test_service_action_start_stops_other_backends(client):
    test_client, llama, vllm, ninfer = client
    llama.get_status.return_value["running"] = True
    vllm.get_status.return_value["running"] = True
    resp = test_client.post(
        "/api/v1/service/action",
        json={"action": "start", "backend": "ninfer", "config": {"performance": {"max_concurrency": 4}}},
    )
    assert resp.status_code == 200
    assert resp.json()["backend"] == "ninfer"
    llama.stop.assert_awaited_once()
    vllm.stop.assert_awaited_once()
    ninfer.start.assert_awaited_once()
    assert ninfer.config["performance"]["max_concurrency"] == 4


def test_service_action_failure_surfaces_last_action_error(client):
    test_client, _llama, _vllm, ninfer = client
    ninfer.start = AsyncMock(return_value=False)
    ninfer.last_action_error = "artifact not found"
    resp = test_client.post("/api/v1/service/action", json={"action": "start", "backend": "ninfer"})
    assert resp.status_code == 500
    assert "artifact not found" in resp.json()["detail"]


def test_chat_proxy_targets_ninfer_when_running(client):
    test_client, _llama, vllm, ninfer = client
    ninfer.is_running.return_value = True

    # Patch the proxy POST so no real HTTP is attempted
    routes = sys.modules["service_routes_test"]
    routes._proxy_non_stream = AsyncMock(return_value={"ok": True})

    resp = test_client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    url = routes._proxy_non_stream.call_args.args[0]
    assert url == "http://ninfer-api:8080/v1/chat/completions"
    # vLLM idle → ninfer must win over llamacpp default
    vllm.is_running.assert_called()


def test_chat_proxy_prefers_vllm_over_ninfer(client):
    test_client, _llama, vllm, ninfer = client
    vllm.is_running.return_value = True
    ninfer.is_running.return_value = True
    vllm.internal_http_base = MagicMock(return_value="http://vllm-api:8080")

    routes = sys.modules["service_routes_test"]
    routes._proxy_non_stream = AsyncMock(return_value={"ok": True})

    resp = test_client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    url = routes._proxy_non_stream.call_args.args[0]
    assert url == "http://vllm-api:8080/v1/chat/completions"
