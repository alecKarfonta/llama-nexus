import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


BACKEND_DIR = Path(__file__).resolve().parents[1]
SERVICE_ROUTE = BACKEND_DIR / "routes" / "service.py"


def load_service_route_module():
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    spec = importlib.util.spec_from_file_location("service_route", SERVICE_ROUTE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


service = load_service_route_module()


class _DummyVLLMManager:
    config = {"server": {"api_key": "vllm-key"}}

    def is_running(self):
        return True

    def internal_http_base(self):
        return "http://vllm-api:8080"


class _DummyLlamaManager:
    config = {"server": {"api_key": "llama-key"}}
    use_docker = True

    def is_running(self):
        return True


def test_active_inference_target_uses_host_reachable_vllm_override(monkeypatch):
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                manager=None,
                vllm_manager=_DummyVLLMManager(),
            ),
        ),
    )

    monkeypatch.setenv("VLLM_PROXY_BASE_URL", "http://127.0.0.1:8080/")

    assert service._active_inference_target(request) == (
        "http://127.0.0.1:8080",
        "vllm-key",
        "vllm",
    )


def test_active_inference_target_uses_vllm_candidate_without_env(monkeypatch):
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                manager=None,
                vllm_manager=_DummyVLLMManager(),
            ),
        ),
    )

    monkeypatch.delenv("VLLM_PROXY_BASE_URL", raising=False)
    monkeypatch.delenv("INFERENCE_BACKEND_URL", raising=False)
    monkeypatch.setattr(
        service,
        "vllm_proxy_base_url",
        lambda internal_base_url: "http://127.0.0.1:8601",
    )

    assert service._active_inference_target(request) == (
        "http://127.0.0.1:8601",
        "vllm-key",
        "vllm",
    )


def test_active_inference_target_uses_generic_llamacpp_override(monkeypatch):
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                manager=_DummyLlamaManager(),
                vllm_manager=None,
            ),
        ),
    )

    monkeypatch.setenv("INFERENCE_BACKEND_URL", "http://127.0.0.1:8080")

    assert service._active_inference_target(request) == (
        "http://127.0.0.1:8080",
        "llama-key",
        "llamacpp",
    )
