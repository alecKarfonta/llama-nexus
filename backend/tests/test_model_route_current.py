import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


BACKEND_DIR = Path(__file__).resolve().parents[1]
MODELS_ROUTE = BACKEND_DIR / "routes" / "models.py"


def load_models_route_module():
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    spec = importlib.util.spec_from_file_location("models_route", MODELS_ROUTE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


models = load_models_route_module()


class _DummyVLLMManager:
    config = {"server": {"api_key": "vllm-key"}}

    def is_running(self):
        return True

    def internal_http_base(self):
        return "http://vllm-api:8080"

    def get_status(self):
        return {
            "model": {
                "name": "stale/config-model",
                "served_name": "stale-served-name",
                "max_model_len": 16384,
            }
        }


def test_current_vllm_model_prefers_live_openai_models(monkeypatch):
    async def fake_live_model_info(base_url, api_key=""):
        assert api_key == "vllm-key"
        if base_url == "http://127.0.0.1:8601":
            return {"id": "Qwen/Qwen3.5-9B", "max_model_len": 8192}
        return None

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                manager=None,
                vllm_manager=_DummyVLLMManager(),
            ),
        ),
    )

    monkeypatch.setenv("VLLM_PROXY_BASE_URL", "http://127.0.0.1:8601")
    monkeypatch.setattr(models, "_live_openai_model_info", fake_live_model_info)

    result = asyncio.run(models.get_current_model(request))

    assert result["name"] == "Qwen/Qwen3.5-9B"
    assert result["context_size"] == 8192
    assert result["backend"] == "vllm"


def test_current_vllm_model_uses_host_candidate_without_env(monkeypatch):
    seen_base_urls = []

    async def fake_live_model_info(base_url, api_key=""):
        seen_base_urls.append(base_url)
        if base_url == "http://127.0.0.1:8601":
            return {"id": "Qwen/Qwen3.5-9B", "max_model_len": 8192}
        return None

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
        models,
        "vllm_proxy_base_url_candidates",
        lambda internal_base_url: ["http://127.0.0.1:8601", internal_base_url],
    )
    monkeypatch.setattr(models, "_live_openai_model_info", fake_live_model_info)

    result = asyncio.run(models.get_current_model(request))

    assert result["name"] == "Qwen/Qwen3.5-9B"
    assert result["context_size"] == 8192
    assert seen_base_urls[0] == "http://127.0.0.1:8601"


def test_current_vllm_model_tries_env_key_when_manager_key_is_stale(monkeypatch):
    class StaleKeyVLLMManager(_DummyVLLMManager):
        config = {"server": {"api_key": "stale-key"}}

    seen_api_keys = []

    async def fake_live_model_info(base_url, api_key=""):
        seen_api_keys.append(api_key)
        if base_url == "http://127.0.0.1:8601" and api_key == "env-key":
            return {"id": "Qwen/Qwen3.5-9B", "max_model_len": 8192}
        return None

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                manager=None,
                vllm_manager=StaleKeyVLLMManager(),
            ),
        ),
    )

    monkeypatch.setenv("VLLM_PROXY_BASE_URL", "http://127.0.0.1:8601")
    monkeypatch.setenv("VLLM_API_KEY", "env-key")
    monkeypatch.setattr(models, "_live_openai_model_info", fake_live_model_info)

    result = asyncio.run(models.get_current_model(request))

    assert result["name"] == "Qwen/Qwen3.5-9B"
    assert result["context_size"] == 8192
    assert "stale-key" in seen_api_keys
    assert "env-key" in seen_api_keys
