from modules import inference_targets


def test_vllm_candidates_try_host_port_before_container_dns_on_desktop(monkeypatch):
    monkeypatch.delenv("VLLM_PROXY_BASE_URL", raising=False)
    monkeypatch.delenv("INFERENCE_BACKEND_URL", raising=False)
    monkeypatch.setattr(inference_targets, "running_inside_container", lambda: False)

    assert inference_targets.vllm_proxy_base_url_candidates("http://vllm-api:8080") == [
        "http://127.0.0.1:8601",
        "http://localhost:8601",
        "http://vllm-api:8080",
    ]


def test_vllm_candidates_keep_container_dns_inside_docker(monkeypatch):
    monkeypatch.delenv("VLLM_PROXY_BASE_URL", raising=False)
    monkeypatch.delenv("INFERENCE_BACKEND_URL", raising=False)
    monkeypatch.setattr(inference_targets, "running_inside_container", lambda: True)

    assert inference_targets.vllm_proxy_base_url_candidates("http://vllm-api:8080") == [
        "http://vllm-api:8080",
    ]


def test_vllm_candidates_prefer_explicit_proxy(monkeypatch):
    monkeypatch.setenv("VLLM_PROXY_BASE_URL", "http://example.test:9000/")
    monkeypatch.setenv("INFERENCE_BACKEND_URL", "http://127.0.0.1:8601")
    monkeypatch.setattr(inference_targets, "running_inside_container", lambda: False)

    assert inference_targets.vllm_proxy_base_url_candidates("http://vllm-api:8080")[:2] == [
        "http://example.test:9000",
        "http://127.0.0.1:8601",
    ]
