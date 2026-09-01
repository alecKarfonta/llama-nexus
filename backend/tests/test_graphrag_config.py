from modules.graphrag.config import get_graphrag_config, reset_config


def test_graphrag_config_uses_docker_helper_defaults(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_ENABLED", "true")
    monkeypatch.setenv("RUNNING_IN_DOCKER", "true")
    monkeypatch.delenv("NER_API_URL", raising=False)
    monkeypatch.delenv("REL_API_URL", raising=False)
    monkeypatch.delenv("CODE_RAG_URL", raising=False)
    reset_config()

    cfg = get_graphrag_config()

    assert cfg.ner_api_url == "http://graphrag-ner:8001"
    assert cfg.rel_api_url == "http://graphrag-rel:8002"
    assert cfg.code_rag_url == "http://graphrag-code:8000"
    assert cfg.service_url_sources["ner"] == "docker_default"
    assert cfg.service_url_sources["rel"] == "docker_default"
    assert cfg.service_url_sources["code_rag"] == "docker_default"
    assert cfg.auto_discovered_service_count == 3

    reset_config()


def test_graphrag_config_leaves_helper_urls_empty_outside_docker(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_ENABLED", "true")
    monkeypatch.delenv("RUNNING_IN_DOCKER", raising=False)
    monkeypatch.delenv("USE_DOCKER", raising=False)
    monkeypatch.delenv("NER_API_URL", raising=False)
    monkeypatch.delenv("REL_API_URL", raising=False)
    monkeypatch.delenv("CODE_RAG_URL", raising=False)
    reset_config()

    cfg = get_graphrag_config()

    assert cfg.ner_api_url is None
    assert cfg.rel_api_url is None
    assert cfg.code_rag_url is None
    assert cfg.service_url_sources["ner"] == "unset"
    assert cfg.auto_discovered_service_count == 0

    reset_config()


def test_graphrag_config_records_explicit_and_alias_url_sources(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_ENABLED", "true")
    monkeypatch.setenv("GRAPHRAG_SERVICE_URL", "http://graph.example")
    monkeypatch.setenv("GRAPHRAG_NER_URL", "http://ner.example")
    monkeypatch.setenv("REL_API_URL", "http://rel.example")
    monkeypatch.delenv("GRAPHRAG_URL", raising=False)
    monkeypatch.delenv("NER_API_URL", raising=False)
    monkeypatch.delenv("RUNNING_IN_DOCKER", raising=False)
    monkeypatch.delenv("USE_DOCKER", raising=False)
    reset_config()

    cfg = get_graphrag_config()

    assert cfg.graphrag_url == "http://graph.example"
    assert cfg.ner_api_url == "http://ner.example"
    assert cfg.rel_api_url == "http://rel.example"
    assert cfg.service_url_sources["graphrag"] == "GRAPHRAG_SERVICE_URL"
    assert cfg.service_url_sources["ner"] == "GRAPHRAG_NER_URL"
    assert cfg.service_url_sources["rel"] == "REL_API_URL"
    assert cfg.auto_discovered_service_count == 0

    reset_config()


def test_graphrag_config_records_image_ocr_flag(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_ENABLE_IMAGE_OCR", "true")
    reset_config()

    cfg = get_graphrag_config()

    assert cfg.image_ocr_enabled is True

    monkeypatch.setenv("GRAPHRAG_ENABLE_IMAGE_OCR", "false")
    reset_config()

    cfg = get_graphrag_config()

    assert cfg.image_ocr_enabled is False

    reset_config()
