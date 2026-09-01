from pathlib import Path

import local_paths


def test_backend_data_dir_honors_explicit_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_NEXUS_BACKEND_DATA_DIR", str(tmp_path))

    assert local_paths.backend_data_dir() == tmp_path
    assert local_paths.backend_data_path("model_registry.db") == tmp_path / "model_registry.db"


def test_backend_data_dir_falls_back_to_workspace_when_container_data_is_not_writable(monkeypatch):
    monkeypatch.delenv("LLAMA_NEXUS_BACKEND_DATA_DIR", raising=False)
    monkeypatch.setattr(local_paths, "_writable_dir", lambda path: False)

    assert local_paths.backend_data_dir() == local_paths.BACKEND_ROOT / "data"


def test_container_or_backend_subdir_uses_workspace_fallback(monkeypatch):
    monkeypatch.delenv("LLAMA_NEXUS_BACKEND_DATA_DIR", raising=False)
    monkeypatch.setattr(local_paths, "_writable_dir", lambda path: False)

    assert local_paths.container_or_backend_subdir("/home/llamacpp/models", "models") == (
        local_paths.BACKEND_ROOT / "data" / "models"
    )


def test_app_or_backend_data_subdir_uses_writable_app_mount(monkeypatch, tmp_path):
    def writable_dir(path: Path) -> bool:
        return path == Path("/app")

    monkeypatch.setattr(local_paths, "_writable_dir", writable_dir)

    assert local_paths.app_or_backend_data_subdir("quantization_output") == Path(
        "/app/quantization_output"
    )
