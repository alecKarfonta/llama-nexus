import json
import base64

from modules.graphrag.pipeline_runs import PipelineRunStore


def test_pipeline_run_store_reuses_cached_reads_and_invalidates_on_file_change(tmp_path, monkeypatch):
    store = PipelineRunStore(tmp_path / "pipeline_runs.json")
    store.upsert({"run_id": "run-a", "value": 1})

    load_count = 0
    original_load = json.load

    def counted_load(handle):
        nonlocal load_count
        load_count += 1
        return original_load(handle)

    monkeypatch.setattr("modules.graphrag.pipeline_runs.json.load", counted_load)

    assert store.list_runs()[0]["run_id"] == "run-a"
    assert store.get("run-a")["value"] == 1
    assert load_count == 0

    store.index_path.write_text(
        json.dumps({"version": 1, "runs": [{"run_id": "run-b", "value": 2}]}),
        encoding="utf-8",
    )

    assert store.list_runs()[0]["run_id"] == "run-b"
    assert store.get("run-b")["value"] == 2
    assert load_count == 1


def test_pipeline_run_store_externalizes_large_artifact_payloads(tmp_path):
    store = PipelineRunStore(tmp_path / "pipeline_runs.json")

    store.upsert({
        "run_id": "run-artifacts",
        "source_document_artifact": {
            "filename": "source.pdf",
            "content_type": "application/pdf",
            "content_base64": base64.b64encode(b"%PDF source bytes").decode("ascii"),
            "size_bytes": 17,
        },
        "structured_ingest_artifact": {
            "filename": "source.structured.md",
            "content_type": "text/markdown",
            "content": "# Structured GraphRAG Ingest\n\nUseful trace text.",
            "char_count": 45,
        },
    })

    stored_text = (tmp_path / "pipeline_runs.json").read_text(encoding="utf-8")

    assert "%PDF source bytes" not in stored_text
    assert "Structured GraphRAG Ingest" not in stored_text
    assert "content_base64" not in stored_text
    assert "content_storage_path" in stored_text

    reloaded = PipelineRunStore(tmp_path / "pipeline_runs.json").get("run-artifacts")

    assert reloaded is not None
    assert reloaded["source_document_artifact"]["externalized"] is True
    assert reloaded["structured_ingest_artifact"]["externalized"] is True
    assert (
        reloaded["source_document_artifact"]["content_storage_path"]
        != reloaded["structured_ingest_artifact"]["content_storage_path"]
    )
