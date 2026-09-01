"""
Durable storage for GraphRAG ingestion pipeline traces.

The upload pipeline returns rich per-stage previews. Keeping the latest runs on
disk makes those traces useful after backend restarts without introducing a
database dependency for this small bounded history.
"""

from __future__ import annotations

from collections import OrderedDict
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any

logger = None
try:
    from enhanced_logger import enhanced_logger as logger
except Exception:  # pragma: no cover - logging must never break storage imports
    import logging

    logger = logging.getLogger(__name__)


DEFAULT_MAX_RUNS = 12
SOURCE_ARTIFACT_FILENAME = "source_document.bin"
STRUCTURED_ARTIFACT_FILENAME = "structured_ingest.md"
PERSISTED_STRING_LIMIT = 1_600
PERSISTED_DEFAULT_LIST_LIMIT = 24
PERSISTED_LIST_LIMITS = {
    "chunks": 64,
    "chunk_evidence_cards": 6,
    "embedding_evidence_cards": 6,
    "embedding_inputs": 3,
    "representative_embedding_inputs": 3,
    "evidence_cards": 8,
    "graph_anchors": 12,
    "graph_context": 16,
    "graph_paths": 16,
    "nodes": 320,
    "edges": 500,
    "path_candidates": 80,
    "bridge_entities": 32,
    "relationships": 12,
    "relationship_evidence_cards": 6,
    "relationship_highlights": 6,
    "entities": 16,
    "retrieval_rehearsals": 4,
    "suggested_questions": 6,
    "starter_prompts": 6,
    "capability_cards": 12,
    "lanes": 8,
}


def default_pipeline_runs_file() -> Path:
    env_path = os.getenv("GRAPHRAG_PIPELINE_RUNS_FILE")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "graphrag"
        / "pipeline_runs.json"
    )


class PipelineRunStore:
    def __init__(self, index_path: Path | None = None, max_runs: int = DEFAULT_MAX_RUNS):
        self.index_path = index_path or default_pipeline_runs_file()
        self.max_runs = max(1, max_runs)
        self._lock = RLock()
        self._cached_signature: tuple[int, int] | None = None
        self._cached_runs: OrderedDict[str, dict[str, Any]] | None = None
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = self.index_path.with_suffix(f"{self.index_path.suffix}.artifacts")
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def _file_signature_unlocked(self) -> tuple[int, int] | None:
        try:
            stat = self.index_path.stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _remember_unlocked(
        self,
        runs: OrderedDict[str, dict[str, Any]],
        signature: tuple[int, int] | None = None,
    ) -> OrderedDict[str, dict[str, Any]]:
        self._cached_signature = signature if signature is not None else self._file_signature_unlocked()
        self._cached_runs = runs
        return runs

    def _read_unlocked(self) -> OrderedDict[str, dict[str, Any]]:
        signature = self._file_signature_unlocked()
        if self._cached_runs is not None and signature == self._cached_signature:
            return self._cached_runs

        try:
            with self.index_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return self._remember_unlocked(OrderedDict(), signature)
        except json.JSONDecodeError:
            logger.warning("GraphRAG pipeline run index is corrupted; ignoring it", extra={"path": str(self.index_path)})
            return self._remember_unlocked(OrderedDict(), signature)

        runs = payload.get("runs", []) if isinstance(payload, dict) else payload
        if not isinstance(runs, list):
            return self._remember_unlocked(OrderedDict(), signature)

        ordered: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for run in runs[-self.max_runs:]:
            if not isinstance(run, dict):
                continue
            run_id = str(run.get("run_id") or "")
            if run_id:
                ordered[run_id] = run
        return self._remember_unlocked(ordered, signature)

    def _write_unlocked(self, runs: OrderedDict[str, dict[str, Any]]) -> None:
        stored_runs = [
            self._compact_run_for_storage(run)
            for run in list(runs.values())[-self.max_runs:]
        ]
        payload = {
            "version": 1,
            "runs": stored_runs,
        }
        tmp_path = self.index_path.with_suffix(f"{self.index_path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        tmp_path.replace(self.index_path)
        self._remember_unlocked(runs)

    def _run_artifact_dir(self, run_id: str) -> Path:
        return self.artifact_dir / run_id

    def _write_artifact_bytes(self, run_id: str, filename: str, content: bytes) -> str:
        run_dir = self._run_artifact_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / filename
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        with tmp_path.open("wb") as handle:
            handle.write(content)
        tmp_path.replace(path)
        return str(path)

    def _compact_response_payload(self, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return self._compact_json_value(payload)
        summary: dict[str, Any] = {
            key: self._compact_json_value(payload.get(key), key)
            for key in (
                "status",
                "processed",
                "failed",
                "total_files",
                "document_count",
                "message",
                "error",
                "detail",
            )
            if key in payload
        }
        if not summary:
            summary["keys"] = sorted(str(key) for key in payload.keys())[:24]
        if isinstance(payload.get("results"), list):
            summary["result_count"] = len(payload["results"])
            summary["results"] = [
                self._compact_json_value(item, "results")
                for item in payload["results"][:5]
            ]
        return summary

    def _compact_json_value(self, value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            compacted: dict[str, Any] = {}
            for child_key, child_value in value.items():
                child_key_text = str(child_key)
                if child_key_text == "response":
                    compacted[child_key_text] = self._compact_response_payload(child_value)
                else:
                    compacted[child_key_text] = self._compact_json_value(child_value, child_key_text)
            return compacted
        if isinstance(value, list):
            limit = PERSISTED_LIST_LIMITS.get(str(key or ""), PERSISTED_DEFAULT_LIST_LIMIT)
            return [
                self._compact_json_value(item, key)
                for item in value[:limit]
            ]
        if isinstance(value, str) and len(value) > PERSISTED_STRING_LIMIT:
            omitted = len(value) - PERSISTED_STRING_LIMIT
            return (
                value[:PERSISTED_STRING_LIMIT].rstrip()
                + f"\n...[truncated {omitted} chars in persisted trace]"
            )
        return value

    def _compact_run_for_storage(self, run: dict[str, Any]) -> dict[str, Any]:
        run_id = str(run.get("run_id") or "")
        stored = deepcopy(run)
        if not run_id:
            return self._compact_json_value(stored)

        source_artifact = stored.get("source_document_artifact")
        if isinstance(source_artifact, dict):
            raw_content: bytes | None = None
            content = source_artifact.pop("content", None)
            if isinstance(content, bytes):
                raw_content = content
            elif isinstance(content, str):
                raw_content = content.encode("utf-8")

            encoded_content = source_artifact.pop("content_base64", None)
            if raw_content is None and encoded_content:
                try:
                    raw_content = base64.b64decode(str(encoded_content), validate=True)
                except Exception:
                    raw_content = None

            if raw_content is not None:
                source_artifact["content_storage_path"] = self._write_artifact_bytes(
                    run_id,
                    SOURCE_ARTIFACT_FILENAME,
                    raw_content,
                )
                source_artifact["externalized"] = True

        structured_artifact = stored.get("structured_ingest_artifact")
        if isinstance(structured_artifact, dict):
            content = structured_artifact.pop("content", None)
            if isinstance(content, str) and content:
                structured_artifact["content_storage_path"] = self._write_artifact_bytes(
                    run_id,
                    STRUCTURED_ARTIFACT_FILENAME,
                    content.encode("utf-8"),
                )
                structured_artifact["externalized"] = True
        return self._compact_json_value(stored)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._read_unlocked().values())

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._read_unlocked().get(str(run_id))

    def upsert(self, run: dict[str, Any]) -> dict[str, Any]:
        run_id = str(run.get("run_id") or "")
        if not run_id:
            return run

        with self._lock:
            runs = self._read_unlocked()
            if run_id in runs:
                runs.pop(run_id)
            runs[run_id] = run
            while len(runs) > self.max_runs:
                runs.popitem(last=False)
            self._write_unlocked(runs)
        return run


_pipeline_run_store: PipelineRunStore | None = None


def get_pipeline_run_store() -> PipelineRunStore:
    global _pipeline_run_store
    if _pipeline_run_store is None:
        _pipeline_run_store = PipelineRunStore()
    return _pipeline_run_store


def reset_pipeline_run_store(store: PipelineRunStore | None = None) -> None:
    global _pipeline_run_store
    _pipeline_run_store = store
