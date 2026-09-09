"""
GraphRAG Ingest Routes

Document upload, batch upload, rebuild KG, entity extraction,
entity linking, disambiguation, hybrid processing.

All endpoints use the shared connection-pooled HTTP client.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response, UploadFile, File, Form
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List
from urllib.parse import quote
import asyncio
import base64
import hashlib
import httpx
import json
import logging
import os
import re
import time

from .helpers import check_graphrag_enabled, compatible_health_urls, compatible_ingest_urls, get_http_client, graphrag_url as _graphrag_url
from .upstream import _build_deployment_diagnostics
from modules.graphrag.config import get_graphrag_config
from modules.graphrag.document_pipeline import build_document_pipeline_preview
from modules.graphrag.pipeline_runs import get_pipeline_run_store

logger = logging.getLogger(__name__)

router = APIRouter()

_PIPELINE_RUNS: dict[str, dict[str, Any]] = {}
_MAX_PIPELINE_RUNS = 100
_PIPELINE_RUNS_LOADED = False
GRAPHRAG_DEPLOYMENT_HINT = "GRAPHRAG_ENABLED=true docker compose --profile graphrag up -d"
GRAPHRAG_SERVICES_STATUS_PATH = "/api/v1/graphrag/services/status"
GRAPHRAG_DEPLOYMENT_DIAGNOSTICS_PATH = "/api/v1/graphrag/deployment/diagnostics"
GRAPHRAG_INGEST_FALLBACK_STATUS_CODES = {404, 405, 415, 422, 501}
DEFAULT_SOURCE_DOCUMENT_ARTIFACT_MAX_BYTES = 8 * 1024 * 1024
DEFAULT_REMOTE_INGEST_TIMEOUT_SECONDS = 8.0
DEFAULT_REMOTE_STRUCTURED_INGEST_TIMEOUT_SECONDS = 20.0
DEFAULT_REMOTE_STRUCTURED_DIGEST_THRESHOLD_CHARS = 80_000
DEFAULT_REMOTE_STRUCTURED_DIGEST_MAX_CHARS = 60_000
DEFAULT_PIPELINE_BATCH_UPLOAD_MAX_BYTES = 128 * 1024 * 1024
REMOTE_STRUCTURED_DIGEST_PRIORITY_RE = re.compile(
    r"chart|table|form|field|connector|diagram|shape|styled|speaker|note|image|"
    r"formula|metadata|relationship|evidence|revenue|risk|forecast|demand|driver|"
    r"source|target|confidence|entity",
    re.IGNORECASE,
)

PIPELINE_ARTIFACT_KIND_ALIASES = {
    "structured": "structured_ingest",
    "structured-text": "structured_ingest",
    "structured_text": "structured_ingest",
    "structured-ingest": "structured_ingest",
    "structured_ingest_document": "structured_ingest",
    "source": "source_document",
    "source-document": "source_document",
    "source_document": "source_document",
    "original": "source_document",
    "original-file": "source_document",
    "original_file": "source_document",
    "graph": "local_graph",
    "local-graph": "local_graph",
    "local_graph_artifact": "local_graph",
    "context": "local_context",
    "local-context": "local_context",
    "local_context_cache": "local_context",
    "reasoning": "local_graph_reasoning",
    "local-reasoning": "local_graph_reasoning",
    "local-graph-reasoning": "local_graph_reasoning",
    "local_graph_reasoning_index": "local_graph_reasoning",
    "handoff": "interaction_handoff",
    "interaction-handoff": "interaction_handoff",
    "interaction": "interaction_handoff",
    "stages": "stages",
    "samples": "samples",
    "profile": "document_profile",
    "document-profile": "document_profile",
    "summary": "summary",
    "manifest": "manifest",
    "full": "full",
    "run": "full",
}


def _file_tuple(filename: str | None, content: bytes, content_type: str | None = None):
    return (
        filename or "document.txt",
        content,
        content_type or "text/plain",
    )


def _pipeline_batch_upload_max_bytes() -> int:
    raw_value = os.getenv("GRAPHRAG_PIPELINE_BATCH_UPLOAD_MAX_BYTES")
    if raw_value in (None, ""):
        return DEFAULT_PIPELINE_BATCH_UPLOAD_MAX_BYTES
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid GRAPHRAG_PIPELINE_BATCH_UPLOAD_MAX_BYTES; using default",
        )
        return DEFAULT_PIPELINE_BATCH_UPLOAD_MAX_BYTES
    return max(1, parsed)


async def _close_upload_file(file: UploadFile) -> None:
    close = getattr(file, "close", None)
    if not close:
        return
    try:
        result = close()
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        logger.debug("Failed to close uploaded GraphRAG file", exc_info=True)


async def _materialize_pipeline_batch_uploads(
    files: list[UploadFile],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """
    Read request-backed upload streams before concurrent pipeline work starts.

    FastAPI UploadFile objects may be backed by spooled temporary files owned by
    the multipart request parser. Keeping graph extraction concurrency on
    immutable bytes avoids overlapping parser/file lifecycle with XML/PDF
    extraction and upstream submission.
    """
    max_total_bytes = _pipeline_batch_upload_max_bytes()
    total_bytes = 0
    materialized: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, file in enumerate(files):
        filename = file.filename or f"file-{index + 1}"
        item_started_at = datetime.now(timezone.utc).isoformat()
        item_start = time.perf_counter()
        try:
            content = await file.read()
            total_bytes += len(content)
            if total_bytes > max_total_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "GraphRAG batch upload is too large "
                        f"({total_bytes} bytes > {max_total_bytes} bytes). "
                        "Set GRAPHRAG_PIPELINE_BATCH_UPLOAD_MAX_BYTES to raise the limit."
                    ),
                )
            materialized.append({
                "index": index,
                "filename": filename,
                "content": content,
                "content_type": file.content_type or "application/octet-stream",
                "started_at": item_started_at,
                "read_duration_ms": _elapsed_ms(item_start),
                "size_bytes": len(content),
            })
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning(f"GraphRAG pipeline batch read failed for {filename}: {exc}")
            failures.append({
                "index": index,
                "filename": filename,
                "success": False,
                "status": "upload_read_failed",
                "error": str(exc),
                "started_at": item_started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": _elapsed_ms(item_start),
                "size_bytes": 0,
            })
        finally:
            await _close_upload_file(file)

    return materialized, failures, total_bytes


def _compatible_ingest_urls(service_url: str | None) -> list[str]:
    return compatible_ingest_urls(service_url)


def _compatible_health_urls(service_url: str | None) -> list[str]:
    return compatible_health_urls(service_url)


def _remote_ingest_timeout_seconds(payload_kind: str = "", payload_bytes: int = 0) -> float:
    raw_value = os.getenv("GRAPHRAG_REMOTE_INGEST_TIMEOUT")
    if raw_value is not None:
        try:
            parsed = float(raw_value)
        except (TypeError, ValueError):
            return DEFAULT_REMOTE_INGEST_TIMEOUT_SECONDS
        return max(1.0, min(parsed, 120.0))

    structured_timeout = os.getenv("GRAPHRAG_REMOTE_STRUCTURED_INGEST_TIMEOUT")
    if (
        str(payload_kind or "").lower() == "structured_text"
        or payload_bytes >= _remote_structured_digest_threshold_chars()
    ):
        if structured_timeout not in (None, ""):
            try:
                parsed = float(structured_timeout)
            except (TypeError, ValueError):
                return DEFAULT_REMOTE_STRUCTURED_INGEST_TIMEOUT_SECONDS
            return max(1.0, min(parsed, 120.0))
        return DEFAULT_REMOTE_STRUCTURED_INGEST_TIMEOUT_SECONDS

    return DEFAULT_REMOTE_INGEST_TIMEOUT_SECONDS


def _remote_structured_digest_threshold_chars() -> int:
    raw_value = os.getenv("GRAPHRAG_REMOTE_STRUCTURED_DIGEST_THRESHOLD_CHARS")
    if raw_value in (None, ""):
        return DEFAULT_REMOTE_STRUCTURED_DIGEST_THRESHOLD_CHARS
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return DEFAULT_REMOTE_STRUCTURED_DIGEST_THRESHOLD_CHARS


def _remote_structured_digest_max_chars() -> int:
    raw_value = os.getenv("GRAPHRAG_REMOTE_STRUCTURED_DIGEST_MAX_CHARS")
    if raw_value in (None, ""):
        return DEFAULT_REMOTE_STRUCTURED_DIGEST_MAX_CHARS
    try:
        return max(4_000, int(raw_value))
    except (TypeError, ValueError):
        return DEFAULT_REMOTE_STRUCTURED_DIGEST_MAX_CHARS


def _load_persisted_pipeline_runs() -> None:
    global _PIPELINE_RUNS_LOADED
    if _PIPELINE_RUNS_LOADED:
        return
    _PIPELINE_RUNS_LOADED = True
    try:
        for run in get_pipeline_run_store().list_runs()[-_MAX_PIPELINE_RUNS:]:
            run_id = str(run.get("run_id") or "")
            if run_id:
                _PIPELINE_RUNS[run_id] = run
    except Exception as exc:
        logger.warning(f"Unable to load persisted GraphRAG pipeline runs: {exc}")


def _remember_pipeline_run(run: dict[str, Any]) -> dict[str, Any]:
    _load_persisted_pipeline_runs()
    run_id = str(run.get("run_id"))
    if not run_id:
        return run
    _PIPELINE_RUNS[run_id] = run
    while len(_PIPELINE_RUNS) > _MAX_PIPELINE_RUNS:
        oldest_id = next(iter(_PIPELINE_RUNS))
        _PIPELINE_RUNS.pop(oldest_id, None)
    try:
        get_pipeline_run_store().upsert(run)
    except Exception as exc:
        logger.warning(f"Unable to persist GraphRAG pipeline run {run_id}: {exc}")
    return run


def _normalize_pipeline_artifact_kind(kind: str | None) -> str:
    normalized = str(kind or "manifest").strip().lower().replace(" ", "-")
    return PIPELINE_ARTIFACT_KIND_ALIASES.get(normalized, normalized.replace("-", "_"))


def _pipeline_artifact_payload(run: dict[str, Any], kind: str) -> tuple[Any, str, str]:
    artifact_kind = _normalize_pipeline_artifact_kind(kind)
    document = run.get("document", {}) if isinstance(run.get("document"), dict) else {}
    filename = str(document.get("filename") or "document")
    run_id = str(run.get("run_id") or "pipeline-run")

    if artifact_kind == "manifest":
        return _pipeline_artifact_manifest(run), "application/json", f"{run_id}.artifacts.json"
    if artifact_kind == "full":
        return run, "application/json", f"{run_id}.json"
    if artifact_kind == "source_document":
        artifact = run.get("source_document_artifact") if isinstance(run.get("source_document_artifact"), dict) else {}
        content = artifact.get("content")
        if content in (None, "") and artifact.get("content_base64"):
            try:
                content = base64.b64decode(str(artifact.get("content_base64")), validate=True)
            except Exception as exc:
                raise HTTPException(status_code=500, detail="Source document artifact is invalid") from exc
        if content in (None, "") and artifact.get("content_storage_path"):
            try:
                content = Path(str(artifact.get("content_storage_path"))).read_bytes()
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Source document artifact file is missing") from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail="Source document artifact file could not be read") from exc
        if content in (None, ""):
            raise HTTPException(status_code=404, detail="Source document artifact is not available for this run")
        return (
            content,
            str(artifact.get("content_type") or document.get("content_type") or "application/octet-stream"),
            str(artifact.get("filename") or filename),
        )
    if artifact_kind == "structured_ingest":
        artifact = run.get("structured_ingest_artifact") if isinstance(run.get("structured_ingest_artifact"), dict) else {}
        content = artifact.get("content")
        if content in (None, "") and artifact.get("content_storage_path"):
            try:
                content = Path(str(artifact.get("content_storage_path"))).read_text(encoding="utf-8")
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Structured ingest artifact file is missing") from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail="Structured ingest artifact file could not be read") from exc
        if content in (None, ""):
            stage_text = None
            for stage in run.get("stages", []) if isinstance(run.get("stages"), list) else []:
                if isinstance(stage, dict) and stage.get("name") == "structured_ingest_document":
                    sample_outputs = stage.get("sample_outputs", {}) if isinstance(stage.get("sample_outputs"), dict) else {}
                    stage_text = sample_outputs.get("text")
                    break
            content = stage_text
        if content in (None, ""):
            raise HTTPException(status_code=404, detail="Structured ingest artifact is not available for this run")
        return str(content), "text/markdown; charset=utf-8", str(artifact.get("filename") or f"{Path(filename).stem or 'document'}.structured.md")
    if artifact_kind == "local_graph":
        return run.get("local_graph") or {}, "application/json", f"{run_id}.local-graph.json"
    if artifact_kind == "local_context":
        return run.get("local_context") or {}, "application/json", f"{run_id}.local-context.json"
    if artifact_kind == "local_graph_reasoning":
        return run.get("local_graph_reasoning") or {}, "application/json", f"{run_id}.local-graph-reasoning.json"
    if artifact_kind == "interaction_handoff":
        return run.get("interaction_handoff") or {}, "application/json", f"{run_id}.interaction-handoff.json"
    if artifact_kind == "stages":
        return run.get("stages") or [], "application/json", f"{run_id}.stages.json"
    if artifact_kind == "samples":
        return run.get("samples") or {}, "application/json", f"{run_id}.samples.json"
    if artifact_kind == "document_profile":
        return run.get("document_profile") or {}, "application/json", f"{run_id}.document-profile.json"
    if artifact_kind == "summary":
        return run.get("summary") or {}, "application/json", f"{run_id}.summary.json"

    raise HTTPException(
        status_code=400,
        detail=f"Unknown pipeline artifact kind '{kind}'. Use manifest, structured, local_graph, local_context, reasoning, handoff, stages, samples, profile, summary, or full.",
    )


def _artifact_available(payload: Any) -> bool:
    if payload is None:
        return False
    if isinstance(payload, (dict, list, tuple, set, str, bytes)):
        return bool(payload)
    return True


def _pipeline_artifact_manifest(run: dict[str, Any]) -> dict[str, Any]:
    run_id = str(run.get("run_id") or "")
    source_artifact = run.get("source_document_artifact") if isinstance(run.get("source_document_artifact"), dict) else {}
    document = run.get("document", {}) if isinstance(run.get("document"), dict) else {}
    structured_artifact = run.get("structured_ingest_artifact") if isinstance(run.get("structured_ingest_artifact"), dict) else {}
    artifact_specs = [
        (
            "source_document",
            "Original source document",
            source_artifact,
            str(source_artifact.get("content_type") or document.get("content_type") or "application/octet-stream"),
        ),
        ("structured_ingest", "Structured ingest Markdown", structured_artifact, "text/markdown"),
        ("local_graph", "Local graph artifact", run.get("local_graph"), "application/json"),
        ("local_context", "Local context cache", run.get("local_context"), "application/json"),
        ("local_graph_reasoning", "Local graph reasoning index", run.get("local_graph_reasoning"), "application/json"),
        ("interaction_handoff", "Interaction handoff", run.get("interaction_handoff"), "application/json"),
        ("stages", "Pipeline stages", run.get("stages"), "application/json"),
        ("samples", "Pipeline samples", run.get("samples"), "application/json"),
        ("document_profile", "Document profile", run.get("document_profile"), "application/json"),
        ("summary", "Run summary", run.get("summary"), "application/json"),
        ("full", "Full pipeline run", run, "application/json"),
    ]
    artifacts = []
    for kind, label, payload, content_type in artifact_specs:
        available = _artifact_available(payload)
        artifact: dict[str, Any] = {
            "kind": kind,
            "label": label,
            "available": available,
            "content_type": content_type,
            "endpoint": f"/api/v1/graphrag/ingest/pipeline/{run_id}/artifacts/{kind}" if run_id else None,
        }
        if kind == "structured_ingest" and structured_artifact:
            artifact.update({
                "filename": structured_artifact.get("filename"),
                "char_count": structured_artifact.get("char_count"),
                "sha256": structured_artifact.get("sha256"),
                "truncated": structured_artifact.get("truncated", False),
                "uses_structured_text": structured_artifact.get("uses_structured_text", False),
            })
        if kind == "source_document" and source_artifact:
            artifact.update({
                "filename": source_artifact.get("filename") or document.get("filename"),
                "size_bytes": source_artifact.get("size_bytes"),
                "sha256": source_artifact.get("sha256"),
            })
        artifacts.append(artifact)

    return {
        "run_id": run_id,
        "status": run.get("status"),
        "document": run.get("document"),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "artifact_count": sum(1 for artifact in artifacts if artifact["available"]),
        "artifacts": artifacts,
    }


def _attach_structured_ingest_artifact(run: dict[str, Any], structured_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = structured_payload if isinstance(structured_payload, dict) else run.pop("_structured_ingest_payload", {})
    if not isinstance(payload, dict):
        payload = {}
    content = str(payload.get("content") or "")
    if not content:
        return run
    structured = run.get("structured_ingest", {}) if isinstance(run.get("structured_ingest"), dict) else {}
    artifact = {
        "filename": payload.get("filename") or structured.get("filename"),
        "content_type": payload.get("content_type") or structured.get("content_type") or "text/markdown",
        "content": content,
        "char_count": len(content),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "truncated": structured.get("truncated", False),
        "uses_structured_text": bool(payload.get("uses_structured_text", structured.get("uses_structured_text", False))),
    }
    run["structured_ingest_artifact"] = artifact
    if isinstance(run.get("structured_ingest"), dict):
        run["structured_ingest"] = {
            **run["structured_ingest"],
            "artifact_available": True,
            "artifact_endpoint": f"/api/v1/graphrag/ingest/pipeline/{run.get('run_id')}/artifacts/structured_ingest",
            "artifact_sha256": artifact["sha256"],
        }
    return run


def _source_document_artifact_max_bytes() -> int:
    raw_value = os.getenv("GRAPHRAG_SOURCE_DOCUMENT_ARTIFACT_MAX_BYTES")
    if raw_value in (None, ""):
        return DEFAULT_SOURCE_DOCUMENT_ARTIFACT_MAX_BYTES
    try:
        return max(0, int(raw_value))
    except ValueError:
        logger.warning(
            "Invalid GRAPHRAG_SOURCE_DOCUMENT_ARTIFACT_MAX_BYTES; using default",
            extra={"value": raw_value},
        )
        return DEFAULT_SOURCE_DOCUMENT_ARTIFACT_MAX_BYTES


def _attach_source_document_artifact(
    run: dict[str, Any],
    *,
    filename: str | None,
    content: bytes,
    content_type: str | None,
) -> dict[str, Any]:
    max_bytes = _source_document_artifact_max_bytes()
    document = run.get("document", {}) if isinstance(run.get("document"), dict) else {}
    if not content or max_bytes <= 0 or len(content) > max_bytes:
        if isinstance(run.get("document"), dict):
            run["document"] = {
                **document,
                "source_artifact_available": False,
                "source_artifact_skipped_reason": (
                    "disabled"
                    if max_bytes <= 0
                    else f"larger than {max_bytes} byte source artifact limit"
                ),
            }
        return run

    source_filename = filename or str(document.get("filename") or "document")
    artifact = {
        "filename": source_filename,
        "content_type": content_type or str(document.get("content_type") or "application/octet-stream"),
        "content_base64": base64.b64encode(content).decode("ascii"),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    run["source_document_artifact"] = artifact
    if isinstance(run.get("document"), dict):
        run["document"] = {
            **document,
            "source_artifact_available": True,
            "source_artifact_endpoint": f"/api/v1/graphrag/ingest/pipeline/{run.get('run_id')}/artifacts/source_document",
            "source_artifact_sha256": artifact["sha256"],
            "source_artifact_size_bytes": artifact["size_bytes"],
        }
    return run


def _get_pipeline_run_or_404(run_id: str) -> dict[str, Any]:
    _load_persisted_pipeline_runs()
    run = _PIPELINE_RUNS.get(run_id)
    if not run:
        try:
            run = get_pipeline_run_store().get(run_id)
            if run:
                _PIPELINE_RUNS[run_id] = run
        except Exception as exc:
            logger.warning(f"Unable to load GraphRAG pipeline run {run_id}: {exc}")
    if not run:
        raise HTTPException(status_code=404, detail="GraphRAG ingestion pipeline run not found")
    return run


def _compact_summary_text(value: Any, *, limit: int = 520) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return f"{text[:max(1, limit - 3)].strip()}..."


def _markdown_digest_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"Preamble": []}
    current = "Preamble"
    for raw_line in str(text or "").splitlines():
        header = re.match(r"^##\s+(.+?)\s*$", raw_line.strip())
        if header:
            current = header.group(1).strip()
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(raw_line)
    return sections


def _digest_line(value: Any, *, limit: int = 520) -> str:
    return _compact_summary_text(value, limit=limit)


def _select_remote_digest_lines(
    lines: list[str],
    *,
    limit: int,
    fallback_limit: int = 0,
    line_limit: int = 520,
) -> list[str]:
    priority: list[str] = []
    fallback: list[str] = []
    seen: set[str] = set()
    for raw_line in lines:
        line = _digest_line(raw_line, limit=line_limit)
        if not line or line.startswith("### "):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        if REMOTE_STRUCTURED_DIGEST_PRIORITY_RE.search(line):
            priority.append(line)
        else:
            fallback.append(line)

    selected = priority[:limit]
    remaining = max(0, limit - len(selected))
    if remaining and fallback_limit:
        selected.extend(fallback[:min(remaining, fallback_limit)])
    return selected


def _remote_digest_chunk_groups(lines: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    for raw_line in lines:
        line = str(raw_line or "").strip()
        if line.startswith("### "):
            if current:
                groups.append(current)
            current = [line]
            continue
        if current:
            current.append(line)
        elif line:
            current = [line]
    if current:
        groups.append(current)
    return groups


def _select_remote_digest_chunks(lines: list[str], *, limit: int) -> list[str]:
    groups = _remote_digest_chunk_groups(lines)
    priority_groups: list[list[str]] = []
    fallback_groups: list[list[str]] = []
    seen: set[str] = set()
    for group in groups:
        group_text = _digest_line(" ".join(group), limit=900)
        if not group_text:
            continue
        key = group_text.lower()
        if key in seen:
            continue
        seen.add(key)
        if REMOTE_STRUCTURED_DIGEST_PRIORITY_RE.search(group_text):
            priority_groups.append(group)
        else:
            fallback_groups.append(group)

    selected_groups = (priority_groups + fallback_groups)[:limit]
    selected: list[str] = []
    for index, group in enumerate(selected_groups, start=1):
        heading = next((line for line in group if str(line).strip().startswith("### ")), f"### representative_chunk_{index}")
        body_lines = [line for line in group if not str(line).strip().startswith("### ") and str(line).strip()]
        selected.append(str(heading).strip())
        selected.append(f"- {_digest_line(' '.join(body_lines), limit=900)}")
    return selected


def _append_remote_digest_section(lines: list[str], title: str, entries: list[str]) -> None:
    entries = [entry for entry in entries if str(entry or "").strip()]
    if not entries:
        return
    if lines and lines[-1] != "":
        lines.append("")
    lines.append(f"## {title}")
    lines.extend(entries)
    lines.append("")


def _clip_remote_structured_digest(text: str, *, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    marker = "\n\n[Remote structured digest truncated by llama-nexus; full structured artifact remains available locally.]\n"
    clip_at = max(0, max_chars - len(marker))
    return text[:clip_at].rstrip() + marker, True


def _build_remote_structured_ingest_digest(
    structured_text: str,
    *,
    filename: str | None,
) -> tuple[str, dict[str, Any]]:
    source_char_count = len(structured_text)
    threshold = _remote_structured_digest_threshold_chars()
    max_chars = _remote_structured_digest_max_chars()
    profile: dict[str, Any] = {
        "remote_digest_used": False,
        "remote_source_char_count": source_char_count,
        "remote_payload_char_count": source_char_count,
        "remote_digest_threshold_chars": threshold,
        "remote_digest_max_chars": max_chars,
    }
    if source_char_count <= threshold:
        return structured_text, profile

    sections = _markdown_digest_sections(structured_text)
    document_name = Path(filename or "document").name
    source_lines = _select_remote_digest_lines(
        sections.get("Source", []),
        limit=18,
        fallback_limit=18,
        line_limit=420,
    )
    source_lines.extend([
        f"- Full structured characters: {source_char_count}",
        f"- Remote payload profile: graph-focused digest; full local artifact retained for citations.",
    ])

    entity_lines = _select_remote_digest_lines(
        sections.get("Candidate Entities", []),
        limit=70,
        fallback_limit=70,
        line_limit=320,
    )
    relationship_lines = _select_remote_digest_lines(
        sections.get("Candidate Graph Relationships", []),
        limit=140,
        fallback_limit=140,
        line_limit=520,
    )
    structure_lines = _select_remote_digest_lines(
        sections.get("Extracted Structure", []),
        limit=100,
        fallback_limit=24,
        line_limit=520,
    )
    chunk_lines = _select_remote_digest_chunks(
        sections.get("Embedding Chunks", []),
        limit=42,
    )

    digest_lines = [
        f"# GraphRAG Remote Ingest Digest: {document_name}",
        "",
        "This digest preserves the highest-signal graph facts for remote indexing.",
    ]
    _append_remote_digest_section(digest_lines, "Source", source_lines)
    _append_remote_digest_section(digest_lines, "Candidate Entities", entity_lines)
    _append_remote_digest_section(digest_lines, "Candidate Graph Relationships", relationship_lines)
    _append_remote_digest_section(digest_lines, "High Signal Extracted Structure", structure_lines)
    _append_remote_digest_section(digest_lines, "Representative Chunks", chunk_lines)

    digest_text = "\n".join(digest_lines).strip() + "\n"
    digest_text, truncated = _clip_remote_structured_digest(digest_text, max_chars=max_chars)
    profile.update({
        "remote_digest_used": True,
        "remote_payload_char_count": len(digest_text),
        "remote_payload_bytes": len(digest_text.encode("utf-8")),
        "remote_digest_truncated": truncated,
        "remote_digest_reason": (
            "Structured document exceeded remote digest threshold; submitted graph-focused digest upstream."
        ),
        "remote_digest_sections": {
            "source": len(source_lines),
            "entities": len(entity_lines),
            "relationships": len(relationship_lines),
            "structure": len(structure_lines),
            "chunks": len(chunk_lines),
        },
    })
    return digest_text, profile


def _compact_prompt_card(prompt: dict[str, Any], *, limit: int = 520) -> dict[str, Any]:
    keys = (
        "id",
        "kind",
        "label",
        "focus",
        "source",
        "target",
        "bridge",
        "relation",
        "score",
        "priority",
    )
    compact = {
        key: prompt.get(key)
        for key in keys
        if prompt.get(key) not in (None, "", [], {})
    }
    if prompt.get("prompt"):
        compact["prompt"] = _compact_summary_text(prompt.get("prompt"), limit=limit)
    return compact


def _compact_capability_card(card: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "kind", "label", "count", "detail", "status", "ready")
    return {
        key: card.get(key)
        for key in keys
        if card.get(key) not in (None, "", [], {})
    }


def _build_trace_summary_prompt(filename: str) -> str:
    safe_filename = _compact_summary_text(filename or "this upload", limit=96)
    return (
        f"Use the recent GraphRAG pipeline trace for {safe_filename}. "
        "Keep the answer bounded and complete: return exactly 4 bullets under "
        "180 words total named Entities, Top relationships, Best path, and "
        "Source proof. For Top relationships, list at most 4 direct "
        "relationships; prioritize gate, block, require, support, drive, "
        "reference, source, visual, or connector edges before incidental "
        "containment or row-order summaries. Cite pages, slides, chunks, or "
        "artifacts when available. Do not add a trailing paragraph, table, or "
        "exhaustive list."
    )


def _is_trace_summary_prompt(prompt: dict[str, Any]) -> bool:
    kind = str(prompt.get("kind") or "").lower()
    label = str(prompt.get("label") or "").lower()
    text = str(prompt.get("prompt") or "").lower()
    return (
        kind == "summary"
        or "summarize" in label
        or "summary" in label
        or "summarize" in text
        or "summary" in text
        or "trace summary" in text
    )


def _pipeline_resume_handoff(run: dict[str, Any]) -> dict[str, Any] | None:
    run_id = str(run.get("run_id") or "").strip()
    if not run_id:
        return None

    document = run.get("document", {}) if isinstance(run.get("document"), dict) else {}
    filename = str(document.get("filename") or "pipeline run")
    domain = str(document.get("domain") or "general")
    summary = run.get("summary", {}) if isinstance(run.get("summary"), dict) else {}
    interaction_handoff = run.get("interaction_handoff", {}) if isinstance(run.get("interaction_handoff"), dict) else {}
    interaction_suggestions = run.get("interaction_suggestions", {}) if isinstance(run.get("interaction_suggestions"), dict) else {}
    operational_handoff = run.get("operational_handoff", {}) if isinstance(run.get("operational_handoff"), dict) else {}
    counts = interaction_handoff.get("counts") if isinstance(interaction_handoff.get("counts"), dict) else {}
    artifact_manifest = _pipeline_artifact_manifest(run)

    starter_prompts = [
        _compact_prompt_card(prompt)
        for prompt in (interaction_handoff.get("starter_prompts", []) or [])
        if isinstance(prompt, dict)
    ]
    if not starter_prompts:
        starter_prompts = [
            _compact_prompt_card(prompt)
            for prompt in (interaction_suggestions.get("suggested_questions", []) or [])
            if isinstance(prompt, dict)
        ][:5]
    else:
        starter_prompts = starter_prompts[:5]
    trace_summary_prompt = _build_trace_summary_prompt(filename)
    if starter_prompts:
        starter_prompts = [
            {
                **item,
                "prompt": trace_summary_prompt,
                "kind": item.get("kind") or "summary",
                "label": item.get("label") or "Summarize trace",
                "focus": item.get("focus") or "trace_summary",
            }
            if _is_trace_summary_prompt(item)
            else item
            for item in starter_prompts
        ]
    else:
        starter_prompts = [
            {
                "id": "source_trace_summary",
                "kind": "summary",
                "label": "Summarize trace",
                "focus": "trace_summary",
                "prompt": trace_summary_prompt,
            }
        ]

    prompt = next(
        (
            str(item.get("prompt"))
            for item in starter_prompts
            if item.get("prompt")
        ),
        trace_summary_prompt,
    )
    chat_context_request = (
        dict(interaction_handoff.get("chat_context_request"))
        if isinstance(interaction_handoff.get("chat_context_request"), dict)
        else {}
    )
    chat_context_request.update({
        "query": chat_context_request.get("query") or prompt,
        "domain": chat_context_request.get("domain") or domain,
        "pipeline_run_ids": [run_id],
        "include_graph": chat_context_request.get("include_graph", True),
        "use_local_pipeline_context": chat_context_request.get("use_local_pipeline_context", True),
    })

    available_artifacts = [
        {
            "kind": artifact.get("kind"),
            "label": artifact.get("label"),
            "endpoint": artifact.get("endpoint"),
            "content_type": artifact.get("content_type"),
        }
        for artifact in artifact_manifest.get("artifacts", [])
        if isinstance(artifact, dict) and artifact.get("available")
    ]
    raw_capability_cards = interaction_handoff.get("capability_cards")
    capability_cards = [
        _compact_capability_card(card)
        for card in (raw_capability_cards if isinstance(raw_capability_cards, list) else [])[:6]
        if isinstance(card, dict)
    ]
    suggested_questions = [
        {
            key: value
            for key, value in {
                "id": prompt_card.get("id"),
                "kind": prompt_card.get("kind"),
                "label": prompt_card.get("label"),
                "prompt": prompt_card.get("prompt"),
                "source": prompt_card.get("source"),
                "target": prompt_card.get("target"),
                "bridge": prompt_card.get("bridge"),
                "relation": prompt_card.get("relation"),
                "focus": prompt_card.get("focus"),
            }.items()
            if value not in (None, "", [], {})
        }
        for prompt_card in starter_prompts[:4]
        if prompt_card.get("prompt")
    ]
    citation_scopes = [
        str(scope)
        for scope in (interaction_handoff.get("citation_scopes", []) or [])
        if scope
    ][:8]
    resume_ready = bool(
        interaction_handoff.get("ready")
        or summary.get("chunk_count", 0)
        or summary.get("relationship_candidate_count", 0)
    )
    raw_actions = operational_handoff.get("actions")
    actions = [
        {
            "id": action.get("id"),
            "label": action.get("label"),
            "ready": action.get("ready", False),
            "target": action.get("target"),
        }
        for action in (raw_actions if isinstance(raw_actions, list) else [])[:6]
        if isinstance(action, dict)
    ]
    if resume_ready and not any(action.get("id") == "ask_chat" for action in actions):
        actions.append({
            "id": "ask_chat",
            "label": "Ask Chat",
            "ready": True,
            "target": "chat",
        })

    route_handoff = {
        "runId": run_id,
        "runIds": [run_id],
        "filename": filename,
        "domain": domain,
        "prompt": prompt,
        "chatContextEndpoint": interaction_handoff.get("chat_context_endpoint") or "/api/v1/graphrag/chat-context-v2",
        "chatContextRequest": chat_context_request,
        "suggestedQuestions": suggested_questions,
        "recommendedMode": interaction_handoff.get("recommended_mode"),
        "answerGuidance": interaction_handoff.get("answer_guidance"),
        "citationScopes": citation_scopes,
        "capabilityCards": capability_cards,
        "graphAnchorCount": counts.get("graph_anchors", len(interaction_handoff.get("graph_anchors", []) or [])),
        "retrievalRehearsalCount": counts.get("retrieval_rehearsals", len(interaction_handoff.get("retrieval_rehearsals", []) or [])),
    }

    return {
        "ready": resume_ready,
        "run_id": run_id,
        "filename": filename,
        "domain": domain,
        "recommended_mode": interaction_handoff.get("recommended_mode"),
        "answer_guidance": interaction_handoff.get("answer_guidance"),
        "citation_scopes": citation_scopes,
        "starter_prompts": starter_prompts,
        "suggested_questions": suggested_questions,
        "chat_context_endpoint": interaction_handoff.get("chat_context_endpoint") or "/api/v1/graphrag/chat-context-v2",
        "chat_context_request": chat_context_request,
        "route_state": {"graphragHandoff": route_handoff},
        "artifact_manifest_endpoint": f"/api/v1/graphrag/ingest/pipeline/{run_id}/artifacts",
        "available_artifacts": available_artifacts[:8],
        "actions": actions,
    }


def _pipeline_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    document = run.get("document", {}) if isinstance(run.get("document"), dict) else {}
    summary = run.get("summary", {}) if isinstance(run.get("summary"), dict) else {}
    document_profile = run.get("document_profile", {}) if isinstance(run.get("document_profile"), dict) else {}
    structured_ingest = run.get("structured_ingest", {}) if isinstance(run.get("structured_ingest"), dict) else {}
    upstream_preflight = run.get("upstream_preflight", {}) if isinstance(run.get("upstream_preflight"), dict) else {}
    upstream_ingest = run.get("upstream_ingest", {}) if isinstance(run.get("upstream_ingest"), dict) else {}
    operational_handoff = run.get("operational_handoff", {}) if isinstance(run.get("operational_handoff"), dict) else {}
    deployment_outcome = run.get("deployment_outcome", {}) if isinstance(run.get("deployment_outcome"), dict) else {}
    pipeline_performance = run.get("pipeline_performance", {}) if isinstance(run.get("pipeline_performance"), dict) else {}
    local_graph = run.get("local_graph", {}) if isinstance(run.get("local_graph"), dict) else {}
    local_context = run.get("local_context", {}) if isinstance(run.get("local_context"), dict) else {}
    interaction_suggestions = run.get("interaction_suggestions", {}) if isinstance(run.get("interaction_suggestions"), dict) else {}
    interaction_handoff = run.get("interaction_handoff", {}) if isinstance(run.get("interaction_handoff"), dict) else {}
    preflight_samples = upstream_preflight.get("sample_outputs", {}) if isinstance(upstream_preflight.get("sample_outputs"), dict) else {}
    upstream_samples = upstream_ingest.get("sample_outputs", {}) if isinstance(upstream_ingest.get("sample_outputs"), dict) else {}
    stages = run.get("stages", []) if isinstance(run.get("stages"), list) else []
    readiness_score = summary.get("readiness_score")
    if readiness_score is None:
        readiness_score = document_profile.get("readiness_score")

    return {
        "run_id": run.get("run_id"),
        "status": run.get("status"),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "document": {
            "filename": document.get("filename"),
            "domain": document.get("domain"),
            "content_type": document.get("content_type"),
            "extension": document.get("extension"),
            "size_bytes": document.get("size_bytes"),
        },
        "summary": {
            "extraction_method": summary.get("extraction_method"),
            "document_complexity": summary.get("document_complexity") or document_profile.get("complexity"),
            "readiness_score": readiness_score,
            "detected_capabilities": summary.get("detected_capabilities") or document_profile.get("detected_capabilities", []),
            "element_count": summary.get("element_count", 0),
            "chunk_count": summary.get("chunk_count", 0),
            "entity_candidate_count": summary.get("entity_candidate_count", 0),
            "relationship_candidate_count": summary.get("relationship_candidate_count", 0),
            "local_graph_node_count": local_graph.get("node_count", 0),
            "local_graph_edge_count": local_graph.get("edge_count", 0),
            "local_graph_truncated": local_graph.get("truncated", False),
            "local_context_chunk_count": local_context.get("cached_chunk_count", 0),
            "local_context_graph_context_relationship_count": local_context.get("graph_context_relationship_count", 0),
            "local_context_truncated": local_context.get("truncated", False),
            "interaction_question_count": interaction_suggestions.get("question_count", 0),
            "interaction_handoff_ready": interaction_handoff.get("ready", False),
            "interaction_recommended_mode": interaction_handoff.get("recommended_mode"),
            "interaction_graph_anchor_count": (interaction_handoff.get("counts") or {}).get("graph_anchors", 0) if isinstance(interaction_handoff.get("counts"), dict) else 0,
            "interaction_capability_card_count": (interaction_handoff.get("counts") or {}).get("capability_cards", 0) if isinstance(interaction_handoff.get("counts"), dict) else 0,
            "interaction_retrieval_rehearsal_count": (interaction_handoff.get("counts") or {}).get("retrieval_rehearsals", 0) if isinstance(interaction_handoff.get("counts"), dict) else 0,
            "interaction_rehearsed_chunk_match_count": (interaction_handoff.get("counts") or {}).get("rehearsed_chunk_matches", 0) if isinstance(interaction_handoff.get("counts"), dict) else 0,
            "interaction_rehearsed_anchor_match_count": (interaction_handoff.get("counts") or {}).get("rehearsed_anchor_matches", 0) if isinstance(interaction_handoff.get("counts"), dict) else 0,
            "warning_count": len(summary.get("warnings", []) or []),
        },
        "document_profile": {
            "complexity": document_profile.get("complexity"),
            "readiness_score": document_profile.get("readiness_score"),
            "detected_capabilities": document_profile.get("detected_capabilities", []),
            "recommended_payload": document_profile.get("recommended_payload"),
            "structured_text_recommended": document_profile.get("structured_text_recommended", False),
            "quality_warning_count": len(document_profile.get("quality_warnings", []) or []),
            "evidence_profile": document_profile.get("evidence_profile") or summary.get("evidence_profile"),
        } if document_profile else None,
        "structured_ingest": {
            "filename": structured_ingest.get("filename"),
            "uses_structured_text": structured_ingest.get("uses_structured_text", False),
            "char_count": structured_ingest.get("char_count", 0),
            "truncated": structured_ingest.get("truncated", False),
            "artifact_available": structured_ingest.get("artifact_available", False) or bool(run.get("structured_ingest_artifact")),
            "artifact_endpoint": structured_ingest.get("artifact_endpoint") or (
                f"/api/v1/graphrag/ingest/pipeline/{run.get('run_id')}/artifacts/structured_ingest"
                if run.get("structured_ingest_artifact") else None
            ),
        },
        "artifacts": _pipeline_artifact_manifest(run),
        "interaction_suggestions": {
            "question_count": interaction_suggestions.get("question_count", 0),
            "suggested_questions": [
                {
                    "id": question.get("id"),
                    "kind": question.get("kind"),
                    "label": question.get("label"),
                }
                for question in interaction_suggestions.get("suggested_questions", [])[:4]
                if isinstance(question, dict)
            ],
        } if interaction_suggestions else None,
        "interaction_handoff": {
            "ready": interaction_handoff.get("ready", False),
            "recommended_mode": interaction_handoff.get("recommended_mode"),
            "chat_context_endpoint": interaction_handoff.get("chat_context_endpoint"),
            "lane_count": len(interaction_handoff.get("lanes", []) or []),
            "capability_card_count": len(interaction_handoff.get("capability_cards", []) or []),
            "graph_anchor_count": len(interaction_handoff.get("graph_anchors", []) or []),
            "starter_prompt_count": len(interaction_handoff.get("starter_prompts", []) or []),
            "retrieval_rehearsal_count": len(interaction_handoff.get("retrieval_rehearsals", []) or []),
            "rehearsed_chunk_match_count": (interaction_handoff.get("counts") or {}).get("rehearsed_chunk_matches", 0) if isinstance(interaction_handoff.get("counts"), dict) else 0,
            "rehearsed_anchor_match_count": (interaction_handoff.get("counts") or {}).get("rehearsed_anchor_matches", 0) if isinstance(interaction_handoff.get("counts"), dict) else 0,
            "citation_scopes": interaction_handoff.get("citation_scopes", []),
            "lanes": [
                {
                    "id": lane.get("id"),
                    "label": lane.get("label"),
                    "ready": lane.get("ready", False),
                    "count": lane.get("count", 0),
                }
                for lane in (interaction_handoff.get("lanes", []) or [])[:6]
                if isinstance(lane, dict)
            ],
        } if interaction_handoff else None,
        "resume_handoff": _pipeline_resume_handoff(run),
        "upstream_preflight": {
            "status": upstream_preflight.get("status"),
            "ready": upstream_preflight.get("ready", False),
            "service_status": upstream_preflight.get("service_status"),
            "status_code": upstream_preflight.get("status_code"),
            "message": upstream_preflight.get("message"),
            "next_step": upstream_preflight.get("next_step") or preflight_samples.get("next_step"),
            "deployment_hint": upstream_preflight.get("deploy_command") or preflight_samples.get("deployment_hint"),
            "payload_kind": upstream_preflight.get("payload_kind") or preflight_samples.get("payload_kind"),
            "submitted_filename": upstream_preflight.get("submitted_filename") or preflight_samples.get("submitted_filename"),
            "submitted_content_type": upstream_preflight.get("submitted_content_type") or preflight_samples.get("submitted_content_type"),
            "payload_bytes": preflight_samples.get("payload_bytes"),
            "payload_sha256": preflight_samples.get("payload_sha256"),
            "target_url": upstream_preflight.get("target_url") or preflight_samples.get("target_url"),
            "health_url": upstream_preflight.get("health_url") or preflight_samples.get("health_url"),
            "services_status_path": preflight_samples.get("services_status_path"),
            "diagnostics_path": preflight_samples.get("diagnostics_path"),
        } if upstream_preflight else None,
        "upstream_ingest": {
            "status": upstream_ingest.get("status"),
            "submitted": upstream_ingest.get("submitted", False),
            "status_code": upstream_ingest.get("status_code"),
            "payload_kind": upstream_samples.get("payload_kind"),
            "submitted_filename": upstream_samples.get("submitted_filename"),
            "target_url": upstream_ingest.get("target_url") or upstream_samples.get("target_url"),
            "selected_endpoint": upstream_ingest.get("selected_endpoint") or upstream_samples.get("selected_endpoint"),
            "attempted_urls": upstream_samples.get("attempted_urls", []),
            "endpoint_attempt_count": len(upstream_samples.get("endpoint_attempts", []) or []),
            "payload_bytes": upstream_samples.get("payload_bytes"),
        } if upstream_ingest else None,
        "operational_handoff": {
            "status": operational_handoff.get("status"),
            "label": operational_handoff.get("label"),
            "local_trace_ready": operational_handoff.get("local_trace_ready", False),
            "chat_handoff_ready": operational_handoff.get("chat_handoff_ready", False),
            "remote_ingest_ready": operational_handoff.get("remote_ingest_ready", False),
            "remote_ingest_submitted": operational_handoff.get("remote_ingest_submitted", False),
            "next_step": operational_handoff.get("next_step"),
            "primary_action": operational_handoff.get("primary_action"),
            "diagnostics_path": operational_handoff.get("diagnostics_path"),
            "services_status_path": operational_handoff.get("services_status_path"),
        } if operational_handoff else None,
        "deployment_outcome": {
            "status": deployment_outcome.get("status"),
            "label": deployment_outcome.get("label"),
            "severity": deployment_outcome.get("severity"),
            "ready": deployment_outcome.get("ready", False),
            "remote_blocked": deployment_outcome.get("remote_blocked", False),
            "remote_graph_available": deployment_outcome.get("remote_graph_available", False),
            "remote_submitted": deployment_outcome.get("remote_submitted", False),
            "fallback_used": deployment_outcome.get("fallback_used", False),
            "preflight": deployment_outcome.get("preflight"),
            "ingest": deployment_outcome.get("ingest"),
            "operator": deployment_outcome.get("operator"),
            "trace_fields": deployment_outcome.get("trace_fields", []),
        } if deployment_outcome else None,
        "pipeline_performance": {
            "stage_count": pipeline_performance.get("stage_count", len(stages)),
            "total_duration_ms": pipeline_performance.get("total_duration_ms"),
            "local_stage_duration_ms": pipeline_performance.get("local_stage_duration_ms"),
            "remote_stage_duration_ms": pipeline_performance.get("remote_stage_duration_ms"),
            "slowest_stage": pipeline_performance.get("slowest_stage"),
            "status_counts": pipeline_performance.get("status_counts", {}),
            "throughput": pipeline_performance.get("throughput", {}),
            "timings": pipeline_performance.get("timings", []),
        } if pipeline_performance else None,
        "stages": [
            {
                "name": stage.get("name"),
                "status": stage.get("status"),
                "duration_ms": stage.get("duration_ms", 0),
            }
            for stage in stages
            if isinstance(stage, dict)
        ],
    }


def _pipeline_operational_handoff(run: dict[str, Any]) -> dict[str, Any]:
    cfg = get_graphrag_config()
    diagnostics = _build_deployment_diagnostics()
    upstream_preflight = run.get("upstream_preflight", {}) if isinstance(run.get("upstream_preflight"), dict) else {}
    upstream_ingest = run.get("upstream_ingest", {}) if isinstance(run.get("upstream_ingest"), dict) else {}
    preflight_samples = upstream_preflight.get("sample_outputs", {}) if isinstance(upstream_preflight.get("sample_outputs"), dict) else {}
    upstream_samples = upstream_ingest.get("sample_outputs", {}) if isinstance(upstream_ingest.get("sample_outputs"), dict) else {}
    summary = run.get("summary", {}) if isinstance(run.get("summary"), dict) else {}
    local_graph = run.get("local_graph", {}) if isinstance(run.get("local_graph"), dict) else {}
    local_context = run.get("local_context", {}) if isinstance(run.get("local_context"), dict) else {}
    interaction_handoff = run.get("interaction_handoff", {}) if isinstance(run.get("interaction_handoff"), dict) else {}
    interaction_counts = interaction_handoff.get("counts", {}) if isinstance(interaction_handoff.get("counts"), dict) else {}

    local_trace_ready = bool(run.get("run_id") and summary.get("chunk_count", 0))
    chat_handoff_ready = bool(cfg.enabled and run.get("run_id") and (
        interaction_handoff.get("ready")
        or local_context.get("cached_chunk_count", 0)
        or local_graph.get("edge_count", 0)
        or summary.get("relationship_candidate_count", 0)
    ))
    remote_ingest_ready = bool(upstream_preflight.get("ready"))
    remote_ingest_submitted = bool(upstream_ingest.get("submitted"))
    service_status = upstream_preflight.get("service_status")
    preflight_status = str(upstream_preflight.get("status") or "")
    ingest_status = str(upstream_ingest.get("status") or "")
    payload_kind = (
        upstream_preflight.get("payload_kind")
        or preflight_samples.get("payload_kind")
        or upstream_samples.get("payload_kind")
    )
    target_url = (
        upstream_ingest.get("target_url")
        or upstream_preflight.get("target_url")
        or upstream_samples.get("target_url")
        or preflight_samples.get("target_url")
    )
    health_url = upstream_preflight.get("health_url") or preflight_samples.get("health_url")
    next_step = (
        upstream_ingest.get("next_step")
        or upstream_preflight.get("next_step")
        or preflight_samples.get("next_step")
    )

    actions: list[dict[str, Any]] = []
    if local_trace_ready:
        actions.append({
            "id": "inspect_trace",
            "label": "Inspect trace",
            "ready": True,
            "target": "pipeline_trace",
        })
    if chat_handoff_ready:
        actions.append({
            "id": "ask_chat",
            "label": "Ask Chat",
            "ready": True,
            "target": "chat",
        })
    if interaction_counts.get("graph_anchors"):
        actions.append({
            "id": "inspect_graph_anchors",
            "label": "Inspect graph anchors",
            "ready": True,
            "target": "pipeline_trace",
        })

    if remote_ingest_submitted:
        status = "remote_graph_submitted"
        label = "Remote Graph Submitted"
        message = "The local trace is ready and the structured payload was submitted to the configured GraphRAG API."
        primary_action = "ask_chat" if chat_handoff_ready else "inspect_trace"
        next_step = next_step or "Use Ask Chat or graph search against the newly submitted document."
    elif ingest_status == "timed_out":
        status = "local_trace_ready"
        label = "Local Trace Ready"
        message = "The local trace, graph candidates, and chat context are ready; remote GraphRAG ingest timed out."
        primary_action = "ask_chat" if chat_handoff_ready else "inspect_trace"
        next_step = next_step or "Retry remote ingest or increase GRAPHRAG_REMOTE_INGEST_TIMEOUT if the configured GraphRAG API needs longer."
        actions.append({
            "id": "retry_upload",
            "label": "Retry remote ingest",
            "ready": True,
            "target": "pipeline_upload",
        })
    elif not cfg.enabled:
        status = "graphrag_disabled"
        label = "Local Trace Built"
        message = "The document trace was built locally, but GraphRAG chat/search routes are disabled."
        primary_action = "enable_graphrag"
        next_step = next_step or "Set GRAPHRAG_ENABLED=true and restart the backend."
        actions.append({
            "id": "enable_graphrag",
            "label": "Enable GraphRAG",
            "ready": False,
            "target": "configuration",
            "command": "GRAPHRAG_ENABLED=true",
        })
    elif not cfg.graphrag_url:
        status = "local_trace_ready"
        label = "Local Trace Ready"
        message = "The local pipeline trace, graph candidates, and run-scoped chat context are ready; remote graph ingest is waiting for GRAPHRAG_URL."
        primary_action = "ask_chat" if chat_handoff_ready else "configure_remote_graph"
        next_step = next_step or "Set GRAPHRAG_URL or GRAPHRAG_SERVICE_URL to the GraphRAG API base URL."
        actions.append({
            "id": "configure_remote_graph",
            "label": "Configure graph API",
            "ready": False,
            "target": "deployment",
            "required_env": ["GRAPHRAG_URL", "GRAPHRAG_SERVICE_URL"],
        })
    elif remote_ingest_ready:
        status = "remote_graph_ready"
        label = "Remote Graph Ready"
        message = "The GraphRAG health probe passed, but remote ingest did not report submission."
        primary_action = "retry_upload"
        next_step = next_step or "Retry the pipeline upload and inspect the upstream ingest response."
        actions.append({
            "id": "retry_upload",
            "label": "Retry upload",
            "ready": True,
            "target": "pipeline_upload",
        })
    else:
        status = "local_trace_ready"
        label = "Local Trace Ready"
        message = "The local pipeline trace and run-scoped chat context are ready; remote graph ingest is blocked by the GraphRAG service health check."
        primary_action = "service_smoke_check"
        next_step = next_step or "Open the Service Smoke Check or inspect GraphRAG service logs before retrying remote ingest."
        actions.extend([
            {
                "id": "service_smoke_check",
                "label": "Service smoke check",
                "ready": False,
                "target": GRAPHRAG_SERVICES_STATUS_PATH,
            },
            {
                "id": "retry_upload",
                "label": "Retry upload",
                "ready": False,
                "target": "pipeline_upload",
            },
        ])

    if remote_ingest_submitted and target_url:
        actions.append({
            "id": "remote_ingest",
            "label": "Remote ingest",
            "ready": True,
            "target": target_url,
        })

    return {
        "status": status,
        "label": label,
        "message": message,
        "local_trace_ready": local_trace_ready,
        "chat_handoff_ready": chat_handoff_ready,
        "remote_ingest_ready": remote_ingest_ready,
        "remote_ingest_submitted": remote_ingest_submitted,
        "remote_ingest_status": ingest_status or None,
        "interaction_ready": interaction_handoff.get("ready", False),
        "interaction_mode": interaction_handoff.get("recommended_mode"),
        "interaction_graph_anchor_count": interaction_counts.get("graph_anchors", 0),
        "interaction_starter_prompt_count": interaction_counts.get("starter_prompts", 0),
        "interaction_capability_card_count": interaction_counts.get("capability_cards", 0),
        "interaction_citation_scopes": interaction_handoff.get("citation_scopes", []),
        "answer_guidance": interaction_handoff.get("answer_guidance"),
        "preflight_status": preflight_status or None,
        "service_status": service_status,
        "payload_kind": payload_kind,
        "target_url": target_url,
        "health_url": health_url,
        "next_step": next_step,
        "primary_action": primary_action,
        "actions": actions[:6],
        "diagnostics_status": diagnostics.get("status"),
        "diagnostics_summary": diagnostics.get("summary"),
        "diagnostics_warnings": diagnostics.get("warnings", [])[:4],
        "diagnostics_path": GRAPHRAG_DEPLOYMENT_DIAGNOSTICS_PATH,
        "services_status_path": GRAPHRAG_SERVICES_STATUS_PATH,
        "deployment_hint": GRAPHRAG_DEPLOYMENT_HINT,
    }


def _pipeline_performance_summary(run: dict[str, Any]) -> dict[str, Any]:
    stages = [
        stage
        for stage in run.get("stages", []) or []
        if isinstance(stage, dict)
    ]
    timings = [
        {
            "name": str(stage.get("name") or ""),
            "status": str(stage.get("status") or ""),
            "duration_ms": int(_number_value(stage.get("duration_ms"), 0)),
        }
        for stage in stages
    ]
    total_duration_ms = sum(item["duration_ms"] for item in timings)
    slowest = max(timings, key=lambda item: item["duration_ms"], default=None)
    status_counts = Counter(item["status"] or "unknown" for item in timings)
    document = run.get("document", {}) if isinstance(run.get("document"), dict) else {}
    summary = run.get("summary", {}) if isinstance(run.get("summary"), dict) else {}
    size_bytes = int(_number_value(document.get("size_bytes"), 0))
    element_count = int(_number_value(summary.get("element_count"), 0))
    chunk_count = int(_number_value(summary.get("chunk_count"), 0))
    relationship_count = int(_number_value(summary.get("relationship_candidate_count"), 0))

    def per_second(count: int | float) -> float:
        if count <= 0:
            return 0.0
        if total_duration_ms <= 0:
            return float(count)
        return round(float(count) / (total_duration_ms / 1000), 3)

    return {
        "stage_count": len(timings),
        "total_duration_ms": total_duration_ms,
        "local_stage_duration_ms": sum(
            item["duration_ms"]
            for item in timings
            if item["name"] not in {"upstream_preflight", "upstream_ingest"}
        ),
        "remote_stage_duration_ms": sum(
            item["duration_ms"]
            for item in timings
            if item["name"] in {"upstream_preflight", "upstream_ingest"}
        ),
        "slowest_stage": slowest,
        "status_counts": dict(status_counts),
        "throughput": {
            "bytes_per_second": per_second(size_bytes),
            "elements_per_second": per_second(element_count),
            "chunks_per_second": per_second(chunk_count),
            "relationships_per_second": per_second(relationship_count),
        },
        "timings": timings,
    }


def _pipeline_deployment_outcome(run: dict[str, Any]) -> dict[str, Any]:
    cfg = get_graphrag_config()
    upstream_preflight = run.get("upstream_preflight", {}) if isinstance(run.get("upstream_preflight"), dict) else {}
    upstream_ingest = run.get("upstream_ingest", {}) if isinstance(run.get("upstream_ingest"), dict) else {}
    operational_handoff = run.get("operational_handoff", {}) if isinstance(run.get("operational_handoff"), dict) else {}
    preflight_samples = upstream_preflight.get("sample_outputs", {}) if isinstance(upstream_preflight.get("sample_outputs"), dict) else {}
    upstream_samples = upstream_ingest.get("sample_outputs", {}) if isinstance(upstream_ingest.get("sample_outputs"), dict) else {}

    endpoint_attempts = [
        attempt
        for attempt in upstream_samples.get("endpoint_attempts", []) or []
        if isinstance(attempt, dict)
    ]
    attempted_urls = [
        str(url)
        for url in upstream_samples.get("attempted_urls", []) or []
        if url
    ]
    if not attempted_urls:
        attempted_urls = [str(attempt.get("url")) for attempt in endpoint_attempts if attempt.get("url")]

    health_attempts = [
        attempt
        for attempt in preflight_samples.get("health_attempts", []) or []
        if isinstance(attempt, dict)
    ]
    attempted_health_urls = [
        str(url)
        for url in preflight_samples.get("attempted_health_urls", []) or []
        if url
    ]
    if not attempted_health_urls:
        attempted_health_urls = [str(attempt.get("url")) for attempt in health_attempts if attempt.get("url")]

    compatible_ingest_urls = [
        str(url)
        for url in (
            upstream_samples.get("compatible_ingest_urls")
            or preflight_samples.get("compatible_ingest_urls")
            or []
        )
        if url
    ]
    compatible_health_urls = [
        str(url)
        for url in preflight_samples.get("compatible_health_urls", []) or []
        if url
    ]
    selected_endpoint = upstream_ingest.get("selected_endpoint") or upstream_samples.get("selected_endpoint")
    target_url = (
        upstream_ingest.get("target_url")
        or upstream_preflight.get("target_url")
        or upstream_samples.get("target_url")
        or preflight_samples.get("target_url")
    )
    health_url = upstream_preflight.get("health_url") or preflight_samples.get("health_url")
    submitted = bool(upstream_ingest.get("submitted"))
    preflight_ready = bool(upstream_preflight.get("ready"))
    ingest_failed = str(upstream_ingest.get("status") or "") == "failed"
    ingest_timed_out = str(upstream_ingest.get("status") or "") == "timed_out"
    service_status = upstream_preflight.get("service_status")
    failed_attempt_count = sum(
        1
        for attempt in endpoint_attempts
        if attempt.get("status") != "complete" or int(_number_value(attempt.get("status_code"), 0)) >= 400
    )
    fallback_used = bool(
        len(endpoint_attempts) > 1
        or (selected_endpoint and compatible_ingest_urls and str(selected_endpoint) != compatible_ingest_urls[0])
    )

    if submitted:
        status = "remote_submitted"
        label = "Remote Graph Submitted"
        severity = "success"
        remote_blocked = False
    elif ingest_timed_out:
        status = "remote_ingest_timed_out"
        label = "Remote Ingest Timed Out"
        severity = "warning"
        remote_blocked = False
    elif ingest_failed:
        status = "remote_ingest_failed"
        label = "Remote Ingest Failed"
        severity = "error"
        remote_blocked = True
    elif not cfg.enabled:
        status = "disabled"
        label = "GraphRAG Disabled"
        severity = "info"
        remote_blocked = True
    elif not cfg.graphrag_url:
        status = "not_configured"
        label = "Remote Graph Not Configured"
        severity = "info"
        remote_blocked = True
    elif preflight_ready:
        status = "remote_ready"
        label = "Remote Graph Ready"
        severity = "warning"
        remote_blocked = False
    elif service_status == "unavailable":
        status = "health_unavailable"
        label = "Remote Health Unavailable"
        severity = "error"
        remote_blocked = True
    elif service_status == "unhealthy":
        status = "health_unhealthy"
        label = "Remote Health Unhealthy"
        severity = "error"
        remote_blocked = True
    elif upstream_preflight:
        status = "preflight_blocked"
        label = "Remote Preflight Blocked"
        severity = "warning"
        remote_blocked = True
    else:
        status = "local_preview"
        label = "Local Preview"
        severity = "info"
        remote_blocked = True

    next_step = (
        operational_handoff.get("next_step")
        or upstream_ingest.get("next_step")
        or upstream_preflight.get("next_step")
        or upstream_samples.get("next_step")
        or preflight_samples.get("next_step")
    )
    primary_action = operational_handoff.get("primary_action")
    if not primary_action:
        primary_action = "ask_chat" if operational_handoff.get("chat_handoff_ready") else "service_smoke_check" if remote_blocked else "retry_upload"

    return {
        "status": status,
        "label": label,
        "severity": severity,
        "ready": submitted,
        "remote_blocked": remote_blocked,
        "remote_graph_available": preflight_ready,
        "remote_submitted": submitted,
        "fallback_used": fallback_used,
        "preflight": {
            "status": upstream_preflight.get("status"),
            "ready": preflight_ready,
            "service_status": service_status,
            "status_code": upstream_preflight.get("status_code") or preflight_samples.get("health_status_code"),
            "health_url": health_url,
            "compatible_health_urls": compatible_health_urls,
            "attempted_health_urls": attempted_health_urls,
            "health_attempt_count": len(health_attempts) or len(attempted_health_urls),
        },
        "ingest": {
            "status": upstream_ingest.get("status"),
            "submitted": submitted,
            "status_code": upstream_ingest.get("status_code"),
            "target_url": target_url,
            "selected_endpoint": selected_endpoint,
            "compatible_ingest_urls": compatible_ingest_urls,
            "attempted_urls": attempted_urls,
            "endpoint_attempt_count": len(endpoint_attempts),
            "failed_attempt_count": failed_attempt_count,
            "payload_kind": upstream_samples.get("payload_kind") or upstream_preflight.get("payload_kind") or preflight_samples.get("payload_kind"),
            "payload_bytes": upstream_samples.get("payload_bytes") or preflight_samples.get("payload_bytes"),
        },
        "operator": {
            "primary_action": primary_action,
            "next_step": next_step,
            "services_status_path": GRAPHRAG_SERVICES_STATUS_PATH,
            "diagnostics_path": GRAPHRAG_DEPLOYMENT_DIAGNOSTICS_PATH,
            "deployment_hint": GRAPHRAG_DEPLOYMENT_HINT,
        },
        "trace_fields": [
            "upstream_preflight.sample_outputs.health_attempts",
            "upstream_preflight.sample_outputs.compatible_health_urls",
            "upstream_ingest.sample_outputs.endpoint_attempts",
            "upstream_ingest.sample_outputs.attempted_urls",
            "upstream_ingest.selected_endpoint",
        ],
    }


def _attach_pipeline_run_rollups(run: dict[str, Any]) -> dict[str, Any]:
    performance = _pipeline_performance_summary(run)
    run["pipeline_performance"] = performance
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    if isinstance(summary, dict):
        summary["pipeline_performance"] = {
            "total_duration_ms": performance["total_duration_ms"],
            "stage_count": performance["stage_count"],
            "slowest_stage": performance["slowest_stage"],
            "throughput": performance["throughput"],
        }
        run["summary"] = summary
    run["deployment_outcome"] = _pipeline_deployment_outcome(run)
    return run


def _number_value(value: Any, default: float = 0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _increment_count(counter: dict[str, int], key: Any) -> None:
    label = str(key or "unknown")
    counter[label] = counter.get(label, 0) + 1


def _evidence_profile_from_run(run: dict[str, Any]) -> dict[str, Any]:
    summary = run.get("summary", {}) if isinstance(run.get("summary"), dict) else {}
    profile = run.get("document_profile", {}) if isinstance(run.get("document_profile"), dict) else {}
    evidence_profile = profile.get("evidence_profile") or summary.get("evidence_profile")
    return evidence_profile if isinstance(evidence_profile, dict) else {}


_BATCH_QUICK_EVIDENCE_TOTALS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("archive_entries", ("archive_entries",)),
    ("attachments", ("attachments",)),
    ("ocr_blocks", ("ocr_blocks",)),
    ("layout_elements", ("layout_elements",)),
    ("distinct_tables", ("tables", "distinct_tables")),
    ("table_elements", ("table_elements",)),
    ("charts", ("charts",)),
    ("diagrams", ("diagrams",)),
    ("shapes", ("shapes",)),
    ("connectors", ("connectors",)),
    ("forms", ("forms",)),
    ("links", ("links",)),
    ("code_blocks", ("code_blocks",)),
    ("comments", ("comments",)),
    ("revisions", ("revisions",)),
    ("speaker_notes", ("speaker_notes",)),
    ("media", ("media",)),
    ("message_body_parts", ("message_body_parts",)),
    ("styled_text_blocks", ("styled_text_blocks",)),
    ("images", ("images",)),
    ("pivot_tables", ("pivot_tables",)),
    ("hidden_slides", ("hidden_slides",)),
    ("hidden_sheets", ("hidden_sheets",)),
    ("legacy_office_streams", ("legacy_office_streams",)),
    ("legacy_office_text_blocks", ("legacy_office_text_blocks",)),
)

_BATCH_GRAPH_EVIDENCE_TOTALS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("graph_relationships", ("relationship_count",)),
    ("semantic_relationships", ("semantic_relationship_count",)),
    ("structural_relationships", ("structural_relationship_count",)),
    ("table_facts", ("table_fact_count",)),
    ("chart_relationships", ("chart_relationship_count",)),
    ("layout_relationships", ("layout_relationship_count",)),
    ("link_relationships", ("link_relationship_count",)),
    ("code_block_relationships", ("code_block_relationship_count",)),
    ("diagram_relationships", ("diagram_relationship_count",)),
    ("styled_text_relationships", ("styled_text_relationship_count",)),
    ("attachment_relationships", ("attachment_relationship_count",)),
    ("archive_relationships", ("archive_relationship_count",)),
    ("ocr_relationships", ("ocr_relationship_count",)),
    ("speaker_notes_relationships", ("speaker_notes_relationship_count",)),
)


def _batch_count_value(source: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        if key in source:
            return int(_number_value(source.get(key)))
    return 0


def _empty_batch_evidence_totals() -> dict[str, int]:
    return {
        target_key: 0
        for target_key, _ in (
            *_BATCH_QUICK_EVIDENCE_TOTALS,
            *_BATCH_GRAPH_EVIDENCE_TOTALS,
        )
    }


def _batch_evidence_totals_from_profile(quick_counts: dict[str, Any], graph_evidence: dict[str, Any]) -> dict[str, int]:
    totals = _empty_batch_evidence_totals()
    for target_key, source_keys in _BATCH_QUICK_EVIDENCE_TOTALS:
        totals[target_key] = _batch_count_value(quick_counts, source_keys)
    for target_key, source_keys in _BATCH_GRAPH_EVIDENCE_TOTALS:
        totals[target_key] = _batch_count_value(graph_evidence, source_keys)
    return totals


def _positive_counts(counts: dict[str, Any]) -> dict[str, int]:
    positives: dict[str, int] = {}
    for key, value in counts.items():
        count = int(_number_value(value))
        if count:
            positives[key] = count
    return positives


def _add_batch_evidence_totals(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(_number_value(value))


def _batch_graph_relation_type(relation: str) -> str:
    normalized = str(relation or "").lower()
    structural_prefixes = (
        "contains",
        "has",
        "part_of",
        "source_",
        "archive_",
        "slide_",
        "page_",
        "sheet_",
    )
    return "layout" if normalized.startswith(structural_prefixes) else "semantic"


def _batch_question_id(*parts: str) -> str:
    text = "_".join(str(part or "").lower() for part in parts if part)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:96] or "batch_graph_question"


def _batch_compact_text(value: Any, limit: int = 56) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return f"{text[:max(1, limit - 3)].strip()}..."


def _batch_file_text(filenames: list[str]) -> str:
    visible = [filename for filename in filenames if filename]
    if not visible:
        return "the uploaded files"
    overflow = f", and {len(visible) - 6} more" if len(visible) > 6 else ""
    return f"{', '.join(visible[:6])}{overflow}"


def _pipeline_batch_graph_questions(
    overview: dict[str, Any],
    *,
    filenames: list[str],
    limit: int = 5,
) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    file_text = _batch_file_text(filenames)

    def push(question: dict[str, Any]) -> None:
        if len(questions) >= limit:
            return
        prompt = str(question.get("prompt") or "").lower()
        label = str(question.get("label") or "").lower()
        if not prompt or not label:
            return
        if any(str(existing.get("prompt") or "").lower() == prompt or str(existing.get("label") or "").lower() == label for existing in questions):
            return
        questions.append(question)

    graph_bits = []
    if overview.get("relationship_count"):
        graph_bits.append(f"{overview['relationship_count']} graph relationships")
    if overview.get("bridge_entity_count"):
        graph_bits.append(f"{overview['bridge_entity_count']} bridge entities")
    if overview.get("cross_document_path_count"):
        graph_bits.append(f"{overview['cross_document_path_count']} cross-document paths")
    graph_sentence = f"The batch graph overview found {', '.join(graph_bits)}. " if graph_bits else ""
    push({
        "id": "batch_summary",
        "kind": "summary",
        "label": "Compare batch graph",
        "prompt": (
            f"Use the GraphRAG pipeline traces for this batch ({file_text}) to compare the documents. "
            f"{graph_sentence}"
            "Explain shared entities, cross-document relationships, graph paths, bridge entities, contradictions or gaps, "
            "and unique evidence from each file. Cite filename plus slide/page/table/chart/connector provenance whenever available."
        ),
    })

    paths = [path for path in overview.get("cross_document_paths", []) if isinstance(path, dict)]
    if paths:
        path = paths[0]
        source = str(path.get("source") or "source entity")
        bridge = str(path.get("bridge") or "bridge entity")
        target = str(path.get("target") or "target entity")
        path_files = _batch_file_text([str(filename) for filename in path.get("filenames", []) if filename])
        push({
            "id": _batch_question_id("batch_path", source, bridge, target),
            "kind": "path",
            "label": _batch_compact_text(f"Path: {source} to {target}"),
            "source": source,
            "bridge": bridge,
            "target": target,
            "prompt": (
                f"Use the GraphRAG pipeline traces for this batch ({file_text}) to explain the cross-document graph path "
                f"\"{source}\" -> \"{bridge}\" -> \"{target}\". "
                f"Compare evidence from {path_files}, explain why \"{bridge}\" is the bridge entity, "
                "and cite filename plus slide/page/table/chart/connector provenance for each hop."
            ),
        })

    bridges = [bridge for bridge in overview.get("bridge_entities", []) if isinstance(bridge, dict)]
    if bridges:
        bridge = bridges[0]
        name = str(bridge.get("name") or "bridge entity")
        bridge_files = _batch_file_text([str(filename) for filename in bridge.get("filenames", []) if filename])
        push({
            "id": _batch_question_id("batch_bridge", name),
            "kind": "bridge",
            "label": _batch_compact_text(f"Bridge: {name}"),
            "bridge": name,
            "prompt": (
                f"Use the GraphRAG pipeline traces for this batch ({file_text}) to analyze \"{name}\" as a bridge entity across documents. "
                f"Compare how it appears in {bridge_files}, list the strongest connected relationships, "
                "identify any contradictions or gaps, and cite provenance for each file."
            ),
        })

    shared_entities = [entity for entity in overview.get("shared_entities", []) if isinstance(entity, dict)]
    if shared_entities:
        entity = shared_entities[0]
        name = str(entity.get("name") or "shared entity")
        push({
            "id": _batch_question_id("batch_shared", name),
            "kind": "shared_entity",
            "label": _batch_compact_text(f"Shared: {name}"),
            "source": name,
            "prompt": (
                f"Use the GraphRAG pipeline traces for this batch ({file_text}) to compare the shared entity \"{name}\" across documents. "
                "Explain which relationships and evidence each file contributes, where the entity meaning changes, and cite source provenance."
            ),
        })

    relationships = [relationship for relationship in overview.get("top_relationships", []) if isinstance(relationship, dict)]
    if relationships:
        relationship = relationships[0]
        source = str(relationship.get("source") or "source")
        relation = str(relationship.get("relation") or "related_to")
        target = str(relationship.get("target") or "target")
        push({
            "id": _batch_question_id("batch_relationship", source, relation, target),
            "kind": "relationship",
            "label": _batch_compact_text(f"{source} to {target}"),
            "source": source,
            "relation": relation,
            "target": target,
            "prompt": (
                f"Use the GraphRAG pipeline traces for this batch ({file_text}) to validate and explain the relationship "
                f"\"{source}\" --[{relation}]--> \"{target}\". "
                "Find supporting or conflicting evidence across files and cite filename plus slide/page/table/chart/connector provenance."
            ),
        })

    return questions[:limit]


def _pipeline_batch_graph_overview(results: list[dict[str, Any]], *, limit: int = 12) -> dict[str, Any]:
    relation_counts: Counter[str] = Counter()
    entity_runs: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "name": "",
        "run_ids": set(),
        "filenames": set(),
        "relationships": set(),
        "incident_edges": [],
    })
    top_relationship_candidates: list[dict[str, Any]] = []
    source_run_ids: set[str] = set()
    source_filenames: set[str] = set()
    node_count = 0
    edge_count = 0

    def add_entity(name: Any, *, run_id: str, filename: str, relation: str | None = None) -> dict[str, Any] | None:
        text = str(name or "").strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered.startswith(("slide ", "page ", "sheet ")):
            return None
        record = entity_runs[lowered]
        record["name"] = text
        if run_id:
            record["run_ids"].add(run_id)
        if filename:
            record["filenames"].add(filename)
        if relation:
            record["relationships"].add(relation)
        return record

    for result in results:
        run = result.get("run") if isinstance(result.get("run"), dict) else {}
        if not run:
            continue
        run_id = str(run.get("run_id") or "")
        document = run.get("document", {}) if isinstance(run.get("document"), dict) else {}
        filename = str(result.get("filename") or document.get("filename") or "")
        graph = run.get("local_graph") if isinstance(run.get("local_graph"), dict) else {}
        nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
        edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
        if nodes or edges:
            if run_id:
                source_run_ids.add(run_id)
            if filename:
                source_filenames.add(filename)
        node_count += len(nodes)
        edge_count += len(edges)

        for node in nodes:
            if not isinstance(node, dict):
                continue
            add_entity(node.get("name") or node.get("label") or node.get("id"), run_id=run_id, filename=filename)

        for edge in edges:
            if not isinstance(edge, dict):
                continue
            relation = str(edge.get("label") or edge.get("relation") or edge.get("type") or "related_to")
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            relation_counts[relation] += 1
            add_entity(source, run_id=run_id, filename=filename, relation=relation)
            add_entity(target, run_id=run_id, filename=filename, relation=relation)
            weight = _number_value(edge.get("weight"), 1.0)
            relationship_type = _batch_graph_relation_type(relation)
            relationship_record = {
                "source": source,
                "relation": relation,
                "target": target,
                "relationship_type": relationship_type,
                "weight": round(weight, 3),
                "score": round(
                    min(1.0, (weight * 0.55) + (0.25 if relationship_type == "semantic" else 0.12)),
                    3,
                ),
                "evidence": edge.get("evidence"),
                "filename": filename,
                "run_id": run_id,
                "metadata": edge.get("metadata") if isinstance(edge.get("metadata"), dict) else {},
            }
            top_relationship_candidates.append(relationship_record)
            for endpoint in (source, target):
                record = add_entity(endpoint, run_id=run_id, filename=filename, relation=relation)
                if record is not None:
                    record["incident_edges"].append(relationship_record)

    shared_entities = [
        {
            "name": record["name"],
            "run_count": len(record["run_ids"]),
            "filenames": sorted(record["filenames"])[:6],
            "relationships": sorted(record["relationships"])[:8],
        }
        for record in entity_runs.values()
        if len(record["run_ids"]) >= 2 or len(record["filenames"]) >= 2
    ]
    top_relationships = sorted(
        top_relationship_candidates,
        key=lambda relationship: (
            relationship.get("score", 0),
            relationship.get("relationship_type") == "semantic",
            relationship.get("weight", 0),
        ),
        reverse=True,
    )[:limit]
    bridge_entities = []
    cross_document_paths = []
    for record in entity_runs.values():
        incident_edges = [
            edge for edge in record.get("incident_edges", [])
            if isinstance(edge, dict)
        ]
        if len(record["run_ids"]) < 2 and len(record["filenames"]) < 2:
            continue
        if len(incident_edges) < 2:
            continue
        bridge_name = record["name"]
        bridge_entities.append({
            "name": bridge_name,
            "run_count": len(record["run_ids"]),
            "filenames": sorted(record["filenames"])[:6],
            "relationship_count": len(incident_edges),
            "relationships": sorted(record["relationships"])[:8],
            "score": round(
                min(1.0, 0.45 + (min(len(record["run_ids"]), 4) * 0.08) + (min(len(incident_edges), 6) * 0.035)),
                3,
            ),
        })
        sorted_edges = sorted(
            incident_edges,
            key=lambda edge: (edge.get("score", 0), edge.get("relationship_type") == "semantic", edge.get("weight", 0)),
            reverse=True,
        )
        for left_index, left in enumerate(sorted_edges[:6]):
            for right in sorted_edges[left_index + 1:6]:
                if left.get("run_id") == right.get("run_id") and left.get("filename") == right.get("filename"):
                    continue
                left_other = left.get("target") if left.get("source") == bridge_name else left.get("source")
                right_other = right.get("target") if right.get("source") == bridge_name else right.get("source")
                if not left_other or not right_other or left_other == right_other:
                    continue
                score = round(min(1.0, ((left.get("score", 0) + right.get("score", 0)) / 2) + 0.08), 3)
                cross_document_paths.append({
                    "bridge": bridge_name,
                    "source": left_other,
                    "target": right_other,
                    "nodes": [left_other, bridge_name, right_other],
                    "hops": 2,
                    "score": score,
                    "filenames": sorted({str(left.get("filename") or ""), str(right.get("filename") or "")} - {""}),
                    "run_ids": sorted({str(left.get("run_id") or ""), str(right.get("run_id") or "")} - {""}),
                    "relationships": [left, right],
                    "text": (
                        f"{left_other} --[{left.get('relation')}]-- {bridge_name} "
                        f"--[{right.get('relation')}]-- {right_other}"
                    ),
                })
                break
            if len(cross_document_paths) >= limit * 2:
                break

    overview = {
        "source_count": len(source_run_ids or source_filenames),
        "node_count": node_count,
        "relationship_count": edge_count,
        "relationship_type_counts": dict(relation_counts.most_common(12)),
        "shared_entity_count": len(shared_entities),
        "shared_entities": sorted(
            shared_entities,
            key=lambda item: (item["run_count"], len(item["relationships"]), item["name"]),
            reverse=True,
        )[:limit],
        "top_relationships": top_relationships,
        "bridge_entity_count": len(bridge_entities),
        "bridge_entities": sorted(
            bridge_entities,
            key=lambda item: (item.get("score", 0), item["run_count"], item["relationship_count"]),
            reverse=True,
        )[:limit],
        "cross_document_path_count": len(cross_document_paths),
        "cross_document_paths": sorted(
            cross_document_paths,
            key=lambda item: item.get("score", 0),
            reverse=True,
        )[:limit],
    }
    suggested_questions = _pipeline_batch_graph_questions(
        overview,
        filenames=sorted(source_filenames),
        limit=5,
    )
    overview["suggested_questions"] = suggested_questions
    overview["question_count"] = len(suggested_questions)
    return overview


def _pipeline_batch_graph_context_hints(overview: dict[str, Any]) -> dict[str, str]:
    paths = [path for path in overview.get("cross_document_paths", []) if isinstance(path, dict)]
    if paths:
        path = paths[0]
        hints = {
            "source": str(path.get("source") or "").strip(),
            "bridge": str(path.get("bridge") or "").strip(),
            "target": str(path.get("target") or "").strip(),
        }
        relationships = [relationship for relationship in path.get("relationships", []) if isinstance(relationship, dict)]
        if relationships:
            relation = str(relationships[0].get("relation") or relationships[0].get("label") or "").strip()
            if relation:
                hints["relation"] = relation
        return {key: value for key, value in hints.items() if value}

    relationships = [relationship for relationship in overview.get("top_relationships", []) if isinstance(relationship, dict)]
    if relationships:
        relationship = relationships[0]
        return {
            key: value
            for key, value in {
                "source": str(relationship.get("source") or "").strip(),
                "relation": str(relationship.get("relation") or "").strip(),
                "target": str(relationship.get("target") or "").strip(),
            }.items()
            if value
        }

    bridges = [bridge for bridge in overview.get("bridge_entities", []) if isinstance(bridge, dict)]
    if bridges:
        name = str(bridges[0].get("name") or "").strip()
        return {"bridge": name} if name else {}

    shared_entities = [entity for entity in overview.get("shared_entities", []) if isinstance(entity, dict)]
    if shared_entities:
        name = str(shared_entities[0].get("name") or "").strip()
        return {"source": name} if name else {}

    return {}


def _pipeline_batch_resume_handoff(
    results: list[dict[str, Any]],
    *,
    graph_overview: dict[str, Any],
) -> dict[str, Any] | None:
    run_ids: list[str] = []
    filenames: list[str] = []
    domains: set[str] = set()
    citation_scopes: list[str] = []
    graph_anchor_count = 0
    retrieval_rehearsal_count = 0
    ready_files = 0
    evidence_totals = _empty_batch_evidence_totals()

    for result in results:
        if result.get("success") is False:
            continue
        run = result.get("run") if isinstance(result.get("run"), dict) else {}
        run_id = str(run.get("run_id") or "").strip()
        if not run_id or run_id in run_ids:
            continue
        document = run.get("document", {}) if isinstance(run.get("document"), dict) else {}
        summary = run.get("summary", {}) if isinstance(run.get("summary"), dict) else {}
        local_graph = run.get("local_graph", {}) if isinstance(run.get("local_graph"), dict) else {}
        local_context = run.get("local_context", {}) if isinstance(run.get("local_context"), dict) else {}
        operational_handoff = run.get("operational_handoff", {}) if isinstance(run.get("operational_handoff"), dict) else {}
        interaction_handoff = run.get("interaction_handoff", {}) if isinstance(run.get("interaction_handoff"), dict) else {}
        interaction_counts = interaction_handoff.get("counts", {}) if isinstance(interaction_handoff.get("counts"), dict) else {}
        edges = local_graph.get("edges") if isinstance(local_graph.get("edges"), list) else []
        chat_ready = bool(
            interaction_handoff.get("ready")
            or operational_handoff.get("chat_handoff_ready")
            or summary.get("chunk_count")
            or summary.get("relationship_candidate_count")
            or summary.get("local_graph_edge_count")
            or summary.get("local_context_chunk_count")
            or local_graph.get("edge_count")
            or edges
            or local_context.get("cached_chunk_count")
        )
        if not chat_ready:
            continue

        run_ids.append(run_id)
        ready_files += 1
        filename = str(result.get("filename") or document.get("filename") or "").strip()
        if filename:
            filenames.append(filename)
        domain = str(document.get("domain") or "").strip()
        if domain:
            domains.add(domain)
        graph_anchor_count += int(_number_value(interaction_counts.get("graph_anchors")))
        retrieval_rehearsal_count += int(_number_value(interaction_counts.get("retrieval_rehearsals")))
        evidence_profile = _evidence_profile_from_run(run)
        quick_counts = evidence_profile.get("quick_counts", {}) if isinstance(evidence_profile.get("quick_counts"), dict) else {}
        graph_evidence = evidence_profile.get("graph_evidence", {}) if isinstance(evidence_profile.get("graph_evidence"), dict) else {}
        _add_batch_evidence_totals(
            evidence_totals,
            _batch_evidence_totals_from_profile(quick_counts, graph_evidence),
        )
        for scope in interaction_handoff.get("citation_scopes", []) or []:
            scope_text = str(scope or "").strip()
            if scope_text and scope_text not in citation_scopes:
                citation_scopes.append(scope_text)

    if not run_ids:
        return None

    sorted_domains = sorted(domains)
    domain = sorted_domains[0] if len(sorted_domains) == 1 else "general"
    suggested_questions = [
        _compact_prompt_card(question)
        for question in (graph_overview.get("suggested_questions", []) or [])
        if isinstance(question, dict) and question.get("prompt")
    ][:5]
    prompt = next(
        (str(question.get("prompt")) for question in suggested_questions if question.get("prompt")),
        (
            f"Use the GraphRAG pipeline traces for this batch ({_batch_file_text(filenames)}) to compare shared entities, "
            "relationships, graph paths, bridge entities, contradictions or gaps, and source provenance across files."
        ),
    )
    context_hints = _pipeline_batch_graph_context_hints(graph_overview)
    evidence_hints = _positive_counts(evidence_totals)
    graph_overview_counts = {
        "relationships": graph_overview.get("relationship_count", 0),
        "shared_entities": graph_overview.get("shared_entity_count", 0),
        "bridge_entities": graph_overview.get("bridge_entity_count", 0),
        "cross_document_paths": graph_overview.get("cross_document_path_count", 0),
    }
    chat_context_request: dict[str, Any] = {
        "query": prompt,
        "domain": domain,
        "pipeline_run_ids": run_ids,
        "include_graph": True,
        "use_local_pipeline_context": True,
        "include_local_pipeline": True,
        "rag_enabled": True,
        "rag_search_mode": "domain" if len(sorted_domains) <= 1 else "global",
        "top_k": min(16, max(8, len(run_ids) * 4)),
        "graph_overview_counts": graph_overview_counts,
        **context_hints,
    }
    if evidence_hints:
        chat_context_request["pipeline_evidence_totals"] = evidence_hints
        chat_context_request["pipeline_evidence_families"] = sorted(evidence_hints)
    if len(sorted_domains) == 1:
        chat_context_request["rag_domains"] = sorted_domains

    filename_label = f"{len(run_ids)} file GraphRAG batch"
    if filenames:
        filename_label = f"{len(run_ids)} file GraphRAG batch: {_batch_file_text(filenames)}"
    route_handoff = {
        "runIds": run_ids,
        "filename": filename_label,
        "domain": domain,
        "prompt": prompt,
        "chatContextEndpoint": "/api/v1/graphrag/chat-context-v2",
        "chatContextRequest": chat_context_request,
        "suggestedQuestions": suggested_questions,
        "recommendedMode": "batch_graph_path_reasoning" if graph_overview.get("cross_document_path_count") else "batch_relationship_comparison",
        "answerGuidance": (
            "Compare the batch graph overview first, then cite each supporting file and its slide/page/table/chart/connector provenance."
        ),
        "citationScopes": citation_scopes[:10],
        "graphAnchorCount": graph_anchor_count,
        "retrievalRehearsalCount": retrieval_rehearsal_count,
        "evidenceTotals": evidence_hints,
        "evidenceFamilies": sorted(evidence_hints),
        "graphOverviewCounts": graph_overview_counts,
    }
    actions = [
        {
            "id": "ask_chat_across_batch",
            "label": "Ask Chat Across Batch",
            "ready": True,
            "target": "chat",
        },
        {
            "id": "inspect_batch_graph",
            "label": "Inspect Batch Graph",
            "ready": bool(graph_overview.get("relationship_count")),
            "target": "pipeline_traces",
        },
    ]

    return {
        "ready": True,
        "run_ids": run_ids,
        "filename": filename_label,
        "filenames": filenames[:12],
        "domain": domain,
        "domains": sorted_domains,
        "ready_files": ready_files,
        "recommended_mode": route_handoff["recommendedMode"],
        "answer_guidance": route_handoff["answerGuidance"],
        "citation_scopes": citation_scopes[:10],
        "starter_prompts": suggested_questions,
        "suggested_questions": suggested_questions,
        "chat_context_endpoint": "/api/v1/graphrag/chat-context-v2",
        "chat_context_request": chat_context_request,
        "route_state": {"graphragHandoff": route_handoff},
        "graph_hints": context_hints,
        "evidence_totals": evidence_hints,
        "evidence_families": sorted(evidence_hints),
        "graph_overview_counts": graph_overview_counts,
        "actions": actions,
    }


def _pipeline_batch_upload_manifest(
    results: list[dict[str, Any]],
    *,
    operational_handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    handoff_status_counts: dict[str, int] = {}
    interaction_mode_counts: dict[str, int] = {}
    payload_kinds: dict[str, int] = {}
    domains: set[str] = set()
    files: list[dict[str, Any]] = []
    next_steps: list[str] = []
    accepted = 0
    failed = 0
    trace_ready = 0
    chat_ready = 0
    remote_ready = 0
    remote_submitted = 0
    remote_waiting = 0
    retrieval_rehearsals = 0
    rehearsed_chunk_matches = 0
    rehearsed_anchor_matches = 0
    total_bytes = 0
    evidence_totals = _empty_batch_evidence_totals()
    largest_file: dict[str, Any] | None = None

    for result in results:
        run = result.get("run") if isinstance(result.get("run"), dict) else {}
        document = run.get("document", {}) if isinstance(run.get("document"), dict) else {}
        summary = run.get("summary", {}) if isinstance(run.get("summary"), dict) else {}
        upstream_preflight = run.get("upstream_preflight", {}) if isinstance(run.get("upstream_preflight"), dict) else {}
        upstream_ingest = run.get("upstream_ingest", {}) if isinstance(run.get("upstream_ingest"), dict) else {}
        preflight_samples = upstream_preflight.get("sample_outputs", {}) if isinstance(upstream_preflight.get("sample_outputs"), dict) else {}
        upstream_samples = upstream_ingest.get("sample_outputs", {}) if isinstance(upstream_ingest.get("sample_outputs"), dict) else {}
        handoff = run.get("operational_handoff", {}) if isinstance(run.get("operational_handoff"), dict) else {}
        deployment_outcome = run.get("deployment_outcome", {}) if isinstance(run.get("deployment_outcome"), dict) else {}
        deployment_ingest = deployment_outcome.get("ingest", {}) if isinstance(deployment_outcome.get("ingest"), dict) else {}
        interaction_handoff = run.get("interaction_handoff", {}) if isinstance(run.get("interaction_handoff"), dict) else {}
        interaction_counts = interaction_handoff.get("counts", {}) if isinstance(interaction_handoff.get("counts"), dict) else {}

        filename = str(result.get("filename") or document.get("filename") or f"file-{result.get('index', len(files)) + 1}")
        status = str(result.get("status") or run.get("status") or "unknown")
        success = result.get("success") is not False
        if success:
            accepted += 1
        else:
            failed += 1
        _increment_count(status_counts, status)

        payload_kind = (
            upstream_samples.get("payload_kind")
            or upstream_preflight.get("payload_kind")
            or preflight_samples.get("payload_kind")
            or "unknown"
        )
        _increment_count(payload_kinds, payload_kind)
        domain = str(document.get("domain") or preflight_samples.get("domain") or "").strip()
        if domain:
            domains.add(domain)

        size_bytes = int(_number_value(document.get("size_bytes"), -1))
        payload_bytes = int(_number_value(upstream_samples.get("payload_bytes") or preflight_samples.get("payload_bytes"), -1))
        upload_bytes = size_bytes if size_bytes >= 0 else max(payload_bytes, 0)
        total_bytes += max(upload_bytes, 0)
        if largest_file is None or upload_bytes > int(largest_file.get("size_bytes") or 0):
            largest_file = {
                "filename": filename,
                "size_bytes": max(upload_bytes, 0),
            }

        run_id = run.get("run_id")
        item_trace_ready = bool(handoff.get("local_trace_ready")) or bool(run_id and summary.get("chunk_count", 0))
        item_chat_ready = bool(handoff.get("chat_handoff_ready")) or bool(run_id and (
            summary.get("relationship_candidate_count", 0)
            or summary.get("local_graph_edge_count", 0)
            or summary.get("local_context_chunk_count", 0)
        ))
        item_remote_ready = bool(handoff.get("remote_ingest_ready")) or bool(upstream_preflight.get("ready"))
        item_remote_submitted = bool(handoff.get("remote_ingest_submitted")) or bool(upstream_ingest.get("submitted"))
        item_remote_waiting = bool(item_trace_ready and not item_remote_submitted)
        if item_trace_ready:
            trace_ready += 1
        if item_chat_ready:
            chat_ready += 1
        if item_remote_ready:
            remote_ready += 1
        if item_remote_submitted:
            remote_submitted += 1
        if item_remote_waiting:
            remote_waiting += 1
        retrieval_rehearsals += int(_number_value(interaction_counts.get("retrieval_rehearsals")))
        rehearsed_chunk_matches += int(_number_value(interaction_counts.get("rehearsed_chunk_matches")))
        rehearsed_anchor_matches += int(_number_value(interaction_counts.get("rehearsed_anchor_matches")))
        evidence_profile = _evidence_profile_from_run(run)
        quick_counts = evidence_profile.get("quick_counts", {}) if isinstance(evidence_profile.get("quick_counts"), dict) else {}
        graph_evidence = evidence_profile.get("graph_evidence", {}) if isinstance(evidence_profile.get("graph_evidence"), dict) else {}
        item_evidence_totals = _batch_evidence_totals_from_profile(quick_counts, graph_evidence)
        item_graph_evidence_counts = _positive_counts(graph_evidence)
        _add_batch_evidence_totals(evidence_totals, item_evidence_totals)
        item_layout_elements = item_evidence_totals["layout_elements"]
        item_layout_relationships = item_evidence_totals["layout_relationships"]

        handoff_status = (
            handoff.get("status")
            or ("remote_graph_submitted" if item_remote_submitted else "local_trace_ready" if item_trace_ready else status)
        )
        _increment_count(handoff_status_counts, handoff_status)
        interaction_mode = interaction_handoff.get("recommended_mode")
        if interaction_mode:
            _increment_count(interaction_mode_counts, interaction_mode)
        next_step = (
            handoff.get("next_step")
            or upstream_ingest.get("next_step")
            or upstream_preflight.get("next_step")
            or upstream_samples.get("next_step")
            or preflight_samples.get("next_step")
        )
        if isinstance(next_step, str) and next_step and next_step not in next_steps:
            next_steps.append(next_step)

        target_url = (
            upstream_ingest.get("target_url")
            or upstream_preflight.get("target_url")
            or upstream_samples.get("target_url")
            or preflight_samples.get("target_url")
        )
        selected_endpoint = upstream_ingest.get("selected_endpoint") or upstream_samples.get("selected_endpoint")
        error = result.get("error") or upstream_ingest.get("error") or upstream_preflight.get("error")
        file_record = {
            "index": result.get("index"),
            "filename": filename,
            "content_type": document.get("content_type"),
            "size_bytes": max(upload_bytes, 0),
            "payload_kind": payload_kind,
            "payload_bytes": max(payload_bytes, 0) if payload_bytes >= 0 else None,
            "status": status,
            "success": success,
            "run_id": run_id,
            "duration_ms": result.get("duration_ms"),
            "completion_order": result.get("completion_order"),
            "trace_ready": item_trace_ready,
            "chat_ready": item_chat_ready,
            "remote_ingest_ready": item_remote_ready,
            "remote_submitted": item_remote_submitted,
            "remote_waiting": item_remote_waiting,
            "deployment_status": deployment_outcome.get("status"),
            "deployment_severity": deployment_outcome.get("severity"),
            "endpoint_attempt_count": deployment_ingest.get("endpoint_attempt_count", len(upstream_samples.get("endpoint_attempts", []) or [])),
            "endpoint_fallback_used": deployment_outcome.get("fallback_used", False),
            "handoff_status": handoff_status,
            "primary_action": handoff.get("primary_action"),
            "interaction_ready": interaction_handoff.get("ready", False),
            "interaction_mode": interaction_mode,
            "graph_anchor_count": interaction_counts.get("graph_anchors", 0),
            "starter_prompt_count": interaction_counts.get("starter_prompts", 0),
            "capability_card_count": interaction_counts.get("capability_cards", 0),
            "retrieval_rehearsal_count": interaction_counts.get("retrieval_rehearsals", 0),
            "rehearsed_chunk_match_count": interaction_counts.get("rehearsed_chunk_matches", 0),
            "rehearsed_anchor_match_count": interaction_counts.get("rehearsed_anchor_matches", 0),
            "evidence_counts": _positive_counts(item_evidence_totals),
            "graph_evidence_counts": item_graph_evidence_counts,
            "layout_element_count": item_layout_elements,
            "layout_relationship_count": item_layout_relationships,
            "distinct_table_count": item_evidence_totals["distinct_tables"],
            "link_count": item_evidence_totals["links"],
            "form_count": item_evidence_totals["forms"],
            "diagram_count": item_evidence_totals["diagrams"],
            "code_block_count": item_evidence_totals["code_blocks"],
            "target_url": target_url,
            "selected_endpoint": selected_endpoint,
            "service_status": handoff.get("service_status") or upstream_preflight.get("service_status"),
            "next_step": next_step,
        }
        if error:
            file_record["error"] = str(error)
        files.append(file_record)

    total_files = len(results)
    handoff_primary_action = operational_handoff.get("primary_action") if isinstance(operational_handoff, dict) else None
    handoff_next_step = operational_handoff.get("next_step") if isinstance(operational_handoff, dict) else None
    if handoff_primary_action:
        primary_action = handoff_primary_action
    elif failed and not chat_ready:
        primary_action = "inspect_errors"
    elif chat_ready > 1:
        primary_action = "ask_chat_across_batch"
    elif chat_ready == 1:
        primary_action = "ask_chat"
    elif trace_ready:
        primary_action = "inspect_traces"
    else:
        primary_action = "inspect_errors"

    if isinstance(handoff_next_step, str) and handoff_next_step:
        next_step = handoff_next_step
    elif next_steps:
        next_step = next_steps[0]
    elif chat_ready:
        next_step = "Ask Chat across the batch to compare entities, relationships, and graph paths."
    elif trace_ready:
        next_step = "Inspect the local pipeline traces, then rerun upload after GraphRAG is reachable."
    else:
        next_step = "Inspect failed files and retry the batch upload."

    lanes = [
        {
            "id": "upload",
            "label": "Upload",
            "ready": accepted > 0,
            "complete": total_files > 0 and failed == 0,
            "count": accepted,
            "waiting": failed,
        },
        {
            "id": "trace",
            "label": "Trace",
            "ready": trace_ready > 0,
            "complete": total_files > 0 and trace_ready == total_files,
            "count": trace_ready,
            "waiting": max(total_files - trace_ready, 0),
        },
        {
            "id": "chat",
            "label": "Chat",
            "ready": chat_ready > 0,
            "complete": total_files > 0 and chat_ready == total_files,
            "count": chat_ready,
            "waiting": max(total_files - chat_ready, 0),
        },
        {
            "id": "remote_graph",
            "label": "Remote Graph",
            "ready": remote_submitted > 0,
            "complete": total_files > 0 and remote_submitted == total_files,
            "count": remote_submitted,
            "waiting": remote_waiting,
        },
    ]

    return {
        "file_count": total_files,
        "accepted": accepted,
        "failed": failed,
        "trace_ready": trace_ready,
        "chat_ready": chat_ready,
        "remote_ingest_ready": remote_ready,
        "remote_submitted": remote_submitted,
        "remote_waiting": remote_waiting,
        "retrieval_rehearsals": retrieval_rehearsals,
        "rehearsed_chunk_matches": rehearsed_chunk_matches,
        "rehearsed_anchor_matches": rehearsed_anchor_matches,
        **evidence_totals,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 3),
        "largest_file": largest_file,
        "domains": sorted(domains),
        "status_counts": status_counts,
        "handoff_status_counts": handoff_status_counts,
        "interaction_mode_counts": interaction_mode_counts,
        "payload_kinds": payload_kinds,
        "primary_action": primary_action,
        "next_step": next_step,
        "lanes": lanes,
        "files": files,
    }


def _pipeline_batch_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    payload_kinds: dict[str, int] = {}
    handoff_status_counts: dict[str, int] = {}
    evidence_totals = _empty_batch_evidence_totals()
    totals = {
        "chunks": 0,
        "entity_candidates": 0,
        "relationship_candidates": 0,
    }
    readiness_scores: list[float] = []
    remote_submitted = 0
    local_preview_only = 0
    trace_ready = 0
    chat_ready = 0
    remote_ready = 0
    remote_waiting = 0
    failed_files: list[str] = []
    next_steps: list[str] = []
    interaction_mode_counts: dict[str, int] = {}
    interaction_rehearsal_totals = {
        "retrieval_rehearsals": 0,
        "chunk_matches": 0,
        "anchor_matches": 0,
    }

    for result in results:
        run = result.get("run") if isinstance(result.get("run"), dict) else {}
        status = str(result.get("status") or run.get("status") or "unknown")
        _increment_count(status_counts, status)

        document = run.get("document", {}) if isinstance(run.get("document"), dict) else {}
        filename = str(result.get("filename") or document.get("filename") or "")
        if result.get("success") is False:
            failed_files.append(filename or f"item-{result.get('index', len(failed_files))}")

        summary = run.get("summary", {}) if isinstance(run.get("summary"), dict) else {}
        totals["chunks"] += int(_number_value(summary.get("chunk_count")))
        totals["entity_candidates"] += int(_number_value(summary.get("entity_candidate_count")))
        totals["relationship_candidates"] += int(_number_value(summary.get("relationship_candidate_count")))

        profile = run.get("document_profile", {}) if isinstance(run.get("document_profile"), dict) else {}
        readiness = profile.get("readiness_score", summary.get("readiness_score"))
        if readiness is not None:
            readiness_scores.append(_number_value(readiness))

        upstream_ingest = run.get("upstream_ingest", {}) if isinstance(run.get("upstream_ingest"), dict) else {}
        upstream_preflight = run.get("upstream_preflight", {}) if isinstance(run.get("upstream_preflight"), dict) else {}
        ingest_samples = upstream_ingest.get("sample_outputs", {}) if isinstance(upstream_ingest.get("sample_outputs"), dict) else {}
        preflight_samples = upstream_preflight.get("sample_outputs", {}) if isinstance(upstream_preflight.get("sample_outputs"), dict) else {}
        payload_kind = ingest_samples.get("payload_kind") or upstream_preflight.get("payload_kind") or preflight_samples.get("payload_kind")
        _increment_count(payload_kinds, payload_kind or "unknown")

        if upstream_ingest.get("submitted") is True:
            remote_submitted += 1
        elif status == "preview_only" or upstream_ingest.get("status") == "skipped":
            local_preview_only += 1

        handoff = run.get("operational_handoff", {}) if isinstance(run.get("operational_handoff"), dict) else {}
        if handoff:
            _increment_count(handoff_status_counts, handoff.get("status") or "unknown")
            if handoff.get("local_trace_ready"):
                trace_ready += 1
            if handoff.get("chat_handoff_ready"):
                chat_ready += 1
            if handoff.get("remote_ingest_ready"):
                remote_ready += 1
            if not handoff.get("remote_ingest_submitted"):
                remote_waiting += 1

        next_step = (
            (run.get("operational_handoff", {}) or {}).get("next_step") if isinstance(run.get("operational_handoff"), dict) else None
        ) or (
            upstream_ingest.get("next_step")
            or upstream_preflight.get("next_step")
            or ingest_samples.get("next_step")
            or preflight_samples.get("next_step")
        )
        if isinstance(next_step, str) and next_step and next_step not in next_steps:
            next_steps.append(next_step)

        evidence_profile = _evidence_profile_from_run(run)
        quick_counts = evidence_profile.get("quick_counts", {}) if isinstance(evidence_profile.get("quick_counts"), dict) else {}
        graph_evidence = evidence_profile.get("graph_evidence", {}) if isinstance(evidence_profile.get("graph_evidence"), dict) else {}
        _add_batch_evidence_totals(
            evidence_totals,
            _batch_evidence_totals_from_profile(quick_counts, graph_evidence),
        )
        interaction_handoff = run.get("interaction_handoff", {}) if isinstance(run.get("interaction_handoff"), dict) else {}
        if interaction_handoff.get("recommended_mode"):
            _increment_count(interaction_mode_counts, interaction_handoff.get("recommended_mode"))
        interaction_counts = interaction_handoff.get("counts", {}) if isinstance(interaction_handoff.get("counts"), dict) else {}
        interaction_rehearsal_totals["retrieval_rehearsals"] += int(_number_value(interaction_counts.get("retrieval_rehearsals")))
        interaction_rehearsal_totals["chunk_matches"] += int(_number_value(interaction_counts.get("rehearsed_chunk_matches")))
        interaction_rehearsal_totals["anchor_matches"] += int(_number_value(interaction_counts.get("rehearsed_anchor_matches")))

    readiness = {
        "average": round(sum(readiness_scores) / len(readiness_scores), 1),
        "minimum": round(min(readiness_scores), 1),
        "maximum": round(max(readiness_scores), 1),
    } if readiness_scores else None
    total_files = len(results)
    if total_files and remote_submitted == total_files:
        batch_status = "remote_graph_submitted"
        batch_label = "Batch Remote Submitted"
        batch_message = "Every file produced a local trace and submitted to the configured GraphRAG API."
        primary_action = "ask_chat_across_batch" if chat_ready else "inspect_traces"
    elif chat_ready:
        batch_status = "local_traces_ready"
        batch_label = "Batch Local Traces Ready"
        batch_message = "Batch traces and run-scoped chat are ready; at least one file is waiting on remote GraphRAG ingest."
        primary_action = "ask_chat_across_batch"
    elif trace_ready:
        batch_status = "traces_ready"
        batch_label = "Batch Traces Ready"
        batch_message = "Batch traces were generated, but chat handoff is not ready for every trace."
        primary_action = "inspect_traces"
    else:
        batch_status = "batch_attention"
        batch_label = "Batch Needs Attention"
        batch_message = "No chat-ready GraphRAG traces were produced for this batch."
        primary_action = "inspect_errors"

    graph_overview = _pipeline_batch_graph_overview(results)
    batch_resume_handoff = _pipeline_batch_resume_handoff(results, graph_overview=graph_overview)
    operational_handoff = {
        "status": batch_status,
        "label": batch_label,
        "message": batch_message,
        "total_files": total_files,
        "trace_ready": trace_ready,
        "chat_ready": chat_ready,
        "remote_ingest_ready": remote_ready,
        "remote_submitted": remote_submitted,
        "remote_waiting": remote_waiting,
        "failed": len(failed_files),
        "handoff_status_counts": handoff_status_counts,
        "next_step": next_steps[0] if next_steps else None,
        "primary_action": primary_action,
        "actions": [
            {
                "id": "ask_chat_across_batch",
                "label": "Ask Chat Across Batch",
                "ready": chat_ready > 0,
                "target": "chat",
            },
            {
                "id": "inspect_traces",
                "label": "Inspect traces",
                "ready": trace_ready > 0,
                "target": "pipeline_traces",
            },
            {
                "id": "service_smoke_check",
                "label": "Service smoke check",
                "ready": remote_waiting == 0,
                "target": GRAPHRAG_SERVICES_STATUS_PATH,
            },
        ],
        "diagnostics_path": GRAPHRAG_DEPLOYMENT_DIAGNOSTICS_PATH,
        "services_status_path": GRAPHRAG_SERVICES_STATUS_PATH,
    }
    if batch_resume_handoff:
        operational_handoff["chat_context_endpoint"] = batch_resume_handoff["chat_context_endpoint"]
        operational_handoff["chat_context_request"] = batch_resume_handoff["chat_context_request"]
        operational_handoff["route_state"] = batch_resume_handoff["route_state"]
        operational_handoff["batch_resume_handoff_ready"] = True
    upload_manifest = _pipeline_batch_upload_manifest(results, operational_handoff=operational_handoff)
    if batch_resume_handoff:
        upload_manifest["chat_context_endpoint"] = batch_resume_handoff["chat_context_endpoint"]
        upload_manifest["chat_context_request"] = batch_resume_handoff["chat_context_request"]
        upload_manifest["route_state"] = batch_resume_handoff["route_state"]
        upload_manifest["batch_resume_handoff"] = batch_resume_handoff

    return {
        "status_counts": status_counts,
        "payload_kinds": payload_kinds,
        "totals": totals,
        "evidence_totals": evidence_totals,
        "graph_overview": graph_overview,
        "interaction_mode_counts": interaction_mode_counts,
        "interaction_rehearsal_totals": interaction_rehearsal_totals,
        "readiness": readiness,
        "remote_submitted": remote_submitted,
        "local_preview_only": local_preview_only,
        "failed_files": failed_files,
        "next_steps": next_steps[:5],
        "operational_handoff": operational_handoff,
        "batch_resume_handoff": batch_resume_handoff,
        "upload_manifest": upload_manifest,
    }


def _stage(name: str, status: str, start: float, **payload: Any) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "duration_ms": _elapsed_ms(start),
        **payload,
    }


def _elapsed_ms(start: float) -> int:
    return max(0, round((time.perf_counter() - start) * 1000))


def _files_per_second(total_files: int, duration_ms: int) -> float:
    if total_files <= 0:
        return 0.0
    if duration_ms <= 0:
        return float(total_files)
    return round(total_files / (duration_ms / 1000), 3)


def _safe_json_response(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


async def _probe_upstream_health(health_urls: str | list[str]) -> dict[str, Any]:
    urls = [health_urls] if isinstance(health_urls, str) else list(health_urls or [])
    attempts: list[dict[str, Any]] = []
    if not urls:
        return {
            "status": "not_configured",
            "ready": False,
            "service_status": "not_configured",
            "health_attempts": attempts,
            "attempted_health_urls": [],
        }

    client = get_http_client()
    response: httpx.Response | None = None
    health_url = urls[0]
    for index, candidate_url in enumerate(urls):
        health_url = candidate_url
        try:
            response = await client.get(candidate_url, timeout=5.0)
        except Exception as exc:
            attempts.append({
                "url": candidate_url,
                "status": "unavailable",
                "error": str(exc),
            })
            return {
                "status": "unavailable",
                "ready": False,
                "service_status": "unavailable",
                "error": str(exc),
                "health_url": candidate_url,
                "health_attempts": attempts,
                "attempted_health_urls": urls[:index + 1],
            }
        attempts.append({
            "url": candidate_url,
            "status": "http_response",
            "status_code": response.status_code,
        })
        if response.status_code in {404, 405} and index < len(urls) - 1:
            attempts[-1]["status"] = "not_found"
            continue
        break

    if response is None:
        return {
            "status": "unavailable",
            "ready": False,
            "service_status": "unavailable",
            "health_url": health_url,
            "health_attempts": attempts,
            "attempted_health_urls": [attempt.get("url") for attempt in attempts if attempt.get("url")],
        }

    payload = _safe_json_response(response)
    service_status = (payload.get("status") or "healthy") if isinstance(payload, dict) else "healthy"
    unhealthy_statuses = {"unhealthy", "unavailable", "disabled", "misconfigured", "error", "failed"}
    if response.status_code >= 400:
        return {
            "status": "unhealthy",
            "ready": False,
            "service_status": "unhealthy",
            "http_unhealthy": True,
            "status_code": response.status_code,
            "response": payload,
            "health_url": health_url,
            "health_attempts": attempts,
            "attempted_health_urls": [attempt.get("url") for attempt in attempts if attempt.get("url")],
        }
    if isinstance(service_status, str) and service_status.lower() in unhealthy_statuses:
        return {
            "status": service_status.lower(),
            "ready": False,
            "service_status": service_status,
            "status_code": response.status_code,
            "response": payload,
            "health_url": health_url,
            "health_attempts": attempts,
            "attempted_health_urls": [attempt.get("url") for attempt in attempts if attempt.get("url")],
        }
    return {
        "status": "ready",
        "ready": True,
        "service_status": service_status,
        "status_code": response.status_code,
        "response": payload,
        "health_url": health_url,
        "health_attempts": attempts,
        "attempted_health_urls": [attempt.get("url") for attempt in attempts if attempt.get("url")],
    }


async def _build_upstream_preflight_stage(
    *,
    filename: str | None,
    content: bytes,
    content_type: str | None,
    domain: str,
    use_semantic_chunking: bool,
    build_knowledge_graph: bool,
    payload_kind: str,
    original_filename: str | None = None,
    health_probe: dict[str, Any] | None = None,
    remote_payload_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = get_graphrag_config()
    start = time.perf_counter()
    service_url = cfg.graphrag_url.rstrip("/") if cfg.graphrag_url else None
    target_url = f"{service_url}/ingest-documents" if service_url else None
    compatible_health_urls = _compatible_health_urls(service_url)
    health_url = compatible_health_urls[0] if compatible_health_urls else None
    compatible_ingest_urls = _compatible_ingest_urls(service_url)
    sample_outputs = {
        "service_base_url": service_url,
        "target_url": target_url,
        "health_url": health_url,
        "compatible_health_urls": compatible_health_urls,
        "services_status_path": GRAPHRAG_SERVICES_STATUS_PATH,
        "diagnostics_path": GRAPHRAG_DEPLOYMENT_DIAGNOSTICS_PATH,
        "compatible_ingest_urls": compatible_ingest_urls,
        "payload_kind": payload_kind,
        "submitted_filename": filename,
        "submitted_content_type": content_type or "application/octet-stream",
        "original_filename": original_filename or filename,
        "domain": domain,
        "use_semantic_chunking": use_semantic_chunking,
        "build_knowledge_graph": build_knowledge_graph,
        "payload_bytes": len(content),
        "payload_sha256": hashlib.sha256(content).hexdigest(),
        "deployment_hint": GRAPHRAG_DEPLOYMENT_HINT,
    }
    if remote_payload_profile:
        sample_outputs.update(remote_payload_profile)

    if not cfg.enabled:
        next_step = "Set GRAPHRAG_ENABLED=true, restart the backend, then rerun the Service Smoke Check."
        sample_outputs["next_step"] = next_step
        return _stage(
            "upstream_preflight",
            "skipped",
            start,
            ready=False,
            submitted=False,
            message="GraphRAG is disabled; upload will remain a local pipeline preview.",
            next_step=next_step,
            deploy_command=GRAPHRAG_DEPLOYMENT_HINT,
            counts={"payload_bytes": len(content)},
            sample_outputs=sample_outputs,
        )

    if not cfg.graphrag_url:
        next_step = "Set GRAPHRAG_URL or GRAPHRAG_SERVICE_URL to the GraphRAG API base URL."
        sample_outputs["next_step"] = next_step
        return _stage(
            "upstream_preflight",
            "skipped",
            start,
            ready=False,
            submitted=False,
            message="GRAPHRAG_URL is not configured; upload will remain a local pipeline preview.",
            next_step=next_step,
            deploy_command=GRAPHRAG_DEPLOYMENT_HINT,
            counts={"payload_bytes": len(content)},
            sample_outputs=sample_outputs,
        )

    probe = dict(health_probe or await _probe_upstream_health(compatible_health_urls))
    if probe.get("health_url"):
        health_url = str(probe["health_url"])
        sample_outputs["health_url"] = health_url
    if probe.get("attempted_health_urls"):
        sample_outputs["attempted_health_urls"] = probe.get("attempted_health_urls")
    if probe.get("health_attempts"):
        sample_outputs["health_attempts"] = probe.get("health_attempts")
    if probe.get("shared"):
        sample_outputs["shared_health_probe"] = True
    if probe.get("error"):
        sample_outputs["health_error"] = probe.get("error")
    if "response" in probe:
        sample_outputs["health_response"] = probe.get("response")
    if probe.get("status_code") is not None:
        sample_outputs["health_status_code"] = probe.get("status_code")

    if probe.get("status") == "unavailable":
        next_step = "Open the Service Smoke Check or call the health URL to verify the configured GraphRAG API is reachable."
        sample_outputs["next_step"] = next_step
        return _stage(
            "upstream_preflight",
            "unavailable",
            start,
            ready=False,
            submitted=False,
            target_url=target_url,
            health_url=health_url,
            service_status="unavailable",
            message="GraphRAG URL is configured, but the service health check failed; upload will remain a local pipeline preview.",
            next_step=next_step,
            deploy_command=GRAPHRAG_DEPLOYMENT_HINT,
            error=probe.get("error"),
            counts={"payload_bytes": len(content)},
            sample_outputs=sample_outputs,
        )

    if probe.get("http_unhealthy"):
        next_step = "Inspect the GraphRAG health response and fix the upstream service before retrying upload."
        sample_outputs["next_step"] = next_step
        return _stage(
            "upstream_preflight",
            "unhealthy",
            start,
            ready=False,
            submitted=False,
            target_url=target_url,
            health_url=health_url,
            service_status="unhealthy",
            status_code=probe.get("status_code"),
            message=f"GraphRAG health check returned HTTP {probe.get('status_code')}; upload will remain a local pipeline preview.",
            next_step=next_step,
            deploy_command=GRAPHRAG_DEPLOYMENT_HINT,
            counts={"payload_bytes": len(content)},
            sample_outputs=sample_outputs,
        )
    service_status = probe.get("service_status") or "healthy"
    if probe.get("ready") is False:
        next_step = "Inspect the GraphRAG health response and service logs before retrying upload."
        sample_outputs["next_step"] = next_step
        return _stage(
            "upstream_preflight",
            str(probe.get("status") or service_status).lower(),
            start,
            ready=False,
            submitted=False,
            target_url=target_url,
            health_url=health_url,
            service_status=service_status,
            status_code=probe.get("status_code"),
            message=f"GraphRAG health check reported {service_status}; upload will remain a local pipeline preview.",
            next_step=next_step,
            deploy_command=GRAPHRAG_DEPLOYMENT_HINT,
            counts={"payload_bytes": len(content)},
            sample_outputs=sample_outputs,
        )

    sample_outputs["next_step"] = "GraphRAG health probe passed; remote ingest can receive this payload."
    return _stage(
        "upstream_preflight",
        "ready",
        start,
        ready=True,
        submitted=False,
        target_url=target_url,
        health_url=health_url,
        service_status=service_status,
        status_code=probe.get("status_code"),
        payload_kind=payload_kind,
        submitted_filename=filename,
        submitted_content_type=content_type or "application/octet-stream",
        counts={"payload_bytes": len(content)},
        sample_outputs=sample_outputs,
    )


def _skipped_upstream_ingest_stage(preflight_stage: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    sample_outputs = preflight_stage.get("sample_outputs", {}) if isinstance(preflight_stage.get("sample_outputs"), dict) else {}
    remote_payload_profile = {
        key: value
        for key, value in sample_outputs.items()
        if str(key).startswith("remote_")
    }
    return _stage(
        "upstream_ingest",
        "skipped",
        start,
        submitted=False,
        reason="preflight_not_ready",
        message=preflight_stage.get("message") or "GraphRAG upstream preflight did not report ready.",
        next_step=preflight_stage.get("next_step") or sample_outputs.get("next_step"),
        target_url=preflight_stage.get("target_url") or sample_outputs.get("target_url"),
        sample_outputs={
            "target_url": preflight_stage.get("target_url") or sample_outputs.get("target_url"),
            "health_url": preflight_stage.get("health_url") or sample_outputs.get("health_url"),
            "payload_kind": sample_outputs.get("payload_kind"),
            "submitted_filename": sample_outputs.get("submitted_filename"),
            "submitted_content_type": sample_outputs.get("submitted_content_type"),
            "original_filename": sample_outputs.get("original_filename"),
            "payload_bytes": sample_outputs.get("payload_bytes"),
            "payload_sha256": sample_outputs.get("payload_sha256"),
            "preflight_status": preflight_stage.get("status"),
            "preflight_message": preflight_stage.get("message"),
            "next_step": preflight_stage.get("next_step") or sample_outputs.get("next_step"),
            "deployment_hint": sample_outputs.get("deployment_hint"),
            "services_status_path": sample_outputs.get("services_status_path"),
            "diagnostics_path": sample_outputs.get("diagnostics_path"),
            **remote_payload_profile,
        },
    )


async def _shared_batch_health_probe() -> dict[str, Any] | None:
    cfg = get_graphrag_config()
    if not cfg.enabled or not cfg.graphrag_url:
        return None
    service_url = cfg.graphrag_url.rstrip("/")
    probe = await _probe_upstream_health(_compatible_health_urls(service_url))
    return {**probe, "shared": True}


async def _forward_ingest_to_graphrag(
    *,
    filename: str | None,
    content: bytes,
    content_type: str | None,
    domain: str,
    use_semantic_chunking: bool,
    build_knowledge_graph: bool,
    payload_kind: str = "original_file",
    original_filename: str | None = None,
    remote_payload_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = get_graphrag_config()
    start = time.perf_counter()
    service_url = cfg.graphrag_url.rstrip("/") if cfg.graphrag_url else None
    compatible_ingest_urls = _compatible_ingest_urls(service_url)
    target_url = compatible_ingest_urls[0] if compatible_ingest_urls else None
    request_sample = {
        "service_base_url": service_url,
        "target_url": target_url,
        "compatible_ingest_urls": compatible_ingest_urls,
        "payload_kind": payload_kind,
        "submitted_filename": filename,
        "submitted_content_type": content_type or "application/octet-stream",
        "original_filename": original_filename or filename,
        "domain": domain,
        "use_semantic_chunking": use_semantic_chunking,
        "build_knowledge_graph": build_knowledge_graph,
        "payload_bytes": len(content),
        "payload_sha256": hashlib.sha256(content).hexdigest(),
    }
    if remote_payload_profile:
        request_sample.update(remote_payload_profile)

    if not cfg.enabled:
        return _stage(
            "upstream_ingest",
            "skipped",
            start,
            message="GraphRAG is disabled. Set GRAPHRAG_ENABLED=true to submit this document.",
            submitted=False,
            target_url=target_url,
            sample_outputs=request_sample,
        )
    if not cfg.graphrag_url:
        return _stage(
            "upstream_ingest",
            "skipped",
            start,
            message="GRAPHRAG_URL is not configured. Preview was generated locally but not submitted.",
            submitted=False,
            target_url=target_url,
            sample_outputs=request_sample,
        )

    client = get_http_client()
    data = {
        "domain": domain,
        "use_semantic_chunking": str(use_semantic_chunking).lower(),
        "build_knowledge_graph": str(build_knowledge_graph).lower(),
    }
    attempts: list[dict[str, Any]] = []
    last_response: httpx.Response | None = None
    last_payload: Any = None
    selected_url = target_url
    ingest_timeout = _remote_ingest_timeout_seconds(payload_kind=payload_kind, payload_bytes=len(content))
    request_sample["timeout_seconds"] = ingest_timeout
    for index, candidate_url in enumerate(compatible_ingest_urls):
        try:
            response = await client.post(
                candidate_url,
                files={"files": _file_tuple(filename, content, content_type or "application/octet-stream")},
                data=data,
                timeout=ingest_timeout,
            )
        except httpx.TimeoutException as e:
            attempts.append({
                "url": candidate_url,
                "status": "timed_out",
                "error": str(e),
                "timeout_seconds": ingest_timeout,
                "fallback_allowed": False,
            })
            next_step = (
                "The local pipeline trace and chat context are ready. "
                "Retry remote ingest or increase GRAPHRAG_REMOTE_INGEST_TIMEOUT if the configured GraphRAG API needs longer."
            )
            return _stage(
                "upstream_ingest",
                "timed_out",
                start,
                submitted=False,
                target_url=candidate_url,
                error=f"GraphRAG remote ingest timed out after {ingest_timeout:g}s: {str(e)}",
                next_step=next_step,
                sample_outputs={
                    **request_sample,
                    "target_url": candidate_url,
                    "attempted_urls": [attempt["url"] for attempt in attempts],
                    "endpoint_attempts": attempts,
                    "next_step": next_step,
                },
            )
        except httpx.HTTPError as e:
            attempts.append({
                "url": candidate_url,
                "status": "error",
                "error": str(e),
                "timeout_seconds": ingest_timeout,
                "fallback_allowed": False,
            })
            return _stage(
                "upstream_ingest",
                "failed",
                start,
                submitted=False,
                target_url=candidate_url,
                error=f"GraphRAG service error: {str(e)}",
                sample_outputs={
                    **request_sample,
                    "target_url": candidate_url,
                    "attempted_urls": [attempt["url"] for attempt in attempts],
                    "endpoint_attempts": attempts,
                },
            )

        payload = _safe_json_response(response)
        last_response = response
        last_payload = payload
        fallback_allowed = (
            response.status_code in GRAPHRAG_INGEST_FALLBACK_STATUS_CODES
            and index < len(compatible_ingest_urls) - 1
        )
        attempts.append({
            "url": candidate_url,
            "status": "complete" if response.status_code < 400 else "failed",
            "status_code": response.status_code,
            "timeout_seconds": ingest_timeout,
            "fallback_allowed": fallback_allowed,
        })
        selected_url = candidate_url
        if response.status_code < 400 or not fallback_allowed:
            break

    if last_response is None:
        return _stage(
            "upstream_ingest",
            "failed",
            start,
            submitted=False,
            target_url=target_url,
            error="No compatible GraphRAG ingest endpoint was available.",
            sample_outputs={
                **request_sample,
                "attempted_urls": [],
                "endpoint_attempts": [],
            },
        )

    response = last_response
    payload = last_payload
    final_status = "complete" if response.status_code < 400 else "failed"
    next_step = None
    if final_status == "failed" and len(attempts) > 1:
        next_step = "All compatible GraphRAG ingest endpoints failed; inspect endpoint_attempts and upstream service logs."
    elif final_status == "complete" and len(attempts) > 1:
        next_step = "Remote ingest succeeded after falling back to a compatible GraphRAG endpoint."

    return _stage(
        "upstream_ingest",
        final_status,
        start,
        submitted=response.status_code < 400,
        target_url=selected_url,
        selected_endpoint=selected_url,
        status_code=response.status_code,
        next_step=next_step,
        sample_outputs={
            **request_sample,
            "target_url": selected_url,
            "selected_endpoint": selected_url,
            "attempted_urls": [attempt["url"] for attempt in attempts],
            "endpoint_attempts": attempts,
            "response": payload,
        },
    )


@router.post("/extract")
async def graphrag_extract_entities(request: Request):
    """Extract entities and relationships from text using graphrag GLiNER."""
    check_graphrag_enabled()

    data = await request.json()
    text = data.get("text")

    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/extract-entities-relations",
            json=data,
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/ingest")
async def graphrag_ingest_document(
    request: Request,
    domain: str = "general",
    use_semantic_chunking: bool = True,
    build_knowledge_graph: bool = True,
):
    """
    Ingest a document through graphrag service.
    Note: For file uploads, use /api/v1/graphrag/ingest/upload endpoint.
    This endpoint is for JSON-based document submission.
    """
    check_graphrag_enabled()

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    text = payload.get("text") or payload.get("content") or payload.get("document")
    filename = payload.get("filename") or payload.get("document_name") or "document.txt"
    content_type = payload.get("content_type") or "text/plain"
    domain = payload.get("domain", domain)
    use_semantic_chunking = payload.get("use_semantic_chunking", use_semantic_chunking)
    build_knowledge_graph = payload.get("build_knowledge_graph", build_knowledge_graph)

    if not text:
        raise HTTPException(status_code=400, detail="text, content, or document is required")

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/ingest-documents",
            files={"files": _file_tuple(filename, str(text).encode("utf-8"), content_type)},
            data={
                "domain": domain,
                "use_semantic_chunking": str(use_semantic_chunking).lower(),
                "build_knowledge_graph": str(build_knowledge_graph).lower(),
            },
            timeout=300.0,
        )
        response.raise_for_status()
        return {
            "status": "success",
            "filename": filename,
            **response.json(),
        }
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/documents")
async def graphrag_list_documents():
    """List documents in graphrag service."""
    check_graphrag_enabled()

    try:
        client = get_http_client()
        response = await client.get(f"{_graphrag_url()}/documents/list", timeout=60.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.delete("/documents/{document_name:path}")
async def graphrag_delete_document(document_name: str):
    """Delete a document from graphrag service."""
    check_graphrag_enabled()

    try:
        client = get_http_client()
        response = await client.delete(f"{_graphrag_url()}/documents/{quote(document_name, safe='')}", timeout=60.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/ingest/upload")
async def graphrag_upload_document(
    file: UploadFile = File(...),
    domain: str = Form("general"),
    use_semantic_chunking: bool = Form(True),
    build_knowledge_graph: bool = Form(True),
):
    """Upload and process a document through GraphRAG pipeline."""
    check_graphrag_enabled()

    try:
        file_content = await file.read()

        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/ingest-documents",
            files={"files": _file_tuple(file.filename, file_content, file.content_type or "application/octet-stream")},
            data={
                "domain": domain,
                "use_semantic_chunking": str(use_semantic_chunking).lower(),
                "build_knowledge_graph": str(build_knowledge_graph).lower(),
            },
            timeout=300.0,
        )

        if response.status_code == 200:
            result = response.json()
            return {
                "status": "success",
                "filename": file.filename,
                **result,
            }
        else:
            error_detail = response.text
            try:
                error_detail = response.json().get("detail", error_detail)
            except Exception:
                pass
            raise HTTPException(
                status_code=response.status_code,
                detail=f"GraphRAG processing failed: {error_detail}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@router.post("/ingest/batch")
async def graphrag_upload_batch(
    files: List[UploadFile] = File(...),
    domain: str = Form("general"),
    use_semantic_chunking: bool = Form(True),
    build_knowledge_graph: bool = Form(True),
):
    """Batch upload documents to GraphRAG."""
    check_graphrag_enabled()

    errors = []

    client = get_http_client()
    upstream_files = []

    for file in files:
        try:
            file_content = await file.read()
            upstream_files.append((
                "files",
                _file_tuple(file.filename, file_content, file.content_type or "application/octet-stream"),
            ))
        except Exception as e:
            errors.append({"filename": file.filename, "success": False, "error": str(e)})

    if not upstream_files:
        return {"processed": 0, "failed": len(errors), "results": errors, "errors": errors}

    try:
        response = await client.post(
            f"{_graphrag_url()}/ingest-documents",
            files=upstream_files,
            data={
                "domain": domain,
                "use_semantic_chunking": str(use_semantic_chunking).lower(),
                "build_knowledge_graph": str(build_knowledge_graph).lower(),
            },
            timeout=300.0,
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")

    if response.status_code >= 400:
        errors.append({"filename": "batch", "success": False, "error": response.text})
        return {"processed": 0, "failed": len(errors), "results": errors, "errors": errors}

    result = response.json()
    if errors:
        result = {**result, "upload_errors": errors}
    return result


@router.post("/ingest/pipeline/preview")
async def graphrag_ingest_pipeline_preview(
    file: UploadFile = File(...),
    domain: str = Form("general"),
    use_semantic_chunking: bool = Form(True),
    build_knowledge_graph: bool = Form(True),
    sample_limit: int = Form(5),
    submit_structured_text: bool = Form(True),
):
    """Preview document extraction, chunking, and graph candidates without upstream ingestion."""
    file_content = await file.read()
    preview = await asyncio.to_thread(
        build_document_pipeline_preview,
        filename=file.filename or "document",
        content=file_content,
        content_type=file.content_type or "application/octet-stream",
        domain=domain,
        use_semantic_chunking=use_semantic_chunking,
        build_knowledge_graph=build_knowledge_graph,
        sample_limit=max(1, min(sample_limit, 80)),
        submit_structured_text=submit_structured_text,
        include_internal_payload=True,
    )
    _attach_source_document_artifact(
        preview,
        filename=file.filename or "document",
        content=file_content,
        content_type=file.content_type or "application/octet-stream",
    )
    _attach_structured_ingest_artifact(preview)
    _attach_pipeline_run_rollups(preview)
    return _remember_pipeline_run(preview)


async def _build_and_submit_pipeline_run(
    *,
    filename: str | None,
    content: bytes,
    content_type: str | None,
    domain: str,
    use_semantic_chunking: bool,
    build_knowledge_graph: bool,
    sample_limit: int,
    submit_structured_text: bool,
    health_probe: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    preview = await asyncio.to_thread(
        build_document_pipeline_preview,
        filename=filename or "document",
        content=content,
        content_type=content_type or "application/octet-stream",
        domain=domain,
        use_semantic_chunking=use_semantic_chunking,
        build_knowledge_graph=build_knowledge_graph,
        sample_limit=max(1, min(sample_limit, 80)),
        submit_structured_text=submit_structured_text,
        include_internal_payload=True,
        run_id=run_id,
    )
    structured_payload = preview.pop("_structured_ingest_payload", {})
    _attach_source_document_artifact(
        preview,
        filename=filename or "document",
        content=content,
        content_type=content_type or "application/octet-stream",
    )
    _attach_structured_ingest_artifact(preview, structured_payload)
    use_structured_text = bool(structured_payload.get("uses_structured_text"))
    upstream_filename = filename
    upstream_content = content
    upstream_content_type = content_type
    payload_kind = "original_file"
    remote_payload_profile: dict[str, Any] = {}
    if use_structured_text:
        upstream_filename = str(structured_payload.get("filename") or f"{filename or 'document'}.structured.md")
        remote_structured_text, remote_payload_profile = _build_remote_structured_ingest_digest(
            str(structured_payload.get("content") or ""),
            filename=filename or upstream_filename,
        )
        upstream_content = remote_structured_text.encode("utf-8")
        upstream_content_type = str(structured_payload.get("content_type") or "text/markdown")
        payload_kind = "structured_text"
        if isinstance(preview.get("structured_ingest"), dict):
            preview["structured_ingest"] = {
                **preview["structured_ingest"],
                "remote_digest_used": bool(remote_payload_profile.get("remote_digest_used")),
                "remote_payload_char_count": remote_payload_profile.get("remote_payload_char_count"),
                "remote_source_char_count": remote_payload_profile.get("remote_source_char_count"),
            }

    preflight_stage = await _build_upstream_preflight_stage(
        filename=upstream_filename,
        content=upstream_content,
        content_type=upstream_content_type,
        domain=domain,
        use_semantic_chunking=use_semantic_chunking,
        build_knowledge_graph=build_knowledge_graph,
        payload_kind=payload_kind,
        original_filename=filename,
        health_probe=health_probe,
        remote_payload_profile=remote_payload_profile,
    )
    preview["stages"].append(preflight_stage)
    preview["upstream_preflight"] = preflight_stage

    if preflight_stage.get("ready") is False:
        upstream_stage = _skipped_upstream_ingest_stage(preflight_stage)
    else:
        upstream_stage = await _forward_ingest_to_graphrag(
            filename=upstream_filename,
            content=upstream_content,
            content_type=upstream_content_type,
            domain=domain,
            use_semantic_chunking=use_semantic_chunking,
            build_knowledge_graph=build_knowledge_graph,
            payload_kind=payload_kind,
            original_filename=filename,
            remote_payload_profile=remote_payload_profile,
        )
    preview["stages"].append(upstream_stage)
    preview["updated_at"] = datetime.now(timezone.utc).isoformat()
    preview["upstream_ingest"] = upstream_stage
    if upstream_stage.get("status") == "complete":
        preview["status"] = "complete"
    elif upstream_stage.get("status") == "failed":
        preview["status"] = "upstream_failed"
    elif upstream_stage.get("status") == "timed_out":
        preview["status"] = "preview_only"
    else:
        preview["status"] = "preview_only"
    if preview["status"] in {"complete", "preview_only", "upstream_failed"}:
        preview["completed_at"] = preview["updated_at"]
    preview["operational_handoff"] = _pipeline_operational_handoff(preview)
    _attach_pipeline_run_rollups(preview)

    return _remember_pipeline_run(preview)


async def _complete_async_pipeline_upload(
    *,
    run_id: str,
    filename: str | None,
    content: bytes,
    content_type: str | None,
    domain: str,
    use_semantic_chunking: bool,
    build_knowledge_graph: bool,
    sample_limit: int,
    submit_structured_text: bool,
) -> None:
    try:
        completed = await _build_and_submit_pipeline_run(
            filename=filename,
            content=content,
            content_type=content_type,
            domain=domain,
            use_semantic_chunking=use_semantic_chunking,
            build_knowledge_graph=build_knowledge_graph,
            sample_limit=sample_limit,
            submit_structured_text=submit_structured_text,
            run_id=run_id,
        )
        completed["async_submission"] = {
            "enabled": True,
            "status": "complete",
            "completed_at": completed.get("updated_at"),
        }
        _remember_pipeline_run(completed)
    except Exception as exc:
        logger.exception("Async GraphRAG pipeline upload failed for %s", filename)
        failed = _PIPELINE_RUNS.get(run_id) or {
            "run_id": run_id,
            "document": {
                "filename": filename or "document",
                "domain": domain,
                "content_type": content_type or "application/octet-stream",
                "size_bytes": len(content),
            },
            "stages": [],
        }
        now = datetime.now(timezone.utc).isoformat()
        failed_stage = _stage(
            "upstream_ingest",
            "failed",
            time.perf_counter(),
            message="Background GraphRAG ingest failed.",
            error=str(exc),
            sample_outputs={"error": str(exc)},
        )
        stages = [stage for stage in failed.get("stages", []) if isinstance(stage, dict)]
        stages = [
            stage
            for stage in stages
            if str(stage.get("name") or "").lower() != "upstream_ingest"
        ]
        failed.update({
            "status": "upstream_failed",
            "updated_at": now,
            "completed_at": now,
            "stages": [*stages, failed_stage],
            "upstream_ingest": failed_stage,
            "async_submission": {
                "enabled": True,
                "status": "failed",
                "error": str(exc),
                "completed_at": now,
            },
        })
        failed["operational_handoff"] = _pipeline_operational_handoff(failed)
        _attach_pipeline_run_rollups(failed)
        _remember_pipeline_run(failed)


def _schedule_async_pipeline_upload(background_tasks: BackgroundTasks | None, **kwargs: Any) -> None:
    if background_tasks is not None:
        background_tasks.add_task(_complete_async_pipeline_upload, **kwargs)
        return
    asyncio.create_task(_complete_async_pipeline_upload(**kwargs))


def _form_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


async def _build_async_pipeline_upload_handoff(
    *,
    filename: str | None,
    content: bytes,
    content_type: str | None,
    domain: str,
    use_semantic_chunking: bool,
    build_knowledge_graph: bool,
    sample_limit: int,
    submit_structured_text: bool,
) -> dict[str, Any]:
    preview = await asyncio.to_thread(
        build_document_pipeline_preview,
        filename=filename or "document",
        content=content,
        content_type=content_type or "application/octet-stream",
        domain=domain,
        use_semantic_chunking=use_semantic_chunking,
        build_knowledge_graph=build_knowledge_graph,
        sample_limit=max(1, min(sample_limit, 80)),
        submit_structured_text=submit_structured_text,
        include_internal_payload=True,
    )
    structured_payload = preview.pop("_structured_ingest_payload", {})
    _attach_source_document_artifact(
        preview,
        filename=filename or "document",
        content=content,
        content_type=content_type or "application/octet-stream",
    )
    _attach_structured_ingest_artifact(preview, structured_payload)
    queued_stage = _stage(
        "upstream_ingest",
        "running",
        time.perf_counter(),
        method="background",
        message="Remote GraphRAG ingest is running in the background.",
        counts={"files": 1, "bytes": len(content)},
        sample_outputs={
            "filename": filename or "document",
            "handoff": "local trace returned; remote GraphRAG ingest continues in the background",
        },
    )
    preview["stages"].append(queued_stage)
    preview["upstream_ingest"] = queued_stage
    preview["status"] = "submitted"
    preview["updated_at"] = datetime.now(timezone.utc).isoformat()
    preview["async_submission"] = {
        "enabled": True,
        "status": "submitted",
        "queued_at": preview["updated_at"],
    }
    preview["operational_handoff"] = _pipeline_operational_handoff(preview)
    _attach_pipeline_run_rollups(preview)
    return _remember_pipeline_run(preview)


@router.post("/ingest/pipeline/upload")
async def graphrag_ingest_pipeline_upload(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    domain: str = Form("general"),
    use_semantic_chunking: bool = Form(True),
    build_knowledge_graph: bool = Form(True),
    sample_limit: int = Form(5),
    submit_structured_text: bool = Form(True),
    submit_async: bool = Form(False),
):
    """
    Upload a document with local pipeline tracing.

    The response includes stage samples for extraction, chunking, graph
    candidate generation, embedding input construction, and upstream ingest.
    """
    file_content = await file.read()
    if _form_bool(submit_async, default=False):
        handoff = await _build_async_pipeline_upload_handoff(
            filename=file.filename,
            content=file_content,
            content_type=file.content_type,
            domain=domain,
            use_semantic_chunking=use_semantic_chunking,
            build_knowledge_graph=build_knowledge_graph,
            sample_limit=sample_limit,
            submit_structured_text=submit_structured_text,
        )
        _schedule_async_pipeline_upload(
            background_tasks,
            run_id=str(handoff.get("run_id")),
            filename=file.filename,
            content=file_content,
            content_type=file.content_type,
            domain=domain,
            use_semantic_chunking=use_semantic_chunking,
            build_knowledge_graph=build_knowledge_graph,
            sample_limit=sample_limit,
            submit_structured_text=submit_structured_text,
        )
        return handoff
    return await _build_and_submit_pipeline_run(
        filename=file.filename,
        content=file_content,
        content_type=file.content_type,
        domain=domain,
        use_semantic_chunking=use_semantic_chunking,
        build_knowledge_graph=build_knowledge_graph,
        sample_limit=sample_limit,
        submit_structured_text=submit_structured_text,
    )


@router.post("/ingest/pipeline/batch")
async def graphrag_ingest_pipeline_batch(
    files: List[UploadFile] = File(...),
    domain: str = Form("general"),
    use_semantic_chunking: bool = Form(True),
    build_knowledge_graph: bool = Form(True),
    sample_limit: int = Form(5),
    submit_structured_text: bool = Form(True),
    max_concurrency: int = Form(3),
):
    """
    Upload multiple documents with local pipeline tracing and bounded concurrency.

    Each file receives the same stage trace as /ingest/pipeline/upload and is
    remembered by run_id for later inspection.
    """
    if not files:
        raise HTTPException(status_code=400, detail="files are required")

    batch_started_at = datetime.now(timezone.utc).isoformat()
    batch_start = time.perf_counter()
    bounded_concurrency = max(1, min(max_concurrency, 8))
    materialized_files, read_failures, materialized_total_bytes = await _materialize_pipeline_batch_uploads(files)
    for completion_order, failure in enumerate(read_failures, start=1):
        failure["completion_order"] = completion_order
    semaphore = asyncio.Semaphore(bounded_concurrency)
    active_lock = asyncio.Lock()
    completion_lock = asyncio.Lock()
    active_items = 0
    peak_concurrency = 0
    completed_items = len(read_failures)
    shared_health_probe = await _shared_batch_health_probe()

    async def process_file(item: dict[str, Any]) -> dict[str, Any]:
        nonlocal active_items, completed_items, peak_concurrency
        async with semaphore:
            index = int(item.get("index", 0))
            filename = str(item.get("filename") or f"file-{index + 1}")
            item_started_at = datetime.now(timezone.utc).isoformat()
            item_start = time.perf_counter()
            async with active_lock:
                active_items += 1
                peak_concurrency = max(peak_concurrency, active_items)
            try:
                run = await _build_and_submit_pipeline_run(
                    filename=filename,
                    content=bytes(item.get("content") or b""),
                    content_type=str(item.get("content_type") or "application/octet-stream"),
                    domain=domain,
                    use_semantic_chunking=use_semantic_chunking,
                    build_knowledge_graph=build_knowledge_graph,
                    sample_limit=sample_limit,
                    submit_structured_text=submit_structured_text,
                    health_probe=shared_health_probe,
                )
                success = run.get("status") != "upstream_failed"
                duration_ms = _elapsed_ms(item_start)
                async with completion_lock:
                    completed_items += 1
                    completion_order = completed_items
                return {
                    "index": index,
                    "filename": filename,
                    "success": success,
                    "status": run.get("status"),
                    "run_id": run.get("run_id"),
                    "started_at": item_started_at,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": duration_ms,
                    "read_duration_ms": item.get("read_duration_ms"),
                    "size_bytes": item.get("size_bytes"),
                    "completion_order": completion_order,
                    "run": run,
                    "summary": _pipeline_run_summary(run),
                }
            except Exception as exc:
                logger.warning(f"GraphRAG pipeline batch item failed for {filename}: {exc}")
                duration_ms = _elapsed_ms(item_start)
                async with completion_lock:
                    completed_items += 1
                    completion_order = completed_items
                return {
                    "index": index,
                    "filename": filename,
                    "success": False,
                    "status": "error",
                    "error": str(exc),
                    "started_at": item_started_at,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": duration_ms,
                    "read_duration_ms": item.get("read_duration_ms"),
                    "size_bytes": item.get("size_bytes"),
                    "completion_order": completion_order,
                }
            finally:
                async with active_lock:
                    active_items = max(0, active_items - 1)

    processed_results = await asyncio.gather(*(process_file(item) for item in materialized_files))
    results_by_index = {
        int(result.get("index", -1)): result
        for result in [*read_failures, *processed_results]
        if isinstance(result.get("index"), int)
    }
    results = [
        results_by_index[index]
        for index in range(len(files))
        if index in results_by_index
    ]
    duration_ms = _elapsed_ms(batch_start)
    batch_completed_at = datetime.now(timezone.utc).isoformat()
    processed = sum(1 for result in results if result.get("success"))
    failed = len(results) - processed
    runs = [result["run"] for result in results if isinstance(result.get("run"), dict)]
    execution = {
        "started_at": batch_started_at,
        "completed_at": batch_completed_at,
        "duration_ms": duration_ms,
        "requested_concurrency": max_concurrency,
        "max_concurrency": bounded_concurrency,
        "peak_concurrency": peak_concurrency,
        "throughput_files_per_second": _files_per_second(len(files), duration_ms),
        "shared_health_probe": shared_health_probe is not None,
        "upload_materialized": True,
        "materialized_file_count": len(materialized_files),
        "upload_read_failures": len(read_failures),
        "materialized_total_bytes": materialized_total_bytes,
        "upload_read_before_processing": True,
    }
    summary = _pipeline_batch_summary(results)
    summary["execution"] = execution
    upload_manifest = summary.get("upload_manifest", {})

    return {
        "status": "complete" if failed == 0 else "partial",
        "total_files": len(files),
        "processed": processed,
        "failed": failed,
        "concurrency": bounded_concurrency,
        "requested_concurrency": max_concurrency,
        "max_concurrency": bounded_concurrency,
        "peak_concurrency": peak_concurrency,
        "duration_ms": duration_ms,
        "throughput_files_per_second": execution["throughput_files_per_second"],
        "started_at": batch_started_at,
        "completed_at": batch_completed_at,
        "summary": summary,
        "upload_manifest": upload_manifest,
        "batch_resume_handoff": summary.get("batch_resume_handoff"),
        "runs": runs,
        "results": [
            {key: value for key, value in result.items() if key != "run"}
            for result in results
        ],
    }


@router.get("/ingest/pipeline")
async def graphrag_list_ingestion_pipeline_runs(
    limit: int = 20,
    status: str | None = None,
    domain: str | None = None,
):
    """List recent tracked GraphRAG ingestion pipeline runs without bulky samples."""
    _load_persisted_pipeline_runs()
    bounded_limit = max(1, min(limit, _MAX_PIPELINE_RUNS))
    runs = list(_PIPELINE_RUNS.values())
    if status:
        runs = [run for run in runs if str(run.get("status", "")).lower() == status.lower()]
    if domain:
        runs = [
            run for run in runs
            if str((run.get("document", {}) or {}).get("domain", "")).lower() == domain.lower()
        ]
    runs = sorted(
        runs,
        key=lambda run: str(run.get("updated_at") or run.get("created_at") or ""),
        reverse=True,
    )[:bounded_limit]

    return {
        "runs": [_pipeline_run_summary(run) for run in runs],
        "total_runs": len(_PIPELINE_RUNS),
        "returned": len(runs),
        "limit": bounded_limit,
        "filters": {
            "status": status,
            "domain": domain,
        },
    }


@router.get("/ingest/pipeline/{run_id}")
async def graphrag_ingest_pipeline_status(run_id: str):
    """Return a remembered pipeline preview/upload run."""
    run = _get_pipeline_run_or_404(run_id)
    return {
        **run,
        "artifacts": _pipeline_artifact_manifest(run),
        "resume_handoff": _pipeline_resume_handoff(run),
    }


@router.get("/ingest/pipeline/{run_id}/artifacts")
async def graphrag_ingest_pipeline_artifacts(run_id: str):
    """Return the exportable artifact manifest for a remembered pipeline run."""
    run = _get_pipeline_run_or_404(run_id)
    return _pipeline_artifact_manifest(run)


@router.get("/ingest/pipeline/{run_id}/artifacts/{artifact_kind}")
async def graphrag_ingest_pipeline_artifact(run_id: str, artifact_kind: str, download: bool = False):
    """Export a structured artifact from a remembered pipeline run."""
    run = _get_pipeline_run_or_404(run_id)
    payload, content_type, filename = _pipeline_artifact_payload(run, artifact_kind)
    if isinstance(payload, str):
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'} if download else None
        return Response(content=payload, media_type=content_type, headers=headers)
    if isinstance(payload, bytes):
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'} if download else None
        return Response(content=payload, media_type=content_type, headers=headers)
    if download:
        content = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    return payload


@router.post("/rebuild")
async def graphrag_rebuild_kg(request: Request):
    """Rebuild knowledge graph from existing documents."""
    check_graphrag_enabled()

    try:
        data = await request.json()
    except Exception:
        data = {}

    domain = data.get("domain")

    try:
        client = get_http_client()
        payload = {}
        if domain:
            payload["domain"] = domain

        response = await client.post(
            f"{_graphrag_url()}/rebuild-knowledge-graph",
            params=payload,
            timeout=600.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/entity/link")
async def graphrag_link_entities(request: Request):
    """Link entities to knowledge graph."""
    check_graphrag_enabled()

    data = await request.json()
    entities = data.get("entities", [])

    if not entities:
        raise HTTPException(status_code=400, detail="entities list is required")

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/entity/link",
            json=entities,
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()
        if "linked_entities" not in result and "links" in result:
            result["linked_entities"] = result["links"]
        return result
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/entity/disambiguate")
async def graphrag_disambiguate_entities(request: Request):
    """Disambiguate entity mentions."""
    check_graphrag_enabled()

    data = await request.json()
    entity = data.get("entity")
    entity_name = data.get("entity_name") or (entity or {}).get("name")

    if not entity_name:
        raise HTTPException(status_code=400, detail="entity or entity_name is required")

    payload = {
        "entity": entity or {
            "name": entity_name,
            "type": data.get("type", data.get("entity_type", "unknown")),
            "description": data.get("description", ""),
            "confidence": data.get("confidence", 1.0),
            "metadata": data.get("metadata", {}),
        },
        "context": data.get("context", ""),
    }

    try:
        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/entity/disambiguate",
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.post("/hybrid/process")
@router.post("/ingest/hybrid")
async def graphrag_hybrid_process(
    file: UploadFile = File(...),
    domain: str = Form("general"),
):
    """Process file with hybrid routing (code to Code RAG, docs to GraphRAG)."""
    check_graphrag_enabled()

    try:
        file_content = await file.read()

        client = get_http_client()
        response = await client.post(
            f"{_graphrag_url()}/hybrid/process",
            files={"file": (file.filename, file_content, file.content_type or "application/octet-stream")},
            data={"domain": domain},
            timeout=300.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")
    except Exception as e:
        logger.error(f"Hybrid processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
