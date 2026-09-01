#!/usr/bin/env python3
"""Browser smoke for the GraphRAG document workflow.

Exercises the real UI path:
  workbench -> optional upload -> ask document chat -> review evidence drawer.

The script intentionally saves screenshots at each checkpoint. It is meant to
make the visual review loop repeatable while keeping assertions focused on the
user-facing workflow.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except Exception as exc:  # pragma: no cover - developer environment guard
    print(f"Playwright is required for UI smoke tests: {exc}", file=sys.stderr)
    sys.exit(2)


DEFAULT_QUESTION = (
    "Show the chart, table, form field, styled text, and connector evidence "
    "extracted from this PowerPoint."
)

DEFAULT_STAGE_SAMPLES = (
    "structure_extraction.elements",
    "chunking.chunks",
    "graph_candidates.relationships",
    "embedding_plan.embedding_inputs",
    "local_graph_artifact.edges",
    "local_graph_reasoning_index.path_candidates",
    "local_context_cache.chunks",
    "graph_interaction_plan.starter_prompts",
)

DEFAULT_ARTIFACTS = (
    "source_document",
    "structured_ingest",
    "local_graph",
    "local_context",
    "local_graph_reasoning",
    "interaction_handoff",
    "samples",
)

EVIDENCE_FAMILY_ALIASES = {
    "charts": {"chart", "charts", "chart_fact", "chart_facts"},
    "connectors": {"connector", "connectors", "semantic_connector", "semantic_connectors"},
    "form_fields": {"form", "forms", "form_field", "form_fields"},
    "layout": {
        "layout",
        "layout_relationship",
        "layout_relationships",
        "visual_layout",
        "visual_relationship",
        "visual_relationships",
        "visually_near",
    },
    "speaker_notes": {"speaker_note", "speaker_notes"},
    "styled_text": {"styled_text", "styled_text_blocks", "styled text"},
    "tables": {"table", "tables", "table_fact", "table_facts", "table_elements"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("GRAPHRAG_UI_BASE_URL", "http://localhost:55846"),
        help="Frontend base URL. Default: %(default)s",
    )
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("GRAPHRAG_BACKEND_URL", "http://127.0.0.1:8799"),
        help="Backend base URL for readiness checks. Default: %(default)s",
    )
    parser.add_argument(
        "--file",
        type=Path,
        action="append",
        default=[],
        help="Optional document to upload through the workbench before asking. Repeat for batch upload.",
    )
    parser.add_argument(
        "--question",
        default=os.environ.get("GRAPHRAG_UI_SMOKE_QUESTION", DEFAULT_QUESTION),
        help="Question to send through the selected document chat input.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("GRAPHRAG_UI_SMOKE_OUT", "/tmp/graphrag-ui-smoke")),
        help="Directory for screenshots and summary JSON. Default: %(default)s",
    )
    parser.add_argument(
        "--expect-evidence-family",
        action="append",
        default=[],
        metavar="LABEL",
        help=(
            "Expected extracted evidence family in uploaded pipeline details. "
            "For batch uploads, the family may appear in any uploaded run. May be repeated."
        ),
    )
    parser.add_argument(
        "--expect-visible-evidence-family",
        action="append",
        default=[],
        metavar="LABEL",
        help=(
            "Expected evidence family label visible in the rendered chat answer and "
            "reference drawer. Use this only when the question should literally surface "
            "the family label. May be repeated."
        ),
    )
    parser.add_argument(
        "--expect-workbench-evidence-text",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Expected text visible in the workbench Evidence tab after upload, "
            "such as charts, tables, forms, or extraction sample labels. May be repeated."
        ),
    )
    parser.add_argument(
        "--expect-workbench-evidence-sample-label",
        action="append",
        default=[],
        metavar="LABEL",
        help=(
            "Expected visible Evidence example card label after upload. Repeated "
            "labels must appear in the visible card order."
        ),
    )
    parser.add_argument(
        "--expect-workbench-source-detail-text",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Expected text visible only after expanding the workbench Evidence tab's "
            "Source details disclosure. May be repeated."
        ),
    )
    parser.add_argument(
        "--expect-workbench-processing-text",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Expected text visible in the workbench Processing tab after upload, "
            "such as stage names, readiness counts, sample titles, or best evidence. May be repeated."
        ),
    )
    parser.add_argument(
        "--expect-workbench-output-text",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Expected text visible after expanding the Processing tab's pipeline outputs, "
            "such as rich evidence labels, sample output titles, relationships, or path cards. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--reject-workbench-text",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Text that must not appear while inspecting the workbench Review, "
            "Evidence, or Pipeline tabs after upload. May be repeated."
        ),
    )
    parser.add_argument(
        "--expect-workbench-ask-first",
        action="store_true",
        help=(
            "On the initial workbench view, assert the selected document ask "
            "control is visually above the upload dropzone."
        ),
    )
    parser.add_argument(
        "--expect-workbench-extraction-evidence-text",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Expected text visible after expanding the workbench Evidence tab's "
            "Extraction evidence drilldown. May be repeated."
        ),
    )
    parser.add_argument(
        "--expect-drawer-reference",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Expected reference row that must be selectable in the evidence drawer. "
            "The selected evidence panel must update to this row. May be repeated."
        ),
    )
    parser.add_argument(
        "--expect-drawer-source",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Expected document/source label in the evidence drawer source selector. "
            "Useful for asserting mobile drawer hierarchy. May be repeated."
        ),
    )
    parser.add_argument(
        "--expect-compact-drawer-search-hidden",
        action="store_true",
        help=(
            "After opening the evidence drawer, assert the compact drawer does not "
            "show the Find source search field."
        ),
    )
    parser.add_argument(
        "--expect-source-text",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Expected text inside the expanded Full document text panel after "
            "reference selection. May be repeated."
        ),
    )
    parser.add_argument(
        "--expect-graph-path",
        action="append",
        default=[],
        metavar="NODE -> NODE",
        help=(
            "Expected ordered graph path in the chat answer and selected evidence drawer, "
            "for example: Product Revenue -> Regional Demand -> Risk Map. May be repeated."
        ),
    )
    parser.add_argument(
        "--expect-answer-text",
        action="append",
        default=[],
        metavar="TEXT",
        help="Expected text in the rendered chat answer page. May be repeated.",
    )
    parser.add_argument(
        "--open-answer-reference-card",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Open a visible answer reference tray card whose text/title matches TEXT, "
            "then assert the evidence drawer focuses that card. May be repeated."
        ),
    )
    parser.add_argument(
        "--open-selected-evidence-details",
        action="store_true",
        help=(
            "After opening the evidence drawer, expand the selected evidence details "
            "section and capture/assert its contents."
        ),
    )
    parser.add_argument(
        "--expect-evidence-detail-text",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Expected text inside the expanded selected evidence details section. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--expect-uploaded-filenames-in-answer",
        action="store_true",
        help="Require every uploaded file basename to appear in the rendered chat answer page.",
    )
    parser.add_argument(
        "--use-first-starter-prompt",
        action="store_true",
        help=(
            "Click the first visible workbench starter prompt instead of typing "
            "the --question into the selected-document ask field."
        ),
    )
    parser.add_argument(
        "--starter-prompt-pattern",
        default="",
        metavar="REGEX",
        help=(
            "Click the first visible workbench starter prompt whose label/text matches "
            "this case-insensitive regex. Implies --use-first-starter-prompt."
        ),
    )
    parser.add_argument(
        "--ask-workbench-evidence-sample",
        default="",
        metavar="REGEX",
        help=(
            "Click the Ask control on the first visible Evidence example card whose "
            "label, title, or text matches this case-insensitive regex."
        ),
    )
    parser.add_argument(
        "--ask-workbench-extraction-evidence-sample",
        default="",
        metavar="REGEX",
        help=(
            "Expand Extraction evidence and click Ask on the first full evidence "
            "sample row whose label, title, or text matches this case-insensitive regex."
        ),
    )
    parser.add_argument(
        "--reject-answer-text",
        action="append",
        default=[],
        metavar="TEXT",
        help="Text that must not appear in the rendered chat answer page. May be repeated.",
    )
    parser.add_argument(
        "--reject-drawer-text",
        action="append",
        default=[],
        metavar="TEXT",
        help="Text that must not appear in the opened evidence drawer. May be repeated.",
    )
    parser.add_argument(
        "--expect-pdf-render",
        action="store_true",
        help="Assert that the evidence drawer renders a PDF canvas with highlights.",
    )
    parser.add_argument(
        "--expect-stage-sample",
        action="append",
        default=[],
        metavar="STAGE.KEY",
        help=(
            "Expected non-empty pipeline stage sample for uploaded runs. "
            "Defaults cover extraction/chunking/graph/embedding/chat-planning."
        ),
    )
    parser.add_argument(
        "--expect-artifact",
        action="append",
        default=[],
        metavar="KIND",
        help=(
            "Expected available pipeline artifact for uploaded runs. "
            "Defaults include source_document, structured_ingest, graph/context, and samples."
        ),
    )
    parser.add_argument(
        "--min-chunks",
        type=int,
        default=int(os.environ.get("GRAPHRAG_UI_SMOKE_MIN_CHUNKS", "1")),
        help="Minimum uploaded pipeline chunk count. Default: %(default)s",
    )
    parser.add_argument(
        "--min-relationships",
        type=int,
        default=int(os.environ.get("GRAPHRAG_UI_SMOKE_MIN_RELATIONSHIPS", "1")),
        help="Minimum uploaded pipeline relationship count. Default: %(default)s",
    )
    parser.add_argument(
        "--min-path-candidates",
        type=int,
        default=int(os.environ.get("GRAPHRAG_UI_SMOKE_MIN_PATH_CANDIDATES", "1")),
        help="Minimum uploaded local graph path candidate count. Default: %(default)s",
    )
    parser.add_argument(
        "--no-deep-pipeline-checks",
        action="store_true",
        help="Skip backend run-detail assertions after UI upload.",
    )
    parser.add_argument(
        "--skip-workbench-tab-inspection",
        action="store_true",
        help="Do not capture Review/Evidence/Processing workbench tab screenshots before chat.",
    )
    parser.add_argument(
        "--skip-layout-checks",
        action="store_true",
        help="Do not fail when the viewport has horizontal overflow or clipped visible content.",
    )
    parser.add_argument(
        "--simulate-upload-network-error",
        action="store_true",
        help=(
            "Abort the GraphRAG pipeline upload request in the browser and assert the "
            "workbench shows an actionable backend-unreachable error."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("GRAPHRAG_UI_SMOKE_TIMEOUT", "90")),
        help="Timeout in seconds for upload/chat waits. Default: %(default)s",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run the selected browser headed instead of headless.",
    )
    parser.add_argument(
        "--browser",
        choices=("auto", "chromium", "firefox", "webkit"),
        default=os.environ.get("GRAPHRAG_UI_SMOKE_BROWSER", "auto"),
        help=(
            "Playwright browser to use. 'auto' tries --browser-fallbacks in order. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--browser-fallbacks",
        default=os.environ.get(
            "GRAPHRAG_UI_SMOKE_BROWSER_FALLBACKS",
            "firefox,chromium,webkit",
        ),
        help=(
            "Comma-separated browser order when --browser=auto. Firefox is first "
            "because it is currently the most reliable renderer for local visual "
            "GraphRAG smoke captures. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--browser-launch-timeout",
        type=int,
        default=int(os.environ.get("GRAPHRAG_UI_SMOKE_BROWSER_LAUNCH_TIMEOUT", "20")),
        help="Browser launch timeout in seconds. Default: %(default)s",
    )
    parser.add_argument(
        "--skip-browser-close",
        action="store_true",
        default=os.environ.get("GRAPHRAG_UI_SMOKE_SKIP_BROWSER_CLOSE", "").lower()
        in {"1", "true", "yes"},
        help=(
            "Write the summary and exit immediately without Playwright browser cleanup. "
            "Useful on hosts where browser.close() hangs after screenshots are saved."
        ),
    )
    parser.add_argument(
        "--viewport-width",
        type=int,
        default=int(os.environ.get("GRAPHRAG_UI_SMOKE_VIEWPORT_WIDTH", "1440")),
        help="Browser viewport width in CSS pixels. Default: %(default)s",
    )
    parser.add_argument(
        "--viewport-height",
        type=int,
        default=int(os.environ.get("GRAPHRAG_UI_SMOKE_VIEWPORT_HEIGHT", "1050")),
        help="Browser viewport height in CSS pixels. Default: %(default)s",
    )
    parser.add_argument(
        "--device-scale-factor",
        type=float,
        default=float(os.environ.get("GRAPHRAG_UI_SMOKE_DEVICE_SCALE", "1")),
        help="Browser device scale factor. Default: %(default)s",
    )
    return parser.parse_args()


def normalize_url(value: str) -> str:
    return value.rstrip("/")


def read_json(url: str, timeout: int = 15) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def nested_value(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def value_count(value: Any) -> int:
    if value in (None, "", [], {}):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0)
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, dict):
        count_keys = ("count", "total", "length", "size")
        for key in count_keys:
            if key in value:
                count = positive_int(value.get(key))
                if count > 0:
                    return count
        return len(value)
    return 1


def normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def family_aliases(label: str) -> set[str]:
    key = normalized_key(label)
    aliases = {key, key.rstrip("s")}
    for canonical, values in EVIDENCE_FAMILY_ALIASES.items():
        normalized_values = {normalized_key(value) for value in values}
        if key == canonical or key in normalized_values:
            aliases.update(normalized_values)
            aliases.add(canonical)
    return {alias for alias in aliases if alias}


def summarize_pipeline_detail(detail: dict[str, Any]) -> dict[str, Any]:
    summary = detail.get("summary") if isinstance(detail.get("summary"), dict) else {}
    local_graph = detail.get("local_graph") if isinstance(detail.get("local_graph"), dict) else {}
    local_context = summary.get("local_context") if isinstance(summary.get("local_context"), dict) else {}
    local_reasoning = summary.get("local_graph_reasoning") if isinstance(summary.get("local_graph_reasoning"), dict) else {}
    return {
        "status": detail.get("status"),
        "chunks": positive_int(summary.get("chunk_count") or local_context.get("chunk_count")),
        "relationships": positive_int(
            summary.get("relationship_candidate_count")
            or summary.get("relationship_count")
            or local_graph.get("edge_count")
            or local_graph.get("relationship_count")
        ),
        "path_candidates": positive_int(local_reasoning.get("path_candidate_count")),
        "artifacts": positive_int(nested_value(detail, "artifacts.artifact_count")),
    }


def stage_sample_count(detail: dict[str, Any], spec: str) -> int:
    if "." not in spec:
        raise AssertionError(f"Stage sample spec must be STAGE.KEY: {spec}")
    stage_name, sample_path = spec.split(".", 1)
    stages = detail.get("stages") if isinstance(detail.get("stages"), list) else []
    for stage in stages:
        if isinstance(stage, dict) and stage.get("name") == stage_name:
            return value_count(nested_value(stage.get("sample_outputs"), sample_path))
    return 0


def available_artifact_kinds(detail: dict[str, Any]) -> set[str]:
    artifacts = nested_value(detail, "artifacts.artifacts")
    if not isinstance(artifacts, list):
        return set()
    return {
        str(artifact.get("kind"))
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("available") is True and artifact.get("kind")
    }


def pipeline_family_count(detail: dict[str, Any], label: str) -> int:
    aliases = family_aliases(label)
    summary = detail.get("summary") if isinstance(detail.get("summary"), dict) else {}
    evidence_profile = summary.get("evidence_profile") if isinstance(summary.get("evidence_profile"), dict) else {}
    quick_counts = evidence_profile.get("quick_counts") if isinstance(evidence_profile.get("quick_counts"), dict) else {}
    graph_evidence = evidence_profile.get("graph_evidence") if isinstance(evidence_profile.get("graph_evidence"), dict) else {}
    detected = summary.get("detected_capabilities") if isinstance(summary.get("detected_capabilities"), list) else []

    count = 0
    for capability in detected:
        if normalized_key(str(capability)) in aliases:
            count += 1
    for key, value in quick_counts.items():
        if normalized_key(str(key)) in aliases:
            count += positive_int(value)
    for key, value in graph_evidence.items():
        normalized = normalized_key(str(key))
        if normalized in aliases or normalized.replace("_relationship_count", "") in aliases:
            count += positive_int(value)

    for stage in detail.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        outputs = stage.get("sample_outputs") if isinstance(stage.get("sample_outputs"), dict) else {}
        categories = outputs.get("relationship_sample_categories")
        if isinstance(categories, dict):
            for key, value in categories.items():
                if normalized_key(str(key)) in aliases:
                    count += positive_int(value)

    return count


def assert_pipeline_detail(
    detail: dict[str, Any],
    *,
    expected_stage_samples: list[str],
    expected_artifacts: list[str],
    expected_families: list[str],
    min_chunks: int,
    min_relationships: int,
    min_path_candidates: int,
) -> dict[str, Any]:
    pipeline_summary = summarize_pipeline_detail(detail)
    errors: list[str] = []
    status = str(pipeline_summary.get("status") or "").strip().lower()
    if status in {"submitted", "pending", "running", "processing"}:
        errors.append(
            "pipeline status stayed non-terminal after run detail was available: "
            f"{status}"
        )
    if pipeline_summary["chunks"] < min_chunks:
        errors.append(f"chunks {pipeline_summary['chunks']} < {min_chunks}")
    if pipeline_summary["relationships"] < min_relationships:
        errors.append(f"relationships {pipeline_summary['relationships']} < {min_relationships}")
    if pipeline_summary["path_candidates"] < min_path_candidates:
        errors.append(f"path_candidates {pipeline_summary['path_candidates']} < {min_path_candidates}")

    stage_counts = {
        spec: stage_sample_count(detail, spec)
        for spec in expected_stage_samples
    }
    missing_stage_samples = [spec for spec, count in stage_counts.items() if count < 1]
    if missing_stage_samples:
        errors.append("missing stage samples: " + ", ".join(missing_stage_samples))

    available_artifacts = available_artifact_kinds(detail)
    missing_artifacts = [
        artifact
        for artifact in expected_artifacts
        if artifact not in available_artifacts
    ]
    if missing_artifacts:
        errors.append("missing artifacts: " + ", ".join(missing_artifacts))

    family_counts = {
        label: pipeline_family_count(detail, label)
        for label in expected_families
    }
    missing_families = [label for label, count in family_counts.items() if count < 1]
    if missing_families:
        errors.append("missing extracted evidence families: " + ", ".join(missing_families))

    if errors:
        raise AssertionError("; ".join(errors))

    return {
        **pipeline_summary,
        "stage_sample_counts": stage_counts,
        "artifact_kinds": sorted(available_artifacts),
        "evidence_family_counts": family_counts,
    }


def uploaded_run_ids(payload: dict[str, Any]) -> list[str]:
    run_id = str(payload.get("run_id") or "").strip()
    if run_id:
        return [run_id]
    runs = payload.get("runs") if isinstance(payload.get("runs"), list) else []
    run_ids: list[str] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        candidate = str(run.get("run_id") or "").strip()
        if candidate:
            run_ids.append(candidate)
    return run_ids


def batch_upload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    runs = payload.get("runs") if isinstance(payload.get("runs"), list) else []
    return {
        "status": payload.get("status"),
        "total_files": positive_int(payload.get("total_files") or len(runs)),
        "processed": positive_int(payload.get("processed")),
        "failed": positive_int(payload.get("failed")),
        "peak_concurrency": positive_int(payload.get("peak_concurrency")),
        "duration_ms": positive_int(payload.get("duration_ms")),
    }


def pipeline_run_filename(run: dict[str, Any]) -> str:
    document = run.get("document") if isinstance(run.get("document"), dict) else {}
    return str(document.get("filename") or "").strip()


def pipeline_run_timestamp(run: dict[str, Any]) -> float:
    value = str(run.get("updated_at") or run.get("created_at") or "").strip()
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def matching_recent_pipeline_runs(
    runs: list[dict[str, Any]],
    file_paths: list[Path],
    *,
    upload_started_at: float,
) -> list[dict[str, Any]]:
    wanted = []
    seen: set[str] = set()
    for path in file_paths:
        filename = path.name.strip().lower()
        if filename and filename not in seen:
            wanted.append(filename)
            seen.add(filename)

    if not wanted:
        return []

    complete_statuses = {"complete", "completed", "ready", "success"}
    cutoff = upload_started_at - 1.0
    candidates = sorted(
        [
            run
            for run in runs
            if str(run.get("run_id") or "").strip()
            and pipeline_run_filename(run).lower() in seen
            and str(run.get("status") or "").strip().lower() in complete_statuses
            and pipeline_run_timestamp(run) >= cutoff
        ],
        key=pipeline_run_timestamp,
        reverse=True,
    )
    matched: list[dict[str, Any]] = []
    for filename in wanted:
        candidate = next(
            (
                run
                for run in candidates
                if pipeline_run_filename(run).lower() == filename
                and run not in matched
            ),
            None,
        )
        if candidate:
            matched.append(candidate)

    return matched if len(matched) == len(wanted) else []


def wait_for_recovered_pipeline_runs(
    backend_url: str,
    file_paths: list[Path],
    *,
    upload_started_at: float,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + max(1.0, min(timeout_seconds, 45.0))
    last_error: Exception | None = None

    while time.monotonic() <= deadline:
        try:
            payload = read_json(
                f"{backend_url}/api/v1/graphrag/ingest/pipeline?limit={max(24, len(file_paths) * 8)}",
                timeout=15,
            )
            runs = [
                run
                for run in payload.get("runs", [])
                if isinstance(run, dict)
            ]
            matched = matching_recent_pipeline_runs(
                runs,
                file_paths,
                upload_started_at=upload_started_at,
            )
            if matched:
                return matched
        except Exception as exc:
            last_error = exc

        time.sleep(1.0)

    raise AssertionError(
        "Upload POST did not return and no matching recovered pipeline runs "
        f"became ready within {min(timeout_seconds, 45.0):.0f}s: {last_error}"
    )


def wait_for_pipeline_detail(
    backend_url: str,
    run_id: str,
    *,
    timeout_seconds: float,
    expected_stage_samples: list[str],
    expected_artifacts: list[str],
    expected_families: list[str],
    min_chunks: int,
    min_relationships: int,
    min_path_candidates: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + max(1.0, timeout_seconds)
    last_error: Exception | None = None
    nonterminal_statuses = {"submitted", "pending", "running", "processing"}

    while time.monotonic() <= deadline:
        try:
            detail_payload = read_json(
                f"{backend_url}/api/v1/graphrag/ingest/pipeline/{run_id}",
                timeout=min(max(1.0, timeout_seconds), 15.0),
            )
        except Exception as exc:
            last_error = exc
        else:
            try:
                ready_summary = assert_pipeline_detail(
                    detail_payload,
                    expected_stage_samples=expected_stage_samples,
                    expected_artifacts=expected_artifacts,
                    expected_families=expected_families,
                    min_chunks=min_chunks,
                    min_relationships=min_relationships,
                    min_path_candidates=min_path_candidates,
                )
                return detail_payload, ready_summary
            except AssertionError as exc:
                pipeline_summary = summarize_pipeline_detail(detail_payload)
                status = str(pipeline_summary.get("status") or "").strip().lower()
                if status and status not in nonterminal_statuses:
                    raise AssertionError(
                        f"Pipeline run {run_id} failed readiness checks: {exc}"
                    ) from exc
                last_error = exc

        time.sleep(1.0)

    raise AssertionError(
        f"Pipeline run {run_id} did not become ready within {timeout_seconds:.0f}s: {last_error}"
    )


def service_ready(status: dict[str, Any]) -> bool:
    meta = status.get("_meta") if isinstance(status.get("_meta"), dict) else {}
    if meta.get("ready") is True or meta.get("operational_ready") is True:
        return True

    services = [
        status.get("graphrag"),
        status.get("ner"),
        status.get("rel"),
    ]
    return all(
        isinstance(service, dict) and str(service.get("status", "")).lower() == "healthy"
        for service in services
    )


def screenshot(page: Any, out_dir: Path, name: str) -> Path:
    path = out_dir / f"{name}.png"
    try:
        page.mouse.move(8, 8)
        page.evaluate(
            """
            () => {
              if (document.getElementById('graphrag-smoke-hide-tooltips')) return;
              const style = document.createElement('style');
              style.id = 'graphrag-smoke-hide-tooltips';
              style.textContent = '[role="tooltip"] { opacity: 0 !important; pointer-events: none !important; }';
              document.head.appendChild(style);
            }
            """
        )
        page.wait_for_timeout(350)
    except Exception:
        pass
    page.screenshot(path=str(path), full_page=False)
    print(f"screenshot={path}")
    return path


def browser_launch_order(args: argparse.Namespace) -> list[str]:
    if args.browser != "auto":
        return [args.browser]

    order: list[str] = []
    for item in str(args.browser_fallbacks or "").split(","):
        name = item.strip().lower()
        if name in {"chromium", "firefox", "webkit"} and name not in order:
            order.append(name)
    return order or ["firefox", "chromium", "webkit"]


def launch_browser(playwright: Any, args: argparse.Namespace) -> tuple[str, Any]:
    errors: list[str] = []
    for browser_name in browser_launch_order(args):
        browser_type = getattr(playwright, browser_name)
        try:
            browser = browser_type.launch(
                headless=not args.headed,
                timeout=max(1, args.browser_launch_timeout) * 1000,
            )
            print(f"browser={browser_name}")
            return browser_name, browser
        except Exception as exc:
            errors.append(f"{browser_name}: {exc}")

    raise RuntimeError("Unable to launch a Playwright browser. " + " | ".join(errors))


PLAYWRIGHT_BROWSER_PROCESS_MARKERS = (
    "/ms-playwright/",
    "chromium_headless_shell",
    "chrome-headless-shell",
    "playwright/driver/package/cli.js run-driver",
)


def playwright_browser_processes() -> set[int]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return set()

    current_pid = os.getpid()
    pids: set[int] = set()
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == current_pid:
            continue
        command = parts[1]
        if any(marker in command for marker in PLAYWRIGHT_BROWSER_PROCESS_MARKERS):
            pids.add(pid)
    return pids


def terminate_processes(pids: set[int]) -> None:
    if not pids:
        return

    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in sorted(pids):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                continue
            except PermissionError:
                continue
        time.sleep(0.25)


def write_summary(out_dir: Path, summary: dict[str, Any], *, stderr: bool = False) -> None:
    payload = json.dumps(summary, indent=2)
    (out_dir / "summary.json").write_text(payload, encoding="utf-8")
    print(payload, file=sys.stderr if stderr else sys.stdout)


def fast_exit(exit_code: int) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def viewport_layout_report(page: Any) -> dict[str, Any]:
    payload = page.evaluate(
        """
        () => {
          const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 0;
          const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
          const scrollWidth = Math.max(
            document.documentElement.scrollWidth || 0,
            document.body?.scrollWidth || 0
          );
          const offenders = [];
          const nodes = Array.from(document.body.querySelectorAll('*'));

          for (const element of nodes) {
            const style = window.getComputedStyle(element);
            if (
              style.display === 'none' ||
              style.visibility === 'hidden' ||
              Number(style.opacity || '1') === 0
            ) continue;

            const rect = element.getBoundingClientRect();
            if (rect.width < 24 || rect.height < 12) continue;
            if (rect.bottom < 0 || rect.top > viewportHeight) continue;

            let insideHorizontalScrollContainer = ['auto', 'scroll'].includes(style.overflowX);
            let ancestor = element.parentElement;
            while (!insideHorizontalScrollContainer && ancestor && ancestor !== document.body) {
              const ancestorStyle = window.getComputedStyle(ancestor);
              const ancestorRect = ancestor.getBoundingClientRect();
              insideHorizontalScrollContainer =
                ['auto', 'scroll'].includes(ancestorStyle.overflowX) &&
                ancestorRect.width < ancestor.scrollWidth;
              ancestor = ancestor.parentElement;
            }
            if (insideHorizontalScrollContainer) continue;

            const clippedLeft = rect.left < -4;
            const clippedRight = rect.right > viewportWidth + 4;
            if (!clippedLeft && !clippedRight) continue;

            const text = (
              element.innerText ||
              element.getAttribute('aria-label') ||
              element.getAttribute('title') ||
              ''
            ).replace(/\\s+/g, ' ').trim();

            offenders.push({
              tag: element.tagName.toLowerCase(),
              className: String(element.className || '').slice(0, 120),
              text: text.slice(0, 100),
              left: Math.round(rect.left),
              right: Math.round(rect.right),
              width: Math.round(rect.width),
            });
            if (offenders.length >= 8) break;
          }

          return {
            viewportWidth,
            viewportHeight,
            scrollWidth,
            pageOverflow: Math.max(0, scrollWidth - viewportWidth),
            offenders,
          };
        }
        """
    )
    report = payload if isinstance(payload, dict) else {}
    report.setdefault("viewportWidth", 0)
    report.setdefault("viewportHeight", 0)
    report.setdefault("scrollWidth", 0)
    report.setdefault("pageOverflow", 0)
    report.setdefault("offenders", [])
    return report


def assert_viewport_layout_stable(
    page: Any,
    context: str,
    summary: dict[str, Any],
) -> None:
    report = viewport_layout_report(page)
    report["context"] = context
    checks = summary.setdefault("layout_checks", [])
    if isinstance(checks, list):
        checks.append(report)

    offenders = report.get("offenders") if isinstance(report.get("offenders"), list) else []
    page_overflow = positive_int(report.get("pageOverflow"))
    if page_overflow > 8 or offenders:
        raise AssertionError(
            f"Viewport layout overflow at {context}: "
            f"pageOverflow={page_overflow}, offenders={offenders[:3]}"
        )


def assert_answer_reference_tray_visible(page: Any, summary: dict[str, Any]) -> None:
    payload = page.evaluate(
        """
        () => {
          const viewportWidth = window.innerWidth;
          const viewportHeight = window.innerHeight;
          const scroller = document.querySelector('[data-chat-messages]');
          const rectPayload = (element) => {
            if (!element) return null;
            const rect = element.getBoundingClientRect();
            return {
              top: Math.round(rect.top),
              bottom: Math.round(rect.bottom),
              left: Math.round(rect.left),
              right: Math.round(rect.right),
              width: Math.round(rect.width),
              height: Math.round(rect.height),
              text: (element.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 120),
            };
          };
          const scrollerRect = rectPayload(scroller) || {
            top: 0,
            bottom: viewportHeight,
            left: 0,
            right: viewportWidth,
            width: viewportWidth,
            height: viewportHeight,
          };
          const visibleHeight = (rect) => rect
            ? Math.max(0, Math.min(rect.bottom, scrollerRect.bottom) - Math.max(rect.top, scrollerRect.top))
            : 0;
          const visibleWidth = (rect) => rect
            ? Math.max(0, Math.min(rect.right, scrollerRect.right) - Math.max(rect.left, scrollerRect.left))
            : 0;
          const intersectsChatPanel = (rect) => Boolean(
            rect &&
            rect.width > 0 &&
            rect.height > 0 &&
            visibleHeight(rect) >= Math.min(40, rect.height * 0.5) &&
            visibleWidth(rect) >= Math.min(80, rect.width * 0.5)
          );
          const intersectsViewport = (rect) => Boolean(
            rect &&
            rect.width > 0 &&
            rect.height > 0 &&
            Math.min(rect.bottom, viewportHeight) - Math.max(rect.top, 0) >= Math.min(40, rect.height * 0.5) &&
            Math.min(rect.right, viewportWidth) - Math.max(rect.left, 0) >= Math.min(80, rect.width * 0.5)
          );
          const tray = document.querySelector('[data-testid="reference-tray"]');
          const review = document.querySelector('[data-testid="reference-tray-review"]');
          const composer = document.querySelector('[data-chat-input]');
          const cards = Array.from(document.querySelectorAll('[data-testid="reference-tray-card"]'));
          const trayRect = rectPayload(tray);
          const reviewRect = rectPayload(review);
          const composerRect = rectPayload(composer);
          const cardRects = cards.map(rectPayload).filter(Boolean);
          const overlaps = (a, b) => Boolean(
            a && b &&
            a.width > 0 &&
            a.height > 0 &&
            b.width > 0 &&
            b.height > 0 &&
            Math.max(a.left, b.left) < Math.min(a.right, b.right) &&
            Math.max(a.top, b.top) < Math.min(a.bottom, b.bottom)
          );
          const visibleCards = cardRects.filter(intersectsChatPanel);
          return {
            viewportWidth,
            viewportHeight,
            chatPanel: scrollerRect,
            tray: trayRect,
            review: reviewRect,
            composer: composerRect,
            cards: cardRects,
            trayDataset: tray ? { ...tray.dataset } : {},
            trayInView: intersectsChatPanel(trayRect),
            reviewInView: intersectsChatPanel(reviewRect),
            composerInViewport: intersectsViewport(composerRect),
            trayOverlapsComposer: overlaps(trayRect, composerRect),
            cardOverlapsComposer: visibleCards.some((rect) => overlaps(rect, composerRect)),
            trayComposerGap: trayRect && composerRect ? Math.round(composerRect.top - trayRect.bottom) : null,
            visibleCardCount: visibleCards.length,
          };
        }
        """
    )
    report = payload if isinstance(payload, dict) else {}
    summary["answer_reference_tray_view"] = report

    if not report.get("trayInView"):
        raise AssertionError(f"Answer reference tray is not visible in the viewport: {report}")
    if not report.get("reviewInView"):
        raise AssertionError(f"Answer reference review action is not visible in the viewport: {report}")
    if positive_int(report.get("visibleCardCount")) < 1:
        raise AssertionError(f"No answer reference cards are visible in the viewport: {report}")
    if report.get("trayOverlapsComposer") or report.get("cardOverlapsComposer"):
        raise AssertionError(f"Answer reference tray overlaps the chat composer: {report}")
    if positive_int(report.get("viewportWidth")) <= 600 and positive_int(report.get("visibleCardCount")) > 4:
        raise AssertionError(f"Mobile answer reference tray shows too many inline cards: {report}")


def install_upload_network_error_route(page: Any) -> None:
    def abort_upload(route: Any) -> None:
        request = route.request
        if request.method == "POST" and "/api/v1/graphrag/ingest/pipeline" in request.url:
            route.abort("failed")
            return
        route.continue_()

    page.route(re.compile(r".*/api/v1/graphrag/ingest/pipeline(?:/.*)?$"), abort_upload)


def wait_for_text(page: Any, pattern: str, timeout_ms: int) -> None:
    page.wait_for_function(
        """pattern => new RegExp(pattern, 'i').test(document.body.innerText || '')""",
        arg=pattern,
        timeout=timeout_ms,
    )


def wait_for_chat_answer_complete(page: Any, timeout_ms: int) -> None:
    wait_for_text(page, r"References|Writing cited answer|Review evidence", timeout_ms)
    page.wait_for_function(
        """
        () => {
          const text = document.body.innerText || '';
          if (/Writing cited answer|Writing\\b|Thinking|Generating|Loading/i.test(text)) {
            return false;
          }
          return /References/i.test(text) || Boolean(
            document.querySelector('[data-testid="reference-tray"]') ||
            document.querySelector('[data-testid="reference-tray-review"]')
          );
        }
        """,
        timeout=timeout_ms,
    )


def click_first_visible_button(page: Any, *names: str, timeout_ms: int = 10_000) -> str:
    for name in names:
        locator = page.get_by_role("button", name=re.compile(name, re.I)).first
        try:
            locator.wait_for(state="visible", timeout=2_000)
            locator.click(timeout=timeout_ms)
            return name
        except PlaywrightTimeoutError:
            continue
    raise AssertionError(f"No visible button matched: {', '.join(names)}")


def click_first_visible_locator(
    locators: list[tuple[str, Any]],
    *,
    timeout_ms: int,
) -> str:
    for label, locator in locators:
        try:
            locator.wait_for(state="visible", timeout=2_000)
            locator.click(timeout=timeout_ms)
            return label
        except PlaywrightTimeoutError:
            continue
    raise AssertionError(
        "No visible locator matched: " + ", ".join(label for label, _ in locators)
    )


def body_text(page: Any) -> str:
    return page.locator("body").inner_text(timeout=10_000)


def assert_text_contains(text: str, label: str, *, context: str = "evidence family label") -> None:
    if re.search(re.escape(label), text, re.I):
        return
    raise AssertionError(f"Expected to find {context}: {label}")


def normalized_visible_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def graph_path_nodes(spec: str) -> list[str]:
    separators = r"\s*(?:->|→|=>|\||/)\s*"
    nodes = [
        part.strip()
        for part in re.split(separators, spec)
        if part.strip()
    ]
    if len(nodes) < 2:
        raise AssertionError(f"Graph path expectation needs at least two nodes: {spec}")
    return nodes


def assert_ordered_labels_visible(text: str, labels: list[str], *, context: str) -> None:
    normalized_text = normalized_visible_text(text)
    cursor = 0
    missing: list[str] = []
    for label in labels:
        normalized_label = normalized_visible_text(label)
        if not normalized_label:
            continue
        index = normalized_text.find(normalized_label, cursor)
        if index < 0:
            missing.append(label)
            break
        cursor = index + len(normalized_label)
    if missing:
        raise AssertionError(
            f"Expected ordered labels in {context}: {' -> '.join(labels)}"
        )


def assert_graph_path_visible(text: str, spec: str, *, context: str) -> list[str]:
    nodes = graph_path_nodes(spec)
    assert_ordered_labels_visible(text, nodes, context=context)
    return nodes


def assert_evidence_family_visible(text: str, label: str) -> None:
    normalized_text = normalized_key(text)
    if any(alias in normalized_text for alias in family_aliases(label)):
        return
    raise AssertionError(f"Expected to find evidence family label: {label}")


def assert_text_missing(text: str, label: str, *, context: str = "text") -> None:
    match = re.search(re.escape(label), text, re.I)
    if not match:
        return
    start = max(0, match.start() - 180)
    end = min(len(text), match.end() + 180)
    snippet = re.sub(r"\s+", " ", text[start:end]).strip()
    raise AssertionError(
        f"Expected not to find {context}: {label}. "
        f"Matched context: {snippet}"
    )


def open_answer_reference_card(
    page: Any,
    label: str,
    out_dir: Path,
    screenshots: list[str],
    *,
    timeout_ms: int,
) -> dict[str, Any]:
    """Open a specific answer reference card and assert the drawer focuses it."""
    cards = page.locator('[data-testid="reference-tray-card"]')
    matching_card = None
    matching_index = -1
    matching_text = ""

    for index in range(cards.count()):
        card = cards.nth(index)
        try:
            if not card.is_visible(timeout=500):
                continue
            card_text = "\n".join(
                [
                    card.get_attribute("data-reference-title") or "",
                    card.inner_text(timeout=1_000).strip(),
                ]
            ).strip()
            if re.search(re.escape(label), card_text, re.I):
                matching_card = card
                matching_index = index
                matching_text = card_text
                break
        except PlaywrightTimeoutError:
            continue

    if matching_card is None:
        visible_cards = []
        for index in range(cards.count()):
            candidate = cards.nth(index)
            try:
                if candidate.is_visible(timeout=500):
                    visible_cards.append(candidate.inner_text(timeout=1_000).strip())
            except PlaywrightTimeoutError:
                continue
        raise AssertionError(
            f"Expected visible answer reference card matching {label!r}. "
            f"Visible cards: {visible_cards[:8]}"
        )

    matching_card.scroll_into_view_if_needed(timeout=timeout_ms)
    matching_card.click(timeout=timeout_ms)
    selected_panel = page.locator('[data-testid="reference-selected-evidence"]').first
    selected_panel.wait_for(state="visible", timeout=timeout_ms)
    page.wait_for_function(
        """
        ({ aliases }) => {
          const panel = document.querySelector('[data-testid="reference-selected-evidence"]');
          if (!panel) return false;
          const normalized = (panel.textContent || '')
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '_')
            .replace(/^_+|_+$/g, '');
          return aliases.some((alias) => normalized.includes(alias));
        }
        """,
        arg={"aliases": sorted(family_aliases(label))},
        timeout=timeout_ms,
    )
    page.wait_for_timeout(450)
    selected_text = selected_panel.inner_text(timeout=timeout_ms).strip()
    screenshots.append(
        str(screenshot(page, out_dir, f"06-evidence-drawer-card-{normalized_key(label).replace('_', '-')}"))
    )
    return {
        "label": label,
        "card_index": matching_index,
        "card_text": matching_text,
        "selected_text": re.sub(r"\s+", " ", selected_text)[:360],
    }


def expand_selected_evidence_details(
    page: Any,
    label: str,
    out_dir: Path,
    screenshots: list[str],
    expected_terms: list[str],
    *,
    timeout_ms: int,
) -> dict[str, Any]:
    details = page.locator("details").filter(has_text=re.compile(r"Evidence details", re.I)).first
    try:
        details.wait_for(state="visible", timeout=timeout_ms)
    except PlaywrightTimeoutError as exc:
        raise AssertionError("Selected evidence drawer did not expose Evidence details.") from exc

    is_open = details.evaluate("(node) => Boolean(node.open)")
    if not is_open:
        details.locator("summary").first.click(timeout=timeout_ms)
    page.wait_for_timeout(350)
    detail_text = details.inner_text(timeout=timeout_ms).strip()
    for expected in expected_terms:
        assert_text_contains(
            detail_text,
            expected,
            context="expanded selected evidence details",
        )

    screenshot_label = label or "selected"
    screenshots.append(
        str(screenshot(page, out_dir, f"06-evidence-details-{normalized_key(screenshot_label).replace('_', '-')}"))
    )
    return {
        "label": label,
        "expected_terms_verified": expected_terms,
        "detail_text": re.sub(r"\s+", " ", detail_text)[:520],
    }


def assert_pdf_canvas_rendered(page: Any, timeout_ms: int) -> dict[str, Any]:
    wait_for_text(page, r"Page\s+\d+\s*/\s*\d+|highlights?", timeout_ms)
    result = page.wait_for_function(
        """
        () => {
          const canvases = Array.from(document.querySelectorAll('canvas'))
            .filter(canvas => canvas.width > 160 && canvas.height > 160);
          for (const canvas of canvases) {
            const context = canvas.getContext('2d', { willReadFrequently: true });
            if (!context) continue;
            const width = canvas.width;
            const height = canvas.height;
            const stepX = Math.max(1, Math.floor(width / 80));
            const stepY = Math.max(1, Math.floor(height / 80));
            let sampled = 0;
            let inkPixels = 0;
            for (let y = 0; y < height; y += stepY) {
              for (let x = 0; x < width; x += stepX) {
                const [red, green, blue, alpha] = context.getImageData(x, y, 1, 1).data;
                sampled += 1;
                if (alpha > 0 && (red < 245 || green < 245 || blue < 245)) {
                  inkPixels += 1;
                }
              }
            }
            if (inkPixels > 0) {
              const highlightCount = document.body.innerText.match(/\\b\\d+\\s+highlights?\\b/i)?.[0] || '';
              return { width, height, sampled, inkPixels, highlightCount };
            }
          }
          return false;
        }
        """,
        timeout=timeout_ms,
    )
    payload = result.json_value()
    if not isinstance(payload, dict) or int(payload.get("inkPixels") or 0) < 1:
        raise AssertionError("PDF canvas did not render visible page content.")
    page_text = body_text(page)
    if not re.search(r"\b\d+\s+highlights?\b", page_text, re.I) and not re.search(r"\bFile-level evidence\b", page_text, re.I):
        raise AssertionError("PDF drawer did not report highlighted or file-level evidence.")
    return payload


def assert_reference_drawer_hierarchy(
    page: Any,
    summary: dict[str, Any],
    expected_sources: list[str],
    *,
    timeout_ms: int,
) -> None:
    """Assert the compact evidence drawer keeps source navigation separate from citations."""
    if int(summary.get("viewport", {}).get("width") or 0) >= 600:
        return

    source_selector = page.locator('[data-testid="reference-mobile-source-selector"]').first
    evidence_list = page.locator('[data-testid="reference-mobile-evidence-list"]').first
    expects_multiple_sources = len({label for label in expected_sources if label}) > 1

    try:
        evidence_list.wait_for(state="visible", timeout=timeout_ms)
    except PlaywrightTimeoutError as exc:
        raise AssertionError(
            "Mobile evidence drawer did not expose a separate Evidence section."
        ) from exc

    source_text = ""
    source_selector_present = False
    try:
        source_selector.wait_for(
            state="visible",
            timeout=timeout_ms if expects_multiple_sources else 1_500,
        )
        source_selector_present = True
        source_text = source_selector.inner_text(timeout=timeout_ms).strip()
    except PlaywrightTimeoutError as exc:
        if expects_multiple_sources:
            raise AssertionError(
                "Mobile evidence drawer did not expose separate Documents and Evidence sections."
            ) from exc

    evidence_text = evidence_list.inner_text(timeout=timeout_ms).strip()
    if source_selector_present:
        if not re.search(r"\bDocuments\b", source_text, re.I):
            raise AssertionError(f"Mobile source selector is missing a Documents label: {source_text!r}")
        if not re.search(r"\bAll documents\b", source_text, re.I):
            raise AssertionError(f"Mobile source selector is missing All documents: {source_text!r}")
    if not re.search(r"\bEvidence\b", evidence_text, re.I):
        raise AssertionError(f"Mobile evidence list is missing an Evidence label: {evidence_text!r}")

    drawer_text = body_text(page)
    for label in expected_sources:
        assert_text_contains(
            source_text if source_selector_present else drawer_text,
            label,
            context="mobile drawer source selector",
        )

    visible_evidence_rows = 0
    row_locator = evidence_list.locator(
        '[data-testid="reference-drawer-row"], [data-testid="reference-drawer-row-pdf"]'
    )
    for index in range(row_locator.count()):
        try:
            if row_locator.nth(index).is_visible(timeout=500):
                visible_evidence_rows += 1
        except PlaywrightTimeoutError:
            continue
    if visible_evidence_rows < 1:
        raise AssertionError("Mobile evidence drawer did not expose visible citation rows.")

    summary["mobile_reference_drawer_hierarchy"] = {
        "source_selector_text": source_text,
        "source_selector_present": source_selector_present,
        "evidence_heading_present": True,
        "visible_evidence_rows": visible_evidence_rows,
        "expected_sources_verified": expected_sources,
    }


def select_pdf_reference_for_render(page: Any, timeout_ms: int) -> str:
    """Select a PDF reference in the open drawer before asserting the canvas."""
    try:
        if re.search(r"Page\s+\d+\s*/\s*\d+|\b\d+\s+highlights?\b", body_text(page), re.I):
            if page.locator("canvas").count() > 0:
                return "already-selected"
    except Exception:
        pass

    pdf_rows = page.locator('[data-testid="reference-drawer-row-pdf"]')
    if pdf_rows.count() == 0:
        show_more = page.get_by_role(
            "button",
            name=re.compile(r"Show\s+\d+\s+more(?:\s+references)?", re.I),
        ).first
        if show_more.count() > 0:
            try:
                show_more.click(timeout=5_000)
                page.wait_for_timeout(250)
            except PlaywrightTimeoutError:
                pass

    pdf_rows = page.locator('[data-testid="reference-drawer-row-pdf"]')
    for index in range(pdf_rows.count()):
        row = pdf_rows.nth(index)
        try:
            if not row.is_visible(timeout=1_000):
                continue
            row.click(timeout=timeout_ms)
            wait_for_text(page, r"PDF|Page\s+\d+\s*/\s*\d+|highlights?", timeout_ms)
            return f"pdf-row-{index}"
        except PlaywrightTimeoutError:
            continue

    raise AssertionError("Expected a selectable PDF reference in the evidence drawer.")


def expand_drawer_references(page: Any) -> bool:
    show_more = page.get_by_role(
        "button",
        name=re.compile(r"Show\s+\d+\s+more(?:\s+references)?", re.I),
    ).first
    try:
        if show_more.count() > 0 and show_more.is_visible(timeout=1_000):
            show_more.click(timeout=5_000)
            page.wait_for_timeout(250)
            return True
    except PlaywrightTimeoutError:
        return False
    return False


def first_visible_locator(locator: Any, *, timeout_ms: int = 1_000) -> tuple[Any | None, int]:
    for index in range(locator.count()):
        candidate = locator.nth(index)
        try:
            if candidate.is_visible(timeout=timeout_ms):
                return candidate, index
        except PlaywrightTimeoutError:
            continue
    return None, -1


def first_visible_bounding_box(locator: Any, *, timeout_ms: int = 1_000) -> tuple[dict[str, float] | None, int]:
    for index in range(locator.count()):
        candidate = locator.nth(index)
        try:
            if not candidate.is_visible(timeout=timeout_ms):
                continue
            box = candidate.bounding_box(timeout=timeout_ms)
            if box:
                return box, index
        except PlaywrightTimeoutError:
            continue
    return None, -1


def assert_workbench_ask_first(page: Any, summary: dict[str, Any]) -> None:
    ask_box, ask_index = first_visible_bounding_box(
        page.locator(
            '[data-testid="graphrag-selected-document-ask-mobile"], '
            '[data-testid="graphrag-selected-document-ask"]'
        ),
        timeout_ms=5_000,
    )
    if not ask_box:
        raise AssertionError("Initial workbench did not expose a selected-document ask control.")

    upload_box, upload_index = first_visible_bounding_box(
        page.locator('[data-testid="graphrag-document-upload-dropzone"]'),
        timeout_ms=1_000,
    )
    summary["workbench_initial_order"] = {
        "ask_index": ask_index,
        "ask_y": round(float(ask_box["y"]), 2),
        "upload_index": upload_index,
        "upload_y": round(float(upload_box["y"]), 2) if upload_box else None,
    }
    if upload_box and ask_box["y"] > upload_box["y"] - 2:
        raise AssertionError(
            "Initial workbench shows upload before selected-document chat: "
            f"ask_y={ask_box['y']:.1f}, upload_y={upload_box['y']:.1f}"
        )


def select_starter_prompt(
    page: Any,
    pattern: str,
    *,
    timeout_ms: int,
) -> tuple[Any, int, str]:
    starter_prompts = page.locator('[data-testid="graphrag-starter-prompt"]')
    starter_prompts.first.wait_for(state="visible", timeout=timeout_ms)

    compiled_pattern = None
    if pattern:
        try:
            compiled_pattern = re.compile(pattern, re.I)
        except re.error as exc:
            raise AssertionError(f"Invalid --starter-prompt-pattern {pattern!r}: {exc}") from exc

    visible_labels: list[str] = []
    for index in range(starter_prompts.count()):
        candidate = starter_prompts.nth(index)
        try:
            if not candidate.is_visible(timeout=1_000):
                continue
            label = (
                candidate.get_attribute("data-prompt-label")
                or candidate.inner_text(timeout=timeout_ms)
                or ""
            ).strip()
            if label:
                visible_labels.append(label)
            if not label:
                continue
            if compiled_pattern is None or compiled_pattern.search(label):
                return candidate, index, label
        except PlaywrightTimeoutError:
            continue

    if compiled_pattern is not None:
        raise AssertionError(
            f"No visible starter prompt matched {pattern!r}. Visible prompts: {visible_labels}"
        )

    raise AssertionError("Starter prompt rendered without a readable visible label.")


def click_workbench_evidence_sample_ask(
    page: Any,
    pattern: str,
    out_dir: Path,
    screenshots: list[str],
    *,
    timeout_ms: int,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_tab = page.get_by_role("tab", name=re.compile(r"Evidence", re.I)).first
    evidence_tab.wait_for(state="visible", timeout=timeout_ms)
    evidence_tab.click(timeout=timeout_ms)
    wait_for_text(page, r"Evidence examples", timeout_ms)

    try:
        compiled_pattern = re.compile(pattern, re.I) if pattern else None
    except re.error as exc:
        raise AssertionError(f"Invalid --ask-workbench-evidence-sample {pattern!r}: {exc}") from exc

    rows = page.locator('[data-testid="graphrag-evidence-example-row"]')
    rows.first.wait_for(state="visible", timeout=timeout_ms)

    visible_samples: list[str] = []
    matching_row = None
    matching_index = -1
    matching_text = ""
    matching_label = ""
    matching_title = ""

    for index in range(rows.count()):
        row = rows.nth(index)
        try:
            if not row.is_visible(timeout=1_000):
                continue
            label = (row.get_attribute("data-evidence-sample-label") or "").strip()
            title = (row.get_attribute("data-evidence-sample-title") or "").strip()
            text = row.inner_text(timeout=timeout_ms).strip()
            candidate_text = "\n".join([label, title, text]).strip()
            if candidate_text:
                visible_samples.append(re.sub(r"\s+", " ", candidate_text)[:220])
            if compiled_pattern is None or compiled_pattern.search(candidate_text):
                matching_row = row
                matching_index = index
                matching_text = text
                matching_label = label
                matching_title = title
                break
        except PlaywrightTimeoutError:
            continue

    if matching_row is None:
        raise AssertionError(
            f"No visible Evidence example matched {pattern!r}. "
            f"Visible samples: {visible_samples[:8]}"
        )

    matching_row.scroll_into_view_if_needed(timeout=timeout_ms)
    page.wait_for_timeout(350)
    screenshots.append(
        str(
            screenshot(
                page,
                out_dir,
                f"04-evidence-sample-ask-ready-{normalized_key(pattern or matching_label).replace('_', '-')}",
            )
        )
    )
    if summary is not None:
        assert_viewport_layout_stable(page, "04-evidence-sample-ask-ready", summary)
    ask_button = matching_row.locator('[data-testid="graphrag-evidence-example-ask"]').first
    ask_button.wait_for(state="visible", timeout=timeout_ms)
    ask_button.click(timeout=timeout_ms)

    return {
        "pattern": pattern,
        "card_index": matching_index,
        "label": matching_label,
        "title": matching_title,
        "card_text": re.sub(r"\s+", " ", matching_text)[:360],
    }


def click_workbench_extraction_evidence_sample_ask(
    page: Any,
    pattern: str,
    out_dir: Path,
    screenshots: list[str],
    *,
    timeout_ms: int,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_tab = page.get_by_role("tab", name=re.compile(r"Evidence", re.I)).first
    evidence_tab.wait_for(state="visible", timeout=timeout_ms)
    evidence_tab.click(timeout=timeout_ms)
    wait_for_text(page, r"Extraction evidence", timeout_ms)

    try:
        compiled_pattern = re.compile(pattern, re.I) if pattern else None
    except re.error as exc:
        raise AssertionError(
            f"Invalid --ask-workbench-extraction-evidence-sample {pattern!r}: {exc}"
        ) from exc

    extraction_details = page.locator('[data-testid="graphrag-extraction-evidence-details"]').first
    extraction_details.wait_for(state="visible", timeout=timeout_ms)
    if not bool(extraction_details.evaluate("(node) => Boolean(node.open)")):
        extraction_details.locator("summary").first.click(timeout=timeout_ms)

    rows = extraction_details.locator('[data-testid="graphrag-extraction-evidence-row"]')
    rows.first.wait_for(state="visible", timeout=timeout_ms)

    visible_samples: list[str] = []
    matching_row = None
    matching_index = -1
    matching_text = ""
    matching_label = ""
    matching_title = ""

    for index in range(rows.count()):
        row = rows.nth(index)
        try:
            if not row.is_visible(timeout=1_000):
                continue
            label = (row.get_attribute("data-evidence-sample-label") or "").strip()
            title = (row.get_attribute("data-evidence-sample-title") or "").strip()
            text = row.inner_text(timeout=timeout_ms).strip()
            candidate_text = "\n".join([label, title, text]).strip()
            if candidate_text:
                visible_samples.append(re.sub(r"\s+", " ", candidate_text)[:220])
            if compiled_pattern is None or compiled_pattern.search(candidate_text):
                matching_row = row
                matching_index = index
                matching_text = text
                matching_label = label
                matching_title = title
                break
        except PlaywrightTimeoutError:
            continue

    if matching_row is None:
        raise AssertionError(
            f"No visible Extraction evidence sample matched {pattern!r}. "
            f"Visible samples: {visible_samples[:8]}"
        )

    matching_row.scroll_into_view_if_needed(timeout=timeout_ms)
    page.wait_for_timeout(350)
    screenshots.append(
        str(
            screenshot(
                page,
                out_dir,
                f"04-extraction-evidence-sample-ask-ready-{normalized_key(pattern or matching_label).replace('_', '-')}",
            )
        )
    )
    if summary is not None:
        assert_viewport_layout_stable(page, "04-extraction-evidence-sample-ask-ready", summary)
    ask_button = matching_row.locator('[data-testid="graphrag-extraction-evidence-ask"]').first
    ask_button.wait_for(state="visible", timeout=timeout_ms)
    ask_button.click(timeout=timeout_ms)

    return {
        "pattern": pattern,
        "row_index": matching_index,
        "label": matching_label,
        "title": matching_title,
        "row_text": re.sub(r"\s+", " ", matching_text)[:360],
    }


def select_drawer_reference(
    page: Any,
    label: str,
    out_dir: Path,
    screenshots: list[str],
    *,
    timeout_ms: int,
) -> dict[str, Any]:
    row_selector = '[data-testid="reference-drawer-row"], [data-testid="reference-drawer-row-pdf"]'
    label_pattern = re.compile(re.escape(label), re.I)
    selection_patterns = [re.escape(label)]
    prose_chain_parts = [
        part.strip()
        for part in re.split(r"\s+to\s+", label, flags=re.I)
        if part.strip()
    ]
    if len(prose_chain_parts) >= 2:
        selection_patterns.append(
            re.escape(prose_chain_parts[0]) +
            r".{0,120}" +
            re.escape(prose_chain_parts[-1])
        )
    rows = page.locator(row_selector).filter(has_text=label_pattern)
    row, index = first_visible_locator(rows)
    if row is None:
        expand_drawer_references(page)
        rows = page.locator(row_selector).filter(has_text=label_pattern)
        row, index = first_visible_locator(rows)
    if row is None:
        visible_rows = []
        all_rows = page.locator(row_selector)
        for row_index in range(all_rows.count()):
            candidate = all_rows.nth(row_index)
            try:
                if candidate.is_visible(timeout=500):
                    visible_rows.append(candidate.inner_text(timeout=1_000).strip())
            except PlaywrightTimeoutError:
                continue
        raise AssertionError(
            f"Expected selectable drawer reference matching {label!r}. "
            f"Visible rows: {visible_rows[:8]}"
        )

    row_text = row.inner_text(timeout=5_000).strip()
    row.click(timeout=timeout_ms)
    selected_panel = page.locator('[data-testid="reference-selected-evidence"]').first
    selected_panel.wait_for(state="visible", timeout=timeout_ms)
    page.wait_for_function(
        """
        ({ patterns }) => {
          const panel = document.querySelector('[data-testid="reference-selected-evidence"]');
          if (!panel) return false;
          const text = panel.textContent || '';
          return patterns.some((pattern) => new RegExp(pattern, 'i').test(text));
        }
        """,
        arg={"patterns": selection_patterns},
        timeout=timeout_ms,
    )
    page.wait_for_timeout(450)
    screenshots.append(
        str(screenshot(page, out_dir, f"06-evidence-drawer-{normalized_key(label).replace('_', '-')}"))
    )
    return {
        "label": label,
        "row_index": index,
        "row_text": row_text,
    }


def select_graph_path_reference(
    page: Any,
    spec: str,
    out_dir: Path,
    screenshots: list[str],
    *,
    timeout_ms: int,
) -> dict[str, Any]:
    nodes = graph_path_nodes(spec)
    row_selector = '[data-testid="reference-drawer-row"], [data-testid="reference-drawer-row-pdf"]'

    def matching_row() -> tuple[Any | None, int, str]:
        rows = page.locator(row_selector)
        for index in range(rows.count()):
            row = rows.nth(index)
            try:
                if not row.is_visible(timeout=500):
                    continue
                row_text = row.inner_text(timeout=1_000).strip()
                title = row.get_attribute("data-reference-title") or ""
                candidate_text = "\n".join([title, row_text]).strip()
                try:
                    assert_ordered_labels_visible(
                        candidate_text,
                        nodes,
                        context=f"drawer row for {spec}",
                    )
                    return row, index, candidate_text
                except AssertionError:
                    continue
            except PlaywrightTimeoutError:
                continue
        return None, -1, ""

    row, index, row_text = matching_row()
    if row is None:
        expand_drawer_references(page)
        row, index, row_text = matching_row()
    if row is None:
        visible_rows = []
        all_rows = page.locator(row_selector)
        for row_index in range(all_rows.count()):
            candidate = all_rows.nth(row_index)
            try:
                if candidate.is_visible(timeout=500):
                    visible_rows.append(
                        "\n".join([
                            candidate.get_attribute("data-reference-title") or "",
                            candidate.inner_text(timeout=1_000).strip(),
                        ]).strip()
                    )
            except PlaywrightTimeoutError:
                continue
        raise AssertionError(
            f"Expected selectable graph path reference matching {spec!r}. "
            f"Visible rows: {visible_rows[:8]}"
        )

    row.click(timeout=timeout_ms)
    selected_panel = page.locator('[data-testid="reference-selected-evidence"]').first
    selected_panel.wait_for(state="visible", timeout=timeout_ms)
    selected_text = selected_panel.inner_text(timeout=timeout_ms)
    assert_ordered_labels_visible(
        selected_text,
        nodes,
        context=f"selected evidence graph path for {spec}",
    )
    page.wait_for_timeout(450)
    screenshots.append(
        str(screenshot(page, out_dir, f"06-evidence-drawer-path-{normalized_key(spec).replace('_', '-')}"))
    )
    return {
        "path": spec,
        "nodes": nodes,
        "row_index": index,
        "row_text": row_text,
    }


def expand_source_text_panel(
    page: Any,
    summary: dict[str, Any],
    expected_terms: list[str],
    *,
    timeout_ms: int,
) -> dict[str, Any] | None:
    show_text = page.get_by_role(
        "button",
        name=re.compile(r"Show (?:full document )?text", re.I),
    ).first
    if show_text.count() <= 0:
        if expected_terms:
            raise AssertionError("Expected Full document text control, but none was present.")
        return None

    show_text.click(timeout=5_000)
    viewer = page.locator('[data-testid="reference-document-text-viewer"]').first
    viewer.wait_for(state="visible", timeout=timeout_ms)
    page.wait_for_timeout(500)
    viewer_text = viewer.inner_text(timeout=timeout_ms)
    for term in expected_terms:
        assert_text_contains(viewer_text, term, context="expanded source text")

    result = {
        "expanded": True,
        "expected_terms_verified": expected_terms,
        "text_preview": re.sub(r"\s+", " ", viewer_text).strip()[:360],
    }
    summary["source_text_expansion"] = result
    return result


def inspect_workbench_tabs(
    page: Any,
    out_dir: Path,
    screenshots: list[str],
    *,
    prefix: str,
    timeout_ms: int,
    require_tabs: bool = False,
    expected_evidence_text: list[str] | None = None,
    expected_evidence_sample_labels: list[str] | None = None,
    expected_source_detail_text: list[str] | None = None,
    expected_extraction_evidence_text: list[str] | None = None,
    expected_processing_text: list[str] | None = None,
    expected_output_text: list[str] | None = None,
    rejected_workbench_text: list[str] | None = None,
    summary: dict[str, Any] | None = None,
) -> None:
    tab_specs = [
        ("Evidence", re.compile(r"Evidence", re.I), r"Source|Evidence examples|Extraction evidence|Key relationships"),
        ("Processing", re.compile(r"Pipeline|Processing", re.I), r"Ready for chat|Flow|View outputs|Diagnostics"),
        ("Review", re.compile(r"Review", re.I), r"Try|Start with|Key relationships|Ask about|chunks|entities|relations"),
    ]

    for label, tab_name_pattern, expected_pattern in tab_specs:
        tab = page.get_by_role("tab", name=tab_name_pattern).first
        try:
            tab.wait_for(state="visible", timeout=5_000)
        except PlaywrightTimeoutError:
            if require_tabs:
                raise AssertionError(
                    f"Workbench did not expose the required {label} tab."
                )
            return
        disabled = tab.get_attribute("aria-disabled")
        if disabled == "true":
            if require_tabs:
                raise AssertionError(
                    f"Workbench {label} tab was disabled after the pipeline became ready."
                )
            continue
        tab.click(timeout=timeout_ms)
        wait_for_text(page, expected_pattern, timeout_ms)
        page.wait_for_timeout(450)
        if rejected_workbench_text:
            tab_text = body_text(page)
            for rejected in rejected_workbench_text:
                assert_text_missing(
                    tab_text,
                    rejected,
                    context=f"workbench {label} tab text",
                )
            if summary is not None:
                summary.setdefault("workbench_rejected_text_verified", {})
                summary["workbench_rejected_text_verified"][label.lower()] = list(
                    rejected_workbench_text
                )
        if label == "Evidence" and expected_evidence_text:
            evidence_text = body_text(page)
            for expected in expected_evidence_text:
                assert_text_contains(
                    evidence_text,
                    expected,
                    context="workbench Evidence tab text",
                )
            if summary is not None:
                summary["workbench_evidence_text_verified"] = list(expected_evidence_text)
        if label == "Evidence" and expected_evidence_sample_labels:
            sample_rows = page.locator('[data-testid="graphrag-evidence-example-row"]')
            sample_rows.first.wait_for(state="visible", timeout=timeout_ms)
            visible_labels: list[str] = []
            for index in range(sample_rows.count()):
                row = sample_rows.nth(index)
                try:
                    if not row.is_visible(timeout=500):
                        continue
                    row_label = (
                        row.get_attribute("data-evidence-sample-label")
                        or row.inner_text(timeout=timeout_ms).splitlines()[0]
                        or ""
                    ).strip()
                    if row_label:
                        visible_labels.append(row_label)
                except PlaywrightTimeoutError:
                    continue
            if not visible_labels:
                raise AssertionError("Evidence tab did not render visible Evidence example cards.")

            cursor = 0
            for expected in expected_evidence_sample_labels:
                expected_pattern = re.compile(re.escape(expected), re.I)
                while cursor < len(visible_labels) and not expected_pattern.search(visible_labels[cursor]):
                    cursor += 1
                if cursor >= len(visible_labels):
                    raise AssertionError(
                        f"Expected Evidence example label order {expected_evidence_sample_labels!r}; "
                        f"visible labels were {visible_labels!r}."
                    )
                cursor += 1
            if summary is not None:
                summary["workbench_evidence_sample_labels"] = visible_labels
                summary["workbench_evidence_sample_labels_verified"] = list(
                    expected_evidence_sample_labels
                )
        if label == "Evidence" and expected_source_detail_text:
            source_details = page.locator(
                '[data-testid="graphrag-source-details"]'
            ).first
            try:
                source_details.wait_for(state="visible", timeout=5_000)
            except PlaywrightTimeoutError as exc:
                raise AssertionError(
                    "Evidence tab did not expose the Source details disclosure."
                ) from exc

            initially_open = bool(source_details.evaluate("(node) => Boolean(node.open)"))
            if initially_open:
                raise AssertionError("Source details should be collapsed in the high-level Evidence view.")
            source_details.scroll_into_view_if_needed(timeout=timeout_ms)
            screenshots.append(
                str(screenshot(page, out_dir, f"{prefix}-source-summary-collapsed"))
            )
            source_details.locator("summary").first.click(timeout=timeout_ms)
            detail_chips = source_details.locator(
                '[data-testid="graphrag-source-detail-signal"], [data-testid="graphrag-source-artifact-chip"]'
            )
            detail_chips.first.wait_for(state="visible", timeout=timeout_ms)
            source_detail_text = source_details.inner_text(timeout=timeout_ms)
            for expected in expected_source_detail_text:
                assert_text_contains(
                    source_detail_text,
                    expected,
                    context="workbench Source details text",
                )
            if summary is not None:
                summary["workbench_source_detail_text_verified"] = list(
                    expected_source_detail_text
                )
                summary["workbench_source_detail_chip_count"] = detail_chips.count()
            page.wait_for_timeout(350)
            screenshots.append(
                str(screenshot(page, out_dir, f"{prefix}-source-details-expanded"))
            )
            source_details.locator("summary").first.click(timeout=timeout_ms)
            page.wait_for_timeout(250)
        if label == "Evidence" and expected_extraction_evidence_text:
            extraction_details = page.locator(
                '[data-testid="graphrag-extraction-evidence-details"]'
            ).first
            try:
                extraction_details.wait_for(state="visible", timeout=5_000)
            except PlaywrightTimeoutError as exc:
                raise AssertionError(
                    "Evidence tab did not expose the Extraction evidence drilldown."
                ) from exc

            is_open = extraction_details.evaluate("(node) => Boolean(node.open)")
            if not is_open:
                extraction_details.locator("summary").first.click(timeout=timeout_ms)
            row_locator = extraction_details.locator(
                '[data-testid="graphrag-extraction-evidence-row"]'
            )
            row_locator.first.wait_for(state="visible", timeout=timeout_ms)
            row_count = row_locator.count()
            if row_count < 1:
                raise AssertionError("Extraction evidence drilldown did not render sample rows.")
            extraction_text = extraction_details.inner_text(timeout=timeout_ms)
            for expected in expected_extraction_evidence_text:
                assert_text_contains(
                    extraction_text,
                    expected,
                    context="workbench Extraction evidence drilldown text",
                )
            extraction_details.scroll_into_view_if_needed(timeout=timeout_ms)
            if summary is not None:
                summary["workbench_extraction_evidence_text_verified"] = list(
                    expected_extraction_evidence_text
                )
                summary["workbench_extraction_evidence_row_count"] = row_count
            page.wait_for_timeout(450)
            screenshots.append(
                str(screenshot(page, out_dir, f"{prefix}-extraction-evidence-details"))
            )
        if label == "Processing" and expected_processing_text:
            processing_text = body_text(page)
            for expected in expected_processing_text:
                assert_text_contains(
                    processing_text,
                    expected,
                    context="workbench Processing tab text",
                )
            if summary is not None:
                summary["workbench_processing_text_verified"] = list(expected_processing_text)
        if label == "Processing":
            output_button = page.get_by_role(
                "button",
                name=re.compile(r"(?:View|Hide) outputs", re.I),
            ).first
            try:
                output_button.wait_for(state="visible", timeout=5_000)
            except PlaywrightTimeoutError as exc:
                raise AssertionError(
                    "Processing tab did not expose the pipeline output drilldown."
                ) from exc

            button_text = output_button.inner_text(timeout=timeout_ms)
            if re.search(r"\bView outputs\b", button_text, re.I):
                output_button.click(timeout=timeout_ms)
            wait_for_text(page, r"Pipeline outputs", timeout_ms)
            output_rows = page.locator('[data-testid="graphrag-pipeline-output-row"]')
            output_rows.first.wait_for(state="visible", timeout=timeout_ms)
            output_count = output_rows.count()
            if output_count < 1:
                raise AssertionError("Pipeline output drilldown did not render output rows.")
            if expected_output_text:
                output_text = body_text(page)
                for expected in expected_output_text:
                    assert_text_contains(
                        output_text,
                        expected,
                        context="workbench Processing output text",
                    )
                if summary is not None:
                    summary["workbench_output_text_verified"] = list(expected_output_text)
            if rejected_workbench_text:
                output_text = body_text(page)
                for rejected in rejected_workbench_text:
                    assert_text_missing(
                        output_text,
                        rejected,
                        context="workbench Processing output text",
                    )
            page.wait_for_timeout(450)
            if summary is not None:
                summary["workbench_processing_output_count"] = output_count
            screenshots.append(
                str(screenshot(page, out_dir, f"{prefix}-processing-outputs"))
            )
        screenshots.append(
            str(screenshot(page, out_dir, f"{prefix}-{label.lower()}"))
        )


def main() -> int:
    args = parse_args()
    base_url = normalize_url(args.base_url)
    backend_url = normalize_url(args.backend_url)
    env_file = os.environ.get("GRAPHRAG_UI_SMOKE_FILE", "").strip()
    file_args = list(args.file or [])
    if not file_args and env_file:
        file_args = [Path(env_file)]
    file_paths = [Path(path).expanduser().resolve() for path in file_args]
    single_file_path = file_paths[0] if len(file_paths) == 1 else None
    timeout_ms = max(args.timeout, 5) * 1000
    args.out_dir.mkdir(parents=True, exist_ok=True)
    expected_stage_samples = (
        list(args.expect_stage_sample)
        if args.expect_stage_sample
        else (list(DEFAULT_STAGE_SAMPLES) if file_paths and not args.no_deep_pipeline_checks else [])
    )
    expected_artifacts = (
        list(args.expect_artifact)
        if args.expect_artifact
        else (list(DEFAULT_ARTIFACTS) if file_paths and not args.no_deep_pipeline_checks else [])
    )
    expect_pdf_render = args.expect_pdf_render or bool(
        single_file_path and single_file_path.suffix.lower() == ".pdf"
    )

    for file_path in file_paths:
        if not file_path.is_file():
            print(f"Upload file not found: {file_path}", file=sys.stderr)
            return 2

    try:
        status = read_json(f"{backend_url}/api/v1/graphrag/services/status")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Could not read GraphRAG service status: {exc}", file=sys.stderr)
        return 1
    if not service_ready(status):
        print(json.dumps(status, indent=2), file=sys.stderr)
        print("GraphRAG services are not ready.", file=sys.stderr)
        return 1

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    screenshots: list[str] = []
    summary: dict[str, Any] = {
        "started": started,
        "base_url": base_url,
        "backend_url": backend_url,
        "viewport": {
            "width": args.viewport_width,
            "height": args.viewport_height,
            "device_scale_factor": args.device_scale_factor,
        },
        "file": str(single_file_path) if single_file_path else "",
        "files": [str(path) for path in file_paths],
        "question": args.question,
        "expected_evidence_families": args.expect_evidence_family,
        "expected_visible_evidence_families": args.expect_visible_evidence_family,
        "expected_workbench_evidence_text": args.expect_workbench_evidence_text,
        "expected_workbench_evidence_sample_labels": args.expect_workbench_evidence_sample_label,
        "expected_workbench_source_detail_text": args.expect_workbench_source_detail_text,
        "expected_workbench_extraction_evidence_text": args.expect_workbench_extraction_evidence_text,
        "expected_workbench_processing_text": args.expect_workbench_processing_text,
        "expected_workbench_output_text": args.expect_workbench_output_text,
        "rejected_workbench_text": args.reject_workbench_text,
        "expect_workbench_ask_first": args.expect_workbench_ask_first,
        "expected_drawer_references": args.expect_drawer_reference,
        "expected_drawer_sources": args.expect_drawer_source,
        "expect_compact_drawer_search_hidden": args.expect_compact_drawer_search_hidden,
        "expected_source_text": args.expect_source_text,
        "expected_graph_paths": args.expect_graph_path,
        "expected_answer_text": args.expect_answer_text,
        "open_answer_reference_cards": args.open_answer_reference_card,
        "open_selected_evidence_details": args.open_selected_evidence_details,
        "expected_evidence_detail_text": args.expect_evidence_detail_text,
        "expect_uploaded_filenames_in_answer": args.expect_uploaded_filenames_in_answer,
        "use_first_starter_prompt": args.use_first_starter_prompt,
        "starter_prompt_pattern": args.starter_prompt_pattern,
        "ask_workbench_evidence_sample": args.ask_workbench_evidence_sample,
        "ask_workbench_extraction_evidence_sample": args.ask_workbench_extraction_evidence_sample,
        "rejected_answer_text": args.reject_answer_text,
        "rejected_drawer_text": args.reject_drawer_text,
        "expect_pdf_render": expect_pdf_render,
        "simulate_upload_network_error": args.simulate_upload_network_error,
        "skip_layout_checks": args.skip_layout_checks,
        "expected_stage_samples": expected_stage_samples,
        "expected_artifacts": expected_artifacts,
        "layout_checks": [],
        "screenshots": screenshots,
    }

    playwright_manager = sync_playwright()
    playwright = playwright_manager.start()
    browser_processes_before = playwright_browser_processes()
    browser = None
    page = None

    def finish(exit_code: int, *, stderr: bool = False) -> int:
        write_summary(args.out_dir, summary, stderr=stderr)
        if args.skip_browser_close:
            terminate_processes(playwright_browser_processes() - browser_processes_before)
            fast_exit(exit_code)
        try:
            if browser is not None:
                browser.close()
        finally:
            playwright.stop()
        return exit_code

    try:
        browser_name, browser = launch_browser(playwright, args)
        summary["browser"] = browser_name
        page = browser.new_page(
            viewport={
                "width": max(320, args.viewport_width),
                "height": max(480, args.viewport_height),
            },
            device_scale_factor=max(0.5, args.device_scale_factor),
        )
        graph_context_requests: list[dict[str, Any]] = []
        summary["graph_context_requests"] = graph_context_requests

        def record_graph_context_request(request: Any) -> None:
            if "/api/v1/graphrag/chat-context" not in request.url:
                return
            post_data = request.post_data or ""
            graph_context_requests.append({
                "method": request.method,
                "url": request.url,
                "post_data_excerpt": post_data[:1200],
            })

        page.on("request", record_graph_context_request)
        page.set_default_timeout(timeout_ms)
        if args.simulate_upload_network_error:
            install_upload_network_error_route(page)

        try:
            page.goto(f"{base_url}/graphrag", wait_until="networkidle", timeout=timeout_ms)
            if args.expect_workbench_ask_first:
                assert_workbench_ask_first(page, summary)
            screenshots.append(str(screenshot(page, args.out_dir, "01-workbench")))
            if not args.skip_layout_checks:
                assert_viewport_layout_stable(page, "01-workbench", summary)

            if file_paths:
                page.locator('input[type="file"]').first.set_input_files(
                    [str(path) for path in file_paths],
                )
                wait_for_text(page, r"Import plan|selected", timeout_ms)
                screenshots.append(str(screenshot(page, args.out_dir, "02-upload-selected")))
                if not args.skip_layout_checks:
                    assert_viewport_layout_stable(page, "02-upload-selected", summary)

                if args.simulate_upload_network_error:
                    click_first_visible_button(
                        page,
                        r"Process selected upload",
                        r"Process \d+ selected uploads",
                        r"Process \d+",
                        timeout_ms=timeout_ms,
                    )
                    wait_for_text(page, r"Backend API did not respond.*ingest/pipeline", timeout_ms)
                    page.wait_for_timeout(650)
                    screenshots.append(str(screenshot(page, args.out_dir, "03-upload-network-error")))
                    if not args.skip_layout_checks:
                        assert_viewport_layout_stable(page, "03-upload-network-error", summary)
                    text = body_text(page)
                    assert_text_contains(text, "Backend API did not respond", context="upload error")
                    assert_text_contains(text, "Verify the Llama Nexus backend is running", context="upload error")
                    summary["simulated_upload_failure"] = "network_error"
                    summary["final_url"] = page.url
                    summary["status"] = "passed"
                    return finish(0)

                upload_started_at = time.time()
                try:
                    with page.expect_response(
                        lambda response: (
                            "/api/v1/graphrag/ingest/pipeline" in response.url
                            and response.request.method == "POST"
                        ),
                        timeout=timeout_ms,
                    ) as upload_response_info:
                        click_first_visible_button(
                            page,
                            r"Process selected upload",
                            r"Process \d+ selected uploads",
                            r"Process \d+",
                            timeout_ms=timeout_ms,
                        )
                    upload_response = upload_response_info.value
                    summary["upload_status"] = upload_response.status
                    run_ids: list[str] = []
                    try:
                        upload_payload = upload_response.json()
                        if isinstance(upload_payload, dict):
                            run_ids = uploaded_run_ids(upload_payload)
                            summary["pipeline_run_id"] = run_ids[0] if len(run_ids) == 1 else ""
                            summary["pipeline_run_ids"] = run_ids
                            if len(file_paths) > 1:
                                summary["batch"] = batch_upload_summary(upload_payload)
                    except Exception:
                        upload_payload = {}
                        run_ids = []
                    if upload_response.status >= 400:
                        raise AssertionError(f"Upload failed with HTTP {upload_response.status}")
                except PlaywrightTimeoutError:
                    recovered_runs = wait_for_recovered_pipeline_runs(
                        backend_url,
                        file_paths,
                        upload_started_at=upload_started_at,
                        timeout_seconds=args.timeout,
                    )
                    run_ids = uploaded_run_ids({"runs": recovered_runs})
                    upload_payload = {
                        "status": "recovered",
                        "total_files": len(file_paths),
                        "processed": len(recovered_runs),
                        "failed": max(0, len(file_paths) - len(recovered_runs)),
                        "runs": recovered_runs,
                    }
                    summary["upload_status"] = "recovered_after_timeout"
                    summary["upload_response_timeout"] = True
                    summary["pipeline_run_id"] = run_ids[0] if len(run_ids) == 1 else ""
                    summary["pipeline_run_ids"] = run_ids
                    if len(file_paths) > 1:
                        summary["batch"] = batch_upload_summary(upload_payload)
                    wait_for_text(
                        page,
                        r"Documents processed|Ready for chat|Chat ready|Context ready|\bAsk\b",
                        timeout_ms,
                    )
                    screenshots.append(str(screenshot(page, args.out_dir, "03-upload-recovered")))
                    if not args.skip_layout_checks:
                        assert_viewport_layout_stable(page, "03-upload-recovered", summary)
                else:
                    pass
                if not run_ids:
                    raise AssertionError("Upload response did not include pipeline run_id values.")
                if len(file_paths) > 1:
                    batch = summary.get("batch") if isinstance(summary.get("batch"), dict) else {}
                    if positive_int(batch.get("failed")) > 0:
                        raise AssertionError(f"Batch upload reported failed files: {batch}")
                    if len(run_ids) < len(file_paths):
                        raise AssertionError(
                            f"Batch upload returned {len(run_ids)} run IDs for {len(file_paths)} files."
                        )
                if not args.no_deep_pipeline_checks:
                    pipeline_summaries = []
                    pipeline_family_totals = {label: 0 for label in args.expect_evidence_family}
                    per_run_expected_families = (
                        args.expect_evidence_family if len(run_ids) == 1 else []
                    )
                    for run_id in run_ids:
                        detail_payload, ready_summary = wait_for_pipeline_detail(
                            backend_url,
                            run_id,
                            timeout_seconds=args.timeout,
                            expected_stage_samples=expected_stage_samples,
                            expected_artifacts=expected_artifacts,
                            expected_families=per_run_expected_families,
                            min_chunks=args.min_chunks,
                            min_relationships=args.min_relationships,
                            min_path_candidates=args.min_path_candidates,
                        )
                        for label in args.expect_evidence_family:
                            pipeline_family_totals[label] += pipeline_family_count(detail_payload, label)
                        pipeline_summaries.append({
                            "run_id": run_id,
                            **ready_summary,
                        })
                    if len(pipeline_summaries) == 1:
                        summary["pipeline"] = pipeline_summaries[0]
                    else:
                        summary["pipelines"] = pipeline_summaries
                    if len(run_ids) > 1 and args.expect_evidence_family:
                        missing_families = [
                            label
                            for label, count in pipeline_family_totals.items()
                            if count < 1
                        ]
                        if missing_families:
                            raise AssertionError(
                                "missing extracted evidence families across batch: "
                                + ", ".join(missing_families)
                            )
                        summary["batch_evidence_family_counts"] = pipeline_family_totals
                for path in file_paths[:3]:
                    wait_for_text(page, re.escape(path.name), timeout_ms)
                wait_for_text(page, r"Ready for chat|Chat ready|Context ready|Ask set|\bAsk\b", timeout_ms)
                screenshots.append(str(screenshot(page, args.out_dir, "03-pipeline-ready")))
                if not args.skip_layout_checks:
                    assert_viewport_layout_stable(page, "03-pipeline-ready", summary)
                if not args.skip_workbench_tab_inspection:
                    inspect_workbench_tabs(
                        page,
                        args.out_dir,
                        screenshots,
                        prefix="03-pipeline",
                        timeout_ms=timeout_ms,
                        require_tabs=True,
                        expected_evidence_text=args.expect_workbench_evidence_text,
                        expected_evidence_sample_labels=args.expect_workbench_evidence_sample_label,
                        expected_source_detail_text=args.expect_workbench_source_detail_text,
                        expected_extraction_evidence_text=args.expect_workbench_extraction_evidence_text,
                        expected_processing_text=args.expect_workbench_processing_text,
                        expected_output_text=args.expect_workbench_output_text,
                        rejected_workbench_text=args.reject_workbench_text,
                        summary=summary,
                    )
            elif not args.skip_workbench_tab_inspection:
                inspect_workbench_tabs(
                    page,
                    args.out_dir,
                    screenshots,
                    prefix="02-workbench",
                    timeout_ms=timeout_ms,
                    expected_evidence_text=args.expect_workbench_evidence_text,
                    expected_evidence_sample_labels=args.expect_workbench_evidence_sample_label,
                    expected_source_detail_text=args.expect_workbench_source_detail_text,
                    expected_extraction_evidence_text=args.expect_workbench_extraction_evidence_text,
                    expected_processing_text=args.expect_workbench_processing_text,
                    expected_output_text=args.expect_workbench_output_text,
                    rejected_workbench_text=args.reject_workbench_text,
                    summary=summary,
                )

            if args.ask_workbench_evidence_sample:
                summary["workbench_evidence_sample_ask"] = click_workbench_evidence_sample_ask(
                    page,
                    args.ask_workbench_evidence_sample,
                    args.out_dir,
                    screenshots,
                    timeout_ms=timeout_ms,
                    summary=None if args.skip_layout_checks else summary,
                )
            elif args.ask_workbench_extraction_evidence_sample:
                summary["workbench_extraction_evidence_sample_ask"] = (
                    click_workbench_extraction_evidence_sample_ask(
                        page,
                        args.ask_workbench_extraction_evidence_sample,
                        args.out_dir,
                        screenshots,
                        timeout_ms=timeout_ms,
                        summary=None if args.skip_layout_checks else summary,
                    )
                )
            elif args.use_first_starter_prompt or args.starter_prompt_pattern:
                review_tab = page.get_by_role("tab", name=re.compile(r"Review", re.I)).first
                try:
                    review_tab.wait_for(state="visible", timeout=5_000)
                    review_tab.click(timeout=timeout_ms)
                    wait_for_text(page, r"Try|Start with", timeout_ms)
                except PlaywrightTimeoutError:
                    pass

                starter_prompt, starter_index, starter_label = select_starter_prompt(
                    page,
                    args.starter_prompt_pattern,
                    timeout_ms=timeout_ms,
                )
                summary["starter_prompt_label"] = starter_label
                summary["starter_prompt_index"] = starter_index
                screenshots.append(str(screenshot(page, args.out_dir, "04-starter-prompt-ready")))
                if not args.skip_layout_checks:
                    assert_viewport_layout_stable(page, "04-starter-prompt-ready", summary)
                starter_prompt.click(timeout=timeout_ms)
            else:
                textarea = page.locator(
                    'textarea[placeholder*="Ask"], input[placeholder*="Ask"]',
                ).first
                textarea.wait_for(state="visible", timeout=timeout_ms)
                textarea.fill(args.question)
                screenshots.append(str(screenshot(page, args.out_dir, "04-question-filled")))
                if not args.skip_layout_checks:
                    assert_viewport_layout_stable(page, "04-question-filled", summary)

                click_first_visible_locator(
                    [
                        (
                            "mobile ask",
                            page.locator(
                                '[data-testid="graphrag-selected-document-ask-mobile"]'
                            ).first,
                        ),
                        (
                            "desktop ask",
                            page.locator(
                                '[data-testid="graphrag-selected-document-ask"]'
                            ).first,
                        ),
                        (
                            "ask button",
                            page.get_by_role(
                                "button",
                                name=re.compile(r"^Ask(?: set)?$", re.I),
                            ).first,
                        ),
                    ],
                    timeout_ms=timeout_ms,
                )

            try:
                page.wait_for_url(re.compile(r"/chat(?:$|[?#])"), timeout=timeout_ms)
            except PlaywrightTimeoutError:
                print(f"Chat navigation wait timed out; current URL: {page.url}")
            try:
                page.wait_for_load_state("networkidle", timeout=timeout_ms)
            except PlaywrightTimeoutError:
                print("Network idle wait timed out after opening chat.")

            wait_for_chat_answer_complete(page, timeout_ms)
            page.wait_for_timeout(800)
            screenshots.append(str(screenshot(page, args.out_dir, "05-chat-answer")))
            if not args.skip_layout_checks:
                assert_viewport_layout_stable(page, "05-chat-answer", summary)
            assert_answer_reference_tray_visible(page, summary)

            text = body_text(page)
            summary["chat_answer_text_excerpt"] = text[-5000:]
            if re.search(r"Writing cited answer|Writing\b|Thinking|Generating|Loading", text, re.I):
                raise AssertionError("Chat answer was still generating after the wait completed.")
            if "References" not in text:
                raise AssertionError("Reference tray did not render.")
            for label in args.expect_visible_evidence_family:
                assert_evidence_family_visible(text, label)
            for label in args.expect_answer_text:
                assert_text_contains(text, label, context="answer text")
            for spec in args.expect_graph_path:
                assert_graph_path_visible(text, spec, context="answer graph path")
            if args.expect_uploaded_filenames_in_answer:
                for path in file_paths:
                    assert_text_contains(
                        text,
                        path.name,
                        context="uploaded filename in answer",
                    )
            for label in args.reject_answer_text:
                assert_text_missing(text, label, context="answer text")

            if args.open_answer_reference_card:
                card_selections = []
                detail_expansions = []
                for index, label in enumerate(args.open_answer_reference_card):
                    if index > 0:
                        page.keyboard.press("Escape")
                        page.wait_for_timeout(450)
                    card_selections.append(
                        open_answer_reference_card(
                            page,
                            label,
                            args.out_dir,
                            screenshots,
                            timeout_ms=timeout_ms,
                        )
                    )
                    if args.open_selected_evidence_details:
                        detail_expansions.append(
                            expand_selected_evidence_details(
                                page,
                                label,
                                args.out_dir,
                                screenshots,
                                (
                                    args.expect_evidence_detail_text
                                    if index == len(args.open_answer_reference_card) - 1
                                    else []
                                ),
                                timeout_ms=timeout_ms,
                            )
                        )
                summary["answer_reference_card_selections"] = card_selections
                if detail_expansions:
                    summary["selected_evidence_detail_expansions"] = detail_expansions
            else:
                click_first_visible_button(page, r"Review evidence", timeout_ms=timeout_ms)
                if args.open_selected_evidence_details:
                    summary["selected_evidence_detail_expansions"] = [
                        expand_selected_evidence_details(
                            page,
                            "review",
                            args.out_dir,
                            screenshots,
                            args.expect_evidence_detail_text,
                            timeout_ms=timeout_ms,
                        )
                    ]
            wait_for_text(page, r"Selected evidence|Evidence details|Full document text", timeout_ms)
            page.wait_for_timeout(800)
            screenshots.append(str(screenshot(page, args.out_dir, "06-evidence-drawer")))
            if not args.skip_layout_checks:
                assert_viewport_layout_stable(page, "06-evidence-drawer", summary)
            assert_reference_drawer_hierarchy(
                page,
                summary,
                args.expect_drawer_source,
                timeout_ms=timeout_ms,
            )
            if args.expect_compact_drawer_search_hidden:
                visible_searches = 0
                search_inputs = page.locator('input[placeholder="Find source"]')
                for index in range(search_inputs.count()):
                    try:
                        if search_inputs.nth(index).is_visible(timeout=500):
                            visible_searches += 1
                    except PlaywrightTimeoutError:
                        continue
                if visible_searches > 0:
                    raise AssertionError(
                        f"Compact evidence drawer still showed {visible_searches} Find source search field(s)."
                    )
                summary["compact_drawer_search_hidden"] = True

            if expect_pdf_render:
                summary["pdf_reference_selection"] = select_pdf_reference_for_render(page, timeout_ms)
                page.wait_for_timeout(450)
                screenshots.append(str(screenshot(page, args.out_dir, "06-evidence-drawer-pdf-ready")))
                if not args.skip_layout_checks:
                    assert_viewport_layout_stable(page, "06-evidence-drawer-pdf-ready", summary)

            drawer_text = body_text(page)
            for label in args.expect_visible_evidence_family:
                assert_evidence_family_visible(drawer_text, label)
            for label in args.reject_drawer_text:
                assert_text_missing(drawer_text, label, context="evidence drawer text")
            if expect_pdf_render:
                summary["pdf_render"] = assert_pdf_canvas_rendered(page, timeout_ms)
            if args.expect_drawer_reference:
                summary["drawer_reference_selections"] = [
                    select_drawer_reference(
                        page,
                        label,
                        args.out_dir,
                        screenshots,
                        timeout_ms=timeout_ms,
                    )
                    for label in args.expect_drawer_reference
                ]
            if args.expect_graph_path:
                summary["graph_path_selections"] = [
                    select_graph_path_reference(
                        page,
                        spec,
                        args.out_dir,
                        screenshots,
                        timeout_ms=timeout_ms,
                    )
                    for spec in args.expect_graph_path
                ]

            source_expansion = None
            if expect_pdf_render and "pdf_render" in summary:
                summary["source_text_expansion"] = {
                    "expanded": False,
                    "skipped_plain_text_for_pdf": True,
                    "expected_terms": args.expect_source_text,
                    "reason": (
                        "PDF sources are verified through the rendered page canvas "
                        "and highlight count instead of the non-PDF text expander."
                    ),
                }
            else:
                source_expansion = expand_source_text_panel(
                    page,
                    summary,
                    args.expect_source_text,
                    timeout_ms=timeout_ms,
                )
            if source_expansion:
                screenshots.append(str(screenshot(page, args.out_dir, "07-source-text-expanded")))

            summary["final_url"] = page.url
            summary["status"] = "passed"
        except Exception as exc:
            summary["status"] = "failed"
            summary["error"] = str(exc)
            if page is not None:
                try:
                    screenshots.append(str(screenshot(page, args.out_dir, "failure")))
                except Exception as screenshot_error:
                    summary["failure_screenshot_error"] = str(screenshot_error)
            return finish(1, stderr=True)

        return finish(0)
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        if page is not None:
            try:
                screenshots.append(str(screenshot(page, args.out_dir, "failure")))
            except Exception as screenshot_error:
                summary["failure_screenshot_error"] = str(screenshot_error)
        return finish(1, stderr=True)


if __name__ == "__main__":
    raise SystemExit(main())
