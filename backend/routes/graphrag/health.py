"""
GraphRAG Health & Status Routes

Service health checks, status, stats, and capability queries.
All endpoints use the shared connection-pooled HTTP client.
"""

from fastapi import APIRouter, HTTPException
import asyncio
import httpx
import importlib.util
import logging
import shlex
import shutil

from .helpers import (
    GRAPHRAG_COMPATIBLE_INGEST_PATHS,
    check_graphrag_enabled,
    compatible_health_urls,
    compatible_ingest_urls,
    get_http_client,
)
from .upstream import _build_deployment_diagnostics
from modules.graphrag.config import get_graphrag_config

logger = logging.getLogger(__name__)

router = APIRouter()


SERVICE_DESCRIPTORS = {
    "graphrag": {
        "required_env": ["GRAPHRAG_URL", "GRAPHRAG_SERVICE_URL"],
        "deploy_command": "Set GRAPHRAG_URL to the external GraphRAG API base URL",
        "description": "External GraphRAG API for graph search, ingestion, exploration, reasoning, and evaluation",
        "optional": False,
    },
    "ner": {
        "required_env": ["NER_API_URL", "GRAPHRAG_NER_URL"],
        "deploy_command": "GRAPHRAG_ENABLED=true docker compose --profile graphrag-ner up -d",
        "description": "GLiNER-based entity extraction",
        "optional": True,
    },
    "rel": {
        "required_env": ["REL_API_URL", "GRAPHRAG_REL_URL"],
        "deploy_command": "GRAPHRAG_ENABLED=true docker compose --profile graphrag-rel up -d",
        "description": "Relationship extraction",
        "optional": True,
    },
    "code_rag": {
        "required_env": ["CODE_RAG_URL", "GRAPHRAG_CODE_RAG_URL"],
        "deploy_command": "GRAPHRAG_ENABLED=true docker compose --profile graphrag-code up -d",
        "description": "Code detection, code search, code ingestion, and code analysis",
        "optional": True,
    },
    "reddit_crawler": {
        "required_env": ["REDDIT_CRAWLER_URL", "GRAPHRAG_REDDIT_CRAWLER_URL"],
        "deploy_command": "GRAPHRAG_ENABLED=true docker compose --profile reddit-crawler up -d",
        "description": "Reddit crawling and optional GraphRAG auto-ingestion",
        "optional": True,
    },
}

SERVICE_STATUS_META_KEY = "_meta"


def _service_descriptor(name: str) -> dict:
    return SERVICE_DESCRIPTORS.get(name, {
        "required_env": [f"{name.upper()}_URL"],
        "deploy_command": None,
        "description": name,
        "optional": True,
    })


def _service_url_source_label(source: str | None) -> str:
    if source == "docker_default":
        return "Docker service default"
    if not source or source == "unset":
        return "Not configured"
    return source


def _service_source_metadata(name: str) -> dict:
    source = get_graphrag_config().service_url_sources.get(name, "unset")
    return {
        "url_source": source,
        "url_source_label": _service_url_source_label(source),
        "auto_discovered": source == "docker_default",
        "explicit": bool(source and source not in {"unset", "docker_default"}),
    }


def _offline_hybrid_status(message: str = "GraphRAG service is disabled") -> dict:
    return {
        "graphrag_healthy": False,
        "code_rag_healthy": False,
        "hybrid_available": False,
        "ner_available": False,
        "rel_available": False,
        "message": message,
    }


def _default_extraction_stats(message: str = "GraphRAG service is disabled") -> dict:
    return {
        "available": False,
        "message": message,
        "available_methods": [],
        "spanbert_available": False,
        "dependency_available": False,
        "entity_linking_available": False,
    }


def _default_supported_formats(cfg=None, message: str = "GraphRAG service is disabled") -> dict:
    cfg = cfg or get_graphrag_config()
    ocr_status = _document_ocr_runtime_status(cfg)
    supported_formats = [
        "txt",
        "md",
        "html",
        "csv",
        "json",
        "pdf",
        "doc",
        "docx",
        "docm",
        "dotx",
        "dotm",
        "ppt",
        "pptx",
        "pptm",
        "potx",
        "potm",
        "ppsx",
        "ppsm",
        "xls",
        "xlsx",
        "xlsm",
        "xltx",
        "xltm",
        "eml",
        "rtf",
        "epub",
        "odt",
        "ods",
        "odp",
        "zip",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "bmp",
        "tif",
        "tiff",
        "webp",
        "svg",
    ]
    return {
        "available": False,
        "message": message,
        "supported_formats": supported_formats,
        "formats": supported_formats,
        "features": {
            "semantic_chunking": False,
            "metadata_extraction": True,
            "content_type_classification": True,
            "structure_preservation": True,
            "archive_extraction": True,
            "embedded_attachment_extraction": True,
            "nested_document_extraction": True,
            "legacy_office_extraction": True,
            "image_metadata_extraction": True,
            "ocr_extraction": True,
            "ocr_extraction_enabled": ocr_status["enabled"],
            "ocr_extraction_ready": ocr_status["ready"],
            "ocr_runtime_available": ocr_status["runtime_available"],
            "pipeline_evidence_profile": True,
            "entity_extraction": cfg.ner_enabled,
            "relationship_extraction": cfg.rel_enabled,
            "code_detection": cfg.code_rag_enabled,
            "reddit_crawling": cfg.reddit_crawler_enabled,
        },
        "document_ocr": ocr_status,
    }


def _document_ocr_runtime_status(cfg=None) -> dict:
    cfg = cfg or get_graphrag_config()
    enabled = bool(getattr(cfg, "image_ocr_enabled", False))

    def module_available(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, AttributeError, ValueError):
            return False

    pillow_available = module_available("PIL")
    pytesseract_available = module_available("pytesseract")
    tesseract_path = shutil.which("tesseract")
    runtime_available = bool(pillow_available and pytesseract_available and tesseract_path)
    ready = bool(enabled and runtime_available)
    missing = [
        label
        for label, available in (
            ("Pillow", pillow_available),
            ("pytesseract", pytesseract_available),
            ("tesseract binary", bool(tesseract_path)),
        )
        if not available
    ]

    if ready:
        status = "ready"
        message = "Image OCR is enabled and the pytesseract runtime is available."
    elif not enabled:
        status = "disabled"
        message = "Set GRAPHRAG_ENABLE_IMAGE_OCR=true to extract text from scanned pages and packaged images."
    elif missing:
        status = "missing_runtime"
        message = f"Image OCR is enabled, but missing runtime dependency: {', '.join(missing)}."
    else:
        status = "blocked"
        message = "Image OCR is enabled, but the runtime is not ready."

    return {
        "enabled": enabled,
        "configured": enabled,
        "available": ready,
        "ready": ready,
        "status": status,
        "engine": "pytesseract",
        "runtime_available": runtime_available,
        "runtime": {
            "pillow": pillow_available,
            "pytesseract": pytesseract_available,
            "tesseract_binary": bool(tesseract_path),
            "tesseract_path": tesseract_path,
        },
        "missing": missing,
        "required_env": ["GRAPHRAG_ENABLE_IMAGE_OCR"],
        "deployment_hint": (
            "Set GRAPHRAG_ENABLE_IMAGE_OCR=true and install Pillow, pytesseract, "
            "and the tesseract-ocr system package."
        ),
        "message": message,
    }


def _document_evidence_capabilities(enabled: bool) -> list[dict]:
    status = "ready" if enabled else "disabled"
    capabilities = [
        {
            "id": "presentation_structure",
            "label": "Presentation Structure",
            "formats": ["pptx", "pptm", "potx", "potm", "ppsx", "ppsm", "odp"],
            "element_types": ["slide_metadata", "slide_title", "slide_text", "speaker_notes"],
            "relationship_types": ["has_slide", "has_hidden_slide", "contains_speaker_notes"],
            "interaction_outputs": ["capability_card", "starter_prompt", "citation_scope"],
            "detail": "Slides, hidden-slide provenance, speaker notes, and slide source parts.",
        },
        {
            "id": "tables",
            "label": "Tables",
            "formats": ["docx", "docm", "pptx", "pptm", "xlsx", "xlsm", "odt", "ods", "odp", "pdf", "html", "csv", "md", "markdown"],
            "element_types": ["table_header", "table_row", "table_cell", "table_definition"],
            "relationship_types": ["contains_table", "table_fact", "emphasizes"],
            "interaction_outputs": ["relationship_evidence_card", "retrieval_rehearsal", "citation_scope"],
            "detail": "Rows, cells, coordinates, spans, styling, and table-derived graph facts.",
        },
        {
            "id": "charts",
            "label": "Charts",
            "formats": ["pptx", "pptm", "xlsx", "xlsm", "ods", "odp"],
            "element_types": ["chart_axis", "chart_legend", "chart_series", "chart_point", "chart_trendline"],
            "relationship_types": ["contains_chart", "contains_metric", "uses_chart_source", "has_axis", "has_trendline"],
            "interaction_outputs": ["capability_card", "starter_prompt", "citation_scope"],
            "detail": "Chart titles, axes, legends, series, points, trendlines, and source tables.",
        },
        {
            "id": "diagrams",
            "label": "Diagrams",
            "formats": ["pptx", "pptm", "docx", "docm", "md", "markdown"],
            "element_types": ["diagram_node", "diagram_relationship"],
            "relationship_types": ["contains_diagram_node", "contains_diagram_relationship"],
            "interaction_outputs": ["capability_card", "starter_prompt", "graph_path_context"],
            "detail": "SmartArt and Mermaid nodes, hierarchy, source/target labels, and diagram edge types.",
        },
        {
            "id": "shapes_connectors",
            "label": "Shapes & Connectors",
            "formats": ["pptx", "pptm", "odp", "odt", "ods"],
            "element_types": ["shape_metadata", "shape_connector", "text_box"],
            "relationship_types": ["contains_shape", "contains_connector", "connects_to"],
            "interaction_outputs": ["capability_card", "starter_prompt", "citation_scope", "graph_path_context"],
            "detail": "Shape ids/titles/descriptions, bboxes, connector endpoints, labels, and relation text.",
        },
        {
            "id": "link_actions",
            "label": "Links & Actions",
            "formats": ["pdf", "docx", "docm", "pptx", "pptm", "xlsx", "xlsm", "odt", "ods", "odp", "epub", "html", "eml", "md", "markdown"],
            "element_types": ["link"],
            "relationship_types": ["links_to", "contains_link"],
            "interaction_outputs": [
                "capability_card",
                "starter_prompt",
                "relationship_evidence_card",
                "graph_anchor",
                "citation_scope",
                "retrieval_rehearsal",
            ],
            "detail": "URI hyperlinks, source-anchored links, and PowerPoint image/layout action links with relationship provenance.",
        },
        {
            "id": "messages",
            "label": "Messages",
            "formats": ["eml"],
            "element_types": ["email_body", "document_metadata", "embedded_object", "link"],
            "relationship_types": ["sent_by", "sent_to", "cc_to", "contains_email_body", "has_attachment"],
            "interaction_outputs": ["capability_card", "relationship_evidence_card", "citation_scope", "retrieval_rehearsal"],
            "detail": "Email headers, sender/recipient graph edges, body text, links, and parsed attachments.",
        },
        {
            "id": "legacy_office",
            "label": "Legacy Office",
            "formats": ["doc", "ppt", "xls"],
            "element_types": ["legacy_office_stream", "legacy_office_text"],
            "relationship_types": ["contains_legacy_office_stream", "contains_legacy_office_text"],
            "interaction_outputs": ["capability_card", "starter_prompt", "structured_ingest_export", "citation_scope"],
            "detail": "Best-effort OLE stream inventory and printable text extraction; convert to OOXML for full layout, chart, table, comment, and relationship extraction.",
        },
        {
            "id": "forms",
            "label": "Forms",
            "formats": ["pdf", "docx", "docm", "pptx", "pptm", "odt", "ods", "odp", "md", "markdown"],
            "element_types": ["form_field"],
            "relationship_types": ["has_field", "contains_form_field", "has_value"],
            "interaction_outputs": ["capability_card", "starter_prompt", "citation_scope"],
            "detail": "Form labels, values, options, selected states, controls, and source parts.",
        },
        {
            "id": "markdown_semantics",
            "label": "Markdown Semantics",
            "formats": ["md", "markdown"],
            "element_types": [
                "document_metadata",
                "heading",
                "link",
                "image",
                "form_field",
                "code_block",
                "diagram_node",
                "diagram_relationship",
                "table_header",
                "table_row",
            ],
            "relationship_types": [
                "has_section",
                "contains_code_block",
                "uses_language",
                "links_to",
                "contains_link",
                "has_field",
                "contains_form_field",
                "contains_diagram_node",
                "contains_diagram_relationship",
            ],
            "interaction_outputs": [
                "capability_card",
                "relationship_evidence_card",
                "graph_path_context",
                "retrieval_rehearsal",
                "structured_ingest_export",
                "citation_scope",
            ],
            "detail": "Frontmatter, headings, task lists, links/images, fenced code, Mermaid graph edges, and Markdown tables with section provenance.",
        },
        {
            "id": "styled_text",
            "label": "Styled Text",
            "formats": ["pdf", "docx", "docm", "pptx", "pptm", "xlsx", "xlsm", "odt", "ods", "odp"],
            "element_types": ["heading", "paragraph", "slide_text", "text_box", "table_cell"],
            "relationship_types": ["emphasizes", "uses_bullet_structure"],
            "interaction_outputs": ["relationship_evidence_card", "retrieval_rehearsal", "citation_scope"],
            "detail": "Bold, italic, underline, color, paragraph styles, emphasized terms, and bullet levels.",
        },
        {
            "id": "spreadsheet_lineage",
            "label": "Spreadsheet Lineage",
            "formats": ["xlsx", "xlsm", "xltx", "xltm", "ods"],
            "element_types": ["formula", "pivot_table", "named_range", "conditional_format"],
            "relationship_types": ["depends_on", "contains_pivot_table", "has_named_range", "has_conditional_format"],
            "interaction_outputs": ["relationship_evidence_card", "graph_path_context", "retrieval_rehearsal"],
            "detail": "Formulas, references, pivots, named ranges, validations, and conditional formatting.",
        },
        {
            "id": "layout_ocr",
            "label": "Layout & OCR",
            "formats": ["pdf", "pptx", "pptm", "docx", "docm", "odp", "odt", "ods", "png", "jpg", "jpeg", "tiff", "webp"],
            "element_types": ["image", "image_ocr_text", "page_ocr_text"],
            "relationship_types": ["contains_image", "image_contains_text", "contains_text"],
            "interaction_outputs": ["capability_card", "starter_prompt", "citation_scope"],
            "detail": "Readable bboxes/zones plus optional OCR for scanned pages and packaged images.",
        },
        {
            "id": "source_artifacts",
            "label": "Source Artifacts",
            "formats": ["zip", "pdf", "docx", "docm", "pptx", "pptm", "xlsx", "xlsm", "epub", "eml"],
            "element_types": ["archive_entry", "embedded_object", "media"],
            "relationship_types": ["contains_archive_entry", "contains_embedded_object", "contains_media"],
            "interaction_outputs": ["artifact_manifest", "structured_ingest_export", "citation_scope"],
            "detail": "Archives, PDF attachments, embedded packages, media, and nested parsed child evidence.",
        },
    ]
    return [
        {
            **capability,
            "ready": enabled,
            "status": status,
        }
        for capability in capabilities
    ]


def _local_document_pipeline_contract(cfg=None, *, enabled: bool | None = None) -> dict:
    cfg = cfg or get_graphrag_config()
    enabled = bool(cfg.enabled if enabled is None else enabled)
    supported = _default_supported_formats(cfg, "Local document pipeline format support")
    ocr_status = _document_ocr_runtime_status(cfg)
    supported_formats = supported["supported_formats"]
    evidence_capabilities = _document_evidence_capabilities(enabled)

    return {
        "available": enabled,
        "ready": enabled,
        "status": "local" if enabled else "disabled",
        "mode": "local_document_evidence_preview",
        "endpoint": "/api/v1/graphrag/ingest/pipeline/preview",
        "upload_endpoint": "/api/v1/graphrag/ingest/pipeline/upload",
        "artifact_endpoint_template": "/api/v1/graphrag/ingest/pipeline/{run_id}/artifacts",
        "supported_formats": supported_formats,
        "format_count": len(supported_formats),
        "sample_limit_cap": 80,
        "relationship_sample_mode": "raw_prefix_plus_representative_highlights",
        "evidence_capability_count": len(evidence_capabilities),
        "evidence_capabilities": evidence_capabilities,
        "stage_outputs": [
            "received",
            "structure_extraction",
            "chunking",
            "graph_candidates",
            "embedding_plan",
            "local_context_cache",
            "graph_interaction_plan",
        ],
        "evidence_outputs": [
            "evidence_profile",
            "relationship_highlights",
            "relationship_evidence_cards",
            "local_graph",
            "local_context",
            "local_graph_reasoning",
            "interaction_handoff",
            "retrieval_rehearsals",
        ],
        "features": {
            **supported["features"],
            "semantic_chunking": True,
            "local_graph_artifact": True,
            "local_context_cache": True,
            "graph_reasoning_index": True,
            "interaction_handoff": True,
            "retrieval_rehearsals": True,
            "balanced_relationship_samples": True,
            "chart_relationships": True,
            "diagram_relationships": True,
            "layout_relationships": True,
            "link_action_relationships": True,
            "markdown_graph_evidence": True,
            "styled_text_relationships": True,
            "source_artifact_profiles": True,
            "ocr_extraction_enabled": ocr_status["enabled"],
            "ocr_extraction_ready": ocr_status["ready"],
        },
        "document_ocr": ocr_status,
        "next_step": (
            "Use Pipeline Preview or Pipeline Upload to create a durable trace and chat handoff."
            if enabled
            else "Set GRAPHRAG_ENABLED=true and restart the backend to enable local pipeline traces."
        ),
    }


async def _check_service(url: str | None, name: str) -> dict:
    """Check if a service is available."""
    descriptor = _service_descriptor(name)
    source_metadata = _service_source_metadata(name)
    if not url:
        return {
            "available": False,
            "configured": False,
            "status": "not_configured",
            **source_metadata,
            "required_env": descriptor["required_env"],
            "deployment_hint": descriptor["deploy_command"],
            "optional": descriptor["optional"],
            "message": f"Set {descriptor['required_env'][0]} to enable",
        }
    health_urls = compatible_health_urls(url)
    health_url = health_urls[0]
    attempts: list[dict] = []
    try:
        client = get_http_client()
        response = None
        for index, candidate_url in enumerate(health_urls):
            health_url = candidate_url
            response = await client.get(candidate_url, timeout=5.0)
            attempts.append({
                "url": candidate_url,
                "status": "http_response",
                "status_code": response.status_code,
            })
            if response.status_code in {404, 405} and index < len(health_urls) - 1:
                attempts[-1]["status"] = "not_found"
                continue
            break
        if response is not None and response.status_code == 200:
            payload = {}
            try:
                payload = response.json()
            except Exception:
                pass
            return {
                "available": True,
                "configured": True,
                "status": "healthy",
                "url": url,
                "health_url": health_url,
                "compatible_health_urls": health_urls,
                "attempted_health_urls": [attempt["url"] for attempt in attempts],
                "health_attempts": attempts,
                **source_metadata,
                "required_env": descriptor["required_env"],
                "deployment_hint": descriptor["deploy_command"],
                "optional": descriptor["optional"],
                "response": payload,
            }
        return {
            "available": False,
            "configured": True,
            "status": "unhealthy",
            "url": url,
            "health_url": health_url,
            "compatible_health_urls": health_urls,
            "attempted_health_urls": [attempt["url"] for attempt in attempts],
            "health_attempts": attempts,
            **source_metadata,
            "required_env": descriptor["required_env"],
            "deployment_hint": descriptor["deploy_command"],
            "optional": descriptor["optional"],
            "code": response.status_code if response is not None else None,
        }
    except Exception as e:
        if not attempts or attempts[-1].get("url") != health_url:
            attempts.append({
                "url": health_url,
                "status": "unavailable",
                "error": str(e),
            })
        return {
            "available": False,
            "configured": True,
            "status": "unavailable",
            "url": url,
            "health_url": health_url,
            "compatible_health_urls": health_urls,
            "attempted_health_urls": [attempt["url"] for attempt in attempts],
            "health_attempts": attempts,
            **source_metadata,
            "required_env": descriptor["required_env"],
            "deployment_hint": descriptor["deploy_command"],
            "optional": descriptor["optional"],
            "error": str(e),
        }


def _workflow_operating_mode(*, status: str, workflows: dict[str, dict], diagnostics: dict) -> dict:
    enabled = bool(diagnostics.get("enabled"))
    local_pipeline_ready = bool(workflows.get("local_pipeline_preview", {}).get("ready"))
    chat_handoff_ready = bool(workflows.get("pipeline_chat_handoff", {}).get("ready"))
    remote_ingest_ready = bool(workflows.get("remote_ingest", {}).get("ready"))
    graph_search_ready = bool(workflows.get("graph_search", {}).get("ready"))
    extraction_ready = bool(workflows.get("extraction", {}).get("ready"))

    if not enabled:
        mode_id = "disabled"
        label = "Disabled"
        summary = "GraphRAG routes are disabled."
        upload_mode = "unavailable"
        chat_mode = "standard_chat"
        recommended_action = "Set GRAPHRAG_ENABLED=true and restart the backend."
    elif remote_ingest_ready and graph_search_ready:
        mode_id = "remote_graph"
        label = "Remote Graph Ready"
        summary = "Uploads can produce local traces and submit structured payloads to the configured GraphRAG API."
        upload_mode = "local_trace_plus_remote_ingest"
        chat_mode = "remote_graph_plus_local_pipeline"
        recommended_action = "Upload documents through the pipeline, then use Ask Chat or graph search."
    elif extraction_ready and local_pipeline_ready:
        mode_id = "hybrid_helpers"
        label = "Hybrid Helpers"
        summary = "Local pipeline and run-scoped chat are ready, with helper extraction services available while remote graph ingest is offline."
        upload_mode = "local_trace_with_helper_extraction"
        chat_mode = "local_pipeline_graph_context"
        recommended_action = "Use pipeline upload now; configure GRAPHRAG_URL when remote graph ingest is needed."
    elif local_pipeline_ready and chat_handoff_ready:
        mode_id = "local_pipeline"
        label = "Local Pipeline"
        summary = "Local extraction traces, graph candidates, cached chunks, bridge/path samples, and run-scoped chat are ready without a remote GraphRAG API."
        upload_mode = "local_trace_only"
        chat_mode = "local_pipeline_graph_context"
        recommended_action = "Upload documents through pipeline preview/upload and use Ask Chat from the trace."
    else:
        mode_id = status or "offline"
        label = "Offline"
        summary = diagnostics.get("summary") or "GraphRAG workflows are not operational yet."
        upload_mode = "unavailable"
        chat_mode = "standard_chat"
        recommended_action = (diagnostics.get("warnings") or ["Check GraphRAG deployment diagnostics."])[0]

    return {
        "id": mode_id,
        "label": label,
        "status": status,
        "ready": bool(local_pipeline_ready or remote_ingest_ready or graph_search_ready or extraction_ready),
        "summary": summary,
        "upload_mode": upload_mode,
        "chat_mode": chat_mode,
        "local_pipeline_available": local_pipeline_ready,
        "chat_handoff_available": chat_handoff_ready,
        "remote_ingest_available": remote_ingest_ready,
        "graph_search_available": graph_search_ready,
        "extraction_available": extraction_ready,
        "recommended_action": recommended_action,
    }


def _quickstart_item(
    *,
    item_id: str,
    label: str,
    status: str,
    summary: str,
    command: str,
    env_vars: list[str] | None = None,
    profiles: list[str] | None = None,
    checks: list[str] | None = None,
    endpoint: str | None = None,
    next_step: str | None = None,
) -> dict:
    status = status or "blocked"
    return {
        "id": item_id,
        "label": label,
        "status": status,
        "ready": status in {"ready", "local", "partial"},
        "summary": summary,
        "command": command,
        "env_vars": [value for value in (env_vars or []) if value],
        "profiles": [value for value in (profiles or []) if value],
        "checks": [value for value in (checks or []) if value],
        "endpoint": endpoint,
        "next_step": next_step,
    }


def _pathway_item(
    *,
    item_id: str,
    label: str,
    status: str,
    summary: str,
    action: str,
    mode: str,
    endpoint: str | None = None,
    secondary_endpoint: str | None = None,
    checks: list[str] | None = None,
    outputs: list[str] | None = None,
    uses_local_pipeline: bool = False,
    uses_remote_graph: bool = False,
    recommended: bool = False,
) -> dict:
    status = status or "blocked"
    return {
        "id": item_id,
        "label": label,
        "status": status,
        "ready": status in {"ready", "local", "partial"},
        "recommended": recommended,
        "summary": summary,
        "action": action,
        "mode": mode,
        "endpoint": endpoint,
        "secondary_endpoint": secondary_endpoint,
        "checks": [check for check in (checks or []) if check],
        "outputs": [output for output in (outputs or []) if output],
        "uses_local_pipeline": uses_local_pipeline,
        "uses_remote_graph": uses_remote_graph,
    }


def _services_interaction_pathways(*, workflows: dict[str, dict], interaction_contract: dict, operating_mode: dict) -> list[dict]:
    upload = interaction_contract.get("upload") if isinstance(interaction_contract.get("upload"), dict) else {}
    chat = interaction_contract.get("chat") if isinstance(interaction_contract.get("chat"), dict) else {}
    search = interaction_contract.get("search") if isinstance(interaction_contract.get("search"), dict) else {}
    deployment = interaction_contract.get("deployment") if isinstance(interaction_contract.get("deployment"), dict) else {}
    document_pipeline = interaction_contract.get("document_pipeline") if isinstance(interaction_contract.get("document_pipeline"), dict) else {}
    document_ocr = deployment.get("document_ocr") if isinstance(deployment.get("document_ocr"), dict) else {}

    local_pipeline = workflows.get("local_pipeline_preview", {})
    chat_handoff = workflows.get("pipeline_chat_handoff", {})
    remote_ingest = workflows.get("remote_ingest", {})
    graph_search = workflows.get("graph_search", {})
    extraction = workflows.get("extraction", {})

    enabled = bool(deployment.get("enabled"))
    upload_available = bool(upload.get("available"))
    chat_available = bool(chat.get("available"))
    remote_submission = bool(upload.get("remote_submission"))
    search_available = bool(search.get("available"))
    search_fallback = str(search.get("fallback") or "")
    local_ready = bool(upload_available and chat_available)
    remote_ready = bool(remote_submission and search_available)
    ocr_ready = bool(document_ocr.get("ready"))
    ocr_enabled = bool(document_ocr.get("enabled") or document_ocr.get("configured"))

    mode_id = str(operating_mode.get("id") or "")
    recommended_local = mode_id in {"local_pipeline", "hybrid_helpers"} or (local_ready and not remote_ready)
    recommended_remote = mode_id == "remote_graph" and remote_ready
    recommended_helpers = bool(extraction.get("ready") and not (recommended_local or recommended_remote))
    recommended_ocr = bool(ocr_ready and not (recommended_local or recommended_remote or recommended_helpers))

    if not enabled:
        local_status = "disabled"
    else:
        local_status = "local" if local_ready else _workflow_status(local_pipeline)
    remote_status = (
        "ready" if remote_ready
        else "local" if local_ready and not search_available
        else "blocked" if enabled
        else "disabled"
    )
    helper_status = _workflow_status(extraction) if enabled else "disabled"
    ocr_status = "ready" if ocr_ready else "blocked" if ocr_enabled else "disabled"

    return [
        _pathway_item(
            item_id="upload_trace_chat",
            label="Upload -> Trace -> Ask Chat",
            status=local_status,
            recommended=recommended_local,
            summary=(
                "Create a durable pipeline trace with stage samples, local graph candidates, cached chunks, and graph-path context for chat."
                if local_ready
                else "Enable GraphRAG routes before upload traces can feed chat."
            ),
            action=(
                "Use Pipeline Upload, then Ask Chat with the generated pipeline_run_id."
                if local_ready
                else local_pipeline.get("next_step") or "Set GRAPHRAG_ENABLED=true and restart the backend."
            ),
            mode=str(chat.get("mode") or operating_mode.get("chat_mode") or "local_pipeline_graph_context"),
            endpoint=upload.get("endpoint") or local_pipeline.get("endpoint") or document_pipeline.get("upload_endpoint"),
            secondary_endpoint=chat.get("endpoint") or chat_handoff.get("endpoint"),
            checks=[
                "stage samples",
                "evidence profile",
                "local graph artifact",
                "local context cache",
                "graph path context",
            ],
            outputs=[
                "pipeline_run_id",
                "interaction_handoff",
                "local_graph",
                "local_context",
                "retrieval_rehearsals",
            ],
            uses_local_pipeline=True,
            uses_remote_graph=bool(chat.get("uses_remote_graph")),
        ),
        _pathway_item(
            item_id="remote_graph_search",
            label="Remote Graph Ingest + Search",
            status=remote_status,
            recommended=recommended_remote,
            summary=(
                "Persist pipeline output to the remote GraphRAG API and use remote graph search/reasoning paths."
                if remote_ready
                else "Local traces are available, but remote graph persistence/search needs a reachable GraphRAG API."
                if local_ready
                else "Remote graph workflows need GraphRAG routes and a reachable GraphRAG API."
            ),
            action=(
                "Upload through the pipeline; adaptive ingest endpoint fallback will record every attempt."
                if remote_submission
                else upload.get("next_step") or remote_ingest.get("next_step") or "Set GRAPHRAG_URL to a reachable GraphRAG API."
            ),
            mode=str(upload.get("endpoint_strategy") or "remote_graph"),
            endpoint=upload.get("preferred_endpoint") or remote_ingest.get("endpoint"),
            secondary_endpoint=search.get("endpoint") or graph_search.get("endpoint"),
            checks=[
                str(upload.get("endpoint_verification") or ""),
                str(upload.get("health_url") or ""),
                *[str(field) for field in upload.get("attempt_trace_fields", []) or []],
            ],
            outputs=[
                "remote_graph_ingest",
                "graph_search",
                "relationship_paths",
                "remote_graph_api",
            ],
            uses_local_pipeline=local_ready,
            uses_remote_graph=remote_ready,
        ),
        _pathway_item(
            item_id="helper_extraction",
            label="Helper Extraction",
            status=helper_status,
            recommended=recommended_helpers,
            summary=(
                extraction.get("message")
                or "Use helper extraction services to strengthen entity and relationship extraction before or alongside remote graph ingest."
            ),
            action=extraction.get("next_step") or "Start graphrag-ner/graphrag-rel or configure GRAPHRAG_URL.",
            mode="helper_services",
            endpoint=extraction.get("endpoint"),
            checks=[
                "NER labels",
                "REL schema",
                "entity extraction",
                "relationship extraction",
            ],
            outputs=[
                "entity_candidates",
                "relationship_candidates",
                "extraction_diagnostics",
            ],
            uses_local_pipeline=local_ready,
            uses_remote_graph=bool(chat.get("uses_remote_graph")),
        ),
        _pathway_item(
            item_id="ocr_enrichment",
            label="OCR Enrichment",
            status=ocr_status,
            recommended=recommended_ocr,
            summary=(
                document_ocr.get("message")
                or "Enable OCR when uploads include scanned PDFs, screenshots, or packaged images."
            ),
            action=(
                "Upload scanned/image-heavy documents; OCR evidence will appear in pipeline traces."
                if ocr_ready
                else document_ocr.get("deployment_hint") or document_ocr.get("message") or "Set GRAPHRAG_ENABLE_IMAGE_OCR=true and install the OCR runtime."
            ),
            mode=str(document_ocr.get("status") or "ocr"),
            endpoint="/api/v1/graphrag/services/status",
            checks=[
                "Pillow import",
                "pytesseract import",
                "tesseract binary",
            ],
            outputs=[
                "page_ocr_text",
                "image_ocr_text",
                "image_contains_text",
            ],
            uses_local_pipeline=local_ready,
            uses_remote_graph=False,
        ),
    ]


def _services_deployment_quickstart(*, workflows: dict[str, dict], interaction_contract: dict, diagnostics: dict) -> list[dict]:
    enabled = bool(diagnostics.get("enabled"))
    local_pipeline = workflows.get("local_pipeline_preview", {})
    chat_handoff = workflows.get("pipeline_chat_handoff", {})
    remote_ingest = workflows.get("remote_ingest", {})
    graph_search = workflows.get("graph_search", {})
    extraction = workflows.get("extraction", {})
    upload = interaction_contract.get("upload") if isinstance(interaction_contract.get("upload"), dict) else {}
    deployment = interaction_contract.get("deployment") if isinstance(interaction_contract.get("deployment"), dict) else {}
    document_ocr = deployment.get("document_ocr") if isinstance(deployment.get("document_ocr"), dict) else _document_ocr_runtime_status()

    local_ready = bool(local_pipeline.get("ready") and chat_handoff.get("ready"))
    remote_ready = bool(remote_ingest.get("ready") and graph_search.get("ready"))
    extraction_ready = bool(extraction.get("ready"))
    ocr_ready = bool(document_ocr.get("ready"))
    ocr_enabled = bool(document_ocr.get("enabled") or document_ocr.get("configured"))

    if not enabled:
        local_status = "disabled"
    else:
        local_status = "ready" if local_ready else "blocked"

    if remote_ready:
        remote_status = "ready"
    elif enabled and local_ready:
        remote_status = "local"
    elif enabled:
        remote_status = "blocked"
    else:
        remote_status = "disabled"

    if extraction_ready:
        helper_status = _workflow_status(extraction)
    elif enabled:
        helper_status = "blocked"
    else:
        helper_status = "disabled"

    if ocr_ready:
        ocr_status = "ready"
    elif ocr_enabled:
        ocr_status = "blocked"
    else:
        ocr_status = "disabled"

    preferred_endpoint = upload.get("preferred_endpoint") or remote_ingest.get("endpoint")
    remote_command = (
        "GRAPHRAG_ENABLED=true GRAPHRAG_URL=http://host.docker.internal:18000 "
        "docker compose --profile graphrag up -d backend-api"
    )

    return [
        _quickstart_item(
            item_id="local_trace_chat",
            label="Local trace + Ask Chat",
            status=local_status,
            summary="Enable GraphRAG routes to upload documents, inspect stage samples, and chat over recent pipeline traces without a remote graph API.",
            command="GRAPHRAG_ENABLED=true docker compose up -d backend-api",
            env_vars=["GRAPHRAG_ENABLED=true"],
            checks=[
                "GET /api/v1/graphrag/services/status",
                "POST /api/v1/graphrag/ingest/pipeline/preview",
                "POST /api/v1/graphrag/chat-context-v2",
            ],
            endpoint="/api/v1/graphrag/ingest/pipeline/preview",
            next_step=None if local_ready else "Set GRAPHRAG_ENABLED=true and restart the backend.",
        ),
        _quickstart_item(
            item_id="helper_extraction",
            label="Helper extraction stack",
            status=helper_status,
            summary="Start the bundled helper services for stronger entity, relationship, and Code-RAG extraction while remote graph persistence is being wired up.",
            command="GRAPHRAG_ENABLED=true docker compose --profile graphrag up -d",
            env_vars=["GRAPHRAG_ENABLED=true"],
            profiles=["graphrag", "graphrag-ner", "graphrag-rel", "graphrag-code"],
            checks=[
                "GET /api/v1/graphrag/services/status",
                "GET http://localhost:8711/health",
                "GET http://localhost:8712/health",
                "GET http://localhost:8713/health",
            ],
            endpoint=extraction.get("endpoint"),
            next_step=None if extraction_ready else extraction.get("next_step") or "Start the graphrag helper profiles.",
        ),
        _quickstart_item(
            item_id="remote_graph",
            label="Remote GraphRAG ingest + search",
            status=remote_status,
            summary="Connect a reachable GraphRAG API so pipeline uploads can persist to the graph and search/reasoning routes can use remote paths.",
            command=remote_command,
            env_vars=["GRAPHRAG_ENABLED=true", "GRAPHRAG_URL=http://host.docker.internal:18000"],
            profiles=["graphrag"],
            checks=[
                "GET /api/v1/graphrag/services/status",
                "GET /api/v1/graphrag/health",
                "POST /api/v1/graphrag/ingest/pipeline/upload",
            ],
            endpoint=preferred_endpoint,
            next_step=None if remote_ready else upload.get("next_step") or remote_ingest.get("next_step"),
        ),
        _quickstart_item(
            item_id="document_ocr",
            label="Scanned image OCR",
            status=ocr_status,
            summary="Enable OCR when uploads include scanned PDFs, screenshots, or packaged document images that need text extraction.",
            command="GRAPHRAG_ENABLE_IMAGE_OCR=true docker compose up -d --build backend-api",
            env_vars=["GRAPHRAG_ENABLE_IMAGE_OCR=true"],
            checks=[
                "Pillow import",
                "pytesseract import",
                "tesseract binary",
            ],
            endpoint="/api/v1/graphrag/services/status",
            next_step=None if ocr_ready else document_ocr.get("message") or document_ocr.get("deployment_hint"),
        ),
    ]


def _services_status_meta(statuses: dict[str, dict], diagnostics: dict) -> dict:
    services = list(statuses.values())
    ready_services = [service for service in services if service.get("ready")]
    configured_services = [service for service in services if service.get("configured")]
    explicit_services = [service for service in services if service.get("explicit")]
    auto_discovered_services = [service for service in services if service.get("auto_discovered")]
    required_services = [service for service in services if not service.get("optional")]
    required_ready = bool(required_services) and all(service.get("ready") for service in required_services)
    blocking_services = [
        service
        for service in required_services
        if not service.get("ready")
    ]
    configured_unavailable = [
        service
        for service in services
        if service.get("configured") and not service.get("ready")
    ]

    next_steps: list[str] = []
    for service in [*blocking_services, *configured_unavailable]:
        command = service.get("deployment_hint") or service.get("deploy_command")
        required_env = service.get("required_env") or []
        if command:
            next_steps.append(command)
        elif required_env:
            next_steps.append(f"Set {required_env[0]} to a reachable service URL.")

    for warning in diagnostics.get("warnings", []) or []:
        if isinstance(warning, str) and warning not in next_steps:
            next_steps.append(warning)

    if not diagnostics.get("enabled"):
        next_steps.insert(0, "Set GRAPHRAG_ENABLED=true and restart the backend.")

    if required_ready:
        status = "ready"
    elif ready_services:
        status = "partial"
    elif diagnostics.get("status"):
        status = str(diagnostics["status"])
    else:
        status = "offline"

    workflows = _service_workflow_readiness(statuses, enabled=bool(diagnostics.get("enabled")))
    available_workflows = [workflow_id for workflow_id, workflow in workflows.items() if workflow.get("ready")]
    blocked_workflows = [workflow_id for workflow_id, workflow in workflows.items() if workflow.get("status") == "blocked"]
    operating_mode = _workflow_operating_mode(status=status, workflows=workflows, diagnostics=diagnostics)
    interaction_contract = _services_interaction_contract(
        statuses=statuses,
        workflows=workflows,
        operating_mode=operating_mode,
        diagnostics=diagnostics,
    )
    interaction_pathways = _services_interaction_pathways(
        workflows=workflows,
        interaction_contract=interaction_contract,
        operating_mode=operating_mode,
    )
    operator_checklist = _services_operator_checklist(
        workflows=workflows,
        interaction_contract=interaction_contract,
        operating_mode=operating_mode,
        diagnostics=diagnostics,
    )
    deployment_quickstart = _services_deployment_quickstart(
        workflows=workflows,
        interaction_contract=interaction_contract,
        diagnostics=diagnostics,
    )
    document_ocr = _document_ocr_runtime_status()

    if auto_discovered_services and explicit_services:
        source_summary = (
            f"Using {len(explicit_services)} explicit service URL(s) and "
            f"{len(auto_discovered_services)} Docker default helper URL(s)."
        )
    elif auto_discovered_services:
        source_summary = f"Using {len(auto_discovered_services)} Docker default helper URL(s)."
    elif explicit_services:
        source_summary = f"Using {len(explicit_services)} explicit service URL(s)."
    else:
        source_summary = "No GraphRAG service URLs are configured."

    return {
        "status": status,
        "ready": required_ready,
        "operational_ready": operating_mode["ready"],
        "local_ready": bool(workflows.get("local_pipeline_preview", {}).get("ready") and workflows.get("pipeline_chat_handoff", {}).get("ready")),
        "summary": diagnostics.get("summary") or "GraphRAG service readiness",
        "source_summary": source_summary,
        "url_sources": {
            name: {
                "source": service.get("url_source"),
                "label": service.get("url_source_label"),
                "url": service.get("url"),
                "configured": service.get("configured"),
                "auto_discovered": service.get("auto_discovered"),
                "explicit": service.get("explicit"),
            }
            for name, service in statuses.items()
        },
        "explicit_url_count": len(explicit_services),
        "auto_discovered_count": len(auto_discovered_services),
        "operating_mode": operating_mode,
        "ready_count": len(ready_services),
        "configured_count": len(configured_services),
        "service_count": len(services),
        "workflow_ready_count": len(available_workflows),
        "workflow_count": len(workflows),
        "required_ready": required_ready,
        "warnings": diagnostics.get("warnings", []),
        "next_steps": list(dict.fromkeys(next_steps))[:6],
        "commands": diagnostics.get("commands", {}),
        "document_ocr": document_ocr,
        "diagnostics_path": "/api/v1/graphrag/deployment/diagnostics",
        "workflows": workflows,
        "available_workflows": available_workflows,
        "blocked_workflows": blocked_workflows,
        "interaction_contract": interaction_contract,
        "interaction_pathways": interaction_pathways,
        "recommended_pathway_id": next(
            (
                pathway.get("id")
                for pathway in interaction_pathways
                if pathway.get("recommended")
            ),
            None,
        ),
        "operator_checklist": operator_checklist,
        "deployment_quickstart": deployment_quickstart,
    }


def _services_interaction_contract(
    *,
    statuses: dict[str, dict],
    workflows: dict[str, dict],
    operating_mode: dict,
    diagnostics: dict,
) -> dict:
    cfg = get_graphrag_config()
    enabled = bool(diagnostics.get("enabled"))
    graph_ready = bool(statuses.get("graphrag", {}).get("ready"))
    graph_url = cfg.graphrag_url
    graph_source = cfg.service_url_sources.get("graphrag", "unset")
    graph_service = statuses.get("graphrag", {}).get("service")
    graph_service = graph_service if isinstance(graph_service, dict) else {}
    health_urls = compatible_health_urls(graph_url)
    health_url = graph_service.get("health_url") or (health_urls[0] if health_urls else None)
    ingest_urls = compatible_ingest_urls(graph_url)
    local_pipeline = workflows.get("local_pipeline_preview", {})
    chat_handoff = workflows.get("pipeline_chat_handoff", {})
    remote_ingest = workflows.get("remote_ingest", {})
    graph_search = workflows.get("graph_search", {})

    upload_available = bool(local_pipeline.get("ready"))
    remote_submission_ready = bool(remote_ingest.get("ready"))
    chat_available = bool(chat_handoff.get("ready"))
    search_available = bool(graph_search.get("ready"))

    if not enabled:
        upload_behavior = "disabled"
        endpoint_strategy = "disabled"
    elif remote_submission_ready:
        upload_behavior = "local_trace_plus_adaptive_remote_ingest"
        endpoint_strategy = "adaptive_fallback"
    elif upload_available:
        upload_behavior = "local_trace_only"
        endpoint_strategy = "not_configured" if not graph_url else "health_blocked"
    else:
        upload_behavior = "unavailable"
        endpoint_strategy = "disabled" if not enabled else "blocked"

    if not enabled:
        endpoint_verification = "disabled"
    elif remote_submission_ready:
        endpoint_verification = "health_preflight_then_adaptive_upload_attempts"
    elif graph_url:
        endpoint_verification = "health_preflight_blocked"
    else:
        endpoint_verification = "not_configured"

    chat_sources = []
    if chat_available:
        chat_sources.extend([
            "pipeline_run_local_context",
            "local_graph_candidates",
            "graph_path_context",
        ])
    if graph_ready:
        chat_sources.append("remote_graph_api")

    attempt_trace_fields = [
        "upstream_ingest.sample_outputs.attempted_urls",
        "upstream_ingest.sample_outputs.endpoint_attempts",
        "upstream_ingest.selected_endpoint",
    ]
    document_pipeline = _local_document_pipeline_contract(cfg, enabled=enabled)
    document_ocr = _document_ocr_runtime_status(cfg)
    smoke_tests = [
        {
            "id": "llama_nexus_services_status",
            "label": "llama-nexus GraphRAG status contract",
            "scope": "llama_nexus",
            "method": "GET",
            "path": "/api/v1/graphrag/services/status",
            "ready": enabled,
            "status": "ready" if enabled else "disabled",
            "expected": "_meta.operational_ready and interaction_contract describe local/remote readiness.",
        },
        {
            "id": "local_pipeline_preview",
            "label": "Local pipeline preview",
            "scope": "llama_nexus",
            "method": "POST",
            "path": "/api/v1/graphrag/ingest/pipeline/preview",
            "ready": upload_available,
            "status": "local" if upload_available else "blocked",
            "expected": "Returns stages, evidence profile, local_graph, local_context, and interaction_handoff.",
        },
        {
            "id": "document_pipeline_contract",
            "label": "Document pipeline evidence contract",
            "scope": "llama_nexus",
            "method": "contract",
            "path": document_pipeline["endpoint"],
            "ready": document_pipeline["ready"],
            "status": document_pipeline["status"],
            "expected": (
                "Reports supported formats, stage outputs, evidence outputs, sample cap, "
                "and representative relationship sample mode."
            ),
            "trace_fields": [
                "summary.sample_limit",
                "samples.relationships",
                "samples.relationship_highlights",
                "document_profile.evidence_profile",
                "document_profile.evidence_profile.graph_evidence.link_relationship_count",
                "interaction_handoff",
                "interaction_handoff.graph_anchors",
                "interaction_handoff.citation_scopes",
            ],
        },
        {
            "id": "document_ocr_runtime",
            "label": "Document OCR runtime",
            "scope": "llama_nexus",
            "method": "runtime_check",
            "ready": document_ocr["ready"],
            "status": document_ocr["status"],
            "expected": "Set GRAPHRAG_ENABLE_IMAGE_OCR=true plus Pillow, pytesseract, and tesseract-ocr for OCR text extraction.",
        },
    ]
    if health_url:
        smoke_tests.append({
            "id": "remote_health_preflight",
            "label": "Remote GraphRAG health preflight",
            "scope": "remote_graphrag",
            "method": "GET",
            "url": health_url,
            "urls": health_urls,
            "command": _curl_command("GET", health_url),
            "ready": graph_ready,
            "status": "ready" if graph_ready else "blocked",
            "expected": "HTTP 200 with a non-unhealthy service status before remote upload.",
        })
    else:
        smoke_tests.append({
            "id": "remote_health_preflight",
            "label": "Remote GraphRAG health preflight",
            "scope": "remote_graphrag",
            "method": "GET",
            "ready": False,
            "status": "not_configured" if enabled else "disabled",
            "expected": "Set GRAPHRAG_URL or GRAPHRAG_SERVICE_URL to enable the remote health probe.",
        })
    if ingest_urls:
        smoke_tests.append({
            "id": "adaptive_remote_ingest",
            "label": "Adaptive remote ingest candidates",
            "scope": "remote_graphrag",
            "method": "POST",
            "urls": ingest_urls,
            "preferred_url": ingest_urls[0],
            "ready": remote_submission_ready,
            "status": "ready" if remote_submission_ready else "blocked",
            "expected": "Pipeline upload records every candidate attempt and selected endpoint.",
            "trace_fields": attempt_trace_fields,
        })

    return {
        "mode": operating_mode.get("id"),
        "label": operating_mode.get("label"),
        "summary": operating_mode.get("summary"),
        "upload": {
            "available": upload_available,
            "behavior": upload_behavior,
            "local_trace": upload_available,
            "remote_submission": remote_submission_ready,
            "endpoint_strategy": endpoint_strategy,
            "endpoint_verification": endpoint_verification,
                "health_url": health_url,
                "compatible_health_urls": health_urls,
                "preferred_endpoint": ingest_urls[0] if ingest_urls else None,
            "compatible_endpoints": ingest_urls,
            "compatible_endpoint_paths": list(GRAPHRAG_COMPATIBLE_INGEST_PATHS),
            "preflight": {
                "method": "GET",
                "url": health_url,
                "status": "required_before_remote_submit" if graph_url else "not_configured",
                "compatible_urls": health_urls,
                "trace_field": "upstream_preflight.sample_outputs.health_url",
            },
            "attempt_trace_fields": attempt_trace_fields,
            "smoke_tests": [
                test for test in smoke_tests
                if test.get("id") in {"remote_health_preflight", "adaptive_remote_ingest"}
            ],
            "endpoint": remote_ingest.get("endpoint"),
            "endpoint_source": graph_source,
            "endpoint_source_label": _service_url_source_label(graph_source),
            "endpoint_explicit": graph_source not in {"unset", "docker_default"},
            "next_step": None if remote_submission_ready else remote_ingest.get("next_step") or operating_mode.get("recommended_action"),
        },
        "chat": {
            "available": chat_available,
            "mode": operating_mode.get("chat_mode"),
            "uses_run_context": chat_available,
            "uses_remote_graph": graph_ready,
            "context_sources": chat_sources,
            "endpoint": chat_handoff.get("endpoint"),
            "next_step": None if chat_available else chat_handoff.get("next_step") or operating_mode.get("recommended_action"),
        },
        "search": {
            "available": search_available,
            "uses_remote_graph": graph_ready,
            "local_pipeline_available": chat_available,
            "endpoint": graph_search.get("endpoint"),
            "fallback": None if search_available else "local_pipeline_search" if chat_available else None,
            "fallback_endpoint": "/api/v1/graphrag/search" if chat_available and not search_available else None,
            "next_step": None if search_available else graph_search.get("next_step") or operating_mode.get("recommended_action"),
        },
        "document_pipeline": document_pipeline,
        "deployment": {
            "enabled": enabled,
            "status": diagnostics.get("status"),
            "recommended_action": operating_mode.get("recommended_action"),
            "diagnostics_path": "/api/v1/graphrag/deployment/diagnostics",
            "services_status_path": "/api/v1/graphrag/services/status",
            "url_sources": cfg.service_url_sources,
            "auto_discovered_service_count": cfg.auto_discovered_service_count,
            "document_ocr": document_ocr,
            "smoke_tests": smoke_tests,
        },
    }


def _checklist_severity(status: str) -> str:
    if status == "ready":
        return "success"
    if status in {"local", "partial"}:
        return "info" if status == "local" else "warning"
    if status in {"disabled", "not_configured"}:
        return "info"
    return "error"


def _operator_step(
    *,
    step_id: str,
    label: str,
    status: str,
    detail: str,
    action: str | None = None,
    endpoint: str | None = None,
    mode: str | None = None,
    checks: list[str] | None = None,
    workflow_id: str | None = None,
) -> dict:
    status = status or "blocked"
    return {
        "id": step_id,
        "label": label,
        "status": status,
        "ready": status in {"ready", "local", "partial"},
        "severity": _checklist_severity(status),
        "detail": detail,
        "action": action,
        "endpoint": endpoint,
        "mode": mode,
        "workflow_id": workflow_id,
        "checks": [check for check in (checks or []) if check],
    }


def _workflow_status(workflow: dict | None) -> str:
    if not isinstance(workflow, dict):
        return "blocked"
    return str(workflow.get("status") or ("ready" if workflow.get("ready") else "blocked"))


def _curl_command(method: str, url: str | None) -> str | None:
    if not url or not str(url).startswith(("http://", "https://")):
        return None
    method = method.upper()
    if method == "GET":
        return f"curl -fsS {shlex.quote(url)}"
    return f"curl -fsS -X {method} {shlex.quote(url)}"


def _services_operator_checklist(
    *,
    workflows: dict[str, dict],
    interaction_contract: dict,
    operating_mode: dict,
    diagnostics: dict,
) -> list[dict]:
    enabled = bool(diagnostics.get("enabled"))
    upload = interaction_contract.get("upload") if isinstance(interaction_contract.get("upload"), dict) else {}
    chat = interaction_contract.get("chat") if isinstance(interaction_contract.get("chat"), dict) else {}
    search = interaction_contract.get("search") if isinstance(interaction_contract.get("search"), dict) else {}
    deployment = interaction_contract.get("deployment") if isinstance(interaction_contract.get("deployment"), dict) else {}

    local_pipeline = workflows.get("local_pipeline_preview", {})
    chat_handoff = workflows.get("pipeline_chat_handoff", {})
    remote_ingest = workflows.get("remote_ingest", {})
    graph_search = workflows.get("graph_search", {})
    extraction = workflows.get("extraction", {})

    upload_available = bool(upload.get("available"))
    remote_submission = bool(upload.get("remote_submission"))
    chat_available = bool(chat.get("available"))
    search_available = bool(search.get("available"))
    search_fallback = str(search.get("fallback") or "")

    enable_status = "ready" if enabled else "disabled"
    remote_status = "ready" if remote_submission else "local" if upload_available else "blocked"
    search_status = "ready" if search_available else "local" if search_fallback else "blocked"

    return [
        _operator_step(
            step_id="enable_graphrag_routes",
            label="Enable GraphRAG routes",
            status=enable_status,
            detail=(
                "GraphRAG routes are enabled in the backend."
                if enabled
                else "GraphRAG routes are disabled, so upload traces and chat handoffs are unavailable."
            ),
            action=None if enabled else "Set GRAPHRAG_ENABLED=true and restart the backend.",
            endpoint="/api/v1/graphrag/services/status",
            mode=str(operating_mode.get("id") or deployment.get("status") or ""),
            checks=[
                f"mode: {operating_mode.get('label') or operating_mode.get('id')}",
                str(deployment.get("diagnostics_path") or ""),
            ],
        ),
        _operator_step(
            step_id="preview_upload_trace",
            label="Preview and trace uploads",
            status=_workflow_status(local_pipeline),
            detail=(
                "Local extraction, chunking, graph candidates, evidence profiles, handoff lanes, and retrieval rehearsals are ready."
                if upload_available
                else "Document pipeline preview is not available yet."
            ),
            action=(
                "Use Pipeline Preview or Pipeline Upload to create a durable run trace."
                if upload_available
                else local_pipeline.get("next_step")
            ),
            endpoint=local_pipeline.get("endpoint") or upload.get("endpoint"),
            mode=str(upload.get("behavior") or ""),
            workflow_id="local_pipeline_preview",
            checks=[
                "stage samples",
                "local graph/context artifacts",
                "retrieval rehearsals",
            ],
        ),
        _operator_step(
            step_id="ask_chat_from_trace",
            label="Ask Chat from pipeline traces",
            status=_workflow_status(chat_handoff),
            detail=(
                "Chat can use pipeline_run_ids, local graph anchors, graph path context, cached chunks, and handoff rehearsals."
                if chat_available
                else "Chat cannot consume pipeline traces until GraphRAG routes are enabled."
            ),
            action=(
                "Open Ask Chat from the trace or pass pipeline_run_ids to chat-context-v2."
                if chat_available
                else chat.get("next_step") or chat_handoff.get("next_step")
            ),
            endpoint=chat_handoff.get("endpoint") or chat.get("endpoint"),
            mode=str(chat.get("mode") or ""),
            workflow_id="pipeline_chat_handoff",
            checks=[
                *[str(source) for source in chat.get("context_sources", []) or []],
            ],
        ),
        _operator_step(
            step_id="remote_graph_ingest",
            label="Submit to remote GraphRAG",
            status=remote_status,
            detail=(
                "Pipeline upload will use adaptive remote ingest endpoint fallback."
                if remote_submission
                else "Uploads still produce local traces; remote graph persistence needs a reachable GraphRAG API."
            ),
            action=(
                "Upload through the pipeline; llama-nexus will try compatible ingest endpoints."
                if remote_submission
                else upload.get("next_step") or remote_ingest.get("next_step")
            ),
            endpoint=upload.get("preferred_endpoint") or remote_ingest.get("endpoint"),
            mode=str(upload.get("endpoint_strategy") or ""),
            workflow_id="remote_ingest",
            checks=[
                f"verification: {upload.get('endpoint_verification')}" if upload.get("endpoint_verification") else "",
                f"health preflight: {upload.get('health_url')}" if upload.get("health_url") else "",
                *[str(path) for path in upload.get("compatible_endpoint_paths", []) or []],
                *[str(field) for field in upload.get("attempt_trace_fields", []) or []],
            ],
        ),
        _operator_step(
            step_id="graph_search_reasoning",
            label="Graph search and reasoning",
            status=search_status,
            detail=(
                "Remote graph search and reasoning are available."
                if search_available
                else "Remote graph search is offline; search and dedicated reasoning routes can use recent local pipeline traces as the fallback."
                if search_fallback
                else "Graph search needs a reachable GraphRAG API."
            ),
            action=(
                "Use Intelligent Search or graph reasoning routes."
                if search_available
                else "Search or reason over recent pipeline traces, or Ask Chat from a trace."
                if search_fallback
                else search.get("next_step") or graph_search.get("next_step")
            ),
            endpoint=search.get("endpoint") or search.get("fallback_endpoint") or graph_search.get("endpoint"),
            mode=str(search.get("fallback") or "remote_graph"),
            workflow_id="graph_search",
            checks=[
                "relationship paths",
                "bridge entities",
                "provenance-aware graph context",
            ],
        ),
        _operator_step(
            step_id="helper_extraction",
            label="Helper extraction services",
            status=_workflow_status(extraction),
            detail=(
                extraction.get("message")
                or "NER/REL helpers improve entity and relationship extraction when they are reachable."
            ),
            action=extraction.get("next_step"),
            endpoint=extraction.get("endpoint"),
            mode="helper_services",
            workflow_id="extraction",
            checks=[
                "NER labels",
                "REL schema",
                "fallback through GraphRAG API",
            ],
        ),
    ]


def _service_workflow_readiness(statuses: dict[str, dict], *, enabled: bool = True) -> dict[str, dict]:
    def ready(service_name: str) -> bool:
        return bool(statuses.get(service_name, {}).get("ready"))

    def configured(service_name: str) -> bool:
        return bool(statuses.get(service_name, {}).get("configured"))

    graph_ready = ready("graphrag")
    ner_ready = ready("ner")
    rel_ready = ready("rel")
    code_ready = ready("code_rag")
    reddit_ready = ready("reddit_crawler")
    extraction_ready = graph_ready or ner_ready or rel_ready
    code_workflow_ready = graph_ready or code_ready

    def workflow(
        *,
        workflow_id: str,
        label: str,
        is_ready: bool,
        depends_on: list[str],
        endpoint: str,
        ready_message: str,
        blocked_message: str,
        next_step: str,
        local: bool = False,
        partial: bool = False,
    ) -> dict:
        status = "ready" if is_ready else "blocked"
        if local and is_ready:
            status = "local"
        elif partial and is_ready:
            status = "partial"
        return {
            "id": workflow_id,
            "label": label,
            "ready": is_ready,
            "status": status,
            "depends_on": depends_on,
            "endpoint": endpoint,
            "message": ready_message if is_ready else blocked_message,
            "next_step": None if is_ready else next_step,
        }

    return {
        "local_pipeline_preview": workflow(
            workflow_id="local_pipeline_preview",
            label="Local document pipeline preview",
            is_ready=enabled,
            depends_on=["backend"],
            endpoint="/api/v1/graphrag/ingest/pipeline/preview",
            ready_message="Local extraction, chunking, evidence profiling, and graph-candidate previews are available without a remote GraphRAG API.",
            blocked_message="The backend is not serving GraphRAG routes.",
            next_step="Start the backend and open the GraphRAG Workbench.",
            local=True,
        ),
        "pipeline_chat_handoff": workflow(
            workflow_id="pipeline_chat_handoff",
            label="Run-scoped upload to Chat",
            is_ready=enabled,
            depends_on=["backend"],
            endpoint="/api/v1/graphrag/chat-context-v2",
            ready_message="Chat can use recent pipeline run IDs, local graph candidates, and cached chunk context before remote GraphRAG ingest is reachable.",
            blocked_message="The backend is not serving GraphRAG chat context routes.",
            next_step="Upload documents through the GraphRAG pipeline, then use Ask Chat from the trace.",
            local=True,
        ),
        "remote_ingest": workflow(
            workflow_id="remote_ingest",
            label="Remote GraphRAG ingest",
            is_ready=graph_ready,
            depends_on=["graphrag"],
            endpoint="/api/v1/graphrag/ingest/pipeline/upload",
            ready_message="Pipeline upload can submit structured payloads to the configured GraphRAG API.",
            blocked_message="Pipeline upload can still produce a local trace, but remote submission needs a reachable GRAPHRAG_URL.",
            next_step="Set GRAPHRAG_URL to a reachable GraphRAG API and confirm /api/v1/graphrag/health.",
        ),
        "graph_search": workflow(
            workflow_id="graph_search",
            label="Graph search and reasoning",
            is_ready=graph_ready,
            depends_on=["graphrag"],
            endpoint="/api/v1/graphrag/search/intelligent",
            ready_message="Graph search, graph exploration, chat graph reasoning, and upstream reasoning routes can use the graph API.",
            blocked_message="Graph search and reasoning need a reachable GRAPHRAG_URL.",
            next_step="Start or connect the GraphRAG API service.",
        ),
        "extraction": workflow(
            workflow_id="extraction",
            label="Entity and relationship extraction",
            is_ready=extraction_ready,
            depends_on=["ner", "rel", "graphrag"],
            endpoint="/api/v1/graphrag/ner/extract",
            ready_message=(
                "Dedicated NER/REL extraction is reachable."
                if ner_ready or rel_ready
                else "Extraction can fall back through the configured GraphRAG API."
            ),
            blocked_message="Entity and relationship extraction need NER_API_URL, REL_API_URL, or GRAPHRAG_URL to be reachable.",
            next_step="Start the graphrag-ner/graphrag-rel profiles or configure GRAPHRAG_URL.",
            partial=bool((ner_ready or rel_ready) and not graph_ready),
        ),
        "code_rag": workflow(
            workflow_id="code_rag",
            label="Code-RAG search and analysis",
            is_ready=code_workflow_ready,
            depends_on=["code_rag", "graphrag"],
            endpoint="/api/v1/graphrag/code/search",
            ready_message=(
                "Dedicated Code-RAG is reachable."
                if code_ready
                else "Code routes can fall back through the configured GraphRAG API."
            ),
            blocked_message="Code search and analysis need CODE_RAG_URL or GRAPHRAG_URL to be reachable.",
            next_step="Start the graphrag-code profile or configure GRAPHRAG_URL.",
            partial=bool(code_ready and not graph_ready),
        ),
        "reddit_crawler": workflow(
            workflow_id="reddit_crawler",
            label="Reddit crawl ingestion",
            is_ready=reddit_ready,
            depends_on=["reddit_crawler"],
            endpoint="/api/v1/graphrag/reddit/crawl",
            ready_message="Reddit crawling routes can use the crawler service.",
            blocked_message=(
                "Reddit crawler URLs are configured but not reachable."
                if configured("reddit_crawler")
                else "Reddit crawling is optional and not configured."
            ),
            next_step="Start the reddit-crawler profile or set REDDIT_CRAWLER_URL.",
        ),
    }


async def _optional_json(url: str, path: str, timeout: float = 5.0) -> dict:
    """Fetch optional service metadata without turning health checks into failures."""
    try:
        client = get_http_client()
        response = await client.get(f"{url.rstrip('/')}/{path.lstrip('/')}", timeout=timeout)
        if response.status_code == 200:
            payload = response.json()
            return payload if isinstance(payload, dict) else {}
    except Exception as e:
        logger.debug(f"Optional GraphRAG service metadata failed for {url}/{path}: {e}")
    return {}


@router.get("/health")
async def graphrag_health():
    """Check GraphRAG system health and service availability."""
    cfg = get_graphrag_config()
    document_ocr = _document_ocr_runtime_status(cfg)

    result = {
        "graphrag_enabled": cfg.enabled,
        "status": "disabled",
        "graph_available": False,
        "extraction_available": False,
        "code_rag_available": False,
        "reddit_crawler_available": False,
        "url_sources": cfg.service_url_sources,
        "auto_discovered_service_count": cfg.auto_discovered_service_count,
        "local_pipeline_available": cfg.enabled,
        "document_ocr_available": document_ocr["ready"],
        "document_ocr": document_ocr,
        "services": {},
    }

    if cfg.enabled:
        result["services"]["graphrag"] = await _check_service(cfg.graphrag_url, "graphrag")
        result["services"]["ner"] = await _check_service(cfg.ner_api_url, "ner")
        result["services"]["rel"] = await _check_service(cfg.rel_api_url, "rel")
        result["services"]["code_rag"] = await _check_service(cfg.code_rag_url, "code_rag")
        result["services"]["reddit_crawler"] = await _check_service(cfg.reddit_crawler_url, "reddit_crawler")

        # Check Neo4j connectivity
        result["services"]["neo4j"] = {
            "configured": cfg.neo4j_uri is not None,
            "uri": cfg.neo4j_uri,
        }

        result["deployment_hints"] = {
            name: descriptor["deploy_command"]
            for name, descriptor in SERVICE_DESCRIPTORS.items()
            if descriptor.get("deploy_command")
        }

        result["graph_available"] = bool(result["services"]["graphrag"].get("available"))
        result["extraction_available"] = bool(
            result["services"]["ner"].get("available") or result["services"]["rel"].get("available")
        )
        result["code_rag_available"] = bool(result["services"]["code_rag"].get("available"))
        result["reddit_crawler_available"] = bool(result["services"]["reddit_crawler"].get("available"))

        if result["graph_available"]:
            result["status"] = "healthy"
        elif any(
            [
                result["extraction_available"],
                result["code_rag_available"],
                result["reddit_crawler_available"],
            ]
        ):
            result["status"] = "partial"
            result["message"] = "GraphRAG helper services are reachable, but GRAPHRAG_URL is not reachable/configured"
        elif cfg.configured_service_count:
            result["status"] = "unavailable"
            result["message"] = "GraphRAG is enabled, but no configured GraphRAG services are reachable"
        else:
            result["status"] = "misconfigured"
            result["message"] = "GraphRAG is enabled, but no GraphRAG service URLs are configured"

    return result


@router.get("/services/status")
async def graphrag_services_status():
    """Get detailed status of all GraphRAG services (for UI display)."""
    cfg = get_graphrag_config()
    service_urls = {
        "graphrag": cfg.graphrag_url,
        "ner": cfg.ner_api_url,
        "rel": cfg.rel_api_url,
        "code_rag": cfg.code_rag_url,
        "reddit_crawler": cfg.reddit_crawler_url,
    }

    if cfg.enabled:
        checks = dict(zip(
            service_urls.keys(),
            await asyncio.gather(*(_check_service(url, name) for name, url in service_urls.items())),
        ))
    else:
        checks = {
            name: {
                "available": False,
                "configured": bool(url),
                "status": "disabled",
                "url": url,
                **_service_source_metadata(name),
                "message": "GraphRAG service is disabled. Set GRAPHRAG_ENABLED=true",
            }
            for name, url in service_urls.items()
        }

    statuses = {}
    for name, url in service_urls.items():
        descriptor = _service_descriptor(name)
        health = checks[name]
        configured = bool(url)
        available = bool(health.get("available"))
        statuses[name] = {
            "enabled": configured,
            "configured": configured,
            "available": available,
            "ready": bool(cfg.enabled and available),
            "status": health.get("status", "unavailable"),
            "url": url,
            **_service_source_metadata(name),
            "deploy_command": descriptor["deploy_command"],
            "deployment_hint": descriptor["deploy_command"],
            "required_env": descriptor["required_env"],
            "description": descriptor["description"],
            "optional": descriptor["optional"],
            "service": health,
        }

    statuses[SERVICE_STATUS_META_KEY] = _services_status_meta(statuses, _build_deployment_diagnostics())
    return statuses


@router.get("/stats")
async def graphrag_get_stats(domain: str | None = None):
    """Get knowledge graph statistics from graphrag service."""
    check_graphrag_enabled()
    cfg = get_graphrag_config()

    if not cfg.graphrag_url:
        return {"available": False, "error": "GRAPHRAG_URL not configured", "nodes": 0, "edges": 0, "communities": 0}

    try:
        client = get_http_client()
        params = {"domain": domain} if domain else None
        response = await client.get(f"{cfg.graphrag_url}/knowledge-graph/stats", params=params, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/domains")
async def graphrag_get_domains():
    """Get available domains from graphrag service."""
    check_graphrag_enabled()
    cfg = get_graphrag_config()

    if not cfg.graphrag_url:
        return {"domains": []}

    try:
        client = get_http_client()
        response = await client.get(f"{cfg.graphrag_url}/knowledge-graph/domains", timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")


@router.get("/ner/status")
async def graphrag_ner_status():
    """Get NER (Named Entity Recognition) service status."""
    check_graphrag_enabled()
    cfg = get_graphrag_config()

    # Check NER directly if available
    if cfg.ner_api_url:
        health = await _check_service(cfg.ner_api_url, "ner")
        model_payload = await _optional_json(cfg.ner_api_url, "model-info") if health.get("available") else {}
        model_info = model_payload.get("model_info", model_payload)
        return {
            "ner_available": bool(health.get("available")),
            "status": health.get("status", "unavailable"),
            "service": health,
            "model_info": model_info if isinstance(model_info, dict) else {},
            "extraction_method": "dedicated_ner_service",
        }

    # Fallback to legacy service
    if cfg.graphrag_url:
        try:
            client = get_http_client()
            response = await client.get(f"{cfg.graphrag_url}/ner/status", timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"GraphRAG service error: {str(e)}")

    return {
        "ner_available": False,
        "status": "not_configured",
        "message": "Set NER_API_URL or GRAPHRAG_URL",
        "model_info": {},
    }


@router.get("/code/health")
async def graphrag_code_health():
    """Check code search service health."""
    cfg = get_graphrag_config()

    if not cfg.enabled:
        return {
            "available": False,
            "status": "disabled",
            "message": "GraphRAG service is disabled. Set GRAPHRAG_ENABLED=true",
        }

    # Check Code-RAG directly if available
    if cfg.code_rag_url:
        return await _check_service(cfg.code_rag_url, "code_rag")

    # Fallback to legacy service
    if cfg.graphrag_url:
        try:
            client = get_http_client()
            response = await client.get(f"{cfg.graphrag_url}/code-detection/health", timeout=30.0)
            return response.json()
        except Exception as e:
            return {"available": False, "error": str(e)}

    return {"available": False, "message": "Set CODE_RAG_URL or GRAPHRAG_URL"}


@router.get("/hybrid/status")
async def graphrag_hybrid_status():
    """Get status of GraphRAG, Code RAG, and hybrid availability."""
    cfg = get_graphrag_config()

    if not cfg.enabled:
        return _offline_hybrid_status()

    if cfg.graphrag_url:
        try:
            client = get_http_client()
            response = await client.get(f"{cfg.graphrag_url}/hybrid/status", timeout=10.0)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.debug(f"GraphRAG hybrid status fallback triggered: {e}")

    graphrag = await _check_service(cfg.graphrag_url, "graphrag")
    code_rag = await _check_service(cfg.code_rag_url, "code_rag")
    ner = await _check_service(cfg.ner_api_url, "ner")
    rel = await _check_service(cfg.rel_api_url, "rel")

    return {
        "graphrag_healthy": bool(graphrag.get("available")),
        "code_rag_healthy": bool(code_rag.get("available")),
        "hybrid_available": bool(graphrag.get("available") and code_rag.get("available")),
        "ner_available": bool(ner.get("available")),
        "rel_available": bool(rel.get("available")),
        "services": {
            "graphrag": graphrag,
            "code_rag": code_rag,
            "ner": ner,
            "rel": rel,
        },
    }


@router.get("/extraction-stats")
async def graphrag_extraction_stats():
    """Get entity extraction method statistics."""
    cfg = get_graphrag_config()

    if not cfg.enabled:
        return _default_extraction_stats()

    if not cfg.graphrag_url:
        return _default_extraction_stats("GRAPHRAG_URL not configured")

    try:
        client = get_http_client()
        response = await client.get(f"{cfg.graphrag_url}/extraction-stats", timeout=10.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Extraction stats error: {e}")
        return _default_extraction_stats(f"GraphRAG service error: {str(e)}")


@router.get("/supported-formats")
async def graphrag_supported_formats():
    """Get supported file formats and features."""
    cfg = get_graphrag_config()

    if not cfg.enabled:
        return _default_supported_formats(cfg)

    if not cfg.graphrag_url:
        return _default_supported_formats(cfg, "GRAPHRAG_URL not configured")

    try:
        client = get_http_client()
        response = await client.get(f"{cfg.graphrag_url}/supported-formats", timeout=10.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Supported formats error: {e}")
        return _default_supported_formats(cfg, f"GraphRAG service error: {str(e)}")
