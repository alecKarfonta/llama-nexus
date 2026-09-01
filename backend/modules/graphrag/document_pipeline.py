"""
GraphRAG document ingestion pipeline preview.

This module keeps local document understanding deliberately dependency-light.
It extracts structured elements from common document formats, produces chunk
samples, and builds graph-construction candidates before upstream GraphRAG
ingestion. Optional libraries are used when available; archive/XML fallbacks
keep Office formats useful even in lean deployments.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
from email import policy
from email.header import decode_header, make_header
from email.message import Message
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime
import hashlib
import html
from html.parser import HTMLParser
import io
import json
import mimetypes
import os
import posixpath
from pathlib import Path
import re
import time
from typing import Any
from uuid import uuid4
import xml.etree.ElementTree as ET
import zipfile


MAX_SAMPLE_TEXT = 900
DEFAULT_CHUNK_CHARS = 1600
DEFAULT_CHUNK_OVERLAP = 180
MAX_STRUCTURED_INGEST_CHARS = 500_000
MAX_EMBEDDED_FILE_PARSE_BYTES = 2_000_000
MAX_ARCHIVE_ENTRIES = 80
MAX_ARCHIVE_ENTRY_PARSE_BYTES = 2_000_000
MAX_LOCAL_GRAPH_RELATIONSHIPS = 500
MAX_LOCAL_GRAPH_REASONING_BRIDGES = 24
MAX_LOCAL_GRAPH_REASONING_PATHS = 48
MAX_LOCAL_GRAPH_REASONING_RELATIONSHIPS = 8
MAX_LOCAL_CONTEXT_CHUNKS = 120
MAX_INTERACTION_SUGGESTIONS = 10
MAX_INTERACTION_HANDOFF_CARDS = 16
MAX_INTERACTION_HANDOFF_ANCHORS = 8
MAX_CHUNK_GRAPH_CONTEXT_RELATIONSHIPS = 8
MAX_CHUNK_GRAPH_CONTEXT_CHARS = 1200
MAX_CHUNK_GRAPH_PATH_CONTEXT = 4
MAX_CHUNK_GRAPH_PATH_CONTEXT_CHARS = 900
MAX_SEMANTIC_CHUNK_TEXT_CHARS = 3600
MAX_VISUAL_LAYOUT_RELATIONSHIPS = 24
MAX_VISUAL_LAYOUT_RELATIONSHIPS_PER_SLIDE = 8
MAX_VISUAL_LAYOUT_NEIGHBORS_PER_ELEMENT = 1
VISUAL_LAYOUT_NEAR_DISTANCE_PCT = 24.0
PPTX_EMUS_PER_INCH = 914400
PDF_POINTS_PER_INCH = 72
STRUCTURED_SUBMIT_METHODS = {
    "archive_bundle",
    "csv_table",
    "docx_xml",
    "email_message",
    "epub_html",
    "fallback_text",
    "html_structure",
    "image_file",
    "json_tree",
    "legacy_office_ole",
    "odp_xml",
    "ods_xml",
    "odt_xml",
    "pdf_blocks",
    "pptx_xml",
    "rtf_text",
    "xlsx_xml",
}

OOXML_WORD_EXTENSIONS = {".docx", ".docm", ".dotx", ".dotm"}
OOXML_PRESENTATION_EXTENSIONS = {".pptx", ".pptm", ".potx", ".potm", ".ppsx", ".ppsm"}
OOXML_SPREADSHEET_EXTENSIONS = {".xlsx", ".xlsm", ".xltx", ".xltm"}
LEGACY_OFFICE_EXTENSIONS = {".doc", ".ppt", ".xls"}

LEGACY_OFFICE_MAX_STREAMS = 32
LEGACY_OFFICE_MAX_STREAM_BYTES = 1_000_000
LEGACY_OFFICE_MAX_STRINGS_PER_STREAM = 80
LEGACY_OFFICE_MAX_TEXT_CHARS = 1_800

OOXML_WORD_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
    "application/vnd.ms-word.document.macroenabled.12",
    "application/vnd.ms-word.template.macroenabled.12",
}
OOXML_PRESENTATION_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
    "application/vnd.openxmlformats-officedocument.presentationml.template",
    "application/vnd.ms-powerpoint.presentation.macroenabled.12",
    "application/vnd.ms-powerpoint.slideshow.macroenabled.12",
    "application/vnd.ms-powerpoint.template.macroenabled.12",
}
OOXML_SPREADSHEET_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.template",
    "application/vnd.ms-excel.sheet.macroenabled.12",
    "application/vnd.ms-excel.template.macroenabled.12",
}
LEGACY_OFFICE_MIME_TYPES = {
    "application/msword",
    "application/vnd.ms-word",
    "application/vnd.ms-powerpoint",
    "application/mspowerpoint",
    "application/powerpoint",
    "application/x-mspowerpoint",
    "application/vnd.ms-excel",
    "application/msexcel",
    "application/x-msexcel",
}

CAPABILITY_ORDER = (
    "metadata",
    "messages",
    "archives",
    "legacy_office",
    "outlines",
    "sections",
    "layout",
    "presentation_structure",
    "workbook_structure",
    "tables",
    "pivot_tables",
    "formulas",
    "named_ranges",
    "conditional_formats",
    "charts",
    "diagrams",
    "shapes",
    "connectors",
    "code_blocks",
    "images",
    "ocr",
    "media",
    "embedded_objects",
    "links",
    "forms",
    "comments",
    "annotations",
    "revisions",
    "speaker_notes",
    "notes",
    "headers",
    "footers",
    "styled_text",
)

CAPABILITY_BY_ELEMENT_TYPE = {
    "annotation": "annotations",
    "archive_entry": "archives",
    "chart_axis": "charts",
    "chart_data_label": "charts",
    "chart_legend": "charts",
    "chart_point": "charts",
    "chart_series": "charts",
    "chart_trendline": "charts",
    "conditional_format": "conditional_formats",
    "comment": "comments",
    "code_block": "code_blocks",
    "diagram_node": "diagrams",
    "diagram_relationship": "diagrams",
    "document_metadata": "metadata",
    "email_body": "messages",
    "embedded_object": "embedded_objects",
    "endnote": "notes",
    "footer": "footers",
    "footnote": "notes",
    "formula": "formulas",
    "form_field": "forms",
    "header": "headers",
    "heading": "sections",
    "image": "images",
    "link": "links",
    "legacy_office_stream": "legacy_office",
    "legacy_office_text": "legacy_office",
    "media": "media",
    "outline_item": "outlines",
    "image_ocr_text": "ocr",
    "page_ocr_text": "ocr",
    "pivot_table": "pivot_tables",
    "revision": "revisions",
    "sheet_metadata": "workbook_structure",
    "shape_metadata": "shapes",
    "shape_group": "shapes",
    "shape_connector": "connectors",
    "slide_metadata": "presentation_structure",
    "slide_title": "sections",
    "speaker_notes": "speaker_notes",
    "table_header": "tables",
    "table_cell": "tables",
    "table_definition": "tables",
    "table_row": "tables",
    "text_box": "shapes",
    "named_range": "named_ranges",
}

COMPLEX_CAPABILITIES = {"charts", "diagrams", "embedded_objects", "forms", "formulas", "media", "pivot_tables", "revisions"}
RICH_CAPABILITIES = {
    "annotations",
    "archives",
    "comments",
    "code_blocks",
    "connectors",
    "images",
    "legacy_office",
    "layout",
    "links",
    "messages",
    "ocr",
    "presentation_structure",
    "shapes",
    "speaker_notes",
    "tables",
    "workbook_structure",
}

VISUAL_LAYOUT_GRAPH_ELEMENT_TYPES = {
    "chart_axis",
    "chart_data_label",
    "chart_legend",
    "chart_series",
    "chart_trendline",
    "form_field",
    "image",
    "shape_group",
    "shape_metadata",
    "slide_text",
    "table_row",
    "text_box",
}

GRAPH_CONTEXT_PROVENANCE_KEYS = (
    "archive_entry_path",
    "cell",
    "email_message_id",
    "email_subject",
    "email_from",
    "email_sender",
    "email_to",
    "email_cc",
    "email_date",
    "email_part_index",
    "email_content_type",
    "email_attachment_index",
    "email_attachment_filename",
    "email_attachment_mime",
    "legacy_office_format",
    "legacy_office_stream_index",
    "legacy_office_stream_path",
    "legacy_office_stream_size_bytes",
    "legacy_office_stream_sha256",
    "legacy_office_extraction_method",
    "legacy_office_string_count",
    "chart_part",
    "chart_source_part",
    "chart_source_extraction_method",
    "chart_source_index",
    "chart_axis_type",
    "chart_axis_title",
    "chart_axis_number_format",
    "chart_legend_position",
    "chart_label_fields",
    "chart_label_position",
    "chart_label_number_format",
    "chart_series_fill_color",
    "chart_series_line_color",
    "chart_series_line_dash",
    "chart_series_visual_summary",
    "chart_marker_symbol",
    "chart_marker_fill_color",
    "chart_marker_line_color",
    "chart_point_fill_color",
    "chart_point_visual_summary",
    "chart_trendline_type",
    "chart_trendline_line_color",
    "chart_trendline_line_dash",
    "chart_trendline_visual_summary",
    "comment_anchor_text",
    "comment_anchor_section",
    "comment_anchor_paragraph",
    "comment_anchor_sheet",
    "comment_anchor_cell",
    "comment_anchor_header",
    "comment_anchor_value",
    "comment_kind",
    "threaded_comment_id",
    "threaded_parent_id",
    "person_id",
    "comment_part",
    "conditional_range",
    "conditional_type",
    "conditional_formula1",
    "conditional_formula_references",
    "formula_cell",
    "formula_expression",
    "formula_references",
    "formula_result",
    "formula_reference_count",
    "merge_range",
    "merge_anchor_cell",
    "named_range_name",
    "named_range_ref",
    "named_range_scope",
    "pivot_table_name",
    "pivot_table_range",
    "pivot_table_source_name",
    "pivot_table_source_ref",
    "pivot_table_source_sheet",
    "pivot_table_row_fields",
    "pivot_table_column_fields",
    "pivot_table_page_fields",
    "pivot_table_data_fields",
    "table_display_name",
    "table_range",
    "table_part",
    "table_columns",
    "table_cell_reference",
    "table_cell_column_header",
    "table_cell_row_header",
    "table_cell_context",
    "table_cell_grid_span",
    "table_cell_row_span",
    "table_cell_h_merge",
    "table_cell_v_merge",
    "table_cell_merge_flags",
    "table_cell_fill_color",
    "table_cell_visual_summary",
    "table_cell_style_name",
    "table_cell_text_anchor",
    "table_cell_vertical_align",
    "table_cell_text_direction",
    "table_cell_width_value",
    "table_cell_width_inches",
    "table_cell_width_type",
    "table_cell_shading_pattern",
    "table_cell_margin_left_inches",
    "table_cell_margin_right_inches",
    "table_cell_margin_top_inches",
    "table_cell_margin_bottom_inches",
    "table_cell_border_left_color",
    "table_cell_border_right_color",
    "table_cell_border_top_color",
    "table_cell_border_bottom_color",
    "table_cell_border_left_width_inches",
    "table_cell_border_right_width_inches",
    "table_cell_border_top_width_inches",
    "table_cell_border_bottom_width_inches",
    "table_cell_column_repeat",
    "table_cell_row_repeat",
    "table_cell_is_covered",
    "cell_value_type",
    "form_kind",
    "field_type",
    "label",
    "value",
    "form_field_alias",
    "form_field_tag",
    "form_field_id",
    "form_field_placeholder",
    "form_field_checked",
    "form_field_options",
    "form_field_option_values",
    "form_field_selected_option",
    "form_field_date_format",
    "form_field_date_full_value",
    "control_id",
    "validation_range",
    "validation_type",
    "validation_formula1",
    "connector_id",
    "connector_name",
    "connector_title",
    "connector_description",
    "source_shape_id",
    "target_shape_id",
    "source_connection_index",
    "target_connection_index",
    "control_part",
    "diagram_part",
    "embedded_file_filename",
    "element_type",
    "emphasis_term",
    "emphasis_terms",
    "style_summary",
    "image_part",
    "image_relationship_id",
    "image_relationship_type",
    "image_target_mode",
    "image_width",
    "image_height",
    "image_mime",
    "image_size_bytes",
    "inherited_from",
    "inherited_part",
    "layout_part",
    "layout_bbox",
    "layout_position",
    "layout_zone",
    "layout_x_inches",
    "layout_y_inches",
    "layout_width_inches",
    "layout_height_inches",
    "layout_center_x_inches",
    "layout_center_y_inches",
    "layout_left_pct",
    "layout_top_pct",
    "layout_width_pct",
    "layout_height_pct",
    "layout_center_x_pct",
    "layout_center_y_pct",
    "layout_x_points",
    "layout_y_points",
    "layout_width_points",
    "layout_height_points",
    "layout_center_x_points",
    "layout_center_y_points",
    "layout_relation",
    "visual_distance_pct",
    "related_element_type",
    "related_label",
    "related_location",
    "target_layout_zone",
    "target_layout_bbox",
    "page_width_points",
    "page_height_points",
    "page_width_inches",
    "page_height_inches",
    "label",
    "link_kind",
    "location",
    "group_id",
    "group_index",
    "group_label",
    "group_child_shape_ids",
    "group_child_labels",
    "group_child_count",
    "parent_group_id",
    "parent_group_label",
    "grouped_visual_label",
    "master_part",
    "media_part",
    "metadata_field",
    "metadata_part",
    "note_anchor_text",
    "note_anchor_section",
    "note_anchor_paragraph",
    "notes_part",
    "notes_relationship_id",
    "page",
    "section_label",
    "shape_id",
    "group_id",
    "group_label",
    "parent_group_id",
    "parent_group_label",
    "sheet_id",
    "sheet_name",
    "sheet_part",
    "sheet_relationship_id",
    "sheet_state",
    "sheet_visibility",
    "slide",
    "slide_id",
    "slide_part",
    "slide_relationship_id",
    "slide_state",
    "slide_title",
    "slide_visibility",
    "source_label",
    "table",
    "row",
    "column",
    "table_name",
    "target_label",
    "target_mode",
    "tooltip",
    "uri",
)

XML_NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "dgm": "http://schemas.openxmlformats.org/drawingml/2006/diagram",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}

IMAGE_MIME_BY_EXTENSION = {
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}

MEDIA_MIME_BY_EXTENSION = {
    ".aac": "audio/aac",
    ".avi": "video/x-msvideo",
    ".bin": "application/octet-stream",
    ".doc": "application/msword",
    ".docm": "application/vnd.ms-word.document.macroEnabled.12",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".dotm": "application/vnd.ms-word.template.macroEnabled.12",
    ".dotx": "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
    ".m4a": "audio/mp4",
    ".m4v": "video/mp4",
    ".mid": "audio/midi",
    ".midi": "audio/midi",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".ogg": "audio/ogg",
    ".pdf": "application/pdf",
    ".potm": "application/vnd.ms-powerpoint.template.macroEnabled.12",
    ".potx": "application/vnd.openxmlformats-officedocument.presentationml.template",
    ".ppsm": "application/vnd.ms-powerpoint.slideshow.macroEnabled.12",
    ".ppsx": "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptm": "application/vnd.ms-powerpoint.presentation.macroEnabled.12",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".wav": "audio/wav",
    ".webm": "video/webm",
    ".wmv": "video/x-ms-wmv",
    ".xls": "application/vnd.ms-excel",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xltm": "application/vnd.ms-excel.template.macroEnabled.12",
    ".xltx": "application/vnd.openxmlformats-officedocument.spreadsheetml.template",
    ".zip": "application/zip",
}

ENTITY_STOPWORDS = {
    "A", "An", "And", "Are", "As", "At", "By", "For", "From", "If", "In", "Into",
    "Is", "It", "Of", "On", "Or", "The", "This", "To", "With", "Without",
}
ENTITY_STOPWORDS_LOWER = {word.lower() for word in ENTITY_STOPWORDS}
ENTITY_TOKEN_PATTERN = r"[A-Z][A-Za-z0-9&/+-]*(?:\.[A-Z0-9][A-Za-z0-9&/+-]*)*"
ENTITY_PHRASE_PATTERN = re.compile(
    rf"\b{ENTITY_TOKEN_PATTERN}(?:\s+(?:{ENTITY_TOKEN_PATTERN}|of|for|and|the)){{0,5}}\b",
)

INTERACTION_REHEARSAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "cite", "do", "does",
    "for", "from", "how", "i", "in", "include", "is", "it", "of", "on", "or",
    "the", "this", "to", "use", "using", "what", "when", "where", "which", "with",
}

GENERIC_POWERPOINT_SHAPE_NAME_RE = re.compile(
    r"^(?:shape|rectangle|rounded rectangle|text box|textbox|title|subtitle|"
    r"content placeholder|object|picture|image|chart|table|placeholder)(?:\s*\d+)?$",
    re.I,
)

OFFICE_METADATA_FIELDS = {
    "docProps/core.xml": {
        "category": "Category",
        "content_status": "Content Status",
        "created": "Created",
        "creator": "Creator",
        "description": "Description",
        "identifier": "Identifier",
        "keywords": "Keywords",
        "language": "Language",
        "last_modified_by": "Last Modified By",
        "modified": "Modified",
        "revision": "Revision",
        "subject": "Subject",
        "title": "Title",
        "version": "Version",
    },
    "docProps/app.xml": {
        "application": "Application",
        "app_version": "Application Version",
        "characters": "Characters",
        "company": "Company",
        "hidden_slides": "Hidden Slides",
        "lines": "Lines",
        "manager": "Manager",
        "notes": "Notes",
        "pages": "Pages",
        "paragraphs": "Paragraphs",
        "presentation_format": "Presentation Format",
        "slides": "Slides",
        "template": "Template",
        "total_time": "Total Time",
        "words": "Words",
    },
}

OFFICE_METADATA_RELATIONS = {
    "application": "created_with",
    "app_version": "has_application_version",
    "author": "created_by",
    "category": "has_category",
    "characters": "has_character_count",
    "company": "has_company",
    "content_status": "has_content_status",
    "created": "created_at",
    "creator": "created_by",
    "creation_date": "created_at",
    "description": "has_description",
    "document_statistic": "has_document_statistic",
    "email_bcc": "bcc_to",
    "email_cc": "cc_to",
    "email_date": "sent_at",
    "email_from": "sent_by",
    "email_in_reply_to": "in_reply_to",
    "email_message_id": "has_message_id",
    "email_references": "references_message",
    "email_reply_to": "has_reply_to",
    "email_sender": "sent_by",
    "email_subject": "has_subject",
    "email_to": "sent_to",
    "hidden_slides": "has_hidden_slide_count",
    "identifier": "has_identifier",
    "initial_creator": "created_by",
    "keywords": "has_keyword",
    "language": "has_language",
    "last_modified_by": "modified_by",
    "lines": "has_line_count",
    "manager": "managed_by",
    "mod_date": "modified_at",
    "modified": "modified_at",
    "notes": "has_notes_count",
    "pages": "has_page_count",
    "paragraphs": "has_paragraph_count",
    "presentation_format": "has_presentation_format",
    "producer": "produced_by",
    "revision": "has_revision",
    "slides": "has_slide_count",
    "subject": "has_subject",
    "template": "uses_template",
    "title": "has_title",
    "total_time": "has_total_time",
    "version": "has_version",
    "words": "has_word_count",
}

PDF_METADATA_FIELDS = {
    "title": ("title", "Title"),
    "author": ("author", "Author"),
    "subject": ("subject", "Subject"),
    "keywords": ("keywords", "Keywords"),
    "creator": ("creator", "Creator"),
    "producer": ("producer", "Producer"),
    "creationDate": ("creation_date", "Creation Date"),
    "modDate": ("mod_date", "Modification Date"),
}

EMAIL_METADATA_FIELDS = (
    ("Subject", "email_subject", "Subject"),
    ("From", "email_from", "From"),
    ("Sender", "email_sender", "Sender"),
    ("To", "email_to", "To"),
    ("Cc", "email_cc", "Cc"),
    ("Bcc", "email_bcc", "Bcc"),
    ("Reply-To", "email_reply_to", "Reply-To"),
    ("Date", "email_date", "Date"),
    ("Message-ID", "email_message_id", "Message ID"),
    ("In-Reply-To", "email_in_reply_to", "In Reply To"),
    ("References", "email_references", "References"),
)

ODF_METADATA_FIELDS = {
    "title": ("title", "Title"),
    "description": ("description", "Description"),
    "subject": ("subject", "Subject"),
    "creator": ("creator", "Creator"),
    "initial-creator": ("initial_creator", "Initial Creator"),
    "creation-date": ("created", "Created"),
    "date": ("modified", "Modified"),
    "keyword": ("keywords", "Keywords"),
    "language": ("language", "Language"),
    "generator": ("application", "Application"),
    "editing-cycles": ("revision", "Revision"),
    "document-statistic": ("document_statistic", "Document Statistics"),
}


@dataclass
class DocumentElement:
    type: str
    text: str
    location: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunkPreview:
    chunk_id: str
    text: str
    token_estimate: int
    source_elements: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphCandidateRelationship:
    source: str
    relation: str
    target: str
    evidence: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


class _HTMLTextCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        self._tag_stack.append(tag)
        if tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str):
        if tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str):
        cleaned = data.strip()
        if cleaned:
            self.parts.append(cleaned)

    def text(self) -> str:
        return _clean_text(" ".join(self.parts))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _duration_ms(start: float) -> int:
    return max(0, round((time.perf_counter() - start) * 1000))


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip()


def _clean_entity_phrase(value: str) -> str:
    text = _clean_text(value).strip(" \"'`([{.,;:!?)]}")
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.I)
    text = re.sub(
        r"\s+(?:for|on|in|at|by|with|from|to|via|of)\s+(?:the\s+)?[a-z0-9][a-z0-9&/+-]*(?:\s+[a-z0-9][a-z0-9&/+-]*){0,5}$",
        "",
        text,
    )
    while True:
        trimmed = re.sub(
            r"\s+(?:and|or|for|of|the|to|with|from|in|on|at|via|by|but|while|when|because)$",
            "",
            text,
            flags=re.I,
        )
        if trimmed == text:
            break
        text = trimmed
    return text.strip(" \"'`([{.,;:!?)]}")


def _clip(value: str, limit: int = MAX_SAMPLE_TEXT) -> str:
    value = _clean_text(value)
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def _clean_multiline_text(value: str) -> str:
    normalized = html.unescape(value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in normalized.split("\n")]
    compact_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank and compact_lines:
                compact_lines.append("")
            previous_blank = True
            continue
        compact_lines.append(line)
        previous_blank = False
    return "\n".join(compact_lines).strip()


def _clip_multiline(value: str, limit: int = MAX_SAMPLE_TEXT) -> str:
    value = _clean_multiline_text(value)
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def _decode_bytes(content: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _xml_text(node: ET.Element, text_tags: tuple[str, ...] = ("t", "v")) -> str:
    values = []
    for child in node.iter():
        local_name = child.tag.rsplit("}", 1)[-1]
        if local_name in text_tags and child.text:
            values.append(child.text)
    return _clean_text(" ".join(values))


def _local_name(value: str) -> str:
    return str(value or "").rsplit("}", 1)[-1]


def _namespace_uri(value: str) -> str:
    text = str(value or "")
    if text.startswith("{") and "}" in text:
        return text[1:].split("}", 1)[0]
    return ""


def _attr_by_local_name(node: ET.Element | None, local_name: str) -> str | None:
    if node is None:
        return None
    for key, value in node.attrib.items():
        if _local_name(key) == local_name:
            return value
    return None


def _read_zip_xml(content: bytes, name: str) -> ET.Element | None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            with archive.open(name) as handle:
                return ET.fromstring(handle.read())
    except (KeyError, ET.ParseError, zipfile.BadZipFile):
        return None


def _snake_case(value: str) -> str:
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", _local_name(value or ""))
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _metadata_values(field: str, value: str) -> list[str]:
    cleaned = _clean_text(value)
    if not cleaned:
        return []
    if field in {"email_from", "email_sender", "email_to", "email_cc", "email_bcc", "email_reply_to"}:
        addresses = []
        for name, address in getaddresses([cleaned]):
            display_name = _clean_text(str(name or ""))
            address_text = _clean_text(str(address or ""))
            if display_name and address_text:
                addresses.append(f"{display_name} <{address_text}>")
            elif address_text:
                addresses.append(address_text)
            elif display_name:
                addresses.append(display_name)
        return _unique_preserve_order(addresses) or [cleaned]
    if field in {"email_references"}:
        return [
            item
            for item in (_clean_text(part) for part in re.split(r"\s+", cleaned))
            if item
        ]
    if field == "keywords":
        return [
            item
            for item in (_clean_text(part) for part in re.split(r"[;,]\s*|\n+", cleaned))
            if item
        ]
    return [cleaned]


def _office_package_metadata_elements(content: bytes) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    seen: set[tuple[str, str, str]] = set()

    for part_name, field_labels in OFFICE_METADATA_FIELDS.items():
        root = _read_zip_xml(content, part_name)
        if root is None:
            continue

        part_label = "core" if part_name.endswith("core.xml") else "app"
        for node in root.iter():
            field = _snake_case(node.tag)
            label = field_labels.get(field)
            if not label:
                continue
            value = _clean_text(" ".join(node.itertext()))
            if not value:
                continue
            key = (part_name, field, value)
            if key in seen:
                continue
            seen.add(key)
            element = _as_element(
                "document_metadata",
                f"{label}: {value}",
                location=f"{part_label} metadata {field}",
                metadata_part=part_name,
                metadata_field=field,
                metadata_label=label,
                metadata_value=value,
            )
            if element:
                elements.append(element)

    return elements


def _zip_names(content: bytes, prefix: str, suffix: str = ".xml") -> list[str]:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            return sorted(
                (name for name in archive.namelist() if name.startswith(prefix) and name.endswith(suffix)),
                key=_natural_key,
            )
    except zipfile.BadZipFile:
        return []


def _read_zip_bytes(content: bytes, name: str) -> bytes | None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            with archive.open(name) as handle:
                return handle.read()
    except (KeyError, zipfile.BadZipFile):
        return None


def _image_mime_from_name(part_name: str | None) -> str | None:
    suffix = Path(part_name or "").suffix.lower()
    if not suffix and part_name:
        suffix = f".{str(part_name).lower().lstrip('.')}"
    return IMAGE_MIME_BY_EXTENSION.get(suffix)


def _mime_from_name(part_name: str | None) -> str | None:
    suffix = Path(part_name or "").suffix.lower()
    if not suffix and part_name:
        suffix = f".{str(part_name).lower().lstrip('.')}"
    guessed, _ = mimetypes.guess_type(str(part_name or ""))
    return IMAGE_MIME_BY_EXTENSION.get(suffix) or MEDIA_MIME_BY_EXTENSION.get(suffix) or guessed


def _is_image_part(part_name: str | None) -> bool:
    mime = _mime_from_name(part_name)
    return bool(mime and mime.startswith("image/"))


def _png_dimensions(content: bytes) -> tuple[int, int] | None:
    if len(content) < 24 or content[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return int.from_bytes(content[16:20], "big"), int.from_bytes(content[20:24], "big")


def _gif_dimensions(content: bytes) -> tuple[int, int] | None:
    if len(content) < 10 or content[:6] not in {b"GIF87a", b"GIF89a"}:
        return None
    return int.from_bytes(content[6:8], "little"), int.from_bytes(content[8:10], "little")


def _jpeg_dimensions(content: bytes) -> tuple[int, int] | None:
    if len(content) < 4 or content[:2] != b"\xff\xd8":
        return None
    index = 2
    while index + 9 < len(content):
        if content[index] != 0xFF:
            index += 1
            continue
        marker = content[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(content):
            return None
        segment_length = int.from_bytes(content[index:index + 2], "big")
        if segment_length < 2 or index + segment_length > len(content):
            return None
        if marker in {
            0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
        }:
            height = int.from_bytes(content[index + 3:index + 5], "big")
            width = int.from_bytes(content[index + 5:index + 7], "big")
            return width, height
        index += segment_length
    return None


def _image_dimensions_from_bytes(content: bytes) -> tuple[int, int] | None:
    dimensions = _png_dimensions(content) or _gif_dimensions(content) or _jpeg_dimensions(content)
    if dimensions:
        return dimensions
    try:
        from PIL import Image

        with Image.open(io.BytesIO(content)) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_image_ocr(content: bytes, *, limit: int = 500) -> str | None:
    if not content or not _env_bool("GRAPHRAG_ENABLE_IMAGE_OCR", False):
        return None
    try:
        from PIL import Image
        import pytesseract

        with Image.open(io.BytesIO(content)) as image:
            return _clip(pytesseract.image_to_string(image), limit) or None
    except Exception:
        return None


def _image_metadata_from_bytes(
    content: bytes | None,
    part_name: str | None = None,
    *,
    enable_ocr: bool = True,
    ocr_limit: int = 500,
) -> dict[str, Any]:
    if not content:
        return {}
    metadata: dict[str, Any] = {
        "image_size_bytes": len(content),
        "image_sha256": hashlib.sha256(content).hexdigest()[:16],
    }
    mime = _image_mime_from_name(part_name)
    if mime:
        metadata["image_mime"] = mime
    dimensions = _image_dimensions_from_bytes(content)
    if dimensions:
        metadata["image_width"], metadata["image_height"] = dimensions
    ocr_text = _optional_image_ocr(content, limit=ocr_limit) if enable_ocr else None
    if ocr_text:
        metadata["ocr_text"] = ocr_text
        metadata["ocr_engine"] = "pytesseract"
    return metadata


def _zip_image_metadata(content: bytes, part_name: str | None) -> dict[str, Any]:
    return _image_metadata_from_bytes(_read_zip_bytes(content, part_name) if part_name else None, part_name)


def _image_link_metadata(
    image_metadata: dict[str, Any],
    *,
    image_part: str | None,
    image_relationship_id: str | None,
    image_relationship: dict[str, str] | None = None,
) -> dict[str, Any]:
    relationship = image_relationship or {}
    metadata: dict[str, Any] = {
        "image_part": image_part,
        "image_relationship_id": image_relationship_id,
        "image_relationship_type": relationship.get("type"),
        "image_target_mode": relationship.get("target_mode"),
    }
    for key in ("image_width", "image_height", "image_mime", "image_size_bytes", "image_sha256"):
        if image_metadata.get(key) not in (None, ""):
            metadata[key] = image_metadata[key]
    return {key: value for key, value in metadata.items() if value not in (None, "", [], {})}


def _image_ocr_text_element(
    image_metadata: dict[str, Any],
    *,
    location: str,
    image_part: str | None,
    ocr_source: str,
    base_metadata: dict[str, Any] | None = None,
) -> DocumentElement | None:
    ocr_text = _clean_text(str(image_metadata.get("ocr_text") or ""))
    if not ocr_text:
        return None
    return _as_element(
        "image_ocr_text",
        ocr_text,
        location=location,
        **(base_metadata or {}),
        image_part=image_part,
        ocr_source=ocr_source,
        ocr_engine=image_metadata.get("ocr_engine"),
        image_width=image_metadata.get("image_width"),
        image_height=image_metadata.get("image_height"),
        image_mime=image_metadata.get("image_mime"),
        image_size_bytes=image_metadata.get("image_size_bytes"),
        image_sha256=image_metadata.get("image_sha256"),
    )


def _extract_image_file(content: bytes, filename: str | None = None, content_type: str | None = None) -> list[DocumentElement]:
    image_metadata = _image_metadata_from_bytes(content, filename, ocr_limit=1600)
    if content_type and content_type.lower().startswith("image/") and not image_metadata.get("image_mime"):
        image_metadata["image_mime"] = content_type
    label = _image_element_text(filename, metadata=image_metadata)
    elements: list[DocumentElement] = []

    image_element = _as_element(
        "image",
        label,
        location="image file",
        image_part=filename,
        ocr_source="image_file" if image_metadata.get("ocr_text") else None,
        **image_metadata,
    )
    if image_element:
        elements.append(image_element)

    ocr_text = _clean_text(str(image_metadata.get("ocr_text") or ""))
    if ocr_text:
        ocr_element = _as_element(
            "image_ocr_text",
            ocr_text,
            location="image OCR",
            image_part=filename,
            ocr_source="image_file",
            ocr_engine=image_metadata.get("ocr_engine"),
            image_width=image_metadata.get("image_width"),
            image_height=image_metadata.get("image_height"),
            image_mime=image_metadata.get("image_mime"),
            image_size_bytes=image_metadata.get("image_size_bytes"),
            image_sha256=image_metadata.get("image_sha256"),
        )
        if ocr_element:
            elements.append(ocr_element)

    return elements


def _binary_metadata_from_bytes(content: bytes | None, part_name: str | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    mime = _mime_from_name(part_name)
    if mime:
        metadata["media_mime"] = mime
    if part_name:
        suffix = Path(part_name).suffix.lower()
        if suffix:
            metadata["media_extension"] = suffix.lstrip(".")
    if content:
        metadata["media_size_bytes"] = len(content)
        metadata["media_sha256"] = hashlib.sha256(content).hexdigest()[:16]
    return metadata


def _zip_binary_metadata(content: bytes, part_name: str | None) -> dict[str, Any]:
    if not part_name:
        return {}
    return _binary_metadata_from_bytes(_read_zip_bytes(content, part_name), part_name)


def _media_kind_from_relationship(relationship_type: str, target: str | None, mime: str | None) -> str:
    rel_type = relationship_type.lower()
    target_path = str(target or "").lower()
    if rel_type.endswith("/audio") or (mime or "").startswith("audio/"):
        return "audio"
    if rel_type.endswith("/video") or (mime or "").startswith("video/"):
        return "video"
    if rel_type.endswith("/oleobject"):
        return "embedded_object"
    if rel_type.endswith("/package") or "/embeddings/" in target_path:
        return "embedded_package"
    return "media"


def _media_element_text(
    *,
    media_kind: str,
    media_part: str | None = None,
    uri: str | None = None,
    media_mime: str | None = None,
    size_bytes: int | None = None,
) -> str:
    target = uri or media_part
    name = Path(str(target or "")).name if target else ""
    parts = [
        _clean_text(f"{media_kind.replace('_', ' ')} {name}".strip()),
        _clean_text(str(target or "")),
        _clean_text(str(media_mime or "")),
    ]
    if size_bytes:
        parts.append(f"{size_bytes} bytes")
    return "; ".join(_unique_preserve_order([part for part in parts if part])) or media_kind.replace("_", " ")


def _image_element_text(*labels: str | None, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    parts = [_clean_text(str(label)) for label in labels if _clean_text(str(label or ""))]
    if metadata.get("image_width") and metadata.get("image_height"):
        parts.append(f"{metadata['image_width']}x{metadata['image_height']} image")
    if metadata.get("image_mime"):
        parts.append(str(metadata["image_mime"]))
    if metadata.get("ocr_text"):
        parts.append(f"OCR text: {metadata['ocr_text']}")
    return "; ".join(_unique_preserve_order(parts)) or "Image"


def _unique_preserve_order(values: list[str]) -> list[str]:
    unique = []
    seen = set()
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def _relationship_part_name(part_name: str) -> str:
    folder, filename = posixpath.split(part_name)
    return posixpath.join(folder, "_rels", f"{filename}.rels")


def _resolve_part_target(part_name: str, target: str | None) -> str | None:
    if not target:
        return None
    if target.startswith("/"):
        return posixpath.normpath(target.lstrip("/"))
    return posixpath.normpath(posixpath.join(posixpath.dirname(part_name), target))


def _part_relationships(content: bytes, part_name: str) -> dict[str, dict[str, str]]:
    rels_root = _read_zip_xml(content, _relationship_part_name(part_name))
    if rels_root is None:
        return {}

    relationships: dict[str, dict[str, str]] = {}
    for relationship in rels_root.findall(".//{*}Relationship"):
        rel_id = relationship.attrib.get("Id")
        if not rel_id:
            continue
        raw_target = relationship.attrib.get("Target")
        target_mode = relationship.attrib.get("TargetMode", "")
        target = raw_target if target_mode.lower() == "external" else _resolve_part_target(part_name, raw_target)
        relationships[rel_id] = {
            "id": rel_id,
            "type": relationship.attrib.get("Type", ""),
            "target": target or "",
            "target_mode": target_mode,
        }
    return relationships


def _relationship_targets_by_type(content: bytes, part_name: str, type_suffix: str) -> list[str]:
    return [
        rel["target"]
        for rel in _part_relationships(content, part_name).values()
        if rel.get("target") and rel.get("type", "").endswith(type_suffix)
    ]


def _natural_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def _as_element(element_type: str, text: str, *, location: str | None = None, **metadata: Any) -> DocumentElement | None:
    cleaned = _clean_text(text)
    if not cleaned:
        return None
    return DocumentElement(type=element_type, text=cleaned, location=location, metadata={k: v for k, v in metadata.items() if v is not None})


def _table_row_text(cells: list[str], headers: list[str] | None = None) -> str:
    cleaned = [_clean_text(cell) for cell in cells]
    if headers:
        pairs = [
            f"{_clean_text(header)}: {cell}"
            for header, cell in zip(headers, cleaned)
            if _clean_text(header) and cell
        ]
        if pairs:
            return "; ".join(pairs)
    return " | ".join(cell for cell in cleaned if cell)


def _table_elements_from_rows(
    rows: list[list[str]],
    *,
    location_prefix: str,
    table: int | None = None,
    base_metadata: dict[str, Any] | None = None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    headers: list[str] | None = None
    metadata_base = base_metadata or {}

    for row_index, row in enumerate(rows, start=1):
        cells = [_clean_text(str(cell)) for cell in row]
        if not any(cells):
            continue

        is_header = headers is None and len([cell for cell in cells if cell]) > 1
        element_type = "table_header" if is_header else "table_row"
        if is_header:
            headers = cells

        row_headers = headers if not is_header else None
        row_location = f"{location_prefix} {row_index}" if location_prefix.endswith("row") else f"{location_prefix} row {row_index}"
        element = _as_element(
            element_type,
            _table_row_text(cells, row_headers),
            location=row_location,
            **metadata_base,
            table=table,
            row=row_index,
            headers=headers if headers else None,
            cells=cells,
        )
        if element:
            elements.append(element)

    return elements


def _table_cell_context_metadata(
    rows: list[list[str]],
    row_index: int,
    cell_index: int,
) -> dict[str, str]:
    if row_index < 1 or cell_index < 1:
        return {}

    cleaned_rows = [
        [_clean_text(str(cell)) for cell in row]
        for row in rows
    ]
    if row_index > len(cleaned_rows):
        return {}

    header_row = cleaned_rows[0] if cleaned_rows and len([cell for cell in cleaned_rows[0] if cell]) > 1 else []
    row = cleaned_rows[row_index - 1]
    column_header = header_row[cell_index - 1] if header_row and cell_index <= len(header_row) and row_index > 1 else ""
    row_header = row[0] if cell_index > 1 and row and row[0] else ""
    context_parts = []
    if column_header:
        context_parts.append(column_header)
    if row_header:
        context_parts.append(f"for {row_header}")
    context = _clean_text(" ".join(context_parts))

    return {
        key: value
        for key, value in {
            "table_cell_column_header": column_header,
            "table_cell_row_header": row_header,
            "table_cell_context": context,
        }.items()
        if value
    }


def _table_shape(rows: list[list[str]]) -> tuple[int, int]:
    row_count = len([row for row in rows if any(_clean_text(str(cell)) for cell in row)])
    column_count = max((len(row) for row in rows), default=0)
    return row_count, column_count


def _markdown_link_elements(
    text: str,
    *,
    location: str,
    section: str | None = None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    link_pattern = re.compile(r"(!)?\[([^\]]+)\]\(([^)\s]+)(?:\s+\"([^\"]+)\")?\)")
    seen: set[tuple[str, str, str]] = set()
    for link_index, match in enumerate(link_pattern.finditer(text), start=1):
        is_image = bool(match.group(1))
        label = _clean_text(match.group(2))
        target = _clean_text(match.group(3))
        title = _clean_text(match.group(4) or "")
        if not target:
            continue
        key = ("image" if is_image else "link", label, target)
        if key in seen:
            continue
        seen.add(key)
        if is_image:
            element = _as_element(
                "image",
                _image_element_text(label, title, target),
                location=f"{location} image {link_index}",
                src=target,
                alt=label,
                title=title or None,
                section=section,
                link_kind="markdown_image",
            )
        else:
            element = _as_element(
                "link",
                f"{label} -> {target}" if label and label != target else target,
                location=f"{location} link {link_index}",
                uri=target,
                label=label or target,
                title=title or None,
                section=section,
                link_kind="markdown_link",
            )
        if element:
            elements.append(element)

    raw_url_pattern = re.compile(r"(?<!\()https?://[^\s<>()\"']+")
    for raw_index, match in enumerate(raw_url_pattern.finditer(text), start=1):
        uri = match.group(0).rstrip(".,;:")
        key = ("raw_link", uri, uri)
        if key in seen or any(element.metadata.get("uri") == uri for element in elements):
            continue
        seen.add(key)
        element = _as_element(
            "link",
            uri,
            location=f"{location} raw link {raw_index}",
            uri=uri,
            label=uri,
            section=section,
            link_kind="markdown_raw_url",
        )
        if element:
            elements.append(element)
    return elements


def _markdown_frontmatter_elements(lines: list[str]) -> tuple[list[DocumentElement], int]:
    if not lines or lines[0].strip() not in {"---", "+++"}:
        return [], 0
    fence = lines[0].strip()
    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == fence:
            end_index = index
            break
    if end_index is None:
        return [], 0

    fields: list[tuple[str, str]] = []
    current_key = ""
    current_values: list[str] = []

    def flush_current() -> None:
        nonlocal current_key, current_values
        if current_key:
            fields.append((current_key, ", ".join(value for value in current_values if value)))
        current_key = ""
        current_values = []

    for line in lines[1:end_index]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        list_item = re.match(r"^[-*]\s+(.+)$", stripped)
        if list_item and current_key:
            current_values.append(_clean_text(list_item.group(1).strip().strip("\"'")))
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*:\s*(.*)$", stripped)
        if match:
            flush_current()
            current_key = _relation_label(match.group(1))
            raw_value = match.group(2).strip()
            if raw_value and raw_value not in {"[]", "{}"}:
                current_values.append(_clean_text(raw_value.strip("\"'")))
    flush_current()

    elements: list[DocumentElement] = []
    for field_index, (field, value) in enumerate(fields, start=1):
        if not field or not value:
            continue
        label = field.replace("_", " ").title()
        element = _as_element(
            "document_metadata",
            f"{label}: {value}",
            location=f"frontmatter {field}",
            metadata_part="markdown_frontmatter",
            metadata_field=field,
            metadata_label=label,
            metadata_value=value,
            markdown_frontmatter_index=field_index,
        )
        if element:
            elements.append(element)
    return elements, end_index + 1


def _markdown_node_label(raw: str) -> tuple[str, str]:
    text = _clean_text(raw.strip().rstrip(";"))
    text = re.sub(r"^&", "", text).strip()
    match = re.match(r"^([A-Za-z0-9_.:-]+)\s*(?:\[([^\]]+)\]|\(([^)]+)\)|\{([^}]+)\}|\"([^\"]+)\")?", text)
    if not match:
        cleaned = text.strip("\"'")
        return cleaned, cleaned
    node_id = _clean_text(match.group(1))
    label = next((match.group(index) for index in range(2, 6) if match.group(index)), None)
    label_text = _clean_text(str(label or node_id).strip("\"'"))
    return node_id, label_text or node_id


def _markdown_mermaid_elements(
    code: str,
    *,
    block_index: int,
    start_line: int,
    end_line: int,
    section: str | None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    node_labels: dict[str, str] = {}
    emitted_nodes: set[str] = set()

    def emit_node(node_id: str, label: str, order: int) -> None:
        if not node_id or node_id in emitted_nodes:
            return
        emitted_nodes.add(node_id)
        element = _as_element(
            "diagram_node",
            label,
            location=f"mermaid block {block_index} node {order}",
            section=section,
            diagram_part=f"markdown mermaid block {block_index}",
            diagram_node_id=node_id,
            diagram_node_type="mermaid",
            diagram_order=order,
            diagram_level=0,
            markdown_code_language="mermaid",
            markdown_code_block_index=block_index,
            markdown_start_line=start_line,
            markdown_end_line=end_line,
        )
        if element:
            elements.append(element)

    edge_pattern = re.compile(r"^\s*(?P<source>.+?)\s*(?P<arrow><-->|-->|---|-.->|==>|--x|--o|<--)\s*(?P<target>.+?)\s*$")
    node_order = 0
    edge_order = 0
    for line_number, raw_line in enumerate(code.splitlines(), start=start_line):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("%%", "graph ", "flowchart ", "sequenceDiagram", "classDiagram", "stateDiagram")):
            continue
        edge_match = edge_pattern.match(stripped)
        if edge_match:
            source_raw = edge_match.group("source")
            target_raw = edge_match.group("target")
            relation = "connects_to"
            label_match = re.match(r"^\|([^|]+)\|\s*(.+)$", target_raw)
            if label_match:
                relation = _clean_text(label_match.group(1)) or relation
                target_raw = label_match.group(2)
            source_id, source_label = _markdown_node_label(source_raw)
            target_id, target_label = _markdown_node_label(target_raw)
            if source_id and source_id not in node_labels:
                node_order += 1
                node_labels[source_id] = source_label
                emit_node(source_id, source_label, node_order)
            if target_id and target_id not in node_labels:
                node_order += 1
                node_labels[target_id] = target_label
                emit_node(target_id, target_label, node_order)
            if source_label and target_label:
                edge_order += 1
                element = _as_element(
                    "diagram_relationship",
                    f"{source_label} --[{relation}]--> {target_label}",
                    location=f"mermaid block {block_index} edge {edge_order}",
                    section=section,
                    diagram_part=f"markdown mermaid block {block_index}",
                    diagram_connection_type=_relation_label(relation),
                    source_diagram_node_id=source_id,
                    target_diagram_node_id=target_id,
                    source_label=source_label,
                    target_label=target_label,
                    relation=relation,
                    markdown_code_language="mermaid",
                    markdown_code_block_index=block_index,
                    markdown_start_line=line_number,
                    markdown_end_line=line_number,
                )
                if element:
                    elements.append(element)
            continue

        node_id, label = _markdown_node_label(stripped)
        if node_id and label and ("[" in stripped or "(" in stripped or "{" in stripped):
            node_order += 1
            node_labels.setdefault(node_id, label)
            emit_node(node_id, label, node_order)
    return elements


def _extract_plain_text(content: bytes, extension: str) -> list[DocumentElement]:
    text = _decode_bytes(content)
    elements: list[DocumentElement] = []

    if extension in {".md", ".markdown"}:
        lines = text.splitlines()
        frontmatter_elements, start_index = _markdown_frontmatter_elements(lines)
        elements.extend(frontmatter_elements)
        current_heading = None
        code_fence: str | None = None
        code_language = ""
        code_lines: list[str] = []
        code_start_line = 0
        code_block_index = 0
        table_index = 0
        in_table = False

        for line_number, line in enumerate(lines[start_index:], start=start_index + 1):
            stripped = line.strip()
            fence_match = re.match(r"^(```|~~~)\s*([A-Za-z0-9_.+-]*)", stripped)
            if code_fence:
                if stripped.startswith(code_fence):
                    code_block_index += 1
                    code_text = "\n".join(code_lines).strip()
                    element = _as_element(
                        "code_block",
                        code_text,
                        location=f"lines {code_start_line}-{line_number}",
                        section=current_heading,
                        markdown_code_language=code_language or None,
                        markdown_code_block_index=code_block_index,
                        markdown_start_line=code_start_line,
                        markdown_end_line=line_number,
                    )
                    if element:
                        elements.append(element)
                    if (code_language or "").lower() in {"mermaid", "mmd"} and code_text:
                        elements.extend(_markdown_mermaid_elements(
                            code_text,
                            block_index=code_block_index,
                            start_line=code_start_line,
                            end_line=line_number,
                            section=current_heading,
                        ))
                    code_fence = None
                    code_language = ""
                    code_lines = []
                    code_start_line = 0
                else:
                    code_lines.append(line)
                continue
            if fence_match:
                code_fence = fence_match.group(1)
                code_language = _clean_text(fence_match.group(2) or "")
                code_start_line = line_number
                code_lines = []
                in_table = False
                continue

            if not stripped:
                in_table = False
                continue
            heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if heading:
                current_heading = heading.group(2).strip()
                in_table = False
                element = _as_element("heading", current_heading, location=f"line {line_number}", level=len(heading.group(1)))
            elif re.match(r"^[-*+]\s+\[[ xX]\]\s+.+$", stripped):
                task = re.match(r"^[-*+]\s+\[([ xX])\]\s+(.+)$", stripped)
                checked = bool(task and task.group(1).lower() == "x")
                label = _clean_text(task.group(2) if task else stripped)
                element = _as_element(
                    "form_field",
                    f"{label}: {checked}",
                    location=f"line {line_number}",
                    section=current_heading,
                    form_kind="markdown_task_list",
                    field_type="checkbox",
                    label=label,
                    value=str(checked),
                    form_field_checked=checked,
                )
                elements.extend(_markdown_link_elements(label, location=f"line {line_number}", section=current_heading))
            elif re.match(r"^\|.*\|$", stripped):
                cells = [cell.strip() for cell in stripped.strip("|").split("|")]
                if all(re.match(r"^:?-{3,}:?$", cell.strip()) for cell in cells if cell.strip()):
                    continue
                if not in_table:
                    table_index += 1
                    in_table = True
                element_type = "table_header" if table_index and not any(
                    existing.type in {"table_header", "table_row"}
                    and existing.metadata.get("table") == table_index
                    for existing in elements[-8:]
                ) else "table_row"
                element = _as_element(
                    element_type,
                    " | ".join(cells),
                    location=f"line {line_number}",
                    cells=cells,
                    section=current_heading,
                    table=table_index,
                    row=line_number,
                )
            else:
                element = _as_element("paragraph", stripped, location=f"line {line_number}", section=current_heading)
                elements.extend(_markdown_link_elements(stripped, location=f"line {line_number}", section=current_heading))
            if element:
                elements.append(element)
        if code_fence and code_lines:
            code_block_index += 1
            code_text = "\n".join(code_lines).strip()
            element = _as_element(
                "code_block",
                code_text,
                location=f"lines {code_start_line}-{len(lines)}",
                section=current_heading,
                markdown_code_language=code_language or None,
                markdown_code_block_index=code_block_index,
                markdown_start_line=code_start_line,
                markdown_end_line=len(lines),
            )
            if element:
                elements.append(element)
        return elements

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    for index, paragraph in enumerate(paragraphs or [text], start=1):
        element_type = "heading" if len(paragraph) < 120 and paragraph.isupper() else "paragraph"
        element = _as_element(element_type, paragraph, location=f"block {index}")
        if element:
            elements.append(element)
    return elements


def _extract_html(content: bytes) -> list[DocumentElement]:
    text = _decode_bytes(content)
    elements: list[DocumentElement] = []
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "html.parser")
        for index, tag in enumerate(soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "a", "img"]), start=1):
            if tag.name == "a":
                href = tag.get("href")
                label = tag.get_text(" ", strip=True) or href
                element = _as_element("link", f"{label} -> {href}" if href and label != href else str(label or ""), location=f"a {index}", uri=href, label=label)
            elif tag.name == "img":
                src = tag.get("src")
                label = _image_element_text(tag.get("alt"), tag.get("title"), src)
                element = _as_element("image", label, location=f"img {index}", src=src, alt=tag.get("alt"), title=tag.get("title"))
            else:
                tag_text = tag.get_text(" ", strip=True)
                if not tag_text:
                    continue
                element_type = "heading" if tag.name.startswith("h") else "table_row" if tag.name == "tr" else "paragraph"
                cells = [cell.get_text(" ", strip=True) for cell in tag.find_all(["td", "th"])] if tag.name == "tr" else None
                element = _as_element(element_type, tag_text, location=f"{tag.name} {index}", cells=cells)
            if element:
                elements.append(element)
    except Exception:
        collector = _HTMLTextCollector()
        collector.feed(text)
        elements = _extract_plain_text(collector.text().encode("utf-8"), ".txt")

    return elements


def _decode_email_header_value(value: str | None) -> str:
    if not value:
        return ""
    try:
        return _clean_text(str(make_header(decode_header(str(value)))))
    except Exception:
        return _clean_text(str(value))


def _email_header_join(message: Message, header_name: str) -> str:
    return "; ".join(
        value
        for value in (_decode_email_header_value(raw_value) for raw_value in message.get_all(header_name, []))
        if value
    )


def _email_payload_bytes(part: Message) -> bytes:
    payload = part.get_payload(decode=True)
    if isinstance(payload, bytes):
        return payload
    try:
        content = part.get_content()
    except Exception:
        content = part.get_payload()
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        charset = part.get_content_charset() or "utf-8"
        return content.encode(charset, errors="replace")
    return b""


def _email_part_text(part: Message) -> str:
    payload = part.get_payload(decode=True)
    charset = part.get_content_charset() or "utf-8"
    if isinstance(payload, bytes):
        try:
            return payload.decode(charset, errors="replace")
        except LookupError:
            return payload.decode("utf-8", errors="replace")
    try:
        content = part.get_content()
    except Exception:
        content = part.get_payload()
    if isinstance(content, bytes):
        try:
            return content.decode(charset, errors="replace")
        except LookupError:
            return content.decode("utf-8", errors="replace")
    return str(content or "")


def _email_base_metadata(message: Message) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    header_to_metadata_key = {
        "Subject": "email_subject",
        "From": "email_from",
        "Sender": "email_sender",
        "To": "email_to",
        "Cc": "email_cc",
        "Date": "email_date",
        "Message-ID": "email_message_id",
        "In-Reply-To": "email_in_reply_to",
    }
    for header_name, metadata_key in header_to_metadata_key.items():
        value = _email_header_join(message, header_name)
        if value:
            metadata[metadata_key] = value
    date_value = metadata.get("email_date")
    if date_value:
        try:
            metadata["email_date_iso"] = parsedate_to_datetime(str(date_value)).isoformat()
        except Exception:
            pass
    return metadata


def _email_body_link_elements(text: str, *, location_prefix: str, base_metadata: dict[str, Any]) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    seen: set[str] = set()
    for link_index, match in enumerate(re.finditer(r"https?://[^\s<>()\"']+", text), start=1):
        uri = match.group(0).rstrip(".,;:)]}")
        if not uri or uri.lower() in seen:
            continue
        seen.add(uri.lower())
        element = _as_element(
            "link",
            uri,
            location=f"{location_prefix} link {link_index}",
            **base_metadata,
            uri=uri,
            label=uri,
            link_kind="email_body",
        )
        if element:
            elements.append(element)
    return elements


def _extract_email(content: bytes) -> list[DocumentElement]:
    message = BytesParser(policy=policy.default).parsebytes(content)
    elements: list[DocumentElement] = []
    base_metadata = _email_base_metadata(message)

    for header_name, metadata_field, label in EMAIL_METADATA_FIELDS:
        value = _email_header_join(message, header_name)
        if not value:
            continue
        if metadata_field in {"email_from", "email_sender", "email_to", "email_cc", "email_bcc", "email_reply_to"}:
            value = ", ".join(_metadata_values(metadata_field, value)) or value
        metadata = {
            **base_metadata,
            "metadata_part": "email_headers",
            "metadata_field": metadata_field,
            "metadata_label": label,
            "metadata_value": value,
            "email_header": header_name,
        }
        element = _as_element(
            "document_metadata",
            f"{label}: {value}",
            location=f"email header {header_name.lower()}",
            **metadata,
        )
        if element:
            elements.append(element)

    body_index = 0
    attachment_index = 0
    parts = list(message.walk()) if message.is_multipart() else [message]
    for part in parts:
        if part.is_multipart():
            continue
        content_type = (part.get_content_type() or "").lower()
        disposition = (part.get_content_disposition() or "").lower()
        filename = _decode_email_header_value(part.get_filename())
        is_attachment = bool(filename) or disposition == "attachment"
        payload = _email_payload_bytes(part)

        if is_attachment:
            attachment_index += 1
            attachment_filename = filename or f"attachment-{attachment_index}"
            attachment_mime = content_type or _mime_from_name(attachment_filename) or "application/octet-stream"
            attachment_metadata = {
                **base_metadata,
                "email_attachment_index": attachment_index,
                "email_attachment_filename": attachment_filename,
                "email_attachment_mime": attachment_mime,
                "email_content_type": content_type,
                "email_disposition": disposition,
                "embedded_file_name": attachment_filename,
                "embedded_file_filename": attachment_filename,
                "embedded_file_mime": attachment_mime,
                "embedded_file_size_bytes": len(payload),
                "embedded_file_sha256": hashlib.sha256(payload).hexdigest()[:16] if payload else None,
            }
            label = _media_element_text(
                media_kind="email_attachment",
                media_part=attachment_filename,
                media_mime=attachment_mime,
                size_bytes=len(payload) if payload else None,
            )
            element = _as_element(
                "embedded_object",
                label,
                location=f"email attachment {attachment_index}",
                **attachment_metadata,
            )
            if element:
                elements.append(element)
            elements.extend(_embedded_file_child_elements(
                payload,
                attachment_filename,
                attachment_index=attachment_index,
                base_metadata=attachment_metadata,
            ))
            continue

        if content_type not in {"text/plain", "text/html"}:
            continue

        body_index += 1
        body_metadata = {
            **base_metadata,
            "email_part_index": body_index,
            "email_part_kind": "html" if content_type == "text/html" else "plain",
            "email_content_type": content_type,
            "email_disposition": disposition,
        }
        if content_type == "text/html":
            html_elements = _extract_html(payload or _email_part_text(part).encode("utf-8"))
            body_text = _clean_text(" ".join(
                element.text
                for element in html_elements
                if element.type not in {"link", "image"}
            )) or _clean_text(_email_part_text(part))
            for element in html_elements[:80]:
                element.location = f"email body {body_index} {element.location or element.type}"
                element.metadata.update(body_metadata)
                elements.append(element)
        else:
            body_text = _clean_text(_email_part_text(part))
            elements.extend(_email_body_link_elements(
                body_text,
                location_prefix=f"email body {body_index}",
                base_metadata=body_metadata,
            ))

        body_element = _as_element(
            "email_body",
            body_text,
            location=f"email body {body_index}",
            **body_metadata,
        )
        if body_element:
            elements.append(body_element)

    return elements or _extract_plain_text(content, ".txt")


def _legacy_office_format(extension: str, content_type: str | None) -> str:
    normalized_content_type = (content_type or "").lower()
    if extension == ".doc" or "msword" in normalized_content_type or "ms-word" in normalized_content_type:
        return "word"
    if extension == ".ppt" or "powerpoint" in normalized_content_type:
        return "powerpoint"
    if extension == ".xls" or "excel" in normalized_content_type:
        return "excel"
    return "office"


def _legacy_office_stream_label(path: Any) -> str:
    raw_parts = path if isinstance(path, (list, tuple)) else [path]
    parts: list[str] = []
    for part in raw_parts:
        text = re.sub(r"[\x00-\x1f]+", "", str(part or ""))
        text = _clean_text(text)
        if text:
            parts.append(text)
    return "/".join(parts) or "stream"


def _binary_printable_strings(
    content: bytes,
    *,
    limit: int = LEGACY_OFFICE_MAX_STRINGS_PER_STREAM,
    min_length: int = 4,
) -> list[str]:
    matches: list[tuple[int, str]] = []

    for match in re.finditer(rb"[\x20-\x7E]{4,}", content):
        matches.append((match.start(), match.group(0).decode("latin-1", errors="ignore")))

    for pattern, encoding in (
        (rb"(?:[\x20-\x7E]\x00){4,}", "utf-16le"),
        (rb"(?:\x00[\x20-\x7E]){4,}", "utf-16be"),
    ):
        for match in re.finditer(pattern, content):
            try:
                matches.append((match.start(), match.group(0).decode(encoding, errors="ignore")))
            except UnicodeDecodeError:
                continue

    strings: list[str] = []
    seen: set[str] = set()
    for _, candidate in sorted(matches, key=lambda item: item[0]):
        cleaned = _clip(candidate.replace("\x00", " "), 240)
        if len(cleaned) < min_length:
            continue
        if not re.search(r"[A-Za-z0-9]", cleaned):
            continue
        alpha_numeric_ratio = sum(1 for char in cleaned if char.isalnum()) / max(len(cleaned), 1)
        if alpha_numeric_ratio < 0.25:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        strings.append(cleaned)
        if len(strings) >= limit:
            break
    return strings


def _legacy_office_base_metadata(
    *,
    office_format: str,
    stream_index: int,
    stream_path: str,
    stream_bytes: bytes,
    extraction_method: str,
    truncated: bool = False,
) -> dict[str, Any]:
    return {
        "legacy_office_format": office_format,
        "legacy_office_stream_index": stream_index,
        "legacy_office_stream_path": stream_path,
        "legacy_office_stream_size_bytes": len(stream_bytes),
        "legacy_office_stream_sha256": hashlib.sha256(stream_bytes).hexdigest()[:16] if stream_bytes else None,
        "legacy_office_stream_truncated": truncated or None,
        "legacy_office_extraction_method": extraction_method,
    }


def _legacy_office_text_element(
    strings: list[str],
    *,
    location: str,
    metadata: dict[str, Any],
) -> DocumentElement | None:
    if not strings:
        return None
    text = _clip("\n".join(strings), LEGACY_OFFICE_MAX_TEXT_CHARS)
    return _as_element(
        "legacy_office_text",
        text,
        location=location,
        **metadata,
        legacy_office_string_count=len(strings),
    )


def _legacy_office_raw_fallback_elements(
    content: bytes,
    *,
    filename: str,
    office_format: str,
) -> list[DocumentElement]:
    stream_path = "raw_content"
    metadata = _legacy_office_base_metadata(
        office_format=office_format,
        stream_index=0,
        stream_path=stream_path,
        stream_bytes=content[:LEGACY_OFFICE_MAX_STREAM_BYTES],
        extraction_method="raw_printable_strings",
        truncated=len(content) > LEGACY_OFFICE_MAX_STREAM_BYTES,
    )
    elements: list[DocumentElement] = []
    stream_element = _as_element(
        "legacy_office_stream",
        f"Legacy Office {office_format} raw content for {Path(filename or 'document').name} ({len(content)} bytes)",
        location="legacy office raw content",
        **metadata,
    )
    if stream_element:
        elements.append(stream_element)
    text_element = _legacy_office_text_element(
        _binary_printable_strings(content, limit=LEGACY_OFFICE_MAX_STRINGS_PER_STREAM * 2),
        location="legacy office raw content text",
        metadata=metadata,
    )
    if text_element:
        elements.append(text_element)
    return elements


def _extract_legacy_office(
    content: bytes,
    filename: str,
    content_type: str | None,
    extension: str,
) -> tuple[list[DocumentElement], list[str]]:
    office_format = _legacy_office_format(extension, content_type)
    warnings = [
        (
            "Legacy binary Office extraction used best-effort OLE stream text; convert to "
            "DOCX/PPTX/XLSX for full layout, chart, table, comment, and relationship extraction."
        )
    ]

    try:
        import olefile  # type: ignore
    except ImportError:
        warnings.append("Optional olefile package is not installed; printable-string fallback was used.")
        elements = _legacy_office_raw_fallback_elements(content, filename=filename, office_format=office_format)
        if not any(element.type == "legacy_office_text" for element in elements):
            warnings.append("No readable legacy Office text strings were found.")
        return elements, warnings

    elements: list[DocumentElement] = []
    try:
        ole = olefile.OleFileIO(io.BytesIO(content))
    except Exception as exc:
        warnings.append(f"OLE stream parsing failed ({exc}); printable-string fallback was used.")
        elements = _legacy_office_raw_fallback_elements(content, filename=filename, office_format=office_format)
        if not any(element.type == "legacy_office_text" for element in elements):
            warnings.append("No readable legacy Office text strings were found.")
        return elements, warnings

    try:
        stream_paths = list(ole.listdir(streams=True, storages=False) or [])
        if len(stream_paths) > LEGACY_OFFICE_MAX_STREAMS:
            warnings.append(
                f"Legacy Office stream inventory was limited to {LEGACY_OFFICE_MAX_STREAMS} of {len(stream_paths)} streams."
            )
        for stream_index, stream_path_parts in enumerate(stream_paths[:LEGACY_OFFICE_MAX_STREAMS], start=1):
            stream_path = _legacy_office_stream_label(stream_path_parts)
            try:
                stream = ole.openstream(stream_path_parts)
                try:
                    stream_bytes = stream.read(LEGACY_OFFICE_MAX_STREAM_BYTES + 1)
                finally:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
            except Exception as exc:
                warnings.append(f"Legacy Office stream {stream_path} could not be read ({exc}).")
                continue

            truncated = len(stream_bytes) > LEGACY_OFFICE_MAX_STREAM_BYTES
            if truncated:
                stream_bytes = stream_bytes[:LEGACY_OFFICE_MAX_STREAM_BYTES]
            metadata = _legacy_office_base_metadata(
                office_format=office_format,
                stream_index=stream_index,
                stream_path=stream_path,
                stream_bytes=stream_bytes,
                extraction_method="ole_stream_printable_strings",
                truncated=truncated,
            )
            stream_element = _as_element(
                "legacy_office_stream",
                f"Legacy Office {office_format} OLE stream {stream_path} ({len(stream_bytes)} bytes)",
                location=f"legacy office stream {stream_index}",
                **metadata,
            )
            if stream_element:
                elements.append(stream_element)
            text_element = _legacy_office_text_element(
                _binary_printable_strings(stream_bytes),
                location=f"legacy office stream {stream_index} text",
                metadata=metadata,
            )
            if text_element:
                elements.append(text_element)
    finally:
        close = getattr(ole, "close", None)
        if callable(close):
            close()

    if not any(element.type == "legacy_office_text" for element in elements):
        fallback_elements = _legacy_office_raw_fallback_elements(content, filename=filename, office_format=office_format)
        if fallback_elements:
            warnings.append("No readable OLE stream text was found; printable-string fallback was used.")
            elements.extend(fallback_elements)
    if not any(element.type == "legacy_office_text" for element in elements):
        warnings.append("No readable legacy Office text strings were found.")
    return elements, warnings


def _extract_csv(content: bytes) -> list[DocumentElement]:
    text = _decode_bytes(content)
    reader = csv.DictReader(io.StringIO(text))
    headers = reader.fieldnames or []
    rows = [headers] if headers else []
    rows.extend([[str(row.get(header, "")).strip() for header in headers] for row in reader])
    return _table_elements_from_rows(rows, location_prefix="row")


def _flatten_json(value: Any, path: str = "$") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        rows = []
        for key, item in value.items():
            rows.extend(_flatten_json(item, f"{path}.{key}"))
        return rows
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_flatten_json(item, f"{path}[{index}]"))
        return rows
    return [(path, str(value))]


def _extract_json(content: bytes) -> list[DocumentElement]:
    text = _decode_bytes(content)
    data = json.loads(text)
    elements: list[DocumentElement] = []
    for path, value in _flatten_json(data):
        element = _as_element("json_value", f"{path}: {value}", location=path, json_path=path)
        if element:
            elements.append(element)
    return elements


def _pdf_rect(value: Any) -> list[float] | None:
    if not value:
        return None
    try:
        return [round(float(part), 2) for part in value]
    except (TypeError, ValueError):
        return None


def _layout_zone(center_x_pct: float | None, center_y_pct: float | None) -> str | None:
    if center_x_pct is None or center_y_pct is None:
        return None
    horizontal = "left" if center_x_pct < 33.4 else "right" if center_x_pct > 66.6 else "center"
    vertical = "top" if center_y_pct < 33.4 else "bottom" if center_y_pct > 66.6 else "middle"
    return f"{vertical}_{horizontal}"


def _pdf_inches(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value / PDF_POINTS_PER_INCH, 3)


def _pdf_pct(value: float | None, total: float | None) -> float | None:
    if value is None or not total:
        return None
    return round((value / total) * 100, 1)


def _pdf_layout_metadata(rect: Any, page_rect: Any) -> dict[str, Any]:
    bbox = _pdf_rect(rect)
    page_bbox = _pdf_rect(page_rect)
    if not bbox or len(bbox) != 4:
        return {}

    page_x0 = page_bbox[0] if page_bbox and len(page_bbox) == 4 else 0.0
    page_y0 = page_bbox[1] if page_bbox and len(page_bbox) == 4 else 0.0
    page_width = max(0.0, page_bbox[2] - page_bbox[0]) if page_bbox and len(page_bbox) == 4 else None
    page_height = max(0.0, page_bbox[3] - page_bbox[1]) if page_bbox and len(page_bbox) == 4 else None

    x = bbox[0] - page_x0
    y = bbox[1] - page_y0
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    center_x = x + width / 2
    center_y = y + height / 2
    center_x_pct = _pdf_pct(center_x, page_width)
    center_y_pct = _pdf_pct(center_y, page_height)

    x_inches = _pdf_inches(x)
    y_inches = _pdf_inches(y)
    width_inches = _pdf_inches(width)
    height_inches = _pdf_inches(height)
    layout_bbox = None
    if None not in (x_inches, y_inches, width_inches, height_inches):
        layout_bbox = f"x={x_inches:.2f}in y={y_inches:.2f}in w={width_inches:.2f}in h={height_inches:.2f}in"

    metadata = {
        "layout_bbox": layout_bbox,
        "layout_zone": _layout_zone(center_x_pct, center_y_pct),
        "layout_x_inches": x_inches,
        "layout_y_inches": y_inches,
        "layout_width_inches": width_inches,
        "layout_height_inches": height_inches,
        "layout_center_x_inches": _pdf_inches(center_x),
        "layout_center_y_inches": _pdf_inches(center_y),
        "layout_left_pct": _pdf_pct(x, page_width),
        "layout_top_pct": _pdf_pct(y, page_height),
        "layout_width_pct": _pdf_pct(width, page_width),
        "layout_height_pct": _pdf_pct(height, page_height),
        "layout_center_x_pct": center_x_pct,
        "layout_center_y_pct": center_y_pct,
        "layout_x_points": round(x, 2),
        "layout_y_points": round(y, 2),
        "layout_width_points": round(width, 2),
        "layout_height_points": round(height, 2),
        "layout_center_x_points": round(center_x, 2),
        "layout_center_y_points": round(center_y, 2),
        "page_width_points": round(page_width, 2) if page_width is not None else None,
        "page_height_points": round(page_height, 2) if page_height is not None else None,
        "page_width_inches": _pdf_inches(page_width),
        "page_height_inches": _pdf_inches(page_height),
    }
    return {key: value for key, value in metadata.items() if value not in (None, "")}


def _rect_area(rect: list[float] | None) -> float:
    if not rect or len(rect) != 4:
        return 0.0
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _rect_overlap_ratio(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != 4 or len(right) != 4:
        return 0.0
    overlap_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    overlap_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    overlap_area = overlap_width * overlap_height
    smaller_area = min(_rect_area(left), _rect_area(right))
    if smaller_area <= 0:
        return 0.0
    return overlap_area / smaller_area


def _pdf_rect_overlaps_any(rect: list[float] | None, candidates: list[list[float]], *, threshold: float = 0.6) -> bool:
    return bool(rect and any(_rect_overlap_ratio(rect, candidate) >= threshold for candidate in candidates))


def _pdf_span_style(span: dict[str, Any]) -> dict[str, Any]:
    font = str(span.get("font") or "")
    font_lower = font.lower()
    flags = int(span.get("flags") or 0)
    return {
        "text": _clip(str(span.get("text") or ""), 240),
        "font": font,
        "size": round(float(span.get("size") or 0), 2),
        "bold": bool(flags & 16) or "bold" in font_lower,
        "italic": bool(flags & 2) or "italic" in font_lower or "oblique" in font_lower,
        "color": span.get("color"),
    }


def _pdf_text_block(block: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    lines = []
    runs = []
    for line in block.get("lines", []) or []:
        line_parts = []
        for span in line.get("spans", []) or []:
            text = str(span.get("text") or "")
            if text:
                line_parts.append(text)
                style = _pdf_span_style(span)
                if style["text"]:
                    runs.append(style)
        line_text = _clean_text(" ".join(line_parts))
        if line_text:
            lines.append(line_text)
    return _clean_text("\n".join(lines)), runs


def _pdf_text_element_type(text: str, runs: list[dict[str, Any]]) -> str:
    if not text:
        return "paragraph"
    max_size = max((float(run.get("size") or 0) for run in runs), default=0)
    bold_chars = sum(len(str(run.get("text") or "")) for run in runs if run.get("bold"))
    bold_ratio = bold_chars / max(len(text), 1)
    if len(text) <= 160 and (max_size >= 15 or bold_ratio >= 0.65 or text.isupper()):
        return "heading"
    return "paragraph"


def _pdf_document_metadata_elements(document: Any) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    metadata = getattr(document, "metadata", {}) or {}
    for raw_field, (metadata_field, label) in PDF_METADATA_FIELDS.items():
        raw_value = metadata.get(raw_field)
        value = _clean_text(str(raw_value or ""))
        if not value or value.lower() in {"none", "null"}:
            continue
        element = _as_element(
            "document_metadata",
            f"{label}: {value}",
            location=f"pdf metadata {metadata_field}",
            metadata_part="pdf:metadata",
            metadata_field=metadata_field,
            metadata_label=label,
            metadata_value=value,
        )
        if element:
            elements.append(element)
    return elements


def _pdf_outline_elements(document: Any) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    try:
        toc = document.get_toc(simple=True) or []
    except Exception:
        return elements

    for outline_index, item in enumerate(toc, start=1):
        try:
            level, title, page_number = item[:3]
        except (TypeError, ValueError):
            continue
        title = _clean_text(str(title or ""))
        if not title:
            continue
        element = _as_element(
            "outline_item",
            title,
            location=f"outline item {outline_index}",
            page=page_number if page_number and page_number > 0 else None,
            outline_level=level,
            outline_title=title,
        )
        if element:
            elements.append(element)
    return elements


def _extract_nested_document_elements(file_bytes: bytes, filename: str, extension: str) -> tuple[list[DocumentElement], str]:
    if extension in IMAGE_MIME_BY_EXTENSION:
        return _extract_image_file(file_bytes, filename), "image_file"
    if extension in {".txt", ".text", ".log", ".rst"}:
        return _extract_plain_text(file_bytes, extension), "plain_text"
    if extension in {".md", ".markdown"}:
        return _extract_plain_text(file_bytes, extension), "markdown_structure"
    if extension in {".html", ".htm"}:
        return _extract_html(file_bytes), "html_structure"
    if extension == ".json":
        return _extract_json(file_bytes), "json_tree"
    if extension == ".pdf":
        return _extract_pdf(file_bytes), "pdf_blocks"
    if extension == ".epub":
        return _extract_epub(file_bytes), "epub_html"
    if extension == ".eml":
        return _extract_email(file_bytes), "email_message"
    if extension in OOXML_WORD_EXTENSIONS:
        return _extract_docx(file_bytes), "docx_xml"
    if extension in OOXML_PRESENTATION_EXTENSIONS:
        return _extract_pptx(file_bytes), "pptx_xml"
    if extension in OOXML_SPREADSHEET_EXTENSIONS:
        return _extract_xlsx(file_bytes), "xlsx_xml"
    if extension in LEGACY_OFFICE_EXTENSIONS:
        elements, _ = _extract_legacy_office(file_bytes, filename, None, extension)
        return elements, "legacy_office_ole"
    if extension in {".odt", ".ods", ".odp"}:
        return _extract_opendocument(file_bytes, extension), f"{extension.lstrip('.')}_xml"
    if extension == ".rtf":
        text = re.sub(r"\\[a-z]+-?\d* ?", " ", _decode_bytes(file_bytes))
        text = re.sub(r"[{}]", " ", text)
        return _extract_plain_text(text.encode("utf-8"), ".txt"), "rtf_text"
    return [], ""


def _child_elements_from_embedded_bytes(
    file_bytes: bytes | None,
    filename: str,
    *,
    location_prefix: str,
    table_index: int,
    base_metadata: dict[str, Any],
    extraction_metadata_key: str,
    max_bytes: int,
) -> list[DocumentElement]:
    if not file_bytes or len(file_bytes) > max_bytes:
        return []

    extension = Path(filename or "").suffix.lower()
    elements: list[DocumentElement] = []
    method = ""

    try:
        if extension == ".csv":
            text = _decode_bytes(file_bytes)
            reader = csv.DictReader(io.StringIO(text))
            headers = reader.fieldnames or []
            rows = [headers] if headers else []
            rows.extend([[str(row.get(header, "")).strip() for header in headers] for row in reader])
            elements = _table_elements_from_rows(
                rows,
                location_prefix=f"{location_prefix} table",
                table=table_index,
                base_metadata={
                    **base_metadata,
                    "table_name": filename,
                    extraction_metadata_key: "csv_table",
                },
            )
            return elements[:80]

        elements, method = _extract_nested_document_elements(file_bytes, filename, extension)
        if method:
            for element in elements:
                element.location = f"{location_prefix} {element.location or element.type}"
                element.metadata.update({
                    **base_metadata,
                    extraction_metadata_key: method,
                })
    except Exception:
        return []

    return elements[:80]


def _embedded_file_child_elements(
    file_bytes: bytes | None,
    filename: str,
    *,
    attachment_index: int,
    base_metadata: dict[str, Any],
) -> list[DocumentElement]:
    return _child_elements_from_embedded_bytes(
        file_bytes,
        filename,
        location_prefix=f"embedded file {attachment_index}",
        table_index=attachment_index,
        base_metadata=base_metadata,
        extraction_metadata_key="embedded_file_extraction_method",
        max_bytes=MAX_EMBEDDED_FILE_PARSE_BYTES,
    )


def _archive_entry_child_elements(
    file_bytes: bytes | None,
    filename: str,
    *,
    entry_index: int,
    base_metadata: dict[str, Any],
) -> list[DocumentElement]:
    return _child_elements_from_embedded_bytes(
        file_bytes,
        filename,
        location_prefix=f"archive entry {entry_index}",
        table_index=entry_index,
        base_metadata=base_metadata,
        extraction_metadata_key="archive_entry_extraction_method",
        max_bytes=MAX_ARCHIVE_ENTRY_PARSE_BYTES,
    )


def _extract_zip_archive(content: bytes) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            entries = entries[:MAX_ARCHIVE_ENTRIES]
            for entry_index, entry in enumerate(entries, start=1):
                entry_path = _clean_text(entry.filename)
                if not entry_path:
                    continue
                entry_bytes: bytes | None = None
                if entry.file_size <= MAX_ARCHIVE_ENTRY_PARSE_BYTES:
                    try:
                        entry_bytes = archive.read(entry)
                    except Exception:
                        entry_bytes = None

                entry_mime = _mime_from_name(entry_path)
                entry_sha = hashlib.sha256(entry_bytes).hexdigest()[:16] if entry_bytes is not None else None
                base_metadata = {
                    "archive_entry_index": entry_index,
                    "archive_entry_path": entry_path,
                    "archive_entry_filename": Path(entry_path).name,
                    "archive_entry_mime": entry_mime,
                    "archive_entry_size_bytes": entry.file_size,
                    "archive_entry_compressed_size_bytes": entry.compress_size,
                    "archive_entry_sha256": entry_sha,
                }
                label = _media_element_text(
                    media_kind="archive_entry",
                    media_part=entry_path,
                    media_mime=entry_mime,
                    size_bytes=entry.file_size,
                )
                element = _as_element(
                    "archive_entry",
                    label,
                    location=f"archive entry {entry_index}",
                    **base_metadata,
                )
                if element:
                    elements.append(element)

                if entry_bytes is not None:
                    elements.extend(_archive_entry_child_elements(
                        entry_bytes,
                        entry_path,
                        entry_index=entry_index,
                        base_metadata=base_metadata,
                    ))
    except zipfile.BadZipFile:
        return _extract_plain_text(content, ".txt")

    return elements


def _pdf_embedded_file_elements(document: Any) -> list[DocumentElement]:
    count = 0
    count_fn = getattr(document, "embfile_count", None)
    if callable(count_fn):
        try:
            count = int(count_fn() or 0)
        except Exception:
            count = 0
    if count <= 0:
        return []

    names: list[Any] = []
    names_fn = getattr(document, "embfile_names", None)
    if callable(names_fn):
        try:
            names = list(names_fn() or [])
        except Exception:
            names = []
    if not names:
        names = list(range(count))

    elements: list[DocumentElement] = []
    for attachment_index, item in enumerate(names, start=1):
        try:
            info = document.embfile_info(item) or {}
        except Exception:
            info = {}

        file_bytes: bytes | None = None
        try:
            raw_bytes = document.embfile_get(item)
            if isinstance(raw_bytes, bytes):
                file_bytes = raw_bytes
        except Exception:
            file_bytes = None

        name = _clean_text(str(info.get("name") or item or f"embedded-file-{attachment_index}"))
        filename = _clean_text(str(info.get("filename") or info.get("ufilename") or name))
        description = _clean_text(str(info.get("description") or ""))
        file_size = len(file_bytes) if file_bytes is not None else info.get("size") or info.get("length")
        try:
            file_size = int(file_size) if file_size is not None else None
        except (TypeError, ValueError):
            file_size = None

        binary_metadata = _binary_metadata_from_bytes(file_bytes, filename or name)
        if file_size is not None and not binary_metadata.get("media_size_bytes"):
            binary_metadata["media_size_bytes"] = file_size
        file_sha = hashlib.sha256(file_bytes).hexdigest()[:16] if file_bytes is not None else None
        if file_sha and not binary_metadata.get("media_sha256"):
            binary_metadata["media_sha256"] = file_sha
        mime = binary_metadata.get("media_mime") or _mime_from_name(filename or name)

        label = _media_element_text(
            media_kind="pdf_attachment",
            media_part=filename or name,
            media_mime=mime,
            size_bytes=file_size,
        )
        if description:
            label = "; ".join(_unique_preserve_order([label, description]))

        element = _as_element(
            "embedded_object",
            label,
            location=f"embedded file {attachment_index}",
            media_kind="pdf_attachment",
            media_part=filename or name,
            embedded_file_index=attachment_index,
            embedded_file_name=name,
            embedded_file_filename=filename,
            embedded_file_description=description,
            embedded_file_mime=mime,
            embedded_file_size_bytes=file_size,
            embedded_file_sha256=file_sha,
            embedded_file_created=info.get("creationDate"),
            embedded_file_modified=info.get("modDate"),
            **binary_metadata,
        )
        if element:
            elements.append(element)
        if file_bytes is not None:
            child_metadata = {
                "media_kind": "pdf_attachment",
                "media_part": filename or name,
                "embedded_file_index": attachment_index,
                "embedded_file_name": name,
                "embedded_file_filename": filename,
                "embedded_file_description": description,
                "embedded_file_mime": mime,
                "embedded_file_size_bytes": file_size,
                "embedded_file_sha256": file_sha,
                "embedded_file_created": info.get("creationDate"),
                "embedded_file_modified": info.get("modDate"),
            }
            elements.extend(_embedded_file_child_elements(
                file_bytes,
                filename or name,
                attachment_index=attachment_index,
                base_metadata=child_metadata,
            ))

    return elements


def _pdf_annotation_elements(page: Any, page_number: int, page_rect: list[float] | None) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    annots = getattr(page, "annots", None)
    if not callable(annots):
        return elements

    try:
        annotations = list(annots() or [])
    except Exception:
        return elements

    for annotation_index, annotation in enumerate(annotations, start=1):
        annotation_type = getattr(annotation, "type", None)
        if isinstance(annotation_type, tuple):
            annotation_type_name = str(annotation_type[1] or annotation_type[0])
        else:
            annotation_type_name = str(annotation_type or "")
        info = getattr(annotation, "info", {}) or {}
        if not isinstance(info, dict):
            info = {}
        author = _clean_text(str(info.get("title") or ""))
        subject = _clean_text(str(info.get("subject") or ""))
        content = _clean_text(str(info.get("content") or info.get("contents") or ""))
        name = _clean_text(str(info.get("name") or ""))
        label_parts = [part for part in (annotation_type_name, subject, content or name) if part]
        annotation_bbox = _pdf_rect(getattr(annotation, "rect", None))
        element = _as_element(
            "annotation",
            ": ".join(label_parts) if label_parts else f"Annotation {annotation_index}",
            location=f"page {page_number} annotation {annotation_index}",
            page=page_number,
            annotation_type=annotation_type_name,
            annotation_author=author,
            annotation_subject=subject,
            annotation_content=content,
            annotation_name=name,
            bbox=annotation_bbox,
            **_pdf_layout_metadata(annotation_bbox, page_rect),
        )
        if element:
            elements.append(element)
    return elements


def _pdf_table_elements(
    page: Any,
    page_number: int,
    page_label: str,
    page_rect: list[float] | None,
) -> list[DocumentElement]:
    find_tables = getattr(page, "find_tables", None)
    if not callable(find_tables):
        return []

    try:
        tables_result = find_tables()
    except Exception:
        return []

    tables = getattr(tables_result, "tables", None)
    if tables is None:
        try:
            tables = list(tables_result or [])
        except TypeError:
            tables = []

    elements: list[DocumentElement] = []
    for table_index, table in enumerate(tables, start=1):
        extract = getattr(table, "extract", None)
        if not callable(extract):
            continue
        try:
            raw_rows = extract() or []
        except Exception:
            continue

        rows = [
            [_clean_text(str(cell or "")) for cell in row]
            for row in raw_rows
            if isinstance(row, (list, tuple))
        ]
        row_count, column_count = _table_shape(rows)
        if row_count <= 0:
            continue

        table_bbox = _pdf_rect(getattr(table, "bbox", None))
        table_layout_metadata = _pdf_layout_metadata(table_bbox, page_rect)
        elements.extend(_table_elements_from_rows(
            rows,
            location_prefix=f"{page_label} table {table_index}",
            table=table_index,
            base_metadata={
                "page": page_number,
                "table_source": "pdf_find_tables",
                "table_bbox": table_bbox,
                "table_row_count": row_count,
                "table_column_count": column_count,
                **table_layout_metadata,
            },
        ))

    return elements


def _pdf_page_ocr_element(
    page: Any,
    page_number: int,
    page_label: str,
    page_rect: list[float] | None,
) -> DocumentElement | None:
    if not _env_bool("GRAPHRAG_ENABLE_IMAGE_OCR", False):
        return None

    try:
        pixmap = page.get_pixmap(alpha=False)
        image_bytes = pixmap.tobytes("png")
    except Exception:
        return None

    ocr_text = _optional_image_ocr(image_bytes, limit=1600)
    if not ocr_text:
        return None

    image_metadata = _image_metadata_from_bytes(image_bytes, "page.png", enable_ocr=False)
    return _as_element(
        "page_ocr_text",
        ocr_text,
        location=f"{page_label} OCR",
        page=page_number,
        bbox=page_rect,
        ocr_source="pdf_page_render",
        ocr_engine="pytesseract",
        **_pdf_layout_metadata(page_rect, page_rect),
        **image_metadata,
    )


def _extract_pdf(content: bytes) -> list[DocumentElement]:
    try:
        import fitz
    except Exception:
        return _extract_plain_text(content, ".txt")

    elements: list[DocumentElement] = []
    with fitz.open(stream=content, filetype="pdf") as document:
        elements.extend(_pdf_document_metadata_elements(document))
        elements.extend(_pdf_outline_elements(document))
        elements.extend(_pdf_embedded_file_elements(document))
        for page_index in range(len(document)):
            page = document.load_page(page_index)
            page_label = f"page {page_index + 1}"
            page_rect = _pdf_rect(getattr(page, "rect", None))
            page_widgets: list[Any] = []
            widget_rects: list[list[float]] = []
            widgets = getattr(page, "widgets", None)
            if callable(widgets):
                try:
                    page_widgets = list(widgets() or [])
                    for widget in page_widgets:
                        rect = _pdf_rect(getattr(widget, "rect", None))
                        if rect:
                            widget_rects.append(rect)
                except Exception:
                    page_widgets = []
                    widget_rects = []

            try:
                page_dict = page.get_text("dict")
            except Exception:
                page_dict = {}

            blocks = page_dict.get("blocks", []) if isinstance(page_dict, dict) else []
            page_has_text = False
            if blocks:
                for block_index, block in enumerate(blocks, start=1):
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == 0:
                        block_text, runs = _pdf_text_block(block)
                        block_bbox = _pdf_rect(block.get("bbox"))
                        block_layout_metadata = _pdf_layout_metadata(block_bbox, page_rect)
                        if _pdf_rect_overlaps_any(block_bbox, widget_rects):
                            continue
                        if block_text:
                            page_has_text = True
                        element_type = _pdf_text_element_type(block_text, runs)
                        emphasis_terms = [run["text"] for run in runs if run.get("bold") or run.get("italic") or run.get("underline")]
                        element = _as_element(
                            element_type,
                            block_text,
                            location=f"{page_label} block {block_index}",
                            page=page_index + 1,
                            bbox=block_bbox,
                            runs=runs[:24] if runs else None,
                            emphasis_terms=emphasis_terms[:12] if emphasis_terms else None,
                            style_summary=_style_summary_from_runs(runs),
                            **block_layout_metadata,
                        )
                    elif block_type == 1:
                        block_bbox = _pdf_rect(block.get("bbox"))
                        image_bytes = block.get("image") if isinstance(block.get("image"), bytes) else None
                        image_metadata = _image_metadata_from_bytes(image_bytes, block.get("ext"))
                        if not image_metadata.get("image_width") and block.get("width"):
                            image_metadata["image_width"] = block.get("width")
                        if not image_metadata.get("image_height") and block.get("height"):
                            image_metadata["image_height"] = block.get("height")
                        label = _image_element_text(f"Image block {block_index}", metadata=image_metadata)
                        element = _as_element(
                            "image",
                            label,
                            location=f"{page_label} image {block_index}",
                            page=page_index + 1,
                            bbox=block_bbox,
                            ext=block.get("ext"),
                            image=block.get("number"),
                            **_pdf_layout_metadata(block_bbox, page_rect),
                            **image_metadata,
                        )
                    else:
                        element = None
                    if element:
                        elements.append(element)
            else:
                fallback_text = page.get_text()
                page_has_text = bool(_clean_text(fallback_text))
                element = _as_element("page_text", fallback_text, location=page_label, page=page_index + 1)
                if element:
                    elements.append(element)

            if not page_has_text:
                element = _pdf_page_ocr_element(page, page_index + 1, page_label, page_rect)
                if element:
                    elements.append(element)

            try:
                for link_index, link in enumerate(page.get_links(), start=1):
                    uri = link.get("uri")
                    if not uri:
                        continue
                    link_bbox = _pdf_rect(link.get("from"))
                    element = _as_element(
                        "link",
                        uri,
                        location=f"{page_label} link {link_index}",
                        page=page_index + 1,
                        bbox=link_bbox,
                        uri=uri,
                        kind=link.get("kind"),
                        **_pdf_layout_metadata(link_bbox, page_rect),
                    )
                    if element:
                        elements.append(element)
            except Exception:
                pass

            if page_widgets:
                try:
                    for widget_index, widget in enumerate(page_widgets, start=1):
                        label = (
                            getattr(widget, "field_label", None)
                            or getattr(widget, "field_name", None)
                            or f"field {widget_index}"
                        )
                        value = getattr(widget, "field_value", None)
                        field_type = getattr(widget, "field_type_string", None) or getattr(widget, "field_type", None)
                        widget_bbox = _pdf_rect(getattr(widget, "rect", None))
                        element = _as_element(
                            "form_field",
                            f"{label}: {value}" if value not in {None, ""} else str(label),
                            location=f"{page_label} form field {widget_index}",
                            page=page_index + 1,
                            label=label,
                            value=value,
                            field_type=field_type,
                            bbox=widget_bbox,
                            **_pdf_layout_metadata(widget_bbox, page_rect),
                        )
                        if element:
                            elements.append(element)
                except Exception:
                    pass

            elements.extend(_pdf_annotation_elements(page, page_index + 1, page_rect))
            elements.extend(_pdf_table_elements(page, page_index + 1, page_label, page_rect))
    return elements


def _word_attr(node: ET.Element | None, local_name: str) -> str | None:
    if node is None:
        return None
    return node.attrib.get(f"{{{XML_NS['w']}}}{local_name}")


def _word_hex_color(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().lstrip("#")
    if cleaned.lower() in {"auto", "none"}:
        return None
    if re.fullmatch(r"[0-9A-Fa-f]{6}", cleaned):
        return f"#{cleaned.upper()}"
    return cleaned


def _word_twips_inches(value: str | None) -> float | None:
    parsed = _first_int(value or "")
    return round(parsed / 1440, 4) if parsed is not None else None


def _word_border_width_inches(value: str | None) -> float | None:
    parsed = _first_int(value or "")
    if parsed is None:
        return None
    return round((parsed / 8) / 72, 4)


def _docx_table_cell_visual_metadata(cell_properties: ET.Element | None) -> dict[str, Any]:
    if cell_properties is None:
        return {}

    metadata: dict[str, Any] = {}
    style_node = cell_properties.find("w:tcStyle", XML_NS)
    width_node = cell_properties.find("w:tcW", XML_NS)
    shading_node = cell_properties.find("w:shd", XML_NS)
    vertical_align_node = cell_properties.find("w:vAlign", XML_NS)
    text_direction_node = cell_properties.find("w:textDirection", XML_NS)

    metadata["table_cell_style_name"] = _word_attr(style_node, "val")
    width_value = _word_attr(width_node, "w")
    width_type = _word_attr(width_node, "type")
    metadata["table_cell_width_value"] = width_value
    metadata["table_cell_width_type"] = width_type
    if not width_type or width_type == "dxa":
        metadata["table_cell_width_inches"] = _word_twips_inches(width_value)
    metadata["table_cell_text_anchor"] = _word_attr(vertical_align_node, "val")
    metadata["table_cell_vertical_align"] = _word_attr(vertical_align_node, "val")
    metadata["table_cell_text_direction"] = _word_attr(text_direction_node, "val")

    fill_color = _word_hex_color(_word_attr(shading_node, "fill"))
    if fill_color:
        metadata["table_cell_fill_color"] = fill_color
    shading_pattern = _word_attr(shading_node, "val")
    if shading_pattern and shading_pattern.lower() not in {"clear", "nil"}:
        metadata["table_cell_shading_pattern"] = shading_pattern

    margins = cell_properties.find("w:tcMar", XML_NS)
    if margins is not None:
        for side in ("left", "right", "top", "bottom"):
            margin = margins.find(f"w:{side}", XML_NS)
            inches = _word_twips_inches(_word_attr(margin, "w"))
            if inches is not None:
                metadata[f"table_cell_margin_{side}_inches"] = inches

    border_parts: list[str] = []
    borders = cell_properties.find("w:tcBorders", XML_NS)
    if borders is not None:
        for side in ("left", "right", "top", "bottom"):
            border = borders.find(f"w:{side}", XML_NS)
            if border is None:
                continue
            border_style = _word_attr(border, "val")
            color = _word_hex_color(_word_attr(border, "color"))
            width = _word_border_width_inches(_word_attr(border, "sz"))
            if color:
                metadata[f"table_cell_border_{side}_color"] = color
            if width is not None:
                metadata[f"table_cell_border_{side}_width_inches"] = width
            border_summary = " ".join(
                part for part in [
                    side,
                    border_style if border_style and border_style != "nil" else "",
                    color or "",
                    f"{width:.4f}in" if width is not None else "",
                ]
                if part
            )
            if border_summary:
                border_parts.append(border_summary)

    summary_parts = []
    if fill_color:
        summary_parts.append(f"fill {fill_color}")
    if metadata.get("table_cell_style_name"):
        summary_parts.append(f"style {metadata['table_cell_style_name']}")
    if metadata.get("table_cell_text_anchor"):
        summary_parts.append(f"text anchor {metadata['table_cell_text_anchor']}")
    if metadata.get("table_cell_text_direction"):
        summary_parts.append(f"text direction {metadata['table_cell_text_direction']}")
    if metadata.get("table_cell_width_inches") is not None:
        width_type = metadata.get("table_cell_width_type")
        width_text = f"width {metadata['table_cell_width_inches']:.3f}in"
        if width_type:
            width_text += f" {width_type}"
        summary_parts.append(width_text)
    if border_parts:
        summary_parts.append("borders " + "; ".join(border_parts))
    margins_summary = [
        (label, metadata.get(key))
        for label, key in (
            ("L", "table_cell_margin_left_inches"),
            ("R", "table_cell_margin_right_inches"),
            ("T", "table_cell_margin_top_inches"),
            ("B", "table_cell_margin_bottom_inches"),
        )
    ]
    if any(value is not None for _, value in margins_summary):
        summary_parts.append(
            "margins "
            + ", ".join(f"{label}={value:.3f}in" for label, value in margins_summary if value is not None)
        )
    if summary_parts:
        metadata["table_cell_visual_summary"] = "; ".join(summary_parts)

    return {key: value for key, value in metadata.items() if value not in (None, "", [], {})}


def _word_run_styles(node: ET.Element) -> list[dict[str, Any]]:
    runs = []
    for run in node.findall(".//w:r", XML_NS):
        text = _xml_text(run, ("t",))
        if not text:
            continue
        run_properties = run.find("w:rPr", XML_NS)
        bold = run_properties.find("w:b", XML_NS) is not None if run_properties is not None else False
        italic = run_properties.find("w:i", XML_NS) is not None if run_properties is not None else False
        underline = run_properties.find("w:u", XML_NS) is not None if run_properties is not None else False
        color = _word_attr(run_properties.find("w:color", XML_NS) if run_properties is not None else None, "val")
        size = _word_attr(run_properties.find("w:sz", XML_NS) if run_properties is not None else None, "val")
        runs.append({
            "text": _clip(text, 240),
            "bold": bold,
            "italic": italic,
            "underline": underline,
            "color": color,
            "size_half_points": size,
        })
    return runs


def _run_style_tags(run: dict[str, Any]) -> list[str]:
    tags = []
    if run.get("bold"):
        tags.append("bold")
    if run.get("italic"):
        tags.append("italic")
    if run.get("underline"):
        tags.append("underline")
    color_tag = _run_color_style_tag("color", run.get("color"))
    if color_tag:
        tags.append(color_tag)
    highlight_tag = _run_color_style_tag("highlight", run.get("highlight_color"))
    if highlight_tag:
        tags.append(highlight_tag)
    size_tag = _run_size_style_tag(run)
    if size_tag:
        tags.append(size_tag)
    return tags


def _run_color_style_tag(label: str, value: Any) -> str | None:
    color = _clean_text(str(value or ""))
    if not color or color.lower() in {"0", "000000", "none", "auto"}:
        return None
    raw = color.lstrip("#")
    if re.fullmatch(r"[0-9A-Fa-f]{6}", raw):
        color = f"#{raw.upper()}"
    return f"{label} {color}"


def _run_size_style_tag(run: dict[str, Any]) -> str | None:
    size_value = run.get("size_1_100_points")
    divisor = 100
    if size_value in (None, ""):
        size_value = run.get("size_half_points")
        divisor = 2
    parsed = _first_int(str(size_value or ""))
    if parsed is None or parsed <= 0:
        return None
    points = parsed / divisor
    return f"{points:g}pt"


def _style_summary_from_runs(runs: list[dict[str, Any]], bullet_levels: list[int] | None = None) -> str | None:
    tags = _unique_preserve_order([
        tag
        for run in runs or []
        for tag in _run_style_tags(run)
    ])
    if bullet_levels:
        levels = ", ".join(str(level) for level in _unique_preserve_order([str(level) for level in bullet_levels[:8]]))
        if levels:
            tags.append(f"bullet levels {levels}")
    return "; ".join(tags) if tags else None


def _styled_text_terms(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    terms: list[dict[str, Any]] = []
    seen: set[str] = set()
    emphasis_terms = metadata.get("emphasis_terms") if isinstance(metadata.get("emphasis_terms"), list) else []
    runs = metadata.get("runs") if isinstance(metadata.get("runs"), list) else []

    def add(text: str, styles: list[str] | None = None) -> None:
        cleaned = _clip(_clean_text(str(text or "")), 120)
        key = cleaned.lower()
        if not cleaned or key in seen:
            return
        seen.add(key)
        terms.append({
            "text": cleaned,
            "styles": _unique_preserve_order(styles or []),
        })

    for term in emphasis_terms:
        term_text = _clean_text(str(term or ""))
        matching_styles = _unique_preserve_order([
            tag
            for run in runs
            if term_text and term_text.lower() in _clean_text(str(run.get("text") or "")).lower()
            for tag in _run_style_tags(run)
        ])
        add(term_text, matching_styles)

    for run in runs:
        style_tags = _run_style_tags(run)
        if style_tags:
            add(str(run.get("text") or ""), style_tags)

    return terms


def _drawing_attr(node: ET.Element | None, local_name: str) -> str | None:
    if node is None:
        return None
    return node.attrib.get(local_name) or node.attrib.get(f"{{{XML_NS['a']}}}{local_name}")


def _pptx_shape_metadata(shape: ET.Element) -> dict[str, str | None]:
    properties = shape.find(".//p:cNvPr", XML_NS)
    return {
        "shape_id": _drawing_attr(properties, "id"),
        "shape_name": _drawing_attr(properties, "name"),
        "shape_title": _drawing_attr(properties, "title"),
        "shape_description": _drawing_attr(properties, "descr"),
    }


def _pptx_shape_metadata_text(metadata: dict[str, str | None]) -> str:
    title = _clean_text(str(metadata.get("shape_title") or ""))
    description = _clean_text(str(metadata.get("shape_description") or ""))
    name = _clean_text(str(metadata.get("shape_name") or ""))
    if title and description and title.lower() != description.lower():
        return f"{title}: {description}"
    if title or description:
        return title or description
    if name and not GENERIC_POWERPOINT_SHAPE_NAME_RE.match(name):
        return name
    return ""


def _pptx_shape_graph_label(metadata: dict[str, str | None], text: str | None = None) -> str:
    title = _clean_text(str(metadata.get("shape_title") or ""))
    text_label = _clean_text(str(text or ""))
    description = _clean_text(str(metadata.get("shape_description") or ""))
    name = _clean_text(str(metadata.get("shape_name") or ""))
    if title:
        return title
    if text_label:
        return _clip(text_label, 120)
    if description:
        return description
    if name and not GENERIC_POWERPOINT_SHAPE_NAME_RE.match(name):
        return name
    return ""


def _pptx_connection_endpoint(connector: ET.Element, local_name: str) -> tuple[str | None, str | None]:
    endpoint = _first_descendant_by_local_name(connector, local_name)
    if endpoint is None:
        return None, None
    return _attr_by_local_name(endpoint, "id"), _attr_by_local_name(endpoint, "idx")


def _pptx_connector_relation(metadata: dict[str, str | None], text: str | None = None) -> str:
    relation = (
        _clean_text(str(text or ""))
        or _clean_text(str(metadata.get("shape_title") or ""))
        or _clean_text(str(metadata.get("shape_description") or ""))
        or _clean_text(str(metadata.get("shape_name") or ""))
    )
    if relation and not GENERIC_POWERPOINT_SHAPE_NAME_RE.match(relation):
        return relation
    return "connects_to"


def _pptx_connectable_shape_labels(root: ET.Element) -> dict[str, str]:
    labels: dict[str, str] = {}
    fallback_kinds = {
        "sp": "shape",
        "pic": "image",
        "graphicFrame": "object",
        "grpSp": "group",
    }
    for node in root.iter():
        kind = fallback_kinds.get(_local_name(node.tag))
        if not kind:
            continue
        metadata = _pptx_shape_metadata(node)
        shape_id = metadata.get("shape_id")
        if not shape_id:
            continue
        text = _xml_text(node, ("t",))
        label = _pptx_shape_graph_label(metadata, text)
        if not label:
            label = _clean_text(str(metadata.get("shape_name") or ""))
        if not label:
            label = f"{kind} {shape_id}"
        labels[str(shape_id)] = label
    return labels


def _pptx_group_records(
    root: ET.Element,
    slide_size_metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    child_context_by_id: dict[str, dict[str, Any]] = {}

    for group_index, group in enumerate(root.findall(".//p:grpSp", XML_NS), start=1):
        group_metadata = _pptx_shape_metadata(group)
        group_id = _clean_text(str(group_metadata.get("shape_id") or ""))
        child_ids: list[str] = []
        child_labels: list[str] = []

        for child in group.iter():
            if child is group or _local_name(child.tag) not in {"sp", "pic", "graphicFrame", "cxnSp", "grpSp"}:
                continue
            child_metadata = _pptx_shape_metadata(child)
            child_id = _clean_text(str(child_metadata.get("shape_id") or ""))
            if not child_id or child_id == group_id:
                continue
            child_text = _xml_text(child, ("t",))
            child_label = (
                _pptx_shape_graph_label(child_metadata, child_text)
                or _clean_text(str(child_metadata.get("shape_name") or ""))
                or f"shape {child_id}"
            )
            if child_id not in child_ids:
                child_ids.append(child_id)
            if child_label and child_label not in child_labels:
                child_labels.append(child_label)

        group_label = (
            _pptx_shape_graph_label(group_metadata)
            or (f"Group: {', '.join(child_labels[:3])}" if child_labels else "")
            or (f"Group {group_id}" if group_id else f"Group {group_index}")
        )
        group_geometry = _pptx_geometry_metadata(group, slide_size_metadata)
        record = {
            "group": group,
            "group_index": group_index,
            "group_id": group_id or None,
            "group_name": group_metadata.get("shape_name"),
            "group_title": group_metadata.get("shape_title"),
            "group_description": group_metadata.get("shape_description"),
            "group_label": group_label,
            "group_child_shape_ids": child_ids[:40] or None,
            "group_child_labels": child_labels[:40] or None,
            "group_child_count": len(child_ids),
            "geometry": group_geometry,
        }
        records.append(record)

        context = {
            "parent_group_id": group_id or None,
            "parent_group_label": group_label,
            "parent_group_index": group_index,
        }
        for child_id in child_ids:
            child_context_by_id[child_id] = context

    return records, child_context_by_id


def _first_descendant_by_local_name(node: ET.Element, *local_names: str) -> ET.Element | None:
    wanted = set(local_names)
    for descendant in node.iter():
        if _local_name(descendant.tag) in wanted:
            return descendant
    return None


def _pptx_int_attr(node: ET.Element | None, local_name: str) -> int | None:
    value = _attr_by_local_name(node, local_name)
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _pptx_inches(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / PPTX_EMUS_PER_INCH, 3)


def _pptx_points(value: int | None) -> float | None:
    if value is None:
        return None
    return round((value / PPTX_EMUS_PER_INCH) * 72, 2)


def _pptx_pct(value: int | None, total: int | None) -> float | None:
    if value is None or not total:
        return None
    return round((value / total) * 100, 1)


def _pptx_layout_zone(center_x_pct: float | None, center_y_pct: float | None) -> str | None:
    return _layout_zone(center_x_pct, center_y_pct)


def _pptx_slide_size_metadata(content: bytes) -> dict[str, Any]:
    root = _read_zip_xml(content, "ppt/presentation.xml")
    if root is None:
        return {}
    slide_size = _first_descendant_by_local_name(root, "sldSz")
    width = _pptx_int_attr(slide_size, "cx")
    height = _pptx_int_attr(slide_size, "cy")
    return {
        "slide_width_emu": width,
        "slide_height_emu": height,
        "slide_width_inches": _pptx_inches(width),
        "slide_height_inches": _pptx_inches(height),
    }


def _pptx_geometry_metadata(node: ET.Element, slide_size_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    xfrm = _first_descendant_by_local_name(node, "xfrm")
    if xfrm is None:
        return {}
    off = _first_descendant_by_local_name(xfrm, "off")
    ext = _first_descendant_by_local_name(xfrm, "ext")
    x = _pptx_int_attr(off, "x")
    y = _pptx_int_attr(off, "y")
    width = _pptx_int_attr(ext, "cx")
    height = _pptx_int_attr(ext, "cy")
    if all(value is None for value in (x, y, width, height)):
        return {}

    slide_size = slide_size_metadata or {}
    slide_width = slide_size.get("slide_width_emu") if isinstance(slide_size.get("slide_width_emu"), int) else None
    slide_height = slide_size.get("slide_height_emu") if isinstance(slide_size.get("slide_height_emu"), int) else None
    center_x = x + width / 2 if x is not None and width is not None else None
    center_y = y + height / 2 if y is not None and height is not None else None
    center_x_pct = _pptx_pct(int(center_x) if center_x is not None else None, slide_width)
    center_y_pct = _pptx_pct(int(center_y) if center_y is not None else None, slide_height)

    x_inches = _pptx_inches(x)
    y_inches = _pptx_inches(y)
    width_inches = _pptx_inches(width)
    height_inches = _pptx_inches(height)
    bbox = None
    if None not in (x_inches, y_inches, width_inches, height_inches):
        bbox = f"x={x_inches:.2f}in y={y_inches:.2f}in w={width_inches:.2f}in h={height_inches:.2f}in"

    rotation_units = _pptx_int_attr(xfrm, "rot")
    rotation_degrees = round(rotation_units / 60000, 3) if rotation_units is not None else None
    metadata = {
        "layout_x_emu": x,
        "layout_y_emu": y,
        "layout_width_emu": width,
        "layout_height_emu": height,
        "layout_x_inches": x_inches,
        "layout_y_inches": y_inches,
        "layout_width_inches": width_inches,
        "layout_height_inches": height_inches,
        "layout_center_x_inches": _pptx_inches(int(center_x) if center_x is not None else None),
        "layout_center_y_inches": _pptx_inches(int(center_y) if center_y is not None else None),
        "layout_left_pct": _pptx_pct(x, slide_width),
        "layout_top_pct": _pptx_pct(y, slide_height),
        "layout_width_pct": _pptx_pct(width, slide_width),
        "layout_height_pct": _pptx_pct(height, slide_height),
        "layout_center_x_pct": center_x_pct,
        "layout_center_y_pct": center_y_pct,
        "layout_zone": _pptx_layout_zone(center_x_pct, center_y_pct),
        "layout_bbox": bbox,
        "layout_rotation_degrees": rotation_degrees,
        "layout_flip_h": _attr_by_local_name(xfrm, "flipH"),
        "layout_flip_v": _attr_by_local_name(xfrm, "flipV"),
    }
    for key in ("slide_width_emu", "slide_height_emu", "slide_width_inches", "slide_height_inches"):
        if slide_size.get(key) is not None:
            metadata[key] = slide_size[key]
    return {key: value for key, value in metadata.items() if value not in (None, "")}


def _pptx_point_position_metadata(node: ET.Element | None, slide_size_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    if node is None:
        return {}
    position = _first_descendant_by_local_name(node, "pos")
    x = _pptx_int_attr(position, "x")
    y = _pptx_int_attr(position, "y")
    if x is None and y is None:
        return {}

    slide_size = slide_size_metadata or {}
    slide_width = slide_size.get("slide_width_emu") if isinstance(slide_size.get("slide_width_emu"), int) else None
    slide_height = slide_size.get("slide_height_emu") if isinstance(slide_size.get("slide_height_emu"), int) else None
    center_x_pct = _pptx_pct(x, slide_width)
    center_y_pct = _pptx_pct(y, slide_height)
    x_inches = _pptx_inches(x)
    y_inches = _pptx_inches(y)
    metadata = {
        "layout_x_emu": x,
        "layout_y_emu": y,
        "layout_x_inches": x_inches,
        "layout_y_inches": y_inches,
        "layout_x_points": _pptx_points(x),
        "layout_y_points": _pptx_points(y),
        "layout_center_x_emu": x,
        "layout_center_y_emu": y,
        "layout_center_x_inches": x_inches,
        "layout_center_y_inches": y_inches,
        "layout_center_x_points": _pptx_points(x),
        "layout_center_y_points": _pptx_points(y),
        "layout_left_pct": center_x_pct,
        "layout_top_pct": center_y_pct,
        "layout_center_x_pct": center_x_pct,
        "layout_center_y_pct": center_y_pct,
        "layout_position": " ".join(
            part for part in [
                f"x={x_inches:.2f}in" if x_inches is not None else None,
                f"y={y_inches:.2f}in" if y_inches is not None else None,
            ]
            if part
        ) or None,
        "layout_zone": _pptx_layout_zone(center_x_pct, center_y_pct),
    }
    for key in ("slide_width_emu", "slide_height_emu", "slide_width_inches", "slide_height_inches"):
        if slide_size.get(key) is not None:
            metadata[key] = slide_size[key]
    return {key: value for key, value in metadata.items() if value not in (None, "")}


def _drawing_run_styles(node: ET.Element) -> list[dict[str, Any]]:
    runs = []
    for run in node.findall(".//a:r", XML_NS):
        text = _xml_text(run, ("t",))
        if not text:
            continue
        run_properties = run.find("a:rPr", XML_NS)
        bold = _drawing_attr(run_properties, "b") in {"1", "true"}
        italic = _drawing_attr(run_properties, "i") in {"1", "true"}
        underline = bool(_drawing_attr(run_properties, "u"))
        size = _drawing_attr(run_properties, "sz")
        color = _drawing_color_value(run_properties)
        highlight = run_properties.find("a:highlight", XML_NS) if run_properties is not None else None
        highlight_color = _drawing_color_value(highlight)
        runs.append({
            "text": _clip(text, 240),
            "bold": bold,
            "italic": italic,
            "underline": underline,
            "color": color,
            "highlight_color": highlight_color,
            "size_1_100_points": size,
        })
    return runs


def _drawing_color_value(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    for color_node in node.iter():
        local = _local_name(color_node.tag)
        if local == "srgbClr":
            value = _attr_by_local_name(color_node, "val")
            return f"#{value.lstrip('#').upper()}" if value else None
        if local == "schemeClr":
            value = _attr_by_local_name(color_node, "val")
            return f"scheme:{value}" if value else None
        if local == "sysClr":
            value = _attr_by_local_name(color_node, "lastClr") or _attr_by_local_name(color_node, "val")
            return f"#{value.lstrip('#').upper()}" if value and re.fullmatch(r"[0-9A-Fa-f]{6}", value) else f"system:{value}" if value else None
        if local == "prstClr":
            value = _attr_by_local_name(color_node, "val")
            return f"preset:{value}" if value else None
    return None


def _pptx_cell_margin_inches(value: str | None) -> float | None:
    parsed = _first_int(value or "")
    return _pptx_inches(parsed) if parsed is not None else None


def _pptx_table_cell_visual_metadata(cell: ET.Element) -> dict[str, Any]:
    properties = cell.find("a:tcPr", XML_NS)
    if properties is None:
        return {}

    metadata: dict[str, Any] = {
        "table_cell_text_anchor": _drawing_attr(properties, "anchor"),
        "table_cell_text_direction": _drawing_attr(properties, "vert"),
        "table_cell_margin_left_inches": _pptx_cell_margin_inches(_drawing_attr(properties, "marL")),
        "table_cell_margin_right_inches": _pptx_cell_margin_inches(_drawing_attr(properties, "marR")),
        "table_cell_margin_top_inches": _pptx_cell_margin_inches(_drawing_attr(properties, "marT")),
        "table_cell_margin_bottom_inches": _pptx_cell_margin_inches(_drawing_attr(properties, "marB")),
    }

    fill_node = properties.find("a:solidFill", XML_NS)
    fill_color = _drawing_color_value(fill_node)
    if fill_color:
        metadata["table_cell_fill_color"] = fill_color

    border_parts: list[str] = []
    for side, local_name in (
        ("left", "lnL"),
        ("right", "lnR"),
        ("top", "lnT"),
        ("bottom", "lnB"),
    ):
        line = properties.find(f"a:{local_name}", XML_NS)
        if line is None:
            continue
        width = _pptx_cell_margin_inches(_drawing_attr(line, "w"))
        color = _drawing_color_value(line)
        if color:
            metadata[f"table_cell_border_{side}_color"] = color
        if width is not None:
            metadata[f"table_cell_border_{side}_width_inches"] = width
        border_summary = " ".join(
            part for part in [
                side,
                color or "",
                f"{width:.3f}in" if width is not None else "",
            ]
            if part
        )
        if border_summary:
            border_parts.append(border_summary)

    summary_parts = []
    if fill_color:
        summary_parts.append(f"fill {fill_color}")
    anchor = metadata.get("table_cell_text_anchor")
    if anchor:
        summary_parts.append(f"text anchor {anchor}")
    margins = [
        metadata.get("table_cell_margin_left_inches"),
        metadata.get("table_cell_margin_right_inches"),
        metadata.get("table_cell_margin_top_inches"),
        metadata.get("table_cell_margin_bottom_inches"),
    ]
    if any(value is not None for value in margins):
        margin_text = ", ".join(
            f"{label}={value:.3f}in"
            for label, value in zip(("L", "R", "T", "B"), margins)
            if value is not None
        )
        summary_parts.append(f"margins {margin_text}")
    if border_parts:
        summary_parts.append("borders " + "; ".join(border_parts))
    if summary_parts:
        metadata["table_cell_visual_summary"] = "; ".join(summary_parts)

    return {key: value for key, value in metadata.items() if value not in (None, "", [], {})}


def _docx_hyperlink_elements(
    root: ET.Element,
    relationships: dict[str, dict[str, str]],
    *,
    location_prefix: str,
    base_metadata: dict[str, Any] | None = None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    metadata_base = base_metadata or {}
    for hyperlink_index, hyperlink in enumerate(root.findall(".//w:hyperlink", XML_NS), start=1):
        label = _xml_text(hyperlink, ("t",))
        relationship_id = hyperlink.attrib.get(f"{{{XML_NS['r']}}}id")
        anchor = _word_attr(hyperlink, "anchor")
        relationship = relationships.get(relationship_id or "", {})
        uri = relationship.get("target") or (f"#{anchor}" if anchor else "")
        element = _as_element(
            "link",
            f"{label} -> {uri}" if uri and label and uri != label else label or uri,
            location=f"{location_prefix} link {hyperlink_index}",
            **metadata_base,
            uri=uri,
            label=label,
            relationship_id=relationship_id,
            target_mode=relationship.get("target_mode"),
            anchor=anchor,
        )
        if element:
            elements.append(element)
    return elements


def _drawing_hyperlink_elements(
    root: ET.Element,
    relationships: dict[str, dict[str, str]],
    *,
    location_prefix: str,
    base_metadata: dict[str, Any] | None = None,
    label_context: str | None = None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    metadata_base = base_metadata or {}
    seen: set[tuple[str | None, str | None, str | None, str]] = set()
    hyperlink_index = 0

    for hyperlink in root.iter():
        local_name = _local_name(hyperlink.tag)
        if local_name not in {"hlinkClick", "hlinkHover"}:
            continue
        relationship_id = _attr_by_local_name(hyperlink, "id")
        relationship = relationships.get(relationship_id or "", {})
        action = _drawing_attr(hyperlink, "action")
        tooltip = _drawing_attr(hyperlink, "tooltip")
        target_frame = _drawing_attr(hyperlink, "tgtFrame")
        uri = relationship.get("target") or action or ""
        label = _clean_text(label_context or tooltip or uri)
        key = (relationship_id, uri, action, label)
        if key in seen:
            continue
        seen.add(key)
        hyperlink_index += 1
        element = _as_element(
            "link",
            f"{label} -> {uri}" if uri and label and uri != label else label or uri,
            location=f"{location_prefix} link {hyperlink_index}",
            **metadata_base,
            uri=uri,
            label=label,
            relationship_id=relationship_id,
            target_mode=relationship.get("target_mode"),
            link_kind="click" if local_name == "hlinkClick" else "hover",
            action=action,
            tooltip=tooltip,
            target_frame=target_frame,
        )
        if element:
            elements.append(element)
    return elements


def _pptx_comment_author_map(content: bytes) -> dict[str, dict[str, str]]:
    authors: dict[str, dict[str, str]] = {}
    author_parts = ["ppt/commentAuthors.xml", *_zip_names(content, "ppt/commentAuthors/commentAuthor")]
    for author_part in author_parts:
        root = _read_zip_xml(content, author_part)
        if root is None:
            continue
        for author in root.iter():
            if _local_name(author.tag) not in {"cmAuthor", "author", "commentAuthor"}:
                continue
            author_id = _attr_by_local_name(author, "id") or _attr_by_local_name(author, "authorId")
            if not author_id:
                continue
            authors[author_id] = {
                key: value
                for key, value in {
                    "author": _attr_by_local_name(author, "name") or _attr_by_local_name(author, "displayName"),
                    "initials": _attr_by_local_name(author, "initials"),
                    "author_part": author_part,
                }.items()
                if value
            }
    return authors


def _pptx_comment_elements(
    content: bytes,
    comment_part: str,
    *,
    author_map: dict[str, dict[str, str]] | None = None,
    slide_number: int | None = None,
    slide_title: str | None = None,
    slide_size_metadata: dict[str, Any] | None = None,
) -> list[DocumentElement]:
    root = _read_zip_xml(content, comment_part)
    if root is None:
        return []

    elements: list[DocumentElement] = []
    authors = author_map or {}
    comment_index = 0
    for comment in root.iter():
        if _local_name(comment.tag) not in {"cm", "comment", "threadedComment", "threadedCommentReply"}:
            continue
        text = _xml_text(comment, ("t",))
        if not text:
            continue
        comment_index += 1
        author_id = (
            _attr_by_local_name(comment, "authorId")
            or _attr_by_local_name(comment, "author")
            or _attr_by_local_name(comment, "userId")
        )
        author_metadata = authors.get(author_id or "", {})
        author = author_metadata.get("author") or _attr_by_local_name(comment, "authorName")
        comment_id = (
            _attr_by_local_name(comment, "id")
            or _attr_by_local_name(comment, "idx")
            or f"{Path(comment_part).stem}-{comment_index}"
        )
        position_metadata = _pptx_point_position_metadata(comment, slide_size_metadata)
        element = _as_element(
            "comment",
            f"{author}: {text}" if author else text,
            location=(
                f"slide {slide_number} comment {comment_index}"
                if slide_number
                else f"{comment_part} comment {comment_index}"
            ),
            slide=slide_number,
            slide_title=slide_title,
            comment_id=comment_id,
            author_id=author_id,
            author=author,
            initials=author_metadata.get("initials") or _attr_by_local_name(comment, "initials"),
            date=_attr_by_local_name(comment, "dt") or _attr_by_local_name(comment, "date"),
            comment_part=comment_part,
            author_part=author_metadata.get("author_part"),
            **position_metadata,
        )
        if element:
            elements.append(element)
    return elements


def _pptx_media_elements(
    content: bytes,
    relationships: dict[str, dict[str, str]],
    *,
    slide_number: int,
    slide_title: str | None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    embedded_index = 0
    for relationship_id, relationship in relationships.items():
        relationship_type = relationship.get("type", "")
        target = relationship.get("target") or ""
        target_mode = relationship.get("target_mode")
        rel_type = relationship_type.lower()
        target_path = target.lower()
        if not target:
            continue
        is_supported_media = (
            rel_type.endswith(("/audio", "/video", "/media", "/oleobject", "/package"))
            or "/embeddings/" in target_path
            or ("/media/" in target_path and not _is_image_part(target))
        )
        if not is_supported_media:
            continue

        is_external = str(target_mode or "").lower() == "external"
        metadata = {} if is_external else _zip_binary_metadata(content, target)
        if is_external:
            metadata.update(_binary_metadata_from_bytes(None, target))
        media_kind = _media_kind_from_relationship(relationship_type, target, metadata.get("media_mime"))
        element_type = "embedded_object" if media_kind in {"embedded_object", "embedded_package"} else "media"
        embedded_metadata: dict[str, Any] = {}
        if element_type == "embedded_object":
            embedded_index += 1
            embedded_filename = Path(target).name if target else ""
            embedded_metadata = {
                "embedded_file_index": embedded_index,
                "embedded_file_name": embedded_filename or target,
                "embedded_file_filename": embedded_filename or target,
                "embedded_file_mime": metadata.get("media_mime"),
                "embedded_file_size_bytes": metadata.get("media_size_bytes"),
                "embedded_file_sha256": metadata.get("media_sha256"),
            }
        element = _as_element(
            element_type,
            _media_element_text(
                media_kind=media_kind,
                media_part=None if is_external else target,
                uri=target if is_external else None,
                media_mime=metadata.get("media_mime"),
                size_bytes=metadata.get("media_size_bytes"),
            ),
            location=f"slide {slide_number} {media_kind.replace('_', ' ')}",
            slide=slide_number,
            slide_title=slide_title,
            relationship_id=relationship_id,
            relationship_type=relationship_type,
            target_mode=target_mode,
            media_kind=media_kind,
            media_part=None if is_external else target,
            uri=target if is_external else None,
            **metadata,
            **embedded_metadata,
        )
        if element:
            elements.append(element)

        if element_type == "embedded_object" and not is_external:
            child_bytes = _read_zip_bytes(content, target)
            if child_bytes:
                elements.extend(_child_elements_from_embedded_bytes(
                    child_bytes,
                    embedded_metadata.get("embedded_file_filename") or target,
                    location_prefix=f"slide {slide_number} embedded object {embedded_index}",
                    table_index=embedded_index,
                    base_metadata={
                        "slide": slide_number,
                        "slide_title": slide_title,
                        "relationship_id": relationship_id,
                        "relationship_type": relationship_type,
                        "target_mode": target_mode,
                        "media_kind": media_kind,
                        "media_part": target,
                        **embedded_metadata,
                    },
                    extraction_metadata_key="embedded_file_extraction_method",
                    max_bytes=MAX_EMBEDDED_FILE_PARSE_BYTES,
                ))
    return elements


def _xml_local_attrs(node: ET.Element | None) -> dict[str, str]:
    if node is None:
        return {}
    return {_snake_case(key): _clean_text(str(value)) for key, value in node.attrib.items() if value is not None}


def _pptx_control_properties(root: ET.Element | None) -> dict[str, str]:
    if root is None:
        return {}

    properties = _xml_local_attrs(root)
    for node in root.iter():
        local = _local_name(node.tag)
        attrs = _xml_local_attrs(node)
        if local in {"ocxPr", "property", "prop"}:
            name = attrs.get("name") or attrs.get("property_name") or attrs.get("id")
            value = attrs.get("value") or attrs.get("val") or attrs.get("text") or _xml_text(node, ("t", "v"))
            if name and value:
                properties[_snake_case(name)] = value
        elif local not in {"ocx"}:
            value = attrs.get("value") or attrs.get("val") or attrs.get("text") or _xml_text(node, ("t", "v"))
            if value:
                properties.setdefault(_snake_case(local), value)
    return {key: value for key, value in properties.items() if value}


def _pptx_control_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _clean_text(str(value or "")).casefold()
    if not text:
        return None
    if text in {"true", "yes", "on", "checked", "selected", "enabled", "1", "-1"}:
        return True
    if text in {"false", "no", "off", "unchecked", "unselected", "disabled", "0"}:
        return False
    return None


def _pptx_control_list_values(value: Any) -> list[str]:
    text = _clean_text(str(value or ""))
    if not text:
        return []
    delimiter = next((candidate for candidate in ("\r\n", "\n", ";", "|", ",") if candidate in text), None)
    if delimiter:
        return [_clean_text(part) for part in text.split(delimiter) if _clean_text(part)]
    return [text]


def _pptx_control_options(properties: dict[str, str]) -> tuple[list[str], list[str]]:
    option_text = next(
        (
            properties.get(key)
            for key in (
                "list",
                "items",
                "options",
                "entries",
                "list_items",
                "combo_box_items",
                "dropdown_items",
            )
            if properties.get(key)
        ),
        None,
    )
    value_text = next(
        (
            properties.get(key)
            for key in (
                "option_values",
                "values",
                "item_values",
                "list_values",
                "bound_values",
            )
            if properties.get(key)
        ),
        None,
    )
    options = _pptx_control_list_values(option_text)
    option_values = _pptx_control_list_values(value_text)
    return options, option_values


def _pptx_control_selected_option(
    properties: dict[str, str],
    value: Any,
    options: list[str],
) -> str | None:
    for key in ("selected_option", "selected_item", "selected_value", "text", "value"):
        selected = _clean_text(str(properties.get(key) or ""))
        if selected:
            return selected

    index_value = properties.get("list_index") or properties.get("selected_index") or properties.get("selection_index")
    if index_value not in {None, ""} and options:
        try:
            index = int(str(index_value))
        except (TypeError, ValueError):
            index = -1
        if 0 <= index < len(options):
            return options[index]
        if 1 <= index <= len(options):
            return options[index - 1]

    selected = _clean_text(str(value or ""))
    return selected or None


def _pptx_control_checked(properties: dict[str, str], value: Any, prog_id: str | None) -> bool | None:
    for key in ("checked", "check_state", "state", "selected", "value"):
        checked = _pptx_control_bool(properties.get(key))
        if checked is not None:
            return checked

    control_type = _clean_text(str(prog_id or "")).casefold()
    if any(kind in control_type for kind in ("checkbox", "check_box", "optionbutton", "radio", "togglebutton")):
        return _pptx_control_bool(value)
    return None


def _pptx_control_elements(
    content: bytes,
    root: ET.Element,
    relationships: dict[str, dict[str, str]],
    *,
    slide_number: int,
    slide_title: str | None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    control_index = 0

    for control in root.iter():
        if _local_name(control.tag) != "control":
            continue
        control_index += 1
        relationship_id = _attr_by_local_name(control, "id")
        relationship = relationships.get(relationship_id or "", {})
        control_part = relationship.get("target") or ""
        control_root = _read_zip_xml(content, control_part) if control_part else None
        properties = _pptx_control_properties(control_root)
        control_name = _attr_by_local_name(control, "name") or properties.get("name")
        prog_id = properties.get("prog_id") or properties.get("progId") or properties.get("class_id")
        caption = properties.get("caption") or properties.get("label") or properties.get("title")
        value = (
            properties.get("value")
            or properties.get("text")
            or properties.get("checked")
            or properties.get("state")
        )
        options, option_values = _pptx_control_options(properties)
        selected_option = _pptx_control_selected_option(properties, value, options) if options else None
        checked = _pptx_control_checked(properties, value, prog_id)
        label = _clean_text(str(caption or control_name or prog_id or control_part or f"Control {control_index}"))
        control_shape_id = _attr_by_local_name(control, "spid")
        control_id = properties.get("id") or properties.get("control_id") or control_shape_id or relationship_id or control_name
        text = f"{label}: {value}" if value not in {None, ""} else label
        element = _as_element(
            "form_field",
            text,
            location=f"slide {slide_number} control {control_index}",
            slide=slide_number,
            slide_title=slide_title,
            label=label,
            value=value,
            field_type=prog_id,
            form_kind="pptx_control",
            control_name=control_name,
            control_part=control_part,
            control_id=control_id,
            control_properties=properties,
            control_shape_id=control_shape_id,
            control_show_as_icon=_attr_by_local_name(control, "showAsIcon"),
            form_field_alias=label,
            form_field_tag=control_name,
            form_field_id=control_id,
            form_field_checked=checked,
            form_field_options=options or None,
            form_field_option_values=option_values or None,
            form_field_selected_option=selected_option,
            relationship_id=relationship_id,
            relationship_type=relationship.get("type"),
            target_mode=relationship.get("target_mode"),
        )
        if element:
            elements.append(element)

    return elements


def _pptx_inherited_part_metadata(
    *,
    part_kind: str,
    part_name: str,
    relationship_id: str | None = None,
    owner_part: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "inherited_from": part_kind,
        "inherited_relationship_id": relationship_id,
    }
    if part_kind == "slide_layout":
        metadata["layout_part"] = part_name
        metadata["layout_relationship_id"] = relationship_id
    elif part_kind == "slide_master":
        metadata["master_part"] = part_name
        metadata["master_relationship_id"] = relationship_id
        metadata["layout_part"] = owner_part
    else:
        metadata["inherited_part"] = part_name
    return {key: value for key, value in metadata.items() if value not in (None, "")}


def _pptx_inherited_part_elements(
    content: bytes,
    part_name: str,
    *,
    part_kind: str,
    slide_number: int,
    slide_title: str | None,
    relationship_id: str | None = None,
    owner_part: str | None = None,
    slide_size_metadata: dict[str, Any] | None = None,
) -> list[DocumentElement]:
    root = _read_zip_xml(content, part_name)
    if root is None:
        return []

    elements: list[DocumentElement] = []
    part_relationships = _part_relationships(content, part_name)
    part_label = "layout" if part_kind == "slide_layout" else "master" if part_kind == "slide_master" else "inherited part"
    inherited_metadata = _pptx_inherited_part_metadata(
        part_kind=part_kind,
        part_name=part_name,
        relationship_id=relationship_id,
        owner_part=owner_part,
    )
    seen_links: set[tuple[Any, ...]] = set()

    for shape_index, shape in enumerate(root.findall(".//p:sp", XML_NS), start=1):
        text = _xml_text(shape, ("t",))
        placeholder = shape.find(".//p:ph", XML_NS)
        placeholder_type = placeholder.attrib.get("type") if placeholder is not None else None
        shape_metadata = _pptx_shape_metadata(shape)
        shape_id = shape_metadata.get("shape_id")
        shape_name = shape_metadata.get("shape_name")
        shape_title = shape_metadata.get("shape_title")
        shape_description = shape_metadata.get("shape_description")
        shape_metadata_text = _pptx_shape_metadata_text(shape_metadata)
        shape_geometry = _pptx_geometry_metadata(shape, slide_size_metadata)
        base_metadata = {
            **inherited_metadata,
            **shape_geometry,
            "slide": slide_number,
            "slide_title": slide_title,
            "placeholder": placeholder_type,
            "shape_index": shape_index,
            "shape_id": shape_id,
            "shape_name": shape_name,
            "shape_title": shape_title,
            "shape_description": shape_description,
        }

        if shape_metadata_text and (shape_title or shape_description or not text):
            element = _as_element(
                "shape_metadata",
                shape_metadata_text,
                location=f"slide {slide_number} {part_label} shape {shape_index}",
                **base_metadata,
            )
            if element:
                elements.append(element)

        if text:
            runs = _drawing_run_styles(shape)
            emphasis_terms = [run["text"] for run in runs if run.get("bold") or run.get("italic") or run.get("underline")]
            bullet_levels = _paragraph_bullet_levels(shape) or None
            element = _as_element(
                "slide_text",
                text,
                location=f"slide {slide_number} {part_label} text {shape_index}",
                **base_metadata,
                bullet_levels=bullet_levels,
                runs=runs[:24] if runs else None,
                emphasis_terms=emphasis_terms[:12] if emphasis_terms else None,
                style_summary=_style_summary_from_runs(runs, bullet_levels),
            )
            if element:
                elements.append(element)

        for link in _drawing_hyperlink_elements(
            shape,
            part_relationships,
            location_prefix=f"slide {slide_number} {part_label} shape {shape_index}",
            base_metadata=base_metadata,
            label_context=text or shape_metadata_text or shape_title or shape_description or shape_name,
        ):
            link_key = (
                link.metadata.get("relationship_id"),
                link.metadata.get("uri"),
                link.metadata.get("link_kind"),
                link.text,
            )
            if link_key in seen_links:
                continue
            seen_links.add(link_key)
            elements.append(link)

    for picture_index, picture in enumerate(root.findall(".//p:pic", XML_NS), start=1):
        picture_properties = picture.find(".//p:cNvPr", XML_NS)
        title = _drawing_attr(picture_properties, "title")
        description = _drawing_attr(picture_properties, "descr")
        name = _drawing_attr(picture_properties, "name")
        blip = picture.find(".//a:blip", XML_NS)
        picture_relationship_id = None
        if blip is not None:
            picture_relationship_id = (
                blip.attrib.get(f"{{{XML_NS['r']}}}embed")
                or blip.attrib.get(f"{{{XML_NS['r']}}}link")
            )
        relationship = part_relationships.get(picture_relationship_id or "", {})
        image_part = relationship.get("target")
        image_metadata = _zip_image_metadata(content, image_part)
        label = _image_element_text(title, description, name, image_part, metadata=image_metadata)
        picture_geometry = _pptx_geometry_metadata(picture, slide_size_metadata)
        element = _as_element(
            "image",
            label,
            location=f"slide {slide_number} {part_label} image {picture_index}",
            **inherited_metadata,
            **picture_geometry,
            slide=slide_number,
            slide_title=slide_title,
            image=picture_index,
            title=title,
            description=description,
            name=name,
            relationship_id=picture_relationship_id,
            image_relationship_id=picture_relationship_id,
            target_mode=relationship.get("target_mode"),
            image_part=image_part,
            ocr_source="pptx_inherited_image" if image_metadata.get("ocr_text") else None,
            **image_metadata,
        )
        if element:
            elements.append(element)
        ocr_element = _image_ocr_text_element(
            image_metadata,
            location=f"slide {slide_number} {part_label} image {picture_index} OCR",
            image_part=image_part,
            ocr_source="pptx_inherited_image",
            base_metadata={
                **inherited_metadata,
                **picture_geometry,
                "slide": slide_number,
                "slide_title": slide_title,
                "image": picture_index,
                "title": title,
                "description": description,
                "name": name,
                "relationship_id": picture_relationship_id,
                "target_mode": relationship.get("target_mode"),
            },
        )
        if ocr_element:
            elements.append(ocr_element)

        for link in _drawing_hyperlink_elements(
            picture,
            part_relationships,
            location_prefix=f"slide {slide_number} {part_label} image {picture_index}",
            base_metadata={
                **inherited_metadata,
                **picture_geometry,
                "slide": slide_number,
                "slide_title": slide_title,
                "image": picture_index,
                "title": title,
                "description": description,
                "name": name,
                **_image_link_metadata(
                    image_metadata,
                    image_part=image_part,
                    image_relationship_id=picture_relationship_id,
                    image_relationship=relationship,
                ),
            },
            label_context=title or description or name or image_part,
        ):
            link_key = (
                link.metadata.get("relationship_id"),
                link.metadata.get("uri"),
                link.metadata.get("link_kind"),
                link.text,
            )
            if link_key in seen_links:
                continue
            seen_links.add(link_key)
            elements.append(link)

    return elements


def _pptx_slide_inherited_elements(
    content: bytes,
    slide_relationships: dict[str, dict[str, str]],
    *,
    slide_number: int,
    slide_title: str | None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    seen_parts: set[tuple[str, str]] = set()
    slide_size_metadata = _pptx_slide_size_metadata(content)
    layout_relationships = [
        relationship
        for relationship in slide_relationships.values()
        if relationship.get("target") and relationship.get("type", "").endswith("/slideLayout")
    ]

    for layout_relationship in _unique_relationships(layout_relationships):
        layout_part = layout_relationship.get("target") or ""
        if not layout_part or ("slide_layout", layout_part) in seen_parts:
            continue
        seen_parts.add(("slide_layout", layout_part))
        elements.extend(_pptx_inherited_part_elements(
            content,
            layout_part,
            part_kind="slide_layout",
            slide_number=slide_number,
            slide_title=slide_title,
            relationship_id=layout_relationship.get("id"),
            slide_size_metadata=slide_size_metadata,
        ))

        layout_relationships_by_id = _part_relationships(content, layout_part)
        master_relationships = [
            relationship
            for relationship in layout_relationships_by_id.values()
            if relationship.get("target") and relationship.get("type", "").endswith("/slideMaster")
        ]
        for master_relationship in _unique_relationships(master_relationships):
            master_part = master_relationship.get("target") or ""
            if not master_part or ("slide_master", master_part) in seen_parts:
                continue
            seen_parts.add(("slide_master", master_part))
            elements.extend(_pptx_inherited_part_elements(
                content,
                master_part,
                part_kind="slide_master",
                slide_number=slide_number,
                slide_title=slide_title,
                relationship_id=master_relationship.get("id"),
                owner_part=layout_part,
                slide_size_metadata=slide_size_metadata,
            ))

    return elements


def _unique_relationships(relationships: list[dict[str, str]]) -> list[dict[str, str]]:
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for relationship in relationships:
        key = (
            str(relationship.get("id") or ""),
            str(relationship.get("type") or ""),
            str(relationship.get("target") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(relationship)
    return unique


def _diagram_connection_relation(kind: str | None) -> str:
    normalized = _relation_label(kind or "related_to")
    return {
        "parof": "contains",
        "presof": "presents",
        "presparof": "presents",
        "unknownrelationship": "related_to",
    }.get(normalized.replace("_", ""), normalized)


def _drawingml_diagram_elements(
    content: bytes,
    diagram_part: str,
    *,
    slide_number: int | None,
    slide_title: str | None,
    relationship_id: str | None = None,
    location_prefix: str | None = None,
    base_metadata: dict[str, Any] | None = None,
) -> list[DocumentElement]:
    root = _read_zip_xml(content, diagram_part)
    if root is None:
        return []

    point_text: dict[str, str] = {}
    point_types: dict[str, str | None] = {}
    point_order: list[str] = []
    for point in root.iter():
        if _local_name(point.tag) != "pt":
            continue
        point_id = _attr_by_local_name(point, "modelId") or _attr_by_local_name(point, "id")
        text = _xml_text(point, ("t",))
        if not point_id or not text:
            continue
        point_text[point_id] = text
        point_types[point_id] = _attr_by_local_name(point, "type")
        point_order.append(point_id)

    parent_map: dict[str, str] = {}
    connections: list[dict[str, str | None]] = []
    for connection in root.iter():
        if _local_name(connection.tag) != "cxn":
            continue
        source_id = _attr_by_local_name(connection, "srcId") or _attr_by_local_name(connection, "sourceId")
        target_id = _attr_by_local_name(connection, "destId") or _attr_by_local_name(connection, "targetId")
        kind = _attr_by_local_name(connection, "type") or _attr_by_local_name(connection, "relationshipType")
        if not source_id or not target_id:
            continue
        connections.append({"source_id": source_id, "target_id": target_id, "kind": kind})
        if str(kind or "").lower() in {"parof", "presparof"} and target_id not in parent_map:
            parent_map[target_id] = source_id

    level_cache: dict[str, int] = {}

    def level_for(point_id: str, seen: set[str] | None = None) -> int:
        if point_id in level_cache:
            return level_cache[point_id]
        seen = seen or set()
        parent_id = parent_map.get(point_id)
        if not parent_id or parent_id in seen:
            level_cache[point_id] = 0
            return 0
        level_cache[point_id] = 1 + level_for(parent_id, {*seen, point_id})
        return level_cache[point_id]

    elements: list[DocumentElement] = []
    resolved_location_prefix = location_prefix or (f"slide {slide_number}" if slide_number else "package")
    common_metadata = {
        **(base_metadata or {}),
        "slide": slide_number,
        "slide_title": slide_title,
        "diagram_part": diagram_part,
        "relationship_id": relationship_id,
    }
    for node_index, point_id in enumerate(point_order, start=1):
        text = point_text.get(point_id, "")
        parent_id = parent_map.get(point_id)
        parent_text = point_text.get(parent_id or "")
        element = _as_element(
            "diagram_node",
            text,
            location=f"{resolved_location_prefix} diagram {Path(diagram_part).stem} node {node_index}",
            **common_metadata,
            diagram_node_id=point_id,
            diagram_node_type=point_types.get(point_id),
            diagram_order=node_index,
            diagram_level=level_for(point_id),
            parent_diagram_node_id=parent_id,
            parent_label=parent_text,
        )
        if element:
            elements.append(element)

    for connection_index, connection in enumerate(connections, start=1):
        source_label = point_text.get(str(connection.get("source_id") or ""))
        target_label = point_text.get(str(connection.get("target_id") or ""))
        if not source_label or not target_label:
            continue
        relation = _diagram_connection_relation(connection.get("kind"))
        element = _as_element(
            "diagram_relationship",
            f"{source_label} --[{relation}]--> {target_label}",
            location=f"{resolved_location_prefix} diagram {Path(diagram_part).stem} connection {connection_index}",
            **common_metadata,
            source_label=source_label,
            target_label=target_label,
            relation=relation,
            diagram_connection_type=connection.get("kind"),
            source_diagram_node_id=connection.get("source_id"),
            target_diagram_node_id=connection.get("target_id"),
        )
        if element:
            elements.append(element)

    return elements


def _docx_part_text_elements(
    content: bytes,
    part_names: list[str],
    *,
    element_type: str,
    location_label: str,
    base_metadata: dict[str, Any] | None = None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    metadata_base = base_metadata or {}
    for part_index, part_name in enumerate(part_names, start=1):
        root = _read_zip_xml(content, part_name)
        if root is None:
            continue
        for paragraph_index, paragraph in enumerate(root.findall(".//w:p", XML_NS), start=1):
            text = _xml_text(paragraph, ("t",))
            element = _as_element(
                element_type,
                text,
                location=f"{location_label} {part_index} paragraph {paragraph_index}",
                **metadata_base,
                part_name=part_name,
                part_index=part_index,
            )
            if element:
                elements.append(element)
        elements.extend(_docx_hyperlink_elements(
            root,
            _part_relationships(content, part_name),
            location_prefix=f"{location_label} {part_index}",
            base_metadata={**metadata_base, "part_name": part_name, "part_index": part_index},
        ))
    return elements


def _docx_comment_anchor_texts(paragraph: ET.Element) -> dict[str, str]:
    active_ranges: set[str] = set()
    referenced_ids: set[str] = set()
    range_texts: dict[str, list[str]] = defaultdict(list)
    paragraph_text = _xml_text(paragraph, ("t", "delText"))

    for node in paragraph.iter():
        local_name = _local_name(node.tag)
        if local_name in {"commentRangeStart", "commentRangeEnd", "commentReference"}:
            comment_id = _word_attr(node, "id") or _attr_by_local_name(node, "id")
            if not comment_id:
                continue
            referenced_ids.add(comment_id)
            if local_name == "commentRangeStart":
                active_ranges.add(comment_id)
            elif local_name == "commentRangeEnd":
                active_ranges.discard(comment_id)
            continue
        if local_name in {"t", "delText"} and node.text:
            for comment_id in active_ranges:
                range_texts[comment_id].append(node.text)

    return {
        comment_id: _clean_text(" ".join(range_texts.get(comment_id) or [paragraph_text]))
        for comment_id in referenced_ids
        if _clean_text(" ".join(range_texts.get(comment_id) or [paragraph_text]))
    }


def _docx_comment_anchor_map(root: ET.Element | None) -> dict[str, dict[str, Any]]:
    if root is None:
        return {}

    anchors: dict[str, dict[str, Any]] = {}
    current_heading = ""
    for paragraph_index, paragraph in enumerate(root.findall(".//w:p", XML_NS), start=1):
        style_node = paragraph.find(".//w:pStyle", XML_NS)
        style = style_node.attrib.get(f"{{{XML_NS['w']}}}val") if style_node is not None else ""
        text = _xml_text(paragraph, ("t", "delText"))
        if style and style.lower().startswith("heading") and text:
            current_heading = text

        for comment_id, anchor_text in _docx_comment_anchor_texts(paragraph).items():
            existing = anchors.get(comment_id)
            if existing:
                combined = _unique_preserve_order([
                    str(existing.get("comment_anchor_text") or ""),
                    anchor_text,
                ])
                existing["comment_anchor_text"] = " | ".join(part for part in combined if part)
                continue
            anchors[comment_id] = {
                "comment_anchor_text": anchor_text,
                "comment_anchor_section": current_heading,
                "comment_anchor_paragraph": paragraph_index,
            }
    return anchors


def _docx_comment_elements(content: bytes, document_root: ET.Element | None = None) -> list[DocumentElement]:
    root = _read_zip_xml(content, "word/comments.xml")
    if root is None:
        return []
    elements: list[DocumentElement] = []
    anchor_map = _docx_comment_anchor_map(document_root)
    for comment_index, comment in enumerate(root.findall(".//w:comment", XML_NS), start=1):
        comment_id = _word_attr(comment, "id")
        author = _word_attr(comment, "author")
        initials = _word_attr(comment, "initials")
        date = _word_attr(comment, "date")
        text = _xml_text(comment, ("t",))
        anchor_metadata = anchor_map.get(comment_id or "", {})
        element = _as_element(
            "comment",
            f"{author}: {text}" if author else text,
            location=f"comment {comment_id or comment_index}",
            comment_id=comment_id,
            author=author,
            initials=initials,
            date=date,
            comment_part="word/comments.xml",
            **anchor_metadata,
        )
        if element:
            elements.append(element)
    return elements


def _docx_note_anchor_map(root: ET.Element | None, reference_name: str) -> dict[str, dict[str, Any]]:
    if root is None:
        return {}

    anchors: dict[str, dict[str, Any]] = {}
    current_heading = ""
    for paragraph_index, paragraph in enumerate(root.findall(".//w:p", XML_NS), start=1):
        style_node = paragraph.find(".//w:pStyle", XML_NS)
        style = style_node.attrib.get(f"{{{XML_NS['w']}}}val") if style_node is not None else ""
        text = _xml_text(paragraph, ("t", "delText"))
        if style and style.lower().startswith("heading") and text:
            current_heading = text

        if not text:
            continue
        for reference in paragraph.iter():
            if _local_name(reference.tag) != reference_name:
                continue
            note_id = _word_attr(reference, "id") or _attr_by_local_name(reference, "id")
            if not note_id:
                continue
            existing = anchors.get(note_id)
            if existing:
                combined = _unique_preserve_order([
                    str(existing.get("note_anchor_text") or ""),
                    text,
                ])
                existing["note_anchor_text"] = " | ".join(part for part in combined if part)
                continue
            anchors[note_id] = {
                "note_anchor_text": text,
                "note_anchor_section": current_heading,
                "note_anchor_paragraph": paragraph_index,
            }
    return anchors


def _docx_note_elements(
    content: bytes,
    part_name: str,
    element_type: str,
    document_root: ET.Element | None = None,
) -> list[DocumentElement]:
    root = _read_zip_xml(content, part_name)
    if root is None:
        return []
    elements: list[DocumentElement] = []
    note_tag = "footnote" if element_type == "footnote" else "endnote"
    reference_name = "footnoteReference" if element_type == "footnote" else "endnoteReference"
    anchor_map = _docx_note_anchor_map(document_root, reference_name)
    for note_index, note in enumerate(root.findall(f".//w:{note_tag}", XML_NS), start=1):
        note_id = _word_attr(note, "id")
        note_kind = _word_attr(note, "type")
        if note_kind in {"separator", "continuationSeparator"}:
            continue
        text = _xml_text(note, ("t",))
        anchor_metadata = anchor_map.get(note_id or "", {})
        element = _as_element(
            element_type,
            text,
            location=f"{element_type} {note_id or note_index}",
            note_id=note_id,
            note_type=note_kind,
            note_part=part_name,
            **anchor_metadata,
        )
        if element:
            elements.append(element)
    return elements


DOCX_REVISION_TYPES = {
    "ins": "inserted",
    "del": "deleted",
    "moveFrom": "moved_from",
    "moveTo": "moved_to",
}


def _docx_revision_elements(
    node: ET.Element,
    *,
    location_prefix: str,
    base_metadata: dict[str, Any] | None = None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    metadata_base = base_metadata or {}
    revision_index = 0
    for revision in node.iter():
        local_name = _local_name(revision.tag)
        revision_type = DOCX_REVISION_TYPES.get(local_name)
        if not revision_type:
            continue
        text = _xml_text(revision, ("t", "delText"))
        if not text:
            continue
        revision_index += 1
        author = _word_attr(revision, "author")
        date = _word_attr(revision, "date")
        revision_id = _word_attr(revision, "id")
        label = revision_type.replace("_", " ").title()
        element = _as_element(
            "revision",
            f"{label}: {text}",
            location=f"{location_prefix} revision {revision_id or revision_index}",
            **metadata_base,
            revision_type=revision_type,
            revision_author=author,
            revision_date=date,
            revision_id=revision_id,
        )
        if element:
            elements.append(element)
    return elements


def _docx_text_box_metadata(container: ET.Element) -> dict[str, Any]:
    properties = _first_descendant_by_local_name(container, "docPr", "cNvPr")
    shape = _first_descendant_by_local_name(container, "shape", "wsp")
    title = (
        _drawing_attr(properties, "title")
        or _attr_by_local_name(shape, "title")
        or _attr_by_local_name(shape, "alt")
    )
    description = (
        _drawing_attr(properties, "descr")
        or _attr_by_local_name(shape, "descr")
        or _attr_by_local_name(shape, "alt")
    )
    name = (
        _drawing_attr(properties, "name")
        or _attr_by_local_name(shape, "name")
        or _attr_by_local_name(shape, "id")
    )
    shape_id = (
        _drawing_attr(properties, "id")
        or _attr_by_local_name(shape, "id")
    )
    style = _attr_by_local_name(shape, "style")
    return {
        "shape_id": shape_id,
        "shape_name": name,
        "shape_title": title,
        "shape_description": description,
        "text_box_style": style,
    }


def _docx_text_box_section_map(root: ET.Element) -> dict[int, str]:
    body = root.find(".//w:body", XML_NS)
    if body is None:
        return {}

    sections: dict[int, str] = {}
    current_heading = ""
    for child in list(body):
        if _local_name(child.tag) == "p":
            style_node = child.find(".//w:pStyle", XML_NS)
            style = style_node.attrib.get(f"{{{XML_NS['w']}}}val") if style_node is not None else ""
            text = _xml_text(child, ("t",))
            if style and style.lower().startswith("heading") and text:
                current_heading = text
        if not current_heading:
            continue
        for node in child.iter():
            if _local_name(node.tag) == "txbxContent":
                sections[id(node)] = current_heading
    return sections


def _docx_text_box_elements(
    root: ET.Element,
    *,
    part_name: str,
    location_label: str = "text box",
    base_metadata: dict[str, Any] | None = None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    metadata_base = base_metadata or {}
    seen_text_boxes: set[int] = set()
    text_box_index = 0
    section_by_text_box_id = _docx_text_box_section_map(root)

    for container in root.iter():
        container_kind = _local_name(container.tag)
        if container_kind not in {"drawing", "pict"}:
            continue
        text_box_nodes = [
            node
            for node in container.iter()
            if _local_name(node.tag) == "txbxContent"
        ]
        if not text_box_nodes:
            continue

        shape_metadata = _docx_text_box_metadata(container)
        for text_box_node in text_box_nodes:
            node_id = id(text_box_node)
            if node_id in seen_text_boxes:
                continue
            seen_text_boxes.add(node_id)
            text = _xml_text(text_box_node, ("t", "delText"))
            if not text:
                continue
            text_box_index += 1
            runs = _word_run_styles(text_box_node)
            emphasis_terms = [
                run["text"]
                for run in runs
                if run.get("bold") or run.get("italic") or run.get("underline")
            ]
            element = _as_element(
                "text_box",
                text,
                location=f"{location_label} {text_box_index}",
                **metadata_base,
                **shape_metadata,
                text_box_index=text_box_index,
                text_box_part=part_name,
                text_box_kind="vml_text_box" if container_kind == "pict" else "drawing_text_box",
                section=section_by_text_box_id.get(node_id),
                runs=runs[:24] if runs else None,
                emphasis_terms=emphasis_terms[:12] if emphasis_terms else None,
                style_summary=_style_summary_from_runs(runs),
            )
            if element:
                elements.append(element)

    return elements


def _paragraph_bullet_levels(node: ET.Element) -> list[int]:
    levels = []
    for paragraph in node.findall(".//a:p", XML_NS):
        paragraph_properties = paragraph.find("a:pPr", XML_NS)
        level = _drawing_attr(paragraph_properties, "lvl")
        if level is not None:
            try:
                levels.append(int(level))
            except ValueError:
                pass
    return levels


def _chart_type(root: ET.Element) -> str | None:
    plot_area = root.find(".//c:plotArea", XML_NS)
    if plot_area is None:
        return None
    for child in plot_area:
        local_name = child.tag.rsplit("}", 1)[-1]
        if local_name.endswith("Chart"):
            return local_name
    return None


def _chart_child_val(root: ET.Element | None, local_name: str) -> str | None:
    if root is None:
        return None
    child = root.find(f"c:{local_name}", XML_NS)
    if child is None:
        return None
    return _attr_by_local_name(child, "val")


def _chart_axis_type(axis_node: ET.Element) -> str:
    local_name = axis_node.tag.rsplit("}", 1)[-1]
    return {
        "catAx": "category",
        "valAx": "value",
        "dateAx": "date",
        "serAx": "series",
    }.get(local_name, local_name)


def _chart_flag_enabled(root: ET.Element | None, local_name: str) -> bool:
    value = _chart_child_val(root, local_name)
    return value is not None and str(value).lower() not in {"0", "false", "off"}


def _chart_label_field_name(local_name: str) -> str:
    return {
        "showLegendKey": "legend key",
        "showVal": "value",
        "showCatName": "category name",
        "showSerName": "series name",
        "showPercent": "percent",
        "showBubbleSize": "bubble size",
        "showLeaderLines": "leader lines",
    }.get(local_name, local_name)


def _chart_shape_visual_metadata(node: ET.Element | None, prefix: str) -> dict[str, Any]:
    if node is None:
        return {}
    shape_properties = node.find("c:spPr", XML_NS)
    if shape_properties is None:
        shape_properties = node.find("a:spPr", XML_NS)
    if shape_properties is None:
        return {}

    metadata: dict[str, Any] = {}
    fill = _drawing_color_value(shape_properties.find("a:solidFill", XML_NS))
    if fill:
        metadata[f"{prefix}_fill_color"] = fill

    line = shape_properties.find("a:ln", XML_NS)
    if line is not None:
        line_fill = line.find("a:solidFill", XML_NS)
        line_color = _drawing_color_value(line_fill if line_fill is not None else line)
        line_width = _pptx_points(_first_int(_drawing_attr(line, "w") or ""))
        dash_node = line.find("a:prstDash", XML_NS)
        line_dash = _drawing_attr(dash_node, "val") if dash_node is not None else None
        if line_color:
            metadata[f"{prefix}_line_color"] = line_color
        if line_width is not None:
            metadata[f"{prefix}_line_width_points"] = line_width
        if line_dash:
            metadata[f"{prefix}_line_dash"] = line_dash

    return metadata


def _chart_visual_summary(metadata: dict[str, Any], prefix: str) -> str | None:
    parts: list[str] = []
    fill = metadata.get(f"{prefix}_fill_color")
    line = metadata.get(f"{prefix}_line_color")
    line_width = metadata.get(f"{prefix}_line_width_points")
    line_dash = metadata.get(f"{prefix}_line_dash")
    if fill:
        parts.append(f"fill {fill}")
    if line:
        line_part = f"line {line}"
        if line_width is not None:
            line_part += f" {line_width:.2f}pt"
        if line_dash:
            line_part += f" {line_dash}"
        parts.append(line_part)
    return "; ".join(parts) if parts else None


def _chart_series_visual_metadata(series: ET.Element) -> dict[str, Any]:
    metadata = _chart_shape_visual_metadata(series, "chart_series")

    marker = series.find("c:marker", XML_NS)
    if marker is not None:
        marker_symbol = _chart_child_val(marker, "symbol")
        marker_size = _first_int(_chart_child_val(marker, "size") or "")
        marker_visual = _chart_shape_visual_metadata(marker, "chart_marker")
        if marker_symbol:
            metadata["chart_marker_symbol"] = marker_symbol
        if marker_size is not None:
            metadata["chart_marker_size"] = marker_size
        metadata.update(marker_visual)

    summary_parts: list[str] = []
    series_summary = _chart_visual_summary(metadata, "chart_series")
    if series_summary:
        summary_parts.append(series_summary)
    marker_parts = []
    if metadata.get("chart_marker_symbol"):
        marker_parts.append(str(metadata["chart_marker_symbol"]))
    if metadata.get("chart_marker_size") is not None:
        marker_parts.append(f"size {metadata['chart_marker_size']}")
    marker_visual_summary = _chart_visual_summary(metadata, "chart_marker")
    if marker_visual_summary:
        marker_parts.append(marker_visual_summary)
    if marker_parts:
        summary_parts.append("marker " + " ".join(marker_parts))
    if summary_parts:
        metadata["chart_series_visual_summary"] = "; ".join(summary_parts)
    return metadata


def _chart_point_visual_metadata(series: ET.Element) -> dict[int, dict[str, Any]]:
    point_visuals: dict[int, dict[str, Any]] = {}
    for point in series.findall("c:dPt", XML_NS):
        index = _first_int(_chart_child_val(point, "idx") or "")
        if index is None:
            continue
        metadata = _chart_shape_visual_metadata(point, "chart_point")
        summary = _chart_visual_summary(metadata, "chart_point")
        if summary:
            metadata["chart_point_visual_summary"] = summary
        if metadata:
            point_visuals[index + 1] = metadata
    return point_visuals


def _chart_data_label_metadata(data_labels: ET.Element | None) -> dict[str, Any]:
    if data_labels is None:
        return {}
    number_format_node = data_labels.find("c:numFmt", XML_NS)
    label_position = _chart_child_val(data_labels, "dLblPos")
    metadata = {
        "chart_label_position": label_position,
        "chart_label_number_format": _attr_by_local_name(number_format_node, "formatCode") if number_format_node is not None else None,
    }
    return {key: value for key, value in metadata.items() if value not in (None, "")}


def _chart_trendline_metadata(trendline: ET.Element) -> dict[str, Any]:
    name_node = trendline.find("c:name", XML_NS)
    metadata: dict[str, Any] = {
        "chart_trendline_type": _chart_child_val(trendline, "trendlineType"),
        "chart_trendline_name": _attr_by_local_name(name_node, "val") if name_node is not None else None,
        "chart_trendline_display_equation": _chart_flag_enabled(trendline, "dispEq"),
        "chart_trendline_display_r_squared": _chart_flag_enabled(trendline, "dispRSqr"),
    }
    metadata.update(_chart_shape_visual_metadata(trendline, "chart_trendline"))
    summary = _chart_visual_summary(metadata, "chart_trendline")
    if summary:
        metadata["chart_trendline_visual_summary"] = summary
    return {key: value for key, value in metadata.items() if value not in (None, "", [], {})}


def _chart_source_relationship_candidates(
    content: bytes,
    chart_part: str | None,
    root: ET.Element | None = None,
) -> list[dict[str, str]]:
    if not chart_part:
        return []

    relationships = _part_relationships(content, chart_part)
    if not relationships:
        return []

    external_data_ids = [
        _attr_by_local_name(node, "id")
        for node in (root.findall(".//c:externalData", XML_NS) if root is not None else [])
    ]
    candidates = [
        relationships[relationship_id]
        for relationship_id in external_data_ids
        if relationship_id and relationship_id in relationships
    ]
    if not candidates:
        candidates = [
            relationship
            for relationship in relationships.values()
            if (
                relationship.get("type", "").lower().endswith(("/package", "/externalLink", "/oleObject"))
                or "/embeddings/" in relationship.get("target", "").lower()
            )
        ]
    return candidates


def _chart_source_relationship_metadata(content: bytes, relationship: dict[str, str]) -> dict[str, Any]:
    target = relationship.get("target") or ""
    is_external = str(relationship.get("target_mode") or "").lower() == "external"
    binary_metadata = {} if is_external else _zip_binary_metadata(content, target)
    if is_external:
        binary_metadata.update(_binary_metadata_from_bytes(None, target))
    source_kind = _media_kind_from_relationship(
        relationship.get("type", ""),
        target,
        binary_metadata.get("media_mime"),
    )
    return {
        "chart_source_part": None if is_external else target,
        "chart_source_uri": target if is_external else None,
        "chart_source_relationship_id": relationship.get("id"),
        "chart_source_relationship_type": relationship.get("type"),
        "chart_source_target_mode": relationship.get("target_mode"),
        "chart_source_kind": source_kind,
        "chart_source_mime": binary_metadata.get("media_mime"),
        "chart_source_size_bytes": binary_metadata.get("media_size_bytes"),
        "chart_source_sha256": binary_metadata.get("media_sha256"),
    }


def _chart_source_metadata(content: bytes, chart_part: str | None, root: ET.Element | None = None) -> dict[str, Any]:
    candidates = _chart_source_relationship_candidates(content, chart_part, root)
    if not candidates:
        return {}
    return _chart_source_relationship_metadata(content, candidates[0])


def _chart_source_child_elements(
    content: bytes,
    chart_part: str | None,
    root: ET.Element | None,
    *,
    location_prefix: str,
    base_metadata: dict[str, Any] | None = None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    metadata_base = base_metadata or {}
    for source_index, relationship in enumerate(
        _chart_source_relationship_candidates(content, chart_part, root)[:3],
        start=1,
    ):
        target = relationship.get("target") or ""
        if not target or str(relationship.get("target_mode") or "").lower() == "external":
            continue
        source_bytes = _read_zip_bytes(content, target)
        if not source_bytes:
            continue
        source_metadata = _chart_source_relationship_metadata(content, relationship)
        elements.extend(_child_elements_from_embedded_bytes(
            source_bytes,
            Path(target).name or target,
            location_prefix=f"{location_prefix} source {source_index}",
            table_index=source_index,
            base_metadata={
                **metadata_base,
                **source_metadata,
                "chart_source_index": source_index,
            },
            extraction_metadata_key="chart_source_extraction_method",
            max_bytes=MAX_EMBEDDED_FILE_PARSE_BYTES,
        ))
    return elements


def _chart_elements(root: ET.Element, *, chart_index: int, location_prefix: str, base_metadata: dict[str, Any] | None = None) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    metadata_base = base_metadata or {}
    chart_kind = _chart_type(root)
    chart_title_node = root.find(".//c:title", XML_NS)
    chart_title = _xml_text(chart_title_node, ("v", "t")) if chart_title_node is not None else None
    series_payloads: list[dict[str, Any]] = []

    for series_index, series in enumerate(root.findall(".//c:ser", XML_NS), start=1):
        title_node = series.find(".//c:tx", XML_NS)
        name = _xml_text(title_node if title_node is not None else series, ("v", "t")) or f"Series {series_index}"
        categories = [_clean_text(node.text or "") for node in series.findall(".//c:cat//c:v", XML_NS)]
        values = [_clean_text(node.text or "") for node in series.findall(".//c:val//c:v", XML_NS)]
        data_labels = series.find("c:dLbls", XML_NS)
        label_fields = [
            _chart_label_field_name(flag_name)
            for flag_name in (
                "showLegendKey",
                "showSerName",
                "showCatName",
                "showVal",
                "showPercent",
                "showBubbleSize",
                "showLeaderLines",
            )
            if _chart_flag_enabled(data_labels, flag_name)
        ]
        label_metadata = _chart_data_label_metadata(data_labels)
        visual_metadata = _chart_series_visual_metadata(series)
        point_visuals = _chart_point_visual_metadata(series)
        trendlines = []
        for trendline_index, trendline in enumerate(series.findall("c:trendline", XML_NS), start=1):
            trendline_metadata = _chart_trendline_metadata(trendline)
            trendline_type = _clean_text(str(trendline_metadata.get("chart_trendline_type") or "trendline"))
            trendline_name = _clean_text(str(trendline_metadata.get("chart_trendline_name") or ""))
            trendline_text_parts = [
                name,
                f"{trendline_name or trendline_type} trendline",
            ]
            if trendline_metadata.get("chart_trendline_display_equation"):
                trendline_text_parts.append("shows equation")
            if trendline_metadata.get("chart_trendline_display_r_squared"):
                trendline_text_parts.append("shows R-squared")
            if trendline_metadata.get("chart_trendline_visual_summary"):
                trendline_text_parts.append(str(trendline_metadata["chart_trendline_visual_summary"]))
            trendlines.append({
                "index": trendline_index,
                "text": "; ".join(part for part in trendline_text_parts if part),
                "metadata": trendline_metadata,
            })
        series_payloads.append({
            "index": series_index,
            "name": name,
            "categories": categories,
            "values": values,
            "label_fields": label_fields,
            "label_metadata": label_metadata,
            "visual_metadata": visual_metadata,
            "point_visuals": point_visuals,
            "trendlines": trendlines,
        })

    for axis_index, axis in enumerate(
        [
            *root.findall(".//c:catAx", XML_NS),
            *root.findall(".//c:valAx", XML_NS),
            *root.findall(".//c:dateAx", XML_NS),
            *root.findall(".//c:serAx", XML_NS),
        ],
        start=1,
    ):
        axis_type = _chart_axis_type(axis)
        axis_title_node = axis.find("c:title", XML_NS)
        axis_title = _xml_text(axis_title_node, ("v", "t")) if axis_title_node is not None else None
        axis_id = _chart_child_val(axis, "axId")
        axis_position = _chart_child_val(axis, "axPos")
        number_format_node = axis.find("c:numFmt", XML_NS)
        number_format = _attr_by_local_name(number_format_node, "formatCode") if number_format_node is not None else None
        text_parts = [f"{axis_type.title()} axis"]
        if axis_title:
            text_parts.append(axis_title)
        if axis_position:
            text_parts.append(f"position {axis_position}")
        if number_format:
            text_parts.append(f"format {number_format}")
        element = _as_element(
            "chart_axis",
            "; ".join(text_parts),
            location=f"{location_prefix} {chart_index} axis {axis_index}",
            **metadata_base,
            chart=chart_index,
            chart_type=chart_kind,
            chart_title=chart_title,
            chart_axis_index=axis_index,
            chart_axis_type=axis_type,
            chart_axis_id=axis_id,
            chart_axis_title=axis_title,
            chart_axis_position=axis_position,
            chart_axis_number_format=number_format,
        )
        if element:
            elements.append(element)

    legend = root.find(".//c:legend", XML_NS)
    if legend is not None:
        legend_position = _chart_child_val(legend, "legendPos")
        legend_overlay = _chart_child_val(legend, "overlay")
        series_names = [str(payload["name"]) for payload in series_payloads if payload.get("name")]
        text_parts = ["Legend"]
        if legend_position:
            text_parts.append(f"position {legend_position}")
        if legend_overlay:
            text_parts.append(f"overlay {legend_overlay}")
        if series_names:
            text_parts.append(f"series {', '.join(series_names[:8])}")
        element = _as_element(
            "chart_legend",
            "; ".join(text_parts),
            location=f"{location_prefix} {chart_index} legend",
            **metadata_base,
            chart=chart_index,
            chart_type=chart_kind,
            chart_title=chart_title,
            chart_legend_position=legend_position,
            chart_legend_overlay=legend_overlay,
            chart_legend_series=series_names[:12],
        )
        if element:
            elements.append(element)

    for payload in series_payloads:
        series_index = int(payload["index"])
        name = str(payload["name"])
        categories = list(payload["categories"])
        values = list(payload["values"])
        label_fields = list(payload["label_fields"])
        label_metadata = dict(payload.get("label_metadata") or {})
        visual_metadata = dict(payload.get("visual_metadata") or {})
        point_visuals = dict(payload.get("point_visuals") or {})
        pairs = [f"{category}: {value}" if category else value for category, value in list(zip(categories, values))[:8]]
        text = "; ".join(part for part in [name, ", ".join(pairs), visual_metadata.get("chart_series_visual_summary")] if part)
        element = _as_element(
            "chart_series",
            text,
            location=f"{location_prefix} {chart_index} series {series_index}",
            **metadata_base,
            **visual_metadata,
            chart=chart_index,
            series=series_index,
            chart_type=chart_kind,
            chart_title=chart_title,
            series_name=name,
            categories=categories[:12],
            values=values[:12],
        )
        if element:
            elements.append(element)

        for point_index, (category, value) in enumerate(zip(categories, values), start=1):
            point_metadata = dict(point_visuals.get(point_index) or {})
            point_label = _clean_text(" ".join(part for part in [name, category] if part))
            point_text = f"{name}; {category}: {value}" if category else f"{name}: {value}"
            if point_metadata.get("chart_point_visual_summary"):
                point_text = f"{point_text}; {point_metadata['chart_point_visual_summary']}"
            point = _as_element(
                "chart_point",
                point_text,
                location=f"{location_prefix} {chart_index} series {series_index} point {point_index}",
                **metadata_base,
                **point_metadata,
                chart=chart_index,
                series=series_index,
                point=point_index,
                chart_label=chart_title,
                chart_point_label=point_label,
                chart_type=chart_kind,
                chart_title=chart_title,
                series_name=name,
                category=category,
                value=value,
            )
            if point:
                elements.append(point)

        if label_fields:
            label_examples = [f"{category}: {value}" if category else value for category, value in list(zip(categories, values))[:8]]
            label_text = f"{name} data labels show {', '.join(label_fields)}"
            if label_examples:
                label_text = f"{label_text}: {', '.join(label_examples)}"
            label_details = []
            if label_metadata.get("chart_label_position"):
                label_details.append(f"position {label_metadata['chart_label_position']}")
            if label_metadata.get("chart_label_number_format"):
                label_details.append(f"format {label_metadata['chart_label_number_format']}")
            if label_details:
                label_text = f"{label_text}; {'; '.join(label_details)}"
            element = _as_element(
                "chart_data_label",
                label_text,
                location=f"{location_prefix} {chart_index} series {series_index} data labels",
                **metadata_base,
                **label_metadata,
                chart=chart_index,
                series=series_index,
                chart_type=chart_kind,
                chart_title=chart_title,
                series_name=name,
                chart_label_fields=label_fields,
                chart_label_examples=label_examples,
            )
            if element:
                elements.append(element)

        for trendline in payload.get("trendlines", []) or []:
            if not isinstance(trendline, dict):
                continue
            trendline_index = int(trendline.get("index") or 1)
            trendline_metadata = dict(trendline.get("metadata") or {})
            element = _as_element(
                "chart_trendline",
                str(trendline.get("text") or f"{name} trendline"),
                location=f"{location_prefix} {chart_index} series {series_index} trendline {trendline_index}",
                **metadata_base,
                **trendline_metadata,
                chart=chart_index,
                series=series_index,
                trendline=trendline_index,
                chart_type=chart_kind,
                chart_title=chart_title,
                series_name=name,
            )
            if element:
                elements.append(element)

    return elements


def _docx_table_section_map(root: ET.Element) -> dict[int, str]:
    body = root.find(".//w:body", XML_NS)
    if body is None:
        return {}

    sections: dict[int, str] = {}
    current_heading = ""
    table_index = 0
    for child in list(body):
        local_name = _local_name(child.tag)
        if local_name == "p":
            style_node = child.find(".//w:pStyle", XML_NS)
            style = style_node.attrib.get(f"{{{XML_NS['w']}}}val") if style_node is not None else ""
            text = _xml_text(child, ("t",))
            if style and style.lower().startswith("heading") and text:
                current_heading = text
        elif local_name == "tbl":
            table_index += 1
            if current_heading:
                sections[table_index] = current_heading
    return sections


def _docx_table_cell_elements(
    table: ET.Element,
    *,
    table_index: int,
    section: str | None = None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    rows = [
        [_xml_text(cell, ("t",)) for cell in row.findall(".//w:tc", XML_NS)]
        for row in table.findall(".//w:tr", XML_NS)
    ]
    for row_index, row in enumerate(table.findall(".//w:tr", XML_NS), start=1):
        for cell_index, cell in enumerate(row.findall(".//w:tc", XML_NS), start=1):
            cell_text = _xml_text(cell, ("t",))
            cell_properties = cell.find("w:tcPr", XML_NS)
            grid_span = _word_attr(cell_properties.find("w:gridSpan", XML_NS) if cell_properties is not None else None, "val")
            h_merge_node = cell_properties.find("w:hMerge", XML_NS) if cell_properties is not None else None
            v_merge_node = cell_properties.find("w:vMerge", XML_NS) if cell_properties is not None else None
            h_merge = _word_attr(h_merge_node, "val") or ("continue" if h_merge_node is not None else None)
            v_merge = _word_attr(v_merge_node, "val") or ("continue" if v_merge_node is not None else None)
            visual_metadata = _docx_table_cell_visual_metadata(cell_properties)
            merge_flags = [
                flag
                for flag, value in (
                    ("horizontal", h_merge),
                    ("vertical", v_merge),
                )
                if value
            ]
            runs = _word_run_styles(cell)
            emphasis_terms = [run["text"] for run in runs if run.get("bold") or run.get("italic") or run.get("underline")]
            context_metadata = _table_cell_context_metadata(rows, row_index, cell_index)
            element = _as_element(
                "table_cell",
                cell_text,
                location=f"table {table_index} cell {row_index}.{cell_index}",
                section=section,
                table=table_index,
                row=row_index,
                column=cell_index,
                table_cell_reference=f"R{row_index}C{cell_index}",
                table_cell_grid_span=_first_int(grid_span) if grid_span else None,
                table_cell_h_merge=h_merge,
                table_cell_v_merge=v_merge,
                table_cell_merge_flags=merge_flags or None,
                **context_metadata,
                **visual_metadata,
                runs=runs[:24] if runs else None,
                emphasis_terms=emphasis_terms[:12] if emphasis_terms else None,
                style_summary=_style_summary_from_runs(runs),
            )
            if element:
                elements.append(element)
    return elements


DOCX_SDT_CONTROL_TYPES = {
    "checkbox": "checkbox",
    "checkboxes": "checkbox",
    "checkboxlist": "checkbox",
    "combobox": "combo_box",
    "date": "date",
    "dropdownlist": "dropdown",
    "group": "group",
    "picture": "picture",
    "plaintext": "plain_text",
    "repeatingsection": "repeating_section",
    "repeatingsectionitem": "repeating_section_item",
    "richtext": "rich_text",
    "text": "plain_text",
}


def _first_descendant_by_local_name_casefold(node: ET.Element | None, *local_names: str) -> ET.Element | None:
    if node is None:
        return None
    wanted = {name.casefold() for name in local_names}
    for descendant in node.iter():
        if _local_name(descendant.tag).casefold() in wanted:
            return descendant
    return None


def _descendants_by_local_name_casefold(node: ET.Element | None, *local_names: str) -> list[ET.Element]:
    if node is None:
        return []
    wanted = {name.casefold() for name in local_names}
    return [
        descendant
        for descendant in node.iter()
        if _local_name(descendant.tag).casefold() in wanted
    ]


def _word_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "on", "yes", "checked"}:
        return True
    if normalized in {"0", "false", "off", "no", "unchecked"}:
        return False
    return None


def _docx_sdt_control_node(properties: ET.Element | None) -> tuple[str | None, ET.Element | None]:
    if properties is None:
        return None, None
    for descendant in properties.iter():
        control_type = DOCX_SDT_CONTROL_TYPES.get(_local_name(descendant.tag).casefold())
        if control_type:
            return control_type, descendant
    checked_node = _first_descendant_by_local_name_casefold(properties, "checked")
    if checked_node is not None:
        return "checkbox", checked_node
    return None, None


def _docx_sdt_list_options(control_node: ET.Element | None) -> tuple[list[str], list[str]]:
    options: list[str] = []
    values: list[str] = []
    for item in _descendants_by_local_name_casefold(control_node, "listItem"):
        display = _attr_by_local_name(item, "displayText")
        value = _attr_by_local_name(item, "value")
        option = _clean_text(str(display or value or ""))
        if option:
            options.append(option)
        value_text = _clean_text(str(value or ""))
        if value_text and value_text != option:
            values.append(value_text)
    return _unique_preserve_order(options), _unique_preserve_order(values)


def _docx_sdt_form_field_metadata(field: ET.Element, field_index: int) -> dict[str, Any]:
    properties = field.find("w:sdtPr", XML_NS)
    content = field.find("w:sdtContent", XML_NS)
    content_text = _xml_text(content if content is not None else field, ("t", "delText"))
    control_type, control_node = _docx_sdt_control_node(properties)

    alias = _word_attr(properties.find("w:alias", XML_NS) if properties is not None else None, "val")
    tag = _word_attr(properties.find("w:tag", XML_NS) if properties is not None else None, "val")
    control_id = _word_attr(properties.find("w:id", XML_NS) if properties is not None else None, "val")
    lock_node = properties.find("w:lock", XML_NS) if properties is not None else None
    placeholder_node = _first_descendant_by_local_name_casefold(properties, "placeholder")
    placeholder_part = _attr_by_local_name(
        _first_descendant_by_local_name_casefold(placeholder_node, "docPart"),
        "val",
    )

    checked_source = control_node if control_node is not None else properties
    checked_node = _first_descendant_by_local_name_casefold(checked_source, "checked")
    checked = _word_bool(_attr_by_local_name(checked_node, "val"))
    if checked is None and content_text:
        checked = _word_bool(content_text)
    if checked is None and content_text in {"\u2611", "\u2612"}:
        checked = True
    elif checked is None and content_text == "\u2610":
        checked = False

    options, option_values = _docx_sdt_list_options(control_node)
    selected_option = ""
    if options and content_text:
        normalized_content = content_text.casefold()
        for option in [*options, *option_values]:
            if normalized_content == option.casefold():
                selected_option = option
                break

    date_format_node = _first_descendant_by_local_name_casefold(control_node, "dateFormat")
    date_full_value = _attr_by_local_name(control_node, "fullDate")
    date_calendar = _attr_by_local_name(
        _first_descendant_by_local_name_casefold(control_node, "calendar"),
        "val",
    )

    label = _clean_text(str(alias or tag or placeholder_part or content_text or f"field {field_index}"))
    if checked is not None:
        value = "checked" if checked else "unchecked"
    elif selected_option:
        value = selected_option
    else:
        value = content_text or date_full_value or ""

    metadata: dict[str, Any] = {
        "form_kind": "docx_content_control",
        "field_type": control_type or "content_control",
        "label": label,
        "value": value,
        "tag": tag,
        "checked": checked,
        "control_id": control_id,
        "control_name": alias or tag,
        "control_part": "word/document.xml",
        "form_field_alias": alias,
        "form_field_tag": tag,
        "form_field_id": control_id,
        "form_field_placeholder": placeholder_part,
        "form_field_checked": checked,
        "form_field_options": options or None,
        "form_field_option_values": option_values or None,
        "form_field_selected_option": selected_option or None,
        "form_field_date_format": _attr_by_local_name(date_format_node, "val"),
        "form_field_date_full_value": date_full_value,
        "form_field_calendar": date_calendar,
        "form_field_lock": _word_attr(lock_node, "val"),
    }
    return {key: value for key, value in metadata.items() if value not in (None, "", [], {})}


def _extract_docx(content: bytes) -> list[DocumentElement]:
    elements = _office_package_metadata_elements(content)
    root = _read_zip_xml(content, "word/document.xml")
    if root is None:
        return [*elements, *_extract_plain_text(content, ".txt")]

    document_relationships = _part_relationships(content, "word/document.xml")
    header_parts = [
        rel["target"]
        for rel in document_relationships.values()
        if rel.get("target") and rel.get("type", "").endswith("/header")
    ] or _zip_names(content, "word/header")
    footer_parts = [
        rel["target"]
        for rel in document_relationships.values()
        if rel.get("target") and rel.get("type", "").endswith("/footer")
    ] or _zip_names(content, "word/footer")

    elements.extend(_docx_part_text_elements(
        content,
        header_parts,
        element_type="header",
        location_label="header",
    ))
    elements.extend(_docx_part_text_elements(
        content,
        footer_parts,
        element_type="footer",
        location_label="footer",
    ))
    elements.extend(_docx_comment_elements(content, document_root=root))
    elements.extend(_docx_note_elements(content, "word/footnotes.xml", "footnote", document_root=root))
    elements.extend(_docx_note_elements(content, "word/endnotes.xml", "endnote", document_root=root))
    elements.extend(_docx_text_box_elements(root, part_name="word/document.xml"))
    table_sections = _docx_table_section_map(root)
    for table_index, table in enumerate(root.findall(".//w:tbl", XML_NS), start=1):
        rows = [
            [_xml_text(cell, ("t",)) for cell in row.findall(".//w:tc", XML_NS)]
            for row in table.findall(".//w:tr", XML_NS)
        ]
        elements.extend(_table_elements_from_rows(
            rows,
            location_prefix=f"table {table_index}",
            table=table_index,
            base_metadata={"section": table_sections.get(table_index)},
        ))
        elements.extend(_docx_table_cell_elements(
            table,
            table_index=table_index,
            section=table_sections.get(table_index),
        ))

    for field_index, field in enumerate(root.findall(".//w:sdt", XML_NS), start=1):
        metadata = _docx_sdt_form_field_metadata(field, field_index)
        label = str(metadata.get("label") or f"field {field_index}")
        value = str(metadata.get("value") or "")
        element = _as_element(
            "form_field",
            f"{label}: {value}" if value else label,
            location=f"form field {field_index}",
            **metadata,
        )
        if element:
            elements.append(element)

    for image_index, drawing in enumerate(root.findall(".//w:drawing", XML_NS), start=1):
        blip = drawing.find(".//a:blip", XML_NS)
        if blip is None:
            continue
        relationship_id = None
        relationship_id = (
            blip.attrib.get(f"{{{XML_NS['r']}}}embed")
            or blip.attrib.get(f"{{{XML_NS['r']}}}link")
        )
        relationship = document_relationships.get(relationship_id or "", {})
        image_part = relationship.get("target")
        properties = _first_descendant_by_local_name(drawing, "docPr", "cNvPr")
        title = _drawing_attr(properties, "title")
        description = _drawing_attr(properties, "descr")
        name = _drawing_attr(properties, "name")
        image_metadata = _zip_image_metadata(content, image_part)
        label = _image_element_text(title, description, name, image_part, metadata=image_metadata)
        element = _as_element(
            "image",
            label,
            location=f"image {image_index}",
            image=image_index,
            title=title,
            description=description,
            name=name,
            relationship_id=relationship_id,
            target_mode=relationship.get("target_mode"),
            image_part=image_part,
            ocr_source="docx_image" if image_metadata.get("ocr_text") else None,
            **image_metadata,
        )
        if element:
            elements.append(element)
        ocr_element = _image_ocr_text_element(
            image_metadata,
            location=f"image {image_index} OCR",
            image_part=image_part,
            ocr_source="docx_image",
            base_metadata={
                "image": image_index,
                "title": title,
                "description": description,
                "name": name,
                "relationship_id": relationship_id,
                "target_mode": relationship.get("target_mode"),
            },
        )
        if ocr_element:
            elements.append(ocr_element)

    seen_chart_parts: set[str] = set()
    chart_sources = [
        (drawing, chart_ref)
        for drawing in root.findall(".//w:drawing", XML_NS)
        for chart_ref in drawing.findall(".//c:chart", XML_NS)
    ]
    for chart_ref_index, (chart_container, chart_ref) in enumerate(chart_sources, start=1):
        relationship_id = chart_ref.attrib.get(f"{{{XML_NS['r']}}}id")
        chart_part = document_relationships.get(relationship_id or "", {}).get("target")
        chart_root = _read_zip_xml(content, chart_part) if chart_part else None
        if chart_root is None:
            continue
        if chart_part in seen_chart_parts:
            continue
        if chart_part:
            seen_chart_parts.add(chart_part)
        chart_index = _first_int(chart_part) or chart_ref_index
        elements.extend(_chart_elements(
            chart_root,
            chart_index=chart_index,
            location_prefix="chart",
            base_metadata={
                "relationship_id": relationship_id,
                "chart_part": chart_part,
                **_chart_source_metadata(content, chart_part, chart_root),
            },
        ))
        elements.extend(_chart_source_child_elements(
            content,
            chart_part,
            chart_root,
            location_prefix=f"chart {chart_index}",
            base_metadata={
                "relationship_id": relationship_id,
                "chart": chart_index,
                "chart_part": chart_part,
            },
        ))

    seen_diagram_parts: set[str] = set()
    diagram_sources = [
        (drawing, rel_ids)
        for drawing in root.findall(".//w:drawing", XML_NS)
        for rel_ids in drawing.findall(".//dgm:relIds", XML_NS)
    ]
    for diagram_ref_index, (diagram_container, rel_ids) in enumerate(diagram_sources, start=1):
        relationship_id = (
            rel_ids.attrib.get(f"{{{XML_NS['r']}}}dm")
            or _attr_by_local_name(rel_ids, "dm")
        )
        diagram_part = document_relationships.get(relationship_id or "", {}).get("target")
        if not diagram_part or diagram_part in seen_diagram_parts:
            continue
        seen_diagram_parts.add(diagram_part)
        elements.extend(_drawingml_diagram_elements(
            content,
            diagram_part,
            slide_number=None,
            slide_title=None,
            relationship_id=relationship_id,
            location_prefix=f"diagram {diagram_ref_index}",
            base_metadata={"source_part": "word/document.xml"},
        ))

    for diagram_name in _zip_names(content, "word/diagrams/data"):
        if diagram_name in seen_diagram_parts:
            continue
        elements.extend(_drawingml_diagram_elements(
            content,
            diagram_name,
            slide_number=None,
            slide_title=None,
            location_prefix="diagram",
            base_metadata={"source_part": "word/document.xml"},
        ))

    for para_index, paragraph in enumerate(root.findall(".//w:p", XML_NS), start=1):
        text = _xml_text(paragraph, ("t",))
        style = None
        style_node = paragraph.find(".//w:pStyle", XML_NS)
        if style_node is not None:
            style = style_node.attrib.get(f"{{{XML_NS['w']}}}val")
        element_type = "heading" if style and style.lower().startswith("heading") else "paragraph"
        runs = _word_run_styles(paragraph)
        emphasis_terms = [run["text"] for run in runs if run.get("bold") or run.get("italic") or run.get("underline")]
        element = _as_element(
            element_type,
            text,
            location=f"paragraph {para_index}",
            style=style,
            runs=runs[:24] if runs else None,
            emphasis_terms=emphasis_terms[:12] if emphasis_terms else None,
            style_summary=_style_summary_from_runs(runs),
        )
        if element:
            elements.append(element)
        elements.extend(_docx_revision_elements(
            paragraph,
            location_prefix=f"paragraph {para_index}",
            base_metadata={"style": style},
        ))
        elements.extend(_docx_hyperlink_elements(
            paragraph,
            document_relationships,
            location_prefix=f"paragraph {para_index}",
            base_metadata={"style": style},
        ))

    return elements


def _pptx_slide_parts(content: bytes) -> list[dict[str, Any]]:
    presentation_root = _read_zip_xml(content, "ppt/presentation.xml")
    relationships = _part_relationships(content, "ppt/presentation.xml")
    slide_parts: list[dict[str, Any]] = []

    if presentation_root is not None and relationships:
        for index, slide_id_node in enumerate(
            [node for node in presentation_root.iter() if _local_name(node.tag) == "sldId"],
            start=1,
        ):
            relationship_id = slide_id_node.attrib.get(f"{{{XML_NS['r']}}}id")
            relationship = relationships.get(relationship_id or "", {})
            slide_part = relationship.get("target") or ""
            if not slide_part:
                continue
            slide_parts.append({
                "index": index,
                "part": slide_part,
                "slide_id": _clean_text(str(slide_id_node.attrib.get("id") or "")),
                "relationship_id": relationship_id,
            })

    if slide_parts:
        return slide_parts

    return [
        {
            "index": index,
            "part": slide_part,
            "slide_id": "",
            "relationship_id": "",
        }
        for index, slide_part in enumerate(_zip_names(content, "ppt/slides/slide"), start=1)
    ]


def _pptx_slide_metadata(slide_info: dict[str, Any], root: ET.Element, slide_title: str | None) -> dict[str, Any]:
    slide_index = int(slide_info.get("index") or 0)
    show_value = _clean_text(str(_attr_by_local_name(root, "show") or ""))
    hidden_values = {"0", "false", "no", "off"}
    visibility = "hidden" if show_value.lower() in hidden_values else "visible"
    return {
        "slide": slide_index,
        "slide_title": _clean_text(str(slide_title or "")),
        "slide_id": _clean_text(str(slide_info.get("slide_id") or "")),
        "slide_part": _clean_text(str(slide_info.get("part") or "")),
        "slide_relationship_id": _clean_text(str(slide_info.get("relationship_id") or "")),
        "slide_state": visibility,
        "slide_visibility": visibility,
    }


def _presentation_slide_metadata_element(slide_metadata: dict[str, Any]) -> DocumentElement | None:
    slide_number = slide_metadata.get("slide")
    slide_title = _clean_text(str(slide_metadata.get("slide_title") or ""))
    visibility = _clean_text(str(slide_metadata.get("slide_visibility") or "visible")) or "visible"
    slide_part = _clean_text(str(slide_metadata.get("slide_part") or ""))
    label = f"Slide {slide_number}: {slide_title}" if slide_title else f"Slide {slide_number}"
    text_parts = [
        label,
        f"visibility: {visibility}",
        f"part: {slide_part}" if slide_part else "",
    ]
    return _as_element(
        "slide_metadata",
        "; ".join(part for part in text_parts if part),
        location=f"presentation slide {slide_number}",
        **slide_metadata,
    )


def _extract_pptx(content: bytes) -> list[DocumentElement]:
    elements = _office_package_metadata_elements(content)
    seen_chart_parts: set[str] = set()
    seen_comment_parts: set[str] = set()
    seen_diagram_parts: set[str] = set()
    seen_notes_parts: set[str] = set()
    comment_authors = _pptx_comment_author_map(content)
    slide_metadata_by_index: dict[int, dict[str, Any]] = {}
    slide_size_metadata = _pptx_slide_size_metadata(content)

    for slide_info in _pptx_slide_parts(content):
        slide_name = str(slide_info.get("part") or "")
        slide_number = int(slide_info.get("index") or (_first_int(slide_name) or len(elements) + 1))
        root = _read_zip_xml(content, slide_name)
        if root is None:
            continue

        slide_relationships = _part_relationships(content, slide_name)
        title_texts = []
        shape_nodes = root.findall(".//p:sp", XML_NS)
        for shape in shape_nodes:
            text = _xml_text(shape, ("t",))
            if not text:
                continue
            placeholder = shape.find(".//p:ph", XML_NS)
            placeholder_type = placeholder.attrib.get("type") if placeholder is not None else None
            if placeholder_type in {"title", "ctrTitle"}:
                title_texts.append(text)

        slide_title = title_texts[0] if title_texts else None
        slide_metadata = _pptx_slide_metadata(slide_info, root, slide_title)
        slide_metadata_by_index[slide_number] = slide_metadata
        slide_metadata_element = _presentation_slide_metadata_element(slide_metadata)
        if slide_metadata_element:
            elements.append(slide_metadata_element)
        seen_slide_links: set[tuple[Any, ...]] = set()
        shape_labels_by_id = _pptx_connectable_shape_labels(root)
        group_records, group_context_by_shape_id = _pptx_group_records(root, slide_size_metadata)

        for group_record in group_records:
            child_labels = group_record.get("group_child_labels") if isinstance(group_record.get("group_child_labels"), list) else []
            group_label = _clean_text(str(group_record.get("group_label") or ""))
            group_text = (
                f"{group_label} groups {', '.join(str(label) for label in child_labels[:8])}"
                if child_labels
                else group_label
            )
            element = _as_element(
                "shape_group",
                group_text,
                location=f"slide {slide_number} group {group_record['group_index']}",
                **group_record.get("geometry", {}),
                slide=slide_number,
                slide_title=slide_title,
                group_index=group_record.get("group_index"),
                group_id=group_record.get("group_id"),
                group_name=group_record.get("group_name"),
                group_title=group_record.get("group_title"),
                group_description=group_record.get("group_description"),
                group_label=group_label,
                group_child_shape_ids=group_record.get("group_child_shape_ids"),
                group_child_labels=group_record.get("group_child_labels"),
                group_child_count=group_record.get("group_child_count"),
            )
            if element:
                elements.append(element)

        for shape_index, shape in enumerate(shape_nodes, start=1):
            text = _xml_text(shape, ("t",))
            placeholder = shape.find(".//p:ph", XML_NS)
            placeholder_type = placeholder.attrib.get("type") if placeholder is not None else None
            shape_metadata = _pptx_shape_metadata(shape)
            shape_id = shape_metadata.get("shape_id")
            shape_name = shape_metadata.get("shape_name")
            shape_title = shape_metadata.get("shape_title")
            shape_description = shape_metadata.get("shape_description")
            shape_metadata_text = _pptx_shape_metadata_text(shape_metadata)
            shape_geometry = _pptx_geometry_metadata(shape, slide_size_metadata)
            shape_group_context = group_context_by_shape_id.get(str(shape_id or ""), {})
            if shape_metadata_text and (shape_title or shape_description or not text):
                element = _as_element(
                    "shape_metadata",
                    shape_metadata_text,
                    location=f"slide {slide_number} shape {shape_index}",
                    **shape_geometry,
                    **shape_group_context,
                    slide=slide_number,
                    slide_title=slide_title,
                    placeholder=placeholder_type,
                    shape_index=shape_index,
                    shape_id=shape_id,
                    shape_name=shape_name,
                    shape_title=shape_title,
                    shape_description=shape_description,
                )
                if element:
                    elements.append(element)

            if text:
                element_type = "slide_title" if placeholder_type in {"title", "ctrTitle"} else "slide_text"
                runs = _drawing_run_styles(shape)
                emphasis_terms = [run["text"] for run in runs if run.get("bold") or run.get("italic") or run.get("underline")]
                bullet_levels = _paragraph_bullet_levels(shape) or None
                element = _as_element(
                    element_type,
                    text,
                    location=f"slide {slide_number}",
                    **shape_geometry,
                    **shape_group_context,
                    slide=slide_number,
                    slide_title=slide_title,
                    placeholder=placeholder_type,
                    shape_index=shape_index,
                    shape_id=shape_id,
                    shape_name=shape_name,
                    shape_title=shape_title,
                    shape_description=shape_description,
                    bullet_levels=bullet_levels,
                    runs=runs[:24] if runs else None,
                    emphasis_terms=emphasis_terms[:12] if emphasis_terms else None,
                    style_summary=_style_summary_from_runs(runs, bullet_levels),
                )
                if element:
                    elements.append(element)

            for link in _drawing_hyperlink_elements(
                shape,
                slide_relationships,
                location_prefix=f"slide {slide_number} shape {shape_index}",
                base_metadata={
                    **shape_geometry,
                    **shape_group_context,
                    "slide": slide_number,
                    "slide_title": slide_title,
                    "placeholder": placeholder_type,
                    "shape_index": shape_index,
                    "shape_id": shape_id,
                    "shape_name": shape_name,
                    "shape_title": shape_title,
                    "shape_description": shape_description,
                },
                label_context=text or shape_metadata_text or shape_title or shape_description or shape_name,
            ):
                link_key = (
                    link.metadata.get("relationship_id"),
                    link.metadata.get("uri"),
                    link.metadata.get("link_kind"),
                    link.text,
                )
                if link_key in seen_slide_links:
                    continue
                seen_slide_links.add(link_key)
                elements.append(link)

        elements.extend(_pptx_slide_inherited_elements(
            content,
            slide_relationships,
            slide_number=slide_number,
            slide_title=slide_title,
        ))

        for connector_index, connector in enumerate(root.findall(".//p:cxnSp", XML_NS), start=1):
            text = _xml_text(connector, ("t",))
            connector_metadata = _pptx_shape_metadata(connector)
            connector_id = connector_metadata.get("shape_id")
            connector_name = connector_metadata.get("shape_name")
            connector_title = connector_metadata.get("shape_title")
            connector_description = connector_metadata.get("shape_description")
            source_shape_id, source_connection_index = _pptx_connection_endpoint(connector, "stCxn")
            target_shape_id, target_connection_index = _pptx_connection_endpoint(connector, "endCxn")
            source_label = shape_labels_by_id.get(str(source_shape_id or ""))
            target_label = shape_labels_by_id.get(str(target_shape_id or ""))
            relation = _pptx_connector_relation(connector_metadata, text)
            connector_label = _pptx_shape_graph_label(connector_metadata, text)
            connector_geometry = _pptx_geometry_metadata(connector, slide_size_metadata)
            connector_group_context = group_context_by_shape_id.get(str(connector_id or ""), {})
            connector_text = (
                f"{source_label} -> {target_label}: {relation}"
                if source_label and target_label
                else connector_label or relation
            )
            element = _as_element(
                "shape_connector",
                connector_text,
                location=f"slide {slide_number} connector {connector_index}",
                **connector_geometry,
                **connector_group_context,
                slide=slide_number,
                slide_title=slide_title,
                connector_index=connector_index,
                connector_id=connector_id,
                connector_name=connector_name,
                connector_title=connector_title,
                connector_description=connector_description,
                source_shape_id=source_shape_id,
                target_shape_id=target_shape_id,
                source_connection_index=source_connection_index,
                target_connection_index=target_connection_index,
                source_label=source_label or None,
                target_label=target_label or None,
                relation=relation,
            )
            if element:
                elements.append(element)

            for link in _drawing_hyperlink_elements(
                connector,
                slide_relationships,
                location_prefix=f"slide {slide_number} connector {connector_index}",
                base_metadata={
                    **connector_geometry,
                    **connector_group_context,
                    "slide": slide_number,
                    "slide_title": slide_title,
                    "connector_index": connector_index,
                    "connector_id": connector_id,
                    "connector_name": connector_name,
                    "connector_title": connector_title,
                    "connector_description": connector_description,
                    "source_shape_id": source_shape_id,
                    "target_shape_id": target_shape_id,
                    "source_label": source_label or None,
                    "target_label": target_label or None,
                    "relation": relation,
                },
                label_context=connector_text,
            ):
                link_key = (
                    link.metadata.get("relationship_id"),
                    link.metadata.get("uri"),
                    link.metadata.get("link_kind"),
                    link.text,
                )
                if link_key in seen_slide_links:
                    continue
                seen_slide_links.add(link_key)
                elements.append(link)

        comment_parts = [
            relationship["target"]
            for relationship in slide_relationships.values()
            if relationship.get("target")
            and (
                relationship.get("type", "").endswith("/comments")
                or "/comments/" in relationship.get("target", "")
                or "/threadedComments/" in relationship.get("target", "")
            )
        ]
        for comment_part in _unique_preserve_order(comment_parts):
            seen_comment_parts.add(comment_part)
            elements.extend(_pptx_comment_elements(
                content,
                comment_part,
                author_map=comment_authors,
                slide_number=slide_number,
                slide_title=slide_title,
                slide_size_metadata=slide_size_metadata,
            ))

        notes_relationships = [
            (relationship_id, relationship)
            for relationship_id, relationship in slide_relationships.items()
            if relationship.get("target")
            and (
                relationship.get("type", "").endswith("/notesSlide")
                or "/notesSlides/" in relationship.get("target", "")
            )
        ]
        seen_slide_notes: set[str] = set()
        for notes_relationship_id, relationship in notes_relationships:
            notes_part = relationship["target"]
            if notes_part in seen_slide_notes:
                continue
            seen_slide_notes.add(notes_part)
            notes_root = _read_zip_xml(content, notes_part)
            if notes_root is None:
                continue
            seen_notes_parts.add(notes_part)
            element = _as_element(
                "speaker_notes",
                _xml_text(notes_root, ("t",)),
                location=f"slide {slide_number} speaker notes",
                **slide_metadata,
                notes_part=notes_part,
                notes_relationship_id=notes_relationship_id,
            )
            if element:
                elements.append(element)

        elements.extend(_pptx_media_elements(
            content,
            slide_relationships,
            slide_number=slide_number,
            slide_title=slide_title,
        ))
        elements.extend(_pptx_control_elements(
            content,
            root,
            slide_relationships,
            slide_number=slide_number,
            slide_title=slide_title,
        ))

        table_sources = [
            (frame, table)
            for frame in root.findall(".//p:graphicFrame", XML_NS)
            for table in frame.findall(".//a:tbl", XML_NS)
        ]
        if not table_sources:
            table_sources = [(table, table) for table in root.findall(".//a:tbl", XML_NS)]
        for table_index, (table_container, table) in enumerate(table_sources, start=1):
            table_geometry = _pptx_geometry_metadata(table_container, slide_size_metadata)
            table_shape_id = _pptx_shape_metadata(table_container).get("shape_id")
            table_group_context = group_context_by_shape_id.get(str(table_shape_id or ""), {})
            rows = [
                [_xml_text(cell, ("t",)) for cell in row.findall(".//a:tc", XML_NS)]
                for row in table.findall(".//a:tr", XML_NS)
            ]
            elements.extend(_table_elements_from_rows(
                rows,
                location_prefix=f"slide {slide_number} table {table_index}",
                table=table_index,
                base_metadata={"slide": slide_number, "slide_title": slide_title, **table_geometry, **table_group_context},
            ))
            for row_index, row in enumerate(table.findall(".//a:tr", XML_NS), start=1):
                for cell_index, cell in enumerate(row.findall(".//a:tc", XML_NS), start=1):
                    cell_text = _xml_text(cell, ("t",))
                    runs = _drawing_run_styles(cell)
                    emphasis_terms = [run["text"] for run in runs if run.get("bold") or run.get("italic") or run.get("underline")]
                    bullet_levels = _paragraph_bullet_levels(cell) or None
                    grid_span = _drawing_attr(cell, "gridSpan")
                    row_span = _drawing_attr(cell, "rowSpan")
                    h_merge = _drawing_attr(cell, "hMerge")
                    v_merge = _drawing_attr(cell, "vMerge")
                    visual_metadata = _pptx_table_cell_visual_metadata(cell)
                    context_metadata = _table_cell_context_metadata(rows, row_index, cell_index)
                    merge_flags = [
                        flag
                        for flag, value in (
                            ("horizontal", h_merge),
                            ("vertical", v_merge),
                        )
                        if str(value or "").lower() in {"1", "true", "on"}
                    ]
                    element = _as_element(
                        "table_cell",
                        cell_text,
                        location=f"slide {slide_number} table {table_index} cell {row_index}.{cell_index}",
                        **table_geometry,
                        **table_group_context,
                        slide=slide_number,
                        slide_title=slide_title,
                        table=table_index,
                        row=row_index,
                        column=cell_index,
                        table_cell_reference=f"R{row_index}C{cell_index}",
                        table_cell_grid_span=_first_int(grid_span) if grid_span else None,
                        table_cell_row_span=_first_int(row_span) if row_span else None,
                        table_cell_h_merge=h_merge,
                        table_cell_v_merge=v_merge,
                        table_cell_merge_flags=merge_flags or None,
                        **context_metadata,
                        **visual_metadata,
                        runs=runs[:24] if runs else None,
                        emphasis_terms=emphasis_terms[:12] if emphasis_terms else None,
                        bullet_levels=bullet_levels,
                        style_summary=_style_summary_from_runs(runs, bullet_levels),
                    )
                    if element:
                        elements.append(element)
                    for link in _drawing_hyperlink_elements(
                        cell,
                        slide_relationships,
                        location_prefix=f"slide {slide_number} table {table_index} cell {row_index}.{cell_index}",
                        base_metadata={
                            **table_geometry,
                            **table_group_context,
                            "slide": slide_number,
                            "slide_title": slide_title,
                            "table": table_index,
                            "row": row_index,
                            "cell": cell_index,
                        },
                        label_context=cell_text or slide_title,
                    ):
                        link_key = (
                            link.metadata.get("relationship_id"),
                            link.metadata.get("uri"),
                            link.metadata.get("link_kind"),
                            link.text,
                        )
                        if link_key in seen_slide_links:
                            continue
                        seen_slide_links.add(link_key)
                        elements.append(link)

        chart_sources = [
            (frame, chart_ref)
            for frame in root.findall(".//p:graphicFrame", XML_NS)
            for chart_ref in frame.findall(".//c:chart", XML_NS)
        ]
        if not chart_sources:
            chart_sources = [(chart_ref, chart_ref) for chart_ref in root.findall(".//c:chart", XML_NS)]
        for chart_ref_index, (chart_container, chart_ref) in enumerate(chart_sources, start=1):
            relationship_id = chart_ref.attrib.get(f"{{{XML_NS['r']}}}id")
            chart_part = slide_relationships.get(relationship_id or "", {}).get("target")
            chart_root = _read_zip_xml(content, chart_part) if chart_part else None
            if chart_root is None:
                continue
            if chart_part:
                seen_chart_parts.add(chart_part)
            chart_index = _first_int(chart_part) or chart_ref_index
            chart_geometry = _pptx_geometry_metadata(chart_container, slide_size_metadata)
            chart_shape_id = _pptx_shape_metadata(chart_container).get("shape_id")
            chart_group_context = group_context_by_shape_id.get(str(chart_shape_id or ""), {})
            elements.extend(_chart_elements(
                chart_root,
                chart_index=chart_index,
                location_prefix=f"slide {slide_number} chart",
                base_metadata={
                    **chart_geometry,
                    **chart_group_context,
                    "slide": slide_number,
                    "slide_title": slide_title,
                    "relationship_id": relationship_id,
                    "chart_part": chart_part,
                    **_chart_source_metadata(content, chart_part, chart_root),
                },
            ))
            elements.extend(_chart_source_child_elements(
                content,
                chart_part,
                chart_root,
                location_prefix=f"slide {slide_number} chart {chart_index}",
                base_metadata={
                    "slide": slide_number,
                    "slide_title": slide_title,
                    "relationship_id": relationship_id,
                    "chart": chart_index,
                    "chart_part": chart_part,
                },
            ))

        for picture_index, picture in enumerate(root.findall(".//p:pic", XML_NS), start=1):
            picture_properties = picture.find(".//p:cNvPr", XML_NS)
            title = _drawing_attr(picture_properties, "title")
            description = _drawing_attr(picture_properties, "descr")
            name = _drawing_attr(picture_properties, "name")
            blip = picture.find(".//a:blip", XML_NS)
            relationship_id = None
            if blip is not None:
                relationship_id = (
                    blip.attrib.get(f"{{{XML_NS['r']}}}embed")
                    or blip.attrib.get(f"{{{XML_NS['r']}}}link")
                )
            relationship = slide_relationships.get(relationship_id or "", {})
            image_part = relationship.get("target")
            image_metadata = _zip_image_metadata(content, image_part)
            label = _image_element_text(title, description, name, image_part, metadata=image_metadata)
            picture_geometry = _pptx_geometry_metadata(picture, slide_size_metadata)
            picture_shape_id = _pptx_shape_metadata(picture).get("shape_id")
            picture_group_context = group_context_by_shape_id.get(str(picture_shape_id or ""), {})
            element = _as_element(
                "image",
                label,
                location=f"slide {slide_number} image {picture_index}",
                **picture_geometry,
                **picture_group_context,
                slide=slide_number,
                image=picture_index,
                title=title,
                description=description,
                name=name,
                relationship_id=relationship_id,
                image_relationship_id=relationship_id,
                target_mode=relationship.get("target_mode"),
                image_part=image_part,
                slide_title=slide_title,
                ocr_source="pptx_image" if image_metadata.get("ocr_text") else None,
                **image_metadata,
            )
            if element:
                elements.append(element)
            ocr_element = _image_ocr_text_element(
                image_metadata,
                location=f"slide {slide_number} image {picture_index} OCR",
                image_part=image_part,
                ocr_source="pptx_image",
                base_metadata={
                    **picture_geometry,
                    **picture_group_context,
                    "slide": slide_number,
                    "slide_title": slide_title,
                    "image": picture_index,
                    "title": title,
                    "description": description,
                    "name": name,
                    "relationship_id": relationship_id,
                    "target_mode": relationship.get("target_mode"),
                },
            )
            if ocr_element:
                elements.append(ocr_element)

            for link in _drawing_hyperlink_elements(
                picture,
                slide_relationships,
                location_prefix=f"slide {slide_number} image {picture_index}",
                base_metadata={
                    **picture_geometry,
                    **picture_group_context,
                    "slide": slide_number,
                    "slide_title": slide_title,
                    "image": picture_index,
                    "title": title,
                    "description": description,
                    "name": name,
                    **_image_link_metadata(
                        image_metadata,
                        image_part=image_part,
                        image_relationship_id=relationship_id,
                        image_relationship=relationship,
                    ),
                },
                label_context=title or description or name or image_part,
            ):
                link_key = (
                    link.metadata.get("relationship_id"),
                    link.metadata.get("uri"),
                    link.metadata.get("link_kind"),
                    link.text,
                )
                if link_key in seen_slide_links:
                    continue
                seen_slide_links.add(link_key)
                elements.append(link)

        for relationship_id, relationship in slide_relationships.items():
            diagram_part = relationship.get("target") or ""
            if not diagram_part:
                continue
            relationship_type = relationship.get("type", "")
            if not (
                relationship_type.endswith("/diagramData")
                or "/diagrams/data" in diagram_part
            ):
                continue
            seen_diagram_parts.add(diagram_part)
            elements.extend(_drawingml_diagram_elements(
                content,
                diagram_part,
                slide_number=slide_number,
                slide_title=slide_title,
                relationship_id=relationship_id,
            ))

    for chart_index, chart_name in enumerate(_zip_names(content, "ppt/charts/chart"), start=1):
        if chart_name in seen_chart_parts:
            continue
        root = _read_zip_xml(content, chart_name)
        if root is None:
            continue
        elements.extend(_chart_elements(
            root,
            chart_index=chart_index,
            location_prefix="package chart",
            base_metadata={"chart_part": chart_name, **_chart_source_metadata(content, chart_name, root)},
        ))
        elements.extend(_chart_source_child_elements(
            content,
            chart_name,
            root,
            location_prefix=f"package chart {chart_index}",
            base_metadata={"chart": chart_index, "chart_part": chart_name},
        ))

    for diagram_name in _zip_names(content, "ppt/diagrams/data"):
        if diagram_name in seen_diagram_parts:
            continue
        elements.extend(_drawingml_diagram_elements(
            content,
            diagram_name,
            slide_number=None,
            slide_title=None,
        ))

    package_comment_parts = [
        *_zip_names(content, "ppt/comments/comment"),
        *_zip_names(content, "ppt/threadedComments/threadedComment"),
    ]
    for comment_part in package_comment_parts:
        if comment_part in seen_comment_parts:
            continue
        elements.extend(_pptx_comment_elements(
            content,
            comment_part,
            author_map=comment_authors,
            slide_size_metadata=slide_size_metadata,
        ))

    for notes_name in _zip_names(content, "ppt/notesSlides/notesSlide"):
        if notes_name in seen_notes_parts:
            continue
        notes_number = _first_int(notes_name)
        root = _read_zip_xml(content, notes_name)
        if root is None:
            continue
        fallback_metadata = dict(slide_metadata_by_index.get(int(notes_number or 0), {}))
        fallback_metadata.setdefault("slide", notes_number)
        element = _as_element(
            "speaker_notes",
            _xml_text(root, ("t",)),
            location=f"notes slide {notes_number}",
            **fallback_metadata,
            notes_part=notes_name,
        )
        if element:
            elements.append(element)

    return elements


def _xlsx_sheet_names(content: bytes) -> list[str]:
    workbook_root = _read_zip_xml(content, "xl/workbook.xml")
    if workbook_root is None:
        return []
    return [
        _clean_text(sheet.attrib.get("name", ""))
        for sheet in workbook_root.findall(".//{*}sheet")
        if _clean_text(sheet.attrib.get("name", ""))
    ]


def _xlsx_sheet_parts(content: bytes) -> list[dict[str, Any]]:
    workbook_root = _read_zip_xml(content, "xl/workbook.xml")
    if workbook_root is None:
        return []

    workbook_relationships = _part_relationships(content, "xl/workbook.xml")
    sheet_parts = []
    for sheet_index, sheet in enumerate(workbook_root.findall(".//{*}sheet"), start=1):
        rel_id = sheet.attrib.get(f"{{{XML_NS['r']}}}id")
        target = workbook_relationships.get(rel_id or "", {}).get("target") or f"xl/worksheets/sheet{sheet_index}.xml"
        sheet_state = _clean_text(str(sheet.attrib.get("state") or "visible")) or "visible"
        sheet_parts.append({
            "index": sheet_index,
            "name": _clean_text(sheet.attrib.get("name", "")) or f"Sheet {sheet_index}",
            "part": target,
            "relationship_id": rel_id,
            "sheet_id": _clean_text(str(sheet.attrib.get("sheetId") or "")),
            "state": sheet_state,
            "visibility": sheet_state,
        })
    return sheet_parts


def _xlsx_sheet_metadata(sheet_info: dict[str, Any]) -> dict[str, Any]:
    sheet_index = int(sheet_info.get("index") or 0)
    sheet_name = _clean_text(str(sheet_info.get("name") or f"Sheet {sheet_index or ''}"))
    sheet_state = _clean_text(str(sheet_info.get("state") or sheet_info.get("visibility") or "visible")) or "visible"
    return {
        "sheet": sheet_index,
        "sheet_name": sheet_name,
        "sheet_id": _clean_text(str(sheet_info.get("sheet_id") or "")),
        "sheet_part": _clean_text(str(sheet_info.get("part") or "")),
        "sheet_relationship_id": _clean_text(str(sheet_info.get("relationship_id") or "")),
        "sheet_state": sheet_state,
        "sheet_visibility": sheet_state,
    }


def _xlsx_sheet_metadata_element(sheet_metadata: dict[str, Any]) -> DocumentElement | None:
    sheet_name = _clean_text(str(sheet_metadata.get("sheet_name") or ""))
    if not sheet_name:
        return None
    visibility = _clean_text(str(sheet_metadata.get("sheet_visibility") or "visible")) or "visible"
    sheet_part = _clean_text(str(sheet_metadata.get("sheet_part") or ""))
    text_parts = [
        f"Sheet {sheet_name}",
        f"visibility: {visibility}",
        f"part: {sheet_part}" if sheet_part else "",
    ]
    return _as_element(
        "sheet_metadata",
        "; ".join(part for part in text_parts if part),
        location=f"workbook sheet {sheet_metadata.get('sheet') or sheet_name}",
        **sheet_metadata,
    )


def _apply_xlsx_sheet_metadata(elements: list[DocumentElement], sheet_metadata: dict[str, Any]) -> list[DocumentElement]:
    for element in elements:
        for key, value in sheet_metadata.items():
            if value not in (None, ""):
                element.metadata.setdefault(key, value)
    return elements


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> tuple[str, str | None]:
    formula_node = cell.find("{*}f")
    formula = _xml_text(formula_node, ("f",)) if formula_node is not None else None
    raw_value = _xml_text(cell, ("v", "t"))
    cell_type = cell.attrib.get("t")

    if cell_type == "s" and raw_value.isdigit():
        index = int(raw_value)
        raw_value = shared_strings[index] if index < len(shared_strings) else raw_value
    elif cell_type == "inlineStr":
        raw_value = _xml_text(cell, ("t",))

    if formula and raw_value:
        return f"{raw_value} (formula: {formula})", formula
    if formula:
        return f"formula: {formula}", formula
    return raw_value, None


def _xlsx_hyperlink_elements(
    root: ET.Element,
    relationships: dict[str, dict[str, str]],
    *,
    sheet_index: int,
    sheet_name: str,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    for link_index, hyperlink in enumerate(root.findall(".//{*}hyperlink"), start=1):
        ref = hyperlink.attrib.get("ref")
        relationship_id = _attr_by_local_name(hyperlink, "id")
        relationship = relationships.get(relationship_id or "", {})
        location = hyperlink.attrib.get("location")
        target = relationship.get("target") or location
        display = _clean_text(hyperlink.attrib.get("display") or hyperlink.attrib.get("tooltip") or "")
        label = " - ".join(part for part in (display, target) if part) or str(ref or "Spreadsheet link")
        element = _as_element(
            "link",
            label,
            location=f"{sheet_name} {ref or 'cell'} hyperlink {link_index}",
            sheet=sheet_index,
            sheet_name=sheet_name,
            cell=ref,
            relationship_id=relationship_id,
            target_mode=relationship.get("target_mode"),
            uri=target,
            link_kind="spreadsheet_hyperlink",
            display=display,
        )
        if element:
            elements.append(element)
    return elements


def _spreadsheet_cell_column(cell_ref: str | None) -> str:
    match = re.match(r"\$?([A-Za-z]+)", str(cell_ref or ""))
    return match.group(1).upper() if match else ""


def _spreadsheet_normalize_cell_ref(cell_ref: str | None) -> str:
    match = re.match(r"\$?([A-Za-z]+)\$?(\d+)", str(cell_ref or ""))
    return f"{match.group(1).upper()}{match.group(2)}" if match else _clean_text(str(cell_ref or ""))


def _spreadsheet_column_index(column: str | None) -> int | None:
    cleaned = _spreadsheet_cell_column(column)
    if not cleaned:
        return None
    index = 0
    for char in cleaned:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index


def _spreadsheet_cell_row(cell_ref: str | None) -> int | None:
    match = re.search(r"(\d+)", str(cell_ref or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _spreadsheet_range_shape(range_ref: str | None) -> dict[str, Any]:
    cleaned = _clean_text(str(range_ref or ""))
    if not cleaned:
        return {}
    start_ref, _, end_ref = cleaned.partition(":")
    start_ref = _spreadsheet_normalize_cell_ref(start_ref)
    end_ref = _spreadsheet_normalize_cell_ref(end_ref or start_ref)
    start_column = _spreadsheet_column_index(start_ref)
    end_column = _spreadsheet_column_index(end_ref)
    start_row = _spreadsheet_cell_row(start_ref)
    end_row = _spreadsheet_cell_row(end_ref)
    if not all(value is not None for value in (start_column, end_column, start_row, end_row)):
        return {"range_ref": cleaned, "anchor_cell": start_ref}
    column_count = abs(int(end_column) - int(start_column)) + 1
    row_count = abs(int(end_row) - int(start_row)) + 1
    return {
        "range_ref": cleaned,
        "anchor_cell": start_ref,
        "row_count": row_count,
        "column_count": column_count,
        "cell_count": row_count * column_count,
    }


def _xlsx_comment_anchor_metadata(
    *,
    sheet_name: str,
    cell_ref: str | None,
    cell_values: dict[str, str],
    column_headers: dict[str, str],
) -> dict[str, Any]:
    cleaned_ref = _spreadsheet_normalize_cell_ref(cell_ref)
    column = _spreadsheet_cell_column(cleaned_ref)
    value = _clean_text(str(cell_values.get(cleaned_ref.upper()) or ""))
    header = _clean_text(str(column_headers.get(column) or ""))
    anchor_text = _clean_text("; ".join(part for part in [header, value] if part)) or cleaned_ref
    return {
        "comment_anchor_text": anchor_text,
        "comment_anchor_sheet": sheet_name,
        "comment_anchor_cell": cleaned_ref,
        "comment_anchor_header": header,
        "comment_anchor_value": value,
    }


def _xlsx_merged_cell_elements(
    root: ET.Element,
    *,
    sheet_index: int,
    sheet_name: str,
    cell_values: dict[str, str],
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    for merge_index, merge_cell in enumerate(root.findall(".//{*}mergeCell"), start=1):
        range_ref = _clean_text(str(merge_cell.attrib.get("ref") or ""))
        range_shape = _spreadsheet_range_shape(range_ref)
        anchor_cell = _clean_text(str(range_shape.get("anchor_cell") or ""))
        value = _clean_text(str(cell_values.get(anchor_cell.upper()) or ""))
        text = value or f"Merged range {range_ref}"
        element = _as_element(
            "heading",
            text,
            location=f"{sheet_name} merged range {range_ref or merge_index}",
            sheet=sheet_index,
            sheet_name=sheet_name,
            cell=anchor_cell,
            level=1,
            heading_kind="spreadsheet_merged_cell",
            merge_range=range_ref,
            merge_anchor_cell=anchor_cell,
            merge_row_count=range_shape.get("row_count"),
            merge_column_count=range_shape.get("column_count"),
            merge_cell_count=range_shape.get("cell_count"),
        )
        if element:
            elements.append(element)
    return elements


def _xlsx_child_text(node: ET.Element, local_name: str) -> str:
    for child in node:
        if _local_name(child.tag) == local_name:
            return _clean_text(" ".join(child.itertext()))
    return ""


def _xlsx_validation_display_value(validation_type: str, formula1: str, formula2: str) -> str:
    cleaned_formula1 = _clean_text(formula1)
    cleaned_formula2 = _clean_text(formula2)
    if validation_type == "list" and len(cleaned_formula1) >= 2 and cleaned_formula1.startswith('"') and cleaned_formula1.endswith('"'):
        choices = [
            _clean_text(part)
            for part in cleaned_formula1.strip('"').split(",")
            if _clean_text(part)
        ]
        if choices:
            return ", ".join(choices)
    if cleaned_formula1 and cleaned_formula2:
        return f"{cleaned_formula1} to {cleaned_formula2}"
    return cleaned_formula1 or cleaned_formula2 or validation_type


def _xlsx_data_validation_elements(
    root: ET.Element,
    *,
    sheet_index: int,
    sheet_name: str,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    for validation_index, validation in enumerate(root.findall(".//{*}dataValidation"), start=1):
        validation_range = _clean_text(str(validation.attrib.get("sqref") or ""))
        validation_type = _clean_text(str(validation.attrib.get("type") or "any"))
        operator = _clean_text(str(validation.attrib.get("operator") or ""))
        formula1 = _xlsx_child_text(validation, "formula1")
        formula2 = _xlsx_child_text(validation, "formula2")
        prompt_title = _xlsx_child_text(validation, "promptTitle") or _clean_text(str(validation.attrib.get("promptTitle") or ""))
        prompt = _xlsx_child_text(validation, "prompt") or _clean_text(str(validation.attrib.get("prompt") or ""))
        error_title = _xlsx_child_text(validation, "errorTitle") or _clean_text(str(validation.attrib.get("errorTitle") or ""))
        error = _xlsx_child_text(validation, "error") or _clean_text(str(validation.attrib.get("error") or ""))
        value = _xlsx_validation_display_value(validation_type, formula1, formula2)
        label = prompt_title or f"{sheet_name} {validation_range or validation_index} validation"
        detail_parts = [
            label,
            f"type: {validation_type}" if validation_type else "",
            f"range: {validation_range}" if validation_range else "",
            f"value: {value}" if value else "",
            prompt,
            error_title,
            error,
        ]
        element = _as_element(
            "form_field",
            "; ".join(part for part in detail_parts if part),
            location=f"{sheet_name} {validation_range or 'range'} validation {validation_index}",
            sheet=sheet_index,
            sheet_name=sheet_name,
            cell=validation_range,
            label=label,
            value=value,
            form_kind="spreadsheet_data_validation",
            field_type=validation_type,
            validation_range=validation_range,
            validation_type=validation_type,
            validation_operator=operator,
            validation_formula1=formula1,
            validation_formula2=formula2,
            validation_allow_blank=validation.attrib.get("allowBlank"),
            validation_prompt_title=prompt_title,
            validation_prompt=prompt,
            validation_error_title=error_title,
            validation_error=error,
        )
        if element:
            elements.append(element)
    return elements


def _xlsx_formula_result(display_value: str, formula: str) -> str:
    cleaned = _clean_text(display_value)
    suffix = f" (formula: {formula})"
    if cleaned.endswith(suffix):
        return _clean_text(cleaned[: -len(suffix)])
    if cleaned == f"formula: {formula}":
        return ""
    return cleaned


def _normalize_formula_reference(reference: str, default_sheet_name: str) -> str:
    cleaned = _clean_text(reference).replace("$", "")
    if "!" not in cleaned:
        return f"{default_sheet_name}!{cleaned}" if default_sheet_name else cleaned
    sheet_name, cell_ref = cleaned.split("!", 1)
    cleaned_sheet_name = sheet_name.strip("'")
    return f"{cleaned_sheet_name}!{cell_ref}"


def _xlsx_formula_references(
    formula: str,
    *,
    sheet_name: str,
    known_named_ranges: set[str],
) -> list[str]:
    references: list[str] = []
    seen: set[str] = set()

    def add(reference: str) -> None:
        cleaned = _clean_text(reference)
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            references.append(cleaned)

    cell_ref_pattern = re.compile(
        r"(?<![A-Za-z0-9_])(?:(?:'([^']+)'|([A-Za-z_][A-Za-z0-9_ .]*))!)?(\$?[A-Z]{1,3}\$?\d+(?::\$?[A-Z]{1,3}\$?\d+)?)"
    )
    for match in cell_ref_pattern.finditer(formula or ""):
        explicit_sheet = _clean_text(str(match.group(1) or match.group(2) or sheet_name))
        cell_ref = _clean_text(str(match.group(3) or ""))
        add(_normalize_formula_reference(f"{explicit_sheet}!{cell_ref}" if explicit_sheet else cell_ref, sheet_name))

    known_names = {name.lower(): name for name in known_named_ranges if _clean_text(name)}
    for token_match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_.]*\b", formula or ""):
        token = token_match.group(0)
        if token.lower() not in known_names:
            continue
        add(known_names[token.lower()])

    return references[:24]


def _xlsx_formula_element(
    *,
    sheet_index: int,
    sheet_name: str,
    cell_ref: str,
    formula: str,
    display_value: str,
    known_named_ranges: set[str],
) -> DocumentElement | None:
    normalized_cell = _spreadsheet_normalize_cell_ref(cell_ref)
    formula_result = _xlsx_formula_result(display_value, formula)
    references = _xlsx_formula_references(
        formula,
        sheet_name=sheet_name,
        known_named_ranges=known_named_ranges,
    )
    cell_label = f"{sheet_name}!{normalized_cell}" if sheet_name and normalized_cell else normalized_cell
    text_parts = [
        f"Formula {cell_label}",
        f"expression: {formula}",
        f"result: {formula_result}" if formula_result else "",
        f"references: {', '.join(references)}" if references else "",
    ]
    return _as_element(
        "formula",
        "; ".join(part for part in text_parts if part),
        location=f"{sheet_name} {normalized_cell} formula",
        sheet=sheet_index,
        sheet_name=sheet_name,
        cell=normalized_cell,
        formula_cell=cell_label,
        formula_expression=formula,
        formula_result=formula_result,
        formula_references=references,
        formula_reference_count=len(references),
    )


def _xlsx_conditional_formatting_elements(
    root: ET.Element,
    *,
    sheet_index: int,
    sheet_name: str,
    known_named_ranges: set[str],
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    for conditional_index, conditional in enumerate(root.findall(".//{*}conditionalFormatting"), start=1):
        conditional_range = _clean_text(str(conditional.attrib.get("sqref") or ""))
        range_label = f"{sheet_name}!{conditional_range}" if sheet_name and conditional_range else conditional_range
        for rule_index, rule in enumerate(conditional.findall("{*}cfRule"), start=1):
            rule_type = _clean_text(str(rule.attrib.get("type") or ""))
            operator = _clean_text(str(rule.attrib.get("operator") or ""))
            priority = _clean_text(str(rule.attrib.get("priority") or ""))
            dxf_id = _clean_text(str(rule.attrib.get("dxfId") or ""))
            stop_if_true = _clean_text(str(rule.attrib.get("stopIfTrue") or ""))
            text = _clean_text(str(rule.attrib.get("text") or ""))
            time_period = _clean_text(str(rule.attrib.get("timePeriod") or ""))
            rank = _clean_text(str(rule.attrib.get("rank") or ""))
            percent = _clean_text(str(rule.attrib.get("percent") or ""))
            formulas = [
                _clean_text(" ".join(formula.itertext()))
                for formula in rule.findall("{*}formula")
                if _clean_text(" ".join(formula.itertext()))
            ]
            formula_references: list[str] = []
            for formula in formulas:
                for reference in _xlsx_formula_references(
                    formula,
                    sheet_name=sheet_name,
                    known_named_ranges=known_named_ranges,
                ):
                    if reference not in formula_references:
                        formula_references.append(reference)
            condition_parts = [
                rule_type,
                operator,
                " ".join(formulas),
                text,
                time_period,
                f"rank {rank}" if rank else "",
            ]
            condition = _clean_text(" ".join(part for part in condition_parts if part)) or "conditional format"
            conditional_label = _clip(f"{range_label} {condition}", 160)
            color_scale = []
            for color in rule.findall(".//{*}color"):
                rgb = _clean_text(str(color.attrib.get("rgb") or ""))
                if rgb:
                    color_scale.append(rgb)
            element = _as_element(
                "conditional_format",
                f"Conditional format {conditional_label}",
                location=f"{sheet_name} {conditional_range or 'range'} conditional format {conditional_index}.{rule_index}",
                sheet=sheet_index,
                sheet_name=sheet_name,
                conditional_label=conditional_label,
                conditional_range=conditional_range,
                conditional_range_label=range_label,
                conditional_type=rule_type,
                conditional_operator=operator,
                conditional_priority=priority,
                conditional_dxf_id=dxf_id,
                conditional_stop_if_true=stop_if_true,
                conditional_text=text,
                conditional_time_period=time_period,
                conditional_rank=rank,
                conditional_percent=percent,
                conditional_formula1=formulas[0] if formulas else None,
                conditional_formula2=formulas[1] if len(formulas) > 1 else None,
                conditional_formulas=formulas,
                conditional_formula_references=formula_references,
                conditional_color_scale=color_scale or None,
            )
            if element:
                elements.append(element)
    return elements


def _xlsx_range_sheet_name(range_ref: str | None) -> str:
    cleaned = _clean_text(str(range_ref or ""))
    if "!" not in cleaned:
        return ""
    sheet_name, _ = cleaned.split("!", 1)
    return _clean_text(sheet_name.strip("'"))


def _xlsx_defined_name_elements(content: bytes, sheet_parts: list[dict[str, Any]]) -> list[DocumentElement]:
    workbook_root = _read_zip_xml(content, "xl/workbook.xml")
    if workbook_root is None:
        return []
    sheet_names_by_local_id = {
        str(index - 1): _clean_text(str(sheet.get("name") or ""))
        for index, sheet in enumerate(sheet_parts, start=1)
    }
    elements: list[DocumentElement] = []
    for name_index, defined_name in enumerate(workbook_root.findall(".//{*}definedName"), start=1):
        name = _clean_text(str(defined_name.attrib.get("name") or ""))
        reference = _clean_text(" ".join(defined_name.itertext()))
        if not name or not reference:
            continue
        local_sheet_id = _clean_text(str(defined_name.attrib.get("localSheetId") or ""))
        sheet_name = sheet_names_by_local_id.get(local_sheet_id, "") or _xlsx_range_sheet_name(reference)
        scope = sheet_name or "workbook"
        hidden = defined_name.attrib.get("hidden")
        comment = _clean_text(str(defined_name.attrib.get("comment") or ""))
        text_parts = [
            f"Named range {name}",
            f"scope: {scope}",
            f"reference: {reference}",
            comment,
        ]
        element = _as_element(
            "named_range",
            "; ".join(part for part in text_parts if part),
            location=f"{scope} named range {name_index}",
            sheet_name=sheet_name,
            named_range_name=name,
            named_range_ref=reference,
            named_range_scope=scope,
            named_range_local_sheet_id=local_sheet_id,
            named_range_hidden=hidden,
            named_range_comment=comment,
        )
        if element:
            elements.append(element)
    return elements


def _xlsx_table_definition_elements(
    content: bytes,
    relationships: dict[str, dict[str, str]],
    *,
    sheet_index: int,
    sheet_name: str,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    table_relationships = [
        rel
        for rel in relationships.values()
        if rel.get("target") and (
            rel.get("type", "").endswith("/table")
            or "/tables/table" in rel.get("target", "")
        )
    ]
    for table_index, relationship in enumerate(table_relationships, start=1):
        table_part = relationship.get("target", "")
        table_root = _read_zip_xml(content, table_part)
        if table_root is None:
            continue
        display_name = _clean_text(str(
            table_root.attrib.get("displayName")
            or table_root.attrib.get("name")
            or f"Table {table_index}"
        ))
        table_name = _clean_text(str(table_root.attrib.get("name") or display_name))
        table_range = _clean_text(str(table_root.attrib.get("ref") or ""))
        columns = [
            _clean_text(str(column.attrib.get("name") or ""))
            for column in table_root.findall(".//{*}tableColumn")
            if _clean_text(str(column.attrib.get("name") or ""))
        ]
        auto_filter = table_root.find(".//{*}autoFilter")
        auto_filter_range = _clean_text(str(auto_filter.attrib.get("ref") or "")) if auto_filter is not None else ""
        text_parts = [
            f"Spreadsheet table {display_name}",
            f"range: {table_range}" if table_range else "",
            f"columns: {', '.join(columns)}" if columns else "",
        ]
        range_shape = _spreadsheet_range_shape(table_range)
        element = _as_element(
            "table_definition",
            "; ".join(part for part in text_parts if part),
            location=f"{sheet_name} table definition {display_name}",
            sheet=sheet_index,
            sheet_name=sheet_name,
            table=display_name,
            table_name=display_name,
            table_display_name=display_name,
            table_definition_name=table_name,
            table_part=table_part,
            table_range=table_range,
            table_columns=columns,
            table_column_count=len(columns) if columns else None,
            table_row_count=range_shape.get("row_count"),
            table_relationship_id=relationship.get("id"),
            auto_filter_range=auto_filter_range,
        )
        if element:
            elements.append(element)
    return elements


def _xlsx_pivot_cache_metadata(content: bytes) -> dict[str, dict[str, Any]]:
    workbook_root = _read_zip_xml(content, "xl/workbook.xml")
    if workbook_root is None:
        return {}
    workbook_relationships = _part_relationships(content, "xl/workbook.xml")
    caches: dict[str, dict[str, Any]] = {}
    for pivot_cache in workbook_root.findall(".//{*}pivotCache"):
        cache_id = _clean_text(str(pivot_cache.attrib.get("cacheId") or ""))
        if not cache_id:
            continue
        relationship_id = _attr_by_local_name(pivot_cache, "id")
        relationship = workbook_relationships.get(relationship_id or "", {})
        cache_part = relationship.get("target") or ""
        metadata: dict[str, Any] = {
            "cache_id": cache_id,
            "cache_relationship_id": relationship_id,
            "cache_relationship_type": relationship.get("type"),
            "cache_part": cache_part,
        }
        cache_root = _read_zip_xml(content, cache_part) if cache_part else None
        if cache_root is not None:
            cache_source = cache_root.find(".//{*}cacheSource")
            if cache_source is not None:
                metadata["source_type"] = _clean_text(str(cache_source.attrib.get("type") or ""))
                worksheet_source = cache_source.find(".//{*}worksheetSource")
                if worksheet_source is not None:
                    metadata["source_ref"] = _clean_text(str(worksheet_source.attrib.get("ref") or ""))
                    metadata["source_name"] = _clean_text(str(worksheet_source.attrib.get("name") or ""))
                    metadata["source_sheet"] = _clean_text(str(worksheet_source.attrib.get("sheet") or ""))
            fields = [
                _clean_text(str(cache_field.attrib.get("name") or ""))
                for cache_field in cache_root.findall(".//{*}cacheField")
                if _clean_text(str(cache_field.attrib.get("name") or ""))
            ]
            if fields:
                metadata["cache_fields"] = fields
            record_count = _clean_text(str(cache_root.attrib.get("recordCount") or ""))
            if record_count:
                metadata["record_count"] = record_count
            refreshed_by = _clean_text(str(cache_root.attrib.get("refreshedBy") or ""))
            if refreshed_by:
                metadata["refreshed_by"] = refreshed_by
        caches[cache_id] = {key: value for key, value in metadata.items() if value not in (None, "", [])}
    return caches


def _xlsx_pivot_field_names(parent: ET.Element | None, field_names: list[str]) -> list[str]:
    if parent is None:
        return []
    names: list[str] = []
    for field in list(parent):
        if _local_name(field.tag) not in {"field", "pageField"}:
            continue
        raw_index = _clean_text(str(field.attrib.get("x") or field.attrib.get("fld") or ""))
        field_name = ""
        try:
            index = int(raw_index)
            if 0 <= index < len(field_names):
                field_name = field_names[index]
        except (TypeError, ValueError):
            field_name = raw_index
        field_name = _clean_text(field_name)
        if field_name and field_name not in names:
            names.append(field_name)
    return names


def _xlsx_pivot_data_fields(root: ET.Element, field_names: list[str]) -> list[dict[str, str]]:
    data_fields: list[dict[str, str]] = []
    for data_field in root.findall(".//{*}dataField"):
        raw_index = _clean_text(str(data_field.attrib.get("fld") or ""))
        source_field = ""
        try:
            index = int(raw_index)
            if 0 <= index < len(field_names):
                source_field = field_names[index]
        except (TypeError, ValueError):
            source_field = raw_index
        display_name = _clean_text(str(data_field.attrib.get("name") or source_field))
        subtotal = _clean_text(str(data_field.attrib.get("subtotal") or ""))
        field_info = {
            "name": display_name,
            "source_field": _clean_text(source_field),
            "subtotal": subtotal,
        }
        data_fields.append({key: value for key, value in field_info.items() if value})
    return data_fields


def _xlsx_pivot_table_elements(
    content: bytes,
    relationships: dict[str, dict[str, str]],
    pivot_caches: dict[str, dict[str, Any]],
    *,
    sheet_index: int,
    sheet_name: str,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    pivot_relationships = [
        rel
        for rel in relationships.values()
        if rel.get("target") and (
            rel.get("type", "").endswith("/pivotTable")
            or "/pivotTables/pivotTable" in rel.get("target", "")
        )
    ]
    for pivot_index, relationship in enumerate(pivot_relationships, start=1):
        pivot_part = relationship.get("target", "")
        pivot_root = _read_zip_xml(content, pivot_part)
        if pivot_root is None:
            continue
        pivot_name = _clean_text(str(pivot_root.attrib.get("name") or f"Pivot Table {pivot_index}"))
        cache_id = _clean_text(str(pivot_root.attrib.get("cacheId") or ""))
        cache_metadata = pivot_caches.get(cache_id, {})
        field_names = [
            _clean_text(str(field))
            for field in cache_metadata.get("cache_fields", [])
            if _clean_text(str(field))
        ]
        location = pivot_root.find(".//{*}location")
        pivot_range = _clean_text(str(location.attrib.get("ref") or "")) if location is not None else ""
        row_fields = _xlsx_pivot_field_names(pivot_root.find(".//{*}rowFields"), field_names)
        column_fields = _xlsx_pivot_field_names(pivot_root.find(".//{*}colFields"), field_names)
        page_fields = _xlsx_pivot_field_names(pivot_root.find(".//{*}pageFields"), field_names)
        data_fields = _xlsx_pivot_data_fields(pivot_root, field_names)
        data_field_labels = [
            _clean_text(str(data_field.get("name") or data_field.get("source_field") or ""))
            for data_field in data_fields
            if _clean_text(str(data_field.get("name") or data_field.get("source_field") or ""))
        ]
        source_name = _clean_text(str(cache_metadata.get("source_name") or ""))
        source_ref = _clean_text(str(cache_metadata.get("source_ref") or ""))
        source_sheet = _clean_text(str(cache_metadata.get("source_sheet") or ""))
        source_label = source_name or _clean_text("!".join(part for part in [source_sheet, source_ref] if part))
        text_parts = [
            f"Pivot table {pivot_name}",
            f"range: {pivot_range}" if pivot_range else "",
            f"source: {source_label}" if source_label else "",
            f"rows: {', '.join(row_fields)}" if row_fields else "",
            f"columns: {', '.join(column_fields)}" if column_fields else "",
            f"filters: {', '.join(page_fields)}" if page_fields else "",
            f"values: {', '.join(data_field_labels)}" if data_field_labels else "",
        ]
        element = _as_element(
            "pivot_table",
            "; ".join(part for part in text_parts if part),
            location=f"{sheet_name} pivot table {pivot_name}",
            sheet=sheet_index,
            sheet_name=sheet_name,
            pivot_table_name=pivot_name,
            pivot_table_cache_id=cache_id,
            pivot_table_part=pivot_part,
            pivot_table_range=pivot_range,
            pivot_table_relationship_id=relationship.get("id"),
            pivot_table_source_type=cache_metadata.get("source_type"),
            pivot_table_source_name=source_name,
            pivot_table_source_ref=source_ref,
            pivot_table_source_sheet=source_sheet,
            pivot_table_cache_part=cache_metadata.get("cache_part"),
            pivot_table_cache_relationship_id=cache_metadata.get("cache_relationship_id"),
            pivot_table_cache_fields=field_names,
            pivot_table_row_fields=row_fields,
            pivot_table_column_fields=column_fields,
            pivot_table_page_fields=page_fields,
            pivot_table_data_fields=data_fields,
            pivot_table_data_field_count=len(data_fields) if data_fields else None,
            pivot_table_record_count=cache_metadata.get("record_count"),
        )
        if element:
            elements.append(element)
    return elements


def _xlsx_threaded_comment_authors(content: bytes) -> dict[str, str]:
    authors: dict[str, str] = {}
    for persons_name in _zip_names(content, "xl/persons/person"):
        root = _read_zip_xml(content, persons_name)
        if root is None:
            continue
        for person in root.iter():
            if _local_name(person.tag) != "person":
                continue
            person_id = _attr_by_local_name(person, "id") or _attr_by_local_name(person, "personId")
            display_name = (
                _attr_by_local_name(person, "displayName")
                or _attr_by_local_name(person, "name")
                or _attr_by_local_name(person, "userId")
            )
            cleaned_name = _clean_text(str(display_name or ""))
            if person_id and cleaned_name:
                authors[person_id] = cleaned_name
    return authors


def _is_xlsx_threaded_comments_relationship(relationship: dict[str, str]) -> bool:
    target = relationship.get("target", "")
    relationship_type = relationship.get("type", "").lower()
    return bool(
        target
        and (
            "/threadedcomments/" in target.lower()
            or relationship_type.endswith("/threadedcomment")
            or relationship_type.endswith("/threadedcomments")
        )
    )


def _xlsx_threaded_comment_elements(
    root: ET.Element,
    *,
    sheet_index: int | None,
    sheet_name: str,
    comment_part: str,
    authors: dict[str, str],
    cell_values: dict[str, str],
    column_headers: dict[str, str],
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    comments = [
        node
        for node in root.iter()
        if _local_name(node.tag) in {"threadedComment", "threadedCommentReply"}
    ]
    for comment_index, comment in enumerate(comments, start=1):
        ref = _attr_by_local_name(comment, "ref")
        person_id = _attr_by_local_name(comment, "personId")
        comment_id = _attr_by_local_name(comment, "id")
        parent_id = _attr_by_local_name(comment, "parentId")
        comment_datetime = (
            _attr_by_local_name(comment, "dT")
            or _attr_by_local_name(comment, "dateTime")
            or _attr_by_local_name(comment, "dt")
        )
        author = authors.get(person_id or "") or _clean_text(str(_attr_by_local_name(comment, "author") or ""))
        text = _xml_text(comment, ("t", "text"))
        element = _as_element(
            "comment",
            f"{author}: {text}" if author else text,
            location=f"{sheet_name} {ref or 'cell'} threaded comment {comment_index}" if sheet_name else f"{Path(comment_part).name} threaded comment {comment_index}",
            sheet=sheet_index,
            sheet_name=sheet_name,
            cell=ref,
            author=author,
            comment_id=comment_id,
            comment_kind="threaded_comment",
            comment_part=comment_part,
            threaded_comment_id=comment_id,
            threaded_parent_id=parent_id,
            person_id=person_id,
            comment_datetime=comment_datetime,
            **_xlsx_comment_anchor_metadata(
                sheet_name=sheet_name,
                cell_ref=ref,
                cell_values=cell_values,
                column_headers=column_headers,
            ),
        )
        if element:
            elements.append(element)
    return elements


def _xlsx_drawing_image_elements(
    content: bytes,
    drawing_root: ET.Element | None,
    drawing_relationships: dict[str, dict[str, str]],
    *,
    sheet_index: int,
    sheet_name: str,
    drawing_part: str,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    if drawing_root is None:
        return elements

    for image_index, picture in enumerate(drawing_root.findall(".//{*}pic"), start=1):
        picture_properties = picture.find(".//{*}cNvPr")
        title = _drawing_attr(picture_properties, "title")
        description = _drawing_attr(picture_properties, "descr")
        name = _drawing_attr(picture_properties, "name")
        blip = picture.find(".//{*}blip")
        relationship_id = None
        if blip is not None:
            relationship_id = (
                blip.attrib.get(f"{{{XML_NS['r']}}}embed")
                or blip.attrib.get(f"{{{XML_NS['r']}}}link")
            )
        relationship = drawing_relationships.get(relationship_id or "", {})
        image_part = relationship.get("target")
        image_metadata = _zip_image_metadata(content, image_part)
        label = _image_element_text(title, description, name, image_part, metadata=image_metadata)
        element = _as_element(
            "image",
            label,
            location=f"{sheet_name} image {image_index}",
            sheet=sheet_index,
            sheet_name=sheet_name,
            image=image_index,
            title=title,
            description=description,
            name=name,
            relationship_id=relationship_id,
            target_mode=relationship.get("target_mode"),
            drawing_part=drawing_part,
            image_part=image_part,
            ocr_source="xlsx_image" if image_metadata.get("ocr_text") else None,
            **image_metadata,
        )
        if element:
            elements.append(element)
        ocr_element = _image_ocr_text_element(
            image_metadata,
            location=f"{sheet_name} image {image_index} OCR",
            image_part=image_part,
            ocr_source="xlsx_image",
            base_metadata={
                "sheet": sheet_index,
                "sheet_name": sheet_name,
                "image": image_index,
                "title": title,
                "description": description,
                "name": name,
                "relationship_id": relationship_id,
                "target_mode": relationship.get("target_mode"),
                "drawing_part": drawing_part,
            },
        )
        if ocr_element:
            elements.append(ocr_element)
    return elements


def _extract_xlsx(content: bytes) -> list[DocumentElement]:
    elements = _office_package_metadata_elements(content)
    shared_strings = []
    shared_root = _read_zip_xml(content, "xl/sharedStrings.xml")
    if shared_root is not None:
        shared_strings = [_xml_text(item, ("t",)) for item in shared_root.findall(".//{*}si")]

    threaded_comment_authors = _xlsx_threaded_comment_authors(content)
    seen_comment_parts: set[str] = set()
    seen_threaded_comment_parts: set[str] = set()
    seen_chart_parts: set[str] = set()
    sheet_parts = _xlsx_sheet_parts(content) or [
        {
            "index": index,
            "name": name,
            "part": sheet_part,
            "relationship_id": None,
            "sheet_id": str(index),
            "state": "visible",
            "visibility": "visible",
        }
        for index, (sheet_part, name) in enumerate(
            zip(_zip_names(content, "xl/worksheets/sheet"), _xlsx_sheet_names(content) or []),
            start=1,
        )
    ]
    if not sheet_parts:
        sheet_parts = [
            {
                "index": index,
                "name": f"Sheet {index}",
                "part": sheet_part,
                "relationship_id": None,
                "sheet_id": str(index),
                "state": "visible",
                "visibility": "visible",
            }
            for index, sheet_part in enumerate(_zip_names(content, "xl/worksheets/sheet"), start=1)
        ]

    defined_name_elements = _xlsx_defined_name_elements(content, sheet_parts)
    known_named_ranges = {
        str(element.metadata.get("named_range_name") or "")
        for element in defined_name_elements
        if element.metadata.get("named_range_name")
    }
    pivot_cache_metadata = _xlsx_pivot_cache_metadata(content)
    elements.extend(defined_name_elements)

    for sheet_info in sheet_parts:
        sheet_index = int(sheet_info["index"])
        sheet_part = str(sheet_info["part"])
        display_sheet_name = str(sheet_info["name"])
        sheet_metadata = _xlsx_sheet_metadata(sheet_info)
        sheet_metadata_element = _xlsx_sheet_metadata_element(sheet_metadata)
        if sheet_metadata_element:
            elements.append(sheet_metadata_element)
        root = _read_zip_xml(content, sheet_part)
        if root is None:
            continue
        rows: list[list[str]] = []
        formulas_by_row: dict[int, list[dict[str, Any]]] = {}
        formula_elements: list[DocumentElement] = []
        cell_values: dict[str, str] = {}
        column_headers: dict[str, str] = {}
        for row in root.findall(".//{*}row"):
            row_number = row.attrib.get("r")
            parsed_row_number = None
            try:
                parsed_row_number = int(row_number or len(rows) + 1)
            except ValueError:
                parsed_row_number = len(rows) + 1
            cells = []
            formulas = []
            for cell in row.findall("{*}c"):
                raw_value, formula = _xlsx_cell_value(cell, shared_strings)
                cell_ref = _clean_text(str(cell.attrib.get("r") or ""))
                if cell_ref:
                    cell_values[cell_ref.upper()] = raw_value
                    if parsed_row_number == 1:
                        column = _spreadsheet_cell_column(cell_ref)
                        if column and raw_value:
                            column_headers[column] = raw_value
                cells.append(raw_value)
                if formula:
                    formula_cell = cell_ref or cell.attrib.get("r") or ""
                    formulas.append({"cell": formula_cell, "formula": formula})
                    formula_element = _xlsx_formula_element(
                        sheet_index=sheet_index,
                        sheet_name=display_sheet_name,
                        cell_ref=formula_cell,
                        formula=formula,
                        display_value=raw_value,
                        known_named_ranges=known_named_ranges,
                    )
                    if formula_element:
                        formula_elements.append(formula_element)
            rows.append(cells)
            if formulas:
                try:
                    formulas_by_row[int(row_number or len(rows))] = formulas
                except ValueError:
                    formulas_by_row[len(rows)] = formulas

        elements.extend(_apply_xlsx_sheet_metadata(formula_elements, sheet_metadata))

        elements.extend(_apply_xlsx_sheet_metadata(_xlsx_merged_cell_elements(
            root,
            sheet_index=sheet_index,
            sheet_name=display_sheet_name,
            cell_values=cell_values,
        ), sheet_metadata))

        sheet_elements = _table_elements_from_rows(
            rows,
            location_prefix=f"sheet {sheet_index}",
            base_metadata=sheet_metadata,
        )
        for element in sheet_elements:
            formulas = formulas_by_row.get(int(element.metadata.get("row", 0) or 0))
            if formulas:
                element.metadata["formulas"] = formulas
            elements.append(element)

        sheet_relationships = _part_relationships(content, sheet_part)
        elements.extend(_apply_xlsx_sheet_metadata(_xlsx_table_definition_elements(
            content,
            sheet_relationships,
            sheet_index=sheet_index,
            sheet_name=display_sheet_name,
        ), sheet_metadata))

        elements.extend(_apply_xlsx_sheet_metadata(_xlsx_pivot_table_elements(
            content,
            sheet_relationships,
            pivot_cache_metadata,
            sheet_index=sheet_index,
            sheet_name=display_sheet_name,
        ), sheet_metadata))

        elements.extend(_apply_xlsx_sheet_metadata(_xlsx_hyperlink_elements(
            root,
            sheet_relationships,
            sheet_index=sheet_index,
            sheet_name=display_sheet_name,
        ), sheet_metadata))

        elements.extend(_apply_xlsx_sheet_metadata(_xlsx_data_validation_elements(
            root,
            sheet_index=sheet_index,
            sheet_name=display_sheet_name,
        ), sheet_metadata))

        elements.extend(_apply_xlsx_sheet_metadata(_xlsx_conditional_formatting_elements(
            root,
            sheet_index=sheet_index,
            sheet_name=display_sheet_name,
            known_named_ranges=known_named_ranges,
        ), sheet_metadata))

        for comment_part in [
            rel["target"]
            for rel in sheet_relationships.values()
            if rel.get("target") and rel.get("type", "").endswith("/comments")
        ]:
            seen_comment_parts.add(comment_part)
            comments_root = _read_zip_xml(content, comment_part)
            if comments_root is None:
                continue
            comment_authors = [
                _xml_text(author, ("t",)) or _clean_text(str(author.text or ""))
                for author in comments_root.findall(".//{*}author")
            ]
            for comment_index, comment in enumerate(comments_root.findall(".//{*}comment"), start=1):
                ref = comment.attrib.get("ref")
                author_id = comment.attrib.get("authorId")
                author = comment_authors[int(author_id)] if str(author_id or "").isdigit() and int(author_id) < len(comment_authors) else None
                text = _xml_text(comment, ("t",))
                element = _as_element(
                    "comment",
                    f"{author}: {text}" if author else text,
                    location=f"{display_sheet_name} {ref or 'cell'} comment {comment_index}",
                    sheet=sheet_index,
                    sheet_name=display_sheet_name,
                    cell=ref,
                    author_id=author_id,
                    author=author,
                    comment_part=comment_part,
                    **_xlsx_comment_anchor_metadata(
                        sheet_name=display_sheet_name,
                        cell_ref=ref,
                        cell_values=cell_values,
                        column_headers=column_headers,
                    ),
                )
                if element:
                    elements.append(element)

        for comment_part in [
            rel["target"]
            for rel in sheet_relationships.values()
            if _is_xlsx_threaded_comments_relationship(rel)
        ]:
            seen_threaded_comment_parts.add(comment_part)
            comments_root = _read_zip_xml(content, comment_part)
            if comments_root is None:
                continue
            elements.extend(_apply_xlsx_sheet_metadata(_xlsx_threaded_comment_elements(
                comments_root,
                sheet_index=sheet_index,
                sheet_name=display_sheet_name,
                comment_part=comment_part,
                authors=threaded_comment_authors,
                cell_values=cell_values,
                column_headers=column_headers,
            ), sheet_metadata))

        for drawing_part in [
            rel["target"]
            for rel in sheet_relationships.values()
            if rel.get("target") and rel.get("type", "").endswith("/drawing")
        ]:
            drawing_relationships = _part_relationships(content, drawing_part)
            drawing_root = _read_zip_xml(content, drawing_part)
            chart_refs = drawing_root.findall(".//c:chart", XML_NS) if drawing_root is not None else []
            if chart_refs:
                for chart_ref_index, chart_ref in enumerate(chart_refs, start=1):
                    relationship_id = chart_ref.attrib.get(f"{{{XML_NS['r']}}}id")
                    chart_part = drawing_relationships.get(relationship_id or "", {}).get("target")
                    chart_root = _read_zip_xml(content, chart_part) if chart_part else None
                    if chart_root is None:
                        continue
                    seen_chart_parts.add(chart_part)
                    chart_index = _first_int(chart_part) or chart_ref_index
                    elements.extend(_apply_xlsx_sheet_metadata(_chart_elements(
                        chart_root,
                        chart_index=chart_index,
                        location_prefix=f"{display_sheet_name} chart",
                        base_metadata={
                            "sheet": sheet_index,
                            "sheet_name": display_sheet_name,
                            "relationship_id": relationship_id,
                            "drawing_part": drawing_part,
                            "chart_part": chart_part,
                            **_chart_source_metadata(content, chart_part, chart_root),
                        },
                    ), sheet_metadata))
            else:
                for rel in drawing_relationships.values():
                    if not rel.get("target") or not rel.get("type", "").endswith("/chart"):
                        continue
                    chart_part = rel["target"]
                    chart_root = _read_zip_xml(content, chart_part)
                    if chart_root is None:
                        continue
                    seen_chart_parts.add(chart_part)
                    chart_index = _first_int(chart_part) or len(seen_chart_parts)
                    elements.extend(_apply_xlsx_sheet_metadata(_chart_elements(
                        chart_root,
                        chart_index=chart_index,
                        location_prefix=f"{display_sheet_name} chart",
                        base_metadata={
                            "sheet": sheet_index,
                            "sheet_name": display_sheet_name,
                            "relationship_id": rel.get("id"),
                            "drawing_part": drawing_part,
                            "chart_part": chart_part,
                            **_chart_source_metadata(content, chart_part, chart_root),
                        },
                    ), sheet_metadata))

            elements.extend(_apply_xlsx_sheet_metadata(_xlsx_drawing_image_elements(
                content,
                drawing_root,
                drawing_relationships,
                sheet_index=sheet_index,
                sheet_name=display_sheet_name,
                drawing_part=drawing_part,
            ), sheet_metadata))

    for comments_name in _zip_names(content, "xl/comments"):
        if comments_name in seen_comment_parts:
            continue
        root = _read_zip_xml(content, comments_name)
        if root is None:
            continue
        comment_authors = [
            _xml_text(author, ("t",)) or _clean_text(str(author.text or ""))
            for author in root.findall(".//{*}author")
        ]
        for comment_index, comment in enumerate(root.findall(".//{*}comment"), start=1):
            ref = comment.attrib.get("ref")
            author_id = comment.attrib.get("authorId")
            author = comment_authors[int(author_id)] if str(author_id or "").isdigit() and int(author_id) < len(comment_authors) else None
            text = _xml_text(comment, ("t",))
            element = _as_element(
                "comment",
                f"{author}: {text}" if author else text,
                location=f"{Path(comments_name).name} comment {comment_index}",
                cell=ref,
                author_id=author_id,
                author=author,
                comment_part=comments_name,
                **_xlsx_comment_anchor_metadata(
                    sheet_name="",
                    cell_ref=ref,
                    cell_values={},
                    column_headers={},
                ),
            )
            if element:
                elements.append(element)

    for comments_name in _zip_names(content, "xl/threadedComments/threadedComment"):
        if comments_name in seen_threaded_comment_parts:
            continue
        root = _read_zip_xml(content, comments_name)
        if root is None:
            continue
        elements.extend(_xlsx_threaded_comment_elements(
            root,
            sheet_index=None,
            sheet_name="",
            comment_part=comments_name,
            authors=threaded_comment_authors,
            cell_values={},
            column_headers={},
        ))

    for chart_index, chart_name in enumerate(_zip_names(content, "xl/charts/chart"), start=1):
        if chart_name in seen_chart_parts:
            continue
        root = _read_zip_xml(content, chart_name)
        if root is None:
            continue
        elements.extend(_chart_elements(
            root,
            chart_index=chart_index,
            location_prefix="workbook chart",
            base_metadata={"chart_part": chart_name, **_chart_source_metadata(content, chart_name, root)},
        ))

    return elements


def _extract_epub(content: bytes) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    chapter_names = [
        name for name in _zip_names(content, "", "")
        if name.lower().endswith((".xhtml", ".html", ".htm"))
        and not name.lower().startswith("meta-inf/")
    ]

    for chapter_index, chapter_name in enumerate(chapter_names, start=1):
        chapter_bytes = _read_zip_bytes(content, chapter_name)
        if not chapter_bytes:
            continue
        chapter_elements = _extract_html(chapter_bytes)
        for element in chapter_elements:
            element.location = f"chapter {chapter_index} {element.location or chapter_name}"
            element.metadata["chapter"] = chapter_index
            element.metadata["chapter_path"] = chapter_name
            if element.type == "image":
                src = element.metadata.get("src")
                image_part = None
                if src and not re.match(r"^[a-z][a-z0-9+.-]*:", str(src), flags=re.I):
                    image_part = _resolve_part_target(chapter_name, str(src))
                image_metadata = _zip_image_metadata(content, image_part)
                if image_part:
                    element.metadata["image_part"] = image_part
                element.metadata.update(image_metadata)
                if image_metadata.get("ocr_text"):
                    element.metadata["ocr_source"] = "epub_image"
                element.text = _image_element_text(
                    element.metadata.get("alt"),
                    element.metadata.get("title"),
                    image_part or str(src or ""),
                    metadata=element.metadata,
                )
            elements.append(element)
            if element.type == "image":
                ocr_element = _image_ocr_text_element(
                    element.metadata,
                    location=f"{element.location or f'chapter {chapter_index} image'} OCR",
                    image_part=element.metadata.get("image_part"),
                    ocr_source="epub_image",
                    base_metadata={
                        key: value
                        for key, value in element.metadata.items()
                        if key != "ocr_text"
                    },
                )
                if ocr_element:
                    elements.append(ocr_element)

    return elements


def _spreadsheet_column_name(index: int) -> str:
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name or "A"


def _odf_table_data(table: ET.Element) -> tuple[list[list[str]], dict[int, list[dict[str, Any]]]]:
    rows = []
    formulas_by_row: dict[int, list[dict[str, Any]]] = {}
    output_row_index = 0
    for row in table.findall(".//{*}table-row"):
        row_repeat = int(_attr_by_local_name(row, "number-rows-repeated") or 1)
        cells = []
        formulas = []
        column_index = 0
        for cell in row.findall("{*}table-cell"):
            cell_repeat = int(_attr_by_local_name(cell, "number-columns-repeated") or 1)
            value = (
                _attr_by_local_name(cell, "value")
                or _attr_by_local_name(cell, "string-value")
                or _odf_visible_text(cell)
            )
            formula = _attr_by_local_name(cell, "formula")
            for repeat_index in range(min(cell_repeat, 20)):
                column_index += 1
                cells.append(value)
                if formula:
                    formulas.append({
                        "cell": f"{_spreadsheet_column_name(column_index)}{output_row_index + 1}",
                        "formula": formula,
                        "value": value,
                        "repeat_index": repeat_index + 1 if cell_repeat > 1 else None,
                    })
        for _ in range(min(row_repeat, 20)):
            output_row_index += 1
            rows.append(cells[:])
            if formulas:
                row_formulas = []
                for formula in formulas:
                    adjusted = dict(formula)
                    column = re.match(r"([A-Z]+)", str(adjusted.get("cell") or ""))
                    adjusted["cell"] = f"{column.group(1) if column else 'A'}{output_row_index}"
                    row_formulas.append({key: value for key, value in adjusted.items() if value is not None})
                formulas_by_row[output_row_index] = row_formulas
    return rows, formulas_by_row


def _odf_table_rows(table: ET.Element) -> list[list[str]]:
    return _odf_table_data(table)[0]


def _odf_positive_int(value: str | None, default: int = 1) -> int:
    if value in (None, ""):
        return default
    parsed = _first_int(str(value))
    return parsed if parsed and parsed > 0 else default


def _odf_visible_text(node: ET.Element) -> str:
    excluded = {"annotation", "annotation-end", "change-info", "tracked-changes"}
    parts: list[str] = []

    def collect(current: ET.Element) -> None:
        if _local_name(current.tag) in excluded:
            return
        if current.text:
            parts.append(current.text)
        for child in list(current):
            collect(child)
            if child.tail:
                parts.append(child.tail)

    collect(node)
    return _clean_text(" ".join(parts))


def _odf_table_cell_text(cell: ET.Element) -> str:
    return _clean_text(str(
        _attr_by_local_name(cell, "value")
        or _attr_by_local_name(cell, "string-value")
        or _odf_visible_text(cell)
    ))


def _odf_style_property_nodes(node: ET.Element) -> list[ET.Element]:
    if _local_name(node.tag) != "style":
        return [
            current
            for current in [node, *list(node)]
            if current is node or _local_name(current.tag) in {"text-properties", "paragraph-properties"}
        ]
    return [
        current
        for current in node.iter()
        if current is node or _local_name(current.tag) in {"text-properties", "paragraph-properties"}
    ]


def _odf_first_style_attr(node: ET.Element, local_names: tuple[str, ...]) -> str | None:
    for current in _odf_style_property_nodes(node):
        for local_name in local_names:
            value = _attr_by_local_name(current, local_name)
            if value not in {None, ""}:
                return value
    return None


def _odf_is_bold(value: str | None) -> bool:
    cleaned = _clean_text(str(value or "")).lower()
    if not cleaned:
        return False
    if "bold" in cleaned:
        return True
    weight = _first_int(cleaned)
    return bool(weight is not None and weight >= 600)


def _odf_is_italic(value: str | None) -> bool:
    cleaned = _clean_text(str(value or "")).lower()
    return cleaned in {"italic", "oblique"} or "italic" in cleaned or "oblique" in cleaned


def _odf_is_underline(value: str | None) -> bool:
    cleaned = _clean_text(str(value or "")).lower()
    return bool(cleaned and cleaned not in {"0", "false", "none", "no"})


def _odf_style_properties(node: ET.Element) -> dict[str, Any]:
    font_weight = _odf_first_style_attr(node, ("font-weight", "font-weight-asian", "font-weight-complex"))
    font_style = _odf_first_style_attr(node, ("font-style", "font-style-asian", "font-style-complex"))
    underline = _odf_first_style_attr(node, (
        "text-underline-style",
        "text-underline-type",
        "text-underline",
    ))
    color = _odf_first_style_attr(node, ("color", "font-color"))
    font_size = _odf_first_style_attr(node, ("font-size", "font-size-asian", "font-size-complex"))

    metadata = {
        "bold": _odf_is_bold(font_weight) or None,
        "italic": _odf_is_italic(font_style) or None,
        "underline": _odf_is_underline(underline) or None,
        "color": color if color and color.lower() not in {"auto", "none", "transparent"} else None,
        "font_size": font_size,
    }
    return {key: value for key, value in metadata.items() if value not in (None, "", [], {})}


def _odf_style_definitions(content: bytes, root: ET.Element) -> dict[str, dict[str, Any]]:
    raw_definitions: dict[str, dict[str, Any]] = {}
    style_roots = [_read_zip_xml(content, "styles.xml"), root]

    for style_root in [candidate for candidate in style_roots if candidate is not None]:
        for style_node in style_root.iter():
            if _local_name(style_node.tag) != "style":
                continue
            style_name = _clean_text(str(_attr_by_local_name(style_node, "name") or ""))
            if not style_name:
                continue
            definition = {
                **raw_definitions.get(style_name, {}),
                **_odf_style_properties(style_node),
                "parent_style_name": _attr_by_local_name(style_node, "parent-style-name"),
                "style_family": _attr_by_local_name(style_node, "family"),
            }
            raw_definitions[style_name] = {
                key: value
                for key, value in definition.items()
                if value not in (None, "", [], {})
            }

    resolved: dict[str, dict[str, Any]] = {}

    def resolve(style_name: str, seen: set[str] | None = None) -> dict[str, Any]:
        if style_name in resolved:
            return resolved[style_name]
        seen = set(seen or set())
        if style_name in seen:
            return raw_definitions.get(style_name, {})
        seen.add(style_name)
        raw = raw_definitions.get(style_name, {})
        parent_name = _clean_text(str(raw.get("parent_style_name") or ""))
        parent = resolve(parent_name, seen) if parent_name and parent_name in raw_definitions else {}
        merged = {**parent, **raw}
        resolved[style_name] = {
            key: value
            for key, value in merged.items()
            if key != "parent_style_name" and value not in (None, "", [], {})
        }
        return resolved[style_name]

    for style_name in raw_definitions:
        resolve(style_name)

    return resolved


def _odf_inline_style(node: ET.Element, style_definitions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    style_name = _clean_text(str(_attr_by_local_name(node, "style-name") or ""))
    style = dict(style_definitions.get(style_name, {}))
    style.update(_odf_style_properties(node))
    if style_name:
        style["style_name"] = style_name
    return {
        key: value
        for key, value in style.items()
        if value not in (None, "", [], {})
    }


def _odf_text_runs(node: ET.Element, style_definitions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    excluded = {"annotation", "annotation-end", "change-info", "tracked-changes"}
    runs: list[dict[str, Any]] = []

    def add_run(text: str | None, style: dict[str, Any]) -> None:
        cleaned = _clean_text(str(text or ""))
        if not cleaned:
            return
        run = {
            "text": _clip(cleaned, 240),
            "bold": bool(style.get("bold")) or None,
            "italic": bool(style.get("italic")) or None,
            "underline": bool(style.get("underline")) or None,
            "color": style.get("color"),
            "font_size": style.get("font_size"),
            "style_name": style.get("style_name"),
            "style_family": style.get("style_family"),
        }
        runs.append({
            key: value
            for key, value in run.items()
            if value not in (None, "", [], {})
        })

    def walk(current: ET.Element, active_style: dict[str, Any]) -> None:
        if _local_name(current.tag) in excluded:
            return
        current_style = {**active_style, **_odf_inline_style(current, style_definitions)}
        add_run(current.text, current_style)
        for child in list(current):
            walk(child, current_style)
            add_run(child.tail, current_style)

    walk(node, {})
    return runs


def _odf_text_style_metadata(
    node: ET.Element,
    style_definitions: dict[str, dict[str, Any]],
    *,
    bullet_levels: list[int] | None = None,
) -> dict[str, Any]:
    runs = _odf_text_runs(node, style_definitions)
    styled_runs = [run for run in runs if _run_style_tags(run)]
    emphasis_terms = [run["text"] for run in styled_runs if run.get("text")]
    style_summary = _style_summary_from_runs(runs, bullet_levels)
    metadata = {
        "runs": runs[:24] if styled_runs else None,
        "emphasis_terms": emphasis_terms[:12] if emphasis_terms else None,
        "style_summary": style_summary,
        "bullet_levels": bullet_levels,
    }
    return {
        key: value
        for key, value in metadata.items()
        if value not in (None, "", [], {})
    }


def _odf_paragraph_bullet_levels(root: ET.Element) -> dict[int, list[int]]:
    levels: dict[int, list[int]] = {}

    def walk(node: ET.Element, depth: int) -> None:
        local_name = _local_name(node.tag)
        next_depth = depth + 1 if local_name == "list" else depth
        if local_name in {"p", "h"} and next_depth:
            levels.setdefault(id(node), []).append(max(0, next_depth - 1))
        for child in list(node):
            walk(child, next_depth)

    walk(root, 0)
    return {
        node_id: list(dict.fromkeys(values))
        for node_id, values in levels.items()
        if values
    }


def _odf_text_section_map(root: ET.Element) -> dict[int, str]:
    sections: dict[int, str] = {}
    current_heading = ""
    excluded = {"annotation", "tracked-changes"}

    def walk(node: ET.Element) -> None:
        nonlocal current_heading
        local_name = _local_name(node.tag)
        if local_name in excluded:
            return
        if local_name == "h":
            heading_text = _odf_visible_text(node)
            if heading_text:
                current_heading = heading_text
        elif local_name == "p" and current_heading:
            sections[id(node)] = current_heading
        for child in list(node):
            walk(child)

    walk(root)
    return sections


def _odf_table_cell_elements(
    table: ET.Element,
    *,
    table_index: int,
    table_name: str | None = None,
    location_prefix: str,
    base_metadata: dict[str, Any] | None = None,
    style_definitions: dict[str, dict[str, Any]] | None = None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    metadata_base = base_metadata or {}
    style_definitions = style_definitions or {}
    output_row_index = 0
    expanded_rows: list[list[str]] = []

    for row in table.findall(".//{*}table-row"):
        row_repeat = _odf_positive_int(_attr_by_local_name(row, "number-rows-repeated"))
        row_cells: list[dict[str, Any]] = []
        column_index = 0
        pending_covered_columns = 0

        for cell in list(row):
            local_name = _local_name(cell.tag)
            if local_name not in {"table-cell", "covered-table-cell"}:
                continue
            if local_name != "covered-table-cell" and pending_covered_columns:
                column_index += pending_covered_columns
                pending_covered_columns = 0

            column_repeat = _odf_positive_int(_attr_by_local_name(cell, "number-columns-repeated"))
            column_span = _odf_positive_int(_attr_by_local_name(cell, "number-columns-spanned"))
            row_span = _odf_positive_int(_attr_by_local_name(cell, "number-rows-spanned"))
            value = _odf_table_cell_text(cell)
            formula = _attr_by_local_name(cell, "formula")
            style_name = _attr_by_local_name(cell, "style-name")
            value_type = _attr_by_local_name(cell, "value-type")

            for repeat_index in range(min(column_repeat, 20)):
                column_index += 1
                if local_name == "covered-table-cell" and pending_covered_columns:
                    pending_covered_columns -= 1
                row_cells.append({
                    "column": column_index,
                    "text": value,
                    "formula": formula,
                    "style_name": style_name,
                    "value_type": value_type,
                    "column_span": column_span,
                    "row_span": row_span,
                    "column_repeat": column_repeat,
                    "row_repeat": row_repeat,
                    "column_repeat_index": repeat_index + 1,
                    "covered": local_name == "covered-table-cell",
                })
                if local_name != "covered-table-cell" and column_span > 1:
                    pending_covered_columns += column_span - 1

        for row_repeat_index in range(min(row_repeat, 20)):
            output_row_index += 1
            row_values = [""] * max((int(cell_info["column"]) for cell_info in row_cells), default=0)
            for cell_info in row_cells:
                column = int(cell_info["column"])
                if column > 0:
                    row_values[column - 1] = str(cell_info.get("text") or "")
            expanded_rows.append(row_values)
            for cell_info in row_cells:
                column = int(cell_info["column"])
                spreadsheet_cell = f"{_spreadsheet_column_name(column)}{output_row_index}"
                formula = cell_info.get("formula")
                value = cell_info.get("text")
                formula_metadata = [{
                    "cell": spreadsheet_cell,
                    "formula": formula,
                    "value": value,
                }] if formula else None
                style_metadata = _odf_text_style_metadata(cell, style_definitions)
                context_metadata = _table_cell_context_metadata(expanded_rows, output_row_index, column)
                element = _as_element(
                    "table_cell",
                    str(value or ""),
                    location=f"{location_prefix} cell {output_row_index}.{column}",
                    **metadata_base,
                    **style_metadata,
                    table=table_index,
                    table_name=table_name,
                    row=output_row_index,
                    column=column,
                    cell=spreadsheet_cell,
                    table_cell_reference=f"R{output_row_index}C{column}",
                    table_cell_grid_span=cell_info["column_span"] if cell_info["column_span"] > 1 else None,
                    table_cell_row_span=cell_info["row_span"] if cell_info["row_span"] > 1 else None,
                    table_cell_style_name=cell_info.get("style_name"),
                    table_cell_column_repeat=cell_info["column_repeat"] if cell_info["column_repeat"] > 1 else None,
                    table_cell_row_repeat=cell_info["row_repeat"] if cell_info["row_repeat"] > 1 else None,
                    table_cell_is_covered=cell_info["covered"] or None,
                    **context_metadata,
                    cell_value_type=cell_info.get("value_type"),
                    formulas=formula_metadata,
                    formula_cell=spreadsheet_cell if formula else None,
                    formula_expression=formula,
                    formula_result=value if formula else None,
                )
                if element:
                    elements.append(element)

    return elements


def _odf_text(node: ET.Element) -> str:
    return _clean_text(" ".join(part for part in node.itertext() if part))


def _odf_metadata_elements(content: bytes) -> list[DocumentElement]:
    root = _read_zip_xml(content, "meta.xml")
    if root is None:
        return []

    elements: list[DocumentElement] = []
    seen: set[tuple[str, str]] = set()
    for node in root.iter():
        local_name = _local_name(node.tag)
        field_info = ODF_METADATA_FIELDS.get(local_name)
        if not field_info:
            continue
        field, label = field_info
        if local_name == "document-statistic":
            stats = {
                _snake_case(key): value
                for key, value in node.attrib.items()
                if value not in {None, ""}
            }
            value = "; ".join(f"{key}: {stats[key]}" for key in sorted(stats))
        else:
            value = _odf_text(node)
        if not value:
            continue
        key = (field, value)
        if key in seen:
            continue
        seen.add(key)
        element = _as_element(
            "document_metadata",
            f"{label}: {value}",
            location=f"odf metadata {field}",
            metadata_part="meta.xml",
            metadata_field=field,
            metadata_label=label,
            metadata_value=value,
        )
        if element:
            elements.append(element)
    return elements


def _odf_link_elements(
    root: ET.Element,
    *,
    extension: str,
    location_prefix: str = "odf",
    base_metadata: dict[str, Any] | None = None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    for link_index, link in enumerate(root.findall(".//{*}a"), start=1):
        href = _attr_by_local_name(link, "href")
        if not href:
            continue
        label = _odf_text(link) or href
        metadata = {"format": extension.lstrip("."), **(base_metadata or {})}
        element = _as_element(
            "link",
            f"{label} - {href}" if label != href else href,
            location=f"{location_prefix} link {link_index}",
            uri=href,
            href=href,
            link_kind="odf_hyperlink",
            **metadata,
        )
        if element:
            elements.append(element)
    return elements


def _odf_annotation_names(node: ET.Element) -> list[str]:
    names: list[str] = []
    for annotation in node.findall(".//{*}annotation"):
        annotation_name = _clean_text(str(_attr_by_local_name(annotation, "name") or ""))
        if annotation_name:
            names.append(annotation_name)
    return _unique_preserve_order(names)


def _odf_annotation_range_anchors(root: ET.Element) -> dict[str, dict[str, Any]]:
    active_annotation_ids: list[str] = []
    texts: dict[str, list[str]] = defaultdict(list)

    def add_active_text(text: str | None) -> None:
        cleaned = _clean_text(str(text or ""))
        if not cleaned:
            return
        for annotation_id in active_annotation_ids:
            texts.setdefault(annotation_id, []).append(cleaned)

    def end_annotation(annotation_id: str) -> None:
        for index in range(len(active_annotation_ids) - 1, -1, -1):
            if active_annotation_ids[index] == annotation_id:
                del active_annotation_ids[index]
                return

    def walk(node: ET.Element) -> None:
        if _local_name(node.tag) == "tracked-changes":
            return
        add_active_text(node.text)
        for child in list(node):
            local_name = _local_name(child.tag)
            annotation_id = _clean_text(str(_attr_by_local_name(child, "name") or ""))
            if local_name == "annotation" and annotation_id:
                active_annotation_ids.append(annotation_id)
            elif local_name == "annotation-end" and annotation_id:
                end_annotation(annotation_id)
            else:
                walk(child)
            add_active_text(child.tail)

    walk(root)
    return {
        annotation_id: {"comment_anchor_text": _clean_text(" ".join(parts))}
        for annotation_id, parts in texts.items()
        if _clean_text(" ".join(parts))
    }


def _odf_annotation_table_anchors(root: ET.Element) -> dict[str, dict[str, Any]]:
    anchors: dict[str, dict[str, Any]] = {}
    for table in root.findall(".//{*}table"):
        table_name = _attr_by_local_name(table, "name") or "Sheet"
        output_row_index = 0
        headers_by_column: dict[int, str] = {}
        for row in table.findall(".//{*}table-row"):
            row_repeat = _odf_positive_int(_attr_by_local_name(row, "number-rows-repeated"))
            row_cells: list[dict[str, Any]] = []
            column_index = 0
            pending_covered_columns = 0
            for cell in list(row):
                local_name = _local_name(cell.tag)
                if local_name not in {"table-cell", "covered-table-cell"}:
                    continue
                if local_name != "covered-table-cell" and pending_covered_columns:
                    column_index += pending_covered_columns
                    pending_covered_columns = 0
                column_repeat = _odf_positive_int(_attr_by_local_name(cell, "number-columns-repeated"))
                column_span = _odf_positive_int(_attr_by_local_name(cell, "number-columns-spanned"))
                value = _odf_table_cell_text(cell)
                annotation_names = _odf_annotation_names(cell)
                for repeat_index in range(min(column_repeat, 20)):
                    column_index += 1
                    if local_name == "covered-table-cell" and pending_covered_columns:
                        pending_covered_columns -= 1
                    row_cells.append({
                        "column": column_index,
                        "value": value,
                        "annotations": annotation_names,
                    })
                    if local_name != "covered-table-cell" and column_span > 1:
                        pending_covered_columns += column_span - 1

            for _ in range(min(row_repeat, 20)):
                output_row_index += 1
                for cell_info in row_cells:
                    column = int(cell_info["column"])
                    value = _clean_text(str(cell_info.get("value") or ""))
                    if output_row_index == 1 and value:
                        headers_by_column[column] = value
                    header = headers_by_column.get(column, "") if output_row_index > 1 else ""
                    anchor_parts = [header, value] if header and value and header != value else [value or header]
                    anchor_text = _clean_text("; ".join(part for part in anchor_parts if part))
                    if not anchor_text:
                        anchor_text = value or header
                    for annotation_name in cell_info.get("annotations", []):
                        anchors[annotation_name] = {
                            "comment_anchor_sheet": table_name,
                            "comment_anchor_cell": f"{_spreadsheet_column_name(column)}{output_row_index}",
                            "comment_anchor_header": header or None,
                            "comment_anchor_value": value or None,
                            "comment_anchor_text": anchor_text or None,
                        }
    return anchors


def _odf_annotation_elements(root: ET.Element, *, extension: str) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    range_anchors = _odf_annotation_range_anchors(root)
    table_anchors = _odf_annotation_table_anchors(root)
    for annotation_index, annotation in enumerate(root.findall(".//{*}annotation"), start=1):
        annotation_name = _attr_by_local_name(annotation, "name") or str(annotation_index)
        anchor_metadata = {
            **range_anchors.get(annotation_name, {}),
            **table_anchors.get(annotation_name, {}),
        }
        author = ""
        date = ""
        text_parts = []
        for child in annotation:
            local_name = _local_name(child.tag)
            if local_name == "creator":
                author = _odf_text(child)
            elif local_name == "date":
                date = _odf_text(child)
            elif local_name not in {"annotation-end"}:
                text = _odf_text(child)
                if text:
                    text_parts.append(text)
        text = _clean_text(" ".join(text_parts)) or _odf_text(annotation)
        element = _as_element(
            "comment",
            f"{author}: {text}" if author else text,
            location=f"odf annotation {annotation_index}",
            format=extension.lstrip("."),
            comment_id=annotation_name,
            comment_kind="odf_annotation",
            author=author,
            date=date,
            comment_part="content.xml",
            **anchor_metadata,
        )
        if element:
            elements.append(element)
    for annotation_end_index, annotation_end in enumerate(root.findall(".//{*}annotation-end"), start=1):
        annotation_name = _attr_by_local_name(annotation_end, "name") or str(annotation_end_index)
        label = f"Annotation range end {annotation_name}"
        element = _as_element(
            "comment",
            label,
            location=f"odf annotation end {annotation_end_index}",
            format=extension.lstrip("."),
            comment_id=annotation_name,
            comment_kind="odf_annotation_end",
            comment_part="content.xml",
        )
        if element:
            elements.append(element)
    return elements


ODF_REVISION_TYPES = {
    "insertion": "inserted",
    "deletion": "deleted",
    "format-change": "format_changed",
}


def _odf_first_descendant_text(node: ET.Element, local_names: set[str]) -> str:
    for descendant in node.iter():
        if descendant is node:
            continue
        if _local_name(descendant.tag) in local_names:
            text = _odf_text(descendant)
            if text:
                return text
    return ""


def _odf_revision_body_text(node: ET.Element) -> str:
    parts: list[str] = []

    def collect(current: ET.Element) -> None:
        if _local_name(current.tag) == "change-info":
            return
        if current.text:
            parts.append(current.text)
        for child in list(current):
            collect(child)
            if child.tail:
                parts.append(child.tail)

    collect(node)
    return _clean_text(" ".join(parts))


def _odf_visible_revision_texts(root: ET.Element) -> dict[str, str]:
    active_change_ids: list[str] = []
    texts: dict[str, list[str]] = defaultdict(list)

    def add_active_text(text: str | None) -> None:
        cleaned = _clean_text(str(text or ""))
        if not cleaned:
            return
        for change_id in active_change_ids:
            texts.setdefault(change_id, []).append(cleaned)

    def end_change(change_id: str) -> None:
        for index in range(len(active_change_ids) - 1, -1, -1):
            if active_change_ids[index] == change_id:
                del active_change_ids[index]
                return

    def walk(node: ET.Element) -> None:
        if _local_name(node.tag) == "tracked-changes":
            return
        add_active_text(node.text)
        for child in list(node):
            local_name = _local_name(child.tag)
            change_id = _clean_text(str(_attr_by_local_name(child, "change-id") or _attr_by_local_name(child, "id") or ""))
            if local_name == "change-start" and change_id:
                active_change_ids.append(change_id)
                add_active_text(child.text)
            elif local_name == "change-end" and change_id:
                end_change(change_id)
            elif local_name == "change" and change_id:
                add_active_text(child.text)
            else:
                walk(child)
            add_active_text(child.tail)

    walk(root)
    return {
        change_id: _clean_text(" ".join(parts))
        for change_id, parts in texts.items()
        if _clean_text(" ".join(parts))
    }


def _odf_revision_elements(root: ET.Element, *, extension: str) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    visible_revision_texts = _odf_visible_revision_texts(root)
    for revision_index, region in enumerate(root.findall(".//{*}changed-region"), start=1):
        revision_id = _clean_text(str(
            _attr_by_local_name(region, "id")
            or _attr_by_local_name(region, "change-id")
            or revision_index
        ))
        revision_node = next(
            (child for child in list(region) if _local_name(child.tag) in ODF_REVISION_TYPES),
            None,
        )
        if revision_node is None:
            continue
        revision_type = ODF_REVISION_TYPES[_local_name(revision_node.tag)]
        author = _odf_first_descendant_text(revision_node, {"creator", "initial-creator"})
        date = _odf_first_descendant_text(revision_node, {"date", "creation-date"})
        visible_text = visible_revision_texts.get(revision_id, "")
        body_text = _odf_revision_body_text(revision_node)
        text = body_text if revision_type == "deleted" else (visible_text or body_text)
        if not text:
            continue
        label = revision_type.replace("_", " ").title()
        element = _as_element(
            "revision",
            f"{label}: {text}",
            location=f"odf revision {revision_id or revision_index}",
            format=extension.lstrip("."),
            revision_type=revision_type,
            revision_author=author,
            revision_date=date,
            revision_id=revision_id,
            revision_part="content.xml",
            revision_change_kind="odf_tracked_change",
            revision_anchor_text=visible_text or None,
        )
        if element:
            elements.append(element)
    return elements


def _odf_image_elements(content: bytes, root: ET.Element, *, extension: str) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    for image_index, image in enumerate(root.findall(".//{*}image"), start=1):
        href = _attr_by_local_name(image, "href")
        title = _xml_text(image, ("title", "desc"))
        image_part = None
        if href and not re.match(r"^[a-z][a-z0-9+.-]*:", href, flags=re.I):
            image_part = _resolve_part_target("content.xml", href)
        image_metadata = _zip_image_metadata(content, image_part)
        element = _as_element(
            "image",
            _image_element_text(title, image_part or href, metadata=image_metadata),
            location=f"image {image_index}",
            format=extension.lstrip("."),
            href=href,
            title=title,
            image_part=image_part,
            ocr_source=f"{extension.lstrip('.')}_image" if image_metadata.get("ocr_text") else None,
            **image_metadata,
        )
        if element:
            elements.append(element)
    return elements


ODF_FORM_CONTROL_TYPES = {
    "button": "button",
    "checkbox": "checkbox",
    "combobox": "combobox",
    "date": "date",
    "file": "file",
    "fixed-text": "label",
    "formatted-text": "formatted_text",
    "frame": "frame",
    "generic-control": "control",
    "grid": "grid",
    "hidden": "hidden",
    "image": "image",
    "image-frame": "image",
    "listbox": "listbox",
    "number": "number",
    "password": "password",
    "radio": "radio",
    "text": "text",
    "textarea": "textarea",
    "time": "time",
    "value-range": "range",
}


def _is_odf_form_control(node: ET.Element) -> bool:
    namespace = _namespace_uri(node.tag)
    return "opendocument:xmlns:form" in namespace and _local_name(node.tag) in ODF_FORM_CONTROL_TYPES


def _is_odf_draw_control(node: ET.Element) -> bool:
    namespace = _namespace_uri(node.tag)
    return "opendocument:xmlns:drawing" in namespace and _local_name(node.tag) == "control"


def _odf_form_control_key(node: ET.Element) -> str:
    return _clean_text(str(
        _attr_by_local_name(node, "id")
        or _attr_by_local_name(node, "name")
        or _attr_by_local_name(node, "label")
        or ""
    ))


def _odf_form_control_definitions(root: ET.Element) -> dict[str, ET.Element]:
    definitions: dict[str, ET.Element] = {}
    for node in root.iter():
        if not _is_odf_form_control(node):
            continue
        for key in (
            _attr_by_local_name(node, "id"),
            _attr_by_local_name(node, "name"),
            _attr_by_local_name(node, "label"),
        ):
            cleaned = _clean_text(str(key or ""))
            if cleaned and cleaned not in definitions:
                definitions[cleaned] = node
    return definitions


def _odf_form_options(node: ET.Element) -> tuple[list[str], list[str], str | None]:
    option_labels: list[str] = []
    option_values: list[str] = []
    selected_option = None
    for option in node.findall(".//{*}option"):
        label = _clean_text(str(_attr_by_local_name(option, "label") or _odf_text(option)))
        value = _clean_text(str(_attr_by_local_name(option, "value") or label))
        if label:
            option_labels.append(label)
        if value:
            option_values.append(value)
        selected = str(
            _attr_by_local_name(option, "selected")
            or _attr_by_local_name(option, "current-selected")
            or ""
        ).lower()
        if label and selected in {"1", "true", "yes", "on", "selected"}:
            selected_option = label
    return _unique_preserve_order(option_labels), _unique_preserve_order(option_values), selected_option


def _odf_form_value(node: ET.Element, field_type: str, selected_option: str | None) -> tuple[str, bool | None]:
    raw_state = _clean_text(str(
        _attr_by_local_name(node, "current-state")
        or _attr_by_local_name(node, "state")
        or _attr_by_local_name(node, "checked")
        or ""
    ))
    checked = None
    if raw_state:
        checked = raw_state.lower() in {"1", "true", "yes", "on", "checked", "selected"}

    raw_value = _clean_text(str(
        _attr_by_local_name(node, "current-value")
        or _attr_by_local_name(node, "value")
        or _attr_by_local_name(node, "selected-value")
        or _attr_by_local_name(node, "selected")
        or selected_option
        or ""
    ))
    if field_type in {"checkbox", "radio"} and checked is not None and not raw_value:
        raw_value = "True" if checked else "False"
    return raw_value, checked


def _odf_form_parent_name(node: ET.Element, root: ET.Element) -> str | None:
    for form in root.findall(".//{*}form"):
        if node is form:
            continue
        if any(descendant is node for descendant in form.iter()):
            return _attr_by_local_name(form, "name")
    return None


def _odf_form_field_element(
    node: ET.Element,
    *,
    field_index: int,
    extension: str,
    location: str,
    source_part: str,
    base_metadata: dict[str, Any] | None = None,
    anchor_id: str | None = None,
    root_for_parent: ET.Element | None = None,
) -> DocumentElement | None:
    local_name = _local_name(node.tag)
    field_type = ODF_FORM_CONTROL_TYPES.get(local_name, local_name)
    control_id = _attr_by_local_name(node, "id")
    control_name = _attr_by_local_name(node, "name")
    label = _clean_text(str(
        _attr_by_local_name(node, "label")
        or _attr_by_local_name(node, "title")
        or control_name
        or control_id
        or f"Field {field_index}"
    ))
    options, option_values, selected_option = _odf_form_options(node)
    value, checked = _odf_form_value(node, field_type, selected_option)
    placeholder = _attr_by_local_name(node, "placeholder")
    form_name = _attr_by_local_name(node, "form-name")
    if not form_name and root_for_parent is not None:
        form_name = _odf_form_parent_name(node, root_for_parent)
    display_value = value or selected_option or ""
    text = f"{label}: {display_value}" if display_value else label
    if options:
        text = f"{text}; options: {', '.join(options[:8])}"
    metadata = {
        "format": extension.lstrip("."),
        **(base_metadata or {}),
    }
    return _as_element(
        "form_field",
        text,
        location=location,
        **metadata,
        form_kind="odf_control",
        field_type=field_type,
        label=label,
        value=value or selected_option,
        form_field_alias=label,
        form_field_tag=control_name,
        form_field_id=control_id,
        form_field_placeholder=placeholder,
        form_field_checked=checked,
        form_field_options=options or None,
        form_field_option_values=option_values or None,
        form_field_selected_option=selected_option or value or None,
        control_id=control_id or anchor_id,
        control_name=control_name,
        control_part=source_part,
        odf_form_name=form_name,
        odf_control_anchor=anchor_id,
        odf_control_element=local_name,
    )


def _odf_form_control_elements(
    root: ET.Element,
    *,
    extension: str,
    location_prefix: str = "odf",
    source_part: str = "content.xml",
    base_metadata: dict[str, Any] | None = None,
    definitions: dict[str, ET.Element] | None = None,
    seen_controls: set[str] | None = None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    seen = seen_controls if seen_controls is not None else set()
    definitions = definitions or _odf_form_control_definitions(root)

    for anchor_index, anchor in enumerate((node for node in root.iter() if _is_odf_draw_control(node)), start=1):
        anchor_id = _clean_text(str(_attr_by_local_name(anchor, "control") or _attr_by_local_name(anchor, "id") or ""))
        if not anchor_id:
            continue
        control = definitions.get(anchor_id)
        if control is None:
            continue
        key = f"{source_part}:{location_prefix}:anchor:{anchor_id}"
        if key in seen:
            continue
        seen.add(key)
        element = _odf_form_field_element(
            control,
            field_index=anchor_index,
            extension=extension,
            location=f"{location_prefix} control {anchor_index}",
            source_part=source_part,
            base_metadata=base_metadata,
            anchor_id=anchor_id,
            root_for_parent=root,
        )
        if element:
            elements.append(element)

    for field_index, control in enumerate((node for node in root.iter() if _is_odf_form_control(node)), start=1):
        control_key = _odf_form_control_key(control) or f"{location_prefix}:{field_index}"
        key = f"{source_part}:{location_prefix}:definition:{control_key}"
        anchored_key = f"{source_part}:{location_prefix}:anchor:{control_key}"
        if key in seen or anchored_key in seen:
            continue
        seen.add(key)
        element = _odf_form_field_element(
            control,
            field_index=field_index,
            extension=extension,
            location=f"{location_prefix} form field {field_index}",
            source_part=source_part,
            base_metadata=base_metadata,
            root_for_parent=root,
        )
        if element:
            elements.append(element)
    return elements


ODF_DRAW_SHAPE_TYPES = {
    "caption",
    "circle",
    "custom-shape",
    "ellipse",
    "path",
    "polygon",
    "polyline",
    "rect",
}

ODF_DRAW_CONNECTOR_TYPES = {"connector", "line"}


def _odf_length_inches(value: str | None) -> float | None:
    cleaned = _clean_text(str(value or ""))
    if not cleaned:
        return None
    match = re.match(r"^(-?\d+(?:\.\d+)?)\s*([a-z%]*)$", cleaned, flags=re.I)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2).lower() or "in"
    conversions = {
        "in": 1.0,
        "cm": 1 / 2.54,
        "mm": 1 / 25.4,
        "pt": 1 / 72,
        "pc": 1 / 6,
        "px": 1 / 96,
    }
    factor = conversions.get(unit)
    if factor is None:
        return None
    return round(number * factor, 3)


def _odf_draw_geometry_metadata(node: ET.Element) -> dict[str, Any]:
    x = _odf_length_inches(_attr_by_local_name(node, "x"))
    y = _odf_length_inches(_attr_by_local_name(node, "y"))
    width = _odf_length_inches(_attr_by_local_name(node, "width"))
    height = _odf_length_inches(_attr_by_local_name(node, "height"))

    x1 = _odf_length_inches(_attr_by_local_name(node, "x1"))
    y1 = _odf_length_inches(_attr_by_local_name(node, "y1"))
    x2 = _odf_length_inches(_attr_by_local_name(node, "x2"))
    y2 = _odf_length_inches(_attr_by_local_name(node, "y2"))
    if x is None and None not in (x1, x2):
        x = min(x1, x2)
        width = abs(x2 - x1)
    if y is None and None not in (y1, y2):
        y = min(y1, y2)
        height = abs(y2 - y1)

    center_x = x + width / 2 if x is not None and width is not None else None
    center_y = y + height / 2 if y is not None and height is not None else None
    bbox = None
    if None not in (x, y, width, height):
        bbox = f"x={x:.2f}in y={y:.2f}in w={width:.2f}in h={height:.2f}in"

    metadata = {
        "layout_x_inches": x,
        "layout_y_inches": y,
        "layout_width_inches": width,
        "layout_height_inches": height,
        "layout_center_x_inches": round(center_x, 3) if center_x is not None else None,
        "layout_center_y_inches": round(center_y, 3) if center_y is not None else None,
        "layout_bbox": bbox,
        "layout_rotation_degrees": _attr_by_local_name(node, "transform"),
    }
    return {key: value for key, value in metadata.items() if value not in (None, "")}


def _odf_draw_label_parts(node: ET.Element) -> tuple[str, str, str]:
    title_node = _odf_first_child(node, {"title"})
    desc_node = _odf_first_child(node, {"desc"})
    title = _odf_text(title_node) if title_node is not None else ""
    description = _odf_text(desc_node) if desc_node is not None else ""
    text = _xml_text(node, ("p", "span"))
    return title, description, text


def _odf_draw_shape_metadata(node: ET.Element, shape_index: int) -> dict[str, Any]:
    title, description, text = _odf_draw_label_parts(node)
    shape_id = (
        _attr_by_local_name(node, "id")
        or _attr_by_local_name(node, "name")
        or _attr_by_local_name(node, "xml:id")
    )
    shape_name = _attr_by_local_name(node, "name")
    shape_type = _local_name(node.tag)
    label = title or text or description or shape_name or shape_id or f"{shape_type} {shape_index}"
    metadata = {
        **_odf_draw_geometry_metadata(node),
        "shape_index": shape_index,
        "shape_id": shape_id,
        "shape_name": shape_name,
        "shape_title": title or label,
        "shape_description": description,
        "shape_type": f"odf_{shape_type}",
    }
    return {key: value for key, value in metadata.items() if value not in (None, "", [], {})}


def _odf_draw_shape_text(metadata: dict[str, Any], node: ET.Element) -> str:
    title = _clean_text(str(metadata.get("shape_title") or ""))
    description = _clean_text(str(metadata.get("shape_description") or ""))
    text = _xml_text(node, ("p", "span"))
    if title and description and title.lower() != description.lower():
        return f"{title}: {description}"
    return title or description or text or _clean_text(str(metadata.get("shape_name") or metadata.get("shape_id") or ""))


def _odf_connectable_shape_labels(root: ET.Element) -> dict[str, str]:
    labels: dict[str, str] = {}
    for shape_index, node in enumerate(root.iter(), start=1):
        if _local_name(node.tag) not in ODF_DRAW_SHAPE_TYPES:
            continue
        metadata = _odf_draw_shape_metadata(node, shape_index)
        label = _clean_text(str(metadata.get("shape_title") or "")) or _odf_draw_shape_text(metadata, node)
        for key in (metadata.get("shape_id"), metadata.get("shape_name")):
            cleaned = _clean_text(str(key or ""))
            if cleaned and label:
                labels[cleaned] = label
    return labels


def _odf_connector_relation(node: ET.Element) -> str:
    title, description, text = _odf_draw_label_parts(node)
    relation = (
        text
        or title
        or description
        or _attr_by_local_name(node, "name")
        or _local_name(node.tag)
    )
    return _clean_text(str(relation or "connects_to")) or "connects_to"


def _odf_shape_elements(
    root: ET.Element,
    *,
    extension: str,
    location_prefix: str = "odf",
    base_metadata: dict[str, Any] | None = None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    metadata_base = {"format": extension.lstrip("."), **(base_metadata or {})}
    shape_labels = _odf_connectable_shape_labels(root)

    shape_index = 0
    for node in root.iter():
        if _local_name(node.tag) not in ODF_DRAW_SHAPE_TYPES:
            continue
        shape_index += 1
        shape_metadata = _odf_draw_shape_metadata(node, shape_index)
        shape_text = _odf_draw_shape_text(shape_metadata, node)
        element = _as_element(
            "shape_metadata",
            shape_text,
            location=f"{location_prefix} shape {shape_index}",
            **metadata_base,
            **shape_metadata,
        )
        if element:
            elements.append(element)

    connector_index = 0
    for node in root.iter():
        if _local_name(node.tag) not in ODF_DRAW_CONNECTOR_TYPES:
            continue
        connector_index += 1
        title, description, text = _odf_draw_label_parts(node)
        connector_id = _attr_by_local_name(node, "id") or _attr_by_local_name(node, "name")
        connector_name = _attr_by_local_name(node, "name")
        source_shape_id = _attr_by_local_name(node, "start-shape")
        target_shape_id = _attr_by_local_name(node, "end-shape")
        source_label = shape_labels.get(str(source_shape_id or ""))
        target_label = shape_labels.get(str(target_shape_id or ""))
        relation = _odf_connector_relation(node)
        connector_label = title or text or description or relation
        connector_text = (
            f"{source_label} -> {target_label}: {relation}"
            if source_label and target_label
            else connector_label
        )
        element = _as_element(
            "shape_connector",
            connector_text,
            location=f"{location_prefix} connector {connector_index}",
            **metadata_base,
            **_odf_draw_geometry_metadata(node),
            connector_index=connector_index,
            connector_id=connector_id,
            connector_name=connector_name,
            connector_title=title or (relation if relation != "connects_to" else None),
            connector_description=description,
            source_shape_id=source_shape_id,
            target_shape_id=target_shape_id,
            source_label=source_label or None,
            target_label=target_label or None,
            source_connection_index=_attr_by_local_name(node, "start-glue-point"),
            target_connection_index=_attr_by_local_name(node, "end-glue-point"),
            relation=relation,
            drawing_part="content.xml",
        )
        if element:
            elements.append(element)
    return elements


def _odf_direct_children(node: ET.Element, local_names: set[str]) -> list[ET.Element]:
    return [child for child in list(node) if _local_name(child.tag) in local_names]


def _odf_first_child(node: ET.Element, local_names: set[str]) -> ET.Element | None:
    children = _odf_direct_children(node, local_names)
    return children[0] if children else None


def _odf_chart_nodes(root: ET.Element) -> list[ET.Element]:
    charts: list[ET.Element] = []
    for node in root.iter():
        if _local_name(node.tag) != "chart":
            continue
        if _attr_by_local_name(node, "class") or _odf_direct_children(node, {"plot-area", "series", "axis", "legend"}):
            charts.append(node)
    return charts


def _odf_chart_object_part(content: bytes, source_part: str, href: str | None) -> str | None:
    resolved = _resolve_part_target(source_part, href)
    if not resolved:
        return None
    candidates = [
        resolved,
        posixpath.join(resolved.rstrip("/"), "content.xml"),
    ]
    for candidate in dict.fromkeys(candidates):
        if _read_zip_xml(content, candidate) is not None:
            return candidate
    return None


def _odf_chart_class_name(value: str | None) -> str | None:
    if not value:
        return None
    return str(value).rsplit(":", 1)[-1]


def _odf_chart_title(chart: ET.Element) -> str | None:
    title = _odf_first_child(chart, {"title"})
    if title is None:
        return None
    return _odf_text(title)


def _odf_chart_series_from_table(chart: ET.Element) -> list[dict[str, Any]]:
    table = next((node for node in chart.findall(".//{*}table") if _local_name(node.tag) == "table"), None)
    if table is None:
        return []

    rows = [
        [_clean_text(str(cell or "")) for cell in row]
        for row in _odf_table_rows(table)
        if any(_clean_text(str(cell or "")) for cell in row)
    ]
    if len(rows) < 2:
        return []

    headers = rows[0]
    categories = [row[0] if row else "" for row in rows[1:]]
    series_payloads: list[dict[str, Any]] = []
    max_columns = max((len(row) for row in rows), default=0)
    for column_index in range(1, max_columns):
        values = [row[column_index] if len(row) > column_index else "" for row in rows[1:]]
        if not any(values):
            continue
        name = headers[column_index] if len(headers) > column_index and headers[column_index] else f"Series {column_index}"
        series_payloads.append({
            "name": name,
            "categories": categories,
            "values": values,
        })
    return series_payloads


def _odf_chart_series_from_ranges(chart: ET.Element) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for series_index, series in enumerate(chart.findall(".//{*}series"), start=1):
        name = (
            _attr_by_local_name(series, "label-cell-address")
            or _attr_by_local_name(series, "name")
            or f"Series {series_index}"
        )
        values_range = _attr_by_local_name(series, "values-cell-range-address")
        categories = [
            _attr_by_local_name(domain, "cell-range-address") or ""
            for domain in series.findall(".//{*}domain")
        ]
        payloads.append({
            "name": name,
            "categories": categories,
            "values": [values_range] if values_range else [],
            "values_range": values_range,
            "label_cell": _attr_by_local_name(series, "label-cell-address"),
        })
    return payloads


def _odf_chart_source_table_elements(
    chart: ET.Element,
    *,
    chart_index: int,
    location_prefix: str,
    base_metadata: dict[str, Any],
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    for table_index, table in enumerate(chart.findall(".//{*}table"), start=1):
        table_name = _attr_by_local_name(table, "name") or f"Chart {chart_index} data"
        rows, formulas_by_row = _odf_table_data(table)
        source_metadata = {
            **base_metadata,
            "chart_source_index": table_index,
            "chart_source_extraction_method": "odf_chart_table",
        }
        elements.extend(_odf_table_cell_elements(
            table,
            table_index=table_index,
            table_name=table_name,
            location_prefix=f"{location_prefix} chart data {table_index}",
            base_metadata=source_metadata,
        ))
        table_elements = _table_elements_from_rows(
            rows,
            location_prefix=f"{location_prefix} chart data {table_index}",
            table=table_index,
            base_metadata={**source_metadata, "table_name": table_name},
        )
        for element in table_elements:
            formulas = formulas_by_row.get(int(element.metadata.get("row", 0) or 0))
            if formulas:
                element.metadata["formulas"] = formulas
            elements.append(element)
    return elements


def _odf_chart_elements_from_chart(
    chart: ET.Element,
    *,
    chart_index: int,
    chart_part: str,
    source_part: str,
    location_prefix: str,
    base_metadata: dict[str, Any] | None = None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    metadata_base = {
        **(base_metadata or {}),
        "chart": chart_index,
        "chart_part": chart_part,
        "chart_source_part": source_part,
        "chart_source_kind": "odf_chart_table",
        "chart_source_extraction_method": "odf_chart_table",
    }
    chart_kind = _odf_chart_class_name(_attr_by_local_name(chart, "class"))
    chart_title = _odf_chart_title(chart)

    for axis_index, axis in enumerate(chart.findall(".//{*}axis"), start=1):
        dimension = _attr_by_local_name(axis, "dimension")
        axis_type = {
            "x": "category",
            "y": "value",
            "z": "series",
        }.get(str(dimension or "").lower(), dimension or "axis")
        axis_title_node = _odf_first_child(axis, {"title"})
        axis_title = _odf_text(axis_title_node) if axis_title_node is not None else None
        axis_name = _attr_by_local_name(axis, "name")
        text_parts = [f"{str(axis_type).title()} axis"]
        if axis_title:
            text_parts.append(axis_title)
        if axis_name:
            text_parts.append(f"name {axis_name}")
        element = _as_element(
            "chart_axis",
            "; ".join(text_parts),
            location=f"{location_prefix} chart {chart_index} axis {axis_index}",
            **metadata_base,
            chart_type=chart_kind,
            chart_title=chart_title,
            chart_axis_index=axis_index,
            chart_axis_type=axis_type,
            chart_axis_title=axis_title,
            chart_axis_id=axis_name,
        )
        if element:
            elements.append(element)

    legend = chart.find(".//{*}legend")
    if legend is not None:
        legend_position = _attr_by_local_name(legend, "legend-position")
        legend_overlay = _attr_by_local_name(legend, "legend-align")
        element = _as_element(
            "chart_legend",
            "; ".join(part for part in ["Legend", f"position {legend_position}" if legend_position else ""] if part),
            location=f"{location_prefix} chart {chart_index} legend",
            **metadata_base,
            chart_type=chart_kind,
            chart_title=chart_title,
            chart_legend_position=legend_position,
            chart_legend_overlay=legend_overlay,
        )
        if element:
            elements.append(element)

    series_payloads = _odf_chart_series_from_table(chart) or _odf_chart_series_from_ranges(chart)
    for series_index, payload in enumerate(series_payloads, start=1):
        name = _clean_text(str(payload.get("name") or f"Series {series_index}"))
        categories = [_clean_text(str(value or "")) for value in payload.get("categories", [])]
        values = [_clean_text(str(value or "")) for value in payload.get("values", [])]
        pairs = [f"{category}: {value}" if category else value for category, value in list(zip(categories, values))[:8]]
        text = "; ".join(part for part in [name, ", ".join(pairs)] if part)
        element = _as_element(
            "chart_series",
            text,
            location=f"{location_prefix} chart {chart_index} series {series_index}",
            **metadata_base,
            chart_type=chart_kind,
            chart_title=chart_title,
            series=series_index,
            series_name=name,
            categories=categories[:12],
            values=values[:12],
        )
        if element:
            elements.append(element)

        for point_index, (category, value) in enumerate(zip(categories, values), start=1):
            if not category and not value:
                continue
            point = _as_element(
                "chart_point",
                f"{name}; {category}: {value}" if category else f"{name}: {value}",
                location=f"{location_prefix} chart {chart_index} series {series_index} point {point_index}",
                **metadata_base,
                chart_type=chart_kind,
                chart_title=chart_title,
                series=series_index,
                point=point_index,
                series_name=name,
                category=category,
                value=value,
            )
            if point:
                elements.append(point)

    elements.extend(_odf_chart_source_table_elements(
        chart,
        chart_index=chart_index,
        location_prefix=location_prefix,
        base_metadata={
            **metadata_base,
            "chart_type": chart_kind,
            "chart_title": chart_title,
        },
    ))
    return elements


def _odf_chart_elements(
    content: bytes,
    root: ET.Element,
    *,
    extension: str,
    location_prefix: str = "odf",
    source_part: str = "content.xml",
    base_metadata: dict[str, Any] | None = None,
    seen_chart_parts: set[str] | None = None,
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    seen = seen_chart_parts if seen_chart_parts is not None else set()
    metadata = {"format": extension.lstrip("."), **(base_metadata or {})}

    for chart_index, chart in enumerate(_odf_chart_nodes(root), start=1):
        chart_part = f"{source_part}#chart{chart_index}"
        if chart_part in seen:
            continue
        seen.add(chart_part)
        elements.extend(_odf_chart_elements_from_chart(
            chart,
            chart_index=chart_index,
            chart_part=chart_part,
            source_part=source_part,
            location_prefix=location_prefix,
            base_metadata=metadata,
        ))

    for object_index, obj in enumerate(root.findall(".//{*}object"), start=1):
        href = _attr_by_local_name(obj, "href")
        chart_part = _odf_chart_object_part(content, source_part, href)
        if not chart_part or chart_part in seen:
            continue
        object_root = _read_zip_xml(content, chart_part)
        if object_root is None:
            continue
        charts = _odf_chart_nodes(object_root)
        if not charts:
            continue
        seen.add(chart_part)
        for nested_index, chart in enumerate(charts, start=1):
            part_label = chart_part if len(charts) == 1 else f"{chart_part}#chart{nested_index}"
            elements.extend(_odf_chart_elements_from_chart(
                chart,
                chart_index=len(seen),
                chart_part=part_label,
                source_part=chart_part,
                location_prefix=f"{location_prefix} object {object_index}",
                base_metadata=metadata,
            ))
    return elements


def _extract_opendocument(content: bytes, extension: str) -> list[DocumentElement]:
    root = _read_zip_xml(content, "content.xml")
    if root is None:
        return _extract_plain_text(content, ".txt")

    elements: list[DocumentElement] = _odf_metadata_elements(content)
    seen_chart_parts: set[str] = set()
    odf_form_definitions = _odf_form_control_definitions(root)
    odf_style_definitions = _odf_style_definitions(content, root)
    odf_bullet_levels = _odf_paragraph_bullet_levels(root)
    odf_sections = _odf_text_section_map(root)
    seen_form_controls: set[str] = set()

    if extension != ".odp":
        for table_index, table in enumerate(root.findall(".//{*}table"), start=1):
            table_name = _attr_by_local_name(table, "name")
            rows, formulas_by_row = _odf_table_data(table)
            table_elements = _table_elements_from_rows(
                rows,
                location_prefix=f"{table_name or 'table'} {table_index}",
                table=table_index,
                base_metadata={"table_name": table_name, "format": extension.lstrip(".")},
            )
            elements.extend(_odf_table_cell_elements(
                table,
                table_index=table_index,
                table_name=table_name,
                location_prefix=f"{table_name or 'table'} {table_index}",
                base_metadata={"format": extension.lstrip(".")},
                style_definitions=odf_style_definitions,
            ))
            for element in table_elements:
                formulas = formulas_by_row.get(int(element.metadata.get("row", 0) or 0))
                if formulas:
                    element.metadata["formulas"] = formulas
                elements.append(element)
        elements.extend(_odf_chart_elements(
            content,
            root,
            extension=extension,
            location_prefix="odf",
            seen_chart_parts=seen_chart_parts,
        ))
        elements.extend(_odf_form_control_elements(
            root,
            extension=extension,
            location_prefix="odf",
            definitions=odf_form_definitions,
            seen_controls=seen_form_controls,
        ))
        elements.extend(_odf_shape_elements(
            root,
            extension=extension,
            location_prefix="odf",
        ))

    if extension == ".odp":
        for slide_index, page in enumerate(root.findall(".//{*}page"), start=1):
            page_name = _attr_by_local_name(page, "name") or f"Slide {slide_index}"
            raw_visibility = _clean_text(str(_attr_by_local_name(page, "visibility") or "visible")) or "visible"
            visibility = "hidden" if raw_visibility.lower() in {"0", "false", "hidden", "no", "off"} else raw_visibility
            slide_metadata = {
                "slide": slide_index,
                "slide_title": page_name,
                "slide_id": page_name,
                "slide_part": "content.xml",
                "slide_state": visibility,
                "slide_visibility": visibility,
                "format": extension.lstrip("."),
            }
            slide_metadata_element = _presentation_slide_metadata_element(slide_metadata)
            if slide_metadata_element:
                elements.append(slide_metadata_element)
            paragraphs = [
                (paragraph, _odf_visible_text(paragraph))
                for paragraph in page.findall(".//{*}p")
            ]
            for paragraph_index, (paragraph, paragraph_text) in enumerate([
                (paragraph, text)
                for paragraph, text in paragraphs
                if text
            ], start=1):
                element_type = "slide_title" if paragraph_index == 1 and len(paragraph_text) < 160 else "slide_text"
                style_metadata = _odf_text_style_metadata(
                    paragraph,
                    odf_style_definitions,
                    bullet_levels=odf_bullet_levels.get(id(paragraph)),
                )
                element = _as_element(
                    element_type,
                    paragraph_text,
                    location=f"slide {slide_index} paragraph {paragraph_index}",
                    **slide_metadata,
                    **style_metadata,
                )
                if element:
                    elements.append(element)
            for table_index, table in enumerate(page.findall(".//{*}table"), start=1):
                table_name = _attr_by_local_name(table, "name")
                rows, formulas_by_row = _odf_table_data(table)
                table_elements = _table_elements_from_rows(
                    rows,
                    location_prefix=f"slide {slide_index} {table_name or 'table'} {table_index}",
                    table=table_index,
                    base_metadata={
                        "table_name": table_name,
                        **slide_metadata,
                    },
                )
                elements.extend(_odf_table_cell_elements(
                    table,
                    table_index=table_index,
                    table_name=table_name,
                    location_prefix=f"slide {slide_index} {table_name or 'table'} {table_index}",
                    base_metadata=slide_metadata,
                    style_definitions=odf_style_definitions,
                ))
                for element in table_elements:
                    formulas = formulas_by_row.get(int(element.metadata.get("row", 0) or 0))
                    if formulas:
                        element.metadata["formulas"] = formulas
                    elements.append(element)
            for image_index, image in enumerate(page.findall(".//{*}image"), start=1):
                href = _attr_by_local_name(image, "href")
                title = _xml_text(image, ("title", "desc"))
                image_part = None
                if href and not re.match(r"^[a-z][a-z0-9+.-]*:", href, flags=re.I):
                    image_part = _resolve_part_target("content.xml", href)
                image_metadata = _zip_image_metadata(content, image_part)
                element = _as_element(
                    "image",
                    _image_element_text(title, image_part or href, metadata=image_metadata),
                    location=f"slide {slide_index} image {image_index}",
                    href=href,
                    image_part=image_part,
                    **slide_metadata,
                    ocr_source="odp_image" if image_metadata.get("ocr_text") else None,
                    **image_metadata,
                )
                if element:
                    elements.append(element)
                ocr_element = _image_ocr_text_element(
                    image_metadata,
                    location=f"slide {slide_index} image {image_index} OCR",
                    image_part=image_part,
                    ocr_source="odp_image",
                    base_metadata={
                        **slide_metadata,
                        "href": href,
                        "title": title,
                    },
                )
                if ocr_element:
                    elements.append(ocr_element)
            elements.extend(_odf_link_elements(
                page,
                extension=extension,
                location_prefix=f"slide {slide_index}",
                base_metadata=slide_metadata,
            ))
            elements.extend(_odf_chart_elements(
                content,
                page,
                extension=extension,
                location_prefix=f"slide {slide_index}",
                base_metadata=slide_metadata,
                seen_chart_parts=seen_chart_parts,
            ))
            elements.extend(_odf_form_control_elements(
                page,
                extension=extension,
                location_prefix=f"slide {slide_index}",
                base_metadata=slide_metadata,
                definitions=odf_form_definitions,
                seen_controls=seen_form_controls,
            ))
            elements.extend(_odf_shape_elements(
                page,
                extension=extension,
                location_prefix=f"slide {slide_index}",
                base_metadata=slide_metadata,
            ))
        elements.extend(_odf_revision_elements(root, extension=extension))
        elements.extend(_odf_annotation_elements(root, extension=extension))
        return elements

    for heading_index, heading in enumerate(root.findall(".//{*}h"), start=1):
        style_metadata = _odf_text_style_metadata(
            heading,
            odf_style_definitions,
            bullet_levels=odf_bullet_levels.get(id(heading)),
        )
        element = _as_element(
            "heading",
            _odf_visible_text(heading),
            location=f"heading {heading_index}",
            level=_attr_by_local_name(heading, "outline-level"),
            format=extension.lstrip("."),
            **style_metadata,
        )
        if element:
            elements.append(element)

    for paragraph_index, paragraph in enumerate(root.findall(".//{*}p"), start=1):
        text = _odf_visible_text(paragraph)
        style_metadata = _odf_text_style_metadata(
            paragraph,
            odf_style_definitions,
            bullet_levels=odf_bullet_levels.get(id(paragraph)),
        )
        element = _as_element(
            "paragraph",
            text,
            location=f"paragraph {paragraph_index}",
            format=extension.lstrip("."),
            section=odf_sections.get(id(paragraph)),
            **style_metadata,
        )
        if element:
            elements.append(element)

    elements.extend(_odf_link_elements(root, extension=extension))
    elements.extend(_odf_annotation_elements(root, extension=extension))
    elements.extend(_odf_revision_elements(root, extension=extension))
    elements.extend(_odf_image_elements(content, root, extension=extension))
    elements.extend(_odf_form_control_elements(
        root,
        extension=extension,
        location_prefix="odf",
        definitions=odf_form_definitions,
        seen_controls=seen_form_controls,
    ))
    elements.extend(_odf_shape_elements(
        root,
        extension=extension,
        location_prefix="odf",
    ))

    return elements


def _first_int(value: str) -> int | None:
    match = re.search(r"(\d+)", value)
    return int(match.group(1)) if match else None


def _extract_elements(filename: str, content: bytes, content_type: str | None) -> tuple[list[DocumentElement], list[str], str]:
    extension = Path(filename or "").suffix.lower()
    warnings: list[str] = []
    method = extension.lstrip(".") or "bytes"
    normalized_content_type = (content_type or "").lower()

    try:
        if extension in IMAGE_MIME_BY_EXTENSION or normalized_content_type.startswith("image/"):
            return _extract_image_file(content, filename or f"image{extension or ''}", content_type), warnings, "image_file"
        if extension in {".txt", ".text", ".log", ".rst"}:
            return _extract_plain_text(content, extension), warnings, "plain_text"
        if extension in {".md", ".markdown"}:
            return _extract_plain_text(content, extension), warnings, "markdown_structure"
        if extension in {".html", ".htm"} or (content_type or "").startswith("text/html"):
            return _extract_html(content), warnings, "html_structure"
        if extension == ".csv":
            return _extract_csv(content), warnings, "csv_table"
        if extension == ".json":
            return _extract_json(content), warnings, "json_tree"
        if extension == ".pdf":
            return _extract_pdf(content), warnings, "pdf_blocks"
        if extension == ".epub":
            return _extract_epub(content), warnings, "epub_html"
        if extension == ".eml" or normalized_content_type in {"message/rfc822", "application/eml"}:
            return _extract_email(content), warnings, "email_message"
        if extension in OOXML_WORD_EXTENSIONS or normalized_content_type in OOXML_WORD_MIME_TYPES:
            return _extract_docx(content), warnings, "docx_xml"
        if extension in OOXML_PRESENTATION_EXTENSIONS or normalized_content_type in OOXML_PRESENTATION_MIME_TYPES:
            return _extract_pptx(content), warnings, "pptx_xml"
        if extension in OOXML_SPREADSHEET_EXTENSIONS or normalized_content_type in OOXML_SPREADSHEET_MIME_TYPES:
            return _extract_xlsx(content), warnings, "xlsx_xml"
        if extension in LEGACY_OFFICE_EXTENSIONS or normalized_content_type in LEGACY_OFFICE_MIME_TYPES:
            elements, legacy_warnings = _extract_legacy_office(content, filename, content_type, extension)
            warnings.extend(legacy_warnings)
            return elements, warnings, "legacy_office_ole"
        if extension in {".odt", ".ods", ".odp"}:
            return _extract_opendocument(content, extension), warnings, f"{extension.lstrip('.')}_xml"
        if extension == ".zip" or normalized_content_type in {"application/zip", "application/x-zip-compressed"}:
            return _extract_zip_archive(content), warnings, "archive_bundle"
        if extension == ".rtf":
            text = re.sub(r"\\[a-z]+-?\d* ?", " ", _decode_bytes(content))
            text = re.sub(r"[{}]", " ", text)
            return _extract_plain_text(text.encode("utf-8"), ".txt"), warnings, "rtf_text"
    except Exception as exc:
        warnings.append(f"{method} extraction failed: {exc}")

    decoded = _decode_bytes(content)
    return _extract_plain_text(decoded.encode("utf-8"), ".txt"), warnings, "fallback_text"


def _element_chunk_scope(element: DocumentElement) -> tuple[str, str, str]:
    metadata = element.metadata if isinstance(element.metadata, dict) else {}

    archive_entry = _clean_text(str(metadata.get("archive_entry_path") or metadata.get("archive_entry_filename") or ""))
    attachment = _clean_text(str(metadata.get("embedded_file_filename") or metadata.get("embedded_file_name") or ""))
    chart_source = _clean_text(str(metadata.get("chart_source_part") or metadata.get("chart_source_uri") or ""))

    if archive_entry:
        base_type = "archive_entry"
        base_key = f"archive_entry:{archive_entry}"
        base_label = f"archive entry {archive_entry}"
    elif attachment:
        base_type = "attachment"
        base_key = f"attachment:{attachment}"
        base_label = f"attachment {attachment}"
    elif chart_source:
        base_type = "chart_source"
        base_key = f"chart_source:{chart_source}"
        base_label = f"chart source {chart_source}"
    elif metadata.get("legacy_office_stream_path"):
        stream_path = _clean_text(str(metadata.get("legacy_office_stream_path") or ""))
        base_type = "legacy_office_stream"
        base_key = f"legacy_office_stream:{stream_path}"
        base_label = f"legacy Office stream {stream_path}" if stream_path else "legacy Office stream"
    elif metadata.get("slide") not in (None, ""):
        slide = str(metadata.get("slide"))
        slide_title = _clean_text(str(metadata.get("slide_title") or ""))
        base_type = "slide"
        base_key = f"slide:{slide}"
        base_label = f"Slide {slide}: {slide_title}" if slide_title else f"slide {slide}"
    elif metadata.get("sheet_name"):
        sheet = _clean_text(str(metadata.get("sheet_name") or ""))
        base_type = "sheet"
        base_key = f"sheet:{sheet}"
        base_label = sheet
    elif metadata.get("page") not in (None, ""):
        page = str(metadata.get("page"))
        base_type = "page"
        base_key = f"page:{page}"
        base_label = f"page {page}"
    elif metadata.get("chapter_path"):
        chapter = _clean_text(str(metadata.get("chapter_path") or ""))
        base_type = "chapter"
        base_key = f"chapter:{chapter}"
        base_label = chapter
    elif metadata.get("section"):
        section = _clean_text(str(metadata.get("section") or ""))
        base_type = "section"
        base_key = f"section:{section}"
        base_label = section
    elif element.type == "document_metadata":
        base_type = "document_metadata"
        base_key = "document_metadata"
        base_label = "document metadata"
    else:
        base_type = "document"
        base_key = "document"
        base_label = "document"

    if element.type in {"table_header", "table_row", "table_cell"}:
        table = metadata.get("table_name") or metadata.get("table")
        if table in (None, "") and base_type in {"archive_entry", "attachment"}:
            table = metadata.get("sheet_name") or metadata.get("sheet")
        if table not in (None, ""):
            table_label = str(table)
            return (
                "table",
                f"{base_key}:table:{table_label}",
                f"{base_label} table {table_label}",
            )

    if element.type in {"chart_axis", "chart_data_label", "chart_legend", "chart_series", "chart_point", "chart_trendline"}:
        chart = _clean_text(str(metadata.get("chart_part") or metadata.get("chart_source_part") or ""))
        if chart:
            return ("chart", f"{base_key}:chart:{chart}", f"{base_label} chart {chart}")

    if element.type == "image":
        image = _clean_text(str(metadata.get("image_part") or metadata.get("src") or ""))
        if image:
            return ("image", f"{base_key}:image:{image}", f"{base_label} image {image}")

    return base_type, base_key, base_label


def _element_evidence_families(element: DocumentElement) -> list[str]:
    metadata = element.metadata if isinstance(element.metadata, dict) else {}
    element_type = str(element.type or "").lower()
    searchable = " ".join(
        str(value)
        for value in (
            element_type,
            element.text[:900],
            *metadata.keys(),
            *metadata.values(),
        )
        if value not in (None, "")
    ).lower()
    families: list[str] = []

    def add(family: str, condition: bool) -> None:
        if condition and family not in families:
            families.append(family)

    add("connector", element_type == "shape_connector" or "connector" in searchable or "arrow" in searchable)
    add("diagram", element_type.startswith("diagram_") or "diagram" in searchable or "smartart" in searchable)
    add("shape", element_type in {"shape_group", "shape_metadata", "text_box"} or bool(metadata.get("group_id") or metadata.get("parent_group_id")))
    add("chart", element_type.startswith("chart_") or bool(metadata.get("chart_part") or metadata.get("chart_source_part")))
    add("table", element_type.startswith("table_") or bool(metadata.get("table_name") or metadata.get("table")))
    add("spreadsheet_layout", bool(metadata.get("heading_kind") == "spreadsheet_merged_cell" or metadata.get("merge_range")))
    add("formula", bool(metadata.get("formula_cell") or metadata.get("formula_expression") or "formula" in searchable))
    add("pivot_table", element_type == "pivot_table" or "pivot_table" in searchable or "pivot table" in searchable)
    add("conditional_format", element_type == "conditional_format" or "conditional_format" in searchable)
    add("code_block", element_type == "code_block" or bool(metadata.get("markdown_code_language")))
    add("styled_text", bool(metadata.get("styled_text") or metadata.get("style_summary") or metadata.get("emphasis_term") or "styled_text" in searchable))
    add("form_field", element_type == "form_field" or bool(metadata.get("form_kind") or metadata.get("field_type") or metadata.get("control_part")))
    add("link", element_type == "link" or bool(metadata.get("uri") or metadata.get("link_kind")))
    add("image", element_type in {"image", "image_ocr_text", "page_ocr_text"} or bool(metadata.get("image_part") or metadata.get("image_relationship_id")))
    add("ocr", bool(metadata.get("ocr_source") or metadata.get("ocr_engine") or "ocr" in searchable))
    add("speaker_notes", element_type == "speaker_notes" or bool(metadata.get("notes_part")))
    add("comment", element_type in {"comment", "annotation"} or bool(metadata.get("comment_id") or metadata.get("annotation_type")))
    add("revision", element_type == "revision" or bool(metadata.get("revision_id") or metadata.get("revision_type")))
    add("media", element_type == "media" or bool(metadata.get("media_part") or metadata.get("media_kind")))
    add("attachment", bool(metadata.get("embedded_file_filename") or metadata.get("embedded_file_name")))
    add("archive_entry", bool(metadata.get("archive_entry_path") or metadata.get("archive_entry_filename")))
    add("presentation_structure", bool(metadata.get("slide_part") or metadata.get("slide_visibility") or metadata.get("slide_state")))
    add("workbook_structure", bool(metadata.get("sheet_part") or metadata.get("sheet_visibility") or metadata.get("sheet_state")))
    add("layout", bool(metadata.get("layout_bbox") or metadata.get("layout_zone") or metadata.get("layout_position")))
    add("text", not families)
    return families


def _evidence_family_priority(family: str) -> int:
    normalized = str(family or "").lower()
    priorities = {
        "connector": 120,
        "chart": 112,
        "spreadsheet_layout": 110,
        "table": 106,
        "styled_text": 104,
        "form_field": 102,
        "diagram": 98,
        "revision": 90,
        "speaker_notes": 88,
        "comment": 84,
        "link": 80,
        "shape": 79,
        "formula": 78,
        "pivot_table": 76,
        "conditional_format": 74,
        "code_block": 72,
        "ocr": 70,
        "image": 66,
        "media": 62,
        "attachment": 58,
        "archive_entry": 54,
        "legacy_office": 50,
        "presentation_structure": 36,
        "workbook_structure": 34,
        "layout": 12,
        "text": 8,
    }
    return priorities.get(normalized, 40)


def _rank_evidence_families(families: list[str]) -> list[str]:
    unique = _unique_preserve_order([str(family) for family in families if family])
    original_index = {family: index for index, family in enumerate(unique)}
    return sorted(
        unique,
        key=lambda family: (-_evidence_family_priority(family), original_index[family]),
    )


def _element_location_hints(element: DocumentElement) -> list[str]:
    metadata = element.metadata if isinstance(element.metadata, dict) else {}
    hints: list[str] = []

    def add(label: str, value: Any, *, clip: int = 96) -> None:
        text = _clean_text(str(value or ""))
        hint = _clip(f"{label} {text}", clip) if text else ""
        if hint and hint not in hints:
            hints.append(hint)

    slide = metadata.get("slide")
    slide_title = _clean_text(str(metadata.get("slide_title") or ""))
    if slide not in (None, ""):
        add("slide", f"{slide}: {slide_title}" if slide_title else slide)
    add("page", metadata.get("page"))
    add("sheet", metadata.get("sheet_name") or metadata.get("sheet"))
    add("table", metadata.get("table_name") or metadata.get("table"))
    add("chart", metadata.get("chart_title") or metadata.get("chart_part") or metadata.get("chart_source_part"))
    add("cell", metadata.get("cell") or metadata.get("formula_cell") or metadata.get("table_cell_reference"))
    add("cell context", metadata.get("table_cell_context"))
    add("diagram", metadata.get("diagram_part") or metadata.get("diagram_node_id"))
    add("group", metadata.get("group_label") or metadata.get("parent_group_label"))
    add("form", metadata.get("label") or metadata.get("field_type") or metadata.get("form_kind"))
    add("image", metadata.get("image_part") or metadata.get("image_relationship_id"))
    add("source", metadata.get("chart_source_part") or metadata.get("embedded_file_filename") or metadata.get("archive_entry_path"))
    add("layout", metadata.get("layout_bbox") or metadata.get("layout_zone"))
    add("style", metadata.get("style_summary"))
    add("uri", metadata.get("uri"), clip=120)
    add("visibility", metadata.get("slide_visibility") or metadata.get("sheet_visibility") or metadata.get("slide_state") or metadata.get("sheet_state"))
    return hints


def _location_hint_priority(hint: str) -> int:
    normalized = _clean_text(hint).lower()
    if normalized.startswith("cell context "):
        return 0
    if normalized.startswith(("form ", "chart ", "table ", "cell ", "diagram ", "group ", "image ", "source ")):
        return 1
    if normalized.startswith(("style ", "uri ", "visibility ")):
        return 2
    if normalized.startswith(("slide ", "page ", "sheet ")):
        return 3
    if normalized.startswith("layout "):
        return 4
    return 5


def _compact_location_hints(hints: list[str], limit: int = 12) -> list[str]:
    unique = _unique_preserve_order(hints)
    if len(unique) <= limit:
        return unique

    primary = unique[0]
    ranked_tail = sorted(
        enumerate(unique[1:], start=1),
        key=lambda item: (_location_hint_priority(item[1]), item[0]),
    )
    compact = [primary]
    for _, hint in ranked_tail:
        if hint not in compact:
            compact.append(hint)
        if len(compact) >= limit:
            break
    return compact


def _source_location_record_priority(record: dict[str, Any]) -> int:
    priority = 0
    high_signal_keys = (
        "revision_type",
        "revision_author",
        "revision_id",
        "revision_change_kind",
        "heading_kind",
        "merge_range",
        "merge_anchor_cell",
        "formula_cell",
        "formula_expression",
        "formula_result",
        "cell_value_type",
        "conditional_range",
        "conditional_range_label",
        "conditional_type",
        "conditional_operator",
        "conditional_formula1",
        "conditional_formula2",
        "conditional_formula_references",
        "validation_range",
        "validation_formula1",
        "validation_type",
        "table_columns",
        "table_part",
        "pivot_table_name",
        "pivot_table_cache_part",
        "pivot_table_source_ref",
        "named_range_name",
        "named_range_ref",
        "sheet_part",
        "sheet_visibility",
        "sheet_state",
        "table_cell_style_name",
        "table_cell_width_value",
        "table_cell_grid_span",
        "table_cell_fill_color",
        "table_cell_visual_summary",
        "table_cell_width_inches",
        "table_cell_border_top_color",
        "comment_anchor_text",
        "comment_anchor_cell",
        "comment_anchor_header",
        "chart_point_label",
        "chart_series_visual_summary",
        "chart_point_visual_summary",
        "connector_id",
        "source_shape_id",
        "target_shape_id",
    )
    for key in high_signal_keys:
        if record.get(key) not in (None, "", [], {}):
            priority += 6
    element_type = str(record.get("element_type") or "").lower()
    if element_type in {"revision", "formula", "table_cell", "chart_point", "shape_connector", "comment"}:
        priority += 8
    if record.get("table_cell_reference"):
        priority += 3
    if record.get("metadata_field"):
        priority -= 4
    return priority


def _unique_source_location_records(records: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not record:
            continue
        compact = {
            key: value
            for key, value in record.items()
            if value not in (None, "", [], {})
        }
        if not compact:
            continue
        key = json.dumps(compact, sort_keys=True, ensure_ascii=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(compact)
    if len(unique) <= limit:
        return unique

    primary = unique[0]
    ranked_tail = sorted(
        enumerate(unique[1:], start=1),
        key=lambda item: (-_source_location_record_priority(item[1]), item[0]),
    )
    compact = [primary]
    for _, record in ranked_tail:
        if record not in compact:
            compact.append(record)
        if len(compact) >= limit:
            break
    return compact


def _provenance_line_priority(line: str) -> int:
    normalized = _clean_text(line).lower()
    priority = 0
    high_signal_tokens = (
        "revision_type=",
        "revision_author=",
        "revision_id=",
        "revision_change_kind=",
        "heading_kind=spreadsheet_merged_cell",
        "merge_range=",
        "formula_cell=",
        "formula_expression=",
        "formula_references=",
        "cell_value_type=",
        "validation_range=",
        "validation_formula1=",
        "conditional_range=",
        "conditional_type=",
        "conditional_formula1=",
        "conditional_formula_references=",
        "table_cell_reference=",
        "table_cell_grid_span=",
        "table_cell_style_name=",
        "table_cell_width_value=",
        "table_cell_width_inches=",
        "table_cell_border_top_color=",
        "table_columns=",
        "table_part=",
        "pivot_table_name=",
        "pivot_table_cache_part=",
        "pivot_table_source_ref=",
        "named_range_name=",
        "named_range_ref=",
        "sheet_part=",
        "sheet_visibility=",
        "sheet_state=",
        "comment_anchor_text=",
        "comment_anchor_cell=",
        "chart_point_label=",
        "chart_series_visual_summary=",
        "chart_point_visual_summary=",
        "connector_id=",
        "source_shape_id=",
        "target_shape_id=",
    )
    for token in high_signal_tokens:
        if token in normalized:
            priority += 5
    if "element_type=document_metadata" in normalized or "metadata_field=" in normalized:
        priority -= 3
    return priority


def _compact_provenance_lines(lines: list[str], *, limit: int = 32) -> list[str]:
    unique = _unique_preserve_order([line for line in lines if _clean_text(line)])
    if len(unique) <= limit:
        return unique
    primary = unique[0]
    ranked_tail = sorted(
        enumerate(unique[1:], start=1),
        key=lambda item: (-_provenance_line_priority(item[1]), item[0]),
    )
    compact = [primary]
    for _, line in ranked_tail:
        if line not in compact:
            compact.append(line)
        if len(compact) >= limit:
            break
    return compact


def _build_chunks(
    elements: list[DocumentElement],
    *,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    respect_semantic_boundaries: bool = True,
) -> list[DocumentChunkPreview]:
    chunks: list[DocumentChunkPreview] = []
    current_parts: list[str] = []
    current_preview_parts: list[str] = []
    source_elements: list[str] = []
    current_provenance: list[str] = []
    current_evidence_families: list[str] = []
    current_location_hints: list[str] = []
    current_source_locations: list[dict[str, Any]] = []
    current_scope: tuple[str, str, str] | None = None
    current_scopes: dict[str, tuple[str, str, str]] = {}
    current_len = 0

    def flush(*, keep_overlap: bool = True) -> None:
        nonlocal current_parts, current_preview_parts, source_elements, current_provenance, current_evidence_families, current_location_hints, current_source_locations, current_scope, current_scopes, current_len
        if not current_parts:
            return
        text = "\n".join(current_parts).strip()
        if not text:
            return
        chunk_id = f"chunk_{len(chunks):04d}"
        metadata = {"char_count": len(text)}
        semantic_text = "\n".join(current_preview_parts).strip()
        if semantic_text:
            metadata["semantic_text"] = _clip_multiline(semantic_text, MAX_SEMANTIC_CHUNK_TEXT_CHARS)
        scope_records = list(current_scopes.values())
        if len(scope_records) == 1:
            current_scope = scope_records[0]
            scope_type, scope_key, scope_label = current_scope
            metadata.update({
                "chunk_scope_type": scope_type,
                "chunk_scope_key": scope_key,
                "chunk_scope_label": scope_label,
            })
        elif len(scope_records) > 1:
            metadata.update({
                "chunk_scope_type": "mixed",
                "chunk_scope_keys": [scope[1] for scope in scope_records[:12]],
                "chunk_scope_labels": [scope[2] for scope in scope_records[:12]],
                "chunk_scope_count": len(scope_records),
            })
        evidence_preview = "\n".join(current_preview_parts).strip()
        if evidence_preview:
            metadata["evidence_preview"] = _clip(evidence_preview, MAX_SAMPLE_TEXT)
        if current_provenance:
            metadata["provenance"] = _compact_provenance_lines(current_provenance)
        evidence_families = _rank_evidence_families(current_evidence_families)
        if evidence_families:
            metadata["primary_evidence_family"] = evidence_families[0]
            metadata["evidence_families"] = evidence_families[:12]
        location_hints = _compact_location_hints(current_location_hints)
        if location_hints:
            metadata["primary_location_hint"] = location_hints[0]
            metadata["location_hints"] = location_hints
        source_locations = _unique_source_location_records(current_source_locations)
        if source_locations:
            metadata["source_locations"] = source_locations
            for key, value in source_locations[0].items():
                metadata.setdefault(key, value)
        chunks.append(DocumentChunkPreview(
            chunk_id=chunk_id,
            text=text,
            token_estimate=max(1, round(len(text) / 4)),
            source_elements=source_elements[:],
            metadata=metadata,
        ))
        if keep_overlap and overlap > 0 and len(text) > overlap:
            current_parts = [text[-overlap:]]
            current_preview_parts = current_preview_parts[-2:]
            source_elements = source_elements[-2:]
            current_provenance = current_provenance[-2:]
            current_evidence_families = current_evidence_families[-8:]
            current_location_hints = current_location_hints[-8:]
            current_source_locations = current_source_locations[-8:]
            current_len = len(current_parts[0])
        else:
            current_parts = []
            current_preview_parts = []
            source_elements = []
            current_provenance = []
            current_evidence_families = []
            current_location_hints = []
            current_source_locations = []
            current_scope = None
            current_scopes = {}
            current_len = 0

    for index, element in enumerate(elements):
        scope = _element_chunk_scope(element)
        if respect_semantic_boundaries and current_parts and current_scope and scope[1] != current_scope[1]:
            flush(keep_overlap=False)
        if not current_parts:
            current_scope = scope
            current_scopes = {scope[1]: scope}
        prefix = f"[{element.type}]"
        if element.location:
            prefix += f" {element.location}:"
        metadata = _metadata_line(element.metadata)
        metadata_suffix = f" ({metadata})" if metadata else ""
        piece = f"{prefix}{metadata_suffix} {element.text}"
        preview_piece = f"{prefix} {element.text}"
        if current_parts and current_len + len(piece) + 1 > max_chars:
            flush(keep_overlap=True)
            if not current_parts:
                current_scope = scope
                current_scopes = {scope[1]: scope}
        current_scopes.setdefault(scope[1], scope)
        current_parts.append(piece)
        current_preview_parts.append(preview_piece)
        source_elements.append(f"{element.type}:{index}")
        current_evidence_families.extend(_element_evidence_families(element))
        current_location_hints.extend(_element_location_hints(element))
        location_record = _relationship_location_hints({
            **(element.metadata or {}),
            "element_type": element.type,
            "location": element.location,
        })
        if location_record:
            current_source_locations.append(location_record)
        if metadata and metadata not in current_provenance:
            current_provenance.append(metadata)
        current_len += len(piece) + 1

    flush(keep_overlap=False)
    return chunks


def _candidate_entities(elements: list[DocumentElement], *, limit: int = 40) -> list[dict[str, Any]]:
    scores: Counter[str] = Counter()
    types: dict[str, str] = {}
    evidence: dict[str, str] = {}

    for element in elements:
        if element.type in {"heading", "slide_metadata", "slide_title", "sheet_metadata", "table_header", "table_definition", "pivot_table", "named_range", "conditional_format", "chart_series", "diagram_node", "shape_group", "shape_metadata", "text_box", "form_field", "outline_item"}:
            if element.type in {"shape_group", "shape_metadata", "text_box"}:
                metadata = element.metadata if isinstance(element.metadata, dict) else {}
                name = _clip(str(
                    metadata.get("group_label")
                    or metadata.get("group_title")
                    or metadata.get("shape_title")
                    or metadata.get("shape_description")
                    or metadata.get("group_description")
                    or metadata.get("shape_name")
                    or metadata.get("group_name")
                    or element.text
                ), 120)
            elif element.type == "table_definition":
                metadata = element.metadata if isinstance(element.metadata, dict) else {}
                name = _clip(str(metadata.get("table_display_name") or metadata.get("table_name") or element.text), 120)
            elif element.type == "pivot_table":
                metadata = element.metadata if isinstance(element.metadata, dict) else {}
                name = _clip(str(metadata.get("pivot_table_name") or element.text), 120)
            elif element.type == "sheet_metadata":
                metadata = element.metadata if isinstance(element.metadata, dict) else {}
                name = _clip(str(metadata.get("sheet_name") or element.text), 120)
            elif element.type == "slide_metadata":
                metadata = element.metadata if isinstance(element.metadata, dict) else {}
                slide = metadata.get("slide")
                title = _clean_text(str(metadata.get("slide_title") or ""))
                name = _clip(f"Slide {slide}: {title}" if title else f"Slide {slide}", 120)
            elif element.type == "named_range":
                metadata = element.metadata if isinstance(element.metadata, dict) else {}
                name = _clip(str(metadata.get("named_range_name") or element.text), 120)
            elif element.type == "conditional_format":
                metadata = element.metadata if isinstance(element.metadata, dict) else {}
                name = _clip(str(metadata.get("conditional_label") or element.text), 120)
            else:
                name = _clip(element.text, 120)
            if 2 < len(name) < 120:
                scores[name] += 4
                types[name] = (
                    "section" if element.type == "heading"
                    else "section" if element.type == "outline_item"
                    else "slide" if element.type == "slide_title"
                    else "slide" if element.type == "slide_metadata"
                    else "sheet" if element.type == "sheet_metadata"
                    else "metric" if element.type == "chart_series"
                    else "diagram" if element.type == "diagram_node"
                    else "shape" if element.type in {"shape_group", "shape_metadata", "text_box"}
                    else "field" if element.type == "form_field"
                    else "named_range" if element.type == "named_range"
                    else "conditional_format" if element.type == "conditional_format"
                    else "pivot_table" if element.type == "pivot_table"
                    else "table"
                )
                evidence.setdefault(name, element.text)

        if element.type == "document_metadata":
            field = str(element.metadata.get("metadata_field") or "")
            value = str(element.metadata.get("metadata_value") or "")
            entity_type = (
                "document" if field == "title"
                else "person" if field in {"author", "creator", "last_modified_by", "manager"}
                else "organization" if field in {"company", "producer"}
                else "topic" if field in {"subject", "category", "keywords"}
                else ""
            )
            if entity_type:
                for name in _metadata_values(field, value):
                    if 2 < len(name) < 120:
                        scores[name] += 4
                        types[name] = entity_type
                        evidence.setdefault(name, element.text)

        for match in ENTITY_PHRASE_PATTERN.finditer(element.text):
            candidate = _clean_entity_phrase(match.group(0))
            if candidate.lower() in ENTITY_STOPWORDS_LOWER or len(candidate) < 3 or len(candidate) > 90:
                continue
            scores[candidate] += 1
            types.setdefault(candidate, "concept")
            evidence.setdefault(candidate, element.text)

    return [
        {
            "name": name,
            "type": types.get(name, "concept"),
            "score": min(1.0, round(0.35 + count * 0.08, 3)),
            "mentions": count,
            "evidence": _clip(evidence.get(name, ""), 240),
        }
        for name, count in scores.most_common(limit)
    ]


SEMANTIC_RELATION_VERBS: tuple[tuple[str, str], ...] = (
    ("drives?", "drives"),
    ("references?|refers\\s+to", "references"),
    ("supports?", "supports"),
    ("blocks?", "blocks"),
    ("gates?", "gates"),
    ("unblocks?", "unblocks"),
    ("owns?", "owns"),
    ("uses?", "uses"),
    ("calculates?", "calculates"),
    ("influences?", "influences"),
    ("causes?", "causes"),
    ("depends\\s+on", "depends_on"),
    ("maps?\\s+to", "maps_to"),
    ("links?\\s+to|connects?\\s+to", "connects_to"),
)
DIRECT_SEMANTIC_RELATIONS = {relation for _, relation in SEMANTIC_RELATION_VERBS}

SEMANTIC_RELATION_VERB_PATTERN = "|".join(f"(?:{pattern})" for pattern, _ in SEMANTIC_RELATION_VERBS)
SEMANTIC_RELATION_SOURCE_PATTERN = r"[A-Z][A-Za-z0-9&/+-]*(?:\s+(?:[A-Z][A-Za-z0-9&/+-]*|[a-z][a-z0-9&/+-]+)){0,5}"
SEMANTIC_RELATION_TARGET_PATTERN = r"[A-Z][A-Za-z0-9&/+-]*(?:\s+(?:[A-Z][A-Za-z0-9&/+-]*|[a-z][a-z0-9&/+-]+)){0,5}"
SEMANTIC_RELATION_PATTERN = re.compile(
    rf"\b(?P<source>{SEMANTIC_RELATION_SOURCE_PATTERN})\s+(?P<verb>(?i:{SEMANTIC_RELATION_VERB_PATTERN}))\s+(?P<target>{SEMANTIC_RELATION_TARGET_PATTERN})\b",
)
SEMANTIC_RELATION_ELEMENT_TYPES = {
    "paragraph",
    "slide_text",
    "text_box",
    "page_text",
    "table_row",
    "table_cell",
    "speaker_notes",
    "comment",
    "annotation",
    "email_body",
    "image_ocr_text",
    "page_ocr_text",
}
SEMANTIC_TABLE_SOURCE_HEADERS = {
    "source",
    "subject",
    "from",
    "head",
    "entity",
    "source_entity",
    "start",
    "origin",
}
SEMANTIC_TABLE_RELATION_HEADERS = {
    "relationship",
    "relation",
    "predicate",
    "edge",
    "link",
    "verb",
    "type",
    "relationship_type",
    "relation_type",
}
SEMANTIC_TABLE_TARGET_HEADERS = {
    "target",
    "object",
    "to",
    "tail",
    "destination",
    "target_entity",
    "end",
}
SEMANTIC_TABLE_EVIDENCE_HEADERS = {
    "evidence",
    "support",
    "supports",
    "rationale",
    "reason",
    "notes",
    "note",
    "description",
    "context",
    "proof",
}


def _semantic_relation_label(verb: str) -> str:
    normalized = _clean_text(verb).lower()
    for pattern, relation in SEMANTIC_RELATION_VERBS:
        if re.fullmatch(pattern, normalized, flags=re.I):
            return relation
    return normalized.replace(" ", "_")


def _semantic_relation_endpoint(value: str) -> str:
    text = _clean_entity_phrase(value)
    return _clip(text, 160)


def _is_semantic_relation_endpoint(value: str) -> bool:
    text = _clean_text(value)
    if len(text) < 3 or len(text) > 160:
        return False
    if text.lower() in ENTITY_STOPWORDS_LOWER:
        return False
    if re.fullmatch(r"[$€£]?\s*-?\d+(?:[.,]\d+)*(?:\s*%|\s*(?:k|m|b|mm|bn))?", text, flags=re.I):
        return False
    return bool(re.search(r"[A-Za-z]", text))


def _semantic_verb_relationships(element: DocumentElement, *, limit: int = 8) -> list[GraphCandidateRelationship]:
    if element.type not in SEMANTIC_RELATION_ELEMENT_TYPES:
        return []

    relationships: list[GraphCandidateRelationship] = []
    seen: set[tuple[str, str, str]] = set()
    sentences = re.split(r"(?<=[.!?;])\s+", element.text)
    for sentence in sentences:
        sentence = _clean_text(sentence)
        if not sentence:
            continue
        for match in SEMANTIC_RELATION_PATTERN.finditer(sentence):
            source = _semantic_relation_endpoint(match.group("source"))
            relation = _semantic_relation_label(match.group("verb"))
            target = _semantic_relation_endpoint(match.group("target"))
            if not (
                _is_semantic_relation_endpoint(source)
                and _is_semantic_relation_endpoint(target)
                and source.lower() != target.lower()
            ):
                continue
            key = (source.lower(), relation, target.lower())
            if key in seen:
                continue
            seen.add(key)
            relationships.append(GraphCandidateRelationship(
                source=source,
                relation=relation,
                target=target,
                evidence=_clip(sentence, 240),
                confidence=0.76,
                metadata=_relationship_metadata(
                    element,
                    semantic_relation_extraction="verb_pattern",
                    relation_verb=_clean_text(match.group("verb")).lower(),
                ),
            ))
            if len(relationships) >= limit:
                return relationships
    return relationships


def _semantic_table_header_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _clean_text(str(value or "")).lower()).strip("_")


def _semantic_table_column(headers: list[Any], aliases: set[str]) -> int | None:
    for index, header in enumerate(headers):
        if _semantic_table_header_key(header) in aliases:
            return index
    return None


def _semantic_table_relation_label(value: Any) -> str:
    text = _clean_text(str(value or ""))
    normalized_for_verb = re.sub(r"[_/-]+", " ", text)
    semantic_label = _semantic_relation_label(normalized_for_verb)
    return _relation_label(semantic_label or text)


def _semantic_table_row_relationship(element: DocumentElement) -> GraphCandidateRelationship | None:
    if element.type != "table_row":
        return None

    metadata = element.metadata if isinstance(element.metadata, dict) else {}
    headers = metadata.get("headers")
    cells = metadata.get("cells")
    if not isinstance(headers, list) or not isinstance(cells, list):
        return None

    source_index = _semantic_table_column(headers, SEMANTIC_TABLE_SOURCE_HEADERS)
    relation_index = _semantic_table_column(headers, SEMANTIC_TABLE_RELATION_HEADERS)
    target_index = _semantic_table_column(headers, SEMANTIC_TABLE_TARGET_HEADERS)
    if source_index is None or relation_index is None or target_index is None:
        return None

    def cell(index: int | None) -> str:
        if index is None or index >= len(cells):
            return ""
        return _clean_text(str(cells[index]))

    source = _semantic_relation_endpoint(cell(source_index))
    relation_text = cell(relation_index)
    target = _semantic_relation_endpoint(cell(target_index))
    if not (
        source
        and relation_text
        and target
        and _is_semantic_relation_endpoint(source)
        and _is_semantic_relation_endpoint(target)
        and source.lower() != target.lower()
    ):
        return None

    evidence_index = _semantic_table_column(headers, SEMANTIC_TABLE_EVIDENCE_HEADERS)
    evidence_text = cell(evidence_index)
    evidence = evidence_text or _clean_text(element.text)
    relation = _semantic_table_relation_label(relation_text)

    return GraphCandidateRelationship(
        source=source,
        relation=relation,
        target=target,
        evidence=_clip(evidence, 240),
        confidence=0.9,
        metadata=_relationship_metadata(
            element,
            semantic_relation_extraction="table_row_triple",
            table_fact_kind="semantic_row_triple",
            table_source="semantic_table_row",
            table_source_header=_clean_text(str(headers[source_index])),
            table_relation_header=_clean_text(str(headers[relation_index])),
            table_target_header=_clean_text(str(headers[target_index])),
            table_evidence_header=_clean_text(str(headers[evidence_index])) if evidence_index is not None else None,
            table_source_value=source,
            table_relation_value=relation_text,
            table_target_value=target,
            table_evidence_value=evidence_text or None,
        ),
    )


def _semantic_candidate_relationships(
    elements: list[DocumentElement],
    *,
    limit: int,
) -> list[GraphCandidateRelationship]:
    relationships: list[GraphCandidateRelationship] = []
    seen: set[tuple[str, str, str]] = set()

    def push(relationship: GraphCandidateRelationship | None) -> None:
        if relationship is None or len(relationships) >= limit:
            return
        key = (
            relationship.source.lower(),
            relationship.relation.lower(),
            relationship.target.lower(),
        )
        if key in seen:
            return
        seen.add(key)
        relationships.append(relationship)

    for element in elements:
        for relationship in _semantic_verb_relationships(element):
            push(relationship)
            if len(relationships) >= limit:
                return relationships
        push(_semantic_table_row_relationship(element))
        if len(relationships) >= limit:
            return relationships

    return relationships


def _relationship_metadata(element: DocumentElement, **extra: Any) -> dict[str, Any]:
    metadata = {"element_type": element.type, "location": element.location}
    for key in (
        "slide",
        "slide_title",
        "slide_id",
        "slide_part",
        "slide_relationship_id",
        "slide_state",
        "slide_visibility",
        "group_index",
        "group_id",
        "group_name",
        "group_title",
        "group_description",
        "group_label",
        "group_child_shape_ids",
        "group_child_labels",
        "group_child_count",
        "grouped_visual_label",
        "parent_group_id",
        "parent_group_label",
        "parent_group_index",
        "connector_index",
        "connector_id",
        "connector_title",
        "connector_description",
        "source_shape_id",
        "target_shape_id",
        "source_label",
        "target_label",
        "relation",
        "inherited_from",
        "inherited_part",
        "inherited_relationship_id",
        "layout_part",
        "layout_relationship_id",
        "master_part",
        "master_relationship_id",
        "layout_bbox",
        "layout_position",
        "layout_zone",
        "layout_x_inches",
        "layout_y_inches",
        "layout_width_inches",
        "layout_height_inches",
        "layout_center_x_inches",
        "layout_center_y_inches",
        "layout_left_pct",
        "layout_top_pct",
        "layout_width_pct",
        "layout_height_pct",
        "layout_center_x_pct",
        "layout_center_y_pct",
        "layout_x_points",
        "layout_y_points",
        "layout_width_points",
        "layout_height_points",
        "layout_center_x_points",
        "layout_center_y_points",
        "page_width_points",
        "page_height_points",
        "page_width_inches",
        "page_height_inches",
        "layout_rotation_degrees",
        "layout_flip_h",
        "layout_flip_v",
        "layout_x_emu",
        "layout_y_emu",
        "layout_width_emu",
        "layout_height_emu",
        "slide_width_emu",
        "slide_height_emu",
        "slide_width_inches",
        "slide_height_inches",
        "page",
        "chapter",
        "chapter_path",
        "outline_level",
        "outline_title",
        "style_summary",
        "emphasis_terms",
        "emphasis_term",
        "emphasis_styles",
        "bullet_levels",
        "styled_text",
        "metadata_part",
        "metadata_field",
        "metadata_label",
        "metadata_value",
        "markdown_code_language",
        "markdown_code_block_index",
        "markdown_start_line",
        "markdown_end_line",
        "markdown_frontmatter_index",
        "email_header",
        "email_message_id",
        "email_subject",
        "email_from",
        "email_sender",
        "email_to",
        "email_cc",
        "email_date",
        "email_part_index",
        "email_part_kind",
        "email_content_type",
        "email_attachment_index",
        "email_attachment_filename",
        "email_attachment_mime",
        "archive_entry_index",
        "archive_entry_path",
        "archive_entry_filename",
        "archive_entry_mime",
        "archive_entry_size_bytes",
        "archive_entry_compressed_size_bytes",
        "archive_entry_sha256",
        "archive_entry_extraction_method",
        "legacy_office_format",
        "legacy_office_stream_index",
        "legacy_office_stream_path",
        "legacy_office_stream_size_bytes",
        "legacy_office_stream_sha256",
        "legacy_office_stream_truncated",
        "legacy_office_extraction_method",
        "legacy_office_string_count",
        "section",
        "level",
        "heading_kind",
        "format",
        "sheet",
        "sheet_name",
        "sheet_id",
        "sheet_part",
        "sheet_relationship_id",
        "sheet_state",
        "sheet_visibility",
        "table_name",
        "table_display_name",
        "table_definition_name",
        "table_part",
        "table_range",
        "table_columns",
        "table_relationship_id",
        "table_cell_reference",
        "table_cell_column_header",
        "table_cell_row_header",
        "table_cell_context",
        "table_cell_grid_span",
        "table_cell_row_span",
        "table_cell_width_value",
        "table_cell_h_merge",
        "table_cell_v_merge",
        "table_cell_merge_flags",
        "table_cell_fill_color",
        "table_cell_visual_summary",
        "table_cell_style_name",
        "table_cell_text_anchor",
        "table_cell_vertical_align",
        "table_cell_text_direction",
        "table_cell_width_value",
        "table_cell_width_inches",
        "table_cell_width_type",
        "table_cell_shading_pattern",
        "table_cell_margin_left_inches",
        "table_cell_margin_right_inches",
        "table_cell_margin_top_inches",
        "table_cell_margin_bottom_inches",
        "table_cell_border_left_color",
        "table_cell_border_right_color",
        "table_cell_border_top_color",
        "table_cell_border_bottom_color",
        "table_cell_border_left_width_inches",
        "table_cell_border_right_width_inches",
        "table_cell_border_top_width_inches",
        "table_cell_border_bottom_width_inches",
        "table_cell_column_repeat",
        "table_cell_row_repeat",
        "table_cell_is_covered",
        "cell_value_type",
        "auto_filter_range",
        "table",
        "row",
        "column",
        "cell",
        "table_source",
        "table_bbox",
        "table_row_count",
        "table_column_count",
        "formulas",
        "conditional_label",
        "conditional_range",
        "conditional_range_label",
        "conditional_type",
        "conditional_operator",
        "conditional_priority",
        "conditional_dxf_id",
        "conditional_stop_if_true",
        "conditional_text",
        "conditional_time_period",
        "conditional_rank",
        "conditional_percent",
        "conditional_formula1",
        "conditional_formula2",
        "conditional_formulas",
        "conditional_formula_references",
        "conditional_color_scale",
        "formula_cell",
        "formula_expression",
        "formula_references",
        "formula_result",
        "formula_reference_count",
        "merge_range",
        "merge_anchor_cell",
        "merge_cell_count",
        "merge_row_count",
        "merge_column_count",
        "named_range_name",
        "named_range_ref",
        "named_range_scope",
        "named_range_local_sheet_id",
        "named_range_hidden",
        "named_range_comment",
        "pivot_table_name",
        "pivot_table_range",
        "pivot_table_source_name",
        "pivot_table_source_ref",
        "pivot_table_source_sheet",
        "pivot_table_row_fields",
        "pivot_table_column_fields",
        "pivot_table_page_fields",
        "pivot_table_data_fields",
        "pivot_table_cache_part",
        "pivot_table_part",
        "pivot_table_cache_fields",
        "pivot_table_cache_id",
        "pivot_table_relationship_id",
        "pivot_table_source_type",
        "pivot_table_cache_relationship_id",
        "pivot_table_data_field_count",
        "pivot_table_record_count",
        "chart_type",
        "chart_title",
        "chart_part",
        "chart_label",
        "chart_point_label",
        "chart_axis_index",
        "chart_axis_type",
        "chart_axis_id",
        "chart_axis_title",
        "chart_axis_position",
        "chart_axis_number_format",
        "chart_legend_position",
        "chart_legend_overlay",
        "chart_legend_series",
        "chart_label_fields",
        "chart_label_position",
        "chart_label_examples",
        "chart_label_number_format",
        "chart_series_fill_color",
        "chart_series_line_color",
        "chart_series_line_width_points",
        "chart_series_line_dash",
        "chart_series_visual_summary",
        "chart_marker_symbol",
        "chart_marker_size",
        "chart_marker_fill_color",
        "chart_marker_line_color",
        "chart_marker_line_width_points",
        "chart_point_fill_color",
        "chart_point_line_color",
        "chart_point_visual_summary",
        "chart_trendline_type",
        "chart_trendline_name",
        "chart_trendline_line_color",
        "chart_trendline_line_width_points",
        "chart_trendline_line_dash",
        "chart_trendline_display_equation",
        "chart_trendline_display_r_squared",
        "chart_trendline_visual_summary",
        "series_name",
        "category",
        "value",
        "chart_source_part",
        "chart_source_extraction_method",
        "chart_source_uri",
        "chart_source_relationship_id",
        "chart_source_relationship_type",
        "chart_source_target_mode",
        "chart_source_kind",
        "chart_source_mime",
        "chart_source_size_bytes",
        "chart_source_sha256",
        "chart_source_index",
        "chart_source_extraction_method",
        "diagram_part",
        "diagram_node_id",
        "diagram_level",
        "diagram_connection_type",
        "source_diagram_node_id",
        "target_diagram_node_id",
        "source_label",
        "target_label",
        "parent_label",
        "relation",
        "group_index",
        "group_id",
        "group_name",
        "group_title",
        "group_description",
        "group_label",
        "group_child_shape_ids",
        "group_child_labels",
        "group_child_count",
        "grouped_visual_label",
        "parent_group_id",
        "parent_group_label",
        "parent_group_index",
        "connector_index",
        "connector_id",
        "connector_name",
        "connector_title",
        "connector_description",
        "source_shape_id",
        "target_shape_id",
        "source_connection_index",
        "target_connection_index",
        "drawing_part",
        "image",
        "image_part",
        "image_relationship_id",
        "image_relationship_type",
        "image_target_mode",
        "image_width",
        "image_height",
        "image_mime",
        "image_size_bytes",
        "image_sha256",
        "ocr_source",
        "ocr_engine",
        "media_part",
        "media_kind",
        "media_mime",
        "media_size_bytes",
        "media_sha256",
        "embedded_file_index",
        "embedded_file_name",
        "embedded_file_filename",
        "embedded_file_description",
        "embedded_file_mime",
        "embedded_file_size_bytes",
        "embedded_file_sha256",
        "embedded_file_created",
        "embedded_file_modified",
        "embedded_file_extraction_method",
        "json_path",
        "relationship_type",
        "relationship_id",
        "target_mode",
        "form_kind",
        "field_type",
        "label",
        "value",
        "form_field_checked",
        "form_field_selected_option",
        "form_field_options",
        "form_field_option_values",
        "control_id",
        "control_name",
        "control_part",
        "form_field_alias",
        "form_field_tag",
        "form_field_id",
        "form_field_placeholder",
        "form_field_date_format",
        "form_field_date_full_value",
        "form_field_calendar",
        "form_field_lock",
        "control_id",
        "action",
        "anchor",
        "tooltip",
        "target_frame",
        "validation_range",
        "validation_type",
        "validation_operator",
        "validation_formula1",
        "validation_formula2",
        "validation_allow_blank",
        "validation_prompt_title",
        "validation_prompt",
        "validation_error_title",
        "validation_error",
        "control_name",
        "control_part",
        "control_shape_id",
        "control_show_as_icon",
        "control_properties",
        "uri",
        "link_kind",
        "display",
        "shape_index",
        "shape_id",
        "shape_name",
        "shape_title",
        "shape_description",
        "text_box_index",
        "text_box_part",
        "text_box_kind",
        "text_box_style",
        "href",
        "src",
        "cell",
        "column",
        "part_name",
        "comment_id",
        "comment_kind",
        "comment_part",
        "comment_anchor_text",
        "comment_anchor_section",
        "comment_anchor_paragraph",
        "comment_anchor_sheet",
        "comment_anchor_cell",
        "comment_anchor_header",
        "comment_anchor_value",
        "threaded_comment_id",
        "threaded_parent_id",
        "person_id",
        "comment_datetime",
        "author_id",
        "author",
        "notes_part",
        "notes_relationship_id",
        "note_id",
        "note_part",
        "note_anchor_text",
        "note_anchor_section",
        "note_anchor_paragraph",
        "revision_type",
        "revision_author",
        "revision_date",
        "revision_id",
        "revision_part",
        "revision_change_kind",
        "revision_anchor_text",
        "annotation_type",
        "annotation_author",
        "annotation_subject",
        "annotation_content",
        "annotation_name",
    ):
        value = element.metadata.get(key)
        if value is not None:
            metadata[key] = value
    metadata.update({key: value for key, value in extra.items() if value is not None})
    return metadata


def _element_graph_label(element: DocumentElement, *, container_label: str | None = None) -> str:
    metadata = element.metadata if isinstance(element.metadata, dict) else {}

    if element.type in {"heading", "slide_title", "slide_text", "text_box", "paragraph", "page_text", "image_ocr_text", "page_ocr_text", "json_value", "speaker_notes", "comment", "footnote", "endnote", "annotation", "outline_item", "revision", "email_body", "legacy_office_text"}:
        return _clip(element.text, 160)
    if element.type == "legacy_office_stream":
        return _clip(str(metadata.get("legacy_office_stream_path") or element.text), 160)
    if element.type == "document_metadata":
        return _clip(str(metadata.get("metadata_value") or element.text), 160)
    if element.type == "archive_entry":
        return _clip(str(metadata.get("archive_entry_path") or metadata.get("archive_entry_filename") or element.text), 160)
    if element.type == "sheet_metadata":
        return _clip(str(metadata.get("sheet_name") or element.text), 160)
    if element.type == "slide_metadata":
        slide = metadata.get("slide")
        title = _clean_text(str(metadata.get("slide_title") or ""))
        return _clip(f"Slide {slide}: {title}" if title else f"Slide {slide}", 160)
    if element.type == "form_field":
        return _clip(str(metadata.get("label") or element.text), 120)
    if element.type == "code_block":
        language = _clean_text(str(metadata.get("markdown_code_language") or "code"))
        block_index = metadata.get("markdown_code_block_index")
        first_line = next((line.strip() for line in element.text.splitlines() if line.strip()), "")
        label = f"{language} code block {block_index}" if block_index not in (None, "") else f"{language} code block"
        if first_line:
            label = f"{label}: {_clip(first_line, 80)}"
        return _clip(label, 160)
    if element.type == "conditional_format":
        return _clip(str(metadata.get("conditional_label") or metadata.get("conditional_range_label") or element.text), 160)
    if element.type == "formula":
        return _clip(str(metadata.get("formula_cell") or element.text), 160)
    if element.type == "table_definition":
        return _clip(str(metadata.get("table_display_name") or metadata.get("table_name") or element.text), 160)
    if element.type == "pivot_table":
        return _clip(str(metadata.get("pivot_table_name") or element.text), 160)
    if element.type == "named_range":
        return _clip(str(metadata.get("named_range_name") or element.text), 160)
    if element.type == "table_cell":
        cell_ref = _clean_text(str(metadata.get("table_cell_reference") or ""))
        return _clip(f"{cell_ref}: {element.text}" if cell_ref else element.text, 160)
    if element.type in {"table_header", "table_row"}:
        table = metadata.get("table")
        if container_label:
            return f"{container_label} table {table}" if table is not None else f"{container_label} table"
        return _clip(element.text, 160)
    if element.type == "chart_axis":
        axis_title = _clean_text(str(metadata.get("chart_axis_title") or ""))
        axis_type = _clean_text(str(metadata.get("chart_axis_type") or "axis"))
        return _clip(f"{axis_type.title()} axis: {axis_title}" if axis_title else f"{axis_type.title()} axis", 160)
    if element.type == "chart_legend":
        position = _clean_text(str(metadata.get("chart_legend_position") or ""))
        return _clip(f"Legend {position}".strip(), 160)
    if element.type == "chart_data_label":
        series_name = _clean_text(str(metadata.get("series_name") or ""))
        return _clip(f"{series_name} data labels" if series_name else element.text, 160)
    if element.type == "chart_trendline":
        series_name = _clean_text(str(metadata.get("series_name") or ""))
        trendline_type = _clean_text(str(metadata.get("chart_trendline_type") or "trendline"))
        return _clip(f"{series_name} {trendline_type} trendline".strip(), 160)
    if element.type in {"chart_series", "chart_point"}:
        return _clip(str(metadata.get("chart_part") or metadata.get("series_name") or element.text), 160)
    if element.type in {"diagram_node", "diagram_relationship"}:
        return _clip(str(metadata.get("target_label") or metadata.get("source_label") or element.text), 160)
    if element.type == "shape_group":
        return _clip(str(
            metadata.get("group_label")
            or metadata.get("group_title")
            or metadata.get("group_description")
            or metadata.get("group_name")
            or element.text
        ), 160)
    if element.type == "shape_metadata":
        return _clip(str(
            metadata.get("shape_title")
            or metadata.get("shape_description")
            or metadata.get("shape_name")
            or element.text
        ), 160)
    if element.type == "shape_connector":
        source_label = _clean_text(str(metadata.get("source_label") or ""))
        target_label = _clean_text(str(metadata.get("target_label") or ""))
        relation = _clean_text(str(metadata.get("relation") or "connects_to"))
        if source_label and target_label:
            return _clip(f"{source_label} -> {target_label}: {relation}", 160)
        return _clip(str(
            metadata.get("connector_title")
            or metadata.get("connector_description")
            or metadata.get("connector_name")
            or element.text
        ), 160)
    if element.type == "image":
        return _clip(str(
            metadata.get("title")
            or metadata.get("description")
            or metadata.get("name")
            or metadata.get("image_part")
            or element.text
        ), 160)
    if element.type in {"media", "embedded_object", "link"}:
        return _clip(str(
            metadata.get("uri")
            or metadata.get("embedded_file_filename")
            or metadata.get("embedded_file_name")
            or metadata.get("media_part")
            or element.text
        ), 160)
    return _clip(element.text, 160)


def _form_field_graph_value(metadata: dict[str, Any]) -> str:
    checked = metadata.get("form_field_checked")
    if isinstance(checked, bool):
        return "checked" if checked else "unchecked"
    if checked is not None:
        checked_text = _clean_text(str(checked)).lower()
        if checked_text in {"true", "1", "yes", "checked", "on"}:
            return "checked"
        if checked_text in {"false", "0", "no", "unchecked", "off"}:
            return "unchecked"

    for key in (
        "form_field_selected_option",
        "value",
        "form_field_date_full_value",
    ):
        value = _clean_text(str(metadata.get(key) or ""))
        if value:
            return _clip(value, 160)

    return ""


def _form_field_relationships(document_name: str, element: DocumentElement) -> list[GraphCandidateRelationship]:
    metadata = element.metadata if isinstance(element.metadata, dict) else {}
    label = _clean_text(str(
        metadata.get("label")
        or metadata.get("form_field_alias")
        or metadata.get("form_field_tag")
        or ""
    ))
    if not label:
        return []

    relationships: list[GraphCandidateRelationship] = [
        GraphCandidateRelationship(
            source=document_name,
            relation="has_field",
            target=label,
            evidence=_clip(element.text, 240),
            confidence=0.7,
            metadata=_relationship_metadata(element),
        )
    ]
    context = _container_context(document_name, element)
    if context and context[0] != document_name:
        relationships.append(GraphCandidateRelationship(
            source=_clip(context[0], 160),
            relation="has_field",
            target=label,
            evidence=_clip(element.text, 240),
            confidence=0.7,
            metadata=_relationship_metadata(element),
        ))

    value = _form_field_graph_value(metadata)
    if value:
        relationships.append(GraphCandidateRelationship(
            source=label,
            relation="has_value",
            target=value,
            evidence=_clip(element.text, 240),
            confidence=0.72,
            metadata=_relationship_metadata(element),
        ))
    checked = metadata.get("form_field_checked")
    if isinstance(checked, bool):
        checked_state = "checked" if checked else "unchecked"
        if checked_state != value:
            relationships.append(GraphCandidateRelationship(
                source=label,
                relation="has_value",
                target=checked_state,
                evidence=_clip(element.text, 240),
                confidence=0.7,
                metadata=_relationship_metadata(element, normalized_form_state=checked_state),
            ))

    return relationships


def _chart_graph_label(element: DocumentElement, document_name: str) -> str:
    metadata = element.metadata if isinstance(element.metadata, dict) else {}
    chart_title = _clean_text(str(metadata.get("chart_title") or ""))
    if chart_title:
        return _clip(chart_title, 160)
    slide_title = _clean_text(str(metadata.get("slide_title") or ""))
    chart_index = metadata.get("chart")
    if slide_title and chart_index not in (None, ""):
        return _clip(f"{slide_title} chart {chart_index}", 160)
    chart_part = _clean_text(str(metadata.get("chart_part") or ""))
    if chart_part:
        return _clip(chart_part, 160)
    if chart_index not in (None, ""):
        return _clip(f"{document_name} chart {chart_index}", 160)
    return _clip(f"{document_name} chart", 160)


def _container_context(document_name: str, element: DocumentElement) -> tuple[str, str, str] | None:
    metadata = element.metadata if isinstance(element.metadata, dict) else {}
    slide = metadata.get("slide")
    if slide is not None:
        slide_title = _clean_text(str(metadata.get("slide_title") or ""))
        label = f"Slide {slide}: {slide_title}" if slide_title else f"{document_name} slide {slide}"
        return label, "has_slide", "slide"

    sheet_name = _clean_text(str(metadata.get("sheet_name") or ""))
    if sheet_name:
        return sheet_name, "has_sheet", "sheet"

    page = metadata.get("page")
    if page is not None:
        return f"{document_name} page {page}", "has_page", "page"

    chapter_path = _clean_text(str(metadata.get("chapter_path") or ""))
    if chapter_path:
        return chapter_path, "has_chapter", "chapter"

    if element.type != "embedded_object":
        attachment = _clean_text(str(metadata.get("embedded_file_filename") or metadata.get("embedded_file_name") or ""))
        if attachment:
            return f"{document_name} attachment {attachment}", "has_attachment", "attachment"

    if element.type != "archive_entry":
        archive_entry = _clean_text(str(metadata.get("archive_entry_path") or metadata.get("archive_entry_filename") or ""))
        if archive_entry:
            return f"{document_name} archive {archive_entry}", "has_archive_entry", "archive_entry"

    legacy_stream = _clean_text(str(metadata.get("legacy_office_stream_path") or ""))
    if legacy_stream:
        if element.type == "legacy_office_stream":
            return document_name, "has_document", "document"
        return f"{document_name} legacy Office stream {legacy_stream}", "has_legacy_office_stream", "legacy_office_stream"

    return None


def _structure_relation_for_element(element: DocumentElement) -> str | None:
    if element.type in {"heading", "slide_title"}:
        return "contains_heading"
    if element.type in {"slide_text", "text_box", "paragraph", "page_text", "image_ocr_text", "page_ocr_text", "json_value"}:
        return "contains_text"
    if element.type == "email_body":
        return "contains_email_body"
    if element.type in {"table_header", "table_row", "table_cell", "table_definition"}:
        return "contains_table"
    if element.type == "pivot_table":
        return "contains_pivot_table"
    if element.type == "named_range":
        return "has_named_range"
    if element.type in {"chart_axis", "chart_data_label", "chart_legend", "chart_series", "chart_point", "chart_trendline"}:
        return "contains_chart"
    if element.type == "diagram_node":
        return "contains_diagram_node"
    if element.type == "diagram_relationship":
        return "contains_diagram_relationship"
    if element.type == "code_block":
        return "contains_code_block"
    if element.type == "shape_metadata":
        return "contains_shape"
    if element.type == "shape_connector":
        return "contains_connector"
    if element.type == "form_field":
        return "contains_form_field"
    if element.type == "conditional_format":
        return "has_conditional_format"
    if element.type == "formula":
        return "contains_formula"
    if element.type == "comment":
        return "contains_comment"
    if element.type == "annotation":
        return "contains_annotation"
    if element.type == "outline_item":
        return "contains_outline_item"
    if element.type == "revision":
        return "contains_revision"
    if element.type == "speaker_notes":
        return "contains_speaker_notes"
    if element.type in {"footnote", "endnote"}:
        return f"contains_{element.type}"
    if element.type == "image":
        return "contains_image"
    if element.type == "media":
        return "contains_media"
    if element.type == "embedded_object":
        return "contains_embedded_object"
    if element.type == "archive_entry":
        return "contains_archive_entry"
    if element.type == "legacy_office_stream":
        return "contains_legacy_office_stream"
    if element.type == "legacy_office_text":
        return "contains_legacy_office_text"
    if element.type == "link":
        return "contains_link"
    return None


def _layout_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not parsed and parsed != 0:
        return None
    return parsed


def _visual_layout_point(element: DocumentElement) -> tuple[float, float] | None:
    metadata = element.metadata if isinstance(element.metadata, dict) else {}
    x_pct = _layout_float(metadata.get("layout_center_x_pct"))
    y_pct = _layout_float(metadata.get("layout_center_y_pct"))
    if x_pct is not None and y_pct is not None:
        return x_pct, y_pct

    x_inches = _layout_float(metadata.get("layout_center_x_inches"))
    y_inches = _layout_float(metadata.get("layout_center_y_inches"))
    width_inches = _layout_float(metadata.get("slide_width_inches") or metadata.get("page_width_inches"))
    height_inches = _layout_float(metadata.get("slide_height_inches") or metadata.get("page_height_inches"))
    if (
        x_inches is not None
        and y_inches is not None
        and width_inches
        and height_inches
    ):
        return round((x_inches / width_inches) * 100, 2), round((y_inches / height_inches) * 100, 2)

    return None


def _visual_layout_family(element: DocumentElement) -> str:
    if element.type.startswith("chart_"):
        return "chart"
    if element.type.startswith("table_"):
        return "table"
    if element.type in {"shape_group", "shape_metadata", "slide_text", "text_box"}:
        return "shape"
    if element.type == "form_field":
        return "form"
    if element.type == "image":
        return "image"
    return element.type


def _visual_layout_label(document_name: str, element: DocumentElement) -> str:
    metadata = element.metadata if isinstance(element.metadata, dict) else {}
    if element.type.startswith("chart_"):
        label = _chart_graph_label(element, document_name)
    elif element.type.startswith("table_"):
        label = _clean_text(str(
            metadata.get("table_name")
            or metadata.get("table_display_name")
            or metadata.get("table_cell_context")
            or ""
        ))
        if not label:
            table = metadata.get("table")
            slide = metadata.get("slide")
            if table not in (None, ""):
                label = f"Slide {slide} table {table}" if slide not in (None, "") else f"Table {table}"
        label = label or _element_graph_label(element)
    else:
        label = _element_graph_label(element)

    label = _clip(_clean_text(label), 160)
    if not label or label.lower() == document_name.lower():
        return ""
    if GENERIC_POWERPOINT_SHAPE_NAME_RE.match(label):
        return ""
    if element.type in {"slide_text", "text_box", "shape_metadata"}:
        word_count = len(label.split())
        if len(label) > 96 or (word_count >= 7 and re.search(r"[.!?;]", label)):
            return ""
    return label


def _visual_layout_relation(source_point: tuple[float, float], target_point: tuple[float, float]) -> str:
    dx = target_point[0] - source_point[0]
    dy = target_point[1] - source_point[1]
    if abs(dx) <= 6 and abs(dy) <= 6:
        return "overlaps"
    if abs(dx) >= abs(dy):
        return "right_of" if dx > 0 else "left_of"
    return "below" if dy > 0 else "above"


def _visual_layout_relationships(
    document_name: str,
    elements: list[DocumentElement],
    *,
    limit: int = MAX_VISUAL_LAYOUT_RELATIONSHIPS,
) -> list[GraphCandidateRelationship]:
    candidates_by_slide: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_candidate_keys: set[tuple[str, str, str]] = set()

    for index, element in enumerate(elements):
        if element.type not in VISUAL_LAYOUT_GRAPH_ELEMENT_TYPES:
            continue
        metadata = element.metadata if isinstance(element.metadata, dict) else {}
        slide = metadata.get("slide")
        if slide in (None, ""):
            continue
        point = _visual_layout_point(element)
        if point is None:
            continue
        label = _visual_layout_label(document_name, element)
        if not label:
            continue
        family = _visual_layout_family(element)
        candidate_key = (str(slide), family, label.lower())
        if candidate_key in seen_candidate_keys:
            continue
        seen_candidate_keys.add(candidate_key)
        candidates_by_slide[str(slide)].append({
            "index": index,
            "element": element,
            "label": label,
            "family": family,
            "point": point,
        })

    relationships: list[GraphCandidateRelationship] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for slide, candidates in candidates_by_slide.items():
        slide_edges = 0
        ordered = sorted(
            candidates,
            key=lambda item: (
                item["point"][1],
                item["point"][0],
                item["index"],
            ),
        )
        for source in ordered:
            pair_scores: list[tuple[float, dict[str, Any]]] = []
            for target in ordered:
                if source is target:
                    continue
                if source["family"] == target["family"]:
                    continue
                dx = float(target["point"][0]) - float(source["point"][0])
                dy = float(target["point"][1]) - float(source["point"][1])
                distance = (dx * dx + dy * dy) ** 0.5
                if distance > VISUAL_LAYOUT_NEAR_DISTANCE_PCT:
                    continue
                pair_scores.append((distance, target))

            for distance, target in sorted(pair_scores, key=lambda item: (item[0], item[1]["index"]))[:MAX_VISUAL_LAYOUT_NEIGHBORS_PER_ELEMENT]:
                source_label = _clip(str(source["label"]), 160)
                target_label = _clip(str(target["label"]), 160)
                if not source_label or not target_label or source_label.lower() == target_label.lower():
                    continue
                edge_pair = tuple(sorted((source_label.lower(), target_label.lower())))
                key = (edge_pair[0], "visually_near", edge_pair[1])
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                relation = _visual_layout_relation(source["point"], target["point"])
                source_element = source["element"]
                target_element = target["element"]
                confidence = round(0.58 + max(0.0, (VISUAL_LAYOUT_NEAR_DISTANCE_PCT - distance) / VISUAL_LAYOUT_NEAR_DISTANCE_PCT) * 0.16, 3)
                relationships.append(GraphCandidateRelationship(
                    source=source_label,
                    relation="visually_near",
                    target=target_label,
                    evidence=_clip(
                        f"Slide {slide} visually places {target_label} {relation.replace('_', ' ')} {source_label}.",
                        240,
                    ),
                    confidence=confidence,
                    metadata=_relationship_metadata(
                        source_element,
                        layout_relation=relation,
                        visual_distance_pct=round(distance, 2),
                        related_element_type=target_element.type,
                        related_label=target_label,
                        related_location=target_element.location,
                        target_layout_zone=(target_element.metadata or {}).get("layout_zone"),
                        target_layout_bbox=(target_element.metadata or {}).get("layout_bbox"),
                    ),
                ))
                slide_edges += 1
                if len(relationships) >= limit or slide_edges >= MAX_VISUAL_LAYOUT_RELATIONSHIPS_PER_SLIDE:
                    break
            if len(relationships) >= limit or slide_edges >= MAX_VISUAL_LAYOUT_RELATIONSHIPS_PER_SLIDE:
                break

        if len(relationships) >= limit:
            break

    return relationships


def _structural_relationships(document_name: str, elements: list[DocumentElement], *, limit: int = 80) -> list[GraphCandidateRelationship]:
    relationships: list[GraphCandidateRelationship] = []
    seen: set[tuple[str, str, str]] = set()

    def add(source: str, relation: str, target: str, element: DocumentElement, confidence: float, **metadata: Any) -> None:
        source = _clip(_clean_text(source), 160)
        target = _clip(_clean_text(target), 160)
        if not source or not target or source == target:
            return
        key = (source.lower(), relation, target.lower())
        if key in seen:
            return
        seen.add(key)
        relationships.append(GraphCandidateRelationship(
            source=source,
            relation=relation,
            target=target,
            evidence=_clip(element.text, 240),
            confidence=confidence,
            metadata=_relationship_metadata(element, **metadata),
        ))

    for element in elements:
        context = _container_context(document_name, element)
        if not context:
            continue
        container_label, container_relation, container_type = context
        add(
            document_name,
            container_relation,
            container_label,
            element,
            0.82,
            container_type=container_type,
        )

        element_relation = _structure_relation_for_element(element)
        element_label = _element_graph_label(element, container_label=container_label)
        if element_relation and element_label:
            add(
                container_label,
                element_relation,
                element_label,
                element,
                0.68,
                container_type=container_type,
                container_label=container_label,
            )
        if len(relationships) >= limit:
            break

    return relationships


def _element_scope_key(element: DocumentElement) -> tuple[str, str]:
    metadata = element.metadata if isinstance(element.metadata, dict) else {}
    for key in ("slide", "sheet_name", "page", "chapter_path", "chapter"):
        value = metadata.get(key)
        if value not in {None, ""}:
            return key, str(value)
    return "document", ""


def _heading_level(element: DocumentElement) -> int:
    metadata = element.metadata if isinstance(element.metadata, dict) else {}
    raw_level = metadata.get("level")
    if raw_level not in {None, ""}:
        try:
            return max(1, int(raw_level))
        except (TypeError, ValueError):
            pass

    style = _clean_text(str(metadata.get("style") or ""))
    match = re.search(r"heading\s*(\d+)", style, flags=re.I)
    if match:
        return max(1, int(match.group(1)))
    return 1 if element.type in {"heading", "slide_title"} else 99


def _section_relationships(document_name: str, elements: list[DocumentElement], *, limit: int = 80) -> list[GraphCandidateRelationship]:
    relationships: list[GraphCandidateRelationship] = []
    seen: set[tuple[str, str, str]] = set()
    section_stacks: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def add(source: str, relation: str, target: str, element: DocumentElement, confidence: float, **metadata: Any) -> None:
        source = _clip(_clean_text(source), 160)
        target = _clip(_clean_text(target), 160)
        if not source or not target or source == target:
            return
        key = (source.lower(), relation, target.lower())
        if key in seen:
            return
        seen.add(key)
        relationships.append(GraphCandidateRelationship(
            source=source,
            relation=relation,
            target=target,
            evidence=_clip(element.text, 240),
            confidence=confidence,
            metadata=_relationship_metadata(element, **metadata),
        ))

    for element in elements:
        scope_key = _element_scope_key(element)
        stack = section_stacks.setdefault(scope_key, [])

        if element.type in {"heading", "slide_title"}:
            section_label = _element_graph_label(element)
            level = _heading_level(element)
            while stack and int(stack[-1].get("level", 1)) >= level:
                stack.pop()
            parent = stack[-1] if stack else None
            if parent:
                add(
                    str(parent["label"]),
                    "has_subsection",
                    section_label,
                    element,
                    0.76,
                    section_label=section_label,
                    section_level=level,
                    parent_section_label=parent.get("label"),
                    scope_type=scope_key[0],
                    scope_value=scope_key[1],
                )
            else:
                context = _container_context(document_name, element)
                section_source = context[0] if context else document_name
                add(
                    section_source,
                    "has_section",
                    section_label,
                    element,
                    0.74,
                    section_label=section_label,
                    section_level=level,
                    scope_type=scope_key[0],
                    scope_value=scope_key[1],
                )
            stack.append({
                "label": section_label,
                "level": level,
                "location": element.location,
            })
            if len(relationships) >= limit:
                break
            continue

        explicit_section = _clean_text(str((element.metadata or {}).get("section") or ""))
        section = {"label": explicit_section, "level": None, "location": None} if explicit_section else (stack[-1] if stack else None)
        if not section:
            continue

        relation = _structure_relation_for_element(element)
        target = _element_graph_label(element)
        if relation and target:
            add(
                str(section["label"]),
                relation,
                target,
                element,
                0.64,
                section_label=section.get("label"),
                section_level=section.get("level"),
                section_location=section.get("location"),
                scope_type=scope_key[0],
                scope_value=scope_key[1],
            )
        if len(relationships) >= limit:
            break

    return relationships


def _candidate_relationships(filename: str, elements: list[DocumentElement], entities: list[dict[str, Any]], *, limit: int = 220) -> list[GraphCandidateRelationship]:
    relationships: list[GraphCandidateRelationship] = []
    co_occurrence_relationships: list[GraphCandidateRelationship] = []
    chart_context_relationships: list[GraphCandidateRelationship] = []
    document_name = Path(filename or "document").name
    entity_names = [entity["name"] for entity in entities[:25]]
    sheet_relationships_seen: set[str] = set()
    semantic_relationships = _semantic_candidate_relationships(
        elements,
        limit=max(1, min(limit, 80)),
    )

    for element in elements:
        metadata = element.metadata if isinstance(element.metadata, dict) else {}
        sheet_name = _clean_text(str(metadata.get("sheet_name") or ""))
        if sheet_name and sheet_name.lower() not in sheet_relationships_seen:
            sheet_relationships_seen.add(sheet_name.lower())
            relationships.append(GraphCandidateRelationship(
                source=document_name,
                relation="has_sheet",
                target=_clip(sheet_name, 160),
                evidence=_clip(element.text, 240),
                confidence=0.82,
                metadata=_relationship_metadata(element, container_type="sheet"),
            ))
        if element.type == "sheet_metadata" and sheet_name:
            sheet_visibility = _clean_text(str(metadata.get("sheet_visibility") or metadata.get("sheet_state") or "visible")) or "visible"
            sheet_part = _clean_text(str(metadata.get("sheet_part") or ""))
            sheet_id = _clean_text(str(metadata.get("sheet_id") or ""))
            relationships.append(GraphCandidateRelationship(
                source=_clip(sheet_name, 160),
                relation="has_sheet_visibility",
                target=_clip(sheet_visibility, 160),
                evidence=_clip(element.text, 240),
                confidence=0.78,
                metadata=_relationship_metadata(element),
            ))
            if sheet_visibility.lower() not in {"", "visible"}:
                relationships.append(GraphCandidateRelationship(
                    source=document_name,
                    relation="has_hidden_sheet",
                    target=_clip(sheet_name, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.8,
                    metadata=_relationship_metadata(element),
                ))
            if sheet_part:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(sheet_name, 160),
                    relation="uses_sheet_part",
                    target=_clip(sheet_part, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.7,
                    metadata=_relationship_metadata(element),
                ))
            if sheet_id:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(sheet_name, 160),
                    relation="has_sheet_id",
                    target=_clip(sheet_id, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.64,
                    metadata=_relationship_metadata(element),
                ))
        if element.type == "slide_metadata":
            slide_label = _element_graph_label(element)
            slide_visibility = _clean_text(str(metadata.get("slide_visibility") or metadata.get("slide_state") or "visible")) or "visible"
            slide_part = _clean_text(str(metadata.get("slide_part") or ""))
            slide_id = _clean_text(str(metadata.get("slide_id") or ""))
            if slide_label:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(slide_label, 160),
                    relation="has_slide_visibility",
                    target=_clip(slide_visibility, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.78,
                    metadata=_relationship_metadata(element),
                ))
                if slide_visibility.lower() not in {"", "visible"}:
                    relationships.append(GraphCandidateRelationship(
                        source=document_name,
                        relation="has_hidden_slide",
                        target=_clip(slide_label, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.8,
                        metadata=_relationship_metadata(element),
                    ))
                if slide_part:
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(slide_label, 160),
                        relation="uses_slide_part",
                        target=_clip(slide_part, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.7,
                        metadata=_relationship_metadata(element),
                    ))
                if slide_id:
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(slide_label, 160),
                        relation="has_slide_id",
                        target=_clip(slide_id, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.64,
                        metadata=_relationship_metadata(element),
                    ))
        if metadata.get("inherited_from"):
            context = _container_context(document_name, element)
            inherited_relation = _structure_relation_for_element(element)
            inherited_target = _element_graph_label(element, container_label=context[0] if context else None)
            if context and inherited_relation and inherited_target:
                container_label, _, container_type = context
                if _clean_text(container_label).lower() != _clean_text(inherited_target).lower():
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(container_label, 160),
                        relation=inherited_relation,
                        target=_clip(inherited_target, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.7,
                        metadata=_relationship_metadata(
                            element,
                            container_type=container_type,
                            container_label=container_label,
                        ),
                    ))

        if element.type in {"heading", "slide_title"}:
            relationships.append(GraphCandidateRelationship(
                source=document_name,
                relation="contains",
                target=_clip(element.text, 120),
                evidence=_clip(element.text, 240),
                confidence=0.78,
                metadata=_relationship_metadata(element),
            ))
            if metadata.get("heading_kind") == "spreadsheet_merged_cell":
                sheet_name = _clean_text(str(metadata.get("sheet_name") or ""))
                merge_range = _clean_text(str(metadata.get("merge_range") or ""))
                range_label = _clean_text(f"{sheet_name}!{merge_range}" if sheet_name and merge_range else merge_range)
                heading_label = _element_graph_label(element)
                if heading_label:
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(sheet_name or document_name, 160),
                        relation="has_merged_label",
                        target=_clip(heading_label, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.74,
                        metadata=_relationship_metadata(element),
                    ))
                if heading_label and range_label:
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(heading_label, 160),
                        relation="covers_range",
                        target=_clip(range_label, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.72,
                        metadata=_relationship_metadata(element, merge_range_label=range_label),
                    ))
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(range_label, 160),
                        relation="labeled_by",
                        target=_clip(heading_label, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.7,
                        metadata=_relationship_metadata(element, merge_range_label=range_label),
                    ))

        if element.type == "document_metadata":
            field = _clean_text(str(element.metadata.get("metadata_field") or ""))
            value = _clean_text(str(element.metadata.get("metadata_value") or ""))
            relation = OFFICE_METADATA_RELATIONS.get(field)
            if relation and value:
                for target in _metadata_values(field, value):
                    relationships.append(GraphCandidateRelationship(
                        source=document_name,
                        relation=relation,
                        target=_clip(target, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.74,
                        metadata=_relationship_metadata(element),
                    ))

        if element.type == "email_body":
            message_id = _clean_text(str(element.metadata.get("email_message_id") or ""))
            subject = _clean_text(str(element.metadata.get("email_subject") or ""))
            sender = _clean_text(str(element.metadata.get("email_from") or element.metadata.get("email_sender") or ""))
            recipients = _clean_text(str(element.metadata.get("email_to") or ""))
            body_label = _clip(subject or message_id or element.text, 160)
            if body_label:
                relationships.append(GraphCandidateRelationship(
                    source=document_name,
                    relation="contains_email_body",
                    target=body_label,
                    evidence=_clip(element.text, 240),
                    confidence=0.66,
                    metadata=_relationship_metadata(element),
                ))
            if sender and body_label:
                for source in _metadata_values("email_from", sender):
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(source, 160),
                        relation="sent_message",
                        target=body_label,
                        evidence=_clip(element.text, 240),
                        confidence=0.7,
                        metadata=_relationship_metadata(element),
                    ))
            if recipients and body_label:
                for target in _metadata_values("email_to", recipients):
                    relationships.append(GraphCandidateRelationship(
                        source=body_label,
                        relation="sent_to",
                        target=_clip(target, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.68,
                        metadata=_relationship_metadata(element),
                    ))

        styled_terms = _styled_text_terms(element.metadata if isinstance(element.metadata, dict) else {})
        if element.type in {
            "heading",
            "slide_title",
            "slide_text",
            "text_box",
            "paragraph",
            "email_body",
            "page_text",
            "image_ocr_text",
            "page_ocr_text",
            "comment",
            "annotation",
            "revision",
            "speaker_notes",
            "table_cell",
            "table_header",
            "table_row",
        } and styled_terms:
            context = _container_context(document_name, element)
            explicit_section = _clean_text(str(element.metadata.get("section") or "")) if isinstance(element.metadata, dict) else ""
            source = context[0] if context else explicit_section or document_name
            for styled_term in styled_terms[:8]:
                term_text = _clean_text(str(styled_term.get("text") or ""))
                if not term_text:
                    continue
                styles = styled_term.get("styles") if isinstance(styled_term.get("styles"), list) else []
                style_summary = _clean_text(str(element.metadata.get("style_summary") or "; ".join(styles)))
                evidence = " | ".join(
                    part
                    for part in [
                        _clean_text(element.text),
                        f"style: {style_summary}" if style_summary else "",
                    ]
                    if part
                )
                relationships.append(GraphCandidateRelationship(
                    source=source,
                    relation="emphasizes",
                    target=_clip(term_text, 160),
                    evidence=_clip(evidence or element.text, 240),
                    confidence=0.64,
                    metadata=_relationship_metadata(
                        element,
                        styled_text=True,
                        emphasis_term=term_text,
                        emphasis_styles=styles,
                        style_summary=style_summary,
                    ),
                ))

        bullet_levels = element.metadata.get("bullet_levels") if isinstance(element.metadata, dict) else None
        if isinstance(bullet_levels, list) and bullet_levels and element.type in {"slide_text", "text_box", "paragraph"}:
            context = _container_context(document_name, element)
            explicit_section = _clean_text(str(element.metadata.get("section") or "")) if isinstance(element.metadata, dict) else ""
            source = context[0] if context else explicit_section or document_name
            level_text = ", ".join(str(level) for level in _unique_preserve_order([str(level) for level in bullet_levels[:8]]))
            if level_text:
                relationships.append(GraphCandidateRelationship(
                    source=source,
                    relation="uses_bullet_structure",
                    target=f"bullet levels {level_text}",
                    evidence=_clip(element.text, 240),
                    confidence=0.56,
                    metadata=_relationship_metadata(
                        element,
                        styled_text=True,
                        bullet_levels=bullet_levels[:12],
                        style_summary=element.metadata.get("style_summary"),
                    ),
                ))

        cells = element.metadata.get("cells") if isinstance(element.metadata, dict) else None
        headers = element.metadata.get("headers") if isinstance(element.metadata, dict) else None
        if element.type == "table_row" and isinstance(cells, list) and cells:
            semantic_table_relationship = _semantic_table_row_relationship(element)
            semantic_table_claim_headers: set[str] = set()
            if semantic_table_relationship:
                semantic_table_claim_headers = (
                    SEMANTIC_TABLE_RELATION_HEADERS
                    | SEMANTIC_TABLE_TARGET_HEADERS
                    | SEMANTIC_TABLE_EVIDENCE_HEADERS
                )
            subject = _clean_text(str(cells[0]))
            if subject:
                for index, value in enumerate(cells[1:6], start=1):
                    value_text = _clean_text(str(value))
                    if not value_text:
                        continue
                    relation = _clean_text(str(headers[index])) if isinstance(headers, list) and index < len(headers) else f"has_column_{index + 1}"
                    if (
                        semantic_table_relationship
                        and _semantic_table_header_key(relation) in semantic_table_claim_headers
                    ):
                        continue
                    relationships.append(GraphCandidateRelationship(
                        source=subject,
                        relation=_relation_label(relation),
                        target=value_text,
                        evidence=_clip(element.text, 240),
                        confidence=0.72,
                        metadata=_relationship_metadata(element),
                    ))

        if element.type == "formula":
            formula_cell = _clean_text(str(element.metadata.get("formula_cell") or element.metadata.get("cell") or ""))
            expression = _clean_text(str(element.metadata.get("formula_expression") or ""))
            result_value = _clean_text(str(element.metadata.get("formula_result") or ""))
            references = element.metadata.get("formula_references") if isinstance(element.metadata, dict) else None
            if formula_cell:
                relationships.append(GraphCandidateRelationship(
                    source=document_name,
                    relation="has_formula",
                    target=_clip(formula_cell, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.66,
                    metadata=_relationship_metadata(element),
                ))
            if formula_cell and expression:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(formula_cell, 160),
                    relation="uses_formula",
                    target=_clip(expression, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.7,
                    metadata=_relationship_metadata(element),
                ))
            if formula_cell and result_value:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(formula_cell, 160),
                    relation="calculates",
                    target=_clip(result_value, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.68,
                    metadata=_relationship_metadata(element),
                ))
            if formula_cell and isinstance(references, list):
                for reference in references[:24]:
                    reference_text = _clean_text(str(reference))
                    if not reference_text:
                        continue
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(formula_cell, 160),
                        relation="depends_on",
                        target=_clip(reference_text, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.74,
                        metadata=_relationship_metadata(element, formula_reference=reference_text),
                    ))

        if element.type == "conditional_format":
            conditional_label = _clean_text(str(element.metadata.get("conditional_label") or element.text))
            range_label = _clean_text(str(element.metadata.get("conditional_range_label") or element.metadata.get("conditional_range") or ""))
            rule_type = _clean_text(str(element.metadata.get("conditional_type") or ""))
            operator = _clean_text(str(element.metadata.get("conditional_operator") or ""))
            formula1 = _clean_text(str(element.metadata.get("conditional_formula1") or ""))
            formula2 = _clean_text(str(element.metadata.get("conditional_formula2") or ""))
            condition = _clean_text(" ".join(part for part in [rule_type, operator, formula1, formula2] if part))
            references = element.metadata.get("conditional_formula_references") if isinstance(element.metadata, dict) else None
            sheet_name = _clean_text(str(element.metadata.get("sheet_name") or ""))
            if conditional_label:
                relationships.append(GraphCandidateRelationship(
                    source=sheet_name or document_name,
                    relation="has_conditional_format",
                    target=_clip(conditional_label, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.66,
                    metadata=_relationship_metadata(element),
                ))
            if conditional_label and range_label:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(conditional_label, 160),
                    relation="applies_to",
                    target=_clip(range_label, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.7,
                    metadata=_relationship_metadata(element),
                ))
                relationships.append(GraphCandidateRelationship(
                    source=_clip(range_label, 160),
                    relation="evaluated_by",
                    target=_clip(conditional_label, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.7,
                    metadata=_relationship_metadata(element),
                ))
            if conditional_label and condition:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(conditional_label, 160),
                    relation="uses_condition",
                    target=_clip(condition, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.68,
                    metadata=_relationship_metadata(element),
                ))
            if conditional_label and isinstance(references, list):
                for reference in references[:24]:
                    reference_text = _clean_text(str(reference))
                    if not reference_text:
                        continue
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(conditional_label, 160),
                        relation="depends_on",
                        target=_clip(reference_text, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.7,
                        metadata=_relationship_metadata(element, conditional_reference=reference_text),
                    ))

        if element.type == "table_definition":
            table_label = _clean_text(str(element.metadata.get("table_display_name") or element.metadata.get("table_name") or element.text))
            table_range = _clean_text(str(element.metadata.get("table_range") or ""))
            table_columns = element.metadata.get("table_columns") if isinstance(element.metadata, dict) else None
            sheet_name = _clean_text(str(element.metadata.get("sheet_name") or ""))
            if sheet_name and table_label:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(sheet_name, 160),
                    relation="contains_table",
                    target=_clip(table_label, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.74,
                    metadata=_relationship_metadata(element),
                ))
            if table_label and table_range:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(table_label, 160),
                    relation="covers_range",
                    target=_clip(table_range, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.7,
                    metadata=_relationship_metadata(element),
                ))
            if table_label and isinstance(table_columns, list):
                for column in table_columns[:24]:
                    column_label = _clean_text(str(column))
                    if not column_label:
                        continue
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(table_label, 160),
                        relation="has_column",
                        target=_clip(column_label, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.68,
                        metadata=_relationship_metadata(element, table_column=column_label),
                    ))

        if element.type == "pivot_table":
            pivot_label = _clean_text(str(element.metadata.get("pivot_table_name") or element.text))
            pivot_range = _clean_text(str(element.metadata.get("pivot_table_range") or ""))
            sheet_name = _clean_text(str(element.metadata.get("sheet_name") or ""))
            source_name = _clean_text(str(element.metadata.get("pivot_table_source_name") or ""))
            source_ref = _clean_text(str(element.metadata.get("pivot_table_source_ref") or ""))
            source_sheet = _clean_text(str(element.metadata.get("pivot_table_source_sheet") or ""))
            source_label = source_name or _clean_text("!".join(part for part in [source_sheet, source_ref] if part))
            row_fields = element.metadata.get("pivot_table_row_fields") if isinstance(element.metadata, dict) else None
            column_fields = element.metadata.get("pivot_table_column_fields") if isinstance(element.metadata, dict) else None
            page_fields = element.metadata.get("pivot_table_page_fields") if isinstance(element.metadata, dict) else None
            data_fields = element.metadata.get("pivot_table_data_fields") if isinstance(element.metadata, dict) else None
            if sheet_name and pivot_label:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(sheet_name, 160),
                    relation="contains_pivot_table",
                    target=_clip(pivot_label, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.74,
                    metadata=_relationship_metadata(element),
                ))
            if pivot_label and pivot_range:
                range_label = _clean_text(f"{sheet_name}!{pivot_range}" if sheet_name else pivot_range)
                relationships.append(GraphCandidateRelationship(
                    source=_clip(pivot_label, 160),
                    relation="covers_range",
                    target=_clip(range_label, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.7,
                    metadata=_relationship_metadata(element, pivot_table_range_label=range_label),
                ))
            if pivot_label and source_label:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(pivot_label, 160),
                    relation="uses_source",
                    target=_clip(source_label, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.76,
                    metadata=_relationship_metadata(element),
                ))
                if source_ref and source_label != source_ref:
                    source_range_label = _clean_text(f"{source_sheet}!{source_ref}" if source_sheet else source_ref)
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(source_label, 160),
                        relation="covers_range",
                        target=_clip(source_range_label, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.68,
                        metadata=_relationship_metadata(element, pivot_table_source_range_label=source_range_label),
                    ))
            if pivot_label and isinstance(row_fields, list):
                for field in row_fields[:16]:
                    field_label = _clean_text(str(field))
                    if field_label:
                        relationships.append(GraphCandidateRelationship(
                            source=_clip(pivot_label, 160),
                            relation="groups_by",
                            target=_clip(field_label, 160),
                            evidence=_clip(element.text, 240),
                            confidence=0.7,
                            metadata=_relationship_metadata(element, pivot_field=field_label, pivot_axis="row"),
                        ))
            if pivot_label and isinstance(column_fields, list):
                for field in column_fields[:16]:
                    field_label = _clean_text(str(field))
                    if field_label:
                        relationships.append(GraphCandidateRelationship(
                            source=_clip(pivot_label, 160),
                            relation="columns_by",
                            target=_clip(field_label, 160),
                            evidence=_clip(element.text, 240),
                            confidence=0.68,
                            metadata=_relationship_metadata(element, pivot_field=field_label, pivot_axis="column"),
                        ))
            if pivot_label and isinstance(page_fields, list):
                for field in page_fields[:16]:
                    field_label = _clean_text(str(field))
                    if field_label:
                        relationships.append(GraphCandidateRelationship(
                            source=_clip(pivot_label, 160),
                            relation="filters_by",
                            target=_clip(field_label, 160),
                            evidence=_clip(element.text, 240),
                            confidence=0.66,
                            metadata=_relationship_metadata(element, pivot_field=field_label, pivot_axis="page"),
                        ))
            if pivot_label and isinstance(data_fields, list):
                for data_field in data_fields[:16]:
                    if not isinstance(data_field, dict):
                        continue
                    data_label = _clean_text(str(data_field.get("name") or data_field.get("source_field") or ""))
                    if not data_label:
                        continue
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(pivot_label, 160),
                        relation="aggregates",
                        target=_clip(data_label, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.72,
                        metadata=_relationship_metadata(
                            element,
                            pivot_data_field=data_label,
                            pivot_source_field=data_field.get("source_field"),
                            pivot_subtotal=data_field.get("subtotal"),
                        ),
                    ))

        if element.type == "named_range":
            range_name = _clean_text(str(element.metadata.get("named_range_name") or element.text))
            range_ref = _clean_text(str(element.metadata.get("named_range_ref") or ""))
            sheet_name = _clean_text(str(element.metadata.get("sheet_name") or ""))
            if range_name:
                relationships.append(GraphCandidateRelationship(
                    source=document_name,
                    relation="has_named_range",
                    target=_clip(range_name, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.68,
                    metadata=_relationship_metadata(element),
                ))
                if sheet_name:
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(sheet_name, 160),
                        relation="has_named_range",
                        target=_clip(range_name, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.7,
                        metadata=_relationship_metadata(element),
                    ))
            if range_name and range_ref:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(range_name, 160),
                    relation="covers_range",
                    target=_clip(range_ref, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.72,
                    metadata=_relationship_metadata(element),
                ))

        if element.type == "chart_series":
            series_name = _clean_text(str(element.metadata.get("series_name") or ""))
            chart_label = _chart_graph_label(element, document_name)
            sheet_name = _clean_text(str(element.metadata.get("sheet_name") or ""))
            context = _container_context(document_name, element)
            chart_container = context[0] if context else sheet_name or document_name
            if chart_label:
                chart_context_relationships.append(GraphCandidateRelationship(
                    source=_clip(chart_container, 160),
                    relation="contains_chart",
                    target=chart_label,
                    evidence=_clip(element.text, 240),
                    confidence=0.68,
                    metadata=_relationship_metadata(element, chart_label=chart_label),
                ))
            if series_name:
                relationships.append(GraphCandidateRelationship(
                    source=document_name,
                    relation="contains_metric",
                    target=series_name,
                    evidence=_clip(element.text, 240),
                    confidence=0.74,
                    metadata=_relationship_metadata(element),
                ))
                if chart_label:
                    chart_context_relationships.append(GraphCandidateRelationship(
                        source=chart_label,
                        relation="contains_chart_series",
                        target=series_name,
                        evidence=_clip(element.text, 240),
                        confidence=0.72,
                        metadata=_relationship_metadata(element, chart_label=chart_label),
                    ))
            chart_source = _clean_text(str(element.metadata.get("chart_source_part") or element.metadata.get("chart_source_uri") or ""))
            if chart_source:
                relationships.append(GraphCandidateRelationship(
                    source=document_name,
                    relation="uses_chart_source",
                    target=_clip(chart_source, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.68,
                    metadata=_relationship_metadata(element),
                ))
                if chart_label:
                    chart_context_relationships.append(GraphCandidateRelationship(
                        source=chart_label,
                        relation="uses_chart_source",
                        target=_clip(chart_source, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.68,
                        metadata=_relationship_metadata(element, chart_label=chart_label),
                    ))

        if element.type == "chart_point":
            series_name = _clean_text(str(element.metadata.get("series_name") or ""))
            category = _clean_text(str(element.metadata.get("category") or ""))
            value = _clean_text(str(element.metadata.get("value") or ""))
            chart_label = _chart_graph_label(element, document_name)
            point_label = _clean_text(" ".join(part for part in [series_name, category] if part))
            if series_name and value:
                relationships.append(GraphCandidateRelationship(
                    source=series_name,
                    relation=_relation_label(f"{category}_value" if category else "has_value"),
                    target=value,
                    evidence=_clip(element.text, 240),
                    confidence=0.76,
                    metadata=_relationship_metadata(element, category=category),
                ))
            if chart_label and point_label:
                chart_context_relationships.append(GraphCandidateRelationship(
                    source=chart_label,
                    relation="contains_chart_point",
                    target=_clip(point_label, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.72,
                    metadata=_relationship_metadata(
                        element,
                        chart_label=chart_label,
                        chart_point_label=point_label,
                        category=category,
                    ),
                ))
            if series_name and category:
                chart_context_relationships.append(GraphCandidateRelationship(
                    source=series_name,
                    relation="has_category",
                    target=_clip(category, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.66,
                    metadata=_relationship_metadata(
                        element,
                        chart_label=chart_label,
                        chart_point_label=point_label,
                        category=category,
                    ),
                ))
            if category and value:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(category, 160),
                    relation="has_chart_value",
                    target=value,
                    evidence=_clip(element.text, 240),
                    confidence=0.72,
                    metadata=_relationship_metadata(
                        element,
                        chart_label=chart_label,
                        chart_point_label=point_label,
                        category=category,
                        series_name=series_name,
                    ),
                ))
                if chart_label:
                    chart_context_relationships.append(GraphCandidateRelationship(
                        source=chart_label,
                        relation="shows_category_value",
                        target=_clip(f"{category}: {value}", 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.7,
                        metadata=_relationship_metadata(
                            element,
                            chart_label=chart_label,
                            chart_point_label=point_label,
                            category=category,
                            series_name=series_name,
                        ),
                    ))
            if point_label and value:
                chart_context_relationships.append(GraphCandidateRelationship(
                    source=_clip(point_label, 160),
                    relation="has_value",
                    target=value,
                    evidence=_clip(element.text, 240),
                    confidence=0.72,
                    metadata=_relationship_metadata(
                        element,
                        chart_label=chart_label,
                        chart_point_label=point_label,
                        category=category,
                    ),
                ))

        if element.type == "chart_axis":
            chart_label = _chart_graph_label(element, document_name)
            axis_label = _element_graph_label(element)
            if chart_label and axis_label:
                chart_context_relationships.append(GraphCandidateRelationship(
                    source=chart_label,
                    relation="has_axis",
                    target=axis_label,
                    evidence=_clip(element.text, 240),
                    confidence=0.72,
                    metadata=_relationship_metadata(element, chart_label=chart_label),
                ))

        if element.type == "chart_legend":
            chart_label = _chart_graph_label(element, document_name)
            legend_label = _element_graph_label(element)
            if chart_label and legend_label:
                chart_context_relationships.append(GraphCandidateRelationship(
                    source=chart_label,
                    relation="has_legend",
                    target=legend_label,
                    evidence=_clip(element.text, 240),
                    confidence=0.68,
                    metadata=_relationship_metadata(element, chart_label=chart_label),
                ))

        if element.type == "chart_data_label":
            chart_label = _chart_graph_label(element, document_name)
            label = _element_graph_label(element)
            if chart_label and label:
                chart_context_relationships.append(GraphCandidateRelationship(
                    source=chart_label,
                    relation="has_data_labels",
                    target=label,
                    evidence=_clip(element.text, 240),
                    confidence=0.7,
                    metadata=_relationship_metadata(element, chart_label=chart_label),
                ))

        if element.type == "chart_trendline":
            chart_label = _chart_graph_label(element, document_name)
            trendline_label = _element_graph_label(element)
            series_name = _clean_text(str(element.metadata.get("series_name") or ""))
            if chart_label and trendline_label:
                chart_context_relationships.append(GraphCandidateRelationship(
                    source=chart_label,
                    relation="has_trendline",
                    target=trendline_label,
                    evidence=_clip(element.text, 240),
                    confidence=0.7,
                    metadata=_relationship_metadata(element, chart_label=chart_label),
                ))
            if series_name and trendline_label:
                chart_context_relationships.append(GraphCandidateRelationship(
                    source=series_name,
                    relation="has_trendline",
                    target=trendline_label,
                    evidence=_clip(element.text, 240),
                    confidence=0.68,
                    metadata=_relationship_metadata(element, chart_label=chart_label),
                ))

        if element.type == "diagram_node":
            label = _clean_text(element.text)
            if label:
                relationships.append(GraphCandidateRelationship(
                    source=document_name,
                    relation="contains_diagram_node",
                    target=_clip(label, 120),
                    evidence=_clip(element.text, 240),
                    confidence=0.66,
                    metadata=_relationship_metadata(element),
                ))
        if element.type == "diagram_relationship":
            source_label = _clean_text(str(element.metadata.get("source_label") or ""))
            target_label = _clean_text(str(element.metadata.get("target_label") or ""))
            relation = _clean_text(str(element.metadata.get("relation") or "related_to"))
            if source_label and target_label:
                relationships.append(GraphCandidateRelationship(
                    source=source_label,
                    relation=_relation_label(relation),
                    target=target_label,
                    evidence=_clip(element.text, 240),
                    confidence=0.72,
                    metadata=_relationship_metadata(element),
                ))

        if element.type == "shape_group":
            group_label = _element_graph_label(element)
            child_labels = element.metadata.get("group_child_labels")
            context = _container_context(document_name, element)
            if group_label:
                relationships.append(GraphCandidateRelationship(
                    source=document_name,
                    relation="contains_shape_group",
                    target=_clip(group_label, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.64,
                    metadata=_relationship_metadata(element),
                ))
                if context and context[0] != document_name:
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(context[0], 160),
                        relation="contains_shape_group",
                        target=_clip(group_label, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.66,
                        metadata=_relationship_metadata(element, container_type=context[2], container_label=context[0]),
                    ))
                if isinstance(child_labels, list):
                    for child_label in child_labels[:24]:
                        child_text = _clean_text(str(child_label or ""))
                        if not child_text or child_text.lower() == group_label.lower():
                            continue
                        relationships.append(GraphCandidateRelationship(
                            source=_clip(group_label, 160),
                            relation="groups_visual",
                            target=_clip(child_text, 160),
                            evidence=_clip(element.text, 240),
                            confidence=0.68,
                            metadata=_relationship_metadata(element, grouped_visual_label=child_text),
                        ))

        if element.type == "code_block":
            code_label = _element_graph_label(element)
            language = _clean_text(str(element.metadata.get("markdown_code_language") or ""))
            section = _clean_text(str(element.metadata.get("section") or ""))
            context = _container_context(document_name, element)
            source_label = section or (context[0] if context else document_name)
            container_type = "section" if section else (context[2] if context else "document")
            if code_label:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(source_label, 160),
                    relation="contains_code_block",
                    target=_clip(code_label, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.68,
                    metadata=_relationship_metadata(
                        element,
                        container_type=container_type,
                        container_label=source_label,
                    ),
                ))
                if source_label != document_name:
                    relationships.append(GraphCandidateRelationship(
                        source=document_name,
                        relation="contains_code_block",
                        target=_clip(code_label, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.62,
                        metadata=_relationship_metadata(element),
                    ))
            if code_label and language:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(code_label, 160),
                    relation="uses_language",
                    target=_clip(language, 80),
                    evidence=_clip(element.text, 240),
                    confidence=0.74,
                    metadata=_relationship_metadata(element),
                ))

        if element.type == "shape_metadata":
            shape_label = _element_graph_label(element)
            shape_description = _clean_text(str(element.metadata.get("shape_description") or ""))
            context = _container_context(document_name, element)
            if shape_label:
                relationships.append(GraphCandidateRelationship(
                    source=document_name,
                    relation="contains_shape",
                    target=_clip(shape_label, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.6,
                    metadata=_relationship_metadata(element),
                ))
                if context and context[0] != document_name:
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(context[0], 160),
                        relation="contains_shape",
                        target=_clip(shape_label, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.6,
                        metadata=_relationship_metadata(element, container_type=context[2], container_label=context[0]),
                    ))
            if shape_label and shape_description and shape_description.lower() != shape_label.lower():
                relationships.append(GraphCandidateRelationship(
                    source=_clip(shape_label, 160),
                    relation="has_description",
                    target=_clip(shape_description, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.58,
                    metadata=_relationship_metadata(element),
                ))

        if element.type == "shape_connector":
            connector_label = _element_graph_label(element)
            source_label = _clean_text(str(element.metadata.get("source_label") or ""))
            target_label = _clean_text(str(element.metadata.get("target_label") or ""))
            relation = _clean_text(str(element.metadata.get("relation") or "connects_to"))
            context = _container_context(document_name, element)
            if connector_label:
                relationships.append(GraphCandidateRelationship(
                    source=document_name,
                    relation="contains_connector",
                    target=_clip(connector_label, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.6,
                    metadata=_relationship_metadata(element),
                ))
                if context and context[0] != document_name:
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(context[0], 160),
                        relation="contains_connector",
                        target=_clip(connector_label, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.6,
                        metadata=_relationship_metadata(element, container_type=context[2], container_label=context[0]),
                    ))
            if source_label and target_label:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(source_label, 160),
                    relation=_relation_label(relation),
                    target=_clip(target_label, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.72,
                    metadata=_relationship_metadata(element),
                ))

        if element.type == "form_field":
            relationships.extend(_form_field_relationships(document_name, element))

        if element.type == "outline_item":
            relationships.append(GraphCandidateRelationship(
                source=document_name,
                relation="has_outline_item",
                target=_clip(element.text, 160),
                evidence=_clip(element.text, 240),
                confidence=0.66,
                metadata=_relationship_metadata(element),
            ))

        if element.type == "annotation":
            relationships.append(GraphCandidateRelationship(
                source=document_name,
                relation="has_annotation",
                target=_clip(element.text, 160),
                evidence=_clip(element.text, 240),
                confidence=0.62,
                metadata=_relationship_metadata(element),
            ))
            author = _clean_text(str(element.metadata.get("annotation_author") or ""))
            if author:
                relationships.append(GraphCandidateRelationship(
                    source=author,
                    relation="annotated",
                    target=document_name,
                    evidence=_clip(element.text, 240),
                    confidence=0.6,
                    metadata=_relationship_metadata(element),
                ))

        if element.type == "comment":
            comment_label = _clean_text(str(element.metadata.get("comment_id") or element.location or "comment"))
            anchor_text = _clean_text(str(element.metadata.get("comment_anchor_text") or ""))
            anchor_section = _clean_text(str(element.metadata.get("comment_anchor_section") or ""))
            anchor_sheet = _clean_text(str(element.metadata.get("comment_anchor_sheet") or ""))
            anchor_cell = _clean_text(str(element.metadata.get("comment_anchor_cell") or ""))
            anchor_header = _clean_text(str(element.metadata.get("comment_anchor_header") or ""))
            anchor_value = _clean_text(str(element.metadata.get("comment_anchor_value") or ""))
            anchor_source = anchor_section or anchor_sheet or document_name
            cell_label = _clean_text(f"{anchor_sheet}!{anchor_cell}" if anchor_sheet and anchor_cell else anchor_cell)
            if element.text:
                relationships.append(GraphCandidateRelationship(
                    source=document_name,
                    relation="has_comment",
                    target=_clip(element.text, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.64,
                    metadata=_relationship_metadata(element, comment_label=comment_label),
                ))
                if anchor_section:
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(anchor_section, 160),
                        relation="has_comment",
                        target=_clip(element.text, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.66,
                        metadata=_relationship_metadata(element, comment_label=comment_label),
                    ))
                if anchor_sheet:
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(anchor_sheet, 160),
                        relation="has_comment",
                        target=_clip(element.text, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.66,
                        metadata=_relationship_metadata(element, comment_label=comment_label),
                    ))
                if cell_label:
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(cell_label, 160),
                        relation="has_comment",
                        target=_clip(element.text, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.7,
                        metadata=_relationship_metadata(element, comment_label=comment_label),
                    ))
            if anchor_text:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(anchor_source, 160),
                    relation="comments_on",
                    target=_clip(anchor_text, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.68,
                    metadata=_relationship_metadata(element, comment_label=comment_label),
                ))
                if cell_label:
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(cell_label, 160),
                        relation="comments_on",
                        target=_clip(anchor_text, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.7,
                        metadata=_relationship_metadata(element, comment_label=comment_label),
                    ))
            if cell_label and anchor_header:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(cell_label, 160),
                    relation="has_header",
                    target=_clip(anchor_header, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.62,
                    metadata=_relationship_metadata(element, comment_label=comment_label),
                ))
            if cell_label and anchor_value:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(cell_label, 160),
                    relation="has_value",
                    target=_clip(anchor_value, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.64,
                    metadata=_relationship_metadata(element, comment_label=comment_label),
                ))
            author = _clean_text(str(element.metadata.get("author") or ""))
            if author and element.text:
                relationships.append(GraphCandidateRelationship(
                    source=author,
                    relation="commented_on",
                    target=document_name,
                    evidence=_clip(element.text, 240),
                    confidence=0.62,
                    metadata=_relationship_metadata(element),
                ))
                if anchor_text:
                    relationships.append(GraphCandidateRelationship(
                        source=author,
                        relation="commented_on",
                        target=_clip(anchor_text, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.68,
                        metadata=_relationship_metadata(element, comment_label=comment_label),
                    ))

        if element.type == "revision":
            relationships.append(GraphCandidateRelationship(
                source=document_name,
                relation="has_revision",
                target=_clip(element.text, 160),
                evidence=_clip(element.text, 240),
                confidence=0.63,
                metadata=_relationship_metadata(element),
            ))
            author = _clean_text(str(element.metadata.get("revision_author") or ""))
            revision_type = _clean_text(str(element.metadata.get("revision_type") or "revised"))
            if author:
                relationships.append(GraphCandidateRelationship(
                    source=author,
                    relation=_relation_label(f"{revision_type}_in"),
                    target=document_name,
                    evidence=_clip(element.text, 240),
                    confidence=0.61,
                    metadata=_relationship_metadata(element),
                ))

        if element.type == "speaker_notes":
            relationships.append(GraphCandidateRelationship(
                source=document_name,
                relation="has_speaker_notes",
                target=_clip(element.text, 160),
                evidence=_clip(element.text, 240),
                confidence=0.61,
                metadata=_relationship_metadata(element),
            ))

        if element.type in {"footnote", "endnote"}:
            relationships.append(GraphCandidateRelationship(
                source=document_name,
                relation=f"has_{element.type}",
                target=_clip(element.text, 160),
                evidence=_clip(element.text, 240),
                confidence=0.62,
                metadata=_relationship_metadata(element),
            ))
            note_anchor_text = _clean_text(str(element.metadata.get("note_anchor_text") or ""))
            note_anchor_section = _clean_text(str(element.metadata.get("note_anchor_section") or ""))
            if note_anchor_section:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(note_anchor_section, 160),
                    relation=f"has_{element.type}",
                    target=_clip(element.text, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.66,
                    metadata=_relationship_metadata(element),
                ))
            if note_anchor_text:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(note_anchor_text, 160),
                    relation=f"has_{element.type}",
                    target=_clip(element.text, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.7,
                    metadata=_relationship_metadata(element),
                ))

        if element.type in {"header", "footer"}:
            relationships.append(GraphCandidateRelationship(
                source=document_name,
                relation=f"has_{element.type}",
                target=_clip(element.text, 120),
                evidence=_clip(element.text, 240),
                confidence=0.6,
                metadata=_relationship_metadata(element),
            ))

        if element.type == "image":
            label = _clean_text(str(
                element.metadata.get("title")
                or element.metadata.get("description")
                or element.metadata.get("name")
                or element.text
            ))
            sheet_name = _clean_text(str(element.metadata.get("sheet_name") or ""))
            if label:
                relationships.append(GraphCandidateRelationship(
                    source=document_name,
                    relation="contains_image",
                    target=_clip(label, 120),
                    evidence=_clip(element.text, 240),
                    confidence=0.58,
                    metadata=_relationship_metadata(element),
                ))
                if sheet_name:
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(sheet_name, 160),
                        relation="contains_image",
                        target=_clip(label, 120),
                        evidence=_clip(element.text, 240),
                        confidence=0.62,
                        metadata=_relationship_metadata(element),
                    ))
            ocr_text = _clean_text(str(element.metadata.get("ocr_text") or ""))
            if ocr_text:
                relationships.append(GraphCandidateRelationship(
                    source=_clip(label or document_name, 120),
                    relation="image_contains_text",
                    target=_clip(ocr_text, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.64,
                    metadata=_relationship_metadata(element),
                ))

        if element.type == "image_ocr_text":
            ocr_text = _clean_text(element.text)
            if ocr_text:
                relationships.append(GraphCandidateRelationship(
                    source=document_name,
                    relation="contains_text",
                    target=_clip(ocr_text, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.66,
                    metadata=_relationship_metadata(element),
                ))

        if element.type == "archive_entry":
            entry_label = _clean_text(str(
                element.metadata.get("archive_entry_path")
                or element.metadata.get("archive_entry_filename")
                or element.text
            ))
            if entry_label:
                relationships.append(GraphCandidateRelationship(
                    source=document_name,
                    relation="contains_archive_entry",
                    target=_clip(entry_label, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.64,
                    metadata=_relationship_metadata(element),
                ))

        if element.type in {"media", "embedded_object"}:
            label = _clean_text(str(
                element.metadata.get("uri")
                or element.metadata.get("embedded_file_filename")
                or element.metadata.get("embedded_file_name")
                or element.metadata.get("media_part")
                or element.text
            ))
            if label:
                relation = "contains_embedded_object" if element.type == "embedded_object" else "contains_media"
                relationships.append(GraphCandidateRelationship(
                    source=document_name,
                    relation=relation,
                    target=_clip(label, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.6,
                    metadata=_relationship_metadata(element),
                ))
                email_subject = _clean_text(str(element.metadata.get("email_subject") or ""))
                email_message_id = _clean_text(str(element.metadata.get("email_message_id") or ""))
                if element.metadata.get("email_attachment_filename") and (email_subject or email_message_id):
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(email_subject or email_message_id, 160),
                        relation="has_attachment",
                        target=_clip(label, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.68,
                        metadata=_relationship_metadata(element),
                    ))
            uri = _clean_text(str(element.metadata.get("uri") or ""))
            if uri:
                relationships.append(GraphCandidateRelationship(
                    source=document_name,
                    relation="links_to",
                    target=_clip(uri, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.64,
                    metadata=_relationship_metadata(element),
                ))

        if element.type == "link":
            uri = _clean_text(str(element.metadata.get("uri") or element.text))
            link_label = _clean_text(str(element.metadata.get("label") or ""))
            if not link_label:
                link_label = _clean_text(re.split(r"\s*(?:->|-)\s*", element.text, maxsplit=1)[0])
            link_label = link_label if link_label and link_label != uri else uri
            context = _container_context(document_name, element)
            group_label = _clean_text(str(element.metadata.get("parent_group_label") or element.metadata.get("group_label") or ""))
            sheet_name = _clean_text(str(element.metadata.get("sheet_name") or ""))
            if uri:
                relationships.append(GraphCandidateRelationship(
                    source=document_name,
                    relation="links_to",
                    target=_clip(uri, 160),
                    evidence=_clip(element.text, 240),
                    confidence=0.66,
                    metadata=_relationship_metadata(element),
                ))
                if context:
                    container_label, _, container_type = context
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(container_label, 160),
                        relation="contains_link",
                        target=_clip(link_label or uri, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.68,
                        metadata=_relationship_metadata(
                            element,
                            container_type=container_type,
                            container_label=container_label,
                        ),
                    ))
                if group_label:
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(group_label, 160),
                        relation="contains_link",
                        target=_clip(link_label or uri, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.7,
                        metadata=_relationship_metadata(
                            element,
                            container_type="shape_group",
                            container_label=group_label,
                        ),
                    ))
                if link_label and link_label != uri:
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(link_label, 160),
                        relation="links_to",
                        target=_clip(uri, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.72,
                        metadata=_relationship_metadata(element, link_label=link_label),
                    ))
                if sheet_name:
                    relationships.append(GraphCandidateRelationship(
                        source=_clip(sheet_name, 160),
                        relation="contains_link",
                        target=_clip(uri, 160),
                        evidence=_clip(element.text, 240),
                        confidence=0.66,
                        metadata=_relationship_metadata(element),
                    ))

        matched = [name for name in entity_names if name.lower() in element.text.lower()]
        if len(matched) >= 2:
            for left, right in zip(matched, matched[1:]):
                co_occurrence_relationships.append(GraphCandidateRelationship(
                    source=left,
                    relation="co_occurs_with",
                    target=right,
                    evidence=_clip(element.text, 240),
                    confidence=0.54,
                    metadata=_relationship_metadata(element),
                ))

        if len(relationships) >= limit:
            break

    deduped = []
    seen = set()
    visual_layout_relationships = _visual_layout_relationships(
        document_name,
        elements,
        limit=min(MAX_VISUAL_LAYOUT_RELATIONSHIPS, max(4, limit // 4)),
    )
    structural_relationships = _structural_relationships(document_name, elements)
    section_relationships = _section_relationships(document_name, elements)
    for relationship in [
        *semantic_relationships,
        *relationships,
        *chart_context_relationships,
        *visual_layout_relationships,
        *structural_relationships,
        *section_relationships,
        *co_occurrence_relationships,
    ]:
        key = (relationship.source.lower(), relationship.relation, relationship.target.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(relationship)
        if len(deduped) >= limit:
            break
    return deduped


def _relation_label(value: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return label or "related_to"


def _stage(name: str, status: str, start: float, **payload: Any) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "duration_ms": _duration_ms(start),
        **payload,
    }


def _sample_elements(elements: list[DocumentElement], limit: int) -> list[dict[str, Any]]:
    return [
        {
            **asdict(element),
            "text": _clip(element.text),
        }
        for element in elements[:limit]
    ]


def _sample_chunks(chunks: list[DocumentChunkPreview], limit: int) -> list[dict[str, Any]]:
    return [
        {
            **asdict(chunk),
            "text": _clip(chunk.text),
        }
        for chunk in chunks[:limit]
    ]


def _chunk_sample_categories(chunk: DocumentChunkPreview) -> list[str]:
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    scope_type = str(metadata.get("chunk_scope_type") or "").lower()
    semantic_searchable = " ".join(
        str(value)
        for value in (
            metadata.get("semantic_text"),
            metadata.get("evidence_preview"),
            chunk.text[:1600],
        )
        if value is not None
    )
    searchable = " ".join(
        str(value)
        for value in (
            scope_type,
            metadata.get("chunk_scope_label"),
            metadata.get("evidence_preview"),
            metadata.get("provenance"),
            metadata.get("source_locations"),
            chunk.text[:1200],
        )
        if value is not None
    ).lower()
    categories: list[str] = []

    def add(category: str, condition: bool) -> None:
        if condition and category not in categories:
            categories.append(category)

    add("semantic_relation", bool(SEMANTIC_RELATION_PATTERN.search(semantic_searchable)))
    add("connector", "shape_connector" in searchable or "connector" in searchable or "arrow" in searchable)
    add("message", "email_" in searchable or "[email_body]" in searchable or scope_type == "message")
    add(
        "diagram",
        bool(
            metadata.get("diagram_part")
            or metadata.get("diagram_node_id")
            or metadata.get("diagram_connection_type")
            or "[diagram_node]" in searchable
            or "[diagram_relationship]" in searchable
            or "smartart" in searchable
        ),
    )
    add("shape", bool(metadata.get("group_id") or metadata.get("parent_group_id") or "shape_group" in searchable))
    add("conditional_format", "conditional_format" in searchable or "conditional_" in searchable)
    add(
        "spreadsheet_layout",
        bool(
            metadata.get("heading_kind") == "spreadsheet_merged_cell"
            or "spreadsheet_merged_cell" in searchable
            or "merge_range" in searchable
        ),
    )
    add("formula", "formula" in searchable or "formula_cell" in searchable or "formula_references" in searchable)
    add("pivot_table", "pivot_table" in searchable or "pivot table" in searchable)
    if (
        "slide_visibility=hidden" in searchable
        or "slide_state=hidden" in searchable
        or "slide_visibility=visible" in searchable
        or "slide_part=" in searchable
        or "[slide_metadata]" in searchable
    ):
        add("presentation_structure", True)
    if (
        "sheet_visibility=hidden" in searchable
        or "sheet_state=hidden" in searchable
        or "visibility: hidden" in searchable
        or "sheet_visibility=veryhidden" in searchable
        or "sheet_state=veryhidden" in searchable
        or "visibility: veryhidden" in searchable
    ):
        add("workbook_structure", True)
    add("chart_source", scope_type in {"chart_source", "chart"} or "chart_source" in searchable or "chart_part" in searchable)
    add("table", scope_type == "table" or "[table_row]" in searchable or "table_name" in searchable or "table_source" in searchable)
    add("code_block", scope_type == "code_block" or "[code_block]" in searchable or "markdown_code_language" in searchable)
    add("styled_text", "styled_text" in searchable or "emphasis_terms" in searchable or "style_summary" in searchable)
    add("link", "[link]" in searchable or "link_kind" in searchable or "uri=" in searchable)
    add(
        "image",
        bool(
            metadata.get("image_part")
            or metadata.get("image_relationship_id")
            or "[image]" in searchable
            or "picture" in searchable
        ),
    )
    add("form_field", "form_field" in searchable or "control_part" in searchable or "field_type" in searchable)
    add("speaker_notes", "speaker_notes" in searchable or bool(metadata.get("notes_part")))
    add("note", "footnote" in searchable or "endnote" in searchable or "note_anchor" in searchable)
    add(
        "comment",
        bool(
            metadata.get("comment_id")
            or metadata.get("annotation_type")
            or metadata.get("comment_anchor_text")
            or "[comment]" in searchable
            or "[annotation]" in searchable
        ),
    )
    add(
        "revision",
        bool(
            metadata.get("revision_type")
            or metadata.get("revision_change_kind")
            or metadata.get("revision_id")
            or "[revision]" in searchable
        ),
    )
    add("ocr", "ocr_source" in searchable or "ocr_engine" in searchable or "[page_ocr_text]" in searchable or "[image_ocr_text]" in searchable)
    add("media", bool(metadata.get("media_part") or metadata.get("media_kind") or "[media]" in searchable))
    add("attachment", scope_type == "attachment" or "embedded_file" in searchable or "attachment" in searchable)
    add("archive_entry", scope_type == "archive_entry" or "archive_entry" in searchable)
    add("legacy_office", scope_type == "legacy_office_stream" or "legacy_office_" in searchable or "[legacy_office_text]" in searchable)
    add("slide", scope_type == "slide")
    add("page", scope_type == "page")
    add("sheet", scope_type == "sheet")
    add("mixed", scope_type == "mixed")
    return categories or ["document"]


def _chunk_sample_category(chunk: DocumentChunkPreview) -> str:
    return _chunk_sample_categories(chunk)[0]


def _chunk_sample_reason(category: str) -> str:
    return {
        "semantic_relation": "Chunk preserves direct subject-relation-object evidence for graph search and path answers",
        "connector": "Chunk preserves PowerPoint connector, arrow, or flow evidence for graph-path answers",
        "conditional_format": "Chunk preserves spreadsheet conditional formatting rules and thresholds",
        "formula": "Chunk preserves spreadsheet formulas and dependency references",
        "pivot_table": "Chunk preserves spreadsheet pivot table layout, source, and summary fields",
        "workbook_structure": "Chunk preserves workbook sheet visibility and source-part provenance",
        "presentation_structure": "Chunk preserves PowerPoint slide visibility and source-part provenance",
        "message": "Chunk preserves email headers, body text, links, or attachment context",
        "chart_source": "Chunk preserves chart source rows or metric evidence for embedding and GraphRAG ingest",
        "spreadsheet_layout": "Chunk preserves merged spreadsheet labels and range layout provenance",
        "table": "Chunk preserves table structure as subject-predicate-object evidence",
        "code_block": "Chunk preserves fenced code evidence with language and section provenance",
        "diagram": "Chunk preserves diagram or SmartArt graph evidence",
        "shape": "Chunk preserves grouped or described visual shape context",
        "link": "Chunk preserves source or action hyperlinks with clicked-object provenance",
        "image": "Chunk preserves image or picture evidence with relationship provenance",
        "form_field": "Chunk preserves form/control labels and values",
        "speaker_notes": "Chunk preserves presenter notes with slide and source-part provenance",
        "note": "Chunk preserves footnote or endnote evidence with source anchors",
        "comment": "Chunk preserves comment or annotation evidence with source anchors",
        "revision": "Chunk preserves inserted or deleted revision evidence",
        "ocr": "Chunk preserves OCR-derived text for scanned pages or images",
        "media": "Chunk preserves audio, video, or media evidence with package provenance",
        "styled_text": "Chunk preserves formatted or emphasized text",
        "attachment": "Chunk preserves embedded attachment evidence",
        "archive_entry": "Chunk preserves nested archive-entry evidence",
        "legacy_office": "Chunk preserves best-effort legacy Office OLE stream evidence",
        "slide": "Chunk preserves slide-local evidence and provenance",
        "page": "Chunk preserves page-local evidence and provenance",
        "sheet": "Chunk preserves worksheet-local evidence and provenance",
        "mixed": "Chunk combines multiple nearby semantic scopes",
        "document": "Representative document chunk",
    }.get(category, "Representative document chunk")


def _representative_chunks(chunks: list[DocumentChunkPreview], limit: int) -> list[DocumentChunkPreview]:
    if limit <= 0:
        return []

    preferred_categories = (
        "semantic_relation",
        "connector",
        "chart_source",
        "spreadsheet_layout",
        "table",
        "code_block",
        "link",
        "styled_text",
        "diagram",
        "image",
        "conditional_format",
        "formula",
        "pivot_table",
        "presentation_structure",
        "message",
        "speaker_notes",
        "workbook_structure",
        "form_field",
        "note",
        "comment",
        "revision",
        "ocr",
        "media",
        "attachment",
        "archive_entry",
        "legacy_office",
        "slide",
        "page",
        "sheet",
        "mixed",
        "document",
    )
    selected: list[DocumentChunkPreview] = []
    selected_ids: set[str] = set()
    selected_categories: set[str] = set()
    indexed = list(enumerate(chunks))
    preferred_rank = {category: index for index, category in enumerate(preferred_categories)}
    chunk_categories_by_id = {
        chunk.chunk_id: _chunk_sample_categories(chunk)
        for _, chunk in indexed
    }

    def chunk_score(chunk: DocumentChunkPreview) -> tuple[int, int, int]:
        metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        categories = chunk_categories_by_id.get(chunk.chunk_id, _chunk_sample_categories(chunk))
        category_score = max(
            (max(0, len(preferred_categories) - preferred_categories.index(category)) for category in categories if category in preferred_categories),
            default=0,
        )
        evidence_score = 2 if metadata.get("evidence_preview") else 0
        graphish_score = 1 if metadata.get("provenance") else 0
        return category_score, evidence_score, graphish_score

    ranked = sorted(indexed, key=lambda item: (*chunk_score(item[1]), -item[0]), reverse=True)

    def uncovered_family_score(chunk: DocumentChunkPreview) -> tuple[int, int, int, int, int]:
        categories = chunk_categories_by_id.get(chunk.chunk_id, _chunk_sample_categories(chunk))
        uncovered = [
            category
            for category in categories
            if category in preferred_rank and category not in selected_categories
        ]
        coverage_score = sum(len(preferred_categories) - preferred_rank[category] for category in uncovered)
        return len(uncovered), coverage_score, *chunk_score(chunk)

    def push(chunk: DocumentChunkPreview) -> bool:
        if len(selected) >= limit or chunk.chunk_id in selected_ids:
            return False
        selected_ids.add(chunk.chunk_id)
        selected.append(chunk)
        selected_categories.update(chunk_categories_by_id.get(chunk.chunk_id, _chunk_sample_categories(chunk)))
        return True

    for category in preferred_categories:
        if len(selected) >= limit:
            break
        if category in selected_categories:
            continue
        candidates = [
            (index, chunk)
            for index, chunk in ranked
            if chunk.chunk_id not in selected_ids
            and category in chunk_categories_by_id.get(chunk.chunk_id, _chunk_sample_categories(chunk))
        ]
        primary_candidates = [
            (index, chunk)
            for index, chunk in candidates
            if chunk_categories_by_id.get(chunk.chunk_id, _chunk_sample_categories(chunk))[0] == category
        ]
        candidate_pool = primary_candidates or candidates
        if candidate_pool:
            _, best_chunk = max(candidate_pool, key=lambda item: (*uncovered_family_score(item[1]), -item[0]))
            push(best_chunk)

    for _, chunk in ranked:
        push(chunk)
        if len(selected) >= limit:
            break

    return selected


def _chunk_evidence_cards(chunks: list[DocumentChunkPreview], limit: int) -> list[dict[str, Any]]:
    cards = []
    for chunk in _representative_chunks(chunks, limit):
        metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        category = _chunk_sample_category(chunk)
        categories = _chunk_sample_categories(chunk)
        scope_label = _clean_text(str(metadata.get("chunk_scope_label") or ""))
        if not scope_label and metadata.get("chunk_scope_labels"):
            scope_label = ", ".join(str(label) for label in metadata.get("chunk_scope_labels", [])[:3])
        source_locations = _unique_source_location_records(
            [
                item
                for item in metadata.get("source_locations", [])
                if isinstance(item, dict)
            ],
            limit=8,
        ) if isinstance(metadata.get("source_locations"), list) else []
        location_hints = _unique_preserve_order([
            str(value)
            for value in metadata.get("location_hints", [])
            if value
        ]) if isinstance(metadata.get("location_hints"), list) else []
        evidence_families = _unique_preserve_order([
            str(value)
            for value in metadata.get("evidence_families", [])
            if value
        ]) if isinstance(metadata.get("evidence_families"), list) else []
        cards.append({
            "chunk_id": chunk.chunk_id,
            "category": category,
            "categories": categories,
            "title": scope_label or chunk.chunk_id,
            "reason": _chunk_sample_reason(category),
            "scope_type": metadata.get("chunk_scope_type"),
            "scope_label": scope_label,
            "token_estimate": chunk.token_estimate,
            "evidence_preview": _clip(str(metadata.get("evidence_preview") or chunk.text), 420),
            "evidence_families": evidence_families[:12],
            "location_hints": location_hints[:8],
            "source_locations": source_locations,
            "metadata": metadata,
        })
    return cards


def _sample_graph_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    samples = []
    for record in records[:limit]:
        sample = dict(record)
        if "evidence" in sample:
            sample["evidence"] = _clip(str(sample.get("evidence") or ""))
        metadata = sample.get("metadata")
        if isinstance(metadata, dict):
            sample["metadata"] = {
                key: _clip(str(value), 220) if isinstance(value, str) else value
                for key, value in metadata.items()
            }
        properties = sample.get("properties")
        if isinstance(properties, dict):
            sample["properties"] = {
                key: _clip(str(value), 220) if isinstance(value, str) else value
                for key, value in properties.items()
            }
        samples.append(sample)
    return samples


def _is_direct_semantic_relationship(relationship: GraphCandidateRelationship) -> bool:
    metadata = relationship.metadata if isinstance(relationship.metadata, dict) else {}
    relation = str(relationship.relation or "").strip().lower()
    if not relation or relation in {"co_occurs_with", "co-occurs-with", "co occurs with", "related_to"}:
        return False
    if relation.startswith(("contains_", "has_")):
        return False
    if metadata.get("semantic_relation_extraction") in {"verb_pattern", "table_row_triple"}:
        return True
    if relation not in DIRECT_SEMANTIC_RELATIONS:
        return False
    return (
        _is_semantic_relation_endpoint(str(relationship.source or ""))
        and _is_semantic_relation_endpoint(str(relationship.target or ""))
    )


def _is_container_wrapper_relationship(relationship: GraphCandidateRelationship) -> bool:
    relation = str(relationship.relation or "").strip().lower()
    if relation in {"contains_connector", "contains_chart", "contains_table", "contains_form_field"}:
        return True
    return relation.startswith(("contains_", "has_"))


def _relationship_sample_category(relationship: GraphCandidateRelationship) -> str:
    metadata = relationship.metadata if isinstance(relationship.metadata, dict) else {}
    relation = str(relationship.relation or "").lower()
    searchable = " ".join(
        str(value)
        for value in (
            relationship.source,
            relationship.relation,
            relationship.target,
            relationship.evidence,
            *metadata.keys(),
            *metadata.values(),
        )
        if value is not None
    ).lower()

    if relation in {"co_occurs_with", "co-occurs-with", "co occurs with", "related_to"}:
        return "weak_mention"
    if _is_direct_semantic_relationship(relationship):
        return "semantic_connector"
    if metadata.get("layout_relation") or relation == "visually_near":
        return "layout"
    if (
        metadata.get("element_type") == "shape_connector"
        or metadata.get("connector_id")
        or metadata.get("source_shape_id")
        or metadata.get("target_shape_id")
        or "shape_connector" in searchable
    ):
        return "connector"
    if "connector" in searchable or relation in {"drives", "references", "connects_to"}:
        return "connector"
    if (
        metadata.get("diagram_part")
        or metadata.get("diagram_node_id")
        or metadata.get("diagram_connection_type")
        or relation in {"contains_diagram_node", "contains_diagram_relationship", "has_diagram_node", "has_diagram_relationship"}
    ):
        return "diagram"
    if (
        metadata.get("group_id")
        or metadata.get("parent_group_id")
        or relation in {"contains_shape_group", "groups_visual"}
    ):
        return "shape"
    if "conditional_format" in searchable or "conditional_" in searchable or relation in {"has_conditional_format", "evaluated_by", "uses_condition"}:
        return "conditional_format"
    if (
        metadata.get("heading_kind") == "spreadsheet_merged_cell"
        or "spreadsheet_merged_cell" in searchable
        or "merge_range" in searchable
        or relation in {"has_merged_label", "labeled_by"}
    ):
        return "spreadsheet_layout"
    if metadata.get("element_type") == "sheet_metadata" or "sheet_metadata" in searchable or relation in {"has_hidden_sheet", "has_sheet_visibility", "uses_sheet_part", "has_sheet_id"}:
        return "workbook_structure"
    if metadata.get("element_type") == "slide_metadata" or "slide_metadata" in searchable or relation in {"has_hidden_slide", "has_slide_visibility", "uses_slide_part", "has_slide_id"}:
        return "presentation_structure"
    if metadata.get("email_message_id") or "email_" in searchable or relation in {"sent_by", "sent_to", "cc_to", "bcc_to", "sent_message", "contains_email_body", "references_message", "in_reply_to"}:
        return "message"
    if relation in {"contains_heading", "has_section", "has_subsection"}:
        return "heading"
    if "formula" in searchable or relation in {"has_formula", "uses_formula", "calculates", "depends_on"}:
        return "formula"
    if metadata.get("pivot_table_name") or "pivot_table" in searchable or "pivot table" in searchable or relation == "contains_pivot_table":
        return "pivot_table"
    if (
        "chart" in searchable
        or relation.endswith("_value")
        or relation in {
            "contains_metric",
            "uses_chart_source",
            "contains_chart",
            "contains_chart_series",
            "contains_chart_point",
            "has_axis",
            "has_category",
            "has_data_labels",
            "has_legend",
            "has_trendline",
            "has_value",
        }
    ):
        return "chart_fact"
    if "table" in searchable or metadata.get("table_name") or metadata.get("table_source"):
        return "table_fact"
    if (
        metadata.get("element_type") == "code_block"
        or metadata.get("markdown_code_language")
        or relation in {"contains_code_block", "uses_language"}
    ):
        return "code_block"
    if metadata.get("uri") or metadata.get("link_kind") or relation in {"links_to", "contains_link"}:
        return "link"
    if (
        relation == "image_contains_text"
        or metadata.get("element_type") in {"image_ocr_text", "page_ocr_text"}
        or (relation == "contains_text" and (metadata.get("ocr_source") or metadata.get("ocr_engine")))
    ):
        return "ocr"
    if (
        relation == "contains_image"
        or metadata.get("element_type") == "image"
        or metadata.get("image_part")
        or metadata.get("image_relationship_id")
    ):
        return "image"
    if (
        "speaker_notes" in searchable
        or metadata.get("notes_part")
        or relation in {"has_speaker_notes", "contains_speaker_notes"}
    ):
        return "speaker_notes"
    if (
        "footnote" in searchable
        or "endnote" in searchable
        or metadata.get("note_id")
        or metadata.get("note_anchor_text")
        or relation in {"has_footnote", "has_endnote"}
    ):
        return "note"
    if (
        "comment" in searchable
        or metadata.get("comment_id")
        or metadata.get("comment_anchor_text")
        or relation in {"has_comment", "commented_on", "comments_on", "has_annotation", "annotated"}
    ):
        return "comment"
    if "revision" in searchable or relation in {"has_revision", "contains_revision", "inserted_in", "deleted_in"}:
        return "revision"
    if (
        metadata.get("element_type") == "form_field"
        or metadata.get("form_kind")
        or metadata.get("field_type")
        or relation in {"has_field", "contains_form_field"}
        or relation.endswith("_field")
    ):
        return "form_field"
    if metadata.get("styled_text") or relation in {"emphasizes", "uses_bullet_structure"}:
        return "styled_text"
    if (
        metadata.get("element_type") == "media"
        or relation == "contains_media"
        or (
            metadata.get("media_part")
            and not (
                metadata.get("embedded_file_filename")
                or metadata.get("embedded_file_name")
            )
        )
    ):
        return "media"
    if metadata.get("embedded_file_filename") or metadata.get("embedded_file_name") or "attachment" in searchable:
        return "attachment"
    if metadata.get("archive_entry_path") or metadata.get("archive_entry_filename") or "archive" in searchable:
        return "archive_entry"
    if metadata.get("legacy_office_stream_path") or "legacy_office_" in searchable:
        return "legacy_office"
    if metadata.get("metadata_field") or relation in OFFICE_METADATA_RELATIONS.values():
        return "document_metadata"
    if relation.startswith(("has_", "contains_")):
        return "layout"
    return "semantic"


def _relationship_sample_reason(category: str) -> str:
    return {
        "semantic_connector": "Direct semantic relationship extracted from document evidence",
        "connector": "PowerPoint connector or diagram edge preserved as a graph relationship",
        "diagram": "PowerPoint SmartArt or diagram nodes and edges promoted into graph evidence",
        "shape": "Grouped or described visual shapes promoted into graph evidence",
        "conditional_format": "Spreadsheet conditional formatting rule promoted into graph evidence",
        "spreadsheet_layout": "Spreadsheet merged label and range layout preserved as graph evidence",
        "workbook_structure": "Workbook sheet visibility and sheet-part provenance promoted into graph evidence",
        "presentation_structure": "PowerPoint slide visibility and slide-part provenance promoted into graph evidence",
        "message": "Email sender, recipient, body, link, or attachment evidence promoted into graph relationships",
        "heading": "Document heading or section boundary preserved as graph evidence",
        "formula": "Spreadsheet formula dependency preserved as graph evidence",
        "pivot_table": "Spreadsheet pivot table source and summary fields promoted into graph evidence",
        "chart_fact": "Chart series, chart source, or metric value promoted into graph evidence",
        "table_fact": "Structured table row converted into typed subject-predicate-object facts",
        "code_block": "Fenced code block preserved with language and section provenance",
        "form_field": "Form/control label and value preserved for graph search",
        "link": "Document hyperlink promoted into graph evidence with source context",
        "image": "Image, picture alt text, or visual asset provenance promoted into graph evidence",
        "ocr": "OCR-derived image or scanned-page text promoted into graph evidence",
        "speaker_notes": "Presenter notes anchored to slide and package source-part provenance",
        "note": "Footnote or endnote anchored to source document evidence",
        "comment": "Comment or annotation anchored to source document evidence",
        "revision": "Inserted or deleted text revision preserved as graph evidence",
        "styled_text": "Formatted or emphasized text surfaced as graph evidence",
        "media": "Audio, video, or packaged media evidence retained with source-part provenance",
        "attachment": "Embedded attachment or package evidence retained with provenance",
        "archive_entry": "Archive entry evidence retained with child provenance",
        "legacy_office": "Legacy Office OLE stream text retained with stream provenance",
        "document_metadata": "Document metadata promoted into graph provenance",
        "layout": "Visual or document layout relationship linking nearby slide, page, sheet, or section evidence",
        "weak_mention": "Low-confidence co-mention retained for recall after stronger evidence",
        "semantic": "Semantic relationship candidate extracted from document content",
    }.get(category, "Representative graph relationship candidate")


def _relationship_evidence_card_category(relationship: dict[str, Any]) -> str:
    category = str(relationship.get("sample_category") or "semantic")
    metadata = relationship.get("metadata") if isinstance(relationship.get("metadata"), dict) else {}
    relation = str(relationship.get("relation") or "").strip().lower()
    if category == "semantic_connector" and (
        metadata.get("element_type") == "shape_connector"
        or metadata.get("connector_id")
        or metadata.get("source_shape_id")
        or metadata.get("target_shape_id")
        or relation in {"drives", "references", "connects_to"}
    ):
        return "connector"
    return category


def _sample_relationships(relationships: list[GraphCandidateRelationship], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    scored = sorted(
        enumerate(relationships),
        key=lambda item: (
            _relationship_priority(item[1]),
            item[1].confidence,
            -item[0],
        ),
        reverse=True,
    )
    preferred_categories = (
        "semantic_connector",
        "connector",
        "diagram",
        "conditional_format",
        "spreadsheet_layout",
        "workbook_structure",
        "presentation_structure",
        "message",
        "heading",
        "speaker_notes",
        "formula",
        "pivot_table",
        "chart_fact",
        "table_fact",
        "code_block",
        "form_field",
        "link",
        "ocr",
        "note",
        "comment",
        "revision",
        "styled_text",
        "media",
        "attachment",
        "legacy_office",
        "document_metadata",
        "layout",
        "semantic",
        "weak_mention",
    )
    selected: list[GraphCandidateRelationship] = []
    selected_keys: set[tuple[str, str, str]] = set()

    def push(relationship: GraphCandidateRelationship) -> None:
        if len(selected) >= limit:
            return
        key = (relationship.source.lower(), relationship.relation.lower(), relationship.target.lower())
        if key in selected_keys:
            return
        selected_keys.add(key)
        selected.append(relationship)

    for category in preferred_categories:
        for _, relationship in scored:
            if _relationship_sample_category(relationship) == category:
                push(relationship)
                break

    preferred_category_minimums = {
        "semantic_connector": 3,
        "connector": 4,
        "chart_fact": 4,
        "table_fact": 4,
        "diagram": 3,
        "form_field": 2,
        "link": 2,
        "styled_text": 2,
    }
    category_counts = Counter(
        _relationship_sample_category(relationship)
        for relationship in selected
    )
    for category, minimum in preferred_category_minimums.items():
        if len(selected) >= limit:
            break
        for _, relationship in scored:
            if category_counts[category] >= minimum:
                break
            if _relationship_sample_category(relationship) != category:
                continue
            before_count = len(selected)
            push(relationship)
            if len(selected) > before_count:
                category_counts[category] += 1
            if len(selected) >= limit:
                break

    for _, relationship in scored:
        push(relationship)
        if len(selected) >= limit:
            break

    samples = []
    for relationship in selected:
        category = _relationship_sample_category(relationship)
        samples.append({
            **asdict(relationship),
            "sample_category": category,
            "sample_reason": _relationship_sample_reason(category),
            "priority": _relationship_priority(relationship),
        })
    return samples


def _relationship_evidence_cards(relationships: list[GraphCandidateRelationship], limit: int) -> list[dict[str, Any]]:
    cards = []
    for relationship in _sample_relationships(relationships, limit):
        metadata = relationship.get("metadata") if isinstance(relationship.get("metadata"), dict) else {}
        provenance = _metadata_line(metadata)
        category = _relationship_evidence_card_category(relationship)
        evidence_preview = _clip(
            " | ".join(
                part
                for part in [
                    str(relationship.get("evidence") or ""),
                    provenance,
                ]
                if _clean_text(part)
            ),
            420,
        )
        cards.append({
            "category": category,
            "title": f"{relationship.get('source')} --[{relationship.get('relation')}]--> {relationship.get('target')}",
            "reason": _relationship_sample_reason(category),
            "evidence": _clip(str(relationship.get("evidence") or ""), 320),
            "evidence_preview": evidence_preview,
            "confidence": relationship.get("confidence"),
            "priority": relationship.get("priority"),
            "provenance": _clip(provenance, 360) if provenance else "",
            "metadata": metadata,
        })
    return cards


def _preview_relationship_samples(relationships: list[GraphCandidateRelationship], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    samples = [asdict(relationship) for relationship in relationships[:limit]]
    selected_keys = {
        (
            str(sample.get("source") or "").lower(),
            str(sample.get("relation") or "").lower(),
            str(sample.get("target") or "").lower(),
        )
        for sample in samples
    }
    extra_limit = limit + min(24, max(4, limit // 2))

    def push_sample(relationship: GraphCandidateRelationship | dict[str, Any]) -> None:
        if len(samples) >= extra_limit:
            return
        sample = asdict(relationship) if isinstance(relationship, GraphCandidateRelationship) else dict(relationship)
        key = (
            str(sample.get("source") or "").lower(),
            str(sample.get("relation") or "").lower(),
            str(sample.get("target") or "").lower(),
        )
        if key in selected_keys:
            return
        selected_keys.add(key)
        for sample_only_key in ("sample_category", "sample_reason", "priority"):
            sample.pop(sample_only_key, None)
        samples.append(sample)

    coverage_relations = {
        "contains_heading",
        "has_section",
        "has_subsection",
        "contains_revision",
        "has_revision",
        "inserted_in",
        "deleted_in",
        "has_footnote",
        "has_endnote",
        "contains_table",
        "contains_chart",
        "contains_chart_series",
        "contains_chart_point",
        "image_contains_text",
        "contains_diagram_node",
        "contains_diagram_relationship",
    }
    for relationship in relationships:
        if relationship.relation in coverage_relations:
            push_sample(relationship)
            if len(samples) >= extra_limit:
                break

    for relationship in _sample_relationships(relationships, limit):
        push_sample(relationship)
        if len(samples) >= extra_limit:
            break

    return samples


def _embedding_input_sample(
    chunk: DocumentChunkPreview,
    *,
    filename: str,
    domain: str,
    content_type: str | None,
    sha256: str,
    relationships: list[GraphCandidateRelationship],
    local_graph_reasoning: dict[str, Any] | None,
) -> dict[str, Any]:
    graph_context = _chunk_graph_context(chunk, relationships)
    graph_paths = _chunk_graph_path_context(chunk, local_graph_reasoning)
    graph_context_reasons = _unique_preserve_order([
        str(reason)
        for relationship in graph_context
        for reason in relationship.get("chunk_match_reasons", [])
        if reason
    ])
    graph_path_reasons = _unique_preserve_order([
        str(reason)
        for path in graph_paths
        for reason in path.get("chunk_match_reasons", [])
        if reason
    ])
    sample_categories = _chunk_sample_categories(chunk)
    sample_category = sample_categories[0]
    return {
        "chunk_id": chunk.chunk_id,
        "text": _enriched_chunk_text(chunk, relationships, local_graph_reasoning=local_graph_reasoning),
        "metadata": {
            **chunk.metadata,
            "filename": filename,
            "domain": domain,
            "source_elements": chunk.source_elements,
            "content_type": content_type,
            "sha256": sha256,
            "sample_category": sample_category,
            "sample_categories": sample_categories,
            "sample_reason": _chunk_sample_reason(sample_category),
            "graph_context_relationship_count": len(graph_context),
            "graph_path_context_count": len(graph_paths),
            "graph_context_max_match_score": max(
                (int(edge.get("chunk_match_score", 0) or 0) for edge in graph_context),
                default=0,
            ),
            "graph_path_max_match_score": max(
                (int(path.get("chunk_match_score", 0) or 0) for path in graph_paths),
                default=0,
            ),
            "graph_context_match_reasons": graph_context_reasons[:12],
            "graph_path_match_reasons": graph_path_reasons[:12],
            "graph_context": graph_context,
            "graph_paths": graph_paths,
        },
    }


def _embedding_evidence_cards(embedding_inputs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    cards = []
    for item in embedding_inputs[:limit]:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        category = str(metadata.get("sample_category") or "document")
        categories = [
            str(value)
            for value in metadata.get("sample_categories", [category])
            if value
        ] if isinstance(metadata.get("sample_categories"), list) else [category]
        scope_label = _clean_text(str(metadata.get("chunk_scope_label") or ""))
        if not scope_label and isinstance(metadata.get("chunk_scope_labels"), list):
            scope_label = ", ".join(str(label) for label in metadata.get("chunk_scope_labels", [])[:3])
        graph_context_count = int(metadata.get("graph_context_relationship_count", 0) or 0)
        graph_path_count = int(metadata.get("graph_path_context_count", 0) or 0)
        source_locations = _unique_source_location_records(
            [
                source_location
                for source_location in metadata.get("source_locations", [])
                if isinstance(source_location, dict)
            ],
            limit=8,
        ) if isinstance(metadata.get("source_locations"), list) else []
        location_hints = _unique_preserve_order([
            str(value)
            for value in metadata.get("location_hints", [])
            if value
        ]) if isinstance(metadata.get("location_hints"), list) else []
        evidence_families = _unique_preserve_order([
            str(value)
            for value in metadata.get("evidence_families", [])
            if value
        ]) if isinstance(metadata.get("evidence_families"), list) else []
        cards.append({
            "chunk_id": item.get("chunk_id"),
            "category": category,
            "categories": categories,
            "title": scope_label or str(item.get("chunk_id") or "embedding input"),
            "reason": metadata.get("sample_reason") or _chunk_sample_reason(category),
            "scope_type": metadata.get("chunk_scope_type"),
            "scope_label": scope_label,
            "graph_context_relationship_count": graph_context_count,
            "graph_path_context_count": graph_path_count,
            "evidence_preview": _clip(str(metadata.get("evidence_preview") or item.get("text") or ""), 420),
            "evidence_families": evidence_families[:12],
            "location_hints": location_hints[:8],
            "source_locations": source_locations,
            "metadata": metadata,
        })
    return cards


def _sample_local_context_chunks(local_context: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    chunks = local_context.get("chunks", []) if isinstance(local_context, dict) else []
    if not isinstance(chunks, list):
        return []
    samples = []
    for chunk in chunks[:limit]:
        if not isinstance(chunk, dict):
            continue
        graph_context = chunk.get("graph_context", [])
        graph_paths = chunk.get("graph_paths", [])
        samples.append({
            **chunk,
            "content": _clip(str(chunk.get("content") or ""), 1200),
            "graph_context": _sample_graph_records(graph_context, limit) if isinstance(graph_context, list) else [],
            "graph_paths": [
                {
                    **path,
                    "text": _clip(str(path.get("text") or ""), 520),
                    "relationships": _sample_graph_records(path.get("relationships", []), limit)
                    if isinstance(path.get("relationships"), list)
                    else [],
                }
                for path in graph_paths[:limit]
                if isinstance(path, dict)
            ] if isinstance(graph_paths, list) else [],
        })
    return samples


def _sample_local_context_graph_context(local_context: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    chunks = local_context.get("chunks", []) if isinstance(local_context, dict) else []
    graph_context: list[dict[str, Any]] = []
    if not isinstance(chunks, list):
        return graph_context
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for relationship in chunk.get("graph_context", []) or []:
            if isinstance(relationship, dict):
                graph_context.append(relationship)
            if len(graph_context) >= limit:
                return _sample_graph_records(graph_context, limit)
    return _sample_graph_records(graph_context, limit)


def _sample_local_context_graph_paths(local_context: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    chunks = local_context.get("chunks", []) if isinstance(local_context, dict) else []
    graph_paths: list[dict[str, Any]] = []
    if not isinstance(chunks, list):
        return graph_paths
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for path in chunk.get("graph_paths", []) or []:
            if isinstance(path, dict):
                graph_paths.append({
                    **path,
                    "text": _clip(str(path.get("text") or ""), 520),
                    "relationships": _sample_graph_records(
                        path.get("relationships", []),
                        limit,
                    ) if isinstance(path.get("relationships"), list) else [],
                })
            if len(graph_paths) >= limit:
                return graph_paths[:limit]
    return graph_paths[:limit]


def _stable_question_id(*parts: str) -> str:
    text = "_".join(_clean_text(part).lower() for part in parts if part)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:96] or "question"


def _format_capability_label(value: str) -> str:
    return re.sub(r"\b\w", lambda match: match.group(0).upper(), str(value or "").replace("_", " "))


def _relationship_priority(relationship: GraphCandidateRelationship) -> int:
    metadata = relationship.metadata or {}
    searchable = " ".join(
        [
            relationship.source,
            relationship.target,
            relationship.relation,
            relationship.evidence,
            *(str(value) for value in metadata.values() if value is not None),
        ]
    ).lower()
    relation = str(relationship.relation or "").lower()
    if relation in {"co_occurs_with", "co-occurs-with", "co occurs with", "related_to"}:
        return 8
    if _is_direct_semantic_relationship(relationship):
        return 74
    if relation == "visually_near" or metadata.get("layout_relation"):
        return 57
    if relation == "contains_table" and metadata.get("table_cell_reference"):
        return 58
    if relation == "contains_chart":
        return 56
    if relation == "contains_connector":
        return 56
    if relation == "contains_form_field":
        return 54
    if _is_container_wrapper_relationship(relationship):
        return 42
    if re.search(r"connector|arrow|line|flow|shape_connector", searchable):
        return 60
    if relation == "emphasizes" or re.search(r"emphasis|styled_text|style_summary|bold|italic|underline", searchable):
        return 55
    if metadata.get("uri") or relation in {"links_to", "contains_link"}:
        return 52
    if metadata.get("markdown_code_language") or relation in {"contains_code_block", "uses_language"}:
        return 52
    if metadata.get("pivot_table_name") or "pivot_table" in searchable or "pivot table" in searchable or relation == "contains_pivot_table":
        return 52
    if metadata.get("element_type") == "sheet_metadata" or relation in {"has_hidden_sheet", "has_sheet_visibility", "uses_sheet_part", "has_sheet_id"}:
        return 51
    if metadata.get("element_type") == "slide_metadata" or relation in {"has_hidden_slide", "has_slide_visibility", "uses_slide_part", "has_slide_id"}:
        return 51
    if metadata.get("notes_part") or relation in {"has_speaker_notes", "contains_speaker_notes"}:
        return 51
    if metadata.get("email_message_id") or relation in {"sent_by", "sent_to", "cc_to", "contains_email_body", "sent_message", "has_attachment"}:
        return 51
    if metadata.get("legacy_office_stream_path") or "legacy_office_" in searchable:
        return 51
    if metadata.get("group_id") or metadata.get("parent_group_id") or relation in {"contains_shape_group", "groups_visual"}:
        return 51
    if re.search(r"chart|table|form|field|image|figure|diagram|code|speaker|notes|embedded", searchable):
        return 50
    if not relation.startswith(("has_", "contains_")):
        return 40
    if relationship.evidence:
        return 30
    return 10


def _relationship_location_hints(metadata: dict[str, Any]) -> dict[str, Any]:
    hint_keys = (
        "email_message_id",
        "email_subject",
        "email_from",
        "email_sender",
        "email_to",
        "email_cc",
        "email_date",
        "email_part_index",
        "email_part_kind",
        "email_content_type",
        "email_attachment_index",
        "email_attachment_filename",
        "email_attachment_mime",
        "metadata_part",
        "metadata_field",
        "metadata_label",
        "metadata_value",
        "page",
        "slide",
        "slide_title",
        "slide_id",
        "slide_part",
        "slide_relationship_id",
        "slide_state",
        "slide_visibility",
        "section",
        "heading_kind",
        "merge_range",
        "merge_anchor_cell",
        "merge_cell_count",
        "merge_row_count",
        "merge_column_count",
        "markdown_code_language",
        "markdown_code_block_index",
        "markdown_start_line",
        "markdown_end_line",
        "notes_part",
        "notes_relationship_id",
        "sheet_name",
        "sheet_id",
        "sheet_part",
        "sheet_relationship_id",
        "sheet_state",
        "sheet_visibility",
        "location",
        "table",
        "table_name",
        "table_display_name",
        "table_definition_name",
        "table_part",
        "table_range",
        "table_columns",
        "table_relationship_id",
        "table_cell_reference",
        "table_cell_column_header",
        "table_cell_row_header",
        "table_cell_context",
        "table_cell_grid_span",
        "table_cell_row_span",
        "table_cell_width_value",
        "table_cell_h_merge",
        "table_cell_v_merge",
        "table_cell_merge_flags",
        "table_cell_fill_color",
        "table_cell_style_name",
        "table_cell_visual_summary",
        "table_cell_text_anchor",
        "table_cell_vertical_align",
        "table_cell_text_direction",
        "table_cell_width_value",
        "table_cell_width_inches",
        "table_cell_width_type",
        "table_cell_shading_pattern",
        "table_cell_margin_left_inches",
        "table_cell_margin_right_inches",
        "table_cell_margin_top_inches",
        "table_cell_margin_bottom_inches",
        "table_cell_border_left_color",
        "table_cell_border_right_color",
        "table_cell_border_top_color",
        "table_cell_border_bottom_color",
        "table_cell_border_left_width_inches",
        "table_cell_border_right_width_inches",
        "table_cell_border_top_width_inches",
        "table_cell_border_bottom_width_inches",
        "table_cell_column_repeat",
        "table_cell_row_repeat",
        "table_cell_is_covered",
        "cell_value_type",
        "row",
        "column",
        "cell",
        "formula_cell",
        "formula_expression",
        "formula_references",
        "formula_result",
        "formula_reference_count",
        "conditional_label",
        "conditional_range",
        "conditional_range_label",
        "conditional_type",
        "conditional_operator",
        "conditional_priority",
        "conditional_dxf_id",
        "conditional_stop_if_true",
        "conditional_text",
        "conditional_time_period",
        "conditional_rank",
        "conditional_percent",
        "conditional_formula1",
        "conditional_formula2",
        "conditional_formulas",
        "conditional_formula_references",
        "conditional_color_scale",
        "validation_range",
        "validation_type",
        "validation_operator",
        "validation_formula1",
        "validation_formula2",
        "validation_allow_blank",
        "validation_prompt_title",
        "validation_prompt",
        "validation_error_title",
        "validation_error",
        "named_range_name",
        "named_range_ref",
        "named_range_scope",
        "named_range_local_sheet_id",
        "named_range_hidden",
        "named_range_comment",
        "pivot_table_name",
        "pivot_table_range",
        "pivot_table_source_name",
        "pivot_table_source_ref",
        "pivot_table_source_sheet",
        "pivot_table_row_fields",
        "pivot_table_column_fields",
        "pivot_table_page_fields",
        "pivot_table_data_fields",
        "pivot_table_cache_part",
        "pivot_table_part",
        "pivot_table_cache_fields",
        "pivot_table_cache_id",
        "pivot_table_relationship_id",
        "pivot_table_source_type",
        "pivot_table_cache_relationship_id",
        "pivot_table_data_field_count",
        "pivot_table_record_count",
        "form_kind",
        "field_type",
        "label",
        "form_field_checked",
        "form_field_options",
        "form_field_selected_option",
        "form_field_alias",
        "form_field_tag",
        "form_field_date_format",
        "form_field_date_full_value",
        "form_field_calendar",
        "control_id",
        "control_part",
        "chart_title",
        "chart_part",
        "chart_label",
        "chart_point_label",
        "chart_axis_index",
        "chart_axis_type",
        "chart_axis_id",
        "chart_axis_title",
        "chart_axis_position",
        "chart_axis_number_format",
        "chart_legend_position",
        "chart_legend_overlay",
        "chart_legend_series",
        "chart_label_fields",
        "chart_label_position",
        "chart_label_examples",
        "chart_label_number_format",
        "chart_series_fill_color",
        "chart_series_line_color",
        "chart_series_line_width_points",
        "chart_series_line_dash",
        "chart_series_visual_summary",
        "chart_marker_symbol",
        "chart_marker_size",
        "chart_marker_fill_color",
        "chart_marker_line_color",
        "chart_marker_line_width_points",
        "chart_point_fill_color",
        "chart_point_line_color",
        "chart_point_visual_summary",
        "chart_trendline_type",
        "chart_trendline_name",
        "chart_trendline_line_color",
        "chart_trendline_line_width_points",
        "chart_trendline_line_dash",
        "chart_trendline_display_equation",
        "chart_trendline_display_r_squared",
        "chart_trendline_visual_summary",
        "series_name",
        "category",
        "value",
        "chart_source_part",
        "chart_source_extraction_method",
        "chart_source_kind",
        "diagram_part",
        "diagram_node_id",
        "diagram_connection_type",
        "source_diagram_node_id",
        "target_diagram_node_id",
        "group_index",
        "group_id",
        "group_name",
        "group_title",
        "group_description",
        "group_label",
        "group_child_shape_ids",
        "group_child_labels",
        "group_child_count",
        "grouped_visual_label",
        "parent_group_id",
        "parent_group_label",
        "parent_group_index",
        "style_summary",
        "emphasis_terms",
        "emphasis_term",
        "emphasis_styles",
        "bullet_levels",
        "styled_text",
        "inherited_from",
        "inherited_part",
        "layout_part",
        "layout_relationship_id",
        "master_part",
        "master_relationship_id",
        "layout_bbox",
        "layout_position",
        "layout_zone",
        "layout_relation",
        "visual_distance_pct",
        "related_element_type",
        "related_label",
        "related_location",
        "target_layout_zone",
        "target_layout_bbox",
        "connector_id",
        "connector_name",
        "connector_title",
        "connector_description",
        "source_label",
        "target_label",
        "source_shape_id",
        "target_shape_id",
        "source_connection_index",
        "target_connection_index",
        "shape_title",
        "shape_description",
        "shape_id",
        "text_box_part",
        "text_box_kind",
        "image_part",
        "image_relationship_id",
        "image_mime",
        "uri",
        "relationship_id",
        "target_mode",
        "link_kind",
        "media_part",
        "media_kind",
        "media_mime",
        "embedded_file_name",
        "embedded_file_filename",
        "embedded_file_mime",
        "comment_id",
        "comment_kind",
        "comment_part",
        "comment_anchor_text",
        "comment_anchor_section",
        "comment_anchor_paragraph",
        "comment_anchor_sheet",
        "comment_anchor_cell",
        "comment_anchor_header",
        "comment_anchor_value",
        "threaded_comment_id",
        "threaded_parent_id",
        "person_id",
        "comment_datetime",
        "author_id",
        "author",
        "revision_type",
        "revision_author",
        "revision_date",
        "revision_id",
        "revision_part",
        "revision_change_kind",
        "revision_anchor_text",
        "archive_entry_path",
        "legacy_office_format",
        "legacy_office_stream_index",
        "legacy_office_stream_path",
        "legacy_office_extraction_method",
        "legacy_office_string_count",
        "element_type",
    )
    hints = {
        key: _clip(str(metadata[key]), 160) if isinstance(metadata.get(key), str) else metadata[key]
        for key in hint_keys
        if metadata.get(key) not in (None, "")
    }
    if not hints.get("uri"):
        for uri_key in ("href", "src"):
            uri_value = _clean_text(str(metadata.get(uri_key) or ""))
            if uri_value:
                hints["uri"] = _clip(uri_value, 160)
                break
    if metadata.get("author") and not hints.get("comment_author"):
        hints["comment_author"] = _clip(str(metadata["author"]), 160)
    return hints


def _push_interaction_suggestion(
    suggestions: list[dict[str, Any]],
    suggestion: dict[str, Any],
    *,
    limit: int = MAX_INTERACTION_SUGGESTIONS,
) -> None:
    if len(suggestions) >= limit:
        return
    prompt_key = str(suggestion.get("prompt") or "").lower()
    label_key = str(suggestion.get("label") or "").lower()
    if any(
        prompt_key == str(existing.get("prompt") or "").lower()
        or label_key == str(existing.get("label") or "").lower()
        for existing in suggestions
    ):
        return
    suggestions.append(suggestion)


def _relationship_suggestion(filename: str, relationship: GraphCandidateRelationship) -> dict[str, Any]:
    metadata = relationship.metadata or {}
    prompt = (
        f'Using the GraphRAG pipeline trace for {filename}, explain how "{relationship.source}" '
        f'connects to "{relationship.target}" through "{relationship.relation}". Find the strongest '
        "direct or multi-hop graph path, cite the trace evidence, and include any slide/page/table/chart/"
        "connector provenance."
    )
    if relationship.evidence:
        prompt += f' The extracted evidence snippet was: "{_clip(relationship.evidence, 220)}".'
    return {
        "id": f"relationship_{_stable_question_id(relationship.source, relationship.target, relationship.relation)}",
        "kind": "relationship",
        "label": _clip(f"{relationship.source} to {relationship.target}", 64),
        "prompt": prompt,
        "source": relationship.source,
        "target": relationship.target,
        "relation": relationship.relation,
        "evidence": _clip(relationship.evidence, 280) if relationship.evidence else "",
        "provenance_hints": _relationship_location_hints(metadata),
        "priority": _relationship_priority(relationship),
    }


def _is_styled_text_relationship(relationship: GraphCandidateRelationship) -> bool:
    metadata = relationship.metadata if isinstance(relationship.metadata, dict) else {}
    return bool(
        metadata.get("styled_text")
        or metadata.get("style_summary")
        or metadata.get("emphasis_term")
        or metadata.get("emphasis_terms")
        or relationship.relation in {"emphasizes", "uses_bullet_structure"}
    )


def _styled_text_relationship_suggestion(
    suggestion: dict[str, Any],
    relationship: GraphCandidateRelationship,
) -> dict[str, Any]:
    metadata = relationship.metadata if isinstance(relationship.metadata, dict) else {}
    emphasis_term = _clean_text(str(metadata.get("emphasis_term") or relationship.target or ""))
    style_summary = _clean_text(str(metadata.get("style_summary") or ""))
    styles = metadata.get("emphasis_styles") if isinstance(metadata.get("emphasis_styles"), list) else []
    style_parts = [
        f'style summary "{style_summary}"' if style_summary else "",
        f"styles {', '.join(_clean_text(str(style)) for style in styles if _clean_text(str(style)))}" if styles else "",
    ]
    style_text = "; ".join(part for part in style_parts if part)
    focused_prompt = (
        f'Using the GraphRAG pipeline trace, start with the emphasized term "{emphasis_term}" '
        f'from the graph edge "{relationship.source}" --[{relationship.relation}]--> "{relationship.target}". '
        "Explain why this styled term matters, which entities or paths it connects to, and cite the exact "
        "style plus slide/page/table/chunk provenance."
    )
    if style_text:
        focused_prompt += f" Preserve the extracted {style_text}."
    if relationship.evidence:
        focused_prompt += f' Evidence snippet: "{_clip(relationship.evidence, 220)}".'

    return {
        **suggestion,
        "prompt": focused_prompt,
        "featured_emphasis_term": emphasis_term,
        "source": relationship.source,
        "target": relationship.target,
        "relation": relationship.relation,
        "style_summary": style_summary,
        "emphasis_styles": styles[:8],
        "evidence": _clip(relationship.evidence, 280) if relationship.evidence else suggestion.get("evidence", ""),
        "provenance_hints": _relationship_location_hints(metadata),
        "priority": _relationship_priority(relationship),
    }


def _path_candidate_suggestion(filename: str, path: dict[str, Any]) -> dict[str, Any]:
    source = _clean_text(str(path.get("source") or "source"))
    bridge = _clean_text(str(path.get("bridge") or "bridge entity"))
    target = _clean_text(str(path.get("target") or "target"))
    prompt = (
        f'Using the GraphRAG pipeline trace for {filename}, explain the two-hop graph path '
        f'"{source}" -> "{bridge}" -> "{target}". Walk each relationship, explain why "{bridge}" '
        "is the bridge entity, cite the extracted evidence, and include slide/page/table/chart/"
        "connector provenance when available."
    )
    if path.get("text"):
        prompt += f' Candidate path: "{_clip(str(path["text"]), 260)}".'
    return {
        "id": f"path_{_stable_question_id(source, bridge, target)}",
        "kind": "path",
        "label": _clip(f"Path: {source} via {bridge}", 64),
        "prompt": prompt,
        "source": source,
        "bridge": bridge,
        "target": target,
        "nodes": path.get("nodes") if isinstance(path.get("nodes"), list) else [source, bridge, target],
        "relationships": path.get("relationships") if isinstance(path.get("relationships"), list) else [],
        "score": path.get("score"),
    }


def _evidence_interaction_suggestions(filename: str, capabilities: set[str]) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []

    if {"connectors", "shapes"} & capabilities:
        suggestions.append({
            "id": "evidence_audit_connectors",
            "kind": "evidence",
            "label": "Audit connectors",
            "prompt": (
                f"Using the GraphRAG pipeline trace for {filename}, inspect the extracted connectors, "
                "arrows, lines, and flow relationships. Explain what they connect, where they appear, "
                "and cite slide or shape provenance for each important link."
            ),
            "focus": "connectors",
        })
    if "presentation_structure" in capabilities:
        suggestions.append({
            "id": "evidence_inspect_slide_structure",
            "kind": "evidence",
            "label": "Inspect slide structure",
            "prompt": (
                f"Using the GraphRAG pipeline trace for {filename}, inspect PowerPoint slide structure. "
                "Identify hidden slides, slide source parts, relationship IDs, and any slide-local evidence "
                "that should be cited in graph-grounded answers."
            ),
            "focus": "presentation_structure",
        })
    if "diagrams" in capabilities:
        suggestions.append({
            "id": "evidence_audit_diagrams",
            "kind": "evidence",
            "label": "Audit diagrams",
            "prompt": (
                f"Using the GraphRAG pipeline trace for {filename}, inspect the extracted diagram and SmartArt "
                "nodes, levels, parent/child edges, and graph relationships. Explain which entities the diagram "
                "connects, where it appears, and cite diagram part plus slide provenance."
            ),
            "focus": "diagrams",
        })
    if "ocr" in capabilities:
        suggestions.append({
            "id": "evidence_review_ocr",
            "kind": "evidence",
            "label": "Review OCR evidence",
            "prompt": (
                f"Using the GraphRAG pipeline trace for {filename}, review OCR-derived evidence from scanned "
                "pages or images. Identify the extracted text, any graph relationships it created, confidence "
                "or quality warnings, and cite page/image/OCR provenance."
            ),
            "focus": "ocr",
        })
    if "styled_text" in capabilities:
        suggestions.append({
            "id": "evidence_inspect_emphasis",
            "kind": "evidence",
            "label": "Inspect emphasized text",
            "prompt": (
                f"Using the GraphRAG pipeline trace for {filename}, inspect the extracted styled text, "
                "bold/italic/underlined terms, bullet hierarchy, and emphasis relationships. Explain what "
                "the emphasized terms highlight, which entities they connect to, and cite style plus "
                "slide/page/chunk provenance."
            ),
            "focus": "styled_text",
        })
    if "messages" in capabilities:
        suggestions.append({
            "id": "evidence_trace_message",
            "kind": "evidence",
            "label": "Trace message",
            "prompt": (
                f"Using the GraphRAG pipeline trace for {filename}, trace the email message evidence. "
                "Identify sender, recipients, subject, body claims, links, and attachments; explain which "
                "child documents or table facts came from attachments; and cite message headers, body part, "
                "link, and attachment provenance."
            ),
            "focus": "messages",
        })
    if "links" in capabilities:
        suggestions.append({
            "id": "evidence_trace_links",
            "kind": "evidence",
            "label": "Trace links",
            "prompt": (
                f"Using the GraphRAG pipeline trace for {filename}, trace extracted hyperlinks and action links. "
                "Explain which document objects or labels link to each URI, identify any clicked image/shape/table "
                "provenance, and cite relationship IDs, slide/page layout, and source context for each important link."
            ),
            "focus": "links",
        })
    if {"charts", "tables", "formulas"} & capabilities:
        suggestions.append({
            "id": "evidence_explain_charts_tables",
            "kind": "evidence",
            "label": "Explain charts and tables",
            "prompt": (
                f"Using the GraphRAG pipeline trace for {filename}, explain the key chart and table evidence. "
                "Identify the entities or metrics they connect to, and cite chart/table/page/slide provenance "
                "from the extraction."
            ),
            "focus": "charts_tables",
        })
    if "layout" in capabilities:
        suggestions.append({
            "id": "evidence_inspect_layout",
            "kind": "evidence",
            "label": "Inspect layout evidence",
            "prompt": (
                f"Using the GraphRAG pipeline trace for {filename}, inspect the extracted layout evidence. "
                "Identify important page or slide regions, readable bboxes, zones, nearby tables/forms/images/"
                "links, and explain how those spatial anchors should be cited in graph-grounded answers."
            ),
            "focus": "layout",
        })
    if {"archives", "embedded_objects", "media"} & capabilities:
        suggestions.append({
            "id": "evidence_trace_source_artifacts",
            "kind": "evidence",
            "label": "Trace source artifacts",
            "prompt": (
                f"Using the GraphRAG pipeline trace for {filename}, inspect embedded files, archive entries, "
                "attachments, and media artifacts. Explain which child documents or source artifacts were "
                "parsed, what graph facts they contributed, and cite filename/path, MIME type, checksum, "
                "and parent document provenance when available."
            ),
            "focus": "source_artifacts",
        })
    if "legacy_office" in capabilities:
        suggestions.append({
            "id": "evidence_trace_legacy_office",
            "kind": "evidence",
            "label": "Trace legacy Office",
            "prompt": (
                f"Using the GraphRAG pipeline trace for {filename}, inspect the legacy Office OLE stream "
                "evidence. Identify which streams produced readable text, what entities or relationships "
                "were recovered, which quality warnings apply, and cite stream path/checksum provenance."
            ),
            "focus": "legacy_office",
        })
    if "forms" in capabilities:
        suggestions.append({
            "id": "evidence_inspect_form_fields",
            "kind": "evidence",
            "label": "Inspect form fields",
            "prompt": (
                f"Using the GraphRAG pipeline trace for {filename}, inspect the extracted forms and fields. "
                "Explain what each field represents, how it relates to nearby document entities, and cite "
                "field/page provenance."
            ),
            "focus": "forms",
        })
    if {"comments", "annotations", "revisions", "speaker_notes", "notes"} & capabilities:
        suggestions.append({
            "id": "evidence_review_markup",
            "kind": "evidence",
            "label": "Review markup",
            "prompt": (
                f"Using the GraphRAG pipeline trace for {filename}, inspect comments, annotations, revisions, "
                "footnotes/endnotes, and speaker notes. Explain who authored the markup, what decision or "
                "risk it captures, how it connects to nearby document entities, and cite page/slide/comment/"
                "revision provenance."
            ),
            "focus": "markup",
        })
    if {"images", "diagrams"} & capabilities:
        suggestions.append({
            "id": "evidence_explain_visuals",
            "kind": "evidence",
            "label": "Explain visuals",
            "prompt": (
                f"Using the GraphRAG pipeline trace for {filename}, explain the extracted images, figures, "
                "and diagrams. Identify related entities, inferred relationships, and visual provenance from "
                "the trace."
            ),
            "focus": "visuals",
        })
    return suggestions


def _evidence_interaction_suggestion(filename: str, capabilities: set[str]) -> dict[str, Any] | None:
    suggestions = _evidence_interaction_suggestions(filename, capabilities)
    return suggestions[0] if suggestions else None


def _build_interaction_suggestions(
    *,
    filename: str,
    domain: str,
    document_profile: dict[str, Any],
    entities: list[dict[str, Any]],
    relationships: list[GraphCandidateRelationship],
    local_graph: dict[str, Any],
    local_graph_reasoning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    capabilities = set(document_profile.get("detected_capabilities") or [])
    suggestions: list[dict[str, Any]] = []
    reasoning = local_graph_reasoning if isinstance(local_graph_reasoning, dict) else {}
    capability_text = (
        " It extracted "
        + ", ".join(_format_capability_label(capability) for capability in list(document_profile.get("detected_capabilities") or [])[:8])
        + "."
        if document_profile.get("detected_capabilities")
        else ""
    )
    _push_interaction_suggestion(suggestions, {
        "id": "summary",
        "kind": "summary",
        "label": "Summarize trace",
        "prompt": (
            f"Use the recent GraphRAG pipeline trace for {filename} to summarize the key evidence, "
            "entities, relationships, and any meaningful graph paths. Cite slide/page/table/chart/connector "
            f"provenance when available.{capability_text}"
        ),
    })

    ranked_relationships = sorted(relationships, key=_relationship_priority, reverse=True)
    if ranked_relationships:
        _push_interaction_suggestion(suggestions, _relationship_suggestion(filename, ranked_relationships[0]))

    styled_relationship = next(
        (relationship for relationship in ranked_relationships if _is_styled_text_relationship(relationship)),
        None,
    )
    evidence_suggestions = _evidence_interaction_suggestions(filename, capabilities)
    if styled_relationship is not None:
        evidence_suggestions = [
            _styled_text_relationship_suggestion(suggestion, styled_relationship)
            if suggestion.get("focus") == "styled_text"
            else suggestion
            for suggestion in evidence_suggestions
        ]
        evidence_suggestions.sort(key=lambda suggestion: 0 if suggestion.get("focus") == "styled_text" else 1)

    for evidence_suggestion in evidence_suggestions:
        _push_interaction_suggestion(suggestions, evidence_suggestion)

    path_candidates = [path for path in reasoning.get("path_candidates", []) if isinstance(path, dict)]
    if path_candidates:
        _push_interaction_suggestion(suggestions, _path_candidate_suggestion(filename, path_candidates[0]))

    for relationship in ranked_relationships[1:4]:
        _push_interaction_suggestion(suggestions, _relationship_suggestion(filename, relationship))

    unique_entities = []
    seen_entities: set[str] = set()
    for entity in entities:
        name = _clean_text(str(entity.get("name") or ""))
        if not name or name.lower() in seen_entities:
            continue
        seen_entities.add(name.lower())
        unique_entities.append(name)
    if len(unique_entities) >= 2:
        source, target = unique_entities[:2]
        _push_interaction_suggestion(suggestions, {
            "id": f"path_{_stable_question_id(source, target)}",
            "kind": "path",
            "label": _clip(f"Path: {source} to {target}", 64),
            "prompt": (
                f'Using the GraphRAG pipeline trace for {filename}, find the strongest graph path connecting '
                f'"{source}" and "{target}". Explain each hop, identify bridge entities, and cite the exact '
                "extracted evidence and provenance for the path."
            ),
            "source": source,
            "target": target,
        })

    return {
        "source": "pipeline_run",
        "filename": filename,
        "domain": domain,
        "question_count": len(suggestions),
        "graph_edge_count": local_graph.get("edge_count", 0),
        "graph_node_count": local_graph.get("node_count", 0),
        "bridge_entity_count": reasoning.get("bridge_entity_count", 0),
        "path_candidate_count": reasoning.get("path_candidate_count", 0),
        "detected_capabilities": list(document_profile.get("detected_capabilities") or []),
        "suggested_questions": suggestions,
    }


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _interaction_mode(
    *,
    local_graph: dict[str, Any],
    local_context: dict[str, Any],
    local_graph_reasoning: dict[str, Any],
) -> str:
    if _safe_int(local_graph_reasoning.get("path_candidate_count")) > 0:
        return "graph_path_reasoning"
    if _safe_int(local_context.get("graph_path_context_count")) > 0:
        return "chunk_path_retrieval"
    if _safe_int(local_context.get("graph_context_relationship_count")) > 0:
        return "graph_augmented_retrieval"
    if _safe_int(local_graph.get("edge_count")) > 0:
        return "relationship_exploration"
    return "source_trace_summary"


def _interaction_lane(
    lane_id: str,
    label: str,
    *,
    ready: bool,
    count: int = 0,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "id": lane_id,
        "label": label,
        "ready": ready,
        "status": "ready" if ready else "waiting",
        "count": count,
        "detail": detail,
    }


def _interaction_capability_cards(
    *,
    document_profile: dict[str, Any],
    local_graph: dict[str, Any],
    local_context: dict[str, Any],
    local_graph_reasoning: dict[str, Any],
) -> list[dict[str, Any]]:
    evidence_profile = document_profile.get("evidence_profile") if isinstance(document_profile.get("evidence_profile"), dict) else {}
    quick_counts = evidence_profile.get("quick_counts") if isinstance(evidence_profile.get("quick_counts"), dict) else {}
    graph_evidence = evidence_profile.get("graph_evidence") if isinstance(evidence_profile.get("graph_evidence"), dict) else {}
    cards: list[dict[str, Any]] = []

    def add(card_id: str, label: str, count: Any, detail: str, *, kind: str = "evidence") -> None:
        numeric_count = _safe_int(count)
        if numeric_count <= 0:
            return
        cards.append({
            "id": card_id,
            "kind": kind,
            "label": label,
            "count": numeric_count,
            "detail": detail,
            "status": "ready",
        })

    add(
        "path_reasoning",
        "Path reasoning",
        local_graph_reasoning.get("path_candidate_count"),
        "Two-hop bridge paths are ready for graph-style questions.",
        kind="graph",
    )
    add(
        "bridge_entities",
        "Bridge entities",
        local_graph_reasoning.get("bridge_entity_count"),
        "Intermediate graph nodes can connect otherwise separate evidence.",
        kind="graph",
    )
    add(
        "chunk_graph_context",
        "Chunk graph context",
        local_context.get("graph_context_relationship_count"),
        "Embedding chunks carry relationship context into chat retrieval.",
        kind="retrieval",
    )
    add(
        "layout",
        "Layout evidence",
        quick_counts.get("layout_elements") or graph_evidence.get("layout_relationship_count"),
        "Page and slide regions include readable bboxes and zones for location-aware answers.",
    )
    add(
        "graph_edges",
        "Local graph edges",
        local_graph.get("edge_count"),
        "Graph relationships are available before remote ingest completes.",
        kind="graph",
    )
    add(
        "charts",
        "Chart evidence",
        quick_counts.get("charts") or graph_evidence.get("chart_relationship_count"),
        "Chart series, categories, and values are available as graph evidence.",
    )
    add(
        "tables",
        "Table evidence",
        quick_counts.get("tables") or graph_evidence.get("table_fact_count"),
        "Structured table rows are available for citation-aware answers.",
    )
    add(
        "messages",
        "Message evidence",
        quick_counts.get("message_body_parts"),
        "Email headers, body text, links, and attachment context are available for graph-aware answers.",
    )
    add(
        "forms",
        "Form evidence",
        quick_counts.get("forms"),
        "Form controls, field labels, and selected values are available as graph evidence.",
    )
    add(
        "connectors",
        "Connector evidence",
        quick_counts.get("connectors"),
        "PowerPoint arrows and connector labels are available as relationships.",
    )
    add(
        "diagrams",
        "Diagram evidence",
        quick_counts.get("diagrams") or graph_evidence.get("diagram_relationship_count"),
        "SmartArt and diagram nodes plus parent/child edges are available as graph evidence.",
    )
    add(
        "presentation_structure",
        "Presentation structure",
        quick_counts.get("slides"),
        (
            "PowerPoint slide order, visibility, hidden-slide evidence, and package source parts "
            "are available for citation."
        ),
    )
    add(
        "speaker_notes",
        "Speaker notes",
        quick_counts.get("speaker_notes") or graph_evidence.get("speaker_notes_relationship_count"),
        "Presenter notes are available with slide and package source-part provenance.",
    )
    add(
        "source_artifacts",
        "Source artifacts",
        _safe_int(quick_counts.get("archive_entries")) + _safe_int(quick_counts.get("attachments")),
        "Attachments, archive entries, and child documents are traceable.",
    )
    add(
        "legacy_office",
        "Legacy Office",
        quick_counts.get("legacy_office_text_blocks") or quick_counts.get("legacy_office_streams"),
        "Best-effort OLE streams and recovered text are traceable with stream provenance.",
    )
    add(
        "styled_text",
        "Styled text",
        quick_counts.get("styled_text_blocks") or graph_evidence.get("styled_text_relationship_count"),
        "Emphasis and formatted text can guide graph-grounded answers.",
    )
    add(
        "links",
        "Link evidence",
        graph_evidence.get("link_relationship_count"),
        "Hyperlinks and object action links are available with URI, layout, and clicked-object provenance.",
    )
    add(
        "ocr",
        "OCR evidence",
        quick_counts.get("ocr_blocks") or graph_evidence.get("ocr_relationship_count"),
        "Scanned or image-derived text is available with source provenance.",
    )
    return cards[:MAX_INTERACTION_HANDOFF_CARDS]


def _interaction_graph_anchors(
    *,
    local_graph: dict[str, Any],
    local_graph_reasoning: dict[str, Any],
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    seen: set[str] = set()

    def push(anchor: dict[str, Any]) -> None:
        if len(anchors) >= MAX_INTERACTION_HANDOFF_ANCHORS:
            return
        key = str(anchor.get("id") or anchor.get("title") or "").lower()
        if not key or key in seen:
            return
        seen.add(key)
        anchors.append(anchor)

    path_candidates = [
        path for path in local_graph_reasoning.get("path_candidates", [])
        if isinstance(path, dict)
    ]
    for path in path_candidates[:3]:
        nodes = [str(node) for node in path.get("nodes", []) if node]
        source = _clean_text(str(path.get("source") or (nodes[0] if nodes else "")))
        bridge = _clean_text(str(path.get("bridge") or (nodes[1] if len(nodes) > 1 else "")))
        target = _clean_text(str(path.get("target") or (nodes[2] if len(nodes) > 2 else "")))
        if not source or not bridge or not target:
            continue
        push({
            "id": str(path.get("id") or f"path_{_stable_question_id(source, bridge, target)}"),
            "kind": "path",
            "title": _clip(f"{source} -> {bridge} -> {target}", 96),
            "label": _clip(f"{source} via {bridge}", 72),
            "score": path.get("score"),
            "nodes": nodes or [source, bridge, target],
            "relationships": path.get("relationships") if isinstance(path.get("relationships"), list) else [],
            "evidence": _clip(str(path.get("explanation") or path.get("text") or ""), 360),
            "evidence_families": path.get("evidence_families") if isinstance(path.get("evidence_families"), list) else [],
            "provenance_summary": path.get("provenance_summary") if isinstance(path.get("provenance_summary"), list) else [],
            "answer_hint": path.get("answer_hint"),
        })

    edges = [
        edge for edge in local_graph.get("edges", [])
        if isinstance(edge, dict)
    ]
    link_edges = sorted(
        [edge for edge in edges if _is_link_graph_edge(edge)],
        key=lambda edge: (_local_graph_edge_score(edge), _local_graph_edge_weight(edge)),
        reverse=True,
    )
    for edge in link_edges[:3]:
        summary = _local_graph_edge_summary(edge)
        source = summary.get("source")
        target = summary.get("target")
        relation = summary.get("relation")
        metadata = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
        uri = _clean_text(str(metadata.get("uri") or target or ""))
        if not source or not target:
            continue
        push({
            "id": str(summary.get("id") or f"link_{_stable_question_id(str(source), str(relation), str(target), uri)}"),
            "kind": "link",
            "title": _clip(f"{source} --[{relation}]--> {target}", 120),
            "label": _clip(f"Link: {source} to {target}", 72),
            "score": summary.get("score"),
            "source": source,
            "target": target,
            "relation": relation,
            "uri": uri,
            "evidence": summary.get("evidence"),
            "metadata": metadata,
        })

    styled_edges = sorted(
        [edge for edge in edges if _is_styled_graph_edge(edge)],
        key=lambda edge: (_local_graph_edge_score(edge), _local_graph_edge_weight(edge)),
        reverse=True,
    )
    for edge in styled_edges[:3]:
        summary = _local_graph_edge_summary(edge)
        source = summary.get("source")
        target = summary.get("target")
        relation = summary.get("relation")
        metadata = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
        emphasis_term = _clean_text(str(metadata.get("emphasis_term") or target or ""))
        style_summary = _clean_text(str(metadata.get("style_summary") or ""))
        label_parts = [emphasis_term]
        if style_summary:
            label_parts.append(style_summary)
        push({
            "id": str(summary.get("id") or f"edge_{_stable_question_id(str(source), str(relation), str(target))}"),
            "kind": "styled_text",
            "title": _clip(f"{source} --[{relation}]--> {target}", 120),
            "label": _clip("Style: " + " · ".join(part for part in label_parts if part), 72),
            "score": summary.get("score"),
            "source": source,
            "target": target,
            "relation": relation,
            "evidence": summary.get("evidence"),
            "metadata": metadata,
        })

    bridge_entities = [
        bridge for bridge in local_graph_reasoning.get("bridge_entities", [])
        if isinstance(bridge, dict)
    ]
    for bridge in bridge_entities[:3]:
        name = _clean_text(str(bridge.get("name") or bridge.get("label") or ""))
        if not name:
            continue
        neighbours = [
            _clean_text(str(neighbour))
            for neighbour in bridge.get("neighbours", [])
            if _clean_text(str(neighbour))
        ] if isinstance(bridge.get("neighbours"), list) else []
        push({
            "id": str(bridge.get("id") or f"bridge_{_stable_question_id(name)}"),
            "kind": "bridge",
            "title": _clip(name, 96),
            "label": _clip(f"Bridge: {name}", 72),
            "score": bridge.get("score"),
            "relationship_count": bridge.get("relationship_count"),
            "path_count": bridge.get("path_count"),
            "neighbours": neighbours[:6],
            "evidence": _clip(str(bridge.get("text") or ""), 280),
        })

    sorted_edges = sorted(
        edges,
        key=lambda edge: (_local_graph_edge_score(edge), _local_graph_edge_weight(edge)),
        reverse=True,
    )
    for edge in sorted_edges[:MAX_INTERACTION_HANDOFF_ANCHORS]:
        summary = _local_graph_edge_summary(edge)
        source = summary.get("source")
        target = summary.get("target")
        relation = summary.get("relation")
        if not source or not target:
            continue
        push({
            "id": str(summary.get("id") or f"edge_{_stable_question_id(str(source), str(relation), str(target))}"),
            "kind": "relationship",
            "title": _clip(f"{source} --[{relation}]--> {target}", 120),
            "label": _clip(f"{source} to {target}", 72),
            "score": summary.get("score"),
            "source": source,
            "target": target,
            "relation": relation,
            "evidence": summary.get("evidence"),
            "metadata": summary.get("metadata"),
        })
    return anchors


def _interaction_citation_scopes(document_profile: dict[str, Any]) -> list[str]:
    capabilities = set(document_profile.get("detected_capabilities") or [])
    scopes = []
    for capability, label in (
        ("sections", "section"),
        ("layout", "layout"),
        ("messages", "message"),
        ("presentation_structure", "slide visibility"),
        ("speaker_notes", "speaker notes"),
        ("tables", "table"),
        ("charts", "chart"),
        ("connectors", "connector"),
        ("diagrams", "diagram"),
        ("shapes", "shape"),
        ("links", "link"),
        ("images", "image"),
        ("ocr", "OCR"),
        ("embedded_objects", "attachment"),
        ("archives", "archive entry"),
        ("legacy_office", "legacy Office stream"),
        ("forms", "form field"),
        ("comments", "comment"),
        ("annotations", "annotation"),
        ("revisions", "revision"),
        ("styled_text", "style"),
    ):
        if capability in capabilities:
            scopes.append(label)
    return scopes[:12]


def _interaction_rehearsal_tokens(value: Any) -> list[str]:
    text = _clean_text(str(value or "")).lower()
    tokens = [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text)
        if token not in INTERACTION_REHEARSAL_STOPWORDS
    ]
    return _unique_preserve_order(tokens)


def _interaction_rehearsal_blob(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, dict):
        return " ".join(
            _interaction_rehearsal_blob(item)
            for item in value.values()
            if item not in (None, "")
        )
    if isinstance(value, (list, tuple, set)):
        return " ".join(_interaction_rehearsal_blob(item) for item in value)
    return _clean_text(str(value))


def _interaction_rehearsal_score(query_tokens: list[str], text: str) -> tuple[float, list[str]]:
    if not query_tokens or not text:
        return 0.0, []
    lowered = text.lower()
    matched = [
        token
        for token in query_tokens
        if token in lowered
    ]
    if not matched:
        return 0.0, []
    unique_matched = _unique_preserve_order(matched)
    coverage = len(unique_matched) / max(1, min(len(query_tokens), 10))
    phrase_bonus = 0.08 if any(
        " ".join(query_tokens[index:index + 2]) in lowered
        for index in range(0, max(0, len(query_tokens) - 1))
    ) else 0.0
    score = min(1.0, (coverage * 0.86) + phrase_bonus)
    return round(score, 3), unique_matched[:12]


def _interaction_anchor_rehearsal_matches(
    query_tokens: list[str],
    anchors: list[dict[str, Any]],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    matches = []
    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        blob = _interaction_rehearsal_blob({
            "title": anchor.get("title") or anchor.get("label"),
            "kind": anchor.get("kind"),
            "source": anchor.get("source"),
            "relation": anchor.get("relation"),
            "target": anchor.get("target"),
            "uri": anchor.get("uri"),
            "nodes": anchor.get("nodes"),
            "neighbours": anchor.get("neighbours"),
            "evidence": anchor.get("evidence"),
            "evidence_families": anchor.get("evidence_families"),
            "provenance_summary": anchor.get("provenance_summary"),
            "answer_hint": anchor.get("answer_hint"),
            "metadata": anchor.get("metadata"),
        })
        score, matched_terms = _interaction_rehearsal_score(query_tokens, blob)
        if score <= 0:
            continue
        matches.append({
            "id": anchor.get("id"),
            "kind": anchor.get("kind"),
            "title": anchor.get("title") or anchor.get("label"),
            "score": score,
            "matched_terms": matched_terms,
            "source": anchor.get("source"),
            "relation": anchor.get("relation"),
            "target": anchor.get("target"),
            "evidence": _clip(str(anchor.get("evidence") or blob), 360),
            "evidence_families": anchor.get("evidence_families") if isinstance(anchor.get("evidence_families"), list) else [],
            "provenance_summary": anchor.get("provenance_summary") if isinstance(anchor.get("provenance_summary"), list) else [],
            "metadata": anchor.get("metadata") if isinstance(anchor.get("metadata"), dict) else {},
        })
    return sorted(matches, key=lambda item: item.get("score", 0), reverse=True)[:limit]


def _interaction_chunk_rehearsal_matches(
    query_tokens: list[str],
    local_context: dict[str, Any],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    chunks = local_context.get("chunks") if isinstance(local_context.get("chunks"), list) else []
    matches = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        blob = _interaction_rehearsal_blob({
            "content": chunk.get("content"),
            "source_elements": chunk.get("source_elements"),
            "graph_context": chunk.get("graph_context"),
            "graph_paths": chunk.get("graph_paths"),
            "metadata": {
                "chunk_scope_label": metadata.get("chunk_scope_label"),
                "chunk_scope_labels": metadata.get("chunk_scope_labels"),
                "provenance": metadata.get("provenance"),
                "evidence_preview": metadata.get("evidence_preview"),
            },
        })
        score, matched_terms = _interaction_rehearsal_score(query_tokens, blob)
        graph_context_count = len(chunk.get("graph_context", []) or []) if isinstance(chunk.get("graph_context"), list) else 0
        graph_path_count = len(chunk.get("graph_paths", []) or []) if isinstance(chunk.get("graph_paths"), list) else 0
        if score <= 0 and not (graph_context_count or graph_path_count):
            continue
        if score <= 0:
            score = 0.08 if graph_context_count else 0.05
        try:
            chunk_index = int(chunk.get("chunk_index", 0) or 0)
        except (TypeError, ValueError):
            chunk_index = 0
        matches.append({
            "chunk_id": chunk.get("id") or chunk.get("chunk_id"),
            "chunk_index": chunk_index,
            "score": round(score, 3),
            "matched_terms": matched_terms,
            "graph_context_relationship_count": graph_context_count,
            "graph_path_context_count": graph_path_count,
            "source_elements": chunk.get("source_elements", [])[:6] if isinstance(chunk.get("source_elements"), list) else [],
            "evidence_preview": _clip(str(metadata.get("evidence_preview") or chunk.get("content") or ""), 420),
            "provenance": metadata.get("provenance", [])[:6] if isinstance(metadata.get("provenance"), list) else metadata.get("provenance"),
        })
    return sorted(matches, key=lambda item: item.get("score", 0), reverse=True)[:limit]


def _build_interaction_rehearsals(
    *,
    starter_prompts: list[dict[str, Any]],
    graph_anchors: list[dict[str, Any]],
    local_context: dict[str, Any],
    limit: int = 5,
) -> list[dict[str, Any]]:
    rehearsals = []
    for index, prompt in enumerate(starter_prompts[:limit], start=1):
        if not isinstance(prompt, dict):
            continue
        prompt_text = _clean_text(str(prompt.get("prompt") or prompt.get("label") or ""))
        if not prompt_text:
            continue
        query_tokens = _interaction_rehearsal_tokens(prompt_text)
        anchor_matches = _interaction_anchor_rehearsal_matches(query_tokens, graph_anchors)
        chunk_matches = _interaction_chunk_rehearsal_matches(query_tokens, local_context)
        matched_terms = _unique_preserve_order([
            term
            for match in [*anchor_matches, *chunk_matches]
            for term in match.get("matched_terms", [])
            if term
        ])
        score = max(
            [match.get("score", 0) for match in [*anchor_matches, *chunk_matches]],
            default=0,
        )
        rehearsals.append({
            "id": f"rehearsal_{_stable_question_id(str(prompt.get('id') or prompt.get('label') or index))}",
            "prompt_id": prompt.get("id"),
            "kind": prompt.get("kind"),
            "label": prompt.get("label") or f"Prompt {index}",
            "prompt": _clip(prompt_text, 520),
            "status": "ready" if anchor_matches or chunk_matches else "no_match",
            "score": round(float(score or 0), 3),
            "matched_terms": matched_terms[:12],
            "matched_anchor_count": len(anchor_matches),
            "matched_chunk_count": len(chunk_matches),
            "top_anchors": anchor_matches,
            "top_chunks": chunk_matches,
        })
    return rehearsals


def _build_interaction_handoff(
    *,
    run_id: str,
    filename: str,
    domain: str,
    document_profile: dict[str, Any],
    interaction_suggestions: dict[str, Any],
    local_graph: dict[str, Any],
    local_context: dict[str, Any],
    local_graph_reasoning: dict[str, Any],
) -> dict[str, Any]:
    graph_edge_count = _safe_int(local_graph.get("edge_count"))
    cached_chunk_count = _safe_int(local_context.get("cached_chunk_count"))
    graph_context_count = _safe_int(local_context.get("graph_context_relationship_count"))
    graph_path_context_count = _safe_int(local_context.get("graph_path_context_count"))
    bridge_count = _safe_int(local_graph_reasoning.get("bridge_entity_count"))
    path_count = _safe_int(local_graph_reasoning.get("path_candidate_count"))
    question_count = _safe_int(interaction_suggestions.get("question_count"))
    recommended_mode = _interaction_mode(
        local_graph=local_graph,
        local_context=local_context,
        local_graph_reasoning=local_graph_reasoning,
    )
    starter_prompts = [
        question for question in interaction_suggestions.get("suggested_questions", [])
        if isinstance(question, dict)
    ][:5]
    lanes = [
        _interaction_lane(
            "trace",
            "Trace",
            ready=True,
            count=_safe_int(document_profile.get("embedding_input_count")),
            detail="Extraction, chunking, graph candidates, and embedding samples are available.",
        ),
        _interaction_lane(
            "graph",
            "Graph",
            ready=graph_edge_count > 0,
            count=graph_edge_count,
            detail="Local graph edges are ready for relationship questions.",
        ),
        _interaction_lane(
            "paths",
            "Paths",
            ready=path_count > 0 or graph_path_context_count > 0,
            count=max(path_count, graph_path_context_count),
            detail="Bridge/path context is ready for multi-hop questions.",
        ),
        _interaction_lane(
            "retrieval",
            "Retrieval",
            ready=cached_chunk_count > 0,
            count=cached_chunk_count,
            detail="Run-scoped chunks can be used by chat before remote ingest completes.",
        ),
        _interaction_lane(
            "chat",
            "Chat",
            ready=question_count > 0,
            count=question_count,
            detail="Citation-aware starter prompts are available.",
        ),
    ]
    graph_anchors = _interaction_graph_anchors(
        local_graph=local_graph,
        local_graph_reasoning=local_graph_reasoning,
    )
    capability_cards = _interaction_capability_cards(
        document_profile=document_profile,
        local_graph=local_graph,
        local_context=local_context,
        local_graph_reasoning=local_graph_reasoning,
    )
    citation_scopes = _interaction_citation_scopes(document_profile)
    answer_guidance = (
        "Use graph anchors first, then support each hop with run-scoped chunks and document provenance. "
        "Prefer layout, slide, page, table, chart, and connector citations when the trace provides them."
    )
    if citation_scopes:
        answer_guidance += f" Available citation scopes include {', '.join(citation_scopes)}."
    retrieval_rehearsals = _build_interaction_rehearsals(
        starter_prompts=starter_prompts,
        graph_anchors=graph_anchors,
        local_context=local_context,
    )
    rehearsed_chunk_matches = sum(item.get("matched_chunk_count", 0) for item in retrieval_rehearsals)
    rehearsed_anchor_matches = sum(item.get("matched_anchor_count", 0) for item in retrieval_rehearsals)

    return {
        "source": "pipeline_run",
        "run_id": run_id,
        "filename": filename,
        "domain": domain,
        "ready": any(lane["ready"] for lane in lanes),
        "recommended_mode": recommended_mode,
        "chat_context_endpoint": "/api/v1/graphrag/chat-context-v2",
        "chat_context_request": {
            "query": starter_prompts[0].get("prompt") if starter_prompts else "",
            "domain": domain,
            "pipeline_run_ids": [run_id],
            "include_graph": True,
            "use_local_pipeline_context": True,
        },
        "lanes": lanes,
        "capability_cards": capability_cards,
        "graph_anchors": graph_anchors,
        "starter_prompts": starter_prompts,
        "retrieval_rehearsals": retrieval_rehearsals,
        "citation_scopes": citation_scopes,
        "answer_guidance": answer_guidance,
        "counts": {
            "graph_edges": graph_edge_count,
            "cached_chunks": cached_chunk_count,
            "graph_context_relationships": graph_context_count,
            "graph_path_context_paths": graph_path_context_count,
            "bridge_entities": bridge_count,
            "path_candidates": path_count,
            "starter_prompts": len(starter_prompts),
            "graph_anchors": len(graph_anchors),
            "capability_cards": len(capability_cards),
            "retrieval_rehearsals": len(retrieval_rehearsals),
            "rehearsed_chunk_matches": rehearsed_chunk_matches,
            "rehearsed_anchor_matches": rehearsed_anchor_matches,
        },
    }


def _build_local_graph_artifact(
    *,
    filename: str,
    domain: str,
    entities: list[dict[str, Any]],
    relationships: list[GraphCandidateRelationship],
) -> dict[str, Any]:
    nodes_by_key: dict[str, dict[str, Any]] = {}

    def add_node(name: str, node_type: str = "concept", source: str = "relationship") -> None:
        cleaned = _clean_text(str(name or ""))
        if not cleaned:
            return
        key = cleaned.lower()
        existing = nodes_by_key.get(key)
        if existing:
            if existing.get("type") == "concept" and node_type != "concept":
                existing["type"] = node_type
            return
        nodes_by_key[key] = {
            "id": cleaned,
            "name": cleaned,
            "label": cleaned,
            "type": node_type or "concept",
            "source": source,
        }

    for entity in entities:
        add_node(
            str(entity.get("name") or ""),
            str(entity.get("type") or "concept").lower(),
            "entity_candidate",
        )

    edges = []
    for index, relationship in enumerate(relationships[:MAX_LOCAL_GRAPH_RELATIONSHIPS], start=1):
        rel = asdict(relationship)
        source = _clean_text(str(rel.get("source") or ""))
        target = _clean_text(str(rel.get("target") or ""))
        relation = _clean_text(str(rel.get("relation") or "related_to")) or "related_to"
        if not source or not target:
            continue
        add_node(source)
        add_node(target)
        confidence = rel.get("confidence", 1.0)
        try:
            weight = float(confidence)
        except (TypeError, ValueError):
            weight = 1.0
        metadata = rel.get("metadata") if isinstance(rel.get("metadata"), dict) else {}
        properties = {
            **metadata,
            "type": relation,
            "relationship_type": relation,
            "weight": max(0.0, min(weight, 1.0)),
            "confidence": confidence,
            "evidence": rel.get("evidence"),
            "source_file": filename,
            "domain": domain,
        }
        edges.append({
            "id": f"local_graph_edge_{index}",
            "source": source,
            "target": target,
            "label": relation,
            "type": relation,
            "weight": properties["weight"],
            "evidence": rel.get("evidence"),
            "metadata": metadata,
            "properties": properties,
        })

    return {
        "source": "pipeline_run",
        "filename": filename,
        "domain": domain,
        "node_count": len(nodes_by_key),
        "relationship_count": len(relationships),
        "edge_count": len(edges),
        "truncated": len(relationships) > MAX_LOCAL_GRAPH_RELATIONSHIPS,
        "nodes": list(nodes_by_key.values()),
        "edges": edges,
    }


def _local_graph_edge_relation(edge: dict[str, Any]) -> str:
    return _clean_text(str(edge.get("label") or edge.get("relation") or edge.get("type") or "related_to")) or "related_to"


def _local_graph_edge_metadata(edge: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for source in (edge.get("metadata"), edge.get("properties")):
        if not isinstance(source, dict):
            continue
        nested = source.get("metadata")
        if isinstance(nested, dict):
            metadata.update(nested)
        metadata.update({key: value for key, value in source.items() if key != "metadata"})
    return metadata


def _local_graph_edge_weight(edge: dict[str, Any]) -> float:
    properties = edge.get("properties") if isinstance(edge.get("properties"), dict) else {}
    try:
        return max(0.0, min(float(edge.get("weight", properties.get("weight", properties.get("confidence", 1.0)))), 1.0))
    except (TypeError, ValueError):
        return 1.0


def _local_graph_edge_score(edge: dict[str, Any]) -> float:
    relation = _local_graph_edge_relation(edge)
    metadata = _local_graph_edge_metadata(edge)
    properties = edge.get("properties") if isinstance(edge.get("properties"), dict) else {}
    searchable = " ".join(
        _clean_text(str(value))
        for value in (
            edge.get("source"),
            edge.get("target"),
            relation,
            edge.get("evidence"),
            *metadata.values(),
            *properties.values(),
        )
        if value not in (None, "")
    ).lower()
    weight = _local_graph_edge_weight(edge)

    if metadata.get("semantic_relation_extraction") in {"verb_pattern", "table_row_triple"}:
        base = 0.94
    elif re.search(r"connector|arrow|line|flow|shape_connector", searchable):
        base = 0.92
    elif relation == "visually_near" or metadata.get("layout_relation"):
        base = 0.82
    elif metadata.get("styled_text") or relation in {"emphasizes", "uses_bullet_structure"}:
        base = 0.84
    elif metadata.get("uri") or metadata.get("link_kind") or relation in {"links_to", "contains_link"}:
        base = 0.82
    elif metadata.get("markdown_code_language") or relation in {"contains_code_block", "uses_language"}:
        base = 0.8
    elif metadata.get("group_id") or metadata.get("parent_group_id") or re.search(r"chart|table|form|field|image|figure|diagram|group|shape|speaker|notes|embedded", searchable):
        base = 0.78
    elif not relation.lower().startswith(("has_", "contains_")):
        base = 0.68
    elif edge.get("evidence"):
        base = 0.48
    else:
        base = 0.32
    return round(min(1.0, (base * 0.72) + (weight * 0.28)), 3)


def _local_graph_edge_summary(edge: dict[str, Any]) -> dict[str, Any]:
    relation = _local_graph_edge_relation(edge)
    metadata = _local_graph_edge_metadata(edge)
    properties = edge.get("properties") if isinstance(edge.get("properties"), dict) else {}
    weight = _local_graph_edge_weight(edge)
    evidence_families = _local_graph_edge_evidence_families(edge)
    provenance_summary = _local_graph_edge_provenance_summary(metadata)
    return {
        "id": edge.get("id"),
        "source": _clean_text(str(edge.get("source") or "")),
        "target": _clean_text(str(edge.get("target") or "")),
        "relation": relation,
        "label": relation,
        "weight": round(weight, 3),
        "score": _local_graph_edge_score(edge),
        "evidence": _clip(str(edge.get("evidence") or properties.get("evidence") or ""), 280),
        "evidence_families": evidence_families,
        "provenance_summary": provenance_summary,
        "rationale": _local_graph_edge_rationale(relation, evidence_families, provenance_summary),
        "metadata": _relationship_location_hints(metadata),
    }


def _local_graph_edge_evidence_families(edge: dict[str, Any]) -> list[str]:
    relation = _local_graph_edge_relation(edge).lower()
    metadata = _local_graph_edge_metadata(edge)
    properties = edge.get("properties") if isinstance(edge.get("properties"), dict) else {}
    searchable = " ".join(
        _clean_text(str(value))
        for value in (
            edge.get("source"),
            edge.get("target"),
            relation,
            edge.get("evidence"),
            *metadata.values(),
            *properties.values(),
        )
        if value not in (None, "")
    ).lower()
    families: list[str] = []

    def add(family: str, condition: bool) -> None:
        if condition and family not in families:
            families.append(family)

    add("connector", "connector" in searchable or "shape_connector" in searchable or "arrow" in searchable)
    add("diagram", bool(metadata.get("diagram_part") or metadata.get("diagram_node_id") or "diagram" in searchable or "smartart" in searchable))
    add("shape", bool(metadata.get("group_id") or metadata.get("parent_group_id") or "shape_group" in searchable))
    add("chart", bool(metadata.get("chart_part") or metadata.get("chart_source_part") or "chart" in searchable))
    add(
        "table",
        bool(
            metadata.get("table_name")
            or metadata.get("table")
            or metadata.get("table_source")
            or metadata.get("table_fact_kind")
            or re.search(r"\btable\b", searchable)
        ),
    )
    add("code_block", bool(metadata.get("markdown_code_language") or metadata.get("element_type") == "code_block" or relation in {"contains_code_block", "uses_language"}))
    add("styled_text", bool(metadata.get("styled_text") or metadata.get("style_summary") or metadata.get("emphasis_term") or relation in {"emphasizes", "uses_bullet_structure"}))
    add("link", bool(metadata.get("uri") or metadata.get("link_kind") or relation in {"links_to", "contains_link"}))
    add("image", bool(metadata.get("image_part") or metadata.get("image_relationship_id") or "image" in searchable or "picture" in searchable))
    add("form_field", bool(metadata.get("form_kind") or metadata.get("field_type") or metadata.get("control_part") or "form_field" in searchable))
    add("speaker_notes", bool(metadata.get("notes_part") or "speaker_notes" in searchable or "speaker note" in searchable))
    add("comment", bool(metadata.get("comment_id") or metadata.get("comment_anchor_text") or "comment" in searchable))
    add("revision", bool(metadata.get("revision_id") or metadata.get("revision_type") or "revision" in searchable))
    add("ocr", bool(metadata.get("ocr_source") or metadata.get("ocr_engine") or "ocr" in searchable))
    add("media", bool(metadata.get("media_part") or metadata.get("media_kind") or "media" in searchable))
    add("attachment", bool(metadata.get("embedded_file_name") or metadata.get("embedded_file_filename") or "attachment" in searchable or "embedded" in searchable))
    add("archive_entry", bool(metadata.get("archive_entry") or metadata.get("archive_entry_path") or "archive_entry" in searchable))
    add("legacy_office", bool(metadata.get("legacy_office_stream_path") or "legacy_office_" in searchable))
    add("formula", bool(metadata.get("formula_cell") or metadata.get("formula_expression") or "formula" in searchable))
    add("pivot_table", bool(metadata.get("pivot_table_name") or metadata.get("pivot_table_range") or "pivot_table" in searchable or "pivot table" in searchable))
    add("conditional_format", bool(metadata.get("conditional_range") or metadata.get("conditional_type") or "conditional_format" in searchable))
    add("presentation_structure", bool(metadata.get("slide_part") or metadata.get("slide_visibility") or metadata.get("slide_state")))
    add("workbook_structure", bool(metadata.get("sheet_part") or metadata.get("sheet_visibility") or metadata.get("sheet_state")))
    add(
        "layout",
        bool(
            metadata.get("layout_relation")
            or relation == "visually_near"
            or metadata.get("layout_bbox")
            or metadata.get("layout_zone")
            or relation.startswith(("contains_", "has_"))
        ),
    )
    add("semantic", not relation.startswith(("contains_", "has_")) or not families)
    return families or ["semantic"]


def _local_graph_edge_provenance_summary(metadata: dict[str, Any]) -> str:
    hints: list[str] = []

    def add(label: str, value: Any) -> None:
        text = _clean_text(str(value or ""))
        if text and f"{label}={text}" not in hints:
            hints.append(f"{label}={text}")

    add("slide", metadata.get("slide_title") or metadata.get("slide"))
    add("page", metadata.get("page"))
    add("sheet", metadata.get("sheet_name") or metadata.get("sheet"))
    add("table", metadata.get("table_name") or metadata.get("table"))
    add("code", metadata.get("markdown_code_language"))
    add("cell", metadata.get("cell") or metadata.get("formula_cell"))
    add("chart", metadata.get("chart_title") or metadata.get("chart_part"))
    add("diagram", metadata.get("diagram_part"))
    add("group", metadata.get("group_label") or metadata.get("parent_group_label"))
    add("grouped_visual", metadata.get("grouped_visual_label"))
    add("connector", metadata.get("connector_title") or metadata.get("connector_name") or metadata.get("connector_id"))
    source_shape = _clean_text(str(metadata.get("source_shape_id") or ""))
    target_shape = _clean_text(str(metadata.get("target_shape_id") or ""))
    if source_shape or target_shape:
        add("shapes", f"{source_shape or '?'} -> {target_shape or '?'}")
    add("image", metadata.get("image_part"))
    add("legacy_stream", metadata.get("legacy_office_stream_path"))
    add("source", metadata.get("chart_source_part") or metadata.get("embedded_file_name") or metadata.get("embedded_file_filename"))
    add("uri", metadata.get("uri"))
    add("layout", metadata.get("layout_bbox") or metadata.get("layout_zone"))
    add("visual", metadata.get("layout_relation"))
    add("near", metadata.get("related_label"))
    add("distance", metadata.get("visual_distance_pct"))
    add("visibility", metadata.get("slide_visibility") or metadata.get("sheet_visibility") or metadata.get("slide_state") or metadata.get("sheet_state"))
    return "; ".join(hints[:6])


def _local_graph_edge_rationale(relation: str, evidence_families: list[str], provenance_summary: str) -> str:
    family_text = ", ".join(format_key.replace("_", " ") for format_key in evidence_families[:4])
    rationale = f"{relation.replace('_', ' ')} evidence"
    if family_text:
        rationale += f" from {family_text}"
    if provenance_summary:
        rationale += f" at {provenance_summary}"
    return _clip(rationale, 240)


def _is_styled_graph_edge(edge: dict[str, Any]) -> bool:
    metadata = _local_graph_edge_metadata(edge)
    relation = _local_graph_edge_relation(edge)
    return bool(
        metadata.get("styled_text")
        or metadata.get("style_summary")
        or metadata.get("emphasis_term")
        or relation in {"emphasizes", "uses_bullet_structure"}
    )


def _is_link_graph_edge(edge: dict[str, Any]) -> bool:
    metadata = _local_graph_edge_metadata(edge)
    relation = _local_graph_edge_relation(edge)
    return bool(
        metadata.get("uri")
        or metadata.get("link_kind")
        or relation in {"links_to", "contains_link"}
    )


def _other_edge_endpoint(edge: dict[str, Any], bridge_key: str) -> str:
    source = _clean_text(str(edge.get("source") or ""))
    target = _clean_text(str(edge.get("target") or ""))
    if source.lower() == bridge_key:
        return target
    if target.lower() == bridge_key:
        return source
    return ""


def _path_candidate_explanation(
    *,
    source: str,
    bridge: str,
    target: str,
    relationships: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_families = _unique_preserve_order([
        str(family)
        for relationship in relationships
        for family in (relationship.get("evidence_families") or [])
        if family
    ])
    provenance = _unique_preserve_order([
        str(relationship.get("provenance_summary") or "")
        for relationship in relationships
        if relationship.get("provenance_summary")
    ])
    relation_text = "; ".join(
        _clip(
            f"{relationship.get('source')} --[{relationship.get('relation')}]-- {relationship.get('target')}",
            160,
        )
        for relationship in relationships
        if relationship.get("source") and relationship.get("target")
    )
    family_text = ", ".join(family.replace("_", " ") for family in evidence_families[:5]) or "graph"
    explanation = (
        f"{bridge} connects {source} and {target} through {family_text} evidence"
        + (f": {relation_text}" if relation_text else ".")
    )
    if provenance:
        explanation += f" Cite {provenance[0]}."
    answer_hint = (
        f"Use the path through {bridge} to explain how {source} and {target} are related, "
        f"then cite the supporting {family_text} evidence."
    )
    return {
        "evidence_families": evidence_families[:8],
        "provenance_summary": provenance[:6],
        "explanation": _clip(explanation, 520),
        "answer_hint": _clip(answer_hint, 360),
    }


def _build_local_graph_reasoning_index(local_graph: dict[str, Any]) -> dict[str, Any]:
    nodes = local_graph.get("nodes") if isinstance(local_graph.get("nodes"), list) else []
    edges = local_graph.get("edges") if isinstance(local_graph.get("edges"), list) else []
    node_lookup: dict[str, dict[str, Any]] = {}
    node_names: dict[str, str] = {}
    incident_edges: dict[str, list[dict[str, Any]]] = {}

    for node in nodes:
        if not isinstance(node, dict):
            continue
        name = _clean_text(str(node.get("name") or node.get("label") or node.get("id") or ""))
        if not name:
            continue
        key = name.lower()
        node_lookup[key] = node
        node_names[key] = name

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = _clean_text(str(edge.get("source") or ""))
        target = _clean_text(str(edge.get("target") or ""))
        if not source or not target or source.lower() == target.lower():
            continue
        for name in (source, target):
            key = name.lower()
            node_names.setdefault(key, name)
            incident_edges.setdefault(key, []).append(edge)

    bridge_entities: list[dict[str, Any]] = []
    for bridge_key, bridge_edges in incident_edges.items():
        if len(bridge_edges) < 2:
            continue
        bridge_name = node_names.get(bridge_key, bridge_key)
        node = node_lookup.get(bridge_key, {})
        sorted_edges = sorted(
            bridge_edges,
            key=lambda edge: (_local_graph_edge_score(edge), _local_graph_edge_weight(edge)),
            reverse=True,
        )
        neighbours: list[str] = []
        for edge in sorted_edges:
            other = _other_edge_endpoint(edge, bridge_key)
            if other and other.lower() != bridge_key and other not in neighbours:
                neighbours.append(other)
        relationship_summaries = [_local_graph_edge_summary(edge) for edge in sorted_edges[:MAX_LOCAL_GRAPH_REASONING_RELATIONSHIPS]]
        average_edge_score = sum(item["score"] for item in relationship_summaries) / max(1, len(relationship_summaries))
        score = round(
            min(
                1.0,
                (average_edge_score * 0.68)
                + (min(len(bridge_edges), 8) * 0.035)
                + (min(len(neighbours), 8) * 0.018),
            ),
            3,
        )
        bridge_entities.append({
            "id": f"bridge_entity_{_stable_question_id(bridge_name)}",
            "name": bridge_name,
            "label": bridge_name,
            "type": node.get("type") or node.get("kind") or "concept",
            "relationship_count": len(bridge_edges),
            "path_count": 0,
            "neighbours": neighbours[:8],
            "relationships": relationship_summaries,
            "score": score,
            "text": (
                f"{bridge_name} links {len(neighbours)} neighbour"
                f"{'' if len(neighbours) == 1 else 's'}: {', '.join(neighbours[:5])}"
            ),
        })

    bridge_entities = sorted(
        bridge_entities,
        key=lambda bridge: (bridge.get("score", 0), bridge.get("relationship_count", 0)),
        reverse=True,
    )

    path_candidates: list[dict[str, Any]] = []
    seen_paths: set[tuple[str, str, str]] = set()
    for bridge in bridge_entities:
        bridge_name = str(bridge.get("name") or "")
        bridge_key = bridge_name.lower()
        sorted_edges = sorted(
            incident_edges.get(bridge_key, []),
            key=lambda edge: (_local_graph_edge_score(edge), _local_graph_edge_weight(edge)),
            reverse=True,
        )[:MAX_LOCAL_GRAPH_REASONING_RELATIONSHIPS]
        for left_index, left_edge in enumerate(sorted_edges):
            left_other = _other_edge_endpoint(left_edge, bridge_key)
            if not left_other:
                continue
            for right_edge in sorted_edges[left_index + 1:]:
                right_other = _other_edge_endpoint(right_edge, bridge_key)
                if not right_other or right_other.lower() in {left_other.lower(), bridge_key}:
                    continue
                unordered_endpoints = "|".join(sorted([left_other.lower(), right_other.lower()]))
                path_key = (unordered_endpoints, bridge_key, _local_graph_edge_relation(left_edge).lower() + "|" + _local_graph_edge_relation(right_edge).lower())
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)
                left_summary = _local_graph_edge_summary(left_edge)
                right_summary = _local_graph_edge_summary(right_edge)
                explanation = _path_candidate_explanation(
                    source=left_other,
                    bridge=bridge_name,
                    target=right_other,
                    relationships=[left_summary, right_summary],
                )
                relationship_score = (left_summary["score"] + right_summary["score"]) / 2
                path_score = round(min(1.0, (relationship_score * 0.82) + (float(bridge.get("score", 0) or 0) * 0.18)), 3)
                path_candidates.append({
                    "id": f"path_candidate_{_stable_question_id(left_other, bridge_name, right_other)}",
                    "source": left_other,
                    "bridge": bridge_name,
                    "target": right_other,
                    "nodes": [left_other, bridge_name, right_other],
                    "hops": 2,
                    "score": path_score,
                    "relationships": [left_summary, right_summary],
                    **explanation,
                    "text": (
                        f"{left_other} --[{left_summary['relation']}]-- {bridge_name} "
                        f"--[{right_summary['relation']}]-- {right_other}"
                    ),
                })
                if len(path_candidates) >= MAX_LOCAL_GRAPH_REASONING_PATHS * 2:
                    break
            if len(path_candidates) >= MAX_LOCAL_GRAPH_REASONING_PATHS * 2:
                break

    path_candidates = sorted(path_candidates, key=lambda path: path.get("score", 0), reverse=True)
    bridge_path_counts = Counter(str(path.get("bridge") or "").lower() for path in path_candidates)
    for bridge in bridge_entities:
        bridge["path_count"] = bridge_path_counts.get(str(bridge.get("name") or "").lower(), 0)

    return {
        "source": "pipeline_run",
        "filename": local_graph.get("filename"),
        "domain": local_graph.get("domain"),
        "graph_edge_count": local_graph.get("edge_count", 0),
        "graph_node_count": local_graph.get("node_count", 0),
        "bridge_entity_count": len(bridge_entities),
        "path_candidate_count": len(path_candidates),
        "truncated": (
            len(bridge_entities) > MAX_LOCAL_GRAPH_REASONING_BRIDGES
            or len(path_candidates) > MAX_LOCAL_GRAPH_REASONING_PATHS
        ),
        "bridge_entities": bridge_entities[:MAX_LOCAL_GRAPH_REASONING_BRIDGES],
        "path_candidates": path_candidates[:MAX_LOCAL_GRAPH_REASONING_PATHS],
    }


def _relationship_value_tokens(metadata: dict[str, Any]) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    for key in GRAPH_CONTEXT_PROVENANCE_KEYS:
        value = metadata.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, (dict, list, tuple, set)):
            rendered = json.dumps(value, ensure_ascii=True)
        else:
            rendered = str(value)
        rendered = _clean_text(rendered)
        if rendered:
            tokens.append((key, rendered))
    return tokens


def _relationship_chunk_match_diagnostics(
    chunk: DocumentChunkPreview,
    relationship: GraphCandidateRelationship,
) -> dict[str, Any]:
    chunk_text = chunk.text.lower()
    provenance_blob = " ".join(str(item) for item in chunk.metadata.get("provenance", []) or []).lower()
    searchable = f"{chunk_text} {provenance_blob}"
    metadata = relationship.metadata or {}
    score = 0
    reasons: list[str] = []

    def add_reason(reason: str, weight: int) -> None:
        nonlocal score
        score += weight
        if reason not in reasons:
            reasons.append(reason)

    evidence = _clean_text(relationship.evidence or "")
    if evidence:
        evidence_lower = evidence.lower()
        if evidence_lower in chunk_text:
            add_reason("evidence_exact", 8)
        else:
            evidence_tokens = [
                token
                for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{3,}", evidence_lower)[:8]
                if token and token in chunk_text
            ]
            if evidence_tokens:
                add_reason(f"evidence_token:{evidence_tokens[0]}", 2)

    for label, value, weight in (
        ("source", relationship.source, 3),
        ("target", relationship.target, 3),
        ("relation", relationship.relation.replace("_", " "), 2),
    ):
        text = _clean_text(str(value or "")).lower()
        if text and text in chunk_text:
            add_reason(f"{label}_text", weight)

    for key, value in _relationship_value_tokens(metadata):
        token = f"{key}={value}".lower()
        value_lower = value.lower()
        if token in searchable:
            add_reason(f"provenance_exact:{key}", 4)
        elif key in {"slide", "page", "table", "cell"} and re.search(rf"\b{re.escape(value_lower)}\b", searchable):
            add_reason(f"provenance_location:{key}", 2)
        elif len(value_lower) >= 4 and value_lower in searchable:
            add_reason(f"provenance_value:{key}", 1)

    element_type = _clean_text(str(metadata.get("element_type") or ""))
    if element_type and any(source.startswith(f"{element_type}:") for source in chunk.source_elements):
        add_reason(f"source_element_type:{element_type}", 2)

    return {
        "score": score,
        "reasons": reasons,
    }


def _relationship_chunk_score(chunk: DocumentChunkPreview, relationship: GraphCandidateRelationship) -> int:
    return int(_relationship_chunk_match_diagnostics(chunk, relationship)["score"])


def _chunk_graph_context_relationships(
    chunk: DocumentChunkPreview,
    relationships: list[GraphCandidateRelationship],
    *,
    limit: int = MAX_CHUNK_GRAPH_CONTEXT_RELATIONSHIPS,
) -> list[GraphCandidateRelationship]:
    scored: list[tuple[int, int, float, GraphCandidateRelationship]] = []
    seen: set[tuple[str, str, str]] = set()

    for relationship in relationships:
        key = (
            _clean_text(relationship.source).lower(),
            _clean_text(relationship.relation).lower(),
            _clean_text(relationship.target).lower(),
        )
        if key in seen:
            continue
        score = _relationship_chunk_score(chunk, relationship)
        if score <= 0:
            continue
        seen.add(key)
        scored.append((score, _relationship_priority(relationship), relationship.confidence, relationship))

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [item[3] for item in scored[:limit]]


def _chunk_graph_context_scored_relationships(
    chunk: DocumentChunkPreview,
    relationships: list[GraphCandidateRelationship],
    *,
    limit: int = MAX_CHUNK_GRAPH_CONTEXT_RELATIONSHIPS,
) -> list[tuple[GraphCandidateRelationship, int, list[str]]]:
    scored: list[tuple[int, int, float, GraphCandidateRelationship, list[str]]] = []
    seen: set[tuple[str, str, str]] = set()

    for relationship in relationships:
        key = (
            _clean_text(relationship.source).lower(),
            _clean_text(relationship.relation).lower(),
            _clean_text(relationship.target).lower(),
        )
        if key in seen:
            continue
        diagnostics = _relationship_chunk_match_diagnostics(chunk, relationship)
        score = int(diagnostics["score"])
        if score <= 0:
            continue
        seen.add(key)
        scored.append((
            score,
            _relationship_priority(relationship),
            relationship.confidence,
            relationship,
            list(diagnostics.get("reasons") or []),
        ))

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

    selected: list[tuple[int, int, float, GraphCandidateRelationship, list[str]]] = []
    selected_keys: set[tuple[str, str, str]] = set()
    preferred_categories = (
        "semantic_connector",
        "connector",
        "chart_fact",
        "table_fact",
        "form_field",
        "diagram",
        "styled_text",
        "conditional_format",
        "formula",
        "pivot_table",
        "speaker_notes",
        "comment",
        "link",
        "ocr",
        "semantic",
        "presentation_structure",
        "workbook_structure",
        "legacy_office",
        "layout",
        "weak_mention",
    )

    def push(item: tuple[int, int, float, GraphCandidateRelationship, list[str]]) -> None:
        if len(selected) >= limit:
            return
        relationship = item[3]
        key = (
            _clean_text(relationship.source).lower(),
            _clean_text(relationship.relation).lower(),
            _clean_text(relationship.target).lower(),
        )
        if key in selected_keys:
            return
        selected_keys.add(key)
        selected.append(item)

    for category in preferred_categories:
        for item in scored:
            if _relationship_sample_category(item[3]) == category:
                push(item)
                break

    for item in scored:
        push(item)
        if len(selected) >= limit:
            break

    return [(item[3], item[0], item[4]) for item in selected]


def _relationship_context_line(relationship: GraphCandidateRelationship) -> str:
    line = f"{relationship.source} --[{relationship.relation}]--> {relationship.target}"
    parts = []
    if relationship.evidence:
        parts.append(f"evidence={_clip(relationship.evidence, 220)}")
    provenance = _relationship_context_provenance_text(relationship.metadata or {})
    if provenance:
        parts.append(f"provenance={_clip(provenance, 260)}")
    if parts:
        line = f"{line}; " + "; ".join(parts)
    return _clip(line, 420)


def _relationship_context_provenance_text(metadata: dict[str, Any]) -> str:
    hints = _relationship_location_hints(metadata or {})
    parts: list[str] = []

    def add(label: str, value: Any, *, prefix: bool = True) -> None:
        if value in (None, "", [], {}):
            return
        if isinstance(value, (list, dict, tuple, set)):
            rendered = json.dumps(value, ensure_ascii=True)
        else:
            rendered = str(value)
        rendered = _clean_text(rendered)
        if not rendered:
            return
        text = f"{label} {rendered}" if prefix else rendered
        if text not in parts:
            parts.append(_clip(text, 120))

    slide = hints.get("slide")
    slide_title = hints.get("slide_title")
    if slide not in (None, ""):
        add("slide", f"{slide}: {slide_title}" if slide_title else slide)
    add("page", hints.get("page"))
    add("sheet", hints.get("sheet_name"))
    add("section", hints.get("section"))
    add("table", hints.get("table_name") or hints.get("table_display_name") or hints.get("table"))
    add("cell", hints.get("table_cell_reference"))
    add("cell context", hints.get("table_cell_context"))
    add("chart", hints.get("chart_title") or hints.get("chart_label"))
    add("chart point", hints.get("chart_point_label"))
    add("axis", hints.get("chart_axis_title"))
    add("series", hints.get("series_name"))
    add("category", hints.get("category"))
    add("value", hints.get("value"))
    add("form", hints.get("label") or hints.get("field_type") or hints.get("form_kind"))
    add("visual", hints.get("layout_relation"))
    add("near", hints.get("related_label"))
    add("distance", hints.get("visual_distance_pct"))
    add("style", hints.get("style_summary") or hints.get("emphasis_term"))
    add("connector", hints.get("connector_title"))
    add("source", hints.get("source_label"))
    add("target", hints.get("target_label"))
    add("notes", hints.get("notes_part"))
    add("comment", hints.get("comment_anchor_text") or hints.get("comment_part"))
    add("media", hints.get("media_part") or hints.get("media_kind"))
    add("image", hints.get("image_part") or hints.get("image_mime"))
    add("attachment", hints.get("embedded_file_filename") or hints.get("embedded_file_name"))
    add("uri", hints.get("uri"))
    add("location", hints.get("location"))
    return "; ".join(parts[:8])


def _chunk_graph_context(
    chunk: DocumentChunkPreview,
    relationships: list[GraphCandidateRelationship],
    *,
    limit: int = MAX_CHUNK_GRAPH_CONTEXT_RELATIONSHIPS,
) -> list[dict[str, Any]]:
    return [
        {
            "rank": index,
            "source": relationship.source,
            "relation": relationship.relation,
            "target": relationship.target,
            "confidence": relationship.confidence,
            "chunk_match_score": score,
            "chunk_match_reasons": reasons,
            "evidence": _clip(relationship.evidence, 280),
            "provenance": _relationship_location_hints(relationship.metadata or {}),
            "text": _relationship_context_line(relationship),
        }
        for index, (relationship, score, reasons) in enumerate(
            _chunk_graph_context_scored_relationships(chunk, relationships, limit=limit),
            start=1,
        )
    ]


def _path_relationship_search_values(relationship: dict[str, Any]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for key in ("source", "bridge", "target", "relation", "label", "evidence", "rationale", "provenance_summary"):
        value = relationship.get(key)
        if value not in (None, ""):
            values.append((key, _clean_text(str(value))))
    for key in ("evidence_families",):
        value = relationship.get(key)
        if isinstance(value, list):
            values.extend((key, _clean_text(str(item))) for item in value if item)
    metadata = relationship.get("metadata") if isinstance(relationship.get("metadata"), dict) else {}
    for key, value in metadata.items():
        if value in (None, ""):
            continue
        rendered = json.dumps(value, ensure_ascii=True) if isinstance(value, (dict, list, tuple, set)) else str(value)
        values.append((f"metadata:{key}", _clean_text(rendered)))
    return [(key, value) for key, value in values if value]


def _path_chunk_match_diagnostics(chunk: DocumentChunkPreview, path: dict[str, Any]) -> dict[str, Any]:
    chunk_text = chunk.text.lower()
    provenance_blob = " ".join(str(item) for item in chunk.metadata.get("provenance", []) or []).lower()
    searchable = f"{chunk_text} {provenance_blob}"
    score = 0
    reasons: list[str] = []

    def add_reason(reason: str, weight: int) -> None:
        nonlocal score
        score += weight
        if reason not in reasons:
            reasons.append(reason)

    for label, value, weight in (
        ("source", path.get("source"), 3),
        ("bridge", path.get("bridge"), 4),
        ("target", path.get("target"), 3),
    ):
        text = _clean_text(str(value or "")).lower()
        if text and text in searchable:
            add_reason(f"{label}_text", weight)

    path_text = _clean_text(str(path.get("text") or ""))
    if path_text:
        path_tokens = [
            token
            for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{3,}", path_text.lower())[:12]
            if token and token in searchable
        ]
        if len(path_tokens) >= 2:
            add_reason("path_text_tokens", min(4, len(path_tokens)))

    for label, value, weight in (
        ("explanation", path.get("explanation"), 3),
        ("answer_hint", path.get("answer_hint"), 2),
    ):
        text = _clean_text(str(value or "")).lower()
        if not text:
            continue
        matched = [
            token
            for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{3,}", text)[:12]
            if token in searchable
        ]
        if matched:
            add_reason(f"path_{label}_tokens", min(weight, len(matched)))

    for family in path.get("evidence_families", []) or []:
        family_text = _clean_text(str(family or "")).replace("_", " ").lower()
        if family_text and family_text in searchable:
            add_reason(f"path_family:{family}", 2)

    for relationship in path.get("relationships", []) or []:
        if not isinstance(relationship, dict):
            continue
        for key, value in _path_relationship_search_values(relationship):
            value_lower = value.lower()
            if not value_lower:
                continue
            if key == "evidence" and value_lower in chunk_text:
                add_reason("path_evidence_exact", 6)
            elif key.startswith("metadata:"):
                metadata_key = key.split(":", 1)[1]
                token = f"{metadata_key}={value}".lower()
                if token in searchable:
                    add_reason(f"path_provenance_exact:{metadata_key}", 4)
                elif len(value_lower) >= 4 and value_lower in searchable:
                    add_reason(f"path_provenance_value:{metadata_key}", 1)
            elif len(value_lower) >= 4 and value_lower in searchable:
                add_reason(f"path_{key}_text", 2 if key in {"source", "target", "relation", "label"} else 1)

    return {
        "score": score,
        "reasons": reasons,
    }


def _chunk_graph_path_context(
    chunk: DocumentChunkPreview,
    local_graph_reasoning: dict[str, Any] | None = None,
    *,
    limit: int = MAX_CHUNK_GRAPH_PATH_CONTEXT,
) -> list[dict[str, Any]]:
    reasoning = local_graph_reasoning if isinstance(local_graph_reasoning, dict) else {}
    path_candidates = reasoning.get("path_candidates") if isinstance(reasoning.get("path_candidates"), list) else []
    scored: list[tuple[int, float, dict[str, Any], list[str]]] = []
    seen: set[tuple[str, str, str]] = set()

    for path in path_candidates:
        if not isinstance(path, dict):
            continue
        source = _clean_text(str(path.get("source") or ""))
        bridge = _clean_text(str(path.get("bridge") or ""))
        target = _clean_text(str(path.get("target") or ""))
        if not source or not bridge or not target:
            continue
        key = (source.lower(), bridge.lower(), target.lower())
        if key in seen:
            continue
        diagnostics = _path_chunk_match_diagnostics(chunk, path)
        score = int(diagnostics.get("score", 0) or 0)
        if score <= 0:
            continue
        seen.add(key)
        try:
            path_score = float(path.get("score", 0) or 0)
        except (TypeError, ValueError):
            path_score = 0.0
        scored.append((score, path_score, path, list(diagnostics.get("reasons") or [])))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    context_paths = []
    for index, (match_score, path_score, path, reasons) in enumerate(scored[:limit], start=1):
        relationships = path.get("relationships") if isinstance(path.get("relationships"), list) else []
        context_paths.append({
            "rank": index,
            "id": path.get("id"),
            "source": path.get("source"),
            "bridge": path.get("bridge"),
            "target": path.get("target"),
            "nodes": path.get("nodes") if isinstance(path.get("nodes"), list) else [
                path.get("source"),
                path.get("bridge"),
                path.get("target"),
            ],
            "hops": path.get("hops", 2),
            "score": round(path_score, 3),
            "chunk_match_score": match_score,
            "chunk_match_reasons": reasons,
            "relationships": relationships[:2],
            "evidence_families": path.get("evidence_families") if isinstance(path.get("evidence_families"), list) else [],
            "provenance_summary": path.get("provenance_summary") if isinstance(path.get("provenance_summary"), list) else [],
            "explanation": _clip(str(path.get("explanation") or ""), 520),
            "answer_hint": _clip(str(path.get("answer_hint") or ""), 360),
            "text": _clip(str(path.get("text") or ""), 520),
        })
    return context_paths


def _chunk_semantic_header_lines(chunk: DocumentChunkPreview) -> list[str]:
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    lines: list[str] = []
    scope_type = _clean_text(str(metadata.get("chunk_scope_type") or ""))
    scope_label = _clean_text(str(metadata.get("chunk_scope_label") or ""))

    if scope_type == "mixed":
        labels = metadata.get("chunk_scope_labels")
        if isinstance(labels, list):
            visible_labels = _unique_preserve_order([
                _clean_text(str(label))
                for label in labels
                if _clean_text(str(label))
            ])
            if visible_labels:
                scope_count = metadata.get("chunk_scope_count")
                count_suffix = f"; {scope_count} scopes" if scope_count else ""
                lines.append(f"Scope: mixed evidence from {', '.join(visible_labels[:4])}{count_suffix}")
    elif scope_label:
        type_suffix = f" ({scope_type})" if scope_type else ""
        lines.append(f"Scope: {scope_label}{type_suffix}")

    evidence_families = [
        _clean_text(str(family)).replace("_", " ")
        for family in metadata.get("evidence_families", [])
        if _clean_text(str(family))
    ] if isinstance(metadata.get("evidence_families"), list) else []
    if evidence_families:
        lines.append(f"Evidence types: {', '.join(_unique_preserve_order(evidence_families)[:12])}")

    location_hints = [
        _clean_text(str(hint))
        for hint in metadata.get("location_hints", [])
        if _clean_text(str(hint))
    ] if isinstance(metadata.get("location_hints"), list) else []
    if any(not hint.lower().startswith("layout ") for hint in location_hints):
        location_hints = [
            hint
            for hint in location_hints
            if not hint.lower().startswith("layout ")
        ]
    if location_hints:
        lines.append(f"Provenance: {'; '.join(_unique_preserve_order(location_hints)[:8])}")

    source_locations = (
        [
            item
            for item in metadata.get("source_locations", [])
            if isinstance(item, dict)
        ]
        if isinstance(metadata.get("source_locations"), list)
        else []
    )
    link_refs = _unique_preserve_order([
        f"{key}={_clip(str(source_location.get(key)), 180)}"
        for source_location in source_locations
        for key in (
            "uri",
            "image_relationship_id",
            "image_part",
        )
        if source_location.get(key)
    ])
    if link_refs and scope_type != "mixed":
        lines.append(f"Link refs: {' | '.join(link_refs[:12])}")

    diagram_refs = _unique_preserve_order([
        f"{key}={_clip(str(source_location.get(key)), 180)}"
        for source_location in source_locations
        for key in (
            "diagram_part",
            "diagram_node_id",
            "diagram_connection_type",
        )
        if source_location.get(key)
    ])
    if diagram_refs and scope_type != "mixed":
        lines.append(f"Diagram refs: {' | '.join(diagram_refs[:12])}")

    comment_refs = _unique_preserve_order([
        " | ".join(
            f"{key}={_clip(str(source_location.get(key)), 180)}"
            for key in (
                "comment_kind",
                "comment_anchor_text",
                "comment_anchor_cell",
                "comment_anchor_header",
                "comment_anchor_value",
                "person_id",
            )
            if source_location.get(key)
        )
        for source_location in source_locations
        if any(
            source_location.get(key)
            for key in (
                "comment_kind",
                "comment_anchor_text",
                "comment_anchor_cell",
                "comment_anchor_header",
                "comment_anchor_value",
                "person_id",
            )
        )
    ])
    if comment_refs and scope_type != "mixed":
        lines.append(f"Comment refs: {' | '.join(comment_refs[:6])}")

    key_ref_fields = (
        "metadata_part",
        "metadata_field",
        "metadata_value",
        "slide_title",
        "slide_visibility",
        "slide_part",
        "heading_kind",
        "merge_range",
        "merge_anchor_cell",
        "layout_bbox",
        "layout_zone",
        "style_summary",
        "emphasis_terms",
        "bullet_levels",
        "inherited_from",
        "layout_part",
        "master_part",
        "chart_part",
        "chart_label",
        "chart_point_label",
        "series_name",
        "category",
        "value",
        "chart_axis_title",
        "chart_legend_position",
        "chart_label_fields",
        "chart_label_position",
        "table_cell_reference",
        "table_cell_column_header",
        "table_cell_row_header",
        "table_cell_context",
        "table_cell_grid_span",
        "table_cell_row_span",
        "table_cell_merge_flags",
        "table_cell_fill_color",
        "table_cell_style_name",
        "table_cell_width_value",
        "table_cell_visual_summary",
        "table_cell_width_inches",
        "table_cell_border_top_color",
        "table_display_name",
        "table_range",
        "table_part",
        "table_columns",
        "cell",
        "cell_value_type",
        "formula_cell",
        "formula_expression",
        "formula_references",
        "formula_result",
        "formula_reference_count",
        "validation_range",
        "validation_type",
        "validation_formula1",
        "form_kind",
        "field_type",
        "label",
        "comment_kind",
        "comment_anchor_text",
        "comment_anchor_cell",
        "comment_anchor_header",
        "comment_anchor_value",
        "person_id",
        "conditional_range",
        "conditional_range_label",
        "conditional_type",
        "conditional_formula1",
        "conditional_formula_references",
        "pivot_table_name",
        "pivot_table_range",
        "pivot_table_source_name",
        "pivot_table_source_ref",
        "pivot_table_source_sheet",
        "pivot_table_row_fields",
        "pivot_table_data_fields",
        "pivot_table_cache_part",
        "named_range_name",
        "named_range_ref",
        "named_range_scope",
        "sheet_name",
        "sheet_part",
        "sheet_visibility",
        "chart_source_part",
        "chart_source_extraction_method",
        "chart_source_kind",
        "embedded_file_filename",
        "media_part",
        "uri",
        "image_relationship_id",
        "shape_title",
        "shape_description",
        "shape_id",
        "text_box_part",
        "text_box_kind",
        "connector_id",
        "connector_name",
        "connector_title",
        "connector_description",
        "source_shape_id",
        "target_shape_id",
        "source_connection_index",
        "target_connection_index",
        "diagram_part",
        "diagram_node_id",
        "control_part",
        "field_type",
        "form_field_checked",
        "form_field_options",
        "form_field_selected_option",
        "form_field_date_format",
        "form_field_date_full_value",
        "form_field_calendar",
        "revision_type",
        "revision_author",
        "revision_date",
        "revision_id",
        "revision_part",
        "revision_change_kind",
        "revision_anchor_text",
    )
    key_refs = _unique_preserve_order([
        f"{key}={_clip(json.dumps(value, ensure_ascii=True) if isinstance(value, (list, dict)) else str(value), 180)}"
        for source_location in source_locations
        for key in key_ref_fields
        for value in [source_location.get(key)]
        if value not in (None, "")
    ])
    if key_refs and scope_type != "mixed":
        lines.append(f"Key refs: {' | '.join(key_refs[:40])}")

    visual_refs = _unique_preserve_order([
        f"{key}={_clip(str(source_location.get(key)), 180)}"
        for source_location in source_locations
        for key in (
            "chart_series_visual_summary",
            "chart_point_visual_summary",
            "chart_trendline_visual_summary",
            "table_cell_visual_summary",
        )
        if source_location.get(key)
    ])
    if visual_refs and scope_type != "mixed":
        lines.append(f"Visual refs: {' | '.join(visual_refs[:8])}")

    source_refs = _unique_preserve_order([
        _metadata_line(source_location)
        for source_location in source_locations
        if _metadata_line(source_location)
    ])
    if source_refs and scope_type != "mixed":
        lines.append(f"Source refs: {' | '.join(source_refs[:12])}")

    evidence_preview = _clean_text(str(metadata.get("evidence_preview") or ""))
    if evidence_preview:
        lines.append(f"Evidence focus: {_clip(evidence_preview, 360)}")

    return lines


def _chunk_text_with_semantic_header(chunk: DocumentChunkPreview, text_limit: int) -> str:
    header_lines = _chunk_semantic_header_lines(chunk)
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    semantic_body = _clean_multiline_text(str(metadata.get("semantic_text") or ""))
    body = semantic_body or chunk.text
    if not header_lines:
        return _clip_multiline(body, text_limit)

    header = "\n".join(header_lines)
    body_limit = max(240, text_limit - len(header) - 2)
    return _clip_multiline(f"{header}\n{_clip_multiline(body, body_limit)}", text_limit)


def _enriched_chunk_text(
    chunk: DocumentChunkPreview,
    relationships: list[GraphCandidateRelationship],
    *,
    local_graph_reasoning: dict[str, Any] | None = None,
    text_limit: int = (DEFAULT_CHUNK_CHARS * 3) + MAX_CHUNK_GRAPH_CONTEXT_CHARS + MAX_CHUNK_GRAPH_PATH_CONTEXT_CHARS,
) -> str:
    graph_context = _chunk_graph_context(chunk, relationships)
    graph_paths = _chunk_graph_path_context(chunk, local_graph_reasoning)
    if not graph_context and not graph_paths:
        return _chunk_text_with_semantic_header(chunk, text_limit)
    sections = []
    if graph_context:
        context_lines = [_relationship["text"] for _relationship in graph_context]
        sections.append("Graph context:\n" + "\n".join(f"- {line}" for line in context_lines))
    if graph_paths:
        path_lines = [
            (
                f"{path['text']} (bridge={path.get('bridge')}; score={path.get('score')}; match={path.get('chunk_match_score')})"
                + (f"; why={path.get('explanation')}" if path.get("explanation") else "")
            )
            for path in graph_paths
            if path.get("text")
        ]
        if path_lines:
            sections.append("Graph path context:\n" + "\n".join(f"- {line}" for line in path_lines))
    section_text = "\n\n".join(sections)
    reserved_context_chars = len(section_text) + 2
    chunk_text_limit = max(DEFAULT_CHUNK_CHARS + MAX_CHUNK_GRAPH_CONTEXT_CHARS, text_limit - reserved_context_chars)
    enriched = _chunk_text_with_semantic_header(chunk, chunk_text_limit).rstrip() + "\n\n" + section_text
    return _clip_multiline(enriched, text_limit)


def _build_local_context_artifact(
    *,
    filename: str,
    domain: str,
    chunks: list[DocumentChunkPreview],
    relationships: list[GraphCandidateRelationship] | None = None,
    local_graph_reasoning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    relationships = relationships or []
    context_chunks = []
    for index, chunk in enumerate(chunks[:MAX_LOCAL_CONTEXT_CHUNKS]):
        graph_context = _chunk_graph_context(chunk, relationships)
        graph_paths = _chunk_graph_path_context(chunk, local_graph_reasoning)
        graph_context_reasons = _unique_preserve_order([
            str(reason)
            for relationship in graph_context
            for reason in relationship.get("chunk_match_reasons", [])
            if reason
        ])
        graph_path_reasons = _unique_preserve_order([
            str(reason)
            for path in graph_paths
            for reason in path.get("chunk_match_reasons", [])
            if reason
        ])
        context_chunks.append({
            "id": chunk.chunk_id,
            "chunk_index": index,
            "content": _enriched_chunk_text(chunk, relationships, local_graph_reasoning=local_graph_reasoning),
            "token_estimate": chunk.token_estimate,
            "source_elements": chunk.source_elements[:],
            "graph_context": graph_context,
            "graph_paths": graph_paths,
            "metadata": {
                **chunk.metadata,
                "filename": filename,
                "domain": domain,
                "graph_context_relationship_count": len(graph_context),
                "graph_path_context_count": len(graph_paths),
                "graph_context_max_match_score": max(
                    (int(edge.get("chunk_match_score", 0) or 0) for edge in graph_context),
                    default=0,
                ),
                "graph_path_max_match_score": max(
                    (int(path.get("chunk_match_score", 0) or 0) for path in graph_paths),
                    default=0,
                ),
                "graph_context_match_reasons": graph_context_reasons[:12],
                "graph_path_match_reasons": graph_path_reasons[:12],
            },
        })

    return {
        "source": "pipeline_run",
        "filename": filename,
        "domain": domain,
        "chunk_count": len(chunks),
        "cached_chunk_count": len(context_chunks),
        "graph_context_relationship_count": sum(
            len(chunk.get("graph_context", []))
            for chunk in context_chunks
            if isinstance(chunk.get("graph_context"), list)
        ),
        "graph_path_context_count": sum(
            len(chunk.get("graph_paths", []))
            for chunk in context_chunks
            if isinstance(chunk.get("graph_paths"), list)
        ),
        "evidence_cards": _chunk_evidence_cards(chunks, min(8, MAX_LOCAL_CONTEXT_CHUNKS)),
        "truncated": len(chunks) > MAX_LOCAL_CONTEXT_CHUNKS,
        "chunks": context_chunks,
    }


def _structured_filename(filename: str) -> str:
    path = Path(filename or "document")
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    return f"{stem or 'document'}.structured.md"


def _metadata_line(metadata: dict[str, Any]) -> str:
    if not metadata:
        return ""
    parts = []
    link_priority_keys = (
        "uri",
        "label",
        "link_kind",
        "relationship_id",
        "target_mode",
        "tooltip",
        "inherited_from",
        "layout_part",
        "master_part",
        "image_relationship_id",
        "image_part",
        "image_mime",
        "layout_bbox",
        "layout_position",
        "layout_zone",
        "slide",
        "slide_title",
    ) if metadata.get("uri") or metadata.get("link_kind") else ()
    form_priority_keys = (
        "form_kind",
        "field_type",
        "label",
        "value",
        "form_field_checked",
        "form_field_selected_option",
        "form_field_options",
        "form_field_option_values",
        "control_id",
        "control_name",
        "control_part",
        "relationship_id",
        "slide",
        "slide_title",
    ) if metadata.get("form_kind") or metadata.get("field_type") or metadata.get("control_part") else ()
    priority_keys = (
        *link_priority_keys,
        *form_priority_keys,
        "metadata_part",
        "metadata_field",
        "metadata_label",
        "email_message_id",
        "email_subject",
        "email_from",
        "email_sender",
        "email_to",
        "email_cc",
        "email_date",
        "email_part_index",
        "email_content_type",
        "email_attachment_index",
        "email_attachment_filename",
        "email_attachment_mime",
        "embedded_file_filename",
        "embedded_file_mime",
        "embedded_file_size_bytes",
        "embedded_file_extraction_method",
        "archive_entry_path",
        "archive_entry_filename",
        "archive_entry_mime",
        "archive_entry_size_bytes",
        "archive_entry_extraction_method",
        "section",
        "level",
        "markdown_code_language",
        "markdown_code_block_index",
        "markdown_start_line",
        "markdown_end_line",
        "markdown_frontmatter_index",
        "section_label",
        "section_level",
        "page",
        "chapter",
        "chapter_path",
        "table_cell_reference",
        "table_cell_grid_span",
        "table_cell_fill_color",
        "table_cell_visual_summary",
        "table_cell_style_name",
        "table_cell_width_value",
        "table_cell_width_inches",
        "table_cell_border_top_color",
        "format",
        "inherited_from",
        "master_part",
        "master_relationship_id",
        "layout_part",
        "layout_relationship_id",
        "inherited_part",
        "inherited_relationship_id",
        "shape_index",
        "shape_id",
        "shape_name",
        "shape_title",
        "shape_description",
        "shape_type",
        "connector_index",
        "connector_id",
        "connector_title",
        "connector_description",
        "source_label",
        "relation",
        "target_label",
        "source_shape_id",
        "target_shape_id",
        "connector_name",
        "source_connection_index",
        "target_connection_index",
        "layout_bbox",
        "layout_position",
        "layout_zone",
        "outline_level",
        "outline_title",
        "slide",
        "slide_title",
        "slide_id",
        "slide_part",
        "slide_relationship_id",
        "slide_state",
        "slide_visibility",
        "style_summary",
        "emphasis_terms",
        "emphasis_term",
        "emphasis_styles",
        "bullet_levels",
        "text_box_index",
        "text_box_part",
        "text_box_kind",
        "text_box_style",
        "sheet",
        "sheet_name",
        "heading_kind",
        "table_name",
        "table_display_name",
        "table_definition_name",
        "table_part",
        "table_range",
        "table_columns",
        "table_relationship_id",
        "table_cell_reference",
        "table_cell_column_header",
        "table_cell_row_header",
        "table_cell_context",
        "table_cell_grid_span",
        "table_cell_row_span",
        "table_cell_h_merge",
        "table_cell_v_merge",
        "table_cell_merge_flags",
        "table_cell_fill_color",
        "table_cell_visual_summary",
        "table_cell_style_name",
        "table_cell_text_anchor",
        "table_cell_vertical_align",
        "table_cell_text_direction",
        "table_cell_width_value",
        "table_cell_width_inches",
        "table_cell_width_type",
        "table_cell_shading_pattern",
        "table_cell_margin_left_inches",
        "table_cell_margin_right_inches",
        "table_cell_margin_top_inches",
        "table_cell_margin_bottom_inches",
        "table_cell_border_left_color",
        "table_cell_border_right_color",
        "table_cell_border_top_color",
        "table_cell_border_bottom_color",
        "table_cell_border_left_width_inches",
        "table_cell_border_right_width_inches",
        "table_cell_border_top_width_inches",
        "table_cell_border_bottom_width_inches",
        "table_cell_column_repeat",
        "table_cell_row_repeat",
        "table_cell_is_covered",
        "cell_value_type",
        "auto_filter_range",
        "table",
        "row",
        "column",
        "cell",
        "table_source",
        "table_bbox",
        "table_row_count",
        "table_column_count",
        "formulas",
        "conditional_label",
        "conditional_range",
        "conditional_range_label",
        "conditional_type",
        "conditional_operator",
        "conditional_priority",
        "conditional_dxf_id",
        "conditional_stop_if_true",
        "conditional_text",
        "conditional_time_period",
        "conditional_rank",
        "conditional_percent",
        "conditional_formula1",
        "conditional_formula2",
        "conditional_formulas",
        "conditional_formula_references",
        "conditional_color_scale",
        "formula_cell",
        "formula_expression",
        "formula_references",
        "formula_result",
        "formula_reference_count",
        "merge_range",
        "merge_anchor_cell",
        "merge_cell_count",
        "merge_row_count",
        "merge_column_count",
        "named_range_name",
        "named_range_ref",
        "named_range_scope",
        "named_range_local_sheet_id",
        "named_range_hidden",
        "named_range_comment",
        "pivot_table_name",
        "pivot_table_range",
        "pivot_table_source_name",
        "pivot_table_source_ref",
        "pivot_table_source_sheet",
        "pivot_table_row_fields",
        "pivot_table_column_fields",
        "pivot_table_page_fields",
        "pivot_table_data_fields",
        "pivot_table_cache_part",
        "pivot_table_part",
        "pivot_table_cache_fields",
        "pivot_table_cache_id",
        "pivot_table_relationship_id",
        "pivot_table_source_type",
        "pivot_table_cache_relationship_id",
        "pivot_table_data_field_count",
        "pivot_table_record_count",
        "form_kind",
        "field_type",
        "label",
        "value",
        "form_field_checked",
        "form_field_selected_option",
        "form_field_options",
        "form_field_option_values",
        "control_id",
        "control_name",
        "control_part",
        "form_field_alias",
        "form_field_tag",
        "form_field_id",
        "form_field_placeholder",
        "form_field_date_format",
        "form_field_date_full_value",
        "form_field_calendar",
        "form_field_lock",
        "validation_range",
        "validation_type",
        "validation_operator",
        "validation_formula1",
        "validation_formula2",
        "validation_allow_blank",
        "validation_prompt_title",
        "validation_prompt",
        "validation_error_title",
        "validation_error",
        "control_shape_id",
        "control_show_as_icon",
        "chart_title",
        "chart_part",
        "chart_label",
        "chart_point_label",
        "chart_axis_index",
        "chart_axis_type",
        "chart_axis_id",
        "chart_axis_title",
        "chart_axis_position",
        "chart_axis_number_format",
        "chart_legend_position",
        "chart_legend_overlay",
        "chart_legend_series",
        "chart_label_fields",
        "chart_label_position",
        "chart_label_examples",
        "chart_label_number_format",
        "chart_series_fill_color",
        "chart_series_line_color",
        "chart_series_line_width_points",
        "chart_series_line_dash",
        "chart_series_visual_summary",
        "chart_marker_symbol",
        "chart_marker_size",
        "chart_marker_fill_color",
        "chart_marker_line_color",
        "chart_marker_line_width_points",
        "chart_point_fill_color",
        "chart_point_line_color",
        "chart_point_visual_summary",
        "chart_trendline_type",
        "chart_trendline_name",
        "chart_trendline_line_color",
        "chart_trendline_line_width_points",
        "chart_trendline_line_dash",
        "chart_trendline_display_equation",
        "chart_trendline_display_r_squared",
        "chart_trendline_visual_summary",
        "series_name",
        "category",
        "value",
        "chart_source_part",
        "chart_source_extraction_method",
        "chart_source_uri",
        "chart_source_relationship_id",
        "chart_source_kind",
        "chart_source_mime",
        "chart_source_index",
        "diagram_part",
        "diagram_node_id",
        "diagram_level",
        "diagram_connection_type",
        "parent_label",
        "group_index",
        "group_id",
        "group_name",
        "group_title",
        "group_description",
        "group_label",
        "group_child_shape_ids",
        "group_child_labels",
        "group_child_count",
        "parent_group_id",
        "parent_group_label",
        "parent_group_index",
        "drawing_part",
        "image",
        "image_part",
        "image_relationship_id",
        "image_relationship_type",
        "image_target_mode",
        "image_width",
        "image_height",
        "image_mime",
        "ocr_source",
        "ocr_engine",
        "media_part",
        "media_kind",
        "media_mime",
        "embedded_file_name",
        "embedded_file_filename",
        "embedded_file_description",
        "embedded_file_mime",
        "embedded_file_size_bytes",
        "embedded_file_created",
        "embedded_file_modified",
        "embedded_file_extraction_method",
        "legacy_office_format",
        "legacy_office_stream_index",
        "legacy_office_stream_path",
        "legacy_office_stream_size_bytes",
        "legacy_office_stream_truncated",
        "legacy_office_extraction_method",
        "legacy_office_string_count",
        "json_path",
        "relationship_id",
        "target_mode",
        "link_kind",
        "uri",
        "label",
        "action",
        "anchor",
        "tooltip",
        "target_frame",
        "display",
        "part_name",
        "layout_position",
        "layout_zone",
        "comment_id",
        "comment_kind",
        "person_id",
        "threaded_comment_id",
        "threaded_parent_id",
        "comment_anchor_text",
        "comment_anchor_section",
        "comment_anchor_paragraph",
        "comment_anchor_sheet",
        "comment_anchor_cell",
        "comment_anchor_header",
        "comment_anchor_value",
        "comment_datetime",
        "author",
        "notes_part",
        "notes_relationship_id",
        "note_id",
        "note_part",
        "note_anchor_text",
        "note_anchor_section",
        "note_anchor_paragraph",
        "revision_type",
        "revision_author",
        "revision_date",
        "revision_id",
        "revision_part",
        "revision_change_kind",
        "revision_anchor_text",
        "annotation_type",
        "annotation_author",
        "annotation_subject",
        "annotation_content",
        "annotation_name",
        "sheet_id",
        "sheet_part",
        "sheet_relationship_id",
        "sheet_state",
        "sheet_visibility",
    )
    ordered_keys = list(dict.fromkeys(key for key in priority_keys if key in metadata))
    ordered_keys.extend(key for key in metadata.keys() if key not in set(ordered_keys))

    for key in ordered_keys:
        value = metadata.get(key)
        if value is None or value == "" or key in {"cells", "headers", "runs", "image_sha256", "media_sha256"}:
            continue
        if isinstance(value, (list, dict)):
            rendered = json.dumps(value, ensure_ascii=True)
        else:
            rendered = str(value)
        if rendered:
            parts.append(f"{key}={_clip(rendered, 180)}")
    return "; ".join(parts[:12])


def _metadata_has_value(metadata: dict[str, Any], key: str) -> bool:
    value = metadata.get(key)
    if value is None or value == "":
        return False
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def _detected_capabilities(elements: list[DocumentElement]) -> list[str]:
    detected: set[str] = set()
    for element in elements:
        capability = CAPABILITY_BY_ELEMENT_TYPE.get(element.type)
        if capability:
            detected.add(capability)
        if _metadata_has_value(element.metadata, "formulas") or _metadata_has_value(element.metadata, "formula"):
            detected.add("formulas")
        if element.metadata.get("runs") or element.metadata.get("emphasis_terms") or element.metadata.get("bullet_levels"):
            detected.add("styled_text")
        if element.metadata.get("ocr_text") or element.metadata.get("ocr_engine") or element.type in {"image_ocr_text", "page_ocr_text"}:
            detected.add("ocr")
        if element.metadata.get("layout_bbox") or element.metadata.get("layout_zone"):
            detected.add("layout")

    return [capability for capability in CAPABILITY_ORDER if capability in detected]


def _document_complexity(capabilities: list[str], elements: list[DocumentElement], relationships: list[GraphCandidateRelationship]) -> str:
    capability_set = set(capabilities)
    if capability_set & COMPLEX_CAPABILITIES or len(capability_set) >= 7 or len(relationships) >= 80:
        return "complex"
    if capability_set & RICH_CAPABILITIES or len(capability_set) >= 4 or len(relationships) >= 25:
        return "rich"
    if capability_set or len(elements) >= 4:
        return "structured"
    return "plain"


def _top_counts(counts: dict[str, int], limit: int = 12) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])


def _artifact_child_count(elements: list[DocumentElement], metadata_key: str, value: Any, exclude_type: str) -> int:
    if value is None or value == "":
        return 0
    return sum(
        1
        for element in elements
        if element.type != exclude_type and element.metadata.get(metadata_key) == value
    )


def _source_artifact_profiles(elements: list[DocumentElement], *, limit: int = 12) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for element in elements:
        if element.type == "archive_entry":
            path = element.metadata.get("archive_entry_path") or element.metadata.get("archive_entry_filename")
            if not path:
                continue
            key = ("archive_entry", str(path))
            if key in seen:
                continue
            seen.add(key)
            artifacts.append({
                "kind": "archive_entry",
                "label": str(path),
                "path": path,
                "filename": element.metadata.get("archive_entry_filename"),
                "mime": element.metadata.get("archive_entry_mime"),
                "size_bytes": element.metadata.get("archive_entry_size_bytes"),
                "sha256": element.metadata.get("archive_entry_sha256"),
                "parsed_child_elements": _artifact_child_count(elements, "archive_entry_path", path, "archive_entry"),
            })
        elif element.type == "embedded_object" and (
            element.metadata.get("embedded_file_filename") or element.metadata.get("media_kind") == "pdf_attachment"
        ):
            filename = (
                element.metadata.get("embedded_file_filename")
                or element.metadata.get("embedded_file_name")
                or element.metadata.get("media_part")
            )
            if not filename:
                continue
            key = ("attachment", str(filename))
            if key in seen:
                continue
            seen.add(key)
            artifacts.append({
                "kind": "attachment",
                "label": str(filename),
                "filename": filename,
                "description": element.metadata.get("embedded_file_description"),
                "mime": element.metadata.get("embedded_file_mime") or element.metadata.get("media_mime"),
                "size_bytes": element.metadata.get("embedded_file_size_bytes") or element.metadata.get("media_size_bytes"),
                "sha256": element.metadata.get("embedded_file_sha256") or element.metadata.get("media_sha256"),
                "parsed_child_elements": _artifact_child_count(elements, "embedded_file_filename", filename, "embedded_object"),
            })
        elif element.type == "legacy_office_stream":
            stream_path = element.metadata.get("legacy_office_stream_path")
            if not stream_path:
                continue
            key = ("legacy_office_stream", str(stream_path))
            if key in seen:
                continue
            seen.add(key)
            artifacts.append({
                "kind": "legacy_office_stream",
                "label": str(stream_path),
                "path": stream_path,
                "format": element.metadata.get("legacy_office_format"),
                "size_bytes": element.metadata.get("legacy_office_stream_size_bytes"),
                "sha256": element.metadata.get("legacy_office_stream_sha256"),
                "extraction_method": element.metadata.get("legacy_office_extraction_method"),
                "parsed_child_elements": _artifact_child_count(elements, "legacy_office_stream_path", stream_path, "legacy_office_stream"),
            })

    return artifacts[:limit]


def _graph_evidence_profile(relationships: list[GraphCandidateRelationship]) -> dict[str, Any]:
    structural_relationship_count = 0
    table_fact_count = 0
    attachment_relationship_count = 0
    archive_relationship_count = 0
    ocr_relationship_count = 0
    layout_relationship_count = 0
    styled_text_relationship_count = 0
    chart_relationship_count = 0
    code_block_relationship_count = 0
    diagram_relationship_count = 0
    shape_relationship_count = 0
    speaker_notes_relationship_count = 0
    link_relationship_count = 0

    for relationship in relationships:
        relation = str(relationship.relation or "").lower()
        metadata = relationship.metadata or {}
        is_structural = relation.startswith(("has_", "contains_"))
        if is_structural:
            structural_relationship_count += 1
        if metadata.get("container_type") or is_structural:
            layout_relationship_count += 1
        if metadata.get("table") is not None and not is_structural:
            table_fact_count += 1
        if metadata.get("embedded_file_filename") or metadata.get("embedded_file_name"):
            attachment_relationship_count += 1
        if metadata.get("archive_entry_path") or metadata.get("archive_entry_filename"):
            archive_relationship_count += 1
        if metadata.get("ocr_source"):
            ocr_relationship_count += 1
        if metadata.get("styled_text") or relation in {"emphasizes", "uses_bullet_structure"}:
            styled_text_relationship_count += 1
        if metadata.get("markdown_code_language") or metadata.get("element_type") == "code_block" or relation in {"contains_code_block", "uses_language"}:
            code_block_relationship_count += 1
        if (
            metadata.get("diagram_part")
            or metadata.get("diagram_node_id")
            or metadata.get("diagram_connection_type")
            or relation in {"contains_diagram_node", "contains_diagram_relationship"}
        ):
            diagram_relationship_count += 1
        if (
            metadata.get("group_id")
            or metadata.get("parent_group_id")
            or relation in {"contains_shape", "contains_shape_group", "groups_visual"}
        ):
            shape_relationship_count += 1
        if metadata.get("notes_part") or relation in {"has_speaker_notes", "contains_speaker_notes"}:
            speaker_notes_relationship_count += 1
        if metadata.get("uri") or metadata.get("link_kind") or relation in {"links_to", "contains_link"}:
            link_relationship_count += 1
        if (
            metadata.get("chart_part")
            or metadata.get("chart_label")
            or metadata.get("chart_point_label")
            or relation in {"contains_metric", "uses_chart_source", "contains_chart", "contains_chart_series", "contains_chart_point", "has_axis", "has_legend", "has_data_labels", "has_trendline"}
            or relation.endswith("_value")
        ):
            chart_relationship_count += 1

    return {
        "relationship_count": len(relationships),
        "semantic_relationship_count": max(0, len(relationships) - structural_relationship_count),
        "structural_relationship_count": structural_relationship_count,
        "layout_relationship_count": layout_relationship_count,
        "table_fact_count": table_fact_count,
        "attachment_relationship_count": attachment_relationship_count,
        "archive_relationship_count": archive_relationship_count,
        "ocr_relationship_count": ocr_relationship_count,
        "styled_text_relationship_count": styled_text_relationship_count,
        "code_block_relationship_count": code_block_relationship_count,
        "diagram_relationship_count": diagram_relationship_count,
        "shape_relationship_count": shape_relationship_count,
        "chart_relationship_count": chart_relationship_count,
        "speaker_notes_relationship_count": speaker_notes_relationship_count,
        "link_relationship_count": link_relationship_count,
    }


def _distinct_table_count(elements: list[DocumentElement]) -> int:
    table_keys: set[tuple[Any, ...]] = set()
    saw_unscoped_table = False

    for element in elements:
        if element.type not in {"table_header", "table_row"}:
            continue
        metadata = element.metadata or {}
        table = metadata.get("table")
        table_name = metadata.get("table_name")
        source = (
            metadata.get("archive_entry_path")
            or metadata.get("embedded_file_filename")
            or metadata.get("sheet_name")
            or metadata.get("slide")
            or metadata.get("page")
            or metadata.get("chapter_path")
            or "document"
        )
        if table is not None or table_name:
            table_keys.add((source, table_name, table))
        else:
            saw_unscoped_table = True

    return len(table_keys) + (1 if saw_unscoped_table else 0)


def _build_evidence_profile(
    elements: list[DocumentElement],
    relationships: list[GraphCandidateRelationship],
) -> dict[str, Any]:
    element_type_counts = Counter(element.type for element in elements)
    table_elements = (
        element_type_counts.get("table_header", 0)
        + element_type_counts.get("table_row", 0)
        + element_type_counts.get("table_cell", 0)
    )
    quick_counts = {
        "archive_entries": element_type_counts.get("archive_entry", 0),
        "attachments": sum(
            1
            for element in elements
            if element.type == "embedded_object"
            and (element.metadata.get("embedded_file_filename") or element.metadata.get("media_kind") == "pdf_attachment")
        ),
        "layout_elements": sum(
            1
            for element in elements
            if element.metadata.get("layout_bbox") or element.metadata.get("layout_zone")
        ),
        "tables": _distinct_table_count(elements),
        "table_elements": table_elements,
        "sheets": element_type_counts.get("sheet_metadata", 0),
        "hidden_sheets": sum(
            1
            for element in elements
            if element.type == "sheet_metadata"
            and str(element.metadata.get("sheet_visibility") or element.metadata.get("sheet_state") or "visible").lower() not in {"", "visible"}
        ),
        "slides": element_type_counts.get("slide_metadata", 0),
        "hidden_slides": sum(
            1
            for element in elements
            if element.type == "slide_metadata"
            and str(element.metadata.get("slide_visibility") or element.metadata.get("slide_state") or "visible").lower() not in {"", "visible"}
        ),
        "pivot_tables": element_type_counts.get("pivot_table", 0),
        "charts": (
            element_type_counts.get("chart_axis", 0)
            + element_type_counts.get("chart_data_label", 0)
            + element_type_counts.get("chart_legend", 0)
            + element_type_counts.get("chart_series", 0)
            + element_type_counts.get("chart_point", 0)
            + element_type_counts.get("chart_trendline", 0)
        ),
        "diagrams": element_type_counts.get("diagram_node", 0) + element_type_counts.get("diagram_relationship", 0),
        "shapes": (
            element_type_counts.get("shape_group", 0)
            + element_type_counts.get("shape_metadata", 0)
            + element_type_counts.get("text_box", 0)
        ),
        "connectors": element_type_counts.get("shape_connector", 0),
        "speaker_notes": element_type_counts.get("speaker_notes", 0),
        "images": element_type_counts.get("image", 0),
        "ocr_blocks": element_type_counts.get("image_ocr_text", 0) + element_type_counts.get("page_ocr_text", 0),
        "forms": element_type_counts.get("form_field", 0),
        "comments": element_type_counts.get("comment", 0) + element_type_counts.get("annotation", 0),
        "revisions": element_type_counts.get("revision", 0),
        "links": element_type_counts.get("link", 0),
        "code_blocks": element_type_counts.get("code_block", 0),
        "media": element_type_counts.get("media", 0),
        "message_body_parts": element_type_counts.get("email_body", 0),
        "legacy_office_streams": element_type_counts.get("legacy_office_stream", 0),
        "legacy_office_text_blocks": element_type_counts.get("legacy_office_text", 0),
        "styled_text_blocks": sum(
            1
            for element in elements
            if element.metadata.get("runs") or element.metadata.get("emphasis_terms") or element.metadata.get("bullet_levels")
        ),
    }

    return {
        "quick_counts": {key: value for key, value in quick_counts.items() if value},
        "source_artifacts": _source_artifact_profiles(elements),
        "graph_evidence": _graph_evidence_profile(relationships),
    }


def _build_document_profile(
    *,
    elements: list[DocumentElement],
    chunks: list[DocumentChunkPreview],
    entities: list[dict[str, Any]],
    relationships: list[GraphCandidateRelationship],
    extraction_method: str,
    warnings: list[str],
    structured_truncated: bool,
    use_structured_text: bool,
    build_knowledge_graph: bool,
) -> dict[str, Any]:
    element_type_counts = Counter(element.type for element in elements)
    relationship_type_counts = Counter(relationship.relation for relationship in relationships)
    capabilities = _detected_capabilities(elements)
    evidence_profile = _build_evidence_profile(elements, relationships)
    metadata_fields = sorted({
        str(element.metadata.get("metadata_field"))
        for element in elements
        if element.type == "document_metadata" and element.metadata.get("metadata_field")
    })

    quality_warnings = list(warnings[:12])
    has_text_evidence = any(
        element.type in {
            "heading",
            "slide_title",
            "slide_text",
            "text_box",
            "paragraph",
            "page_text",
            "image_ocr_text",
            "page_ocr_text",
            "legacy_office_text",
            "table_header",
            "table_row",
            "table_cell",
            "chart_axis",
            "chart_data_label",
            "chart_legend",
            "chart_series",
            "chart_point",
            "chart_trendline",
            "code_block",
            "shape_group",
            "diagram_node",
            "form_field",
            "comment",
            "annotation",
            "revision",
            "email_body",
        }
        for element in elements
    )
    if not elements:
        quality_warnings.append("No extractable document structure was found.")
    if element_type_counts.get("image") and not has_text_evidence and "ocr" not in capabilities:
        quality_warnings.append("Document appears image-only; set GRAPHRAG_ENABLE_IMAGE_OCR=true to extract scanned text.")
    if not chunks:
        quality_warnings.append("No embedding chunks were planned.")
    if build_knowledge_graph and not relationships:
        quality_warnings.append("No graph candidate relationships were found.")
    if use_structured_text and structured_truncated:
        quality_warnings.append("Structured ingest text was truncated before upstream submission.")

    score = 0
    if elements:
        score += 25
    if chunks:
        score += 25
    if relationships or not build_knowledge_graph:
        score += 20
    if entities:
        score += 10
    if capabilities:
        score += 10
    if extraction_method in STRUCTURED_SUBMIT_METHODS:
        score += 10
    score -= min(20, len(warnings) * 4)
    if structured_truncated:
        score -= 10
    readiness_score = max(0, min(100, score))

    return {
        "complexity": _document_complexity(capabilities, elements, relationships),
        "readiness_score": readiness_score,
        "detected_capabilities": capabilities,
        "recommended_payload": "structured_text" if use_structured_text else "original_file",
        "structured_text_recommended": use_structured_text,
        "extraction_method": extraction_method,
        "element_count": len(elements),
        "chunk_count": len(chunks),
        "entity_candidate_count": len(entities),
        "graph_candidate_count": len(relationships),
        "embedding_input_count": len(chunks),
        "metadata_fields": metadata_fields[:40],
        "element_type_counts": dict(element_type_counts),
        "relationship_type_counts": dict(relationship_type_counts),
        "evidence_profile": evidence_profile,
        "quality_warnings": quality_warnings,
    }


def _should_submit_structured_text(extraction_method: str, elements: list[DocumentElement], submit_structured_text: bool) -> bool:
    if not submit_structured_text:
        return False
    if extraction_method in STRUCTURED_SUBMIT_METHODS:
        return True
    return any(element.type in {
        "chart_point",
        "code_block",
        "chart_axis",
        "chart_data_label",
        "chart_legend",
        "chart_series",
        "chart_trendline",
        "table_cell",
        "shape_group",
        "diagram_node",
        "diagram_relationship",
        "document_metadata",
        "embedded_object",
        "email_body",
        "form_field",
        "image",
        "media",
        "legacy_office_text",
        "annotation",
        "outline_item",
        "image_ocr_text",
        "page_ocr_text",
        "revision",
        "table_row",
        "text_box",
    } for element in elements)


def _build_structured_ingest_text(
    *,
    filename: str,
    domain: str,
    content_type: str | None,
    sha256: str,
    extraction_method: str,
    elements: list[DocumentElement],
    chunks: list[DocumentChunkPreview],
    entities: list[dict[str, Any]],
    relationships: list[GraphCandidateRelationship],
    warnings: list[str],
    local_graph_reasoning: dict[str, Any] | None = None,
) -> tuple[str, bool]:
    lines = [
        f"# Structured GraphRAG Ingest: {Path(filename or 'document').name}",
        "",
        "## Source",
        f"- Filename: {filename or 'document'}",
        f"- Domain: {domain}",
        f"- Content type: {content_type or 'application/octet-stream'}",
        f"- Extraction method: {extraction_method}",
        f"- SHA256: {sha256}",
        "",
    ]

    if warnings:
        lines.extend(["## Extraction Warnings", *[f"- {warning}" for warning in warnings[:20]], ""])

    lines.extend(["## Extracted Structure"])
    for index, element in enumerate(elements, start=1):
        metadata = _metadata_line(element.metadata)
        location = f" | {element.location}" if element.location else ""
        metadata_suffix = f" | {metadata}" if metadata else ""
        lines.append(f"{index}. [{element.type}{location}{metadata_suffix}] {element.text}")
    lines.append("")

    if entities:
        lines.extend(["## Candidate Entities"])
        for entity in entities[:80]:
            lines.append(
                f"- {entity.get('name', '')} ({entity.get('type', 'concept')}) "
                f"score={entity.get('score', 0)} mentions={entity.get('mentions', 1)}"
            )
        lines.append("")

    if relationships:
        lines.extend(["## Candidate Graph Relationships"])
        for relationship in relationships[:150]:
            lines.append(
                f"- {relationship.source} --[{relationship.relation}]--> {relationship.target} "
                f"confidence={relationship.confidence}; evidence={_clip(relationship.evidence, 280)}"
            )
        lines.append("")

    if chunks:
        lines.extend(["## Embedding Chunks"])
        for chunk in chunks:
            lines.append(f"### {chunk.chunk_id}")
            lines.append(_enriched_chunk_text(chunk, relationships, local_graph_reasoning=local_graph_reasoning))
            lines.append("")

    text = "\n".join(lines).strip() + "\n"
    truncated = False
    if len(text) > MAX_STRUCTURED_INGEST_CHARS:
        text = (
            text[:MAX_STRUCTURED_INGEST_CHARS].rstrip()
            + "\n\n[Structured ingest truncated by llama-nexus before upstream submission]\n"
        )
        truncated = True
    return text, truncated


def build_document_pipeline_preview(
    *,
    filename: str,
    content: bytes,
    content_type: str | None = None,
    domain: str = "general",
    use_semantic_chunking: bool = True,
    build_knowledge_graph: bool = True,
    sample_limit: int = 5,
    submit_structured_text: bool = True,
    include_internal_payload: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build a tracked, local preview of the document-to-GraphRAG pipeline."""
    run_id = run_id or f"graphrag_ingest_{uuid4().hex[:12]}"
    created_at = _utc_now()
    extension = Path(filename or "").suffix.lower()
    sha256 = hashlib.sha256(content).hexdigest()
    stages: list[dict[str, Any]] = []

    start = time.perf_counter()
    stages.append(_stage(
        "received",
        "complete",
        start,
        counts={"bytes": len(content)},
        sample_outputs={
            "filename": filename,
            "content_type": content_type or "application/octet-stream",
            "extension": extension,
            "sha256": sha256,
        },
    ))

    start = time.perf_counter()
    elements, warnings, extraction_method = _extract_elements(filename, content, content_type)
    counts_by_type = Counter(element.type for element in elements)
    stages.append(_stage(
        "structure_extraction",
        "complete" if elements else "warning",
        start,
        method=extraction_method,
        counts={"elements": len(elements), "by_type": dict(counts_by_type)},
        warnings=warnings,
        sample_outputs={"elements": _sample_elements(elements, sample_limit)},
    ))

    start = time.perf_counter()
    chunks = _build_chunks(elements, respect_semantic_boundaries=use_semantic_chunking)
    chunk_scope_counts = Counter(
        str(chunk.metadata.get("chunk_scope_type") or "document")
        for chunk in chunks
    )
    chunk_scope_keys = {
        str(chunk.metadata.get("chunk_scope_key") or "")
        for chunk in chunks
        if chunk.metadata.get("chunk_scope_key")
    }
    stages.append(_stage(
        "chunking",
        "complete" if chunks else "warning",
        start,
        strategy="semantic_structure_aware" if use_semantic_chunking else "structure_aware",
        counts={
            "chunks": len(chunks),
            "estimated_tokens": sum(chunk.token_estimate for chunk in chunks),
            "semantic_scopes": len(chunk_scope_keys),
            "by_scope_type": dict(chunk_scope_counts),
        },
        sample_outputs={
            "chunks": _sample_chunks(chunks, sample_limit),
            "chunk_evidence_cards": _chunk_evidence_cards(chunks, sample_limit),
            "semantic_scopes": len(chunk_scope_keys),
            "scope_type_counts": dict(chunk_scope_counts),
        },
    ))

    start = time.perf_counter()
    entities = _candidate_entities(elements)
    relationships = _candidate_relationships(filename, elements, entities)
    stages.append(_stage(
        "graph_candidates",
        "complete" if build_knowledge_graph else "skipped",
        start,
        counts={"entities": len(entities), "relationships": len(relationships)},
        sample_outputs={
            "entities": entities[:sample_limit],
            "relationships": _sample_relationships(relationships, sample_limit),
            "relationship_evidence_cards": _relationship_evidence_cards(relationships, sample_limit),
            "relationship_sample_categories": dict(Counter(
                _relationship_sample_category(relationship)
                for relationship in relationships
            )),
        },
    ))

    start = time.perf_counter()
    local_graph = _build_local_graph_artifact(
        filename=filename,
        domain=domain,
        entities=entities,
        relationships=relationships,
    ) if build_knowledge_graph else {
        "source": "pipeline_run",
        "filename": filename,
        "domain": domain,
        "node_count": 0,
        "relationship_count": 0,
        "edge_count": 0,
        "truncated": False,
        "nodes": [],
        "edges": [],
    }
    stages.append(_stage(
        "local_graph_artifact",
        "complete" if build_knowledge_graph and local_graph.get("edge_count", 0) else "skipped" if not build_knowledge_graph else "warning",
        start,
        counts={
            "nodes": local_graph.get("node_count", 0),
            "edges": local_graph.get("edge_count", 0),
            "relationships": local_graph.get("relationship_count", 0),
            "truncated": local_graph.get("truncated", False),
        },
        sample_outputs={
            "nodes": _sample_graph_records(local_graph.get("nodes", []), sample_limit),
            "edges": _sample_graph_records(local_graph.get("edges", []), sample_limit),
            "source": local_graph.get("source"),
            "filename": local_graph.get("filename"),
            "domain": local_graph.get("domain"),
        },
    ))

    start = time.perf_counter()
    local_graph_reasoning = _build_local_graph_reasoning_index(local_graph) if build_knowledge_graph else {
        "source": "pipeline_run",
        "filename": filename,
        "domain": domain,
        "graph_edge_count": 0,
        "graph_node_count": 0,
        "bridge_entity_count": 0,
        "path_candidate_count": 0,
        "truncated": False,
        "bridge_entities": [],
        "path_candidates": [],
    }
    stages.append(_stage(
        "local_graph_reasoning_index",
        (
            "complete"
            if local_graph_reasoning.get("bridge_entity_count", 0) or local_graph_reasoning.get("path_candidate_count", 0)
            else "skipped"
            if not build_knowledge_graph
            else "warning"
        ),
        start,
        counts={
            "bridge_entities": local_graph_reasoning.get("bridge_entity_count", 0),
            "path_candidates": local_graph_reasoning.get("path_candidate_count", 0),
            "graph_edges": local_graph_reasoning.get("graph_edge_count", 0),
            "graph_nodes": local_graph_reasoning.get("graph_node_count", 0),
            "truncated": local_graph_reasoning.get("truncated", False),
        },
        sample_outputs={
            "bridge_entities": local_graph_reasoning.get("bridge_entities", [])[:sample_limit],
            "path_candidates": local_graph_reasoning.get("path_candidates", [])[:sample_limit],
            "filename": local_graph_reasoning.get("filename"),
            "domain": local_graph_reasoning.get("domain"),
        },
    ))

    start = time.perf_counter()
    representative_chunks = _representative_chunks(chunks, sample_limit)
    embedding_inputs = [
        _embedding_input_sample(
            chunk,
            filename=filename,
            domain=domain,
            content_type=content_type,
            sha256=sha256,
            relationships=relationships,
            local_graph_reasoning=local_graph_reasoning,
        )
        for chunk in representative_chunks
    ]
    representative_embedding_inputs = [
        _embedding_input_sample(
            chunk,
            filename=filename,
            domain=domain,
            content_type=content_type,
            sha256=sha256,
            relationships=relationships,
            local_graph_reasoning=local_graph_reasoning,
        )
        for chunk in representative_chunks
    ]
    representative_embedding_family_counts = Counter(
        str(category)
        for item in representative_embedding_inputs
        if isinstance(item.get("metadata"), dict)
        for category in (
            item["metadata"].get("sample_categories")
            if isinstance(item["metadata"].get("sample_categories"), list)
            else [item["metadata"].get("sample_category") or "document"]
        )
        if category
    )
    stages.append(_stage(
        "embedding_plan",
        "complete" if chunks else "warning",
        start,
        counts={
            "embedding_inputs": len(chunks),
            "graph_context_relationships": sum(
                len(item["metadata"].get("graph_context", []))
                for item in embedding_inputs
                if isinstance(item.get("metadata"), dict)
            ),
            "graph_path_context_paths": sum(
                len(item["metadata"].get("graph_paths", []))
                for item in embedding_inputs
                if isinstance(item.get("metadata"), dict)
            ),
            "representative_graph_context_relationships": sum(
                len(item["metadata"].get("graph_context", []))
                for item in representative_embedding_inputs
                if isinstance(item.get("metadata"), dict)
            ),
            "representative_graph_path_context_paths": sum(
                len(item["metadata"].get("graph_paths", []))
                for item in representative_embedding_inputs
                if isinstance(item.get("metadata"), dict)
            ),
            "graph_enriched_representative_inputs": len([
                item
                for item in representative_embedding_inputs
                if isinstance(item.get("metadata"), dict)
                and (
                    item["metadata"].get("graph_context_relationship_count", 0)
                    or item["metadata"].get("graph_path_context_count", 0)
                )
            ]),
        },
        sample_outputs={
            "embedding_inputs": embedding_inputs,
            "representative_embedding_inputs": representative_embedding_inputs,
            "embedding_evidence_cards": _embedding_evidence_cards(representative_embedding_inputs, sample_limit),
            "embedding_sample_categories": dict(Counter(
                str((item.get("metadata") or {}).get("sample_category") or "document")
                for item in representative_embedding_inputs
            )),
            "embedding_evidence_family_counts": dict(representative_embedding_family_counts),
        },
    ))

    start = time.perf_counter()
    structured_text, structured_truncated = _build_structured_ingest_text(
        filename=filename,
        domain=domain,
        content_type=content_type,
        sha256=sha256,
        extraction_method=extraction_method,
        elements=elements,
        chunks=chunks,
        entities=entities,
        relationships=relationships,
        local_graph_reasoning=local_graph_reasoning,
        warnings=warnings,
    )
    use_structured_text = _should_submit_structured_text(extraction_method, elements, submit_structured_text)
    structured_filename = _structured_filename(filename)

    structured_stage = _stage(
        "structured_ingest_document",
        "complete" if structured_text else "warning",
        start,
        strategy="structured_text" if use_structured_text else "original_file",
        counts={
            "characters": len(structured_text),
            "estimated_tokens": max(1, round(len(structured_text) / 4)) if structured_text else 0,
        },
        warnings=["Structured ingest text was truncated before upstream submission."] if structured_truncated else [],
        sample_outputs={
            "filename": structured_filename,
            "content_type": "text/markdown",
            "uses_structured_text": use_structured_text,
            "text": _clip(structured_text, 1800),
        },
    )

    start = time.perf_counter()
    document_profile = _build_document_profile(
        elements=elements,
        chunks=chunks,
        entities=entities,
        relationships=relationships,
        extraction_method=extraction_method,
        warnings=warnings,
        structured_truncated=structured_truncated,
        use_structured_text=use_structured_text,
        build_knowledge_graph=build_knowledge_graph,
    )
    stages.append(_stage(
        "document_profile",
        "complete" if document_profile["readiness_score"] >= 70 else "warning",
        start,
        counts={
            "capabilities": len(document_profile["detected_capabilities"]),
            "metadata_fields": len(document_profile["metadata_fields"]),
            "embedding_inputs": document_profile["embedding_input_count"],
            "graph_candidates": document_profile["graph_candidate_count"],
        },
        warnings=document_profile["quality_warnings"][:5],
        sample_outputs={
            "complexity": document_profile["complexity"],
            "readiness_score": document_profile["readiness_score"],
            "detected_capabilities": document_profile["detected_capabilities"],
            "recommended_payload": document_profile["recommended_payload"],
            "structured_text_recommended": document_profile["structured_text_recommended"],
            "element_type_counts": _top_counts(document_profile["element_type_counts"]),
            "relationship_type_counts": _top_counts(document_profile["relationship_type_counts"]),
            "metadata_fields": document_profile["metadata_fields"][:20],
            "evidence_profile": document_profile["evidence_profile"],
        },
    ))

    status = "complete" if elements and chunks else "warning"
    start = time.perf_counter()
    local_context = _build_local_context_artifact(
        filename=filename,
        domain=domain,
        chunks=chunks,
        relationships=relationships,
        local_graph_reasoning=local_graph_reasoning,
    )
    stages.append(_stage(
        "local_context_cache",
        "complete" if local_context.get("cached_chunk_count", 0) else "warning",
        start,
        counts={
            "chunks": local_context.get("chunk_count", 0),
            "cached_chunks": local_context.get("cached_chunk_count", 0),
            "graph_context_relationships": local_context.get("graph_context_relationship_count", 0),
            "graph_path_context_paths": local_context.get("graph_path_context_count", 0),
            "truncated": local_context.get("truncated", False),
        },
        sample_outputs={
            "evidence_cards": local_context.get("evidence_cards", [])[:sample_limit],
            "chunks": _sample_local_context_chunks(local_context, sample_limit),
            "graph_context": _sample_local_context_graph_context(local_context, sample_limit),
            "graph_paths": _sample_local_context_graph_paths(local_context, sample_limit),
            "filename": local_context.get("filename"),
            "domain": local_context.get("domain"),
        },
    ))

    start = time.perf_counter()
    interaction_suggestions = _build_interaction_suggestions(
        filename=filename,
        domain=domain,
        document_profile=document_profile,
        entities=entities,
        relationships=relationships,
        local_graph=local_graph,
        local_graph_reasoning=local_graph_reasoning,
    )
    interaction_handoff = _build_interaction_handoff(
        run_id=run_id,
        filename=filename,
        domain=domain,
        document_profile=document_profile,
        interaction_suggestions=interaction_suggestions,
        local_graph=local_graph,
        local_context=local_context,
        local_graph_reasoning=local_graph_reasoning,
    )
    stages.append(_stage(
        "graph_interaction_plan",
        "complete" if interaction_handoff["ready"] else "warning",
        start,
        counts={
            "suggested_questions": interaction_suggestions["question_count"],
            "graph_edges": interaction_suggestions["graph_edge_count"],
            "graph_nodes": interaction_suggestions["graph_node_count"],
            "bridge_entities": interaction_suggestions["bridge_entity_count"],
            "path_candidates": interaction_suggestions["path_candidate_count"],
            "handoff_lanes": len(interaction_handoff["lanes"]),
            "graph_anchors": interaction_handoff["counts"]["graph_anchors"],
            "capability_cards": interaction_handoff["counts"]["capability_cards"],
            "retrieval_rehearsals": interaction_handoff["counts"]["retrieval_rehearsals"],
            "rehearsed_chunk_matches": interaction_handoff["counts"]["rehearsed_chunk_matches"],
            "rehearsed_anchor_matches": interaction_handoff["counts"]["rehearsed_anchor_matches"],
        },
        sample_outputs={
            "suggested_questions": interaction_suggestions["suggested_questions"][:sample_limit],
            "starter_prompts": interaction_handoff["starter_prompts"][:sample_limit],
            "retrieval_rehearsals": interaction_handoff["retrieval_rehearsals"][:sample_limit],
            "handoff_lanes": interaction_handoff["lanes"],
            "capability_cards": interaction_handoff["capability_cards"][:sample_limit],
            "graph_anchors": interaction_handoff["graph_anchors"][:sample_limit],
            "recommended_mode": interaction_handoff["recommended_mode"],
            "answer_guidance": interaction_handoff["answer_guidance"],
            "detected_capabilities": interaction_suggestions["detected_capabilities"],
            "graph_edge_count": interaction_suggestions["graph_edge_count"],
            "graph_node_count": interaction_suggestions["graph_node_count"],
            "bridge_entity_count": interaction_suggestions["bridge_entity_count"],
            "path_candidate_count": interaction_suggestions["path_candidate_count"],
        },
    ))
    stages.append(structured_stage)

    result = {
        "run_id": run_id,
        "status": status,
        "created_at": created_at,
        "updated_at": _utc_now(),
        "document": {
            "filename": filename,
            "domain": domain,
            "content_type": content_type or "application/octet-stream",
            "extension": extension,
            "size_bytes": len(content),
            "sha256": sha256,
        },
        "options": {
            "use_semantic_chunking": use_semantic_chunking,
            "build_knowledge_graph": build_knowledge_graph,
            "sample_limit": sample_limit,
            "submit_structured_text": submit_structured_text,
        },
        "summary": {
            "extraction_method": extraction_method,
            "document_complexity": document_profile["complexity"],
            "readiness_score": document_profile["readiness_score"],
            "detected_capabilities": document_profile["detected_capabilities"],
            "element_count": len(elements),
            "chunk_count": len(chunks),
            "entity_candidate_count": len(entities),
            "relationship_candidate_count": len(relationships),
            "interaction_question_count": interaction_suggestions["question_count"],
            "evidence_profile": document_profile["evidence_profile"],
            "local_graph": {
                "node_count": local_graph["node_count"],
                "edge_count": local_graph["edge_count"],
                "relationship_count": local_graph["relationship_count"],
                "truncated": local_graph["truncated"],
            },
            "local_context": {
                "chunk_count": local_context["chunk_count"],
                "cached_chunk_count": local_context["cached_chunk_count"],
                "graph_context_relationship_count": local_context["graph_context_relationship_count"],
                "graph_path_context_count": local_context["graph_path_context_count"],
                "truncated": local_context["truncated"],
            },
            "local_graph_reasoning": {
                "bridge_entity_count": local_graph_reasoning["bridge_entity_count"],
                "path_candidate_count": local_graph_reasoning["path_candidate_count"],
                "truncated": local_graph_reasoning["truncated"],
            },
            "interaction_handoff": {
                "ready": interaction_handoff["ready"],
                "recommended_mode": interaction_handoff["recommended_mode"],
                "lane_count": len(interaction_handoff["lanes"]),
                "graph_anchor_count": interaction_handoff["counts"]["graph_anchors"],
                "capability_card_count": interaction_handoff["counts"]["capability_cards"],
                "starter_prompt_count": interaction_handoff["counts"]["starter_prompts"],
                "retrieval_rehearsal_count": interaction_handoff["counts"]["retrieval_rehearsals"],
                "rehearsed_chunk_match_count": interaction_handoff["counts"]["rehearsed_chunk_matches"],
                "rehearsed_anchor_match_count": interaction_handoff["counts"]["rehearsed_anchor_matches"],
                "citation_scopes": interaction_handoff["citation_scopes"],
            },
            "warnings": warnings,
        },
        "document_profile": document_profile,
        "interaction_suggestions": interaction_suggestions,
        "interaction_handoff": interaction_handoff,
        "local_graph": local_graph,
        "local_context": local_context,
        "local_graph_reasoning": local_graph_reasoning,
        "stages": stages,
        "samples": {
            "elements": _sample_elements(elements, sample_limit),
            "chunks": _sample_chunks(chunks, sample_limit),
            "entities": entities[:sample_limit],
            "relationships": _preview_relationship_samples(relationships, sample_limit),
            "relationship_highlights": _sample_relationships(relationships, sample_limit),
            "relationship_evidence_cards": _relationship_evidence_cards(relationships, sample_limit),
        },
        "structured_ingest": {
            "filename": structured_filename,
            "content_type": "text/markdown",
            "char_count": len(structured_text),
            "estimated_tokens": max(1, round(len(structured_text) / 4)) if structured_text else 0,
            "uses_structured_text": use_structured_text,
            "truncated": structured_truncated,
        },
    }
    if include_internal_payload:
        result["_structured_ingest_payload"] = {
            "filename": structured_filename,
            "content": structured_text,
            "content_type": "text/markdown",
            "uses_structured_text": use_structured_text,
        }
    return result
