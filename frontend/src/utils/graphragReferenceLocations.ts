export interface GraphRAGReferenceSourceLocation {
  label: string;
  value: string;
  category?: string;
}

const REFERENCE_SOURCE_LOCATION_FIELDS: Array<{
  key: string;
  label: string;
  category: string;
}> = [
  { key: "evidence_family", label: "Evidence type", category: "evidence" },
  { key: "pipeline_evidence_family", label: "Evidence type", category: "evidence" },
  { key: "evidence_count", label: "Evidence count", category: "evidence" },
  { key: "pipeline_evidence_count", label: "Evidence count", category: "evidence" },
  { key: "filename", label: "File", category: "document" },
  { key: "local_graph_filename", label: "File", category: "document" },
  { key: "page", label: "Page", category: "page" },
  { key: "slide", label: "Slide", category: "slide" },
  { key: "slide_title", label: "Slide title", category: "slide" },
  { key: "slide_visibility", label: "Slide visibility", category: "slide" },
  { key: "slide_part", label: "Slide part", category: "slide" },
  { key: "sheet", label: "Sheet", category: "sheet" },
  { key: "sheet_name", label: "Sheet", category: "sheet" },
  { key: "sheet_id", label: "Sheet id", category: "sheet" },
  { key: "sheet_visibility", label: "Sheet visibility", category: "sheet" },
  { key: "sheet_part", label: "Sheet part", category: "sheet" },
  { key: "table_name", label: "Table", category: "table" },
  { key: "table_display_name", label: "Table", category: "table" },
  { key: "table", label: "Table", category: "table" },
  { key: "table_range", label: "Table range", category: "table" },
  { key: "cell", label: "Cell", category: "table" },
  { key: "formula_cell", label: "Formula cell", category: "table" },
  { key: "table_cell_reference", label: "Cell", category: "table" },
  { key: "table_cell_text_anchor", label: "Cell text", category: "table" },
  { key: "row", label: "Row", category: "table" },
  { key: "column", label: "Column", category: "table" },
  { key: "chart_label", label: "Chart", category: "chart" },
  { key: "chart_title", label: "Chart title", category: "chart" },
  { key: "chart_part", label: "Chart", category: "chart" },
  { key: "chart_source_part", label: "Chart source", category: "chart" },
  { key: "chart_type", label: "Chart type", category: "chart" },
  { key: "chart_point_label", label: "Chart point", category: "chart" },
  { key: "chart_series_label", label: "Chart series", category: "chart" },
  { key: "series_name", label: "Series", category: "chart" },
  { key: "category", label: "Category", category: "chart" },
  { key: "value", label: "Value", category: "chart" },
  { key: "chart_axis_title", label: "Chart axis", category: "chart" },
  { key: "chart_axis_type", label: "Chart axis type", category: "chart" },
  { key: "chart_axis_number_format", label: "Chart axis format", category: "chart" },
  { key: "chart_label_fields", label: "Chart labels", category: "chart" },
  { key: "chart_label_position", label: "Chart label position", category: "chart" },
  { key: "chart_trendline_name", label: "Trendline", category: "chart" },
  { key: "chart_trendline_type", label: "Trendline type", category: "chart" },
  { key: "pivot_table", label: "Pivot", category: "table" },
  { key: "pivot_table_name", label: "Pivot", category: "table" },
  { key: "pivot_table_range", label: "Pivot range", category: "table" },
  { key: "pivot_table_source", label: "Pivot source", category: "table" },
  { key: "pivot_table_source_name", label: "Pivot source", category: "table" },
  { key: "pivot_table_source_ref", label: "Pivot source range", category: "table" },
  { key: "pivot_subtotal", label: "Pivot subtotal", category: "table" },
  { key: "form_kind", label: "Form", category: "form" },
  { key: "field_type", label: "Field", category: "form" },
  { key: "label", label: "Label", category: "form" },
  { key: "form_field_checked", label: "Checked", category: "form" },
  { key: "form_field_selected_option", label: "Selected", category: "form" },
  { key: "form_field_options", label: "Options", category: "form" },
  { key: "form_field_option_values", label: "Option values", category: "form" },
  { key: "form_field_date_format", label: "Date format", category: "form" },
  { key: "form_field_date_full_value", label: "Date value", category: "form" },
  { key: "form_field_calendar", label: "Calendar", category: "form" },
  { key: "control_name", label: "Control", category: "form" },
  { key: "control_part", label: "Control part", category: "form" },
  { key: "form_field_alias", label: "Field alias", category: "form" },
  { key: "form_field_tag", label: "Field tag", category: "form" },
  { key: "control_id", label: "Control", category: "form" },
  { key: "group_label", label: "Group", category: "layout" },
  { key: "group_id", label: "Group id", category: "layout" },
  { key: "group_child_labels", label: "Group items", category: "layout" },
  { key: "group_child_count", label: "Group size", category: "layout" },
  { key: "parent_group_label", label: "Parent group", category: "layout" },
  { key: "parent_group_id", label: "Parent group id", category: "layout" },
  { key: "grouped_visual_label", label: "Grouped visual", category: "layout" },
  { key: "shape", label: "Shape", category: "layout" },
  { key: "shape_id", label: "Shape", category: "layout" },
  { key: "shape_title", label: "Shape title", category: "layout" },
  { key: "shape_description", label: "Shape note", category: "layout" },
  { key: "text_box_kind", label: "Text box type", category: "layout" },
  { key: "text_box_part", label: "Text box part", category: "layout" },
  { key: "source_shape_id", label: "Source shape", category: "layout" },
  { key: "target_shape_id", label: "Target shape", category: "layout" },
  { key: "connector_id", label: "Connector", category: "layout" },
  { key: "connector_name", label: "Connector", category: "layout" },
  { key: "connector_title", label: "Connector title", category: "layout" },
  { key: "connector_description", label: "Connector note", category: "layout" },
  { key: "source_label", label: "Source label", category: "layout" },
  { key: "target_label", label: "Target label", category: "layout" },
  { key: "diagram_part", label: "Diagram", category: "diagram" },
  { key: "diagram_node_id", label: "Diagram node", category: "diagram" },
  { key: "diagram_connection_type", label: "Diagram edge", category: "diagram" },
  { key: "diagram_source", label: "Diagram source", category: "diagram" },
  { key: "diagram_target", label: "Diagram target", category: "diagram" },
  { key: "source_diagram_node_id", label: "Source diagram node", category: "diagram" },
  { key: "target_diagram_node_id", label: "Target diagram node", category: "diagram" },
  { key: "markdown_code_language", label: "Code language", category: "code" },
  { key: "markdown_code_block_index", label: "Code block", category: "code" },
  { key: "section", label: "Section", category: "document" },
  { key: "comment_id", label: "Comment", category: "comment" },
  { key: "comment_author", label: "Comment author", category: "comment" },
  { key: "author", label: "Comment author", category: "comment" },
  { key: "comment_anchor", label: "Comment anchor", category: "comment" },
  { key: "comment_anchor_text", label: "Comment anchor", category: "comment" },
  { key: "comment_kind", label: "Comment kind", category: "comment" },
  { key: "notes_part", label: "Notes", category: "notes" },
  { key: "notes_relationship_id", label: "Notes link", category: "notes" },
  { key: "emphasis_term", label: "Emphasis", category: "style" },
  { key: "emphasis_styles", label: "Styles", category: "style" },
  { key: "style_summary", label: "Style", category: "style" },
  { key: "image_part", label: "Image", category: "media" },
  { key: "ocr_source", label: "OCR", category: "media" },
  { key: "media_part", label: "Media", category: "media" },
  { key: "media_kind", label: "Media kind", category: "media" },
  { key: "media_mime", label: "Media type", category: "media" },
  { key: "embedded_file_filename", label: "Embedded file", category: "media" },
  { key: "embedded_file_name", label: "Embedded file", category: "media" },
  { key: "embedded_file_mime", label: "Embedded type", category: "media" },
  { key: "archive_entry_path", label: "Archive entry", category: "archive" },
  { key: "email_subject", label: "Subject", category: "message" },
  { key: "email_message_id", label: "Message id", category: "message" },
  { key: "email_from", label: "From", category: "message" },
  { key: "email_to", label: "To", category: "message" },
  { key: "email_cc", label: "Cc", category: "message" },
  { key: "email_attachment_filename", label: "Attachment", category: "message" },
  { key: "email_attachment_mime", label: "Attachment type", category: "message" },
  { key: "legacy_office_format", label: "Legacy format", category: "document" },
  { key: "legacy_office_stream_path", label: "Legacy stream", category: "document" },
  { key: "section_label", label: "Section", category: "document" },
  { key: "location", label: "Location", category: "document" },
  { key: "layout_zone", label: "Zone", category: "layout" },
  { key: "layout_bbox", label: "Region", category: "layout" },
];

const referenceLocationValue = (value: unknown): string => {
  if (typeof value === "string") return value.trim();
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (Array.isArray(value)) {
    return value
      .map(referenceLocationValue)
      .filter(Boolean)
      .slice(0, 4)
      .join(", ");
  }
  return "";
};

const formatReferenceLocationFieldValue = (key: string, value: unknown): string => {
  const text = referenceLocationValue(value);
  if (!text) return "";

  if (key === "evidence_family" || key === "pipeline_evidence_family") {
    return text.replace(/[_-]+/g, " ");
  }
  if (key === "evidence_count" || key === "pipeline_evidence_count") {
    const count = Number(text);
    return Number.isFinite(count) && count > 0 ? String(count) : "";
  }

  return text;
};

export const buildGraphRAGReferenceSourceLocations = (
  ...records: Array<Record<string, unknown> | undefined>
): GraphRAGReferenceSourceLocation[] => {
  const seen = new Set<string>();
  const locations: GraphRAGReferenceSourceLocation[] = [];

  const addLocation = (
    label: string,
    value: unknown,
    category: string,
  ): void => {
    const text = referenceLocationValue(value);
    if (!text) return;
    const dedupeKey = `${label.toLowerCase()}::${text.toLowerCase()}`;
    if (seen.has(dedupeKey)) return;
    seen.add(dedupeKey);
    locations.push({ label, value: text, category });
  };

  records.forEach((record) => {
    if (!record) return;
    const lineStart = referenceLocationValue(record.markdown_start_line);
    const lineEnd = referenceLocationValue(record.markdown_end_line);
    if (lineStart || lineEnd) {
      addLocation(
        lineStart && lineEnd && lineStart !== lineEnd ? "Lines" : "Line",
        lineStart && lineEnd && lineStart !== lineEnd
          ? `${lineStart}-${lineEnd}`
          : lineStart || lineEnd,
        "code",
      );
    }
    const layoutPercent = [
      referenceLocationValue(record.layout_left_pct),
      referenceLocationValue(record.layout_top_pct),
      referenceLocationValue(record.layout_width_pct),
      referenceLocationValue(record.layout_height_pct),
    ];
    if (layoutPercent.every(Boolean)) {
      addLocation("Region", layoutPercent.join(", "), "layout");
    }
    REFERENCE_SOURCE_LOCATION_FIELDS.forEach(({ key, label, category }) => {
      addLocation(label, formatReferenceLocationFieldValue(key, record[key]), category);
    });
  });

  return locations.slice(0, 24);
};
