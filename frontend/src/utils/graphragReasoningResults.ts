type JsonRecord = Record<string, unknown>;

export interface NormalizedReasoningHop {
  hop: number;
  entity: string;
  relation: string;
  source?: string;
  target?: string;
  confidence?: number;
  evidence?: string;
  provenance?: string;
  metadata?: JsonRecord;
  text: string;
}

export interface NormalizedReasoningPath {
  text: string;
  nodes: string[];
  hops?: number;
  score?: number;
  pathType?: string;
  relationships: NormalizedReasoningRelationship[];
}

export interface NormalizedReasoningRelationship {
  text: string;
  source?: string;
  relation?: string;
  target?: string;
  score?: number;
  weight?: number;
  evidence?: string;
  provenance?: string;
  metadata?: JsonRecord;
}

export interface NormalizedReasoningEvidence {
  content: string;
  source?: string;
  score?: number;
  sourceType?: string;
  metadata?: JsonRecord;
}

export interface NormalizedGraphRAGReasoningResult {
  reasoningSource: string;
  reasoningTask?: string;
  reasoningMethod?: string;
  warnings: string[];
  hops: NormalizedReasoningHop[];
  paths: NormalizedReasoningPath[];
  relationships: NormalizedReasoningRelationship[];
  evidence: NormalizedReasoningEvidence[];
  bridgeEntities: JsonRecord[];
  localPipeline?: JsonRecord;
  graphOverview?: JsonRecord;
}

const isRecord = (value: unknown): value is JsonRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const asArray = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

const asString = (value: unknown, fallback = ''): string =>
  typeof value === 'string' && value.trim()
    ? value.trim()
    : typeof value === 'number' && Number.isFinite(value)
      ? String(value)
      : fallback;

const asNumber = (value: unknown): number | undefined => {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
};

const uniqueStrings = (values: unknown[]): string[] => {
  const seen = new Set<string>();
  const unique: string[] = [];
  values.forEach(value => {
    const text = asString(value);
    if (!text || seen.has(text)) return;
    seen.add(text);
    unique.push(text);
  });
  return unique;
};

const localPayload = (record: JsonRecord): JsonRecord => (
  isRecord(record.local_pipeline_reasoning) ? record.local_pipeline_reasoning : {}
);

const relationshipText = (relationship: JsonRecord): string => {
  const source = asString(relationship.source);
  const relation = asString(relationship.relation, 'related_to');
  const target = asString(relationship.target);
  const evidence = asString(relationship.evidence);
  const base = [source, relation ? `--[${relation}]-->` : '-->', target]
    .filter(Boolean)
    .join(' ');
  return evidence ? `${base}: ${evidence}` : base;
};

const normalizeRelationship = (value: unknown): NormalizedReasoningRelationship | null => {
  if (!isRecord(value)) return null;
  const metadata = isRecord(value.metadata) ? value.metadata : undefined;
  const text = asString(value.text) || relationshipText(value);
  if (!text.trim()) return null;

  return {
    text,
    source: asString(value.source) || undefined,
    relation: asString(value.relation) || undefined,
    target: asString(value.target) || undefined,
    score: asNumber(value.score),
    weight: asNumber(value.weight),
    evidence: asString(value.evidence) || undefined,
    provenance: asString(value.provenance) || asString(metadata?.provenance) || undefined,
    metadata,
  };
};

const normalizeHop = (value: unknown, index: number): NormalizedReasoningHop | null => {
  if (!isRecord(value)) return null;
  const metadata = isRecord(value.metadata) ? value.metadata : undefined;
  const source = asString(value.source);
  const target = asString(value.target);
  const relation = asString(value.relation, 'related_to');
  const entity = asString(value.entity, target || source || `Hop ${index + 1}`);
  const text = source && target
    ? `${source} --[${relation}]--> ${target}`
    : `${entity} (${relation})`;

  return {
    hop: asNumber(value.hop) ?? index + 1,
    entity,
    relation,
    source: source || undefined,
    target: target || undefined,
    confidence: asNumber(value.confidence),
    evidence: asString(value.evidence) || undefined,
    provenance: asString(value.provenance) || asString(metadata?.provenance) || undefined,
    metadata,
    text,
  };
};

const normalizePath = (value: unknown): NormalizedReasoningPath | null => {
  if (Array.isArray(value)) {
    const nodes = value.map(item => asString(item)).filter(Boolean);
    if (nodes.length === 0) return null;
    return {
      text: nodes.join(' -> '),
      nodes,
      hops: Math.max(0, nodes.length - 1),
      relationships: [],
    };
  }

  if (!isRecord(value)) return null;
  const nodes = asArray(value.nodes).map(item => asString(item)).filter(Boolean);
  const relationships = asArray(value.relationships)
    .map(normalizeRelationship)
    .filter((relationship): relationship is NormalizedReasoningRelationship => Boolean(relationship));
  const text = asString(value.text) || (nodes.length > 0 ? nodes.join(' -> ') : '');
  if (!text) return null;

  return {
    text,
    nodes,
    hops: asNumber(value.hops),
    score: asNumber(value.score),
    pathType: asString(value.path_type) || asString(value.pathType) || undefined,
    relationships,
  };
};

const normalizeEvidence = (value: unknown): NormalizedReasoningEvidence | null => {
  if (!isRecord(value)) {
    const content = asString(value);
    return content ? { content } : null;
  }
  const metadata = isRecord(value.metadata) ? value.metadata : undefined;
  const content = (
    asString(value.content)
    || asString(value.text)
    || asString(value.explanation)
    || asString(value.source)
  );
  if (!content) return null;

  return {
    content,
    source: asString(value.source) || undefined,
    score: asNumber(value.score),
    sourceType: asString(value.source_type) || asString(value.sourceType) || undefined,
    metadata,
  };
};

const dedupeByText = <T extends { text?: string; content?: string; source?: string }>(items: T[]): T[] => {
  const seen = new Set<string>();
  const deduped: T[] = [];
  items.forEach(item => {
    const key = `${item.source || ''}|${item.text || item.content || ''}`.toLowerCase();
    if (!key.trim() || seen.has(key)) return;
    seen.add(key);
    deduped.push(item);
  });
  return deduped;
};

export const formatGraphRAGReasoningLocation = (metadata?: JsonRecord): string => {
  if (!metadata) return '';
  const tableCellWidthInches = asString(metadata.table_cell_width_inches);
  const tableCellWidthValue = asString(metadata.table_cell_width_value);
  const tableCellWidthType = asString(metadata.table_cell_width_type);
  const tableCellWidth = tableCellWidthInches
    ? `width ${tableCellWidthInches}in${tableCellWidthType ? ` ${tableCellWidthType}` : ''}`
    : tableCellWidthValue
      ? `width ${tableCellWidthValue}${tableCellWidthType ? ` ${tableCellWidthType}` : ''}`
      : '';
  const formChecked = typeof metadata.form_field_checked === 'boolean'
    ? metadata.form_field_checked ? 'checked' : 'unchecked'
    : asString(metadata.form_field_checked);
  const formOptions = asArray(metadata.form_field_options)
    .map(option => asString(option))
    .filter(Boolean)
    .slice(0, 4)
    .join(', ');
  const emphasisStyles = asArray(metadata.emphasis_styles)
    .map(style => asString(style))
    .filter(Boolean)
    .slice(0, 4)
    .join(', ');
  const bulletLevels = asArray(metadata.bullet_levels)
    .map(level => asString(level))
    .filter(Boolean)
    .slice(0, 6)
    .join(', ');
  const notesPart = asString(metadata.notes_part) || asString(metadata.note_part);
  const commentAuthor = (
    asString(metadata.comment_author)
    || asString(metadata.author)
    || asString(metadata.annotation_author)
    || asString(metadata.revision_author)
  );
  const commentAnchor = (
    asString(metadata.comment_anchor)
    || asString(metadata.comment_anchor_text)
    || asString(metadata.comment_anchor_cell)
    || asString(metadata.comment_anchor_section)
    || asString(metadata.note_anchor_text)
  );
  const diagramParent = asString(metadata.parent_label) || asString(metadata.parent_diagram_node_id);
  const diagramSource = (
    asString(metadata.diagram_source)
    || asString(metadata.source_label)
    || asString(metadata.source_diagram_node_id)
  );
  const diagramTarget = (
    asString(metadata.diagram_target)
    || asString(metadata.target_label)
    || asString(metadata.target_diagram_node_id)
  );
  const diagramEdge = diagramSource && diagramTarget
    ? `diagram edge ${diagramSource} to ${diagramTarget}`
    : '';
  const emailSubject = asString(metadata.email_subject);
  const emailMessageId = asString(metadata.email_message_id);
  const emailMessageLabel = emailSubject || emailMessageId;
  const emailSender = asString(metadata.email_from) || asString(metadata.email_sender);
  const emailTo = asString(metadata.email_to);
  const emailCc = asString(metadata.email_cc);
  const emailBcc = asString(metadata.email_bcc);
  const emailReplyTo = asString(metadata.email_reply_to);
  const emailDate = asString(metadata.email_date_iso) || asString(metadata.email_date);
  const emailReferences = asArray(metadata.email_references)
    .map(reference => asString(reference))
    .filter(Boolean)
    .slice(0, 3)
    .join(', ') || asString(metadata.email_references);
  const hasEmailBodyPart = (
    metadata.email_part_index !== undefined
    || Boolean(asString(metadata.email_part_kind))
    || Boolean(asString(metadata.email_content_type))
  );
  const emailBodyPart = hasEmailBodyPart
    ? [
        'message body',
        metadata.email_part_index !== undefined ? `part ${asString(metadata.email_part_index)}` : '',
        asString(metadata.email_part_kind),
        asString(metadata.email_content_type),
      ].filter(Boolean).join(' ')
    : '';
  const hasEmailAttachment = (
    Boolean(asString(metadata.email_attachment_filename))
    || metadata.email_attachment_index !== undefined
  );
  const emailAttachment = hasEmailAttachment
    ? [
        'email attachment',
        asString(metadata.email_attachment_filename),
        asString(metadata.email_attachment_mime),
        metadata.email_attachment_index !== undefined ? `index ${asString(metadata.email_attachment_index)}` : '',
      ].filter(Boolean).join(' ')
    : '';
  const parts = [
    asString(metadata.local_graph_filename),
    emailMessageLabel ? `message ${emailMessageLabel}` : '',
    emailSender ? `from ${emailSender}` : '',
    emailTo ? `to ${emailTo}` : '',
    emailCc ? `cc ${emailCc}` : '',
    emailBcc ? `bcc ${emailBcc}` : '',
    emailDate ? `sent ${emailDate}` : '',
    emailMessageId && emailSubject ? `message id ${emailMessageId}` : '',
    asString(metadata.email_header) ? `email header ${asString(metadata.email_header)}` : '',
    emailBodyPart,
    emailAttachment,
    emailReplyTo ? `reply-to ${emailReplyTo}` : '',
    asString(metadata.email_in_reply_to) ? `in reply to ${asString(metadata.email_in_reply_to)}` : '',
    emailReferences ? `references ${emailReferences}` : '',
    metadata.slide !== undefined
      ? `slide ${asString(metadata.slide)}${asString(metadata.slide_title) ? `: ${asString(metadata.slide_title)}` : ''}`
      : asString(metadata.slide_title),
    metadata.page !== undefined ? `page ${asString(metadata.page)}` : '',
    asString(metadata.sheet_name) || asString(metadata.sheet),
    asString(metadata.table_name) || asString(metadata.table),
    asString(metadata.table_cell_reference),
    asString(metadata.table_cell_visual_summary),
    asString(metadata.table_cell_style_name) ? `style ${asString(metadata.table_cell_style_name)}` : '',
    asString(metadata.table_cell_fill_color) ? `fill ${asString(metadata.table_cell_fill_color)}` : '',
    asString(metadata.table_cell_shading_pattern) ? `shading ${asString(metadata.table_cell_shading_pattern)}` : '',
    asString(metadata.table_cell_vertical_align) ? `align ${asString(metadata.table_cell_vertical_align)}` : '',
    asString(metadata.table_cell_text_direction) ? `direction ${asString(metadata.table_cell_text_direction)}` : '',
    tableCellWidth,
    asString(metadata.label) ? `field ${asString(metadata.label)}` : '',
    asString(metadata.field_type) ? `type ${asString(metadata.field_type)}` : '',
    asString(metadata.value) ? `value ${asString(metadata.value)}` : '',
    formChecked ? `checked ${formChecked}` : '',
    asString(metadata.form_field_selected_option) ? `selected ${asString(metadata.form_field_selected_option)}` : '',
    formOptions ? `options ${formOptions}` : '',
    asString(metadata.form_field_date_format) ? `date ${asString(metadata.form_field_date_format)}` : '',
    asString(metadata.chart_title),
    asString(metadata.chart_series_visual_summary),
    asString(metadata.chart_series_line_color) ? `line ${asString(metadata.chart_series_line_color)}` : '',
    asString(metadata.chart_point_visual_summary),
    asString(metadata.chart_trendline_type) ? `trendline ${asString(metadata.chart_trendline_type)}` : '',
    asString(metadata.chart_trendline_visual_summary),
    notesPart ? `speaker notes ${notesPart}` : '',
    asString(metadata.notes_relationship_id) ? `notes relationship ${asString(metadata.notes_relationship_id)}` : '',
    asString(metadata.comment_id) ? `comment ${asString(metadata.comment_id)}` : '',
    asString(metadata.comment_kind) ? `comment kind ${asString(metadata.comment_kind)}` : '',
    commentAuthor ? `author ${commentAuthor}` : '',
    commentAnchor ? `anchor ${commentAnchor}` : '',
    asString(metadata.revision_type) ? `revision ${asString(metadata.revision_type)}` : '',
    asString(metadata.revision_id) ? `revision id ${asString(metadata.revision_id)}` : '',
    asString(metadata.annotation_type) ? `annotation ${asString(metadata.annotation_type)}` : '',
    asString(metadata.annotation_subject) ? `annotation subject ${asString(metadata.annotation_subject)}` : '',
    asString(metadata.emphasis_term) ? `emphasis ${asString(metadata.emphasis_term)}` : '',
    emphasisStyles ? `styles ${emphasisStyles}` : '',
    bulletLevels ? `bullet levels ${bulletLevels}` : '',
    asString(metadata.style_summary) ? `style summary ${asString(metadata.style_summary)}` : '',
    asString(metadata.media_kind) ? `media ${asString(metadata.media_kind)}` : '',
    asString(metadata.media_part) ? `media part ${asString(metadata.media_part)}` : '',
    asString(metadata.media_mime) ? `media type ${asString(metadata.media_mime)}` : '',
    metadata.media_size_bytes !== undefined ? `media size ${asString(metadata.media_size_bytes)} bytes` : '',
    asString(metadata.diagram_part) ? `diagram part ${asString(metadata.diagram_part)}` : '',
    asString(metadata.diagram_node_id) ? `diagram node ${asString(metadata.diagram_node_id)}` : '',
    asString(metadata.diagram_node_type) ? `diagram node type ${asString(metadata.diagram_node_type)}` : '',
    metadata.diagram_order !== undefined ? `diagram order ${asString(metadata.diagram_order)}` : '',
    metadata.diagram_level !== undefined ? `diagram level ${asString(metadata.diagram_level)}` : '',
    diagramParent ? `diagram parent ${diagramParent}` : '',
    diagramEdge,
    asString(metadata.diagram_connection_type) ? `diagram connection ${asString(metadata.diagram_connection_type)}` : '',
    asString(metadata.connector_title) || asString(metadata.connector),
    asString(metadata.layout_bbox),
    asString(metadata.layout_zone),
    asString(metadata.provenance),
  ];
  return uniqueStrings(parts).join(' | ');
};

export const normalizeGraphRAGReasoningResult = (result: unknown): NormalizedGraphRAGReasoningResult => {
  const record = isRecord(result) ? result : {};
  const local = localPayload(record);
  const localPipeline = isRecord(record.local_pipeline)
    ? record.local_pipeline
    : isRecord(local.local_pipeline)
      ? local.local_pipeline
      : undefined;
  const graphOverview = isRecord(record.graph_overview)
    ? record.graph_overview
    : isRecord(local.graph_overview)
      ? local.graph_overview
      : undefined;

  const hops = [
    ...asArray(record.reasoning_path),
    ...asArray(local.reasoning_path),
  ].map(normalizeHop).filter((hop): hop is NormalizedReasoningHop => Boolean(hop));

  const paths = dedupeByText([
    ...asArray(record.paths),
    ...asArray(record.reasoning_paths),
    ...asArray(local.paths),
    ...asArray(local.reasoning_paths),
  ].map(normalizePath).filter((path): path is NormalizedReasoningPath => Boolean(path)));

  const relationships = dedupeByText([
    ...asArray(record.direct_relationships),
    ...asArray(local.direct_relationships),
  ].map(normalizeRelationship).filter((relationship): relationship is NormalizedReasoningRelationship => Boolean(relationship)));

  const evidence = dedupeByText([
    ...asArray(record.sources),
    ...asArray(record.evidence),
    ...asArray(local.sources),
    ...asArray(local.evidence),
  ].map(normalizeEvidence).filter((item): item is NormalizedReasoningEvidence => Boolean(item)));

  return {
    reasoningSource: (
      asString(record.reasoning_source)
      || asString(local.reasoning_source)
      || (localPipeline ? 'local_pipeline' : 'remote_graph')
    ),
    reasoningTask: asString(record.reasoning_task) || asString(local.reasoning_task) || undefined,
    reasoningMethod: asString(record.reasoning_method) || asString(local.reasoning_method) || undefined,
    warnings: uniqueStrings([
      ...asArray(record.warnings),
      ...asArray(local.warnings),
    ]),
    hops,
    paths,
    relationships,
    evidence,
    bridgeEntities: [
      ...asArray(record.bridge_entities),
      ...asArray(local.bridge_entities),
    ].filter(isRecord),
    localPipeline,
    graphOverview,
  };
};
