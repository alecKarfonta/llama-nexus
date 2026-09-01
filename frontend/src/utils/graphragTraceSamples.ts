export type GraphRAGTraceSampleRole = 'evidence' | 'interaction' | 'operational' | 'debug';

export interface GraphRAGTraceSampleCandidate {
  key: string;
  record: Record<string, unknown>;
  index: number;
  score: number;
}

export interface GraphRAGTraceSampleGroup {
  key: string;
  count: number;
  role: GraphRAGTraceSampleRole;
  priority: number;
}

export interface GraphRAGTraceSampleLinkTarget {
  label: string;
  url: string;
}

const SAMPLE_GROUP_LABELS: Record<string, { singular: string; plural?: string }> = {
  capability_cards: { singular: 'capability', plural: 'capabilities' },
  chunk_evidence_cards: { singular: 'chunk' },
  chunks: { singular: 'chunk' },
  content_type: { singular: 'file type' },
  detected_capabilities: { singular: 'capability', plural: 'capabilities' },
  edges: { singular: 'relationship' },
  elements: { singular: 'text block' },
  embedding_evidence_cards: { singular: 'search match' },
  embedding_evidence_families: { singular: 'category', plural: 'categories' },
  embedding_sample_categories: { singular: 'category', plural: 'categories' },
  entities: { singular: 'entity', plural: 'entities' },
  evidence_families: { singular: 'category', plural: 'categories' },
  extension: { singular: 'extension' },
  filename: { singular: 'file name' },
  graph_anchors: { singular: 'anchor' },
  graph_context: { singular: 'graph context' },
  graph_paths: { singular: 'path' },
  handoff_lanes: { singular: 'chat lane' },
  nodes: { singular: 'entity', plural: 'entities' },
  path_candidates: { singular: 'path' },
  relationship_evidence_cards: { singular: 'relationship' },
  relationship_sample_categories: { singular: 'category', plural: 'categories' },
  relationships: { singular: 'relationship' },
  retrieval_rehearsals: { singular: 'chat preview' },
  starter_prompts: { singular: 'question' },
  suggested_questions: { singular: 'question' },
  text: { singular: 'text sample' },
};

const OPERATIONAL_SAMPLE_KEYS = new Set([
  'content_type',
  'detected_capabilities',
  'extension',
  'filename',
  'mime_type',
  'readiness_score',
]);

const EVIDENCE_KEY_PRIORITIES: Record<string, number> = {
  relationship_evidence_cards: 150,
  relationships: 145,
  edges: 140,
  graph_context: 140,
  graph_paths: 134,
  path_candidates: 132,
  bridge_entities: 124,
  chunk_evidence_cards: 118,
  embedding_evidence_cards: 112,
  chunks: 108,
  elements: 104,
  text: 100,
  graph_anchors: 92,
  retrieval_rehearsals: 90,
  suggested_questions: 86,
  starter_prompts: 84,
  capability_cards: 78,
  handoff_lanes: 76,
  nodes: 70,
  representative_embedding_inputs: 48,
  embedding_inputs: 42,
  endpoint_attempts: 24,
  selected_endpoint: 22,
};

const INTERACTION_KEYS = new Set([
  'graph_anchors',
  'retrieval_rehearsals',
  'suggested_questions',
  'starter_prompts',
  'capability_cards',
  'handoff_lanes',
]);

const OPERATIONAL_KEY_PATTERN =
  /(?:endpoint|health|status_code|target_url|payload|service_status|submitted|deployment|fallback)/i;

const DEBUG_KEY_PATTERN =
  /(?:sha256|raw|json|xml|rels|embedding_inputs|representative_embedding_inputs|endpoint_attempts|selected_endpoint)/i;

const INTERNAL_ARTIFACT_PATTERN =
  /(?:^|[/"'\\\s])(?:ppt|word|xl|docProps|_rels)\/|(?:chartWorkbook|oleObject|embeddings\/|media\/image|\.rels\b|\.xml\b|\.bin\b|vmlDrawing|theme\/theme)/i;

const EXTERNAL_URL_PATTERN =
  /\b(?:https?:\/\/|www\.)\S+/i;

const URL_VALUE_PATTERN =
  /^(?:https?:\/\/|www\.)\S+$/i;

const DOCUMENT_LINK_RELATION_PATTERN =
  /^(?:links?_to|contains?_link|has_link|references?_url)$/i;

const DOCUMENT_CONTAINER_PATTERN =
  /(?:contains?_(?:document|slide|page|chart|table|image|shape|part)|has_(?:slide|page|chart|table|image|shape|part)|part_of|document_container|source_artifact)/i;

const DOMAIN_RELATION_PATTERN =
  /\b(?:drives?|supports?|blocks?|references?|depends?|influences?|causes?|contains?|maps?|links?|requires?|approves?|rejects?|summarizes?|compares?|forecasts?|trends?|signals?|explains?)\b/i;

const DIRECT_EVIDENCE_RELATION_PATTERN =
  /\b(?:approves?|blocks?|depends?|drives?|gates?|influences?|requires?|supports?)\b/i;

const GROUPED_VISUAL_EVIDENCE_PATTERN =
  /\b(?:contains[_\s-]?shape[_\s-]?group|groups[_\s-]?visual|grouped visual|shape group)\b/i;

const CONNECTOR_EVIDENCE_PATTERN =
  /\b(?:arrow|connector|connects?|flow|gates?)\b/i;

const FILELIKE_NODE_PATTERN =
  /\b(?:source document|[\w.-]+\.(?:csv|docx?|eml|html?|md|pdf|pptx?|txt|xlsx?|zip))\b/i;

const RICH_DOCUMENT_EVIDENCE_PATTERN =
  /\b(?:chart|table|figure|slide|page|section|row|column|cell|form|checkbox|signature|heading|styled|bold|italic|highlight|speaker note|footnote|caption|paragraph|chunk)\b/i;

const LOW_VALUE_OPERATIONAL_BOILERPLATE_PATTERN =
  /\b(?:unauthenticated requests?|hf[_\s-]?token|hf hub|hugging\s*face hub|authentication token|rate limits?)\b/i;

const DOCUMENT_EVIDENCE_SIGNAL_PATTERN =
  /\b(?:approval|approved|blocker|dependency|decision|finding|forecast|gate|launch|metric|model|owner|phase|quant|requirement|result|revenue|risk|throughput|validation|workflow)\b/i;

const WEAK_RELATION_KEY_PATTERN =
  /^(?:appears?_with|co_occurs(?:_with)?|mentioned?_with|mentions?|related_to)$/i;

const LOW_VALUE_NODE_LABELS = new Set([
  'hf hub',
  'hf token',
  'hf_token',
  'hub',
  'please',
  'token',
  'warning',
  'you',
]);

const LOW_VALUE_MATCH_TERMS = new Set([
  'co occurs with',
  'co_occurs_with',
  'graph',
  'hub',
  'relationship',
  'related to',
  'related_to',
]);

const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};

const asString = (value: unknown): string =>
  typeof value === 'string' ? value.trim() : '';

const stringifyValue = (value: unknown): string => {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (Array.isArray(value)) {
    return value.map(stringifyValue).filter(Boolean).slice(0, 8).join(' ');
  }
  try {
    return JSON.stringify(value);
  } catch {
    return '';
  }
};

export const graphRAGTraceSampleCount = (value: unknown): number => {
  if (Array.isArray(value)) return value.length;
  if (value && typeof value === 'object') return Object.keys(asRecord(value)).length > 0 ? 1 : 0;
  return stringifyValue(value).trim() ? 1 : 0;
};

export const graphRAGTraceSampleGroupRole = (key: string): GraphRAGTraceSampleRole => {
  if (OPERATIONAL_SAMPLE_KEYS.has(key)) return 'operational';
  if (DEBUG_KEY_PATTERN.test(key)) return 'debug';
  if (OPERATIONAL_KEY_PATTERN.test(key)) return 'operational';
  if (INTERACTION_KEYS.has(key)) return 'interaction';
  return 'evidence';
};

export const graphRAGTraceSampleGroupPriority = (key: string): number => {
  const role = graphRAGTraceSampleGroupRole(key);
  const base = EVIDENCE_KEY_PRIORITIES[key] ?? (role === 'evidence' ? 64 : role === 'interaction' ? 56 : role === 'operational' ? 32 : 10);
  return base;
};

const fallbackSampleGroupNoun = (key: string): string =>
  key
    .replace(/[_-]+/g, ' ')
    .replace(/\b(?:evidence cards?|sample outputs?|sample categories?)\b/gi, '')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase() || 'item';

const pluralSampleGroupNoun = (noun: string): string => {
  if (noun.endsWith('y') && !/[aeiou]y$/i.test(noun)) return `${noun.slice(0, -1)}ies`;
  if (noun.endsWith('s')) return noun;
  return `${noun}s`;
};

export const graphRAGTraceSampleGroupDisplayLabel = (
  key: string,
  count: number,
): string => {
  const noun = SAMPLE_GROUP_LABELS[key] || { singular: fallbackSampleGroupNoun(key) };
  const label = count === 1
    ? noun.singular
    : noun.plural || pluralSampleGroupNoun(noun.singular);
  return `${count.toLocaleString()} ${label}`;
};

const textForRecord = (record: Record<string, unknown>): string =>
  [
    record.title,
    record.label,
    record.source,
    record.relation,
    record.type,
    record.target,
    record.evidence,
    record.evidence_preview,
    record.reason,
    record.answer_hint,
    record.kind,
    record.focus,
    record.prompt,
    record.text,
    record.content,
    record.nodes,
    record.relationships,
    record.evidence_families,
    record.provenance,
    record.provenance_summary,
    record.location,
    record.metadata,
  ].map(stringifyValue).filter(Boolean).join(' ');

const relationKey = (value: unknown): string =>
  asString(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');

const metadataValue = (
  record: Record<string, unknown>,
  key: string,
): string =>
  asString(record[key]) ||
  asString(asRecord(record.metadata)[key]) ||
  asString(asRecord(record.provenance)[key]);

export const graphRAGTraceSampleExternalUrl = (
  record: Record<string, unknown>,
): string => {
  for (const key of ['uri', 'href', 'url', 'target_url', 'source_url']) {
    const value = metadataValue(record, key);
    if (URL_VALUE_PATTERN.test(value)) return value;
  }

  for (const value of [record.target, record.source, record.evidence, record.evidence_preview]) {
    const text = asString(value);
    const match = text.match(EXTERNAL_URL_PATTERN);
    if (match) return match[0].replace(/[),.;\]]+$/, '');
  }

  return '';
};

export const graphRAGTraceSampleLinkTarget = (
  record: Record<string, unknown>,
): GraphRAGTraceSampleLinkTarget | null => {
  const url = graphRAGTraceSampleExternalUrl(record);
  if (!url) return null;

  const labelCandidates = [
    metadataValue(record, 'label'),
    asString(record.label),
    asString(record.source),
    asString(record.title).split(/\s+--\[/)[0],
  ];
  const label = labelCandidates.find(candidate => candidate && !URL_VALUE_PATTERN.test(candidate)) || 'External link';
  return { label, url };
};

const recordHasDocumentLinkEvidence = (
  key: string,
  record: Record<string, unknown>,
  relationKeys: string[],
  relationships: Record<string, unknown>[],
): boolean => {
  if (graphRAGTraceSampleGroupRole(key) !== 'evidence') return false;

  const metadata = asRecord(record.metadata);
  const provenance = asRecord(record.provenance);
  const evidenceFamilies = [
    ...String(record.evidence_families || '').split(/[\s,|]+/),
    ...String(metadata.evidence_families || '').split(/[\s,|]+/),
  ].map(item => item.toLowerCase());
  const hasLinkMetadata =
    Boolean(metadataValue(record, 'uri')) ||
    Boolean(metadataValue(record, 'href')) ||
    Boolean(metadataValue(record, 'link_kind')) ||
    Boolean(metadataValue(record, 'relationship_id')) && (
      asString(metadata.element_type).toLowerCase() === 'link' ||
      asString(provenance.element_type).toLowerCase() === 'link'
    );
  const hasLinkRelation = relationKeys.some(item => DOCUMENT_LINK_RELATION_PATTERN.test(item));
  const nestedLinkEvidence = relationships.some(item => {
    const nestedRelation = relationKey(item.relation || item.label || item.type);
    const nestedMetadata = asRecord(item.metadata);
    return (
      DOCUMENT_LINK_RELATION_PATTERN.test(nestedRelation) ||
      Boolean(nestedMetadata.uri) ||
      Boolean(nestedMetadata.href) ||
      asString(nestedMetadata.element_type).toLowerCase() === 'link'
    ) && Boolean(graphRAGTraceSampleExternalUrl(item));
  });

  return Boolean(graphRAGTraceSampleExternalUrl(record)) && (
    hasLinkMetadata ||
    hasLinkRelation ||
    evidenceFamilies.includes('link') ||
    nestedLinkEvidence
  );
};

const relationshipRecords = (record: Record<string, unknown>): Record<string, unknown>[] =>
  Array.isArray(record.relationships)
    ? record.relationships
        .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object' && !Array.isArray(item))
    : [];

const compactDisplayText = (value: string, limit = 180): string => {
  const text = value
    .replace(/\s+/g, ' ')
    .replace(/\s*;\s*(?:prompt|answer_hint|question)\s*[:=]\s*[^;.]+/gi, '')
    .trim();
  return text.length > limit ? `${text.slice(0, Math.max(1, limit - 1)).trim()}...` : text;
};

const splitTraceSentences = (value: string): string[] =>
  value.match(/.+?(?:[.!?](?=\s|$)|$)/g)?.map(sentence => sentence.trim()).filter(Boolean) || [];

const normalizedNodeLabel = (value: unknown): string =>
  stringifyValue(value)
    .toLowerCase()
    .replace(/\bhf[_\s-]?token\b/g, 'hf token')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();

const isLowValueNodeLabel = (value: unknown): boolean => {
  const text = normalizedNodeLabel(value);
  if (!text) return false;
  if (LOW_VALUE_NODE_LABELS.has(text)) return true;
  return text.length <= 3 && /^(?:hub|you)$/i.test(text);
};

const isLowValueMatchedTerm = (value: unknown): boolean => {
  const text = normalizedNodeLabel(value);
  if (!text) return false;
  return (
    isLowValueNodeLabel(value) ||
    LOW_VALUE_MATCH_TERMS.has(text) ||
    WEAK_RELATION_KEY_PATTERN.test(relationKey(value))
  );
};

const stripLowValueOperationalBoilerplate = (value: string): string => {
  const text = value
    .replace(/(?:^|\s)[=_-]{6,}(?=\s|$)/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  if (!text) return '';

  const sentenceMatches = splitTraceSentences(text);
  const keptSentences = sentenceMatches
    .filter(sentence =>
      sentence &&
      (!LOW_VALUE_OPERATIONAL_BOILERPLATE_PATTERN.test(sentence) ||
        DOCUMENT_EVIDENCE_SIGNAL_PATTERN.test(sentence)),
    );
  const stripped = keptSentences.join(' ').replace(/\s+/g, ' ').trim();
  const candidate = stripped || text;
  const firstEvidenceSignal = candidate.search(DOCUMENT_EVIDENCE_SIGNAL_PATTERN);

  if (
    firstEvidenceSignal > 0 &&
    LOW_VALUE_OPERATIONAL_BOILERPLATE_PATTERN.test(candidate.slice(0, firstEvidenceSignal))
  ) {
    return candidate.slice(firstEvidenceSignal).replace(/^[^a-z0-9]+/i, '').trim();
  }

  return candidate;
};

export const cleanGraphRAGTraceSampleText = (
  value: string,
  limit = 180,
): string => {
  const cleaned = stripLowValueOperationalBoilerplate(value);
  if (!cleaned) return '';

  if (cleaned.length > limit) {
    const sentences = splitTraceSentences(cleaned);
    const evidenceSentence = sentences
      .map(sentence => sentence.trim())
      .find(sentence => DOCUMENT_EVIDENCE_SIGNAL_PATTERN.test(sentence));
    if (evidenceSentence) {
      return compactDisplayText(evidenceSentence, limit);
    }
  }

  return compactDisplayText(cleaned, limit);
};

export const isLowValueGraphRAGTraceSampleText = (value: string): boolean => {
  const text = value.replace(/\s+/g, ' ').trim();
  if (!text) return false;

  const cleaned = stripLowValueOperationalBoilerplate(text);
  if (
    LOW_VALUE_OPERATIONAL_BOILERPLATE_PATTERN.test(text) &&
    !DOCUMENT_EVIDENCE_SIGNAL_PATTERN.test(cleaned)
  ) {
    return true;
  }

  const normalized = normalizedNodeLabel(cleaned);
  const lowValueMentions = [...LOW_VALUE_NODE_LABELS].filter(label =>
    new RegExp(`\\b${label.replace(/\s+/g, '\\s+')}\\b`, 'i').test(normalized),
  ).length;
  const relationLike = /\b(?:appears with|co occurs|co_occurs|related to|through|to)\b|->/i.test(cleaned);

  return relationLike && lowValueMentions >= 2 && !DOCUMENT_EVIDENCE_SIGNAL_PATTERN.test(cleaned);
};

const graphRelationDisplayLabel = (value: unknown): string => {
  const normalized = relationKey(value);
  if (!normalized) return 'relates to';
  if (normalized === 'links_to' || normalized === 'link_to') return 'links to';
  if (normalized === 'contains_shape_group') return 'contains group';
  if (normalized === 'groups_visual') return 'groups';
  if (normalized === 'contains_image') return 'contains image';
  if (normalized === 'contains_link') return 'contains link';
  if (normalized === 'co_occurs_with') return 'appears with';
  return normalized.replace(/_/g, ' ');
};

const compactNodeLabel = (value: unknown, fallback = ''): string => {
  const text = stringifyValue(value).replace(/\s+/g, ' ').trim();
  if (!text) return fallback;
  return text.length > 56 ? `${text.slice(0, 55).trim()}...` : text;
};

const relationshipDisplaySummary = (relationship: Record<string, unknown>): string => {
  const source = compactNodeLabel(relationship.source, 'Source');
  const target = compactNodeLabel(relationship.target, 'Target');
  const relation = graphRelationDisplayLabel(relationship.relation || relationship.label || relationship.type);
  return `${source} ${relation} ${target}`;
};

export const graphRAGTraceSampleEvidenceSummary = (
  key: string,
  record: Record<string, unknown>,
  limit = 180,
): string => {
  if (key === 'path_candidates' || key === 'graph_paths') {
    const relationships = relationshipRecords(record);
    if (relationships.length > 0) {
      return compactDisplayText(
        relationships
          .slice(0, 2)
          .map(relationshipDisplaySummary)
          .join(' · '),
        limit,
      );
    }

    const nodes = Array.isArray(record.nodes)
      ? record.nodes.map((node) => compactNodeLabel(node)).filter(Boolean)
      : [];
    if (nodes.length > 1) {
      const hopCount = Math.max(1, nodes.length - 1);
      return compactDisplayText(`${hopCount} hop path: ${nodes.slice(0, 4).join(' -> ')}`, limit);
    }
  }

  if (key === 'retrieval_rehearsals') {
    const matchedTerms = Array.isArray(record.matched_terms)
      ? record.matched_terms.map((term) => compactNodeLabel(term)).filter(Boolean)
      : [];
    if (matchedTerms.length > 0) {
      return compactDisplayText(`Matched ${matchedTerms.slice(0, 4).join(', ')}`, limit);
    }

    const topAnchors = Array.isArray(record.top_anchors)
      ? record.top_anchors.map((anchor) => compactNodeLabel(anchor)).filter(Boolean)
      : [];
    if (topAnchors.length > 0) {
      return compactDisplayText(`Top anchors: ${topAnchors.slice(0, 3).join(', ')}`, limit);
    }
  }

  return '';
};

const recordHasFilelikeNode = (record: Record<string, unknown>, text: string): boolean =>
  FILELIKE_NODE_PATTERN.test(text) ||
  [record.source, record.target, record.title, record.label]
    .map(stringifyValue)
    .some(value => FILELIKE_NODE_PATTERN.test(value));

export const graphRAGTraceSampleRecordScore = (
  key: string,
  record: Record<string, unknown>,
): number => {
  const relation = asString(record.relation) || asString(record.label) || asString(record.type);
  const source = asString(record.source);
  const target = asString(record.target);
  const title = asString(record.title) || asString(record.label);
  const evidence = asString(record.evidence) || asString(record.evidence_preview) || asString(record.reason);
  const text = textForRecord(record);
  const relationships = relationshipRecords(record);
  const relationKeys = [relation, ...relationships.map(item => asString(item.relation) || asString(item.label) || asString(item.type))]
    .map(relationKey)
    .filter(Boolean);
  const structuralRelationshipCount = relationKeys.filter(item =>
    DOCUMENT_CONTAINER_PATTERN.test(item) || item.startsWith('contains_') || item.startsWith('has_'),
  ).length;
  const hasDocumentLinkEvidence = recordHasDocumentLinkEvidence(key, record, relationKeys, relationships);
  const hasDirectRelation =
    relationKeys.some(item => DIRECT_EVIDENCE_RELATION_PATTERN.test(item)) ||
    DIRECT_EVIDENCE_RELATION_PATTERN.test(text);
  const hasGroupedVisualEvidence =
    relationKeys.some(item => GROUPED_VISUAL_EVIDENCE_PATTERN.test(item)) ||
    GROUPED_VISUAL_EVIDENCE_PATTERN.test(text);
  const hasConnectorEvidence =
    relationKeys.some(item => CONNECTOR_EVIDENCE_PATTERN.test(item)) ||
    CONNECTOR_EVIDENCE_PATTERN.test(text);
  const hasFilelikeNode = recordHasFilelikeNode(record, text);
  const chartHeavyPath =
    Array.isArray(record.nodes) &&
    /\b(?:chart|metric|revenue plan|trendline|q[1-4]|series)\b/i.test(text) &&
    hasFilelikeNode;
  const structuralOnlyPath =
    relationships.length > 0 &&
    structuralRelationshipCount === relationships.length &&
    !hasDirectRelation &&
    !hasGroupedVisualEvidence &&
    !hasConnectorEvidence;
  const interactionQuestionKey = ['retrieval_rehearsals', 'starter_prompts', 'suggested_questions'].includes(key);
  const graphAnchorPath = key === 'graph_anchors' && asString(record.kind).toLowerCase() === 'path';
  const weakInteractionReference = /\b(?:co[_\s-]?occurs(?:[_\s-]?with)?|related[_\s-]?to|mentioned?[_\s-]?with)\b/i.test(
    [
      record.id,
      record.prompt_id,
      record.prompt,
      record.label,
    ].map(stringifyValue).filter(Boolean).join(' '),
  );
  const weakRelation =
    relationKeys.some(item => WEAK_RELATION_KEY_PATTERN.test(item)) ||
    weakInteractionReference;
  const lowValueEndpointCount = [source, target, title].filter(isLowValueNodeLabel).length;
  const lowValueText = isLowValueGraphRAGTraceSampleText(text);
  const matchedTerms = Array.isArray(record.matched_terms) ? record.matched_terms : [];
  const lowValueMatchedTermCount = matchedTerms.filter(isLowValueMatchedTerm).length;
  const lowValueMatchedTermRatio =
    matchedTerms.length > 0 ? lowValueMatchedTermCount / matchedTerms.length : 0;
  const lowValueMatchedTerms =
    matchedTerms.length > 0 &&
    (
      matchedTerms.every(isLowValueMatchedTerm) ||
      (
        lowValueMatchedTermRatio >= 0.5 &&
        LOW_VALUE_OPERATIONAL_BOILERPLATE_PATTERN.test(text)
      )
    );
  const pathNodes = Array.isArray(record.nodes) ? record.nodes : [];
  const lowValuePathNodeCount = pathNodes.filter(isLowValueNodeLabel).length;
  const lowValuePath =
    (key === 'graph_anchors' || key === 'path_candidates' || key === 'graph_paths') &&
    lowValuePathNodeCount >= 2 &&
    weakRelation;
  const lowValueBridge =
    (key === 'graph_anchors' || key === 'bridge_entities') &&
    asString(record.kind).toLowerCase() === 'bridge' &&
    isLowValueNodeLabel(record.title || record.name || record.label);
  let score = graphRAGTraceSampleGroupPriority(key);

  if (source && target) score += 36;
  if (source && target && relation) score += 24;
  if (relation && DOMAIN_RELATION_PATTERN.test(relation)) score += 22;
  if (hasDirectRelation) score += 48;
  if (hasGroupedVisualEvidence) score += 42;
  if (hasConnectorEvidence) score += 30;
  if (hasDocumentLinkEvidence) score += key === 'path_candidates' || key === 'graph_paths' ? 38 : 58;
  if (interactionQuestionKey && asString(record.kind).toLowerCase() === 'relationship') score += 84;
  if (interactionQuestionKey && /\b(?:audit|inspect|question|trace)\b/i.test(text)) score += 22;
  if (key === 'bridge_entities' && (hasDirectRelation || hasGroupedVisualEvidence || hasConnectorEvidence)) score += 44;
  if (DOMAIN_RELATION_PATTERN.test(title) || DOMAIN_RELATION_PATTERN.test(evidence)) score += 12;
  if (evidence) score += 18;
  if (record.source_locations || record.location || record.provenance || record.chunk_id) score += 12;
  if (RICH_DOCUMENT_EVIDENCE_PATTERN.test(text)) score += 10;
  if (DOCUMENT_EVIDENCE_SIGNAL_PATTERN.test(stripLowValueOperationalBoilerplate(text))) score += 16;
  if (Array.isArray(record.nodes) && record.nodes.length >= 3) score += 18;
  if (Array.isArray(record.relationships) && record.relationships.length > 0) score += 14;

  if (weakRelation) score -= 28;
  if (weakRelation && lowValueEndpointCount > 0) score -= 96;
  if (lowValueEndpointCount >= 2) score -= 64;
  if (lowValueText) score -= 84;
  if (lowValuePath) score -= 220;
  if (lowValueBridge) score -= 130;
  if (
    interactionQuestionKey &&
    weakRelation &&
    (
      lowValueEndpointCount > 0 ||
      lowValueMatchedTermCount >= 2 ||
      isLowValueGraphRAGTraceSampleText(`${asString(record.label)} ${asString(record.prompt_id)}`)
    )
  ) score -= 130;
  if (key === 'retrieval_rehearsals' && lowValueMatchedTerms) score -= 120;
  if (DOCUMENT_CONTAINER_PATTERN.test(relation) || DOCUMENT_CONTAINER_PATTERN.test(title)) score -= 46;
  if ((key === 'path_candidates' || graphAnchorPath) && hasFilelikeNode) score -= 48;
  if (chartHeavyPath) score -= 34;
  if (structuralOnlyPath) score -= 30;
  if (INTERNAL_ARTIFACT_PATTERN.test(text)) score -= 72;
  if (EXTERNAL_URL_PATTERN.test(text) && !hasDocumentLinkEvidence) score -= 44;
  if (/\b(?:api\/v\d|sha256|payload|status_code|health_url|target_url)\b/i.test(text)) score -= 26;
  if (!source && !target && !evidence && text.length < 18) score -= 16;

  return score;
};

const candidateRecords = (
  key: string,
  value: unknown,
): Array<Omit<GraphRAGTraceSampleCandidate, 'score'>> => {
  if (Array.isArray(value)) {
    return value
      .map((item, index) => {
        if (item && typeof item === 'object' && !Array.isArray(item)) {
          return { key, record: item as Record<string, unknown>, index };
        }
        const text = stringifyValue(item).trim();
        return text ? { key, record: { text }, index } : null;
      })
      .filter((item): item is Omit<GraphRAGTraceSampleCandidate, 'score'> => Boolean(item));
  }

  if (value && typeof value === 'object') {
    return [{ key, record: value as Record<string, unknown>, index: 0 }];
  }

  const text = stringifyValue(value).trim();
  return text ? [{ key, record: { text }, index: 0 }] : [];
};

export const rankGraphRAGTraceSampleRecords = (
  key: string,
  records: Record<string, unknown>[],
): GraphRAGTraceSampleCandidate[] =>
  records
    .map((record, index) => ({
      key,
      record,
      index,
      score: graphRAGTraceSampleRecordScore(key, record),
    }))
    .sort((left, right) => right.score - left.score || left.index - right.index);

export const collectGraphRAGTraceSampleCandidates = (
  sampleOutputs: Record<string, unknown>,
  keys: string[],
): GraphRAGTraceSampleCandidate[] =>
  keys
    .flatMap((key) =>
      candidateRecords(key, sampleOutputs[key]).map((candidate) => ({
        ...candidate,
        score: graphRAGTraceSampleRecordScore(candidate.key, candidate.record),
      })),
    )
    .sort((left, right) => right.score - left.score || keys.indexOf(left.key) - keys.indexOf(right.key) || left.index - right.index);

export const pickBestGraphRAGTraceSample = (
  sampleOutputs: Record<string, unknown>,
  keys: string[],
): GraphRAGTraceSampleCandidate | null =>
  collectGraphRAGTraceSampleCandidates(sampleOutputs, keys)[0] || null;

export const rankGraphRAGTraceSampleGroups = (
  sampleOutputs: Record<string, unknown>,
): GraphRAGTraceSampleGroup[] =>
  Object.entries(sampleOutputs)
    .map(([key, value]) => ({
      key,
      count: graphRAGTraceSampleCount(value),
      role: graphRAGTraceSampleGroupRole(key),
      priority: graphRAGTraceSampleGroupPriority(key),
    }))
    .filter((group) => group.count > 0)
    .sort((left, right) => {
      const roleDelta =
        Number(left.role === 'debug') - Number(right.role === 'debug') ||
        Number(left.role === 'operational') - Number(right.role === 'operational');
      if (roleDelta !== 0) return roleDelta;
      return right.priority - left.priority || right.count - left.count || left.key.localeCompare(right.key);
    });
