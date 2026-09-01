import { normalizeGraphRAGPipelineArtifactDownloads } from './graphragPipelineArtifacts';

type JsonRecord = Record<string, unknown>;

export interface GraphRAGTraceQuestion {
  id: string;
  label: string;
  prompt: string;
  kind: 'summary' | 'relationship' | 'path' | 'evidence' | 'bridge' | 'shared_entity';
  source?: string;
  target?: string;
  bridge?: string;
  relation?: string;
  focus?: string;
}

export interface GraphRAGTraceCapabilityCard {
  id: string;
  label: string;
  detail: string;
  count?: number;
}

export interface GraphRAGTraceChatContextRequest {
  query?: string;
  top_k?: number;
  domain?: string;
  pipeline_run_id?: string;
  pipeline_run_ids?: string[];
  include_graph?: boolean;
  use_local_pipeline_context?: boolean;
  include_local_pipeline?: boolean;
  rag_enabled?: boolean;
  rag_domains?: string[];
  rag_search_mode?: 'domain' | 'global';
  source?: string;
  target?: string;
  bridge?: string;
  relation?: string;
  pipeline_evidence_totals?: JsonRecord;
  pipeline_evidence_families?: string[];
  graph_overview_counts?: JsonRecord;
  response_mode?: 'chat' | 'compact' | 'full';
  context_text_max_chars?: number;
  response_chunk_content_chars?: number;
  response_graph_context_limit?: number;
  response_graph_path_limit?: number;
  response_evidence_card_limit?: number;
  response_relationship_limit?: number;
  response_path_limit?: number;
}

export interface GraphRAGTraceChatContextSettings {
  ragTopK?: number;
  ragSelectedDomains?: string[];
  ragSearchMode?: 'domain' | 'domains' | 'global';
  ragEnabled?: boolean;
  graphEnabled?: boolean;
}

export interface GraphRAGTraceChatHandoff {
  runId?: string;
  runIds?: string[];
  filename?: string;
  domain?: string;
  prompt?: string;
  displayPrompt?: string;
  contextQuery?: string;
  autoSubmit?: boolean;
  chatContextEndpoint?: string;
  chatContextRequest?: GraphRAGTraceChatContextRequest;
  suggestedQuestions?: GraphRAGTraceQuestion[];
  recommendedMode?: string;
  answerGuidance?: string;
  citationScopes?: string[];
  capabilityCards?: GraphRAGTraceCapabilityCard[];
  graphAnchorCount?: number;
  retrievalRehearsalCount?: number;
  evidenceTotals?: JsonRecord;
  evidenceFamilies?: string[];
  graphOverviewCounts?: JsonRecord;
}

const isRecord = (value: unknown): value is JsonRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const asArray = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

const asString = (value: unknown, fallback = ''): string =>
  typeof value === 'string' && value.trim() ? value.trim() : fallback;

const asNumber = (value: unknown, fallback = 0): number => {
  const parsed = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : Number.NaN;
  return Number.isFinite(parsed) ? parsed : fallback;
};

const asBoolean = (value: unknown): boolean | undefined =>
  typeof value === 'boolean' ? value : undefined;

const positiveNumber = (value: unknown): number | undefined => {
  const parsed = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : Number.NaN;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
};

const compact = (value: string, limit = 48): string => {
  const text = value.replace(/\s+/g, ' ').trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, Math.max(1, limit - 3)).trim()}...`;
};

const formatKey = (value: string): string =>
  value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, character => character.toUpperCase());

const normalizeKey = (value: string): string =>
  value.replace(/[^a-zA-Z0-9]+/g, '_').replace(/^_+|_+$/g, '').toLowerCase();

const getDocument = (result: JsonRecord): JsonRecord =>
  isRecord(result.document) ? result.document : {};

const getSummary = (result: JsonRecord): JsonRecord =>
  isRecord(result.summary) ? result.summary : {};

const getDocumentProfile = (result: JsonRecord): JsonRecord =>
  isRecord(result.document_profile) ? result.document_profile : {};

const getArtifactManifest = (result: JsonRecord): JsonRecord =>
  isRecord(result.artifacts) ? result.artifacts : {};

const asStringArray = (value: unknown): string[] =>
  asArray(value)
    .map(item => asString(item))
    .filter(Boolean);

const collectSampleOutputs = (result: JsonRecord): JsonRecord[] =>
  asArray(result.stages)
    .filter(isRecord)
    .map(stage => stage.sample_outputs)
    .filter(isRecord);

const getLocalGraph = (result: JsonRecord): JsonRecord =>
  isRecord(result.local_graph) ? result.local_graph : {};

const getLocalContext = (result: JsonRecord): JsonRecord =>
  isRecord(result.local_context) ? result.local_context : {};

const getLocalGraphReasoning = (result: JsonRecord): JsonRecord =>
  isRecord(result.local_graph_reasoning) ? result.local_graph_reasoning : {};

const collectCapabilities = (summary: JsonRecord, documentProfile: JsonRecord, sampleOutputs: JsonRecord[]): string[] => {
  const seen = new Set<string>();
  const capabilities: string[] = [];
  const add = (value: unknown) => {
    const text = asString(value);
    const key = text.toLowerCase();
    if (text && !seen.has(key)) {
      seen.add(key);
      capabilities.push(text);
    }
  };

  asStringArray(documentProfile.detected_capabilities).forEach(add);
  asStringArray(summary.detected_capabilities).forEach(add);
  for (const sample of sampleOutputs) {
    asStringArray(sample.detected_capabilities).forEach(add);
  }
  return capabilities;
};

const describeArtifactsForPrompt = (artifactManifest: unknown, limit = 5): string => {
  const artifacts = normalizeGraphRAGPipelineArtifactDownloads(artifactManifest, { includeManifest: false })
    .filter(artifact => artifact.kind !== 'full')
    .slice(0, limit);
  if (artifacts.length === 0) return '';
  return artifacts
    .map(artifact => {
      const endpointText = artifact.endpoint ? ` at ${artifact.endpoint}` : '';
      return `${artifact.shortLabel} (${artifact.downloadFilename}${endpointText})`;
    })
    .join(', ');
};

const artifactPromptText = (artifactManifest: unknown): string => {
  const artifactText = describeArtifactsForPrompt(artifactManifest);
  if (!artifactText) return '';
  return ` Exportable pipeline artifacts available: ${artifactText}. Use structured Markdown for source text, local graph JSON for nodes and edges, local context caches for retrieval chunks, and graph reasoning artifacts for bridge/path evidence when helpful.`;
};

export const buildGraphRAGTraceDisplayPrompt = (
  filename: string,
  prompt = '',
): string => {
  const fileLabel = compact(filename || 'document', 80);
  const text = prompt.replace(/\s+/g, ' ').trim();
  if (!text) return `Ask ${fileLabel}`;

  const isGeneratedTracePrompt =
    /^(?:use|using) the (?:recent )?graphrag pipeline trace for\b/i.test(text) ||
    /^(?:use|using) the recent document import for\b/i.test(text) ||
    /\bexportable pipeline artifacts available\b/i.test(text);
  if (!isGeneratedTracePrompt) return compact(text, 160);

  const relationshipMatch = text.match(
    /explain how "([^"]+)" connects to "([^"]+)"(?: through "([^"]+)")?/i,
  );
  if (relationshipMatch) {
    const relation = relationshipMatch[3]
      ? ` (${relationshipMatch[3]})`
      : '';
    return compact(
      `Trace ${relationshipMatch[1]} -> ${relationshipMatch[2]}${relation}`,
      160,
    );
  }

  const pathMatch = text.match(
    /(?:path connecting|graph path ")(?:"?([^"]+)"?\s*(?:and|->)\s*"?([^"]+)"?)/i,
  );
  if (pathMatch) {
    return compact(`Trace ${pathMatch[1]} -> ${pathMatch[2]}`, 160);
  }

  if (/\bsummarize\b/i.test(text)) return `Summarize ${fileLabel}`;
  if (/\bforms?\b|\bfields?\b/i.test(text)) return `Inspect form fields in ${fileLabel}`;
  if (/\bstyled text\b|\bemphasis\b|\bbold\b|\bitalic\b/i.test(text)) {
    return `Inspect emphasized text in ${fileLabel}`;
  }
  if (/\bcharts?\b|\btables?\b/i.test(text)) return `Explain charts and tables in ${fileLabel}`;
  if (/\bconnectors?\b|\barrows?\b|\bflow\b/i.test(text)) return `Trace connectors in ${fileLabel}`;
  if (/\bimages?\b|\bfigures?\b|\bdiagrams?\b/i.test(text)) return `Explain visuals in ${fileLabel}`;
  return `Summarize ${fileLabel}`;
};

const graphRAGPromptLooksGenerated = (prompt = ''): boolean => {
  const text = prompt.replace(/\s+/g, ' ').trim();
  if (!text) return false;
  return (
    /^(?:use|using) the (?:recent )?graphrag pipeline trace for\b/i.test(text) ||
    /^(?:use|using) the recent document import for\b/i.test(text) ||
    /^use the document context for this batch\b/i.test(text) ||
    /\bexportable pipeline artifacts available\b/i.test(text) ||
    /\breturn exactly 4 bullets\b/i.test(text)
  );
};

export const buildGraphRAGTraceContextQuery = (
  filename: string,
  prompt = '',
): string => {
  const fileLabel = compact(filename || 'document', 96);
  const text = prompt.replace(/\s+/g, ' ').trim();
  if (!text) {
    return `Extracted document evidence for ${fileLabel}: entities, semantic relationships, graph paths, source provenance, charts, tables, forms, styled text, connectors, and chunks.`;
  }
  if (!graphRAGPromptLooksGenerated(text)) return text;

  const relationshipMatch = text.match(
    /explain how "([^"]+)" connects to "([^"]+)"(?: through "([^"]+)")?/i,
  );
  if (relationshipMatch) {
    const [, source, target, relation] = relationshipMatch;
    const relationText = relation ? ` ${relation}` : '';
    return compact(
      `Document evidence for ${source}${relationText} ${target} in ${fileLabel}: semantic relationships, graph path hops, source provenance, and supporting chunks.`,
      360,
    );
  }

  const selectedRelationshipMatch = text.match(
    /"([^"]+)"\s*--\[(.+?)\]-->\s*"([^"]+)"/i,
  );
  if (selectedRelationshipMatch) {
    const [, source, relation, target] = selectedRelationshipMatch;
    return compact(
      `Document evidence for ${source} ${relation} ${target} in ${fileLabel}: source provenance, supporting chunks, and related graph context.`,
      360,
    );
  }

  const pathMatch = text.match(
    /(?:path connecting|graph path ")(?:"?([^"]+)"?\s*(?:and|->)\s*"?([^"]+)"?)/i,
  );
  if (pathMatch) {
    return compact(
      `Document evidence for the graph path from ${pathMatch[1]} to ${pathMatch[2]} in ${fileLabel}: bridge entities, path hops, source provenance, and supporting chunks.`,
      360,
    );
  }

  if (/^use the document context for this batch\b/i.test(text)) {
    return compact(
      `Compare extracted evidence across ${fileLabel}: shared entities, semantic relationships, graph paths, bridge entities, contradictions, unique source proof, and document provenance.`,
      360,
    );
  }

  if (/\bsummarize\b/i.test(text)) {
    return `Extracted document evidence for ${fileLabel}: entities, semantic relationships, graph paths, source provenance, charts, tables, forms, styled text, connectors, and chunks.`;
  }

  if (/\bforms?\b|\bfields?\b/i.test(text)) {
    return `Form field evidence in ${fileLabel}: field labels, values, source provenance, related entities, and supporting chunks.`;
  }
  if (/\bstyled text\b|\bemphasis\b|\bbold\b|\bitalic\b/i.test(text)) {
    return `Styled text evidence in ${fileLabel}: emphasized phrases, style metadata, source provenance, related entities, and supporting chunks.`;
  }
  if (/\bcharts?\b|\btables?\b/i.test(text)) {
    return `Chart and table evidence in ${fileLabel}: series, cells, values, source provenance, related entities, and supporting chunks.`;
  }
  if (/\bconnectors?\b|\barrows?\b|\bflow\b/i.test(text)) {
    return `Connector evidence in ${fileLabel}: source labels, target labels, relationship labels, source provenance, and supporting chunks.`;
  }
  if (/\bimages?\b|\bfigures?\b|\bdiagrams?\b/i.test(text)) {
    return `Visual evidence in ${fileLabel}: figures, diagrams, image labels, related entities, source provenance, and supporting chunks.`;
  }

  return `Extracted document evidence for ${fileLabel}: entities, semantic relationships, graph paths, source provenance, charts, tables, forms, styled text, connectors, and chunks.`;
};

export const buildGraphRAGTracePrompt = (
  filename: string,
  summary: JsonRecord = {},
  documentProfile: JsonRecord = {},
  sampleOutputs: JsonRecord[] = [],
  artifactManifest: unknown = undefined,
): string => {
  const capabilities = collectCapabilities(summary, documentProfile, sampleOutputs);
  const capabilityText = capabilities.length > 0
    ? ` It extracted ${capabilities.slice(0, 8).map(formatKey).join(', ')}.`
    : '';
  return [
    `Use the recent GraphRAG pipeline trace for ${filename} to summarize the key evidence, entities, relationships, and any meaningful graph paths.`,
    'Keep the answer bounded and complete: return exactly 4 bullets, under 180 words total, named Entities, Top relationships, Best path, and Source proof. Do not add a trailing paragraph, table, or exhaustive list.',
    'In Top relationships, include at most 4 direct graph relationships and prioritize gate/block/require/support/drive/reference edges before incidental containment or row-order summaries.',
    `Cite slide/page/table/chart/connector provenance when available.${capabilityText}${artifactPromptText(artifactManifest)}`,
  ].join(' ');
};

interface RelationshipCandidate {
  source: string;
  target: string;
  relation: string;
  evidence: string;
  metadata: JsonRecord;
}

const relationshipFromRecord = (record: JsonRecord): RelationshipCandidate | null => {
  const metadata = isRecord(record.metadata)
    ? record.metadata
    : isRecord(record.properties)
      ? record.properties
      : {};
  const source = asString(record.source);
  const target = asString(record.target);
  const relation = asString(record.relation) || asString(record.label) || asString(record.type) || 'related_to';
  if (!source || !target) return null;
  return {
    source,
    target,
    relation,
    evidence: asString(record.evidence) || asString(metadata.evidence),
    metadata,
  };
};

const relationshipPriority = (relationship: RelationshipCandidate): number => {
  const searchable = [
    relationship.source,
    relationship.target,
    relationship.relation,
    relationship.evidence,
    ...Object.values(relationship.metadata).map(value => String(value ?? '')),
  ].join(' ').toLowerCase();

  if (/(connector|arrow|line|flow|shape_connector)/.test(searchable)) return 5;
  if (/(chart|table|form|field|image|figure|diagram|speaker|notes|embedded)/.test(searchable)) return 4;
  if (!/^(has|contains)_/.test(relationship.relation)) return 3;
  if (relationship.evidence) return 2;
  return 1;
};

const collectRelationships = (result: JsonRecord, sampleOutputs: JsonRecord[]): RelationshipCandidate[] => {
  const relationships: RelationshipCandidate[] = [];
  const seen = new Set<string>();
  const add = (record: unknown) => {
    if (!isRecord(record)) return;
    const relationship = relationshipFromRecord(record);
    if (!relationship) return;
    const key = `${relationship.source.toLowerCase()}|${relationship.relation.toLowerCase()}|${relationship.target.toLowerCase()}`;
    if (seen.has(key)) return;
    seen.add(key);
    relationships.push(relationship);
  };

  for (const sample of sampleOutputs) {
    asArray(sample.relationships).forEach(add);
    asArray(sample.edges).forEach(add);
    asArray(sample.graph_context).forEach(add);
    for (const path of [...asArray(sample.path_candidates), ...asArray(sample.paths), ...asArray(sample.graph_paths)]) {
      if (!isRecord(path)) continue;
      asArray(path.relationships).forEach(add);
    }
    for (const bridge of asArray(sample.bridge_entities)) {
      if (!isRecord(bridge)) continue;
      asArray(bridge.relationships).forEach(add);
    }
  }

  const localGraph = getLocalGraph(result);
  asArray(localGraph.edges).forEach(add);

  const localGraphReasoning = getLocalGraphReasoning(result);
  for (const path of [...asArray(localGraphReasoning.path_candidates), ...asArray(localGraphReasoning.paths), ...asArray(localGraphReasoning.graph_paths)]) {
    if (!isRecord(path)) continue;
    asArray(path.relationships).forEach(add);
  }
  for (const bridge of asArray(localGraphReasoning.bridge_entities)) {
    if (!isRecord(bridge)) continue;
    asArray(bridge.relationships).forEach(add);
  }

  const localContext = getLocalContext(result);
  for (const chunk of asArray(localContext.chunks)) {
    if (!isRecord(chunk)) continue;
    asArray(chunk.graph_context).forEach(add);
  }

  return relationships.sort((a, b) => relationshipPriority(b) - relationshipPriority(a));
};

interface EntityCandidate {
  name: string;
  type: string;
}

interface PathCandidate {
  source: string;
  bridge: string;
  target: string;
  nodes: string[];
  text: string;
}

const pathFromRecord = (record: JsonRecord): PathCandidate | null => {
  const nodes = asArray(record.nodes)
    .map(item => asString(item))
    .filter(Boolean);
  const source = asString(record.source) || nodes[0] || '';
  const bridge = asString(record.bridge) || nodes[1] || '';
  const target = asString(record.target) || nodes[2] || '';
  if (!source || !bridge || !target) return null;
  return {
    source,
    bridge,
    target,
    nodes: nodes.length >= 3 ? nodes : [source, bridge, target],
    text: asString(record.text),
  };
};

const collectPathCandidates = (result: JsonRecord, sampleOutputs: JsonRecord[]): PathCandidate[] => {
  const paths: PathCandidate[] = [];
  const seen = new Set<string>();
  const add = (record: unknown) => {
    if (!isRecord(record)) return;
    const path = pathFromRecord(record);
    if (!path) return;
    const key = `${path.source.toLowerCase()}|${path.bridge.toLowerCase()}|${path.target.toLowerCase()}`;
    if (seen.has(key)) return;
    seen.add(key);
    paths.push(path);
  };

  for (const sample of sampleOutputs) {
    asArray(sample.path_candidates).forEach(add);
    asArray(sample.paths).forEach(add);
    asArray(sample.graph_paths).forEach(add);
  }

  const localGraphReasoning = getLocalGraphReasoning(result);
  asArray(localGraphReasoning.path_candidates).forEach(add);
  asArray(localGraphReasoning.paths).forEach(add);
  asArray(localGraphReasoning.graph_paths).forEach(add);

  return paths;
};

const collectEntities = (
  result: JsonRecord,
  sampleOutputs: JsonRecord[],
  relationships: RelationshipCandidate[] = [],
): EntityCandidate[] => {
  const entities: EntityCandidate[] = [];
  const seen = new Set<string>();
  const add = (nameValue: unknown, typeValue: unknown = 'entity') => {
    const name = asString(nameValue);
    if (!name) return;
    const key = name.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    entities.push({ name, type: asString(typeValue, 'entity') });
  };

  for (const sample of sampleOutputs) {
    for (const entity of asArray(sample.entities)) {
      if (!isRecord(entity)) continue;
      add(entity.name, entity.type);
    }
    for (const node of asArray(sample.nodes)) {
      if (!isRecord(node)) continue;
      add(node.name || node.label || node.id, node.type || node.kind);
    }
    for (const bridge of asArray(sample.bridge_entities)) {
      if (!isRecord(bridge)) continue;
      add(bridge.name || bridge.label || bridge.id, bridge.type || 'bridge_entity');
    }
    for (const path of [...asArray(sample.path_candidates), ...asArray(sample.paths), ...asArray(sample.graph_paths)]) {
      if (!isRecord(path)) continue;
      add(path.source, 'graph_node');
      add(path.bridge, 'bridge_entity');
      add(path.target, 'graph_node');
      asArray(path.nodes).forEach(node => add(node, 'graph_node'));
    }
  }

  const localGraph = getLocalGraph(result);
  for (const node of asArray(localGraph.nodes)) {
    if (!isRecord(node)) continue;
    add(node.name || node.label || node.id, node.type || node.kind);
  }

  const localGraphReasoning = getLocalGraphReasoning(result);
  for (const bridge of asArray(localGraphReasoning.bridge_entities)) {
    if (!isRecord(bridge)) continue;
    add(bridge.name || bridge.label || bridge.id, bridge.type || 'bridge_entity');
  }
  for (const path of [...asArray(localGraphReasoning.path_candidates), ...asArray(localGraphReasoning.paths), ...asArray(localGraphReasoning.graph_paths)]) {
    if (!isRecord(path)) continue;
    add(path.source, 'graph_node');
    add(path.bridge, 'bridge_entity');
    add(path.target, 'graph_node');
    asArray(path.nodes).forEach(node => add(node, 'graph_node'));
  }

  for (const relationship of relationships) {
    add(relationship.source, 'graph_node');
    add(relationship.target, 'graph_node');
  }

  return entities;
};

const collectEvidenceTerms = (result: JsonRecord, sampleOutputs: JsonRecord[], capabilities: string[]): Set<string> => {
  const evidence = new Set<string>();
  const addText = (value: unknown) => {
    const text = asString(value).toLowerCase();
    if (!text) return;
    for (const term of [
      'connector',
      'arrow',
      'archive',
      'attachment',
      'annotation',
      'chart',
      'comment',
      'table',
      'form',
      'field',
      'image',
      'figure',
      'diagram',
      'ocr',
      'revision',
      'markup',
      'speaker_notes',
      'note',
      'embedded_object',
      'embedded_objects',
      'styled_text',
      'emphasis',
      'graph_context',
      'relationship',
      'path',
      'bridge',
    ]) {
      if (text.includes(term)) evidence.add(term);
    }
  };

  capabilities.forEach(addText);
  for (const sample of sampleOutputs) {
    for (const element of asArray(sample.elements)) {
      if (!isRecord(element)) continue;
      addText(element.type);
      addText(element.text);
      if (isRecord(element.metadata)) {
        Object.values(element.metadata).forEach(addText);
      }
    }
    for (const edge of [...asArray(sample.edges), ...asArray(sample.graph_context)]) {
      if (!isRecord(edge)) continue;
      addText(edge.label || edge.relation || edge.type);
      addText(edge.evidence);
      if (isRecord(edge.metadata)) Object.values(edge.metadata).forEach(addText);
      if (isRecord(edge.properties)) Object.values(edge.properties).forEach(addText);
    }
    for (const bridge of asArray(sample.bridge_entities)) {
      if (!isRecord(bridge)) continue;
      addText('bridge');
      addText(bridge.name);
      addText(bridge.text);
      asArray(bridge.relationships).forEach(relationship => {
        if (isRecord(relationship)) Object.values(relationship).forEach(addText);
        else addText(relationship);
      });
    }
    for (const path of [...asArray(sample.path_candidates), ...asArray(sample.paths), ...asArray(sample.graph_paths)]) {
      if (!isRecord(path)) continue;
      addText('path');
      addText(path.source);
      addText(path.bridge);
      addText(path.target);
      addText(path.text);
      asArray(path.relationships).forEach(relationship => {
        if (isRecord(relationship)) Object.values(relationship).forEach(addText);
        else addText(relationship);
      });
    }
  }

  const localGraph = getLocalGraph(result);
  for (const edge of asArray(localGraph.edges)) {
    if (!isRecord(edge)) continue;
    addText(edge.label);
    addText(edge.type);
    addText(edge.evidence);
    if (isRecord(edge.metadata)) Object.values(edge.metadata).forEach(addText);
    if (isRecord(edge.properties)) Object.values(edge.properties).forEach(addText);
  }

  const localGraphReasoning = getLocalGraphReasoning(result);
  for (const bridge of asArray(localGraphReasoning.bridge_entities)) {
    if (!isRecord(bridge)) continue;
    addText('bridge');
    addText(bridge.name);
    addText(bridge.text);
    asArray(bridge.relationships).forEach(relationship => {
      if (isRecord(relationship)) Object.values(relationship).forEach(addText);
      else addText(relationship);
    });
  }
  for (const path of [...asArray(localGraphReasoning.path_candidates), ...asArray(localGraphReasoning.paths), ...asArray(localGraphReasoning.graph_paths)]) {
    if (!isRecord(path)) continue;
    addText('path');
    addText(path.source);
    addText(path.bridge);
    addText(path.target);
    addText(path.text);
    asArray(path.relationships).forEach(relationship => {
      if (isRecord(relationship)) Object.values(relationship).forEach(addText);
      else addText(relationship);
    });
  }

  const localContext = getLocalContext(result);
  for (const chunk of asArray(localContext.chunks)) {
    if (!isRecord(chunk)) continue;
    addText(chunk.content || chunk.text);
    const graphContextRelationships = asArray(chunk.graph_context);
    if (graphContextRelationships.length > 0) addText('graph_context');
    for (const relationship of graphContextRelationships) {
      if (!isRecord(relationship)) continue;
      Object.values(relationship).forEach(addText);
      if (isRecord(relationship.metadata)) Object.values(relationship.metadata).forEach(addText);
    }
  }
  return evidence;
};

const normalizeBackendQuestion = (question: JsonRecord, index: number): GraphRAGTraceQuestion | null => {
  const prompt = asString(question.prompt);
  const label = asString(question.label, prompt ? compact(prompt) : '');
  if (!prompt || !label) return null;
  const kind = asString(question.kind);
  return {
    id: asString(question.id, `backend-question-${index}`),
    label,
    prompt,
    kind: kind === 'relationship' || kind === 'path' || kind === 'evidence' || kind === 'summary'
      ? kind
      : 'evidence',
    source: asString(question.source) || undefined,
    target: asString(question.target) || undefined,
    bridge: asString(question.bridge) || undefined,
    relation: asString(question.relation) || undefined,
    focus: asString(question.focus) || undefined,
  };
};

const collectBackendQuestions = (result: JsonRecord): GraphRAGTraceQuestion[] => {
  const questions: GraphRAGTraceQuestion[] = [];
  const addQuestions = (value: unknown) => {
    for (const question of asArray(value)) {
      if (!isRecord(question)) continue;
      const normalized = normalizeBackendQuestion(question, questions.length);
      if (normalized) questions.push(normalized);
    }
  };

  const interactionSuggestions = isRecord(result.interaction_suggestions) ? result.interaction_suggestions : {};
  addQuestions(interactionSuggestions.suggested_questions);
  const interactionHandoff = isRecord(result.interaction_handoff) ? result.interaction_handoff : {};
  addQuestions(interactionHandoff.starter_prompts);

  for (const sample of collectSampleOutputs(result)) {
    addQuestions(sample.suggested_questions);
    addQuestions(sample.starter_prompts);
  }
  return questions;
};

const pushUniqueQuestion = (
  questions: GraphRAGTraceQuestion[],
  question: GraphRAGTraceQuestion,
  limit: number,
) => {
  if (questions.length >= limit) return;
  const key = question.prompt.toLowerCase();
  if (questions.some(existing => existing.prompt.toLowerCase() === key || existing.label.toLowerCase() === question.label.toLowerCase())) {
    return;
  }
  questions.push(question);
};

const relationshipPrompt = (filename: string, relationship: RelationshipCandidate): string => {
  const evidenceText = relationship.evidence
    ? ` The extracted evidence snippet was: "${compact(relationship.evidence, 220)}".`
    : '';
  return `Using the GraphRAG pipeline trace for ${filename}, explain how "${relationship.source}" connects to "${relationship.target}" through "${relationship.relation}". Find the strongest direct or multi-hop graph path, cite the trace evidence, and include any slide/page/table/chart/connector provenance.${evidenceText}`;
};

const pathPrompt = (filename: string, source: string, target: string): string =>
  `Using the GraphRAG pipeline trace for ${filename}, find the strongest graph path connecting "${source}" and "${target}". Explain each hop, identify bridge entities, and cite the exact extracted evidence and provenance for the path.`;

const pathCandidatePrompt = (filename: string, path: PathCandidate): string => {
  const pathText = path.text ? ` Candidate path: "${compact(path.text, 260)}".` : '';
  return `Using the GraphRAG pipeline trace for ${filename}, explain the two-hop graph path "${path.source}" -> "${path.bridge}" -> "${path.target}". Walk each relationship, explain why "${path.bridge}" is the bridge entity, cite the extracted evidence, and include slide/page/table/chart/connector provenance when available.${pathText}`;
};

const evidencePrompts = (filename: string, terms: Set<string>): Array<{ label: string; prompt: string }> => {
  const prompts: Array<{ label: string; prompt: string }> = [];

  if (terms.has('connector') || terms.has('arrow')) {
    prompts.push({
      label: 'Audit connectors',
      prompt: `Using the GraphRAG pipeline trace for ${filename}, inspect the extracted connectors, arrows, lines, and flow relationships. Explain what they connect, where they appear, and cite slide or shape provenance for each important link.`,
    });
  }
  if (terms.has('diagram')) {
    prompts.push({
      label: 'Audit diagrams',
      prompt: `Using the GraphRAG pipeline trace for ${filename}, inspect the extracted diagram and SmartArt nodes, levels, parent/child edges, and graph relationships. Explain which entities the diagram connects, where it appears, and cite diagram part plus slide provenance.`,
    });
  }
  if (terms.has('styled_text') || terms.has('emphasis')) {
    prompts.push({
      label: 'Inspect emphasized text',
      prompt: `Using the GraphRAG pipeline trace for ${filename}, inspect the extracted styled text, bold/italic/underlined terms, bullet hierarchy, and emphasis relationships. Explain what the emphasized terms highlight, which entities they connect to, and cite style plus slide/page/chunk provenance.`,
    });
  }
  if (terms.has('chart') || terms.has('table')) {
    prompts.push({
      label: 'Explain charts and tables',
      prompt: `Using the GraphRAG pipeline trace for ${filename}, explain the key chart and table evidence. Identify the entities or metrics they connect to, and cite chart/table/page/slide provenance from the extraction.`,
    });
  }
  if (terms.has('archive') || terms.has('attachment') || terms.has('embedded_object') || terms.has('embedded_objects')) {
    prompts.push({
      label: 'Trace source artifacts',
      prompt: `Using the GraphRAG pipeline trace for ${filename}, inspect embedded files, archive entries, attachments, and media artifacts. Explain which child documents or source artifacts were parsed, what graph facts they contributed, and cite filename/path, MIME type, checksum, and parent document provenance when available.`,
    });
  }
  if (terms.has('ocr')) {
    prompts.push({
      label: 'Review OCR evidence',
      prompt: `Using the GraphRAG pipeline trace for ${filename}, review OCR-derived evidence from scanned pages or images. Identify the extracted text, any graph relationships it created, confidence or quality warnings, and cite page/image/OCR provenance.`,
    });
  }
  if (terms.has('form') || terms.has('field')) {
    prompts.push({
      label: 'Inspect form fields',
      prompt: `Using the GraphRAG pipeline trace for ${filename}, inspect the extracted forms and fields. Explain what each field represents, how it relates to nearby document entities, and cite field/page provenance.`,
    });
  }
  if (terms.has('comment') || terms.has('annotation') || terms.has('revision') || terms.has('speaker_notes') || terms.has('note') || terms.has('markup')) {
    prompts.push({
      label: 'Review markup',
      prompt: `Using the GraphRAG pipeline trace for ${filename}, inspect comments, annotations, revisions, footnotes/endnotes, and speaker notes. Explain who authored the markup, what decision or risk it captures, how it connects to nearby document entities, and cite page/slide/comment/revision provenance.`,
    });
  }
  if (terms.has('image') || terms.has('figure') || terms.has('diagram')) {
    prompts.push({
      label: 'Explain visuals',
      prompt: `Using the GraphRAG pipeline trace for ${filename}, explain the extracted images, figures, and diagrams. Identify related entities, inferred relationships, and visual provenance from the trace.`,
    });
  }
  if (terms.has('graph_context') || terms.has('relationship') || terms.has('path') || terms.has('bridge')) {
    prompts.push({
      label: 'Inspect graph context',
      prompt: `Using the GraphRAG pipeline trace for ${filename}, inspect the graph context attached to embedding chunks. Explain which relationships should improve retrieval, which graph paths they support, and cite chunk plus document provenance.`,
    });
  }
  return prompts;
};

const artifactEvidencePrompts = (filename: string, artifactManifest: unknown): Array<{ label: string; prompt: string }> => {
  const artifactText = describeArtifactsForPrompt(artifactManifest, 6);
  if (!artifactText) return [];
  return [
    {
      label: 'Review run artifacts',
      prompt: `Using the GraphRAG pipeline trace for ${filename}, review the exportable run artifacts: ${artifactText}. Explain which artifact should be used for structured source text, local graph nodes and edges, retrieval chunk context, and bridge/path reasoning. Cite artifact filenames and endpoints when relevant.`,
    },
  ];
};

const normalizeTraceChatContextRequest = (
  value: unknown,
  {
    prompt,
    contextQuery,
    domain,
    runId,
  }: {
    prompt: string;
    contextQuery?: string;
    domain?: string;
    runId?: string;
  },
): GraphRAGTraceChatContextRequest | undefined => {
  const record = isRecord(value) ? value : {};
  const recordQuery = asString(record.query);
  const normalizedQuery = recordQuery && !graphRAGPromptLooksGenerated(recordQuery)
    ? recordQuery
    : asString(contextQuery, recordQuery || prompt);
  const pipelineRunIds = asStringArray(record.pipeline_run_ids);
  const runIds = pipelineRunIds.length > 0 ? pipelineRunIds : runId ? [runId] : [];
  const request: GraphRAGTraceChatContextRequest = {
    query: normalizedQuery || undefined,
    domain: asString(record.domain, domain || '') || undefined,
    pipeline_run_id: asString(record.pipeline_run_id) || undefined,
    pipeline_run_ids: runIds.length > 0 ? runIds : undefined,
    include_graph: asBoolean(record.include_graph) ?? true,
    use_local_pipeline_context: asBoolean(record.use_local_pipeline_context) ?? (runIds.length > 0 ? true : undefined),
    include_local_pipeline: asBoolean(record.include_local_pipeline) ?? (runIds.length > 0 ? true : undefined),
    rag_enabled: asBoolean(record.rag_enabled),
    rag_domains: asStringArray(record.rag_domains).length > 0 ? asStringArray(record.rag_domains) : undefined,
    rag_search_mode: record.rag_search_mode === 'global' ? 'global' : record.rag_search_mode === 'domain' ? 'domain' : undefined,
    source: asString(record.source) || undefined,
    target: asString(record.target) || undefined,
    bridge: asString(record.bridge) || undefined,
    relation: asString(record.relation) || undefined,
  };
  const responseMode = asString(record.response_mode).toLowerCase();
  if (responseMode === 'chat' || responseMode === 'compact' || responseMode === 'full') {
    request.response_mode = responseMode;
  }
  const topK = asNumber(record.top_k, Number.NaN);
  if (Number.isFinite(topK) && topK > 0) request.top_k = topK;
  for (const [sourceKey, targetKey] of [
    ['context_text_max_chars', 'context_text_max_chars'],
    ['response_chunk_content_chars', 'response_chunk_content_chars'],
    ['response_graph_context_limit', 'response_graph_context_limit'],
    ['response_graph_path_limit', 'response_graph_path_limit'],
    ['response_evidence_card_limit', 'response_evidence_card_limit'],
    ['response_relationship_limit', 'response_relationship_limit'],
    ['response_path_limit', 'response_path_limit'],
  ] as const) {
    const value = asNumber(record[sourceKey], Number.NaN);
    if (Number.isFinite(value) && value > 0) {
      request[targetKey] = value;
    }
  }
  return Object.values(request).some(value => value !== undefined) ? request : undefined;
};

const graphRAGChatSearchModeForRequest = (
  value: GraphRAGTraceChatContextRequest['rag_search_mode'] | GraphRAGTraceChatContextSettings['ragSearchMode'] | undefined,
): 'domain' | 'global' => {
  if (value === 'global') return 'global';
  return 'domain';
};

const graphRAGHandoffRunIds = (handoff?: GraphRAGTraceChatHandoff | null): string[] => {
  const ids = [
    ...(Array.isArray(handoff?.runIds) ? handoff?.runIds || [] : []),
    handoff?.runId || '',
  ];
  const seen = new Set<string>();
  return ids
    .map(id => asString(id))
    .filter((id) => {
      if (!id || seen.has(id)) return false;
      seen.add(id);
      return true;
    });
};

interface GraphRAGQueryTargetHint {
  source?: string;
  target?: string;
  bridge?: string;
  relation?: string;
}

const GRAPH_RAG_ANCHOR_STOP_WORDS = new Set([
  'a',
  'an',
  'and',
  'between',
  'bridge',
  'connect',
  'connected',
  'connection',
  'do',
  'does',
  'find',
  'how',
  'is',
  'link',
  'linked',
  'path',
  'relationship',
  'show',
  'the',
  'to',
  'what',
]);

const cleanGraphRAGQueryAnchor = (value: string): string =>
  value
    .replace(/\s+/g, ' ')
    .replace(/^[\s"'`([{]+|[\s"'`)\]},.?!:;]+$/g, '')
    .replace(/^(?:the|a|an|entity|node|metric|concept)\s+/i, '')
    .trim();

const normalizeGraphRAGAnchorText = (value: string): string =>
  cleanGraphRAGQueryAnchor(value)
    .toLowerCase()
    .replace(/&/g, ' and ')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();

const isUsefulGraphRAGAnchor = (value: string): boolean => {
  const normalized = normalizeGraphRAGAnchorText(value);
  if (normalized.length < 2) return false;
  const words = normalized.split(/\s+/).filter(Boolean);
  return words.length > 0 && words.some(word => !GRAPH_RAG_ANCHOR_STOP_WORDS.has(word));
};

const graphRAGHandoffQueryCandidates = (handoff?: GraphRAGTraceChatHandoff | null): string[] => {
  const candidates: string[] = [];
  const seen = new Set<string>();
  const add = (value: unknown) => {
    const candidate = asString(value);
    const normalized = normalizeGraphRAGAnchorText(candidate);
    if (!candidate || !normalized || seen.has(normalized) || !isUsefulGraphRAGAnchor(candidate)) return;
    seen.add(normalized);
    candidates.push(candidate);
  };

  add(handoff?.chatContextRequest?.source);
  add(handoff?.chatContextRequest?.target);
  add(handoff?.chatContextRequest?.bridge);
  for (const question of Array.isArray(handoff?.suggestedQuestions) ? handoff?.suggestedQuestions || [] : []) {
    add(question.source);
    add(question.target);
    add(question.bridge);
    add(question.focus);
  }
  return candidates;
};

const resolveGraphRAGAnchor = (value: string, candidates: string[]): string | undefined => {
  const cleaned = cleanGraphRAGQueryAnchor(value);
  const normalized = normalizeGraphRAGAnchorText(cleaned);
  if (!normalized) return undefined;
  const resolved = candidates
    .map(candidate => ({ candidate, normalized: normalizeGraphRAGAnchorText(candidate) }))
    .filter(candidate =>
      candidate.normalized &&
      (candidate.normalized === normalized ||
        ` ${normalized} `.includes(` ${candidate.normalized} `))
    )
    .sort((a, b) => b.normalized.length - a.normalized.length)[0]?.candidate;
  if (resolved) return resolved;
  return isUsefulGraphRAGAnchor(cleaned) ? cleaned : undefined;
};

const graphRAGMentionedCandidateAnchors = (query: string, candidates: string[]): string[] => {
  const normalizedQuery = ` ${normalizeGraphRAGAnchorText(query)} `;
  const spans = candidates
    .map(candidate => {
      const normalized = normalizeGraphRAGAnchorText(candidate);
      const phrase = ` ${normalized} `;
      const start = normalized ? normalizedQuery.indexOf(phrase) : -1;
      return {
        candidate,
        start,
        end: start + phrase.length,
        length: normalized.length,
      };
    })
    .filter(span => span.start >= 0)
    .sort((a, b) => b.length - a.length || a.start - b.start);

  const selected: typeof spans = [];
  for (const span of spans) {
    if (!selected.some(existing => span.start < existing.end && existing.start < span.end)) {
      selected.push(span);
    }
  }
  return selected.sort((a, b) => a.start - b.start).map(span => span.candidate);
};

const GRAPH_RAG_RELATION_CUE_PATTERN =
  /\b(?:affect|affects|block|blocks|bridge|cause|causes|connect|connected|connection|depend(?:s)?\s+on|drive|drives|enable|enables|gate|gates|influence|influences|link|linked|map(?:s)?\s+to|own|owns|path|reference|references|relate|related|relationship|route|support|supports|unblock|unblocks|use|uses)\b/i;

const relationCueForGraphRAGQuery = (query: string): boolean =>
  GRAPH_RAG_RELATION_CUE_PATTERN.test(query);

const GRAPH_RAG_RELATION_HINT_BETWEEN_ANCHORS_PATTERN =
  /\b(affect|affects|block|blocks|cause|causes|depend(?:s)?\s+on|drive|drives|enable|enables|gate|gates|influence|influences|map(?:s)?\s+to|own|owns|reference|references|support|supports|unblock|unblocks|use|uses)\b/i;

const graphRAGRelationHintBetweenAnchors = (
  query: string,
  source: string,
  target: string,
): string | undefined => {
  const normalizedQuery = normalizeGraphRAGAnchorText(query);
  const normalizedSource = normalizeGraphRAGAnchorText(source);
  const normalizedTarget = normalizeGraphRAGAnchorText(target);
  if (!normalizedQuery || !normalizedSource || !normalizedTarget) return undefined;

  const sourceStart = normalizedQuery.indexOf(normalizedSource);
  const targetStart = normalizedQuery.indexOf(normalizedTarget, sourceStart + normalizedSource.length);
  if (sourceStart < 0 || targetStart < 0 || targetStart <= sourceStart) return undefined;

  const between = normalizedQuery
    .slice(sourceStart + normalizedSource.length, targetStart)
    .replace(/\b(?:the|a|an|to|with)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  const relation = between.match(GRAPH_RAG_RELATION_HINT_BETWEEN_ANCHORS_PATTERN)?.[1];
  return relation ? cleanGraphRAGRelationHint(relation) : undefined;
};

const inferGraphRAGBridgeAnchor = (
  query: string,
  candidates: string[],
  source?: string,
  target?: string,
): string | undefined => {
  const match = query.match(/\b(?:through|via)\s+(.+?)(?:[?.!,;]|$)/i);
  const bridge = match ? resolveGraphRAGAnchor(match[1] || '', candidates) : undefined;
  if (!bridge) return undefined;
  const normalizedBridge = normalizeGraphRAGAnchorText(bridge);
  if (normalizedBridge === normalizeGraphRAGAnchorText(source || '') || normalizedBridge === normalizeGraphRAGAnchorText(target || '')) {
    return undefined;
  }
  return bridge;
};

const inferGraphRAGArrowTargetHint = (query: string, candidates: string[]): GraphRAGQueryTargetHint => {
  if (!/(?:-{1,2}>|=>)/.test(query)) return {};
  const parts = query
    .split(/\s*(?:-{1,2}>|=>)\s*/)
    .map(part => cleanGraphRAGQueryAnchor(part))
    .filter(Boolean);
  if (parts.length < 2) return {};
  const source = resolveGraphRAGAnchor(parts[0], candidates);
  const target = resolveGraphRAGAnchor(parts[parts.length - 1], candidates);
  if (!source || !target) return {};
  const bridge = parts.length > 2 ? resolveGraphRAGAnchor(parts[1], candidates) : undefined;
  return { source, target, bridge };
};

const cleanGraphRAGRelationHint = (value: string): string | undefined => {
  const relation = asString(value)
    .replace(/\s+/g, '_')
    .replace(/[^a-z0-9_]+/gi, '')
    .toLowerCase();
  return relation || undefined;
};

const graphRAGQueryTargetPatterns: Array<{ pattern: RegExp; source: number; target: number; relation?: number }> = [
  {
    pattern: /\b(?:trace|find|show|explain)\s+(?:the\s+)?(?:strongest\s+)?(?:graph\s+)?(?:path|connection|relationship|route|bridge)\s+(?:between|from)\s+(.+?)\s+(?:and|to)\s+(.+?)(?:[?.!,;]|$)/i,
    source: 1,
    target: 2,
  },
  {
    pattern: /\b(?:path|connection|relationship|route|bridge)\s+between\s+(.+?)\s+(?:and|to)\s+(.+?)(?:[?.!,;]|$)/i,
    source: 1,
    target: 2,
  },
  {
    pattern: /\b(?:path|route)\s+from\s+(.+?)\s+to\s+(.+?)(?:[?.!,;]|$)/i,
    source: 1,
    target: 2,
  },
  {
    pattern: /\bhow\s+(?:does|do)\s+(.+?)\s+(affect|affects|block|blocks|cause|causes|connect(?:ed)?|depend(?:s)?\s+on|drive|drives|enable|enables|gate|gates|influence|influences|link(?:ed)?|map(?:s)?\s+to|own|owns|reference|references|relate(?:d)?|support|supports|unblock|unblocks|use|uses)\s+(?:to|with)?\s*(.+?)(?:[?.!,;]|$)/i,
    source: 1,
    relation: 2,
    target: 3,
  },
  {
    pattern: /\bhow\s+(?:is|are)\s+(.+?)\s+(?:connected|related|linked)\s+(?:to|with)\s+(.+?)(?:[?.!,;]|$)/i,
    source: 1,
    target: 2,
  },
  {
    pattern: /\bhow\s+(?:are|do)\s+(.+?)\s+(?:and|with)\s+(.+?)\s+(?:connected|related|linked|connect|relate|link)\b/i,
    source: 1,
    target: 2,
  },
  {
    pattern: /\b(?:what|which)\s+(?:path|relationship|connection|route|bridge)\s+(?:links|connects|relates)\s+(.+?)\s+(?:to|with|and)\s+(.+?)(?:[?.!,;]|$)/i,
    source: 1,
    target: 2,
  },
];

const inferGraphRAGPatternTargetHint = (query: string, candidates: string[]): GraphRAGQueryTargetHint => {
  for (const targetPattern of graphRAGQueryTargetPatterns) {
    const match = query.match(targetPattern.pattern);
    if (!match) continue;
    if (graphRAGTargetTextLooksBroad(match[targetPattern.target] || '')) continue;
    const source = resolveGraphRAGAnchor(match[targetPattern.source] || '', candidates);
    const target = resolveGraphRAGAnchor(match[targetPattern.target] || '', candidates);
    if (!source || !target) continue;
    return {
      source,
      target,
      relation: targetPattern.relation ? cleanGraphRAGRelationHint(match[targetPattern.relation] || '') : undefined,
      bridge: inferGraphRAGBridgeAnchor(query, candidates, source, target),
    };
  }
  return {};
};

const graphRAGTargetTextLooksBroad = (value: string): boolean => {
  const targetText = cleanGraphRAGQueryAnchor(value);
  if (!targetText) return false;
  const pieces = targetText
    .split(/\s*(?:,|\band\b|\bor\b)\s*/i)
    .map(piece => cleanGraphRAGQueryAnchor(piece))
    .filter(isUsefulGraphRAGAnchor);
  return pieces.length > 1;
};

const GRAPH_RAG_RELATION_VERB_LIST_PATTERN =
  /\bhow\s+(?:does|do)\s+.+?\s+(?:affect|affects|block|blocks|cause|causes|connect(?:ed)?|depend(?:s)?\s+on|drive|drives|enable|enables|gate|gates|influence|influences|link(?:ed)?|map(?:s)?\s+to|own|owns|reference|references|relate(?:d)?|support|supports|unblock|unblocks|use|uses)\s+(?:to|with)?\s*(.+?)(?:[?.!;]|$)/i;

const graphRAGQueryTargetsEntityList = (query: string): boolean => {
  const relationListMatch = query.match(GRAPH_RAG_RELATION_VERB_LIST_PATTERN);
  if (relationListMatch && graphRAGTargetTextLooksBroad(relationListMatch[1] || '')) return true;
  for (const targetPattern of graphRAGQueryTargetPatterns) {
    const match = query.match(targetPattern.pattern);
    if (match && graphRAGTargetTextLooksBroad(match[targetPattern.target] || '')) return true;
  }
  return false;
};

const graphRAGQueryLooksLikeOrderedPathChain = (query: string): boolean => {
  if (!/\b(?:bridge|connect|connection|path|route|relationship)\b/i.test(query)) return false;
  if (/\b(?:through|via)\b/i.test(query)) return false;
  if (!/\bto\b[\s\S]+\bto\b/i.test(query)) return false;
  return query
    .split(/\bto\b/i)
    .map(piece => cleanGraphRAGQueryAnchor(piece))
    .filter(isUsefulGraphRAGAnchor)
    .length >= 3;
};

const inferGraphRAGQueryTargetHint = (
  query: string,
  handoff?: GraphRAGTraceChatHandoff | null,
): GraphRAGQueryTargetHint => {
  const queryText = asString(query);
  if (!queryText) return {};
  const candidates = graphRAGHandoffQueryCandidates(handoff);
  const broadEntityListQuery =
    graphRAGQueryTargetsEntityList(queryText) ||
    graphRAGQueryLooksLikeOrderedPathChain(queryText);

  const arrowHint = inferGraphRAGArrowTargetHint(queryText, candidates);
  if (arrowHint.source && arrowHint.target) return arrowHint;

  const mentioned = graphRAGMentionedCandidateAnchors(queryText, candidates);
  if (mentioned.length >= 2 && relationCueForGraphRAGQuery(queryText) && !broadEntityListQuery) {
    const explicitBridge = inferGraphRAGBridgeAnchor(queryText, candidates);
    const mentionedWithoutBridge = explicitBridge
      ? mentioned.filter(candidate => normalizeGraphRAGAnchorText(candidate) !== normalizeGraphRAGAnchorText(explicitBridge))
      : mentioned;
    if (mentionedWithoutBridge.length < 2) return {};
    const source = mentionedWithoutBridge[0];
    const target = mentionedWithoutBridge[mentionedWithoutBridge.length - 1];
    return {
      source,
      target,
      relation: graphRAGRelationHintBetweenAnchors(queryText, source, target),
      bridge: explicitBridge || (mentioned.length > 2 ? mentioned[1] : inferGraphRAGBridgeAnchor(queryText, candidates, source, target)),
    };
  }

  if (!broadEntityListQuery) {
    const patternHint = inferGraphRAGPatternTargetHint(queryText, candidates);
    if (patternHint.source && patternHint.target) return patternHint;
  }

  return {};
};

export const normalizeGraphRAGChatContextEndpoint = (
  endpoint?: string,
  fallback: '/api/v1/graphrag/chat-context' | '/api/v1/graphrag/chat-context-v2' = '/api/v1/graphrag/chat-context-v2',
): '/api/v1/graphrag/chat-context' | '/api/v1/graphrag/chat-context-v2' => {
  const text = asString(endpoint);
  if (!text) return fallback;
  const path = text.replace(/^https?:\/\/[^/]+/i, '').split(/[?#]/)[0].replace(/\/+$/, '');
  if (path === '/api/v1/graphrag/chat-context') return '/api/v1/graphrag/chat-context';
  if (path === '/api/v1/graphrag/chat-context-v2') return '/api/v1/graphrag/chat-context-v2';
  return fallback;
};

export const shouldUseUnifiedGraphRAGChatContext = (
  endpoint?: string,
  settings: GraphRAGTraceChatContextSettings = {},
): boolean =>
  settings.graphEnabled !== false &&
  normalizeGraphRAGChatContextEndpoint(endpoint) === '/api/v1/graphrag/chat-context-v2';

export const buildGraphRAGChatContextPayload = (
  handoff: GraphRAGTraceChatHandoff | null | undefined,
  query: string,
  settings: GraphRAGTraceChatContextSettings = {},
): GraphRAGTraceChatContextRequest & { query: string } => {
  const request = handoff?.chatContextRequest || {};
  const liveQuery = asString(query);
  const handoffPrompt = asString(handoff?.prompt);
  const requestQuery = asString(request.query);
  const autoSubmittedHandoffPrompt =
    Boolean(liveQuery && handoffPrompt) && liveQuery.toLowerCase() === handoffPrompt.toLowerCase();
  const generatedContextQuery =
    autoSubmittedHandoffPrompt
      ? buildGraphRAGTraceContextQuery(asString(handoff?.filename, 'document'), liveQuery)
      : '';
  const preparedContextQuery = asString(
    handoff?.contextQuery,
    requestQuery && !graphRAGPromptLooksGenerated(requestQuery)
      ? requestQuery
      : generatedContextQuery,
  );
  const payloadQuery =
    autoSubmittedHandoffPrompt && preparedContextQuery
      ? preparedContextQuery
      : liveQuery;
  const selectedQuestion = Array.isArray(handoff?.suggestedQuestions)
    ? handoff?.suggestedQuestions?.find(question => asString(question.prompt).toLowerCase() === asString(query).toLowerCase())
    : undefined;
  const requestRunIds = asStringArray(request.pipeline_run_ids);
  const handoffRunIds = graphRAGHandoffRunIds(handoff);
  const pipelineRunIds = requestRunIds.length > 0 ? requestRunIds : handoffRunIds;
  const requestDomains = asStringArray(request.rag_domains);
  const selectedDomains = asStringArray(settings.ragSelectedDomains);
  const ragDomains = requestDomains.length > 0 ? requestDomains : selectedDomains;
  const topK = positiveNumber(request.top_k) || positiveNumber(settings.ragTopK);
  const responseMode = request.response_mode || 'chat';
  const compactResponse = responseMode === 'chat' || responseMode === 'compact';
  const inferredTargetHint = inferGraphRAGQueryTargetHint(payloadQuery, handoff);
  const requestQueryText = requestQuery.toLowerCase();
  const liveQueryText = payloadQuery.toLowerCase();
  const preparedRequestMatchesQuery =
    Boolean(requestQueryText) &&
    (requestQueryText === liveQueryText ||
      (autoSubmittedHandoffPrompt && requestQueryText === handoffPrompt.toLowerCase()));

  return {
    query: payloadQuery,
    top_k: topK,
    domain: asString(request.domain, handoff?.domain || '') || undefined,
    pipeline_run_id: asString(request.pipeline_run_id) || undefined,
    pipeline_run_ids: pipelineRunIds.length > 0 ? pipelineRunIds : undefined,
    include_graph: typeof request.include_graph === 'boolean' ? request.include_graph : true,
    use_local_pipeline_context: typeof request.use_local_pipeline_context === 'boolean'
      ? request.use_local_pipeline_context
      : pipelineRunIds.length > 0
        ? true
        : undefined,
    include_local_pipeline: typeof request.include_local_pipeline === 'boolean'
      ? request.include_local_pipeline
      : pipelineRunIds.length > 0
        ? true
        : undefined,
    rag_enabled: typeof request.rag_enabled === 'boolean'
      ? request.rag_enabled
      : typeof settings.ragEnabled === 'boolean'
        ? settings.ragEnabled
        : true,
    rag_domains: ragDomains.length > 0 ? ragDomains : undefined,
    rag_search_mode: graphRAGChatSearchModeForRequest(request.rag_search_mode || settings.ragSearchMode),
    source: asString(
      selectedQuestion?.source,
      preparedRequestMatchesQuery
        ? asString(request.source, asString(inferredTargetHint.source))
        : asString(inferredTargetHint.source),
    ) || undefined,
    target: asString(
      selectedQuestion?.target,
      preparedRequestMatchesQuery
        ? asString(request.target, asString(inferredTargetHint.target))
        : asString(inferredTargetHint.target),
    ) || undefined,
    bridge: asString(
      selectedQuestion?.bridge,
      preparedRequestMatchesQuery
        ? asString(request.bridge, asString(inferredTargetHint.bridge))
        : asString(inferredTargetHint.bridge),
    ) || undefined,
    relation: asString(
      selectedQuestion?.relation,
      preparedRequestMatchesQuery
        ? asString(request.relation, asString(inferredTargetHint.relation))
        : asString(inferredTargetHint.relation),
    ) || undefined,
    pipeline_evidence_totals: isRecord(request.pipeline_evidence_totals)
      ? request.pipeline_evidence_totals
      : isRecord(handoff?.evidenceTotals)
        ? handoff?.evidenceTotals
        : undefined,
    pipeline_evidence_families: asStringArray(request.pipeline_evidence_families).length > 0
      ? asStringArray(request.pipeline_evidence_families)
      : asStringArray(handoff?.evidenceFamilies).length > 0
        ? asStringArray(handoff?.evidenceFamilies)
        : undefined,
    graph_overview_counts: isRecord(request.graph_overview_counts)
      ? request.graph_overview_counts
      : isRecord(handoff?.graphOverviewCounts)
        ? handoff?.graphOverviewCounts
        : undefined,
    response_mode: responseMode,
    context_text_max_chars: compactResponse
      ? positiveNumber(request.context_text_max_chars) || 12000
      : positiveNumber(request.context_text_max_chars),
    response_chunk_content_chars: compactResponse
      ? positiveNumber(request.response_chunk_content_chars) || 700
      : positiveNumber(request.response_chunk_content_chars),
    response_graph_context_limit: compactResponse
      ? positiveNumber(request.response_graph_context_limit) || 2
      : positiveNumber(request.response_graph_context_limit),
    response_graph_path_limit: compactResponse
      ? positiveNumber(request.response_graph_path_limit) || 4
      : positiveNumber(request.response_graph_path_limit),
    response_evidence_card_limit: compactResponse
      ? positiveNumber(request.response_evidence_card_limit) || 3
      : positiveNumber(request.response_evidence_card_limit),
    response_relationship_limit: compactResponse
      ? positiveNumber(request.response_relationship_limit) || 8
      : positiveNumber(request.response_relationship_limit),
    response_path_limit: compactResponse
      ? positiveNumber(request.response_path_limit) || 4
      : positiveNumber(request.response_path_limit),
  };
};

export const buildGraphRAGTraceQuestions = (result: unknown, limit = 4): GraphRAGTraceQuestion[] => {
  if (!isRecord(result) || limit <= 0) return [];

  const document = getDocument(result);
  const filename = asString(document.filename, asString(result.filename, 'Document'));
  const summary = getSummary(result);
  const documentProfile = getDocumentProfile(result);
  const artifactManifest = getArtifactManifest(result);
  const sampleOutputs = collectSampleOutputs(result);
  const capabilities = collectCapabilities(summary, documentProfile, sampleOutputs);
  const relationships = collectRelationships(result, sampleOutputs);
  const pathCandidates = collectPathCandidates(result, sampleOutputs);
  const entities = collectEntities(result, sampleOutputs, relationships);
  const evidenceTerms = collectEvidenceTerms(result, sampleOutputs, capabilities);
  const questions: GraphRAGTraceQuestion[] = [];

  for (const question of collectBackendQuestions(result)) {
    pushUniqueQuestion(questions, question, limit);
  }

  pushUniqueQuestion(questions, {
    id: 'summary',
    label: 'Summarize trace',
    kind: 'summary',
    prompt: buildGraphRAGTracePrompt(filename, summary, documentProfile, sampleOutputs, artifactManifest),
  }, limit);

  for (const artifactQuestion of artifactEvidencePrompts(filename, artifactManifest)) {
    pushUniqueQuestion(questions, {
      id: `evidence-${normalizeKey(artifactQuestion.label)}`,
      label: artifactQuestion.label,
      kind: 'evidence',
      prompt: artifactQuestion.prompt,
    }, limit);
  }

  const [primaryRelationship, ...remainingRelationships] = relationships;
  if (primaryRelationship) {
    pushUniqueQuestion(questions, {
      id: `relationship-${normalizeKey(primaryRelationship.source)}-${normalizeKey(primaryRelationship.target)}-${normalizeKey(primaryRelationship.relation)}`,
      label: compact(`${primaryRelationship.source} to ${primaryRelationship.target}`),
      kind: 'relationship',
      prompt: relationshipPrompt(filename, primaryRelationship),
      source: primaryRelationship.source,
      target: primaryRelationship.target,
      relation: primaryRelationship.relation,
    }, limit);
  }

  for (const evidenceQuestion of evidencePrompts(filename, evidenceTerms)) {
    pushUniqueQuestion(questions, {
      id: `evidence-${normalizeKey(evidenceQuestion.label)}`,
      label: evidenceQuestion.label,
      kind: 'evidence',
      prompt: evidenceQuestion.prompt,
    }, limit);
  }

  for (const relationship of remainingRelationships.slice(0, 2)) {
    pushUniqueQuestion(questions, {
      id: `relationship-${normalizeKey(relationship.source)}-${normalizeKey(relationship.target)}-${normalizeKey(relationship.relation)}`,
      label: compact(`${relationship.source} to ${relationship.target}`),
      kind: 'relationship',
      prompt: relationshipPrompt(filename, relationship),
      source: relationship.source,
      target: relationship.target,
      relation: relationship.relation,
    }, limit);
  }

  const primaryPath = pathCandidates[0];
  if (primaryPath) {
    pushUniqueQuestion(questions, {
      id: `path-${normalizeKey(primaryPath.source)}-${normalizeKey(primaryPath.bridge)}-${normalizeKey(primaryPath.target)}`,
      label: compact(`Path: ${primaryPath.source} via ${primaryPath.bridge}`),
      kind: 'path',
      prompt: pathCandidatePrompt(filename, primaryPath),
      source: primaryPath.source,
      target: primaryPath.target,
      bridge: primaryPath.bridge,
    }, limit);
  } else if (entities.length >= 2) {
    const [source, target] = entities;
    pushUniqueQuestion(questions, {
      id: `path-${normalizeKey(source.name)}-${normalizeKey(target.name)}`,
      label: compact(`Path: ${source.name} to ${target.name}`),
      kind: 'path',
      prompt: pathPrompt(filename, source.name, target.name),
      source: source.name,
      target: target.name,
    }, limit);
  }

  return questions.slice(0, limit);
};

export const buildGraphRAGTraceChatHandoff = (
  result: unknown,
  options: { prompt?: string; questionLimit?: number } = {},
): GraphRAGTraceChatHandoff => {
  if (!isRecord(result)) return {};

  const document = getDocument(result);
  const summary = getSummary(result);
  const documentProfile = getDocumentProfile(result);
  const artifactManifest = getArtifactManifest(result);
  const sampleOutputs = collectSampleOutputs(result);
  const interactionHandoff = isRecord(result.interaction_handoff) ? result.interaction_handoff : {};
  const counts = isRecord(interactionHandoff.counts) ? interactionHandoff.counts : {};
  const filename = asString(document.filename, asString(result.filename, 'Document'));
  const runId = asString(result.run_id);
  const suggestedQuestions = buildGraphRAGTraceQuestions(result, options.questionLimit ?? 6);
  const prompt = asString(
    options.prompt,
    suggestedQuestions[0]?.prompt
      || buildGraphRAGTracePrompt(filename, summary, documentProfile, sampleOutputs, artifactManifest),
  );
  const contextQuery = buildGraphRAGTraceContextQuery(filename, prompt);
  const capabilityCards = asArray(interactionHandoff.capability_cards)
    .filter(isRecord)
    .map((card, index) => ({
      id: asString(card.id, `capability-${index + 1}`),
      label: asString(card.label, asString(card.title, `Capability ${index + 1}`)),
      detail: asString(card.detail, asString(card.description)),
      count: card.count !== undefined ? asNumber(card.count) : undefined,
    }))
    .filter(card => card.label)
    .slice(0, 8);

  return {
    runId: runId || undefined,
    runIds: runId ? [runId] : undefined,
    filename,
    domain: asString(document.domain, asString(result.domain)) || undefined,
    prompt,
    displayPrompt: buildGraphRAGTraceDisplayPrompt(filename, prompt),
    contextQuery,
    chatContextEndpoint: asString(interactionHandoff.chat_context_endpoint, '/api/v1/graphrag/chat-context-v2'),
    chatContextRequest: normalizeTraceChatContextRequest(interactionHandoff.chat_context_request, {
      prompt,
      contextQuery,
      domain: asString(document.domain, asString(result.domain)) || undefined,
      runId: runId || undefined,
    }),
    suggestedQuestions,
    recommendedMode: asString(interactionHandoff.recommended_mode) || undefined,
    answerGuidance: asString(interactionHandoff.answer_guidance) || undefined,
    citationScopes: asStringArray(interactionHandoff.citation_scopes),
    capabilityCards,
    graphAnchorCount: asNumber(counts.graph_anchors, asArray(interactionHandoff.graph_anchors).length),
    retrievalRehearsalCount: asNumber(counts.retrieval_rehearsals, asArray(interactionHandoff.retrieval_rehearsals).length),
  };
};
