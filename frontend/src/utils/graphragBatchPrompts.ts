type JsonRecord = Record<string, unknown>;

export interface GraphRAGBatchQuestion {
  id: string;
  label: string;
  prompt: string;
  kind: 'summary' | 'bridge' | 'path' | 'relationship' | 'shared_entity';
  source?: string;
  target?: string;
  bridge?: string;
  relation?: string;
  focus?: string;
}

interface BatchPromptOptions {
  filenames?: string[];
  overview?: unknown;
}

const isRecord = (value: unknown): value is JsonRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const asArray = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

const asString = (value: unknown, fallback = ''): string =>
  typeof value === 'string' && value.trim() ? value.trim() : fallback;

const compact = (value: string, limit = 56): string => {
  const text = value.replace(/\s+/g, ' ').trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, Math.max(1, limit - 3)).trim()}...`;
};

const normalizeKey = (value: string): string =>
  value.replace(/[^a-zA-Z0-9]+/g, '_').replace(/^_+|_+$/g, '').toLowerCase();

const formatFileText = (filenames: string[] = []): string => {
  const visible = filenames.filter(Boolean);
  if (visible.length === 0) return 'the uploaded files';
  const overflow = visible.length > 6 ? `, and ${visible.length - 6} more` : '';
  return `${visible.slice(0, 6).join(', ')}${overflow}`;
};

const overviewRecord = (overview: unknown): JsonRecord =>
  isRecord(overview) ? overview : {};

const overviewList = (overview: JsonRecord, key: string): JsonRecord[] =>
  asArray(overview[key]).filter(isRecord);

const pushUniqueQuestion = (
  questions: GraphRAGBatchQuestion[],
  question: GraphRAGBatchQuestion,
  limit: number,
) => {
  if (questions.length >= limit) return;
  const key = question.prompt.toLowerCase();
  if (questions.some(existing => existing.prompt.toLowerCase() === key || existing.label.toLowerCase() === question.label.toLowerCase())) {
    return;
  }
  questions.push(question);
};

const normalizeBackendQuestion = (question: JsonRecord, index: number): GraphRAGBatchQuestion | null => {
  const prompt = asString(question.prompt);
  const label = asString(question.label, prompt ? compact(prompt) : '');
  if (!prompt || !label) return null;
  const kind = asString(question.kind);
  return {
    id: asString(question.id, `backend-batch-question-${index}`),
    label,
    prompt,
    kind: kind === 'summary' || kind === 'bridge' || kind === 'path' || kind === 'relationship' || kind === 'shared_entity'
      ? kind
      : 'relationship',
    source: asString(question.source) || undefined,
    target: asString(question.target) || undefined,
    bridge: asString(question.bridge) || undefined,
    relation: asString(question.relation) || undefined,
    focus: asString(question.focus) || undefined,
  };
};

export const buildGraphRAGBatchPrompt = ({ filenames = [], overview }: BatchPromptOptions): string => {
  const graphOverview = overviewRecord(overview);
  const bridgeCount = Number(graphOverview.bridge_entity_count || 0);
  const pathCount = Number(graphOverview.cross_document_path_count || 0);
  const relationshipCount = Number(graphOverview.relationship_count || 0);
  const fileText = formatFileText(filenames);
  const graphText = [
    relationshipCount > 0 ? `${relationshipCount} graph relationships` : '',
    bridgeCount > 0 ? `${bridgeCount} bridge entities` : '',
    pathCount > 0 ? `${pathCount} cross-document paths` : '',
  ].filter(Boolean).join(', ');

  return (
    `Use the document context for this batch (${fileText}) to compare the sources. `
    + `${graphText ? `The batch graph overview found ${graphText}. ` : ''}`
    + 'Explain shared entities, cross-document relationships, graph paths, bridge entities, contradictions or gaps, '
    + 'and unique evidence from each file. Cite filename plus slide/page/table/chart/connector provenance whenever available.'
  );
};

export const buildGraphRAGBatchQuestions = (
  options: BatchPromptOptions,
  limit = 4,
): GraphRAGBatchQuestion[] => {
  if (limit <= 0) return [];

  const graphOverview = overviewRecord(options.overview);
  const filenames = options.filenames || [];
  const fileText = formatFileText(filenames);
  const questions: GraphRAGBatchQuestion[] = [];
  const paths = overviewList(graphOverview, 'cross_document_paths');
  const bridges = overviewList(graphOverview, 'bridge_entities');
  const sharedEntities = overviewList(graphOverview, 'shared_entities');
  const relationships = overviewList(graphOverview, 'top_relationships');

  for (const question of overviewList(graphOverview, 'suggested_questions')) {
    const normalized = normalizeBackendQuestion(question, questions.length);
    if (normalized) pushUniqueQuestion(questions, normalized, limit);
  }

  pushUniqueQuestion(questions, {
    id: 'batch-summary',
    label: 'Compare batch graph',
    kind: 'summary',
    prompt: buildGraphRAGBatchPrompt(options),
  }, limit);

  const path = paths[0];
  if (path) {
    const source = asString(path.source, 'source entity');
    const bridge = asString(path.bridge, 'bridge entity');
    const target = asString(path.target, 'target entity');
    const pathFiles = asArray(path.filenames).map(item => asString(item)).filter(Boolean);
    pushUniqueQuestion(questions, {
      id: `batch-path-${normalizeKey(source)}-${normalizeKey(bridge)}-${normalizeKey(target)}`,
      label: compact(`Path: ${source} to ${target}`),
      kind: 'path',
      source,
      bridge,
      target,
      prompt: (
        `Use the document context for this batch (${fileText}) to explain the cross-document graph path `
        + `"${source}" -> "${bridge}" -> "${target}". `
        + `Compare evidence from ${pathFiles.length ? pathFiles.join(' and ') : 'the relevant files'}, explain why "${bridge}" is the bridge entity, `
        + 'and cite filename plus slide/page/table/chart/connector provenance for each hop.'
      ),
    }, limit);
  }

  const bridge = bridges[0];
  if (bridge) {
    const name = asString(bridge.name, 'bridge entity');
    const bridgeFiles = asArray(bridge.filenames).map(item => asString(item)).filter(Boolean);
    pushUniqueQuestion(questions, {
      id: `batch-bridge-${normalizeKey(name)}`,
      label: compact(`Bridge: ${name}`),
      kind: 'bridge',
      bridge: name,
      prompt: (
        `Use the document context for this batch (${fileText}) to analyze "${name}" as a bridge entity across documents. `
        + `Compare how it appears in ${bridgeFiles.length ? bridgeFiles.join(', ') : 'the uploaded files'}, list the strongest connected relationships, `
        + 'identify any contradictions or gaps, and cite provenance for each file.'
      ),
    }, limit);
  }

  const sharedEntity = sharedEntities[0];
  if (sharedEntity) {
    const name = asString(sharedEntity.name, 'shared entity');
    pushUniqueQuestion(questions, {
      id: `batch-shared-${normalizeKey(name)}`,
      label: compact(`Shared: ${name}`),
      kind: 'shared_entity',
      source: name,
      prompt: (
        `Use the document context for this batch (${fileText}) to compare the shared entity "${name}" across documents. `
        + 'Explain which relationships and evidence each file contributes, where the entity meaning changes, and cite source provenance.'
      ),
    }, limit);
  }

  const relationship = relationships[0];
  if (relationship) {
    const source = asString(relationship.source, 'source');
    const relation = asString(relationship.relation, 'related_to');
    const target = asString(relationship.target, 'target');
    pushUniqueQuestion(questions, {
      id: `batch-relationship-${normalizeKey(source)}-${normalizeKey(relation)}-${normalizeKey(target)}`,
      label: compact(`${source} to ${target}`),
      kind: 'relationship',
      source,
      relation,
      target,
      prompt: (
        `Use the document context for this batch (${fileText}) to validate and explain the relationship `
        + `"${source}" --[${relation}]--> "${target}". `
        + 'Find supporting or conflicting evidence across files and cite filename plus slide/page/table/chart/connector provenance.'
      ),
    }, limit);
  }

  return questions.slice(0, limit);
};
