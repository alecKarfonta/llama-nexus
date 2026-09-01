import type { GraphRAGBatchQuestion } from './graphragBatchPrompts';

type JsonRecord = Record<string, unknown>;

export interface GraphRAGDocumentPromptGraphRow {
  kind?: 'relationship' | 'path';
  label?: string;
  relation?: string;
  source?: string;
  target?: string;
  bridge?: string;
  title?: string;
  text?: string;
}

interface GraphRAGDocumentPromptOptions {
  filename?: string;
  overview?: unknown;
  graphRows?: GraphRAGDocumentPromptGraphRow[];
  evidenceFamilies?: Array<{ label: string; value?: number }>;
  preferredQuestions?: GraphRAGBatchQuestion[];
}

const isRecord = (value: unknown): value is JsonRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const asRecord = (value: unknown): JsonRecord => (isRecord(value) ? value : {});

const asString = (value: unknown, fallback = ''): string =>
  typeof value === 'string' && value.trim() ? value.trim() : fallback;

const asNumber = (value: unknown): number => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.max(0, value);
  }
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return Math.max(0, parsed);
  }
  return 0;
};

const compact = (value: string, limit = 56): string => {
  const text = value.replace(/\s+/g, ' ').trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, Math.max(1, limit - 3)).trim()}...`;
};

const normalizeKey = (value: string): string =>
  value.replace(/[^a-zA-Z0-9]+/g, '_').replace(/^_+|_+$/g, '').toLowerCase();

const documentName = (filename?: string): string =>
  asString(filename, 'this document');

const WEAK_GRAPH_RELATIONS = new Set([
  'co_occurs',
  'co_occurs_with',
  'co-occurs-with',
  'co occurs with',
  'mentioned_with',
  'mentioned with',
  'related_to',
  'related to',
]);

const LOW_VALUE_ENTITY_LABELS = new Set([
  'a',
  'an',
  'and',
  'for',
  'it',
  'please',
  'the',
  'this',
  'to',
  'you',
]);

const normalizedRelation = (value?: string): string =>
  asString(value)
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();

const GRAPH_DOCUMENT_NAME_PATTERN = /\.(?:csv|docx?|html?|md|pdf|pptx?|txt|xlsx?)$/i;

const EVIDENCE_SENTENCE_RELATION_PATTERN =
  /\b(?:approves?|blocks?|depends?|drives?|gates?|influences?|owns?|references?|requires?|supports?)\b/i;

const sourceDocumentLabel = (value?: string): boolean =>
  GRAPH_DOCUMENT_NAME_PATTERN.test(asString(value));

const evidenceSentenceLabel = (value?: string): boolean => {
  const text = asString(value);
  if (!/[.!?]$/.test(text)) return false;
  const words = text.split(/\s+/).filter(Boolean);
  return words.length >= 4 && EVIDENCE_SENTENCE_RELATION_PATTERN.test(text);
};

const meaningfulEntityLabel = (value?: string): boolean => {
  const normalized = asString(value)
    .replace(/[^a-zA-Z0-9]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
  if (!normalized || LOW_VALUE_ENTITY_LABELS.has(normalized)) return false;
  if (normalized.length < 4) return false;
  if (sourceDocumentLabel(value) || evidenceSentenceLabel(value)) return false;
  return true;
};

const starterWorthyGraphRow = (row: GraphRAGDocumentPromptGraphRow): boolean => {
  const relation = normalizedRelation(row.relation || row.label);
  const weakRelation = WEAK_GRAPH_RELATIONS.has(relation);
  const pathLike = row.kind === 'path' || Boolean(asString(row.bridge));
  if (pathLike) {
    const bridge = asString(row.bridge);
    return (
      meaningfulEntityLabel(row.source) &&
      meaningfulEntityLabel(row.target) &&
      (!bridge || meaningfulEntityLabel(bridge))
    );
  }

  return (
    !weakRelation &&
    meaningfulEntityLabel(row.source) &&
    meaningfulEntityLabel(row.target)
  );
};

const starterWorthyPreferredQuestion = (question: GraphRAGBatchQuestion): boolean => {
  if (question.kind === 'summary' || question.focus === 'provenance') return true;
  if (question.kind === 'path' || question.kind === 'bridge') {
    return starterWorthyGraphRow({
      kind: 'path',
      source: question.source,
      bridge: question.bridge,
      target: question.target,
    });
  }
  if (question.kind === 'relationship') {
    return starterWorthyGraphRow({
      kind: 'relationship',
      source: question.source,
      relation: question.relation,
      target: question.target,
    });
  }
  if (question.kind === 'shared_entity') {
    return meaningfulEntityLabel(question.bridge || question.source || question.target);
  }
  return false;
};

const evidenceFamilyQuestionLabel = (label?: string): string => {
  const cleaned = asString(label);
  if (!cleaned) return 'Show source evidence';
  if (/^(?:semantic|semantic edges?|graph edges?|relations?|relationships?)$/i.test(cleaned)) {
    return 'Show source evidence';
  }
  return compact(`Show ${cleaned}`);
};

const overviewCounts = (overview: unknown) => {
  const record = asRecord(overview);
  const relationships = asNumber(
    record.relationship_candidate_count ?? record.relationship_count,
  );
  const entities = asNumber(record.entity_candidate_count ?? record.entity_count);
  const chunks = asNumber(record.chunk_count ?? record.chunks);
  const paths = asNumber(record.path_count ?? record.cross_document_path_count);
  return { chunks, entities, relationships, paths };
};

const pushUniqueQuestion = (
  questions: GraphRAGBatchQuestion[],
  question: GraphRAGBatchQuestion,
  limit: number,
) => {
  if (questions.length >= limit) return;
  if (!question.prompt.trim() || !question.label.trim()) return;
  const key = question.prompt.toLowerCase();
  if (
    questions.some(
      (existing) =>
        existing.prompt.toLowerCase() === key ||
        existing.label.toLowerCase() === question.label.toLowerCase(),
    )
  ) {
    return;
  }
  questions.push(question);
};

export const buildGraphRAGDocumentPrompt = (
  options: GraphRAGDocumentPromptOptions,
): string => {
  const filename = documentName(options.filename);
  const counts = overviewCounts(options.overview);
  const countSummary = [
    counts.chunks > 0 ? `${counts.chunks} chunks` : '',
    counts.entities > 0 ? `${counts.entities} entities` : '',
    counts.relationships > 0 ? `${counts.relationships} relationships` : '',
    counts.paths > 0 ? `${counts.paths} paths` : '',
  ]
    .filter(Boolean)
    .join(', ');

  return (
    `Use the GraphRAG context for ${filename} to summarize the document. `
    + `${countSummary ? `The current import includes ${countSummary}. ` : ''}`
    + 'Explain the key facts, decisions, risks, entities, relationships, and source provenance. '
    + 'When graph relationships or paths are available, use them to connect ideas instead of only quoting passages.'
  );
};

export const buildGraphRAGDocumentQuestions = (
  options: GraphRAGDocumentPromptOptions,
  limit = 4,
): GraphRAGBatchQuestion[] => {
  if (limit <= 0) return [];

  const filename = documentName(options.filename);
  const counts = overviewCounts(options.overview);
  const graphRows = (options.graphRows || []).filter(
    (row) => asString(row.source) || asString(row.target) || asString(row.text),
  );
  const starterGraphRows = graphRows.filter(starterWorthyGraphRow);
  const weakOnlyGraphRows = graphRows.length > 0 && starterGraphRows.length === 0;
  const questions: GraphRAGBatchQuestion[] = [];

  for (const preferred of options.preferredQuestions || []) {
    if (!starterWorthyPreferredQuestion(preferred)) continue;
    pushUniqueQuestion(questions, preferred, limit);
  }

  pushUniqueQuestion(
    questions,
    {
      id: 'document-summary',
      label: 'Summarize document',
      kind: 'summary',
      prompt: buildGraphRAGDocumentPrompt(options),
    },
    limit,
  );

  const primaryGraphRow = starterGraphRows[0];
  if (primaryGraphRow) {
    const source = asString(primaryGraphRow.source, 'source entity');
    const target = asString(primaryGraphRow.target, asString(primaryGraphRow.text, 'target entity'));
    const bridge = asString(primaryGraphRow.bridge);
    const relation = asString(primaryGraphRow.relation, asString(primaryGraphRow.label, 'related_to'));
    const isPath = primaryGraphRow.kind === 'path' || Boolean(bridge);
    pushUniqueQuestion(
      questions,
      {
        id: isPath
          ? `document-path-${normalizeKey(source)}-${normalizeKey(bridge)}-${normalizeKey(target)}`
          : `document-relationship-${normalizeKey(source)}-${normalizeKey(relation)}-${normalizeKey(target)}`,
        label: compact(isPath ? `Trace ${source} to ${target}` : `${source} to ${target}`),
        kind: isPath ? 'path' : 'relationship',
        source,
        target,
        bridge: bridge || undefined,
        relation,
        prompt: isPath
          ? (
              `Use the GraphRAG context for ${filename} to explain the graph path `
              + `"${source}" -> "${bridge || 'bridge entity'}" -> "${target}". `
              + 'Cite the source passages, pages, slides, tables, charts, and relationship evidence for each hop.'
            )
          : (
              `Use the GraphRAG context for ${filename} to validate and explain the relationship `
              + `"${source}" --[${relation}]--> "${target}". `
              + 'Cite the source passages, pages, slides, tables, charts, and relationship evidence.'
            ),
      },
      limit,
    );
  } else if (counts.relationships > 0 && !weakOnlyGraphRows) {
    pushUniqueQuestion(
      questions,
      {
        id: 'document-relationship-overview',
        label: 'Map relationships',
        kind: 'relationship',
        prompt:
          `Use the GraphRAG context for ${filename} to identify the most important relationships in the document. `
          + 'Group them by theme, explain why they matter, and cite the strongest evidence for each one.',
      },
      limit,
    );
  }

  const strongestEvidenceFamily = (options.evidenceFamilies || [])
    .filter((family) => asString(family.label))
    .sort((left, right) => asNumber(right.value) - asNumber(left.value))[0];
  pushUniqueQuestion(
    questions,
    {
      id: 'document-provenance',
      label: evidenceFamilyQuestionLabel(strongestEvidenceFamily?.label),
      kind: 'summary',
      focus: 'provenance',
      prompt:
        `Use the GraphRAG context for ${filename} to show the strongest source evidence. `
        + `${strongestEvidenceFamily ? `Focus first on ${strongestEvidenceFamily.label}. ` : ''}`
        + 'Include page, slide, table, chart, section, or file provenance whenever available.',
    },
    limit,
  );

  return questions.slice(0, limit);
};
