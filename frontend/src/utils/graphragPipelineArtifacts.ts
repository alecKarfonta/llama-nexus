type JsonRecord = Record<string, unknown>;

export interface GraphRAGPipelineArtifactDownload {
  kind: string;
  label: string;
  shortLabel: string;
  contentType: string;
  endpoint: string;
  available: boolean;
  charCount?: number;
  filename?: string;
  downloadFilename: string;
  detail: string;
}

interface NormalizeArtifactOptions {
  includeManifest?: boolean;
  includeUnavailable?: boolean;
  runId?: string;
  documentFilename?: string;
}

const ARTIFACT_SUFFIX_BY_KIND: Record<string, string> = {
  structured_ingest: 'structured',
  local_graph: 'local-graph',
  local_context: 'local-context',
  local_graph_reasoning: 'local-graph-reasoning',
  interaction_handoff: 'interaction-handoff',
  stages: 'stages',
  samples: 'samples',
  document_profile: 'document-profile',
  summary: 'summary',
  full: 'run',
  manifest: 'artifacts',
};

const CONTENT_TYPE_EXTENSION: Array<[string, string]> = [
  ['text/markdown', 'md'],
  ['application/markdown', 'md'],
  ['application/json', 'json'],
  ['text/json', 'json'],
  ['text/csv', 'csv'],
  ['text/plain', 'txt'],
];

const isRecord = (value: unknown): value is JsonRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const asString = (value: unknown, fallback = ''): string =>
  typeof value === 'string' && value.trim() ? value.trim() : fallback;

const asNumber = (value: unknown): number | undefined => {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
};

const asArray = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

const asBoolean = (value: unknown): boolean => value === true;

const normalizeKind = (value: unknown): string =>
  asString(value, 'artifact').toLowerCase().replace(/[\s-]+/g, '_');

const formatKey = (value: string): string =>
  value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, character => character.toUpperCase());

const basename = (value: string): string =>
  value.split(/[\\/]/).filter(Boolean).pop() || value;

const sanitizeFileName = (value: string, fallback: string): string => {
  const safe = basename(value)
    .replace(/[^A-Za-z0-9._-]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^[._-]+|[._-]+$/g, '');
  return safe || fallback;
};

const fileStem = (value: string, fallback: string): string => {
  const safe = sanitizeFileName(value, fallback);
  return safe.replace(/\.[A-Za-z0-9]{1,12}$/, '') || fallback;
};

const extensionForContentType = (contentType: string): string => {
  const normalized = contentType.toLowerCase();
  const match = CONTENT_TYPE_EXTENSION.find(([prefix]) => normalized.includes(prefix));
  return match?.[1] || 'bin';
};

const hasFileExtension = (filename: string): boolean =>
  /\.[A-Za-z0-9]{1,12}$/.test(filename);

const formatCount = (value: number): string => value.toLocaleString();

const artifactEndpoint = (runId: string, kind: string): string =>
  runId ? `/api/v1/graphrag/ingest/pipeline/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(kind)}` : '';

const artifactAvailable = (value: unknown): boolean => {
  if (value === undefined || value === null) return false;
  if (Array.isArray(value)) return value.length > 0;
  if (isRecord(value)) return Object.keys(value).length > 0;
  if (typeof value === 'string') return value.trim().length > 0;
  return true;
};

const artifactRecord = (
  run: JsonRecord,
  kind: string,
  label: string,
  contentType: string,
  available: boolean,
  extra: JsonRecord = {},
): JsonRecord => {
  const runId = asString(run.run_id);
  return {
    kind,
    label,
    available,
    content_type: contentType,
    endpoint: artifactEndpoint(runId, kind),
    ...extra,
  };
};

export const buildGraphRAGPipelineArtifactManifest = (runValue: unknown): JsonRecord => {
  const run = isRecord(runValue) ? runValue : {};
  const existingManifest = isRecord(run.artifacts) ? run.artifacts : {};
  if (asArray(existingManifest.artifacts).length > 0) {
    return {
      ...existingManifest,
      run_id: asString(existingManifest.run_id, asString(run.run_id)),
      status: asString(existingManifest.status, asString(run.status)),
      document: isRecord(existingManifest.document)
        ? existingManifest.document
        : isRecord(run.document)
          ? run.document
          : undefined,
    };
  }

  const runId = asString(run.run_id);
  if (!runId) return {};

  const document = isRecord(run.document) ? run.document : {};
  const sourceArtifact = isRecord(run.source_document_artifact) ? run.source_document_artifact : {};
  const structuredArtifact = isRecord(run.structured_ingest_artifact) ? run.structured_ingest_artifact : {};
  const structuredIngest = isRecord(run.structured_ingest) ? run.structured_ingest : {};
  const filename = asString(document.filename, asString(run.filename, 'document'));

  const artifacts = [
    artifactRecord(
      run,
      'source_document',
      'Original source document',
      asString(sourceArtifact.content_type, asString(document.content_type, 'application/octet-stream')),
      artifactAvailable(sourceArtifact) || asBoolean(document.source_artifact_available),
      {
        filename: asString(sourceArtifact.filename, filename),
        size_bytes: asNumber(sourceArtifact.size_bytes) ?? asNumber(document.source_artifact_size_bytes) ?? asNumber(document.size_bytes),
        sha256: asString(sourceArtifact.sha256, asString(document.source_artifact_sha256)),
      },
    ),
    artifactRecord(
      run,
      'structured_ingest',
      'Structured ingest Markdown',
      asString(structuredArtifact.content_type, 'text/markdown'),
      artifactAvailable(structuredArtifact) || asBoolean(structuredIngest.artifact_available),
      {
        filename: asString(structuredArtifact.filename, asString(structuredIngest.filename, `${fileStem(filename, 'document')}.structured.md`)),
        char_count: asNumber(structuredArtifact.char_count),
        sha256: asString(structuredArtifact.sha256, asString(structuredIngest.artifact_sha256)),
        truncated: structuredArtifact.truncated ?? structuredIngest.truncated,
        uses_structured_text: structuredArtifact.uses_structured_text ?? structuredIngest.uses_structured_text,
      },
    ),
    artifactRecord(run, 'local_graph', 'Local graph artifact', 'application/json', artifactAvailable(run.local_graph)),
    artifactRecord(run, 'local_context', 'Local context cache', 'application/json', artifactAvailable(run.local_context)),
    artifactRecord(run, 'local_graph_reasoning', 'Local graph reasoning index', 'application/json', artifactAvailable(run.local_graph_reasoning)),
    artifactRecord(run, 'interaction_handoff', 'Interaction handoff', 'application/json', artifactAvailable(run.interaction_handoff)),
    artifactRecord(run, 'stages', 'Pipeline stages', 'application/json', asArray(run.stages).length > 0),
    artifactRecord(run, 'samples', 'Pipeline samples', 'application/json', artifactAvailable(run.samples)),
    artifactRecord(run, 'document_profile', 'Document profile', 'application/json', artifactAvailable(run.document_profile)),
    artifactRecord(run, 'summary', 'Run summary', 'application/json', artifactAvailable(run.summary)),
    artifactRecord(run, 'full', 'Full pipeline run', 'application/json', true),
  ];

  return {
    run_id: runId,
    status: asString(run.status),
    document,
    created_at: run.created_at,
    updated_at: run.updated_at,
    artifact_count: artifacts.filter(artifact => artifact.available === true).length,
    artifacts,
  };
};

const buildDownloadFilename = (
  artifact: JsonRecord,
  manifest: JsonRecord,
  kind: string,
  contentType: string,
  options: NormalizeArtifactOptions,
): string => {
  const explicitFilename = asString(artifact.filename);
  if (explicitFilename) {
    const safeExplicit = sanitizeFileName(explicitFilename, `${kind}.${extensionForContentType(contentType)}`);
    return hasFileExtension(safeExplicit)
      ? safeExplicit
      : `${safeExplicit}.${extensionForContentType(contentType)}`;
  }

  const runId = asString(options.runId, asString(manifest.run_id, 'graphrag-pipeline-run'));
  const document = isRecord(manifest.document) ? manifest.document : {};
  const documentFilename = asString(options.documentFilename, asString(document.filename));
  const runBase = sanitizeFileName(runId, 'graphrag-pipeline-run');
  const extension = extensionForContentType(contentType);

  if (kind === 'structured_ingest') {
    const structuredBase = fileStem(documentFilename || runBase, runBase);
    return `${structuredBase}.${ARTIFACT_SUFFIX_BY_KIND.structured_ingest}.${extension}`;
  }

  if (kind === 'full') {
    return `${runBase}.${extension}`;
  }

  const suffix = ARTIFACT_SUFFIX_BY_KIND[kind] || kind.replace(/_/g, '-');
  return `${runBase}.${suffix}.${extension}`;
};

const buildDetail = (artifact: JsonRecord, contentType: string, endpoint: string): string => {
  const charCount = asNumber(artifact.char_count);
  return [
    contentType,
    charCount !== undefined ? `${formatCount(charCount)} chars` : '',
    endpoint,
  ].filter(Boolean).join(' • ');
};

const toDescriptor = (
  artifact: JsonRecord,
  manifest: JsonRecord,
  options: NormalizeArtifactOptions,
): GraphRAGPipelineArtifactDownload | null => {
  const kind = normalizeKind(artifact.kind);
  const available = artifact.available === true;
  if (!available && !options.includeUnavailable) return null;

  const contentType = asString(artifact.content_type, 'application/octet-stream');
  const endpoint = asString(
    artifact.endpoint,
    artifactEndpoint(asString(options.runId, asString(manifest.run_id)), kind),
  );
  const label = asString(artifact.label, formatKey(kind));
  const filename = asString(artifact.filename);
  return {
    kind,
    label,
    shortLabel: formatKey(kind),
    contentType,
    endpoint,
    available,
    charCount: asNumber(artifact.char_count),
    filename: filename || undefined,
    downloadFilename: buildDownloadFilename(artifact, manifest, kind, contentType, options),
    detail: buildDetail(artifact, contentType, endpoint),
  };
};

export const normalizeGraphRAGPipelineArtifactDownloads = (
  manifestValue: unknown,
  options: NormalizeArtifactOptions = {},
): GraphRAGPipelineArtifactDownload[] => {
  const manifest = isRecord(manifestValue) ? manifestValue : {};
  const runId = asString(options.runId, asString(manifest.run_id));
  const artifacts = asArray(manifest.artifacts)
    .filter(isRecord)
    .map(artifact => toDescriptor(artifact, manifest, options))
    .filter((artifact): artifact is GraphRAGPipelineArtifactDownload => artifact !== null);

  if (options.includeManifest !== false && runId) {
    const manifestArtifact = toDescriptor(
      {
        kind: 'manifest',
        label: 'Artifact manifest',
        available: true,
        content_type: 'application/json',
        endpoint: artifactEndpoint(runId, 'manifest'),
      },
      manifest,
      { ...options, runId },
    );
    if (manifestArtifact) artifacts.unshift(manifestArtifact);
  }

  const seen = new Set<string>();
  return artifacts.filter(artifact => {
    if (seen.has(artifact.kind)) return false;
    seen.add(artifact.kind);
    return true;
  });
};
