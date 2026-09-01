import {
  rankGraphRAGTraceSampleGroups,
  type GraphRAGTraceSampleGroup,
} from './graphragTraceSamples';

type JsonRecord = Record<string, unknown>;

export interface GraphRAGTraceStageTimelineItem {
  id: string;
  name: string;
  label: string;
  status: string;
  statusLabel: string;
  durationMs: number;
  percentOfTotal: number;
  warningCount: number;
  sampleGroupCount: number;
  sampleItemCount: number;
  hasSamples: boolean;
  sampleGroups: string[];
  debugSampleGroups: string[];
}

export interface GraphRAGTraceTimeline {
  totalDurationMs: number;
  stageCount: number;
  completedStageCount: number;
  warningStageCount: number;
  failedStageCount: number;
  sampleStageCount: number;
  slowestStage?: GraphRAGTraceStageTimelineItem;
  stages: GraphRAGTraceStageTimelineItem[];
}

const isRecord = (value: unknown): value is JsonRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const asArray = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

const asString = (value: unknown, fallback = ''): string =>
  typeof value === 'string' && value.trim() ? value.trim() : fallback;

const asNumber = (value: unknown): number => {
  if (typeof value === 'number' && Number.isFinite(value)) return Math.max(0, value);
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return Math.max(0, parsed);
  }
  return 0;
};

const FRIENDLY_TIMELINE_LABELS: Record<string, string> = {
  capability_cards: 'Chat capabilities',
  chunk_evidence_cards: 'Evidence cards',
  compatible_health_urls: 'Health routes',
  compatible_ingest_urls: 'Ingest routes',
  detected_capabilities: 'Detected abilities',
  embedding_evidence_cards: 'Embedding evidence',
  embedding_evidence_family_counts: 'Embedding evidence mix',
  embedding_inputs: 'Embedding text',
  embedding_sample_categories: 'Embedding categories',
  endpoint_attempts: 'Endpoint checks',
  graph_context: 'Relationships',
  graph_interaction_plan: 'Chat context',
  graph_paths: 'Graph paths',
  local_context_cache: 'Retrieval cache',
  local_graph_artifact: 'Graph index',
  local_graph_reasoning_index: 'Reasoning paths',
  relationship_evidence_cards: 'Relationship evidence',
  relationship_sample_categories: 'Relationship categories',
  representative_embedding_inputs: 'Embedding examples',
  retrieval_rehearsals: 'Search rehearsals',
  selected_endpoint: 'Selected endpoint',
  structured_ingest_document: 'Structured ingest',
  upstream_ingest: 'GraphRAG ingest',
  upstream_preflight: 'GraphRAG connection',
};

const friendlyKey = (value: string): string =>
  value.trim().toLowerCase().replace(/[\s-]+/g, '_');

const formatKey = (value: string): string => {
  const friendly = FRIENDLY_TIMELINE_LABELS[friendlyKey(value)];
  if (friendly) return friendly;

  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, character => character.toUpperCase());
};

const isCompleteStatus = (status: string): boolean =>
  ['complete', 'completed', 'success', 'submitted', 'ready'].includes(status);

const isWarningStatus = (status: string): boolean =>
  ['warning', 'preview_only', 'skipped', 'partial', 'not_configured'].includes(status);

const isFailedStatus = (status: string): boolean =>
  ['failed', 'error', 'upstream_failed', 'unavailable', 'unhealthy', 'misconfigured'].includes(status);

const sampleGroups = (sampleOutputs: unknown): GraphRAGTraceSampleGroup[] => {
  if (!isRecord(sampleOutputs)) return [];

  return rankGraphRAGTraceSampleGroups(sampleOutputs);
};

export const buildGraphRAGTraceTimeline = (result: unknown): GraphRAGTraceTimeline => {
  const stages = isRecord(result)
    ? asArray(result.stages).filter(isRecord)
    : [];
  const rawTotalDuration = stages.reduce((total, stage) => total + asNumber(stage.duration_ms), 0);
  const totalDurationMs = Math.round(rawTotalDuration);
  const denominator = rawTotalDuration > 0 ? rawTotalDuration : Math.max(stages.length, 1);

  const items = stages.map((stage, index) => {
    const name = asString(stage.name, `stage_${index + 1}`);
    const status = asString(stage.status, 'unknown').toLowerCase();
    const durationMs = Math.round(asNumber(stage.duration_ms));
    const groups = sampleGroups(stage.sample_outputs);
    const visibleGroups = groups.filter(group => !['debug', 'operational'].includes(group.role));
    const displayGroups = visibleGroups.length > 0 ? visibleGroups : groups;
    const percentBasis = rawTotalDuration > 0 ? durationMs : 1;
    const percentOfTotal = Math.round((percentBasis / denominator) * 1000) / 10;

    return {
      id: `${name}-${index}`,
      name,
      label: formatKey(name),
      status,
      statusLabel: formatKey(status),
      durationMs,
      percentOfTotal,
      warningCount: asArray(stage.warnings).length + (asString(stage.error) ? 1 : 0),
      sampleGroupCount: groups.length,
      sampleItemCount: groups.reduce((total, group) => total + group.count, 0),
      hasSamples: groups.length > 0,
      sampleGroups: displayGroups.map(group => formatKey(group.key)),
      debugSampleGroups: groups
        .filter(group => ['debug', 'operational'].includes(group.role))
        .map(group => formatKey(group.key)),
    };
  });

  const slowestStage = [...items].sort((left, right) => right.durationMs - left.durationMs)[0];

  return {
    totalDurationMs,
    stageCount: items.length,
    completedStageCount: items.filter(item => isCompleteStatus(item.status)).length,
    warningStageCount: items.filter(item => !isFailedStatus(item.status) && (isWarningStatus(item.status) || item.warningCount > 0)).length,
    failedStageCount: items.filter(item => isFailedStatus(item.status)).length,
    sampleStageCount: items.filter(item => item.hasSamples).length,
    slowestStage,
    stages: items,
  };
};
