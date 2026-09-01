import React from 'react';
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Grid,
  LinearProgress,
  Stack,
  Tooltip,
  Typography,
  alpha,
} from '@mui/material';
import {
  AccountTree as GraphIcon,
  QuestionAnswer as ChatIcon,
  CheckCircle as CheckCircleIcon,
  Download as DownloadIcon,
  ErrorOutline as ErrorOutlineIcon,
  ExpandMore as ExpandMoreIcon,
  Schema as SchemaIcon,
  WarningAmber as WarningAmberIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import apiService from '@/services/api';
import { getRoutePath } from '@/config/navigation';
import { useToast } from '@/contexts/ToastContext';
import {
  buildGraphRAGPipelineArtifactManifest,
  normalizeGraphRAGPipelineArtifactDownloads,
  type GraphRAGPipelineArtifactDownload,
} from '@/utils/graphragPipelineArtifacts';
import {
  buildGraphRAGTraceChatHandoff,
  buildGraphRAGTracePrompt,
  buildGraphRAGTraceQuestions,
} from '@/utils/graphragPipelinePrompts';
import {
  buildGraphRAGTraceTimeline,
  type GraphRAGTraceTimeline,
} from '@/utils/graphragPipelineTraceTimeline';
import {
  graphRAGTraceSampleLinkTarget,
  rankGraphRAGTraceSampleGroups,
  rankGraphRAGTraceSampleRecords,
} from '@/utils/graphragTraceSamples';
import { displayPathNodes } from '@/utils/graphragPathDisplay';
import {
  normalizeGraphRAGPipelineRunRollups,
  type GraphRAGDeploymentOutcomeRollup,
  type GraphRAGPipelinePerformanceRollup,
} from '@/utils/graphragPipelineRunRollups';

type JsonRecord = Record<string, unknown>;
type ChipColor = 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning';
type AlertSeverity = 'success' | 'warning' | 'error' | 'info';

interface GraphRAGPipelineTraceProps {
  result: unknown;
  title?: string;
  dense?: boolean;
}

const isRecord = (value: unknown): value is JsonRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const asString = (value: unknown, fallback = ''): string =>
  typeof value === 'string' && value.trim() ? value : fallback;

const asNumber = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

const asArray = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

const asStringArray = (value: unknown): string[] =>
  asArray(value)
    .map(item => asString(item))
    .filter(Boolean);

const FRIENDLY_TRACE_LABELS: Record<string, string> = {
  capability_cards: 'Chat capabilities',
  chunk_evidence_cards: 'Evidence cards',
  compatible_health_urls: 'Health routes',
  compatible_ingest_urls: 'Ingest routes',
  detected_capabilities: 'Detected abilities',
  embedding_evidence_cards: 'Embedding evidence',
  embedding_evidence_family_counts: 'Embedding evidence mix',
  embedding_inputs: 'Embedding text',
  embedding_sample_categories: 'Embedding categories',
  graph_interaction_plan: 'Chat context',
  graph_paths: 'Graph paths',
  graph_rehearsals: 'Search rehearsals',
  local_context_cache: 'Retrieval cache',
  local_graph_artifact: 'Graph index',
  local_graph_reasoning_index: 'Reasoning paths',
  relationship_evidence_cards: 'Relationship evidence',
  relationship_sample_categories: 'Relationship categories',
  representative_embedding_inputs: 'Embedding examples',
  retrieval_rehearsals: 'Search rehearsals',
  structured_ingest_document: 'Structured ingest',
  upstream_ingest: 'GraphRAG ingest',
  upstream_preflight: 'GraphRAG connection',
};

const friendlyKey = (value: string): string =>
  value.trim().toLowerCase().replace(/[\s-]+/g, '_');

const formatKey = (value: string): string => {
  const friendly = FRIENDLY_TRACE_LABELS[friendlyKey(value)];
  if (friendly) return friendly;

  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, character => character.toUpperCase());
};

const formatStageName = (value: unknown): string => formatKey(asString(value, 'stage'));

const formatCompactStageName = (value: unknown): string => {
  const raw = asString(value);
  if (friendlyKey(raw) === 'upstream_ingest') return 'ingest';
  return formatStageName(value).replace(/^GraphRAG\s+/i, '');
};

const FRIENDLY_VALUE_LABELS: Record<string, string> = {
  pptx_xml: 'PPTX structured extraction',
  remote_graph_submitted: 'Remote graph submitted',
  structured_text: 'Structured text',
};

const formatValue = (value: unknown): string => {
  if (typeof value === 'number') return value.toLocaleString();
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'string') return FRIENDLY_VALUE_LABELS[friendlyKey(value)] || value;
  if (value == null) return 'None';
  return JSON.stringify(value);
};

const graphAnchorLabel = (anchor: JsonRecord): string => {
  const kind = asString(anchor.kind, 'anchor');
  const metadata = isRecord(anchor.metadata) ? anchor.metadata : {};
  if (kind === 'path') {
    const nodeValues = asArray(anchor.nodes).map(item => formatValue(item)).filter(Boolean);
    const parsedNodes = nodeValues.length > 0
      ? nodeValues
      : asString(anchor.title, asString(anchor.label))
          .replace(/^path:\s*/i, '')
          .split(/\s*->\s*/)
          .map(item => item.trim())
          .filter(Boolean);
    const nodes = displayPathNodes(parsedNodes);
    if (nodes.length >= 2) return nodes.join(' -> ');
  }
  if (kind === 'styled_text') {
    const term = asString(metadata.emphasis_term, asString(anchor.target, asString(anchor.label, 'Styled text')));
    const style = asString(metadata.style_summary);
    return `Style: ${term}${style ? ` · ${style}` : ''}`;
  }
  if (kind === 'link') {
    const source = asString(anchor.source, asString(anchor.label, 'Link'));
    const target = asString(anchor.uri, asString(metadata.uri, asString(anchor.target, 'target')));
    return `${source} -> ${target}`;
  }
  return asString(anchor.title, asString(anchor.label, 'Graph anchor'));
};

const graphAnchorTooltip = (anchor: JsonRecord): string => {
  const metadata = isRecord(anchor.metadata) ? anchor.metadata : {};
  const parts = [
    asString(anchor.title, asString(anchor.label)),
    anchor.source && (anchor.relation || anchor.target)
      ? `${formatValue(anchor.source)} --[${formatValue(anchor.relation || 'related_to')}]--> ${formatValue(anchor.target || 'target')}`
      : '',
    asString(metadata.emphasis_term) ? `Emphasis: ${asString(metadata.emphasis_term)}` : '',
    asString(metadata.style_summary) ? `Style: ${asString(metadata.style_summary)}` : '',
    asStringArray(metadata.emphasis_styles).length > 0 ? `Styles: ${asStringArray(metadata.emphasis_styles).join(', ')}` : '',
    asString(anchor.uri, asString(metadata.uri)) ? `URI: ${asString(anchor.uri, asString(metadata.uri))}` : '',
    asString(metadata.image_relationship_id) ? `Image relationship: ${asString(metadata.image_relationship_id)}` : '',
    asString(metadata.layout_bbox) ? `Layout: ${asString(metadata.layout_bbox)}` : '',
    asString(anchor.evidence),
  ].filter(Boolean);
  return parts.join('\n');
};

const graphAnchorColor = (kind: string): ChipColor => {
  if (kind === 'path') return 'primary';
  if (kind === 'bridge') return 'secondary';
  if (kind === 'link') return 'success';
  if (kind === 'styled_text') return 'warning';
  return 'info';
};

const getStatusColor = (status: unknown): ChipColor => {
  const normalized = asString(status).toLowerCase();
  if (['complete', 'completed', 'success', 'submitted', 'ready', 'remote_graph_ready', 'remote_graph_submitted', 'remote_submitted'].includes(normalized)) return 'success';
  if (['warning', 'preview_only', 'skipped', 'partial', 'not_configured', 'local_trace_ready', 'graphrag_disabled', 'disabled', 'preflight_blocked', 'remote_ready'].includes(normalized)) return 'warning';
  if (['failed', 'error', 'upstream_failed', 'unavailable', 'unhealthy', 'misconfigured', 'remote_ingest_failed', 'health_unavailable', 'health_unhealthy'].includes(normalized)) return 'error';
  if (['processing', 'running', 'pending'].includes(normalized)) return 'info';
  return 'default';
};

const getAlertSeverity = (status: unknown): AlertSeverity => {
  const color = getStatusColor(status);
  if (color === 'success') return 'success';
  if (color === 'error') return 'error';
  if (color === 'warning') return 'warning';
  return 'info';
};

const getStatusIcon = (status: unknown): React.ReactElement => {
  const color = getStatusColor(status);
  if (color === 'success') return <CheckCircleIcon />;
  if (color === 'error') return <ErrorOutlineIcon />;
  if (color === 'warning') return <WarningAmberIcon />;
  return <GraphIcon />;
};

const graphRAGTraceDisplayStatus = ({
  status,
  localTraceReady,
  chatReady,
  remoteSubmitted,
  remoteReady,
}: {
  status: string;
  localTraceReady: boolean;
  chatReady: boolean;
  remoteSubmitted: boolean;
  remoteReady: boolean;
}): { label: string; color: ChipColor; icon: React.ReactElement } => {
  const normalized = status.toLowerCase();
  if (
    (normalized === 'preview_only' || normalized === 'local_trace_ready') &&
    (localTraceReady || chatReady) &&
    !remoteSubmitted &&
    !remoteReady
  ) {
    return {
      label: 'Local Ready',
      color: 'success',
      icon: <CheckCircleIcon />,
    };
  }

  return {
    label: formatStageName(status),
    color: getStatusColor(status),
    icon: getStatusIcon(status),
  };
};

const countEntries = (value: unknown, prefix = ''): Array<{ label: string; value: unknown }> => {
  if (!isRecord(value)) return [];

  return Object.entries(value).flatMap(([key, entry]) => {
    const label = prefix ? `${prefix} ${formatKey(key)}` : formatKey(key);
    if (isRecord(entry)) return countEntries(entry, label);
    return [{ label, value: entry }];
  });
};

const summarizeJson = (value: unknown): string => {
  const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2) || '';
  return text.length > 2800 ? `${text.slice(0, 2800)}\n...` : text;
};

const downloadBlob = (blob: Blob, filename: string) => {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
};

const sampleRecords = (sampleOutputs: unknown, key: string): JsonRecord[] =>
  isRecord(sampleOutputs) ? asArray(sampleOutputs[key]).filter(isRecord) : [];

const rankedSampleRecords = (sampleOutputs: unknown, key: string): JsonRecord[] =>
  rankGraphRAGTraceSampleRecords(key, sampleRecords(sampleOutputs, key))
    .map(candidate => candidate.record);

const compactText = (value: unknown, limit = 220): string => {
  const text = typeof value === 'string' ? value.replace(/\s+/g, ' ').trim() : formatValue(value);
  return text.length > limit ? `${text.slice(0, limit - 1).trim()}...` : text;
};

const SampleLinkTargetRow: React.FC<{ record: JsonRecord; dense?: boolean }> = ({ record, dense }) => {
  const target = graphRAGTraceSampleLinkTarget(record);
  if (!target) return null;

  return (
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns: { xs: '1fr', sm: 'auto minmax(0, 1fr)' },
        alignItems: 'baseline',
        gap: { xs: 0.15, sm: 0.55 },
        minWidth: 0,
        mt: 0.6,
        px: 0.65,
        py: 0.45,
        borderRadius: 1,
        bgcolor: theme => alpha(theme.palette.success.main, 0.075),
      }}
    >
      <Typography
        variant="caption"
        noWrap
        sx={{
          color: 'success.main',
          fontSize: dense ? '0.6rem' : '0.64rem',
          fontWeight: 800,
          lineHeight: 1.15,
        }}
      >
        Link target
      </Typography>
      <Typography
        variant="caption"
        title={target.url}
        sx={{
          minWidth: 0,
          display: '-webkit-box',
          overflow: 'hidden',
          WebkitBoxOrient: 'vertical',
          WebkitLineClamp: dense ? { xs: 2, sm: 1 } : 2,
          overflowWrap: 'anywhere',
          wordBreak: 'break-word',
          color: 'text.primary',
          fontSize: dense ? '0.62rem' : '0.66rem',
          fontWeight: 700,
          lineHeight: 1.18,
        }}
      >
        {target.label !== 'External link' ? `${target.label}: ` : ''}
        {target.url}
      </Typography>
    </Box>
  );
};

const formatBytes = (value: unknown): string => {
  const bytes = asNumber(value);
  if (bytes === null) return formatValue(value);
  if (bytes < 1024) return `${bytes.toLocaleString()} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const formatRate = (key: string, value: number): string => {
  if (/bytes?/i.test(key)) return `${formatBytes(value)}/s`;
  if (value >= 100) return `${Math.round(value).toLocaleString()}/s`;
  if (value >= 10) return `${value.toFixed(1)}/s`;
  return `${value.toFixed(2)}/s`;
};

const formatDuration = (durationMs: number): string => {
  if (durationMs < 1000) return `${durationMs.toLocaleString()} ms`;
  return `${(durationMs / 1000).toFixed(durationMs < 10000 ? 1 : 0)} s`;
};

const progressColor = (status: string): 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' => {
  const color = getStatusColor(status);
  return color === 'default' ? 'primary' : color;
};

const positiveCountEntries = (value: unknown): Array<{ key: string; label: string; value: number }> => {
  if (!isRecord(value)) return [];

  return Object.entries(value)
    .map(([key, entry]) => ({ key, label: formatKey(key), value: asNumber(entry) ?? 0 }))
    .filter(item => item.value > 0);
};

const evidenceFamilyValues = (record: JsonRecord): string[] => {
  const metadata = isRecord(record.metadata) ? record.metadata : {};
  const families = [
    ...asStringArray(record.categories),
    ...asStringArray(record.sample_categories),
    ...asStringArray(record.evidence_families),
    ...asStringArray(metadata.sample_categories),
  ];
  const primary = asString(record.category, asString(metadata.sample_category));
  if (primary) families.unshift(primary);
  return [...new Set(families)].filter(Boolean);
};

const evidenceFamilyCountEntries = (sampleOutputs: unknown): Array<{ key: string; label: string; value: number }> => {
  if (!isRecord(sampleOutputs)) return [];
  return positiveCountEntries(sampleOutputs.embedding_evidence_family_counts)
    .sort((left, right) => right.value - left.value || left.label.localeCompare(right.label));
};

const sampleRecordTitle = (key: string, record: JsonRecord, index: number): string => {
  if (key === 'elements') return `${formatValue(record.type || 'element')} ${record.location ? `- ${formatValue(record.location)}` : ''}`;
  if (key === 'chunks') return formatValue(record.chunk_id || `chunk ${index + 1}`);
  if (key === 'chunk_evidence_cards') return `${formatValue(record.category || 'chunk')}: ${formatValue(record.title || record.chunk_id || `chunk ${index + 1}`)}`;
  if (key === 'entities') return `${formatValue(record.name || `entity ${index + 1}`)}${record.type ? ` (${formatValue(record.type)})` : ''}`;
  if (key === 'nodes') return `${formatValue(record.name || record.label || record.id || `node ${index + 1}`)}${record.type ? ` (${formatValue(record.type)})` : ''}`;
  if (key === 'edges') return `${formatValue(record.source)} --[${formatValue(record.label || record.relation || record.type)}]--> ${formatValue(record.target)}`;
  if (key === 'graph_context') return `${formatValue(record.source)} --[${formatValue(record.relation || record.label || record.type)}]--> ${formatValue(record.target)}`;
  if (key === 'relationships') return `${formatValue(record.source)} --[${formatValue(record.relation)}]--> ${formatValue(record.target)}`;
  if (key === 'relationship_evidence_cards') {
    const source = asString(record.source);
    const target = asString(record.target);
    const relation = asString(record.relation, asString(record.label, asString(record.type, 'related')));
    if (source || target) return `${formatValue(source || 'Source')} --[${formatValue(relation)}]--> ${formatValue(target || 'Target')}`;
    return `${formatValue(record.category || 'evidence')}: ${formatValue(record.title || `relationship ${index + 1}`)}`;
  }
  if (key === 'endpoint_attempts') return `${formatValue(record.status || 'attempt')} ${record.status_code ? `(${formatValue(record.status_code)})` : ''}`;
  if (key === 'starter_prompts') return formatValue(record.label || record.id || `prompt ${index + 1}`);
  if (key === 'handoff_lanes') return `${formatValue(record.label || record.id || `lane ${index + 1}`)}: ${record.ready ? 'ready' : 'waiting'}`;
  if (key === 'capability_cards') return `${formatValue(record.label || record.id || `capability ${index + 1}`)}${record.count !== undefined ? `: ${formatValue(record.count)}` : ''}`;
  if (key === 'graph_anchors') return `${formatValue(record.kind || 'anchor')}: ${formatValue(record.title || record.label || `anchor ${index + 1}`)}`;
  if (key === 'retrieval_rehearsals') return `${formatValue(record.label || record.prompt_id || `rehearsal ${index + 1}`)}: ${formatValue(record.status || 'ready')}`;
  if (key === 'bridge_entities') return `Bridge: ${formatValue(record.name || record.label || `entity ${index + 1}`)}`;
  if (key === 'path_candidates' || key === 'graph_paths') {
    const nodes = displayPathNodes(
      asArray(record.nodes).map(item => formatValue(item)).filter(Boolean),
    );
    if (nodes.length >= 2) return nodes.join(' -> ');
    return `${formatValue(record.source)} -> ${formatValue(record.bridge)} -> ${formatValue(record.target)}`;
  }
  if (key === 'embedding_inputs' || key === 'representative_embedding_inputs') return formatValue(record.chunk_id || `embedding ${index + 1}`);
  if (key === 'embedding_evidence_cards') return `${formatValue(record.category || 'embedding')}: ${formatValue(record.title || record.chunk_id || `embedding ${index + 1}`)}`;
  return `${formatKey(key)} ${index + 1}`;
};

const sampleRecordBody = (key: string, record: JsonRecord): string => {
  if (key === 'chunks') {
    const metadata = isRecord(record.metadata) ? record.metadata : {};
    return compactText(metadata.evidence_preview || record.text || record.content || record.metadata || record);
  }
  if (key === 'chunk_evidence_cards') return compactText(record.reason || record.evidence_preview || record.metadata || record);
  if (key === 'nodes') return compactText(record.source || record.metadata || record);
  if (key === 'edges' || key === 'graph_context') return compactText(record.evidence || record.metadata || record.properties || record);
  if (key === 'relationships') return compactText(record.evidence || record.metadata || record);
  if (key === 'relationship_evidence_cards') return compactText(record.reason || record.evidence || record.provenance || record);
  if (key === 'endpoint_attempts') {
    const fallback = record.fallback_allowed ? 'fallback allowed' : 'final attempt';
    return compactText(`${formatValue(record.url)} - ${fallback}`);
  }
  if (key === 'starter_prompts') return compactText(record.prompt || record.label || record);
  if (key === 'handoff_lanes') return compactText(record.detail || record.status || record);
  if (key === 'capability_cards') return compactText(record.detail || record.label || record);
  if (key === 'graph_anchors') return compactText(record.evidence || record.metadata || record.relationships || record.nodes || record);
  if (key === 'retrieval_rehearsals') return compactText(record.prompt || record.matched_terms || record.top_anchors || record.top_chunks || record);
  if (key === 'entities') return compactText(record.evidence || record.score || record);
  if (key === 'bridge_entities') return compactText(record.text || record.relationships || record.neighbours || record);
  if (key === 'path_candidates' || key === 'graph_paths') return compactText(record.explanation || record.answer_hint || record.text || record.relationships || record);
  if (key === 'embedding_evidence_cards') return compactText(record.reason || record.evidence_preview || record);
  return compactText(record.text || record.content || record.metadata || record);
};

const SAMPLE_KEYS = ['relationship_evidence_cards', 'relationships', 'edges', 'graph_context', 'path_candidates', 'graph_paths', 'bridge_entities', 'chunk_evidence_cards', 'chunks', 'embedding_evidence_cards', 'elements', 'entities', 'nodes', 'graph_anchors', 'retrieval_rehearsals', 'starter_prompts', 'handoff_lanes', 'capability_cards', 'embedding_inputs', 'representative_embedding_inputs', 'endpoint_attempts'];

const visibleSampleGroups = (sampleOutputs: unknown): Array<{ key: string; records: JsonRecord[] }> => {
  if (!isRecord(sampleOutputs)) return [];

  return rankGraphRAGTraceSampleGroups(sampleOutputs)
    .filter(group => SAMPLE_KEYS.includes(group.key))
    .filter(group => !['debug', 'operational'].includes(group.role))
    .map(group => ({ key: group.key, records: rankedSampleRecords(sampleOutputs, group.key) }))
    .filter(group => group.records.length > 0);
};

const hasSampleHighlights = (sampleOutputs: unknown): boolean =>
  visibleSampleGroups(sampleOutputs).length > 0
  || evidenceFamilyCountEntries(sampleOutputs).length > 0;

const hasStructuredTextHighlights = (sampleOutputs: unknown): boolean =>
  isRecord(sampleOutputs) && typeof sampleOutputs.text === 'string' && sampleOutputs.text.trim().length > 0;

const hasSuggestedQuestionHighlights = (sampleOutputs: unknown): boolean =>
  sampleRecords(sampleOutputs, 'suggested_questions').length > 0;

const hasProfileHighlights = (sampleOutputs: unknown): boolean => {
  if (!isRecord(sampleOutputs)) return false;
  return (
    asStringArray(sampleOutputs.detected_capabilities).length > 0 ||
    isRecord(sampleOutputs.evidence_profile) ||
    sampleOutputs.readiness_score !== undefined ||
    sampleOutputs.complexity !== undefined ||
    sampleOutputs.recommended_payload !== undefined
  );
};

const RichEvidenceProfileHighlights: React.FC<{ evidenceProfile: unknown; dense?: boolean }> = ({ evidenceProfile, dense }) => {
  if (!isRecord(evidenceProfile)) return null;

  const quickCounts = positiveCountEntries(evidenceProfile.quick_counts);
  const graphCounts = positiveCountEntries(evidenceProfile.graph_evidence)
    .filter(item => item.key !== 'relationship_count');
  const artifacts = asArray(evidenceProfile.source_artifacts).filter(isRecord);
  const visibleArtifacts = artifacts.slice(0, dense ? 2 : 4);

  if (quickCounts.length === 0 && graphCounts.length === 0 && visibleArtifacts.length === 0) return null;

  return (
    <Stack spacing={1}>
      {quickCounts.length > 0 && (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
          {quickCounts.slice(0, dense ? 8 : 12).map(item => (
            <Chip
              key={item.key}
              label={`${item.label}: ${formatValue(item.value)}`}
              size="small"
              color="secondary"
              variant="outlined"
            />
          ))}
        </Box>
      )}

      {graphCounts.length > 0 && (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
          {graphCounts.slice(0, dense ? 5 : 8).map(item => (
            <Chip
              key={item.key}
              label={`${item.label}: ${formatValue(item.value)}`}
              size="small"
              color="primary"
              variant="outlined"
            />
          ))}
        </Box>
      )}

      {visibleArtifacts.length > 0 && (
        <Grid container spacing={1}>
          {visibleArtifacts.map((artifact, index) => {
            const label = asString(artifact.label, asString(artifact.path, asString(artifact.filename, `Source artifact ${index + 1}`)));
            const kind = asString(artifact.kind, 'artifact');
            const details = [
              asString(artifact.mime),
              artifact.size_bytes !== undefined ? formatBytes(artifact.size_bytes) : '',
              artifact.parsed_child_elements !== undefined ? `${formatValue(artifact.parsed_child_elements)} parsed elements` : '',
            ].filter(Boolean).join(' · ');

            return (
              <Grid item xs={12} sm={dense ? 12 : 6} key={`${kind}-${label}-${index}`}>
                <Box
                  sx={{
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: 1,
                    p: 1,
                    bgcolor: theme => alpha(theme.palette.text.primary, 0.03),
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, minWidth: 0 }}>
                    <Chip label={formatKey(kind)} size="small" sx={{ height: 20, flexShrink: 0 }} />
                    <Typography variant="caption" fontWeight={700} noWrap title={label} sx={{ minWidth: 0 }}>
                      {label}
                    </Typography>
                  </Box>
                  {details && (
                    <Typography variant="caption" color="text.secondary" component="div" noWrap title={details} sx={{ mt: 0.5 }}>
                      {details}
                    </Typography>
                  )}
                </Box>
              </Grid>
            );
          })}
        </Grid>
      )}
    </Stack>
  );
};

const ProfileOutputHighlights: React.FC<{ sampleOutputs: unknown }> = ({ sampleOutputs }) => {
  if (!isRecord(sampleOutputs)) return null;

  const capabilities = asStringArray(sampleOutputs.detected_capabilities);
  const metadataFields = asStringArray(sampleOutputs.metadata_fields);
  const evidenceProfile = sampleOutputs.evidence_profile;
  const profileItems = [
    { label: 'Complexity', value: sampleOutputs.complexity },
    { label: 'Readiness', value: sampleOutputs.readiness_score !== undefined ? `${formatValue(sampleOutputs.readiness_score)}%` : undefined },
    { label: 'Payload', value: sampleOutputs.recommended_payload },
  ].filter(item => item.value !== undefined && item.value !== null && item.value !== '');

  return (
    <Stack spacing={1}>
      {profileItems.length > 0 && (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
          {profileItems.map(item => (
            <Chip key={item.label} label={`${item.label}: ${formatValue(item.value)}`} size="small" variant="outlined" />
          ))}
        </Box>
      )}
      {capabilities.length > 0 && (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
          {capabilities.slice(0, 14).map(capability => (
            <Chip key={capability} label={formatKey(capability)} size="small" color="info" variant="outlined" />
          ))}
        </Box>
      )}
      {metadataFields.length > 0 && (
        <Typography variant="caption" color="text.secondary" component="div">
          Metadata: {metadataFields.slice(0, 10).map(formatKey).join(', ')}
        </Typography>
      )}
      <RichEvidenceProfileHighlights evidenceProfile={evidenceProfile} />
    </Stack>
  );
};

const StructuredTextHighlights: React.FC<{ sampleOutputs: unknown; dense?: boolean }> = ({ sampleOutputs, dense }) => {
  if (!isRecord(sampleOutputs) || typeof sampleOutputs.text !== 'string' || !sampleOutputs.text.trim()) return null;

  const text = sampleOutputs.text.trim();
  const preview = text.length > (dense ? 520 : 900)
    ? `${text.slice(0, dense ? 520 : 900).trim()}...`
    : text;
  const meta = [
    { label: 'File', value: sampleOutputs.filename },
    { label: 'Type', value: sampleOutputs.content_type },
    { label: 'Structured', value: sampleOutputs.uses_structured_text },
  ].filter(item => item.value !== undefined && item.value !== null && item.value !== '');

  return (
    <Box
      sx={{
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1,
        p: 1,
        bgcolor: theme => alpha(theme.palette.text.primary, 0.035),
      }}
    >
      {meta.length > 0 && (
        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mb: 0.75 }}>
          {meta.map(item => (
            <Chip
              key={item.label}
              label={`${item.label}: ${formatValue(item.value)}`}
              size="small"
              variant="outlined"
              sx={{ height: 20 }}
            />
          ))}
        </Box>
      )}
      <Box
        component="pre"
        sx={{
          m: 0,
          maxHeight: dense ? 180 : 260,
          overflow: 'auto',
          fontSize: 12,
          lineHeight: 1.45,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {preview}
      </Box>
    </Box>
  );
};

const SuggestedQuestionHighlights: React.FC<{ sampleOutputs: unknown; dense?: boolean }> = ({ sampleOutputs, dense }) => {
  const questions = sampleRecords(sampleOutputs, 'suggested_questions');
  if (questions.length === 0) return null;

  return (
    <Box>
      <Typography variant="caption" color="text.secondary" component="div" sx={{ mb: 0.5 }}>
        Suggested graph questions
      </Typography>
      <Grid container spacing={1}>
        {questions.slice(0, dense ? 3 : 5).map((question, index) => {
          const label = asString(question.label, asString(question.kind, `Question ${index + 1}`));
          const prompt = asString(question.prompt);
          const kind = asString(question.kind, 'question');

          return (
            <Grid item xs={12} sm={dense ? 12 : 6} key={asString(question.id, `${kind}-${index}`)}>
              <Box
                sx={{
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 1,
                  p: 1,
                  bgcolor: theme => alpha(theme.palette.info.main, 0.055),
                  minHeight: 74,
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 0 }}>
                  <Chip label={formatKey(kind)} size="small" color="info" variant="outlined" sx={{ height: 20, flexShrink: 0 }} />
                  <Typography variant="caption" fontWeight={700} noWrap title={label} sx={{ minWidth: 0 }}>
                    {label}
                  </Typography>
                </Box>
                {prompt && (
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    component="div"
                    sx={{ mt: 0.5, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}
                    title={prompt}
                  >
                    {prompt}
                  </Typography>
                )}
              </Box>
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
};

const InteractionHandoffHighlights: React.FC<{
  handoff: JsonRecord;
  dense?: boolean;
  onAskPrompt: (prompt: string) => void;
}> = ({ handoff, dense, onAskPrompt }) => {
  const lanes = asArray(handoff.lanes).filter(isRecord);
  const capabilityCards = asArray(handoff.capability_cards).filter(isRecord);
  const graphAnchors = asArray(handoff.graph_anchors).filter(isRecord);
  const starterPrompts = asArray(handoff.starter_prompts).filter(isRecord);
  const retrievalRehearsals = asArray(handoff.retrieval_rehearsals).filter(isRecord);
  const citationScopes = asStringArray(handoff.citation_scopes);
  const recommendedMode = asString(handoff.recommended_mode);
  const answerGuidance = asString(handoff.answer_guidance);

  if (
    lanes.length === 0
    && capabilityCards.length === 0
    && graphAnchors.length === 0
    && starterPrompts.length === 0
    && retrievalRehearsals.length === 0
  ) {
    return null;
  }

  return (
    <Box
      sx={{
        mt: 1.25,
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1,
        p: 1,
        bgcolor: theme => alpha(theme.palette.success.main, 0.045),
      }}
    >
      <Stack spacing={0.85}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
          <Typography variant="caption" fontWeight={800} color="text.secondary">
            Interaction handoff
          </Typography>
          {recommendedMode && (
            <Chip
              label={formatKey(recommendedMode)}
              size="small"
              color="success"
              variant="outlined"
              sx={{ height: 20 }}
            />
          )}
          {citationScopes.slice(0, dense ? 4 : 8).map(scope => (
            <Chip
              key={scope}
              label={formatKey(scope)}
              size="small"
              variant="outlined"
              sx={{ height: 20 }}
            />
          ))}
        </Box>

        {lanes.length > 0 && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
            {lanes.slice(0, dense ? 5 : 7).map(lane => {
              const label = asString(lane.label, asString(lane.id, 'Lane'));
              const count = lane.count !== undefined ? ` ${formatValue(lane.count)}` : '';
              return (
                <Tooltip key={asString(lane.id, label)} title={asString(lane.detail, asString(lane.status))}>
                  <Chip
                    label={`${label}${count}`}
                    size="small"
                    color={lane.ready ? 'success' : 'default'}
                    variant="outlined"
                    sx={{ height: 22 }}
                  />
                </Tooltip>
              );
            })}
          </Box>
        )}

        {capabilityCards.length > 0 && (
          <Grid container spacing={0.75}>
            {capabilityCards.slice(0, dense ? 3 : 6).map(card => {
              const label = asString(card.label, asString(card.id, 'Capability'));
              return (
                <Grid item xs={12} sm={dense ? 12 : 6} key={asString(card.id, label)}>
                  <Box
                    sx={{
                      border: '1px solid',
                      borderColor: 'divider',
                      borderRadius: 1,
                      p: 0.75,
                      bgcolor: theme => alpha(theme.palette.background.paper, 0.52),
                    }}
                  >
                    <Typography variant="caption" fontWeight={700} component="div" noWrap title={label}>
                      {label}{card.count !== undefined ? `: ${formatValue(card.count)}` : ''}
                    </Typography>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      component="div"
                      sx={{ mt: 0.25, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}
                      title={asString(card.detail)}
                    >
                      {asString(card.detail)}
                    </Typography>
                  </Box>
                </Grid>
              );
            })}
          </Grid>
        )}

        {graphAnchors.length > 0 && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
            {graphAnchors.slice(0, dense ? 4 : 8).map(anchor => {
              const kind = asString(anchor.kind, 'anchor');
              const label = graphAnchorLabel(anchor);
              return (
                <Tooltip key={asString(anchor.id, label)} title={graphAnchorTooltip(anchor) || label}>
                  <Chip
                    label={`${formatKey(kind)}: ${label}`}
                    size="small"
                    color={graphAnchorColor(kind)}
                    variant="outlined"
                    sx={{
                      height: 22,
                      maxWidth: '100%',
                      '& .MuiChip-label': {
                        display: 'block',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                      },
                    }}
                  />
                </Tooltip>
              );
            })}
          </Box>
        )}

        {starterPrompts.length > 0 && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
            {starterPrompts.slice(0, dense ? 2 : 4).map(prompt => {
              const label = asString(prompt.label, asString(prompt.kind, 'Ask Chat'));
              const promptText = asString(prompt.prompt);
              if (!promptText) return null;
              return (
                <Tooltip key={asString(prompt.id, label)} title={promptText} describeChild>
                  <Button
                    size="small"
                    variant={asString(prompt.kind) === 'summary' ? 'contained' : 'outlined'}
                    color={asString(prompt.kind) === 'summary' ? 'primary' : 'inherit'}
                    startIcon={<ChatIcon fontSize="small" />}
                    onClick={() => onAskPrompt(promptText)}
                    sx={{
                      minHeight: 28,
                      maxWidth: { xs: '100%', sm: 260 },
                      borderRadius: 1,
                      textTransform: 'none',
                      justifyContent: 'flex-start',
                      '& .MuiButton-startIcon': { flexShrink: 0 },
                    }}
                  >
                    <Box component="span" sx={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {label}
                    </Box>
                  </Button>
                </Tooltip>
              );
            })}
          </Box>
        )}

        {retrievalRehearsals.length > 0 && (
          <Grid container spacing={0.75}>
            {retrievalRehearsals.slice(0, dense ? 2 : 4).map(rehearsal => {
              const label = asString(rehearsal.label, asString(rehearsal.prompt_id, 'Retrieval rehearsal'));
              const matchedTerms = asStringArray(rehearsal.matched_terms);
              const anchorCount = asNumber(rehearsal.matched_anchor_count) ?? 0;
              const chunkCount = asNumber(rehearsal.matched_chunk_count) ?? 0;
              const prompt = asString(rehearsal.prompt);

              return (
                <Grid item xs={12} sm={dense ? 12 : 6} key={asString(rehearsal.id, label)}>
                  <Tooltip title={prompt || label}>
                    <Box
                      sx={{
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 1,
                        p: 0.75,
                        bgcolor: theme => alpha(theme.palette.info.main, 0.045),
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 0 }}>
                        <Typography variant="caption" fontWeight={700} noWrap title={label} sx={{ minWidth: 0 }}>
                          {label}
                        </Typography>
                        <Chip
                          label={formatKey(asString(rehearsal.status, 'ready'))}
                          size="small"
                          color={asString(rehearsal.status) === 'ready' ? 'success' : 'default'}
                          variant="outlined"
                          sx={{ height: 18, fontSize: '0.6rem', ml: 'auto' }}
                        />
                      </Box>
                      <Box sx={{ display: 'flex', gap: 0.35, flexWrap: 'wrap', mt: 0.5 }}>
                        <Chip label={`${formatValue(anchorCount)} anchors`} size="small" variant="outlined" sx={{ height: 18, fontSize: '0.6rem' }} />
                        <Chip label={`${formatValue(chunkCount)} chunks`} size="small" variant="outlined" sx={{ height: 18, fontSize: '0.6rem' }} />
                        {matchedTerms.slice(0, dense ? 2 : 4).map(term => (
                          <Chip key={term} label={formatKey(term)} size="small" sx={{ height: 18, fontSize: '0.6rem' }} />
                        ))}
                      </Box>
                    </Box>
                  </Tooltip>
                </Grid>
              );
            })}
          </Grid>
        )}

        {answerGuidance && !dense && (
          <Typography variant="caption" color="text.secondary" component="div">
            {answerGuidance}
          </Typography>
        )}
      </Stack>
    </Box>
  );
};

const SampleOutputHighlights: React.FC<{ sampleOutputs: unknown; dense?: boolean }> = ({ sampleOutputs, dense }) => {
  const groups = visibleSampleGroups(sampleOutputs);
  const familyCounts = evidenceFamilyCountEntries(sampleOutputs);

  if (groups.length === 0 && familyCounts.length === 0) return null;

  return (
    <Stack spacing={1}>
      {familyCounts.length > 0 && (
        <Box>
          <Typography variant="caption" color="text.secondary" component="div" sx={{ mb: 0.5 }}>
            Evidence families
          </Typography>
          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
            {familyCounts.slice(0, dense ? 8 : 14).map(item => (
              <Chip
                key={item.key}
                label={`${item.label}: ${formatValue(item.value)}`}
                size="small"
                color="info"
                variant="outlined"
                sx={{ height: 20, fontSize: '0.68rem' }}
              />
            ))}
          </Box>
        </Box>
      )}
      {groups.map(group => (
        <Box key={group.key}>
          <Typography variant="caption" color="text.secondary" component="div" sx={{ mb: 0.5 }}>
            {formatKey(group.key.replace(/_evidence_cards$/, '').replace(/^graph_context$/, 'relationships'))}
          </Typography>
          <Grid container spacing={1}>
            {group.records.slice(0, dense ? 2 : 4).map((record, index) => {
              const matchScore = asNumber(record.chunk_match_score);
              const matchReasons = asStringArray(record.chunk_match_reasons);
              const reasoningScore = ['bridge_entities', 'path_candidates', 'graph_paths'].includes(group.key) ? asNumber(record.score) : null;
              const relationshipCount = group.key === 'bridge_entities' ? asNumber(record.relationship_count) : null;
              const pathCount = group.key === 'bridge_entities' ? asNumber(record.path_count) : null;
              const hops = ['path_candidates', 'graph_paths'].includes(group.key) ? asNumber(record.hops) : null;
              const evidenceFamilies = ['chunk_evidence_cards', 'embedding_evidence_cards', 'representative_embedding_inputs', 'path_candidates', 'graph_paths'].includes(group.key)
                ? evidenceFamilyValues(record)
                : [];

              return (
                <Grid item xs={12} sm={['chunk_evidence_cards', 'relationship_evidence_cards', 'relationships', 'edges', 'graph_context', 'path_candidates', 'graph_paths', 'representative_embedding_inputs', 'embedding_evidence_cards'].includes(group.key) ? 12 : 6} key={`${group.key}-${index}`}>
                  <Box
                    sx={{
                      border: '1px solid',
                      borderColor: 'divider',
                      borderRadius: 1,
                      p: 1,
                      bgcolor: theme => alpha(theme.palette.text.primary, 0.035),
                      minHeight: 72,
                    }}
                  >
                    <Typography variant="caption" fontWeight={700} component="div" noWrap title={sampleRecordTitle(group.key, record, index)}>
                      {sampleRecordTitle(group.key, record, index)}
                    </Typography>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      component="div"
                      sx={{ mt: 0.25, display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}
                      title={sampleRecordBody(group.key, record)}
                    >
                      {sampleRecordBody(group.key, record)}
                    </Typography>
                    <SampleLinkTargetRow record={record} dense={dense} />
                    {(matchScore !== null || matchReasons.length > 0 || reasoningScore !== null || relationshipCount !== null || pathCount !== null || hops !== null) && (
                      <Box sx={{ display: 'flex', gap: 0.35, flexWrap: 'wrap', mt: 0.75 }}>
                        {matchScore !== null && (
                          <Chip label={`Match ${formatValue(matchScore)}`} size="small" color="success" variant="outlined" sx={{ height: 18, fontSize: '0.6rem' }} />
                        )}
                        {reasoningScore !== null && (
                          <Chip label={`Score ${formatValue(reasoningScore)}`} size="small" color="success" variant="outlined" sx={{ height: 18, fontSize: '0.6rem' }} />
                        )}
                        {relationshipCount !== null && (
                          <Chip label={`${formatValue(relationshipCount)} links`} size="small" variant="outlined" sx={{ height: 18, fontSize: '0.6rem' }} />
                        )}
                        {pathCount !== null && (
                          <Chip label={`${formatValue(pathCount)} paths`} size="small" variant="outlined" sx={{ height: 18, fontSize: '0.6rem' }} />
                        )}
                        {hops !== null && (
                          <Chip label={`${formatValue(hops)} hops`} size="small" variant="outlined" sx={{ height: 18, fontSize: '0.6rem' }} />
                        )}
                        {matchReasons.slice(0, dense ? 2 : 4).map(reason => (
                          <Chip
                            key={reason}
                            label={formatKey(reason)}
                            size="small"
                            variant="outlined"
                            sx={{ height: 18, fontSize: '0.6rem' }}
                          />
                        ))}
                      </Box>
                    )}
                    {evidenceFamilies.length > 0 && (
                      <Box sx={{ display: 'flex', gap: 0.35, flexWrap: 'wrap', mt: 0.75 }}>
                        {evidenceFamilies.slice(0, dense ? 4 : 8).map(family => (
                          <Chip
                            key={family}
                            label={formatKey(family)}
                            size="small"
                            color="info"
                            variant="outlined"
                            sx={{ height: 18, fontSize: '0.6rem' }}
                          />
                        ))}
                      </Box>
                    )}
                  </Box>
                </Grid>
              );
            })}
          </Grid>
        </Box>
      ))}
    </Stack>
  );
};

const PipelineTimeline: React.FC<{ timeline: GraphRAGTraceTimeline; dense?: boolean }> = ({ timeline, dense }) => {
  if (timeline.stageCount === 0) return null;

  const visibleStages = timeline.stages.slice(0, dense ? 6 : 10);
  const hiddenStageCount = Math.max(0, timeline.stages.length - visibleStages.length);

  return (
    <Box
      sx={{
        mt: 1.25,
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1,
        p: 1,
        bgcolor: theme => alpha(theme.palette.text.primary, 0.025),
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap', mb: 1 }}>
        <Typography variant="caption" fontWeight={800} color="text.secondary">
          Pipeline timeline
        </Typography>
        <Chip label={`${timeline.stageCount} stages`} size="small" variant="outlined" sx={{ height: 20 }} />
        <Chip label={`Total ${formatDuration(timeline.totalDurationMs)}`} size="small" variant="outlined" sx={{ height: 20 }} />
        {timeline.sampleStageCount > 0 && (
          <Chip label={`${timeline.sampleStageCount} with evidence`} size="small" color="info" variant="outlined" sx={{ height: 20 }} />
        )}
        {timeline.warningStageCount > 0 && (
          <Chip label={`${timeline.warningStageCount} warning${timeline.warningStageCount === 1 ? '' : 's'}`} size="small" color="warning" variant="outlined" sx={{ height: 20 }} />
        )}
        {timeline.failedStageCount > 0 && (
          <Chip label={`${timeline.failedStageCount} failed`} size="small" color="error" variant="outlined" sx={{ height: 20 }} />
        )}
        {timeline.slowestStage && timeline.slowestStage.durationMs > 0 && (
          <Tooltip title={`${timeline.slowestStage.label}: ${formatDuration(timeline.slowestStage.durationMs)}`}>
            <Chip
              label={`Slowest ${formatCompactStageName(timeline.slowestStage.name || timeline.slowestStage.label)}`}
              size="small"
              variant="outlined"
              sx={{ height: 20, maxWidth: '100%' }}
            />
          </Tooltip>
        )}
      </Box>

      <Stack spacing={0.85}>
        {visibleStages.map(stage => (
          <Box key={stage.id} sx={{ minWidth: 0 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mb: 0.35, minWidth: 0 }}>
              <Chip
                label={stage.statusLabel}
                color={getStatusColor(stage.status)}
                size="small"
                sx={{ height: 20, flexShrink: 0, fontSize: '0.68rem' }}
              />
              <Typography variant="caption" fontWeight={700} noWrap title={stage.label} sx={{ minWidth: 0 }}>
                {stage.label}
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ ml: 'auto', flexShrink: 0 }}>
                {formatDuration(stage.durationMs)}
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0 }}>
                {stage.percentOfTotal.toFixed(stage.percentOfTotal % 1 === 0 ? 0 : 1)}%
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={Math.max(3, Math.min(100, stage.percentOfTotal))}
              color={progressColor(stage.status)}
              sx={{ height: 5, borderRadius: 1 }}
            />
            {(stage.hasSamples || stage.warningCount > 0) && (
              <Box sx={{ display: 'flex', gap: 0.35, flexWrap: 'wrap', mt: 0.45 }}>
                {stage.hasSamples && (
                  <Tooltip title={stage.sampleGroups.slice(0, 8).join(', ')}>
                    <Chip
                      label={`${stage.sampleItemCount} sample${stage.sampleItemCount === 1 ? '' : 's'}`}
                      size="small"
                      variant="outlined"
                      sx={{ height: 18, fontSize: '0.6rem' }}
                    />
                  </Tooltip>
                )}
                {stage.warningCount > 0 && (
                  <Chip
                    label={`${stage.warningCount} issue${stage.warningCount === 1 ? '' : 's'}`}
                    size="small"
                    color="warning"
                    variant="outlined"
                    sx={{ height: 18, fontSize: '0.6rem' }}
                  />
                )}
                {stage.sampleGroups.slice(0, dense ? 2 : 4).map(group => (
                  <Chip
                    key={`${stage.id}-${group}`}
                    label={group}
                    size="small"
                    sx={{ height: 18, fontSize: '0.6rem', bgcolor: theme => alpha(theme.palette.info.main, 0.08) }}
                  />
                ))}
              </Box>
            )}
          </Box>
        ))}
        {hiddenStageCount > 0 && (
          <Typography variant="caption" color="text.secondary">
            {hiddenStageCount} more stage{hiddenStageCount === 1 ? '' : 's'} in the detailed trace.
          </Typography>
        )}
      </Stack>
    </Box>
  );
};

const DeploymentOutcomeCard: React.FC<{ deployment: GraphRAGDeploymentOutcomeRollup; dense?: boolean }> = ({ deployment, dense }) => {
  if (!deployment.available) return null;

  const endpoint = deployment.selectedEndpoint || deployment.targetUrl;
  const showLinks = deployment.servicesStatusPath || deployment.diagnosticsPath;
  const chips = [
    {
      label: deployment.remoteSubmitted ? 'Remote submitted' : deployment.remoteGraphAvailable ? 'Remote ready' : 'Remote waiting',
      color: deployment.remoteSubmitted || deployment.remoteGraphAvailable ? 'success' : 'warning',
    },
    deployment.fallbackUsed
      ? { label: 'Endpoint fallback used', color: 'info' }
      : null,
    deployment.endpointAttemptCount
      ? { label: `${formatValue(deployment.endpointAttemptCount)} endpoint attempts`, color: deployment.failedAttemptCount ? 'warning' : 'info' }
      : null,
    deployment.failedAttemptCount
      ? { label: `${formatValue(deployment.failedAttemptCount)} failed attempts`, color: 'error' }
      : null,
    deployment.healthAttemptCount
      ? { label: `${formatValue(deployment.healthAttemptCount)} health probe${deployment.healthAttemptCount === 1 ? '' : 's'}`, color: 'info' }
      : null,
    deployment.payloadKind
      ? { label: formatKey(deployment.payloadKind), color: 'default' }
      : null,
    deployment.payloadBytes !== undefined
      ? { label: formatBytes(deployment.payloadBytes), color: 'default' }
      : null,
  ].filter(Boolean) as Array<{ label: string; color: ChipColor }>;

  return (
    <Alert
      severity={deployment.severity}
      icon={getStatusIcon(deployment.status)}
      sx={{ mt: 1.25 }}
    >
      <Stack spacing={0.75}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
          <Typography variant="body2" fontWeight={800}>
            Deployment outcome
          </Typography>
          <Chip
            label={deployment.label || formatStageName(deployment.status)}
            color={getStatusColor(deployment.status)}
            size="small"
            sx={{ height: 22 }}
          />
          {deployment.primaryAction && (
            <Chip
              label={formatKey(deployment.primaryAction)}
              color={deployment.ready ? 'success' : 'default'}
              variant="outlined"
              size="small"
              sx={{ height: 22 }}
            />
          )}
        </Box>

        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
          {chips.slice(0, dense ? 5 : 8).map(chip => (
            <Chip
              key={chip.label}
              label={chip.label}
              color={chip.color}
              size="small"
              variant="outlined"
            />
          ))}
        </Box>

        {(endpoint || deployment.healthUrl) && (
          <Grid container spacing={0.75}>
            {deployment.healthUrl && (
              <Grid item xs={12} sm={dense ? 12 : 6}>
                <Typography variant="caption" color="text.secondary" component="div">
                  Health
                </Typography>
                <Typography variant="caption" component="div" noWrap title={deployment.healthUrl}>
                  Health check configured
                </Typography>
              </Grid>
            )}
            {endpoint && (
              <Grid item xs={12} sm={dense ? 12 : 6}>
                <Typography variant="caption" color="text.secondary" component="div">
                  Ingest
                </Typography>
                <Typography variant="caption" component="div" noWrap title={endpoint}>
                  Ingest route configured
                </Typography>
              </Grid>
            )}
          </Grid>
        )}

        {deployment.nextStep && (
          <Typography variant="caption" color="text.secondary" component="div">
            {deployment.nextStep}
          </Typography>
        )}

        {showLinks && !dense && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
            {deployment.servicesStatusPath && (
              <Chip label={`Status ${deployment.servicesStatusPath}`} size="small" variant="outlined" sx={{ height: 20 }} />
            )}
            {deployment.diagnosticsPath && (
              <Chip label={`Diagnostics ${deployment.diagnosticsPath}`} size="small" variant="outlined" sx={{ height: 20 }} />
            )}
          </Box>
        )}
      </Stack>
    </Alert>
  );
};

const PerformanceRollupCard: React.FC<{ performance: GraphRAGPipelinePerformanceRollup; dense?: boolean }> = ({ performance, dense }) => {
  if (!performance.available) return null;

  const throughputEntries = Object.entries(performance.throughput || {})
    .filter(([, value]) => value > 0)
    .slice(0, dense ? 2 : 4);
  const visibleTimings = performance.timings.slice(0, dense ? 4 : 6);

  return (
    <Box
      sx={{
        mt: 1.25,
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1,
        p: 1,
        bgcolor: theme => alpha(theme.palette.primary.main, 0.035),
      }}
    >
      <Stack spacing={0.85}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
          <Typography variant="caption" fontWeight={800} color="text.secondary">
            Upload performance
          </Typography>
          <Chip label={`${formatValue(performance.stageCount)} stages`} size="small" variant="outlined" sx={{ height: 20 }} />
          <Chip label={`Total ${formatDuration(performance.totalDurationMs)}`} size="small" variant="outlined" sx={{ height: 20 }} />
          {performance.localStageDurationMs > 0 && (
            <Chip label={`Local ${formatDuration(performance.localStageDurationMs)}`} size="small" color="info" variant="outlined" sx={{ height: 20 }} />
          )}
          {performance.remoteStageDurationMs > 0 && (
            <Chip label={`Remote ${formatDuration(performance.remoteStageDurationMs)}`} size="small" color="secondary" variant="outlined" sx={{ height: 20 }} />
          )}
          {performance.slowestStage && (
            <Tooltip title={`${formatStageName(performance.slowestStage.name)}: ${formatDuration(performance.slowestStage.durationMs)}`}>
              <Chip
                label={`Slowest ${formatCompactStageName(performance.slowestStage.name)}`}
                size="small"
                variant="outlined"
                sx={{ height: 20, maxWidth: '100%' }}
              />
            </Tooltip>
          )}
        </Box>

        {throughputEntries.length > 0 && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
            {throughputEntries.map(([key, value]) => (
              <Chip
                key={key}
                label={`${formatKey(key.replace(/_per_second$/, ''))}: ${formatRate(key, value)}`}
                size="small"
                variant="outlined"
                sx={{ height: 20 }}
              />
            ))}
          </Box>
        )}

        {visibleTimings.length > 0 && !dense && (
          <Grid container spacing={0.75}>
            {visibleTimings.map(timing => (
              <Grid item xs={12} sm={6} md={4} key={`${timing.name}-${timing.status}`}>
                <Box sx={{ minWidth: 0 }}>
                  <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center', minWidth: 0 }}>
                    <Chip label={formatStageName(timing.status)} color={getStatusColor(timing.status)} size="small" sx={{ height: 18, fontSize: '0.6rem' }} />
                    <Typography variant="caption" noWrap title={formatStageName(timing.name)} sx={{ minWidth: 0, fontWeight: 700 }}>
                      {formatStageName(timing.name)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ ml: 'auto', flexShrink: 0 }}>
                      {formatDuration(timing.durationMs)}
                    </Typography>
                  </Box>
                </Box>
              </Grid>
            ))}
          </Grid>
        )}
      </Stack>
    </Box>
  );
};

const getStages = (result: JsonRecord): JsonRecord[] =>
  asArray(result.stages).filter(isRecord);

const getSummary = (result: JsonRecord): JsonRecord =>
  isRecord(result.summary) ? result.summary : {};

const getDocumentProfile = (result: JsonRecord): JsonRecord =>
  isRecord(result.document_profile) ? result.document_profile : {};

const getDocument = (result: JsonRecord): JsonRecord =>
  isRecord(result.document) ? result.document : {};

const getWarnings = (stageOrSummary: JsonRecord): string[] =>
  asArray(stageOrSummary.warnings)
    .map(warning => asString(warning))
    .filter(Boolean);

export const GraphRAGPipelineTrace: React.FC<GraphRAGPipelineTraceProps> = ({
  result,
  title = 'GraphRAG pipeline trace',
  dense = false,
}) => {
  const navigate = useNavigate();
  const toast = useToast();
  const [downloadingArtifactKind, setDownloadingArtifactKind] = React.useState<string | null>(null);
  if (!isRecord(result)) return null;

  const status = asString(result.status, 'unknown');
  const stages = getStages(result);
  const timeline = buildGraphRAGTraceTimeline(result);
  const rollups = normalizeGraphRAGPipelineRunRollups(result);
  const summary = getSummary(result);
  const documentProfile = getDocumentProfile(result);
  const document = getDocument(result);
  const filename = asString(document.filename, asString(result.filename, 'Document'));
  const domain = asString(document.domain);
  const runId = asString(result.run_id);
  const localGraphSummary = isRecord(summary.local_graph) ? summary.local_graph : isRecord(result.local_graph) ? result.local_graph : {};
  const localContextSummary = isRecord(summary.local_context) ? summary.local_context : isRecord(result.local_context) ? result.local_context : {};
  const localGraphReasoningSummary = isRecord(summary.local_graph_reasoning)
    ? summary.local_graph_reasoning
    : isRecord(result.local_graph_reasoning)
      ? result.local_graph_reasoning
      : {};
  const operationalHandoff = isRecord(result.operational_handoff) ? result.operational_handoff : {};
  const hasOperationalHandoff = Object.keys(operationalHandoff).length > 0;
  const handoffStatus = asString(operationalHandoff.status, 'preview_only');
  const handoffLabel = asString(operationalHandoff.label, formatStageName(handoffStatus));
  const handoffMessage = asString(operationalHandoff.message);
  const handoffNextStep = asString(operationalHandoff.next_step);
  const handoffPayloadKind = asString(operationalHandoff.payload_kind);
  const handoffActions = asArray(operationalHandoff.actions).filter(isRecord);
  const handoffLocalTraceReady = Boolean(operationalHandoff.local_trace_ready);
  const handoffChatReady = Boolean(operationalHandoff.chat_handoff_ready);
  const handoffRemoteIngestReady = Boolean(operationalHandoff.remote_ingest_ready);
  const handoffRemoteSubmitted = Boolean(operationalHandoff.remote_ingest_submitted);
  const displayStatus = graphRAGTraceDisplayStatus({
    status,
    localTraceReady: handoffLocalTraceReady,
    chatReady: handoffChatReady,
    remoteSubmitted: handoffRemoteSubmitted,
    remoteReady: handoffRemoteIngestReady,
  });
  const interactionHandoff = isRecord(result.interaction_handoff) ? result.interaction_handoff : {};
  const hasInteractionHandoff = Object.keys(interactionHandoff).length > 0;
  const artifactManifest = buildGraphRAGPipelineArtifactManifest(result);
  const exportableArtifacts = normalizeGraphRAGPipelineArtifactDownloads(artifactManifest, {
    runId,
    documentFilename: filename,
  });
  const traceSampleOutputs = stages
    .map(stage => stage.sample_outputs)
    .filter(isRecord);
  const warnings = getWarnings(summary);
  const complexity = documentProfile.complexity ?? summary.document_complexity;
  const readinessScore = documentProfile.readiness_score ?? summary.readiness_score;
  const detectedCapabilities = asStringArray(documentProfile.detected_capabilities).length > 0
    ? asStringArray(documentProfile.detected_capabilities)
    : asStringArray(summary.detected_capabilities);
  const suggestedQuestions = buildGraphRAGTraceQuestions(result, dense ? 3 : 4);
  const handleAskChat = (prompt = buildGraphRAGTracePrompt(filename, summary, documentProfile, traceSampleOutputs, artifactManifest)) => {
    navigate(getRoutePath('chat'), {
      state: {
        graphragHandoff: {
          ...buildGraphRAGTraceChatHandoff(result, {
            prompt,
            questionLimit: dense ? 3 : 6,
          }),
          runId: runId || undefined,
          filename,
          domain: domain || undefined,
          prompt,
        },
      },
    });
  };
  const handleDownloadArtifact = async (artifact: GraphRAGPipelineArtifactDownload) => {
    if (!runId) return;
    setDownloadingArtifactKind(artifact.kind);
    try {
      const blob = await apiService.exportGraphRAGIngestionPipelineArtifact(runId, artifact.kind, true);
      downloadBlob(blob, artifact.downloadFilename);
      toast.success(`${artifact.label} saved as ${artifact.downloadFilename}`, 'GraphRAG artifact exported');
    } catch (error) {
      const message = error instanceof Error ? error.message : `Failed to export ${artifact.label}`;
      toast.error(message, 'GraphRAG artifact export failed');
    } finally {
      setDownloadingArtifactKind(null);
    }
  };
  const summaryItems = [
    { label: 'Extraction', value: summary.extraction_method },
    { label: 'Complexity', value: complexity },
    { label: 'Readiness', value: readinessScore !== undefined && readinessScore !== null ? `${formatValue(readinessScore)}%` : undefined },
    { label: 'Elements', value: summary.element_count },
    { label: 'Chunks', value: summary.chunk_count },
    { label: 'Entities', value: summary.entity_candidate_count },
    { label: 'Relationships', value: summary.relationship_candidate_count },
    { label: 'Graph nodes', value: localGraphSummary.node_count },
    { label: 'Graph edges', value: localGraphSummary.edge_count ?? localGraphSummary.relationship_count },
    { label: 'Graph context', value: localContextSummary.graph_context_relationship_count },
    { label: 'Graph paths', value: localContextSummary.graph_path_context_count },
    { label: 'Bridge entities', value: localGraphReasoningSummary.bridge_entity_count },
    { label: 'Path candidates', value: localGraphReasoningSummary.path_candidate_count },
  ].filter(item => item.value !== undefined && item.value !== null && item.value !== '');

  return (
    <Box sx={{ mt: dense ? 1.5 : 2 }}>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          gap: 1,
          mb: 1,
          flexWrap: { xs: 'wrap', sm: 'nowrap' },
        }}
      >
        <Box sx={{ minWidth: 0, flex: { xs: '1 1 100%', sm: '1 1 auto' } }}>
          <Typography
            variant={dense ? 'subtitle2' : 'subtitle1'}
            fontWeight={700}
            noWrap
            title={title}
            sx={{
              fontSize: { xs: dense ? '0.86rem' : '0.95rem', sm: undefined },
              lineHeight: 1.22,
            }}
          >
            {title}
          </Typography>
          <Typography variant="caption" color="text.secondary" component="div" noWrap title={filename}>
            {filename}{domain ? ` in ${domain}` : ''}
          </Typography>
          {runId && (
            <Typography variant="caption" color="text.secondary" component="div" noWrap title={runId}>
              Import trace available
            </Typography>
          )}
        </Box>
        <Chip
          icon={displayStatus.icon}
          label={displayStatus.label}
          color={displayStatus.color}
          size="small"
          sx={{ flexShrink: 0, fontWeight: 700 }}
        />
        {runId && (
          <Button
            size="small"
            variant="outlined"
            startIcon={<ChatIcon fontSize="small" />}
            onClick={() => handleAskChat()}
            sx={{ flexShrink: 0, borderRadius: 1, minHeight: 28 }}
          >
            Ask Chat
          </Button>
        )}
      </Box>

      {summaryItems.length > 0 && (
        <Grid container spacing={1}>
          {summaryItems.map(item => (
            <Grid item xs={6} sm={dense ? 4 : 3} key={item.label}>
              <Box
                sx={{
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 1,
                  px: 1,
                  py: 0.75,
                  minHeight: 56,
                  bgcolor: alpha('#1976d2', 0.035),
                }}
              >
                <Typography variant="caption" color="text.secondary" component="div">
                  {item.label}
                </Typography>
                <Typography variant="body2" fontWeight={700} noWrap title={formatValue(item.value)}>
                  {formatValue(item.value)}
                </Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
      )}

      <DeploymentOutcomeCard deployment={rollups.deployment} dense={dense} />
      <PerformanceRollupCard performance={rollups.performance} dense={dense} />

      {hasOperationalHandoff && (
        <Alert
          severity={getAlertSeverity(handoffStatus)}
          icon={getStatusIcon(handoffStatus)}
          sx={{ mt: 1.25 }}
        >
          <Stack spacing={0.75}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
              <Typography variant="body2" fontWeight={700}>
                {handoffLabel}
              </Typography>
              <Chip
                label={formatStageName(handoffStatus)}
                color={getStatusColor(handoffStatus)}
                size="small"
                sx={{ height: 22 }}
              />
            </Box>
            {handoffMessage && (
              <Typography variant="body2">
                {handoffMessage}
              </Typography>
            )}
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
              <Chip
                label={`Trace ${handoffLocalTraceReady ? 'ready' : 'pending'}`}
                color={handoffLocalTraceReady ? 'success' : 'default'}
                size="small"
                variant="outlined"
              />
              <Chip
                label={`Chat ${handoffChatReady ? 'ready' : 'offline'}`}
                color={handoffChatReady ? 'success' : 'warning'}
                size="small"
                variant="outlined"
              />
              <Chip
                label={`Remote ingest ${handoffRemoteSubmitted ? 'submitted' : handoffRemoteIngestReady ? 'ready' : 'waiting'}`}
                color={handoffRemoteSubmitted || handoffRemoteIngestReady ? 'success' : 'warning'}
                size="small"
                variant="outlined"
              />
              {handoffPayloadKind && (
                <Chip
                  label={formatKey(handoffPayloadKind)}
                  color="info"
                  size="small"
                  variant="outlined"
                />
              )}
            </Box>
            {(handoffNextStep || handoffActions.length > 0) && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
                {handoffNextStep && (
                  <Typography variant="caption" color="text.secondary" sx={{ mr: 0.5 }}>
                    {handoffNextStep}
                  </Typography>
                )}
                {runId && handoffChatReady && (
                  <Button
                    size="small"
                    variant="contained"
                    startIcon={<ChatIcon fontSize="small" />}
                    onClick={() => handleAskChat()}
                    sx={{ borderRadius: 1, minHeight: 28 }}
                  >
                    Ask Chat
                  </Button>
                )}
                {handoffActions
                  .filter(action => asString(action.id) !== 'ask_chat')
                  .slice(0, 3)
                  .map(action => (
                    <Chip
                      key={asString(action.id, asString(action.label))}
                      label={asString(action.label, 'Action')}
                      size="small"
                      color={action.ready ? 'success' : 'default'}
                      variant="outlined"
                    />
                  ))}
              </Box>
            )}
          </Stack>
        </Alert>
      )}

      {hasInteractionHandoff && (
        <InteractionHandoffHighlights
          handoff={interactionHandoff}
          dense={dense}
          onAskPrompt={handleAskChat}
        />
      )}

      {exportableArtifacts.length > 0 && (
        <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 0.75, mt: 1 }}>
          <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 700 }}>
            Artifacts
          </Typography>
          {exportableArtifacts.slice(0, dense ? 6 : 10).map(artifact => {
            const isDownloading = downloadingArtifactKind === artifact.kind;

            return (
              <Tooltip key={artifact.kind} title={artifact.detail || artifact.label} describeChild>
                <span>
                  <Button
                    size="small"
                    variant={artifact.kind === 'structured_ingest' ? 'contained' : 'outlined'}
                    color={artifact.kind === 'structured_ingest' ? 'primary' : 'inherit'}
                    startIcon={isDownloading ? <CircularProgress size={14} /> : <DownloadIcon fontSize="small" />}
                    disabled={!runId || downloadingArtifactKind !== null}
                    onClick={() => handleDownloadArtifact(artifact)}
                    sx={{
                      minHeight: 26,
                      maxWidth: { xs: '100%', sm: 230 },
                      borderRadius: 1,
                      textTransform: 'none',
                      px: 1,
                      '& .MuiButton-startIcon': { flexShrink: 0, mr: 0.5 },
                    }}
                  >
                    <Box
                      component="span"
                      sx={{
                        minWidth: 0,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {artifact.shortLabel}
                    </Box>
                  </Button>
                </span>
              </Tooltip>
            );
          })}
          {exportableArtifacts.length > (dense ? 6 : 10) && (
            <Chip
              label={`+${exportableArtifacts.length - (dense ? 6 : 10)} more`}
              size="small"
              variant="outlined"
              sx={{ height: 24 }}
            />
          )}
        </Box>
      )}

      {warnings.length > 0 && (
        <Alert severity="warning" sx={{ mt: 1 }}>
          {warnings.slice(0, 3).join(' ')}
        </Alert>
      )}

      {detectedCapabilities.length > 0 && (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mt: 1 }}>
          {detectedCapabilities.slice(0, dense ? 8 : 14).map(capability => (
            <Chip key={capability} label={formatKey(capability)} size="small" color="info" variant="outlined" />
          ))}
        </Box>
      )}

      <Box sx={{ mt: 1 }}>
        <RichEvidenceProfileHighlights evidenceProfile={documentProfile.evidence_profile} dense={dense} />
      </Box>

      <PipelineTimeline timeline={timeline} dense={dense} />

      {runId && suggestedQuestions.length > 0 && (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mt: 1 }}>
          {suggestedQuestions.map(question => (
            <Tooltip key={question.id} title={question.prompt} describeChild>
              <Button
                size="small"
                variant={question.kind === 'summary' ? 'contained' : 'outlined'}
                color={question.kind === 'summary' ? 'primary' : 'inherit'}
                startIcon={<ChatIcon fontSize="small" />}
                onClick={() => handleAskChat(question.prompt)}
                sx={{
                  minHeight: 28,
                  maxWidth: { xs: '100%', sm: 260 },
                  borderRadius: 1,
                  textTransform: 'none',
                  justifyContent: 'flex-start',
                  '& .MuiButton-startIcon': { flexShrink: 0 },
                }}
              >
                <Box component="span" sx={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {question.label}
                </Box>
              </Button>
            </Tooltip>
          ))}
        </Box>
      )}

      {stages.length > 0 ? (
        <Stack spacing={1} sx={{ mt: 1.5 }}>
          {stages.map((stage, index) => {
            const stageName = asString(stage.name, `stage_${index + 1}`);
            const stageStatus = asString(stage.status, 'unknown');
            const duration = asNumber(stage.duration_ms);
            const sampleOutputs = stage.sample_outputs;
            const stageWarnings = getWarnings(stage);
            const stageCounts = countEntries(stage.counts);
            const stageHasSampleHighlights = hasSampleHighlights(sampleOutputs);
            const stageHasProfileHighlights = hasProfileHighlights(sampleOutputs);
            const stageHasStructuredTextHighlights = hasStructuredTextHighlights(sampleOutputs);
            const stageHasSuggestedQuestionHighlights = hasSuggestedQuestionHighlights(sampleOutputs);
            const stageHasHighlights = (
              stageHasSampleHighlights
              || stageHasProfileHighlights
              || stageHasStructuredTextHighlights
              || stageHasSuggestedQuestionHighlights
            );
            const sampleOutputRecord = isRecord(sampleOutputs) ? sampleOutputs : {};
            const stageMeta = [
              'method',
              'strategy',
              'ready',
              'submitted',
              'target_url',
              'payload_kind',
              'submitted_filename',
              'submitted_content_type',
              'payload_bytes',
              'payload_sha256',
              'service_status',
              'status_code',
              'health_url',
              'selected_endpoint',
            ]
              .map(key => ({ key, value: stage[key] ?? sampleOutputRecord[key] }))
              .filter(item => item.value !== undefined && item.value !== null && item.value !== '');
            const message = asString(stage.message) || asString(stage.error);

            return (
              <Accordion
                key={`${stageName}-${index}`}
                disableGutters
                defaultExpanded={!dense && index < 2}
                sx={{
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 1,
                  boxShadow: 'none',
                  '&:before': { display: 'none' },
                }}
              >
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%', minWidth: 0 }}>
                    <SchemaIcon fontSize="small" color="action" />
                    <Typography variant="body2" fontWeight={700} noWrap sx={{ minWidth: 0 }}>
                      {formatStageName(stageName)}
                    </Typography>
                    <Chip
                      label={formatStageName(stageStatus)}
                      color={getStatusColor(stageStatus)}
                      size="small"
                      sx={{ ml: 'auto', flexShrink: 0 }}
                    />
                    {duration !== null && (
                      <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0 }}>
                        {duration} ms
                      </Typography>
                    )}
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Stack spacing={1}>
                    {message && (
                      <Alert severity={getAlertSeverity(stageStatus)} sx={{ py: 0.5 }}>
                        {message}
                      </Alert>
                    )}

                    {stageWarnings.length > 0 && (
                      <Alert severity="warning" sx={{ py: 0.5 }}>
                        {stageWarnings.slice(0, 3).join(' ')}
                      </Alert>
                    )}

                    {stageCounts.length > 0 && (
                      <Grid container spacing={1}>
                        {stageCounts.slice(0, 8).map(item => (
                          <Grid item xs={6} sm={3} key={item.label}>
                            <Box
                              sx={{
                                border: '1px solid',
                                borderColor: 'divider',
                                borderRadius: 1,
                                px: 1,
                                py: 0.5,
                              }}
                            >
                              <Typography variant="caption" color="text.secondary" component="div" noWrap title={item.label}>
                                {item.label}
                              </Typography>
                              <Typography variant="caption" fontWeight={700} component="div" noWrap title={formatValue(item.value)}>
                                {formatValue(item.value)}
                              </Typography>
                            </Box>
                          </Grid>
                        ))}
                      </Grid>
                    )}

                    {stageMeta.length > 0 && (
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
                        {stageMeta.map(item => (
                          <Chip
                            key={item.key}
                            label={`${formatKey(item.key)}: ${formatValue(item.value)}`}
                            title={`${formatKey(item.key)}: ${formatValue(item.value)}`}
                            size="small"
                            variant="outlined"
                            sx={{
                              maxWidth: '100%',
                              '& .MuiChip-label': {
                                display: 'block',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                              },
                            }}
                          />
                        ))}
                      </Box>
                    )}

                    {sampleOutputs !== undefined && (
                      <Box>
                        <Typography variant="caption" color="text.secondary" component="div" sx={{ mb: 0.5 }}>
                          Evidence preview
                        </Typography>
                        {stageHasProfileHighlights && (
                          <ProfileOutputHighlights sampleOutputs={sampleOutputs} />
                        )}
                        {stageHasStructuredTextHighlights && (
                          <StructuredTextHighlights sampleOutputs={sampleOutputs} dense={dense} />
                        )}
                        {stageHasSampleHighlights && (
                          <SampleOutputHighlights sampleOutputs={sampleOutputs} dense={dense} />
                        )}
                        {stageHasSuggestedQuestionHighlights && (
                          <SuggestedQuestionHighlights sampleOutputs={sampleOutputs} dense={dense} />
                        )}
                        <Box
                          component={stageHasHighlights ? 'details' : 'div'}
                          sx={{ mt: stageHasHighlights ? 1 : 0 }}
                        >
                          {stageHasHighlights && (
                            <Typography component="summary" variant="caption" color="text.secondary" sx={{ cursor: 'pointer', mb: 0.5 }}>
                              Technical payload
                            </Typography>
                          )}
                          <Box
                            component="pre"
                            sx={{
                              m: 0,
                              p: 1,
                              maxHeight: dense ? 180 : 260,
                              overflow: 'auto',
                              borderRadius: 1,
                              bgcolor: theme => alpha(theme.palette.text.primary, 0.055),
                              color: 'text.primary',
                              fontSize: 12,
                              lineHeight: 1.45,
                              whiteSpace: 'pre-wrap',
                              wordBreak: 'break-word',
                            }}
                          >
                            {summarizeJson(sampleOutputs)}
                          </Box>
                        </Box>
                      </Box>
                    )}
                  </Stack>
                </AccordionDetails>
              </Accordion>
            );
          })}
        </Stack>
      ) : (
        <Box sx={{ mt: 1 }}>
          <LinearProgress variant="determinate" value={status === 'unknown' ? 0 : 100} />
        </Box>
      )}
    </Box>
  );
};

export default GraphRAGPipelineTrace;
