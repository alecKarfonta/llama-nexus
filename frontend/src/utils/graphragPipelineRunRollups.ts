type JsonRecord = Record<string, unknown>;

export interface GraphRAGPipelineTiming {
  name: string;
  status: string;
  durationMs: number;
}

export interface GraphRAGPipelinePerformanceRollup {
  available: boolean;
  stageCount: number;
  totalDurationMs: number;
  localStageDurationMs: number;
  remoteStageDurationMs: number;
  slowestStage?: GraphRAGPipelineTiming;
  statusCounts: Record<string, number>;
  throughput: Record<string, number>;
  timings: GraphRAGPipelineTiming[];
}

export interface GraphRAGDeploymentOutcomeRollup {
  available: boolean;
  status: string;
  label: string;
  severity: 'success' | 'warning' | 'error' | 'info';
  ready: boolean;
  remoteBlocked: boolean;
  remoteGraphAvailable: boolean;
  remoteSubmitted: boolean;
  fallbackUsed: boolean;
  healthUrl?: string;
  selectedEndpoint?: string;
  targetUrl?: string;
  endpointAttemptCount: number;
  failedAttemptCount: number;
  healthAttemptCount: number;
  payloadKind?: string;
  payloadBytes?: number;
  primaryAction?: string;
  nextStep?: string;
  servicesStatusPath?: string;
  diagnosticsPath?: string;
  traceFields: string[];
}

export interface GraphRAGPipelineRunRollups {
  performance: GraphRAGPipelinePerformanceRollup;
  deployment: GraphRAGDeploymentOutcomeRollup;
}

const isRecord = (value: unknown): value is JsonRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const asArray = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

const asString = (value: unknown, fallback = ''): string =>
  typeof value === 'string' && value.trim() ? value.trim() : fallback;

const asNumber = (value: unknown, fallback = 0): number => {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
};

const severityValue = (value: unknown): GraphRAGDeploymentOutcomeRollup['severity'] => {
  const severity = asString(value).toLowerCase();
  if (severity === 'success' || severity === 'warning' || severity === 'error' || severity === 'info') {
    return severity;
  }
  return 'info';
};

const normalizeTiming = (value: unknown, index: number): GraphRAGPipelineTiming | null => {
  if (!isRecord(value)) return null;
  const name = asString(value.name, `stage_${index + 1}`);
  return {
    name,
    status: asString(value.status, 'unknown'),
    durationMs: Math.max(0, Math.round(asNumber(value.duration_ms ?? value.durationMs))),
  };
};

const normalizePerformance = (result: JsonRecord): GraphRAGPipelinePerformanceRollup => {
  const raw = isRecord(result.pipeline_performance)
    ? result.pipeline_performance
    : isRecord(isRecord(result.summary) ? result.summary.pipeline_performance : undefined)
      ? (result.summary as JsonRecord).pipeline_performance as JsonRecord
      : {};
  const stageTimings = asArray(result.stages).map(normalizeTiming).filter((timing): timing is GraphRAGPipelineTiming => Boolean(timing));
  const timings = asArray(raw.timings).map(normalizeTiming).filter((timing): timing is GraphRAGPipelineTiming => Boolean(timing));
  const finalTimings = timings.length > 0 ? timings : stageTimings;
  const totalDurationMs = Math.max(
    0,
    Math.round(asNumber(raw.total_duration_ms ?? raw.totalDurationMs, finalTimings.reduce((total, item) => total + item.durationMs, 0))),
  );
  const slowestRecord = isRecord(raw.slowest_stage) ? raw.slowest_stage : undefined;
  const slowestStage = slowestRecord
    ? normalizeTiming(slowestRecord, 0) || undefined
    : finalTimings.length > 0
      ? [...finalTimings].sort((left, right) => right.durationMs - left.durationMs)[0]
      : undefined;
  const statusCounts = isRecord(raw.status_counts)
    ? Object.fromEntries(Object.entries(raw.status_counts).map(([key, value]) => [key, asNumber(value)]))
    : {};
  const throughput = isRecord(raw.throughput)
    ? Object.fromEntries(Object.entries(raw.throughput).map(([key, value]) => [key, asNumber(value)]))
    : {};

  return {
    available: Boolean(Object.keys(raw).length || finalTimings.length),
    stageCount: Math.round(asNumber(raw.stage_count ?? raw.stageCount, finalTimings.length)),
    totalDurationMs,
    localStageDurationMs: Math.max(0, Math.round(asNumber(raw.local_stage_duration_ms ?? raw.localStageDurationMs))),
    remoteStageDurationMs: Math.max(0, Math.round(asNumber(raw.remote_stage_duration_ms ?? raw.remoteStageDurationMs))),
    slowestStage,
    statusCounts,
    throughput,
    timings: finalTimings,
  };
};

const normalizeDeployment = (result: JsonRecord): GraphRAGDeploymentOutcomeRollup => {
  const outcome = isRecord(result.deployment_outcome) ? result.deployment_outcome : {};
  const preflight = isRecord(outcome.preflight) ? outcome.preflight : {};
  const ingest = isRecord(outcome.ingest) ? outcome.ingest : {};
  const operator = isRecord(outcome.operator) ? outcome.operator : {};
  const operationalHandoff = isRecord(result.operational_handoff) ? result.operational_handoff : {};
  const fallbackStatus = asString(operationalHandoff.status, asString(result.status, 'unknown'));
  const remoteSubmitted = Boolean(outcome.remote_submitted ?? operationalHandoff.remote_ingest_submitted);
  const remoteGraphAvailable = Boolean(outcome.remote_graph_available ?? operationalHandoff.remote_ingest_ready);

  return {
    available: Boolean(Object.keys(outcome).length || Object.keys(operationalHandoff).length),
    status: asString(outcome.status, fallbackStatus),
    label: asString(outcome.label, asString(operationalHandoff.label, fallbackStatus.replace(/_/g, ' '))),
    severity: severityValue(outcome.severity),
    ready: Boolean(outcome.ready ?? remoteSubmitted),
    remoteBlocked: Boolean(outcome.remote_blocked ?? (!remoteSubmitted && !remoteGraphAvailable)),
    remoteGraphAvailable,
    remoteSubmitted,
    fallbackUsed: Boolean(outcome.fallback_used),
    healthUrl: asString(preflight.health_url) || asString(operationalHandoff.health_url) || undefined,
    selectedEndpoint: asString(ingest.selected_endpoint) || asString(operationalHandoff.target_url) || undefined,
    targetUrl: asString(ingest.target_url) || asString(operationalHandoff.target_url) || undefined,
    endpointAttemptCount: Math.round(asNumber(ingest.endpoint_attempt_count)),
    failedAttemptCount: Math.round(asNumber(ingest.failed_attempt_count)),
    healthAttemptCount: Math.round(asNumber(preflight.health_attempt_count)),
    payloadKind: asString(ingest.payload_kind, asString(operationalHandoff.payload_kind)) || undefined,
    payloadBytes: ingest.payload_bytes !== undefined ? Math.max(0, Math.round(asNumber(ingest.payload_bytes))) : undefined,
    primaryAction: asString(operator.primary_action, asString(operationalHandoff.primary_action)) || undefined,
    nextStep: asString(operator.next_step, asString(operationalHandoff.next_step)) || undefined,
    servicesStatusPath: asString(operator.services_status_path, asString(operationalHandoff.services_status_path)) || undefined,
    diagnosticsPath: asString(operator.diagnostics_path, asString(operationalHandoff.diagnostics_path)) || undefined,
    traceFields: asArray(outcome.trace_fields).map(item => asString(item)).filter(Boolean),
  };
};

export const normalizeGraphRAGPipelineRunRollups = (result: unknown): GraphRAGPipelineRunRollups => {
  const record = isRecord(result) ? result : {};
  return {
    performance: normalizePerformance(record),
    deployment: normalizeDeployment(record),
  };
};
