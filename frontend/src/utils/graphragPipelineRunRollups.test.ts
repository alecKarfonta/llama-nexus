import { describe, expect, it } from 'vitest';
import { normalizeGraphRAGPipelineRunRollups } from './graphragPipelineRunRollups';

describe('graphrag pipeline run rollups', () => {
  it('normalizes backend performance and deployment outcome rollups', () => {
    const rollups = normalizeGraphRAGPipelineRunRollups({
      pipeline_performance: {
        stage_count: 4,
        total_duration_ms: 1250,
        local_stage_duration_ms: 900,
        remote_stage_duration_ms: 350,
        slowest_stage: { name: 'structure_extraction', status: 'complete', duration_ms: 720 },
        status_counts: { complete: 3, warning: 1 },
        throughput: {
          bytes_per_second: 4096,
          chunks_per_second: 12.5,
        },
        timings: [
          { name: 'received', status: 'complete', duration_ms: 10 },
          { name: 'structure_extraction', status: 'complete', duration_ms: 720 },
          { name: 'upstream_ingest', status: 'complete', duration_ms: 340 },
        ],
      },
      deployment_outcome: {
        status: 'remote_submitted',
        label: 'Remote Graph Submitted',
        severity: 'success',
        ready: true,
        remote_blocked: false,
        remote_graph_available: true,
        remote_submitted: true,
        fallback_used: true,
        preflight: {
          health_url: 'http://graphrag.test/health',
          health_attempt_count: 1,
        },
        ingest: {
          selected_endpoint: 'http://graphrag.test/process-document',
          target_url: 'http://graphrag.test/process-document',
          endpoint_attempt_count: 2,
          failed_attempt_count: 1,
          payload_kind: 'structured_text',
          payload_bytes: 7821,
        },
        operator: {
          primary_action: 'ask_chat',
          next_step: 'Use Ask Chat or graph search.',
          services_status_path: '/api/v1/graphrag/services/status',
          diagnostics_path: '/api/v1/graphrag/deployment/diagnostics',
        },
        trace_fields: ['upstream_ingest.sample_outputs.endpoint_attempts'],
      },
    });

    expect(rollups.performance).toMatchObject({
      available: true,
      stageCount: 4,
      totalDurationMs: 1250,
      localStageDurationMs: 900,
      remoteStageDurationMs: 350,
      slowestStage: {
        name: 'structure_extraction',
        durationMs: 720,
      },
      statusCounts: { complete: 3, warning: 1 },
      throughput: {
        bytes_per_second: 4096,
        chunks_per_second: 12.5,
      },
    });
    expect(rollups.performance.timings).toHaveLength(3);
    expect(rollups.deployment).toMatchObject({
      available: true,
      status: 'remote_submitted',
      label: 'Remote Graph Submitted',
      severity: 'success',
      ready: true,
      remoteBlocked: false,
      remoteGraphAvailable: true,
      remoteSubmitted: true,
      fallbackUsed: true,
      healthUrl: 'http://graphrag.test/health',
      selectedEndpoint: 'http://graphrag.test/process-document',
      endpointAttemptCount: 2,
      failedAttemptCount: 1,
      healthAttemptCount: 1,
      payloadKind: 'structured_text',
      payloadBytes: 7821,
      primaryAction: 'ask_chat',
      servicesStatusPath: '/api/v1/graphrag/services/status',
    });
    expect(rollups.deployment.traceFields).toContain('upstream_ingest.sample_outputs.endpoint_attempts');
  });

  it('falls back to stage timings and operational handoff for older traces', () => {
    const rollups = normalizeGraphRAGPipelineRunRollups({
      status: 'preview_only',
      stages: [
        { name: 'received', status: 'complete', duration_ms: 5 },
        { name: 'chunking', status: 'warning', duration_ms: 20 },
      ],
      operational_handoff: {
        status: 'local_trace_ready',
        label: 'Local Trace Ready',
        chat_handoff_ready: true,
        remote_ingest_submitted: false,
        remote_ingest_ready: false,
        primary_action: 'ask_chat',
        next_step: 'Set GRAPHRAG_URL later.',
      },
    });

    expect(rollups.performance).toMatchObject({
      available: true,
      stageCount: 2,
      totalDurationMs: 25,
      slowestStage: {
        name: 'chunking',
        durationMs: 20,
      },
    });
    expect(rollups.deployment).toMatchObject({
      available: true,
      status: 'local_trace_ready',
      label: 'Local Trace Ready',
      ready: false,
      remoteSubmitted: false,
      remoteGraphAvailable: false,
      remoteBlocked: true,
      primaryAction: 'ask_chat',
      nextStep: 'Set GRAPHRAG_URL later.',
    });
  });
});
