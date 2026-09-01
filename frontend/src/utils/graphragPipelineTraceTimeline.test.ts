import { describe, expect, it } from 'vitest';
import { buildGraphRAGTraceTimeline } from './graphragPipelineTraceTimeline';

describe('graphrag pipeline trace timeline', () => {
  it('summarizes stage durations, statuses, and sample richness', () => {
    const timeline = buildGraphRAGTraceTimeline({
      stages: [
        {
          name: 'document_intake',
          status: 'complete',
          duration_ms: 10,
          sample_outputs: {
            filename: 'strategy.pptx',
            detected_capabilities: ['chart_series'],
          },
        },
        {
          name: 'structure_extraction',
          status: 'warning',
          duration_ms: 30,
          warnings: ['Some speaker notes were empty.'],
          sample_outputs: {
            elements: [{ type: 'chart', text: 'Revenue Forecast' }],
          },
        },
        {
          name: 'upstream_ingest',
          status: 'failed',
          duration_ms: 10,
          error: 'GraphRAG API unavailable',
        },
      ],
    });

    expect(timeline.totalDurationMs).toBe(50);
    expect(timeline.stageCount).toBe(3);
    expect(timeline.completedStageCount).toBe(1);
    expect(timeline.warningStageCount).toBe(1);
    expect(timeline.failedStageCount).toBe(1);
    expect(timeline.sampleStageCount).toBe(2);
    expect(timeline.slowestStage?.name).toBe('structure_extraction');
    expect(timeline.stages.map(stage => stage.percentOfTotal)).toEqual([20, 60, 20]);
    expect(timeline.stages[1]).toMatchObject({
      label: 'Structure Extraction',
      warningCount: 1,
      sampleGroupCount: 1,
      sampleItemCount: 1,
      sampleGroups: ['Elements'],
    });
  });

  it('still distributes progress when stages have no duration', () => {
    const timeline = buildGraphRAGTraceTimeline({
      stages: [
        { name: 'chunking', status: 'complete', sample_outputs: { chunks: [{ id: 'c1' }, { id: 'c2' }] } },
        { name: 'graph_candidates', status: 'skipped', sample_outputs: {} },
      ],
    });

    expect(timeline.totalDurationMs).toBe(0);
    expect(timeline.stages.map(stage => stage.percentOfTotal)).toEqual([50, 50]);
    expect(timeline.warningStageCount).toBe(1);
    expect(timeline.sampleStageCount).toBe(1);
    expect(timeline.stages[0].sampleItemCount).toBe(2);
  });

  it('counts adaptive upstream endpoint attempts as trace samples', () => {
    const timeline = buildGraphRAGTraceTimeline({
      stages: [
        {
          name: 'upstream_ingest',
          status: 'complete',
          sample_outputs: {
            selected_endpoint: 'http://graphrag.test/process-document',
            endpoint_attempts: [
              { url: 'http://graphrag.test/ingest-documents', status_code: 404 },
              { url: 'http://graphrag.test/process-document', status_code: 200 },
            ],
          },
        },
      ],
    });

    expect(timeline.sampleStageCount).toBe(1);
    expect(timeline.stages[0]).toMatchObject({
      sampleGroupCount: 2,
      sampleItemCount: 3,
      sampleGroups: ['Endpoint checks', 'Selected endpoint'],
    });
  });

  it('summarizes high-value evidence groups before debug plumbing', () => {
    const timeline = buildGraphRAGTraceTimeline({
      stages: [
        {
          name: 'graph_candidates',
          status: 'complete',
          sample_outputs: {
            selected_endpoint: 'http://graphrag.test/process-document',
            endpoint_attempts: [{ status_code: 200 }],
            relationships: [
              {
                source: 'Product Revenue',
                relation: 'drives',
                target: 'Regional Demand',
                evidence: 'Slide 4 explains the relationship.',
              },
            ],
            chunks: [{ chunk_id: 'chunk-1', text: 'Revenue demand evidence.' }],
          },
        },
      ],
    });

    expect(timeline.stages[0].sampleGroups.slice(0, 2)).toEqual([
      'Relationships',
      'Chunks',
    ]);
    expect(timeline.stages[0].debugSampleGroups).toEqual([
      'Endpoint checks',
      'Selected endpoint',
    ]);
  });
});
