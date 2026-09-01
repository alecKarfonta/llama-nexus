import { describe, expect, it } from 'vitest';
import {
  buildGraphRAGPipelineArtifactManifest,
  normalizeGraphRAGPipelineArtifactDownloads,
} from './graphragPipelineArtifacts';

describe('graphrag pipeline artifact downloads', () => {
  it('adds a manifest export and normalizes available artifact metadata', () => {
    const artifacts = normalizeGraphRAGPipelineArtifactDownloads({
      run_id: 'run-123',
      document: { filename: 'research.md' },
      artifacts: [
        {
          kind: 'structured_ingest',
          label: 'Structured ingest Markdown',
          available: true,
          content_type: 'text/markdown',
          filename: 'research.structured.md',
          char_count: 2048,
          endpoint: '/api/v1/graphrag/ingest/pipeline/run-123/artifacts/structured_ingest',
        },
        {
          kind: 'local_graph',
          label: 'Local graph artifact',
          available: true,
          content_type: 'application/json',
          endpoint: '/api/v1/graphrag/ingest/pipeline/run-123/artifacts/local_graph',
        },
        {
          kind: 'local_context',
          label: 'Local context cache',
          available: false,
          content_type: 'application/json',
        },
      ],
    });

    expect(artifacts.map(artifact => artifact.kind)).toEqual([
      'manifest',
      'structured_ingest',
      'local_graph',
    ]);
    expect(artifacts[0]).toMatchObject({
      label: 'Artifact manifest',
      downloadFilename: 'run-123.artifacts.json',
    });
    expect(artifacts[1]).toMatchObject({
      shortLabel: 'Structured Ingest',
      downloadFilename: 'research.structured.md',
      detail: 'text/markdown • 2,048 chars • /api/v1/graphrag/ingest/pipeline/run-123/artifacts/structured_ingest',
    });
    expect(artifacts[2]).toMatchObject({
      shortLabel: 'Local Graph',
      downloadFilename: 'run-123.local-graph.json',
    });
  });

  it('builds predictable safe filenames when the manifest has no explicit filename', () => {
    const artifacts = normalizeGraphRAGPipelineArtifactDownloads(
      {
        run_id: 'run:with spaces/and/slashes',
        document: { filename: '../Quarterly Revenue Deck.pptx' },
        artifacts: [
          {
            kind: 'structured_ingest',
            available: true,
            content_type: 'text/markdown; charset=utf-8',
          },
          {
            kind: 'local_graph_reasoning',
            available: true,
            content_type: 'application/json',
          },
          {
            kind: 'full',
            available: true,
            content_type: 'application/json',
          },
        ],
      },
      { includeManifest: false },
    );

    expect(artifacts.map(artifact => artifact.downloadFilename)).toEqual([
      'Quarterly-Revenue-Deck.structured.md',
      'slashes.local-graph-reasoning.json',
      'slashes.json',
    ]);
  });

  it('can include unavailable artifacts for diagnostics', () => {
    const artifacts = normalizeGraphRAGPipelineArtifactDownloads(
      {
        artifacts: [
          {
            kind: 'samples',
            label: 'Pipeline samples',
            available: false,
            content_type: 'application/json',
          },
        ],
      },
      { includeUnavailable: true, includeManifest: false, runId: 'diagnostic-run' },
    );

    expect(artifacts).toHaveLength(1);
    expect(artifacts[0]).toMatchObject({
      kind: 'samples',
      available: false,
      downloadFilename: 'diagnostic-run.samples.json',
    });
  });

  it('synthesizes an artifact manifest from a full pipeline run', () => {
    const manifest = buildGraphRAGPipelineArtifactManifest({
      run_id: 'run-local-ready',
      status: 'preview_only',
      document: {
        filename: 'deck.pptx',
        content_type: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        source_artifact_available: true,
        source_artifact_size_bytes: 28060,
      },
      source_document_artifact: {
        filename: 'deck.pptx',
        content_type: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        size_bytes: 28060,
      },
      structured_ingest_artifact: {
        filename: 'deck.structured.md',
        content_type: 'text/markdown',
        char_count: 227266,
        uses_structured_text: true,
      },
      local_graph: { node_count: 107, edge_count: 220 },
      local_context: { chunk_count: 27 },
    });

    const artifacts = normalizeGraphRAGPipelineArtifactDownloads(
      manifest,
      { includeManifest: false },
    );

    expect(manifest).toMatchObject({
      run_id: 'run-local-ready',
      artifact_count: 5,
    });
    expect(artifacts.map(artifact => artifact.kind).slice(0, 4)).toEqual([
      'source_document',
      'structured_ingest',
      'local_graph',
      'local_context',
    ]);
    expect(artifacts[0]).toMatchObject({
      label: 'Original source document',
      downloadFilename: 'deck.pptx',
      endpoint: '/api/v1/graphrag/ingest/pipeline/run-local-ready/artifacts/source_document',
    });
    expect(artifacts[1]).toMatchObject({
      downloadFilename: 'deck.structured.md',
      charCount: 227266,
    });
  });
});
