import { describe, expect, it } from 'vitest';
import {
  normalizeGraphRAGDocumentOCRStatus,
  normalizeGraphRAGServiceStatus,
} from './graphragServiceStatus';

describe('graphrag service status utilities', () => {
  it('normalizes a ready OCR runtime into a workbench card', () => {
    const card = normalizeGraphRAGDocumentOCRStatus({
      enabled: true,
      ready: true,
      status: 'ready',
      engine: 'pytesseract',
      runtime: {
        tesseract_path: '/usr/bin/tesseract',
      },
      message: 'Image OCR is enabled and the pytesseract runtime is available.',
    });

    expect(card).toEqual({
      id: 'ocr',
      label: 'OCR',
      status: 'ready',
      ready: true,
      enabled: true,
      present: true,
      detail: 'Image OCR is enabled and the pytesseract runtime is available.',
      meta: 'pytesseract · /usr/bin/tesseract',
    });
  });

  it('turns missing OCR runtime dependencies into an actionable blocked card', () => {
    const card = normalizeGraphRAGDocumentOCRStatus({
      enabled: true,
      ready: false,
      status: 'missing_runtime',
      missing: ['Pillow', 'tesseract binary'],
      required_env: ['GRAPHRAG_ENABLE_IMAGE_OCR'],
    });

    expect(card.status).toBe('blocked');
    expect(card.enabled).toBe(true);
    expect(card.detail).toBe('Image OCR is enabled, but the runtime needs attention.');
    expect(card.meta).toBe('Missing Pillow, Tesseract Binary');
  });

  it('keeps unreported OCR status hidden from readiness rows', () => {
    const card = normalizeGraphRAGDocumentOCRStatus(undefined);

    expect(card.present).toBe(false);
    expect(card.status).toBe('unavailable');
  });

  it('normalizes local-only deployment contracts for the workbench', () => {
    const normalized = normalizeGraphRAGServiceStatus({
      graphrag: {
        configured: false,
        ready: false,
        status: 'not_configured',
        required_env: ['GRAPHRAG_URL', 'GRAPHRAG_SERVICE_URL'],
        optional: false,
        message: 'Set GRAPHRAG_URL to enable',
      },
      ner: {
        configured: true,
        ready: true,
        status: 'healthy',
        url: 'http://graphrag-ner:8001',
        url_source: 'docker_default',
        url_source_label: 'Docker service default',
        auto_discovered: true,
        optional: true,
        service: {
          health_url: 'http://graphrag-ner:8001/health',
          compatible_health_urls: [
            'http://graphrag-ner:8001/health',
            'http://graphrag-ner:8001/api/health',
            'http://graphrag-ner:8001/api/v1/health',
          ],
          attempted_health_urls: [
            'http://graphrag-ner:8001/health',
            'http://graphrag-ner:8001/api/health',
          ],
          health_attempts: [
            {
              url: 'http://graphrag-ner:8001/health',
              status: 'not_found',
              status_code: 404,
            },
            {
              url: 'http://graphrag-ner:8001/api/health',
              status: 'http_response',
              status_code: 200,
            },
          ],
        },
      },
      _meta: {
        status: 'local',
        summary: 'Local pipeline preview and chat handoff are available without remote GraphRAG.',
        source_summary: '1 Docker default helper URL detected.',
        next_steps: ['Set GRAPHRAG_URL for remote graph ingest.'],
        operating_mode: {
          label: 'Local pipeline mode',
          summary: 'Uploads generate local traces.',
          recommended_action: 'Use pipeline upload now.',
        },
        interaction_contract: {
          document_pipeline: {
            available: true,
            status: 'local',
            mode: 'local_document_evidence_preview',
            supported_formats: ['pdf', 'docx', 'pptx', 'xlsx'],
            format_count: 4,
            evidence_capabilities: [
              {
                id: 'charts',
                label: 'Charts',
                status: 'ready',
                ready: true,
                formats: ['pptx', 'xlsx'],
                element_types: ['chart_series', 'chart_point'],
                relationship_types: ['contains_chart', 'contains_metric'],
                interaction_outputs: ['capability_card', 'citation_scope'],
                detail: 'Chart evidence.',
              },
              {
                id: 'shapes_connectors',
                label: 'Shapes & Connectors',
                status: 'ready',
                ready: true,
                formats: ['pptx', 'odp'],
                element_types: ['shape_metadata', 'shape_connector'],
                relationship_types: ['contains_shape', 'contains_connector'],
                interaction_outputs: ['capability_card', 'graph_path_context'],
                detail: 'Shape and connector evidence.',
              },
              {
                id: 'link_actions',
                label: 'Links & Actions',
                status: 'ready',
                ready: true,
                formats: ['pdf', 'pptx', 'html'],
                element_types: ['link'],
                relationship_types: ['links_to', 'contains_link'],
                interaction_outputs: ['capability_card', 'graph_anchor', 'citation_scope', 'retrieval_rehearsal'],
                detail: 'Link and action evidence.',
              },
              {
                id: 'markdown_semantics',
                label: 'Markdown Semantics',
                status: 'ready',
                ready: true,
                formats: ['md', 'markdown'],
                element_types: ['document_metadata', 'code_block', 'diagram_relationship', 'form_field', 'link'],
                relationship_types: ['contains_code_block', 'uses_language', 'contains_diagram_relationship'],
                interaction_outputs: ['capability_card', 'relationship_evidence_card', 'structured_ingest_export'],
                detail: 'Markdown graph evidence.',
              },
            ],
          },
          upload: {
            available: true,
            remote_submission: false,
            behavior: 'local_trace_only',
            endpoint_strategy: 'adaptive',
          },
          chat: {
            available: true,
            mode: 'local_pipeline_handoff',
            context_sources: ['pipeline_runs', 'local_graph'],
          },
          search: {
            available: false,
            fallback: 'local_pipeline_search',
          },
          deployment: {
            smoke_tests: [
              {
                id: 'remote_health_preflight',
                label: 'Remote GraphRAG health preflight',
                status: 'not_configured',
                method: 'GET',
                urls: [
                  'http://graphrag.test/health',
                  'http://graphrag.test/api/health',
                  'http://graphrag.test/api/v1/health',
                ],
                expected: 'Set GRAPHRAG_URL.',
              },
            ],
          },
        },
        workflows: {
          local_pipeline_preview: {
            label: 'Local document pipeline preview',
            status: 'local',
            ready: true,
            endpoint: '/api/v1/graphrag/ingest/pipeline/preview',
          },
        },
        operator_checklist: [
          {
            id: 'remote_ingest',
            label: 'Remote GraphRAG ingest',
            status: 'local',
            ready: true,
            detail: 'Local traces are ready.',
            checks: ['remote_health_preflight'],
          },
        ],
        deployment_quickstart: [
          {
            id: 'local_trace_chat',
            label: 'Local trace + Ask Chat',
            status: 'ready',
            ready: true,
            summary: 'Enable GraphRAG routes to upload and chat over traces.',
            command: 'GRAPHRAG_ENABLED=true docker compose up -d backend-api',
            env_vars: ['GRAPHRAG_ENABLED=true'],
            checks: ['GET /api/v1/graphrag/services/status'],
            endpoint: '/api/v1/graphrag/ingest/pipeline/preview',
          },
          {
            id: 'remote_graph',
            label: 'Remote GraphRAG ingest + search',
            status: 'local',
            ready: true,
            summary: 'Connect a reachable GraphRAG API.',
            command: 'GRAPHRAG_ENABLED=true GRAPHRAG_URL=http://host.docker.internal:18000 docker compose --profile graphrag up -d backend-api',
            env_vars: ['GRAPHRAG_ENABLED=true', 'GRAPHRAG_URL=http://host.docker.internal:18000'],
            profiles: ['graphrag'],
            checks: ['GET /api/v1/graphrag/health', 'POST /api/v1/graphrag/ingest/pipeline/upload'],
            endpoint: 'http://graphrag.test/ingest-documents',
            next_step: 'Set GRAPHRAG_URL for remote graph ingest.',
          },
        ],
        interaction_pathways: [
          {
            id: 'upload_trace_chat',
            label: 'Upload -> Trace -> Ask Chat',
            status: 'local',
            ready: true,
            recommended: true,
            summary: 'Create a local trace and ask chat from the run.',
            action: 'Upload documents through the pipeline.',
            mode: 'local_pipeline_graph_context',
            endpoint: '/api/v1/graphrag/ingest/pipeline/upload',
            secondary_endpoint: '/api/v1/graphrag/chat-context-v2',
            checks: ['stage samples', 'local graph artifact'],
            outputs: ['pipeline_run_id', 'interaction_handoff', 'local_context'],
            uses_local_pipeline: true,
            uses_remote_graph: false,
          },
          {
            id: 'remote_graph_search',
            label: 'Remote Graph Ingest + Search',
            status: 'local',
            ready: true,
            summary: 'Local traces are ready while remote graph is not configured.',
            action: 'Set GRAPHRAG_URL.',
            mode: 'not_configured',
            checks: ['health preflight'],
            outputs: ['graph_search'],
            uses_local_pipeline: true,
            uses_remote_graph: false,
          },
        ],
        recommended_pathway_id: 'upload_trace_chat',
      },
    });

    expect(normalized.smokeStatus).toBe('local');
    expect(normalized.readyCount).toBe(1);
    expect(normalized.configuredCount).toBe(1);
    expect(normalized.remoteGraphRAGReady).toBe(false);
    expect(normalized.entries.find(service => service.name === 'ner')).toMatchObject({
      autoDiscovered: true,
      urlSourceLabel: 'Docker service default',
      healthUrl: 'http://graphrag-ner:8001/health',
      compatibleHealthUrls: [
        'http://graphrag-ner:8001/health',
        'http://graphrag-ner:8001/api/health',
        'http://graphrag-ner:8001/api/v1/health',
      ],
      attemptedHealthUrls: [
        'http://graphrag-ner:8001/health',
        'http://graphrag-ner:8001/api/health',
      ],
      healthAttempts: [
        expect.objectContaining({
          url: 'http://graphrag-ner:8001/health',
          status: 'not_found',
          statusCode: 404,
        }),
        expect.objectContaining({
          url: 'http://graphrag-ner:8001/api/health',
          status: 'http_response',
          statusCode: 200,
        }),
      ],
    });
    expect(normalized.contractItems).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: 'pipeline', status: 'local', detail: 'Local Document Evidence Preview', meta: '4 formats' }),
      expect.objectContaining({ id: 'upload', status: 'local', detail: 'Local Trace Only' }),
      expect.objectContaining({ id: 'chat', status: 'ready', meta: 'Pipeline Runs + Local Graph' }),
      expect.objectContaining({ id: 'search', status: 'local', detail: 'Local Pipeline Search' }),
      expect.objectContaining({ id: 'evidence', status: 'local', detail: 'Charts + Shapes & Connectors + Links & Actions + Markdown Semantics', meta: '4 families' }),
    ]));
    expect(normalized.evidenceCapabilities).toEqual(expect.arrayContaining([
      expect.objectContaining({
        id: 'shapes_connectors',
        label: 'Shapes & Connectors',
        formats: ['pptx', 'odp'],
        elementTypes: ['shape_metadata', 'shape_connector'],
        relationshipTypes: ['contains_shape', 'contains_connector'],
        interactionOutputs: ['capability_card', 'graph_path_context'],
      }),
      expect.objectContaining({
        id: 'link_actions',
        label: 'Links & Actions',
        relationshipTypes: ['links_to', 'contains_link'],
        interactionOutputs: ['capability_card', 'graph_anchor', 'citation_scope', 'retrieval_rehearsal'],
      }),
      expect.objectContaining({
        id: 'markdown_semantics',
        label: 'Markdown Semantics',
        formats: ['md', 'markdown'],
        elementTypes: ['document_metadata', 'code_block', 'diagram_relationship', 'form_field', 'link'],
        relationshipTypes: ['contains_code_block', 'uses_language', 'contains_diagram_relationship'],
        interactionOutputs: ['capability_card', 'relationship_evidence_card', 'structured_ingest_export'],
      }),
    ]));
    expect(normalized.smokeTests[0]).toMatchObject({
      id: 'remote_health_preflight',
      status: 'not_configured',
      method: 'GET',
      candidates: [
        'http://graphrag.test/health',
        'http://graphrag.test/api/health',
        'http://graphrag.test/api/v1/health',
      ],
    });
    expect(normalized.workflowEntries[0]).toMatchObject({
      id: 'local_pipeline_preview',
      ready: true,
      status: 'local',
    });
    expect(normalized.operatorChecklist[0].checks).toEqual(['remote_health_preflight']);
    expect(normalized.deploymentQuickstart).toEqual(expect.arrayContaining([
      expect.objectContaining({
        id: 'local_trace_chat',
        status: 'ready',
        command: 'GRAPHRAG_ENABLED=true docker compose up -d backend-api',
        envVars: ['GRAPHRAG_ENABLED=true'],
        endpoint: '/api/v1/graphrag/ingest/pipeline/preview',
      }),
      expect.objectContaining({
        id: 'remote_graph',
        status: 'local',
        profiles: ['graphrag'],
        checks: ['GET /api/v1/graphrag/health', 'POST /api/v1/graphrag/ingest/pipeline/upload'],
        nextStep: 'Set GRAPHRAG_URL for remote graph ingest.',
      }),
    ]));
    expect(normalized.recommendedPathwayId).toBe('upload_trace_chat');
    expect(normalized.interactionPathways).toEqual(expect.arrayContaining([
      expect.objectContaining({
        id: 'upload_trace_chat',
        status: 'local',
        ready: true,
        recommended: true,
        endpoint: '/api/v1/graphrag/ingest/pipeline/upload',
        secondaryEndpoint: '/api/v1/graphrag/chat-context-v2',
        checks: ['stage samples', 'local graph artifact'],
        outputs: ['pipeline_run_id', 'interaction_handoff', 'local_context'],
        usesLocalPipeline: true,
        usesRemoteGraph: false,
      }),
      expect.objectContaining({
        id: 'remote_graph_search',
        mode: 'not_configured',
        usesLocalPipeline: true,
        usesRemoteGraph: false,
      }),
    ]));
    expect(normalized.severity).toBe('info');
  });

  it('falls back to disabled status when no services are ready', () => {
    const normalized = normalizeGraphRAGServiceStatus({
      graphrag: {
        configured: false,
        ready: false,
        optional: false,
        required_env: ['GRAPHRAG_URL'],
      },
    }, {
      graphrag_enabled: false,
    });

    expect(normalized.loaded).toBe(true);
    expect(normalized.smokeStatus).toBe('disabled');
    expect(normalized.graphRAGCoreService?.message).toBe('Set GRAPHRAG_URL to enable.');
    expect(normalized.severity).toBe('info');
  });
});
