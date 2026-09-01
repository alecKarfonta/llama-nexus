import { describe, expect, it } from 'vitest';
import {
  buildGraphRAGChatContextPayload,
  buildGraphRAGTraceChatHandoff,
  buildGraphRAGTraceContextQuery,
  buildGraphRAGTraceDisplayPrompt,
  buildGraphRAGTracePrompt,
  buildGraphRAGTraceQuestions,
  normalizeGraphRAGChatContextEndpoint,
  shouldUseUnifiedGraphRAGChatContext,
} from './graphragPipelinePrompts';

describe('graphrag pipeline prompts', () => {
  it('builds a summary prompt with detected capabilities', () => {
    const prompt = buildGraphRAGTracePrompt(
      'strategy.pptx',
      {},
      { detected_capabilities: ['shape_connector', 'chart_series'] },
    );

    expect(prompt).toContain('strategy.pptx');
    expect(prompt).toContain('Shape Connector');
    expect(prompt).toContain('Chart Series');
    expect(prompt).toContain('return exactly 4 bullets');
    expect(prompt).toContain('under 180 words total');
    expect(prompt).toContain('named Entities, Top relationships, Best path, and Source proof');
    expect(prompt).toContain('Do not add a trailing paragraph');
    expect(prompt).toContain('prioritize gate/block/require/support/drive/reference edges');
    expect(prompt).toContain('Cite slide/page/table/chart/connector provenance');
  });

  it('keeps generated trace handoff prompts concise in the visible chat bubble', () => {
    const generatedPrompt = buildGraphRAGTracePrompt(
      'graphrag-ux-rich-e2e.pptx',
      {},
      { detected_capabilities: ['tables', 'charts'] },
      [],
      {
        artifacts: [
          {
            kind: 'local_graph',
            available: true,
            endpoint: '/api/v1/graphrag/ingest/pipeline/run-rich/artifacts/local_graph',
          },
        ],
      },
    );

    expect(buildGraphRAGTraceDisplayPrompt('graphrag-ux-rich-e2e.pptx', generatedPrompt)).toBe(
      'Summarize graphrag-ux-rich-e2e.pptx',
    );
    expect(
      buildGraphRAGTraceDisplayPrompt(
        'graphrag-ux-rich-e2e.pptx',
        'What chart evidence was extracted?',
      ),
    ).toBe('What chart evidence was extracted?');
  });

  it('uses retrieval-focused context queries for generated trace summaries', () => {
    const generatedPrompt = buildGraphRAGTracePrompt(
      'graphrag-ux-rich-e2e.pptx',
      {},
      { detected_capabilities: ['tables', 'charts', 'styled_text'] },
      [],
      {
        artifacts: [
          {
            kind: 'local_graph',
            available: true,
            endpoint: '/api/v1/graphrag/ingest/pipeline/run-rich/artifacts/local_graph',
          },
        ],
      },
    );
    const contextQuery = buildGraphRAGTraceContextQuery('graphrag-ux-rich-e2e.pptx', generatedPrompt);

    expect(contextQuery).toContain('Extracted document evidence');
    expect(contextQuery).toContain('semantic relationships');
    expect(contextQuery).toContain('source provenance');
    expect(contextQuery).not.toContain('GraphRAG pipeline trace');
    expect(contextQuery).not.toContain('Best path');
    expect(contextQuery).not.toContain('Source proof');
    expect(contextQuery).not.toContain('return exactly 4 bullets');
    expect(contextQuery).not.toContain('Exportable pipeline artifacts');
  });

  it('adds exportable artifact context to trace handoff prompts', () => {
    const prompt = buildGraphRAGTracePrompt(
      'research.md',
      {},
      {},
      [],
      {
        run_id: 'run-123',
        document: { filename: 'research.md' },
        artifacts: [
          {
            kind: 'structured_ingest',
            available: true,
            content_type: 'text/markdown',
            filename: 'research.structured.md',
            endpoint: '/api/v1/graphrag/ingest/pipeline/run-123/artifacts/structured_ingest',
          },
          {
            kind: 'local_graph',
            available: true,
            content_type: 'application/json',
            endpoint: '/api/v1/graphrag/ingest/pipeline/run-123/artifacts/local_graph',
          },
        ],
      },
    );

    expect(prompt).toContain('Exportable pipeline artifacts available');
    expect(prompt).toContain('Structured Ingest (research.structured.md');
    expect(prompt).toContain('Local Graph (run-123.local-graph.json');
    expect(prompt).toContain('local graph JSON for nodes and edges');
  });

  it('suggests run-scoped graph questions from relationships, entities, and connector evidence', () => {
    const questions = buildGraphRAGTraceQuestions({
      run_id: 'graphrag_ingest_local_001',
      document: { filename: 'strategy.pptx', domain: 'slides' },
      document_profile: { detected_capabilities: ['shape_connector', 'table'] },
      stages: [
        {
          name: 'structure_extraction',
          sample_outputs: {
            elements: [
              {
                type: 'shape_connector',
                text: 'Product Revenue -> Regional Demand: drives',
                metadata: {
                  slide: 1,
                  slide_title: 'Revenue Architecture',
                  source_label: 'Product Revenue',
                  target_label: 'Regional Demand',
                },
              },
            ],
          },
        },
        {
          name: 'graph_candidates',
          sample_outputs: {
            entities: [
              { name: 'Product Revenue', type: 'metric' },
              { name: 'Regional Demand', type: 'market_signal' },
            ],
            relationships: [
              {
                source: 'Product Revenue',
                relation: 'drives',
                target: 'Regional Demand',
                evidence: 'Product revenue drives regional demand on slide 1.',
                metadata: {
                  slide: 1,
                  element_type: 'shape_connector',
                },
              },
            ],
          },
        },
      ],
      local_graph: {
        edges: [
          {
            source: 'Slide 1: Revenue Architecture',
            label: 'contains_connector',
            target: 'Product Revenue -> Regional Demand: drives',
            evidence: 'Product Revenue -> Regional Demand: drives',
            metadata: {
              slide: 1,
              element_type: 'shape_connector',
            },
          },
        ],
      },
    });

    expect(questions.map(question => question.kind)).toContain('summary');
    expect(questions.some(question => question.prompt.includes('"Product Revenue" connects to "Regional Demand"'))).toBe(true);
    expect(questions.some(question => question.label === 'Audit connectors')).toBe(true);
    expect(questions.every(question => question.prompt.includes('strategy.pptx'))).toBe(true);
  });

  it('suggests an artifact review question when run artifacts are exportable', () => {
    const questions = buildGraphRAGTraceQuestions({
      run_id: 'graphrag_ingest_local_001',
      document: { filename: 'research.md' },
      artifacts: {
        run_id: 'graphrag_ingest_local_001',
        document: { filename: 'research.md' },
        artifacts: [
          {
            kind: 'structured_ingest',
            available: true,
            content_type: 'text/markdown',
            filename: 'research.structured.md',
            endpoint: '/api/v1/graphrag/ingest/pipeline/graphrag_ingest_local_001/artifacts/structured_ingest',
          },
          {
            kind: 'local_graph_reasoning',
            available: true,
            content_type: 'application/json',
            endpoint: '/api/v1/graphrag/ingest/pipeline/graphrag_ingest_local_001/artifacts/local_graph_reasoning',
          },
        ],
      },
    }, 3);

    expect(questions.map(question => question.label)).toContain('Review run artifacts');
    expect(questions.find(question => question.label === 'Review run artifacts')?.prompt).toContain('bridge/path reasoning');
    expect(questions.find(question => question.label === 'Summarize trace')?.prompt).toContain('Exportable pipeline artifacts available');
  });

  it('prefers backend interaction suggestions when the trace provides them', () => {
    const questions = buildGraphRAGTraceQuestions({
      document: { filename: 'deck.pptx' },
      interaction_suggestions: {
        suggested_questions: [
          {
            id: 'backend_connector_audit',
            kind: 'evidence',
            label: 'Audit connectors',
            prompt: 'Using the GraphRAG pipeline trace for deck.pptx, audit backend connector evidence.',
          },
        ],
      },
      stages: [
        {
          name: 'graph_candidates',
          sample_outputs: {
            relationships: [
              {
                source: 'A',
                relation: 'connects_to',
                target: 'B',
              },
            ],
          },
        },
      ],
    });

    expect(questions[0]).toMatchObject({
      id: 'backend_connector_audit',
      label: 'Audit connectors',
      kind: 'evidence',
    });
    expect(questions[0].prompt).toContain('backend connector evidence');
  });

  it('uses interaction handoff starter prompts before mined frontend questions', () => {
    const questions = buildGraphRAGTraceQuestions({
      document: { filename: 'handoff.pptx' },
      interaction_handoff: {
        starter_prompts: [
          {
            id: 'handoff-chart',
            kind: 'evidence',
            label: 'Explain chart handoff',
            prompt: 'Use the handoff chart anchors for handoff.pptx.',
          },
        ],
      },
      stages: [
        {
          name: 'graph_candidates',
          sample_outputs: {
            relationships: [
              {
                source: 'Revenue',
                relation: 'has_value',
                target: 'Q2',
              },
            ],
          },
        },
      ],
    }, 4);

    expect(questions[0]).toMatchObject({
      id: 'handoff-chart',
      label: 'Explain chart handoff',
      kind: 'evidence',
    });
    expect(questions[0].prompt).toContain('handoff chart anchors');
  });

  it('packages trace starter prompts and citation scopes for chat route handoff', () => {
    const handoff = buildGraphRAGTraceChatHandoff({
      run_id: 'run-link-001',
      document: { filename: 'risk-map.pptx', domain: 'risk' },
      interaction_handoff: {
        recommended_mode: 'graph_path_reasoning',
        answer_guidance: 'Cite link and image relationship provenance.',
        citation_scopes: ['slide', 'link', 'layout'],
        chat_context_endpoint: '/api/v1/graphrag/chat-context-v2',
        chat_context_request: {
          query: 'Trace clicked-image links',
          domain: 'risk',
          pipeline_run_ids: ['run-link-001'],
          include_graph: true,
          use_local_pipeline_context: true,
          include_local_pipeline: true,
          source: 'Risk Map',
          target: 'Supplier Portal',
        },
        counts: {
          graph_anchors: 3,
          retrieval_rehearsals: 2,
        },
        capability_cards: [
          {
            id: 'links',
            label: 'Link evidence',
            detail: 'URI and image relationship anchors',
            count: 3,
          },
        ],
        starter_prompts: [
          {
            id: 'trace-links',
            kind: 'evidence',
            label: 'Trace links',
            prompt: 'Use the risk-map.pptx handoff to trace clicked-image links.',
          },
        ],
      },
    }, { questionLimit: 3 });

    expect(handoff).toMatchObject({
      runId: 'run-link-001',
      runIds: ['run-link-001'],
      filename: 'risk-map.pptx',
      domain: 'risk',
      chatContextEndpoint: '/api/v1/graphrag/chat-context-v2',
      recommendedMode: 'graph_path_reasoning',
      answerGuidance: 'Cite link and image relationship provenance.',
      citationScopes: ['slide', 'link', 'layout'],
      graphAnchorCount: 3,
      retrievalRehearsalCount: 2,
    });
    expect(handoff.prompt).toContain('clicked-image links');
    expect(handoff.displayPrompt).toBe('Use the risk-map.pptx handoff to trace clicked-image links.');
    expect(handoff.suggestedQuestions?.[0]).toMatchObject({
      id: 'trace-links',
      label: 'Trace links',
      prompt: 'Use the risk-map.pptx handoff to trace clicked-image links.',
    });
    expect(handoff.capabilityCards?.[0]).toMatchObject({
      id: 'links',
      label: 'Link evidence',
      count: 3,
    });
    expect(handoff.chatContextRequest).toMatchObject({
      query: 'Trace clicked-image links',
      domain: 'risk',
      pipeline_run_ids: ['run-link-001'],
      include_graph: true,
      use_local_pipeline_context: true,
      include_local_pipeline: true,
      source: 'Risk Map',
      target: 'Supplier Portal',
    });
  });

  it('normalizes GraphRAG chat context endpoints to known same-origin routes', () => {
    expect(normalizeGraphRAGChatContextEndpoint('/api/v1/graphrag/chat-context-v2')).toBe('/api/v1/graphrag/chat-context-v2');
    expect(normalizeGraphRAGChatContextEndpoint('https://llama.local/api/v1/graphrag/chat-context?debug=1')).toBe('/api/v1/graphrag/chat-context');
    expect(normalizeGraphRAGChatContextEndpoint('https://example.com/not-graphrag')).toBe('/api/v1/graphrag/chat-context-v2');
    expect(normalizeGraphRAGChatContextEndpoint('/api/v1/graphrag/chat-context-v2/')).toBe('/api/v1/graphrag/chat-context-v2');
  });

  it('keeps v2 chat context available for graph-only document conversations', () => {
    expect(shouldUseUnifiedGraphRAGChatContext('/api/v1/graphrag/chat-context-v2', {
      graphEnabled: true,
      ragEnabled: false,
    })).toBe(true);
    expect(shouldUseUnifiedGraphRAGChatContext('/api/v1/graphrag/chat-context', {
      graphEnabled: true,
      ragEnabled: false,
    })).toBe(false);
    expect(shouldUseUnifiedGraphRAGChatContext('/api/v1/graphrag/chat-context-v2', {
      graphEnabled: false,
      ragEnabled: true,
    })).toBe(false);
  });

  it('builds chat context payloads from prepared handoff requests while preserving the live query', () => {
    const payload = buildGraphRAGChatContextPayload({
      runId: 'fallback-run',
      runIds: ['fallback-run'],
      domain: 'slides',
      suggestedQuestions: [
        {
          id: 'selected-path',
          kind: 'path',
          label: 'Path: Product Revenue via Regional Demand',
          prompt: 'live user question',
          source: 'Product Revenue',
          bridge: 'Regional Demand',
          target: 'Risk Map',
        },
      ],
      chatContextRequest: {
        query: 'old prepared prompt',
        top_k: 9,
        domain: 'risk',
        pipeline_run_ids: ['run-a', 'run-b'],
        include_graph: true,
        use_local_pipeline_context: true,
        include_local_pipeline: true,
        rag_domains: ['pipeline-domain'],
        rag_search_mode: 'global',
        source: 'Old Source',
        target: 'Old Target',
        pipeline_evidence_totals: {
          code_blocks: 3,
          diagram_relationships: 5,
        },
        pipeline_evidence_families: ['code_blocks', 'diagram_relationships'],
        graph_overview_counts: {
          cross_document_paths: 2,
        },
      },
    }, 'live user question', {
      ragTopK: 3,
      ragSelectedDomains: ['ui-domain'],
      ragSearchMode: 'domains',
    });

    expect(payload).toMatchObject({
      query: 'live user question',
      top_k: 9,
      domain: 'risk',
      pipeline_run_ids: ['run-a', 'run-b'],
      include_graph: true,
      use_local_pipeline_context: true,
      include_local_pipeline: true,
      rag_enabled: true,
      rag_domains: ['pipeline-domain'],
      rag_search_mode: 'global',
      source: 'Product Revenue',
      bridge: 'Regional Demand',
      target: 'Risk Map',
      pipeline_evidence_totals: {
        code_blocks: 3,
        diagram_relationships: 5,
      },
      pipeline_evidence_families: ['code_blocks', 'diagram_relationships'],
      graph_overview_counts: {
        cross_document_paths: 2,
      },
      response_mode: 'chat',
      context_text_max_chars: 12000,
      response_chunk_content_chars: 700,
      response_graph_context_limit: 2,
      response_graph_path_limit: 4,
      response_evidence_card_limit: 3,
      response_relationship_limit: 8,
      response_path_limit: 4,
    });
  });

  it('uses prepared context queries for auto-submitted generated prompts and typed queries for follow-ups', () => {
    const generatedPrompt = buildGraphRAGTracePrompt(
      'graphrag-ux-rich-e2e.pptx',
      {},
      { detected_capabilities: ['tables', 'charts'] },
    );
    const contextQuery = buildGraphRAGTraceContextQuery(
      'graphrag-ux-rich-e2e.pptx',
      generatedPrompt,
    );
    const handoff = {
      runId: 'run-rich',
      runIds: ['run-rich'],
      filename: 'graphrag-ux-rich-e2e.pptx',
      prompt: generatedPrompt,
      contextQuery,
      chatContextRequest: {
        query: contextQuery,
        pipeline_run_ids: ['run-rich'],
        include_graph: true,
        use_local_pipeline_context: true,
        include_local_pipeline: true,
      },
    };

    const autoSubmitPayload = buildGraphRAGChatContextPayload(handoff, generatedPrompt);
    expect(autoSubmitPayload.query).toBe(contextQuery);
    expect(autoSubmitPayload.query).not.toContain('GraphRAG pipeline trace');
    expect(autoSubmitPayload.query).not.toContain('Best path');
    expect(autoSubmitPayload.query).not.toContain('Source proof');
    expect(autoSubmitPayload.query).not.toContain('return exactly 4 bullets');

    const followUpPayload = buildGraphRAGChatContextPayload(
      handoff,
      'What chart evidence was found?',
    );
    expect(followUpPayload.query).toBe('What chart evidence was found?');
  });

  it('preserves prepared source-target-relation hints for selected reference handoffs', () => {
    const preparedPrompt =
      'Use this selected GraphRAG relationship as the primary evidence: Approval Dependency Cluster --[groups_visual]--> Approval required before launch workflow.';
    const payload = buildGraphRAGChatContextPayload({
      runId: 'graphrag_ingest_selected_edge',
      runIds: ['graphrag_ingest_selected_edge'],
      domain: 'slides',
      chatContextRequest: {
        query: preparedPrompt,
        pipeline_run_ids: ['graphrag_ingest_selected_edge'],
        source: 'Approval Dependency Cluster',
        relation: 'groups_visual',
        target: 'Approval required before launch workflow.',
      },
    }, preparedPrompt);

    expect(payload).toMatchObject({
      query: preparedPrompt,
      pipeline_run_ids: ['graphrag_ingest_selected_edge'],
      source: 'Approval Dependency Cluster',
      relation: 'groups_visual',
      target: 'Approval required before launch workflow.',
      include_graph: true,
      use_local_pipeline_context: true,
      include_local_pipeline: true,
    });
  });

  it('honors the live RAG toggle when building v2 graph chat payloads', () => {
    const payload = buildGraphRAGChatContextPayload({
      runId: 'run-graph-only',
      runIds: ['run-graph-only'],
      domain: 'slides',
    }, 'What does Product Revenue drive?', {
      ragEnabled: false,
      graphEnabled: true,
      ragTopK: 5,
    });

    expect(payload).toMatchObject({
      query: 'What does Product Revenue drive?',
      domain: 'slides',
      pipeline_run_ids: ['run-graph-only'],
      include_graph: true,
      use_local_pipeline_context: true,
      include_local_pipeline: true,
      rag_enabled: false,
      top_k: 5,
    });
  });

  it('infers source and target anchors from natural graph chat questions', () => {
    const payload = buildGraphRAGChatContextPayload({
      runId: 'run-path-001',
      runIds: ['run-path-001'],
      domain: 'slides',
      suggestedQuestions: [
        {
          id: 'prepared-path',
          kind: 'path',
          label: 'Path: Product Revenue to Regional Demand',
          prompt: 'prepared starter prompt',
          source: 'Product Revenue',
          target: 'Regional Demand',
        },
      ],
      chatContextRequest: {
        pipeline_run_ids: ['run-path-001'],
        source: 'Old Source',
        target: 'Old Target',
      },
    }, 'How does Product Revenue connect to Regional Demand?');

    expect(payload).toMatchObject({
      query: 'How does Product Revenue connect to Regional Demand?',
      pipeline_run_ids: ['run-path-001'],
      source: 'Product Revenue',
      target: 'Regional Demand',
      include_graph: true,
      use_local_pipeline_context: true,
    });
  });

  it('infers anchors from natural domain relationship verbs', () => {
    const payload = buildGraphRAGChatContextPayload({
      runId: 'run-drive-001',
      runIds: ['run-drive-001'],
      domain: 'slides',
      suggestedQuestions: [
        {
          id: 'prepared-drive',
          kind: 'relationship',
          label: 'Product Revenue drives Regional Demand',
          prompt: 'prepared relationship prompt',
          source: 'Product Revenue',
          relation: 'drives',
          target: 'Regional Demand',
        },
      ],
    }, 'How does Product Revenue drive Regional Demand?');

    expect(payload).toMatchObject({
      query: 'How does Product Revenue drive Regional Demand?',
      pipeline_run_ids: ['run-drive-001'],
      source: 'Product Revenue',
      relation: 'drive',
      target: 'Regional Demand',
      include_graph: true,
      use_local_pipeline_context: true,
    });
  });

  it('does not split entity names whose last word looks like a relation verb', () => {
    const payload = buildGraphRAGChatContextPayload({
      runId: 'run-gate-001',
      runIds: ['run-gate-001'],
      domain: 'general',
      suggestedQuestions: [
        {
          id: 'trace-gates',
          kind: 'relationship',
          label: 'Trace Gates: Launch Workflow',
          prompt: 'Trace the launch gate.',
          source: 'Approval Gate',
          relation: 'gates',
          target: 'Launch Workflow',
        },
      ],
    }, 'How does the Approval Gate relate to the Launch Workflow? Cite the relationship and evidence.');

    expect(payload).toMatchObject({
      query: 'How does the Approval Gate relate to the Launch Workflow? Cite the relationship and evidence.',
      pipeline_run_ids: ['run-gate-001'],
      source: 'Approval Gate',
      target: 'Launch Workflow',
      include_graph: true,
      use_local_pipeline_context: true,
    });
    expect(payload.relation).toBeUndefined();
  });

  it('keeps typed entity anchors narrower than broad connector labels', () => {
    const payload = buildGraphRAGChatContextPayload({
      runId: 'run-connector-001',
      suggestedQuestions: [
        {
          id: 'prepared-connector',
          kind: 'relationship',
          label: 'Connector evidence',
          prompt: 'prepared connector prompt',
          source: 'graphrag-complex-office-source.pptx',
          target: 'Product Revenue -> Regional Demand: drives',
          relation: 'contains_connector',
        },
      ],
      chatContextRequest: {
        pipeline_run_ids: ['run-connector-001'],
      },
    }, 'How does Product Revenue connect to Regional Demand?');

    expect(payload).toMatchObject({
      source: 'Product Revenue',
      target: 'Regional Demand',
    });
  });

  it('infers bridge anchors from path questions with via or through phrasing', () => {
    const payload = buildGraphRAGChatContextPayload({
      runId: 'run-bridge-001',
      suggestedQuestions: [
        {
          id: 'prepared-bridge',
          kind: 'path',
          label: 'Path: Alpha Team via Beta Program',
          prompt: 'prepared bridge prompt',
          source: 'Alpha Team',
          bridge: 'Beta Program',
          target: 'Omega Launch',
        },
      ],
    }, 'Find the graph path from Alpha Team to Omega Launch through Beta Program.');

    expect(payload).toMatchObject({
      pipeline_run_ids: ['run-bridge-001'],
      source: 'Alpha Team',
      bridge: 'Beta Program',
      target: 'Omega Launch',
    });
  });

  it('infers two-hop graph path anchors from arrow notation', () => {
    const payload = buildGraphRAGChatContextPayload({
      runId: 'run-arrow-001',
      suggestedQuestions: [
        {
          id: 'prepared-arrow',
          kind: 'path',
          label: 'Path: Revenue via Demand',
          prompt: 'prepared arrow prompt',
          source: 'Product Revenue',
          bridge: 'Regional Demand',
          target: 'Risk Map',
        },
      ],
    }, 'Product Revenue -> Regional Demand -> Risk Map');

    expect(payload).toMatchObject({
      pipeline_run_ids: ['run-arrow-001'],
      source: 'Product Revenue',
      bridge: 'Regional Demand',
      target: 'Risk Map',
    });
  });

  it('does not carry prepared graph anchors into an unrelated free-form question', () => {
    const payload = buildGraphRAGChatContextPayload({
      runId: 'run-summary-001',
      chatContextRequest: {
        pipeline_run_ids: ['run-summary-001'],
        source: 'Prepared Source',
        target: 'Prepared Target',
      },
    }, 'Summarize the uploaded document.');

    expect(payload.source).toBeUndefined();
    expect(payload.target).toBeUndefined();
  });

  it('leaves broad multi-entity questions open for backend graph path ranking', () => {
    const payload = buildGraphRAGChatContextPayload({
      runId: 'run-path-ranking-001',
      suggestedQuestions: [
        {
          id: 'stale-prepared-path',
          kind: 'path',
          label: 'Path: Product Revenue via Risk Map',
          prompt: 'prepared narrow starter',
          source: 'Product Revenue',
          bridge: 'Risk Map',
          target: 'Regional Demand',
        },
      ],
      chatContextRequest: {
        pipeline_run_ids: ['run-path-ranking-001'],
        source: 'Product Revenue',
        bridge: 'Risk Map',
        target: 'Regional Demand',
      },
    }, 'How does Product Revenue influence Regional Demand, Risk Map, and Launch Priority?');

    expect(payload).toMatchObject({
      query: 'How does Product Revenue influence Regional Demand, Risk Map, and Launch Priority?',
      pipeline_run_ids: ['run-path-ranking-001'],
    });
    expect(payload.source).toBeUndefined();
    expect(payload.bridge).toBeUndefined();
    expect(payload.target).toBeUndefined();
  });

  it('leaves ordered multi-hop path chains open for backend graph path ranking', () => {
    const payload = buildGraphRAGChatContextPayload({
      runId: 'run-pdf-path-ranking-001',
      suggestedQuestions: [
        {
          id: 'prepared-first-hop',
          kind: 'relationship',
          label: 'Product Revenue to Regional Demand',
          prompt: 'prepared narrow starter',
          source: 'Product Revenue',
          target: 'Regional Demand',
        },
      ],
      chatContextRequest: {
        pipeline_run_ids: ['run-pdf-path-ranking-001'],
        source: 'Product Revenue',
        target: 'Regional Demand',
      },
    }, 'Explain the Product Revenue to Regional Demand to Risk Map path. Cite the PDF page evidence.');

    expect(payload).toMatchObject({
      query: 'Explain the Product Revenue to Regional Demand to Risk Map path. Cite the PDF page evidence.',
      pipeline_run_ids: ['run-pdf-path-ranking-001'],
    });
    expect(payload.source).toBeUndefined();
    expect(payload.target).toBeUndefined();
  });

  it('mines local graph and chunk graph context when sample entities are sparse', () => {
    const questions = buildGraphRAGTraceQuestions({
      document: { filename: 'brief.pdf' },
      stages: [
        {
          name: 'embedding_plan',
          sample_outputs: {
            embedding_inputs: [
              {
                chunk_id: 'chunk_1',
                text: 'Regional demand depends on product revenue.',
              },
            ],
          },
        },
      ],
      local_graph: {
        nodes: [
          { id: 'Product Revenue', name: 'Product Revenue', type: 'metric' },
          { id: 'Regional Demand', name: 'Regional Demand', type: 'market_signal' },
        ],
        edges: [
          {
            source: 'Product Revenue',
            label: 'supports',
            target: 'Regional Demand',
            evidence: 'Revenue support is visible in the regional demand brief.',
          },
        ],
      },
      local_context: {
        chunks: [
          {
            chunk_id: 'chunk_1',
            graph_context: [
              {
                source: 'Product Revenue',
                relation: 'supports',
                target: 'Regional Demand',
                evidence: 'Attached as graph context for retrieval.',
              },
            ],
          },
        ],
      },
    }, 5);

    expect(questions.some(question => question.prompt.includes('"Product Revenue" connects to "Regional Demand"'))).toBe(true);
    expect(questions.some(question => question.label === 'Inspect graph context')).toBe(true);
    expect(questions.some(question => question.label.includes('Path: Product Revenue to Regional Demand'))).toBe(true);
  });

  it('uses local graph reasoning samples for bridge path questions', () => {
    const questions = buildGraphRAGTraceQuestions({
      document: { filename: 'research.md' },
      stages: [
        {
          name: 'local_graph_reasoning_index',
          sample_outputs: {
            bridge_entities: [
              {
                name: 'Beta Program',
                text: 'Beta Program links Alpha Team and Omega Launch',
              },
            ],
            path_candidates: [
              {
                source: 'Alpha Team',
                bridge: 'Beta Program',
                target: 'Omega Launch',
                nodes: ['Alpha Team', 'Beta Program', 'Omega Launch'],
                text: 'Alpha Team --[owns]-- Beta Program --[unblocks]-- Omega Launch',
                relationships: [
                  { source: 'Alpha Team', relation: 'owns', target: 'Beta Program' },
                ],
              },
            ],
          },
        },
      ],
    }, 6);

    expect(questions.some(question => question.label.includes('Path: Alpha Team via Beta Program'))).toBe(true);
    expect(questions.some(question => question.prompt.includes('"Alpha Team" -> "Beta Program" -> "Omega Launch"'))).toBe(true);
  });

  it('uses local context graph path samples for bridge path questions', () => {
    const questions = buildGraphRAGTraceQuestions({
      document: { filename: 'deck.pptx' },
      stages: [
        {
          name: 'local_context_cache',
          sample_outputs: {
            graph_paths: [
              {
                source: 'Product Revenue',
                bridge: 'Regional Demand',
                target: 'Risk Map',
                nodes: ['Product Revenue', 'Regional Demand', 'Risk Map'],
                text: 'Product Revenue --[drives]-- Regional Demand --[references]-- Risk Map',
              },
            ],
          },
        },
      ],
    }, 6);

    expect(questions.some(question => question.label.includes('Path: Product Revenue via Regional Demand'))).toBe(true);
    expect(questions.some(question => question.prompt.includes('"Product Revenue" -> "Regional Demand" -> "Risk Map"'))).toBe(true);
  });

  it('suggests deep evidence questions for artifacts, OCR, markup, and styled text', () => {
    const questions = buildGraphRAGTraceQuestions({
      document: { filename: 'annual-strategy.pdf' },
      document_profile: {
        detected_capabilities: [
          'embedded_objects',
          'ocr',
          'annotations',
          'revisions',
          'styled_text',
          'forms',
          'diagrams',
        ],
      },
      stages: [
        {
          name: 'document_profile',
          sample_outputs: {
            detected_capabilities: ['embedded_objects', 'ocr', 'annotations', 'styled_text', 'forms', 'diagrams'],
          },
        },
      ],
    }, 9);

    const labels = questions.map(question => question.label);
    expect(labels).toContain('Trace source artifacts');
    expect(labels).toContain('Review OCR evidence');
    expect(labels).toContain('Audit diagrams');
    expect(labels).toContain('Review markup');
    expect(labels).toContain('Inspect emphasized text');
    expect(labels).toContain('Inspect form fields');
    expect(questions.find(question => question.label === 'Trace source artifacts')?.prompt).toContain('child documents');
    expect(questions.find(question => question.label === 'Audit diagrams')?.prompt).toContain('SmartArt nodes');
    expect(questions.find(question => question.label === 'Inspect emphasized text')?.prompt).toContain('bold/italic/underlined');
  });
});
