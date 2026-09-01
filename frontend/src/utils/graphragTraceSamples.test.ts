import { describe, expect, it } from 'vitest';
import {
  cleanGraphRAGTraceSampleText,
  graphRAGTraceSampleEvidenceSummary,
  graphRAGTraceSampleGroupDisplayLabel,
  graphRAGTraceSampleLinkTarget,
  isLowValueGraphRAGTraceSampleText,
  pickBestGraphRAGTraceSample,
  rankGraphRAGTraceSampleGroups,
  rankGraphRAGTraceSampleRecords,
} from './graphragTraceSamples';

describe('graphrag trace sample ranking', () => {
  it('prefers semantic document evidence over Office internals and endpoint plumbing', () => {
    const sampleOutputs = {
      endpoint_attempts: [
        { url: 'http://graphrag.test/ingest-documents', status_code: 404 },
      ],
      relationships: [
        {
          source: 'ppt/embeddings/Microsoft_Excel_Worksheet1.xlsx',
          relation: 'contains_chart_workbook',
          target: 'Regional Revenue Forecast',
          evidence: 'Internal embedded workbook relationship.',
        },
        {
          source: 'Product Revenue',
          relation: 'drives',
          target: 'Regional Demand',
          evidence: 'Slide 4 states regional demand tracks product revenue growth.',
          source_locations: [{ page: 4, label: 'slide 4' }],
        },
      ],
      chunks: [
        {
          chunk_id: 'chunk-3',
          text: 'Regional demand tracks product revenue growth in enterprise accounts.',
        },
      ],
    };

    const best = pickBestGraphRAGTraceSample(sampleOutputs, [
      'endpoint_attempts',
      'relationships',
      'chunks',
    ]);

    expect(best?.key).toBe('relationships');
    expect(best?.record).toMatchObject({
      source: 'Product Revenue',
      relation: 'drives',
      target: 'Regional Demand',
    });
  });

  it('sorts records and groups so evidence is visible before debug samples', () => {
    const relationshipRecords = [
      {
        source: 'https://example.com',
        relation: 'external_link',
        target: 'Brand Guidelines',
      },
      {
        source: 'Approval Required',
        relation: 'has_value',
        target: 'checked',
        evidence: 'The intake form has the approval checkbox checked.',
      },
    ];

    expect(rankGraphRAGTraceSampleRecords('relationships', relationshipRecords)[0].record).toMatchObject({
      source: 'Approval Required',
      target: 'checked',
    });

    const groups = rankGraphRAGTraceSampleGroups({
      filename: 'launch-smoke.pdf',
      selected_endpoint: 'http://graphrag.test/process-document',
      endpoint_attempts: [{ status_code: 200 }],
      relationship_evidence_cards: relationshipRecords,
      graph_paths: [{ nodes: ['A', 'B', 'C'], explanation: 'A connects to C through B.' }],
    });

    expect(groups.slice(0, 2).map(group => group.key)).toEqual([
      'relationship_evidence_cards',
      'graph_paths',
    ]);
    expect(groups.find(group => group.key === 'filename')?.role).toBe('operational');
    expect(groups.at(-1)?.role).toBe('debug');
  });

  it('formats sample group labels as user-facing evidence counts', () => {
    expect(graphRAGTraceSampleGroupDisplayLabel('relationship_evidence_cards', 8)).toBe('8 relationships');
    expect(graphRAGTraceSampleGroupDisplayLabel('chunk_evidence_cards', 1)).toBe('1 chunk');
    expect(graphRAGTraceSampleGroupDisplayLabel('relationship_sample_categories', 2)).toBe('2 categories');
    expect(graphRAGTraceSampleGroupDisplayLabel('starter_prompts', 3)).toBe('3 questions');
  });

  it('keeps document link evidence visible without promoting generic external URL noise', () => {
    const records = [
      {
        source: 'https://example.com',
        relation: 'external_link',
        target: 'Brand Guidelines',
      },
      {
        source: 'Evidence Snapshot',
        relation: 'links_to',
        target: 'https://example.com/evidence/approval-gate',
        evidence: 'Evidence Snapshot -> https://example.com/evidence/approval-gate',
        metadata: {
          label: 'Evidence Snapshot',
          uri: 'https://example.com/evidence/approval-gate',
          element_type: 'link',
          slide_title: 'Launch Evidence',
        },
      },
    ];

    expect(rankGraphRAGTraceSampleRecords('relationship_evidence_cards', records)[0].record).toMatchObject({
      source: 'Evidence Snapshot',
      relation: 'links_to',
    });
  });

  it('extracts a compact link target from relationship provenance', () => {
    expect(graphRAGTraceSampleLinkTarget({
      source: 'Launch Evidence',
      relation: 'contains_link',
      target: 'Evidence Snapshot',
      evidence: 'Evidence Snapshot -> https://example.com/evidence/approval-gate',
      provenance: {
        label: 'Evidence Snapshot',
        uri: 'https://example.com/evidence/approval-gate',
        element_type: 'link',
      },
    })).toEqual({
      label: 'Evidence Snapshot',
      url: 'https://example.com/evidence/approval-gate',
    });
  });

  it('prefers direct visual graph evidence over file-led chart paths in stage previews', () => {
    const sampleOutputs = {
      path_candidates: [
        {
          id: 'file-chart-path',
          kind: 'path',
          nodes: ['quarterly-report.pptx', 'Approval', 'Revenue Plan'],
          relationships: [
            {
              source: 'quarterly-report.pptx',
              relation: 'contains_metric',
              target: 'Approval',
            },
            {
              source: 'Revenue Plan',
              relation: 'contains_chart_series',
              target: 'Approval',
            },
          ],
          answer_hint: 'Use the chart path through Approval to explain Revenue Plan.',
        },
      ],
      bridge_entities: [
        {
          id: 'approval-cluster',
          name: 'Approval Dependency Cluster',
          relationships: [
            {
              source: 'Approval Dependency Cluster',
              relation: 'groups_visual',
              target: 'Approval required before launch workflow.',
            },
            {
              source: 'Approval required before launch workflow.',
              relation: 'gates',
              target: 'Launch workflow',
            },
          ],
          text: 'Approval Dependency Cluster groups the approval gate and launch workflow.',
        },
      ],
    };

    const best = pickBestGraphRAGTraceSample(sampleOutputs, [
      'path_candidates',
      'bridge_entities',
    ]);

    expect(best?.key).toBe('bridge_entities');
    expect(best?.record).toMatchObject({
      name: 'Approval Dependency Cluster',
    });
  });

  it('prefers relationship starter prompts over incidental graph anchors for chat context samples', () => {
    const sampleOutputs = {
      graph_anchors: [
        {
          id: 'chart-anchor',
          kind: 'path',
          title: 'deck.pptx -> Approval -> Revenue Plan',
          nodes: ['deck.pptx', 'Approval', 'Revenue Plan'],
          evidence_families: ['chart', 'layout'],
        },
      ],
      starter_prompts: [
        {
          id: 'relationship-gate',
          kind: 'relationship',
          label: 'Approval required before launch workflow. to Launch workflow',
          source: 'Approval required before launch workflow.',
          relation: 'gates',
          target: 'Launch workflow',
          prompt: 'Explain how the approval requirement gates the launch workflow and cite connector evidence.',
        },
      ],
    };

    const best = pickBestGraphRAGTraceSample(sampleOutputs, [
      'graph_anchors',
      'starter_prompts',
    ]);

    expect(best?.key).toBe('starter_prompts');
    expect(best?.record).toMatchObject({
      id: 'relationship-gate',
    });
  });

  it('summarizes reasoning paths as graph facts instead of chat prompts', () => {
    const summary = graphRAGTraceSampleEvidenceSummary('path_candidates', {
      nodes: [
        'Evidence Snapshot',
        'Slide 1: Launch Evidence',
        'https://example.com/evidence/approval-gate',
      ],
      relationships: [
        {
          source: 'Evidence Snapshot',
          relation: 'links_to',
          target: 'https://example.com/evidence/approval-gate',
        },
      ],
      answer_hint: 'Use the path through Slide 1 to explain the URL relationship.',
    });

    expect(summary).toBe('Evidence Snapshot links to https://example.com/evidence/approval-gate');
    expect(summary).not.toMatch(/\buse the path|explain\b/i);
  });

  it('summarizes retrieval rehearsals from matches without leaking raw prompts', () => {
    const summary = graphRAGTraceSampleEvidenceSummary('retrieval_rehearsals', {
      matched_terms: ['Evidence Snapshot', 'approval gate'],
      prompt: 'Using the GraphRAG pipeline trace, explain this relationship.',
    });

    expect(summary).toBe('Matched Evidence Snapshot, approval gate');
    expect(summary).not.toMatch(/Using the GraphRAG pipeline trace/i);
  });

  it('keeps validation evidence ahead of HF auth boilerplate in text previews', () => {
    const rawText = [
      'Warning: You are sending unauthenticated requests to the HF Hub.',
      'Please set a HF_TOKEN to avoid rate limits.',
      '======================================================================',
      'TURBOQUANT PHASE A: REAL MODEL VALIDATION Model: Qwen/Qwen3-1.7B result: pass.',
    ].join(' ');

    expect(cleanGraphRAGTraceSampleText(rawText, 140)).toBe(
      'TURBOQUANT PHASE A: REAL MODEL VALIDATION Model: Qwen/Qwen3-1.7B result: pass.',
    );
    expect(isLowValueGraphRAGTraceSampleText(rawText)).toBe(false);
  });

  it('demotes weak warning co-occurrence relationships below real document evidence', () => {
    const sampleOutputs = {
      relationships: [
        {
          source: 'Warning',
          relation: 'co_occurs_with',
          target: 'You',
          evidence: 'Warning appears with You in an HF Hub authentication notice.',
        },
        {
          source: 'REAL MODEL VALIDATION Model',
          relation: 'co_occurs_with',
          target: 'Qwen/Qwen3-1.7B',
          evidence: 'TURBOQUANT PHASE A: REAL MODEL VALIDATION Model: Qwen/Qwen3-1.7B result: pass.',
        },
      ],
      chunks: [
        {
          text: 'TURBOQUANT PHASE A: REAL MODEL VALIDATION Model: Qwen/Qwen3-1.7B result: pass.',
        },
      ],
    };

    expect(rankGraphRAGTraceSampleRecords('relationships', sampleOutputs.relationships)[0].record).toMatchObject({
      source: 'REAL MODEL VALIDATION Model',
      target: 'Qwen/Qwen3-1.7B',
    });

    const best = pickBestGraphRAGTraceSample(sampleOutputs, [
      'relationships',
      'chunks',
    ]);

    expect(best?.record).not.toMatchObject({
      source: 'Warning',
      target: 'You',
    });
  });

  it('does not use warning-only retrieval rehearsals as the chat context preview', () => {
    const sampleOutputs = {
      retrieval_rehearsals: [
        {
          label: 'Warning to You',
          prompt_id: 'relationship_warning_you_co_occurs_with',
          prompt:
            'Explain how Warning connects to You through co_occurs_with. The extracted evidence snippet was: Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN.',
          matched_terms: [
            'warning',
            'you',
            'co_occurs_with',
            'graph',
            'sending',
            'unauthenticated',
            'requests',
            'hub',
            'please',
            'set',
            'hf_token',
            'enable',
          ],
          top_chunks: [
            {
              evidence_preview:
                'Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN. TURBOQUANT PHASE A: REAL MODEL VALIDATION.',
            },
          ],
        },
      ],
      graph_anchors: [
        {
          kind: 'path',
          title: 'Warning -> You -> HF Hub',
          nodes: ['Warning', 'You', 'HF Hub'],
          relationships: [
            {
              source: 'Warning',
              relation: 'co_occurs_with',
              target: 'You',
            },
            {
              source: 'You',
              relation: 'co_occurs_with',
              target: 'HF Hub',
            },
          ],
          evidence:
            'You connects Warning and HF Hub through semantic evidence: Warning --[co_occurs_with]-- You; You --[co_occurs_with]-- HF Hub',
        },
      ],
      starter_prompts: [
        {
          id: 'summary',
          kind: 'summary',
          label: 'Summarize trace',
          prompt: 'Summarize the document evidence.',
        },
      ],
    };

    const best = pickBestGraphRAGTraceSample(sampleOutputs, [
      'retrieval_rehearsals',
      'graph_anchors',
      'starter_prompts',
    ]);

    expect(best?.key).toBe('starter_prompts');
    expect(best?.record).toMatchObject({
      id: 'summary',
    });
  });
});
