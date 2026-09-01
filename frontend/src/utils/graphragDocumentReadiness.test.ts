import { describe, expect, it } from 'vitest';
import { buildGraphRAGDocumentReadiness } from './graphragDocumentReadiness';

describe('graphrag document readiness', () => {
  it('turns pipeline lanes into a chat-ready summary', () => {
    const readiness = buildGraphRAGDocumentReadiness({
      lanes: [
        {
          key: 'content',
          label: 'Extracted content',
          status: 'complete',
          sampleCount: 27,
          sampleCountLabel: '27 items',
          sampleTitle: 'Title slide',
        },
        {
          key: 'relationships',
          label: 'Graph links',
          status: 'complete',
          sampleCount: 220,
          sampleCountLabel: '220 links',
          sampleBody: 'Revenue Forecast owner Ada Lovelace',
        },
        {
          key: 'chunks',
          label: 'Retrieval chunks',
          status: 'complete',
          sampleCount: 27,
          sampleCountLabel: '27 chunks',
        },
      ],
      evidence: [
        { label: 'Semantic', value: 6 },
        { label: 'Chart', value: 5 },
      ],
      questionCount: 2,
    });

    expect(readiness).toMatchObject({
      headline: 'Ready for chat',
      detail: '3/3 checks · 274 items',
      statusLabel: '3/3 ready',
      tone: 'success',
      chips: ['6 Semantic', '5 Chart', '2 suggested questions'],
    });
    expect(readiness.cards.map(card => card.label)).toEqual([
      'Relationships ready',
      'References ready',
      'Text parsed',
    ]);
    expect(readiness.cards[0]).toMatchObject({
      title: '220 relationships',
      detail: 'Revenue Forecast owner Ada Lovelace',
      countLabel: '220 links',
      statusLabel: 'Complete',
    });
    expect(readiness.primaryCard).toMatchObject({
      key: 'relationships',
      label: 'Relationships ready',
      title: '220 relationships',
      countLabel: '220 links',
    });
  });

  it('surfaces warnings before claiming the import is fully ready', () => {
    const readiness = buildGraphRAGDocumentReadiness({
      lanes: [
        {
          key: 'content',
          label: 'Extracted content',
          status: 'complete',
          sampleCount: 8,
        },
        {
          key: 'paths',
          label: 'Reasoning paths',
          status: 'preview_only',
          sampleCount: 2,
          warningCount: 1,
        },
      ],
    });

    expect(readiness.headline).toBe('Import needs review');
    expect(readiness.tone).toBe('warning');
    expect(readiness.detail).toContain('1 issue');
    expect(readiness.cards.find(card => card.key === 'paths')).toMatchObject({
      statusLabel: '1 issue',
      tone: 'warning',
    });
  });

  it('uses a meaningful relationship sample title before aggregate counts', () => {
    const readiness = buildGraphRAGDocumentReadiness({
      lanes: [
        {
          key: 'relationships',
          label: 'Graph links',
          status: 'complete',
          sampleCount: 43,
          sampleCountLabel: '43 links',
          sampleTitle: 'Links To: https://example.com/evidence-index',
          sampleBody: 'GraphRAG evidence links to the evidence index.',
        },
        {
          key: 'chunks',
          label: 'Retrieval chunks',
          status: 'complete',
          sampleCount: 8,
          sampleCountLabel: '8 chunks',
        },
      ],
    });

    expect(readiness.primaryCard).toMatchObject({
      key: 'relationships',
      title: 'Links To: https://example.com/evidence-index',
      countLabel: '43 links',
    });
  });

  it('uses the first meaningful evidence card when graph samples are weak boilerplate', () => {
    const readiness = buildGraphRAGDocumentReadiness({
      lanes: [
        {
          key: 'relationships',
          label: 'Graph links',
          status: 'complete',
          sampleCount: 19,
          sampleTitle: 'Co Occurs With: Warning -> You',
          sampleBody: 'Warning appears with You in an HF Hub authentication notice.',
        },
        {
          key: 'chunks',
          label: 'Retrieval chunks',
          status: 'complete',
          sampleCount: 12,
          sampleTitle: 'TURBOQUANT PHASE A',
          sampleBody: 'REAL MODEL VALIDATION Model: Qwen/Qwen3-1.7B result: pass.',
        },
      ],
    });

    expect(readiness.cards[0]).toMatchObject({
      key: 'relationships',
    });
    expect(readiness.primaryCard).toMatchObject({
      key: 'chunks',
      label: 'References ready',
      detail: 'REAL MODEL VALIDATION Model: Qwen/Qwen3-1.7B result: pass.',
    });
  });
});
