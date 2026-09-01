import { describe, expect, it } from 'vitest';
import {
  buildGraphRAGDocumentPrompt,
  buildGraphRAGDocumentQuestions,
} from './graphragDocumentPrompts';

describe('graphrag document prompts', () => {
  it('builds a summary prompt with document graph counts', () => {
    const prompt = buildGraphRAGDocumentPrompt({
      filename: 'strategy.pptx',
      overview: {
        chunk_count: 27,
        entity_candidate_count: 40,
        relationship_candidate_count: 220,
      },
    });

    expect(prompt).toContain('strategy.pptx');
    expect(prompt).toContain('27 chunks');
    expect(prompt).toContain('40 entities');
    expect(prompt).toContain('220 relationships');
    expect(prompt).toContain('connect ideas instead of only quoting passages');
  });

  it('suggests summary, graph relationship, and provenance questions', () => {
    const questions = buildGraphRAGDocumentQuestions({
      filename: 'strategy.pptx',
      overview: {
        relationship_candidate_count: 220,
      },
      graphRows: [
        {
          kind: 'relationship',
          source: 'Regional Demand',
          relation: 'drives',
          target: 'Revenue Forecast',
        },
      ],
      evidenceFamilies: [{ label: 'Chart edges', value: 12 }],
    });

    expect(questions.map((question) => question.kind)).toEqual([
      'summary',
      'relationship',
      'summary',
    ]);
    expect(questions[1]).toMatchObject({
      source: 'Regional Demand',
      relation: 'drives',
      target: 'Revenue Forecast',
    });
    expect(questions[1].prompt).toContain(
      '"Regional Demand" --[drives]--> "Revenue Forecast"',
    );
    expect(questions[2].label).toBe('Show Chart edges');
    expect(questions[2].prompt).toContain('page, slide, table, chart');
  });

  it('does not turn weak co-occurrence edges into starter graph questions', () => {
    const questions = buildGraphRAGDocumentQuestions({
      filename: 'turboquant_validation_output.txt',
      overview: {
        relationship_candidate_count: 9,
      },
      graphRows: [
        {
          kind: 'relationship',
          source: 'Warning',
          relation: 'co_occurs_with',
          target: 'You',
        },
        {
          kind: 'relationship',
          source: 'HF Hub',
          relation: 'co_occurs_with',
          target: 'Please',
        },
      ],
      evidenceFamilies: [{ label: 'Semantic edges', value: 9 }],
    });

    expect(questions.map((question) => question.label)).toEqual([
      'Summarize document',
      'Show source evidence',
    ]);
    expect(questions.every((question) => question.kind === 'summary')).toBe(true);
    expect(questions.map((question) => question.prompt).join('\n')).not.toContain(
      '"Warning" --[co_occurs_with]--> "You"',
    );
  });

  it('builds path questions and keeps preferred questions first', () => {
    const questions = buildGraphRAGDocumentQuestions(
      {
        filename: 'deck.pptx',
        preferredQuestions: [
          {
            id: 'backend-summary',
            label: 'Backend summary',
            kind: 'summary',
            prompt: 'Use backend-ranked starter evidence.',
          },
        ],
        graphRows: [
          {
            kind: 'path',
            source: 'Revenue Forecast',
            bridge: 'Regional Demand',
            target: 'Launch Risk',
          },
        ],
      },
      3,
    );

    expect(questions[0]).toMatchObject({
      id: 'backend-summary',
      label: 'Backend summary',
    });
    expect(questions[1].kind).toBe('summary');
    expect(questions[2]).toMatchObject({
      kind: 'path',
      source: 'Revenue Forecast',
      bridge: 'Regional Demand',
      target: 'Launch Risk',
    });
    expect(questions[2].prompt).toContain(
      '"Revenue Forecast" -> "Regional Demand" -> "Launch Risk"',
    );
  });

  it('filters weak backend-preferred trace prompts before they reach the UI', () => {
    const questions = buildGraphRAGDocumentQuestions({
      filename: 'turboquant_validation_output.txt',
      preferredQuestions: [
        {
          id: 'weak-backend-path',
          label: 'Trace Warning via You',
          kind: 'path',
          source: 'Warning',
          bridge: 'You',
          target: 'HF Hub',
          prompt: 'Trace Warning through You to HF Hub.',
        },
        {
          id: 'weak-backend-relationship',
          label: 'Warning to You',
          kind: 'relationship',
          source: 'Warning',
          relation: 'co_occurs_with',
          target: 'You',
          prompt: 'Explain Warning co-occurs with You.',
        },
      ],
      overview: { relationship_candidate_count: 9 },
      graphRows: [
        {
          kind: 'relationship',
          source: 'Warning',
          relation: 'co_occurs_with',
          target: 'You',
        },
      ],
      evidenceFamilies: [{ label: 'Semantic edges', value: 9 }],
    });

    expect(questions.map((question) => question.label)).toEqual([
      'Summarize document',
      'Show source evidence',
    ]);
  });

  it('filters structural document-section paths from primary starter prompts', () => {
    const questions = buildGraphRAGDocumentQuestions(
      {
        filename: 'graphrag-upload-smoke.md',
        preferredQuestions: [
          {
            id: 'product-regional',
            label: 'Product Revenue to Regional Demand',
            kind: 'relationship',
            source: 'Product Revenue',
            relation: 'drives',
            target: 'Regional Demand',
            prompt: 'Explain Product Revenue drives Regional Demand.',
          },
          {
            id: 'source-section-path',
            label: 'Path: graphrag-upload-smoke.md via GraphRAG Upload Smoke Deck',
            kind: 'path',
            source: 'graphrag-upload-smoke.md',
            bridge: 'GraphRAG Upload Smoke Deck',
            target: 'Product Revenue drives Regional Demand.',
            prompt: 'Explain the source document to section to sentence path.',
          },
          {
            id: 'regional-risk',
            label: 'Regional Demand to Risk Map',
            kind: 'relationship',
            source: 'Regional Demand',
            relation: 'references',
            target: 'Risk Map',
            prompt: 'Explain Regional Demand references Risk Map.',
          },
        ],
        evidenceFamilies: [{ label: 'Semantic edges', value: 20 }],
      },
      4,
    );

    expect(questions.map((question) => question.id)).toEqual([
      'product-regional',
      'regional-risk',
      'document-summary',
      'document-provenance',
    ]);
    expect(questions.map((question) => question.label).join('\n')).not.toContain(
      'graphrag-upload-smoke.md via GraphRAG Upload Smoke Deck',
    );
  });
});
