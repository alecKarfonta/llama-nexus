import { describe, expect, it } from 'vitest';
import {
  buildGraphRAGBatchPrompt,
  buildGraphRAGBatchQuestions,
} from './graphragBatchPrompts';

describe('graphrag batch prompts', () => {
  const overview = {
    relationship_count: 4,
    bridge_entity_count: 1,
    cross_document_path_count: 1,
    shared_entities: [
      {
        name: 'Revenue Forecast',
        run_count: 2,
        filenames: ['strategy.pptx', 'scorecard.xlsx'],
        relationships: ['drives', 'tracks'],
      },
    ],
    bridge_entities: [
      {
        name: 'Revenue Forecast',
        run_count: 2,
        filenames: ['strategy.pptx', 'scorecard.xlsx'],
        relationship_count: 4,
        relationships: ['drives', 'tracks', 'influences'],
      },
    ],
    cross_document_paths: [
      {
        source: 'Regional Demand',
        bridge: 'Revenue Forecast',
        target: 'Launch Risk',
        filenames: ['strategy.pptx', 'scorecard.xlsx'],
        text: 'Regional Demand --[drives]-- Revenue Forecast --[influences]-- Launch Risk',
      },
    ],
    top_relationships: [
      {
        source: 'Revenue Forecast',
        relation: 'influences',
        target: 'Launch Risk',
      },
    ],
  };

  it('builds a summary prompt with batch graph overview counts', () => {
    const prompt = buildGraphRAGBatchPrompt({
      filenames: ['strategy.pptx', 'scorecard.xlsx'],
      overview,
    });

    expect(prompt).toContain('strategy.pptx, scorecard.xlsx');
    expect(prompt).toContain('4 graph relationships');
    expect(prompt).toContain('1 bridge entities');
    expect(prompt).toContain('1 cross-document paths');
    expect(prompt).toContain('Cite filename plus slide/page/table/chart/connector provenance');
  });

  it('suggests path, bridge, shared entity, and relationship questions', () => {
    const questions = buildGraphRAGBatchQuestions({
      filenames: ['strategy.pptx', 'scorecard.xlsx'],
      overview,
    }, 5);

    expect(questions.map(question => question.kind)).toEqual([
      'summary',
      'path',
      'bridge',
      'shared_entity',
      'relationship',
    ]);
    expect(questions[1]).toMatchObject({
      source: 'Regional Demand',
      bridge: 'Revenue Forecast',
      target: 'Launch Risk',
    });
    expect(questions[2]).toMatchObject({ bridge: 'Revenue Forecast' });
    expect(questions[3]).toMatchObject({ source: 'Revenue Forecast' });
    expect(questions[4]).toMatchObject({
      source: 'Revenue Forecast',
      relation: 'influences',
      target: 'Launch Risk',
    });
    expect(questions[1].prompt).toContain('"Regional Demand" -> "Revenue Forecast" -> "Launch Risk"');
    expect(questions[2].prompt).toContain('"Revenue Forecast" as a bridge entity');
    expect(questions[3].prompt).toContain('shared entity "Revenue Forecast"');
    expect(questions.every(question => question.prompt.includes('strategy.pptx'))).toBe(true);
  });

  it('prefers backend suggested questions when provided by the batch overview', () => {
    const questions = buildGraphRAGBatchQuestions({
      filenames: ['strategy.pptx', 'scorecard.xlsx'],
      overview: {
        ...overview,
        suggested_questions: [
          {
            id: 'backend-path',
            kind: 'path',
            label: 'Backend path',
            prompt: 'Use backend-ranked cross-document path evidence.',
            source: 'Regional Demand',
            bridge: 'Revenue Forecast',
            target: 'Launch Risk',
          },
        ],
      },
    }, 3);

    expect(questions[0]).toMatchObject({
      id: 'backend-path',
      kind: 'path',
      label: 'Backend path',
      source: 'Regional Demand',
      bridge: 'Revenue Forecast',
      target: 'Launch Risk',
    });
    expect(questions[0].prompt).toContain('backend-ranked');
  });
});
