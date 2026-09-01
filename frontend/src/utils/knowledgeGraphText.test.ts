import { describe, expect, it } from 'vitest';
import {
  formatGraphEvidenceSnippetText,
  getGraphTextSearchSnippet,
  normalizeGraphDisplayText,
  normalizeGraphSearchResultText,
  normalizeGraphText,
} from './knowledgeGraphText';

describe('knowledge graph text utilities', () => {
  it('normalizes whitespace without dropping graph text characters', () => {
    expect(normalizeGraphText('DenseEngine  uses\tGraphRAG\nand system context')).toBe(
      'DenseEngine uses GraphRAG and system context'
    );
  });

  it('normalizes markdown-like graph evidence into readable display text', () => {
    expect(normalizeGraphDisplayText('The **system** uses `DenseEngine` and *GraphRAG*.')).toBe(
      'The system uses DenseEngine and GraphRAG.'
    );
  });

  it('normalizes html-like graph evidence into readable display text', () => {
    expect(
      normalizeGraphDisplayText('<p>GraphRAG &amp; DenseEngine<br/>use context&nbsp;engine evidence.</p>')
    ).toBe('GraphRAG & DenseEngine use context engine evidence.');
  });

  it('returns a bounded search snippet while preserving the original casing', () => {
    expect(
      getGraphTextSearchSnippet(
        'The premise: stop benchmarking the model and start benchmarking the system with context engine evidence.',
        'system',
        12,
        18
      )
    ).toBe('...marking the system with context engi...');
  });

  it('matches queries case-insensitively', () => {
    expect(getGraphTextSearchSnippet('DenseEngine affects context engine', 'denseengine')).toBe(
      'DenseEngine affects context engine'
    );
  });

  it('returns clean snippets for markdown-like evidence text', () => {
    const snippet = getGraphTextSearchSnippet(
      'The premise: stop benchmarking the *model* and start benchmarking the **system** with `context engine` evidence.',
      'system',
      34,
      22
    );

    expect(snippet).toContain('model and start benchmarking the system with context engine');
    expect(snippet).not.toContain('*');
    expect(snippet).not.toContain('`');
  });

  it('returns clean snippets for html-like evidence text', () => {
    const snippet = getGraphTextSearchSnippet(
      '<div>GraphRAG &amp; DenseEngine<br>combine metadata, relation types, and entity labels.</div>',
      'relation',
      36,
      20
    );

    expect(snippet).toContain('combine metadata, relation types, and entity');
    expect(snippet).not.toContain('<');
    expect(snippet).not.toContain('&amp;');
  });

  it('removes divider noise from graph search result text', () => {
    expect(
      normalizeGraphSearchResultText(
        '| DatasetProcessor | TrainingManager | AdapterStore | ExportManager | +----------------------------------------------'
      )
    ).toBe('DatasetProcessor / TrainingManager / AdapterStore / ExportManager');
  });

  it('formats clipped GraphRAG evidence snippets without leading word fragments', () => {
    expect(
      formatGraphEvidenceSnippetText(
        'maCPP)\nThe primary inference engine supporting:\n- GGUF model format with multiple quantization levels'
      )
    ).toBe('... The primary inference engine supporting: - GGUF model format with multiple quantization levels');

    expect(
      formatGraphEvidenceSnippetText(
        'mory usage\n- Continuous batching for throughput\n- OpenAI-compatible API endpoints'
      )
    ).toBe('... Continuous batching for throughput - OpenAI-compatible API endpoints');

    expect(
      formatGraphEvidenceSnippetText(
        'odes**: Entity search, code search, reasoning\n- **Document Nodes**: Ingestion, chunking, processing'
      )
    ).toBe('... Entity search, code search, reasoning - Document Nodes: Ingestion, chunking, processing');
  });

  it('leaves complete evidence snippets readable and unprefixed', () => {
    expect(formatGraphEvidenceSnippetText('GraphRAG supports relation metadata rendering.')).toBe(
      'GraphRAG supports relation metadata rendering.'
    );
  });
});
