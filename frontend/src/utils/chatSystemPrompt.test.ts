import { describe, expect, it } from 'vitest';

import { applyKnowledgeSystemPrompt, isExplicitGraphRAGQuestion } from './chatSystemPrompt';

describe('applyKnowledgeSystemPrompt', () => {
  it('keeps graph context when updating an existing system message with reasoning', () => {
    const messages = [
      { role: 'system', content: 'Answer as an analyst. Reasoning: low' },
      { role: 'user', content: 'How are Alpha and Omega connected?' },
    ];

    const result = applyKnowledgeSystemPrompt(messages, {
      contextText: '\n\n--- Unified Knowledge Context ---\n=== Graph Paths ===\nAlpha --[supports]-- Omega\n--- End Knowledge Context ---',
      reasoningLevel: 'high',
    });

    expect(result[0].content).toContain('Answer as an analyst.');
    expect(result[0].content).toContain('Reasoning: high');
    expect(result[0].content).toContain('GraphRAG answer contract:');
    expect(result[0].content).toContain('Question mode: explicit graph reasoning.');
    expect(result[0].content).toContain('lead with graph relationships, graph paths, and hop labels');
    expect(result[0].content).toContain('ordinary document questions');
    expect(result[0].content).toContain('source-document evidence first');
    expect(result[0].content).toContain('explicitly asks for a graph path');
    expect(result[0].content).toContain('directly answer an explicit graph');
    expect(result[0].content).toContain('follow its strategy and ordered hops for explicit graph/path questions');
    expect(result[0].content).toContain('Evidence checklist');
    expect(result[0].content).toContain('citation order');
    expect(result[0].content).toContain('answer from that path first only for explicit path/connection questions');
    expect(result[0].content).toContain('ordinary source, citation, or document-summary questions');
    expect(result[0].content).toContain('retrieved passage or source-document evidence first');
    expect(result[0].content).toContain('Prefer a direct graph relationship over an indirect graph path');
    expect(result[0].content).toContain('For grouped visual evidence');
    expect(result[0].content).toContain('specific cited evidence label');
    expect(result[0].content).toContain('chart series/value');
    expect(result[0].content).toContain('address each requested family explicitly');
    expect(result[0].content).toContain('Requested Evidence Coverage section is present');
    expect(result[0].content).toContain('do not say a listed family is missing');
    expect(result[0].content).toContain('direct graph relationship');
    expect(result[0].content).toContain('indirect graph path');
    expect(result[0].content).toContain('Graph path sentence');
    expect(result[0].content).toContain('source-document evidence');
    expect(result[0].content).toContain('source document/source evidence nodes as provenance bridges');
    expect(result[0].content).toContain('do not answer as though nothing was found');
    expect(result[0].content).toContain('source-first and plain-language');
    expect(result[0].content).toContain('upload trace summaries');
    expect(result[0].content).toContain('compact enough to complete');
    expect(result[0].content).toContain('Do not expose internal retrieval details');
    expect(result[0].content).toContain('run ids');
    expect(result[0].content).toContain('chunk match score');
    expect(result[0].content).toContain('Never print raw numeric graph scores');
    expect(result[0].content).toContain('=== Graph Paths ===');
    expect(result[0].content).toContain('Alpha --[supports]-- Omega');
  });

  it('uses document-answer mode when citation wording is not a graph request', () => {
    const messages = [
      {
        role: 'user',
        content: 'What validation result does this document report? Cite the source evidence.',
      },
    ];

    const result = applyKnowledgeSystemPrompt(messages, {
      contextText: [
        '\n\n--- Unified Knowledge Context ---',
        '=== Knowledge Context ===',
        'TURBOQUANT PHASE A: REAL MODEL VALIDATION Model: Qwen/Qwen3-1.7B',
        '=== Relevant Graph Relationships ===',
        'REAL MODEL VALIDATION Model --[co_occurs_with]--> Qwen/Qwen3-1.7B',
        '--- End Knowledge Context ---',
      ].join('\n'),
      reasoningLevel: 'none',
    });

    expect(result[0].content).toContain('Question mode: document answer.');
    expect(result[0].content).toContain('Citation, evidence, source, and support words alone do not make this a graph/path question.');
    expect(result[0].content).toContain('Lead with the answer from the selected document');
    expect(result[0].content).toContain('Do not lead with or create headings named Direct Graph Relationship');
  });

  it('keeps explicit bridge/path wording in graph-answer mode', () => {
    const query = 'Trace the path from Approval required before launch workflow to Launch workflow through Approval Dependency Cluster. Show each hop label and cite the slide evidence.';

    const result = applyKnowledgeSystemPrompt([{ role: 'user', content: query }], {
      contextText: [
        '\n\n--- Unified Knowledge Context ---',
        '=== Graph Answer Plan ===',
        'Strategy: walk_strongest_path',
        'Ordered hops:',
        '  1. Approval required before launch workflow --[supports]-- Approval Dependency Cluster',
        '  2. Approval Dependency Cluster --[supports]-- Launch workflow',
        '--- End Knowledge Context ---',
      ].join('\n'),
      reasoningLevel: 'none',
    });

    expect(isExplicitGraphRAGQuestion(query)).toBe(true);
    expect(result[0].content).toContain('Question mode: explicit graph reasoning.');
    expect(result[0].content).not.toContain('Question mode: document answer.');
  });

  it('treats plain connect wording as explicit graph reasoning', () => {
    const query = 'How does Product Revenue connect to Regional Demand and Risk Map? Cite the evidence.';

    const result = applyKnowledgeSystemPrompt([{ role: 'user', content: query }], {
      contextText: [
        '\n\n--- Unified Knowledge Context ---',
        '=== Graph Paths ===',
        'Product Revenue --[drives]-- Regional Demand --[references]-- Risk Map',
        '--- End Knowledge Context ---',
      ].join('\n'),
      reasoningLevel: 'none',
    });

    expect(isExplicitGraphRAGQuestion(query)).toBe(true);
    expect(result[0].content).toContain('Question mode: explicit graph reasoning.');
  });

  it('adds graph reasoning instructions for an explicit graph answer plan', () => {
    const messages = [{ role: 'user', content: 'Explain the connection.' }];

    const result = applyKnowledgeSystemPrompt(messages, {
      contextText: [
        '\n\n--- Unified Knowledge Context ---',
        '=== Graph Answer Plan ===',
        'Strategy: walk_strongest_path',
        'Ordered hops:',
        '  1. Alpha --[supports]-- Omega',
        'Evidence checklist:',
        '  - Hop 1: Alpha --[supports]--> Omega | cite: slide 2',
        '--- End Knowledge Context ---',
      ].join('\n'),
      reasoningLevel: 'none',
    });

    expect(result[0].content).toContain('GraphRAG answer contract:');
    expect(result[0].content).toContain('follow its strategy and ordered hops for explicit graph/path questions');
    expect(result[0].content).toContain('use it as the citation order');
    expect(result[0].content).toContain('=== Graph Answer Plan ===');
    expect(result[0].content).toContain('Strategy: walk_strongest_path');
    expect(result[0].content).toContain('Evidence checklist:');
  });

  it('treats requested evidence coverage as GraphRAG context', () => {
    const messages = [{ role: 'user', content: 'Which evidence families were extracted?' }];

    const result = applyKnowledgeSystemPrompt(messages, {
      contextText: [
        '=== Requested Evidence Coverage ===',
        'The user explicitly asked about: Form Fields; Speaker Notes.',
        '- Form Fields: label: Approval required; checked: true',
        '- Speaker Notes: Speaker note: cite the Finance Review Board approval evidence.',
      ].join('\n'),
      reasoningLevel: 'none',
    });

    expect(result[0].content).toContain('GraphRAG answer contract:');
    expect(result[0].content).toContain('Requested Evidence Coverage section is present');
    expect(result[0].content).toContain('Approval required');
    expect(result[0].content).toContain('Speaker note');
  });

  it('instructs GraphRAG answers to keep raw scoring details out of user-facing prose', () => {
    const messages = [{ role: 'user', content: 'What connects Product Revenue to Regional Demand?' }];

    const result = applyKnowledgeSystemPrompt(messages, {
      contextText: [
        '\n\n--- Unified Knowledge Context ---',
        '=== Graph Paths ===',
        'Product Revenue --[drives]-- Regional Demand',
        'score=0.94 match=14 chunk_match_reasons=product,revenue',
        '--- End Knowledge Context ---',
      ].join('\n'),
      reasoningLevel: 'none',
    });

    expect(result[0].content).toContain('source-first and plain-language');
    expect(result[0].content).toContain('Do not expose internal retrieval details');
    expect(result[0].content).toContain('chunk match score');
    expect(result[0].content).toContain('path score');
    expect(result[0].content).toContain('token match reasons');
    expect(result[0].content).toContain('Pipeline Trace');
    expect(result[0].content).toContain('79% score');
    expect(result[0].content).toContain('explain confidence through source agreement');
    expect(result[0].content).toContain('not through semantic matching or token matching');
    expect(result[0].content).toContain('Leave raw scoring details to the source review UI');
    expect(result[0].content).toContain('score=0.94 match=14');
  });

  it('adds graph reasoning instructions for a pipeline interaction handoff', () => {
    const messages = [{ role: 'user', content: 'Explain the uploaded chart.' }];

    const result = applyKnowledgeSystemPrompt(messages, {
      contextText: [
        '\n\n--- Unified Knowledge Context ---',
        '=== Pipeline Interaction Handoff ===',
        'Run: chart-deck.pptx | mode: graph_path_reasoning | ready: true',
        '  Graph anchors:',
        '    - relationship: Revenue Q2 has value 18 | evidence: Revenue Q2 = 18',
        '  Retrieval rehearsals:',
        '    - Explain Q2 chart | ready | anchors: 1 | chunks: 1 | matched: revenue, q2, chart',
        '--- End Knowledge Context ---',
      ].join('\n'),
      reasoningLevel: 'none',
    });

    expect(result[0].content).toContain('GraphRAG answer contract:');
    expect(result[0].content).toContain('Pipeline Interaction Handoff');
    expect(result[0].content).toContain('use its graph anchors, lanes, starter prompts, retrieval rehearsals, and citation scopes silently');
    expect(result[0].content).toContain('do not describe the handoff mechanics');
    expect(result[0].content).toContain('Explain Q2 chart | ready | anchors: 1 | chunks: 1');
    expect(result[0].content).toContain('Revenue Q2 has value 18');
  });

  it('adds graph-only context even when reasoning is disabled', () => {
    const messages = [{ role: 'user', content: 'Where is the risk chart?' }];

    const result = applyKnowledgeSystemPrompt(messages, {
      contextText: '\n\n--- Knowledge Graph Context ---\nRelevant graph relationships:\nSlide 4 --[contains_chart]-- Risk Chart\n--- End Knowledge Graph Context ---',
      reasoningLevel: 'none',
    });

    expect(result[0].role).toBe('system');
    expect(result[0].content).toContain('GraphRAG answer contract:');
    expect(result[0].content).toContain('explicit connection');
    expect(result[0].content).toContain('relationship chain hop by hop');
    expect(result[0].content).toContain('Relevant graph relationships');
    expect(result[0].content).toContain('Slide 4 --[contains_chart]-- Risk Chart');
    expect(result[0].content).not.toContain('Reasoning:');
  });

  it('creates a new system message with context and reasoning', () => {
    const messages = [{ role: 'user', content: 'Summarize the graph evidence.' }];

    const result = applyKnowledgeSystemPrompt(messages, {
      contextText: '\n\n--- Unified Knowledge Context ---\nGraph evidence\n--- End Knowledge Context ---',
      reasoningLevel: 'medium',
    });

    expect(result[0].content).toContain('You are a helpful AI assistant.');
    expect(result[0].content).toContain('Graph evidence');
    expect(result[0].content).toContain('Reasoning: medium');
    expect(result[1]).toBe(messages[0]);
  });

  it('does not add graph reasoning instructions for plain retrieved context', () => {
    const messages = [{ role: 'user', content: 'What does the document say?' }];

    const result = applyKnowledgeSystemPrompt(messages, {
      contextText: '\n\n--- Retrieved Context ---\n[1] The handbook lists deployment steps.\n--- End Context ---',
      reasoningLevel: 'none',
    });

    expect(result[0].content).toContain('--- Retrieved Context ---');
    expect(result[0].content).not.toContain('GraphRAG answer contract:');
  });

  it('does not duplicate graph reasoning instructions on an existing system message', () => {
    const messages = [
      {
        role: 'system',
        content: [
          'Answer as an analyst.',
          'GraphRAG answer contract:',
          '- Treat graph paths and relationships as structured evidence, not as ordinary retrieved snippets.',
        ].join('\n'),
      },
      { role: 'user', content: 'How are Alpha and Omega connected?' },
    ];

    const result = applyKnowledgeSystemPrompt(messages, {
      contextText: '\n\n--- Unified Knowledge Context ---\n=== Graph Paths ===\nAlpha --[supports]-- Omega\n--- End Knowledge Context ---',
      reasoningLevel: 'medium',
    });

    const content = String(result[0].content);
    expect(content.match(/GraphRAG answer contract:/g)).toHaveLength(1);
    expect(content).toContain('Reasoning: medium');
    expect(content).toContain('Alpha --[supports]-- Omega');
  });

  it('does not add a system message when no context or reasoning is requested', () => {
    const messages = [{ role: 'user', content: 'Hello' }];

    const result = applyKnowledgeSystemPrompt(messages, {
      contextText: '',
      reasoningLevel: 'none',
    });

    expect(result).toBe(messages);
  });
});
