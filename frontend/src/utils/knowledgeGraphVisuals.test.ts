import { describe, expect, it } from 'vitest';
import {
  getKnowledgeGraphAmbientLabelLimit,
  getKnowledgeGraphAnchorLabelLimit,
  getKnowledgeGraphEntityVisualType,
  getKnowledgeGraphPriorityLabelQualityFloor,
  getKnowledgeGraphRelationVisualKind,
} from './knowledgeGraphVisuals';

describe('knowledge graph visuals', () => {
  it('keeps backend GraphRAG entity variants visually meaningful', () => {
    expect(getKnowledgeGraphEntityVisualType('solution', 'Solution')).toBe('product');
    expect(getKnowledgeGraphEntityVisualType('product', 'Solution')).toBe('product');
    expect(getKnowledgeGraphEntityVisualType('specification', 'Speculative decoding format')).toBe('concept');
    expect(getKnowledgeGraphEntityVisualType('requirement', 'Context Size')).toBe('config');
    expect(getKnowledgeGraphEntityVisualType('safety', 'API token guardrail')).toBe('security');
    expect(getKnowledgeGraphEntityVisualType('organisation', 'NVIDIA')).toBe('organization');
  });

  it('uses labels to refine generic technical entities', () => {
    expect(getKnowledgeGraphEntityVisualType('technology', 'Qwen/Qwen3.5-9B')).toBe('model');
    expect(getKnowledgeGraphEntityVisualType('concept', 'GraphRAG retrieval index')).toBe('vector');
    expect(getKnowledgeGraphEntityVisualType('product', 'Docker deployment workflow')).toBe('runtime');
  });

  it('lets specific technical labels beat vendor organization names', () => {
    expect(getKnowledgeGraphEntityVisualType('concept', 'OpenAI API endpoint')).toBe('api');
    expect(getKnowledgeGraphEntityVisualType('concept', 'Microsoft OAuth token')).toBe('security');
    expect(getKnowledgeGraphEntityVisualType('technology', 'NVIDIA H100 GPU')).toBe('hardware');
    expect(getKnowledgeGraphEntityVisualType('technology', 'HuggingFace dataset cache')).toBe('data');
  });

  it('keeps standard GraphRAG graphs readable without rich-label mode', () => {
    const limit = getKnowledgeGraphAmbientLabelLimit('auto', 'priority', 'standard', 'far', 1800, 58, 222);
    expect(limit).toBeGreaterThanOrEqual(32);
    expect(limit).toBeLessThanOrEqual(58);
  });

  it('still throttles labels for performance and focus modes', () => {
    expect(getKnowledgeGraphAmbientLabelLimit('auto', 'focus', 'standard', 'far', 1800, 58, 222)).toBe(0);
    expect(getKnowledgeGraphAmbientLabelLimit('performance', 'priority', 'standard', 'far', 1800, 58, 222)).toBeLessThan(32);
    expect(getKnowledgeGraphAnchorLabelLimit('auto', 'focus', 'standard', 'middle', 1800, 222)).toBe(28);
  });

  it('uses lower label quality floors for standard graphs than massive graphs', () => {
    expect(getKnowledgeGraphPriorityLabelQualityFloor('standard')).toBeLessThan(getKnowledgeGraphPriorityLabelQualityFloor('massive'));
  });

  it('maps real GraphRAG relationship verbs to useful visual families', () => {
    expect(getKnowledgeGraphRelationVisualKind('manages')).toBe('controls');
    expect(getKnowledgeGraphRelationVisualKind('operates')).toBe('controls');
    expect(getKnowledgeGraphRelationVisualKind('regulates')).toBe('controls');
    expect(getKnowledgeGraphRelationVisualKind('monitors')).toBe('evaluates');
    expect(getKnowledgeGraphRelationVisualKind('part of')).toBe('contains');
    expect(getKnowledgeGraphRelationVisualKind('contains')).toBe('contains');
    expect(getKnowledgeGraphRelationVisualKind('depends on')).toBe('supports');
    expect(getKnowledgeGraphRelationVisualKind('requires')).toBe('supports');
    expect(getKnowledgeGraphRelationVisualKind('enables')).toBe('supports');
    expect(getKnowledgeGraphRelationVisualKind('prevents')).toBe('guards');
    expect(getKnowledgeGraphRelationVisualKind('protects')).toBe('guards');
    expect(getKnowledgeGraphRelationVisualKind('replaces')).toBe('transforms');
    expect(getKnowledgeGraphRelationVisualKind('supplies')).toBe('creates');
    expect(getKnowledgeGraphRelationVisualKind('specifies')).toBe('evaluates');
    expect(getKnowledgeGraphRelationVisualKind('applies to')).toBe('evaluates');
    expect(getKnowledgeGraphRelationVisualKind('connects to')).toBe('flows');
    expect(getKnowledgeGraphRelationVisualKind('associated with')).toBe('connects');
  });
});
