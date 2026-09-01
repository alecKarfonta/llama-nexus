import { describe, expect, it } from 'vitest';
import { blendGraphSearchResults } from './knowledgeGraphSearch';

describe('knowledge graph search blending', () => {
  it('interleaves entity and relation hits so relations are visible immediately', () => {
    expect(
      blendGraphSearchResults(['entity-a', 'entity-b', 'entity-c'], ['relation-a', 'relation-b'])
    ).toEqual(['entity-a', 'relation-a', 'entity-b', 'relation-b', 'entity-c']);
  });

  it('preserves primary-only result order', () => {
    expect(blendGraphSearchResults(['entity-a', 'entity-b'], [])).toEqual(['entity-a', 'entity-b']);
  });

  it('respects result limits while blending', () => {
    expect(
      blendGraphSearchResults(['entity-a', 'entity-b', 'entity-c'], ['relation-a', 'relation-b'], 3)
    ).toEqual(['entity-a', 'relation-a', 'entity-b']);
  });
});
