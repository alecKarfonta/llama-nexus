import { describe, expect, it } from 'vitest';
import { getNodeIconSpriteWorldSize, getNodeLabelOffset } from './knowledgeGraphLabelLayout';

describe('knowledge graph label layout', () => {
  it('keeps vertical labels clear of large visible node symbols', () => {
    const size = 6;
    const labelScale = { x: 68, y: 18 };
    const iconSize = getNodeIconSpriteWorldSize(size, false, false, 1.2);
    const above = getNodeLabelOffset('above', labelScale, size, false, false, true, 1.2, 'rich');
    const below = getNodeLabelOffset('below', labelScale, size, false, false, true, 1.2, 'rich');
    const iconRadius = iconSize * 0.5;
    const labelHalfHeight = labelScale.y * 0.5;

    expect(above.y).toBeGreaterThan(iconRadius + labelHalfHeight + 5);
    expect(below.y).toBeLessThan(-(iconRadius + labelHalfHeight + 5));
  });

  it('pushes diagonal labels away from the node symbol and keeps anchors mirrored', () => {
    const size = 5.5;
    const labelScale = { x: 82, y: 20 };
    const aboveLeft = getNodeLabelOffset('above-left', labelScale, size, true, false, true, 1.1, 'annotation');
    const aboveRight = getNodeLabelOffset('above-right', labelScale, size, true, false, true, 1.1, 'annotation');
    const belowRight = getNodeLabelOffset('below-right', labelScale, size, true, false, true, 1.1, 'annotation');

    expect(aboveLeft.x).toBeLessThan(-48);
    expect(aboveLeft.y).toBeGreaterThan(24);
    expect(aboveRight.x).toBeCloseTo(-aboveLeft.x, 6);
    expect(aboveRight.y).toBeCloseTo(aboveLeft.y, 6);
    expect(belowRight.x).toBeCloseTo(aboveRight.x, 6);
    expect(belowRight.y).toBeCloseTo(-aboveRight.y, 6);
  });

  it('keeps side labels lifted but outside the node hit target', () => {
    const size = 7;
    const labelScale = { x: 72, y: 18 };
    const left = getNodeLabelOffset('left', labelScale, size, false, true, true, 1, 'compact');
    const right = getNodeLabelOffset('right', labelScale, size, false, true, true, 1, 'compact');

    expect(left.x).toBeLessThan(-50);
    expect(right.x).toBeGreaterThan(50);
    expect(left.y).toBeGreaterThan(2);
    expect(left.y).toBeCloseTo(right.y, 6);
    expect(left.x).toBeCloseTo(-right.x, 6);
  });
});
