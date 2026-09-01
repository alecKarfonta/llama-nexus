export type KnowledgeGraphNodeLabelStyle = 'anchor' | 'annotation' | 'compact' | 'neighbor' | 'rich';

export type KnowledgeGraphNodeLabelAnchor =
  | 'above'
  | 'above-left'
  | 'above-right'
  | 'below'
  | 'below-left'
  | 'below-right'
  | 'left'
  | 'right';

export interface KnowledgeGraphLabelScale {
  x: number;
  y: number;
}

export interface KnowledgeGraphLabelOffset {
  x: number;
  y: number;
  z: number;
}

export const getNodeIconSpriteWorldSize = (
  size: number,
  highlighted: boolean,
  focused: boolean,
  worldScale = 1
) => Math.max(
  highlighted ? 25.4 : focused ? 23.6 : 21.8,
  size * (highlighted ? 3.78 : focused ? 3.58 : 3.36)
) * worldScale;

export const getNodeLabelOffset = (
  anchor: KnowledgeGraphNodeLabelAnchor,
  labelScale: KnowledgeGraphLabelScale,
  size: number,
  highlighted: boolean,
  focused: boolean,
  iconVisible: boolean,
  symbolWorldScale: number,
  labelStyle: KnowledgeGraphNodeLabelStyle
): KnowledgeGraphLabelOffset => {
  const labelHalfWidth = labelScale.x * 0.5;
  const labelHalfHeight = labelScale.y * 0.5;
  const iconSize = iconVisible
    ? getNodeIconSpriteWorldSize(size, highlighted, focused, symbolWorldScale)
    : size * 2.1;
  const meshClearance = size * (highlighted ? 1.24 : focused ? 1.14 : 1.04);
  const horizontalClearance = Math.max(meshClearance, iconSize * 0.5);
  const verticalClearance = Math.max(meshClearance, iconSize * 0.62);
  const diagonalClearance = Math.max(meshClearance, iconSize * 0.56);
  const sideGap = Math.max(size * 0.44, labelStyle === 'rich' ? 3.2 : labelStyle === 'annotation' ? 2.65 : 2.15);
  const verticalGap = Math.max(size * 0.42, labelStyle === 'rich' ? 3.25 : labelStyle === 'annotation' ? 2.8 : 2.35);
  const sideX = horizontalClearance + labelHalfWidth + sideGap;
  const verticalY = verticalClearance + labelHalfHeight + verticalGap;
  const horizontalLift = Math.min(verticalY * 0.34, Math.max(size * 0.58, 2.2));
  const diagonalX = diagonalClearance * 0.9 + labelHalfWidth * 0.9 + sideGap;
  const diagonalY = diagonalClearance * 0.86 + labelHalfHeight * 0.92 + verticalGap;

  if (anchor === 'left') return { x: -sideX, y: horizontalLift, z: 0 };
  if (anchor === 'above') return { x: 0, y: verticalY, z: 0 };
  if (anchor === 'below') return { x: 0, y: -verticalY, z: 0 };
  if (anchor === 'above-left') return { x: -diagonalX, y: diagonalY, z: 0 };
  if (anchor === 'above-right') return { x: diagonalX, y: diagonalY, z: 0 };
  if (anchor === 'below-left') return { x: -diagonalX, y: -diagonalY, z: 0 };
  if (anchor === 'below-right') return { x: diagonalX, y: -diagonalY, z: 0 };
  return { x: sideX, y: horizontalLift, z: 0 };
};
