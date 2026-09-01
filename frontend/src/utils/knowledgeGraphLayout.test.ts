import { describe, expect, it } from 'vitest';
import {
  buildCompactGraphTopology,
  getKnowledgeGraphVisibleGraph,
  type GraphLayoutPoint,
  type KnowledgeGraphLayoutEdge,
} from './knowledgeGraphLayout';

const distance = (a: GraphLayoutPoint, b: GraphLayoutPoint) =>
  Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

const getBounds = (points: GraphLayoutPoint[]) => {
  const bounds = points.reduce(
    (acc, point) => ({
      minX: Math.min(acc.minX, point.x),
      maxX: Math.max(acc.maxX, point.x),
      minY: Math.min(acc.minY, point.y),
      maxY: Math.max(acc.maxY, point.y),
      minZ: Math.min(acc.minZ, point.z),
      maxZ: Math.max(acc.maxZ, point.z),
    }),
    {
      minX: points[0].x,
      maxX: points[0].x,
      minY: points[0].y,
      maxY: points[0].y,
      minZ: points[0].z,
      maxZ: points[0].z,
    }
  );

  return {
    ...bounds,
    spanX: bounds.maxX - bounds.minX,
    spanY: bounds.maxY - bounds.minY,
    spanZ: bounds.maxZ - bounds.minZ,
  };
};

describe('knowledge graph layout', () => {
  it('spreads compact hub graphs enough for readable node labels', () => {
    const nodes = [
      { id: 'Solution' },
      ...Array.from({ length: 8 }, (_, index) => ({ id: `neighbor-${index + 1}` })),
    ];
    const edges = nodes.slice(1).map((node, index) => ({
      source: 'Solution',
      target: node.id,
      relationCount: index + 1,
    }));

    const topology = buildCompactGraphTopology(nodes, edges);
    const points = nodes.map((node) => topology.positions.get(node.id));

    expect(points.every(Boolean)).toBe(true);
    points.forEach((point) => {
      expect(Number.isFinite(point?.x)).toBe(true);
      expect(Number.isFinite(point?.y)).toBe(true);
      expect(Number.isFinite(point?.z)).toBe(true);
    });

    const solution = topology.positions.get('Solution')!;
    const neighborDistances = nodes.slice(1).map((node) => distance(solution, topology.positions.get(node.id)!));
    const bounds = getBounds(points as GraphLayoutPoint[]);

    expect(topology.metrics.get('Solution')).toEqual({ links: 8, relations: 36 });
    expect(Math.min(...neighborDistances)).toBeGreaterThan(44);
    expect(bounds.spanX).toBeGreaterThan(130);
    expect(bounds.spanY).toBeGreaterThan(86);
    expect(bounds.spanX / Math.max(1, bounds.spanY)).toBeGreaterThan(1.05);
  });

  it('keeps secondary components and isolated nodes outside the core cluster', () => {
    const nodes = [
      { id: 'hub' },
      { id: 'a' },
      { id: 'b' },
      { id: 'secondary-1' },
      { id: 'secondary-2' },
      { id: 'isolated' },
    ];
    const edges = [
      { source: 'hub', target: 'a', relationCount: 8 },
      { source: 'hub', target: 'b', relationCount: 5 },
      { source: 'secondary-1', target: 'secondary-2', relationCount: 2 },
    ];

    const topology = buildCompactGraphTopology(nodes, edges);
    const hub = topology.positions.get('hub')!;
    const secondary = topology.positions.get('secondary-1')!;
    const isolated = topology.positions.get('isolated')!;

    expect(distance(hub, secondary)).toBeGreaterThan(60);
    expect(distance(hub, isolated)).toBeGreaterThan(74);
    expect(topology.metrics.get('secondary-1')).toEqual({ links: 1, relations: 2 });
    expect(topology.metrics.get('isolated')).toEqual({ links: 0, relations: 0 });
  });

  it('retains an isolated selected entity in linked scope', () => {
    const nodes = [
      { id: 'hub' },
      { id: 'a' },
      { id: 'b' },
      { id: 'Solution' },
    ];
    const edges = [
      { id: 'hub-a', source: 'hub', target: 'a' },
      { id: 'hub-b', source: 'hub', target: 'b' },
    ];

    const visible = getKnowledgeGraphVisibleGraph(nodes, edges, {
      focusNeighborhood: false,
      graphScope: 'linked',
      selectedNodeId: 'Solution',
    });

    expect(visible.nodeIds.has('Solution')).toBe(true);
    expect(visible.nodes.map((node) => node.id)).toContain('Solution');
    expect(visible.edges).toHaveLength(2);
  });

  it('retains selected relationship endpoints outside the core component', () => {
    const nodes = [
      { id: 'hub' },
      { id: 'a' },
      { id: 'b' },
      { id: 'side-1' },
      { id: 'side-2' },
    ];
    const edges: KnowledgeGraphLayoutEdge[] = [
      { id: 'hub-a', source: 'hub', target: 'a' },
      { id: 'hub-b', source: 'hub', target: 'b' },
      { id: 'side-link', source: 'side-1', target: 'side-2' },
    ];
    const primaryIds = new Set(['hub', 'a', 'b']);

    const visible = getKnowledgeGraphVisibleGraph(nodes, edges, {
      focusNeighborhood: false,
      getEdgeKey: (edge) => `${edge.source}::${edge.target}`,
      getPrimaryConnectedComponentIds: () => primaryIds,
      graphScope: 'core',
      selectedEdgeId: 'side-1::side-2',
    });

    expect(visible.nodeIds.has('side-1')).toBe(true);
    expect(visible.nodeIds.has('side-2')).toBe(true);
    expect(visible.edges.some((edge) => edge.id === 'side-link')).toBe(true);
  });
});
