export interface KnowledgeGraphLayoutNode {
  id: string;
}

export interface KnowledgeGraphLayoutEdge {
  id?: string;
  source: string;
  target: string;
  relationCount?: number;
}

export type KnowledgeGraphScopeMode = 'core' | 'linked' | 'all';

export interface KnowledgeGraphVisibleGraph<TNode extends KnowledgeGraphLayoutNode, TEdge extends KnowledgeGraphLayoutEdge> {
  nodes: TNode[];
  edges: TEdge[];
  nodeIds: Set<string>;
}

export interface KnowledgeGraphVisibleGraphOptions<TNode extends KnowledgeGraphLayoutNode, TEdge extends KnowledgeGraphLayoutEdge> {
  focusNeighborhood: boolean;
  focusNodeId?: string | null;
  getEdgeKey?: (edge: TEdge) => string;
  getPrimaryConnectedComponentIds?: (nodes: TNode[], edges: TEdge[]) => Set<string>;
  graphScope: KnowledgeGraphScopeMode;
  selectedComponentNodeIds?: Set<string> | null;
  selectedEdgeId?: string | null;
  selectedNodeId?: string | null;
  selectedRelationshipTypesCount?: number;
}

export interface GraphLayoutPoint {
  x: number;
  y: number;
  z: number;
}

export interface GraphTopology {
  metrics: Map<string, { links: number; relations: number }>;
  positions: Map<string, GraphLayoutPoint>;
}

const GRAPH_GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

export const getKnowledgeGraphVisibleGraph = <
  TNode extends KnowledgeGraphLayoutNode,
  TEdge extends KnowledgeGraphLayoutEdge
>(
  nodes: TNode[],
  edges: TEdge[],
  options: KnowledgeGraphVisibleGraphOptions<TNode, TEdge>
): KnowledgeGraphVisibleGraph<TNode, TEdge> => {
  let candidateNodes = nodes;
  let candidateEdges = edges;
  const nodeIds = new Set(nodes.map((node) => node.id));
  const {
    focusNeighborhood,
    focusNodeId,
    graphScope,
    selectedComponentNodeIds,
    selectedEdgeId,
    selectedNodeId,
    selectedRelationshipTypesCount = 0,
  } = options;
  const applyScope =
    !selectedComponentNodeIds &&
    !focusNeighborhood &&
    selectedRelationshipTypesCount === 0 &&
    candidateEdges.length > 0;

  if (selectedComponentNodeIds) {
    candidateNodes = candidateNodes.filter((node) => selectedComponentNodeIds.has(node.id));
    candidateEdges = candidateEdges.filter((edge) =>
      selectedComponentNodeIds.has(edge.source) && selectedComponentNodeIds.has(edge.target)
    );
  }

  if (applyScope && graphScope === 'core') {
    const primaryIds = options.getPrimaryConnectedComponentIds?.(candidateNodes, candidateEdges);
    if (primaryIds && primaryIds.size >= 3 && primaryIds.size < candidateNodes.length) {
      candidateNodes = candidateNodes.filter((node) => primaryIds.has(node.id));
    }
  } else if (applyScope && graphScope === 'linked') {
    const connectedIds = new Set<string>();
    candidateEdges.forEach((edge) => {
      connectedIds.add(edge.source);
      connectedIds.add(edge.target);
    });

    const connectedNodes = candidateNodes.filter((node) => connectedIds.has(node.id));
    if (connectedNodes.length >= 3) {
      candidateNodes = connectedNodes;
    }
  }

  if (focusNeighborhood && focusNodeId) {
    const focusIds = new Set<string>([focusNodeId]);
    candidateEdges.forEach((edge) => {
      if (edge.source === focusNodeId) focusIds.add(edge.target);
      if (edge.target === focusNodeId) focusIds.add(edge.source);
    });
    candidateNodes = candidateNodes.filter((node) => focusIds.has(node.id));
  }

  const retainedNodeIds = new Set<string>();
  const retainedEdges: TEdge[] = [];
  if (selectedNodeId && nodeIds.has(selectedNodeId)) {
    retainedNodeIds.add(selectedNodeId);
  }
  if (selectedEdgeId) {
    const getEdgeKey = options.getEdgeKey || ((edge: TEdge) => edge.id || `${edge.source}::${edge.target}`);
    edges
      .filter((edge) => edge.id === selectedEdgeId || getEdgeKey(edge) === selectedEdgeId)
      .forEach((edge) => {
        retainedEdges.push(edge);
        retainedNodeIds.add(edge.source);
        retainedNodeIds.add(edge.target);
      });
  }
  if (retainedNodeIds.size > 0) {
    const candidateNodeIds = new Set(candidateNodes.map((node) => node.id));
    candidateNodes = [
      ...candidateNodes,
      ...nodes.filter((node) => retainedNodeIds.has(node.id) && !candidateNodeIds.has(node.id)),
    ];
    const candidateEdgeIds = new Set(candidateEdges.map((edge) => edge.id || `${edge.source}::${edge.target}`));
    candidateEdges = [
      ...candidateEdges,
      ...retainedEdges.filter((edge) => !candidateEdgeIds.has(edge.id || `${edge.source}::${edge.target}`)),
    ];
  }

  const visibleIds = new Set(candidateNodes.map((node) => node.id));
  candidateEdges = candidateEdges.filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target));

  return { nodes: candidateNodes, edges: candidateEdges, nodeIds: visibleIds };
};

const relaxGraphTopologyPositions = (
  positions: Map<string, GraphLayoutPoint>,
  nodeCount: number
) => {
  if (positions.size < 3 || nodeCount > 420) return;

  const ids = Array.from(positions.keys());
  const minDistance = nodeCount <= 40 ? 36 : nodeCount <= 120 ? 28 : 17;
  const iterations = nodeCount <= 80 ? 30 : nodeCount <= 160 ? 22 : 14;

  for (let pass = 0; pass < iterations; pass += 1) {
    const adjustments = new Map<string, GraphLayoutPoint>();
    ids.forEach((id) => adjustments.set(id, { x: 0, y: 0, z: 0 }));
    let moved = false;

    for (let i = 0; i < ids.length; i += 1) {
      const aId = ids[i];
      const a = positions.get(aId);
      const aAdjustment = adjustments.get(aId);
      if (!a || !aAdjustment) continue;

      for (let j = i + 1; j < ids.length; j += 1) {
        const bId = ids[j];
        const b = positions.get(bId);
        const bAdjustment = adjustments.get(bId);
        if (!b || !bAdjustment) continue;

        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let dz = (a.z - b.z) * 0.72;
        let distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (distance < 0.001) {
          const angle = (i * 31 + j * 17) * GRAPH_GOLDEN_ANGLE;
          dx = Math.cos(angle);
          dy = Math.sin(angle);
          dz = Math.sin(angle * 1.7) * 0.48;
          distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
        }

        if (distance >= minDistance) continue;

        const strength = ((minDistance - distance) / minDistance) * (pass < 4 ? 1.05 : 0.68);
        const pushX = (dx / distance) * strength;
        const pushY = (dy / distance) * strength;
        const pushZ = (dz / distance) * strength;

        aAdjustment.x += pushX;
        aAdjustment.y += pushY;
        aAdjustment.z += pushZ;
        bAdjustment.x -= pushX;
        bAdjustment.y -= pushY;
        bAdjustment.z -= pushZ;
        moved = true;
      }
    }

    if (!moved) break;

    const damping = pass < 4 ? 0.92 : 0.72;
    ids.forEach((id) => {
      const point = positions.get(id);
      const adjustment = adjustments.get(id);
      if (!point || !adjustment) return;
      positions.set(id, {
        x: point.x + adjustment.x * damping,
        y: point.y + adjustment.y * damping,
        z: point.z + adjustment.z * damping,
      });
    });
  }
};

export const buildCompactGraphTopology = (
  nodes: KnowledgeGraphLayoutNode[],
  edges: KnowledgeGraphLayoutEdge[]
): GraphTopology => {
  const adjacency = new Map<string, Set<string>>();
  const metrics = new Map<string, { links: number; relations: number }>();
  const positions = new Map<string, GraphLayoutPoint>();

  nodes.forEach((node) => {
    adjacency.set(node.id, new Set());
    metrics.set(node.id, { links: 0, relations: 0 });
  });

  edges.forEach((edge) => {
    if (!adjacency.has(edge.source) || !adjacency.has(edge.target)) return;
    const relationCount = Math.max(1, Number(edge.relationCount || 1));
    adjacency.get(edge.source)?.add(edge.target);
    adjacency.get(edge.target)?.add(edge.source);

    const source = metrics.get(edge.source);
    const target = metrics.get(edge.target);
    if (source) {
      source.links += 1;
      source.relations += relationCount;
    }
    if (target) {
      target.links += 1;
      target.relations += relationCount;
    }
  });

  const visited = new Set<string>();
  const components: string[][] = [];

  nodes.forEach((node) => {
    if (visited.has(node.id)) return;
    const stack = [node.id];
    const component: string[] = [];
    visited.add(node.id);

    while (stack.length > 0) {
      const id = stack.pop()!;
      component.push(id);
      adjacency.get(id)?.forEach((neighborId) => {
        if (!visited.has(neighborId)) {
          visited.add(neighborId);
          stack.push(neighborId);
        }
      });
    }

    components.push(component);
  });

  const relationScore = (ids: string[]) =>
    ids.reduce((total, id) => total + (metrics.get(id)?.relations || 0), 0);
  const sortedComponents = components.sort((a, b) => b.length - a.length || relationScore(b) - relationScore(a));
  const connectedComponents = sortedComponents.filter((component) => component.length > 1);
  const isolatedComponents = sortedComponents.filter((component) => component.length === 1);
  const compactLayout = nodes.length <= 40;
  const spreadScale = nodes.length > 3000 ? 0.68 : nodes.length > 350 ? 0.88 : nodes.length > 120 ? 1.08 : compactLayout ? 1.58 : 1.34;

  connectedComponents.forEach((component, componentIndex) => {
    const orderedIds = [...component].sort((a, b) => {
      const aMetrics = metrics.get(a) || { links: 0, relations: 0 };
      const bMetrics = metrics.get(b) || { links: 0, relations: 0 };
      return bMetrics.relations - aMetrics.relations || bMetrics.links - aMetrics.links || a.localeCompare(b);
    });
    const componentAngle = componentIndex * GRAPH_GOLDEN_ANGLE;
    const componentSpread = compactLayout ? 0.98 : nodes.length <= 120 ? 1.12 : 1;
    const componentDistance = componentIndex === 0
      ? 0
      : (34 + Math.sqrt(componentIndex) * 20 + Math.sqrt(orderedIds.length) * 6.4) * spreadScale * componentSpread;
    const center = {
      x: Math.cos(componentAngle) * componentDistance,
      y: Math.sin(componentAngle) * componentDistance,
      z: Math.sin(componentAngle * 0.7) * componentDistance * 0.36,
    };
    const localRadius = Math.max(
      17,
      Math.min(
        nodes.length > 3000 ? 52 : nodes.length > 300 ? 70 : nodes.length > 120 ? 92 : compactLayout ? 168 : 148,
        (18 + Math.sqrt(orderedIds.length) * 13) * spreadScale
      )
    );

    orderedIds.forEach((id, nodeIndex) => {
      if (orderedIds.length === 1) {
        positions.set(id, center);
        return;
      }

      if (orderedIds.length === 2) {
        const angle = componentAngle + nodeIndex * Math.PI;
        positions.set(id, {
          x: center.x + Math.cos(angle) * localRadius * 0.64,
          y: center.y + Math.sin(angle) * localRadius * 0.64,
          z: center.z + (nodeIndex === 0 ? -1 : 1) * Math.min(18, localRadius * 0.28),
        });
        return;
      }

      if (nodeIndex === 0 && orderedIds.length > 2) {
        positions.set(id, center);
        return;
      }

      const localIndex = orderedIds.length > 2 ? nodeIndex - 1 : nodeIndex;
      const localTotal = Math.max(1, orderedIds.length - (orderedIds.length > 2 ? 2 : 1));
      const angle = localIndex * GRAPH_GOLDEN_ANGLE;
      const distance = orderedIds.length === 2
        ? localRadius * 0.64
        : localRadius * Math.sqrt((localIndex + 1) / localTotal);
      positions.set(id, {
        x: center.x + Math.cos(angle) * distance,
        y: center.y + Math.sin(angle) * distance,
        z: center.z + Math.sin(localIndex * GRAPH_GOLDEN_ANGLE * 1.7) * Math.min(42, localRadius * 0.42),
      });
    });
  });

  const isolateInnerRadius = connectedComponents.length > 0 ? (nodes.length > 3000 ? 120 : nodes.length > 300 ? 112 : compactLayout ? 96 : 124) : 32;
  const isolateGap = nodes.length > 3000 ? 32 : nodes.length > 300 ? 31 : compactLayout ? 28 : 32;
  const firstRingCapacity = connectedComponents.length > 0 ? 24 : 18;

  isolatedComponents.forEach((component, isolateIndex) => {
    let slot = isolateIndex;
    let ring = 0;
    let capacity = firstRingCapacity;

    while (slot >= capacity) {
      slot -= capacity;
      ring += 1;
      capacity += nodes.length > 300 ? 14 : 10;
    }

    const radius = isolateInnerRadius + ring * isolateGap;
    const angle = ((slot + 0.5) / capacity) * Math.PI * 2 + ring * 0.38;
    positions.set(component[0], {
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
      z: Math.sin(angle * 2.1 + ring * 0.7) * Math.min(34, 12 + ring * 4),
    });
  });

  if (positions.size > 3) {
    const points = Array.from(positions.values());
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
    const spanX = Math.max(1, bounds.maxX - bounds.minX);
    const spanY = Math.max(1, bounds.maxY - bounds.minY);
    const currentAspect = spanX / spanY;
    const targetAspect = nodes.length > 350
      ? 1.7
      : nodes.length > 120
        ? 1.62
        : compactLayout
          ? 1.34
          : 1.74;

    if (currentAspect < targetAspect * 0.92) {
      const factor = targetAspect / currentAspect;
      const horizontalStretch = Math.max(1, Math.min(nodes.length > 350 ? 1.56 : 1.66, Math.sqrt(factor) * 1.04));
      const verticalTighten = Math.max(nodes.length > 350 ? 0.75 : 0.7, Math.min(1, 1 / Math.pow(factor, 0.28)));
      const depthStretch = Math.max(0.98, Math.min(1.2, 1 + (horizontalStretch - 1) * 0.32));
      const center = {
        x: (bounds.minX + bounds.maxX) / 2,
        y: (bounds.minY + bounds.maxY) / 2,
        z: (bounds.minZ + bounds.maxZ) / 2,
      };

      positions.forEach((point, id) => {
        positions.set(id, {
          x: center.x + (point.x - center.x) * horizontalStretch,
          y: center.y + (point.y - center.y) * verticalTighten,
          z: center.z + (point.z - center.z) * depthStretch,
        });
      });
    }
  }

  relaxGraphTopologyPositions(positions, nodes.length);

  return { metrics, positions };
};
