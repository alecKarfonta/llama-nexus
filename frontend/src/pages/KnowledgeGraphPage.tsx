import { useState, useEffect, useCallback, useRef, useMemo, type ReactElement, type ReactNode } from 'react';
import {
  Box,
  Typography,
  Grid,
  TextField,
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  Select,
  MenuItem,
  Chip,
  List,
  ListItemButton,
  ListItemText,
  ButtonBase,
  Tooltip,
  Alert,
  CircularProgress,
  Divider,
  Tabs,
  Tab,
    Switch,
    FormControlLabel,
    Collapse,
    ToggleButton,
    ToggleButtonGroup,
  } from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
  Search as SearchIcon,
  Refresh as RefreshIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  Hub as HubIcon,
  Upload as UploadIcon,
  Fullscreen as FullscreenIcon,
  FullscreenExit as FullscreenExitIcon,
  CenterFocusStrong as CenterIcon,
    Explore as ExploreIcon,
    Settings as SettingsIcon,
    PlayArrow as PlayIcon,
    Pause as PauseIcon,
    Info as InfoIcon,
    Person as PersonIcon,
    Business as OrganizationIcon,
    Place as LocationIcon,
    Event as EventIcon,
    Category as ConceptIcon,
    Memory as TechnologyIcon,
    ModelTraining as ModelIcon,
    Api as ApiIcon,
    DataObject as CodeIcon,
    Storage as DataIcon,
    Functions as MetricIcon,
    Key as SecurityIcon,
    CloudQueue as ServiceIcon,
    DeveloperBoard as HardwareIcon,
    Article as PromptIcon,
    BubbleChart as VectorIcon,
    Speed as EvaluationIcon,
    RocketLaunch as RuntimeIcon,
    AccountTree as RelationIcon,
    Label as LabelIcon,
    KeyboardArrowDown as ExpandBundleIcon,
    KeyboardArrowUp as PreviousResultIcon,
      KeyboardArrowLeft as BrowseBackIcon,
      KeyboardArrowRight as CollapseBundleIcon,
    Close as CloseIcon,
    MoreHoriz as MoreIcon,
    } from '@mui/icons-material';
import {
  PageShell,
  SectionCard,
} from '../components/common';
import { debugLog } from '@/utils/debug';
import { useSearchParams } from 'react-router-dom';
import type { ForceGraphMethods as ForceGraph3DMethods } from 'react-force-graph-3d';
import * as THREE from 'three';
import { apiService as api } from '../services/api';
import { accentColors, gradients } from '../utils/theme';
import {
  getNodeIconSpriteWorldSize,
  getNodeLabelOffset,
  type KnowledgeGraphNodeLabelAnchor,
  type KnowledgeGraphNodeLabelStyle,
} from '../utils/knowledgeGraphLabelLayout';
import { buildCompactGraphTopology, getKnowledgeGraphVisibleGraph, type GraphTopology } from '../utils/knowledgeGraphLayout';
import { blendGraphSearchResults } from '../utils/knowledgeGraphSearch';
import {
  formatGraphEvidenceSnippetText,
  getGraphTextSearchSnippet,
  isGraphEvidenceExcerpt,
  normalizeGraphDisplayText,
  normalizeGraphSearchResultText,
  normalizeGraphText,
} from '../utils/knowledgeGraphText';
import {
  getKnowledgeGraphAmbientLabelLimit as getGraphAmbientLabelLimit,
  getKnowledgeGraphAnchorLabelLimit as getGraphAnchorLabelLimit,
  getKnowledgeGraphEntityVisualAlias,
  getKnowledgeGraphEntityVisualType as getEntityVisualType,
  getKnowledgeGraphPriorityLabelQualityFloor,
  getKnowledgeGraphRelationVisualKind as getRelationVisualKind,
  isKnowledgeGraphGenericEntityVisualType,
  normalizeKnowledgeGraphType as normalizeType,
  type KnowledgeGraphDetailMode as GraphDetailMode,
  type KnowledgeGraphLabelDensity as GraphLabelDensity,
  type KnowledgeGraphRelationVisualKind as RelationVisualKind,
  type KnowledgeGraphScaleTier as GraphScaleTier,
  type KnowledgeGraphZoomTier as GraphZoomTier,
} from '../utils/knowledgeGraphVisuals';

// Types
interface Entity {
  id: string;
  name: string;
  entity_type: string;
  description: string;
  aliases: string[];
  properties: Record<string, unknown>;
  confidence: number;
}

interface Relationship {
  id: string;
  source_id: string;
  target_id: string;
  relationship_type: string;
  description: string;
  weight: number;
}

interface GraphNode {
  id: string;
  label: string;
  type: string;
  description?: string;
  confidence?: number;
  occurrence?: number;
  properties?: Record<string, unknown>;
  val?: number; // Node size for react-force-graph
}

interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  type?: string;
  weight: number;
  relationCount?: number;
  searchText?: string;
  properties?: Record<string, unknown>;
}

type RenderGraphNode = GraphNode & { x?: number; y?: number; z?: number; val?: number };

type RenderGraphEdge = Omit<GraphEdge, 'source' | 'target'> & {
  source: string | RenderGraphNode;
  target: string | RenderGraphNode;
};

type ForceGraph3DComponent = typeof import('react-force-graph-3d').default;
type ForceGraphRuntimeHandle = ForceGraph3DMethods<RenderGraphNode, RenderGraphEdge> & {
  emitParticle?: (link: RenderGraphEdge) => void;
};
type GraphPointerEvent = MouseEvent & {
  detail?: number;
};

let forceGraph3DLoader: Promise<ForceGraph3DComponent> | null = null;

const loadForceGraph3D = () => {
  if (!forceGraph3DLoader) {
    forceGraph3DLoader = import('react-force-graph-3d').then((module) => module.default);
  }
  return forceGraph3DLoader;
};

interface RelationTypeCount {
  type: string;
  count: number;
}

interface RelationTypePayload {
  type?: string | null;
  count?: number | string | null;
}

interface GraphSearchResult {
  id: string;
  kind: 'entity' | 'relation';
  label: string;
  secondary: string;
  snippet?: string;
  color: string;
  badges?: { label: string; color?: string }[];
  entity?: Entity;
  edge?: GraphEdge;
  route?: {
    sourceLabel: string;
    sourceType?: string;
    targetLabel: string;
    targetType?: string;
    relationType: string;
    relationLabel: string;
    relationCount: number;
  };
}

interface GraphBrowseEntry {
  kind: 'entity' | 'edge';
  id: string;
  label: string;
  color: string;
}

interface GraphOverviewHover {
  kind: 'entity' | 'relation';
  id: string;
  label: string;
  secondary: string;
  color: string;
  entityType?: string;
  relationType?: string;
}

interface GraphSemanticTypePreview {
  kind: 'entity' | 'relation';
  type: string;
}

const isSameGraphBrowseEntry = (a?: GraphBrowseEntry | null, b?: GraphBrowseEntry | null) =>
  Boolean(a && b && a.kind === b.kind && a.id === b.id);

interface GraphPathSegment {
  fromId: string;
  toId: string;
  edgeId: string;
  visualEdgeId: string;
  type: string;
  label: string;
  color: string;
  score: number;
}

interface GraphPathPreview {
  sourceId: string;
  targetId: string;
  nodes: Entity[];
  segments: GraphPathSegment[];
}

interface LinkLabelSpriteOptions {
  rich?: boolean;
  sourceLabel?: string;
  targetLabel?: string;
  relationTypes?: RelationTypeCount[];
}

interface GraphComponentSummary {
  id: string;
  label: string;
  nodeIds: Set<string>;
  nodeCount: number;
  linkCount: number;
  relationCount: number;
  topNodeId?: string;
  topNodeLabel?: string;
  topNodeType?: string;
  entityTypes: { type: string; count: number }[];
}

interface RelationshipEvidence {
  context: string;
  domain?: string;
  type: string;
  weight: number;
}

interface HoveredNodeStrongestRelation {
  color: string;
  count: number;
  otherId: string;
  otherNode?: GraphNode;
  type: string;
  weight: number;
}

interface HoveredNodeRelationPreview {
  relationTypes: Array<RelationTypeCount & { color: string }>;
  strongest: HoveredNodeStrongestRelation | null;
}

interface RawEvidenceSnippet {
  context?: unknown;
  domain?: unknown;
  type?: string | null;
  weight?: number | string | null;
}

interface GraphRAGNodePayload {
  id: string;
  label: string;
  type?: string;
  occurrence?: number;
  properties?: Record<string, unknown>;
}

interface GraphRAGEdgePayload {
  id: string;
  source: string;
  target: string;
  label?: string;
  type?: string;
  weight?: number;
  relationCount?: number;
  relation_count?: number;
  properties?: Record<string, unknown>;
}

  type MetadataEntry = [string, unknown];
  type MetadataGroupId = 'source' | 'evidence' | 'metrics' | 'graph' | 'model' | 'other';
  type MetadataEntryGroup = {
    id: MetadataGroupId;
    label: string;
    color: string;
    priority: number;
    entries: MetadataEntry[];
  };
  type GraphMetadataFact = {
    id: string;
    label: string;
    value: string;
    color: string;
    icon: ReactElement;
    priority: number;
  };
  type GraphScope = 'core' | 'linked' | 'all';
  type GraphLabelModeValue = 'hidden' | GraphLabelDensity;
  type GraphSearchKindFilter = 'all' | 'entity' | 'relation';
  type GraphSettingsSection = 'display' | 'scope' | 'entities' | 'components' | 'relationships';
  type GraphDepthBand = 'front' | 'middle' | 'back';
  type GraphNodeLabelStyle = KnowledgeGraphNodeLabelStyle;
  type GraphNodeLabelAnchor = KnowledgeGraphNodeLabelAnchor;
  type GraphSelectionPlacementEdge = 'left' | 'right' | 'top' | 'bottom';

  interface GraphNodeLabelLayout {
    anchor: GraphNodeLabelAnchor;
    pinned: boolean;
    screenX: number;
    screenY: number;
    selectionContext?: boolean;
  }

  interface GraphPersistedLabel {
    layout: GraphNodeLabelLayout;
    style: GraphNodeLabelStyle;
    worldScale?: number;
    visibleUntil: number;
    visibleSince: number;
    baseOpacity: number;
    fadeOutMs?: number;
    pinned: boolean;
    retention?: 'ambient' | 'focus';
  }

  interface GraphFocusHaloLabelHint {
    color: string;
    count: number;
    type: string;
  }

  const GRAPH_MANUAL_NAVIGATION_SETTLE_MS = 1450;
  const GRAPH_LABEL_FADE_SETTLE_MS = 5600;
  const GRAPH_LABEL_FADE_IN_MS = 1400;
  const GRAPH_LABEL_FADE_OUT_MS = 9800;
  const GRAPH_ROUTE_LABEL_HOLD_MS = 12000;
  const GRAPH_ROUTE_LABEL_FADE_OUT_MS = 7200;
  const GRAPH_LABEL_STABLE_AMBIENT_SCORE = 24000;
  const GRAPH_LABEL_STABLE_PINNED_SCORE = 60000;
  const GRAPH_RECENT_AUTOFIT_SKIP_MS = 3600;
  const GRAPH_OVERVIEW_DIRECT_FIT_LIMIT = 350;
  const GRAPH_HOVER_INTENT_MS = 280;
  const GRAPH_HOVER_CLEAR_INTENT_MS = 320;
  const GRAPH_HOVER_CARD_ENTER_MS = 260;
  const GRAPH_HOVER_CARD_EXIT_MS = 3000;
  const GRAPH_HOVER_CARD_CONTEXT_CLEAR_MS = 180;
  const GRAPH_HOVER_FOCUS_FADE_IN_MS = 1320;
  const GRAPH_HOVER_FOCUS_FADE_OUT_MS = 2600;
  const GRAPH_ROUTE_FOCUS_FADE_IN_MS = 1120;
  const GRAPH_ROUTE_FOCUS_FADE_OUT_MS = 2400;
  const GRAPH_SELECTION_REFIT_AFTER_MANUAL_MS = 1250;

  const getGraphFocusHaloLabelBudget = (tier: GraphScaleTier, zoomTier: GraphZoomTier) => {
    const limits: Record<GraphScaleTier, Record<GraphZoomTier, number>> = {
      massive: { far: 0, middle: 0, near: 1 },
      large: { far: 0, middle: 1, near: 2 },
      standard: { far: 0, middle: 2, near: 3 },
      compact: { far: 1, middle: 3, near: 4 },
    };

    return limits[tier][zoomTier];
  };

  const getGraphNavigationRetainLimit = (tier: GraphScaleTier, viewportWidth: number) => {
    const compactViewport = viewportWidth < 700;
    const limits: Record<GraphScaleTier, number> = compactViewport
      ? { massive: 18, large: 24, standard: 34, compact: 42 }
      : { massive: 36, large: 48, standard: 64, compact: 72 };

    return limits[tier];
  };

  const getGraphFocusContextBudget = (
    tier: GraphScaleTier,
    zoomTier: GraphZoomTier,
    activeNodeCount: number,
    searchActive: boolean,
    edgeFocused: boolean
  ) => {
    const baseLimits: Record<GraphScaleTier, Record<GraphZoomTier, number>> = {
      massive: { far: 2, middle: 6, near: 10 },
      large: { far: 5, middle: 12, near: 18 },
      standard: { far: 8, middle: 16, near: 24 },
      compact: { far: 10, middle: 20, near: 30 },
    };
    const activeFocusPenalty = activeNodeCount > 90
      ? 0.38
      : activeNodeCount > 42
        ? 0.55
        : activeNodeCount > 16
          ? 0.72
          : 1;
      const intentPenalty = searchActive ? 0.9 : edgeFocused ? 0.72 : 1;

    return Math.round(baseLimits[tier][zoomTier] * activeFocusPenalty * intentPenalty);
  };

  const getGraphFocusedNameplateLimit = (tier: GraphScaleTier, viewportWidth: number) => {
    const compactViewport = viewportWidth < 700;
    if (tier === 'massive') return compactViewport ? 6 : 10;
    if (tier === 'large') return compactViewport ? 8 : 16;
    if (tier === 'standard') return compactViewport ? 10 : 22;
    return compactViewport ? 12 : 28;
  };

  const getGraphActiveIdentityLimit = (
    tier: GraphScaleTier,
    zoomTier: GraphZoomTier,
    viewportWidth: number,
    activeNodeCount: number
  ) => {
    const compactViewport = viewportWidth < 700;
    const baseLimits: Record<GraphScaleTier, Record<GraphZoomTier, number>> = compactViewport
      ? {
        massive: { far: 5, middle: 7, near: 10 },
        large: { far: 7, middle: 10, near: 14 },
        standard: { far: 9, middle: 14, near: 18 },
        compact: { far: 11, middle: 16, near: 22 },
      }
      : {
        massive: { far: 8, middle: 12, near: 16 },
        large: { far: 12, middle: 18, near: 26 },
        standard: { far: 16, middle: 26, near: 38 },
        compact: { far: 20, middle: 34, near: 48 },
      };

    return Math.min(activeNodeCount, baseLimits[tier][zoomTier]);
  };

  const getGraphLabelFadeValue = (value: number) => {
    const clamped = Math.max(0, Math.min(1, value));
    return clamped * clamped * (3 - 2 * clamped);
  };

  const getGraphPersistedLabelOpacity = (label: GraphPersistedLabel, now: number) => {
    const fadeInFloor = label.pinned ? 0.72 : label.retention === 'focus' ? 0.54 : 0.38;
    const fadeIn = Math.max(fadeInFloor, getGraphLabelFadeValue((now - label.visibleSince) / GRAPH_LABEL_FADE_IN_MS));
    if (label.pinned) return fadeIn;

    const remaining = Math.max(0, label.visibleUntil - now);
    const fadeOutMs = Math.max(1, label.fadeOutMs || GRAPH_LABEL_FADE_OUT_MS);
    const fadeOut = remaining < fadeOutMs
      ? getGraphLabelFadeValue(remaining / fadeOutMs)
      : 1;
    return Math.max(0, Math.min(label.baseOpacity, fadeIn, fadeOut));
  };

  interface GraphScreenRect {
    left: number;
    top: number;
    right: number;
    bottom: number;
  }

  interface GraphSelectionAnchor {
    key: string;
    kind: 'node' | 'edge';
    x: number;
    y: number;
    depth: number;
    width: number;
    height: number;
  }

  interface GraphOverlayPlacement {
    mode: 'anchored' | 'docked';
    edge?: GraphSelectionPlacementEdge;
    left?: number;
    top?: number;
    right?: number;
    bottom?: number;
    transformOrigin: string;
    tether?: {
      angle: number;
      length: number;
      left: number;
      top: number;
    };
  }

  interface GraphCameraLod {
    key: string;
    zoomTier: GraphZoomTier;
    distance: number;
    position: { x: number; y: number; z: number };
    target: { x: number; y: number; z: number };
    direction: { x: number; y: number; z: number };
    moving: boolean;
  }

  interface GraphCameraFocusRequest {
    id: number;
    target: { x: number; y: number; z: number };
    minDistance?: number;
    maxDistance?: number;
    duration?: number;
  }

  const INITIAL_GRAPH_CAMERA_LOD: GraphCameraLod = {
    key: 'initial',
    zoomTier: 'far',
    distance: 260,
    position: { x: 0, y: 0, z: 260 },
    target: { x: 0, y: 0, z: 0 },
    direction: { x: 0, y: 0, z: -1 },
    moving: false,
  };

  interface GraphRenderProfile {
    tier: GraphScaleTier;
    nodeRelSize: number;
    nodeResolution: number;
    linkResolution: number;
    arrowResolution: number;
    warmupTicks: number;
    cooldownTicks: number;
    maxPixelRatio: number;
    nodeBudget: Record<GraphZoomTier, number>;
    labelBudget: Record<GraphZoomTier, number>;
    iconBudget: Record<GraphZoomTier, number>;
    linkBudget: Record<GraphZoomTier, number>;
    alwaysUseCustomNodes: boolean;
  }

  const GRAPH_ENTITY_FETCH_LIMIT = 3000;
  const GRAPH_RELATIONSHIP_FETCH_LIMIT = 9000;

  const TYPE_COLORS: Record<string, { main: string; light: string; glow: string }> = {
  person: { main: accentColors.success, light: accentColors.success, glow: alpha(accentColors.success, 0.6) },
  organization: { main: accentColors.info, light: accentColors.info, glow: alpha(accentColors.info, 0.6) },
  location: { main: accentColors.warning, light: accentColors.warning, glow: alpha(accentColors.warning, 0.6) },
  date: { main: accentColors.purple, light: accentColors.purple, glow: alpha(accentColors.purple, 0.6) },
  event: { main: accentColors.rose, light: accentColors.rose, glow: alpha(accentColors.rose, 0.6) },
  product: { main: accentColors.info, light: accentColors.info, glow: alpha(accentColors.info, 0.6) },
  technology: { main: accentColors.primary, light: accentColors.primary, glow: alpha(accentColors.primary, 0.6) },
  model: { main: accentColors.purple, light: accentColors.purple, glow: alpha(accentColors.purple, 0.6) },
  code: { main: accentColors.info, light: accentColors.info, glow: alpha(accentColors.info, 0.6) },
  data: { main: accentColors.warning, light: accentColors.warning, glow: alpha(accentColors.warning, 0.6) },
  prompt: { main: '#eab308', light: '#fde047', glow: alpha('#eab308', 0.6) },
  vector: { main: '#14b8a6', light: '#2dd4bf', glow: alpha('#14b8a6', 0.6) },
  api: { main: '#38bdf8', light: '#7dd3fc', glow: alpha('#38bdf8', 0.6) },
  config: { main: '#a3e635', light: '#bef264', glow: alpha('#a3e635', 0.55) },
  memory: { main: '#2dd4bf', light: '#5eead4', glow: alpha('#2dd4bf', 0.58) },
  security: { main: '#fb7185', light: '#fda4af', glow: alpha('#fb7185', 0.58) },
  evaluation: { main: '#f97316', light: '#fb923c', glow: alpha('#f97316', 0.6) },
  runtime: { main: '#22c55e', light: '#4ade80', glow: alpha('#22c55e', 0.6) },
  service: { main: accentColors.success, light: accentColors.success, glow: alpha(accentColors.success, 0.6) },
  hardware: { main: accentColors.slate, light: accentColors.slate, glow: alpha(accentColors.slate, 0.6) },
  metric: { main: accentColors.rose, light: accentColors.rose, glow: alpha(accentColors.rose, 0.6) },
  concept: { main: accentColors.rose, light: accentColors.rose, glow: alpha(accentColors.rose, 0.6) },
  process: { main: accentColors.success, light: accentColors.success, glow: alpha(accentColors.success, 0.6) },
    custom: { main: accentColors.slate, light: accentColors.slate, glow: alpha(accentColors.slate, 0.6) },
  };

  const RELATION_PALETTE = [
    accentColors.primary,
    accentColors.success,
    accentColors.info,
    accentColors.warning,
    accentColors.rose,
    accentColors.purple,
    accentColors.slate,
  ];
  const RELATION_KIND_COLORS: Record<RelationVisualKind, string> = {
    connects: accentColors.slate,
    contains: accentColors.purple,
    controls: accentColors.warning,
    conflicts: accentColors.rose,
    creates: accentColors.success,
    evaluates: '#f97316',
    flows: accentColors.info,
    guards: '#fb7185',
    influences: '#f59e0b',
    references: '#38bdf8',
    sequence: '#818cf8',
    supports: '#22c55e',
    transforms: '#a855f7',
  };
  const RELATION_CURVE_PROFILES: Record<RelationVisualKind, { curvature: number; rotation: number }> = {
    connects: { curvature: 0.042, rotation: 0 },
    contains: { curvature: -0.054, rotation: Math.PI * 0.12 },
    controls: { curvature: 0.074, rotation: Math.PI * 0.18 },
    conflicts: { curvature: -0.092, rotation: Math.PI * 0.48 },
    creates: { curvature: 0.066, rotation: Math.PI * 0.08 },
    evaluates: { curvature: -0.068, rotation: Math.PI * 0.32 },
    flows: { curvature: 0.086, rotation: Math.PI * 0.24 },
    guards: { curvature: -0.076, rotation: Math.PI * 0.38 },
    influences: { curvature: 0.096, rotation: Math.PI * 0.3 },
    references: { curvature: -0.048, rotation: Math.PI * 0.16 },
    sequence: { curvature: 0.062, rotation: Math.PI * 0.42 },
    supports: { curvature: -0.058, rotation: Math.PI * 0.22 },
    transforms: { curvature: 0.082, rotation: Math.PI * 0.36 },
  };

  const ENTITY_METADATA_HIDDEN_KEYS = new Set(['aliases', 'confidence', 'description', 'entity_type', 'id', 'name', 'occurrence', 'type']);
  const ENTITY_SEARCH_METADATA_HIDDEN_KEYS = new Set([
    ...ENTITY_METADATA_HIDDEN_KEYS,
    'extractedby',
    'extracted_by',
    'extractor',
    'modelconfidence',
    'model_confidence',
    'nermodel',
    'ner_model',
  ]);
  const EDGE_METADATA_HIDDEN_KEYS = new Set(['context', 'domain', 'domains', 'evidencesnippets', 'id', 'relationcount', 'relationtypes', 'source', 'target', 'type', 'weight']);
  const METADATA_KEY_PRIORITY = new Map([
    ['domain', 0],
    ['domains', 0],
    ['source', 1],
    ['sources', 1],
    ['document', 2],
    ['documents', 2],
    ['file', 2],
    ['path', 2],
    ['url', 2],
    ['category', 3],
    ['tags', 3],
    ['score', 4],
    ['rank', 4],
    ['confidence', 4],
    ['createdat', 5],
    ['updatedat', 5],
    ['timestamp', 5],
  ]);

  const getTypeLabel = (value?: string | null) =>
    normalizeType(value)
      .replace(/[_-]+/g, ' ')
      .replace(/\b\w/g, (char) => char.toUpperCase());

  const getTypeSearchAliases = (value?: string | null) => {
    const normalized = normalizeType(value);
    const original = typeof value === 'string' ? value.trim() : '';
    return Array.from(new Set([
      original,
      normalized,
      normalized.replace(/[_-]+/g, ' '),
      getTypeLabel(normalized),
    ].filter(Boolean)));
  };

  const getEntityDisplayTypeLabel = (type?: string | null, label?: string | null) => {
    const aliased = getKnowledgeGraphEntityVisualAlias(type);
    const visualType = getEntityVisualType(type, label);
    if (isKnowledgeGraphGenericEntityVisualType(aliased) && visualType !== aliased) {
      return getTypeLabel(visualType);
    }
    return getTypeLabel(type || visualType);
  };

  const getTypeColors = (type?: string, label?: string | null) =>
    TYPE_COLORS[getEntityVisualType(type, label)] || TYPE_COLORS.custom;

  const getRelationshipColor = (type?: string) => {
    const normalized = normalizeType(type);
    const relationText = normalized.replace(/[_-]+/g, ' ');
    const visualKind = getRelationVisualKind(relationText);
    if (visualKind !== 'connects') return RELATION_KIND_COLORS[visualKind];
    if (/\b(related|connect|associate|reference|mention|link|map|same|similar)\b/.test(relationText)) {
      return RELATION_KIND_COLORS.connects;
    }
    const hash = normalized.split('').reduce((sum, char) => sum + char.charCodeAt(0), 0);
    return RELATION_PALETTE[hash % RELATION_PALETTE.length];
  };

  const getVisualEdgeKey = (edge: Pick<GraphEdge, 'source' | 'target' | 'type' | 'label'>) =>
    [edge.source, edge.target].join('::');

  const getStableGraphHash = (value: string) => {
    let hash = 0;
    for (let index = 0; index < value.length; index += 1) {
      hash = ((hash << 5) - hash + value.charCodeAt(index)) | 0;
    }
    return Math.abs(hash);
  };

  const getRenderEdgeEndpointId = (endpoint: string | RenderGraphNode | undefined) =>
    typeof endpoint === 'string' ? endpoint : endpoint?.id;

  const getRelationMarkerPlacement = (
    link: RenderGraphEdge,
    focusNodeId?: string | null
  ) => {
    const hash = getStableGraphHash(`${link.id}:${link.type || link.label}`);
    const jitter = ((hash % 9) - 4) * 0.012;
    const sourceId = getRenderEdgeEndpointId(link.source);
    const targetId = getRenderEdgeEndpointId(link.target);

    if (focusNodeId && sourceId === focusNodeId) {
      return Math.max(0.74, Math.min(0.9, 0.82 + jitter));
    }
    if (focusNodeId && targetId === focusNodeId) {
      return Math.max(0.1, Math.min(0.26, 0.18 - jitter));
    }

    return 0.42 + (hash % 7) * 0.035;
  };

  const getRelationMarkerLane = (
    link: RenderGraphEdge,
    focusNodeId?: string | null
  ) => {
    const hash = getStableGraphHash(`${focusNodeId || 'ambient'}:${link.id}:${link.type || link.label}`);
    if (focusNodeId) return (hash % 7) - 3;
    return (hash % 5) - 2;
  };

  const getPreviousEntityBrowseEntry = (
    history: GraphBrowseEntry[],
    historyIndex: number,
    selectedEntityId: string
  ) => {
    const startIndex = historyIndex >= 0
      ? Math.min(historyIndex, history.length - 1)
      : history.length - 1;

    for (let index = startIndex; index >= 0; index -= 1) {
      const entry = history[index];
      if (entry?.kind === 'entity' && entry.id !== selectedEntityId) return entry;
    }

    return null;
  };

  const buildEntityPathPreview = (
    sourceId: string,
    targetId: string,
    entityById: Map<string, Entity>,
    edges: GraphEdge[]
  ): GraphPathPreview | null => {
    if (sourceId === targetId || !entityById.has(sourceId) || !entityById.has(targetId)) return null;

    const adjacency = new Map<string, GraphPathSegment[]>();
    const addSegment = (fromId: string, toId: string, edge: GraphEdge) => {
      if (!entityById.has(fromId) || !entityById.has(toId)) return;
      const type = normalizeType(edge.type || edge.label);
      const relationCount = Math.max(1, Number(edge.relationCount || 1));
      const score = relationCount * 12 + Number(edge.weight || 0);
      const nextSegment: GraphPathSegment = {
        fromId,
        toId,
        edgeId: edge.id,
        visualEdgeId: getVisualEdgeKey(edge),
        type,
        label: getTypeLabel(type),
        color: getRelationshipColor(type),
        score,
      };
      const segments = adjacency.get(fromId) || [];
      const existingIndex = segments.findIndex((segment) => segment.toId === toId);

      if (existingIndex >= 0) {
        if (segments[existingIndex].score < score) {
          segments[existingIndex] = nextSegment;
        }
      } else {
        segments.push(nextSegment);
      }

      adjacency.set(fromId, segments);
    };

    edges.forEach((edge) => {
      addSegment(edge.source, edge.target, edge);
      addSegment(edge.target, edge.source, edge);
    });
    adjacency.forEach((segments) => segments.sort((a, b) => b.score - a.score || a.label.localeCompare(b.label)));

    const entityCount = entityById.size;
    const maxDepth = entityCount > 1200 ? 4 : entityCount > 350 ? 5 : 6;
    const maxVisited = entityCount > 3000 ? 900 : entityCount > 1200 ? 1200 : 1800;
    const queue: Array<{ id: string; nodeIds: string[]; segments: GraphPathSegment[] }> = [{
      id: sourceId,
      nodeIds: [sourceId],
      segments: [],
    }];
    const visited = new Set<string>([sourceId]);

    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const item = queue[cursor];
      if (item.segments.length >= maxDepth) continue;

      const nextSegments = adjacency.get(item.id) || [];
      for (const segment of nextSegments) {
        if (visited.has(segment.toId)) continue;
        const nodeIds = [...item.nodeIds, segment.toId];
        const segments = [...item.segments, segment];

        if (segment.toId === targetId) {
          const pathNodes = nodeIds
            .map((id) => entityById.get(id))
            .filter((entity): entity is Entity => Boolean(entity));

          if (pathNodes.length !== nodeIds.length) return null;
          return {
            sourceId,
            targetId,
            nodes: pathNodes,
            segments,
          };
        }

        visited.add(segment.toId);
        if (visited.size >= maxVisited) return null;
        queue.push({ id: segment.toId, nodeIds, segments });
      }
    }

    return null;
  };

    const getEntityIcon = (type?: string, color = 'inherit', label?: string | null) => {
    const sx = { fontSize: 16, color, flexShrink: 0 };
    switch (getEntityVisualType(type, label)) {
      case 'person': return <PersonIcon sx={sx} />;
      case 'organization': return <OrganizationIcon sx={sx} />;
      case 'location': return <LocationIcon sx={sx} />;
      case 'date': return <EventIcon sx={sx} />;
      case 'event': return <EventIcon sx={sx} />;
      case 'product': return <LabelIcon sx={sx} />;
      case 'technology': return <TechnologyIcon sx={sx} />;
      case 'model': return <ModelIcon sx={sx} />;
      case 'api': return <ApiIcon sx={sx} />;
      case 'config': return <SettingsIcon sx={sx} />;
      case 'memory': return <TechnologyIcon sx={sx} />;
      case 'security': return <SecurityIcon sx={sx} />;
      case 'code': return <CodeIcon sx={sx} />;
      case 'data': return <DataIcon sx={sx} />;
      case 'prompt': return <PromptIcon sx={sx} />;
      case 'vector': return <VectorIcon sx={sx} />;
      case 'evaluation': return <EvaluationIcon sx={sx} />;
      case 'runtime': return <RuntimeIcon sx={sx} />;
      case 'service': return <ServiceIcon sx={sx} />;
      case 'hardware': return <HardwareIcon sx={sx} />;
      case 'metric': return <MetricIcon sx={sx} />;
      case 'process': return <RelationIcon sx={sx} />;
      case 'concept': return <ConceptIcon sx={sx} />;
      default: return <HubIcon sx={sx} />;
    }
  };

  const getEntitySymbolShapeSx = (visualType: string) => {
    switch (visualType) {
      case 'organization':
      case 'code':
      case 'config':
        return { borderRadius: '5px', clipPath: 'none', rotate: '-8deg' };
      case 'location':
        return { borderRadius: '5px', clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)', rotate: '0deg' };
      case 'event':
      case 'date':
        return { borderRadius: '4px', clipPath: 'polygon(50% 0%, 100% 88%, 0% 88%)', rotate: '0deg' };
      case 'model':
        return { borderRadius: '5px', clipPath: 'polygon(50% 0%, 62% 32%, 97% 35%, 70% 56%, 79% 92%, 50% 72%, 21% 92%, 30% 56%, 3% 35%, 38% 32%)', rotate: '0deg' };
      case 'product':
        return { borderRadius: '4px', clipPath: 'polygon(10% 0%, 78% 0%, 100% 24%, 100% 100%, 10% 100%)', rotate: '-5deg' };
      case 'concept':
        return { borderRadius: '5px', clipPath: 'polygon(50% 0%, 88% 22%, 88% 78%, 50% 100%, 12% 78%, 12% 22%)', rotate: '0deg' };
      case 'api':
        return { borderRadius: '999px', clipPath: 'polygon(18% 24%, 42% 24%, 50% 40%, 58% 24%, 82% 24%, 82% 76%, 58% 76%, 50% 60%, 42% 76%, 18% 76%)', rotate: '0deg' };
      case 'data':
      case 'memory':
        return { borderRadius: '999px / 62%', clipPath: 'ellipse(48% 42% at 50% 50%)', rotate: '0deg' };
      case 'prompt':
        return { borderRadius: '4px', clipPath: 'polygon(14% 0%, 70% 0%, 86% 20%, 86% 100%, 14% 100%)', rotate: '0deg' };
      case 'security':
        return { borderRadius: '5px', clipPath: 'polygon(50% 0%, 86% 18%, 76% 72%, 50% 100%, 24% 72%, 14% 18%)', rotate: '0deg' };
      case 'vector':
        return { borderRadius: '5px', clipPath: 'polygon(50% 0%, 92% 26%, 92% 74%, 50% 100%, 8% 74%, 8% 26%)', rotate: '0deg' };
      case 'evaluation':
      case 'metric':
        return { borderRadius: '5px', clipPath: 'polygon(50% 0%, 98% 38%, 80% 100%, 20% 100%, 2% 38%)', rotate: '0deg' };
      case 'runtime':
        return { borderRadius: '5px', clipPath: 'polygon(50% 0%, 83% 68%, 58% 60%, 50% 100%, 42% 60%, 17% 68%)', rotate: '0deg' };
      case 'technology':
      case 'hardware':
      case 'process':
        return { borderRadius: '5px', clipPath: 'polygon(50% 0%, 94% 25%, 94% 75%, 50% 100%, 6% 75%, 6% 25%)', rotate: '0deg' };
      default:
        return { borderRadius: '999px', clipPath: 'none', rotate: '0deg' };
    }
  };

  const renderEntitySymbolMark = (
    type?: string,
    label?: string | null,
    options: { selected?: boolean; muted?: boolean; size?: number } = {}
  ) => {
    const visualType = getEntityVisualType(type, label);
    const colors = getTypeColors(type, label);
    const size = options.size || 30;
    const inset = Math.max(3, Math.round(size * 0.15));
    const shape = getEntitySymbolShapeSx(visualType);
    const muted = Boolean(options.muted);
    const selected = Boolean(options.selected);

    return (
      <Box
        component="span"
        data-kg-entity-symbol={visualType}
        aria-hidden="true"
        sx={{
          width: size,
          height: size,
          position: 'relative',
          display: 'grid',
          placeItems: 'center',
          flexShrink: 0,
          color: muted ? alpha(colors.light, 0.58) : colors.light,
          filter: selected ? `drop-shadow(0 0 12px ${alpha(colors.main, 0.42)})` : 'none',
        }}
      >
        <Box
          component="span"
          sx={{
            position: 'absolute',
            inset,
            borderRadius: shape.borderRadius,
            clipPath: shape.clipPath,
            transform: `rotateX(56deg) rotateZ(${shape.rotate})`,
            transformStyle: 'preserve-3d',
            background: muted
              ? `linear-gradient(135deg, ${alpha(colors.light, 0.36)} 0%, ${alpha(colors.main, 0.2)} 54%, ${alpha('#020617', 0.42)} 100%)`
              : `linear-gradient(135deg, ${alpha(colors.light, 0.96)} 0%, ${colors.main} 48%, ${alpha('#020617', 0.36)} 100%)`,
            border: '1px solid',
            borderColor: alpha(selected ? '#fff' : colors.light, selected ? 0.72 : muted ? 0.22 : 0.44),
            boxShadow: muted
              ? `inset -3px -4px 9px ${alpha('#020617', 0.42)}`
              : `inset -4px -5px 10px ${alpha('#020617', 0.36)}, inset 3px 3px 8px ${alpha('#fff', 0.18)}, 0 8px 18px ${alpha(colors.main, selected ? 0.34 : 0.18)}`,
            opacity: muted ? 0.62 : 1,
          }}
        />
        <Box
          component="span"
          sx={{
            position: 'relative',
            zIndex: 1,
            width: Math.max(16, Math.round(size * 0.54)),
            height: Math.max(16, Math.round(size * 0.54)),
            borderRadius: '7px',
            display: 'grid',
            placeItems: 'center',
            bgcolor: alpha('#020617', muted ? 0.18 : 0.28),
            border: '1px solid',
            borderColor: alpha('#fff', muted ? 0.12 : 0.2),
            boxShadow: `0 5px 12px ${alpha('#020617', 0.28)}`,
            '& .MuiSvgIcon-root': {
              fontSize: Math.max(12, Math.round(size * 0.48)),
            },
          }}
        >
          {getEntityIcon(type, muted ? alpha(colors.light, 0.6) : colors.light, label)}
        </Box>
      </Box>
    );
  };

  const getRelationIcon = (type?: string | null, color = 'inherit', size = 15) => {
    const sx = { fontSize: size, color, flexShrink: 0 };
    switch (getRelationVisualKind(type)) {
      case 'contains': return <HubIcon sx={sx} />;
      case 'controls': return <SettingsIcon sx={sx} />;
      case 'conflicts': return <CloseIcon sx={sx} />;
      case 'creates': return <LabelIcon sx={sx} />;
      case 'evaluates': return <InfoIcon sx={sx} />;
      case 'flows': return <CollapseBundleIcon sx={sx} />;
      case 'guards': return <SecurityIcon sx={sx} />;
      case 'influences': return <MetricIcon sx={sx} />;
      case 'references': return <CodeIcon sx={sx} />;
      case 'sequence': return <BrowseBackIcon sx={{ ...sx, transform: 'rotate(180deg)' }} />;
      case 'supports': return <ServiceIcon sx={sx} />;
      case 'transforms': return <RuntimeIcon sx={sx} />;
      default: return <RelationIcon sx={sx} />;
    }
  };

  const renderRelationSymbolMark = (
    type?: string | null,
    color = accentColors.primary,
    options: { muted?: boolean; selected?: boolean; size?: number } = {}
  ) => {
    const size = options.size || 30;
    const muted = Boolean(options.muted);
    const selected = Boolean(options.selected);
    const relationColor = muted ? alpha(color, 0.58) : color;

    return (
      <Box
        component="span"
        data-kg-relation-symbol={getRelationVisualKind(type)}
        aria-hidden="true"
        sx={{
          width: size,
          height: size,
          position: 'relative',
          display: 'grid',
          placeItems: 'center',
          flexShrink: 0,
          color: relationColor,
          filter: selected ? `drop-shadow(0 0 10px ${alpha(color, 0.36)})` : 'none',
          opacity: muted ? 0.62 : 1,
        }}
      >
        <Box
          component="span"
          sx={{
            position: 'absolute',
            left: Math.round(size * 0.12),
            right: Math.round(size * 0.12),
            top: '50%',
            height: Math.max(2, Math.round(size * 0.08)),
            borderRadius: '999px',
            transform: 'translateY(-50%) rotate(-18deg)',
            transformOrigin: 'center',
            background: `linear-gradient(90deg, ${alpha(color, muted ? 0.18 : 0.34)} 0%, ${relationColor} 62%, ${alpha('#fff', muted ? 0.2 : 0.5)} 100%)`,
            boxShadow: `0 5px 14px ${alpha(color, muted ? 0.12 : 0.24)}`,
          }}
        />
        <Box
          component="span"
          sx={{
            position: 'absolute',
            right: Math.round(size * 0.09),
            top: Math.round(size * 0.32),
            width: 0,
            height: 0,
            borderTop: `${Math.max(4, Math.round(size * 0.14))}px solid transparent`,
            borderBottom: `${Math.max(4, Math.round(size * 0.14))}px solid transparent`,
            borderLeft: `${Math.max(6, Math.round(size * 0.18))}px solid ${relationColor}`,
            transform: 'rotate(-18deg)',
            filter: `drop-shadow(0 3px 6px ${alpha(color, muted ? 0.12 : 0.28)})`,
          }}
        />
        <Box
          component="span"
          sx={{
            position: 'relative',
            zIndex: 1,
            width: Math.max(16, Math.round(size * 0.54)),
            height: Math.max(16, Math.round(size * 0.54)),
            borderRadius: '999px',
            display: 'grid',
            placeItems: 'center',
            bgcolor: alpha('#020617', muted ? 0.22 : 0.36),
            border: '1px solid',
            borderColor: alpha(color, muted ? 0.18 : selected ? 0.58 : 0.34),
            boxShadow: `inset 0 1px 0 ${alpha('#fff', 0.16)}, 0 5px 12px ${alpha('#020617', 0.32)}`,
          }}
        >
          {getRelationIcon(type, relationColor, Math.max(12, Math.round(size * 0.46)))}
        </Box>
      </Box>
    );
  };

  const drawEntityShape = (
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    size: number,
    type?: string,
    label?: string | null
  ) => {
    const normalized = getEntityVisualType(type, label);
    ctx.beginPath();
    if (normalized === 'organization') {
      ctx.rect(x - size, y - size, size * 2, size * 2);
      return;
    }
    if (normalized === 'location') {
      ctx.moveTo(x, y - size * 1.25);
      ctx.lineTo(x + size, y);
      ctx.lineTo(x, y + size * 1.25);
      ctx.lineTo(x - size, y);
      ctx.closePath();
      return;
    }
    if (normalized === 'event' || normalized === 'date') {
      ctx.moveTo(x, y - size * 1.15);
      ctx.lineTo(x + size * 1.05, y + size * 0.75);
      ctx.lineTo(x - size * 1.05, y + size * 0.75);
      ctx.closePath();
      return;
    }
    if (normalized === 'model') {
      for (let i = 0; i < 8; i += 1) {
        const radius = i % 2 === 0 ? size * 1.08 : size * 0.82;
        const angle = Math.PI / 8 + (i * Math.PI) / 4;
        const px = x + Math.cos(angle) * radius;
        const py = y + Math.sin(angle) * radius;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      return;
    }
    if (normalized === 'product') {
      ctx.moveTo(x - size * 0.94, y - size * 0.9);
      ctx.lineTo(x + size * 0.44, y - size * 0.9);
      ctx.lineTo(x + size * 0.94, y - size * 0.4);
      ctx.lineTo(x + size * 0.94, y + size * 0.9);
      ctx.lineTo(x - size * 0.94, y + size * 0.9);
      ctx.closePath();
      return;
    }
    if (normalized === 'concept') {
      for (let i = 0; i < 6; i += 1) {
        const angle = Math.PI / 6 + (i * Math.PI) / 3;
        const px = x + Math.cos(angle) * size;
        const py = y + Math.sin(angle) * size;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      return;
    }
    if (normalized === 'code') {
      ctx.rect(x - size * 1.08, y - size * 0.82, size * 2.16, size * 1.64);
      return;
    }
    if (normalized === 'data') {
      ctx.ellipse(x, y - size * 0.55, size * 0.98, size * 0.42, 0, 0, Math.PI * 2);
      ctx.moveTo(x + size * 0.98, y - size * 0.55);
      ctx.lineTo(x + size * 0.98, y + size * 0.5);
      ctx.ellipse(x, y + size * 0.5, size * 0.98, size * 0.42, 0, 0, Math.PI * 2);
      ctx.lineTo(x - size * 0.98, y - size * 0.55);
      return;
    }
    if (normalized === 'prompt') {
      ctx.moveTo(x - size * 0.84, y - size);
      ctx.lineTo(x + size * 0.36, y - size);
      ctx.lineTo(x + size * 0.84, y - size * 0.52);
      ctx.lineTo(x + size * 0.84, y + size);
      ctx.lineTo(x - size * 0.84, y + size);
      ctx.closePath();
      return;
    }
    if (normalized === 'security') {
      ctx.moveTo(x, y - size * 1.12);
      ctx.lineTo(x + size * 0.84, y - size * 0.62);
      ctx.lineTo(x + size * 0.68, y + size * 0.42);
      ctx.lineTo(x, y + size * 1.08);
      ctx.lineTo(x - size * 0.68, y + size * 0.42);
      ctx.lineTo(x - size * 0.84, y - size * 0.62);
      ctx.closePath();
      return;
    }
    if (normalized === 'config') {
      ctx.rect(x - size * 1.02, y - size * 0.72, size * 2.04, size * 1.44);
      return;
    }
    if (normalized === 'memory') {
      ctx.ellipse(x, y - size * 0.5, size * 0.92, size * 0.36, 0, 0, Math.PI * 2);
      ctx.moveTo(x + size * 0.92, y - size * 0.5);
      ctx.lineTo(x + size * 0.92, y + size * 0.52);
      ctx.ellipse(x, y + size * 0.52, size * 0.92, size * 0.36, 0, 0, Math.PI * 2);
      ctx.lineTo(x - size * 0.92, y - size * 0.5);
      return;
    }
    if (normalized === 'api') {
      ctx.arc(x - size * 0.42, y, size * 0.44, 0, Math.PI * 2);
      ctx.moveTo(x + size * 0.86, y);
      ctx.arc(x + size * 0.42, y, size * 0.44, 0, Math.PI * 2);
      return;
    }
    if (normalized === 'vector') {
      for (let i = 0; i < 8; i += 1) {
        const angle = Math.PI / 8 + (i * Math.PI) / 4;
        const px = x + Math.cos(angle) * size;
        const py = y + Math.sin(angle) * size;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      return;
    }
    if (normalized === 'evaluation') {
      ctx.moveTo(x, y - size * 1.08);
      ctx.lineTo(x + size * 0.98, y - size * 0.18);
      ctx.lineTo(x + size * 0.58, y + size * 0.98);
      ctx.lineTo(x - size * 0.58, y + size * 0.98);
      ctx.lineTo(x - size * 0.98, y - size * 0.18);
      ctx.closePath();
      return;
    }
    if (normalized === 'runtime') {
      ctx.moveTo(x, y - size * 1.22);
      ctx.lineTo(x + size * 0.76, y + size * 0.42);
      ctx.lineTo(x + size * 0.25, y + size * 0.32);
      ctx.lineTo(x, y + size * 1.05);
      ctx.lineTo(x - size * 0.25, y + size * 0.32);
      ctx.lineTo(x - size * 0.76, y + size * 0.42);
      ctx.closePath();
      return;
    }
    if (normalized === 'service') {
      ctx.arc(x - size * 0.42, y + size * 0.08, size * 0.52, Math.PI * 0.82, Math.PI * 1.92);
      ctx.arc(x + size * 0.05, y - size * 0.2, size * 0.58, Math.PI * 1.1, Math.PI * 2.08);
      ctx.arc(x + size * 0.56, y + size * 0.12, size * 0.46, Math.PI * 1.22, Math.PI * 0.3, true);
      ctx.lineTo(x - size * 0.74, y + size * 0.58);
      ctx.closePath();
      return;
    }
    if (normalized === 'metric') {
      ctx.moveTo(x, y - size * 1.06);
      for (let i = 1; i < 5; i += 1) {
        const angle = -Math.PI / 2 + (i * Math.PI * 2) / 5;
        ctx.lineTo(x + Math.cos(angle) * size, y + Math.sin(angle) * size);
      }
      ctx.closePath();
      return;
    }
    if (normalized === 'technology' || normalized === 'hardware' || normalized === 'process') {
      for (let i = 0; i < 6; i += 1) {
        const angle = Math.PI / 6 + (i * Math.PI) / 3;
        const px = x + Math.cos(angle) * size;
        const py = y + Math.sin(angle) * size;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      return;
    }
      ctx.arc(x, y, size, 0, 2 * Math.PI);
    };

    const drawEntityGlyph = (
      ctx: CanvasRenderingContext2D,
      x: number,
      y: number,
      size: number,
      type?: string,
      label?: string | null
    ) => {
      const normalized = getEntityVisualType(type, label);
      const stroke = 'rgba(255, 255, 255, 0.92)';
      const fill = 'rgba(255, 255, 255, 0.94)';
      const lineWidth = Math.max(1.2, size * 0.12);
      const dot = Math.max(1.6, size * 0.15);
      const r = size * 0.46;

      ctx.save();
      ctx.shadowColor = 'rgba(3, 7, 18, 0.7)';
      ctx.shadowBlur = Math.max(1, size * 0.08);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.strokeStyle = stroke;
      ctx.fillStyle = fill;
      ctx.lineWidth = lineWidth;

      if (normalized === 'model') {
        const layers = [
          [[x - size * 0.5, y - size * 0.3], [x - size * 0.5, y + size * 0.3]],
          [[x, y - size * 0.42], [x, y], [x, y + size * 0.42]],
          [[x + size * 0.5, y - size * 0.26], [x + size * 0.5, y + size * 0.26]],
        ];
        ctx.beginPath();
        layers[0].forEach(([fromX, fromY]) => {
          layers[1].forEach(([toX, toY]) => {
            ctx.moveTo(fromX, fromY);
            ctx.lineTo(toX, toY);
          });
        });
        layers[1].forEach(([fromX, fromY]) => {
          layers[2].forEach(([toX, toY]) => {
            ctx.moveTo(fromX, fromY);
            ctx.lineTo(toX, toY);
          });
        });
        ctx.stroke();
        layers.flat().forEach(([px, py]) => {
          ctx.beginPath();
          ctx.arc(px, py, dot * 0.94, 0, Math.PI * 2);
          ctx.fill();
        });
      ctx.restore();
      return;
    }

      if (normalized === 'product') {
        const w = size * 0.82;
        const h = size * 0.72;
        ctx.strokeRect(x - w / 2, y - h / 2, w, h);
        ctx.beginPath();
        ctx.moveTo(x - w * 0.26, y - h * 0.14);
        ctx.lineTo(x + w * 0.26, y - h * 0.14);
        ctx.moveTo(x - w * 0.26, y + h * 0.14);
        ctx.lineTo(x + w * 0.16, y + h * 0.14);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x + w * 0.32, y - h * 0.34, dot * 0.72, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        return;
      }

      if (normalized === 'concept') {
        ctx.beginPath();
        ctx.arc(x, y - size * 0.1, size * 0.38, Math.PI * 1.1, Math.PI * 1.9);
        ctx.arc(x, y - size * 0.1, size * 0.38, Math.PI * 1.9, Math.PI * 2.9);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x - size * 0.24, y + size * 0.3);
        ctx.lineTo(x + size * 0.24, y + size * 0.3);
        ctx.moveTo(x - size * 0.18, y + size * 0.52);
        ctx.lineTo(x + size * 0.18, y + size * 0.52);
        ctx.stroke();
        ctx.restore();
        return;
      }

      if (normalized === 'code') {
        ctx.beginPath();
        ctx.moveTo(x - size * 0.18, y - size * 0.48);
        ctx.lineTo(x - size * 0.44, y);
        ctx.lineTo(x - size * 0.18, y + size * 0.48);
        ctx.moveTo(x + size * 0.18, y - size * 0.48);
        ctx.lineTo(x + size * 0.44, y);
        ctx.lineTo(x + size * 0.18, y + size * 0.48);
        ctx.moveTo(x + size * 0.06, y - size * 0.54);
        ctx.lineTo(x - size * 0.06, y + size * 0.54);
        ctx.stroke();
        ctx.restore();
        return;
      }

      if (normalized === 'data') {
        const w = size * 0.86;
        const h = size * 0.38;
        ctx.ellipse(x, y - size * 0.42, w / 2, h / 2, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x - w / 2, y - size * 0.42);
        ctx.lineTo(x - w / 2, y + size * 0.38);
        ctx.ellipse(x, y + size * 0.38, w / 2, h / 2, 0, 0, Math.PI);
        ctx.moveTo(x + w / 2, y - size * 0.42);
        ctx.lineTo(x + w / 2, y + size * 0.38);
        ctx.stroke();
        ctx.restore();
        return;
      }

      if (normalized === 'prompt') {
        const w = size * 0.78;
        const h = size * 0.92;
        ctx.strokeRect(x - w / 2, y - h / 2, w, h);
        ctx.beginPath();
        ctx.moveTo(x + w * 0.12, y - h / 2);
        ctx.lineTo(x + w / 2, y - h * 0.12);
        ctx.moveTo(x - w * 0.24, y - h * 0.16);
        ctx.lineTo(x + w * 0.24, y - h * 0.16);
        ctx.moveTo(x - w * 0.24, y + h * 0.08);
        ctx.lineTo(x + w * 0.18, y + h * 0.08);
        ctx.moveTo(x - w * 0.24, y + h * 0.3);
        ctx.lineTo(x + w * 0.12, y + h * 0.3);
        ctx.stroke();
        ctx.restore();
        return;
      }

      if (normalized === 'security') {
        const w = size * 0.72;
        const h = size * 0.5;
        ctx.beginPath();
        ctx.arc(x, y - size * 0.16, w * 0.38, Math.PI, 0);
        ctx.stroke();
        ctx.strokeRect(x - w / 2, y - size * 0.1, w, h);
        ctx.beginPath();
        ctx.arc(x, y + size * 0.12, dot, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        return;
      }

      if (normalized === 'config') {
        [-0.34, 0, 0.34].forEach((offset, index) => {
          const yy = y + offset * size;
          ctx.beginPath();
          ctx.moveTo(x - size * 0.48, yy);
          ctx.lineTo(x + size * 0.48, yy);
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(x + (index === 1 ? size * 0.22 : -size * 0.18), yy, dot, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.restore();
        return;
      }

      if (normalized === 'memory') {
        const w = size * 0.78;
        [-0.28, 0.06, 0.4].forEach((offset) => {
          ctx.beginPath();
          ctx.moveTo(x - w / 2, y + offset * size);
          ctx.lineTo(x + w / 2, y + offset * size);
          ctx.stroke();
        });
        ctx.beginPath();
        ctx.rect(x - w / 2, y - size * 0.44, w, size * 0.96);
        ctx.stroke();
        ctx.restore();
        return;
      }

      if (normalized === 'api') {
        const nodes = [
          [x - size * 0.44, y - size * 0.28],
          [x - size * 0.44, y + size * 0.28],
          [x + size * 0.42, y],
        ];
        ctx.beginPath();
        ctx.moveTo(nodes[0][0], nodes[0][1]);
        ctx.lineTo(nodes[2][0], nodes[2][1]);
        ctx.lineTo(nodes[1][0], nodes[1][1]);
        ctx.stroke();
        nodes.forEach(([px, py]) => {
          ctx.beginPath();
          ctx.arc(px, py, dot, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.restore();
        return;
      }

      if (normalized === 'vector') {
        const nodes = [
          [x - size * 0.45, y - size * 0.22],
          [x - size * 0.12, y + size * 0.38],
          [x + size * 0.36, y + size * 0.18],
          [x + size * 0.24, y - size * 0.44],
        ];
        ctx.beginPath();
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]].forEach(([from, to]) => {
          ctx.moveTo(nodes[from][0], nodes[from][1]);
          ctx.lineTo(nodes[to][0], nodes[to][1]);
        });
        ctx.stroke();
        nodes.forEach(([px, py]) => {
          ctx.beginPath();
          ctx.arc(px, py, dot, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.restore();
        return;
      }

      if (normalized === 'evaluation') {
        ctx.beginPath();
        ctx.arc(x, y + size * 0.18, size * 0.48, Math.PI * 1.08, Math.PI * 1.92);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y + size * 0.18);
        ctx.lineTo(x + size * 0.32, y - size * 0.18);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y + size * 0.18, dot, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        return;
      }

      if (normalized === 'runtime') {
        ctx.beginPath();
        ctx.moveTo(x, y - size * 0.58);
        ctx.quadraticCurveTo(x + size * 0.44, y - size * 0.12, x + size * 0.22, y + size * 0.34);
        ctx.lineTo(x, y + size * 0.18);
        ctx.lineTo(x - size * 0.22, y + size * 0.34);
        ctx.quadraticCurveTo(x - size * 0.44, y - size * 0.12, x, y - size * 0.58);
        ctx.closePath();
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y - size * 0.1, dot * 0.92, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(x - size * 0.16, y + size * 0.42);
        ctx.lineTo(x, y + size * 0.62);
        ctx.lineTo(x + size * 0.16, y + size * 0.42);
        ctx.stroke();
        ctx.restore();
        return;
      }

      if (normalized === 'service') {
        const hubY = y - size * 0.22;
        const endpoints = [
          [x - size * 0.34, y + size * 0.38],
          [x, y + size * 0.5],
          [x + size * 0.34, y + size * 0.38],
        ];
        ctx.beginPath();
        endpoints.forEach(([px, py]) => {
          ctx.moveTo(x, hubY);
          ctx.lineTo(px, py);
        });
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, hubY, dot * 1.05, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x, y + size * 0.08, size * 0.46, Math.PI * 1.08, Math.PI * 1.92);
        ctx.stroke();
        endpoints.forEach(([px, py]) => {
          ctx.beginPath();
          ctx.arc(px, py, dot * 0.8, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.restore();
        return;
      }

      if (normalized === 'metric') {
        [-0.34, 0, 0.34].forEach((offset, index) => {
          const barHeight = size * (0.42 + index * 0.18);
          ctx.fillRect(x + offset * size - dot / 2, y + size * 0.42 - barHeight, dot, barHeight);
        });
        ctx.beginPath();
        ctx.moveTo(x - size * 0.48, y + size * 0.48);
        ctx.lineTo(x + size * 0.54, y + size * 0.48);
        ctx.stroke();
        ctx.restore();
        return;
      }

      if (normalized === 'hardware') {
        const box = size * 0.68;
        ctx.strokeRect(x - box / 2, y - box / 2, box, box);
        [-0.5, -0.18, 0.18, 0.5].forEach((offset) => {
          ctx.beginPath();
          ctx.moveTo(x - box / 2 - size * 0.18, y + offset * box);
          ctx.lineTo(x - box / 2, y + offset * box);
          ctx.moveTo(x + box / 2, y + offset * box);
          ctx.lineTo(x + box / 2 + size * 0.18, y + offset * box);
          ctx.moveTo(x + offset * box, y - box / 2 - size * 0.18);
          ctx.lineTo(x + offset * box, y - box / 2);
          ctx.moveTo(x + offset * box, y + box / 2);
          ctx.lineTo(x + offset * box, y + box / 2 + size * 0.18);
          ctx.stroke();
        });
        ctx.fillRect(x - box * 0.2, y - box * 0.2, box * 0.4, box * 0.4);
        ctx.restore();
        return;
      }

      if (normalized === 'technology') {
        const box = size * 0.64;
        ctx.strokeRect(x - box / 2, y - box / 2, box, box);
        ctx.beginPath();
        ctx.moveTo(x - box * 0.26, y);
        ctx.lineTo(x, y - box * 0.24);
        ctx.lineTo(x + box * 0.28, y + box * 0.14);
        ctx.stroke();
        [[x - box * 0.26, y], [x, y - box * 0.24], [x + box * 0.28, y + box * 0.14]].forEach(([px, py]) => {
          ctx.beginPath();
          ctx.arc(px, py, dot * 0.78, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.fillRect(x - box * 0.14, y - box * 0.14, box * 0.28, box * 0.28);
        ctx.restore();
        return;
      }

      if (normalized === 'custom') {
        const points = [
          [x - r * 0.78, y - r * 0.22],
          [x + r * 0.68, y - r * 0.45],
          [x + r * 0.2, y + r * 0.72],
        ];
        ctx.beginPath();
        ctx.moveTo(points[0][0], points[0][1]);
        ctx.lineTo(points[1][0], points[1][1]);
        ctx.lineTo(points[2][0], points[2][1]);
        ctx.lineTo(points[0][0], points[0][1]);
        ctx.stroke();
        points.forEach(([px, py]) => {
          ctx.beginPath();
          ctx.arc(px, py, dot, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.restore();
        return;
      }

      if (normalized === 'person') {
        ctx.beginPath();
        ctx.arc(x, y - size * 0.24, size * 0.18, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x, y + size * 0.34, size * 0.36, Math.PI * 1.08, Math.PI * 1.92);
        ctx.stroke();
        ctx.restore();
        return;
      }

      if (normalized === 'date' || normalized === 'event') {
        const w = size * 0.78;
        const h = size * 0.62;
        ctx.strokeRect(x - w / 2, y - h / 2, w, h);
        ctx.beginPath();
        ctx.moveTo(x - w / 2, y - h * 0.16);
        ctx.lineTo(x + w / 2, y - h * 0.16);
        ctx.stroke();
        ctx.fillRect(x - w * 0.22, y + h * 0.08, dot, dot);
        ctx.fillRect(x + w * 0.08, y + h * 0.08, dot, dot);
        ctx.restore();
        return;
      }

      if (normalized === 'process') {
        const topY = y - size * 0.42;
        const bottomY = y + size * 0.42;
        ctx.beginPath();
        ctx.moveTo(x, topY);
        ctx.lineTo(x, y);
        ctx.lineTo(x - size * 0.36, bottomY);
        ctx.moveTo(x, y);
        ctx.lineTo(x + size * 0.36, bottomY);
        ctx.stroke();
        [[x, topY], [x - size * 0.36, bottomY], [x + size * 0.36, bottomY]].forEach(([px, py]) => {
          ctx.beginPath();
          ctx.arc(px, py, dot, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.restore();
        return;
      }

      if (normalized === 'organization') {
        const w = size * 0.78;
        const h = size * 0.68;
        ctx.strokeRect(x - w / 2, y - h / 2, w, h);
        [-0.22, 0.22].forEach((offset) => {
          ctx.beginPath();
          ctx.moveTo(x + offset * w, y - h / 2);
          ctx.lineTo(x + offset * w, y + h / 2);
          ctx.stroke();
        });
        ctx.restore();
        return;
      }

      ctx.beginPath();
      ctx.arc(x, y, size * 0.18, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, y, size * 0.44, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    };

    const drawRelationGlyph = (
      ctx: CanvasRenderingContext2D,
      x: number,
      y: number,
      size: number,
      type?: string | null
    ) => {
      const kind = getRelationVisualKind(type);
      const dot = Math.max(1.4, size * 0.2);
      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.lineWidth = Math.max(1.4, size * 0.18);

      if (kind === 'flows') {
        ctx.beginPath();
        ctx.moveTo(x - size * 0.62, y);
        ctx.lineTo(x + size * 0.42, y);
        ctx.moveTo(x + size * 0.2, y - size * 0.3);
        ctx.lineTo(x + size * 0.52, y);
        ctx.lineTo(x + size * 0.2, y + size * 0.3);
        ctx.stroke();
        ctx.restore();
        return;
      }

      if (kind === 'influences') {
        ctx.beginPath();
        ctx.moveTo(x - size * 0.58, y + size * 0.32);
        ctx.quadraticCurveTo(x - size * 0.06, y - size * 0.5, x + size * 0.52, y - size * 0.12);
        ctx.moveTo(x + size * 0.22, y - size * 0.34);
        ctx.lineTo(x + size * 0.58, y - size * 0.1);
        ctx.lineTo(x + size * 0.18, y + size * 0.06);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x - size * 0.38, y + size * 0.24, dot * 0.92, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x + size * 0.42, y - size * 0.14, dot * 0.72, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        return;
      }

      if (kind === 'references') {
        const w = size * 0.56;
        const h = size * 0.46;
        ctx.strokeRect(x - size * 0.5, y - h * 0.5, w, h);
        ctx.strokeRect(x - size * 0.06, y - h * 0.5, w, h);
        ctx.beginPath();
        ctx.moveTo(x - size * 0.02, y);
        ctx.lineTo(x + size * 0.06, y);
        ctx.stroke();
        ctx.restore();
        return;
      }

      if (kind === 'conflicts') {
        ctx.beginPath();
        ctx.moveTo(x - size * 0.46, y - size * 0.46);
        ctx.lineTo(x + size * 0.46, y + size * 0.46);
        ctx.moveTo(x + size * 0.46, y - size * 0.46);
        ctx.lineTo(x - size * 0.46, y + size * 0.46);
        ctx.stroke();
        ctx.restore();
        return;
      }

      if (kind === 'sequence') {
        ctx.beginPath();
        ctx.moveTo(x - size * 0.58, y);
        ctx.lineTo(x + size * 0.46, y);
        ctx.moveTo(x + size * 0.18, y - size * 0.3);
        ctx.lineTo(x + size * 0.52, y);
        ctx.lineTo(x + size * 0.18, y + size * 0.3);
        ctx.moveTo(x - size * 0.18, y - size * 0.26);
        ctx.lineTo(x + size * 0.02, y);
        ctx.lineTo(x - size * 0.18, y + size * 0.26);
        ctx.stroke();
        ctx.restore();
        return;
      }

      if (kind === 'contains') {
        ctx.strokeRect(x - size * 0.46, y - size * 0.36, size * 0.92, size * 0.72);
        ctx.beginPath();
        ctx.arc(x, y, dot, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        return;
      }

      if (kind === 'controls') {
        [-0.34, 0, 0.34].forEach((offset, index) => {
          const yy = y + offset * size;
          ctx.beginPath();
          ctx.moveTo(x - size * 0.5, yy);
          ctx.lineTo(x + size * 0.5, yy);
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(x + (index === 1 ? size * 0.18 : -size * 0.18), yy, dot, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.restore();
        return;
      }

      if (kind === 'creates') {
        ctx.beginPath();
        ctx.moveTo(x - size * 0.5, y);
        ctx.lineTo(x + size * 0.5, y);
        ctx.moveTo(x, y - size * 0.5);
        ctx.lineTo(x, y + size * 0.5);
        ctx.stroke();
        ctx.restore();
        return;
      }

      if (kind === 'evaluates') {
        ctx.beginPath();
        ctx.moveTo(x - size * 0.52, y + size * 0.02);
        ctx.lineTo(x - size * 0.12, y + size * 0.42);
        ctx.lineTo(x + size * 0.56, y - size * 0.42);
        ctx.stroke();
        ctx.restore();
        return;
      }

      if (kind === 'guards') {
        ctx.beginPath();
        ctx.moveTo(x, y - size * 0.58);
        ctx.lineTo(x + size * 0.42, y - size * 0.32);
        ctx.lineTo(x + size * 0.34, y + size * 0.24);
        ctx.quadraticCurveTo(x, y + size * 0.58, x - size * 0.34, y + size * 0.24);
        ctx.lineTo(x - size * 0.42, y - size * 0.32);
        ctx.closePath();
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x - size * 0.18, y);
        ctx.lineTo(x - size * 0.03, y + size * 0.17);
        ctx.lineTo(x + size * 0.25, y - size * 0.16);
        ctx.stroke();
        ctx.restore();
        return;
      }

      if (kind === 'supports') {
        ctx.beginPath();
        ctx.moveTo(x - size * 0.48, y + size * 0.28);
        ctx.lineTo(x, y - size * 0.42);
        ctx.lineTo(x + size * 0.48, y + size * 0.28);
        ctx.moveTo(x - size * 0.26, y + size * 0.28);
        ctx.lineTo(x + size * 0.26, y + size * 0.28);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y + size * 0.08, dot, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        return;
      }

      if (kind === 'transforms') {
        ctx.beginPath();
        ctx.moveTo(x - size * 0.54, y - size * 0.2);
        ctx.lineTo(x + size * 0.12, y - size * 0.2);
        ctx.moveTo(x - size * 0.1, y - size * 0.42);
        ctx.lineTo(x + size * 0.24, y - size * 0.2);
        ctx.lineTo(x - size * 0.1, y + size * 0.02);
        ctx.moveTo(x + size * 0.54, y + size * 0.2);
        ctx.lineTo(x - size * 0.12, y + size * 0.2);
        ctx.moveTo(x + size * 0.1, y + size * 0.42);
        ctx.lineTo(x - size * 0.24, y + size * 0.2);
        ctx.lineTo(x + size * 0.1, y - size * 0.02);
        ctx.stroke();
        ctx.restore();
        return;
      }

      const points = [
        [x - size * 0.5, y],
        [x + size * 0.5, y],
        [x, y - size * 0.42],
      ];
      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);
      ctx.lineTo(points[1][0], points[1][1]);
      ctx.lineTo(points[2][0], points[2][1]);
      ctx.lineTo(points[0][0], points[0][1]);
      ctx.stroke();
      points.forEach(([px, py]) => {
        ctx.beginPath();
        ctx.arc(px, py, dot, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.restore();
    };

    const formatMetadataValue = (value: unknown): string => {
      if (value === null || value === undefined || value === '') return 'Not set';
      if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(3);
      if (typeof value === 'boolean') return value ? 'true' : 'false';
      if (Array.isArray(value)) return value.map(formatMetadataValue).join(', ');
      if (typeof value === 'object') {
        const entries = Object.entries(value as Record<string, unknown>)
          .filter(([, item]) => !isBlankMetadataValue(item))
          .slice(0, 4)
          .map(([key, item]) => `${getTypeLabel(key)}: ${formatMetadataValue(item)}`);
        return entries.length > 0 ? entries.join(' / ') : 'Not set';
      }
      return String(value);
    };

      const isBlankMetadataValue = (value: unknown) =>
        value === null ||
        value === undefined ||
        value === '' ||
        (Array.isArray(value) && value.length === 0) ||
        (typeof value === 'object' && !Array.isArray(value) && Object.keys(value as Record<string, unknown>).length === 0);

      const isGraphExtractionBoilerplate = (value: unknown) => {
        const text = normalizeGraphSearchResultText(value).toLowerCase();
        if (!text) return false;
        return /^extracted by\s+[\w .-]+\s+model(?:\s*\(confidence:\s*\d+(?:\.\d+)?\))?\.?$/.test(text);
      };

      const getUsefulEntityDescription = (value: unknown) => {
        const text = normalizeGraphSearchResultText(value);
        return text && !isGraphExtractionBoilerplate(text) ? text : '';
      };

      const getSearchableEntityMetadataText = (properties?: Record<string, unknown>) =>
        Object.entries(properties || {})
          .filter(([key, value]) =>
            !ENTITY_SEARCH_METADATA_HIDDEN_KEYS.has(normalizeType(key)) &&
            !isBlankMetadataValue(value) &&
            !isGraphExtractionBoilerplate(formatMetadataValue(value))
          )
          .map(([key, value]) => `${getTypeLabel(key)} ${formatMetadataValue(value)}`)
          .join(' ');

      const getFirstMetadataValue = (
        properties: Record<string, unknown> | undefined,
        keys: string[]
      ) => {
        const entries = Object.entries(properties || {});
        for (const key of keys) {
          const normalizedKey = normalizeType(key);
          const match = entries.find(([entryKey, value]) =>
            normalizeType(entryKey) === normalizedKey && !isBlankMetadataValue(value)
          );
          if (match) return match[1];
        }
        return null;
      };

      const getFiniteMetadataNumber = (...values: unknown[]) => {
        for (const value of values) {
          const numberValue = Number(value);
          if (Number.isFinite(numberValue) && numberValue > 0) return numberValue;
        }
        return null;
      };

      const formatMetadataFactValue = (value: unknown, maxLength = 28) =>
        truncateGraphLabel(formatMetadataValue(value), maxLength);

      const getMetadataGroupConfig = (key: string, fallbackColor: string): Omit<MetadataEntryGroup, 'entries'> => {
        const normalized = normalizeType(key);
        if ([
          'source',
          'sources',
          'document',
          'documents',
          'file',
          'path',
          'url',
          'domain',
          'domains',
          'namespace',
          'collection',
        ].includes(normalized)) {
          return { id: 'source', label: 'Source', color: accentColors.info, priority: 0 };
        }
        if ([
          'evidence',
          'evidence_snippets',
          'context',
          'snippet',
          'summary',
          'text',
          'description',
        ].includes(normalized)) {
          return { id: 'evidence', label: 'Evidence', color: accentColors.success, priority: 1 };
        }
        if ([
          'confidence',
          'score',
          'weight',
          'rank',
          'occurrence',
          'hits',
          'count',
          'degree',
          'relation_count',
          'relations',
        ].includes(normalized)) {
          return { id: 'metrics', label: 'Metrics', color: accentColors.warning, priority: 2 };
        }
        if ([
          'component',
          'community',
          'entity_type',
          'type',
          'label',
          'name',
          'aliases',
          'neighbors',
        ].includes(normalized)) {
          return { id: 'graph', label: 'Graph', color: fallbackColor, priority: 3 };
        }
        if ([
          'model',
          'adapter',
          'runtime',
          'backend',
          'provider',
          'device',
          'gpu',
          'embedding_model',
        ].includes(normalized)) {
          return { id: 'model', label: 'Model', color: accentColors.purple, priority: 4 };
        }
        return { id: 'other', label: 'Other', color: accentColors.slate, priority: 5 };
      };

      const getMetadataGroupIcon = (groupId: MetadataGroupId, color: string) => {
        const sx = { fontSize: 13, color: 'inherit' };
        switch (groupId) {
          case 'source':
            return <DataIcon sx={sx} />;
          case 'evidence':
            return <InfoIcon sx={sx} />;
          case 'metrics':
            return <MetricIcon sx={sx} />;
          case 'graph':
            return <HubIcon sx={sx} />;
          case 'model':
            return <ModelIcon sx={sx} />;
          default:
            return <LabelIcon sx={{ ...sx, color }} />;
        }
      };

      const groupMetadataEntries = (
        entries: MetadataEntry[],
        fallbackColor: string,
        limit = 8
      ): MetadataEntryGroup[] => {
        const groups = new Map<MetadataGroupId, MetadataEntryGroup>();
        entries.slice(0, limit).forEach((entry) => {
          const config = getMetadataGroupConfig(entry[0], fallbackColor);
          const group = groups.get(config.id) || { ...config, entries: [] };
          group.entries.push(entry);
          groups.set(config.id, group);
        });
        return Array.from(groups.values()).sort((a, b) => a.priority - b.priority || a.label.localeCompare(b.label));
      };

      const getMetadataEntries = (
        properties: Record<string, unknown> | undefined,
        hiddenKeys: Set<string>,
      limit = 12
    ): MetadataEntry[] =>
      Object.entries(properties || {})
        .filter(([key, value]) => !hiddenKeys.has(normalizeType(key)) && !isBlankMetadataValue(value))
        .sort(([aKey], [bKey]) => {
          const aNormalized = normalizeType(aKey);
          const bNormalized = normalizeType(bKey);
          const aPriority = METADATA_KEY_PRIORITY.get(aNormalized) ?? 100;
          const bPriority = METADATA_KEY_PRIORITY.get(bNormalized) ?? 100;
          return aPriority - bPriority || getTypeLabel(aKey).localeCompare(getTypeLabel(bKey));
        })
        .slice(0, limit);

    type MetadataScalarKind = 'boolean' | 'number' | 'url' | 'path' | 'code' | 'text';

    const getMetadataScalarKind = (value: unknown, metadataKey?: string): MetadataScalarKind => {
      const key = normalizeType(metadataKey || '');
      if (typeof value === 'boolean') return 'boolean';
      if (typeof value === 'number') return 'number';
      if (typeof value !== 'string') return 'text';

      const text = value.trim();
      if (/^https?:\/\//i.test(text)) return 'url';
      if (/(^|[/\\])[\w.-]+\.(tsx?|jsx?|py|json|ya?ml|md|txt|gguf|safetensors|bin|csv|parquet)$/i.test(text) || /[/\\][\w.-]+[/\\]/.test(text)) return 'path';
      if (
        /\b(path|file|source|document|module|route|endpoint|model|adapter|class|function|method|symbol|commit|sha|hash)\b/.test(key) ||
        /^[\w./:-]+$/.test(text) && /[./:_-]/.test(text) && text.length <= 96
      ) {
        return 'code';
      }
      return 'text';
    };

    const MetadataScalarValue = ({
      metadataKey,
      value,
    }: {
      metadataKey?: string;
      value: unknown;
    }) => {
      const kind = getMetadataScalarKind(value, metadataKey);
      const rawText = formatMetadataValue(value);
      const valueText = kind === 'boolean'
        ? (value === true ? 'Yes' : 'No')
        : rawText;
      const color = kind === 'boolean'
        ? (value === true ? accentColors.success : accentColors.slate)
        : kind === 'number'
          ? accentColors.warning
          : kind === 'url'
            ? accentColors.info
            : kind === 'path' || kind === 'code'
              ? accentColors.primary
              : 'text.secondary';

      if (kind === 'boolean' || kind === 'number') {
        return (
          <Chip
            size="small"
            label={valueText}
            sx={{
              height: 20,
              maxWidth: '100%',
              color,
              bgcolor: alpha(color, 0.1),
              border: '1px solid',
              borderColor: alpha(color, 0.24),
              '& .MuiChip-label': {
                px: 0.62,
                fontSize: '0.6rem',
                fontWeight: 880,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              },
            }}
          />
        );
      }

      if (kind === 'url') {
        return (
          <Box
            component="a"
            href={rawText}
            target="_blank"
            rel="noreferrer"
            title={rawText}
            sx={{
              display: 'inline-block',
              maxWidth: '100%',
              color,
              fontSize: '0.67rem',
              fontWeight: 800,
              lineHeight: 1.3,
              textDecoration: 'none',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              '&:hover': { textDecoration: 'underline' },
            }}
          >
            {truncateGraphLabel(rawText.replace(/^https?:\/\//i, ''), 72)}
          </Box>
        );
      }

      if (kind === 'path' || kind === 'code') {
        return (
          <Box
            component="code"
            title={rawText}
            sx={{
              display: 'inline-block',
              maxWidth: '100%',
              px: 0.55,
              py: 0.18,
              borderRadius: '6px',
              color,
              bgcolor: alpha(color, 0.08),
              border: '1px solid',
              borderColor: alpha(color, 0.16),
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
              fontSize: '0.62rem',
              fontWeight: 750,
              lineHeight: 1.35,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {truncateGraphLabel(rawText, 88)}
          </Box>
        );
      }

      return (
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{
            display: '-webkit-box',
            WebkitLineClamp: 3,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
            lineHeight: 1.35,
            wordBreak: 'break-word',
          }}
        >
          {rawText}
        </Typography>
      );
    };

    const MetadataValue = ({ metadataKey, value, depth = 0 }: { metadataKey?: string; value: unknown; depth?: number }) => {
      if (isBlankMetadataValue(value)) {
        return (
          <Typography variant="caption" color="text.disabled">
            Not set
          </Typography>
        );
      }

      if (Array.isArray(value)) {
        const visibleItems = value.filter((item) => !isBlankMetadataValue(item)).slice(0, 6);
        const hiddenCount = Math.max(0, value.length - visibleItems.length);
        const hasComplexItems = visibleItems.some((item) => typeof item === 'object' && item !== null);
        if (hasComplexItems) {
          return (
            <Box data-kg-metadata-array="records" sx={{ display: 'grid', gap: 0.45, minWidth: 0 }}>
              {visibleItems.slice(0, 3).map((item, index) => (
                <Box
                  key={`${formatMetadataValue(item)}-${index}`}
                  sx={{
                    minWidth: 0,
                    pl: 0.65,
                    py: 0.35,
                    borderLeft: `2px solid ${alpha(accentColors.primary, 0.34)}`,
                    bgcolor: alpha('#fff', 0.025),
                    borderRadius: '0 7px 7px 0',
                  }}
                >
                  {typeof item === 'object' && item !== null ? (
                    <MetadataValue value={item} depth={depth + 1} />
                  ) : (
                    <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1.35, wordBreak: 'break-word' }}>
                      {formatMetadataValue(item)}
                    </Typography>
                  )}
                </Box>
              ))}
              {hiddenCount + Math.max(0, visibleItems.length - 3) > 0 && (
                <Typography variant="caption" color="text.disabled" sx={{ fontWeight: 800 }}>
                  +{hiddenCount + Math.max(0, visibleItems.length - 3)} more records
                </Typography>
              )}
            </Box>
          );
        }

        return (
          <Box sx={{ display: 'flex', gap: 0.4, flexWrap: 'wrap', minWidth: 0 }}>
            {visibleItems.map((item, index) => (
              <Chip
                key={`${formatMetadataValue(item)}-${index}`}
                size="small"
                label={truncateGraphLabel(formatMetadataValue(item), 42)}
                variant="outlined"
                sx={{
                  height: 20,
                  maxWidth: '100%',
                  bgcolor: alpha('#fff', 0.04),
                  '& .MuiChip-label': {
                    px: 0.62,
                    fontSize: '0.61rem',
                    fontWeight: 750,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  },
                }}
              />
            ))}
            {hiddenCount > 0 && (
              <Chip
                size="small"
                label={`+${hiddenCount}`}
                sx={{
                  height: 20,
                  bgcolor: alpha('#fff', 0.06),
                  '& .MuiChip-label': { px: 0.62, fontSize: '0.61rem', fontWeight: 800 },
                }}
              />
            )}
          </Box>
        );
      }

      if (typeof value === 'object') {
        const allEntries = Object.entries(value as Record<string, unknown>)
          .filter(([, item]) => !isBlankMetadataValue(item))
          .sort(([aKey], [bKey]) => {
            const aPriority = METADATA_KEY_PRIORITY.get(normalizeType(aKey)) ?? 100;
            const bPriority = METADATA_KEY_PRIORITY.get(normalizeType(bKey)) ?? 100;
            return aPriority - bPriority || getTypeLabel(aKey).localeCompare(getTypeLabel(bKey));
          });
        const entries = allEntries.slice(0, depth > 0 ? 4 : 6);
        const hiddenCount = Math.max(0, allEntries.length - entries.length);

        if (entries.length === 0) return null;

        return (
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: 'minmax(76px, 0.38fr) minmax(0, 1fr)',
              gap: 0.45,
              minWidth: 0,
            }}
          >
            {entries.map(([key, item]) => (
              <Box key={key} sx={{ display: 'contents' }}>
                <Typography variant="caption" color="text.disabled" noWrap>
                  {getTypeLabel(key)}
                </Typography>
                {depth < 1 && typeof item === 'object' && item !== null ? (
                  <MetadataValue metadataKey={key} value={item} depth={depth + 1} />
                ) : (
                  <MetadataScalarValue metadataKey={key} value={item} />
                )}
              </Box>
            ))}
            {hiddenCount > 0 && (
              <>
                <Typography variant="caption" color="text.disabled" noWrap>
                  More
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  +{hiddenCount} fields
                </Typography>
              </>
            )}
          </Box>
        );
      }

      return <MetadataScalarValue metadataKey={metadataKey} value={value} />;
    };

    const MetadataGrid = ({ entries }: { entries: MetadataEntry[] }) => (
      <Box sx={{ display: 'grid', gridTemplateColumns: 'minmax(90px, 0.36fr) minmax(0, 1fr)', gap: 0.6 }}>
        {entries.map(([key, value]) => (
          <Box key={key} sx={{ display: 'contents' }}>
            <Typography variant="caption" color="text.disabled" noWrap>
              {getTypeLabel(key)}
            </Typography>
            <MetadataValue metadataKey={key} value={value} />
          </Box>
        ))}
      </Box>
    );

    const getMetadataPreviewItems = (entries: MetadataEntry[], maxItems = 3) =>
      entries.slice(0, maxItems).map(([key, value]) => ({
        key,
        label: `${getTypeLabel(key)}: ${truncateGraphLabel(formatMetadataValue(value), 44)}`,
      }));

    const normalizeEvidenceText = normalizeGraphText;
    const formatEvidenceDisplayText = normalizeGraphDisplayText;
    const isEvidenceExcerpt = isGraphEvidenceExcerpt;
    const formatEvidenceSnippetDisplayText = formatGraphEvidenceSnippetText;
    const getCompactEvidencePreviewText = (value?: string | null) => {
      const text = formatEvidenceSnippetDisplayText(value);
      if (!text) return '';
      if (!isEvidenceExcerpt(text)) return text;
      return text;
    };

    const getEvidenceDetailScore = (evidence: RelationshipEvidence) => {
      const text = formatEvidenceDisplayText(evidence.context);
      const lengthScore = Math.min(240, text.length / 3);
      const completenessScore = isEvidenceExcerpt(text) ? -120 : 80;
      const domainScore = evidence.domain ? 8 : 0;
      const weightScore = Math.min(60, Math.max(0, evidence.weight) * 4);
      return completenessScore + lengthScore + domainScore + weightScore;
    };

    const getRelationshipEvidence = (edge: Pick<GraphEdge, 'properties' | 'type' | 'label' | 'weight'>): RelationshipEvidence | null => {
      const context = normalizeEvidenceText(edge.properties?.context);
      if (!context) return null;
      const domain = typeof edge.properties?.domain === 'string' ? edge.properties.domain : undefined;
      const formattedContext = /^[a-z),.;:]/.test(context) ? `...${context}` : context;
      return {
        context: formattedContext,
        domain,
        type: normalizeType(edge.type || edge.label),
        weight: Number(edge.weight || 0),
      };
    };

    const getRelationshipEvidenceSnippets = (
      edge: Pick<GraphEdge, 'properties' | 'type' | 'label' | 'weight'>,
      limit = 3
    ): RelationshipEvidence[] => {
      const rawEvidence: RawEvidenceSnippet[] = Array.isArray(edge.properties?.evidenceSnippets)
        ? edge.properties.evidenceSnippets as RawEvidenceSnippet[]
        : [];
      const snippets = rawEvidence
        .map((item) => ({
          context: normalizeEvidenceText(item.context),
          domain: typeof item.domain === 'string' ? item.domain : undefined,
          type: normalizeType(item.type || edge.type || edge.label),
          weight: Number(item.weight || edge.weight || 0),
        }))
        .filter((item: RelationshipEvidence) => item.context);

      const fallbackEvidence = getRelationshipEvidence(edge);
      const combined = fallbackEvidence ? [...snippets, fallbackEvidence] : snippets;
      const seen = new Set<string>();
      return combined.filter((item) => {
        const key = item.context.toLowerCase();
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      }).slice(0, limit);
    };

      const getEntitySearchText = (entity: Entity | GraphNode) => {
        const aliases = 'aliases' in entity
          ? entity.aliases?.join(' ')
          : Array.isArray(entity.properties?.aliases)
            ? entity.properties.aliases.map(String).join(' ')
            : '';
        return [
          'name' in entity ? entity.name : entity.label,
          'type' in entity ? entity.type : entity.entity_type,
          getUsefulEntityDescription(entity.description),
          aliases,
          getSearchableEntityMetadataText(entity.properties),
        ].filter(Boolean).join(' ').toLowerCase();
      };

      const getEdgeSearchText = (edge: GraphEdge) => [
        edge.searchText,
        edge.source,
        edge.target,
        ...getTypeSearchAliases(edge.label),
        ...getTypeSearchAliases(edge.type),
        ...(Array.isArray(edge.properties?.relationTypes)
          ? (edge.properties.relationTypes as RelationTypePayload[]).flatMap((item) => getTypeSearchAliases(item.type))
          : []),
        formatMetadataValue(edge.properties || {}),
      ].filter(Boolean).join(' ').toLowerCase();

    const getTextSearchSnippet = getGraphTextSearchSnippet;

    const getEvidenceSearchSnippet = (items: RawEvidenceSnippet[], query: string) => {
      if (!query) return '';
      const matched = items
        .map((item) => normalizeEvidenceText(item.context))
        .find((context) => context.toLowerCase().includes(query));
      if (!matched) return '';
      return getTextSearchSnippet(matched, query);
    };

    const getEntitySearchSnippet = (entity: Entity | GraphNode, query: string) => {
      if (!query) return '';
      const aliases = 'aliases' in entity
        ? entity.aliases?.join(', ')
        : Array.isArray(entity.properties?.aliases)
          ? entity.properties.aliases.map(String).join(', ')
          : '';
      const detailText = [
        getUsefulEntityDescription(entity.description),
        aliases,
        getSearchableEntityMetadataText(entity.properties),
      ].filter(Boolean).join(' ');
      return getTextSearchSnippet(detailText, query, 56, 118);
    };

  const getGraphRenderProfile = (nodeCount: number, linkCount: number): GraphRenderProfile => {
    if (nodeCount > 3000 || linkCount > 7000) {
      return {
        tier: 'massive',
        nodeRelSize: 2.55,
        nodeResolution: 4,
        linkResolution: 1,
        arrowResolution: 4,
        warmupTicks: 8,
        cooldownTicks: 18,
        maxPixelRatio: 1,
          nodeBudget: { far: 620, middle: 1300, near: 2600 },
            labelBudget: { far: 10, middle: 24, near: 48 },
            iconBudget: { far: 148, middle: 360, near: 760 },
        linkBudget: { far: 650, middle: 1100, near: 1800 },
        alwaysUseCustomNodes: false,
      };
    }

    if (nodeCount > 1200 || linkCount > 2400) {
      return {
        tier: 'massive',
        nodeRelSize: 2.68,
        nodeResolution: 5,
        linkResolution: 2,
        arrowResolution: 5,
        warmupTicks: 12,
        cooldownTicks: 28,
        maxPixelRatio: 1.25,
          nodeBudget: { far: 560, middle: 1150, near: 2200 },
            labelBudget: { far: 12, middle: 26, near: 54 },
            iconBudget: { far: 168, middle: 390, near: 820 },
        linkBudget: { far: 820, middle: 1450, near: 2400 },
        alwaysUseCustomNodes: false,
      };
    }

    if (nodeCount > 350 || linkCount > 900) {
      return {
        tier: 'large',
        nodeRelSize: 3.05,
        nodeResolution: 6,
        linkResolution: 3,
        arrowResolution: 6,
        warmupTicks: 28,
        cooldownTicks: 64,
        maxPixelRatio: 1.5,
          nodeBudget: { far: 420, middle: 840, near: 1600 },
            labelBudget: { far: 24, middle: 54, near: 96 },
            iconBudget: { far: 360, middle: 780, near: 1400 },
        linkBudget: { far: 1200, middle: 2200, near: 3600 },
        alwaysUseCustomNodes: true,
      };
    }

    if (nodeCount > 90 || linkCount > 240) {
      return {
        tier: 'standard',
        nodeRelSize: 3.35,
        nodeResolution: 8,
        linkResolution: 4,
        arrowResolution: 8,
        warmupTicks: 92,
        cooldownTicks: 42,
        maxPixelRatio: 1.75,
          nodeBudget: { far: Number.POSITIVE_INFINITY, middle: Number.POSITIVE_INFINITY, near: Number.POSITIVE_INFINITY },
            labelBudget: { far: 58, middle: 108, near: 156 },
            iconBudget: { far: Number.POSITIVE_INFINITY, middle: Number.POSITIVE_INFINITY, near: Number.POSITIVE_INFINITY },
        linkBudget: { far: 1800, middle: 3200, near: 5200 },
        alwaysUseCustomNodes: true,
      };
    }

    return {
      tier: 'compact',
      nodeRelSize: 3.8,
      nodeResolution: 12,
      linkResolution: 6,
      arrowResolution: 10,
      warmupTicks: 110,
      cooldownTicks: 220,
      maxPixelRatio: 2,
        nodeBudget: { far: Number.POSITIVE_INFINITY, middle: Number.POSITIVE_INFINITY, near: Number.POSITIVE_INFINITY },
          labelBudget: { far: 72, middle: 128, near: 188 },
      iconBudget: { far: Number.POSITIVE_INFINITY, middle: Number.POSITIVE_INFINITY, near: Number.POSITIVE_INFINITY },
      linkBudget: { far: Number.POSITIVE_INFINITY, middle: Number.POSITIVE_INFINITY, near: Number.POSITIVE_INFINITY },
      alwaysUseCustomNodes: true,
    };
  };

  const scaleGraphBudget = (budget: Record<GraphZoomTier, number>, scale: number) => ({
    far: Number.isFinite(budget.far) ? (budget.far <= 0 ? 0 : Math.max(4, Math.round(budget.far * scale))) : budget.far,
    middle: Number.isFinite(budget.middle) ? (budget.middle <= 0 ? 0 : Math.max(8, Math.round(budget.middle * scale))) : budget.middle,
    near: Number.isFinite(budget.near) ? (budget.near <= 0 ? 0 : Math.max(12, Math.round(budget.near * scale))) : budget.near,
  });

  const scaleGraphLabelBudget = (budget: Record<GraphZoomTier, number>, scale: number) => ({
    far: Number.isFinite(budget.far) ? Math.max(0, Math.round(budget.far * scale)) : budget.far,
    middle: Number.isFinite(budget.middle) ? Math.max(0, Math.round(budget.middle * scale)) : budget.middle,
    near: Number.isFinite(budget.near) ? Math.max(0, Math.round(budget.near * scale)) : budget.near,
  });

  const applyGraphDetailMode = (
    profile: GraphRenderProfile,
    detailMode: GraphDetailMode
  ): GraphRenderProfile => {
    if (detailMode === 'auto') return profile;

    if (detailMode === 'performance') {
      return {
        ...profile,
        nodeResolution: Math.max(4, Math.floor(profile.nodeResolution * 0.72)),
        linkResolution: Math.max(1, Math.floor(profile.linkResolution * 0.64)),
        arrowResolution: Math.max(4, Math.floor(profile.arrowResolution * 0.7)),
        warmupTicks: Math.max(10, Math.floor(profile.warmupTicks * 0.65)),
        cooldownTicks: Math.max(24, Math.floor(profile.cooldownTicks * 0.62)),
        maxPixelRatio: Math.min(profile.maxPixelRatio, 1.15),
        nodeBudget: scaleGraphBudget(profile.nodeBudget, 0.72),
        labelBudget: scaleGraphLabelBudget(profile.labelBudget, 0.72),
        iconBudget: scaleGraphBudget(profile.iconBudget, 0.82),
        linkBudget: scaleGraphBudget(profile.linkBudget, 0.7),
      };
    }

    return {
      ...profile,
      nodeResolution: profile.tier === 'massive' ? Math.max(profile.nodeResolution, 6) : Math.min(12, Math.ceil(profile.nodeResolution * 1.25)),
      linkResolution: profile.tier === 'massive' ? Math.max(profile.linkResolution, 3) : Math.min(6, Math.ceil(profile.linkResolution * 1.25)),
      arrowResolution: Math.min(10, Math.ceil(profile.arrowResolution * 1.2)),
      warmupTicks: Math.min(profile.tier === 'massive' ? 54 : 160, Math.ceil(profile.warmupTicks * 1.25)),
      cooldownTicks: Math.min(profile.tier === 'massive' ? 96 : 260, Math.ceil(profile.cooldownTicks * 1.25)),
      maxPixelRatio: Math.min(profile.tier === 'massive' ? 1.4 : 2, profile.maxPixelRatio + 0.25),
      nodeBudget: scaleGraphBudget(profile.nodeBudget, profile.tier === 'massive' ? 1.25 : 1.45),
      labelBudget: scaleGraphLabelBudget(profile.labelBudget, profile.tier === 'massive' ? 1.25 : 1.45),
      iconBudget: scaleGraphBudget(profile.iconBudget, profile.tier === 'massive' ? 1.35 : 1.65),
      linkBudget: scaleGraphBudget(profile.linkBudget, 1.4),
    };
  };

  const getGraphZoomThresholds = (nodeCount: number) => {
    const scale = nodeCount > 3000 ? 2.7 : nodeCount > 1200 ? 2.1 : nodeCount > 350 ? 1.55 : nodeCount > 90 ? 1.22 : 1;
    return {
      near: 125 * scale,
      middle: 360 * scale,
    };
  };

  const getGraphZoomTier = (distance: number, nodeCount: number): GraphZoomTier => {
    const thresholds = getGraphZoomThresholds(nodeCount);
    if (distance < thresholds.near) return 'near';
    if (distance < thresholds.middle) return 'middle';
    return 'far';
  };

  const getStableGraphZoomTier = (
    distance: number,
    nodeCount: number,
    currentTier: GraphZoomTier,
    moving: boolean
  ): GraphZoomTier => {
    const thresholds = getGraphZoomThresholds(nodeCount);
    const buffer = moving ? 0.24 : 0.14;
    if (currentTier === 'near' && distance < thresholds.near * (1 + buffer)) return 'near';
    if (currentTier === 'middle') {
      if (distance < thresholds.near * (1 - buffer)) return 'near';
      if (distance < thresholds.middle * (1 + buffer)) return 'middle';
      return 'far';
    }
    if (currentTier === 'far' && distance > thresholds.middle * (1 - buffer)) return 'far';
    return getGraphZoomTier(distance, nodeCount);
  };

  const getGraphCameraKey = (position: THREE.Vector3, target: THREE.Vector3, nodeCount: number) => {
    const distance = position.distanceTo(target);
    const direction = position.clone().sub(target).normalize();
    const zoomTier = getGraphZoomTier(distance, nodeCount);
    return [
      zoomTier,
      Math.round(distance / 96),
      Math.round(direction.x * 2.2),
      Math.round(direction.y * 2.2),
      Math.round(direction.z * 2.2),
    ].join(':');
  };

  const getNodeImportanceScore = (
    node: GraphNode,
    summary?: { links: number; relations: number }
  ) => {
    const relations = summary?.relations || 0;
    const links = summary?.links || 0;
    const occurrence = Number(node.occurrence || node.properties?.occurrence || 0);
    const confidence = Number(node.confidence || node.properties?.confidence || 0);
    return relations * 11 + links * 28 + Math.log2(occurrence + 1) * 18 + confidence;
  };

  const SELECTION_CONTEXT_TYPE_BONUS: Record<string, number> = {
    api: 126,
    code: 154,
    data: 84,
    evaluation: 112,
    hardware: 74,
    memory: 92,
    model: 86,
    organization: 76,
    process: 148,
    prompt: 74,
    runtime: 128,
    security: 76,
    service: 150,
    technology: 88,
    vector: 144,
  };

  const LOW_VALUE_CONTEXT_LABELS = new Set([
    'date',
    'entity',
    'id',
    'label',
    'model',
    'name',
    'node',
    'nodes',
    'relation',
    'relations',
    'score',
    'text',
    'time',
    'type',
    'value',
  ]);

  const getSelectionContextLabelQualityScore = (
    node: GraphNode,
    summary?: { links: number; relations: number }
  ) => {
    const displayLabel = getGraphNodeDisplayLabel(node.label || node.id, node.type).trim();
    const normalizedLabel = normalizeGraphDisplayText(displayLabel).toLowerCase();
    if (!normalizedLabel) return Number.NEGATIVE_INFINITY;

    const visualType = getEntityVisualType(node.type, node.label);
    const baseScore = getNodeImportanceScore(node, summary);
    let qualityScore = baseScore + (SELECTION_CONTEXT_TYPE_BONUS[visualType] || 0);

    if (visualType === 'date') return Number.NEGATIVE_INFINITY;
    if (visualType === 'metric') qualityScore -= 96;
    if (visualType === 'config') qualityScore -= 86;
    if (visualType === 'custom') qualityScore -= 38;

    if (LOW_VALUE_CONTEXT_LABELS.has(normalizedLabel)) return Number.NEGATIVE_INFINITY;
    if (/^[+\-–]?\d+(?:[.,]\d+)?(?:\s*(?:ms|s|sec|secs|second|seconds|minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years))?$/i.test(displayLabel)) {
      return Number.NEGATIVE_INFINITY;
    }
    if (/^\d{4}(?:[-/]\d{1,2}){0,2}$/.test(normalizedLabel)) return Number.NEGATIVE_INFINITY;
    if (/^(?:q\d+_[a-z0-9_]+|fp\d+|bf\d+|int\d+|f\d+)$/i.test(displayLabel)) return Number.NEGATIVE_INFINITY;
    if (/^(?:maximum|minimum|min|max)\s+(?:text|token|context|sequence)\s+(?:length|size)$/i.test(displayLabel)) return Number.NEGATIVE_INFINITY;
    if (normalizedLabel.length <= 4 && !/[/\\.]/.test(displayLabel)) qualityScore -= 120;

    if (/[./\\]/.test(displayLabel)) qualityScore += 118;
    if (/[a-z][A-Z]/.test(displayLabel) || /\b(?:Page|Service|Manager|Engine|Processor|Dataset|Workflow|GraphRAG|Backend|Frontend|Runtime|Registry|Router|Store)\b/.test(displayLabel)) {
      qualityScore += 92;
    }
    if (/\b(?:service|manager|engine|processor|workflow|dataset|document|frontend|backend|runtime|adapter|store|route|page|registry|deploy|graph|rag)\b/i.test(displayLabel)) {
      qualityScore += 68;
    }

    return qualityScore;
  };

  const getNodeCameraBand = (
    node: RenderGraphNode,
    camera: GraphCameraLod
  ): { depthBand: GraphDepthBand; offAxis: boolean } => {
    const nodeVector = new THREE.Vector3(
      Number(node.x || 0) - camera.position.x,
      Number(node.y || 0) - camera.position.y,
      Number(node.z || 0) - camera.position.z
    );

    const direction = new THREE.Vector3(camera.direction.x, camera.direction.y, camera.direction.z);
    const forwardDepth = nodeVector.dot(direction);
    const lateralDistance = nodeVector.clone().sub(direction.multiplyScalar(forwardDepth)).length();
    const lateralLimit = camera.distance * (
      camera.zoomTier === 'near' ? 0.72 : camera.zoomTier === 'middle' ? 1.05 : 1.42
    );

    if (forwardDepth < camera.distance * 0.2) {
      return { depthBand: 'back', offAxis: true };
    }
    if (forwardDepth <= camera.distance * 1.04) {
      return { depthBand: 'front', offAxis: lateralDistance > lateralLimit };
    }
    if (forwardDepth <= camera.distance * 1.58) {
      return { depthBand: 'middle', offAxis: lateralDistance > lateralLimit * 1.18 };
    }
    return { depthBand: 'back', offAxis: lateralDistance > lateralLimit * 1.32 };
  };

  const buildGraphComponentSummaries = (nodes: GraphNode[], edges: GraphEdge[]): GraphComponentSummary[] => {
    const nodeById = new Map(nodes.map((node) => [node.id, node]));
    const adjacency = new Map<string, Set<string>>();
    const metrics = new Map<string, { links: number; relations: number }>();
    const typeCounts = new Map<string, Map<string, number>>();

    nodes.forEach((node) => {
      adjacency.set(node.id, new Set());
      metrics.set(node.id, { links: 0, relations: 0 });
    });

    edges.forEach((edge) => {
      if (!nodeById.has(edge.source) || !nodeById.has(edge.target)) return;
      const relationCount = Math.max(1, Number(edge.relationCount || 1));
      adjacency.get(edge.source)?.add(edge.target);
      adjacency.get(edge.target)?.add(edge.source);

      const sourceMetrics = metrics.get(edge.source);
      const targetMetrics = metrics.get(edge.target);
      if (sourceMetrics) {
        sourceMetrics.links += 1;
        sourceMetrics.relations += relationCount;
      }
      if (targetMetrics) {
        targetMetrics.links += 1;
        targetMetrics.relations += relationCount;
      }
    });

    const visited = new Set<string>();
    const components: GraphComponentSummary[] = [];

    nodes.forEach((node) => {
      if (visited.has(node.id)) return;
      const nodeIds = new Set<string>();
      const stack = [node.id];
      let linkCount = 0;
      let relationCount = 0;
      visited.add(node.id);

      while (stack.length > 0) {
        const id = stack.pop()!;
        nodeIds.add(id);
        const nodeMetrics = metrics.get(id);
        linkCount += nodeMetrics?.links || 0;
        relationCount += nodeMetrics?.relations || 0;

        const nodeType = normalizeType(nodeById.get(id)?.type);
        const componentTypeCounts = typeCounts.get(node.id) || new Map<string, number>();
        componentTypeCounts.set(nodeType, (componentTypeCounts.get(nodeType) || 0) + 1);
        typeCounts.set(node.id, componentTypeCounts);

        adjacency.get(id)?.forEach((neighborId) => {
          if (!visited.has(neighborId)) {
            visited.add(neighborId);
            stack.push(neighborId);
          }
        });
      }

      const componentNodes = Array.from(nodeIds)
        .map((id) => nodeById.get(id))
        .filter((item): item is GraphNode => Boolean(item));
      const topNode = componentNodes
        .sort((a, b) =>
          getNodeImportanceScore(b, metrics.get(b.id)) -
          getNodeImportanceScore(a, metrics.get(a.id)) ||
          a.label.localeCompare(b.label)
        )[0];
      const entityTypes = Array.from((typeCounts.get(node.id) || new Map<string, number>()).entries())
        .map(([type, count]) => ({ type, count }))
        .sort((a, b) => b.count - a.count || a.type.localeCompare(b.type));

      components.push({
        id: `component:${components.length + 1}`,
        label: `Component ${components.length + 1}`,
        nodeIds,
        nodeCount: nodeIds.size,
        linkCount: Math.round(linkCount / 2),
        relationCount: Math.round(relationCount / 2),
        topNodeId: topNode?.id,
        topNodeLabel: topNode?.label,
        topNodeType: topNode?.type,
        entityTypes,
      });
    });

    return components.sort((a, b) =>
      b.relationCount - a.relationCount ||
      b.linkCount - a.linkCount ||
      b.nodeCount - a.nodeCount ||
      (a.topNodeLabel || a.label).localeCompare(b.topNodeLabel || b.label)
    ).map((component, index) => ({
      ...component,
      id: `component:${index + 1}`,
      label: `Component ${index + 1}`,
    }));
  };

    const getPrimaryConnectedComponentIds = (nodes: GraphNode[], edges: GraphEdge[]) => {
      const nodeIds = new Set(nodes.map((node) => node.id));
      const adjacency = new Map<string, Set<string>>();
      const relationWeight = new Map<string, number>();

      nodeIds.forEach((id) => {
        adjacency.set(id, new Set());
        relationWeight.set(id, 0);
      });

      edges.forEach((edge) => {
        if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) return;
        adjacency.get(edge.source)?.add(edge.target);
        adjacency.get(edge.target)?.add(edge.source);
        relationWeight.set(edge.source, (relationWeight.get(edge.source) || 0) + 1);
        relationWeight.set(edge.target, (relationWeight.get(edge.target) || 0) + 1);
      });

      const visited = new Set<string>();
      const components: { ids: Set<string>; score: number }[] = [];

      nodes.forEach((node) => {
        if (visited.has(node.id)) return;
        const ids = new Set<string>();
        const stack = [node.id];
        let score = 0;
        visited.add(node.id);

        while (stack.length > 0) {
          const id = stack.pop()!;
          ids.add(id);
          score += relationWeight.get(id) || 0;
          adjacency.get(id)?.forEach((neighborId) => {
            if (!visited.has(neighborId)) {
              visited.add(neighborId);
              stack.push(neighborId);
            }
          });
        }

        components.push({ ids, score });
      });

      return components.sort((a, b) => b.score - a.score || b.ids.size - a.ids.size)[0]?.ids || nodeIds;
    };

const truncateGraphLabel = (value: string, maxLength = 28) =>
  value.length > maxLength ? `${value.slice(0, maxLength - 1)}...` : value;

const getGraphNodeDisplayLabel = (value: string, type?: string) => {
  const normalized = value.trim();
  if (!normalized) return value;
  const looksLikePath = /[/\\]/.test(normalized);
  if (!looksLikePath) return normalized;

  const segments = normalized
    .split(/[/\\]+/)
    .map((segment) => segment.trim())
    .filter(Boolean);
  if (segments.length <= 1) return normalized;

  const leaf = segments[segments.length - 1];
  const parent = segments[segments.length - 2];
  const codeLike = /\.(?:c|cc|cpp|css|cu|go|h|hpp|html|java|js|jsx|json|md|py|rs|sh|sql|ts|tsx|yaml|yml)$/i.test(leaf) ||
    ['code', 'data', 'prompt', 'config'].includes(getEntityVisualType(type, normalized));
  if (!codeLike) return leaf.length <= 24 ? leaf : `${parent}/${leaf}`;
  if (/^(?:__init__|index|main|app|types?)\.[a-z0-9]+$/i.test(leaf)) return `${parent}/${leaf}`;
  return leaf;
};

const truncateMeasuredGraphLabel = (
  ctx: CanvasRenderingContext2D,
  value: string,
  maxWidth: number
) => {
  const normalized = value.trim();
  if (!normalized) return '';
  if (ctx.measureText(normalized).width <= maxWidth) return normalized;
  const suffix = '...';
  let low = 1;
  let high = normalized.length;
  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    if (ctx.measureText(`${normalized.slice(0, mid).trimEnd()}${suffix}`).width <= maxWidth) {
      low = mid;
    } else {
      high = mid - 1;
    }
  }
  return `${normalized.slice(0, Math.max(1, low)).trimEnd()}${suffix}`;
};

const wrapMeasuredGraphLabel = (
  ctx: CanvasRenderingContext2D,
  value: string,
  maxWidth: number,
  maxChars: number,
  maxLines: number
) => {
  const normalized = truncateGraphLabel(value.trim(), maxChars);
  if (!normalized) return [''];
  if (maxLines <= 1 || ctx.measureText(normalized).width <= maxWidth) {
    return [truncateMeasuredGraphLabel(ctx, normalized, maxWidth)];
  }

  const lines: string[] = [];
  let remaining = normalized;
  while (remaining && lines.length < maxLines) {
    if (lines.length === maxLines - 1) {
      lines.push(truncateMeasuredGraphLabel(ctx, remaining, maxWidth));
      break;
    }

    let fitEnd = 1;
    while (
      fitEnd < remaining.length &&
      ctx.measureText(remaining.slice(0, fitEnd + 1).trim()).width <= maxWidth
    ) {
      fitEnd += 1;
    }

    const fitText = remaining.slice(0, fitEnd);
    const minimumBreak = Math.max(4, Math.floor(fitEnd * 0.48));
    const delimiterBreaks = Array.from(fitText.matchAll(/[\s/_.-]/g))
      .map((match) => match.index ?? -1)
      .filter((index) => index >= 0)
      .map((index) => index + 1);
    const camelBreaks = Array.from(fitText.matchAll(/[a-z0-9][A-Z]/g))
      .map((match) => (match.index ?? -1) + 1)
      .filter((index) => index > 0);
    const breakAtCandidate = [...delimiterBreaks, ...camelBreaks]
      .filter((index) => index >= minimumBreak)
      .sort((a, b) => a - b)
      .pop();
    const breakAt = breakAtCandidate && breakAtCandidate > 0 ? breakAtCandidate : fitEnd;
    const nextLine = remaining.slice(0, breakAt).trim().replace(/[./_-]+$/, '');
    if (nextLine) lines.push(nextLine);
    remaining = remaining.slice(breakAt).trim().replace(/^[./_-]+/, '');
  }

  return lines.length > 0 ? lines : [truncateMeasuredGraphLabel(ctx, normalized, maxWidth)];
};

const getRelationDisplayLabel = (link: Pick<GraphEdge, 'label' | 'type' | 'relationCount'>) => {
  const count = Math.max(1, Number(link.relationCount || 1));
  const baseLabel = truncateGraphLabel(getTypeLabel(link.type || link.label || 'related'), count > 1 ? 19 : 24);
  return count > 1 ? `${baseLabel} x${count.toLocaleString()}` : baseLabel;
};

const getLinkRelationTypeCounts = (
  link: Pick<GraphEdge, 'label' | 'type' | 'relationCount' | 'properties'>,
  limit = 5
): RelationTypeCount[] => {
  const rawRelationTypes: RelationTypePayload[] = Array.isArray(link.properties?.relationTypes)
    ? link.properties.relationTypes as RelationTypePayload[]
    : [];
  const relationTypes = (rawRelationTypes.length > 0
    ? rawRelationTypes
    : [{ type: normalizeType(link.type || link.label), count: link.relationCount || 1 }]
  )
    .map((item) => ({
      type: normalizeType(item.type),
      count: Math.max(1, Number(item.count || 0)),
    }))
    .filter((item): item is RelationTypeCount => Boolean(item.type && item.count > 0))
    .sort((a, b) => b.count - a.count || a.type.localeCompare(b.type));

  return relationTypes.slice(0, limit);
};

const getLinkCurveProfile = (link: Pick<GraphEdge, 'label' | 'type' | 'relationCount' | 'properties'>) => {
  const relationTypes = getLinkRelationTypeCounts(link, 3);
  const primaryType = relationTypes[0]?.type || normalizeType(link.type || link.label);
  const kind = getRelationVisualKind(primaryType);
  const base = RELATION_CURVE_PROFILES[kind];
  const relationCount = Math.max(1, Number(link.relationCount || relationTypes.reduce((sum, item) => sum + item.count, 0) || 1));
  const countBoost = Math.min(0.036, Math.log2(relationCount + 1) * 0.0045);
  const mixedTypeBoost = relationTypes.length > 1 ? 0.014 : 0;
  const direction = base.curvature < 0 ? -1 : 1;

  return {
    curvature: direction * Math.min(0.14, Math.abs(base.curvature) + countBoost + mixedTypeBoost),
    rotation: base.rotation + (relationTypes.length > 1 ? Math.PI * 0.045 : 0),
  };
};

const estimateNodeLabelScreenSize = (
  node: Pick<RenderGraphNode, 'id' | 'label' | 'type'>,
  style: GraphNodeLabelStyle,
  zoomTier: GraphZoomTier
) => {
  const rawLabel = getGraphNodeDisplayLabel(node.label || node.id, node.type);
  const maxChars = style === 'rich' ? 52 : style === 'annotation' ? 34 : style === 'anchor' ? 30 : style === 'neighbor' ? 34 : 30;
  const wrapped = rawLabel.length > (style === 'rich' ? 24 : style === 'annotation' ? 17 : style === 'anchor' ? 16 : style === 'neighbor' ? 18 : 18);
  const visibleLabelChars = Math.min(maxChars, wrapped ? Math.ceil(Math.min(rawLabel.length, maxChars) * 0.58) : rawLabel.length);
  const typeLabel = getEntityDisplayTypeLabel(node.type, node.label);
    const zoomScale = (zoomTier === 'near' ? 1.18 : zoomTier === 'middle' ? 1.1 : 1.04) * 1.16;
    if (style === 'rich') {
      return {
        width: Math.min(334, Math.max(156, Math.max(visibleLabelChars * 10.6, typeLabel.length * 6.4 + 84) + 78)) * zoomScale,
        height: (wrapped ? 68 : 54) * zoomScale,
      };
    }
    if (style === 'annotation') {
      return {
        width: Math.min(246, Math.max(122, visibleLabelChars * 10.2 + 58)) * zoomScale,
        height: (wrapped ? 48 : 38) * zoomScale,
      };
    }
    if (style === 'anchor') {
      return {
        width: Math.min(258, Math.max(114, visibleLabelChars * 9.2 + 48)) * zoomScale,
        height: (wrapped ? 48 : 34) * zoomScale,
      };
    }
    if (style === 'neighbor') {
      return {
        width: Math.min(238, Math.max(122, visibleLabelChars * 8.8 + 66)) * zoomScale,
        height: (wrapped ? 52 : 42) * zoomScale,
      };
    }

    return {
      width: Math.min(238, Math.max(112, visibleLabelChars * 9.2 + 48)) * zoomScale,
      height: (wrapped ? 48 : 36) * zoomScale,
    };
  };

const projectGraphPointToScreen = (
  point: { x?: number; y?: number; z?: number },
  camera: THREE.PerspectiveCamera,
  width: number,
  height: number
) => {
  if (!Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) return null;
  const projected = new THREE.Vector3(Number(point.x), Number(point.y), Number(point.z)).project(camera);
  if (!Number.isFinite(projected.x) || !Number.isFinite(projected.y) || !Number.isFinite(projected.z)) return null;
  if (projected.z < -1 || projected.z > 1 || Math.abs(projected.x) > 1.42 || Math.abs(projected.y) > 1.42) return null;
  return {
    x: (projected.x + 1) * 0.5 * width,
    y: (1 - projected.y) * 0.5 * height,
    depth: projected.z,
  };
};

const projectNodeToScreen = (
  node: RenderGraphNode,
  camera: THREE.PerspectiveCamera,
  width: number,
  height: number
) => projectGraphPointToScreen(node, camera, width, height);

const getLabelRectForAnchor = (
  screenX: number,
  screenY: number,
  labelWidth: number,
  labelHeight: number,
  anchor: GraphNodeLabelAnchor,
  gap: number
): GraphScreenRect => {
  if (anchor === 'left') {
    return {
      left: screenX - labelWidth - gap,
      right: screenX - gap,
      top: screenY - labelHeight * 0.5,
      bottom: screenY + labelHeight * 0.5,
    };
  }
  if (anchor === 'above') {
    return {
      left: screenX - labelWidth * 0.5,
      right: screenX + labelWidth * 0.5,
      top: screenY - labelHeight - gap,
      bottom: screenY - gap,
    };
  }
  if (anchor === 'below') {
    return {
      left: screenX - labelWidth * 0.5,
      right: screenX + labelWidth * 0.5,
      top: screenY + gap,
      bottom: screenY + labelHeight + gap,
    };
  }
  if (anchor === 'above-right') {
    const left = screenX + gap * 0.72;
    const bottom = screenY - gap * 0.58;
    return {
      left,
      right: left + labelWidth,
      top: bottom - labelHeight,
      bottom,
    };
  }
  if (anchor === 'above-left') {
    const right = screenX - gap * 0.72;
    const bottom = screenY - gap * 0.58;
    return {
      left: right - labelWidth,
      right,
      top: bottom - labelHeight,
      bottom,
    };
  }
  if (anchor === 'below-right') {
    const left = screenX + gap * 0.72;
    const top = screenY + gap * 0.58;
    return {
      left,
      right: left + labelWidth,
      top,
      bottom: top + labelHeight,
    };
  }
  if (anchor === 'below-left') {
    const right = screenX - gap * 0.72;
    const top = screenY + gap * 0.58;
    return {
      left: right - labelWidth,
      right,
      top,
      bottom: top + labelHeight,
    };
  }
  return {
    left: screenX + gap,
    right: screenX + labelWidth + gap,
    top: screenY - labelHeight * 0.5,
    bottom: screenY + labelHeight * 0.5,
  };
};

const padGraphRect = (rect: GraphScreenRect, padding: number): GraphScreenRect => ({
  left: rect.left - padding,
  top: rect.top - padding,
  right: rect.right + padding,
  bottom: rect.bottom + padding,
});

const graphRectsOverlap = (a: GraphScreenRect, b: GraphScreenRect) =>
  a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;

const escapeGraphSearchRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const graphRectFitsViewport = (rect: GraphScreenRect, width: number, height: number) =>
  rect.left >= 8 && rect.top >= 8 && rect.right <= width - 8 && rect.bottom <= height - 8;

const clampGraphValue = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value));

const lerpGraphValue = (from: number, to: number, amount: number) =>
  from + (to - from) * Math.max(0, Math.min(1, amount));

const getRectOverlapArea = (a: GraphScreenRect, b: GraphScreenRect) => {
  if (!graphRectsOverlap(a, b)) return 0;
  return Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left)) *
    Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
};

const getSelectionOverlayPlacement = (
  anchor: GraphSelectionAnchor | null,
  cardWidth: number,
  cardHeight: number,
  expanded: boolean,
  dockBottom = 82,
  avoidRects: GraphScreenRect[] = [],
  preferredEdge?: GraphSelectionPlacementEdge | null
): GraphOverlayPlacement => {
  const docked = {
    mode: 'docked' as const,
    right: 12,
    bottom: dockBottom,
    transformOrigin: 'right bottom',
  };
  if (!anchor || anchor.width < 720 || anchor.height < 520) return docked;
  if (expanded && (anchor.width < 960 || anchor.height < 620)) return docked;

  const gap = expanded
    ? anchor.kind === 'edge' ? 52 : 46
    : anchor.kind === 'edge' ? 38 : 34;
  const bounds = {
    left: 12,
    top: 12,
    right: anchor.width - 12,
    bottom: anchor.height - 12,
  };
  const reservedRects: GraphScreenRect[] = [
    { left: 0, top: 0, right: Math.min(anchor.width, 468), bottom: 126 },
    { left: Math.max(0, anchor.width - 288), top: 0, right: anchor.width, bottom: 92 },
    { left: 0, top: Math.max(0, anchor.height - 72), right: Math.min(anchor.width, 540), bottom: anchor.height },
    { left: Math.max(0, anchor.width - 88), top: Math.max(0, anchor.height - 158), right: anchor.width, bottom: anchor.height },
  ];
  const centralGraphRect: GraphScreenRect = {
    left: anchor.width * 0.25,
    top: anchor.height * 0.2,
    right: anchor.width * 0.75,
    bottom: anchor.height * 0.78,
  };
  const edgeInset = 20;

  const rawCandidates: Array<{
    edge: GraphSelectionPlacementEdge;
    left: number;
    origin: string;
    preference: number;
    top: number;
  }> = [
    {
      left: anchor.x + gap,
      top: anchor.y - cardHeight * 0.5,
      origin: 'left center',
      edge: 'left',
      preference: 0,
    },
    {
      left: anchor.x - cardWidth - gap,
      top: anchor.y - cardHeight * 0.5,
      origin: 'right center',
      edge: 'right',
      preference: 12,
    },
    {
      left: anchor.x - cardWidth * 0.5,
      top: anchor.y + gap,
      origin: 'center top',
      edge: 'top',
      preference: 22,
    },
    {
      left: anchor.x - cardWidth * 0.5,
      top: anchor.y - cardHeight - gap,
      origin: 'center bottom',
      edge: 'bottom',
      preference: 30,
    },
    {
      left: anchor.width - cardWidth - edgeInset,
      top: anchor.y - cardHeight * 0.5,
      origin: 'left center',
      edge: 'left',
      preference: anchor.x < anchor.width * 0.72 ? 18 : 76,
    },
    {
      left: edgeInset,
      top: anchor.y - cardHeight * 0.5,
      origin: 'right center',
      edge: 'right',
      preference: anchor.x > anchor.width * 0.28 ? 26 : 84,
    },
    {
      left: anchor.x - cardWidth * 0.5,
      top: anchor.height - cardHeight - Math.max(dockBottom + 12, 92),
      origin: 'center top',
      edge: 'top',
      preference: 58,
    },
    {
      left: anchor.x - cardWidth * 0.5,
      top: edgeInset,
      origin: 'center bottom',
      edge: 'bottom',
      preference: 72,
    },
  ];

  const candidates = rawCandidates.map((candidate) => {
    const left = clampGraphValue(candidate.left, bounds.left, bounds.right - cardWidth);
    const top = clampGraphValue(candidate.top, bounds.top, bounds.bottom - cardHeight);
    const rect = { left, top, right: left + cardWidth, bottom: top + cardHeight };
    const overlapPenalty = reservedRects.reduce((total, reserved) => total + getRectOverlapArea(rect, reserved), 0) / (expanded ? 8 : 6);
    const focusOverlap = avoidRects.reduce((total, reserved) => total + getRectOverlapArea(rect, reserved), 0);
    const focusOverlapPenalty = focusOverlap / (expanded ? 1.45 : 3.2);
    const centralOverlapPenalty = getRectOverlapArea(rect, centralGraphRect) / (expanded ? 92 : 26);
    const clampPenalty = Math.abs(left - candidate.left) + Math.abs(top - candidate.top);
    const viewportPenalty = graphRectFitsViewport(rect, anchor.width, anchor.height) ? 0 : 800;
    const distancePenalty = Math.hypot(left + cardWidth * 0.5 - anchor.x, top + cardHeight * 0.5 - anchor.y) * (expanded ? 0.006 : 0.02);
    const edgeDistance = Math.min(rect.left, anchor.width - rect.right, rect.top, anchor.height - rect.bottom);
    const edgePenalty = Math.max(0, edgeDistance - (expanded ? 54 : 96)) * (expanded ? 0.38 : 0.72);
    const hysteresisPenalty = preferredEdge && candidate.edge !== preferredEdge ? (expanded ? 28 : 84) : 0;
    const preferredEdgeBonus = preferredEdge === candidate.edge ? (expanded ? -12 : -24) : 0;
    return {
      ...candidate,
      left,
      top,
      rect,
      score: candidate.preference + overlapPenalty + focusOverlapPenalty + centralOverlapPenalty + clampPenalty + viewportPenalty + distancePenalty + edgePenalty + hysteresisPenalty + preferredEdgeBonus,
    };
  }).sort((a, b) => a.score - b.score);

  const best = candidates[0];
  if (!best || (!expanded && best.score > 780)) return docked;

  const endX = best.edge === 'left'
    ? best.left
    : best.edge === 'right'
      ? best.left + cardWidth
      : clampGraphValue(anchor.x, best.left + 18, best.left + cardWidth - 18);
  const endY = best.edge === 'top'
    ? best.top
    : best.edge === 'bottom'
      ? best.top + cardHeight
      : clampGraphValue(anchor.y, best.top + 18, best.top + cardHeight - 18);
  const length = Math.hypot(endX - anchor.x, endY - anchor.y);
  const angle = Math.atan2(endY - anchor.y, endX - anchor.x) * (180 / Math.PI);
  const maxUsefulTetherLength = Math.max(180, Math.min(360, Math.min(anchor.width, anchor.height) * 0.42));
  const tether = length <= maxUsefulTetherLength
    ? {
      angle,
      length,
      left: anchor.x,
      top: anchor.y,
    }
    : undefined;

  return {
    mode: 'anchored',
    edge: best.edge,
    left: best.left,
    top: best.top,
    transformOrigin: best.origin,
    ...(tether ? { tether } : {}),
  };
};

const getSelectionActionRailPlacement = (
  anchor: GraphSelectionAnchor | null,
  railWidth: number,
  railHeight: number,
  cardPlacement: GraphOverlayPlacement,
  expanded: boolean
): GraphOverlayPlacement | null => {
  if (!anchor || expanded || anchor.width < 720 || anchor.height < 520) return null;

  const bounds = {
    left: 12,
    top: 12,
    right: anchor.width - 12,
    bottom: anchor.height - 12,
  };
  const reservedRects: GraphScreenRect[] = [
    { left: 0, top: 0, right: Math.min(anchor.width, 468), bottom: 126 },
    { left: Math.max(0, anchor.width - 288), top: 0, right: anchor.width, bottom: 92 },
    { left: 0, top: Math.max(0, anchor.height - 72), right: Math.min(anchor.width, 540), bottom: anchor.height },
    { left: Math.max(0, anchor.width - 88), top: Math.max(0, anchor.height - 158), right: anchor.width, bottom: anchor.height },
  ];

  if (
    cardPlacement.mode === 'anchored' &&
    Number.isFinite(cardPlacement.left) &&
    Number.isFinite(cardPlacement.top)
  ) {
    const estimatedCardWidth = Math.min(420, Math.max(300, railWidth * 1.75));
    const estimatedCardHeight = Math.min(390, Math.max(150, railHeight * 5.2));
    const cardLeft = Number(cardPlacement.left);
    const cardTop = Number(cardPlacement.top);
    reservedRects.push({
      left: cardLeft - 8,
      top: cardTop - 8,
      right: cardLeft + estimatedCardWidth + 8,
      bottom: cardTop + estimatedCardHeight + 8,
    });
  }

  const gap = anchor.kind === 'edge' ? 24 : 22;
  const rawCandidates = [
    {
      left: anchor.x - railWidth * 0.5,
      top: anchor.y - railHeight - gap,
      origin: 'center bottom',
      preference: anchor.kind === 'edge' ? 2 : 0,
    },
    {
      left: anchor.x + gap,
      top: anchor.y - railHeight * 0.5,
      origin: 'left center',
      preference: 6,
    },
    {
      left: anchor.x - railWidth * 0.5,
      top: anchor.y + gap,
      origin: 'center top',
      preference: 10,
    },
    {
      left: anchor.x - railWidth - gap,
      top: anchor.y - railHeight * 0.5,
      origin: 'right center',
      preference: 14,
    },
  ];

  const candidates = rawCandidates.map((candidate) => {
    const left = clampGraphValue(candidate.left, bounds.left, bounds.right - railWidth);
    const top = clampGraphValue(candidate.top, bounds.top, bounds.bottom - railHeight);
    const rect = { left, top, right: left + railWidth, bottom: top + railHeight };
    const overlapPenalty = reservedRects.reduce((total, reserved) => total + getRectOverlapArea(rect, reserved), 0) / 2.5;
    const clampPenalty = Math.abs(left - candidate.left) + Math.abs(top - candidate.top);
    const viewportPenalty = graphRectFitsViewport(rect, anchor.width, anchor.height) ? 0 : 500;
    return {
      ...candidate,
      left,
      top,
      score: candidate.preference + overlapPenalty + clampPenalty + viewportPenalty,
    };
  }).sort((a, b) => a.score - b.score);

  const best = candidates[0];
  if (!best || best.score > 620) return null;

  return {
    mode: 'anchored',
    left: best.left,
    top: best.top,
    transformOrigin: best.origin,
  };
};

const chooseNodeLabelLayout = (
  screenX: number,
  screenY: number,
  labelWidth: number,
  labelHeight: number,
  width: number,
  height: number,
  reservedRects: GraphScreenRect[],
  preferredAnchor: GraphNodeLabelAnchor,
  pinned: boolean
): { anchor: GraphNodeLabelAnchor; rect: GraphScreenRect } | null => {
  const horizontalOutward: GraphNodeLabelAnchor = screenX > width * 0.5 ? 'left' : 'right';
  const verticalOutward: GraphNodeLabelAnchor = screenY > height * 0.5 ? 'above' : 'below';
  const diagonalOutward: GraphNodeLabelAnchor = screenY > height * 0.5
    ? screenX > width * 0.5 ? 'above-left' : 'above-right'
    : screenX > width * 0.5 ? 'below-left' : 'below-right';
  const diagonalAlternateA: GraphNodeLabelAnchor = screenY > height * 0.5
    ? screenX > width * 0.5 ? 'below-left' : 'below-right'
    : screenX > width * 0.5 ? 'above-left' : 'above-right';
  const diagonalAlternateB: GraphNodeLabelAnchor = screenY > height * 0.5
    ? screenX > width * 0.5 ? 'above-right' : 'above-left'
    : screenX > width * 0.5 ? 'below-right' : 'below-left';
  const anchorCandidates: GraphNodeLabelAnchor[] = [
    preferredAnchor,
    preferredAnchor === 'right' ? 'left' : 'right',
    diagonalOutward,
    horizontalOutward,
    verticalOutward,
    diagonalAlternateA,
    diagonalAlternateB,
    verticalOutward === 'above' ? 'below' : 'above',
  ];
  const anchors = anchorCandidates.filter((anchor, index, list) => list.indexOf(anchor) === index);
  const gap = pinned ? 30 : 18;
  const padding = pinned ? 13 : 10;

  const candidates = anchors.map((anchor, index) => {
    const rect = getLabelRectForAnchor(screenX, screenY, labelWidth, labelHeight, anchor, gap);
    const padded = padGraphRect(rect, padding);
    const collisions = reservedRects.reduce((count, item) => count + (graphRectsOverlap(padded, item) ? 1 : 0), 0);
    const overflow =
      Math.max(0, 8 - rect.left) +
      Math.max(0, 8 - rect.top) +
      Math.max(0, rect.right - (width - 8)) +
      Math.max(0, rect.bottom - (height - 8));
    return { anchor, rect, padded, collisions, overflow, preference: index };
  }).sort((a, b) =>
    Number(a.collisions > 0) - Number(b.collisions > 0) ||
    a.collisions - b.collisions ||
    a.overflow - b.overflow ||
    a.preference - b.preference
  );

  const fit = candidates.find((candidate) =>
    candidate.collisions === 0 && graphRectFitsViewport(candidate.rect, width, height)
  );
  if (fit) return { anchor: fit.anchor, rect: fit.padded };
  if (!pinned) return null;
  const fallback = candidates.find((candidate) => graphRectFitsViewport(candidate.rect, width, height)) || candidates[0];
  return fallback ? { anchor: fallback.anchor, rect: fallback.padded } : null;
};

const drawRoundedRect = (
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number
) => {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
};

const NODE_GEOMETRY_CACHE = new Map<string, THREE.BufferGeometry>();

const getNodeGeometry = (
  type: string | undefined,
  size: number,
  label?: string | null,
  detail: 'full' | 'lite' = 'full'
) => {
  const visualType = getEntityVisualType(type, label);
  const sizeBucket = Math.round(size * 4) / 4;
  const cacheKey = [visualType, sizeBucket, detail].join(':');
  const cached = NODE_GEOMETRY_CACHE.get(cacheKey);
  if (cached) return cached;

  let geometry: THREE.BufferGeometry;
  switch (getEntityVisualType(type, label)) {
    case 'person':
      geometry = new THREE.SphereGeometry(size * 0.96, detail === 'full' ? 22 : 8, detail === 'full' ? 14 : 6);
      break;
    case 'organization':
      geometry = new THREE.BoxGeometry(size * 1.45, size * 1.45, size * 1.45);
      break;
    case 'location':
      geometry = new THREE.OctahedronGeometry(size * 1.05, 0);
      break;
    case 'event':
    case 'date':
      geometry = new THREE.ConeGeometry(size * 0.95, size * 1.75, 3);
      break;
    case 'technology':
      geometry = new THREE.IcosahedronGeometry(size * 1.02, detail === 'full' ? 1 : 0);
      break;
    case 'model':
      geometry = new THREE.OctahedronGeometry(size * 1.14, detail === 'full' ? 1 : 0);
      break;
    case 'product':
      geometry = new THREE.BoxGeometry(size * 1.58, size * 1.12, size * 0.82);
      break;
    case 'concept':
      geometry = new THREE.TetrahedronGeometry(size * 1.2, detail === 'full' ? 1 : 0);
      break;
    case 'api':
      geometry = new THREE.TorusGeometry(size * 0.78, Math.max(0.08, size * 0.18), detail === 'full' ? 12 : 6, detail === 'full' ? 32 : 12);
      break;
    case 'config':
      geometry = new THREE.BoxGeometry(size * 1.54, size * 0.95, size * 0.56);
      break;
    case 'memory':
      geometry = new THREE.CylinderGeometry(size * 0.74, size * 0.88, size * 1.42, detail === 'full' ? 18 : 8);
      break;
    case 'security':
      geometry = new THREE.ConeGeometry(size * 0.98, size * 1.58, 4);
      break;
    case 'code':
      geometry = new THREE.BoxGeometry(size * 1.62, size * 1.1, size * 0.46);
      break;
    case 'data':
      geometry = new THREE.CylinderGeometry(size * 0.88, size * 0.88, size * 1.18, detail === 'full' ? 28 : 10);
      break;
    case 'prompt':
      geometry = new THREE.BoxGeometry(size * 1.08, size * 1.44, size * 0.34);
      break;
    case 'vector':
      geometry = new THREE.TetrahedronGeometry(size * 1.14, detail === 'full' ? 1 : 0);
      break;
    case 'evaluation':
      geometry = new THREE.DodecahedronGeometry(size * 1.02, detail === 'full' ? 1 : 0);
      break;
    case 'runtime':
      geometry = new THREE.ConeGeometry(size * 0.9, size * 1.72, 5);
      break;
    case 'service':
      geometry = new THREE.SphereGeometry(size * 1.02, detail === 'full' ? 20 : 8, detail === 'full' ? 12 : 6);
      break;
    case 'hardware':
      geometry = new THREE.BoxGeometry(size * 1.38, size * 1.38, size * 0.82);
      break;
    case 'metric':
      geometry = new THREE.ConeGeometry(size * 0.98, size * 1.4, 5);
      break;
    case 'process':
      geometry = new THREE.DodecahedronGeometry(size * 0.96, 0);
      break;
    default:
      geometry = new THREE.SphereGeometry(size, detail === 'full' ? 24 : 8, detail === 'full' ? 16 : 6);
  }

  NODE_GEOMETRY_CACHE.set(cacheKey, geometry);
  return geometry;
};

const getNodeLabelWorldPosition = (
  anchor: GraphNodeLabelAnchor,
  labelScale: THREE.Vector3,
  size: number,
  highlighted: boolean,
  focused: boolean,
  iconVisible: boolean,
  symbolWorldScale: number,
  labelStyle: GraphNodeLabelStyle
) => {
  const offset = getNodeLabelOffset(
    anchor,
    labelScale,
    size,
    highlighted,
    focused,
    iconVisible,
    symbolWorldScale,
    labelStyle
  );
  return new THREE.Vector3(offset.x, offset.y, offset.z);
};

const getGraphSpriteTextureDpr = (minimumDpr = 2) => {
  const deviceDpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
  return Math.min(3, Math.max(minimumDpr, deviceDpr));
};

const createGraphCanvasTexture = (canvas: HTMLCanvasElement) => {
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.generateMipmaps = false;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  return texture;
};

const createNodeIconSprite = (
  node: RenderGraphNode,
  colors: { main: string; light: string; glow: string },
  size: number,
  highlighted: boolean,
  focused: boolean,
  muted: boolean,
  worldScale = 1
) => {
  const canvas = document.createElement('canvas');
  const dpr = getGraphSpriteTextureDpr(2);
  const canvasSize = 128;
  canvas.width = canvasSize * dpr;
  canvas.height = canvasSize * dpr;

  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, canvasSize, canvasSize);

  const center = canvasSize / 2;
  const radius = highlighted ? 44 : focused ? 43 : 42;
  const depthOffset = highlighted ? 7.6 : focused ? 6.6 : 5.8;
  ctx.globalAlpha = muted ? 0.68 : highlighted || focused ? 1 : 0.99;

  ctx.save();
  ctx.translate(depthOffset * 0.5, depthOffset * 0.62);
  ctx.shadowColor = alpha(colors.glow, muted ? 0.16 : highlighted ? 0.38 : 0.28);
  ctx.shadowBlur = highlighted ? 20 : focused ? 16 : 12;
  ctx.fillStyle = alpha('#020617', muted ? 0.46 : 0.58);
  drawEntityShape(ctx, center, center, radius, node.type, node.label);
  ctx.fill();
  ctx.restore();

  ctx.save();
  ctx.shadowColor = colors.glow;
  ctx.shadowBlur = highlighted ? 25 : focused ? 20 : 15;
  const fill = ctx.createLinearGradient(center - radius, center - radius, center + radius, center + radius);
  fill.addColorStop(0, muted ? alpha(colors.light, 0.52) : alpha(colors.light, highlighted ? 0.98 : 0.9));
  fill.addColorStop(0.48, alpha(colors.main, muted ? 0.62 : highlighted ? 0.98 : 0.94));
  fill.addColorStop(1, alpha('#020617', muted ? 0.42 : 0.28));
  ctx.fillStyle = fill;
  drawEntityShape(ctx, center, center, radius, node.type, node.label);
  ctx.fill();
  ctx.clip();
  ctx.shadowBlur = 0;
  ctx.fillStyle = alpha('#ffffff', muted ? 0.08 : highlighted ? 0.24 : focused ? 0.2 : 0.15);
  ctx.beginPath();
  ctx.ellipse(center - radius * 0.24, center - radius * 0.36, radius * 0.68, radius * 0.28, -0.48, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = alpha('#020617', muted ? 0.28 : 0.2);
  ctx.beginPath();
  ctx.ellipse(center + radius * 0.32, center + radius * 0.38, radius * 0.62, radius * 0.26, -0.4, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  ctx.shadowBlur = 0;
  ctx.strokeStyle = highlighted ? '#ffffff' : alpha('#ffffff', focused ? 0.9 : muted ? 0.58 : 0.82);
  ctx.lineWidth = highlighted ? 4.35 : focused ? 3.55 : 3.1;
  drawEntityShape(ctx, center, center, radius, node.type, node.label);
  ctx.stroke();
  drawEntityGlyph(ctx, center, center, radius, node.type, node.label);
  ctx.globalAlpha = 1;

  const texture = createGraphCanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    opacity: muted ? 0.78 : highlighted || focused ? 1 : 0.995,
  });
  const sprite = new THREE.Sprite(material);
  sprite.userData.kgIconSprite = true;
  sprite.userData.kgEntityVisualType = getEntityVisualType(node.type, node.label);
  sprite.renderOrder = highlighted ? 13 : focused ? 12 : 10;
  const spriteSize = getNodeIconSpriteWorldSize(size, highlighted, focused, worldScale);
  sprite.scale.set(spriteSize, spriteSize, 1);
  sprite.position.set(0, 0, size * (highlighted ? 1.78 : focused ? 1.66 : 1.56) * Math.max(0.94, Math.min(1.08, worldScale)));
  return sprite;
};

const createNodeRelationHaloSprite = (
  relationTypes: RelationTypeCount[],
  size: number,
  highlighted: boolean,
  focused: boolean,
  muted: boolean
) => {
  const visibleTypes = relationTypes
    .filter((item) => item.type && item.count > 0)
    .slice(0, highlighted ? 6 : 4);
  if (visibleTypes.length === 0) return null;

  const richHalo = highlighted && !muted;
  const dpr = getGraphSpriteTextureDpr(2);
  const canvas = document.createElement('canvas');
  const canvasSize = richHalo ? 220 : 148;
  canvas.width = canvasSize * dpr;
  canvas.height = canvasSize * dpr;

  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, canvasSize, canvasSize);

  const center = canvasSize / 2;
  const total = Math.max(1, visibleTypes.reduce((sum, item) => sum + item.count, 0));
  const radius = richHalo ? 76 : highlighted ? 57 : focused ? 54 : 51;
  const lineWidth = richHalo ? 9 : highlighted ? 8 : 6.5;
  let start = -Math.PI / 2;

  ctx.save();
  ctx.lineCap = 'round';
  ctx.shadowColor = richHalo ? 'rgba(255, 255, 255, 0.22)' : highlighted ? 'rgba(255, 255, 255, 0.18)' : 'rgba(125, 211, 252, 0.16)';
  ctx.shadowBlur = richHalo ? 20 : highlighted ? 16 : 11;
  ctx.strokeStyle = muted ? 'rgba(148, 163, 184, 0.2)' : 'rgba(226, 232, 240, 0.16)';
  ctx.lineWidth = lineWidth + 2;
  ctx.beginPath();
  ctx.arc(center, center, radius, 0, Math.PI * 2);
  ctx.stroke();

  visibleTypes.forEach((item, index) => {
    const share = Math.max(0.08, item.count / total);
    const end = index === visibleTypes.length - 1 ? Math.PI * 1.5 : start + share * Math.PI * 2;
    const color = getRelationshipColor(item.type);
    ctx.strokeStyle = muted ? alpha(color, 0.28) : alpha(color, highlighted ? 0.88 : 0.62);
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.arc(center, center, radius, start + 0.025, end - 0.025);
    ctx.stroke();

    if (highlighted || focused) {
      const angle = start + (end - start) / 2;
      const glyphX = center + Math.cos(angle) * (radius + lineWidth * (richHalo ? 1.1 : 0.68));
      const glyphY = center + Math.sin(angle) * (radius + lineWidth * (richHalo ? 1.1 : 0.68));
      const glyphRadius = richHalo ? 9.6 : 7.4;
      ctx.fillStyle = muted ? alpha(color, 0.18) : alpha(color, 0.26);
      ctx.beginPath();
      ctx.arc(glyphX, glyphY, glyphRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = muted ? 'rgba(226, 232, 240, 0.42)' : 'rgba(248, 250, 252, 0.86)';
      ctx.lineWidth = richHalo ? 1.15 : 1;
      ctx.stroke();
      ctx.fillStyle = muted ? 'rgba(226, 232, 240, 0.42)' : 'rgba(248, 250, 252, 0.9)';
      drawRelationGlyph(ctx, glyphX, glyphY, richHalo ? 5.4 : 4.2, item.type);

      if (richHalo && index < 4) {
        const label = truncateGraphLabel(getTypeLabel(item.type), 16);
        const countLabel = item.count >= 1000 ? `${Math.round(item.count / 100) / 10}k` : String(item.count);
        ctx.font = '780 8.8px Inter, Arial, sans-serif';
        const labelWidth = Math.min(
          90,
          Math.max(58, Math.ceil(ctx.measureText(label).width + ctx.measureText(countLabel).width + 24))
        );
        const labelHeight = 20;
        const rawLabelX = center + Math.cos(angle) * (radius + 31);
        const rawLabelY = center + Math.sin(angle) * (radius + 31);
        const labelX = clampGraphValue(
          Math.cos(angle) >= 0 ? rawLabelX : rawLabelX - labelWidth,
          7,
          canvasSize - labelWidth - 7
        );
        const labelY = clampGraphValue(rawLabelY - labelHeight / 2, 7, canvasSize - labelHeight - 7);
        const labelAnchorX = Math.cos(angle) >= 0 ? labelX : labelX + labelWidth;
        const labelAnchorY = labelY + labelHeight / 2;

        ctx.save();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = alpha(color, 0.42);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(glyphX + Math.cos(angle) * 5.5, glyphY + Math.sin(angle) * 5.5);
        ctx.lineTo(labelAnchorX, labelAnchorY);
        ctx.stroke();

        ctx.fillStyle = 'rgba(4, 8, 20, 0.82)';
        drawRoundedRect(ctx, labelX, labelY, labelWidth, labelHeight, 7);
        ctx.fill();
        ctx.strokeStyle = alpha(color, 0.38);
        ctx.lineWidth = 0.9;
        ctx.stroke();
        ctx.fillStyle = 'rgba(248, 250, 252, 0.93)';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, labelX + 7, labelY + labelHeight / 2 + 0.2);
        ctx.fillStyle = alpha(color, 0.94);
        ctx.font = '850 8px Inter, Arial, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(countLabel, labelX + labelWidth - 7, labelY + labelHeight / 2 + 0.2);
        ctx.textAlign = 'left';
        ctx.restore();
      }
    }

    start = end;
  });
  ctx.restore();

  const texture = createGraphCanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    opacity: muted ? 0.34 : highlighted ? 0.96 : 0.7,
  });
  const sprite = new THREE.Sprite(material);
  sprite.userData.kgRelationHaloSprite = true;
  sprite.renderOrder = highlighted ? 12 : focused ? 9 : 8;
  const spriteSize = Math.max(
    richHalo ? 24 : highlighted ? 17.5 : focused ? 16 : 14.2,
    size * (richHalo ? 4.4 : highlighted ? 3.42 : focused ? 3.2 : 2.92)
  );
  sprite.scale.set(spriteSize, spriteSize, 1);
  sprite.position.set(0, 0, size * (richHalo ? 1.72 : highlighted ? 1.5 : focused ? 1.42 : 1.32));
  return sprite;
};

const createNodeLabelSprite = (
  node: RenderGraphNode,
  colors: { main: string; light: string; glow: string },
  highlighted: boolean,
  muted: boolean,
  style: GraphNodeLabelStyle,
  opacity = 1,
  worldScale = 1,
  relationHint?: GraphFocusHaloLabelHint,
  relationSummary?: { links: number; relations: number }
) => {
  const typeLabel = getEntityDisplayTypeLabel(node.type, node.label);
  const relationCount = Math.max(0, Number(relationSummary?.relations || 0));
  const richDetailLabel = [
    typeLabel,
    relationCount > 0 ? `${relationCount.toLocaleString()} relation${relationCount === 1 ? '' : 's'}` : '',
  ].filter(Boolean).join(' / ');
  const relationLabel = relationHint ? `${getTypeLabel(relationHint.type)} ${relationHint.count}` : '';
  const dpr = getGraphSpriteTextureDpr(2);
  const canvas = document.createElement('canvas');
  const measurement = document.createElement('canvas').getContext('2d');
  if (!measurement) return null;

    const rich = style === 'rich';
    const annotation = style === 'annotation';
    const anchor = style === 'anchor';
    const neighbor = style === 'neighbor';
    const compact = style === 'compact';
      const labelFont = rich
        ? highlighted ? '840 18.4px Inter, Arial, sans-serif' : '810 16.4px Inter, Arial, sans-serif'
        : annotation
          ? highlighted ? '840 14.4px Inter, Arial, sans-serif' : '790 13.5px Inter, Arial, sans-serif'
          : anchor
            ? highlighted ? '860 15.4px Inter, Arial, sans-serif' : '830 14.3px Inter, Arial, sans-serif'
          : neighbor
            ? highlighted ? '850 14px Inter, Arial, sans-serif' : '820 13px Inter, Arial, sans-serif'
          : highlighted ? '860 14.9px Inter, Arial, sans-serif' : '830 13.9px Inter, Arial, sans-serif';
    const typeFont = rich ? '720 9px Inter, Arial, sans-serif' : annotation ? '730 8px Inter, Arial, sans-serif' : anchor ? '770 8.3px Inter, Arial, sans-serif' : neighbor ? '760 8px Inter, Arial, sans-serif' : '750 8.4px Inter, Arial, sans-serif';
  measurement.font = labelFont;
  const displayLabel = getGraphNodeDisplayLabel(node.label || node.id, node.type);
  const maxLabelWidth = rich ? 222 : annotation ? 152 : anchor ? 184 : neighbor ? 164 : 176;
  const labelLines = wrapMeasuredGraphLabel(
    measurement,
    displayLabel,
    maxLabelWidth,
    rich ? 52 : annotation ? 34 : anchor ? 30 : neighbor ? 34 : 30,
    rich || annotation || anchor || neighbor || (compact && displayLabel.length > 18) ? 2 : 1
  );
  const labelWidth = Math.max(...labelLines.map((line) => measurement.measureText(line).width));
  measurement.font = typeFont;
  const typeWidth = rich ? measurement.measureText(richDetailLabel).width : neighbor ? measurement.measureText(relationLabel).width : 0;
    const textInset = rich ? 52 : annotation ? 17 : anchor ? 22 : neighbor ? 33 : 22;
    const width = rich
      ? Math.min(326, Math.max(150, Math.ceil(Math.max(labelWidth, typeWidth) + 74)))
      : annotation
        ? Math.min(222, Math.max(92, Math.ceil(labelWidth + textInset + 16)))
        : anchor
          ? Math.min(254, Math.max(108, Math.ceil(labelWidth + textInset + 20)))
        : neighbor
          ? Math.min(236, Math.max(116, Math.ceil(Math.max(labelWidth, typeWidth + 16) + textInset + 16)))
      : Math.min(238, Math.max(106, Math.ceil(labelWidth + textInset + 18)));
    const labelLineHeight = rich ? 17 : annotation ? 13.4 : anchor ? 14.2 : neighbor ? 13.2 : 13.2;
    const multilineLabel = labelLines.length > 1;
    const height = rich
      ? multilineLabel ? 66 : 52
      : annotation
        ? multilineLabel ? 42 : 28
        : anchor
          ? multilineLabel ? 44 : 30
          : neighbor
            ? multilineLabel ? 50 : 38
            : multilineLabel ? 46 : 34;
  canvas.width = width * dpr;
  canvas.height = height * dpr;

  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  ctx.globalAlpha = muted ? 0.36 : 1;
    ctx.shadowColor = highlighted ? alpha(colors.main, 0.44) : 'rgba(0, 0, 0, 0.62)';
      ctx.shadowBlur = highlighted ? 15 : rich ? 10 : annotation ? 6 : neighbor ? 8 : 7;
      ctx.fillStyle = highlighted
        ? 'rgba(8, 13, 28, 0.92)'
        : rich
          ? 'rgba(8, 12, 26, 0.82)'
          : annotation
            ? 'rgba(5, 9, 20, 0.86)'
            : anchor
              ? 'rgba(5, 9, 20, 0.91)'
            : neighbor
              ? 'rgba(5, 9, 20, 0.84)'
             : 'rgba(5, 9, 20, 0.91)';
    drawRoundedRect(ctx, 2, 4, width - 4, height - 8, rich ? 10 : annotation ? 7 : anchor ? 8 : neighbor ? 8 : 7);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.strokeStyle = alpha(neighbor && relationHint ? relationHint.color : colors.main, highlighted ? 0.88 : rich ? 0.5 : annotation ? 0.34 : anchor ? 0.56 : neighbor ? 0.48 : 0.26);
  ctx.lineWidth = highlighted ? 2 : rich ? 1 : annotation ? 0.9 : anchor ? 1.1 : neighbor ? 1.05 : 0.8;
  ctx.stroke();

    if (rich) {
      ctx.fillStyle = alpha(colors.main, highlighted ? 0.98 : 0.84);
      drawEntityShape(ctx, 26, height / 2, 12, node.type, node.label);
      ctx.fill();
      ctx.strokeStyle = alpha('#ffffff', highlighted ? 0.9 : 0.68);
      ctx.lineWidth = 1.5;
      ctx.stroke();
      drawEntityGlyph(ctx, 26, height / 2, 12, node.type, node.label);
    } else if (neighbor && relationHint) {
      const iconX = 16;
      const iconY = height / 2;
      ctx.fillStyle = alpha(colors.main, highlighted ? 0.98 : 0.76);
      drawEntityShape(ctx, iconX, iconY, 7.4, node.type, node.label);
      ctx.fill();
      ctx.strokeStyle = alpha(relationHint.color, highlighted ? 0.85 : 0.54);
      ctx.lineWidth = 1.1;
      ctx.stroke();
      drawEntityGlyph(ctx, iconX, iconY, 7.4, node.type, node.label);
    } else {
      const iconX = annotation ? 8.2 : anchor ? 10 : 11.2;
      const iconRadius = annotation ? 4.7 : anchor ? 5.5 : 6.8;
      ctx.fillStyle = alpha(colors.main, highlighted ? 0.98 : annotation ? 0.82 : anchor ? 0.92 : 0.86);
      drawEntityShape(ctx, iconX, height / 2, iconRadius, node.type, node.label);
      ctx.fill();
      ctx.strokeStyle = alpha('#ffffff', highlighted ? 0.82 : anchor ? 0.56 : 0.46);
      ctx.lineWidth = highlighted ? 1.2 : 0.95;
      ctx.stroke();
      drawEntityGlyph(ctx, iconX, height / 2, iconRadius, node.type, node.label);
    }

      ctx.fillStyle = muted ? 'rgba(255, 255, 255, 0.68)' : annotation ? 'rgba(255, 255, 255, 0.9)' : anchor ? 'rgba(255, 255, 255, 0.96)' : '#ffffff';
    ctx.font = labelFont;
    ctx.textBaseline = 'middle';
    const labelBlockCenterY = rich
      ? multilineLabel ? 24 : 19
      : neighbor
        ? multilineLabel ? 19 : 15
        : height / 2 + 0.2;
    const labelStartY = labelBlockCenterY - ((labelLines.length - 1) * labelLineHeight) / 2;
    labelLines.forEach((line, index) => {
      ctx.fillText(line, textInset, labelStartY + index * labelLineHeight);
    });
    if (rich) {
      ctx.fillStyle = muted ? 'rgba(206, 214, 230, 0.46)' : alpha(colors.light, 0.82);
      ctx.font = typeFont;
      ctx.fillText(richDetailLabel, textInset + 1, multilineLabel ? 52 : 36);
  } else if (neighbor && relationHint) {
      const relationY = multilineLabel ? 39 : 28.5;
      ctx.fillStyle = alpha(relationHint.color, muted ? 0.46 : 0.88);
      drawRelationGlyph(ctx, textInset + 3, relationY, 4.6, relationHint.type);
      ctx.font = typeFont;
      ctx.fillStyle = muted ? 'rgba(206, 214, 230, 0.46)' : alpha('#dbeafe', 0.72);
      ctx.fillText(relationLabel, textInset + 14, relationY + 0.2);
  }
  ctx.globalAlpha = 1;

  const texture = createGraphCanvasTexture(canvas);
    const baseOpacity = muted ? 0.64 : rich ? 0.99 : annotation ? 0.98 : anchor ? 1 : neighbor ? 0.97 : 1;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    opacity: baseOpacity * Math.max(0, Math.min(1, opacity)),
  });
  material.userData.kgBaseOpacity = baseOpacity;
  const sprite = new THREE.Sprite(material);
  sprite.userData.kgLabelSprite = true;
  sprite.renderOrder = highlighted ? 14 : rich ? 11 : annotation ? 10 : anchor ? 12 : neighbor ? 11 : 12;
      const scale = rich ? 5.16 : annotation ? 4.9 : anchor ? 4.54 : neighbor ? 4.9 : 4.34;
  sprite.scale.set((width / scale) * worldScale, (height / scale) * worldScale, 1);
  return sprite;
};

  const createLinkLabelSprite = (
    link: RenderGraphEdge,
    highlighted: boolean,
    muted: boolean,
    options: LinkLabelSpriteOptions = {}
  ) => {
    const label = getRelationDisplayLabel(link);
    const color = getRelationshipColor(link.type || link.label);
    const relationType = link.type || link.label;
    const relationTypes = (options.relationTypes?.length ? options.relationTypes : getLinkRelationTypeCounts(link, 4)).slice(0, 4);
    const relationCount = Math.max(1, Number(link.relationCount || 1));
    const sourceLabel = truncateGraphLabel(options.sourceLabel || '', 16);
    const targetLabel = truncateGraphLabel(options.targetLabel || '', 16);
    const rich = Boolean(options.rich && sourceLabel && targetLabel);
    const dpr = getGraphSpriteTextureDpr(2);
    const canvas = document.createElement('canvas');
    const measurement = document.createElement('canvas').getContext('2d');
    if (!measurement) return null;

    const labelFont = rich
      ? '850 14px Inter, Arial, sans-serif'
      : highlighted ? '850 15px Inter, Arial, sans-serif' : '760 13px Inter, Arial, sans-serif';
    const routeFont = rich ? '760 10px Inter, Arial, sans-serif' : '760 11px Inter, Arial, sans-serif';
    const countFont = '820 10px Inter, Arial, sans-serif';
    measurement.font = labelFont;
    const labelWidth = measurement.measureText(label).width;
    measurement.font = routeFont;
    const routeLabel = rich ? `${sourceLabel} -> ${targetLabel}` : '';
    const routeWidth = rich ? measurement.measureText(routeLabel).width : 0;
    measurement.font = countFont;
    const countLabel = relationCount > 1 ? `${relationCount.toLocaleString()} rel` : '1 rel';
    const countWidth = rich ? Math.max(38, measurement.measureText(countLabel).width + 14) : 0;
    const iconInset = rich ? 40 : 38;
    const mixInset = rich ? Math.min(62, relationTypes.length * 14 + (relationTypes.length > 3 ? 14 : 0)) : 0;
    const width = rich
      ? Math.min(292, Math.max(184, Math.ceil(Math.max(labelWidth + countWidth + 14, routeWidth) + iconInset + mixInset + 16)))
      : Math.min(240, Math.max(86, Math.ceil(labelWidth + 44)));
    const height = rich ? 48 : 30;
    canvas.width = width * dpr;
    canvas.height = height * dpr;

  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  ctx.globalAlpha = muted ? 0.48 : 1;
  ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
    ctx.shadowBlur = highlighted ? 16 : 9;
    ctx.fillStyle = highlighted ? 'rgba(9, 14, 30, 0.92)' : 'rgba(8, 12, 26, 0.74)';
    drawRoundedRect(ctx, 1, 3, width - 2, height - 6, rich ? 13 : 10);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = alpha(color, highlighted ? 0.84 : 0.48);
    ctx.lineWidth = highlighted ? 1.8 : 1;
    ctx.stroke();

    const iconX = rich ? 21 : 16;
    const iconY = height / 2;
    const iconRadius = rich ? 9 : 8;
    ctx.fillStyle = alpha(color, highlighted ? 0.24 : 0.16);
    ctx.beginPath();
    ctx.arc(iconX, iconY, iconRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = alpha(color, highlighted ? 0.92 : 0.66);
    ctx.lineWidth = rich ? 1.3 : 1.1;
    ctx.beginPath();
    ctx.arc(iconX, iconY, iconRadius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.strokeStyle = muted ? 'rgba(230, 235, 248, 0.5)' : '#f8fbff';
    ctx.fillStyle = muted ? 'rgba(230, 235, 248, 0.5)' : '#f8fbff';
    drawRelationGlyph(ctx, iconX, iconY, rich ? 6 : 5.6, relationType);

    const textX = rich ? 39 : 38;
    ctx.fillStyle = muted ? 'rgba(230, 235, 248, 0.62)' : '#f8fbff';
    ctx.font = labelFont;
    ctx.textBaseline = 'middle';
    ctx.fillText(label, textX, rich ? 17 : height / 2 + 0.4);

    if (rich) {
      const countX = Math.min(width - mixInset - countWidth - 10, textX + labelWidth + 8);
      ctx.fillStyle = alpha(color, 0.11);
      drawRoundedRect(ctx, countX, 8, countWidth, 16, 8);
      ctx.fill();
      ctx.strokeStyle = alpha(color, 0.34);
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = muted ? 'rgba(230, 235, 248, 0.5)' : alpha(color, 0.96);
      ctx.font = countFont;
      ctx.textBaseline = 'middle';
      ctx.fillText(countLabel, countX + 7, 16.5);

      ctx.fillStyle = muted ? 'rgba(206, 214, 230, 0.48)' : 'rgba(206, 214, 230, 0.82)';
      ctx.font = routeFont;
      ctx.fillText(routeLabel, textX, 32.8);

      const visibleMix = relationTypes.slice(0, 3);
      const mixStartX = Math.max(textX + routeWidth + 10, width - (visibleMix.length * 14) - (relationTypes.length > 3 ? 18 : 9));
      visibleMix.forEach((item, index) => {
        const mixColor = getRelationshipColor(item.type);
        const x = mixStartX + index * 14;
        ctx.fillStyle = alpha(mixColor, 0.22);
        ctx.beginPath();
        ctx.arc(x, 33, 5.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = alpha(mixColor, 0.74);
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.strokeStyle = muted ? 'rgba(230, 235, 248, 0.48)' : '#f8fbff';
        ctx.fillStyle = muted ? 'rgba(230, 235, 248, 0.48)' : '#f8fbff';
        drawRelationGlyph(ctx, x, 33, 3.5, item.type);
      });
      if (relationTypes.length > visibleMix.length) {
        ctx.fillStyle = muted ? 'rgba(206, 214, 230, 0.48)' : 'rgba(206, 214, 230, 0.82)';
        ctx.font = '820 9px Inter, Arial, sans-serif';
        ctx.fillText(`+${relationTypes.length - visibleMix.length}`, mixStartX + visibleMix.length * 14 - 1, 33.5);
      }
    }
    ctx.globalAlpha = 1;

    const texture = createGraphCanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    opacity: muted ? 0.5 : 1,
    });
    const sprite = new THREE.Sprite(material);
    sprite.userData.linkLabel = true;
    sprite.userData.richLinkLabel = rich;
    sprite.userData.highlightedLinkLabel = highlighted;
    sprite.renderOrder = highlighted ? 11 : 8;
    sprite.scale.set(width / (rich ? 12.8 : 11.4), height / (rich ? 12.8 : 11.4), 1);
    return sprite;
  };

const createRelationMarkerSprite = (
  link: RenderGraphEdge,
  highlighted: boolean,
  muted: boolean
) => {
  const relationTypes = getLinkRelationTypeCounts(link, 3);
  const relationType = relationTypes[0]?.type || link.type || link.label;
  const color = getRelationshipColor(relationType);
  const relationCount = Math.max(1, Number(link.relationCount || 1));
  const dpr = getGraphSpriteTextureDpr(2);
  const label = truncateGraphLabel(getTypeLabel(relationType), highlighted ? 22 : 19);
  const labelFont = `${highlighted ? 880 : 850} ${highlighted ? 13.8 : 12.7}px Inter, Arial, sans-serif`;
  const countFont = `${highlighted ? 860 : 830} 10.6px Inter, Arial, sans-serif`;
  const measurement = document.createElement('canvas').getContext('2d');
  if (!measurement) return null;
  measurement.font = labelFont;
  const labelWidth = measurement.measureText(label).width;
  measurement.font = countFont;
  const countLabel = relationCount > 1 ? (relationCount > 99 ? '99+' : relationCount.toLocaleString()) : '';
  const countWidth = countLabel ? Math.max(21, measurement.measureText(countLabel).width + 11) : 0;
  const mixWidth = relationTypes.length > 1 ? Math.min(42, relationTypes.length * 14 + 2) : 0;
  const width = Math.min(highlighted ? 184 : 168, Math.max(highlighted ? 112 : 102, Math.ceil(labelWidth + countWidth + mixWidth + 72)));
  const height = 68;
  const canvas = document.createElement('canvas');
  canvas.width = width * dpr;
  canvas.height = height * dpr;

  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  ctx.globalAlpha = muted ? 0.56 : 1;

  const centerX = 27;
  const centerY = height / 2;
  const radius = Math.min(19.5, (highlighted ? 15.1 : 13.5) + Math.log2(relationCount + 1) * 0.76);
  ctx.shadowColor = alpha(color, highlighted ? 0.54 : 0.34);
  ctx.shadowBlur = highlighted ? 15 : 10;
  ctx.fillStyle = highlighted ? 'rgba(7, 11, 24, 0.95)' : 'rgba(7, 11, 24, 0.89)';
  drawRoundedRect(ctx, 3, 9, width - 6, height - 18, 16);
  ctx.fill();

  ctx.shadowBlur = 0;
  ctx.strokeStyle = alpha(color, highlighted ? 0.78 : 0.56);
  ctx.lineWidth = highlighted ? 1.55 : 1.15;
  ctx.stroke();

  ctx.fillStyle = alpha(color, highlighted ? 0.96 : 0.84);
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = highlighted ? alpha('#ffffff', 0.92) : alpha('#ffffff', 0.76);
  ctx.lineWidth = highlighted ? 2.25 : 1.72;
  ctx.stroke();

  ctx.strokeStyle = muted ? 'rgba(226, 232, 240, 0.5)' : '#f8fbff';
  ctx.fillStyle = muted ? 'rgba(226, 232, 240, 0.5)' : '#f8fbff';
  drawRelationGlyph(ctx, centerX, centerY, radius * 0.58, relationType);

  const labelX = 49;
  ctx.font = labelFont;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = muted ? 'rgba(226, 232, 240, 0.64)' : alpha('#f8fbff', highlighted ? 0.99 : 0.94);
  ctx.fillText(label, labelX + labelWidth / 2, centerY + 0.3);

  const countLeft = countLabel ? width - countWidth - 10 : width - 10;
  if (countLabel) {
    const badgeX = countLeft + countWidth / 2;
    ctx.fillStyle = alpha(color, highlighted ? 0.21 : 0.15);
    drawRoundedRect(ctx, badgeX - countWidth / 2, centerY - 8.5, countWidth, 17, 8.5);
    ctx.fill();
    ctx.strokeStyle = alpha(color, highlighted ? 0.62 : 0.44);
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.font = countFont;
    ctx.fillStyle = muted ? 'rgba(226, 232, 240, 0.56)' : alpha(color, 0.96);
    ctx.fillText(countLabel, badgeX, centerY + 0.4);
  }

  if (relationTypes.length > 1) {
    const visibleMix = relationTypes.slice(1, 3);
    const mixEndX = countLabel ? countLeft - 10 : width - 16;
    const mixStartX = Math.max(labelX + labelWidth + 12, mixEndX - (visibleMix.length - 1) * 14);
    visibleMix.forEach((item, index) => {
      const mixColor = getRelationshipColor(item.type);
      const x = mixStartX + index * 14;
      if (x > mixEndX || x > width - 13) return;
      ctx.fillStyle = alpha(mixColor, highlighted ? 0.28 : 0.22);
      ctx.beginPath();
      ctx.arc(x, centerY, 5.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = alpha(mixColor, highlighted ? 0.76 : 0.62);
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.strokeStyle = muted ? 'rgba(226, 232, 240, 0.52)' : '#f8fbff';
      ctx.fillStyle = muted ? 'rgba(226, 232, 240, 0.52)' : '#f8fbff';
      drawRelationGlyph(ctx, x, centerY, 3.5, item.type);
    });
  }

  const directionY = height - 17.5;
  const directionStartX = Math.min(width - 50, Math.max(49, labelX + Math.min(labelWidth, 58)));
  const directionEndX = width - 16;
  if (directionEndX - directionStartX > 16) {
    ctx.strokeStyle = alpha(color, muted ? 0.2 : highlighted ? 0.58 : 0.38);
    ctx.lineWidth = highlighted ? 1.35 : 1.05;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(directionStartX, directionY);
    ctx.lineTo(directionEndX, directionY);
    ctx.lineTo(directionEndX - 5.2, directionY - 3.8);
    ctx.moveTo(directionEndX, directionY);
    ctx.lineTo(directionEndX - 5.2, directionY + 3.8);
    ctx.stroke();
  }

  ctx.globalAlpha = 1;
  const texture = createGraphCanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    opacity: muted ? 0.44 : highlighted ? 1 : 0.94,
  });
  const sprite = new THREE.Sprite(material);
  sprite.userData.relationMarker = true;
  sprite.renderOrder = highlighted ? 10 : 7;
  const scale = highlighted ? 8.1 : 8.55;
  sprite.scale.set(width / scale, height / scale, 1);
  return sprite;
};

const disposeThreeMaterial = (material?: THREE.Material | THREE.Material[]) => {
  const materials = Array.isArray(material) ? material : material ? [material] : [];
  materials.forEach((item) => {
    const withMap = item as THREE.Material & { map?: THREE.Texture };
    withMap.map?.dispose();
    item.dispose();
  });
};

const disposeGraphSceneObject = (object: THREE.Object3D) => {
  object.traverse((child) => {
    if (child.userData.kgSharedHitTarget) return;
    const mesh = child as THREE.Mesh;
    mesh.geometry?.dispose?.();
    disposeThreeMaterial(mesh.material as THREE.Material | THREE.Material[] | undefined);
  });
};

const GRAPH_NODE_HIT_TARGET_GEOMETRY = new THREE.SphereGeometry(1, 8, 6);
const GRAPH_NODE_HIT_TARGET_MATERIAL = (() => {
  const material = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0,
    depthTest: false,
    depthWrite: false,
  });
  material.colorWrite = false;
  return material;
})();

const createGraphNodeHitTarget = (radius: number) => {
  const hitTarget = new THREE.Mesh(GRAPH_NODE_HIT_TARGET_GEOMETRY, GRAPH_NODE_HIT_TARGET_MATERIAL);
  hitTarget.name = 'kg-node-hit-target';
  hitTarget.scale.setScalar(Math.max(1, radius));
  hitTarget.renderOrder = -100;
  hitTarget.userData.kgSharedHitTarget = true;
  return hitTarget;
};

const createGraphNodeLabelConnector = (
  labelPosition: THREE.Vector3,
  size: number,
  color: string,
  highlighted: boolean,
  focused: boolean,
  muted: boolean,
  labelStyle: GraphNodeLabelStyle
) => {
  const distance = labelPosition.length();
  if (distance < size * 1.35) return null;

  const direction = labelPosition.clone().normalize();
  const labelInset = labelStyle === 'rich'
    ? size * 0.86
    : labelStyle === 'annotation'
      ? size * 0.74
      : labelStyle === 'neighbor'
        ? size * 0.68
        : size * 0.58;
  const start = direction.clone().multiplyScalar(size * (highlighted ? 1.18 : focused ? 1.08 : 0.98));
  const end = labelPosition.clone().sub(direction.clone().multiplyScalar(labelInset));
  if (start.distanceTo(end) < size * 0.42) return null;

  const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
  const material = new THREE.LineBasicMaterial({
    color: new THREE.Color(color),
    transparent: true,
    opacity: muted ? 0.1 : highlighted ? 0.42 : focused ? 0.28 : 0.18,
    depthTest: false,
    depthWrite: false,
  });
  const line = new THREE.Line(geometry, material);
  line.userData.kgLabelConnector = true;
  line.renderOrder = highlighted ? 10 : focused ? 8 : 6;
  return line;
};

const createDepthReferenceLine = (
  points: THREE.Vector3[],
  color: number,
  opacity: number
) => {
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthWrite: false,
    depthTest: true,
  });
  const line = new THREE.Line(geometry, material);
  line.renderOrder = -8;
  return line;
};

const createGraphDepthReferenceObject = (
  nodes: RenderGraphNode[],
  tier: GraphScaleTier
) => {
  if (nodes.length < 2) return null;

  const bounds = nodes.reduce(
    (acc, node) => ({
      minX: Math.min(acc.minX, Number(node.x || 0)),
      maxX: Math.max(acc.maxX, Number(node.x || 0)),
      minY: Math.min(acc.minY, Number(node.y || 0)),
      maxY: Math.max(acc.maxY, Number(node.y || 0)),
      minZ: Math.min(acc.minZ, Number(node.z || 0)),
      maxZ: Math.max(acc.maxZ, Number(node.z || 0)),
    }),
    { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, minZ: Infinity, maxZ: -Infinity }
  );
  const center = new THREE.Vector3(
    (bounds.minX + bounds.maxX) / 2,
    (bounds.minY + bounds.maxY) / 2,
    (bounds.minZ + bounds.maxZ) / 2
  );
  const span = Math.max(
    bounds.maxX - bounds.minX,
    bounds.maxY - bounds.minY,
    bounds.maxZ - bounds.minZ,
    48
  );
  const radius = Math.min(
    tier === 'massive' ? 1600 : tier === 'large' ? 1120 : 760,
    Math.max(72, span * 0.66)
  );
  const zSpan = Math.max(42, Math.min(radius * 1.24, bounds.maxZ - bounds.minZ + radius * 0.34));
  const group = new THREE.Group();
  group.userData.kgDepthReference = true;

  [0.38, 0.68, 1].forEach((scale, index) => {
    const ringRadius = radius * scale;
    const points = Array.from({ length: 145 }, (_, pointIndex) => {
      const angle = (pointIndex / 144) * Math.PI * 2;
      return new THREE.Vector3(
        center.x + Math.cos(angle) * ringRadius,
        center.y + Math.sin(angle) * ringRadius,
        center.z
      );
    });
    group.add(createDepthReferenceLine(points, 0x4452a3, index === 2 ? 0.09 : 0.055));
  });

  group.add(createDepthReferenceLine([
    new THREE.Vector3(center.x - radius, center.y, center.z),
    new THREE.Vector3(center.x + radius, center.y, center.z),
  ], 0x7dd3fc, 0.12));
  group.add(createDepthReferenceLine([
    new THREE.Vector3(center.x, center.y - radius, center.z),
    new THREE.Vector3(center.x, center.y + radius, center.z),
  ], 0x8b5cf6, 0.1));
  group.add(createDepthReferenceLine([
    new THREE.Vector3(center.x, center.y, center.z - zSpan * 0.5),
    new THREE.Vector3(center.x, center.y, center.z + zSpan * 0.5),
  ], 0xf59e0b, 0.12));

  const tickCount = tier === 'massive' ? 4 : 6;
  for (let i = 0; i <= tickCount; i += 1) {
    const t = i / tickCount;
    const z = center.z - zSpan * 0.5 + zSpan * t;
    const tickRadius = radius * (i === 0 || i === tickCount ? 0.04 : 0.026);
    group.add(createDepthReferenceLine([
      new THREE.Vector3(center.x - tickRadius, center.y, z),
      new THREE.Vector3(center.x + tickRadius, center.y, z),
    ], 0xfbbf24, 0.12));
  }

  return group;
};

const setGraphNodeObjectLabelOpacity = (object: THREE.Object3D, opacity: number) => {
  const nextOpacity = Math.max(0, Math.min(1, opacity));
  object.traverse((child) => {
    if (!child.userData.kgLabelSprite) return;
    const material = (child as THREE.Sprite).material;
    if (!(material instanceof THREE.SpriteMaterial)) return;
    const baseOpacity = Number(material.userData.kgBaseOpacity ?? 1);
    const nextMaterialOpacity = baseOpacity * nextOpacity;
    if (Math.abs(material.opacity - nextMaterialOpacity) < 0.003) return;
    material.opacity = nextMaterialOpacity;
  });
};

const createGraphNodeObject = ({
  node,
  relationSummary,
  highlighted,
  focused,
  muted,
  softMuted,
  labelVisible,
  iconVisible,
  labelStyle,
  labelLayout,
  labelOpacity,
  labelWorldScale,
  symbolWorldScale = 1,
  relationHint,
  relationTypeMix,
  lowDetail,
  contextVisible = false,
  softMuteStrength = 1,
}: {
  node: RenderGraphNode;
  relationSummary?: { links: number; relations: number };
  highlighted: boolean;
  focused: boolean;
  muted: boolean;
  softMuted: boolean;
  labelVisible: boolean;
  iconVisible: boolean;
  labelStyle: GraphNodeLabelStyle;
  labelLayout?: GraphNodeLabelLayout;
  labelOpacity?: number;
  labelWorldScale?: number;
  symbolWorldScale?: number;
  relationHint?: GraphFocusHaloLabelHint;
  relationTypeMix?: RelationTypeCount[];
  lowDetail: boolean;
  contextVisible?: boolean;
  softMuteStrength?: number;
}) => {
  const colors = getTypeColors(node.type, node.label);
  const relationBoost = Math.log2((relationSummary?.relations || 0) + 1);
  const val = Math.max(1, Number(node.val || 5));
  const contextMuted = muted || softMuted;
  const softFocusAmount = softMuted ? Math.max(0, Math.min(1, softMuteStrength)) : 0;
  const normalValBoost = highlighted ? 0.44 : focused ? 0.36 : lowDetail ? 0.3 : 0.38;
  const normalMinSize = highlighted ? 2.25 : focused ? 2.0 : lowDetail ? 1.55 : 1.84;
  const normalMaxSize = highlighted ? 4.8 : focused ? 4.1 : lowDetail ? 3.55 : 4.45;
  const normalBaseSize = highlighted ? 1.72 : focused ? 1.52 : lowDetail ? 1.18 : 1.38;
  const normalRelationBoost = highlighted ? 0.07 : focused ? 0.056 : lowDetail ? 0.04 : 0.052;
  const normalEmissive = highlighted ? 0.34 : focused ? 0.18 : lowDetail ? 0.045 : 0.07;
  const normalOpacity = highlighted ? 0.98 : focused ? 0.9 : lowDetail ? 0.68 : 0.82;
  const visualValBoost = softFocusAmount > 0 ? lerpGraphValue(normalValBoost, 0.34, softFocusAmount) : normalValBoost;
  const visualMinSize = softFocusAmount > 0 ? lerpGraphValue(normalMinSize, 1.54, softFocusAmount) : normalMinSize;
  const visualMaxSize = softFocusAmount > 0 ? lerpGraphValue(normalMaxSize, 3.34, softFocusAmount) : normalMaxSize;
  const visualBaseSize = softFocusAmount > 0 ? lerpGraphValue(normalBaseSize, 1.13, softFocusAmount) : normalBaseSize;
  const visualRelationBoost = softFocusAmount > 0 ? lerpGraphValue(normalRelationBoost, 0.04, softFocusAmount) : normalRelationBoost;
  const visualEmissive = softFocusAmount > 0 ? lerpGraphValue(normalEmissive, 0.052, softFocusAmount) : normalEmissive;
  const visualOpacity = softFocusAmount > 0 ? lerpGraphValue(normalOpacity, 0.58, softFocusAmount) : normalOpacity;
  const valBoost = Math.sqrt(val) * (muted ? 0.24 : visualValBoost);
  const size = Math.max(
    muted ? 1.18 : visualMinSize,
    Math.min(
      muted ? 2.7 : visualMaxSize,
      (muted ? 0.92 : visualBaseSize) +
        valBoost +
        relationBoost * (muted ? 0.018 : visualRelationBoost)
    )
  );
  const group = new THREE.Group();
  group.userData.nodeId = node.id;
  group.userData.highlighted = highlighted;
  group.userData.focused = focused;

  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(colors.main),
    emissive: new THREE.Color(colors.main),
    emissiveIntensity: muted ? (contextVisible ? 0.055 : 0.032) : visualEmissive,
    roughness: lowDetail ? 0.62 : 0.5,
    metalness: lowDetail ? 0.04 : 0.12,
    transparent: true,
    opacity: muted ? (contextVisible ? 0.24 : 0.13) : visualOpacity,
  });
  const mesh = new THREE.Mesh(getNodeGeometry(node.type, size, node.label, lowDetail ? 'lite' : 'full'), material);
  mesh.userData.nodeId = node.id;
  group.add(mesh);

  if (iconVisible || highlighted || focused || contextVisible) {
    const outlineGeometry = new THREE.EdgesGeometry(mesh.geometry, 28);
    const outlineMaterial = new THREE.LineBasicMaterial({
      color: new THREE.Color(highlighted ? '#ffffff' : colors.light),
      transparent: true,
      opacity: muted
        ? contextVisible ? 0.26 : 0.13
        : highlighted ? 0.62 : focused ? 0.44 : iconVisible ? 0.28 : 0.2,
      depthTest: true,
      depthWrite: false,
    });
    const outline = new THREE.LineSegments(outlineGeometry, outlineMaterial);
    outline.userData.kgNodeShapeOutline = true;
    outline.renderOrder = highlighted ? 5 : focused ? 4 : 2;
    group.add(outline);
  }

  const hitRadius = Math.min(
    lowDetail ? 9.5 : 13.5,
    Math.max(
      size * (highlighted ? 2.85 : focused ? 2.55 : labelVisible ? 2.35 : 2.08),
      highlighted ? 8.5 : focused || labelVisible ? 7.2 : 5.8
    )
  );
  const hitTarget = createGraphNodeHitTarget(hitRadius);
  hitTarget.userData.nodeId = node.id;
  group.userData.hitRadius = hitRadius;
  group.add(hitTarget);

  if (highlighted || focused) {
    const halo = new THREE.Mesh(
      new THREE.SphereGeometry(size * (highlighted ? 1.28 : focused ? 1.18 : 1.12), 24, 12),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(colors.main),
        transparent: true,
        opacity: highlighted ? 0.075 : 0.026,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
    );
    halo.userData.pulseHalo = highlighted || focused;
    group.add(halo);
  }

  if (focused && !highlighted) {
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(size * 1.18, Math.max(0.04, size * 0.02), 8, 42),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(colors.light),
        transparent: true,
        opacity: 0.08,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
    );
    ring.rotation.x = Math.PI / 2;
    ring.userData.focusRing = true;
    group.add(ring);
  }

  if ((highlighted || focused) && relationTypeMix?.length) {
    const relationHalo = createNodeRelationHaloSprite(relationTypeMix, size, highlighted, focused, contextMuted);
    if (relationHalo) group.add(relationHalo);
  }

  if (iconVisible) {
    const iconSprite = createNodeIconSprite(node, colors, size, highlighted, focused, contextMuted, symbolWorldScale);
    if (iconSprite) group.add(iconSprite);
  }

  if (labelVisible) {
    const labelSprite = createNodeLabelSprite(node, colors, highlighted, contextMuted, labelStyle, labelOpacity, labelWorldScale, relationHint, relationSummary);
    if (labelSprite) {
      const anchor = labelLayout?.anchor || 'right';
      labelSprite.position.copy(getNodeLabelWorldPosition(
        anchor,
        labelSprite.scale,
        size,
        highlighted,
        focused,
        iconVisible,
        symbolWorldScale,
        labelStyle
      ));
      const labelConnector = createGraphNodeLabelConnector(
        labelSprite.position.clone(),
        size,
        relationHint?.color || colors.light,
        highlighted,
        focused,
        contextMuted,
        labelStyle
      );
      if (labelConnector) group.add(labelConnector);
      group.add(labelSprite);
    }
  }

  return group;
};

const detectWebGLSupport = () => {
  if (typeof document === 'undefined') return true;
  try {
    const canvas = document.createElement('canvas');
    return Boolean(canvas.getContext('webgl2') || canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
  } catch {
    return false;
  }
};

// Force-Directed Graph Visualization using Three.js / react-force-graph-3d
  interface ForceGraphProps {
    nodes: GraphNode[];
    edges: GraphEdge[];
    selectedNode: string | null;
    selectedEdge: string | null;
    focusNodeId: string | null;
    hoveredNode: string | null;
    activeNodeIds: Set<string>;
    activeEdgeIds: Set<string>;
    pathFocusNodeIds: string[];
    pathFocusVersion: number;
    searchTerm: string;
    showLabels: boolean;
    showEdgeLabels: boolean;
    labelDensity: GraphLabelDensity;
      compactSelectedNodeLabel: boolean;
      compactSelectedEdgeEndpointLabels: boolean;
      selectionExpanded: boolean;
      detailMode: GraphDetailMode;
    cameraFocusRequest?: GraphCameraFocusRequest | null;
    viewSignature: string;
  particleFlow: boolean;
  onNodeClick: (node: GraphNode) => void;
  onLinkClick: (edge: GraphEdge) => void;
  onNodeDive?: (node: GraphNode) => void;
  onLinkDive?: (edge: GraphEdge) => void;
    onNodeHover: (node: GraphNode | null) => void;
    onBackgroundClick: () => void;
    onSelectionAnchorChange?: (anchor: GraphSelectionAnchor | null) => void;
    onCameraLodChange?: (camera: GraphCameraLod) => void;
    overlayReservedRects?: GraphScreenRect[];
    topology: GraphTopology;
  isFullscreen: boolean;
  physicsEnabled: boolean;
  suppressHoverInteractions: boolean;
}

const ForceGraphVisualization = ({
    nodes,
    edges,
    selectedNode,
    selectedEdge,
    focusNodeId,
    hoveredNode,
    activeNodeIds,
    activeEdgeIds,
    pathFocusNodeIds,
    pathFocusVersion,
    searchTerm,
    showLabels,
    showEdgeLabels,
      labelDensity,
      compactSelectedNodeLabel,
      compactSelectedEdgeEndpointLabels,
      selectionExpanded,
      detailMode,
  cameraFocusRequest,
  viewSignature,
  particleFlow,
  onNodeClick,
  onLinkClick,
  onNodeDive,
  onLinkDive,
  onNodeHover,
  onBackgroundClick,
  onSelectionAnchorChange,
  onCameraLodChange,
  overlayReservedRects = [],
  topology,
  isFullscreen,
  physicsEnabled,
  suppressHoverInteractions,
  }: ForceGraphProps) => {
      const fgRef = useRef<ForceGraphRuntimeHandle>();
      const containerRef = useRef<HTMLDivElement>(null);
        const nodeObjectCacheRef = useRef(new Map<string, { key: string; object: THREE.Object3D }>());
        const linkObjectCacheRef = useRef(new Map<string, { key: string; object: THREE.Object3D }>());
        const animatedNodeObjectsRef = useRef(new Map<string, THREE.Object3D>());
        const componentRegionGroupRef = useRef<THREE.Group | null>(null);
        const depthReferenceGroupRef = useRef<THREE.Group | null>(null);
        const pendingAutoFitRef = useRef(false);
        const autoFitTimersRef = useRef<number[]>([]);
        const lastUserCameraInteractionRef = useRef(0);
      const stableLabelLayoutsRef = useRef(new Map<string, GraphNodeLabelLayout>());
      const persistedLabelStateRef = useRef(new Map<string, GraphPersistedLabel>());
      const cameraLodRef = useRef<GraphCameraLod>(INITIAL_GRAPH_CAMERA_LOD);
      const visualCameraLodRef = useRef<GraphCameraLod>(INITIAL_GRAPH_CAMERA_LOD);
      const cameraLodTickRef = useRef(0);
      const cameraMovingUntilRef = useRef(0);
      const cameraFlightFrameRef = useRef<number | null>(null);
      const cameraFlightTokenRef = useRef(0);
      const programmaticCameraUntilRef = useRef(0);
      const manualCameraInteractionFrameRef = useRef<number | null>(null);
      const manualCameraReleaseTimerRef = useRef<number | null>(null);
      const visualCameraSettleTimerRef = useRef<number | null>(null);
      const lastCameraFlightRef = useRef<{ signature: string; at: number } | null>(null);
      const lastAutoFitAtRef = useRef(0);
      const initialViewportReadyTimerRef = useRef<number | null>(null);
      const hoverIntentTimerRef = useRef<number | null>(null);
      const hoverCardExitTimerRef = useRef<number | null>(null);
      const hoverFocusAnimationFrameRef = useRef<number | null>(null);
      const hoverFocusWeightRef = useRef(0);
      const hoverFocusNodeIdsRef = useRef(new Set<string>());
      const hoverFocusEdgeIdsRef = useRef(new Set<string>());
      const routeFocusAnimationFrameRef = useRef<number | null>(null);
      const routeFocusWeightRef = useRef(0);
      const routeFocusNodeIdsRef = useRef(new Set<string>());
      const routeFocusEdgeIdsRef = useRef(new Set<string>());
      const pointerInsideGraphRef = useRef(false);
      const lastSearchFocusSignatureRef = useRef<string | null>(null);
      const lastSelectionFocusSignatureRef = useRef<string | null>(null);
    const [webglSupported] = useState(detectWebGLSupport);
    const [ForceGraph3DRuntime, setForceGraph3DRuntime] = useState<ForceGraph3DComponent | null>(null);
    const [graphRuntimeError, setGraphRuntimeError] = useState<string | null>(null);
    const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [cameraLod, setCameraLod] = useState(cameraLodRef.current);
  const [visualCameraLod, setVisualCameraLod] = useState(visualCameraLodRef.current);
  const [manualCameraInteractionVersion, setManualCameraInteractionVersion] = useState(0);
  const [graphLayoutVersion, setGraphLayoutVersion] = useState(0);
  const [graphInitialViewportReady, setGraphInitialViewportReady] = useState(false);
  const [cursorPosition, setCursorPosition] = useState({ x: 16, y: 16 });
  const [hoveredLink, setHoveredLink] = useState<RenderGraphEdge | null>(null);
  const [hoverFocusWeight, setHoverFocusWeight] = useState(0);
  const [routeFocusWeight, setRouteFocusWeight] = useState(0);
  const cursorFrameRef = useRef<number | null>(null);
  const pendingCursorPositionRef = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    let active = true;
    setGraphRuntimeError(null);
    loadForceGraph3D()
      .then((component) => {
        if (active) setForceGraph3DRuntime(() => component);
      })
      .catch((error: unknown) => {
        if (!active) return;
        setGraphRuntimeError(error instanceof Error ? error.message : 'Unable to load 3D graph runtime');
      });

    return () => {
      active = false;
    };
  }, []);

  const markInitialViewportReady = useCallback((delay = 0) => {
    if (initialViewportReadyTimerRef.current !== null) {
      window.clearTimeout(initialViewportReadyTimerRef.current);
      initialViewportReadyTimerRef.current = null;
    }

    if (delay <= 0) {
      setGraphInitialViewportReady(true);
      return;
    }

    initialViewportReadyTimerRef.current = window.setTimeout(() => {
      initialViewportReadyTimerRef.current = null;
      setGraphInitialViewportReady(true);
    }, delay);
  }, []);

  useEffect(() => {
    setGraphInitialViewportReady(false);
    if (initialViewportReadyTimerRef.current !== null) {
      window.clearTimeout(initialViewportReadyTimerRef.current);
      initialViewportReadyTimerRef.current = null;
    }
  }, [ForceGraph3DRuntime, viewSignature]);

  useEffect(() => () => {
    if (initialViewportReadyTimerRef.current !== null) {
      window.clearTimeout(initialViewportReadyTimerRef.current);
      initialViewportReadyTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!ForceGraph3DRuntime || !nodes.length || graphInitialViewportReady) return undefined;
    const fallbackTimer = window.setTimeout(() => {
      setGraphInitialViewportReady(true);
    }, nodes.length > 1200 ? 4200 : 3200);

    return () => {
      window.clearTimeout(fallbackTimer);
    };
  }, [ForceGraph3DRuntime, graphInitialViewportReady, nodes.length, viewSignature]);

  const renderProfile = useMemo(
    () => applyGraphDetailMode(getGraphRenderProfile(nodes.length, edges.length), detailMode),
    [detailMode, edges.length, nodes.length]
  );

  const nodeRelationSummary = useMemo(() => {
    const summary = new Map<string, { links: number; relations: number }>();
    const ensure = (id: string) => {
      if (!summary.has(id)) summary.set(id, { links: 0, relations: 0 });
      return summary.get(id)!;
    };

    edges.forEach((edge) => {
      const relationCount = Math.max(1, edge.relationCount || 1);
      const source = ensure(edge.source);
      const target = ensure(edge.target);
      source.links += 1;
      target.links += 1;
      source.relations += relationCount;
      target.relations += relationCount;
    });

    return summary;
  }, [edges]);

  const nodeRelationTypeMix = useMemo(() => {
    const mix = new Map<string, Map<string, number>>();
    const ensure = (id: string) => {
      if (!mix.has(id)) mix.set(id, new Map<string, number>());
      return mix.get(id)!;
    };

    edges.forEach((edge) => {
      const relationTypes = getLinkRelationTypeCounts(edge, 6);
      const source = ensure(edge.source);
      const target = ensure(edge.target);
      relationTypes.forEach(({ type, count }) => {
        source.set(type, (source.get(type) || 0) + count);
        target.set(type, (target.get(type) || 0) + count);
      });
    });

    const normalized = new Map<string, RelationTypeCount[]>();
    mix.forEach((counts, nodeId) => {
      normalized.set(
        nodeId,
        Array.from(counts.entries())
          .map(([type, count]) => ({ type, count }))
          .sort((a, b) => b.count - a.count || a.type.localeCompare(b.type))
          .slice(0, 6)
      );
    });
    return normalized;
  }, [edges]);

  const hoveredNodeDetails = useMemo(
    () => nodes.find((node) => node.id === hoveredNode) || null,
    [hoveredNode, nodes]
  );

  const nodeById = useMemo(() => new Map(nodes.map((node) => [node.id, node])), [nodes]);

  const hoveredNodeRelationPreview = useMemo<HoveredNodeRelationPreview | null>(() => {
    if (!hoveredNodeDetails) return null;

    const typeCounts = new Map<string, number>();
    let strongest: HoveredNodeStrongestRelation | null = null;

    edges.forEach((edge) => {
      const sourceId = typeof edge.source === 'string' ? edge.source : (edge.source as RenderGraphNode | undefined)?.id;
      const targetId = typeof edge.target === 'string' ? edge.target : (edge.target as RenderGraphNode | undefined)?.id;
      if (sourceId !== hoveredNodeDetails.id && targetId !== hoveredNodeDetails.id) return;

      const otherId = sourceId === hoveredNodeDetails.id ? targetId : sourceId;
      if (!otherId) return;

      const rawRelationTypes: RelationTypePayload[] = Array.isArray(edge.properties?.relationTypes)
        ? edge.properties.relationTypes as RelationTypePayload[]
        : [];
      const relationTypes = (rawRelationTypes.length > 0
        ? rawRelationTypes
        : [{ type: normalizeType(edge.type || edge.label), count: edge.relationCount || 1 }]
      )
        .map((item) => ({
          type: normalizeType(item.type),
          count: Math.max(1, Number(item.count || 0)),
        }))
        .filter((item: RelationTypeCount) => item.type && item.count > 0)
        .sort((a: RelationTypeCount, b: RelationTypeCount) => b.count - a.count || a.type.localeCompare(b.type));

      relationTypes.forEach(({ type, count }) => {
        typeCounts.set(type, (typeCounts.get(type) || 0) + count);
      });

      const primaryType = relationTypes[0]?.type || normalizeType(edge.type || edge.label);
      const relationCount = Math.max(1, Number(edge.relationCount || relationTypes.reduce((sum, item) => sum + item.count, 0) || 1));
      const weight = Number(edge.weight || 0);
      const score = relationCount * 10 + weight;
      const strongestScore = strongest ? strongest.count * 10 + strongest.weight : -1;
      if (score > strongestScore) {
        strongest = {
          color: getRelationshipColor(primaryType),
          count: relationCount,
          otherId,
          otherNode: nodeById.get(otherId),
          type: primaryType,
          weight,
        };
      }
    });

    const relationTypes = Array.from(typeCounts.entries())
      .map(([type, count]) => ({
        color: getRelationshipColor(type),
        count,
        type,
      }))
      .sort((a, b) => b.count - a.count || a.type.localeCompare(b.type))
      .slice(0, 4);

    return relationTypes.length || strongest ? { relationTypes, strongest } : null;
  }, [edges, hoveredNodeDetails, nodeById]);

  const selectedEdgeEndpointIds = useMemo(() => {
    if (!selectedEdge) return new Set<string>();
    const edge = edges.find((item) => item.id === selectedEdge);
    if (!edge) return new Set<string>();

    return new Set([edge.source, edge.target].filter((id): id is string => Boolean(id)));
  }, [edges, selectedEdge]);

  const prioritizedNodeIds = useMemo(() => {
    const nodeBudget = renderProfile.nodeBudget[visualCameraLod.zoomTier];
    const labelBudget = renderProfile.labelBudget[visualCameraLod.zoomTier];
    const iconBudget = renderProfile.iconBudget[visualCameraLod.zoomTier];
    const anchorLabelBudget = getGraphAnchorLabelLimit(
      detailMode,
      labelDensity,
      renderProfile.tier,
      visualCameraLod.zoomTier,
      dimensions.width,
      nodes.length
    );
    const sorted = [...nodes].sort((a, b) =>
      getNodeImportanceScore(b, nodeRelationSummary.get(b.id)) -
      getNodeImportanceScore(a, nodeRelationSummary.get(a.id)) ||
      a.label.localeCompare(b.label)
    );

    const anchorLabelCandidateLimit = anchorLabelBudget > 0
      ? Math.min(sorted.length, Math.max(anchorLabelBudget, anchorLabelBudget * 5, anchorLabelBudget + 8))
      : 0;
    const fallbackLabelBudget = labelDensity === 'focus'
      ? 0
      : dimensions.width < 700
        ? renderProfile.tier === 'massive'
          ? 4
          : renderProfile.tier === 'large'
            ? 6
            : 8
        : renderProfile.tier === 'massive'
          ? visualCameraLod.zoomTier === 'near' ? 10 : visualCameraLod.zoomTier === 'middle' ? 7 : 4
          : renderProfile.tier === 'large'
            ? visualCameraLod.zoomTier === 'near' ? 18 : visualCameraLod.zoomTier === 'middle' ? 14 : 9
            : renderProfile.tier === 'standard'
              ? visualCameraLod.zoomTier === 'near' ? 28 : visualCameraLod.zoomTier === 'middle' ? 22 : 14
              : visualCameraLod.zoomTier === 'near' ? 34 : visualCameraLod.zoomTier === 'middle' ? 26 : 16;

    return {
      visible: new Set(
        Number.isFinite(nodeBudget)
          ? sorted.slice(0, nodeBudget).map((node) => node.id)
          : sorted.map((node) => node.id)
      ),
      labels: new Set(
        Number.isFinite(labelBudget)
          ? sorted.slice(0, labelBudget).map((node) => node.id)
          : sorted.map((node) => node.id)
      ),
      anchorLabels: new Set(
        anchorLabelCandidateLimit > 0
          ? sorted.slice(0, anchorLabelCandidateLimit).map((node) => node.id)
          : []
      ),
      fallbackLabels: new Set(
        fallbackLabelBudget > 0
          ? sorted.slice(0, Math.min(sorted.length, fallbackLabelBudget)).map((node) => node.id)
          : []
      ),
      anchorLabelLimit: anchorLabelBudget,
      icons: new Set(
        Number.isFinite(iconBudget)
          ? sorted.slice(0, iconBudget).map((node) => node.id)
          : sorted.map((node) => node.id)
      ),
    };
  }, [detailMode, dimensions.width, labelDensity, nodeRelationSummary, nodes, renderProfile, visualCameraLod.zoomTier]);

  const prioritizedLinkIds = useMemo(() => {
    const linkBudget = renderProfile.linkBudget[visualCameraLod.zoomTier];
    if (!Number.isFinite(linkBudget) || edges.length <= linkBudget) return null;

    return new Set(
      [...edges]
        .sort((a, b) =>
          Number(b.relationCount || 1) - Number(a.relationCount || 1) ||
          Number(b.weight || 0) - Number(a.weight || 0) ||
          a.id.localeCompare(b.id)
        )
        .slice(0, linkBudget)
      .map((edge) => edge.id)
    );
  }, [edges, renderProfile, visualCameraLod.zoomTier]);

  const ambientDirectionCueLinkIds = useMemo(() => {
    const searchActive = Boolean(searchTerm.trim());
    const hasExplicitFocus = activeEdgeIds.size > 0 ||
      activeNodeIds.size > 0 ||
      Boolean(selectedEdge) ||
      Boolean(hoveredLink?.id);
    if (cameraLod.moving || searchActive || hasExplicitFocus || detailMode === 'performance') {
      return new Set<string>();
    }

    const cueBudget =
      renderProfile.tier === 'massive'
        ? 0
        : renderProfile.tier === 'large'
          ? visualCameraLod.zoomTier === 'near' ? 1 : 0
          : renderProfile.tier === 'standard'
            ? visualCameraLod.zoomTier === 'far' ? 0 : visualCameraLod.zoomTier === 'middle' ? 1 : 2
            : visualCameraLod.zoomTier === 'far' ? 0 : visualCameraLod.zoomTier === 'middle' ? 1 : 2;
    if (cueBudget <= 0) return new Set<string>();

    return new Set(
      [...edges]
        .filter((edge) => !prioritizedLinkIds || prioritizedLinkIds.has(edge.id))
        .sort((a, b) =>
          Number(b.relationCount || 1) - Number(a.relationCount || 1) ||
          Number(b.weight || 0) - Number(a.weight || 0) ||
          a.id.localeCompare(b.id)
        )
        .slice(0, cueBudget)
        .map((edge) => edge.id)
    );
  }, [
    activeEdgeIds.size,
    activeNodeIds.size,
    cameraLod.moving,
    detailMode,
    edges,
    hoveredLink?.id,
    prioritizedLinkIds,
    renderProfile.tier,
    searchTerm,
    selectedEdge,
    visualCameraLod.zoomTier,
  ]);

  const relationMarkerLinkIds = useMemo(() => {
    const markerBudget =
      renderProfile.tier === 'massive'
        ? visualCameraLod.zoomTier === 'near' ? 4 : visualCameraLod.zoomTier === 'middle' ? 2 : 0
        : renderProfile.tier === 'large'
          ? visualCameraLod.zoomTier === 'far' ? 3 : visualCameraLod.zoomTier === 'middle' ? 5 : 8
          : renderProfile.tier === 'standard'
            ? visualCameraLod.zoomTier === 'far' ? 5 : visualCameraLod.zoomTier === 'middle' ? 8 : 12
            : visualCameraLod.zoomTier === 'far' ? 6 : visualCameraLod.zoomTier === 'middle' ? 10 : 14;

    if (markerBudget <= 0) return new Set<string>();
    const rankedEdges = [...edges]
      .filter((edge) => !prioritizedLinkIds || prioritizedLinkIds.has(edge.id))
      .sort((a, b) =>
        Number(b.relationCount || 1) - Number(a.relationCount || 1) ||
        Number(b.weight || 0) - Number(a.weight || 0) ||
        a.id.localeCompare(b.id)
    );

    const chosen = new Set<string>();
    const chosenKinds = new Set<RelationVisualKind>();
    const chosenTypes = new Set<string>();
    const addEdge = (edge: GraphEdge, mode: 'kind' | 'type' | 'fill') => {
      if (chosen.size >= markerBudget || chosen.has(edge.id)) return;
      const relationType = getLinkRelationTypeCounts(edge, 1)[0]?.type || normalizeType(edge.type || edge.label);
      const visualKind = getRelationVisualKind(relationType);
      if (mode === 'kind' && chosenKinds.has(visualKind)) return;
      if (mode === 'type' && chosenTypes.has(relationType)) return;
      chosen.add(edge.id);
      chosenKinds.add(visualKind);
      chosenTypes.add(relationType);
    };

    rankedEdges.forEach((edge) => addEdge(edge, 'kind'));
    rankedEdges.forEach((edge) => addEdge(edge, 'type'));
    rankedEdges.forEach((edge) => addEdge(edge, 'fill'));

    return chosen;
  }, [edges, prioritizedLinkIds, renderProfile.tier, visualCameraLod.zoomTier]);

  const focusedRelationMarkerLinkIds = useMemo(() => {
    if (activeEdgeIds.size === 0) return new Set<string>();
    const baseMarkerBudget =
      renderProfile.tier === 'massive'
        ? visualCameraLod.zoomTier === 'near' ? 1 : 0
        : renderProfile.tier === 'large'
          ? visualCameraLod.zoomTier === 'far' ? 0 : visualCameraLod.zoomTier === 'middle' ? 0 : 2
          : renderProfile.tier === 'standard'
            ? visualCameraLod.zoomTier === 'far' ? 0 : visualCameraLod.zoomTier === 'middle' ? 1 : 2
            : visualCameraLod.zoomTier === 'far' ? 0 : visualCameraLod.zoomTier === 'middle' ? 2 : 3;
    const markerBudget = selectedNode && activeEdgeIds.size > 2
      ? Math.min(baseMarkerBudget, visualCameraLod.zoomTier === 'near' ? 2 : 1)
      : baseMarkerBudget;
    if (markerBudget <= 0) return new Set<string>();

    const rankedEdges = edges
      .filter((edge) => activeEdgeIds.has(edge.id))
      .sort((a, b) =>
        Number(b.relationCount || 1) - Number(a.relationCount || 1) ||
        Number(b.weight || 0) - Number(a.weight || 0) ||
        a.id.localeCompare(b.id)
      );
    const chosen = new Set<string>();
    const chosenTypes = new Set<string>();
    rankedEdges.forEach((edge) => {
      if (chosen.size >= markerBudget) return;
      const relationType = getLinkRelationTypeCounts(edge, 1)[0]?.type || normalizeType(edge.type || edge.label);
      if (chosenTypes.has(relationType)) return;
      chosen.add(edge.id);
      chosenTypes.add(relationType);
    });
    if (!selectedNode) {
      rankedEdges.forEach((edge) => {
        if (chosen.size >= markerBudget) return;
        chosen.add(edge.id);
      });
    }

    return chosen;
  }, [activeEdgeIds, edges, renderProfile.tier, selectedNode, visualCameraLod.zoomTier]);

  const hoveredLinkDetails = useMemo(() => {
    if (!hoveredLink) return null;
    const sourceId = typeof hoveredLink.source === 'string' ? hoveredLink.source : hoveredLink.source?.id;
    const targetId = typeof hoveredLink.target === 'string' ? hoveredLink.target : hoveredLink.target?.id;
    if (!sourceId || !targetId) return null;

    const rawRelationTypes: RelationTypePayload[] = Array.isArray(hoveredLink.properties?.relationTypes)
      ? hoveredLink.properties.relationTypes as RelationTypePayload[]
      : [];
    const relationTypes = (rawRelationTypes.length > 0
      ? rawRelationTypes
      : [{ type: normalizeType(hoveredLink.type || hoveredLink.label), count: hoveredLink.relationCount || 1 }]
    )
      .map((item) => ({
        type: normalizeType(item.type),
        count: Number(item.count || 0),
      }))
      .filter((item: RelationTypeCount) => item.type && item.count > 0)
      .sort((a: RelationTypeCount, b: RelationTypeCount) => b.count - a.count || a.type.localeCompare(b.type));

      return {
        source: nodeById.get(sourceId),
        target: nodeById.get(targetId),
        sourceId,
        targetId,
        label: hoveredLink.label || hoveredLink.type || 'related',
        relationCount: Number(hoveredLink.relationCount || 1),
        weight: Number(hoveredLink.weight || 0),
        relationTypes,
        evidenceSnippets: getRelationshipEvidenceSnippets(hoveredLink, 2),
        domains: [
          typeof hoveredLink.properties?.domain === 'string' ? hoveredLink.properties.domain : '',
          ...(Array.isArray(hoveredLink.properties?.domains) ? hoveredLink.properties.domains.map(String) : []),
        ].map((domain) => domain.trim()).filter(Boolean),
      };
  }, [hoveredLink, nodeById]);
  const manualCameraNavigationActive = manualCameraInteractionVersion >= 0 &&
    typeof window !== 'undefined' &&
    window.performance.now() - lastUserCameraInteractionRef.current < GRAPH_MANUAL_NAVIGATION_SETTLE_MS;
  const navigationSettlingActive = cameraLod.moving || manualCameraNavigationActive;
  const motionSimplificationActive = cameraLod.moving && !manualCameraNavigationActive && (
    renderProfile.tier === 'large' ||
    renderProfile.tier === 'massive'
  );
  const searchInteractionActive = Boolean(searchTerm.trim());
  const selectedContextHoverNodeActive = Boolean(
    hoveredNodeDetails &&
    (selectedNode || selectedEdge) &&
    (!selectedNode || hoveredNodeDetails.id !== selectedNode)
  );
  const selectedHoverRouteActive = Boolean(
    selectedNode &&
    hoveredNode &&
    hoveredNode !== selectedNode &&
    activeEdgeIds.size > 0 &&
    !selectedEdge &&
    !searchTerm.trim()
  );
  const hoverInteractionSuppressed = suppressHoverInteractions || searchInteractionActive || navigationSettlingActive;
  const hoverCardsSuppressed =
    suppressHoverInteractions ||
    searchInteractionActive ||
    (navigationSettlingActive && !selectedContextHoverNodeActive) ||
    (Boolean(selectedNode || selectedEdge) && !selectedContextHoverNodeActive);
  const trackHoverCursor = !hoverInteractionSuppressed || selectedContextHoverNodeActive;
  const hoverSoftFocusActive = Boolean((hoveredNode || hoveredLink?.id) && !selectedNode && !selectedEdge && !searchInteractionActive);
  const getLinkEndpointId = useCallback((endpoint: string | RenderGraphNode | undefined) => (
    typeof endpoint === 'string' ? endpoint : endpoint?.id
  ), []);
  const hoveredLinkFocusNodeIds = useMemo(() => {
    const nodeIds = new Set<string>();
    if (!hoveredLink) return nodeIds;
    const sourceId = getLinkEndpointId(hoveredLink.source);
    const targetId = getLinkEndpointId(hoveredLink.target);
    if (sourceId) nodeIds.add(sourceId);
    if (targetId) nodeIds.add(targetId);
    return nodeIds;
  }, [getLinkEndpointId, hoveredLink]);
  const hoveredLinkFocusEdgeIds = useMemo(() => (
    hoveredLink?.id ? new Set([hoveredLink.id]) : new Set<string>()
  ), [hoveredLink?.id]);
  const liveHoverFocusNodeIds = hoveredLink?.id && activeNodeIds.size === 0
    ? hoveredLinkFocusNodeIds
    : activeNodeIds;
  const liveHoverFocusEdgeIds = hoveredLink?.id && activeEdgeIds.size === 0
    ? hoveredLinkFocusEdgeIds
    : activeEdgeIds;
  const hoverFocusRetainingContext = Boolean(
    hoverFocusWeight > 0.01 &&
    !hoverSoftFocusActive &&
    !selectedNode &&
    !selectedEdge &&
    !searchInteractionActive &&
    hoverFocusNodeIdsRef.current.size > 0
  );
  const routeFocusRetainingContext = Boolean(
    routeFocusWeight > 0.01 &&
    !selectedHoverRouteActive &&
    selectedNode &&
    !selectedEdge &&
    !searchInteractionActive &&
    routeFocusNodeIdsRef.current.size > 0
  );
  const visualActiveNodeIds = hoverFocusRetainingContext
    ? hoverFocusNodeIdsRef.current
    : routeFocusRetainingContext
      ? routeFocusNodeIdsRef.current
      : hoverSoftFocusActive
      ? liveHoverFocusNodeIds
      : activeNodeIds;
  const visualActiveEdgeIds = hoverFocusRetainingContext
    ? hoverFocusEdgeIdsRef.current
    : routeFocusRetainingContext
      ? routeFocusEdgeIdsRef.current
      : hoverSoftFocusActive
      ? liveHoverFocusEdgeIds
      : activeEdgeIds;
  const routeFocusVisualActive = Boolean(
    (selectedHoverRouteActive || routeFocusRetainingContext) &&
    selectedNode &&
    !selectedEdge &&
    !searchInteractionActive &&
    visualActiveNodeIds.size > 0
  );
  const hoverFocusVisualActive = Boolean(
    (hoverSoftFocusActive || hoverFocusRetainingContext) &&
    !selectedNode &&
    !selectedEdge &&
    !searchInteractionActive
  );
  const graphLinkHoverPrecision =
    renderProfile.tier === 'massive'
      ? 6
      : renderProfile.tier === 'large'
        ? 9
        : renderProfile.tier === 'standard'
          ? 12
          : 14;
  const pointerInteractionEnabled = !motionSimplificationActive;
  const nodeDragEnabled = renderProfile.tier !== 'massive' && !motionSimplificationActive;

  useEffect(() => {
    if (hoverFocusAnimationFrameRef.current !== null) {
      window.cancelAnimationFrame(hoverFocusAnimationFrameRef.current);
      hoverFocusAnimationFrameRef.current = null;
    }

    const startValue = hoverFocusWeightRef.current;
    const targetValue = hoverSoftFocusActive ? 1 : 0;
    if (Math.abs(startValue - targetValue) < 0.01) {
      hoverFocusWeightRef.current = targetValue;
      setHoverFocusWeight(targetValue);
      return undefined;
    }

    const startedAt = window.performance.now();
    const duration = targetValue > startValue
      ? GRAPH_HOVER_FOCUS_FADE_IN_MS
      : GRAPH_HOVER_FOCUS_FADE_OUT_MS;
    const ease = (value: number) => {
      const clamped = Math.max(0, Math.min(1, value));
      return clamped * clamped * clamped * (clamped * (clamped * 6 - 15) + 10);
    };

    const animate = (timestamp: number) => {
      const progress = Math.max(0, Math.min(1, (timestamp - startedAt) / Math.max(1, duration)));
      const nextValue = lerpGraphValue(startValue, targetValue, ease(progress));
      const rounded = Math.round(nextValue * 1000) / 1000;
      hoverFocusWeightRef.current = rounded;
      setHoverFocusWeight((current) => (Math.abs(current - rounded) < 0.002 ? current : rounded));
      if (progress < 1) {
        hoverFocusAnimationFrameRef.current = window.requestAnimationFrame(animate);
        return;
      }
      hoverFocusAnimationFrameRef.current = null;
      hoverFocusWeightRef.current = targetValue;
      setHoverFocusWeight(targetValue);
    };

    hoverFocusAnimationFrameRef.current = window.requestAnimationFrame(animate);
    return () => {
      if (hoverFocusAnimationFrameRef.current !== null) {
        window.cancelAnimationFrame(hoverFocusAnimationFrameRef.current);
        hoverFocusAnimationFrameRef.current = null;
      }
    };
  }, [hoverSoftFocusActive]);

  useEffect(() => {
    if (routeFocusAnimationFrameRef.current !== null) {
      window.cancelAnimationFrame(routeFocusAnimationFrameRef.current);
      routeFocusAnimationFrameRef.current = null;
    }

    const startValue = selectedHoverRouteActive
      ? routeFocusWeightRef.current
      : Math.max(routeFocusWeightRef.current, routeFocusNodeIdsRef.current.size > 0 ? 1 : 0);
    const targetValue = selectedHoverRouteActive ? 1 : 0;
    if (Math.abs(startValue - targetValue) < 0.01) {
      routeFocusWeightRef.current = targetValue;
      setRouteFocusWeight(targetValue);
      return undefined;
    }

    const startedAt = window.performance.now();
    const duration = targetValue > startValue
      ? GRAPH_ROUTE_FOCUS_FADE_IN_MS
      : GRAPH_ROUTE_FOCUS_FADE_OUT_MS;
    const ease = (value: number) => {
      const clamped = Math.max(0, Math.min(1, value));
      return clamped * clamped * clamped * (clamped * (clamped * 6 - 15) + 10);
    };

    const animate = (timestamp: number) => {
      const progress = Math.max(0, Math.min(1, (timestamp - startedAt) / Math.max(1, duration)));
      const nextValue = lerpGraphValue(startValue, targetValue, ease(progress));
      const rounded = Math.round(nextValue * 1000) / 1000;
      routeFocusWeightRef.current = rounded;
      setRouteFocusWeight((current) => (Math.abs(current - rounded) < 0.002 ? current : rounded));
      if (progress < 1) {
        routeFocusAnimationFrameRef.current = window.requestAnimationFrame(animate);
        return;
      }
      routeFocusAnimationFrameRef.current = null;
      routeFocusWeightRef.current = targetValue;
      setRouteFocusWeight(targetValue);
    };

    routeFocusAnimationFrameRef.current = window.requestAnimationFrame(animate);
    return () => {
      if (routeFocusAnimationFrameRef.current !== null) {
        window.cancelAnimationFrame(routeFocusAnimationFrameRef.current);
        routeFocusAnimationFrameRef.current = null;
      }
    };
  }, [selectedHoverRouteActive]);

  useEffect(() => {
    if (hoverSoftFocusActive && liveHoverFocusNodeIds.size > 0) {
      hoverFocusNodeIdsRef.current = new Set(liveHoverFocusNodeIds);
      hoverFocusEdgeIdsRef.current = new Set(liveHoverFocusEdgeIds);
      return;
    }

    if (!hoverSoftFocusActive && hoverFocusWeight <= 0.01) {
      hoverFocusNodeIdsRef.current.clear();
      hoverFocusEdgeIdsRef.current.clear();
    }
  }, [hoverFocusWeight, hoverSoftFocusActive, liveHoverFocusEdgeIds, liveHoverFocusNodeIds]);

  useEffect(() => {
    if (selectedHoverRouteActive && activeNodeIds.size > 0) {
      routeFocusNodeIdsRef.current = new Set(activeNodeIds);
      routeFocusEdgeIdsRef.current = new Set(activeEdgeIds);
      return;
    }

    if (!selectedHoverRouteActive && routeFocusWeight <= 0.01) {
      routeFocusNodeIdsRef.current.clear();
      routeFocusEdgeIdsRef.current.clear();
    }
  }, [activeEdgeIds, activeNodeIds, routeFocusWeight, selectedHoverRouteActive]);

  useEffect(() => {
    const moving = navigationSettlingActive;
    const applyVisualCameraLod = () => {
      const current = visualCameraLodRef.current;
      const zoomTier = getStableGraphZoomTier(cameraLod.distance, nodes.length, current.zoomTier, moving);
      const visualKey = [
        zoomTier,
        Math.round(cameraLod.distance / (moving ? 360 : 180)),
        Math.round(cameraLod.direction.x * (moving ? 0.42 : 0.86)),
        Math.round(cameraLod.direction.y * (moving ? 0.42 : 0.86)),
        Math.round(cameraLod.direction.z * (moving ? 0.42 : 0.86)),
      ].join(':');
      const distanceThreshold = moving
        ? Math.max(64, cameraLod.distance * 0.26)
        : Math.max(18, cameraLod.distance * 0.075);
      if (
        current.key === visualKey &&
        current.moving === cameraLod.moving &&
        Math.abs(current.distance - cameraLod.distance) < distanceThreshold
      ) {
        return;
      }

      const next: GraphCameraLod = {
        ...cameraLod,
        key: visualKey,
        zoomTier,
      };
      visualCameraLodRef.current = next;
      setVisualCameraLod(next);
    };

    if (visualCameraSettleTimerRef.current !== null) {
      window.clearTimeout(visualCameraSettleTimerRef.current);
      visualCameraSettleTimerRef.current = null;
    }

    if (!moving) {
      applyVisualCameraLod();
      return undefined;
    }

    visualCameraSettleTimerRef.current = window.setTimeout(() => {
      visualCameraSettleTimerRef.current = null;
      applyVisualCameraLod();
    }, 940);

    return () => {
      if (visualCameraSettleTimerRef.current !== null) {
        window.clearTimeout(visualCameraSettleTimerRef.current);
        visualCameraSettleTimerRef.current = null;
      }
    };
  }, [cameraLod, navigationSettlingActive, nodes.length]);

  const updateCursorPosition = useCallback((x: number, y: number) => {
    pendingCursorPositionRef.current = { x, y };
    if (cursorFrameRef.current !== null) return;

    cursorFrameRef.current = window.requestAnimationFrame(() => {
      cursorFrameRef.current = null;
      const next = pendingCursorPositionRef.current;
      pendingCursorPositionRef.current = null;
      if (!next) return;

      setCursorPosition((current) => (
        Math.abs(current.x - next.x) < 2 && Math.abs(current.y - next.y) < 2
          ? current
          : next
      ));
    });
  }, []);

  const clearHoverIntentTimer = useCallback(() => {
    if (hoverIntentTimerRef.current === null) return;
    window.clearTimeout(hoverIntentTimerRef.current);
    hoverIntentTimerRef.current = null;
  }, []);

  const commitGraphHover = useCallback((
    target:
      | { kind: 'node'; node: RenderGraphNode }
      | { kind: 'link'; link: RenderGraphEdge }
      | { kind: 'none' }
  ) => {
    if (target.kind === 'node') {
      setHoveredLink(null);
      onNodeHover(target.node);
      return;
    }

    if (target.kind === 'link') {
      onNodeHover(null);
      setHoveredLink(target.link);
      return;
    }

    setHoveredLink(null);
    onNodeHover(null);
  }, [onNodeHover]);

  const scheduleGraphHover = useCallback((
    target:
      | { kind: 'node'; node: RenderGraphNode }
      | { kind: 'link'; link: RenderGraphEdge }
      | { kind: 'none' }
  ) => {
    clearHoverIntentTimer();

    if (target.kind === 'none' || hoverInteractionSuppressed) {
      hoverIntentTimerRef.current = window.setTimeout(() => {
        hoverIntentTimerRef.current = null;
        commitGraphHover({ kind: 'none' });
      }, GRAPH_HOVER_CLEAR_INTENT_MS);
      return;
    }

    hoverIntentTimerRef.current = window.setTimeout(() => {
      hoverIntentTimerRef.current = null;
      if (!pointerInsideGraphRef.current) return;
      commitGraphHover(target);
    }, GRAPH_HOVER_INTENT_MS);
  }, [clearHoverIntentTimer, commitGraphHover, hoverInteractionSuppressed]);

  useEffect(() => () => {
    if (cursorFrameRef.current !== null) {
      window.cancelAnimationFrame(cursorFrameRef.current);
      cursorFrameRef.current = null;
    }
    clearHoverIntentTimer();
  }, [clearHoverIntentTimer]);

  useEffect(() => {
    if (
      selectedContextHoverNodeActive ||
      (!cameraLod.moving && !suppressHoverInteractions && !navigationSettlingActive)
    ) {
      return;
    }
    pendingCursorPositionRef.current = null;
    clearHoverIntentTimer();
    setHoveredLink(null);
    onNodeHover(null);
  }, [
    cameraLod.moving,
    clearHoverIntentTimer,
    navigationSettlingActive,
    onNodeHover,
    selectedContextHoverNodeActive,
    suppressHoverInteractions,
  ]);

  // Keep the WebGL canvas locked to the actual flex/card size, including after data loads.
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const fallbackHeight = isFullscreen ? window.innerHeight : Math.max(420, window.innerHeight - 72);
        const bounds = containerRef.current.getBoundingClientRect();
        const measuredHeight = bounds.height || containerRef.current.clientHeight || fallbackHeight;
        setDimensions({
          width: Math.max(320, Math.floor(bounds.width || containerRef.current.clientWidth || 800)),
          height: Math.max(360, Math.floor(measuredHeight)),
        });
      }
    };
    updateDimensions();
    const resizeObserver = typeof ResizeObserver !== 'undefined' && containerRef.current
      ? new ResizeObserver(updateDimensions)
      : null;
    if (containerRef.current) {
      resizeObserver?.observe(containerRef.current);
    }
    window.addEventListener('resize', updateDimensions);
    return () => {
      resizeObserver?.disconnect();
      window.removeEventListener('resize', updateDimensions);
    };
  }, [isFullscreen]);

  // Prepare graph data for the Three.js force graph.
  const graphData = useMemo(() => {
    const anchorDeterministicLayout = true;
    return {
      nodes: nodes.map((node) => {
        const metrics = topology.metrics.get(node.id) || { links: 0, relations: 0 };
        const seed = topology.positions.get(node.id);
        const occurrenceBoost = Math.min(2, Math.log2(Math.max(1, node.occurrence || 0) + 1) * 0.75);
        const relationBoost = Math.log2(metrics.relations + 1) * 1.1;
        const seededPosition = seed || { x: 0, y: 0, z: 0 };
        const maxNodeSize = nodes.length > 3000 ? 8 : nodes.length > 1200 ? 9.5 : nodes.length > 350 ? 11.5 : 16;
        const minNodeSize = nodes.length > 3000 ? 2.8 : nodes.length > 1200 ? 3.2 : nodes.length > 350 ? 3.8 : 4.5;
        return {
          ...node,
          ...seededPosition,
          ...(anchorDeterministicLayout ? {
            fx: seededPosition.x,
            fy: seededPosition.y,
            fz: seededPosition.z,
          } : {}),
          val: Math.max(minNodeSize, Math.min(maxNodeSize, minNodeSize + relationBoost + occurrenceBoost)),
        };
      }),
      links: edges.map(e => ({
        id: e.id,
        source: e.source,
        target: e.target,
        label: e.label,
        type: e.type || e.label,
        weight: e.weight,
        relationCount: e.relationCount || 1,
        properties: e.properties || {},
      })),
    };
    }, [nodes, edges, topology]);

    const graphNodeById = useMemo(() => new Map(graphData.nodes.map((node) => [node.id, node])), [graphData.nodes]);

    const focusContextNodeIds = useMemo(() => {
      const query = searchTerm.trim();
      const hasIntentionalFocus = Boolean(
        activeNodeIds.size > 0 &&
        (selectedNode || selectedEdge || hoveredNode || query)
      );
      if (!hasIntentionalFocus) return new Set<string>();

      const budget = getGraphFocusContextBudget(
        renderProfile.tier,
        visualCameraLod.zoomTier,
        activeNodeIds.size,
        Boolean(query),
        Boolean(selectedEdge)
      );
      if (budget <= 0) return new Set<string>();

        const contextBoosts = new Map<string, number>();
        if (query) {
            graphData.links.forEach((edge) => {
            const sourceId = getLinkEndpointId(edge.source);
            const targetId = getLinkEndpointId(edge.target);
            if (!sourceId || !targetId) return;
            const sourceActive = activeNodeIds.has(sourceId);
            const targetActive = activeNodeIds.has(targetId);
            const edgeActive = activeEdgeIds.has(edge.id);
            if (!sourceActive && !targetActive && !edgeActive) return;
            const relationStrength = Math.max(1, Number(edge.relationCount || 1)) * 6 + Math.max(0, Number(edge.weight || 0));
            const addContextBoost = (id: string) => {
              if (activeNodeIds.has(id) || selectedEdgeEndpointIds.has(id)) return;
              contextBoosts.set(id, Math.max(contextBoosts.get(id) || 0, 1000 + relationStrength));
            };
            if (sourceActive) addContextBoost(targetId);
            if (targetActive) addContextBoost(sourceId);
            if (edgeActive) {
              addContextBoost(sourceId);
              addContextBoost(targetId);
            }
          });
        }

        return new Set(
          graphData.nodes
            .filter((node) => {
              if (activeNodeIds.has(node.id) || selectedEdgeEndpointIds.has(node.id)) return false;
              const { depthBand, offAxis } = getNodeCameraBand(node, visualCameraLod);
              return !offAxis && depthBand !== 'back';
            })
            .sort((a, b) =>
              (contextBoosts.get(b.id) || 0) - (contextBoosts.get(a.id) || 0) ||
              getNodeImportanceScore(b, nodeRelationSummary.get(b.id)) -
              getNodeImportanceScore(a, nodeRelationSummary.get(a.id)) ||
              a.label.localeCompare(b.label)
            )
          .slice(0, budget)
          .map((node) => node.id)
      );
    }, [
        activeNodeIds,
        activeEdgeIds,
        getLinkEndpointId,
          graphData.links,
        graphData.nodes,
        hoveredNode,
        nodeRelationSummary,
      renderProfile.tier,
      searchTerm,
      selectedEdge,
      selectedEdgeEndpointIds,
      selectedNode,
      visualCameraLod,
    ]);

    const focusHaloLabelHints = useMemo(() => {
      if (detailMode === 'performance' || selectedEdge || activeNodeIds.size <= 1) {
        return new Map<string, GraphFocusHaloLabelHint>();
      }

      const centerId = selectedHoverRouteActive && hoveredNode
        ? hoveredNode
        : focusNodeId || selectedNode || hoveredNode;
      const hasIntentionalFocus = Boolean(
        centerId ||
        pathFocusNodeIds.length > 0 ||
        searchTerm.trim()
      );
      if (!hasIntentionalFocus) return new Map<string, GraphFocusHaloLabelHint>();

      const baseBudget = getGraphFocusHaloLabelBudget(renderProfile.tier, visualCameraLod.zoomTier);

      const focusSizeAdjustedBudget = activeNodeIds.size > 90
        ? Math.min(baseBudget, 2)
        : activeNodeIds.size > 42
          ? Math.min(baseBudget, 3)
          : baseBudget;
      if (focusSizeAdjustedBudget <= 0) return new Map<string, GraphFocusHaloLabelHint>();

      const relationshipScore = new Map<string, number>();
      const relationshipHints = new Map<string, GraphFocusHaloLabelHint & { score: number }>();
      if (centerId) {
        graphData.links.forEach((link) => {
          const source = link.source as string | RenderGraphNode | undefined;
          const target = link.target as string | RenderGraphNode | undefined;
          const sourceId = typeof source === 'string' ? source : source?.id;
          const targetId = typeof target === 'string' ? target : target?.id;
          if (sourceId !== centerId && targetId !== centerId) return;

          const otherId = sourceId === centerId ? targetId : sourceId;
          if (!otherId || !activeNodeIds.has(otherId)) return;
          const relationTypes = getLinkRelationTypeCounts(link, 1);
          const type = relationTypes[0]?.type || normalizeType(link.type || link.label || 'related');
          const count = Math.max(1, Number(relationTypes[0]?.count || link.relationCount || 1));
          const score = count * 12 + Math.max(0, Number(link.weight || 0));
          relationshipScore.set(otherId, (relationshipScore.get(otherId) || 0) + score);
          const existing = relationshipHints.get(otherId);
          if (!existing || score > existing.score) {
            relationshipHints.set(otherId, {
              color: getRelationshipColor(type),
              count,
              score,
              type,
            });
          }
        });
      }

      const candidates = graphData.nodes
        .filter((node) =>
          activeNodeIds.has(node.id) &&
          node.id !== centerId &&
          node.id !== selectedNode &&
          node.id !== hoveredNode &&
          !selectedEdgeEndpointIds.has(node.id)
        )
        .sort((a, b) =>
          (relationshipScore.get(b.id) || 0) - (relationshipScore.get(a.id) || 0) ||
          getNodeImportanceScore(b, nodeRelationSummary.get(b.id)) -
            getNodeImportanceScore(a, nodeRelationSummary.get(a.id)) ||
          a.label.localeCompare(b.label)
        );

      const hints = new Map<string, GraphFocusHaloLabelHint>();
      candidates.slice(0, focusSizeAdjustedBudget).forEach((node) => {
        const relationHint = relationshipHints.get(node.id);
        hints.set(node.id, relationHint
          ? {
            color: relationHint.color,
            count: relationHint.count,
            type: relationHint.type,
          }
          : {
            color: accentColors.slate,
            count: Math.max(1, Math.round((relationshipScore.get(node.id) || 12) / 12)),
            type: 'related',
          });
      });
      return hints;
    }, [
      activeNodeIds,
      detailMode,
      focusNodeId,
      graphData.links,
      graphData.nodes,
      hoveredNode,
      nodeRelationSummary,
      pathFocusNodeIds.length,
      renderProfile.tier,
      searchTerm,
      selectedEdge,
      selectedEdgeEndpointIds,
      selectedHoverRouteActive,
      selectedNode,
      visualCameraLod.zoomTier,
    ]);

      const focusHaloLabelIds = useMemo(
        () => new Set(focusHaloLabelHints.keys()),
        [focusHaloLabelHints]
      );

      const activeIdentityNodeIds = useMemo(() => {
        const query = searchTerm.trim().toLowerCase();
        const hasActiveInspection = Boolean(selectedNode || selectedEdge || hoveredNode || query);
        if (!hasActiveInspection || detailMode === 'performance') return new Set<string>();

        const forcedIds = new Set(
          [selectedNode, hoveredNode, focusNodeId, ...selectedEdgeEndpointIds]
            .filter((id): id is string => Boolean(id))
        );
        const candidateIds = new Set<string>(visualActiveNodeIds);
        forcedIds.forEach((id) => candidateIds.add(id));
        if (candidateIds.size === 0) return new Set<string>();

        const limit = Math.max(
          forcedIds.size,
          getGraphActiveIdentityLimit(
            renderProfile.tier,
            visualCameraLod.zoomTier,
            dimensions.width,
            candidateIds.size
          )
        );
        const chosen = new Set<string>();
        graphData.nodes
          .filter((node) =>
            candidateIds.has(node.id) &&
            (
              forcedIds.has(node.id) ||
              getSelectionContextLabelQualityScore(node, nodeRelationSummary.get(node.id)) >= 120
            )
          )
          .sort((a, b) => {
            const scoreNode = (node: RenderGraphNode) => {
              const relationSummary = nodeRelationSummary.get(node.id);
              return (
                (node.id === selectedNode ? 100000 : 0) +
                (node.id === hoveredNode ? 90000 : 0) +
                (node.id === focusNodeId ? 80000 : 0) +
                (selectedEdgeEndpointIds.has(node.id) ? 70000 : 0) +
                (query && getEntitySearchText(node).includes(query) ? 60000 : 0) +
                (visualActiveNodeIds.has(node.id) ? 3000 : 0) +
                Math.max(
                  getNodeImportanceScore(node, relationSummary),
                  getSelectionContextLabelQualityScore(node, relationSummary)
                )
              );
            };
            return scoreNode(b) - scoreNode(a) || a.label.localeCompare(b.label);
          })
          .slice(0, limit)
          .forEach((node) => chosen.add(node.id));
        forcedIds.forEach((id) => {
          if (graphNodeById.has(id)) chosen.add(id);
        });
        return chosen;
      }, [
        detailMode,
        dimensions.width,
        focusNodeId,
        graphData.nodes,
        graphNodeById,
        hoveredNode,
        nodeRelationSummary,
        renderProfile.tier,
        searchTerm,
        selectedEdge,
        selectedEdgeEndpointIds,
        selectedNode,
        visualActiveNodeIds,
        visualCameraLod.zoomTier,
      ]);

      useEffect(() => {
        if (componentRegionGroupRef.current) {
        const previous = componentRegionGroupRef.current;
        previous.parent?.remove(previous);
        disposeGraphSceneObject(previous);
        componentRegionGroupRef.current = null;
      }
      return undefined;
    }, [graphData.nodes.length]);

    const getSelectionScreenAnchor = useCallback((): GraphSelectionAnchor | null => {
    const graph = fgRef.current;
    const camera = graph?.camera() as THREE.PerspectiveCamera | undefined;
    if (!camera || dimensions.width <= 0 || dimensions.height <= 0) return null;

    if (selectedNode) {
      const node = graphNodeById.get(selectedNode);
      const screenPoint = node ? projectGraphPointToScreen(node, camera, dimensions.width, dimensions.height) : null;
      if (!screenPoint) return null;
      return {
        key: `node:${selectedNode}`,
        kind: 'node',
        x: screenPoint.x,
        y: screenPoint.y,
        depth: screenPoint.depth,
        width: dimensions.width,
        height: dimensions.height,
      };
    }

    if (selectedEdge) {
      const edge = graphData.links.find((item) => item.id === selectedEdge);
      if (!edge) return null;
      const sourceEndpoint = edge.source as string | RenderGraphNode;
      const targetEndpoint = edge.target as string | RenderGraphNode;
      const sourceId = typeof sourceEndpoint === 'string' ? sourceEndpoint : sourceEndpoint?.id;
      const targetId = typeof targetEndpoint === 'string' ? targetEndpoint : targetEndpoint?.id;
      const source = sourceId ? graphNodeById.get(sourceId) || (typeof sourceEndpoint === 'string' ? null : sourceEndpoint) : null;
      const target = targetId ? graphNodeById.get(targetId) || (typeof targetEndpoint === 'string' ? null : targetEndpoint) : null;
      if (!source || !target) return null;
      const midpoint = {
        x: (Number(source.x) + Number(target.x)) / 2,
        y: (Number(source.y) + Number(target.y)) / 2,
        z: (Number(source.z) + Number(target.z)) / 2,
      };
      const screenPoint = projectGraphPointToScreen(midpoint, camera, dimensions.width, dimensions.height);
      if (!screenPoint) return null;
      return {
        key: `edge:${selectedEdge}`,
        kind: 'edge',
        x: screenPoint.x,
        y: screenPoint.y,
        depth: screenPoint.depth,
        width: dimensions.width,
        height: dimensions.height,
      };
    }

    return null;
  }, [dimensions.height, dimensions.width, graphData.links, graphNodeById, selectedEdge, selectedNode]);

  useEffect(() => {
    if (!onSelectionAnchorChange) return undefined;
    if (!selectedNode && !selectedEdge) {
      onSelectionAnchorChange(null);
      return undefined;
    }

    let frameId = 0;
    let active = true;
    let lastPublishedAt = 0;
    const publish = () => {
      onSelectionAnchorChange(getSelectionScreenAnchor());
    };
    const publishLoop = (timestamp: number) => {
      if (!active) return;
      const fastTracking =
        cameraLodRef.current.moving ||
        window.performance.now() - lastUserCameraInteractionRef.current < 2600;
      const publishInterval = fastTracking ? 64 : 220;
      if (timestamp - lastPublishedAt >= publishInterval) {
        lastPublishedAt = timestamp;
        publish();
      }
      frameId = window.requestAnimationFrame(publishLoop);
    };

    publish();
    frameId = window.requestAnimationFrame((timestamp) => {
      lastPublishedAt = timestamp - 1000;
      publishLoop(timestamp);
    });

    return () => {
      active = false;
      window.cancelAnimationFrame(frameId);
    };
  }, [cameraLod.key, cameraLod.moving, getSelectionScreenAnchor, onSelectionAnchorChange, selectedEdge, selectedNode]);

  const screenReadableLabelLayouts = useMemo(() => {
    const forcedReadableLabelIntent = Boolean(selectedNode || selectedEdge || hoveredNode || searchTerm.trim());
    if (!showLabels && !forcedReadableLabelIntent) {
      stableLabelLayoutsRef.current.clear();
      return new Map<string, GraphNodeLabelLayout>();
    }

      const navigationRetainLimit = getGraphNavigationRetainLimit(renderProfile.tier, dimensions.width);

    if (navigationSettlingActive && stableLabelLayoutsRef.current.size > 0) {
      const retainedLayouts = new Map<string, GraphNodeLabelLayout>();
      stableLabelLayoutsRef.current.forEach((layout, nodeId) => {
        const compactRelationshipEndpoint = compactSelectedEdgeEndpointLabels && selectedEdgeEndpointIds.has(nodeId);
        if (!compactRelationshipEndpoint && (layout.pinned || selectedNode === nodeId || selectedEdgeEndpointIds.has(nodeId))) {
          retainedLayouts.set(nodeId, layout);
        }
      });

      if (showLabels && detailMode !== 'performance' && !searchTerm.trim()) {
        let retainedAmbientLabels = 0;
        stableLabelLayoutsRef.current.forEach((layout, nodeId) => {
          if (retainedLayouts.has(nodeId) || layout.pinned) return;
          if (retainedAmbientLabels >= navigationRetainLimit) return;
          retainedLayouts.set(nodeId, layout);
          retainedAmbientLabels += 1;
        });
      }

      if (retainedLayouts.size > 0) return retainedLayouts;
    }

    const layouts = new Map<string, GraphNodeLabelLayout>();
    const layoutVersionSalt = graphLayoutVersion * 0;
    if (dimensions.width < 360 || dimensions.height < 320) return layouts;

    const camera = new THREE.PerspectiveCamera(45, dimensions.width / Math.max(1, dimensions.height), 1, 12000);
    camera.position.set(visualCameraLod.position.x, visualCameraLod.position.y, visualCameraLod.position.z);
    camera.lookAt(new THREE.Vector3(visualCameraLod.target.x, visualCameraLod.target.y, visualCameraLod.target.z));
    camera.updateMatrixWorld();
    camera.updateProjectionMatrix();

    const query = searchTerm.trim().toLowerCase();
    const priorityLabelQualityFloor = getKnowledgeGraphPriorityLabelQualityFloor(renderProfile.tier);
    const focusedNodeLimit = renderProfile.tier === 'massive'
      ? 44
      : renderProfile.tier === 'large'
        ? 64
        : renderProfile.tier === 'standard'
          ? 96
          : 140;
    const labelBudget = renderProfile.labelBudget[visualCameraLod.zoomTier];
    const baseNormalLabelLimit = Number.isFinite(labelBudget)
      ? Math.max(3, Number(labelBudget))
      : nodes.length <= 36
        ? nodes.length
        : visualCameraLod.zoomTier === 'near'
          ? 92
          : visualCameraLod.zoomTier === 'middle'
            ? 56
            : 34;
    const navigationLabelsSettling = navigationSettlingActive;
    const settledAmbientLabelLimit = showLabels
      ? getGraphAmbientLabelLimit(
        detailMode,
        labelDensity,
        renderProfile.tier,
        visualCameraLod.zoomTier,
        dimensions.width,
        baseNormalLabelLimit,
        nodes.length
      )
      : 0;
      const navigationAmbientLabelLimit = showLabels && navigationLabelsSettling && detailMode !== 'performance'
        ? Math.min(
          navigationRetainLimit,
          settledAmbientLabelLimit
        )
      : 0;
    const normalLabelLimit = navigationLabelsSettling
      ? navigationAmbientLabelLimit
      : settledAmbientLabelLimit;
    const focusedNeighborhoodBaseLimit = renderProfile.tier === 'massive'
      ? 8
      : renderProfile.tier === 'large'
        ? 11
        : renderProfile.tier === 'standard'
          ? 14
          : 18;
    const focusContextLabelsEnabled = showLabels || Boolean(selectedNode || selectedEdge || query);
    const focusedNeighborhoodLabelLimit = focusContextLabelsEnabled &&
      activeNodeIds.size > 0 &&
      activeNodeIds.size <= focusedNodeLimit &&
      detailMode !== 'performance'
      ? Math.min(
        activeNodeIds.size,
        dimensions.width < 700
          ? Math.max(5, Math.round(focusedNeighborhoodBaseLimit * 0.78))
          : focusedNeighborhoodBaseLimit
      )
      : 0;
    const compactSelectedOverview = Boolean(
      selectedNode &&
      !selectedEdge &&
      !selectionExpanded &&
      !query &&
      activeNodeIds.size === 0 &&
      activeEdgeIds.size === 0
    );
    const selectionContextLabelLimit = showLabels &&
      !query &&
      Boolean(selectedNode || selectedEdge) &&
      !compactSelectedOverview &&
      detailMode !== 'performance'
      ? Math.min(
        normalLabelLimit > 0
          ? normalLabelLimit
          : Math.max(1, Math.min(nodes.length, prioritizedNodeIds.anchorLabelLimit || 0)),
        renderProfile.tier === 'massive'
          ? dimensions.width < 1200 ? 3 : 4
          : renderProfile.tier === 'large'
            ? dimensions.width < 1200 ? 4 : 6
            : dimensions.width < 700
              ? 3
              : dimensions.width < 1200
                ? 5
                : dimensions.width < 1600
                  ? 7
                  : 9
      )
      : 0;
    const ambientLabelFloorCap = labelDensity === 'focus'
      ? Math.max(0, prioritizedNodeIds.anchorLabelLimit || 0)
      : normalLabelLimit;
    const minimumAmbientLabelLimit = showLabels &&
      !query &&
      (!selectedNode || compactSelectedOverview) &&
      !selectedEdge &&
      activeNodeIds.size === 0 &&
      activeEdgeIds.size === 0 &&
      detailMode !== 'performance' &&
      ambientLabelFloorCap > 0
      ? Math.min(
        ambientLabelFloorCap,
        nodes.length,
          dimensions.width < 700
            ? labelDensity === 'focus' ? 6 : labelDensity === 'rich' ? 10 : 8
            : dimensions.width < 1200
              ? labelDensity === 'focus' ? 10 : labelDensity === 'rich' ? 18 : 16
              : labelDensity === 'focus' ? 14 : labelDensity === 'rich' ? 30 : 22
      )
      : 0;
    const reservedRects: GraphScreenRect[] = [
      {
        left: 0,
        top: 0,
        right: Math.min(dimensions.width, dimensions.width < 700 ? 72 : 86),
        bottom: dimensions.width < 700 ? 70 : 82,
      },
      { left: Math.max(0, dimensions.width - 126), top: 0, right: dimensions.width, bottom: 58 },
      { left: 0, top: Math.max(0, dimensions.height - 64), right: Math.min(dimensions.width, 76), bottom: dimensions.height },
      { left: Math.max(0, dimensions.width - 72), top: Math.max(0, dimensions.height - 126), right: dimensions.width, bottom: dimensions.height },
      ...overlayReservedRects,
    ];

    type LabelCandidate = {
      anchor: boolean;
      focused: boolean;
      highlighted: boolean;
      focusHalo: boolean;
      relationshipEndpoint: boolean;
      selectionContext: boolean;
      node: RenderGraphNode;
      pinned: boolean;
      preferredAnchor: GraphNodeLabelAnchor;
      searchMatch: boolean;
      screenPoint: { x: number; y: number; depth: number };
      score: number;
      size: { width: number; height: number };
    };

  const searchLabelLimit = query
      ? dimensions.width < 700
        ? labelDensity === 'rich' ? 7 : 5
      : labelDensity === 'rich'
        ? renderProfile.tier === 'massive'
          ? 4
          : renderProfile.tier === 'large'
            ? 6
            : 9
        : renderProfile.tier === 'massive'
          ? 3
          : renderProfile.tier === 'large'
            ? 4
            : 6
    : Number.POSITIVE_INFINITY;
    const candidates = graphData.nodes
      .map((node): LabelCandidate | null => {
        const isSelected = selectedNode === node.id;
        const isHovered = hoveredNode === node.id;
        const isSearchMatch = Boolean(query && getEntitySearchText(node).includes(query));
        const isSearchPreviewFocus = Boolean(query && focusNodeId === node.id && isSearchMatch);
        const isRelationshipEndpoint = selectedEdgeEndpointIds.has(node.id);
        const isFocused = activeNodeIds.has(node.id);
        const isFocusedNeighbor = isFocused && activeNodeIds.size > 0 && activeNodeIds.size <= focusedNodeLimit;
        const isFocusHaloLabel = focusHaloLabelIds.has(node.id);
        const focusedSearchMatch = Boolean(
          isSearchMatch &&
          (isSelected || isHovered || isRelationshipEndpoint || isSearchPreviewFocus)
        );
        const isolatedSelectedNodeVisual = Boolean(
          isSelected &&
          selectedNode &&
          !selectedEdge &&
          activeEdgeIds.size === 0 &&
          !query
        );
        const compactSelectedFocusLabel = Boolean(
          isSelected &&
          compactSelectedNodeLabel &&
          !isHovered &&
          !query &&
          !isolatedSelectedNodeVisual
        );
        const compactSearchPreviewLabel = Boolean(isSearchPreviewFocus && !isSelected && !isHovered && !isRelationshipEndpoint);
        const pinned = isSelected || isHovered || focusedSearchMatch || isRelationshipEndpoint;
        const hasCommittedSelectionOverlay = (isSelected || isHovered) && !query;
        const compactRelationshipEndpoint = Boolean(
          isRelationshipEndpoint &&
          compactSelectedEdgeEndpointLabels &&
          !isHovered &&
          !query
        );
        const relationSummary = nodeRelationSummary.get(node.id);
        const labelQualityScore = getSelectionContextLabelQualityScore(node, relationSummary);
        const selectionContextLabelQualityScore = selectionContextLabelLimit > 0
          ? labelQualityScore
          : Number.NEGATIVE_INFINITY;
        const searchCandidate = Boolean(
          showLabels &&
          !navigationLabelsSettling &&
          isSearchMatch &&
          (isFocused || prioritizedNodeIds.labels.has(node.id) || nodes.length <= 180)
        );
        const focusHaloCandidate = Boolean(
          isFocusHaloLabel &&
          !navigationLabelsSettling &&
          detailMode !== 'performance'
        );
        const anchorLabelCandidate = Boolean(
          showLabels &&
          labelDensity === 'focus' &&
          !navigationLabelsSettling &&
          !query &&
          !selectedNode &&
          !selectedEdge &&
          activeNodeIds.size === 0 &&
          activeEdgeIds.size === 0 &&
          detailMode !== 'performance' &&
          prioritizedNodeIds.anchorLabels.has(node.id)
        );
        const focusedNeighborhoodLabelAllowed = Boolean(
          isFocusedNeighbor &&
          focusedNeighborhoodLabelLimit > 0
        );
        const selectionContextLabelAllowed = Boolean(
          selectionContextLabelLimit > 0 &&
          !pinned &&
          !isFocusedNeighbor &&
          !isRelationshipEndpoint &&
          !navigationLabelsSettling &&
          showLabels &&
          !query &&
          Boolean(selectedNode || selectedEdge) &&
          detailMode !== 'performance' &&
          selectionContextLabelQualityScore >= 120 &&
          (
            prioritizedNodeIds.fallbackLabels.has(node.id) ||
            prioritizedNodeIds.anchorLabels.has(node.id) ||
            prioritizedNodeIds.labels.has(node.id)
          )
        );
        const selectionFocusActive = Boolean(selectedNode || selectedEdge) && !query && !compactSelectedOverview;
        const priorityAmbientLabel = Boolean(
          labelDensity === 'priority' &&
          !pinned &&
          !searchCandidate &&
          !focusHaloCandidate &&
          !anchorLabelCandidate &&
          !selectionContextLabelAllowed &&
          !focusedNeighborhoodLabelAllowed &&
          !selectionFocusActive &&
          !query
        );
        if (priorityAmbientLabel && labelQualityScore < priorityLabelQualityFloor) return null;
        const ambientLabelAllowed = focusedNeighborhoodLabelAllowed || (showLabels && !selectionFocusActive && (
          !navigationLabelsSettling && (
            labelDensity !== 'focus' && (
              isFocusedNeighbor ||
              (labelDensity === 'rich' && renderProfile.tier === 'compact') ||
              prioritizedNodeIds.labels.has(node.id) ||
              normalLabelLimit > 0
            )
          )
        ));
        const labelBudgetAllows =
          pinned ||
          searchCandidate ||
          focusHaloCandidate ||
          anchorLabelCandidate ||
          selectionContextLabelAllowed ||
          ambientLabelAllowed;
        if (!labelBudgetAllows) return null;

        const { depthBand, offAxis } = getNodeCameraBand(node, visualCameraLod);
        const offAxisLabelAllowed = Boolean(
          prioritizedNodeIds.labels.has(node.id) ||
          isFocusedNeighbor ||
          isSearchMatch ||
          minimumAmbientLabelLimit > 0
        );
        if (!pinned && depthBand === 'back') return null;
        if (!pinned && offAxis && !offAxisLabelAllowed) return null;

        const screenPoint = projectNodeToScreen(node, camera, dimensions.width, dimensions.height);
        if (!screenPoint) return null;

        const previousStableLayout = stableLabelLayoutsRef.current.get(node.id);
        const style: GraphNodeLabelStyle = compactRelationshipEndpoint
          ? 'annotation'
          : compactSelectedFocusLabel
            ? 'annotation'
          : compactSearchPreviewLabel
            ? 'compact'
            : (isolatedSelectedNodeVisual || pinned || isRelationshipEndpoint)
            ? 'rich'
            : anchorLabelCandidate
              ? 'anchor'
            : isFocusHaloLabel
              ? 'neighbor'
            : focusedNeighborhoodLabelAllowed && Boolean(selectedNode || selectedEdge || query)
              ? 'annotation'
              : isFocusedNeighbor && labelDensity === 'focus'
                ? 'anchor'
              : 'compact';
        const size = estimateNodeLabelScreenSize(node, style, visualCameraLod.zoomTier);
        const computedPreferredAnchor: GraphNodeLabelAnchor = hasCommittedSelectionOverlay
          ? 'above'
          : compactSearchPreviewLabel
            ? screenPoint.y > dimensions.height * 0.38 ? 'above' : 'below'
          : focusedNeighborhoodLabelAllowed && screenPoint.x > dimensions.width * 0.68
            ? 'left'
          : focusedNeighborhoodLabelAllowed && screenPoint.x < dimensions.width * 0.32
            ? 'right'
          : compactRelationshipEndpoint || (!pinned && !isRelationshipEndpoint)
            ? screenPoint.y > dimensions.height * 0.18 ? 'above' : 'below'
            : screenPoint.y > dimensions.height * 0.54
              ? 'above'
              : 'below';
        const preferredAnchor = previousStableLayout?.anchor || computedPreferredAnchor;
        const depthScore = depthBand === 'front' ? 28 : depthBand === 'middle' ? 10 : -18;
        const rawImportanceScore = getNodeImportanceScore(node, relationSummary);
        const importanceScore = selectionContextLabelAllowed
          ? selectionContextLabelQualityScore
          : priorityAmbientLabel
            ? Math.max(rawImportanceScore, labelQualityScore) + Math.max(0, labelQualityScore) * 0.35
            : rawImportanceScore;
        const stabilityScore = previousStableLayout
          ? previousStableLayout.pinned || pinned
            ? GRAPH_LABEL_STABLE_PINNED_SCORE
            : GRAPH_LABEL_STABLE_AMBIENT_SCORE
          : 0;
        const score =
          (pinned ? 100000 : 0) +
          (isFocusedNeighbor ? 9000 : 0) +
          stabilityScore +
          importanceScore +
          depthScore -
          Math.max(0, screenPoint.depth) * 14 +
          layoutVersionSalt;

        return {
          anchor: anchorLabelCandidate,
          focused: isFocusedNeighbor,
          focusHalo: isFocusHaloLabel,
          highlighted: isSelected || isHovered || focusedSearchMatch,
          relationshipEndpoint: isRelationshipEndpoint,
          selectionContext: selectionContextLabelAllowed,
          node,
          pinned,
          searchMatch: isSearchMatch,
          preferredAnchor,
          screenPoint,
          score,
          size,
        };
      })
      .filter((item): item is LabelCandidate => Boolean(item))
      .sort((a, b) =>
        Number(b.pinned) - Number(a.pinned) ||
        b.score - a.score ||
        a.node.label.localeCompare(b.node.label)
      );

    let acceptedNormalLabels = 0;
    let acceptedFocusHaloLabels = 0;
    let acceptedSearchLabels = 0;
    let acceptedAnchorLabels = 0;
    let acceptedSelectionContextLabels = 0;
    const acceptedAmbientLabelPoints: Array<{ radius: number; x: number; y: number }> = [];
    const acceptedAnchorLabelPoints: Array<{ radius: number; x: number; y: number }> = [];
    const acceptedFocusedLabelPoints: Array<{ x: number; y: number }> = [];
    const anchorLabelLimit = prioritizedNodeIds.anchorLabelLimit || 0;
    const ambientLabelMinDistance = dimensions.width < 700
      ? 74
      : dimensions.width < 1300
        ? 104
        : 132;
    const anchorLabelMinDistance = dimensions.width < 900
      ? 108
      : dimensions.width < 1300
        ? 128
        : 148;
    const activeFocusLabelSpacing = Boolean(selectedNode || selectedEdge || query);
    const focusedLabelMinDistance = activeFocusLabelSpacing
      ? dimensions.width < 700
        ? 70
        : dimensions.width < 1300
          ? 86
          : 104
      : dimensions.width < 700
        ? 88
        : dimensions.width < 1300
          ? 112
          : 136;
    candidates.forEach((candidate) => {
      const limitedSearchLabel = Boolean(query && candidate.searchMatch && !candidate.pinned);
      const focusHaloLabel = Boolean(candidate.focusHalo && !candidate.pinned);
      const anchorLabel = Boolean(candidate.anchor && !candidate.pinned);
      const focusedNeighborhoodLabel = Boolean(candidate.focused && !candidate.pinned && focusedNeighborhoodLabelLimit > 0);
      const selectionContextLabel = Boolean(candidate.selectionContext && !candidate.pinned);
      const ambientLimit = Math.max(
        normalLabelLimit,
        focusedNeighborhoodLabel ? focusedNeighborhoodLabelLimit : 0,
        anchorLabel ? anchorLabelLimit : 0,
        selectionContextLabel ? selectionContextLabelLimit : 0
      );
      if (limitedSearchLabel && acceptedSearchLabels >= searchLabelLimit) return;
      if (focusHaloLabel && acceptedFocusHaloLabels >= focusHaloLabelIds.size) return;
      if (anchorLabel && acceptedAnchorLabels >= anchorLabelLimit) return;
      if (selectionContextLabel && acceptedSelectionContextLabels >= selectionContextLabelLimit) return;
      if (
        anchorLabel
      ) {
        const candidateAnchorRadius = Math.max(
          anchorLabelMinDistance * 0.52,
          Math.min(260, candidate.size.width * 0.68)
        );
        if (
          acceptedAnchorLabelPoints.some((point) =>
            Math.hypot(point.x - candidate.screenPoint.x, point.y - candidate.screenPoint.y) <
            point.radius + candidateAnchorRadius
          )
        ) {
          return;
        }
      }
      if (
        focusedNeighborhoodLabel &&
        acceptedFocusedLabelPoints.some((point) =>
          Math.hypot(point.x - candidate.screenPoint.x, point.y - candidate.screenPoint.y) < focusedLabelMinDistance
        )
      ) {
        return;
      }
      if (
        !candidate.pinned &&
        !limitedSearchLabel &&
        !focusHaloLabel &&
        !anchorLabel &&
        !focusedNeighborhoodLabel
      ) {
        const candidateAmbientRadius = Math.max(
          ambientLabelMinDistance * 0.48,
          Math.min(240, candidate.size.width * 0.64)
        );
        if (
          acceptedAmbientLabelPoints.some((point) =>
            Math.hypot(point.x - candidate.screenPoint.x, point.y - candidate.screenPoint.y) <
            point.radius + candidateAmbientRadius
          )
        ) {
          return;
        }
      }
      if (!limitedSearchLabel && !focusHaloLabel && !candidate.pinned && !selectionContextLabel && acceptedNormalLabels >= ambientLimit) return;
      const layout = chooseNodeLabelLayout(
        candidate.screenPoint.x,
        candidate.screenPoint.y,
        candidate.size.width,
        candidate.size.height,
        dimensions.width,
        dimensions.height,
        reservedRects,
        candidate.preferredAnchor,
        candidate.pinned
      );
      if (!layout) return;
      if (!candidate.pinned && overlayReservedRects.some((rect) => graphRectsOverlap(layout.rect, rect))) return;
      const focusClearance = candidate.pinned
        ? visualCameraLod.zoomTier === 'near' ? 18 : visualCameraLod.zoomTier === 'middle' ? 15 : 12
        : candidate.focused
          ? visualCameraLod.zoomTier === 'near' ? 14 : 10
        : candidate.searchMatch
          ? 10
        : candidate.size.width > 190
          ? visualCameraLod.zoomTier === 'far' ? 34 : 28
          : candidate.size.width > 150
            ? visualCameraLod.zoomTier === 'far' ? 28 : 22
            : 16;
      const focusRadius = candidate.highlighted
        ? visualCameraLod.zoomTier === 'near' ? 38 : visualCameraLod.zoomTier === 'middle' ? 32 : 26
        : candidate.relationshipEndpoint
          ? visualCameraLod.zoomTier === 'near' ? 30 : 24
        : candidate.focused
          ? visualCameraLod.zoomTier === 'near' ? 22 : 16
          : 0;

      if (focusRadius > 0) {
        reservedRects.push({
          left: candidate.screenPoint.x - focusRadius,
          top: candidate.screenPoint.y - focusRadius,
          right: candidate.screenPoint.x + focusRadius,
          bottom: candidate.screenPoint.y + focusRadius,
        });
      }
      reservedRects.push(focusClearance > 0 ? padGraphRect(layout.rect, focusClearance) : layout.rect);
      layouts.set(candidate.node.id, {
        anchor: layout.anchor,
        pinned: candidate.pinned,
        selectionContext: selectionContextLabel,
        screenX: candidate.screenPoint.x,
        screenY: candidate.screenPoint.y,
      });
      if (limitedSearchLabel) {
        acceptedSearchLabels += 1;
      } else if (focusHaloLabel) {
        acceptedFocusHaloLabels += 1;
      } else if (!candidate.pinned) {
        if (!selectionContextLabel) {
          acceptedNormalLabels += 1;
        }
        if (anchorLabel) {
          acceptedAnchorLabels += 1;
          acceptedAnchorLabelPoints.push({
            radius: Math.max(anchorLabelMinDistance * 0.52, Math.min(260, candidate.size.width * 0.68)),
            x: candidate.screenPoint.x,
            y: candidate.screenPoint.y,
          });
        }
        if (selectionContextLabel) {
          acceptedSelectionContextLabels += 1;
        }
        if (focusedNeighborhoodLabel) {
          acceptedFocusedLabelPoints.push({ x: candidate.screenPoint.x, y: candidate.screenPoint.y });
        }
        if (!anchorLabel && !focusedNeighborhoodLabel) {
          acceptedAmbientLabelPoints.push({
            radius: Math.max(ambientLabelMinDistance * 0.48, Math.min(240, candidate.size.width * 0.64)),
            x: candidate.screenPoint.x,
            y: candidate.screenPoint.y,
          });
        }
      }
    });

    if (minimumAmbientLabelLimit > layouts.size) {
      type AmbientFallbackLabelCandidate = {
        node: (typeof graphData.nodes)[number];
        preferredAnchor: GraphNodeLabelAnchor;
        screenPoint: { x: number; y: number; depth: number };
        score: number;
      };
      const centerX = dimensions.width * 0.5;
      const centerY = dimensions.height * 0.48;
      graphData.nodes
        .filter((node) => !layouts.has(node.id))
        .map((node): AmbientFallbackLabelCandidate | null => {
          const screenPoint = projectNodeToScreen(node, camera, dimensions.width, dimensions.height);
          if (!screenPoint) return null;
          const { depthBand, offAxis } = getNodeCameraBand(node, visualCameraLod);
          const relationSummary = nodeRelationSummary.get(node.id);
          const labelQualityScore = getSelectionContextLabelQualityScore(node, relationSummary);
          if (labelDensity === 'priority' && labelQualityScore < priorityLabelQualityFloor) return null;
          const centerDistance = Math.hypot(screenPoint.x - centerX, screenPoint.y - centerY);
          const preferredAnchor: GraphNodeLabelAnchor = screenPoint.y > dimensions.height * 0.18 ? 'above' : 'below';
          const depthScore = depthBand === 'front' ? 72 : depthBand === 'middle' ? 32 : -120;
          return {
            node,
            preferredAnchor,
            screenPoint,
            score:
              Math.max(getNodeImportanceScore(node, relationSummary), labelQualityScore) +
              Math.max(0, labelQualityScore) * 0.35 +
              depthScore -
              (offAxis ? 90 : 0) -
              centerDistance * 0.07 -
              Math.max(0, screenPoint.depth) * 12,
          };
        })
        .filter((item): item is AmbientFallbackLabelCandidate => Boolean(item))
        .sort((a, b) => b.score - a.score || a.node.label.localeCompare(b.node.label))
        .some((candidate) => {
          if (layouts.size >= minimumAmbientLabelLimit) return true;
          const size = estimateNodeLabelScreenSize(candidate.node, 'compact', visualCameraLod.zoomTier);
          const layout = chooseNodeLabelLayout(
            candidate.screenPoint.x,
            candidate.screenPoint.y,
            size.width,
            size.height,
            dimensions.width,
            dimensions.height,
            reservedRects,
            candidate.preferredAnchor,
            false
          );
          if (!layout) return false;
          if (overlayReservedRects.some((rect) => graphRectsOverlap(layout.rect, rect))) return false;
          const fallbackClearance = size.width > 190
            ? visualCameraLod.zoomTier === 'far' ? 34 : 28
            : size.width > 150
              ? visualCameraLod.zoomTier === 'far' ? 28 : 22
              : 16;
          reservedRects.push(padGraphRect(layout.rect, fallbackClearance));
          layouts.set(candidate.node.id, {
            anchor: layout.anchor,
            pinned: false,
            screenX: candidate.screenPoint.x,
            screenY: candidate.screenPoint.y,
          });
          return false;
        });
    }

    if (
      focusContextLabelsEnabled &&
      layouts.size === 0 &&
      !query &&
      focusedNeighborhoodLabelLimit > 0 &&
      activeNodeIds.size > 0 &&
      activeNodeIds.size <= focusedNodeLimit &&
      detailMode !== 'performance'
    ) {
      type FocusedFallbackLabelCandidate = {
        node: (typeof graphData.nodes)[number];
        screenPoint: { x: number; y: number; depth: number };
        score: number;
      };
      graphData.nodes
        .filter((node) => activeNodeIds.has(node.id))
        .map((node): FocusedFallbackLabelCandidate | null => {
          const screenPoint = projectNodeToScreen(node, camera, dimensions.width, dimensions.height);
          if (!screenPoint) return null;
          const relationSummary = nodeRelationSummary.get(node.id);
          const isHovered = hoveredNode === node.id;
          const isSelected = selectedNode === node.id;
          return {
            node,
            screenPoint,
            score:
              (isHovered ? 100000 : 0) +
              (isSelected ? 90000 : 0) +
              (focusNodeId === node.id ? 80000 : 0) +
              getNodeImportanceScore(node, relationSummary) -
              Math.max(0, screenPoint.depth) * 12,
          };
        })
        .filter((item): item is FocusedFallbackLabelCandidate => Boolean(item))
        .sort((a, b) => b.score - a.score || a.node.label.localeCompare(b.node.label))
        .slice(0, focusedNeighborhoodLabelLimit)
        .forEach(({ node, screenPoint }) => {
          const size = estimateNodeLabelScreenSize(node, 'anchor', visualCameraLod.zoomTier);
          const preferredAnchor: GraphNodeLabelAnchor = screenPoint.x > dimensions.width * 0.62
            ? 'left'
            : screenPoint.x < dimensions.width * 0.38
              ? 'right'
              : screenPoint.y > dimensions.height * 0.52
                ? 'above'
                : 'below';
          const layout = chooseNodeLabelLayout(
            screenPoint.x,
            screenPoint.y,
            size.width,
            size.height,
            dimensions.width,
            dimensions.height,
            reservedRects,
            preferredAnchor,
            true
          );
          if (!layout) return;
          reservedRects.push(padGraphRect(layout.rect, 6));
          layouts.set(node.id, {
            anchor: layout.anchor,
            pinned: node.id === hoveredNode || node.id === selectedNode,
            screenX: screenPoint.x,
            screenY: screenPoint.y,
          });
        });
    }

    if (
      showLabels &&
      detailMode === 'detail' &&
      labelDensity !== 'focus' &&
      layouts.size === 0 &&
      normalLabelLimit > 1 &&
      nodes.length <= 90 &&
      !query &&
      !selectedNode &&
      !selectedEdge
    ) {
      type FallbackLabelCandidate = {
        node: (typeof graphData.nodes)[number];
        screenPoint: { x: number; y: number; depth: number };
        score: number;
      };
      const fallbackLabel = graphData.nodes
        .map((node): FallbackLabelCandidate | null => {
          const screenPoint = projectNodeToScreen(node, camera, dimensions.width, dimensions.height);
          if (!screenPoint) return null;
          const relationSummary = nodeRelationSummary.get(node.id);
          const centerDistance = Math.hypot(
            screenPoint.x - dimensions.width * 0.5,
            screenPoint.y - dimensions.height * 0.48
          );
          return {
            node,
            screenPoint,
            score:
              getNodeImportanceScore(node, relationSummary) -
              centerDistance * 0.08 -
              Math.max(0, screenPoint.depth) * 8,
          };
        })
        .filter((item): item is FallbackLabelCandidate => item !== null)
        .sort((a, b) => b.score - a.score || a.node.label.localeCompare(b.node.label))[0];

      if (fallbackLabel) {
        const size = estimateNodeLabelScreenSize(fallbackLabel.node, 'compact', visualCameraLod.zoomTier);
        const preferredAnchor: GraphNodeLabelAnchor = fallbackLabel.screenPoint.x > dimensions.width * 0.64 ? 'left' : 'right';
        const layout = chooseNodeLabelLayout(
          fallbackLabel.screenPoint.x,
          fallbackLabel.screenPoint.y,
          size.width,
          size.height,
          dimensions.width,
          dimensions.height,
          reservedRects,
          preferredAnchor,
          true
        );
        if (layout) {
          layouts.set(fallbackLabel.node.id, {
            anchor: layout.anchor,
            pinned: false,
            screenX: fallbackLabel.screenPoint.x,
            screenY: fallbackLabel.screenPoint.y,
          });
        }
      }
    }

    if (
      showLabels &&
      layouts.size === 0 &&
      stableLabelLayoutsRef.current.size > 0 &&
      !query &&
      navigationSettlingActive
    ) {
      return stableLabelLayoutsRef.current;
    }

    stableLabelLayoutsRef.current = layouts;
    return layouts;
  }, [
    activeEdgeIds.size,
    activeNodeIds,
    compactSelectedEdgeEndpointLabels,
    compactSelectedNodeLabel,
    detailMode,
    dimensions.height,
    dimensions.width,
    graphData,
    graphLayoutVersion,
    focusHaloLabelIds,
    focusNodeId,
    labelDensity,
    hoveredNode,
    navigationSettlingActive,
    nodeRelationSummary,
    nodes.length,
    overlayReservedRects,
    prioritizedNodeIds.anchorLabelLimit,
    prioritizedNodeIds.anchorLabels,
    prioritizedNodeIds.fallbackLabels,
    prioritizedNodeIds.labels,
    renderProfile.labelBudget,
    renderProfile.tier,
    searchTerm,
    selectedEdge,
    selectedEdgeEndpointIds,
    selectedNode,
    selectionExpanded,
    showLabels,
    visualCameraLod,
  ]);

  useEffect(() => {
    const labelAnimationActive = showLabels ||
      Boolean(selectedNode || selectedEdge || hoveredNode || searchTerm.trim()) ||
      navigationSettlingActive;
    if (!labelAnimationActive) return undefined;

    let frameId = 0;
    const animateLabelOpacity = () => {
      const now = window.performance.now();
      let keepAnimating = false;

      persistedLabelStateRef.current.forEach((label, nodeId) => {
        const cached = nodeObjectCacheRef.current.get(nodeId);
        if (!cached) return;

        const opacity = getGraphPersistedLabelOpacity(label, now);
        setGraphNodeObjectLabelOpacity(cached.object, opacity);
        if (label.pinned || label.visibleUntil > now || opacity > 0.01) {
          keepAnimating = true;
        }
      });

      if (keepAnimating || labelAnimationActive) {
        frameId = window.requestAnimationFrame(animateLabelOpacity);
      }
    };

    frameId = window.requestAnimationFrame(animateLabelOpacity);
    return () => {
      window.cancelAnimationFrame(frameId);
    };
  }, [
    hoveredNode,
    navigationSettlingActive,
    screenReadableLabelLayouts,
    searchTerm,
    selectedEdge,
    selectedNode,
    showLabels,
  ]);

    const cancelCameraFlight = useCallback((clearProgrammatic = true) => {
      cameraFlightTokenRef.current += 1;
      if (cameraFlightFrameRef.current !== null) {
        window.cancelAnimationFrame(cameraFlightFrameRef.current);
        cameraFlightFrameRef.current = null;
      }
      if (clearProgrammatic) {
        programmaticCameraUntilRef.current = 0;
      }
    }, []);

    useEffect(() => () => {
      cancelCameraFlight();
    }, [cancelCameraFlight]);

    const flyCamera = useCallback((
      position: THREE.Vector3,
      target: THREE.Vector3,
      duration = 720,
      options?: { force?: boolean; kind?: 'fit' | 'focus' | 'dive' | 'zoom' }
    ): boolean => {
      const graph = fgRef.current;
      const camera = graph?.camera() as THREE.PerspectiveCamera | undefined;
      const controls = graph?.controls?.() as { target?: THREE.Vector3; update?: () => void } | undefined;
      if (!graph || !camera) return false;

      const nextPosition = position.clone();
      const nextTarget = target.clone();
      if (
        !Number.isFinite(nextPosition.x) ||
        !Number.isFinite(nextPosition.y) ||
        !Number.isFinite(nextPosition.z) ||
        !Number.isFinite(nextTarget.x) ||
        !Number.isFinite(nextTarget.y) ||
        !Number.isFinite(nextTarget.z)
      ) {
        return false;
      }

      const now = window.performance.now();
      const currentTarget = controls?.target || new THREE.Vector3(0, 0, 0);
      const currentDistance = camera.position.distanceTo(currentTarget) || cameraLodRef.current.distance || 1;
      const cameraShift = camera.position.distanceTo(nextPosition);
      const targetShift = currentTarget.distanceTo(nextTarget);
      const settleEpsilon = Math.max(1.8, currentDistance * (nodes.length > 1200 ? 0.014 : 0.009));
      const signature = [
        Math.round(nextPosition.x / settleEpsilon),
        Math.round(nextPosition.y / settleEpsilon),
        Math.round(nextPosition.z / settleEpsilon),
        Math.round(nextTarget.x / settleEpsilon),
        Math.round(nextTarget.y / settleEpsilon),
        Math.round(nextTarget.z / settleEpsilon),
      ].join(':');
      const previousFlight = lastCameraFlightRef.current;

      if (
        !options?.force &&
        previousFlight?.signature === signature &&
        now - previousFlight.at < 520
      ) {
        return true;
      }

      if (
        !options?.force &&
        cameraShift < settleEpsilon &&
        targetShift < settleEpsilon * 0.72
      ) {
        lastCameraFlightRef.current = { signature, at: now };
        return true;
      }

      const movingNow = now < cameraMovingUntilRef.current;
      const travelRatio = Math.min(2.6, (cameraShift + targetShift * 0.55) / Math.max(32, currentDistance));
      const maximumDuration = options?.kind === 'zoom'
        ? 820
        : options?.kind === 'fit'
          ? 1380
          : options?.kind === 'dive'
            ? 1080
            : 1240;
      const adaptiveDuration = Math.round(Math.max(
        options?.kind === 'zoom' ? 420 : 620,
        Math.min(maximumDuration, duration * (movingNow ? 1.18 : 1) + travelRatio * 135)
      ));

      const startPosition = camera.position.clone();
      const startTarget = currentTarget.clone();
      const token = cameraFlightTokenRef.current + 1;
      const startTime = now;
      const scratchPosition = new THREE.Vector3();
      const scratchTarget = new THREE.Vector3();
      const easeCameraFlight = (value: number) => {
        const clamped = Math.max(0, Math.min(1, value));
        return clamped < 0.5
          ? 4 * clamped * clamped * clamped
          : 1 - Math.pow(-2 * clamped + 2, 3) / 2;
      };

      cancelCameraFlight(false);
      cameraFlightTokenRef.current = token;
      programmaticCameraUntilRef.current = now + adaptiveDuration + 320;
      lastCameraFlightRef.current = { signature, at: now };
      cameraMovingUntilRef.current = now + adaptiveDuration + 220;
      graph.resumeAnimation();
      const applyCameraFrame = (timestamp: number) => {
        if (cameraFlightTokenRef.current !== token) return;

        const progress = Math.max(0, Math.min(1, (timestamp - startTime) / Math.max(1, adaptiveDuration)));
        const eased = easeCameraFlight(progress);
        scratchPosition.lerpVectors(startPosition, nextPosition, eased);
        scratchTarget.lerpVectors(startTarget, nextTarget, eased);
        camera.position.copy(scratchPosition);
        if (controls?.target) {
          controls.target.copy(scratchTarget);
        } else {
          camera.lookAt(scratchTarget);
        }
        camera.updateMatrixWorld();
        controls?.update?.();
        graph.resumeAnimation();

        if (progress < 1) {
          cameraFlightFrameRef.current = window.requestAnimationFrame(applyCameraFrame);
          return;
        }

        camera.position.copy(nextPosition);
        if (controls?.target) {
          controls.target.copy(nextTarget);
        } else {
          camera.lookAt(nextTarget);
        }
        camera.updateMatrixWorld();
        controls?.update?.();
        cameraFlightFrameRef.current = null;
        programmaticCameraUntilRef.current = window.performance.now() + 180;
        cameraMovingUntilRef.current = window.performance.now() + 180;
      };

      cameraFlightFrameRef.current = window.requestAnimationFrame(applyCameraFrame);
      return true;
    }, [cancelCameraFlight, nodes.length]);

    const fitGraph = useCallback((duration = 640, options?: { lockDirection?: boolean }): boolean => {
      const graph = fgRef.current;
      if (!graph || nodes.length === 0) return false;

      const bbox = nodes.length > 1800 ? null : graph.getGraphBbox?.();
      const bboxCoordinates = bbox && [
        bbox.x[0],
        bbox.x[1],
        bbox.y[0],
        bbox.y[1],
        bbox.z[0],
        bbox.z[1],
      ].every(Number.isFinite)
        ? [
          { x: bbox.x[0], y: bbox.y[0], z: bbox.z[0] },
          { x: bbox.x[0], y: bbox.y[0], z: bbox.z[1] },
          { x: bbox.x[0], y: bbox.y[1], z: bbox.z[0] },
          { x: bbox.x[0], y: bbox.y[1], z: bbox.z[1] },
          { x: bbox.x[1], y: bbox.y[0], z: bbox.z[0] },
          { x: bbox.x[1], y: bbox.y[0], z: bbox.z[1] },
          { x: bbox.x[1], y: bbox.y[1], z: bbox.z[0] },
          { x: bbox.x[1], y: bbox.y[1], z: bbox.z[1] },
        ]
        : [];
      const nodeCoordinates = graphData.nodes
        .map((node) => ({
          x: Number(node.x),
          y: Number(node.y),
          z: Number(node.z),
        }))
        .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z));
      const coordinates = nodeCoordinates.length > 0 ? nodeCoordinates : bboxCoordinates;

      if (coordinates.length === 0) return false;

      const extents = coordinates.reduce(
        (bounds, point) => ({
          minX: Math.min(bounds.minX, point.x),
          maxX: Math.max(bounds.maxX, point.x),
          minY: Math.min(bounds.minY, point.y),
          maxY: Math.max(bounds.maxY, point.y),
          minZ: Math.min(bounds.minZ, point.z),
          maxZ: Math.max(bounds.maxZ, point.z),
        }),
        {
          minX: coordinates[0].x,
          maxX: coordinates[0].x,
          minY: coordinates[0].y,
          maxY: coordinates[0].y,
          minZ: coordinates[0].z,
          maxZ: coordinates[0].z,
        }
      );

      const center = {
        x: (extents.minX + extents.maxX) / 2,
        y: (extents.minY + extents.maxY) / 2,
        z: (extents.minZ + extents.maxZ) / 2,
      };
        const renderPadding = nodes.length > 3000 ? 96 : nodes.length > 1200 ? 72 : nodes.length > 350 ? 46 : nodes.length > 120 ? 30 : nodes.length <= 12 ? 20 : nodes.length <= 40 ? 24 : 28;
      const span = Math.max(
        extents.maxX - extents.minX,
        extents.maxY - extents.minY,
        extents.maxZ - extents.minZ,
        60
      ) + renderPadding;
      const camera = graph.camera() as THREE.PerspectiveCamera | undefined;
      const controls = graph.controls?.() as { target?: THREE.Vector3 } | undefined;
      const fov = ((camera?.fov || 45) * Math.PI) / 180;
      const aspect = dimensions.width / Math.max(1, dimensions.height);
      const portraitViewport = aspect < 1;
      const currentTarget = controls?.target || new THREE.Vector3(0, 0, 0);
      const currentViewDirection = camera
        ? camera.position.clone().sub(currentTarget)
        : new THREE.Vector3(0.62, -0.34, 1);
      if (currentViewDirection.lengthSq() === 0) currentViewDirection.set(0.62, -0.34, 1);
      currentViewDirection.normalize();
          const minimumDistance = nodes.length > 3000 ? 260 : nodes.length > 1200 ? 190 : nodes.length > 350 ? 112 : nodes.length > 120 ? 72 : nodes.length <= 12 ? 36 : nodes.length <= 40 ? 42 : 46;
        const viewPadding = nodes.length > 3000 ? 0.96 : nodes.length > 1200 ? 0.9 : nodes.length > 350 ? 0.78 : nodes.length > 120 ? 0.7 : nodes.length <= 12 ? 0.64 : nodes.length <= 40 ? 0.68 : 0.7;
      const aspectPadding = aspect < 1 ? 1.28 : 1;
      const maximumDistance = nodes.length > 3000 ? 6600 : nodes.length > 1200 ? 4200 : nodes.length > 350 ? 2600 : nodes.length > 120 ? 1600 : 920;
      const roughDistance = Math.max(
        minimumDistance,
        Math.min(maximumDistance, (span / (2 * Math.tan(fov / 2))) * viewPadding * aspectPadding)
      );
      const centerVector = new THREE.Vector3(center.x, center.y, center.z);
      const sampleStep = Math.max(1, Math.ceil(nodeCoordinates.length / 1800));
      const fitCoordinates = nodeCoordinates.length > 0
        ? nodeCoordinates.filter((_, index) => index % sampleStep === 0)
        : bboxCoordinates;
      const projectionCoordinates = fitCoordinates.length > 0 ? fitCoordinates : coordinates;
      const viewportPadding = nodes.length > 3000
        ? 18
        : nodes.length > 1200
          ? 20
          : nodes.length > 350
            ? 22
            : nodes.length > 120
              ? 28
              : nodes.length <= 40
                ? 34
                : 42;
      const innerBounds = portraitViewport
        ? {
          left: Math.min(dimensions.width * 0.08, viewportPadding),
          right: dimensions.width - Math.min(dimensions.width * 0.08, viewportPadding),
          top: Math.min(dimensions.height * 0.14, viewportPadding + 16),
          bottom: dimensions.height - Math.min(dimensions.height * 0.14, viewportPadding + 34),
        }
        : {
          left: Math.min(dimensions.width * 0.18, viewportPadding + 18),
          right: dimensions.width - Math.min(dimensions.width * 0.14, viewportPadding + 18),
          top: Math.min(dimensions.height * 0.16, viewportPadding + 16),
          bottom: dimensions.height - Math.min(dimensions.height * 0.18, viewportPadding + 44),
        };
      const tempCamera = new THREE.PerspectiveCamera(
        camera?.fov || 45,
        aspect,
        camera?.near || 1,
        camera?.far || 12000
      );
      const projectedPoint = new THREE.Vector3();
      const innerWidth = Math.max(1, innerBounds.right - innerBounds.left);
      const innerHeight = Math.max(1, innerBounds.bottom - innerBounds.top);
      const innerCenterX = (innerBounds.left + innerBounds.right) / 2;
      const innerCenterY = (innerBounds.top + innerBounds.bottom) / 2;
      const evaluateViewportFit = (viewDirection: THREE.Vector3, distance: number) => {
        tempCamera.position.copy(centerVector).addScaledVector(viewDirection, distance);
        tempCamera.lookAt(centerVector);
        tempCamera.updateMatrixWorld();
        tempCamera.updateProjectionMatrix();

        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;

        for (const point of projectionCoordinates) {
          projectedPoint.set(point.x, point.y, point.z).project(tempCamera);
          if (
            !Number.isFinite(projectedPoint.x) ||
            !Number.isFinite(projectedPoint.y) ||
            !Number.isFinite(projectedPoint.z) ||
            projectedPoint.z < -1 ||
            projectedPoint.z > 1
          ) {
            return null;
          }

          const screenX = (projectedPoint.x + 1) * 0.5 * dimensions.width;
          const screenY = (1 - projectedPoint.y) * 0.5 * dimensions.height;
          minX = Math.min(minX, screenX);
          maxX = Math.max(maxX, screenX);
          minY = Math.min(minY, screenY);
          maxY = Math.max(maxY, screenY);
        }

        const projectedWidth = Math.max(1, maxX - minX);
        const projectedHeight = Math.max(1, maxY - minY);
        const centerOffset =
          Math.abs((minX + maxX) / 2 - innerCenterX) / innerWidth +
          Math.abs((minY + maxY) / 2 - innerCenterY) / innerHeight;

        return {
          fits: minX >= innerBounds.left &&
            maxX <= innerBounds.right &&
            minY >= innerBounds.top &&
            maxY <= innerBounds.bottom,
          fillRatio: Math.min(1, (projectedWidth * projectedHeight) / (innerWidth * innerHeight)),
          widthRatio: Math.min(1, projectedWidth / innerWidth),
          heightRatio: Math.min(1, projectedHeight / innerHeight),
          centerOffset,
        };
      };
      const baseDirection = options?.lockDirection
        ? currentViewDirection
        : new THREE.Vector3(0.62, portraitViewport ? -0.08 : -0.34, 1).normalize();
      const wideViewport = aspect >= 1.35;
      const lateralReach = wideViewport ? 0.98 : 0.76;
      const azimuthDirections = Array.from({ length: wideViewport ? 12 : 8 }, (_, index) => {
        const angle = -0.36 + (index / (wideViewport ? 12 : 8)) * Math.PI * 2;
        const zLift = wideViewport
          ? (index % 3 === 0 ? 0.86 : index % 3 === 1 ? 1.04 : 1.2)
          : (index % 2 === 0 ? 0.94 : 1.12);
        return new THREE.Vector3(
          Math.cos(angle) * lateralReach,
          Math.sin(angle) * lateralReach,
          zLift
        ).normalize();
      });
      const candidateDirections = options?.lockDirection
        ? [baseDirection]
        : [
          baseDirection,
          new THREE.Vector3(0.38, -0.72, 1).normalize(),
          new THREE.Vector3(0.78, 0.14, 1).normalize(),
          new THREE.Vector3(-0.36, -0.42, 1).normalize(),
          new THREE.Vector3(-0.22, 0.68, 1).normalize(),
          ...azimuthDirections,
        ];
      const widthReward = wideViewport ? 0.42 : 0.16;
      const heightReward = wideViewport ? 0.08 : 0.18;
      const centerPenalty = wideViewport ? 0.12 : 0.46;
      const fitCandidates = candidateDirections.map((viewDirection, index) => {
        let lowDistance = minimumDistance;
        let highDistance = Math.min(maximumDistance, Math.max(roughDistance, minimumDistance * 1.4));
        let highEvaluation = evaluateViewportFit(viewDirection, highDistance);
        while ((!highEvaluation || !highEvaluation.fits) && highDistance < maximumDistance) {
          highDistance = Math.min(maximumDistance, highDistance * 1.35);
          highEvaluation = evaluateViewportFit(viewDirection, highDistance);
        }

        if (!highEvaluation || !highEvaluation.fits) {
          const fallbackEvaluation = evaluateViewportFit(viewDirection, roughDistance);
          return {
            direction: viewDirection,
            distance: roughDistance,
            fits: false,
            score: fallbackEvaluation
              ? fallbackEvaluation.fillRatio +
                Math.min(fallbackEvaluation.widthRatio, fallbackEvaluation.heightRatio) * 0.18 +
                fallbackEvaluation.widthRatio * widthReward +
                fallbackEvaluation.heightRatio * heightReward -
                fallbackEvaluation.centerOffset * centerPenalty -
                index * 0.006
              : -Infinity,
          };
        }

        for (let iteration = 0; iteration < 18; iteration += 1) {
          const midpoint = (lowDistance + highDistance) / 2;
          const midpointEvaluation = evaluateViewportFit(viewDirection, midpoint);
          if (midpointEvaluation?.fits) {
            highDistance = midpoint;
          } else {
            lowDistance = midpoint;
          }
        }

        const fitDistanceSafety = nodes.length <= 120
          ? 1.085
          : nodes.length <= 350
            ? 1.06
            : 1.03;
        const finalDistance = Math.min(maximumDistance, highDistance * fitDistanceSafety);
        const finalEvaluation = evaluateViewportFit(viewDirection, finalDistance) || highEvaluation;
        return {
          direction: viewDirection,
          distance: finalDistance,
          fits: true,
          score:
            finalEvaluation.fillRatio +
            Math.min(finalEvaluation.widthRatio, finalEvaluation.heightRatio) * 0.24 +
            finalEvaluation.widthRatio * widthReward +
            finalEvaluation.heightRatio * heightReward -
            finalEvaluation.centerOffset * centerPenalty -
            finalDistance / maximumDistance * 0.015 -
            index * 0.006,
        };
      });
      const bestFit = fitCandidates
        .filter((candidate) => candidate.fits)
        .sort((a, b) => b.score - a.score)[0] ||
        fitCandidates.sort((a, b) => b.score - a.score)[0] ||
        { direction: baseDirection, distance: roughDistance };
      const direction = bestFit.direction;
      const distance = bestFit.distance;

      return flyCamera(
        new THREE.Vector3(
          center.x + direction.x * distance,
          center.y + direction.y * distance,
          center.z + direction.z * distance
        ),
        centerVector,
        duration,
        { kind: 'fit' }
      );
    }, [dimensions.height, dimensions.width, flyCamera, graphData.nodes, nodes.length]);

    const getGraphNodePoint = useCallback((node?: RenderGraphNode | null) => {
      if (!node) return null;
      const point = new THREE.Vector3(
        Number(node.x),
        Number(node.y),
        Number(node.z)
      );
      return Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z) ? point : null;
    }, []);

    const focusGraphPoints = useCallback((points: THREE.Vector3[], duration = 740, options?: {
      composeFit?: boolean;
      maxDistance?: number;
      maxHeightRatio?: number;
      maxWidthRatio?: number;
      minDistance?: number;
      ignoreBottomCornerOverlays?: boolean;
      pointScale?: number;
      targetHeightRatio?: number;
      targetWidthRatio?: number;
    }): boolean => {
      const graph = fgRef.current;
      const camera = graph?.camera() as THREE.PerspectiveCamera | undefined;
      const controls = graph?.controls() as { target?: THREE.Vector3 } | undefined;
      const validPoints = points.filter((point) =>
        Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z)
      );
      if (!graph || !camera || validPoints.length === 0) return false;

      const currentTarget = controls?.target || new THREE.Vector3(0, 0, 0);
      const currentDirection = camera.position.clone().sub(currentTarget);
      const direction = currentDirection.lengthSq() > 0
        ? currentDirection.normalize()
        : new THREE.Vector3(0.55, -0.3, 1).normalize();
      const extents = validPoints.reduce(
        (bounds, point) => ({
          minX: Math.min(bounds.minX, point.x),
          maxX: Math.max(bounds.maxX, point.x),
          minY: Math.min(bounds.minY, point.y),
          maxY: Math.max(bounds.maxY, point.y),
          minZ: Math.min(bounds.minZ, point.z),
          maxZ: Math.max(bounds.maxZ, point.z),
        }),
        {
          minX: validPoints[0].x,
          maxX: validPoints[0].x,
          minY: validPoints[0].y,
          maxY: validPoints[0].y,
          minZ: validPoints[0].z,
          maxZ: validPoints[0].z,
        }
      );
      const center = new THREE.Vector3(
        (extents.minX + extents.maxX) / 2,
        (extents.minY + extents.maxY) / 2,
        (extents.minZ + extents.maxZ) / 2
      );
      const rawSpan = Math.max(
        extents.maxX - extents.minX,
        extents.maxY - extents.minY,
        extents.maxZ - extents.minZ
      );
      const pointScale = options?.pointScale || 1;
      const focusFloor = validPoints.length === 1 ? 42 : nodes.length <= 40 ? 30 : nodes.length <= 120 ? 32 : 38;
      const focusPadding = validPoints.length === 1 ? 46 : nodes.length <= 40 ? 20 : nodes.length <= 120 ? 22 : 32;
      const span = Math.max(focusFloor, rawSpan * pointScale + focusPadding);
      const fov = ((camera.fov || 45) * Math.PI) / 180;
      const aspect = dimensions.width / Math.max(1, dimensions.height);
      const computedMinDistance = validPoints.length === 1
        ? nodes.length <= 12 ? 60 : nodes.length <= 40 ? 66 : nodes.length <= 120 ? 74 : nodes.length <= 350 ? 96 : 118
        : nodes.length <= 12 ? 38 : nodes.length <= 40 ? 44 : nodes.length <= 120 ? 48 : 68;
      const computedMaxDistance = nodes.length <= 12 ? 180 : nodes.length <= 40 ? 220 : nodes.length <= 120 ? 360 : 560;
      const minDistance = options?.minDistance ?? computedMinDistance;
      const maxDistance = options?.maxDistance ?? computedMaxDistance;
      let distance = Math.max(
        minDistance,
        Math.min(maxDistance, (span / (2 * Math.tan(fov / 2))) * (aspect < 1 ? 1.08 : 1))
      );
      let targetCenter = center.clone();
      const fitReservedRects = options?.ignoreBottomCornerOverlays
        ? overlayReservedRects.filter((rect) => {
          const bottomCornerOverlay =
            rect.top > dimensions.height * 0.58 &&
            (
              rect.left > dimensions.width * 0.42 ||
              rect.right < dimensions.width * 0.58
            );
          return !bottomCornerOverlay;
        })
        : overlayReservedRects;

      if (options?.composeFit && validPoints.length > 0 && dimensions.width > 0 && dimensions.height > 0) {
        const horizontalPadding = dimensions.width < 700 ? 36 : 78;
        const verticalPadding = dimensions.height < 560 ? 36 : 72;
        const focusBounds = {
          left: horizontalPadding,
          right: dimensions.width - horizontalPadding,
          top: verticalPadding,
          bottom: dimensions.height - verticalPadding,
        };

        fitReservedRects.forEach((rect) => {
          const rectCenterX = (rect.left + rect.right) / 2;
          const rectCenterY = (rect.top + rect.bottom) / 2;
          const rectHeight = Math.max(0, rect.bottom - rect.top);
          const sideBlockingOverlay =
            rectHeight > dimensions.height * 0.42 ||
            (
              rect.bottom > dimensions.height * 0.26 &&
              rect.top < dimensions.height * 0.64
            );
          if (
            sideBlockingOverlay &&
            rectCenterX >= dimensions.width * 0.5 &&
            rect.left > Math.min(220, dimensions.width * 0.28)
          ) {
            focusBounds.right = Math.min(focusBounds.right, Math.max(focusBounds.left + 120, rect.left - 28));
          }
          if (
            sideBlockingOverlay &&
            rectCenterX < dimensions.width * 0.5 &&
            rect.right < dimensions.width - Math.min(220, dimensions.width * 0.28)
          ) {
            focusBounds.left = Math.max(focusBounds.left, Math.min(focusBounds.right - 120, rect.right + 28));
          }
          if (rectCenterY >= dimensions.height * 0.5 && rect.top > dimensions.height * (dimensions.width < 700 ? 0.24 : 0.34)) {
            focusBounds.bottom = Math.min(focusBounds.bottom, Math.max(focusBounds.top + 120, rect.top - 28));
          }
          if (rectCenterY < dimensions.height * 0.5 && rect.bottom < dimensions.height - Math.min(180, dimensions.height * 0.28)) {
            focusBounds.top = Math.max(focusBounds.top, Math.min(focusBounds.bottom - 120, rect.bottom + 28));
          }
        });

        const focusWidth = Math.max(1, focusBounds.right - focusBounds.left);
        const focusHeight = Math.max(1, focusBounds.bottom - focusBounds.top);
        const focusCenterX = (focusBounds.left + focusBounds.right) / 2;
        const focusCenterY = (focusBounds.top + focusBounds.bottom) / 2;
        const focusShiftX = focusCenterX - dimensions.width * 0.5;
        const focusShiftY = focusCenterY - dimensions.height * 0.5;
        const tempCamera = new THREE.PerspectiveCamera(
          camera.fov || 45,
          aspect,
          camera.near || 1,
          camera.far || 12000
        );
        const projectedPoint = new THREE.Vector3();
        const getTargetCenterForDistance = (candidateDistance: number) => {
          if (Math.abs(focusShiftX) < 1 && Math.abs(focusShiftY) < 1) return center;

          tempCamera.position.copy(center).addScaledVector(direction, candidateDistance);
          tempCamera.lookAt(center);
          tempCamera.updateMatrixWorld();
          const right = new THREE.Vector3().setFromMatrixColumn(tempCamera.matrixWorld, 0);
          const up = new THREE.Vector3().setFromMatrixColumn(tempCamera.matrixWorld, 1);
          const halfViewHeight = candidateDistance * Math.tan(fov / 2);
          const halfViewWidth = halfViewHeight * aspect;
          const desiredNdcX = (focusCenterX / dimensions.width) * 2 - 1;
          const desiredNdcY = 1 - (focusCenterY / dimensions.height) * 2;
          return center.clone()
            .addScaledVector(right, -desiredNdcX * halfViewWidth)
            .addScaledVector(up, -desiredNdcY * halfViewHeight);
        };
        const evaluateDistance = (candidateDistance: number) => {
          const candidateTarget = getTargetCenterForDistance(candidateDistance);
          tempCamera.position.copy(candidateTarget).addScaledVector(direction, candidateDistance);
          tempCamera.lookAt(candidateTarget);
          tempCamera.updateMatrixWorld();
          tempCamera.updateProjectionMatrix();

          let minX = Infinity;
          let maxX = -Infinity;
          let minY = Infinity;
          let maxY = -Infinity;

          for (const point of validPoints) {
            projectedPoint.copy(point).project(tempCamera);
            if (
              !Number.isFinite(projectedPoint.x) ||
              !Number.isFinite(projectedPoint.y) ||
              !Number.isFinite(projectedPoint.z) ||
              projectedPoint.z < -1 ||
              projectedPoint.z > 1
            ) {
              return null;
            }

            const screenX = (projectedPoint.x + 1) * 0.5 * dimensions.width;
            const screenY = (1 - projectedPoint.y) * 0.5 * dimensions.height;
            minX = Math.min(minX, screenX);
            maxX = Math.max(maxX, screenX);
            minY = Math.min(minY, screenY);
            maxY = Math.max(maxY, screenY);
          }

          const rect = { left: minX, right: maxX, top: minY, bottom: maxY };
          const projectedWidth = Math.max(1, maxX - minX);
          const projectedHeight = Math.max(1, maxY - minY);
          const projectedArea = projectedWidth * projectedHeight;
          const overlayPenalty = fitReservedRects.reduce(
            (total, overlayRect) => total + getRectOverlapArea(rect, overlayRect) / Math.max(1, projectedArea),
            0
          );

          return {
            centerOffset:
              Math.abs((minX + maxX) / 2 - focusCenterX) / focusWidth +
              Math.abs((minY + maxY) / 2 - focusCenterY) / focusHeight,
            fillRatio: Math.min(1, projectedArea / Math.max(1, focusWidth * focusHeight)),
            fits: minX >= focusBounds.left &&
              maxX <= focusBounds.right &&
              minY >= focusBounds.top &&
              maxY <= focusBounds.bottom,
            heightRatio: Math.min(1, projectedHeight / focusHeight),
            overlayPenalty,
            widthRatio: Math.min(1, projectedWidth / focusWidth),
          };
        };

        const desiredWidthRatio = options.targetWidthRatio ?? (
          validPoints.length <= 6 ? 0.34 : validPoints.length <= 18 ? 0.42 : 0.5
        );
        const desiredHeightRatio = options.targetHeightRatio ?? (
          validPoints.length <= 6 ? 0.28 : validPoints.length <= 18 ? 0.34 : 0.42
        );
        const maxWidthRatio = options.maxWidthRatio ?? 0.62;
        const maxHeightRatio = options.maxHeightRatio ?? 0.62;
        const candidateFloor = Math.max(minDistance, distance * 0.42);
        const candidateCeiling = Math.max(distance, candidateFloor);
        const candidateDistances = Array.from({ length: 16 }, (_, index) => {
          const progress = index / 15;
          return candidateFloor + (candidateCeiling - candidateFloor) * Math.pow(progress, 1.45);
        });

        type FocusDistanceCandidate = { distance: number; score: number };
        let bestCandidate: FocusDistanceCandidate | null = null;
        for (const candidateDistance of candidateDistances) {
          const evaluation = evaluateDistance(candidateDistance);
          if (!evaluation?.fits) continue;

          const overage =
            Math.max(0, evaluation.widthRatio - maxWidthRatio) +
            Math.max(0, evaluation.heightRatio - maxHeightRatio);
          const score =
            evaluation.fillRatio * 0.52 -
            Math.abs(evaluation.widthRatio - desiredWidthRatio) * 0.95 -
            Math.abs(evaluation.heightRatio - desiredHeightRatio) * 0.82 -
            evaluation.centerOffset * 0.22 -
            evaluation.overlayPenalty * 1.8 -
            overage * 2.4 -
            candidateDistance / Math.max(1, maxDistance) * 0.015;

          if (!bestCandidate || score > bestCandidate.score) {
            bestCandidate = { distance: candidateDistance, score };
          }
        }

        if (bestCandidate) {
          distance = bestCandidate.distance;
        }
        targetCenter = getTargetCenterForDistance(distance).clone();
      }

      return flyCamera(
        new THREE.Vector3(
          targetCenter.x + direction.x * distance,
          targetCenter.y + direction.y * distance,
          targetCenter.z + direction.z * distance
        ),
        targetCenter,
        duration,
        { kind: 'focus' }
      );
    }, [dimensions.height, dimensions.width, flyCamera, nodes.length, overlayReservedRects]);

    const getOverviewFitOptions = useCallback(() => {
      const compactViewport = dimensions.width < 700;
      if (nodes.length <= 40) {
        return {
          composeFit: true,
          maxDistance: compactViewport ? 230 : 205,
          maxHeightRatio: compactViewport ? 0.76 : 0.72,
          maxWidthRatio: compactViewport ? 0.86 : 0.82,
          minDistance: compactViewport ? 42 : 32,
          pointScale: compactViewport ? 0.96 : 0.86,
          targetHeightRatio: compactViewport ? 0.58 : 0.54,
          targetWidthRatio: compactViewport ? 0.74 : 0.7,
        };
      }

      if (nodes.length <= 90) {
        return {
          composeFit: true,
          maxDistance: compactViewport ? 360 : 345,
          maxHeightRatio: compactViewport ? 0.68 : 0.58,
          maxWidthRatio: compactViewport ? 0.78 : 0.64,
          minDistance: compactViewport ? 56 : 44,
          pointScale: compactViewport ? 1.14 : 1.04,
          targetHeightRatio: compactViewport ? 0.52 : 0.42,
          targetWidthRatio: compactViewport ? 0.64 : 0.54,
        };
      }

      return {
        composeFit: true,
        maxDistance: compactViewport ? 760 : 620,
        maxHeightRatio: compactViewport ? 0.72 : 0.7,
        maxWidthRatio: compactViewport ? 0.84 : 0.82,
        minDistance: compactViewport ? 84 : 72,
        pointScale: compactViewport ? 1.08 : 0.92,
        targetHeightRatio: compactViewport ? 0.58 : 0.56,
        targetWidthRatio: compactViewport ? 0.72 : 0.68,
      };
    }, [dimensions.width, nodes.length]);

    const getCompactSelectedOverviewFitOptions = useCallback(() => {
      const base = getOverviewFitOptions();
      if (dimensions.width < 700) {
        return {
          ...base,
          ignoreBottomCornerOverlays: true,
        };
      }

      return {
        ...base,
        ignoreBottomCornerOverlays: true,
        maxDistance: Math.min(base.maxDistance, nodes.length > 350 ? 560 : nodes.length > 120 ? 460 : 300),
        maxHeightRatio: Math.max(base.maxHeightRatio, 0.78),
        maxWidthRatio: Math.max(base.maxWidthRatio, 0.9),
        pointScale: Math.min(base.pointScale, nodes.length > 120 ? 0.78 : 0.84),
        targetHeightRatio: Math.max(base.targetHeightRatio, 0.62),
        targetWidthRatio: Math.max(base.targetWidthRatio, 0.78),
      };
    }, [dimensions.width, getOverviewFitOptions, nodes.length]);

    const diveIntoGraphPoint = useCallback((point: THREE.Vector3, duration = 720, distance = 66) => {
      const graph = fgRef.current;
      const camera = graph?.camera() as THREE.PerspectiveCamera | undefined;
      const controls = graph?.controls() as { target?: THREE.Vector3 } | undefined;
      if (!graph || !camera) return;

      const target = controls?.target || new THREE.Vector3(0, 0, 0);
      const direction = camera.position.clone().sub(target);
      if (direction.lengthSq() === 0) direction.set(0.55, -0.3, 1);
      direction.normalize();
      const next = point.clone().add(direction.multiplyScalar(distance));
      flyCamera(next, point, duration, { force: true, kind: 'dive' });
    }, [flyCamera]);

    const emitRelationTrace = useCallback((link: RenderGraphEdge, count = 4) => {
      const graph = fgRef.current;
      if (!graph?.emitParticle) return;
      for (let index = 0; index < count; index += 1) {
        window.setTimeout(() => {
          graph.emitParticle?.(link);
        }, index * 86);
      }
    }, []);

    const selectedHoverRouteSignature = useMemo(() => {
      if (!selectedHoverRouteActive || activeEdgeIds.size === 0) return '';
      return Array.from(activeEdgeIds).sort().slice(0, 8).join('|');
    }, [activeEdgeIds, selectedHoverRouteActive]);

    useEffect(() => {
      if (!selectedHoverRouteSignature || cameraLod.moving) return undefined;
      const graph = fgRef.current;
      if (!graph?.emitParticle) return undefined;

      const routeEdgeIds = selectedHoverRouteSignature.split('|').filter(Boolean);
      const routeLinks: RenderGraphEdge[] = [];
      routeEdgeIds.forEach((edgeId) => {
        const link = graphData.links.find((item) => item.id === edgeId);
        if (link) routeLinks.push(link as RenderGraphEdge);
      });
      if (routeLinks.length === 0) return undefined;

      const pulseCount = renderProfile.tier === 'massive'
        ? 1
        : routeLinks.length <= 3
          ? 3
          : 2;
      const timers: number[] = [];

      routeLinks.forEach((link, linkIndex) => {
        for (let pulseIndex = 0; pulseIndex < pulseCount; pulseIndex += 1) {
          timers.push(window.setTimeout(() => {
            graph.emitParticle?.(link);
          }, pulseIndex * 240 + linkIndex * 72));
        }
      });

      return () => {
        timers.forEach((timer) => window.clearTimeout(timer));
      };
    }, [cameraLod.moving, graphData.links, renderProfile.tier, selectedHoverRouteSignature]);

    const focusNodeInGraph = useCallback((
      nodeId: string,
      duration = 760,
      options?: {
        composeFit?: boolean;
        maxDistance?: number;
        maxHeightRatio?: number;
        maxWidthRatio?: number;
        minDistance?: number;
        neighborhoodLimit?: number;
        pointScale?: number;
        targetHeightRatio?: number;
        targetWidthRatio?: number;
      }
    ): boolean => {
      const node = graphData.nodes.find((item) => item.id === nodeId);
      if (!node) return false;
      const nodeById = new Map(graphData.nodes.map((item) => [item.id, item]));
      const selectedPoint = getGraphNodePoint(node);
      const neighborhoodLimit = options?.neighborhoodLimit ?? (nodes.length > 1200 ? 64 : nodes.length > 350 ? 74 : nodes.length > 120 ? 84 : 36);
      const neighborhoodPoints = graphData.links
        .map((edge) => {
          const sourceEndpoint = edge.source as string | RenderGraphNode;
          const targetEndpoint = edge.target as string | RenderGraphNode;
          const sourceId = typeof sourceEndpoint === 'string' ? sourceEndpoint : sourceEndpoint?.id;
          const targetId = typeof targetEndpoint === 'string' ? targetEndpoint : targetEndpoint?.id;
          if (sourceId !== nodeId && targetId !== nodeId) return null;
          const otherId = sourceId === nodeId ? targetId : sourceId;
          const otherNode = otherId ? nodeById.get(otherId) : null;
          const point = getGraphNodePoint(otherNode);
          if (!point) return null;
          return {
            point,
            score: Number(edge.relationCount || 1) * 12 + Number(edge.weight || 0),
          };
        })
        .filter((item): item is { point: THREE.Vector3; score: number } => Boolean(item))
        .sort((a, b) => b.score - a.score)
        .slice(0, neighborhoodLimit)
        .map((item) => item.point);

      return focusGraphPoints([
        ...(selectedPoint ? [selectedPoint] : []),
        ...neighborhoodPoints,
      ], duration, {
        composeFit: options?.composeFit,
        maxDistance: options?.maxDistance,
        maxHeightRatio: options?.maxHeightRatio,
        maxWidthRatio: options?.maxWidthRatio,
        minDistance: options?.minDistance,
        pointScale: options?.pointScale ?? (neighborhoodPoints.length > 1 ? 1.12 : 1),
        targetHeightRatio: options?.targetHeightRatio,
        targetWidthRatio: options?.targetWidthRatio,
      });
    }, [focusGraphPoints, getGraphNodePoint, graphData.links, graphData.nodes, nodes.length]);

    const diveIntoNodeInGraph = useCallback((nodeId: string, duration = 720) => {
      const node = graphData.nodes.find((item) => item.id === nodeId);
      const point = getGraphNodePoint(node);
      if (!point) return;
      const distance = nodes.length > 1200 ? 156 : nodes.length > 350 ? 136 : nodes.length > 120 ? 116 : 88;
      diveIntoGraphPoint(point, duration, distance);
    }, [diveIntoGraphPoint, getGraphNodePoint, graphData.nodes, nodes.length]);

    const focusEdgeInGraph = useCallback((
      edgeId: string,
      duration = 760,
      options?: {
        composeFit?: boolean;
        maxDistance?: number;
        maxHeightRatio?: number;
        maxWidthRatio?: number;
        minDistance?: number;
        pointScale?: number;
        targetHeightRatio?: number;
        targetWidthRatio?: number;
      }
    ): boolean => {
      const edge = graphData.links.find((item) => item.id === edgeId);
      if (!edge) return false;

      const sourceEndpoint = edge.source as string | RenderGraphNode;
      const targetEndpoint = edge.target as string | RenderGraphNode;
      const sourceId = typeof sourceEndpoint === 'string' ? sourceEndpoint : sourceEndpoint.id;
      const targetId = typeof targetEndpoint === 'string' ? targetEndpoint : targetEndpoint.id;
      const source = graphData.nodes.find((item) => item.id === sourceId);
      const target = graphData.nodes.find((item) => item.id === targetId);
      if (!source || !target) return false;

      const narrowViewport = dimensions.width < 640 || dimensions.width / Math.max(1, dimensions.height) < 0.72;
      return focusGraphPoints([
        getGraphNodePoint(source),
        getGraphNodePoint(target),
      ].filter((point): point is THREE.Vector3 => Boolean(point)), duration, {
        composeFit: options?.composeFit,
        maxDistance: options?.maxDistance ?? (
          narrowViewport
            ? nodes.length > 350 ? 420 : nodes.length > 120 ? 360 : 300
            : nodes.length > 350 ? 380 : nodes.length > 120 ? 320 : 270
        ),
        maxHeightRatio: options?.maxHeightRatio,
        maxWidthRatio: options?.maxWidthRatio,
        minDistance: options?.minDistance ?? (
          narrowViewport
            ? nodes.length > 350 ? 180 : nodes.length > 120 ? 152 : 126
            : nodes.length > 350 ? 132 : nodes.length > 120 ? 116 : 104
        ),
        pointScale: options?.pointScale ?? (narrowViewport ? 1.16 : 0.96),
        targetHeightRatio: options?.targetHeightRatio,
        targetWidthRatio: options?.targetWidthRatio,
      });
    }, [dimensions.height, dimensions.width, focusGraphPoints, getGraphNodePoint, graphData.links, graphData.nodes, nodes.length]);

    const diveIntoEdgeInGraph = useCallback((link: RenderGraphEdge, duration = 720) => {
      const sourceId = typeof link.source === 'string' ? link.source : link.source?.id;
      const targetId = typeof link.target === 'string' ? link.target : link.target?.id;
      const source = sourceId ? graphData.nodes.find((item) => item.id === sourceId) : null;
      const target = targetId ? graphData.nodes.find((item) => item.id === targetId) : null;
      const sourcePoint = getGraphNodePoint(source);
      const targetPoint = getGraphNodePoint(target);
      if (sourcePoint && targetPoint) {
        focusGraphPoints([sourcePoint, targetPoint], duration, {
          maxDistance: nodes.length > 350 ? 440 : nodes.length > 120 ? 360 : 300,
          minDistance: nodes.length > 350 ? 136 : nodes.length > 120 ? 118 : 104,
          pointScale: dimensions.width < 640 ? 1.18 : 1.04,
          targetHeightRatio: dimensions.width < 640 ? 0.42 : 0.32,
          targetWidthRatio: dimensions.width < 640 ? 0.5 : 0.42,
        });
        return;
      }
      const availablePoint = sourcePoint || targetPoint;
      if (availablePoint) diveIntoGraphPoint(availablePoint, duration);
    }, [dimensions.width, diveIntoGraphPoint, focusGraphPoints, getGraphNodePoint, graphData.nodes, nodes.length]);

    const focusPathInGraph = useCallback((nodeIds: string[], duration = 680) => {
      if (nodeIds.length === 0) return;
      const nodeIdSet = new Set(nodeIds);
      const points = graphData.nodes
        .filter((node) => nodeIdSet.has(node.id))
        .map((node) => getGraphNodePoint(node))
        .filter((point): point is THREE.Vector3 => Boolean(point));

      if (points.length === 0) return;
      focusGraphPoints(points, duration, { pointScale: points.length > 2 ? 1.42 : 1.26 });
    }, [focusGraphPoints, getGraphNodePoint, graphData.nodes]);

    useEffect(() => {
      if (!cameraFocusRequest) return;
      const graph = fgRef.current;
      const camera = graph?.camera() as THREE.PerspectiveCamera | undefined;
      const controls = graph?.controls() as { target?: THREE.Vector3; update?: () => void } | undefined;
      if (!graph || !camera) return;

      const currentTarget = controls?.target || new THREE.Vector3(0, 0, 0);
      const direction = camera.position.clone().sub(currentTarget);
      if (direction.lengthSq() === 0) direction.set(0.55, -0.3, 1);
      const requestedDistance = camera.position.distanceTo(currentTarget) || cameraLodRef.current.distance;
      const minDistance = cameraFocusRequest.minDistance ?? 24;
      const maxDistance = cameraFocusRequest.maxDistance ?? 7600;
      const distance = Math.max(minDistance, Math.min(maxDistance, requestedDistance));
      const target = new THREE.Vector3(
        cameraFocusRequest.target.x,
        cameraFocusRequest.target.y,
        cameraFocusRequest.target.z
      );
      const next = target.clone().add(direction.normalize().multiplyScalar(distance));
      const duration = cameraFocusRequest.duration ?? 720;

      flyCamera(next, target, duration, { kind: 'focus' });
    }, [cameraFocusRequest, flyCamera]);

    const isGraphOverlayEventTarget = useCallback((target: EventTarget | null) => {
      const element = target as Element | null;
      return Boolean(element?.closest([
        '[data-kg-search-tray="true"]',
        '[data-kg-selection-card="true"]',
        '[data-kg-selection-quick-rail="true"]',
        '[data-kg-graph-tools="true"]',
        '[data-kg-graph-stats="true"]',
        '[data-kg-graph-inspector="true"]',
        '[data-kg-graph-overview="true"]',
      ].join(',')));
    }, []);

    const markManualCameraInteraction = useCallback(() => {
      const now = window.performance.now();
      cancelCameraFlight();
      lastUserCameraInteractionRef.current = now;
      pendingAutoFitRef.current = false;
      clearHoverIntentTimer();
      setHoveredLink(null);
      onNodeHover(null);
      autoFitTimersRef.current.forEach((timer) => window.clearTimeout(timer));
      autoFitTimersRef.current = [];
      cameraMovingUntilRef.current = now + 620;
      if (manualCameraReleaseTimerRef.current !== null) {
        window.clearTimeout(manualCameraReleaseTimerRef.current);
      }
      manualCameraReleaseTimerRef.current = window.setTimeout(() => {
        manualCameraReleaseTimerRef.current = null;
        setManualCameraInteractionVersion((version) => version + 1);
      }, GRAPH_MANUAL_NAVIGATION_SETTLE_MS + 80);
      if (manualCameraInteractionFrameRef.current === null) {
        manualCameraInteractionFrameRef.current = window.requestAnimationFrame(() => {
          manualCameraInteractionFrameRef.current = null;
          setManualCameraInteractionVersion((version) => version + 1);
        });
      }
    }, [cancelCameraFlight, clearHoverIntentTimer, onNodeHover]);

    useEffect(() => () => {
      if (manualCameraInteractionFrameRef.current !== null) {
        window.cancelAnimationFrame(manualCameraInteractionFrameRef.current);
        manualCameraInteractionFrameRef.current = null;
      }
      if (manualCameraReleaseTimerRef.current !== null) {
        window.clearTimeout(manualCameraReleaseTimerRef.current);
        manualCameraReleaseTimerRef.current = null;
      }
    }, []);

    useEffect(() => {
      const element = containerRef.current;
      if (!element) return undefined;
      const handleManualInput = (event: Event) => {
        if (isGraphOverlayEventTarget(event.target)) return;
        markManualCameraInteraction();
      };
      const handleManualPointerMove = (event: PointerEvent) => {
        if (isGraphOverlayEventTarget(event.target)) return;
        if (event.buttons) markManualCameraInteraction();
      };
      element.addEventListener('wheel', handleManualInput, { capture: true, passive: true });
      element.addEventListener('pointermove', handleManualPointerMove, { capture: true, passive: true });
      return () => {
        element.removeEventListener('wheel', handleManualInput, { capture: true });
        element.removeEventListener('pointermove', handleManualPointerMove, { capture: true });
      };
    }, [ForceGraph3DRuntime, isGraphOverlayEventTarget, markManualCameraInteraction]);

    const updateCameraLod = useCallback((isInteraction = false) => {
      const now = window.performance.now();
      const programmaticCameraActive = now < programmaticCameraUntilRef.current;
      if (isInteraction && !programmaticCameraActive) {
        markManualCameraInteraction();
      }
      const updateInterval = isInteraction ? 140 : 240;
      if (now - cameraLodTickRef.current < updateInterval) return;
      cameraLodTickRef.current = now;

      const graph = fgRef.current;
      const camera = graph?.camera() as THREE.PerspectiveCamera | undefined;
      const controls = graph?.controls() as { target?: THREE.Vector3 } | undefined;
      if (!camera) return;

      const target = controls?.target || new THREE.Vector3(0, 0, 0);
      const key = getGraphCameraKey(camera.position, target, nodes.length);
      const distance = camera.position.distanceTo(target);
      const direction = target.clone().sub(camera.position);
      if (direction.lengthSq() === 0) direction.set(0, 0, -1);
      direction.normalize();
      const moving = now < cameraMovingUntilRef.current;
      if (key === cameraLodRef.current.key && moving === cameraLodRef.current.moving) return;

      const next = {
        key,
        zoomTier: getGraphZoomTier(distance, nodes.length),
        distance,
        position: {
          x: camera.position.x,
          y: camera.position.y,
          z: camera.position.z,
        },
        target: {
          x: target.x,
          y: target.y,
          z: target.z,
        },
        direction: {
          x: direction.x,
          y: direction.y,
          z: direction.z,
        },
        moving,
      };
      cameraLodRef.current = next;
      setCameraLod(next);
      onCameraLodChange?.(next);
    }, [markManualCameraInteraction, nodes.length, onCameraLodChange]);

    useEffect(() => {
      updateCameraLod();
      const timer = window.setInterval(updateCameraLod, 420);
      return () => window.clearInterval(timer);
    }, [updateCameraLod]);

      useEffect(() => {
        if (!ForceGraph3DRuntime) return;
        if (!nodes.length) return;
        if (selectedNode || selectedEdge) {
        pendingAutoFitRef.current = false;
        return;
      }
      pendingAutoFitRef.current = true;
      autoFitTimersRef.current.forEach((timer) => window.clearTimeout(timer));
      const runAutoFit = (duration: number, lockDirection = false) => {
        if (
          lastUserCameraInteractionRef.current > 0 &&
          window.performance.now() - lastUserCameraInteractionRef.current < 1800
        ) {
          pendingAutoFitRef.current = false;
          return;
        }
        if (nodes.length <= GRAPH_OVERVIEW_DIRECT_FIT_LIMIT) {
          const fitPoints = graphData.nodes
            .map((node) => getGraphNodePoint(node))
            .filter((point): point is THREE.Vector3 => Boolean(point));
          if (fitPoints.length > 0) {
            const focused = focusGraphPoints(fitPoints, duration, getOverviewFitOptions());
            if (focused) {
              lastAutoFitAtRef.current = window.performance.now();
              pendingAutoFitRef.current = false;
              markInitialViewportReady(180);
              return;
            }
          }
          pendingAutoFitRef.current = true;
          return;
        }
        const fitted = fitGraph(duration, { lockDirection });
        if (fitted) {
          lastAutoFitAtRef.current = window.performance.now();
          pendingAutoFitRef.current = false;
          markInitialViewportReady(180);
        } else {
          pendingAutoFitRef.current = true;
        }
      };
        const initialFit = window.setTimeout(() => {
          runAutoFit(920);
        }, 260);
        autoFitTimersRef.current = [initialFit];
      return () => {
        window.clearTimeout(initialFit);
        autoFitTimersRef.current = autoFitTimersRef.current.filter(
          (timer) => timer !== initialFit
          );
        };
      }, [ForceGraph3DRuntime, dimensions.height, dimensions.width, edges.length, fitGraph, focusGraphPoints, getGraphNodePoint, getOverviewFitOptions, graphData.nodes, markInitialViewportReady, nodes.length, selectedEdge, selectedNode, viewSignature]);

      const selectedNodeHasVisibleEdges = useMemo(() => {
        if (!selectedNode) return false;
        return graphData.links.some((edge) =>
          getLinkEndpointId(edge.source) === selectedNode ||
          getLinkEndpointId(edge.target) === selectedNode
        );
      }, [getLinkEndpointId, graphData.links, selectedNode]);

        useEffect(() => {
          if (!ForceGraph3DRuntime) return;
          if (!selectedNode && !selectedEdge) {
            lastSelectionFocusSignatureRef.current = null;
            return;
          }
          const searchPreviewActive = Boolean(searchTerm.trim());
          const expandedSelectionFocus = selectionExpanded && !searchPreviewActive;
          const compactSelectionViewport = dimensions.width < 700;
          const selectionFocusSignature = [
            selectedNode ? `node:${selectedNode}` : `edge:${selectedEdge || ''}`,
            selectedNode ? (selectedNodeHasVisibleEdges ? 'linked' : 'isolated') : 'relation',
            searchPreviewActive ? 'search' : 'browse',
            expandedSelectionFocus ? 'expanded' : 'compact',
            nodes.length,
            edges.length,
            dimensions.width,
            dimensions.height,
          ].join('|');
          if (lastSelectionFocusSignatureRef.current === selectionFocusSignature) return;
          lastSelectionFocusSignatureRef.current = selectionFocusSignature;
          const scheduledAt = window.performance.now();
          const focusFullGraph = (duration: number): boolean => {
            if (nodes.length <= GRAPH_OVERVIEW_DIRECT_FIT_LIMIT) {
              const fitPoints = graphData.nodes
                .map((node) => getGraphNodePoint(node))
                .filter((point): point is THREE.Vector3 => Boolean(point));
              if (fitPoints.length > 0) {
                return focusGraphPoints(
                  fitPoints,
                  duration,
                  selectedNode && !expandedSelectionFocus && !searchPreviewActive
                    ? getCompactSelectedOverviewFitOptions()
                    : getOverviewFitOptions()
                );
              }
              return false;
            }

            return fitGraph(duration, { lockDirection: true });
          };
          const focusSelection = (duration: number) => {
            if (lastUserCameraInteractionRef.current > scheduledAt) return;
            if (window.performance.now() - lastUserCameraInteractionRef.current < GRAPH_SELECTION_REFIT_AFTER_MANUAL_MS) return;
            if (selectedNode) {
            if (!selectedNodeHasVisibleEdges && !searchPreviewActive) {
              const focused = focusFullGraph(duration);
              pendingAutoFitRef.current = !focused;
              if (focused) markInitialViewportReady(180);
              return;
            }
            if (!expandedSelectionFocus && !searchPreviewActive) {
              const focused = focusFullGraph(duration);
              pendingAutoFitRef.current = !focused;
              if (focused) markInitialViewportReady(180);
              return;
            }
            const focused = focusNodeInGraph(selectedNode, duration, searchPreviewActive
              ? {
                composeFit: true,
              maxDistance: compactSelectionViewport
                ? nodes.length > 350 ? 660 : nodes.length > 120 ? 560 : 480
                : nodes.length > 350 ? 520 : nodes.length > 120 ? 440 : 340,
              maxHeightRatio: compactSelectionViewport ? 0.58 : 0.6,
              maxWidthRatio: compactSelectionViewport ? 0.78 : 0.62,
              minDistance: compactSelectionViewport
                ? nodes.length > 350 ? 232 : nodes.length > 120 ? 198 : 168
                : nodes.length > 350 ? 150 : nodes.length > 120 ? 132 : 104,
              neighborhoodLimit: nodes.length > 350 ? 82 : nodes.length > 120 ? 68 : 36,
              pointScale: compactSelectionViewport
                ? nodes.length > 350 ? 1.72 : nodes.length > 120 ? 1.58 : 1.44
                : nodes.length > 350 ? 1.22 : nodes.length > 120 ? 1.12 : 1,
              targetHeightRatio: compactSelectionViewport
                ? nodes.length > 350 ? 0.38 : nodes.length > 120 ? 0.36 : 0.34
                : nodes.length > 350 ? 0.3 : nodes.length > 120 ? 0.28 : 0.25,
              targetWidthRatio: compactSelectionViewport
                ? nodes.length > 350 ? 0.56 : nodes.length > 120 ? 0.54 : 0.5
                : nodes.length > 350 ? 0.38 : nodes.length > 120 ? 0.35 : 0.31,
              }
              : {
                composeFit: true,
                maxDistance: expandedSelectionFocus
                  ? compactSelectionViewport
                    ? nodes.length > 350 ? 660 : nodes.length > 120 ? 560 : 480
                    : nodes.length > 350 ? 580 : nodes.length > 120 ? 500 : 420
                    : compactSelectionViewport
                      ? nodes.length > 350 ? 500 : nodes.length > 120 ? 430 : 360
                      : nodes.length > 350 ? 500 : nodes.length > 120 ? 430 : 390,
                maxHeightRatio: compactSelectionViewport ? 0.64 : expandedSelectionFocus ? 0.62 : 0.72,
                maxWidthRatio: compactSelectionViewport ? 0.86 : expandedSelectionFocus ? 0.76 : 0.82,
                minDistance: expandedSelectionFocus
                  ? compactSelectionViewport
                    ? nodes.length > 350 ? 280 : nodes.length > 120 ? 248 : 220
                    : nodes.length > 350 ? 240 : nodes.length > 120 ? 210 : 184
                    : compactSelectionViewport
                      ? nodes.length > 350 ? 220 : nodes.length > 120 ? 190 : 160
                      : nodes.length > 350 ? 230 : nodes.length > 120 ? 200 : 148,
                neighborhoodLimit: expandedSelectionFocus
                  ? nodes.length > 350 ? 68 : nodes.length > 120 ? 52 : 30
                  : nodes.length > 350 ? 32 : nodes.length > 120 ? 24 : 14,
                pointScale: expandedSelectionFocus
                  ? compactSelectionViewport
                    ? nodes.length > 350 ? 1.8 : nodes.length > 120 ? 1.66 : 1.52
                    : nodes.length > 350 ? 1.34 : nodes.length > 120 ? 1.24 : 1.14
                    : compactSelectionViewport
                      ? nodes.length > 350 ? 1.34 : nodes.length > 120 ? 1.22 : 1.1
                      : nodes.length > 350 ? 1.14 : nodes.length > 120 ? 1.06 : 0.98,
                targetHeightRatio: compactSelectionViewport
                  ? expandedSelectionFocus
                    ? nodes.length > 350 ? 0.42 : nodes.length > 120 ? 0.39 : 0.36
                    : nodes.length > 350 ? 0.32 : nodes.length > 120 ? 0.3 : 0.28
                  : expandedSelectionFocus
                    ? nodes.length > 350 ? 0.38 : nodes.length > 120 ? 0.35 : 0.32
                    : nodes.length > 350 ? 0.32 : nodes.length > 120 ? 0.29 : 0.34,
                targetWidthRatio: compactSelectionViewport
                  ? expandedSelectionFocus
                    ? nodes.length > 350 ? 0.66 : nodes.length > 120 ? 0.62 : 0.58
                    : nodes.length > 350 ? 0.52 : nodes.length > 120 ? 0.48 : 0.58
                  : expandedSelectionFocus
                    ? nodes.length > 350 ? 0.52 : nodes.length > 120 ? 0.48 : 0.43
                    : nodes.length > 350 ? 0.44 : nodes.length > 120 ? 0.4 : 0.5,
              });
            pendingAutoFitRef.current = !focused;
            if (focused) markInitialViewportReady(180);
            return;
          }
          if (selectedEdge) {
              const focused = focusEdgeInGraph(selectedEdge, duration, searchPreviewActive
            ? {
              composeFit: true,
              maxDistance: compactSelectionViewport
                ? nodes.length > 350 ? 700 : nodes.length > 120 ? 600 : 520
                : nodes.length > 350 ? 560 : nodes.length > 120 ? 480 : 400,
              maxHeightRatio: compactSelectionViewport ? 0.54 : 0.6,
              maxWidthRatio: compactSelectionViewport ? 0.78 : 0.7,
              minDistance: compactSelectionViewport
                ? nodes.length > 350 ? 300 : nodes.length > 120 ? 264 : 232
                : nodes.length > 350 ? 220 : nodes.length > 120 ? 190 : 162,
              pointScale: compactSelectionViewport ? 1.82 : 1.22,
              targetHeightRatio: compactSelectionViewport ? 0.34 : 0.36,
              targetWidthRatio: compactSelectionViewport ? 0.58 : 0.5,
              }
              : {
                composeFit: true,
                maxDistance: expandedSelectionFocus
                  ? compactSelectionViewport
                    ? nodes.length > 350 ? 760 : nodes.length > 120 ? 660 : 560
                    : nodes.length > 350 ? 620 : nodes.length > 120 ? 520 : 440
                  : compactSelectionViewport
                    ? nodes.length > 350 ? 660 : nodes.length > 120 ? 560 : 480
                    : nodes.length > 350 ? 480 : nodes.length > 120 ? 390 : 320,
                maxHeightRatio: compactSelectionViewport ? 0.58 : expandedSelectionFocus ? 0.72 : 0.56,
                maxWidthRatio: compactSelectionViewport ? 0.8 : expandedSelectionFocus ? 0.82 : 0.64,
                minDistance: expandedSelectionFocus
                  ? compactSelectionViewport
                    ? nodes.length > 350 ? 340 : nodes.length > 120 ? 300 : 260
                    : nodes.length > 350 ? 240 : nodes.length > 120 ? 210 : 178
                  : compactSelectionViewport
                    ? nodes.length > 350 ? 250 : nodes.length > 120 ? 216 : 188
                    : nodes.length > 350 ? 142 : nodes.length > 120 ? 124 : 108,
                pointScale: compactSelectionViewport ? 1.96 : expandedSelectionFocus ? 1.32 : 1.04,
                targetHeightRatio: compactSelectionViewport ? 0.38 : expandedSelectionFocus ? 0.48 : 0.3,
                targetWidthRatio: compactSelectionViewport ? 0.62 : expandedSelectionFocus ? 0.66 : 0.42,
              });
              pendingAutoFitRef.current = !focused;
              if (focused) markInitialViewportReady(180);
            }
          };

        const compactOverviewFocus = Boolean(selectedNode && !expandedSelectionFocus && !searchPreviewActive);
        const focusDelays = compactOverviewFocus
          ? [220, 560, 960, 1480, 2200]
          : [searchPreviewActive ? 140 : 220, 620, 1180];
        const focusTimers = focusDelays.map((delay, index) => window.setTimeout(() => {
          if (compactOverviewFocus && index > 0) {
            updateCameraLod(false);
            const currentCamera = cameraLodRef.current;
            const targetDistanceFromOrigin = Math.hypot(
              currentCamera.target.x,
              currentCamera.target.y,
              currentCamera.target.z
            );
            const stillInDefaultOverview =
              targetDistanceFromOrigin < 32 &&
              currentCamera.distance > (nodes.length > 120 ? 820 : 620);
            if (!pendingAutoFitRef.current && !stillInDefaultOverview) return;
          }

          focusSelection(searchPreviewActive ? 980 : index > 0 ? 620 : 920);
        }, delay));

        return () => {
          focusTimers.forEach((timer) => window.clearTimeout(timer));
        };
      }, [
        ForceGraph3DRuntime,
        dimensions.height,
        dimensions.width,
        edges.length,
        fitGraph,
        focusEdgeInGraph,
        focusGraphPoints,
        focusNodeInGraph,
        getGraphNodePoint,
        getCompactSelectedOverviewFitOptions,
        getOverviewFitOptions,
        graphData.nodes,
        markInitialViewportReady,
        nodes.length,
        searchTerm,
        selectedEdge,
        selectedNode,
        selectedNodeHasVisibleEdges,
        selectionExpanded,
        updateCameraLod,
      ]);

    useEffect(() => {
      if (!ForceGraph3DRuntime) return;
      const query = searchTerm.trim();
      if (!query || selectedNode || selectedEdge) {
        lastSearchFocusSignatureRef.current = null;
        return undefined;
      }

      let previewEdgeId: string | null = null;
      if (!focusNodeId && activeEdgeIds.size > 0) {
        const availableEdgeIds = new Set(graphData.links.map((link) => link.id));
        for (const edgeId of activeEdgeIds) {
          if (availableEdgeIds.has(edgeId)) {
            previewEdgeId = edgeId;
            break;
          }
        }
      }

      const signature = focusNodeId
        ? `node:${query}:${focusNodeId}`
        : previewEdgeId
          ? `edge:${query}:${previewEdgeId}`
          : null;
      if (!signature || lastSearchFocusSignatureRef.current === signature) return undefined;
      lastSearchFocusSignatureRef.current = signature;

      const scheduledAt = window.performance.now();
      const focusTimer = window.setTimeout(() => {
        if (lastUserCameraInteractionRef.current > scheduledAt) return;
        if (window.performance.now() - lastUserCameraInteractionRef.current < GRAPH_SELECTION_REFIT_AFTER_MANUAL_MS) return;

        if (focusNodeId) {
          focusNodeInGraph(focusNodeId, 820, {
            composeFit: true,
            maxDistance: nodes.length > 350 ? 360 : nodes.length > 120 ? 300 : 236,
            maxHeightRatio: 0.6,
            maxWidthRatio: 0.62,
            minDistance: nodes.length > 350 ? 110 : nodes.length > 120 ? 88 : 66,
            neighborhoodLimit: nodes.length > 350 ? 64 : nodes.length > 120 ? 52 : 30,
            pointScale: nodes.length > 350 ? 1.02 : nodes.length > 120 ? 0.96 : 0.9,
            targetHeightRatio: nodes.length > 350 ? 0.34 : nodes.length > 120 ? 0.31 : 0.28,
            targetWidthRatio: nodes.length > 350 ? 0.44 : nodes.length > 120 ? 0.4 : 0.36,
          });
          return;
        }

        if (previewEdgeId) {
          focusEdgeInGraph(previewEdgeId, 780, {
            composeFit: true,
            maxDistance: nodes.length > 350 ? 480 : nodes.length > 120 ? 400 : 340,
            maxHeightRatio: 0.58,
            maxWidthRatio: 0.68,
            minDistance: nodes.length > 350 ? 190 : nodes.length > 120 ? 162 : 136,
            pointScale: 0.96,
            targetHeightRatio: 0.32,
            targetWidthRatio: 0.46,
          });
        }
      }, 160);

      return () => {
        window.clearTimeout(focusTimer);
      };
    }, [
      ForceGraph3DRuntime,
      activeEdgeIds,
      focusEdgeInGraph,
      focusNodeId,
      focusNodeInGraph,
      graphData.links,
      nodes.length,
      searchTerm,
      selectedEdge,
      selectedNode,
    ]);

    useEffect(() => {
      if (!ForceGraph3DRuntime) return undefined;
      if (pathFocusVersion <= 0 || pathFocusNodeIds.length === 0) return undefined;

      const scheduledAt = window.performance.now();
      const focusPath = () => {
        if (lastUserCameraInteractionRef.current > scheduledAt) return;
        focusPathInGraph(pathFocusNodeIds);
      };
      const focusTimer = window.setTimeout(focusPath, 150);

      return () => {
        window.clearTimeout(focusTimer);
      };
    }, [ForceGraph3DRuntime, focusPathInGraph, graphData.nodes, pathFocusNodeIds, pathFocusVersion]);

  useEffect(() => {
    const graph = fgRef.current;
    const charge = graph?.d3Force('charge') as { strength?: (value: number) => void } | undefined;
    const linkForce = graph?.d3Force('link') as {
      distance?: (value: number | ((link: unknown) => number)) => void;
      strength?: (value: number) => void;
    } | undefined;

    const chargeStrength = nodes.length <= 40
      ? -52
      : nodes.length <= 120
        ? -82
        : nodes.length <= 350
          ? -58
          : nodes.length <= 1200
            ? -38
            : -30;
    const linkScale = nodes.length <= 40
      ? 1.34
      : nodes.length <= 120
        ? 1.42
        : nodes.length <= 350
          ? 1.12
          : nodes.length <= 1200
            ? 0.86
            : 0.72;
    charge?.strength?.(chargeStrength);
    linkForce?.strength?.(nodes.length <= 120 ? 0.78 : 0.86);
    linkForce?.distance?.((link: unknown) => {
      const relationCount = Number((link as { relationCount?: number })?.relationCount || 1);
      const relationCompaction = Math.min(relationCount, 48) * (nodes.length <= 120 ? 0.22 : 0.42);
      const minimumLinkDistance = nodes.length <= 40 ? 48 : nodes.length <= 120 ? 42 : 34;
      return Math.max(minimumLinkDistance, (68 - relationCompaction) * linkScale);
    });
    graph?.d3ReheatSimulation?.();
    }, [edges.length, nodes.length]);

  useEffect(() => {
    const graph = fgRef.current;
    if (!graph) return;
    const renderer = graph.renderer();
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, renderProfile.maxPixelRatio));
    const ambient = new THREE.AmbientLight(0xffffff, 1.45);
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(140, 220, 180);
    const rim = new THREE.PointLight(0x72f0ff, 1.4, 480);
    rim.position.set(-180, -120, 220);
    graph.lights([ambient, key, rim]);
  }, [dimensions.height, dimensions.width, renderProfile.maxPixelRatio]);

  useEffect(() => {
    const graph = fgRef.current;
    const scene = graph?.scene();
    if (!scene) return undefined;

    if (depthReferenceGroupRef.current) {
      const previous = depthReferenceGroupRef.current;
      previous.parent?.remove(previous);
      disposeGraphSceneObject(previous);
      depthReferenceGroupRef.current = null;
    }

    if (detailMode === 'performance') return undefined;

    const depthReference = createGraphDepthReferenceObject(graphData.nodes, renderProfile.tier);
    if (!depthReference) return undefined;
    scene.add(depthReference);
    depthReferenceGroupRef.current = depthReference;

    return () => {
      if (depthReferenceGroupRef.current !== depthReference) return;
      depthReference.parent?.remove(depthReference);
      disposeGraphSceneObject(depthReference);
      depthReferenceGroupRef.current = null;
    };
  }, [ForceGraph3DRuntime, detailMode, graphData.nodes, renderProfile.tier, viewSignature]);

  useEffect(() => {
    if (!selectedNode && !selectedEdge && !hoveredNode) return undefined;
    let frameId = 0;
    const animateFocus = () => {
      const wave = (Math.sin(window.performance.now() * 0.0024) + 1) / 2;
      animatedNodeObjectsRef.current.forEach((object) => {
        object.scale.setScalar(object.userData.focused ? 1 + wave * 0.018 : 1 + wave * 0.04);
        object.children.forEach((child) => {
          const material = (child as THREE.Mesh).material;
          if ((child.userData.pulseHalo || child.userData.focusRing) && material instanceof THREE.MeshBasicMaterial) {
            material.opacity = child.userData.focusRing
              ? 0.18 + wave * 0.1
              : object.userData.focused
                ? 0.055 + wave * 0.06
                : 0.08 + wave * 0.08;
          }
        });
      });
      frameId = window.requestAnimationFrame(animateFocus);
    };

    frameId = window.requestAnimationFrame(animateFocus);
    return () => window.cancelAnimationFrame(frameId);
  }, [hoveredNode, selectedEdge, selectedNode]);

  useEffect(() => {
    const graph = fgRef.current;
    const controls = graph?.controls() as {
      enableDamping?: boolean;
      dampingFactor?: number;
      autoRotate?: boolean;
      autoRotateSpeed?: number;
      enablePan?: boolean;
      screenSpacePanning?: boolean;
      panSpeed?: number;
      rotateSpeed?: number;
      zoomSpeed?: number;
      minDistance?: number;
      maxDistance?: number;
      addEventListener?: (type: string, listener: () => void) => void;
      removeEventListener?: (type: string, listener: () => void) => void;
      update?: () => void;
    } | undefined;
    if (!controls) return;

    controls.enableDamping = true;
    controls.dampingFactor = 0.042;
    controls.autoRotate = false;
    controls.autoRotateSpeed = 0;
    controls.enablePan = true;
    controls.screenSpacePanning = true;
    controls.panSpeed = nodes.length > 3000 ? 0.82 : nodes.length > 350 ? 0.76 : nodes.length > 120 ? 0.7 : 0.62;
    controls.rotateSpeed = nodes.length > 3000 ? 0.26 : nodes.length > 1200 ? 0.32 : 0.42;
    controls.zoomSpeed = nodes.length > 3000 ? 0.82 : nodes.length > 350 ? 0.76 : 0.68;
    controls.minDistance = nodes.length > 3000 ? 72 : nodes.length > 1200 ? 48 : nodes.length > 120 ? 32 : 24;
    controls.maxDistance = nodes.length > 3000 ? 7600 : nodes.length > 1200 ? 5200 : nodes.length > 350 ? 3200 : nodes.length > 120 ? 1900 : 980;
    controls.update?.();
    const handleControlsChange = () => updateCameraLod(false);
    controls.addEventListener?.('change', handleControlsChange);
    return () => {
      controls.autoRotate = false;
      controls.removeEventListener?.('change', handleControlsChange);
    };
  }, [nodes.length, updateCameraLod]);

  const moveCamera = useCallback((scale: number) => {
    const graph = fgRef.current;
    const camera = graph?.camera() as THREE.PerspectiveCamera | undefined;
    const controls = graph?.controls() as { target?: THREE.Vector3; update?: () => void } | undefined;
    if (!graph || !camera) return;
    const target = controls?.target || new THREE.Vector3(0, 0, 0);
    const next = camera.position.clone().sub(target).multiplyScalar(scale).add(target);
    const duration = 520;
    flyCamera(next, target, duration, { kind: 'zoom' });
  }, [flyCamera]);

  // Handle camera controls
  const handleZoomIn = useCallback(() => {
    moveCamera(0.72);
  }, [moveCamera]);

  const handleZoomOut = useCallback(() => {
    moveCamera(1.38);
  }, [moveCamera]);

    const handleCenterGraph = useCallback((): boolean => {
      const fitOptions = selectedNode && !selectedEdge && !selectionExpanded && !searchTerm.trim()
        ? getCompactSelectedOverviewFitOptions()
        : getOverviewFitOptions();

      if (nodes.length <= GRAPH_OVERVIEW_DIRECT_FIT_LIMIT) {
        const fitPoints = graphData.nodes
          .map((node) => getGraphNodePoint(node))
          .filter((point): point is THREE.Vector3 => Boolean(point));
        if (fitPoints.length > 0) {
          return focusGraphPoints(fitPoints, 720, fitOptions);
        }
      }

      return fitGraph(720);
    }, [fitGraph, focusGraphPoints, getGraphNodePoint, getCompactSelectedOverviewFitOptions, getOverviewFitOptions, graphData.nodes, nodes.length, searchTerm, selectedEdge, selectedNode, selectionExpanded]);

    useEffect(() => {
      if (!ForceGraph3DRuntime) return undefined;
      if (!selectedNode || selectedEdge || selectionExpanded || searchTerm.trim()) return undefined;

      const scheduledAt = window.performance.now();
      let fitComplete = false;
      const runSelectedOverviewFit = () => {
        if (fitComplete) return;
        if (lastUserCameraInteractionRef.current > scheduledAt) return;
        if (window.performance.now() - lastUserCameraInteractionRef.current < GRAPH_SELECTION_REFIT_AFTER_MANUAL_MS) return;
        const focused = handleCenterGraph();
        if (focused) {
          fitComplete = true;
          pendingAutoFitRef.current = false;
          lastAutoFitAtRef.current = window.performance.now();
          markInitialViewportReady(180);
        }
      };
      const timers = [320, 1120, 2400].map((delay) => window.setTimeout(runSelectedOverviewFit, delay));

      return () => {
        timers.forEach((timer) => window.clearTimeout(timer));
      };
    }, [
      ForceGraph3DRuntime,
      handleCenterGraph,
      markInitialViewportReady,
      searchTerm,
      selectedEdge,
      selectedNode,
      selectionExpanded,
      viewSignature,
    ]);

  // Pause/resume physics
  useEffect(() => {
    if (fgRef.current) {
      if (physicsEnabled) {
        fgRef.current.resumeAnimation();
      } else {
        fgRef.current.pauseAnimation();
      }
    }
  }, [physicsEnabled]);

    useEffect(() => {
      nodeObjectCacheRef.current.clear();
      animatedNodeObjectsRef.current.clear();
      stableLabelLayoutsRef.current.clear();
      persistedLabelStateRef.current.clear();
    }, [nodes]);

    useEffect(() => {
      linkObjectCacheRef.current.clear();
    }, [edges]);

    const isLinkActive = useCallback((link: RenderGraphEdge) => {
      const sourceId = getLinkEndpointId(link.source);
      const targetId = getLinkEndpointId(link.target);
    const hasFocus = visualActiveEdgeIds.size > 0 || visualActiveNodeIds.size > 0;
    if (!hasFocus) return true;
    if (visualActiveEdgeIds.has(link.id)) return true;
    if (visualActiveEdgeIds.size > 0) return false;
    return Boolean(sourceId && visualActiveNodeIds.has(sourceId)) ||
        Boolean(targetId && visualActiveNodeIds.has(targetId));
    }, [getLinkEndpointId, visualActiveEdgeIds, visualActiveNodeIds]);

    const linkMatchesSearch = useCallback((link: RenderGraphEdge) => {
      const query = searchTerm.trim().toLowerCase();
      if (!query) return false;
      const sourceId = getLinkEndpointId(link.source);
      const targetId = getLinkEndpointId(link.target);
      const source = sourceId ? nodeById.get(sourceId) : null;
      const target = targetId ? nodeById.get(targetId) : null;
      return [
        link.searchText,
        link.label,
        link.type,
        sourceId,
        targetId,
        source?.label,
        source?.type,
        target?.label,
        target?.type,
        formatMetadataValue(link.properties || {}),
      ].filter(Boolean).join(' ').toLowerCase().includes(query);
    }, [getLinkEndpointId, nodeById, searchTerm]);

    const shouldRenderNode = useCallback((node: RenderGraphNode) => {
      const query = searchTerm.trim().toLowerCase();
      const isSelected = selectedNode === node.id;
      const isHovered = hoveredNode === node.id;
      const isSearchMatch = Boolean(query && getEntitySearchText(node).includes(query));
      const isFocused = visualActiveNodeIds.has(node.id);
      const hasFocus = visualActiveNodeIds.size > 0 || visualActiveEdgeIds.size > 0;
      const selectionCullingAllowed = renderProfile.tier === 'large' || renderProfile.tier === 'massive';
      const focusCullingActive = Boolean(
        hasFocus &&
        (
          query ||
          (selectionCullingAllowed && selectedEdge) ||
          (selectionCullingAllowed && selectedNode && visualActiveEdgeIds.size > 0)
        )
      );
      if (showLabels && screenReadableLabelLayouts.has(node.id)) return true;
      const persistedLabel = persistedLabelStateRef.current.get(node.id);
      const now = typeof window !== 'undefined' ? window.performance.now() : Date.now();
      if (
        persistedLabel &&
        persistedLabel.visibleUntil > now &&
        (
          showLabels ||
          persistedLabel.retention === 'focus' ||
          Boolean(query && isSearchMatch)
        )
      ) return true;
      if (isSelected || isHovered || isSearchMatch || isFocused) return true;
      const rankedVisible = prioritizedNodeIds.visible.has(node.id);
      const { depthBand, offAxis } = getNodeCameraBand(node, visualCameraLod);
        if (focusCullingActive) {
          if (motionSimplificationActive) return false;
          if (focusContextNodeIds.has(node.id) && rankedVisible && !offAxis && depthBand !== 'back') return true;
          if (hoverFocusVisualActive && !query && !selectedNode && !selectedEdge) {
            return rankedVisible &&
              !offAxis &&
              (depthBand !== 'back' || visualCameraLod.zoomTier === 'near') &&
              (
                renderProfile.tier === 'compact' ||
                renderProfile.tier === 'standard' ||
                visualCameraLod.zoomTier !== 'far'
              );
          }
          if (routeFocusVisualActive && !query && selectedNode && !selectedEdge) {
            return rankedVisible &&
              !offAxis &&
              (
                depthBand !== 'back' ||
                visualCameraLod.zoomTier === 'near' ||
                renderProfile.tier === 'standard'
              );
          }
          return false;
        }
      if (renderProfile.tier === 'compact' || renderProfile.tier === 'standard') return true;

      const nodeBudget = renderProfile.nodeBudget[visualCameraLod.zoomTier];
      if (!Number.isFinite(nodeBudget) || nodes.length <= nodeBudget) return true;
      if (!Number.isFinite(node.x) || !Number.isFinite(node.y) || !Number.isFinite(node.z)) return true;

      if (motionSimplificationActive) {
        return rankedVisible && !(offAxis && depthBand === 'back');
      }
      if (rankedVisible) return true;
      if (renderProfile.tier === 'large' && visualCameraLod.zoomTier === 'near' && !offAxis && depthBand === 'front') return true;
      return false;
      }, [focusContextNodeIds, hoverFocusVisualActive, hoveredNode, motionSimplificationActive, nodes.length, prioritizedNodeIds.visible, renderProfile.nodeBudget, renderProfile.tier, routeFocusVisualActive, screenReadableLabelLayouts, searchTerm, selectedEdge, selectedNode, showLabels, visualActiveEdgeIds.size, visualActiveNodeIds, visualCameraLod]);

    const shouldRenderLink = useCallback((link: RenderGraphEdge) => {
      if (!prioritizedLinkIds) return true;
      if (hoveredLink?.id === link.id || selectedEdge === link.id || linkMatchesSearch(link)) return true;
      if ((visualActiveEdgeIds.size > 0 || visualActiveNodeIds.size > 0) && isLinkActive(link)) return true;
      if (!prioritizedLinkIds.has(link.id)) return false;

      const sourceId = getLinkEndpointId(link.source);
      const targetId = getLinkEndpointId(link.target);
      const source = typeof link.source === 'string' ? (sourceId ? nodeById.get(sourceId) : null) : link.source;
      const target = typeof link.target === 'string' ? (targetId ? nodeById.get(targetId) : null) : link.target;
      if (!source || !target) return true;
      return shouldRenderNode(source as RenderGraphNode) && shouldRenderNode(target as RenderGraphNode);
    }, [getLinkEndpointId, hoveredLink, isLinkActive, linkMatchesSearch, nodeById, prioritizedLinkIds, selectedEdge, shouldRenderNode, visualActiveEdgeIds.size, visualActiveNodeIds.size]);

    const shouldShowLinkLabel = useCallback((link: RenderGraphEdge) => {
      const directlyHovered = hoveredLink?.id === link.id;
      const directlySelected = selectedEdge === link.id;
      const searchActive = Boolean(searchTerm.trim());
      const matchesSearch = searchActive && linkMatchesSearch(link);
      const focusedSearchLink = searchActive && matchesSearch && activeEdgeIds.has(link.id) && activeEdgeIds.size <= 1;
      if (focusedSearchLink) return true;
      if (directlySelected) return true;
      if (directlyHovered) return false;
      if (routeFocusVisualActive && isLinkActive(link) && visualActiveEdgeIds.size <= 8 && routeFocusWeight > 0.14) return true;
      if (navigationSettlingActive) return false;
      if (selectedEdge) return false;
      if (motionSimplificationActive) return false;
      if (!isLinkActive(link)) return false;
      const focusedEdgeLabelLimit = renderProfile.tier === 'massive'
        ? 0
        : renderProfile.tier === 'large'
          ? 0
          : renderProfile.tier === 'standard'
            ? 1
            : 2;
      const activeLabelLimit = searchActive ? Math.min(1, focusedEdgeLabelLimit) : focusedEdgeLabelLimit;
      const focusedEdgeLabelsVisible =
        activeLabelLimit > 0 &&
        activeEdgeIds.size > 0 &&
        activeEdgeIds.size <= activeLabelLimit &&
        nodes.length <= (searchActive ? 80 : 140) &&
        (nodes.length <= 36 || visualCameraLod.zoomTier !== 'far');
      if (searchActive && focusedEdgeLabelsVisible && !matchesSearch) return false;
      if (focusedEdgeLabelsVisible) return true;
      if (
        ambientDirectionCueLinkIds.has(link.id) &&
        nodes.length <= 180 &&
        visualCameraLod.zoomTier !== 'far'
      ) {
        return true;
      }
      if (!showEdgeLabels) return false;
      if (renderProfile.tier === 'compact' && nodes.length <= 36 && activeEdgeIds.size <= 2) return true;
      if (visualCameraLod.zoomTier === 'near' && nodes.length <= 90 && activeEdgeIds.size <= 2) return true;
      return false;
    }, [activeEdgeIds, ambientDirectionCueLinkIds, hoveredLink, isLinkActive, linkMatchesSearch, motionSimplificationActive, navigationSettlingActive, nodes.length, renderProfile.tier, routeFocusVisualActive, routeFocusWeight, searchTerm, selectedEdge, showEdgeLabels, visualActiveEdgeIds.size, visualCameraLod.zoomTier]);

    const shouldShowRelationMarker = useCallback((link: RenderGraphEdge) => {
      const directlyHovered = hoveredLink?.id === link.id;
      const directlySelected = selectedEdge === link.id;
      const searchActive = Boolean(searchTerm.trim());
      const matchesSearch = searchActive && linkMatchesSearch(link);
      const hasFocus = visualActiveEdgeIds.size > 0 || visualActiveNodeIds.size > 0;
      if (directlySelected) return true;
      if (directlyHovered) return !navigationSettlingActive && !motionSimplificationActive;
      if (searchActive) {
        if (selectedNode && activeEdgeIds.size > 2) {
          return Boolean(matchesSearch && focusedRelationMarkerLinkIds.has(link.id));
        }
        return Boolean(matchesSearch && activeEdgeIds.has(link.id));
      }
      if (routeFocusVisualActive && routeFocusWeight > 0.1 && isLinkActive(link)) return true;
      if (navigationSettlingActive) return false;
      if (motionSimplificationActive) return false;
      if (!isLinkActive(link)) return false;

      if (hasFocus) {
        const compactNodeFocus = dimensions.width < 700 && !selectedEdge && visualActiveNodeIds.size > 0;
        if (compactNodeFocus) {
          return visualActiveEdgeIds.size > 0 && focusedRelationMarkerLinkIds.has(link.id);
        }
        if (selectedNode && visualActiveEdgeIds.size > 2) {
          return focusedRelationMarkerLinkIds.has(link.id);
        }
        if (visualActiveEdgeIds.size > 0 && visualActiveEdgeIds.size <= 12) return true;
        return visualActiveEdgeIds.size > 0 && focusedRelationMarkerLinkIds.has(link.id);
      }

      if (relationMarkerLinkIds.has(link.id)) return true;
      if (!showEdgeLabels) return false;
      return nodes.length <= 120 && visualCameraLod.zoomTier === 'near';
    }, [
      activeEdgeIds,
      dimensions.width,
      focusedRelationMarkerLinkIds,
      hoveredLink,
      isLinkActive,
      linkMatchesSearch,
      motionSimplificationActive,
      navigationSettlingActive,
      nodes.length,
      relationMarkerLinkIds,
      routeFocusVisualActive,
      routeFocusWeight,
      searchTerm,
      selectedEdge,
      selectedNode,
      showEdgeLabels,
      visualActiveEdgeIds.size,
      visualActiveNodeIds.size,
      visualCameraLod.zoomTier,
    ]);

    const nodeThreeObject = useCallback((node: RenderGraphNode) => {
    const isSelected = selectedNode === node.id;
    const isHovered = hoveredNode === node.id;
    const query = searchTerm.trim().toLowerCase();
    const hoverOnlyFocusActive = hoverFocusVisualActive && !query;
    const isSearchMatch = Boolean(query && getEntitySearchText(node).includes(query));
    const isSearchPreviewFocus = Boolean(query && focusNodeId === node.id && isSearchMatch);
    const hasFocus = visualActiveNodeIds.size > 0;
    const isConnectedFocus = !hasFocus || visualActiveNodeIds.has(node.id);
    const isRelationshipEndpoint = selectedEdgeEndpointIds.has(node.id);
      const isFocusHaloLabel = focusHaloLabelIds.has(node.id);
      const focusHaloLabelHint = focusHaloLabelHints.get(node.id);
      const isAmbientFocusContextNode = focusContextNodeIds.has(node.id);
      const isActiveIdentityNode = activeIdentityNodeIds.has(node.id);
      const focusedNodeLimit = renderProfile.tier === 'massive'
        ? 44
        : renderProfile.tier === 'large'
          ? 64
          : renderProfile.tier === 'standard'
          ? 96
          : 140;
    const isFocusedContext = hasFocus && visualActiveNodeIds.has(node.id) && visualActiveNodeIds.size <= focusedNodeLimit;
    const focusedSearchMatch = Boolean(
      isSearchMatch &&
      (isSelected || isHovered || isRelationshipEndpoint || isSearchPreviewFocus)
    );
    const isolatedSelectedNodeVisual = Boolean(
      isSelected &&
      selectedNode &&
      !selectedEdge &&
      visualActiveEdgeIds.size === 0 &&
      !query
    );
    const isHighlighted = isSelected || isHovered || focusedSearchMatch;
    const compactSelectedFocusLabel = Boolean(
      isSelected &&
      compactSelectedNodeLabel &&
      !isHovered &&
      !query &&
      !isolatedSelectedNodeVisual
    );
    const compactSearchPreviewLabel = Boolean(isSearchPreviewFocus && !isSelected && !isHovered && !isRelationshipEndpoint);
    const compactRelationshipEndpoint = Boolean(isRelationshipEndpoint && compactSelectedEdgeEndpointLabels && !isHovered && !query);
    const richLabelFocus =
      isolatedSelectedNodeVisual ||
      (isHighlighted && !compactSelectedFocusLabel && !compactSearchPreviewLabel) ||
      (isRelationshipEndpoint && !compactRelationshipEndpoint);
    const isFocusContextNode = (isFocusedContext || isAmbientFocusContextNode) && !isHighlighted;
      const hoverFocusedNeighbor = Boolean(
        isFocusContextNode &&
        hoverOnlyFocusActive &&
        showLabels &&
        !query &&
        detailMode !== 'performance'
      );
    const activeSelectionFocus = Boolean(selectedNode || selectedEdge || query);
      const isMuted = hasFocus && !isConnectedFocus;
      const routeContextFocusActive = Boolean(routeFocusVisualActive && !query && hasFocus);
      const isSoftMuted = (hoverOnlyFocusActive || routeContextFocusActive) && isMuted;
      const isHardMuted = isMuted && !isSoftMuted;
      const softMuteStrengthForNode = routeContextFocusActive ? routeFocusWeight : hoverFocusWeight;
      const activeIdentityVisible = Boolean(
        activeSelectionFocus &&
        detailMode !== 'performance' &&
        isActiveIdentityNode &&
        !isHardMuted
      );
      const selectedFocusNameplateVisible = Boolean(
        activeIdentityVisible
      );
      const selectedFocusSymbolVisible = Boolean(
        activeIdentityVisible ||
        (
          activeSelectionFocus &&
          isActiveIdentityNode &&
          !isHardMuted
        )
      );
      const isFocusedNeighbor = isFocusContextNode && (
        hoverFocusedNeighbor ||
        isRelationshipEndpoint ||
        isFocusHaloLabel ||
        activeSelectionFocus ||
        selectedFocusNameplateVisible
      );
    const relationSummary = nodeRelationSummary.get(node.id);
    const relationTypeMix = nodeRelationTypeMix.get(node.id) || [];
    const { depthBand, offAxis } = getNodeCameraBand(node, visualCameraLod);
    const rankedVisible = prioritizedNodeIds.visible.has(node.id);
    const prioritizedLabelAllows = prioritizedNodeIds.labels.has(node.id);
    const fallbackLabelAllows = prioritizedNodeIds.fallbackLabels.has(node.id);
    const navigationLabelsSettling = navigationSettlingActive;
    let suppressWhileMoving =
      motionSimplificationActive &&
      renderProfile.tier !== 'compact' &&
      renderProfile.tier !== 'standard' &&
      !isHighlighted &&
      !isFocusedNeighbor;
    const depthAllowsDetail =
      renderProfile.tier === 'compact' ||
      isHighlighted ||
      isFocusedNeighbor ||
      (!offAxis && depthBand === 'front') ||
      (!offAxis && visualCameraLod.zoomTier === 'near' && depthBand !== 'back') ||
      (!offAxis && visualCameraLod.zoomTier === 'middle' && depthBand === 'middle');
    const now = typeof window !== 'undefined' ? window.performance.now() : Date.now();
    const persistedLabels = persistedLabelStateRef.current;
    const existingPersistedLabel = persistedLabels.get(node.id);
    if (!showLabels && existingPersistedLabel) {
      persistedLabels.delete(node.id);
    }
    const stableFocusLabel = (
      richLabelFocus ||
      compactRelationshipEndpoint ||
      compactSelectedFocusLabel ||
      compactSearchPreviewLabel
    );
    const forcedReadableFocusLabel = Boolean(
      stableFocusLabel ||
      isRelationshipEndpoint ||
      selectedFocusNameplateVisible ||
      (isFocusHaloLabel && activeSelectionFocus) ||
      (isFocusedNeighbor && activeSelectionFocus && !isHardMuted)
    );
    const hoverReadableLabel = Boolean(hoverFocusedNeighbor && !isHardMuted);
    const fallbackLabelTarget = Math.min(
      prioritizedNodeIds.fallbackLabels.size,
      dimensions.width < 700
        ? 4
        : renderProfile.tier === 'massive'
          ? 4
          : renderProfile.tier === 'large'
            ? 7
            : 10
    );
    const ambientFallbackLabelIntent = Boolean(
      showLabels &&
      fallbackLabelAllows &&
      (labelDensity === 'priority' || labelDensity === 'rich' || detailMode === 'detail') &&
      screenReadableLabelLayouts.size < fallbackLabelTarget &&
      !query &&
      !selectedNode &&
      !selectedEdge &&
      detailMode !== 'performance' &&
      !isHardMuted &&
      !offAxis &&
      depthBand !== 'back' &&
      (
        renderProfile.tier !== 'massive' ||
        visualCameraLod.zoomTier !== 'far' ||
        nodes.length <= 1200
      )
    );
    const priorityNameplateIntent = Boolean(
      showLabels &&
      fallbackLabelAllows &&
      (labelDensity === 'priority' || labelDensity === 'rich' || detailMode === 'detail') &&
      !query &&
      !selectedNode &&
      !selectedEdge &&
      detailMode !== 'performance' &&
      !isHardMuted &&
      !offAxis &&
      depthBand !== 'back'
    );
    const anchorLabelCandidate = Boolean(
      showLabels &&
      labelDensity === 'focus' &&
      !navigationSettlingActive &&
      !query &&
      !selectedNode &&
      !selectedEdge &&
      visualActiveNodeIds.size === 0 &&
      visualActiveEdgeIds.size === 0 &&
      detailMode !== 'performance' &&
      prioritizedNodeIds.anchorLabels.has(node.id)
    );
    const persistedLabelUsable = Boolean(showLabels && existingPersistedLabel && existingPersistedLabel.visibleUntil > now);
    let labelLayout = screenReadableLabelLayouts.get(node.id);
    if (!labelLayout && stableFocusLabel) {
        labelLayout = {
          anchor: existingPersistedLabel?.layout.anchor || 'above',
          pinned: true,
          screenX: 0,
          screenY: 0,
      };
    } else if (!labelLayout && forcedReadableFocusLabel && isFocusedNeighbor && !stableFocusLabel) {
      const focusDx = Number(node.x || 0) - visualCameraLod.target.x;
      const focusDy = Number(node.y || 0) - visualCameraLod.target.y;
      const anchor: GraphNodeLabelAnchor = Math.abs(focusDx) > Math.abs(focusDy)
        ? focusDx >= 0 ? 'right' : 'left'
        : focusDy >= 0 ? 'above' : 'below';
      labelLayout = {
        anchor,
        pinned: isRelationshipEndpoint || isFocusHaloLabel,
        screenX: 0,
        screenY: 0,
      };
    } else if (!labelLayout && hoverFocusedNeighbor) {
      labelLayout = {
        anchor: Number(node.x || 0) > 0 ? 'left' : 'right',
        pinned: false,
        screenX: 0,
        screenY: 0,
      };
    } else if (!labelLayout && selectedFocusNameplateVisible) {
      const focusDx = Number(node.x || 0) - visualCameraLod.target.x;
      const focusDy = Number(node.y || 0) - visualCameraLod.target.y;
      const anchor: GraphNodeLabelAnchor = Math.abs(focusDx) > Math.abs(focusDy) * 1.05
        ? focusDx >= 0 ? 'right' : 'left'
        : focusDy >= 0 ? 'above' : 'below';
      labelLayout = {
        anchor,
        pinned: false,
        screenX: 0,
        screenY: 0,
      };
    } else if (!labelLayout && isFocusContextNode && activeSelectionFocus && showLabels && !isHardMuted) {
      const focusDx = Number(node.x || 0) - visualCameraLod.target.x;
      const focusDy = Number(node.y || 0) - visualCameraLod.target.y;
      const anchor: GraphNodeLabelAnchor = Math.abs(focusDx) > Math.abs(focusDy)
        ? focusDx >= 0 ? 'right' : 'left'
        : focusDy >= 0 ? 'above' : 'below';
      labelLayout = {
        anchor,
        pinned: false,
        screenX: 0,
        screenY: 0,
      };
    } else if (!labelLayout && ambientFallbackLabelIntent) {
      const focusDx = Number(node.x || 0) - visualCameraLod.target.x;
      const focusDy = Number(node.y || 0) - visualCameraLod.target.y;
      const anchor: GraphNodeLabelAnchor = Math.abs(focusDx) > Math.abs(focusDy) * 1.1
        ? focusDx >= 0 ? 'right' : 'left'
        : focusDy >= 0 ? 'above' : 'below';
      labelLayout = {
        anchor,
        pinned: false,
        screenX: 0,
        screenY: 0,
      };
    } else if (!labelLayout && priorityNameplateIntent) {
      const focusDx = Number(node.x || 0) - visualCameraLod.target.x;
      const focusDy = Number(node.y || 0) - visualCameraLod.target.y;
      const anchor: GraphNodeLabelAnchor = Math.abs(focusDx) > Math.abs(focusDy) * 1.08
        ? focusDx >= 0 ? 'right' : 'left'
        : focusDy >= 0 ? 'above' : 'below';
      labelLayout = {
        anchor,
        pinned: false,
        screenX: 0,
        screenY: 0,
      };
    }
    const selectionContextLayout = Boolean(labelLayout?.selectionContext);
    const labelCollisionAllows = Boolean(labelLayout);
    const stableAmbientIdentityVisible = Boolean(
      showLabels &&
      !query &&
      !selectedNode &&
      !selectedEdge &&
      detailMode !== 'performance' &&
      isConnectedFocus &&
      !isHardMuted &&
      rankedVisible &&
      !offAxis &&
      depthBand !== 'back' &&
      (
        labelCollisionAllows ||
        prioritizedNodeIds.icons.has(node.id) ||
        prioritizedLabelAllows ||
        fallbackLabelAllows
      ) &&
      (
        renderProfile.tier === 'compact' ||
        renderProfile.tier === 'standard' ||
        visualCameraLod.zoomTier !== 'far' ||
        nodes.length <= (renderProfile.tier === 'large' ? 1800 : 900)
      )
    );
    const stableAmbientLabelMotionAllowed = stableAmbientIdentityVisible && labelCollisionAllows;
    const orientationDetailAllowedDuringMotion = Boolean(
      showLabels &&
      (labelCollisionAllows || stableAmbientIdentityVisible) &&
      rankedVisible &&
      !isHardMuted &&
      (
        isHighlighted ||
        isFocusedNeighbor ||
        isRelationshipEndpoint ||
        (
          !offAxis &&
          depthBand !== 'back' &&
          !(renderProfile.tier === 'massive' && nodes.length > 3000 && visualCameraLod.zoomTier === 'far')
        )
      )
    );
    if (orientationDetailAllowedDuringMotion) {
      suppressWhileMoving = false;
    }
    const stableNavigationLabel = Boolean(
      navigationLabelsSettling &&
      labelCollisionAllows &&
      showLabels &&
      !query &&
      detailMode !== 'performance'
    );
    const navigationAllowsLabel =
      !navigationLabelsSettling ||
      stableFocusLabel ||
      persistedLabelUsable ||
      stableNavigationLabel;
    const selectedFocusLabelVisible = Boolean(activeSelectionFocus && labelCollisionAllows);
    const focusedLabelVisible = isFocusedNeighbor && (
      isRelationshipEndpoint ||
      (
        (showLabels || isFocusHaloLabel || forcedReadableFocusLabel) &&
        navigationAllowsLabel &&
        (!motionSimplificationActive || forcedReadableFocusLabel) &&
        (
          isFocusHaloLabel ||
          forcedReadableFocusLabel ||
          selectedFocusLabelVisible ||
          (query ? isSearchMatch : (nodes.length <= 120 || visualCameraLod.zoomTier !== 'far'))
        )
      )
    );
    const baseIconBudgetAllows =
      prioritizedNodeIds.icons.has(node.id) ||
      isHighlighted ||
      isFocusedNeighbor ||
      isFocusContextNode ||
      isRelationshipEndpoint ||
      labelCollisionAllows ||
      Boolean(query && isSearchMatch && !isHardMuted) ||
      renderProfile.tier === 'compact' ||
      renderProfile.tier === 'standard' ||
      (renderProfile.tier === 'large' && nodes.length <= 720 && !offAxis && depthBand !== 'back');
    const labelBudgetAllows = prioritizedLabelAllows || labelCollisionAllows;
    let labelStyle: GraphNodeLabelStyle = compactRelationshipEndpoint
      ? 'annotation'
      : compactSelectedFocusLabel
        ? 'annotation'
      : compactSearchPreviewLabel
        ? 'compact'
        : richLabelFocus
        ? 'rich'
        : anchorLabelCandidate
          ? 'anchor'
        : isFocusHaloLabel
          ? 'neighbor'
          : isFocusedNeighbor && Boolean(selectedNode || selectedEdge || query)
            ? 'annotation'
          : isFocusedNeighbor && labelDensity === 'focus'
            ? 'anchor'
          : 'compact';
    const searchLabelVisible = Boolean(
      showLabels &&
      query &&
      isSearchMatch &&
      labelCollisionAllows &&
      depthAllowsDetail &&
      navigationAllowsLabel &&
      !suppressWhileMoving &&
      !isHardMuted
    );
    const focusHaloLabelVisible = Boolean(
      isFocusHaloLabel &&
      labelCollisionAllows &&
      depthAllowsDetail &&
      navigationAllowsLabel &&
      !suppressWhileMoving &&
      !isHardMuted
    );
    const stableAmbientLayoutLabel = Boolean(
      showLabels &&
      !query &&
      labelDensity !== 'focus' &&
      labelCollisionAllows &&
      screenReadableLabelLayouts.has(node.id)
    );
    const readableLayoutLabelVisible = Boolean(
      showLabels &&
      labelCollisionAllows &&
      screenReadableLabelLayouts.has(node.id) &&
      navigationAllowsLabel &&
      (!suppressWhileMoving || stableAmbientLabelMotionAllowed) &&
      !isHardMuted
    );
    const selectionContextLabelVisible = Boolean(
      showLabels &&
      activeSelectionFocus &&
      !query &&
      selectionContextLayout &&
      labelCollisionAllows &&
      screenReadableLabelLayouts.has(node.id) &&
      !isConnectedFocus &&
      !motionSimplificationActive &&
      detailMode !== 'performance'
    );
    const ambientLabelVisible = isConnectedFocus &&
      showLabels &&
      navigationAllowsLabel &&
      labelCollisionAllows &&
      (depthAllowsDetail || stableAmbientLayoutLabel) &&
      (!suppressWhileMoving || stableAmbientLabelMotionAllowed) &&
      (renderProfile.tier === 'compact' || labelBudgetAllows);
    const strictlyLabelVisible = compactSelectedFocusLabel ||
      richLabelFocus ||
      searchLabelVisible ||
      focusHaloLabelVisible ||
      readableLayoutLabelVisible ||
      selectionContextLabelVisible ||
      (priorityNameplateIntent && labelCollisionAllows) ||
      (selectedFocusNameplateVisible && labelCollisionAllows) ||
      (forcedReadableFocusLabel && labelCollisionAllows) ||
      (focusedLabelVisible && labelCollisionAllows) ||
      ambientLabelVisible;
    let labelVisible = strictlyLabelVisible;
    let labelOpacity = 1;
    const identitySymbolIntent = Boolean(
      isHighlighted ||
      isFocusedNeighbor ||
      isFocusContextNode ||
      isRelationshipEndpoint ||
      selectedFocusSymbolVisible ||
      selectedFocusNameplateVisible ||
      forcedReadableFocusLabel ||
      selectionContextLabelVisible ||
      stableAmbientIdentityVisible ||
      labelCollisionAllows ||
      labelVisible
    );
    const iconBudgetAllows = baseIconBudgetAllows || identitySymbolIntent;
    const compactViewport = dimensions.width < 700;
      const baseLabelWorldScale = labelStyle === 'rich'
        ? visualCameraLod.zoomTier === 'near'
          ? compactViewport ? 0.56 : 0.62
          : visualCameraLod.zoomTier === 'middle'
            ? compactViewport ? 0.82 : 0.94
            : compactViewport ? 1.04 : 1.24
        : labelStyle === 'annotation'
          ? visualCameraLod.zoomTier === 'near'
            ? 0.58
            : visualCameraLod.zoomTier === 'middle'
              ? 0.82
              : 1
              : visualCameraLod.zoomTier === 'near'
              ? compactViewport ? 0.72 : 0.78
              : visualCameraLod.zoomTier === 'middle'
                ? compactViewport ? 1.02 : 1.1
                : compactViewport ? 1.34 : 1.58;
      const computedLabelWorldScale = baseLabelWorldScale;
      const activeIdentityLabelScaleBoost = activeIdentityVisible
        ? isolatedSelectedNodeVisual
          ? visualCameraLod.zoomTier === 'far' ? 2.12 : visualCameraLod.zoomTier === 'middle' ? 1.52 : 1.16
          : compactSelectedFocusLabel
          ? visualCameraLod.zoomTier === 'far' ? 1.34 : visualCameraLod.zoomTier === 'middle' ? 1.18 : 1
          : labelStyle === 'rich'
            ? visualCameraLod.zoomTier === 'far' ? 1.78 : visualCameraLod.zoomTier === 'middle' ? 1.28 : 1
            : visualCameraLod.zoomTier === 'far' ? 1.82 : visualCameraLod.zoomTier === 'middle' ? 1.34 : 1.04
        : 1;
      const activeIdentityMinLabelWorldScale = computedLabelWorldScale * activeIdentityLabelScaleBoost;
      const persistedLabelWorldScale = Math.max(
        existingPersistedLabel?.worldScale ?? computedLabelWorldScale,
        activeIdentityMinLabelWorldScale
      );
      let labelWorldScale = navigationLabelsSettling && existingPersistedLabel
        ? persistedLabelWorldScale
        : Math.max(computedLabelWorldScale, activeIdentityMinLabelWorldScale);
      const viewportIdentityScale = dimensions.width >= 1600
        ? 1.08
        : dimensions.width >= 1280
        ? 1.04
        : dimensions.width < 700
          ? 0.9
          : 1;
    const graphDensityIdentityScale = nodes.length <= 90
      ? 1.06
      : nodes.length <= 180
        ? 1.03
        : renderProfile.tier === 'large'
          ? 0.97
          : renderProfile.tier === 'massive'
            ? 0.92
            : 1;
      const focusIdentityScale = isHighlighted
        ? isolatedSelectedNodeVisual ? 1.3 : 1.16
        : activeIdentityVisible
          ? 1.12
        : isFocusedNeighbor || isRelationshipEndpoint
          ? 1.04
          : labelVisible
            ? 0.96
            : 1;
      const symbolWorldScale = Math.max(
        0.96,
        Math.min(isolatedSelectedNodeVisual ? 1.62 : activeIdentityVisible ? 1.36 : 1.24, viewportIdentityScale * graphDensityIdentityScale * focusIdentityScale)
      );
    if (
      isFocusedNeighbor &&
      Boolean(selectedNode || selectedEdge || query) &&
      !isHighlighted &&
      !isRelationshipEndpoint &&
      labelStyle === 'annotation'
    ) {
      labelWorldScale *= 0.96;
    }
    if (hoverReadableLabel) {
      labelWorldScale = Math.max(
        labelWorldScale,
        computedLabelWorldScale * (visualCameraLod.zoomTier === 'far' ? 1.2 : 1.08)
      );
    }
    if (compactRelationshipEndpoint) {
      const endpointScaleFloor = visualCameraLod.zoomTier === 'far'
        ? 1.12
        : visualCameraLod.zoomTier === 'middle'
          ? 0.98
          : 0.84;
      const endpointScaleCap = visualCameraLod.zoomTier === 'far'
        ? 1.24
        : visualCameraLod.zoomTier === 'middle'
          ? 1.08
          : 0.92;
      labelWorldScale = Math.min(Math.max(labelWorldScale, endpointScaleFloor), endpointScaleCap);
    }
    const routeContextLabel = Boolean(
      selectedHoverRouteActive &&
      isFocusedNeighbor &&
      !isHardMuted &&
      !query
    );
    const labelHoldMs = stableFocusLabel
      ? 60000
      : routeContextLabel
        ? GRAPH_ROUTE_LABEL_HOLD_MS
      : hoverReadableLabel
        ? 5600
        : isFocusHaloLabel
        ? 4200
        : navigationLabelsSettling
          ? GRAPH_LABEL_FADE_SETTLE_MS * 2
          : 14000;
      if (strictlyLabelVisible && labelLayout) {
      if (
        existingPersistedLabel &&
        stableFocusLabel &&
        !compactRelationshipEndpoint &&
        !compactSelectedFocusLabel &&
        !compactSearchPreviewLabel
      ) {
        labelLayout = {
          ...existingPersistedLabel.layout,
          pinned: true,
        };
        labelStyle = existingPersistedLabel.style;
        labelWorldScale = persistedLabelWorldScale;
      } else if (
        existingPersistedLabel &&
        existingPersistedLabel.visibleUntil > now &&
        existingPersistedLabel.style === labelStyle &&
        existingPersistedLabel.retention === 'ambient'
      ) {
        labelLayout = existingPersistedLabel.layout;
        labelWorldScale = persistedLabelWorldScale;
      } else if (navigationLabelsSettling && existingPersistedLabel && !stableFocusLabel) {
        labelLayout = existingPersistedLabel.layout;
        labelStyle = existingPersistedLabel.style;
        labelWorldScale = persistedLabelWorldScale;
      }
      const nextPersistedLabel: GraphPersistedLabel = {
        layout: labelLayout,
        style: labelStyle,
        worldScale: labelWorldScale,
        visibleUntil: now + labelHoldMs,
        visibleSince: existingPersistedLabel?.visibleSince || now,
          baseOpacity: stableFocusLabel ? 1 : routeContextLabel ? 0.96 : hoverReadableLabel ? 0.98 : selectionContextLabelVisible ? 0.82 : isFocusHaloLabel ? 0.86 : labelStyle === 'anchor' ? 0.96 : navigationLabelsSettling ? 0.74 : 0.9,
        fadeOutMs: stableFocusLabel
          ? GRAPH_LABEL_FADE_IN_MS
          : routeContextLabel
            ? GRAPH_ROUTE_LABEL_FADE_OUT_MS
          : hoverReadableLabel
            ? 4200
          : isFocusHaloLabel
            ? 3600
            : navigationLabelsSettling
              ? GRAPH_LABEL_FADE_SETTLE_MS
              : GRAPH_LABEL_FADE_OUT_MS,
        pinned: stableFocusLabel || routeContextLabel || hoverReadableLabel,
        retention: stableFocusLabel || isFocusHaloLabel || routeContextLabel || hoverReadableLabel ? 'focus' : 'ambient',
      };
      labelOpacity = getGraphPersistedLabelOpacity(nextPersistedLabel, now);
      persistedLabels.set(node.id, nextPersistedLabel);
    } else {
      const persistedLabel = persistedLabels.get(node.id);
      const focusRetainedLabel = persistedLabel?.retention === 'focus';
      const routeRetainedLabel = Boolean(focusRetainedLabel && selectedNode && !query);
      const keepAmbientPersistedLabel = Boolean(
        showLabels &&
        persistedLabel &&
        persistedLabel.visibleUntil > now &&
        !query &&
        !selectedNode &&
        !selectedEdge &&
        navigationLabelsSettling
      );
      const canUsePersistedLabel =
        showLabels &&
        Boolean(persistedLabel) &&
        persistedLabel!.visibleUntil > now &&
        (!compactRelationshipEndpoint || Boolean(labelLayout)) &&
        (showLabels || Boolean(query && isSearchMatch) || isHighlighted || isRelationshipEndpoint || (focusRetainedLabel && isFocusHaloLabel)) &&
        (isConnectedFocus || keepAmbientPersistedLabel || routeRetainedLabel) &&
        (!isHardMuted || keepAmbientPersistedLabel || routeRetainedLabel) &&
        (
          stableFocusLabel ||
          navigationLabelsSettling ||
          !motionSimplificationActive ||
          isHighlighted ||
          isFocusedNeighbor
        );
      if (canUsePersistedLabel && persistedLabel) {
        persistedLabel.pinned = stableFocusLabel;
        if (!stableFocusLabel) {
            persistedLabel.baseOpacity = Math.min(persistedLabel.baseOpacity, routeRetainedLabel ? 0.84 : navigationLabelsSettling ? 0.78 : 0.88);
        }
        if (navigationLabelsSettling && !stableFocusLabel) {
          persistedLabel.visibleUntil = Math.min(persistedLabel.visibleUntil, now + GRAPH_LABEL_FADE_SETTLE_MS);
        } else if (routeRetainedLabel) {
          persistedLabel.visibleUntil = Math.min(persistedLabel.visibleUntil, now + GRAPH_ROUTE_LABEL_FADE_OUT_MS);
        } else if (!stableFocusLabel) {
          persistedLabel.visibleUntil = Math.min(persistedLabel.visibleUntil, now + GRAPH_LABEL_FADE_OUT_MS);
        }
        labelLayout = persistedLabel.layout;
        labelStyle = persistedLabel.style;
        labelWorldScale = persistedLabel.worldScale ?? computedLabelWorldScale;
        labelVisible = true;
        labelOpacity = getGraphPersistedLabelOpacity(persistedLabel, now);
      } else if (
        persistedLabel &&
        persistedLabel.visibleUntil > now &&
        !compactRelationshipEndpoint &&
        showLabels &&
        !query
      ) {
        persistedLabel.pinned = false;
        persistedLabel.fadeOutMs = navigationLabelsSettling
          ? GRAPH_LABEL_FADE_SETTLE_MS
          : GRAPH_LABEL_FADE_OUT_MS * 0.62;
        persistedLabel.visibleUntil = Math.min(persistedLabel.visibleUntil, now + persistedLabel.fadeOutMs);
          persistedLabel.baseOpacity = Math.min(persistedLabel.baseOpacity, navigationLabelsSettling ? 0.62 : 0.56);
        labelLayout = persistedLabel.layout;
        labelStyle = persistedLabel.style;
        labelWorldScale = persistedLabel.worldScale ?? computedLabelWorldScale;
        labelOpacity = getGraphPersistedLabelOpacity(persistedLabel, now);
        labelVisible = labelOpacity > 0.006;
      } else if (persistedLabel) {
        persistedLabels.delete(node.id);
      }
    }
    if (compactRelationshipEndpoint && labelVisible) {
      const endpointScaleFloor = visualCameraLod.zoomTier === 'far'
        ? 1.12
        : visualCameraLod.zoomTier === 'middle'
          ? 0.98
          : 0.84;
      const endpointScaleCap = visualCameraLod.zoomTier === 'far'
        ? 1.24
        : visualCameraLod.zoomTier === 'middle'
          ? 1.08
          : 0.92;
      labelWorldScale = Math.min(Math.max(labelWorldScale, endpointScaleFloor), endpointScaleCap);
    }
      const passiveSymbolNodeCount = renderProfile.tier === 'compact'
        ? Number.POSITIVE_INFINITY
        : renderProfile.tier === 'standard'
          ? Number.POSITIVE_INFINITY
          : renderProfile.tier === 'large'
            ? 1200
            : 520;
    const symbolBudgetAllows =
      identitySymbolIntent ||
      iconBudgetAllows ||
      labelCollisionAllows ||
      orientationDetailAllowedDuringMotion ||
      (rankedVisible && !offAxis && depthBand !== 'back') ||
      (
        renderProfile.tier !== 'massive' &&
        rankedVisible &&
        visualCameraLod.zoomTier !== 'far'
      );
    const symbolMotionAllows =
      identitySymbolIntent ||
      stableAmbientIdentityVisible ||
      !suppressWhileMoving ||
      isHighlighted ||
      isFocusedNeighbor ||
      isRelationshipEndpoint ||
      labelVisible ||
      (showLabels && visualCameraLod.zoomTier !== 'far' && renderProfile.tier !== 'massive');
    const priorityTypeSymbolVisible = isConnectedFocus &&
      symbolMotionAllows &&
      symbolBudgetAllows &&
      rankedVisible &&
      (
        depthAllowsDetail ||
        visualCameraLod.zoomTier !== 'far' ||
        Boolean(selectedNode || selectedEdge || query)
      );
    const ambientTypeSymbolVisible = isConnectedFocus &&
      !isHardMuted &&
      symbolMotionAllows &&
      symbolBudgetAllows &&
      rankedVisible &&
      (
        renderProfile.tier !== 'massive' ||
        visualCameraLod.zoomTier !== 'far' ||
        nodes.length <= 1800
      );
    const passiveIconVisible = isConnectedFocus &&
      (!suppressWhileMoving || stableAmbientIdentityVisible) &&
      iconBudgetAllows &&
      (
        depthAllowsDetail ||
        labelCollisionAllows ||
        Boolean(query && isSearchMatch) ||
        renderProfile.tier === 'compact' ||
        (!offAxis && visualCameraLod.zoomTier !== 'far')
      ) &&
      (
        isFocusedNeighbor ||
        labelVisible ||
        labelCollisionAllows ||
        nodes.length <= passiveSymbolNodeCount ||
        (visualCameraLod.zoomTier === 'near' && nodes.length <= 340) ||
        nodes.length <= 42
      );
    const persistentTypeSymbolVisible = isConnectedFocus &&
      !isHardMuted &&
      rankedVisible &&
      prioritizedNodeIds.icons.has(node.id) &&
      (
        isHighlighted ||
        isFocusedNeighbor ||
        isRelationshipEndpoint ||
        labelVisible ||
        labelCollisionAllows ||
        visualCameraLod.zoomTier !== 'far' ||
        (!offAxis && depthBand !== 'back') ||
        nodes.length <= passiveSymbolNodeCount
      ) &&
      (
        renderProfile.tier !== 'massive' ||
        visualCameraLod.zoomTier !== 'far' ||
        nodes.length <= 1200 ||
        isHighlighted ||
        isFocusedNeighbor ||
        isRelationshipEndpoint
      );
    const stableTypeSymbolVisible = isConnectedFocus &&
      !isHardMuted &&
      iconBudgetAllows &&
      rankedVisible &&
      (
        isHighlighted ||
        isFocusedNeighbor ||
        labelVisible ||
        labelCollisionAllows ||
        fallbackLabelAllows ||
        (!offAxis && depthBand !== 'back')
      ) &&
      (
        renderProfile.tier !== 'massive' ||
        visualCameraLod.zoomTier !== 'far' ||
        nodes.length <= 1200
      );
    const orientationTypeSymbolVisible = isConnectedFocus &&
      !isHardMuted &&
      symbolMotionAllows &&
      symbolBudgetAllows &&
      (
        isHighlighted ||
        isFocusedNeighbor ||
        labelCollisionAllows ||
        (
          rankedVisible &&
          !offAxis &&
          (
            renderProfile.tier === 'compact' ||
            renderProfile.tier === 'standard' ||
            depthBand !== 'back' ||
            visualCameraLod.zoomTier === 'near'
          )
        )
      );
    const landmarkTypeSymbolVisible = isConnectedFocus &&
      !isHardMuted &&
      symbolMotionAllows &&
      rankedVisible &&
      prioritizedNodeIds.icons.has(node.id) &&
      (
        !offAxis ||
        visualCameraLod.zoomTier !== 'far' ||
        isHighlighted ||
        isFocusedNeighbor
      ) &&
      (
        depthBand !== 'back' ||
        isHighlighted ||
        isFocusedNeighbor ||
        visualCameraLod.zoomTier === 'near'
      );
      const hoverAmbientSymbolVisible = hoverOnlyFocusActive &&
        isMuted &&
        !suppressWhileMoving &&
        iconBudgetAllows &&
        rankedVisible &&
        !offAxis &&
        (depthBand !== 'back' || visualCameraLod.zoomTier === 'near') &&
        (
          renderProfile.tier === 'compact' ||
          renderProfile.tier === 'standard' ||
          visualCameraLod.zoomTier !== 'far'
        );
    const iconVisible = identitySymbolIntent ||
      isHighlighted ||
      isFocusedNeighbor ||
      selectedFocusSymbolVisible ||
      labelVisible ||
      labelCollisionAllows ||
      priorityTypeSymbolVisible ||
      ambientTypeSymbolVisible ||
      persistentTypeSymbolVisible ||
      passiveIconVisible ||
      stableTypeSymbolVisible ||
      orientationTypeSymbolVisible ||
      landmarkTypeSymbolVisible ||
      hoverAmbientSymbolVisible;
    const ambientCustomShapeVisible = isConnectedFocus &&
      (!suppressWhileMoving || stableAmbientIdentityVisible) &&
      (
        renderProfile.tier === 'large'
          ? rankedVisible
          : renderProfile.tier === 'massive'
            ? rankedVisible && (
              depthAllowsDetail ||
              visualCameraLod.zoomTier !== 'far' ||
              priorityTypeSymbolVisible ||
              nodes.length <= 5000
            )
            : true
      );
    const customShapeVisible = renderProfile.alwaysUseCustomNodes ||
      isHighlighted ||
      isFocusedNeighbor ||
      isFocusContextNode ||
      labelVisible ||
      labelCollisionAllows ||
      ambientTypeSymbolVisible ||
      persistentTypeSymbolVisible ||
      passiveIconVisible ||
      stableTypeSymbolVisible ||
      orientationTypeSymbolVisible ||
      landmarkTypeSymbolVisible ||
      hoverAmbientSymbolVisible ||
      ambientCustomShapeVisible;
      const lowDetail = renderProfile.tier !== 'compact' &&
        !isHighlighted &&
        !isFocusedNeighbor &&
        !iconVisible &&
        !(rankedVisible && !offAxis && visualCameraLod.zoomTier !== 'far') &&
        (Boolean(selectedEdge) || !labelVisible || visualCameraLod.zoomTier === 'far');

    if (!customShapeVisible && (offAxis || depthBand === 'back' || !iconVisible)) {
      return null as unknown as THREE.Object3D;
    }

    const cacheKey = [
      node.id,
      node.label,
      node.type,
      node.val,
      relationSummary?.relations || 0,
      isHighlighted,
      isRelationshipEndpoint,
      isFocusedNeighbor,
      isFocusContextNode,
      isAmbientFocusContextNode,
      isHardMuted,
      isSoftMuted,
      isSoftMuted ? softMuteStrengthForNode.toFixed(2) : '0',
      routeFocusVisualActive,
      routeContextFocusActive ? routeFocusWeight.toFixed(2) : '0',
      selectionContextLabelVisible,
      labelVisible,
      iconVisible,
      customShapeVisible,
      labelStyle,
      labelLayout?.anchor || 'none',
      labelWorldScale.toFixed(2),
      symbolWorldScale.toFixed(2),
      focusHaloLabelHint ? `${focusHaloLabelHint.type}:${focusHaloLabelHint.count}` : '',
      lowDetail,
      renderProfile.tier,
      relationTypeMix.slice(0, 5).map(({ type, count }) => `${type}:${count}`).join(','),
    ].join('|');
    const shouldAnimateNode = isSelected ||
      isHovered ||
      isRelationshipEndpoint ||
      (!query && isFocusedNeighbor && renderProfile.tier !== 'massive');
    const cached = nodeObjectCacheRef.current.get(node.id);
    if (cached?.key === cacheKey) {
      if (labelVisible) {
        setGraphNodeObjectLabelOpacity(cached.object, labelOpacity);
      }
      if (shouldAnimateNode) {
        animatedNodeObjectsRef.current.set(node.id, cached.object);
      } else {
        animatedNodeObjectsRef.current.delete(node.id);
      }
      return cached.object;
    }

    const object = createGraphNodeObject({
      node,
      relationSummary,
      highlighted: isHighlighted,
      focused: isFocusedNeighbor,
      muted: isHardMuted,
      softMuted: isSoftMuted,
      softMuteStrength: softMuteStrengthForNode,
      labelVisible,
      iconVisible,
      labelStyle,
      labelLayout,
      labelOpacity,
      labelWorldScale,
      symbolWorldScale,
      relationHint: focusHaloLabelHint,
      relationTypeMix: isHighlighted || isFocusedNeighbor || isRelationshipEndpoint ? relationTypeMix : undefined,
      lowDetail,
      contextVisible: isAmbientFocusContextNode,
    });
    if (shouldAnimateNode) {
      animatedNodeObjectsRef.current.set(node.id, object);
    } else {
      animatedNodeObjectsRef.current.delete(node.id);
    }
    nodeObjectCacheRef.current.set(node.id, { key: cacheKey, object });
      return object;
      }, [activeIdentityNodeIds, compactSelectedEdgeEndpointLabels, compactSelectedNodeLabel, detailMode, dimensions.width, focusContextNodeIds, focusHaloLabelHints, focusHaloLabelIds, focusNodeId, hoverFocusVisualActive, hoverFocusWeight, hoveredNode, labelDensity, motionSimplificationActive, navigationSettlingActive, nodeRelationSummary, nodeRelationTypeMix, nodes.length, prioritizedNodeIds, renderProfile, routeFocusVisualActive, routeFocusWeight, screenReadableLabelLayouts, searchTerm, selectedEdge, selectedEdgeEndpointIds, selectedHoverRouteActive, selectedNode, showLabels, visualActiveEdgeIds.size, visualActiveNodeIds, visualCameraLod]);

      const linkThreeObject = useCallback((link: RenderGraphEdge) => {
        const showLabel = shouldShowLinkLabel(link);
        const showMarker = !showLabel && shouldShowRelationMarker(link);
        if (!showLabel && !showMarker) return null as unknown as THREE.Object3D;
        const directlyHovered = hoveredLink?.id === link.id;
        const directlySelected = selectedEdge === link.id;
        const matchesSearch = linkMatchesSearch(link);
        const muted = !isLinkActive(link);
        const selectedHoverRouteLink = routeFocusVisualActive && routeFocusWeight > 0.08 && !muted;
        const highlighted = directlyHovered || directlySelected || matchesSearch || selectedHoverRouteLink;
        const neighborhoodMarkerEmphasis = Boolean(
          showMarker &&
          !highlighted &&
          !muted &&
          activeEdgeIds.size > 0 &&
          activeEdgeIds.size <= 12 &&
          !selectedEdge &&
          !searchTerm.trim()
        );
        const markerHighlighted = highlighted || neighborhoodMarkerEmphasis;
        const sourceId = getLinkEndpointId(link.source);
        const targetId = getLinkEndpointId(link.target);
        const sourceNode = typeof link.source === 'string' ? (sourceId ? nodeById.get(sourceId) : undefined) : link.source;
        const targetNode = typeof link.target === 'string' ? (targetId ? nodeById.get(targetId) : undefined) : link.target;
        const relationTypes = getLinkRelationTypeCounts(link, 4);
        const richRoute =
          directlyHovered ||
          directlySelected ||
          selectedHoverRouteLink ||
          Boolean(searchTerm.trim() && matchesSearch && !selectedEdge);
        const relationMarkerFocusNodeId = selectedNode || hoveredNode || focusNodeId;
        const relationMarkerPlacement = showMarker
          ? getRelationMarkerPlacement(link, relationMarkerFocusNodeId)
          : undefined;
        const relationMarkerLane = showMarker
          ? getRelationMarkerLane(link, relationMarkerFocusNodeId)
          : 0;
        const relationMarkerFocused = Boolean(showMarker && relationMarkerFocusNodeId);
        const cacheKey = [
          link.id,
          showLabel ? getRelationDisplayLabel(link) : `relation-marker:${getRelationDisplayLabel(link)}`,
          showLabel ? highlighted : markerHighlighted,
          muted,
          richRoute,
          relationMarkerPlacement?.toFixed(2) || '',
          relationMarkerLane,
          sourceNode?.label || sourceId || '',
          targetNode?.label || targetId || '',
          relationTypes.map((item) => `${item.type}:${item.count}`).join(','),
        ].join('|');
        const cached = linkObjectCacheRef.current.get(link.id);
        if (cached?.key === cacheKey) {
          cached.object.userData.relationMarkerPlacement = relationMarkerPlacement;
          cached.object.userData.relationMarkerLane = relationMarkerLane;
          cached.object.userData.relationMarkerFocused = relationMarkerFocused;
          return cached.object;
        }

        const object = showLabel
          ? createLinkLabelSprite(link, highlighted, muted, {
            rich: richRoute,
            sourceLabel: sourceNode?.label || sourceId,
            targetLabel: targetNode?.label || targetId,
            relationTypes,
          })
          : createRelationMarkerSprite(link, markerHighlighted, muted);
        if (!object) return null as unknown as THREE.Object3D;
        object.userData.relationMarkerPlacement = relationMarkerPlacement;
        object.userData.relationMarkerLane = relationMarkerLane;
        object.userData.relationMarkerFocused = relationMarkerFocused;
        linkObjectCacheRef.current.set(link.id, { key: cacheKey, object });
        return object;
      }, [activeEdgeIds.size, focusNodeId, getLinkEndpointId, hoveredLink, hoveredNode, isLinkActive, linkMatchesSearch, nodeById, routeFocusVisualActive, routeFocusWeight, searchTerm, selectedEdge, selectedNode, shouldShowLinkLabel, shouldShowRelationMarker]);

  const linkPositionUpdate = useCallback((
    object: THREE.Object3D | undefined,
    { start, end }: { start: { x: number; y: number; z: number }; end: { x: number; y: number; z: number } }
  ) => {
    if (!object?.position) return false;
    const relationMarker = object.userData.relationMarker === true;
    const focusedRelationMarker = relationMarker && object.userData.relationMarkerFocused === true;
    const richLinkLabel = object.userData.richLinkLabel === true;
    const markerPlacement = Number(object.userData.relationMarkerPlacement);
    const markerLane = Number(object.userData.relationMarkerLane || 0);
    const placement = relationMarker
      ? Number.isFinite(markerPlacement)
        ? markerPlacement
        : 0.57
      : richLinkLabel ? 0.42 : 0.5;
    const middle = {
      x: start.x + (end.x - start.x) * placement,
        y: start.y + (end.y - start.y) * placement,
        z: start.z + (end.z - start.z) * placement,
      };
      const deltaX = end.x - start.x;
      const deltaY = end.y - start.y;
      const deltaZ = end.z - start.z;
      const distance = Math.hypot(deltaX, deltaY, deltaZ);
      const lift = relationMarker
        ? focusedRelationMarker
          ? Math.min(20, Math.max(7.2, distance * 0.044))
          : Math.min(13, Math.max(4.2, distance * 0.028))
        : richLinkLabel
          ? Math.min(18, Math.max(7, distance * 0.038))
          : Math.min(13, Math.max(4, distance * 0.026));
      const lane = relationMarker && Number.isFinite(markerLane) ? markerLane : 0;
      const laneSpread = relationMarker
        ? focusedRelationMarker
          ? Math.min(13.5, Math.max(4.8, distance * 0.028))
          : Math.min(8.4, Math.max(3.2, distance * 0.018))
        : 0;
      const flatLength = Math.hypot(deltaX, deltaY) || distance || 1;
      const laneX = (-deltaY / flatLength) * lane * laneSpread;
      const laneY = (deltaX / flatLength) * lane * laneSpread;
      object.position.set(middle.x + laneX, middle.y + laneY, middle.z + lift + Math.abs(lane) * 0.85);
      return true;
    }, []);

  const linkColor = useCallback((link: RenderGraphEdge) => {
    const relationColor = getRelationshipColor(link.type || link.label);
    const active = isLinkActive(link);
    const hasFocus = visualActiveEdgeIds.size > 0 || visualActiveNodeIds.size > 0;
    const searchActive = Boolean(searchTerm.trim());
    const matchesSearch = Boolean(searchActive && linkMatchesSearch(link));
    const hoverOnlyFocusActive = hoverFocusVisualActive;
    const routeContextFocusActive = Boolean(routeFocusVisualActive && hasFocus && !searchActive);
    const directlyFocused = hoveredLink?.id === link.id || selectedEdge === link.id;
    const searchFocused = matchesSearch && (!hasFocus || active);
    const selectedNeighborhoodLink = Boolean(
      selectedNode &&
      active &&
      visualActiveEdgeIds.size > 0 &&
      visualActiveEdgeIds.size <= 12 &&
      !selectedEdge &&
      !searchTerm.trim()
    );
    const ambientOpacity = renderProfile.tier === 'compact'
      ? 0.34
      : renderProfile.tier === 'standard'
        ? 0.26
        : renderProfile.tier === 'large'
          ? 0.18
          : 0.1;
    const focusedAmbientBase = renderProfile.tier === 'massive' ? 0.14 : 0.22;
    const focusedAmbientTarget = renderProfile.tier === 'massive' ? 0.22 : 0.38;
    const inactiveFocusBase = renderProfile.tier === 'massive' ? 0.05 : renderProfile.tier === 'large' ? 0.068 : 0.082;
    const inactiveFocusTarget = renderProfile.tier === 'massive' ? 0.12 : renderProfile.tier === 'large' ? 0.16 : 0.21;
    const inactiveRouteContextOpacity = renderProfile.tier === 'massive' ? 0.1 : renderProfile.tier === 'large' ? 0.13 : 0.16;
    const focusedAmbientOpacity = hoverOnlyFocusActive
      ? lerpGraphValue(focusedAmbientBase, focusedAmbientTarget, hoverFocusWeight)
      : routeContextFocusActive
        ? lerpGraphValue(focusedAmbientBase, focusedAmbientTarget, routeFocusWeight)
        : focusedAmbientBase;
    const inactiveFocusOpacity = hoverOnlyFocusActive
      ? lerpGraphValue(inactiveFocusBase, inactiveFocusTarget, hoverFocusWeight)
      : routeContextFocusActive
        ? lerpGraphValue(inactiveFocusBase, inactiveRouteContextOpacity, routeFocusWeight)
        : inactiveFocusBase;
    const opacity = directlyFocused
      ? 0.86
      : searchFocused
        ? 0.58
        : selectedNeighborhoodLink
          ? lerpGraphValue(focusedAmbientBase, 0.36, routeContextFocusActive ? routeFocusWeight : 1)
        : active
          ? hasFocus ? focusedAmbientOpacity : ambientOpacity
          : hasFocus ? inactiveFocusOpacity : Math.min(0.07, ambientOpacity * 0.28);
    return alpha(relationColor, opacity);
  }, [hoverFocusVisualActive, hoverFocusWeight, hoveredLink, isLinkActive, linkMatchesSearch, renderProfile.tier, routeFocusVisualActive, routeFocusWeight, searchTerm, selectedEdge, selectedNode, visualActiveEdgeIds.size, visualActiveNodeIds.size]);

  const linkWidth = useCallback((link: RenderGraphEdge) => {
    const relationCount = Math.max(1, Number(link.relationCount || 1));
    const active = isLinkActive(link);
    const hasFocus = visualActiveEdgeIds.size > 0 || visualActiveNodeIds.size > 0;
    const searchActive = Boolean(searchTerm.trim());
    const matchesSearch = Boolean(searchActive && linkMatchesSearch(link));
    const hoverOnlyFocusActive = hoverFocusVisualActive;
    const routeContextFocusActive = Boolean(routeFocusVisualActive && hasFocus && !searchActive);
    const directlyFocused = hoveredLink?.id === link.id || selectedEdge === link.id;
    const searchFocused = matchesSearch && (!hasFocus || active);
    if (!active) {
      return hasFocus
        ? hoverOnlyFocusActive
          ? lerpGraphValue(0.1, renderProfile.tier === 'massive' ? 0.13 : 0.23, hoverFocusWeight)
          : routeContextFocusActive
            ? lerpGraphValue(0.05, renderProfile.tier === 'massive' ? 0.08 : renderProfile.tier === 'large' ? 0.09 : 0.11, routeFocusWeight)
            : 0.05
        : 0.16;
    }
    const widthScale = renderProfile.tier === 'massive' ? 0.44 : renderProfile.tier === 'large' ? 0.54 : renderProfile.tier === 'standard' ? 0.7 : 0.92;
    const relationScale = Math.log2(relationCount + 1);
    if (directlyFocused) return Math.max(1.08, Math.min(2.2, relationScale * 0.48 * widthScale));
    if (searchFocused) return Math.max(0.72, Math.min(1.42, relationScale * 0.32 * widthScale));
    return hasFocus
      ? Math.max(0.22, Math.min(0.74, relationScale * 0.2 * widthScale))
      : Math.max(renderProfile.tier === 'compact' ? 0.24 : 0.14, Math.min(0.86, relationScale * 0.16 * widthScale));
  }, [hoverFocusVisualActive, hoverFocusWeight, hoveredLink, isLinkActive, linkMatchesSearch, renderProfile.tier, routeFocusVisualActive, routeFocusWeight, searchTerm, selectedEdge, visualActiveEdgeIds.size, visualActiveNodeIds.size]);

  const linkCurvature = useCallback((link: RenderGraphEdge) => {
    const profile = getLinkCurveProfile(link);
    const directlyFocused = hoveredLink?.id === link.id || selectedEdge === link.id;
    const searchFocused = Boolean(searchTerm.trim() && linkMatchesSearch(link));
    const focusBoost = directlyFocused || searchFocused ? 1.18 : 1;
    return profile.curvature * focusBoost;
  }, [hoveredLink, linkMatchesSearch, searchTerm, selectedEdge]);

  const linkCurveRotation = useCallback((link: RenderGraphEdge) => getLinkCurveProfile(link).rotation, []);

  const linkArrowLength = useCallback((link: RenderGraphEdge) => {
    const active = isLinkActive(link);
    const hasFocus = visualActiveEdgeIds.size > 0 || visualActiveNodeIds.size > 0;
    const searchActive = Boolean(searchTerm.trim());
    const matchesSearch = searchActive && linkMatchesSearch(link);
    const hoverOnlyFocusActive = hoverFocusVisualActive && !searchActive;
    const directlyFocused = hoveredLink?.id === link.id || selectedEdge === link.id;
    const searchFocused = matchesSearch && (!hasFocus || active);
    const emphasized = directlyFocused || searchFocused;
    const ambientDirectionCue = !hasFocus && !searchActive && ambientDirectionCueLinkIds.has(link.id);
    const ambientRelationMarkerCue = !hasFocus && !searchActive && relationMarkerLinkIds.has(link.id);
    const selectedNeighborhoodLink = Boolean(
      selectedNode &&
      active &&
      visualActiveEdgeIds.size > 0 &&
      visualActiveEdgeIds.size <= 12 &&
      !selectedEdge &&
      !searchActive
    );
    if (!active) return hoverOnlyFocusActive ? lerpGraphValue(0, 0.32, hoverFocusWeight) : 0;
    if (ambientDirectionCue) return renderProfile.tier === 'compact' ? 1.85 : 1.42;
    if (ambientRelationMarkerCue) return renderProfile.tier === 'compact' ? 1.58 : 1.22;
    if (!emphasized && visualActiveEdgeIds.size > 10) return 0;
    if (searchActive && !emphasized && renderProfile.tier !== 'compact') return 0;
    if (!emphasized && visualActiveEdgeIds.size === 0 && visualActiveNodeIds.size === 0) {
      return 0;
    }
    if (selectedNeighborhoodLink) return lerpGraphValue(1.2, 3.1, routeFocusVisualActive ? routeFocusWeight : 1);
    return directlyFocused ? 4.1 : searchFocused ? 2.8 : hasFocus ? 1.7 : 1.8;
  }, [ambientDirectionCueLinkIds, hoverFocusVisualActive, hoverFocusWeight, hoveredLink, isLinkActive, linkMatchesSearch, relationMarkerLinkIds, renderProfile.tier, routeFocusVisualActive, routeFocusWeight, searchTerm, selectedEdge, selectedNode, visualActiveEdgeIds.size, visualActiveNodeIds.size]);

  const linkArrowRelPos = useCallback((link: RenderGraphEdge) => {
    if (ambientDirectionCueLinkIds.has(link.id)) return 0.76;
    if (relationMarkerLinkIds.has(link.id)) return 0.72;
    if (!selectedNode || selectedEdge || visualActiveEdgeIds.size === 0 || visualActiveEdgeIds.size > 12 || searchTerm.trim()) return 1;
    if (!isLinkActive(link)) return 1;
    const sourceId = getLinkEndpointId(link.source);
    const targetId = getLinkEndpointId(link.target);
    if (targetId === selectedNode) return 0.78;
    if (sourceId === selectedNode) return 0.84;
    return 0.82;
  }, [ambientDirectionCueLinkIds, getLinkEndpointId, isLinkActive, relationMarkerLinkIds, searchTerm, selectedEdge, selectedNode, visualActiveEdgeIds.size]);

  const getFallbackHoverPosition = (width: number, height: number) => ({
    left: cursorPosition.x > dimensions.width / 2 ? 14 : Math.max(14, dimensions.width - width - 14),
    top: cursorPosition.y > dimensions.height / 2 ? 14 : Math.max(14, dimensions.height - height - 14),
    transformOrigin: cursorPosition.x > dimensions.width / 2 ? 'left center' : 'right center',
  });

  const getHoverScreenAnchor = (kind: 'node' | 'relation') => {
    const graph = fgRef.current;
    const camera = graph?.camera() as THREE.PerspectiveCamera | undefined;
    if (!camera || dimensions.width <= 0 || dimensions.height <= 0) return null;

    if (kind === 'node' && hoveredNodeDetails) {
      const graphNode = graphNodeById.get(hoveredNodeDetails.id);
      return graphNode ? projectGraphPointToScreen(graphNode, camera, dimensions.width, dimensions.height) : null;
    }

    if (kind === 'relation' && hoveredLinkDetails) {
      const source = graphNodeById.get(hoveredLinkDetails.sourceId);
      const target = graphNodeById.get(hoveredLinkDetails.targetId);
      if (!source || !target) return null;
      return projectGraphPointToScreen({
        x: (Number(source.x) + Number(target.x)) / 2,
        y: (Number(source.y) + Number(target.y)) / 2,
        z: (Number(source.z) + Number(target.z)) / 2,
      }, camera, dimensions.width, dimensions.height);
    }

    return null;
  };

  const getAnchoredHoverPosition = (kind: 'node' | 'relation', width: number, height: number) => {
    const anchor = getHoverScreenAnchor(kind);
    if (!anchor) return getFallbackHoverPosition(width, height);

    const margin = 12;
    const gap = kind === 'node' ? 92 : 28;
    const focusClearanceRect: GraphScreenRect | null = kind === 'node'
      ? {
        left: anchor.x - 172,
        top: anchor.y - 146,
        right: anchor.x + 172,
        bottom: anchor.y + 164,
      }
      : null;
    const reservedRects: GraphScreenRect[] = [
      { left: 0, top: 0, right: Math.min(dimensions.width, dimensions.width < 700 ? dimensions.width : 370), bottom: 82 },
      { left: Math.max(0, dimensions.width - 78), top: 0, right: dimensions.width, bottom: 118 },
      { left: 0, top: Math.max(0, dimensions.height - 78), right: 82, bottom: dimensions.height },
      { left: Math.max(0, dimensions.width - 82), top: Math.max(0, dimensions.height - 148), right: dimensions.width, bottom: dimensions.height },
      ...overlayReservedRects,
    ];
    const candidates = [
      {
        side: 'right',
        left: anchor.x + gap,
        top: anchor.y - height * 0.5,
        transformOrigin: 'left center',
        preference: kind === 'node' ? (anchor.x < dimensions.width * 0.68 ? 8 : 22) : (anchor.x < dimensions.width * 0.68 ? 0 : 20),
      },
      {
        side: 'left',
        left: anchor.x - width - gap,
        top: anchor.y - height * 0.5,
        transformOrigin: 'right center',
        preference: kind === 'node' ? (anchor.x > dimensions.width * 0.32 ? 8 : 22) : (anchor.x > dimensions.width * 0.32 ? 0 : 20),
      },
      {
        side: 'below',
        left: anchor.x - width * 0.5,
        top: anchor.y + gap,
        transformOrigin: 'top center',
        preference: kind === 'node' ? (anchor.y < dimensions.height * 0.68 ? 0 : 20) : (anchor.y < dimensions.height * 0.68 ? 8 : 24),
      },
      {
        side: 'above',
        left: anchor.x - width * 0.5,
        top: anchor.y - height - gap,
        transformOrigin: 'bottom center',
        preference: kind === 'node' ? (anchor.y > dimensions.height * 0.32 ? 0 : 20) : (anchor.y > dimensions.height * 0.32 ? 8 : 24),
      },
      ...(kind === 'node'
        ? [
          {
            side: 'above-right',
            left: anchor.x + 78,
            top: anchor.y - height - 82,
            transformOrigin: 'bottom left',
            preference: 6,
          },
          {
            side: 'above-left',
            left: anchor.x - width - 78,
            top: anchor.y - height - 82,
            transformOrigin: 'bottom right',
            preference: 6,
          },
          {
            side: 'below-right',
            left: anchor.x + 78,
            top: anchor.y + 82,
            transformOrigin: 'top left',
            preference: 14,
          },
          {
            side: 'below-left',
            left: anchor.x - width - 78,
            top: anchor.y + 82,
            transformOrigin: 'top right',
            preference: 14,
          },
        ]
        : []),
    ];

    const scored = candidates.map((candidate) => {
      const left = clampGraphValue(candidate.left, margin, Math.max(margin, dimensions.width - width - margin));
      const top = clampGraphValue(candidate.top, margin, Math.max(margin, dimensions.height - height - margin));
      const rect = { left, top, right: left + width, bottom: top + height };
      const clampPenalty = Math.abs(left - candidate.left) + Math.abs(top - candidate.top);
      const overlapPenalty = reservedRects.reduce((total, reserved) => total + getRectOverlapArea(rect, reserved), 0);
      const focusClearancePenalty = focusClearanceRect ? getRectOverlapArea(rect, focusClearanceRect) : 0;
      const anchorDistancePenalty = Math.hypot((left + width * 0.5) - anchor.x, (top + height * 0.5) - anchor.y) * 0.08;
      return {
        ...candidate,
        left,
        top,
        score: candidate.preference +
          clampPenalty * 3 +
          overlapPenalty * 0.55 +
          focusClearancePenalty * 1.4 +
          anchorDistancePenalty,
      };
    }).sort((a, b) => a.score - b.score);

    const placement = scored[0] || getFallbackHoverPosition(width, height);
    return {
      left: placement.left,
      top: placement.top,
      transformOrigin: placement.transformOrigin,
    };
  };

  const maxHoverCardWidth = Math.max(248, dimensions.width - 24);
  const nodeHoverCardWidth = Math.round(Math.min(maxHoverCardWidth, clampGraphValue(dimensions.width * 0.26, 320, 480)));
  const relationHoverCardWidth = Math.round(Math.min(maxHoverCardWidth, clampGraphValue(dimensions.width * 0.27, 340, 520)));
  const nodeHoverCardHeightEstimate = hoveredNodeRelationPreview?.strongest
    ? hoveredNodeDetails?.description ? 174 : 138
    : hoveredNodeDetails?.description ? 116 : 92;
  const relationHoverCardHeightEstimate = hoveredLinkDetails?.evidenceSnippets.length ? 232 : 182;
  const nodeHoverCardPosition = getAnchoredHoverPosition('node', nodeHoverCardWidth, nodeHoverCardHeightEstimate);
  const relationHoverCardPosition = getAnchoredHoverPosition('relation', relationHoverCardWidth, relationHoverCardHeightEstimate);
  type HoverCardPosition = ReturnType<typeof getAnchoredHoverPosition>;
  type HoverCardSnapshot =
    | {
      kind: 'node';
      node: RenderGraphNode;
      relationPreview: HoveredNodeRelationPreview | null;
      position: HoverCardPosition;
    }
    | {
      kind: 'relation';
      relation: NonNullable<typeof hoveredLinkDetails>;
      position: HoverCardPosition;
    };
  const [hoverCardSnapshot, setHoverCardSnapshot] = useState<HoverCardSnapshot | null>(null);
  const [hoverCardVisible, setHoverCardVisible] = useState(false);

  useEffect(() => {
    if (hoverCardExitTimerRef.current !== null) {
      window.clearTimeout(hoverCardExitTimerRef.current);
      hoverCardExitTimerRef.current = null;
    }

    if (!hoverCardsSuppressed && hoveredNodeDetails) {
      setHoverCardSnapshot({
        kind: 'node',
        node: hoveredNodeDetails,
        relationPreview: hoveredNodeRelationPreview,
        position: {
          left: nodeHoverCardPosition.left,
          top: nodeHoverCardPosition.top,
          transformOrigin: nodeHoverCardPosition.transformOrigin,
        },
      });
      setHoverCardVisible(true);
      return undefined;
    }

    if (!hoverCardsSuppressed && hoveredLinkDetails && !hoveredNodeDetails) {
      setHoverCardSnapshot({
        kind: 'relation',
        relation: hoveredLinkDetails,
        position: {
          left: relationHoverCardPosition.left,
          top: relationHoverCardPosition.top,
          transformOrigin: relationHoverCardPosition.transformOrigin,
        },
      });
      setHoverCardVisible(true);
      return undefined;
    }

    const hoverCardExitDelay = hoverCardsSuppressed
      ? GRAPH_HOVER_CARD_CONTEXT_CLEAR_MS
      : GRAPH_HOVER_CARD_EXIT_MS;

    setHoverCardVisible(false);
    hoverCardExitTimerRef.current = window.setTimeout(() => {
      hoverCardExitTimerRef.current = null;
      setHoverCardSnapshot(null);
    }, hoverCardExitDelay);
    return undefined;
  }, [
    hoverCardsSuppressed,
    hoveredLinkDetails,
    hoveredNodeDetails,
    hoveredNodeRelationPreview,
    nodeHoverCardPosition.left,
    nodeHoverCardPosition.top,
    nodeHoverCardPosition.transformOrigin,
    relationHoverCardPosition.left,
    relationHoverCardPosition.top,
    relationHoverCardPosition.transformOrigin,
  ]);

  useEffect(() => () => {
    if (hoverCardExitTimerRef.current !== null) {
      window.clearTimeout(hoverCardExitTimerRef.current);
    }
  }, []);

  const displayedHoverNodeDetails = hoverCardSnapshot?.kind === 'node' ? hoverCardSnapshot.node : null;
  const displayedHoverNodeRelationPreview = hoverCardSnapshot?.kind === 'node' ? hoverCardSnapshot.relationPreview : null;
  const displayedHoverLinkDetails = hoverCardSnapshot?.kind === 'relation' ? hoverCardSnapshot.relation : null;
  const renderHoverFactStrip = (
    facts: Array<{
      color: string;
      id: string;
      label: string;
      value: string;
    }>
  ) => {
    const visibleFacts = facts.filter((fact) => fact.value).slice(0, 4);
    if (visibleFacts.length === 0) return null;

    return (
      <Box
        data-kg-hover-facts="true"
        sx={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 0.36,
          minWidth: 0,
          mb: 0.58,
        }}
      >
        {visibleFacts.map((fact) => (
          <Chip
            key={fact.id}
            size="small"
            label={`${fact.label} ${fact.value}`}
            sx={{
              height: 18,
              maxWidth: '100%',
              color: fact.color,
              bgcolor: alpha(fact.color, 0.095),
              border: '1px solid',
              borderColor: alpha(fact.color, 0.24),
              '& .MuiChip-label': {
                px: 0.52,
                fontSize: '0.56rem',
                fontWeight: 880,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              },
            }}
          />
        ))}
      </Box>
    );
  };
  const renderHoverRelationTypeStrip = (
    relationTypes: Array<{ color?: string; count: number; type: string }>,
    fallbackColor: string
  ) => {
    const visibleTypes = relationTypes
      .filter((item) => item.type && item.count > 0)
      .slice(0, 4);
    if (visibleTypes.length === 0) return null;

    const maxCount = Math.max(1, ...visibleTypes.map((item) => item.count));
    return (
      <Box
        data-kg-hover-relation-types="true"
        sx={{
          display: 'grid',
          gridTemplateColumns: visibleTypes.length > 2 ? 'repeat(2, minmax(0, 1fr))' : '1fr',
          gap: 0.38,
          mb: 0.58,
        }}
      >
        {visibleTypes.map((item) => {
          const color = item.color || getRelationshipColor(item.type) || fallbackColor;
          return (
            <Box
              key={`hover-relation-type-${item.type}`}
              sx={{
                minWidth: 0,
                height: 25,
                px: 0.5,
                display: 'grid',
                gridTemplateColumns: '15px minmax(0, 1fr) auto',
                alignItems: 'center',
                gap: 0.35,
                borderRadius: '7px',
                color,
                bgcolor: alpha(color, 0.07),
                border: '1px solid',
                borderColor: alpha(color, 0.22),
                position: 'relative',
                overflow: 'hidden',
              }}
            >
              <Box
                aria-hidden="true"
                sx={{
                  position: 'absolute',
                  left: 0,
                  right: `${100 - Math.max(10, Math.round((item.count / maxCount) * 100))}%`,
                  bottom: 0,
                  height: 2,
                  bgcolor: alpha(color, 0.76),
                }}
              />
              {renderRelationSymbolMark(item.type, color, { selected: true, size: 18 })}
              <Typography variant="caption" noWrap sx={{ minWidth: 0, fontSize: '0.61rem', fontWeight: 900, lineHeight: 1 }}>
                {getTypeLabel(item.type)}
              </Typography>
              <Typography variant="caption" sx={{ fontSize: '0.58rem', fontWeight: 950, lineHeight: 1 }}>
                {formatNumber(item.count)}
              </Typography>
            </Box>
          );
        })}
      </Box>
    );
  };
    const viewportControlsVisible = !(dimensions.width < 560 && Boolean(selectedNode || selectedEdge));
    const effectiveVisibleLabelIds = useMemo(() => {
      const ids = new Set(screenReadableLabelLayouts.keys());
      activeIdentityNodeIds.forEach((nodeId) => ids.add(nodeId));
      const query = searchTerm.trim();
    if (showLabels && query && detailMode !== 'performance') {
      const normalizedQuery = query.toLowerCase();
      const searchLabelLimit = dimensions.width < 700
        ? labelDensity === 'rich' ? 7 : 5
        : labelDensity === 'rich' ? 12 : 8;
      graphData.nodes
        .filter((node) => getEntitySearchText(node).includes(normalizedQuery))
        .sort((a, b) =>
          Number(visualActiveNodeIds.has(b.id)) - Number(visualActiveNodeIds.has(a.id)) ||
          getNodeImportanceScore(b, nodeRelationSummary.get(b.id)) -
            getNodeImportanceScore(a, nodeRelationSummary.get(a.id)) ||
          a.label.localeCompare(b.label)
        )
        .slice(0, searchLabelLimit)
        .forEach((node) => ids.add(node.id));
    }
    if (
      showLabels &&
      hoverFocusVisualActive &&
      !query &&
      detailMode !== 'performance' &&
      visualActiveNodeIds.size > 0
    ) {
      const focusedNodeLimit = renderProfile.tier === 'massive'
        ? 44
        : renderProfile.tier === 'large'
          ? 64
          : renderProfile.tier === 'standard'
            ? 96
            : 140;
      const focusedNeighborhoodBaseLimit = renderProfile.tier === 'massive'
        ? 8
        : renderProfile.tier === 'large'
          ? 11
          : renderProfile.tier === 'standard'
            ? 14
            : 18;
      const labelLimit = visualActiveNodeIds.size <= focusedNodeLimit
        ? Math.min(
          visualActiveNodeIds.size,
          dimensions.width < 700
            ? Math.max(5, Math.round(focusedNeighborhoodBaseLimit * 0.78))
            : focusedNeighborhoodBaseLimit
        )
        : 0;

      if (labelLimit > 0) {
        graphData.nodes
          .filter((node) =>
            visualActiveNodeIds.has(node.id) &&
            (
              node.id === hoveredNode ||
              node.id === selectedNode ||
              selectedEdgeEndpointIds.has(node.id) ||
              getSelectionContextLabelQualityScore(node, nodeRelationSummary.get(node.id)) >= 120
            )
          )
          .sort((a, b) =>
            Number(b.id === hoveredNode) - Number(a.id === hoveredNode) ||
            getNodeImportanceScore(b, nodeRelationSummary.get(b.id)) -
              getNodeImportanceScore(a, nodeRelationSummary.get(a.id)) ||
            a.label.localeCompare(b.label)
          )
          .slice(0, labelLimit)
          .forEach((node) => ids.add(node.id));
      }
      }

      if (
      showLabels &&
      !query &&
      detailMode !== 'performance' &&
      Boolean(selectedNode || selectedEdge) &&
      visualActiveNodeIds.size > 0
    ) {
      const labelLimit = getGraphFocusedNameplateLimit(renderProfile.tier, dimensions.width);
      if (visualActiveNodeIds.size <= labelLimit) {
        graphData.nodes
          .filter((node) =>
            visualActiveNodeIds.has(node.id) &&
            (
              node.id === selectedNode ||
              selectedEdgeEndpointIds.has(node.id) ||
              getSelectionContextLabelQualityScore(node, nodeRelationSummary.get(node.id)) >= 120
            )
          )
          .sort((a, b) =>
            Number(b.id === selectedNode) - Number(a.id === selectedNode) ||
            getNodeImportanceScore(b, nodeRelationSummary.get(b.id)) -
              getNodeImportanceScore(a, nodeRelationSummary.get(a.id)) ||
            a.label.localeCompare(b.label)
          )
          .forEach((node) => ids.add(node.id));
      }
      }

      if (selectedNode) ids.add(selectedNode);
    if (showLabels && hoveredNode) ids.add(hoveredNode);
    selectedEdgeEndpointIds.forEach((nodeId) => ids.add(nodeId));
    return Array.from(ids);
    }, [
      activeIdentityNodeIds,
      detailMode,
    dimensions.width,
    graphData.nodes,
    hoverFocusVisualActive,
    hoveredNode,
    labelDensity,
    nodeRelationSummary,
    renderProfile.tier,
    screenReadableLabelLayouts,
    searchTerm,
    selectedEdge,
    selectedEdgeEndpointIds,
    selectedNode,
    showLabels,
    visualActiveNodeIds,
  ]);

  return (
      <Box
        ref={containerRef}
        data-kg-graph-canvas-root="true"
        data-kg-path-focus-version={pathFocusVersion}
        data-kg-path-focus-count={pathFocusNodeIds.length}
        data-kg-active-nodes={activeNodeIds.size}
        data-kg-active-edges={activeEdgeIds.size}
        data-kg-focus-context-nodes={focusContextNodeIds.size}
        data-kg-label-density={showLabels ? labelDensity : 'hidden'}
        data-kg-visible-labels={effectiveVisibleLabelIds.length}
        data-kg-visible-label-ids={effectiveVisibleLabelIds.join(',')}
        data-kg-layout-labels={screenReadableLabelLayouts.size}
        data-kg-camera-moving={cameraLod.moving ? 'true' : 'false'}
        data-kg-camera-distance={Math.round(cameraLod.distance)}
        data-kg-initial-viewport-ready={graphInitialViewportReady ? 'true' : 'false'}
        data-kg-camera-target={[
          Math.round(cameraLod.target.x),
          Math.round(cameraLod.target.y),
          Math.round(cameraLod.target.z),
        ].join(',')}
        data-kg-view-signature={viewSignature}
        data-kg-hover-focus-weight={hoverFocusWeight.toFixed(2)}
        data-kg-hover-retaining-context={hoverFocusRetainingContext ? 'true' : 'false'}
        data-kg-route-focus-weight={routeFocusWeight.toFixed(2)}
        data-kg-route-retaining-context={routeFocusRetainingContext ? 'true' : 'false'}
        data-kg-route-visual-active={routeFocusVisualActive ? 'true' : 'false'}
        data-kg-selected-hover-peek={selectedContextHoverNodeActive ? 'true' : 'false'}
        data-kg-selected-hover-route={selectedHoverRouteActive ? 'true' : 'false'}
        data-kg-hover-cards-suppressed={hoverCardsSuppressed ? 'true' : 'false'}
        data-kg-visual-active-nodes={visualActiveNodeIds.size}
        data-kg-visual-active-edges={visualActiveEdgeIds.size}
      onWheelCapture={(event) => {
        if (!isGraphOverlayEventTarget(event.target)) markManualCameraInteraction();
      }}
      onWheel={(event) => {
        if (!isGraphOverlayEventTarget(event.target)) markManualCameraInteraction();
      }}
      onMouseMove={(event) => {
        pointerInsideGraphRef.current = true;
        if (event.buttons && !isGraphOverlayEventTarget(event.target)) markManualCameraInteraction();
        if (!trackHoverCursor) return;
        const bounds = event.currentTarget.getBoundingClientRect();
        updateCursorPosition(event.clientX - bounds.left, event.clientY - bounds.top);
      }}
      onMouseLeave={() => {
        pointerInsideGraphRef.current = false;
        pendingCursorPositionRef.current = null;
        scheduleGraphHover({ kind: 'none' });
      }}
      sx={{
        position: 'relative',
        width: '100%',
        height: '100%',
        minHeight: isFullscreen ? 0 : 360,
        maxHeight: 'none',
        background: '#070a12',
        borderRadius: '8px',
        overflow: 'hidden',
      }}
    >
      {!webglSupported ? (
        <Box
          sx={{
            height: '100%',
            display: 'grid',
            placeItems: 'center',
            px: 3,
            textAlign: 'center',
          }}
        >
          <Box sx={{ maxWidth: 420 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 900, mb: 0.75 }}>
              WebGL is unavailable
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Enable hardware acceleration or use a WebGL-capable browser to view the 3D graph.
            </Typography>
          </Box>
        </Box>
      ) : graphRuntimeError ? (
        <Box
          data-kg-runtime-error="true"
          sx={{
            height: '100%',
            display: 'grid',
            placeItems: 'center',
            px: 3,
            textAlign: 'center',
          }}
        >
          <Box sx={{ maxWidth: 420 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 900, mb: 0.75 }}>
              3D graph failed to load
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {graphRuntimeError}
            </Typography>
          </Box>
        </Box>
      ) : !ForceGraph3DRuntime ? (
        <Box
          data-kg-runtime-loading="true"
          sx={{
            height: '100%',
            display: 'grid',
            placeItems: 'center',
            color: 'text.secondary',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CircularProgress size={18} sx={{ color: accentColors.primary }} />
            <Typography variant="caption" sx={{ fontWeight: 850 }}>
              Loading 3D graph
            </Typography>
          </Box>
        </Box>
      ) : (
        <>
      <Box
        data-kg-graph-viewport-state={graphInitialViewportReady ? 'ready' : 'preparing'}
        sx={{
          width: '100%',
          height: '100%',
          opacity: graphInitialViewportReady ? 1 : 0,
          pointerEvents: graphInitialViewportReady ? 'auto' : 'none',
          transition: 'opacity 420ms cubic-bezier(0.2, 0.8, 0.2, 1)',
        }}
      >
      <ForceGraph3DRuntime
        ref={fgRef}
        graphData={graphData}
        width={dimensions.width}
        height={dimensions.height}
        backgroundColor="rgba(0,0,0,0)"
        showNavInfo={false}
        numDimensions={3}
        nodeRelSize={renderProfile.nodeRelSize}
        nodeVal="val"
        nodeVisibility={shouldRenderNode}
        nodeColor={(node: RenderGraphNode) => getTypeColors(node.type, node.label).main}
        nodeResolution={renderProfile.nodeResolution}
        nodeThreeObject={nodeThreeObject}
        nodeThreeObjectExtend={false}
        nodeLabel=""
        linkLabel=""
        linkColor={linkColor}
        linkVisibility={shouldRenderLink}
        linkWidth={linkWidth}
        linkOpacity={0.82}
        linkResolution={renderProfile.linkResolution}
          linkCurvature={linkCurvature}
          linkCurveRotation={linkCurveRotation}
          linkThreeObject={linkThreeObject}
          linkThreeObjectExtend={true}
          linkPositionUpdate={linkPositionUpdate}
          linkDirectionalArrowLength={linkArrowLength}
        linkDirectionalArrowColor={(link: RenderGraphEdge) => (
          alpha(getRelationshipColor(link.type || link.label), ambientDirectionCueLinkIds.has(link.id) ? 0.74 : 0.98)
        )}
        linkDirectionalArrowRelPos={linkArrowRelPos}
        linkDirectionalArrowResolution={renderProfile.arrowResolution}
        onNodeClick={(node: RenderGraphNode, event?: GraphPointerEvent) => {
          onNodeClick(node);
          if (event?.detail && event.detail > 1) {
            diveIntoNodeInGraph(node.id);
            onNodeDive?.(node);
          }
        }}
        onNodeRightClick={(node: RenderGraphNode, event?: GraphPointerEvent) => {
          event?.preventDefault();
          onNodeClick(node);
          diveIntoNodeInGraph(node.id, 620);
          onNodeDive?.(node);
        }}
        onLinkClick={(link: RenderGraphEdge, event?: GraphPointerEvent) => {
          onLinkClick(link as GraphEdge);
          emitRelationTrace(link);
          if (event?.detail && event.detail > 1) {
            diveIntoEdgeInGraph(link);
            onLinkDive?.(link as GraphEdge);
          }
        }}
        onLinkRightClick={(link: RenderGraphEdge, event?: GraphPointerEvent) => {
          event?.preventDefault();
          onLinkClick(link as GraphEdge);
          emitRelationTrace(link, 5);
          diveIntoEdgeInGraph(link, 620);
          onLinkDive?.(link as GraphEdge);
        }}
        onBackgroundClick={() => {
          clearHoverIntentTimer();
          commitGraphHover({ kind: 'none' });
          onBackgroundClick();
        }}
        onNodeHover={(node: RenderGraphNode | null) => {
          if (hoverInteractionSuppressed) {
            clearHoverIntentTimer();
            commitGraphHover({ kind: 'none' });
            return;
          }
          scheduleGraphHover(node ? { kind: 'node', node } : { kind: 'none' });
        }}
        onLinkHover={(link: RenderGraphEdge | null) => {
          if (hoverInteractionSuppressed) {
            clearHoverIntentTimer();
            commitGraphHover({ kind: 'none' });
            return;
          }
          scheduleGraphHover(link ? { kind: 'link', link } : { kind: 'none' });
        }}
        linkHoverPrecision={graphLinkHoverPrecision}
        enableNodeDrag={nodeDragEnabled}
        enableNavigationControls={true}
        enablePointerInteraction={pointerInteractionEnabled}
        linkDirectionalParticles={(link: RenderGraphEdge) => {
          const emphasized = hoveredLink?.id === link.id || selectedEdge === link.id;
          const selectedHoverRouteLink =
            routeFocusVisualActive &&
            routeFocusWeight > 0.28 &&
            visualActiveEdgeIds.size <= 8 &&
            isLinkActive(link);
          const focusParticleLimit = renderProfile.tier === 'compact' ? 16 : renderProfile.tier === 'standard' ? 10 : 0;
          const searchActive = Boolean(searchTerm.trim());
          const searchParticleLimit = searchActive ? Math.min(4, focusParticleLimit) : focusParticleLimit;
          const focusedEdgeParticles =
            activeEdgeIds.size > 0 &&
            activeEdgeIds.size <= searchParticleLimit &&
            !cameraLod.moving &&
            isLinkActive(link);
          if (emphasized) return 1;
          if (selectedHoverRouteLink) return renderProfile.tier === 'massive' ? 1 : 2;
          if (focusedEdgeParticles && particleFlow) return 1;
          if (cameraLod.moving && renderProfile.tier !== 'compact' && !emphasized) return 0;
          if (particleFlow) {
            if (!isLinkActive(link)) return 0;
            return renderProfile.tier === 'compact' ? 2 : emphasized ? 1 : 0;
          }
          return 0;
        }}
        linkDirectionalParticleSpeed={0.006}
        linkDirectionalParticleWidth={(link: RenderGraphEdge) => {
          const selectedHoverRouteLink =
            routeFocusVisualActive &&
            routeFocusWeight > 0.28 &&
            visualActiveEdgeIds.size <= 8 &&
            isLinkActive(link);
          if (hoveredLink?.id === link.id || selectedEdge === link.id) return 2.3;
          if (selectedHoverRouteLink) return lerpGraphValue(1.45, 2.05, routeFocusWeight);
          return activeEdgeIds.size > 0 ? 1.5 : 1.6;
        }}
          linkDirectionalParticleColor={(link: RenderGraphEdge) => {
            return getRelationshipColor(link.type || link.label);
          }}
          linkDirectionalParticleResolution={renderProfile.tier === 'compact' ? 8 : 4}
          d3AlphaDecay={renderProfile.tier === 'compact' ? 0.02 : renderProfile.tier === 'standard' ? 0.075 : 0.06}
          d3VelocityDecay={renderProfile.tier === 'compact' ? 0.34 : renderProfile.tier === 'standard' ? 0.62 : 0.48}
          warmupTicks={renderProfile.warmupTicks}
          cooldownTicks={renderProfile.cooldownTicks}
          onEngineStop={() => {
            setGraphLayoutVersion((version) => version + 1);
            updateCameraLod(false);
            if (!pendingAutoFitRef.current) return;
            if (selectedNode || selectedEdge) {
              const focused = selectedNode
                ? !selectionExpanded && !searchTerm.trim()
                  ? (() => {
                    const fitPoints = graphData.nodes
                      .map((node) => getGraphNodePoint(node))
                      .filter((point): point is THREE.Vector3 => Boolean(point));
                    return fitPoints.length > 0
                      ? focusGraphPoints(fitPoints, 780, getCompactSelectedOverviewFitOptions())
                      : fitGraph(780, { lockDirection: true });
                  })()
                  : selectedNodeHasVisibleEdges || searchTerm.trim()
                  ? focusNodeInGraph(selectedNode, 780, { composeFit: true })
                  : (() => {
                    const fitPoints = graphData.nodes
                      .map((node) => getGraphNodePoint(node))
                      .filter((point): point is THREE.Vector3 => Boolean(point));
                    return fitPoints.length > 0
                      ? focusGraphPoints(fitPoints, 780, getCompactSelectedOverviewFitOptions())
                      : fitGraph(780, { lockDirection: true });
                  })()
                : selectedEdge
                  ? focusEdgeInGraph(selectedEdge, 780, { composeFit: true })
                  : false;
              pendingAutoFitRef.current = !focused;
              if (focused) {
                lastAutoFitAtRef.current = window.performance.now();
                markInitialViewportReady(180);
              }
              return;
            }
            if (window.performance.now() - lastAutoFitAtRef.current < GRAPH_RECENT_AUTOFIT_SKIP_MS) {
              pendingAutoFitRef.current = false;
              return;
            }
            const fitted = fitGraph(780, { lockDirection: true });
            pendingAutoFitRef.current = !fitted;
            if (fitted) {
              lastAutoFitAtRef.current = window.performance.now();
              markInitialViewportReady(180);
            }
          }}
        />
      </Box>
      {!graphInitialViewportReady && (
        <Box
          data-kg-initial-viewport-loading="true"
          sx={{
            position: 'absolute',
            inset: 0,
            display: 'grid',
            placeItems: 'center',
            pointerEvents: 'none',
            zIndex: 2,
          }}
        >
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              px: 1.2,
              py: 0.75,
              borderRadius: '8px',
              bgcolor: alpha('#050814', 0.82),
              border: `1px solid ${alpha('#8ba4ff', 0.2)}`,
              color: 'text.secondary',
            }}
          >
            <CircularProgress size={16} sx={{ color: accentColors.primary }} />
            <Typography variant="caption" sx={{ fontWeight: 850 }}>
              Preparing 3D graph
            </Typography>
          </Box>
        </Box>
      )}
      </>
      )}

      {hoverCardSnapshot?.kind === 'node' && displayedHoverNodeDetails && (() => {
            const hoveredNodeDetails = displayedHoverNodeDetails;
            const hoveredNodeRelationPreview = displayedHoverNodeRelationPreview;
            const nodeHoverCardPosition = hoverCardSnapshot.position;
            const nodeHoverCardActive = hoverCardVisible && hoverCardSnapshot.kind === 'node';
            const nodeHoverCardTransitionMs = nodeHoverCardActive ? GRAPH_HOVER_CARD_ENTER_MS : GRAPH_HOVER_CARD_EXIT_MS;
            const nodeHoverCardEase = nodeHoverCardActive ? 'cubic-bezier(0.2, 0.8, 0.2, 1)' : 'cubic-bezier(0.4, 0, 0.2, 1)';
            const nodeColors = getTypeColors(hoveredNodeDetails.type, hoveredNodeDetails.label);
            const nodeSummary = nodeRelationSummary.get(hoveredNodeDetails.id);
            const nodeProperties = hoveredNodeDetails.properties || {};
            const occurrence = getFiniteMetadataNumber(
              hoveredNodeDetails.occurrence,
              nodeProperties.occurrence,
              nodeProperties.hits,
              nodeProperties.count
            );
            const confidence = getFiniteMetadataNumber(hoveredNodeDetails.confidence, nodeProperties.confidence);
            const domain = getFirstMetadataValue(nodeProperties, ['domain', 'domains', 'namespace', 'collection']);
            const source = getFirstMetadataValue(nodeProperties, ['source', 'sources', 'document', 'documents', 'file', 'path', 'url']);
            const hoverNodeFacts = [
              nodeSummary?.relations
                ? { id: 'relations', label: 'Relations', value: formatNumber(nodeSummary.relations), color: nodeColors.main }
                : null,
              occurrence
                ? { id: 'hits', label: 'Hits', value: formatNumber(Math.round(occurrence)), color: accentColors.success }
                : null,
              confidence
                ? {
                  id: 'confidence',
                  label: 'Confidence',
                  value: confidence <= 1 ? confidence.toFixed(2) : formatNumber(Math.round(confidence)),
                  color: accentColors.slate,
                }
                : null,
              domain
                ? { id: 'domain', label: 'Domain', value: formatMetadataFactValue(domain, 16), color: accentColors.primary }
                : null,
              !domain && source
                ? { id: 'source', label: 'Source', value: formatMetadataFactValue(source, 16), color: accentColors.info }
                : null,
            ].filter((fact): fact is { color: string; id: string; label: string; value: string } => Boolean(fact));
            return (
        <Box
          data-kg-hover-card="node"
          data-kg-hover-card-width={nodeHoverCardWidth}
          sx={{
            position: 'absolute',
            ...nodeHoverCardPosition,
            width: nodeHoverCardWidth,
            maxWidth: 'calc(100% - 24px)',
            p: dimensions.width >= 1180 ? 0.92 : 0.78,
            borderRadius: '8px',
            bgcolor: alpha('#050816', 0.78),
            border: `1px solid ${alpha(nodeColors.main, 0.34)}`,
            boxShadow: `0 12px 28px ${alpha('#000', 0.24)}`,
            backdropFilter: 'blur(10px)',
            pointerEvents: 'none',
            zIndex: 3,
            opacity: nodeHoverCardActive ? 1 : 0,
            transform: nodeHoverCardActive ? 'translate3d(0, 0, 0) scale(1)' : 'translate3d(4px, 4px, 0) scale(0.985)',
            transition: `opacity ${nodeHoverCardTransitionMs}ms ${nodeHoverCardEase}, transform ${nodeHoverCardTransitionMs}ms ${nodeHoverCardEase}`,
            willChange: 'opacity, transform',
          }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, minWidth: 0, mb: hoverNodeFacts.length > 0 || hoveredNodeRelationPreview?.strongest ? 0.58 : 0 }}>
                {renderEntitySymbolMark(hoveredNodeDetails.type, hoveredNodeDetails.label, {
                  selected: true,
                  size: 30,
                })}
                <Box sx={{ minWidth: 0 }}>
                <Typography variant="body2" noWrap sx={{ fontWeight: 900, lineHeight: 1.16 }}>
                  {hoveredNodeDetails.label}
                </Typography>
                <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block', lineHeight: 1.18 }}>
                  {[
                    getEntityDisplayTypeLabel(hoveredNodeDetails.type, hoveredNodeDetails.label),
                    `${nodeRelationSummary.get(hoveredNodeDetails.id)?.links || 0} links`,
                  ].join(' / ')}
                </Typography>
              </Box>
            </Box>

            {renderHoverFactStrip(hoverNodeFacts)}
            {renderHoverRelationTypeStrip(hoveredNodeRelationPreview?.relationTypes || [], nodeColors.main)}

            {hoveredNodeRelationPreview?.strongest && (
              <Box data-kg-hover-relations="node" sx={{ mb: hoveredNodeDetails.description ? 0.58 : 0 }}>
                  <Box
                    data-kg-hover-strongest="node"
                    sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 0.48,
                      minHeight: 24,
                      px: 0.55,
                      py: 0.34,
                      borderRadius: '7px',
                    border: '1px solid',
                    borderColor: alpha(hoveredNodeRelationPreview.strongest.color, 0.2),
                    bgcolor: alpha(hoveredNodeRelationPreview.strongest.color, 0.055),
                  }}
                >
                  {renderRelationSymbolMark(hoveredNodeRelationPreview.strongest.type, hoveredNodeRelationPreview.strongest.color, {
                    selected: true,
                    size: 19,
                  })}
                  <Typography variant="caption" color="text.secondary" noWrap sx={{ flex: 1, minWidth: 0, lineHeight: 1.15 }}>
                    {getTypeLabel(hoveredNodeRelationPreview.strongest.type)} {hoveredNodeRelationPreview.strongest.otherNode?.label || hoveredNodeRelationPreview.strongest.otherId}
                  </Typography>
                  <Chip
                    size="small"
                    label={hoveredNodeRelationPreview.strongest.count}
                    sx={{
                      height: 17,
                      minWidth: 22,
                      color: hoveredNodeRelationPreview.strongest.color,
                      bgcolor: alpha(hoveredNodeRelationPreview.strongest.color, 0.13),
                      '& .MuiChip-label': { px: 0.42, fontSize: '0.54rem', fontWeight: 900 },
                    }}
                    />
                  </Box>
              </Box>
            )}

          {hoveredNodeDetails.description && (
            <Typography
              variant="caption"
                color="text.secondary"
                sx={{
                  display: '-webkit-box',
                  WebkitLineClamp: nodeHoverCardWidth >= 320 ? 2 : 1,
                  WebkitBoxOrient: 'vertical',
                  overflow: 'hidden',
                  lineHeight: 1.28,
                }}
              >
                {hoveredNodeDetails.description}
              </Typography>
            )}
          </Box>
            );
          })()}

      {hoverCardSnapshot?.kind === 'relation' && displayedHoverLinkDetails && (() => {
            const hoveredLinkDetails = displayedHoverLinkDetails;
            const relationHoverCardPosition = hoverCardSnapshot.position;
            const relationHoverCardActive = hoverCardVisible && hoverCardSnapshot.kind === 'relation';
            const primaryRelation = hoveredLinkDetails.relationTypes[0] || {
              count: hoveredLinkDetails.relationCount,
              type: normalizeType(hoveredLinkDetails.label),
            };
            const relationColor = getRelationshipColor(primaryRelation.type || hoveredLinkDetails.label);
            const sourceLabel = hoveredLinkDetails.source?.label || hoveredLinkDetails.sourceId;
            const targetLabel = hoveredLinkDetails.target?.label || hoveredLinkDetails.targetId;
            const sourceColors = getTypeColors(hoveredLinkDetails.source?.type, sourceLabel);
            const targetColors = getTypeColors(hoveredLinkDetails.target?.type, targetLabel);
            const totalRelationCount = Math.max(hoveredLinkDetails.relationCount, Number(primaryRelation.count || 0));
            const relationWeight = Number(hoveredLinkDetails.weight || 0);
            const hoverRelationFacts = [
              relationWeight > 0
                ? { id: 'weight', label: 'Weight', value: formatNumber(Math.round(relationWeight)), color: accentColors.warning }
                : null,
              hoveredLinkDetails.evidenceSnippets.length > 0
                ? { id: 'evidence', label: 'Evidence', value: formatNumber(hoveredLinkDetails.evidenceSnippets.length), color: relationColor }
                : null,
              hoveredLinkDetails.relationTypes.length > 1
                ? { id: 'types', label: 'Types', value: formatNumber(hoveredLinkDetails.relationTypes.length), color: relationColor }
                : null,
              hoveredLinkDetails.domains[0]
                ? { id: 'domain', label: 'Domain', value: truncateGraphLabel(hoveredLinkDetails.domains[0], 16), color: accentColors.primary }
                : null,
            ].filter((fact): fact is { color: string; id: string; label: string; value: string } => Boolean(fact));
            const relationHoverCardTransitionMs = relationHoverCardActive ? GRAPH_HOVER_CARD_ENTER_MS : GRAPH_HOVER_CARD_EXIT_MS;
            const relationHoverCardEase = relationHoverCardActive ? 'cubic-bezier(0.2, 0.8, 0.2, 1)' : 'cubic-bezier(0.4, 0, 0.2, 1)';
            return (
        <Box
          data-kg-hover-card="relation"
          data-kg-hover-card-width={relationHoverCardWidth}
          sx={{
            position: 'absolute',
            ...relationHoverCardPosition,
            width: relationHoverCardWidth,
            maxWidth: 'calc(100% - 24px)',
            maxHeight: Math.max(112, dimensions.height - 24),
            overflow: 'hidden',
            p: dimensions.width >= 1180 ? 0.92 : 0.78,
            borderRadius: '8px',
            bgcolor: alpha('#050816', 0.8),
            border: `1px solid ${alpha(relationColor, 0.34)}`,
            boxShadow: `0 12px 28px ${alpha('#000', 0.26)}`,
            backdropFilter: 'blur(10px)',
            pointerEvents: 'none',
            zIndex: 3,
            opacity: relationHoverCardActive ? 1 : 0,
            transform: relationHoverCardActive ? 'translate3d(0, 0, 0) scale(1)' : 'translate3d(4px, 4px, 0) scale(0.985)',
            transition: `opacity ${relationHoverCardTransitionMs}ms ${relationHoverCardEase}, transform ${relationHoverCardTransitionMs}ms ${relationHoverCardEase}`,
            willChange: 'opacity, transform',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.65, mb: 0.62 }}>
            <Box
              sx={{
                width: 20,
                height: 20,
                display: 'grid',
                placeItems: 'center',
                borderRadius: '7px',
                color: relationColor,
                bgcolor: alpha(relationColor, 0.14),
                border: `1px solid ${alpha(relationColor, 0.28)}`,
                flexShrink: 0,
              }}
            >
              {renderRelationSymbolMark(primaryRelation.type, relationColor, { selected: true, size: 18 })}
            </Box>
            <Box sx={{ minWidth: 0, flex: 1 }}>
              <Typography variant="body2" noWrap sx={{ fontWeight: 900, lineHeight: 1.15 }}>
                {getTypeLabel(primaryRelation.type || hoveredLinkDetails.label)}
              </Typography>
              <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block', lineHeight: 1.18 }}>
                {formatNumber(totalRelationCount)} relation{totalRelationCount === 1 ? '' : 's'}
              </Typography>
            </Box>
          </Box>

          {renderHoverFactStrip(hoverRelationFacts)}
          {hoveredLinkDetails.relationTypes.length > 1 && renderHoverRelationTypeStrip(hoveredLinkDetails.relationTypes, relationColor)}

          <Box
            data-kg-hover-relation-route="true"
            sx={{
              display: 'grid',
              gridTemplateColumns: 'minmax(0, 1fr) 24px minmax(0, 1fr)',
              alignItems: 'center',
              gap: 0.45,
            }}
          >
            <Box
              sx={{
                minWidth: 0,
                height: 26,
                px: 0.5,
                display: 'flex',
                alignItems: 'center',
                gap: 0.36,
                borderRadius: '7px',
                border: '1px solid',
                borderColor: alpha(sourceColors.main, 0.22),
                bgcolor: alpha(sourceColors.main, 0.06),
              }}
            >
              {renderEntitySymbolMark(hoveredLinkDetails.source?.type, sourceLabel, {
                selected: true,
                size: 22,
              })}
              <Typography variant="caption" noWrap sx={{ minWidth: 0, fontWeight: 900, lineHeight: 1, fontSize: '0.64rem' }}>
                {sourceLabel}
              </Typography>
            </Box>
            <Box
              sx={{
                width: 24,
                height: 24,
                display: 'grid',
                placeItems: 'center',
                borderRadius: '999px',
                color: relationColor,
                bgcolor: alpha(relationColor, 0.14),
                border: `1px solid ${alpha(relationColor, 0.28)}`,
              }}
            >
              {renderRelationSymbolMark(primaryRelation.type, relationColor, { selected: true, size: 18 })}
            </Box>
            <Box
              sx={{
                minWidth: 0,
                height: 26,
                px: 0.5,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
                gap: 0.36,
                borderRadius: '7px',
                border: '1px solid',
                borderColor: alpha(targetColors.main, 0.22),
                bgcolor: alpha(targetColors.main, 0.06),
              }}
            >
              <Typography variant="caption" noWrap sx={{ minWidth: 0, fontWeight: 900, lineHeight: 1, fontSize: '0.64rem' }}>
                {targetLabel}
              </Typography>
              {renderEntitySymbolMark(hoveredLinkDetails.target?.type, targetLabel, {
                selected: true,
                size: 22,
              })}
            </Box>
          </Box>
        </Box>
            );
          })()}

      {/* Controls overlay */}
      {viewportControlsVisible && (
        <Box
          data-kg-viewport-controls="true"
          sx={{
            position: 'absolute',
            bottom: 14,
            right: 14,
            display: 'flex',
            flexDirection: 'column',
            gap: 0.35,
            background: alpha('#020617', 0.2),
            backdropFilter: 'blur(10px)',
            borderRadius: '8px',
            border: `1px solid ${alpha('#fff', 0.08)}`,
            p: 0.4,
            boxShadow: `0 12px 30px ${alpha('#000', 0.18)}`,
              opacity: 0.18,
              transform: 'translateX(4px) scale(0.97)',
              transition: 'opacity 190ms ease, transform 190ms cubic-bezier(0.2, 0.8, 0.2, 1), background-color 190ms ease, border-color 190ms ease',
              '&:hover, &:focus-within': {
                opacity: 1,
                transform: 'translateX(0) scale(1)',
                background: alpha('#020617', 0.72),
                borderColor: alpha('#fff', 0.13),
              },
          }}
        >
          <Tooltip title="Zoom In">
            <IconButton
              size="small"
              onClick={handleZoomIn}
              sx={{
                width: 29,
                height: 29,
                borderRadius: '8px',
                color: 'rgba(255,255,255,0.88)',
                transition: 'transform 150ms ease, background-color 150ms ease',
                '&:hover': { transform: 'translateY(-1px)', bgcolor: alpha('#fff', 0.1), color: '#fff' },
              }}
            >
              <ZoomInIcon sx={{ fontSize: 18 }} />
            </IconButton>
          </Tooltip>
          <Tooltip title="Zoom Out">
            <IconButton
              size="small"
              onClick={handleZoomOut}
              sx={{
                width: 29,
                height: 29,
                borderRadius: '8px',
                color: 'rgba(255,255,255,0.88)',
                transition: 'transform 150ms ease, background-color 150ms ease',
                '&:hover': { transform: 'translateY(-1px)', bgcolor: alpha('#fff', 0.1), color: '#fff' },
              }}
            >
              <ZoomOutIcon sx={{ fontSize: 18 }} />
            </IconButton>
          </Tooltip>
          <Tooltip title="Fit to View">
            <IconButton
              size="small"
              onClick={handleCenterGraph}
              sx={{
                width: 29,
                height: 29,
                borderRadius: '8px',
                color: 'rgba(255,255,255,0.88)',
                transition: 'transform 150ms ease, background-color 150ms ease',
                '&:hover': { transform: 'translateY(-1px)', bgcolor: alpha('#fff', 0.1), color: '#fff' },
              }}
            >
              <CenterIcon sx={{ fontSize: 18 }} />
            </IconButton>
          </Tooltip>
        </Box>
      )}
    </Box>
  );
};

// Main Page Component
const formatNumber = (value: number): string => {
  if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
  if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
  return value.toString();
};

  const KnowledgeGraphPage = () => {
    const [urlSearchParams, setUrlSearchParams] = useSearchParams();
    const initialGraphSearch = urlSearchParams.get('kgSearch')?.trim() || '';
    const initialGraphEntity = urlSearchParams.get('kgEntity')?.trim() || '';
    const initialGraphRelation = (urlSearchParams.get('kgRelation') || urlSearchParams.get('kgEdge'))?.trim() || '';
    const initialGraphDetails = ['1', 'true', 'yes'].includes((urlSearchParams.get('kgDetails') || '').trim().toLowerCase());
    const initialGraphLens = ['1', 'true', 'yes'].includes((urlSearchParams.get('kgLens') || urlSearchParams.get('kgLegend') || '').trim().toLowerCase());
    const initialGraphSearchExpanded = ['1', 'true', 'yes', 'expanded'].includes((urlSearchParams.get('kgSearchExpanded') || '').trim().toLowerCase());
  const [entities, setEntities] = useState<Entity[]>([]);
  const [graphNodes, setGraphNodes] = useState<GraphNode[]>([]);
  const [graphEdges, setGraphEdges] = useState<GraphEdge[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState(initialGraphSearch);
  const [searchResultIndex, setSearchResultIndex] = useState(0);
  const [searchKindFilter, setSearchKindFilter] = useState<GraphSearchKindFilter>('all');
  const [searchTrayOpen, setSearchTrayOpen] = useState(Boolean(initialGraphSearch));
  const [searchTrayExpanded, setSearchTrayExpanded] = useState(Boolean(initialGraphSearch && initialGraphSearchExpanded));
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [searchPreviewNodeId, setSearchPreviewNodeId] = useState<string | null>(null);
  const [searchPreviewEdgeId, setSearchPreviewEdgeId] = useState<string | null>(null);
  const [selectionAnchor, setSelectionAnchor] = useState<GraphSelectionAnchor | null>(null);
  const [browseHistory, setBrowseHistory] = useState<GraphBrowseEntry[]>([]);
  const [browseHistoryIndex, setBrowseHistoryIndex] = useState(-1);
  const [selectedEvidenceIndex, setSelectedEvidenceIndex] = useState(0);
  const selectionPlacementEdgeRef = useRef<GraphSelectionPlacementEdge | null>(null);
  const selectionPlacementKeyRef = useRef<string | null>(null);
    const searchAutoFocusSignatureRef = useRef<string | null>(null);
    const graphSearchUrlSyncRef = useRef(initialGraphSearch);
    const graphSearchExpandedUrlSyncRef = useRef(initialGraphSearchExpanded);
    const graphSelectionUrlSyncRef = useRef({
      entity: initialGraphEntity,
      relation: initialGraphRelation,
      details: initialGraphDetails,
    });
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [expandedRelationshipBundleId, setExpandedRelationshipBundleId] = useState<string | null>(null);
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [relationships, setRelationships] = useState<Relationship[]>([]);

  // GraphRAG service state
  const [graphRAGStatus, setGraphRAGStatus] = useState<{
    available: boolean;
    status: string;
    message?: string;
    nerModel?: string;
    device?: string;
    gpuName?: string;
  }>({ available: false, status: 'checking' });
  const [graphRAGDomains, setGraphRAGDomains] = useState<string[]>([]);
  const [selectedDomain, setSelectedDomain] = useState<string>('');

  // View settings
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showLabels, setShowLabels] = useState(true);
  const [graphLabelDensity, setGraphLabelDensity] = useState<GraphLabelDensity>('priority');
      const [showEdgeLabels, setShowEdgeLabels] = useState(false);
      const [graphDetailMode, setGraphDetailMode] = useState<GraphDetailMode>('auto');
      const [particleFlow, setParticleFlow] = useState(false);
      const [physicsEnabled, setPhysicsEnabled] = useState(true);
        const [settingsOpen, setSettingsOpen] = useState(false);
        const [settingsSection, setSettingsSection] = useState<GraphSettingsSection | null>(null);
        const [graphScopeOpen, setGraphScopeOpen] = useState(false);
        const [graphOverviewOpen, setGraphOverviewOpen] = useState(false);
        const [graphOverviewHover, setGraphOverviewHover] = useState<GraphOverviewHover | null>(null);
        const [graphCameraLod, setGraphCameraLod] = useState<GraphCameraLod>(INITIAL_GRAPH_CAMERA_LOD);
        const [graphCameraFocusRequest, setGraphCameraFocusRequest] = useState<GraphCameraFocusRequest | null>(null);
        const graphOverviewDragRef = useRef({ active: false, moved: false, lastMoveAt: 0 });
        const [graphToolsOpen, setGraphToolsOpen] = useState(false);
        const [graphLegendOpen, setGraphLegendOpen] = useState(initialGraphLens);
        const [graphLegendEntitiesExpanded, setGraphLegendEntitiesExpanded] = useState(false);
        const [graphLegendRelationsExpanded, setGraphLegendRelationsExpanded] = useState(false);
        const [graphStatsOpen, setGraphStatsOpen] = useState(false);
        const [graphStatsDetailsOpen, setGraphStatsDetailsOpen] = useState(false);
          const [selectionPanelExpanded, setSelectionPanelExpanded] = useState(initialGraphDetails);
        const [selectionActionsOpen, setSelectionActionsOpen] = useState(false);
        const [neighborOrbitExpanded, setNeighborOrbitExpanded] = useState(false);
        const [expandedSelectionMetadataGroups, setExpandedSelectionMetadataGroups] = useState<Record<string, boolean>>({});
          const [inspectorOpen, setInspectorOpen] = useState(false);
        const [pathFocusVersion, setPathFocusVersion] = useState(0);
      const [focusNeighborhood, setFocusNeighborhood] = useState(false);
      const [graphScope, setGraphScope] = useState<GraphScope>('linked');
      const [selectedComponentId, setSelectedComponentId] = useState<string | null>(null);
      const [selectedEntityTypes, setSelectedEntityTypes] = useState<string[]>([]);
    const [selectedRelationshipTypes, setSelectedRelationshipTypes] = useState<string[]>([]);
    const [semanticTypePreview, setSemanticTypePreview] = useState<GraphSemanticTypePreview | null>(null);
    const [tabValue, setTabValue] = useState(0);

  // Dialogs
  const [extractDialogOpen, setExtractDialogOpen] = useState(false);

  // Form state
  const [extractText, setExtractText] = useState('');

  useEffect(() => {
    if (!isFullscreen) return undefined;

    const previousBodyOverflow = document.body.style.overflow;
    const previousBodyOverscroll = document.body.style.overscrollBehavior;
    const appChromeElements = Array.from(document.querySelectorAll<HTMLElement>('[data-app-chrome]'));
    const previousChromeState = appChromeElements.map((element) => ({
      element,
      pointerEvents: element.style.pointerEvents,
      visibility: element.style.visibility,
    }));
    document.body.style.overflow = 'hidden';
    document.body.style.overscrollBehavior = 'none';
    appChromeElements.forEach((element) => {
      element.style.pointerEvents = 'none';
      element.style.visibility = 'hidden';
    });
    setSettingsOpen(false);
    setSettingsSection(null);
    setGraphScopeOpen(false);
    setGraphOverviewOpen(false);
    setGraphToolsOpen(false);
    setGraphLegendOpen(false);
    setGraphStatsOpen(false);
    setGraphStatsDetailsOpen(false);
    setSearchTrayOpen(false);
    setSearchTrayExpanded(false);
    setSelectionActionsOpen(false);

    const handleFullscreenEscape = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest('input, textarea, [contenteditable="true"]')) return;
      if (event.key === 'Escape') setIsFullscreen(false);
    };

    window.addEventListener('keydown', handleFullscreenEscape);
    return () => {
      document.body.style.overflow = previousBodyOverflow;
      document.body.style.overscrollBehavior = previousBodyOverscroll;
      previousChromeState.forEach(({ element, pointerEvents, visibility }) => {
        element.style.pointerEvents = pointerEvents;
        element.style.visibility = visibility;
      });
      window.removeEventListener('keydown', handleFullscreenEscape);
    };
  }, [isFullscreen]);

  useEffect(() => {
    if (!searchTrayOpen) return undefined;

    const frameId = window.requestAnimationFrame(() => {
      searchInputRef.current?.focus();
      searchInputRef.current?.select();
    });

    return () => window.cancelAnimationFrame(frameId);
  }, [searchTrayOpen]);

  const graphSearchUrlValue = urlSearchParams.get('kgSearch')?.trim() || '';
  const graphSearchExpandedUrlValue = ['1', 'true', 'yes', 'expanded'].includes((urlSearchParams.get('kgSearchExpanded') || '').trim().toLowerCase());

  useEffect(() => {
    const normalizedSearchTerm = searchTerm.trim();
    const currentUrlSearch = urlSearchParams.get('kgSearch')?.trim() || '';
    const currentUrlExpanded = ['1', 'true', 'yes', 'expanded'].includes((urlSearchParams.get('kgSearchExpanded') || '').trim().toLowerCase());
    const expandedParamNeedsUpdate = normalizedSearchTerm
      ? currentUrlExpanded !== searchTrayExpanded
      : urlSearchParams.has('kgSearchExpanded');
    if (currentUrlSearch === normalizedSearchTerm && !expandedParamNeedsUpdate) {
      graphSearchUrlSyncRef.current = currentUrlSearch;
      graphSearchExpandedUrlSyncRef.current = Boolean(normalizedSearchTerm && currentUrlExpanded);
      return;
    }

    const nextSearchParams = new URLSearchParams(urlSearchParams);
    if (normalizedSearchTerm) {
      nextSearchParams.set('kgSearch', normalizedSearchTerm);
      if (searchTrayExpanded) {
        nextSearchParams.set('kgSearchExpanded', '1');
      } else {
        nextSearchParams.delete('kgSearchExpanded');
      }
    } else {
      nextSearchParams.delete('kgSearch');
      nextSearchParams.delete('kgSearchExpanded');
    }

    graphSearchUrlSyncRef.current = normalizedSearchTerm;
    graphSearchExpandedUrlSyncRef.current = Boolean(normalizedSearchTerm && searchTrayExpanded);
    setUrlSearchParams(nextSearchParams, { replace: true });
  }, [searchTerm, searchTrayExpanded, setUrlSearchParams, urlSearchParams]);

  useEffect(() => {
    const searchChanged = graphSearchUrlValue !== graphSearchUrlSyncRef.current;
    const expandedChanged = graphSearchExpandedUrlValue !== graphSearchExpandedUrlSyncRef.current;
    if (!searchChanged && !expandedChanged) return;

    graphSearchUrlSyncRef.current = graphSearchUrlValue;
    graphSearchExpandedUrlSyncRef.current = graphSearchExpandedUrlValue;
    if (searchChanged) {
      setSearchTerm(graphSearchUrlValue);
      setSearchResultIndex(0);
      setSearchPreviewNodeId(null);
      setSearchPreviewEdgeId(null);
    }
    if (graphSearchUrlValue) {
      setSearchTrayOpen(true);
      setSearchTrayExpanded(graphSearchExpandedUrlValue);
    } else {
      setSearchTrayExpanded(false);
    }
  }, [graphSearchExpandedUrlValue, graphSearchUrlValue]);

  const handleSelectionAnchorChange = useCallback((nextAnchor: GraphSelectionAnchor | null) => {
    setSelectionAnchor((currentAnchor) => {
      if (!nextAnchor && !currentAnchor) return currentAnchor;
      if (!nextAnchor || !currentAnchor) return nextAnchor;
      const sameSelection = currentAnchor.key === nextAnchor.key && currentAnchor.kind === nextAnchor.kind;
      const sameViewport = currentAnchor.width === nextAnchor.width && currentAnchor.height === nextAnchor.height;
      const samePosition =
        Math.abs(currentAnchor.x - nextAnchor.x) < 0.8 &&
        Math.abs(currentAnchor.y - nextAnchor.y) < 0.8 &&
        Math.abs(currentAnchor.depth - nextAnchor.depth) < 0.01;
      return sameSelection && sameViewport && samePosition ? currentAnchor : nextAnchor;
    });
  }, []);

  const handleGraphCameraLodChange = useCallback((nextCamera: GraphCameraLod) => {
    setGraphCameraLod((currentCamera) => {
      const sameCamera =
        currentCamera.key === nextCamera.key &&
        currentCamera.moving === nextCamera.moving &&
        Math.abs(currentCamera.distance - nextCamera.distance) < 0.8 &&
        Math.abs(currentCamera.target.x - nextCamera.target.x) < 0.8 &&
        Math.abs(currentCamera.target.y - nextCamera.target.y) < 0.8 &&
        Math.abs(currentCamera.target.z - nextCamera.target.z) < 0.8;
      return sameCamera ? currentCamera : nextCamera;
    });
  }, []);

  useEffect(() => {
    if (selectedNodeId || selectedEdgeId) return;
    setSelectionAnchor(null);
  }, [selectedEdgeId, selectedNodeId]);

    useEffect(() => {
      setNeighborOrbitExpanded(false);
    }, [selectedEdgeId, selectedNodeId]);

  // Check GraphRAG service status
  const checkGraphRAGStatus = useCallback(async () => {
    try {
      const healthRes = await api.getGraphRAGHealth();
      const graphService = healthRes.services?.graphrag;
      const graphAvailable = healthRes.graph_available === true || graphService?.available === true;

      if (graphAvailable) {
        setGraphRAGStatus({
          available: true,
          status: healthRes.status || 'healthy',
          message: healthRes.message,
        });
        // Get NER status for more details
        try {
          const nerRes = await api.getGraphRAGNERStatus();
          setGraphRAGStatus(prev => ({
            ...prev,
            nerModel: nerRes.model_info?.model_name,
            device: nerRes.model_info?.device,
            gpuName: nerRes.model_info?.gpu_name,
          }));
        } catch {
          // NER status is optional
        }
        // Get available domains
        try {
          const domainsRes = await api.getGraphRAGDomains();
          setGraphRAGDomains(domainsRes.domains || []);
        } catch {
          // Domains are optional
        }
      } else {
        setGraphRAGStatus({
          available: false,
          status: healthRes.status || 'unavailable',
          message: healthRes.message || 'Set GRAPHRAG_URL to a reachable GraphRAG API to load graph data.',
        });
      }
    } catch {
      setGraphRAGStatus({
        available: false,
        status: 'unavailable',
      });
    }
  }, []);

  // Load data from GraphRAG service
  const loadData = useCallback(async () => {
    if (!graphRAGStatus.available) {
      return;
    }

    setLoading(true);
    setError(null);
    try {
      // Load from GraphRAG service
      const graphRes = await api.getGraphRAGGraph({
        domain: selectedDomain || undefined,
        max_entities: GRAPH_ENTITY_FETCH_LIMIT,
        max_relationships: GRAPH_RELATIONSHIP_FETCH_LIMIT,
      });

      // Transform nodes for visualization
        const nodes: GraphNode[] = ((graphRes.nodes || []) as GraphRAGNodePayload[]).map((n) => {
          const properties = n.properties && typeof n.properties === 'object' ? n.properties : {};
          const description = getUsefulEntityDescription(properties.description);
          return {
            id: n.id,
            label: n.label,
            type: n.type || 'custom',
            description,
            confidence: Number(properties.confidence ?? n.occurrence ?? 1),
            occurrence: Number(n.occurrence ?? properties.occurrence ?? 0),
            properties,
          };
        });

        // Transform edges
        const edges: GraphEdge[] = ((graphRes.edges || []) as GraphRAGEdgePayload[]).map((e) => {
          const properties = e.properties && typeof e.properties === 'object' ? e.properties : {};
          const label = e.label || e.type || 'related';
          const type = e.type || e.label || 'related';
          const relationCount = Number(
            e.relationCount
            || e.relation_count
            || properties.relationCount
            || properties.relation_count
            || 1
          );
          const searchText = [
            e.source,
            e.target,
            label,
            type,
            `${relationCount} relations`,
            typeof properties.domain === 'string' ? properties.domain : '',
            typeof properties.context === 'string' ? properties.context : '',
            formatMetadataValue(properties),
          ].filter(Boolean).join(' ').toLowerCase();

          return {
            id: e.id,
            source: e.source,
            target: e.target,
            label,
            type,
            weight: Number(e.weight || 1),
            relationCount: Math.max(1, Number.isFinite(relationCount) ? relationCount : 1),
            searchText,
            properties,
          };
        });

      setGraphNodes(nodes);
      setGraphEdges(edges);

      // Transform to entities for sidebar
        const entitiesList = nodes.map((n) => ({
          id: n.id,
          name: n.label,
          entity_type: n.type,
          description: n.description || '',
          aliases: Array.isArray(n.properties?.aliases) ? n.properties.aliases.map(String) : [],
          properties: n.properties || {},
          confidence: n.confidence ?? 1,
        }));
      setEntities(entitiesList);

      // Get stats
      const statsRes = await api.getGraphRAGStats();
      setStats({
        entity_count: statsRes.nodes,
        relationship_count: statsRes.edges,
        community_count: statsRes.communities || 0,
        entities_by_type: {},
        relationships_by_type: {},
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to load graph data';
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [graphRAGStatus.available, selectedDomain]);

  // Check GraphRAG status on mount
  useEffect(() => {
    checkGraphRAGStatus();
  }, [checkGraphRAGStatus]);

  // Load data when GraphRAG becomes available or domain changes
  useEffect(() => {
    if (graphRAGStatus.available) {
      loadData();
    }
  }, [loadData, graphRAGStatus.available]);

  // Load relationships for selected entity from graph edges
  useEffect(() => {
    setExpandedRelationshipBundleId(null);
    if (selectedNodeId && graphRAGStatus.available) {
      // Filter edges that involve this node
      const nodeRels = graphEdges
        .filter(e => e.source === selectedNodeId || e.target === selectedNodeId)
        .map(e => ({
          id: e.id,
          source_id: e.source,
          target_id: e.target,
          relationship_type: e.type || e.label,
          description: '',
          weight: e.weight,
        }));
      setRelationships(nodeRels);
    } else {
      setRelationships([]);
    }
  }, [selectedNodeId, graphRAGStatus.available, graphEdges]);

  // Extract entities using GraphRAG GLiNER
  const handleExtract = async () => {
    if (!graphRAGStatus.available) {
      setError('GraphRAG service is not available');
      return;
    }

    try {
      setLoading(true);
      const result = await api.extractGraphRAGEntities({
        text: extractText,
        domain: selectedDomain || 'general',
      });
      // Log extraction results
      debugLog('knowledge-graph', `Extracted ${result.entity_count} entities and ${result.relationship_count} relationships using ${result.extraction_method}`, result);
      setExtractDialogOpen(false);
      setExtractText('');
      // Reload graph data
      loadData();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to extract entities');
    } finally {
      setLoading(false);
    }
  };

  const closeGraphOverlaysForFocus = useCallback(() => {
    setSearchTerm('');
    setSearchTrayOpen(false);
    setSearchTrayExpanded(false);
    setSearchPreviewNodeId(null);
    setSearchPreviewEdgeId(null);
    setSettingsOpen(false);
    setSettingsSection(null);
    setGraphScopeOpen(false);
    setGraphOverviewOpen(false);
    setGraphToolsOpen(false);
    setGraphLegendOpen(false);
    setGraphStatsOpen(false);
    setGraphStatsDetailsOpen(false);
    setInspectorOpen(false);
    setSelectionActionsOpen(false);
    setSelectionPanelExpanded(false);
    setNeighborOrbitExpanded(false);
  }, []);

  const handleNodeClick = (node: GraphNode) => {
    closeGraphOverlaysForFocus();
    setSelectedEdgeId(null);
    setSelectedNodeId(node.id);
    const entity = entities.find(e => e.id === node.id);
    if (entity) setSelectedEntity(entity);
    setTabValue(2);
  };

  const handleLinkClick = (edge: GraphEdge) => {
    closeGraphOverlaysForFocus();
    setSelectedEdgeId(edge.id);
    setSelectedNodeId(null);
    setSelectedEntity(null);
    setExpandedRelationshipBundleId(null);
    setTabValue(2);
  };

  const openGraphEntitySelection = useCallback((
    entity: Entity,
    options: {
      clearSearch?: boolean;
      collapsePanel?: boolean;
      closeInspector?: boolean;
      closeNeighborOrbit?: boolean;
    } = {}
  ) => {
    if (options.clearSearch) {
      setSearchTerm('');
      setSearchPreviewNodeId(null);
      setSearchPreviewEdgeId(null);
    }
    setSelectedEdgeId(null);
    setSelectedEntity(entity);
    setSelectedNodeId(entity.id);
    setExpandedRelationshipBundleId(null);
    setSelectionActionsOpen(false);
    if (options.collapsePanel) setSelectionPanelExpanded(false);
    if (options.closeNeighborOrbit) setNeighborOrbitExpanded(false);
    if (options.closeInspector) setInspectorOpen(false);
    setTabValue(2);
  }, []);

  const openGraphRelationshipDetails = (
    edge: GraphEdge | null | undefined,
    options: { expandPanel?: boolean } = {}
  ) => {
    if (!edge) return;
    handleLinkClick(edge);
    if (options.expandPanel) setSelectionPanelExpanded(true);
    setNeighborOrbitExpanded(false);
    setInspectorOpen(false);
  };

  const handleNodeDive = useCallback((node: GraphNode) => {
    setSelectedComponentId(null);
    setSelectedRelationshipTypes([]);
    setFocusNeighborhood(true);
    setGraphScope('linked');
    setGraphStatsOpen(false);
    setGraphStatsDetailsOpen(false);
    setSelectionPanelExpanded(false);
    setNeighborOrbitExpanded(false);
    setHoveredNodeId(null);
    setSelectedNodeId(node.id);
    const entity = entities.find((item) => item.id === node.id);
    if (entity) setSelectedEntity(entity);
  }, [entities]);

  const handleLinkDive = useCallback((edge?: GraphEdge) => {
    if (edge) {
      setSelectedEdgeId(edge.id);
      setSelectedNodeId(null);
      setSelectedEntity(null);
      setExpandedRelationshipBundleId(null);
      setTabValue(2);
    }
    setSelectedComponentId(null);
    setFocusNeighborhood(false);
    setGraphScope('linked');
    setGraphStatsOpen(false);
    setGraphStatsDetailsOpen(false);
    setSelectionPanelExpanded(false);
    setNeighborOrbitExpanded(false);
  }, []);

  const clearGraphSelection = useCallback(() => {
    setSelectedEdgeId(null);
    setSelectedNodeId(null);
    setSelectedEntity(null);
    setSearchPreviewNodeId(null);
    setSearchPreviewEdgeId(null);
    setExpandedRelationshipBundleId(null);
    setSelectionActionsOpen(false);
  }, []);

  const clearGraphCanvasContext = useCallback(() => {
    clearGraphSelection();
    setInspectorOpen(false);
      setSelectionPanelExpanded(false);
      setNeighborOrbitExpanded(false);
      setSettingsOpen(false);
    setSettingsSection(null);
    setGraphScopeOpen(false);
    setGraphOverviewOpen(false);
    setGraphToolsOpen(false);
    setGraphLegendOpen(false);
    setGraphStatsOpen(false);
    setSearchTrayOpen(false);
    setSearchTrayExpanded(false);
    setSelectionActionsOpen(false);
  }, [clearGraphSelection]);

  useEffect(() => {
    setSelectionActionsOpen(false);
  }, [selectedEdgeId, selectedNodeId]);

  const openGraphSearch = useCallback(() => {
    setSettingsOpen(false);
    setSettingsSection(null);
    setGraphScopeOpen(false);
    setGraphOverviewOpen(false);
    setGraphToolsOpen(false);
    setGraphLegendOpen(false);
    setGraphStatsOpen(false);
    setGraphStatsDetailsOpen(false);
    setInspectorOpen(false);
    setSelectionActionsOpen(false);
    setHoveredNodeId(null);
    setSearchTrayOpen(true);
    window.requestAnimationFrame(() => {
      searchInputRef.current?.focus();
      searchInputRef.current?.select();
    });
  }, []);

  useEffect(() => {
    const isTextEntryTarget = (target: EventTarget | null) => {
      if (!(target instanceof HTMLElement)) return false;
      return Boolean(target.closest('input, textarea, select, [contenteditable="true"], [role="textbox"]'));
    };

    const handleGraphKeyboard = (event: KeyboardEvent) => {
      const textEntryTarget = isTextEntryTarget(event.target);
      if (
        event.key === '/' &&
        !textEntryTarget &&
        !event.metaKey &&
        !event.ctrlKey &&
        !event.altKey
        ) {
          event.preventDefault();
          openGraphSearch();
          return;
        }

      if (event.key !== 'Escape' || textEntryTarget) return;

      const hasTransientContext = Boolean(
        searchTrayOpen ||
        searchTerm ||
        searchTrayExpanded ||
        selectedNodeId ||
        selectedEdgeId ||
        settingsOpen ||
        graphScopeOpen ||
        graphOverviewOpen ||
        graphToolsOpen ||
        graphLegendOpen ||
          graphStatsOpen ||
          graphStatsDetailsOpen ||
          inspectorOpen ||
          selectionPanelExpanded ||
          selectionActionsOpen ||
          neighborOrbitExpanded
        );

      if (hasTransientContext) {
        event.preventDefault();
        event.stopImmediatePropagation();
        setSearchTerm('');
        setSearchTrayOpen(false);
        setSearchTrayExpanded(false);
        clearGraphCanvasContext();
      } else if (isFullscreen) {
        event.preventDefault();
        setIsFullscreen(false);
      }
    };

    window.addEventListener('keydown', handleGraphKeyboard, { capture: true });
    return () => window.removeEventListener('keydown', handleGraphKeyboard, { capture: true });
    }, [
      clearGraphCanvasContext,
      graphLegendOpen,
      graphOverviewOpen,
      graphScopeOpen,
      graphStatsDetailsOpen,
      graphStatsOpen,
      graphToolsOpen,
      inspectorOpen,
      isFullscreen,
      neighborOrbitExpanded,
      openGraphSearch,
      searchTerm,
      selectionActionsOpen,
      searchTrayOpen,
      searchTrayExpanded,
    selectedEdgeId,
    selectedNodeId,
    selectionPanelExpanded,
    settingsOpen,
  ]);

  const openGraphDetails = () => {
    setSettingsOpen(false);
    setSettingsSection(null);
    setGraphScopeOpen(false);
    setGraphOverviewOpen(false);
    setGraphToolsOpen(false);
    setGraphLegendOpen(false);
    setGraphStatsOpen(false);
    setGraphStatsDetailsOpen(false);
    setTabValue(2);
    setInspectorOpen(false);
    setSelectionActionsOpen(false);
    setSelectionPanelExpanded(true);
  };

    const entityTypeSummary = useMemo(() => {
      const counts = new Map<string, number>();
      graphNodes.forEach((node) => {
        const type = normalizeType(node.type);
        counts.set(type, (counts.get(type) || 0) + 1);
      });
      return Array.from(counts.entries())
        .map(([type, count]) => ({ type, count }))
        .sort((a, b) => b.count - a.count || a.type.localeCompare(b.type));
    }, [graphNodes]);

    const relationshipTypeSummary = useMemo(() => {
      const counts = new Map<string, number>();
      graphEdges.forEach((edge) => {
        const type = normalizeType(edge.type || edge.label);
        counts.set(type, (counts.get(type) || 0) + 1);
      });
      return Array.from(counts.entries())
        .map(([type, count]) => ({ type, count, color: getRelationshipColor(type) }))
        .sort((a, b) => b.count - a.count || a.type.localeCompare(b.type));
    }, [graphEdges]);

      const selectedRelationshipTypeSummary = useMemo(() => {
        if (selectedRelationshipTypes.length === 0) return [];
        const summaryByType = new Map(relationshipTypeSummary.map((item) => [item.type, item]));
        return selectedRelationshipTypes.map((type) =>
        summaryByType.get(type) || { type, count: 0, color: getRelationshipColor(type) }
      );
    }, [relationshipTypeSummary, selectedRelationshipTypes]);

      const degreeMap = useMemo(() => {
      const map = new Map<string, { in: number; out: number; total: number }>();
      const ensure = (id: string) => {
        if (!map.has(id)) map.set(id, { in: 0, out: 0, total: 0 });
        return map.get(id)!;
      };
      graphEdges.forEach((edge) => {
        const source = ensure(edge.source);
        source.out += 1;
        source.total += 1;
        const target = ensure(edge.target);
        target.in += 1;
        target.total += 1;
      });
        return map;
      }, [graphEdges]);

        const entityById = useMemo(
          () => new Map(entities.map((entity) => [entity.id, entity])),
          [entities]
        );
        const entityByLookupKey = useMemo(() => {
          const map = new Map<string, Entity>();
          entities.forEach((entity) => {
            map.set(entity.id, entity);
            map.set(entity.name, entity);
          });
          return map;
        }, [entities]);

      const graphEntityUrlValue = urlSearchParams.get('kgEntity')?.trim() || '';
      const graphRelationUrlValue = (urlSearchParams.get('kgRelation') || urlSearchParams.get('kgEdge'))?.trim() || '';
      const graphDetailsUrlValue = ['1', 'true', 'yes'].includes((urlSearchParams.get('kgDetails') || '').trim().toLowerCase());
      const selectedGraphEntityUrlId = selectedEdgeId ? '' : selectedEntity?.id || selectedNodeId || '';
      const selectedGraphRelationUrlId = selectedEdgeId || '';

        const resolveGraphEntityDeepLink = useCallback((value: string) => {
          const normalizedValue = normalizeGraphText(value);
          if (!normalizedValue) return null;
          return entities.find((entity) =>
            entity.id === value ||
          entity.name === value ||
          normalizeGraphText(entity.id) === normalizedValue ||
            normalizeGraphText(entity.name) === normalizedValue
          ) || null;
        }, [entities]);

        const resolveGraphRouteDeepLinkId = useCallback((sourceValue: string, targetValue: string) => {
          const buildEndpointKeys = (rawValue: string) => {
            const value = rawValue.trim();
            const entity = resolveGraphEntityDeepLink(value);
            return new Set(
              [value, entity?.id, entity?.name]
                .filter((item): item is string => Boolean(item))
                .map(normalizeGraphText)
                .filter(Boolean)
            );
          };
          const sourceKeys = buildEndpointKeys(sourceValue);
          const targetKeys = buildEndpointKeys(targetValue);
          if (sourceKeys.size === 0 || targetKeys.size === 0) return '';

          const edgeEndpointMatches = (edge: GraphEdge, sourceAliases: Set<string>, targetAliases: Set<string>) => {
            const edgeSourceName = entityById.get(edge.source)?.name || edge.source;
            const edgeTargetName = entityById.get(edge.target)?.name || edge.target;
            const edgeSourceKeys = [edge.source, edgeSourceName].map(normalizeGraphText).filter(Boolean);
            const edgeTargetKeys = [edge.target, edgeTargetName].map(normalizeGraphText).filter(Boolean);
            return edgeSourceKeys.some((key) => sourceAliases.has(key)) &&
              edgeTargetKeys.some((key) => targetAliases.has(key));
          };

          const directedEdge = graphEdges.find((edge) => edgeEndpointMatches(edge, sourceKeys, targetKeys));
          if (directedEdge) return getVisualEdgeKey(directedEdge);

          const reverseEdge = graphEdges.find((edge) => edgeEndpointMatches(edge, targetKeys, sourceKeys));
          return reverseEdge ? getVisualEdgeKey(reverseEdge) : '';
        }, [entityById, graphEdges, resolveGraphEntityDeepLink]);

        const resolveGraphRelationDeepLink = useCallback((value: string) => {
          const normalizedValue = normalizeGraphText(value);
          if (!normalizedValue) return null;
          return graphEdges.find((edge) => {
            if (edge.id === value || normalizeGraphText(edge.id) === normalizedValue) return true;
          const sourceName = entityById.get(edge.source)?.name || edge.source;
          const targetName = entityById.get(edge.target)?.name || edge.target;
          const routeLabel = `${sourceName} -> ${targetName}`;
          return normalizeGraphText(routeLabel) === normalizedValue ||
            normalizeGraphText(`${routeLabel} ${edge.type || edge.label}`) === normalizedValue;
        }) || null;
      }, [entityById, graphEdges]);

        const resolveGraphRelationDeepLinkId = useCallback((value: string) => {
          const edge = resolveGraphRelationDeepLink(value);
          if (edge) return getVisualEdgeKey(edge);
          const routeParts = value.split('::');
          if (routeParts.length >= 2) {
            const sourceValue = routeParts[0];
            const targetValue = routeParts.slice(1).join('::');
            const routeId = resolveGraphRouteDeepLinkId(sourceValue, targetValue);
            if (routeId) return routeId;
          }
          const arrowRouteMatch = value.match(/^\s*(.+?)\s*(?:--?>|=>|->|→|\bto\b)\s*(.+?)\s*$/i);
          if (arrowRouteMatch) {
            const routeId = resolveGraphRouteDeepLinkId(arrowRouteMatch[1], arrowRouteMatch[2]);
            if (routeId) return routeId;
          }
          return '';
        }, [resolveGraphRelationDeepLink, resolveGraphRouteDeepLinkId]);

      useEffect(() => {
        if (graphRelationUrlValue) {
          if (graphEdges.length === 0) return;
          const relationId = resolveGraphRelationDeepLinkId(graphRelationUrlValue);
          if (!relationId) return;
          graphSelectionUrlSyncRef.current = {
            entity: '',
            relation: graphRelationUrlValue,
            details: graphDetailsUrlValue,
          };
          setSearchPreviewNodeId(null);
          setSearchPreviewEdgeId(null);
          setSelectedEdgeId(relationId);
          setSelectedNodeId(null);
          setSelectedEntity(null);
          setExpandedRelationshipBundleId(null);
          setSelectionActionsOpen(false);
          setNeighborOrbitExpanded(false);
          setInspectorOpen(false);
          setSelectedEntityTypes([]);
          setSelectedRelationshipTypes([]);
          setSelectedComponentId(null);
          setFocusNeighborhood(false);
          setGraphScope('all');
          setTabValue(2);
          setSelectionPanelExpanded(graphDetailsUrlValue);
          return;
        }

        if (graphEntityUrlValue) {
          if (entities.length === 0) return;
          const entity = resolveGraphEntityDeepLink(graphEntityUrlValue);
          if (!entity) return;
          graphSelectionUrlSyncRef.current = {
            entity: graphEntityUrlValue,
            relation: '',
            details: graphDetailsUrlValue,
          };
          setSearchPreviewNodeId(null);
          setSearchPreviewEdgeId(null);
          setSelectedEdgeId(null);
          setSelectedEntity(entity);
          setSelectedNodeId(entity.id);
          setExpandedRelationshipBundleId(null);
          setSelectionActionsOpen(false);
          setNeighborOrbitExpanded(false);
          setInspectorOpen(false);
          setSelectedEntityTypes([]);
          setSelectedRelationshipTypes([]);
          setSelectedComponentId(null);
          setFocusNeighborhood(false);
          setGraphScope('all');
          setTabValue(2);
          setSelectionPanelExpanded(graphDetailsUrlValue);
        }
      }, [
        entities.length,
        graphDetailsUrlValue,
        graphEdges.length,
        graphEntityUrlValue,
        graphRelationUrlValue,
        resolveGraphEntityDeepLink,
        resolveGraphRelationDeepLinkId,
      ]);

      useEffect(() => {
        const pendingEntityDeepLink = Boolean(graphEntityUrlValue && !selectedGraphEntityUrlId && !selectedGraphRelationUrlId && entities.length === 0);
        const pendingRelationDeepLink = Boolean(graphRelationUrlValue && !selectedGraphRelationUrlId && graphEdges.length === 0);
        const hydratingEntityDeepLink = Boolean(
          graphEntityUrlValue &&
          !selectedGraphEntityUrlId &&
          !selectedGraphRelationUrlId &&
          entities.length > 0 &&
          resolveGraphEntityDeepLink(graphEntityUrlValue)
        );
        const hydratingRelationDeepLink = Boolean(
          graphRelationUrlValue &&
          !selectedGraphRelationUrlId &&
          graphEdges.length > 0 &&
          resolveGraphRelationDeepLinkId(graphRelationUrlValue)
        );
        if (pendingEntityDeepLink || pendingRelationDeepLink || hydratingEntityDeepLink || hydratingRelationDeepLink) return;

        const currentUrlEntity = urlSearchParams.get('kgEntity')?.trim() || '';
        const currentUrlRelation = (urlSearchParams.get('kgRelation') || urlSearchParams.get('kgEdge'))?.trim() || '';
        const currentUrlDetails = ['1', 'true', 'yes'].includes((urlSearchParams.get('kgDetails') || '').trim().toLowerCase());
        const nextUrlDetails = Boolean((selectedGraphEntityUrlId || selectedGraphRelationUrlId) && selectionPanelExpanded);

        if (
          currentUrlEntity === selectedGraphEntityUrlId &&
          currentUrlRelation === selectedGraphRelationUrlId &&
          currentUrlDetails === nextUrlDetails
        ) {
          graphSelectionUrlSyncRef.current = {
            entity: currentUrlEntity,
            relation: currentUrlRelation,
            details: currentUrlDetails,
          };
          return;
        }

        const nextSearchParams = new URLSearchParams(urlSearchParams);
        if (selectedGraphRelationUrlId) {
          nextSearchParams.set('kgRelation', selectedGraphRelationUrlId);
          nextSearchParams.delete('kgEdge');
          nextSearchParams.delete('kgEntity');
        } else if (selectedGraphEntityUrlId) {
          nextSearchParams.set('kgEntity', selectedGraphEntityUrlId);
          nextSearchParams.delete('kgRelation');
          nextSearchParams.delete('kgEdge');
        } else {
          nextSearchParams.delete('kgEntity');
          nextSearchParams.delete('kgRelation');
          nextSearchParams.delete('kgEdge');
        }

        if (nextUrlDetails) {
          nextSearchParams.set('kgDetails', '1');
        } else {
          nextSearchParams.delete('kgDetails');
        }

        graphSelectionUrlSyncRef.current = {
          entity: selectedGraphEntityUrlId,
          relation: selectedGraphRelationUrlId,
          details: nextUrlDetails,
        };
        setUrlSearchParams(nextSearchParams, { replace: true });
      }, [
        entities.length,
        graphEdges.length,
        graphEntityUrlValue,
        graphRelationUrlValue,
        resolveGraphEntityDeepLink,
        resolveGraphRelationDeepLinkId,
        selectedGraphEntityUrlId,
        selectedGraphRelationUrlId,
        selectionPanelExpanded,
        setUrlSearchParams,
        urlSearchParams,
      ]);

      const toggleEntityType = useCallback((type: string) => {
      setSemanticTypePreview(null);
      setSelectedEntityTypes((current) =>
        current.includes(type) ? current.filter((item) => item !== type) : [...current, type]
      );
    }, []);

    const toggleRelationshipType = useCallback((type: string) => {
      setSemanticTypePreview(null);
      setSelectedRelationshipTypes((current) =>
        current.includes(type) ? current.filter((item) => item !== type) : [...current, type]
      );
    }, []);

    const focusRelationshipType = useCallback((type: string) => {
      setSemanticTypePreview(null);
      setSearchTerm('');
      setSelectedRelationshipTypes([normalizeType(type)]);
      setFocusNeighborhood(false);
      setSelectedComponentId(null);
      setGraphScope('linked');
      setTabValue(1);
    }, []);

    const toggleRelationshipTypeFocus = useCallback((type: string) => {
      const normalized = normalizeType(type);
      setSemanticTypePreview(null);
      setSearchTerm('');
      setSelectedEdgeId(null);
      setSelectedEntity(null);
      setSelectedNodeId(null);
      setFocusNeighborhood(false);
      setSelectedComponentId(null);
      setGraphScope('linked');
      setSelectedRelationshipTypes((current) =>
        current.length === 1 && current[0] === normalized ? [] : [normalized]
      );
      setTabValue(1);
    }, []);

    const clearGraphFilters = useCallback(() => {
      setSemanticTypePreview(null);
      setSearchTerm('');
      setSelectedEntityTypes([]);
      setSelectedRelationshipTypes([]);
      setFocusNeighborhood(false);
      setSelectedComponentId(null);
    }, []);

    const searchQuery = searchTerm.trim().toLowerCase();
    const searchTrayActive = searchTrayOpen || Boolean(searchQuery) || searchTrayExpanded;
    const focusNodeId = selectedNodeId || (searchQuery ? searchPreviewNodeId || hoveredNodeId : hoveredNodeId);
    const selectedHoverRouteNodeId = selectedEntity &&
      selectedNodeId &&
      hoveredNodeId &&
      hoveredNodeId !== selectedNodeId
      ? hoveredNodeId
      : null;
    const graphFiltersActive = Boolean(
      searchQuery ||
      selectedEntityTypes.length > 0 ||
      selectedRelationshipTypes.length > 0 ||
      Boolean(selectedComponentId) ||
      focusNeighborhood
    );
    const graphTypeLensFilterCount = selectedEntityTypes.length + selectedRelationshipTypes.length;

    useEffect(() => {
      if (graphLegendOpen) return;
      setSemanticTypePreview(null);
    }, [graphLegendOpen]);

    useEffect(() => {
      if (!searchQuery) {
        setSearchPreviewNodeId(null);
        setSearchPreviewEdgeId(null);
        return;
      }

      setSelectedEdgeId(null);
      setSelectedNodeId(null);
      setSelectedEntity(null);
      setExpandedRelationshipBundleId(null);
      setSelectionPanelExpanded(false);
      setNeighborOrbitExpanded(false);
      setInspectorOpen(false);
    }, [searchQuery]);

      const baseFilteredGraph = useMemo(() => {
        let candidateNodes = graphNodes.filter((node) =>
          selectedEntityTypes.length === 0 || selectedEntityTypes.includes(normalizeType(node.type))
        );
        const selectedRelationshipTypeSet = new Set(selectedRelationshipTypes.map((type) => normalizeType(type)));
        let candidateEdges = graphEdges.filter((edge) =>
          selectedRelationshipTypeSet.size === 0 ||
          getLinkRelationTypeCounts(edge, 12).some((item) => selectedRelationshipTypeSet.has(normalizeType(item.type))) ||
          selectedRelationshipTypeSet.has(normalizeType(edge.type || edge.label))
      );

      if (selectedRelationshipTypes.length > 0) {
        const relationNodeIds = new Set<string>();
        candidateEdges.forEach((edge) => {
          relationNodeIds.add(edge.source);
          relationNodeIds.add(edge.target);
        });
        candidateNodes = candidateNodes.filter((node) => relationNodeIds.has(node.id));
      }

      const visibleIds = new Set(candidateNodes.map((node) => node.id));
      candidateEdges = candidateEdges.filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target));

        return { nodes: candidateNodes, edges: candidateEdges, nodeIds: visibleIds };
      }, [graphEdges, graphNodes, selectedEntityTypes, selectedRelationshipTypes]);

      const componentSummaries = useMemo(
        () => buildGraphComponentSummaries(baseFilteredGraph.nodes, baseFilteredGraph.edges),
        [baseFilteredGraph.edges, baseFilteredGraph.nodes]
      );

      const browsableComponentSummaries = useMemo(
        () => componentSummaries
          .filter((component) => component.nodeCount > 1 || component.relationCount > 0)
          .slice(0, 16),
        [componentSummaries]
      );

      const selectedComponent = useMemo(
        () => componentSummaries.find((component) => component.id === selectedComponentId) || null,
        [componentSummaries, selectedComponentId]
      );

      useEffect(() => {
        if (!selectedComponentId) return;
        if (componentSummaries.some((component) => component.id === selectedComponentId)) return;
        setSelectedComponentId(null);
      }, [componentSummaries, selectedComponentId]);

      const focusComponent = useCallback((component: GraphComponentSummary) => {
        setFocusNeighborhood(false);
        setSelectedComponentId((current) => current === component.id ? null : component.id);
        setGraphScope('all');
      }, []);

      const visibleGraph = useMemo(() => {
        return getKnowledgeGraphVisibleGraph(baseFilteredGraph.nodes, baseFilteredGraph.edges, {
          focusNeighborhood,
          focusNodeId,
          getEdgeKey: getVisualEdgeKey,
          getPrimaryConnectedComponentIds,
          graphScope,
          selectedComponentNodeIds: selectedComponent?.nodeIds || null,
          selectedEdgeId,
          selectedNodeId,
          selectedRelationshipTypesCount: selectedRelationshipTypes.length,
        });
      }, [baseFilteredGraph.edges, baseFilteredGraph.nodes, focusNeighborhood, focusNodeId, graphScope, selectedComponent, selectedEdgeId, selectedNodeId, selectedRelationshipTypes.length]);
        const visualGraph = useMemo(() => {
          const visualEdges = new Map<string, {
            firstEdge: GraphEdge;
            relationCount: number;
            typeCounts: Map<string, number>;
            weight: number;
            domains: Set<string>;
            evidence: RelationshipEvidence[];
          }>();

          const includeEvidence = (items: RelationshipEvidence[], edge: GraphEdge) => {
            const evidence = getRelationshipEvidence(edge);
            if (!evidence) return;
            const exists = items.some((item) => item.context === evidence.context);
            if (exists) return;
            const matchesSearch = Boolean(searchQuery && evidence.context.toLowerCase().includes(searchQuery));
            if (matchesSearch) {
              items.unshift(evidence);
              if (items.length > 8) items.pop();
              return;
            }
            if (items.length < 8) items.push(evidence);
          };

          visibleGraph.edges.forEach((edge) => {
            const aggregateKey = getVisualEdgeKey(edge);
            const existing = visualEdges.get(aggregateKey);
          const relationType = normalizeType(edge.type || edge.label);
          const relationTypes = getLinkRelationTypeCounts(edge, 12);
          const relationTypeTotal = relationTypes.reduce((sum, item) => sum + item.count, 0);
          const edgeRelationCount = Math.max(1, Number(edge.relationCount || 0), relationTypeTotal);
          const edgeWeight = Number(edge.weight || 0);
          const domain = typeof edge.properties?.domain === 'string' ? edge.properties.domain : null;

          const addTypeCounts = (typeCounts: Map<string, number>) => {
            if (relationTypes.length > 0) {
              relationTypes.forEach(({ type, count }) => {
                typeCounts.set(type, (typeCounts.get(type) || 0) + count);
              });
              return;
            }
            typeCounts.set(relationType, (typeCounts.get(relationType) || 0) + edgeRelationCount);
          };

          if (existing) {
              existing.relationCount += edgeRelationCount;
              existing.weight += edgeWeight;
              addTypeCounts(existing.typeCounts);
              if (domain) existing.domains.add(domain);
              includeEvidence(existing.evidence, edge);
              return;
            }

            const evidence: RelationshipEvidence[] = [];
            includeEvidence(evidence, edge);
            const typeCounts = new Map<string, number>();
            addTypeCounts(typeCounts);
            visualEdges.set(aggregateKey, {
              firstEdge: edge,
              relationCount: edgeRelationCount,
            typeCounts,
            weight: edgeWeight,
            domains: new Set(domain ? [domain] : []),
            evidence,
          });
        });

      return {
        nodes: visibleGraph.nodes,
        edges: Array.from(visualEdges.entries()).map(([id, group]) => {
          const relationTypes = Array.from(group.typeCounts.entries())
            .map(([type, count]) => ({ type, count }))
            .sort((a, b) => b.count - a.count || a.type.localeCompare(b.type));
          const dominantType = relationTypes[0]?.type || normalizeType(group.firstEdge.type || group.firstEdge.label);
          const label = relationTypes.length > 1
            ? `${getTypeLabel(dominantType)} +${relationTypes.length - 1}`
            : getTypeLabel(dominantType);

            const evidence = searchQuery
              ? [...group.evidence].sort((a, b) =>
                Number(b.context.toLowerCase().includes(searchQuery)) -
                Number(a.context.toLowerCase().includes(searchQuery))
              )
              : group.evidence;

            return {
              ...group.firstEdge,
              id,
              label,
            type: dominantType,
            weight: group.weight,
            relationCount: group.relationCount,
              properties: {
                ...group.firstEdge.properties,
                relationCount: group.relationCount,
                  relationTypes,
                  domains: Array.from(group.domains).sort(),
                  evidenceSnippets: evidence.slice(0, 5),
                },
              };
          }),
        };
        }, [searchQuery, visibleGraph]);
      const graphOverviewTopology = useMemo(
        () => buildCompactGraphTopology(visualGraph.nodes, visualGraph.edges),
        [visualGraph.edges, visualGraph.nodes]
      );
      const focusGraphCameraAt = useCallback((
        target: { x: number; y: number; z?: number },
        options?: Pick<GraphCameraFocusRequest, 'minDistance' | 'maxDistance' | 'duration'>
      ) => {
        const x = Number(target.x);
        const y = Number(target.y);
        const z = Number(target.z || 0);
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return false;

        setGraphCameraFocusRequest((current) => ({
          id: (current?.id || 0) + 1,
          target: { x, y, z },
          minDistance: options?.minDistance,
          maxDistance: options?.maxDistance,
          duration: options?.duration,
        }));
        return true;
      }, []);
      const focusGraphNodeInTopology = useCallback((nodeId: string) => {
        const position = graphOverviewTopology.positions.get(nodeId);
        if (!position) return false;
        const focused = focusGraphCameraAt(position, {
          minDistance: 58,
          maxDistance: 170,
          duration: 880,
        });
        if (focused) setGraphOverviewHover(null);
        return focused;
      }, [focusGraphCameraAt, graphOverviewTopology.positions]);
      const focusGraphEdgeInTopology = useCallback((edge: GraphEdge) => {
        const source = graphOverviewTopology.positions.get(edge.source);
        const target = graphOverviewTopology.positions.get(edge.target);
        if (!source || !target) return false;
        const focused = focusGraphCameraAt(
          {
            x: (source.x + target.x) / 2,
            y: (source.y + target.y) / 2,
            z: (source.z + target.z) / 2,
          },
          {
            minDistance: 124,
            maxDistance: 440,
            duration: 880,
          }
        );
        if (focused) setGraphOverviewHover(null);
        return focused;
      }, [focusGraphCameraAt, graphOverviewTopology.positions]);
      useEffect(() => {
        if (!graphOverviewOpen) {
          setGraphOverviewHover(null);
        }
      }, [graphOverviewOpen, visualGraph.edges.length, visualGraph.nodes.length]);
      const graphScopeActive = graphScope !== 'core';
      const graphUsesConnectedCore =
        graphScope === 'core' &&
        !graphFiltersActive &&
        visibleGraph.nodes.length > 0 &&
        visibleGraph.nodes.length < graphNodes.length;

          const selectedEdgeDetails = useMemo(() => {
            if (!selectedEdgeId) return null;
            const edge = visualGraph.edges.find((item) => item.id === selectedEdgeId) || (() => {
              const normalizedSelectedEdgeId = normalizeGraphText(selectedEdgeId);
              const rawEdge = graphEdges.find((item) =>
                item.id === selectedEdgeId ||
                normalizeGraphText(item.id) === normalizedSelectedEdgeId
              );
              if (!rawEdge) return null;
              const visualEdgeId = getVisualEdgeKey(rawEdge);
              return visualGraph.edges.find((item) => item.id === visualEdgeId) || null;
            })();
            if (!edge) return null;
            const edgeProperties = (edge.properties || {}) as Record<string, unknown>;

          const rawRelationTypes: RelationTypePayload[] = Array.isArray(edgeProperties.relationTypes)
            ? edgeProperties.relationTypes as RelationTypePayload[]
            : [];
          const relationTypes = (rawRelationTypes.length > 0
            ? rawRelationTypes
            : [{ type: normalizeType(edge.type || edge.label), count: edge.relationCount || 1 }]
          )
          .map((item) => {
            const type = normalizeType(item.type || undefined);
            return {
              type,
            count: Number(item.count || 0),
              color: getRelationshipColor(type),
            };
          })
            .filter((item) => item.type && item.count > 0)
            .sort((a, b) => b.count - a.count || a.type.localeCompare(b.type));

          const rawEvidence: RawEvidenceSnippet[] = Array.isArray(edgeProperties.evidenceSnippets)
            ? edgeProperties.evidenceSnippets as RawEvidenceSnippet[]
            : [];
          const fallbackEvidence = getRelationshipEvidence(edge);
          const evidenceCandidates = [
            ...rawEvidence
            .map((item) => ({
              context: normalizeEvidenceText(item.context),
              domain: typeof item.domain === 'string' ? item.domain : undefined,
              type: normalizeType(item.type),
              weight: Number(item.weight || 0),
            })),
            ...(fallbackEvidence ? [fallbackEvidence] : []),
          ].filter((item: RelationshipEvidence) => item.context);
          const seenEvidence = new Set<string>();
          const evidenceSnippets = evidenceCandidates
            .filter((item) => {
              const key = `${item.context.toLowerCase()}::${item.domain || ''}`;
              if (seenEvidence.has(key)) return false;
              seenEvidence.add(key);
              return true;
            })
            .sort((a, b) =>
              getEvidenceDetailScore(b) - getEvidenceDetailScore(a) ||
              a.context.localeCompare(b.context)
            )
            .slice(0, 5);
          const rawDomains = Array.isArray(edgeProperties.domains)
            ? edgeProperties.domains
            : (typeof edgeProperties.domain === 'string' ? [edgeProperties.domain] : []);
          const domains = rawDomains
            .map((domain: unknown) => String(domain || '').trim())
            .filter(Boolean);

            const source = entityById.get(edge.source);
            const target = entityById.get(edge.target);
          return {
            edge,
          source,
          target,
          sourceId: edge.source,
          targetId: edge.target,
          relationCount: Number(edge.relationCount || 1),
            weight: Number(edge.weight || 0),
            relationTypes,
            evidenceSnippets,
            domains,
            dominantColor: relationTypes[0]?.color || getRelationshipColor(edge.type || edge.label),
          };
        }, [entityById, graphEdges, selectedEdgeId, visualGraph.edges]);

      const selectedEdgeMetadataEntries = useMemo(() => {
        if (!selectedEdgeDetails) return [];
        return getMetadataEntries(selectedEdgeDetails.edge.properties, EDGE_METADATA_HIDDEN_KEYS, 8);
      }, [selectedEdgeDetails]);

      useEffect(() => {
        setSelectedEvidenceIndex(0);
      }, [selectedEdgeId]);

        useEffect(() => {
          if (!selectedEdgeDetails) return;
          setSelectedEvidenceIndex((current) =>
            Math.min(current, Math.max(0, selectedEdgeDetails.evidenceSnippets.length - 1))
          );
        }, [selectedEdgeDetails]);

          useEffect(() => {
            setExpandedSelectionMetadataGroups({});
          }, [selectedEdgeId, selectedEntity?.id]);

        const selectedEntityPathPreview = useMemo<GraphPathPreview | null>(() => {
          if (!selectedEntity) return null;
          const previousEntry = getPreviousEntityBrowseEntry(browseHistory, browseHistoryIndex, selectedEntity.id);
          if (!previousEntry) return null;
          return buildEntityPathPreview(previousEntry.id, selectedEntity.id, entityById, graphEdges);
        }, [browseHistory, browseHistoryIndex, entityById, graphEdges, selectedEntity]);

        const selectedHoverPathPreview = useMemo<GraphPathPreview | null>(() => {
          if (!selectedEntity || !selectedHoverRouteNodeId) return null;
          const hoveredEntity = entityById.get(selectedHoverRouteNodeId);
          if (!hoveredEntity || hoveredEntity.id === selectedEntity.id) return null;
          return buildEntityPathPreview(selectedEntity.id, hoveredEntity.id, entityById, visualGraph.edges);
        }, [entityById, selectedEntity, selectedHoverRouteNodeId, visualGraph.edges]);

        const activeEntityPathPreview = selectedHoverPathPreview || selectedEntityPathPreview;

          const pathFocusNodeIds = useMemo(
            () => activeEntityPathPreview?.nodes.map((node) => node.id) || [],
            [activeEntityPathPreview]
          );

          const revealSelectedEntityPath = useCallback(() => {
            if (!activeEntityPathPreview) return;
            setSearchTerm('');
            setSelectedEntityTypes([]);
            setSelectedRelationshipTypes([]);
            setSelectedComponentId(null);
            setFocusNeighborhood(false);
            setGraphScope('all');
            setParticleFlow(true);
            setPathFocusVersion((value) => value + 1);
          }, [activeEntityPathPreview]);

        const activeGraphFocus = useMemo(() => {
        const activeNodes = new Set<string>();
        const activeEdges = new Set<string>();
        const addActiveEdge = (edge: GraphEdge) => {
          activeEdges.add(edge.id);
          activeEdges.add(getVisualEdgeKey(edge));
          activeNodes.add(edge.source);
          activeNodes.add(edge.target);
        };
        const getHoverContinuationEdges = (hoverNodeId: string, excludedNodeIds: Set<string>) => {
          const continuationLimit = visibleGraph.nodes.length > 350
            ? 2
            : visibleGraph.nodes.length > 120
              ? 3
              : 4;
          return visualGraph.edges
            .filter((edge) => {
              if (edge.source !== hoverNodeId && edge.target !== hoverNodeId) return false;
              const otherId = edge.source === hoverNodeId ? edge.target : edge.source;
              return !excludedNodeIds.has(otherId);
            })
            .sort((a, b) => {
              const aOutgoing = a.source === hoverNodeId ? 1 : 0;
              const bOutgoing = b.source === hoverNodeId ? 1 : 0;
              if (aOutgoing !== bOutgoing) return bOutgoing - aOutgoing;
              return (
                Number(b.relationCount || 1) - Number(a.relationCount || 1) ||
                Number(b.weight || 0) - Number(a.weight || 0) ||
                getTypeLabel(a.type || a.label).localeCompare(getTypeLabel(b.type || b.label))
              );
            })
            .slice(0, continuationLimit);
        };

        const addHoverContinuationContext = (hoverNodeId: string, excludedNodeIds: Set<string>) => {
          getHoverContinuationEdges(hoverNodeId, excludedNodeIds).forEach(addActiveEdge);
        };
        const addSelectedHoverRouteContext = (hoverNodeId: string) => {
          if (!selectedEntity || !selectedNodeId) return;
          const hoveredNode = visibleGraph.nodes.find((node) => node.id === hoverNodeId);
          const hoveredEntity = entityById.get(hoverNodeId);
          const selectedKeys = new Set(
            [selectedNodeId, selectedEntity.id, selectedEntity.name].filter((value): value is string => Boolean(value))
          );
          const hoveredKeys = new Set(
            [hoverNodeId, hoveredNode?.label, hoveredEntity?.name].filter((value): value is string => Boolean(value))
          );
          const routeEdges = visibleGraph.edges
            .filter((edge) =>
              (selectedKeys.has(edge.source) && hoveredKeys.has(edge.target)) ||
              (selectedKeys.has(edge.target) && hoveredKeys.has(edge.source))
            )
            .sort((a, b) =>
              Number(b.relationCount || 1) - Number(a.relationCount || 1) ||
              Number(b.weight || 0) - Number(a.weight || 0) ||
              getTypeLabel(a.type || a.label).localeCompare(getTypeLabel(b.type || b.label))
            )
            .slice(0, visibleGraph.nodes.length > 350 ? 2 : 4);

          if (routeEdges.length === 0) return;
          activeNodes.add(selectedNodeId);
          activeNodes.add(hoverNodeId);
          routeEdges.forEach(addActiveEdge);
          addHoverContinuationContext(hoverNodeId, new Set([selectedNodeId, hoverNodeId]));
        };
        const addSearchPreviewNodeContext = (nodeId: string) => {
          activeNodes.add(nodeId);
          const query = searchQuery.trim();
          const maxPreviewEdges = visibleGraph.nodes.length > 350 ? 3 : visibleGraph.nodes.length > 120 ? 4 : 5;
          visibleGraph.edges
            .filter((edge) => edge.source === nodeId || edge.target === nodeId)
            .sort((a, b) => {
              const aMatches = query && getEdgeSearchText(a).includes(query) ? 1 : 0;
              const bMatches = query && getEdgeSearchText(b).includes(query) ? 1 : 0;
              if (aMatches !== bMatches) return bMatches - aMatches;
              return (
                Number(b.relationCount || 1) - Number(a.relationCount || 1) ||
                Number(b.weight || 0) - Number(a.weight || 0) ||
                getTypeLabel(a.type || a.label).localeCompare(getTypeLabel(b.type || b.label))
              );
            })
            .slice(0, maxPreviewEdges)
            .forEach(addActiveEdge);
        };
        const addSemanticTypePreviewContext = () => {
          if (!semanticTypePreview) return false;
          const previewType = normalizeType(semanticTypePreview.type);
          if (!previewType) return false;

          if (semanticTypePreview.kind === 'entity') {
            visibleGraph.nodes.forEach((node) => {
              if (normalizeType(node.type) === previewType) activeNodes.add(node.id);
            });
            if (activeNodes.size === 0) return false;
            visualGraph.edges.forEach((edge) => {
              if (activeNodes.has(edge.source) || activeNodes.has(edge.target)) addActiveEdge(edge);
            });
            return true;
          }

          visualGraph.edges.forEach((edge) => {
            const matchesType =
              normalizeType(edge.type || edge.label) === previewType ||
              getLinkRelationTypeCounts(edge, 12).some((item) => normalizeType(item.type) === previewType);
            if (matchesType) addActiveEdge(edge);
          });
          return activeEdges.size > 0 || activeNodes.size > 0;
        };

        if (addSemanticTypePreviewContext()) {
          return { activeNodes, activeEdges };
        }

        const passiveSelectedNodeFocus = Boolean(
          selectedNodeId &&
          focusNodeId === selectedNodeId &&
          !focusNeighborhood &&
          !selectionPanelExpanded &&
          !searchQuery
        );

        if (selectedEdgeDetails) {
          activeEdges.add(selectedEdgeDetails.edge.id);
          activeNodes.add(selectedEdgeDetails.sourceId);
          activeNodes.add(selectedEdgeDetails.targetId);
        } else if (selectedHoverPathPreview) {
          const pathNodeIds = new Set(selectedHoverPathPreview.nodes.map((node) => node.id));
          selectedHoverPathPreview.nodes.forEach((node) => activeNodes.add(node.id));
          selectedHoverPathPreview.segments.forEach((segment) => {
            activeEdges.add(segment.edgeId);
            activeEdges.add(segment.visualEdgeId);
          });
          addHoverContinuationContext(selectedHoverPathPreview.targetId, pathNodeIds);
        } else if (selectedHoverRouteNodeId && visibleGraph.nodeIds.has(selectedHoverRouteNodeId)) {
          addSelectedHoverRouteContext(selectedHoverRouteNodeId);
        } else if (selectedEntityPathPreview) {
          selectedEntityPathPreview.nodes.forEach((node) => activeNodes.add(node.id));
          selectedEntityPathPreview.segments.forEach((segment) => {
            activeEdges.add(segment.edgeId);
            activeEdges.add(segment.visualEdgeId);
          });
        } else if (searchQuery && searchPreviewNodeId && visibleGraph.nodeIds.has(searchPreviewNodeId)) {
          addSearchPreviewNodeContext(searchPreviewNodeId);
        } else if (focusNodeId && visibleGraph.nodeIds.has(focusNodeId) && !passiveSelectedNodeFocus) {
          const compactFocus = !focusNeighborhood && !selectionPanelExpanded;
          const maxFocusEdges = compactFocus
            ? visibleGraph.nodes.length > 350
              ? 5
              : visibleGraph.nodes.length > 120
                ? 7
                : 9
            : Number.POSITIVE_INFINITY;
          const focusEdges = visibleGraph.edges
            .filter((edge) => edge.source === focusNodeId || edge.target === focusNodeId)
            .sort((a, b) =>
              Number(b.relationCount || 1) - Number(a.relationCount || 1) ||
              Number(b.weight || 0) - Number(a.weight || 0) ||
              getTypeLabel(a.type || a.label).localeCompare(getTypeLabel(b.type || b.label))
            )
            .slice(0, maxFocusEdges);
          if (focusEdges.length > 0 || searchQuery || focusNeighborhood) {
            activeNodes.add(focusNodeId);
          }
          focusEdges.forEach(addActiveEdge);
      } else if (searchQuery && searchPreviewEdgeId) {
          const previewEdge =
            visibleGraph.edges.find((edge) => edge.id === searchPreviewEdgeId) ||
            visualGraph.edges.find((edge) => edge.id === searchPreviewEdgeId);
          if (previewEdge) addActiveEdge(previewEdge);
      } else if (searchQuery) {
        visibleGraph.nodes.forEach((node) => {
          if (getEntitySearchText(node).includes(searchQuery)) {
            activeNodes.add(node.id);
          }
          });
          visibleGraph.edges.forEach((edge) => {
            if (
              activeNodes.has(edge.source) ||
              activeNodes.has(edge.target) ||
              getEdgeSearchText(edge).includes(searchQuery)
            ) {
              activeEdges.add(edge.id);
              activeEdges.add(getVisualEdgeKey(edge));
              activeNodes.add(edge.source);
            activeNodes.add(edge.target);
          }
        });
      }
        return { activeNodes, activeEdges };
      }, [
        entityById,
        focusNeighborhood,
        focusNodeId,
        searchPreviewEdgeId,
        searchPreviewNodeId,
        searchQuery,
        semanticTypePreview,
        selectedEdgeDetails,
        selectedEntity,
        selectedEntityPathPreview,
        selectedHoverPathPreview,
        selectedHoverRouteNodeId,
        selectedNodeId,
        selectionPanelExpanded,
        visibleGraph,
        visualGraph.edges,
      ]);

      const visibleEntityTypeSummary = useMemo(() => {
      const counts = new Map<string, number>();
      visibleGraph.nodes.forEach((node) => {
        const type = normalizeType(node.type);
        counts.set(type, (counts.get(type) || 0) + 1);
      });
      return Array.from(counts.entries())
        .map(([type, count]) => ({ type, count }))
          .sort((a, b) => b.count - a.count || a.type.localeCompare(b.type));
      }, [visibleGraph.nodes]);

      const relationBrowserRows = useMemo(() => {
        return visualGraph.edges
          .map((edge) => {
            const rawRelationTypes: RelationTypePayload[] = Array.isArray(edge.properties?.relationTypes)
              ? edge.properties.relationTypes as RelationTypePayload[]
              : [];
            const relationTypes = (rawRelationTypes.length > 0
              ? rawRelationTypes
              : [{ type: normalizeType(edge.type || edge.label), count: edge.relationCount || 1 }]
            )
              .map((item) => {
                const type = normalizeType(item.type || undefined);
                return {
                  type,
                count: Number(item.count || 0),
                  color: getRelationshipColor(type),
                };
              })
              .filter((item) => item.type && item.count > 0)
              .sort((a, b) => b.count - a.count || a.type.localeCompare(b.type));
            const source = entityById.get(edge.source);
            const target = entityById.get(edge.target);
            const sourceLabel = source?.name || edge.source;
            const targetLabel = target?.name || edge.target;
            const rawEvidence: RawEvidenceSnippet[] = Array.isArray(edge.properties?.evidenceSnippets)
              ? edge.properties.evidenceSnippets as RawEvidenceSnippet[]
              : [];
              const evidenceText = rawEvidence
                .map((item) => normalizeEvidenceText(item.context))
                .filter(Boolean)
                .join(' ');
              const previewEvidence = rawEvidence
                .map((item) => normalizeEvidenceText(item.context))
                .filter(Boolean)
                .sort((a, b) =>
                  Number(isEvidenceExcerpt(a)) - Number(isEvidenceExcerpt(b)) ||
                  b.length - a.length
                )[0] || '';
              const matchedEvidence = getEvidenceSearchSnippet(rawEvidence, searchQuery);
              const domains = Array.isArray(edge.properties?.domains)
                ? edge.properties.domains.map((domain: unknown) => String(domain || '').trim()).filter(Boolean)
                : [];
            const searchableText = [
              sourceLabel,
              targetLabel,
              ...getTypeSearchAliases(edge.label),
              ...getTypeSearchAliases(edge.type),
              ...domains,
              evidenceText,
              ...relationTypes.flatMap((item) => getTypeSearchAliases(item.type)),
            ].filter(Boolean).join(' ').toLowerCase();

            return {
              edge,
              source,
              target,
              sourceLabel,
              targetLabel,
              relationCount: Number(edge.relationCount || 1),
              weight: Number(edge.weight || 0),
                relationTypes,
                domains: Array.from(new Set(domains)).slice(0, 2),
                dominantColor: relationTypes[0]?.color || getRelationshipColor(edge.type || edge.label),
                searchableText,
                matchedEvidence,
                  previewEvidence,
                  evidenceCount: rawEvidence.length,
                };
            })
          .filter((row) => !searchQuery || row.searchableText.includes(searchQuery))
          .sort((a, b) =>
            b.relationCount - a.relationCount ||
            b.weight - a.weight ||
            a.sourceLabel.localeCompare(b.sourceLabel) ||
            a.targetLabel.localeCompare(b.targetLabel)
          );
      }, [entityById, searchQuery, visualGraph.edges]);

      const relationTypeMatrix = useMemo(() => {
        const buckets = new Map<string, { type: string; count: number; color: string; rows: number; weight: number }>();
        relationBrowserRows.forEach((row) => {
          row.relationTypes.forEach((relationType) => {
            const type = normalizeType(relationType.type);
            const existing = buckets.get(type) || {
              type,
              count: 0,
              color: relationType.color || getRelationshipColor(type),
              rows: 0,
              weight: 0,
            };
            existing.count += relationType.count;
            existing.rows += 1;
            existing.weight += row.weight || 0;
            buckets.set(type, existing);
          });
        });
        const items = Array.from(buckets.values()).sort((a, b) =>
          b.count - a.count ||
          b.rows - a.rows ||
          a.type.localeCompare(b.type)
        );
        const total = items.reduce((sum, item) => sum + item.count, 0);
        const max = items[0]?.count || 1;
        return {
          items: items.map((item) => ({
            ...item,
            share: total > 0 ? item.count / total : 0,
            intensity: max > 0 ? item.count / max : 0,
          })),
          total,
          max,
        };
        }, [relationBrowserRows]);

        const globalLegendRelationshipTypeSummary = relationshipTypeSummary.map((item) => ({
          ...item,
          share: 0,
          intensity: 0,
          rows: item.count,
          weight: 0,
        }));
        const legendEntityTypeSummary = selectedEntityTypes.length > 0
          ? entityTypeSummary
          : visibleEntityTypeSummary.length > 0
          ? visibleEntityTypeSummary
          : entityTypeSummary;
        const legendRelationshipTypeSummary = selectedRelationshipTypes.length > 0
          ? globalLegendRelationshipTypeSummary
          : relationTypeMatrix.items.length > 0
          ? relationTypeMatrix.items
          : globalLegendRelationshipTypeSummary;
        const visibleLegendEntityTypes = graphLegendEntitiesExpanded
          ? legendEntityTypeSummary
          : legendEntityTypeSummary.slice(0, 8);
        const visibleLegendRelationshipTypes = graphLegendRelationsExpanded
          ? legendRelationshipTypeSummary
          : legendRelationshipTypeSummary.slice(0, 8);
        const maxLegendEntityTypeCount = Math.max(1, legendEntityTypeSummary[0]?.count || 1);
        const maxLegendRelationshipTypeCount = Math.max(1, legendRelationshipTypeSummary[0]?.count || 1);
        const semanticTypePreviewSummary = useMemo(() => {
          if (!semanticTypePreview) return null;
          const previewType = normalizeType(semanticTypePreview.type);
          if (!previewType) return null;

          if (semanticTypePreview.kind === 'entity') {
            const colors = getTypeColors(previewType);
            const nodeIds = new Set(
              visibleGraph.nodes
                .filter((node) => normalizeType(node.type) === previewType)
                .map((node) => node.id)
            );
            const linkCount = visualGraph.edges.filter((edge) =>
              nodeIds.has(edge.source) || nodeIds.has(edge.target)
            ).length;
            return {
              color: colors.main,
              label: getTypeLabel(previewType),
              text: `${formatNumber(nodeIds.size)} nodes / ${formatNumber(linkCount)} links`,
            };
          }

          const color = getRelationshipColor(previewType);
          const nodeIds = new Set<string>();
          let relationCount = 0;
          let linkCount = 0;
          visualGraph.edges.forEach((edge) => {
            const relationTypes = getLinkRelationTypeCounts(edge, 12);
            const matchingTypes = relationTypes.filter((item) => normalizeType(item.type) === previewType);
            const matchesDominant = normalizeType(edge.type || edge.label) === previewType;
            if (matchingTypes.length === 0 && !matchesDominant) return;
            linkCount += 1;
            relationCount += matchingTypes.reduce((sum, item) => sum + item.count, 0) || Math.max(1, Number(edge.relationCount || 1));
            nodeIds.add(edge.source);
            nodeIds.add(edge.target);
          });
          return {
            color,
            label: getTypeLabel(previewType),
            text: `${formatNumber(relationCount)} relations / ${formatNumber(linkCount)} links / ${formatNumber(nodeIds.size)} nodes`,
          };
        }, [semanticTypePreview, visibleGraph.nodes, visualGraph.edges]);

      const selectedRelationshipGroups = useMemo(() => {
      if (!selectedEntity) return [];
      const counts = new Map<string, number>();
      const selectedEntityKeys = new Set([selectedEntity.id, selectedEntity.name].filter(Boolean));
      relationships.forEach((rel) => {
        if (!selectedEntityKeys.has(rel.source_id) && !selectedEntityKeys.has(rel.target_id)) return;
        const type = normalizeType(rel.relationship_type);
        counts.set(type, (counts.get(type) || 0) + 1);
      });
      return Array.from(counts.entries())
        .map(([type, count]) => ({ type, count, color: getRelationshipColor(type) }))
        .sort((a, b) => b.count - a.count || a.type.localeCompare(b.type));
    }, [relationships, selectedEntity]);

    const selectedRelationshipBundles = useMemo(() => {
      if (!selectedEntity) return [];

        const bundles = new Map<string, {
          otherId: string;
          otherEntity?: Entity;
          primaryEdge?: GraphEdge;
          inCount: number;
          outCount: number;
          totalWeight: number;
          typeCounts: Map<string, number>;
        relationships: Relationship[];
      }>();

      relationships.forEach((rel) => {
        const isOutgoing = rel.source_id === selectedEntity.id;
        const otherId = isOutgoing ? rel.target_id : rel.source_id;
        const existing = bundles.get(otherId) || {
          otherId,
          otherEntity: entities.find((entity) => entity.id === otherId || entity.name === otherId),
          inCount: 0,
          outCount: 0,
          totalWeight: 0,
          typeCounts: new Map<string, number>(),
          relationships: [],
        };
        const type = normalizeType(rel.relationship_type);

        if (isOutgoing) {
          existing.outCount += 1;
        } else {
          existing.inCount += 1;
        }
        existing.totalWeight += rel.weight || 0;
          existing.typeCounts.set(type, (existing.typeCounts.get(type) || 0) + 1);
          existing.relationships.push(rel);
          bundles.set(otherId, existing);
        });

        return Array.from(bundles.values())
          .map((bundle) => {
            const edgeKeys = new Set(
              bundle.relationships.flatMap((rel) => [
                [rel.source_id, rel.target_id].join('::'),
                [rel.target_id, rel.source_id].join('::'),
              ])
            );
            const otherEntity = bundle.otherEntity || entities.find((entity) => entity.id === bundle.otherId || entity.name === bundle.otherId);
            const selectedEntityKeys = new Set([selectedEntity.id, selectedEntity.name].filter(Boolean));
            const otherEntityKeys = new Set([bundle.otherId, otherEntity?.id, otherEntity?.name].filter(Boolean));
            const primaryEdge = visualGraph.edges
              .filter((edge) =>
                edgeKeys.has(edge.id) ||
                (selectedEntityKeys.has(edge.source) && otherEntityKeys.has(edge.target)) ||
                (selectedEntityKeys.has(edge.target) && otherEntityKeys.has(edge.source))
              )
              .sort((a, b) =>
                Number(b.relationCount || 1) - Number(a.relationCount || 1) ||
                Number(b.weight || 0) - Number(a.weight || 0)
              )[0];

            return {
              ...bundle,
              otherEntity,
              primaryEdge,
              total: bundle.relationships.length,
              types: Array.from(bundle.typeCounts.entries())
                .map(([type, count]) => ({ type, count, color: getRelationshipColor(type) }))
                .sort((a, b) => b.count - a.count || a.type.localeCompare(b.type)),
            };
          })
          .sort((a, b) => b.total - a.total || (a.otherEntity?.name || a.otherId).localeCompare(b.otherEntity?.name || b.otherId));
      }, [entities, relationships, selectedEntity, visualGraph.edges]);

    const selectedEntityGraphRole = useMemo(() => {
      if (!selectedEntity) return null;

      const rankedEntities = entities
        .map((entity) => ({
          entity,
          degree: degreeMap.get(entity.id) || { in: 0, out: 0, total: 0 },
        }))
        .sort((a, b) =>
          b.degree.total - a.degree.total ||
          Number(b.entity.confidence || 0) - Number(a.entity.confidence || 0) ||
          a.entity.name.localeCompare(b.entity.name)
        );
      const rank = rankedEntities.findIndex((item) => item.entity.id === selectedEntity.id) + 1;
      const componentIndex = componentSummaries.findIndex((component) => component.nodeIds.has(selectedEntity.id));
      const component = componentIndex >= 0 ? componentSummaries[componentIndex] : null;
      const strongestBundle = selectedRelationshipBundles[0] || null;
      const totalRelations = selectedRelationshipBundles.reduce((sum, bundle) => sum + bundle.total, 0);

      return {
        rank: rank > 0 ? rank : null,
        totalEntities: entities.length,
        component,
        componentLabel: component ? `C${componentIndex + 1}` : null,
        strongestBundle,
        totalRelations,
      };
    }, [componentSummaries, degreeMap, entities, selectedEntity, selectedRelationshipBundles]);

      const selectedMetadataEntries = useMemo(() => {
        if (!selectedEntity) return [];
        return getMetadataEntries(selectedEntity.properties, ENTITY_METADATA_HIDDEN_KEYS, 12);
      }, [selectedEntity]);

      const selectedEntityMetadataFacts = useMemo<GraphMetadataFact[]>(() => {
        if (!selectedEntity) return [];
        const properties = selectedEntity.properties || {};
        const entityColors = getTypeColors(selectedEntity.entity_type, selectedEntity.name);
        const degree = degreeMap.get(selectedEntity.id) || { in: 0, out: 0, total: 0 };
        const occurrence = getFiniteMetadataNumber(properties.occurrence, properties.hits, properties.count);
        const confidence = getFiniteMetadataNumber(selectedEntity.confidence, properties.confidence);
        const domain = getFirstMetadataValue(properties, ['domain', 'domains', 'namespace', 'collection']);
        const source = getFirstMetadataValue(properties, ['source', 'sources', 'document', 'documents', 'file', 'path', 'url']);
        const facts: GraphMetadataFact[] = [];

        if (degree.total > 0) {
          facts.push({
            id: 'entity-relations',
            label: 'Relations',
            value: formatNumber(degree.total),
            color: entityColors.main,
            icon: <HubIcon sx={{ fontSize: 13 }} />,
            priority: 0,
          });
        }
        if (selectedEntityGraphRole?.component) {
          facts.push({
            id: 'entity-component',
            label: selectedEntityGraphRole.componentLabel || 'Component',
            value: `${formatNumber(selectedEntityGraphRole.component.nodeCount)} nodes`,
            color: entityColors.main,
            icon: <ExploreIcon sx={{ fontSize: 13 }} />,
            priority: 1,
          });
        }
        if (occurrence) {
          facts.push({
            id: 'entity-occurrence',
            label: 'Hits',
            value: formatNumber(Math.round(occurrence)),
            color: accentColors.success,
            icon: <MetricIcon sx={{ fontSize: 13 }} />,
            priority: 2,
          });
        }
        if (confidence) {
          facts.push({
            id: 'entity-confidence',
            label: 'Confidence',
            value: confidence <= 1 ? confidence.toFixed(2) : formatNumber(Math.round(confidence)),
            color: accentColors.slate,
            icon: <InfoIcon sx={{ fontSize: 13 }} />,
            priority: 3,
          });
        }
        if (domain) {
          facts.push({
            id: 'entity-domain',
            label: 'Domain',
            value: formatMetadataFactValue(domain, 24),
            color: accentColors.primary,
            icon: <DataIcon sx={{ fontSize: 13 }} />,
            priority: 4,
          });
        }
        if (source) {
          facts.push({
            id: 'entity-source',
            label: 'Source',
            value: formatMetadataFactValue(source, 24),
            color: accentColors.info,
            icon: <CodeIcon sx={{ fontSize: 13 }} />,
            priority: 5,
          });
        }
        return facts.sort((a, b) => a.priority - b.priority).slice(0, 6);
      }, [degreeMap, selectedEntity, selectedEntityGraphRole]);

      const selectedRelationMetadataFacts = useMemo<GraphMetadataFact[]>(() => {
        if (!selectedEdgeDetails) return [];
        const facts: GraphMetadataFact[] = [];
        facts.push({
          id: 'relation-count',
          label: 'Relations',
          value: formatNumber(selectedEdgeDetails.relationCount),
          color: selectedEdgeDetails.dominantColor,
          icon: <RelationIcon sx={{ fontSize: 13 }} />,
          priority: 0,
        });

        if (selectedEdgeDetails.weight > 0) {
          facts.push({
            id: 'relation-weight',
            label: 'Weight',
            value: formatNumber(Math.round(selectedEdgeDetails.weight)),
            color: accentColors.warning,
            icon: <MetricIcon sx={{ fontSize: 13 }} />,
            priority: 1,
          });
        }
        if (selectedEdgeDetails.evidenceSnippets.length > 0) {
          facts.push({
            id: 'relation-evidence',
            label: 'Evidence',
            value: formatNumber(selectedEdgeDetails.evidenceSnippets.length),
            color: selectedEdgeDetails.dominantColor,
            icon: <InfoIcon sx={{ fontSize: 13 }} />,
            priority: 2,
          });
        }
        if (selectedEdgeDetails.domains.length > 0) {
          facts.push({
            id: 'relation-domain',
            label: 'Domain',
            value: truncateGraphLabel(selectedEdgeDetails.domains[0], 22),
            color: accentColors.primary,
            icon: <DataIcon sx={{ fontSize: 13 }} />,
            priority: 3,
          });
        }
        return facts.sort((a, b) => a.priority - b.priority).slice(0, 5);
      }, [selectedEdgeDetails]);

      const filteredEntities = useMemo(() => {
      const relevanceScore = (entity: Entity) => {
        const degree = degreeMap.get(entity.id) || { in: 0, out: 0, total: 0 };
        const occurrence = Number(entity.properties?.occurrence || 0);
        if (!searchQuery) {
          return degree.total * 100 + occurrence * 8 + Number(entity.confidence || 0);
        }

        const name = entity.name.toLowerCase();
        const type = normalizeType(entity.entity_type);
        const searchText = getEntitySearchText(entity);
        const directNameMatch = name === searchQuery ? 5000 : name.startsWith(searchQuery) ? 2600 : name.includes(searchQuery) ? 1800 : 0;
        const typeMatch = type.includes(searchQuery) ? 700 : 0;
        const metadataMatch = searchText.includes(searchQuery) ? 350 : 0;
        return directNameMatch + typeMatch + metadataMatch + degree.total * 80 + occurrence * 8 + Number(entity.confidence || 0);
      };

      return entities
        .filter(entity => visibleGraph.nodeIds.has(entity.id))
        .sort((a, b) => relevanceScore(b) - relevanceScore(a) || a.name.localeCompare(b.name));
    }, [degreeMap, entities, searchQuery, visibleGraph.nodeIds]);

    const allSearchResults = useMemo<GraphSearchResult[]>(() => {
      if (!searchQuery) return [];

      const entityResults = filteredEntities
        .filter((entity) => getEntitySearchText(entity).includes(searchQuery))
        .map((entity) => {
          const colors = getTypeColors(entity.entity_type, entity.name);
          const degree = degreeMap.get(entity.id) || { total: 0 };
          return {
            id: `entity:${entity.id}`,
            kind: 'entity' as const,
            label: entity.name,
            secondary: `${getEntityDisplayTypeLabel(entity.entity_type, entity.name)} • ${degree.total} relations`,
            snippet: getEntitySearchSnippet(entity, searchQuery) || undefined,
              color: colors.main,
              badges: [
                { label: getEntityDisplayTypeLabel(entity.entity_type, entity.name), color: colors.light },
                ...(degree.total > 0
                  ? [{ label: `${degree.total.toLocaleString()} rel`, color: accentColors.primary }]
                  : []),
                ...(Number(entity.properties?.occurrence || 0) > 0
                  ? [{ label: `${Number(entity.properties?.occurrence || 0).toLocaleString()} hits`, color: accentColors.success }]
                  : []),
              ...(Number(entity.confidence || 0) > 0
                ? [{ label: `${Number(entity.confidence || 0).toFixed(2)} conf`, color: accentColors.slate }]
                : []),
            ].slice(0, 4),
            entity,
          };
        });

      const relationResults = relationBrowserRows.map((row) => {
        const primaryType = row.relationTypes[0];
        const typeLabel = primaryType ? getTypeLabel(primaryType.type) : 'Related';
        const typeMix = row.relationTypes
          .slice(0, 2)
          .map(({ type, count }) => `${getTypeLabel(type)} ${count}`)
          .join(' / ');
        const graphSummary = [
          typeMix || typeLabel,
          `${row.relationCount.toLocaleString()} relation${row.relationCount === 1 ? '' : 's'}`,
          row.domains[0],
        ].filter(Boolean).join(' • ');
        const evidenceSummary = normalizeGraphSearchResultText(row.matchedEvidence || '');
        const snippet = evidenceSummary && evidenceSummary !== graphSummary
          ? truncateGraphLabel(evidenceSummary, 128)
          : graphSummary;
        return {
          id: `relation:${row.edge.id}`,
          kind: 'relation' as const,
          label: `${row.sourceLabel} -> ${row.targetLabel}`,
          secondary: `${typeLabel} • ${row.relationCount} relation${row.relationCount === 1 ? '' : 's'}${row.domains[0] ? ` • ${row.domains[0]}` : ''}`,
          snippet,
          color: row.dominantColor,
          route: {
            sourceLabel: row.sourceLabel,
            sourceType: row.source?.entity_type,
            targetLabel: row.targetLabel,
            targetType: row.target?.entity_type,
            relationType: primaryType?.type || row.edge.type || row.edge.label || 'related',
            relationLabel: typeLabel,
            relationCount: row.relationCount,
          },
          badges: [
              ...row.relationTypes.slice(0, 2).map(({ type, count, color }) => ({
                label: `${getTypeLabel(type)} ${count}`,
                color,
              })),
              ...(row.evidenceCount > 0
                ? [{ label: `${row.evidenceCount.toLocaleString()} ev`, color: accentColors.success }]
                : []),
              ...row.domains.slice(0, 1).map((domain) => ({ label: domain, color: accentColors.slate })),
            ].slice(0, 4),
          edge: row.edge,
        };
      });

      return blendGraphSearchResults<GraphSearchResult>(entityResults, relationResults, 80);
    }, [degreeMap, filteredEntities, relationBrowserRows, searchQuery]);

    const searchResultCounts = useMemo(() => ({
      all: allSearchResults.length,
      entities: allSearchResults.filter((result) => result.kind === 'entity').length,
      relations: allSearchResults.filter((result) => result.kind === 'relation').length,
    }), [allSearchResults]);

    const searchResults = useMemo(
      () => searchKindFilter === 'all'
        ? allSearchResults
        : allSearchResults.filter((result) => result.kind === searchKindFilter),
      [allSearchResults, searchKindFilter]
    );

      useEffect(() => {
        setSearchResultIndex(0);
        if (!searchQuery) {
          setSearchTrayExpanded(false);
          setSearchKindFilter('all');
        }
      }, [searchQuery]);

      useEffect(() => {
        setSearchResultIndex(0);
      }, [searchKindFilter]);

    useEffect(() => {
      setSearchResultIndex((current) => {
        if (searchResults.length === 0) return 0;
        return Math.min(current, searchResults.length - 1);
      });
    }, [searchResults.length]);

    const selectedSearchResult = searchResults[searchResultIndex] || null;
    const fallbackViewportWidth = typeof window !== 'undefined' ? window.innerWidth : isFullscreen ? 1440 : 1180;
    const fallbackViewportHeight = typeof window !== 'undefined' ? window.innerHeight : isFullscreen ? 820 : 680;
    const selectionViewportWidth = selectionAnchor?.width || fallbackViewportWidth;
    const selectionViewportHeight = selectionAnchor?.height || fallbackViewportHeight;
    const searchResultWindow = useMemo(() => {
      if (searchResults.length === 0) return [];
      const maxVisibleResults = selectionViewportWidth < 700
        ? searchTrayExpanded ? 3 : 2
        : searchTrayExpanded
          ? selectionViewportWidth >= 1600 ? 5 : selectionViewportWidth < 1300 ? 3 : 4
        : 3;
      const start = Math.min(
        Math.max(0, searchResultIndex - (searchTrayExpanded ? 3 : 1)),
        Math.max(0, searchResults.length - maxVisibleResults)
      );
      return searchResults
        .slice(start, start + maxVisibleResults)
        .map((result, offset) => ({ result, index: start + offset }));
    }, [searchResultIndex, searchResults, searchTrayExpanded, selectionViewportWidth]);
    const searchHighlightTerms = useMemo(() => {
      const normalized = normalizeGraphText(searchQuery);
      if (!normalized) return [];
      const terms = [
        normalized,
        ...normalized.split(/\s+/).filter((term) => term.length > 1 || normalized.length === 1),
      ];
      return Array.from(new Set(terms))
        .sort((a, b) => b.length - a.length || a.localeCompare(b))
        .slice(0, 10);
    }, [searchQuery]);
    const renderHighlightedSearchText = useCallback((
      value: string | undefined,
      color = accentColors.primary,
      maxLength?: number
    ): ReactNode => {
      const display = maxLength ? truncateGraphLabel(value || '', maxLength) : value || '';
      if (!display || searchHighlightTerms.length === 0) return display;

      const matcher = new RegExp(`(${searchHighlightTerms.map(escapeGraphSearchRegExp).join('|')})`, 'ig');
      const parts = display.split(matcher).filter((part) => part.length > 0);
      return parts.map((part, index) => {
        const matched = searchHighlightTerms.includes(part.toLowerCase());
        if (!matched) {
          return (
            <Box key={`${part}-${index}`} component="span">
              {part}
            </Box>
          );
        }

        return (
          <Box
            key={`${part}-${index}`}
            component="mark"
            data-kg-search-match="true"
            sx={{
              display: 'inline',
              px: 0.18,
              mx: 0.03,
              borderRadius: '3px',
              color: 'inherit',
              bgcolor: alpha(color, 0.24),
              boxShadow: `0 0 0 1px ${alpha(color, 0.22)}`,
            }}
          >
            {part}
          </Box>
        );
      });
    }, [searchHighlightTerms]);

    const focusSearchResult = useCallback((result: GraphSearchResult, openDetails = false) => {
      setInspectorOpen(false);
      setSelectionActionsOpen(false);
      setNeighborOrbitExpanded(false);
      setSelectionPanelExpanded(openDetails);

      if (result.kind === 'entity' && result.entity) {
        setSearchPreviewNodeId(null);
        setSearchPreviewEdgeId(null);
        setSelectedEdgeId(null);
        setSelectedEntity(result.entity);
        setSelectedNodeId(result.entity.id);
        setExpandedRelationshipBundleId(null);
        if (openDetails) {
          setTabValue(2);
          setSelectionPanelExpanded(true);
        }
        setInspectorOpen(false);
        return;
      }

      if (result.kind === 'relation' && result.edge) {
        setSearchPreviewNodeId(null);
        setSearchPreviewEdgeId(null);
        setSelectedEdgeId(result.edge.id);
        setSelectedNodeId(null);
        setSelectedEntity(null);
        setExpandedRelationshipBundleId(null);
        if (openDetails) {
          setTabValue(2);
          setSelectionPanelExpanded(true);
        }
        setInspectorOpen(false);
      }
    }, []);

    const shouldAutoOpenSearchDetails = useCallback((result: GraphSearchResult) => {
      if (result.kind === 'relation') return true;
      if (!result.entity) return false;

      const entityDegree = degreeMap.get(result.entity.id);
      return Boolean(
        getUsefulEntityDescription(result.entity.description) ||
        result.entity.aliases.length > 0 ||
        getMetadataEntries(result.entity.properties, ENTITY_METADATA_HIDDEN_KEYS, 1).length > 0 ||
        (entityDegree && entityDegree.total > 0)
      );
    }, [degreeMap]);

    const stageSearchPreview = useCallback((result: GraphSearchResult) => {
      if (result.kind === 'entity' && result.entity) {
        setSearchPreviewNodeId(result.entity.id);
        setSearchPreviewEdgeId(null);
        return;
      }

      if (result.kind === 'relation' && result.edge) {
        setSearchPreviewNodeId(null);
        setSearchPreviewEdgeId(result.edge.id);
        return;
      }

      setSearchPreviewNodeId(null);
      setSearchPreviewEdgeId(null);
    }, []);

      const commitSearchResult = useCallback((
        result: GraphSearchResult,
        openDetails = false,
        options: { preserveSearch?: boolean; autoDetails?: boolean } = {}
      ) => {
        const revealDetails = openDetails || Boolean(options.autoDetails && shouldAutoOpenSearchDetails(result));
        const preserveSearch = options.preserveSearch ?? true;
        if (preserveSearch) {
          setSearchTrayOpen(true);
          setSearchTrayExpanded(!revealDetails);
        } else {
          setSearchTerm('');
          setSearchTrayOpen(false);
          setSearchTrayExpanded(false);
        }
        setSelectedEntityTypes([]);
      setSelectedRelationshipTypes([]);
      setSelectedComponentId(null);
      setFocusNeighborhood(false);
      setGraphScope('all');
      setHoveredNodeId(null);
      focusSearchResult(result, revealDetails);
    }, [focusSearchResult, shouldAutoOpenSearchDetails]);

    const selectedSearchResultSignature = selectedSearchResult
      ? [
        searchQuery,
        searchResultIndex,
        selectedSearchResult.kind,
        selectedSearchResult.kind === 'entity'
          ? selectedSearchResult.entity?.id
          : selectedSearchResult.edge?.id,
      ].filter(Boolean).join('::')
      : null;

    useEffect(() => {
      if (!searchQuery || !selectedSearchResult || !selectedSearchResultSignature) {
        searchAutoFocusSignatureRef.current = null;
        return;
      }
      if (searchAutoFocusSignatureRef.current === selectedSearchResultSignature) return;

      searchAutoFocusSignatureRef.current = selectedSearchResultSignature;
      stageSearchPreview(selectedSearchResult);
    }, [stageSearchPreview, searchQuery, selectedSearchResult, selectedSearchResultSignature]);

    const previewSearchResult = useCallback((index: number) => {
      if (searchResults.length === 0) return;
      const nextIndex = ((index % searchResults.length) + searchResults.length) % searchResults.length;
      const result = searchResults[nextIndex];
      setSearchResultIndex(nextIndex);
      stageSearchPreview(result);
    }, [searchResults, stageSearchPreview]);

    const activateSearchResult = useCallback((index: number, openDetails = false) => {
      if (searchResults.length === 0) return;
      const nextIndex = ((index % searchResults.length) + searchResults.length) % searchResults.length;
      const result = searchResults[nextIndex];
      setSearchResultIndex(nextIndex);
      commitSearchResult(result, openDetails, { autoDetails: true });
    }, [commitSearchResult, searchResults]);

    const activatePreviousSearchResult = useCallback(() => {
      previewSearchResult(searchResultIndex - 1);
    }, [previewSearchResult, searchResultIndex]);

    const activateNextSearchResult = useCallback(() => {
      previewSearchResult(searchResultIndex + 1);
    }, [previewSearchResult, searchResultIndex]);

    const hoveredNode = graphNodes.find(node => node.id === hoveredNodeId) || null;
    const selectedEntityColors = selectedEntity ? getTypeColors(selectedEntity.entity_type, selectedEntity.name) : getTypeColors();
    const selectedDegree = selectedEntity ? (degreeMap.get(selectedEntity.id) || { in: 0, out: 0, total: 0 }) : null;
    const selectedEntityHasRelationships = Boolean(selectedDegree && selectedDegree.total > 0);
    const selectedEvidence = selectedEdgeDetails?.evidenceSnippets[selectedEvidenceIndex] || null;
    const selectedEvidenceIsExcerpt = Boolean(selectedEvidence && isEvidenceExcerpt(selectedEvidence.context));
      const selectedEvidenceDisplayText = selectedEvidence
        ? formatEvidenceSnippetDisplayText(selectedEvidence.context)
        : '';
      const selectedEvidenceMatchesSearch = Boolean(
        selectedEvidenceDisplayText &&
        searchQuery &&
        normalizeGraphSearchResultText(selectedEvidenceDisplayText).includes(searchQuery)
      );
      const selectedEvidenceColor = selectedEvidence
        ? getRelationshipColor(selectedEvidence.type || selectedEdgeDetails?.edge.type || selectedEdgeDetails?.edge.label)
        : selectedEdgeDetails?.dominantColor || accentColors.primary;
      const selectionAccentColor = selectedEntity
        ? selectedEntityColors.main
        : selectedEdgeDetails?.dominantColor || accentColors.primary;
      const responsiveSelectionWidth = (baseWidth: number, viewportRatio: number, maxWidth: number) => {
        const availableWidth = Math.max(baseWidth, selectionViewportWidth - 24);
        return Math.round(Math.min(
          availableWidth,
          clampGraphValue(selectionViewportWidth * viewportRatio, baseWidth, maxWidth)
        ));
      };
      const selectionCardWidth = selectionPanelExpanded
        ? selectedEntity
          ? responsiveSelectionWidth(620, 0.52, 960)
          : responsiveSelectionWidth(640, 0.5, 980)
        : selectedEntity
          ? responsiveSelectionWidth(440, 0.36, 620)
          : responsiveSelectionWidth(460, 0.34, 620);
      const selectionCardLarge = selectionCardWidth >= 540;
      const selectionCardSpacious = selectionCardWidth >= 720;
      const selectionCardHeightEstimate = selectionPanelExpanded
        ? selectedEdgeDetails ? 492 : selectionCardSpacious ? 430 : selectionCardLarge ? 394 : 360
        : selectedEntity
          ? neighborOrbitExpanded
            ? (activeEntityPathPreview ? 382 : selectionCardLarge ? 344 : 318)
            : selectedRelationshipGroups.length > 0 ? (activeEntityPathPreview ? 262 : selectionCardLarge ? 236 : 214) : activeEntityPathPreview ? 188 : 124
          : selectedEdgeDetails
            ? selectedEdgeDetails.evidenceSnippets.length > 0 ? 176 : 146
            : 142;
      const selectionCardMaxHeight = selectionPanelExpanded
        ? selectedEdgeDetails
          ? 'min(calc(100% - 96px), 640px)'
          : 'calc(100% - 138px)'
        : selectedEdgeDetails
          ? selectionViewportHeight >= 760 ? 'min(34vh, 252px)' : 'min(31vh, 206px)'
          : selectedEntity
            ? neighborOrbitExpanded
              ? selectionViewportHeight >= 760 ? 'min(44vh, 388px)' : 'min(40vh, 306px)'
              : selectionViewportHeight >= 760 ? 'min(30vh, 252px)' : 'min(28vh, 218px)'
            : 'min(24vh, 124px)';
      const inspectorOverlayItemLimit = selectionViewportWidth >= 1536
        ? 8
        : selectionViewportWidth >= 1180 ? 6 : 5;
      const inspectorOverlayRelationTypeLimit = selectionViewportWidth >= 1536
        ? 8
        : selectionViewportWidth >= 1180 ? 6 : 4;
      const selectionDockBottom = isFullscreen ? 18 : 58;
      const selectionGraphAvoidRects = useMemo<GraphScreenRect[]>(() => {
        if (!selectionAnchor) return [];
        const camera = new THREE.PerspectiveCamera(45, selectionAnchor.width / Math.max(1, selectionAnchor.height), 1, 12000);
        camera.position.set(graphCameraLod.position.x, graphCameraLod.position.y, graphCameraLod.position.z);
        camera.lookAt(new THREE.Vector3(graphCameraLod.target.x, graphCameraLod.target.y, graphCameraLod.target.z));
        camera.updateMatrixWorld();
        camera.updateProjectionMatrix();

        const projectPosition = (nodeId: string) => {
          const position = graphOverviewTopology.positions.get(nodeId);
          return position ? projectGraphPointToScreen(position, camera, selectionAnchor.width, selectionAnchor.height) : null;
        };
        const nodeRadius = graphCameraLod.zoomTier === 'near'
          ? 132
          : graphCameraLod.zoomTier === 'middle'
            ? 92
            : 58;
        const labelPad = graphCameraLod.zoomTier === 'near' ? 42 : 28;
        const rects: GraphScreenRect[] = [];
        const addPointRect = (point: { x: number; y: number } | null, radius: number) => {
          if (!point) return;
          rects.push({
            left: clampGraphValue(point.x - radius, 0, selectionAnchor.width),
            top: clampGraphValue(point.y - radius * 0.86, 0, selectionAnchor.height),
            right: clampGraphValue(point.x + radius, 0, selectionAnchor.width),
            bottom: clampGraphValue(point.y + radius * 0.86, 0, selectionAnchor.height),
          });
        };

        if (selectedEdgeDetails) {
          const sourcePoint = projectPosition(selectedEdgeDetails.sourceId);
          const targetPoint = projectPosition(selectedEdgeDetails.targetId);
          addPointRect(sourcePoint, nodeRadius);
          addPointRect(targetPoint, nodeRadius);
          if (sourcePoint && targetPoint) {
            const minX = Math.min(sourcePoint.x, targetPoint.x);
            const maxX = Math.max(sourcePoint.x, targetPoint.x);
            const minY = Math.min(sourcePoint.y, targetPoint.y);
            const maxY = Math.max(sourcePoint.y, targetPoint.y);
            const corridorPad = graphCameraLod.zoomTier === 'near' ? 38 : graphCameraLod.zoomTier === 'middle' ? 28 : 18;
            rects.push({
              left: clampGraphValue(minX - corridorPad, 0, selectionAnchor.width),
              top: clampGraphValue(minY - corridorPad, 0, selectionAnchor.height),
              right: clampGraphValue(maxX + corridorPad, 0, selectionAnchor.width),
              bottom: clampGraphValue(maxY + corridorPad, 0, selectionAnchor.height),
            });
          }
          addPointRect({ x: selectionAnchor.x, y: selectionAnchor.y }, Math.max(42, nodeRadius * 0.58));
        } else if (selectedNodeId) {
          addPointRect(projectPosition(selectedNodeId), nodeRadius + labelPad);
        }

        return rects.filter((rect) => rect.right - rect.left > 2 && rect.bottom - rect.top > 2);
      }, [
        graphCameraLod.position.x,
        graphCameraLod.position.y,
        graphCameraLod.position.z,
        graphCameraLod.target.x,
        graphCameraLod.target.y,
        graphCameraLod.target.z,
        graphCameraLod.zoomTier,
        graphOverviewTopology.positions,
        selectedEdgeDetails,
        selectedNodeId,
        selectionAnchor,
      ]);
      const selectionCardPlacement = useMemo(
        () => {
          const selectionKey = selectionAnchor?.key || null;
          if (selectionPlacementKeyRef.current !== selectionKey) {
            selectionPlacementKeyRef.current = selectionKey;
            selectionPlacementEdgeRef.current = null;
          }

          const placement = getSelectionOverlayPlacement(
          selectionAnchor,
          selectionCardWidth,
        selectionCardHeightEstimate,
        selectionPanelExpanded,
        selectionDockBottom,
        selectionGraphAvoidRects,
        selectionPlacementEdgeRef.current
        );

          if (placement.mode === 'anchored' && placement.edge) {
            selectionPlacementEdgeRef.current = placement.edge;
          }

          return placement;
        },
        [selectionAnchor, selectionCardHeightEstimate, selectionCardWidth, selectionDockBottom, selectionGraphAvoidRects, selectionPanelExpanded]
      );
      const selectedEdgePrimaryRelationType = selectedEdgeDetails?.relationTypes[0]?.type ||
        selectedEdgeDetails?.edge.type ||
        selectedEdgeDetails?.edge.label ||
        null;
      const selectionQuickActionCount = selectedEntity
        ? [
          selectedEntityHasRelationships,
          Boolean(activeEntityPathPreview),
          Boolean(selectedEntityGraphRole?.component && selectedComponentId !== selectedEntityGraphRole.component.id),
          true,
          true,
        ].filter(Boolean).length
        : selectedEdgeDetails
          ? [
            Boolean(selectedEdgeDetails.source),
            Boolean(selectedEdgeDetails.target),
            Boolean(selectedEdgePrimaryRelationType),
            true,
            true,
          ].filter(Boolean).length
          : 0;
      const selectionQuickRailWidth = Math.max(36, 8 + selectionQuickActionCount * 28 + Math.max(0, selectionQuickActionCount - 1) * 4);
      const selectionQuickRailPlacement = useMemo(
        () => getSelectionActionRailPlacement(
          selectionAnchor,
          selectionQuickRailWidth,
          36,
          selectionCardPlacement,
          selectionPanelExpanded
        ),
        [selectionAnchor, selectionCardPlacement, selectionPanelExpanded, selectionQuickRailWidth]
      );
      const selectionCardAnchored = selectionCardPlacement.mode === 'anchored';
      const selectionOverlayReservedRects = useMemo<GraphScreenRect[]>(() => {
        if ((!selectedEntity && !selectedEdgeDetails) || !selectionAnchor) return [];

        const cardWidth = Math.min(selectionCardWidth, Math.max(0, selectionAnchor.width - 24));
        const cardHeight = Math.min(selectionCardHeightEstimate, Math.max(0, selectionAnchor.height - 24));
        const cardLeft = selectionCardPlacement.mode === 'anchored' &&
          Number.isFinite(selectionCardPlacement.left)
          ? Number(selectionCardPlacement.left)
          : selectionAnchor.width - cardWidth - 12;
        const cardTop = selectionCardPlacement.mode === 'anchored' &&
          Number.isFinite(selectionCardPlacement.top)
          ? Number(selectionCardPlacement.top)
          : selectionAnchor.height - selectionDockBottom - cardHeight;
        const rects = [
          padGraphRect({
            left: clampGraphValue(cardLeft, 8, Math.max(8, selectionAnchor.width - cardWidth - 8)),
            top: clampGraphValue(cardTop, 8, Math.max(8, selectionAnchor.height - cardHeight - 8)),
            right: clampGraphValue(cardLeft, 8, Math.max(8, selectionAnchor.width - cardWidth - 8)) + cardWidth,
            bottom: clampGraphValue(cardTop, 8, Math.max(8, selectionAnchor.height - cardHeight - 8)) + cardHeight,
          }, 16),
        ];

        if (
          selectionActionsOpen &&
          selectionQuickRailPlacement?.mode === 'anchored' &&
          Number.isFinite(selectionQuickRailPlacement.left) &&
          Number.isFinite(selectionQuickRailPlacement.top)
        ) {
          rects.push(padGraphRect({
            left: Number(selectionQuickRailPlacement.left),
            top: Number(selectionQuickRailPlacement.top),
            right: Number(selectionQuickRailPlacement.left) + selectionQuickRailWidth,
            bottom: Number(selectionQuickRailPlacement.top) + 36,
          }, 12));
        }

        return rects;
      }, [
        selectedEdgeDetails,
        selectedEntity,
        selectionActionsOpen,
        selectionAnchor,
        selectionCardHeightEstimate,
        selectionCardPlacement.left,
        selectionCardPlacement.mode,
        selectionCardPlacement.top,
        selectionCardWidth,
        selectionDockBottom,
        selectionQuickRailPlacement,
        selectionQuickRailWidth,
      ]);
      const searchTrayReservedRects = useMemo<GraphScreenRect[]>(() => {
        if (!searchTrayActive) return [];

        const viewportWidth = selectionViewportWidth;
        const viewportHeight = selectionViewportHeight;
        const trayWidth = viewportWidth < 900
          ? Math.max(38, viewportWidth - 24)
          : viewportWidth < 1200
            ? searchQuery
            ? searchTrayExpanded ? 440 : 340
              : 320
            : viewportWidth < 1536
              ? searchQuery
                ? searchTrayExpanded ? Math.min(520, Math.round(viewportWidth * 0.36)) : 360
                : 340
              : searchQuery
                ? searchTrayExpanded ? Math.min(560, Math.round(viewportWidth * 0.32)) : 420
                : 380;
        const visibleResultRows = searchQuery
          ? searchTrayExpanded
            ? Math.min(viewportWidth >= 1600 ? 5 : viewportWidth < 1300 ? 3 : 4, Math.max(1, searchResults.length))
            : selectedSearchResult ? 1 : 0
          : 0;
        const searchResultColumns = searchTrayExpanded
          ? viewportWidth >= 1300 ? 2 : 1
          : 1;
        const reservedResultRows = searchResultColumns > 1 && visibleResultRows > 1
          ? 1 + Math.ceil((visibleResultRows - 1) / searchResultColumns)
          : visibleResultRows;
        const trayHeight = searchQuery
          ? searchTrayExpanded
            ? 140 + reservedResultRows * (viewportWidth >= 1300 ? 62 : 54)
            : selectedSearchResult ? 166 : 104
          : 58;

        return [padGraphRect({
          left: 12,
          top: 12,
          right: Math.min(viewportWidth - 12, 12 + trayWidth),
          bottom: Math.min(viewportHeight - 12, 12 + trayHeight),
        }, 18)];
      }, [
        searchQuery,
        searchResults.length,
        searchTrayActive,
        searchTrayExpanded,
        selectedSearchResult,
        selectionViewportHeight,
        selectionViewportWidth,
      ]);
      const toolsOverlayReservedRects = useMemo<GraphScreenRect[]>(() => {
        if (!graphToolsOpen) return [];

        const viewportWidth = selectionViewportWidth;
        const viewportHeight = selectionViewportHeight;
        const panelWidth = viewportWidth < 700 ? Math.max(38, viewportWidth - 24) : 394;
        const panelHeight = 246;
        const top = viewportWidth < 900 && searchTrayActive ? (searchQuery ? 112 : 62) : 12;
        return [padGraphRect({
          left: Math.max(12, viewportWidth - panelWidth - 12),
          top,
          right: viewportWidth - 12,
          bottom: Math.min(viewportHeight - 12, top + panelHeight),
        }, 16)];
      }, [
        graphToolsOpen,
        searchQuery,
        searchTrayActive,
        selectionViewportHeight,
        selectionViewportWidth,
      ]);
      const panelOverlayReservedRects = useMemo<GraphScreenRect[]>(() => {
        const viewportWidth = selectionViewportWidth;
        const viewportHeight = selectionViewportHeight;
        const rects: GraphScreenRect[] = [];
        const topAfterSearch = viewportWidth < 900 && searchTrayActive ? (searchQuery ? 112 : 62) : 12;
        const reserveRightPanel = (panelWidth: number, panelHeight: number, top: number, padding = 16) => {
          const width = Math.min(Math.max(38, panelWidth), Math.max(38, viewportWidth - 24));
          const left = Math.max(12, viewportWidth - width - 12);
          rects.push(padGraphRect({
            left,
            top: clampGraphValue(top, 12, Math.max(12, viewportHeight - 72)),
            right: viewportWidth - 12,
            bottom: Math.min(viewportHeight - 12, top + panelHeight),
          }, padding));
        };

        if (settingsOpen) {
          const panelWidth = viewportWidth < 600
            ? viewportWidth - 24
            : viewportWidth < 1200
              ? 380
              : viewportWidth < 1536
                ? 440
                : 480;
          const panelTop = viewportWidth < 900 ? 58 : 52;
          const panelHeight = Math.min(
            viewportHeight - panelTop - 12,
            viewportWidth < 900 ? Math.min(viewportHeight * 0.62, 520) : Math.min(viewportHeight * 0.52, 560)
          );
          reserveRightPanel(panelWidth, panelHeight, panelTop, 18);
        }

        if (graphLegendOpen) {
          const panelWidth = viewportWidth < 600
            ? viewportWidth - 42
            : viewportWidth < 1200
              ? 430
              : viewportWidth < 1536
                ? Math.min(viewportWidth * 0.48, 560)
                : 600;
          const panelHeight = Math.min(
            viewportHeight - topAfterSearch - 12,
            viewportWidth < 700 ? Math.min(viewportHeight * 0.52, 430) : Math.min(viewportHeight * 0.62, 560)
          );
          reserveRightPanel(panelWidth, panelHeight, topAfterSearch, 18);
        }

        if (graphOverviewOpen) {
          reserveRightPanel(
            viewportWidth < 700 ? Math.min(252, viewportWidth - 24) : viewportWidth < 1536 ? Math.min(viewportWidth * 0.3, 340) : 360,
            graphOverviewHover ? 226 : 182,
            topAfterSearch,
            16
          );
        }

        if (inspectorOpen && graphNodes.length > 0) {
          const inspectorTop = viewportWidth < 900 && searchTrayActive ? (searchQuery ? 164 : 112) : 56;
          reserveRightPanel(viewportWidth < 600 ? viewportWidth - 24 : viewportWidth < 1200 ? 320 : 380, 360, inspectorTop, 18);
        }

        if (graphStatsOpen) {
          const statsWidth = Math.min(viewportWidth < 700 ? viewportWidth - 86 : 384, viewportWidth - 24);
          const statsHeight = graphStatsDetailsOpen ? 500 : 190;
          rects.push(padGraphRect({
            left: 12,
            top: Math.max(12, viewportHeight - 12 - statsHeight - 42),
            right: Math.min(viewportWidth - 12, 12 + statsWidth),
            bottom: viewportHeight - 12,
          }, 16));
        }

        return rects.filter((rect) => rect.right - rect.left > 2 && rect.bottom - rect.top > 2);
      }, [
        graphLegendOpen,
        graphNodes.length,
        graphOverviewHover,
        graphOverviewOpen,
        graphStatsDetailsOpen,
        graphStatsOpen,
        inspectorOpen,
        searchQuery,
        searchTrayActive,
        selectionViewportHeight,
        selectionViewportWidth,
        settingsOpen,
      ]);
      const graphOverlayReservedRects = useMemo<GraphScreenRect[]>(() => [
        ...selectionOverlayReservedRects,
        ...searchTrayReservedRects,
        ...toolsOverlayReservedRects,
        ...panelOverlayReservedRects,
      ], [panelOverlayReservedRects, searchTrayReservedRects, selectionOverlayReservedRects, toolsOverlayReservedRects]);
    const selectedBrowseEntry = useMemo<GraphBrowseEntry | null>(() => {
      if (selectedEntity) {
        return {
          kind: 'entity',
          id: selectedEntity.id,
          label: selectedEntity.name,
          color: selectedEntityColors.main,
        };
      }

      if (selectedEdgeDetails) {
        return {
          kind: 'edge',
          id: selectedEdgeDetails.edge.id,
          label: `${selectedEdgeDetails.source?.name || selectedEdgeDetails.sourceId} -> ${selectedEdgeDetails.target?.name || selectedEdgeDetails.targetId}`,
          color: selectedEdgeDetails.dominantColor,
        };
      }

      return null;
    }, [selectedEdgeDetails, selectedEntity, selectedEntityColors.main]);

    const restoreBrowseEntry = useCallback((entry: GraphBrowseEntry) => {
      if (entry.kind === 'entity') {
        const entity = entities.find((item) => item.id === entry.id);
        if (!entity) return false;

        setSelectedEdgeId(null);
        setSelectedEntity(entity);
        setSelectedNodeId(entity.id);
        setExpandedRelationshipBundleId(null);
        setInspectorOpen(false);
        setTabValue(2);
        return true;
      }

      const edge = visualGraph.edges.find((item) => item.id === entry.id);
      if (!edge) return false;

      setSelectedEdgeId(edge.id);
      setSelectedNodeId(null);
      setSelectedEntity(null);
      setExpandedRelationshipBundleId(null);
      setInspectorOpen(false);
      setTabValue(2);
      return true;
    }, [entities, visualGraph.edges]);

    useEffect(() => {
      if (!selectedBrowseEntry) return;

      setBrowseHistory((current) => {
        const currentEntry = browseHistoryIndex >= 0 ? current[browseHistoryIndex] : null;
        if (isSameGraphBrowseEntry(currentEntry, selectedBrowseEntry)) return current;

        const base = browseHistoryIndex >= 0 ? current.slice(0, browseHistoryIndex + 1) : current;
        const lastEntry = base[base.length - 1];

        if (isSameGraphBrowseEntry(lastEntry, selectedBrowseEntry)) {
          setBrowseHistoryIndex(Math.max(0, base.length - 1));
          return base.length === current.length ? current : base;
        }

        const next = [...base, selectedBrowseEntry].slice(-24);
        setBrowseHistoryIndex(next.length - 1);
        return next;
      });
    }, [browseHistoryIndex, selectedBrowseEntry]);

    const canBrowseBack = browseHistoryIndex > 0;
    const canBrowseForward = browseHistoryIndex >= 0 && browseHistoryIndex < browseHistory.length - 1;
    const selectedEntityNeighborhoodFocused = Boolean(
      selectedEntity && focusNeighborhood && selectedNodeId === selectedEntity.id
    );

    const navigateBrowseHistory = useCallback((direction: -1 | 1) => {
      const nextIndex = browseHistoryIndex + direction;
      const entry = browseHistory[nextIndex];
      if (!entry) return;
      if (!restoreBrowseEntry(entry)) return;
      setBrowseHistoryIndex(nextIndex);
    }, [browseHistory, browseHistoryIndex, restoreBrowseEntry]);

    const toggleSelectedEntityNeighborhood = useCallback(() => {
      if (!selectedEntity) return;
      if (selectedEntityNeighborhoodFocused) {
        setFocusNeighborhood(false);
        return;
      }

      setSelectedNodeId(selectedEntity.id);
      setSelectedEdgeId(null);
      setSelectedComponentId(null);
      setFocusNeighborhood(true);
      setGraphScope('linked');
      setInspectorOpen(false);
    }, [selectedEntity, selectedEntityNeighborhoodFocused]);

    const entityTypeCount = entityTypeSummary.length;
    const graphEntityCount = (stats as Record<string, number> | null)?.entity_count ?? graphNodes.length;
    const graphRelationshipCount =
      (stats as Record<string, number> | null)?.relationship_count ??
      (stats as Record<string, number> | null)?.edges ??
      graphEdges.length;
  const graphCommunityCount = (stats as Record<string, number> | null)?.community_count ?? 0;
  const activeFocusLabel = selectedEntity?.name ||
    (selectedEdgeDetails
      ? `${selectedEdgeDetails.source?.name || selectedEdgeDetails.sourceId} -> ${selectedEdgeDetails.target?.name || selectedEdgeDetails.targetId}`
      : hoveredNode?.label || 'No entity selected');
  const hasActiveFocus = Boolean(selectedEntity || selectedEdgeDetails || hoveredNode);
  const graphStatusLabel = graphRAGStatus.available
    ? 'GraphRAG ready'
    : graphRAGStatus.status === 'checking'
      ? 'Checking service'
      : 'GraphRAG offline';
  const graphStatusColor = graphRAGStatus.available
    ? accentColors.success
    : graphRAGStatus.status === 'checking'
      ? accentColors.warning
      : accentColors.rose;
  const graphLabelModeValue: GraphLabelModeValue = showLabels ? graphLabelDensity : 'hidden';
  const graphLabelModeLabel = graphLabelModeValue === 'hidden'
    ? 'Hidden'
    : graphLabelModeValue === 'focus'
      ? 'Focus'
      : graphLabelModeValue === 'rich'
        ? 'Rich'
        : 'Priority';
  const graphLabelModeSummary = graphLabelModeValue === 'hidden'
    ? 'names hidden'
    : graphLabelModeValue === 'focus'
      ? 'focus names'
      : graphLabelModeValue === 'rich'
        ? 'rich names'
        : 'priority names';
  const setGraphLabelMode = useCallback((mode: GraphLabelModeValue) => {
    if (mode === 'hidden') {
      setShowLabels(false);
      return;
    }
    setShowLabels(true);
    setGraphLabelDensity(mode);
  }, []);
  const cycleGraphLabelMode = useCallback(() => {
    const modes: GraphLabelModeValue[] = ['rich', 'priority', 'focus'];
    const currentIndex = modes.indexOf(graphLabelModeValue);
    const nextIndex = currentIndex < 0 ? 0 : (currentIndex + 1) % modes.length;
    setGraphLabelMode(modes[nextIndex]);
  }, [graphLabelModeValue, setGraphLabelMode]);
  const graphIsEmpty = graphNodes.length === 0;
  const graphUnavailable = !graphRAGStatus.available && graphRAGStatus.status !== 'checking';
  const useGraphCanvasLayout = isFullscreen || loading || !graphIsEmpty;
  const inspectorOverlayOpen = inspectorOpen && !graphIsEmpty;
  const selectionOverlayOpen = Boolean(selectedEntity || selectedEdgeDetails);
  const graphOverlayInteractionActive = Boolean(
    searchTrayActive ||
    graphToolsOpen ||
    graphScopeOpen ||
    graphOverviewOpen ||
    graphLegendOpen ||
    settingsOpen ||
    inspectorOverlayOpen ||
    graphStatsOpen ||
    graphStatsDetailsOpen ||
      (selectionOverlayOpen && selectionActionsOpen)
  );
  const showInspector = false;
  const showGraphSummaryRow = graphIsEmpty && (graphRAGStatus.available || Boolean(selectedDomain) || hasActiveFocus);
  const inspectorCountLabel = tabValue === 1
    ? `${formatNumber(relationBrowserRows.length)} links`
    : `${formatNumber(filteredEntities.length)} entities`;
    const graphScopeLabel = searchQuery
      ? 'Search results'
      : selectedComponent
        ? selectedComponent.label
        : graphScope === 'core'
        ? 'Core component'
        : graphScope === 'linked'
          ? 'Linked entities'
          : 'All entities';
    const graphContextLabel = [
      !graphIsEmpty ? graphScopeLabel : null,
      selectedDomain ? `Domain: ${selectedDomain}` : !graphIsEmpty ? 'All domains' : null,
      hasActiveFocus ? activeFocusLabel : null,
    ].filter(Boolean).join(' / ');
  const graphRenderProfilePreview = useMemo(
    () => applyGraphDetailMode(getGraphRenderProfile(visualGraph.nodes.length, visualGraph.edges.length), graphDetailMode),
    [graphDetailMode, visualGraph.edges.length, visualGraph.nodes.length]
  );
  const graphOverview = useMemo(() => {
    const rawNodes = visualGraph.nodes
      .map((node) => {
        const position = graphOverviewTopology.positions.get(node.id);
        if (!position) return null;
        return { node, x: position.x, y: position.y, z: position.z };
      })
      .filter((item): item is { node: GraphNode; x: number; y: number; z: number } => Boolean(item));

    if (rawNodes.length === 0) {
      return {
        edges: [] as Array<{ edge: GraphEdge; source: string; target: string; color: string }>,
        nodes: [] as Array<{ node: GraphNode; x: number; y: number; r: number; color: string; selected: boolean; focused: boolean }>,
        focusBounds: null as { x: number; y: number; width: number; height: number; nodeCount: number } | null,
        cameraTarget: null as { x: number; y: number; r: number; zoomTier: GraphZoomTier; moving: boolean } | null,
        projection: null as {
          bounds: { minX: number; maxX: number; minY: number; maxY: number };
          height: number;
          offsetX: number;
          offsetY: number;
          scale: number;
          width: number;
          z: number;
        } | null,
        hiddenNodeCount: 0,
      };
    }

    const bounds = rawNodes.reduce(
      (acc, item) => ({
        minX: Math.min(acc.minX, item.x),
        maxX: Math.max(acc.maxX, item.x),
        minY: Math.min(acc.minY, item.y),
        maxY: Math.max(acc.maxY, item.y),
      }),
      { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity }
    );
    const width = 208;
    const height = 132;
    const padding = 13;
    const spanX = Math.max(1, bounds.maxX - bounds.minX);
    const spanY = Math.max(1, bounds.maxY - bounds.minY);
    const scale = Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY);
    const offsetX = (width - spanX * scale) / 2;
    const offsetY = (height - spanY * scale) / 2;
    const nodeLimit = graphRenderProfilePreview.tier === 'massive'
      ? 180
      : graphRenderProfilePreview.tier === 'large'
        ? 240
        : 360;
    const rankedNodeIds = [...visualGraph.nodes]
      .sort((a, b) =>
        getNodeImportanceScore(b, graphOverviewTopology.metrics.get(b.id)) -
        getNodeImportanceScore(a, graphOverviewTopology.metrics.get(a.id)) ||
        a.label.localeCompare(b.label)
      )
      .slice(0, nodeLimit)
      .map((node) => node.id);
    const visibleNodeIds = new Set(rankedNodeIds);
    if (selectedNodeId) visibleNodeIds.add(selectedNodeId);
    if (selectedEdgeDetails) {
      visibleNodeIds.add(selectedEdgeDetails.sourceId);
      visibleNodeIds.add(selectedEdgeDetails.targetId);
    }
    activeGraphFocus.activeNodes.forEach((id) => visibleNodeIds.add(id));

    const overviewNodes = rawNodes
      .filter(({ node }) => visibleNodeIds.has(node.id))
      .map(({ node, x, y }) => {
        const summary = graphOverviewTopology.metrics.get(node.id);
        const color = getTypeColors(node.type, node.label).main;
        const selected = node.id === selectedNodeId;
        const focused = activeGraphFocus.activeNodes.has(node.id);
        return {
          node,
          x: offsetX + (x - bounds.minX) * scale,
          y: height - (offsetY + (y - bounds.minY) * scale),
          r: selected ? 4.7 : focused ? 3.8 : Math.max(1.6, Math.min(3.4, 1.6 + Math.log2((summary?.relations || 0) + 1) * 0.38)),
          color,
          selected,
          focused,
        };
      });
    const focusedOverviewNodes = overviewNodes.filter(({ selected, focused }) => selected || focused);
    const focusBounds = focusedOverviewNodes.length > 0
      ? (() => {
        const bounds = focusedOverviewNodes.reduce(
          (acc, item) => ({
            minX: Math.min(acc.minX, item.x),
            maxX: Math.max(acc.maxX, item.x),
            minY: Math.min(acc.minY, item.y),
            maxY: Math.max(acc.maxY, item.y),
          }),
          {
            minX: focusedOverviewNodes[0].x,
            maxX: focusedOverviewNodes[0].x,
            minY: focusedOverviewNodes[0].y,
            maxY: focusedOverviewNodes[0].y,
          }
        );
        const focusPadding = focusedOverviewNodes.length === 1 ? 10 : 8;
        const focusX = Math.max(5, bounds.minX - focusPadding);
        const focusY = Math.max(5, bounds.minY - focusPadding);
        const focusRight = Math.min(width - 5, bounds.maxX + focusPadding);
        const focusBottom = Math.min(height - 5, bounds.maxY + focusPadding);
        return {
          x: focusX,
          y: focusY,
          width: Math.max(16, focusRight - focusX),
          height: Math.max(16, focusBottom - focusY),
          nodeCount: focusedOverviewNodes.length,
        };
      })()
      : null;
    const cameraTarget = (() => {
      const targetX = Number(graphCameraLod.target.x);
      const targetY = Number(graphCameraLod.target.y);
      if (!Number.isFinite(targetX) || !Number.isFinite(targetY)) return null;

      const mappedX = offsetX + (targetX - bounds.minX) * scale;
      const mappedY = height - (offsetY + (targetY - bounds.minY) * scale);
      const radius = graphCameraLod.zoomTier === 'near'
        ? 9.5
        : graphCameraLod.zoomTier === 'middle'
          ? 15
          : 22;
      return {
        x: clampGraphValue(mappedX, 7, width - 7),
        y: clampGraphValue(mappedY, 7, height - 7),
        r: radius,
        zoomTier: graphCameraLod.zoomTier,
        moving: graphCameraLod.moving,
      };
    })();
    const overviewNodeSet = new Set(overviewNodes.map(({ node }) => node.id));
    const overviewEdges = [...visualGraph.edges]
      .filter((edge) => overviewNodeSet.has(edge.source) && overviewNodeSet.has(edge.target))
      .sort((a, b) =>
        Number(b.relationCount || 1) - Number(a.relationCount || 1) ||
        Number(b.weight || 0) - Number(a.weight || 0)
      )
      .slice(0, graphRenderProfilePreview.tier === 'massive' ? 220 : 340)
      .map((edge) => ({
        edge,
        source: edge.source,
        target: edge.target,
        color: getRelationshipColor(edge.type || edge.label),
      }));

    return {
      edges: overviewEdges,
      nodes: overviewNodes,
      focusBounds,
      cameraTarget,
      projection: {
        bounds,
        height,
        offsetX,
        offsetY,
        scale,
        width,
        z: Number.isFinite(Number(graphCameraLod.target.z)) ? Number(graphCameraLod.target.z) : 0,
      },
      hiddenNodeCount: Math.max(0, visualGraph.nodes.length - overviewNodes.length),
    };
  }, [activeGraphFocus.activeNodes, graphCameraLod, graphOverviewTopology, graphRenderProfilePreview.tier, selectedEdgeDetails, selectedNodeId, visualGraph.edges, visualGraph.nodes]);

  const getGraphPointFromOverview = useCallback((clientX: number, clientY: number, bounds: DOMRect) => {
    const projection = graphOverview.projection;
    if (!projection) return null;

    const overviewX = ((clientX - bounds.left) / Math.max(1, bounds.width)) * projection.width;
    const overviewY = ((clientY - bounds.top) / Math.max(1, bounds.height)) * projection.height;
    const graphX = projection.bounds.minX + (overviewX - projection.offsetX) / projection.scale;
    const graphY = projection.bounds.minY + ((projection.height - overviewY) - projection.offsetY) / projection.scale;

    if (!Number.isFinite(graphX) || !Number.isFinite(graphY)) return null;
    return {
      x: graphX,
      y: graphY,
      z: projection.z,
    };
  }, [graphOverview.projection]);

  const focusGraphOverviewAt = useCallback((clientX: number, clientY: number, bounds: DOMRect) => {
    const target = getGraphPointFromOverview(clientX, clientY, bounds);
    if (!target) return false;
    const focused = focusGraphCameraAt(target);
    if (focused) setGraphOverviewHover(null);
    return focused;
  }, [focusGraphCameraAt, getGraphPointFromOverview]);

  const focusGraphOverviewNode = useCallback((nodeId: string) => {
    return focusGraphNodeInTopology(nodeId);
  }, [focusGraphNodeInTopology]);

  const focusGraphOverviewEdge = useCallback((edge: GraphEdge) => {
    return focusGraphEdgeInTopology(edge);
  }, [focusGraphEdgeInTopology]);

  const graphNodeBudgetLabel = (['far', 'middle', 'near'] as GraphZoomTier[])
    .map((tier) => graphRenderProfilePreview.nodeBudget[tier])
    .every(Number.isFinite)
    ? `${formatNumber(graphRenderProfilePreview.nodeBudget.far)} / ${formatNumber(graphRenderProfilePreview.nodeBudget.middle)} / ${formatNumber(graphRenderProfilePreview.nodeBudget.near)} node LOD`
    : 'Full node detail';
  const graphLinkBudgetLabel = (['far', 'middle', 'near'] as GraphZoomTier[])
    .map((tier) => graphRenderProfilePreview.linkBudget[tier])
    .every(Number.isFinite)
    ? `${formatNumber(graphRenderProfilePreview.linkBudget.far)} / ${formatNumber(graphRenderProfilePreview.linkBudget.middle)} / ${formatNumber(graphRenderProfilePreview.linkBudget.near)} link LOD`
    : 'Full link detail';
  const graphCanvasKey = [
    graphScope,
    selectedDomain || 'all-domains',
    selectedEntityTypes.join(',') || 'all-entity-types',
    selectedRelationshipTypes.join(',') || 'all-relation-types',
    selectedComponentId || 'all-components',
    focusNeighborhood ? 'neighborhood' : 'global',
    visualGraph.nodes.length,
    visualGraph.edges.length,
  ].join('|');
  const graphSelectedNodeId = selectedNodeId;
  const graphSelectedEdgeId = selectedEdgeId;
  const commandIconButtonSx = {
    width: 32,
    height: 32,
    borderRadius: '8px',
    border: '1px solid',
    borderColor: 'divider',
    bgcolor: 'action.hover',
    '&:hover': { bgcolor: 'action.selected' },
    '&.Mui-disabled': {
      bgcolor: 'action.disabledBackground',
      color: 'action.disabled',
    },
  };
  const graphOverlayButtonSx = {
    ...commandIconButtonSx,
    width: 30,
    height: 30,
    color: 'text.secondary',
    bgcolor: alpha('#070a12', 0.72),
    borderColor: alpha('#fff', 0.12),
    backdropFilter: 'blur(14px)',
    boxShadow: `0 8px 24px ${alpha('#000', 0.18)}`,
    transition: 'transform 160ms ease, background-color 160ms ease, border-color 160ms ease, color 160ms ease',
    '&:hover': {
      transform: 'translateY(-1px)',
      bgcolor: alpha('#111827', 0.92),
      borderColor: alpha(accentColors.primary, 0.42),
      color: 'text.primary',
    },
    '&:active': {
      transform: 'translateY(0) scale(0.96)',
    },
  };
  const selectionFooterActionSx = (color = accentColors.primary, active = false) => ({
    width: 28,
    height: 28,
    borderRadius: '8px',
    border: '1px solid',
    borderColor: active ? alpha(color, 0.34) : alpha('#fff', 0.1),
    color: active ? color : 'text.secondary',
    bgcolor: active ? alpha(color, 0.13) : alpha('#020617', 0.42),
    transition: 'transform 150ms ease, background-color 150ms ease, border-color 150ms ease, color 150ms ease',
    '&:hover': {
      transform: 'translateY(-1px)',
      borderColor: alpha(color, 0.38),
      color,
      bgcolor: alpha(color, 0.1),
    },
  });
  const graphMenuActionSx = (active = false, color = accentColors.primary) => ({
    minHeight: { xs: 34, md: 42 },
    width: { xs: 34, md: 164, xl: 178 },
    height: { xs: 34, md: 42 },
    minWidth: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: { xs: 'center', md: 'flex-start' },
    gap: { xs: 0, md: 0.8 },
    px: { xs: 0, md: 0.9 },
    py: 0,
    borderRadius: '8px',
    border: '1px solid',
    borderColor: active ? alpha(color, 0.44) : alpha('#fff', 0.08),
    bgcolor: active ? alpha(color, 0.13) : alpha('#020617', 0.34),
    color: active ? color : 'text.secondary',
    textAlign: 'left',
    transition: 'background-color 150ms ease, border-color 150ms ease, color 150ms ease, transform 150ms ease',
    '&:hover': {
      transform: 'translateY(-1px)',
      borderColor: alpha(color, 0.42),
      bgcolor: alpha(color, active ? 0.17 : 0.09),
      color: active ? color : 'text.primary',
    },
    '&.Mui-disabled': {
      color: 'text.disabled',
      borderColor: alpha('#fff', 0.05),
      bgcolor: alpha('#020617', 0.18),
    },
    '& > .MuiBox-root': {
      display: { xs: 'none', md: 'block' },
      flex: 1,
      minWidth: 0,
    },
  });
  const graphOverlayPanelSx = {
      bgcolor: alpha('#050816', 0.58),
      border: '1px solid',
      borderColor: alpha('#fff', 0.1),
      boxShadow: `0 12px 34px ${alpha('#000', 0.22)}`,
      backdropFilter: 'blur(18px)',
      borderRadius: '8px',
      animation: 'kgOverlayIn 180ms ease-out',
      transition: 'background-color 160ms ease, border-color 160ms ease, box-shadow 160ms ease',
      '&:hover, &:focus-within': {
        bgcolor: alpha('#050816', 0.86),
        borderColor: alpha('#fff', 0.16),
        boxShadow: `0 14px 38px ${alpha('#000', 0.28)}`,
      },
    };
    const graphOverlayScrollSx = {
      overscrollBehavior: 'contain',
      scrollbarWidth: 'thin',
      scrollbarColor: `${alpha(accentColors.primary, 0.42)} ${alpha('#020617', 0.35)}`,
      '&::-webkit-scrollbar': {
        width: 8,
        height: 8,
      },
      '&::-webkit-scrollbar-track': {
        bgcolor: alpha('#020617', 0.26),
        borderRadius: '999px',
      },
      '&::-webkit-scrollbar-thumb': {
        bgcolor: alpha(accentColors.primary, 0.32),
        borderRadius: '999px',
        border: `2px solid ${alpha('#020617', 0.38)}`,
      },
      '&::-webkit-scrollbar-thumb:hover': {
        bgcolor: alpha(accentColors.primary, 0.5),
      },
    };
  const renderBrowseHistoryControls = (color: string) => {
    if (!canBrowseBack && !canBrowseForward) return null;
    return (
      <Box
        data-kg-browse-nav="true"
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 0.25,
          p: 0.25,
          borderRadius: '8px',
          bgcolor: alpha('#020617', 0.52),
          border: `1px solid ${alpha(color, 0.16)}`,
          flexShrink: 0,
        }}
      >
        <Tooltip title={browseHistory[browseHistoryIndex - 1]?.label || 'Back'}>
          <span>
            <IconButton
              size="small"
              data-kg-browse-back="true"
              aria-label="Back through graph selection history"
              disabled={!canBrowseBack}
              onClick={() => navigateBrowseHistory(-1)}
              sx={{
                width: 24,
                height: 24,
                borderRadius: '7px',
                color: canBrowseBack ? color : 'text.disabled',
              }}
            >
              <BrowseBackIcon sx={{ fontSize: 17 }} />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title={browseHistory[browseHistoryIndex + 1]?.label || 'Forward'}>
          <span>
            <IconButton
              size="small"
              data-kg-browse-forward="true"
              aria-label="Forward through graph selection history"
              disabled={!canBrowseForward}
              onClick={() => navigateBrowseHistory(1)}
              sx={{
                width: 24,
                height: 24,
                borderRadius: '7px',
                color: canBrowseForward ? color : 'text.disabled',
              }}
            >
              <CollapseBundleIcon sx={{ fontSize: 17 }} />
            </IconButton>
          </span>
        </Tooltip>
      </Box>
    );
  };
  const graphSettingsSectionHeaderSx = (active: boolean) => ({
    minHeight: 34,
    px: 1,
    py: 0.65,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 1,
    cursor: 'pointer',
    color: active ? 'text.primary' : 'text.secondary',
    bgcolor: active ? alpha(accentColors.primary, 0.1) : 'transparent',
    transition: 'background-color 160ms ease, color 160ms ease',
    '&:hover': {
      bgcolor: active ? alpha(accentColors.primary, 0.13) : alpha('#fff', 0.045),
      color: 'text.primary',
    },
  });
  const graphSettingsSectionSx = {
    overflow: 'hidden',
    borderRadius: '8px',
    border: '1px solid',
    borderColor: alpha('#fff', 0.1),
    bgcolor: alpha('#020617', 0.42),
  };
  const toggleGraphSettings = useCallback(() => {
    setSettingsOpen((open) => {
      const next = !open;
      if (next) {
        setSettingsSection(null);
        setGraphScopeOpen(false);
        setGraphOverviewOpen(false);
        setGraphToolsOpen(false);
        setGraphLegendOpen(false);
        setGraphStatsOpen(false);
        setGraphStatsDetailsOpen(false);
        setSelectionActionsOpen(false);
      }
      return next;
    });
  }, []);
  const toggleSettingsSection = useCallback((section: GraphSettingsSection) => {
    setSettingsSection((current) => current === section ? null : section);
  }, []);
      const renderSettingsDisclosure = (
        section: GraphSettingsSection,
        title: string,
        summary: string,
        children: ReactNode
  ) => {
    const expanded = settingsSection === section;
    return (
      <Box sx={graphSettingsSectionSx}>
        <Box
          role="button"
          tabIndex={0}
          onClick={() => toggleSettingsSection(section)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault();
              toggleSettingsSection(section);
            }
          }}
          sx={graphSettingsSectionHeaderSx(expanded)}
        >
          <Box sx={{ minWidth: 0 }}>
            <Typography variant="caption" sx={{ display: 'block', fontWeight: 900, lineHeight: 1.1 }}>
              {title}
            </Typography>
            <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', lineHeight: 1.2 }}>
              {summary}
            </Typography>
          </Box>
          {expanded ? <PreviousResultIcon sx={{ fontSize: 18, flexShrink: 0 }} /> : <ExpandBundleIcon sx={{ fontSize: 18, flexShrink: 0 }} />}
        </Box>
        <Collapse in={expanded} unmountOnExit>
          <Box sx={{ px: 1, pt: 0.75, pb: 1 }}>
            {children}
          </Box>
        </Collapse>
          </Box>
        );
      };
      const renderMetadataFactStrip = (
        facts: GraphMetadataFact[],
        kind: 'entity' | 'relation'
      ) => {
        if (facts.length === 0) return null;

        return (
          <Box
            data-kg-metadata-facts={kind}
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.4,
              flexWrap: 'wrap',
              minWidth: 0,
              mb: 0.45,
            }}
          >
            {facts.map((fact) => (
              <Chip
                key={fact.id}
                size="small"
                icon={fact.icon}
                label={`${fact.label}: ${fact.value}`}
                variant="outlined"
                sx={{
                  height: 20,
                  maxWidth: '100%',
                  color: fact.color,
                  bgcolor: alpha(fact.color, 0.085),
                  borderColor: alpha(fact.color, 0.28),
                  '& .MuiChip-label': {
                    px: 0.58,
                    fontSize: '0.58rem',
                    fontWeight: 850,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  },
                  '& .MuiChip-icon': { ml: 0.5, mr: -0.2, color: 'inherit' },
                }}
              />
            ))}
          </Box>
        );
      };
      const renderCompactMetadataFactStrip = (
        facts: GraphMetadataFact[],
        kind: 'entity' | 'relation',
        hiddenFactIds: string[] = []
      ) => {
        if (facts.length === 0) return null;
        const wideCollapsedCard = selectionCardWidth >= 390;
        const visibleFacts = facts.filter((fact) => !hiddenFactIds.includes(fact.id));
        const compactFacts = (visibleFacts.length > 0 ? visibleFacts : facts)
          .slice(0, wideCollapsedCard ? 3 : 2);
        if (compactFacts.length === 0) return null;

        return (
          <Box
            data-kg-selection-compact-metadata={kind}
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.42,
              flexWrap: wideCollapsedCard ? 'wrap' : 'nowrap',
              minWidth: 0,
              overflow: 'hidden',
              mt: -0.1,
              mb: 0.46,
            }}
          >
            {compactFacts.map((fact) => (
              <Chip
                key={`${kind}-compact-fact-${fact.id}`}
                size="small"
                icon={fact.icon}
                label={`${fact.label}: ${fact.value}`}
                variant="outlined"
                sx={{
                  height: 20,
                  maxWidth: wideCollapsedCard ? 158 : 126,
                  color: fact.color,
                  bgcolor: alpha(fact.color, 0.08),
                  borderColor: alpha(fact.color, 0.25),
                  '& .MuiChip-label': {
                    px: 0.58,
                    fontSize: '0.57rem',
                    fontWeight: 850,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  },
                  '& .MuiChip-icon': { ml: 0.5, mr: -0.22, color: 'inherit' },
                }}
              />
            ))}
          </Box>
        );
      };
      const renderSelectionMetadataBlock = (
        entries: MetadataEntry[],
        color: string,
      kind: 'entity' | 'relation'
    ) => {
      if (entries.length === 0) return null;
      const visibleLimit = kind === 'relation' ? 9 : 8;
      const groupedEntries = groupMetadataEntries(entries, color, visibleLimit);
      const visibleEntryCount = groupedEntries.reduce((sum, group) => sum + group.entries.length, 0);
      const hiddenCount = Math.max(0, entries.length - visibleEntryCount);

      return (
        <Box
          data-kg-selection-metadata={kind}
          sx={{
            mb: 0.85,
            p: 0.75,
            borderRadius: '8px',
            border: '1px solid',
            borderColor: alpha(color, 0.18),
            bgcolor: alpha('#020617', 0.26),
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.55 }}>
            <Typography variant="caption" color="text.secondary" sx={{ flex: 1, minWidth: 0, fontWeight: 900 }} noWrap>
              Metadata
            </Typography>
            <Chip
              size="small"
              label={hiddenCount > 0
                ? `${visibleEntryCount}/${entries.length}`
                : `${entries.length} ${entries.length === 1 ? 'field' : 'fields'}`}
              sx={{
                height: 18,
                color,
                bgcolor: alpha(color, 0.12),
                '& .MuiChip-label': { px: 0.55, fontSize: '0.58rem', fontWeight: 900 },
              }}
            />
          </Box>
          <Box sx={{ display: 'grid', gap: 0.42 }}>
            {groupedEntries.map((group) => {
              const groupKey = `${kind}:${group.id}`;
              const expanded = expandedSelectionMetadataGroups[groupKey] ?? false;
              const previewItems = getMetadataPreviewItems(group.entries, kind === 'relation' ? 2 : 3);
              return (
                <Box
                  key={`${kind}-metadata-${group.id}`}
                  data-kg-selection-metadata-group={group.id}
                  data-kg-selection-metadata-group-expanded={expanded ? 'true' : 'false'}
                  sx={{
                    display: 'grid',
                    gap: expanded ? 0.48 : 0,
                    minWidth: 0,
                    borderRadius: '8px',
                    border: '1px solid',
                    borderColor: alpha(group.color, expanded ? 0.22 : 0.12),
                    bgcolor: alpha(group.color, expanded ? 0.055 : 0.025),
                    overflow: 'hidden',
                  }}
                >
                  <ButtonBase
                    aria-label={`${expanded ? 'Collapse' : 'Expand'} ${group.label} metadata`}
                    onClick={() =>
                      setExpandedSelectionMetadataGroups((current) => ({
                        ...current,
                        [groupKey]: !expanded,
                      }))
                    }
                    sx={{
                      minWidth: 0,
                      minHeight: 28,
                      px: 0.58,
                      py: 0.36,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 0.45,
                      justifyContent: 'flex-start',
                      color: expanded ? group.color : 'text.secondary',
                      textAlign: 'left',
                      transition: 'background-color 140ms ease, color 140ms ease',
                      '&:hover': {
                        bgcolor: alpha(group.color, 0.08),
                        color: group.color,
                      },
                    }}
                  >
                    <Box
                      sx={{
                        width: 20,
                        height: 20,
                        borderRadius: '7px',
                        display: 'grid',
                        placeItems: 'center',
                        flexShrink: 0,
                        color: group.color,
                        bgcolor: alpha(group.color, 0.11),
                        border: '1px solid',
                        borderColor: alpha(group.color, 0.22),
                      }}
                    >
                      {getMetadataGroupIcon(group.id, group.color)}
                    </Box>
                    <Typography variant="caption" noWrap sx={{ flex: 1, minWidth: 0, fontWeight: 900, lineHeight: 1.1 }}>
                      {group.label}
                    </Typography>
                    <Chip
                      size="small"
                      label={group.entries.length}
                      sx={{
                        height: 17,
                        color: group.color,
                        bgcolor: alpha(group.color, 0.1),
                        '& .MuiChip-label': { px: 0.45, fontSize: '0.54rem', fontWeight: 900 },
                      }}
                    />
                    {expanded ? <PreviousResultIcon sx={{ fontSize: 16, flexShrink: 0 }} /> : <ExpandBundleIcon sx={{ fontSize: 16, flexShrink: 0 }} />}
                  </ButtonBase>
                  {!expanded && previewItems.length > 0 && (
                    <Box
                      data-kg-selection-metadata-preview={group.id}
                      sx={{
                        display: 'flex',
                        gap: 0.35,
                        flexWrap: 'wrap',
                        minWidth: 0,
                        px: 0.68,
                        pb: 0.62,
                      }}
                    >
                      {previewItems.map((item) => (
                        <Chip
                          key={`${group.id}-${item.key}`}
                          size="small"
                          label={item.label}
                          sx={{
                            height: 18,
                            maxWidth: '100%',
                            color: group.color,
                            bgcolor: alpha(group.color, 0.08),
                            border: '1px solid',
                            borderColor: alpha(group.color, 0.16),
                            '& .MuiChip-label': {
                              px: 0.54,
                              fontSize: '0.56rem',
                              fontWeight: 800,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                            },
                          }}
                        />
                      ))}
                      {group.entries.length > previewItems.length && (
                        <Chip
                          size="small"
                          label={`+${group.entries.length - previewItems.length}`}
                          sx={{
                            height: 18,
                            color: 'text.secondary',
                            bgcolor: alpha('#fff', 0.05),
                            '& .MuiChip-label': { px: 0.54, fontSize: '0.56rem', fontWeight: 850 },
                          }}
                        />
                      )}
                    </Box>
                  )}
                  <Collapse in={expanded} timeout={150} unmountOnExit>
                    <Box sx={{ px: 0.68, pb: 0.65 }}>
                      <MetadataGrid entries={group.entries} />
                    </Box>
                  </Collapse>
                </Box>
              );
            })}
            {hiddenCount > 0 && (
              <Typography variant="caption" color="text.disabled" sx={{ fontWeight: 800 }}>
                +{hiddenCount} lower-priority fields hidden
              </Typography>
            )}
          </Box>
        </Box>
      );
    };
    const openSelectedRelationshipBundle = (
      bundle: typeof selectedRelationshipBundles[number] | null | undefined
    ) => {
      if (!bundle) return;
      if (bundle.otherEntity) {
        openGraphEntitySelection(bundle.otherEntity, {
          clearSearch: true,
          collapsePanel: true,
          closeInspector: true,
          closeNeighborOrbit: true,
        });
        return;
      }
      openGraphRelationshipDetails(bundle.primaryEdge);
    };

    const renderEntityRoleStrip = (color: string) => {
      if (!selectedEntity || !selectedDegree) return null;
      const topRelation = selectedRelationshipGroups[0] || null;
      const relationColor = topRelation?.color || color;
      const componentLabel = selectedEntityGraphRole?.component
        ? `${selectedEntityGraphRole.componentLabel} · ${formatNumber(selectedEntityGraphRole.component.nodeCount)} nodes`
        : selectedEntityHasRelationships ? 'No component' : 'Isolated';
      const rankLabel = selectedEntityGraphRole?.rank
        ? `${formatNumber(selectedEntityGraphRole.rank)} of ${formatNumber(selectedEntityGraphRole.totalEntities)}`
        : 'Unranked';
      const directionTotal = Math.max(1, selectedDegree.in + selectedDegree.out);
      const outPercent = Math.max(0, Math.min(100, (selectedDegree.out / directionTotal) * 100));
      const linkedCount = selectedRelationshipBundles.length || selectedDegree.total;

      const roleCellSx = {
        minWidth: 0,
        px: 0.58,
        py: 0.48,
        borderRadius: '7px',
        border: '1px solid',
        borderColor: alpha(color, 0.14),
        bgcolor: alpha('#020617', 0.22),
        overflow: 'hidden',
      };

      return (
        <Box
          data-kg-entity-role-strip="true"
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: 'repeat(2, minmax(0, 1fr))', sm: 'repeat(4, minmax(0, 1fr))' },
            gap: 0.42,
            mb: 0.5,
            p: 0.48,
            borderRadius: '8px',
            border: '1px solid',
            borderColor: alpha(color, 0.16),
            bgcolor: alpha(color, 0.045),
          }}
        >
          <Box sx={roleCellSx}>
            <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', fontSize: '0.54rem', fontWeight: 850, lineHeight: 1.05 }}>
              Rank
            </Typography>
            <Typography variant="caption" noWrap sx={{ display: 'block', fontSize: '0.63rem', fontWeight: 920, lineHeight: 1.12 }}>
              {rankLabel}
            </Typography>
          </Box>
          <Box sx={roleCellSx}>
            <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', fontSize: '0.54rem', fontWeight: 850, lineHeight: 1.05 }}>
              Component
            </Typography>
            <Typography variant="caption" noWrap sx={{ display: 'block', fontSize: '0.63rem', fontWeight: 920, lineHeight: 1.12 }}>
              {componentLabel}
            </Typography>
          </Box>
          <Box sx={roleCellSx}>
            <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', fontSize: '0.54rem', fontWeight: 850, lineHeight: 1.05 }}>
              Direction
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.35, minWidth: 0 }}>
              <Typography variant="caption" noWrap sx={{ fontSize: '0.63rem', fontWeight: 920, lineHeight: 1.12 }}>
                Out {formatNumber(selectedDegree.out)}
              </Typography>
              <Typography variant="caption" color="text.disabled" noWrap sx={{ fontSize: '0.55rem', lineHeight: 1.12 }}>
                / In {formatNumber(selectedDegree.in)}
              </Typography>
            </Box>
            <Box sx={{ mt: 0.28, height: 3, borderRadius: 999, overflow: 'hidden', bgcolor: alpha('#fff', 0.08) }}>
              <Box sx={{ width: `${outPercent}%`, height: '100%', bgcolor: color, borderRadius: 999 }} />
            </Box>
          </Box>
          <Box sx={{ ...roleCellSx, borderColor: alpha(relationColor, 0.18), bgcolor: alpha(relationColor, 0.055) }}>
            <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', fontSize: '0.54rem', fontWeight: 850, lineHeight: 1.05 }}>
              Top Link
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.3, minWidth: 0 }}>
              {topRelation ? renderRelationSymbolMark(topRelation.type, relationColor, { selected: true, size: 17 }) : <HubIcon sx={{ fontSize: 12, color }} />}
              <Typography variant="caption" noWrap sx={{ color: relationColor, fontSize: '0.63rem', fontWeight: 920, lineHeight: 1.12 }}>
                {topRelation ? `${getTypeLabel(topRelation.type)} ${formatNumber(topRelation.count)}` : `${formatNumber(linkedCount)} linked`}
              </Typography>
            </Box>
          </Box>
        </Box>
      );
    };
      const renderCollapsedEntityRelationMix = () => {
        if (!selectedEntity || selectionPanelExpanded || selectedRelationshipGroups.length === 0) return null;
      const wideCollapsedCard = selectionCardWidth >= 390;
      const spaciousCollapsedCard = selectionCardWidth >= 540;
      const visibleTypes = selectedRelationshipGroups.slice(0, spaciousCollapsedCard ? 3 : wideCollapsedCard ? 2 : 1);
      const hiddenTypeCount = Math.max(0, selectedRelationshipGroups.length - visibleTypes.length);
      const linkedCount = selectedRelationshipBundles.length || selectedDegree?.total || 0;
      const summaryItems = [
        selectedDegree
          ? { id: 'linked', label: `${formatNumber(linkedCount)} linked`, color: selectedEntityColors.main }
          : null,
        selectedDegree
          ? { id: 'direction', label: `${formatNumber(selectedDegree.out)} out / ${formatNumber(selectedDegree.in)} in`, color: selectedEntityColors.main }
          : null,
      ].filter((item): item is { id: string; label: string; color: string } => Boolean(item));

      return (
        <Box
          data-kg-selection-relation-mix="true"
          sx={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0, 1fr)',
            alignItems: 'center',
            gap: 0.34,
            minWidth: 0,
            mt: -0.08,
            mb: 0.18,
            overflow: 'hidden',
          }}
        >
          {summaryItems.length > 0 && (
            <Box
              data-kg-selection-summary-strip="true"
              sx={{
                display: 'flex',
              alignItems: 'center',
              gap: 0.34,
              minWidth: 0,
              flexWrap: wideCollapsedCard ? 'wrap' : 'nowrap',
              overflow: 'hidden',
            }}
            >
              {summaryItems.slice(0, wideCollapsedCard ? 2 : 1).map(({ id, label, color }) => (
                <Chip
                  key={`selection-summary-${selectedEntity.id}-${id}`}
                  size="small"
                  label={label}
                  sx={{
                    height: 19,
                    maxWidth: spaciousCollapsedCard ? 160 : wideCollapsedCard ? 138 : 112,
                    color: alpha(color, 0.94),
                    bgcolor: alpha(color, 0.08),
                    border: '1px solid',
                    borderColor: alpha(color, 0.18),
                    '& .MuiChip-label': {
                      px: 0.62,
                      fontSize: '0.55rem',
                      fontWeight: 850,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    },
                  }}
                />
              ))}
            </Box>
          )}
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-start',
              gap: 0.36,
              minWidth: 0,
              flexWrap: wideCollapsedCard ? 'wrap' : 'nowrap',
              overflow: 'hidden',
            }}
          >
            {visibleTypes.map(({ type, count, color }) => (
              <ButtonBase
                key={`selection-relation-mix-${selectedEntity.id}-${type}`}
                data-kg-selection-relation-mix-item="true"
                aria-label={`Focus ${getTypeLabel(type)} relationships`}
                onClick={(event) => {
                  event.stopPropagation();
                  focusRelationshipType(type);
                }}
                sx={{
                  position: 'relative',
                  minWidth: 0,
                  height: 24,
                  maxWidth: spaciousCollapsedCard ? 178 : wideCollapsedCard ? 154 : 126,
                  px: 0.32,
                  pr: 0.48,
                  display: 'grid',
                  gridTemplateColumns: '23px minmax(0, 1fr) auto',
                  alignItems: 'center',
                  gap: 0.38,
                  borderRadius: '8px',
                  border: '1px solid',
                  borderColor: alpha(color, 0.28),
                  bgcolor: alpha(color, 0.105),
                  color,
                  overflow: 'hidden',
                  textAlign: 'left',
                  cursor: 'pointer',
                  transition: 'transform 140ms ease, background-color 140ms ease, border-color 140ms ease',
                  '&:hover': {
                    transform: 'translateY(-1px)',
                    borderColor: alpha(color, 0.48),
                    bgcolor: alpha(color, 0.16),
                  },
                  '&::after': {
                    content: '""',
                    position: 'absolute',
                    left: 0,
                    right: `${100 - Math.max(14, Math.min(100, Math.round((count / Math.max(1, selectedRelationshipGroups[0]?.count || count)) * 100)))}%`,
                    bottom: 0,
                    height: 2,
                    bgcolor: alpha(color, 0.72),
                  },
                }}
              >
                {renderRelationSymbolMark(type, color, { selected: true, size: 23 })}
                <Typography variant="caption" noWrap sx={{ minWidth: 0, fontSize: '0.58rem', fontWeight: 900, lineHeight: 1.05 }}>
                  {getTypeLabel(type)}
                </Typography>
                <Typography variant="caption" noWrap sx={{ flexShrink: 0, fontSize: '0.56rem', fontWeight: 950, lineHeight: 1 }}>
                  {formatNumber(count)}
                </Typography>
              </ButtonBase>
            ))}
            {hiddenTypeCount > 0 && (
              <Chip
                size="small"
                label={`+${hiddenTypeCount}`}
                sx={{
                  height: 21,
                  color: 'text.secondary',
                  bgcolor: alpha('#fff', 0.055),
                  border: '1px solid',
                  borderColor: alpha('#fff', 0.1),
                  '& .MuiChip-label': { px: 0.56, fontSize: '0.57rem', fontWeight: 850 },
                }}
              />
            )}
          </Box>
          </Box>
        );
      };
      const renderCollapsedEdgeEvidencePreview = () => {
      if (!selectedEdgeDetails) return null;
      const primaryRelationType = selectedEdgeDetails.relationTypes[0] || null;
      const hiddenRelationTypeCount = Math.max(0, selectedEdgeDetails.relationTypes.length - 1);
      const previewColor = selectedEvidence
        ? getRelationshipColor(selectedEvidence.type || selectedEdgeDetails.edge.type || selectedEdgeDetails.edge.label)
        : primaryRelationType?.color || selectedEdgeDetails.dominantColor;
      const relationMixPreview = selectedEdgeDetails.relationTypes
        .slice(0, 3)
        .map((item) => `${getTypeLabel(item.type)} ${formatNumber(item.count)}`)
        .join(' / ');
      const fallbackPreview = [
        relationMixPreview || getTypeLabel(selectedEdgeDetails.edge.type || selectedEdgeDetails.edge.label),
        `${formatNumber(selectedEdgeDetails.relationCount)} relation${selectedEdgeDetails.relationCount === 1 ? '' : 's'}`,
        selectedEdgeDetails.weight > 0 ? `${formatNumber(Math.round(selectedEdgeDetails.weight))} weight` : '',
        selectedEdgeDetails.domains[0] ? truncateGraphLabel(selectedEdgeDetails.domains[0], 28) : '',
      ].filter(Boolean).join(' / ');
      const evidencePreview = selectedEvidence
        ? getCompactEvidencePreviewText(selectedEvidence.context)
        : fallbackPreview;
      const previewDomain = selectedEvidence?.domain || selectedEdgeDetails.domains[0] || '';
      const evidenceMeta = [
        selectedEvidence
          ? {
          id: 'count',
          label: `${formatNumber(selectedEdgeDetails.evidenceSnippets.length)} ev`,
            color: previewColor,
          }
          : {
            id: 'count',
            label: `${formatNumber(selectedEdgeDetails.relationCount)} rel`,
            color: previewColor,
          },
        primaryRelationType
          ? {
            id: 'type',
            label: `${truncateGraphLabel(getTypeLabel(primaryRelationType.type), 14)} ${formatNumber(primaryRelationType.count)}${hiddenRelationTypeCount > 0 ? ` +${hiddenRelationTypeCount}` : ''}`,
            color: primaryRelationType.color,
          }
          : null,
        selectedEdgeDetails.weight > 0
          ? { id: 'weight', label: `${formatNumber(Math.round(selectedEdgeDetails.weight))} w`, color: accentColors.warning }
          : null,
        previewDomain
          ? { id: 'domain', label: truncateGraphLabel(previewDomain, 22), color: accentColors.info }
          : null,
      ].filter((item): item is { id: string; label: string; color: string } => Boolean(item));
      const previewTitle = selectedEvidence ? 'Evidence' : 'Relationship';

      return (
        <ButtonBase
          data-kg-edge-evidence-preview="true"
          aria-label={selectedEvidence ? 'Open relationship evidence and metadata' : 'Open relationship metadata'}
          onClick={() => {
            setSelectionActionsOpen(false);
            setSelectionPanelExpanded(true);
          }}
          sx={{
            width: '100%',
            minWidth: 0,
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: 0.65,
            mb: 0,
            px: 0.62,
            py: 0.52,
            borderRadius: '8px',
            border: '1px solid',
            borderColor: alpha(previewColor, 0.2),
            bgcolor: alpha('#020617', 0.24),
            color: 'text.primary',
            textAlign: 'left',
            transition: 'transform 140ms ease, background-color 140ms ease, border-color 140ms ease',
            '&:hover': {
              transform: 'translateY(-1px)',
              borderColor: alpha(previewColor, 0.42),
              bgcolor: alpha(previewColor, 0.075),
            },
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 0.52, minWidth: 0, flex: 1 }}>
            <Box
              sx={{
                width: 21,
                height: 21,
                mt: 0.08,
                borderRadius: '7px',
                display: 'grid',
                placeItems: 'center',
                flexShrink: 0,
                color: previewColor,
                bgcolor: alpha(previewColor, 0.12),
                border: '1px solid',
                borderColor: alpha(previewColor, 0.24),
              }}
            >
              {selectedEvidence
                ? <InfoIcon sx={{ fontSize: 13, color: 'inherit' }} />
                : renderRelationSymbolMark(primaryRelationType?.type || selectedEdgeDetails.edge.type || selectedEdgeDetails.edge.label, previewColor, {
                  selected: true,
                  size: 18,
                })}
            </Box>
            <Box sx={{ minWidth: 0, display: 'grid', gap: 0.22, flex: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.35, minWidth: 0, flexWrap: 'wrap' }}>
                <Typography variant="caption" noWrap sx={{ display: 'block', fontSize: '0.62rem', fontWeight: 900, lineHeight: 1.12 }}>
                  {previewTitle}
                </Typography>
                {evidenceMeta.map((item) => (
                  <Chip
                    key={`collapsed-edge-evidence-${item.id}`}
                    size="small"
                    label={item.label}
                    sx={{
                      height: 16,
                      maxWidth: 96,
                      color: item.color,
                      bgcolor: alpha(item.color, 0.1),
                      border: `1px solid ${alpha(item.color, 0.2)}`,
                      '& .MuiChip-label': {
                        px: 0.42,
                        fontSize: '0.5rem',
                        fontWeight: 900,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                      },
                    }}
                  />
                ))}
              </Box>
              <Typography
                variant="caption"
                sx={{
                  display: '-webkit-box',
                  WebkitLineClamp: 2,
                  WebkitBoxOrient: 'vertical',
                  overflow: 'hidden',
                  color: 'text.secondary',
                  fontSize: '0.59rem',
                  fontWeight: 780,
                  lineHeight: 1.2,
                  minWidth: 0,
                  maxWidth: '100%',
                }}
              >
                {evidencePreview}
              </Typography>
            </Box>
          </Box>
            <Box
              sx={{
                width: 22,
                height: 22,
                mt: 0.08,
                borderRadius: '7px',
                display: 'grid',
                placeItems: 'center',
                flexShrink: 0,
                color: previewColor,
                bgcolor: alpha(previewColor, 0.1),
                border: '1px solid',
                borderColor: alpha(previewColor, 0.22),
              }}
            >
              <ExpandBundleIcon sx={{ fontSize: 15 }} />
            </Box>
        </ButtonBase>
      );
    };
    const renderEntityPathPreview = (preview: GraphPathPreview, color: string) => {
      const source = preview.nodes[0];
      const target = preview.nodes[preview.nodes.length - 1];
      const hopCount = preview.segments.length;
      const primarySegment = preview.segments[0];
      const primaryColor = primarySegment?.color || color;
      const revealPath = () => {
        setSearchTerm('');
        setSelectedEntityTypes([]);
        setSelectedRelationshipTypes([]);
        setSelectedComponentId(null);
        setFocusNeighborhood(false);
        setGraphScope('all');
        setParticleFlow(true);
        setPathFocusVersion((value) => value + 1);
      };
      const openPathEntity = (entity: Entity) => {
        setSearchTerm('');
        setSelectedEdgeId(null);
        setSelectedEntity(entity);
        setSelectedNodeId(entity.id);
        setExpandedRelationshipBundleId(null);
        setSelectionPanelExpanded(false);
        setTabValue(2);
        setPathFocusVersion((value) => value + 1);
      };
      const openPathSegment = (segment: GraphPathSegment) => {
        const edge = visualGraph.edges.find((item) =>
          item.id === segment.visualEdgeId ||
          item.id === segment.edgeId ||
          (
            item.source === segment.fromId &&
            item.target === segment.toId
          ) ||
          (
            item.source === segment.toId &&
            item.target === segment.fromId
          )
        );

        if (!edge) {
          revealPath();
          return;
        }

        setSearchTerm('');
        setSelectedEntityTypes([]);
        setSelectedRelationshipTypes([]);
        setSelectedComponentId(null);
        setFocusNeighborhood(false);
        setGraphScope('all');
        handleLinkClick(edge);
        setSelectionPanelExpanded(false);
        setInspectorOpen(false);
        setPathFocusVersion((value) => value + 1);
      };

      return (
        <Box
          data-kg-path-preview="true"
          sx={{
            width: '100%',
            px: 0.65,
            py: 0.5,
            mb: 0.18,
            borderRadius: '8px',
            border: '1px solid',
            borderColor: alpha(primaryColor, 0.26),
            bgcolor: alpha(primaryColor, 0.065),
            color: 'text.primary',
            textAlign: 'left',
            overflow: 'hidden',
            animation: 'kgPathPreviewIn 180ms ease-out both',
            transition: 'background-color 140ms ease, border-color 140ms ease, transform 140ms ease',
            '@keyframes kgPathPreviewIn': {
              from: { opacity: 0, transform: 'translateX(5px) scale(0.99)' },
              to: { opacity: 1, transform: 'translateX(0) scale(1)' },
            },
            '&:hover': {
              borderColor: alpha(primaryColor, 0.46),
              bgcolor: alpha(primaryColor, 0.11),
            },
          }}
        >
          <ButtonBase
            data-kg-path-focus="true"
            aria-label={`Frame path from ${source?.name || preview.sourceId} to ${target?.name || preview.targetId}`}
            onClick={revealPath}
            sx={{
              width: '100%',
              minHeight: 21,
              justifyContent: 'flex-start',
              alignItems: 'center',
              gap: 0.55,
              color: 'text.primary',
              textAlign: 'left',
              borderRadius: '7px',
              transition: 'transform 140ms ease',
              '&:hover': { transform: 'translateX(1px)' },
            }}
          >
            <RelationIcon sx={{ fontSize: 15, color: primaryColor, flexShrink: 0 }} />
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.42, minWidth: 0, flex: 1 }}>
              <Typography variant="caption" noWrap sx={{ fontWeight: 900, lineHeight: 1.08 }}>
                {hopCount} hop{hopCount === 1 ? '' : 's'}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.3, minWidth: 0, overflow: 'hidden' }}>
                {preview.segments.slice(0, 3).map((segment, index) => (
                  <Chip
                    key={`${segment.edgeId}-${index}`}
                    size="small"
                    data-kg-path-segment="true"
                    icon={renderRelationSymbolMark(segment.type, segment.color, { selected: true, size: 15 })}
                    label={segment.label}
                    onClick={(event) => {
                      event.stopPropagation();
                      openPathSegment(segment);
                    }}
                    sx={{
                      height: 17,
                      maxWidth: 96,
                      color: segment.color,
                      bgcolor: alpha(segment.color, 0.1),
                      borderColor: alpha(segment.color, 0.26),
                      cursor: 'pointer',
                      '& .MuiChip-label': {
                        px: 0.48,
                        fontSize: '0.54rem',
                        fontWeight: 850,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                      },
                      '& .MuiChip-icon': { ml: 0.42, mr: -0.26, color: 'inherit' },
                    }}
                  />
                ))}
                {preview.segments.length > 3 && (
                  <Chip
                    size="small"
                    label={`+${preview.segments.length - 3}`}
                    sx={{ height: 17, '& .MuiChip-label': { px: 0.48, fontSize: '0.54rem', fontWeight: 850 } }}
                  />
                )}
              </Box>
            </Box>
          </ButtonBase>
          <Box
            data-kg-path-nodes="true"
            sx={{ display: 'flex', alignItems: 'center', gap: 0.35, mt: 0.42, minWidth: 0, overflow: 'hidden' }}
          >
            {preview.nodes.slice(0, 5).map((node, index) => {
              const nodeColors = getTypeColors(node.entity_type, node.name);
              const isCurrent = node.id === selectedEntity?.id;
              return (
                <Box key={`path-node-${node.id}-${index}`} sx={{ display: 'flex', alignItems: 'center', gap: 0.35, minWidth: 0, flexShrink: index === preview.nodes.length - 1 ? 1 : 0 }}>
                  {index > 0 && (
                    <Typography variant="caption" color="text.disabled" sx={{ lineHeight: 1, flexShrink: 0 }}>
                      {'>'}
                    </Typography>
                  )}
                  <Chip
                    size="small"
                    data-kg-path-node={isCurrent ? 'current' : 'hop'}
                    icon={getEntityIcon(node.entity_type, isCurrent ? '#fff' : nodeColors.light, node.name)}
                    label={node.name}
                    onClick={(event) => {
                      event.stopPropagation();
                      if (!isCurrent) openPathEntity(node);
                      else revealPath();
                    }}
                    variant={isCurrent ? 'filled' : 'outlined'}
                    sx={{
                      height: 19,
                      maxWidth: index === preview.nodes.length - 1 ? 132 : 116,
                      color: isCurrent ? '#fff' : nodeColors.light,
                      bgcolor: isCurrent ? alpha(nodeColors.main, 0.34) : alpha(nodeColors.main, 0.1),
                      borderColor: alpha(nodeColors.main, isCurrent ? 0.52 : 0.34),
                      cursor: 'pointer',
                      '& .MuiChip-label': {
                        px: 0.5,
                        fontSize: '0.56rem',
                        fontWeight: 850,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                      },
                      '& .MuiChip-icon': { ml: 0.42, mr: -0.28, color: 'inherit' },
                    }}
                  />
                </Box>
              );
            })}
            {preview.nodes.length > 5 && (
              <Chip
                size="small"
                label={`+${preview.nodes.length - 5}`}
                sx={{ height: 19, '& .MuiChip-label': { px: 0.5, fontSize: '0.56rem', fontWeight: 850 } }}
              />
            )}
          </Box>
        </Box>
      );
    };
    const renderIsolatedEntityNotice = (color: string) => {
      if (!selectedEntity || selectedEntityHasRelationships) return null;
      return (
        <Box
          data-kg-isolated-entity-notice="true"
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 0.45,
            minWidth: 0,
            mb: 0.38,
            px: 0.58,
            py: 0.42,
            borderRadius: '8px',
            border: '1px solid',
            borderColor: alpha(color, 0.16),
            bgcolor: alpha(color, 0.055),
            color: alpha(color, 0.94),
          }}
        >
          <HubIcon sx={{ fontSize: 13, flexShrink: 0 }} />
          <Typography variant="caption" noWrap sx={{ minWidth: 0, fontSize: '0.58rem', fontWeight: 850, lineHeight: 1.1 }}>
            No links in current graph
          </Typography>
        </Box>
      );
    };
    const renderEntityNeighborOrbit = (
      bundles: typeof selectedRelationshipBundles,
      color: string
    ) => {
      if (!selectedEntity || bundles.length === 0) return null;
      const compactNeighborRail = !neighborOrbitExpanded;
      const mobileCompactNeighborRail = compactNeighborRail && selectionViewportWidth < 560;
      const wideCompactNeighborRail = compactNeighborRail && selectionCardWidth >= 470;
      const spaciousCompactNeighborRail = compactNeighborRail && selectionCardWidth >= 540;
      const expandedNeighborColumns = selectionCardWidth >= 760 ? 3 : selectionCardWidth >= 520 ? 2 : 1;
      const compactNeighborColumns = mobileCompactNeighborRail ? Math.min(2, bundles.length) : spaciousCompactNeighborRail ? 3 : wideCompactNeighborRail ? 2 : 1;
      const visibleLimit = neighborOrbitExpanded
        ? expandedNeighborColumns >= 3 ? 9 : expandedNeighborColumns === 2 ? 6 : 4
        : mobileCompactNeighborRail ? 2 : spaciousCompactNeighborRail ? 4 : wideCompactNeighborRail ? 2 : 2;
      const visibleBundles = bundles.slice(0, visibleLimit);
      const hiddenCount = Math.max(0, bundles.length - visibleBundles.length);
      const canExpand = bundles.length > visibleLimit || neighborOrbitExpanded;
      const openBundle = (bundle: typeof selectedRelationshipBundles[number]) => {
        openSelectedRelationshipBundle(bundle);
      };
      const previewBundle = (bundle: typeof selectedRelationshipBundles[number]) => {
        if (bundle.otherId) setHoveredNodeId(bundle.otherId);
      };
      const clearBundlePreview = (bundle: typeof selectedRelationshipBundles[number]) => {
        if (!bundle.otherId) return;
        setHoveredNodeId((current) => (current === bundle.otherId ? null : current));
      };

      return (
        <Box
          data-kg-neighbor-orbit="true"
          data-kg-neighbor-orbit-expanded={neighborOrbitExpanded ? 'true' : 'false'}
          sx={{
            width: '100%',
            mb: 0.08,
            p: compactNeighborRail ? 0.38 : 0.62,
            borderRadius: '8px',
            border: '1px solid',
            borderColor: alpha(color, 0.18),
            bgcolor: alpha('#020617', 0.26),
            overflow: 'hidden',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.38, mb: compactNeighborRail ? 0.24 : 0.48, minWidth: 0 }}>
            <RelationIcon sx={{ fontSize: 14, color, flexShrink: 0 }} />
            <Typography variant="caption" color="text.secondary" noWrap sx={{ flex: 1, minWidth: 0, fontWeight: 900, lineHeight: 1.05 }}>
              {formatNumber(bundles.length)} linked
            </Typography>
            <Chip
              size="small"
              label={`${formatNumber(selectedEntityGraphRole?.totalRelations || selectedDegree?.total || 0)} rel`}
              sx={{
                height: 18,
                color,
                bgcolor: alpha(color, 0.12),
                '& .MuiChip-label': { px: 0.5, fontSize: '0.56rem', fontWeight: 900 },
              }}
            />
            {canExpand && (
              <Chip
                size="small"
                data-kg-neighbor-orbit-toggle="true"
                data-kg-neighbor-orbit-more="true"
                label={neighborOrbitExpanded ? 'Less' : `+${Math.max(0, bundles.length - visibleLimit)}`}
                onClick={() => setNeighborOrbitExpanded((open) => !open)}
                sx={{
                  height: 18,
                  color,
                  bgcolor: alpha(color, 0.1),
                  border: '1px solid',
                  borderColor: alpha(color, 0.24),
                  cursor: 'pointer',
                  '& .MuiChip-label': { px: 0.5, fontSize: '0.56rem', fontWeight: 900 },
                }}
              />
            )}
          </Box>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: neighborOrbitExpanded
                ? {
                  xs: '1fr',
                  sm: expandedNeighborColumns >= 2 ? 'repeat(2, minmax(0, 1fr))' : '1fr',
                  lg: expandedNeighborColumns >= 3 ? 'repeat(3, minmax(0, 1fr))' : expandedNeighborColumns >= 2 ? 'repeat(2, minmax(0, 1fr))' : '1fr',
                }
                : compactNeighborColumns >= 3 ? 'repeat(3, minmax(0, 1fr))' : compactNeighborColumns === 2 ? 'repeat(2, minmax(0, 1fr))' : '1fr',
              gap: compactNeighborRail ? 0.22 : 0.4,
              alignItems: 'stretch',
            }}
          >
            {visibleBundles.map((bundle) => {
              const topTypes = bundle.types.slice(0, 3);
              const topType = topTypes[0];
              const hiddenTypeCount = Math.max(0, bundle.types.length - topTypes.length);
              const otherEntity = bundle.otherEntity;
              const otherLabel = otherEntity?.name || bundle.otherId;
              const dominantColor = topType?.color || getTypeColors(otherEntity?.entity_type, otherEntity?.name).main;
              const hasOutgoing = bundle.outCount > 0;
              const hasIncoming = bundle.inCount > 0;
              const directionLabel = hasOutgoing && hasIncoming
                ? `${formatNumber(bundle.outCount)} out / ${formatNumber(bundle.inCount)} in`
                : hasOutgoing
                  ? `${formatNumber(bundle.outCount)} outgoing`
                  : `${formatNumber(bundle.inCount)} incoming`;
              const compactDirectionLabel = hasOutgoing && hasIncoming
                ? `${formatNumber(bundle.outCount)}/${formatNumber(bundle.inCount)}`
                : hasOutgoing
                  ? `${formatNumber(bundle.outCount)} out`
                  : `${formatNumber(bundle.inCount)} in`;
              const directionIcon = hasOutgoing && hasIncoming
                ? <RelationIcon sx={{ fontSize: 11, color: alpha(dominantColor, 0.9), flexShrink: 0 }} />
                : hasOutgoing
                  ? <CollapseBundleIcon sx={{ fontSize: 12, color: alpha(dominantColor, 0.95), flexShrink: 0 }} />
                  : <BrowseBackIcon sx={{ fontSize: 12, color: alpha(dominantColor, 0.95), flexShrink: 0 }} />;
              const routeLabel = hasOutgoing && hasIncoming
                ? `${selectedEntity.name} and ${otherLabel}`
                : hasOutgoing
                  ? `${selectedEntity.name} to ${otherLabel}`
                  : `${otherLabel} to ${selectedEntity.name}`;
              return (
                <ButtonBase
                  component="div"
                  role="button"
                  key={`neighbor-orbit-${selectedEntity.id}-${bundle.otherId}`}
                  data-kg-neighbor-orbit-item="true"
                  data-kg-neighbor-orbit-action={otherEntity ? 'entity' : bundle.primaryEdge ? 'relation' : 'none'}
                  data-kg-neighbor-direction={hasOutgoing && hasIncoming ? 'both' : hasOutgoing ? 'outgoing' : 'incoming'}
                  aria-label={otherEntity ? `Open linked entity ${otherLabel}` : `Open ${directionLabel} relationship: ${routeLabel}`}
                  tabIndex={0}
                  onFocus={() => previewBundle(bundle)}
                  onBlur={() => clearBundlePreview(bundle)}
                  onPointerEnter={() => previewBundle(bundle)}
                  onPointerOver={() => previewBundle(bundle)}
                  onPointerMove={() => previewBundle(bundle)}
                  onPointerLeave={() => clearBundlePreview(bundle)}
                  onMouseEnter={() => previewBundle(bundle)}
                    onMouseOver={() => previewBundle(bundle)}
                    onMouseLeave={() => clearBundlePreview(bundle)}
                    onClick={() => openBundle(bundle)}
                    sx={{
                      position: 'relative',
                      minWidth: 0,
                      minHeight: compactNeighborRail ? 36 : 46,
                      justifyContent: 'flex-start',
                      alignItems: compactNeighborRail ? 'flex-start' : 'stretch',
                      gap: compactNeighborRail ? 0.32 : 0.48,
                      px: compactNeighborRail ? 0.4 : 0.55,
                      pr: compactNeighborRail && bundle.primaryEdge ? 2.45 : compactNeighborRail ? 0.48 : 0.55,
                      py: compactNeighborRail ? 0.2 : 0.45,
                      borderRadius: '8px',
                      border: '1px solid',
                    borderColor: alpha(dominantColor, 0.2),
                    bgcolor: alpha(dominantColor, 0.055),
                    color: 'text.primary',
                    textAlign: 'left',
                    overflow: 'hidden',
                    transition: 'transform 140ms ease, border-color 140ms ease, background-color 140ms ease',
                    '&:hover': {
                        transform: 'translateY(-1px)',
                        borderColor: alpha(dominantColor, 0.4),
                        bgcolor: alpha(dominantColor, 0.1),
                      },
                      '&:hover [data-kg-neighbor-orbit-relation-details="true"], &:focus-within [data-kg-neighbor-orbit-relation-details="true"]': {
                        opacity: 1,
                      },
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', minHeight: compactNeighborRail ? 30 : 32, flexShrink: 0 }}>
                      {renderEntitySymbolMark(otherEntity?.entity_type, otherEntity?.name || bundle.otherId, {
                        selected: selectedNodeId === otherEntity?.id,
                        size: compactNeighborRail ? 22 : 28,
                      })}
                    </Box>
                    <Box sx={{ minWidth: 0, flex: 1, display: 'grid', gap: compactNeighborRail ? 0.18 : 0.32 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', minWidth: 0, gap: 0.45 }}>
                        <Typography variant="caption" noWrap sx={{ display: 'block', flex: 1, minWidth: 0, fontWeight: 900, lineHeight: 1.08 }}>
                          {otherLabel}
                        </Typography>
                        <Box
                          component="span"
                          data-kg-neighbor-orbit-count="true"
                          sx={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            flexShrink: 0,
                            minWidth: 0,
                            px: compactNeighborRail ? 0.38 : 0.45,
                            py: 0.08,
                            borderRadius: '6px',
                            color: alpha(dominantColor, 0.98),
                            bgcolor: alpha(dominantColor, 0.12),
                            border: '1px solid',
                            borderColor: alpha(dominantColor, 0.18),
                          }}
                        >
                          <Typography variant="caption" noWrap sx={{ fontSize: compactNeighborRail ? '0.53rem' : '0.56rem', fontWeight: 900, lineHeight: 1.08, color: 'inherit' }}>
                            {formatNumber(bundle.total)} rel
                          </Typography>
                        </Box>
                        {bundle.primaryEdge && (
                          <Tooltip title="Open relationship details" disableHoverListener disableFocusListener disableTouchListener>
                            <IconButton
                              size="small"
                              data-kg-neighbor-orbit-relation-details="true"
                              aria-label={`Open relationship details for ${routeLabel}`}
                              onClick={(event) => {
                                event.stopPropagation();
                                openGraphRelationshipDetails(bundle.primaryEdge, { expandPanel: true });
                              }}
                              onMouseDown={(event) => event.stopPropagation()}
                              sx={{
                                position: compactNeighborRail ? 'absolute' : 'static',
                                right: compactNeighborRail ? 4 : 'auto',
                                top: compactNeighborRail ? 4 : 'auto',
                                width: 19,
                                height: 19,
                                ml: compactNeighborRail ? 0 : -0.18,
                                flexShrink: 0,
                                color: alpha(dominantColor, 0.9),
                                opacity: compactNeighborRail ? 0 : 0.88,
                                transition: 'opacity 140ms ease, background-color 140ms ease, color 140ms ease',
                                '&:hover': {
                                  opacity: 1,
                                  color: dominantColor,
                                  bgcolor: alpha(dominantColor, 0.14),
                                },
                                '& .MuiSvgIcon-root': { fontSize: 13 },
                              }}
                            >
                              <RelationIcon sx={{ fontSize: 13 }} />
                            </IconButton>
                          </Tooltip>
                        )}
                      </Box>
                      {compactNeighborRail && (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.28, minWidth: 0, color: alpha(dominantColor, 0.88) }}>
                          {topType
                            ? renderRelationSymbolMark(topType.type, dominantColor, { selected: true, size: 18 })
                            : <RelationIcon sx={{ fontSize: 10.5, color: dominantColor, flexShrink: 0 }} />}
                        <Typography
                          variant="caption"
                          noWrap
                          sx={{ minWidth: 0, color: 'inherit', fontSize: '0.55rem', fontWeight: 850, lineHeight: 1.06 }}
                        >
                            {topType ? `${getTypeLabel(topType.type)} ${formatNumber(topType.count)}` : 'Related'}
                          </Typography>
                        {hiddenTypeCount > 0 && spaciousCompactNeighborRail && (
                          <Typography variant="caption" noWrap sx={{ flexShrink: 0, color: alpha(dominantColor, 0.78), fontSize: '0.52rem', fontWeight: 850, lineHeight: 1.06 }}>
                            +{hiddenTypeCount}
                          </Typography>
                        )}
                        <Typography variant="caption" aria-hidden="true" sx={{ color: alpha('#fff', 0.34), fontSize: '0.58rem', lineHeight: 1 }}>
                          ·
                        </Typography>
                        <Typography variant="caption" noWrap sx={{ minWidth: 0, flexShrink: 0, fontSize: '0.54rem', fontWeight: 850, lineHeight: 1.06, color: alpha(dominantColor, 0.9) }}>
                          {compactDirectionLabel}
                        </Typography>
                      </Box>
                    )}
                    {!compactNeighborRail && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.34, minWidth: 0 }}>
                      {topType && renderRelationSymbolMark(topType.type, dominantColor, { selected: true, size: 19 })}
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        noWrap
                        title={routeLabel}
                        sx={{ display: 'block', minWidth: 0, lineHeight: 1.08, fontSize: '0.57rem', flex: 1 }}
                      >
                        {topType ? getTypeLabel(topType.type) : 'Related'}
                      </Typography>
                      <Box
                        component="span"
                        sx={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: 0.18,
                          minWidth: 0,
                          maxWidth: '54%',
                          px: 0.35,
                          py: 0.08,
                          borderRadius: '6px',
                          color: alpha(dominantColor, 0.95),
                          bgcolor: alpha(dominantColor, 0.1),
                          border: '1px solid',
                          borderColor: alpha(dominantColor, 0.14),
                          overflow: 'hidden',
                        }}
                      >
                        {directionIcon}
                        <Typography
                          variant="caption"
                          noWrap
                          sx={{ minWidth: 0, fontSize: '0.54rem', fontWeight: 850, lineHeight: 1.08, color: 'inherit' }}
                        >
                          {directionLabel}
                        </Typography>
                      </Box>
                    </Box>
                    )}
                    {!compactNeighborRail && (
                    <Box
                      aria-hidden="true"
                      sx={{
                        display: 'flex',
                        height: 3,
                        borderRadius: '999px',
                        overflow: 'hidden',
                        bgcolor: alpha(dominantColor, 0.08),
                      }}
                    >
                      {(topTypes.length > 0 ? topTypes : [{ type: 'related', count: 1, color: dominantColor }]).map((type) => (
                        <Box
                          key={`${bundle.otherId}-${type.type}`}
                          sx={{
                            flex: Math.max(1, type.count),
                            bgcolor: type.color,
                            opacity: 0.78,
                          }}
                        />
                      ))}
                    </Box>
                    )}
                    {neighborOrbitExpanded && topTypes.length > 1 && (
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.28, minWidth: 0, overflow: 'hidden' }}>
                        {topTypes.slice(1).map((type) => (
                          <Typography
                            key={`${bundle.otherId}-${type.type}-label`}
                            variant="caption"
                            noWrap
                            sx={{
                              minWidth: 0,
                              color: alpha(type.color, 0.95),
                              fontSize: '0.52rem',
                              fontWeight: 850,
                              lineHeight: 1,
                            }}
                          >
                            {getTypeLabel(type.type)} {formatNumber(type.count)}
                          </Typography>
                        ))}
                        {hiddenTypeCount > 0 && (
                          <Typography variant="caption" noWrap sx={{ color: 'text.disabled', fontSize: '0.52rem', fontWeight: 850, lineHeight: 1 }}>
                            +{hiddenTypeCount}
                          </Typography>
                        )}
                      </Box>
                    )}
                  </Box>
                  <Box
                    aria-hidden="true"
                    sx={{
                      alignSelf: 'stretch',
                      width: 2,
                      borderRadius: '999px',
                      bgcolor: alpha(dominantColor, hasOutgoing && hasIncoming ? 0.42 : 0.3),
                      flexShrink: 0,
                    }}
                  />
                </ButtonBase>
              );
            })}
          </Box>
          {neighborOrbitExpanded && hiddenCount > 0 && (
            <Typography
              variant="caption"
              color="text.disabled"
              data-kg-neighbor-orbit-hidden="true"
              noWrap
              sx={{ display: 'block', mt: 0.42, px: 0.2, fontSize: '0.58rem', lineHeight: 1.1 }}
            >
              +{formatNumber(hiddenCount)} more linked entities
            </Typography>
          )}
        </Box>
      );
    };
    const renderGraphOverview = () => {
      const overviewNodeById = new Map(graphOverview.nodes.map((item) => [item.node.id, item]));
      const visibleEdgeCount = graphOverview.edges.length;
      const hoveredOverviewId = graphOverviewHover?.id || null;
      const hoveredOverviewKind = graphOverviewHover?.kind || null;
      const clearOverviewHover = (id: string) => {
        setGraphOverviewHover((current) => (current?.id === id ? null : current));
      };
      const renderOverviewNodeGlyph = ({
        node,
        x,
        y,
        r,
        color,
        selected,
        focused,
        hovered,
      }: {
        node: GraphNode;
        x: number;
        y: number;
        r: number;
        color: string;
        selected: boolean;
        focused: boolean;
        hovered: boolean;
      }) => {
        const visualType = getEntityVisualType(node.type, node.label);
        const size = Math.max(3.4, r * (selected ? 1.9 : focused ? 1.72 : hovered ? 1.82 : 1.58));
        const fillOpacity = selected ? 1 : focused ? 0.86 : hovered ? 0.92 : 0.64;
        const stroke = selected ? '#fff' : alpha('#fff', focused || hovered ? 0.58 : 0.34);
        const strokeWidth = selected ? 1.15 : focused || hovered ? 0.82 : 0.48;
        const common = {
          fill: color,
          fillOpacity,
          stroke,
          strokeWidth,
        };

        if (visualType === 'organization' || visualType === 'service' || visualType === 'runtime') {
          return (
            <rect
              x={x - size}
              y={y - size * 0.72}
              width={size * 2}
              height={size * 1.44}
              rx={Math.max(1.5, size * 0.28)}
              {...common}
            />
          );
        }

        if (visualType === 'location') {
          return (
            <path
              d={`M ${x} ${y + size * 1.05} C ${x - size * 0.84} ${y + size * 0.14}, ${x - size * 0.74} ${y - size * 0.9}, ${x} ${y - size * 1.08} C ${x + size * 0.74} ${y - size * 0.9}, ${x + size * 0.84} ${y + size * 0.14}, ${x} ${y + size * 1.05} Z`}
              {...common}
            />
          );
        }

        if (visualType === 'model' || visualType === 'hardware') {
          const points = Array.from({ length: 6 }, (_, index) => {
            const angle = (Math.PI / 3) * index - Math.PI / 6;
            return `${x + Math.cos(angle) * size},${y + Math.sin(angle) * size}`;
          }).join(' ');
          return <polygon points={points} {...common} />;
        }

        if (visualType === 'product') {
          return (
            <path
              d={`M ${x - size} ${y - size * 0.72} L ${x + size * 0.48} ${y - size * 0.72} L ${x + size} ${y - size * 0.2} L ${x + size} ${y + size * 0.72} L ${x - size} ${y + size * 0.72} Z`}
              {...common}
            />
          );
        }

        if (visualType === 'concept') {
          const points = Array.from({ length: 6 }, (_, index) => {
            const angle = (Math.PI / 3) * index + Math.PI / 6;
            return `${x + Math.cos(angle) * size},${y + Math.sin(angle) * size}`;
          }).join(' ');
          return <polygon points={points} {...common} />;
        }

        if (visualType === 'code') {
          return (
            <g>
              <rect
                x={x - size}
                y={y - size}
                width={size * 2}
                height={size * 2}
                rx={Math.max(1.2, size * 0.22)}
                {...common}
              />
              <path
                d={`M ${x - size * 0.42} ${y - size * 0.18} L ${x - size * 0.68} ${y} L ${x - size * 0.42} ${y + size * 0.18} M ${x + size * 0.42} ${y - size * 0.18} L ${x + size * 0.68} ${y} L ${x + size * 0.42} ${y + size * 0.18}`}
                fill="none"
                stroke={alpha('#020617', 0.72)}
                strokeWidth={Math.max(0.55, size * 0.16)}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </g>
          );
        }

        if (visualType === 'data') {
          return (
            <g>
              <rect
                x={x - size * 0.9}
                y={y - size * 0.62}
                width={size * 1.8}
                height={size * 1.24}
                rx={Math.max(1, size * 0.12)}
                {...common}
              />
              <ellipse
                cx={x}
                cy={y - size * 0.62}
                rx={size * 0.9}
                ry={size * 0.32}
                fill={alpha('#fff', focused || selected ? 0.22 : 0.14)}
                stroke={stroke}
                strokeWidth={strokeWidth}
              />
            </g>
          );
        }

        if (visualType === 'prompt') {
          return (
            <path
              d={`M ${x - size * 0.78} ${y - size} L ${x + size * 0.36} ${y - size} L ${x + size * 0.78} ${y - size * 0.56} L ${x + size * 0.78} ${y + size} L ${x - size * 0.78} ${y + size} Z`}
              {...common}
            />
          );
        }

        if (visualType === 'vector' || visualType === 'metric' || visualType === 'evaluation') {
          return (
            <polygon
              points={`${x},${y - size * 1.06} ${x + size * 1.06},${y} ${x},${y + size * 1.06} ${x - size * 1.06},${y}`}
              {...common}
            />
          );
        }

        return <circle cx={x} cy={y} r={size * 0.88} {...common} />;
      };

      return (
        <Box
          data-kg-overview-map="true"
          data-kg-overview-hover-kind={hoveredOverviewKind || 'none'}
          sx={{
            width: { xs: 252, md: 'min(30vw, 340px)', xl: 360 },
            maxWidth: '100%',
            p: 0.75,
            display: 'grid',
            gap: 0.55,
            ...graphOverlayPanelSx,
            borderColor: graphOverviewHover ? alpha(graphOverviewHover.color, 0.26) : alpha('#fff', 0.1),
            pointerEvents: 'auto',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.65, minWidth: 0 }}>
            <HubIcon sx={{ fontSize: 16, color: accentColors.primary, flexShrink: 0 }} />
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography variant="caption" noWrap sx={{ display: 'block', fontWeight: 900, lineHeight: 1.1 }}>
                Topology
              </Typography>
              <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', lineHeight: 1.1, fontSize: '0.58rem' }}>
                {formatNumber(graphOverview.nodes.length)} visible / {formatNumber(visibleEdgeCount)} links
              </Typography>
            </Box>
            {graphOverview.focusBounds && (
              <Chip
                size="small"
                data-kg-overview-focus-count="true"
                label={`${formatNumber(graphOverview.focusBounds.nodeCount)} focus`}
                sx={{
                  height: 18,
                  color: accentColors.primary,
                  bgcolor: alpha(accentColors.primary, 0.12),
                  border: `1px solid ${alpha(accentColors.primary, 0.28)}`,
                  '& .MuiChip-label': { px: 0.48, fontSize: '0.56rem', fontWeight: 900 },
                }}
              />
            )}
            {graphOverview.hiddenNodeCount > 0 && (
              <Chip
                size="small"
                label={`+${formatNumber(graphOverview.hiddenNodeCount)}`}
                sx={{
                  height: 18,
                  color: 'text.secondary',
                  bgcolor: alpha('#fff', 0.055),
                  '& .MuiChip-label': { px: 0.48, fontSize: '0.56rem', fontWeight: 900 },
                }}
              />
            )}
            <Tooltip title="Close overview" disableHoverListener disableFocusListener disableTouchListener>
              <IconButton
                size="small"
                data-kg-overview-close="true"
                aria-label="Close topology overview"
                onClick={() => setGraphOverviewOpen(false)}
                sx={{ width: 24, height: 24, borderRadius: '8px', color: 'text.secondary', flexShrink: 0 }}
              >
                <CloseIcon sx={{ fontSize: 15 }} />
              </IconButton>
            </Tooltip>
          </Box>
          <Box
            component="svg"
            viewBox="0 0 208 132"
            role="button"
            tabIndex={0}
            data-kg-overview-navigation="true"
            aria-label="Compact graph topology overview. Click or drag blank space to move the 3D camera."
            onClick={(event) => {
              if (graphOverviewDragRef.current.moved) {
                graphOverviewDragRef.current.moved = false;
                return;
              }
              const target = event.target as Element | null;
              if (target?.closest('[data-kg-overview-node], [data-kg-overview-edge]')) return;
              focusGraphOverviewAt(event.clientX, event.clientY, event.currentTarget.getBoundingClientRect());
            }}
            onPointerDown={(event) => {
              const target = event.target as Element | null;
              if (target?.closest('[data-kg-overview-node], [data-kg-overview-edge]')) return;
              if (!graphOverview.projection) return;
              graphOverviewDragRef.current = { active: true, moved: false, lastMoveAt: 0 };
              event.currentTarget.setPointerCapture(event.pointerId);
            }}
            onPointerMove={(event) => {
              const dragState = graphOverviewDragRef.current;
              if (!dragState.active) return;
              event.preventDefault();
              const now = typeof window !== 'undefined' ? window.performance.now() : Date.now();
              if (now - dragState.lastMoveAt < 120) return;
              dragState.lastMoveAt = now;
              dragState.moved = true;
              focusGraphOverviewAt(event.clientX, event.clientY, event.currentTarget.getBoundingClientRect());
            }}
            onPointerUp={(event) => {
              graphOverviewDragRef.current.active = false;
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId);
              }
            }}
            onPointerCancel={(event) => {
              graphOverviewDragRef.current = { active: false, moved: false, lastMoveAt: 0 };
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId);
              }
            }}
            sx={{
              width: '100%',
              height: 132,
              borderRadius: '8px',
              border: '1px solid',
              borderColor: alpha('#fff', 0.08),
              bgcolor: alpha('#020617', 0.38),
              overflow: 'hidden',
              cursor: graphOverview.projection ? 'grab' : 'default',
            }}
          >
            <defs>
              <radialGradient id="kg-overview-vignette" cx="50%" cy="50%" r="70%">
                <stop offset="0%" stopColor={alpha(accentColors.primary, 0.16)} />
                <stop offset="100%" stopColor={alpha('#020617', 0)} />
              </radialGradient>
              <filter id="kg-overview-focus-glow" x="-40%" y="-40%" width="180%" height="180%">
                <feGaussianBlur stdDeviation="2.6" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>
            <rect x="0" y="0" width="208" height="132" fill="url(#kg-overview-vignette)" />
            {graphOverview.focusBounds && (
              <g data-kg-overview-focus-lens="true">
                <rect
                  x={graphOverview.focusBounds.x}
                  y={graphOverview.focusBounds.y}
                  width={graphOverview.focusBounds.width}
                  height={graphOverview.focusBounds.height}
                  rx="7"
                  fill={alpha(accentColors.primary, 0.08)}
                  stroke={alpha(accentColors.primary, 0.68)}
                  strokeWidth="1.1"
                  strokeDasharray="4 3"
                  filter="url(#kg-overview-focus-glow)"
                >
                  <animate attributeName="stroke-dashoffset" values="0;-14" dur="1.8s" repeatCount="indefinite" />
                </rect>
                <rect
                  x={graphOverview.focusBounds.x + 2.5}
                  y={graphOverview.focusBounds.y + 2.5}
                  width={Math.max(2, graphOverview.focusBounds.width - 5)}
                  height={Math.max(2, graphOverview.focusBounds.height - 5)}
                  rx="5"
                  fill="none"
                  stroke={alpha('#fff', 0.22)}
                  strokeWidth="0.45"
                />
              </g>
            )}
            {graphOverview.edges.map(({ edge, source, target, color }) => {
              const sourceNode = overviewNodeById.get(source);
              const targetNode = overviewNodeById.get(target);
              if (!sourceNode || !targetNode) return null;
              const hovered = hoveredOverviewKind === 'relation' && hoveredOverviewId === edge.id;
              const active = hovered || selectedEdgeId === edge.id || activeGraphFocus.activeEdges.has(edge.id);
              return (
                <line
                  key={`overview-edge-${edge.id}`}
                  data-kg-overview-edge={edge.id}
                  x1={sourceNode.x}
                  y1={sourceNode.y}
                  x2={targetNode.x}
                  y2={targetNode.y}
                  stroke={color}
                  strokeWidth={active ? hovered ? 2.35 : 1.9 : 0.72}
                  strokeOpacity={active ? hovered ? 0.95 : 0.82 : 0.18}
                  strokeLinecap="round"
                  onMouseEnter={() => {
                    setGraphOverviewHover({
                      kind: 'relation',
                      id: edge.id,
                      label: `${sourceNode.node.label} -> ${targetNode.node.label}`,
                      secondary: `${getTypeLabel(edge.type || edge.label)} / ${Math.max(1, Number(edge.relationCount || 1)).toLocaleString()} relation${Number(edge.relationCount || 1) === 1 ? '' : 's'}`,
                      color,
                      relationType: edge.type || edge.label,
                    });
                  }}
                  onMouseLeave={() => clearOverviewHover(edge.id)}
                  onClick={(event) => {
                    event.stopPropagation();
                    setSelectedEdgeId(edge.id);
                    setSelectedNodeId(null);
                    setSelectedEntity(null);
                    setExpandedRelationshipBundleId(null);
                    setSelectionPanelExpanded(false);
                    setTabValue(2);
                    focusGraphOverviewEdge(edge);
                  }}
                  onDoubleClick={(event) => {
                    event.stopPropagation();
                    focusGraphOverviewEdge(edge);
                    handleLinkDive(edge);
                    setGraphOverviewOpen(false);
                  }}
                  style={{ cursor: 'pointer' }}
                >
                  <title>{`${getTypeLabel(edge.type || edge.label)}: ${source} to ${target}`}</title>
                </line>
              );
            })}
            {graphOverview.cameraTarget && (
              <g
                data-kg-overview-camera="true"
                data-kg-overview-camera-zoom={graphOverview.cameraTarget.zoomTier}
                pointerEvents="none"
              >
                <circle
                  cx={graphOverview.cameraTarget.x}
                  cy={graphOverview.cameraTarget.y}
                  r={graphOverview.cameraTarget.r}
                  fill={alpha(accentColors.info, graphOverview.cameraTarget.moving ? 0.09 : 0.055)}
                  stroke={alpha(accentColors.info, graphOverview.cameraTarget.moving ? 0.78 : 0.54)}
                  strokeWidth="0.9"
                  strokeDasharray="3 3"
                >
                  {graphOverview.cameraTarget.moving && <animate attributeName="stroke-dashoffset" values="0;-12" dur="1s" repeatCount="indefinite" />}
                </circle>
                <line
                  x1={graphOverview.cameraTarget.x - Math.min(6, graphOverview.cameraTarget.r * 0.42)}
                  y1={graphOverview.cameraTarget.y}
                  x2={graphOverview.cameraTarget.x + Math.min(6, graphOverview.cameraTarget.r * 0.42)}
                  y2={graphOverview.cameraTarget.y}
                  stroke={alpha(accentColors.info, 0.72)}
                  strokeWidth="0.75"
                  strokeLinecap="round"
                />
                <line
                  x1={graphOverview.cameraTarget.x}
                  y1={graphOverview.cameraTarget.y - Math.min(6, graphOverview.cameraTarget.r * 0.42)}
                  x2={graphOverview.cameraTarget.x}
                  y2={graphOverview.cameraTarget.y + Math.min(6, graphOverview.cameraTarget.r * 0.42)}
                  stroke={alpha(accentColors.info, 0.72)}
                  strokeWidth="0.75"
                  strokeLinecap="round"
                />
              </g>
            )}
            {graphOverview.nodes.map((overviewNode) => {
              const { node, x, y, r, color, selected, focused } = overviewNode;
              const visualType = getEntityVisualType(node.type, node.label);
              const hovered = hoveredOverviewKind === 'entity' && hoveredOverviewId === node.id;
              return (
              <g
                key={`overview-node-${node.id}`}
                data-kg-overview-node={node.id}
                data-kg-overview-node-type={visualType}
                data-kg-overview-node-hovered={hovered ? 'true' : 'false'}
                onMouseEnter={() => {
                  setHoveredNodeId(node.id);
                  setGraphOverviewHover({
                    kind: 'entity',
                    id: node.id,
                    label: node.label,
                    secondary: `${getEntityDisplayTypeLabel(node.type, node.label)} / ${formatNumber(graphOverviewTopology.metrics.get(node.id)?.links || 0)} links`,
                    color,
                    entityType: node.type,
                  });
                }}
                onMouseLeave={() => {
                  setHoveredNodeId((current) => (current === node.id ? null : current));
                  clearOverviewHover(node.id);
                }}
                onClick={(event) => {
                  event.stopPropagation();
                  setSelectedEdgeId(null);
                  setSelectedNodeId(node.id);
                  const entity = entities.find((item) => item.id === node.id);
                  if (entity) setSelectedEntity(entity);
                  setExpandedRelationshipBundleId(null);
                  setSelectionPanelExpanded(false);
                  setTabValue(2);
                  focusGraphOverviewNode(node.id);
                }}
                onDoubleClick={(event) => {
                  event.stopPropagation();
                  focusGraphOverviewNode(node.id);
                  handleNodeDive(node);
                  setGraphOverviewOpen(false);
                }}
                style={{ cursor: 'pointer' }}
              >
                {(selected || focused || hovered) && (
                  <circle
                    cx={x}
                    cy={y}
                    r={r + (hovered ? 5.8 : 4)}
                    fill="none"
                    stroke={color}
                    strokeWidth={selected || hovered ? 1.6 : 1}
                    strokeOpacity={selected || hovered ? 0.72 : 0.36}
                  >
                    {hovered && <animate attributeName="r" values={`${r + 4.8};${r + 7};${r + 4.8}`} dur="1.45s" repeatCount="indefinite" />}
                  </circle>
                )}
                {renderOverviewNodeGlyph({ ...overviewNode, hovered })}
                <title>{`${node.label} (${getEntityDisplayTypeLabel(node.type, node.label)})`}</title>
              </g>
              );
            })}
          </Box>
          <Collapse in={Boolean(graphOverviewHover)} timeout={140} unmountOnExit>
            <Box
              data-kg-overview-hover-readout="true"
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.55,
                minWidth: 0,
                px: 0.58,
                py: 0.48,
                borderRadius: '8px',
                border: '1px solid',
                borderColor: alpha(graphOverviewHover?.color || accentColors.primary, 0.22),
                bgcolor: alpha(graphOverviewHover?.color || accentColors.primary, 0.075),
                animation: 'kgOverviewReadoutIn 140ms ease-out',
              }}
            >
              <Box
                sx={{
                  width: 22,
                  height: 22,
                  borderRadius: '7px',
                  display: 'grid',
                  placeItems: 'center',
                  flexShrink: 0,
                  color: graphOverviewHover?.color || accentColors.primary,
                  bgcolor: alpha(graphOverviewHover?.color || accentColors.primary, 0.12),
                  border: '1px solid',
                  borderColor: alpha(graphOverviewHover?.color || accentColors.primary, 0.26),
                }}
              >
                {graphOverviewHover?.kind === 'relation'
                  ? renderRelationSymbolMark(graphOverviewHover.relationType, graphOverviewHover.color, { selected: true, size: 18 })
                  : getEntityIcon(graphOverviewHover?.entityType, graphOverviewHover?.color || accentColors.primary, graphOverviewHover?.label)}
              </Box>
              <Box sx={{ minWidth: 0, flex: 1 }}>
                <Typography variant="caption" noWrap sx={{ display: 'block', fontWeight: 900, lineHeight: 1.12 }}>
                  {graphOverviewHover?.label}
                </Typography>
                <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block', lineHeight: 1.12, fontSize: '0.58rem' }}>
                  {graphOverviewHover?.secondary}
                </Typography>
              </Box>
            </Box>
          </Collapse>
        </Box>
      );
    };

      const getEntitySearchPreviewBundles = (entity: Entity, limit = 4) => {
        const entityKeys = new Set([entity.id, entity.name].filter((value): value is string => Boolean(value)));
        const bundles = new Map<string, {
          otherId: string;
          otherEntity?: Entity;
          primaryEdge: GraphEdge;
          relationCount: number;
          totalWeight: number;
          inCount: number;
          outCount: number;
          typeCounts: Map<string, number>;
        }>();

        visualGraph.edges.forEach((edge) => {
          const sourceMatches = entityKeys.has(edge.source);
          const targetMatches = entityKeys.has(edge.target);
          if (!sourceMatches && !targetMatches) return;

          const otherId = sourceMatches ? edge.target : edge.source;
          if (!otherId || entityKeys.has(otherId)) return;

          const relationTypes = getLinkRelationTypeCounts(edge, 4);
          const relationCount = Math.max(
            1,
            Number(edge.relationCount || 0) ||
            relationTypes.reduce((sum, item) => sum + item.count, 0)
          );
          const existing = bundles.get(otherId) || {
            otherId,
            otherEntity: entityByLookupKey.get(otherId),
            primaryEdge: edge,
            relationCount: 0,
            totalWeight: 0,
            inCount: 0,
            outCount: 0,
            typeCounts: new Map<string, number>(),
          };
          const existingScore = Number(existing.primaryEdge.relationCount || 1) * 10 + Number(existing.primaryEdge.weight || 0);
          const nextScore = relationCount * 10 + Number(edge.weight || 0);

          if (nextScore > existingScore) {
            existing.primaryEdge = edge;
          }
          existing.relationCount += relationCount;
          existing.totalWeight += Number(edge.weight || 0);
          if (sourceMatches) {
            existing.outCount += relationCount;
          } else {
            existing.inCount += relationCount;
          }
          relationTypes.forEach((item) => {
            const type = normalizeType(item.type);
            existing.typeCounts.set(type, (existing.typeCounts.get(type) || 0) + item.count);
          });
          if (!relationTypes.length) {
            const type = normalizeType(edge.type || edge.label);
            existing.typeCounts.set(type, (existing.typeCounts.get(type) || 0) + relationCount);
          }
          bundles.set(otherId, existing);
        });

        return Array.from(bundles.values())
          .map((bundle) => {
            const types = Array.from(bundle.typeCounts.entries())
              .map(([type, count]) => ({ type, count, color: getRelationshipColor(type) }))
              .sort((a, b) => b.count - a.count || a.type.localeCompare(b.type));
            return {
              ...bundle,
              otherEntity: bundle.otherEntity || entityByLookupKey.get(bundle.otherId),
              types,
            };
          })
          .sort((a, b) =>
            b.relationCount - a.relationCount ||
            b.totalWeight - a.totalWeight ||
            (a.otherEntity?.name || a.otherId).localeCompare(b.otherEntity?.name || b.otherId)
          )
          .slice(0, limit);
      };

      const renderSearchEntityNeighborhood = (result: GraphSearchResult, compact = false): ReactNode => {
        if (result.kind !== 'entity' || !result.entity) return null;
        const bundles = getEntitySearchPreviewBundles(result.entity, compact ? 3 : 4);
        if (bundles.length === 0) return null;

        const color = result.color || getTypeColors(result.entity.entity_type, result.entity.name).main;
        return (
          <Box
            data-kg-search-entity-neighborhood="true"
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: compact ? 0.32 : 0.42,
              flexWrap: 'wrap',
              minWidth: 0,
              mt: compact ? 0.35 : 0.48,
              overflow: 'hidden',
              maxHeight: compact ? 46 : 54,
            }}
          >
            {bundles.map((bundle) => {
              const primaryType = bundle.types[0];
              const relationColor = primaryType?.color || getRelationshipColor(bundle.primaryEdge.type || bundle.primaryEdge.label);
              const otherEntity = bundle.otherEntity;
              const otherLabel = truncateGraphLabel(otherEntity?.name || bundle.otherId, compact ? 15 : 18);
              const directionLabel = bundle.inCount > 0 && bundle.outCount > 0
                ? 'both'
                : bundle.outCount > 0
                  ? 'out'
                  : 'in';

              return (
                <ButtonBase
                  key={`search-neighbor-${result.entity?.id}-${bundle.otherId}`}
                  data-kg-search-entity-neighbor="true"
                  aria-label={`Preview ${result.entity?.name} relationship with ${otherEntity?.name || bundle.otherId}`}
                  onFocus={() => {
                    setSearchPreviewNodeId(null);
                    setSearchPreviewEdgeId(bundle.primaryEdge.id);
                  }}
                  onBlur={() => {
                    setSearchPreviewNodeId(result.entity?.id || null);
                    setSearchPreviewEdgeId(null);
                  }}
                  onPointerEnter={() => {
                    setSearchPreviewNodeId(null);
                    setSearchPreviewEdgeId(bundle.primaryEdge.id);
                  }}
                  onPointerOver={() => {
                    setSearchPreviewNodeId(null);
                    setSearchPreviewEdgeId(bundle.primaryEdge.id);
                  }}
                  onPointerLeave={() => {
                    setSearchPreviewNodeId(result.entity?.id || null);
                    setSearchPreviewEdgeId(null);
                  }}
                  onMouseEnter={() => {
                    setSearchPreviewNodeId(null);
                    setSearchPreviewEdgeId(bundle.primaryEdge.id);
                  }}
                  onMouseOver={() => {
                    setSearchPreviewNodeId(null);
                    setSearchPreviewEdgeId(bundle.primaryEdge.id);
                  }}
                  onMouseLeave={() => {
                    setSearchPreviewNodeId(result.entity?.id || null);
                    setSearchPreviewEdgeId(null);
                  }}
                  onClick={(event) => {
                    event.stopPropagation();
                    handleLinkClick(bundle.primaryEdge);
                  }}
                  sx={{
                    minWidth: 0,
                    maxWidth: compact ? 156 : 196,
                    height: compact ? 27 : 31,
                    px: compact ? 0.45 : 0.55,
                    borderRadius: '8px',
                    border: '1px solid',
                    borderColor: alpha(relationColor, 0.24),
                    bgcolor: alpha(relationColor, 0.08),
                    color: 'text.primary',
                    display: 'flex',
                    alignItems: 'center',
                    gap: compact ? 0.34 : 0.42,
                    justifyContent: 'flex-start',
                    textAlign: 'left',
                    transition: 'transform 140ms ease, background-color 140ms ease, border-color 140ms ease',
                    '&:hover, &:focus-visible': {
                      transform: 'translateY(-1px)',
                      borderColor: alpha(relationColor, 0.46),
                      bgcolor: alpha(relationColor, 0.13),
                    },
                  }}
                >
                  {renderRelationSymbolMark(primaryType?.type || bundle.primaryEdge.type || bundle.primaryEdge.label, relationColor, {
                    selected: true,
                    size: compact ? 18 : 20,
                  })}
                  {otherEntity
                    ? renderEntitySymbolMark(otherEntity.entity_type, otherEntity.name, {
                      size: compact ? 18 : 20,
                    })
                    : <HubIcon sx={{ fontSize: compact ? 12 : 13, color: alpha(color, 0.82), flexShrink: 0 }} />}
                  <Typography variant="caption" noWrap sx={{ minWidth: 0, flex: 1, fontSize: compact ? '0.57rem' : '0.62rem', fontWeight: 900, lineHeight: 1 }}>
                    {renderHighlightedSearchText(otherLabel, relationColor)}
                  </Typography>
                  <Typography component="span" variant="caption" sx={{ flexShrink: 0, color: relationColor, fontSize: compact ? '0.51rem' : '0.54rem', fontWeight: 920, lineHeight: 1 }}>
                    {formatNumber(bundle.relationCount)} {directionLabel}
                  </Typography>
                </ButtonBase>
              );
            })}
          </Box>
        );
      };

      const renderSearchRelationRoute = (result: GraphSearchResult, compact = false): ReactNode => {
        if (result.kind !== 'relation' || !result.route) return null;

      const route = result.route;
      const sourceColors = getTypeColors(route.sourceType, route.sourceLabel);
      const targetColors = getTypeColors(route.targetType, route.targetLabel);
      const relationColor = result.color || getRelationshipColor(route.relationType);
      const sourceLabel = truncateGraphLabel(route.sourceLabel, compact ? 16 : 20);
      const targetLabel = truncateGraphLabel(route.targetLabel, compact ? 16 : 20);
      const relationLabel = truncateGraphLabel(route.relationLabel, compact ? 14 : 16);

      const endpointSx = (colors: { main: string; light: string }) => ({
        minWidth: 0,
        height: compact ? 25 : 29,
        px: compact ? 0.45 : 0.55,
        display: 'flex',
        alignItems: 'center',
        gap: 0.34,
        borderRadius: '7px',
        border: '1px solid',
        borderColor: alpha(colors.main, compact ? 0.2 : 0.26),
        bgcolor: alpha(colors.main, compact ? 0.055 : 0.075),
        color: 'text.primary',
      });

      return (
        <Box
          data-kg-search-relation-route="true"
          sx={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0, 1fr) auto minmax(0, 1fr)',
            alignItems: 'center',
            gap: compact ? 0.34 : 0.45,
            width: '100%',
            minWidth: 0,
          }}
        >
          <Box sx={endpointSx(sourceColors)}>
            {renderEntitySymbolMark(route.sourceType, route.sourceLabel, {
              size: compact ? 18 : 21,
            })}
            <Typography variant="caption" noWrap sx={{ minWidth: 0, fontWeight: 900, lineHeight: 1, fontSize: compact ? '0.63rem' : '0.68rem' }}>
              {renderHighlightedSearchText(sourceLabel, sourceColors.main)}
            </Typography>
          </Box>
          <Box
            sx={{
              height: compact ? 25 : 29,
              px: compact ? 0.44 : 0.58,
              display: 'flex',
              alignItems: 'center',
              gap: 0.3,
              borderRadius: '999px',
              border: '1px solid',
              borderColor: alpha(relationColor, compact ? 0.36 : 0.44),
              bgcolor: alpha(relationColor, compact ? 0.13 : 0.16),
              color: relationColor,
              minWidth: 0,
              maxWidth: compact ? 92 : 116,
            }}
          >
            {renderRelationSymbolMark(route.relationType, relationColor, {
              selected: true,
              size: compact ? 18 : 21,
            })}
            <Typography variant="caption" noWrap sx={{ minWidth: 0, fontWeight: 900, lineHeight: 1, fontSize: compact ? '0.58rem' : '0.62rem' }}>
              {renderHighlightedSearchText(relationLabel, relationColor)}
            </Typography>
            {!compact && route.relationCount > 1 && (
              <Typography component="span" variant="caption" sx={{ flexShrink: 0, fontWeight: 900, lineHeight: 1, fontSize: '0.55rem', opacity: 0.82 }}>
                {formatNumber(route.relationCount)}
              </Typography>
            )}
          </Box>
          <Box sx={endpointSx(targetColors)}>
            {renderEntitySymbolMark(route.targetType, route.targetLabel, {
              size: compact ? 18 : 21,
            })}
            <Typography variant="caption" noWrap sx={{ minWidth: 0, fontWeight: 900, lineHeight: 1, fontSize: compact ? '0.63rem' : '0.68rem' }}>
              {renderHighlightedSearchText(targetLabel, targetColors.main)}
            </Typography>
          </Box>
          </Box>
        );
      };

      const renderSearchResultActionBar = (result: GraphSearchResult): ReactNode => {
        const color = result.color || accentColors.primary;
        const resultType = result.kind === 'entity'
          ? normalizeType(result.entity?.entity_type)
          : normalizeType(result.route?.relationType || result.edge?.type || result.edge?.label);
        const resultTypeLabel = result.kind === 'entity'
          ? getEntityDisplayTypeLabel(result.entity?.entity_type, result.entity?.name || result.label)
          : getTypeLabel(resultType);
        const actionButtonSx = (actionColor = color) => ({
          width: 27,
          height: 27,
          borderRadius: '8px',
          color: actionColor,
          bgcolor: alpha(actionColor, 0.1),
          border: '1px solid',
          borderColor: alpha(actionColor, 0.22),
          transition: 'transform 140ms ease, background-color 140ms ease, border-color 140ms ease',
          '&:hover': {
            transform: 'translateY(-1px)',
            bgcolor: alpha(actionColor, 0.16),
            borderColor: alpha(actionColor, 0.4),
          },
        });
        const focusResultType = () => {
          setSearchTerm('');
          setSearchTrayOpen(false);
          setSearchTrayExpanded(false);
          setSearchPreviewNodeId(null);
          setSearchPreviewEdgeId(null);
          setSelectedEdgeId(null);
          setSelectedNodeId(null);
          setSelectedEntity(null);
          setExpandedRelationshipBundleId(null);
          setSelectionPanelExpanded(false);
          setInspectorOpen(false);
          setGraphScope('linked');
          setSelectedComponentId(null);
          setFocusNeighborhood(false);

          if (result.kind === 'entity') {
            setSelectedEntityTypes([resultType]);
            setSelectedRelationshipTypes([]);
            setTabValue(0);
            return;
          }

          setSelectedEntityTypes([]);
          setSelectedRelationshipTypes([resultType]);
          setTabValue(1);
        };

        return (
          <Box
            data-kg-search-result-actions="true"
            sx={{
              mt: 0.55,
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
              minWidth: 0,
            }}
          >
            <Box
              data-kg-search-result-facts="true"
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.35,
                flex: 1,
                minWidth: 0,
                overflow: 'hidden',
              }}
            >
              {(result.badges || []).slice(0, 4).map((badge) => {
                const badgeColor = badge.color || color;
                return (
                  <Chip
                    key={`active-search-fact-${result.id}-${badge.label}`}
                    size="small"
                    label={badge.label}
                    sx={{
                      height: 19,
                      maxWidth: { xs: 104, sm: 132 },
                      color: badgeColor,
                      bgcolor: alpha(badgeColor, 0.12),
                      border: '1px solid',
                      borderColor: alpha(badgeColor, 0.24),
                      '& .MuiChip-label': {
                        px: 0.55,
                        fontSize: '0.58rem',
                        fontWeight: 850,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                      },
                    }}
                  />
                );
              })}
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.35, flexShrink: 0 }}>
              <Tooltip title="Open result">
                <IconButton
                  size="small"
                  data-kg-search-open-result="true"
                  aria-label={`Open search result ${result.label}`}
                  onClick={() => commitSearchResult(result)}
                  sx={actionButtonSx(color)}
                >
                  <CenterIcon sx={{ fontSize: 15 }} />
                </IconButton>
              </Tooltip>
              <Tooltip title="Open details">
                <IconButton
                  size="small"
                  data-kg-search-open-details="true"
                  aria-label={`Open details for search result ${result.label}`}
                  onClick={() => commitSearchResult(result, true)}
                  sx={actionButtonSx(accentColors.info)}
                >
                  <InfoIcon sx={{ fontSize: 15 }} />
                </IconButton>
              </Tooltip>
              {resultType && (
                <Tooltip title={`Focus ${resultTypeLabel}`}>
                  <IconButton
                    size="small"
                    data-kg-search-focus-type="true"
                    aria-label={`Focus ${resultTypeLabel}`}
                    onClick={focusResultType}
                    sx={actionButtonSx(result.kind === 'entity'
                      ? getTypeColors(resultType, result.label).main
                      : getRelationshipColor(resultType))}
                  >
                    {result.kind === 'entity'
                      ? renderEntitySymbolMark(resultType, result.label, { selected: true, size: 21 })
                      : renderRelationSymbolMark(resultType, getRelationshipColor(resultType), { selected: true, size: 21 })}
                  </IconButton>
                </Tooltip>
              )}
            </Box>
          </Box>
        );
      };

      const graphHeaderActions = graphUnavailable && graphIsEmpty ? (
    <Button
      size="small"
      variant="outlined"
      startIcon={<RefreshIcon />}
      onClick={checkGraphRAGStatus}
      sx={{ height: 34, borderRadius: '8px', borderColor: 'divider', fontWeight: 800 }}
    >
      Retry
      </Button>
    ) : (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
        {!graphIsEmpty && (
          <ToggleButtonGroup
            exclusive
            size="small"
            value={graphScope}
            onChange={(_, value: GraphScope | null) => {
              if (value) setGraphScope(value);
            }}
            aria-label="Graph scope"
            sx={{
              height: 32,
              '& .MuiToggleButton-root': {
                height: 32,
                px: 1,
                borderColor: 'divider',
                color: 'text.secondary',
                fontSize: '0.72rem',
                fontWeight: 850,
                textTransform: 'none',
                '&.Mui-selected': {
                  color: accentColors.primary,
                  bgcolor: alpha(accentColors.primary, 0.16),
                  '&:hover': { bgcolor: alpha(accentColors.primary, 0.2) },
                },
              },
            }}
          >
            <ToggleButton value="core" aria-label="Show core component">
              Core
            </ToggleButton>
            <ToggleButton value="linked" aria-label="Show linked entities">
              Linked
            </ToggleButton>
            <ToggleButton value="all" aria-label="Show all entities">
              All
            </ToggleButton>
          </ToggleButtonGroup>
        )}

        <FormControl size="small" disabled={!graphRAGStatus.available || graphRAGDomains.length === 0} sx={{ minWidth: 164 }}>
        <Select
          value={selectedDomain}
          onChange={(e) => setSelectedDomain(e.target.value)}
          displayEmpty
          sx={{ height: 32, fontSize: '0.8125rem', borderRadius: '8px' }}
        >
          <MenuItem value="">All Domains</MenuItem>
          {graphRAGDomains.map(d => (
            <MenuItem key={d} value={d}>{d}</MenuItem>
          ))}
        </Select>
      </FormControl>

      {!graphIsEmpty && (
        <>
          <Tooltip title="Toggle physics">
            <IconButton
              size="small"
              aria-label="Toggle graph physics"
              onClick={() => setPhysicsEnabled(!physicsEnabled)}
              sx={{ ...commandIconButtonSx, color: physicsEnabled ? accentColors.success : 'text.secondary' }}
            >
              {physicsEnabled ? <PlayIcon sx={{ fontSize: 17 }} /> : <PauseIcon sx={{ fontSize: 17 }} />}
            </IconButton>
          </Tooltip>
          <Tooltip title="Graph display settings">
            <IconButton
              size="small"
              aria-label="Graph display settings"
              onClick={toggleGraphSettings}
              sx={commandIconButtonSx}
            >
              <SettingsIcon sx={{ fontSize: 17 }} />
            </IconButton>
          </Tooltip>
        </>
      )}

      <Tooltip title={graphUnavailable ? 'Retry GraphRAG status' : 'Reload graph'}>
        <span>
          <IconButton
            size="small"
            aria-label={graphUnavailable ? 'Retry GraphRAG status' : 'Reload graph'}
            onClick={graphUnavailable ? checkGraphRAGStatus : loadData}
            disabled={loading}
            sx={commandIconButtonSx}
          >
            <RefreshIcon sx={{ fontSize: 17 }} />
          </IconButton>
        </span>
      </Tooltip>
      <Tooltip title="Extract from text">
        <span>
          <IconButton
            size="small"
            aria-label="Extract entities from text"
            onClick={() => setExtractDialogOpen(true)}
            disabled={!graphRAGStatus.available}
            sx={commandIconButtonSx}
          >
            <UploadIcon sx={{ fontSize: 17 }} />
          </IconButton>
        </span>
      </Tooltip>
      <Tooltip title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen canvas'}>
        <IconButton
          size="small"
          aria-label={isFullscreen ? 'Exit fullscreen' : 'Open graph fullscreen'}
          onClick={() => setIsFullscreen(!isFullscreen)}
          sx={commandIconButtonSx}
        >
          {isFullscreen ? <FullscreenExitIcon sx={{ fontSize: 17 }} /> : <FullscreenIcon sx={{ fontSize: 17 }} />}
        </IconButton>
      </Tooltip>
    </Box>
  );
  const graphSummaryRow = showGraphSummaryRow ? (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
      <Chip
        size="small"
        label={graphStatusLabel}
        sx={{
          height: 22,
          fontSize: '0.68rem',
          fontWeight: 800,
          color: graphStatusColor,
          bgcolor: alpha(graphStatusColor, 0.1),
          border: `1px solid ${alpha(graphStatusColor, 0.28)}`,
        }}
      />
      {graphEntityCount > 0 && (
        <Chip size="small" label={`${formatNumber(graphEntityCount)} entities`} sx={{ height: 22, fontSize: '0.68rem', fontWeight: 800 }} />
      )}
      {graphRelationshipCount > 0 && (
        <Chip size="small" label={`${formatNumber(graphRelationshipCount)} stored edges`} sx={{ height: 22, fontSize: '0.68rem', fontWeight: 800 }} />
      )}
        {(graphFiltersActive || graphScopeActive || graphUsesConnectedCore) && (
        <Chip
          size="small"
          label={`${formatNumber(visibleGraph.nodes.length)} shown / ${formatNumber(visualGraph.edges.length)} links / ${formatNumber(visibleGraph.edges.length)} relations`}
          sx={{
            height: 22,
            fontSize: '0.68rem',
            fontWeight: 800,
            color: accentColors.primary,
            bgcolor: alpha(accentColors.primary, 0.1),
            border: `1px solid ${alpha(accentColors.primary, 0.28)}`,
          }}
        />
      )}
      {graphCommunityCount > 0 && (
        <Chip size="small" label={`${formatNumber(graphCommunityCount)} communities`} sx={{ height: 22, fontSize: '0.68rem', fontWeight: 800 }} />
      )}
      {entityTypeCount > 0 && (
        <Chip size="small" label={`${entityTypeCount} types`} sx={{ height: 22, fontSize: '0.68rem', fontWeight: 800 }} />
      )}
      {graphContextLabel && (
        <Typography variant="caption" color="text.secondary" noWrap sx={{ minWidth: 0 }}>
          {graphContextLabel}
        </Typography>
      )}
    </Box>
  ) : null;

  return (
    <PageShell
      sx={{
        ...(isFullscreen ? {
          position: 'fixed',
          inset: 0,
          zIndex: 2147483000,
          width: '100vw',
          maxWidth: '100vw',
          m: 0,
          p: '0 !important',
          boxSizing: 'border-box',
          animation: 'kgFullscreenIn 180ms ease-out both',
          '@keyframes kgFullscreenIn': {
            from: { opacity: 0.72, transform: 'scale(0.992)' },
            to: { opacity: 1, transform: 'scale(1)' },
          },
        } : {}),
        height: isFullscreen ? '100dvh' : useGraphCanvasLayout ? 'calc(100dvh - 56px)' : 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: isFullscreen ? 0 : 0.75,
        px: isFullscreen ? 0 : { xs: 1, sm: 1.25, md: 1.5 },
        py: isFullscreen ? 0 : 0.75,
        background: isFullscreen ? '#0f0f1a' : 'transparent',
        overflow: useGraphCanvasLayout ? 'hidden' : 'visible',
      }}
    >
      {error && (
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* GraphRAG Unavailable Warning */}
      {graphUnavailable && !isFullscreen && graphNodes.length > 0 && (
        <Alert
          severity="warning"
          action={
            <Button color="inherit" size="small" onClick={checkGraphRAGStatus}>
              Retry
            </Button>
          }
        >
          {graphRAGStatus.message || 'GraphRAG service is offline. Start the GraphRAG containers to extract or load entities.'}
        </Alert>
      )}

      {/* Loading indicator while checking status */}
      {graphRAGStatus.status === 'checking' && !isFullscreen && (
        <Alert
          severity="info"
          icon={<CircularProgress size={20} />}
        >
          Checking GraphRAG service status...
        </Alert>
      )}

      <Grid
        container
        spacing={isFullscreen ? 0 : 0.75}
        data-knowledge-graph-workspace
        data-kg-fullscreen={isFullscreen ? 'true' : 'false'}
        sx={{
          flex: useGraphCanvasLayout ? 1 : '0 0 auto',
          minHeight: useGraphCanvasLayout ? 0 : 'auto',
          overflow: useGraphCanvasLayout ? 'hidden' : 'visible',
        }}
      >
        {/* Graph Visualization */}
          <Grid
            item
            xs={12}
            lg={showInspector ? 8 : 12}
            xl={showInspector ? 9 : 12}
            sx={{
              minHeight: useGraphCanvasLayout ? 0 : 'auto',
              height: useGraphCanvasLayout ? '100%' : 'auto',
              display: 'flex',
            }}
          >
          <SectionCard
            title="Graph Explorer"
            subtitle={graphIsEmpty ? undefined : `${formatNumber(visibleGraph.nodes.length)} nodes, ${formatNumber(visualGraph.edges.length)} visual links`}
            icon={<ExploreIcon />}
            accentColor={accentColors.primary}
            fullHeight={useGraphCanvasLayout}
            hideHeader={!graphIsEmpty}
            action={graphIsEmpty ? graphHeaderActions : undefined}
            contentSx={!graphIsEmpty ? { p: 0, '&:last-child': { pb: 0 } } : undefined}
            sx={{
              position: 'relative',
              overflow: 'hidden',
              width: '100%',
              minHeight: 0,
              bgcolor: graphIsEmpty ? 'background.paper' : '#050816',
              borderColor: graphIsEmpty ? 'divider' : alpha('#fff', 0.1),
              borderRadius: isFullscreen ? 0 : '8px',
              ...(isFullscreen ? {
                borderColor: 'transparent',
                boxShadow: 'none',
              } : {}),
              ...(useGraphCanvasLayout ? {
                flex: 1,
                height: '100%',
                minHeight: 0,
                display: 'flex',
                flexDirection: 'column',
              } : {}),
              '& .MuiCardContent-root': {
                ...(useGraphCanvasLayout ? {
                  flex: 1,
                  minHeight: 0,
                  display: 'flex',
                  flexDirection: 'column',
                } : {}),
              },
              '& .MuiCardContent-root > .MuiBox-root:last-child': {
                ...(useGraphCanvasLayout ? {
                  flex: 1,
                  minHeight: 0,
                  display: 'flex',
                  flexDirection: 'column',
                } : {}),
              },
            }}
          >
            {!isFullscreen && graphSummaryRow}

            {/* Settings panel */}
              {settingsOpen && (
                <Box data-kg-settings-panel="true" sx={{
                position: 'absolute',
                top: { xs: 58, md: 52 },
                right: 12,
                width: { xs: 'calc(100% - 24px)', sm: 380, lg: 440, xl: 480 },
                maxWidth: 'calc(100% - 24px)',
                zIndex: 9,
                p: 0.85,
                ...graphOverlayPanelSx,
                  bgcolor: alpha('#050816', 0.94),
                  maxHeight: { xs: 'min(62dvh, calc(100% - 82px))', md: 'min(52dvh, calc(100% - 74px))' },
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden',
                }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, mb: 0.85 }}>
                  <Box sx={{ minWidth: 0 }}>
                    <Typography variant="body2" sx={{ fontWeight: 900, lineHeight: 1.1 }}>
                      Settings
                    </Typography>
                    <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block' }}>
                      {graphScopeLabel} / {graphDetailMode}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Button
                      size="small"
                      onClick={clearGraphFilters}
                      disabled={!graphFiltersActive}
                      sx={{ height: 28, borderRadius: '8px' }}
                    >
                      Clear
                    </Button>
                  <Tooltip title="Close settings">
                    <IconButton
                      size="small"
                      aria-label="Close graph display settings"
                      onClick={() => setSettingsOpen(false)}
                      sx={{ width: 28, height: 28, borderRadius: '8px', color: 'text.secondary' }}
                    >
                      <CloseIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  </Tooltip>
                  </Box>
                </Box>

                <Box sx={{ display: 'grid', gap: 0.65, minHeight: 0, overflowY: 'auto', overflowX: 'hidden', pr: 0.25, ...graphOverlayScrollSx }}>
                  {renderSettingsDisclosure(
                    'display',
                    'Display',
                    `${graphLabelModeSummary} / ${showEdgeLabels ? 'edge labels' : 'clean edges'} / ${particleFlow ? 'flow' : 'still'}`,
                    <Box sx={{ display: 'grid', gap: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
                        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 850 }}>
                          Detail
                        </Typography>
                        <ToggleButtonGroup
                          exclusive
                          size="small"
                          value={graphDetailMode}
                          onChange={(_, value: GraphDetailMode | null) => {
                            if (value) setGraphDetailMode(value);
                          }}
                          aria-label="Graph detail mode"
                          sx={{
                            height: 28,
                            '& .MuiToggleButton-root': {
                              height: 28,
                              px: 0.9,
                              borderColor: 'divider',
                              color: 'text.secondary',
                              fontSize: '0.68rem',
                              fontWeight: 850,
                              textTransform: 'none',
                              '&.Mui-selected': {
                                color: accentColors.primary,
                                bgcolor: alpha(accentColors.primary, 0.16),
                                '&:hover': { bgcolor: alpha(accentColors.primary, 0.2) },
                              },
                            },
                          }}
                        >
                          <ToggleButton value="auto" aria-label="Automatic detail">Auto</ToggleButton>
                          <ToggleButton value="performance" aria-label="Fast detail">Fast</ToggleButton>
                          <ToggleButton value="detail" aria-label="Rich detail">Rich</ToggleButton>
                        </ToggleButtonGroup>
                      </Box>
                      <Box
                        sx={{
                          display: 'grid',
                          gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, minmax(0, 1fr))' },
                          gap: 0.45,
                          alignItems: 'center',
                        }}
                      >
                        <Box sx={{ gridColumn: '1 / -1', display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
                          <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 850 }}>
                            Names
                          </Typography>
                          <ToggleButtonGroup
                            exclusive
                            size="small"
                            value={graphLabelModeValue}
                            onChange={(_, value: GraphLabelModeValue | null) => {
                              if (value) setGraphLabelMode(value);
                            }}
                            aria-label="Entity name density"
                            sx={{
                              height: 28,
                              '& .MuiToggleButton-root': {
                                height: 28,
                                px: 0.82,
                                borderColor: 'divider',
                                color: 'text.secondary',
                                fontSize: '0.65rem',
                                fontWeight: 850,
                                textTransform: 'none',
                                '&.Mui-selected': {
                                  color: accentColors.info,
                                  bgcolor: alpha(accentColors.info, 0.15),
                                  '&:hover': { bgcolor: alpha(accentColors.info, 0.2) },
                                },
                              },
                            }}
                          >
                            <ToggleButton value="hidden" aria-label="Hide entity names">Hidden</ToggleButton>
                            <ToggleButton value="focus" aria-label="Show focused entity names">Focus</ToggleButton>
                            <ToggleButton value="priority" aria-label="Show priority entity names">Priority</ToggleButton>
                            <ToggleButton value="rich" aria-label="Show rich entity names">Rich</ToggleButton>
                          </ToggleButtonGroup>
                        </Box>
                        <FormControlLabel
                          control={<Switch size="small" checked={showEdgeLabels} onChange={(e) => setShowEdgeLabels(e.target.checked)} />}
                          label="Edge labels"
                          sx={{ mr: 0, '& .MuiFormControlLabel-label': { fontSize: '0.78rem', fontWeight: 750 } }}
                        />
                        <FormControlLabel
                          control={<Switch size="small" checked={particleFlow} onChange={(e) => setParticleFlow(e.target.checked)} />}
                          label="Flow"
                          sx={{ mr: 0, '& .MuiFormControlLabel-label': { fontSize: '0.78rem', fontWeight: 750 } }}
                        />
                        <FormControlLabel
                          control={<Switch size="small" checked={focusNeighborhood} onChange={(e) => setFocusNeighborhood(e.target.checked)} />}
                          label="Neighbors"
                          sx={{ mr: 0, '& .MuiFormControlLabel-label': { fontSize: '0.78rem', fontWeight: 750 } }}
                        />
                        <FormControlLabel
                          control={<Switch size="small" checked={physicsEnabled} onChange={(e) => setPhysicsEnabled(e.target.checked)} />}
                          label="Physics"
                          sx={{ mr: 0, '& .MuiFormControlLabel-label': { fontSize: '0.78rem', fontWeight: 750 } }}
                        />
                      </Box>
                    </Box>
                  )}

                  {renderSettingsDisclosure(
                    'scope',
                    'Scope',
                    `${graphScopeLabel} / ${selectedDomain || 'all domains'}`,
                    <Box sx={{ display: 'grid', gap: 1 }}>
                      <ToggleButtonGroup
                        exclusive
                        size="small"
                        value={graphScope}
                        onChange={(_, value: GraphScope | null) => {
                          if (value) setGraphScope(value);
                        }}
                        aria-label="Graph scope"
                        sx={{
                          height: 30,
                          width: 'fit-content',
                          '& .MuiToggleButton-root': {
                            height: 30,
                            px: 0.9,
                            borderColor: 'divider',
                            color: 'text.secondary',
                            fontSize: '0.68rem',
                            fontWeight: 850,
                            textTransform: 'none',
                            '&.Mui-selected': {
                              color: accentColors.primary,
                              bgcolor: alpha(accentColors.primary, 0.16),
                              '&:hover': { bgcolor: alpha(accentColors.primary, 0.2) },
                            },
                          },
                        }}
                      >
                        <ToggleButton value="core" aria-label="Show core component">Core</ToggleButton>
                        <ToggleButton value="linked" aria-label="Show linked entities">Linked</ToggleButton>
                        <ToggleButton value="all" aria-label="Show all entities">All</ToggleButton>
                      </ToggleButtonGroup>
                      <FormControl size="small" disabled={!graphRAGStatus.available || graphRAGDomains.length === 0} sx={{ maxWidth: 260 }}>
                        <Select
                          value={selectedDomain}
                          onChange={(e) => setSelectedDomain(e.target.value)}
                          displayEmpty
                          sx={{ height: 32, fontSize: '0.8125rem', borderRadius: '8px' }}
                        >
                          <MenuItem value="">All Domains</MenuItem>
                          {graphRAGDomains.map(d => (
                            <MenuItem key={d} value={d}>{d}</MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Box>
                  )}

                  {entityTypeSummary.length > 0 && renderSettingsDisclosure(
                    'entities',
                    'Entity Types',
                    selectedEntityTypes.length > 0 ? `${selectedEntityTypes.length} selected` : `${entityTypeSummary.length} types`,
                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                      {entityTypeSummary.map(({ type, count }) => {
                        const colors = getTypeColors(type);
                        const active = selectedEntityTypes.length === 0 || selectedEntityTypes.includes(type);
                        return (
                          <Chip
                            key={type}
                            size="small"
                            icon={getEntityIcon(type, active ? colors.light : alpha(colors.light, 0.55))}
                            label={`${getTypeLabel(type)} ${count}`}
                            onClick={() => toggleEntityType(type)}
                            variant={selectedEntityTypes.includes(type) ? 'filled' : 'outlined'}
                            sx={{
                              height: 24,
                              opacity: active ? 1 : 0.45,
                              bgcolor: selectedEntityTypes.includes(type) ? alpha(colors.main, 0.22) : 'transparent',
                              borderColor: alpha(colors.main, active ? 0.65 : 0.28),
                              color: active ? colors.light : 'text.secondary',
                              '& .MuiChip-label': { px: 0.75, fontSize: '0.68rem', fontWeight: 800 },
                            }}
                          />
                        );
                      })}
                    </Box>
                  )}

                  {browsableComponentSummaries.length > 1 && renderSettingsDisclosure(
                    'components',
                    'Components',
                    selectedComponent ? selectedComponent.label : `${browsableComponentSummaries.length} visible groups`,
                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                      {browsableComponentSummaries.map((component) => {
                        const selected = selectedComponentId === component.id;
                        const dominantType = component.entityTypes[0]?.type || component.topNodeType || 'custom';
                        const colors = getTypeColors(dominantType, component.topNodeLabel);
                        const label = `${component.label.replace('Component ', 'C')} ${component.nodeCount} nodes`;
                        return (
                          <Chip
                            key={component.id}
                            size="small"
                            icon={getEntityIcon(dominantType, selected ? colors.light : alpha(colors.light, 0.62), component.topNodeLabel)}
                            label={component.topNodeLabel ? `${label} / ${truncateGraphLabel(component.topNodeLabel, 18)}` : label}
                            onClick={() => focusComponent(component)}
                            variant={selected ? 'filled' : 'outlined'}
                            sx={{
                              height: 24,
                              bgcolor: selected ? alpha(colors.main, 0.22) : 'transparent',
                              borderColor: alpha(colors.main, selected ? 0.7 : 0.32),
                              color: selected ? colors.light : 'text.secondary',
                              '& .MuiChip-label': { px: 0.75, fontSize: '0.68rem', fontWeight: 800 },
                            }}
                          />
                        );
                      })}
                      {componentSummaries.length > browsableComponentSummaries.length && (
                        <Chip
                          size="small"
                          label={`+${componentSummaries.length - browsableComponentSummaries.length} smaller`}
                          sx={{
                            height: 24,
                            bgcolor: alpha('#fff', 0.05),
                            color: 'text.secondary',
                            '& .MuiChip-label': { px: 0.75, fontSize: '0.68rem', fontWeight: 800 },
                          }}
                          />
                        )}
                    </Box>
                  )}

                  {relationshipTypeSummary.length > 0 && renderSettingsDisclosure(
                    'relationships',
                    'Relationships',
                    selectedRelationshipTypes.length > 0 ? `${selectedRelationshipTypes.length} selected` : `${relationshipTypeSummary.length} types`,
                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                      {relationshipTypeSummary.map(({ type, count, color }) => {
                        const active = selectedRelationshipTypes.length === 0 || selectedRelationshipTypes.includes(type);
                        return (
                            <Chip
                              key={type}
                              size="small"
                              icon={renderRelationSymbolMark(type, color, { muted: !active, selected: selectedRelationshipTypes.includes(type), size: 20 })}
                              label={`${getTypeLabel(type)} ${count}`}
                              onClick={() => toggleRelationshipType(type)}
                            variant={selectedRelationshipTypes.includes(type) ? 'filled' : 'outlined'}
                            sx={{
                              height: 24,
                              opacity: active ? 1 : 0.45,
                              bgcolor: selectedRelationshipTypes.includes(type) ? alpha(color, 0.22) : 'transparent',
                              borderColor: alpha(color, active ? 0.65 : 0.28),
                              color: active ? color : 'text.secondary',
                              '& .MuiChip-label': { px: 0.75, fontSize: '0.68rem', fontWeight: 800 },
                            }}
                          />
                        );
                      })}
                    </Box>
                  )}
                </Box>
              </Box>
            )}

            {/* Graph */}
            {loading ? (
              <Box sx={{ flex: 1, minHeight: 280, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <CircularProgress sx={{ color: accentColors.primary }} />
              </Box>
            ) : graphNodes.length === 0 ? (
              <Box
                data-knowledge-graph-empty-state
                sx={{
                  p: 1.25,
                  borderRadius: '8px',
                  border: '1px solid',
                  borderColor: graphUnavailable ? alpha(accentColors.warning, 0.35) : 'divider',
                  bgcolor: alpha('#000', 0.16),
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 1.25,
                  flexWrap: 'wrap',
                }}
              >
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 1.5,
                    flexWrap: { xs: 'wrap', sm: 'nowrap' },
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25, minWidth: 0 }}>
                    <Box
                      sx={{
                        width: 38,
                        height: 38,
                        borderRadius: '8px',
                        display: 'grid',
                        placeItems: 'center',
                        bgcolor: alpha(graphUnavailable ? accentColors.warning : accentColors.primary, 0.16),
                        color: graphUnavailable ? accentColors.warning : accentColors.primary,
                        flex: '0 0 auto',
                      }}
                    >
                      <HubIcon sx={{ fontSize: 22 }} />
                    </Box>
                    <Box sx={{ minWidth: 0 }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 800 }}>
                          {graphUnavailable ? 'GraphRAG offline' : 'No graph data yet'}
                        </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 680 }}>
	                        {graphUnavailable
	                          ? graphRAGStatus.message || 'Start the GraphRAG service, then refresh.'
	                          : 'Extract text or upload documents to populate entities.'}
                      </Typography>
                    </Box>
                  </Box>

                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap', justifyContent: { sm: 'flex-end' } }}>
                    {!graphUnavailable && (
                      <Button
                        size="small"
                        startIcon={<RefreshIcon />}
                        onClick={loadData}
                      >
                        Reload
                      </Button>
                    )}
                    {graphRAGStatus.available && (
                      <Button
                        size="small"
                        variant="outlined"
                        startIcon={<UploadIcon />}
                        onClick={() => setExtractDialogOpen(true)}
                      >
                        Extract
                      </Button>
                    )}
                  </Box>
                </Box>
              </Box>
            ) : (
              <Box
                sx={{
                  flex: 1,
                  minHeight: 0,
                  position: 'relative',
                  overflow: 'hidden',
                    '@keyframes kgOverlayIn': {
                      from: { opacity: 0, transform: 'translateY(-6px)' },
                      to: { opacity: 1, transform: 'translateY(0)' },
                    },
                    '@keyframes kgSelectionIn': {
                      from: { opacity: 0, transform: 'translate3d(8px, 8px, 0) scale(0.985)' },
                      to: { opacity: 1, transform: 'translate3d(0, 0, 0) scale(1)' },
                    },
                    '@keyframes kgAnchorPulse': {
                      '0%, 100%': { transform: 'scale(1)', opacity: 0.82 },
                      '50%': { transform: 'scale(1.18)', opacity: 1 },
                    },
                    '@keyframes kgStatusBreath': {
                      '0%, 100%': { transform: 'scale(1)', opacity: 0.78 },
                      '50%': { transform: 'scale(1.28)', opacity: 1 },
                    },
                    '@keyframes kgInspectorIn': {
                      from: { opacity: 0, transform: 'translate3d(10px, -4px, 0) scale(0.985)' },
                      to: { opacity: 1, transform: 'translate3d(0, 0, 0) scale(1)' },
                    },
                    '@keyframes kgToolFanIn': {
                      from: { opacity: 0, transform: 'translate3d(8px, -6px, 0) scale(0.96)' },
                      to: { opacity: 1, transform: 'translate3d(0, 0, 0) scale(1)' },
                    },
                    '@keyframes kgToolButtonIn': {
                      from: { opacity: 0, transform: 'translate3d(0, -4px, 0) scale(0.86)' },
                      to: { opacity: 1, transform: 'translate3d(0, 0, 0) scale(1)' },
                    },
                    '@keyframes kgOverviewReadoutIn': {
                      from: { opacity: 0, transform: 'translateY(-3px) scale(0.985)' },
                      to: { opacity: 1, transform: 'translateY(0) scale(1)' },
                    },
                    '@keyframes kgRouteCueIn': {
                      from: { opacity: 0, transform: 'translateX(-5px) scale(0.985)' },
                      to: { opacity: 1, transform: 'translateX(0) scale(1)' },
                    },
                  }}
                >
                  <ForceGraphVisualization
                  nodes={visualGraph.nodes}
                  edges={visualGraph.edges}
                  selectedNode={graphSelectedNodeId}
                    selectedEdge={graphSelectedEdgeId}
                    focusNodeId={focusNodeId}
                    hoveredNode={hoveredNodeId}
                  activeNodeIds={activeGraphFocus.activeNodes}
                  activeEdgeIds={activeGraphFocus.activeEdges}
                  pathFocusNodeIds={pathFocusNodeIds}
                  pathFocusVersion={pathFocusVersion}
                  searchTerm={searchTerm}
                  showLabels={showLabels}
                  showEdgeLabels={showEdgeLabels}
                    labelDensity={graphLabelDensity}
                    compactSelectedNodeLabel={Boolean(selectionOverlayOpen && selectedEntity && !selectedEdgeDetails)}
                    compactSelectedEdgeEndpointLabels={Boolean(selectionOverlayOpen && selectedEdgeDetails)}
                    selectionExpanded={selectionPanelExpanded}
                    detailMode={graphDetailMode}
                  cameraFocusRequest={graphCameraFocusRequest}
                  viewSignature={graphCanvasKey}
                  particleFlow={particleFlow}
                  onNodeClick={handleNodeClick}
                  onLinkClick={handleLinkClick}
                  onNodeDive={handleNodeDive}
                  onLinkDive={handleLinkDive}
                  onNodeHover={(node) => setHoveredNodeId(node?.id || null)}
                  onBackgroundClick={clearGraphCanvasContext}
                  onSelectionAnchorChange={handleSelectionAnchorChange}
                  onCameraLodChange={handleGraphCameraLodChange}
                  overlayReservedRects={graphOverlayReservedRects}
                  topology={graphOverviewTopology}
                  isFullscreen={isFullscreen}
                  physicsEnabled={physicsEnabled}
                  suppressHoverInteractions={graphOverlayInteractionActive}
                />
                <Box
                  data-kg-search-tray="true"
                  data-kg-search-active={searchTrayActive ? 'true' : 'false'}
                  data-kg-search-results-expanded={searchTrayExpanded ? 'true' : 'false'}
                  sx={{
                    position: 'absolute',
                    top: 12,
                    left: 12,
                    zIndex: 7,
                    width: {
                      xs: searchTrayActive ? 'calc(100% - 24px)' : 38,
                      md: searchTrayActive ? (searchQuery ? (searchTrayExpanded ? 440 : 340) : 320) : 38,
                      lg: searchTrayActive ? (searchQuery ? (searchTrayExpanded ? 'min(36vw, 520px)' : 360) : 340) : 38,
                      xl: searchTrayActive ? (searchQuery ? (searchTrayExpanded ? 'min(32vw, 560px)' : 420) : 380) : 38,
                      },
                      maxWidth: 'calc(100% - 24px)',
                      maxHeight: searchTrayActive
                        ? {
                          xs: searchQuery
                            ? searchTrayExpanded ? 'min(62dvh, calc(100dvh - 132px))' : 'min(42dvh, calc(100dvh - 188px))'
                            : 'calc(100dvh - 128px)',
                          md: searchQuery && searchTrayExpanded ? 'min(58dvh, calc(100dvh - 108px))' : 'calc(100dvh - 96px)',
                        }
                        : 38,
                      p: searchTrayActive ? { xs: 0.55, sm: 0.75 } : 0.35,
                      ...graphOverlayPanelSx,
                      display: 'flex',
                      flexDirection: 'column',
                      minHeight: 0,
                      overflow: 'hidden',
                      transition: 'width 190ms cubic-bezier(0.2, 0.8, 0.2, 1), padding 190ms ease, background-color 180ms ease',
                    }}
                  >
                  {!searchTrayActive ? (
                    <Tooltip title="Search graph (/)" disableHoverListener disableFocusListener disableTouchListener>
                      <IconButton
                          size="small"
                          data-kg-search-open="true"
                          aria-label="Open graph search"
                          onClick={openGraphSearch}
                        sx={{
                          ...graphOverlayButtonSx,
                          bgcolor: 'transparent',
                          borderColor: 'transparent',
                        }}
                      >
                        <SearchIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                    </Tooltip>
                  ) : (
                    <>
                      <TextField
                        fullWidth
                        size="small"
                        placeholder="Search entities or relations"
                        inputRef={searchInputRef}
                        inputProps={{ 'data-kg-search-input': 'true' }}
                        value={searchTerm}
                        onFocus={() => setSearchTrayOpen(true)}
                          onChange={(e) => {
                            const nextValue = e.target.value;
                            const hasQuery = Boolean(nextValue.trim());
                            setSearchTrayOpen(true);
                            setSearchTerm(nextValue);
                            setSearchTrayExpanded(hasQuery);
                            setHoveredNodeId(null);
                            if (!hasQuery) {
                              setSearchPreviewNodeId(null);
                              setSearchPreviewEdgeId(null);
                            }
                          }}
                        onKeyDown={(event) => {
                          if (event.key === 'Escape') {
                            if (searchQuery && searchTrayExpanded) {
                              event.preventDefault();
                              setSearchTrayExpanded(false);
                              return;
                            }
                            event.preventDefault();
                            if (searchQuery) {
                              setSearchTerm('');
                              setSearchTrayExpanded(false);
                              return;
                            }
                            setSearchTrayOpen(false);
                            setSearchTrayExpanded(false);
                            clearGraphCanvasContext();
                            return;
                          }
                          if (searchResults.length === 0) return;
                          if (event.key === 'ArrowDown') {
                            event.preventDefault();
                            previewSearchResult(searchResultIndex + 1);
                            return;
                          }
                          if (event.key === 'ArrowUp') {
                            event.preventDefault();
                            previewSearchResult(searchResultIndex - 1);
                            return;
                          }
                          if (event.key === 'Enter') {
                            event.preventDefault();
                            activateSearchResult(searchResultIndex, event.shiftKey);
                          }
                        }}
                        InputProps={{
                          startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.disabled', fontSize: 18 }} />,
                          endAdornment: searchTerm ? (
                            <Tooltip title="Clear search" disableHoverListener disableFocusListener disableTouchListener>
                              <IconButton
                                size="small"
                                aria-label="Clear graph search"
                                onClick={() => {
                                  setSearchTerm('');
                                  setSearchTrayOpen(false);
                                  setSearchTrayExpanded(false);
                                  setSearchPreviewNodeId(null);
                                  setSearchPreviewEdgeId(null);
                                }}
                                sx={{ width: 24, height: 24, color: 'text.secondary' }}
                              >
                                <CloseIcon sx={{ fontSize: 16 }} />
                              </IconButton>
                            </Tooltip>
                          ) : null,
                        }}
                        sx={{
                          '& .MuiOutlinedInput-root': {
                            height: 34,
                            bgcolor: alpha('#020617', 0.66),
                            '& fieldset': { borderColor: alpha('#fff', 0.12) },
                            '&:hover fieldset': { borderColor: alpha(accentColors.primary, 0.34) },
                            '&.Mui-focused fieldset': { borderColor: alpha(accentColors.primary, 0.7) },
                          },
                        }}
                      />

                      {searchQuery && (
                        <Box sx={{ mt: 0.65, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 0.6,
                          minWidth: 0,
                          flexWrap: 'nowrap',
                        }}
                      >
                        <Chip
                          size="small"
                          label={searchResults.length > 0 ? `${searchResultIndex + 1}/${searchResults.length}` : '0'}
                          sx={{
                            height: 22,
                            flexShrink: 0,
                            color: searchResults.length > 0 ? accentColors.primary : 'text.secondary',
                            bgcolor: searchResults.length > 0 ? alpha(accentColors.primary, 0.14) : alpha('#fff', 0.05),
                            '& .MuiChip-label': { px: 0.65, fontSize: '0.66rem', fontWeight: 850 },
                          }}
                        />
                        <Typography variant="caption" color="text.secondary" noWrap sx={{ flex: 1, minWidth: 0 }}>
                          {searchResults.length > 0
                            ? `${searchResultCounts.entities} entities / ${searchResultCounts.relations} relations`
                            : searchResultCounts.all > 0 && searchKindFilter !== 'all'
                              ? `No ${searchKindFilter === 'entity' ? 'entity' : 'relation'} matches`
                              : 'No matching entities or relations'}
                        </Typography>
                        <Tooltip title="Previous result" disableHoverListener disableFocusListener disableTouchListener>
                          <span>
                            <IconButton
                              size="small"
                              aria-label="Previous graph search result"
                              disabled={searchResults.length === 0}
                              onClick={activatePreviousSearchResult}
                              sx={graphOverlayButtonSx}
                            >
                              <PreviousResultIcon sx={{ fontSize: 18 }} />
                            </IconButton>
                          </span>
                        </Tooltip>
                        <Tooltip title="Next result" disableHoverListener disableFocusListener disableTouchListener>
                          <span>
                            <IconButton
                              size="small"
                              aria-label="Next graph search result"
                              disabled={searchResults.length === 0}
                              onClick={activateNextSearchResult}
                              sx={graphOverlayButtonSx}
                            >
                              <ExpandBundleIcon sx={{ fontSize: 18 }} />
                            </IconButton>
                          </span>
                        </Tooltip>
                        <Tooltip title={searchTrayExpanded ? 'Collapse results' : 'Show results'} disableHoverListener disableFocusListener disableTouchListener>
                          <span>
                            <IconButton
                              size="small"
                              data-kg-search-results-toggle="true"
                              aria-label={searchTrayExpanded ? 'Collapse graph search results' : 'Show graph search results'}
                              disabled={searchResults.length === 0}
                              onClick={() => {
                                setSearchTrayExpanded((value) => !value);
                                window.setTimeout(() => searchInputRef.current?.focus(), 0);
                              }}
                              sx={{
                                ...graphOverlayButtonSx,
                                color: searchTrayExpanded ? accentColors.primary : 'text.secondary',
                              }}
                            >
                              {searchTrayExpanded ? <PreviousResultIcon sx={{ fontSize: 18 }} /> : <ExpandBundleIcon sx={{ fontSize: 18 }} />}
                            </IconButton>
                          </span>
                        </Tooltip>
                        </Box>

                        {selectedSearchResult && renderSearchResultActionBar(selectedSearchResult)}

                        {selectedSearchResult && !searchTrayExpanded && (
                          <ButtonBase
                          data-kg-search-active-summary="true"
                          aria-label={`Open active graph search result ${selectedSearchResult.label}`}
                          onClick={() => activateSearchResult(searchResultIndex)}
                          sx={{
                            width: '100%',
                            mt: 0.55,
                            p: 0.48,
                            borderRadius: '8px',
                            border: '1px solid',
                            borderColor: alpha(selectedSearchResult.color, 0.28),
                            bgcolor: alpha(selectedSearchResult.color, 0.075),
                            color: 'text.primary',
                            display: 'flex',
                            alignItems: 'center',
                            gap: 0.55,
                            minWidth: 0,
                            textAlign: 'left',
                            transition: 'background-color 140ms ease, border-color 140ms ease, transform 140ms ease',
                            '&:hover': {
                              bgcolor: alpha(selectedSearchResult.color, 0.12),
                              borderColor: alpha(selectedSearchResult.color, 0.42),
                              transform: 'translateY(-1px)',
                            },
                          }}
                        >
                          {selectedSearchResult.kind === 'relation' && selectedSearchResult.route ? (
                            renderSearchRelationRoute(selectedSearchResult, true)
                          ) : (
                            <>
                              <Box
                                sx={{
                                  width: 24,
                                  height: 24,
                                  display: 'grid',
                                  placeItems: 'center',
                                  flexShrink: 0,
                                  borderRadius: '8px',
                                  color: selectedSearchResult.color,
                                  bgcolor: alpha(selectedSearchResult.color, 0.14),
                                  border: '1px solid',
                                  borderColor: alpha(selectedSearchResult.color, 0.28),
                                }}
                              >
                                {selectedSearchResult.kind === 'entity'
                                  ? renderEntitySymbolMark(selectedSearchResult.entity?.entity_type, selectedSearchResult.entity?.name || selectedSearchResult.label, {
                                    selected: true,
                                    size: 23,
                                  })
                                  : renderRelationSymbolMark(selectedSearchResult.edge?.type || selectedSearchResult.edge?.label, selectedSearchResult.color, {
                                    selected: true,
                                    size: 23,
                                  })}
                              </Box>
                              <Box sx={{ minWidth: 0, flex: 1 }}>
                                <Typography variant="caption" noWrap sx={{ display: 'block', fontWeight: 900, lineHeight: 1.14 }}>
                                  {renderHighlightedSearchText(selectedSearchResult.label, selectedSearchResult.color)}
                                </Typography>
                                <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block', lineHeight: 1.14, fontSize: '0.58rem' }}>
                                  {renderHighlightedSearchText(selectedSearchResult.snippet || selectedSearchResult.secondary, selectedSearchResult.color)}
                                </Typography>
                                {selectedSearchResult.kind === 'entity' && renderSearchEntityNeighborhood(selectedSearchResult, true)}
                              </Box>
                            </>
                          )}
                        </ButtonBase>
                      )}

                      <Collapse
                        in={Boolean(searchQuery && searchResultCounts.all > 0 && (searchTrayExpanded || searchKindFilter !== 'all'))}
                        timeout={150}
                        unmountOnExit
                      >
                        <Box
                          data-kg-search-kind-filter="true"
                          sx={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            gap: 0.65,
                            mt: 0.55,
                            minWidth: 0,
                          }}
                        >
                          <ToggleButtonGroup
                            exclusive
                            size="small"
                            value={searchKindFilter}
                            onChange={(_, value: GraphSearchKindFilter | null) => {
                              if (value) setSearchKindFilter(value);
                            }}
                            aria-label="Filter graph search results"
                            sx={{
                              height: 26,
                              maxWidth: '100%',
                              overflow: 'hidden',
                              '& .MuiToggleButton-root': {
                                height: 26,
                                minWidth: 0,
                                px: 0.65,
                                gap: 0.34,
                                borderColor: alpha('#fff', 0.1),
                                color: 'text.secondary',
                                fontSize: '0.61rem',
                                fontWeight: 850,
                                textTransform: 'none',
                                '&.Mui-selected': {
                                  color: accentColors.primary,
                                  bgcolor: alpha(accentColors.primary, 0.16),
                                  '&:hover': { bgcolor: alpha(accentColors.primary, 0.2) },
                                },
                                '&.Mui-disabled': {
                                  opacity: 0.36,
                                },
                              },
                            }}
                          >
                            <ToggleButton value="all" aria-label="Show all search results">
                              <SearchIcon sx={{ fontSize: 13 }} />
                              {formatNumber(searchResultCounts.all)}
                            </ToggleButton>
                            <ToggleButton value="entity" aria-label="Show entity search results" disabled={searchResultCounts.entities === 0}>
                              <HubIcon sx={{ fontSize: 13 }} />
                              {formatNumber(searchResultCounts.entities)}
                            </ToggleButton>
                            <ToggleButton value="relation" aria-label="Show relationship search results" disabled={searchResultCounts.relations === 0}>
                              <RelationIcon sx={{ fontSize: 13 }} />
                              {formatNumber(searchResultCounts.relations)}
                            </ToggleButton>
                          </ToggleButtonGroup>
                          {searchKindFilter !== 'all' && searchResults.length === 0 && (
                            <Typography variant="caption" color="text.disabled" noWrap sx={{ minWidth: 0, fontSize: '0.58rem', fontWeight: 800 }}>
                              No {searchKindFilter === 'entity' ? 'entities' : 'relations'}
                            </Typography>
                          )}
                        </Box>
                      </Collapse>

                      <Collapse
                        in={searchTrayExpanded && searchResultWindow.length > 0}
                        timeout={180}
                        unmountOnExit
                        sx={{ minWidth: 0, maxWidth: '100%', overflow: 'hidden' }}
                      >
                        <Box
                          data-kg-search-results-list="true"
                          sx={{
                            width: '100%',
                            maxWidth: '100%',
                            mt: { xs: 0.65, md: searchTrayExpanded ? 0.78 : 0.65 },
                            display: 'grid',
                            gridTemplateColumns: {
                              xs: 'minmax(0, 1fr)',
                              lg: searchTrayExpanded ? 'repeat(2, minmax(0, 1fr))' : 'minmax(0, 1fr)',
                              xl: searchTrayExpanded ? 'repeat(2, minmax(0, 1fr))' : 'minmax(0, 1fr)',
                            },
                            gap: { xs: 0.38, sm: 0.48, xl: searchTrayExpanded ? 0.58 : 0.48 },
                              maxHeight: {
                                xs: searchTrayExpanded ? 'min(40dvh, 300px)' : 'min(24dvh, 188px)',
                                md: searchTrayExpanded ? 'min(44dvh, 420px)' : 'min(54dvh, 430px)',
                              },
                              overflowY: 'auto',
                              overflowX: 'hidden',
                              ...graphOverlayScrollSx,
                              pr: 0.25,
                            }}
                        >
                          {searchResultWindow.map(({ result, index }) => {
                            const active = index === searchResultIndex;
                            const color = result.color || accentColors.primary;
                            const searchResultSummary = result.kind === 'relation' && !active
                              ? result.secondary
                              : result.snippet || result.secondary;
                            return (
                              <ButtonBase
                                key={result.id}
                                data-kg-search-result={result.kind}
                                data-kg-search-result-active={active ? 'true' : 'false'}
                                data-kg-search-active-result={active ? 'true' : undefined}
                                aria-current={active ? 'true' : undefined}
                                tabIndex={0}
                                onFocus={() => previewSearchResult(index)}
                                onPointerEnter={() => {
                                  if (index !== searchResultIndex) previewSearchResult(index);
                                }}
                                onPointerMove={() => {
                                  if (index !== searchResultIndex) previewSearchResult(index);
                                }}
                                onMouseEnter={() => {
                                  if (index !== searchResultIndex) previewSearchResult(index);
                                }}
                                onMouseDown={(event) => {
                                  if (event.button !== 0) return;
                                  event.preventDefault();
                                  setSearchResultIndex(index);
                                  commitSearchResult(result, false, { autoDetails: true });
                                }}
                                onClick={() => {
                                  setSearchResultIndex(index);
                                  commitSearchResult(result, false, { autoDetails: true });
                                }}
                                onDoubleClick={() => {
                                  setSearchResultIndex(index);
                                  commitSearchResult(result, true);
                                }}
                                sx={{
                                  gridColumn: {
                                    xs: 'auto',
                                    lg: searchTrayExpanded && active ? '1 / -1' : 'auto',
                                    xl: searchTrayExpanded && active ? '1 / -1' : 'auto',
                                  },
                                  display: 'flex',
                                  width: '100%',
                                  maxWidth: '100%',
                                  minWidth: 0,
                                  minHeight: {
                                    xs: active ? (result.kind === 'relation' ? 54 : 48) : 38,
                                    sm: active ? (result.kind === 'relation' ? 64 : 56) : 46,
                                    xl: active ? (result.kind === 'relation' ? 68 : 58) : 50,
                                  },
                                  justifyContent: 'flex-start',
                                  alignItems: 'flex-start',
                                  gap: { xs: 0.52, sm: 0.68 },
                                  px: { xs: 0.58, sm: 0.75, xl: searchTrayExpanded ? 0.85 : 0.75 },
                                  py: { xs: active ? 0.44 : 0.38, sm: active ? 0.58 : 0.5 },
                                  borderRadius: '8px',
                                  border: '1px solid',
                                  borderColor: active ? alpha(color, 0.54) : alpha('#fff', 0.08),
                                  bgcolor: active ? alpha(color, 0.13) : alpha('#020617', 0.34),
                                  color: 'text.primary',
                                  textAlign: 'left',
                                  overflow: 'hidden',
                                  transition: 'background-color 140ms ease, border-color 140ms ease, transform 140ms ease',
                                  '&:hover': {
                                    transform: 'translateX(1px)',
                                    borderColor: alpha(color, 0.44),
                                    bgcolor: alpha(color, 0.1),
                                  },
                                }}
                              >
                                <Box
                                  sx={{
                                    width: { xs: active ? 26 : 23, sm: active ? 28 : 25 },
                                    height: { xs: active ? 26 : 23, sm: active ? 28 : 25 },
                                    mt: active ? 0.08 : 0,
                                    borderRadius: '8px',
                                    display: 'grid',
                                    placeItems: 'center',
                                    flexShrink: 0,
                                    color,
                                    bgcolor: alpha(color, active ? 0.2 : 0.12),
                                    border: '1px solid',
                                    borderColor: alpha(color, active ? 0.46 : 0.24),
                                  }}
                                >
                                  {result.kind === 'entity'
                                    ? renderEntitySymbolMark(result.entity?.entity_type, result.entity?.name || result.label, {
                                      selected: active,
                                      size: active ? 24 : 21,
                                    })
                                    : renderRelationSymbolMark(result.edge?.type || result.edge?.label, color, {
                                      selected: active,
                                      size: active ? 24 : 21,
                                    })}
                                </Box>
                                <Box sx={{ minWidth: 0, maxWidth: '100%', flex: 1, overflow: 'hidden', display: 'grid', gap: active ? 0.24 : 0.12 }}>
                                  {active && result.kind === 'relation' && result.route ? (
                                    <Box sx={{ mb: 0.35 }}>
                                      {renderSearchRelationRoute(result)}
                                    </Box>
                                  ) : (
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.6, minWidth: 0 }}>
                                      <Typography variant="caption" noWrap sx={{ flex: 1, minWidth: 0, fontWeight: 900, lineHeight: 1.18, fontSize: { xs: '0.66rem', sm: '0.7rem' } }}>
                                        {renderHighlightedSearchText(result.label, color)}
                                      </Typography>
                                      <Typography
                                        component="span"
                                        variant="caption"
                                        sx={{
                                          flexShrink: 0,
                                          color,
                                          bgcolor: alpha(color, active ? 0.14 : 0.08),
                                          border: '1px solid',
                                          borderColor: alpha(color, active ? 0.3 : 0.18),
                                          borderRadius: '999px',
                                          px: 0.45,
                                          py: 0.12,
                                          fontSize: '0.55rem',
                                          fontWeight: 900,
                                          lineHeight: 1,
                                        }}
                                      >
                                        {result.kind === 'entity' ? 'Entity' : 'Relation'}
                                      </Typography>
                                    </Box>
                                  )}
                                  <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    noWrap={!active}
                                    sx={{
                                      display: active ? '-webkit-box' : 'block',
                                      WebkitLineClamp: active ? 2 : undefined,
                                      WebkitBoxOrient: active ? 'vertical' : undefined,
                                      maxWidth: '100%',
                                      overflow: 'hidden',
                                      textOverflow: 'ellipsis',
                                      lineHeight: { xs: 1.14, sm: 1.22 },
                                      fontSize: { xs: active ? '0.66rem' : '0.61rem', sm: active ? '0.68rem' : '0.63rem' },
                                    }}
                                  >
                                    {renderHighlightedSearchText(searchResultSummary, color)}
                                  </Typography>
                                  {active && result.kind === 'entity' && renderSearchEntityNeighborhood(result)}
                                  {active && result.badges && result.badges.length > 0 && (
                                    <Box sx={{ display: { xs: 'none', sm: 'flex' }, alignItems: 'center', gap: 0.35, flexWrap: 'wrap', mt: 0.42, minWidth: 0, maxWidth: '100%', overflow: 'hidden', maxHeight: 40 }}>
                                      {result.badges.slice(0, 3).map((badge) => {
                                        const badgeColor = badge.color || color;
                                        return (
                                          <Chip
                                            key={`${result.id}-${badge.label}`}
                                            data-kg-search-badge="true"
                                            size="small"
                                            label={badge.label}
                                            sx={{
                                              height: 17,
                                              maxWidth: result.kind === 'relation' ? 126 : 96,
                                              flexShrink: 0,
                                              color: badgeColor,
                                              bgcolor: alpha(badgeColor, active ? 0.15 : 0.09),
                                              border: '1px solid',
                                              borderColor: alpha(badgeColor, active ? 0.32 : 0.18),
                                              '& .MuiChip-label': {
                                                px: 0.5,
                                                fontSize: '0.56rem',
                                                fontWeight: 850,
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                              },
                                            }}
                                          />
                                        );
                                      })}
                                    </Box>
                                  )}
                                </Box>
                              </ButtonBase>
                            );
                          })}
                        </Box>
                      </Collapse>
                    </Box>
                      )}
                    </>
                  )}
                </Box>

                  <Box
                    data-kg-graph-tools="true"
                      sx={{
                        position: 'absolute',
                        top: { xs: searchTrayActive ? (searchQuery ? 112 : 62) : 12, sm: 12 },
                        right: 12,
                        left: 'auto',
                        zIndex: 7,
                        display: { xs: searchTrayActive ? 'none' : 'flex', sm: 'flex' },
                        flexDirection: 'column',
                        alignItems: 'flex-end',
                      gap: 0.5,
                      maxWidth: 'calc(100% - 24px)',
                      pointerEvents: 'none',
                    }}
                  >
                        <Box
                          data-kg-control-dock="true"
                          sx={{
                            alignSelf: 'flex-end',
                            display: { xs: graphLegendOpen ? 'none' : 'flex', sm: 'flex' },
                            alignItems: 'center',
                            justifyContent: 'flex-end',
                            gap: 0.25,
                            p: 0.28,
                            ...graphOverlayPanelSx,
                            pointerEvents: 'auto',
                          }}
                        >
                          <Tooltip
                            title={graphLegendOpen
                              ? 'Hide type lens'
                              : graphTypeLensFilterCount > 0
                                ? `${graphTypeLensFilterCount} active type ${graphTypeLensFilterCount === 1 ? 'filter' : 'filters'}`
                                : 'Show entity symbols and relation colors'}
                            disableInteractive
                          >
                            <ButtonBase
                              data-kg-semantic-lens-toggle="true"
                              data-kg-semantic-lens-strip="true"
                              data-kg-active-type-filters={graphTypeLensFilterCount || undefined}
                              aria-label={graphLegendOpen ? 'Hide graph type lens' : 'Show graph type lens'}
                              onClick={() => {
                                setSettingsOpen(false);
                                setGraphScopeOpen(false);
                                setGraphOverviewOpen(false);
                                setInspectorOpen(false);
                                setGraphStatsOpen(false);
                                setGraphStatsDetailsOpen(false);
                                setSelectionActionsOpen(false);
                                setGraphToolsOpen(false);
                                setGraphLegendOpen((open) => !open);
                              }}
                              sx={{
                                ...graphOverlayButtonSx,
                                position: 'relative',
                                overflow: 'visible',
                                width: {
                                  xs: 46,
                                  sm: graphLegendOpen || graphTypeLensFilterCount > 0 ? 146 : 132,
                                  xl: graphLegendOpen || graphTypeLensFilterCount > 0 ? 158 : 144,
                                },
                                minWidth: 0,
                                px: { xs: 0, sm: 0.62 },
                                display: 'grid',
                                gridTemplateColumns: { xs: '1fr', sm: '50px minmax(0, 1fr)' },
                                alignItems: 'center',
                                justifyItems: { xs: 'center', sm: 'start' },
                                gap: { xs: 0, sm: 0.5 },
                                color: graphLegendOpen || graphTypeLensFilterCount > 0 ? accentColors.warning : 'text.secondary',
                                borderColor: graphLegendOpen || graphTypeLensFilterCount > 0 ? alpha(accentColors.warning, 0.34) : alpha('#fff', 0.08),
                                bgcolor: graphLegendOpen
                                  ? alpha(accentColors.warning, 0.13)
                                  : graphTypeLensFilterCount > 0
                                    ? alpha(accentColors.warning, 0.09)
                                    : 'transparent',
                              }}
                            >
                              <Box
                                aria-hidden="true"
                                sx={{
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  width: { xs: '100%', sm: 50 },
                                  minWidth: 0,
                                }}
                              >
                              {visibleLegendEntityTypes.length > 0 ? (
                                <Box
                                  aria-hidden="true"
                                  sx={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    gap: 0.15,
                                    width: '100%',
                                  }}
                                >
                                  {visibleLegendEntityTypes.slice(0, 3).map(({ type }, index) => {
                                    return (
                                      <Box
                                        key={`semantic-lens-preview-${type}`}
                                        sx={{
                                          ml: index === 0 ? 0 : -0.48,
                                          display: 'grid',
                                          placeItems: 'center',
                                        }}
                                      >
                                        {renderEntitySymbolMark(type, undefined, {
                                          muted: !graphLegendOpen && selectedEntityTypes.length > 0 && !selectedEntityTypes.includes(type),
                                          selected: graphLegendOpen || selectedEntityTypes.includes(type),
                                          size: 24,
                                        })}
                                      </Box>
                                    );
                                  })}
                                </Box>
                              ) : (
                                <LabelIcon sx={{ fontSize: 17 }} />
                              )}
                              </Box>
                              <Box
                                sx={{
                                  display: { xs: 'none', sm: 'grid' },
                                  gap: 0.08,
                                  minWidth: 0,
                                  textAlign: 'left',
                                  lineHeight: 1,
                                }}
                              >
                                <Typography variant="caption" noWrap sx={{ minWidth: 0, fontSize: '0.62rem', fontWeight: 950, lineHeight: 1.05 }}>
                                  Types
                                </Typography>
                                <Typography variant="caption" noWrap sx={{ minWidth: 0, color: 'text.disabled', fontSize: '0.55rem', fontWeight: 850, lineHeight: 1.05 }}>
                                  {graphTypeLensFilterCount > 0
                                    ? `${graphTypeLensFilterCount} active`
                                    : `${legendEntityTypeSummary.length} entity / ${legendRelationshipTypeSummary.length} relation`}
                                </Typography>
                              </Box>
                              {graphTypeLensFilterCount > 0 && (
                                <Box
                                  data-kg-type-filter-badge="true"
                                  aria-hidden="true"
                                  sx={{
                                    position: 'absolute',
                                    right: -4,
                                    top: -5,
                                    minWidth: 16,
                                    height: 16,
                                    px: 0.28,
                                    borderRadius: '999px',
                                    display: 'grid',
                                    placeItems: 'center',
                                    color: '#fff',
                                    bgcolor: accentColors.warning,
                                    border: '1px solid',
                                    borderColor: alpha('#050816', 0.94),
                                    boxShadow: `0 5px 14px ${alpha(accentColors.warning, 0.34)}`,
                                    fontSize: '0.56rem',
                                    fontWeight: 950,
                                    lineHeight: 1,
                                  }}
                                >
                                  {graphTypeLensFilterCount}
                                </Box>
                              )}
                            </ButtonBase>
                          </Tooltip>
                          <Tooltip title={graphToolsOpen ? '' : 'Graph menu'} disableInteractive>
                            <IconButton
                            size="small"
                            data-kg-actions-toggle="true"
                            aria-label={graphToolsOpen ? 'Close graph menu' : 'Open graph menu'}
                            onClick={() => {
                              setSettingsOpen(false);
                              setGraphScopeOpen(false);
                              setGraphOverviewOpen(false);
                              setGraphLegendOpen(false);
                              setInspectorOpen(false);
                              setGraphStatsOpen(false);
                              setGraphStatsDetailsOpen(false);
                              setSelectionActionsOpen(false);
                            setGraphToolsOpen((open) => !open);
                          }}
                          sx={{ ...graphOverlayButtonSx, color: graphToolsOpen ? accentColors.primary : 'text.secondary' }}
                        >
                          <MoreIcon
                            sx={{
                              fontSize: 18,
                              transition: 'transform 160ms ease',
                              transform: graphToolsOpen ? 'rotate(90deg)' : 'rotate(0deg)',
                            }}
                            />
                          </IconButton>
                        </Tooltip>
                      </Box>

                    {!settingsOpen && (
                      <Collapse in={graphScopeOpen} timeout={180} unmountOnExit>
                        <Box
                          data-kg-scope-panel="true"
                          sx={{
                            alignSelf: { xs: 'stretch', md: 'flex-end' },
                            display: 'grid',
                            gap: 0.55,
                            p: 0.65,
                            ...graphOverlayPanelSx,
                            pointerEvents: 'auto',
                          }}
                        >
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.65, minWidth: 0 }}>
                            <ExploreIcon sx={{ fontSize: 16, color: accentColors.primary, flexShrink: 0 }} />
                            <Box sx={{ minWidth: 0, flex: 1 }}>
                              <Typography variant="caption" noWrap sx={{ display: 'block', fontWeight: 900, lineHeight: 1.1 }}>
                                Scope
                              </Typography>
                              <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', lineHeight: 1.1, fontSize: '0.58rem' }}>
                                {graphScopeLabel}
                              </Typography>
                            </Box>
                            <Tooltip title="Close scope" disableHoverListener disableFocusListener disableTouchListener>
                              <IconButton
                                size="small"
                                data-kg-scope-close="true"
                                aria-label="Close graph scope"
                                onClick={() => setGraphScopeOpen(false)}
                                sx={{ width: 24, height: 24, borderRadius: '8px', color: 'text.secondary', flexShrink: 0 }}
                              >
                                <CloseIcon sx={{ fontSize: 15 }} />
                              </IconButton>
                            </Tooltip>
                          </Box>
                          <ToggleButtonGroup
                            exclusive
                            size="small"
                            value={graphScope}
                            onChange={(_, value: GraphScope | null) => {
                              if (value) {
                                setGraphScope(value);
                                setGraphScopeOpen(false);
                              }
                            }}
                            aria-label="Graph scope"
                            sx={{
                              height: 28,
                              p: 0,
                              overflow: 'hidden',
                              flexShrink: 0,
                              '& .MuiToggleButton-root': {
                                height: 28,
                                px: 0.82,
                                borderColor: alpha('#fff', 0.1),
                                color: 'text.secondary',
                                fontSize: '0.66rem',
                                fontWeight: 850,
                                textTransform: 'none',
                                '&.Mui-selected': {
                                  color: accentColors.primary,
                                  bgcolor: alpha(accentColors.primary, 0.18),
                                  '&:hover': { bgcolor: alpha(accentColors.primary, 0.22) },
                                },
                              },
                            }}
                          >
                            <ToggleButton value="core" aria-label="Show core component">Core</ToggleButton>
                            <ToggleButton value="linked" aria-label="Show linked entities">Linked</ToggleButton>
                            <ToggleButton value="all" aria-label="Show all entities">All</ToggleButton>
                          </ToggleButtonGroup>
                        </Box>
                      </Collapse>
                    )}

                    {!settingsOpen && graphOverview.nodes.length > 0 && (
                      <Collapse in={graphOverviewOpen} timeout={180} unmountOnExit>
                        <Box sx={{ alignSelf: { xs: 'stretch', md: 'flex-end' }, pointerEvents: 'auto' }}>
                          {renderGraphOverview()}
                        </Box>
                      </Collapse>
                    )}

                    {!settingsOpen && (
                      <Collapse in={graphToolsOpen} timeout={180} unmountOnExit>
                        <Box
                          data-kg-tools-panel="true"
                          sx={{
                            alignSelf: { xs: 'stretch', md: 'flex-end' },
                            width: { xs: 'fit-content', md: 'min(364px, calc(100vw - 24px))', xl: 'min(396px, calc(100vw - 24px))' },
                            maxWidth: 'calc(100vw - 24px)',
                            p: { xs: 0.38, md: 0.68 },
                            display: 'grid',
                            gridTemplateColumns: { xs: 'repeat(3, 34px)', sm: 'repeat(6, 34px)', md: 'repeat(2, minmax(0, 1fr))' },
                            gap: { xs: 0.32, md: 0.58 },
                            ...graphOverlayPanelSx,
                            pointerEvents: 'auto',
                            animation: 'kgToolFanIn 180ms cubic-bezier(0.2, 0.8, 0.2, 1) both',
                            transformOrigin: 'top right',
                          }}
                        >
                          <Box sx={{ display: 'contents' }}>
                            <ButtonBase
                              data-kg-menu-action="fullscreen"
                              aria-label={isFullscreen ? 'Exit fullscreen canvas' : 'Open graph fullscreen'}
                              title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
                              onClick={() => {
                                setIsFullscreen(!isFullscreen);
                                setGraphToolsOpen(false);
                              }}
                              sx={graphMenuActionSx(isFullscreen, accentColors.primary)}
                            >
                              {isFullscreen ? <FullscreenExitIcon sx={{ fontSize: 17, flexShrink: 0 }} /> : <FullscreenIcon sx={{ fontSize: 17, flexShrink: 0 }} />}
                              <Box sx={{ minWidth: 0 }}>
                                <Typography variant="caption" noWrap sx={{ display: 'block', fontWeight: 900, lineHeight: 1.08 }}>
                                  {isFullscreen ? 'Exit' : 'Fullscreen'}
                                </Typography>
                                <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', fontSize: '0.58rem', lineHeight: 1.08 }}>
                                  Canvas
                                </Typography>
                              </Box>
                            </ButtonBase>

                            <ButtonBase
                              data-kg-menu-action="labels"
                              data-kg-label-mode={graphLabelModeValue}
                              aria-label="Cycle entity name density"
                              title={`Names: ${graphLabelModeLabel}`}
                              onClick={() => {
                                setSettingsOpen(false);
                                setGraphScopeOpen(false);
                                setGraphOverviewOpen(false);
                                setGraphLegendOpen(false);
                                setInspectorOpen(false);
                                setGraphStatsOpen(false);
                                setGraphStatsDetailsOpen(false);
                                setSelectionActionsOpen(false);
                                cycleGraphLabelMode();
                                setGraphToolsOpen(false);
                              }}
                              sx={graphMenuActionSx(showLabels, accentColors.info)}
                            >
                              <LabelIcon sx={{ fontSize: 17, flexShrink: 0 }} />
                              <Box sx={{ minWidth: 0 }}>
                                <Typography variant="caption" noWrap sx={{ display: 'block', fontWeight: 900, lineHeight: 1.08 }}>
                                  Names
                                </Typography>
                                <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', fontSize: '0.58rem', lineHeight: 1.08 }}>
                                  {graphLabelModeLabel}
                                </Typography>
                              </Box>
                            </ButtonBase>

                            <ButtonBase
                              data-kg-menu-action="scope"
                              data-kg-scope-toggle="true"
                              aria-label="Open graph scope controls"
                              title="Scope"
                              onClick={() => {
                                setSettingsOpen(false);
                                setGraphOverviewOpen(false);
                                setGraphLegendOpen(false);
                                setInspectorOpen(false);
                                setGraphStatsOpen(false);
                                setGraphStatsDetailsOpen(false);
                                setSelectionActionsOpen(false);
                                setGraphScopeOpen((open) => !open);
                                setGraphToolsOpen(false);
                              }}
                              sx={graphMenuActionSx(graphScopeOpen, accentColors.primary)}
                            >
                              <ExploreIcon sx={{ fontSize: 17, flexShrink: 0 }} />
                              <Box sx={{ minWidth: 0 }}>
                                <Typography variant="caption" noWrap sx={{ display: 'block', fontWeight: 900, lineHeight: 1.08 }}>
                                  Scope
                                </Typography>
                                <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', fontSize: '0.58rem', lineHeight: 1.08 }}>
                                  {graphScopeLabel}
                                </Typography>
                              </Box>
                            </ButtonBase>

                            <ButtonBase
                              data-kg-menu-action="overview"
                              data-kg-overview-toggle="true"
                              aria-label={graphOverviewOpen ? 'Hide topology overview' : 'Show topology overview'}
                              title="Overview"
                              onClick={() => {
                                setSettingsOpen(false);
                                setGraphScopeOpen(false);
                                setGraphLegendOpen(false);
                                setInspectorOpen(false);
                                setGraphStatsOpen(false);
                                setGraphStatsDetailsOpen(false);
                                setSelectionActionsOpen(false);
                                setGraphOverviewOpen((open) => !open);
                                setGraphToolsOpen(false);
                              }}
                              sx={graphMenuActionSx(graphOverviewOpen, accentColors.success)}
                            >
                              <HubIcon sx={{ fontSize: 17, flexShrink: 0 }} />
                              <Box sx={{ minWidth: 0 }}>
                                <Typography variant="caption" noWrap sx={{ display: 'block', fontWeight: 900, lineHeight: 1.08 }}>
                                  Overview
                                </Typography>
                                <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', fontSize: '0.58rem', lineHeight: 1.08 }}>
                                  Topology
                                </Typography>
                              </Box>
                            </ButtonBase>

                            <ButtonBase
                              data-kg-menu-action="legend"
                              data-kg-legend-toggle="true"
                              aria-label={graphLegendOpen ? 'Hide graph legend' : 'Show graph legend'}
                              title="Legend"
                              onClick={() => {
                                setSettingsOpen(false);
                                setGraphScopeOpen(false);
                                setGraphOverviewOpen(false);
                                setInspectorOpen(false);
                                setGraphStatsOpen(false);
                                setGraphStatsDetailsOpen(false);
                                setSelectionActionsOpen(false);
                                setGraphLegendOpen((open) => !open);
                                setGraphToolsOpen(false);
                              }}
                              sx={graphMenuActionSx(graphLegendOpen, accentColors.warning)}
                            >
                              <LabelIcon sx={{ fontSize: 17, flexShrink: 0 }} />
                              <Box sx={{ minWidth: 0 }}>
                                <Typography variant="caption" noWrap sx={{ display: 'block', fontWeight: 900, lineHeight: 1.08 }}>
                                  Legend
                                </Typography>
                                <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', fontSize: '0.58rem', lineHeight: 1.08 }}>
                                  Types
                                </Typography>
                              </Box>
                            </ButtonBase>

                            <ButtonBase
                              data-kg-menu-action="inspector"
                              aria-label={inspectorOpen ? 'Hide graph inspector' : 'Show graph inspector'}
                              title="Inspector"
                              onClick={() => {
                                setSettingsOpen(false);
                                setGraphScopeOpen(false);
                                setGraphOverviewOpen(false);
                                setGraphLegendOpen(false);
                                setGraphStatsOpen(false);
                                setGraphStatsDetailsOpen(false);
                                setSelectionActionsOpen(false);
                                setInspectorOpen((open) => {
                                  const next = !open;
                                  if (next && (selectedEntity || selectedEdgeDetails)) setTabValue(2);
                                  return next;
                                });
                                setGraphToolsOpen(false);
                              }}
                              sx={graphMenuActionSx(inspectorOpen, accentColors.info)}
                            >
                              <InfoIcon sx={{ fontSize: 17, flexShrink: 0 }} />
                              <Box sx={{ minWidth: 0 }}>
                                <Typography variant="caption" noWrap sx={{ display: 'block', fontWeight: 900, lineHeight: 1.08 }}>
                                  Inspector
                                </Typography>
                                <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', fontSize: '0.58rem', lineHeight: 1.08 }}>
                                  Details
                                </Typography>
                              </Box>
                            </ButtonBase>

                            <ButtonBase
                              data-kg-menu-action="display"
                              aria-label="Graph display settings"
                              title="Display"
                              onClick={() => {
                                setInspectorOpen(false);
                                setSelectionActionsOpen(false);
                                toggleGraphSettings();
                                setGraphToolsOpen(false);
                              }}
                              sx={graphMenuActionSx(settingsOpen, accentColors.purple)}
                            >
                              <SettingsIcon sx={{ fontSize: 17, flexShrink: 0 }} />
                              <Box sx={{ minWidth: 0 }}>
                                <Typography variant="caption" noWrap sx={{ display: 'block', fontWeight: 900, lineHeight: 1.08 }}>
                                  Display
                                </Typography>
                                <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', fontSize: '0.58rem', lineHeight: 1.08 }}>
                                  {graphDetailMode}
                                </Typography>
                              </Box>
                            </ButtonBase>
                          </Box>

                          <Divider sx={{ display: 'none' }} />

                          <Box sx={{ display: 'contents' }}>
                            <ButtonBase
                              data-kg-menu-action="reload"
                              aria-label="Reload graph"
                              title="Reload"
                              disabled={loading}
                              onClick={() => {
                                setGraphToolsOpen(false);
                                loadData();
                              }}
                              sx={graphMenuActionSx(false, accentColors.success)}
                            >
                              <RefreshIcon sx={{ fontSize: 17, flexShrink: 0 }} />
                              <Box sx={{ minWidth: 0 }}>
                                <Typography variant="caption" noWrap sx={{ display: 'block', fontWeight: 900, lineHeight: 1.08 }}>
                                  Reload
                                </Typography>
                                <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', fontSize: '0.58rem', lineHeight: 1.08 }}>
                                  Graph
                                </Typography>
                              </Box>
                            </ButtonBase>

                            <ButtonBase
                              data-kg-menu-action="extract"
                              aria-label="Extract entities from text"
                              title="Extract"
                              disabled={!graphRAGStatus.available}
                              onClick={() => {
                                setGraphToolsOpen(false);
                                setExtractDialogOpen(true);
                              }}
                              sx={graphMenuActionSx(false, accentColors.rose)}
                            >
                              <UploadIcon sx={{ fontSize: 17, flexShrink: 0 }} />
                              <Box sx={{ minWidth: 0 }}>
                                <Typography variant="caption" noWrap sx={{ display: 'block', fontWeight: 900, lineHeight: 1.08 }}>
                                  Extract
                                </Typography>
                                <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', fontSize: '0.58rem', lineHeight: 1.08 }}>
                                  Text
                                </Typography>
                              </Box>
                            </ButtonBase>
                          </Box>
                        </Box>
                      </Collapse>
                    )}
                      {!settingsOpen && (
                        <Collapse in={graphLegendOpen} timeout={180} unmountOnExit>
                          <Box
                            data-kg-legend-panel="true"
                            sx={{
                              alignSelf: { xs: 'stretch', md: 'flex-end' },
                              width: { xs: 'calc(100vw - 42px)', sm: 430, lg: 'min(48vw, 560px)', xl: 600 },
                              maxWidth: 'calc(100vw - 24px)',
                              maxHeight: { xs: 'min(52dvh, 430px)', md: 'min(62dvh, 560px)' },
                              overflowY: 'auto',
                              ...graphOverlayScrollSx,
                              p: { xs: 0.72, md: 0.9 },
                              display: 'grid',
                              gap: 0.85,
                              ...graphOverlayPanelSx,
                              bgcolor: alpha('#050816', 0.9),
                              pointerEvents: 'auto',
                            }}
                          >
                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 0.75, minWidth: 0 }}>
                              <Box
                                sx={{
                                  width: 34,
                                  height: 34,
                                  borderRadius: '8px',
                                  display: 'grid',
                                  placeItems: 'center',
                                  color: accentColors.warning,
                                  bgcolor: alpha(accentColors.warning, 0.12),
                                  border: '1px solid',
                                  borderColor: alpha(accentColors.warning, 0.24),
                                  flexShrink: 0,
                                }}
                              >
                                <LabelIcon sx={{ fontSize: 18 }} />
                              </Box>
                              <Box sx={{ minWidth: 0, flex: 1 }}>
                                <Typography variant="body2" noWrap sx={{ display: 'block', fontWeight: 900, lineHeight: 1.08 }}>
                                  Type lens
                                </Typography>
                                <Typography variant="caption" color="text.disabled" noWrap sx={{ display: { xs: 'none', sm: 'block' }, lineHeight: 1.15 }}>
                                  Entity symbols, relation colors, and active filters
                                </Typography>
                              </Box>
                              <Chip
                                size="small"
                                data-kg-type-preview-summary={semanticTypePreviewSummary ? 'true' : undefined}
                                label={semanticTypePreviewSummary
                                  ? `${semanticTypePreviewSummary.label}: ${semanticTypePreviewSummary.text}`
                                  : `${legendEntityTypeSummary.length} entity types / ${legendRelationshipTypeSummary.length} relation types`}
                                sx={{
                                  height: 21,
                                  color: semanticTypePreviewSummary ? semanticTypePreviewSummary.color : 'text.secondary',
                                  bgcolor: semanticTypePreviewSummary ? alpha(semanticTypePreviewSummary.color, 0.11) : alpha('#fff', 0.05),
                                  border: '1px solid',
                                  borderColor: semanticTypePreviewSummary ? alpha(semanticTypePreviewSummary.color, 0.32) : alpha('#fff', 0.1),
                                  '& .MuiChip-label': { px: 0.64, fontSize: '0.6rem', fontWeight: 850 },
                                }}
                              />
                              {graphFiltersActive && (
                                <Tooltip title="Clear graph filters">
                                  <IconButton
                                    size="small"
                                    data-kg-legend-clear="true"
                                    aria-label="Clear graph filters"
                                    onClick={clearGraphFilters}
                                    sx={{
                                      width: 26,
                                      height: 26,
                                      borderRadius: '8px',
                                      color: accentColors.info,
                                      bgcolor: alpha(accentColors.info, 0.1),
                                      border: '1px solid',
                                      borderColor: alpha(accentColors.info, 0.22),
                                      flexShrink: 0,
                                    }}
                                  >
                                    <RefreshIcon sx={{ fontSize: 15 }} />
                                  </IconButton>
                                </Tooltip>
                              )}
                              <Tooltip title="Close legend" disableHoverListener disableFocusListener disableTouchListener>
                                <IconButton
                                  size="small"
                                data-kg-legend-close="true"
                                aria-label="Close graph legend"
                                onClick={() => setGraphLegendOpen(false)}
                                sx={{ width: 24, height: 24, borderRadius: '8px', color: 'text.secondary', flexShrink: 0 }}
                              >
                                <CloseIcon sx={{ fontSize: 15 }} />
                                </IconButton>
                              </Tooltip>
                            </Box>

                            <Box
                              data-kg-type-lens-grid="true"
                              sx={{
                                display: 'grid',
                                gridTemplateColumns: { xs: '1fr', lg: 'repeat(2, minmax(0, 1fr))' },
                                gap: 0.75,
                                alignItems: 'start',
                              }}
                            >
                              {legendEntityTypeSummary.length > 0 && (
                                <Box data-kg-legend-entities="true" sx={{ display: 'grid', gap: 0.5, minWidth: 0 }}>
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.55, minWidth: 0 }}>
                                    <HubIcon sx={{ fontSize: 15, color: accentColors.info, flexShrink: 0 }} />
                                    <Typography variant="caption" noWrap sx={{ flex: 1, minWidth: 0, color: 'text.secondary', fontWeight: 900 }}>
                                      Entity symbols
                                    </Typography>
                                    <Typography variant="caption" color="text.disabled" noWrap sx={{ fontSize: '0.58rem', fontWeight: 850 }}>
                                      click to filter
                                    </Typography>
                                  </Box>
                                  <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2, minmax(0, 1fr))', sm: 'repeat(2, minmax(0, 1fr))' }, gap: 0.42 }}>
                                    {visibleLegendEntityTypes.map(({ type, count }) => {
                                      const colors = getTypeColors(type);
                                      const selected = selectedEntityTypes.includes(type);
                                      const active = selectedEntityTypes.length === 0 || selected;
                                      const previewing = semanticTypePreview?.kind === 'entity' && semanticTypePreview.type === type;
                                      const share = Math.max(8, Math.round((count / maxLegendEntityTypeCount) * 100));
                                      return (
                                        <ButtonBase
                                          key={`legend-entity-${type}`}
                                          data-kg-legend-entity-row="true"
                                          data-kg-legend-selected={selected ? 'true' : 'false'}
                                          data-kg-type-preview-active={semanticTypePreview?.kind === 'entity' && semanticTypePreview.type === type ? 'true' : undefined}
                                          aria-label={`${selected ? 'Remove' : 'Filter to'} ${getTypeLabel(type)} entities`}
                                          onClick={() => toggleEntityType(type)}
                                          onMouseEnter={() => setSemanticTypePreview({ kind: 'entity', type })}
                                          onMouseLeave={() => {
                                            setSemanticTypePreview((current) =>
                                              current?.kind === 'entity' && current.type === type ? null : current
                                            );
                                          }}
                                          onFocus={() => setSemanticTypePreview({ kind: 'entity', type })}
                                          onBlur={() => {
                                            setSemanticTypePreview((current) =>
                                              current?.kind === 'entity' && current.type === type ? null : current
                                            );
                                          }}
                                          sx={{
                                            minWidth: 0,
                                            minHeight: 46,
                                            px: 0.62,
                                            py: 0.55,
                                            display: 'grid',
                                            gridTemplateColumns: '30px minmax(0, 1fr)',
                                            alignItems: 'center',
                                            gap: 0.6,
                                            borderRadius: '8px',
                                            border: '1px solid',
                                            borderColor: alpha(colors.main, previewing ? 0.72 : selected ? 0.58 : active ? 0.25 : 0.13),
                                            bgcolor: previewing ? alpha(colors.main, 0.2) : selected ? alpha(colors.main, 0.17) : alpha(colors.main, active ? 0.075 : 0.035),
                                            color: active ? 'text.primary' : 'text.secondary',
                                            opacity: active ? 1 : 0.5,
                                            textAlign: 'left',
                                            overflow: 'hidden',
                                            boxShadow: previewing ? `0 0 0 1px ${alpha(colors.light, 0.18)}, 0 10px 26px ${alpha(colors.main, 0.18)}` : 'none',
                                            transition: 'transform 150ms ease, background-color 150ms ease, border-color 150ms ease, opacity 150ms ease',
                                            '&:hover': {
                                              transform: 'translateY(-1px)',
                                              borderColor: alpha(colors.main, 0.48),
                                              bgcolor: alpha(colors.main, 0.13),
                                              opacity: 1,
                                            },
                                          }}
                                        >
                                          {renderEntitySymbolMark(type, undefined, {
                                            muted: !active,
                                            selected: selected || previewing,
                                            size: 30,
                                          })}
                                          <Box sx={{ minWidth: 0, display: 'grid', gap: 0.34 }}>
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.4, minWidth: 0 }}>
                                              <Typography variant="caption" noWrap sx={{ flex: 1, minWidth: 0, fontWeight: 900, lineHeight: 1.05 }}>
                                                {getTypeLabel(type)}
                                              </Typography>
                                              <Typography variant="caption" noWrap sx={{ color: colors.light, fontSize: '0.58rem', fontWeight: 900, lineHeight: 1 }}>
                                                {formatNumber(count)}
                                              </Typography>
                                            </Box>
                                            <Box sx={{ height: 3, borderRadius: '999px', overflow: 'hidden', bgcolor: alpha('#fff', 0.07) }}>
                                              <Box sx={{ width: `${share}%`, height: '100%', borderRadius: '999px', bgcolor: colors.main, opacity: active ? 0.86 : 0.48 }} />
                                            </Box>
                                          </Box>
                                        </ButtonBase>
                                      );
                                    })}
                                  </Box>
                                  {legendEntityTypeSummary.length > 8 && (
                                    <ButtonBase
                                      data-kg-legend-entities-toggle="true"
                                      onClick={() => setGraphLegendEntitiesExpanded((expanded) => !expanded)}
                                      sx={{
                                        minHeight: 28,
                                        px: 0.75,
                                        borderRadius: '8px',
                                        justifyContent: 'space-between',
                                        color: graphLegendEntitiesExpanded ? accentColors.primary : 'text.secondary',
                                        bgcolor: graphLegendEntitiesExpanded ? alpha(accentColors.primary, 0.11) : alpha('#fff', 0.045),
                                        border: '1px solid',
                                        borderColor: graphLegendEntitiesExpanded ? alpha(accentColors.primary, 0.3) : alpha('#fff', 0.1),
                                      }}
                                    >
                                      <Typography variant="caption" sx={{ fontWeight: 900 }}>
                                        {graphLegendEntitiesExpanded ? 'Show fewer entity types' : `Show ${legendEntityTypeSummary.length - 8} more entity types`}
                                      </Typography>
                                      {graphLegendEntitiesExpanded ? <PreviousResultIcon sx={{ fontSize: 17 }} /> : <ExpandBundleIcon sx={{ fontSize: 17 }} />}
                                    </ButtonBase>
                                  )}
                                </Box>
                              )}

                              {legendRelationshipTypeSummary.length > 0 && (
                                <Box data-kg-legend-relations="true" sx={{ display: 'grid', gap: 0.5, minWidth: 0 }}>
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.55, minWidth: 0 }}>
                                    <RelationIcon sx={{ fontSize: 15, color: accentColors.warning, flexShrink: 0 }} />
                                    <Typography variant="caption" noWrap sx={{ flex: 1, minWidth: 0, color: 'text.secondary', fontWeight: 900 }}>
                                      Relation colors
                                    </Typography>
                                    <Typography variant="caption" color="text.disabled" noWrap sx={{ fontSize: '0.58rem', fontWeight: 850 }}>
                                      click to filter
                                    </Typography>
                                  </Box>
                                  <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2, minmax(0, 1fr))', sm: 'repeat(2, minmax(0, 1fr))' }, gap: 0.42 }}>
                                    {visibleLegendRelationshipTypes.map(({ type, count, color }) => {
                                      const selected = selectedRelationshipTypes.includes(type);
                                      const active = selectedRelationshipTypes.length === 0 || selected;
                                      const previewing = semanticTypePreview?.kind === 'relation' && semanticTypePreview.type === type;
                                      const share = Math.max(8, Math.round((count / maxLegendRelationshipTypeCount) * 100));
                                      return (
                                        <ButtonBase
                                          key={`legend-relation-${type}`}
                                          data-kg-legend-relation-row="true"
                                          data-kg-legend-selected={selected ? 'true' : 'false'}
                                          data-kg-type-preview-active={semanticTypePreview?.kind === 'relation' && semanticTypePreview.type === type ? 'true' : undefined}
                                          aria-label={`${selected ? 'Remove' : 'Filter to'} ${getTypeLabel(type)} relationships`}
                                          onClick={() => toggleRelationshipType(type)}
                                          onMouseEnter={() => setSemanticTypePreview({ kind: 'relation', type })}
                                          onMouseLeave={() => {
                                            setSemanticTypePreview((current) =>
                                              current?.kind === 'relation' && current.type === type ? null : current
                                            );
                                          }}
                                          onFocus={() => setSemanticTypePreview({ kind: 'relation', type })}
                                          onBlur={() => {
                                            setSemanticTypePreview((current) =>
                                              current?.kind === 'relation' && current.type === type ? null : current
                                            );
                                          }}
                                          sx={{
                                            minWidth: 0,
                                            minHeight: 46,
                                            px: 0.62,
                                            py: 0.55,
                                            display: 'grid',
                                            gridTemplateColumns: '30px minmax(0, 1fr)',
                                            alignItems: 'center',
                                            gap: 0.6,
                                            borderRadius: '8px',
                                            border: '1px solid',
                                            borderColor: alpha(color, previewing ? 0.72 : selected ? 0.58 : active ? 0.25 : 0.13),
                                            bgcolor: previewing ? alpha(color, 0.19) : selected ? alpha(color, 0.16) : alpha(color, active ? 0.07 : 0.035),
                                            color: active ? 'text.primary' : 'text.secondary',
                                            opacity: active ? 1 : 0.5,
                                            textAlign: 'left',
                                            overflow: 'hidden',
                                            boxShadow: previewing ? `0 0 0 1px ${alpha(color, 0.18)}, 0 10px 26px ${alpha(color, 0.16)}` : 'none',
                                            transition: 'transform 150ms ease, background-color 150ms ease, border-color 150ms ease, opacity 150ms ease',
                                            '&:hover': {
                                              transform: 'translateY(-1px)',
                                              borderColor: alpha(color, 0.48),
                                              bgcolor: alpha(color, 0.12),
                                              opacity: 1,
                                            },
                                          }}
                                        >
                                          {renderRelationSymbolMark(type, color, {
                                            muted: !active,
                                            selected: selected || previewing,
                                            size: 30,
                                          })}
                                          <Box sx={{ minWidth: 0, display: 'grid', gap: 0.34 }}>
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.4, minWidth: 0 }}>
                                              <Typography variant="caption" noWrap sx={{ flex: 1, minWidth: 0, fontWeight: 900, lineHeight: 1.05 }}>
                                                {getTypeLabel(type)}
                                              </Typography>
                                              <Typography variant="caption" noWrap sx={{ color, fontSize: '0.58rem', fontWeight: 900, lineHeight: 1 }}>
                                                {formatNumber(count)}
                                              </Typography>
                                            </Box>
                                            <Box sx={{ height: 3, borderRadius: '999px', overflow: 'hidden', bgcolor: alpha('#fff', 0.07) }}>
                                              <Box sx={{ width: `${share}%`, height: '100%', borderRadius: '999px', bgcolor: color, opacity: active ? 0.86 : 0.48 }} />
                                            </Box>
                                          </Box>
                                        </ButtonBase>
                                      );
                                    })}
                                  </Box>
                                  {legendRelationshipTypeSummary.length > 8 && (
                                    <ButtonBase
                                      data-kg-legend-relations-toggle="true"
                                      onClick={() => setGraphLegendRelationsExpanded((expanded) => !expanded)}
                                      sx={{
                                        minHeight: 28,
                                        px: 0.75,
                                        borderRadius: '8px',
                                        justifyContent: 'space-between',
                                        color: graphLegendRelationsExpanded ? accentColors.warning : 'text.secondary',
                                        bgcolor: graphLegendRelationsExpanded ? alpha(accentColors.warning, 0.11) : alpha('#fff', 0.045),
                                        border: '1px solid',
                                        borderColor: graphLegendRelationsExpanded ? alpha(accentColors.warning, 0.3) : alpha('#fff', 0.1),
                                      }}
                                    >
                                      <Typography variant="caption" sx={{ fontWeight: 900 }}>
                                        {graphLegendRelationsExpanded ? 'Show fewer relation types' : `Show ${legendRelationshipTypeSummary.length - 8} more relation types`}
                                      </Typography>
                                      {graphLegendRelationsExpanded ? <PreviousResultIcon sx={{ fontSize: 17 }} /> : <ExpandBundleIcon sx={{ fontSize: 17 }} />}
                                    </ButtonBase>
                                  )}
                                </Box>
                              )}
                            </Box>
                          </Box>
                        </Collapse>
                      )}
                  </Box>

                  {inspectorOverlayOpen && (
                    <Box
                      data-kg-inspector-overlay="true"
                      sx={{
                        position: 'absolute',
                        top: { xs: searchTrayActive ? (searchQuery ? 164 : 112) : 62, md: 56 },
                        right: 12,
                        zIndex: 8,
                        width: { xs: 'calc(100% - 24px)', sm: 380, md: 440, lg: 520, xl: 600 },
                        maxWidth: 'calc(100% - 24px)',
                        maxHeight: { xs: 'min(50dvh, 400px)', md: 'min(68dvh, 720px)' },
                        display: 'flex',
                        flexDirection: 'column',
                        overflow: 'hidden',
                        ...graphOverlayPanelSx,
                        bgcolor: alpha('#050816', 0.88),
                        animation: 'kgInspectorIn 210ms cubic-bezier(0.2, 0.8, 0.2, 1)',
                        pointerEvents: 'auto',
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, px: 1, py: 0.85, borderBottom: `1px solid ${alpha('#fff', 0.08)}` }}>
                        <Box sx={{ minWidth: 0, flex: 1 }}>
                          <Typography variant="caption" sx={{ display: 'block', fontWeight: 900, lineHeight: 1.1 }}>
                            Graph Inspector
                          </Typography>
                          <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', lineHeight: 1.15 }}>
                            {activeFocusLabel}
                          </Typography>
                        </Box>
                        <Chip
                          size="small"
                          label={inspectorCountLabel}
                          sx={{ height: 20, '& .MuiChip-label': { px: 0.6, fontSize: '0.6rem', fontWeight: 850 } }}
                        />
                        <Tooltip title="Close inspector" disableHoverListener disableFocusListener disableTouchListener>
                          <IconButton
                            size="small"
                            aria-label="Close graph inspector"
                            onClick={() => setInspectorOpen(false)}
                            sx={{ width: 26, height: 26, borderRadius: '8px', color: 'text.secondary' }}
                          >
                            <CloseIcon sx={{ fontSize: 16 }} />
                          </IconButton>
                        </Tooltip>
                      </Box>

                      <Tabs
                        value={tabValue}
                        onChange={(_, value) => setTabValue(value)}
                        variant="fullWidth"
                        sx={{
                          minHeight: 32,
                          borderBottom: `1px solid ${alpha('#fff', 0.08)}`,
                          '& .MuiTab-root': {
                            minHeight: 32,
                            px: 0.65,
                            fontSize: '0.65rem',
                            fontWeight: 850,
                            textTransform: 'none',
                            color: 'text.secondary',
                          },
                          '& .Mui-selected': { color: accentColors.primary },
                          '& .MuiTabs-indicator': { height: 2, bgcolor: accentColors.primary },
                        }}
                      >
                        <Tab label="Entities" icon={<HubIcon sx={{ fontSize: 15 }} />} iconPosition="start" />
                        <Tab label="Relations" icon={<RelationIcon sx={{ fontSize: 15 }} />} iconPosition="start" />
                        <Tab label="Details" icon={<InfoIcon sx={{ fontSize: 15 }} />} iconPosition="start" />
                      </Tabs>

                        <Box sx={{ minHeight: 0, overflowY: 'auto', ...graphOverlayScrollSx, px: 0.75, py: 0.7 }}>
                        {tabValue === 0 && (
                          <Box sx={{ display: 'grid', gap: 0.45 }}>
                            {selectedEntityTypes.length > 0 && visibleEntityTypeSummary.length > 0 && (
                              <Box sx={{ display: 'flex', gap: 0.4, flexWrap: 'wrap', mb: 0.25 }}>
                                {visibleEntityTypeSummary.slice(0, 4).map(({ type, count }) => {
                                  const colors = getTypeColors(type);
                                  const selected = selectedEntityTypes.includes(type);
                                  const active = selectedEntityTypes.length === 0 || selected;
                                  return (
                                    <Chip
                                      key={`overlay-entity-filter-${type}`}
                                      size="small"
                                      icon={getEntityIcon(type, active ? colors.light : alpha(colors.light, 0.55))}
                                      label={`${getTypeLabel(type)} ${count}`}
                                      onClick={() => toggleEntityType(type)}
                                      variant={selected ? 'filled' : 'outlined'}
                                      sx={{
                                        height: 20,
                                        opacity: active ? 1 : 0.45,
                                        bgcolor: selected ? alpha(colors.main, 0.2) : alpha(colors.main, 0.07),
                                        borderColor: alpha(colors.main, active ? 0.44 : 0.2),
                                        color: active ? colors.light : 'text.secondary',
                                        '& .MuiChip-label': { px: 0.55, fontSize: '0.58rem', fontWeight: 850 },
                                        '& .MuiChip-icon': { ml: 0.5, mr: -0.25, color: 'inherit' },
                                      }}
                                    />
                                  );
                                })}
                              </Box>
                            )}
                            {filteredEntities.slice(0, inspectorOverlayItemLimit).map((entity) => {
                              const colors = getTypeColors(entity.entity_type, entity.name);
                              const degree = degreeMap.get(entity.id) || { in: 0, out: 0, total: 0 };
                              return (
                                <ButtonBase
                                  key={`overlay-entity-${entity.id}`}
                                  onClick={() => {
                                    setSelectedEdgeId(null);
                                    setSelectedEntity(entity);
                                    setSelectedNodeId(entity.id);
                                    setSelectionPanelExpanded(false);
                                    setTabValue(2);
                                  }}
                                  sx={{
                                    width: '100%',
                                    minHeight: 34,
                                    justifyContent: 'flex-start',
                                    alignItems: 'center',
                                    gap: 0.6,
                                    px: 0.65,
                                    py: 0.42,
                                    borderRadius: '8px',
                                    border: '1px solid',
                                    borderColor: selectedEntity?.id === entity.id ? alpha(colors.main, 0.42) : alpha('#fff', 0.07),
                                    bgcolor: selectedEntity?.id === entity.id ? alpha(colors.main, 0.12) : alpha('#020617', 0.24),
                                    color: 'text.primary',
                                    textAlign: 'left',
                                    transition: 'transform 140ms ease, border-color 140ms ease, background-color 140ms ease',
                                    '&:hover': {
                                      transform: 'translateX(1px)',
                                      borderColor: alpha(colors.main, 0.36),
                                      bgcolor: alpha(colors.main, 0.09),
                                    },
                                  }}
                                >
                                  {renderEntitySymbolMark(entity.entity_type, entity.name, {
                                    selected: selectedEntity?.id === entity.id,
                                    size: 24,
                                  })}
                                  <Box sx={{ minWidth: 0, flex: 1 }}>
                                    <Typography variant="caption" noWrap sx={{ display: 'block', fontWeight: 900, lineHeight: 1.12 }}>
                                      {entity.name}
                                    </Typography>
                                    <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', lineHeight: 1.12 }}>
                                      {getEntityDisplayTypeLabel(entity.entity_type, entity.name)}
                                    </Typography>
                                  </Box>
                                  <Chip
                                    size="small"
                                    label={degree.total}
                                    sx={{
                                      height: 18,
                                      minWidth: 25,
                                      color: colors.light,
                                      bgcolor: alpha(colors.main, 0.14),
                                      '& .MuiChip-label': { px: 0.48, fontSize: '0.56rem', fontWeight: 900 },
                                    }}
                                  />
                                </ButtonBase>
                              );
                            })}
                            {filteredEntities.length > inspectorOverlayItemLimit && (
                              <Typography variant="caption" color="text.disabled" sx={{ px: 0.5, pt: 0.25 }}>
                                +{formatNumber(filteredEntities.length - inspectorOverlayItemLimit)} more entities
                              </Typography>
                            )}
                          </Box>
                        )}

                        {tabValue === 1 && (
                          <Box sx={{ display: 'grid', gap: 0.45 }}>
                            {relationTypeMatrix.items.length > 0 && (
                              <Box
                                data-kg-relation-matrix="true"
                                sx={{
                                  display: 'grid',
                                  gap: 0.42,
                                  p: 0.52,
                                  borderRadius: '8px',
                                  border: '1px solid',
                                  borderColor: alpha(accentColors.primary, 0.16),
                                  bgcolor: alpha('#020617', 0.24),
                                  overflow: 'hidden',
                                }}
                              >
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.45, minWidth: 0 }}>
                                  <RelationIcon sx={{ fontSize: 13, color: accentColors.primary, flexShrink: 0 }} />
                                  <Typography variant="caption" noWrap sx={{ flex: 1, minWidth: 0, fontSize: '0.6rem', fontWeight: 900, lineHeight: 1 }}>
                                    Relation mix
                                  </Typography>
                                  <Chip
                                    size="small"
                                    label={`${formatNumber(relationTypeMatrix.total)} rel`}
                                    sx={{
                                      height: 18,
                                      color: accentColors.primary,
                                      bgcolor: alpha(accentColors.primary, 0.1),
                                      '& .MuiChip-label': { px: 0.5, fontSize: '0.56rem', fontWeight: 900 },
                                    }}
                                  />
                                </Box>
                                <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 0.32 }}>
                                  {relationTypeMatrix.items.slice(0, inspectorOverlayRelationTypeLimit).map((item) => {
                                    const selected = selectedRelationshipTypes.includes(item.type);
                                    return (
                                      <ButtonBase
                                        key={`overlay-relation-matrix-${item.type}`}
                                        data-kg-relation-matrix-item="true"
                                        data-kg-relation-matrix-selected={selected ? 'true' : 'false'}
                                        aria-label={`${selected ? 'Clear' : 'Focus'} ${getTypeLabel(item.type)} relationships`}
                                        onClick={() => toggleRelationshipTypeFocus(item.type)}
                                        sx={{
                                          minWidth: 0,
                                          minHeight: 30,
                                          px: 0.45,
                                          py: 0.34,
                                          display: 'grid',
                                          gap: 0.25,
                                          borderRadius: '7px',
                                          border: '1px solid',
                                          borderColor: alpha(item.color, selected ? 0.5 : 0.2),
                                          bgcolor: selected ? alpha(item.color, 0.15) : alpha(item.color, 0.06),
                                          color: 'text.primary',
                                          textAlign: 'left',
                                          overflow: 'hidden',
                                          '&:hover': {
                                            borderColor: alpha(item.color, 0.42),
                                            bgcolor: alpha(item.color, 0.11),
                                          },
                                        }}
                                      >
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.28, minWidth: 0 }}>
                                          {renderRelationSymbolMark(item.type, item.color, { selected, size: 18 })}
                                          <Typography variant="caption" noWrap sx={{ minWidth: 0, flex: 1, fontSize: '0.56rem', fontWeight: 900, lineHeight: 1.05 }}>
                                            {getTypeLabel(item.type)}
                                          </Typography>
                                          <Typography variant="caption" noWrap sx={{ color: item.color, fontSize: '0.54rem', fontWeight: 900, lineHeight: 1 }}>
                                            {formatNumber(item.count)}
                                          </Typography>
                                        </Box>
                                        <Box sx={{ height: 3, borderRadius: '999px', overflow: 'hidden', bgcolor: alpha('#fff', 0.08) }}>
                                          <Box
                                            sx={{
                                              width: `${Math.max(7, Math.round(item.intensity * 100))}%`,
                                              height: '100%',
                                              borderRadius: '999px',
                                              bgcolor: item.color,
                                              opacity: selected ? 0.95 : 0.76,
                                            }}
                                          />
                                        </Box>
                                      </ButtonBase>
                                    );
                                  })}
                                </Box>
                              </Box>
                            )}
                            {selectedRelationshipTypeSummary.length > 0 && (
                              <Box data-kg-active-relation-filter="true" sx={{ display: 'flex', gap: 0.4, flexWrap: 'wrap', mb: 0.25 }}>
                                {selectedRelationshipTypeSummary.slice(0, 3).map(({ type, count, color }) => (
                                  <Chip
                                    key={`overlay-active-relation-filter-${type}`}
                                    size="small"
                                    icon={renderRelationSymbolMark(type, color, { selected: true, size: 18 })}
                                    label={`${getTypeLabel(type)} ${count}`}
                                    onClick={() => {
                                      setSelectedEdgeId(null);
                                      setSelectedEntity(null);
                                      setSelectedNodeId(null);
                                      toggleRelationshipType(type);
                                    }}
                                    sx={{
                                      height: 20,
                                      color,
                                      bgcolor: alpha(color, 0.2),
                                      border: '1px solid',
                                      borderColor: alpha(color, 0.44),
                                      '& .MuiChip-label': { px: 0.55, fontSize: '0.58rem', fontWeight: 850 },
                                      '& .MuiChip-icon': { ml: 0.5, mr: -0.25, color: 'inherit' },
                                    }}
                                  />
                                ))}
                                <Chip
                                  size="small"
                                  label="Clear"
                                  onClick={() => setSelectedRelationshipTypes([])}
                                  sx={{
                                    height: 20,
                                    color: 'text.secondary',
                                    bgcolor: alpha('#fff', 0.06),
                                    border: '1px solid',
                                    borderColor: alpha('#fff', 0.12),
                                    '& .MuiChip-label': { px: 0.6, fontSize: '0.58rem', fontWeight: 850 },
                                  }}
                                />
                              </Box>
                            )}
                            {relationBrowserRows.slice(0, inspectorOverlayItemLimit).map((row) => {
                              const primaryType = row.relationTypes[0];
                              return (
                                <ButtonBase
                                  key={`overlay-relation-${row.edge.id}`}
                                  onClick={() => {
                                    handleLinkClick(row.edge);
                                    setSelectionPanelExpanded(false);
                                    setTabValue(2);
                                  }}
                                  sx={{
                                    width: '100%',
                                    minHeight: 38,
                                    justifyContent: 'flex-start',
                                    alignItems: 'center',
                                    gap: 0.6,
                                    px: 0.65,
                                    py: 0.45,
                                    borderRadius: '8px',
                                    border: '1px solid',
                                    borderColor: selectedEdgeId === row.edge.id ? alpha(row.dominantColor, 0.44) : alpha('#fff', 0.07),
                                    bgcolor: selectedEdgeId === row.edge.id ? alpha(row.dominantColor, 0.12) : alpha('#020617', 0.24),
                                    color: 'text.primary',
                                    textAlign: 'left',
                                    transition: 'transform 140ms ease, border-color 140ms ease, background-color 140ms ease',
                                    '&:hover': {
                                      transform: 'translateX(1px)',
                                      borderColor: alpha(row.dominantColor, 0.36),
                                      bgcolor: alpha(row.dominantColor, 0.09),
                                    },
                                  }}
                                >
                                  {renderRelationSymbolMark(primaryType?.type || row.edge.type || row.edge.label, row.dominantColor, {
                                    selected: selectedEdgeId === row.edge.id,
                                    size: 24,
                                  })}
                                  <Box sx={{ minWidth: 0, flex: 1 }}>
                                    <Typography variant="caption" noWrap sx={{ display: 'block', fontWeight: 900, lineHeight: 1.12 }}>
                                      {row.sourceLabel} {'->'} {row.targetLabel}
                                    </Typography>
                                    <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', lineHeight: 1.12 }}>
                                      {primaryType ? getTypeLabel(primaryType.type) : 'Related'} / {row.relationCount} relations
                                    </Typography>
                                  </Box>
                                  <Chip
                                    size="small"
                                    label={row.relationCount}
                                    sx={{
                                      height: 18,
                                      minWidth: 25,
                                      color: row.dominantColor,
                                      bgcolor: alpha(row.dominantColor, 0.14),
                                      '& .MuiChip-label': { px: 0.48, fontSize: '0.56rem', fontWeight: 900 },
                                    }}
                                  />
                                </ButtonBase>
                              );
                            })}
                            {relationBrowserRows.length > 4 && (
                              <Typography variant="caption" color="text.disabled" sx={{ px: 0.5, pt: 0.25 }}>
                                +{formatNumber(relationBrowserRows.length - 4)} more relations
                              </Typography>
                            )}
                          </Box>
                        )}

                        {tabValue === 2 && selectedEntity && (
                          <Box sx={{ display: 'grid', gap: 0.7 }}>
                            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 0.75, minWidth: 0 }}>
                              {renderEntitySymbolMark(selectedEntity.entity_type, selectedEntity.name, {
                                selected: true,
                                size: 30,
                              })}
                              <Box sx={{ minWidth: 0, flex: 1 }}>
                                <Typography variant="body2" noWrap sx={{ fontWeight: 900, lineHeight: 1.12 }}>
                                  {selectedEntity.name}
                                </Typography>
                                <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', lineHeight: 1.12 }}>
                                  {getEntityDisplayTypeLabel(selectedEntity.entity_type, selectedEntity.name)}
                                </Typography>
                              </Box>
                            </Box>
                            <Box sx={{ display: 'flex', gap: 0.4, flexWrap: 'wrap' }}>
                              {selectedDegree && (
                                <Chip size="small" label={`${selectedDegree.total} relations`} sx={{ height: 20, '& .MuiChip-label': { px: 0.58, fontSize: '0.58rem', fontWeight: 850 } }} />
                              )}
                              {selectedEntityGraphRole?.rank && (
                                <Chip size="small" label={`rank #${selectedEntityGraphRole.rank}`} sx={{ height: 20, color: selectedEntityColors.light, bgcolor: alpha(selectedEntityColors.main, 0.14), '& .MuiChip-label': { px: 0.58, fontSize: '0.58rem', fontWeight: 850 } }} />
                              )}
                              {selectedEntityGraphRole?.component && (
                                <Chip size="small" label={`${selectedEntityGraphRole.componentLabel} / ${formatNumber(selectedEntityGraphRole.component.nodeCount)}`} sx={{ height: 20, '& .MuiChip-label': { px: 0.58, fontSize: '0.58rem', fontWeight: 850 } }} />
                              )}
                            </Box>
                            {selectedEntity.description && (
                              <Typography variant="caption" color="text.secondary" sx={{ display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden', lineHeight: 1.35 }}>
                                {selectedEntity.description}
                              </Typography>
                            )}
                            {renderSelectionMetadataBlock(selectedMetadataEntries, selectedEntityColors.main, 'entity')}
                          </Box>
                        )}

                        {tabValue === 2 && selectedEdgeDetails && !selectedEntity && (
                          <Box sx={{ display: 'grid', gap: 0.7 }}>
                            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 0.75, minWidth: 0 }}>
                              {renderRelationSymbolMark(selectedEdgeDetails.edge.type || selectedEdgeDetails.edge.label, selectedEdgeDetails.dominantColor, {
                                selected: true,
                                size: 28,
                              })}
                              <Box sx={{ minWidth: 0, flex: 1 }}>
                                <Typography variant="body2" noWrap sx={{ fontWeight: 900, lineHeight: 1.12 }}>
                                  {selectedEdgeDetails.source?.name || selectedEdgeDetails.sourceId}
                                </Typography>
                                <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', lineHeight: 1.12 }}>
                                  {'->'} {selectedEdgeDetails.target?.name || selectedEdgeDetails.targetId}
                                </Typography>
                              </Box>
                            </Box>
                            <Box sx={{ display: 'flex', gap: 0.4, flexWrap: 'wrap' }}>
                              <Chip size="small" label={`${selectedEdgeDetails.relationCount} relations`} sx={{ height: 20, color: selectedEdgeDetails.dominantColor, bgcolor: alpha(selectedEdgeDetails.dominantColor, 0.14), '& .MuiChip-label': { px: 0.58, fontSize: '0.58rem', fontWeight: 850 } }} />
                              {selectedEdgeDetails.weight > 0 && (
                                <Chip size="small" label={`w ${formatNumber(Math.round(selectedEdgeDetails.weight))}`} sx={{ height: 20, '& .MuiChip-label': { px: 0.58, fontSize: '0.58rem', fontWeight: 850 } }} />
                              )}
                            </Box>
                            <Box sx={{ display: 'flex', gap: 0.4, flexWrap: 'wrap' }}>
                              {selectedEdgeDetails.relationTypes.slice(0, 5).map(({ type, count, color }) => (
                                <Chip
                                  key={`overlay-edge-type-${selectedEdgeDetails.edge.id}-${type}`}
                                  size="small"
                                  icon={renderRelationSymbolMark(type, color, { selected: true, size: 17 })}
                                  label={`${getTypeLabel(type)} ${count}`}
                                  onClick={() => focusRelationshipType(type)}
                                  variant="outlined"
                                  sx={{ height: 20, color, bgcolor: alpha(color, 0.1), borderColor: alpha(color, 0.36), '& .MuiChip-label': { px: 0.58, fontSize: '0.58rem', fontWeight: 850 }, '& .MuiChip-icon': { ml: 0.5, mr: -0.25, color: 'inherit' } }}
                                />
                              ))}
                            </Box>
                            {selectedEvidence && (
                              <Typography variant="caption" color="text.secondary" sx={{ display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden', lineHeight: 1.35 }}>
                                {formatEvidenceSnippetDisplayText(selectedEvidence.context)}
                              </Typography>
                            )}
                            {renderSelectionMetadataBlock(selectedEdgeMetadataEntries, selectedEdgeDetails.dominantColor, 'relation')}
                          </Box>
                        )}

                        {tabValue === 2 && !selectedEntity && !selectedEdgeDetails && (
                          <Box sx={{ py: 3, textAlign: 'center' }}>
                            <InfoIcon sx={{ fontSize: 28, color: 'text.disabled', mb: 0.75 }} />
                            <Typography variant="caption" color="text.secondary">
                              Select a node or relation for details.
                            </Typography>
                          </Box>
                        )}
                      </Box>
                    </Box>
                  )}

                    {selectionOverlayOpen && (
                    <>
                    {selectionCardAnchored && selectionAnchor && selectionCardPlacement.tether && (
                      <>
                        <Box
                          data-kg-selection-tether="true"
                          sx={{
                            position: 'absolute',
                            display: { xs: 'none', sm: 'block' },
                            left: selectionCardPlacement.tether.left,
                            top: selectionCardPlacement.tether.top,
                            width: Math.max(0, selectionCardPlacement.tether.length),
                            height: 2,
                            zIndex: 7,
                            pointerEvents: 'none',
                            transformOrigin: 'left center',
                            transform: `rotate(${selectionCardPlacement.tether.angle}deg)`,
                            background: `linear-gradient(90deg, ${alpha(selectionAccentColor, 0.42)}, ${alpha(selectionAccentColor, 0.04)})`,
                            filter: 'none',
                            opacity: 0.42,
                            transition: 'left 620ms cubic-bezier(0.2, 0.8, 0.2, 1), top 620ms cubic-bezier(0.2, 0.8, 0.2, 1), width 620ms cubic-bezier(0.2, 0.8, 0.2, 1), transform 620ms cubic-bezier(0.2, 0.8, 0.2, 1), opacity 180ms ease',
                          }}
                        />
                        <Box
                          data-kg-selection-anchor-dot="true"
                          sx={{
                            position: 'absolute',
                            display: { xs: 'none', sm: 'block' },
                            left: selectionAnchor.x - 5,
                            top: selectionAnchor.y - 5,
                            width: 10,
                            height: 10,
                            borderRadius: '999px',
                            zIndex: 7,
                            pointerEvents: 'none',
                            border: `1px solid ${alpha(selectionAccentColor, 0.86)}`,
                            bgcolor: alpha(selectionAccentColor, 0.28),
                            boxShadow: `0 0 0 4px ${alpha(selectionAccentColor, 0.06)}`,
                            animation: 'none',
                            transition: 'left 620ms cubic-bezier(0.2, 0.8, 0.2, 1), top 620ms cubic-bezier(0.2, 0.8, 0.2, 1), border-color 180ms ease, background-color 180ms ease',
                          }}
                          />
                        </>
                      )}
                      {selectionPanelExpanded && selectionActionsOpen && selectionQuickRailPlacement && selectionQuickActionCount > 0 && (
                        <Box
                          data-kg-selection-quick-rail="true"
                          data-kg-selection-quick-kind={selectedEntity ? 'entity' : 'relation'}
                          data-kg-selection-quick-open={selectionActionsOpen ? 'true' : 'false'}
                          onClick={() => setSelectionActionsOpen(false)}
                          sx={{
                            position: 'absolute',
                            display: { xs: 'none', sm: 'flex' },
                            alignItems: 'center',
                            gap: 0.45,
                            left: selectionQuickRailPlacement.left,
                            top: selectionQuickRailPlacement.top,
                            zIndex: 9,
                            p: 0.35,
                            borderRadius: '8px',
                            border: '1px solid',
                            borderColor: alpha(selectionAccentColor, 0.22),
                            bgcolor: alpha('#050816', 0.82),
                            boxShadow: `0 14px 42px ${alpha('#000', 0.34)}`,
                            backdropFilter: 'blur(16px)',
                            pointerEvents: 'auto',
                            transformOrigin: selectionQuickRailPlacement.transformOrigin,
                            animation: 'kgSelectionIn 190ms cubic-bezier(0.2, 0.8, 0.2, 1)',
                            transition: 'left 620ms cubic-bezier(0.2, 0.8, 0.2, 1), top 620ms cubic-bezier(0.2, 0.8, 0.2, 1), transform 220ms cubic-bezier(0.2, 0.8, 0.2, 1), opacity 180ms ease',
                          }}
                        >
                          {selectedEntity && selectedEntityHasRelationships && (
                            <Tooltip title={selectedEntityNeighborhoodFocused ? 'Show wider graph' : 'Focus neighborhood'} disableHoverListener disableFocusListener disableTouchListener>
                              <IconButton
                                size="small"
                                data-kg-quick-neighborhood="true"
                                aria-label={selectedEntityNeighborhoodFocused ? 'Show wider graph context' : 'Focus selected entity neighborhood'}
                                onClick={toggleSelectedEntityNeighborhood}
                                sx={{
                                  ...graphOverlayButtonSx,
                                  width: 28,
                                  height: 28,
                                  color: selectedEntityNeighborhoodFocused ? selectedEntityColors.main : 'text.secondary',
                                  bgcolor: selectedEntityNeighborhoodFocused ? alpha(selectedEntityColors.main, 0.14) : alpha('#070a12', 0.58),
                                }}
                              >
                                <HubIcon sx={{ fontSize: 16 }} />
                              </IconButton>
                            </Tooltip>
                          )}
                          {selectedEntity && activeEntityPathPreview && (
                            <Tooltip
                              title={selectedHoverPathPreview
                                ? `Reveal ${activeEntityPathPreview.segments.length} hop hover path`
                                : `Reveal ${activeEntityPathPreview.segments.length} hop path`}
                              disableHoverListener
                              disableFocusListener
                              disableTouchListener
                            >
                              <IconButton
                                size="small"
                                data-kg-quick-path="true"
                                aria-label="Reveal selected entity path"
                                onClick={revealSelectedEntityPath}
                                sx={{
                                  ...graphOverlayButtonSx,
                                  width: 28,
                                  height: 28,
                                  color: selectionAccentColor,
                                  bgcolor: alpha(selectionAccentColor, 0.13),
                                }}
                              >
                                <RelationIcon sx={{ fontSize: 16 }} />
                              </IconButton>
                            </Tooltip>
                          )}
                          {selectedEntity && selectedEntityGraphRole?.component && selectedComponentId !== selectedEntityGraphRole.component.id && (
                            <Tooltip title={`Focus ${selectedEntityGraphRole.componentLabel || 'component'}`} disableHoverListener disableFocusListener disableTouchListener>
                              <IconButton
                                size="small"
                                data-kg-quick-component="true"
                                aria-label="Focus selected entity component"
                                onClick={() => focusComponent(selectedEntityGraphRole.component!)}
                                sx={{ ...graphOverlayButtonSx, width: 28, height: 28, color: 'text.secondary' }}
                              >
                                <ExploreIcon sx={{ fontSize: 16 }} />
                              </IconButton>
                            </Tooltip>
                          )}
                          {selectedEdgeDetails?.source && (
                            <Tooltip title="Open source entity" disableHoverListener disableFocusListener disableTouchListener>
                              <IconButton
                                size="small"
                                data-kg-quick-source="true"
                                aria-label={`Open source entity ${selectedEdgeDetails.source.name}`}
                                onClick={() => {
                                  openGraphEntitySelection(selectedEdgeDetails.source!, {
                                    clearSearch: true,
                                    closeInspector: true,
                                  });
                                }}
                                sx={{ ...graphOverlayButtonSx, width: 28, height: 28, color: getTypeColors(selectedEdgeDetails.source.entity_type, selectedEdgeDetails.source.name).main }}
                              >
                                <BrowseBackIcon sx={{ fontSize: 16 }} />
                              </IconButton>
                            </Tooltip>
                          )}
                          {selectedEdgeDetails?.target && (
                            <Tooltip title="Open target entity" disableHoverListener disableFocusListener disableTouchListener>
                              <IconButton
                                size="small"
                                data-kg-quick-target="true"
                                aria-label={`Open target entity ${selectedEdgeDetails.target.name}`}
                                onClick={() => {
                                  openGraphEntitySelection(selectedEdgeDetails.target!, {
                                    clearSearch: true,
                                    closeInspector: true,
                                  });
                                }}
                                sx={{ ...graphOverlayButtonSx, width: 28, height: 28, color: getTypeColors(selectedEdgeDetails.target.entity_type, selectedEdgeDetails.target.name).main }}
                              >
                                <CollapseBundleIcon sx={{ fontSize: 16 }} />
                              </IconButton>
                            </Tooltip>
                          )}
                          {selectedEdgeDetails && selectedEdgePrimaryRelationType && (
                            <Tooltip title={`Focus ${getTypeLabel(selectedEdgePrimaryRelationType)} relations`} disableHoverListener disableFocusListener disableTouchListener>
                              <IconButton
                                size="small"
                                data-kg-quick-relation-type="true"
                                aria-label="Focus selected relationship type"
                                onClick={() => focusRelationshipType(selectedEdgePrimaryRelationType)}
                                sx={{ ...graphOverlayButtonSx, width: 28, height: 28, color: selectedEdgeDetails.dominantColor }}
                              >
                                <RelationIcon sx={{ fontSize: 16 }} />
                              </IconButton>
                            </Tooltip>
                          )}
                          <Tooltip title="Open inspector" disableHoverListener disableFocusListener disableTouchListener>
                            <IconButton
                              size="small"
                              data-kg-quick-details="true"
                              aria-label="Open selected graph details"
                              onClick={openGraphDetails}
                              sx={{ ...graphOverlayButtonSx, width: 28, height: 28, color: 'text.secondary' }}
                            >
                              <InfoIcon sx={{ fontSize: 16 }} />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Clear selection" disableHoverListener disableFocusListener disableTouchListener>
                            <IconButton
                              size="small"
                              data-kg-quick-clear="true"
                              aria-label="Clear graph selection"
                              onClick={clearGraphSelection}
                              sx={{ ...graphOverlayButtonSx, width: 28, height: 28, color: 'text.secondary' }}
                            >
                              <CloseIcon sx={{ fontSize: 15 }} />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      )}
                      <Box
                        data-kg-selection-card="true"
                        data-kg-selection-anchor-mode={selectionCardPlacement.mode}
                      data-kg-selection-anchor-kind={selectionAnchor?.kind || 'none'}
                      data-kg-selection-expanded={selectionPanelExpanded ? 'true' : 'false'}
                      data-kg-selection-width={selectionCardWidth}
                      sx={{
                      position: 'absolute',
                      zIndex: 8,
                        ...(selectionCardAnchored
                          ? {
                            left: { xs: 'auto', sm: selectionCardPlacement.left },
                            top: { xs: 'auto', sm: selectionCardPlacement.top },
                            right: { xs: 12, sm: 'auto' },
                            bottom: { xs: selectionDockBottom, sm: 'auto' },
                            transformOrigin: { xs: 'right bottom', sm: selectionCardPlacement.transformOrigin },
                          }
                          : {
                            right: 12,
                            bottom: selectionDockBottom,
                            transformOrigin: 'right bottom',
                          }),
                        width: {
                          xs: selectionPanelExpanded
                            ? 'calc(100% - 24px)'
                            : selectedEdgeDetails
                              ? 'min(calc(100% - 24px), 372px)'
                              : 'min(calc(100% - 24px), 332px)',
                          sm: selectionCardWidth,
                        },
                          maxWidth: 'calc(100% - 24px)',
                          maxHeight: selectionCardMaxHeight,
                        overflowX: 'hidden',
                        overflowY: selectionPanelExpanded || neighborOrbitExpanded ? 'auto' : 'hidden',
                        display: { xs: inspectorOverlayOpen ? 'none' : 'block', md: 'block' },
                        p: selectionPanelExpanded ? 1 : 0.82,
                        ...graphOverlayPanelSx,
                        bgcolor: alpha('#050816', selectionPanelExpanded ? 0.9 : 0.84),
                        animation: 'kgSelectionIn 220ms cubic-bezier(0.2, 0.8, 0.2, 1)',
                        transition: 'left 620ms cubic-bezier(0.2, 0.8, 0.2, 1), top 620ms cubic-bezier(0.2, 0.8, 0.2, 1), width 220ms ease, max-height 220ms ease, transform 260ms cubic-bezier(0.2, 0.8, 0.2, 1)',
                      }}
                    >
                    {selectedEntity ? (
                      <>
                        <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 1, mb: 0.8 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, minWidth: 0 }}>
                            {renderEntitySymbolMark(selectedEntity.entity_type, selectedEntity.name, {
                              selected: true,
                              size: 32,
                            })}
                            <Box sx={{ minWidth: 0 }}>
                              <Typography variant="body2" noWrap sx={{ fontWeight: 900, lineHeight: 1.15 }}>
                                {selectedEntity.name}
                              </Typography>
                              <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block', lineHeight: 1.15 }}>
                                {[
                                  getEntityDisplayTypeLabel(selectedEntity.entity_type, selectedEntity.name),
                                  selectedDegree
                                    ? selectedEntityHasRelationships
                                      ? `${selectedDegree.total} relation${selectedDegree.total === 1 ? '' : 's'}`
                                      : 'Isolated'
                                    : null,
                                ].filter(Boolean).join(' • ')}
                              </Typography>
                            </Box>
                          </Box>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.4, flexShrink: 0 }}>
                              {selectionPanelExpanded && renderBrowseHistoryControls(selectedEntityColors.main)}
                              {selectedEntityHasRelationships && (
                                <Tooltip title={selectedEntityNeighborhoodFocused ? 'Show wider graph' : 'Focus neighborhood'} disableHoverListener disableFocusListener disableTouchListener>
                                  <IconButton
                                    size="small"
                                    data-kg-focus-neighborhood="true"
                                    aria-label={selectedEntityNeighborhoodFocused ? 'Show wider graph context' : 'Focus selected entity neighborhood'}
                                    onClick={toggleSelectedEntityNeighborhood}
                                    sx={{
                                      width: 26,
                                      height: 26,
                                      borderRadius: '8px',
                                      color: selectedEntityNeighborhoodFocused ? selectedEntityColors.main : 'text.secondary',
                                      bgcolor: selectedEntityNeighborhoodFocused ? alpha(selectedEntityColors.main, 0.13) : 'transparent',
                                      border: `1px solid ${selectedEntityNeighborhoodFocused ? alpha(selectedEntityColors.main, 0.28) : 'transparent'}`,
                                    }}
                                  >
                                    <HubIcon sx={{ fontSize: 16 }} />
                                  </IconButton>
                                </Tooltip>
                              )}
                              {selectionPanelExpanded && selectionQuickActionCount > 0 && (
                                <Tooltip title={selectionActionsOpen ? 'Hide selection actions' : 'Selection actions'} disableHoverListener disableFocusListener disableTouchListener>
                                  <IconButton
                                    size="small"
                                    data-kg-selection-actions-toggle="true"
                                    aria-label={selectionActionsOpen ? 'Hide selected entity actions' : 'Show selected entity actions'}
                                    onClick={() => setSelectionActionsOpen((open) => !open)}
                                    sx={{
                                      width: 26,
                                      height: 26,
                                      borderRadius: '8px',
                                      color: selectionActionsOpen ? selectedEntityColors.main : 'text.secondary',
                                      bgcolor: selectionActionsOpen ? alpha(selectedEntityColors.main, 0.13) : 'transparent',
                                    }}
                                  >
                                    <MoreIcon sx={{ fontSize: 17 }} />
                                  </IconButton>
                                </Tooltip>
                              )}
                              <Tooltip title={selectionPanelExpanded ? 'Collapse details' : 'Show details'} disableHoverListener disableFocusListener disableTouchListener>
                                <IconButton
                                  size="small"
                                  data-kg-selection-expand="true"
                                  aria-label={selectionPanelExpanded ? 'Collapse selected entity details' : 'Show selected entity details'}
                                  onClick={() => {
                                    setSelectionActionsOpen(false);
                                    setSelectionPanelExpanded((open) => !open);
                                  }}
                                  sx={{
                                    width: 26,
                                    height: 26,
                                    borderRadius: '8px',
                                    color: selectionPanelExpanded ? selectedEntityColors.main : 'text.secondary',
                                    bgcolor: selectionPanelExpanded ? alpha(selectedEntityColors.main, 0.13) : 'transparent',
                                  }}
                                >
                                  {selectionPanelExpanded ? <PreviousResultIcon sx={{ fontSize: 17 }} /> : <ExpandBundleIcon sx={{ fontSize: 17 }} />}
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="Clear selection" disableHoverListener disableFocusListener disableTouchListener>
                                <IconButton
                                size="small"
                                aria-label="Clear graph selection"
                                onClick={clearGraphSelection}
                                sx={{ width: 26, height: 26, borderRadius: '8px', color: 'text.secondary', flexShrink: 0 }}
                              >
                                <CloseIcon sx={{ fontSize: 16 }} />
                              </IconButton>
                            </Tooltip>
                          </Box>
                          </Box>

                          {!selectionPanelExpanded && renderCompactMetadataFactStrip(selectedEntityMetadataFacts, 'entity', ['entity-relations'])}
                          {renderCollapsedEntityRelationMix()}
                          {!selectionPanelExpanded && renderIsolatedEntityNotice(selectedEntityColors.main)}
                          {!selectionPanelExpanded && renderEntityNeighborOrbit(selectedRelationshipBundles, selectedEntityColors.main)}

                          <Collapse in={selectionPanelExpanded} timeout={180} unmountOnExit>
                            <Box>

                          {selectedEntity.description && (
                            <Typography
                            variant="caption"
                            color="text.secondary"
                            sx={{
                              display: '-webkit-box',
                              WebkitLineClamp: 2,
                              WebkitBoxOrient: 'vertical',
                              overflow: 'hidden',
                              lineHeight: 1.35,
                              mb: 0.75,
                            }}
                          >
                              {selectedEntity.description}
                            </Typography>
                          )}

                            {renderMetadataFactStrip(selectedEntityMetadataFacts, 'entity')}
                            {renderEntityRoleStrip(selectedEntityColors.main)}
                            {renderIsolatedEntityNotice(selectedEntityColors.main)}

                            {activeEntityPathPreview && renderEntityPathPreview(activeEntityPathPreview, selectedEntityColors.main)}

                          {renderSelectionMetadataBlock(selectedMetadataEntries, selectedEntityColors.main, 'entity')}

                            {renderEntityNeighborOrbit(selectedRelationshipBundles, selectedEntityColors.main)}

                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 0.45, mt: 0.1 }}>
                            <Tooltip title="Open inspector">
                              <IconButton
                                size="small"
                                data-kg-selection-footer-details="true"
                                aria-label="Open selected entity inspector"
                                onClick={openGraphDetails}
                                sx={selectionFooterActionSx(selectedEntityColors.main)}
                              >
                                <InfoIcon sx={{ fontSize: 16 }} />
                              </IconButton>
                            </Tooltip>
                            {selectedEntityGraphRole?.component && selectedComponentId !== selectedEntityGraphRole.component.id && (
                              <Tooltip title={`Focus ${selectedEntityGraphRole.componentLabel || 'component'}`}>
                                <IconButton
                                  size="small"
                                  data-kg-selection-footer-component="true"
                                  aria-label="Focus selected entity component"
                                  onClick={() => focusComponent(selectedEntityGraphRole.component!)}
                                  sx={selectionFooterActionSx(selectedEntityColors.main)}
                                >
                                  <ExploreIcon sx={{ fontSize: 16 }} />
                                </IconButton>
                              </Tooltip>
                            )}
                          </Box>

                            </Box>
                          </Collapse>
                        </>
                    ) : selectedEdgeDetails ? (
                      <>
                        <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 1, mb: 0.65 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, minWidth: 0 }}>
                              {renderRelationSymbolMark(
                                selectedEdgeDetails.relationTypes[0]?.type || selectedEdgeDetails.edge.type || selectedEdgeDetails.edge.label,
                                selectedEdgeDetails.dominantColor,
                                { selected: true, size: 30 }
                              )}
                            <Box sx={{ minWidth: 0 }}>
                              <Typography variant="body2" noWrap sx={{ fontWeight: 900, lineHeight: 1.15 }}>
                                {getTypeLabel(selectedEdgeDetails.relationTypes[0]?.type || selectedEdgeDetails.edge.type || selectedEdgeDetails.edge.label)}
                              </Typography>
                              <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block', lineHeight: 1.15 }}>
                                {`${selectedEdgeDetails.relationCount} ${selectedEdgeDetails.relationCount === 1 ? 'relation' : 'relations'}${selectedEdgeDetails.weight > 0 ? ` · ${formatNumber(Math.round(selectedEdgeDetails.weight))} weight` : ''}`}
                              </Typography>
                            </Box>
                          </Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.4, flexShrink: 0 }}>
                              {selectionPanelExpanded && renderBrowseHistoryControls(selectedEdgeDetails.dominantColor)}
                              {selectionPanelExpanded && selectionQuickActionCount > 0 && (
                                <Tooltip title={selectionActionsOpen ? 'Hide selection actions' : 'Selection actions'} disableHoverListener disableFocusListener disableTouchListener>
                                  <IconButton
                                    size="small"
                                    data-kg-selection-actions-toggle="true"
                                    aria-label={selectionActionsOpen ? 'Hide selected relationship actions' : 'Show selected relationship actions'}
                                    onClick={() => setSelectionActionsOpen((open) => !open)}
                                    sx={{
                                      width: 26,
                                      height: 26,
                                      borderRadius: '8px',
                                      color: selectionActionsOpen ? selectedEdgeDetails.dominantColor : 'text.secondary',
                                      bgcolor: selectionActionsOpen ? alpha(selectedEdgeDetails.dominantColor, 0.13) : 'transparent',
                                    }}
                                  >
                                    <MoreIcon sx={{ fontSize: 17 }} />
                                  </IconButton>
                                </Tooltip>
                              )}
                              <Tooltip title={selectionPanelExpanded ? 'Collapse details' : 'Show details'} disableHoverListener disableFocusListener disableTouchListener>
                                <IconButton
                                  size="small"
                                  data-kg-selection-expand="true"
                                  aria-label={selectionPanelExpanded ? 'Collapse selected relationship details' : 'Show selected relationship details'}
                                  onClick={() => {
                                    setSelectionActionsOpen(false);
                                    setSelectionPanelExpanded((open) => !open);
                                  }}
                                  sx={{
                                    width: 26,
                                    height: 26,
                                    borderRadius: '8px',
                                    color: selectionPanelExpanded ? selectedEdgeDetails.dominantColor : 'text.secondary',
                                    bgcolor: selectionPanelExpanded ? alpha(selectedEdgeDetails.dominantColor, 0.13) : 'transparent',
                                  }}
                                >
                                  {selectionPanelExpanded ? <PreviousResultIcon sx={{ fontSize: 17 }} /> : <ExpandBundleIcon sx={{ fontSize: 17 }} />}
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="Clear selection" disableHoverListener disableFocusListener disableTouchListener>
                                <IconButton
                                size="small"
                                aria-label="Clear graph selection"
                                onClick={clearGraphSelection}
                                sx={{ width: 26, height: 26, borderRadius: '8px', color: 'text.secondary', flexShrink: 0 }}
                              >
                                <CloseIcon sx={{ fontSize: 16 }} />
                              </IconButton>
                            </Tooltip>
                          </Box>
                        </Box>

                        {!selectionPanelExpanded && renderCompactMetadataFactStrip(selectedRelationMetadataFacts, 'relation', ['relation-count'])}

                        <Box
                          data-kg-edge-route="true"
                          data-kg-edge-route-compact={selectionPanelExpanded ? 'false' : 'true'}
                          sx={{
                              position: 'relative',
                              display: 'grid',
                              gridTemplateColumns: selectionPanelExpanded
                                ? { xs: 'minmax(0, 1fr)', sm: 'minmax(118px, 1fr) auto minmax(118px, 1fr)' }
                                : 'minmax(0, 1fr) auto minmax(0, 1fr)',
                              alignItems: 'stretch',
                              gap: selectionPanelExpanded ? 0.62 : 0.42,
                            mb: selectionPanelExpanded ? 0.65 : 0.45,
                            p: selectionPanelExpanded ? 0.55 : 0.42,
                            borderRadius: '8px',
                            border: '1px solid',
                            borderColor: alpha(selectedEdgeDetails.dominantColor, 0.22),
                            bgcolor: alpha(selectedEdgeDetails.dominantColor, 0.055),
                            overflow: 'hidden',
                            '&::before': {
                              content: '""',
                              position: 'absolute',
                              left: 44,
                              right: 44,
                              top: '50%',
                                height: 1,
                                transform: 'translateY(-50%)',
                                background: `linear-gradient(90deg, ${alpha(selectedEdgeDetails.dominantColor, 0.08)}, ${alpha(selectedEdgeDetails.dominantColor, 0.5)}, ${alpha(selectedEdgeDetails.dominantColor, 0.08)})`,
                                opacity: 0.78,
                                display: selectionPanelExpanded ? { xs: 'none', sm: 'block' } : 'block',
                              },
                              '&::after': {
                                content: '""',
                                position: 'absolute',
                              left: 44,
                              right: 44,
                              top: '50%',
                              height: 2,
                              transform: 'translateY(-50%)',
                                background: `repeating-linear-gradient(90deg, transparent 0 12px, ${alpha(selectedEdgeDetails.dominantColor, 0.76)} 12px 20px)`,
                                backgroundSize: '64px 2px',
                                opacity: 0.42,
                                animation: 'kgRouteFlow 2.4s linear infinite',
                                display: selectionPanelExpanded ? { xs: 'none', sm: 'block' } : 'block',
                              },
                            '@keyframes kgRouteFlow': {
                              from: { backgroundPosition: '0 0' },
                              to: { backgroundPosition: '64px 0' },
                            },
                          }}
                        >
                          <ButtonBase
                            data-kg-edge-route-source="true"
                            aria-label={`Open source entity ${selectedEdgeDetails.source?.name || selectedEdgeDetails.sourceId}`}
                            disabled={!selectedEdgeDetails.source}
                            onClick={() => {
                              if (!selectedEdgeDetails.source) return;
                              openGraphEntitySelection(selectedEdgeDetails.source, {
                                clearSearch: true,
                                closeInspector: true,
                              });
                            }}
                            sx={{
                              position: 'relative',
                                zIndex: 1,
                                minWidth: 0,
                                width: '100%',
                                minHeight: selectionPanelExpanded ? 46 : 34,
                              justifyContent: 'flex-start',
                              alignItems: 'center',
                              gap: selectionPanelExpanded ? 0.5 : 0.38,
                              px: selectionPanelExpanded ? 0.55 : 0.42,
                              py: selectionPanelExpanded ? 0.35 : 0.26,
                              borderRadius: '7px',
                              color: 'text.primary',
                              textAlign: 'left',
                              bgcolor: alpha('#050816', 0.5),
                              transition: 'background-color 140ms ease, transform 140ms ease',
                              '&:hover': { transform: 'translateY(-1px)', bgcolor: alpha(selectedEdgeDetails.dominantColor, 0.09) },
                              '&.Mui-disabled': { color: 'text.primary', opacity: 0.72 },
                            }}
                          >
                            {renderEntitySymbolMark(
                              selectedEdgeDetails.source?.entity_type,
                              selectedEdgeDetails.source?.name || selectedEdgeDetails.sourceId,
                              {
                                muted: !selectedEdgeDetails.source,
                                selected: Boolean(selectedEdgeDetails.source),
                                size: selectionPanelExpanded ? 28 : 22,
                              }
                            )}
                            <Box sx={{ minWidth: 0 }}>
                              <Typography
                                variant="caption"
                                sx={{
                                  display: '-webkit-box',
                                  WebkitLineClamp: selectionPanelExpanded ? 2 : 1,
                                  WebkitBoxOrient: 'vertical',
                                  overflow: 'hidden',
                                  fontWeight: 900,
                                  lineHeight: 1.08,
                                  fontSize: selectionPanelExpanded ? undefined : '0.62rem',
                                }}
                              >
                                {selectedEdgeDetails.source?.name || selectedEdgeDetails.sourceId}
                              </Typography>
                              {selectionPanelExpanded && (
                              <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', fontSize: '0.58rem', fontWeight: 800, lineHeight: 1.1 }}>
                                {selectedEdgeDetails.source ? getEntityDisplayTypeLabel(selectedEdgeDetails.source.entity_type, selectedEdgeDetails.source.name) : 'Source'}
                              </Typography>
                              )}
                            </Box>
                          </ButtonBase>

                          <Box
                            data-kg-edge-route-relation="true"
                            sx={{
                              position: 'relative',
                                zIndex: 1,
                                minWidth: 0,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                py: selectionPanelExpanded ? { xs: 0.1, sm: 0 } : 0,
                              }}
                          >
                            <Chip
                              size="small"
                              icon={renderRelationSymbolMark(
                                selectedEdgeDetails.relationTypes[0]?.type || selectedEdgeDetails.edge.type || selectedEdgeDetails.edge.label,
                                selectedEdgeDetails.relationTypes[0]?.color || selectedEdgeDetails.dominantColor,
                                { selected: true, size: 18 }
                              )}
                              label={getTypeLabel(selectedEdgeDetails.relationTypes[0]?.type || selectedEdgeDetails.edge.type || selectedEdgeDetails.edge.label)}
                              sx={{
                                  height: selectionPanelExpanded ? 22 : 20,
                                  maxWidth: selectionPanelExpanded ? { xs: '100%', sm: 132 } : 96,
                                color: selectedEdgeDetails.relationTypes[0]?.color || selectedEdgeDetails.dominantColor,
                                bgcolor: alpha(selectedEdgeDetails.relationTypes[0]?.color || selectedEdgeDetails.dominantColor, 0.14),
                                border: `1px solid ${alpha(selectedEdgeDetails.relationTypes[0]?.color || selectedEdgeDetails.dominantColor, 0.34)}`,
                                '& .MuiChip-label': {
                                  px: 0.6,
                                  display: 'block',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  fontSize: selectionPanelExpanded ? '0.62rem' : '0.56rem',
                                  fontWeight: 900,
                                },
                              }}
                            />
                          </Box>

                          <ButtonBase
                            data-kg-edge-route-target="true"
                            aria-label={`Open target entity ${selectedEdgeDetails.target?.name || selectedEdgeDetails.targetId}`}
                            disabled={!selectedEdgeDetails.target}
                            onClick={() => {
                              if (!selectedEdgeDetails.target) return;
                              openGraphEntitySelection(selectedEdgeDetails.target, {
                                clearSearch: true,
                                closeInspector: true,
                              });
                            }}
                            sx={{
                                position: 'relative',
                                zIndex: 1,
                                minWidth: 0,
                                width: '100%',
                                minHeight: selectionPanelExpanded ? 46 : 34,
                              justifyContent: 'flex-end',
                              alignItems: 'center',
                              gap: selectionPanelExpanded ? 0.5 : 0.38,
                              px: selectionPanelExpanded ? 0.55 : 0.42,
                              py: selectionPanelExpanded ? 0.35 : 0.26,
                              borderRadius: '7px',
                              color: 'text.primary',
                              textAlign: 'right',
                              bgcolor: alpha('#050816', 0.5),
                              transition: 'background-color 140ms ease, transform 140ms ease',
                              '&:hover': { transform: 'translateY(-1px)', bgcolor: alpha(selectedEdgeDetails.dominantColor, 0.09) },
                              '&.Mui-disabled': { color: 'text.primary', opacity: 0.72 },
                            }}
                          >
                            <Box sx={{ minWidth: 0 }}>
                              <Typography
                                variant="caption"
                                sx={{
                                  display: '-webkit-box',
                                  WebkitLineClamp: selectionPanelExpanded ? 2 : 1,
                                  WebkitBoxOrient: 'vertical',
                                  overflow: 'hidden',
                                  fontWeight: 900,
                                  lineHeight: 1.08,
                                  fontSize: selectionPanelExpanded ? undefined : '0.62rem',
                                }}
                              >
                                {selectedEdgeDetails.target?.name || selectedEdgeDetails.targetId}
                              </Typography>
                              {selectionPanelExpanded && (
                              <Typography variant="caption" color="text.disabled" noWrap sx={{ display: 'block', fontSize: '0.58rem', fontWeight: 800, lineHeight: 1.1 }}>
                                {selectedEdgeDetails.target ? getEntityDisplayTypeLabel(selectedEdgeDetails.target.entity_type, selectedEdgeDetails.target.name) : 'Target'}
                              </Typography>
                              )}
                            </Box>
                            {renderEntitySymbolMark(
                              selectedEdgeDetails.target?.entity_type,
                              selectedEdgeDetails.target?.name || selectedEdgeDetails.targetId,
                              {
                                muted: !selectedEdgeDetails.target,
                                selected: Boolean(selectedEdgeDetails.target),
                                size: selectionPanelExpanded ? 28 : 22,
                              }
                            )}
                          </ButtonBase>
                          </Box>

                            {!selectionPanelExpanded && renderCollapsedEdgeEvidencePreview()}

                          <Collapse in={selectionPanelExpanded} timeout={180} unmountOnExit>
                              <Box>

                              <Box sx={{ display: 'flex', gap: 0.45, flexWrap: 'wrap', mb: selectedEdgeDetails.evidenceSnippets[0] ? 0.75 : 0.85 }}>
                              {selectedEdgeDetails.relationTypes.slice(0, 5).map(({ type, count, color }) => (
                              <Chip
                                key={`selected-edge-${selectedEdgeDetails.edge.id}-${type}`}
                                size="small"
                                icon={renderRelationSymbolMark(type, color, { selected: true, size: 18 })}
                                label={`${getTypeLabel(type)} ${count}`}
                                variant="outlined"
                              sx={{
                                height: 20,
                                color,
                                borderColor: alpha(color, 0.42),
                                bgcolor: alpha(color, 0.08),
                                '& .MuiChip-label': { px: 0.6, fontSize: '0.62rem', fontWeight: 800 },
                              }}
                            />
                            ))}
                          </Box>

                            {renderMetadataFactStrip(selectedRelationMetadataFacts, 'relation')}

                          {selectedEvidence && (
                          <Box
                            data-kg-edge-evidence="true"
                            sx={{
                              mb: 0.9,
                              p: 0.92,
                              borderRadius: '8px',
                              border: '1px solid',
                              borderColor: alpha(selectedEvidenceColor, 0.3),
                              bgcolor: alpha(selectedEvidenceColor, 0.065),
                            }}
                          >
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.45 }}>
                              <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 900, flex: 1, minWidth: 0 }} noWrap>
                                {selectedEvidenceIsExcerpt ? 'Evidence excerpt' : 'Evidence'}
                              </Typography>
                              <Chip
                                size="small"
                                label={`${selectedEvidenceIndex + 1}/${selectedEdgeDetails.evidenceSnippets.length}`}
                                sx={{
                                  height: 18,
                                  color: selectedEdgeDetails.dominantColor,
                                  bgcolor: alpha(selectedEdgeDetails.dominantColor, 0.12),
                                  '& .MuiChip-label': { px: 0.55, fontSize: '0.58rem', fontWeight: 900 },
                                }}
                              />
                              {selectedEvidenceIsExcerpt && (
                                <Chip
                                  size="small"
                                  label="excerpt"
                                  sx={{
                                    height: 18,
                                    color: selectedEvidenceColor,
                                    bgcolor: alpha(selectedEvidenceColor, 0.11),
                                    '& .MuiChip-label': { px: 0.55, fontSize: '0.56rem', fontWeight: 900 },
                                  }}
                                />
                              )}
                              {selectedEvidenceMatchesSearch && (
                                <Chip
                                  size="small"
                                  data-kg-evidence-search-match="true"
                                  label="search match"
                                  sx={{
                                    height: 18,
                                    color: accentColors.info,
                                    bgcolor: alpha(accentColors.info, 0.11),
                                    '& .MuiChip-label': { px: 0.55, fontSize: '0.56rem', fontWeight: 900 },
                                  }}
                                />
                              )}
                              {selectedEdgeDetails.evidenceSnippets.length > 1 && (
                                <Box sx={{ display: 'flex', gap: 0.25 }}>
                                  <Tooltip title="Previous evidence">
                                    <span>
                                      <IconButton
                                        size="small"
                                        data-kg-evidence-prev="true"
                                        aria-label="Previous relationship evidence"
                                        disabled={selectedEvidenceIndex === 0}
                                        onClick={() => setSelectedEvidenceIndex((current) => Math.max(0, current - 1))}
                                        sx={{ width: 22, height: 22, borderRadius: '7px', color: 'text.secondary' }}
                                      >
                                        <BrowseBackIcon sx={{ fontSize: 15 }} />
                                      </IconButton>
                                    </span>
                                  </Tooltip>
                                  <Tooltip title="Next evidence">
                                    <span>
                                      <IconButton
                                        size="small"
                                        data-kg-evidence-next="true"
                                        aria-label="Next relationship evidence"
                                        disabled={selectedEvidenceIndex >= selectedEdgeDetails.evidenceSnippets.length - 1}
                                        onClick={() => setSelectedEvidenceIndex((current) => Math.min(selectedEdgeDetails.evidenceSnippets.length - 1, current + 1))}
                                        sx={{ width: 22, height: 22, borderRadius: '7px', color: 'text.secondary' }}
                                      >
                                        <CollapseBundleIcon sx={{ fontSize: 15 }} />
                                      </IconButton>
                                    </span>
                                  </Tooltip>
                                </Box>
                              )}
                            </Box>
                            <Box
                              data-kg-edge-evidence-provenance="true"
                              sx={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 0.35,
                                flexWrap: 'wrap',
                                mb: 0.62,
                                minWidth: 0,
                              }}
                            >
                              {[
                                selectedEvidence.type
                                  ? {
                                    id: 'type',
                                    label: getTypeLabel(selectedEvidence.type),
                                    color: getRelationshipColor(selectedEvidence.type),
                                    icon: renderRelationSymbolMark(selectedEvidence.type, getRelationshipColor(selectedEvidence.type), {
                                      selected: true,
                                      size: 17,
                                    }),
                                  }
                                  : null,
                                selectedEvidence.domain
                                  ? {
                                    id: 'domain',
                                    label: truncateGraphLabel(selectedEvidence.domain, 30),
                                    color: accentColors.info,
                                    icon: <DataIcon sx={{ fontSize: 11 }} />,
                                  }
                                  : null,
                                selectedEvidence.weight > 0
                                  ? {
                                    id: 'weight',
                                    label: `${formatNumber(Math.round(selectedEvidence.weight))} weight`,
                                    color: accentColors.warning,
                                    icon: <MetricIcon sx={{ fontSize: 11 }} />,
                                  }
                                  : null,
                              ].filter((item): item is { id: string; label: string; color: string; icon: ReactElement } => Boolean(item)).map((item) => (
                                <Chip
                                  key={`selected-evidence-provenance-${item.id}`}
                                  size="small"
                                  icon={item.icon}
                                  label={item.label}
                                  variant="outlined"
                                  sx={{
                                    height: 18,
                                    maxWidth: 150,
                                    color: item.color,
                                    bgcolor: alpha(item.color, 0.09),
                                    borderColor: alpha(item.color, 0.28),
                                    '& .MuiChip-label': {
                                      px: 0.48,
                                      fontSize: '0.55rem',
                                      fontWeight: 850,
                                      overflow: 'hidden',
                                      textOverflow: 'ellipsis',
                                    },
                                    '& .MuiChip-icon': { ml: 0.45, mr: -0.24, color: 'inherit' },
                                  }}
                                />
                              ))}
                            </Box>
                            <Typography
                              variant="body2"
                              color="text.primary"
                              data-kg-edge-evidence-text="true"
                              sx={{
                                display: 'block',
                                p: 0.78,
                                borderRadius: '8px',
                                border: '1px solid',
                                borderColor: alpha(selectedEvidenceColor, selectedEvidenceMatchesSearch ? 0.36 : 0.2),
                                bgcolor: alpha('#020617', 0.34),
                                boxShadow: selectedEvidenceMatchesSearch
                                  ? `inset 3px 0 0 ${alpha(selectedEvidenceColor, 0.72)}`
                                  : `inset 3px 0 0 ${alpha(selectedEvidenceColor, 0.34)}`,
                                lineHeight: 1.52,
                                fontSize: '0.82rem',
                                wordBreak: 'break-word',
                                '& [data-kg-search-match="true"]': {
                                  color: 'inherit',
                                  bgcolor: alpha(selectedEvidenceColor, 0.24),
                                  boxShadow: `0 0 0 1px ${alpha(selectedEvidenceColor, 0.24)}`,
                                },
                              }}
                            >
                              {renderHighlightedSearchText(selectedEvidenceDisplayText, selectedEvidenceColor)}
                            </Typography>
                          </Box>
                        )}

                        {renderSelectionMetadataBlock(selectedEdgeMetadataEntries, selectedEdgeDetails.dominantColor, 'relation')}

                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 0.45, mt: 0.1 }}>
                          <Tooltip title="Open inspector">
                            <IconButton
                              size="small"
                              data-kg-selection-footer-details="true"
                              aria-label="Open selected relationship inspector"
                              onClick={openGraphDetails}
                              sx={selectionFooterActionSx(selectedEdgeDetails.dominantColor)}
                            >
                              <InfoIcon sx={{ fontSize: 16 }} />
                            </IconButton>
                          </Tooltip>
                            {selectedEdgeDetails.relationTypes[0] && (
                            <Tooltip title={`Focus ${getTypeLabel(selectedEdgeDetails.relationTypes[0].type)} relations`}>
                              <IconButton
                                size="small"
                                data-kg-selection-footer-relation-type="true"
                                aria-label="Focus selected relationship type"
                                onClick={() => focusRelationshipType(selectedEdgeDetails.relationTypes[0].type)}
                                sx={selectionFooterActionSx(selectedEdgeDetails.dominantColor)}
                              >
                                <RelationIcon sx={{ fontSize: 16 }} />
                              </IconButton>
                            </Tooltip>
                            )}
                            {selectedEdgeDetails.source && (
                            <Tooltip title={`Open ${selectedEdgeDetails.source.name}`}>
                              <IconButton
                                size="small"
                                data-kg-selection-footer-source="true"
                                aria-label={`Open source entity ${selectedEdgeDetails.source.name}`}
                                onClick={() => {
                                  openGraphEntitySelection(selectedEdgeDetails.source!, {
                                    clearSearch: true,
                                    closeInspector: true,
                                  });
                                }}
                                sx={selectionFooterActionSx(getTypeColors(selectedEdgeDetails.source.entity_type, selectedEdgeDetails.source.name).main)}
                              >
                                <BrowseBackIcon sx={{ fontSize: 16 }} />
                              </IconButton>
                            </Tooltip>
                            )}
                            {selectedEdgeDetails.target && (
                            <Tooltip title={`Open ${selectedEdgeDetails.target.name}`}>
                              <IconButton
                                size="small"
                                data-kg-selection-footer-target="true"
                                aria-label={`Open target entity ${selectedEdgeDetails.target.name}`}
                                onClick={() => {
                                  openGraphEntitySelection(selectedEdgeDetails.target!, {
                                    clearSearch: true,
                                    closeInspector: true,
                                  });
                                }}
                                sx={selectionFooterActionSx(getTypeColors(selectedEdgeDetails.target.entity_type, selectedEdgeDetails.target.name).main)}
                              >
                                <CollapseBundleIcon sx={{ fontSize: 16 }} />
                              </IconButton>
                            </Tooltip>
                              )}
                          </Box>

                            </Box>
                          </Collapse>
                        </>
                    ) : null}
                  </Box>
                  </>
                )}

                    <Box
                      sx={{
                        position: 'absolute',
                        left: 12,
                        bottom: 12,
                        zIndex: 7,
                        display: 'flex',
                        flexDirection: 'column-reverse',
                        alignItems: 'flex-start',
                        gap: 0.45,
                  maxWidth: { xs: 'calc(100% - 86px)', md: 384 },
                        pointerEvents: 'auto',
                      }}
                    >
                      <Tooltip title={`${formatNumber(visibleGraph.nodes.length)} nodes / ${formatNumber(visualGraph.edges.length)} links / ${graphScopeLabel}`}>
                        <ButtonBase
                          data-kg-stats-toggle="true"
                          aria-label={graphStatsOpen ? 'Hide graph stats' : 'Show graph stats'}
                          onClick={() => {
                            setSettingsOpen(false);
                            setSettingsSection(null);
                            setGraphScopeOpen(false);
                            setGraphOverviewOpen(false);
                            setGraphToolsOpen(false);
                            setGraphLegendOpen(false);
                            setInspectorOpen(false);
                            setSelectionActionsOpen(false);
                            if (graphStatsOpen) setGraphStatsDetailsOpen(false);
                            setGraphStatsOpen((open) => !open);
                          }}
                          sx={{
                            width: 34,
                            height: 34,
                            justifyContent: 'center',
                            alignItems: 'center',
                            p: 0,
                            ...graphOverlayPanelSx,
                            color: graphStatsOpen ? accentColors.primary : 'text.secondary',
                            textAlign: 'left',
                            transition: 'transform 160ms ease, border-color 160ms ease, background-color 160ms ease, color 160ms ease',
                            '&:hover': {
                              transform: 'translateY(-1px)',
                              borderColor: alpha(accentColors.primary, 0.34),
                              bgcolor: alpha('#0b1220', 0.84),
                              color: 'text.primary',
                            },
                          }}
                        >
                          <HubIcon sx={{ fontSize: 18 }} />
                          <Box
                            sx={{
                              position: 'absolute',
                              right: 7,
                              top: 7,
                              width: 7,
                              height: 7,
                              borderRadius: '999px',
                              bgcolor: graphStatusColor,
                              boxShadow: `0 0 8px ${alpha(graphStatusColor, 0.24)}`,
                              animation: 'none',
                            }}
                          />
                        </ButtonBase>
                      </Tooltip>

                    <Collapse in={graphStatsOpen} timeout={180} unmountOnExit>
                      <Box
                        data-kg-stats-panel="true"
                        sx={{
                          display: 'grid',
                          gap: 0.6,
                          p: 0.65,
                          width: { xs: 'min(100%, 356px)', md: 384 },
                          maxWidth: '100%',
                          ...graphOverlayPanelSx,
                        }}
                      >
                        <Box
                          data-kg-scope-quick="true"
                          sx={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            gap: 0.75,
                            minWidth: 0,
                          }}
                        >
                          <Typography variant="caption" color="text.secondary" noWrap sx={{ fontWeight: 900, minWidth: 0 }}>
                            Scope
                          </Typography>
                          <ToggleButtonGroup
                            exclusive
                            size="small"
                            value={graphScope}
                            onChange={(_, value: GraphScope | null) => {
                              if (value) setGraphScope(value);
                            }}
                            aria-label="Quick graph scope"
                            sx={{
                              height: 26,
                              flexShrink: 0,
                              '& .MuiToggleButton-root': {
                                height: 26,
                                minWidth: 0,
                                px: { xs: 0.72, sm: 0.85 },
                                borderColor: alpha('#fff', 0.1),
                                color: 'text.secondary',
                                fontSize: '0.62rem',
                                fontWeight: 900,
                                textTransform: 'none',
                                '&.Mui-selected': {
                                  color: accentColors.primary,
                                  bgcolor: alpha(accentColors.primary, 0.17),
                                  '&:hover': { bgcolor: alpha(accentColors.primary, 0.22) },
                                },
                              },
                            }}
                          >
                            <ToggleButton value="core" aria-label="Show core component">Core</ToggleButton>
                            <ToggleButton value="linked" aria-label="Show linked entities">Linked</ToggleButton>
                            <ToggleButton value="all" aria-label="Show all entities">All</ToggleButton>
                          </ToggleButtonGroup>
                        </Box>

                        <Box
                          data-kg-stats-facts="true"
                          sx={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 0.45,
                            flexWrap: 'wrap',
                            minWidth: 0,
                          }}
                        >
                          <Chip
                            size="small"
                            label={`${formatNumber(visibleGraph.nodes.length)} nodes`}
                            sx={{ height: 22, '& .MuiChip-label': { px: 0.65, fontSize: '0.65rem', fontWeight: 850 } }}
                          />
                          <Chip
                            size="small"
                            label={`${formatNumber(visualGraph.edges.length)} links`}
                            sx={{ height: 22, '& .MuiChip-label': { px: 0.65, fontSize: '0.65rem', fontWeight: 850 } }}
                          />
                          {entityTypeCount > 0 && (
                            <Chip
                              size="small"
                              label={`${entityTypeCount} types`}
                              sx={{ height: 22, '& .MuiChip-label': { px: 0.65, fontSize: '0.65rem', fontWeight: 850 } }}
                            />
                          )}
                          {graphCommunityCount > 0 && (
                            <Chip
                              size="small"
                              label={`${formatNumber(graphCommunityCount)} communities`}
                              sx={{ height: 22, '& .MuiChip-label': { px: 0.65, fontSize: '0.65rem', fontWeight: 850 } }}
                            />
                          )}
                          <Box sx={{ flex: 1, minWidth: 4 }} />
                          <Tooltip title={graphStatsDetailsOpen ? 'Hide graph details' : 'Graph details'}>
                            <IconButton
                              size="small"
                              data-kg-stats-details-toggle="true"
                              aria-label={graphStatsDetailsOpen ? 'Hide graph details' : 'Show graph details'}
                              onClick={() => setGraphStatsDetailsOpen((open) => !open)}
                              sx={{
                                width: 24,
                                height: 24,
                                borderRadius: '8px',
                                color: graphStatsDetailsOpen ? accentColors.primary : 'text.secondary',
                                bgcolor: graphStatsDetailsOpen ? alpha(accentColors.primary, 0.12) : alpha('#fff', 0.04),
                                '&:hover': {
                                  color: 'text.primary',
                                  bgcolor: alpha(accentColors.primary, 0.14),
                                },
                              }}
                            >
                              {graphStatsDetailsOpen ? <PreviousResultIcon sx={{ fontSize: 16 }} /> : <MoreIcon sx={{ fontSize: 16 }} />}
                            </IconButton>
                          </Tooltip>
                        </Box>

                        <Collapse in={graphStatsDetailsOpen} timeout={160} unmountOnExit>
                          <Box
                            data-kg-stats-details="true"
                            sx={{
                              display: 'grid',
                              gap: 0.55,
                              pt: 0.65,
                              borderTop: `1px solid ${alpha('#fff', 0.08)}`,
                            }}
                          >
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.45, flexWrap: 'wrap', minWidth: 0 }}>
                              <Chip
                                size="small"
                                label={graphStatusLabel}
                                sx={{
                                  height: 22,
                                  color: graphStatusColor,
                                  bgcolor: alpha(graphStatusColor, 0.12),
                                  border: `1px solid ${alpha(graphStatusColor, 0.3)}`,
                                  '& .MuiChip-label': { px: 0.65, fontSize: '0.65rem', fontWeight: 850 },
                                }}
                              />
                              <Chip
                                size="small"
                                label={`${formatNumber(graphRelationshipCount)} stored relations`}
                                sx={{ height: 22, '& .MuiChip-label': { px: 0.65, fontSize: '0.65rem', fontWeight: 850 } }}
                              />
                              <Chip
                                size="small"
                                label={`${graphRenderProfilePreview.tier} / ${graphDetailMode}`}
                                sx={{
                                  height: 22,
                                  color: accentColors.primary,
                                  bgcolor: alpha(accentColors.primary, 0.1),
                                  border: `1px solid ${alpha(accentColors.primary, 0.24)}`,
                                  '& .MuiChip-label': { px: 0.65, fontSize: '0.65rem', fontWeight: 850 },
                                }}
                              />
                              <Chip
                                size="small"
                                label={graphNodeBudgetLabel}
                                sx={{ height: 22, '& .MuiChip-label': { px: 0.65, fontSize: '0.65rem', fontWeight: 850 } }}
                              />
                              <Chip
                                size="small"
                                label={graphLinkBudgetLabel}
                                sx={{ height: 22, '& .MuiChip-label': { px: 0.65, fontSize: '0.65rem', fontWeight: 850 } }}
                              />
                            </Box>
                            {(graphContextLabel || graphFiltersActive) && (
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.65, minWidth: 0 }}>
                                {graphContextLabel && (
                                  <Typography variant="caption" color="text.secondary" noWrap sx={{ flex: 1, minWidth: 0 }}>
                                    {graphContextLabel}
                                  </Typography>
                                )}
                                {graphFiltersActive && (
                                  <Button
                                    size="small"
                                    onClick={clearGraphFilters}
                                    sx={{
                                      height: 24,
                                      minWidth: 0,
                                      px: 0.8,
                                      borderRadius: '8px',
                                      fontSize: '0.64rem',
                                      fontWeight: 900,
                                      flexShrink: 0,
                                    }}
                                  >
                                    Clear
                                  </Button>
                                )}
                              </Box>
                            )}
                          </Box>
                        </Collapse>
                      </Box>
                    </Collapse>
                  </Box>
              </Box>
            )}
          </SectionCard>
        </Grid>

        {/* Sidebar */}
        {showInspector && (
            <Grid item xs={12} lg={4} xl={3} sx={{ minHeight: 0, height: '100%', display: 'flex' }}>
            <SectionCard
              title="Graph Inspector"
              subtitle={activeFocusLabel}
              icon={<InfoIcon />}
                accentColor={selectedEntity ? selectedEntityColors.main : selectedEdgeDetails?.dominantColor || accentColors.primary}
              fullHeight
                action={(
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Chip size="small" label={inspectorCountLabel} />
                    <Tooltip title="Hide inspector">
                      <IconButton
                        size="small"
                        aria-label="Hide graph inspector"
                        onClick={() => setInspectorOpen(false)}
                        sx={{ width: 28, height: 28, borderRadius: '8px' }}
                      >
                        <CloseIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Tooltip>
                  </Box>
                )}
              sx={{
                flex: 1,
                minHeight: 0,
                display: 'flex',
                flexDirection: 'column',
                '& .MuiCardContent-root': {
                  flex: 1,
                  minHeight: 0,
                  display: 'flex',
                  flexDirection: 'column',
                },
                '& .MuiCardContent-root > .MuiBox-root:last-child': {
                  flex: 1,
                  minHeight: 0,
                  overflow: 'auto',
                },
              }}
            >
              <TextField
                fullWidth
                size="small"
                  placeholder="Search graph"
                  value={searchTerm}
                  onChange={(e) => {
                    const nextValue = e.target.value;
                    const hasQuery = Boolean(nextValue.trim());
                    setSearchTerm(nextValue);
                    setSearchTrayOpen(hasQuery);
                    setSearchTrayExpanded(hasQuery);
                    setHoveredNodeId(null);
                    if (!hasQuery) {
                      setSearchPreviewNodeId(null);
                      setSearchPreviewEdgeId(null);
                    }
                  }}
                  onKeyDown={(event) => {
                    if (event.key === 'Escape') {
                      setSearchTerm('');
                      setSearchTrayOpen(false);
                      setSearchTrayExpanded(false);
                      setSearchPreviewNodeId(null);
                      setSearchPreviewEdgeId(null);
                      return;
                    }
                    if (searchResults.length === 0) return;
                    if (event.key === 'ArrowDown') {
                      event.preventDefault();
                      previewSearchResult(searchResultIndex + 1);
                      return;
                    }
                    if (event.key === 'ArrowUp') {
                      event.preventDefault();
                      previewSearchResult(searchResultIndex - 1);
                      return;
                    }
                    if (event.key === 'Enter') {
                      event.preventDefault();
                      activateSearchResult(searchResultIndex, event.shiftKey);
                    }
                  }}
                  InputProps={{
                    startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.disabled' }} />,
                    endAdornment: searchTerm ? (
                    <Tooltip title="Clear search">
                      <IconButton
                        size="small"
                        aria-label="Clear graph search"
                        onClick={() => {
                          setSearchTerm('');
                          setSearchTrayOpen(false);
                          setSearchTrayExpanded(false);
                          setSearchPreviewNodeId(null);
                          setSearchPreviewEdgeId(null);
                        }}
                        sx={{ width: 24, height: 24, color: 'text.secondary' }}
                      >
                        <CloseIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Tooltip>
                  ) : null,
                  }}
                  sx={{ mb: 1 }}
                />

                {searchQuery && (
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 0.65,
                      mb: 1,
                      minHeight: 30,
                      minWidth: 0,
                    }}
                  >
                    <Chip
                      size="small"
                      label={searchResults.length > 0 ? `${searchResultIndex + 1}/${searchResults.length}` : '0 matches'}
                      sx={{
                        height: 24,
                        flexShrink: 0,
                        color: searchResults.length > 0 ? accentColors.primary : 'text.secondary',
                        bgcolor: searchResults.length > 0 ? alpha(accentColors.primary, 0.12) : alpha('#fff', 0.05),
                        '& .MuiChip-label': { px: 0.75, fontSize: '0.68rem', fontWeight: 850 },
                      }}
                    />
                    <Box
                      onClick={() => activateSearchResult(searchResultIndex)}
                      sx={{
                        flex: 1,
                        minWidth: 0,
                        minHeight: selectedSearchResult?.kind === 'relation' && selectedSearchResult.route ? 44 : 30,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 0.65,
                        px: 0.75,
                        borderRadius: '8px',
                        border: '1px solid',
                        borderColor: selectedSearchResult ? alpha(selectedSearchResult.color, 0.28) : 'divider',
                        bgcolor: selectedSearchResult ? alpha(selectedSearchResult.color, 0.08) : alpha('#fff', 0.025),
                        cursor: selectedSearchResult ? 'pointer' : 'default',
                      }}
                    >
                        {selectedSearchResult
                          ? selectedSearchResult.kind === 'entity'
                            ? renderEntitySymbolMark(selectedSearchResult.entity?.entity_type, selectedSearchResult.entity?.name || selectedSearchResult.label, {
                              selected: true,
                              size: 23,
                            })
                            : renderRelationSymbolMark(
                              selectedSearchResult.edge?.type || selectedSearchResult.edge?.label,
                              selectedSearchResult.color,
                              { selected: true, size: 23 }
                            )
                          : <SearchIcon sx={{ fontSize: 16, color: 'text.disabled', flexShrink: 0 }} />}
                      <Box sx={{ minWidth: 0, flex: 1 }}>
                        {selectedSearchResult?.kind === 'relation' && selectedSearchResult.route ? (
                          <>
                            {renderSearchRelationRoute(selectedSearchResult, true)}
                            <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block', mt: 0.28, lineHeight: 1.05, fontSize: '0.58rem' }}>
                              {renderHighlightedSearchText(selectedSearchResult.snippet || selectedSearchResult.secondary, selectedSearchResult.color)}
                            </Typography>
                          </>
                        ) : (
                          <>
                            <Typography variant="caption" noWrap sx={{ display: 'block', fontWeight: 850, lineHeight: 1.15 }}>
                              {selectedSearchResult
                                ? renderHighlightedSearchText(selectedSearchResult.label, selectedSearchResult.color)
                                : 'No matches'}
                            </Typography>
                            {selectedSearchResult && (
                              <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block', lineHeight: 1.15 }}>
                                {renderHighlightedSearchText(selectedSearchResult.snippet || selectedSearchResult.secondary, selectedSearchResult.color)}
                              </Typography>
                            )}
                          </>
                        )}
                      </Box>
                    </Box>
                    <Tooltip title="Previous result">
                      <span>
                        <IconButton
                          size="small"
                          aria-label="Previous graph search result"
                          disabled={searchResults.length === 0}
                          onClick={activatePreviousSearchResult}
                          sx={{ width: 28, height: 28, borderRadius: '8px', border: '1px solid', borderColor: 'divider' }}
                        >
                          <PreviousResultIcon sx={{ fontSize: 18 }} />
                        </IconButton>
                      </span>
                    </Tooltip>
                    <Tooltip title="Next result">
                      <span>
                        <IconButton
                          size="small"
                          aria-label="Next graph search result"
                          disabled={searchResults.length === 0}
                          onClick={activateNextSearchResult}
                          sx={{ width: 28, height: 28, borderRadius: '8px', border: '1px solid', borderColor: 'divider' }}
                        >
                          <ExpandBundleIcon sx={{ fontSize: 18 }} />
                        </IconButton>
                      </span>
                    </Tooltip>
                  </Box>
                )}

                <Tabs
                value={tabValue}
                onChange={(_, v) => setTabValue(v)}
                    sx={{
                      mb: 1.5,
                      minHeight: 36,
                    '& .MuiTab-root': { minWidth: 96, minHeight: 36, px: 1.25 },
                  }}
                >
                  <Tab label="Entities" icon={<HubIcon />} iconPosition="start" />
                  <Tab label="Relations" icon={<RelationIcon />} iconPosition="start" />
                  <Tab label="Details" icon={<InfoIcon />} iconPosition="start" />
                </Tabs>

                {tabValue === 0 && (
                  <>
                      {visibleEntityTypeSummary.length > 0 && (
                      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
                        {visibleEntityTypeSummary.slice(0, 6).map(({ type, count }) => {
                          const colors = getTypeColors(type);
                          const selected = selectedEntityTypes.includes(type);
                          const active = selectedEntityTypes.length === 0 || selected;
                          return (
                            <Chip
                              key={type}
                              size="small"
                              icon={getEntityIcon(type, active ? colors.light : alpha(colors.light, 0.55))}
                              label={`${getTypeLabel(type)} ${count}`}
                              onClick={() => toggleEntityType(type)}
                              sx={{
                                height: 22,
                                opacity: active ? 1 : 0.45,
                                bgcolor: selected ? alpha(colors.main, 0.22) : alpha(colors.main, 0.08),
                                borderColor: alpha(colors.main, active ? 0.48 : 0.22),
                                color: active ? colors.light : 'text.secondary',
                                '& .MuiChip-label': { px: 0.7, fontSize: '0.68rem', fontWeight: 800 },
                              }}
                              variant={selected ? 'filled' : 'outlined'}
                            />
                          );
                        })}
                      </Box>
                    )}
                    <List sx={{ flex: 1, overflow: 'auto', pr: 0.5, py: 0 }}>
                    {filteredEntities.map(entity => {
                      const colors = getTypeColors(entity.entity_type, entity.name);
                      const degree = degreeMap.get(entity.id) || { in: 0, out: 0, total: 0 };
                      const entitySearchSnippet = searchQuery ? getEntitySearchSnippet(entity, searchQuery) : '';
                      return (
                        <ListItemButton
                          key={entity.id}
                          selected={selectedEntity?.id === entity.id}
                              onClick={() => {
                                setSelectedEdgeId(null);
                                setSelectedEntity(entity);
                                setSelectedNodeId(entity.id);
                                setTabValue(2);
                              }}
                            sx={{
                              borderRadius: '8px',
                              mb: 0.25,
                              py: 0.42,
                              px: 0.75,
                              borderLeft: `3px solid ${colors.main}`,
                              bgcolor: selectedEntity?.id === entity.id ? alpha(colors.main, 0.1) : 'transparent',
                              '&:hover': { bgcolor: alpha(colors.main, 0.1) },
                            }}
                          >
                            <ListItemText
                              sx={{ my: 0 }}
                              primary={
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, minWidth: 0 }}>
                                  {getEntityIcon(entity.entity_type, colors.light, entity.name)}
                                  <Typography variant="body2" noWrap sx={{ fontWeight: 800, minWidth: 0, flex: 1 }}>
                                    {entity.name}
                                  </Typography>
                                  <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    noWrap
                                    sx={{ flex: '0 1 74px', textAlign: 'right' }}
                                  >
                                    {getEntityDisplayTypeLabel(entity.entity_type, entity.name)}
                                  </Typography>
                                  <Chip
                                    label={degree.total}
                                    size="small"
                                  sx={{
                                    height: 18,
                                    minWidth: 24,
                                    bgcolor: alpha(colors.main, 0.16),
                                    color: colors.light,
                                    '& .MuiChip-label': { px: 0.6, fontSize: '0.65rem', fontWeight: 900 },
                                    }}
                                  />
                                </Box>
                              }
                              secondary={entitySearchSnippet ? (
                                <Typography
                                  variant="caption"
                                  color="text.secondary"
                                  sx={{
                                    display: '-webkit-box',
                                    WebkitLineClamp: 1,
                                    WebkitBoxOrient: 'vertical',
                                    overflow: 'hidden',
                                    lineHeight: 1.35,
                                    mt: 0.25,
                                    pl: 2.75,
                                  }}
                                >
                                  {entitySearchSnippet}
                                </Typography>
                              ) : null}
                            />
                          </ListItemButton>
                      );
                    })}
                    </List>
                  </>
                )}

                  {tabValue === 1 && (
                    <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
                      {relationTypeMatrix.items.length > 0 && (
                        <Box
                          data-kg-relation-matrix="true"
                          sx={{
                            display: 'grid',
                            gap: 0.55,
                            mb: 0.85,
                            p: 0.68,
                            borderRadius: '8px',
                            border: '1px solid',
                            borderColor: alpha(accentColors.primary, 0.16),
                            bgcolor: alpha('#020617', 0.22),
                          }}
                        >
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.55, minWidth: 0 }}>
                            <RelationIcon sx={{ fontSize: 15, color: accentColors.primary, flexShrink: 0 }} />
                            <Typography variant="caption" noWrap sx={{ flex: 1, minWidth: 0, fontWeight: 900, lineHeight: 1.1 }}>
                              Relation mix
                            </Typography>
                            <Chip
                              size="small"
                              label={`${formatNumber(relationTypeMatrix.total)} relations`}
                              sx={{
                                height: 19,
                                color: accentColors.primary,
                                bgcolor: alpha(accentColors.primary, 0.1),
                                '& .MuiChip-label': { px: 0.58, fontSize: '0.58rem', fontWeight: 900 },
                              }}
                            />
                            <Chip
                              size="small"
                              label={`${relationTypeMatrix.items.length} types`}
                              sx={{ height: 19, '& .MuiChip-label': { px: 0.58, fontSize: '0.58rem', fontWeight: 850 } }}
                            />
                          </Box>
                          <Box
                            sx={{
                              display: 'grid',
                              gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, minmax(0, 1fr))' },
                              gap: 0.38,
                            }}
                          >
                            {relationTypeMatrix.items.slice(0, 6).map((item) => {
                              const selected = selectedRelationshipTypes.includes(item.type);
                              const shareLabel = `${Math.max(1, Math.round(item.share * 100))}%`;
                              return (
                                <ButtonBase
                                  key={`relation-matrix-${item.type}`}
                                  data-kg-relation-matrix-item="true"
                                  data-kg-relation-matrix-selected={selected ? 'true' : 'false'}
                                  aria-label={`${selected ? 'Clear' : 'Focus'} ${getTypeLabel(item.type)} relationships`}
                                  onClick={() => toggleRelationshipTypeFocus(item.type)}
                                  sx={{
                                    minWidth: 0,
                                    minHeight: 34,
                                    px: 0.55,
                                    py: 0.42,
                                    display: 'grid',
                                    gap: 0.32,
                                    borderRadius: '8px',
                                    border: '1px solid',
                                    borderColor: alpha(item.color, selected ? 0.5 : 0.2),
                                    bgcolor: selected ? alpha(item.color, 0.15) : alpha(item.color, 0.055),
                                    color: 'text.primary',
                                    textAlign: 'left',
                                    overflow: 'hidden',
                                    transition: 'background-color 140ms ease, border-color 140ms ease, transform 140ms ease',
                                    '&:hover': {
                                      transform: 'translateY(-1px)',
                                      borderColor: alpha(item.color, 0.42),
                                      bgcolor: alpha(item.color, 0.11),
                                    },
                                  }}
                                >
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.42, minWidth: 0 }}>
                                    {renderRelationSymbolMark(item.type, item.color, { selected, size: 20 })}
                                    <Typography variant="caption" noWrap sx={{ minWidth: 0, flex: 1, fontSize: '0.62rem', fontWeight: 900, lineHeight: 1.05 }}>
                                      {getTypeLabel(item.type)}
                                    </Typography>
                                    <Typography variant="caption" noWrap sx={{ color: item.color, fontSize: '0.58rem', fontWeight: 900, lineHeight: 1 }}>
                                      {formatNumber(item.count)}
                                    </Typography>
                                    <Typography variant="caption" noWrap sx={{ color: 'text.disabled', fontSize: '0.55rem', fontWeight: 850, lineHeight: 1 }}>
                                      {shareLabel}
                                    </Typography>
                                  </Box>
                                  <Box sx={{ height: 3, borderRadius: '999px', overflow: 'hidden', bgcolor: alpha('#fff', 0.08) }}>
                                    <Box
                                      sx={{
                                        width: `${Math.max(7, Math.round(item.intensity * 100))}%`,
                                        height: '100%',
                                        borderRadius: '999px',
                                        bgcolor: item.color,
                                        opacity: selected ? 0.95 : 0.76,
                                      }}
                                    />
                                  </Box>
                                </ButtonBase>
                              );
                            })}
                          </Box>
                          {relationTypeMatrix.items.length > 6 && (
                            <Typography variant="caption" color="text.disabled" noWrap sx={{ px: 0.15, fontSize: '0.58rem', lineHeight: 1 }}>
                              +{relationTypeMatrix.items.length - 6} lower-volume relation types
                            </Typography>
                          )}
                        </Box>
                      )}
                      {selectedRelationshipTypeSummary.length > 0 && (
                      <Box data-kg-active-relation-filter="true" sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
                        {selectedRelationshipTypeSummary.map(({ type, count, color }) => (
                          <Chip
                            key={`active-relation-filter-${type}`}
                            size="small"
                            icon={renderRelationSymbolMark(type, color, { selected: true, size: 20 })}
                            label={`${getTypeLabel(type)} ${count}`}
                            onClick={() => {
                              setSelectedEdgeId(null);
                              setSelectedEntity(null);
                              setSelectedNodeId(null);
                              toggleRelationshipType(type);
                            }}
                            sx={{
                              height: 22,
                              color,
                              bgcolor: alpha(color, 0.22),
                              border: '1px solid',
                              borderColor: alpha(color, 0.48),
                              '& .MuiChip-label': { px: 0.7, fontSize: '0.66rem', fontWeight: 800 },
                              '& .MuiChip-icon': { color: 'inherit' },
                            }}
                          />
                        ))}
                        <Chip
                          size="small"
                          label="Clear"
                          onClick={() => setSelectedRelationshipTypes([])}
                          sx={{
                            height: 22,
                            color: 'text.secondary',
                            bgcolor: alpha('#fff', 0.07),
                            border: '1px solid',
                            borderColor: alpha('#fff', 0.12),
                            '& .MuiChip-label': { px: 0.7, fontSize: '0.66rem', fontWeight: 800 },
                          }}
                        />
                      </Box>
                    )}

                      <List sx={{ flex: 1, minHeight: 0, overflow: 'auto', pr: 0.5, py: 0 }}>
                        {relationBrowserRows.map((row) => {
                          const primaryType = row.relationTypes[0];
                          const hiddenTypeCount = Math.max(0, row.relationTypes.length - 1);
                          const evidencePreview = row.matchedEvidence || row.previewEvidence;
                          return (
                            <ListItemButton
                            key={row.edge.id}
                            selected={selectedEdgeId === row.edge.id}
                            onClick={() => handleLinkClick(row.edge)}
                              sx={{
                                borderRadius: '8px',
                                mb: 0.3,
                                py: 0.52,
                                px: 0.85,
                                alignItems: 'flex-start',
                                borderLeft: `3px solid ${alpha(row.dominantColor, 0.8)}`,
                                bgcolor: selectedEdgeId === row.edge.id ? alpha(row.dominantColor, 0.12) : 'transparent',
                              '&:hover': { bgcolor: alpha(row.dominantColor, 0.1) },
                            }}
                            >
                              <ListItemText
                                sx={{ my: 0 }}
                                primary={
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, minWidth: 0 }}>
                                    {getEntityIcon(row.source?.entity_type, getTypeColors(row.source?.entity_type, row.source?.name).light, row.source?.name)}
                                    <Box sx={{ minWidth: 0, flex: 1 }}>
                                      <Typography variant="body2" noWrap sx={{ fontWeight: 850, minWidth: 0 }}>
                                        {renderHighlightedSearchText(`${row.sourceLabel} -> ${row.targetLabel}`, row.dominantColor)}
                                      </Typography>
                                      <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block' }}>
                                        {primaryType ? getTypeLabel(primaryType.type) : 'Related'}
                                        {hiddenTypeCount > 0 ? ` +${hiddenTypeCount}` : ''} / {row.relationCount} relations
                                        {row.weight > 0 ? ` / w ${formatNumber(Math.round(row.weight))}` : ''}
                                        {row.domains[0] ? ` / ${truncateGraphLabel(row.domains[0], 22)}` : ''}
                                      </Typography>
                                    </Box>
                                    {primaryType && (
                                      <Chip
                                        size="small"
                                        icon={renderRelationSymbolMark(primaryType.type, primaryType.color, { selected: true, size: 18 })}
                                        label={primaryType.count}
                                        variant="outlined"
                                        sx={{
                                          height: 19,
                                          minWidth: 26,
                                          color: primaryType.color,
                                          bgcolor: alpha(primaryType.color, 0.1),
                                          borderColor: alpha(primaryType.color, 0.38),
                                          '& .MuiChip-label': { px: 0.56, fontSize: '0.61rem', fontWeight: 900 },
                                        }}
                                      />
                                    )}
                                  </Box>
                                }
                                secondary={
                                  evidencePreview ? (
                                    <Typography
                                      variant="caption"
                                      color="text.secondary"
                                      sx={{
                                        display: '-webkit-box',
                                        WebkitLineClamp: 1,
                                        WebkitBoxOrient: 'vertical',
                                        overflow: 'hidden',
                                        lineHeight: 1.35,
                                        mt: 0.35,
                                      }}
                                    >
                                      {renderHighlightedSearchText(evidencePreview, row.dominantColor)}
                                    </Typography>
                                  ) : null
                                }
                              />
                          </ListItemButton>
                        );
                      })}
                    </List>

                    {relationBrowserRows.length === 0 && (
                      <Box sx={{ py: 6, textAlign: 'center' }}>
                        <RelationIcon sx={{ fontSize: 38, color: 'text.disabled', mb: 1 }} />
                        <Typography variant="body2" color="text.secondary">
                          No relationships match the current filters.
                        </Typography>
                      </Box>
                    )}
                  </Box>
                )}

                {tabValue === 2 && selectedEdgeDetails && !selectedEntity && (
                  <Box sx={{ flex: 1, overflow: 'auto' }}>
                  <Box
                    sx={{
                      p: 1.5,
                      borderRadius: '8px',
                      bgcolor: alpha(selectedEdgeDetails.dominantColor, 0.1),
                      border: `1px solid ${alpha(selectedEdgeDetails.dominantColor, 0.22)}`,
                      mb: 1.5,
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 1, mb: 1 }}>
                      <Box sx={{ minWidth: 0 }}>
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.35, fontWeight: 800 }}>
                          Relationship
                        </Typography>
                        <Typography variant="subtitle1" noWrap sx={{ fontWeight: 900, lineHeight: 1.15 }}>
                          {selectedEdgeDetails.source?.name || selectedEdgeDetails.sourceId}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" noWrap>
                          {selectedEdgeDetails.target?.name || selectedEdgeDetails.targetId}
                        </Typography>
                      </Box>
                      <Chip
                        size="small"
                        label={`${selectedEdgeDetails.relationCount} relations`}
                        sx={{
                          height: 22,
                          flexShrink: 0,
                          color: selectedEdgeDetails.dominantColor,
                          bgcolor: alpha(selectedEdgeDetails.dominantColor, 0.14),
                          '& .MuiChip-label': { px: 0.75, fontSize: '0.66rem', fontWeight: 900 },
                        }}
                      />
                    </Box>

                    <Box sx={{ display: 'flex', gap: 0.6, flexWrap: 'wrap' }}>
                      {selectedEdgeDetails.source && (
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => {
                            if (!selectedEdgeDetails.source) return;
                            openGraphEntitySelection(selectedEdgeDetails.source, {
                              clearSearch: true,
                              closeInspector: true,
                            });
                          }}
                          sx={{ height: 26, borderRadius: '8px' }}
                        >
                          Source
                        </Button>
                      )}
                      {selectedEdgeDetails.target && (
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => {
                            if (!selectedEdgeDetails.target) return;
                            openGraphEntitySelection(selectedEdgeDetails.target, {
                              clearSearch: true,
                              closeInspector: true,
                            });
                          }}
                          sx={{ height: 26, borderRadius: '8px' }}
                        >
                          Target
                        </Button>
                      )}
                    </Box>
                  </Box>

                  <Box sx={{ display: 'grid', gridTemplateColumns: '84px minmax(0, 1fr)', gap: 0.55, mb: 1.5 }}>
                    <Typography variant="caption" color="text.disabled" noWrap>
                      Direction
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ wordBreak: 'break-word' }}>
                      {selectedEdgeDetails.source?.name || selectedEdgeDetails.sourceId} -&gt; {selectedEdgeDetails.target?.name || selectedEdgeDetails.targetId}
                    </Typography>
                    {selectedEdgeDetails.weight > 0 && (
                      <>
                        <Typography variant="caption" color="text.disabled" noWrap>
                          Weight
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {formatNumber(Math.round(selectedEdgeDetails.weight))}
                        </Typography>
                      </>
                    )}
                  </Box>

                  <Divider sx={{ my: 1.5 }} />

                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1, fontWeight: 800 }}>
                    Relation Types
                  </Typography>
                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mb: 1.5 }}>
                      {selectedEdgeDetails.relationTypes.map(({ type, count, color }) => (
                        <Chip
                          key={`${selectedEdgeDetails.edge.id}-${type}`}
                        icon={renderRelationSymbolMark(type, color, { selected: true, size: 18 })}
                        label={`${getTypeLabel(type)} ${count}`}
                        size="small"
                        onClick={() => focusRelationshipType(type)}
                        sx={{
                          height: 22,
                          bgcolor: alpha(color, 0.12),
                          borderColor: alpha(color, 0.45),
                          color,
                          cursor: 'pointer',
                          '& .MuiChip-label': { px: 0.7, fontSize: '0.66rem', fontWeight: 800 },
                        }}
                        variant="outlined"
                      />
                      ))}
                    </Box>

                    {selectedEdgeMetadataEntries.length > 0 && (
                      <>
                        <Divider sx={{ my: 1.5 }} />
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1, fontWeight: 800 }}>
                          Metadata
                        </Typography>
                        <MetadataGrid entries={selectedEdgeMetadataEntries} />
                      </>
                    )}

                    {(selectedEdgeDetails.domains.length > 0 || selectedEdgeDetails.evidenceSnippets.length > 0) && (
                      <>
                        <Divider sx={{ my: 1.5 }} />
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, mb: 1 }}>
                          <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 800 }}>
                            Evidence
                          </Typography>
                          {selectedEdgeDetails.domains.length > 0 && (
                            <Box sx={{ display: 'flex', gap: 0.4, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                              {selectedEdgeDetails.domains.slice(0, 2).map((domain) => (
                                <Chip
                                  key={`${selectedEdgeDetails.edge.id}-${domain}`}
                                  label={domain}
                                  size="small"
                                  sx={{
                                    height: 19,
                                    bgcolor: alpha('#fff', 0.05),
                                    '& .MuiChip-label': { px: 0.62, fontSize: '0.61rem', fontWeight: 800 },
                                  }}
                                />
                              ))}
                            </Box>
                          )}
                        </Box>

                        <Box sx={{ display: 'grid', gap: 0.75 }}>
                          {selectedEdgeDetails.evidenceSnippets.map((evidence, index) => {
                            const color = getRelationshipColor(evidence.type);
                            const evidenceMatchesSearch = Boolean(searchQuery && evidence.context.toLowerCase().includes(searchQuery));
                            return (
                              <Box
                                key={`${selectedEdgeDetails.edge.id}-evidence-${index}`}
                                sx={{
                                  p: 1,
                                  borderRadius: '8px',
                                  bgcolor: evidenceMatchesSearch ? alpha(color, 0.08) : alpha('#000', 0.14),
                                  border: `1px solid ${alpha(color, evidenceMatchesSearch ? 0.42 : 0.2)}`,
                                }}
                              >
                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, mb: 0.55 }}>
                                  <Chip
                                    label={getTypeLabel(evidence.type)}
                                    size="small"
                                    variant="outlined"
                                    sx={{
                                      height: 19,
                                      color,
                                      bgcolor: alpha(color, 0.1),
                                      borderColor: alpha(color, 0.4),
                                      '& .MuiChip-label': { px: 0.6, fontSize: '0.61rem', fontWeight: 800 },
                                    }}
                                  />
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.4, minWidth: 0 }}>
                                    {evidenceMatchesSearch && (
                                      <Chip
                                        label="match"
                                        size="small"
                                        sx={{
                                          height: 18,
                                          color,
                                          bgcolor: alpha(color, 0.14),
                                          '& .MuiChip-label': { px: 0.55, fontSize: '0.58rem', fontWeight: 850 },
                                        }}
                                      />
                                    )}
                                    {evidence.domain && (
                                      <Chip
                                        label={evidence.domain}
                                        size="small"
                                        sx={{
                                          height: 18,
                                          maxWidth: 116,
                                          bgcolor: alpha('#fff', 0.05),
                                          '& .MuiChip-label': {
                                            px: 0.55,
                                            fontSize: '0.58rem',
                                            fontWeight: 800,
                                            overflow: 'hidden',
                                            textOverflow: 'ellipsis',
                                          },
                                        }}
                                      />
                                    )}
                                    {evidence.weight > 0 && (
                                      <Typography variant="caption" color="text.disabled" noWrap>
                                        w {formatNumber(Math.round(evidence.weight))}
                                      </Typography>
                                    )}
                                  </Box>
                                </Box>
                                <Typography
                                  variant="caption"
                                  color="text.secondary"
                                  sx={{
                                    display: '-webkit-box',
                                    WebkitLineClamp: 3,
                                    WebkitBoxOrient: 'vertical',
                                    overflow: 'hidden',
                                    lineHeight: 1.45,
                                  }}
                                >
                                  {formatEvidenceSnippetDisplayText(evidence.context)}
                                </Typography>
                              </Box>
                            );
                          })}
                        </Box>
                      </>
                    )}
                  </Box>
                )}

                {tabValue === 2 && selectedEntity && (
                <Box sx={{ flex: 1, overflow: 'auto' }}>
                  <Box sx={{
                    p: 1.5,
                    borderRadius: '8px',
                    bgcolor: alpha(selectedEntityColors.main, 0.1),
                    mb: 1.5,
                  }}>
                    <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, mb: 1 }}>
                      {renderEntitySymbolMark(selectedEntity.entity_type, selectedEntity.name, {
                        selected: true,
                        size: 38,
                      })}
                      <Box sx={{ minWidth: 0, flex: 1 }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 900, lineHeight: 1.15 }}>
                          {selectedEntity.name}
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 0.65 }}>
                          <Chip
                          label={getEntityDisplayTypeLabel(selectedEntity.entity_type, selectedEntity.name)}
                            size="small"
                            sx={{
                              height: 22,
                              bgcolor: alpha(selectedEntityColors.main, 0.22),
                              color: selectedEntityColors.light,
                              '& .MuiChip-label': { px: 0.8, fontSize: '0.68rem', fontWeight: 800 },
                            }}
                          />
                          {selectedDegree && (
                            <Chip
                              label={`${selectedDegree.total} relations`}
                              size="small"
                              sx={{
                                height: 22,
                                bgcolor: alpha('#fff', 0.06),
                                '& .MuiChip-label': { px: 0.8, fontSize: '0.68rem', fontWeight: 800 },
                              }}
                            />
                          )}
                          {selectedEntity.confidence > 0 && (
                            <Chip
                              label={`Confidence ${Number(selectedEntity.confidence).toFixed(2)}`}
                              size="small"
                              sx={{
                                height: 22,
                                bgcolor: alpha('#fff', 0.06),
                                '& .MuiChip-label': { px: 0.8, fontSize: '0.68rem', fontWeight: 800 },
                              }}
                            />
                          )}
                        </Box>
                      </Box>
                    </Box>
                  </Box>

                  {selectedEntityGraphRole && (
                    <Box sx={{ mb: 1.5 }}>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.65, fontWeight: 800 }}>
                        Graph Role
                      </Typography>
                      <Box
                        sx={{
                          display: 'grid',
                          gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                          gap: 0.65,
                        }}
                      >
                        <Box sx={{ minWidth: 0 }}>
                          <Typography variant="caption" color="text.disabled" noWrap>
                            Rank
                          </Typography>
                          <Typography variant="body2" noWrap sx={{ fontWeight: 850 }}>
                            {selectedEntityGraphRole.rank
                              ? `#${selectedEntityGraphRole.rank} of ${formatNumber(selectedEntityGraphRole.totalEntities)}`
                              : 'Not ranked'}
                          </Typography>
                        </Box>
                        <Box sx={{ minWidth: 0 }}>
                          <Typography variant="caption" color="text.disabled" noWrap>
                            Component
                          </Typography>
                          <Typography variant="body2" noWrap sx={{ fontWeight: 850 }}>
                            {selectedEntityGraphRole.component
                              ? `${selectedEntityGraphRole.componentLabel} / ${formatNumber(selectedEntityGraphRole.component.nodeCount)} nodes`
                              : 'Isolated'}
                          </Typography>
                        </Box>
                        <Box sx={{ minWidth: 0 }}>
                          <Typography variant="caption" color="text.disabled" noWrap>
                            Direction
                          </Typography>
                          <Typography variant="body2" noWrap sx={{ fontWeight: 850 }}>
                            {selectedDegree ? `${selectedDegree.out} out / ${selectedDegree.in} in` : '0 out / 0 in'}
                          </Typography>
                        </Box>
                        <Box sx={{ minWidth: 0 }}>
                          <Typography variant="caption" color="text.disabled" noWrap>
                            Strongest Link
                          </Typography>
                          <Typography variant="body2" noWrap sx={{ fontWeight: 850 }}>
                            {selectedEntityGraphRole.strongestBundle?.otherEntity?.name ||
                              selectedEntityGraphRole.strongestBundle?.otherId ||
                              'None'}
                          </Typography>
                        </Box>
                      </Box>
                      {selectedEntityGraphRole.component && selectedComponentId !== selectedEntityGraphRole.component.id && (
                        <Button
                          size="small"
                          variant="text"
                          onClick={() => focusComponent(selectedEntityGraphRole.component!)}
                          sx={{ mt: 0.75, px: 0, minWidth: 0, height: 24, fontWeight: 850 }}
                        >
                          Focus component
                        </Button>
                      )}
                    </Box>
                  )}

                  {selectedEntity.description && (
                    <Box sx={{ mb: 1.5 }}>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5, fontWeight: 800 }}>
                        Description
                      </Typography>
                      <Typography variant="body2" sx={{ lineHeight: 1.5 }}>
                        {selectedEntity.description}
                      </Typography>
                    </Box>
                  )}

                  {selectedEntity.aliases.length > 0 && (
                    <Box sx={{ mb: 1.5 }}>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5, fontWeight: 800 }}>
                        Aliases
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {selectedEntity.aliases.map((alias, i) => (
                          <Chip key={i} label={alias} size="small" variant="outlined" />
                        ))}
                      </Box>
                    </Box>
                  )}

                  {selectedMetadataEntries.length > 0 && (
                    <Box sx={{ mb: 1.5 }}>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5, fontWeight: 800 }}>
                        Metadata
                      </Typography>
                      <MetadataGrid entries={selectedMetadataEntries} />
                    </Box>
                  )}

                  <Divider sx={{ my: 1.5 }} />

                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, mb: 1 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 800 }}>
                      Relationships ({relationships.length})
                    </Typography>
                    {selectedDegree && (
                      <Typography variant="caption" color="text.disabled">
                        {selectedDegree.out} out / {selectedDegree.in} in
                      </Typography>
                    )}
                  </Box>

                  {selectedRelationshipGroups.length > 0 && (
                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
                      {selectedRelationshipGroups.map(({ type, count, color }) => (
                        <Chip
                          key={type}
                          icon={renderRelationSymbolMark(type, color, { selected: true, size: 18 })}
                          label={`${getTypeLabel(type)} ${count}`}
                          size="small"
                          onClick={() => focusRelationshipType(type)}
                          sx={{
                            height: 21,
                            bgcolor: alpha(color, 0.12),
                            borderColor: alpha(color, 0.45),
                            color,
                            cursor: 'pointer',
                            '& .MuiChip-label': { px: 0.7, fontSize: '0.66rem', fontWeight: 800 },
                          }}
                          variant="outlined"
                        />
                      ))}
                    </Box>
                  )}

                  <List dense disablePadding>
                    {selectedRelationshipBundles.map(bundle => {
                      const otherEntity = bundle.otherEntity;
                      const topTypes = bundle.types.slice(0, 4);
                      const hiddenTypeCount = Math.max(0, bundle.types.length - topTypes.length);
                      const isExpanded = expandedRelationshipBundleId === bundle.otherId;
                      const directionLabel = [
                        bundle.outCount > 0 ? `${bundle.outCount} out` : null,
                        bundle.inCount > 0 ? `${bundle.inCount} in` : null,
                      ].filter(Boolean).join(' / ');
                      const dominantColor = topTypes[0]?.color || accentColors.primary;

                      return (
                          <Box key={bundle.otherId} sx={{ mb: 0.75 }}>
                          <ListItemButton
                              data-kg-neighbor-row="true"
                              data-kg-neighbor-row-action={otherEntity ? 'entity' : bundle.primaryEdge ? 'relation' : 'none'}
                              aria-label={otherEntity ? `Open linked entity ${otherEntity.name}` : bundle.primaryEdge ? `Open relationship details for ${bundle.otherId}` : `Open linked item ${bundle.otherId}`}
                              tabIndex={0}
                              onFocus={() => {
                                if (bundle.otherId) setHoveredNodeId(bundle.otherId);
                              }}
                              onBlur={() => {
                                if (!bundle.otherId) return;
                                setHoveredNodeId((current) => (current === bundle.otherId ? null : current));
                              }}
                              onMouseEnter={() => {
                                if (bundle.otherId) setHoveredNodeId(bundle.otherId);
                              }}
                              onPointerEnter={() => {
                                if (bundle.otherId) setHoveredNodeId(bundle.otherId);
                              }}
                              onPointerOver={() => {
                                if (bundle.otherId) setHoveredNodeId(bundle.otherId);
                              }}
                              onPointerMove={() => {
                                if (bundle.otherId) setHoveredNodeId(bundle.otherId);
                              }}
                              onPointerLeave={() => {
                                if (!bundle.otherId) return;
                                setHoveredNodeId((current) => (current === bundle.otherId ? null : current));
                              }}
                              onMouseLeave={() => {
                                if (!bundle.otherId) return;
                                setHoveredNodeId((current) => (current === bundle.otherId ? null : current));
                              }}
                              onMouseOver={() => {
                                if (bundle.otherId) setHoveredNodeId(bundle.otherId);
                              }}
                              onClick={() => {
                                if (otherEntity) {
                                  openGraphEntitySelection(otherEntity, {
                                    clearSearch: true,
                                    closeInspector: true,
                                  });
                                  return;
                                }
                                openGraphRelationshipDetails(bundle.primaryEdge, { expandPanel: true });
                            }}
                          sx={{
                            borderRadius: '8px',
                            bgcolor: alpha('#fff', 0.02),
                            borderLeft: `2px solid ${alpha(dominantColor, 0.75)}`,
                            py: 0.85,
                            px: 1,
                            alignItems: 'flex-start',
                            '&:hover': { bgcolor: alpha(dominantColor, 0.1) },
                          }}
                        >
                          <ListItemText
                            primary={
                              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, minWidth: 0 }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, minWidth: 0 }}>
                                  {renderEntitySymbolMark(otherEntity?.entity_type, otherEntity?.name || bundle.otherId, {
                                    selected: selectedNodeId === otherEntity?.id,
                                    size: 26,
                                  })}
                                  <Typography variant="body2" noWrap sx={{ minWidth: 0, fontWeight: 800 }}>
                                    {otherEntity?.name || bundle.otherId}
                                  </Typography>
                                </Box>
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.4, flexShrink: 0 }}>
                                    <Chip
                                      size="small"
                                      label={`${bundle.total} relations`}
                                    sx={{
                                      height: 20,
                                      bgcolor: alpha(dominantColor, 0.12),
                                      color: dominantColor,
                                        '& .MuiChip-label': { px: 0.65, fontSize: '0.64rem', fontWeight: 800 },
                                      }}
                                    />
                                    {bundle.primaryEdge && (
                                      <Tooltip title="Open relationship details">
                                        <IconButton
                                          size="small"
                                          data-kg-neighbor-relation-details="true"
                                          aria-label={`Open relationship details for ${otherEntity?.name || bundle.otherId}`}
                                          onClick={(event) => {
                                            event.stopPropagation();
                                            openGraphRelationshipDetails(bundle.primaryEdge, { expandPanel: true });
                                          }}
                                          sx={{
                                            width: 22,
                                            height: 22,
                                            color: dominantColor,
                                            '&:hover': {
                                              bgcolor: alpha(dominantColor, 0.14),
                                            },
                                            '& .MuiSvgIcon-root': { fontSize: 15 },
                                          }}
                                        >
                                          <RelationIcon sx={{ fontSize: 15 }} />
                                        </IconButton>
                                      </Tooltip>
                                    )}
                                    <Tooltip title={isExpanded ? 'Hide relation mix' : 'Show full relation mix'}>
                                      <IconButton
                                      size="small"
                                      aria-label={isExpanded ? 'Hide relation mix' : 'Show full relation mix'}
                                      onClick={(event) => {
                                        event.stopPropagation();
                                        setExpandedRelationshipBundleId(isExpanded ? null : bundle.otherId);
                                      }}
                                      sx={{
                                        width: 22,
                                        height: 22,
                                        color: 'text.secondary',
                                        '&:hover': { bgcolor: alpha(dominantColor, 0.14), color: dominantColor },
                                      }}
                                    >
                                      {isExpanded ? <ExpandBundleIcon sx={{ fontSize: 18 }} /> : <CollapseBundleIcon sx={{ fontSize: 18 }} />}
                                    </IconButton>
                                  </Tooltip>
                                </Box>
                              </Box>
                            }
                            secondary={
                              <Box sx={{ mt: 0.75 }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, mb: 0.65 }}>
                                  <Typography variant="caption" color="text.secondary" noWrap>
                                    {getEntityDisplayTypeLabel(otherEntity?.entity_type || 'custom', otherEntity?.name)} {directionLabel ? `- ${directionLabel}` : ''}
                                  </Typography>
                                  {bundle.totalWeight > 0 && (
                                    <Typography variant="caption" color="text.disabled">
                                      w {formatNumber(Math.round(bundle.totalWeight))}
                                    </Typography>
                                  )}
                                </Box>
                                <Box sx={{ display: 'flex', gap: 0.45, flexWrap: 'wrap' }}>
                                  {topTypes.map(({ type, count, color }) => (
                                    <Chip
                                      key={`${bundle.otherId}-${type}`}
                                      icon={renderRelationSymbolMark(type, color, { selected: true, size: 17 })}
                                      label={`${getTypeLabel(type)} ${count}`}
                                      size="small"
                                      onClick={(event) => {
                                        event.stopPropagation();
                                        focusRelationshipType(type);
                                      }}
                                      sx={{
                                        height: 19,
                                        bgcolor: alpha(color, 0.12),
                                        borderColor: alpha(color, 0.4),
                                        color,
                                        cursor: 'pointer',
                                        '& .MuiChip-label': { px: 0.6, fontSize: '0.62rem', fontWeight: 800 },
                                      }}
                                      variant="outlined"
                                    />
                                  ))}
                                  {hiddenTypeCount > 0 && (
                                    <Chip
                                      label={`+${hiddenTypeCount}`}
                                      size="small"
                                      sx={{
                                        height: 19,
                                        bgcolor: alpha('#fff', 0.05),
                                        '& .MuiChip-label': { px: 0.6, fontSize: '0.62rem', fontWeight: 800 },
                                      }}
                                    />
                                  )}
                                </Box>
                              </Box>
                            }
                          />
                        </ListItemButton>
                        <Collapse in={isExpanded} timeout="auto" unmountOnExit>
                          <Box
                            sx={{
                              ml: 1,
                              mt: 0.5,
                              mb: 0.25,
                              p: 1,
                              borderRadius: '8px',
                              bgcolor: alpha('#000', 0.16),
                              border: `1px solid ${alpha(dominantColor, 0.18)}`,
                            }}
                          >
                            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.75, fontWeight: 800 }}>
                              Full relation mix
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 0.45, flexWrap: 'wrap' }}>
                              {bundle.types.map(({ type, count, color }) => (
                                <Chip
                                  key={`${bundle.otherId}-expanded-${type}`}
                                  icon={renderRelationSymbolMark(type, color, { selected: true, size: 17 })}
                                  label={`${getTypeLabel(type)} ${count}`}
                                  size="small"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    focusRelationshipType(type);
                                  }}
                                  sx={{
                                    height: 20,
                                    bgcolor: alpha(color, 0.12),
                                    borderColor: alpha(color, 0.44),
                                    color,
                                    cursor: 'pointer',
                                    '& .MuiChip-label': { px: 0.65, fontSize: '0.63rem', fontWeight: 800 },
                                  }}
                                  variant="outlined"
                                />
                              ))}
                            </Box>
                          </Box>
                        </Collapse>
                        </Box>
                      );
                    })}
                  </List>
                </Box>
              )}

                {tabValue === 2 && !selectedEntity && !selectedEdgeDetails && (
                <Box sx={{ textAlign: 'center', py: 8 }}>
                  <InfoIcon sx={{ fontSize: 48, color: 'text.disabled', mb: 2 }} />
                  <Typography color="text.secondary">
                    Select an entity to view details
                  </Typography>
                </Box>
              )}
            </SectionCard>
          </Grid>
        )}
      </Grid>

      {/* Extract from Text Dialog */}
      <Dialog open={extractDialogOpen} onClose={() => setExtractDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle sx={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <UploadIcon sx={{ color: accentColors.success }} />
            Extract Entities from Text
          </Box>
        </DialogTitle>
        <DialogContent sx={{ pt: 3 }}>
          <Alert severity="info" sx={{ mb: 2 }}>
            Paste text below to automatically extract entities and relationships using GLiNER NER model.
            Extracted entities will be added to the Neo4j knowledge graph.
          </Alert>
          {selectedDomain && (
            <Alert severity="success" sx={{ mb: 2 }} icon={false}>
              Extracting to domain: <strong>{selectedDomain}</strong>
            </Alert>
          )}
          <TextField
            fullWidth
            multiline
            rows={12}
            placeholder="Enter or paste text here to extract entities and relationships..."
            value={extractText}
            onChange={(e) => setExtractText(e.target.value)}
            sx={{
              '& .MuiOutlinedInput-root': {
                fontFamily: 'monospace',
              }
            }}
          />
        </DialogContent>
        <DialogActions sx={{ p: 2, borderTop: '1px solid rgba(255,255,255,0.1)' }}>
          <Button onClick={() => setExtractDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleExtract} 
            variant="contained" 
            disabled={loading || !extractText.trim() || !graphRAGStatus.available}
            sx={{ background: gradients.success }}
          >
            {loading ? <CircularProgress size={20} /> : 'Extract Entities'}
          </Button>
        </DialogActions>
      </Dialog>
    </PageShell>
  );
};

export default KnowledgeGraphPage;
