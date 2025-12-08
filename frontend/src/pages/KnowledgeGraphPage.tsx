import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  Box,
  Paper,
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
  InputLabel,
  Select,
  MenuItem,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Tooltip,
  Alert,
  CircularProgress,
  Divider,
  Card,
  CardContent,
  Tabs,
  Tab,
  Autocomplete,
  Slider,
  Switch,
  FormControlLabel,
  Fade,
  Zoom as MuiZoom,
  Collapse,
  Badge,
  alpha,
  useTheme,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Search as SearchIcon,
  Refresh as RefreshIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  FitScreen as FitScreenIcon,
  AccountTree as GraphIcon,
  Hub as HubIcon,
  Link as LinkIcon,
  MergeType as MergeIcon,
  Upload as UploadIcon,
  Download as DownloadIcon,
  Fullscreen as FullscreenIcon,
  FullscreenExit as FullscreenExitIcon,
  FilterList as FilterIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  CenterFocusStrong as CenterIcon,
  Explore as ExploreIcon,
  Timeline as TimelineIcon,
  Layers as LayersIcon,
  Settings as SettingsIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Info as InfoIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import { apiService as api } from '../services/api';

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
  x: number;
  y: number;
  vx: number;
  vy: number;
  fx?: number | null;
  fy?: number | null;
  connections: number;
}

interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  weight: number;
}

// Constants
const ENTITY_TYPES = [
  'person', 'organization', 'location', 'date', 'event',
  'product', 'technology', 'concept', 'process', 'custom'
];

const RELATIONSHIP_TYPES = [
  'is_a', 'part_of', 'has_property', 'located_in', 'works_for',
  'created_by', 'depends_on', 'related_to', 'causes', 'precedes',
  'follows', 'contradicts', 'custom'
];

// Beautiful color palette for entity types
const TYPE_COLORS: Record<string, { main: string; light: string; glow: string }> = {
  person: { main: '#10B981', light: '#34D399', glow: 'rgba(16, 185, 129, 0.6)' },
  organization: { main: '#3B82F6', light: '#60A5FA', glow: 'rgba(59, 130, 246, 0.6)' },
  location: { main: '#F59E0B', light: '#FBBF24', glow: 'rgba(245, 158, 11, 0.6)' },
  date: { main: '#8B5CF6', light: '#A78BFA', glow: 'rgba(139, 92, 246, 0.6)' },
  event: { main: '#EF4444', light: '#F87171', glow: 'rgba(239, 68, 68, 0.6)' },
  product: { main: '#06B6D4', light: '#22D3EE', glow: 'rgba(6, 182, 212, 0.6)' },
  technology: { main: '#6366F1', light: '#818CF8', glow: 'rgba(99, 102, 241, 0.6)' },
  concept: { main: '#EC4899', light: '#F472B6', glow: 'rgba(236, 72, 153, 0.6)' },
  process: { main: '#14B8A6', light: '#2DD4BF', glow: 'rgba(20, 184, 166, 0.6)' },
  custom: { main: '#6B7280', light: '#9CA3AF', glow: 'rgba(107, 114, 128, 0.6)' },
};

// World-class Force-Directed Graph Visualization
const ForceGraphVisualization: React.FC<{
  nodes: GraphNode[];
  edges: GraphEdge[];
  selectedNode: string | null;
  hoveredNode: string | null;
  searchTerm: string;
  showLabels: boolean;
  showEdgeLabels: boolean;
  particleFlow: boolean;
  onNodeClick: (node: GraphNode) => void;
  onNodeHover: (node: GraphNode | null) => void;
  onNodeDragStart: (node: GraphNode) => void;
  onNodeDrag: (node: GraphNode, x: number, y: number) => void;
  onNodeDragEnd: (node: GraphNode) => void;
  isFullscreen: boolean;
  physicsEnabled: boolean;
}> = ({
  nodes,
  edges,
  selectedNode,
  hoveredNode,
  searchTerm,
  showLabels,
  showEdgeLabels,
  particleFlow,
  onNodeClick,
  onNodeHover,
  onNodeDragStart,
  onNodeDrag,
  onNodeDragEnd,
  isFullscreen,
  physicsEnabled,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [draggingNode, setDraggingNode] = useState<string | null>(null);
  const [particles, setParticles] = useState<Array<{ edge: string; progress: number; speed: number }>>([]);
  const nodesRef = useRef<Map<string, GraphNode>>(new Map());
  const theme = useTheme();

  // Initialize node positions
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;

    // Count connections for each node
    const connectionCount = new Map<string, number>();
    edges.forEach(edge => {
      connectionCount.set(edge.source, (connectionCount.get(edge.source) || 0) + 1);
      connectionCount.set(edge.target, (connectionCount.get(edge.target) || 0) + 1);
    });

    nodes.forEach((node, i) => {
      if (!nodesRef.current.has(node.id)) {
        const angle = (2 * Math.PI * i) / Math.max(nodes.length, 1);
        const radius = Math.min(width, height) * 0.35;
        nodesRef.current.set(node.id, {
          ...node,
          x: centerX + radius * Math.cos(angle) + (Math.random() - 0.5) * 50,
          y: centerY + radius * Math.sin(angle) + (Math.random() - 0.5) * 50,
          vx: 0,
          vy: 0,
          connections: connectionCount.get(node.id) || 0,
        });
      } else {
        const existing = nodesRef.current.get(node.id)!;
        existing.label = node.label;
        existing.type = node.type;
        existing.connections = connectionCount.get(node.id) || 0;
      }
    });

    // Remove nodes that no longer exist
    nodesRef.current.forEach((_, id) => {
      if (!nodes.find(n => n.id === id)) {
        nodesRef.current.delete(id);
      }
    });
  }, [nodes, edges]);

  // Initialize particles for edge flow
  useEffect(() => {
    if (particleFlow && edges.length > 0) {
      const newParticles = edges.flatMap(edge => 
        Array.from({ length: 3 }, () => ({
          edge: edge.id,
          progress: Math.random(),
          speed: 0.002 + Math.random() * 0.003,
        }))
      );
      setParticles(newParticles);
    } else {
      setParticles([]);
    }
  }, [particleFlow, edges.length]);

  // Physics simulation and rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    const render = () => {
      // Clear canvas with gradient background
      const gradient = ctx.createRadialGradient(
        width / 2, height / 2, 0,
        width / 2, height / 2, Math.max(width, height)
      );
      gradient.addColorStop(0, '#1a1a2e');
      gradient.addColorStop(0.5, '#16162a');
      gradient.addColorStop(1, '#0f0f1a');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);

      // Draw grid pattern
      ctx.save();
      ctx.translate(pan.x, pan.y);
      ctx.scale(zoom, zoom);

      // Subtle grid
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
      ctx.lineWidth = 1;
      const gridSize = 50;
      const startX = Math.floor(-pan.x / zoom / gridSize) * gridSize - gridSize;
      const startY = Math.floor(-pan.y / zoom / gridSize) * gridSize - gridSize;
      const endX = startX + width / zoom + gridSize * 2;
      const endY = startY + height / zoom + gridSize * 2;

      for (let x = startX; x < endX; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, startY);
        ctx.lineTo(x, endY);
        ctx.stroke();
      }
      for (let y = startY; y < endY; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(startX, y);
        ctx.lineTo(endX, y);
        ctx.stroke();
      }

      // Physics simulation
      if (physicsEnabled) {
        const nodeArray = Array.from(nodesRef.current.values());
        
        nodeArray.forEach(node => {
          if (node.fx !== null && node.fx !== undefined) return;

          let fx = 0, fy = 0;

          // Repulsion from other nodes
          nodeArray.forEach(other => {
            if (other.id === node.id) return;
            const dx = node.x - other.x;
            const dy = node.y - other.y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            const force = 8000 / (dist * dist);
            fx += (dx / dist) * force;
            fy += (dy / dist) * force;
          });

          // Attraction along edges
          edges.forEach(edge => {
            if (edge.source !== node.id && edge.target !== node.id) return;
            const otherId = edge.source === node.id ? edge.target : edge.source;
            const other = nodesRef.current.get(otherId);
            if (!other) return;
            const dx = other.x - node.x;
            const dy = other.y - node.y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            const force = dist * 0.03;
            fx += (dx / dist) * force;
            fy += (dy / dist) * force;
          });

          // Center gravity
          fx += (width / 2 - node.x) * 0.0005;
          fy += (height / 2 - node.y) * 0.0005;

          // Damping
          node.vx = (node.vx + fx) * 0.85;
          node.vy = (node.vy + fy) * 0.85;

          // Update position
          node.x += node.vx;
          node.y += node.vy;

          // Boundaries
          node.x = Math.max(50, Math.min(width - 50, node.x));
          node.y = Math.max(50, Math.min(height - 50, node.y));
        });
      }

      // Draw edges
      edges.forEach(edge => {
        const source = nodesRef.current.get(edge.source);
        const target = nodesRef.current.get(edge.target);
        if (!source || !target) return;

        const isHighlighted = selectedNode === edge.source || selectedNode === edge.target ||
                            hoveredNode === edge.source || hoveredNode === edge.target;

        // Edge glow for highlighted
        if (isHighlighted) {
          ctx.shadowColor = 'rgba(99, 102, 241, 0.8)';
          ctx.shadowBlur = 15;
        }

        // Draw curved edge
        const midX = (source.x + target.x) / 2;
        const midY = (source.y + target.y) / 2;
        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const offset = Math.sqrt(dx * dx + dy * dy) * 0.1;
        const ctrlX = midX - dy * 0.2;
        const ctrlY = midY + dx * 0.2;

        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.quadraticCurveTo(ctrlX, ctrlY, target.x, target.y);
        
        // Gradient edge
        const edgeGradient = ctx.createLinearGradient(source.x, source.y, target.x, target.y);
        const sourceColor = TYPE_COLORS[source.type]?.main || TYPE_COLORS.custom.main;
        const targetColor = TYPE_COLORS[target.type]?.main || TYPE_COLORS.custom.main;
        edgeGradient.addColorStop(0, alpha(sourceColor, isHighlighted ? 0.8 : 0.4));
        edgeGradient.addColorStop(1, alpha(targetColor, isHighlighted ? 0.8 : 0.4));
        
        ctx.strokeStyle = edgeGradient;
        ctx.lineWidth = isHighlighted ? 3 : 1.5;
        ctx.stroke();

        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;

        // Draw arrow
        const angle = Math.atan2(target.y - ctrlY, target.x - ctrlX);
        const arrowSize = isHighlighted ? 12 : 8;
        const arrowX = target.x - Math.cos(angle) * 25;
        const arrowY = target.y - Math.sin(angle) * 25;

        ctx.beginPath();
        ctx.moveTo(arrowX, arrowY);
        ctx.lineTo(
          arrowX - arrowSize * Math.cos(angle - Math.PI / 6),
          arrowY - arrowSize * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
          arrowX - arrowSize * Math.cos(angle + Math.PI / 6),
          arrowY - arrowSize * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fillStyle = alpha(targetColor, isHighlighted ? 0.9 : 0.6);
        ctx.fill();

        // Edge label
        if (showEdgeLabels && isHighlighted) {
          ctx.font = '10px Inter, sans-serif';
          ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
          ctx.textAlign = 'center';
          ctx.fillText(edge.label, ctrlX, ctrlY - 5);
        }
      });

      // Draw particles
      if (particleFlow) {
        particles.forEach((particle, i) => {
          const edge = edges.find(e => e.id === particle.edge);
          if (!edge) return;
          
          const source = nodesRef.current.get(edge.source);
          const target = nodesRef.current.get(edge.target);
          if (!source || !target) return;

          const t = particle.progress;
          const midX = (source.x + target.x) / 2;
          const midY = (source.y + target.y) / 2;
          const dx = target.x - source.x;
          const dy = target.y - source.y;
          const ctrlX = midX - dy * 0.2;
          const ctrlY = midY + dx * 0.2;

          // Quadratic bezier point
          const px = (1-t)*(1-t)*source.x + 2*(1-t)*t*ctrlX + t*t*target.x;
          const py = (1-t)*(1-t)*source.y + 2*(1-t)*t*ctrlY + t*t*target.y;

          const sourceColor = TYPE_COLORS[source.type]?.light || TYPE_COLORS.custom.light;
          
          ctx.beginPath();
          ctx.arc(px, py, 3, 0, Math.PI * 2);
          ctx.fillStyle = sourceColor;
          ctx.shadowColor = sourceColor;
          ctx.shadowBlur = 10;
          ctx.fill();
          ctx.shadowBlur = 0;

          // Update particle
          particles[i].progress += particle.speed;
          if (particles[i].progress > 1) {
            particles[i].progress = 0;
          }
        });
      }

      // Draw nodes
      Array.from(nodesRef.current.values()).forEach(node => {
        const colors = TYPE_COLORS[node.type] || TYPE_COLORS.custom;
        const isSelected = selectedNode === node.id;
        const isHovered = hoveredNode === node.id;
        const isSearchMatch = searchTerm && node.label.toLowerCase().includes(searchTerm.toLowerCase());
        const isHighlighted = isSelected || isHovered || isSearchMatch;
        
        // Node size based on connections
        const baseSize = 20 + Math.min(node.connections * 3, 15);
        const size = isHighlighted ? baseSize * 1.2 : baseSize;

        // Outer glow
        if (isHighlighted) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, size + 15, 0, Math.PI * 2);
          const glowGradient = ctx.createRadialGradient(
            node.x, node.y, size,
            node.x, node.y, size + 15
          );
          glowGradient.addColorStop(0, colors.glow);
          glowGradient.addColorStop(1, 'transparent');
          ctx.fillStyle = glowGradient;
          ctx.fill();
        }

        // Node shadow
        ctx.shadowColor = colors.glow;
        ctx.shadowBlur = isHighlighted ? 25 : 15;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;

        // Node gradient fill
        const nodeGradient = ctx.createRadialGradient(
          node.x - size * 0.3, node.y - size * 0.3, 0,
          node.x, node.y, size
        );
        nodeGradient.addColorStop(0, colors.light);
        nodeGradient.addColorStop(0.7, colors.main);
        nodeGradient.addColorStop(1, alpha(colors.main, 0.8));

        ctx.beginPath();
        ctx.arc(node.x, node.y, size, 0, Math.PI * 2);
        ctx.fillStyle = nodeGradient;
        ctx.fill();

        // Node border
        ctx.shadowBlur = 0;
        ctx.strokeStyle = isHighlighted ? '#fff' : alpha('#fff', 0.3);
        ctx.lineWidth = isHighlighted ? 3 : 1;
        ctx.stroke();

        // Inner highlight
        ctx.beginPath();
        ctx.arc(node.x - size * 0.25, node.y - size * 0.25, size * 0.3, 0, Math.PI * 2);
        ctx.fillStyle = alpha('#fff', 0.3);
        ctx.fill();

        // Labels
        if (showLabels || isHighlighted) {
          ctx.font = `${isHighlighted ? 'bold ' : ''}${isHighlighted ? 13 : 11}px Inter, sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          
          // Text shadow for readability
          ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
          ctx.shadowBlur = 4;
          ctx.fillStyle = '#fff';
          
          const label = node.label.length > 15 ? node.label.slice(0, 15) + '...' : node.label;
          ctx.fillText(label, node.x, node.y + size + 15);
          
          // Type badge
          ctx.font = '9px Inter, sans-serif';
          ctx.fillStyle = alpha('#fff', 0.7);
          ctx.fillText(node.type, node.x, node.y + size + 28);
          
          ctx.shadowBlur = 0;
        }
      });

      ctx.restore();

      animationRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [nodes, edges, selectedNode, hoveredNode, searchTerm, showLabels, showEdgeLabels, 
      particleFlow, particles, zoom, pan, physicsEnabled, theme]);

  // Mouse handlers
  const getMousePos = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left - pan.x) / zoom,
      y: (e.clientY - rect.top - pan.y) / zoom,
    };
  }, [zoom, pan]);

  const findNodeAtPos = useCallback((x: number, y: number): GraphNode | null => {
    for (const node of nodesRef.current.values()) {
      const size = 20 + Math.min(node.connections * 3, 15);
      const dx = x - node.x;
      const dy = y - node.y;
      if (dx * dx + dy * dy < size * size) {
        return node;
      }
    }
    return null;
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    const pos = getMousePos(e);
    const node = findNodeAtPos(pos.x, pos.y);
    
    if (node) {
      setDraggingNode(node.id);
      node.fx = node.x;
      node.fy = node.y;
      onNodeDragStart(node);
    } else {
      setIsPanning(true);
      setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const pos = getMousePos(e);
    
    if (draggingNode) {
      const node = nodesRef.current.get(draggingNode);
      if (node) {
        node.x = pos.x;
        node.y = pos.y;
        node.fx = pos.x;
        node.fy = pos.y;
        onNodeDrag(node, pos.x, pos.y);
      }
    } else if (isPanning) {
      setPan({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y,
      });
    } else {
      const node = findNodeAtPos(pos.x, pos.y);
      onNodeHover(node);
    }
  };

  const handleMouseUp = () => {
    if (draggingNode) {
      const node = nodesRef.current.get(draggingNode);
      if (node) {
        node.fx = null;
        node.fy = null;
        onNodeDragEnd(node);
      }
      setDraggingNode(null);
    }
    setIsPanning(false);
  };

  const handleClick = (e: React.MouseEvent) => {
    const pos = getMousePos(e);
    const node = findNodeAtPos(pos.x, pos.y);
    if (node) {
      onNodeClick(node);
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(0.1, Math.min(3, zoom * delta));
    
    // Zoom towards mouse position
    const rect = canvasRef.current?.getBoundingClientRect();
    if (rect) {
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const newPanX = mouseX - (mouseX - pan.x) * (newZoom / zoom);
      const newPanY = mouseY - (mouseY - pan.y) * (newZoom / zoom);
      setPan({ x: newPanX, y: newPanY });
    }
    
    setZoom(newZoom);
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  return (
    <Box sx={{ position: 'relative', width: '100%', height: isFullscreen ? '100vh' : 600 }}>
      <canvas
        ref={canvasRef}
        width={1200}
        height={isFullscreen ? 900 : 600}
        style={{
          width: '100%',
          height: '100%',
          cursor: draggingNode ? 'grabbing' : isPanning ? 'grabbing' : 'grab',
          borderRadius: 8,
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onClick={handleClick}
        onWheel={handleWheel}
      />
      
      {/* Controls overlay */}
      <Box
        sx={{
          position: 'absolute',
          bottom: 16,
          right: 16,
          display: 'flex',
          gap: 1,
          background: alpha('#000', 0.6),
          backdropFilter: 'blur(10px)',
          borderRadius: 2,
          p: 1,
        }}
      >
        <Tooltip title="Zoom In">
          <IconButton size="small" onClick={() => setZoom(z => Math.min(3, z * 1.2))} sx={{ color: '#fff' }}>
            <ZoomInIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="Zoom Out">
          <IconButton size="small" onClick={() => setZoom(z => Math.max(0.1, z * 0.8))} sx={{ color: '#fff' }}>
            <ZoomOutIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="Reset View">
          <IconButton size="small" onClick={resetView} sx={{ color: '#fff' }}>
            <CenterIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Zoom indicator */}
      <Box
        sx={{
          position: 'absolute',
          bottom: 16,
          left: 16,
          background: alpha('#000', 0.6),
          backdropFilter: 'blur(10px)',
          borderRadius: 1,
          px: 2,
          py: 0.5,
        }}
      >
        <Typography variant="caption" sx={{ color: '#fff' }}>
          {Math.round(zoom * 100)}%
        </Typography>
      </Box>
    </Box>
  );
};

// Main Page Component
const KnowledgeGraphPage: React.FC = () => {
  const theme = useTheme();
  const [entities, setEntities] = useState<Entity[]>([]);
  const [graphNodes, setGraphNodes] = useState<GraphNode[]>([]);
  const [graphEdges, setGraphEdges] = useState<GraphEdge[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [relationships, setRelationships] = useState<Relationship[]>([]);

  // View settings
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showLabels, setShowLabels] = useState(true);
  const [showEdgeLabels, setShowEdgeLabels] = useState(false);
  const [particleFlow, setParticleFlow] = useState(true);
  const [physicsEnabled, setPhysicsEnabled] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [tabValue, setTabValue] = useState(0);

  // Dialogs
  const [entityDialogOpen, setEntityDialogOpen] = useState(false);
  const [relationshipDialogOpen, setRelationshipDialogOpen] = useState(false);
  const [extractDialogOpen, setExtractDialogOpen] = useState(false);
  const [mergeDialogOpen, setMergeDialogOpen] = useState(false);

  // Form state
  const [entityForm, setEntityForm] = useState({
    name: '',
    entity_type: 'concept',
    description: '',
    aliases: '',
  });
  const [relationshipForm, setRelationshipForm] = useState({
    source_id: '',
    target_id: '',
    relationship_type: 'related_to',
    description: '',
  });
  const [extractText, setExtractText] = useState('');
  const [mergeIds, setMergeIds] = useState<string[]>([]);
  const [mergeName, setMergeName] = useState('');

  // Load data
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const entitiesRes = await api.get('/api/v1/rag/graph/entities', {
        params: { limit: 500, search: searchTerm || undefined }
      });
      setEntities(entitiesRes.data.entities || []);

      const graphRes = await api.get('/api/v1/rag/graph/visualize', {
        params: { limit: 500 }
      });
      
      // Transform nodes with initial positions
      const nodes = (graphRes.data.nodes || []).map((n: Record<string, unknown>, i: number) => ({
        ...n,
        x: 600 + Math.cos(2 * Math.PI * i / Math.max(graphRes.data.nodes.length, 1)) * 300,
        y: 300 + Math.sin(2 * Math.PI * i / Math.max(graphRes.data.nodes.length, 1)) * 200,
        vx: 0,
        vy: 0,
        connections: 0,
      }));
      
      setGraphNodes(nodes);
      setGraphEdges(graphRes.data.edges || []);

      const statsRes = await api.get('/api/v1/rag/graph/statistics');
      setStats(statsRes.data);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to load graph data';
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [searchTerm]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Load relationships for selected entity
  useEffect(() => {
    if (selectedNodeId) {
      api.get('/api/v1/rag/graph/relationships', {
        params: { entity_id: selectedNodeId }
      }).then(res => {
        setRelationships(res.data.relationships || []);
      }).catch(() => {});
    }
  }, [selectedNodeId]);

  // Handlers
  const handleCreateEntity = async () => {
    try {
      await api.post('/api/v1/rag/graph/entities', {
        name: entityForm.name,
        entity_type: entityForm.entity_type,
        description: entityForm.description,
        aliases: entityForm.aliases.split(',').map(a => a.trim()).filter(Boolean),
      });
      setEntityDialogOpen(false);
      setEntityForm({ name: '', entity_type: 'concept', description: '', aliases: '' });
      loadData();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to create entity');
    }
  };

  const handleDeleteEntity = async (id: string) => {
    if (!confirm('Delete this entity and all its relationships?')) return;
    try {
      await api.delete(`/api/v1/rag/graph/entities/${id}`);
      setSelectedNodeId(null);
      setSelectedEntity(null);
      loadData();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to delete entity');
    }
  };

  const handleCreateRelationship = async () => {
    try {
      await api.post('/api/v1/rag/graph/relationships', relationshipForm);
      setRelationshipDialogOpen(false);
      setRelationshipForm({ source_id: '', target_id: '', relationship_type: 'related_to', description: '' });
      loadData();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to create relationship');
    }
  };

  const handleExtract = async () => {
    try {
      setLoading(true);
      await api.post('/api/v1/rag/graph/extract', { text: extractText, save: true });
      setExtractDialogOpen(false);
      setExtractText('');
      loadData();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to extract entities');
    } finally {
      setLoading(false);
    }
  };

  const handleMerge = async () => {
    if (mergeIds.length < 2) {
      setError('Select at least 2 entities to merge');
      return;
    }
    try {
      await api.post('/api/v1/rag/graph/entities/merge', { entity_ids: mergeIds, merged_name: mergeName });
      setMergeDialogOpen(false);
      setMergeIds([]);
      setMergeName('');
      loadData();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to merge entities');
    }
  };

  const handleNodeClick = (node: GraphNode) => {
    setSelectedNodeId(node.id);
    const entity = entities.find(e => e.id === node.id);
    if (entity) setSelectedEntity(entity);
    setTabValue(1);
  };

  return (
    <Box sx={{ 
      p: isFullscreen ? 0 : 3, 
      height: isFullscreen ? '100vh' : 'auto',
      background: isFullscreen ? '#0f0f1a' : 'transparent',
    }}>
      {/* Header */}
      {!isFullscreen && (
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box>
            <Typography variant="h4" sx={{ 
              fontWeight: 700, 
              mb: 0.5,
              background: 'linear-gradient(135deg, #6366F1 0%, #EC4899 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}>
              Knowledge Graph
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Interactive visualization of entities and relationships
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              startIcon={<UploadIcon />}
              onClick={() => setExtractDialogOpen(true)}
              variant="outlined"
              sx={{ borderColor: alpha('#6366F1', 0.5) }}
            >
              Extract from Text
            </Button>
            <Button
              startIcon={<MergeIcon />}
              onClick={() => setMergeDialogOpen(true)}
              variant="outlined"
              sx={{ borderColor: alpha('#EC4899', 0.5) }}
            >
              Merge
            </Button>
            <Button
              startIcon={<AddIcon />}
              onClick={() => setEntityDialogOpen(true)}
              variant="contained"
              sx={{ 
                background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
                '&:hover': { background: 'linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%)' }
              }}
            >
              Add Entity
            </Button>
          </Box>
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Stats Cards */}
      {!isFullscreen && stats && (
        <Grid container spacing={2} sx={{ mb: 3 }}>
          {[
            { label: 'Entities', value: (stats as Record<string, number>).entity_count || 0, color: '#10B981', icon: <HubIcon /> },
            { label: 'Relationships', value: (stats as Record<string, number>).relationship_count || 0, color: '#3B82F6', icon: <LinkIcon /> },
            { label: 'Communities', value: (stats as Record<string, number>).community_count || 0, color: '#F59E0B', icon: <LayersIcon /> },
            { label: 'Entity Types', value: Object.keys((stats as Record<string, Record<string, number>>).entities_by_type || {}).length, color: '#8B5CF6', icon: <GraphIcon /> },
          ].map((stat, i) => (
            <Grid item xs={3} key={i}>
              <Card sx={{ 
                background: `linear-gradient(135deg, ${alpha(stat.color, 0.15)} 0%, ${alpha(stat.color, 0.05)} 100%)`,
                border: `1px solid ${alpha(stat.color, 0.3)}`,
                backdropFilter: 'blur(10px)',
              }}>
                <CardContent sx={{ display: 'flex', alignItems: 'center', gap: 2, py: 2 }}>
                  <Box sx={{ 
                    p: 1.5, 
                    borderRadius: 2, 
                    background: alpha(stat.color, 0.2),
                    color: stat.color,
                  }}>
                    {stat.icon}
                  </Box>
                  <Box>
                    <Typography variant="h4" sx={{ fontWeight: 700, color: stat.color }}>
                      {stat.value}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {stat.label}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      <Grid container spacing={3}>
        {/* Graph Visualization */}
        <Grid item xs={isFullscreen ? 12 : 8}>
          <Paper sx={{ 
            p: 2, 
            position: 'relative',
            background: 'linear-gradient(135deg, rgba(26, 26, 46, 0.9) 0%, rgba(15, 15, 26, 0.9) 100%)',
            border: '1px solid rgba(99, 102, 241, 0.2)',
            overflow: 'hidden',
          }}>
            {/* Graph header */}
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center', 
              mb: 2,
              pb: 2,
              borderBottom: '1px solid rgba(255,255,255,0.1)',
            }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <ExploreIcon sx={{ color: '#6366F1' }} />
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Graph Explorer
                </Typography>
                <Chip 
                  label={`${graphNodes.length} nodes`} 
                  size="small" 
                  sx={{ bgcolor: alpha('#6366F1', 0.2) }}
                />
                <Chip 
                  label={`${graphEdges.length} edges`} 
                  size="small" 
                  sx={{ bgcolor: alpha('#EC4899', 0.2) }}
                />
              </Box>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Tooltip title="Toggle Physics">
                  <IconButton 
                    onClick={() => setPhysicsEnabled(!physicsEnabled)}
                    sx={{ color: physicsEnabled ? '#10B981' : '#6B7280' }}
                  >
                    {physicsEnabled ? <PlayIcon /> : <PauseIcon />}
                  </IconButton>
                </Tooltip>
                <Tooltip title="Settings">
                  <IconButton onClick={() => setSettingsOpen(!settingsOpen)}>
                    <SettingsIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Refresh">
                  <IconButton onClick={loadData}>
                    <RefreshIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}>
                  <IconButton onClick={() => setIsFullscreen(!isFullscreen)}>
                    {isFullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>

            {/* Settings panel */}
            <Collapse in={settingsOpen}>
              <Box sx={{ 
                mb: 2, 
                p: 2, 
                borderRadius: 2, 
                bgcolor: alpha('#000', 0.3),
                display: 'flex',
                gap: 3,
                flexWrap: 'wrap',
              }}>
                <FormControlLabel
                  control={<Switch checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} />}
                  label="Show Labels"
                />
                <FormControlLabel
                  control={<Switch checked={showEdgeLabels} onChange={(e) => setShowEdgeLabels(e.target.checked)} />}
                  label="Edge Labels"
                />
                <FormControlLabel
                  control={<Switch checked={particleFlow} onChange={(e) => setParticleFlow(e.target.checked)} />}
                  label="Particle Flow"
                />
              </Box>
            </Collapse>

            {/* Graph */}
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', py: 15 }}>
                <CircularProgress sx={{ color: '#6366F1' }} />
              </Box>
            ) : graphNodes.length === 0 ? (
              <Box sx={{ textAlign: 'center', py: 15 }}>
                <HubIcon sx={{ fontSize: 80, color: alpha('#6366F1', 0.3), mb: 2 }} />
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  No entities in the knowledge graph
                </Typography>
                <Typography variant="body2" color="text.disabled" sx={{ mb: 3 }}>
                  Add entities manually or extract from text to get started
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<AddIcon />}
                  onClick={() => setEntityDialogOpen(true)}
                  sx={{ 
                    background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
                    mr: 2,
                  }}
                >
                  Add Entity
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<UploadIcon />}
                  onClick={() => setExtractDialogOpen(true)}
                >
                  Extract from Text
                </Button>
              </Box>
            ) : (
              <ForceGraphVisualization
                nodes={graphNodes}
                edges={graphEdges}
                selectedNode={selectedNodeId}
                hoveredNode={hoveredNodeId}
                searchTerm={searchTerm}
                showLabels={showLabels}
                showEdgeLabels={showEdgeLabels}
                particleFlow={particleFlow}
                onNodeClick={handleNodeClick}
                onNodeHover={(node) => setHoveredNodeId(node?.id || null)}
                onNodeDragStart={() => {}}
                onNodeDrag={() => {}}
                onNodeDragEnd={() => {}}
                isFullscreen={isFullscreen}
                physicsEnabled={physicsEnabled}
              />
            )}

            {/* Legend */}
            <Box sx={{ 
              mt: 2, 
              pt: 2,
              borderTop: '1px solid rgba(255,255,255,0.1)',
              display: 'flex', 
              flexWrap: 'wrap', 
              gap: 1,
              justifyContent: 'center',
            }}>
              {Object.entries(TYPE_COLORS).map(([type, colors]) => (
                <Chip
                  key={type}
                  label={type}
                  size="small"
                  sx={{
                    bgcolor: alpha(colors.main, 0.2),
                    borderColor: colors.main,
                    color: colors.light,
                    '& .MuiChip-label': { textTransform: 'capitalize' },
                  }}
                  variant="outlined"
                />
              ))}
            </Box>
          </Paper>
        </Grid>

        {/* Sidebar */}
        {!isFullscreen && (
          <Grid item xs={4}>
            <Paper sx={{ 
              p: 2, 
              height: 700, 
              display: 'flex', 
              flexDirection: 'column',
              background: 'linear-gradient(135deg, rgba(26, 26, 46, 0.9) 0%, rgba(15, 15, 26, 0.9) 100%)',
              border: '1px solid rgba(99, 102, 241, 0.2)',
            }}>
              <Tabs 
                value={tabValue} 
                onChange={(_, v) => setTabValue(v)} 
                sx={{ 
                  mb: 2,
                  '& .MuiTab-root': { minWidth: 0, flex: 1 },
                }}
              >
                <Tab label="Entities" icon={<HubIcon />} iconPosition="start" />
                <Tab label="Details" icon={<InfoIcon />} iconPosition="start" />
              </Tabs>

              {tabValue === 0 && (
                <>
                  <TextField
                    fullWidth
                    size="small"
                    placeholder="Search entities..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    InputProps={{
                      startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.disabled' }} />,
                    }}
                    sx={{ mb: 2 }}
                  />
                  <List sx={{ flex: 1, overflow: 'auto' }}>
                    {entities.filter(e => 
                      !searchTerm || e.name.toLowerCase().includes(searchTerm.toLowerCase())
                    ).map(entity => {
                      const colors = TYPE_COLORS[entity.entity_type] || TYPE_COLORS.custom;
                      return (
                        <ListItem
                          key={entity.id}
                          button
                          selected={selectedEntity?.id === entity.id}
                          onClick={() => {
                            setSelectedEntity(entity);
                            setSelectedNodeId(entity.id);
                            setTabValue(1);
                          }}
                          sx={{
                            borderRadius: 1,
                            mb: 0.5,
                            borderLeft: `3px solid ${colors.main}`,
                            bgcolor: selectedEntity?.id === entity.id ? alpha(colors.main, 0.1) : 'transparent',
                            '&:hover': { bgcolor: alpha(colors.main, 0.1) },
                          }}
                        >
                          <ListItemText
                            primary={entity.name}
                            secondary={
                              <Chip 
                                label={entity.entity_type} 
                                size="small" 
                                sx={{ 
                                  height: 20, 
                                  bgcolor: alpha(colors.main, 0.2),
                                  color: colors.light,
                                  '& .MuiChip-label': { px: 1, fontSize: 10 },
                                }}
                              />
                            }
                          />
                          <ListItemSecondaryAction>
                            <IconButton
                              size="small"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteEntity(entity.id);
                              }}
                              sx={{ '&:hover': { color: '#EF4444' } }}
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </ListItemSecondaryAction>
                        </ListItem>
                      );
                    })}
                  </List>
                </>
              )}

              {tabValue === 1 && selectedEntity && (
                <Box sx={{ flex: 1, overflow: 'auto' }}>
                  <Box sx={{ 
                    p: 2, 
                    borderRadius: 2, 
                    bgcolor: alpha(TYPE_COLORS[selectedEntity.entity_type]?.main || '#6B7280', 0.1),
                    mb: 2,
                  }}>
                    <Typography variant="h5" sx={{ fontWeight: 700, mb: 1 }}>
                      {selectedEntity.name}
                    </Typography>
                    <Chip
                      label={selectedEntity.entity_type}
                      sx={{
                        bgcolor: alpha(TYPE_COLORS[selectedEntity.entity_type]?.main || '#6B7280', 0.3),
                        color: TYPE_COLORS[selectedEntity.entity_type]?.light || '#fff',
                        textTransform: 'capitalize',
                      }}
                    />
                  </Box>
                  
                  {selectedEntity.description && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Description
                      </Typography>
                      <Typography variant="body2">
                        {selectedEntity.description}
                      </Typography>
                    </Box>
                  )}

                  {selectedEntity.aliases.length > 0 && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Aliases
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {selectedEntity.aliases.map((alias, i) => (
                          <Chip key={i} label={alias} size="small" variant="outlined" />
                        ))}
                      </Box>
                    </Box>
                  )}

                  <Divider sx={{ my: 2 }} />

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Relationships ({relationships.length})
                    </Typography>
                    <Button
                      size="small"
                      startIcon={<LinkIcon />}
                      onClick={() => {
                        setRelationshipForm(prev => ({ ...prev, source_id: selectedEntity.id }));
                        setRelationshipDialogOpen(true);
                      }}
                    >
                      Add
                    </Button>
                  </Box>

                  <List dense>
                    {relationships.map(rel => {
                      const isSource = rel.source_id === selectedEntity.id;
                      const otherId = isSource ? rel.target_id : rel.source_id;
                      const otherEntity = entities.find(e => e.id === otherId);
                      return (
                        <ListItem 
                          key={rel.id}
                          sx={{ 
                            borderRadius: 1, 
                            bgcolor: alpha('#fff', 0.02),
                            mb: 0.5,
                          }}
                        >
                          <ListItemText
                            primary={
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Chip 
                                  label={rel.relationship_type} 
                                  size="small" 
                                  sx={{ height: 20, fontSize: 10 }}
                                />
                                <Typography variant="body2">
                                  {isSource ? 'to' : 'from'} <strong>{otherEntity?.name || 'Unknown'}</strong>
                                </Typography>
                              </Box>
                            }
                            secondary={rel.description}
                          />
                        </ListItem>
                      );
                    })}
                  </List>

                  <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                    <Button
                      fullWidth
                      variant="outlined"
                      startIcon={<EditIcon />}
                      size="small"
                    >
                      Edit
                    </Button>
                    <Button
                      fullWidth
                      variant="outlined"
                      color="error"
                      startIcon={<DeleteIcon />}
                      size="small"
                      onClick={() => handleDeleteEntity(selectedEntity.id)}
                    >
                      Delete
                    </Button>
                  </Box>
                </Box>
              )}

              {tabValue === 1 && !selectedEntity && (
                <Box sx={{ textAlign: 'center', py: 8 }}>
                  <InfoIcon sx={{ fontSize: 48, color: 'text.disabled', mb: 2 }} />
                  <Typography color="text.secondary">
                    Select an entity to view details
                  </Typography>
                </Box>
              )}
            </Paper>
          </Grid>
        )}
      </Grid>

      {/* Create Entity Dialog */}
      <Dialog open={entityDialogOpen} onClose={() => setEntityDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AddIcon sx={{ color: '#6366F1' }} />
            Add New Entity
          </Box>
        </DialogTitle>
        <DialogContent sx={{ pt: 3 }}>
          <TextField
            fullWidth
            label="Entity Name"
            value={entityForm.name}
            onChange={(e) => setEntityForm(prev => ({ ...prev, name: e.target.value }))}
            sx={{ mb: 2 }}
          />
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Entity Type</InputLabel>
            <Select
              value={entityForm.entity_type}
              label="Entity Type"
              onChange={(e) => setEntityForm(prev => ({ ...prev, entity_type: e.target.value }))}
            >
              {ENTITY_TYPES.map(type => (
                <MenuItem key={type} value={type}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ 
                      width: 12, 
                      height: 12, 
                      borderRadius: '50%', 
                      bgcolor: TYPE_COLORS[type]?.main || TYPE_COLORS.custom.main 
                    }} />
                    <span style={{ textTransform: 'capitalize' }}>{type}</span>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <TextField
            fullWidth
            label="Description"
            value={entityForm.description}
            onChange={(e) => setEntityForm(prev => ({ ...prev, description: e.target.value }))}
            multiline
            rows={3}
            sx={{ mb: 2 }}
          />
          <TextField
            fullWidth
            label="Aliases (comma-separated)"
            value={entityForm.aliases}
            onChange={(e) => setEntityForm(prev => ({ ...prev, aliases: e.target.value }))}
            placeholder="alias1, alias2, alias3"
          />
        </DialogContent>
        <DialogActions sx={{ p: 2, borderTop: '1px solid rgba(255,255,255,0.1)' }}>
          <Button onClick={() => setEntityDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleCreateEntity} 
            variant="contained"
            disabled={!entityForm.name}
            sx={{ background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)' }}
          >
            Create Entity
          </Button>
        </DialogActions>
      </Dialog>

      {/* Create Relationship Dialog */}
      <Dialog open={relationshipDialogOpen} onClose={() => setRelationshipDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <LinkIcon sx={{ color: '#EC4899' }} />
            Add Relationship
          </Box>
        </DialogTitle>
        <DialogContent sx={{ pt: 3 }}>
          <Autocomplete
            options={entities}
            getOptionLabel={(option) => option.name}
            value={entities.find(e => e.id === relationshipForm.source_id) || null}
            onChange={(_, value) => setRelationshipForm(prev => ({ ...prev, source_id: value?.id || '' }))}
            renderInput={(params) => <TextField {...params} label="Source Entity" />}
            sx={{ mb: 2 }}
          />
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Relationship Type</InputLabel>
            <Select
              value={relationshipForm.relationship_type}
              label="Relationship Type"
              onChange={(e) => setRelationshipForm(prev => ({ ...prev, relationship_type: e.target.value }))}
            >
              {RELATIONSHIP_TYPES.map(type => (
                <MenuItem key={type} value={type}>
                  <span style={{ textTransform: 'capitalize' }}>{type.replace('_', ' ')}</span>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Autocomplete
            options={entities}
            getOptionLabel={(option) => option.name}
            value={entities.find(e => e.id === relationshipForm.target_id) || null}
            onChange={(_, value) => setRelationshipForm(prev => ({ ...prev, target_id: value?.id || '' }))}
            renderInput={(params) => <TextField {...params} label="Target Entity" />}
            sx={{ mb: 2 }}
          />
          <TextField
            fullWidth
            label="Description (optional)"
            value={relationshipForm.description}
            onChange={(e) => setRelationshipForm(prev => ({ ...prev, description: e.target.value }))}
          />
        </DialogContent>
        <DialogActions sx={{ p: 2, borderTop: '1px solid rgba(255,255,255,0.1)' }}>
          <Button onClick={() => setRelationshipDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleCreateRelationship} 
            variant="contained"
            disabled={!relationshipForm.source_id || !relationshipForm.target_id}
            sx={{ background: 'linear-gradient(135deg, #EC4899 0%, #8B5CF6 100%)' }}
          >
            Create Relationship
          </Button>
        </DialogActions>
      </Dialog>

      {/* Extract from Text Dialog */}
      <Dialog open={extractDialogOpen} onClose={() => setExtractDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle sx={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <UploadIcon sx={{ color: '#10B981' }} />
            Extract Entities from Text
          </Box>
        </DialogTitle>
        <DialogContent sx={{ pt: 3 }}>
          <Alert severity="info" sx={{ mb: 2 }}>
            Paste text below to automatically extract entities and relationships using AI.
            The extracted information will be added to your knowledge graph.
          </Alert>
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
            disabled={loading || !extractText.trim()}
            sx={{ background: 'linear-gradient(135deg, #10B981 0%, #06B6D4 100%)' }}
          >
            {loading ? <CircularProgress size={20} /> : 'Extract & Save'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Merge Entities Dialog */}
      <Dialog open={mergeDialogOpen} onClose={() => setMergeDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <MergeIcon sx={{ color: '#F59E0B' }} />
            Merge Entities
          </Box>
        </DialogTitle>
        <DialogContent sx={{ pt: 3 }}>
          <Alert severity="warning" sx={{ mb: 2 }}>
            Merging will combine the selected entities into one. All relationships will be transferred to the merged entity.
          </Alert>
          <Autocomplete
            multiple
            options={entities}
            getOptionLabel={(option) => option.name}
            value={entities.filter(e => mergeIds.includes(e.id))}
            onChange={(_, values) => setMergeIds(values.map(v => v.id))}
            renderInput={(params) => <TextField {...params} label="Select Entities to Merge" />}
            sx={{ mb: 2 }}
          />
          <TextField
            fullWidth
            label="Name for Merged Entity"
            value={mergeName}
            onChange={(e) => setMergeName(e.target.value)}
            placeholder="Enter the name for the resulting entity"
          />
        </DialogContent>
        <DialogActions sx={{ p: 2, borderTop: '1px solid rgba(255,255,255,0.1)' }}>
          <Button onClick={() => setMergeDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleMerge} 
            variant="contained" 
            disabled={mergeIds.length < 2 || !mergeName.trim()}
            sx={{ background: 'linear-gradient(135deg, #F59E0B 0%, #EF4444 100%)' }}
          >
            Merge Entities
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default KnowledgeGraphPage;
