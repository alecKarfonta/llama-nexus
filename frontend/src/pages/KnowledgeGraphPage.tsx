import React, { useState, useEffect, useCallback, useRef } from 'react';
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
  Select,
  MenuItem,
  Chip,
  List,
  ListItem,
  ListItemText,
  Tooltip,
  Alert,
  CircularProgress,
  Divider,
  Card,
  CardContent,
  Tabs,
  Tab,
  Switch,
  FormControlLabel,
  Collapse,
  alpha,
  useTheme,
} from '@mui/material';
import {
  Add as AddIcon,
  Search as SearchIcon,
  Refresh as RefreshIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  AccountTree as GraphIcon,
  Hub as HubIcon,
  Link as LinkIcon,
  Upload as UploadIcon,
  Fullscreen as FullscreenIcon,
  FullscreenExit as FullscreenExitIcon,
  CenterFocusStrong as CenterIcon,
  Explore as ExploreIcon,
  Layers as LayersIcon,
  Settings as SettingsIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import ForceGraph2D, { ForceGraphMethods } from 'react-force-graph-2d';
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
  val?: number; // Node size for react-force-graph
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

// Force-Directed Graph Visualization using react-force-graph-2d
interface ForceGraphProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  selectedNode: string | null;
  searchTerm: string;
  showLabels: boolean;
  showEdgeLabels: boolean;
  particleFlow: boolean;
  onNodeClick: (node: GraphNode) => void;
  onNodeHover: (node: GraphNode | null) => void;
  isFullscreen: boolean;
  physicsEnabled: boolean;
}

const ForceGraphVisualization: React.FC<ForceGraphProps> = ({
  nodes,
  edges,
  selectedNode,
  searchTerm,
  showLabels,
  showEdgeLabels,
  particleFlow,
  onNodeClick,
  onNodeHover,
  isFullscreen,
  physicsEnabled,
}) => {
  const fgRef = useRef<ForceGraphMethods>();
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: isFullscreen ? window.innerHeight - 100 : 550,
        });
      }
    };
    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, [isFullscreen]);

  // Prepare graph data for react-force-graph
  const graphData = React.useMemo(() => ({
    nodes: nodes.map(n => ({
      ...n,
      val: Math.max(3, Math.min(15, (n.confidence || 1) * 5)), // Node size based on confidence
    })),
    links: edges.map(e => ({
      source: e.source,
      target: e.target,
      label: e.label,
      weight: e.weight,
    })),
  }), [nodes, edges]);

  // Handle zoom controls
  const handleZoomIn = useCallback(() => {
    fgRef.current?.zoom(fgRef.current.zoom() * 1.3, 400);
  }, []);

  const handleZoomOut = useCallback(() => {
    fgRef.current?.zoom(fgRef.current.zoom() / 1.3, 400);
  }, []);

  const handleCenterGraph = useCallback(() => {
    fgRef.current?.zoomToFit(400, 50);
  }, []);

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

  // Custom node rendering
  const paintNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const colors = TYPE_COLORS[node.type] || TYPE_COLORS.custom;
    const isSelected = selectedNode === node.id;
    const isSearchMatch = searchTerm && node.label?.toLowerCase().includes(searchTerm.toLowerCase());
    const isHighlighted = isSelected || isSearchMatch;
    
    const nodeSize = node.val || 5;
    const size = isHighlighted ? nodeSize * 1.4 : nodeSize;

    // Glow effect for highlighted nodes
    if (isHighlighted) {
      ctx.shadowColor = colors.glow;
      ctx.shadowBlur = 15;
    }

    // Draw node circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
    ctx.fillStyle = colors.main;
    ctx.fill();
    
    // Border
    ctx.strokeStyle = isHighlighted ? '#fff' : alpha(colors.light, 0.5);
    ctx.lineWidth = isHighlighted ? 2 : 1;
    ctx.stroke();
    
    ctx.shadowBlur = 0;

    // Draw label if enabled or highlighted
    if (showLabels || isHighlighted) {
      const label = node.label || node.id;
      const fontSize = Math.max(10, 12 / globalScale);
      ctx.font = `${isHighlighted ? 'bold ' : ''}${fontSize}px Inter, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      
      // Text background for readability
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(node.x - textWidth / 2 - 2, node.y + size + 2, textWidth + 4, fontSize + 4);
      
      ctx.fillStyle = isHighlighted ? '#fff' : 'rgba(255, 255, 255, 0.8)';
      ctx.fillText(label.length > 20 ? label.slice(0, 20) + '...' : label, node.x, node.y + size + 4);
    }
  }, [selectedNode, searchTerm, showLabels]);

  // Custom link rendering
  const paintLink = useCallback((link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const sourceNode = link.source;
    const targetNode = link.target;
    
    if (!sourceNode.x || !targetNode.x) return;

    const sourceColors = TYPE_COLORS[sourceNode.type] || TYPE_COLORS.custom;
    const targetColors = TYPE_COLORS[targetNode.type] || TYPE_COLORS.custom;
    
    // Create gradient
    const gradient = ctx.createLinearGradient(sourceNode.x, sourceNode.y, targetNode.x, targetNode.y);
    gradient.addColorStop(0, alpha(sourceColors.main, 0.6));
    gradient.addColorStop(1, alpha(targetColors.main, 0.6));
    
    ctx.strokeStyle = gradient;
    ctx.lineWidth = Math.max(0.5, Math.min(3, (link.weight || 1) / 50));
    
    ctx.beginPath();
    ctx.moveTo(sourceNode.x, sourceNode.y);
    ctx.lineTo(targetNode.x, targetNode.y);
    ctx.stroke();

    // Draw arrow
    const angle = Math.atan2(targetNode.y - sourceNode.y, targetNode.x - sourceNode.x);
    const arrowLength = 6;
    const targetSize = targetNode.val || 5;
    const arrowX = targetNode.x - Math.cos(angle) * (targetSize + 3);
    const arrowY = targetNode.y - Math.sin(angle) * (targetSize + 3);
    
    ctx.fillStyle = targetColors.main;
    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(arrowX - arrowLength * Math.cos(angle - Math.PI / 6), arrowY - arrowLength * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(arrowX - arrowLength * Math.cos(angle + Math.PI / 6), arrowY - arrowLength * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();

    // Draw edge label if enabled
    if (showEdgeLabels && link.label) {
      const midX = (sourceNode.x + targetNode.x) / 2;
      const midY = (sourceNode.y + targetNode.y) / 2;
      const fontSize = Math.max(8, 10 / globalScale);
      ctx.font = `${fontSize}px Inter, sans-serif`;
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.textAlign = 'center';
      ctx.fillText(link.label, midX, midY);
    }
  }, [showEdgeLabels]);

  return (
    <Box 
      ref={containerRef} 
      sx={{ 
        position: 'relative', 
        width: '100%', 
        height: isFullscreen ? 'calc(100vh - 100px)' : 550,
        background: 'linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%)',
        borderRadius: 2,
        overflow: 'hidden',
      }}
    >
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        width={dimensions.width}
        height={dimensions.height}
        backgroundColor="rgba(0,0,0,0)"
        nodeCanvasObject={paintNode}
        linkCanvasObject={paintLink}
        onNodeClick={(node: any) => onNodeClick(node as GraphNode)}
        onNodeHover={(node: any) => onNodeHover(node as GraphNode | null)}
        enableNodeDrag={true}
        enableZoomInteraction={true}
        enablePanInteraction={true}
        linkDirectionalParticles={particleFlow ? 2 : 0}
        linkDirectionalParticleSpeed={0.005}
        linkDirectionalParticleWidth={3}
        linkDirectionalParticleColor={(link: any) => {
          const sourceColors = TYPE_COLORS[link.source?.type] || TYPE_COLORS.custom;
          return sourceColors.light;
        }}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
        warmupTicks={100}
        cooldownTicks={200}
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
          <IconButton size="small" onClick={handleZoomIn} sx={{ color: '#fff' }}>
            <ZoomInIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="Zoom Out">
          <IconButton size="small" onClick={handleZoomOut} sx={{ color: '#fff' }}>
            <ZoomOutIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="Fit to View">
          <IconButton size="small" onClick={handleCenterGraph} sx={{ color: '#fff' }}>
            <CenterIcon />
          </IconButton>
        </Tooltip>
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

  // GraphRAG service state
  const [graphRAGStatus, setGraphRAGStatus] = useState<{
    available: boolean;
    status: string;
    nerModel?: string;
    device?: string;
    gpuName?: string;
  }>({ available: false, status: 'checking' });
  const [graphRAGDomains, setGraphRAGDomains] = useState<string[]>([]);
  const [selectedDomain, setSelectedDomain] = useState<string>('');

  // View settings
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showLabels, setShowLabels] = useState(true);
  const [showEdgeLabels, setShowEdgeLabels] = useState(false);
  const [particleFlow, setParticleFlow] = useState(true);
  const [physicsEnabled, setPhysicsEnabled] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [tabValue, setTabValue] = useState(0);

  // Dialogs
  const [extractDialogOpen, setExtractDialogOpen] = useState(false);

  // Form state
  const [extractText, setExtractText] = useState('');

  // Check GraphRAG service status
  const checkGraphRAGStatus = useCallback(async () => {
    try {
      const healthRes = await api.getGraphRAGHealth();
      if (healthRes.status === 'healthy') {
        setGraphRAGStatus({
          available: true,
          status: 'healthy',
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
        max_entities: 500,
        max_relationships: 500,
      });
      
      // Transform nodes for visualization
      const nodes = (graphRes.nodes || []).map((n: any) => ({
        id: n.id,
        label: n.label,
        type: n.type,
        description: n.properties?.description,
        confidence: n.properties?.confidence || n.occurrence || 1,
      }));
      
      // Transform edges
      const edges = (graphRes.edges || []).map((e: any) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        label: e.label || e.type,
        weight: e.weight || 1,
      }));
      
      setGraphNodes(nodes);
      setGraphEdges(edges);
      
      // Transform to entities for sidebar
      const entitiesList = nodes.map((n: any) => ({
        id: n.id,
        name: n.label,
        entity_type: n.type,
        description: n.description || '',
        aliases: [],
        properties: {},
        confidence: n.confidence,
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
    if (selectedNodeId && graphRAGStatus.available) {
      // Filter edges that involve this node
      const nodeRels = graphEdges
        .filter(e => e.source === selectedNodeId || e.target === selectedNodeId)
        .map(e => ({
          id: e.id,
          source_id: e.source,
          target_id: e.target,
          relationship_type: e.label,
          description: '',
          weight: e.weight,
        }));
      setRelationships(nodeRels);
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
      console.log(`Extracted ${result.entity_count} entities and ${result.relationship_count} relationships using ${result.extraction_method}`, result);
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
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="body2" color="text.secondary">
                Powered by GraphRAG with Neo4j + GLiNER
              </Typography>
              <Tooltip title={
                graphRAGStatus.available 
                  ? `${graphRAGStatus.nerModel || 'GLiNER'} on ${graphRAGStatus.gpuName || graphRAGStatus.device || 'GPU'}`
                  : graphRAGStatus.status === 'checking' ? 'Checking service...' : 'GraphRAG service unavailable'
              }>
                <Box sx={{ 
                  width: 10, 
                  height: 10, 
                  borderRadius: '50%', 
                  bgcolor: graphRAGStatus.available ? '#10B981' : graphRAGStatus.status === 'checking' ? '#F59E0B' : '#EF4444',
                }} />
              </Tooltip>
            </Box>
          </Box>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            {/* Domain selector */}
            {graphRAGStatus.available && graphRAGDomains.length > 0 && (
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <Select
                  value={selectedDomain}
                  onChange={(e) => setSelectedDomain(e.target.value)}
                  displayEmpty
                  sx={{ height: 36 }}
                >
                  <MenuItem value="">All Domains</MenuItem>
                  {graphRAGDomains.map(d => (
                    <MenuItem key={d} value={d}>{d}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            )}
            <Button
              startIcon={<UploadIcon />}
              onClick={() => setExtractDialogOpen(true)}
              variant="outlined"
              disabled={!graphRAGStatus.available}
              sx={{ borderColor: alpha('#6366F1', 0.5) }}
            >
              Extract from Text
            </Button>
            <Button
              startIcon={<RefreshIcon />}
              onClick={loadData}
              variant="outlined"
              disabled={!graphRAGStatus.available || loading}
            >
              Refresh
            </Button>
          </Box>
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* GraphRAG Unavailable Warning */}
      {!graphRAGStatus.available && graphRAGStatus.status !== 'checking' && !isFullscreen && (
        <Alert 
          severity="warning" 
          sx={{ mb: 2 }}
          action={
            <Button color="inherit" size="small" onClick={checkGraphRAGStatus}>
              Retry
            </Button>
          }
        >
          GraphRAG service is not available. Make sure the graphrag containers are running (docker compose up in the graphrag directory).
        </Alert>
      )}

      {/* Loading indicator while checking status */}
      {graphRAGStatus.status === 'checking' && !isFullscreen && (
        <Alert 
          severity="info" 
          sx={{ mb: 2 }}
          icon={<CircularProgress size={20} />}
        >
          Checking GraphRAG service status...
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
                searchTerm={searchTerm}
                showLabels={showLabels}
                showEdgeLabels={showEdgeLabels}
                particleFlow={particleFlow}
                onNodeClick={handleNodeClick}
                onNodeHover={(node) => setHoveredNodeId(node?.id || null)}
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

                  <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                    Relationships ({relationships.length})
                  </Typography>

                  <List dense>
                    {relationships.map(rel => {
                      const isSource = rel.source_id === selectedEntity.id;
                      const otherId = isSource ? rel.target_id : rel.source_id;
                      const otherEntity = entities.find(e => e.id === otherId);
                      return (
                        <ListItem 
                          key={rel.id}
                          button
                          onClick={() => {
                            // Select the other entity
                            if (otherEntity) {
                              setSelectedEntity(otherEntity);
                              setSelectedNodeId(otherEntity.id);
                            }
                          }}
                          sx={{ 
                            borderRadius: 1, 
                            bgcolor: alpha('#fff', 0.02),
                            mb: 0.5,
                            '&:hover': { bgcolor: alpha('#6366F1', 0.1) },
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
                                  {isSource ? 'to' : 'from'} <strong>{otherEntity?.name || otherId}</strong>
                                </Typography>
                              </Box>
                            }
                            secondary={rel.description}
                          />
                        </ListItem>
                      );
                    })}
                  </List>
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
            sx={{ background: 'linear-gradient(135deg, #10B981 0%, #06B6D4 100%)' }}
          >
            {loading ? <CircularProgress size={20} /> : 'Extract Entities'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default KnowledgeGraphPage;
