import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  IconButton,
  ToggleButton,
  ToggleButtonGroup,
  Slider,
  FormControl,
  FormControlLabel,
  Switch,
  Select,
  MenuItem,
  InputLabel,
  Tooltip,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Button,
  alpha,
  useTheme,
} from '@mui/material';
import {
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  CenterFocusStrong as CenterIcon,
  Fullscreen as FullscreenIcon,
  FullscreenExit as FullscreenExitIcon,
  Timeline as TimelineIcon,
  AccountTree as HierarchicalIcon,
  Hub as RadialIcon,
  GridOn as GridIcon,
  ScatterPlot as ForceIcon,
  Settings as SettingsIcon,
  ExpandMore as ExpandIcon,
  Palette as PaletteIcon,
  Speed as SpeedIcon,
  Visibility as VisibilityIcon,
  FilterList as FilterIcon,
  PhotoCamera as ScreenshotIcon,
  Download as ExportIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Replay as ResetIcon,
} from '@mui/icons-material';

interface GraphVisualizationControlsProps {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onCenter: () => void;
  onFullscreen: () => void;
  onLayoutChange: (layout: string) => void;
  onExport: (format: 'png' | 'svg' | 'json') => void;
  onSettingsChange: (settings: GraphSettings) => void;
  isFullscreen: boolean;
  currentLayout: string;
  settings: GraphSettings;
}

export interface GraphSettings {
  nodeSize: number;
  edgeWidth: number;
  labelSize: number;
  animationSpeed: number;
  showLabels: boolean;
  showEdgeLabels: boolean;
  showNodeStats: boolean;
  clusterByType: boolean;
  highlightConnections: boolean;
  physics: {
    enabled: boolean;
    forceStrength: number;
    linkDistance: number;
    chargeStrength: number;
  };
  filters: {
    minConfidence: number;
    entityTypes: string[];
    relationshipTypes: string[];
    maxNodes: number;
  };
  colors: {
    scheme: 'default' | 'category' | 'confidence' | 'custom';
    nodeColors: Record<string, string>;
    edgeColors: Record<string, string>;
  };
}

const defaultSettings: GraphSettings = {
  nodeSize: 30,
  edgeWidth: 2,
  labelSize: 12,
  animationSpeed: 1,
  showLabels: true,
  showEdgeLabels: false,
  showNodeStats: false,
  clusterByType: false,
  highlightConnections: true,
  physics: {
    enabled: true,
    forceStrength: -30,
    linkDistance: 100,
    chargeStrength: -300,
  },
  filters: {
    minConfidence: 0.5,
    entityTypes: [],
    relationshipTypes: [],
    maxNodes: 500,
  },
  colors: {
    scheme: 'category',
    nodeColors: {},
    edgeColors: {},
  },
};

export const GraphVisualizationControls: React.FC<GraphVisualizationControlsProps> = ({
  onZoomIn,
  onZoomOut,
  onCenter,
  onFullscreen,
  onLayoutChange,
  onExport,
  onSettingsChange,
  isFullscreen,
  currentLayout,
  settings = defaultSettings,
}) => {
  const theme = useTheme();
  const [physicsRunning, setPhysicsRunning] = useState(settings.physics.enabled);
  const [expandedPanel, setExpandedPanel] = useState<string | false>('display');

  const handleSettingChange = (key: string, value: any) => {
    const newSettings = { ...settings };
    const keys = key.split('.');
    let target: any = newSettings;
    
    for (let i = 0; i < keys.length - 1; i++) {
      target = target[keys[i]];
    }
    target[keys[keys.length - 1]] = value;
    
    onSettingsChange(newSettings);
  };

  const togglePhysics = () => {
    const newState = !physicsRunning;
    setPhysicsRunning(newState);
    handleSettingChange('physics.enabled', newState);
  };

  const layouts = [
    { id: 'force', label: 'Force-Directed', icon: <ForceIcon /> },
    { id: 'hierarchical', label: 'Hierarchical', icon: <HierarchicalIcon /> },
    { id: 'radial', label: 'Radial', icon: <RadialIcon /> },
    { id: 'grid', label: 'Grid', icon: <GridIcon /> },
    { id: 'timeline', label: 'Timeline', icon: <TimelineIcon /> },
  ];

  const colorSchemes = [
    { id: 'default', label: 'Default' },
    { id: 'category', label: 'By Category' },
    { id: 'confidence', label: 'By Confidence' },
    { id: 'custom', label: 'Custom' },
  ];

  return (
    <Paper
      sx={{
        position: 'absolute',
        top: 16,
        right: 16,
        width: 320,
        maxHeight: 'calc(100vh - 32px)',
        overflow: 'auto',
        bgcolor: alpha(theme.palette.background.paper, 0.95),
        backdropFilter: 'blur(10px)',
        zIndex: 1000,
      }}
    >
      {/* Quick Controls */}
      <Box sx={{ p: 2, borderBottom: `1px solid ${theme.palette.divider}` }}>
        <Typography variant="subtitle2" gutterBottom>
          Quick Controls
        </Typography>
        <Box sx={{ display: 'flex', gap: 0.5, mb: 1 }}>
          <Tooltip title="Zoom In">
            <IconButton size="small" onClick={onZoomIn}>
              <ZoomInIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Zoom Out">
            <IconButton size="small" onClick={onZoomOut}>
              <ZoomOutIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Center View">
            <IconButton size="small" onClick={onCenter}>
              <CenterIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}>
            <IconButton size="small" onClick={onFullscreen}>
              {isFullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
            </IconButton>
          </Tooltip>
          <Divider orientation="vertical" flexItem sx={{ mx: 0.5 }} />
          <Tooltip title={physicsRunning ? "Pause Physics" : "Start Physics"}>
            <IconButton size="small" onClick={togglePhysics}>
              {physicsRunning ? <PauseIcon /> : <PlayIcon />}
            </IconButton>
          </Tooltip>
          <Tooltip title="Reset Layout">
            <IconButton size="small" onClick={() => onLayoutChange(currentLayout)}>
              <ResetIcon />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Layout Selection */}
        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
          Layout
        </Typography>
        <ToggleButtonGroup
          value={currentLayout}
          exclusive
          onChange={(e, value) => value && onLayoutChange(value)}
          size="small"
          fullWidth
          sx={{ flexWrap: 'wrap' }}
        >
          {layouts.map(layout => (
            <ToggleButton
              key={layout.id}
              value={layout.id}
              sx={{ flex: '1 1 30%' }}
            >
              <Tooltip title={layout.label}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  {layout.icon}
                </Box>
              </Tooltip>
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
      </Box>

      {/* Detailed Settings */}
      <Accordion
        expanded={expandedPanel === 'display'}
        onChange={(e, isExpanded) => setExpandedPanel(isExpanded ? 'display' : false)}
      >
        <AccordionSummary expandIcon={<ExpandIcon />}>
          <VisibilityIcon sx={{ mr: 1 }} />
          <Typography>Display Settings</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ px: 1 }}>
            <Typography variant="caption" color="text.secondary">
              Node Size
            </Typography>
            <Slider
              value={settings.nodeSize}
              onChange={(e, value) => handleSettingChange('nodeSize', value)}
              min={10}
              max={100}
              valueLabelDisplay="auto"
              size="small"
            />

            <Typography variant="caption" color="text.secondary">
              Edge Width
            </Typography>
            <Slider
              value={settings.edgeWidth}
              onChange={(e, value) => handleSettingChange('edgeWidth', value)}
              min={1}
              max={10}
              valueLabelDisplay="auto"
              size="small"
            />

            <Typography variant="caption" color="text.secondary">
              Label Size
            </Typography>
            <Slider
              value={settings.labelSize}
              onChange={(e, value) => handleSettingChange('labelSize', value)}
              min={8}
              max={24}
              valueLabelDisplay="auto"
              size="small"
            />

            <FormControlLabel
              control={
                <Switch
                  checked={settings.showLabels}
                  onChange={(e) => handleSettingChange('showLabels', e.target.checked)}
                  size="small"
                />
              }
              label="Show Node Labels"
            />

            <FormControlLabel
              control={
                <Switch
                  checked={settings.showEdgeLabels}
                  onChange={(e) => handleSettingChange('showEdgeLabels', e.target.checked)}
                  size="small"
                />
              }
              label="Show Edge Labels"
            />

            <FormControlLabel
              control={
                <Switch
                  checked={settings.showNodeStats}
                  onChange={(e) => handleSettingChange('showNodeStats', e.target.checked)}
                  size="small"
                />
              }
              label="Show Node Statistics"
            />

            <FormControlLabel
              control={
                <Switch
                  checked={settings.highlightConnections}
                  onChange={(e) => handleSettingChange('highlightConnections', e.target.checked)}
                  size="small"
                />
              }
              label="Highlight on Hover"
            />
          </Box>
        </AccordionDetails>
      </Accordion>

      <Accordion
        expanded={expandedPanel === 'physics'}
        onChange={(e, isExpanded) => setExpandedPanel(isExpanded ? 'physics' : false)}
      >
        <AccordionSummary expandIcon={<ExpandIcon />}>
          <SpeedIcon sx={{ mr: 1 }} />
          <Typography>Physics & Animation</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ px: 1 }}>
            <Typography variant="caption" color="text.secondary">
              Animation Speed
            </Typography>
            <Slider
              value={settings.animationSpeed}
              onChange={(e, value) => handleSettingChange('animationSpeed', value)}
              min={0.1}
              max={3}
              step={0.1}
              valueLabelDisplay="auto"
              size="small"
            />

            <Typography variant="caption" color="text.secondary">
              Force Strength
            </Typography>
            <Slider
              value={settings.physics.forceStrength}
              onChange={(e, value) => handleSettingChange('physics.forceStrength', value)}
              min={-100}
              max={0}
              valueLabelDisplay="auto"
              size="small"
              disabled={!physicsRunning}
            />

            <Typography variant="caption" color="text.secondary">
              Link Distance
            </Typography>
            <Slider
              value={settings.physics.linkDistance}
              onChange={(e, value) => handleSettingChange('physics.linkDistance', value)}
              min={50}
              max={300}
              valueLabelDisplay="auto"
              size="small"
              disabled={!physicsRunning}
            />

            <Typography variant="caption" color="text.secondary">
              Charge Strength
            </Typography>
            <Slider
              value={settings.physics.chargeStrength}
              onChange={(e, value) => handleSettingChange('physics.chargeStrength', value)}
              min={-1000}
              max={0}
              valueLabelDisplay="auto"
              size="small"
              disabled={!physicsRunning}
            />
          </Box>
        </AccordionDetails>
      </Accordion>

      <Accordion
        expanded={expandedPanel === 'filters'}
        onChange={(e, isExpanded) => setExpandedPanel(isExpanded ? 'filters' : false)}
      >
        <AccordionSummary expandIcon={<ExpandIcon />}>
          <FilterIcon sx={{ mr: 1 }} />
          <Typography>Filters</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ px: 1 }}>
            <Typography variant="caption" color="text.secondary">
              Minimum Confidence
            </Typography>
            <Slider
              value={settings.filters.minConfidence}
              onChange={(e, value) => handleSettingChange('filters.minConfidence', value)}
              min={0}
              max={1}
              step={0.1}
              valueLabelDisplay="auto"
              valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
              size="small"
            />

            <Typography variant="caption" color="text.secondary">
              Max Nodes to Display
            </Typography>
            <Slider
              value={settings.filters.maxNodes}
              onChange={(e, value) => handleSettingChange('filters.maxNodes', value)}
              min={50}
              max={1000}
              step={50}
              valueLabelDisplay="auto"
              size="small"
            />

            <FormControlLabel
              control={
                <Switch
                  checked={settings.clusterByType}
                  onChange={(e) => handleSettingChange('clusterByType', e.target.checked)}
                  size="small"
                />
              }
              label="Cluster by Type"
            />
          </Box>
        </AccordionDetails>
      </Accordion>

      <Accordion
        expanded={expandedPanel === 'colors'}
        onChange={(e, isExpanded) => setExpandedPanel(isExpanded ? 'colors' : false)}
      >
        <AccordionSummary expandIcon={<ExpandIcon />}>
          <PaletteIcon sx={{ mr: 1 }} />
          <Typography>Colors</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ px: 1 }}>
            <FormControl fullWidth size="small">
              <InputLabel>Color Scheme</InputLabel>
              <Select
                value={settings.colors.scheme}
                onChange={(e) => handleSettingChange('colors.scheme', e.target.value)}
                label="Color Scheme"
              >
                {colorSchemes.map(scheme => (
                  <MenuItem key={scheme.id} value={scheme.id}>
                    {scheme.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Export Options */}
      <Box sx={{ p: 2, borderTop: `1px solid ${theme.palette.divider}` }}>
        <Typography variant="subtitle2" gutterBottom>
          Export
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            size="small"
            startIcon={<ScreenshotIcon />}
            onClick={() => onExport('png')}
            variant="outlined"
          >
            PNG
          </Button>
          <Button
            size="small"
            startIcon={<ExportIcon />}
            onClick={() => onExport('svg')}
            variant="outlined"
          >
            SVG
          </Button>
          <Button
            size="small"
            startIcon={<ExportIcon />}
            onClick={() => onExport('json')}
            variant="outlined"
          >
            JSON
          </Button>
        </Box>
      </Box>
    </Paper>
  );
};
