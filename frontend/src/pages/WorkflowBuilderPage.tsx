/**
 * WorkflowBuilderPage - Visual workflow editor with drag-and-drop canvas
 */
import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  IconButton,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Chip,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Tooltip,
  alpha,
  CircularProgress,
} from '@mui/material';
import {
  Add as AddIcon,
  Save as SaveIcon,
  FolderOpen as OpenIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Settings as SettingsIcon,
  MoreVert as MoreIcon,
  Delete as DeleteIcon,
  ContentCopy as DuplicateIcon,
  Download as ExportIcon,
  Upload as ImportIcon,
  Undo as UndoIcon,
  Redo as RedoIcon,
  ZoomIn as ZoomInIcon,
  AccountTree as WorkflowIcon,
} from '@mui/icons-material';
import { ReactFlowProvider } from 'reactflow';
import 'reactflow/dist/style.css';

import { WorkflowCanvas, NodePalette, PropertyPanel, ExecutionPanel } from '@/components/workflow';
import {
  Workflow,
  WorkflowNode,
  WorkflowConnection,
  WorkflowExecution,
  NodeExecution,
  getNodeTypeDefinition,
} from '@/types/workflow';
import { workflowApi } from '@/services/workflowApi';

// Initial empty workflow
const createEmptyWorkflow = (): Workflow => ({
  id: `wf-${Date.now()}`,
  name: 'New Workflow',
  description: '',
  nodes: [],
  connections: [],
  variables: {},
  settings: {},
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  version: 1,
  isActive: true,
});

const WorkflowBuilderPage: React.FC = () => {
  // Workflow state
  const [workflow, setWorkflow] = useState<Workflow | null>(null);
  const [savedWorkflows, setSavedWorkflows] = useState<Workflow[]>([]);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Selection state
  const [selectedNode, setSelectedNode] = useState<WorkflowNode | null>(null);

  // Execution state
  const [execution, setExecution] = useState<WorkflowExecution | null>(null);
  const [nodeExecutions, setNodeExecutions] = useState<Record<string, NodeExecution>>({});
  const [isRunning, setIsRunning] = useState(false);
  const [executionId, setExecutionId] = useState<string | null>(null);

  // Dialog state
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [openDialogOpen, setOpenDialogOpen] = useState(false);
  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false);

  // Menu state
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);

  // Load saved workflows from API
  useEffect(() => {
    loadWorkflows();
  }, []);

  const loadWorkflows = async () => {
    try {
      setLoading(true);
      const response = await workflowApi.listWorkflows({ limit: 100 });
      setSavedWorkflows(response.workflows);
    } catch (e) {
      console.error('Failed to load workflows:', e);
      // Fallback to localStorage if API fails
      const stored = localStorage.getItem('llama-nexus-workflows');
      if (stored) {
        try {
          setSavedWorkflows(JSON.parse(stored));
        } catch (parseError) {
          console.error('Failed to parse stored workflows:', parseError);
        }
      }
    } finally {
      setLoading(false);
    }
  };

  // Create new workflow
  const handleNewWorkflow = () => {
    if (hasUnsavedChanges) {
      if (!confirm('You have unsaved changes. Create new workflow anyway?')) {
        return;
      }
    }
    setWorkflow(createEmptyWorkflow());
    setSelectedNode(null);
    setHasUnsavedChanges(false);
    setError(null);
  };

  // Save workflow
  const handleSaveWorkflow = async () => {
    if (!workflow) return;

    try {
      setLoading(true);
      setError(null);

      let savedWorkflow: Workflow;
      
      // Check if workflow already exists in saved list
      const existingIndex = savedWorkflows.findIndex((w) => w.id === workflow.id);
      
      if (existingIndex >= 0) {
        // Update existing workflow
        savedWorkflow = await workflowApi.updateWorkflow(workflow.id, {
          name: workflow.name,
          description: workflow.description,
          nodes: workflow.nodes,
          connections: workflow.connections,
          variables: workflow.variables,
          settings: workflow.settings,
        });
      } else {
        // Create new workflow
        savedWorkflow = await workflowApi.createWorkflow({
          name: workflow.name,
          description: workflow.description,
          nodes: workflow.nodes,
          connections: workflow.connections,
          variables: workflow.variables,
          settings: workflow.settings,
        });
      }

      setWorkflow(savedWorkflow);
      setHasUnsavedChanges(false);
      setSaveDialogOpen(false);
      
      // Refresh workflow list
      await loadWorkflows();
    } catch (e: any) {
      console.error('Failed to save workflow:', e);
      setError(e.message || 'Failed to save workflow');
    } finally {
      setLoading(false);
    }
  };

  // Open workflow
  const handleOpenWorkflow = async (workflowToOpen: Workflow) => {
    if (hasUnsavedChanges) {
      if (!confirm('You have unsaved changes. Open another workflow anyway?')) {
        return;
      }
    }
    
    try {
      setLoading(true);
      // Fetch fresh copy from API
      const freshWorkflow = await workflowApi.getWorkflow(workflowToOpen.id);
      setWorkflow(freshWorkflow);
      setSelectedNode(null);
      setHasUnsavedChanges(false);
      setOpenDialogOpen(false);
      setError(null);
    } catch (e: any) {
      console.error('Failed to open workflow:', e);
      // Fallback to passed workflow
      setWorkflow(workflowToOpen);
      setSelectedNode(null);
      setHasUnsavedChanges(false);
      setOpenDialogOpen(false);
    } finally {
      setLoading(false);
    }
  };

  // Delete workflow
  const handleDeleteWorkflow = async (workflowId: string) => {
    try {
      setLoading(true);
      await workflowApi.deleteWorkflow(workflowId);
      
      if (workflow?.id === workflowId) {
        setWorkflow(null);
        setSelectedNode(null);
      }
      
      await loadWorkflows();
    } catch (e: any) {
      console.error('Failed to delete workflow:', e);
      setError(e.message || 'Failed to delete workflow');
    } finally {
      setLoading(false);
    }
  };

  // Handle nodes change
  const handleNodesChange = useCallback((nodes: WorkflowNode[]) => {
    setWorkflow((prev) => prev ? { ...prev, nodes } : prev);
    setHasUnsavedChanges(true);
  }, []);

  // Handle connections change
  const handleConnectionsChange = useCallback((connections: WorkflowConnection[]) => {
    setWorkflow((prev) => prev ? { ...prev, connections } : prev);
    setHasUnsavedChanges(true);
  }, []);

  // Handle node selection
  const handleNodeSelect = useCallback((node: WorkflowNode | null) => {
    setSelectedNode(node);
  }, []);

  // Handle node update from property panel
  const handleNodeUpdate = useCallback((updatedNode: WorkflowNode) => {
    setWorkflow((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        nodes: prev.nodes.map((n) => (n.id === updatedNode.id ? updatedNode : n)),
      };
    });
    setSelectedNode(updatedNode);
    setHasUnsavedChanges(true);
  }, []);

  // Handle node delete
  const handleNodeDelete = useCallback((nodeId: string) => {
    setWorkflow((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        nodes: prev.nodes.filter((n) => n.id !== nodeId),
        connections: prev.connections.filter(
          (c) => c.source !== nodeId && c.target !== nodeId
        ),
      };
    });
    if (selectedNode?.id === nodeId) {
      setSelectedNode(null);
    }
    setHasUnsavedChanges(true);
  }, [selectedNode]);

  // Handle node duplicate
  const handleNodeDuplicate = useCallback((nodeId: string) => {
    setWorkflow((prev) => {
      if (!prev) return prev;
      const nodeToDuplicate = prev.nodes.find((n) => n.id === nodeId);
      if (!nodeToDuplicate) return prev;

      const newNode: WorkflowNode = {
        ...nodeToDuplicate,
        id: `node-${Date.now()}`,
        position: {
          x: nodeToDuplicate.position.x + 50,
          y: nodeToDuplicate.position.y + 50,
        },
        data: {
          ...nodeToDuplicate.data,
          label: `${nodeToDuplicate.data.label} (copy)`,
        },
      };

      return {
        ...prev,
        nodes: [...prev.nodes, newNode],
      };
    });
    setHasUnsavedChanges(true);
  }, []);

  // Run workflow via API
  const handleRunWorkflow = async () => {
    if (!workflow || workflow.nodes.length === 0) return;

    // Save workflow first if there are unsaved changes
    if (hasUnsavedChanges) {
      await handleSaveWorkflow();
    }

    try {
      setIsRunning(true);
      setNodeExecutions({});
      setError(null);

      // Start execution via API
      const response = await workflowApi.executeWorkflow(workflow.id, {});
      setExecutionId(response.execution_id);

      // Poll for execution status
      await workflowApi.pollExecution(
        response.execution_id,
        (exec) => {
          // Update node executions from the API response
          if (exec.nodeExecutions) {
            const nodeExecs: Record<string, NodeExecution> = {};
            for (const [nodeId, nodeExec] of Object.entries(exec.nodeExecutions)) {
              nodeExecs[nodeId] = nodeExec as NodeExecution;
            }
            setNodeExecutions(nodeExecs);
          }
          setExecution(exec);
        },
        1000, // Poll every second
        300   // Max 5 minutes
      );

      setIsRunning(false);
    } catch (e: any) {
      console.error('Workflow execution failed:', e);
      setError(e.message || 'Workflow execution failed');
      setIsRunning(false);
    }
  };

  // Stop workflow execution
  const handleStopWorkflow = async () => {
    if (!executionId) {
      setIsRunning(false);
      return;
    }

    try {
      await workflowApi.cancelExecution(executionId);
      setIsRunning(false);
      setExecutionId(null);
    } catch (e: any) {
      console.error('Failed to stop workflow:', e);
      setError(e.message || 'Failed to stop workflow');
    }
  };

  // Export workflow
  const handleExportWorkflow = () => {
    if (!workflow) return;

    const dataStr = JSON.stringify(workflow, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${workflow.name.replace(/\s+/g, '-').toLowerCase()}.json`;
    link.click();
    URL.revokeObjectURL(url);
    setMenuAnchor(null);
  };

  // Import workflow
  const handleImportWorkflow = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;

      try {
        const text = await file.text();
        const imported = JSON.parse(text) as Workflow;
        imported.id = `wf-${Date.now()}`; // Generate new ID
        setWorkflow(imported);
        setHasUnsavedChanges(true);
      } catch (err) {
        console.error('Failed to import workflow:', err);
        alert('Failed to import workflow. Please check the file format.');
      }
    };
    input.click();
    setMenuAnchor(null);
  };

  return (
    <Box
      sx={{
        width: '100%',
        height: 'calc(100vh - 64px)',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          px: 2,
          py: 1.5,
          borderBottom: '1px solid',
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          gap: 2,
          bgcolor: 'background.paper',
        }}
      >
        {/* Left: Title and workflow info */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <WorkflowIcon sx={{ color: '#6366f1' }} />
          <Typography variant="h6" fontWeight={600}>
            Workflow Builder
          </Typography>
          {workflow && (
            <>
              <Chip
                label={workflow.name}
                size="small"
                sx={{
                  bgcolor: alpha('#6366f1', 0.1),
                  color: '#818cf8',
                  fontWeight: 500,
                }}
              />
              {hasUnsavedChanges && (
                <Chip
                  label="Unsaved"
                  size="small"
                  sx={{
                    bgcolor: alpha('#f59e0b', 0.1),
                    color: '#f59e0b',
                    fontSize: '0.7rem',
                  }}
                />
              )}
            </>
          )}
        </Box>

        {/* Center: Workflow actions */}
        <Box sx={{ flex: 1, display: 'flex', justifyContent: 'center', gap: 1 }}>
          {!workflow ? (
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={handleNewWorkflow}
            >
              New Workflow
            </Button>
          ) : (
            <>
              <Tooltip title="Save">
                <IconButton onClick={() => setSaveDialogOpen(true)} disabled={loading}>
                  {loading ? <CircularProgress size={20} /> : <SaveIcon />}
                </IconButton>
              </Tooltip>
              <Tooltip title="Open">
                <IconButton onClick={() => setOpenDialogOpen(true)}>
                  <OpenIcon />
                </IconButton>
              </Tooltip>
              <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />
              <Button
                variant="contained"
                startIcon={isRunning ? <CircularProgress size={16} color="inherit" /> : <PlayIcon />}
                onClick={isRunning ? handleStopWorkflow : handleRunWorkflow}
                disabled={workflow.nodes.length === 0}
                color={isRunning ? 'error' : 'success'}
                sx={{ minWidth: 100 }}
              >
                {isRunning ? 'Stop' : 'Run'}
              </Button>
            </>
          )}
        </Box>

        {/* Right: More actions */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {workflow && (
            <>
              <Tooltip title="Settings">
                <IconButton onClick={() => setSettingsDialogOpen(true)}>
                  <SettingsIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="More actions">
                <IconButton onClick={(e) => setMenuAnchor(e.currentTarget)}>
                  <MoreIcon />
                </IconButton>
              </Tooltip>
            </>
          )}
        </Box>
      </Box>

      {/* Error display */}
      {error && (
        <Alert 
          severity="error" 
          onClose={() => setError(null)}
          sx={{ mx: 2, mt: 1 }}
        >
          {error}
        </Alert>
      )}

      {/* Main content */}
      <Box sx={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Node Palette */}
        <NodePalette disabled={!workflow || loading} />

        {/* Canvas */}
        <Box sx={{ flex: 1, position: 'relative' }}>
          {workflow ? (
            <ReactFlowProvider>
              <WorkflowCanvas
                nodes={workflow.nodes}
                connections={workflow.connections}
                nodeExecutions={nodeExecutions}
                onNodesChange={handleNodesChange}
                onConnectionsChange={handleConnectionsChange}
                onNodeSelect={handleNodeSelect}
                selectedNodeId={selectedNode?.id}
              />
            </ReactFlowProvider>
          ) : (
            <Box
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 2,
                bgcolor: 'background.default',
              }}
            >
              <WorkflowIcon sx={{ fontSize: 80, color: 'text.disabled', opacity: 0.3 }} />
              <Typography variant="h6" color="text.secondary">
                Create or open a workflow to get started
              </Typography>
              <Box sx={{ display: 'flex', gap: 2 }}>
                <Button variant="contained" startIcon={<AddIcon />} onClick={handleNewWorkflow}>
                  New Workflow
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<OpenIcon />}
                  onClick={() => setOpenDialogOpen(true)}
                  disabled={savedWorkflows.length === 0}
                >
                  Open Workflow
                </Button>
              </Box>
            </Box>
          )}
        </Box>

        {/* Right Side Panel: Property or Execution */}
        {workflow && (
          selectedNode ? (
            <PropertyPanel
              node={selectedNode}
              onNodeUpdate={handleNodeUpdate}
              onNodeDelete={handleNodeDelete}
              onNodeDuplicate={handleNodeDuplicate}
              onClose={() => setSelectedNode(null)}
            />
          ) : (
            <ExecutionPanel
              workflowId={workflow.id}
              currentExecution={execution}
              nodeExecutions={nodeExecutions}
              isRunning={isRunning}
            />
          )
        )}
      </Box>

      {/* More actions menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={() => setMenuAnchor(null)}
      >
        <MenuItem onClick={handleNewWorkflow}>
          <ListItemIcon><AddIcon fontSize="small" /></ListItemIcon>
          <ListItemText>New Workflow</ListItemText>
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleExportWorkflow}>
          <ListItemIcon><ExportIcon fontSize="small" /></ListItemIcon>
          <ListItemText>Export Workflow</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleImportWorkflow}>
          <ListItemIcon><ImportIcon fontSize="small" /></ListItemIcon>
          <ListItemText>Import Workflow</ListItemText>
        </MenuItem>
        <Divider />
        <MenuItem
          onClick={() => {
            if (workflow && confirm('Delete this workflow?')) {
              handleDeleteWorkflow(workflow.id);
            }
            setMenuAnchor(null);
          }}
          sx={{ color: 'error.main' }}
        >
          <ListItemIcon><DeleteIcon fontSize="small" color="error" /></ListItemIcon>
          <ListItemText>Delete Workflow</ListItemText>
        </MenuItem>
      </Menu>

      {/* Save Dialog */}
      <Dialog open={saveDialogOpen} onClose={() => setSaveDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Save Workflow</DialogTitle>
        <DialogContent>
          <TextField
            label="Workflow Name"
            value={workflow?.name || ''}
            onChange={(e) => setWorkflow((prev) => prev ? { ...prev, name: e.target.value } : prev)}
            fullWidth
            margin="normal"
            autoFocus
          />
          <TextField
            label="Description (optional)"
            value={workflow?.description || ''}
            onChange={(e) => setWorkflow((prev) => prev ? { ...prev, description: e.target.value } : prev)}
            fullWidth
            margin="normal"
            multiline
            rows={3}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSaveDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={handleSaveWorkflow}>
            Save
          </Button>
        </DialogActions>
      </Dialog>

      {/* Open Dialog */}
      <Dialog open={openDialogOpen} onClose={() => setOpenDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Open Workflow</DialogTitle>
        <DialogContent>
          {savedWorkflows.length === 0 ? (
            <Alert severity="info" sx={{ mt: 1 }}>
              No saved workflows found. Create a new workflow to get started.
            </Alert>
          ) : (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mt: 1 }}>
              {savedWorkflows.map((wf) => (
                <Box
                  key={wf.id}
                  sx={{
                    p: 2,
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: 1,
                    cursor: 'pointer',
                    '&:hover': { bgcolor: 'action.hover' },
                    display: 'flex',
                    alignItems: 'center',
                    gap: 2,
                  }}
                  onClick={() => handleOpenWorkflow(wf)}
                >
                  <WorkflowIcon sx={{ color: '#6366f1' }} />
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="subtitle2">{wf.name}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {wf.nodes.length} nodes | Updated {new Date(wf.updatedAt).toLocaleDateString()}
                    </Typography>
                  </Box>
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (confirm(`Delete "${wf.name}"?`)) {
                        handleDeleteWorkflow(wf.id);
                      }
                    }}
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </Box>
              ))}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialogOpen(false)}>Cancel</Button>
        </DialogActions>
      </Dialog>

      {/* Settings Dialog */}
      <Dialog open={settingsDialogOpen} onClose={() => setSettingsDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Workflow Settings</DialogTitle>
        <DialogContent>
          <TextField
            label="Timeout (seconds)"
            type="number"
            value={workflow?.settings.timeout || 300}
            onChange={(e) => setWorkflow((prev) => 
              prev ? { ...prev, settings: { ...prev.settings, timeout: parseInt(e.target.value) } } : prev
            )}
            fullWidth
            margin="normal"
          />
          <TextField
            label="Max Retries"
            type="number"
            value={workflow?.settings.maxRetries || 3}
            onChange={(e) => setWorkflow((prev) => 
              prev ? { ...prev, settings: { ...prev.settings, maxRetries: parseInt(e.target.value) } } : prev
            )}
            fullWidth
            margin="normal"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default WorkflowBuilderPage;
