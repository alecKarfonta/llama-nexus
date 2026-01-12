/**
 * ExecutionPanel - Shows workflow execution history and status
 */
import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  IconButton,
  Collapse,
  LinearProgress,
  Divider,
  Button,
  alpha,
  Tooltip,
} from '@mui/material';
import {
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  HourglassEmpty as PendingIcon,
  PlayArrow as RunningIcon,
  Cancel as CancelledIcon,
  ExpandMore as ExpandIcon,
  ExpandLess as CollapseIcon,
  Refresh as RefreshIcon,
  History as HistoryIcon,
  Timeline as TimelineIcon,
} from '@mui/icons-material';
import { WorkflowExecution, NodeExecution, ExecutionStatus } from '@/types/workflow';
import { workflowApi } from '@/services/workflowApi';

interface ExecutionPanelProps {
  workflowId: string;
  currentExecution?: WorkflowExecution | null;
  nodeExecutions?: Record<string, NodeExecution>;
  isRunning?: boolean;
}

// Status colors and icons
const statusConfig: Record<ExecutionStatus, { color: string; icon: React.ReactNode; label: string }> = {
  idle: { color: '#64748b', icon: <PendingIcon fontSize="small" />, label: 'Idle' },
  pending: { color: '#f59e0b', icon: <PendingIcon fontSize="small" />, label: 'Pending' },
  running: { color: '#6366f1', icon: <RunningIcon fontSize="small" />, label: 'Running' },
  success: { color: '#10b981', icon: <SuccessIcon fontSize="small" />, label: 'Success' },
  error: { color: '#ef4444', icon: <ErrorIcon fontSize="small" />, label: 'Error' },
  cancelled: { color: '#64748b', icon: <CancelledIcon fontSize="small" />, label: 'Cancelled' },
  paused: { color: '#f59e0b', icon: <PendingIcon fontSize="small" />, label: 'Paused' },
};

export const ExecutionPanel: React.FC<ExecutionPanelProps> = ({
  workflowId,
  currentExecution,
  nodeExecutions,
  isRunning,
}) => {
  const [tabValue, setTabValue] = useState(0);
  const [executions, setExecutions] = useState<WorkflowExecution[]>([]);
  const [loading, setLoading] = useState(false);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());

  // Load execution history
  const loadExecutions = async () => {
    if (!workflowId) return;
    
    try {
      setLoading(true);
      const response = await workflowApi.listExecutions(workflowId, { limit: 20 });
      setExecutions(response.executions);
    } catch (e) {
      console.error('Failed to load executions:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadExecutions();
  }, [workflowId]);

  const toggleNodeExpand = (nodeId: string) => {
    setExpandedNodes((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  };

  const formatDuration = (ms: number | undefined) => {
    if (!ms) return '-';
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  const formatDate = (dateStr: string | undefined) => {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    return date.toLocaleString();
  };

  return (
    <Box
      sx={{
        width: 320,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        borderLeft: '1px solid',
        borderColor: 'divider',
        bgcolor: 'background.paper',
      }}
    >
      {/* Header */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs
          value={tabValue}
          onChange={(_, v) => setTabValue(v)}
          variant="fullWidth"
        >
          <Tab
            icon={<TimelineIcon fontSize="small" />}
            label="Current"
            sx={{ minHeight: 48 }}
          />
          <Tab
            icon={<HistoryIcon fontSize="small" />}
            label="History"
            sx={{ minHeight: 48 }}
          />
        </Tabs>
      </Box>

      {/* Current Execution Tab */}
      {tabValue === 0 && (
        <Box sx={{ flex: 1, overflow: 'auto' }}>
          {currentExecution || (nodeExecutions && Object.keys(nodeExecutions).length > 0) ? (
            <>
              {/* Execution status */}
              <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  {isRunning && (
                    <LinearProgress
                      sx={{ flex: 1, height: 4, borderRadius: 2 }}
                    />
                  )}
                  {!isRunning && currentExecution && (
                    <Chip
                      icon={statusConfig[currentExecution.status]?.icon}
                      label={statusConfig[currentExecution.status]?.label}
                      size="small"
                      sx={{
                        bgcolor: alpha(statusConfig[currentExecution.status]?.color || '#64748b', 0.15),
                        color: statusConfig[currentExecution.status]?.color,
                      }}
                    />
                  )}
                </Box>
                {currentExecution && (
                  <Typography variant="caption" color="text.secondary">
                    Started: {formatDate(currentExecution.startedAt)}
                  </Typography>
                )}
              </Box>

              {/* Node execution list */}
              <List dense>
                {nodeExecutions && Object.entries(nodeExecutions).map(([nodeId, nodeExec]) => {
                  const status = statusConfig[nodeExec.status] || statusConfig.idle;
                  const isExpanded = expandedNodes.has(nodeId);

                  return (
                    <React.Fragment key={nodeId}>
                      <ListItem
                        sx={{
                          cursor: 'pointer',
                          '&:hover': { bgcolor: 'action.hover' },
                        }}
                        onClick={() => toggleNodeExpand(nodeId)}
                      >
                        <ListItemIcon sx={{ minWidth: 32 }}>
                          <Box sx={{ color: status.color }}>{status.icon}</Box>
                        </ListItemIcon>
                        <ListItemText
                          primary={nodeExec.nodeName}
                          secondary={formatDuration(nodeExec.duration)}
                          primaryTypographyProps={{ variant: 'body2', fontWeight: 500 }}
                          secondaryTypographyProps={{ variant: 'caption' }}
                        />
                        <IconButton size="small">
                          {isExpanded ? <CollapseIcon fontSize="small" /> : <ExpandIcon fontSize="small" />}
                        </IconButton>
                      </ListItem>
                      
                      <Collapse in={isExpanded}>
                        <Box sx={{ px: 2, pb: 2, bgcolor: 'action.hover' }}>
                          {/* Inputs */}
                          {nodeExec.inputs && Object.keys(nodeExec.inputs).length > 0 && (
                            <Box sx={{ mb: 1 }}>
                              <Typography variant="caption" color="text.secondary" fontWeight={600}>
                                Inputs
                              </Typography>
                              <Paper
                                variant="outlined"
                                sx={{
                                  mt: 0.5,
                                  p: 1,
                                  fontFamily: 'monospace',
                                  fontSize: '0.7rem',
                                  maxHeight: 100,
                                  overflow: 'auto',
                                }}
                              >
                                <pre style={{ margin: 0 }}>
                                  {JSON.stringify(nodeExec.inputs, null, 2)}
                                </pre>
                              </Paper>
                            </Box>
                          )}

                          {/* Outputs */}
                          {nodeExec.outputs && Object.keys(nodeExec.outputs).length > 0 && (
                            <Box sx={{ mb: 1 }}>
                              <Typography variant="caption" color="text.secondary" fontWeight={600}>
                                Outputs
                              </Typography>
                              <Paper
                                variant="outlined"
                                sx={{
                                  mt: 0.5,
                                  p: 1,
                                  fontFamily: 'monospace',
                                  fontSize: '0.7rem',
                                  maxHeight: 100,
                                  overflow: 'auto',
                                }}
                              >
                                <pre style={{ margin: 0 }}>
                                  {JSON.stringify(nodeExec.outputs, null, 2)}
                                </pre>
                              </Paper>
                            </Box>
                          )}

                          {/* Error */}
                          {nodeExec.error && (
                            <Box sx={{ mb: 1 }}>
                              <Typography variant="caption" color="error" fontWeight={600}>
                                Error
                              </Typography>
                              <Paper
                                variant="outlined"
                                sx={{
                                  mt: 0.5,
                                  p: 1,
                                  fontSize: '0.7rem',
                                  color: 'error.main',
                                  bgcolor: alpha('#ef4444', 0.05),
                                }}
                              >
                                {nodeExec.error}
                              </Paper>
                            </Box>
                          )}

                          {/* Logs */}
                          {nodeExec.logs && nodeExec.logs.length > 0 && (
                            <Box>
                              <Typography variant="caption" color="text.secondary" fontWeight={600}>
                                Logs
                              </Typography>
                              <Paper
                                variant="outlined"
                                sx={{
                                  mt: 0.5,
                                  p: 1,
                                  fontFamily: 'monospace',
                                  fontSize: '0.65rem',
                                  maxHeight: 80,
                                  overflow: 'auto',
                                }}
                              >
                                {nodeExec.logs.map((log, i) => (
                                  <div key={i}>{log}</div>
                                ))}
                              </Paper>
                            </Box>
                          )}
                        </Box>
                      </Collapse>
                    </React.Fragment>
                  );
                })}
              </List>

              {/* Final output */}
              {currentExecution?.finalOutput && (
                <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
                  <Typography variant="caption" color="text.secondary" fontWeight={600}>
                    Final Output
                  </Typography>
                  <Paper
                    variant="outlined"
                    sx={{
                      mt: 0.5,
                      p: 1,
                      fontFamily: 'monospace',
                      fontSize: '0.75rem',
                      maxHeight: 150,
                      overflow: 'auto',
                      bgcolor: alpha('#10b981', 0.05),
                    }}
                  >
                    <pre style={{ margin: 0 }}>
                      {JSON.stringify(currentExecution.finalOutput, null, 2)}
                    </pre>
                  </Paper>
                </Box>
              )}
            </>
          ) : (
            <Box
              sx={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                p: 3,
              }}
            >
              <TimelineIcon sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
              <Typography variant="body2" color="text.secondary" textAlign="center">
                Run the workflow to see execution details
              </Typography>
            </Box>
          )}
        </Box>
      )}

      {/* History Tab */}
      {tabValue === 1 && (
        <Box sx={{ flex: 1, overflow: 'auto' }}>
          <Box sx={{ p: 1, display: 'flex', justifyContent: 'flex-end' }}>
            <Tooltip title="Refresh">
              <IconButton size="small" onClick={loadExecutions} disabled={loading}>
                <RefreshIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>

          {loading && <LinearProgress />}

          {executions.length === 0 && !loading ? (
            <Box
              sx={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                p: 3,
              }}
            >
              <HistoryIcon sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
              <Typography variant="body2" color="text.secondary" textAlign="center">
                No execution history yet
              </Typography>
            </Box>
          ) : (
            <List dense>
              {executions.map((exec) => {
                const status = statusConfig[exec.status] || statusConfig.idle;
                const nodeCount = Object.keys(exec.nodeExecutions || {}).length;
                
                return (
                  <ListItem
                    key={exec.id}
                    sx={{
                      borderBottom: 1,
                      borderColor: 'divider',
                    }}
                  >
                    <ListItemIcon sx={{ minWidth: 32 }}>
                      <Box sx={{ color: status.color }}>{status.icon}</Box>
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="body2" fontWeight={500}>
                            {exec.id.slice(0, 12)}
                          </Typography>
                          <Chip
                            label={status.label}
                            size="small"
                            sx={{
                              height: 18,
                              fontSize: '0.65rem',
                              bgcolor: alpha(status.color, 0.15),
                              color: status.color,
                            }}
                          />
                        </Box>
                      }
                      secondary={
                        <Typography variant="caption" color="text.secondary">
                          {formatDate(exec.startedAt)} | {nodeCount} nodes
                        </Typography>
                      }
                    />
                  </ListItem>
                );
              })}
            </List>
          )}
        </Box>
      )}
    </Box>
  );
};

export default ExecutionPanel;











