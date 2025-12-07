import React, { useState, useCallback } from 'react'
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  IconButton,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Tooltip,
  alpha,
  Divider,
  Grid,
  Paper,
  CircularProgress,
} from '@mui/material'
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  PlayArrow as PlayIcon,
  Save as SaveIcon,
  ContentCopy as CopyIcon,
  Edit as EditIcon,
  ArrowForward as ArrowIcon,
  AccountTree as WorkflowIcon,
  Psychology as LlmIcon,
  Code as CodeIcon,
  FilterAlt as FilterIcon,
  Loop as LoopIcon,
  CallSplit as BranchIcon,
  Input as InputIcon,
  Output as OutputIcon,
} from '@mui/icons-material'

// Workflow node types
type NodeType = 'input' | 'llm' | 'code' | 'condition' | 'output' | 'loop'

interface WorkflowNode {
  id: string
  type: NodeType
  name: string
  config: Record<string, any>
  position: { x: number; y: number }
  connections: string[] // IDs of connected nodes
}

interface Workflow {
  id: string
  name: string
  description: string
  nodes: WorkflowNode[]
  createdAt: string
  updatedAt: string
}

// Node type configurations
const nodeTypeConfig: Record<NodeType, { label: string; color: string; icon: React.ReactNode; fields: string[] }> = {
  input: {
    label: 'Input',
    color: '#10b981',
    icon: <InputIcon />,
    fields: ['variableName', 'defaultValue', 'description'],
  },
  llm: {
    label: 'LLM Call',
    color: '#6366f1',
    icon: <LlmIcon />,
    fields: ['prompt', 'systemPrompt', 'temperature', 'maxTokens', 'model'],
  },
  code: {
    label: 'Code',
    color: '#f59e0b',
    icon: <CodeIcon />,
    fields: ['code', 'language'],
  },
  condition: {
    label: 'Condition',
    color: '#8b5cf6',
    icon: <BranchIcon />,
    fields: ['condition', 'trueBranch', 'falseBranch'],
  },
  output: {
    label: 'Output',
    color: '#ef4444',
    icon: <OutputIcon />,
    fields: ['outputVariable', 'format'],
  },
  loop: {
    label: 'Loop',
    color: '#06b6d4',
    icon: <LoopIcon />,
    fields: ['iterations', 'breakCondition'],
  },
}

// Node component
interface NodeCardProps {
  node: WorkflowNode
  onEdit: (node: WorkflowNode) => void
  onDelete: (id: string) => void
  onConnect: (fromId: string) => void
  isConnecting: boolean
  connectingFrom: string | null
}

const NodeCard: React.FC<NodeCardProps> = ({ node, onEdit, onDelete, onConnect, isConnecting, connectingFrom }) => {
  const config = nodeTypeConfig[node.type]

  return (
    <Card
      sx={{
        width: 220,
        background: 'linear-gradient(145deg, rgba(30, 30, 63, 0.8) 0%, rgba(26, 26, 46, 0.9) 100%)',
        backdropFilter: 'blur(12px)',
        border: `2px solid ${connectingFrom === node.id ? config.color : 'rgba(255, 255, 255, 0.06)'}`,
        borderRadius: 2,
        transition: 'all 0.2s ease-in-out',
        cursor: 'pointer',
        '&:hover': {
          borderColor: alpha(config.color, 0.5),
          transform: 'translateY(-2px)',
          boxShadow: `0 8px 24px ${alpha(config.color, 0.2)}`,
        },
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 1.5,
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
          background: alpha(config.color, 0.1),
        }}
      >
        <Box sx={{ color: config.color, display: 'flex' }}>{config.icon}</Box>
        <Typography variant="subtitle2" sx={{ flex: 1, fontWeight: 600, color: 'text.primary' }}>
          {node.name}
        </Typography>
        <Chip
          label={config.label}
          size="small"
          sx={{
            height: 20,
            fontSize: '0.625rem',
            bgcolor: alpha(config.color, 0.15),
            color: config.color,
            fontWeight: 600,
          }}
        />
      </Box>

      {/* Content Preview */}
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
            fontSize: '0.6875rem',
          }}
        >
          {node.type === 'llm' && node.config.prompt
            ? node.config.prompt.substring(0, 80) + '...'
            : `${config.label} node`}
        </Typography>

        {/* Actions */}
        <Box sx={{ display: 'flex', gap: 0.5, mt: 1, justifyContent: 'flex-end' }}>
          <Tooltip title="Connect">
            <IconButton
              size="small"
              onClick={(e) => {
                e.stopPropagation()
                onConnect(node.id)
              }}
              sx={{
                bgcolor: isConnecting && connectingFrom !== node.id ? alpha('#10b981', 0.2) : 'transparent',
                '&:hover': { bgcolor: alpha(config.color, 0.2) },
              }}
            >
              <ArrowIcon sx={{ fontSize: 16 }} />
            </IconButton>
          </Tooltip>
          <Tooltip title="Edit">
            <IconButton
              size="small"
              onClick={(e) => {
                e.stopPropagation()
                onEdit(node)
              }}
            >
              <EditIcon sx={{ fontSize: 16 }} />
            </IconButton>
          </Tooltip>
          <Tooltip title="Delete">
            <IconButton
              size="small"
              onClick={(e) => {
                e.stopPropagation()
                onDelete(node.id)
              }}
              sx={{ '&:hover': { color: '#ef4444' } }}
            >
              <DeleteIcon sx={{ fontSize: 16 }} />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Connections indicator */}
        {node.connections.length > 0 && (
          <Box sx={{ mt: 1, pt: 1, borderTop: '1px solid rgba(255, 255, 255, 0.06)' }}>
            <Typography variant="caption" color="text.secondary">
              Connects to: {node.connections.length} node(s)
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  )
}

// Node Editor Dialog
interface NodeEditorProps {
  open: boolean
  node: WorkflowNode | null
  onClose: () => void
  onSave: (node: WorkflowNode) => void
}

const NodeEditor: React.FC<NodeEditorProps> = ({ open, node, onClose, onSave }) => {
  const [editedNode, setEditedNode] = useState<WorkflowNode | null>(null)

  React.useEffect(() => {
    setEditedNode(node)
  }, [node])

  if (!editedNode) return null

  const config = nodeTypeConfig[editedNode.type]

  const handleConfigChange = (field: string, value: any) => {
    setEditedNode({
      ...editedNode,
      config: { ...editedNode.config, [field]: value },
    })
  }

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
        <Box sx={{ color: config.color, display: 'flex' }}>{config.icon}</Box>
        Edit {config.label} Node
      </DialogTitle>
      <DialogContent>
        <Box sx={{ pt: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
          <TextField
            label="Node Name"
            value={editedNode.name}
            onChange={(e) => setEditedNode({ ...editedNode, name: e.target.value })}
            fullWidth
            size="small"
          />

          {/* Type-specific fields */}
          {editedNode.type === 'llm' && (
            <>
              <TextField
                label="System Prompt"
                value={editedNode.config.systemPrompt || ''}
                onChange={(e) => handleConfigChange('systemPrompt', e.target.value)}
                fullWidth
                multiline
                rows={2}
                size="small"
              />
              <TextField
                label="Prompt Template"
                value={editedNode.config.prompt || ''}
                onChange={(e) => handleConfigChange('prompt', e.target.value)}
                fullWidth
                multiline
                rows={4}
                size="small"
                placeholder="Use {{variable}} for template variables"
              />
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    label="Temperature"
                    type="number"
                    value={editedNode.config.temperature || 0.7}
                    onChange={(e) => handleConfigChange('temperature', parseFloat(e.target.value))}
                    fullWidth
                    size="small"
                    inputProps={{ min: 0, max: 2, step: 0.1 }}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Max Tokens"
                    type="number"
                    value={editedNode.config.maxTokens || 1024}
                    onChange={(e) => handleConfigChange('maxTokens', parseInt(e.target.value))}
                    fullWidth
                    size="small"
                  />
                </Grid>
              </Grid>
            </>
          )}

          {editedNode.type === 'input' && (
            <>
              <TextField
                label="Variable Name"
                value={editedNode.config.variableName || ''}
                onChange={(e) => handleConfigChange('variableName', e.target.value)}
                fullWidth
                size="small"
              />
              <TextField
                label="Default Value"
                value={editedNode.config.defaultValue || ''}
                onChange={(e) => handleConfigChange('defaultValue', e.target.value)}
                fullWidth
                size="small"
              />
              <TextField
                label="Description"
                value={editedNode.config.description || ''}
                onChange={(e) => handleConfigChange('description', e.target.value)}
                fullWidth
                multiline
                rows={2}
                size="small"
              />
            </>
          )}

          {editedNode.type === 'code' && (
            <>
              <FormControl fullWidth size="small">
                <InputLabel>Language</InputLabel>
                <Select
                  value={editedNode.config.language || 'javascript'}
                  label="Language"
                  onChange={(e) => handleConfigChange('language', e.target.value)}
                >
                  <MenuItem value="javascript">JavaScript</MenuItem>
                  <MenuItem value="python">Python</MenuItem>
                </Select>
              </FormControl>
              <TextField
                label="Code"
                value={editedNode.config.code || ''}
                onChange={(e) => handleConfigChange('code', e.target.value)}
                fullWidth
                multiline
                rows={6}
                size="small"
                sx={{ fontFamily: 'monospace' }}
              />
            </>
          )}

          {editedNode.type === 'condition' && (
            <>
              <TextField
                label="Condition Expression"
                value={editedNode.config.condition || ''}
                onChange={(e) => handleConfigChange('condition', e.target.value)}
                fullWidth
                size="small"
                placeholder="e.g., {{response}}.includes('yes')"
              />
              <Alert severity="info" sx={{ fontSize: '0.75rem' }}>
                Use JavaScript expressions. Variables are available as {'{{'} variableName {'}}'}
              </Alert>
            </>
          )}

          {editedNode.type === 'output' && (
            <>
              <TextField
                label="Output Variable"
                value={editedNode.config.outputVariable || ''}
                onChange={(e) => handleConfigChange('outputVariable', e.target.value)}
                fullWidth
                size="small"
              />
              <FormControl fullWidth size="small">
                <InputLabel>Format</InputLabel>
                <Select
                  value={editedNode.config.format || 'text'}
                  label="Format"
                  onChange={(e) => handleConfigChange('format', e.target.value)}
                >
                  <MenuItem value="text">Plain Text</MenuItem>
                  <MenuItem value="json">JSON</MenuItem>
                  <MenuItem value="markdown">Markdown</MenuItem>
                </Select>
              </FormControl>
            </>
          )}

          {editedNode.type === 'loop' && (
            <>
              <TextField
                label="Max Iterations"
                type="number"
                value={editedNode.config.iterations || 5}
                onChange={(e) => handleConfigChange('iterations', parseInt(e.target.value))}
                fullWidth
                size="small"
              />
              <TextField
                label="Break Condition"
                value={editedNode.config.breakCondition || ''}
                onChange={(e) => handleConfigChange('breakCondition', e.target.value)}
                fullWidth
                size="small"
                placeholder="e.g., {{result}} === 'done'"
              />
            </>
          )}
        </Box>
      </DialogContent>
      <DialogActions sx={{ p: 2 }}>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" onClick={() => onSave(editedNode)}>
          Save
        </Button>
      </DialogActions>
    </Dialog>
  )
}

// Main Workflow Builder Page
const WorkflowBuilderPage: React.FC = () => {
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [currentWorkflow, setCurrentWorkflow] = useState<Workflow | null>(null)
  const [editingNode, setEditingNode] = useState<WorkflowNode | null>(null)
  const [isConnecting, setIsConnecting] = useState(false)
  const [connectingFrom, setConnectingFrom] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [runResult, setRunResult] = useState<string | null>(null)

  // Create new workflow
  const createWorkflow = () => {
    const newWorkflow: Workflow = {
      id: `wf-${Date.now()}`,
      name: 'New Workflow',
      description: '',
      nodes: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    }
    setCurrentWorkflow(newWorkflow)
    setWorkflows([...workflows, newWorkflow])
  }

  // Add node
  const addNode = (type: NodeType) => {
    if (!currentWorkflow) return

    const newNode: WorkflowNode = {
      id: `node-${Date.now()}`,
      type,
      name: `${nodeTypeConfig[type].label} ${currentWorkflow.nodes.filter((n) => n.type === type).length + 1}`,
      config: {},
      position: { x: 100, y: 100 + currentWorkflow.nodes.length * 150 },
      connections: [],
    }

    setCurrentWorkflow({
      ...currentWorkflow,
      nodes: [...currentWorkflow.nodes, newNode],
      updatedAt: new Date().toISOString(),
    })
  }

  // Delete node
  const deleteNode = (id: string) => {
    if (!currentWorkflow) return

    setCurrentWorkflow({
      ...currentWorkflow,
      nodes: currentWorkflow.nodes
        .filter((n) => n.id !== id)
        .map((n) => ({
          ...n,
          connections: n.connections.filter((c) => c !== id),
        })),
      updatedAt: new Date().toISOString(),
    })
  }

  // Handle connection
  const handleConnect = (nodeId: string) => {
    if (!currentWorkflow) return

    if (!isConnecting) {
      setIsConnecting(true)
      setConnectingFrom(nodeId)
    } else if (connectingFrom && connectingFrom !== nodeId) {
      // Create connection
      setCurrentWorkflow({
        ...currentWorkflow,
        nodes: currentWorkflow.nodes.map((n) =>
          n.id === connectingFrom ? { ...n, connections: [...n.connections, nodeId] } : n
        ),
        updatedAt: new Date().toISOString(),
      })
      setIsConnecting(false)
      setConnectingFrom(null)
    } else {
      setIsConnecting(false)
      setConnectingFrom(null)
    }
  }

  // Save node
  const saveNode = (node: WorkflowNode) => {
    if (!currentWorkflow) return

    setCurrentWorkflow({
      ...currentWorkflow,
      nodes: currentWorkflow.nodes.map((n) => (n.id === node.id ? node : n)),
      updatedAt: new Date().toISOString(),
    })
    setEditingNode(null)
  }

  // Run workflow (simulation)
  const runWorkflow = async () => {
    if (!currentWorkflow || currentWorkflow.nodes.length === 0) return

    setRunning(true)
    setRunResult(null)

    // Simulate workflow execution
    await new Promise((resolve) => setTimeout(resolve, 2000))

    setRunResult('Workflow executed successfully!\n\nOutput:\n{\n  "result": "Sample output from workflow",\n  "steps_executed": ' + currentWorkflow.nodes.length + '\n}')
    setRunning(false)
  }

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
          p: 2,
          borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 2,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <WorkflowIcon sx={{ color: '#6366f1' }} />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Workflow Builder
          </Typography>
          {currentWorkflow && (
            <Chip
              label={currentWorkflow.name}
              size="small"
              sx={{
                bgcolor: alpha('#6366f1', 0.1),
                border: `1px solid ${alpha('#6366f1', 0.2)}`,
                color: '#818cf8',
              }}
            />
          )}
        </Box>

        <Box sx={{ display: 'flex', gap: 1 }}>
          {!currentWorkflow ? (
            <Button variant="contained" startIcon={<AddIcon />} onClick={createWorkflow}>
              New Workflow
            </Button>
          ) : (
            <>
              <Button
                variant="outlined"
                startIcon={<SaveIcon />}
                onClick={() => {
                  // Save to localStorage or API
                  localStorage.setItem(`workflow-${currentWorkflow.id}`, JSON.stringify(currentWorkflow))
                }}
              >
                Save
              </Button>
              <Button
                variant="contained"
                startIcon={running ? <CircularProgress size={16} /> : <PlayIcon />}
                onClick={runWorkflow}
                disabled={running || currentWorkflow.nodes.length === 0}
                sx={{ bgcolor: '#10b981', '&:hover': { bgcolor: '#059669' } }}
              >
                {running ? 'Running...' : 'Run'}
              </Button>
            </>
          )}
        </Box>
      </Box>

      {/* Main Content */}
      <Box sx={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Node Palette */}
        <Box
          sx={{
            width: 200,
            p: 2,
            borderRight: '1px solid rgba(255, 255, 255, 0.06)',
            overflowY: 'auto',
          }}
        >
          <Typography variant="overline" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
            Add Nodes
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {(Object.keys(nodeTypeConfig) as NodeType[]).map((type) => {
              const config = nodeTypeConfig[type]
              return (
                <Button
                  key={type}
                  variant="outlined"
                  startIcon={config.icon}
                  onClick={() => addNode(type)}
                  disabled={!currentWorkflow}
                  sx={{
                    justifyContent: 'flex-start',
                    borderColor: alpha(config.color, 0.3),
                    color: config.color,
                    '&:hover': {
                      borderColor: config.color,
                      bgcolor: alpha(config.color, 0.1),
                    },
                  }}
                >
                  {config.label}
                </Button>
              )
            })}
          </Box>

          {isConnecting && (
            <Alert severity="info" sx={{ mt: 2, fontSize: '0.75rem' }}>
              Click another node to connect, or click the same node to cancel.
            </Alert>
          )}
        </Box>

        {/* Canvas */}
        <Box
          sx={{
            flex: 1,
            p: 3,
            overflowY: 'auto',
            bgcolor: 'rgba(0, 0, 0, 0.2)',
            backgroundImage: `
              linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
              linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px)
            `,
            backgroundSize: '20px 20px',
          }}
        >
          {!currentWorkflow ? (
            <Box
              sx={{
                height: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexDirection: 'column',
                gap: 2,
              }}
            >
              <WorkflowIcon sx={{ fontSize: 64, color: 'text.secondary', opacity: 0.3 }} />
              <Typography color="text.secondary">Create a new workflow to get started</Typography>
              <Button variant="contained" startIcon={<AddIcon />} onClick={createWorkflow}>
                New Workflow
              </Button>
            </Box>
          ) : currentWorkflow.nodes.length === 0 ? (
            <Box
              sx={{
                height: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexDirection: 'column',
                gap: 2,
              }}
            >
              <Typography color="text.secondary">Add nodes from the left panel to build your workflow</Typography>
            </Box>
          ) : (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
              {currentWorkflow.nodes.map((node) => (
                <NodeCard
                  key={node.id}
                  node={node}
                  onEdit={setEditingNode}
                  onDelete={deleteNode}
                  onConnect={handleConnect}
                  isConnecting={isConnecting}
                  connectingFrom={connectingFrom}
                />
              ))}
            </Box>
          )}
        </Box>

        {/* Run Result Panel */}
        {runResult && (
          <Box
            sx={{
              width: 300,
              p: 2,
              borderLeft: '1px solid rgba(255, 255, 255, 0.06)',
              overflowY: 'auto',
            }}
          >
            <Typography variant="overline" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
              Run Result
            </Typography>
            <Box
              sx={{
                bgcolor: 'rgba(0, 0, 0, 0.3)',
                borderRadius: 1,
                p: 1.5,
                fontFamily: 'monospace',
                fontSize: '0.75rem',
                whiteSpace: 'pre-wrap',
              }}
            >
              {runResult}
            </Box>
          </Box>
        )}
      </Box>

      {/* Node Editor Dialog */}
      <NodeEditor open={!!editingNode} node={editingNode} onClose={() => setEditingNode(null)} onSave={saveNode} />
    </Box>
  )
}

export default WorkflowBuilderPage
