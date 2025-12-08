import React, { useState, useEffect, useCallback, useRef } from 'react'
import {
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Tooltip,
  Alert,
  CircularProgress,
  Divider,
  Paper,
  Tab,
  Tabs,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material'
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
  CloudUpload as UploadIcon,
  Download as DownloadIcon,
  Visibility as ViewIcon,
  Add as AddIcon,
  History as HistoryIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  HourglassEmpty as PendingIcon,
  PlayCircle as RunningIcon,
} from '@mui/icons-material'

interface BatchJob {
  id: string
  name: string
  status: string
  total_items: number
  completed_items: number
  failed_items: number
  progress: number
  created_at: string
  completed_at?: string
  config?: {
    max_tokens?: number
    temperature?: number
    system_prompt?: string
  }
}

interface BatchItem {
  id: string
  index: number
  input_text: string
  output_text?: string
  status: string
  error?: string
  tokens_used: number
  processing_time_ms: number
}

interface BatchStats {
  total_jobs: number
  completed_jobs: number
  running_jobs: number
  total_items: number
  processed_items: number
}

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'completed': return <SuccessIcon color="success" />
    case 'failed': return <ErrorIcon color="error" />
    case 'running': return <RunningIcon color="info" />
    case 'cancelled': return <StopIcon color="warning" />
    default: return <PendingIcon color="disabled" />
  }
}

const getStatusColor = (status: string): "success" | "error" | "info" | "warning" | "default" => {
  switch (status) {
    case 'completed': return 'success'
    case 'failed': return 'error'
    case 'running': return 'info'
    case 'cancelled': return 'warning'
    default: return 'default'
  }
}

export default function BatchProcessingPage() {
  // State
  const [tabValue, setTabValue] = useState(0)
  const [jobs, setJobs] = useState<BatchJob[]>([])
  const [stats, setStats] = useState<BatchStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  // Create job state
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [jobName, setJobName] = useState('')
  const [inputMethod, setInputMethod] = useState<'text' | 'file'>('text')
  const [inputText, setInputText] = useState('')
  const [fileContent, setFileContent] = useState('')
  const [fileType, setFileType] = useState('txt')
  const [maxTokens, setMaxTokens] = useState(512)
  const [temperature, setTemperature] = useState(0.7)
  const [systemPrompt, setSystemPrompt] = useState('')
  const [creating, setCreating] = useState(false)
  
  // View job state
  const [viewDialogOpen, setViewDialogOpen] = useState(false)
  const [selectedJob, setSelectedJob] = useState<BatchJob | null>(null)
  const [jobItems, setJobItems] = useState<BatchItem[]>([])
  const [itemsLoading, setItemsLoading] = useState(false)
  
  // Polling
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // Load data
  const loadData = useCallback(async () => {
    try {
      const [jobsRes, statsRes] = await Promise.all([
        fetch('/api/v1/batch/jobs?limit=50').then(r => r.json()),
        fetch('/api/v1/batch/stats').then(r => r.json()),
      ])
      setJobs(jobsRes.jobs || [])
      setStats(statsRes)
      
      // Check if any jobs are running
      const hasRunning = (jobsRes.jobs || []).some((j: BatchJob) => j.status === 'running')
      if (hasRunning && !pollIntervalRef.current) {
        pollIntervalRef.current = setInterval(loadData, 3000)
      } else if (!hasRunning && pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
        pollIntervalRef.current = null
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load batch jobs')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadData()
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
    }
  }, [loadData])

  // Create job
  const handleCreateJob = async () => {
    setCreating(true)
    setError(null)
    
    try {
      let requestBody: any = {
        name: jobName || `Batch Job ${new Date().toLocaleString()}`,
        config: {
          max_tokens: maxTokens,
          temperature: temperature,
          system_prompt: systemPrompt || undefined,
        },
      }
      
      if (inputMethod === 'text') {
        // Parse text input (one prompt per line)
        const lines = inputText.trim().split('\n').filter(l => l.trim())
        requestBody.items = lines.map(line => ({ input: line.trim() }))
      } else {
        requestBody.content = fileContent
        requestBody.file_type = fileType
      }
      
      const response = await fetch('/api/v1/batch/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      })
      
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to create job')
      }
      
      setCreateDialogOpen(false)
      resetCreateForm()
      loadData()
    } catch (err: any) {
      setError(err.message || 'Failed to create job')
    } finally {
      setCreating(false)
    }
  }

  const resetCreateForm = () => {
    setJobName('')
    setInputText('')
    setFileContent('')
    setFileType('txt')
    setMaxTokens(512)
    setTemperature(0.7)
    setSystemPrompt('')
  }

  // Run job
  const handleRunJob = async (jobId: string) => {
    try {
      const response = await fetch(`/api/v1/batch/jobs/${jobId}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to start job')
      }
      
      loadData()
    } catch (err: any) {
      setError(err.message || 'Failed to start job')
    }
  }

  // Cancel job
  const handleCancelJob = async (jobId: string) => {
    try {
      await fetch(`/api/v1/batch/jobs/${jobId}/cancel`, {
        method: 'POST',
      })
      loadData()
    } catch (err: any) {
      setError(err.message || 'Failed to cancel job')
    }
  }

  // Delete job
  const handleDeleteJob = async (jobId: string) => {
    if (!confirm('Delete this job and all its items?')) return
    try {
      await fetch(`/api/v1/batch/jobs/${jobId}`, {
        method: 'DELETE',
      })
      loadData()
    } catch (err: any) {
      setError(err.message || 'Failed to delete job')
    }
  }

  // View job details
  const handleViewJob = async (job: BatchJob) => {
    setSelectedJob(job)
    setItemsLoading(true)
    setViewDialogOpen(true)
    
    try {
      const response = await fetch(`/api/v1/batch/jobs/${job.id}/items?limit=100`)
      const data = await response.json()
      setJobItems(data.items || [])
    } catch (err: any) {
      setError(err.message || 'Failed to load job items')
    } finally {
      setItemsLoading(false)
    }
  }

  // Export job
  const handleExportJob = async (jobId: string, format: string) => {
    try {
      const response = await fetch(`/api/v1/batch/jobs/${jobId}/export?format=${format}`)
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `batch_${jobId}.${format}`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err: any) {
      setError(err.message || 'Failed to export job')
    }
  }

  // Handle file upload
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    
    const reader = new FileReader()
    reader.onload = (e) => {
      setFileContent(e.target?.result as string)
      
      // Detect file type
      if (file.name.endsWith('.json')) {
        setFileType('json')
      } else if (file.name.endsWith('.csv')) {
        setFileType('csv')
      } else {
        setFileType('txt')
      }
    }
    reader.readAsText(file)
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" sx={{ fontWeight: 600 }}>
          Batch Processing
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title="Refresh">
            <IconButton onClick={loadData}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setCreateDialogOpen(true)}
          >
            New Batch Job
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Stats Cards */}
      {stats && (
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={6} md={2.4}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <HistoryIcon color="primary" />
                <Typography variant="h5">{stats.total_jobs}</Typography>
                <Typography variant="caption" color="text.secondary">Total Jobs</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={2.4}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <SuccessIcon color="success" />
                <Typography variant="h5">{stats.completed_jobs}</Typography>
                <Typography variant="caption" color="text.secondary">Completed</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={2.4}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <RunningIcon color="info" />
                <Typography variant="h5">{stats.running_jobs}</Typography>
                <Typography variant="caption" color="text.secondary">Running</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={2.4}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <PendingIcon color="disabled" />
                <Typography variant="h5">{stats.total_items}</Typography>
                <Typography variant="caption" color="text.secondary">Total Items</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={2.4}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <SuccessIcon color="success" />
                <Typography variant="h5">{stats.processed_items}</Typography>
                <Typography variant="caption" color="text.secondary">Processed</Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Jobs List */}
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>Batch Jobs</Typography>
        
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress />
          </Box>
        ) : jobs.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography color="text.secondary">
              No batch jobs yet. Create a new job to get started.
            </Typography>
          </Box>
        ) : (
          <List>
            {jobs.map((job) => (
              <ListItem key={job.id} divider>
                <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
                  {getStatusIcon(job.status)}
                </Box>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body1" sx={{ fontWeight: 500 }}>
                        {job.name}
                      </Typography>
                      <Chip
                        size="small"
                        label={job.status}
                        color={getStatusColor(job.status)}
                      />
                    </Box>
                  }
                  secondary={
                    <Box sx={{ mt: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 0.5 }}>
                        <Typography variant="caption">
                          Items: {job.completed_items + job.failed_items} / {job.total_items}
                        </Typography>
                        {job.failed_items > 0 && (
                          <Typography variant="caption" color="error">
                            Failed: {job.failed_items}
                          </Typography>
                        )}
                        <Typography variant="caption" color="text.secondary">
                          Created: {new Date(job.created_at).toLocaleString()}
                        </Typography>
                      </Box>
                      {job.status === 'running' && (
                        <LinearProgress 
                          variant="determinate" 
                          value={job.progress} 
                          sx={{ mt: 1, height: 6, borderRadius: 3 }}
                        />
                      )}
                    </Box>
                  }
                />
                <ListItemSecondaryAction>
                  <Box sx={{ display: 'flex', gap: 0.5 }}>
                    {job.status === 'pending' && (
                      <Tooltip title="Run Job">
                        <IconButton size="small" onClick={() => handleRunJob(job.id)} color="primary">
                          <StartIcon />
                        </IconButton>
                      </Tooltip>
                    )}
                    {job.status === 'running' && (
                      <Tooltip title="Cancel Job">
                        <IconButton size="small" onClick={() => handleCancelJob(job.id)} color="warning">
                          <StopIcon />
                        </IconButton>
                      </Tooltip>
                    )}
                    <Tooltip title="View Details">
                      <IconButton size="small" onClick={() => handleViewJob(job)}>
                        <ViewIcon />
                      </IconButton>
                    </Tooltip>
                    {job.status === 'completed' && (
                      <Tooltip title="Export Results">
                        <IconButton size="small" onClick={() => handleExportJob(job.id, 'json')}>
                          <DownloadIcon />
                        </IconButton>
                      </Tooltip>
                    )}
                    <Tooltip title="Delete Job">
                      <IconButton size="small" onClick={() => handleDeleteJob(job.id)} color="error">
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        )}
      </Paper>

      {/* Create Job Dialog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create Batch Job</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <TextField
              label="Job Name"
              value={jobName}
              onChange={(e) => setJobName(e.target.value)}
              fullWidth
              placeholder="Optional - auto-generated if empty"
            />

            <Divider />

            <Typography variant="subtitle2">Input Method</Typography>
            <Tabs value={inputMethod} onChange={(_, v) => setInputMethod(v)}>
              <Tab label="Text Input" value="text" />
              <Tab label="File Upload" value="file" />
            </Tabs>

            {inputMethod === 'text' ? (
              <TextField
                label="Prompts (one per line)"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                multiline
                rows={8}
                fullWidth
                placeholder="Enter one prompt per line..."
              />
            ) : (
              <Box>
                <Button
                  variant="outlined"
                  component="label"
                  startIcon={<UploadIcon />}
                >
                  Upload File (JSON, CSV, or TXT)
                  <input
                    type="file"
                    hidden
                    accept=".json,.csv,.txt"
                    onChange={handleFileUpload}
                  />
                </Button>
                {fileContent && (
                  <Typography variant="caption" sx={{ ml: 2 }}>
                    File loaded ({fileType.toUpperCase()}, {fileContent.length} characters)
                  </Typography>
                )}
              </Box>
            )}

            <Divider />

            <Typography variant="subtitle2">Configuration</Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <TextField
                  label="Max Tokens"
                  type="number"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value) || 512)}
                  fullWidth
                  size="small"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Temperature"
                  type="number"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value) || 0.7)}
                  fullWidth
                  size="small"
                  inputProps={{ step: 0.1, min: 0, max: 2 }}
                />
              </Grid>
            </Grid>

            <TextField
              label="System Prompt (optional)"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              multiline
              rows={3}
              fullWidth
              placeholder="Optional system prompt to prepend to all items..."
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleCreateJob}
            disabled={creating || (!inputText.trim() && !fileContent)}
            startIcon={creating ? <CircularProgress size={20} /> : <AddIcon />}
          >
            Create Job
          </Button>
        </DialogActions>
      </Dialog>

      {/* View Job Dialog */}
      <Dialog open={viewDialogOpen} onClose={() => setViewDialogOpen(false)} maxWidth="lg" fullWidth>
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {selectedJob?.name}
            <Chip
              size="small"
              label={selectedJob?.status}
              color={getStatusColor(selectedJob?.status || '')}
            />
          </Box>
        </DialogTitle>
        <DialogContent>
          {selectedJob && (
            <Box>
              {/* Job Summary */}
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={3}>
                  <Typography variant="caption" color="text.secondary">Total Items</Typography>
                  <Typography variant="h6">{selectedJob.total_items}</Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="caption" color="text.secondary">Completed</Typography>
                  <Typography variant="h6" color="success.main">{selectedJob.completed_items}</Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="caption" color="text.secondary">Failed</Typography>
                  <Typography variant="h6" color="error.main">{selectedJob.failed_items}</Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="caption" color="text.secondary">Progress</Typography>
                  <Typography variant="h6">{selectedJob.progress}%</Typography>
                </Grid>
              </Grid>

              {selectedJob.status === 'running' && (
                <LinearProgress 
                  variant="determinate" 
                  value={selectedJob.progress} 
                  sx={{ mb: 3, height: 8, borderRadius: 4 }}
                />
              )}

              <Divider sx={{ my: 2 }} />

              {/* Items Table */}
              <Typography variant="subtitle2" sx={{ mb: 2 }}>Items</Typography>
              {itemsLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                  <CircularProgress />
                </Box>
              ) : (
                <TableContainer sx={{ maxHeight: 400 }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell width={50}>#</TableCell>
                        <TableCell>Input</TableCell>
                        <TableCell>Output</TableCell>
                        <TableCell width={80}>Status</TableCell>
                        <TableCell width={80}>Tokens</TableCell>
                        <TableCell width={80}>Time</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {jobItems.map((item) => (
                        <TableRow key={item.id}>
                          <TableCell>{item.index + 1}</TableCell>
                          <TableCell>
                            <Typography variant="body2" sx={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {item.input_text}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" sx={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {item.output_text || item.error || '-'}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Chip size="small" label={item.status} color={getStatusColor(item.status)} />
                          </TableCell>
                          <TableCell>{item.tokens_used || '-'}</TableCell>
                          <TableCell>{item.processing_time_ms ? `${(item.processing_time_ms / 1000).toFixed(1)}s` : '-'}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          {selectedJob?.status === 'completed' && (
            <>
              <Button onClick={() => handleExportJob(selectedJob.id, 'json')} startIcon={<DownloadIcon />}>
                Export JSON
              </Button>
              <Button onClick={() => handleExportJob(selectedJob.id, 'csv')} startIcon={<DownloadIcon />}>
                Export CSV
              </Button>
            </>
          )}
          <Button onClick={() => setViewDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}
