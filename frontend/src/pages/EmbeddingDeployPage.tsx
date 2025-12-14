import React, { useEffect, useState } from 'react'
import {
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  TextField,
  Select,
  MenuItem,
  Button,
  Alert,
  Chip,
  CircularProgress,
  FormHelperText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
} from '@mui/material'
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  RestartAlt as RestartIcon,
  Save as SaveIcon,
  Science as TestIcon,
  CloudDownload as DownloadIcon,
  Delete as DeleteIcon,
  Info as InfoIcon,
  CheckCircle as CheckIcon,
  Cancel as CancelIcon,
} from '@mui/icons-material'

// Embedding model repository mappings
const EMBEDDING_MODEL_REPOS = {
  'nomic-embed-text-v1.5': {
    repo: 'nomic-ai/nomic-embed-text-v1.5-GGUF',
    files: ['nomic-embed-text-v1.5.Q8_0.gguf', 'nomic-embed-text-v1.5.Q4_K_M.gguf', 'nomic-embed-text-v1.5.Q4_0.gguf']
  },
  'e5-mistral-7b': {
    repo: 'intfloat/e5-mistral-7b-instruct-GGUF',
    files: ['e5-mistral-7b-instruct.Q8_0.gguf', 'e5-mistral-7b-instruct.Q4_K_M.gguf', 'e5-mistral-7b-instruct.Q4_0.gguf']
  },
  'bge-m3': {
    repo: 'BAAI/bge-m3-GGUF',
    files: ['bge-m3.Q8_0.gguf', 'bge-m3.Q4_K_M.gguf', 'bge-m3.Q4_0.gguf']
  },
  'gte-Qwen2-1.5B': {
    repo: 'Alibaba-NLP/gte-Qwen2-1.5B-instruct-GGUF',
    files: ['gte-Qwen2-1.5B-instruct.Q8_0.gguf', 'gte-Qwen2-1.5B-instruct.Q4_K_M.gguf', 'gte-Qwen2-1.5B-instruct.Q4_0.gguf']
  }
}

export const EmbeddingDeployPage: React.FC = () => {
  // Embedding service state
  const [embeddingStatus, setEmbeddingStatus] = useState<any>(null)
  const [embeddingConfig, setEmbeddingConfig] = useState<any>(null)
  const [embeddingLoading, setEmbeddingLoading] = useState<'start' | 'stop' | 'restart' | null>(null)
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState<any[]>([])
  const [embeddingError, setEmbeddingError] = useState<string | null>(null)
  const [embeddingSuccess, setEmbeddingSuccess] = useState<string | null>(null)
  
  // RAG embedding configuration
  const [ragEmbeddingConfig, setRagEmbeddingConfig] = useState<any>(null)
  const [ragEmbeddingLoading, setRagEmbeddingLoading] = useState(false)
  
  // Test embedding state
  const [testText, setTestText] = useState('The quick brown fox jumps over the lazy dog.')
  const [testResult, setTestResult] = useState<any>(null)
  const [testLoading, setTestLoading] = useState(false)
  const [testError, setTestError] = useState<string | null>(null)
  
  // Model download state
  const [downloadDialogOpen, setDownloadDialogOpen] = useState(false)
  const [selectedModelForDownload, setSelectedModelForDownload] = useState<string>('')
  const [selectedVariant, setSelectedVariant] = useState<string>('Q8_0')
  const [downloadLoading, setDownloadLoading] = useState(false)
  const [activeDownloads, setActiveDownloads] = useState<any[]>([])
  const [localModels, setLocalModels] = useState<any[]>([])
  
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const init = async () => {
      try {
        setLoading(true)
        
        // Load embedding service config and status
        const embedCfgRes = await fetch('/api/v1/embedding/config')
        if (embedCfgRes.ok) {
          const embedCfg = await embedCfgRes.json()
          setEmbeddingConfig(embedCfg.config)
          setAvailableEmbeddingModels(embedCfg.available_models || [])
        }
        
        const embedStatusRes = await fetch('/api/v1/embedding/status')
        if (embedStatusRes.ok) {
          const embedStatus = await embedStatusRes.json()
          setEmbeddingStatus(embedStatus)
        }
        
        // Load RAG embedding configuration
        const ragEmbedCfgRes = await fetch('/api/v1/rag/embeddings/config')
        if (ragEmbedCfgRes.ok) {
          const ragEmbedCfg = await ragEmbedCfgRes.json()
          setRagEmbeddingConfig(ragEmbedCfg)
        }
        
        // Load local models and downloads
        await Promise.all([
          loadLocalModels(),
          loadActiveDownloads()
        ])
      } catch (e) {
        console.error('Failed to fetch embedding service info:', e)
      } finally {
        setLoading(false)
      }
    }
    
    init()
  }, [])

  // Refresh embedding status and downloads periodically
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch('/api/v1/embedding/status')
        if (res.ok) {
          const status = await res.json()
          setEmbeddingStatus(status)
        }
        
        // Also refresh downloads if there are active ones
        if (activeDownloads.length > 0) {
          await loadActiveDownloads()
        }
      } catch (e) {
        // Silently fail
      }
    }, 5000)
    
    return () => clearInterval(interval)
  }, [activeDownloads.length])

  // Embedding service action handlers
  const runEmbeddingAction = async (action: 'start' | 'stop' | 'restart') => {
    try {
      setEmbeddingLoading(action)
      setEmbeddingError(null)
      
      const res = await fetch(`/api/v1/embedding/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      
      if (!res.ok) {
        throw new Error(`Failed to ${action} embedding service`)
      }
      
      const result = await res.json()
      setEmbeddingStatus(result.status)
      setEmbeddingSuccess(result.message)
      
      // Clear success message after 3 seconds
      setTimeout(() => setEmbeddingSuccess(null), 3000)
    } catch (e) {
      setEmbeddingError(e instanceof Error ? e.message : `Failed to ${action} embedding service`)
    } finally {
      setEmbeddingLoading(null)
    }
  }
  
  const updateEmbeddingConfig = (path: string, value: any) => {
    if (!embeddingConfig) return
    
    const keys = path.split('.')
    const newConfig = JSON.parse(JSON.stringify(embeddingConfig))
    
    let current = newConfig
    for (let i = 0; i < keys.length - 1; i++) {
      if (!current[keys[i]]) current[keys[i]] = {}
      current = current[keys[i]]
    }
    current[keys[keys.length - 1]] = value
    
    setEmbeddingConfig(newConfig)
  }
  
  const saveEmbeddingConfig = async () => {
    try {
      const res = await fetch('/api/v1/embedding/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(embeddingConfig)
      })
      
      if (!res.ok) {
        throw new Error('Failed to save embedding config')
      }
      
      const result = await res.json()
      setEmbeddingSuccess(result.message)
      setTimeout(() => setEmbeddingSuccess(null), 3000)
    } catch (e) {
      setEmbeddingError(e instanceof Error ? e.message : 'Failed to save embedding config')
    }
  }
  
  const updateRagEmbeddingConfig = (field: string, value: any) => {
    if (!ragEmbeddingConfig) return
    setRagEmbeddingConfig({ ...ragEmbeddingConfig, [field]: value })
  }
  
  const saveRagEmbeddingConfig = async () => {
    try {
      setRagEmbeddingLoading(true)
      const res = await fetch('/api/v1/rag/embeddings/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ragEmbeddingConfig)
      })
      
      if (!res.ok) {
        throw new Error('Failed to save RAG embedding config')
      }
      
      const result = await res.json()
      setEmbeddingSuccess('RAG embedding configuration saved successfully')
      setTimeout(() => setEmbeddingSuccess(null), 3000)
    } catch (e) {
      setEmbeddingError(e instanceof Error ? e.message : 'Failed to save RAG embedding config')
    } finally {
      setRagEmbeddingLoading(false)
    }
  }
  
  const testEmbedding = async () => {
    if (!testText.trim()) {
      setTestError('Please enter some text to test')
      return
    }
    
    try {
      setTestLoading(true)
      setTestError(null)
      setTestResult(null)
      
      // Use the backend API endpoint which proxies to the embedding service
      const res = await fetch('/api/v1/embedding/test', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: testText
        })
      })
      
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(errorData.detail || `Service returned ${res.status}`)
      }
      
      const result = await res.json()
      
      setTestResult({
        model: result.model,
        usage: result.usage,
        data: result.data,
        timeTaken: result.timeTaken,
        vectorLength: result.vectorLength
      })
      
    } catch (e) {
      setTestError(e instanceof Error ? e.message : 'Failed to test embedding')
    } finally {
      setTestLoading(false)
    }
  }
  
  // Model download functions
  const loadLocalModels = async () => {
    try {
      const res = await fetch('/v1/models')
      if (res.ok) {
        const data = await res.json()
        setLocalModels(data.data || [])
      }
    } catch (e) {
      console.error('Failed to load local models:', e)
    }
  }
  
  const loadActiveDownloads = async () => {
    try {
      const res = await fetch('/v1/models/downloads')
      if (res.ok) {
        const data = await res.json()
        setActiveDownloads(data.data || [])
      }
    } catch (e) {
      console.error('Failed to load downloads:', e)
    }
  }
  
  const startModelDownload = async () => {
    if (!selectedModelForDownload || !selectedVariant) {
      setEmbeddingError('Please select a model and variant')
      return
    }
    
    const modelRepo = EMBEDDING_MODEL_REPOS[selectedModelForDownload as keyof typeof EMBEDDING_MODEL_REPOS]
    if (!modelRepo) {
      setEmbeddingError('Invalid model selected')
      return
    }
    
    const filename = modelRepo.files.find(f => f.includes(selectedVariant))
    if (!filename) {
      setEmbeddingError('Invalid variant selected')
      return
    }
    
    try {
      setDownloadLoading(true)
      setEmbeddingError(null)
      
      const res = await fetch('/v1/models/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repositoryId: modelRepo.repo,
          filename: filename,
          priority: 'normal'
        })
      })
      
      if (!res.ok) {
        const errorData = await res.json()
        throw new Error(errorData.detail || 'Failed to start download')
      }
      
      setEmbeddingSuccess('Model download started successfully')
      setDownloadDialogOpen(false)
      setSelectedModelForDownload('')
      setSelectedVariant('Q8_0')
      
      // Refresh downloads and models
      await Promise.all([loadActiveDownloads(), loadLocalModels()])
      
      setTimeout(() => setEmbeddingSuccess(null), 3000)
    } catch (e) {
      setEmbeddingError(e instanceof Error ? e.message : 'Failed to start download')
    } finally {
      setDownloadLoading(false)
    }
  }
  
  const cancelDownload = async (modelId: string) => {
    try {
      const res = await fetch(`/v1/models/downloads/${modelId}`, {
        method: 'DELETE'
      })
      
      if (res.ok) {
        setEmbeddingSuccess('Download cancelled')
        await loadActiveDownloads()
        setTimeout(() => setEmbeddingSuccess(null), 3000)
      }
    } catch (e) {
      setEmbeddingError('Failed to cancel download')
    }
  }
  
  const deleteLocalModel = async (filePath: string) => {
    try {
      const res = await fetch('/v1/models/local-files', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_path: filePath })
      })
      
      if (res.ok) {
        setEmbeddingSuccess('Model deleted successfully')
        await loadLocalModels()
        setTimeout(() => setEmbeddingSuccess(null), 3000)
      }
    } catch (e) {
      setEmbeddingError('Failed to delete model')
    }
  }
  
  const getModelStatus = (modelName: string, variant: string) => {
    const expectedFilename = EMBEDDING_MODEL_REPOS[modelName as keyof typeof EMBEDDING_MODEL_REPOS]?.files
      .find(f => f.includes(variant))
    
    if (!expectedFilename) return 'unknown'
    
    // Check if model is downloaded
    const isDownloaded = localModels.some(model => 
      model.filename === expectedFilename || model.path?.includes(expectedFilename)
    )
    
    if (isDownloaded) return 'downloaded'
    
    // Check if model is downloading
    const isDownloading = activeDownloads.some(download => 
      download.filename === expectedFilename && download.status === 'downloading'
    )
    
    if (isDownloading) return 'downloading'
    
    return 'not_downloaded'
  }

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    )
  }

  return (
    <Box sx={{ p: { xs: 2, sm: 3, md: 4 }, maxWidth: '100%', mx: 'auto' }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
          Embedding Model Deployment
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Deploy and manage dedicated embedding models for RAG operations
        </Typography>
      </Box>

      {embeddingError && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setEmbeddingError(null)}>
          {embeddingError}
        </Alert>
      )}
      
      {embeddingSuccess && (
        <Alert severity="success" sx={{ mb: 3 }} onClose={() => setEmbeddingSuccess(null)}>
          {embeddingSuccess}
        </Alert>
      )}

      {embeddingConfig && (
        <Box>
          {/* Status and Control */}
          <Card sx={{
            mb: 3,
            borderRadius: 2,
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
            border: '1px solid',
            borderColor: 'divider',
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontSize: '1.125rem', fontWeight: 600 }}>
                Service Status
              </Typography>
              
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
                <Chip
                  label={embeddingStatus?.running ? 'Running' : 'Stopped'}
                  color={embeddingStatus?.running ? 'success' : 'default'}
                  size="small"
                />
                {embeddingStatus?.running && (
                  <>
                    <Typography variant="body2" color="text.secondary">
                      Uptime: {Math.floor(embeddingStatus.uptime / 60)}m {Math.floor(embeddingStatus.uptime % 60)}s
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Model: {embeddingStatus.model?.name}
                    </Typography>
                  </>
                )}
              </Box>
              
              {embeddingStatus?.endpoint && (
                <Alert severity="info" sx={{ mb: 3 }}>
                  <strong>Service Endpoint:</strong> {embeddingStatus.endpoint}
                </Alert>
              )}
              
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <Button
                  variant="contained"
                  color="primary"
                  size="large"
                  startIcon={embeddingLoading === 'start' ? <CircularProgress size={20} /> : <StartIcon />}
                  onClick={() => runEmbeddingAction('start')}
                  disabled={embeddingLoading !== null || embeddingStatus?.running}
                  sx={{ borderRadius: 1.5, fontWeight: 600, px: 3 }}
                >
                  Start Service
                </Button>
                <Button
                  variant="contained"
                  color="warning"
                  size="large"
                  startIcon={embeddingLoading === 'restart' ? <CircularProgress size={20} /> : <RestartIcon />}
                  onClick={() => runEmbeddingAction('restart')}
                  disabled={embeddingLoading !== null}
                  sx={{ borderRadius: 1.5, fontWeight: 600, px: 3 }}
                >
                  Restart
                </Button>
                <Button
                  variant="outlined"
                  color="error"
                  size="large"
                  startIcon={embeddingLoading === 'stop' ? <CircularProgress size={20} /> : <StopIcon />}
                  onClick={() => runEmbeddingAction('stop')}
                  disabled={embeddingLoading !== null || !embeddingStatus?.running}
                  sx={{ borderRadius: 1.5, fontWeight: 600, px: 3 }}
                >
                  Stop
                </Button>
              </Box>
            </CardContent>
          </Card>

          {/* Model Configuration */}
          <Card sx={{
            mb: 3,
            borderRadius: 2,
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
            border: '1px solid',
            borderColor: 'divider',
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontSize: '1.125rem', fontWeight: 600 }}>
                Model Configuration
              </Typography>
              
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                    Embedding Model
                  </Typography>
                  <Select
                    fullWidth
                    value={embeddingConfig?.model?.name || ''}
                    onChange={(e) => updateEmbeddingConfig('model.name', e.target.value)}
                    sx={{ fontSize: '0.875rem' }}
                  >
                    {availableEmbeddingModels.map((model) => (
                      <MenuItem key={model.name} value={model.name}>
                        {model.name} ({model.dimensions}D, {model.max_tokens} tokens)
                      </MenuItem>
                    ))}
                  </Select>
                  <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                    {availableEmbeddingModels.find(m => m.name === embeddingConfig?.model?.name)?.description}
                  </FormHelperText>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                    Model Variant (Quantization)
                  </Typography>
                  <Select
                    fullWidth
                    value={embeddingConfig?.model?.variant || 'Q8_0'}
                    onChange={(e) => updateEmbeddingConfig('model.variant', e.target.value)}
                    sx={{ fontSize: '0.875rem' }}
                  >
                    <MenuItem value="Q8_0">Q8_0 (Recommended - High Quality)</MenuItem>
                    <MenuItem value="Q4_K_M">Q4_K_M (Balanced)</MenuItem>
                    <MenuItem value="Q4_0">Q4_0 (Smaller, Faster)</MenuItem>
                  </Select>
                  <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                    Higher quantization = better quality but larger size
                  </FormHelperText>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                    Context Size
                  </Typography>
                  <TextField
                    fullWidth
                    type="number"
                    value={embeddingConfig?.model?.context_size || 8192}
                    onChange={(e) => updateEmbeddingConfig('model.context_size', parseInt(e.target.value))}
                    inputProps={{ min: 512, max: 32768 }}
                  />
                  <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                    Maximum context length for embeddings
                  </FormHelperText>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                    GPU Layers
                  </Typography>
                  <TextField
                    fullWidth
                    type="number"
                    value={embeddingConfig?.model?.gpu_layers || 999}
                    onChange={(e) => updateEmbeddingConfig('model.gpu_layers', parseInt(e.target.value))}
                    inputProps={{ min: 0, max: 999 }}
                  />
                  <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                    Number of layers to offload to GPU (999 = all)
                  </FormHelperText>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                    Execution Mode
                  </Typography>
                  <Select
                    fullWidth
                    value={embeddingConfig?.execution?.mode || 'gpu'}
                    onChange={(e) => updateEmbeddingConfig('execution.mode', e.target.value)}
                    sx={{ fontSize: '0.875rem' }}
                  >
                    <MenuItem value="gpu">GPU Acceleration</MenuItem>
                    <MenuItem value="cpu">CPU Only</MenuItem>
                  </Select>
                  <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                    Choose GPU or CPU execution
                  </FormHelperText>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                    CUDA Devices
                  </Typography>
                  <TextField
                    fullWidth
                    value={embeddingConfig?.execution?.cuda_devices || 'all'}
                    onChange={(e) => updateEmbeddingConfig('execution.cuda_devices', e.target.value)}
                    disabled={embeddingConfig?.execution?.mode === 'cpu'}
                  />
                  <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                    GPU devices to use (all, 0, 0,1, etc.)
                  </FormHelperText>
                </Grid>
              </Grid>
              
              <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<SaveIcon />}
                  onClick={saveEmbeddingConfig}
                  sx={{ borderRadius: 1.5, fontWeight: 600, px: 3 }}
                >
                  Save Configuration
                </Button>
              </Box>
            </CardContent>
          </Card>

          {/* Model Downloads */}
          <Card sx={{
            mb: 3,
            borderRadius: 2,
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
            border: '1px solid',
            borderColor: 'divider',
          }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                <Box>
                  <Typography variant="h6" gutterBottom sx={{ fontSize: '1.125rem', fontWeight: 600 }}>
                    Embedding Models
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Download and manage embedding models for the service
                  </Typography>
                </Box>
                <Button
                  variant="contained"
                  startIcon={<DownloadIcon />}
                  onClick={() => setDownloadDialogOpen(true)}
                  sx={{ borderRadius: 1.5, fontWeight: 600, px: 3 }}
                >
                  Download Model
                </Button>
              </Box>
              
              {/* Available Models Status */}
              <Grid container spacing={2}>
                {availableEmbeddingModels.map((model) => {
                  const modelStatus = getModelStatus(model.name, embeddingConfig?.model?.variant || 'Q8_0')
                  const isCurrentModel = embeddingConfig?.model?.name === model.name
                  
                  return (
                    <Grid item xs={12} md={6} key={model.name}>
                      <Card variant="outlined" sx={{ 
                        p: 2,
                        bgcolor: isCurrentModel ? 'primary.50' : 'background.paper',
                        borderColor: isCurrentModel ? 'primary.main' : 'divider'
                      }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                          <Typography variant="h6" sx={{ fontSize: '0.9rem', fontWeight: 600 }}>
                            {model.name}
                          </Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {isCurrentModel && (
                              <Chip label="Current" size="small" color="primary" />
                            )}
                            <Chip 
                              label={
                                modelStatus === 'downloaded' ? 'Downloaded' :
                                modelStatus === 'downloading' ? 'Downloading' :
                                'Not Downloaded'
                              }
                              size="small"
                              color={
                                modelStatus === 'downloaded' ? 'success' :
                                modelStatus === 'downloading' ? 'warning' :
                                'default'
                              }
                              icon={
                                modelStatus === 'downloaded' ? <CheckIcon sx={{ fontSize: '0.8rem' }} /> :
                                modelStatus === 'downloading' ? <CircularProgress size={12} /> :
                                undefined
                              }
                            />
                          </Box>
                        </Box>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                          {model.description}
                        </Typography>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Typography variant="caption" color="text.secondary">
                            {model.dimensions}D • {model.max_tokens} tokens
                          </Typography>
                          {modelStatus === 'downloading' && (
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <LinearProgress sx={{ width: 60, height: 4 }} />
                              <Typography variant="caption">
                                {activeDownloads.find(d => d.filename?.includes(model.name))?.progress || 0}%
                              </Typography>
                            </Box>
                          )}
                        </Box>
                      </Card>
                    </Grid>
                  )
                })}
              </Grid>
              
              {/* Active Downloads */}
              {activeDownloads.length > 0 && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ fontSize: '1rem', fontWeight: 600 }}>
                    Active Downloads
                  </Typography>
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Model</TableCell>
                          <TableCell>Progress</TableCell>
                          <TableCell>Speed</TableCell>
                          <TableCell>ETA</TableCell>
                          <TableCell align="right">Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {activeDownloads.map((download) => (
                          <TableRow key={download.id}>
                            <TableCell>
                              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                {download.filename}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {download.repository_id}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <LinearProgress 
                                  variant="determinate" 
                                  value={download.progress || 0} 
                                  sx={{ width: 80, height: 6 }}
                                />
                                <Typography variant="caption">
                                  {Math.round(download.progress || 0)}%
                                </Typography>
                              </Box>
                            </TableCell>
                            <TableCell>
                              <Typography variant="caption">
                                {download.download_speed || 'N/A'}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="caption">
                                {download.eta || 'N/A'}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Tooltip title="Cancel Download">
                                <IconButton 
                                  size="small" 
                                  onClick={() => cancelDownload(download.id)}
                                  color="error"
                                >
                                  <CancelIcon fontSize="small" />
                                </IconButton>
                              </Tooltip>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Box>
              )}
            </CardContent>
          </Card>

          {/* RAG Integration */}
          {ragEmbeddingConfig && (
            <Card sx={{
              mb: 3,
              borderRadius: 2,
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
              border: '1px solid',
              borderColor: 'divider',
            }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontSize: '1.125rem', fontWeight: 600 }}>
                  RAG System Integration
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Configure how the RAG system uses embeddings for document processing and retrieval.
                </Typography>
                
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <Box sx={{ flex: 1 }}>
                        <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                          Embedding Source
                        </Typography>
                        <FormHelperText sx={{ mt: 0.5, mb: 2, fontSize: '0.75rem' }}>
                          When enabled, the RAG system will use this deployed embedding service instead of local sentence-transformers.
                          The service must be running for this to work.
                        </FormHelperText>
                      </Box>
                      <Select
                        value={ragEmbeddingConfig?.use_deployed_service ? 'deployed' : 'local'}
                        onChange={(e) => updateRagEmbeddingConfig('use_deployed_service', e.target.value === 'deployed')}
                        sx={{ minWidth: 220, fontSize: '0.875rem' }}
                      >
                        <MenuItem value="local">Local (Sentence Transformers)</MenuItem>
                        <MenuItem value="deployed">Deployed Service</MenuItem>
                      </Select>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12}>
                    <Alert 
                      severity={ragEmbeddingConfig?.service_running ? 'success' : 'warning'} 
                      sx={{ mb: 2 }}
                    >
                      {ragEmbeddingConfig?.service_running ? (
                        <>
                          <strong>Embedding service is running</strong>
                          <br />
                          {ragEmbeddingConfig.use_deployed_service 
                            ? 'RAG system is configured to use the deployed service.' 
                            : 'Enable deployed service above to use it for RAG.'}
                        </>
                      ) : (
                        <>
                          <strong>Embedding service is not running</strong>
                          <br />
                          Start the service above to enable RAG integration. The system will fall back to local embeddings if the service is unavailable.
                        </>
                      )}
                    </Alert>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                      Default Embedding Model
                    </Typography>
                    <TextField
                      fullWidth
                      value={ragEmbeddingConfig?.default_model || ''}
                      onChange={(e) => updateRagEmbeddingConfig('default_model', e.target.value)}
                    />
                    <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                      Default model to use when no model is specified
                    </FormHelperText>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                      Service URL
                    </Typography>
                    <TextField
                      fullWidth
                      value={ragEmbeddingConfig?.service_url || ''}
                      onChange={(e) => updateRagEmbeddingConfig('service_url', e.target.value)}
                    />
                    <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                      URL of the embedding service endpoint
                    </FormHelperText>
                  </Grid>
                </Grid>
                
                <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                  <Button
                    variant="contained"
                    startIcon={ragEmbeddingLoading ? <CircularProgress size={20} /> : <SaveIcon />}
                    onClick={saveRagEmbeddingConfig}
                    disabled={ragEmbeddingLoading}
                    sx={{ borderRadius: 1.5, fontWeight: 600, px: 3 }}
                  >
                    Save RAG Configuration
                  </Button>
                </Box>
              </CardContent>
            </Card>
          )}

          {/* Test Embedding */}
          {embeddingStatus?.running && (
            <Card sx={{
              mb: 3,
              borderRadius: 2,
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
              border: '1px solid',
              borderColor: 'divider',
            }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontSize: '1.125rem', fontWeight: 600 }}>
                  Test Embedding
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Test the embedding service with sample text to verify it's working correctly.
                </Typography>
                
                {testError && (
                  <Alert severity="error" sx={{ mb: 2 }} onClose={() => setTestError(null)}>
                    {testError}
                  </Alert>
                )}
                
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                      Test Text
                    </Typography>
                    <TextField
                      fullWidth
                      multiline
                      rows={3}
                      value={testText}
                      onChange={(e) => setTestText(e.target.value)}
                      placeholder="Enter text to generate embeddings..."
                    />
                    <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                      Enter any text to see how the model generates embeddings
                    </FormHelperText>
                  </Grid>
                  
                  <Grid item xs={12}>
                    <Button
                      variant="contained"
                      color="primary"
                      size="large"
                      startIcon={testLoading ? <CircularProgress size={20} /> : <TestIcon />}
                      onClick={testEmbedding}
                      disabled={testLoading || !testText.trim()}
                      sx={{ borderRadius: 1.5, fontWeight: 600, px: 3 }}
                    >
                      Generate Embedding
                    </Button>
                  </Grid>
                  
                  {testResult && (
                    <Grid item xs={12}>
                      <Alert severity="success" sx={{ mb: 2 }}>
                        <strong>Embedding Generated Successfully!</strong>
                      </Alert>
                      
                      <Grid container spacing={2}>
                        <Grid item xs={12} sm={6} md={3}>
                          <Card variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Vector Dimensions
                            </Typography>
                            <Typography variant="h5" sx={{ fontWeight: 600, color: 'primary.main' }}>
                              {testResult.vectorLength}
                            </Typography>
                          </Card>
                        </Grid>
                        
                        <Grid item xs={12} sm={6} md={3}>
                          <Card variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Processing Time
                            </Typography>
                            <Typography variant="h5" sx={{ fontWeight: 600, color: 'success.main' }}>
                              {testResult.timeTaken}ms
                            </Typography>
                          </Card>
                        </Grid>
                        
                        <Grid item xs={12} sm={6} md={3}>
                          <Card variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Model Used
                            </Typography>
                            <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '0.9rem' }}>
                              {testResult.model || 'N/A'}
                            </Typography>
                          </Card>
                        </Grid>
                        
                        <Grid item xs={12} sm={6} md={3}>
                          <Card variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Usage
                            </Typography>
                            <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '0.9rem' }}>
                              {testResult.usage?.total_tokens || 'N/A'} tokens
                            </Typography>
                          </Card>
                        </Grid>
                        
                        <Grid item xs={12}>
                          <Typography variant="body2" gutterBottom sx={{ fontWeight: 500 }}>
                            Embedding Vector (first 10 dimensions):
                          </Typography>
                          <Box sx={{ 
                            p: 2, 
                            bgcolor: 'grey.100', 
                            borderRadius: 1, 
                            fontFamily: 'monospace',
                            fontSize: '0.75rem',
                            overflowX: 'auto',
                            maxHeight: 200,
                            overflowY: 'auto'
                          }}>
                            {testResult.data?.[0]?.embedding?.slice(0, 10).map((val: number, idx: number) => (
                              <span key={idx}>
                                {val.toFixed(6)}
                                {idx < 9 ? ', ' : '...'}
                              </span>
                            )) || 'No embedding data'}
                          </Box>
                          <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                            Showing first 10 of {testResult.vectorLength} dimensions
                          </FormHelperText>
                        </Grid>
                      </Grid>
                    </Grid>
                  )}
                </Grid>
              </CardContent>
            </Card>
          )}

          {/* Info */}
          <Card sx={{
            borderRadius: 2,
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
            border: '1px solid',
            borderColor: 'divider',
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontSize: '1.125rem', fontWeight: 600 }}>
                About Embedding Service
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                The embedding service provides text embeddings for the RAG (Retrieval Augmented Generation) system.
                It runs as a separate llama.cpp server optimized for embedding generation.
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                After starting the service, it will be available at the endpoint shown above.
                The service uses the same Docker infrastructure as the main LLM service.
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Configure the RAG Integration section above to connect your RAG system to this deployed embedding service
                for faster and more efficient embedding generation during document processing and retrieval.
              </Typography>
            </CardContent>
          </Card>
        </Box>
      )}

      {/* Download Model Dialog */}
      <Dialog open={downloadDialogOpen} onClose={() => setDownloadDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          <Box display="flex" alignItems="center" gap={1}>
            <DownloadIcon />
            <Typography variant="h6">Download Embedding Model</Typography>
          </Box>
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, pt: 1 }}>
            <Box>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Select Model
              </Typography>
              <Select
                fullWidth
                value={selectedModelForDownload}
                onChange={(e) => setSelectedModelForDownload(e.target.value)}
                displayEmpty
              >
                <MenuItem value="" disabled>Choose an embedding model...</MenuItem>
                {availableEmbeddingModels.map((model) => (
                  <MenuItem key={model.name} value={model.name}>
                    {model.name} ({model.dimensions}D, {model.max_tokens} tokens)
                  </MenuItem>
                ))}
              </Select>
              {selectedModelForDownload && (
                <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                  {availableEmbeddingModels.find(m => m.name === selectedModelForDownload)?.description}
                </FormHelperText>
              )}
            </Box>

            <Box>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Model Variant (Quantization)
              </Typography>
              <Select
                fullWidth
                value={selectedVariant}
                onChange={(e) => setSelectedVariant(e.target.value)}
              >
                <MenuItem value="Q8_0">Q8_0 (Recommended - High Quality)</MenuItem>
                <MenuItem value="Q4_K_M">Q4_K_M (Balanced)</MenuItem>
                <MenuItem value="Q4_0">Q4_0 (Smaller, Faster)</MenuItem>
              </Select>
              <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                Higher quantization = better quality but larger size
              </FormHelperText>
            </Box>

            {selectedModelForDownload && (
              <Alert severity="info" sx={{ mt: 2 }}>
                <Typography variant="body2">
                  <strong>Repository:</strong> {EMBEDDING_MODEL_REPOS[selectedModelForDownload as keyof typeof EMBEDDING_MODEL_REPOS]?.repo}
                </Typography>
                <Typography variant="body2">
                  <strong>File:</strong> {EMBEDDING_MODEL_REPOS[selectedModelForDownload as keyof typeof EMBEDDING_MODEL_REPOS]?.files.find(f => f.includes(selectedVariant))}
                </Typography>
              </Alert>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDownloadDialogOpen(false)} disabled={downloadLoading}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={startModelDownload}
            disabled={downloadLoading || !selectedModelForDownload}
            startIcon={downloadLoading ? <CircularProgress size={20} /> : <DownloadIcon />}
          >
            {downloadLoading ? 'Starting Download...' : 'Download'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default EmbeddingDeployPage
