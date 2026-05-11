import React, { useEffect, useState, useCallback } from 'react'
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
  Tabs,
  Tab,
  Divider,
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
  Speed as SpeedIcon,
  Refresh as RefreshIcon,
  Assessment as AssessmentIcon,
  History as HistoryIcon,
} from '@mui/icons-material'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Legend,
  LineChart,
  Line,
} from 'recharts'

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

  // Benchmark state
  const [benchType, setBenchType] = useState<string>('embedding_throughput')
  const [benchRunning, setBenchRunning] = useState(false)
  const [benchResult, setBenchResult] = useState<any>(null)
  const [benchError, setBenchError] = useState<string | null>(null)
  const [benchHistory, setBenchHistory] = useState<any[]>([])
  const [benchHistoryLoading, setBenchHistoryLoading] = useState(false)
  const [benchTab, setBenchTab] = useState(0)
  const [benchConfig, setBenchConfig] = useState({
    collection_sizes: '100,1000,5000',
    text_lengths: '50,200,500',
    batch_sizes: '1,8,32,64',
    search_top_k: '1,5,10,50',
    search_iterations: 25,
    runs: 1,
  })

  const BENCHMARK_TYPES = [
    { id: 'embedding_throughput', name: 'Embedding Throughput', description: 'Raw embedding speed across batch sizes and text lengths' },
    { id: 'insertion', name: 'Insertion', description: 'Embed + upsert documents at various collection sizes' },
    { id: 'search', name: 'Search', description: 'Vector search latency at various collection sizes and top_k' },
    { id: 'update', name: 'Update', description: 'Re-embed + re-upsert latency for existing documents' },
    { id: 'deletion', name: 'Deletion', description: 'Single and batch delete latency' },
    { id: 'full_pipeline', name: 'Full Pipeline', description: 'Complete lifecycle: insert, search, update, delete' },
  ]

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

  // Benchmark functions
  const loadBenchHistory = useCallback(async () => {
    try {
      setBenchHistoryLoading(true)
      const res = await fetch('/api/v1/embedding-benchmark/results?limit=20')
      if (res.ok) {
        const data = await res.json()
        setBenchHistory(data.results || [])
      }
    } catch (e) {
      console.error('Failed to load benchmark history:', e)
    } finally {
      setBenchHistoryLoading(false)
    }
  }, [])

  useEffect(() => {
    loadBenchHistory()
  }, [loadBenchHistory])

  const runBenchmark = async () => {
    try {
      setBenchRunning(true)
      setBenchError(null)
      setBenchResult(null)

      const parseNumList = (s: string) => s.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v))

      const payload: any = {
        type: benchType,
        name: `${BENCHMARK_TYPES.find(t => t.id === benchType)?.name} - ${new Date().toLocaleString()}`,
        runs: benchConfig.runs,
        search_iterations: benchConfig.search_iterations,
      }

      if (['insertion', 'search', 'update', 'deletion', 'full_pipeline'].includes(benchType)) {
        payload.collection_sizes = parseNumList(benchConfig.collection_sizes)
      }
      if (['embedding_throughput'].includes(benchType)) {
        payload.text_lengths = parseNumList(benchConfig.text_lengths)
        payload.batch_sizes = parseNumList(benchConfig.batch_sizes)
      }
      if (['search'].includes(benchType)) {
        payload.search_top_k = parseNumList(benchConfig.search_top_k)
      }

      const res = await fetch('/api/v1/embedding-benchmark/run-sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Benchmark failed' }))
        throw new Error(err.detail || `Benchmark failed (${res.status})`)
      }

      const data = await res.json()
      setBenchResult(data)
      await loadBenchHistory()
    } catch (e) {
      setBenchError(e instanceof Error ? e.message : 'Benchmark failed')
    } finally {
      setBenchRunning(false)
    }
  }

  const deleteBenchResult = async (id: string) => {
    try {
      await fetch(`/api/v1/embedding-benchmark/results/${id}`, { method: 'DELETE' })
      await loadBenchHistory()
    } catch (e) {
      console.error('Failed to delete benchmark result:', e)
    }
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

          {/* Performance Benchmark */}
          <Card sx={{
            mb: 3,
            borderRadius: 2,
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
            border: '1px solid',
            borderColor: 'divider',
          }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <SpeedIcon color="primary" />
                <Typography variant="h6" sx={{ fontSize: '1.125rem', fontWeight: 600 }}>
                  Performance Benchmark
                </Typography>
              </Box>

              {/* Config Row */}
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} md={4}>
                  <Typography variant="body2" sx={{ mb: 1, fontWeight: 500 }}>Benchmark Type</Typography>
                  <Select
                    fullWidth
                    size="small"
                    value={benchType}
                    onChange={(e) => setBenchType(e.target.value)}
                  >
                    {BENCHMARK_TYPES.map(t => (
                      <MenuItem key={t.id} value={t.id}>{t.name}</MenuItem>
                    ))}
                  </Select>
                  <FormHelperText>{BENCHMARK_TYPES.find(t => t.id === benchType)?.description}</FormHelperText>
                </Grid>

                {['insertion', 'search', 'update', 'deletion', 'full_pipeline'].includes(benchType) && (
                  <Grid item xs={12} md={4}>
                    <Typography variant="body2" sx={{ mb: 1, fontWeight: 500 }}>Collection Sizes</Typography>
                    <TextField
                      fullWidth
                      size="small"
                      value={benchConfig.collection_sizes}
                      onChange={(e) => setBenchConfig({ ...benchConfig, collection_sizes: e.target.value })}
                      placeholder="100, 1000, 5000"
                      helperText="Comma-separated list"
                    />
                  </Grid>
                )}

                {benchType === 'embedding_throughput' && (
                  <>
                    <Grid item xs={6} md={3}>
                      <Typography variant="body2" sx={{ mb: 1, fontWeight: 500 }}>Text Lengths (words)</Typography>
                      <TextField
                        fullWidth
                        size="small"
                        value={benchConfig.text_lengths}
                        onChange={(e) => setBenchConfig({ ...benchConfig, text_lengths: e.target.value })}
                        placeholder="50, 200, 500"
                      />
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Typography variant="body2" sx={{ mb: 1, fontWeight: 500 }}>Batch Sizes</Typography>
                      <TextField
                        fullWidth
                        size="small"
                        value={benchConfig.batch_sizes}
                        onChange={(e) => setBenchConfig({ ...benchConfig, batch_sizes: e.target.value })}
                        placeholder="1, 8, 32, 64"
                      />
                    </Grid>
                  </>
                )}

                {benchType === 'search' && (
                  <Grid item xs={6} md={2}>
                    <Typography variant="body2" sx={{ mb: 1, fontWeight: 500 }}>top_k Values</Typography>
                    <TextField
                      fullWidth
                      size="small"
                      value={benchConfig.search_top_k}
                      onChange={(e) => setBenchConfig({ ...benchConfig, search_top_k: e.target.value })}
                      placeholder="1, 5, 10, 50"
                    />
                  </Grid>
                )}

                <Grid item xs={6} md={2}>
                  <Typography variant="body2" sx={{ mb: 1, fontWeight: 500 }}>Runs</Typography>
                  <TextField
                    fullWidth
                    size="small"
                    type="number"
                    value={benchConfig.runs}
                    onChange={(e) => setBenchConfig({ ...benchConfig, runs: parseInt(e.target.value) || 1 })}
                    inputProps={{ min: 1, max: 10 }}
                  />
                </Grid>

                <Grid item xs={12} md={2} sx={{ display: 'flex', alignItems: 'flex-end', pb: 1.5 }}>
                  <Button
                    variant="contained"
                    fullWidth
                    onClick={runBenchmark}
                    disabled={benchRunning || !embeddingStatus?.running}
                    startIcon={benchRunning ? <CircularProgress size={18} /> : <StartIcon />}
                    sx={{ minHeight: 40 }}
                  >
                    {benchRunning ? 'Running...' : 'Run'}
                  </Button>
                </Grid>
              </Grid>

              {!embeddingStatus?.running && (
                <Alert severity="warning" sx={{ mb: 2 }}>
                  Start the embedding service before running benchmarks.
                </Alert>
              )}

              {benchError && (
                <Alert severity="error" sx={{ mb: 2 }} onClose={() => setBenchError(null)}>
                  {benchError}
                </Alert>
              )}

              {benchRunning && (
                <Box sx={{ mb: 2 }}>
                  <LinearProgress />
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
                    Running benchmark... This may take several minutes depending on collection sizes.
                  </Typography>
                </Box>
              )}

              {/* Results Tabs */}
              <Tabs value={benchTab} onChange={(_, v) => setBenchTab(v)} sx={{ mb: 2 }}>
                <Tab label="Latest Result" icon={<AssessmentIcon />} iconPosition="start" />
                <Tab label={`History (${benchHistory.length})`} icon={<HistoryIcon />} iconPosition="start" />
              </Tabs>

              {/* Latest Result Tab */}
              {benchTab === 0 && benchResult && (
                <Box>
                  {benchResult.type === 'embedding_throughput' && (() => {
                    const br = benchResult.metrics?.batch_results || {}
                    const chartData: any[] = []
                    Object.entries(br).forEach(([tlen, batches]: [string, any]) => {
                      Object.entries(batches).forEach(([bsize, data]: [string, any]) => {
                        chartData.push({
                          name: `${tlen}w/b${bsize}`,
                          textsPerSec: data.texts_per_second,
                          perTextMs: data.per_text_ms?.mean || 0,
                          textLen: tlen,
                          batchSize: bsize,
                        })
                      })
                    })
                    return (
                      <Box>
                        <Typography variant="subtitle2" sx={{ mb: 1 }}>
                          Throughput by Text Length / Batch Size
                        </Typography>
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                            <YAxis yAxisId="left" label={{ value: 'texts/sec', angle: -90, position: 'insideLeft' }} />
                            <YAxis yAxisId="right" orientation="right" label={{ value: 'ms/text', angle: 90, position: 'insideRight' }} />
                            <RechartsTooltip />
                            <Legend />
                            <Bar yAxisId="left" dataKey="textsPerSec" name="Texts/sec" fill="#4fc3f7" radius={[4, 4, 0, 0]} />
                            <Bar yAxisId="right" dataKey="perTextMs" name="ms/text" fill="#ffb74d" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                        <TableContainer component={Paper} sx={{ mt: 2 }} variant="outlined">
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>Text Length</TableCell>
                                <TableCell>Batch Size</TableCell>
                                <TableCell align="right">Texts/sec</TableCell>
                                <TableCell align="right">ms/text</TableCell>
                                <TableCell align="right">Total ms (mean)</TableCell>
                                <TableCell align="right">Tokens</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {Object.entries(br).flatMap(([tlen, batches]: [string, any]) =>
                                Object.entries(batches).map(([bsize, data]: [string, any]) => (
                                  <TableRow key={`${tlen}-${bsize}`}>
                                    <TableCell>{tlen} words</TableCell>
                                    <TableCell>{data.batch_size}</TableCell>
                                    <TableCell align="right">{data.texts_per_second?.toFixed(1)}</TableCell>
                                    <TableCell align="right">{data.per_text_ms?.mean?.toFixed(2)}</TableCell>
                                    <TableCell align="right">{data.total_latency_ms?.mean?.toFixed(1)}</TableCell>
                                    <TableCell align="right">{data.token_count}</TableCell>
                                  </TableRow>
                                ))
                              )}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </Box>
                    )
                  })()}

                  {benchResult.type === 'insertion' && (() => {
                    const ir = benchResult.metrics?.insertion_results || {}
                    const chartData = Object.entries(ir).map(([size, data]: [string, any]) => ({
                      name: `${parseInt(size).toLocaleString()}`,
                      docsPerSec: data.docs_per_second,
                      embedMs: data.embedding_time_ms?.mean || 0,
                      upsertMs: data.upsert_time_ms?.mean || 0,
                      msPerDoc: data.avg_ms_per_doc,
                    }))
                    return (
                      <Box>
                        <Typography variant="subtitle2" sx={{ mb: 1 }}>Insertion Performance by Collection Size</Typography>
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis dataKey="name" label={{ value: 'Collection Size', position: 'insideBottom', offset: -5 }} />
                            <YAxis yAxisId="left" label={{ value: 'docs/sec', angle: -90, position: 'insideLeft' }} />
                            <YAxis yAxisId="right" orientation="right" label={{ value: 'ms', angle: 90, position: 'insideRight' }} />
                            <RechartsTooltip />
                            <Legend />
                            <Bar yAxisId="left" dataKey="docsPerSec" name="Docs/sec" fill="#81c784" radius={[4, 4, 0, 0]} />
                            <Bar yAxisId="right" dataKey="embedMs" name="Embed (ms)" fill="#4fc3f7" radius={[4, 4, 0, 0]} />
                            <Bar yAxisId="right" dataKey="upsertMs" name="Upsert (ms)" fill="#ffb74d" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                        <TableContainer component={Paper} sx={{ mt: 2 }} variant="outlined">
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>Size</TableCell>
                                <TableCell align="right">Total (ms)</TableCell>
                                <TableCell align="right">Embed (ms)</TableCell>
                                <TableCell align="right">Upsert (ms)</TableCell>
                                <TableCell align="right">Docs/sec</TableCell>
                                <TableCell align="right">ms/doc</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {Object.entries(ir).map(([size, data]: [string, any]) => (
                                <TableRow key={size}>
                                  <TableCell>{parseInt(size).toLocaleString()}</TableCell>
                                  <TableCell align="right">{data.total_time_ms?.toFixed(0)}</TableCell>
                                  <TableCell align="right">{data.embedding_time_ms?.mean?.toFixed(0)}</TableCell>
                                  <TableCell align="right">{data.upsert_time_ms?.mean?.toFixed(0)}</TableCell>
                                  <TableCell align="right">{data.docs_per_second?.toFixed(1)}</TableCell>
                                  <TableCell align="right">{data.avg_ms_per_doc?.toFixed(2)}</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </Box>
                    )
                  })()}

                  {benchResult.type === 'search' && (() => {
                    const sr = benchResult.metrics?.search_results || {}
                    const chartData: any[] = []
                    Object.entries(sr).forEach(([size, data]: [string, any]) => {
                      Object.entries(data.search_results || {}).forEach(([tk, sd]: [string, any]) => {
                        chartData.push({
                          name: `${parseInt(size).toLocaleString()} / k=${sd.top_k}`,
                          p50: sd.search_only_ms?.p50,
                          p95: sd.search_only_ms?.p95,
                          p99: sd.search_only_ms?.p99,
                          qps: sd.queries_per_second,
                          size: parseInt(size),
                          topK: sd.top_k,
                        })
                      })
                    })
                    return (
                      <Box>
                        <Typography variant="subtitle2" sx={{ mb: 1 }}>Search Latency (ms) by Collection Size / top_k</Typography>
                        <ResponsiveContainer width="100%" height={350}>
                          <BarChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-25} textAnchor="end" height={60} />
                            <YAxis label={{ value: 'ms', angle: -90, position: 'insideLeft' }} />
                            <RechartsTooltip />
                            <Legend />
                            <Bar dataKey="p50" name="p50" fill="#4fc3f7" radius={[2, 2, 0, 0]} />
                            <Bar dataKey="p95" name="p95" fill="#ffb74d" radius={[2, 2, 0, 0]} />
                            <Bar dataKey="p99" name="p99" fill="#ef5350" radius={[2, 2, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                        <TableContainer component={Paper} sx={{ mt: 2 }} variant="outlined">
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>Collection</TableCell>
                                <TableCell>top_k</TableCell>
                                <TableCell align="right">p50 (ms)</TableCell>
                                <TableCell align="right">p95 (ms)</TableCell>
                                <TableCell align="right">p99 (ms)</TableCell>
                                <TableCell align="right">Mean (ms)</TableCell>
                                <TableCell align="right">QPS</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {Object.entries(sr).flatMap(([size, data]: [string, any]) =>
                                Object.entries(data.search_results || {}).map(([tk, sd]: [string, any]) => (
                                  <TableRow key={`${size}-${tk}`}>
                                    <TableCell>{parseInt(size).toLocaleString()}</TableCell>
                                    <TableCell>{sd.top_k}</TableCell>
                                    <TableCell align="right">{sd.search_only_ms?.p50?.toFixed(2)}</TableCell>
                                    <TableCell align="right">{sd.search_only_ms?.p95?.toFixed(2)}</TableCell>
                                    <TableCell align="right">{sd.search_only_ms?.p99?.toFixed(2)}</TableCell>
                                    <TableCell align="right">{sd.search_only_ms?.mean?.toFixed(2)}</TableCell>
                                    <TableCell align="right">{sd.queries_per_second?.toFixed(0)}</TableCell>
                                  </TableRow>
                                ))
                              )}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </Box>
                    )
                  })()}

                  {benchResult.type === 'full_pipeline' && (() => {
                    const pr = benchResult.metrics?.pipeline_results || {}
                    const chartData = Object.entries(pr).map(([size, data]: [string, any]) => ({
                      name: parseInt(size).toLocaleString(),
                      Insert: data.phases?.insert_ms,
                      Search: data.phases?.search_total_ms,
                      Update: data.phases?.update_ms,
                      Delete: data.phases?.delete_ms,
                    }))
                    return (
                      <Box>
                        <Typography variant="subtitle2" sx={{ mb: 1 }}>Pipeline Phase Breakdown (ms)</Typography>
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis dataKey="name" label={{ value: 'Collection Size', position: 'insideBottom', offset: -5 }} />
                            <YAxis label={{ value: 'ms', angle: -90, position: 'insideLeft' }} />
                            <RechartsTooltip />
                            <Legend />
                            <Bar dataKey="Insert" stackId="a" fill="#4fc3f7" />
                            <Bar dataKey="Search" stackId="a" fill="#81c784" />
                            <Bar dataKey="Update" stackId="a" fill="#ffb74d" />
                            <Bar dataKey="Delete" stackId="a" fill="#ef5350" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                        <TableContainer component={Paper} sx={{ mt: 2 }} variant="outlined">
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>Size</TableCell>
                                <TableCell align="right">Insert (ms)</TableCell>
                                <TableCell align="right">Search (ms)</TableCell>
                                <TableCell align="right">Update (ms)</TableCell>
                                <TableCell align="right">Delete (ms)</TableCell>
                                <TableCell align="right">Total (ms)</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {Object.entries(pr).map(([size, data]: [string, any]) => (
                                <TableRow key={size}>
                                  <TableCell>{parseInt(size).toLocaleString()}</TableCell>
                                  <TableCell align="right">{data.phases?.insert_ms?.toLocaleString()}</TableCell>
                                  <TableCell align="right">{data.phases?.search_total_ms?.toLocaleString()}</TableCell>
                                  <TableCell align="right">{data.phases?.update_ms?.toLocaleString()}</TableCell>
                                  <TableCell align="right">{data.phases?.delete_ms?.toLocaleString()}</TableCell>
                                  <TableCell align="right">{data.phases?.total_pipeline_ms?.toLocaleString()}</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </Box>
                    )
                  })()}

                  {['update', 'deletion'].includes(benchResult.type) && (() => {
                    const metrics = benchResult.metrics
                    const key = benchResult.type === 'update' ? 'update_results' : 'deletion_results'
                    const results = metrics?.[key] || {}
                    const modelInfo = metrics?.model_info
                    return (
                      <Box>
                        {modelInfo && (
                          <Alert severity="info" sx={{ mb: 2 }}>
                            Model: {modelInfo.name} ({modelInfo.dimensions}D, {modelInfo.provider})
                          </Alert>
                        )}
                        <TableContainer component={Paper} variant="outlined">
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>Collection Size</TableCell>
                                {benchResult.type === 'update' ? (
                                  <>
                                    <TableCell align="right">Total (ms)</TableCell>
                                    <TableCell align="right">Re-embed (ms)</TableCell>
                                    <TableCell align="right">Upsert (ms)</TableCell>
                                    <TableCell align="right">ms/update</TableCell>
                                    <TableCell align="right">Updates/sec</TableCell>
                                  </>
                                ) : (
                                  <>
                                    <TableCell align="right">Single p50 (ms)</TableCell>
                                    <TableCell align="right">Single p99 (ms)</TableCell>
                                    <TableCell align="right">Batch p50 (ms)</TableCell>
                                    <TableCell align="right">Single/sec</TableCell>
                                    <TableCell align="right">Batch/sec</TableCell>
                                  </>
                                )}
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {Object.entries(results).map(([size, data]: [string, any]) => (
                                <TableRow key={size}>
                                  <TableCell>{data.collection_size?.toLocaleString()}</TableCell>
                                  {benchResult.type === 'update' ? (
                                    <>
                                      <TableCell align="right">{data.total_time_ms?.toFixed(0)}</TableCell>
                                      <TableCell align="right">{data.re_embed_time_ms?.mean?.toFixed(0)}</TableCell>
                                      <TableCell align="right">{data.upsert_time_ms?.mean?.toFixed(0)}</TableCell>
                                      <TableCell align="right">{data.avg_ms_per_update?.toFixed(2)}</TableCell>
                                      <TableCell align="right">{data.updates_per_second?.toFixed(1)}</TableCell>
                                    </>
                                  ) : (
                                    <>
                                      <TableCell align="right">{data.single_delete_ms?.p50?.toFixed(2)}</TableCell>
                                      <TableCell align="right">{data.single_delete_ms?.p99?.toFixed(2)}</TableCell>
                                      <TableCell align="right">{data.batch_delete_ms?.p50?.toFixed(2)}</TableCell>
                                      <TableCell align="right">{data.deletes_per_second_single?.toFixed(0)}</TableCell>
                                      <TableCell align="right">{data.deletes_per_second_batch?.toFixed(0)}</TableCell>
                                    </>
                                  )}
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </Box>
                    )
                  })()}

                  {benchResult && !benchResult.metrics && (
                    <Typography color="text.secondary">No metrics returned.</Typography>
                  )}
                </Box>
              )}

              {benchTab === 0 && !benchResult && !benchRunning && (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <AssessmentIcon sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
                  <Typography color="text.secondary">
                    Select a benchmark type and click Run to measure performance.
                  </Typography>
                </Box>
              )}

              {/* History Tab */}
              {benchTab === 1 && (
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>
                    <Button size="small" startIcon={<RefreshIcon />} onClick={loadBenchHistory} disabled={benchHistoryLoading}>
                      Refresh
                    </Button>
                  </Box>
                  {benchHistoryLoading && <LinearProgress sx={{ mb: 1 }} />}
                  {benchHistory.length === 0 ? (
                    <Typography color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
                      No benchmark results yet. Run a benchmark to see results here.
                    </Typography>
                  ) : (
                    <TableContainer component={Paper} variant="outlined">
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>ID</TableCell>
                            <TableCell>Name</TableCell>
                            <TableCell>Type</TableCell>
                            <TableCell>Status</TableCell>
                            <TableCell>Created</TableCell>
                            <TableCell align="right">Actions</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {benchHistory.map((r) => (
                            <TableRow key={r.id}>
                              <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>{r.id}</TableCell>
                              <TableCell>{r.name || '-'}</TableCell>
                              <TableCell>
                                <Chip label={r.type} size="small" variant="outlined" />
                              </TableCell>
                              <TableCell>
                                <Chip
                                  label={r.status}
                                  size="small"
                                  color={r.status === 'completed' ? 'success' : r.status === 'failed' ? 'error' : 'warning'}
                                />
                              </TableCell>
                              <TableCell sx={{ fontSize: '0.8rem' }}>{r.created_at}</TableCell>
                              <TableCell align="right">
                                <Tooltip title="Load result">
                                  <IconButton size="small" onClick={async () => {
                                    try {
                                      const res = await fetch(`/api/v1/embedding-benchmark/results/${r.id}`)
                                      if (res.ok) {
                                        const data = await res.json()
                                        if (data.metrics) {
                                          setBenchResult({
                                            benchmark_id: data.id,
                                            type: data.type,
                                            metrics: JSON.parse(data.metrics),
                                          })
                                          setBenchTab(0)
                                        }
                                      }
                                    } catch (e) { console.error(e) }
                                  }}>
                                    <AssessmentIcon fontSize="small" />
                                  </IconButton>
                                </Tooltip>
                                <Tooltip title="Delete">
                                  <IconButton size="small" onClick={() => deleteBenchResult(r.id)}>
                                    <DeleteIcon fontSize="small" />
                                  </IconButton>
                                </Tooltip>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>

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
