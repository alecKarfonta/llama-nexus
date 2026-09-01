import React, { useEffect, useState, useCallback, useMemo } from 'react'
import {
  Box,
  Grid,
  Typography,
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
  useMediaQuery,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  RestartAlt as RestartIcon,
  Save as SaveIcon,
  Science as TestIcon,
  CloudDownload as DownloadIcon,
  Delete as DeleteIcon,
  CheckCircle as CheckIcon,
  Cancel as CancelIcon,
  Speed as SpeedIcon,
  Refresh as RefreshIcon,
  Assessment as AssessmentIcon,
  History as HistoryIcon,
  Hub as EmbeddingIcon,
  Inventory2 as InventoryIcon,
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
} from 'recharts'
import {
  ConfirmationDialog,
  compactCollapsibleCardSx,
  MetricTile,
  optionalSectionGridSx,
  PageShell,
  SectionCard,
} from '@/components/common'
import { useConfirmableAction } from '@/hooks/useConfirmableAction'
import { debugLog } from '@/utils/debug'
import { accentColors } from '@/utils/theme'

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

const formatBytes = (bytes?: number) => {
  if (!bytes || bytes <= 0) return 'Unknown size'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  return `${(bytes / Math.pow(1024, index)).toFixed(index === 0 ? 0 : 1)} ${units[index]}`
}

const localModelPath = (model: any) => model.localPath || model.path || model.filename || model.name || ''
const localModelName = (model: any) => model.filename || model.name || 'Unknown model'
const localModelSource = (model: any) => model.repositoryId || model.repository_id || localModelPath(model)
const localModelModifiedLabel = (model: any) => {
  const modified = model.lastModified || model.modified
  return modified ? new Date(modified).toLocaleDateString() : 'Unknown'
}
const embeddingModelNeedles = [
  ...Object.keys(EMBEDDING_MODEL_REPOS),
  ...Object.values(EMBEDDING_MODEL_REPOS).flatMap((model) => model.files),
].map((value) => value.toLowerCase())

const isEmbeddingLocalModel = (model: any) => {
  const searchable = [
    model.filename,
    model.name,
    model.path,
    model.localPath,
    model.repositoryId,
    model.repository_id,
  ].filter(Boolean).join(' ').toLowerCase()

  return embeddingModelNeedles.some((needle) => searchable.includes(needle))
}

export const EmbeddingDeployPage: React.FC = () => {
  const isNarrow = useMediaQuery('(max-width:599.95px)')
  const isShortViewport = useMediaQuery('(max-height:799.95px)')

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
  const {
    confirmationAction: confirmAction,
    confirmationLoading: confirmingAction,
    requestConfirmation,
    closeConfirmation: closeConfirmDialog,
    confirmAction: handleConfirmAction,
  } = useConfirmableAction()
  
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
        debugLog('embedding', 'Failed to fetch embedding service info', e)
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
      debugLog('embedding', 'Failed to load local models', e)
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
      debugLog('embedding', 'Failed to load downloads', e)
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
  
  const deleteLocalModel = (filePath: string) => {
    if (!filePath) return
    requestConfirmation({
      title: 'Delete local model file?',
      message: `Delete local model file "${filePath}"?`,
      confirmLabel: 'Delete file',
      destructive: true,
      onConfirm: async () => {
        try {
          const res = await fetch(`/v1/models/local-files?file_path=${encodeURIComponent(filePath)}`, {
            method: 'DELETE'
          })

          if (res.ok) {
            setEmbeddingSuccess('Model deleted successfully')
            await loadLocalModels()
            setTimeout(() => setEmbeddingSuccess(null), 3000)
          }
        } catch (e) {
          setEmbeddingError('Failed to delete model')
        }
      },
    })
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
      debugLog('embedding', 'Failed to load benchmark history', e)
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
      debugLog('embedding', 'Failed to delete benchmark result', e)
    }
  }

  const serviceRunning = Boolean(embeddingStatus?.running)
  const serviceStatusColor = serviceRunning ? accentColors.success : accentColors.warning
  const serviceStatusLabel = serviceRunning ? 'Service running' : 'Service stopped'
  const showStartAction = !serviceRunning || embeddingLoading === 'start'
  const showRestartAction = serviceRunning || embeddingLoading === 'restart'
  const showStopAction = serviceRunning || embeddingLoading === 'stop'
  const selectedModel = embeddingConfig?.model?.name || 'No model selected'
  const selectedVariantLabel = embeddingConfig?.model?.variant || 'Q8_0'
  const embeddingLocalModels = useMemo(() => localModels.filter(isEmbeddingLocalModel), [localModels])
  const totalLocalModelSize = embeddingLocalModels.reduce((total, model) => total + (model.size || model.size_bytes || 0), 0)
  const ragModeLabel = ragEmbeddingConfig?.use_deployed_service ? 'Deployed service' : 'Local fallback'
  const endpointLabel = embeddingStatus?.endpoint || ragEmbeddingConfig?.service_url || 'Endpoint pending'
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
  }
  const embeddingHeaderActions = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
      <Tooltip title="Refresh service status">
        <IconButton
          size="small"
          aria-label="Refresh embedding service status"
          onClick={async () => {
            const res = await fetch('/api/v1/embedding/status')
            if (res.ok) {
              setEmbeddingStatus(await res.json())
            }
            await Promise.all([loadLocalModels(), loadActiveDownloads()])
          }}
          sx={commandIconButtonSx}
        >
          <RefreshIcon sx={{ fontSize: 17 }} />
        </IconButton>
      </Tooltip>
      <Tooltip title="Save settings">
        <span>
          <IconButton
            size="small"
            aria-label="Save embedding settings"
            onClick={saveEmbeddingConfig}
            disabled={!embeddingConfig}
            sx={commandIconButtonSx}
          >
            <SaveIcon sx={{ fontSize: 17 }} />
          </IconButton>
        </span>
      </Tooltip>
      {showStartAction && (
        <Tooltip title="Start service">
          <span>
            <IconButton
              size="small"
              aria-label="Start embedding service"
              onClick={() => runEmbeddingAction('start')}
              disabled={embeddingLoading !== null || serviceRunning}
              sx={{
                width: 32,
                height: 32,
                borderRadius: '8px',
                bgcolor: 'primary.main',
                color: 'primary.contrastText',
                '&:hover': { bgcolor: 'primary.dark' },
                '&.Mui-disabled': {
                  bgcolor: 'action.disabledBackground',
                  color: 'action.disabled',
                },
              }}
            >
              {embeddingLoading === 'start' ? <CircularProgress size={15} color="inherit" /> : <StartIcon sx={{ fontSize: 17 }} />}
            </IconButton>
          </span>
        </Tooltip>
      )}
      {showRestartAction && (
        <Tooltip title="Restart service">
          <span>
            <IconButton
              size="small"
              aria-label="Restart embedding service"
              onClick={() => runEmbeddingAction('restart')}
              disabled={embeddingLoading !== null}
              sx={commandIconButtonSx}
            >
              {embeddingLoading === 'restart' ? <CircularProgress size={15} /> : <RestartIcon sx={{ fontSize: 17 }} />}
            </IconButton>
          </span>
        </Tooltip>
      )}
      {showStopAction && (
        <Tooltip title="Stop service">
          <span>
            <IconButton
              size="small"
              aria-label="Stop embedding service"
              color="error"
              onClick={() => runEmbeddingAction('stop')}
              disabled={embeddingLoading !== null || !serviceRunning}
              sx={commandIconButtonSx}
            >
              {embeddingLoading === 'stop' ? <CircularProgress size={15} /> : <StopIcon sx={{ fontSize: 17 }} />}
            </IconButton>
          </span>
        </Tooltip>
      )}
    </Box>
  )
  const embeddingSummaryRow = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
      <Chip
        size="small"
        label={serviceStatusLabel}
        sx={{
          height: 22,
          fontSize: '0.68rem',
          fontWeight: 800,
          color: serviceStatusColor,
          bgcolor: alpha(serviceStatusColor, 0.1),
          border: `1px solid ${alpha(serviceStatusColor, 0.28)}`,
        }}
      />
      <Chip
        size="small"
        label={`${selectedVariantLabel}`}
        sx={{ height: 22, fontSize: '0.68rem', fontWeight: 800, display: { xs: 'none', sm: 'inline-flex' } }}
      />
      <Chip
        size="small"
        label={ragModeLabel}
        sx={{ height: 22, fontSize: '0.68rem', fontWeight: 800, display: { xs: 'none', sm: 'inline-flex' } }}
      />
      <Chip
        size="small"
        label={`${embeddingLocalModels.length} local / ${activeDownloads.length} active`}
        sx={{ height: 22, fontSize: '0.68rem', fontWeight: 800, display: { xs: activeDownloads.length > 0 ? 'inline-flex' : 'none', sm: 'inline-flex' } }}
      />
      <Typography variant="caption" color="text.secondary" noWrap sx={{ minWidth: 0, display: { xs: 'none', sm: 'block' } }}>
        {selectedModel} / {endpointLabel}
      </Typography>
    </Box>
  )
  const embeddingRuntimeSettingItems = (
    <>
      <Grid item xs={6} sm={6} lg={4}>
        <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
          Context Size
        </Typography>
        <TextField
          fullWidth
          size="small"
          type="number"
          value={embeddingConfig?.model?.context_size || 8192}
          onChange={(e) => updateEmbeddingConfig('model.context_size', parseInt(e.target.value))}
          inputProps={{ min: 512, max: 32768 }}
        />
      </Grid>

      <Grid item xs={6} sm={6} lg={4}>
        <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
          GPU Layers
        </Typography>
        <TextField
          fullWidth
          size="small"
          type="number"
          value={embeddingConfig?.model?.gpu_layers || 999}
          onChange={(e) => updateEmbeddingConfig('model.gpu_layers', parseInt(e.target.value))}
          inputProps={{ min: 0, max: 999 }}
        />
      </Grid>

      <Grid item xs={6} sm={6} lg={4}>
        <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
          Execution Mode
        </Typography>
        <Select
          fullWidth
          size="small"
          value={embeddingConfig?.execution?.mode || 'gpu'}
          onChange={(e) => updateEmbeddingConfig('execution.mode', e.target.value)}
          sx={{ fontSize: '0.875rem' }}
        >
          <MenuItem value="gpu">GPU Acceleration</MenuItem>
          <MenuItem value="cpu">CPU Only</MenuItem>
        </Select>
      </Grid>

      <Grid item xs={6} sm={6} lg={4}>
        <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
          CUDA Devices
        </Typography>
        <TextField
          fullWidth
          size="small"
          value={embeddingConfig?.execution?.cuda_devices || 'all'}
          onChange={(e) => updateEmbeddingConfig('execution.cuda_devices', e.target.value)}
          disabled={embeddingConfig?.execution?.mode === 'cpu'}
        />
      </Grid>
    </>
  )
  const embeddingRuntimeSettings = (
    <Grid container spacing={1.25}>
      {embeddingRuntimeSettingItems}
    </Grid>
  )

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    )
  }

  return (
    <PageShell sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      {embeddingError && (
        <Alert severity="error" onClose={() => setEmbeddingError(null)}>
          {embeddingError}
        </Alert>
      )}
      
      {embeddingSuccess && (
        <Alert severity="success" onClose={() => setEmbeddingSuccess(null)}>
          {embeddingSuccess}
        </Alert>
      )}

      {embeddingConfig && (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: {
                xs: '1fr',
                lg: 'minmax(0, 0.95fr) minmax(0, 1.05fr)',
              },
              gap: 1.25,
              alignItems: 'start',
            }}
          >
            <SectionCard
              title="Model settings"
              icon={<EmbeddingIcon />}
              accentColor={accentColors.info}
              action={embeddingHeaderActions}
            >
              {embeddingSummaryRow}
              <Grid container spacing={1.25}>
                <Grid item xs={12} sm={6} lg={4}>
                  <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                    Embedding Model
                  </Typography>
                  <Select
                    fullWidth
                    size="small"
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
                </Grid>
                
                <Grid item xs={12} sm={6} lg={4}>
                  <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                    Variant
                  </Typography>
                  <Select
                    fullWidth
                    size="small"
                    value={embeddingConfig?.model?.variant || 'Q8_0'}
                    onChange={(e) => updateEmbeddingConfig('model.variant', e.target.value)}
                    sx={{ fontSize: '0.875rem' }}
                  >
                    <MenuItem value="Q8_0">Q8_0 (Recommended - High Quality)</MenuItem>
                    <MenuItem value="Q4_K_M">Q4_K_M (Balanced)</MenuItem>
                    <MenuItem value="Q4_0">Q4_0 (Smaller, Faster)</MenuItem>
                  </Select>
                </Grid>
                
                {!isNarrow && embeddingRuntimeSettingItems}
              </Grid>

              {isNarrow && (
                <Box
                  component="details"
                  sx={{
                    mt: 1,
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: '8px',
                    bgcolor: 'action.hover',
                    '& summary': { listStyle: 'none' },
                    '& summary::-webkit-details-marker': { display: 'none' },
                  }}
                >
                  <Box
                    component="summary"
                    sx={{
                      cursor: 'pointer',
                      px: 1,
                      py: 0.875,
                      fontSize: '0.8125rem',
                      fontWeight: 700,
                      color: 'text.secondary',
                    }}
                  >
                    Runtime settings
                  </Box>
                  <Box sx={{ px: 1, pb: 1 }}>
                    {embeddingRuntimeSettings}
                  </Box>
                </Box>
              )}
              
            </SectionCard>

            <SectionCard
              title="Model files"
              icon={<InventoryIcon />}
              accentColor={accentColors.primary}
              collapsible
              defaultExpanded={!isNarrow}
              sx={compactCollapsibleCardSx}
              action={
                <Tooltip title="Download model">
                  <IconButton
                    aria-label="Download embedding model"
                    onClick={() => setDownloadDialogOpen(true)}
                    sx={commandIconButtonSx}
                  >
                    <DownloadIcon sx={{ fontSize: 17 }} />
                  </IconButton>
                </Tooltip>
              }
            >
              <Grid container spacing={1}>
                {availableEmbeddingModels.map((model) => {
                  const modelStatus = getModelStatus(model.name, embeddingConfig?.model?.variant || 'Q8_0')
                  const isCurrentModel = embeddingConfig?.model?.name === model.name
                  
                  return (
                    <Grid item xs={12} md={6} key={model.name}>
                      <Box sx={{
                        p: 1.25,
                        borderRadius: '8px',
                        border: '1px solid',
                        bgcolor: isCurrentModel ? alpha(accentColors.info, 0.08) : 'action.hover',
                        borderColor: isCurrentModel ? alpha(accentColors.info, 0.38) : 'divider'
                      }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1 }}>
                          <Box sx={{ minWidth: 0 }}>
                            <Typography variant="body2" sx={{ fontWeight: 600 }} noWrap title={model.name}>
                              {model.name}
                            </Typography>
                            <Typography variant="caption" color="text.secondary" noWrap title={model.description}>
                              {model.dimensions}D / {model.max_tokens} tokens
                            </Typography>
                          </Box>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexShrink: 0 }}>
                            {isCurrentModel && (
                              <Chip label="Current" size="small" color="primary" />
                            )}
                            <Chip 
                              label={
                                modelStatus === 'downloaded' ? 'Downloaded' :
                                modelStatus === 'downloading' ? 'Downloading' :
                                'Missing'
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
                        {modelStatus === 'downloading' && (
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                            <LinearProgress sx={{ flex: 1, height: 4 }} />
                            <Typography variant="caption">
                              {activeDownloads.find(d => d.filename?.includes(model.name))?.progress || 0}%
                            </Typography>
                          </Box>
                        )}
                      </Box>
                    </Grid>
                  )
                })}
              </Grid>

              {activeDownloads.length > 0 && (
                <Box sx={{ mt: 1.5 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Active downloads
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
                              <Tooltip title="Cancel download">
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

              {embeddingLocalModels.length > 0 && (!isShortViewport || isNarrow) && (
                <Box sx={{ mt: 1.5 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Box>
                      <Typography variant="subtitle2">
                        Local inventory
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {embeddingLocalModels.length} files / {formatBytes(totalLocalModelSize)}
                      </Typography>
                    </Box>
                    <Tooltip title="Refresh local models">
                      <IconButton size="small" aria-label="Refresh local embedding models" onClick={loadLocalModels} sx={commandIconButtonSx}>
                        <RefreshIcon sx={{ fontSize: 17 }} />
                      </IconButton>
                    </Tooltip>
                  </Box>
                  <TableContainer
                    component={Paper}
                    variant="outlined"
                    sx={{ display: { xs: 'none', sm: 'block' } }}
                  >
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Model</TableCell>
                          <TableCell>Variant</TableCell>
                          <TableCell align="right">Size</TableCell>
                          <TableCell>Modified</TableCell>
                          <TableCell align="right">Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {embeddingLocalModels.map((model) => {
                          const filePath = localModelPath(model)
                          const modelSize = model.size || model.size_bytes
                          return (
                            <TableRow key={filePath || model.name || model.filename}>
                              <TableCell>
                                <Typography variant="body2" sx={{ fontWeight: 600 }} noWrap title={localModelName(model)}>
                                  {localModelName(model)}
                                </Typography>
                                <Typography variant="caption" color="text.secondary" noWrap title={localModelSource(model)}>
                                  {localModelSource(model)}
                                </Typography>
                              </TableCell>
                              <TableCell>
                                <Chip label={model.variant || model.extension || 'unknown'} size="small" variant="outlined" />
                              </TableCell>
                              <TableCell align="right">{formatBytes(modelSize)}</TableCell>
                              <TableCell>
                                <Typography variant="caption" color="text.secondary">
                                  {localModelModifiedLabel(model)}
                                </Typography>
                              </TableCell>
                              <TableCell align="right">
                                <Tooltip title="Delete local file">
                                  <span>
                                    <IconButton
                                      size="small"
                                      color="error"
                                      disabled={!filePath}
                                      onClick={() => deleteLocalModel(filePath)}
                                    >
                                      <DeleteIcon fontSize="small" />
                                    </IconButton>
                                  </span>
                                </Tooltip>
                              </TableCell>
                            </TableRow>
                          )
                        })}
                      </TableBody>
                    </Table>
                  </TableContainer>
                  <Box sx={{ display: { xs: 'grid', sm: 'none' }, gap: 0.75 }}>
                    {embeddingLocalModels.map((model) => {
                      const filePath = localModelPath(model)
                      const modelSize = model.size || model.size_bytes

                      return (
                        <Box
                          key={filePath || model.name || model.filename}
                          sx={{
                            p: 1,
                            borderRadius: '8px',
                            border: '1px solid',
                            borderColor: 'divider',
                            bgcolor: 'background.paper',
                            display: 'grid',
                            gridTemplateColumns: 'minmax(0, 1fr) auto',
                            gap: 1,
                            alignItems: 'center',
                          }}
                        >
                          <Box sx={{ minWidth: 0 }}>
                            <Typography variant="body2" sx={{ fontWeight: 700 }} noWrap title={localModelName(model)}>
                              {localModelName(model)}
                            </Typography>
                            <Typography variant="caption" color="text.secondary" noWrap title={localModelSource(model)}>
                              {localModelSource(model)}
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap', mt: 0.75 }}>
                              <Chip label={model.variant || model.extension || 'unknown'} size="small" variant="outlined" />
                              <Typography variant="caption" color="text.secondary">
                                {formatBytes(modelSize)}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {localModelModifiedLabel(model)}
                              </Typography>
                            </Box>
                          </Box>
                          <Tooltip title="Delete local file">
                            <span>
                              <IconButton
                                size="small"
                                color="error"
                                disabled={!filePath}
                                onClick={() => deleteLocalModel(filePath)}
                                sx={commandIconButtonSx}
                              >
                                <DeleteIcon fontSize="small" />
                              </IconButton>
                            </span>
                          </Tooltip>
                        </Box>
                      )
                    })}
                  </Box>
                </Box>
              )}
            </SectionCard>
          </Box>

          <Box sx={optionalSectionGridSx}>
            {ragEmbeddingConfig && (
              <SectionCard
                title="RAG integration"
                icon={<AssessmentIcon />}
                accentColor={accentColors.rose}
                collapsible
                defaultExpanded={false}
                sx={compactCollapsibleCardSx}
              >
                <Grid container spacing={1.5}>
                  <Grid item xs={12}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap' }}>
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
                        size="small"
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
                      size="small"
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
                      size="small"
                      value={ragEmbeddingConfig?.service_url || ''}
                      onChange={(e) => updateRagEmbeddingConfig('service_url', e.target.value)}
                    />
                    <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                      URL of the embedding service endpoint
                    </FormHelperText>
                  </Grid>
                </Grid>
                
                <Box sx={{ mt: 1.5, display: 'flex', gap: 1 }}>
                  <Button
                    variant="contained"
                    size="small"
                    startIcon={ragEmbeddingLoading ? <CircularProgress size={20} /> : <SaveIcon />}
                    onClick={saveRagEmbeddingConfig}
                    disabled={ragEmbeddingLoading}
                    sx={{ fontWeight: 600 }}
                  >
                    Save
                  </Button>
                </Box>
              </SectionCard>
            )}

            {embeddingStatus?.running && (
              <SectionCard
                title="Embedding test"
                icon={<TestIcon />}
                accentColor={accentColors.success}
                collapsible
                defaultExpanded={false}
                sx={compactCollapsibleCardSx}
              >
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
                      size="small"
                      startIcon={testLoading ? <CircularProgress size={20} /> : <TestIcon />}
                      onClick={testEmbedding}
                      disabled={testLoading || !testText.trim()}
                      sx={{ fontWeight: 600 }}
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
                          <MetricTile
                            label="Vector Dimensions"
                            value={testResult.vectorLength}
                            detail="embedding width"
                            icon={<EmbeddingIcon />}
                            color={accentColors.primary}
                          />
                        </Grid>

                        <Grid item xs={12} sm={6} md={3}>
                          <MetricTile
                            label="Processing Time"
                            value={`${testResult.timeTaken}ms`}
                            detail="round trip"
                            icon={<SpeedIcon />}
                            color={accentColors.success}
                          />
                        </Grid>

                        <Grid item xs={12} sm={6} md={3}>
                          <MetricTile
                            label="Model Used"
                            value={testResult.model || 'N/A'}
                            detail="service response"
                            icon={<EmbeddingIcon />}
                            color={accentColors.info}
                            valueSx={{ fontSize: '0.9rem' }}
                          />
                        </Grid>

                        <Grid item xs={12} sm={6} md={3}>
                          <MetricTile
                            label="Usage"
                            value={`${testResult.usage?.total_tokens || 'N/A'} tokens`}
                            detail="reported total"
                            icon={<AssessmentIcon />}
                            color={accentColors.purple}
                            valueSx={{ fontSize: '0.9rem' }}
                          />
                        </Grid>

                        <Grid item xs={12}>
                          <Typography variant="body2" gutterBottom sx={{ fontWeight: 500 }}>
                            Embedding Vector (first 10 dimensions):
                          </Typography>
	                          <Box sx={{
	                            p: 2,
	                            bgcolor: 'grey.900',
	                            borderRadius: '8px',
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
              </SectionCard>
            )}

            <SectionCard
              title="Benchmarks"
              icon={<SpeedIcon />}
              accentColor={accentColors.purple}
              collapsible
              defaultExpanded={false}
              sx={compactCollapsibleCardSx}
            >
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
                    size="small"
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
                            <CartesianGrid strokeDasharray="3 3" stroke={alpha(accentColors.slate, 0.24)} />
                            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                            <YAxis yAxisId="left" label={{ value: 'texts/sec', angle: -90, position: 'insideLeft' }} />
                            <YAxis yAxisId="right" orientation="right" label={{ value: 'ms/text', angle: 90, position: 'insideRight' }} />
                            <RechartsTooltip />
                            <Legend />
                            <Bar yAxisId="left" dataKey="textsPerSec" name="Texts/sec" fill={accentColors.info} radius={[4, 4, 0, 0]} />
                            <Bar yAxisId="right" dataKey="perTextMs" name="ms/text" fill={accentColors.warning} radius={[4, 4, 0, 0]} />
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
                            <CartesianGrid strokeDasharray="3 3" stroke={alpha(accentColors.slate, 0.24)} />
                            <XAxis dataKey="name" label={{ value: 'Collection Size', position: 'insideBottom', offset: -5 }} />
                            <YAxis yAxisId="left" label={{ value: 'docs/sec', angle: -90, position: 'insideLeft' }} />
                            <YAxis yAxisId="right" orientation="right" label={{ value: 'ms', angle: 90, position: 'insideRight' }} />
                            <RechartsTooltip />
                            <Legend />
                            <Bar yAxisId="left" dataKey="docsPerSec" name="Docs/sec" fill={accentColors.success} radius={[4, 4, 0, 0]} />
                            <Bar yAxisId="right" dataKey="embedMs" name="Embed (ms)" fill={accentColors.info} radius={[4, 4, 0, 0]} />
                            <Bar yAxisId="right" dataKey="upsertMs" name="Upsert (ms)" fill={accentColors.warning} radius={[4, 4, 0, 0]} />
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
	                      Object.values(data.search_results || {}).forEach((sd: any) => {
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
                            <CartesianGrid strokeDasharray="3 3" stroke={alpha(accentColors.slate, 0.24)} />
                            <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-25} textAnchor="end" height={60} />
                            <YAxis label={{ value: 'ms', angle: -90, position: 'insideLeft' }} />
                            <RechartsTooltip />
                            <Legend />
                            <Bar dataKey="p50" name="p50" fill={accentColors.info} radius={[2, 2, 0, 0]} />
                            <Bar dataKey="p95" name="p95" fill={accentColors.warning} radius={[2, 2, 0, 0]} />
                            <Bar dataKey="p99" name="p99" fill={accentColors.rose} radius={[2, 2, 0, 0]} />
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
                            <CartesianGrid strokeDasharray="3 3" stroke={alpha(accentColors.slate, 0.24)} />
                            <XAxis dataKey="name" label={{ value: 'Collection Size', position: 'insideBottom', offset: -5 }} />
                            <YAxis label={{ value: 'ms', angle: -90, position: 'insideLeft' }} />
                            <RechartsTooltip />
                            <Legend />
                            <Bar dataKey="Insert" stackId="a" fill={accentColors.info} />
                            <Bar dataKey="Search" stackId="a" fill={accentColors.success} />
                            <Bar dataKey="Update" stackId="a" fill={accentColors.warning} />
                            <Bar dataKey="Delete" stackId="a" fill={accentColors.rose} radius={[4, 4, 0, 0]} />
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
                                    } catch (e) { debugLog('embedding', 'Failed to load benchmark result', e) }
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
            </SectionCard>
          </Box>

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

      <ConfirmationDialog
        action={confirmAction}
        loading={confirmingAction}
        onClose={closeConfirmDialog}
        onConfirm={handleConfirmAction}
      />
    </PageShell>
  )
}

export default EmbeddingDeployPage
