import React, { useState, useEffect, useMemo, useRef } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import {
  Typography,
  Box,
  Grid,
  Button,
  Alert,
  Snackbar,
  CircularProgress,
  TextField,
  InputAdornment,
  LinearProgress,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Menu,
  MenuItem,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  useMediaQuery,
  alpha
} from '@mui/material'
import {
  Refresh as RefreshIcon,
  CloudDownload as CloudDownloadIcon,
  Search as SearchIcon,
  Stop as StopIcon,
  Info as InfoIcon,
  RocketLaunch as DeployIcon,
  Delete as DeleteIcon,
  FolderOpen as FolderIcon,
  Storage as StorageIcon,
  Archive as ArchiveIcon,
  Unarchive as UnarchiveIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon,
  Compress as QuantizeIcon,
  Speed as BenchmarkIcon,
  Tune as TuneIcon,
  MoreVert as MoreVertIcon,
  Close as CloseIcon,
  MergeType as GraftIcon,
} from '@mui/icons-material'
import { ModelInfo, ModelDownload, ModelArchiveRequest } from '@/types/api'
import { quantizationModelHandoffFromModelInfo } from '@/types/quantizationHandoff'
import { trainingModelHandoffFromModelInfo } from '@/types/trainingHandoff'
import { apiService } from '@/services/api'
import { DownloadModelDialog } from '@/components/DownloadModelDialog'
import { PageShell, SectionCard, StatusChip, type StatusChipConfigMap } from '@/components/common'
import { getRoutePath } from '@/config/navigation'
import { accentColors } from '@/utils/theme'

// Local type definitions to replace missing imports
export type ModelFramework = 'transformers' | 'gguf' | 'onnx' | 'pytorch' | 'safetensors';
export type ModelStatus = 'available' | 'downloading' | 'loading' | 'running' | 'stopped' | 'error';

export interface ModelFormData {
  name: string;
  framework: ModelFramework;
  port: number;
  path: string;
}

const isFineTunableModel = (model: ModelInfo) => (
  model.framework === 'transformers' || (Boolean(model.repositoryId) && model.framework !== 'llama.cpp' && model.framework !== 'ggml')
)

const getModelIdentity = (model: ModelInfo) => `${model.id ?? model.name}:${model.variant ?? model.quantization ?? ''}`

const MODELS_STATUS_CONFIG: StatusChipConfigMap = {
  available: { color: accentColors.info, label: 'Available' },
  downloading: { color: accentColors.primary, label: 'Downloading' },
  error: { color: accentColors.rose, label: 'Error' },
  library: { color: accentColors.purple, label: 'Model library' },
  file: { color: accentColors.info, label: 'File' },
  loading: { color: accentColors.info, label: 'Loading' },
  local: { color: accentColors.warning, label: 'Local files' },
  mtp: { color: accentColors.warning, label: 'MTP' },
  parameters: { color: accentColors.slate, label: 'Parameters' },
  all: { color: accentColors.purple, label: 'All' },
  count: { color: accentColors.info, label: 'Count' },
  filter: { color: accentColors.slate, label: 'Filter' },
  lora: { color: accentColors.purple, label: 'LoRA' },
  running: { color: accentColors.success, label: 'Running' },
  stopped: { color: accentColors.slate, label: 'Stopped' },
  trained: { color: accentColors.success, label: 'Trained' },
}

export const ModelsPage: React.FC = () => {
  const navigate = useNavigate()
  const location = useLocation()
  const isNarrow = useMediaQuery('(max-width:599.95px)')
  const hasInlineModelDetails = useMediaQuery('(min-width:1200px)')
  const benchmarkPath = getRoutePath('benchmark')
  const deployPath = getRoutePath('deploy')
  const finetuningPath = getRoutePath('finetuning')
  const quantizationPath = getRoutePath('quantization')

  // State for filtering and models
  const [searchTerm, setSearchTerm] = useState('')
  const [activeFilter, setActiveFilter] = useState('all')
  const [activeTab, setActiveTab] = useState(0)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [archivedModels, setArchivedModels] = useState<ModelInfo[]>([])
  const [localFiles, setLocalFiles] = useState<any[]>([])
  const [totalDiskUsage, setTotalDiskUsage] = useState(0)
  const [isLoading, setIsLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [activeDownloads, setActiveDownloads] = useState<ModelDownload[]>([])
  const activeDownloadsRef = useRef<ModelDownload[]>([])
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info' | 'warning';
  }>({
    open: false,
    message: '',
    severity: 'info'
  })

  // State for download dialog
  const [downloadDialogOpen, setDownloadDialogOpen] = useState(false)

  // State for model info dialog
  const [infoDialogOpen, setInfoDialogOpen] = useState(false)
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null)
  const [modelActionsAnchorEl, setModelActionsAnchorEl] = useState<HTMLElement | null>(null)

  // State for delete confirmation
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [fileToDelete, setFileToDelete] = useState<any | null>(null)
  const [deleting, setDeleting] = useState(false)

  // State for archive dialog
  const [archiveDialogOpen, setArchiveDialogOpen] = useState(false)
  const [modelToArchive, setModelToArchive] = useState<ModelInfo | null>(null)
  const [archiveRating, setArchiveRating] = useState(0)
  const [archiveNotes, setArchiveNotes] = useState('')
  const [performanceNotes, setPerformanceNotes] = useState('')
  const [deleteLocalFiles, setDeleteLocalFiles] = useState(true)
  const [archiving, setArchiving] = useState(false)

  // State for model delete confirmation
  const [modelDeleteDialogOpen, setModelDeleteDialogOpen] = useState(false)
  const [modelToDelete, setModelToDelete] = useState<ModelInfo | null>(null)
  const [deletingModel, setDeletingModel] = useState(false)

  const [graftDialogOpen, setGraftDialogOpen] = useState(false)
  const [graftTargetModel, setGraftTargetModel] = useState<ModelInfo | null>(null)
  const [graftDonorRepo, setGraftDonorRepo] = useState('')
  const [graftDonorFilename, setGraftDonorFilename] = useState('')
  const [graftBusy, setGraftBusy] = useState(false)
  const [graftError, setGraftError] = useState<string | null>(null)
  const [graftLogs, setGraftLogs] = useState<string[]>([])

  // State for highlighting a local file when linked from model
  const [highlightedFilePath, setHighlightedFilePath] = useState<string | null>(null)

  // Keep ref in sync with state to avoid stale closures in interval
  useEffect(() => {
    activeDownloadsRef.current = activeDownloads
  }, [activeDownloads])

  // Fetch models on component mount
  useEffect(() => {
    fetchModels()
    fetchArchivedModels()
    fetchLocalFiles()

    // Start polling for active downloads - always poll to catch ongoing downloads
    const interval = window.setInterval(() => {
      fetchDownloads()
    }, 2000) // Poll every 2 seconds for more responsive updates

    return () => {
      if (interval) {
        window.clearInterval(interval)
      }
    }
  }, [])

  useEffect(() => {
    const state = location.state as { openMtpGraft?: boolean; model?: ModelInfo } | null
    if (state?.openMtpGraft && state.model) {
      handleGraftMtp(state.model)
      navigate(location.pathname, { replace: true, state: null })
    }
  }, [location.pathname, location.state, navigate])

  // Fetch models from API
  const fetchModels = async () => {
    setIsLoading(true)
    setLoadError(null)

    try {
      const modelData = await apiService.getModels()
      setModels(modelData)

      // Also fetch any active downloads
      fetchDownloads()
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : 'Failed to load models')
      // Fallback to sample data in development
      if (process.env.NODE_ENV === 'development') {
        setModels(getSampleModels())
      }
    } finally {
      setIsLoading(false)
    }
  }

  // Fetch local model files
  const fetchLocalFiles = async () => {
    try {
      const data = await apiService.getLocalModelFiles()
      setLocalFiles(data.files || [])
      setTotalDiskUsage(data.total_size || 0)
    } catch (error) {
      console.error('Failed to fetch local files:', error)
    }
  }

  // Fetch archived models
  const fetchArchivedModels = async () => {
    try {
      const archived = await apiService.getArchivedModels()
      setArchivedModels(archived)
    } catch (error) {
      console.error('Failed to fetch archived models:', error)
    }
  }

  // Fetch active downloads
  const fetchDownloads = async () => {
    try {
      const downloads = await apiService.getModelDownloads()
      setActiveDownloads(downloads.filter(d =>
        d.status === 'downloading' || d.status === 'queued'
      ))

      // Update model download progress
      if (downloads.length > 0) {
        setModels(prev => {
          const updatedModels: ModelInfo[] = prev.map((model): ModelInfo => {
            const download = downloads.find(d => d.modelId.includes(model.name))
            if (download) {
              return {
                ...model,
                status: download.status === 'completed' ? 'available' : 'downloading',
                downloadProgress: download.progress
              }
            }
            return model
          })

          // Add any new downloads that aren't in the models list yet
          downloads.forEach(download => {
            const modelExists = updatedModels.some(model => download.modelId.includes(model.name))
            if (!modelExists && (download.status === 'downloading' || download.status === 'queued' || download.status === 'completed')) {
              // Extract model name from download ID
              const modelName = download.modelId.split('-').slice(0, -1).join('-')
              // Extract quantization from download ID
              const quantization = download.modelId.split('-').pop()
              const variant = quantization || 'unknown'

              updatedModels.push({
                // Same stem the backend uses for downloads and DELETE /v1/models/{model_id}
                id: download.modelId,
                name: modelName,
                variant,
                size: 0,
                framework: 'transformers',
                status: download.status === 'completed' ? 'available' : 'downloading',
                downloadProgress: download.status === 'completed' ? undefined : download.progress,
                parameters: modelName.includes('30B') ? '30B' :
                  modelName.includes('7B') ? '7B' :
                    modelName.includes('13B') ? '13B' : '?B',
                quantization: variant,
                contextLength: 32768
              })
            }
          })

          return updatedModels
        })
      }
    } catch (error) {
      console.error('Failed to fetch downloads:', error)
    }
  }

  // Handle download start
  const handleDownloadStart = () => {
    setSnackbar({
      open: true,
      message: 'Model download started!',
      severity: 'success'
    })
    // Refresh downloads immediately
    fetchDownloads()
    // Also refresh models list
    setTimeout(() => fetchModels(), 1000)
  }

  // Model action handlers
  const handleStopModel = async (id: string | number) => {
    const model = models.find(m => m.id === id)
    if (!model) return

    try {
      setSnackbar({
        open: true,
        message: `Stopping ${model.name}...`,
        severity: 'info'
      })
      // TODO: Call backend API to stop the service
      await apiService.performServiceAction({ action: 'stop' })
      setTimeout(() => fetchModels(), 1000)
    } catch (error) {
      setSnackbar({
        open: true,
        message: `Failed to stop ${model.name}`,
        severity: 'error'
      })
    }
  }

  const handleShowInfo = (model: ModelInfo) => {
    setSelectedModel(model)
    setInfoDialogOpen(true)
  }

  const handleModelSelect = (model: ModelInfo) => {
    setSelectedModel(model)
    if (!hasInlineModelDetails) {
      setInfoDialogOpen(true)
    }
  }

  const handleDeploy = (model: ModelInfo) => {
    navigate(deployPath, { state: { model } })
  }

  const handleGraftMtp = (model: ModelInfo) => {
    setGraftTargetModel(model)
    setGraftDonorRepo('')
    setGraftDonorFilename('')
    setGraftError(null)
    setGraftLogs([])
    setGraftDialogOpen(true)
  }

  const handleStartGraft = async () => {
    if (!graftTargetModel) return
    setGraftBusy(true)
    setGraftError(null)
    try {
      const request = {
        target: {
          name: graftTargetModel.name,
          variant: graftTargetModel.variant,
          filename: graftTargetModel.filename || graftTargetModel.localPath,
        },
        donor: {
          repo_id: graftDonorRepo.trim() || undefined,
          filename: graftDonorFilename.trim() || undefined,
        },
      }
      const preflight = await apiService.mtpGraftPreflight(request)
      if (!preflight.ok) {
        setGraftError(preflight.errors.join(' ') || 'Preflight failed')
        setGraftBusy(false)
        return
      }
      const job = await apiService.startMtpGraft(request)
      setGraftLogs(job.logs || [`Started graft job ${job.job_id}`])
      setSnackbar({
        open: true,
        message: `MTP graft started (${job.job_id})`,
        severity: 'success',
      })
      setGraftDialogOpen(false)
      fetchModels()
    } catch (error) {
      setGraftError(error instanceof Error ? error.message : 'Failed to start MTP graft')
    } finally {
      setGraftBusy(false)
    }
  }

  const handleBenchmark = (model: ModelInfo) => {
    navigate(benchmarkPath, { state: { model } })
  }

  const handleQuantize = (model: ModelInfo) => {
    navigate(quantizationPath, { state: { model: quantizationModelHandoffFromModelInfo(model) } })
  }

  const handleFineTune = (model: ModelInfo) => {
    navigate(finetuningPath, { state: { model: trainingModelHandoffFromModelInfo(model) } })
  }

  const handleDeleteClick = (file: any) => {
    setFileToDelete(file)
    setDeleteDialogOpen(true)
  }

  const handleArchiveClick = (model: ModelInfo) => {
    setModelToArchive(model)
    setArchiveRating(0)
    setArchiveNotes('')
    setPerformanceNotes('')
    setDeleteLocalFiles(true)
    setArchiveDialogOpen(true)
  }

  const handleModelDeleteClick = (model: ModelInfo) => {
    setModelToDelete(model)
    setModelDeleteDialogOpen(true)
  }

  // Handle clicking on a model's local file link
  const handleLocalFileClick = (localPath: string) => {
    setHighlightedFilePath(localPath)
    setActiveTab(1) // Switch to Downloaded Files tab
    // Clear highlight after 3 seconds
    setTimeout(() => setHighlightedFilePath(null), 3000)
  }

  const handleDeleteConfirm = async () => {
    if (!fileToDelete) return

    setDeleting(true)
    try {
      await apiService.deleteLocalModelFile(fileToDelete.path)
      setSnackbar({
        open: true,
        message: `Deleted ${fileToDelete.name} (freed ${formatBytes(fileToDelete.size)})`,
        severity: 'success'
      })
      setDeleteDialogOpen(false)
      setFileToDelete(null)
      // Refresh the lists
      fetchLocalFiles()
      fetchModels()
    } catch (error) {
      setSnackbar({
        open: true,
        message: `Failed to delete: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      })
    } finally {
      setDeleting(false)
    }
  }

  const handleArchiveConfirm = async () => {
    if (!modelToArchive) return

    setArchiving(true)
    try {
      const request: ModelArchiveRequest = {
        modelId: modelToArchive.id,
        rating: archiveRating > 0 ? archiveRating : undefined,
        notes: archiveNotes.trim() || undefined,
        performanceNotes: performanceNotes.trim() || undefined,
        deleteLocalFiles
      }

      const result = await apiService.archiveModel(request)

      setSnackbar({
        open: true,
        message: `Archived ${modelToArchive.name}${result.sizeFreed ? ` (freed ${formatBytes(result.sizeFreed)})` : ''}`,
        severity: 'success'
      })

      setArchiveDialogOpen(false)
      setModelToArchive(null)

      // Refresh the lists
      fetchModels()
      fetchArchivedModels()
      if (deleteLocalFiles) {
        fetchLocalFiles()
      }
    } catch (error) {
      setSnackbar({
        open: true,
        message: `Failed to archive: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      })
    } finally {
      setArchiving(false)
    }
  }

  const handleModelDeleteConfirm = async () => {
    if (!modelToDelete) return

    setDeletingModel(true)
    try {
      const m = modelToDelete
      const deleteKey =
        typeof m.id === 'number'
          ? `${m.name}:${m.variant ?? m.quantization ?? 'unknown'}`
          : String(m.id)
      const result = await apiService.deleteModel(deleteKey)

      setSnackbar({
        open: true,
        message: `Deleted ${modelToDelete.name}${result.sizeFreed ? ` (freed ${formatBytes(result.sizeFreed)})` : ''}`,
        severity: 'success'
      })

      setModelDeleteDialogOpen(false)
      setModelToDelete(null)

      // Refresh the lists
      fetchModels()
      fetchArchivedModels()
      fetchLocalFiles()
    } catch (error) {
      setSnackbar({
        open: true,
        message: `Failed to delete: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      })
    } finally {
      setDeletingModel(false)
    }
  }

  const handleUnarchiveModel = async (model: ModelInfo) => {
    try {
      await apiService.unarchiveModel(model.id)

      setSnackbar({
        open: true,
        message: `Unarchived ${model.name}`,
        severity: 'success'
      })

      // Refresh the lists
      fetchModels()
      fetchArchivedModels()
    } catch (error) {
      setSnackbar({
        open: true,
        message: `Failed to unarchive: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      })
    }
  }

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i]
  }

  // Filter models based on search term and active filter
  const filteredModels = useMemo(() => {
    return models.filter(model => {
      // Filter by search term
      const matchesSearch = model.name?.toLowerCase().includes(searchTerm?.toLowerCase() || '') ||
        (model.variant?.toLowerCase().includes(searchTerm?.toLowerCase() || '') || false);

      const matchesFilter =
        activeFilter === 'all' ||
        model.status === activeFilter ||
        (activeFilter === 'trained' && model.source === 'trained') ||
        (activeFilter === 'mtp' && Boolean(model.mtpCapable))

      return matchesSearch && matchesFilter;
    });
  }, [models, searchTerm, activeFilter]);

  const modelFilterOptions = [
    { id: 'all', label: 'All', color: accentColors.purple },
    { id: 'running', label: 'Running', color: accentColors.success },
    { id: 'available', label: 'Available', color: accentColors.info },
    { id: 'downloading', label: 'Downloading', color: accentColors.primary },
    { id: 'trained', label: 'Trained', color: accentColors.success },
    { id: 'mtp', label: 'MTP', color: accentColors.warning },
  ]
  const modelViewOptions = [
    { id: 0, label: 'Models', count: models.length, icon: <StorageIcon sx={{ fontSize: 16 }} /> },
    { id: 1, label: 'Files', count: localFiles.length, icon: <FolderIcon sx={{ fontSize: 16 }} /> },
    { id: 2, label: 'Archived', count: archivedModels.length, icon: <ArchiveIcon sx={{ fontSize: 16 }} /> },
  ]
  const selectedActiveModel = useMemo(() => {
    if (filteredModels.length === 0) return null
    if (!selectedModel) return filteredModels[0]

    return (
      filteredModels.find((model) => getModelIdentity(model) === getModelIdentity(selectedModel)) ||
      filteredModels[0]
    )
  }, [filteredModels, selectedModel])
  const selectedActiveDownload = selectedActiveModel
    ? activeDownloads.find((download) => download.modelId.includes(selectedActiveModel.name))
    : undefined
  const selectedActiveIsDownloading = Boolean(
    selectedActiveModel && (selectedActiveModel.status === 'downloading' || selectedActiveDownload)
  )
  const modelActionsMenuOpen = Boolean(modelActionsAnchorEl)
  const selectedFileLabel = selectedActiveModel
    ? selectedActiveModel.filename || selectedActiveModel.localPath || selectedActiveModel.path || 'No local file linked'
    : ''
  const selectedModelVariantLabel = selectedActiveModel
    ? [selectedActiveModel.quantization || selectedActiveModel.variant, selectedActiveModel.parameters].filter(Boolean).join(' / ') || 'No variant'
    : ''
  const selectedModelFacts = selectedActiveModel
    ? [
        ['Size', selectedActiveModel.size > 0 ? formatBytes(selectedActiveModel.size) : '-'],
        ['Context', selectedActiveModel.contextLength ? selectedActiveModel.contextLength.toLocaleString() : '-'],
        ['Framework', selectedActiveModel.framework || '-'],
        ['Source', selectedActiveModel.source || 'local'],
      ]
    : []
  const modelActionButtonSx = {
    height: 32,
    justifyContent: 'flex-start',
    borderRadius: '8px',
    fontSize: '0.75rem',
    fontWeight: 800,
    px: 1,
  }
  const modelPrimaryActionButtonSx = {
    ...modelActionButtonSx,
    minWidth: 116,
    justifyContent: 'center',
    boxShadow: 'none',
    '&:hover': {
      boxShadow: 'none',
    },
  }
  const modelUtilityIconButtonSx = {
    width: 32,
    height: 32,
    borderRadius: '8px',
    border: '1px solid',
    borderColor: 'divider',
    bgcolor: 'action.hover',
    '&:hover': { bgcolor: 'action.selected' },
  }
  const modelLibraryControls = (
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns: activeTab === 0
          ? { xs: 'minmax(0, 1fr) 92px', sm: 'max-content minmax(180px, 1fr) 124px auto' }
          : { xs: '1fr', sm: 'auto auto' },
        gap: 0.75,
        alignItems: 'center',
        mb: 1,
      }}
    >
      <Box
        sx={{
          minWidth: 0,
          display: 'flex',
          alignItems: 'center',
          gap: 0.5,
          flexWrap: 'nowrap',
          gridColumn: { xs: activeTab === 0 ? '1 / 2' : 'auto', sm: 'auto' },
          gridRow: { xs: activeTab === 0 ? '1' : 'auto', sm: 'auto' },
        }}
      >
        {modelViewOptions.map((view) => {
          const selected = activeTab === view.id

          return (
            <Tooltip key={view.id} title={isNarrow && !selected ? view.label : ''}>
              <Button
                size="small"
                variant={selected ? 'contained' : 'outlined'}
                startIcon={view.icon}
                onClick={() => setActiveTab(view.id)}
                sx={{
                  height: 30,
                  minHeight: 30,
                  minWidth: { xs: selected ? 84 : 34, sm: 'auto' },
                  borderRadius: '8px',
                  borderColor: selected ? 'transparent' : 'divider',
                  fontWeight: 800,
                  px: { xs: selected ? 1 : 0.65, sm: 1 },
                  py: 0,
                  whiteSpace: 'nowrap',
                  '& .MuiButton-startIcon': {
                    mr: { xs: selected ? 0.5 : 0, sm: 0.5 },
                    ml: 0,
                  },
                }}
              >
                {isNarrow && !selected ? '' : view.label}
              </Button>
            </Tooltip>
          )
        })}
      </Box>

      {activeTab === 0 && (
        <>
          <TextField
            fullWidth
            size="small"
            placeholder="Search models..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              )
            }}
            sx={{
              gridColumn: { xs: '1 / 2', sm: 'auto' },
              gridRow: { xs: '2', sm: 'auto' },
              '& .MuiInputBase-root': { height: 32 },
            }}
          />

          <TextField
            select
            size="small"
            aria-label="Filter models"
            value={activeFilter}
            onChange={(event) => setActiveFilter(event.target.value)}
            sx={{
              gridColumn: { xs: '2 / 3', sm: 'auto' },
              gridRow: { xs: '2', sm: 'auto' },
              '& .MuiInputBase-root': { height: 32 },
            }}
          >
            {modelFilterOptions.map((filter) => (
              <MenuItem key={filter.id} value={filter.id}>
                {filter.label}
              </MenuItem>
            ))}
          </TextField>
        </>
      )}

      <Box
        sx={{
          display: 'flex',
          gap: 0.5,
          flexWrap: 'nowrap',
          justifyContent: 'flex-end',
          gridColumn: { xs: activeTab === 0 ? '2 / 3' : 'auto', sm: 'auto' },
          gridRow: { xs: activeTab === 0 ? '1' : 'auto', sm: 'auto' },
        }}
      >
        <Tooltip title="Download model">
          <IconButton
            size="small"
            aria-label="Download model"
            onClick={() => setDownloadDialogOpen(true)}
            sx={{
              width: 32,
              height: 32,
              borderRadius: '8px',
              bgcolor: 'primary.main',
              color: 'primary.contrastText',
              '&:hover': { bgcolor: 'primary.dark' },
            }}
          >
            <CloudDownloadIcon sx={{ fontSize: 17 }} />
          </IconButton>
        </Tooltip>
        <Tooltip title={isLoading ? 'Refreshing models' : 'Refresh models'}>
          <span>
            <IconButton
              size="small"
              aria-label="Refresh models"
              onClick={() => {
                fetchModels()
                fetchArchivedModels()
                fetchLocalFiles()
              }}
              disabled={isLoading}
              sx={modelUtilityIconButtonSx}
            >
              {isLoading ? <CircularProgress size={15} /> : <RefreshIcon sx={{ fontSize: 17 }} />}
            </IconButton>
          </span>
        </Tooltip>
      </Box>
    </Box>
  )

  // Handle snackbar close
  const handleSnackbarClose = () => {
    setSnackbar(prev => ({ ...prev, open: false }))
  }

  // Sample models for development
  const getSampleModels = (): ModelInfo[] => [
    {
      id: 1,
      name: 'Qwen3-Coder-30B',
      variant: 'Q4_K_M',
      size: 0,
      framework: 'transformers',
      status: 'running',
      port: 8001,
      latency: '156ms',
      memory: '22.5GB',
      path: 'unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF',
      parameters: '30B',
      quantization: 'Q4_K_M',
      contextLength: 128000
    },
    {
      id: 2,
      name: 'Llama-3-8B-Instruct',
      variant: 'Q4_K_M',
      size: 0,
      framework: 'transformers',
      status: 'stopped',
      port: 8002,
      latency: '--',
      memory: '--',
      path: 'TheBloke/Llama-3-8B-Instruct-GGUF',
      parameters: '8B',
      quantization: 'Q4_K_M',
      contextLength: 32768
    },
    {
      id: 3,
      name: 'Mistral-7B-Instruct-v0.2',
      variant: 'Q5_K_M',
      size: 0,
      framework: 'transformers',
      status: 'stopped',
      port: 8003,
      latency: '--',
      memory: '--',
      path: 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
      parameters: '7B',
      quantization: 'Q5_K_M',
      contextLength: 32768
    }
  ]

  return (
    <PageShell
      sx={{
        py: 1.25,
      }}
    >
      {loadError && (
        <Alert
          severity="error"
          sx={{
            mb: 2,
            borderRadius: 0.5,
            fontSize: '0.8125rem'
          }}
        >
          {loadError}
        </Alert>
      )}

      {/* Tab Content */}
      {activeTab === 0 && (
        <Grid
          container
          spacing={1.5}
          data-models-workspace
          sx={{ alignItems: 'flex-start' }}
        >
          <Grid item xs={12} lg={8.4} sx={{ display: 'flex' }}>
            <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 1.5 }}>
              <SectionCard
                title="Model inventory"
                icon={<StorageIcon />}
                accentColor={accentColors.purple}
                action={<StatusChip status="library" label={`${filteredModels.length} models / ${localFiles.length} files`} statusConfig={MODELS_STATUS_CONFIG} />}
                sx={{ flex: '0 0 auto' }}
              >
              {modelLibraryControls}
              {isLoading && models.length === 0 ? (
                <Box sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  alignItems: 'center',
                  minHeight: 160,
                  bgcolor: 'action.hover',
                  borderRadius: '8px',
                  border: '1px solid',
                  borderColor: 'divider'
                }}>
                  <CircularProgress size={32} sx={{ mb: 1.5 }} />
                  <Typography variant="body1" color="text.secondary" sx={{ mb: 0.5, fontSize: '0.875rem' }}>
                    Loading models...
                  </Typography>
                </Box>
              ) : (
                <TableContainer
                  component={Paper}
                  variant="outlined"
                  sx={{
                    display: { xs: 'none', sm: 'block' },
                    width: '100%',
                    maxWidth: '100%',
                    borderRadius: '8px',
                    bgcolor: 'background.paper',
                    borderColor: 'divider',
                    overflow: 'hidden',
                    '& .MuiTable-root': {
                      width: '100%',
                      tableLayout: { sm: 'fixed' },
                      display: { xs: 'block', sm: 'table' },
                    },
                    '& .MuiTableHead-root': {
                      display: { xs: 'none', sm: 'table-header-group' },
                    },
                    '& .MuiTableBody-root': {
                      display: { xs: 'block', sm: 'table-row-group' },
                    },
                    '& .MuiTableRow-root': {
                      display: { xs: 'grid', sm: 'table-row' },
                      gridTemplateColumns: { xs: 'minmax(0, 1fr)', sm: 'none' },
                      gap: { xs: 0.75, sm: 0 },
                      p: { xs: 1.25, sm: 0 },
                      borderBottom: { xs: '1px solid', sm: 0 },
                      borderColor: { xs: 'divider', sm: 'transparent' },
                      '&:last-child': {
                        borderBottom: 0,
                      },
                    },
                    '& .MuiTableCell-root': {
                      display: { xs: 'block', sm: 'table-cell' },
                      width: { xs: '100%', sm: 'auto' },
                      borderColor: 'divider',
                      py: { xs: 0, sm: 0.75 },
                      px: { xs: 0, sm: 2 },
                      minWidth: 0,
                      borderBottom: { xs: 0 },
                    },
                  }}
                >
                  <Table size="small" aria-label="Active model catalog">
                    <TableHead>
                      <TableRow sx={{ bgcolor: 'action.hover' }}>
                        <TableCell sx={{ width: '40%', fontSize: '0.75rem', fontWeight: 800 }}>Model</TableCell>
                        <TableCell sx={{ width: '28%', fontSize: '0.75rem', fontWeight: 800 }}>State</TableCell>
                        <TableCell sx={{ width: '32%', fontSize: '0.75rem', fontWeight: 800 }}>Size / File</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {filteredModels.map((model: ModelInfo) => {
                        const download = activeDownloads.find(d => d.modelId.includes(model.name))
                        const isDownloading = model.status === 'downloading' || Boolean(download)
                        const modelStatus = model.status || 'available'
                        const fileLabel = model.filename || model.localPath || model.path || 'No local file linked'
                        const selected = Boolean(selectedActiveModel && getModelIdentity(model) === getModelIdentity(selectedActiveModel))

                        return (
                          <TableRow
                            key={getModelIdentity(model)}
                            hover
                            selected={selected}
                            aria-selected={selected}
                            tabIndex={0}
                            onClick={() => handleModelSelect(model)}
                            onKeyDown={(event) => {
                              if (event.key === 'Enter' || event.key === ' ') {
                                event.preventDefault()
                                handleModelSelect(model)
                              }
                            }}
                            sx={{
                              cursor: 'pointer',
                              '&:last-child td': { borderBottom: 0 },
                              ...(selected && {
                                bgcolor: alpha(accentColors.purple, 0.12),
                                '&:hover': { bgcolor: alpha(accentColors.purple, 0.16) },
                                '& td:first-of-type': {
                                  boxShadow: `inset 3px 0 0 ${accentColors.purple}`,
                                  pl: { xs: 1.25, sm: 2 },
                                },
                              }),
                            }}
                          >
                            <TableCell>
                              <Box sx={{ minWidth: 0 }}>
                                <Typography variant="body2" title={model.name} sx={{ fontWeight: 800, lineHeight: 1.2 }} noWrap>
                                  {model.name}
                                </Typography>
                                <Typography variant="caption" color="text.secondary" noWrap>
                                  {[model.quantization || model.variant, model.parameters].filter(Boolean).join(' / ') || 'No variant'}
                                </Typography>
                              </Box>
                            </TableCell>
                            <TableCell>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
                                <StatusChip status={modelStatus} statusConfig={MODELS_STATUS_CONFIG} height={22} />
                                {model.source === 'trained' && (
                                  <StatusChip status="trained" statusConfig={MODELS_STATUS_CONFIG} height={22} />
                                )}
                                {model.mtpCapable && (
                                  <StatusChip
                                    status="mtp"
                                    label={model.mtpNextnLayers ? `MTP x${model.mtpNextnLayers}` : 'MTP'}
                                    statusConfig={MODELS_STATUS_CONFIG}
                                    height={22}
                                    title="GGUF includes NextN prediction heads (verified from file metadata)"
                                  />
                                )}
                              </Box>
                              {isDownloading && (
                                <Box sx={{ mt: 0.75, maxWidth: 260 }}>
                                  <Box display="flex" justifyContent="space-between" mb={0.25}>
                                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                                      Downloading
                                    </Typography>
                                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                                      {download ? `${Math.round(download.progress)}%` : `${Math.round(model.downloadProgress || 0)}%`}
                                    </Typography>
                                  </Box>
                                  <LinearProgress
                                    variant="determinate"
                                    value={download?.progress || model.downloadProgress || 0}
                                    sx={{
                                      height: 4,
                                      borderRadius: '3px',
                                      bgcolor: alpha(accentColors.primary, 0.12),
                                      '& .MuiLinearProgress-bar': {
                                        borderRadius: '3px',
                                        bgcolor: accentColors.primary,
                                      },
                                    }}
                                  />
                                </Box>
                              )}
                            </TableCell>
                            <TableCell>
                              <Box sx={{ minWidth: 0 }}>
                                {model.size > 0 && (
                                  <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: '0.7rem' }}>
                                    {formatBytes(model.size)}
                                  </Typography>
                                )}
                                <Box
                                  sx={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 0.5,
                                    color: 'text.secondary',
                                    minWidth: 0,
                                  }}
                                >
                                  <FolderIcon sx={{ fontSize: '0.8rem', flexShrink: 0 }} />
                                  <Typography
                                    variant="caption"
                                    title={fileLabel}
                                    sx={{
                                      fontSize: '0.65rem',
                                      fontFamily: 'monospace',
                                      overflow: 'hidden',
                                      textOverflow: 'ellipsis',
                                      whiteSpace: 'nowrap',
                                    }}
                                  >
                                    {fileLabel}
                                  </Typography>
                                </Box>
                              </Box>
                            </TableCell>
                          </TableRow>
                        )
                      })}

                      {filteredModels.length === 0 && (
                        <TableRow>
                          <TableCell colSpan={3} align="center" sx={{ py: 4 }}>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                              No models found
                            </Typography>
                            <Button
                              variant="contained"
                              startIcon={<CloudDownloadIcon sx={{ fontSize: '1rem' }} />}
                              onClick={() => setDownloadDialogOpen(true)}
                              size="small"
                              sx={{
                                height: 32,
                                fontWeight: 700,
                                fontSize: '0.75rem',
                                py: 0,
                              }}
                            >
                              Download model
                            </Button>
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}

              {!isLoading && (
                <Box role="listbox" aria-label="Models" sx={{ display: { xs: 'grid', sm: 'none' }, gap: 0.5 }}>
                  {filteredModels.map((model: ModelInfo) => {
                    const download = activeDownloads.find(d => d.modelId.includes(model.name))
                    const isDownloading = model.status === 'downloading' || Boolean(download)
                    const modelStatus = model.status || 'available'
                    const fileLabel = model.filename || model.localPath || model.path || 'No local file linked'
                    const selected = Boolean(selectedActiveModel && getModelIdentity(model) === getModelIdentity(selectedActiveModel))

                    return (
                      <Box
                        key={getModelIdentity(model)}
                        role="option"
                        tabIndex={0}
                        aria-selected={selected}
                        onClick={() => handleModelSelect(model)}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault()
                            handleModelSelect(model)
                          }
                        }}
                        sx={{
                          cursor: 'pointer',
                          p: 1,
                          borderRadius: '8px',
                          border: '1px solid',
                          borderColor: selected ? alpha(accentColors.purple, 0.52) : 'divider',
                          bgcolor: selected ? alpha(accentColors.purple, 0.13) : 'background.paper',
                          boxShadow: selected ? `inset 3px 0 0 ${accentColors.purple}` : 'none',
                          display: 'grid',
                          gridTemplateColumns: 'minmax(0, 1fr) auto',
                          gap: 0.75,
                          alignItems: 'start',
                        }}
                      >
                        <Box sx={{ minWidth: 0 }}>
                          <Typography variant="body2" title={model.name} sx={{ fontWeight: 850, lineHeight: 1.15 }} noWrap>
                            {model.name}
                          </Typography>
                          <Typography variant="caption" color="text.secondary" noWrap>
                            {[
                              model.quantization || model.variant,
                              model.parameters,
                              model.size > 0 ? formatBytes(model.size) : null,
                            ].filter(Boolean).join(' / ') || 'No variant'}
                          </Typography>
                          <Box sx={{ mt: 0.25, display: 'flex', alignItems: 'center', gap: 0.5, color: 'text.secondary', minWidth: 0 }}>
                            <FolderIcon sx={{ fontSize: '0.75rem', flexShrink: 0 }} />
                            <Typography
                              variant="caption"
                              title={fileLabel}
                              sx={{
                                fontSize: '0.62rem',
                                fontFamily: 'monospace',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                              }}
                            >
                              {fileLabel}
                            </Typography>
                          </Box>
                          {isDownloading && (
                            <LinearProgress
                              variant="determinate"
                              value={download?.progress || model.downloadProgress || 0}
                              sx={{
                                mt: 0.75,
                                height: 4,
                                borderRadius: '3px',
                                bgcolor: alpha(accentColors.primary, 0.12),
                                '& .MuiLinearProgress-bar': {
                                  borderRadius: '3px',
                                  bgcolor: accentColors.primary,
                                },
                              }}
                            />
                          )}
                        </Box>
                        <StatusChip status={modelStatus} statusConfig={MODELS_STATUS_CONFIG} height={22} />
                      </Box>
                    )
                  })}

                  {filteredModels.length === 0 && (
                    <Box
                      sx={{
                        py: 2,
                        px: 1,
                        textAlign: 'center',
                        borderRadius: '8px',
                        border: '1px dashed',
                        borderColor: 'divider',
                        bgcolor: 'action.hover',
                      }}
                    >
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        No models found
                      </Typography>
                      <Button
                        variant="contained"
                        startIcon={<CloudDownloadIcon sx={{ fontSize: '1rem' }} />}
                        onClick={() => setDownloadDialogOpen(true)}
                        size="small"
                        sx={{ height: 32, fontWeight: 800 }}
                      >
                        Download
                      </Button>
                    </Box>
                  )}
                </Box>
              )}
              </SectionCard>
            </Box>
          </Grid>

          <Grid item xs={12} lg={3.6} sx={{ display: { xs: 'none', lg: 'block' } }}>
            <SectionCard
              title={selectedActiveModel ? selectedActiveModel.name : 'Selected model'}
              subtitle={selectedActiveModel ? selectedModelVariantLabel : undefined}
              icon={<InfoIcon />}
              accentColor={accentColors.info}
              action={selectedActiveModel ? (
                <StatusChip status={selectedActiveModel.status || 'available'} statusConfig={MODELS_STATUS_CONFIG} />
              ) : undefined}
              collapsible={isNarrow}
              defaultExpanded={!isNarrow}
            >
              {selectedActiveModel ? (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25 }}>
                  {(selectedActiveModel.source === 'trained' || selectedActiveModel.mtpCapable || selectedActiveIsDownloading) && (
                    <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap' }}>
                      {selectedActiveModel.source === 'trained' && (
                        <StatusChip status="trained" statusConfig={MODELS_STATUS_CONFIG} height={22} />
                      )}
                      {selectedActiveModel.mtpCapable && (
                        <StatusChip
                          status="mtp"
                          label={selectedActiveModel.mtpNextnLayers ? `MTP x${selectedActiveModel.mtpNextnLayers}` : 'MTP'}
                          statusConfig={MODELS_STATUS_CONFIG}
                          height={22}
                        />
                      )}
                      {selectedActiveIsDownloading && (
                        <StatusChip status="downloading" statusConfig={MODELS_STATUS_CONFIG} height={22} />
                      )}
                    </Box>
                  )}

                  <Box
                    sx={{
                      p: 1,
                      borderRadius: '8px',
                      border: '1px solid',
                      borderColor: 'divider',
                      bgcolor: 'action.hover',
                      display: 'grid',
                      gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                      gap: 0.75,
                      minWidth: 0,
                    }}
                  >
                    {selectedModelFacts.map(([label, value]) => (
                      <Box key={label} sx={{ minWidth: 0 }}>
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontSize: '0.68rem', fontWeight: 800, textTransform: 'uppercase', lineHeight: 1.15 }}>
                          {label}
                        </Typography>
                        <Typography variant="body2" title={value} sx={{ fontWeight: 800, lineHeight: 1.25 }} noWrap>
                          {value}
                        </Typography>
                      </Box>
                    ))}
                    <Box sx={{ gridColumn: '1 / -1', minWidth: 0, pt: 0.25, borderTop: '1px solid', borderColor: 'divider' }}>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontSize: '0.68rem', fontWeight: 800, textTransform: 'uppercase', lineHeight: 1.15 }}>
                        File
                      </Typography>
                      <Typography
                        variant="caption"
                        title={selectedFileLabel}
                        sx={{
                          display: 'block',
                          fontFamily: 'monospace',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {selectedFileLabel}
                      </Typography>
                    </Box>
                  </Box>

                  {selectedActiveIsDownloading && (
                    <Box>
                      <Box display="flex" justifyContent="space-between" mb={0.25}>
                        <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                          Download progress
                        </Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                          {selectedActiveDownload ? `${Math.round(selectedActiveDownload.progress)}%` : `${Math.round(selectedActiveModel.downloadProgress || 0)}%`}
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={selectedActiveDownload?.progress || selectedActiveModel.downloadProgress || 0}
                        sx={{
                          height: 5,
                          borderRadius: '4px',
                          bgcolor: alpha(accentColors.primary, 0.12),
                          '& .MuiLinearProgress-bar': {
                            borderRadius: '4px',
                            bgcolor: accentColors.primary,
                          },
                        }}
                      />
                    </Box>
                  )}

                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
                    {selectedActiveModel.status === 'running' ? (
                      <Button
                        color="error"
                        variant="contained"
                        size="small"
                        startIcon={<StopIcon sx={{ fontSize: 16 }} />}
                        onClick={() => handleStopModel(selectedActiveModel.id)}
                        sx={modelPrimaryActionButtonSx}
                      >
                        Stop
                      </Button>
                    ) : (
                      <Button
                        variant="contained"
                        size="small"
                        startIcon={<DeployIcon sx={{ fontSize: 16 }} />}
                        onClick={() => handleDeploy(selectedActiveModel)}
                        disabled={selectedActiveIsDownloading}
                        sx={modelPrimaryActionButtonSx}
                      >
                        Deploy
                      </Button>
                    )}

                    <Tooltip title="More model actions">
                      <IconButton
                        size="small"
                        aria-label="More model actions"
                        aria-controls={modelActionsMenuOpen ? 'model-actions-menu' : undefined}
                        aria-haspopup="menu"
                        aria-expanded={modelActionsMenuOpen ? 'true' : undefined}
                        onClick={(event) => setModelActionsAnchorEl(event.currentTarget)}
                        sx={{ ...modelUtilityIconButtonSx, ml: 'auto' }}
                      >
                        <MoreVertIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>
              ) : (
                <Box
                  sx={{
                    py: 3,
                    px: 2,
                    textAlign: 'center',
                    borderRadius: '8px',
                    border: '1px dashed',
                    borderColor: 'divider',
                    bgcolor: 'action.hover',
                  }}
                >
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    Select or download a model.
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={<CloudDownloadIcon sx={{ fontSize: '1rem' }} />}
                    onClick={() => setDownloadDialogOpen(true)}
                    size="small"
                    sx={{ height: 32, fontWeight: 800 }}
                  >
                    Download
                  </Button>
                </Box>
              )}
            </SectionCard>
          </Grid>
        </Grid>
      )}

      {/* Archived Models Tab */}
      {activeTab === 2 && (
        <Box data-models-archived-workspace>
          <SectionCard
            title="Archived models"
            subtitle="Retained model notes, ratings, and restore/delete actions."
            icon={<ArchiveIcon />}
            accentColor={accentColors.warning}
            action={<StatusChip status="archived" label={`${archivedModels.length} archived`} statusConfig={MODELS_STATUS_CONFIG} />}
          >
          {modelLibraryControls}
          {archivedModels.length === 0 ? (
            <Box sx={{
              textAlign: 'center',
              py: 3,
              bgcolor: 'action.hover',
              borderRadius: '8px',
              border: '1px dashed',
              borderColor: 'divider'
            }}>
              <ArchiveIcon sx={{ fontSize: '2rem', color: 'text.secondary', mb: 1 }} />
              <Typography variant="h6" color="text.secondary" sx={{ mb: 0.5, fontWeight: 600, fontSize: '1rem' }}>
                No archived models
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.875rem' }}>
                Models you archive will appear here with their ratings and notes
              </Typography>
            </Box>
          ) : (
            <Grid container spacing={1.5} sx={{ px: 0 }}>
              {archivedModels.map((model: any) => (
                <Grid item key={model.id || model.name} xs={12} sm={6} md={4} lg={3} xl={2}>
                  <Box
                    sx={{
                      height: '100%',
                      minHeight: 230,
                      p: 2,
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 1.25,
                      borderRadius: '8px',
                      border: '1px solid',
                      borderColor: 'divider',
                      bgcolor: 'background.paper',
                      opacity: 0.86,
                      transition: 'border-color 0.2s ease, background-color 0.2s ease, transform 0.2s ease',
                      '&:hover': {
                        borderColor: alpha(accentColors.warning, 0.42),
                        bgcolor: alpha(accentColors.warning, 0.05),
                        transform: 'translateY(-1px)',
                      },
                    }}
                  >
                    <Box sx={{ minWidth: 0 }}>
                      <Typography variant="h6" title={model.name} sx={{ fontSize: '0.95rem', fontWeight: 700, lineHeight: 1.25, wordBreak: 'break-word' }}>
                        {model.name}
                      </Typography>
                      <Typography color="text.secondary" sx={{ fontSize: '0.75rem', wordBreak: 'break-word' }}>
                        {model.quantization || model.variant}
                      </Typography>
                    </Box>

                    <Box display="flex" alignItems="center" gap={0.75} flexWrap="wrap">
                      <StatusChip status="archived" statusConfig={MODELS_STATUS_CONFIG} height={22} />
                      {model.parameters && (
                        <StatusChip status="parameters" label={model.parameters} statusConfig={MODELS_STATUS_CONFIG} height={22} />
                      )}
                    </Box>

                    {/* Rating Display */}
                    {model.archiveRating && (
                      <Box display="flex" alignItems="center" gap={0.5}>
                        <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                          Rating:
                        </Typography>
                        {[1, 2, 3, 4, 5].map((star) => (
                          <Box key={star} sx={{ color: star <= model.archiveRating ? 'warning.main' : 'grey.300' }}>
                            <StarIcon sx={{ fontSize: '0.8rem' }} />
                          </Box>
                        ))}
                      </Box>
                    )}

                    {/* Archive Notes */}
                    {model.archiveNotes && (
                      <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: '0.7rem', fontStyle: 'italic' }}>
                        "{model.archiveNotes}"
                      </Typography>
                    )}

                    {/* Performance Notes */}
                    {model.performanceNotes && (
                      <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: '0.7rem' }}>
                        Performance: {model.performanceNotes}
                      </Typography>
                    )}

                    {/* Archive Date */}
                    {model.archivedAt && (
                      <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: '0.65rem' }}>
                        Archived: {new Date(model.archivedAt).toLocaleDateString()}
                      </Typography>
                    )}
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mt: 'auto' }}>
                      <Button
                        size="small"
                        onClick={() => handleShowInfo(model)}
                        startIcon={<InfoIcon sx={{ fontSize: '0.9rem' }} />}
                        sx={{ borderRadius: '8px', fontSize: '0.7rem', minWidth: 'auto', px: 0.75 }}
                      >
                        Info
                      </Button>
                      <Tooltip title="Restore this model">
                        <Button
                          size="small"
                          onClick={() => handleUnarchiveModel(model)}
                          color="success"
                          startIcon={<UnarchiveIcon sx={{ fontSize: '0.9rem' }} />}
                          sx={{ borderRadius: '8px', fontSize: '0.7rem', minWidth: 'auto', px: 0.75 }}
                        >
                          Restore
                        </Button>
                      </Tooltip>
                      <Tooltip title="Delete permanently">
                        <Button
                          size="small"
                          onClick={() => handleModelDeleteClick(model)}
                          color="error"
                          startIcon={<DeleteIcon sx={{ fontSize: '0.9rem' }} />}
                          sx={{ borderRadius: '8px', fontSize: '0.7rem', minWidth: 'auto', px: 0.75 }}
                        >
                          Delete
                        </Button>
                      </Tooltip>
                    </Box>
                  </Box>
                </Grid>
              ))}
            </Grid>
          )}
          </SectionCard>
        </Box>
      )}

      {/* Downloaded Files Tab */}
      {activeTab === 1 && (
        <Box data-models-files-workspace sx={{ flex: 1, minHeight: 0, overflow: { xs: 'visible', md: 'auto' } }}>
          <SectionCard
            title="Downloaded files"
            subtitle={`${formatBytes(totalDiskUsage)} on disk`}
            icon={<FolderIcon />}
            accentColor={accentColors.warning}
            action={<StatusChip status="local" label={`${localFiles.length} files`} statusConfig={MODELS_STATUS_CONFIG} />}
          >
          {modelLibraryControls}
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow sx={{ bgcolor: 'action.hover' }}>
                  <TableCell sx={{ fontWeight: 600, fontSize: '0.8rem' }}>File Name</TableCell>
                  <TableCell sx={{ fontWeight: 600, fontSize: '0.8rem' }}>Size</TableCell>
                  <TableCell sx={{ fontWeight: 600, fontSize: '0.8rem' }}>Modified</TableCell>
                  <TableCell sx={{ fontWeight: 600, fontSize: '0.8rem' }}>Type</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 600, fontSize: '0.8rem' }}>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {localFiles.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={5} align="center" sx={{ py: 4 }}>
                      <Typography color="text.secondary" sx={{ fontSize: '0.875rem' }}>
                        No downloaded files found
                      </Typography>
                      <Button
                        variant="contained"
                        startIcon={<CloudDownloadIcon />}
                        onClick={() => setDownloadDialogOpen(true)}
                        size="small"
                        sx={{ mt: 2, fontSize: '0.75rem' }}
                      >
                        Download Your First Model
                      </Button>
                    </TableCell>
                  </TableRow>
                ) : (
                  localFiles.map((file) => (
                    <TableRow
                      key={file.path}
                      sx={{
                        '&:hover': { bgcolor: 'grey.50' },
                        '&:last-child td, &:last-child th': { border: 0 },
                        ...(highlightedFilePath === file.path && {
                          bgcolor: 'primary.50',
                          animation: 'highlight-pulse 1s ease-in-out 2',
                          '@keyframes highlight-pulse': {
                            '0%, 100%': { bgcolor: 'primary.50' },
                            '50%': { bgcolor: 'primary.100' }
                          }
                        })
                      }}
                    >
                      <TableCell sx={{ fontSize: '0.8rem' }}>
                        <Tooltip title={file.full_path}>
                          <Box sx={{ fontFamily: 'monospace' }}>{file.name}</Box>
                        </Tooltip>
                      </TableCell>
                      <TableCell sx={{ fontSize: '0.8rem' }}>{formatBytes(file.size)}</TableCell>
                      <TableCell sx={{ fontSize: '0.8rem' }}>
                        {new Date(file.modified).toLocaleDateString()} {new Date(file.modified).toLocaleTimeString()}
                      </TableCell>
                      <TableCell>
                        <StatusChip status="file" label={file.extension.toUpperCase()} statusConfig={MODELS_STATUS_CONFIG} height={22} />
                      </TableCell>
                      <TableCell align="right">
                        <Tooltip title="Delete file">
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => handleDeleteClick(file)}
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
          </SectionCard>
        </Box>
      )}

      {/* Snackbar for notifications */}
      <Menu
        id="model-actions-menu"
        anchorEl={modelActionsAnchorEl}
        open={modelActionsMenuOpen}
        onClose={() => setModelActionsAnchorEl(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        PaperProps={{
          sx: {
            minWidth: 190,
            borderRadius: '8px',
            border: '1px solid',
            borderColor: 'divider',
          },
        }}
      >
        {selectedActiveModel && [
          <MenuItem
            key="benchmark"
            dense
            disabled={selectedActiveIsDownloading}
            onClick={() => {
              setModelActionsAnchorEl(null)
              handleBenchmark(selectedActiveModel)
            }}
          >
            <BenchmarkIcon sx={{ mr: 1.25, fontSize: 18 }} />
            Benchmark
          </MenuItem>,
          <MenuItem
            key="quantize"
            dense
            disabled={selectedActiveIsDownloading}
            onClick={() => {
              setModelActionsAnchorEl(null)
              handleQuantize(selectedActiveModel)
            }}
          >
            <QuantizeIcon sx={{ mr: 1.25, fontSize: 18 }} />
            Quantize
          </MenuItem>,
          <MenuItem
            key="fine-tune"
            dense
            disabled={selectedActiveIsDownloading || !isFineTunableModel(selectedActiveModel)}
            onClick={() => {
              setModelActionsAnchorEl(null)
              handleFineTune(selectedActiveModel)
            }}
          >
            <TuneIcon sx={{ mr: 1.25, fontSize: 18 }} />
            Fine-tune
          </MenuItem>,
          <MenuItem
            key="graft-mtp"
            dense
            disabled={selectedActiveIsDownloading || !selectedActiveModel.localPath || Boolean(selectedActiveModel.mtpCapable)}
            onClick={() => {
              setModelActionsAnchorEl(null)
              handleGraftMtp(selectedActiveModel)
            }}
          >
            <GraftIcon sx={{ mr: 1.25, fontSize: 18 }} />
            Add MTP heads
          </MenuItem>,
          <MenuItem
            key="details"
            dense
            onClick={() => {
              setModelActionsAnchorEl(null)
              handleShowInfo(selectedActiveModel)
            }}
          >
            <InfoIcon sx={{ mr: 1.25, fontSize: 18 }} />
            Details
          </MenuItem>,
          <MenuItem
            key="file"
            dense
            disabled={!selectedActiveModel.localPath}
            onClick={() => {
              setModelActionsAnchorEl(null)
              selectedActiveModel.localPath && handleLocalFileClick(selectedActiveModel.localPath)
            }}
          >
            <FolderIcon sx={{ mr: 1.25, fontSize: 18 }} />
            Local file
          </MenuItem>,
          <MenuItem
            key="archive"
            dense
            disabled={selectedActiveIsDownloading || selectedActiveModel.status === 'running'}
            onClick={() => {
              setModelActionsAnchorEl(null)
              handleArchiveClick(selectedActiveModel)
            }}
          >
            <ArchiveIcon sx={{ mr: 1.25, fontSize: 18 }} />
            Archive
          </MenuItem>,
          <MenuItem
            key="delete"
            dense
            disabled={selectedActiveIsDownloading || selectedActiveModel.status === 'running'}
            onClick={() => {
              setModelActionsAnchorEl(null)
              handleModelDeleteClick(selectedActiveModel)
            }}
            sx={{ color: 'error.main' }}
          >
            <DeleteIcon sx={{ mr: 1.25, fontSize: 18 }} />
            Delete
          </MenuItem>,
        ]}
      </Menu>

      <Dialog open={graftDialogOpen} onClose={() => !graftBusy && setGraftDialogOpen(false)} fullWidth maxWidth="sm">
        <DialogTitle>Graft MTP heads</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Copy MTP prediction heads from a donor GGUF onto {graftTargetModel?.name || 'this model'}.
          </Typography>
          <TextField
            fullWidth
            size="small"
            label="Donor Hugging Face repo (optional)"
            value={graftDonorRepo}
            onChange={(e) => setGraftDonorRepo(e.target.value)}
            sx={{ mb: 1.5 }}
          />
          <TextField
            fullWidth
            size="small"
            label="Donor filename (optional)"
            value={graftDonorFilename}
            onChange={(e) => setGraftDonorFilename(e.target.value)}
          />
          {graftError && (
            <Alert severity="error" sx={{ mt: 2 }}>{graftError}</Alert>
          )}
          {graftLogs.length > 0 && (
            <Typography variant="caption" component="pre" sx={{ mt: 2, whiteSpace: 'pre-wrap' }}>
              {graftLogs.join('\n')}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setGraftDialogOpen(false)} disabled={graftBusy}>Cancel</Button>
          <Button onClick={() => void handleStartGraft()} disabled={graftBusy || !graftTargetModel} variant="contained">
            {graftBusy ? 'Starting…' : 'Start graft'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
      >
        <Alert
          onClose={handleSnackbarClose}
          severity={snackbar.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>

      {/* Download Model Dialog */}
      <DownloadModelDialog
        open={downloadDialogOpen}
        onClose={() => setDownloadDialogOpen(false)}
        onDownloadStart={handleDownloadStart}
      />

      {/* Model Info Dialog */}
      <Dialog
        open={infoDialogOpen}
        onClose={() => setInfoDialogOpen(false)}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            borderRadius: '8px',
            mx: { xs: 1.5, sm: 3 },
          },
        }}
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, pr: 1.5 }}>
          <Box sx={{ minWidth: 0, flex: 1 }}>
            <Typography variant="h6" noWrap>{selectedModel?.name}</Typography>
            <Typography variant="caption" color="text.secondary">
              {selectedModel?.variant || selectedModel?.quantization}
            </Typography>
          </Box>
          <Tooltip title="Close">
            <IconButton
              size="small"
              aria-label="Close model details"
              onClick={() => setInfoDialogOpen(false)}
              sx={modelUtilityIconButtonSx}
            >
              <CloseIcon sx={{ fontSize: 17 }} />
            </IconButton>
          </Tooltip>
        </DialogTitle>
        <DialogContent>
          {selectedModel && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box>
                <Typography variant="subtitle2" color="text.secondary">Status</Typography>
                <StatusChip status={selectedModel.status || 'available'} statusConfig={MODELS_STATUS_CONFIG} sx={{ mt: 0.5 }} />
              </Box>

              {selectedModel.parameters && (
                <Box>
                  <Typography variant="subtitle2" color="text.secondary">Parameters</Typography>
                  <Typography variant="body1">{selectedModel.parameters}</Typography>
                </Box>
              )}

              {selectedModel.size > 0 && (
                <Box>
                  <Typography variant="subtitle2" color="text.secondary">Size</Typography>
                  <Typography variant="body1">{formatBytes(selectedModel.size)}</Typography>
                </Box>
              )}

              {selectedModel.contextLength && (
                <Box>
                  <Typography variant="subtitle2" color="text.secondary">Context Length</Typography>
                  <Typography variant="body1">{selectedModel.contextLength.toLocaleString()} tokens</Typography>
                </Box>
              )}

              {selectedModel.repositoryId && (
                <Box>
                  <Typography variant="subtitle2" color="text.secondary">Repository</Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {selectedModel.repositoryId}
                  </Typography>
                </Box>
              )}

              {selectedModel.localPath && (
                <Box>
                  <Typography variant="subtitle2" color="text.secondary">Local File</Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                    <FolderIcon sx={{ fontSize: '1rem', color: 'text.secondary' }} />
                    <Typography
                      variant="body2"
                      sx={{
                        fontFamily: 'monospace',
                        cursor: 'pointer',
                        '&:hover': { color: 'primary.main', textDecoration: 'underline' }
                      }}
                      onClick={() => {
                        setInfoDialogOpen(false)
                        handleLocalFileClick(selectedModel.localPath!)
                      }}
                    >
                      {selectedModel.filename || selectedModel.localPath}
                    </Typography>
                  </Box>
                </Box>
              )}

              {selectedModel.description && (
                <Box>
                  <Typography variant="subtitle2" color="text.secondary">Description</Typography>
                  <Typography variant="body2">{selectedModel.description}</Typography>
                </Box>
              )}

              {selectedModel.vramRequired && (
                <Box>
                  <Typography variant="subtitle2" color="text.secondary">VRAM Required</Typography>
                  <Typography variant="body1">{selectedModel.vramRequired} GB</Typography>
                </Box>
              )}

              {selectedModel.license && (
                <Box>
                  <Typography variant="subtitle2" color="text.secondary">License</Typography>
                  <Typography variant="body2">{selectedModel.license}</Typography>
                </Box>
              )}

              {/* Training Metadata Section (for fine-tuned models) */}
              {selectedModel.source === 'trained' && (
                <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                      <StatusChip status="trained" label="Fine-Tuned Model" statusConfig={MODELS_STATUS_CONFIG} />
                    </Box>

                  {selectedModel.baseModel && (
                    <Box sx={{ mb: 1.5 }}>
                      <Typography variant="subtitle2" color="text.secondary">Base Model</Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                        {selectedModel.baseModel}
                      </Typography>
                    </Box>
                  )}

                  {selectedModel.finalLoss != null && (
                    <Box sx={{ mb: 1.5 }}>
                      <Typography variant="subtitle2" color="text.secondary">Final Training Loss</Typography>
                      <Typography variant="body1" sx={{ fontWeight: 500 }}>
                        {selectedModel.finalLoss.toFixed(4)}
                      </Typography>
                    </Box>
                  )}

                  {selectedModel.totalSteps && (
                    <Box sx={{ mb: 1.5 }}>
                      <Typography variant="subtitle2" color="text.secondary">Training Steps</Typography>
                      <Typography variant="body1">{selectedModel.totalSteps.toLocaleString()}</Typography>
                    </Box>
                  )}

                  {selectedModel.loraConfig && (
                    <Box sx={{ mb: 1.5 }}>
                      <Typography variant="subtitle2" color="text.secondary">LoRA Configuration</Typography>
                      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 0.5 }}>
                        {selectedModel.loraConfig.r && (
                          <StatusChip
                            status="lora"
                            label={`Rank: ${selectedModel.loraConfig.r}`}
                            variant="outlined"
                            statusConfig={MODELS_STATUS_CONFIG}
                          />
                        )}
                        {selectedModel.loraConfig.lora_alpha && (
                          <StatusChip
                            status="lora"
                            label={`Alpha: ${selectedModel.loraConfig.lora_alpha}`}
                            variant="outlined"
                            statusConfig={MODELS_STATUS_CONFIG}
                          />
                        )}
                        {selectedModel.loraConfig.lora_dropout != null && (
                          <StatusChip
                            status="lora"
                            label={`Dropout: ${selectedModel.loraConfig.lora_dropout}`}
                            variant="outlined"
                            statusConfig={MODELS_STATUS_CONFIG}
                          />
                        )}
                      </Box>
                    </Box>
                  )}

                  {selectedModel.mergedAt && (
                    <Box>
                      <Typography variant="subtitle2" color="text.secondary">Merged At</Typography>
                      <Typography variant="body2">
                        {new Date(selectedModel.mergedAt).toLocaleString()}
                      </Typography>
                    </Box>
                  )}

                  {selectedModel.trainingJobId && (
                    <Box sx={{ mt: 1.5 }}>
                      <Typography variant="subtitle2" color="text.secondary">Training Job ID</Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                        {selectedModel.trainingJobId}
                      </Typography>
                    </Box>
                  )}
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions
          sx={{
            px: { xs: 2, sm: 3 },
            py: { xs: 1.5, sm: 2 },
            gap: 0.75,
            flexWrap: 'wrap',
            '& .MuiButton-root': {
              minHeight: 32,
              height: 32,
              borderRadius: '8px',
              fontSize: '0.75rem',
              fontWeight: 800,
              lineHeight: 1,
              minWidth: 'auto',
              px: 1.1,
              whiteSpace: 'nowrap',
            },
            '& .MuiButton-startIcon': {
              ml: 0,
              mr: 0.5,
            },
            '& .MuiButton-startIcon > *:nth-of-type(1)': {
              fontSize: 16,
            },
          }}
        >
          {selectedModel && selectedModel.status === 'running' && (
            <Button
              variant="outlined"
              startIcon={<BenchmarkIcon />}
              onClick={() => {
                setInfoDialogOpen(false)
                handleBenchmark(selectedModel)
              }}
            >
              Benchmark
            </Button>
          )}
          {selectedModel && selectedModel.status !== 'downloading' && (
            <Button
              variant="outlined"
              startIcon={<QuantizeIcon />}
              onClick={() => {
                setInfoDialogOpen(false)
                handleQuantize(selectedModel)
              }}
            >
              Quantize
            </Button>
          )}
          {selectedModel && selectedModel.status !== 'downloading' && isFineTunableModel(selectedModel) && (
            <Button
              variant="outlined"
              startIcon={<TuneIcon />}
              onClick={() => {
                setInfoDialogOpen(false)
                handleFineTune(selectedModel)
              }}
            >
              Fine-tune
            </Button>
          )}
          {selectedModel && selectedModel.status !== 'downloading' && (
            <Button
              variant="contained"
              startIcon={<DeployIcon />}
              onClick={() => {
                setInfoDialogOpen(false)
                handleDeploy(selectedModel)
              }}
            >
              {isNarrow ? 'Deploy' : 'Deploy Model'}
            </Button>
          )}
        </DialogActions>
      </Dialog>

      {/* Archive Model Dialog */}
      <Dialog
        open={archiveDialogOpen}
        onClose={() => !archiving && setArchiveDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ fontSize: '1rem', fontWeight: 600 }}>
          Archive Model
        </DialogTitle>
        <DialogContent>
          {modelToArchive && (
            <>
              <Typography variant="body2" sx={{ mb: 3 }}>
                Archive <strong>{modelToArchive.name}</strong>? This will hide it from the main models list but keep your rating and notes for future reference.
              </Typography>

              {/* Rating */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                  Rate this model (optional)
                </Typography>
                <Box display="flex" alignItems="center" gap={0.5}>
                  {[1, 2, 3, 4, 5].map((star) => (
                    <IconButton
                      key={star}
                      size="small"
                      onClick={() => setArchiveRating(star === archiveRating ? 0 : star)}
                      sx={{
                        color: star <= archiveRating ? 'warning.main' : 'grey.300',
                        p: 0.25
                      }}
                    >
                      {star <= archiveRating ? <StarIcon /> : <StarBorderIcon />}
                    </IconButton>
                  ))}
                  {archiveRating > 0 && (
                    <Typography variant="caption" sx={{ ml: 1 }}>
                      {archiveRating} star{archiveRating !== 1 ? 's' : ''}
                    </Typography>
                  )}
                </Box>
              </Box>

              {/* General Notes */}
              <Box sx={{ mb: 3 }}>
                <TextField
                  fullWidth
                  multiline
                  rows={3}
                  label="Notes (optional)"
                  placeholder="General thoughts about this model..."
                  value={archiveNotes}
                  onChange={(e) => setArchiveNotes(e.target.value)}
                  variant="outlined"
                  size="small"
                />
              </Box>

              {/* Performance Notes */}
              <Box sx={{ mb: 3 }}>
                <TextField
                  fullWidth
                  multiline
                  rows={2}
                  label="Performance Notes (optional)"
                  placeholder="Speed, quality, memory usage, etc..."
                  value={performanceNotes}
                  onChange={(e) => setPerformanceNotes(e.target.value)}
                  variant="outlined"
                  size="small"
                />
              </Box>

              {/* Delete Local Files Option */}
              <Box sx={{ mb: 2 }}>
                <Box display="flex" alignItems="center" gap={1}>
                  <input
                    type="checkbox"
                    id="deleteLocalFiles"
                    checked={deleteLocalFiles}
                    onChange={(e) => setDeleteLocalFiles(e.target.checked)}
                  />
                  <label htmlFor="deleteLocalFiles">
                    <Typography variant="body2">
                      Delete local model files to free up disk space
                    </Typography>
                  </label>
                </Box>
                {modelToArchive.size > 0 && deleteLocalFiles && (
                  <Typography variant="caption" color="text.secondary" sx={{ ml: 3, fontSize: '0.75rem' }}>
                    This will free up approximately {formatBytes(modelToArchive.size)}
                  </Typography>
                )}
              </Box>
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => setArchiveDialogOpen(false)}
            disabled={archiving}
            sx={{ fontSize: '0.8rem' }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleArchiveConfirm}
            color="warning"
            variant="contained"
            disabled={archiving}
            startIcon={archiving ? <CircularProgress size={16} /> : <ArchiveIcon />}
            sx={{ fontSize: '0.8rem' }}
          >
            {archiving ? 'Archiving...' : 'Archive Model'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Model Delete Confirmation Dialog */}
      <Dialog
        open={modelDeleteDialogOpen}
        onClose={() => !deletingModel && setModelDeleteDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ fontSize: '1rem', fontWeight: 600 }}>
          Delete Model Permanently?
        </DialogTitle>
        <DialogContent>
          {modelToDelete && (
            <>
              <Typography variant="body2" sx={{ mb: 2 }}>
                Are you sure you want to permanently delete <strong>{modelToDelete.name}</strong>? This action cannot be undone and will remove all associated files and data.
              </Typography>
              <Box sx={{ bgcolor: 'error.50', p: 2, borderRadius: 1, border: '1px solid', borderColor: 'error.200' }}>
                <Typography variant="body2" sx={{ fontWeight: 600, color: 'error.main' }}>
                  {modelToDelete.name}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {modelToDelete.variant || modelToDelete.quantization}
                </Typography>
                {modelToDelete.size > 0 && (
                  <Typography variant="caption" color="text.secondary" display="block">
                    Size: {formatBytes(modelToDelete.size)}
                  </Typography>
                )}
              </Box>
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => setModelDeleteDialogOpen(false)}
            disabled={deletingModel}
            sx={{ fontSize: '0.8rem' }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleModelDeleteConfirm}
            color="error"
            variant="contained"
            disabled={deletingModel}
            startIcon={deletingModel ? <CircularProgress size={16} /> : <DeleteIcon />}
            sx={{ fontSize: '0.8rem' }}
          >
            {deletingModel ? 'Deleting...' : 'Delete Permanently'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete File Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => !deleting && setDeleteDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ fontSize: '1rem', fontWeight: 600 }}>
          Delete Model File?
        </DialogTitle>
        <DialogContent>
          {fileToDelete && (
            <>
              <Typography variant="body2" sx={{ mb: 2 }}>
                Are you sure you want to delete this file? This action cannot be undone.
              </Typography>
              <Box sx={{ bgcolor: 'grey.100', p: 2, borderRadius: 1, fontFamily: 'monospace', fontSize: '0.85rem' }}>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                  {fileToDelete.name}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Size: {formatBytes(fileToDelete.size)}
                </Typography>
              </Box>
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => setDeleteDialogOpen(false)}
            disabled={deleting}
            sx={{ fontSize: '0.8rem' }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleDeleteConfirm}
            color="error"
            variant="contained"
            disabled={deleting}
            startIcon={deleting ? <CircularProgress size={16} /> : <DeleteIcon />}
            sx={{ fontSize: '0.8rem' }}
          >
            {deleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </PageShell>
  )
}
