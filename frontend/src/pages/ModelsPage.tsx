import React, { useState, useEffect, useMemo, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Typography,
  Box,
  Grid,
  Button,
  Fab,
  Menu,
  MenuItem,
  Alert,
  Snackbar,
  CircularProgress,
  Card,
  CardContent,
  CardActions,
  Chip,
  TextField,
  InputAdornment,
  LinearProgress,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton
} from '@mui/material'
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  CloudDownload as CloudDownloadIcon,
  Search as SearchIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Info as InfoIcon,
  RocketLaunch as DeployIcon,
  Delete as DeleteIcon,
  FolderOpen as FolderIcon,
  Storage as StorageIcon,
  Archive as ArchiveIcon,
  Unarchive as UnarchiveIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon
} from '@mui/icons-material'
import { ModelInfo, ModelDownload, ModelDownloadRequest, ModelArchiveRequest } from '@/types/api'
import { apiService } from '@/services/api'
import { DownloadModelDialog } from '@/components/DownloadModelDialog'

// Local type definitions to replace missing imports
export type ModelFramework = 'transformers' | 'gguf' | 'onnx' | 'pytorch' | 'safetensors';
export type ModelStatus = 'available' | 'downloading' | 'loading' | 'running' | 'stopped' | 'error';

export interface ModelFormData {
  name: string;
  framework: ModelFramework;
  port: number;
  path: string;
}

export const ModelsPage: React.FC = () => {
  const navigate = useNavigate()

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

  // State for add model menu
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null)
  const addMenuOpen = Boolean(anchorEl)

  // State for download dialog
  const [downloadDialogOpen, setDownloadDialogOpen] = useState(false)

  // State for model info dialog
  const [infoDialogOpen, setInfoDialogOpen] = useState(false)
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null)

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
          const updatedModels = prev.map(model => {
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

              updatedModels.push({
                id: prev.length + updatedModels.length + 1,
                name: modelName,
                framework: 'transformers' as any, // Cast to any or specific type if known
                status: download.status === 'completed' ? 'available' : 'downloading',
                downloadProgress: download.status === 'completed' ? undefined : download.progress,
                parameters: modelName.includes('30B') ? '30B' :
                  modelName.includes('7B') ? '7B' :
                    modelName.includes('13B') ? '13B' : '?B',
                quantization: quantization || 'unknown',
                contextLength: 32768
              } as any) // Cast to any to bypass strict type checking for now
            }
          })

          return updatedModels
        })
      }
    } catch (error) {
      console.error('Failed to fetch downloads:', error)
    }
  }

  // Open add model menu
  const handleAddMenuClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget)
  }

  // Close add model menu
  const handleAddMenuClose = () => {
    setAnchorEl(null)
  }

  // Open deploy modal with specific type
  const handleDeployTypeSelect = (type: 'huggingface' | 'local' | 'docker') => {
    handleAddMenuClose()

    if (type === 'huggingface') {
      setDownloadDialogOpen(true)
    } else {
      setSnackbar({
        open: true,
        message: `Deploy feature (${type}) coming soon!`,
        severity: 'info'
      })
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
  const handleStartModel = async (id: number) => {
    const model = models.find(m => m.id === id)
    if (!model) return

    try {
      setSnackbar({
        open: true,
        message: `Starting ${model.name}...`,
        severity: 'info'
      })
      // TODO: Call backend API to start this specific model
      // For now, navigate to deploy page to configure and start
      navigate('/deploy', { state: { model } })
    } catch (error) {
      setSnackbar({
        open: true,
        message: `Failed to start ${model.name}`,
        severity: 'error'
      })
    }
  }

  const handleStopModel = async (id: number) => {
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

  const handleDeploy = (model: ModelInfo) => {
    navigate('/deploy', { state: { model } })
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
      const result = await apiService.deleteModel(modelToDelete.id)

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

  const formatSpeed = (bytesPerSecond: number) => {
    return formatBytes(bytesPerSecond) + '/s'
  }

  const formatETA = (seconds: number) => {
    if (seconds < 60) return `${Math.round(seconds)}s`
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`
    return `${Math.round(seconds / 3600)}h`
  }

  // Filter models based on search term and active filter
  const filteredModels = useMemo(() => {
    return models.filter(model => {
      // Filter by search term
      const matchesSearch = model.name?.toLowerCase().includes(searchTerm?.toLowerCase() || '') ||
        (model.variant?.toLowerCase().includes(searchTerm?.toLowerCase() || '') || false);

      // Filter by framework
      // const matchesFilter = activeFilter === 'all' || model.framework === activeFilter;
      const matchesFilter = true; // Simplified for now

      return matchesSearch && matchesFilter;
    });
  }, [models, searchTerm, activeFilter]);

  // Handle snackbar close
  const handleSnackbarClose = () => {
    setSnackbar(prev => ({ ...prev, open: false }))
  }

  // Sample models for development
  const getSampleModels = () => [
    {
      id: 1,
      name: 'Qwen3-Coder-30B',
      framework: 'transformers' as ModelFramework,
      status: 'running' as ModelStatus,
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
      framework: 'transformers' as ModelFramework,
      status: 'stopped' as ModelStatus,
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
      framework: 'transformers' as ModelFramework,
      status: 'stopped' as ModelStatus,
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
    <Box sx={{
      width: '100%',
      overflow: 'hidden',
      px: 3,
      py: 2
    }}>
      {/* Modern Header Section */}
      <Box sx={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        mb: 2,
        flexDirection: { xs: 'column', sm: 'row' },
        gap: 2
      }}>
        {/* Title and Description */}
        <Box>
          <Typography
            variant="h1"
            sx={{
              fontWeight: 700,
              color: 'text.primary',
              mb: 0.5,
              fontSize: { xs: '1.25rem', sm: '1.5rem' },
              lineHeight: 1
            }}
          >
            ML Model Manager
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: 'text.secondary',
              fontSize: '0.8125rem',
              fontWeight: 400,
              maxWidth: '400px'
            }}
          >
            Deploy, manage, and monitor your machine learning models
          </Typography>
        </Box>

        {/* Action Buttons */}
        <Box sx={{
          display: 'flex',
          gap: 1,
          flexWrap: 'wrap',
          alignItems: 'center'
        }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon sx={{ fontSize: '1rem' }} />}
            onClick={() => {
              fetchModels()
              fetchArchivedModels()
              fetchLocalFiles()
            }}
            disabled={isLoading}
            size="small"
            sx={{
              borderColor: 'grey.300',
              color: 'text.primary',
              fontWeight: 500,
              fontSize: '0.75rem',
              '&:hover': {
                borderColor: 'primary.main',
                bgcolor: 'primary.50'
              }
            }}
          >
            {isLoading ? 'Refreshing' : 'Refresh'}
          </Button>

          <Button
            variant="outlined"
            startIcon={<CloudDownloadIcon sx={{ fontSize: '1rem' }} />}
            onClick={() => setDownloadDialogOpen(true)}
            size="small"
            sx={{
              borderColor: 'grey.300',
              color: 'text.primary',
              fontWeight: 500,
              fontSize: '0.75rem',
              '&:hover': {
                borderColor: 'info.main',
                bgcolor: 'info.50'
              }
            }}
          >
            Download
          </Button>

          <Button
            variant="contained"
            startIcon={<AddIcon sx={{ fontSize: '1rem' }} />}
            onClick={handleAddMenuClick}
            size="small"
            sx={{
              fontWeight: 600,
              fontSize: '0.75rem'
            }}
          >
            Add Model
          </Button>
        </Box>

        <Menu
          anchorEl={anchorEl}
          open={addMenuOpen}
          onClose={handleAddMenuClose}
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'right',
          }}
          transformOrigin={{
            vertical: 'top',
            horizontal: 'right',
          }}
          PaperProps={{
            sx: {
              mt: 0.5,
              borderRadius: 0.5,
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
              border: '1px solid',
              borderColor: 'grey.200',
              minWidth: 200
            }
          }}
        >
          <MenuItem
            onClick={() => handleDeployTypeSelect('huggingface')}
            sx={{
              px: 1.5,
              py: 1,
              fontSize: '0.8125rem',
              '&:hover': { bgcolor: 'grey.50' }
            }}
          >
            <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
              <Typography sx={{ fontSize: '0.8125rem', fontWeight: 500 }}>HuggingFace Model</Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.6875rem' }}>
                Deploy from huggingface.co
              </Typography>
            </Box>
          </MenuItem>
          <MenuItem
            onClick={() => handleDeployTypeSelect('local')}
            sx={{
              px: 1.5,
              py: 1,
              fontSize: '0.8125rem',
              '&:hover': { bgcolor: 'grey.50' }
            }}
          >
            <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
              <Typography sx={{ fontSize: '0.8125rem', fontWeight: 500 }}>Local Model</Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.6875rem' }}>
                Deploy from local file system
              </Typography>
            </Box>
          </MenuItem>
          <MenuItem
            onClick={() => handleDeployTypeSelect('docker')}
            sx={{
              px: 1.5,
              py: 1,
              fontSize: '0.8125rem',
              '&:hover': { bgcolor: 'grey.50' }
            }}
          >
            <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
              <Typography sx={{ fontSize: '0.8125rem', fontWeight: 500 }}>Docker Container</Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.6875rem' }}>
                Deploy using Docker image
              </Typography>
            </Box>
          </MenuItem>
        </Menu>
      </Box>

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
        <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
          <Tab
            label={
              <Box display="flex" alignItems="center" gap={1}>
                <StorageIcon sx={{ fontSize: '1rem' }} />
                <span>Models</span>
                {models.length > 0 && (
                  <Chip label={models.length} size="small" sx={{ height: 18, fontSize: '0.7rem' }} />
                )}
              </Box>
            }
            sx={{ fontSize: '0.8rem', textTransform: 'none', fontWeight: 500 }}
          />
          <Tab
            label={
              <Box display="flex" alignItems="center" gap={1}>
                <FolderIcon sx={{ fontSize: '1rem' }} />
                <span>Downloaded Files</span>
                {localFiles.length > 0 && (
                  <Chip label={localFiles.length} size="small" sx={{ height: 18, fontSize: '0.7rem' }} />
                )}
              </Box>
            }
            sx={{ fontSize: '0.8rem', textTransform: 'none', fontWeight: 500 }}
          />
          <Tab
            label={
              <Box display="flex" alignItems="center" gap={1}>
                <ArchiveIcon sx={{ fontSize: '1rem' }} />
                <span>Archived</span>
                {archivedModels.length > 0 && (
                  <Chip label={archivedModels.length} size="small" sx={{ height: 18, fontSize: '0.7rem' }} />
                )}
              </Box>
            }
            sx={{ fontSize: '0.8rem', textTransform: 'none', fontWeight: 500 }}
          />
        </Tabs>
      </Box>

      {/* Search bar - only show on models tab */}
      {activeTab === 0 && (
        <Box sx={{ mb: 3 }}>
          <TextField
            fullWidth
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
            sx={{ maxWidth: 500 }}
          />
        </Box>
      )}

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
        <>
          {isLoading && models.length === 0 ? (
            <Box sx={{
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              height: 300,
              bgcolor: 'background.paper',
              borderRadius: 0.5,
              border: '1px solid',
              borderColor: 'grey.200'
            }}>
              <CircularProgress size={32} sx={{ mb: 1.5 }} />
              <Typography variant="body1" color="text.secondary" sx={{ mb: 0.5, fontSize: '0.875rem' }}>
                Loading models...
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.75rem' }}>
                Please wait while we fetch your ML models
              </Typography>
            </Box>
          ) : (
            <Grid container spacing={1.5} sx={{ px: 0 }}>
              {filteredModels.map((model: any) => {
                const download = activeDownloads.find(d => d.modelId.includes(model.name))
                const isDownloading = model.status === 'downloading' || download

                return (
                  <Grid item key={model.id || model.name} xs={12} sm={6} md={4} lg={3} xl={2}>
                    <Card variant="outlined" sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                      <CardContent sx={{ flexGrow: 1 }}>
                        <Typography variant="h6" noWrap title={model.name} sx={{ fontSize: '0.95rem', fontWeight: 600 }}>
                          {model.name}
                        </Typography>
                        <Typography color="textSecondary" gutterBottom sx={{ fontSize: '0.75rem' }}>
                          {model.quantization || model.variant}
                        </Typography>

                        <Box display="flex" alignItems="center" gap={0.5} mb={1} flexWrap="wrap">
                          <Chip
                            label={model.status}
                            size="small"
                            color={model.status === 'running' ? 'success' : model.status === 'downloading' ? 'info' : 'default'}
                            sx={{ fontSize: '0.7rem', height: 20 }}
                          />
                          {model.source === 'trained' && (
                            <Chip
                              label="Trained"
                              size="small"
                              sx={{
                                fontSize: '0.65rem',
                                height: 18,
                                bgcolor: '#059669',
                                color: 'white',
                                fontWeight: 600
                              }}
                            />
                          )}
                          {model.parameters && (
                            <Chip label={model.parameters} size="small" variant="outlined" sx={{ fontSize: '0.7rem', height: 20 }} />
                          )}
                        </Box>

                        {/* Download Progress */}
                        {isDownloading && (
                          <Box sx={{ mt: 1 }}>
                            <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
                              <Typography variant="caption" color="text.secondary">
                                Downloading...
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {download ? `${Math.round(download.progress)}%` : `${Math.round(model.downloadProgress || 0)}%`}
                              </Typography>
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={download?.progress || model.downloadProgress || 0}
                              sx={{ height: 4, borderRadius: 2 }}
                            />
                            {download && (
                              <Box display="flex" justifyContent="space-between" mt={0.5}>
                                <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                                  {formatSpeed(download.speed)}
                                </Typography>
                                <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                                  ETA: {formatETA(download.eta)}
                                </Typography>
                              </Box>
                            )}
                          </Box>
                        )}

                        {/* Model Info */}
                        {model.size > 0 && (
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5, fontSize: '0.7rem' }}>
                            Size: {formatBytes(model.size)}
                          </Typography>
                        )}

                        {/* Local File Link */}
                        {model.localPath && (
                          <Tooltip title={`View file: ${model.filename || model.localPath}`}>
                            <Box
                              onClick={() => handleLocalFileClick(model.localPath!)}
                              sx={{
                                mt: 0.5,
                                display: 'flex',
                                alignItems: 'center',
                                gap: 0.5,
                                cursor: 'pointer',
                                '&:hover': {
                                  '& .MuiTypography-root': {
                                    color: 'primary.main',
                                    textDecoration: 'underline'
                                  }
                                }
                              }}
                            >
                              <FolderIcon sx={{ fontSize: '0.8rem', color: 'text.secondary' }} />
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{
                                  fontSize: '0.65rem',
                                  fontFamily: 'monospace',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  whiteSpace: 'nowrap',
                                  maxWidth: '150px'
                                }}
                              >
                                {model.filename || model.localPath}
                              </Typography>
                            </Box>
                          </Tooltip>
                        )}
                      </CardContent>
                      <CardActions sx={{ pt: 0, pb: 1, px: 1, gap: 0.5, flexWrap: 'wrap' }}>
                        <Button
                          size="small"
                          onClick={() => handleShowInfo(model)}
                          startIcon={<InfoIcon sx={{ fontSize: '0.9rem' }} />}
                          sx={{ fontSize: '0.7rem', minWidth: 'auto', px: 0.5 }}
                        >
                          Info
                        </Button>
                        {!isDownloading && model.status !== 'running' && (
                          <Tooltip title="Deploy this model">
                            <Button
                              size="small"
                              onClick={() => handleDeploy(model)}
                              color="primary"
                              startIcon={<DeployIcon sx={{ fontSize: '0.9rem' }} />}
                              sx={{ fontSize: '0.7rem', minWidth: 'auto', px: 0.5 }}
                            >
                              Deploy
                            </Button>
                          </Tooltip>
                        )}
                        {model.status === 'stopped' && !isDownloading && (
                          <Button
                            size="small"
                            onClick={() => handleStartModel(model.id)}
                            color="success"
                            startIcon={<PlayIcon sx={{ fontSize: '0.9rem' }} />}
                            sx={{ fontSize: '0.7rem', minWidth: 'auto', px: 0.5 }}
                          >
                            Start
                          </Button>
                        )}
                        {model.status === 'running' && (
                          <Button
                            size="small"
                            onClick={() => handleStopModel(model.id)}
                            color="error"
                            startIcon={<StopIcon sx={{ fontSize: '0.9rem' }} />}
                            sx={{ fontSize: '0.7rem', minWidth: 'auto', px: 0.5 }}
                          >
                            Stop
                          </Button>
                        )}
                        {!isDownloading && model.status !== 'running' && (
                          <Tooltip title="Archive this model">
                            <Button
                              size="small"
                              onClick={() => handleArchiveClick(model)}
                              color="warning"
                              startIcon={<ArchiveIcon sx={{ fontSize: '0.9rem' }} />}
                              sx={{ fontSize: '0.7rem', minWidth: 'auto', px: 0.5 }}
                            >
                              Archive
                            </Button>
                          </Tooltip>
                        )}
                        {!isDownloading && model.status !== 'running' && (
                          <Tooltip title="Delete this model permanently">
                            <Button
                              size="small"
                              onClick={() => handleModelDeleteClick(model)}
                              color="error"
                              startIcon={<DeleteIcon sx={{ fontSize: '0.9rem' }} />}
                              sx={{ fontSize: '0.7rem', minWidth: 'auto', px: 0.5 }}
                            >
                              Delete
                            </Button>
                          </Tooltip>
                        )}
                      </CardActions>
                    </Card>
                  </Grid>
                )
              })}

              {filteredModels.length === 0 && (
                <Grid item xs={12}>
                  <Box sx={{
                    textAlign: 'center',
                    py: 6,
                    bgcolor: 'background.paper',
                    borderRadius: 0.5,
                    border: '1px dashed',
                    borderColor: 'grey.300'
                  }}>
                    <Typography variant="h6" color="text.secondary" sx={{ mb: 0.5, fontWeight: 600, fontSize: '1rem' }}>
                      No models found
                    </Typography>
                    <Button
                      variant="contained"
                      startIcon={<AddIcon sx={{ fontSize: '1rem' }} />}
                      onClick={handleAddMenuClick}
                      size="small"
                      sx={{
                        fontWeight: 600,
                        fontSize: '0.75rem',
                        mt: 2
                      }}
                    >
                      Add Your First Model
                    </Button>
                  </Box>
                </Grid>
              )}
            </Grid>
          )}
        </>
      )}

      {/* Archived Models Tab */}
      {activeTab === 2 && (
        <Box>
          {archivedModels.length === 0 ? (
            <Box sx={{
              textAlign: 'center',
              py: 6,
              bgcolor: 'background.paper',
              borderRadius: 0.5,
              border: '1px dashed',
              borderColor: 'grey.300'
            }}>
              <ArchiveIcon sx={{ fontSize: '3rem', color: 'text.secondary', mb: 2 }} />
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
                  <Card variant="outlined" sx={{ height: '100%', display: 'flex', flexDirection: 'column', opacity: 0.8 }}>
                    <CardContent sx={{ flexGrow: 1 }}>
                      <Typography variant="h6" noWrap title={model.name} sx={{ fontSize: '0.95rem', fontWeight: 600 }}>
                        {model.name}
                      </Typography>
                      <Typography color="textSecondary" gutterBottom sx={{ fontSize: '0.75rem' }}>
                        {model.quantization || model.variant}
                      </Typography>

                      <Box display="flex" alignItems="center" gap={0.5} mb={1} flexWrap="wrap">
                        <Chip
                          label="archived"
                          size="small"
                          color="warning"
                          sx={{ fontSize: '0.7rem', height: 20 }}
                        />
                        {model.parameters && (
                          <Chip label={model.parameters} size="small" variant="outlined" sx={{ fontSize: '0.7rem', height: 20 }} />
                        )}
                      </Box>

                      {/* Rating Display */}
                      {model.archiveRating && (
                        <Box display="flex" alignItems="center" gap={0.5} mb={1}>
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
                        <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5, fontSize: '0.7rem', fontStyle: 'italic' }}>
                          "{model.archiveNotes}"
                        </Typography>
                      )}

                      {/* Performance Notes */}
                      {model.performanceNotes && (
                        <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5, fontSize: '0.7rem' }}>
                          Performance: {model.performanceNotes}
                        </Typography>
                      )}

                      {/* Archive Date */}
                      {model.archivedAt && (
                        <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5, fontSize: '0.65rem' }}>
                          Archived: {new Date(model.archivedAt).toLocaleDateString()}
                        </Typography>
                      )}
                    </CardContent>
                    <CardActions sx={{ pt: 0, pb: 1, px: 1, gap: 0.5 }}>
                      <Button
                        size="small"
                        onClick={() => handleShowInfo(model)}
                        startIcon={<InfoIcon sx={{ fontSize: '0.9rem' }} />}
                        sx={{ fontSize: '0.7rem', minWidth: 'auto', px: 0.5 }}
                      >
                        Info
                      </Button>
                      <Tooltip title="Restore this model">
                        <Button
                          size="small"
                          onClick={() => handleUnarchiveModel(model)}
                          color="success"
                          startIcon={<UnarchiveIcon sx={{ fontSize: '0.9rem' }} />}
                          sx={{ fontSize: '0.7rem', minWidth: 'auto', px: 0.5 }}
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
                          sx={{ fontSize: '0.7rem', minWidth: 'auto', px: 0.5 }}
                        >
                          Delete
                        </Button>
                      </Tooltip>
                    </CardActions>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}
        </Box>
      )}

      {/* Downloaded Files Tab */}
      {activeTab === 1 && (
        <Box>
          {/* Disk Usage Summary */}
          <Card variant="outlined" sx={{ mb: 2, bgcolor: 'background.paper' }}>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Box>
                  <Typography variant="subtitle2" color="text.secondary" sx={{ fontSize: '0.8rem' }}>
                    Total Disk Usage
                  </Typography>
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    {formatBytes(totalDiskUsage)}
                  </Typography>
                </Box>
                <Box textAlign="right">
                  <Typography variant="subtitle2" color="text.secondary" sx={{ fontSize: '0.8rem' }}>
                    Downloaded Files
                  </Typography>
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    {localFiles.length}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>

          {/* Files Table */}
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow sx={{ bgcolor: 'grey.50' }}>
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
                        <Chip
                          label={file.extension.toUpperCase()}
                          size="small"
                          variant="outlined"
                          sx={{ fontSize: '0.7rem', height: 20 }}
                        />
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
        </Box>
      )}

      {/* Floating action button for mobile */}
      <Fab
        color="primary"
        aria-label="download model"
        size="small"
        sx={{
          position: 'fixed',
          bottom: 16,
          right: 16,
          display: { md: 'none' },
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
          '&:hover': {
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)'
          }
        }}
        onClick={() => setDownloadDialogOpen(true)}
      >
        <CloudDownloadIcon sx={{ fontSize: '1.25rem' }} />
      </Fab>

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
      <Dialog open={infoDialogOpen} onClose={() => setInfoDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          <Typography variant="h6">{selectedModel?.name}</Typography>
          <Typography variant="caption" color="text.secondary">
            {selectedModel?.variant || selectedModel?.quantization}
          </Typography>
        </DialogTitle>
        <DialogContent>
          {selectedModel && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box>
                <Typography variant="subtitle2" color="text.secondary">Status</Typography>
                <Chip
                  label={selectedModel.status}
                  size="small"
                  color={selectedModel.status === 'running' ? 'success' : selectedModel.status === 'downloading' ? 'info' : 'default'}
                  sx={{ mt: 0.5 }}
                />
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
                    <Chip
                      label="Fine-Tuned Model"
                      size="small"
                      sx={{
                        bgcolor: '#059669',
                        color: 'white',
                        fontWeight: 600,
                        fontSize: '0.7rem'
                      }}
                    />
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
                          <Chip label={`Rank: ${selectedModel.loraConfig.r}`} size="small" variant="outlined" sx={{ fontSize: '0.7rem' }} />
                        )}
                        {selectedModel.loraConfig.lora_alpha && (
                          <Chip label={`Alpha: ${selectedModel.loraConfig.lora_alpha}`} size="small" variant="outlined" sx={{ fontSize: '0.7rem' }} />
                        )}
                        {selectedModel.loraConfig.lora_dropout != null && (
                          <Chip label={`Dropout: ${selectedModel.loraConfig.lora_dropout}`} size="small" variant="outlined" sx={{ fontSize: '0.7rem' }} />
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
        <DialogActions>
          <Button onClick={() => setInfoDialogOpen(false)}>Close</Button>
          {selectedModel && selectedModel.status !== 'downloading' && (
            <Button
              variant="contained"
              startIcon={<DeployIcon />}
              onClick={() => {
                setInfoDialogOpen(false)
                handleDeploy(selectedModel)
              }}
            >
              Deploy Model
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
    </Box>
  )
}
