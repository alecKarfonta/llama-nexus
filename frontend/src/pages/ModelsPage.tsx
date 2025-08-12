import React, { useState, useEffect, useMemo } from 'react'
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
  CircularProgress
} from '@mui/material'
import { 
  Add as AddIcon, 
  Refresh as RefreshIcon,
  CloudDownload as CloudDownloadIcon
} from '@mui/icons-material'
import { ModelCard, ModelFramework, ModelStatus } from '@/components/models/ModelCard'
import { ModelFilters } from '@/components/models/ModelFilters'
import { ModelDeployModal, ModelFormData } from '@/components/models/ModelDeployModal'
import { ModelInfoModal } from '@/components/models/ModelInfoModal'
import { ModelSwitchModal } from '@/components/models/ModelSwitchModal'
import { ModelDownloadModal } from '@/components/models/ModelDownloadModal'
import { ModelInfo, ModelDownload, ModelDownloadRequest } from '@/types/api'
import { apiService } from '@/services/api'

export const ModelsPage: React.FC = () => {
  // State for filtering and models
  const [searchTerm, setSearchTerm] = useState('')
  const [activeFilter, setActiveFilter] = useState('all')
  const [models, setModels] = useState<ModelInfo[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [activeDownloads, setActiveDownloads] = useState<ModelDownload[]>([])
  const [refreshInterval, setRefreshInterval] = useState<number | null>(null)
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info' | 'warning';
  }>({
    open: false,
    message: '',
    severity: 'info'
  })
  
  // State for model deployment modal
  const [deployModalOpen, setDeployModalOpen] = useState(false)
  const [deployType, setDeployType] = useState<'huggingface' | 'local' | 'docker'>('huggingface')
  
  // State for model info modal
  const [infoModalOpen, setInfoModalOpen] = useState(false)
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null)
  
  // State for model switch modal
  const [switchModalOpen, setSwitchModalOpen] = useState(false)
  
  // State for model download modal
  const [downloadModalOpen, setDownloadModalOpen] = useState(false)
  
  // State for add model menu
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null)
  const addMenuOpen = Boolean(anchorEl)

  // Fetch models on component mount
  useEffect(() => {
    fetchModels()
    
    // Start polling for active downloads
    const interval = window.setInterval(() => {
      if (activeDownloads.length > 0) {
        fetchDownloads()
      }
    }, 3000)
    
    setRefreshInterval(interval)
    
    return () => {
      if (refreshInterval) {
        window.clearInterval(refreshInterval)
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
                framework: 'transformers',
                status: download.status === 'completed' ? 'available' : 'downloading',
                downloadProgress: download.status === 'completed' ? undefined : download.progress,
                parameters: modelName.includes('30B') ? '30B' : 
                          modelName.includes('7B') ? '7B' : 
                          modelName.includes('13B') ? '13B' : '?B',
                quantization: quantization || 'unknown',
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
    setDeployType(type)
    setDeployModalOpen(true)
    handleAddMenuClose()
  }
  
  // Model action handlers
  const handleStartModel = async (id: number) => {
    const model = models.find(m => m.id === id)
    if (!model) return
    
    // Update UI immediately
    setModels(models.map(m => {
      if (m.id === id) {
        return { ...m, status: 'loading' as ModelStatus }
      }
      return m
    }))
    
    try {
      // In a real implementation, this would call the API
      await apiService.performServiceAction({
        action: 'start',
        config: {
          model: {
            name: model.name,
            variant: model.quantization || 'Q4_K_M',
            contextSize: model.contextLength || 32768,
            gpuLayers: 999
          }
        }
      })
      
      // Update model status after successful API call
      setTimeout(() => {
        setModels(prev => prev.map(m => {
          if (m.id === id) {
            return {
              ...m,
              status: 'running' as ModelStatus,
              latency: Math.floor(Math.random() * 200 + 50) + 'ms',
              memory: (Math.random() * 2 + 0.5).toFixed(1) + 'GB'
            }
          }
          return m
        }))
        
        setSnackbar({
          open: true,
          message: `${model.name} started successfully`,
          severity: 'success'
        })
      }, 2000)
    } catch (error) {
      // Handle error
      setModels(prev => prev.map(m => {
        if (m.id === id) {
          return { ...m, status: 'stopped' as ModelStatus }
        }
        return m
      }))
      
      setSnackbar({
        open: true,
        message: `Failed to start ${model.name}: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      })
    }
  }
  
  const handleStopModel = async (id: number) => {
    const model = models.find(m => m.id === id)
    if (!model) return
    
    // Update UI immediately
    setModels(models.map(m => {
      if (m.id === id) {
        return { 
          ...m, 
          status: 'loading' as ModelStatus,
          latency: '--',
          memory: '--'
        }
      }
      return m
    }))
    
    try {
      // In a real implementation, this would call the API
      await apiService.performServiceAction({
        action: 'stop'
      })
      
      // Update model status after successful API call
      setTimeout(() => {
        setModels(prev => prev.map(m => {
          if (m.id === id) {
            return {
              ...m,
              status: 'stopped' as ModelStatus
            }
          }
          return m
        }))
        
        setSnackbar({
          open: true,
          message: `${model.name} stopped successfully`,
          severity: 'success'
        })
      }, 1500)
    } catch (error) {
      // Handle error
      setModels(prev => prev.map(m => {
        if (m.id === id) {
          return { ...m, status: 'running' as ModelStatus }
        }
        return m
      }))
      
      setSnackbar({
        open: true,
        message: `Failed to stop ${model.name}: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      })
    }
  }
  
  const handleTestModel = (id: number) => {
    const model = models.find(m => m.id === id)
    if (!model) return
    
    // In a real application, this would open a test interface
    setSnackbar({
      open: true,
      message: `Testing ${model.name}...`,
      severity: 'info'
    })
  }
  
  const handleShowLogs = (id: number) => {
    const model = models.find(m => m.id === id)
    if (!model) return
    
    // In a real application, this would open a logs modal
    setSnackbar({
      open: true,
      message: `Showing logs for ${model.name}...`,
      severity: 'info'
    })
  }
  
  const handleShowInfo = (id: number) => {
    const model = models.find(m => m.id === id)
    if (!model) return
    
    // Convert to ModelInfo type
    const modelInfo: ModelInfo = {
      name: model.name,
      variant: model.quantization || 'unknown',
      size: 10 * 1024 * 1024 * 1024, // 10GB placeholder
      status: model.status === 'running' ? 'available' : model.status === 'stopped' ? 'available' : 'downloading',
      parameters: model.parameters || '30B',
      quantization: model.quantization || 'Q4_K_M',
      contextLength: model.contextLength || 32768,
      description: `${model.name} is a large language model optimized for ${model.framework} applications.`,
      vramRequired: 16,
      lastModified: new Date(),
      repositoryId: model.path?.includes('/') ? model.path : `huggingface/${model.path}`
    }
    
    setSelectedModel(modelInfo)
    setInfoModalOpen(true)
  }
  
  const handleSwitchModel = (id: number) => {
    const model = models.find(m => m.id === id)
    if (!model) return
    
    setSwitchModalOpen(true)
  }
  
  const handleDownloadModel = (id: number) => {
    const model = models.find(m => m.id === id)
    if (!model) return
    
    setDownloadModalOpen(true)
  }
  
  // Handle model deployment
  const handleModelDeploy = (modelData: ModelFormData) => {
    // In a real application, this would make an API call to deploy the model
    const newModel = {
      id: models.length + 1,
      name: modelData.name,
      framework: modelData.framework,
      status: 'running' as ModelStatus,
      port: modelData.port,
      latency: Math.floor(Math.random() * 200 + 50) + 'ms',
      memory: (Math.random() * 2 + 0.5).toFixed(1) + 'GB',
      path: modelData.path,
      parameters: '7B',
      quantization: 'Q4_K_M',
      contextLength: 32768
    }
    
    setModels(prev => [...prev, newModel])
    
    setSnackbar({
      open: true,
      message: `${modelData.name} deployed successfully`,
      severity: 'success'
    })
  }
  
  // Handle model switch
  const handleModelSwitch = async (modelId: string) => {
    try {
      // In a real implementation, this would call the API
      await apiService.performServiceAction({
        action: 'restart',
        config: {
          model: {
            name: modelId,
            variant: 'Q4_K_M',
            contextSize: 32768,
            gpuLayers: 999
          }
        }
      })
      
      // Update models after successful switch
      setModels(prev => prev.map(model => {
        if (model.name === modelId) {
          return { 
            ...model, 
            status: 'running' as ModelStatus,
            latency: Math.floor(Math.random() * 200 + 50) + 'ms',
            memory: (Math.random() * 2 + 0.5).toFixed(1) + 'GB'
          }
        } else if (model.status === 'running') {
          return { 
            ...model, 
            status: 'stopped' as ModelStatus,
            latency: '--',
            memory: '--'
          }
        }
        return model
      }))
      
      setSnackbar({
        open: true,
        message: `Switched to ${modelId} successfully`,
        severity: 'success'
      })
    } catch (error) {
      setSnackbar({
        open: true,
        message: `Failed to switch model: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      })
      throw error
    }
  }
  
  // Handle model download
  const handleModelDownload = async (request: ModelDownloadRequest) => {
    try {
      // In a real implementation, this would call the API
      await apiService.downloadModel(request)
      
      // Extract model name from filename
      const modelName = request.filename.split('.')[0]
      
      // Add new model to the list
      const newModel: ModelInfo = {
        name: modelName,
        variant: request.filename.includes('Q') ? request.filename.split('.')[1] : 'Q4_K_M',
        size: 10 * 1024 * 1024 * 1024, // 10GB placeholder
        status: 'downloading',
        downloadProgress: 0,
        repositoryId: request.repositoryId,
        parameters: request.repositoryId.includes('30B') ? '30B' : 
                   request.repositoryId.includes('7B') ? '7B' : 
                   request.repositoryId.includes('13B') ? '13B' : '?B',
        quantization: request.filename.includes('Q') ? request.filename.split('.')[1] : 'Q4_K_M',
        contextLength: 32768
      }
      
      // Convert to model card format
      const newModelCard = {
        id: models.length + 1,
        name: modelName,
        framework: 'transformers' as ModelFramework,
        status: 'loading' as ModelStatus,
        path: request.repositoryId,
        parameters: newModel.parameters,
        quantization: newModel.quantization,
        contextLength: newModel.contextLength,
        downloadProgress: 0
      }
      
      setModels(prev => [...prev, newModelCard])
      
      // Add to active downloads
      setActiveDownloads(prev => [...prev, {
        modelId: modelName,
        progress: 0,
        status: 'downloading',
        totalSize: 10 * 1024 * 1024 * 1024, // 10GB placeholder
        downloadedSize: 0,
        speed: 10 * 1024 * 1024, // 10MB/s placeholder
        eta: 1000 // 1000 seconds placeholder
      }])
      
      setSnackbar({
        open: true,
        message: `Started downloading ${modelName}`,
        severity: 'success'
      })
      
      // Simulate download progress in development
      if (process.env.NODE_ENV === 'development') {
        let progress = 0
        const interval = window.setInterval(() => {
          progress += Math.random() * 5
          if (progress >= 100) {
            progress = 100
            clearInterval(interval)
            
            // Update model status but keep it in the list
            setModels(prev => prev.map(model => {
              if (model.name === modelName) {
                return { 
                  ...model, 
                  status: 'available' as ModelStatus,
                  downloadProgress: undefined
                }
              }
              return model
            }))
            
            // Update active downloads status to completed instead of removing
            setActiveDownloads(prev => prev.map(d => {
              if (d.modelId === modelName) {
                return {
                  ...d,
                  status: 'completed',
                  progress: 100
                }
              }
              return d
            }))
            
            setSnackbar({
              open: true,
              message: `${modelName} downloaded successfully`,
              severity: 'success'
            })
          } else {
            // Update download progress
            setModels(prev => prev.map(model => {
              if (model.name === modelName) {
                return { 
                  ...model, 
                  downloadProgress: progress
                }
              }
              return model
            }))
            
            // Update active downloads
            setActiveDownloads(prev => prev.map(d => {
              if (d.modelId === modelName) {
                const downloadedSize = (d.totalSize * progress) / 100
                return {
                  ...d,
                  progress,
                  downloadedSize,
                  speed: 10 * 1024 * 1024 + Math.random() * 5 * 1024 * 1024,
                  eta: (d.totalSize - downloadedSize) / (10 * 1024 * 1024)
                }
              }
              return d
            }))
          }
        }, 1000)
      }
    } catch (error) {
      setSnackbar({
        open: true,
        message: `Failed to start download: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      })
      throw error
    }
  }
  
  // Filter models based on search term and active filter
  const filteredModels = useMemo(() => {
    return models.filter(model => {
      // Filter by search term
      const matchesSearch = model.name?.toLowerCase().includes(searchTerm?.toLowerCase() || '') ||
                           (model.framework?.toLowerCase().includes(searchTerm?.toLowerCase() || '') || false);
      
      // Filter by framework
      const matchesFilter = activeFilter === 'all' || model.framework === activeFilter;
      
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
            onClick={fetchModels}
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
            onClick={() => setDownloadModalOpen(true)}
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
      
      <ModelFilters
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
        activeFilter={activeFilter}
        onFilterChange={setActiveFilter}
      />
      
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
          {filteredModels.map((model) => (
            <Grid item key={model.id} xs={12} sm={6} md={4} lg={3} xl={2}>
              <ModelCard
                {...model}
                onStart={handleStartModel}
                onStop={handleStopModel}
                onTest={handleTestModel}
                onLogs={handleShowLogs}
                onInfo={handleShowInfo}
                onSwitch={handleSwitchModel}
                onDownload={handleDownloadModel}
              />
            </Grid>
          ))}
          
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
                <Box
                  component="img"
                  src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='48' height='48' viewBox='0 0 24 24' fill='none' stroke='%239ca3af' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M13 2L3 14h9l-1 8 10-12h-9l1-8z'/%3E%3C/svg%3E"
                  alt="No models"
                  sx={{ 
                    width: 48, 
                    height: 48, 
                    opacity: 0.5,
                    mb: 1.5 
                  }}
                />
                <Typography variant="h6" color="text.secondary" sx={{ mb: 0.5, fontWeight: 600, fontSize: '1rem' }}>
                  No models found
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2, maxWidth: 300, mx: 'auto', fontSize: '0.8125rem' }}>
                  {searchTerm || activeFilter !== 'all' 
                    ? 'No models match your current filters.'
                    : 'Get started by adding your first model.'
                  }
                </Typography>
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
                  Add Your First Model
                </Button>
              </Box>
            </Grid>
          )}
        </Grid>
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
        onClick={() => setDownloadModalOpen(true)}
      >
        <CloudDownloadIcon sx={{ fontSize: '1.25rem' }} />
      </Fab>
      
      {/* Model deploy modal */}
      <ModelDeployModal
        open={deployModalOpen}
        onClose={() => setDeployModalOpen(false)}
        onDeploy={handleModelDeploy}
        deployType={deployType}
      />
      
      {/* Model info modal */}
      <ModelInfoModal
        open={infoModalOpen}
        onClose={() => setInfoModalOpen(false)}
        model={selectedModel}
      />
      
      {/* Model switch modal */}
      <ModelSwitchModal
        open={switchModalOpen}
        onClose={() => setSwitchModalOpen(false)}
        models={models.map(m => ({
          name: m.name,
          variant: m.quantization || 'unknown',
          size: 10 * 1024 * 1024 * 1024, // 10GB placeholder
          status: m.status === 'running' ? 'available' : m.status === 'stopped' ? 'available' : 'downloading',
          parameters: m.parameters,
          quantization: m.quantization,
          contextLength: m.contextLength
        }))}
        currentModelId={models.find(m => m.status === 'running')?.name}
        onSwitch={handleModelSwitch}
      />
      
      {/* Model download modal */}
      <ModelDownloadModal
        open={downloadModalOpen}
        onClose={() => setDownloadModalOpen(false)}
        onDownload={handleModelDownload}
        activeDownloads={activeDownloads}
      />
      
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
    </Box>
  )
}