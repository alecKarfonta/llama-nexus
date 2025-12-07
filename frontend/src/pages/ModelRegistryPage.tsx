import React, { useState, useEffect, useCallback } from 'react'
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
  Tooltip,
  Alert,
  CircularProgress,
  Divider,
  InputAdornment,
  Paper,
  Rating,
  LinearProgress,
  Tab,
  Tabs,
} from '@mui/material'
import {
  Search as SearchIcon,
  Star as StarIcon,
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
  Info as InfoIcon,
  BarChart as StatsIcon,
  Download as DownloadIcon,
  CloudDownload as CloudIcon,
} from '@mui/icons-material'
import { apiService } from '@/services/api'

interface CachedModel {
  id: string
  repo_id: string
  name: string
  description?: string
  author?: string
  downloads: number
  likes: number
  tags: string[]
  model_type?: string
  license?: string
  last_modified?: string
  created_at: string
  updated_at: string
}

interface ModelVariant {
  id: number
  model_id: string
  filename: string
  quantization: string
  size_bytes?: number
  vram_required_mb?: number
  quality_score?: number
  speed_score?: number
}

interface ModelUsage {
  model_id: string
  variant?: string
  load_count: number
  inference_count: number
  total_tokens_generated: number
  last_used?: string
  name?: string
  repo_id?: string
}

interface RegistryStats {
  cached_models: number
  total_variants: number
  total_loads: number
  total_inferences: number
  rated_models: number
}

const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const formatNumber = (num: number): string => {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M'
  if (num >= 1000) return (num / 1000).toFixed(1) + 'K'
  return num.toString()
}

export default function ModelRegistryPage() {
  // State
  const [models, setModels] = useState<CachedModel[]>([])
  const [stats, setStats] = useState<RegistryStats | null>(null)
  const [usageStats, setUsageStats] = useState<ModelUsage[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [tabValue, setTabValue] = useState(0)
  
  // Dialog states
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false)
  const [selectedModel, setSelectedModel] = useState<CachedModel | null>(null)
  const [modelVariants, setModelVariants] = useState<ModelVariant[]>([])
  const [modelRating, setModelRating] = useState<number | null>(null)
  const [ratingNotes, setRatingNotes] = useState('')
  
  // Cache model dialog
  const [cacheDialogOpen, setCacheDialogOpen] = useState(false)
  const [cacheFormData, setCacheFormData] = useState({
    repo_id: '',
    name: '',
    description: '',
    author: '',
    model_type: '',
    license: '',
  })

  // Load data
  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [modelsRes, statsRes, usageRes] = await Promise.all([
        apiService.listCachedModels({
          search: searchQuery || undefined,
          limit: 100,
        }),
        apiService.getRegistryStats(),
        apiService.getModelUsageStats(),
      ])
      setModels(modelsRes.models)
      setStats(statsRes)
      setUsageStats(usageRes.usage)
    } catch (err: any) {
      setError(err.message || 'Failed to load model registry')
    } finally {
      setLoading(false)
    }
  }, [searchQuery])

  useEffect(() => {
    loadData()
  }, [loadData])

  // Handlers
  const handleViewDetails = async (model: CachedModel) => {
    setSelectedModel(model)
    try {
      const [variantsRes, ratingRes] = await Promise.all([
        apiService.getModelVariants(model.repo_id),
        apiService.getModelRating(model.repo_id),
      ])
      setModelVariants(variantsRes.variants)
      setModelRating(ratingRes.rating || null)
      setRatingNotes(ratingRes.notes || '')
      setDetailsDialogOpen(true)
    } catch (err: any) {
      setError(err.message || 'Failed to load model details')
    }
  }

  const handleSaveRating = async () => {
    if (!selectedModel || modelRating === null) return
    try {
      await apiService.setModelRating(selectedModel.repo_id, {
        rating: modelRating,
        notes: ratingNotes,
      })
      loadData()
    } catch (err: any) {
      setError(err.message || 'Failed to save rating')
    }
  }

  const handleDeleteModel = async (model: CachedModel) => {
    if (!confirm(`Delete "${model.name}" from cache?`)) return
    try {
      await apiService.deleteCachedModel(model.repo_id)
      loadData()
    } catch (err: any) {
      setError(err.message || 'Failed to delete model')
    }
  }

  const handleCacheModel = async () => {
    try {
      await apiService.cacheModel(cacheFormData)
      setCacheDialogOpen(false)
      setCacheFormData({
        repo_id: '',
        name: '',
        description: '',
        author: '',
        model_type: '',
        license: '',
      })
      loadData()
    } catch (err: any) {
      setError(err.message || 'Failed to cache model')
    }
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" sx={{ fontWeight: 600 }}>
          Model Registry
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title="Refresh">
            <IconButton onClick={loadData}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
          <Button
            variant="contained"
            startIcon={<CloudIcon />}
            onClick={() => setCacheDialogOpen(true)}
          >
            Cache Model
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
                <MemoryIcon color="primary" />
                <Typography variant="h5">{stats.cached_models}</Typography>
                <Typography variant="caption" color="text.secondary">Cached Models</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={2.4}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <StorageIcon color="secondary" />
                <Typography variant="h5">{stats.total_variants}</Typography>
                <Typography variant="caption" color="text.secondary">Variants</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={2.4}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <DownloadIcon color="info" />
                <Typography variant="h5">{formatNumber(stats.total_loads)}</Typography>
                <Typography variant="caption" color="text.secondary">Total Loads</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={2.4}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <SpeedIcon color="success" />
                <Typography variant="h5">{formatNumber(stats.total_inferences)}</Typography>
                <Typography variant="caption" color="text.secondary">Inferences</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={2.4}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <StarIcon color="warning" />
                <Typography variant="h5">{stats.rated_models}</Typography>
                <Typography variant="caption" color="text.secondary">Rated</Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Tabs */}
      <Tabs value={tabValue} onChange={(_, v) => setTabValue(v)} sx={{ mb: 2 }}>
        <Tab label="Cached Models" />
        <Tab label="Usage Statistics" />
      </Tabs>

      {tabValue === 0 && (
        <>
          {/* Search Bar */}
          <TextField
            fullWidth
            placeholder="Search models..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            sx={{ mb: 2 }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
          />

          {/* Models List */}
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress />
            </Box>
          ) : models.length === 0 ? (
            <Paper sx={{ p: 4, textAlign: 'center' }}>
              <Typography color="text.secondary">
                No cached models found. Cache a model to get started.
              </Typography>
            </Paper>
          ) : (
            <Grid container spacing={2}>
              {models.map(model => (
                <Grid item xs={12} md={6} key={model.id}>
                  <Card sx={{ 
                    '&:hover': { boxShadow: 4 },
                    height: '100%',
                  }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <Box sx={{ flex: 1 }}>
                          <Typography variant="h6" sx={{ fontSize: '1rem', fontWeight: 600 }}>
                            {model.name}
                          </Typography>
                          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                            {model.repo_id}
                          </Typography>
                          {model.description && (
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                              {model.description.length > 100
                                ? model.description.substring(0, 100) + '...'
                                : model.description}
                            </Typography>
                          )}
                          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center' }}>
                            {model.model_type && (
                              <Chip size="small" label={model.model_type} color="primary" variant="outlined" />
                            )}
                            {model.license && (
                              <Chip size="small" label={model.license} variant="outlined" />
                            )}
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              <DownloadIcon sx={{ fontSize: 14, opacity: 0.6 }} />
                              <Typography variant="caption">{formatNumber(model.downloads)}</Typography>
                            </Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              <StarIcon sx={{ fontSize: 14, opacity: 0.6 }} />
                              <Typography variant="caption">{formatNumber(model.likes)}</Typography>
                            </Box>
                          </Box>
                        </Box>
                        <Box sx={{ display: 'flex', gap: 0.5 }}>
                          <Tooltip title="View Details">
                            <IconButton size="small" onClick={() => handleViewDetails(model)}>
                              <InfoIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete from cache">
                            <IconButton size="small" onClick={() => handleDeleteModel(model)} color="error">
                              <DeleteIcon />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}
        </>
      )}

      {tabValue === 1 && (
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>Model Usage Statistics</Typography>
          {usageStats.length === 0 ? (
            <Typography color="text.secondary">No usage data recorded yet.</Typography>
          ) : (
            <List>
              {usageStats.map((usage, index) => (
                <ListItem key={index} divider>
                  <ListItemText
                    primary={usage.name || usage.repo_id || 'Unknown Model'}
                    secondary={
                      <Box sx={{ display: 'flex', gap: 2, mt: 0.5 }}>
                        <Typography variant="caption">
                          Loads: {usage.load_count}
                        </Typography>
                        <Typography variant="caption">
                          Inferences: {usage.inference_count}
                        </Typography>
                        <Typography variant="caption">
                          Tokens: {formatNumber(usage.total_tokens_generated)}
                        </Typography>
                        {usage.last_used && (
                          <Typography variant="caption" color="text.secondary">
                            Last used: {new Date(usage.last_used).toLocaleDateString()}
                          </Typography>
                        )}
                      </Box>
                    }
                  />
                </ListItem>
              ))}
            </List>
          )}
        </Paper>
      )}

      {/* Model Details Dialog */}
      <Dialog open={detailsDialogOpen} onClose={() => setDetailsDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          {selectedModel?.name}
        </DialogTitle>
        <DialogContent>
          {selectedModel && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
              <Typography variant="body2" color="text.secondary">
                {selectedModel.repo_id}
              </Typography>
              
              {selectedModel.description && (
                <Typography variant="body1">
                  {selectedModel.description}
                </Typography>
              )}

              <Divider />

              <Box sx={{ display: 'flex', gap: 2 }}>
                {selectedModel.author && (
                  <Typography variant="body2">
                    <strong>Author:</strong> {selectedModel.author}
                  </Typography>
                )}
                {selectedModel.license && (
                  <Typography variant="body2">
                    <strong>License:</strong> {selectedModel.license}
                  </Typography>
                )}
              </Box>

              <Box sx={{ display: 'flex', gap: 2 }}>
                <Typography variant="body2">
                  <strong>Downloads:</strong> {formatNumber(selectedModel.downloads)}
                </Typography>
                <Typography variant="body2">
                  <strong>Likes:</strong> {formatNumber(selectedModel.likes)}
                </Typography>
              </Box>

              {selectedModel.tags.length > 0 && (
                <Box>
                  <Typography variant="body2" sx={{ mb: 1 }}><strong>Tags:</strong></Typography>
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    {selectedModel.tags.map(tag => (
                      <Chip key={tag} size="small" label={tag} />
                    ))}
                  </Box>
                </Box>
              )}

              <Divider />

              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                Quantization Variants ({modelVariants.length})
              </Typography>
              
              {modelVariants.length === 0 ? (
                <Typography color="text.secondary">No variants recorded.</Typography>
              ) : (
                <List dense>
                  {modelVariants.map(variant => (
                    <ListItem key={variant.id}>
                      <ListItemText
                        primary={variant.filename}
                        secondary={
                          <Box sx={{ display: 'flex', gap: 2 }}>
                            <Chip size="small" label={variant.quantization} color="primary" />
                            {variant.size_bytes && (
                              <Typography variant="caption">
                                Size: {formatBytes(variant.size_bytes)}
                              </Typography>
                            )}
                            {variant.vram_required_mb && (
                              <Typography variant="caption">
                                VRAM: {variant.vram_required_mb} MB
                              </Typography>
                            )}
                          </Box>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              )}

              <Divider />

              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                Your Rating
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Rating
                  value={modelRating}
                  onChange={(_, value) => setModelRating(value)}
                  size="large"
                />
                <Typography variant="body2" color="text.secondary">
                  {modelRating ? `${modelRating} star${modelRating > 1 ? 's' : ''}` : 'Not rated'}
                </Typography>
              </Box>
              <TextField
                label="Notes"
                value={ratingNotes}
                onChange={(e) => setRatingNotes(e.target.value)}
                multiline
                rows={2}
                fullWidth
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsDialogOpen(false)}>Close</Button>
          <Button
            variant="contained"
            onClick={handleSaveRating}
            disabled={modelRating === null}
          >
            Save Rating
          </Button>
        </DialogActions>
      </Dialog>

      {/* Cache Model Dialog */}
      <Dialog open={cacheDialogOpen} onClose={() => setCacheDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Cache Model Metadata</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <TextField
              label="Repository ID"
              value={cacheFormData.repo_id}
              onChange={(e) => setCacheFormData({ ...cacheFormData, repo_id: e.target.value })}
              fullWidth
              required
              placeholder="e.g., meta-llama/Llama-3.1-8B"
            />
            <TextField
              label="Model Name"
              value={cacheFormData.name}
              onChange={(e) => setCacheFormData({ ...cacheFormData, name: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="Description"
              value={cacheFormData.description}
              onChange={(e) => setCacheFormData({ ...cacheFormData, description: e.target.value })}
              fullWidth
              multiline
              rows={2}
            />
            <TextField
              label="Author"
              value={cacheFormData.author}
              onChange={(e) => setCacheFormData({ ...cacheFormData, author: e.target.value })}
              fullWidth
            />
            <TextField
              label="Model Type"
              value={cacheFormData.model_type}
              onChange={(e) => setCacheFormData({ ...cacheFormData, model_type: e.target.value })}
              fullWidth
              placeholder="e.g., text-generation, chat"
            />
            <TextField
              label="License"
              value={cacheFormData.license}
              onChange={(e) => setCacheFormData({ ...cacheFormData, license: e.target.value })}
              fullWidth
              placeholder="e.g., Apache-2.0, MIT"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCacheDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleCacheModel}
            disabled={!cacheFormData.repo_id || !cacheFormData.name}
          >
            Cache Model
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}
