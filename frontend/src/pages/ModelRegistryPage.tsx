import { useState, useEffect, useCallback } from 'react'
import {
  Box,
  Grid,
  Typography,
  TextField,
  Button,
  IconButton,
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
  Rating,
  alpha,
} from '@mui/material'
import {
  Search as SearchIcon,
  Star as StarIcon,
  Memory as MemoryIcon,
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
  Info as InfoIcon,
  BarChart as StatsIcon,
  Download as DownloadIcon,
  CloudDownload as CloudIcon,
} from '@mui/icons-material'
import { ConfirmationDialog, PageShell, SectionCard, StatusChip, type StatusChipConfigMap } from '@/components/common'
import { useConfirmableAction } from '@/hooks/useConfirmableAction'
import { apiService } from '@/services/api'
import { accentColors } from '@/utils/theme'

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

const MODEL_REGISTRY_STATUS_CONFIG: StatusChipConfigMap = {
  catalog: { color: accentColors.info, label: 'Catalog' },
  registry: { color: accentColors.primary, label: 'Registry' },
  type: { color: accentColors.primary, label: 'Type' },
  license: { color: accentColors.slate, label: 'License' },
  usage: { color: accentColors.success, label: 'Usage' },
  coverage: { color: accentColors.warning, label: 'Coverage' },
  tag: { color: accentColors.info, label: 'Tag' },
  quantization: { color: accentColors.primary, label: 'Quantization' },
}

export default function ModelRegistryPage() {
  // State
  const [models, setModels] = useState<CachedModel[]>([])
  const [stats, setStats] = useState<RegistryStats | null>(null)
  const [usageStats, setUsageStats] = useState<ModelUsage[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  
  // Dialog states
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false)
  const [selectedModel, setSelectedModel] = useState<CachedModel | null>(null)
  const [modelVariants, setModelVariants] = useState<ModelVariant[]>([])
  const [modelRating, setModelRating] = useState<number | null>(null)
  const [ratingNotes, setRatingNotes] = useState('')
  const {
    confirmationAction: confirmAction,
    confirmationLoading: confirmingAction,
    requestConfirmation,
    closeConfirmation: closeConfirmDialog,
    confirmAction: handleConfirmAction,
  } = useConfirmableAction()
  
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
  const canCacheModel = cacheFormData.repo_id.trim().length > 0 && cacheFormData.name.trim().length > 0

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

  const deleteModel = async (model: CachedModel) => {
    try {
      await apiService.deleteCachedModel(model.repo_id)
      loadData()
    } catch (err: any) {
      setError(err.message || 'Failed to delete model')
    }
  }

  const handleDeleteModel = (model: CachedModel) => {
    requestConfirmation({
      title: 'Delete cached model?',
      message: `Delete "${model.name}" from cache? The model metadata and usage history will be removed from this registry.`,
      confirmLabel: 'Delete model',
      destructive: true,
      onConfirm: () => deleteModel(model),
    })
  }

  const handleCacheModel = async () => {
    if (!canCacheModel) return

    try {
      await apiService.cacheModel({
        ...cacheFormData,
        repo_id: cacheFormData.repo_id.trim(),
        name: cacheFormData.name.trim(),
      })
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

  const totalTokenCount = usageStats.reduce((total, usage) => total + usage.total_tokens_generated, 0)
  const cachedModelCount = stats?.cached_models ?? models.length
  const ratedModelCount = stats?.rated_models ?? 0
  const ratingCoverage = cachedModelCount > 0 ? Math.round((ratedModelCount / cachedModelCount) * 100) : 0
  const hasSearchQuery = searchQuery.trim().length > 0
  const showRegistryList = loading || cachedModelCount > 0 || models.length > 0 || hasSearchQuery
  const shouldShowRegistrySearch = cachedModelCount > 0 || models.length > 0 || hasSearchQuery
  const prioritizeCacheForm = !showRegistryList
  const showRegistrySummaryRow = shouldShowRegistrySearch
    || cachedModelCount > 0
    || ratedModelCount > 0
    || (stats?.total_variants ?? 0) > 0
    || (stats?.total_inferences ?? 0) > 0
    || totalTokenCount > 0
  const registryRefreshAction = (
    <Tooltip title="Refresh registry">
      <IconButton
        aria-label="Refresh registry"
        size="small"
        onClick={loadData}
        sx={{
          width: 32,
          height: 32,
          borderRadius: '8px',
          border: '1px solid',
          borderColor: 'divider',
          bgcolor: 'action.hover',
        }}
      >
        <RefreshIcon sx={{ fontSize: 18 }} />
      </IconButton>
    </Tooltip>
  )

  return (
    <PageShell sx={{ py: 1.25 }}>
      {error && (
        <Alert severity="error" sx={{ mb: 1.25 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={1.5} alignItems="flex-start">
        {showRegistryList && (
          <Grid
            item
            xs={12}
            lg={8}
            sx={{ order: { xs: 2, lg: prioritizeCacheForm ? 2 : 1 } }}
          >
            <SectionCard
              title="Cached models"
              icon={<MemoryIcon />}
              accentColor={accentColors.primary}
              action={registryRefreshAction}
            >
              {showRegistrySummaryRow && (
                <Box
                  sx={{
                    display: 'grid',
                    gridTemplateColumns: { xs: '1fr', md: 'minmax(0, 1fr) auto' },
                    alignItems: 'center',
                    gap: 0.75,
                    mb: 1,
                  }}
                >
                  <Box sx={{ minWidth: 0, display: 'flex', alignItems: 'center', gap: 0.6, flexWrap: 'wrap' }}>
                    {cachedModelCount > 0 && (
                      <StatusChip status="catalog" label={`${cachedModelCount} cached`} statusConfig={MODEL_REGISTRY_STATUS_CONFIG} height={22} />
                    )}
                    {models.length > 0 && models.length !== cachedModelCount && (
                      <StatusChip status="catalog" label={`${models.length} visible`} statusConfig={MODEL_REGISTRY_STATUS_CONFIG} height={22} />
                    )}
                    {cachedModelCount > 0 && ratedModelCount > 0 && (
                      <StatusChip status="coverage" label={`${ratingCoverage}% rated`} statusConfig={MODEL_REGISTRY_STATUS_CONFIG} height={22} />
                    )}
                    {(stats?.total_variants ?? 0) > 0 && (
                      <StatusChip status="registry" label={`${stats?.total_variants ?? 0} variants`} statusConfig={MODEL_REGISTRY_STATUS_CONFIG} height={22} />
                    )}
                    {(stats?.total_inferences ?? 0) > 0 && (
                      <StatusChip status="usage" label={`${formatNumber(stats?.total_inferences ?? 0)} inferences`} statusConfig={MODEL_REGISTRY_STATUS_CONFIG} height={22} />
                    )}
                    {totalTokenCount > 0 && (
                      <Typography variant="caption" color="text.secondary" sx={{ minWidth: 0 }}>
                        {formatNumber(totalTokenCount)} generated tokens recorded.
                      </Typography>
                    )}
                  </Box>

                  {shouldShowRegistrySearch && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 0, justifyContent: { xs: 'flex-start', md: 'flex-end' } }}>
                      <TextField
                        placeholder="Search cached models"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        InputProps={{
                          startAdornment: (
                            <InputAdornment position="start">
                              <SearchIcon sx={{ fontSize: 18 }} />
                            </InputAdornment>
                          ),
                        }}
                        size="small"
                        sx={{
                          minWidth: { xs: 0, sm: 220 },
                          width: { xs: '100%', sm: 260 },
                          '& .MuiInputBase-root': { height: 32, borderRadius: '8px' },
                        }}
                      />
                    </Box>
                  )}
                </Box>
              )}

              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', py: 3, minHeight: 120 }}>
                  <CircularProgress />
                </Box>
              ) : models.length === 0 ? (
                <Box
                  sx={{
                    p: 1.25,
                    bgcolor: alpha(accentColors.primary, 0.04),
                    borderRadius: '8px',
                    border: '1px dashed',
                    borderColor: 'divider',
                  }}
                >
                  <Typography variant="body2" sx={{ fontWeight: 800 }}>
                    {hasSearchQuery ? 'No matching cached models' : 'No cached models'}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {hasSearchQuery
                      ? 'Adjust the search term or refresh the registry.'
                      : 'Cache a repository.'}
                  </Typography>
                </Box>
              ) : (
                <Grid container spacing={1.25}>
                  {models.map(model => (
                    <Grid item xs={12} md={6} key={model.id}>
                      <Box
                        sx={{
                          height: '100%',
                          p: 1.5,
                          borderRadius: '8px',
                          border: '1px solid',
                          borderColor: 'divider',
                          bgcolor: 'background.paper',
                          transition: 'border-color 0.2s ease, background-color 0.2s ease, transform 0.2s ease',
                          '&:hover': {
                            borderColor: alpha(accentColors.primary, 0.42),
                            bgcolor: alpha(accentColors.primary, 0.05),
                            transform: 'translateY(-1px)',
                          },
                        }}
                      >
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 1.5 }}>
                          <Box sx={{ flex: 1, minWidth: 0 }}>
                            <Typography variant="h6" sx={{ fontSize: '0.95rem', fontWeight: 700, lineHeight: 1.25, wordBreak: 'break-word' }}>
                              {model.name}
                            </Typography>
                            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1, wordBreak: 'break-all' }}>
                              {model.repo_id}
                            </Typography>
                            {model.description && (
                              <Typography variant="body2" color="text.secondary" sx={{ mb: 1.25 }}>
                                {model.description.length > 100
                                  ? model.description.substring(0, 100) + '...'
                                  : model.description}
                              </Typography>
                            )}
                          </Box>
                          <Box sx={{ display: 'flex', gap: 0.5, flexShrink: 0 }}>
                            <Tooltip title="View details">
                              <IconButton
                                size="small"
                                onClick={() => handleViewDetails(model)}
                                aria-label={`View details for ${model.name}`}
                                sx={{
                                  width: 32,
                                  height: 32,
                                  borderRadius: '8px',
                                  border: '1px solid',
                                  borderColor: 'divider',
                                }}
                              >
                                <InfoIcon sx={{ fontSize: 17 }} />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Delete from cache">
                              <IconButton
                                size="small"
                                onClick={() => handleDeleteModel(model)}
                                color="error"
                                aria-label={`Delete ${model.name} from cache`}
                                sx={{
                                  width: 32,
                                  height: 32,
                                  borderRadius: '8px',
                                  border: '1px solid',
                                  borderColor: alpha(accentColors.rose, 0.3),
                                }}
                              >
                                <DeleteIcon sx={{ fontSize: 17 }} />
                              </IconButton>
                            </Tooltip>
                          </Box>
                        </Box>

                        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center', mt: 1.25 }}>
                          {model.model_type && (
                            <StatusChip status="type" label={model.model_type} statusConfig={MODEL_REGISTRY_STATUS_CONFIG} />
                          )}
                          {model.license && (
                            <StatusChip status="license" label={model.license} statusConfig={MODEL_REGISTRY_STATUS_CONFIG} />
                          )}
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: 'text.secondary' }}>
                            <DownloadIcon sx={{ fontSize: 14 }} />
                            <Typography variant="caption">{formatNumber(model.downloads)}</Typography>
                          </Box>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: 'text.secondary' }}>
                            <StarIcon sx={{ fontSize: 14 }} />
                            <Typography variant="caption">{formatNumber(model.likes)}</Typography>
                          </Box>
                        </Box>
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              )}
            </SectionCard>
          </Grid>
        )}

        <Grid
          item
          xs={12}
          lg={showRegistryList ? 4 : 8}
          xl={showRegistryList ? 4 : 7}
          sx={{ order: { xs: 1, lg: prioritizeCacheForm ? 1 : 2 } }}
        >
          <Box sx={{ display: 'grid', gap: 1.5 }}>
            <SectionCard
              title="Cache model"
              icon={<CloudIcon />}
              accentColor={accentColors.info}
              action={!showRegistryList ? registryRefreshAction : undefined}
            >
              <Box
                sx={{
                  display: 'grid',
                  gap: 1.25,
                  gridTemplateColumns: {
                    xs: '1fr',
                    sm: prioritizeCacheForm || !showRegistryList ? 'repeat(2, minmax(0, 1fr))' : '1fr',
                    xl: prioritizeCacheForm || !showRegistryList ? 'repeat(2, minmax(0, 1fr))' : '1fr',
                  },
                }}
              >
                <TextField
                  label="Repository ID"
                  value={cacheFormData.repo_id}
                  onChange={(e) => setCacheFormData({ ...cacheFormData, repo_id: e.target.value })}
                  fullWidth
                  required
                  size="small"
                  placeholder="meta-llama/Llama-3.1-8B"
                />
                <TextField
                  label="Model name"
                  value={cacheFormData.name}
                  onChange={(e) => setCacheFormData({ ...cacheFormData, name: e.target.value })}
                  fullWidth
                  required
                  size="small"
                />
                <TextField
                  label="Type"
                  value={cacheFormData.model_type}
                  onChange={(e) => setCacheFormData({ ...cacheFormData, model_type: e.target.value })}
                  fullWidth
                  size="small"
                  placeholder="chat"
                />
                <TextField
                  label="License"
                  value={cacheFormData.license}
                  onChange={(e) => setCacheFormData({ ...cacheFormData, license: e.target.value })}
                  fullWidth
                  size="small"
                  placeholder="Apache-2.0"
                />
                <Box sx={{ gridColumn: '1 / -1', display: 'flex', justifyContent: 'space-between', gap: 1, flexWrap: 'wrap' }}>
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => setCacheDialogOpen(true)}
                    sx={{ height: 34, borderRadius: '8px', borderColor: 'divider', fontWeight: 800 }}
                  >
                    More fields
                  </Button>
                  {canCacheModel && (
                    <Button
                      variant="contained"
                      size="small"
                      onClick={handleCacheModel}
                      sx={{ height: 34, borderRadius: '8px', fontWeight: 800 }}
                    >
                      Cache
                    </Button>
                  )}
                </Box>
              </Box>
            </SectionCard>

            {usageStats.length > 0 && (
              <SectionCard
                title="Usage"
                subtitle="Loads, inferences, and token activity."
                icon={<StatsIcon />}
                accentColor={accentColors.success}
                action={<StatusChip status="usage" label={`${usageStats.length} records`} statusConfig={MODEL_REGISTRY_STATUS_CONFIG} />}
              >
                <List disablePadding sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  {usageStats.map((usage, index) => (
                    <ListItem
                      key={`${usage.model_id}-${usage.variant ?? index}`}
                      sx={{
                        px: 1.25,
                        py: 1,
                        alignItems: 'flex-start',
                        borderRadius: '8px',
                        border: '1px solid',
                        borderColor: 'divider',
                        bgcolor: 'background.paper',
                      }}
                    >
                      <ListItemText
                        primary={usage.name || usage.repo_id || 'Unknown Model'}
                        primaryTypographyProps={{ fontWeight: 700, fontSize: '0.85rem', sx: { wordBreak: 'break-word' } }}
                        secondary={
                          <Box sx={{ display: 'flex', gap: 1.25, flexWrap: 'wrap', mt: 0.5 }}>
                            <Typography variant="caption">
                              Loads: {usage.load_count}
                            </Typography>
                            <Typography variant="caption">
                              Runs: {usage.inference_count}
                            </Typography>
                            <Typography variant="caption">
                              Tokens: {formatNumber(usage.total_tokens_generated)}
                            </Typography>
                            {usage.last_used && (
                              <Typography variant="caption" color="text.secondary">
                                Last: {new Date(usage.last_used).toLocaleDateString()}
                              </Typography>
                            )}
                          </Box>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </SectionCard>
            )}
          </Box>
        </Grid>
      </Grid>

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
                      <StatusChip
                        key={tag}
                        status="tag"
                        label={tag}
                        variant="outlined"
                        statusConfig={MODEL_REGISTRY_STATUS_CONFIG}
                      />
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
                            <StatusChip
                              status="quantization"
                              label={variant.quantization}
                              statusConfig={MODEL_REGISTRY_STATUS_CONFIG}
                            />
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
          {canCacheModel && (
            <Button
              variant="contained"
              onClick={handleCacheModel}
            >
              Cache Model
            </Button>
          )}
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
