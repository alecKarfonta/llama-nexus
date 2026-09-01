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
  ListItemButton,
  ListItemText,
  ListItemSecondaryAction,
  Tooltip,
  Alert,
  CircularProgress,
  Divider,
  InputAdornment,
  Menu,
  MenuItem,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import {
  Add as AddIcon,
  Search as SearchIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon,
  ContentCopy as CopyIcon,
  History as HistoryIcon,
  Code as CodeIcon,
  Chat as ChatIcon,
  Settings as SettingsIcon,
  Folder as FolderIcon,
  BarChart as ChartIcon,
  Download as DownloadIcon,
  FilterAltOff as ClearFiltersIcon,
  MoreVert as MoreIcon,
  PlayArrow as UseIcon,
} from '@mui/icons-material'
import { PageShell, SectionCard, StatusChip, type StatusChipConfigMap } from '@/components/common'
import { useClipboard } from '@/hooks/useClipboard'
import { apiService } from '@/services/api'
import { accentColors } from '@/utils/theme'

interface PromptTemplate {
  id: string
  name: string
  description?: string
  content: string
  category: string
  tags: string[]
  variables: string[]
  is_system_prompt: boolean
  is_favorite: boolean
  use_count: number
  created_at: string
  updated_at: string
}

interface PromptCategory {
  id: string
  name: string
  description?: string
  color: string
  icon: string
  prompt_count: number
}

interface PromptVersion {
  id: number
  version: number
  content: string
  change_note?: string
  created_at: string
}

const getCategoryIcon = (icon: string) => {
  switch (icon) {
    case 'code': return <CodeIcon />
    case 'chat': return <ChatIcon />
    case 'settings': return <SettingsIcon />
    case 'chart': return <ChartIcon />
    case 'edit': return <EditIcon />
    default: return <FolderIcon />
  }
}

const formatNumber = (value: number): string => {
  if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`
  if (value >= 1000) return `${(value / 1000).toFixed(1)}K`
  return value.toString()
}

const PROMPT_LIBRARY_STATUS_CONFIG: StatusChipConfigMap = {
  all: { color: accentColors.primary, label: 'All prompts' },
  categories: { color: accentColors.success, label: 'Categories' },
  category: { color: accentColors.info, label: 'Category' },
  favorites: { color: accentColors.warning, label: 'Favorites' },
  library: { color: accentColors.success, label: 'Prompt library' },
  prompts: { color: accentColors.primary, label: 'Prompts' },
  system: { color: accentColors.rose, label: 'System' },
  tag: { color: accentColors.slate, label: 'Tag' },
  tools: { color: accentColors.info, label: 'Tools' },
  variables: { color: accentColors.primary, label: 'Variables' },
}

const promptLibraryPanelSx = {
  borderRadius: '8px',
  bgcolor: 'background.paper',
  border: '1px solid',
  borderColor: 'divider',
} as const

const promptLibraryIconButtonSx = {
  width: 34,
  height: 34,
  borderRadius: '8px',
  border: '1px solid',
  borderColor: 'divider',
  bgcolor: 'action.hover',
  '&:hover': { bgcolor: 'action.selected' },
} as const

export default function PromptLibraryPage() {
  const copyToClipboard = useClipboard()

  // State
  const [prompts, setPrompts] = useState<PromptTemplate[]>([])
  const [categories, setCategories] = useState<PromptCategory[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false)
  
  // Dialog states
  const [editDialogOpen, setEditDialogOpen] = useState(false)
  const [editingPrompt, setEditingPrompt] = useState<PromptTemplate | null>(null)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [promptToDelete, setPromptToDelete] = useState<PromptTemplate | null>(null)
  const [historyDialogOpen, setHistoryDialogOpen] = useState(false)
  const [promptHistory, setPromptHistory] = useState<PromptVersion[]>([])
  const [historyPromptId, setHistoryPromptId] = useState<string | null>(null)
  const [useDialogOpen, setUseDialogOpen] = useState(false)
  const [promptToUse, setPromptToUse] = useState<PromptTemplate | null>(null)
  const [variableValues, setVariableValues] = useState<Record<string, string>>({})
  const [renderedContent, setRenderedContent] = useState<string | null>(null)
  
  // Menu state
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null)
  const [menuPrompt, setMenuPrompt] = useState<PromptTemplate | null>(null)
  
  // Form state
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    content: '',
    category: 'general',
    tags: [] as string[],
    is_system_prompt: false,
  })
  const [tagInput, setTagInput] = useState('')

  // Load data
  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [promptsRes, categoriesRes] = await Promise.all([
        apiService.listPrompts({
          category: selectedCategory || undefined,
          search: searchQuery || undefined,
          is_favorite: showFavoritesOnly ? true : undefined,
          limit: 100,
        }),
        apiService.listPromptCategories(),
      ])
      setPrompts(promptsRes.prompts)
      setCategories(categoriesRes.categories)
    } catch (err: any) {
      setError(err.message || 'Failed to load prompts')
    } finally {
      setLoading(false)
    }
  }, [selectedCategory, searchQuery, showFavoritesOnly])

  useEffect(() => {
    loadData()
  }, [loadData])

  // Handlers
  const handleCreatePrompt = () => {
    setEditingPrompt(null)
    setFormData({
      name: '',
      description: '',
      content: '',
      category: selectedCategory || 'general',
      tags: [],
      is_system_prompt: false,
    })
    setEditDialogOpen(true)
  }

  const handleEditPrompt = (prompt: PromptTemplate) => {
    setEditingPrompt(prompt)
    setFormData({
      name: prompt.name,
      description: prompt.description || '',
      content: prompt.content,
      category: prompt.category,
      tags: prompt.tags,
      is_system_prompt: prompt.is_system_prompt,
    })
    setEditDialogOpen(true)
    setMenuAnchor(null)
  }

  const handleSavePrompt = async () => {
    try {
      if (editingPrompt) {
        await apiService.updatePrompt(editingPrompt.id, formData)
      } else {
        await apiService.createPrompt(formData)
      }
      setEditDialogOpen(false)
      loadData()
    } catch (err: any) {
      setError(err.message || 'Failed to save prompt')
    }
  }

  const handleQuickCreatePrompt = async () => {
    if (!formData.name.trim() || !formData.content.trim()) return

    try {
      await apiService.createPrompt({
        ...formData,
        name: formData.name.trim(),
        content: formData.content.trim(),
        description: formData.description.trim(),
      })
      setFormData({
        name: '',
        description: '',
        content: '',
        category: selectedCategory || 'general',
        tags: [],
        is_system_prompt: false,
      })
      loadData()
    } catch (err: any) {
      setError(err.message || 'Failed to save prompt')
    }
  }

  const handleDeletePrompt = async () => {
    if (!promptToDelete) return
    try {
      await apiService.deletePrompt(promptToDelete.id)
      setDeleteDialogOpen(false)
      setPromptToDelete(null)
      loadData()
    } catch (err: any) {
      setError(err.message || 'Failed to delete prompt')
    }
  }

  const handleToggleFavorite = async (prompt: PromptTemplate) => {
    try {
      await apiService.updatePrompt(prompt.id, {
        is_favorite: !prompt.is_favorite,
      })
      loadData()
    } catch (err: any) {
      setError(err.message || 'Failed to update prompt')
    }
  }

  const handleCopyContent = (content: string) => {
    copyToClipboard(content, {
      successMessage: 'Prompt content copied.',
      successTitle: 'Prompt copied',
    })
  }

  const handleViewHistory = async (prompt: PromptTemplate) => {
    setHistoryPromptId(prompt.id)
    try {
      const res = await apiService.getPromptVersions(prompt.id)
      setPromptHistory(res.versions)
      setHistoryDialogOpen(true)
    } catch (err: any) {
      setError(err.message || 'Failed to load history')
    }
    setMenuAnchor(null)
  }

  const handleRestoreVersion = async (version: number) => {
    if (!historyPromptId) return
    try {
      await apiService.restorePromptVersion(historyPromptId, version)
      setHistoryDialogOpen(false)
      loadData()
    } catch (err: any) {
      setError(err.message || 'Failed to restore version')
    }
  }

  const handleUsePrompt = (prompt: PromptTemplate) => {
    setPromptToUse(prompt)
    setVariableValues({})
    setRenderedContent(null)
    if (prompt.variables.length > 0) {
      // Initialize variable values
      const values: Record<string, string> = {}
      prompt.variables.forEach(v => { values[v] = '' })
      setVariableValues(values)
      setUseDialogOpen(true)
    } else {
      // No variables, just copy content
      handleCopyContent(prompt.content)
    }
    setMenuAnchor(null)
  }

  const handleRenderPrompt = async () => {
    if (!promptToUse) return
    try {
      const res = await apiService.renderPrompt(promptToUse.id, variableValues)
      setRenderedContent(res.rendered)
    } catch (err: any) {
      setError(err.message || 'Failed to render prompt')
    }
  }

  const handleAddTag = () => {
    if (tagInput.trim() && !formData.tags.includes(tagInput.trim())) {
      setFormData({ ...formData, tags: [...formData.tags, tagInput.trim()] })
      setTagInput('')
    }
  }

  const handleRemoveTag = (tag: string) => {
    setFormData({ ...formData, tags: formData.tags.filter(t => t !== tag) })
  }

  const handleExport = async () => {
    try {
      const res = await apiService.exportPrompts()
      const blob = new Blob([res.data], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'prompts-export.json'
      a.click()
      URL.revokeObjectURL(url)
    } catch (err: any) {
      setError(err.message || 'Failed to export prompts')
    }
  }

  const selectedCategoryMeta = categories.find((category) => category.id === selectedCategory)
  const totalLibraryPrompts = categories.reduce((total, category) => total + category.prompt_count, 0)
  const favoriteCount = prompts.filter((prompt) => prompt.is_favorite).length
  const systemPromptCount = prompts.filter((prompt) => prompt.is_system_prompt).length
  const variablePromptCount = prompts.filter((prompt) => prompt.variables.length > 0).length
  const totalUses = prompts.reduce((total, prompt) => total + prompt.use_count, 0)
  const libraryHasPrompts = totalLibraryPrompts > 0 || prompts.length > 0
  const activeFilterLabel = showFavoritesOnly
    ? 'Favorites'
    : selectedCategoryMeta?.name || 'All prompts'
  const activeFilterStatus = showFavoritesOnly
    ? 'favorites'
    : selectedCategoryMeta
      ? 'category'
      : 'all'
  const hasActivePromptFilters = Boolean(searchQuery || selectedCategory || showFavoritesOnly)
  const showPromptLibraryTools = libraryHasPrompts || hasActivePromptFilters
  const hasGeneralCategory = categories.some((category) => category.id === 'general')
  const canQuickCreatePrompt = formData.name.trim().length > 0 && formData.content.trim().length > 0
  const canSavePrompt = formData.name.trim().length > 0 && formData.content.trim().length > 0

  const handleClearPromptFilters = () => {
    setSearchQuery('')
    setSelectedCategory(null)
    setShowFavoritesOnly(false)
  }
  const promptHeaderActions = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
      <TextField
        placeholder="Search prompts..."
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        size="small"
        sx={{ width: { xs: '100%', sm: 250, lg: 320 }, '& .MuiInputBase-root': { height: 34 } }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon />
            </InputAdornment>
          ),
        }}
      />
      {hasActivePromptFilters && (
        <Tooltip title="Clear filters">
          <IconButton
            aria-label="Clear filters"
            size="small"
            onClick={handleClearPromptFilters}
            sx={promptLibraryIconButtonSx}
          >
            <ClearFiltersIcon sx={{ fontSize: 18 }} />
          </IconButton>
        </Tooltip>
      )}
      <Tooltip title={showFavoritesOnly ? 'Show all prompts' : 'Show favorites'}>
        <IconButton
          aria-label={showFavoritesOnly ? 'Show all prompts' : 'Show favorites'}
          size="small"
          onClick={() => setShowFavoritesOnly(!showFavoritesOnly)}
          sx={{
            ...promptLibraryIconButtonSx,
            borderColor: showFavoritesOnly ? alpha(accentColors.warning, 0.45) : 'divider',
            color: accentColors.warning,
            bgcolor: showFavoritesOnly ? alpha(accentColors.warning, 0.12) : 'transparent',
          }}
        >
          <StarIcon sx={{ fontSize: 18 }} />
        </IconButton>
      </Tooltip>
      {libraryHasPrompts && (
        <Tooltip title="Export prompts">
          <IconButton
            aria-label="Export prompts"
            size="small"
            onClick={handleExport}
            sx={promptLibraryIconButtonSx}
          >
            <DownloadIcon sx={{ fontSize: 18 }} />
          </IconButton>
        </Tooltip>
      )}
      <Button size="small" variant="contained" onClick={handleCreatePrompt} startIcon={<AddIcon />} sx={{ height: 34, borderRadius: '8px', fontWeight: 800, px: 1.25 }}>
        New
      </Button>
    </Box>
  )
  const promptSummaryRow = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
      <StatusChip status={activeFilterStatus} label={activeFilterLabel} statusConfig={PROMPT_LIBRARY_STATUS_CONFIG} height={22} />
      <StatusChip status="prompts" label={`${formatNumber(prompts.length)} visible`} statusConfig={PROMPT_LIBRARY_STATUS_CONFIG} height={22} />
      {favoriteCount > 0 && (
        <StatusChip status="favorites" label={`${formatNumber(favoriteCount)} favorites`} statusConfig={PROMPT_LIBRARY_STATUS_CONFIG} height={22} />
      )}
      {(totalUses > 0 || systemPromptCount > 0 || variablePromptCount > 0) && (
        <Typography variant="caption" color="text.secondary" sx={{ minWidth: 0 }}>
          {totalUses > 0 ? `${formatNumber(totalUses)} uses` : ''}
          {systemPromptCount > 0 ? `${totalUses > 0 ? ' / ' : ''}${formatNumber(systemPromptCount)} system` : ''}
          {variablePromptCount > 0 ? `${totalUses > 0 || systemPromptCount > 0 ? ' / ' : ''}${formatNumber(variablePromptCount)} variables` : ''}
        </Typography>
      )}
    </Box>
  )

  return (
    <PageShell
      sx={{
        py: 1.25,
        display: 'flex',
        flexDirection: 'column',
        gap: 1.25,
        overflow: 'visible',
      }}
    >
      {error && (
        <Alert severity="error" sx={{ mb: 1.25 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid
        container
        spacing={1.5}
        data-prompt-library-workspace
        sx={{
          alignItems: 'flex-start',
          overflow: 'visible',
        }}
      >
        {/* Sidebar - Categories */}
        {showPromptLibraryTools && (
        <Grid item xs={12} md={3} lg={2.4}>
          <SectionCard
              title="Categories"
              icon={<FolderIcon />}
              accentColor={accentColors.success}
              action={<StatusChip status="categories" label={`${formatNumber(categories.length)} groups`} statusConfig={PROMPT_LIBRARY_STATUS_CONFIG} height={22} />}
            >
            <List dense disablePadding>
              <ListItemButton
                selected={selectedCategory === null}
                onClick={() => setSelectedCategory(null)}
                  sx={{ borderRadius: '8px', mb: 0.25, minHeight: 34, py: 0.25, px: 1 }}
                >
                  <ListItemText
                    primary="All Prompts"
                    primaryTypographyProps={{ noWrap: true, fontSize: '0.84rem', fontWeight: 700 }}
                  />
                  <StatusChip status="prompts" label={formatNumber(totalLibraryPrompts || prompts.length)} statusConfig={PROMPT_LIBRARY_STATUS_CONFIG} height={22} />
                </ListItemButton>
              {categories.map(cat => (
                <ListItemButton
                  key={cat.id}
                  selected={selectedCategory === cat.id}
                  onClick={() => setSelectedCategory(cat.id)}
                  sx={{ borderRadius: '8px', mb: 0.25, minHeight: 34, py: 0.25, px: 1 }}
                >
                  <Box sx={{ mr: 0.75, color: cat.color, display: 'flex', '& .MuiSvgIcon-root': { fontSize: 18 } }}>
                    {getCategoryIcon(cat.icon)}
                  </Box>
                    <ListItemText
                      primary={cat.name}
                      primaryTypographyProps={{ noWrap: true, fontSize: '0.84rem', fontWeight: 700 }}
                    />
                    <StatusChip status="category" label={formatNumber(cat.prompt_count)} statusConfig={PROMPT_LIBRARY_STATUS_CONFIG} height={22} />
                  </ListItemButton>
                ))}
            </List>
          </SectionCard>
        </Grid>
        )}

        {/* Main Content */}
        <Grid item xs={12} md={showPromptLibraryTools ? 9 : 12} lg={showPromptLibraryTools ? 9.6 : 12}>
          <SectionCard
            title="Prompts"
            subtitle={showPromptLibraryTools ? `${formatNumber(prompts.length)} visible in ${activeFilterLabel.toLowerCase()}` : undefined}
            icon={<ChatIcon />}
            accentColor={accentColors.primary}
            action={showPromptLibraryTools ? promptHeaderActions : undefined}
          >
            {showPromptLibraryTools && promptSummaryRow}
            {loading ? (
              <Box sx={{ minHeight: 120, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <CircularProgress />
              </Box>
            ) : prompts.length === 0 ? (
              <Box
                data-prompt-library-empty-state
                sx={{
                  p: 1.25,
                  borderRadius: '8px',
                  border: '1px dashed',
                  borderColor: 'divider',
                  bgcolor: alpha(accentColors.primary, 0.04),
                }}
              >
                <Box
                  sx={{
                    width: '100%',
                    display: 'grid',
                    gridTemplateColumns: { xs: '1fr', lg: hasActivePromptFilters ? 'minmax(0, 1fr) auto' : '1fr' },
                    gap: 1,
                    alignItems: 'start',
                  }}
                >
                  <Box sx={{ minWidth: 0 }}>
                    <Typography variant="body2" sx={{ fontWeight: 800 }}>
                      {hasActivePromptFilters ? 'No matching prompts' : 'No prompts yet'}
                    </Typography>
                    {hasActivePromptFilters && (
                      <Typography variant="caption" color="text.secondary">
                        {activeFilterLabel}
                      </Typography>
                    )}
                  </Box>

                  {hasActivePromptFilters ? (
                    <Box sx={{ display: 'flex', gap: 0.75, justifyContent: { xs: 'flex-start', lg: 'flex-end' }, flexWrap: 'wrap' }}>
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={handleClearPromptFilters}
                        sx={{ height: 34, borderRadius: '8px', borderColor: 'divider', fontWeight: 800 }}
                      >
                        Clear
                      </Button>
                    </Box>
                  ) : (
                    <Box
                      sx={{
                        display: 'grid',
                        gridTemplateColumns: {
                          xs: '1fr',
                          sm: canQuickCreatePrompt
                            ? 'minmax(180px, 0.7fr) minmax(140px, 0.35fr) auto'
                            : 'minmax(180px, 0.7fr) minmax(140px, 0.35fr)',
                        },
                        gap: 0.75,
                        alignItems: 'start',
                      }}
                    >
                      <TextField
                        label="Name"
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                        size="small"
                        fullWidth
                      />
                      <TextField
                        select
                        label="Category"
                        value={formData.category}
                        onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                        size="small"
                        fullWidth
                      >
                        {!hasGeneralCategory && <MenuItem value="general">General</MenuItem>}
                        {categories.map(category => (
                          <MenuItem key={category.id} value={category.id}>{category.name}</MenuItem>
                        ))}
                      </TextField>
                      <TextField
                        label="Prompt"
                        value={formData.content}
                        onChange={(e) => setFormData({ ...formData, content: e.target.value })}
                        size="small"
                        fullWidth
                        multiline
                        minRows={3}
                        sx={{ gridColumn: { xs: 'auto', sm: '1 / -1' } }}
                      />
                      {canQuickCreatePrompt && (
                        <Button
                          size="small"
                          variant="contained"
                          onClick={handleQuickCreatePrompt}
                          sx={{
                            height: 34,
                            minWidth: 96,
                            borderRadius: '8px',
                            fontWeight: 800,
                            px: 1.25,
                            whiteSpace: 'nowrap',
                            justifySelf: { xs: 'end', sm: 'stretch' },
                            gridColumn: { xs: 'auto', sm: '3 / 4' },
                            gridRow: { xs: 'auto', sm: '1' },
                          }}
                        >
                          Create
                        </Button>
                      )}
                    </Box>
                  )}
                </Box>
              </Box>
            ) : (
              <Grid
                container
                spacing={1.25}
                sx={{ alignContent: 'flex-start' }}
              >
                {prompts.map(prompt => {
                  const categoryColor = categories.find(c => c.id === prompt.category)?.color || accentColors.slate

                  return (
                    <Grid item xs={12} key={prompt.id}>
                        <Box
                          sx={{
                            height: '100%',
                            p: 2,
                            borderRadius: '8px',
                            bgcolor: 'background.paper',
                            border: '1px solid',
                            borderColor: 'divider',
                            borderLeft: `4px solid ${categoryColor}`,
                            transition: 'border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease',
                            '&:hover': {
                              borderColor: alpha(accentColors.success, 0.36),
                              boxShadow: `0 16px 42px ${alpha('#000', 0.18)}`,
                              transform: 'translateY(-1px)',
                            },
                          }}
                        >
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <Box sx={{ flex: 1 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                            <Typography variant="h6" sx={{ fontSize: '1rem', fontWeight: 600 }}>
                              {prompt.name}
                            </Typography>
                              {prompt.is_system_prompt && (
                                <StatusChip status="system" statusConfig={PROMPT_LIBRARY_STATUS_CONFIG} />
                              )}
                            {prompt.is_favorite && (
                              <StarIcon sx={{ color: 'warning.main', fontSize: 18 }} />
                            )}
                          </Box>
                          {prompt.description && (
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                              {prompt.description}
                            </Typography>
                          )}
                          <Typography
                            variant="body2"
                            sx={{
                              bgcolor: 'action.hover',
                              p: 1,
                              borderRadius: '8px',
                              fontFamily: 'monospace',
                              fontSize: '0.8rem',
                              maxHeight: 80,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'pre-wrap',
                            }}
                          >
                            {prompt.content.length > 200
                              ? prompt.content.substring(0, 200) + '...'
                              : prompt.content}
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 1, mt: 1, flexWrap: 'wrap', alignItems: 'center' }}>
                              {prompt.variables.length > 0 && (
                                <StatusChip
                                  status="variables"
                                  label={`${prompt.variables.length} variable${prompt.variables.length > 1 ? 's' : ''}`}
                                  statusConfig={PROMPT_LIBRARY_STATUS_CONFIG}
                                  height={22}
                                />
                              )}
                              {prompt.tags.map(tag => (
                                <StatusChip key={tag} status="tag" label={tag} statusConfig={PROMPT_LIBRARY_STATUS_CONFIG} height={22} />
                              ))}
                            <Typography variant="caption" color="text.secondary" sx={{ ml: 'auto' }}>
                              Used {prompt.use_count} times
                            </Typography>
                          </Box>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <Tooltip title="Use prompt">
                            <IconButton size="small" onClick={() => handleUsePrompt(prompt)}>
                              <UseIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Copy content">
                            <IconButton size="small" onClick={() => handleCopyContent(prompt.content)}>
                              <CopyIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title={prompt.is_favorite ? 'Remove from favorites' : 'Add to favorites'}>
                            <IconButton size="small" onClick={() => handleToggleFavorite(prompt)}>
                              {prompt.is_favorite ? <StarIcon color="warning" /> : <StarBorderIcon />}
                            </IconButton>
                          </Tooltip>
                          <IconButton
                            size="small"
                            onClick={(e) => {
                              setMenuAnchor(e.currentTarget)
                              setMenuPrompt(prompt)
                            }}
                          >
                            <MoreIcon />
                          </IconButton>
                          </Box>
                        </Box>
                        </Box>
                      </Grid>
                  )
                })}
              </Grid>
            )}
          </SectionCard>
        </Grid>
      </Grid>

      {/* Context Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={() => setMenuAnchor(null)}
      >
        <MenuItem onClick={() => menuPrompt && handleEditPrompt(menuPrompt)}>
          <EditIcon sx={{ mr: 1 }} /> Edit
        </MenuItem>
        <MenuItem onClick={() => menuPrompt && handleViewHistory(menuPrompt)}>
          <HistoryIcon sx={{ mr: 1 }} /> Version History
        </MenuItem>
        <Divider />
        <MenuItem
          onClick={() => {
            setPromptToDelete(menuPrompt)
            setDeleteDialogOpen(true)
            setMenuAnchor(null)
          }}
          sx={{ color: 'error.main' }}
        >
          <DeleteIcon sx={{ mr: 1 }} /> Delete
        </MenuItem>
      </Menu>

      {/* Edit/Create Dialog */}
      <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          {editingPrompt ? 'Edit Prompt' : 'Create New Prompt'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <TextField
              label="Name"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="Description"
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              fullWidth
              multiline
              rows={2}
            />
            <TextField
              label="Content"
              value={formData.content}
              onChange={(e) => setFormData({ ...formData, content: e.target.value })}
              fullWidth
              multiline
              rows={8}
              required
              placeholder="Enter your prompt here. Use {{variable}} for template variables."
              sx={{ fontFamily: 'monospace' }}
            />
            <TextField
              select
              label="Category"
              value={formData.category}
              onChange={(e) => setFormData({ ...formData, category: e.target.value })}
              fullWidth
            >
              {categories.map(cat => (
                <MenuItem key={cat.id} value={cat.id}>
                  {cat.name}
                </MenuItem>
              ))}
            </TextField>
            <Box>
              <Typography variant="body2" sx={{ mb: 1 }}>Tags</Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 1 }}>
                  {formData.tags.map(tag => (
                    <StatusChip
                      key={tag}
                      status="tag"
                      label={tag}
                      statusConfig={PROMPT_LIBRARY_STATUS_CONFIG}
                      onDelete={() => handleRemoveTag(tag)}
                      height={22}
                    />
                  ))}
                </Box>
              <TextField
                size="small"
                placeholder="Add tag..."
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault()
                    handleAddTag()
                  }
                }}
                InputProps={{
                  endAdornment: (
                    <Button size="small" onClick={handleAddTag}>Add</Button>
                  ),
                }}
              />
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
          {canSavePrompt && (
            <Button
              variant="contained"
              onClick={handleSavePrompt}
            >
              Save
            </Button>
          )}
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Prompt</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete "{promptToDelete?.name}"? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" color="error" onClick={handleDeletePrompt}>
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Version History Dialog */}
      <Dialog open={historyDialogOpen} onClose={() => setHistoryDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Version History</DialogTitle>
        <DialogContent>
          <List>
            {promptHistory.map((version, index) => (
              <ListItem key={version.id} divider>
                <ListItemText
                  primary={`Version ${version.version}${index === 0 ? ' (Current)' : ''}`}
                  secondary={
                    <>
                      <Typography variant="caption" display="block">
                        {new Date(version.created_at).toLocaleString()}
                        {version.change_note && ` - ${version.change_note}`}
                      </Typography>
                      <Typography
                        variant="body2"
                        sx={{
                          mt: 1,
                          p: 1,
                          bgcolor: 'action.hover',
                          borderRadius: '8px',
                          fontFamily: 'monospace',
                          fontSize: '0.75rem',
                          maxHeight: 100,
                          overflow: 'auto',
                          whiteSpace: 'pre-wrap',
                        }}
                      >
                        {version.content.length > 300
                          ? version.content.substring(0, 300) + '...'
                          : version.content}
                      </Typography>
                    </>
                  }
                />
                {index > 0 && (
                  <ListItemSecondaryAction>
                    <Button
                      size="small"
                      onClick={() => handleRestoreVersion(version.version)}
                    >
                      Restore
                    </Button>
                  </ListItemSecondaryAction>
                )}
              </ListItem>
            ))}
          </List>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setHistoryDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Use Prompt Dialog (for variables) */}
      <Dialog open={useDialogOpen} onClose={() => setUseDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Use Prompt: {promptToUse?.name}</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Fill in the template variables below:
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {promptToUse?.variables.map(variable => (
              <TextField
                key={variable}
                label={variable}
                value={variableValues[variable] || ''}
                onChange={(e) => setVariableValues({
                  ...variableValues,
                  [variable]: e.target.value,
                })}
                fullWidth
              />
            ))}
          </Box>
          {renderedContent && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>Rendered Output:</Typography>
                <Box sx={{ p: 2, ...promptLibraryPanelSx }}>
                  <Typography
                    sx={{
                      fontFamily: 'monospace',
                    fontSize: '0.85rem',
                    whiteSpace: 'pre-wrap',
                  }}
                  >
                    {renderedContent}
                  </Typography>
                </Box>
              </Box>
            )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUseDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleRenderPrompt}>Preview</Button>
          <Button
            variant="contained"
            onClick={() => {
              if (renderedContent) {
                handleCopyContent(renderedContent)
                setUseDialogOpen(false)
              } else {
                handleRenderPrompt()
              }
            }}
            startIcon={<CopyIcon />}
          >
            {renderedContent ? 'Copy to Clipboard' : 'Render & Copy'}
          </Button>
        </DialogActions>
      </Dialog>
    </PageShell>
  )
}
