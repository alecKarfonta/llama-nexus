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
  ListItemSecondaryAction,
  Tooltip,
  Alert,
  CircularProgress,
  Divider,
  InputAdornment,
  Paper,
  Tabs,
  Tab,
  Menu,
  MenuItem,
  Badge,
} from '@mui/material'
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
  Upload as UploadIcon,
  MoreVert as MoreIcon,
  PlayArrow as UseIcon,
} from '@mui/icons-material'
import { apiService } from '@/services/api'

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

export default function PromptLibraryPage() {
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
    navigator.clipboard.writeText(content)
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

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" sx={{ fontWeight: 600 }}>
          Prompt Library
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title="Export all prompts">
            <IconButton onClick={handleExport}>
              <DownloadIcon />
            </IconButton>
          </Tooltip>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={handleCreatePrompt}
          >
            New Prompt
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Sidebar - Categories */}
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
              Categories
            </Typography>
            <List dense>
              <ListItem
                button
                selected={selectedCategory === null}
                onClick={() => setSelectedCategory(null)}
              >
                <ListItemText primary="All Prompts" />
                <Chip size="small" label={prompts.length} />
              </ListItem>
              {categories.map(cat => (
                <ListItem
                  key={cat.id}
                  button
                  selected={selectedCategory === cat.id}
                  onClick={() => setSelectedCategory(cat.id)}
                >
                  <Box sx={{ mr: 1, color: cat.color }}>
                    {getCategoryIcon(cat.icon)}
                  </Box>
                  <ListItemText primary={cat.name} />
                  <Chip size="small" label={cat.prompt_count} />
                </ListItem>
              ))}
            </List>
            <Divider sx={{ my: 2 }} />
            <Button
              fullWidth
              variant={showFavoritesOnly ? 'contained' : 'outlined'}
              color="warning"
              startIcon={<StarIcon />}
              onClick={() => setShowFavoritesOnly(!showFavoritesOnly)}
              size="small"
            >
              Favorites Only
            </Button>
          </Paper>
        </Grid>

        {/* Main Content */}
        <Grid item xs={12} md={9}>
          {/* Search Bar */}
          <TextField
            fullWidth
            placeholder="Search prompts..."
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

          {/* Prompts List */}
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress />
            </Box>
          ) : prompts.length === 0 ? (
            <Paper sx={{ p: 4, textAlign: 'center' }}>
              <Typography color="text.secondary">
                No prompts found. Create your first prompt to get started.
              </Typography>
            </Paper>
          ) : (
            <Grid container spacing={2}>
              {prompts.map(prompt => (
                <Grid item xs={12} key={prompt.id}>
                  <Card sx={{ 
                    '&:hover': { boxShadow: 4 },
                    borderLeft: `4px solid ${categories.find(c => c.id === prompt.category)?.color || '#6B7280'}`,
                  }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <Box sx={{ flex: 1 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                            <Typography variant="h6" sx={{ fontSize: '1rem', fontWeight: 600 }}>
                              {prompt.name}
                            </Typography>
                            {prompt.is_system_prompt && (
                              <Chip size="small" label="System" color="error" />
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
                              borderRadius: 1,
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
                              <Chip
                                size="small"
                                variant="outlined"
                                label={`${prompt.variables.length} variable${prompt.variables.length > 1 ? 's' : ''}`}
                              />
                            )}
                            {prompt.tags.map(tag => (
                              <Chip key={tag} size="small" label={tag} />
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
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}
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
                  <Chip
                    key={tag}
                    label={tag}
                    onDelete={() => handleRemoveTag(tag)}
                    size="small"
                  />
                ))}
              </Box>
              <TextField
                size="small"
                placeholder="Add tag..."
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleAddTag()}
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
          <Button
            variant="contained"
            onClick={handleSavePrompt}
            disabled={!formData.name || !formData.content}
          >
            Save
          </Button>
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
                          borderRadius: 1,
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
              <Paper sx={{ p: 2, bgcolor: 'action.hover' }}>
                <Typography
                  sx={{
                    fontFamily: 'monospace',
                    fontSize: '0.85rem',
                    whiteSpace: 'pre-wrap',
                  }}
                >
                  {renderedContent}
                </Typography>
              </Paper>
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
    </Box>
  )
}
