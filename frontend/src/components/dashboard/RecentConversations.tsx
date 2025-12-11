/**
 * RecentConversations Component
 * Displays recent conversations with research initiative tracking
 */

import React, { useState, useEffect, useCallback } from 'react'
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemIcon,
  IconButton,
  Chip,
  TextField,
  InputAdornment,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  CircularProgress,
  Tooltip,
  alpha,
  Divider,
  Autocomplete,
  Stack,
} from '@mui/material'
import {
  Chat as ChatIcon,
  Search as SearchIcon,
  MoreVert as MoreIcon,
  Delete as DeleteIcon,
  Archive as ArchiveIcon,
  Download as DownloadIcon,
  Label as LabelIcon,
  Add as AddIcon,
  Refresh as RefreshIcon,
  AccessTime as TimeIcon,
  Message as MessageIcon,
  Science as ResearchIcon,
  TrendingUp as TrendingIcon,
  LocalOffer as TagIcon,
} from '@mui/icons-material'
import { useNavigate } from 'react-router-dom'
import { apiService } from '@/services/api'

interface Conversation {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
  model?: string
  tags: string[]
  is_archived: boolean
  preview?: string
}

interface ConversationStats {
  total_conversations: number
  total_messages: number
  recent_activity: number
  tags_used: string[]
  models_used: string[]
}

// Predefined research tags for quick categorization
const RESEARCH_TAGS = [
  { label: 'Research', color: '#8b5cf6' },
  { label: 'Development', color: '#06b6d4' },
  { label: 'Testing', color: '#f59e0b' },
  { label: 'Documentation', color: '#10b981' },
  { label: 'Bug Fix', color: '#ef4444' },
  { label: 'Feature', color: '#3b82f6' },
  { label: 'Experiment', color: '#ec4899' },
  { label: 'Analysis', color: '#14b8a6' },
]

const getTagColor = (tag: string): string => {
  const preset = RESEARCH_TAGS.find(t => t.label.toLowerCase() === tag.toLowerCase())
  if (preset) return preset.color
  // Generate consistent color from tag name
  let hash = 0
  for (let i = 0; i < tag.length; i++) {
    hash = tag.charCodeAt(i) + ((hash << 5) - hash)
  }
  const colors = ['#8b5cf6', '#06b6d4', '#f59e0b', '#10b981', '#ef4444', '#3b82f6', '#ec4899', '#14b8a6']
  return colors[Math.abs(hash) % colors.length]
}

interface RecentConversationsProps {
  limit?: number
  showSearch?: boolean
  showStats?: boolean
  compact?: boolean
}

export const RecentConversations: React.FC<RecentConversationsProps> = ({
  limit = 5,
  showSearch = true,
  showStats = true,
  compact = false,
}) => {
  const navigate = useNavigate()
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [stats, setStats] = useState<ConversationStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null)
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [tagDialogOpen, setTagDialogOpen] = useState(false)
  const [newTags, setNewTags] = useState<string[]>([])
  const [filterTag, setFilterTag] = useState<string | null>(null)

  // Fetch conversations
  const fetchConversations = useCallback(async () => {
    setLoading(true)
    try {
      const params: any = { limit, search: search || undefined }
      if (filterTag) params.tags = filterTag
      
      const response = await apiService.listConversations(params)
      setConversations(response.conversations)
    } catch (error) {
      console.error('Failed to fetch conversations:', error)
    } finally {
      setLoading(false)
    }
  }, [limit, search, filterTag])

  // Fetch stats
  const fetchStats = useCallback(async () => {
    try {
      const response = await apiService.get('/api/v1/conversations/stats')
      setStats(response.data)
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    }
  }, [])

  useEffect(() => {
    fetchConversations()
    if (showStats) {
      fetchStats()
    }
  }, [fetchConversations, fetchStats, showStats])

  // Format timestamp
  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    
    if (diff < 60000) return 'Just now'
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
    if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`
    return date.toLocaleDateString()
  }

  // Handle menu
  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, conversation: Conversation) => {
    event.stopPropagation()
    setMenuAnchor(event.currentTarget)
    setSelectedConversation(conversation)
  }

  const handleMenuClose = () => {
    setMenuAnchor(null)
    setSelectedConversation(null)
  }

  // Handle delete
  const handleDelete = async () => {
    if (!selectedConversation) return
    try {
      await apiService.deleteConversation(selectedConversation.id)
      setConversations(prev => prev.filter(c => c.id !== selectedConversation.id))
    } catch (error) {
      console.error('Failed to delete:', error)
    } finally {
      setDeleteDialogOpen(false)
      handleMenuClose()
    }
  }

  // Handle archive
  const handleArchive = async () => {
    if (!selectedConversation) return
    try {
      await apiService.updateConversation(selectedConversation.id, { is_archived: true })
      setConversations(prev => prev.filter(c => c.id !== selectedConversation.id))
    } catch (error) {
      console.error('Failed to archive:', error)
    } finally {
      handleMenuClose()
    }
  }

  // Handle tag update
  const handleUpdateTags = async () => {
    if (!selectedConversation) return
    try {
      await apiService.updateConversation(selectedConversation.id, { tags: newTags })
      setConversations(prev => 
        prev.map(c => c.id === selectedConversation.id ? { ...c, tags: newTags } : c)
      )
    } catch (error) {
      console.error('Failed to update tags:', error)
    } finally {
      setTagDialogOpen(false)
      handleMenuClose()
    }
  }

  // Handle export
  const handleExport = async (format: 'json' | 'markdown') => {
    if (!selectedConversation) return
    try {
      const content = await apiService.exportConversation(selectedConversation.id, format)
      const blob = new Blob([content], { type: format === 'json' ? 'application/json' : 'text/markdown' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${selectedConversation.title.slice(0, 30)}.${format === 'json' ? 'json' : 'md'}`
      a.click()
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Failed to export:', error)
    } finally {
      handleMenuClose()
    }
  }

  // Navigate to conversation
  const handleOpenConversation = (conversationId: string) => {
    navigate(`/chat?conversation=${conversationId}`)
  }

  // Open tag dialog
  const openTagDialog = () => {
    if (selectedConversation) {
      setNewTags(selectedConversation.tags || [])
    }
    setTagDialogOpen(true)
  }

  // Get all unique tags from conversations
  const allTags = Array.from(new Set([
    ...RESEARCH_TAGS.map(t => t.label),
    ...conversations.flatMap(c => c.tags || []),
    ...(stats?.tags_used || []),
  ]))

  return (
    <Box>
      {/* Stats Row */}
      {showStats && stats && (
        <Box sx={{ 
          display: 'flex', 
          gap: 2, 
          mb: 2, 
          pb: 2, 
          borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
          flexWrap: 'wrap',
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ChatIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
            <Typography variant="body2" color="text.secondary">
              <strong>{stats.total_conversations}</strong> conversations
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <MessageIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
            <Typography variant="body2" color="text.secondary">
              <strong>{stats.total_messages}</strong> messages
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TrendingIcon sx={{ fontSize: 16, color: '#10b981' }} />
            <Typography variant="body2" color="text.secondary">
              <strong>{stats.recent_activity}</strong> this week
            </Typography>
          </Box>
        </Box>
      )}

      {/* Search and Filter Bar */}
      {showSearch && (
        <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
          <TextField
            size="small"
            placeholder="Search conversations..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" sx={{ color: 'text.secondary' }} />
                </InputAdornment>
              ),
            }}
            sx={{ 
              flex: 1,
              minWidth: 200,
              '& .MuiOutlinedInput-root': {
                bgcolor: 'rgba(255, 255, 255, 0.02)',
                fontSize: '0.875rem',
              },
            }}
          />
          <Autocomplete
            size="small"
            options={allTags}
            value={filterTag}
            onChange={(_, value) => setFilterTag(value)}
            renderInput={(params) => (
              <TextField 
                {...params} 
                placeholder="Filter by tag" 
                sx={{ minWidth: 150 }}
              />
            )}
            sx={{ 
              '& .MuiOutlinedInput-root': {
                bgcolor: 'rgba(255, 255, 255, 0.02)',
              },
            }}
          />
          <Tooltip title="Refresh">
            <IconButton size="small" onClick={fetchConversations}>
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      )}

      {/* Quick Tag Filters */}
      {!compact && (
        <Box sx={{ display: 'flex', gap: 0.5, mb: 2, flexWrap: 'wrap' }}>
          {RESEARCH_TAGS.slice(0, 4).map((tag) => (
            <Chip
              key={tag.label}
              label={tag.label}
              size="small"
              onClick={() => setFilterTag(filterTag === tag.label ? null : tag.label)}
              sx={{
                height: 24,
                fontSize: '0.6875rem',
                bgcolor: filterTag === tag.label ? alpha(tag.color, 0.2) : 'transparent',
                border: '1px solid',
                borderColor: filterTag === tag.label ? tag.color : 'rgba(255, 255, 255, 0.1)',
                color: filterTag === tag.label ? tag.color : 'text.secondary',
                '&:hover': {
                  bgcolor: alpha(tag.color, 0.1),
                  borderColor: tag.color,
                },
              }}
            />
          ))}
        </Box>
      )}

      {/* Conversation List */}
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
          <CircularProgress size={24} />
        </Box>
      ) : conversations.length === 0 ? (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <ResearchIcon sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
          <Typography variant="body2" color="text.secondary">
            {search || filterTag ? 'No matching conversations' : 'No conversations yet'}
          </Typography>
          <Button
            size="small"
            startIcon={<AddIcon />}
            onClick={() => navigate('/chat')}
            sx={{ mt: 2, textTransform: 'none' }}
          >
            Start a conversation
          </Button>
        </Box>
      ) : (
        <List disablePadding sx={{ mx: -1 }}>
          {conversations.map((conversation, index) => (
            <ListItem
              key={conversation.id}
              disablePadding
              sx={{
                borderBottom: index < conversations.length - 1 ? '1px solid rgba(255, 255, 255, 0.04)' : 'none',
              }}
            >
              <ListItemButton
                onClick={() => handleOpenConversation(conversation.id)}
                sx={{ 
                  px: 1,
                  py: compact ? 1 : 1.5,
                  borderRadius: 1,
                  '&:hover': {
                    bgcolor: 'rgba(255, 255, 255, 0.03)',
                  },
                }}
              >
                <ListItemIcon sx={{ minWidth: 36 }}>
                  <Box
                    sx={{
                      width: 28,
                      height: 28,
                      borderRadius: 1,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      bgcolor: alpha('#06b6d4', 0.1),
                    }}
                  >
                    <ChatIcon sx={{ fontSize: 14, color: '#06b6d4' }} />
                  </Box>
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography
                        variant="body2"
                        sx={{
                          fontWeight: 500,
                          fontSize: '0.8125rem',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          flex: 1,
                        }}
                      >
                        {conversation.title}
                      </Typography>
                      {conversation.tags?.slice(0, 2).map((tag) => (
                        <Chip
                          key={tag}
                          label={tag}
                          size="small"
                          sx={{
                            height: 18,
                            fontSize: '0.625rem',
                            bgcolor: alpha(getTagColor(tag), 0.15),
                            color: getTagColor(tag),
                            border: 'none',
                            '& .MuiChip-label': { px: 0.75 },
                          }}
                        />
                      ))}
                    </Box>
                  }
                  secondary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mt: 0.5 }}>
                      <Typography
                        variant="caption"
                        sx={{ 
                          color: 'text.secondary',
                          display: 'flex',
                          alignItems: 'center',
                          gap: 0.5,
                        }}
                      >
                        <TimeIcon sx={{ fontSize: 12 }} />
                        {formatTime(conversation.updated_at)}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {conversation.message_count} messages
                      </Typography>
                      {conversation.model && (
                        <Typography 
                          variant="caption" 
                          sx={{ 
                            color: alpha('#fff', 0.4),
                            display: { xs: 'none', sm: 'block' },
                          }}
                        >
                          {conversation.model.split('/').pop()}
                        </Typography>
                      )}
                    </Box>
                  }
                />
                <IconButton
                  size="small"
                  onClick={(e) => handleMenuOpen(e, conversation)}
                  sx={{ ml: 1 }}
                >
                  <MoreIcon fontSize="small" />
                </IconButton>
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      )}

      {/* View All Button */}
      {conversations.length > 0 && (
        <Box sx={{ mt: 2, textAlign: 'center' }}>
          <Button
            size="small"
            onClick={() => navigate('/chat')}
            sx={{ 
              textTransform: 'none', 
              fontSize: '0.75rem',
              color: 'text.secondary',
              '&:hover': { color: 'primary.main' },
            }}
          >
            View all conversations
          </Button>
        </Box>
      )}

      {/* Context Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
        PaperProps={{
          sx: {
            bgcolor: 'background.paper',
            border: '1px solid rgba(255, 255, 255, 0.1)',
          },
        }}
      >
        <MenuItem onClick={openTagDialog}>
          <LabelIcon fontSize="small" sx={{ mr: 1.5 }} />
          Manage Tags
        </MenuItem>
        <MenuItem onClick={() => handleExport('markdown')}>
          <DownloadIcon fontSize="small" sx={{ mr: 1.5 }} />
          Export Markdown
        </MenuItem>
        <MenuItem onClick={() => handleExport('json')}>
          <DownloadIcon fontSize="small" sx={{ mr: 1.5 }} />
          Export JSON
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleArchive}>
          <ArchiveIcon fontSize="small" sx={{ mr: 1.5 }} />
          Archive
        </MenuItem>
        <MenuItem onClick={() => setDeleteDialogOpen(true)} sx={{ color: 'error.main' }}>
          <DeleteIcon fontSize="small" sx={{ mr: 1.5 }} />
          Delete
        </MenuItem>
      </Menu>

      {/* Delete Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Conversation?</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete "{selectedConversation?.title}"? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDelete} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Tag Management Dialog */}
      <Dialog 
        open={tagDialogOpen} 
        onClose={() => setTagDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TagIcon />
            Manage Tags
          </Box>
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Add tags to organize your research and conversations. Tags help you find related work later.
          </Typography>
          
          {/* Quick Tags */}
          <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
            Quick add:
          </Typography>
          <Box sx={{ display: 'flex', gap: 0.5, mb: 2, flexWrap: 'wrap' }}>
            {RESEARCH_TAGS.map((tag) => (
              <Chip
                key={tag.label}
                label={tag.label}
                size="small"
                onClick={() => {
                  if (!newTags.includes(tag.label)) {
                    setNewTags([...newTags, tag.label])
                  }
                }}
                disabled={newTags.includes(tag.label)}
                sx={{
                  bgcolor: alpha(tag.color, 0.1),
                  borderColor: tag.color,
                  color: tag.color,
                  '&:hover': { bgcolor: alpha(tag.color, 0.2) },
                  '&.Mui-disabled': {
                    bgcolor: alpha(tag.color, 0.3),
                    color: tag.color,
                    opacity: 0.7,
                  },
                }}
              />
            ))}
          </Box>

          {/* Custom Tags Input */}
          <Autocomplete
            multiple
            freeSolo
            options={allTags.filter(t => !newTags.includes(t))}
            value={newTags}
            onChange={(_, value) => setNewTags(value as string[])}
            renderTags={(value, getTagProps) =>
              value.map((option, index) => (
                <Chip
                  {...getTagProps({ index })}
                  key={option}
                  label={option}
                  size="small"
                  sx={{
                    bgcolor: alpha(getTagColor(option), 0.15),
                    color: getTagColor(option),
                  }}
                />
              ))
            }
            renderInput={(params) => (
              <TextField
                {...params}
                placeholder="Add custom tags..."
                helperText="Press Enter to add custom tags"
              />
            )}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTagDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleUpdateTags} variant="contained">
            Save Tags
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default RecentConversations
