/**
 * ConversationSidebar Component
 * Displays and manages saved conversations
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Typography,
  TextField,
  InputAdornment,
  Divider,
  Button,
  Tooltip,
  CircularProgress,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Search as SearchIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Archive as ArchiveIcon,
  Download as DownloadIcon,
  MoreVert as MoreIcon,
  Chat as ChatIcon,
} from '@mui/icons-material';
import type { ConversationListItem, ConversationListResponse } from '@/types/api';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || '';

interface ConversationSidebarProps {
  open: boolean;
  onClose: () => void;
  onSelectConversation: (conversation: ConversationListItem) => void;
  onNewConversation: () => void;
  currentConversationId?: string;
}

export const ConversationSidebar: React.FC<ConversationSidebarProps> = ({
  open,
  onClose,
  onSelectConversation,
  onNewConversation,
  currentConversationId,
}) => {
  const [conversations, setConversations] = useState<ConversationListItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState('');
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [selectedItem, setSelectedItem] = useState<ConversationListItem | null>(null);
  const [deleteDialog, setDeleteDialog] = useState(false);

  // Fetch conversations
  const fetchConversations = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (search) params.append('search', search);
      params.append('limit', '50');
      
      const response = await fetch(`${BACKEND_URL}/api/v1/conversations?${params}`);
      if (response.ok) {
        const data: ConversationListResponse = await response.json();
        setConversations(data.conversations);
      }
    } catch (error) {
      console.error('Failed to fetch conversations:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (open) {
      fetchConversations();
    }
  }, [open, search]);

  // Handle menu open
  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, item: ConversationListItem) => {
    event.stopPropagation();
    setMenuAnchor(event.currentTarget);
    setSelectedItem(item);
  };

  // Handle menu close
  const handleMenuClose = () => {
    setMenuAnchor(null);
    setSelectedItem(null);
  };

  // Handle delete conversation
  const handleDelete = async () => {
    if (!selectedItem) return;
    
    try {
      const response = await fetch(`${BACKEND_URL}/api/v1/conversations/${selectedItem.id}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        setConversations(prev => prev.filter(c => c.id !== selectedItem.id));
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error);
    } finally {
      setDeleteDialog(false);
      handleMenuClose();
    }
  };

  // Handle archive conversation
  const handleArchive = async () => {
    if (!selectedItem) return;
    
    try {
      const response = await fetch(`${BACKEND_URL}/api/v1/conversations/${selectedItem.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ is_archived: true }),
      });
      if (response.ok) {
        setConversations(prev => prev.filter(c => c.id !== selectedItem.id));
      }
    } catch (error) {
      console.error('Failed to archive conversation:', error);
    } finally {
      handleMenuClose();
    }
  };

  // Handle export conversation
  const handleExport = async (format: 'json' | 'markdown') => {
    if (!selectedItem) return;
    
    try {
      const response = await fetch(
        `${BACKEND_URL}/api/v1/conversations/${selectedItem.id}/export?format=${format}`
      );
      if (response.ok) {
        const data = await response.json();
        const blob = new Blob([data.content], { type: data.content_type });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `conversation-${selectedItem.id.slice(0, 8)}.${format === 'json' ? 'json' : 'md'}`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Failed to export conversation:', error);
    } finally {
      handleMenuClose();
    }
  };

  // Format timestamp
  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <>
      <Drawer
        anchor="left"
        open={open}
        onClose={onClose}
        sx={{
          '& .MuiDrawer-paper': {
            width: 320,
            backgroundColor: 'background.default',
            borderRight: '1px solid',
            borderColor: 'divider',
          },
        }}
      >
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Conversations
          </Typography>
          
          {/* New conversation button */}
          <Button
            fullWidth
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => {
              onNewConversation();
              onClose();
            }}
            sx={{ mb: 2 }}
          >
            New Conversation
          </Button>
          
          {/* Search */}
          <TextField
            fullWidth
            size="small"
            placeholder="Search conversations..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" />
                </InputAdornment>
              ),
            }}
            sx={{ mb: 2 }}
          />
        </Box>
        
        <Divider />
        
        {/* Conversation list */}
        <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
              <CircularProgress size={24} />
            </Box>
          ) : conversations.length === 0 ? (
            <Box sx={{ p: 4, textAlign: 'center' }}>
              <ChatIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
              <Typography variant="body2" color="text.secondary">
                No conversations yet
              </Typography>
            </Box>
          ) : (
            <List disablePadding>
              {conversations.map((conversation) => (
                <ListItem
                  key={conversation.id}
                  disablePadding
                  sx={{
                    borderLeft: conversation.id === currentConversationId ? '3px solid' : 'none',
                    borderColor: 'primary.main',
                  }}
                >
                  <ListItemButton
                    onClick={() => {
                      onSelectConversation(conversation);
                      onClose();
                    }}
                    selected={conversation.id === currentConversationId}
                    sx={{ pr: 6 }}
                  >
                    <ListItemText
                      primary={
                        <Typography
                          variant="body2"
                          sx={{
                            fontWeight: conversation.id === currentConversationId ? 600 : 400,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {conversation.title}
                        </Typography>
                      }
                      secondary={
                        <Box component="span" sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="caption" color="text.secondary">
                            {conversation.message_count} messages
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {formatTime(conversation.updated_at)}
                          </Typography>
                        </Box>
                      }
                    />
                  </ListItemButton>
                  <ListItemSecondaryAction>
                    <IconButton
                      size="small"
                      onClick={(e) => handleMenuOpen(e, conversation)}
                    >
                      <MoreIcon fontSize="small" />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          )}
        </Box>
      </Drawer>

      {/* Context menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => handleExport('json')}>
          <DownloadIcon fontSize="small" sx={{ mr: 1 }} />
          Export as JSON
        </MenuItem>
        <MenuItem onClick={() => handleExport('markdown')}>
          <DownloadIcon fontSize="small" sx={{ mr: 1 }} />
          Export as Markdown
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleArchive}>
          <ArchiveIcon fontSize="small" sx={{ mr: 1 }} />
          Archive
        </MenuItem>
        <MenuItem onClick={() => setDeleteDialog(true)} sx={{ color: 'error.main' }}>
          <DeleteIcon fontSize="small" sx={{ mr: 1 }} />
          Delete
        </MenuItem>
      </Menu>

      {/* Delete confirmation dialog */}
      <Dialog open={deleteDialog} onClose={() => setDeleteDialog(false)}>
        <DialogTitle>Delete Conversation?</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete "{selectedItem?.title}"? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog(false)}>Cancel</Button>
          <Button onClick={handleDelete} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default ConversationSidebar;
