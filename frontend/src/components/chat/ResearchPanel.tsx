/**
 * ResearchPanel Component
 * Displays research session progress, saved notes, and tool call history
 */

import React, { useState, useEffect } from 'react'
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Chip,
  Collapse,
  Divider,
  Button,
  Tooltip,
  LinearProgress,
  alpha,
  Badge,
} from '@mui/material'
import {
  Science as ResearchIcon,
  Search as SearchIcon,
  Language as WebIcon,
  Storage as KnowledgeIcon,
  Note as NoteIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  ContentCopy as CopyIcon,
  Download as DownloadIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  HourglassEmpty as PendingIcon,
  Link as LinkIcon,
  LocalOffer as TagIcon,
} from '@mui/icons-material'
import { ToolsService } from '@/services/tools'

interface ToolCallRecord {
  id: string
  name: string
  arguments: Record<string, any>
  result?: string
  status: 'pending' | 'executing' | 'success' | 'error'
  timestamp: string
  duration?: number
}

interface ResearchPanelProps {
  toolCalls: ToolCallRecord[]
  isResearching: boolean
  onClearNotes?: () => void
  onExportNotes?: () => void
  compact?: boolean
}

const getToolIcon = (name: string) => {
  switch (name) {
    case 'web_search': return <WebIcon />;
    case 'knowledge_search': return <KnowledgeIcon />;
    case 'fetch_url': return <LinkIcon />;
    case 'save_research_note': return <NoteIcon />;
    default: return <SearchIcon />;
  }
};

const getToolColor = (name: string) => {
  switch (name) {
    case 'web_search': return '#3b82f6';
    case 'knowledge_search': return '#a855f7';
    case 'fetch_url': return '#06b6d4';
    case 'save_research_note': return '#10b981';
    default: return '#64748b';
  }
};

export const ResearchPanel: React.FC<ResearchPanelProps> = ({
  toolCalls,
  isResearching,
  onClearNotes,
  onExportNotes,
  compact = false,
}) => {
  const [expanded, setExpanded] = useState(!compact);
  const [notesExpanded, setNotesExpanded] = useState(true);
  const [queriesExpanded, setQueriesExpanded] = useState(true);
  const [notes, setNotes] = useState(ToolsService.getResearchNotes());

  // Refresh notes periodically
  useEffect(() => {
    const interval = setInterval(() => {
      setNotes(ToolsService.getResearchNotes());
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Stats
  const successfulQueries = toolCalls.filter(t => t.status === 'success').length;
  const webSearches = toolCalls.filter(t => t.name === 'web_search').length;
  const knowledgeSearches = toolCalls.filter(t => t.name === 'knowledge_search').length;

  // Export notes as markdown
  const handleExport = () => {
    const markdown = `# Research Notes\n\nGenerated: ${new Date().toLocaleString()}\n\n---\n\n` +
      notes.map(note => 
        `## ${note.title}\n\n${note.content}\n\n` +
        (note.sources.length > 0 ? `**Sources:** ${note.sources.join(', ')}\n\n` : '') +
        (note.tags.length > 0 ? `**Tags:** ${note.tags.map(t => `#${t}`).join(' ')}\n\n` : '') +
        `*Saved: ${new Date(note.timestamp).toLocaleString()}*\n\n---\n\n`
      ).join('');

    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `research-notes-${new Date().toISOString().split('T')[0]}.md`;
    a.click();
    URL.revokeObjectURL(url);
    
    if (onExportNotes) onExportNotes();
  };

  // Clear notes
  const handleClear = () => {
    ToolsService.clearResearchNotes();
    setNotes([]);
    if (onClearNotes) onClearNotes();
  };

  if (toolCalls.length === 0 && notes.length === 0 && !isResearching) {
    return null;
  }

  return (
    <Paper
      variant="outlined"
      sx={{
        mb: 2,
        bgcolor: alpha('#8b5cf6', 0.03),
        borderColor: alpha('#8b5cf6', 0.2),
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          px: 2,
          py: 1,
          cursor: 'pointer',
          '&:hover': { bgcolor: alpha('#8b5cf6', 0.05) },
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <Badge
          badgeContent={isResearching ? '...' : successfulQueries}
          color="primary"
          sx={{ mr: 1.5 }}
        >
          <Box
            sx={{
              width: 28,
              height: 28,
              borderRadius: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: alpha('#8b5cf6', 0.15),
              color: '#8b5cf6',
            }}
          >
            <ResearchIcon sx={{ fontSize: 16 }} />
          </Box>
        </Badge>
        
        <Typography variant="subtitle2" sx={{ fontWeight: 600, color: '#8b5cf6' }}>
          Research Session
        </Typography>

        {/* Quick Stats */}
        <Box sx={{ display: 'flex', gap: 0.75, ml: 2 }}>
          {webSearches > 0 && (
            <Chip
              size="small"
              icon={<WebIcon sx={{ fontSize: 12 }} />}
              label={webSearches}
              sx={{
                height: 20,
                fontSize: '0.625rem',
                bgcolor: alpha('#3b82f6', 0.1),
                color: '#3b82f6',
              }}
            />
          )}
          {knowledgeSearches > 0 && (
            <Chip
              size="small"
              icon={<KnowledgeIcon sx={{ fontSize: 12 }} />}
              label={knowledgeSearches}
              sx={{
                height: 20,
                fontSize: '0.625rem',
                bgcolor: alpha('#a855f7', 0.1),
                color: '#a855f7',
              }}
            />
          )}
          {notes.length > 0 && (
            <Chip
              size="small"
              icon={<NoteIcon sx={{ fontSize: 12 }} />}
              label={notes.length}
              sx={{
                height: 20,
                fontSize: '0.625rem',
                bgcolor: alpha('#10b981', 0.1),
                color: '#10b981',
              }}
            />
          )}
        </Box>

        {isResearching && (
          <Typography variant="caption" sx={{ ml: 2, color: '#8b5cf6', fontStyle: 'italic' }}>
            Researching...
          </Typography>
        )}

        <IconButton size="small" sx={{ ml: 'auto' }}>
          {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        </IconButton>
      </Box>

      {isResearching && <LinearProgress sx={{ height: 2 }} color="secondary" />}

      {/* Content */}
      <Collapse in={expanded}>
        <Box sx={{ px: 2, py: 1.5, borderTop: '1px solid', borderColor: alpha('#8b5cf6', 0.1) }}>
          {/* Research Notes Section */}
          {notes.length > 0 && (
            <Box sx={{ mb: 2 }}>
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  mb: 1,
                  cursor: 'pointer',
                }}
                onClick={() => setNotesExpanded(!notesExpanded)}
              >
                <NoteIcon sx={{ fontSize: 16, color: '#10b981', mr: 1 }} />
                <Typography variant="caption" sx={{ fontWeight: 600, color: '#10b981' }}>
                  Research Notes ({notes.length})
                </Typography>
                <Box sx={{ ml: 'auto', display: 'flex', gap: 0.5 }}>
                  <Tooltip title="Export as Markdown">
                    <IconButton size="small" onClick={(e) => { e.stopPropagation(); handleExport(); }}>
                      <DownloadIcon sx={{ fontSize: 14 }} />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Clear Notes">
                    <IconButton size="small" onClick={(e) => { e.stopPropagation(); handleClear(); }}>
                      <DeleteIcon sx={{ fontSize: 14 }} />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>
              
              <Collapse in={notesExpanded}>
                <List dense disablePadding>
                  {notes.map((note) => (
                    <ListItem
                      key={note.id}
                      sx={{
                        bgcolor: alpha('#10b981', 0.05),
                        borderRadius: 1,
                        mb: 0.5,
                        border: '1px solid',
                        borderColor: alpha('#10b981', 0.1),
                      }}
                    >
                      <ListItemIcon sx={{ minWidth: 32 }}>
                        <NoteIcon sx={{ fontSize: 16, color: '#10b981' }} />
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Typography variant="body2" sx={{ fontWeight: 500, fontSize: '0.8125rem' }}>
                            {note.title}
                          </Typography>
                        }
                        secondary={
                          <Box>
                            <Typography
                              variant="caption"
                              sx={{
                                color: 'text.secondary',
                                display: '-webkit-box',
                                WebkitLineClamp: 2,
                                WebkitBoxOrient: 'vertical',
                                overflow: 'hidden',
                              }}
                            >
                              {note.content}
                            </Typography>
                            {note.tags.length > 0 && (
                              <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5 }}>
                                {note.tags.map((tag) => (
                                  <Chip
                                    key={tag}
                                    label={tag}
                                    size="small"
                                    sx={{ height: 16, fontSize: '0.5625rem' }}
                                  />
                                ))}
                              </Box>
                            )}
                          </Box>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </Collapse>
            </Box>
          )}

          {/* Tool Calls Section */}
          {toolCalls.length > 0 && (
            <Box>
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  mb: 1,
                  cursor: 'pointer',
                }}
                onClick={() => setQueriesExpanded(!queriesExpanded)}
              >
                <SearchIcon sx={{ fontSize: 16, color: 'text.secondary', mr: 1 }} />
                <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                  Queries ({toolCalls.length})
                </Typography>
              </Box>
              
              <Collapse in={queriesExpanded}>
                <List dense disablePadding>
                  {toolCalls.slice(-5).reverse().map((call) => (
                    <ListItem
                      key={call.id}
                      sx={{
                        bgcolor: alpha(getToolColor(call.name), 0.05),
                        borderRadius: 1,
                        mb: 0.5,
                        py: 0.5,
                      }}
                    >
                      <ListItemIcon sx={{ minWidth: 28 }}>
                        <Box sx={{ color: getToolColor(call.name), '& .MuiSvgIcon-root': { fontSize: 14 } }}>
                          {getToolIcon(call.name)}
                        </Box>
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Typography variant="caption" sx={{ fontWeight: 500 }}>
                            {call.arguments.query || call.arguments.url || call.arguments.title || call.name}
                          </Typography>
                        }
                        secondary={
                          <Typography variant="caption" sx={{ color: 'text.disabled', fontSize: '0.625rem' }}>
                            {call.name} {call.duration ? `(${call.duration}ms)` : ''}
                          </Typography>
                        }
                      />
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        {call.status === 'success' && <SuccessIcon sx={{ fontSize: 14, color: '#10b981' }} />}
                        {call.status === 'error' && <ErrorIcon sx={{ fontSize: 14, color: '#ef4444' }} />}
                        {call.status === 'executing' && <PendingIcon sx={{ fontSize: 14, color: '#f59e0b' }} />}
                      </Box>
                    </ListItem>
                  ))}
                </List>
              </Collapse>
            </Box>
          )}
        </Box>
      </Collapse>
    </Paper>
  );
};

export default ResearchPanel;
