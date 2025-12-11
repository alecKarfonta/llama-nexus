import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Dialog,
  Box,
  TextField,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  InputAdornment,
  Chip,
  alpha,
  Fade,
} from '@mui/material'
import {
  Search as SearchIcon,
  Dashboard as DashboardIcon,
  Chat as ChatIcon,
  Memory as ModelsIcon,
  Settings as ConfigIcon,
  RocketLaunch as DeployIcon,
  Science as TestingIcon,
  Insights as MonitoringIcon,
  Description as TemplatesIcon,
  MenuBook as PromptsIcon,
  Inventory2 as RegistryIcon,
  Speed as BenchmarkIcon,
  BatchPrediction as BatchIcon,
  CompareArrows as CompareIcon,
  Api as ApiIcon,
  AccountTree as WorkflowIcon,
  Storage as KnowledgeIcon,
  Hub as GraphIcon,
  Folder as DocumentsIcon,
  TravelExplore as DiscoveryIcon,
  Mic as MicIcon,
  RecordVoiceOver as TTSIcon,
  TextFields as EmbeddingIcon,
  KeyboardReturn as EnterIcon,
} from '@mui/icons-material'

interface CommandItem {
  id: string
  label: string
  description: string
  path: string
  icon: React.ElementType
  color: string
  section: string
  keywords: string[]
}

const commands: CommandItem[] = [
  // Overview
  { id: 'dashboard', label: 'Dashboard', description: 'System overview and metrics', path: '/dashboard', icon: DashboardIcon, color: '#6366f1', section: 'Overview', keywords: ['home', 'overview', 'stats'] },
  { id: 'models', label: 'Models', description: 'Browse and download models', path: '/models', icon: ModelsIcon, color: '#8b5cf6', section: 'Overview', keywords: ['download', 'gguf', 'llm'] },
  { id: 'registry', label: 'Registry', description: 'Model metadata cache', path: '/registry', icon: RegistryIcon, color: '#a855f7', section: 'Overview', keywords: ['cache', 'metadata'] },
  
  // Deployment
  { id: 'deploy', label: 'Deploy LLM', description: 'Launch language model', path: '/deploy', icon: DeployIcon, color: '#f59e0b', section: 'Deployment', keywords: ['start', 'launch', 'run', 'llm'] },
  { id: 'embedding', label: 'Deploy Embedding', description: 'Launch embedding service', path: '/embedding-deploy', icon: EmbeddingIcon, color: '#06b6d4', section: 'Deployment', keywords: ['embed', 'vector'] },
  { id: 'stt', label: 'Deploy STT', description: 'Speech-to-text service', path: '/stt-deploy', icon: MicIcon, color: '#10b981', section: 'Deployment', keywords: ['speech', 'whisper', 'transcribe'] },
  { id: 'tts', label: 'Deploy TTS', description: 'Text-to-speech service', path: '/tts-deploy', icon: TTSIcon, color: '#8b5cf6', section: 'Deployment', keywords: ['voice', 'speak', 'audio'] },
  
  // Development
  { id: 'chat', label: 'Chat', description: 'Interactive conversation', path: '/chat', icon: ChatIcon, color: '#06b6d4', section: 'Development', keywords: ['conversation', 'talk', 'message'] },
  { id: 'prompts', label: 'Prompt Library', description: 'Manage prompt templates', path: '/prompts', icon: PromptsIcon, color: '#14b8a6', section: 'Development', keywords: ['template', 'system'] },
  { id: 'templates', label: 'Chat Templates', description: 'Jinja chat templates', path: '/templates', icon: TemplatesIcon, color: '#10b981', section: 'Development', keywords: ['jinja', 'format'] },
  { id: 'workflows', label: 'Workflows', description: 'Build multi-step pipelines', path: '/workflows', icon: WorkflowIcon, color: '#a855f7', section: 'Development', keywords: ['pipeline', 'chain', 'flow'] },
  { id: 'testing', label: 'Testing', description: 'Function calling playground', path: '/testing', icon: TestingIcon, color: '#f97316', section: 'Development', keywords: ['tools', 'function'] },
  { id: 'benchmark', label: 'Benchmark', description: 'Performance testing', path: '/benchmark', icon: BenchmarkIcon, color: '#ef4444', section: 'Development', keywords: ['performance', 'speed', 'bfcl'] },
  { id: 'batch', label: 'Batch Processing', description: 'Process multiple requests', path: '/batch', icon: BatchIcon, color: '#ec4899', section: 'Development', keywords: ['bulk', 'queue'] },
  { id: 'compare', label: 'Compare Models', description: 'Side-by-side comparison', path: '/compare', icon: CompareIcon, color: '#f43f5e', section: 'Development', keywords: ['diff', 'versus'] },
  
  // Knowledge & RAG
  { id: 'documents', label: 'Documents', description: 'Upload and manage documents', path: '/documents', icon: DocumentsIcon, color: '#10b981', section: 'Knowledge', keywords: ['upload', 'pdf', 'file'] },
  { id: 'knowledge-graph', label: 'Knowledge Graph', description: 'Entity relationships', path: '/knowledge-graph', icon: GraphIcon, color: '#6366f1', section: 'Knowledge', keywords: ['graph', 'entity', 'neo4j'] },
  { id: 'discovery', label: 'Discovery', description: 'Web search and review', path: '/discovery', icon: DiscoveryIcon, color: '#8b5cf6', section: 'Knowledge', keywords: ['search', 'web'] },
  { id: 'knowledge', label: 'RAG Search', description: 'Semantic retrieval', path: '/knowledge', icon: KnowledgeIcon, color: '#14b8a6', section: 'Knowledge', keywords: ['rag', 'semantic', 'vector'] },
  
  // Operations
  { id: 'monitoring', label: 'Monitoring', description: 'System metrics and logs', path: '/monitoring', icon: MonitoringIcon, color: '#0ea5e9', section: 'Operations', keywords: ['logs', 'metrics', 'status'] },
  { id: 'api-docs', label: 'API Docs', description: 'Interactive documentation', path: '/api-docs', icon: ApiIcon, color: '#14b8a6', section: 'Operations', keywords: ['swagger', 'openapi'] },
  { id: 'configuration', label: 'Settings', description: 'Application configuration', path: '/configuration', icon: ConfigIcon, color: '#64748b', section: 'Operations', keywords: ['config', 'preferences'] },
]

interface CommandPaletteProps {
  open: boolean
  onClose: () => void
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({ open, onClose }) => {
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)

  // Filter commands based on query
  const filteredCommands = useMemo(() => {
    if (!query.trim()) return commands
    
    const lowerQuery = query.toLowerCase()
    return commands.filter(cmd => 
      cmd.label.toLowerCase().includes(lowerQuery) ||
      cmd.description.toLowerCase().includes(lowerQuery) ||
      cmd.keywords.some(kw => kw.includes(lowerQuery))
    )
  }, [query])

  // Group filtered commands by section
  const groupedCommands = useMemo(() => {
    const groups: Record<string, CommandItem[]> = {}
    filteredCommands.forEach(cmd => {
      if (!groups[cmd.section]) groups[cmd.section] = []
      groups[cmd.section].push(cmd)
    })
    return groups
  }, [filteredCommands])

  // Reset selection when query changes
  useEffect(() => {
    setSelectedIndex(0)
  }, [query])

  // Reset state when dialog opens
  useEffect(() => {
    if (open) {
      setQuery('')
      setSelectedIndex(0)
    }
  }, [open])

  // Handle keyboard navigation
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault()
        setSelectedIndex(prev => Math.min(prev + 1, filteredCommands.length - 1))
        break
      case 'ArrowUp':
        e.preventDefault()
        setSelectedIndex(prev => Math.max(prev - 1, 0))
        break
      case 'Enter':
        e.preventDefault()
        if (filteredCommands[selectedIndex]) {
          navigate(filteredCommands[selectedIndex].path)
          onClose()
        }
        break
      case 'Escape':
        onClose()
        break
    }
  }, [filteredCommands, selectedIndex, navigate, onClose])

  const handleSelect = (path: string) => {
    navigate(path)
    onClose()
  }

  let globalIndex = 0

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      TransitionComponent={Fade}
      TransitionProps={{ timeout: 150 }}
      PaperProps={{
        sx: {
          bgcolor: 'rgba(15, 15, 35, 0.98)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: 3,
          overflow: 'hidden',
          mt: -10,
        }
      }}
      slotProps={{
        backdrop: {
          sx: {
            bgcolor: 'rgba(0, 0, 0, 0.7)',
            backdropFilter: 'blur(4px)',
          }
        }
      }}
    >
      {/* Search Input */}
      <Box sx={{ p: 2, borderBottom: '1px solid rgba(255, 255, 255, 0.08)' }}>
        <TextField
          autoFocus
          fullWidth
          placeholder="Search pages..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon sx={{ color: 'text.secondary' }} />
              </InputAdornment>
            ),
            endAdornment: (
              <InputAdornment position="end">
                <Chip 
                  label="esc" 
                  size="small" 
                  sx={{ 
                    height: 20, 
                    fontSize: '0.625rem',
                    bgcolor: 'rgba(255,255,255,0.05)',
                    color: 'text.secondary',
                  }} 
                />
              </InputAdornment>
            ),
          }}
          sx={{
            '& .MuiOutlinedInput-root': {
              bgcolor: 'rgba(255, 255, 255, 0.03)',
              borderRadius: 2,
              '& fieldset': { border: 'none' },
              '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.05)' },
              '&.Mui-focused': { bgcolor: 'rgba(255, 255, 255, 0.05)' },
            },
            '& input': {
              fontSize: '0.9375rem',
            }
          }}
        />
      </Box>

      {/* Results */}
      <Box sx={{ maxHeight: 400, overflowY: 'auto', p: 1 }}>
        {filteredCommands.length === 0 ? (
          <Box sx={{ p: 4, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              No results found for "{query}"
            </Typography>
          </Box>
        ) : (
          Object.entries(groupedCommands).map(([section, items]) => (
            <Box key={section} sx={{ mb: 1 }}>
              <Typography
                variant="caption"
                sx={{
                  px: 1.5,
                  py: 0.5,
                  display: 'block',
                  color: 'text.secondary',
                  fontSize: '0.6875rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.08em',
                  fontWeight: 600,
                }}
              >
                {section}
              </Typography>
              <List dense disablePadding>
                {items.map((cmd) => {
                  const currentIndex = globalIndex++
                  const isSelected = currentIndex === selectedIndex
                  const Icon = cmd.icon
                  
                  return (
                    <ListItem key={cmd.id} disablePadding>
                      <ListItemButton
                        selected={isSelected}
                        onClick={() => handleSelect(cmd.path)}
                        sx={{
                          py: 1,
                          px: 1.5,
                          borderRadius: 2,
                          mb: 0.25,
                          bgcolor: isSelected ? alpha(cmd.color, 0.12) : 'transparent',
                          '&:hover': {
                            bgcolor: alpha(cmd.color, 0.08),
                          },
                          '&.Mui-selected': {
                            bgcolor: alpha(cmd.color, 0.12),
                            '&:hover': {
                              bgcolor: alpha(cmd.color, 0.16),
                            },
                          },
                        }}
                      >
                        <ListItemIcon sx={{ minWidth: 36 }}>
                          <Box
                            sx={{
                              width: 28,
                              height: 28,
                              borderRadius: 1.5,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              bgcolor: alpha(cmd.color, 0.15),
                            }}
                          >
                            <Icon sx={{ fontSize: 16, color: cmd.color }} />
                          </Box>
                        </ListItemIcon>
                        <ListItemText
                          primary={cmd.label}
                          secondary={cmd.description}
                          primaryTypographyProps={{
                            fontSize: '0.875rem',
                            fontWeight: 500,
                          }}
                          secondaryTypographyProps={{
                            fontSize: '0.75rem',
                            color: 'text.secondary',
                          }}
                        />
                        {isSelected && (
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Chip
                              label="enter"
                              size="small"
                              icon={<EnterIcon sx={{ fontSize: '12px !important' }} />}
                              sx={{
                                height: 20,
                                fontSize: '0.625rem',
                                bgcolor: 'rgba(255,255,255,0.08)',
                                color: 'text.secondary',
                                '& .MuiChip-icon': { ml: 0.5 },
                              }}
                            />
                          </Box>
                        )}
                      </ListItemButton>
                    </ListItem>
                  )
                })}
              </List>
            </Box>
          ))
        )}
      </Box>

      {/* Footer */}
      <Box
        sx={{
          p: 1.5,
          borderTop: '1px solid rgba(255, 255, 255, 0.08)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          bgcolor: 'rgba(0, 0, 0, 0.2)',
        }}
      >
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Chip label="arrow" size="small" sx={{ height: 18, fontSize: '0.625rem', bgcolor: 'rgba(255,255,255,0.05)' }} />
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.6875rem' }}>
              navigate
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Chip label="enter" size="small" sx={{ height: 18, fontSize: '0.625rem', bgcolor: 'rgba(255,255,255,0.05)' }} />
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.6875rem' }}>
              select
            </Typography>
          </Box>
        </Box>
        <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.6875rem' }}>
          {filteredCommands.length} results
        </Typography>
      </Box>
    </Dialog>
  )
}

export default CommandPalette
