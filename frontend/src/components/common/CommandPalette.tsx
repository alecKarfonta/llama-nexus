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
  alpha,
  Fade,
} from '@mui/material'
import {
  Search as SearchIcon,
} from '@mui/icons-material'
import { searchableNavigationItems, type SearchableNavigationItem } from '@/config/navigation'

type ScoredNavigationItem = SearchableNavigationItem & { score: number }

const normalizeQuery = (value: string) => value.trim().toLowerCase()

const scoreCommand = (command: SearchableNavigationItem, normalizedQuery: string): number => {
  const terms = normalizedQuery.split(/\s+/).filter(Boolean)

  return terms.reduce((score, term) => {
    let termScore = 0
    const label = command.label.toLowerCase()
    const title = command.title.toLowerCase()
    const description = command.description.toLowerCase()
    const section = command.section.toLowerCase()
    const path = command.path.toLowerCase()
    const parentLabels = command.parentLabels.map((label) => label.toLowerCase())
    const keywords = command.keywords.map((keyword) => keyword.toLowerCase())

    if (label === term || title === term) termScore += 120
    if (label.startsWith(term) || title.startsWith(term)) termScore += 80
    if (label.includes(term) || title.includes(term)) termScore += 50
    if (keywords.some((keyword) => keyword === term)) termScore += 45
    if (keywords.some((keyword) => keyword.includes(term))) termScore += 30
    if (parentLabels.some((label) => label.includes(term))) termScore += 28
    if (section.includes(term)) termScore += 24
    if (description.includes(term)) termScore += 18
    if (path.includes(term)) termScore += 14

    return score + termScore
  }, 0)
}

const getCommandContext = (command: SearchableNavigationItem): string => (
  [command.section, ...command.parentLabels].filter(Boolean).join(' / ')
)

interface CommandPaletteProps {
  open: boolean
  onClose: () => void
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({ open, onClose }) => {
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)

  // Filter commands based on query
  const filteredCommands = useMemo<ScoredNavigationItem[]>(() => {
    const normalizedQuery = normalizeQuery(query)

    if (!normalizedQuery) {
      return searchableNavigationItems.map((command) => ({ ...command, score: 0 }))
    }

    return searchableNavigationItems
      .map((command) => ({ ...command, score: scoreCommand(command, normalizedQuery) }))
      .filter((command) => command.score > 0)
      .sort((a, b) => (
        b.score - a.score ||
        a.section.localeCompare(b.section) ||
        a.label.localeCompare(b.label)
      ))
  }, [query])

  // Group filtered commands by section
  const groupedCommands = useMemo(() => {
    const groups: Record<string, ScoredNavigationItem[]> = {}
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

  useEffect(() => {
    setSelectedIndex((current) => Math.min(current, Math.max(filteredCommands.length - 1, 0)))
  }, [filteredCommands.length])

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
        if (filteredCommands.length > 0) {
          setSelectedIndex(prev => Math.min(prev + 1, filteredCommands.length - 1))
        }
        break
      case 'ArrowUp':
        e.preventDefault()
        if (filteredCommands.length > 0) {
          setSelectedIndex(prev => Math.max(prev - 1, 0))
        }
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
          borderRadius: '8px',
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
          }}
          sx={{
            '& .MuiOutlinedInput-root': {
              bgcolor: 'rgba(255, 255, 255, 0.03)',
                borderRadius: '8px',
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
                  letterSpacing: 0,
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
                            borderRadius: '8px',
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
                              borderRadius: '8px',
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
                          secondary={
                            <Box component="span" sx={{ display: 'grid', gap: 0.25, minWidth: 0 }}>
                              <Typography
                                component="span"
                                sx={{
                                  color: 'text.secondary',
                                  fontSize: '0.75rem',
                                  lineHeight: 1.35,
                                }}
                              >
                                {cmd.description}
                              </Typography>
                              <Typography
                                component="span"
                                sx={{
                                  color: alpha(cmd.color, 0.82),
                                  fontSize: '0.6875rem',
                                  lineHeight: 1.25,
                                }}
                              >
                                {getCommandContext(cmd)}
                              </Typography>
                            </Box>
                          }
                          primaryTypographyProps={{
                            fontSize: '0.875rem',
                            fontWeight: 500,
                          }}
                          secondaryTypographyProps={{
                            fontSize: '0.75rem',
                            color: 'text.secondary',
                          }}
                        />
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
          justifyContent: 'flex-end',
          alignItems: 'center',
          bgcolor: 'rgba(0, 0, 0, 0.2)',
        }}
      >
        <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.6875rem' }}>
          {filteredCommands.length} results
        </Typography>
      </Box>
    </Dialog>
  )
}

export default CommandPalette
