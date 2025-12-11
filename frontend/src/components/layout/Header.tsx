import React from 'react'
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Chip,
  Tooltip,
  alpha,
  useTheme,
} from '@mui/material'
import { 
  Menu as MenuIcon,
  Circle as CircleIcon,
  AutoAwesome as AutoAwesomeIcon,
  Search as SearchIcon,
} from '@mui/icons-material'
import { useLocation } from 'react-router-dom'

interface HeaderProps {
  onMenuClick: () => void
  onCommandPalette?: () => void
}

// Page title configuration with icons and descriptions
const pageConfig: Record<string, { title: string; description: string; color: string }> = {
  '/dashboard': { title: 'Dashboard', description: 'System overview and metrics', color: '#6366f1' },
  '/models': { title: 'Models', description: 'Manage your ML models', color: '#8b5cf6' },
  '/registry': { title: 'Registry', description: 'Model metadata and cache', color: '#a855f7' },
  '/chat': { title: 'Chat', description: 'Interactive conversations', color: '#06b6d4' },
  '/templates': { title: 'Templates', description: 'Chat template management', color: '#10b981' },
  '/prompts': { title: 'Prompts', description: 'Prompt library', color: '#14b8a6' },
  '/deploy': { title: 'Deploy', description: 'Configure and launch models', color: '#f59e0b' },
  '/testing': { title: 'Testing', description: 'Function calling playground', color: '#f97316' },
  '/benchmark': { title: 'Benchmark', description: 'Performance testing', color: '#ef4444' },
  '/batch': { title: 'Batch', description: 'Batch processing jobs', color: '#ec4899' },
  '/compare': { title: 'Compare', description: 'Model comparison', color: '#f43f5e' },
  '/workflows': { title: 'Workflows', description: 'Multi-step LLM pipelines', color: '#a855f7' },
  '/knowledge': { title: 'RAG Search', description: 'Semantic search and retrieval', color: '#06b6d4' },
  '/knowledge-graph': { title: 'Knowledge Graph', description: 'Entity relationships visualization', color: '#6366f1' },
  '/documents': { title: 'Documents', description: 'Document management', color: '#10b981' },
  '/discovery': { title: 'Discovery', description: 'Web search and document review', color: '#8b5cf6' },
  '/monitoring': { title: 'Monitoring', description: 'System monitoring', color: '#0ea5e9' },
  '/api-docs': { title: 'API Docs', description: 'Interactive API documentation', color: '#14b8a6' },
  '/configuration': { title: 'Settings', description: 'Application settings', color: '#64748b' },
}

export const Header: React.FC<HeaderProps> = ({ onMenuClick, onCommandPalette }) => {
  const theme = useTheme()
  const location = useLocation()

  const config = pageConfig[location.pathname] || { 
    title: 'Dashboard', 
    description: 'System overview', 
    color: '#6366f1' 
  }

  return (
    <AppBar 
      position="fixed" 
      sx={{ 
        zIndex: (theme) => theme.zIndex.drawer + 1,
        background: 'linear-gradient(90deg, rgba(15, 15, 35, 0.95) 0%, rgba(26, 26, 46, 0.95) 100%)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
        height: '64px'
      }}
      elevation={0}
    >
      <Toolbar sx={{ 
        minHeight: '64px !important', 
        height: '64px', 
        py: 0,
        px: { xs: 2, sm: 3 },
      }}>
        {/* Menu Button */}
        <IconButton
          edge="start"
          color="inherit"
          aria-label="menu"
          onClick={onMenuClick}
          sx={{ 
            mr: 2,
            width: 40,
            height: 40,
            borderRadius: 2,
            border: '1px solid rgba(255, 255, 255, 0.08)',
            background: 'rgba(255, 255, 255, 0.03)',
            transition: 'all 0.2s ease-in-out',
            '&:hover': {
              background: 'rgba(255, 255, 255, 0.08)',
              borderColor: 'rgba(255, 255, 255, 0.15)',
              transform: 'scale(1.05)',
            }
          }}
        >
          <MenuIcon sx={{ fontSize: 20 }} />
        </IconButton>
        
        {/* Logo and Brand */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Box sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 36,
            height: 36,
            borderRadius: 2,
            background: 'linear-gradient(135deg, #6366f1 0%, #a855f7 100%)',
            boxShadow: '0 4px 12px rgba(99, 102, 241, 0.4)',
          }}>
            <AutoAwesomeIcon sx={{ fontSize: 20, color: 'white' }} />
          </Box>
          
          <Box sx={{ display: { xs: 'none', sm: 'block' } }}>
            <Typography 
              variant="h6" 
              sx={{ 
                fontWeight: 700,
                fontSize: '1.125rem',
                letterSpacing: '-0.02em',
                background: 'linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                lineHeight: 1,
              }}
            >
              Llama Nexus
            </Typography>
            <Typography
              variant="caption"
              sx={{
                color: 'text.secondary',
                fontSize: '0.6875rem',
                letterSpacing: '0.02em',
              }}
            >
              ML Model Manager
            </Typography>
          </Box>
        </Box>

        {/* Divider */}
        <Box sx={{ 
          width: 1, 
          height: 32, 
          bgcolor: 'rgba(255, 255, 255, 0.08)', 
          mx: 3,
          display: { xs: 'none', md: 'block' }
        }} />

        {/* Page Title */}
        <Box sx={{ flexGrow: 1, display: { xs: 'none', md: 'flex' }, alignItems: 'center', gap: 2 }}>
          <Box sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1.5,
          }}>
            <Box sx={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              bgcolor: config.color,
              boxShadow: `0 0 12px ${config.color}`,
            }} />
            <Typography 
              variant="h6" 
              sx={{ 
                fontWeight: 600,
                fontSize: '1rem',
                color: 'text.primary',
              }}
            >
              {config.title}
            </Typography>
          </Box>
          <Typography
            variant="body2"
            sx={{
              color: 'text.secondary',
              fontSize: '0.8125rem',
            }}
          >
            {config.description}
          </Typography>
        </Box>

        {/* Mobile Title */}
        <Box sx={{ flexGrow: 1, display: { xs: 'block', md: 'none' } }}>
          <Typography 
            variant="h6" 
            sx={{ 
              fontWeight: 600,
              fontSize: '1rem',
              color: 'text.primary',
            }}
          >
            {config.title}
          </Typography>
        </Box>

        {/* Search / Command Palette */}
        <Tooltip title="Search (Cmd+K)">
          <Box
            onClick={onCommandPalette}
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              px: 1.5,
              py: 0.75,
              borderRadius: 2,
              bgcolor: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              cursor: 'pointer',
              transition: 'all 0.2s ease-in-out',
              '&:hover': {
                bgcolor: 'rgba(255, 255, 255, 0.06)',
                borderColor: 'rgba(255, 255, 255, 0.12)',
              },
              mr: 2,
            }}
          >
            <SearchIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
            <Typography
              variant="body2"
              sx={{
                color: 'text.secondary',
                fontSize: '0.8125rem',
                display: { xs: 'none', md: 'block' },
              }}
            >
              Search...
            </Typography>
            <Chip
              label={navigator.platform.includes('Mac') ? 'Cmd+K' : 'Ctrl+K'}
              size="small"
              sx={{
                height: 20,
                fontSize: '0.625rem',
                bgcolor: 'rgba(255,255,255,0.06)',
                color: 'text.secondary',
                display: { xs: 'none', sm: 'flex' },
              }}
            />
          </Box>
        </Tooltip>

        {/* Status Indicators */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <Tooltip title="System Online">
            <Chip
              icon={<CircleIcon sx={{ fontSize: '8px !important', color: '#10b981 !important' }} />}
              label="Online"
              size="small"
              sx={{
                height: 28,
                bgcolor: 'rgba(16, 185, 129, 0.1)',
                border: '1px solid rgba(16, 185, 129, 0.2)',
                color: '#34d399',
                fontWeight: 500,
                fontSize: '0.75rem',
                '& .MuiChip-icon': {
                  ml: 1,
                },
                display: { xs: 'none', sm: 'flex' },
              }}
            />
          </Tooltip>
        </Box>
      </Toolbar>
    </AppBar>
  )
}
