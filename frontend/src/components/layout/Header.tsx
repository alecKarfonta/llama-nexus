import React from 'react'
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Tooltip,
} from '@mui/material'
import { 
  Menu as MenuIcon,
  AutoAwesome as AutoAwesomeIcon,
  Search as SearchIcon,
} from '@mui/icons-material'
import { useLocation } from 'react-router-dom'
import { ServiceStatusBar } from '@/components/common/ServiceStatusBar'
import { getRouteMeta } from '@/config/navigation'

interface HeaderProps {
  onMenuClick: () => void
  onCommandPalette?: () => void
}

export const Header: React.FC<HeaderProps> = ({ onMenuClick, onCommandPalette }) => {
  const location = useLocation()
  const config = getRouteMeta(location.pathname)

  return (
    <AppBar 
      data-app-chrome="header"
      position="fixed" 
      sx={{ 
        zIndex: (theme) => theme.zIndex.drawer + 1,
        bgcolor: 'rgba(15, 15, 35, 0.97)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
        boxShadow: 'none',
        height: '56px'
      }}
      elevation={0}
    >
      <Toolbar sx={{ 
        minHeight: '56px !important',
        height: '56px',
        py: 0,
        px: { xs: 1.5, sm: 2 },
      }}>
        {/* Menu Button */}
        <IconButton
          edge="start"
          color="inherit"
          aria-label="menu"
          onClick={onMenuClick}
          sx={{ 
            mr: 1.25,
            width: 34,
            height: 34,
            borderRadius: 2,
            border: '1px solid rgba(255, 255, 255, 0.08)',
            background: 'rgba(255, 255, 255, 0.03)',
            transition: 'background-color 0.15s, border-color 0.15s',
            '&:hover': {
              background: 'rgba(255, 255, 255, 0.08)',
              borderColor: 'rgba(255, 255, 255, 0.15)',
            }
          }}
        >
          <MenuIcon sx={{ fontSize: 20 }} />
        </IconButton>
        
        {/* Logo and Brand */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25 }}>
          <Box sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 30,
            height: 30,
            borderRadius: 2,
            bgcolor: '#6366f1',
          }}>
            <AutoAwesomeIcon sx={{ fontSize: 17, color: 'white' }} />
          </Box>
          
          <Box sx={{ display: { xs: 'none', sm: 'block' } }}>
            <Typography 
              variant="h6" 
              sx={{ 
                fontWeight: 700,
                fontSize: '1rem',
                letterSpacing: 0,
                color: 'text.primary',
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
                letterSpacing: 0,
              }}
            >
              ML Model Manager
            </Typography>
          </Box>
        </Box>

        {/* Divider */}
        <Box sx={{ 
          width: '1px',
          height: 28,
          bgcolor: 'rgba(255, 255, 255, 0.08)', 
          mx: 2,
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
            }} />
            <Typography 
              variant="h6" 
              sx={{ 
                fontWeight: 600,
                fontSize: '0.9375rem',
                color: 'text.primary',
              }}
            >
              {config.title}
            </Typography>
          </Box>
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
        <Tooltip title="Open search">
          <Box
            onClick={onCommandPalette}
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.75,
              px: 1.25,
              py: 0.5,
              borderRadius: 2,
              bgcolor: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              cursor: 'pointer',
              transition: 'background-color 0.15s, border-color 0.15s',
              '&:hover': {
                bgcolor: 'rgba(255, 255, 255, 0.06)',
                borderColor: 'rgba(255, 255, 255, 0.12)',
              },
              mr: 1.5,
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
          </Box>
        </Tooltip>

        {/* Service Status */}
        <Box sx={{ display: { xs: 'none', md: 'block' } }}>
          <ServiceStatusBar compact />
        </Box>
      </Toolbar>
    </AppBar>
  )
}
