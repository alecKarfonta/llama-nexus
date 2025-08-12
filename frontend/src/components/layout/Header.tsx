import React from 'react'
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Button,
  useTheme,
} from '@mui/material'
import { 
  Menu as MenuIcon, 
  Refresh as RefreshIcon,
  Add as AddIcon,
} from '@mui/icons-material'
import { useLocation } from 'react-router-dom'

interface HeaderProps {
  onMenuClick: () => void
}

export const Header: React.FC<HeaderProps> = ({ onMenuClick }) => {
  const theme = useTheme()
  const location = useLocation()

  const getPageTitle = (path: string): string => {
    switch (path) {
      case '/dashboard':
        return 'Dashboard'
      case '/models':
        return 'Models'
      case '/chat':
        return 'Chat'
      case '/deploy':
        return 'Deploy'
      case '/testing':
        return 'Testing'
      case '/monitoring':
        return 'Monitoring'
      case '/configuration':
        return 'Settings'
      default:
        return 'Dashboard'
    }
  }

  const renderHeaderActions = (path: string) => {
    // Remove duplicate buttons from header since they're already in the page
    return null
  }
  
  return (
    <AppBar 
      position="fixed" 
      sx={{ 
        zIndex: (theme) => theme.zIndex.drawer + 1,
        background: 'white',
        color: theme.palette.text.primary,
        boxShadow: '0 1px 0 0 #e2e8f0',
        height: '48px'
      }}
      elevation={0}
    >
      <Toolbar sx={{ minHeight: '48px !important', height: '48px', py: 0 }}>
        <IconButton
          edge="start"
          color="inherit"
          aria-label="menu"
          onClick={onMenuClick}
          sx={{ mr: 2 }}
        >
          <MenuIcon />
        </IconButton>
        
        <Typography variant="h6" component="div" sx={{ 
          flexGrow: 1,
          fontWeight: 600,
          fontSize: '1rem'
        }}>
          {getPageTitle(location.pathname)}
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {renderHeaderActions(location.pathname)}
        </Box>
      </Toolbar>
    </AppBar>
  )
}
