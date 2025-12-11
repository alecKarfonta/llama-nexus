import React from 'react'
import { useLocation, useNavigate, Link as RouterLink } from 'react-router-dom'
import { Breadcrumbs as MuiBreadcrumbs, Link, Typography, Box, alpha } from '@mui/material'
import {
  Home as HomeIcon,
  ChevronRight as ChevronRightIcon,
} from '@mui/icons-material'

// Route to breadcrumb label mapping
const routeLabels: Record<string, string> = {
  '': 'Home',
  'dashboard': 'Dashboard',
  'chat': 'Chat',
  'models': 'Models',
  'registry': 'Registry',
  'deploy': 'Deploy LLM',
  'embedding-deploy': 'Deploy Embedding',
  'stt-deploy': 'Deploy STT',
  'tts-deploy': 'Deploy TTS',
  'prompts': 'Prompt Library',
  'templates': 'Chat Templates',
  'workflows': 'Workflows',
  'testing': 'Testing',
  'benchmark': 'Benchmark',
  'batch': 'Batch Processing',
  'compare': 'Model Comparison',
  'documents': 'Documents',
  'knowledge-graph': 'Knowledge Graph',
  'discovery': 'Discovery',
  'knowledge': 'RAG Search',
  'monitoring': 'Monitoring',
  'api-docs': 'API Documentation',
  'configuration': 'Settings',
}

// Route hierarchy for grouping
const routeGroups: Record<string, string[]> = {
  'deploy': ['deploy', 'embedding-deploy', 'stt-deploy', 'tts-deploy'],
  'knowledge': ['documents', 'knowledge-graph', 'discovery', 'knowledge'],
  'development': ['chat', 'prompts', 'templates', 'workflows', 'testing', 'benchmark', 'batch', 'compare'],
}

interface BreadcrumbsProps {
  maxItems?: number
  showHome?: boolean
}

export const Breadcrumbs: React.FC<BreadcrumbsProps> = ({ 
  maxItems = 3,
  showHome = true 
}) => {
  const location = useLocation()
  const navigate = useNavigate()
  
  // Parse the current path
  const pathSegments = location.pathname.split('/').filter(Boolean)
  
  // Build breadcrumb items
  const items: Array<{ label: string; path: string; isLast: boolean }> = []
  
  // Add home if enabled
  if (showHome) {
    items.push({ label: 'Home', path: '/dashboard', isLast: pathSegments.length === 0 })
  }
  
  // Add path segments
  let currentPath = ''
  pathSegments.forEach((segment, index) => {
    currentPath += `/${segment}`
    const label = routeLabels[segment] || segment.charAt(0).toUpperCase() + segment.slice(1)
    items.push({
      label,
      path: currentPath,
      isLast: index === pathSegments.length - 1
    })
  })

  // Don't render if we're at root or only have home
  if (items.length <= 1) {
    return null
  }

  return (
    <Box sx={{ mb: 2 }}>
      <MuiBreadcrumbs
        separator={<ChevronRightIcon sx={{ fontSize: 16, color: 'text.disabled' }} />}
        maxItems={maxItems}
        sx={{
          '& .MuiBreadcrumbs-ol': {
            alignItems: 'center',
          },
        }}
      >
        {items.map((item, index) => {
          if (item.isLast) {
            return (
              <Typography
                key={item.path}
                variant="body2"
                sx={{
                  color: 'text.primary',
                  fontWeight: 500,
                  fontSize: '0.8125rem',
                }}
              >
                {item.label}
              </Typography>
            )
          }

          return (
            <Link
              key={item.path}
              component={RouterLink}
              to={item.path}
              underline="hover"
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                color: 'text.secondary',
                fontSize: '0.8125rem',
                transition: 'color 0.2s ease',
                '&:hover': {
                  color: 'primary.main',
                },
              }}
            >
              {index === 0 && showHome && (
                <HomeIcon sx={{ fontSize: 16 }} />
              )}
              {item.label}
            </Link>
          )
        })}
      </MuiBreadcrumbs>
    </Box>
  )
}

export default Breadcrumbs
