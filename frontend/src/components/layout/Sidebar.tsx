import React from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Toolbar,
  Typography,
  Box,
  styled
} from '@mui/material'
import {
  Dashboard as DashboardIcon,
  Chat as ChatIcon,
  Memory as ModelsIcon,
  Settings as ConfigIcon,
  Storage as DeployIcon,
  Speed as TestingIcon,
  Insights as MonitoringIcon,
  Description as TemplatesIcon,
  LibraryBooks as PromptsIcon,
  Hub as RegistryIcon,
  Timer as BenchmarkIcon,
  BatchPrediction as BatchIcon,
} from '@mui/icons-material'
import type { NavigationItem, NavigationSection } from '@/types'

interface SidebarProps {
  open: boolean
  onToggle: () => void
}

// Styled components for the sidebar
const SidebarHeader = styled(Box)(({ theme }) => ({
  padding: theme.spacing(1.5),
  borderBottom: `1px solid ${theme.palette.grey[200]}`,
}))

const Logo = styled(Typography)(({ theme }) => ({
  fontSize: 14,
  fontWeight: 700,
  color: theme.palette.text.primary,
  letterSpacing: '-0.01em',
}))

const SectionTitle = styled(Typography)(({ theme }) => ({
  fontSize: 10,
  textTransform: 'uppercase',
  color: theme.palette.text.secondary,
  marginBottom: theme.spacing(0.5),
  marginTop: theme.spacing(1.5),
  marginLeft: theme.spacing(1.5),
  letterSpacing: 0.5,
  fontWeight: 600,
  opacity: 0.7,
}))

// Navigation sections with items
const navigationSections: NavigationSection[] = [
  {
    id: 'overview',
    title: 'Overview',
    items: [
      {
        id: 'dashboard',
        label: 'Dashboard',
        path: '/dashboard',
        icon: 'dashboard',
      },
      {
        id: 'models',
        label: 'Models',
        path: '/models',
        icon: 'models',
      },
      {
        id: 'registry',
        label: 'Registry',
        path: '/registry',
        icon: 'registry',
      },
    ],
  },
  {
    id: 'development',
    title: 'Development',
    items: [
      {
        id: 'chat',
        label: 'Chat',
        path: '/chat',
        icon: 'chat',
      },
      {
        id: 'templates',
        label: 'Templates',
        path: '/templates',
        icon: 'templates',
      },
      {
        id: 'prompts',
        label: 'Prompts',
        path: '/prompts',
        icon: 'prompts',
      },
      {
        id: 'deploy',
        label: 'Deploy',
        path: '/deploy',
        icon: 'deploy',
      },
      {
        id: 'testing',
        label: 'Testing',
        path: '/testing',
        icon: 'testing',
      },
      {
        id: 'benchmark',
        label: 'Benchmark',
        path: '/benchmark',
        icon: 'benchmark',
      },
      {
        id: 'batch',
        label: 'Batch',
        path: '/batch',
        icon: 'batch',
      },
    ],
  },
  {
    id: 'operations',
    title: 'Operations',
    items: [
      {
        id: 'monitoring',
        label: 'Monitoring',
        path: '/monitoring',
        icon: 'monitoring',
      },
      {
        id: 'configuration',
        label: 'Settings',
        path: '/configuration',
        icon: 'config',
      },
    ],
  },
]

const getIcon = (iconName: string) => {
  switch (iconName) {
    case 'dashboard':
      return <DashboardIcon />
    case 'chat':
      return <ChatIcon />
    case 'models':
      return <ModelsIcon />
    case 'config':
      return <ConfigIcon />
    case 'deploy':
      return <DeployIcon />
    case 'testing':
      return <TestingIcon />
    case 'monitoring':
      return <MonitoringIcon />
    case 'templates':
      return <TemplatesIcon />
    case 'prompts':
      return <PromptsIcon />
    case 'registry':
      return <RegistryIcon />
    case 'benchmark':
      return <BenchmarkIcon />
    case 'batch':
      return <BatchIcon />
    default:
      return <DashboardIcon />
  }
}

export const Sidebar: React.FC<SidebarProps> = ({ open }) => {
  const location = useLocation()
  const navigate = useNavigate()

  const handleNavigation = (path: string) => {
    navigate(path)
  }

  return (
    <Drawer
      variant="persistent"
      anchor="left"
      open={open}
      sx={{
        width: open ? 200 : 0,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 200,
          boxSizing: 'border-box',
          borderRight: '1px solid',
          borderColor: 'grey.200',
        },
      }}
    >
      <Toolbar sx={{ minHeight: '48px !important' }} />
      <SidebarHeader>
        <Logo>ML Manager</Logo>
      </SidebarHeader>
      
      {navigationSections.map((section) => (
        <React.Fragment key={section.id}>
          <SectionTitle>{section.title}</SectionTitle>
          <List sx={{ py: 0 }}>
            {section.items.map((item) => (
              <ListItem key={item.id} disablePadding>
                <ListItemButton
                  selected={location.pathname === item.path}
                  onClick={() => handleNavigation(item.path)}
                  sx={{
                    py: 0.75,
                    px: 1.5,
                    borderLeft: '2px solid transparent',
                    minHeight: 36,
                    '&.Mui-selected': {
                      backgroundColor: 'primary.50',
                      borderLeftColor: 'primary.main',
                      '& .MuiListItemIcon-root': {
                        color: 'primary.main',
                      },
                      '& .MuiListItemText-primary': {
                        color: 'primary.main',
                        fontWeight: 600,
                      },
                    },
                    '&:hover': {
                      backgroundColor: 'grey.50',
                    },
                  }}
                >
                  <ListItemIcon sx={{ 
                    minWidth: 28, 
                    color: 'inherit', 
                    opacity: 0.7,
                    '& .MuiSvgIcon-root': {
                      fontSize: '1.125rem'
                    }
                  }}>
                    {getIcon(item.icon || 'dashboard')}
                  </ListItemIcon>
                  <ListItemText 
                    primary={item.label} 
                    primaryTypographyProps={{ 
                      fontSize: '0.8125rem',
                      fontWeight: location.pathname === item.path ? 600 : 500,
                      lineHeight: 1.2
                    }} 
                  />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </React.Fragment>
      ))}
    </Drawer>
  )
}
