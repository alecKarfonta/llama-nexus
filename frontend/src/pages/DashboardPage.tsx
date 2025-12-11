import React, { useState } from 'react'
import {
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  IconButton,
  Tooltip,
  Collapse,
  alpha,
  Chip,
} from '@mui/material'
import {
  Settings as SettingsIcon,
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  Schedule as ScheduleIcon,
  Chat as ChatIcon,
  RocketLaunch as DeployIcon,
  Folder as DocumentsIcon,
  Insights as MonitoringIcon,
  AccountTree as WorkflowIcon,
  Hub as GraphIcon,
} from '@mui/icons-material'
import { useNavigate } from 'react-router-dom'
import { RealTimeMetricsDisplay } from '@/components/monitoring'
import { ServiceStatusDisplay } from '@/components/monitoring/ServiceStatusDisplay'
import { LogViewer } from '@/components/monitoring/LogViewer'
import { TokenUsageTracker } from '@/components/monitoring/TokenUsageTracker'
import { StatCard } from '@/components/dashboard/StatCard'

// Section wrapper component for consistent styling
interface SectionCardProps {
  title: string
  subtitle?: string
  icon?: React.ReactNode
  accentColor?: string
  children: React.ReactNode
  action?: React.ReactNode
}

const SectionCard: React.FC<SectionCardProps> = ({ 
  title, 
  subtitle, 
  icon, 
  accentColor = '#6366f1', 
  children,
  action 
}) => (
  <Card
    sx={{
      position: 'relative',
      overflow: 'hidden',
      background: 'linear-gradient(145deg, rgba(30, 30, 63, 0.6) 0%, rgba(26, 26, 46, 0.8) 100%)',
      backdropFilter: 'blur(12px)',
      border: '1px solid rgba(255, 255, 255, 0.06)',
      borderRadius: 3,
      transition: 'all 0.3s ease-in-out',
      '&:hover': {
        borderColor: alpha(accentColor, 0.2),
        boxShadow: `0 8px 32px ${alpha(accentColor, 0.15)}`,
      },
    }}
  >
    {/* Top accent gradient */}
    <Box
      sx={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: 3,
        background: `linear-gradient(90deg, ${accentColor} 0%, ${alpha(accentColor, 0.3)} 100%)`,
      }}
    />
    
    <CardContent sx={{ p: 2.5 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          {icon && (
            <Box
              sx={{
                width: 36,
                height: 36,
                borderRadius: 2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                bgcolor: alpha(accentColor, 0.1),
                color: accentColor,
                '& .MuiSvgIcon-root': {
                  fontSize: 20,
                },
              }}
            >
              {icon}
            </Box>
          )}
          <Box>
            <Typography
              variant="h6"
              sx={{
                fontWeight: 600,
                fontSize: '1rem',
                color: 'text.primary',
                lineHeight: 1.3,
              }}
            >
              {title}
            </Typography>
            {subtitle && (
              <Typography
                variant="caption"
                sx={{
                  color: 'text.secondary',
                  fontSize: '0.75rem',
                }}
              >
                {subtitle}
              </Typography>
            )}
          </Box>
        </Box>
        {action}
      </Box>
      
      {/* Content */}
      <Box sx={{ maxWidth: '100%', overflow: 'auto' }}>
        {children}
      </Box>
    </CardContent>
  </Card>
)

// Quick action item type
interface QuickAction {
  id: string
  label: string
  description: string
  icon: React.ReactNode
  path: string
  color: string
  shortcut?: string
}

const quickActions: QuickAction[] = [
  { id: 'chat', label: 'Chat', description: 'Start conversation', icon: <ChatIcon />, path: '/chat', color: '#06b6d4', shortcut: 'C' },
  { id: 'deploy', label: 'Deploy', description: 'Launch model', icon: <DeployIcon />, path: '/deploy', color: '#f59e0b', shortcut: 'D' },
  { id: 'documents', label: 'Documents', description: 'Upload files', icon: <DocumentsIcon />, path: '/documents', color: '#10b981', shortcut: 'U' },
  { id: 'workflows', label: 'Workflows', description: 'Build pipeline', icon: <WorkflowIcon />, path: '/workflows', color: '#a855f7', shortcut: 'W' },
  { id: 'graph', label: 'Knowledge', description: 'View graph', icon: <GraphIcon />, path: '/knowledge-graph', color: '#6366f1', shortcut: 'G' },
  { id: 'monitoring', label: 'Monitor', description: 'System health', icon: <MonitoringIcon />, path: '/monitoring', color: '#0ea5e9', shortcut: 'M' },
]

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate()
  
  // Stats for dashboard (would come from API in production)
  const stats = {
    totalModels: 12,
    activeServices: 8,
    frameworks: 4,
    performance: '95%',
  }
  
  const [showSettings, setShowSettings] = useState(false)
  
  return (
    <Box sx={{ 
      width: '100%', 
      maxWidth: '100vw',
      overflow: 'hidden',
      px: { xs: 2, sm: 3, md: 4 },
      py: 3,
      boxSizing: 'border-box',
      minHeight: '100%',
    }}>
      {/* Header */}
      <Box sx={{ 
        display: 'flex', 
        flexDirection: { xs: 'column', sm: 'row' },
        justifyContent: 'space-between', 
        alignItems: { xs: 'flex-start', sm: 'center' }, 
        mb: 4,
        gap: 2,
      }}>
        <Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
            <Typography 
              variant="h1" 
              sx={{ 
                fontWeight: 700, 
                color: 'text.primary',
                fontSize: { xs: '1.5rem', sm: '1.75rem', md: '2rem' },
                lineHeight: 1,
                background: 'linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              Dashboard
            </Typography>
            <Chip
              size="small"
              label="Live"
              sx={{
                height: 22,
                bgcolor: alpha('#10b981', 0.1),
                border: `1px solid ${alpha('#10b981', 0.2)}`,
                color: '#34d399',
                fontWeight: 600,
                fontSize: '0.6875rem',
                '& .MuiChip-label': {
                  px: 1,
                },
              }}
            />
          </Box>
          <Typography 
            variant="body2" 
            color="text.secondary" 
            sx={{ 
              fontSize: '0.875rem',
              maxWidth: 400,
            }}
          >
            Monitor your ML models and infrastructure performance in real-time
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Tooltip title="Refresh Data">
            <IconButton 
              size="small"
              sx={{
                width: 38,
                height: 38,
                borderRadius: 2,
                border: '1px solid rgba(255, 255, 255, 0.08)',
                bgcolor: 'rgba(255, 255, 255, 0.03)',
                transition: 'all 0.2s ease-in-out',
                '&:hover': {
                  bgcolor: 'rgba(255, 255, 255, 0.08)',
                  borderColor: 'rgba(255, 255, 255, 0.15)',
                },
              }}
            >
              <RefreshIcon sx={{ fontSize: 18 }} />
            </IconButton>
          </Tooltip>
          <Tooltip title="Dashboard Settings">
            <IconButton 
              onClick={() => setShowSettings(!showSettings)}
              size="small"
              sx={{
                width: 38,
                height: 38,
                borderRadius: 2,
                border: '1px solid',
                borderColor: showSettings ? alpha('#6366f1', 0.3) : 'rgba(255, 255, 255, 0.08)',
                bgcolor: showSettings ? alpha('#6366f1', 0.1) : 'rgba(255, 255, 255, 0.03)',
                transition: 'all 0.2s ease-in-out',
                '&:hover': {
                  bgcolor: showSettings ? alpha('#6366f1', 0.15) : 'rgba(255, 255, 255, 0.08)',
                  borderColor: showSettings ? alpha('#6366f1', 0.4) : 'rgba(255, 255, 255, 0.15)',
                },
              }}
            >
              <SettingsIcon sx={{ fontSize: 18 }} />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Settings Panel */}
      <Collapse in={showSettings}>
        <Card 
          sx={{ 
            mb: 3, 
            background: 'linear-gradient(145deg, rgba(30, 30, 63, 0.6) 0%, rgba(26, 26, 46, 0.8) 100%)',
            backdropFilter: 'blur(12px)',
            border: '1px solid rgba(255, 255, 255, 0.06)',
            borderRadius: 3,
          }}
        >
          <CardContent sx={{ p: 2.5 }}>
            <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
              Dashboard Settings
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Configure dashboard refresh rate, displayed metrics, and other preferences.
            </Typography>
          </CardContent>
        </Card>
      </Collapse>

      {/* Quick Actions */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
          <Typography 
            variant="overline" 
            sx={{ 
              color: 'text.secondary',
              fontSize: '0.6875rem',
              fontWeight: 600,
              letterSpacing: '0.08em',
            }}
          >
            Quick Actions
          </Typography>
        </Box>
        <Box sx={{ 
          display: 'grid', 
          gridTemplateColumns: { 
            xs: 'repeat(2, 1fr)', 
            sm: 'repeat(3, 1fr)', 
            md: 'repeat(6, 1fr)' 
          }, 
          gap: 1.5 
        }}>
          {quickActions.map((action) => (
            <Box
              key={action.id}
              onClick={() => navigate(action.path)}
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 1,
                p: 2,
                borderRadius: 2,
                bgcolor: 'rgba(255, 255, 255, 0.02)',
                border: '1px solid rgba(255, 255, 255, 0.06)',
                cursor: 'pointer',
                transition: 'all 0.2s ease-in-out',
                '&:hover': {
                  bgcolor: alpha(action.color, 0.08),
                  borderColor: alpha(action.color, 0.3),
                  transform: 'translateY(-2px)',
                  '& .action-icon': {
                    bgcolor: alpha(action.color, 0.2),
                    color: action.color,
                  },
                  '& .action-label': {
                    color: 'text.primary',
                  },
                },
              }}
            >
              <Box
                className="action-icon"
                sx={{
                  width: 40,
                  height: 40,
                  borderRadius: 2,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  bgcolor: 'rgba(255, 255, 255, 0.05)',
                  color: 'text.secondary',
                  transition: 'all 0.2s ease-in-out',
                  '& .MuiSvgIcon-root': { fontSize: 20 },
                }}
              >
                {action.icon}
              </Box>
              <Box sx={{ textAlign: 'center' }}>
                <Typography
                  className="action-label"
                  variant="body2"
                  sx={{
                    fontWeight: 600,
                    fontSize: '0.8125rem',
                    color: 'text.secondary',
                    transition: 'color 0.2s ease-in-out',
                    lineHeight: 1.2,
                  }}
                >
                  {action.label}
                </Typography>
                <Typography
                  variant="caption"
                  sx={{
                    color: alpha('#fff', 0.4),
                    fontSize: '0.6875rem',
                    display: { xs: 'none', sm: 'block' },
                  }}
                >
                  {action.description}
                </Typography>
              </Box>
            </Box>
          ))}
        </Box>
      </Box>
      
      {/* Stats Cards */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
          <TrendingUpIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
          <Typography 
            variant="overline" 
            sx={{ 
              color: 'text.secondary',
              fontSize: '0.6875rem',
              fontWeight: 600,
              letterSpacing: '0.08em',
            }}
          >
            Key Metrics
          </Typography>
        </Box>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} lg={3}>
            <StatCard 
              variant="models" 
              value={stats.totalModels} 
              label="Total Models"
              trend={{ value: 12, positive: true }}
            />
          </Grid>
          <Grid item xs={12} sm={6} lg={3}>
            <StatCard 
              variant="active" 
              value={stats.activeServices} 
              label="Active Services"
              trend={{ value: 5, positive: true }}
            />
          </Grid>
          <Grid item xs={12} sm={6} lg={3}>
            <StatCard 
              variant="frameworks" 
              value={stats.frameworks} 
              label="Frameworks" 
            />
          </Grid>
          <Grid item xs={12} sm={6} lg={3}>
            <StatCard 
              variant="performance" 
              value={stats.performance} 
              label="Avg Performance"
              trend={{ value: 3, positive: true }}
            />
          </Grid>
        </Grid>
      </Box>
      
      {/* Main Content */}
      <Grid container spacing={3}>
        {/* Service Status */}
        <Grid item xs={12} lg={6}>
          <SectionCard
            title="Service Status"
            subtitle="Infrastructure health overview"
            accentColor="#10b981"
            icon={<ScheduleIcon />}
          >
            <ServiceStatusDisplay />
          </SectionCard>
        </Grid>
        
        {/* Real-Time Metrics */}
        <Grid item xs={12} lg={6}>
          <SectionCard
            title="Real-Time Metrics"
            subtitle="System resource utilization"
            accentColor="#06b6d4"
            icon={<TrendingUpIcon />}
          >
            <RealTimeMetricsDisplay showWebSocketStatus />
          </SectionCard>
        </Grid>
        
        {/* Token Usage */}
        <Grid item xs={12}>
          <SectionCard
            title="Token Usage Tracking"
            subtitle="Monitor API consumption"
            accentColor="#8b5cf6"
          >
            <TokenUsageTracker />
          </SectionCard>
        </Grid>

        {/* System Logs */}
        <Grid item xs={12}>
          <SectionCard
            title="System Logs"
            subtitle="Real-time log stream"
            accentColor="#f59e0b"
          >
            <LogViewer />
          </SectionCard>
        </Grid>
      </Grid>
    </Box>
  )
}
