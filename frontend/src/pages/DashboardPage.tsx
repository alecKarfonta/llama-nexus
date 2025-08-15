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
  Divider,
  Paper
} from '@mui/material'
import {
  Settings as SettingsIcon
} from '@mui/icons-material'
import { RealTimeMetricsDisplay } from '@/components/monitoring'
import { ServiceStatusDisplay } from '@/components/monitoring/ServiceStatusDisplay'
import { LogViewer } from '@/components/monitoring/LogViewer'
import { TokenUsageTracker } from '@/components/monitoring/TokenUsageTracker'
import { StatCard } from '@/components/dashboard/StatCard'

export const DashboardPage: React.FC = () => {
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
      px: 3,
      py: 2,
      boxSizing: 'border-box'
    }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box>
          <Typography 
            variant="h1" 
            sx={{ 
              fontWeight: 700, 
              color: 'text.primary',
              mb: 0.5,
              fontSize: { xs: '1.25rem', sm: '1.5rem' },
              lineHeight: 1
            }}
          >
            ML Model Manager Dashboard
          </Typography>
          <Typography 
            variant="body2" 
            color="text.secondary" 
            sx={{ 
              fontSize: '0.8125rem',
              mb: { xs: 1, sm: 2 }
            }}
          >
            Monitor your ML models and service performance in real-time
          </Typography>
        </Box>
        <Box>
          <Tooltip title="Dashboard Settings">
            <IconButton 
              onClick={() => setShowSettings(!showSettings)}
              size="small"
              sx={{
                bgcolor: showSettings ? 'action.selected' : 'transparent',
                '&:hover': { bgcolor: 'action.hover' }
              }}
            >
              <SettingsIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Refresh Data">
            <IconButton 
              size="small"
              sx={{
                '&:hover': { bgcolor: 'action.hover' }
              }}
            >
              ⟲
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Settings Panel */}
      <Collapse in={showSettings}>
        <Card sx={{ mb: 2, borderRadius: 0.5, boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.1)', border: '1px solid', borderColor: 'grey.200' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Dashboard Settings
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Configure dashboard refresh rate, displayed metrics, and other preferences.
            </Typography>
            <Divider sx={{ my: 2 }} />
            <Typography variant="body2" color="text.secondary">
              Settings panel content would go here.
            </Typography>
          </CardContent>
        </Card>
      </Collapse>
      
      {/* Stats Cards */}
      <Paper sx={{ 
        p: 2, 
        mb: 3, 
        borderRadius: 0.5,
        bgcolor: 'background.paper',
        border: '1px solid',
        borderColor: 'grey.200'
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          {/* Dashboard icon would go here */}
          <Typography variant="h6">Key Metrics</Typography>
        </Box>
        <Grid container spacing={1.5} sx={{ width: 'auto', mx: 0 }}>
          <Grid item xs={12} sm={6} lg={3}>
            <StatCard 
              variant="models" 
              value={stats.totalModels} 
              label="Total Models" 
            />
          </Grid>
          <Grid item xs={12} sm={6} lg={3}>
            <StatCard 
              variant="active" 
              value={stats.activeServices} 
              label="Active Services" 
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
            />
          </Grid>
        </Grid>
      </Paper>
      
      {/* Main Content */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', maxWidth: '100%' }}>
        <Grid container spacing={1.5} sx={{ width: 'auto', mx: 0 }}>
          <Grid item xs={12}>
            <Paper sx={{ 
              p: 2, 
              borderRadius: 0.5,
              bgcolor: 'background.paper',
              border: '1px solid',
              borderColor: 'grey.200'
            }}>
              <Typography variant="h6" gutterBottom>Service Status</Typography>
              <Box sx={{ maxWidth: '100%', overflow: 'auto' }}>
                <ServiceStatusDisplay />
              </Box>
            </Paper>
          </Grid>
          
          <Grid item xs={12}>
            <Paper sx={{ 
              p: 2, 
              borderRadius: 0.5,
              bgcolor: 'background.paper',
              border: '1px solid',
              borderColor: 'grey.200'
            }}>
              <Typography variant="h6" gutterBottom>Real-Time Metrics</Typography>
              <Box sx={{ maxWidth: '100%', overflow: 'auto' }}>
                <RealTimeMetricsDisplay showWebSocketStatus />
              </Box>
            </Paper>
          </Grid>
          
          <Grid item xs={12}>
            <Paper sx={{ 
              p: 2, 
              borderRadius: 0.5,
              bgcolor: 'background.paper',
              border: '1px solid',
              borderColor: 'grey.200'
            }}>
              <Typography variant="h6" gutterBottom>Token Usage Tracking</Typography>
              <Box sx={{ maxWidth: '100%', overflow: 'auto' }}>
                <TokenUsageTracker />
              </Box>
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Paper sx={{ 
              p: 2, 
              borderRadius: 0.5,
              bgcolor: 'background.paper',
              border: '1px solid',
              borderColor: 'grey.200'
            }}>
              <Typography variant="h6" gutterBottom>System Logs</Typography>
              <Box sx={{ maxWidth: '100%', overflow: 'auto' }}>
                <LogViewer />
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </Box>
    </Box>
  )
}
