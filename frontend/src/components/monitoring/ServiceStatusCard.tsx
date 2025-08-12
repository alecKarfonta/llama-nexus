/**
 * Service Status Card Component - Shows service health and status
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Divider,
  Grid,
  useTheme,
  LinearProgress,
} from '@mui/material';
import {
  CheckCircle as HealthyIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  HourglassEmpty as StartingIcon,
  Stop as StoppedIcon,
  AccessTime as UptimeIcon,
  Speed as PerformanceIcon,
} from '@mui/icons-material';
import type { ServiceStatus } from '@/types/api';

interface ServiceStatusCardProps {
  status?: ServiceStatus;
  loading?: boolean;
  error?: string;
}

export const ServiceStatusCard: React.FC<ServiceStatusCardProps> = ({
  status,
  loading = false,
  error,
}) => {
  const theme = useTheme();

  const getStatusInfo = (health: ServiceStatus['health']) => {
    switch (health) {
      case 'healthy':
        return {
          icon: <HealthyIcon />,
          color: theme.palette.success.main,
          chipColor: 'success' as const,
          label: 'Healthy',
        };
      case 'degraded':
        return {
          icon: <WarningIcon />,
          color: theme.palette.warning.main,
          chipColor: 'warning' as const,
          label: 'Degraded',
        };
      case 'unhealthy':
        return {
          icon: <ErrorIcon />,
          color: theme.palette.error.main,
          chipColor: 'error' as const,
          label: 'Unhealthy',
        };
      case 'starting':
        return {
          icon: <StartingIcon />,
          color: theme.palette.info.main,
          chipColor: 'info' as const,
          label: 'Starting',
        };
      case 'stopped':
        return {
          icon: <StoppedIcon />,
          color: theme.palette.grey[500],
          chipColor: 'default' as const,
          label: 'Stopped',
        };
      default:
        return {
          icon: <ErrorIcon />,
          color: theme.palette.grey[500],
          chipColor: 'default' as const,
          label: 'Unknown',
        };
    }
  };

  const formatUptime = (seconds: number): string => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) {
      return `${days}d ${hours}h ${minutes}m`;
    }
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const getEndpointStatus = (endpoints: ServiceStatus['endpoints']) => {
    const total = Object.keys(endpoints).length;
    const active = Object.values(endpoints).filter(Boolean).length;
    return { active, total };
  };

  if (loading) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Service Status
          </Typography>
          <LinearProgress />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Checking service status...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Service Status
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <ErrorIcon color="error" />
            <Chip label="Connection Error" color="error" size="small" />
          </Box>
          <Typography variant="body2" color="error">
            {error}
          </Typography>
        </CardContent>
      </Card>
    );
  }

  if (!status) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Service Status
          </Typography>
          <Typography variant="body2" color="text.secondary">
            No status data available
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const statusInfo = getStatusInfo(status.health);
  const endpointStatus = getEndpointStatus(status.endpoints);

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Service Status
        </Typography>
        
        {/* Main Status */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <Box sx={{ color: statusInfo.color }}>
            {statusInfo.icon}
          </Box>
          <Box sx={{ flexGrow: 1 }}>
            <Chip 
              label={statusInfo.label} 
              color={statusInfo.chipColor} 
              size="small" 
              variant="filled"
            />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
              Version {status.version}
            </Typography>
          </Box>
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Status Details */}
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <UptimeIcon fontSize="small" color="action" />
              <Typography variant="body2" color="text.secondary">
                Uptime
              </Typography>
            </Box>
            <Typography variant="body1" fontWeight="medium">
              {formatUptime(status.uptime)}
            </Typography>
          </Grid>
          
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <PerformanceIcon fontSize="small" color="action" />
              <Typography variant="body2" color="text.secondary">
                Endpoints
              </Typography>
            </Box>
            <Typography variant="body1" fontWeight="medium">
              {endpointStatus.active}/{endpointStatus.total} active
            </Typography>
          </Grid>
        </Grid>

        {/* Model Status */}
        {status.modelLoaded && status.modelName && (
          <>
            <Divider sx={{ my: 2 }} />
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Current Model
              </Typography>
              <Typography variant="body1" fontWeight="medium">
                {status.modelName.split('/').pop()?.replace('.gguf', '') || 'Unknown'}
              </Typography>
              <Chip 
                label="Loaded" 
                color="success" 
                size="small" 
                variant="outlined"
                sx={{ mt: 1 }}
              />
            </Box>
          </>
        )}

        {/* Error Display */}
        {status.lastError && (
          <>
            <Divider sx={{ my: 2 }} />
            <Box>
              <Typography variant="body2" color="error" gutterBottom>
                Last Error
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {status.lastError}
              </Typography>
            </Box>
          </>
        )}

        {/* Last Updated */}
        <Box sx={{ mt: 2, pt: 1, borderTop: `1px solid ${theme.palette.divider}` }}>
          <Typography variant="caption" color="text.secondary">
            Last updated: {new Date(status.timestamp).toLocaleTimeString()}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};
