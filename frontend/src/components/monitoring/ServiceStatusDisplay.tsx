import React, { useEffect, useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Button,
  CircularProgress,
  Alert,
  Grid,
  LinearProgress,
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Refresh as RestartIcon,
  Thermostat as TempIcon,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface ServiceStatus {
  running: boolean;
  pid?: number;
  uptime?: number;
  start_time?: string;
  mode?: string;
  config?: any;
  model?: {
    name: string;
    variant: string;
    context_size: number;
    gpu_layers: number;
  };
  resources?: {
    cpu_percent: number;
    memory_mb: number;
    memory_percent: number;
    num_threads: number;
  };
  gpu?: {
    vram_used_mb: number;
    vram_total_mb: number;
    gpu_usage_percent: number;
    temperature_c: number;
  };
  llamacpp_health?: {
    healthy: boolean;
    status_code?: number;
    error?: string;
  };
}

interface ServiceStatusDisplayProps {
  // No props needed since we use the API service
}

export const ServiceStatusDisplay: React.FC<ServiceStatusDisplayProps> = () => {
  const theme = useTheme();
  const [status, setStatus] = useState<ServiceStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      const backendStatus = await fetch('/api/v1/service/status');
      if (backendStatus.ok) {
        const data = await backendStatus.json();
        setStatus(data);
        setError(null);
        return;
      }
      // Fallback to basic health endpoint if detailed status fails
      const healthRes = await fetch('/api/health');
      if (healthRes.ok) {
        const health = await healthRes.json();
        setStatus({
          running: health.status === 'healthy',
          llamacpp_health: { healthy: health.status === 'healthy' }
        } as ServiceStatus);
        setError(null);
        return;
      }
      throw new Error(`Failed to fetch status: ${backendStatus.status}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch status');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000); // Poll every 2 seconds
    return () => clearInterval(interval);
  }, []);

  const handleServiceAction = async (action: 'start' | 'stop' | 'restart') => {
    setActionLoading(action);
    try {
      const response = await fetch(`/api/v1/service/${action}`, {
        method: 'POST',
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to ${action} service`);
      }
      await fetchStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to ${action} service`);
    } finally {
      setActionLoading(null);
    }
  };

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours}h ${minutes}m ${secs}s`;
  };

  const getStatusColor = (running: boolean, health?: boolean) => {
    if (!running) return theme.palette.grey[500];
    if (health === false) return theme.palette.warning.main;
    return theme.palette.success.main;
  };

  const getStatusText = (running: boolean, health?: boolean) => {
    if (!running) return 'Stopped';
    if (health === false) return 'Unhealthy';
    return 'Running';
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
            <CircularProgress />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error && !status) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Box display="flex" alignItems="center" gap={2}>
            <Typography variant="h5">LlamaCPP Service Status</Typography>
            <Chip
              label={getStatusText(status?.running || false, status?.llamacpp_health?.healthy)}
              size="small"
              sx={{
                backgroundColor: getStatusColor(status?.running || false, status?.llamacpp_health?.healthy),
                color: 'white',
              }}
            />
          </Box>
          
          <Box display="flex" gap={1}>
            <Button
              variant="contained"
              color="success"
              startIcon={actionLoading === 'start' ? <CircularProgress size={20} /> : <StartIcon />}
              onClick={() => handleServiceAction('start')}
              disabled={status?.running || actionLoading !== null}
              size="small"
            >
              Start
            </Button>
            <Button
              variant="contained"
              color="error"
              startIcon={actionLoading === 'stop' ? <CircularProgress size={20} /> : <StopIcon />}
              onClick={() => handleServiceAction('stop')}
              disabled={!status?.running || actionLoading !== null}
              size="small"
            >
              Stop
            </Button>
            <Button
              variant="contained"
              color="warning"
              startIcon={actionLoading === 'restart' ? <CircularProgress size={20} /> : <RestartIcon />}
              onClick={() => handleServiceAction('restart')}
              disabled={actionLoading !== null}
              size="small"
            >
              Restart
            </Button>
          </Box>
        </Box>

        {error && (
          <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Grid container spacing={3}>
          {/* Service Info */}
          <Grid item xs={12} md={6}>
            <Box mb={2}>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Service Information
              </Typography>
              <Box display="flex" flexDirection="column" gap={1}>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2">Status:</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {status?.running ? 'Running' : 'Stopped'}
                  </Typography>
                </Box>
                {status?.running && (
                  <>
                    <Box display="flex" justifyContent="space-between">
                      <Typography variant="body2">PID:</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {status.pid || 'N/A'}
                      </Typography>
                    </Box>
                    <Box display="flex" justifyContent="space-between">
                      <Typography variant="body2">Uptime:</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {status.uptime ? formatUptime(status.uptime) : 'N/A'}
                      </Typography>
                    </Box>
                    <Box display="flex" justifyContent="space-between">
                      <Typography variant="body2">Mode:</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {status.mode || 'subprocess'}
                      </Typography>
                    </Box>
                  </>
                )}
              </Box>
            </Box>
          </Grid>

          {/* Model Info */}
          <Grid item xs={12} md={6}>
            <Box mb={2}>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Model Configuration
              </Typography>
              <Box display="flex" flexDirection="column" gap={1}>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2">Model:</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {status?.model?.name || 'None'}
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2">Variant:</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {status?.model?.variant || 'N/A'}
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2">Context Size:</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {status?.model?.context_size?.toLocaleString() || 'N/A'}
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2">GPU Layers:</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {status?.model?.gpu_layers || 'N/A'}
                  </Typography>
                </Box>
              </Box>
            </Box>
          </Grid>

          {/* Resource Usage */}
          {status?.running && (
            <>
              <Grid item xs={12} md={6}>
                <Box mb={2}>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    System Resources
                  </Typography>
                  <Box display="flex" flexDirection="column" gap={2}>
                    {status.resources && (
                      <>
                        <Box>
                          <Box display="flex" justifyContent="space-between" mb={0.5}>
                            <Typography variant="body2">CPU Usage</Typography>
                            <Typography variant="body2" fontWeight="bold">
                              {status.resources.cpu_percent?.toFixed(1)}%
                            </Typography>
                          </Box>
                          <LinearProgress
                            variant="determinate"
                            value={status.resources.cpu_percent || 0}
                            sx={{
                              height: 6,
                              borderRadius: 3,
                              backgroundColor: theme.palette.grey[300],
                              '& .MuiLinearProgress-bar': {
                                backgroundColor: status.resources.cpu_percent > 80 
                                  ? theme.palette.error.main 
                                  : theme.palette.primary.main,
                              },
                            }}
                          />
                        </Box>
                        <Box>
                          <Box display="flex" justifyContent="space-between" mb={0.5}>
                            <Typography variant="body2">Memory</Typography>
                            <Typography variant="body2" fontWeight="bold">
                              {(status.resources.memory_mb / 1024).toFixed(1)} GB ({status.resources.memory_percent?.toFixed(1)}%)
                            </Typography>
                          </Box>
                          <LinearProgress
                            variant="determinate"
                            value={status.resources.memory_percent || 0}
                            sx={{
                              height: 6,
                              borderRadius: 3,
                              backgroundColor: theme.palette.grey[300],
                              '& .MuiLinearProgress-bar': {
                                backgroundColor: status.resources.memory_percent > 80 
                                  ? theme.palette.error.main 
                                  : theme.palette.primary.main,
                              },
                            }}
                          />
                        </Box>
                      </>
                    )}
                  </Box>
                </Box>
              </Grid>

              <Grid item xs={12} md={6}>
                <Box mb={2}>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    GPU Resources
                  </Typography>
                  {status.gpu ? (
                    <Box display="flex" flexDirection="column" gap={2}>
                      <Box>
                        <Box display="flex" justifyContent="space-between" mb={0.5}>
                          <Typography variant="body2">GPU Usage</Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {status.gpu.gpu_usage_percent?.toFixed(1)}%
                          </Typography>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={status.gpu.gpu_usage_percent || 0}
                          sx={{
                            height: 6,
                            borderRadius: 3,
                            backgroundColor: theme.palette.grey[300],
                            '& .MuiLinearProgress-bar': {
                              backgroundColor: status.gpu.gpu_usage_percent > 80 
                                ? theme.palette.error.main 
                                : theme.palette.success.main,
                            },
                          }}
                        />
                      </Box>
                      <Box>
                        <Box display="flex" justifyContent="space-between" mb={0.5}>
                          <Typography variant="body2">VRAM</Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {(status.gpu.vram_used_mb / 1024).toFixed(1)} / {(status.gpu.vram_total_mb / 1024).toFixed(1)} GB
                          </Typography>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={(status.gpu.vram_used_mb / status.gpu.vram_total_mb) * 100}
                          sx={{
                            height: 6,
                            borderRadius: 3,
                            backgroundColor: theme.palette.grey[300],
                            '& .MuiLinearProgress-bar': {
                              backgroundColor: (status.gpu.vram_used_mb / status.gpu.vram_total_mb) > 0.8 
                                ? theme.palette.error.main 
                                : theme.palette.success.main,
                            },
                          }}
                        />
                      </Box>
                      <Box display="flex" alignItems="center" gap={1}>
                        <TempIcon 
                          fontSize="small" 
                          sx={{ 
                            color: status.gpu.temperature_c > 85 ? theme.palette.error.main : 
                                   status.gpu.temperature_c > 75 ? theme.palette.warning.main : 
                                   theme.palette.success.main 
                          }} 
                        />
                        <Typography variant="body2">
                          Temperature: {status.gpu.temperature_c?.toFixed(0)}°C
                        </Typography>
                        {status.gpu.temperature_c > 85 && (
                          <Chip label="Critical" size="small" color="error" variant="outlined" />
                        )}
                        {status.gpu.temperature_c > 75 && status.gpu.temperature_c <= 85 && (
                          <Chip label="Warning" size="small" color="warning" variant="outlined" />
                        )}
                      </Box>
                    </Box>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      GPU information not available
                    </Typography>
                  )}
                </Box>
              </Grid>
            </>
          )}
        </Grid>
      </CardContent>
    </Card>
  );
};
