import React, { useEffect, useState } from 'react';
import {
  Typography,
  Box,
  Chip,
  Button,
  CircularProgress,
  Alert,
  Grid,
  LinearProgress,
  alpha,
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Refresh as RestartIcon,
  Thermostat as TempIcon,
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
} from '@mui/icons-material';

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

export const ServiceStatusDisplay: React.FC = () => {
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
    const interval = setInterval(fetchStatus, 2000);
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
    if (!running) return '#64748b';
    if (health === false) return '#f59e0b';
    return '#10b981';
  };

  const getStatusText = (running: boolean, health?: boolean) => {
    if (!running) return 'Stopped';
    if (health === false) return 'Unhealthy';
    return 'Running';
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
        <CircularProgress size={32} sx={{ color: '#6366f1' }} />
      </Box>
    );
  }

  if (error && !status) {
    return (
      <Alert 
        severity="error" 
        onClose={() => setError(null)}
        sx={{
          bgcolor: alpha('#ef4444', 0.1),
          border: `1px solid ${alpha('#ef4444', 0.2)}`,
          color: '#f87171',
        }}
      >
        {error}
      </Alert>
    );
  }

  const statusColor = getStatusColor(status?.running || false, status?.llamacpp_health?.healthy);

  return (
    <Box>
      {/* Header with Status and Actions */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3} flexWrap="wrap" gap={2}>
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>LlamaCPP Service</Typography>
          <Chip
            label={getStatusText(status?.running || false, status?.llamacpp_health?.healthy)}
            size="small"
            sx={{
              bgcolor: alpha(statusColor, 0.1),
              border: `1px solid ${alpha(statusColor, 0.3)}`,
              color: statusColor,
              fontWeight: 600,
              fontSize: '0.75rem',
              '& .MuiChip-label': { px: 1.5 },
            }}
          />
        </Box>
        
        <Box display="flex" gap={1}>
          <Button
            variant="contained"
            startIcon={actionLoading === 'start' ? <CircularProgress size={16} sx={{ color: 'white' }} /> : <StartIcon />}
            onClick={() => handleServiceAction('start')}
            disabled={status?.running || actionLoading !== null}
            size="small"
            sx={{
              bgcolor: '#10b981',
              '&:hover': { bgcolor: '#059669' },
              '&.Mui-disabled': { bgcolor: alpha('#10b981', 0.3) },
            }}
          >
            Start
          </Button>
          <Button
            variant="contained"
            startIcon={actionLoading === 'stop' ? <CircularProgress size={16} sx={{ color: 'white' }} /> : <StopIcon />}
            onClick={() => handleServiceAction('stop')}
            disabled={!status?.running || actionLoading !== null}
            size="small"
            sx={{
              bgcolor: '#ef4444',
              '&:hover': { bgcolor: '#dc2626' },
              '&.Mui-disabled': { bgcolor: alpha('#ef4444', 0.3) },
            }}
          >
            Stop
          </Button>
          <Button
            variant="contained"
            startIcon={actionLoading === 'restart' ? <CircularProgress size={16} sx={{ color: 'white' }} /> : <RestartIcon />}
            onClick={() => handleServiceAction('restart')}
            disabled={actionLoading !== null}
            size="small"
            sx={{
              bgcolor: '#f59e0b',
              '&:hover': { bgcolor: '#d97706' },
              '&.Mui-disabled': { bgcolor: alpha('#f59e0b', 0.3) },
            }}
          >
            Restart
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert 
          severity="error" 
          onClose={() => setError(null)} 
          sx={{ 
            mb: 2,
            bgcolor: alpha('#ef4444', 0.1),
            border: `1px solid ${alpha('#ef4444', 0.2)}`,
            color: '#f87171',
          }}
        >
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Service Info */}
        <Grid item xs={12} md={6}>
          <Box 
            sx={{ 
              p: 2, 
              borderRadius: 2, 
              bgcolor: 'rgba(255, 255, 255, 0.02)',
              border: '1px solid rgba(255, 255, 255, 0.04)',
            }}
          >
            <Box display="flex" alignItems="center" gap={1} mb={1.5}>
              <StorageIcon sx={{ fontSize: 18, color: '#6366f1' }} />
              <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                Service Information
              </Typography>
            </Box>
            <Box display="flex" flexDirection="column" gap={1}>
              <InfoRow label="Status" value={status?.running ? 'Running' : 'Stopped'} color={statusColor} />
              {status?.running && (
                <>
                  <InfoRow label="PID" value={status.pid?.toString() || 'N/A'} />
                  <InfoRow label="Uptime" value={status.uptime ? formatUptime(status.uptime) : 'N/A'} />
                  <InfoRow label="Mode" value={status.mode || 'subprocess'} />
                </>
              )}
            </Box>
          </Box>
        </Grid>

        {/* Model Info */}
        <Grid item xs={12} md={6}>
          <Box 
            sx={{ 
              p: 2, 
              borderRadius: 2, 
              bgcolor: 'rgba(255, 255, 255, 0.02)',
              border: '1px solid rgba(255, 255, 255, 0.04)',
            }}
          >
            <Box display="flex" alignItems="center" gap={1} mb={1.5}>
              <MemoryIcon sx={{ fontSize: 18, color: '#8b5cf6' }} />
              <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                Model Configuration
              </Typography>
            </Box>
            <Box display="flex" flexDirection="column" gap={1}>
              <InfoRow label="Model" value={status?.model?.name || 'None'} />
              <InfoRow label="Variant" value={status?.model?.variant || 'N/A'} />
              <InfoRow label="Context Size" value={status?.model?.context_size?.toLocaleString() || 'N/A'} />
              <InfoRow label="GPU Layers" value={status?.model?.gpu_layers?.toString() || 'N/A'} />
            </Box>
          </Box>
        </Grid>

        {/* Resource Usage */}
        {status?.running && (
          <>
            <Grid item xs={12} md={6}>
              <Box 
                sx={{ 
                  p: 2, 
                  borderRadius: 2, 
                  bgcolor: 'rgba(255, 255, 255, 0.02)',
                  border: '1px solid rgba(255, 255, 255, 0.04)',
                }}
              >
                <Box display="flex" alignItems="center" gap={1} mb={1.5}>
                  <SpeedIcon sx={{ fontSize: 18, color: '#06b6d4' }} />
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                    System Resources
                  </Typography>
                </Box>
                {status.resources && (
                  <Box display="flex" flexDirection="column" gap={2}>
                    <ResourceBar 
                      label="CPU Usage" 
                      value={status.resources.cpu_percent} 
                      max={100}
                      unit="%"
                      color="#6366f1"
                    />
                    <ResourceBar 
                      label="Memory" 
                      value={status.resources.memory_mb / 1024} 
                      max={100}
                      displayValue={`${(status.resources.memory_mb / 1024).toFixed(1)} GB (${status.resources.memory_percent?.toFixed(1)}%)`}
                      color="#8b5cf6"
                      progress={status.resources.memory_percent}
                    />
                  </Box>
                )}
              </Box>
            </Grid>

            <Grid item xs={12} md={6}>
              <Box 
                sx={{ 
                  p: 2, 
                  borderRadius: 2, 
                  bgcolor: 'rgba(255, 255, 255, 0.02)',
                  border: '1px solid rgba(255, 255, 255, 0.04)',
                }}
              >
                <Box display="flex" alignItems="center" gap={1} mb={1.5}>
                  <MemoryIcon sx={{ fontSize: 18, color: '#10b981' }} />
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                    GPU Resources
                  </Typography>
                </Box>
                {status.gpu ? (
                  <Box display="flex" flexDirection="column" gap={2}>
                    <ResourceBar 
                      label="GPU Usage" 
                      value={status.gpu.gpu_usage_percent} 
                      max={100}
                      unit="%"
                      color="#10b981"
                    />
                    <ResourceBar 
                      label="VRAM" 
                      value={(status.gpu.vram_used_mb / status.gpu.vram_total_mb) * 100} 
                      max={100}
                      displayValue={`${(status.gpu.vram_used_mb / 1024).toFixed(1)} / ${(status.gpu.vram_total_mb / 1024).toFixed(1)} GB`}
                      color="#14b8a6"
                    />
                    <Box display="flex" alignItems="center" gap={1.5}>
                      <TempIcon 
                        sx={{ 
                          fontSize: 18,
                          color: status.gpu.temperature_c > 85 ? '#ef4444' : 
                                 status.gpu.temperature_c > 75 ? '#f59e0b' : '#10b981'
                        }} 
                      />
                      <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                        Temperature: <strong>{status.gpu.temperature_c?.toFixed(0)}°C</strong>
                      </Typography>
                      {status.gpu.temperature_c > 85 && (
                        <Chip 
                          label="Critical" 
                          size="small" 
                          sx={{ 
                            height: 20,
                            bgcolor: alpha('#ef4444', 0.1),
                            border: `1px solid ${alpha('#ef4444', 0.3)}`,
                            color: '#f87171',
                            fontSize: '0.625rem',
                          }} 
                        />
                      )}
                      {status.gpu.temperature_c > 75 && status.gpu.temperature_c <= 85 && (
                        <Chip 
                          label="Warning" 
                          size="small" 
                          sx={{ 
                            height: 20,
                            bgcolor: alpha('#f59e0b', 0.1),
                            border: `1px solid ${alpha('#f59e0b', 0.3)}`,
                            color: '#fbbf24',
                            fontSize: '0.625rem',
                          }} 
                        />
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
    </Box>
  );
};

// Helper component for info rows
const InfoRow: React.FC<{ label: string; value: string; color?: string }> = ({ label, value, color }) => (
  <Box display="flex" justifyContent="space-between" alignItems="center">
    <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.8125rem' }}>
      {label}
    </Typography>
    <Typography 
      variant="body2" 
      sx={{ 
        fontWeight: 600, 
        fontSize: '0.8125rem',
        color: color || 'text.primary',
      }}
    >
      {value}
    </Typography>
  </Box>
);

// Helper component for resource bars
const ResourceBar: React.FC<{ 
  label: string; 
  value: number; 
  max: number;
  unit?: string;
  displayValue?: string;
  color: string;
  progress?: number;
}> = ({ label, value, max, unit = '', displayValue, color, progress }) => {
  const percentage = progress ?? Math.min((value / max) * 100, 100);
  const isHigh = percentage > 80;
  const barColor = isHigh ? '#ef4444' : color;
  
  return (
    <Box>
      <Box display="flex" justifyContent="space-between" mb={0.5}>
        <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.8125rem' }}>
          {label}
        </Typography>
        <Typography variant="body2" sx={{ fontWeight: 600, fontSize: '0.8125rem' }}>
          {displayValue || `${value.toFixed(1)}${unit}`}
        </Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={percentage}
        sx={{
          height: 6,
          borderRadius: 3,
          bgcolor: alpha(barColor, 0.15),
          '& .MuiLinearProgress-bar': {
            borderRadius: 3,
            background: `linear-gradient(90deg, ${barColor} 0%, ${alpha(barColor, 0.7)} 100%)`,
          },
        }}
      />
    </Box>
  );
};
