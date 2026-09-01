import React, { useEffect, useState } from 'react';
import {
  Typography,
  Box,
  CircularProgress,
  Alert,
  Collapse,
  LinearProgress,
  IconButton,
  Tooltip,
  alpha,
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Refresh as RestartIcon,
  ExpandLess as ExpandLessIcon,
  ExpandMore as ExpandMoreIcon,
  Thermostat as TempIcon,
  Memory as MemoryIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material';
import { StatusChip, type StatusChipConfigMap } from '@/components/common';
import { accentColors } from '@/utils/theme';

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
  llamacpp_build?: {
    tag?: string;
    build_number?: number;
    cli_version?: string;
    mtp_supported?: boolean;
  };
  mtp_supported?: boolean;
}

const serviceStatusConfig: StatusChipConfigMap = {
  running: { color: accentColors.success, label: 'Running' },
  stopped: { color: accentColors.slate, label: 'Stopped' },
  unhealthy: { color: accentColors.warning, label: 'Unhealthy' },
  critical: { color: accentColors.rose, label: 'Critical' },
  warning: { color: accentColors.warning, label: 'Warning' },
};

const serviceActionButtonSx = {
  width: 32,
  height: 32,
  borderRadius: '8px',
  border: '1px solid',
  borderColor: 'divider',
  bgcolor: 'action.hover',
  '& .MuiSvgIcon-root': { fontSize: 18 },
  '&:hover': { bgcolor: 'action.selected' },
} as const;

const servicePanelSx = {
  p: 1.5,
  borderRadius: '8px',
  bgcolor: 'rgba(255, 255, 255, 0.02)',
  border: '1px solid rgba(255, 255, 255, 0.04)',
} as const;

export const ServiceStatusDisplay: React.FC = () => {
  const [status, setStatus] = useState<ServiceStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

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
    const interval = setInterval(fetchStatus, 10000);
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
    if (!running) return accentColors.slate;
    if (health === false) return accentColors.warning;
    return accentColors.success;
  };

  const getStatusKey = (running: boolean, health?: boolean) => {
    if (!running) return 'stopped';
    if (health === false) return 'unhealthy';
    return 'running';
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={120}>
        <CircularProgress size={32} sx={{ color: accentColors.primary }} />
      </Box>
    );
  }

  if (error && !status) {
    return (
      <Alert
        severity="error"
        onClose={() => setError(null)}
        sx={{
          bgcolor: alpha(accentColors.rose, 0.1),
          border: `1px solid ${alpha(accentColors.rose, 0.2)}`,
          color: accentColors.rose,
        }}
      >
        {error}
      </Alert>
    );
  }

  const serviceRunning = Boolean(status?.running);
  const statusKey = getStatusKey(serviceRunning, status?.llamacpp_health?.healthy);
  const statusColor = getStatusColor(serviceRunning, status?.llamacpp_health?.healthy);
  const buildLabel = status?.llamacpp_build?.tag || status?.llamacpp_build?.cli_version || 'unknown';
  const summaryItems = [
    { label: 'Model', value: status?.model?.name || 'None', color: accentColors.purple },
    { label: 'Variant', value: status?.model?.variant || 'N/A', color: accentColors.info },
    { label: 'Context', value: status?.model?.context_size?.toLocaleString() || 'N/A', color: accentColors.primary },
    { label: 'GPU Layers', value: status?.model?.gpu_layers?.toString() || 'N/A', color: accentColors.success },
    { label: 'Build', value: buildLabel, color: accentColors.warning },
    {
      label: 'MTP',
      value: status?.mtp_supported == null ? 'unknown' : status.mtp_supported ? 'yes' : 'no - upgrade b9193+',
      color: status?.mtp_supported ? accentColors.success : accentColors.warning,
    },
  ];

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1} flexWrap="wrap" gap={1}>
        <Box display="flex" alignItems="center" gap={1}>
          <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>LlamaCPP Service</Typography>
          <StatusChip
            status={statusKey}
            statusConfig={serviceStatusConfig}
            sx={{
              fontSize: '0.7rem',
              '& .MuiChip-label': { px: 1 },
            }}
          />
        </Box>

        <Box display="flex" gap={0.5}>
          {!serviceRunning && (
            <Tooltip title="Start service">
              <span>
                <IconButton
                  aria-label="Start service"
                  onClick={() => handleServiceAction('start')}
                  disabled={actionLoading !== null}
                  size="small"
                  sx={{
                    ...serviceActionButtonSx,
                    color: accentColors.success,
                    borderColor: alpha(accentColors.success, 0.34),
                  }}
                >
                  {actionLoading === 'start' ? <CircularProgress size={15} color="inherit" /> : <StartIcon />}
                </IconButton>
              </span>
            </Tooltip>
          )}
          {serviceRunning && (
            <>
              <Tooltip title="Stop service">
                <span>
                  <IconButton
                    aria-label="Stop service"
                    onClick={() => handleServiceAction('stop')}
                    disabled={actionLoading !== null}
                    size="small"
                    sx={{
                      ...serviceActionButtonSx,
                      color: accentColors.rose,
                      borderColor: alpha(accentColors.rose, 0.32),
                    }}
                  >
                    {actionLoading === 'stop' ? <CircularProgress size={15} color="inherit" /> : <StopIcon />}
                  </IconButton>
                </span>
              </Tooltip>
              <Tooltip title="Restart service">
                <span>
                  <IconButton
                    aria-label="Restart service"
                    onClick={() => handleServiceAction('restart')}
                    disabled={actionLoading !== null}
                    size="small"
                    sx={{
                      ...serviceActionButtonSx,
                      color: accentColors.warning,
                      borderColor: alpha(accentColors.warning, 0.34),
                    }}
                  >
                    {actionLoading === 'restart' ? <CircularProgress size={15} color="inherit" /> : <RestartIcon />}
                  </IconButton>
                </span>
              </Tooltip>
            </>
          )}
        </Box>
      </Box>

      {error && (
        <Alert
          severity="error"
          onClose={() => setError(null)}
          sx={{
            mb: 2,
            bgcolor: alpha(accentColors.rose, 0.1),
            border: `1px solid ${alpha(accentColors.rose, 0.2)}`,
            color: accentColors.rose,
          }}
        >
          {error}
        </Alert>
      )}

      <Box
        sx={{
          ...servicePanelSx,
          display: 'grid',
          gridTemplateColumns: { xs: 'repeat(2, minmax(0, 1fr))', sm: 'repeat(3, minmax(0, 1fr))' },
          gap: 0.75,
          p: 1,
        }}
      >
        {summaryItems.map((item) => (
          <Box key={item.label} sx={{ minWidth: 0, borderLeft: `2px solid ${alpha(item.color, 0.7)}`, pl: 0.75 }}>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', lineHeight: 1.2 }}>
              {item.label}
            </Typography>
            <Typography
              variant="body2"
              title={item.value}
              noWrap
              sx={{ fontWeight: 800, color: item.color === accentColors.warning && item.label === 'MTP' ? item.color : 'text.primary' }}
            >
              {item.value}
            </Typography>
          </Box>
        ))}
      </Box>

      {(status?.running || status?.resources || status?.gpu) && (
        <>
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 0.5 }}>
            <Tooltip title={detailsOpen ? 'Hide runtime details' : 'Show runtime details'}>
              <IconButton
                size="small"
                aria-label={detailsOpen ? 'Hide runtime details' : 'Show runtime details'}
                onClick={() => setDetailsOpen((value) => !value)}
                sx={{ ...serviceActionButtonSx, width: 28, height: 28 }}
              >
                {detailsOpen ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              </IconButton>
            </Tooltip>
          </Box>

          <Collapse in={detailsOpen} unmountOnExit>
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', md: 'repeat(2, minmax(0, 1fr))' },
                gap: 1,
                mt: 1,
              }}
            >
              <Box sx={servicePanelSx}>
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <SpeedIcon sx={{ fontSize: 18, color: accentColors.info }} />
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                    Runtime
                  </Typography>
                </Box>
                <Box display="flex" flexDirection="column" gap={1}>
                  <InfoRow label="Status" value={status?.running ? 'Running' : 'Stopped'} color={statusColor} />
                  <InfoRow label="PID" value={status?.pid?.toString() || 'N/A'} />
                  <InfoRow label="Uptime" value={status?.uptime ? formatUptime(status.uptime) : 'N/A'} />
                  <InfoRow label="Mode" value={status?.mode || 'subprocess'} />
                </Box>
              </Box>

              {status?.resources && (
                <Box sx={servicePanelSx}>
                  <Box display="flex" alignItems="center" gap={1} mb={1}>
                    <MemoryIcon sx={{ fontSize: 18, color: accentColors.purple }} />
                    <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                      Resources
                    </Typography>
                  </Box>
                  <Box display="flex" flexDirection="column" gap={1.25}>
                    <ResourceBar
                      label="CPU Usage"
                      value={status.resources.cpu_percent}
                      max={100}
                      unit="%"
                      color={accentColors.primary}
                    />
                    <ResourceBar
                      label="Memory"
                      value={status.resources.memory_mb / 1024}
                      max={100}
                      displayValue={`${(status.resources.memory_mb / 1024).toFixed(1)} GB (${status.resources.memory_percent?.toFixed(1)}%)`}
                      color={accentColors.purple}
                      progress={status.resources.memory_percent}
                    />
                  </Box>
                </Box>
              )}

              {status?.gpu && (
                <Box sx={servicePanelSx}>
                  <Box display="flex" alignItems="center" gap={1} mb={1}>
                    <MemoryIcon sx={{ fontSize: 18, color: accentColors.success }} />
                    <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                      GPU
                    </Typography>
                  </Box>
                  <Box display="flex" flexDirection="column" gap={1.25}>
                    <ResourceBar
                      label="GPU Usage"
                      value={status.gpu.gpu_usage_percent}
                      max={100}
                      unit="%"
                      color={accentColors.success}
                    />
                    <ResourceBar
                      label="VRAM"
                      value={(status.gpu.vram_used_mb / status.gpu.vram_total_mb) * 100}
                      max={100}
                      displayValue={`${(status.gpu.vram_used_mb / 1024).toFixed(1)} / ${(status.gpu.vram_total_mb / 1024).toFixed(1)} GB`}
                      color={accentColors.info}
                    />
                    <Box display="flex" alignItems="center" gap={1}>
                      <TempIcon
                        sx={{
                          fontSize: 18,
                          color: status.gpu.temperature_c > 85 ? accentColors.rose :
                            status.gpu.temperature_c > 75 ? accentColors.warning : accentColors.success,
                        }}
                      />
                      <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                        Temperature: <strong>{status.gpu.temperature_c?.toFixed(0)}°C</strong>
                      </Typography>
                    </Box>
                  </Box>
                </Box>
              )}
            </Box>
          </Collapse>
        </>
      )}
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
  const barColor = isHigh ? accentColors.rose : color;

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
          borderRadius: '3px',
          bgcolor: alpha(barColor, 0.15),
          '& .MuiLinearProgress-bar': {
            borderRadius: '3px',
            background: `linear-gradient(90deg, ${barColor} 0%, ${alpha(barColor, 0.7)} 100%)`,
          },
        }}
      />
    </Box>
  );
};
