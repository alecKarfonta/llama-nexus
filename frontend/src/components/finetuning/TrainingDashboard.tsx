/**
 * Real-time Training Dashboard Component
 * Shows live training metrics, loss curves, GPU utilization, and logs
 */
import React, { useMemo } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  LinearProgress,
  Chip,
  IconButton,
  Tooltip,
  Alert,
  Divider,
  alpha,
} from '@mui/material';
import {
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  Timer as TimerIcon,
  TrendingDown as LossIcon,
  Thermostat as TempIcon,
  BoltOutlined as PowerIcon,
  FiberManualRecord as LiveIcon,
  Refresh as RefreshIcon,
  Timeline as ChartIcon,
  Terminal as LogIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  HourglassEmpty as PendingIcon,
  PlayArrow as RunningIcon,
} from '@mui/icons-material';
import { useTrainingMetrics, LossHistoryPoint, GPUMetrics, TrainingMetrics } from '@/hooks/useTrainingMetrics';

// Accent colors
const accentColors = {
  primary: '#6366f1',
  success: '#10b981',
  warning: '#f59e0b',
  info: '#06b6d4',
  purple: '#8b5cf6',
  rose: '#f43f5e',
};

// Animated Live Loss Chart
const LiveLossChart: React.FC<{ data: LossHistoryPoint[]; height?: number }> = ({ data, height = 200 }) => {
  if (data.length < 2) {
    return (
      <Box
        sx={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: 'rgba(0, 0, 0, 0.2)',
          borderRadius: 2,
        }}
      >
        <Box sx={{ textAlign: 'center' }}>
          <ChartIcon sx={{ fontSize: 40, color: 'text.secondary', opacity: 0.3, mb: 1 }} />
          <Typography variant="body2" color="text.secondary">
            Waiting for training data...
          </Typography>
        </Box>
      </Box>
    );
  }

  const width = 100;
  const chartHeight = 100;
  const padding = { left: 8, right: 8, top: 10, bottom: 10 };

  const losses = data.map((d) => d.loss);
  const steps = data.map((d) => d.step);
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const minStep = Math.min(...steps);
  const maxStep = Math.max(...steps);
  const lossRange = maxLoss - minLoss || 0.001;

  const scaleX = (step: number) =>
    padding.left + ((step - minStep) / (maxStep - minStep || 1)) * (width - padding.left - padding.right);
  const scaleY = (loss: number) =>
    chartHeight - padding.bottom - ((loss - minLoss) / lossRange) * (chartHeight - padding.top - padding.bottom);

  const pathD = data.map((d, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(d.step)} ${scaleY(d.loss)}`).join(' ');
  const areaD = `${pathD} L ${scaleX(data[data.length - 1].step)} ${chartHeight - padding.bottom} L ${scaleX(data[0].step)} ${chartHeight - padding.bottom} Z`;

  const currentLoss = data[data.length - 1]?.loss;
  const currentStep = data[data.length - 1]?.step;

  return (
    <Box sx={{ position: 'relative' }}>
      <svg viewBox={`0 0 ${width} ${chartHeight}`} style={{ width: '100%', height }}>
        <defs>
          <linearGradient id="lossGradientLive" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor={accentColors.success} stopOpacity="0.4" />
            <stop offset="100%" stopColor={accentColors.success} stopOpacity="0.02" />
          </linearGradient>
          <linearGradient id="lineGradientLive" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={accentColors.info} />
            <stop offset="100%" stopColor={accentColors.success} />
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="1.5" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Grid lines */}
        {[0.25, 0.5, 0.75].map((pct) => (
          <line
            key={pct}
            x1={padding.left}
            y1={padding.top + pct * (chartHeight - padding.top - padding.bottom)}
            x2={width - padding.right}
            y2={padding.top + pct * (chartHeight - padding.top - padding.bottom)}
            stroke="rgba(255, 255, 255, 0.05)"
            strokeDasharray="2,2"
          />
        ))}

        {/* Area fill */}
        <path d={areaD} fill="url(#lossGradientLive)" />

        {/* Loss curve */}
        <path
          d={pathD}
          fill="none"
          stroke="url(#lineGradientLive)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          filter="url(#glow)"
        />

        {/* Current point indicator with pulse animation */}
        <circle cx={scaleX(currentStep)} cy={scaleY(currentLoss)} r="4" fill={accentColors.success}>
          <animate attributeName="r" values="4;6;4" dur="1.5s" repeatCount="indefinite" />
          <animate attributeName="opacity" values="1;0.6;1" dur="1.5s" repeatCount="indefinite" />
        </circle>
      </svg>

      {/* Labels */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', px: 1, mt: 1 }}>
        <Box>
          <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.65rem' }}>
            Current Loss
          </Typography>
          <Typography
            sx={{
              fontWeight: 700,
              fontSize: '1.1rem',
              color: accentColors.success,
              fontFamily: 'monospace',
            }}
          >
            {currentLoss?.toFixed(4)}
          </Typography>
        </Box>
        <Box sx={{ textAlign: 'right' }}>
          <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.65rem' }}>
            Min Loss
          </Typography>
          <Typography
            sx={{
              fontWeight: 600,
              fontSize: '0.9rem',
              color: accentColors.info,
              fontFamily: 'monospace',
            }}
          >
            {minLoss.toFixed(4)}
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

// GPU Gauge Component
const GPUGauge: React.FC<{ value: number; max: number; label: string; unit: string; color: string }> = ({
  value,
  max,
  label,
  unit,
  color,
}) => {
  const percentage = max > 0 ? (value / max) * 100 : 0;

  return (
    <Box sx={{ textAlign: 'center' }}>
      <Box sx={{ position: 'relative', width: 80, height: 80, mx: 'auto' }}>
        <svg viewBox="0 0 36 36" style={{ width: '100%', height: '100%', transform: 'rotate(-90deg)' }}>
          {/* Background circle */}
          <circle
            cx="18"
            cy="18"
            r="15"
            fill="none"
            stroke="rgba(255, 255, 255, 0.1)"
            strokeWidth="3"
          />
          {/* Progress circle */}
          <circle
            cx="18"
            cy="18"
            r="15"
            fill="none"
            stroke={color}
            strokeWidth="3"
            strokeLinecap="round"
            strokeDasharray={`${percentage * 0.942} 100`}
            style={{
              transition: 'stroke-dasharray 0.5s ease-in-out',
              filter: `drop-shadow(0 0 4px ${alpha(color, 0.5)})`,
            }}
          />
        </svg>
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center',
          }}
        >
          <Typography sx={{ fontWeight: 700, fontSize: '0.9rem', lineHeight: 1 }}>
            {value.toFixed(1)}
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.6rem' }}>
            {unit}
          </Typography>
        </Box>
      </Box>
      <Typography variant="caption" sx={{ color: 'text.secondary', mt: 0.5, display: 'block' }}>
        {label}
      </Typography>
    </Box>
  );
};

// Stat Card Component
interface StatCardProps {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
  subtitle?: string;
  pulse?: boolean;
}

const StatCard: React.FC<StatCardProps> = ({ label, value, icon, color, subtitle, pulse }) => (
  <Box
    sx={{
      p: 2,
      borderRadius: 2,
      bgcolor: alpha(color, 0.08),
      border: `1px solid ${alpha(color, 0.15)}`,
      display: 'flex',
      alignItems: 'center',
      gap: 1.5,
      position: 'relative',
      overflow: 'hidden',
    }}
  >
    {pulse && (
      <Box
        sx={{
          position: 'absolute',
          top: 8,
          right: 8,
          width: 8,
          height: 8,
          borderRadius: '50%',
          bgcolor: color,
          animation: 'pulse 1.5s infinite',
          '@keyframes pulse': {
            '0%, 100%': { opacity: 1 },
            '50%': { opacity: 0.4 },
          },
        }}
      />
    )}
    <Box
      sx={{
        width: 40,
        height: 40,
        borderRadius: 2,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: `linear-gradient(135deg, ${color} 0%, ${alpha(color, 0.7)} 100%)`,
        boxShadow: `0 4px 12px ${alpha(color, 0.4)}`,
        '& .MuiSvgIcon-root': { fontSize: 20, color: '#fff' },
      }}
    >
      {icon}
    </Box>
    <Box>
      <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem', display: 'block' }}>
        {label}
      </Typography>
      <Typography sx={{ fontWeight: 700, fontSize: '1.1rem', color: 'text.primary', lineHeight: 1.2 }}>
        {value}
      </Typography>
      {subtitle && (
        <Typography variant="caption" sx={{ color: alpha(color, 0.8), fontSize: '0.65rem' }}>
          {subtitle}
        </Typography>
      )}
    </Box>
  </Box>
);

// Status indicator
const StatusIndicator: React.FC<{ status: string }> = ({ status }) => {
  const config: Record<string, { color: string; icon: React.ReactNode; label: string }> = {
    training: { color: accentColors.info, icon: <RunningIcon />, label: 'Training' },
    running: { color: accentColors.info, icon: <RunningIcon />, label: 'Running' },
    completed: { color: accentColors.success, icon: <SuccessIcon />, label: 'Completed' },
    failed: { color: accentColors.rose, icon: <ErrorIcon />, label: 'Failed' },
    pending: { color: accentColors.warning, icon: <PendingIcon />, label: 'Pending' },
    queued: { color: accentColors.purple, icon: <PendingIcon />, label: 'Queued' },
  };

  const { color, icon, label } = config[status.toLowerCase()] || config.pending;

  return (
    <Chip
      icon={icon as React.ReactElement}
      label={label}
      size="small"
      sx={{
        bgcolor: alpha(color, 0.1),
        color: color,
        border: `1px solid ${alpha(color, 0.3)}`,
        fontWeight: 600,
        fontSize: '0.75rem',
        '& .MuiChip-icon': { color: color },
      }}
    />
  );
};

// Main Dashboard Component
interface TrainingDashboardProps {
  jobId?: string;
  jobName?: string;
  onClose?: () => void;
}

export const TrainingDashboard: React.FC<TrainingDashboardProps> = ({ jobId, jobName, onClose }) => {
  const {
    isConnected,
    connectionStatus,
    error,
    currentMetrics,
    gpuMetrics,
    lossHistory,
    logs,
    tokensPerSecond,
    estimatedTimeRemaining,
    connect,
    disconnect,
    clearHistory,
  } = useTrainingMetrics({ jobId, autoConnect: true });

  const progress = currentMetrics?.progress ?? 0;
  const step = currentMetrics?.step ?? 0;
  const totalSteps = currentMetrics?.totalSteps ?? 0;

  return (
    <Card
      sx={{
        background: 'linear-gradient(145deg, rgba(30, 30, 63, 0.8) 0%, rgba(26, 26, 46, 0.95) 100%)',
        backdropFilter: 'blur(12px)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: 3,
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          px: 2.5,
          py: 2,
          borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Box
            sx={{
              width: 44,
              height: 44,
              borderRadius: 2,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: `linear-gradient(135deg, ${accentColors.primary} 0%, ${accentColors.purple} 100%)`,
              boxShadow: `0 4px 14px ${alpha(accentColors.primary, 0.4)}`,
            }}
          >
            <ChartIcon sx={{ color: '#fff', fontSize: 24 }} />
          </Box>
          <Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
              <Typography variant="h6" sx={{ fontWeight: 700, fontSize: '1rem' }}>
                {jobName || 'Training Monitor'}
              </Typography>
              {isConnected && (
                <Chip
                  icon={<LiveIcon sx={{ fontSize: '10px !important' }} />}
                  label="LIVE"
                  size="small"
                  sx={{
                    height: 20,
                    bgcolor: alpha(accentColors.success, 0.1),
                    color: accentColors.success,
                    border: `1px solid ${alpha(accentColors.success, 0.3)}`,
                    fontWeight: 700,
                    fontSize: '0.6rem',
                    '& .MuiChip-icon': {
                      color: accentColors.success,
                      animation: 'pulse 1s infinite',
                    },
                  }}
                />
              )}
            </Box>
            <Typography variant="caption" color="text.secondary">
              Step {step.toLocaleString()} / {totalSteps.toLocaleString()}
            </Typography>
          </Box>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {currentMetrics && <StatusIndicator status={currentMetrics.status} />}
          <Tooltip title={isConnected ? 'Disconnect' : 'Reconnect'}>
            <IconButton
              size="small"
              onClick={isConnected ? disconnect : connect}
              sx={{
                bgcolor: alpha(isConnected ? accentColors.success : accentColors.warning, 0.1),
                '&:hover': { bgcolor: alpha(isConnected ? accentColors.success : accentColors.warning, 0.2) },
              }}
            >
              <RefreshIcon sx={{ fontSize: 18, color: isConnected ? accentColors.success : accentColors.warning }} />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Progress Bar */}
      <Box sx={{ px: 2.5, py: 1.5, bgcolor: 'rgba(0, 0, 0, 0.2)' }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
          <Typography variant="caption" color="text.secondary">
            Training Progress
          </Typography>
          <Typography variant="caption" sx={{ fontWeight: 700, color: accentColors.success }}>
            {progress.toFixed(1)}%
          </Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={progress}
          sx={{
            height: 8,
            borderRadius: 4,
            bgcolor: 'rgba(255, 255, 255, 0.1)',
            '& .MuiLinearProgress-bar': {
              borderRadius: 4,
              background: `linear-gradient(90deg, ${accentColors.info} 0%, ${accentColors.success} 100%)`,
              boxShadow: `0 0 10px ${alpha(accentColors.success, 0.5)}`,
            },
          }}
        />
      </Box>

      <CardContent sx={{ p: 2.5 }}>
        {error && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Grid container spacing={2.5}>
          {/* Loss Chart */}
          <Grid item xs={12} lg={8}>
            <Box
              sx={{
                p: 2,
                borderRadius: 2,
                bgcolor: 'rgba(0, 0, 0, 0.2)',
                border: '1px solid rgba(255, 255, 255, 0.06)',
              }}
            >
              <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                <LossIcon sx={{ fontSize: 18, color: accentColors.success }} />
                Training Loss
              </Typography>
              <LiveLossChart data={lossHistory} height={180} />
            </Box>
          </Grid>

          {/* GPU Metrics */}
          <Grid item xs={12} lg={4}>
            <Box
              sx={{
                p: 2,
                borderRadius: 2,
                bgcolor: 'rgba(0, 0, 0, 0.2)',
                border: '1px solid rgba(255, 255, 255, 0.06)',
                height: '100%',
              }}
            >
              <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                <MemoryIcon sx={{ fontSize: 18, color: accentColors.info }} />
                GPU Status
              </Typography>

              {gpuMetrics ? (
                <Box sx={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 2 }}>
                  <GPUGauge
                    value={gpuMetrics.gpuUtilization}
                    max={100}
                    label="GPU Usage"
                    unit="%"
                    color={accentColors.info}
                  />
                  <GPUGauge
                    value={gpuMetrics.vramUsedGb}
                    max={gpuMetrics.vramTotalGb}
                    label="VRAM"
                    unit="GB"
                    color={accentColors.purple}
                  />
                  <GPUGauge
                    value={gpuMetrics.temperatureC}
                    max={100}
                    label="Temp"
                    unit="C"
                    color={gpuMetrics.temperatureC > 80 ? accentColors.rose : accentColors.success}
                  />
                </Box>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <MemoryIcon sx={{ fontSize: 40, color: 'text.secondary', opacity: 0.3, mb: 1 }} />
                  <Typography variant="body2" color="text.secondary">
                    Waiting for GPU data...
                  </Typography>
                </Box>
              )}
            </Box>
          </Grid>

          {/* Stats Row */}
          <Grid item xs={6} sm={3}>
            <StatCard
              label="Tokens/sec"
              value={tokensPerSecond ? `${tokensPerSecond.toFixed(0)}` : '--'}
              icon={<SpeedIcon />}
              color={accentColors.info}
              pulse={isConnected}
            />
          </Grid>
          <Grid item xs={6} sm={3}>
            <StatCard
              label="Time Remaining"
              value={estimatedTimeRemaining || '--'}
              icon={<TimerIcon />}
              color={accentColors.warning}
            />
          </Grid>
          <Grid item xs={6} sm={3}>
            <StatCard
              label="Current Loss"
              value={currentMetrics?.loss?.toFixed(4) || '--'}
              icon={<LossIcon />}
              color={accentColors.success}
              pulse={isConnected && currentMetrics?.status === 'training'}
            />
          </Grid>
          <Grid item xs={6} sm={3}>
            <StatCard
              label="GPU Power"
              value={gpuMetrics?.powerW ? `${gpuMetrics.powerW.toFixed(0)}W` : '--'}
              icon={<PowerIcon />}
              color={accentColors.purple}
            />
          </Grid>

          {/* Logs */}
          <Grid item xs={12}>
            <Box
              sx={{
                p: 2,
                borderRadius: 2,
                bgcolor: 'rgba(0, 0, 0, 0.3)',
                border: '1px solid rgba(255, 255, 255, 0.06)',
              }}
            >
              <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1.5, display: 'flex', alignItems: 'center', gap: 1 }}>
                <LogIcon sx={{ fontSize: 18, color: accentColors.warning }} />
                Training Logs
              </Typography>
              <Box
                sx={{
                  maxHeight: 150,
                  overflow: 'auto',
                  fontFamily: 'monospace',
                  fontSize: '0.75rem',
                  color: accentColors.success,
                  '& > div': {
                    py: 0.25,
                    borderBottom: '1px solid rgba(255, 255, 255, 0.03)',
                  },
                }}
              >
                {logs.length > 0 ? (
                  logs.slice(-50).map((log, i) => (
                    <div key={i}>{log.message}</div>
                  ))
                ) : (
                  <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'inherit' }}>
                    No logs yet...
                  </Typography>
                )}
              </Box>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default TrainingDashboard;
