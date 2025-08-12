/**
 * Metrics Card Component - Displays real-time system metrics
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Chip,
  useTheme,
} from '@mui/material';
import {
  Memory as MemoryIcon,
  Speed as CpuIcon,
  Videocam as GpuIcon,
  Timer as TimerIcon,
} from '@mui/icons-material';

interface MetricsCardProps {
  title: string;
  value: number;
  unit: string;
  icon: React.ReactNode;
  color?: 'primary' | 'secondary' | 'error' | 'warning' | 'success';
  max?: number;
  subtitle?: string;
  showProgress?: boolean;
  trend?: 'up' | 'down' | 'stable';
  isPlaceholder?: boolean;
}

export const MetricsCard: React.FC<MetricsCardProps> = ({
  title,
  value,
  unit,
  icon,
  color = 'primary',
  max,
  subtitle,
  showProgress = false,
  trend,
  isPlaceholder = false,
}) => {
  const theme = useTheme();
  
  const percentage = max ? (value / max) * 100 : value;
  const isHighUsage = percentage > 80;
  const displayColor = isPlaceholder ? 'primary' : (isHighUsage ? 'warning' : color);

  const formatValue = (val: number): string => {
    if (unit === 'GB' && val > 1024) {
      return `${(val / 1024).toFixed(1)} TB`;
    }
    if (unit === 'MB' && val > 1024) {
      return `${(val / 1024).toFixed(1)} GB`;
    }
    if (val > 1000000) {
      return `${(val / 1000000).toFixed(1)}M`;
    }
    if (val > 1000) {
      return `${(val / 1000).toFixed(1)}K`;
    }
    return val.toFixed(1);
  };

  const getTrendIcon = () => {
    if (!trend) return null;
    switch (trend) {
      case 'up':
        return <span style={{ color: theme.palette.error.main }}>↗</span>;
      case 'down':
        return <span style={{ color: theme.palette.success.main }}>↘</span>;
      case 'stable':
        return <span style={{ color: theme.palette.text.secondary }}>→</span>;
      default:
        return null;
    }
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Box sx={{ 
            p: 1, 
            borderRadius: 1, 
            backgroundColor: isPlaceholder ? theme.palette.action.hover : theme.palette[displayColor].light + '20',
            color: isPlaceholder ? theme.palette.action.active : theme.palette[displayColor].main,
            mr: 2,
          }}>
            {icon}
          </Box>
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="h6" component="div" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {title}
              {getTrendIcon()}
            </Typography>
            {subtitle && (
              <Typography variant="body2" color="text.secondary">
                {subtitle}
              </Typography>
            )}
          </Box>
        </Box>

        <Box sx={{ mb: showProgress ? 2 : 0 }}>
          {isPlaceholder ? (
            <>
              <Typography variant="h4" component="div" color="text.disabled">
                --
                <Typography component="span" variant="h6" color="text.secondary" sx={{ ml: 1 }}>
                  {unit}
                </Typography>
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Awaiting backend integration
              </Typography>
            </>
          ) : (
            <>
              <Typography variant="h4" component="div" color={displayColor}>
                {formatValue(value)}
                <Typography component="span" variant="h6" color="text.secondary" sx={{ ml: 1 }}>
                  {unit}
                </Typography>
              </Typography>
              
              {max && (
                <Typography variant="body2" color="text.secondary">
                  of {formatValue(max)} {unit}
                </Typography>
              )}
            </>
          )}
        </Box>

        {showProgress && max && (
          <Box sx={{ mb: 2 }}>
            <LinearProgress 
              variant="determinate" 
              value={Math.min(percentage, 100)} 
              color={displayColor}
              sx={{ height: 8, borderRadius: 4 }}
            />
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
              <Typography variant="body2" color="text.secondary">
                {percentage.toFixed(1)}%
              </Typography>
              {isHighUsage && (
                <Chip label="High Usage" size="small" color="warning" variant="outlined" />
              )}
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

// Specialized metric cards for common use cases
export const CpuMetricsCard: React.FC<{ usage: number; cores?: number }> = ({ usage, cores }) => (
  <MetricsCard
    title="CPU Usage"
    value={usage}
    unit="%"
    icon={<CpuIcon />}
    color="primary"
    max={100}
    subtitle={cores ? `${cores} cores` : undefined}
    showProgress
  />
);

export const MemoryMetricsCard: React.FC<{ used: number; total: number }> = ({ used, total }) => (
  <MetricsCard
    title="Memory"
    value={used}
    unit="GB"
    icon={<MemoryIcon />}
    color="secondary"
    max={total}
    subtitle="System RAM"
    showProgress
  />
);

export const VramMetricsCard: React.FC<{ used: number; total: number }> = ({ used, total }) => (
  <MetricsCard
    title="VRAM"
    value={used}
    unit="GB"
    icon={<GpuIcon />}
    color="success"
    max={total}
    subtitle="GPU Memory"
    showProgress
  />
);

export const TokensPerSecondCard: React.FC<{ tokens: number; type?: 'prompt' | 'generation' }> = ({ 
  tokens, 
  type = 'generation' 
}) => (
  <MetricsCard
    title={`${type === 'prompt' ? 'Prompt' : 'Generation'} Speed`}
    value={tokens}
    unit="tok/s"
    icon={<TimerIcon />}
    color={type === 'prompt' ? 'primary' : 'secondary'}
    subtitle={`${type === 'prompt' ? 'Input' : 'Output'} throughput`}
  />
);
