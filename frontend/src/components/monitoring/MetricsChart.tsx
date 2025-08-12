/**
 * Metrics Chart Component - Real-time performance charts
 */

import React, { useMemo } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  ToggleButton,
  ToggleButtonGroup,
  useTheme,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Area,
  AreaChart,
} from 'recharts';
import type { ResourceMetrics } from '@/types/api';

interface MetricsChartProps {
  title: string;
  data: ResourceMetrics[];
  metrics: string[];
  colors?: string[];
  unit?: string;
  height?: number;
  type?: 'line' | 'area';
  showLegend?: boolean;
  timeRange?: '5m' | '15m' | '1h' | '6h' | '24h';
  onTimeRangeChange?: (range: '5m' | '15m' | '1h' | '6h' | '24h') => void;
}

export const MetricsChart: React.FC<MetricsChartProps> = ({
  title,
  data,
  metrics,
  colors,
  unit = '',
  height = 300,
  type = 'line',
  showLegend = true,
  timeRange = '1h',
  onTimeRangeChange,
}) => {
  const theme = useTheme();

  const defaultColors = [
    theme.palette.primary.main,
    theme.palette.secondary.main,
    theme.palette.error.main,
    theme.palette.warning.main,
    theme.palette.success.main,
    theme.palette.info.main,
  ];

  const chartColors = colors || defaultColors;

  const chartData = useMemo(() => {
    return data.map(item => ({
      ...item,
      timestamp: new Date(item.timestamp).getTime(),
      time: new Date(item.timestamp).toLocaleTimeString('en-US', { 
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
      }),
    }));
  }, [data]);

  const formatValue = (value: number): string => {
    if (unit === '%') {
      return `${value.toFixed(1)}%`;
    }
    if (unit === 'GB') {
      return `${value.toFixed(2)} GB`;
    }
    if (unit === 'tok/s') {
      return `${value.toFixed(0)} tok/s`;
    }
    return value.toFixed(2);
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Box
          sx={{
            backgroundColor: theme.palette.background.paper,
            border: `1px solid ${theme.palette.divider}`,
            borderRadius: 1,
            p: 1,
            boxShadow: theme.shadows[3],
          }}
        >
          <Typography variant="body2" sx={{ mb: 1 }}>
            {new Date(label).toLocaleString()}
          </Typography>
          {payload.map((entry: any, index: number) => (
            <Typography
              key={index}
              variant="body2"
              sx={{ color: entry.color }}
            >
              {entry.name}: {formatValue(entry.value)}
            </Typography>
          ))}
        </Box>
      );
    }
    return null;
  };

  const timeRangeOptions = [
    { value: '5m', label: '5m' },
    { value: '15m', label: '15m' },
    { value: '1h', label: '1h' },
    { value: '6h', label: '6h' },
    { value: '24h', label: '24h' },
  ];

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">{title}</Typography>
          {onTimeRangeChange && (
            <ToggleButtonGroup
              size="small"
              value={timeRange}
              exclusive
              onChange={(_, value) => value && onTimeRangeChange(value)}
            >
              {timeRangeOptions.map(option => (
                <ToggleButton key={option.value} value={option.value}>
                  {option.label}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>
          )}
        </Box>

        <ResponsiveContainer width="100%" height={height}>
          {type === 'area' ? (
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
              <XAxis 
                dataKey="time" 
                stroke={theme.palette.text.secondary}
                fontSize={12}
              />
              <YAxis 
                stroke={theme.palette.text.secondary}
                fontSize={12}
                tickFormatter={formatValue}
              />
              <Tooltip content={<CustomTooltip />} />
              {showLegend && <Legend />}
              {metrics.map((metric, index) => (
                <Area
                  key={metric}
                  type="monotone"
                  dataKey={metric}
                  stroke={chartColors[index % chartColors.length]}
                  fill={chartColors[index % chartColors.length]}
                  fillOpacity={0.2}
                  strokeWidth={2}
                />
              ))}
            </AreaChart>
          ) : (
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
              <XAxis 
                dataKey="time" 
                stroke={theme.palette.text.secondary}
                fontSize={12}
              />
              <YAxis 
                stroke={theme.palette.text.secondary}
                fontSize={12}
                tickFormatter={formatValue}
              />
              <Tooltip content={<CustomTooltip />} />
              {showLegend && <Legend />}
              {metrics.map((metric, index) => (
                <Line
                  key={metric}
                  type="monotone"
                  dataKey={metric}
                  stroke={chartColors[index % chartColors.length]}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
              ))}
            </LineChart>
          )}
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

// Specialized chart components
export const ResourceUsageChart: React.FC<{
  data: ResourceMetrics[];
  timeRange?: '5m' | '15m' | '1h' | '6h' | '24h';
  onTimeRangeChange?: (range: '5m' | '15m' | '1h' | '6h' | '24h') => void;
}> = ({ data, timeRange, onTimeRangeChange }) => (
  <MetricsChart
    title="Resource Usage"
    data={data}
    metrics={['cpuUsage', 'memoryUsage', 'gpuUsage', 'vramUsage']}
    unit="%"
    type="area"
    timeRange={timeRange}
    onTimeRangeChange={onTimeRangeChange}
  />
);

export const ThroughputChart: React.FC<{
  data: ResourceMetrics[];
  timeRange?: '5m' | '15m' | '1h' | '6h' | '24h';
  onTimeRangeChange?: (range: '5m' | '15m' | '1h' | '6h' | '24h') => void;
}> = ({ data, timeRange, onTimeRangeChange }) => (
  <MetricsChart
    title="Throughput Performance"
    data={data}
    metrics={['tokensPerSecond', 'requestRate']}
    unit="tok/s"
    type="line"
    timeRange={timeRange}
    onTimeRangeChange={onTimeRangeChange}
  />
);

export const ResponseTimeChart: React.FC<{
  data: ResourceMetrics[];
  timeRange?: '5m' | '15m' | '1h' | '6h' | '24h';
  onTimeRangeChange?: (range: '5m' | '15m' | '1h' | '6h' | '24h') => void;
}> = ({ data, timeRange, onTimeRangeChange }) => {
  const theme = useTheme();
  
  return (
    <MetricsChart
      title="Response Times"
      data={data}
      metrics={['responseTime']}
      unit="ms"
      type="line"
      colors={[theme.palette.warning.main]}
      timeRange={timeRange}
      onTimeRangeChange={onTimeRangeChange}
    />
  );
};
