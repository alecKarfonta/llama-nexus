/**
 * Real-time Metrics Display - Main monitoring dashboard component
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Grid,
  Box,
  Typography,
  Alert,
  Card,
  CardContent,
  Button,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Wifi as ConnectedIcon,
  Stop as DisconnectIcon,
} from '@mui/icons-material';

import { MetricsCard, TokensPerSecondCard } from './MetricsCard';
import { ServiceStatusCard } from './ServiceStatusCard';
import { MetricsChart, ResourceUsageChart, ThroughputChart, ResponseTimeChart } from './MetricsChart';

import { useRealTimeMetrics, usePrometheusMetrics, useServiceStatus, useMetrics, useMetricsHistory } from '@/hooks/useMetrics';
import type { ResourceMetrics } from '@/types/api';

interface RealTimeMetricsDisplayProps {
  showWebSocketStatus?: boolean;
}

export const RealTimeMetricsDisplay: React.FC<RealTimeMetricsDisplayProps> = ({
  showWebSocketStatus = true,
}) => {

  const [timeRange, setTimeRange] = useState<'5m' | '15m' | '1h' | '6h' | '24h'>('1h');

  // Real-time WebSocket connection (not used until endpoint is implemented)
  const { metrics, connectionStatus, error, disconnect, isConnected } = useRealTimeMetrics();

  // Current metrics from API
  const { data: currentMetrics, isLoading: metricsLoading, error: metricsError, refetch: refetchMetrics } = useMetrics();

  // Metrics history for charts
  const { data: metricsHistory, isLoading: historyLoading, error: historyError } = useMetricsHistory(timeRange);

  // Prometheus metrics parsing
  const { parsedMetrics, fetchMetrics } = usePrometheusMetrics();

  // Service status
  const { data: serviceStatus, isLoading: statusLoading, error: statusError } = useServiceStatus();



  // Auto-connect on mount (disabled until WebSocket endpoint exists)
  useEffect(() => {
    // Skip WebSocket connection attempt entirely since endpoint doesn't exist yet
    // if (autoConnect && !isConnected) {
    //   connect();
    // }

    // Fetch initial metrics
    fetchMetrics();

    // Set up polling for Prometheus metrics (primary mode for now)
    const interval = setInterval(fetchMetrics, 10000);

    return () => {
      clearInterval(interval);
      if (isConnected) {
        disconnect();
      }
    };
  }, [isConnected, fetchMetrics, disconnect]);

  // Use real metrics from WebSocket if available, otherwise from API
  const displayMetrics = metrics || currentMetrics;

  const handleRefresh = () => {
    fetchMetrics();
    refetchMetrics();
  };

  const renderConnectionStatus = () => {
    // Only show WebSocket status if actually connected (for future use)
    if (!showWebSocketStatus || !isConnected) return null;

    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Chip
          icon={<ConnectedIcon />}
          label="WebSocket: Connected"
          color="success"
          size="small"
          variant="outlined"
        />
        <Button
          size="small"
          startIcon={<DisconnectIcon />}
          onClick={disconnect}
          color="secondary"
        >
          Disconnect
        </Button>
      </Box>
    );
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Real-time Monitoring
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {renderConnectionStatus()}
          <Tooltip title="Refresh metrics">
            <IconButton onClick={handleRefresh}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* WebSocket Status Info - Only show if user tries to connect */}
      {connectionStatus === 'error' && showWebSocketStatus && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>Note:</strong> WebSocket endpoint is not implemented yet.
            Using polling mode with 5-second updates.
          </Typography>
        </Alert>
      )}

      {/* Error Alert for metrics or other errors */}
      {(error && !error.includes('WebSocket not available')) || metricsError ? (
        <Alert severity="error" sx={{ mb: 3 }}>
          Error: {error || metricsError?.message || 'Failed to load metrics'}
        </Alert>
      ) : null}

      {/* Temperature Alert - Will be enabled when backend provides temperature data */}
      {/* {serviceStatus?.gpu?.temperature_c && serviceStatus.gpu.temperature_c > 80 && (
        <Alert 
          severity={serviceStatus.gpu.temperature_c > 85 ? "error" : "warning"} 
          sx={{ mb: 3 }}
        >
          <Typography variant="body2">
            <strong>GPU Temperature Alert:</strong> GPU running at {serviceStatus.gpu.temperature_c.toFixed(0)}°C
            {serviceStatus.gpu.temperature_c > 85 
              ? " - Critical temperature! Consider reducing load or improving cooling."
              : " - Temperature elevated. Monitor closely."
            }
          </Typography>
        </Alert>
      )} */}

      {/* Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6} lg={3}>
          <ServiceStatusCard
            status={serviceStatus}
            loading={statusLoading}
            error={statusError?.message}
          />
        </Grid>

        <Grid item xs={12} md={6} lg={3}>
          <MetricsCard
            title="CPU Usage"
            value={displayMetrics?.cpuUsage || 0}
            unit="%"
            icon={<RefreshIcon />}
            color="primary"
            max={100}
            subtitle="System CPU"
            showProgress={true}
            isPlaceholder={metricsLoading && !displayMetrics}
          />
        </Grid>

        <Grid item xs={12} md={6} lg={3}>
          <MetricsCard
            title="Memory"
            value={displayMetrics?.memoryUsed || 0}
            unit="GB"
            icon={<RefreshIcon />}
            color="secondary"
            max={displayMetrics?.memoryTotal || 64}
            subtitle="System RAM"
            showProgress={true}
            isPlaceholder={metricsLoading && !displayMetrics}
          />
        </Grid>

        <Grid item xs={12} md={6} lg={3}>
          <MetricsCard
            title="VRAM"
            value={displayMetrics?.vramUsed || 0}
            unit="GB"
            icon={<RefreshIcon />}
            color="success"
            max={displayMetrics?.vramTotal || 32}
            subtitle="GPU Memory"
            showProgress={true}
            isPlaceholder={metricsLoading && !displayMetrics}
          />
        </Grid>
      </Grid>

      {/* Performance Metrics */}
      {displayMetrics && (
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={4}>
            <TokensPerSecondCard
              tokens={parsedMetrics['llamacpp:prompt_tokens_seconds'] || 0}
              type="prompt"
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <TokensPerSecondCard
              tokens={parsedMetrics['llamacpp:predicted_tokens_seconds'] || 0}
              type="generation"
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <MetricsCard
              title="Active Requests"
              value={parsedMetrics['llamacpp:requests_processing'] || 0}
              unit="req"
              icon={<RefreshIcon />}
              color="secondary"
              subtitle="Currently processing"
            />
          </Grid>
        </Grid>
      )}

      {/* Historical Charts */}
      {metricsHistory && metricsHistory.length > 0 && (
        <>
          <Typography variant="h5" sx={{ mb: 3, mt: 2 }}>
            Historical Performance
          </Typography>

          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} lg={6}>
              <ResourceUsageChart
                data={metricsHistory}
                timeRange={timeRange}
                onTimeRangeChange={setTimeRange}
              />
            </Grid>

            <Grid item xs={12} lg={6}>
              <ThroughputChart
                data={metricsHistory}
                timeRange={timeRange}
                onTimeRangeChange={setTimeRange}
              />
            </Grid>
          </Grid>

          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} lg={6}>
              <ResponseTimeChart
                data={metricsHistory}
                timeRange={timeRange}
                onTimeRangeChange={setTimeRange}
              />
            </Grid>

            <Grid item xs={12} lg={6}>
              <MetricsChart
                title="GPU Utilization & Temperature"
                data={metricsHistory}
                metrics={['gpuUsage', 'vramUsage']}
                unit="%"
                type="area"
                timeRange={timeRange}
                onTimeRangeChange={setTimeRange}
              />
            </Grid>
          </Grid>
        </>
      )}

      {/* Fallback for no data */}
      {!displayMetrics && !metricsLoading && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              No Metrics Data
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Unable to fetch metrics from the service. Please check your connection and try refreshing.
            </Typography>
            <Button startIcon={<RefreshIcon />} onClick={handleRefresh} sx={{ mt: 2 }}>
              Retry
            </Button>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};
