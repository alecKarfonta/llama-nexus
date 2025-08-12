/**
 * React hooks for metrics data fetching and real-time updates
 */

import { useState, useEffect, useCallback } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiService } from '@/services/api';
import { websocketService } from '@/services/websocket';
import type { ResourceMetrics } from '@/types/api';

// Hook for fetching current metrics
export const useMetrics = (enabled: boolean = true) => {
  return useQuery({
    queryKey: ['metrics'],
    queryFn: () => apiService.getResourceMetrics(),
    enabled,
    refetchInterval: 5000, // Fallback polling every 5 seconds
    staleTime: 4000, // Consider data stale after 4 seconds
  });
};

// Hook for fetching metrics history
export const useMetricsHistory = (duration: string = '1h') => {
  return useQuery({
    queryKey: ['metrics', 'history', duration],
    queryFn: () => apiService.getMetricsHistory(duration),
    refetchInterval: 30000, // Refresh history every 30 seconds
    staleTime: 25000,
  });
};

// Hook for service status
export const useServiceStatus = (enabled: boolean = true) => {
  return useQuery({
    queryKey: ['service', 'status'],
    queryFn: () => apiService.getServiceStatus(),
    enabled,
    refetchInterval: 10000, // Check status every 10 seconds
    staleTime: 8000,
  });
};

// Hook for real-time metrics with WebSocket
export const useRealTimeMetrics = () => {
  const [metrics, setMetrics] = useState<ResourceMetrics | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<string>('disconnected');
  const [error, setError] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const connect = useCallback(async () => {
    try {
      setError(null);
      await websocketService.connect();
      setConnectionStatus(websocketService.connectionState);
    } catch (err) {
      // Don't show error for expected WebSocket unavailability
      const errorMessage = err instanceof Error ? err.message : 'Failed to connect';
      if (errorMessage.includes('WebSocket not available')) {
        console.info('WebSocket endpoint not available, using polling mode');
        setError(null);
      } else {
        setError(errorMessage);
      }
      setConnectionStatus('disconnected');
    }
  }, []);

  const disconnect = useCallback(() => {
    websocketService.disconnect();
    setConnectionStatus('disconnected');
  }, []);

  useEffect(() => {
    // Handle metrics updates
    const handleMetrics = (update: any) => {
      if (update.type === 'metrics') {
        setMetrics(update.payload);
        // Update React Query cache
        queryClient.setQueryData(['metrics'], update.payload);
      }
    };

    // Handle status updates
    const handleStatus = (update: any) => {
      if (update.type === 'status') {
        queryClient.setQueryData(['service', 'status'], update.payload);
      }
    };

    // Handle connection state changes
    const updateConnectionStatus = () => {
      setConnectionStatus(websocketService.connectionState);
    };

    // Subscribe to events
    websocketService.onMetrics(handleMetrics);
    websocketService.onStatus(handleStatus);
    websocketService.onError((error) => {
      // Don't show errors for expected WebSocket unavailability
      const errorMessage = error.message || 'WebSocket error';
      if (!errorMessage.includes('WebSocket disconnected')) {
        setError(errorMessage);
        setConnectionStatus('error');
      }
    });

    // Monitor connection state
    const connectionMonitor = setInterval(updateConnectionStatus, 1000);

    return () => {
      clearInterval(connectionMonitor);
      websocketService.off('metrics', handleMetrics);
      websocketService.off('status', handleStatus);
    };
  }, [queryClient]);

  return {
    metrics,
    connectionStatus,
    error,
    connect,
    disconnect,
    isConnected: websocketService.isConnected,
  };
};

// Hook for parsing Prometheus metrics from llamacpp
export const usePrometheusMetrics = () => {
  const [parsedMetrics, setParsedMetrics] = useState<Record<string, number>>({});

  const parseMetrics = useCallback((metricsText: string) => {
    const metrics: Record<string, number> = {};
    const lines = metricsText.split('\n');
    
    for (const line of lines) {
      // Skip comments and empty lines
      if (line.startsWith('#') || !line.trim()) continue;
      
      // Parse metric lines (format: metric_name value)
      const match = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([0-9.-]+)/);
      if (match) {
        const [, name, value] = match;
        metrics[name] = parseFloat(value);
      }
    }
    
    setParsedMetrics(metrics);
    return metrics;
  }, []);

  // Fetch and parse current metrics
  const fetchMetrics = useCallback(async () => {
    try {
      const text = await apiService.getPrometheusMetrics();
      return parseMetrics(text);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
    return {};
  }, [parseMetrics]);

  return {
    parsedMetrics,
    parseMetrics,
    fetchMetrics,
  };
};

// Hook for health check with automatic retry
export const useHealthCheck = () => {
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);
  const [lastCheck, setLastCheck] = useState<Date | null>(null);

  const checkHealth = useCallback(async () => {
    try {
      const result = await apiService.healthCheck();
      const healthy = result.status === 'healthy';
      setIsHealthy(healthy);
      setLastCheck(new Date());
      return healthy;
    } catch (error) {
      setIsHealthy(false);
      setLastCheck(new Date());
      return false;
    }
  }, []);

  useEffect(() => {
    // Initial check
    checkHealth();

    // Check every 30 seconds
    const interval = setInterval(checkHealth, 30000);

    return () => clearInterval(interval);
  }, [checkHealth]);

  return {
    isHealthy,
    lastCheck,
    checkHealth,
  };
};
