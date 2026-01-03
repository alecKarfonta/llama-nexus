/**
 * Hook for real-time training metrics via WebSocket
 */
import { useState, useEffect, useCallback, useRef } from 'react';

export interface TrainingMetrics {
  jobId: string;
  step: number;
  totalSteps: number;
  loss: number | null;
  status: string;
  progress: number;
  adapterPath?: string;
  metrics?: Record<string, any>;
  timestamp: string;
}

export interface GPUMetrics {
  vramUsedGb: number;
  vramTotalGb: number;
  gpuUtilization: number;
  temperatureC: number;
  powerW: number | null;
}

export interface LossHistoryPoint {
  step: number;
  loss: number;
  timestamp: string;
}

export interface TrainingLog {
  jobId: string;
  message: string;
  timestamp: string;
}

interface UseTrainingMetricsOptions {
  jobId?: string;
  autoConnect?: boolean;
  maxHistoryPoints?: number;
}

interface UseTrainingMetricsReturn {
  // Connection state
  isConnected: boolean;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  error: string | null;

  // Metrics
  currentMetrics: TrainingMetrics | null;
  gpuMetrics: GPUMetrics | null;
  lossHistory: LossHistoryPoint[];
  logs: TrainingLog[];

  // Derived values
  tokensPerSecond: number | null;
  estimatedTimeRemaining: string | null;

  // Actions
  connect: () => void;
  disconnect: () => void;
  clearHistory: () => void;
}

export function useTrainingMetrics(options: UseTrainingMetricsOptions = {}): UseTrainingMetricsReturn {
  const { jobId, autoConnect = true, maxHistoryPoints = 500 } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [error, setError] = useState<string | null>(null);

  const [currentMetrics, setCurrentMetrics] = useState<TrainingMetrics | null>(null);
  const [gpuMetrics, setGpuMetrics] = useState<GPUMetrics | null>(null);
  const [lossHistory, setLossHistory] = useState<LossHistoryPoint[]>([]);
  const [logs, setLogs] = useState<TrainingLog[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const lastStepRef = useRef<number>(0);
  const stepTimestampsRef = useRef<{ step: number; time: number }[]>([]);

  // Calculate tokens per second based on step progress
  const tokensPerSecond = useCallback(() => {
    if (stepTimestampsRef.current.length < 2) return null;
    
    const recent = stepTimestampsRef.current.slice(-10);
    if (recent.length < 2) return null;
    
    const firstEntry = recent[0];
    const lastEntry = recent[recent.length - 1];
    const stepsDelta = lastEntry.step - firstEntry.step;
    const timeDelta = (lastEntry.time - firstEntry.time) / 1000; // Convert to seconds
    
    if (timeDelta <= 0) return null;
    
    // Assuming average ~512 tokens per step (this could be made configurable)
    const tokensPerStep = 512;
    return (stepsDelta * tokensPerStep) / timeDelta;
  }, []);

  // Calculate estimated time remaining
  const estimatedTimeRemaining = useCallback((): string | null => {
    if (!currentMetrics || currentMetrics.totalSteps <= 0) return null;
    if (stepTimestampsRef.current.length < 2) return null;

    const recent = stepTimestampsRef.current.slice(-20);
    if (recent.length < 2) return null;

    const firstEntry = recent[0];
    const lastEntry = recent[recent.length - 1];
    const stepsDelta = lastEntry.step - firstEntry.step;
    const timeDelta = lastEntry.time - firstEntry.time;

    if (stepsDelta <= 0) return null;

    const msPerStep = timeDelta / stepsDelta;
    const remainingSteps = currentMetrics.totalSteps - currentMetrics.step;
    const remainingMs = remainingSteps * msPerStep;

    // Format time
    const seconds = Math.floor(remainingMs / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  }, [currentMetrics]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setConnectionStatus('connecting');
    setError(null);

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const url = `${protocol}//${host}/ws/training`;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setConnectionStatus('connected');
        setError(null);
        console.log('Training WebSocket connected');
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          switch (message.type) {
            case 'connected':
              console.log('Training WebSocket confirmed:', message.data);
              break;

            case 'training_status':
              const payload = message.payload;
              // Only update if it's for our job (or no job filter)
              if (!jobId || payload.job_id === jobId) {
                const metrics: TrainingMetrics = {
                  jobId: payload.job_id,
                  step: payload.step,
                  totalSteps: payload.total_steps,
                  loss: payload.loss,
                  status: payload.status,
                  progress: payload.progress,
                  adapterPath: payload.adapter_path,
                  metrics: payload.metrics,
                  timestamp: payload.timestamp,
                };
                setCurrentMetrics(metrics);

                // Track step timestamps for speed calculation
                if (payload.step > lastStepRef.current) {
                  lastStepRef.current = payload.step;
                  stepTimestampsRef.current.push({
                    step: payload.step,
                    time: Date.now(),
                  });
                  // Keep only last 50 entries
                  if (stepTimestampsRef.current.length > 50) {
                    stepTimestampsRef.current = stepTimestampsRef.current.slice(-50);
                  }
                }

                // Add to loss history if loss is available
                if (payload.loss !== null && payload.loss !== undefined) {
                  setLossHistory((prev) => {
                    const newPoint: LossHistoryPoint = {
                      step: payload.step,
                      loss: payload.loss,
                      timestamp: payload.timestamp,
                    };
                    const updated = [...prev, newPoint];
                    // Keep only last N points
                    return updated.slice(-maxHistoryPoints);
                  });
                }
              }
              break;

            case 'training_log':
              const logPayload = message.payload;
              if (!jobId || logPayload.job_id === jobId) {
                setLogs((prev) => {
                  const newLog: TrainingLog = {
                    jobId: logPayload.job_id,
                    message: logPayload.message,
                    timestamp: logPayload.timestamp,
                  };
                  const updated = [...prev, newLog];
                  // Keep only last 200 logs
                  return updated.slice(-200);
                });
              }
              break;

            case 'gpu_metrics':
              const gpu = message.payload;
              if (gpu && Object.keys(gpu).length > 0) {
                setGpuMetrics({
                  vramUsedGb: gpu.vram_used_gb,
                  vramTotalGb: gpu.vram_total_gb,
                  gpuUtilization: gpu.gpu_utilization,
                  temperatureC: gpu.temperature_c,
                  powerW: gpu.power_w,
                });
              }
              break;
          }
        } catch (e) {
          console.error('Failed to parse training WebSocket message:', e);
        }
      };

      ws.onclose = (event) => {
        setIsConnected(false);
        setConnectionStatus('disconnected');
        wsRef.current = null;
        console.log('Training WebSocket disconnected:', event.code, event.reason);

        // Auto-reconnect after 3 seconds if not intentionally closed
        if (event.code !== 1000) {
          reconnectTimeoutRef.current = window.setTimeout(() => {
            connect();
          }, 3000);
        }
      };

      ws.onerror = () => {
        setConnectionStatus('error');
        setError('Failed to connect to training WebSocket');
        console.warn('Training WebSocket error');
      };
    } catch (e) {
      setConnectionStatus('error');
      setError('Failed to create WebSocket connection');
    }
  }, [jobId, maxHistoryPoints]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnecting');
      wsRef.current = null;
    }
    setIsConnected(false);
    setConnectionStatus('disconnected');
  }, []);

  const clearHistory = useCallback(() => {
    setLossHistory([]);
    setLogs([]);
    stepTimestampsRef.current = [];
    lastStepRef.current = 0;
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  // Reset history when job changes
  useEffect(() => {
    clearHistory();
  }, [jobId, clearHistory]);

  return {
    isConnected,
    connectionStatus,
    error,
    currentMetrics,
    gpuMetrics,
    lossHistory,
    logs,
    tokensPerSecond: tokensPerSecond(),
    estimatedTimeRemaining: estimatedTimeRemaining(),
    connect,
    disconnect,
    clearHistory,
  };
}

export default useTrainingMetrics;
