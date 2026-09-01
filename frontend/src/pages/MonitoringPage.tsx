import { useState, useEffect, useCallback } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  Box,
  Grid,
  Typography,
  Button,
  IconButton,
  Alert,
  LinearProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tooltip,
  Switch,
  FormControlLabel,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import {
  Refresh as RefreshIcon,
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  Timer as TimerIcon,
  TrendingUp as TrendingIcon,
  DataUsage as DataUsageIcon,
  Cached as CachedIcon,
  Token as TokenIcon,
  Chat as ChatIcon,
  RocketLaunch as DeployIcon,
} from '@mui/icons-material'
import { useMetrics, useServiceStatus, useRealTimeMetrics } from '@/hooks/useMetrics'
import { TokenUsageTracker } from '@/components/monitoring/TokenUsageTracker'
import { MetricTile, PageShell, SectionCard, StatusChip, type StatusChipConfigMap } from '@/components/common'
import { getRoutePath } from '@/config/navigation'
import type { DeploymentRouteContext, RuntimeRouteState } from '@/types/runtimeHandoff'
import { modelInfoFromDeployment, runtimeBackendLabel } from '@/types/runtimeHandoff'
import { accentColors } from '@/utils/theme'

const MONITORING_STATUS_CONFIG: StatusChipConfigMap = {
  checking: { color: accentColors.warning, label: 'Checking health' },
  connected: { color: accentColors.success, label: 'Live websocket' },
  deferred: { color: accentColors.warning, label: 'Deferred' },
  degraded: { color: accentColors.warning, label: 'Degraded' },
  good: { color: accentColors.success, label: 'Good cache performance' },
  healthy: { color: accentColors.success, label: 'Healthy' },
  info: { color: accentColors.info, label: 'Info' },
  operations: { color: accentColors.primary, label: 'Operations' },
  polling: { color: accentColors.info, label: 'Polling mode' },
  processing: { color: accentColors.success, label: 'Processing' },
  starting: { color: accentColors.warning, label: 'Starting' },
  stopped: { color: accentColors.rose, label: 'Stopped' },
  target: { color: accentColors.info, label: 'Runtime target' },
  unavailable: { color: accentColors.slate, label: 'Metrics unavailable' },
  warning: { color: accentColors.warning, label: 'Warning' },
}

const monitoringPanelSx = {
  borderRadius: '8px',
  bgcolor: 'background.paper',
  border: '1px solid',
  borderColor: 'divider',
} as const

interface PrometheusMetrics {
  // Request metrics
  llamacpp_requests_processing?: number
  llamacpp_requests_deferred?: number
  llamacpp_requests_total?: number
  
  // Token metrics
  llamacpp_tokens_predicted_total?: number
  llamacpp_tokens_evaluated_total?: number
  llamacpp_tokens_cached_total?: number
  
  // Timing metrics
  llamacpp_prompt_eval_time_seconds_total?: number
  llamacpp_tokens_eval_time_seconds_total?: number
  llamacpp_time_to_first_token_seconds_sum?: number
  llamacpp_time_to_first_token_seconds_count?: number
  
  // KV Cache metrics
  llamacpp_kv_cache_tokens?: number
  llamacpp_kv_cache_max_tokens?: number
  llamacpp_kv_cache_usage_ratio?: number
  
  // Slot metrics
  llamacpp_slot_active?: number
  llamacpp_slot_idle?: number
  
  // Other metrics
  llamacpp_n_decode_total?: number
  llamacpp_n_busy_slots_per_decode?: number
}

export default function MonitoringPage() {
  const location = useLocation()
  const navigate = useNavigate()
  const benchmarkPath = getRoutePath('benchmark')
  const chatPath = getRoutePath('chat')
  const deployPath = getRoutePath('deploy')
  const routeState = location.state as RuntimeRouteState | null
  const incomingDeployment = routeState?.deployment
  const [tabValue, setTabValue] = useState(0)
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [prometheusMetrics, setPrometheusMetrics] = useState<PrometheusMetrics>({})
  const [deploymentHandoff, setDeploymentHandoff] = useState<DeploymentRouteContext | null>(null)
  const [error, setError] = useState<string | null>(null)
  
  const { data: metrics, isLoading: metricsLoading, refetch: refetchMetrics } = useMetrics()
  const { data: serviceStatus, isLoading: statusLoading } = useServiceStatus()
  const { connect, disconnect, connectionStatus } = useRealTimeMetrics()

  useEffect(() => {
    if (!incomingDeployment?.modelName) return
    setDeploymentHandoff(incomingDeployment)
    setAutoRefresh(true)
  }, [
    incomingDeployment?.backend,
    incomingDeployment?.endpoint,
    incomingDeployment?.modelName,
  ])
  
  // Fetch Prometheus metrics from llama.cpp
  const fetchPrometheusMetrics = useCallback(async () => {
    try {
      const response = await fetch('/metrics')
      if (!response.ok) {
        // LlamaCPP not running, return empty metrics
        setPrometheusMetrics({})
        return
      }
      
      const text = await response.text()
      const parsed: PrometheusMetrics = {}
      
      // Parse Prometheus text format
      const lines = text.split('\n')
      for (const line of lines) {
        if (line.startsWith('#') || !line.trim()) continue
        
        // Handle simple metrics: metric_name value
        const simpleMatch = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([0-9.eE+-]+)/)
        if (simpleMatch) {
          const [, name, value] = simpleMatch
          parsed[name as keyof PrometheusMetrics] = parseFloat(value)
        }
        
        // Handle labeled metrics: metric_name{labels} value
        const labeledMatch = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)\{[^}]*\}\s+([0-9.eE+-]+)/)
        if (labeledMatch) {
          const [, name, value] = labeledMatch
          // For now, just take the first value for each metric name
          if (!(name in parsed)) {
            parsed[name as keyof PrometheusMetrics] = parseFloat(value)
          }
        }
      }
      
      setPrometheusMetrics(parsed)
      setError(null)
    } catch (err: any) {
      // Don't show error if LlamaCPP just isn't running
      if (!err.message?.includes('Failed to fetch')) {
        setError(err.message)
      }
      setPrometheusMetrics({})
    }
  }, [])
  
  // Initial fetch and auto-refresh
  useEffect(() => {
    fetchPrometheusMetrics()
    
    let interval: NodeJS.Timeout | null = null
    if (autoRefresh) {
      interval = setInterval(fetchPrometheusMetrics, 5000)
    }
    
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [fetchPrometheusMetrics, autoRefresh])
  
  // Connect to WebSocket for real-time updates
  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])
  
  // Calculate derived metrics
  const avgTimeToFirstToken = prometheusMetrics.llamacpp_time_to_first_token_seconds_count
    ? (prometheusMetrics.llamacpp_time_to_first_token_seconds_sum || 0) / prometheusMetrics.llamacpp_time_to_first_token_seconds_count * 1000
    : 0
  
  const tokensPerSecond = prometheusMetrics.llamacpp_tokens_eval_time_seconds_total
    ? (prometheusMetrics.llamacpp_tokens_predicted_total || 0) / prometheusMetrics.llamacpp_tokens_eval_time_seconds_total
    : 0
  
  const kvCacheUsage = prometheusMetrics.llamacpp_kv_cache_usage_ratio
    ? prometheusMetrics.llamacpp_kv_cache_usage_ratio * 100
    : 0
  
  const cacheHitRate = (prometheusMetrics.llamacpp_tokens_cached_total || 0) > 0 && (prometheusMetrics.llamacpp_tokens_evaluated_total || 0) > 0
    ? ((prometheusMetrics.llamacpp_tokens_cached_total || 0) / 
       ((prometheusMetrics.llamacpp_tokens_cached_total || 0) + (prometheusMetrics.llamacpp_tokens_evaluated_total || 0))) * 100
    : 0
  const inferenceRequestTotal = prometheusMetrics.llamacpp_requests_total || 0
  const inferenceProcessing = prometheusMetrics.llamacpp_requests_processing || 0
  const inferenceDeferred = prometheusMetrics.llamacpp_requests_deferred || 0
  const inferenceSlotActive = prometheusMetrics.llamacpp_slot_active || 0
  const inferenceSlotIdle = prometheusMetrics.llamacpp_slot_idle || 0
  const inferenceTokensPredicted = prometheusMetrics.llamacpp_tokens_predicted_total || 0
  const inferenceTokensEvaluated = prometheusMetrics.llamacpp_tokens_evaluated_total || 0
  const hasInferenceTraffic = [
    inferenceRequestTotal,
    inferenceProcessing,
    inferenceDeferred,
    inferenceSlotActive,
    inferenceTokensPredicted,
    inferenceTokensEvaluated,
    tokensPerSecond,
    avgTimeToFirstToken,
  ].some((value) => value > 0)

  const handleRefresh = () => {
    fetchPrometheusMetrics()
    refetchMetrics()
  }

  const serviceHealth = serviceStatus?.health || 'stopped'
  const connectionLabel = connectionStatus === 'connected' ? 'Live' : 'Polling'
  const connectionStatusKey = connectionStatus === 'connected' ? 'connected' : 'polling'
  const activeEndpoints = serviceStatus?.endpoints
    ? Object.values(serviceStatus.endpoints).filter(Boolean).length
    : 0
  const totalEndpoints = serviceStatus?.endpoints ? Object.keys(serviceStatus.endpoints).length : 0
  const showEndpointStatus = activeEndpoints > 0
  const activeDeploymentContext = deploymentHandoff ?? (
    serviceStatus?.modelLoaded && serviceStatus.modelName
      ? {
          backend: 'llamacpp' as const,
          modelName: serviceStatus.modelName,
          endpoint: 'Backend proxy',
        }
      : null
  )
  const monitoringTargetLabel = activeDeploymentContext?.modelName || serviceStatus?.modelName || 'No model loaded'
  const monitoringBackendLabel = runtimeBackendLabel(activeDeploymentContext?.backend)
  const monitoringEndpointLabel = activeDeploymentContext?.endpoint || 'Backend proxy'
  const runtimeActions = activeDeploymentContext
    ? [
        {
          label: 'Chat',
          icon: <ChatIcon />,
          onClick: () => navigate(chatPath, { state: { deployment: activeDeploymentContext } }),
        },
        {
          label: 'Benchmark',
          icon: <SpeedIcon />,
          onClick: () =>
            navigate(benchmarkPath, {
              state: {
                deployment: activeDeploymentContext,
                model: modelInfoFromDeployment(activeDeploymentContext),
              },
            }),
        },
        {
          label: 'Deploy',
          icon: <DeployIcon />,
          onClick: () => navigate(deployPath, { state: { backend: activeDeploymentContext.backend } }),
        },
      ]
    : []
  const monitoringIconButtonSx = {
    width: 34,
    height: 34,
    borderRadius: '8px',
    border: '1px solid',
    borderColor: 'divider',
    bgcolor: 'action.hover',
    '&:hover': { bgcolor: alpha(accentColors.primary, 0.12) },
    '& .MuiSvgIcon-root': { fontSize: 18 },
  } as const
  const monitoringHeaderActions = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
      {['Inference', 'KV Cache', 'Tokens'].map((label, index) => (
        <Button
          key={label}
          size="small"
          variant={tabValue === index ? 'contained' : 'outlined'}
          onClick={() => setTabValue(index)}
          sx={{ height: 34, borderRadius: '8px', fontWeight: 800, px: 1.1 }}
        >
          {label}
        </Button>
      ))}
      <FormControlLabel
        control={
          <Switch
            checked={autoRefresh}
            onChange={(e) => setAutoRefresh(e.target.checked)}
            size="small"
          />
        }
        label="Auto refresh"
        sx={{ ml: 0, mr: 0.5, '& .MuiFormControlLabel-label': { fontSize: '0.8rem' } }}
      />
      {runtimeActions.map((action) => (
        <Tooltip key={action.label} title={action.label}>
          <IconButton
            size="small"
            aria-label={action.label}
            onClick={action.onClick}
            sx={monitoringIconButtonSx}
          >
            {action.icon}
          </IconButton>
        </Tooltip>
      ))}
      <Tooltip title="Refresh metrics">
        <IconButton
          size="small"
          aria-label="Refresh metrics"
          onClick={handleRefresh}
          sx={monitoringIconButtonSx}
        >
          <RefreshIcon sx={{ fontSize: 18 }} />
        </IconButton>
      </Tooltip>
    </Box>
  )
  const monitoringSummaryRow = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
      <StatusChip status={statusLoading ? 'checking' : serviceHealth} statusConfig={MONITORING_STATUS_CONFIG} height={22} sx={{ textTransform: 'capitalize' }} />
      <StatusChip status={connectionStatusKey} label={connectionLabel} statusConfig={MONITORING_STATUS_CONFIG} height={22} />
      <StatusChip status="target" label={monitoringTargetLabel} statusConfig={MONITORING_STATUS_CONFIG} height={22} />
      {showEndpointStatus && (
        <StatusChip status="info" label={`${activeEndpoints}/${totalEndpoints || 4} endpoints`} statusConfig={MONITORING_STATUS_CONFIG} height={22} />
      )}
      <Typography variant="caption" color="text.secondary" noWrap sx={{ minWidth: 0 }}>
        {monitoringBackendLabel} at {monitoringEndpointLabel} / {metrics ? `${metrics.cpuUsage.toFixed(1)}% CPU, ${metrics.memoryUsage.toFixed(1)}% memory` : metricsLoading ? 'loading resources' : 'resources unavailable'}
      </Typography>
    </Box>
  )

  return (
    <PageShell
      sx={{
        py: 1.25,
        display: 'flex',
        flexDirection: 'column',
        gap: 1.25,
        overflow: 'visible',
      }}
    >
      {error && (
        <Alert severity="warning" sx={{ mb: 1.25 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <SectionCard title="Resources" icon={<SpeedIcon />} accentColor={accentColors.success} action={monitoringHeaderActions}>
        {monitoringSummaryRow}
        <Grid container spacing={1}>
          <Grid item xs={6} sm={6} lg={3}>
            <MetricTile
              variant="horizontal"
              label="CPU"
              value={metrics ? `${metrics.cpuUsage.toFixed(1)}%` : '--'}
              detail={metricsLoading ? 'loading' : 'system load'}
              icon={<SpeedIcon />}
              color={accentColors.success}
            />
          </Grid>
          <Grid item xs={6} sm={6} lg={3}>
            <MetricTile
              variant="horizontal"
              label="Memory"
              value={metrics ? `${metrics.memoryUsage.toFixed(1)}%` : '--'}
              detail={metrics ? `${metrics.memoryUsed.toFixed(1)} / ${metrics.memoryTotal.toFixed(1)} GB` : metricsLoading ? 'loading' : 'unavailable'}
              icon={<MemoryIcon />}
              color={accentColors.info}
            />
          </Grid>
          <Grid item xs={6} sm={6} lg={3}>
            <MetricTile
              variant="horizontal"
              label="GPU"
              value={metrics ? `${metrics.gpuUsage.toFixed(1)}%` : '--'}
              detail={metrics?.gpuUsage ? 'utilization' : 'no active load'}
              icon={<DataUsageIcon />}
              color={accentColors.warning}
            />
          </Grid>
          <Grid item xs={6} sm={6} lg={3}>
            <MetricTile
              variant="horizontal"
              label="VRAM"
              value={metrics ? `${metrics.vramUsage.toFixed(1)}%` : '--'}
              detail={metrics?.vramTotal ? `${metrics.vramUsed.toFixed(1)} / ${metrics.vramTotal.toFixed(1)} GB` : 'none reported'}
              icon={<StorageIcon />}
              color={accentColors.primary}
            />
          </Grid>
        </Grid>
      </SectionCard>
      
      <Box data-monitoring-workspace sx={{ minWidth: 0 }}>
        {/* Inference Metrics Tab */}
        {tabValue === 0 && hasInferenceTraffic && (
          <SectionCard
            title="Inference"
            icon={<DataUsageIcon />}
            accentColor={accentColors.info}
            action={
              inferenceProcessing > 0 ? (
                <StatusChip
                  status="processing"
                  label={`${inferenceProcessing} active`}
                  statusConfig={MONITORING_STATUS_CONFIG}
                />
              ) : inferenceRequestTotal > 0 ? (
                <StatusChip
                  status="info"
                  label={`${inferenceRequestTotal.toLocaleString()} requests`}
                  statusConfig={MONITORING_STATUS_CONFIG}
                />
              ) : undefined
            }
          >
            <Grid container spacing={1}>
              {inferenceRequestTotal > 0 && (
                <Grid item xs={6} sm={6} lg={4} xl={3}>
                  <MetricTile
                    variant="horizontal"
                    label="Total requests"
                    value={inferenceRequestTotal.toLocaleString()}
                    icon={<DataUsageIcon />}
                    color={accentColors.info}
                  />
                </Grid>
              )}
              {inferenceProcessing > 0 && (
                <Grid item xs={6} sm={6} lg={4} xl={3}>
                  <MetricTile
                    variant="horizontal"
                    label="Processing"
                    value={inferenceProcessing}
                    detail="active queue"
                    icon={<SpeedIcon />}
                    color={accentColors.success}
                  />
                </Grid>
              )}
              {inferenceDeferred > 0 && (
                <Grid item xs={6} sm={6} lg={4} xl={3}>
                  <MetricTile
                    variant="horizontal"
                    label="Deferred"
                    value={inferenceDeferred}
                    detail="waiting queue"
                    icon={<TimerIcon />}
                    color={accentColors.warning}
                  />
                </Grid>
              )}
              {(inferenceSlotActive > 0 || inferenceSlotIdle > 0) && (
                <Grid item xs={6} sm={6} lg={4} xl={3}>
                  <MetricTile
                    variant="horizontal"
                    label="Slots"
                    value={`${inferenceSlotActive} active / ${inferenceSlotIdle} idle`}
                    icon={<CachedIcon />}
                    color={accentColors.primary}
                  />
                </Grid>
              )}
              {tokensPerSecond > 0 && (
                <Grid item xs={6} sm={6} lg={4} xl={3}>
                  <MetricTile
                    variant="horizontal"
                    label="Tokens/sec"
                    value={tokensPerSecond.toFixed(1)}
                    icon={<SpeedIcon />}
                    color={accentColors.success}
                  />
                </Grid>
              )}
              {avgTimeToFirstToken > 0 && (
                <Grid item xs={6} sm={6} lg={4} xl={3}>
                  <MetricTile
                    variant="horizontal"
                    label="Avg TTFT"
                    value={`${avgTimeToFirstToken.toFixed(0)}ms`}
                    icon={<TimerIcon />}
                    color={accentColors.info}
                  />
                </Grid>
              )}
              {inferenceTokensPredicted > 0 && (
                <Grid item xs={6} sm={6} lg={4} xl={3}>
                  <MetricTile
                    variant="horizontal"
                    label="Tokens predicted"
                    value={inferenceTokensPredicted.toLocaleString()}
                    icon={<TokenIcon />}
                    color={accentColors.primary}
                  />
                </Grid>
              )}
              {inferenceTokensEvaluated > 0 && (
                <Grid item xs={6} sm={6} lg={4} xl={3}>
                  <MetricTile
                    variant="horizontal"
                    label="Tokens evaluated"
                    value={inferenceTokensEvaluated.toLocaleString()}
                    icon={<DataUsageIcon />}
                    color={accentColors.warning}
                  />
                </Grid>
              )}
            </Grid>
          </SectionCard>
        )}

        {/* KV Cache Tab */}
        {tabValue === 1 && (
          <SectionCard
            title="KV Cache"
            icon={<CachedIcon />}
            accentColor={cacheHitRate > 50 ? accentColors.success : cacheHitRate > 20 ? accentColors.warning : accentColors.info}
            action={
              <StatusChip
                status={cacheHitRate > 50 ? 'good' : cacheHitRate > 20 ? 'warning' : 'unavailable'}
                icon={<CachedIcon sx={{ fontSize: 15 }} />}
                label={`${cacheHitRate.toFixed(1)}% hit`}
                statusConfig={MONITORING_STATUS_CONFIG}
              />
            }
          >
            <Grid container spacing={1}>
              <Grid item xs={12} md={6} xl={3}>
                <Box sx={{ ...monitoringPanelSx, p: 1.25, height: '100%', minHeight: 58 }}>
                  <Box sx={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 1, mb: 0.75 }}>
                    <Typography variant="caption" color="text.secondary" fontWeight={700}>
                      Cache utilization
                    </Typography>
                    <Typography variant="body2" fontWeight={800}>
                      {kvCacheUsage.toFixed(1)}%
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={kvCacheUsage}
                    sx={{
                      height: 7,
                      borderRadius: 4,
                      bgcolor: 'action.hover',
                      '& .MuiLinearProgress-bar': {
                        bgcolor: kvCacheUsage > 90 ? accentColors.rose : kvCacheUsage > 70 ? accentColors.warning : accentColors.primary,
                      },
                    }}
                  />
                </Box>
              </Grid>
              <Grid item xs={12} sm={6} xl={3}>
                <MetricTile
                  variant="horizontal"
                  label="Cache tokens"
                  value={`${(prometheusMetrics.llamacpp_kv_cache_tokens || 0).toLocaleString()} / ${(prometheusMetrics.llamacpp_kv_cache_max_tokens || 0).toLocaleString()}`}
                  icon={<CachedIcon />}
                  color={accentColors.primary}
                />
              </Grid>
              <Grid item xs={12} sm={6} xl={3}>
                <MetricTile
                  variant="horizontal"
                  label="Cached tokens"
                  value={(prometheusMetrics.llamacpp_tokens_cached_total || 0).toLocaleString()}
                  icon={<CachedIcon />}
                  color={accentColors.success}
                />
              </Grid>
              <Grid item xs={12} sm={6} xl={3}>
                <MetricTile
                  variant="horizontal"
                  label="Cache hit rate"
                  value={`${cacheHitRate.toFixed(1)}%`}
                  detail="cached / total"
                  icon={<TrendingIcon />}
                  color={cacheHitRate > 50 ? accentColors.success : accentColors.warning}
                />
              </Grid>
            </Grid>

            {(kvCacheUsage > 90 || cacheHitRate < 20 || (kvCacheUsage === 0 && prometheusMetrics.llamacpp_kv_cache_max_tokens === undefined)) && (
              <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
                {kvCacheUsage > 90 && (
                  <StatusChip status="warning" label="Reduce context size" statusConfig={MONITORING_STATUS_CONFIG} />
                )}
                {cacheHitRate < 20 && (
                  <StatusChip status="info" label="Low cache reuse" statusConfig={MONITORING_STATUS_CONFIG} />
                )}
                {kvCacheUsage === 0 && prometheusMetrics.llamacpp_kv_cache_max_tokens === undefined && (
                  <StatusChip status="unavailable" label="Metrics unavailable" statusConfig={MONITORING_STATUS_CONFIG} />
                )}
              </Box>
            )}
          </SectionCard>
        )}
      
        {/* Token Usage Tab */}
        {tabValue === 2 && (
          <SectionCard
            title="Token Usage"
            icon={<TokenIcon />}
            accentColor={accentColors.primary}
            action={
              <FormControl size="small" sx={{ minWidth: 118 }}>
                <InputLabel>Range</InputLabel>
                <Select
                  value={timeRange}
                  label="Range"
                  onChange={(e) => setTimeRange(e.target.value as any)}
                >
                  <MenuItem value="1h">1 hour</MenuItem>
                  <MenuItem value="24h">24 hours</MenuItem>
                  <MenuItem value="7d">7 days</MenuItem>
                  <MenuItem value="30d">30 days</MenuItem>
                </Select>
              </FormControl>
            }
          >
            <TokenUsageTracker timeRange={timeRange} compact />
          </SectionCard>
        )}
      </Box>
    </PageShell>
  )
}
