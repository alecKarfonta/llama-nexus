import React, { useState, useEffect, useCallback } from 'react'
import {
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Chip,
  Alert,
  CircularProgress,
  Divider,
  Paper,
  Tab,
  Tabs,
  LinearProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Switch,
  FormControlLabel,
} from '@mui/material'
import {
  Refresh as RefreshIcon,
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  Timer as TimerIcon,
  TrendingUp as TrendingIcon,
  Insights as InsightsIcon,
  DataUsage as DataUsageIcon,
  Cached as CachedIcon,
  Token as TokenIcon,
} from '@mui/icons-material'
import { useMetrics, useServiceStatus, useTokenUsage, useRealTimeMetrics } from '@/hooks/useMetrics'
import { TokenUsageTracker } from '@/components/monitoring/TokenUsageTracker'

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

const MetricCard: React.FC<{
  title: string
  value: string | number
  subtitle?: string
  icon: React.ReactNode
  color?: string
  trend?: number
}> = ({ title, value, subtitle, icon, color = 'primary.main', trend }) => (
  <Card variant="outlined">
    <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box>
          <Typography variant="caption" color="text.secondary">
            {title}
          </Typography>
          <Typography variant="h5" sx={{ color, fontWeight: 600 }}>
            {value}
          </Typography>
          {subtitle && (
            <Typography variant="caption" color="text.secondary">
              {subtitle}
            </Typography>
          )}
        </Box>
        <Box sx={{ color, opacity: 0.8 }}>
          {icon}
        </Box>
      </Box>
      {trend !== undefined && (
        <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
          <TrendingIcon sx={{ fontSize: 14, color: trend >= 0 ? 'success.main' : 'error.main', mr: 0.5 }} />
          <Typography variant="caption" color={trend >= 0 ? 'success.main' : 'error.main'}>
            {trend >= 0 ? '+' : ''}{trend.toFixed(1)}%
          </Typography>
        </Box>
      )}
    </CardContent>
  </Card>
)

export default function MonitoringPage() {
  const [tabValue, setTabValue] = useState(0)
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [prometheusMetrics, setPrometheusMetrics] = useState<PrometheusMetrics>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  const { data: metrics, isLoading: metricsLoading, refetch: refetchMetrics } = useMetrics()
  const { data: serviceStatus, isLoading: statusLoading } = useServiceStatus()
  const { connect, disconnect, connectionStatus } = useRealTimeMetrics()
  
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
    } finally {
      setLoading(false)
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

  const handleRefresh = () => {
    fetchPrometheusMetrics()
    refetchMetrics()
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
          <InsightsIcon /> System Monitoring
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                size="small"
              />
            }
            label="Auto Refresh"
          />
          <Chip
            label={connectionStatus === 'connected' ? 'Live' : 'Polling'}
            color={connectionStatus === 'connected' ? 'success' : 'default'}
            size="small"
          />
          <Tooltip title="Refresh Now">
            <IconButton onClick={handleRefresh}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      
      {error && (
        <Alert severity="warning" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      {/* System Resources */}
      <Typography variant="subtitle1" sx={{ mb: 1.5, fontWeight: 600 }}>
        System Resources
      </Typography>
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={6} md={3}>
          <MetricCard
            title="CPU Usage"
            value={metrics?.cpu?.percent?.toFixed(1) + '%' || 'N/A'}
            subtitle={`${metrics?.cpu?.count || 0} cores`}
            icon={<SpeedIcon sx={{ fontSize: 32 }} />}
            color="primary.main"
          />
        </Grid>
        <Grid item xs={6} md={3}>
          <MetricCard
            title="Memory Used"
            value={metrics?.memory?.used_mb 
              ? `${(metrics.memory.used_mb / 1024).toFixed(1)} GB`
              : 'N/A'}
            subtitle={metrics?.memory?.total_mb 
              ? `of ${(metrics.memory.total_mb / 1024).toFixed(1)} GB`
              : ''}
            icon={<MemoryIcon sx={{ fontSize: 32 }} />}
            color="info.main"
          />
        </Grid>
        <Grid item xs={6} md={3}>
          <MetricCard
            title="GPU Memory"
            value={metrics?.gpu?.memory_used_mb
              ? `${(metrics.gpu.memory_used_mb / 1024).toFixed(1)} GB`
              : 'N/A'}
            subtitle={metrics?.gpu?.memory_total_mb
              ? `of ${(metrics.gpu.memory_total_mb / 1024).toFixed(1)} GB`
              : ''}
            icon={<StorageIcon sx={{ fontSize: 32 }} />}
            color="success.main"
          />
        </Grid>
        <Grid item xs={6} md={3}>
          <MetricCard
            title="GPU Utilization"
            value={metrics?.gpu?.utilization?.toFixed(1) + '%' || 'N/A'}
            subtitle={metrics?.gpu?.name || 'No GPU'}
            icon={<DataUsageIcon sx={{ fontSize: 32 }} />}
            color="warning.main"
          />
        </Grid>
      </Grid>
      
      <Tabs value={tabValue} onChange={(_, v) => setTabValue(v)} sx={{ mb: 2 }}>
        <Tab label="Inference Metrics" />
        <Tab label="KV Cache" />
        <Tab label="Token Usage" />
      </Tabs>
      
      {/* Inference Metrics Tab */}
      {tabValue === 0 && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardHeader
                title="Request Statistics"
                titleTypographyProps={{ variant: 'subtitle1' }}
              />
              <Divider />
              <CardContent>
                <TableContainer>
                  <Table size="small">
                    <TableBody>
                      <TableRow>
                        <TableCell>Total Requests</TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" fontWeight={600}>
                            {prometheusMetrics.llamacpp_requests_total?.toLocaleString() || 0}
                          </Typography>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Processing</TableCell>
                        <TableCell align="right">
                          <Chip 
                            size="small" 
                            label={prometheusMetrics.llamacpp_requests_processing || 0}
                            color={prometheusMetrics.llamacpp_requests_processing ? 'success' : 'default'}
                          />
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Deferred</TableCell>
                        <TableCell align="right">
                          <Chip 
                            size="small" 
                            label={prometheusMetrics.llamacpp_requests_deferred || 0}
                            color={prometheusMetrics.llamacpp_requests_deferred ? 'warning' : 'default'}
                          />
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Active Slots</TableCell>
                        <TableCell align="right">
                          {prometheusMetrics.llamacpp_slot_active || 0}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Idle Slots</TableCell>
                        <TableCell align="right">
                          {prometheusMetrics.llamacpp_slot_idle || 0}
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardHeader
                title="Performance Metrics"
                titleTypographyProps={{ variant: 'subtitle1' }}
              />
              <Divider />
              <CardContent>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <MetricCard
                      title="Tokens/Second"
                      value={tokensPerSecond.toFixed(1)}
                      icon={<SpeedIcon />}
                      color="success.main"
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <MetricCard
                      title="Avg TTFT"
                      value={`${avgTimeToFirstToken.toFixed(0)}ms`}
                      icon={<TimerIcon />}
                      color="info.main"
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <MetricCard
                      title="Tokens Predicted"
                      value={(prometheusMetrics.llamacpp_tokens_predicted_total || 0).toLocaleString()}
                      icon={<TokenIcon />}
                      color="primary.main"
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <MetricCard
                      title="Tokens Evaluated"
                      value={(prometheusMetrics.llamacpp_tokens_evaluated_total || 0).toLocaleString()}
                      icon={<DataUsageIcon />}
                      color="warning.main"
                    />
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
      
      {/* KV Cache Tab */}
      {tabValue === 1 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Card variant="outlined">
              <CardHeader
                title="KV Cache Statistics"
                titleTypographyProps={{ variant: 'subtitle1' }}
                action={
                  <Chip 
                    icon={<CachedIcon />}
                    label={`${cacheHitRate.toFixed(1)}% Hit Rate`}
                    color={cacheHitRate > 50 ? 'success' : cacheHitRate > 20 ? 'warning' : 'default'}
                    size="small"
                  />
                }
              />
              <Divider />
              <CardContent>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={4}>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Cache Utilization
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <LinearProgress 
                          variant="determinate" 
                          value={kvCacheUsage} 
                          sx={{ flexGrow: 1, height: 8, borderRadius: 4 }}
                          color={kvCacheUsage > 90 ? 'error' : kvCacheUsage > 70 ? 'warning' : 'primary'}
                        />
                        <Typography variant="body2" fontWeight={600}>
                          {kvCacheUsage.toFixed(1)}%
                        </Typography>
                      </Box>
                    </Box>
                    <Box>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Cache Tokens
                      </Typography>
                      <Typography variant="h5" fontWeight={600}>
                        {(prometheusMetrics.llamacpp_kv_cache_tokens || 0).toLocaleString()}
                        <Typography component="span" variant="body2" color="text.secondary" sx={{ ml: 1 }}>
                          / {(prometheusMetrics.llamacpp_kv_cache_max_tokens || 0).toLocaleString()}
                        </Typography>
                      </Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <MetricCard
                      title="Cached Tokens"
                      value={(prometheusMetrics.llamacpp_tokens_cached_total || 0).toLocaleString()}
                      subtitle="Total tokens served from cache"
                      icon={<CachedIcon sx={{ fontSize: 32 }} />}
                      color="success.main"
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <MetricCard
                      title="Cache Hit Rate"
                      value={`${cacheHitRate.toFixed(1)}%`}
                      subtitle="Cached / (Cached + Evaluated)"
                      icon={<TrendingIcon sx={{ fontSize: 32 }} />}
                      color={cacheHitRate > 50 ? 'success.main' : 'warning.main'}
                    />
                  </Grid>
                </Grid>
                
                <Divider sx={{ my: 2 }} />
                
                <Typography variant="subtitle2" gutterBottom>
                  Cache Performance Tips
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {kvCacheUsage > 90 && (
                    <Chip size="small" label="Consider reducing context size" color="warning" />
                  )}
                  {cacheHitRate < 20 && (
                    <Chip size="small" label="Low cache hit rate - prompts vary significantly" color="info" />
                  )}
                  {kvCacheUsage === 0 && prometheusMetrics.llamacpp_kv_cache_max_tokens === undefined && (
                    <Chip size="small" label="LlamaCPP server not running or metrics unavailable" color="default" />
                  )}
                  {kvCacheUsage > 0 && cacheHitRate > 50 && (
                    <Chip size="small" label="Good cache performance" color="success" />
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
      
      {/* Token Usage Tab */}
      {tabValue === 2 && (
        <Box>
          <Box sx={{ mb: 2, display: 'flex', justifyContent: 'flex-end' }}>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Time Range</InputLabel>
              <Select
                value={timeRange}
                label="Time Range"
                onChange={(e) => setTimeRange(e.target.value as any)}
              >
                <MenuItem value="1h">Last Hour</MenuItem>
                <MenuItem value="24h">Last 24 Hours</MenuItem>
                <MenuItem value="7d">Last 7 Days</MenuItem>
                <MenuItem value="30d">Last 30 Days</MenuItem>
              </Select>
            </FormControl>
          </Box>
          <TokenUsageTracker timeRange={timeRange} />
        </Box>
      )}
    </Box>
  )
}
