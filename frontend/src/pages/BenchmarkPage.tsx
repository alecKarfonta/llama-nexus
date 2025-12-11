import React, { useState, useEffect, useCallback, useRef } from 'react'
import {
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  IconButton,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Tooltip,
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
  Slider,
  Switch,
  FormControlLabel,
  Collapse,
  Stack,
} from '@mui/material'
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
  Speed as SpeedIcon,
  Timer as TimerIcon,
  Memory as TokenIcon,
  TrendingUp as TrendingIcon,
  History as HistoryIcon,
  Add as AddIcon,
  Remove as RemoveIcon,
  Download as DownloadIcon,
  Compare as CompareIcon,
  Timeline as TimelineIcon,
  Storage as StorageIcon,
  ExpandMore as ExpandIcon,
  ExpandLess as CollapseIcon,
} from '@mui/icons-material'

// Types
interface RunMetrics {
  run_number: number
  prompt_tokens: number
  completion_tokens: number
  time_to_first_token_ms: number
  tokens_per_second: number
  total_time_ms: number
  timestamp: string
  response?: string
}

interface AggregateMetrics {
  tokens_per_second: { min: number; max: number; mean: number; median: number; p90: number; p99: number; stdev: number }
  time_to_first_token_ms: { min: number; max: number; mean: number; median: number; p90: number; p99: number; stdev: number }
  total_time_ms: { min: number; max: number; mean: number; median: number; p90: number; p99: number; stdev: number }
  total_tokens: number
  successful_runs: number
  failed_runs: number
}

interface BenchmarkResult {
  id: string
  name: string | null
  type: string
  endpoint: string | null
  model_name: string | null
  config: Record<string, any>
  metrics: AggregateMetrics | null
  runs: RunMetrics[]
  status: string
  created_at: string
  completed_at: string | null
}

interface ComparisonResult {
  name: string
  endpoint: string
  metrics: AggregateMetrics | null
  runs: RunMetrics[]
  errors: any[] | null
  status: string
}

interface ContextScalingResult {
  target_context_tokens: number
  actual_prompt_tokens?: number
  completion_tokens?: number
  time_to_first_token_ms?: number
  tokens_per_second?: number
  total_time_ms?: number
  status: string
  error?: string
}

interface ThroughputResult {
  concurrency: number
  total_requests: number
  successful_requests: number
  failed_requests: number
  total_time_seconds: number
  total_tokens_generated: number
  aggregate_throughput_tps: number
  avg_tokens_per_second: number
  avg_time_to_first_token_ms: number
  requests_per_second: number
}

interface EndpointConfig {
  name: string
  url: string
  api_key: string
}

// Tab panel component
function TabPanel({ children, value, index, ...other }: { children?: React.ReactNode; value: number; index: number; [key: string]: any }) {
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
    </div>
  )
}

// Metric card component
function MetricCard({ icon, value, label, unit, color = 'primary' }: { icon: React.ReactNode; value: string | number; label: string; unit?: string; color?: string }) {
  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent sx={{ textAlign: 'center', py: 1.5 }}>
        <Box sx={{ color: `${color}.main`, mb: 0.5 }}>{icon}</Box>
        <Typography variant="h5" sx={{ fontFamily: 'monospace' }}>
          {value}{unit && <Typography component="span" variant="body2" color="text.secondary">{unit}</Typography>}
        </Typography>
        <Typography variant="caption" color="text.secondary">{label}</Typography>
      </CardContent>
    </Card>
  )
}

export default function BenchmarkPage() {
  const [tabValue, setTabValue] = useState(0)
  
  // Speed test state
  const [speedTestPrompt, setSpeedTestPrompt] = useState('Explain the concept of neural networks in simple terms.')
  const [speedTestMaxTokens, setSpeedTestMaxTokens] = useState(256)
  const [speedTestNumRuns, setSpeedTestNumRuns] = useState(3)
  const [speedTestEndpoint, setSpeedTestEndpoint] = useState('')
  const [speedTestModelName, setSpeedTestModelName] = useState('')
  const [speedTestResult, setSpeedTestResult] = useState<BenchmarkResult | null>(null)
  const [speedTestRunning, setSpeedTestRunning] = useState(false)
  const [speedTestProgress, setSpeedTestProgress] = useState({ current: 0, total: 0 })
  
  // Comparison state
  const [comparePrompt, setComparePrompt] = useState('Write a short story about a robot learning to paint.')
  const [compareMaxTokens, setCompareMaxTokens] = useState(256)
  const [compareNumRuns, setCompareNumRuns] = useState(3)
  const [compareEndpoints, setCompareEndpoints] = useState<EndpointConfig[]>([
    { name: 'Local LLM', url: '', api_key: '' }
  ])
  const [compareResults, setCompareResults] = useState<ComparisonResult[]>([])
  const [compareRunning, setCompareRunning] = useState(false)
  
  // Context scaling state
  const [contextPrompt, setContextPrompt] = useState('This is a sample document that will be repeated to test how context length affects inference speed. ')
  const [contextSizes, setContextSizes] = useState([100, 500, 1000, 2000, 4000])
  const [contextMaxTokens, setContextMaxTokens] = useState(100)
  const [contextResults, setContextResults] = useState<ContextScalingResult[]>([])
  const [contextRunning, setContextRunning] = useState(false)
  
  // Throughput state
  const [throughputPrompt, setThroughputPrompt] = useState('What is 2+2?')
  const [throughputMaxTokens, setThroughputMaxTokens] = useState(64)
  const [throughputConcurrency, setThroughputConcurrency] = useState([1, 2, 4])
  const [throughputRequestsPerLevel, setThroughputRequestsPerLevel] = useState(3)
  const [throughputResults, setThroughputResults] = useState<ThroughputResult[]>([])
  const [throughputRunning, setThroughputRunning] = useState(false)
  
  // History state
  const [historyResults, setHistoryResults] = useState<BenchmarkResult[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyTotal, setHistoryTotal] = useState(0)
  
  // General state
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState<{ total_benchmarks: number; by_type: Record<string, number>; average_tps_recent: number } | null>(null)
  
  const abortControllerRef = useRef<AbortController | null>(null)

  // Load stats and history on mount
  useEffect(() => {
    loadStats()
    loadHistory()
  }, [])

  const loadStats = async () => {
    try {
      const response = await fetch('/api/v1/benchmark/stats')
      if (response.ok) {
        setStats(await response.json())
      }
    } catch (err) {
      console.error('Failed to load stats:', err)
    }
  }

  const loadHistory = async () => {
    setHistoryLoading(true)
    try {
      const response = await fetch('/api/v1/benchmark/results?limit=50')
      if (response.ok) {
        const data = await response.json()
        setHistoryResults(data.results || [])
        setHistoryTotal(data.total || 0)
      }
    } catch (err) {
      console.error('Failed to load history:', err)
    } finally {
      setHistoryLoading(false)
    }
  }

  // Speed Test
  const runSpeedTest = async () => {
    setError(null)
    setSpeedTestRunning(true)
    setSpeedTestResult(null)
    setSpeedTestProgress({ current: 0, total: speedTestNumRuns })
    
    abortControllerRef.current = new AbortController()
    
    try {
      const response = await fetch('/api/v1/benchmark/speed-test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: speedTestPrompt,
          max_tokens: speedTestMaxTokens,
          num_runs: speedTestNumRuns,
          endpoint: speedTestEndpoint || null,
          model_name: speedTestModelName || null,
          save_result: true
        }),
        signal: abortControllerRef.current.signal
      })
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      
      const reader = response.body?.getReader()
      if (!reader) throw new Error('No response body')
      
      const decoder = new TextDecoder()
      let buffer = ''
      
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              
              if (data.event === 'run_start') {
                setSpeedTestProgress({ current: data.run, total: data.total })
              } else if (data.event === 'run_complete') {
                setSpeedTestProgress({ current: data.run, total: speedTestNumRuns })
              } else if (data.event === 'complete') {
                setSpeedTestResult(data.result)
              } else if (data.event === 'run_error') {
                console.error('Run error:', data.error)
              }
            } catch (e) {
              // Ignore parse errors
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setError(err.message || 'Speed test failed')
      }
    } finally {
      setSpeedTestRunning(false)
      loadStats()
      loadHistory()
    }
  }

  // Compare Endpoints
  const runComparison = async () => {
    setError(null)
    setCompareRunning(true)
    setCompareResults([])
    
    try {
      const response = await fetch('/api/v1/benchmark/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: comparePrompt,
          max_tokens: compareMaxTokens,
          num_runs: compareNumRuns,
          endpoints: compareEndpoints.map(ep => ({
            name: ep.name,
            url: ep.url || null,
            api_key: ep.api_key || null
          }))
        })
      })
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      
      const data = await response.json()
      setCompareResults(data.results || [])
    } catch (err: any) {
      setError(err.message || 'Comparison failed')
    } finally {
      setCompareRunning(false)
      loadStats()
      loadHistory()
    }
  }

  // Context Scaling Test
  const runContextScaling = async () => {
    setError(null)
    setContextRunning(true)
    setContextResults([])
    
    try {
      const response = await fetch('/api/v1/benchmark/context-scaling', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_prompt: contextPrompt,
          context_sizes: contextSizes,
          max_tokens: contextMaxTokens
        })
      })
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      
      const data = await response.json()
      setContextResults(data.results || [])
    } catch (err: any) {
      setError(err.message || 'Context scaling test failed')
    } finally {
      setContextRunning(false)
      loadStats()
      loadHistory()
    }
  }

  // Throughput Test
  const runThroughput = async () => {
    setError(null)
    setThroughputRunning(true)
    setThroughputResults([])
    
    try {
      const response = await fetch('/api/v1/benchmark/throughput', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: throughputPrompt,
          max_tokens: throughputMaxTokens,
          concurrent_requests: throughputConcurrency,
          requests_per_level: throughputRequestsPerLevel
        })
      })
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      
      const data = await response.json()
      setThroughputResults(data.results || [])
    } catch (err: any) {
      setError(err.message || 'Throughput test failed')
    } finally {
      setThroughputRunning(false)
      loadStats()
      loadHistory()
    }
  }

  // Delete result
  const deleteResult = async (id: string) => {
    try {
      await fetch(`/api/v1/benchmark/results/${id}`, { method: 'DELETE' })
      loadHistory()
      loadStats()
    } catch (err) {
      console.error('Failed to delete result:', err)
    }
  }

  // Export result
  const exportResult = async (id: string, format: 'json' | 'csv') => {
    window.open(`/api/v1/benchmark/export/${id}?format=${format}`, '_blank')
  }

  // Add/remove endpoints for comparison
  const addEndpoint = () => {
    setCompareEndpoints([...compareEndpoints, { name: `Endpoint ${compareEndpoints.length + 1}`, url: '', api_key: '' }])
  }
  
  const removeEndpoint = (index: number) => {
    setCompareEndpoints(compareEndpoints.filter((_, i) => i !== index))
  }
  
  const updateEndpoint = (index: number, field: keyof EndpointConfig, value: string) => {
    const updated = [...compareEndpoints]
    updated[index] = { ...updated[index], [field]: value }
    setCompareEndpoints(updated)
  }

  const formatNumber = (num: number | undefined, decimals: number = 1) => {
    if (num === undefined || num === null) return '-'
    return num.toFixed(decimals)
  }

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'speed_test': return 'primary'
      case 'comparison': return 'secondary'
      case 'context_scaling': return 'info'
      case 'throughput': return 'warning'
      default: return 'default'
    }
  }

  const getTypeLabel = (type: string) => {
    switch (type) {
      case 'speed_test': return 'Speed Test'
      case 'comparison': return 'Comparison'
      case 'context_scaling': return 'Context Scaling'
      case 'throughput': return 'Throughput'
      default: return type
    }
  }

  return (
    <Box sx={{ p: 3, maxWidth: 1400, mx: 'auto' }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 600 }}>
            LLM Benchmark Suite
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Measure and compare language model inference performance
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title="Refresh">
            <IconButton onClick={() => { loadStats(); loadHistory(); }}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Stats Overview */}
      {stats && (
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={6} sm={3}>
            <MetricCard
              icon={<HistoryIcon />}
              value={stats.total_benchmarks}
              label="Total Benchmarks"
              color="primary"
            />
          </Grid>
          <Grid item xs={6} sm={3}>
            <MetricCard
              icon={<SpeedIcon />}
              value={formatNumber(stats.average_tps_recent)}
              label="Avg TPS (Recent)"
              unit=" t/s"
              color="success"
            />
          </Grid>
          <Grid item xs={6} sm={3}>
            <MetricCard
              icon={<TimelineIcon />}
              value={stats.by_type?.speed_test || 0}
              label="Speed Tests"
              color="info"
            />
          </Grid>
          <Grid item xs={6} sm={3}>
            <MetricCard
              icon={<CompareIcon />}
              value={stats.by_type?.comparison || 0}
              label="Comparisons"
              color="secondary"
            />
          </Grid>
        </Grid>
      )}

      {/* Main Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={tabValue}
          onChange={(_, v) => setTabValue(v)}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab icon={<SpeedIcon />} label="Speed Test" iconPosition="start" />
          <Tab icon={<CompareIcon />} label="Compare" iconPosition="start" />
          <Tab icon={<TimelineIcon />} label="Context Scaling" iconPosition="start" />
          <Tab icon={<StorageIcon />} label="Throughput" iconPosition="start" />
          <Tab icon={<HistoryIcon />} label="History" iconPosition="start" />
        </Tabs>

        <Box sx={{ p: 3 }}>
          {/* Speed Test Tab */}
          <TabPanel value={tabValue} index={0}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={5}>
                <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
                  Quick Speed Test
                </Typography>
                
                <TextField
                  label="Test Prompt"
                  multiline
                  rows={4}
                  value={speedTestPrompt}
                  onChange={(e) => setSpeedTestPrompt(e.target.value)}
                  fullWidth
                  sx={{ mb: 2 }}
                />
                
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <TextField
                      label="Max Tokens"
                      type="number"
                      value={speedTestMaxTokens}
                      onChange={(e) => setSpeedTestMaxTokens(parseInt(e.target.value) || 256)}
                      fullWidth
                      size="small"
                      inputProps={{ min: 1, max: 4096 }}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      label="Number of Runs"
                      type="number"
                      value={speedTestNumRuns}
                      onChange={(e) => setSpeedTestNumRuns(parseInt(e.target.value) || 1)}
                      fullWidth
                      size="small"
                      inputProps={{ min: 1, max: 10 }}
                    />
                  </Grid>
                </Grid>
                
                <TextField
                  label="Custom Endpoint (optional)"
                  value={speedTestEndpoint}
                  onChange={(e) => setSpeedTestEndpoint(e.target.value)}
                  fullWidth
                  size="small"
                  placeholder="Leave empty for local LLM"
                  sx={{ mb: 2 }}
                />
                
                <TextField
                  label="Model Name (optional)"
                  value={speedTestModelName}
                  onChange={(e) => setSpeedTestModelName(e.target.value)}
                  fullWidth
                  size="small"
                  placeholder="For labeling results"
                  sx={{ mb: 2 }}
                />
                
                <Button
                  variant="contained"
                  fullWidth
                  startIcon={speedTestRunning ? <CircularProgress size={20} color="inherit" /> : <StartIcon />}
                  onClick={runSpeedTest}
                  disabled={speedTestRunning || !speedTestPrompt.trim()}
                  size="large"
                >
                  {speedTestRunning ? `Running ${speedTestProgress.current}/${speedTestProgress.total}...` : 'Run Speed Test'}
                </Button>
                
                {speedTestRunning && (
                  <LinearProgress sx={{ mt: 2 }} variant="determinate" value={(speedTestProgress.current / speedTestProgress.total) * 100} />
                )}
              </Grid>
              
              <Grid item xs={12} md={7}>
                {speedTestResult ? (
                  <Box>
                    <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
                      Results
                    </Typography>
                    
                    {/* Key Metrics */}
                    <Grid container spacing={2} sx={{ mb: 3 }}>
                      <Grid item xs={6} sm={3}>
                        <MetricCard
                          icon={<SpeedIcon sx={{ fontSize: 28 }} />}
                          value={formatNumber(speedTestResult.metrics?.tokens_per_second.mean)}
                          label="Tokens/Second"
                          unit=" t/s"
                          color="primary"
                        />
                      </Grid>
                      <Grid item xs={6} sm={3}>
                        <MetricCard
                          icon={<TimerIcon sx={{ fontSize: 28 }} />}
                          value={formatNumber(speedTestResult.metrics?.time_to_first_token_ms.mean, 0)}
                          label="TTFT"
                          unit=" ms"
                          color="info"
                        />
                      </Grid>
                      <Grid item xs={6} sm={3}>
                        <MetricCard
                          icon={<TokenIcon sx={{ fontSize: 28 }} />}
                          value={speedTestResult.metrics?.total_tokens || 0}
                          label="Total Tokens"
                          color="success"
                        />
                      </Grid>
                      <Grid item xs={6} sm={3}>
                        <MetricCard
                          icon={<TrendingIcon sx={{ fontSize: 28 }} />}
                          value={formatNumber((speedTestResult.metrics?.total_time_ms.mean || 0) / 1000)}
                          label="Avg Time"
                          unit=" s"
                          color="warning"
                        />
                      </Grid>
                    </Grid>
                    
                    {/* Detailed Stats */}
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>Detailed Statistics</Typography>
                    <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Metric</TableCell>
                            <TableCell align="right">Min</TableCell>
                            <TableCell align="right">Max</TableCell>
                            <TableCell align="right">Mean</TableCell>
                            <TableCell align="right">Median</TableCell>
                            <TableCell align="right">P90</TableCell>
                            <TableCell align="right">P99</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          <TableRow>
                            <TableCell>Tokens/Second</TableCell>
                            <TableCell align="right">{formatNumber(speedTestResult.metrics?.tokens_per_second.min)}</TableCell>
                            <TableCell align="right">{formatNumber(speedTestResult.metrics?.tokens_per_second.max)}</TableCell>
                            <TableCell align="right" sx={{ fontWeight: 600 }}>{formatNumber(speedTestResult.metrics?.tokens_per_second.mean)}</TableCell>
                            <TableCell align="right">{formatNumber(speedTestResult.metrics?.tokens_per_second.median)}</TableCell>
                            <TableCell align="right">{formatNumber(speedTestResult.metrics?.tokens_per_second.p90)}</TableCell>
                            <TableCell align="right">{formatNumber(speedTestResult.metrics?.tokens_per_second.p99)}</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>TTFT (ms)</TableCell>
                            <TableCell align="right">{formatNumber(speedTestResult.metrics?.time_to_first_token_ms.min, 0)}</TableCell>
                            <TableCell align="right">{formatNumber(speedTestResult.metrics?.time_to_first_token_ms.max, 0)}</TableCell>
                            <TableCell align="right" sx={{ fontWeight: 600 }}>{formatNumber(speedTestResult.metrics?.time_to_first_token_ms.mean, 0)}</TableCell>
                            <TableCell align="right">{formatNumber(speedTestResult.metrics?.time_to_first_token_ms.median, 0)}</TableCell>
                            <TableCell align="right">{formatNumber(speedTestResult.metrics?.time_to_first_token_ms.p90, 0)}</TableCell>
                            <TableCell align="right">{formatNumber(speedTestResult.metrics?.time_to_first_token_ms.p99, 0)}</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>Total Time (ms)</TableCell>
                            <TableCell align="right">{formatNumber(speedTestResult.metrics?.total_time_ms.min, 0)}</TableCell>
                            <TableCell align="right">{formatNumber(speedTestResult.metrics?.total_time_ms.max, 0)}</TableCell>
                            <TableCell align="right" sx={{ fontWeight: 600 }}>{formatNumber(speedTestResult.metrics?.total_time_ms.mean, 0)}</TableCell>
                            <TableCell align="right">{formatNumber(speedTestResult.metrics?.total_time_ms.median, 0)}</TableCell>
                            <TableCell align="right">{formatNumber(speedTestResult.metrics?.total_time_ms.p90, 0)}</TableCell>
                            <TableCell align="right">{formatNumber(speedTestResult.metrics?.total_time_ms.p99, 0)}</TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </TableContainer>
                    
                    {/* Individual Runs */}
                    {speedTestResult.runs && speedTestResult.runs.length > 0 && (
                      <>
                        <Typography variant="subtitle2" sx={{ mb: 1 }}>Individual Runs</Typography>
                        <TableContainer component={Paper} variant="outlined">
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>Run</TableCell>
                                <TableCell align="right">Prompt</TableCell>
                                <TableCell align="right">Output</TableCell>
                                <TableCell align="right">TTFT</TableCell>
                                <TableCell align="right">TPS</TableCell>
                                <TableCell align="right">Total</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {speedTestResult.runs.map((run) => (
                                <TableRow key={run.run_number}>
                                  <TableCell>{run.run_number}</TableCell>
                                  <TableCell align="right">{run.prompt_tokens}</TableCell>
                                  <TableCell align="right">{run.completion_tokens}</TableCell>
                                  <TableCell align="right">{formatNumber(run.time_to_first_token_ms, 0)} ms</TableCell>
                                  <TableCell align="right" sx={{ fontWeight: 600 }}>{formatNumber(run.tokens_per_second)} t/s</TableCell>
                                  <TableCell align="right">{formatNumber(run.total_time_ms / 1000)}s</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </>
                    )}
                  </Box>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 8 }}>
                    <SpeedIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                    <Typography color="text.secondary">
                      Run a speed test to measure inference performance
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      Enter a prompt and click "Run Speed Test"
                    </Typography>
                  </Box>
                )}
              </Grid>
            </Grid>
          </TabPanel>

          {/* Compare Tab */}
          <TabPanel value={tabValue} index={1}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={5}>
                <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
                  Compare Endpoints
                </Typography>
                
                <TextField
                  label="Test Prompt"
                  multiline
                  rows={3}
                  value={comparePrompt}
                  onChange={(e) => setComparePrompt(e.target.value)}
                  fullWidth
                  sx={{ mb: 2 }}
                />
                
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <TextField
                      label="Max Tokens"
                      type="number"
                      value={compareMaxTokens}
                      onChange={(e) => setCompareMaxTokens(parseInt(e.target.value) || 256)}
                      fullWidth
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      label="Runs per Endpoint"
                      type="number"
                      value={compareNumRuns}
                      onChange={(e) => setCompareNumRuns(parseInt(e.target.value) || 3)}
                      fullWidth
                      size="small"
                    />
                  </Grid>
                </Grid>
                
                <Divider sx={{ my: 2 }} />
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="subtitle2">Endpoints</Typography>
                  <Button size="small" startIcon={<AddIcon />} onClick={addEndpoint}>
                    Add Endpoint
                  </Button>
                </Box>
                
                {compareEndpoints.map((ep, index) => (
                  <Paper key={index} variant="outlined" sx={{ p: 2, mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <TextField
                        label="Name"
                        value={ep.name}
                        onChange={(e) => updateEndpoint(index, 'name', e.target.value)}
                        size="small"
                        sx={{ flex: 1, mr: 1 }}
                      />
                      {compareEndpoints.length > 1 && (
                        <IconButton size="small" onClick={() => removeEndpoint(index)} color="error">
                          <RemoveIcon />
                        </IconButton>
                      )}
                    </Box>
                    <TextField
                      label="URL (empty for local)"
                      value={ep.url}
                      onChange={(e) => updateEndpoint(index, 'url', e.target.value)}
                      fullWidth
                      size="small"
                      placeholder="http://api.openai.com/v1"
                      sx={{ mb: 1 }}
                    />
                    <TextField
                      label="API Key (optional)"
                      value={ep.api_key}
                      onChange={(e) => updateEndpoint(index, 'api_key', e.target.value)}
                      fullWidth
                      size="small"
                      type="password"
                    />
                  </Paper>
                ))}
                
                <Button
                  variant="contained"
                  fullWidth
                  startIcon={compareRunning ? <CircularProgress size={20} color="inherit" /> : <CompareIcon />}
                  onClick={runComparison}
                  disabled={compareRunning || !comparePrompt.trim() || compareEndpoints.length === 0}
                  size="large"
                >
                  {compareRunning ? 'Comparing...' : 'Run Comparison'}
                </Button>
              </Grid>
              
              <Grid item xs={12} md={7}>
                {compareResults.length > 0 ? (
                  <Box>
                    <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
                      Comparison Results
                    </Typography>
                    
                    <TableContainer component={Paper} variant="outlined">
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>Endpoint</TableCell>
                            <TableCell align="right">TPS (Mean)</TableCell>
                            <TableCell align="right">TTFT (Mean)</TableCell>
                            <TableCell align="right">Total Time</TableCell>
                            <TableCell align="center">Status</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {compareResults.map((result, index) => (
                            <TableRow key={index}>
                              <TableCell>
                                <Typography variant="body2" sx={{ fontWeight: 600 }}>{result.name}</Typography>
                                <Typography variant="caption" color="text.secondary">{result.endpoint}</Typography>
                              </TableCell>
                              <TableCell align="right" sx={{ fontFamily: 'monospace', fontWeight: 600, color: 'primary.main' }}>
                                {result.metrics ? formatNumber(result.metrics.tokens_per_second.mean) : '-'} t/s
                              </TableCell>
                              <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                                {result.metrics ? formatNumber(result.metrics.time_to_first_token_ms.mean, 0) : '-'} ms
                              </TableCell>
                              <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                                {result.metrics ? formatNumber(result.metrics.total_time_ms.mean / 1000) : '-'} s
                              </TableCell>
                              <TableCell align="center">
                                <Chip
                                  size="small"
                                  label={result.status}
                                  color={result.status === 'completed' ? 'success' : 'error'}
                                />
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                    
                    {/* Visual comparison bar chart */}
                    <Box sx={{ mt: 3 }}>
                      <Typography variant="subtitle2" sx={{ mb: 2 }}>Tokens Per Second Comparison</Typography>
                      {compareResults.filter(r => r.metrics).map((result, index) => {
                        const maxTps = Math.max(...compareResults.filter(r => r.metrics).map(r => r.metrics!.tokens_per_second.mean))
                        const percentage = maxTps > 0 ? (result.metrics!.tokens_per_second.mean / maxTps) * 100 : 0
                        return (
                          <Box key={index} sx={{ mb: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                              <Typography variant="body2">{result.name}</Typography>
                              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                {formatNumber(result.metrics!.tokens_per_second.mean)} t/s
                              </Typography>
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={percentage}
                              sx={{ height: 20, borderRadius: 1 }}
                            />
                          </Box>
                        )
                      })}
                    </Box>
                  </Box>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 8 }}>
                    <CompareIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                    <Typography color="text.secondary">
                      Compare performance across multiple endpoints
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      Add endpoints and run a comparison
                    </Typography>
                  </Box>
                )}
              </Grid>
            </Grid>
          </TabPanel>

          {/* Context Scaling Tab */}
          <TabPanel value={tabValue} index={2}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={5}>
                <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
                  Context Length Scaling
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Test how inference speed changes with different context lengths
                </Typography>
                
                <TextField
                  label="Base Text (will be repeated)"
                  multiline
                  rows={3}
                  value={contextPrompt}
                  onChange={(e) => setContextPrompt(e.target.value)}
                  fullWidth
                  sx={{ mb: 2 }}
                />
                
                <TextField
                  label="Max Output Tokens"
                  type="number"
                  value={contextMaxTokens}
                  onChange={(e) => setContextMaxTokens(parseInt(e.target.value) || 100)}
                  fullWidth
                  size="small"
                  sx={{ mb: 2 }}
                />
                
                <Typography variant="subtitle2" sx={{ mb: 1 }}>Context Sizes to Test (tokens)</Typography>
                <TextField
                  value={contextSizes.join(', ')}
                  onChange={(e) => {
                    const sizes = e.target.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0)
                    if (sizes.length > 0) setContextSizes(sizes)
                  }}
                  fullWidth
                  size="small"
                  placeholder="100, 500, 1000, 2000, 4000"
                  sx={{ mb: 2 }}
                  helperText="Comma-separated list of token counts"
                />
                
                <Button
                  variant="contained"
                  fullWidth
                  startIcon={contextRunning ? <CircularProgress size={20} color="inherit" /> : <TimelineIcon />}
                  onClick={runContextScaling}
                  disabled={contextRunning || !contextPrompt.trim()}
                  size="large"
                >
                  {contextRunning ? 'Testing...' : 'Run Scaling Test'}
                </Button>
              </Grid>
              
              <Grid item xs={12} md={7}>
                {contextResults.length > 0 ? (
                  <Box>
                    <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
                      Scaling Results
                    </Typography>
                    
                    <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>Context (tokens)</TableCell>
                            <TableCell align="right">TPS</TableCell>
                            <TableCell align="right">TTFT</TableCell>
                            <TableCell align="right">Total Time</TableCell>
                            <TableCell align="center">Status</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {contextResults.map((result, index) => (
                            <TableRow key={index}>
                              <TableCell sx={{ fontWeight: 600 }}>{result.target_context_tokens}</TableCell>
                              <TableCell align="right" sx={{ fontFamily: 'monospace', color: 'primary.main' }}>
                                {result.tokens_per_second ? formatNumber(result.tokens_per_second) + ' t/s' : '-'}
                              </TableCell>
                              <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                                {result.time_to_first_token_ms ? formatNumber(result.time_to_first_token_ms, 0) + ' ms' : '-'}
                              </TableCell>
                              <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                                {result.total_time_ms ? formatNumber(result.total_time_ms / 1000) + ' s' : '-'}
                              </TableCell>
                              <TableCell align="center">
                                <Chip
                                  size="small"
                                  label={result.status}
                                  color={result.status === 'success' ? 'success' : 'error'}
                                />
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                    
                    {/* Visual scaling chart */}
                    <Typography variant="subtitle2" sx={{ mb: 2 }}>TPS vs Context Length</Typography>
                    {contextResults.filter(r => r.tokens_per_second).map((result, index) => {
                      const maxTps = Math.max(...contextResults.filter(r => r.tokens_per_second).map(r => r.tokens_per_second!))
                      const percentage = maxTps > 0 ? (result.tokens_per_second! / maxTps) * 100 : 0
                      return (
                        <Box key={index} sx={{ mb: 2 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                            <Typography variant="body2">{result.target_context_tokens} tokens</Typography>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              {formatNumber(result.tokens_per_second)} t/s
                            </Typography>
                          </Box>
                          <LinearProgress
                            variant="determinate"
                            value={percentage}
                            sx={{ height: 16, borderRadius: 1 }}
                            color={percentage > 80 ? 'success' : percentage > 50 ? 'warning' : 'error'}
                          />
                        </Box>
                      )
                    })}
                  </Box>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 8 }}>
                    <TimelineIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                    <Typography color="text.secondary">
                      Test how context length affects inference speed
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      See performance degradation patterns
                    </Typography>
                  </Box>
                )}
              </Grid>
            </Grid>
          </TabPanel>

          {/* Throughput Tab */}
          <TabPanel value={tabValue} index={3}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={5}>
                <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
                  Throughput / Stress Test
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Test system performance under concurrent load
                </Typography>
                
                <TextField
                  label="Test Prompt"
                  multiline
                  rows={2}
                  value={throughputPrompt}
                  onChange={(e) => setThroughputPrompt(e.target.value)}
                  fullWidth
                  sx={{ mb: 2 }}
                />
                
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <TextField
                      label="Max Tokens"
                      type="number"
                      value={throughputMaxTokens}
                      onChange={(e) => setThroughputMaxTokens(parseInt(e.target.value) || 64)}
                      fullWidth
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      label="Requests/Level"
                      type="number"
                      value={throughputRequestsPerLevel}
                      onChange={(e) => setThroughputRequestsPerLevel(parseInt(e.target.value) || 3)}
                      fullWidth
                      size="small"
                    />
                  </Grid>
                </Grid>
                
                <Typography variant="subtitle2" sx={{ mb: 1 }}>Concurrency Levels</Typography>
                <TextField
                  value={throughputConcurrency.join(', ')}
                  onChange={(e) => {
                    const levels = e.target.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0)
                    if (levels.length > 0) setThroughputConcurrency(levels)
                  }}
                  fullWidth
                  size="small"
                  placeholder="1, 2, 4, 8"
                  sx={{ mb: 2 }}
                  helperText="Comma-separated concurrent request counts"
                />
                
                <Button
                  variant="contained"
                  fullWidth
                  startIcon={throughputRunning ? <CircularProgress size={20} color="inherit" /> : <StorageIcon />}
                  onClick={runThroughput}
                  disabled={throughputRunning || !throughputPrompt.trim()}
                  size="large"
                >
                  {throughputRunning ? 'Testing...' : 'Run Throughput Test'}
                </Button>
              </Grid>
              
              <Grid item xs={12} md={7}>
                {throughputResults.length > 0 ? (
                  <Box>
                    <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
                      Throughput Results
                    </Typography>
                    
                    <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>Concurrency</TableCell>
                            <TableCell align="right">Requests</TableCell>
                            <TableCell align="right">Throughput</TableCell>
                            <TableCell align="right">Avg TPS</TableCell>
                            <TableCell align="right">Req/s</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {throughputResults.map((result, index) => (
                            <TableRow key={index}>
                              <TableCell sx={{ fontWeight: 600 }}>{result.concurrency}x</TableCell>
                              <TableCell align="right">
                                {result.successful_requests}/{result.total_requests}
                              </TableCell>
                              <TableCell align="right" sx={{ fontFamily: 'monospace', color: 'primary.main', fontWeight: 600 }}>
                                {formatNumber(result.aggregate_throughput_tps)} t/s
                              </TableCell>
                              <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                                {formatNumber(result.avg_tokens_per_second)} t/s
                              </TableCell>
                              <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                                {formatNumber(result.requests_per_second)}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                    
                    {/* Visual throughput chart */}
                    <Typography variant="subtitle2" sx={{ mb: 2 }}>Aggregate Throughput by Concurrency</Typography>
                    {throughputResults.map((result, index) => {
                      const maxThroughput = Math.max(...throughputResults.map(r => r.aggregate_throughput_tps))
                      const percentage = maxThroughput > 0 ? (result.aggregate_throughput_tps / maxThroughput) * 100 : 0
                      return (
                        <Box key={index} sx={{ mb: 2 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                            <Typography variant="body2">{result.concurrency}x concurrent</Typography>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              {formatNumber(result.aggregate_throughput_tps)} t/s
                            </Typography>
                          </Box>
                          <LinearProgress
                            variant="determinate"
                            value={percentage}
                            sx={{ height: 16, borderRadius: 1 }}
                          />
                        </Box>
                      )
                    })}
                  </Box>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 8 }}>
                    <StorageIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                    <Typography color="text.secondary">
                      Test system throughput under load
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      See how concurrency affects performance
                    </Typography>
                  </Box>
                )}
              </Grid>
            </Grid>
          </TabPanel>

          {/* History Tab */}
          <TabPanel value={tabValue} index={4}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                Benchmark History ({historyTotal} results)
              </Typography>
              <Button
                size="small"
                color="error"
                onClick={async () => {
                  await fetch('/api/v1/benchmark/results', { method: 'DELETE' })
                  loadHistory()
                  loadStats()
                }}
              >
                Clear All
              </Button>
            </Box>
            
            {historyLoading ? (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <CircularProgress />
              </Box>
            ) : historyResults.length === 0 ? (
              <Box sx={{ textAlign: 'center', py: 8 }}>
                <HistoryIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                <Typography color="text.secondary">
                  No benchmark history yet
                </Typography>
              </Box>
            ) : (
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Name</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell align="right">TPS</TableCell>
                      <TableCell align="right">TTFT</TableCell>
                      <TableCell>Date</TableCell>
                      <TableCell align="center">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {historyResults.map((result) => (
                      <TableRow key={result.id}>
                        <TableCell>
                          <Typography variant="body2" sx={{ fontWeight: 500 }}>
                            {result.name || result.id}
                          </Typography>
                          {result.model_name && (
                            <Typography variant="caption" color="text.secondary">
                              {result.model_name}
                            </Typography>
                          )}
                        </TableCell>
                        <TableCell>
                          <Chip
                            size="small"
                            label={getTypeLabel(result.type)}
                            color={getTypeColor(result.type) as any}
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                          {result.metrics?.tokens_per_second?.mean
                            ? formatNumber(result.metrics.tokens_per_second.mean) + ' t/s'
                            : '-'}
                        </TableCell>
                        <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                          {result.metrics?.time_to_first_token_ms?.mean
                            ? formatNumber(result.metrics.time_to_first_token_ms.mean, 0) + ' ms'
                            : '-'}
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {new Date(result.created_at).toLocaleDateString()}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {new Date(result.created_at).toLocaleTimeString()}
                          </Typography>
                        </TableCell>
                        <TableCell align="center">
                          <Tooltip title="Export JSON">
                            <IconButton size="small" onClick={() => exportResult(result.id, 'json')}>
                              <DownloadIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton size="small" color="error" onClick={() => deleteResult(result.id)}>
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </TabPanel>
        </Box>
      </Paper>
    </Box>
  )
}
