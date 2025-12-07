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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
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
  Settings as SettingsIcon,
} from '@mui/icons-material'
import { apiService } from '@/services/api'

interface BenchmarkConfig {
  prompt_tokens: number
  max_output_tokens: number
  num_runs: number
  warmup_runs: number
  temperature: number
  preset: string
}

interface BenchmarkStats {
  min: number
  max: number
  mean: number
  median: number
  stdev: number
}

interface BenchmarkResult {
  id: string
  config: BenchmarkConfig
  model_name: string
  model_variant: string
  status: string
  progress: number
  current_run: number
  total_runs: number
  runs: Array<{
    run_number: number
    prompt_tokens: number
    completion_tokens: number
    time_to_first_token_ms: number
    tokens_per_second: number
    total_time_ms: number
    timestamp: string
  }>
  statistics?: {
    tokens_per_second: BenchmarkStats
    time_to_first_token_ms: BenchmarkStats
    total_time_ms: BenchmarkStats
    num_successful_runs: number
    total_tokens_generated: number
  }
  started_at: string
  completed_at?: string
  error?: string
}

interface PresetConfig {
  prompt_tokens: number
  max_output_tokens: number
  num_runs: number
  warmup_runs: number
  temperature: number
  preset: string
}

const PRESET_DESCRIPTIONS: Record<string, string> = {
  quick: 'Fast sanity check with minimal runs',
  standard: 'Balanced test for typical use cases',
  long_context: 'Test performance with longer prompts',
  max_speed: 'Optimized for measuring peak speed',
}

export default function BenchmarkPage() {
  // State
  const [tabValue, setTabValue] = useState(0)
  const [presets, setPresets] = useState<Record<string, PresetConfig>>({})
  const [selectedPreset, setSelectedPreset] = useState('standard')
  const [customConfig, setCustomConfig] = useState<BenchmarkConfig>({
    prompt_tokens: 512,
    max_output_tokens: 256,
    num_runs: 5,
    warmup_runs: 1,
    temperature: 0.0,
    preset: 'standard',
  })
  const [modelName, setModelName] = useState('')
  const [modelVariant, setModelVariant] = useState('')
  
  const [activeBenchmark, setActiveBenchmark] = useState<BenchmarkResult | null>(null)
  const [benchmarkHistory, setBenchmarkHistory] = useState<BenchmarkResult[]>([])
  const [stats, setStats] = useState<{
    total_benchmarks: number
    completed: number
    failed: number
    average_tps_recent: number
  } | null>(null)
  
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // Load initial data
  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [presetsRes, historyRes, statsRes] = await Promise.all([
        fetch('/backend/api/v1/benchmark/presets').then(r => r.json()),
        fetch('/backend/api/v1/benchmark/history?limit=20').then(r => r.json()),
        fetch('/backend/api/v1/benchmark/stats').then(r => r.json()),
      ])
      setPresets(presetsRes.presets || {})
      setBenchmarkHistory(historyRes.benchmarks || [])
      setStats(statsRes)
      
      // Set default config from standard preset
      if (presetsRes.presets?.standard) {
        setCustomConfig(presetsRes.presets.standard)
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load benchmark data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadData()
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
    }
  }, [loadData])

  // Poll for active benchmark status
  const pollBenchmarkStatus = useCallback(async (benchmarkId: string) => {
    try {
      const response = await fetch(`/backend/api/v1/benchmark/${benchmarkId}`)
      if (response.ok) {
        const result = await response.json()
        setActiveBenchmark(result)
        
        if (result.status === 'completed' || result.status === 'failed') {
          setRunning(false)
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current)
            pollIntervalRef.current = null
          }
          loadData() // Refresh history
        }
      }
    } catch (err) {
      console.error('Failed to poll benchmark status:', err)
    }
  }, [loadData])

  // Start a benchmark
  const handleStartBenchmark = async () => {
    setError(null)
    setRunning(true)
    setActiveBenchmark(null)
    
    try {
      const response = await fetch('/backend/api/v1/benchmark/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          preset: selectedPreset,
          prompt_tokens: customConfig.prompt_tokens,
          max_output_tokens: customConfig.max_output_tokens,
          num_runs: customConfig.num_runs,
          warmup_runs: customConfig.warmup_runs,
          temperature: customConfig.temperature,
          model_name: modelName || 'unknown',
          model_variant: modelVariant || 'unknown',
        }),
      })
      
      if (!response.ok) {
        throw new Error('Failed to start benchmark')
      }
      
      const data = await response.json()
      
      if (data.benchmark_id) {
        // Start polling
        pollIntervalRef.current = setInterval(() => {
          pollBenchmarkStatus(data.benchmark_id)
        }, 1000)
        
        // Initial poll
        await pollBenchmarkStatus(data.benchmark_id)
      }
    } catch (err: any) {
      setError(err.message || 'Failed to start benchmark')
      setRunning(false)
    }
  }

  // Handle preset change
  const handlePresetChange = (preset: string) => {
    setSelectedPreset(preset)
    if (presets[preset]) {
      setCustomConfig(presets[preset])
    }
  }

  // Delete a benchmark
  const handleDeleteBenchmark = async (benchmarkId: string) => {
    try {
      await fetch(`/backend/api/v1/benchmark/${benchmarkId}`, {
        method: 'DELETE',
      })
      loadData()
    } catch (err: any) {
      setError(err.message || 'Failed to delete benchmark')
    }
  }

  // View benchmark details
  const handleViewBenchmark = async (benchmarkId: string) => {
    try {
      const response = await fetch(`/backend/api/v1/benchmark/${benchmarkId}`)
      if (response.ok) {
        const result = await response.json()
        setActiveBenchmark(result)
        setTabValue(0) // Switch to results tab
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load benchmark')
    }
  }

  const formatNumber = (num: number, decimals: number = 1) => {
    return num.toFixed(decimals)
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" sx={{ fontWeight: 600 }}>
          Inference Benchmark
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title="Refresh">
            <IconButton onClick={loadData}>
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

      {/* Stats Cards */}
      {stats && (
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={6} md={3}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <HistoryIcon color="primary" />
                <Typography variant="h5">{stats.total_benchmarks}</Typography>
                <Typography variant="caption" color="text.secondary">Total Benchmarks</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={3}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <SpeedIcon color="success" />
                <Typography variant="h5">{stats.completed}</Typography>
                <Typography variant="caption" color="text.secondary">Completed</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={3}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <TrendingIcon color="info" />
                <Typography variant="h5">{formatNumber(stats.average_tps_recent)}</Typography>
                <Typography variant="caption" color="text.secondary">Avg TPS (Recent)</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={3}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <TimerIcon color="warning" />
                <Typography variant="h5">{stats.failed}</Typography>
                <Typography variant="caption" color="text.secondary">Failed</Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      <Grid container spacing={3}>
        {/* Configuration Panel */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
              <SettingsIcon /> Configuration
            </Typography>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Preset</InputLabel>
              <Select
                value={selectedPreset}
                label="Preset"
                onChange={(e) => handlePresetChange(e.target.value)}
              >
                {Object.keys(presets).map(preset => (
                  <MenuItem key={preset} value={preset}>
                    {preset.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {PRESET_DESCRIPTIONS[selectedPreset] && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 2 }}>
                {PRESET_DESCRIPTIONS[selectedPreset]}
              </Typography>
            )}

            <Divider sx={{ my: 2 }} />

            <Typography variant="subtitle2" sx={{ mb: 1 }}>Custom Settings</Typography>

            <TextField
              label="Prompt Tokens"
              type="number"
              value={customConfig.prompt_tokens}
              onChange={(e) => setCustomConfig({ ...customConfig, prompt_tokens: parseInt(e.target.value) || 0 })}
              fullWidth
              size="small"
              sx={{ mb: 2 }}
            />

            <TextField
              label="Max Output Tokens"
              type="number"
              value={customConfig.max_output_tokens}
              onChange={(e) => setCustomConfig({ ...customConfig, max_output_tokens: parseInt(e.target.value) || 0 })}
              fullWidth
              size="small"
              sx={{ mb: 2 }}
            />

            <TextField
              label="Number of Runs"
              type="number"
              value={customConfig.num_runs}
              onChange={(e) => setCustomConfig({ ...customConfig, num_runs: parseInt(e.target.value) || 1 })}
              fullWidth
              size="small"
              sx={{ mb: 2 }}
              inputProps={{ min: 1, max: 20 }}
            />

            <TextField
              label="Warmup Runs"
              type="number"
              value={customConfig.warmup_runs}
              onChange={(e) => setCustomConfig({ ...customConfig, warmup_runs: parseInt(e.target.value) || 0 })}
              fullWidth
              size="small"
              sx={{ mb: 2 }}
              inputProps={{ min: 0, max: 5 }}
            />

            <Divider sx={{ my: 2 }} />

            <TextField
              label="Model Name (optional)"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              fullWidth
              size="small"
              sx={{ mb: 2 }}
              placeholder="e.g., Llama-3.1-8B"
            />

            <TextField
              label="Model Variant (optional)"
              value={modelVariant}
              onChange={(e) => setModelVariant(e.target.value)}
              fullWidth
              size="small"
              sx={{ mb: 2 }}
              placeholder="e.g., Q4_K_M"
            />

            <Button
              variant="contained"
              fullWidth
              startIcon={running ? <CircularProgress size={20} color="inherit" /> : <StartIcon />}
              onClick={handleStartBenchmark}
              disabled={running || loading}
              sx={{ mt: 1 }}
            >
              {running ? 'Running...' : 'Start Benchmark'}
            </Button>
          </Paper>
        </Grid>

        {/* Results Panel */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Tabs value={tabValue} onChange={(_, v) => setTabValue(v)} sx={{ mb: 2 }}>
              <Tab label="Current Results" />
              <Tab label="History" />
            </Tabs>

            {tabValue === 0 && (
              <Box>
                {/* Progress Bar */}
                {running && activeBenchmark && (
                  <Box sx={{ mb: 3 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">
                        Run {activeBenchmark.current_run} of {activeBenchmark.total_runs}
                      </Typography>
                      <Typography variant="body2">
                        {activeBenchmark.progress}%
                      </Typography>
                    </Box>
                    <LinearProgress variant="determinate" value={activeBenchmark.progress} />
                  </Box>
                )}

                {/* Results Summary */}
                {activeBenchmark?.statistics && (
                  <Grid container spacing={2} sx={{ mb: 3 }}>
                    <Grid item xs={6} md={3}>
                      <Card variant="outlined">
                        <CardContent sx={{ textAlign: 'center', py: 1.5 }}>
                          <SpeedIcon color="primary" sx={{ fontSize: 28 }} />
                          <Typography variant="h5">
                            {formatNumber(activeBenchmark.statistics.tokens_per_second.mean)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Tokens/Second
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Card variant="outlined">
                        <CardContent sx={{ textAlign: 'center', py: 1.5 }}>
                          <TimerIcon color="info" sx={{ fontSize: 28 }} />
                          <Typography variant="h5">
                            {formatNumber(activeBenchmark.statistics.time_to_first_token_ms.mean, 0)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            TTFT (ms)
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Card variant="outlined">
                        <CardContent sx={{ textAlign: 'center', py: 1.5 }}>
                          <TokenIcon color="success" sx={{ fontSize: 28 }} />
                          <Typography variant="h5">
                            {activeBenchmark.statistics.total_tokens_generated}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Total Tokens
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Card variant="outlined">
                        <CardContent sx={{ textAlign: 'center', py: 1.5 }}>
                          <TrendingIcon color="warning" sx={{ fontSize: 28 }} />
                          <Typography variant="h5">
                            {formatNumber(activeBenchmark.statistics.total_time_ms.mean / 1000)}s
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Avg Time
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                )}

                {/* Detailed Statistics */}
                {activeBenchmark?.statistics && (
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>Detailed Statistics</Typography>
                    <TableContainer>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Metric</TableCell>
                            <TableCell align="right">Min</TableCell>
                            <TableCell align="right">Max</TableCell>
                            <TableCell align="right">Mean</TableCell>
                            <TableCell align="right">Median</TableCell>
                            <TableCell align="right">Std Dev</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          <TableRow>
                            <TableCell>Tokens/Second</TableCell>
                            <TableCell align="right">{formatNumber(activeBenchmark.statistics.tokens_per_second.min)}</TableCell>
                            <TableCell align="right">{formatNumber(activeBenchmark.statistics.tokens_per_second.max)}</TableCell>
                            <TableCell align="right">{formatNumber(activeBenchmark.statistics.tokens_per_second.mean)}</TableCell>
                            <TableCell align="right">{formatNumber(activeBenchmark.statistics.tokens_per_second.median)}</TableCell>
                            <TableCell align="right">{formatNumber(activeBenchmark.statistics.tokens_per_second.stdev)}</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>TTFT (ms)</TableCell>
                            <TableCell align="right">{formatNumber(activeBenchmark.statistics.time_to_first_token_ms.min, 0)}</TableCell>
                            <TableCell align="right">{formatNumber(activeBenchmark.statistics.time_to_first_token_ms.max, 0)}</TableCell>
                            <TableCell align="right">{formatNumber(activeBenchmark.statistics.time_to_first_token_ms.mean, 0)}</TableCell>
                            <TableCell align="right">{formatNumber(activeBenchmark.statistics.time_to_first_token_ms.median, 0)}</TableCell>
                            <TableCell align="right">{formatNumber(activeBenchmark.statistics.time_to_first_token_ms.stdev, 0)}</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>Total Time (ms)</TableCell>
                            <TableCell align="right">{formatNumber(activeBenchmark.statistics.total_time_ms.min, 0)}</TableCell>
                            <TableCell align="right">{formatNumber(activeBenchmark.statistics.total_time_ms.max, 0)}</TableCell>
                            <TableCell align="right">{formatNumber(activeBenchmark.statistics.total_time_ms.mean, 0)}</TableCell>
                            <TableCell align="right">{formatNumber(activeBenchmark.statistics.total_time_ms.median, 0)}</TableCell>
                            <TableCell align="right">{formatNumber(activeBenchmark.statistics.total_time_ms.stdev, 0)}</TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Box>
                )}

                {/* Individual Runs */}
                {activeBenchmark?.runs && activeBenchmark.runs.length > 0 && (
                  <Box>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>Individual Runs</Typography>
                    <TableContainer>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Run</TableCell>
                            <TableCell align="right">Prompt Tokens</TableCell>
                            <TableCell align="right">Output Tokens</TableCell>
                            <TableCell align="right">TTFT (ms)</TableCell>
                            <TableCell align="right">TPS</TableCell>
                            <TableCell align="right">Total Time</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {activeBenchmark.runs.map((run) => (
                            <TableRow key={run.run_number}>
                              <TableCell>{run.run_number}</TableCell>
                              <TableCell align="right">{run.prompt_tokens}</TableCell>
                              <TableCell align="right">{run.completion_tokens}</TableCell>
                              <TableCell align="right">{formatNumber(run.time_to_first_token_ms, 0)}</TableCell>
                              <TableCell align="right">{formatNumber(run.tokens_per_second)}</TableCell>
                              <TableCell align="right">{formatNumber(run.total_time_ms / 1000)}s</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Box>
                )}

                {/* No Results Yet */}
                {!activeBenchmark && !running && (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <SpeedIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                    <Typography color="text.secondary">
                      No benchmark results yet. Configure and start a benchmark to measure inference speed.
                    </Typography>
                  </Box>
                )}
              </Box>
            )}

            {tabValue === 1 && (
              <Box>
                {benchmarkHistory.length === 0 ? (
                  <Typography color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
                    No benchmark history yet.
                  </Typography>
                ) : (
                  <List>
                    {benchmarkHistory.map((benchmark) => (
                      <ListItem
                        key={benchmark.id}
                        divider
                        secondaryAction={
                          <Box>
                            <Tooltip title="View Details">
                              <IconButton size="small" onClick={() => handleViewBenchmark(benchmark.id)}>
                                <SpeedIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Delete">
                              <IconButton size="small" onClick={() => handleDeleteBenchmark(benchmark.id)} color="error">
                                <DeleteIcon />
                              </IconButton>
                            </Tooltip>
                          </Box>
                        }
                      >
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="body1">
                                {benchmark.model_name || 'Unknown Model'}
                              </Typography>
                              <Chip
                                size="small"
                                label={benchmark.status}
                                color={benchmark.status === 'completed' ? 'success' : benchmark.status === 'failed' ? 'error' : 'default'}
                              />
                              {benchmark.config?.preset && (
                                <Chip size="small" label={benchmark.config.preset} variant="outlined" />
                              )}
                            </Box>
                          }
                          secondary={
                            <Box sx={{ display: 'flex', gap: 2, mt: 0.5 }}>
                              {benchmark.statistics?.tokens_per_second && (
                                <Typography variant="caption">
                                  TPS: {formatNumber(benchmark.statistics.tokens_per_second.mean)}
                                </Typography>
                              )}
                              {benchmark.statistics?.time_to_first_token_ms && (
                                <Typography variant="caption">
                                  TTFT: {formatNumber(benchmark.statistics.time_to_first_token_ms.mean, 0)}ms
                                </Typography>
                              )}
                              <Typography variant="caption" color="text.secondary">
                                {new Date(benchmark.started_at).toLocaleString()}
                              </Typography>
                            </Box>
                          }
                        />
                      </ListItem>
                    ))}
                  </List>
                )}
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  )
}
