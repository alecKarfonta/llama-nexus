import React, { useState, useEffect } from 'react'
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Container,
  FormControl,
  FormControlLabel,
  FormLabel,
  Grid,
  LinearProgress,
  MenuItem,
  Radio,
  RadioGroup,
  Select,
  TextField,
  Typography,
  Alert,
  Chip,
  Divider
} from '@mui/material'
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  Assessment as BenchmarkIcon
} from '@mui/icons-material'
import { apiService } from '@/services/api'

interface BenchmarkResult {
  status: 'starting' | 'running' | 'completed' | 'failed' | 'cancelled'
  test_category?: string
  max_examples?: number
  started_at?: string
  completed_at?: string
  progress?: number
  total?: number
  current_test?: string
  results?: {
    total: number
    correct: number
    accuracy: number
    errors: string[]
  }
  error?: string
}

export const TestingPage: React.FC = () => {
  const [benchmarkStatus, setBenchmarkStatus] = useState<BenchmarkResult>({ status: 'starting' })
  const [isRunning, setIsRunning] = useState(false)
  const [testCategory, setTestCategory] = useState('tool_calling')
  const [maxExamples, setMaxExamples] = useState(10)
  const [benchmarkId, setBenchmarkId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [quickTestRunning, setQuickTestRunning] = useState(false)
  const [quickTestResult, setQuickTestResult] = useState<string | null>(null)

  // Poll for benchmark status
  useEffect(() => {
    let intervalId: NodeJS.Timeout
    
    if (isRunning) {
      intervalId = setInterval(async () => {
        try {
          const status = await apiService.getBfclBenchmarkStatus(benchmarkId || undefined)
          setBenchmarkStatus(status)
          
          if (status.status === 'completed' || status.status === 'failed' || status.status === 'cancelled') {
            setIsRunning(false)
          }
        } catch (err) {
          console.error('Error fetching benchmark status:', err)
          setError('Failed to fetch benchmark status')
        }
      }, 2000)
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId)
    }
  }, [isRunning, benchmarkId])

  const startBenchmark = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await apiService.startBfclBenchmark({
        test_category: testCategory,
        max_examples: maxExamples
      })
      
      setBenchmarkId(response.benchmark_id)
      setIsRunning(true)
      setBenchmarkStatus({ 
        status: 'starting',
        test_category: testCategory,
        max_examples: maxExamples
      })
    } catch (err) {
      console.error('Error starting benchmark:', err)
      setError('Failed to start benchmark')
    } finally {
      setLoading(false)
    }
  }

  const stopBenchmark = async () => {
    if (!benchmarkId) return
    
    setLoading(true)
    try {
      await apiService.stopBfclBenchmark(benchmarkId)
      setIsRunning(false)
      setBenchmarkStatus(prev => ({ ...prev, status: 'cancelled' }))
    } catch (err) {
      console.error('Error stopping benchmark:', err)
      setError('Failed to stop benchmark')
    } finally {
      setLoading(false)
    }
  }

  const clearResults = async () => {
    setLoading(true)
    try {
      await apiService.clearCompletedBenchmarks()
      setBenchmarkStatus({ status: 'starting' })
      setBenchmarkId(null)
    } catch (err) {
      console.error('Error clearing results:', err)
      setError('Failed to clear results')
    } finally {
      setLoading(false)
    }
  }

  const runQuickTest = async () => {
    setQuickTestRunning(true)
    setQuickTestResult(null)
    setError(null)
    
    try {
      // Run a simple tool calling test
      const response = await apiService.createChatCompletion({
        messages: [
          { role: 'user', content: 'What is 15 * 23? Please use the calculator tool.' }
        ],
        tools: [
          {
            type: 'function',
            function: {
              name: 'calculate',
              description: 'Perform mathematical calculations and evaluate expressions',
              parameters: {
                type: 'object',
                properties: {
                  expression: {
                    type: 'string',
                    description: 'Mathematical expression to evaluate, e.g., "2 + 2 * 3" or "sqrt(16)"'
                  }
                },
                required: ['expression']
              }
            }
          }
        ],
        tool_choice: 'auto',
        temperature: 0.1,
        max_tokens: 200
      })
      
      // Check if tool was called correctly
      const message = response.choices?.[0]?.message
      const toolCalls = message?.tool_calls || []
      
      if (toolCalls.length > 0) {
        const toolCall = toolCalls[0]
        if (toolCall.function.name === 'calculate') {
          try {
            const args = JSON.parse(toolCall.function.arguments)
            if (args.expression && args.expression.includes('15') && args.expression.includes('23')) {
              setQuickTestResult('✅ SUCCESS: Model correctly called the calculator tool with the right expression!')
            } else {
              setQuickTestResult('⚠️ PARTIAL: Tool called but with unexpected expression: ' + args.expression)
            }
          } catch {
            setQuickTestResult('❌ FAILED: Tool called but arguments are malformed')
          }
        } else {
          setQuickTestResult('❌ FAILED: Wrong tool called: ' + toolCall.function.name)
        }
      } else {
        setQuickTestResult('❌ FAILED: No tool calls made. Response: ' + (message?.content || 'No response'))
      }
    } catch (err) {
      console.error('Quick test error:', err)
      setError('Quick test failed: ' + (err instanceof Error ? err.message : 'Unknown error'))
      setQuickTestResult('❌ ERROR: Test execution failed')
    } finally {
      setQuickTestRunning(false)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success'
      case 'failed': return 'error'
      case 'running': return 'info'
      case 'starting': return 'warning'
      default: return 'default'
    }
  }

  const formatTime = (dateString?: string) => {
    if (!dateString) return 'N/A'
    return new Date(dateString).toLocaleTimeString()
  }

  return (
    <Container maxWidth="lg" sx={{ py: 3 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          <BenchmarkIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Tool Calling Benchmark
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Test and evaluate your model's tool calling capabilities
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Configuration Panel */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Benchmark Configuration
              </Typography>
              
              <FormControl fullWidth sx={{ mb: 2 }}>
                <FormLabel>Test Category</FormLabel>
                <Select
                  value={testCategory}
                  onChange={(e) => setTestCategory(e.target.value)}
                  disabled={isRunning}
                >
                  <MenuItem value="tool_calling">Tool Calling</MenuItem>
                  <MenuItem value="basic">Basic Functionality</MenuItem>
                  <MenuItem value="advanced">Advanced Reasoning</MenuItem>
                </Select>
              </FormControl>
              
              <FormControl fullWidth sx={{ mb: 3 }}>
                <FormLabel>Max Examples</FormLabel>
                <TextField
                  type="number"
                  value={maxExamples}
                  onChange={(e) => setMaxExamples(Number(e.target.value))}
                  InputProps={{ inputProps: { min: 1, max: 100 } }}
                  disabled={isRunning}
                />
              </FormControl>
              
              <Box sx={{ display: 'flex', gap: 1, flexDirection: 'column' }}>
                {!isRunning ? (
                  <Button
                    variant="contained"
                    startIcon={<PlayIcon />}
                    onClick={startBenchmark}
                    disabled={loading}
                    fullWidth
                  >
                    {loading ? <CircularProgress size={24} /> : 'Start Full Benchmark'}
                  </Button>
                ) : (
                  <Button
                    variant="outlined"
                    color="error"
                    startIcon={<StopIcon />}
                    onClick={stopBenchmark}
                    disabled={loading}
                    fullWidth
                  >
                    {loading ? <CircularProgress size={24} /> : 'Stop Benchmark'}
                  </Button>
                )}
                
                <Button
                  variant="outlined"
                  color="primary"
                  startIcon={quickTestRunning ? <CircularProgress size={20} /> : <PlayIcon />}
                  onClick={runQuickTest}
                  disabled={quickTestRunning || loading}
                  fullWidth
                >
                  {quickTestRunning ? 'Testing...' : 'Quick Tool Test'}
                </Button>
                
                <Button
                  variant="outlined"
                  startIcon={<RefreshIcon />}
                  onClick={clearResults}
                  disabled={loading || isRunning}
                  size="small"
                >
                  Clear
                </Button>
              </Box>
            </CardContent>
          </Card>
          
          {/* Quick Test Result */}
          {quickTestResult && (
            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Quick Test Result
                </Typography>
                <Alert 
                  severity={
                    quickTestResult.includes('SUCCESS') ? 'success' : 
                    quickTestResult.includes('PARTIAL') ? 'warning' : 'error'
                  }
                  sx={{ mt: 1 }}
                >
                  {quickTestResult}
                </Alert>
              </CardContent>
            </Card>
          )}

          {/* Benchmark Info */}
          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                About Tool Calling Benchmark
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                This benchmark evaluates your model's ability to properly call tools and functions. 
                It tests various scenarios including:
              </Typography>
              <ul>
                <li>Function detection and invocation</li>
                <li>Parameter extraction and formatting</li>
                <li>Error handling in tool calls</li>
                <li>Multi-step tool chaining</li>
              </ul>
              <Typography variant="body2" color="text.secondary" paragraph sx={{ mt: 2 }}>
                <strong>Quick Tool Test:</strong> Runs a single test to verify basic tool calling functionality.
                <br />
                <strong>Full Benchmark:</strong> Runs comprehensive BFCL-compatible tests across multiple categories.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Results Panel */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Benchmark Results
                </Typography>
                <Chip 
                  label={benchmarkStatus.status || 'Not Started'} 
                  color={getStatusColor(benchmarkStatus.status)} 
                  size="small"
                />
              </Box>
              
              {benchmarkStatus.status !== 'starting' && (
                <>
                  <Grid container spacing={2} sx={{ mb: 3 }}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Test Category
                      </Typography>
                      <Typography variant="h6">
                        {benchmarkStatus.test_category || 'N/A'}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Max Examples
                      </Typography>
                      <Typography variant="h6">
                        {benchmarkStatus.max_examples || 'N/A'}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Started At
                      </Typography>
                      <Typography variant="h6">
                        {formatTime(benchmarkStatus.started_at)}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Completed At
                      </Typography>
                      <Typography variant="h6">
                        {formatTime(benchmarkStatus.completed_at)}
                      </Typography>
                    </Grid>
                  </Grid>
                  
                  {(benchmarkStatus.status === 'running' || benchmarkStatus.progress) && (
                    <Box sx={{ mb: 3 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2" color="text.secondary">
                          Progress
                        </Typography>
                        <Typography variant="body2">
                          {benchmarkStatus.progress || 0}/{benchmarkStatus.total || '?'}
                        </Typography>
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={benchmarkStatus.total ? (benchmarkStatus.progress || 0) / benchmarkStatus.total * 100 : 0} 
                      />
                      {benchmarkStatus.current_test && (
                        <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                          Current test: {benchmarkStatus.current_test}
                        </Typography>
                      )}
                    </Box>
                  )}
                  
                  {benchmarkStatus.results && (
                    <Box>
                      <Typography variant="h6" gutterBottom>
                        Results Summary
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid item xs={4}>
                          <Card variant="outlined">
                            <CardContent sx={{ textAlign: 'center' }}>
                              <Typography variant="h4" color="primary">
                                {benchmarkStatus.results.accuracy.toFixed(1)}%
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                Accuracy
                              </Typography>
                            </CardContent>
                          </Card>
                        </Grid>
                        <Grid item xs={4}>
                          <Card variant="outlined">
                            <CardContent sx={{ textAlign: 'center' }}>
                              <Typography variant="h4" color="success.main">
                                {benchmarkStatus.results.correct}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                Correct
                              </Typography>
                            </CardContent>
                          </Card>
                        </Grid>
                        <Grid item xs={4}>
                          <Card variant="outlined">
                            <CardContent sx={{ textAlign: 'center' }}>
                              <Typography variant="h4" color="error.main">
                                {benchmarkStatus.results.total - benchmarkStatus.results.correct}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                Errors
                              </Typography>
                            </CardContent>
                          </Card>
                        </Grid>
                      </Grid>
                      
                      {benchmarkStatus.results.errors.length > 0 && (
                        <Box sx={{ mt: 3 }}>
                          <Typography variant="h6" gutterBottom>
                            Errors
                          </Typography>
                          {benchmarkStatus.results.errors.map((error, index) => (
                            <Alert key={index} severity="warning" sx={{ mb: 1 }}>
                              {error}
                            </Alert>
                          ))}
                        </Box>
                      )}
                    </Box>
                  )}
                  
                  {benchmarkStatus.error && (
                    <Alert severity="error" sx={{ mt: 2 }}>
                      <Typography variant="subtitle2">Benchmark Error:</Typography>
                      <Typography variant="body2">{benchmarkStatus.error}</Typography>
                    </Alert>
                  )}
                </>
              )}
              
              {benchmarkStatus.status === 'starting' && (
                <Box sx={{ textAlign: 'center', py: 5 }}>
                  <Typography variant="body1" color="text.secondary">
                    Configure and start a benchmark to see results
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  )
}
