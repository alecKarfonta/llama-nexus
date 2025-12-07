import React, { useState, useEffect, useCallback } from 'react'
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
  Divider,
  Tabs,
  Tab,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Paper,
  Tooltip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Switch,
  InputLabel,
} from '@mui/material'
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  Assessment as BenchmarkIcon,
  Build as ToolIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Save as SaveIcon,
  ContentCopy as CopyIcon,
  ExpandMore as ExpandIcon,
  Code as CodeIcon,
  Send as SendIcon,
  Visibility as ViewIcon,
  Functions as FunctionIcon,
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

interface ToolParameter {
  name: string
  type: 'string' | 'number' | 'boolean' | 'array' | 'object'
  description: string
  required: boolean
  enum?: string[]
}

interface CustomTool {
  id: string
  name: string
  description: string
  parameters: ToolParameter[]
  mockResponse?: string
}

interface ToolCall {
  id: string
  name: string
  arguments: Record<string, any>
  response?: string
  timestamp: Date
}

// Default example tools
const DEFAULT_TOOLS: CustomTool[] = [
  {
    id: 'calc-1',
    name: 'calculate',
    description: 'Perform mathematical calculations',
    parameters: [
      { name: 'expression', type: 'string', description: 'Math expression to evaluate', required: true }
    ],
    mockResponse: '{"result": 345}'
  },
  {
    id: 'weather-1',
    name: 'get_weather',
    description: 'Get current weather for a location',
    parameters: [
      { name: 'location', type: 'string', description: 'City name or coordinates', required: true },
      { name: 'units', type: 'string', description: 'Temperature units', required: false, enum: ['celsius', 'fahrenheit'] }
    ],
    mockResponse: '{"temperature": 22, "condition": "sunny", "humidity": 45}'
  },
  {
    id: 'search-1',
    name: 'web_search',
    description: 'Search the web for information',
    parameters: [
      { name: 'query', type: 'string', description: 'Search query', required: true },
      { name: 'num_results', type: 'number', description: 'Number of results to return', required: false }
    ],
    mockResponse: '{"results": [{"title": "Example Result", "url": "https://example.com", "snippet": "..."}]}'
  }
]

export const TestingPage: React.FC = () => {
  // Tab state
  const [tabValue, setTabValue] = useState(0)
  
  // Benchmark state
  const [benchmarkStatus, setBenchmarkStatus] = useState<BenchmarkResult>({ status: 'starting' })
  const [isRunning, setIsRunning] = useState(false)
  const [testCategory, setTestCategory] = useState('tool_calling')
  const [maxExamples, setMaxExamples] = useState(10)
  const [benchmarkId, setBenchmarkId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [quickTestRunning, setQuickTestRunning] = useState(false)
  const [quickTestResult, setQuickTestResult] = useState<string | null>(null)
  
  // Tool Playground state
  const [customTools, setCustomTools] = useState<CustomTool[]>(DEFAULT_TOOLS)
  const [selectedTools, setSelectedTools] = useState<string[]>(['calc-1'])
  const [playgroundMessage, setPlaygroundMessage] = useState('')
  const [playgroundResponse, setPlaygroundResponse] = useState<string | null>(null)
  const [toolCalls, setToolCalls] = useState<ToolCall[]>([])
  const [playgroundLoading, setPlaygroundLoading] = useState(false)
  const [useMockResponses, setUseMockResponses] = useState(true)
  
  // Tool Builder state
  const [toolBuilderOpen, setToolBuilderOpen] = useState(false)
  const [editingTool, setEditingTool] = useState<CustomTool | null>(null)
  const [newTool, setNewTool] = useState<CustomTool>({
    id: '',
    name: '',
    description: '',
    parameters: [],
    mockResponse: ''
  })

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
                    description: 'Mathematical expression to evaluate'
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
      
      const message = response.choices?.[0]?.message
      const calls = message?.tool_calls || []
      
      if (calls.length > 0) {
        const toolCall = calls[0]
        if (toolCall.function.name === 'calculate') {
          try {
            const args = JSON.parse(toolCall.function.arguments)
            if (args.expression && args.expression.includes('15') && args.expression.includes('23')) {
              setQuickTestResult('SUCCESS: Model correctly called the calculator tool with the right expression!')
            } else {
              setQuickTestResult('PARTIAL: Tool called but with unexpected expression: ' + args.expression)
            }
          } catch {
            setQuickTestResult('FAILED: Tool called but arguments are malformed')
          }
        } else {
          setQuickTestResult('FAILED: Wrong tool called: ' + toolCall.function.name)
        }
      } else {
        setQuickTestResult('FAILED: No tool calls made. Response: ' + (message?.content || 'No response'))
      }
    } catch (err) {
      console.error('Quick test error:', err)
      setError('Quick test failed: ' + (err instanceof Error ? err.message : 'Unknown error'))
      setQuickTestResult('ERROR: Test execution failed')
    } finally {
      setQuickTestRunning(false)
    }
  }

  // Tool Playground functions
  const convertToolToOpenAI = (tool: CustomTool) => {
    const properties: Record<string, any> = {}
    const required: string[] = []
    
    tool.parameters.forEach(param => {
      properties[param.name] = {
        type: param.type,
        description: param.description,
        ...(param.enum ? { enum: param.enum } : {})
      }
      if (param.required) {
        required.push(param.name)
      }
    })
    
    return {
      type: 'function' as const,
      function: {
        name: tool.name,
        description: tool.description,
        parameters: {
          type: 'object',
          properties,
          required
        }
      }
    }
  }

  const runPlaygroundTest = async () => {
    if (!playgroundMessage.trim()) return
    
    setPlaygroundLoading(true)
    setPlaygroundResponse(null)
    setError(null)
    
    const selectedToolObjects = customTools.filter(t => selectedTools.includes(t.id))
    const openAITools = selectedToolObjects.map(convertToolToOpenAI)
    
    try {
      const response = await apiService.createChatCompletion({
        messages: [{ role: 'user', content: playgroundMessage }],
        tools: openAITools,
        tool_choice: 'auto',
        temperature: 0.7,
        max_tokens: 500
      })
      
      const message = response.choices?.[0]?.message
      const calls = message?.tool_calls || []
      
      if (calls.length > 0) {
        // Process tool calls
        const newToolCalls: ToolCall[] = calls.map((call: any) => {
          let args = {}
          try {
            args = JSON.parse(call.function.arguments)
          } catch {}
          
          // Find mock response if enabled
          let mockResp = undefined
          if (useMockResponses) {
            const tool = selectedToolObjects.find(t => t.name === call.function.name)
            mockResp = tool?.mockResponse
          }
          
          return {
            id: call.id,
            name: call.function.name,
            arguments: args,
            response: mockResp,
            timestamp: new Date()
          }
        })
        
        setToolCalls(prev => [...newToolCalls, ...prev])
        
        // If using mock responses, continue the conversation
        if (useMockResponses && newToolCalls.every(tc => tc.response)) {
          const toolMessages = newToolCalls.map(tc => ({
            role: 'tool' as const,
            tool_call_id: tc.id,
            content: tc.response || ''
          }))
          
          const followUp = await apiService.createChatCompletion({
            messages: [
              { role: 'user', content: playgroundMessage },
              {
                role: 'assistant',
                content: null,
                tool_calls: calls
              },
              ...toolMessages
            ],
            tools: openAITools,
            temperature: 0.7,
            max_tokens: 500
          })
          
          setPlaygroundResponse(followUp.choices?.[0]?.message?.content || 'No response')
        } else {
          setPlaygroundResponse(`Tool calls made: ${calls.map((c: any) => c.function.name).join(', ')}`)
        }
      } else {
        setPlaygroundResponse(message?.content || 'No response')
      }
    } catch (err) {
      console.error('Playground error:', err)
      setError('Playground test failed: ' + (err instanceof Error ? err.message : 'Unknown error'))
    } finally {
      setPlaygroundLoading(false)
    }
  }

  // Tool Builder functions
  const openToolBuilder = (tool?: CustomTool) => {
    if (tool) {
      setEditingTool(tool)
      setNewTool({ ...tool })
    } else {
      setEditingTool(null)
      setNewTool({
        id: `tool-${Date.now()}`,
        name: '',
        description: '',
        parameters: [],
        mockResponse: ''
      })
    }
    setToolBuilderOpen(true)
  }

  const saveTool = () => {
    if (!newTool.name.trim()) return
    
    if (editingTool) {
      setCustomTools(prev => prev.map(t => t.id === editingTool.id ? newTool : t))
    } else {
      setCustomTools(prev => [...prev, newTool])
    }
    setToolBuilderOpen(false)
  }

  const deleteTool = (id: string) => {
    setCustomTools(prev => prev.filter(t => t.id !== id))
    setSelectedTools(prev => prev.filter(tid => tid !== id))
  }

  const addParameter = () => {
    setNewTool(prev => ({
      ...prev,
      parameters: [...prev.parameters, { name: '', type: 'string', description: '', required: false }]
    }))
  }

  const updateParameter = (index: number, field: keyof ToolParameter, value: any) => {
    setNewTool(prev => ({
      ...prev,
      parameters: prev.parameters.map((p, i) => i === index ? { ...p, [field]: value } : p)
    }))
  }

  const removeParameter = (index: number) => {
    setNewTool(prev => ({
      ...prev,
      parameters: prev.parameters.filter((_, i) => i !== index)
    }))
  }

  const toggleToolSelection = (id: string) => {
    setSelectedTools(prev => 
      prev.includes(id) ? prev.filter(tid => tid !== id) : [...prev, id]
    )
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
    <Container maxWidth="xl" sx={{ py: 3 }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 600 }}>
          Function Calling Playground
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Test, build, and benchmark tool calling capabilities
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Tabs value={tabValue} onChange={(_, v) => setTabValue(v)} sx={{ mb: 3 }}>
        <Tab icon={<ToolIcon />} label="Playground" iconPosition="start" />
        <Tab icon={<CodeIcon />} label="Tool Builder" iconPosition="start" />
        <Tab icon={<BenchmarkIcon />} label="Benchmark" iconPosition="start" />
      </Tabs>

      {/* Playground Tab */}
      {tabValue === 0 && (
        <Grid container spacing={3}>
          {/* Tool Selection */}
          <Grid item xs={12} md={4}>
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Available Tools</Typography>
                  <Tooltip title="Create new tool">
                    <IconButton size="small" onClick={() => openToolBuilder()}>
                      <AddIcon />
                    </IconButton>
                  </Tooltip>
                </Box>
                
                <List dense>
                  {customTools.map(tool => (
                    <ListItem 
                      key={tool.id} 
                      button 
                      onClick={() => toggleToolSelection(tool.id)}
                      sx={{ 
                        borderRadius: 1, 
                        mb: 0.5,
                        bgcolor: selectedTools.includes(tool.id) ? 'action.selected' : 'transparent'
                      }}
                    >
                      <ListItemText 
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <FunctionIcon fontSize="small" color={selectedTools.includes(tool.id) ? 'primary' : 'disabled'} />
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>{tool.name}</Typography>
                          </Box>
                        }
                        secondary={tool.description}
                      />
                      <ListItemSecondaryAction>
                        <IconButton size="small" onClick={(e) => { e.stopPropagation(); openToolBuilder(tool); }}>
                          <EditIcon fontSize="small" />
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
                
                {customTools.length === 0 && (
                  <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                    No tools defined. Create one to get started.
                  </Typography>
                )}
              </CardContent>
            </Card>
            
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Settings</Typography>
                <FormControlLabel
                  control={
                    <Switch 
                      checked={useMockResponses} 
                      onChange={(e) => setUseMockResponses(e.target.checked)} 
                    />
                  }
                  label="Use mock responses"
                />
                <Typography variant="caption" color="text.secondary" display="block">
                  When enabled, tool calls will use predefined mock responses to continue the conversation
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          {/* Chat Area */}
          <Grid item xs={12} md={8}>
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Test Message</Typography>
                <TextField
                  fullWidth
                  multiline
                  rows={3}
                  placeholder="Enter a message that should trigger tool usage..."
                  value={playgroundMessage}
                  onChange={(e) => setPlaygroundMessage(e.target.value)}
                  sx={{ mb: 2 }}
                />
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    variant="contained"
                    startIcon={playgroundLoading ? <CircularProgress size={20} /> : <SendIcon />}
                    onClick={runPlaygroundTest}
                    disabled={playgroundLoading || !playgroundMessage.trim() || selectedTools.length === 0}
                  >
                    Send
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<RefreshIcon />}
                    onClick={() => { setPlaygroundResponse(null); setToolCalls([]); }}
                  >
                    Clear
                  </Button>
                </Box>
                
                {selectedTools.length === 0 && (
                  <Alert severity="info" sx={{ mt: 2 }}>
                    Select at least one tool from the list to test function calling
                  </Alert>
                )}
              </CardContent>
            </Card>
            
            {/* Tool Calls */}
            {toolCalls.length > 0 && (
              <Card sx={{ mb: 2 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Tool Calls</Typography>
                  {toolCalls.slice(0, 5).map((call, idx) => (
                    <Accordion key={call.id + idx} defaultExpanded={idx === 0}>
                      <AccordionSummary expandIcon={<ExpandIcon />}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <FunctionIcon color="primary" />
                          <Typography sx={{ fontWeight: 500 }}>{call.name}</Typography>
                          <Chip label="called" size="small" color="success" />
                        </Box>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Typography variant="subtitle2" gutterBottom>Arguments:</Typography>
                        <Paper sx={{ p: 1, bgcolor: 'grey.900', mb: 1 }}>
                          <pre style={{ margin: 0, fontSize: '0.8rem', overflow: 'auto' }}>
                            {JSON.stringify(call.arguments, null, 2)}
                          </pre>
                        </Paper>
                        {call.response && (
                          <>
                            <Typography variant="subtitle2" gutterBottom>Mock Response:</Typography>
                            <Paper sx={{ p: 1, bgcolor: 'grey.900' }}>
                              <pre style={{ margin: 0, fontSize: '0.8rem', overflow: 'auto' }}>
                                {call.response}
                              </pre>
                            </Paper>
                          </>
                        )}
                      </AccordionDetails>
                    </Accordion>
                  ))}
                </CardContent>
              </Card>
            )}
            
            {/* Response */}
            {playgroundResponse && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Response</Typography>
                  <Paper sx={{ p: 2, bgcolor: 'grey.900' }}>
                    <Typography sx={{ whiteSpace: 'pre-wrap' }}>{playgroundResponse}</Typography>
                  </Paper>
                </CardContent>
              </Card>
            )}
          </Grid>
        </Grid>
      )}

      {/* Tool Builder Tab */}
      {tabValue === 1 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Custom Tools</Typography>
                  <Button startIcon={<AddIcon />} onClick={() => openToolBuilder()}>
                    New Tool
                  </Button>
                </Box>
                
                {customTools.map(tool => (
                  <Paper key={tool.id} sx={{ p: 2, mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <Box>
                        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>{tool.name}</Typography>
                        <Typography variant="body2" color="text.secondary">{tool.description}</Typography>
                        <Box sx={{ mt: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                          {tool.parameters.map(p => (
                            <Chip 
                              key={p.name} 
                              label={`${p.name}: ${p.type}${p.required ? '*' : ''}`} 
                              size="small" 
                              variant="outlined"
                            />
                          ))}
                        </Box>
                      </Box>
                      <Box>
                        <IconButton size="small" onClick={() => openToolBuilder(tool)}>
                          <EditIcon />
                        </IconButton>
                        <IconButton size="small" color="error" onClick={() => deleteTool(tool.id)}>
                          <DeleteIcon />
                        </IconButton>
                      </Box>
                    </Box>
                  </Paper>
                ))}
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>OpenAI Schema Preview</Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  This is the JSON schema that will be sent to the model
                </Typography>
                <Paper sx={{ p: 2, bgcolor: 'grey.900', maxHeight: 500, overflow: 'auto' }}>
                  <pre style={{ margin: 0, fontSize: '0.75rem' }}>
                    {JSON.stringify(
                      customTools.map(convertToolToOpenAI),
                      null,
                      2
                    )}
                  </pre>
                </Paper>
                <Button 
                  sx={{ mt: 2 }}
                  startIcon={<CopyIcon />}
                  onClick={() => navigator.clipboard.writeText(JSON.stringify(customTools.map(convertToolToOpenAI), null, 2))}
                >
                  Copy Schema
                </Button>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Benchmark Tab */}
      {tabValue === 2 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Benchmark Configuration</Typography>
                
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
            
            {quickTestResult && (
              <Card sx={{ mt: 2 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Quick Test Result</Typography>
                  <Alert 
                    severity={
                      quickTestResult.includes('SUCCESS') ? 'success' : 
                      quickTestResult.includes('PARTIAL') ? 'warning' : 'error'
                    }
                  >
                    {quickTestResult}
                  </Alert>
                </CardContent>
              </Card>
            )}
          </Grid>
          
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Benchmark Results</Typography>
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
                        <Typography variant="body2" color="text.secondary">Test Category</Typography>
                        <Typography variant="h6">{benchmarkStatus.test_category || 'N/A'}</Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">Max Examples</Typography>
                        <Typography variant="h6">{benchmarkStatus.max_examples || 'N/A'}</Typography>
                      </Grid>
                    </Grid>
                    
                    {(benchmarkStatus.status === 'running' || benchmarkStatus.progress) && (
                      <Box sx={{ mb: 3 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                          <Typography variant="body2" color="text.secondary">Progress</Typography>
                          <Typography variant="body2">
                            {benchmarkStatus.progress || 0}/{benchmarkStatus.total || '?'}
                          </Typography>
                        </Box>
                        <LinearProgress 
                          variant="determinate" 
                          value={benchmarkStatus.total ? (benchmarkStatus.progress || 0) / benchmarkStatus.total * 100 : 0} 
                        />
                      </Box>
                    )}
                    
                    {benchmarkStatus.results && (
                      <Grid container spacing={2}>
                        <Grid item xs={4}>
                          <Card variant="outlined">
                            <CardContent sx={{ textAlign: 'center' }}>
                              <Typography variant="h4" color="primary">
                                {benchmarkStatus.results.accuracy.toFixed(1)}%
                              </Typography>
                              <Typography variant="body2" color="text.secondary">Accuracy</Typography>
                            </CardContent>
                          </Card>
                        </Grid>
                        <Grid item xs={4}>
                          <Card variant="outlined">
                            <CardContent sx={{ textAlign: 'center' }}>
                              <Typography variant="h4" color="success.main">
                                {benchmarkStatus.results.correct}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">Correct</Typography>
                            </CardContent>
                          </Card>
                        </Grid>
                        <Grid item xs={4}>
                          <Card variant="outlined">
                            <CardContent sx={{ textAlign: 'center' }}>
                              <Typography variant="h4" color="error.main">
                                {benchmarkStatus.results.total - benchmarkStatus.results.correct}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">Errors</Typography>
                            </CardContent>
                          </Card>
                        </Grid>
                      </Grid>
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
      )}

      {/* Tool Builder Dialog */}
      <Dialog open={toolBuilderOpen} onClose={() => setToolBuilderOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>{editingTool ? 'Edit Tool' : 'Create New Tool'}</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <TextField
              label="Function Name"
              value={newTool.name}
              onChange={(e) => setNewTool(prev => ({ ...prev, name: e.target.value.replace(/\s/g, '_') }))}
              placeholder="e.g., get_weather"
              helperText="Use snake_case, no spaces"
              fullWidth
            />
            
            <TextField
              label="Description"
              value={newTool.description}
              onChange={(e) => setNewTool(prev => ({ ...prev, description: e.target.value }))}
              placeholder="Describe what this function does"
              multiline
              rows={2}
              fullWidth
            />
            
            <Divider />
            
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="subtitle1">Parameters</Typography>
              <Button startIcon={<AddIcon />} size="small" onClick={addParameter}>
                Add Parameter
              </Button>
            </Box>
            
            {newTool.parameters.map((param, idx) => (
              <Paper key={idx} sx={{ p: 2 }}>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={4}>
                    <TextField
                      label="Name"
                      value={param.name}
                      onChange={(e) => updateParameter(idx, 'name', e.target.value)}
                      size="small"
                      fullWidth
                    />
                  </Grid>
                  <Grid item xs={12} sm={3}>
                    <FormControl fullWidth size="small">
                      <InputLabel>Type</InputLabel>
                      <Select
                        value={param.type}
                        label="Type"
                        onChange={(e) => updateParameter(idx, 'type', e.target.value)}
                      >
                        <MenuItem value="string">string</MenuItem>
                        <MenuItem value="number">number</MenuItem>
                        <MenuItem value="boolean">boolean</MenuItem>
                        <MenuItem value="array">array</MenuItem>
                        <MenuItem value="object">object</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} sm={3}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={param.required}
                          onChange={(e) => updateParameter(idx, 'required', e.target.checked)}
                          size="small"
                        />
                      }
                      label="Required"
                    />
                  </Grid>
                  <Grid item xs={12} sm={2}>
                    <IconButton color="error" onClick={() => removeParameter(idx)}>
                      <DeleteIcon />
                    </IconButton>
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      label="Description"
                      value={param.description}
                      onChange={(e) => updateParameter(idx, 'description', e.target.value)}
                      size="small"
                      fullWidth
                    />
                  </Grid>
                </Grid>
              </Paper>
            ))}
            
            <Divider />
            
            <TextField
              label="Mock Response (JSON)"
              value={newTool.mockResponse}
              onChange={(e) => setNewTool(prev => ({ ...prev, mockResponse: e.target.value }))}
              placeholder='{"result": "example"}'
              multiline
              rows={3}
              fullWidth
              helperText="JSON response to return when this tool is called (for testing)"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setToolBuilderOpen(false)}>Cancel</Button>
          <Button 
            variant="contained" 
            onClick={saveTool}
            disabled={!newTool.name.trim()}
            startIcon={<SaveIcon />}
          >
            Save Tool
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  )
}
