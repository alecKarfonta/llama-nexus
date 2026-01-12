import React, { useState, useEffect, useCallback } from 'react'
import {
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  CardHeader,
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
  Slider,
  Switch,
  FormControlLabel,
  Collapse,
} from '@mui/material'
import {
  PlayArrow as RunIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Compare as CompareIcon,
  ContentCopy as CopyIcon,
  Download as DownloadIcon,
  ExpandMore as ExpandIcon,
  ExpandLess as CollapseIcon,
  Speed as SpeedIcon,
  Timer as TimerIcon,
  Memory as TokenIcon,
  Settings as SettingsIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material'
import { settingsManager } from '@/utils/settings'

interface ModelConfig {
  id: string
  name: string
  endpoint: string
  apiKey: string
  temperature: number
  maxTokens: number
  topP: number
}

interface ComparisonResult {
  modelId: string
  modelName: string
  response: string
  promptTokens: number
  completionTokens: number
  totalTokens: number
  timeToFirstToken: number
  totalTime: number
  tokensPerSecond: number
  error?: string
}

interface ComparisonRun {
  id: string
  prompt: string
  systemPrompt: string
  results: ComparisonResult[]
  timestamp: string
}

const DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant.'

const DEFAULT_MODELS: ModelConfig[] = [
  {
    id: '1',
    name: 'Model A',
    endpoint: '/v1/chat/completions',
    apiKey: '',
    temperature: 0.7,
    maxTokens: 1024,
    topP: 0.9,
  },
  {
    id: '2',
    name: 'Model B',
    endpoint: '/v1/chat/completions',
    apiKey: '',
    temperature: 0.7,
    maxTokens: 1024,
    topP: 0.9,
  },
]

export default function ModelComparisonPage() {
  // State
  const [models, setModels] = useState<ModelConfig[]>(DEFAULT_MODELS)
  const [prompt, setPrompt] = useState('')
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT)
  const [showSystemPrompt, setShowSystemPrompt] = useState(false)
  
  const [currentResults, setCurrentResults] = useState<ComparisonResult[]>([])
  const [runHistory, setRunHistory] = useState<ComparisonRun[]>([])
  
  const [running, setRunning] = useState(false)
  const [runningModels, setRunningModels] = useState<Set<string>>(new Set())
  const [error, setError] = useState<string | null>(null)
  
  const [expandedResults, setExpandedResults] = useState<Set<string>>(new Set())
  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false)
  const [selectedModelIndex, setSelectedModelIndex] = useState<number | null>(null)
  
  // Load history from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem('model-comparison-history')
      if (saved) {
        setRunHistory(JSON.parse(saved))
      }
      const savedModels = localStorage.getItem('model-comparison-configs')
      if (savedModels) {
        setModels(JSON.parse(savedModels))
      }
    } catch (e) {
      console.error('Failed to load comparison history:', e)
    }
  }, [])
  
  // Save models to localStorage
  const saveModels = (newModels: ModelConfig[]) => {
    setModels(newModels)
    try {
      localStorage.setItem('model-comparison-configs', JSON.stringify(newModels))
    } catch (e) {
      console.error('Failed to save model configs:', e)
    }
  }
  
  // Save history to localStorage
  const saveHistory = (newHistory: ComparisonRun[]) => {
    setRunHistory(newHistory)
    try {
      localStorage.setItem('model-comparison-history', JSON.stringify(newHistory.slice(0, 50)))
    } catch (e) {
      console.error('Failed to save comparison history:', e)
    }
  }
  
  // Add a new model slot
  const addModel = () => {
    const newId = String(Date.now())
    saveModels([
      ...models,
      {
        id: newId,
        name: `Model ${models.length + 1}`,
        endpoint: '/v1/chat/completions',
        apiKey: '',
        temperature: 0.7,
        maxTokens: 1024,
        topP: 0.9,
      },
    ])
  }
  
  // Remove a model slot
  const removeModel = (index: number) => {
    if (models.length <= 2) {
      setError('Need at least 2 models for comparison')
      return
    }
    const newModels = models.filter((_, i) => i !== index)
    saveModels(newModels)
  }
  
  // Update a model config
  const updateModel = (index: number, updates: Partial<ModelConfig>) => {
    const newModels = [...models]
    newModels[index] = { ...newModels[index], ...updates }
    saveModels(newModels)
  }
  
  // Run comparison for a single model
  const runModelComparison = async (model: ModelConfig): Promise<ComparisonResult> => {
    const startTime = performance.now()
    let firstTokenTime = 0
    let response = ''
    let promptTokens = 0
    let completionTokens = 0
    
    try {
      const res = await fetch(model.endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(model.apiKey ? { 'Authorization': `Bearer ${model.apiKey}` } : {}),
        },
        body: JSON.stringify({
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: prompt },
          ],
          temperature: model.temperature,
          max_tokens: model.maxTokens,
          top_p: model.topP,
          stream: false,
        }),
      })
      
      if (!res.ok) {
        const errorText = await res.text()
        throw new Error(`HTTP ${res.status}: ${errorText.substring(0, 200)}`)
      }
      
      const data = await res.json()
      firstTokenTime = performance.now() - startTime
      
      response = data.choices?.[0]?.message?.content || ''
      promptTokens = data.usage?.prompt_tokens || 0
      completionTokens = data.usage?.completion_tokens || 0
      
      const totalTime = performance.now() - startTime
      const tokensPerSecond = completionTokens > 0 ? (completionTokens / (totalTime / 1000)) : 0
      
      return {
        modelId: model.id,
        modelName: model.name,
        response,
        promptTokens,
        completionTokens,
        totalTokens: promptTokens + completionTokens,
        timeToFirstToken: firstTokenTime,
        totalTime,
        tokensPerSecond,
      }
    } catch (err: any) {
      const totalTime = performance.now() - startTime
      return {
        modelId: model.id,
        modelName: model.name,
        response: '',
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
        timeToFirstToken: 0,
        totalTime,
        tokensPerSecond: 0,
        error: err.message,
      }
    }
  }
  
  // Run comparison for all models
  const runComparison = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt')
      return
    }
    
    setError(null)
    setRunning(true)
    setCurrentResults([])
    setRunningModels(new Set(models.map(m => m.id)))
    
    // Run all models in parallel
    const results = await Promise.all(
      models.map(async (model) => {
        const result = await runModelComparison(model)
        setRunningModels(prev => {
          const next = new Set(prev)
          next.delete(model.id)
          return next
        })
        setCurrentResults(prev => [...prev, result])
        return result
      })
    )
    
    // Save to history
    const run: ComparisonRun = {
      id: String(Date.now()),
      prompt,
      systemPrompt,
      results,
      timestamp: new Date().toISOString(),
    }
    saveHistory([run, ...runHistory])
    
    setRunning(false)
  }
  
  // Export results
  const exportResults = () => {
    if (currentResults.length === 0) return
    
    const exportData = {
      prompt,
      systemPrompt,
      timestamp: new Date().toISOString(),
      results: currentResults.map(r => ({
        model: r.modelName,
        response: r.response,
        metrics: {
          promptTokens: r.promptTokens,
          completionTokens: r.completionTokens,
          totalTime: r.totalTime,
          tokensPerSecond: r.tokensPerSecond,
        },
        error: r.error,
      })),
    }
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `model-comparison-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }
  
  // Copy response to clipboard
  const copyResponse = (text: string) => {
    navigator.clipboard.writeText(text)
  }
  
  // Toggle result expansion
  const toggleExpand = (modelId: string) => {
    setExpandedResults(prev => {
      const next = new Set(prev)
      if (next.has(modelId)) {
        next.delete(modelId)
      } else {
        next.add(modelId)
      }
      return next
    })
  }
  
  // Format time
  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`
    return `${(ms / 1000).toFixed(2)}s`
  }
  
  // Load a previous run
  const loadRun = (run: ComparisonRun) => {
    setPrompt(run.prompt)
    setSystemPrompt(run.systemPrompt)
    setCurrentResults(run.results)
  }

  return (
    <Box sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
          <CompareIcon /> Model Comparison
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title="Export Results">
            <span>
              <IconButton onClick={exportResults} disabled={currentResults.length === 0}>
                <DownloadIcon />
              </IconButton>
            </span>
          </Tooltip>
          <Button
            variant="outlined"
            startIcon={<AddIcon />}
            onClick={addModel}
            size="small"
          >
            Add Model
          </Button>
        </Box>
      </Box>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      <Grid container spacing={2} sx={{ flexGrow: 1, overflow: 'hidden' }}>
        {/* Left Panel - Configuration */}
        <Grid item xs={12} md={4} sx={{ display: 'flex', flexDirection: 'column' }}>
          <Paper sx={{ p: 2, flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'auto' }}>
            {/* Model Configurations */}
            <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 600 }}>
              Model Configurations
            </Typography>
            
            {models.map((model, index) => (
              <Card key={model.id} variant="outlined" sx={{ mb: 1.5 }}>
                <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <TextField
                      value={model.name}
                      onChange={(e) => updateModel(index, { name: e.target.value })}
                      size="small"
                      variant="standard"
                      sx={{ fontWeight: 600 }}
                      inputProps={{ style: { fontWeight: 600 } }}
                    />
                    <Box>
                      <Tooltip title="Settings">
                        <IconButton
                          size="small"
                          onClick={() => {
                            setSelectedModelIndex(index)
                            setSettingsDialogOpen(true)
                          }}
                        >
                          <SettingsIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Remove">
                        <IconButton
                          size="small"
                          onClick={() => removeModel(index)}
                          disabled={models.length <= 2}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </Box>
                  
                  <TextField
                    label="Endpoint"
                    value={model.endpoint}
                    onChange={(e) => updateModel(index, { endpoint: e.target.value })}
                    size="small"
                    fullWidth
                    sx={{ mb: 1 }}
                  />
                  
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Chip
                      size="small"
                      label={`T: ${model.temperature}`}
                      variant="outlined"
                    />
                    <Chip
                      size="small"
                      label={`Max: ${model.maxTokens}`}
                      variant="outlined"
                    />
                    {model.apiKey && (
                      <Chip
                        size="small"
                        label="API Key Set"
                        color="success"
                        variant="outlined"
                      />
                    )}
                  </Box>
                </CardContent>
              </Card>
            ))}
            
            <Divider sx={{ my: 2 }} />
            
            {/* Prompt Input */}
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                System Prompt
              </Typography>
              <IconButton
                size="small"
                onClick={() => setShowSystemPrompt(!showSystemPrompt)}
              >
                {showSystemPrompt ? <CollapseIcon /> : <ExpandIcon />}
              </IconButton>
            </Box>
            
            <Collapse in={showSystemPrompt}>
              <TextField
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                multiline
                rows={3}
                fullWidth
                size="small"
                sx={{ mb: 2 }}
              />
            </Collapse>
            
            <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 600 }}>
              Prompt
            </Typography>
            <TextField
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              multiline
              rows={4}
              fullWidth
              placeholder="Enter your prompt here..."
              sx={{ mb: 2 }}
            />
            
            <Button
              variant="contained"
              startIcon={running ? <CircularProgress size={20} color="inherit" /> : <RunIcon />}
              onClick={runComparison}
              disabled={running || !prompt.trim()}
              fullWidth
            >
              {running ? 'Running...' : 'Run Comparison'}
            </Button>
            
            {/* History */}
            {runHistory.length > 0 && (
              <>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 600 }}>
                  Recent Comparisons
                </Typography>
                <List dense sx={{ maxHeight: 200, overflow: 'auto' }}>
                  {runHistory.slice(0, 10).map((run) => (
                    <ListItem
                      key={run.id}
                      button
                      onClick={() => loadRun(run)}
                      sx={{ borderRadius: 1 }}
                    >
                      <ListItemText
                        primary={run.prompt.substring(0, 50) + (run.prompt.length > 50 ? '...' : '')}
                        secondary={new Date(run.timestamp).toLocaleString()}
                        primaryTypographyProps={{ fontSize: '0.875rem' }}
                        secondaryTypographyProps={{ fontSize: '0.75rem' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </>
            )}
          </Paper>
        </Grid>
        
        {/* Right Panel - Results */}
        <Grid item xs={12} md={8} sx={{ display: 'flex', flexDirection: 'column' }}>
          <Paper sx={{ p: 2, flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'auto' }}>
            <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
              Comparison Results
            </Typography>
            
            {/* Metrics Summary */}
            {currentResults.length > 0 && (
              <TableContainer sx={{ mb: 2 }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Model</TableCell>
                      <TableCell align="right">Prompt Tokens</TableCell>
                      <TableCell align="right">Output Tokens</TableCell>
                      <TableCell align="right">Time</TableCell>
                      <TableCell align="right">TPS</TableCell>
                      <TableCell align="right">Status</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {currentResults.map((result) => (
                      <TableRow key={result.modelId}>
                        <TableCell>{result.modelName}</TableCell>
                        <TableCell align="right">{result.promptTokens}</TableCell>
                        <TableCell align="right">{result.completionTokens}</TableCell>
                        <TableCell align="right">{formatTime(result.totalTime)}</TableCell>
                        <TableCell align="right">{result.tokensPerSecond.toFixed(1)}</TableCell>
                        <TableCell align="right">
                          {result.error ? (
                            <Chip size="small" label="Error" color="error" />
                          ) : (
                            <Chip size="small" label="OK" color="success" />
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
            
            {/* Response Cards */}
            <Grid container spacing={2}>
              {models.map((model) => {
                const result = currentResults.find(r => r.modelId === model.id)
                const isRunning = runningModels.has(model.id)
                const isExpanded = expandedResults.has(model.id)
                
                return (
                  <Grid item xs={12} md={6} key={model.id}>
                    <Card variant="outlined" sx={{ height: '100%' }}>
                      <CardHeader
                        title={model.name}
                        titleTypographyProps={{ variant: 'subtitle2' }}
                        action={
                          result && !result.error && (
                            <Box>
                              <Tooltip title="Copy Response">
                                <IconButton size="small" onClick={() => copyResponse(result.response)}>
                                  <CopyIcon fontSize="small" />
                                </IconButton>
                              </Tooltip>
                              <IconButton size="small" onClick={() => toggleExpand(model.id)}>
                                {isExpanded ? <CollapseIcon /> : <ExpandIcon />}
                              </IconButton>
                            </Box>
                          )
                        }
                        sx={{ py: 1, px: 2 }}
                      />
                      <Divider />
                      <CardContent sx={{ py: 1.5 }}>
                        {isRunning && (
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', py: 4 }}>
                            <CircularProgress size={24} />
                            <Typography sx={{ ml: 1 }} color="text.secondary">
                              Generating...
                            </Typography>
                          </Box>
                        )}
                        
                        {result && result.error && (
                          <Alert severity="error" sx={{ mb: 1 }}>
                            {result.error}
                          </Alert>
                        )}
                        
                        {result && !result.error && (
                          <>
                            {/* Metrics */}
                            <Box sx={{ display: 'flex', gap: 1, mb: 1, flexWrap: 'wrap' }}>
                              <Chip
                                size="small"
                                icon={<TokenIcon sx={{ fontSize: '14px !important' }} />}
                                label={`${result.completionTokens} tokens`}
                                variant="outlined"
                              />
                              <Chip
                                size="small"
                                icon={<TimerIcon sx={{ fontSize: '14px !important' }} />}
                                label={formatTime(result.totalTime)}
                                variant="outlined"
                              />
                              <Chip
                                size="small"
                                icon={<SpeedIcon sx={{ fontSize: '14px !important' }} />}
                                label={`${result.tokensPerSecond.toFixed(1)} t/s`}
                                variant="outlined"
                              />
                            </Box>
                            
                            {/* Response */}
                            <Box
                              sx={{
                                maxHeight: isExpanded ? 'none' : 200,
                                overflow: isExpanded ? 'visible' : 'auto',
                                bgcolor: 'grey.50',
                                p: 1.5,
                                borderRadius: 1,
                                fontSize: '0.875rem',
                                whiteSpace: 'pre-wrap',
                                wordBreak: 'break-word',
                              }}
                            >
                              {result.response || '(empty response)'}
                            </Box>
                          </>
                        )}
                        
                        {!isRunning && !result && (
                          <Typography color="text.secondary" sx={{ fontStyle: 'italic', py: 4, textAlign: 'center' }}>
                            Run comparison to see results
                          </Typography>
                        )}
                      </CardContent>
                    </Card>
                  </Grid>
                )
              })}
            </Grid>
          </Paper>
        </Grid>
      </Grid>
      
      {/* Model Settings Dialog */}
      <Dialog
        open={settingsDialogOpen}
        onClose={() => setSettingsDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          Model Settings
          {selectedModelIndex !== null && ` - ${models[selectedModelIndex]?.name}`}
        </DialogTitle>
        <DialogContent>
          {selectedModelIndex !== null && models[selectedModelIndex] && (
            <Box sx={{ pt: 1 }}>
              <TextField
                label="API Key"
                type="password"
                value={models[selectedModelIndex].apiKey}
                onChange={(e) => updateModel(selectedModelIndex, { apiKey: e.target.value })}
                fullWidth
                sx={{ mb: 2 }}
                placeholder="Leave empty for no authentication"
              />
              
              <Typography gutterBottom>Temperature: {models[selectedModelIndex].temperature}</Typography>
              <Slider
                value={models[selectedModelIndex].temperature}
                onChange={(_, v) => updateModel(selectedModelIndex, { temperature: v as number })}
                min={0}
                max={2}
                step={0.1}
                marks={[
                  { value: 0, label: '0' },
                  { value: 1, label: '1' },
                  { value: 2, label: '2' },
                ]}
                sx={{ mb: 3 }}
              />
              
              <Typography gutterBottom>Top P: {models[selectedModelIndex].topP}</Typography>
              <Slider
                value={models[selectedModelIndex].topP}
                onChange={(_, v) => updateModel(selectedModelIndex, { topP: v as number })}
                min={0}
                max={1}
                step={0.05}
                marks={[
                  { value: 0, label: '0' },
                  { value: 0.5, label: '0.5' },
                  { value: 1, label: '1' },
                ]}
                sx={{ mb: 3 }}
              />
              
              <TextField
                label="Max Tokens"
                type="number"
                value={models[selectedModelIndex].maxTokens}
                onChange={(e) => updateModel(selectedModelIndex, { maxTokens: parseInt(e.target.value) || 0 })}
                fullWidth
                inputProps={{ min: 1, max: 32768 }}
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}











