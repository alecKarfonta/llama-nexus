import React, { useEffect, useState } from 'react'
import {
  Box,
  Typography,
  Chip,
  IconButton,
  Tooltip,
  TextField,
  Button,
  Alert,
  ButtonBase,
  alpha,
  CircularProgress,
} from '@mui/material'
import {
  ContentCopy as CopyIcon,
  PlayArrow as RunIcon,
  Code as CodeIcon,
  Api as ApiIcon,
  Storage as StorageIcon,
  Chat as ChatIcon,
  Speed as SpeedIcon,
  Settings as SettingsIcon,
  Search as SearchIcon,
} from '@mui/icons-material'
import { useClipboard } from '@/hooks/useClipboard'

// API endpoint definitions
interface ApiEndpoint {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH'
  path: string
  description: string
  category: string
  requestBody?: object
  responseExample?: object
  parameters?: { name: string; type: string; description: string; required?: boolean }[]
}

const apiEndpoints: ApiEndpoint[] = [
  // Health & Status
  {
    method: 'GET',
    path: '/api/health',
    description: 'Check backend health status',
    category: 'System',
    responseExample: { status: 'healthy', timestamp: '2025-12-07T20:00:00Z', mode: 'docker' },
  },
  {
    method: 'GET',
    path: '/v1/service/status',
    description: 'Get detailed service status including model info and resources',
    category: 'System',
    responseExample: { running: true, pid: 1234, uptime: 3600, model: { name: 'model.gguf' } },
  },
  {
    method: 'GET',
    path: '/v1/resources',
    description: 'Get system resource usage (CPU, memory, GPU)',
    category: 'System',
    responseExample: { cpu: { percent: 25.5 }, memory: { total_mb: 32000, used_mb: 16000 }, gpu: { vram_used_mb: 8000 } },
  },
  // Chat Completions
  {
    method: 'POST',
    path: '/v1/chat/completions',
    description: 'Create a chat completion (OpenAI compatible)',
    category: 'Chat',
    requestBody: {
      model: 'model-name',
      messages: [{ role: 'user', content: 'Hello!' }],
      temperature: 0.7,
      max_tokens: 1024,
      stream: false,
    },
    responseExample: {
      id: 'chatcmpl-123',
      choices: [{ message: { role: 'assistant', content: 'Hello! How can I help?' } }],
      usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
    },
  },
  // Conversations
  {
    method: 'GET',
    path: '/api/v1/conversations',
    description: 'List all conversations',
    category: 'Conversations',
    parameters: [
      { name: 'limit', type: 'number', description: 'Max results to return', required: false },
      { name: 'include_archived', type: 'boolean', description: 'Include archived conversations', required: false },
    ],
    responseExample: { conversations: [{ id: '123', title: 'My Chat', message_count: 5 }] },
  },
  {
    method: 'POST',
    path: '/api/v1/conversations',
    description: 'Create a new conversation',
    category: 'Conversations',
    requestBody: { title: 'New Conversation', messages: [], system_prompt: 'You are helpful.' },
    responseExample: { id: '456', title: 'New Conversation', created_at: '2025-12-07T20:00:00Z' },
  },
  {
    method: 'GET',
    path: '/api/v1/conversations/{id}',
    description: 'Get a specific conversation by ID',
    category: 'Conversations',
    responseExample: { id: '123', title: 'My Chat', messages: [], settings: {} },
  },
  {
    method: 'DELETE',
    path: '/api/v1/conversations/{id}',
    description: 'Delete a conversation',
    category: 'Conversations',
    responseExample: { success: true },
  },
  // Prompts
  {
    method: 'GET',
    path: '/api/v1/prompts',
    description: 'List all prompt templates',
    category: 'Prompts',
    responseExample: { prompts: [{ id: '1', name: 'Code Review', category: 'development' }] },
  },
  {
    method: 'POST',
    path: '/api/v1/prompts',
    description: 'Create a new prompt template',
    category: 'Prompts',
    requestBody: { name: 'My Prompt', content: 'Analyze {{topic}}', category: 'general' },
    responseExample: { id: '2', name: 'My Prompt', created_at: '2025-12-07T20:00:00Z' },
  },
  // Registry
  {
    method: 'GET',
    path: '/api/v1/registry/models',
    description: 'List cached models in registry',
    category: 'Registry',
    responseExample: { models: [{ repo_id: 'TheBloke/model-GGUF', cached_at: '2025-12-07' }] },
  },
  {
    method: 'POST',
    path: '/api/v1/registry/models',
    description: 'Cache a model from HuggingFace',
    category: 'Registry',
    requestBody: { repo_id: 'TheBloke/Llama-2-7B-GGUF' },
    responseExample: { success: true, model: { repo_id: 'TheBloke/Llama-2-7B-GGUF' } },
  },
  // Benchmark
  {
    method: 'POST',
    path: '/api/v1/benchmark/run',
    description: 'Run an inference benchmark',
    category: 'Benchmark',
    requestBody: { prompt: 'Hello', num_runs: 5, max_tokens: 100 },
    responseExample: { benchmark_id: '123', status: 'running' },
  },
  {
    method: 'GET',
    path: '/api/v1/benchmark/{id}',
    description: 'Get benchmark results',
    category: 'Benchmark',
    responseExample: { id: '123', status: 'completed', results: { avg_tokens_per_second: 50 } },
  },
  // Batch Processing
  {
    method: 'POST',
    path: '/api/v1/batch/jobs',
    description: 'Create a batch processing job',
    category: 'Batch',
    requestBody: { prompts: ['prompt1', 'prompt2'], max_tokens: 100 },
    responseExample: { job_id: '456', status: 'pending', total_items: 2 },
  },
  {
    method: 'GET',
    path: '/api/v1/batch/jobs/{id}',
    description: 'Get batch job status and results',
    category: 'Batch',
    responseExample: { job_id: '456', status: 'completed', completed_items: 2, results: [] },
  },
  // VRAM Estimation
  {
    method: 'POST',
    path: '/api/v1/vram/estimate',
    description: 'Estimate VRAM requirements for a model',
    category: 'Tools',
    requestBody: { model_name: 'llama-7b-Q4_K_M.gguf', context_size: 4096 },
    responseExample: { total_gb: 5.2, model_gb: 4.0, kv_cache_gb: 1.0, fits_vram: true },
  },
  {
    method: 'GET',
    path: '/api/v1/vram/quantizations',
    description: 'Get supported quantization types',
    category: 'Tools',
    responseExample: { quantizations: { Q4_K_M: 4.5, Q8_0: 8.0, F16: 16.0 } },
  },
  // Models
  {
    method: 'GET',
    path: '/v1/models',
    description: 'List available models (OpenAI compatible)',
    category: 'Models',
    responseExample: { data: [{ id: 'model-1', object: 'model' }] },
  },
  {
    method: 'POST',
    path: '/v1/models/download',
    description: 'Download a model from HuggingFace',
    category: 'Models',
    requestBody: { repo_id: 'TheBloke/Llama-2-7B-GGUF', filename: 'model.Q4_K_M.gguf' },
    responseExample: { success: true, message: 'Download started' },
  },
]

const methodColors: Record<string, string> = {
  GET: '#10b981',
  POST: '#6366f1',
  PUT: '#f59e0b',
  DELETE: '#ef4444',
  PATCH: '#8b5cf6',
}

const categoryIcons: Record<string, React.ReactElement> = {
  System: <SettingsIcon />,
  Chat: <ChatIcon />,
  Conversations: <ChatIcon />,
  Prompts: <CodeIcon />,
  Registry: <StorageIcon />,
  Benchmark: <SpeedIcon />,
  Batch: <StorageIcon />,
  Tools: <ApiIcon />,
  Models: <StorageIcon />,
}

const categories = ['System', 'Chat', 'Conversations', 'Prompts', 'Registry', 'Benchmark', 'Batch', 'Tools', 'Models']

const endpointKey = (endpoint: ApiEndpoint) => `${endpoint.method}-${endpoint.path}`

const formatJson = (value?: object | string | null) => {
  if (!value) return ''
  return typeof value === 'string' ? value : JSON.stringify(value, null, 2)
}

const truncatePreview = (text: string, max = 1200) =>
  text.length > max ? `${text.slice(0, max).trimEnd()}\n...` : text

const buildCurl = (endpoint: ApiEndpoint) => {
  const parts = [`curl -X ${endpoint.method} "${endpoint.path}"`]
  if (endpoint.requestBody && endpoint.method !== 'GET') {
    parts.push('-H "Content-Type: application/json"')
    parts.push(`-d '${JSON.stringify(endpoint.requestBody)}'`)
  }
  return parts.join(' \\\n  ')
}

const CodePreview: React.FC<{
  title: string
  value?: object | string | null
  onCopy: (text: string) => void
  emptyText?: string
}> = ({ title, value, onCopy, emptyText = 'Not required' }) => {
  const text = truncatePreview(formatJson(value))

  return (
    <Box
      sx={{
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '8px',
        bgcolor: 'rgba(2, 6, 23, 0.52)',
        minWidth: 0,
      }}
    >
      <Box sx={{ px: 1.25, py: 0.75, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1 }}>
        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 800, textTransform: 'uppercase' }}>
          {title}
        </Typography>
        {text && (
          <Tooltip title={`Copy ${title.toLowerCase()}`}>
            <IconButton size="small" onClick={() => onCopy(formatJson(value))} sx={{ width: 26, height: 26 }}>
              <CopyIcon sx={{ fontSize: 15 }} />
            </IconButton>
          </Tooltip>
        )}
      </Box>
      <Box
        component="pre"
        sx={{
          m: 0,
          px: 1.25,
          pb: 1.25,
          color: text ? 'text.primary' : 'text.disabled',
          fontFamily: 'monospace',
          fontSize: '0.72rem',
          lineHeight: 1.45,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {text || emptyText}
      </Box>
    </Box>
  )
}

const ApiDocsPage: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState(0)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedEndpointKey, setSelectedEndpointKey] = useState(endpointKey(apiEndpoints[0]))
  const [response, setResponse] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const copyToClipboard = useClipboard()

  const activeCategory = categories[selectedCategory]
  const categoryEndpoints = apiEndpoints.filter(
    (endpoint) => endpoint.category === activeCategory
  )
  const filteredEndpoints = categoryEndpoints.filter(
    (endpoint) => {
      const query = searchTerm.trim().toLowerCase()
      if (!query) return true
      return [
        endpoint.method,
        endpoint.path,
        endpoint.description,
        endpoint.category,
      ].some(value => value.toLowerCase().includes(query))
    }
  )
  const selectedEndpoint = filteredEndpoints.find(endpoint => endpointKey(endpoint) === selectedEndpointKey) || filteredEndpoints[0]

  useEffect(() => {
    if (filteredEndpoints.length > 0 && !filteredEndpoints.some(endpoint => endpointKey(endpoint) === selectedEndpointKey)) {
      setSelectedEndpointKey(endpointKey(filteredEndpoints[0]))
      setResponse(null)
      setError(null)
    }
  }, [filteredEndpoints, selectedEndpointKey])

  const copyExample = (text: string) => {
    copyToClipboard(text, {
      successMessage: 'API example copied.',
      successTitle: 'Example copied',
    })
  }

  const handleTryIt = async () => {
    if (!selectedEndpoint) return
    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const options: RequestInit = {
        method: selectedEndpoint.method,
        headers: { 'Content-Type': 'application/json' },
      }
      if (selectedEndpoint.requestBody && selectedEndpoint.method !== 'GET') {
        options.body = JSON.stringify(selectedEndpoint.requestBody)
      }
      const res = await fetch(selectedEndpoint.path, options)
      const data = await res.json()
      setResponse(JSON.stringify(data, null, 2))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box
      sx={{
        width: '100%',
        maxWidth: '100vw',
        overflow: 'visible',
        px: { xs: 2, sm: 3, md: 4 },
        py: 1.25,
        boxSizing: 'border-box',
        display: 'flex',
        flexDirection: 'column',
        gap: 1,
      }}
    >
      <Box
        sx={{
          p: 1.25,
          borderRadius: '8px',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          bgcolor: 'rgba(15, 23, 42, 0.58)',
        }}
      >
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: { xs: 'flex-start', md: 'center' },
            flexDirection: { xs: 'column', md: 'row' },
            gap: 1.5,
          }}
        >
          <Box sx={{ minWidth: 0, display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
            <Typography variant="body2" sx={{ fontWeight: 800, mr: 0.5 }}>
              API Docs
            </Typography>
            <Chip
              size="small"
              icon={<ApiIcon sx={{ fontSize: 15 }} />}
              label="API console"
              sx={{
                height: 24,
                bgcolor: alpha('#14B8A6', 0.14),
                border: `1px solid ${alpha('#14B8A6', 0.24)}`,
                color: '#5EEAD4',
                fontWeight: 700,
                '& .MuiChip-icon': { color: '#5EEAD4' },
              }}
            />
            <Chip
              size="small"
              label={`${apiEndpoints.length} endpoints`}
              sx={{
                height: 24,
                bgcolor: alpha('#6366F1', 0.12),
                border: `1px solid ${alpha('#6366F1', 0.24)}`,
                color: '#A5B4FC',
                fontWeight: 600,
              }}
            />
            <Chip
              size="small"
              label={`${filteredEndpoints.length} visible`}
              sx={{
                height: 24,
                bgcolor: alpha('#10B981', 0.1),
                border: `1px solid ${alpha('#10B981', 0.22)}`,
                color: '#86EFAC',
                fontWeight: 600,
              }}
            />
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, alignSelf: { xs: 'stretch', md: 'center' } }}>
            <TextField
              size="small"
              placeholder="Search endpoints"
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
              sx={{ minWidth: { xs: 0, sm: 260 }, flex: { xs: 1, md: 'initial' } }}
              InputProps={{
                startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary', fontSize: 18 }} />,
              }}
            />
            <Tooltip title="Copy base URL">
              <IconButton
                size="small"
                onClick={() => copyToClipboard(window.location.origin, {
                  successMessage: 'Base URL copied.',
                  successTitle: 'URL copied',
                })}
                sx={{
                  width: 36,
                  height: 36,
                  borderRadius: '8px',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  bgcolor: 'rgba(255, 255, 255, 0.04)',
                  '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.08)' },
                }}
              >
                <CopyIcon sx={{ fontSize: 18 }} />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </Box>

      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', lg: '360px minmax(0, 1fr)' },
          gap: 1.25,
          alignItems: 'start',
        }}
      >
        <Box
          sx={{
            borderRadius: '8px',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            bgcolor: 'rgba(15, 23, 42, 0.58)',
            p: 1,
            overflow: 'visible',
          }}
        >
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
              gap: 0.5,
              mb: 1,
            }}
          >
            {categories.map((category, index) => {
              const selected = selectedCategory === index
              const color = selected ? '#8B5CF6' : 'text.secondary'
              return (
                <ButtonBase
                  key={category}
                  onClick={() => {
                    setSelectedCategory(index)
                    setResponse(null)
                    setError(null)
                  }}
                  sx={{
                    minHeight: 36,
                    borderRadius: '8px',
                    border: '1px solid',
                    borderColor: selected ? alpha('#8B5CF6', 0.55) : 'rgba(255, 255, 255, 0.08)',
                    bgcolor: selected ? alpha('#8B5CF6', 0.12) : 'rgba(255, 255, 255, 0.03)',
                    color,
                    px: 0.75,
                    gap: 0.5,
                    justifyContent: 'flex-start',
                    '& svg': { fontSize: 15 },
                  }}
                >
                  {categoryIcons[category]}
                  <Typography
                    variant="caption"
                    sx={{
                      minWidth: 0,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      fontWeight: 800,
                    }}
                  >
                    {category}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ ml: 'auto' }}>
                    {apiEndpoints.filter(endpoint => endpoint.category === category).length}
                  </Typography>
                </ButtonBase>
              )
            })}
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', px: 0.25, mb: 0.75 }}>
            <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 800, textTransform: 'uppercase' }}>
              Endpoints
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {filteredEndpoints.length}
            </Typography>
          </Box>

          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
            {filteredEndpoints.length > 0 ? (
              filteredEndpoints.map((endpoint) => {
                const selected = selectedEndpoint && endpointKey(endpoint) === endpointKey(selectedEndpoint)
                return (
                  <Box
                    key={endpointKey(endpoint)}
                    sx={{
                      minHeight: 36,
                      display: 'grid',
                      gridTemplateColumns: 'minmax(0, 188px) minmax(0, 1fr)',
                      alignItems: 'center',
                      gap: 0.65,
                      px: 0.25,
                      py: 0.25,
                      borderRadius: '8px',
                      border: '1px solid',
                      borderColor: selected ? alpha(methodColors[endpoint.method], 0.48) : 'transparent',
                      bgcolor: selected ? alpha(methodColors[endpoint.method], 0.1) : 'transparent',
                      '&:hover': {
                        bgcolor: alpha(methodColors[endpoint.method], 0.08),
                      },
                    }}
                  >
                    <ButtonBase
                      onClick={() => {
                        setSelectedEndpointKey(endpointKey(endpoint))
                        setResponse(null)
                        setError(null)
                      }}
                      title={`${endpoint.method} ${endpoint.path}`}
                      sx={{
                        minWidth: 0,
                        width: '100%',
                        height: 28,
                        alignItems: 'center',
                        justifyContent: 'flex-start',
                        gap: 0.5,
                        borderRadius: '7px',
                        px: 0.5,
                        textAlign: 'left',
                        '&:focus-visible': {
                          outline: `2px solid ${alpha(methodColors[endpoint.method], 0.45)}`,
                          outlineOffset: 2,
                        },
                      }}
                    >
                      <Box
                        sx={{
                          width: 38,
                          height: 18,
                          borderRadius: '6px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0,
                          bgcolor: alpha(methodColors[endpoint.method], 0.15),
                          color: methodColors[endpoint.method],
                          fontSize: '0.58rem',
                          fontWeight: 900,
                        }}
                      >
                        {endpoint.method}
                      </Box>
                      <Typography
                        variant="caption"
                        title={endpoint.path}
                        sx={{
                          minWidth: 0,
                          fontFamily: 'monospace',
                          fontWeight: 900,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {endpoint.path}
                      </Typography>
                    </ButtonBase>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{
                        minWidth: 0,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                      title={endpoint.description}
                    >
                      {endpoint.description}
                    </Typography>
                  </Box>
                )
              })
            ) : (
              <Box
                sx={{
                  minHeight: 120,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: '8px',
                  border: '1px dashed rgba(255, 255, 255, 0.14)',
                  color: 'text.secondary',
                  textAlign: 'center',
                  px: 2,
                }}
              >
                <Typography variant="body2">No endpoints match this search.</Typography>
              </Box>
            )}
          </Box>
        </Box>

        <Box
          sx={{
            borderRadius: '8px',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            bgcolor: 'rgba(15, 23, 42, 0.58)',
            p: 1.25,
            overflow: 'visible',
          }}
        >
          {selectedEndpoint ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25 }}>
              <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 1.25 }}>
                <Box sx={{ minWidth: 0 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mb: 0.75, minWidth: 0 }}>
                    <Chip
                      label={selectedEndpoint.method}
                      size="small"
                      sx={{
                        height: 24,
                        bgcolor: alpha(methodColors[selectedEndpoint.method], 0.15),
                        color: methodColors[selectedEndpoint.method],
                        fontWeight: 900,
                        minWidth: 58,
                      }}
                    />
                    <Typography
                      variant="h6"
                      title={selectedEndpoint.path}
                      sx={{ fontFamily: 'monospace', fontWeight: 900, fontSize: '1rem' }}
                      noWrap
                    >
                      {selectedEndpoint.path}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {selectedEndpoint.description}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexShrink: 0 }}>
                  <Tooltip title="Copy endpoint path">
                    <IconButton
                      size="small"
                      onClick={() => copyExample(selectedEndpoint.path)}
                      sx={{
                        width: 32,
                        height: 32,
                        borderRadius: '8px',
                        border: '1px solid rgba(255, 255, 255, 0.1)',
                      }}
                    >
                      <CopyIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  </Tooltip>
                  <Button
                    variant="contained"
                    size="small"
                    startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <RunIcon />}
                    onClick={handleTryIt}
                    disabled={loading}
                    sx={{
                      height: 32,
                      borderRadius: '8px',
                      textTransform: 'none',
                      fontWeight: 900,
                      boxShadow: 'none',
                      bgcolor: methodColors[selectedEndpoint.method],
                      '&:hover': { bgcolor: alpha(methodColors[selectedEndpoint.method], 0.82), boxShadow: 'none' },
                    }}
                  >
                    Try
                  </Button>
                </Box>
              </Box>

              {selectedEndpoint.parameters && selectedEndpoint.parameters.length > 0 && (
                <Box
                  sx={{
                    border: '1px solid rgba(255, 255, 255, 0.08)',
                    borderRadius: '8px',
                    p: 1,
                    bgcolor: 'rgba(255, 255, 255, 0.03)',
                  }}
                >
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.75, fontWeight: 800, textTransform: 'uppercase' }}>
                    Parameters
                  </Typography>
                  <Box sx={{ display: 'grid', gap: 0.5 }}>
                    {selectedEndpoint.parameters.map((param) => (
                      <Box key={param.name} sx={{ display: 'grid', gridTemplateColumns: '120px 90px minmax(0, 1fr)', gap: 1 }}>
                        <Typography variant="caption" sx={{ fontFamily: 'monospace', color: '#67E8F9', fontWeight: 800 }}>{param.name}</Typography>
                        <Typography variant="caption" color="text.secondary">{param.type}{param.required ? ' *' : ''}</Typography>
                        <Typography variant="caption" color="text.secondary" noWrap title={param.description}>{param.description}</Typography>
                      </Box>
                    ))}
                  </Box>
                </Box>
              )}

              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: { xs: '1fr', md: 'repeat(2, minmax(0, 1fr))' },
                  gap: 1,
                }}
              >
                {selectedEndpoint.requestBody && (
                  <CodePreview title="Request" value={selectedEndpoint.requestBody} onCopy={copyExample} />
                )}
                <CodePreview title="Example response" value={selectedEndpoint.responseExample} onCopy={copyExample} />
              </Box>

              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: { xs: '1fr', md: 'repeat(2, minmax(0, 1fr))' },
                  gap: 1,
                }}
              >
                <CodePreview title="cURL" value={buildCurl(selectedEndpoint)} onCopy={copyExample} />
                {response && (
                  <CodePreview title="Live response" value={response} onCopy={copyExample} />
                )}
              </Box>

              {error && (
                <Alert severity="error" sx={{ mt: 'auto' }}>
                  {error}
                </Alert>
              )}
            </Box>
          ) : (
            <Box
              sx={{
                minHeight: 120,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                borderRadius: '8px',
                border: '1px dashed rgba(255, 255, 255, 0.14)',
              }}
            >
              <Typography color="text.secondary">No endpoint selected.</Typography>
            </Box>
          )}
        </Box>
      </Box>
    </Box>
  )
}

export default ApiDocsPage
