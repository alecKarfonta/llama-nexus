import React, { useState } from 'react'
import {
  Box,
  Typography,
  Card,
  CardContent,
  Tabs,
  Tab,
  Chip,
  IconButton,
  Tooltip,
  TextField,
  Button,
  Alert,
  Collapse,
  alpha,
  Divider,
  CircularProgress,
} from '@mui/material'
import {
  ContentCopy as CopyIcon,
  ExpandMore as ExpandIcon,
  ExpandLess as CollapseIcon,
  PlayArrow as RunIcon,
  Code as CodeIcon,
  Api as ApiIcon,
  Storage as StorageIcon,
  Chat as ChatIcon,
  Speed as SpeedIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material'

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

const categoryIcons: Record<string, React.ReactNode> = {
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

interface EndpointCardProps {
  endpoint: ApiEndpoint
}

const EndpointCard: React.FC<EndpointCardProps> = ({ endpoint }) => {
  const [expanded, setExpanded] = useState(false)
  const [response, setResponse] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const handleTryIt = async () => {
    setLoading(true)
    setError(null)
    try {
      const options: RequestInit = {
        method: endpoint.method,
        headers: { 'Content-Type': 'application/json' },
      }
      if (endpoint.requestBody && endpoint.method !== 'GET') {
        options.body = JSON.stringify(endpoint.requestBody)
      }
      const res = await fetch(endpoint.path, options)
      const data = await res.json()
      setResponse(JSON.stringify(data, null, 2))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card
      sx={{
        mb: 2,
        background: 'linear-gradient(145deg, rgba(30, 30, 63, 0.6) 0%, rgba(26, 26, 46, 0.8) 100%)',
        backdropFilter: 'blur(12px)',
        border: '1px solid rgba(255, 255, 255, 0.06)',
        borderRadius: 2,
        transition: 'all 0.2s ease-in-out',
        '&:hover': {
          borderColor: alpha(methodColors[endpoint.method], 0.3),
        },
      }}
    >
      <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
        {/* Header */}
        <Box
          sx={{ display: 'flex', alignItems: 'center', gap: 2, cursor: 'pointer' }}
          onClick={() => setExpanded(!expanded)}
        >
          <Chip
            label={endpoint.method}
            size="small"
            sx={{
              bgcolor: alpha(methodColors[endpoint.method], 0.15),
              color: methodColors[endpoint.method],
              fontWeight: 700,
              fontSize: '0.6875rem',
              minWidth: 60,
            }}
          />
          <Typography
            variant="body2"
            sx={{
              fontFamily: 'monospace',
              color: 'text.primary',
              fontWeight: 500,
              flex: 1,
            }}
          >
            {endpoint.path}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ flex: 2, display: { xs: 'none', md: 'block' } }}>
            {endpoint.description}
          </Typography>
          <IconButton size="small" sx={{ color: 'text.secondary' }}>
            {expanded ? <CollapseIcon /> : <ExpandIcon />}
          </IconButton>
        </Box>

        {/* Expanded Content */}
        <Collapse in={expanded}>
          <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid rgba(255, 255, 255, 0.06)' }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              {endpoint.description}
            </Typography>

            {/* Parameters */}
            {endpoint.parameters && endpoint.parameters.length > 0 && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
                  Parameters
                </Typography>
                <Box sx={{ pl: 2 }}>
                  {endpoint.parameters.map((param) => (
                    <Box key={param.name} sx={{ mb: 0.5 }}>
                      <Typography variant="body2" component="span" sx={{ fontFamily: 'monospace', color: '#06b6d4' }}>
                        {param.name}
                      </Typography>
                      <Typography variant="body2" component="span" color="text.secondary">
                        {' '}({param.type}){param.required ? ' *' : ''} - {param.description}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </Box>
            )}

            {/* Request Body */}
            {endpoint.requestBody && (
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="subtitle2" sx={{ color: 'text.secondary' }}>
                    Request Body
                  </Typography>
                  <Tooltip title="Copy">
                    <IconButton size="small" onClick={() => handleCopy(JSON.stringify(endpoint.requestBody, null, 2))}>
                      <CopyIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  </Tooltip>
                </Box>
                <Box
                  sx={{
                    bgcolor: 'rgba(0, 0, 0, 0.3)',
                    borderRadius: 1,
                    p: 1.5,
                    fontFamily: 'monospace',
                    fontSize: '0.75rem',
                    overflow: 'auto',
                    maxHeight: 200,
                  }}
                >
                  <pre style={{ margin: 0 }}>{JSON.stringify(endpoint.requestBody, null, 2)}</pre>
                </Box>
              </Box>
            )}

            {/* Response Example */}
            {endpoint.responseExample && (
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="subtitle2" sx={{ color: 'text.secondary' }}>
                    Response Example
                  </Typography>
                  <Tooltip title="Copy">
                    <IconButton size="small" onClick={() => handleCopy(JSON.stringify(endpoint.responseExample, null, 2))}>
                      <CopyIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  </Tooltip>
                </Box>
                <Box
                  sx={{
                    bgcolor: 'rgba(0, 0, 0, 0.3)',
                    borderRadius: 1,
                    p: 1.5,
                    fontFamily: 'monospace',
                    fontSize: '0.75rem',
                    overflow: 'auto',
                    maxHeight: 200,
                  }}
                >
                  <pre style={{ margin: 0 }}>{JSON.stringify(endpoint.responseExample, null, 2)}</pre>
                </Box>
              </Box>
            )}

            {/* Try It Button */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Button
                variant="contained"
                size="small"
                startIcon={loading ? <CircularProgress size={16} /> : <RunIcon />}
                onClick={handleTryIt}
                disabled={loading}
                sx={{
                  bgcolor: methodColors[endpoint.method],
                  '&:hover': { bgcolor: alpha(methodColors[endpoint.method], 0.8) },
                }}
              >
                Try It
              </Button>
            </Box>

            {/* Response */}
            {response && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" sx={{ color: 'text.secondary', mb: 1 }}>
                  Response
                </Typography>
                <Box
                  sx={{
                    bgcolor: alpha('#10b981', 0.1),
                    border: `1px solid ${alpha('#10b981', 0.2)}`,
                    borderRadius: 1,
                    p: 1.5,
                    fontFamily: 'monospace',
                    fontSize: '0.75rem',
                    overflow: 'auto',
                    maxHeight: 300,
                  }}
                >
                  <pre style={{ margin: 0 }}>{response}</pre>
                </Box>
              </Box>
            )}

            {/* Error */}
            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}
          </Box>
        </Collapse>
      </CardContent>
    </Card>
  )
}

const ApiDocsPage: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState(0)

  const filteredEndpoints = apiEndpoints.filter(
    (endpoint) => endpoint.category === categories[selectedCategory]
  )

  return (
    <Box
      sx={{
        width: '100%',
        maxWidth: '100vw',
        overflow: 'hidden',
        px: { xs: 2, sm: 3, md: 4 },
        py: 3,
        boxSizing: 'border-box',
      }}
    >
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
          <Typography
            variant="h1"
            sx={{
              fontWeight: 700,
              fontSize: { xs: '1.5rem', sm: '1.75rem', md: '2rem' },
              background: 'linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            API Documentation
          </Typography>
          <Chip
            label="v1"
            size="small"
            sx={{
              height: 22,
              bgcolor: alpha('#6366f1', 0.1),
              border: `1px solid ${alpha('#6366f1', 0.2)}`,
              color: '#818cf8',
              fontWeight: 600,
              fontSize: '0.6875rem',
            }}
          />
        </Box>
        <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 600 }}>
          Interactive API documentation for Llama Nexus. Explore endpoints, view examples, and test requests directly.
        </Typography>
      </Box>

      {/* Base URL */}
      <Card
        sx={{
          mb: 3,
          background: 'linear-gradient(145deg, rgba(30, 30, 63, 0.6) 0%, rgba(26, 26, 46, 0.8) 100%)',
          backdropFilter: 'blur(12px)',
          border: '1px solid rgba(255, 255, 255, 0.06)',
          borderRadius: 2,
        }}
      >
        <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Base URL:
            </Typography>
            <Typography
              variant="body2"
              sx={{
                fontFamily: 'monospace',
                bgcolor: 'rgba(0, 0, 0, 0.3)',
                px: 1.5,
                py: 0.5,
                borderRadius: 1,
              }}
            >
              {window.location.origin}
            </Typography>
            <Tooltip title="Copy">
              <IconButton size="small" onClick={() => navigator.clipboard.writeText(window.location.origin)}>
                <CopyIcon sx={{ fontSize: 16 }} />
              </IconButton>
            </Tooltip>
          </Box>
        </CardContent>
      </Card>

      {/* Category Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'rgba(255, 255, 255, 0.06)', mb: 3 }}>
        <Tabs
          value={selectedCategory}
          onChange={(_, value) => setSelectedCategory(value)}
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            '& .MuiTabs-indicator': {
              background: 'linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%)',
            },
          }}
        >
          {categories.map((category, index) => (
            <Tab
              key={category}
              label={category}
              icon={categoryIcons[category]}
              iconPosition="start"
              sx={{
                textTransform: 'none',
                minHeight: 48,
                fontWeight: 500,
                '& .MuiSvgIcon-root': { fontSize: 18, mr: 0.5 },
              }}
            />
          ))}
        </Tabs>
      </Box>

      {/* Endpoints */}
      <Box>
        {filteredEndpoints.length > 0 ? (
          filteredEndpoints.map((endpoint, index) => (
            <EndpointCard key={`${endpoint.method}-${endpoint.path}-${index}`} endpoint={endpoint} />
          ))
        ) : (
          <Typography color="text.secondary">No endpoints in this category.</Typography>
        )}
      </Box>
    </Box>
  )
}

export default ApiDocsPage
