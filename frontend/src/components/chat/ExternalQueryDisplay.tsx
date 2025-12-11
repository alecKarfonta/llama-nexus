/**
 * ExternalQueryDisplay Component
 * Displays external queries (tool calls, web searches, RAG queries) in a clean UI
 */

import React, { useState, memo } from 'react'
import {
  Box,
  Typography,
  Paper,
  Collapse,
  IconButton,
  Chip,
  LinearProgress,
  Divider,
  alpha,
  Tooltip,
} from '@mui/material'
import {
  ExpandMore as ExpandMoreIcon,
  Search as SearchIcon,
  Code as CodeIcon,
  Cloud as ApiIcon,
  Storage as DatabaseIcon,
  Calculate as CalculatorIcon,
  WbSunny as WeatherIcon,
  Computer as SystemIcon,
  Language as WebIcon,
  Psychology as RagIcon,
  Check as CheckIcon,
  Error as ErrorIcon,
  HourglassEmpty as PendingIcon,
  ContentCopy as CopyIcon,
} from '@mui/icons-material'

interface ToolCall {
  id: string
  type: string
  function: {
    name: string
    arguments: string
  }
}

interface ExternalQueryDisplayProps {
  toolCall: ToolCall
  result?: string
  status: 'pending' | 'executing' | 'success' | 'error'
  executionTime?: number
  defaultExpanded?: boolean
}

// Map tool names to icons and colors
const getToolMeta = (toolName: string): { icon: React.ReactNode; color: string; label: string } => {
  const toolMap: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
    'web_search': { icon: <WebIcon />, color: '#3b82f6', label: 'Web Search' },
    'search': { icon: <SearchIcon />, color: '#3b82f6', label: 'Search' },
    'code_execution': { icon: <CodeIcon />, color: '#10b981', label: 'Code Execution' },
    'execute_code': { icon: <CodeIcon />, color: '#10b981', label: 'Code Execution' },
    'python': { icon: <CodeIcon />, color: '#10b981', label: 'Python' },
    'calculator': { icon: <CalculatorIcon />, color: '#f59e0b', label: 'Calculator' },
    'get_weather': { icon: <WeatherIcon />, color: '#06b6d4', label: 'Weather' },
    'weather': { icon: <WeatherIcon />, color: '#06b6d4', label: 'Weather' },
    'system_info': { icon: <SystemIcon />, color: '#8b5cf6', label: 'System Info' },
    'get_system_info': { icon: <SystemIcon />, color: '#8b5cf6', label: 'System Info' },
    'database_query': { icon: <DatabaseIcon />, color: '#ec4899', label: 'Database' },
    'sql': { icon: <DatabaseIcon />, color: '#ec4899', label: 'SQL Query' },
    'api_call': { icon: <ApiIcon />, color: '#14b8a6', label: 'API Call' },
    'rag_search': { icon: <RagIcon />, color: '#a855f7', label: 'Knowledge Search' },
    'retrieve': { icon: <RagIcon />, color: '#a855f7', label: 'RAG Retrieval' },
  }

  // Check for partial matches
  const lowerName = toolName.toLowerCase()
  for (const [key, value] of Object.entries(toolMap)) {
    if (lowerName.includes(key) || key.includes(lowerName)) {
      return value
    }
  }

  // Default
  return { icon: <ApiIcon />, color: '#64748b', label: toolName }
}

const getStatusMeta = (status: string): { icon: React.ReactNode; color: string; label: string } => {
  switch (status) {
    case 'pending':
      return { icon: <PendingIcon />, color: '#64748b', label: 'Pending' }
    case 'executing':
      return { icon: <PendingIcon />, color: '#f59e0b', label: 'Executing' }
    case 'success':
      return { icon: <CheckIcon />, color: '#10b981', label: 'Success' }
    case 'error':
      return { icon: <ErrorIcon />, color: '#ef4444', label: 'Failed' }
    default:
      return { icon: <PendingIcon />, color: '#64748b', label: status }
  }
}

export const ExternalQueryDisplay: React.FC<ExternalQueryDisplayProps> = memo(({
  toolCall,
  result,
  status,
  executionTime,
  defaultExpanded = false,
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded)
  const [copied, setCopied] = useState(false)
  
  const toolMeta = getToolMeta(toolCall.function.name)
  const statusMeta = getStatusMeta(status)

  // Parse arguments
  let parsedArgs: Record<string, any> = {}
  try {
    parsedArgs = JSON.parse(toolCall.function.arguments)
  } catch {
    parsedArgs = { raw: toolCall.function.arguments }
  }

  const handleCopy = async () => {
    const content = JSON.stringify({ toolCall, result }, null, 2)
    await navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Paper
      variant="outlined"
      sx={{
        my: 1,
        backgroundColor: alpha(toolMeta.color, 0.03),
        borderColor: alpha(toolMeta.color, 0.2),
        overflow: 'hidden',
        transition: 'all 0.2s ease',
        '&:hover': {
          borderColor: alpha(toolMeta.color, 0.4),
        },
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          px: 2,
          py: 1,
          cursor: 'pointer',
          '&:hover': {
            backgroundColor: alpha(toolMeta.color, 0.05),
          },
        }}
        onClick={() => setExpanded(!expanded)}
      >
        {/* Tool Icon */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 28,
            height: 28,
            borderRadius: 1,
            bgcolor: alpha(toolMeta.color, 0.1),
            color: toolMeta.color,
            mr: 1.5,
            '& .MuiSvgIcon-root': { fontSize: 16 },
          }}
        >
          {toolMeta.icon}
        </Box>

        {/* Tool Name */}
        <Typography
          variant="subtitle2"
          sx={{
            fontWeight: 600,
            color: toolMeta.color,
            fontSize: '0.8125rem',
          }}
        >
          {toolMeta.label}
        </Typography>

        {/* Status Chip */}
        <Chip
          size="small"
          icon={<Box sx={{ '& .MuiSvgIcon-root': { fontSize: 12 } }}>{statusMeta.icon}</Box>}
          label={statusMeta.label}
          sx={{
            ml: 1.5,
            height: 22,
            fontSize: '0.6875rem',
            bgcolor: alpha(statusMeta.color, 0.1),
            color: statusMeta.color,
            border: `1px solid ${alpha(statusMeta.color, 0.3)}`,
            '& .MuiChip-icon': { color: statusMeta.color },
          }}
        />

        {/* Execution Time */}
        {executionTime && status === 'success' && (
          <Typography
            variant="caption"
            sx={{ ml: 1.5, color: 'text.secondary', fontSize: '0.6875rem' }}
          >
            {executionTime}ms
          </Typography>
        )}

        {/* Copy Button */}
        <Tooltip title={copied ? 'Copied!' : 'Copy'}>
          <IconButton
            size="small"
            onClick={(e) => {
              e.stopPropagation()
              handleCopy()
            }}
            sx={{ ml: 'auto', mr: 0.5 }}
          >
            {copied ? <CheckIcon fontSize="small" /> : <CopyIcon fontSize="small" />}
          </IconButton>
        </Tooltip>

        {/* Expand Button */}
        <IconButton
          size="small"
          sx={{
            transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.2s',
          }}
        >
          <ExpandMoreIcon fontSize="small" />
        </IconButton>
      </Box>

      {/* Executing Progress */}
      {status === 'executing' && (
        <LinearProgress
          sx={{
            height: 2,
            bgcolor: alpha(toolMeta.color, 0.1),
            '& .MuiLinearProgress-bar': {
              bgcolor: toolMeta.color,
            },
          }}
        />
      )}

      {/* Content */}
      <Collapse in={expanded}>
        <Box
          sx={{
            px: 2,
            py: 1.5,
            borderTop: '1px solid',
            borderColor: alpha(toolMeta.color, 0.1),
            backgroundColor: alpha('#000', 0.15),
          }}
        >
          {/* Arguments Section */}
          <Typography
            variant="caption"
            sx={{
              color: 'text.secondary',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              fontSize: '0.625rem',
            }}
          >
            Query Parameters
          </Typography>
          <Box
            sx={{
              mt: 0.5,
              mb: 1.5,
              p: 1,
              bgcolor: alpha('#000', 0.2),
              borderRadius: 1,
              fontFamily: 'monospace',
              fontSize: '0.75rem',
              overflow: 'auto',
              maxHeight: 150,
            }}
          >
            {Object.entries(parsedArgs).map(([key, value]) => (
              <Box key={key} sx={{ display: 'flex', mb: 0.5 }}>
                <Typography
                  component="span"
                  sx={{
                    color: '#a855f7',
                    fontFamily: 'monospace',
                    fontSize: '0.75rem',
                    mr: 1,
                  }}
                >
                  {key}:
                </Typography>
                <Typography
                  component="span"
                  sx={{
                    color: '#10b981',
                    fontFamily: 'monospace',
                    fontSize: '0.75rem',
                    wordBreak: 'break-all',
                  }}
                >
                  {typeof value === 'string' ? `"${value}"` : JSON.stringify(value)}
                </Typography>
              </Box>
            ))}
          </Box>

          {/* Result Section */}
          {result && (
            <>
              <Divider sx={{ my: 1 }} />
              <Typography
                variant="caption"
                sx={{
                  color: 'text.secondary',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  fontSize: '0.625rem',
                }}
              >
                Result
              </Typography>
              <Box
                sx={{
                  mt: 0.5,
                  p: 1,
                  bgcolor: status === 'error' ? alpha('#ef4444', 0.1) : alpha('#10b981', 0.05),
                  borderRadius: 1,
                  border: '1px solid',
                  borderColor: status === 'error' ? alpha('#ef4444', 0.2) : alpha('#10b981', 0.1),
                  fontFamily: 'monospace',
                  fontSize: '0.75rem',
                  overflow: 'auto',
                  maxHeight: 200,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}
              >
                {result}
              </Box>
            </>
          )}
        </Box>
      </Collapse>
    </Paper>
  )
})

ExternalQueryDisplay.displayName = 'ExternalQueryDisplay'

export default ExternalQueryDisplay
