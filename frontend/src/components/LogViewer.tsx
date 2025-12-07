import React, { useState, useEffect, useRef, forwardRef, useImperativeHandle } from 'react'
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  CircularProgress,
  Chip
} from '@mui/material'
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Clear as ClearIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material'

interface LogEntry {
  message: string
  timestamp: string
  level?: string
}

interface LogViewerProps {
  containerName?: string
  maxLines?: number
  height?: number | string
}

export interface LogViewerRef {
  clearLogs: () => void
  refreshLogs: () => void
}

export const LogViewer = forwardRef<LogViewerRef, LogViewerProps>(({
  containerName = 'llamacpp-api',
  maxLines = 500,
  height = 400
}, ref) => {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [autoScroll, setAutoScroll] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)
  const eventSourceRef = useRef<EventSource | null>(null)
  const logsContainerRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs, autoScroll])

  // Fetch initial logs
  const fetchInitialLogs = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/v1/logs/container?lines=100')
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const data = await response.json()
      setLogs(data.logs || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch logs')
      console.error('Error fetching logs:', err)
    } finally {
      setIsLoading(false)
    }
  }

  // Start streaming logs
  const startStreaming = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }

    setError(null)
    setIsStreaming(true)

    const eventSource = new EventSource('/api/v1/logs/container/stream')
    eventSourceRef.current = eventSource

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.error) {
          setError(data.error)
          return
        }
        
        const newLog: LogEntry = {
          message: data.message,
          timestamp: data.timestamp || new Date().toISOString(),
          level: 'INFO'
        }
        
        setLogs((prev) => {
          const updated = [...prev, newLog]
          // Keep only last maxLines
          return updated.slice(-maxLines)
        })
      } catch (err) {
        console.error('Error parsing log event:', err)
      }
    }

    eventSource.onerror = (err) => {
      console.error('EventSource error:', err)
      setError('Log stream disconnected')
      setIsStreaming(false)
      eventSource.close()
    }
  }

  // Stop streaming logs
  const stopStreaming = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
    setIsStreaming(false)
  }

  // Clear logs
  const clearLogs = () => {
    setLogs([])
    setError(null)
  }

  // Download logs
  const downloadLogs = () => {
    const logText = logs.map(log => 
      `[${new Date(log.timestamp).toLocaleString()}] ${log.message}`
    ).join('\n')
    
    const blob = new Blob([logText], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${containerName}-logs-${new Date().toISOString()}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  // Toggle streaming
  const toggleStreaming = () => {
    if (isStreaming) {
      stopStreaming()
    } else {
      startStreaming()
    }
  }

  // Expose methods to parent component via ref
  useImperativeHandle(ref, () => ({
    clearLogs,
    refreshLogs: fetchInitialLogs
  }))

  // Load initial logs on mount
  useEffect(() => {
    fetchInitialLogs()
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
      }
    }
  }, [])

  return (
    <Paper 
      elevation={0} 
      sx={{ 
        border: '1px solid',
        borderColor: 'grey.300',
        borderRadius: 1,
        overflow: 'hidden'
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          p: 1.5,
          borderBottom: '1px solid',
          borderColor: 'grey.200',
          bgcolor: 'grey.50'
        }}
      >
        <Box display="flex" alignItems="center" gap={1}>
          <Typography variant="subtitle2" fontWeight={600}>
            Container Logs
          </Typography>
          {isStreaming && (
            <Chip 
              label="LIVE" 
              size="small" 
              color="success" 
              sx={{ height: 20, fontSize: '0.7rem' }}
            />
          )}
          {logs.length > 0 && (
            <Typography variant="caption" color="text.secondary">
              ({logs.length} lines)
            </Typography>
          )}
        </Box>
        
        <Box display="flex" alignItems="center" gap={0.5}>
          <FormControlLabel
            control={
              <Switch
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                size="small"
              />
            }
            label={<Typography variant="caption">Auto-scroll</Typography>}
            sx={{ mr: 1 }}
          />
          
          <Tooltip title={isStreaming ? "Pause streaming" : "Start streaming"}>
            <IconButton size="small" onClick={toggleStreaming} color={isStreaming ? "error" : "primary"}>
              {isStreaming ? <PauseIcon fontSize="small" /> : <PlayIcon fontSize="small" />}
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Refresh logs">
            <IconButton size="small" onClick={fetchInitialLogs} disabled={isLoading || isStreaming}>
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Clear logs">
            <IconButton size="small" onClick={clearLogs}>
              <ClearIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Download logs">
            <IconButton size="small" onClick={downloadLogs} disabled={logs.length === 0}>
              <DownloadIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Log content */}
      <Box
        ref={logsContainerRef}
        sx={{
          height,
          overflow: 'auto',
          bgcolor: '#1e1e1e',
          color: '#d4d4d4',
          fontFamily: 'monospace',
          fontSize: '0.75rem',
          p: 1.5,
          '&::-webkit-scrollbar': {
            width: '8px',
          },
          '&::-webkit-scrollbar-track': {
            bgcolor: '#2d2d2d',
          },
          '&::-webkit-scrollbar-thumb': {
            bgcolor: '#555',
            borderRadius: '4px',
            '&:hover': {
              bgcolor: '#666',
            },
          },
        }}
      >
        {isLoading && (
          <Box display="flex" alignItems="center" justifyContent="center" height="100%">
            <CircularProgress size={24} />
          </Box>
        )}
        
        {error && (
          <Box p={2} bgcolor="error.dark" borderRadius={1}>
            <Typography variant="body2" color="error.light">
              {error}
            </Typography>
          </Box>
        )}
        
        {!isLoading && logs.length === 0 && !error && (
          <Box display="flex" alignItems="center" justifyContent="center" height="100%" flexDirection="column" gap={1}>
            <Typography variant="body2" color="text.secondary">
              No logs available
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Click the play button to start streaming
            </Typography>
          </Box>
        )}
        
        {logs.map((log, index) => (
          <Box
            key={index}
            sx={{
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              lineHeight: 1.5,
              mb: 0.25,
              '&:hover': {
                bgcolor: 'rgba(255, 255, 255, 0.05)',
              },
            }}
          >
            <span style={{ opacity: 0.5, marginRight: '8px' }}>
              {new Date(log.timestamp).toLocaleTimeString()}
            </span>
            <span>{log.message}</span>
          </Box>
        ))}
        <div ref={logsEndRef} />
      </Box>
    </Paper>
  )
})

LogViewer.displayName = 'LogViewer'

export default LogViewer

