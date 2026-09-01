import { useState, useEffect, useRef, forwardRef, useImperativeHandle } from 'react'
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
import { debugLog } from '@/utils/debug'

interface LogEntry {
  message: string
  timestamp: string
  level?: string
}

interface LogViewerProps {
  containerName?: string
  maxLines?: number
  height?: number | string
  backend?: 'llamacpp' | 'vllm'
  compact?: boolean
  collapseEmpty?: boolean
}

export interface LogViewerRef {
  clearLogs: () => void
  refreshLogs: () => void
}

export const LogViewer = forwardRef<LogViewerRef, LogViewerProps>(({
  containerName = 'llamacpp-api',
  maxLines = 500,
  height = 400,
  backend = 'llamacpp',
  compact = false,
  collapseEmpty = false
}, ref) => {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [autoScroll, setAutoScroll] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const eventSourceRef = useRef<EventSource | null>(null)
  const logsContainerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (autoScroll && logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  const normalizeLogs = (rawLogs: unknown): LogEntry[] => {
    if (!rawLogs) return []
    if (Array.isArray(rawLogs)) {
      return rawLogs
        .map((entry): LogEntry | null => {
          if (typeof entry === 'string') {
            return { message: entry, timestamp: new Date().toISOString(), level: 'INFO' }
          }
          if (entry && typeof entry === 'object') {
            const obj = entry as Record<string, unknown>
            const message = String(obj.message ?? obj.text ?? '')
            if (!message) return null
            return {
              message,
              timestamp: String(obj.timestamp ?? new Date().toISOString()),
              level: typeof obj.level === 'string' ? obj.level : 'INFO'
            }
          }
          return null
        })
        .filter((entry): entry is LogEntry => entry !== null)
    }
    if (typeof rawLogs === 'string') {
      return rawLogs
        .split('\n')
        .filter((line) => line.trim().length > 0)
        .map((line) => ({
          message: line,
          timestamp: new Date().toISOString(),
          level: 'INFO'
        }))
    }
    return []
  }

  // Fetch initial logs
  const fetchInitialLogs = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await fetch(`/api/v1/logs/container?lines=100&backend=${backend}`)
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const data = await response.json()
      setLogs(normalizeLogs(data.logs))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch logs')
      debugLog('logs', 'Container logs unavailable', err)
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

    const eventSource = new EventSource(`/api/v1/logs/container/stream?backend=${backend}`)
    eventSourceRef.current = eventSource

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'status' && data.available === false) {
          setError(null)
          setIsStreaming(false)
          eventSource.close()
          return
        }

        if (data.error) {
          setError(data.error)
          return
        }
        
        const message = data.message ?? data.text
        if (!message) {
          return
        }

        const newLog: LogEntry = {
          message,
          timestamp: data.timestamp || new Date().toISOString(),
          level: 'INFO'
        }
        
        setLogs((prev) => {
          const updated = [...prev, newLog]
          // Keep only last maxLines
          return updated.slice(-maxLines)
        })
      } catch (err) {
        debugLog('logs', 'Ignoring malformed log event', err)
      }
    }

    eventSource.onerror = (err) => {
      debugLog('logs', 'Log stream disconnected', err)
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

  const compactEmpty = compact && !isLoading && logs.length === 0 && !error && !isStreaming
  const shouldCollapseEmpty = compactEmpty && collapseEmpty
  const hasLogs = logs.length > 0
  const showAutoScrollControl = hasLogs || isStreaming || !compact
  const showClearLogsControl = hasLogs || Boolean(error)
  const showDownloadLogsControl = hasLogs

  return (
    <Paper 
      elevation={0} 
      sx={{ 
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: '8px',
        bgcolor: 'background.paper',
        overflow: 'hidden'
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 1,
          px: compact ? 0.75 : 1,
          py: compact ? 0.25 : 0.5,
          borderBottom: '1px solid',
          borderColor: 'divider',
          bgcolor: 'action.hover'
        }}
      >
        <Box display="flex" alignItems="center" gap={1} minWidth={0}>
          {!compact && (
            <Typography variant="caption" sx={{ fontWeight: 800, color: 'text.secondary' }}>
              Logs
            </Typography>
          )}
          {isStreaming && (
            <Chip 
              label="LIVE" 
              size="small" 
              color="success" 
              sx={{ height: 20, fontSize: '0.7rem' }}
            />
          )}
          {hasLogs && (
            <Typography variant="caption" color="text.secondary">
              ({logs.length} lines)
            </Typography>
          )}
        </Box>
        
        <Box display="flex" alignItems="center" gap={0.5} flexShrink={0}>
          {showAutoScrollControl && (
            compact ? (
              <Tooltip title={autoScroll ? 'Auto-scroll on' : 'Auto-scroll off'}>
                <Switch
                  checked={autoScroll}
                  onChange={(e) => setAutoScroll(e.target.checked)}
                  size="small"
                  inputProps={{ 'aria-label': 'Auto-scroll logs' }}
                />
              </Tooltip>
            ) : (
              <FormControlLabel
                control={
                  <Switch
                    checked={autoScroll}
                    onChange={(e) => setAutoScroll(e.target.checked)}
                    size="small"
                  />
                }
                label={<Typography variant="caption">Auto-scroll</Typography>}
                sx={{ mr: 0.5, color: 'text.secondary' }}
              />
            )
          )}
          
          <Tooltip title={isStreaming ? "Pause streaming" : "Start streaming"}>
            <IconButton size="small" onClick={toggleStreaming} color={isStreaming ? "error" : "primary"} sx={{ width: compact ? 24 : 28, height: compact ? 24 : 28, borderRadius: '8px' }}>
              {isStreaming ? <PauseIcon fontSize="small" /> : <PlayIcon fontSize="small" />}
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Refresh logs">
            <IconButton size="small" onClick={fetchInitialLogs} disabled={isLoading || isStreaming} sx={{ width: compact ? 24 : 28, height: compact ? 24 : 28, borderRadius: '8px' }}>
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          
          {showClearLogsControl && (
            <Tooltip title="Clear logs">
              <IconButton size="small" onClick={clearLogs} sx={{ width: compact ? 24 : 28, height: compact ? 24 : 28, borderRadius: '8px' }}>
                <ClearIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
          
          {showDownloadLogsControl && (
            <Tooltip title="Download logs">
              <IconButton size="small" onClick={downloadLogs} sx={{ width: compact ? 24 : 28, height: compact ? 24 : 28, borderRadius: '8px' }}>
                <DownloadIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
        </Box>
      </Box>

      {!shouldCollapseEmpty && (
      <Box
        ref={logsContainerRef}
        sx={{
          height: compactEmpty ? 42 : height,
          overflow: compactEmpty ? 'hidden' : 'auto',
          bgcolor: 'rgba(0, 0, 0, 0.24)',
          color: 'text.secondary',
          fontFamily: 'monospace',
          fontSize: compact ? '0.7rem' : '0.75rem',
          lineHeight: compact ? 1.35 : 1.5,
          p: compact ? 0.75 : 1,
          '&::-webkit-scrollbar': {
            width: 6,
          },
          '&::-webkit-scrollbar-track': {
            bgcolor: 'transparent',
          },
          '&::-webkit-scrollbar-thumb': {
            bgcolor: 'rgba(255, 255, 255, 0.14)',
            borderRadius: '4px',
            '&:hover': {
              bgcolor: 'rgba(255, 255, 255, 0.22)',
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
          <Box px={1} py={0.75} bgcolor="error.dark" borderRadius="8px">
            <Typography variant="caption" color="error.light" sx={{ fontFamily: 'monospace' }}>
              {error}
            </Typography>
          </Box>
        )}
        
        {!isLoading && logs.length === 0 && !error && (
          <Box display="flex" alignItems="center" justifyContent="center" height="100%">
            <Typography variant="caption" color="text.secondary">
              {isStreaming ? 'Waiting for logs' : 'No logs'}
            </Typography>
          </Box>
        )}
        
        {logs.map((log, index) => (
          <Box
            key={index}
            sx={{
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              lineHeight: compact ? 1.35 : 1.5,
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
      </Box>
      )}
    </Paper>
  )
})

LogViewer.displayName = 'LogViewer'

export default LogViewer
