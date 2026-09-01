import React, { useEffect, useState, useRef } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Paper,
  IconButton,
  Tooltip,
  Chip,
  TextField,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Clear as ClearIcon,
  Download as DownloadIcon,
  Pause as PauseIcon,
  PlayArrow as PlayIcon,
  Search as SearchIcon,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { debugLog } from '@/utils/debug';

interface LogEntry {
  message: string;
  timestamp: string;
}

interface LogViewerProps {
  maxLines?: number;
  height?: number | string;
  compact?: boolean;
  collapseEmpty?: boolean;
}

export const LogViewer: React.FC<LogViewerProps> = ({
  maxLines = 1000,
  height = 500,
  compact = false,
  collapseEmpty = false,
}) => {
  const theme = useTheme();
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filteredLogs, setFilteredLogs] = useState<LogEntry[]>([]);
  const [connected, setConnected] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [paused, setPaused] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const logContainerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Fetch initial logs
    fetchLogs();
    
    // Connect to WebSocket for real-time logs
    if (!paused) {
      connectWebSocket();
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [paused]);

  useEffect(() => {
    // Filter logs based on search term
    if (searchTerm) {
      setFilteredLogs(
        logs.filter(log => 
          log.message.toLowerCase().includes(searchTerm.toLowerCase())
        )
      );
    } else {
      setFilteredLogs(logs);
    }
  }, [logs, searchTerm]);

  useEffect(() => {
    // Auto-scroll to bottom when new logs arrive
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [filteredLogs, autoScroll]);

  const fetchLogs = async () => {
    try {
      const response = await fetch(`/api/v1/logs?lines=100`);
      if (response.ok) {
        const data = await response.json();
        setLogs(data.logs || []);
      }
    } catch (error) {
      console.error('Failed to fetch logs:', error);
    }
  };

  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    // WebSocket endpoint for logs is not implemented yet, so we'll rely on polling
    debugLog('logs', 'WebSocket log streaming not implemented yet, using polling mode');
    setConnected(false);
  };

  const handleClearLogs = async () => {
    try {
      const response = await fetch(`/api/v1/logs`, {
        method: 'DELETE',
      });
      if (response.ok) {
        setLogs([]);
      }
    } catch (error) {
      console.error('Failed to clear logs:', error);
    }
  };

  const handleDownloadLogs = () => {
    const logText = logs.map(log => 
      `[${new Date(log.timestamp).toLocaleString()}] ${log.message}`
    ).join('\n');
    
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `llamacpp-logs-${new Date().toISOString()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleTogglePause = () => {
    setPaused(!paused);
    if (paused && wsRef.current?.readyState === WebSocket.CLOSED) {
      connectWebSocket();
    } else if (!paused && wsRef.current) {
      wsRef.current.close();
    }
  };

  const getLogLevelColor = (message: string) => {
    const lowerMessage = message.toLowerCase();
    if (lowerMessage.includes('error') || lowerMessage.includes('fail')) {
      return theme.palette.error.main;
    }
    if (lowerMessage.includes('warn')) {
      return theme.palette.warning.main;
    }
    if (lowerMessage.includes('info')) {
      return theme.palette.info.main;
    }
    if (lowerMessage.includes('success') || lowerMessage.includes('loaded')) {
      return theme.palette.success.main;
    }
    return theme.palette.text.primary;
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const time = date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
    const ms = date.getMilliseconds().toString().padStart(3, '0');
    return `${time}.${ms}`;
  };
  const compactEmpty = compact && filteredLogs.length === 0 && !searchTerm;
  const shouldCollapseEmpty = compactEmpty && collapseEmpty;
  const effectiveHeight = compactEmpty ? 42 : height;

  return (
    <Card
      elevation={compact ? 0 : undefined}
      sx={compact ? {
        bgcolor: 'transparent',
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: '8px',
        boxShadow: 'none',
      } : undefined}
    >
      <CardContent sx={{ p: compact ? 0.75 : 2, '&:last-child': { pb: compact ? 0.75 : 2 } }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={compact ? 0.5 : 2} gap={1}>
          <Box display="flex" alignItems="center" gap={compact ? 1 : 2} minWidth={0}>
            {!compact && (
              <Typography variant="h6">
                LlamaCPP Logs
              </Typography>
            )}
            <Chip
              label={connected ? 'Connected' : 'Disconnected'}
              size="small"
              color={connected ? 'success' : 'default'}
              sx={compact ? { height: 20, fontSize: '0.68rem' } : undefined}
            />
            {paused && (
              <Chip
                label="Paused"
                size="small"
                color="warning"
                sx={compact ? { height: 20, fontSize: '0.68rem' } : undefined}
              />
            )}
          </Box>
          
          <Box display="flex" alignItems="center" gap={1}>
            {!compact && (
              <TextField
                size="small"
                placeholder="Search logs..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                InputProps={{
                  startAdornment: <SearchIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />,
                }}
                sx={{ minWidth: 200 }}
              />
            )}
            
            {!compact && (
              <FormControlLabel
                control={
                  <Switch
                    checked={autoScroll}
                    onChange={(e) => setAutoScroll(e.target.checked)}
                    size="small"
                  />
                }
                label="Auto-scroll"
                sx={{ ml: 2 }}
              />
            )}
            
            <Tooltip title={paused ? 'Resume' : 'Pause'}>
              <IconButton onClick={handleTogglePause} size="small" sx={compact ? { width: 28, height: 28, borderRadius: '8px' } : undefined}>
                {paused ? <PlayIcon fontSize="small" /> : <PauseIcon fontSize="small" />}
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Clear logs">
              <IconButton onClick={handleClearLogs} size="small" disabled={logs.length === 0} sx={compact ? { width: 28, height: 28, borderRadius: '8px' } : undefined}>
                <ClearIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Download logs">
              <IconButton onClick={handleDownloadLogs} size="small" disabled={logs.length === 0} sx={compact ? { width: 28, height: 28, borderRadius: '8px' } : undefined}>
                <DownloadIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {shouldCollapseEmpty ? (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              minHeight: 28,
              px: 1,
              borderRadius: '8px',
              bgcolor: 'action.hover',
              color: 'text.secondary',
            }}
          >
            <Typography variant="caption" sx={{ fontWeight: 700 }}>
              Idle
            </Typography>
          </Box>
        ) : (
          <Paper
            ref={logContainerRef}
            sx={{
              p: compact ? 0.75 : 2,
              backgroundColor: compactEmpty ? 'action.hover' : '#1e1e1e',
              color: compactEmpty ? 'text.secondary' : '#d4d4d4',
              fontFamily: '"Consolas", "Monaco", "Courier New", monospace',
              fontSize: compact ? '0.75rem' : '0.875rem',
              lineHeight: compact ? 1.35 : 1.5,
              height: effectiveHeight,
              overflowY: compactEmpty ? 'hidden' : 'auto',
              overflowX: compactEmpty ? 'hidden' : 'auto',
              borderRadius: '8px',
              '&::-webkit-scrollbar': {
                width: 8,
                height: 8,
              },
              '&::-webkit-scrollbar-track': {
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
              },
              '&::-webkit-scrollbar-thumb': {
                backgroundColor: 'rgba(255, 255, 255, 0.3)',
                borderRadius: 4,
              },
            }}
          >
            {filteredLogs.length === 0 ? (
              <Box
                display="flex"
                justifyContent="center"
                alignItems="center"
                height="100%"
                color="text.secondary"
              >
                <Typography variant="body2">
                  {searchTerm ? 'No logs match your search' : 'No logs'}
                </Typography>
              </Box>
            ) : (
              filteredLogs.map((log, index) => (
                <Box
                  key={index}
                  sx={{
                    mb: 0.5,
                    '&:hover': {
                      backgroundColor: 'rgba(255, 255, 255, 0.05)',
                    },
                  }}
                >
                  <span style={{ color: '#858585', marginRight: 8 }}>
                    [{formatTimestamp(log.timestamp)}]
                  </span>
                  <span style={{ color: getLogLevelColor(log.message) }}>
                    {log.message}
                  </span>
                </Box>
              ))
            )}
          </Paper>
        )}

        {!compactEmpty && (
          <Box display="flex" justifyContent="space-between" alignItems="center" mt={compact ? 0.5 : 1}>
            <Typography variant="caption" color="text.secondary">
              Showing {filteredLogs.length} of {logs.length} logs
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Max {maxLines} lines retained
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};
