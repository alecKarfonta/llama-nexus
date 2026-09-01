import { Component, ErrorInfo, ReactNode } from 'react'
import { Link as RouterLink } from 'react-router-dom'
import {
  Alert,
  Box,
  Button,
  Chip,
  Collapse,
  Divider,
  Paper,
  Stack,
  Typography,
  alpha,
} from '@mui/material'
import {
  BugReport as BugReportIcon,
  Dashboard as DashboardIcon,
  ErrorOutline as ErrorIcon,
  Refresh as RetryIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material'
import { getRoutePath } from '@/config/navigation'

interface Props {
  children: ReactNode
  resetKey?: string
}

interface State {
  hasError: boolean
  error?: Error
  errorInfo?: ErrorInfo
  showDetails: boolean
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, showDetails: false }
  }

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, showDetails: false }
  }

  public componentDidUpdate(prevProps: Props) {
    if (this.state.hasError && prevProps.resetKey !== this.props.resetKey) {
      this.handleReset()
    }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo)
    this.setState({ errorInfo })
  }

  private handleReset = () => {
    this.setState({
      hasError: false,
      error: undefined,
      errorInfo: undefined,
      showDetails: false,
    })
  }

  private toggleDetails = () => {
    this.setState((state) => ({ showDetails: !state.showDetails }))
  }

  public render() {
    if (this.state.hasError) {
      const message = this.state.error?.message || 'An unexpected rendering error occurred.'
      const stack = this.state.errorInfo?.componentStack?.trim()

      return (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '100vh',
            px: { xs: 2, sm: 3 },
            py: 4,
            bgcolor: 'background.default',
          }}
        >
          <Paper
            elevation={0}
            sx={{
              width: '100%',
              maxWidth: 760,
              p: { xs: 2, sm: 3 },
              borderRadius: '8px',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              bgcolor: 'rgba(15, 23, 42, 0.78)',
              background:
                'linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(99, 102, 241, 0.08) 55%, rgba(20, 184, 166, 0.08) 100%)',
              overflow: 'hidden',
            }}
          >
            <Stack spacing={2.5}>
              <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.5 }}>
                <Box
                  sx={{
                    width: 44,
                    height: 44,
                    borderRadius: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#f87171',
                    bgcolor: alpha('#ef4444', 0.14),
                    flexShrink: 0,
                  }}
                >
                  <ErrorIcon />
                </Box>
                <Box sx={{ minWidth: 0, flex: 1 }}>
                  <Chip
                    size="small"
                    label="Recovery mode"
                    sx={{
                      mb: 1,
                      height: 24,
                      color: '#fca5a5',
                      bgcolor: alpha('#ef4444', 0.12),
                      border: `1px solid ${alpha('#ef4444', 0.25)}`,
                      fontWeight: 700,
                    }}
                  />
                  <Typography variant="h4" sx={{ fontWeight: 800, fontSize: { xs: '1.35rem', sm: '1.75rem' } }}>
                    This workspace hit a rendering error
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 0.75 }}>
                    The app shell is still running. Try the view again, jump back to the dashboard, or open settings to check service state.
                  </Typography>
                </Box>
              </Box>

              <Alert severity="error" variant="outlined" sx={{ borderRadius: '8px' }}>
                <Typography variant="body2" sx={{ overflowWrap: 'anywhere' }}>
                  {message}
                </Typography>
              </Alert>

              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
                <Button variant="contained" startIcon={<RetryIcon />} onClick={this.handleReset}>
                  Try Again
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<DashboardIcon />}
                  component={RouterLink}
                  to={getRoutePath('dashboard')}
                  onClick={this.handleReset}
                >
                  Dashboard
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<SettingsIcon />}
                  component={RouterLink}
                  to={getRoutePath('configuration')}
                  onClick={this.handleReset}
                >
                  Settings
                </Button>
                <Button variant="text" startIcon={<BugReportIcon />} onClick={this.toggleDetails}>
                  {this.state.showDetails ? 'Hide Details' : 'Show Details'}
                </Button>
              </Stack>

              <Collapse in={this.state.showDetails}>
                <Divider sx={{ mb: 2 }} />
                <Box
                  component="pre"
                  sx={{
                    m: 0,
                    p: 1.5,
                    maxHeight: 260,
                    overflow: 'auto',
                    borderRadius: '8px',
                    bgcolor: 'rgba(0, 0, 0, 0.28)',
                    color: 'text.secondary',
                    fontSize: '0.75rem',
                    whiteSpace: 'pre-wrap',
                    overflowWrap: 'anywhere',
                  }}
                >
                  {`${this.state.error?.stack || message}${stack ? `\n\nComponent stack:\n${stack}` : ''}`}
                </Box>
              </Collapse>
            </Stack>
          </Paper>
        </Box>
      )
    }

    return this.props.children
  }
}
