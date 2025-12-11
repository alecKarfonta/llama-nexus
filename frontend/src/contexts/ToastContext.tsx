import React, { createContext, useContext, useState, useCallback } from 'react'
import { Snackbar, Alert, AlertColor, Slide, SlideProps, Box, Typography, IconButton, alpha } from '@mui/material'
import {
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Close as CloseIcon,
} from '@mui/icons-material'

interface Toast {
  id: string
  message: string
  severity: AlertColor
  duration?: number
  title?: string
}

interface ToastContextValue {
  showToast: (message: string, severity?: AlertColor, options?: { title?: string; duration?: number }) => void
  success: (message: string, title?: string) => void
  error: (message: string, title?: string) => void
  warning: (message: string, title?: string) => void
  info: (message: string, title?: string) => void
}

const ToastContext = createContext<ToastContextValue | null>(null)

function SlideTransition(props: SlideProps) {
  return <Slide {...props} direction="up" />
}

const iconMap: Record<AlertColor, React.ReactNode> = {
  success: <SuccessIcon />,
  error: <ErrorIcon />,
  warning: <WarningIcon />,
  info: <InfoIcon />,
}

const colorMap: Record<AlertColor, string> = {
  success: '#10b981',
  error: '#ef4444',
  warning: '#f59e0b',
  info: '#6366f1',
}

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<Toast[]>([])

  const showToast = useCallback((
    message: string, 
    severity: AlertColor = 'info',
    options?: { title?: string; duration?: number }
  ) => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const toast: Toast = {
      id,
      message,
      severity,
      duration: options?.duration ?? 4000,
      title: options?.title,
    }
    setToasts((prev) => [...prev, toast])
  }, [])

  const success = useCallback((message: string, title?: string) => {
    showToast(message, 'success', { title })
  }, [showToast])

  const error = useCallback((message: string, title?: string) => {
    showToast(message, 'error', { title, duration: 6000 })
  }, [showToast])

  const warning = useCallback((message: string, title?: string) => {
    showToast(message, 'warning', { title })
  }, [showToast])

  const info = useCallback((message: string, title?: string) => {
    showToast(message, 'info', { title })
  }, [showToast])

  const handleClose = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  return (
    <ToastContext.Provider value={{ showToast, success, error, warning, info }}>
      {children}
      {toasts.map((toast, index) => (
        <Snackbar
          key={toast.id}
          open
          autoHideDuration={toast.duration}
          onClose={() => handleClose(toast.id)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
          TransitionComponent={SlideTransition}
          sx={{
            bottom: { xs: 16 + index * 80, sm: 24 + index * 80 },
            right: { xs: 16, sm: 24 },
          }}
        >
          <Box
            sx={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: 1.5,
              p: 1.5,
              pr: 2,
              minWidth: 300,
              maxWidth: 450,
              bgcolor: 'rgba(15, 15, 35, 0.98)',
              backdropFilter: 'blur(20px)',
              border: '1px solid',
              borderColor: alpha(colorMap[toast.severity], 0.3),
              borderRadius: 2,
              boxShadow: `0 8px 32px rgba(0, 0, 0, 0.4), 0 0 0 1px ${alpha(colorMap[toast.severity], 0.1)}`,
            }}
          >
            {/* Icon */}
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 32,
                height: 32,
                borderRadius: 1.5,
                bgcolor: alpha(colorMap[toast.severity], 0.15),
                color: colorMap[toast.severity],
                flexShrink: 0,
                '& svg': { fontSize: 18 },
              }}
            >
              {iconMap[toast.severity]}
            </Box>

            {/* Content */}
            <Box sx={{ flex: 1, minWidth: 0 }}>
              {toast.title && (
                <Typography
                  variant="subtitle2"
                  sx={{
                    fontWeight: 600,
                    fontSize: '0.875rem',
                    color: 'text.primary',
                    lineHeight: 1.3,
                  }}
                >
                  {toast.title}
                </Typography>
              )}
              <Typography
                variant="body2"
                sx={{
                  fontSize: '0.8125rem',
                  color: toast.title ? 'text.secondary' : 'text.primary',
                  lineHeight: 1.4,
                }}
              >
                {toast.message}
              </Typography>
            </Box>

            {/* Close button */}
            <IconButton
              size="small"
              onClick={() => handleClose(toast.id)}
              sx={{
                color: 'text.secondary',
                p: 0.5,
                '&:hover': {
                  color: 'text.primary',
                  bgcolor: 'rgba(255, 255, 255, 0.05)',
                },
              }}
            >
              <CloseIcon sx={{ fontSize: 16 }} />
            </IconButton>
          </Box>
        </Snackbar>
      ))}
    </ToastContext.Provider>
  )
}

export const useToast = (): ToastContextValue => {
  const context = useContext(ToastContext)
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider')
  }
  return context
}

export default ToastContext
