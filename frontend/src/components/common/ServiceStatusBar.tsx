import React, { useState, useEffect } from 'react'
import { Box, Tooltip, Typography, Chip, alpha, CircularProgress } from '@mui/material'
import {
  Memory as LLMIcon,
  TextFields as EmbeddingIcon,
  Mic as STTIcon,
  RecordVoiceOver as TTSIcon,
  CheckCircle as OnlineIcon,
  Cancel as OfflineIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material'
import { apiService } from '@/services/api'

interface ServiceStatus {
  id: string
  name: string
  icon: React.ReactNode
  running: boolean
  loading: boolean
  color: string
  path: string
}

interface ServiceStatusBarProps {
  compact?: boolean
  showLabels?: boolean
  onServiceClick?: (serviceId: string, path: string) => void
}

export const ServiceStatusBar: React.FC<ServiceStatusBarProps> = ({
  compact = false,
  showLabels = true,
  onServiceClick,
}) => {
  const [services, setServices] = useState<ServiceStatus[]>([
    { id: 'llm', name: 'LLM', icon: <LLMIcon />, running: false, loading: true, color: '#f59e0b', path: '/deploy' },
    { id: 'embedding', name: 'Embed', icon: <EmbeddingIcon />, running: false, loading: true, color: '#06b6d4', path: '/embedding-deploy' },
    { id: 'stt', name: 'STT', icon: <STTIcon />, running: false, loading: true, color: '#10b981', path: '/stt-deploy' },
    { id: 'tts', name: 'TTS', icon: <TTSIcon />, running: false, loading: true, color: '#8b5cf6', path: '/tts-deploy' },
  ])
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  const checkServices = async () => {
    const updates = await Promise.all([
      // LLM service
      apiService.getServiceStatus()
        .then(status => ({ id: 'llm', running: status.health === 'healthy' && status.modelLoaded }))
        .catch(() => ({ id: 'llm', running: false })),
      // Embedding service (check if llamacpp-embed is running)
      apiService.get('/api/v1/embedding/status')
        .then((response) => ({ id: 'embedding', running: response.data?.running ?? false }))
        .catch(() => ({ id: 'embedding', running: false })),
      // STT service
      apiService.getSTTStatus()
        .then(status => ({ id: 'stt', running: status.running }))
        .catch(() => ({ id: 'stt', running: false })),
      // TTS service
      apiService.getTTSStatus()
        .then(status => ({ id: 'tts', running: status.running }))
        .catch(() => ({ id: 'tts', running: false })),
    ])

    setServices(prev => prev.map(service => {
      const update = updates.find(u => u.id === service.id)
      return {
        ...service,
        running: update?.running ?? false,
        loading: false,
      }
    }))
    setLastUpdated(new Date())
  }

  useEffect(() => {
    checkServices()
    const interval = setInterval(checkServices, 30000) // Check every 30s
    return () => clearInterval(interval)
  }, [])

  const runningCount = services.filter(s => s.running).length
  const totalCount = services.length

  if (compact) {
    return (
      <Tooltip
        title={
          <Box sx={{ p: 0.5 }}>
            <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 1 }}>
              Services: {runningCount}/{totalCount} running
            </Typography>
            {services.map(service => (
              <Box key={service.id} sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Box sx={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  bgcolor: service.loading ? 'grey.500' : service.running ? '#10b981' : '#ef4444'
                }} />
                <Typography variant="caption">{service.name}</Typography>
              </Box>
            ))}
            {lastUpdated && (
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 1, fontSize: '0.625rem' }}>
                Updated {lastUpdated.toLocaleTimeString()}
              </Typography>
            )}
          </Box>
        }
      >
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
            px: 1,
            py: 0.5,
            borderRadius: 1.5,
            bgcolor: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            cursor: 'pointer',
          }}
        >
          {services.map(service => (
            <Box
              key={service.id}
              sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                bgcolor: service.loading
                  ? 'grey.600'
                  : service.running
                    ? '#10b981'
                    : alpha('#ef4444', 0.5),
                transition: 'all 0.3s ease',
              }}
            />
          ))}
        </Box>
      </Tooltip>
    )
  }

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        p: 1,
        borderRadius: 2,
        bgcolor: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid rgba(255, 255, 255, 0.06)',
      }}
    >
      {services.map(service => (
        <Tooltip
          key={service.id}
          title={`${service.name}: ${service.loading ? 'Checking...' : service.running ? 'Running' : 'Stopped'}`}
        >
          <Chip
            icon={
              service.loading ? (
                <CircularProgress size={12} sx={{ color: 'text.secondary' }} />
              ) : service.running ? (
                <OnlineIcon sx={{ fontSize: '14px !important', color: '#10b981 !important' }} />
              ) : (
                <OfflineIcon sx={{ fontSize: '14px !important', color: alpha('#ef4444', 0.6) + ' !important' }} />
              )
            }
            label={showLabels ? service.name : undefined}
            size="small"
            onClick={() => onServiceClick?.(service.id, service.path)}
            sx={{
              height: 28,
              bgcolor: service.running ? alpha('#10b981', 0.1) : 'rgba(255, 255, 255, 0.03)',
              border: '1px solid',
              borderColor: service.running ? alpha('#10b981', 0.2) : 'rgba(255, 255, 255, 0.06)',
              color: service.running ? '#34d399' : 'text.secondary',
              fontWeight: 500,
              fontSize: '0.75rem',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              '&:hover': {
                bgcolor: service.running ? alpha('#10b981', 0.15) : alpha(service.color, 0.1),
                borderColor: service.running ? alpha('#10b981', 0.3) : alpha(service.color, 0.3),
              },
              '& .MuiChip-icon': {
                ml: showLabels ? 0.5 : 0,
                mr: showLabels ? -0.5 : 0,
              },
              ...(!showLabels && {
                width: 28,
                '& .MuiChip-label': { display: 'none' },
              }),
            }}
          />
        </Tooltip>
      ))}

      <Tooltip title="Refresh status">
        <Box
          onClick={() => {
            setServices(prev => prev.map(s => ({ ...s, loading: true })))
            checkServices()
          }}
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 28,
            height: 28,
            borderRadius: 1,
            cursor: 'pointer',
            color: 'text.secondary',
            transition: 'all 0.2s ease',
            '&:hover': {
              bgcolor: 'rgba(255, 255, 255, 0.05)',
              color: 'text.primary',
            },
          }}
        >
          <RefreshIcon sx={{ fontSize: 16 }} />
        </Box>
      </Tooltip>
    </Box>
  )
}

export default ServiceStatusBar
