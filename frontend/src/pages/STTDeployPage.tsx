import React, { useEffect, useState, useRef } from 'react'
import {
  Box,
  Grid,
  Typography,
  TextField,
  Select,
  MenuItem,
  Button,
  Alert,
  Chip,
  CircularProgress,
  FormHelperText,
  IconButton,
  LinearProgress,
  Tooltip,
  useMediaQuery,
} from '@mui/material'
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  RestartAlt as RestartIcon,
  Save as SaveIcon,
  Mic as MicIcon,
  MicOff as MicOffIcon,
  CloudUpload as UploadIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material'
import {
  compactCollapsibleCardSx,
  PageShell,
  SectionCard,
} from '@/components/common'
import { alpha } from '@mui/material/styles'
import { debugLog } from '@/utils/debug'
import { accentColors } from '@/utils/theme'

interface STTConfig {
  model: {
    name: string
    size: string
    language: string
    task: string
  }
  server: {
    host: string
    port: number
    api_key: string
  }
  execution: {
    mode: string
    cuda_devices: string
    compute_type: string
  }
}

interface STTStatus {
  running: boolean
  uptime: number
  endpoint: string
  model?: {
    name: string
    size: string
  }
}

// Available Whisper models
const WHISPER_MODELS = [
  { name: 'tiny', size: 'tiny', params: '39M', vram: '~1GB', speed: 'Fastest', quality: 'Lower' },
  { name: 'base', size: 'base', params: '74M', vram: '~1GB', speed: 'Fast', quality: 'Good' },
  { name: 'small', size: 'small', params: '244M', vram: '~2GB', speed: 'Medium', quality: 'Better' },
  { name: 'medium', size: 'medium', params: '769M', vram: '~5GB', speed: 'Slower', quality: 'Great' },
  { name: 'large-v3', size: 'large-v3', params: '1.5B', vram: '~10GB', speed: 'Slowest', quality: 'Best' },
  { name: 'distil-large-v3.5-ct2', size: 'distil-large-v3.5-ct2', params: '580M', vram: '~8GB', speed: 'Faster', quality: 'High' },
]

const LANGUAGES = [
  { code: 'auto', name: 'Auto-detect' },
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'it', name: 'Italian' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'ru', name: 'Russian' },
  { code: 'ja', name: 'Japanese' },
  { code: 'ko', name: 'Korean' },
  { code: 'zh', name: 'Chinese' },
  { code: 'ar', name: 'Arabic' },
  { code: 'hi', name: 'Hindi' },
]

export const STTDeployPage: React.FC = () => {
  const isNarrow = useMediaQuery('(max-width:599.95px)')

  // Service state
  const [status, setStatus] = useState<STTStatus | null>(null)
  const [config, setConfig] = useState<STTConfig>({
    model: {
      name: 'base',
      size: 'base',
      language: 'auto',
      task: 'transcribe',
    },
    server: {
      host: '0.0.0.0',
      port: 8603,
      api_key: 'stt-api-key',
    },
    execution: {
      mode: 'gpu',
      cuda_devices: '0',
      compute_type: 'float16',
    },
  })
  const [loading, setLoading] = useState<'start' | 'stop' | 'restart' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [pageLoading, setPageLoading] = useState(true)

  // Test state
  const [testFile, setTestFile] = useState<File | null>(null)
  const [testResult, setTestResult] = useState<any>(null)
  const [testLoading, setTestLoading] = useState(false)
  const [testError, setTestError] = useState<string | null>(null)
  
  // Recording state
  const [isRecording, setIsRecording] = useState(false)
  const [recordedAudio, setRecordedAudio] = useState<Blob | null>(null)
  const [recordingTime, setRecordingTime] = useState(0)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const init = async () => {
      try {
        setPageLoading(true)
        // Try to load existing config and status
        const configRes = await fetch('/api/v1/stt/config')
        if (configRes.ok) {
          const data = await configRes.json()
          if (data.config) setConfig(data.config)
        }

        const statusRes = await fetch('/api/v1/stt/status')
        if (statusRes.ok) {
          const data = await statusRes.json()
          setStatus(data)
        }
      } catch (e) {
        // Service might not exist yet - that's ok
        debugLog('stt', 'STT service not configured yet')
      } finally {
        setPageLoading(false)
      }
    }
    init()
  }, [])

  // Poll status
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch('/api/v1/stt/status')
        if (res.ok) {
          const data = await res.json()
          setStatus(data)
        }
      } catch (e) {
        // Silently fail
      }
    }, 5000)
    return () => clearInterval(interval)
  }, [])

  const runAction = async (action: 'start' | 'stop' | 'restart') => {
    try {
      setLoading(action)
      setError(null)

      const res = await fetch(`/api/v1/stt/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || `Failed to ${action} STT service`)
      }
      
      const result = await res.json()
      setStatus(result.status)
      setSuccess(result.message || `STT service ${action}ed successfully`)
      setTimeout(() => setSuccess(null), 3000)
    } catch (e) {
      setError(e instanceof Error ? e.message : `Failed to ${action} STT service`)
    } finally {
      setLoading(null)
    }
  }

  const updateConfig = (path: string, value: any) => {
    setConfig((prev: STTConfig | null) => {
      if (!prev) return prev
      const next = JSON.parse(JSON.stringify(prev))
      const keys = path.split('.')
      let current: any = next
      for (let i = 0; i < keys.length - 1; i++) {
        if (!current[keys[i]]) current[keys[i]] = {}
        current = current[keys[i]]
      }
      current[keys[keys.length - 1]] = value
      return next
    })
  }

  const saveConfig = async () => {
    try {
      const res = await fetch('/api/v1/stt/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      
      if (!res.ok) throw new Error('Failed to save configuration')
      
      setSuccess('Configuration saved successfully')
      setTimeout(() => setSuccess(null), 3000)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save configuration')
    }
  }

  // Recording functions
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        setRecordedAudio(blob)
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)
      setRecordingTime(0)
      
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1)
      }, 1000)
    } catch (e) {
      setTestError('Failed to access microphone. Please ensure microphone permissions are granted.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setTestFile(file)
      setRecordedAudio(null)
    }
  }

  const testTranscription = async () => {
    const audioData = recordedAudio || testFile
    if (!audioData) {
      setTestError('Please record audio or upload a file first')
      return
    }

    try {
      setTestLoading(true)
      setTestError(null)
      setTestResult(null)

      const formData = new FormData()
      formData.append('file', audioData, audioData instanceof File ? audioData.name : 'recording.webm')
      formData.append('model', config.model.name)
      formData.append('language', config.model.language)
      formData.append('response_format', 'verbose_json')

      const startTime = Date.now()
      
      // Use backend proxy to avoid CORS issues
      const res = await fetch('/api/v1/stt/transcribe', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const errorText = await res.text()
        throw new Error(`Service returned ${res.status}: ${errorText}`)
      }

      const result = await res.json()
      const endTime = Date.now()

      setTestResult({
        ...result,
        processingTime: endTime - startTime,
      })
    } catch (e) {
      setTestError(e instanceof Error ? e.message : 'Failed to transcribe audio')
    } finally {
      setTestLoading(false)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const refreshStatus = async () => {
    try {
      const res = await fetch('/api/v1/stt/status')
      if (res.ok) {
        setStatus(await res.json())
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to refresh STT status')
    }
  }

  const serviceRunning = Boolean(status?.running)
  const serviceStatusColor = serviceRunning ? accentColors.success : accentColors.warning
  const serviceStatusLabel = serviceRunning ? 'Service running' : 'Service stopped'
  const showStartAction = !serviceRunning || loading === 'start'
  const showRestartAction = serviceRunning || loading === 'restart'
  const showStopAction = serviceRunning || loading === 'stop'
  const selectedLanguage = LANGUAGES.find(lang => lang.code === config.model.language)?.name || config.model.language
  const activeAudioSource = isRecording
    ? `Recording ${formatTime(recordingTime)}`
    : recordedAudio
      ? 'Recording ready'
      : testFile
        ? testFile.name
        : 'No audio loaded'
  const commandIconButtonSx = {
    width: 32,
    height: 32,
    borderRadius: '8px',
    border: '1px solid',
    borderColor: 'divider',
    bgcolor: 'action.hover',
    '&:hover': { bgcolor: 'action.selected' },
    '&.Mui-disabled': {
      bgcolor: 'action.disabledBackground',
      color: 'action.disabled',
    },
  }
  const compactSectionCardSx = compactCollapsibleCardSx as Record<string, unknown>
  const sttHeaderActions = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: { xs: 'nowrap', sm: 'wrap' }, justifyContent: 'flex-end' }}>
      {!isNarrow && (
        <Tooltip title="Refresh STT status">
          <IconButton
            size="small"
            aria-label="Refresh STT status"
            onClick={refreshStatus}
            sx={commandIconButtonSx}
          >
            <RefreshIcon sx={{ fontSize: 17 }} />
          </IconButton>
        </Tooltip>
      )}
      <Tooltip title="Save settings">
        <IconButton
          size="small"
          aria-label="Save STT settings"
          onClick={saveConfig}
          sx={commandIconButtonSx}
        >
          <SaveIcon sx={{ fontSize: 17 }} />
        </IconButton>
      </Tooltip>
      {showStartAction && (
        <Tooltip title="Start service">
          <span>
            <IconButton
              size="small"
              aria-label="Start STT service"
              onClick={() => runAction('start')}
              disabled={loading !== null || serviceRunning}
              sx={{
                width: 32,
                height: 32,
                borderRadius: '8px',
                bgcolor: 'primary.main',
                color: 'primary.contrastText',
                '&:hover': { bgcolor: 'primary.dark' },
                '&.Mui-disabled': {
                  bgcolor: 'action.disabledBackground',
                  color: 'action.disabled',
                },
              }}
            >
              {loading === 'start' ? <CircularProgress size={15} color="inherit" /> : <StartIcon sx={{ fontSize: 17 }} />}
            </IconButton>
          </span>
        </Tooltip>
      )}
      {!isNarrow && showRestartAction && (
        <Tooltip title="Restart service">
          <span>
            <IconButton
              size="small"
              aria-label="Restart STT service"
              onClick={() => runAction('restart')}
              disabled={loading !== null}
              sx={commandIconButtonSx}
            >
              {loading === 'restart' ? <CircularProgress size={15} /> : <RestartIcon sx={{ fontSize: 17 }} />}
            </IconButton>
          </span>
        </Tooltip>
      )}
      {(!isNarrow || serviceRunning) && showStopAction && (
        <Tooltip title="Stop service">
          <span>
            <IconButton
              size="small"
              aria-label="Stop STT service"
              color="error"
              onClick={() => runAction('stop')}
              disabled={loading !== null || !serviceRunning}
              sx={commandIconButtonSx}
            >
              {loading === 'stop' ? <CircularProgress size={15} /> : <StopIcon sx={{ fontSize: 17 }} />}
            </IconButton>
          </span>
        </Tooltip>
      )}
    </Box>
  )
  const sttSummaryRow = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
      <Chip
        size="small"
        label={serviceStatusLabel}
        sx={{
          height: 22,
          fontSize: '0.68rem',
          fontWeight: 800,
          color: serviceStatusColor,
          bgcolor: alpha(serviceStatusColor, 0.1),
          border: `1px solid ${alpha(serviceStatusColor, 0.28)}`,
        }}
      />
      <Chip size="small" label={`${config.model.name} / ${config.execution.compute_type}`} sx={{ height: 22, fontSize: '0.68rem', fontWeight: 800 }} />
      <Chip size="small" label={`${selectedLanguage} / ${config.model.task}`} sx={{ height: 22, fontSize: '0.68rem', fontWeight: 800 }} />
      <Typography variant="caption" color="text.secondary" noWrap sx={{ minWidth: 0 }}>
        /v1/audio/transcriptions / {activeAudioSource}
      </Typography>
    </Box>
  )

  if (pageLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    )
  }

  return (
    <PageShell sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      {error && (
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: {
            xs: '1fr',
            lg: 'minmax(0, 0.95fr) minmax(0, 1.05fr)',
          },
          gap: 1.25,
          alignItems: 'start',
        }}
      >
        <SectionCard
          title="Model settings"
          subtitle={isNarrow ? `${serviceStatusLabel} / ${config.model.name} / ${config.execution.compute_type}` : undefined}
          icon={<MicIcon />}
          accentColor={accentColors.info}
          action={sttHeaderActions}
          collapsible={isNarrow}
          defaultExpanded={!isNarrow}
          sx={{ ...compactSectionCardSx, order: { xs: 2, lg: 1 } }}
        >
          {sttSummaryRow}
          <Grid container spacing={1.25}>
            <Grid item xs={12} sm={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Whisper Model
              </Typography>
              <Select
                fullWidth
                size="small"
                value={config.model.name}
                onChange={(e) => {
                  const newName = e.target.value as string
                  const selectedModel = WHISPER_MODELS.find(m => m.name === newName)
                  setConfig((prev: STTConfig | null) => {
                    if (!prev) return prev
                    const next = JSON.parse(JSON.stringify(prev))
                    next.model.name = newName
                    if (selectedModel) next.model.size = selectedModel.size
                    return next
                  })
                }}
              >
                {WHISPER_MODELS.map((model) => (
                  <MenuItem key={model.name} value={model.name}>
                    {model.name} ({model.params}, {model.vram})
                  </MenuItem>
                ))}
              </Select>
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Language
              </Typography>
              <Select
                fullWidth
                size="small"
                value={config.model.language}
                onChange={(e) => updateConfig('model.language', e.target.value)}
              >
                {LANGUAGES.map((lang) => (
                  <MenuItem key={lang.code} value={lang.code}>
                    {lang.name}
                  </MenuItem>
                ))}
              </Select>
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Task
              </Typography>
              <Select
                fullWidth
                size="small"
                value={config.model.task}
                onChange={(e) => updateConfig('model.task', e.target.value)}
              >
                <MenuItem value="transcribe">Transcribe (same language)</MenuItem>
                <MenuItem value="translate">Translate to English</MenuItem>
              </Select>
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Compute Type
              </Typography>
              <Select
                fullWidth
                size="small"
                value={config.execution.compute_type}
                onChange={(e) => updateConfig('execution.compute_type', e.target.value)}
              >
                <MenuItem value="float16">float16 (Recommended for GPU)</MenuItem>
                <MenuItem value="float32">float32 (Higher precision)</MenuItem>
                <MenuItem value="int8">int8 (Lower memory)</MenuItem>
              </Select>
            </Grid>

            <Grid item xs={6} sm={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Port
              </Typography>
              <TextField
                fullWidth
                size="small"
                type="number"
                value={config.server.port}
                onChange={(e) => updateConfig('server.port', parseInt(e.target.value))}
                inputProps={{ min: 1024, max: 65535 }}
              />
            </Grid>
            
            <Grid item xs={6} sm={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                API Key
              </Typography>
              <TextField
                fullWidth
                size="small"
                value={config.server.api_key}
                onChange={(e) => updateConfig('server.api_key', e.target.value)}
              />
            </Grid>
            
            <Grid item xs={6} sm={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Execution Mode
              </Typography>
              <Select
                fullWidth
                size="small"
                value={config.execution.mode}
                onChange={(e) => updateConfig('execution.mode', e.target.value)}
              >
                <MenuItem value="gpu">GPU (CUDA)</MenuItem>
                <MenuItem value="cpu">CPU Only</MenuItem>
              </Select>
            </Grid>
            
            <Grid item xs={6} sm={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                CUDA Device
              </Typography>
              <TextField
                fullWidth
                size="small"
                value={config.execution.cuda_devices}
                onChange={(e) => updateConfig('execution.cuda_devices', e.target.value)}
                disabled={config.execution.mode === 'cpu'}
              />
            </Grid>
          </Grid>
          
        </SectionCard>

        <SectionCard
          title="Transcription test"
          icon={<UploadIcon />}
          accentColor={accentColors.success}
          collapsible
          defaultExpanded
          sx={{ ...compactSectionCardSx, order: { xs: 1, lg: 2 } }}
        >
          {testError && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setTestError(null)}>
              {testError}
            </Alert>
          )}
          
          {!serviceRunning ? (
            <Box
              sx={{
                p: 1.25,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                borderRadius: '8px',
                border: '1px solid',
                borderColor: alpha(accentColors.warning, 0.35),
                bgcolor: alpha(accentColors.warning, 0.08),
              }}
            >
              <Chip
                label="Service stopped"
                size="small"
                sx={{
                  height: 22,
                  bgcolor: alpha(accentColors.warning, 0.14),
                  color: accentColors.warning,
                  fontWeight: 800,
                }}
              />
              <Typography variant="body2" color="text.secondary">
                Start the STT service to record, upload, and transcribe audio.
              </Typography>
            </Box>
          ) : (
          <Grid container spacing={1.5}>
            {/* Recording Section */}
            <Grid item xs={12} md={6}>
              <Box
	                sx={{
	                  p: 1.5,
	                  height: '100%',
                  borderRadius: '8px',
                  border: '1px solid',
                  borderColor: 'divider',
                  bgcolor: 'action.hover',
                }}
              >
                <Typography variant="subtitle2" gutterBottom>
                  Record Audio
                </Typography>
	                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: isRecording ? 1 : 0 }}>
	                  <IconButton
	                    size="small"
	                    color={isRecording ? 'error' : 'primary'}
	                    onClick={isRecording ? stopRecording : startRecording}
	                    sx={{
	                      width: 40,
	                      height: 40,
	                      bgcolor: isRecording ? 'error.light' : 'primary.light',
                      '&:hover': {
                        bgcolor: isRecording ? 'error.main' : 'primary.main',
                      }
                    }}
                  >
                    {isRecording ? <MicOffIcon /> : <MicIcon />}
                  </IconButton>
                  <Box>
                    <Typography variant="body2">
                      {isRecording ? 'Recording...' : recordedAudio ? 'Recording ready' : 'Click to record'}
                    </Typography>
                    {isRecording && (
                      <Typography variant="caption" color="error">
                        {formatTime(recordingTime)}
                      </Typography>
                    )}
                    {recordedAudio && !isRecording && (
                      <Typography variant="caption" color="success.main">
                        Ready to transcribe
                      </Typography>
                    )}
                  </Box>
                </Box>
                {isRecording && <LinearProgress color="error" />}
              </Box>
            </Grid>
            
            {/* Upload Section */}
            <Grid item xs={12} md={6}>
              <Box
	                sx={{
	                  p: 1.5,
	                  height: '100%',
                  borderRadius: '8px',
                  border: '1px solid',
                  borderColor: 'divider',
                  bgcolor: 'action.hover',
                }}
              >
                <Typography variant="subtitle2" gutterBottom>
                  Upload Audio File
                </Typography>
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleFileUpload}
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                />
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
	                  <Button
	                    variant="outlined"
	                    size="small"
	                    startIcon={<UploadIcon />}
	                    onClick={() => fileInputRef.current?.click()}
                  >
                    Choose File
                  </Button>
                  {testFile && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body2" noWrap sx={{ maxWidth: 150 }}>
                        {testFile.name}
                      </Typography>
                      <IconButton size="small" onClick={() => setTestFile(null)}>
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Box>
                  )}
                </Box>
	                <FormHelperText>
	                  Supports WAV, MP3, M4A, FLAC, WebM
	                </FormHelperText>
              </Box>
            </Grid>
            
            <Grid item xs={12}>
	              <Button
	                variant="contained"
	                color="primary"
	                size="small"
	                startIcon={testLoading ? <CircularProgress size={20} /> : <MicIcon />}
                onClick={testTranscription}
                disabled={testLoading || (!recordedAudio && !testFile) || !serviceRunning}
                sx={{ fontWeight: 600 }}
              >
                {testLoading ? 'Transcribing...' : 'Transcribe'}
              </Button>
              {!serviceRunning && (
                <Typography variant="caption" color="error" sx={{ ml: 2 }}>
                  Start the service first
                </Typography>
              )}
            </Grid>
            
            {/* Results */}
            {testResult && (
              <Grid item xs={12}>
                <Alert severity="success" sx={{ mb: 2 }}>
                  Transcription completed in {testResult.processingTime}ms
                </Alert>
                
                <Box
                  sx={{
                    p: 2,
                    mb: 2,
                    borderRadius: '8px',
                    border: '1px solid',
                    borderColor: 'divider',
                    bgcolor: 'action.hover',
                  }}
                >
                  <Typography variant="subtitle2" gutterBottom>
                    Transcribed Text
                  </Typography>
                  <Typography variant="body1" sx={{ 
                    p: 2, 
                    bgcolor: 'grey.900', 
                    borderRadius: '8px',
                    fontStyle: testResult.text ? 'normal' : 'italic',
                    color: 'grey.100'
                  }}>
                    {testResult.text || '(No speech detected)'}
                  </Typography>
                </Box>
                
                <Grid container spacing={2}>
                  <Grid item xs={6} sm={3}>
                    <Box
                      sx={{
                        p: 2,
                        textAlign: 'center',
                        borderRadius: '8px',
                        border: '1px solid',
                        borderColor: 'divider',
                        bgcolor: 'action.hover',
                      }}
                    >
                      <Typography variant="caption" color="text.secondary">
                        Language
                      </Typography>
                      <Typography variant="h6">
                        {testResult.language || 'N/A'}
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Box
                      sx={{
                        p: 2,
                        textAlign: 'center',
                        borderRadius: '8px',
                        border: '1px solid',
                        borderColor: 'divider',
                        bgcolor: 'action.hover',
                      }}
                    >
                      <Typography variant="caption" color="text.secondary">
                        Duration
                      </Typography>
                      <Typography variant="h6">
                        {testResult.duration ? `${testResult.duration.toFixed(1)}s` : 'N/A'}
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Box
                      sx={{
                        p: 2,
                        textAlign: 'center',
                        borderRadius: '8px',
                        border: '1px solid',
                        borderColor: 'divider',
                        bgcolor: 'action.hover',
                      }}
                    >
                      <Typography variant="caption" color="text.secondary">
                        Processing
                      </Typography>
                      <Typography variant="h6">
                        {testResult.processingTime}ms
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Box
                      sx={{
                        p: 2,
                        textAlign: 'center',
                        borderRadius: '8px',
                        border: '1px solid',
                        borderColor: 'divider',
                        bgcolor: 'action.hover',
                      }}
                    >
                      <Typography variant="caption" color="text.secondary">
                        Words
                      </Typography>
                      <Typography variant="h6">
                        {testResult.text?.split(/\s+/).filter(Boolean).length || 0}
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Grid>
            )}
          </Grid>
          )}
        </SectionCard>
      </Box>

      <SectionCard
        title="API usage"
        icon={<SpeedIcon />}
        accentColor={accentColors.primary}
        collapsible
        defaultExpanded={false}
        sx={compactSectionCardSx}
      >
        <Typography variant="subtitle2" gutterBottom>
          Endpoint
        </Typography>
        <Box sx={{ p: 1.25, bgcolor: 'grey.900', borderRadius: '8px', mb: 1.25, minWidth: 0 }}>
          <code style={{ color: accentColors.success, whiteSpace: 'normal', wordBreak: 'break-word' }}>
            POST {status?.endpoint || `http://localhost:${config.server.port}`}/v1/audio/transcriptions
          </code>
        </Box>
        <Typography variant="subtitle2" gutterBottom>
          Example (cURL)
        </Typography>
        <Box sx={{ p: 1.25, bgcolor: 'grey.900', borderRadius: '8px', fontFamily: 'monospace', fontSize: '0.8rem', minWidth: 0 }}>
          <pre style={{ margin: 0, color: 'inherit', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
{`curl ${status?.endpoint || `http://localhost:${config.server.port}`}/v1/audio/transcriptions \\
  -H "Authorization: Bearer ${config.server.api_key}" \\
  -F file="@audio.mp3" \\
  -F model="${config.model.name}" \\
  -F language="${config.model.language}"`}
          </pre>
        </Box>
      </SectionCard>
    </PageShell>
  )
}

export default STTDeployPage
