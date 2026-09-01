import React, { useEffect, useState, useRef } from 'react'
import {
  Box,
  Grid,
  Typography,
  TextField,
  Select,
  MenuItem,
  Alert,
  Chip,
  CircularProgress,
  FormHelperText,
  Slider,
  IconButton,
  Tooltip,
  useMediaQuery,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  RestartAlt as RestartIcon,
  Save as SaveIcon,
  RecordVoiceOver as TTSIcon,
  VolumeUp as VolumeIcon,
  PlayCircle as PlayIcon,
  StopCircle as StopPlayIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material'
import {
  compactCollapsibleCardSx,
  optionalSectionGridSx,
  PageShell,
  SectionCard,
} from '@/components/common'
import { debugLog } from '@/utils/debug'
import { accentColors } from '@/utils/theme'

interface TTSConfig {
  model: {
    name: string
    voice: string
    language: string
  }
  server: {
    host: string
    port: number
    api_key: string
  }
  execution: {
    mode: string
    cuda_devices: string
  }
  audio: {
    speed: number
    format: string
    sample_rate: number
  }
}

interface TTSStatus {
  running: boolean
  uptime: number
  endpoint: string
  model?: {
    name: string
    voice: string
  }
  available_voices?: string[]
}

// Available TTS models (OpenAI-compatible)
const TTS_MODELS = [
  { 
    name: 'tts-1', 
    provider: 'Piper', 
    description: 'Standard quality, fast generation',
    quality: 'Standard',
    speed: 'Fast'
  },
  { 
    name: 'tts-1-hd', 
    provider: 'Piper', 
    description: 'High definition audio quality',
    quality: 'HD',
    speed: 'Slower'
  },
]

const OPENAI_VOICES = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']

const AUDIO_FORMATS = [
  { value: 'mp3', label: 'MP3 (Compressed)' },
  { value: 'wav', label: 'WAV (Uncompressed)' },
  { value: 'opus', label: 'Opus (Web optimized)' },
  { value: 'flac', label: 'FLAC (Lossless)' },
]

export const TTSDeployPage: React.FC = () => {
  const isNarrow = useMediaQuery('(max-width:599.95px)')

  // Service state
  const [status, setStatus] = useState<TTSStatus | null>(null)
  const [config, setConfig] = useState<TTSConfig>({
    model: {
      name: 'tts-1',
      voice: 'alloy',
      language: 'en',
    },
    server: {
      host: '0.0.0.0',
      port: 8604,
      api_key: 'tts-api-key',
    },
    execution: {
      mode: 'cpu',
      cuda_devices: '0',
    },
    audio: {
      speed: 1.0,
      format: 'mp3',
      sample_rate: 22050,
    },
  })
  const [loading, setLoading] = useState<'start' | 'stop' | 'restart' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [pageLoading, setPageLoading] = useState(true)

  // Test state
  const [testText, setTestText] = useState('Hello! This is a test of the text-to-speech system. The quick brown fox jumps over the lazy dog.')
  const [testResult, setTestResult] = useState<{ audioUrl: string; processingTime: number } | null>(null)
  const [testLoading, setTestLoading] = useState(false)
  const [testError, setTestError] = useState<string | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    const init = async () => {
      try {
        setPageLoading(true)
        const configRes = await fetch('/api/v1/tts/config')
        if (configRes.ok) {
          const data = await configRes.json()
          if (data.config) setConfig(data.config)
        }
        
        const statusRes = await fetch('/api/v1/tts/status')
        if (statusRes.ok) {
          const data = await statusRes.json()
          setStatus(data)
        }
      } catch (e) {
        debugLog('tts', 'TTS service not configured yet')
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
        const res = await fetch('/api/v1/tts/status')
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

  // Clean up audio on unmount
  useEffect(() => {
    return () => {
      if (testResult?.audioUrl) {
        URL.revokeObjectURL(testResult.audioUrl)
      }
    }
  }, [testResult])

  const runAction = async (action: 'start' | 'stop' | 'restart') => {
    try {
      setLoading(action)
      setError(null)
      
      const res = await fetch(`/api/v1/tts/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || `Failed to ${action} TTS service`)
      }
      
      const result = await res.json()
      setStatus(result.status)
      setSuccess(result.message || `TTS service ${action}ed successfully`)
      setTimeout(() => setSuccess(null), 3000)
    } catch (e) {
      setError(e instanceof Error ? e.message : `Failed to ${action} TTS service`)
    } finally {
      setLoading(null)
    }
  }

  const updateConfig = (path: string, value: any) => {
    const keys = path.split('.')
    const newConfig = JSON.parse(JSON.stringify(config))
    let current: any = newConfig
    for (let i = 0; i < keys.length - 1; i++) {
      if (!current[keys[i]]) current[keys[i]] = {}
      current = current[keys[i]]
    }
    current[keys[keys.length - 1]] = value
    setConfig(newConfig)
  }

  const saveConfig = async () => {
    try {
      const res = await fetch('/api/v1/tts/config', {
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

  const testSpeech = async () => {
    if (!testText.trim()) {
      setTestError('Please enter some text to synthesize')
      return
    }

    try {
      setTestLoading(true)
      setTestError(null)
      
      // Clean up previous audio
      if (testResult?.audioUrl) {
        URL.revokeObjectURL(testResult.audioUrl)
      }
      setTestResult(null)

      const startTime = Date.now()
      
      // Use backend proxy to avoid CORS issues
      const res = await fetch('/api/v1/tts/synthesize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: config.model.name,
          input: testText,
          voice: config.model.voice,
          speed: config.audio.speed,
          response_format: config.audio.format,
        }),
      })

      if (!res.ok) {
        const errorText = await res.text()
        throw new Error(`Service returned ${res.status}: ${errorText}`)
      }

      const blob = await res.blob()
      const audioUrl = URL.createObjectURL(blob)
      const endTime = Date.now()

      setTestResult({
        audioUrl,
        processingTime: endTime - startTime,
      })
    } catch (e) {
      setTestError(e instanceof Error ? e.message : 'Failed to generate speech')
    } finally {
      setTestLoading(false)
    }
  }

  const playAudio = () => {
    if (audioRef.current && testResult?.audioUrl) {
      if (isPlaying) {
        audioRef.current.pause()
        audioRef.current.currentTime = 0
        setIsPlaying(false)
      } else {
        audioRef.current.play()
        setIsPlaying(true)
      }
    }
  }

  const downloadAudio = () => {
    if (testResult?.audioUrl) {
      const a = document.createElement('a')
      a.href = testResult.audioUrl
      a.download = `tts-output.${config.audio.format}`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
    }
  }

  const handleAudioEnded = () => {
    setIsPlaying(false)
  }

  const refreshStatus = async () => {
    try {
      const res = await fetch('/api/v1/tts/status')
      if (res.ok) {
        setStatus(await res.json())
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to refresh TTS status')
    }
  }

  const serviceRunning = Boolean(status?.running)
  const serviceStatusColor = serviceRunning ? accentColors.success : accentColors.warning
  const serviceStatusLabel = serviceRunning ? 'Service running' : 'Service stopped'
  const showStartAction = !serviceRunning || loading === 'start'
  const showRestartAction = serviceRunning || loading === 'restart'
  const showStopAction = serviceRunning || loading === 'stop'
  const availableVoiceCount = (status?.available_voices || OPENAI_VOICES).length
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
  const ttsHeaderActions = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: { xs: 'nowrap', sm: 'wrap' }, justifyContent: 'flex-end' }}>
      {!isNarrow && (
        <Tooltip title="Refresh TTS status">
          <IconButton size="small" aria-label="Refresh TTS status" onClick={refreshStatus} sx={commandIconButtonSx}>
            <RefreshIcon sx={{ fontSize: 17 }} />
          </IconButton>
        </Tooltip>
      )}
      {showStartAction && (
        <Tooltip title="Start service">
          <span>
            <IconButton
              size="small"
              aria-label="Start TTS service"
              onClick={() => runAction('start')}
              disabled={loading !== null || serviceRunning}
              sx={{
                ...commandIconButtonSx,
                bgcolor: 'primary.main',
                color: 'primary.contrastText',
                borderColor: 'primary.main',
                '&:hover': { bgcolor: 'primary.dark', borderColor: 'primary.dark' },
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
              aria-label="Restart TTS service"
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
              aria-label="Stop TTS service"
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
      <Tooltip title="Save TTS settings">
        <span>
          <IconButton
            size="small"
            aria-label="Save TTS settings"
            onClick={saveConfig}
            sx={commandIconButtonSx}
          >
            <SaveIcon sx={{ fontSize: 17 }} />
          </IconButton>
        </span>
      </Tooltip>
    </Box>
  )
  const ttsSummaryRow = (
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
      <Chip size="small" label={`${config.model.name} / ${config.model.voice}`} sx={{ height: 22, fontSize: '0.68rem', fontWeight: 800 }} />
      <Chip size="small" label={`${config.audio.format.toUpperCase()} / ${config.audio.speed.toFixed(2)}x`} sx={{ height: 22, fontSize: '0.68rem', fontWeight: 800, display: { xs: 'none', sm: 'inline-flex' } }} />
      <Chip size="small" label={`${availableVoiceCount.toLocaleString()} voices`} sx={{ height: 22, fontSize: '0.68rem', fontWeight: 800, display: { xs: 'none', sm: 'inline-flex' } }} />
      <Typography variant="caption" color="text.secondary" noWrap sx={{ minWidth: 0, display: { xs: 'none', sm: 'block' } }}>
        /v1/audio/speech
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

      <Grid container spacing={1.5} alignItems="flex-start">
        <Grid item xs={12} lg={7.2}>
          <SectionCard
            title="Speech test"
            icon={<VolumeIcon />}
            accentColor={accentColors.success}
            action={ttsHeaderActions}
          >
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
                  Start the TTS service to generate and play speech.
                </Typography>
              </Box>
            ) : (
              <>
                {ttsSummaryRow}
                {testError && (
                  <Alert severity="error" sx={{ mb: 1.25 }} onClose={() => setTestError(null)}>
                    {testError}
                  </Alert>
                )}

                <TextField
                  fullWidth
                  multiline
                  minRows={3}
                  maxRows={3}
                  value={testText}
                  onChange={(e) => setTestText(e.target.value)}
                  placeholder="Enter text to convert to speech..."
                  sx={{
                    '& .MuiInputBase-input': {
                      fontSize: '0.875rem',
                      lineHeight: 1.45,
                    },
                  }}
                />

                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1.25, minWidth: 0 }}>
                  <Tooltip title="Generate speech">
                    <span>
                      <IconButton
                        color="primary"
                        aria-label="Generate speech"
                        onClick={testSpeech}
                        disabled={testLoading || !testText.trim()}
                        sx={{
                          width: 36,
                          height: 36,
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
                        {testLoading ? <CircularProgress size={16} color="inherit" /> : <VolumeIcon sx={{ fontSize: 18 }} />}
                      </IconButton>
                    </span>
                  </Tooltip>

                  <Box sx={{ minWidth: 0, flex: 1 }}>
                    <Typography variant="body2" sx={{ fontWeight: 800, lineHeight: 1.2 }}>
                      {testLoading ? 'Generating audio' : testResult ? `Generated in ${testResult.processingTime}ms` : `${testText.length} characters`}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" noWrap>
                      {`${config.model.voice} / ${config.audio.format.toUpperCase()} / ${config.audio.speed.toFixed(2)}x`}
                    </Typography>
                  </Box>

                  {testResult && (
                    <>
                      <Tooltip title={isPlaying ? 'Stop playback' : 'Play generated audio'}>
                        <IconButton size="small" color="primary" onClick={playAudio} sx={commandIconButtonSx}>
                          {isPlaying ? <StopPlayIcon sx={{ fontSize: 17 }} /> : <PlayIcon sx={{ fontSize: 17 }} />}
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Download generated audio">
                        <IconButton size="small" onClick={downloadAudio} sx={commandIconButtonSx}>
                          <DownloadIcon sx={{ fontSize: 17 }} />
                        </IconButton>
                      </Tooltip>
                    </>
                  )}
                </Box>

                {testResult && (
                  <audio
                    ref={audioRef}
                    src={testResult.audioUrl}
                    onEnded={handleAudioEnded}
                    style={{ display: 'none' }}
                  />
                )}
              </>
            )}
          </SectionCard>
        </Grid>

        <Grid item xs={12} lg={4.8}>
          <SectionCard
            title="Voice settings"
            subtitle={isNarrow ? `${config.model.name} / ${config.model.voice} / ${config.audio.speed.toFixed(2)}x` : undefined}
            icon={<TTSIcon />}
            accentColor={accentColors.purple}
            collapsible={isNarrow}
            defaultExpanded={!isNarrow}
            sx={{ ...compactSectionCardSx }}
          >
            <Grid container spacing={1.25}>
              <Grid item xs={12} sm={6}>
                <Select
                  fullWidth
                  size="small"
                  value={config.model.name}
                  onChange={(e) => updateConfig('model.name', e.target.value)}
                  displayEmpty
                >
                  {TTS_MODELS.map((model) => (
                    <MenuItem key={model.name} value={model.name}>
                      {model.name} ({model.provider})
                    </MenuItem>
                  ))}
                </Select>
                <FormHelperText sx={{ mx: 0.25 }}>Model</FormHelperText>
              </Grid>

              <Grid item xs={12} sm={6}>
                <Select
                  fullWidth
                  size="small"
                  value={config.model.voice}
                  onChange={(e) => updateConfig('model.voice', e.target.value)}
                >
                  {(status?.available_voices || OPENAI_VOICES).map((voice) => (
                    <MenuItem key={voice} value={voice}>
                      {voice}
                    </MenuItem>
                  ))}
                </Select>
                <FormHelperText sx={{ mx: 0.25 }}>Voice</FormHelperText>
              </Grid>

              <Grid item xs={12}>
                <Box sx={{ px: 0.5 }}>
                  <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }}>
                    Speed: {config.audio.speed.toFixed(2)}x
                  </Typography>
                  <Slider
                    size="small"
                    value={config.audio.speed}
                    onChange={(_, value) => updateConfig('audio.speed', value)}
                    min={0.25}
                    max={4.0}
                    step={0.05}
                    valueLabelDisplay="auto"
                    sx={{ mt: 0.25 }}
                  />
                </Box>
              </Grid>

              <Grid item xs={12} sm={6}>
                <Select
                  fullWidth
                  size="small"
                  value={config.audio.format}
                  onChange={(e) => updateConfig('audio.format', e.target.value)}
                >
                  {AUDIO_FORMATS.map((format) => (
                    <MenuItem key={format.value} value={format.value}>
                      {format.label}
                    </MenuItem>
                  ))}
                </Select>
                <FormHelperText sx={{ mx: 0.25 }}>Format</FormHelperText>
              </Grid>

              <Grid item xs={12} sm={6}>
                <Select
                  fullWidth
                  size="small"
                  value={config.audio.sample_rate}
                  onChange={(e) => updateConfig('audio.sample_rate', e.target.value)}
                >
                  <MenuItem value={16000}>16000 Hz</MenuItem>
                  <MenuItem value={22050}>22050 Hz</MenuItem>
                  <MenuItem value={44100}>44100 Hz</MenuItem>
                  <MenuItem value={48000}>48000 Hz</MenuItem>
                </Select>
                <FormHelperText sx={{ mx: 0.25 }}>Sample rate</FormHelperText>
              </Grid>

            </Grid>
          </SectionCard>
        </Grid>
      </Grid>

      <Box sx={optionalSectionGridSx}>
        <SectionCard
          title="API / service"
          icon={<SpeedIcon />}
          accentColor={accentColors.primary}
          collapsible
          defaultExpanded={false}
          sx={compactCollapsibleCardSx}
        >
          <Grid container spacing={1.25} sx={{ mb: 2 }}>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                size="small"
                label="Port"
                type="number"
                value={config.server.port}
                onChange={(e) => updateConfig('server.port', parseInt(e.target.value))}
                inputProps={{ min: 1024, max: 65535 }}
              />
            </Grid>

            <Grid item xs={12} sm={4}>
              <Select
                fullWidth
                size="small"
                value={config.execution.mode}
                onChange={(e) => updateConfig('execution.mode', e.target.value)}
              >
                <MenuItem value="cpu">CPU</MenuItem>
                <MenuItem value="gpu">GPU</MenuItem>
              </Select>
              <FormHelperText sx={{ mx: 0.25 }}>Execution</FormHelperText>
            </Grid>

            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                size="small"
                label="API key"
                value={config.server.api_key}
                onChange={(e) => updateConfig('server.api_key', e.target.value)}
              />
            </Grid>
          </Grid>

          <Typography variant="subtitle2" gutterBottom>
            Endpoint
          </Typography>
          <Box sx={{ p: 2, bgcolor: 'grey.900', borderRadius: '8px', mb: 2 }}>
            <code style={{ color: accentColors.purple }}>
              POST {status?.endpoint || `http://localhost:${config.server.port}`}/v1/audio/speech
            </code>
          </Box>
          
          <Typography variant="subtitle2" gutterBottom>
            Example (cURL)
          </Typography>
          <Box sx={{ p: 2, bgcolor: 'grey.900', borderRadius: '8px', fontFamily: 'monospace', fontSize: '0.8rem', overflowX: 'auto' }}>
            <pre style={{ margin: 0, color: 'inherit' }}>
{`curl ${status?.endpoint || `http://localhost:${config.server.port}`}/v1/audio/speech \\
  -H "Authorization: Bearer ${config.server.api_key}" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${config.model.name}",
    "input": "Hello, this is a test.",
    "voice": "${config.model.voice}",
    "speed": ${config.audio.speed}
  }' \\
  --output speech.${config.audio.format}`}
            </pre>
          </Box>
          
          <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
            Example (Python)
          </Typography>
          <Box sx={{ p: 2, bgcolor: 'grey.900', borderRadius: '8px', fontFamily: 'monospace', fontSize: '0.8rem', overflowX: 'auto' }}>
            <pre style={{ margin: 0, color: 'inherit' }}>
{`from openai import OpenAI

client = OpenAI(
    api_key="${config.server.api_key}",
    base_url="${status?.endpoint || `http://localhost:${config.server.port}`}/v1"
)

response = client.audio.speech.create(
    model="${config.model.name}",
    voice="${config.model.voice}",
    input="Hello, this is a test.",
    speed=${config.audio.speed}
)

response.stream_to_file("speech.${config.audio.format}")`}
            </pre>
          </Box>
        </SectionCard>
      </Box>
    </PageShell>
  )
}

export default TTSDeployPage
