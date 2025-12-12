import React, { useEffect, useState, useRef } from 'react'
import {
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  TextField,
  Select,
  MenuItem,
  Button,
  Alert,
  Chip,
  CircularProgress,
  FormHelperText,
  Slider,
  IconButton,
} from '@mui/material'
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
} from '@mui/icons-material'

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

// Legacy models (for reference)
const LEGACY_TTS_MODELS = [
  { 
    name: 'piper-en-gb-alan-medium', 
    provider: 'Piper', 
    language: 'English (UK)', 
    voice: 'alan',
    quality: 'Balanced',
    size: '~60MB'
  },
  { 
    name: 'coqui-tacotron2-DDC', 
    provider: 'Coqui', 
    language: 'English', 
    voice: 'default',
    quality: 'High quality',
    size: '~200MB'
  },
  { 
    name: 'coqui-vits', 
    provider: 'Coqui', 
    language: 'Multi', 
    voice: 'default',
    quality: 'Best quality',
    size: '~150MB'
  },
  { 
    name: 'xtts-v2', 
    provider: 'Coqui', 
    language: 'Multi', 
    voice: 'cloneable',
    quality: 'Voice cloning capable',
    size: '~2GB'
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
        console.log('TTS service not configured yet')
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

  if (pageLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    )
  }

  return (
    <Box sx={{ p: { xs: 2, sm: 3, md: 4 }, maxWidth: '100%', mx: 'auto' }}>
      {/* Header */}
      <Box sx={{ mb: 4, display: 'flex', alignItems: 'center', gap: 2 }}>
        <TTSIcon sx={{ fontSize: 40, color: '#8b5cf6' }} />
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700 }}>
            Text-to-Speech Deployment
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Deploy and manage TTS models for voice synthesis
          </Typography>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      {success && (
        <Alert severity="success" sx={{ mb: 3 }} onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      {/* Service Status */}
      <Card sx={{ mb: 3, borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
            Service Status
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
            <Chip
              label={status?.running ? 'Running' : 'Stopped'}
              color={status?.running ? 'success' : 'default'}
              size="small"
            />
            {status?.running && (
              <>
                <Typography variant="body2" color="text.secondary">
                  Uptime: {Math.floor((status.uptime || 0) / 60)}m {Math.floor((status.uptime || 0) % 60)}s
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Model: {status.model?.name || config.model.name}
                </Typography>
              </>
            )}
          </Box>
          
          {status?.endpoint && (
            <Alert severity="info" sx={{ mb: 3 }}>
              <strong>Service Endpoint:</strong> {status.endpoint}
              <br />
              <Typography variant="caption" color="text.secondary">
                OpenAI-compatible endpoint: POST {status.endpoint}/v1/audio/speech
              </Typography>
            </Alert>
          )}
          
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <Button
              variant="contained"
              color="primary"
              startIcon={loading === 'start' ? <CircularProgress size={20} /> : <StartIcon />}
              onClick={() => runAction('start')}
              disabled={loading !== null || status?.running}
              sx={{ borderRadius: 1.5, fontWeight: 600 }}
            >
              Start Service
            </Button>
            <Button
              variant="contained"
              color="warning"
              startIcon={loading === 'restart' ? <CircularProgress size={20} /> : <RestartIcon />}
              onClick={() => runAction('restart')}
              disabled={loading !== null}
              sx={{ borderRadius: 1.5, fontWeight: 600 }}
            >
              Restart
            </Button>
            <Button
              variant="outlined"
              color="error"
              startIcon={loading === 'stop' ? <CircularProgress size={20} /> : <StopIcon />}
              onClick={() => runAction('stop')}
              disabled={loading !== null || !status?.running}
              sx={{ borderRadius: 1.5, fontWeight: 600 }}
            >
              Stop
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Model Configuration */}
      <Card sx={{ mb: 3, borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
            Model Configuration
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                TTS Model
              </Typography>
              <Select
                fullWidth
                value={config.model.name}
                onChange={(e) => updateConfig('model.name', e.target.value)}
              >
                {TTS_MODELS.map((model) => (
                  <MenuItem key={model.name} value={model.name}>
                    {model.name} ({model.provider})
                  </MenuItem>
                ))}
              </Select>
              <FormHelperText>
                {TTS_MODELS.find(m => m.name === config.model.name)?.description || 'Select a model'}
              </FormHelperText>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Voice (OpenAI-compatible)
              </Typography>
              <Select
                fullWidth
                value={config.model.voice}
                onChange={(e) => updateConfig('model.voice', e.target.value)}
              >
                {(status?.available_voices || OPENAI_VOICES).map((voice) => (
                  <MenuItem key={voice} value={voice}>
                    {voice}
                  </MenuItem>
                ))}
              </Select>
              <FormHelperText>
                Voice style for synthesis
              </FormHelperText>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Speed: {config.audio.speed.toFixed(2)}x
              </Typography>
              <Slider
                value={config.audio.speed}
                onChange={(_, value) => updateConfig('audio.speed', value)}
                min={0.25}
                max={4.0}
                step={0.05}
                marks={[
                  { value: 0.5, label: '0.5x' },
                  { value: 1.0, label: '1x' },
                  { value: 2.0, label: '2x' },
                  { value: 4.0, label: '4x' },
                ]}
              />
              <FormHelperText>
                Playback speed multiplier
              </FormHelperText>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Audio Format
              </Typography>
              <Select
                fullWidth
                value={config.audio.format}
                onChange={(e) => updateConfig('audio.format', e.target.value)}
              >
                {AUDIO_FORMATS.map((format) => (
                  <MenuItem key={format.value} value={format.value}>
                    {format.label}
                  </MenuItem>
                ))}
              </Select>
              <FormHelperText>
                Output audio format
              </FormHelperText>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Sample Rate
              </Typography>
              <Select
                fullWidth
                value={config.audio.sample_rate}
                onChange={(e) => updateConfig('audio.sample_rate', e.target.value)}
              >
                <MenuItem value={16000}>16000 Hz (Low)</MenuItem>
                <MenuItem value={22050}>22050 Hz (Standard)</MenuItem>
                <MenuItem value={44100}>44100 Hz (CD Quality)</MenuItem>
                <MenuItem value={48000}>48000 Hz (High)</MenuItem>
              </Select>
              <FormHelperText>
                Audio sample rate
              </FormHelperText>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Port
              </Typography>
              <TextField
                fullWidth
                type="number"
                value={config.server.port}
                onChange={(e) => updateConfig('server.port', parseInt(e.target.value))}
                inputProps={{ min: 1024, max: 65535 }}
              />
              <FormHelperText>
                Service port (default: 8604)
              </FormHelperText>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                API Key
              </Typography>
              <TextField
                fullWidth
                value={config.server.api_key}
                onChange={(e) => updateConfig('server.api_key', e.target.value)}
              />
              <FormHelperText>
                API key for authentication
              </FormHelperText>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Execution Mode
              </Typography>
              <Select
                fullWidth
                value={config.execution.mode}
                onChange={(e) => updateConfig('execution.mode', e.target.value)}
              >
                <MenuItem value="cpu">CPU (Recommended for Piper)</MenuItem>
                <MenuItem value="gpu">GPU (For Coqui/XTTS)</MenuItem>
              </Select>
              <FormHelperText>
                Piper is optimized for CPU, Coqui benefits from GPU
              </FormHelperText>
            </Grid>
          </Grid>
          
          <Box sx={{ mt: 3 }}>
            <Button
              variant="contained"
              startIcon={<SaveIcon />}
              onClick={saveConfig}
              sx={{ borderRadius: 1.5, fontWeight: 600 }}
            >
              Save Configuration
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Test Speech Synthesis */}
      <Card sx={{ mb: 3, borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
            Test Speech Synthesis
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Enter text to generate speech and test the TTS service.
          </Typography>
          
          {testError && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setTestError(null)}>
              {testError}
            </Alert>
          )}
          
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Text to Speak
              </Typography>
              <TextField
                fullWidth
                multiline
                rows={4}
                value={testText}
                onChange={(e) => setTestText(e.target.value)}
                placeholder="Enter text to convert to speech..."
              />
              <FormHelperText>
                {testText.length} characters
              </FormHelperText>
            </Grid>
            
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                <Button
                  variant="contained"
                  color="primary"
                  size="large"
                  startIcon={testLoading ? <CircularProgress size={20} /> : <VolumeIcon />}
                  onClick={testSpeech}
                  disabled={testLoading || !testText.trim() || !status?.running}
                  sx={{ borderRadius: 1.5, fontWeight: 600 }}
                >
                  {testLoading ? 'Generating...' : 'Generate Speech'}
                </Button>
                {!status?.running && (
                  <Typography variant="caption" color="error">
                    Start the service first
                  </Typography>
                )}
              </Box>
            </Grid>
            
            {/* Audio Player */}
            {testResult && (
              <Grid item xs={12}>
                <Alert severity="success" sx={{ mb: 2 }}>
                  Speech generated in {testResult.processingTime}ms
                </Alert>
                
                <Card variant="outlined" sx={{ p: 3 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <IconButton
                      color="primary"
                      onClick={playAudio}
                      sx={{ 
                        width: 64, 
                        height: 64,
                        bgcolor: 'primary.light',
                        '&:hover': { bgcolor: 'primary.main' }
                      }}
                    >
                      {isPlaying ? <StopPlayIcon sx={{ fontSize: 32 }} /> : <PlayIcon sx={{ fontSize: 32 }} />}
                    </IconButton>
                    
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="subtitle1" fontWeight={600}>
                        Generated Audio
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Format: {config.audio.format.toUpperCase()} | Speed: {config.audio.speed}x | Voice: {config.model.voice}
                      </Typography>
                    </Box>
                    
                    <Button
                      variant="outlined"
                      startIcon={<DownloadIcon />}
                      onClick={downloadAudio}
                    >
                      Download
                    </Button>
                  </Box>
                  
                  <audio
                    ref={audioRef}
                    src={testResult.audioUrl}
                    onEnded={handleAudioEnded}
                    style={{ display: 'none' }}
                  />
                </Card>
              </Grid>
            )}
          </Grid>
        </CardContent>
      </Card>

      {/* API Documentation */}
      <Card sx={{ borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
            API Usage
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            The TTS service provides an OpenAI-compatible API for speech synthesis.
          </Typography>
          
          <Typography variant="subtitle2" gutterBottom>
            Endpoint
          </Typography>
          <Box sx={{ p: 2, bgcolor: 'grey.900', borderRadius: 1, mb: 2 }}>
            <code style={{ color: '#8b5cf6' }}>
              POST {status?.endpoint || `http://localhost:${config.server.port}`}/v1/audio/speech
            </code>
          </Box>
          
          <Typography variant="subtitle2" gutterBottom>
            Example (cURL)
          </Typography>
          <Box sx={{ p: 2, bgcolor: 'grey.900', borderRadius: 1, fontFamily: 'monospace', fontSize: '0.8rem', overflowX: 'auto' }}>
            <pre style={{ margin: 0, color: '#e2e8f0' }}>
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
          <Box sx={{ p: 2, bgcolor: 'grey.900', borderRadius: 1, fontFamily: 'monospace', fontSize: '0.8rem', overflowX: 'auto' }}>
            <pre style={{ margin: 0, color: '#e2e8f0' }}>
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
        </CardContent>
      </Card>
    </Box>
  )
}

export default TTSDeployPage


