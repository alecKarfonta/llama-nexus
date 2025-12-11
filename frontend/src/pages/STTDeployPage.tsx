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
  IconButton,
  LinearProgress,
  Slider,
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
} from '@mui/icons-material'

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
        console.log('STT service not configured yet')
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
        <MicIcon sx={{ fontSize: 40, color: '#10b981' }} />
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700 }}>
            Speech-to-Text Deployment
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Deploy and manage Whisper STT models for audio transcription
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
                OpenAI-compatible endpoint: POST {status.endpoint}/v1/audio/transcriptions
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
                Whisper Model
              </Typography>
              <Select
                fullWidth
                value={config.model.name}
                onChange={(e) => {
                  const model = WHISPER_MODELS.find(m => m.name === e.target.value)
                  updateConfig('model.name', e.target.value)
                  if (model) updateConfig('model.size', model.size)
                }}
              >
                {WHISPER_MODELS.map((model) => (
                  <MenuItem key={model.name} value={model.name}>
                    {model.name} ({model.params}, {model.vram})
                  </MenuItem>
                ))}
              </Select>
              <FormHelperText>
                {WHISPER_MODELS.find(m => m.name === config.model.name)?.quality} quality, {WHISPER_MODELS.find(m => m.name === config.model.name)?.speed}
              </FormHelperText>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Language
              </Typography>
              <Select
                fullWidth
                value={config.model.language}
                onChange={(e) => updateConfig('model.language', e.target.value)}
              >
                {LANGUAGES.map((lang) => (
                  <MenuItem key={lang.code} value={lang.code}>
                    {lang.name}
                  </MenuItem>
                ))}
              </Select>
              <FormHelperText>
                Auto-detect works well but specifying language improves accuracy
              </FormHelperText>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Task
              </Typography>
              <Select
                fullWidth
                value={config.model.task}
                onChange={(e) => updateConfig('model.task', e.target.value)}
              >
                <MenuItem value="transcribe">Transcribe (same language)</MenuItem>
                <MenuItem value="translate">Translate to English</MenuItem>
              </Select>
              <FormHelperText>
                Translate will convert any language to English
              </FormHelperText>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                Compute Type
              </Typography>
              <Select
                fullWidth
                value={config.execution.compute_type}
                onChange={(e) => updateConfig('execution.compute_type', e.target.value)}
              >
                <MenuItem value="float16">float16 (Recommended for GPU)</MenuItem>
                <MenuItem value="float32">float32 (Higher precision)</MenuItem>
                <MenuItem value="int8">int8 (Lower memory)</MenuItem>
              </Select>
              <FormHelperText>
                float16 offers best speed/quality balance on GPU
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
                Service port (default: 8603)
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
                <MenuItem value="gpu">GPU (CUDA)</MenuItem>
                <MenuItem value="cpu">CPU Only</MenuItem>
              </Select>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                CUDA Device
              </Typography>
              <TextField
                fullWidth
                value={config.execution.cuda_devices}
                onChange={(e) => updateConfig('execution.cuda_devices', e.target.value)}
                disabled={config.execution.mode === 'cpu'}
              />
              <FormHelperText>
                GPU device ID (0, 1, etc.)
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

      {/* Test Transcription */}
      <Card sx={{ mb: 3, borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
            Test Transcription
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Record audio or upload a file to test the STT service.
          </Typography>
          
          {testError && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setTestError(null)}>
              {testError}
            </Alert>
          )}
          
          <Grid container spacing={3}>
            {/* Recording Section */}
            <Grid item xs={12} md={6}>
              <Card variant="outlined" sx={{ p: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Record Audio
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                  <IconButton
                    color={isRecording ? 'error' : 'primary'}
                    onClick={isRecording ? stopRecording : startRecording}
                    sx={{ 
                      width: 56, 
                      height: 56,
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
              </Card>
            </Grid>
            
            {/* Upload Section */}
            <Grid item xs={12} md={6}>
              <Card variant="outlined" sx={{ p: 2 }}>
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
              </Card>
            </Grid>
            
            <Grid item xs={12}>
              <Button
                variant="contained"
                color="primary"
                size="large"
                startIcon={testLoading ? <CircularProgress size={20} /> : <MicIcon />}
                onClick={testTranscription}
                disabled={testLoading || (!recordedAudio && !testFile) || !status?.running}
                sx={{ borderRadius: 1.5, fontWeight: 600 }}
              >
                {testLoading ? 'Transcribing...' : 'Transcribe'}
              </Button>
              {!status?.running && (
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
                
                <Card variant="outlined" sx={{ p: 2, mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Transcribed Text
                  </Typography>
                  <Typography variant="body1" sx={{ 
                    p: 2, 
                    bgcolor: 'grey.900', 
                    borderRadius: 1,
                    fontStyle: testResult.text ? 'normal' : 'italic',
                    color: 'grey.100'
                  }}>
                    {testResult.text || '(No speech detected)'}
                  </Typography>
                </Card>
                
                <Grid container spacing={2}>
                  <Grid item xs={6} sm={3}>
                    <Card variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="caption" color="text.secondary">
                        Language
                      </Typography>
                      <Typography variant="h6">
                        {testResult.language || 'N/A'}
                      </Typography>
                    </Card>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Card variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="caption" color="text.secondary">
                        Duration
                      </Typography>
                      <Typography variant="h6">
                        {testResult.duration ? `${testResult.duration.toFixed(1)}s` : 'N/A'}
                      </Typography>
                    </Card>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Card variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="caption" color="text.secondary">
                        Processing
                      </Typography>
                      <Typography variant="h6">
                        {testResult.processingTime}ms
                      </Typography>
                    </Card>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Card variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="caption" color="text.secondary">
                        Words
                      </Typography>
                      <Typography variant="h6">
                        {testResult.text?.split(/\s+/).filter(Boolean).length || 0}
                      </Typography>
                    </Card>
                  </Grid>
                </Grid>
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
            The STT service provides an OpenAI-compatible API for audio transcription.
          </Typography>
          
          <Typography variant="subtitle2" gutterBottom>
            Endpoint
          </Typography>
          <Box sx={{ p: 2, bgcolor: 'grey.900', borderRadius: 1, mb: 2 }}>
            <code style={{ color: '#10b981' }}>
              POST {status?.endpoint || `http://localhost:${config.server.port}`}/v1/audio/transcriptions
            </code>
          </Box>
          
          <Typography variant="subtitle2" gutterBottom>
            Example (cURL)
          </Typography>
          <Box sx={{ p: 2, bgcolor: 'grey.900', borderRadius: 1, fontFamily: 'monospace', fontSize: '0.8rem', overflowX: 'auto' }}>
            <pre style={{ margin: 0, color: '#e2e8f0' }}>
{`curl ${status?.endpoint || `http://localhost:${config.server.port}`}/v1/audio/transcriptions \\
  -H "Authorization: Bearer ${config.server.api_key}" \\
  -F file="@audio.mp3" \\
  -F model="${config.model.name}" \\
  -F language="${config.model.language}"`}
            </pre>
          </Box>
        </CardContent>
      </Card>
    </Box>
  )
}

export default STTDeployPage

