import React, { useEffect, useState, useRef, useCallback } from 'react'
import {
    Box,
    Grid,
    Typography,
    Card,
    CardContent,
    TextField,
    Button,
    Alert,
    Chip,
    CircularProgress,
    FormHelperText,
    IconButton,
    LinearProgress,
    Slider,
    Paper,
} from '@mui/material'
import {
    PlayArrow as StartIcon,
    Stop as StopIcon,
    RestartAlt as RestartIcon,
    Save as SaveIcon,
    Mic as MicIcon,
    MicOff as MicOffIcon,
    GraphicEq as WaveformIcon,
    RecordVoiceOver as SpeakingIcon,
} from '@mui/icons-material'

interface StreamingSTTConfig {
    model: {
        name: string
    }
    server: {
        host: string
        port: number
        internal_port: number
    }
    vad: {
        threshold: number
        silence_threshold: number
        hysteresis_ms: number
    }
    processing: {
        chunk_size_ms: number
        sample_rate: number
        sentence_end_silence_ms: number
        min_sentence_words: number
        model_buffer_ms: number
    }
    logging: {
        level: string
    }
}

interface StreamingSTTStatus {
    running: boolean
    uptime: number
    endpoint: string
    websocket_url: string
    relay_url: string
    model?: {
        name: string
    }
}

export const StreamingSTTDeployPage: React.FC = () => {
    // Service state
    const [status, setStatus] = useState<StreamingSTTStatus | null>(null)
    const [config, setConfig] = useState<StreamingSTTConfig | null>(null)
    const [loading, setLoading] = useState<'start' | 'stop' | 'restart' | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [success, setSuccess] = useState<string | null>(null)
    const [pageLoading, setPageLoading] = useState(true)

    // Streaming test state
    const [isStreaming, setIsStreaming] = useState(false)
    const [isSpeaking, setIsSpeaking] = useState(false)
    const [partialText, setPartialText] = useState('')
    const [sentences, setSentences] = useState<string[]>([])
    const [streamError, setStreamError] = useState<string | null>(null)

    // Refs
    const wsRef = useRef<WebSocket | null>(null)
    const audioContextRef = useRef<AudioContext | null>(null)
    const processorRef = useRef<ScriptProcessorNode | null>(null)
    const streamRef = useRef<MediaStream | null>(null)

    useEffect(() => {
        const init = async () => {
            try {
                setPageLoading(true)
                const [configRes, statusRes] = await Promise.all([
                    fetch('/api/v1/streaming-stt/config'),
                    fetch('/api/v1/streaming-stt/status')
                ])

                if (configRes.ok) {
                    const data = await configRes.json()
                    if (data.config) setConfig(data.config)
                }

                if (statusRes.ok) {
                    const data = await statusRes.json()
                    setStatus(data)
                }
            } catch (e) {
                console.log('Streaming STT service not configured yet')
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
                const res = await fetch('/api/v1/streaming-stt/status')
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

            const res = await fetch(`/api/v1/streaming-stt/${action}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            })

            if (!res.ok) {
                const err = await res.json()
                throw new Error(err.detail || `Failed to ${action} Streaming STT service`)
            }

            const result = await res.json()
            setStatus(result.status)
            setSuccess(result.message || `Streaming STT service ${action}ed successfully`)
            setTimeout(() => setSuccess(null), 3000)
        } catch (e) {
            setError(e instanceof Error ? e.message : `Failed to ${action} Streaming STT service`)
        } finally {
            setLoading(null)
        }
    }

    const updateConfig = (path: string, value: any) => {
        setConfig((prev) => {
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
            const res = await fetch('/api/v1/streaming-stt/config', {
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

    // WebSocket Streaming Functions
    const startStreaming = useCallback(async () => {
        try {
            setStreamError(null)
            setSentences([])
            setPartialText('')

            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                }
            })
            streamRef.current = stream

            // Create audio context
            const audioContext = new AudioContext({ sampleRate: 16000 })
            audioContextRef.current = audioContext

            // Connect WebSocket
            const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/v1/streaming-stt/ws`
            const ws = new WebSocket(wsUrl)
            wsRef.current = ws

            ws.onopen = () => {
                console.log('WebSocket connected')
                setIsStreaming(true)

                // Set up audio processing
                const source = audioContext.createMediaStreamSource(stream)
                const processor = audioContext.createScriptProcessor(1280, 1, 1) // 80ms at 16kHz
                processorRef.current = processor

                processor.onaudioprocess = (e) => {
                    if (ws.readyState === WebSocket.OPEN) {
                        const inputData = e.inputBuffer.getChannelData(0)
                        const float32Array = new Float32Array(inputData)

                        // Convert to base64
                        const uint8 = new Uint8Array(float32Array.buffer)
                        let binary = ''
                        for (let i = 0; i < uint8.byteLength; i++) {
                            binary += String.fromCharCode(uint8[i])
                        }
                        const base64 = btoa(binary)

                        ws.send(JSON.stringify({
                            type: 'audio_chunk',
                            data: base64
                        }))
                    }
                }

                source.connect(processor)
                processor.connect(audioContext.destination)
            }

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data)

                switch (msg.type) {
                    case 'partial':
                        setPartialText(msg.text || '')
                        break
                    case 'sentence':
                        setSentences(prev => [...prev, msg.text])
                        setPartialText('')
                        break
                    case 'vad':
                        setIsSpeaking(msg.is_speech)
                        break
                    case 'final':
                        if (msg.text) {
                            setSentences(prev => [...prev, msg.text])
                        }
                        setPartialText('')
                        break
                    case 'error':
                        setStreamError(msg.message)
                        break
                }
            }

            ws.onclose = () => {
                console.log('WebSocket closed')
                stopStreaming()
            }

            ws.onerror = (e) => {
                console.error('WebSocket error:', e)
                setStreamError('WebSocket connection error')
                stopStreaming()
            }

        } catch (e) {
            setStreamError(e instanceof Error ? e.message : 'Failed to start streaming')
            stopStreaming()
        }
    }, [])

    const stopStreaming = useCallback(() => {
        // Close WebSocket
        if (wsRef.current) {
            if (wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({ type: 'end' }))
            }
            wsRef.current.close()
            wsRef.current = null
        }

        // Stop audio processing
        if (processorRef.current) {
            processorRef.current.disconnect()
            processorRef.current = null
        }

        if (audioContextRef.current) {
            audioContextRef.current.close()
            audioContextRef.current = null
        }

        // Stop media stream
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop())
            streamRef.current = null
        }

        setIsStreaming(false)
        setIsSpeaking(false)
    }, [])

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
                <WaveformIcon sx={{ fontSize: 40, color: '#8b5cf6' }} />
                <Box>
                    <Typography variant="h4" sx={{ fontWeight: 700 }}>
                        Streaming STT Deployment
                    </Typography>
                    <Typography variant="body1" color="text.secondary">
                        Real-time speech-to-text using NVIDIA Nemotron with WebSocket streaming
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
                                    Model: {status.model?.name || 'Nemotron-0.6B'}
                                </Typography>
                            </>
                        )}
                    </Box>

                    {status?.running && (
                        <Alert severity="info" sx={{ mb: 3 }}>
                            <strong>WebSocket Endpoint:</strong> {status.relay_url}
                            <br />
                            <Typography variant="caption" color="text.secondary">
                                Direct service: {status.websocket_url}
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

            {/* Live Streaming Test */}
            <Card sx={{ mb: 3, borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
                <CardContent>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                        Live Streaming Test
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                        Test real-time transcription with your microphone. Click the button to start streaming.
                    </Typography>

                    {streamError && (
                        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setStreamError(null)}>
                            {streamError}
                        </Alert>
                    )}

                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
                        <IconButton
                            color={isStreaming ? 'error' : 'primary'}
                            onClick={isStreaming ? stopStreaming : startStreaming}
                            disabled={!status?.running}
                            sx={{
                                width: 64,
                                height: 64,
                                bgcolor: isStreaming ? 'error.light' : 'primary.light',
                                '&:hover': {
                                    bgcolor: isStreaming ? 'error.main' : 'primary.main',
                                }
                            }}
                        >
                            {isStreaming ? <MicOffIcon sx={{ fontSize: 32 }} /> : <MicIcon sx={{ fontSize: 32 }} />}
                        </IconButton>
                        <Box>
                            <Typography variant="body1">
                                {isStreaming ? 'Streaming...' : 'Click to start'}
                            </Typography>
                            {isStreaming && (
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    <SpeakingIcon sx={{ color: isSpeaking ? 'success.main' : 'text.disabled', fontSize: 18 }} />
                                    <Typography variant="caption" color={isSpeaking ? 'success.main' : 'text.disabled'}>
                                        {isSpeaking ? 'Speaking detected' : 'Listening...'}
                                    </Typography>
                                </Box>
                            )}
                        </Box>
                        {!status?.running && (
                            <Typography variant="caption" color="error">
                                Start the service first
                            </Typography>
                        )}
                    </Box>

                    {isStreaming && <LinearProgress color="primary" sx={{ mb: 2 }} />}

                    {/* Transcription Output */}
                    <Paper
                        variant="outlined"
                        sx={{
                            p: 2,
                            minHeight: 150,
                            maxHeight: 300,
                            overflow: 'auto',
                            bgcolor: 'grey.900',
                            fontFamily: 'monospace',
                        }}
                    >
                        {sentences.map((sentence, i) => (
                            <Typography key={i} sx={{ color: 'grey.100', mb: 1 }}>
                                {sentence}
                            </Typography>
                        ))}
                        {partialText && (
                            <Typography sx={{ color: 'grey.500', fontStyle: 'italic' }}>
                                {partialText}...
                            </Typography>
                        )}
                        {!sentences.length && !partialText && (
                            <Typography sx={{ color: 'grey.600', fontStyle: 'italic' }}>
                                {isStreaming ? 'Waiting for speech...' : 'Start streaming to see transcription here'}
                            </Typography>
                        )}
                    </Paper>
                </CardContent>
            </Card>

            {/* Configuration */}
            {config && (
                <Card sx={{ mb: 3, borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
                    <CardContent>
                        <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                            Configuration
                        </Typography>

                        <Grid container spacing={3}>
                            <Grid item xs={12} md={4}>
                                <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                                    VAD Threshold
                                </Typography>
                                <Slider
                                    value={config.vad?.threshold || 0.005}
                                    onChange={(_, v) => updateConfig('vad.threshold', v)}
                                    min={0.001}
                                    max={0.05}
                                    step={0.001}
                                    valueLabelDisplay="auto"
                                />
                                <FormHelperText>
                                    Speech detection sensitivity (lower = more sensitive)
                                </FormHelperText>
                            </Grid>

                            <Grid item xs={12} md={4}>
                                <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                                    Sentence End Silence (ms)
                                </Typography>
                                <TextField
                                    fullWidth
                                    type="number"
                                    value={config.processing?.sentence_end_silence_ms || 800}
                                    onChange={(e) => updateConfig('processing.sentence_end_silence_ms', parseInt(e.target.value))}
                                    inputProps={{ min: 200, max: 2000 }}
                                />
                                <FormHelperText>
                                    Silence duration before committing a sentence
                                </FormHelperText>
                            </Grid>

                            <Grid item xs={12} md={4}>
                                <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                                    Model Buffer (ms)
                                </Typography>
                                <TextField
                                    fullWidth
                                    type="number"
                                    value={config.processing?.model_buffer_ms || 896}
                                    onChange={(e) => updateConfig('processing.model_buffer_ms', parseInt(e.target.value))}
                                    inputProps={{ min: 400, max: 2000 }}
                                />
                                <FormHelperText>
                                    Audio buffer before inference (~900ms optimal)
                                </FormHelperText>
                            </Grid>

                            <Grid item xs={12} md={6}>
                                <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                                    Min Sentence Words
                                </Typography>
                                <TextField
                                    fullWidth
                                    type="number"
                                    value={config.processing?.min_sentence_words || 2}
                                    onChange={(e) => updateConfig('processing.min_sentence_words', parseInt(e.target.value))}
                                    inputProps={{ min: 1, max: 10 }}
                                />
                                <FormHelperText>
                                    Minimum words required to commit a sentence
                                </FormHelperText>
                            </Grid>

                            <Grid item xs={12} md={6}>
                                <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                                    Log Level
                                </Typography>
                                <TextField
                                    fullWidth
                                    value={config.logging?.level || 'INFO'}
                                    onChange={(e) => updateConfig('logging.level', e.target.value)}
                                />
                                <FormHelperText>
                                    DEBUG, INFO, WARNING, ERROR
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
            )}

            {/* API Documentation */}
            <Card sx={{ borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
                <CardContent>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                        WebSocket Protocol
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                        Connect to the WebSocket endpoint for real-time streaming transcription.
                    </Typography>

                    <Typography variant="subtitle2" gutterBottom>
                        Endpoint
                    </Typography>
                    <Box sx={{ p: 2, bgcolor: 'grey.900', borderRadius: 1, mb: 2 }}>
                        <code style={{ color: '#8b5cf6' }}>
                            ws://localhost:8700/api/v1/streaming-stt/ws
                        </code>
                    </Box>

                    <Typography variant="subtitle2" gutterBottom>
                        Client → Server Messages
                    </Typography>
                    <Box sx={{ p: 2, bgcolor: 'grey.900', borderRadius: 1, mb: 2, fontFamily: 'monospace', fontSize: '0.875rem' }}>
                        <Box sx={{ color: 'grey.400' }}>{'// Audio chunk (80ms PCM float32, base64)'}</Box>
                        <Box sx={{ color: 'grey.100' }}>{'{"type": "audio_chunk", "data": "<base64>"}'}</Box>
                        <Box sx={{ color: 'grey.400', mt: 1 }}>{'// End stream'}</Box>
                        <Box sx={{ color: 'grey.100' }}>{'{"type": "end"}'}</Box>
                    </Box>

                    <Typography variant="subtitle2" gutterBottom>
                        Server → Client Messages
                    </Typography>
                    <Box sx={{ p: 2, bgcolor: 'grey.900', borderRadius: 1, fontFamily: 'monospace', fontSize: '0.875rem' }}>
                        <Box sx={{ color: 'grey.400' }}>{'// Partial transcription'}</Box>
                        <Box sx={{ color: 'grey.100' }}>{'{"type": "partial", "text": "Hello wo..."}'}</Box>
                        <Box sx={{ color: 'grey.400', mt: 1 }}>{'// Complete sentence'}</Box>
                        <Box sx={{ color: 'grey.100' }}>{'{"type": "sentence", "text": "Hello world.", "confidence": 0.95}'}</Box>
                        <Box sx={{ color: 'grey.400', mt: 1 }}>{'// VAD state change'}</Box>
                        <Box sx={{ color: 'grey.100' }}>{'{"type": "vad", "is_speech": true}'}</Box>
                    </Box>
                </CardContent>
            </Card>
        </Box>
    )
}

export default StreamingSTTDeployPage
