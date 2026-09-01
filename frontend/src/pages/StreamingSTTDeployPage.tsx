import React, { useEffect, useState, useRef, useCallback } from 'react'
import {
    Box,
    Grid,
    Typography,
    TextField,
    Alert,
    Chip,
    CircularProgress,
    FormHelperText,
    IconButton,
    LinearProgress,
    Slider,
    Tooltip,
    useMediaQuery,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import {
    PlayArrow as StartIcon,
    Stop as StopIcon,
    RestartAlt as RestartIcon,
    Save as SaveIcon,
    Mic as MicIcon,
    MicOff as MicOffIcon,
    GraphicEq as WaveformIcon,
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
    const isNarrow = useMediaQuery('(max-width:599.95px)')

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

    const refreshStatus = useCallback(async () => {
        try {
            const res = await fetch('/api/v1/streaming-stt/status')
            if (res.ok) {
                const data = await res.json()
                setStatus(data)
            }
        } catch (e) {
            // Status polling should not interrupt the active workflow.
        }
    }, [])

    useEffect(() => {
        const init = async () => {
            try {
                setPageLoading(true)
                const [configRes] = await Promise.all([
                    fetch('/api/v1/streaming-stt/config'),
                    refreshStatus()
                ])

                if (configRes.ok) {
                    const data = await configRes.json()
                    if (data.config) setConfig(data.config)
                }
            } catch (e) {
                debugLog('streaming-stt', 'Streaming STT service not configured yet')
            } finally {
                setPageLoading(false)
            }
        }
        init()
    }, [refreshStatus])

    // Poll status
    useEffect(() => {
        const interval = setInterval(() => {
            void refreshStatus()
        }, 5000)
        return () => clearInterval(interval)
    }, [refreshStatus])

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
                debugLog('streaming-stt', 'WebSocket connected')
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
                debugLog('streaming-stt', 'WebSocket closed')
                stopStreaming()
            }

            ws.onerror = (e) => {
                debugLog('streaming-stt', 'WebSocket error', e)
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

    const serviceRunning = Boolean(status?.running)
    const serviceStatusColor = serviceRunning ? accentColors.success : accentColors.warning
    const serviceStatusLabel = serviceRunning ? 'Service running' : 'Service stopped'
    const showStartAction = !serviceRunning || loading === 'start'
    const showRestartAction = serviceRunning || loading === 'restart'
    const showStopAction = serviceRunning || loading === 'stop'
    const modelName = status?.model?.name || config?.model.name || 'Nemotron-0.6B'
    const sampleRate = config?.processing?.sample_rate ?? 16000
    const sampleRateLabel = sampleRate >= 1000 ? `${Number((sampleRate / 1000).toFixed(1))} kHz` : `${sampleRate} Hz`
    const modelBufferMs = config?.processing?.model_buffer_ms ?? 896
    const sentenceSilenceMs = config?.processing?.sentence_end_silence_ms ?? 800
    const vadThreshold = config?.vad?.threshold ?? 0.005
    const transcriptLabel = partialText ? 'Live partial' : `${sentences.length} committed`
    const hasTranscriptContent = Boolean(partialText || sentences.length)
    const micLabel = isStreaming ? (isSpeaking ? 'Speaking' : 'Listening') : 'Mic idle'
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
    const streamingHeaderActions = (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: { xs: 'nowrap', sm: 'wrap' }, justifyContent: 'flex-end' }}>
            {!isNarrow && (
                <Tooltip title="Refresh streaming STT status">
                    <IconButton size="small" aria-label="Refresh streaming STT status" onClick={refreshStatus} sx={commandIconButtonSx}>
                        <RefreshIcon sx={{ fontSize: 17 }} />
                    </IconButton>
                </Tooltip>
            )}
            {showStartAction && (
                <Tooltip title="Start service">
                    <span>
                        <IconButton
                            size="small"
                            aria-label="Start streaming STT service"
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
                            aria-label="Restart streaming STT service"
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
                            aria-label="Stop streaming STT service"
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
            <Tooltip title="Save stream settings">
                <span>
                    <IconButton
                        size="small"
                        aria-label="Save stream settings"
                        onClick={saveConfig}
                        disabled={!config}
                        sx={commandIconButtonSx}
                    >
                        <SaveIcon sx={{ fontSize: 17 }} />
                    </IconButton>
                </span>
            </Tooltip>
        </Box>
    )
    const streamingSummaryRow = (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
            <Chip
                size="small"
                label={`${serviceStatusLabel} / ${micLabel}`}
                sx={{
                    height: 22,
                    fontSize: '0.68rem',
                    fontWeight: 800,
                    color: isStreaming ? accentColors.success : serviceStatusColor,
                    bgcolor: alpha(isStreaming ? accentColors.success : serviceStatusColor, 0.1),
                    border: `1px solid ${alpha(isStreaming ? accentColors.success : serviceStatusColor, 0.28)}`,
                }}
            />
            <Chip size="small" label={`${modelBufferMs} ms buffer`} sx={{ height: 22, fontSize: '0.68rem', fontWeight: 800, display: { xs: 'none', sm: 'inline-flex' } }} />
            <Chip size="small" label={`${sentenceSilenceMs} ms silence`} sx={{ height: 22, fontSize: '0.68rem', fontWeight: 800, display: { xs: 'none', sm: 'inline-flex' } }} />
            <Typography variant="caption" color="text.secondary" noWrap sx={{ minWidth: 0, display: { xs: 'none', sm: 'block' } }}>
                {modelName} / {sampleRateLabel} / {transcriptLabel}
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
                        title="Live transcript"
                        icon={<WaveformIcon />}
                        accentColor={accentColors.primary}
                        action={streamingHeaderActions}
                    >
                        {!serviceRunning ? (
                            <Box
                                sx={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 1,
                                    flexWrap: 'wrap',
                                    p: 1.25,
                                    borderRadius: '8px',
                                    border: `1px solid ${alpha(accentColors.warning, 0.45)}`,
                                    bgcolor: alpha(accentColors.warning, 0.12),
                                }}
                            >
                                <Chip
                                    size="small"
                                    label="Service stopped"
                                    sx={{
                                        height: 22,
                                        fontSize: '0.68rem',
                                        fontWeight: 800,
                                        color: accentColors.warning,
                                        bgcolor: alpha(accentColors.warning, 0.14),
                                    }}
                                />
                                <Typography variant="body2" color="text.secondary">
                                    Start the streaming STT service to open the microphone and capture live transcripts.
                                </Typography>
                            </Box>
                        ) : (
                            <>
                                {streamingSummaryRow}
                                {streamError && (
                                    <Alert severity="error" sx={{ mb: 1.25 }} onClose={() => setStreamError(null)}>
                                        {streamError}
                                    </Alert>
                                )}

                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.25, minWidth: 0 }}>
                                    <IconButton
                                        color={isStreaming ? 'error' : 'primary'}
                                        onClick={isStreaming ? stopStreaming : startStreaming}
                                        sx={{
                                            width: 36,
                                            height: 36,
                                            borderRadius: '8px',
                                            bgcolor: isStreaming ? 'error.main' : 'primary.main',
                                            color: 'primary.contrastText',
                                            '&:hover': {
                                                bgcolor: isStreaming ? 'error.dark' : 'primary.dark',
                                            },
                                        }}
                                    >
                                        {isStreaming ? <MicOffIcon sx={{ fontSize: 18 }} /> : <MicIcon sx={{ fontSize: 18 }} />}
                                    </IconButton>
                                    <Box sx={{ minWidth: 0 }}>
                                        <Typography variant="body2" sx={{ fontWeight: 800, lineHeight: 1.2 }}>
                                            {isStreaming ? micLabel : 'Microphone'}
                                        </Typography>
                                        <Typography variant="caption" color="text.secondary" noWrap>
                                            {`${transcriptLabel} / ${sampleRateLabel}`}
                                        </Typography>
                                    </Box>
                                </Box>

                                {isStreaming && <LinearProgress color="primary" sx={{ mb: 1 }} />}

                                <Box
                                    sx={{
                                        p: 1.5,
                                        height: hasTranscriptContent || isStreaming
                                            ? { xs: 180, lg: streamError ? 150 : 190 }
                                            : 96,
                                        overflow: 'auto',
                                        borderRadius: '8px',
                                        border: '1px solid',
                                        borderColor: 'divider',
                                        bgcolor: 'grey.900',
                                        fontFamily: 'monospace',
                                    }}
                                >
                                    {sentences.map((sentence, i) => (
                                        <Typography key={i} sx={{ color: 'grey.100', mb: 1, fontSize: '0.875rem' }}>
                                            {sentence}
                                        </Typography>
                                    ))}
                                    {partialText && (
                                        <Typography sx={{ color: 'grey.500', fontStyle: 'italic', fontSize: '0.875rem' }}>
                                            {partialText}...
                                        </Typography>
                                    )}
                                    {!sentences.length && !partialText && (
                                        <Typography sx={{ color: 'grey.600', fontStyle: 'italic', fontSize: '0.875rem' }}>
                                            {isStreaming ? 'Waiting for speech...' : 'No transcript yet'}
                                        </Typography>
                                    )}
                                </Box>
                            </>
                        )}
                    </SectionCard>
                </Grid>

                <Grid item xs={12} lg={4.8}>
                    {config && (
                        <SectionCard
                            title="Stream settings"
                            subtitle={isNarrow ? `VAD ${vadThreshold.toFixed(3)} / ${sentenceSilenceMs} ms silence / ${modelBufferMs} ms buffer` : undefined}
                            icon={<SpeedIcon />}
                            accentColor={accentColors.info}
                            collapsible={isNarrow}
                            defaultExpanded={!isNarrow}
                            sx={{ ...compactSectionCardSx }}
                        >
                            <Grid container spacing={1.25}>
                                <Grid item xs={12}>
                                    <Box sx={{ px: 0.5 }}>
                                        <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }}>
                                            VAD threshold: {vadThreshold.toFixed(3)}
                                        </Typography>
                                        <Slider
                                            size="small"
                                            value={config.vad?.threshold || 0.005}
                                            onChange={(_, v) => updateConfig('vad.threshold', v)}
                                            min={0.001}
                                            max={0.05}
                                            step={0.001}
                                            valueLabelDisplay="auto"
                                            sx={{ mt: 0.25 }}
                                        />
                                    </Box>
                                </Grid>

                                <Grid item xs={12} sm={6}>
                                    <TextField
                                        fullWidth
                                        size="small"
                                        label="Sentence silence"
                                        type="number"
                                        value={config.processing?.sentence_end_silence_ms || 800}
                                        onChange={(e) => updateConfig('processing.sentence_end_silence_ms', parseInt(e.target.value))}
                                        inputProps={{ min: 200, max: 2000 }}
                                    />
                                    <FormHelperText sx={{ mx: 0.25 }}>ms before commit</FormHelperText>
                                </Grid>

                                <Grid item xs={12} sm={6}>
                                    <TextField
                                        fullWidth
                                        size="small"
                                        label="Model buffer"
                                        type="number"
                                        value={config.processing?.model_buffer_ms || 896}
                                        onChange={(e) => updateConfig('processing.model_buffer_ms', parseInt(e.target.value))}
                                        inputProps={{ min: 400, max: 2000 }}
                                    />
                                    <FormHelperText sx={{ mx: 0.25 }}>ms before inference</FormHelperText>
                                </Grid>

                                <Grid item xs={12} sm={6}>
                                    <TextField
                                        fullWidth
                                        size="small"
                                        label="Min words"
                                        type="number"
                                        value={config.processing?.min_sentence_words || 2}
                                        onChange={(e) => updateConfig('processing.min_sentence_words', parseInt(e.target.value))}
                                        inputProps={{ min: 1, max: 10 }}
                                    />
                                </Grid>

                                <Grid item xs={12} sm={6}>
                                    <TextField
                                        fullWidth
                                        size="small"
                                        label="Log level"
                                        value={config.logging?.level || 'INFO'}
                                        onChange={(e) => updateConfig('logging.level', e.target.value)}
                                    />
                                </Grid>
                            </Grid>
                        </SectionCard>
                    )}
                </Grid>
            </Grid>

            <Box sx={optionalSectionGridSx}>
                <SectionCard
                    title="WebSocket protocol"
                    icon={<SpeedIcon />}
                    accentColor={accentColors.purple}
                    collapsible
                    defaultExpanded={false}
                    sx={compactCollapsibleCardSx}
                >
                    <Typography variant="subtitle2" gutterBottom>
                        Endpoint
                    </Typography>
                    <Box sx={{ p: 2, bgcolor: 'grey.900', borderRadius: '8px', mb: 2 }}>
                        <code style={{ color: accentColors.purple }}>
                            ws://localhost:8700/api/v1/streaming-stt/ws
                        </code>
                    </Box>

                    <Typography variant="subtitle2" gutterBottom>
                        Client to server messages
                    </Typography>
                    <Box sx={{ p: 2, bgcolor: 'grey.900', borderRadius: '8px', mb: 2, fontFamily: 'monospace', fontSize: '0.875rem' }}>
                        <Box sx={{ color: 'grey.400' }}>{'// Audio chunk (80ms PCM float32, base64)'}</Box>
                        <Box sx={{ color: 'grey.100' }}>{'{"type": "audio_chunk", "data": "<base64>"}'}</Box>
                        <Box sx={{ color: 'grey.400', mt: 1 }}>{'// End stream'}</Box>
                        <Box sx={{ color: 'grey.100' }}>{'{"type": "end"}'}</Box>
                    </Box>

                    <Typography variant="subtitle2" gutterBottom>
                        Server to client messages
                    </Typography>
                    <Box sx={{ p: 2, bgcolor: 'grey.900', borderRadius: '8px', fontFamily: 'monospace', fontSize: '0.875rem' }}>
                        <Box sx={{ color: 'grey.400' }}>{'// Partial transcription'}</Box>
                        <Box sx={{ color: 'grey.100' }}>{'{"type": "partial", "text": "Hello wo..."}'}</Box>
                        <Box sx={{ color: 'grey.400', mt: 1 }}>{'// Complete sentence'}</Box>
                        <Box sx={{ color: 'grey.100' }}>{'{"type": "sentence", "text": "Hello world.", "confidence": 0.95}'}</Box>
                        <Box sx={{ color: 'grey.400', mt: 1 }}>{'// VAD state change'}</Box>
                        <Box sx={{ color: 'grey.100' }}>{'{"type": "vad", "is_speech": true}'}</Box>
                    </Box>
                </SectionCard>
            </Box>
        </PageShell>
    )
}

export default StreamingSTTDeployPage
