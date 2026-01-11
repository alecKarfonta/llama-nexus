import React, { useState, useEffect, useCallback } from 'react'
import {
    Box,
    Typography,
    Card,
    CardContent,
    Button,
    IconButton,
    TextField,
    Chip,
    Switch,
    Slider,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Alert,
    CircularProgress,
    Divider,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    alpha,
} from '@mui/material'
import {
    Reddit as RedditIcon,
    PlayArrow as PlayIcon,
    Stop as StopIcon,
    Refresh as RefreshIcon,
    Visibility as ViewIcon,
    Settings as SettingsIcon,
    Timer as TimerIcon,
    Dataset as DatasetIcon,
} from '@mui/icons-material'

interface CrawlerStatus {
    running: boolean
    last_run: string | null
    next_run: string | null
    last_run_examples: number
    total_examples: number
    error: string | null
}

interface CrawlerConfig {
    enabled: boolean
    interval_hours: number
    max_per_run: number
    subreddits: string[]
}

interface CrawlerStats {
    total_examples: number
    subreddit_counts: Record<string, number>
    avg_output_length: number
    instruction_types: Record<string, number>
    seen_posts: number
    estimated_tokens: number
}

interface Sample {
    instruction: string
    input: string
    output: string
}

const RedditCrawlerCard: React.FC = () => {
    const [status, setStatus] = useState<CrawlerStatus | null>(null)
    const [config, setConfig] = useState<CrawlerConfig | null>(null)
    const [stats, setStats] = useState<CrawlerStats | null>(null)
    const [loading, setLoading] = useState(false)
    const [runningNow, setRunningNow] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [success, setSuccess] = useState<string | null>(null)

    // Dialogs
    const [settingsOpen, setSettingsOpen] = useState(false)
    const [samplesOpen, setSamplesOpen] = useState(false)
    const [samples, setSamples] = useState<Sample[]>([])
    const [samplesLoading, setSamplesLoading] = useState(false)

    // Temp config for editing
    const [tempConfig, setTempConfig] = useState<CrawlerConfig | null>(null)

    const fetchStatus = useCallback(async () => {
        try {
            const res = await fetch('/api/v1/reddit/status')
            const data = await res.json()
            setStatus(data.status)
            setConfig(data.config)
        } catch (e) {
            console.error('Failed to fetch status:', e)
        }
    }, [])

    const fetchStats = useCallback(async () => {
        try {
            const res = await fetch('/api/v1/reddit/stats')
            const data = await res.json()
            setStats(data)
        } catch (e) {
            console.error('Failed to fetch stats:', e)
        }
    }, [])

    useEffect(() => {
        fetchStatus()
        fetchStats()
        const interval = setInterval(() => {
            fetchStatus()
            fetchStats()
        }, 10000)
        return () => clearInterval(interval)
    }, [fetchStatus, fetchStats])

    const handleToggle = async () => {
        setLoading(true)
        setError(null)
        try {
            const endpoint = status?.running ? '/api/v1/reddit/stop' : '/api/v1/reddit/start'
            const res = await fetch(endpoint, { method: 'POST' })
            const data = await res.json()
            setStatus(data.status)
            setSuccess(status?.running ? 'Crawler stopped' : 'Crawler started')
            setTimeout(() => setSuccess(null), 3000)
        } catch (e) {
            setError('Failed to toggle crawler')
        }
        setLoading(false)
    }

    const handleRunNow = async () => {
        setRunningNow(true)
        setError(null)
        try {
            const res = await fetch('/api/v1/reddit/run-now', { method: 'POST' })
            const data = await res.json()
            if (data.result?.success) {
                setSuccess(`Added ${data.result.examples_added} examples`)
                fetchStats()
            } else {
                setError(data.result?.error || 'Run failed')
            }
            setStatus(data.status)
            setTimeout(() => setSuccess(null), 3000)
        } catch (e) {
            setError('Failed to run crawler')
        }
        setRunningNow(false)
    }

    const handleSaveConfig = async () => {
        if (!tempConfig) return
        setLoading(true)
        try {
            const res = await fetch('/api/v1/reddit/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(tempConfig),
            })
            const data = await res.json()
            setConfig(data.config)
            setSettingsOpen(false)
            setSuccess('Settings saved')
            setTimeout(() => setSuccess(null), 3000)
        } catch (e) {
            setError('Failed to save settings')
        }
        setLoading(false)
    }

    const handleViewSamples = async () => {
        setSamplesLoading(true)
        setSamplesOpen(true)
        try {
            const res = await fetch('/api/v1/reddit/samples?limit=20')
            const data = await res.json()
            setSamples(data.samples || [])
        } catch (e) {
            console.error('Failed to fetch samples:', e)
        }
        setSamplesLoading(false)
    }

    const formatTime = (iso: string | null) => {
        if (!iso) return 'Never'
        return new Date(iso).toLocaleString()
    }

    return (
        <>
            <Card
                sx={{
                    mt: 2,
                    background: 'linear-gradient(145deg, rgba(255, 87, 34, 0.05) 0%, rgba(26, 26, 46, 0.8) 100%)',
                    backdropFilter: 'blur(12px)',
                    border: '1px solid rgba(255, 87, 34, 0.15)',
                    borderRadius: 2,
                }}
            >
                <CardContent sx={{ p: 2 }}>
                    {/* Header */}
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                            <RedditIcon sx={{ color: '#ff5722', fontSize: 24 }} />
                            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                                Reddit Crawler
                            </Typography>
                            {status?.running && (
                                <Chip
                                    label="Running"
                                    size="small"
                                    sx={{
                                        height: 20,
                                        bgcolor: alpha('#10b981', 0.1),
                                        color: '#34d399',
                                        fontSize: '0.625rem',
                                    }}
                                />
                            )}
                        </Box>
                        <Box sx={{ display: 'flex', gap: 0.5 }}>
                            <IconButton size="small" onClick={() => { setTempConfig(config); setSettingsOpen(true); }}>
                                <SettingsIcon fontSize="small" />
                            </IconButton>
                            <IconButton size="small" onClick={fetchStatus}>
                                <RefreshIcon fontSize="small" />
                            </IconButton>
                        </Box>
                    </Box>

                    {error && <Alert severity="error" sx={{ mb: 2, py: 0.5 }}>{error}</Alert>}
                    {success && <Alert severity="success" sx={{ mb: 2, py: 0.5 }}>{success}</Alert>}

                    {/* Stats */}
                    {stats && (
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mb: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="caption" color="text.secondary">Total Examples</Typography>
                                <Typography variant="body2" fontWeight={600} color="#ff5722">
                                    {stats.total_examples.toLocaleString()}
                                </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="caption" color="text.secondary">Unique Posts</Typography>
                                <Typography variant="body2" fontWeight={600}>{stats.seen_posts.toLocaleString()}</Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="caption" color="text.secondary">Est. Tokens</Typography>
                                <Typography variant="body2" fontWeight={600}>{(stats.estimated_tokens / 1000).toFixed(0)}k</Typography>
                            </Box>
                            {status?.last_run && (
                                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <Typography variant="caption" color="text.secondary">Last Run</Typography>
                                    <Typography variant="caption">{formatTime(status.last_run)}</Typography>
                                </Box>
                            )}
                            {status?.running && status?.next_run && (
                                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <Typography variant="caption" color="text.secondary">Next Run</Typography>
                                    <Typography variant="caption">{formatTime(status.next_run)}</Typography>
                                </Box>
                            )}
                        </Box>
                    )}

                    {/* Top Subreddits */}
                    {stats && Object.keys(stats.subreddit_counts).length > 0 && (
                        <Box sx={{ mb: 2 }}>
                            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                                Top Subreddits
                            </Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                                {Object.entries(stats.subreddit_counts).slice(0, 5).map(([sub, count]) => (
                                    <Chip
                                        key={sub}
                                        label={`r/${sub} (${count})`}
                                        size="small"
                                        sx={{
                                            height: 20,
                                            fontSize: '0.6rem',
                                            bgcolor: alpha('#ff5722', 0.1),
                                            color: '#ff7043',
                                        }}
                                    />
                                ))}
                            </Box>
                        </Box>
                    )}

                    <Divider sx={{ my: 1.5 }} />

                    {/* Actions */}
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                        <Button
                            size="small"
                            variant={status?.running ? 'outlined' : 'contained'}
                            startIcon={status?.running ? <StopIcon /> : <PlayIcon />}
                            onClick={handleToggle}
                            disabled={loading}
                            sx={status?.running ? {} : { bgcolor: '#ff5722', '&:hover': { bgcolor: '#f4511e' } }}
                        >
                            {status?.running ? 'Stop' : 'Start'}
                        </Button>
                        <Button
                            size="small"
                            variant="outlined"
                            startIcon={runningNow ? <CircularProgress size={14} /> : <RefreshIcon />}
                            onClick={handleRunNow}
                            disabled={runningNow}
                        >
                            Run Now
                        </Button>
                        <Button
                            size="small"
                            variant="outlined"
                            startIcon={<ViewIcon />}
                            onClick={handleViewSamples}
                        >
                            View Samples
                        </Button>
                    </Box>
                </CardContent>
            </Card>

            {/* Settings Dialog */}
            <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)} maxWidth="sm" fullWidth>
                <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <SettingsIcon sx={{ color: '#ff5722' }} />
                    Reddit Crawler Settings
                </DialogTitle>
                <DialogContent>
                    {tempConfig && (
                        <Box sx={{ pt: 1, display: 'flex', flexDirection: 'column', gap: 3 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                <Typography variant="body2">Auto-start on boot</Typography>
                                <Switch
                                    checked={tempConfig.enabled}
                                    onChange={(e) => setTempConfig({ ...tempConfig, enabled: e.target.checked })}
                                    color="warning"
                                />
                            </Box>

                            <Box>
                                <Typography variant="body2" gutterBottom>
                                    Crawl Interval: {tempConfig.interval_hours} hours
                                </Typography>
                                <Slider
                                    value={tempConfig.interval_hours}
                                    onChange={(_, v) => setTempConfig({ ...tempConfig, interval_hours: v as number })}
                                    min={1}
                                    max={24}
                                    marks={[
                                        { value: 1, label: '1h' },
                                        { value: 6, label: '6h' },
                                        { value: 12, label: '12h' },
                                        { value: 24, label: '24h' },
                                    ]}
                                    sx={{ color: '#ff5722' }}
                                />
                            </Box>

                            <Box>
                                <Typography variant="body2" gutterBottom>
                                    Max Examples Per Run: {tempConfig.max_per_run}
                                </Typography>
                                <Slider
                                    value={tempConfig.max_per_run}
                                    onChange={(_, v) => setTempConfig({ ...tempConfig, max_per_run: v as number })}
                                    min={10}
                                    max={200}
                                    step={10}
                                    marks={[
                                        { value: 10, label: '10' },
                                        { value: 50, label: '50' },
                                        { value: 100, label: '100' },
                                        { value: 200, label: '200' },
                                    ]}
                                    sx={{ color: '#ff5722' }}
                                />
                            </Box>

                            <TextField
                                fullWidth
                                multiline
                                rows={4}
                                label="Subreddits (comma-separated)"
                                value={tempConfig.subreddits.join(', ')}
                                onChange={(e) =>
                                    setTempConfig({
                                        ...tempConfig,
                                        subreddits: e.target.value.split(',').map((s) => s.trim()).filter(Boolean),
                                    })
                                }
                                size="small"
                                helperText={`${tempConfig.subreddits.length} subreddits configured`}
                            />
                        </Box>
                    )}
                </DialogContent>
                <DialogActions sx={{ p: 2 }}>
                    <Button onClick={() => setSettingsOpen(false)}>Cancel</Button>
                    <Button variant="contained" onClick={handleSaveConfig} disabled={loading} sx={{ bgcolor: '#ff5722' }}>
                        Save Settings
                    </Button>
                </DialogActions>
            </Dialog>

            {/* Samples Dialog */}
            <Dialog open={samplesOpen} onClose={() => setSamplesOpen(false)} maxWidth="md" fullWidth>
                <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <DatasetIcon sx={{ color: '#ff5722' }} />
                    Reddit Dataset Samples ({stats?.total_examples || 0} total)
                </DialogTitle>
                <DialogContent>
                    {samplesLoading ? (
                        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                            <CircularProgress />
                        </Box>
                    ) : samples.length === 0 ? (
                        <Alert severity="info">No samples yet. Run the crawler to collect data.</Alert>
                    ) : (
                        <TableContainer sx={{ maxHeight: 400 }}>
                            <Table stickyHeader size="small">
                                <TableHead>
                                    <TableRow>
                                        <TableCell sx={{ width: '40%' }}>Instruction</TableCell>
                                        <TableCell>Output (truncated)</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {samples.map((sample, i) => (
                                        <TableRow key={i} hover>
                                            <TableCell>
                                                <Typography variant="caption" sx={{ display: 'block', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                                    {sample.instruction}
                                                </Typography>
                                            </TableCell>
                                            <TableCell>
                                                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                                    {sample.output.substring(0, 150)}...
                                                </Typography>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    )}
                </DialogContent>
                <DialogActions sx={{ p: 2 }}>
                    <Button onClick={() => setSamplesOpen(false)}>Close</Button>
                </DialogActions>
            </Dialog>
        </>
    )
}

export default RedditCrawlerCard
