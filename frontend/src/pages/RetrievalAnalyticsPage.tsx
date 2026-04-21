import React, { useState, useEffect, useCallback } from 'react'
import {
    Box,
    Typography,
    Paper,
    Chip,
    Tooltip,
    Select,
    MenuItem,
    FormControl,
} from '@mui/material'
import {
    TrendingUp as TrendingUpIcon,
    Warning as WarningIcon,
} from '@mui/icons-material'

// ── Types ─────────────────────────────────────────────────────────────

interface OverallStats {
    total_feedback: number
    positive: number
    negative: number
    neutral: number
    satisfaction_rate: number
    mean_relevance: number
    mean_chunks: number
    mean_entities: number
    mean_cross_refs: number
    mean_graph_expansion: number
}

interface DailyTrend {
    date: string
    count: number
    positive: number
    negative: number
    relevance: number
}

interface DomainStat {
    domain: string
    count: number
    satisfaction_rate: number
    relevance: number
}

interface ProblemQuery {
    query: string
    comment: string
    relevance: number
    date: string
}

interface AnalyticsData {
    period_days: number
    domain_filter: string | null
    overall: OverallStats
    daily_trends: DailyTrend[]
    domain_stats: DomainStat[]
    problem_queries: ProblemQuery[]
}

// ── Helpers ───────────────────────────────────────────────────────────

const getScoreColor = (score: number): string => {
    if (score >= 0.7) return '#10B981'
    if (score >= 0.4) return '#F59E0B'
    return '#EF4444'
}

// ── Component ─────────────────────────────────────────────────────────

export const RetrievalAnalyticsPage: React.FC = () => {
    const [data, setData] = useState<AnalyticsData | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [period, setPeriod] = useState(30)

    const fetchAnalytics = useCallback(async () => {
        setLoading(true)
        setError(null)
        try {
            const response = await fetch(`/api/v1/graphrag/analytics?days=${period}`)
            if (response.ok) {
                setData(await response.json())
            } else {
                setError('Failed to load analytics')
            }
        } catch (err) {
            setError('Connection error')
        } finally {
            setLoading(false)
        }
    }, [period])

    useEffect(() => {
        fetchAnalytics()
    }, [fetchAnalytics])

    return (
        <Box sx={{ p: 3, maxWidth: 1200, mx: 'auto' }}>
            {/* Header */}
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <TrendingUpIcon sx={{ fontSize: 28, color: 'primary.main' }} />
                    <Typography variant="h5" sx={{ fontWeight: 700 }}>
                        Retrieval Quality Dashboard
                    </Typography>
                </Box>
                <FormControl size="small" sx={{ minWidth: 120 }}>
                    <Select
                        value={period}
                        onChange={(e: any) => setPeriod(Number(e.target.value))}
                        sx={{ fontSize: '0.85rem' }}
                    >
                        <MenuItem value={7}>Last 7 days</MenuItem>
                        <MenuItem value={30}>Last 30 days</MenuItem>
                        <MenuItem value={90}>Last 90 days</MenuItem>
                    </Select>
                </FormControl>
            </Box>

            {loading && (
                <Typography sx={{ textAlign: 'center', py: 4, color: 'text.secondary' }}>
                    Loading analytics...
                </Typography>
            )}

            {error && (
                <Paper sx={{ p: 2, bgcolor: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 2 }}>
                    <Typography color="error">{error}</Typography>
                </Paper>
            )}

            {data && !loading && (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                    {/* Summary cards */}
                    <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2 }}>
                        <Paper elevation={0} sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>Total Feedback</Typography>
                            <Typography variant="h4" sx={{ fontWeight: 700 }}>{data.overall.total_feedback}</Typography>
                        </Paper>
                        <Paper elevation={0} sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>Satisfaction Rate</Typography>
                            <Typography variant="h4" sx={{ fontWeight: 700, color: getScoreColor(data.overall.satisfaction_rate) }}>
                                {(data.overall.satisfaction_rate * 100).toFixed(0)}%
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                                <Chip label={`👍 ${data.overall.positive}`} size="small" sx={{ height: 20, fontSize: '0.65rem', bgcolor: 'rgba(16,185,129,0.1)' }} />
                                <Chip label={`👎 ${data.overall.negative}`} size="small" sx={{ height: 20, fontSize: '0.65rem', bgcolor: 'rgba(239,68,68,0.1)' }} />
                            </Box>
                        </Paper>
                        <Paper elevation={0} sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>Avg Relevance</Typography>
                            <Typography variant="h4" sx={{ fontWeight: 700, color: getScoreColor(data.overall.mean_relevance) }}>
                                {(data.overall.mean_relevance * 100).toFixed(0)}%
                            </Typography>
                        </Paper>
                        <Paper elevation={0} sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>Avg Graph Expansion</Typography>
                            <Typography variant="h4" sx={{ fontWeight: 700 }}>{data.overall.mean_graph_expansion.toFixed(1)}</Typography>
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>entities discovered/query</Typography>
                        </Paper>
                    </Box>

                    {/* Daily trends */}
                    {data.daily_trends.length > 0 && (
                        <Paper elevation={0} sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
                            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1.5 }}>Daily Trends</Typography>
                            <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'flex-end', height: 120 }}>
                                {data.daily_trends.slice(0, 14).reverse().map((day: DailyTrend) => {
                                    const maxCount = Math.max(...data.daily_trends.map((d: DailyTrend) => d.count), 1)
                                    const height = (day.count / maxCount) * 100
                                    const posRatio = day.positive / Math.max(day.count, 1)
                                    return (
                                        <Tooltip key={day.date} title={`${day.date}: ${day.count} feedback (${day.positive}👍 ${day.negative}👎)`}>
                                            <Box
                                                sx={{
                                                    flex: 1,
                                                    height: `${Math.max(height, 4)}%`,
                                                    borderRadius: '4px 4px 0 0',
                                                    bgcolor: getScoreColor(posRatio),
                                                    opacity: 0.7,
                                                    cursor: 'pointer',
                                                    '&:hover': { opacity: 1 },
                                                    transition: 'opacity 0.2s',
                                                }}
                                            />
                                        </Tooltip>
                                    )
                                })}
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                                <Typography variant="caption" sx={{ fontSize: '0.6rem', color: 'text.secondary' }}>
                                    {data.daily_trends.length > 1
                                        ? data.daily_trends[Math.min(13, data.daily_trends.length - 1)]?.date || ''
                                        : data.daily_trends[0]?.date || ''}
                                </Typography>
                                <Typography variant="caption" sx={{ fontSize: '0.6rem', color: 'text.secondary' }}>
                                    {data.daily_trends[0]?.date || ''}
                                </Typography>
                            </Box>
                        </Paper>
                    )}

                    {/* Domain stats */}
                    {data.domain_stats.length > 0 && (
                        <Paper elevation={0} sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
                            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1.5 }}>Quality by Domain</Typography>
                            {data.domain_stats.map((ds: DomainStat) => (
                                <Box key={ds.domain} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', py: 0.5, borderBottom: '1px solid', borderColor: 'divider' }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <Typography variant="body2" sx={{ fontWeight: 500 }}>{ds.domain}</Typography>
                                        <Chip label={`${ds.count} queries`} size="small" sx={{ height: 18, fontSize: '0.6rem' }} />
                                    </Box>
                                    <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                                        <Typography variant="caption" sx={{ color: getScoreColor(ds.satisfaction_rate) }}>
                                            {(ds.satisfaction_rate * 100).toFixed(0)}% satisfaction
                                        </Typography>
                                        <Typography variant="caption" sx={{ color: getScoreColor(ds.relevance) }}>
                                            {(ds.relevance * 100).toFixed(0)}% relevance
                                        </Typography>
                                    </Box>
                                </Box>
                            ))}
                        </Paper>
                    )}

                    {/* Problem queries */}
                    {data.problem_queries.length > 0 && (
                        <Paper elevation={0} sx={{ p: 2, border: '1px solid', borderColor: 'rgba(239,68,68,0.3)', borderRadius: 2, bgcolor: 'rgba(239,68,68,0.03)' }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                                <WarningIcon sx={{ fontSize: 18, color: '#EF4444' }} />
                                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>Poorly Rated Queries</Typography>
                            </Box>
                            {data.problem_queries.map((pq: ProblemQuery, idx: number) => (
                                <Box key={idx} sx={{ py: 0.5, borderBottom: '1px solid', borderColor: 'divider' }}>
                                    <Typography variant="body2" sx={{ fontWeight: 500, mb: 0.3 }}>{pq.query}</Typography>
                                    <Box sx={{ display: 'flex', gap: 1 }}>
                                        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                            Relevance: {(pq.relevance * 100).toFixed(0)}%
                                        </Typography>
                                        {pq.comment && (
                                            <Typography variant="caption" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>
                                                — {pq.comment}
                                            </Typography>
                                        )}
                                    </Box>
                                </Box>
                            ))}
                        </Paper>
                    )}

                    {/* Empty state */}
                    {data.overall.total_feedback === 0 && (
                        <Paper elevation={0} sx={{ p: 4, textAlign: 'center', border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
                            <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>No Feedback Yet</Typography>
                            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                                Use the 👍/👎 buttons on retrieved context in chat to start collecting feedback data.
                            </Typography>
                        </Paper>
                    )}
                </Box>
            )}
        </Box>
    )
}
