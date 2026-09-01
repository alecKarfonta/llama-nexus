import { useCallback, useEffect, useState } from 'react'
import {
  Alert,
  Box,
  CircularProgress,
  FormControl,
  Grid,
  IconButton,
  InputLabel,
  LinearProgress,
  MenuItem,
  Select,
  Tooltip,
  Typography,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import {
  Domain as DomainIcon,
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  Warning as WarningIcon,
} from '@mui/icons-material'
import { PageShell, SectionCard, StatusChip, type StatusChipConfigMap } from '@/components/common'
import { accentColors } from '@/utils/theme'

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

const emptyOverall: OverallStats = {
  total_feedback: 0,
  positive: 0,
  negative: 0,
  neutral: 0,
  satisfaction_rate: 0,
  mean_relevance: 0,
  mean_chunks: 0,
  mean_entities: 0,
  mean_cross_refs: 0,
  mean_graph_expansion: 0,
}

const getScoreColor = (score: number): string => {
  if (score >= 0.7) return accentColors.success
  if (score >= 0.4) return accentColors.warning
  return accentColors.rose
}

const formatNumber = (value: number): string => {
  if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`
  if (value >= 1000) return `${(value / 1000).toFixed(1)}K`
  return value.toString()
}

const formatPercent = (value: number): string => `${Math.round(value * 100)}%`

const RETRIEVAL_ANALYTICS_STATUS_CONFIG: StatusChipConfigMap = {
  problem: { color: accentColors.rose, label: 'Needs review' },
  quality: { color: accentColors.success, label: 'Retrieval quality' },
  satisfaction: { color: accentColors.success, label: 'Satisfaction' },
  tools: { color: accentColors.info, label: 'Analytics tools' },
}

const retrievalAnalyticsPanelSx = {
  borderRadius: '8px',
  bgcolor: 'background.paper',
  border: '1px solid',
  borderColor: 'divider',
} as const

export function RetrievalAnalyticsPage() {
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
        setError('Failed to load retrieval analytics')
      }
    } catch {
      setError('Connection error')
    } finally {
      setLoading(false)
    }
  }, [period])

  useEffect(() => {
    fetchAnalytics()
  }, [fetchAnalytics])

  const overall = data?.overall || emptyOverall
  const trendWindow = data?.daily_trends.slice(0, 14).reverse() || []
  const maxTrendCount = Math.max(...(data?.daily_trends.map((day) => day.count) || [0]), 1)
  const topDomain = data?.domain_stats.reduce<DomainStat | null>((best, domain) => {
    if (!best) return domain
    return domain.count > best.count ? domain : best
  }, null)
  const problemCount = data?.problem_queries.length || 0
  const hasAnalyticsData = trendWindow.length > 0 || (data?.domain_stats.length || 0) > 0 || problemCount > 0
  const hasFeedback = overall.total_feedback > 0
  const retrievalHeaderActions = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
      <FormControl size="small" sx={{ width: { xs: '100%', sm: 150 } }}>
        <InputLabel id="retrieval-period-label">Period</InputLabel>
        <Select
          labelId="retrieval-period-label"
          label="Period"
          value={period}
          onChange={(event) => setPeriod(Number(event.target.value))}
          sx={{ height: 34, '& .MuiSelect-select': { py: 0.75 } }}
        >
          <MenuItem value={7}>Last 7 days</MenuItem>
          <MenuItem value={30}>Last 30 days</MenuItem>
          <MenuItem value={90}>Last 90 days</MenuItem>
        </Select>
      </FormControl>
      <Tooltip title="Refresh analytics">
        <IconButton
          aria-label="Refresh analytics"
          size="small"
          onClick={fetchAnalytics}
          disabled={loading}
          sx={{
            width: 34,
            height: 34,
            borderRadius: '8px',
            border: '1px solid',
            borderColor: 'divider',
            bgcolor: 'action.hover',
            '&:hover': { bgcolor: alpha(accentColors.success, 0.12) },
          }}
        >
          {loading ? <CircularProgress size={16} /> : <RefreshIcon sx={{ fontSize: 18 }} />}
        </IconButton>
      </Tooltip>
    </Box>
  )
  const retrievalSummaryRow = hasFeedback || topDomain || problemCount > 0 ? (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap' }}>
      {hasFeedback && (
        <>
          <StatusChip
            status="satisfaction"
            label={`${formatPercent(overall.satisfaction_rate)} satisfaction`}
            statusConfig={RETRIEVAL_ANALYTICS_STATUS_CONFIG}
            height={22}
          />
          <StatusChip
            status="quality"
            label={`${formatPercent(overall.mean_relevance)} relevance`}
            statusConfig={RETRIEVAL_ANALYTICS_STATUS_CONFIG}
            height={22}
          />
        </>
      )}
      {problemCount > 0 && (
        <StatusChip
          status="problem"
          label={`${formatNumber(problemCount)} reviews`}
          statusConfig={RETRIEVAL_ANALYTICS_STATUS_CONFIG}
          height={22}
        />
      )}
      {(hasFeedback || topDomain) && (
        <Typography variant="caption" color="text.secondary" noWrap sx={{ minWidth: 0 }}>
          {formatNumber(overall.total_feedback)} feedback / {data?.domain_stats.length || 0} domains
          {topDomain ? ` / ${topDomain.domain} leads` : ''}
        </Typography>
      )}
    </Box>
  ) : null
  const retrievalHeaderContent = (() => {
    if (loading && !data) {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25, minHeight: 24 }}>
          <CircularProgress size={18} />
          <Typography variant="body2" color="text.secondary">
            Loading retrieval analytics...
          </Typography>
        </Box>
      )
    }

    if (!hasAnalyticsData) {
      return (
        <Typography variant="body2" color="text.secondary">
          No retrieval feedback yet. Rated chat answers will populate quality and low-rated query review.
        </Typography>
      )
    }

    return retrievalSummaryRow
  })()

  return (
    <PageShell
      sx={{
        display: 'flex',
        flexDirection: 'column',
        gap: 1.25,
        py: 1.25,
      }}
    >
      <SectionCard
        title="Retrieval Quality"
        subtitle={hasFeedback ? `${formatNumber(overall.total_feedback)} rated answers` : 'No rated answers in selected period'}
        icon={<TrendingUpIcon />}
        accentColor={accentColors.success}
        action={retrievalHeaderActions}
        sx={!hasAnalyticsData ? { width: 'min(100%, 760px)' } : undefined}
      >
        {retrievalHeaderContent}
      </SectionCard>

      <Box
        data-retrieval-analytics-workspace
        sx={{
          display: 'flex',
          flexDirection: 'column',
          gap: 1.25,
          overflow: 'visible',
        }}
      >
        {error && (
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {data && hasAnalyticsData ? (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25 }}>
          {trendWindow.length > 0 && (
            <SectionCard
              title="Daily Trends"
              subtitle={`${trendWindow.length} days of feedback activity`}
              icon={<TrendingUpIcon />}
              accentColor={accentColors.info}
            >
              <Box sx={{ display: 'flex', gap: 0.75, alignItems: 'flex-end', height: 132 }}>
                {trendWindow.map((day) => {
                  const height = (day.count / maxTrendCount) * 100
                  const positiveRatio = day.positive / Math.max(day.count, 1)
                  return (
                    <Tooltip key={day.date} title={`${day.date}: ${day.count} feedback, ${formatPercent(day.relevance)} relevance`}>
                      <Box
                        sx={{
                          flex: 1,
                          minWidth: 12,
                          height: `${Math.max(height, 4)}%`,
                          borderRadius: '4px 4px 0 0',
                          bgcolor: getScoreColor(positiveRatio),
                          opacity: 0.78,
                          cursor: 'pointer',
                          transition: 'opacity 0.2s ease, transform 0.2s ease',
                          '&:hover': { opacity: 1, transform: 'translateY(-2px)' },
                        }}
                      />
                    </Tooltip>
                  )
                })}
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.75 }}>
                <Typography variant="caption" color="text.secondary">
                  {trendWindow[0]?.date || ''}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {trendWindow[trendWindow.length - 1]?.date || ''}
                </Typography>
              </Box>
            </SectionCard>
          )}

          <Grid container spacing={1.25}>
            <Grid item xs={12} lg={7}>
              <SectionCard
                title="Quality by Domain"
                subtitle={`${formatNumber(data.domain_stats.length)} tracked domains`}
                icon={<DomainIcon />}
                accentColor={accentColors.success}
                fullHeight
              >
                {data.domain_stats.length === 0 ? (
                  <Typography variant="body2" color="text.secondary">
                    No domain quality data yet.
                  </Typography>
                ) : (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25 }}>
                    {data.domain_stats.map((domain) => {
                      const satisfactionColor = getScoreColor(domain.satisfaction_rate)

                      return (
                        <Box
                          key={domain.domain}
                          sx={{
                            ...retrievalAnalyticsPanelSx,
                            p: 1.25,
                            bgcolor: alpha(satisfactionColor, 0.03),
                          }}
                        >
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, mb: 0.75 }}>
                            <Box sx={{ minWidth: 0 }}>
                              <Typography variant="body2" sx={{ fontWeight: 700 }} noWrap>
                                {domain.domain}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {formatNumber(domain.count)} queries
                              </Typography>
                            </Box>
                            <StatusChip
                              status="satisfaction"
                              label={`${formatPercent(domain.satisfaction_rate)} satisfaction`}
                              statusConfig={RETRIEVAL_ANALYTICS_STATUS_CONFIG}
                              sx={{
                                bgcolor: alpha(satisfactionColor, 0.12),
                                border: `1px solid ${alpha(satisfactionColor, 0.3)}`,
                                color: satisfactionColor,
                                flexShrink: 0,
                              }}
                            />
                          </Box>
                          <LinearProgress
                            variant="determinate"
                            value={domain.relevance * 100}
                            sx={{
                              height: 6,
                              borderRadius: '3px',
                              bgcolor: 'action.hover',
                              '& .MuiLinearProgress-bar': {
                                borderRadius: '3px',
                                bgcolor: getScoreColor(domain.relevance),
                              },
                            }}
                          />
                        </Box>
                      )
                    })}
                  </Box>
                )}
              </SectionCard>
            </Grid>

            <Grid item xs={12} lg={5}>
              <SectionCard
                title="Poorly Rated Queries"
                subtitle={problemCount > 0 ? 'Queries needing review' : 'No low-rated queries'}
                icon={<WarningIcon />}
                accentColor={problemCount > 0 ? accentColors.rose : accentColors.success}
                fullHeight
                action={
                  <StatusChip
                    status={problemCount > 0 ? 'problem' : 'satisfaction'}
                    label={formatNumber(problemCount)}
                    statusConfig={RETRIEVAL_ANALYTICS_STATUS_CONFIG}
                  />
                }
              >
                {problemCount === 0 ? (
                  <Typography variant="body2" color="text.secondary">
                    No low-rated queries in this period.
                  </Typography>
                ) : (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    {data.problem_queries.map((query, index) => (
                      <Box
                        key={`${query.date}-${index}`}
                        sx={{
                          ...retrievalAnalyticsPanelSx,
                          p: 1.25,
                          bgcolor: alpha(accentColors.rose, 0.04),
                        }}
                      >
                        <Typography variant="body2" sx={{ fontWeight: 700, mb: 0.5 }}>
                          {query.query}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Relevance {formatPercent(query.relevance)}
                          {query.comment ? ` - ${query.comment}` : ''}
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                )}
              </SectionCard>
            </Grid>
          </Grid>
        </Box>
      ) : null}
      </Box>
    </PageShell>
  )
}

export default RetrievalAnalyticsPage
