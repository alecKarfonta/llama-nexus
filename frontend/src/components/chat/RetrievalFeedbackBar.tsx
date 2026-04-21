import React, { useState } from 'react'
import {
    Box,
    IconButton,
    Tooltip,
    Typography,
    Chip,
} from '@mui/material'
import {
    ThumbUp as ThumbUpIcon,
    ThumbDown as ThumbDownIcon,
} from '@mui/icons-material'

interface QualitySignals {
    avg_chunk_relevance: number
    entity_coverage: number
    graph_expansion_count: number
    cross_reference_count: number
    total_chunks: number
    total_entities: number
}

interface RetrievalFeedbackBarProps {
    query: string
    qualitySignals?: QualitySignals | null
    domain?: string
    onFeedbackSubmitted?: (rating: number) => void
}

export const RetrievalFeedbackBar: React.FC<RetrievalFeedbackBarProps> = ({
    query,
    qualitySignals,
    domain = 'general',
    onFeedbackSubmitted,
}) => {
    const [submitted, setSubmitted] = useState<number | null>(null)
    const [submitting, setSubmitting] = useState(false)

    const submitFeedback = async (rating: number) => {
        if (submitting || submitted !== null) return
        setSubmitting(true)

        try {
            await fetch('/api/v1/graphrag/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    rating,
                    quality_signals: qualitySignals || {},
                    domain,
                }),
            })
            setSubmitted(rating)
            onFeedbackSubmitted?.(rating)
        } catch (error) {
            console.warn('Failed to submit feedback:', error)
        } finally {
            setSubmitting(false)
        }
    }

    if (!query) return null

    return (
        <Box
            sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                py: 0.5,
                px: 1,
                borderRadius: 1,
                bgcolor: submitted !== null
                    ? submitted === 1
                        ? 'rgba(16,185,129,0.08)'
                        : 'rgba(239,68,68,0.08)'
                    : 'transparent',
            }}
        >
            <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>
                {submitted !== null
                    ? submitted === 1 ? '✅ Thanks!' : '📝 Noted'
                    : 'Was this context helpful?'}
            </Typography>

            {submitted === null && (
                <>
                    <Tooltip title="Good context">
                        <IconButton
                            size="small"
                            onClick={() => submitFeedback(1)}
                            disabled={submitting}
                            sx={{ color: 'text.secondary', '&:hover': { color: '#10B981' } }}
                        >
                            <ThumbUpIcon sx={{ fontSize: 16 }} />
                        </IconButton>
                    </Tooltip>
                    <Tooltip title="Poor context">
                        <IconButton
                            size="small"
                            onClick={() => submitFeedback(-1)}
                            disabled={submitting}
                            sx={{ color: 'text.secondary', '&:hover': { color: '#EF4444' } }}
                        >
                            <ThumbDownIcon sx={{ fontSize: 16 }} />
                        </IconButton>
                    </Tooltip>
                </>
            )}

            {qualitySignals && (
                <Chip
                    label={`${(qualitySignals.avg_chunk_relevance * 100).toFixed(0)}% rel`}
                    size="small"
                    sx={{ height: 18, fontSize: '0.6rem', ml: 'auto' }}
                    variant="outlined"
                />
            )}
        </Box>
    )
}
