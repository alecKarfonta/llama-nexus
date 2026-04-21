import React, { useState } from 'react'
import {
    Box,
    Typography,
    Collapse,
    IconButton,
    Paper,
    Chip,
    Divider,
    Tooltip,
} from '@mui/material'
import {
    ExpandMore as ExpandMoreIcon,
    ExpandLess as ExpandLessIcon,
    Hub as HubIcon,
    ArrowForward as ArrowIcon,
    Search as SearchIcon,
    AccountTree as LinkIcon,
} from '@mui/icons-material'

// ── Types ─────────────────────────────────────────────────────────────

export interface UnifiedChunk {
    id: string
    content: string
    score: number
    boosted_score?: number
    entity_boost?: number
    matched_entities?: string[]
    document_id?: string
    document_title?: string
    chunk_index?: number
    metadata?: Record<string, unknown>
}

export interface UnifiedEntity {
    name: string
    type: string
    score: number
    source?: string  // 'graph_expansion' for discovered entities
}

export interface UnifiedRelationship {
    source: string
    target: string
    relation: string
    score: number
}

export interface CrossReference {
    chunk_index: number
    chunk_id: string
    matched_entities: string[]
    overlap_score: number
}

export interface QualitySignals {
    avg_chunk_relevance: number
    entity_coverage: number
    graph_expansion_count: number
    cross_reference_count: number
    total_chunks: number
    total_entities: number
}

interface UnifiedContextBlockProps {
    chunks: UnifiedChunk[]
    entities: UnifiedEntity[]
    relationships: UnifiedRelationship[]
    graphConnections?: string[]
    crossReferences?: CrossReference[]
    qualitySignals?: QualitySignals
    isLoading?: boolean
    onEntityClick?: (entityName: string, entityType: string, anchorEl: HTMLElement) => void
    onAskAbout?: (entityName: string) => void
}

// ── Color palette for entity types ────────────────────────────────────

const TYPE_COLORS: Record<string, string> = {
    PER: '#10B981',
    PERSON: '#10B981',
    ORG: '#3B82F6',
    ORGANIZATION: '#3B82F6',
    LOC: '#F59E0B',
    LOCATION: '#F59E0B',
    MISC: '#8B5CF6',
    MONEY: '#EC4899',
    DATE: '#6366F1',
    TIME: '#06B6D4',
    ENTITY: '#6B7280',
    UNKNOWN: '#6B7280',
}

const getEntityColor = (type: string): string => {
    return TYPE_COLORS[type.toUpperCase()] || TYPE_COLORS.ENTITY
}

// ── Relevance color ───────────────────────────────────────────────────

const getRelevanceColor = (score: number): string => {
    if (score >= 0.7) return '#10B981'
    if (score >= 0.4) return '#F59E0B'
    return '#EF4444'
}

// ── Component ─────────────────────────────────────────────────────────

export const UnifiedContextBlock: React.FC<UnifiedContextBlockProps> = ({
    chunks,
    entities,
    relationships,
    graphConnections = [],
    crossReferences = [],
    qualitySignals,
    isLoading = false,
    onEntityClick,
    onAskAbout: _onAskAbout,
}) => {
    const [expanded, setExpanded] = useState(true)
    const [showChunks, setShowChunks] = useState(true)

    const hasContent = chunks.length > 0 || entities.length > 0 || relationships.length > 0
    if (!hasContent && !isLoading) return null

    const extractedEntities = entities.filter(e => e.source !== 'graph_expansion')
    const discoveredEntities = entities.filter(e => e.source === 'graph_expansion')

    return (
        <Box sx={{ mb: 2 }}>
            <Paper
                elevation={0}
                sx={{
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: 2,
                    overflow: 'hidden',
                    bgcolor: 'background.paper',
                }}
            >
                {/* Header */}
                <Box
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        px: 2,
                        py: 1,
                        background: 'linear-gradient(135deg, rgba(99,102,241,0.08), rgba(16,185,129,0.08))',
                        cursor: 'pointer',
                    }}
                    onClick={() => setExpanded(!expanded)}
                >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <HubIcon sx={{ fontSize: 18, color: 'primary.main' }} />
                        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                            Unified Knowledge Context
                        </Typography>
                        {chunks.length > 0 && (
                            <Chip
                                label={`${chunks.length} chunks`}
                                size="small"
                                sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'action.selected' }}
                            />
                        )}
                        {entities.length > 0 && (
                            <Chip
                                label={`${entities.length} entities`}
                                size="small"
                                sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'action.selected' }}
                            />
                        )}
                        {crossReferences.length > 0 && (
                            <Chip
                                icon={<LinkIcon sx={{ fontSize: 12 }} />}
                                label={`${crossReferences.length} cross-refs`}
                                size="small"
                                sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'rgba(99,102,241,0.15)' }}
                            />
                        )}
                    </Box>
                    <IconButton size="small">
                        {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                    </IconButton>
                </Box>

                {isLoading && (
                    <Box sx={{ height: 2, width: '100%', bgcolor: 'action.hover', overflow: 'hidden', position: 'relative' }}>
                        <Box sx={{
                            height: '100%',
                            width: '30%',
                            bgcolor: 'primary.main',
                            position: 'absolute',
                            '@keyframes slideProgress': {
                                '0%': { left: '-30%' },
                                '100%': { left: '100%' },
                            },
                            animation: 'slideProgress 1.5s ease-in-out infinite',
                        }} />
                    </Box>
                )}

                <Collapse in={expanded}>
                    <Box sx={{ p: 2 }}>
                        {/* Quality signals bar */}
                        {qualitySignals && (
                            <Box sx={{ display: 'flex', gap: 2, mb: 2, flexWrap: 'wrap' }}>
                                <Tooltip title="Average relevance score of retrieved chunks">
                                    <Chip
                                        label={`Relevance: ${(qualitySignals.avg_chunk_relevance * 100).toFixed(0)}%`}
                                        size="small"
                                        sx={{
                                            height: 22,
                                            fontSize: '0.7rem',
                                            borderColor: getRelevanceColor(qualitySignals.avg_chunk_relevance),
                                            color: getRelevanceColor(qualitySignals.avg_chunk_relevance),
                                        }}
                                        variant="outlined"
                                    />
                                </Tooltip>
                                {qualitySignals.graph_expansion_count > 0 && (
                                    <Tooltip title="New entities discovered via knowledge graph expansion">
                                        <Chip
                                            label={`✨ ${qualitySignals.graph_expansion_count} discovered`}
                                            size="small"
                                            sx={{ height: 22, fontSize: '0.7rem', bgcolor: 'rgba(99,102,241,0.12)' }}
                                        />
                                    </Tooltip>
                                )}
                                {qualitySignals.cross_reference_count > 0 && (
                                    <Tooltip title="Number of chunks that mention extracted entities">
                                        <Chip
                                            label={`🔗 ${qualitySignals.cross_reference_count} linked`}
                                            size="small"
                                            sx={{ height: 22, fontSize: '0.7rem', bgcolor: 'rgba(16,185,129,0.12)' }}
                                        />
                                    </Tooltip>
                                )}
                            </Box>
                        )}

                        {/* Entities section */}
                        {extractedEntities.length > 0 && (
                            <Box sx={{ mb: 2 }}>
                                <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', mb: 0.5, display: 'block' }}>
                                    Extracted Entities
                                </Typography>
                                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                                    {extractedEntities.map((entity, idx) => (
                                        <Chip
                                            key={`ext-${idx}`}
                                            label={entity.name}
                                            size="small"
                                            onClick={(e: any) => onEntityClick?.(entity.name, entity.type, e.currentTarget)}
                                            sx={{
                                                height: 24,
                                                fontSize: '0.75rem',
                                                borderColor: getEntityColor(entity.type),
                                                color: getEntityColor(entity.type),
                                                cursor: onEntityClick ? 'pointer' : 'default',
                                                '&:hover': onEntityClick ? { bgcolor: `${getEntityColor(entity.type)}15` } : {},
                                            }}
                                            variant="outlined"
                                        />
                                    ))}
                                </Box>
                            </Box>
                        )}

                        {discoveredEntities.length > 0 && (
                            <Box sx={{ mb: 2 }}>
                                <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', mb: 0.5, display: 'block' }}>
                                    ✨ Discovered via Graph
                                </Typography>
                                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                                    {discoveredEntities.map((entity, idx) => (
                                        <Chip
                                            key={`disc-${idx}`}
                                            label={entity.name}
                                            size="small"
                                            onClick={(e: any) => onEntityClick?.(entity.name, entity.type, e.currentTarget)}
                                            sx={{
                                                height: 24,
                                                fontSize: '0.75rem',
                                                borderColor: getEntityColor(entity.type),
                                                color: getEntityColor(entity.type),
                                                borderStyle: 'dashed',
                                                cursor: onEntityClick ? 'pointer' : 'default',
                                                '&:hover': onEntityClick ? { bgcolor: `${getEntityColor(entity.type)}15` } : {},
                                            }}
                                            variant="outlined"
                                        />
                                    ))}
                                </Box>
                            </Box>
                        )}

                        {/* Graph connections */}
                        {graphConnections.length > 0 && (
                            <Box sx={{ mb: 2 }}>
                                <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', mb: 0.5, display: 'block' }}>
                                    Knowledge Graph Connections
                                </Typography>
                                {graphConnections.slice(0, 8).map((conn, idx) => (
                                    <Typography key={idx} variant="caption" sx={{ display: 'block', color: 'text.secondary', pl: 1, fontSize: '0.7rem' }}>
                                        • {conn}
                                    </Typography>
                                ))}
                            </Box>
                        )}

                        {/* Relationships */}
                        {relationships.length > 0 && (
                            <Box sx={{ mb: 2 }}>
                                <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', mb: 0.5, display: 'block' }}>
                                    Extracted Relationships
                                </Typography>
                                {relationships.slice(0, 6).map((rel, idx) => (
                                    <Box key={idx} sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.3 }}>
                                        <Chip label={rel.source} size="small" sx={{ height: 20, fontSize: '0.65rem' }} />
                                        <ArrowIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
                                        <Typography variant="caption" sx={{ color: 'primary.main', fontStyle: 'italic', fontSize: '0.65rem' }}>
                                            {rel.relation}
                                        </Typography>
                                        <ArrowIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
                                        <Chip label={rel.target} size="small" sx={{ height: 20, fontSize: '0.65rem' }} />
                                    </Box>
                                ))}
                            </Box>
                        )}

                        <Divider sx={{ my: 1 }} />

                        {/* Retrieved chunks section */}
                        <Box
                            sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer', mb: 1 }}
                            onClick={() => setShowChunks(!showChunks)}
                        >
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <SearchIcon sx={{ fontSize: 14, color: 'text.secondary' }} />
                                <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                                    Retrieved Passages ({chunks.length})
                                </Typography>
                            </Box>
                            <IconButton size="small">
                                {showChunks ? <ExpandLessIcon sx={{ fontSize: 16 }} /> : <ExpandMoreIcon sx={{ fontSize: 16 }} />}
                            </IconButton>
                        </Box>

                        <Collapse in={showChunks}>
                            {chunks.slice(0, 5).map((chunk, idx) => (
                                <Paper
                                    key={chunk.id || idx}
                                    elevation={0}
                                    sx={{
                                        p: 1.5,
                                        mb: 1,
                                        border: '1px solid',
                                        borderColor: chunk.matched_entities?.length ? 'rgba(99,102,241,0.3)' : 'divider',
                                        borderRadius: 1,
                                        bgcolor: chunk.matched_entities?.length ? 'rgba(99,102,241,0.03)' : 'transparent',
                                    }}
                                >
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                        <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.65rem' }}>
                                            {chunk.document_title || `Chunk ${idx + 1}`}
                                        </Typography>
                                        <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center' }}>
                                            <Tooltip title="Relevance score">
                                                <Typography
                                                    variant="caption"
                                                    sx={{ fontSize: '0.65rem', color: getRelevanceColor(chunk.score) }}
                                                >
                                                    {(chunk.score * 100).toFixed(0)}%
                                                </Typography>
                                            </Tooltip>
                                            {chunk.entity_boost !== undefined && chunk.entity_boost > 0 && (
                                                <Tooltip title={`Entity boost: +${(chunk.entity_boost * 100).toFixed(0)}%`}>
                                                    <Typography variant="caption" sx={{ fontSize: '0.6rem', color: '#6366F1' }}>
                                                        +🔗
                                                    </Typography>
                                                </Tooltip>
                                            )}
                                        </Box>
                                    </Box>
                                    <Typography variant="body2" sx={{ fontSize: '0.8rem', lineHeight: 1.5, color: 'text.primary' }}>
                                        {chunk.content.length > 300 ? chunk.content.slice(0, 300) + '...' : chunk.content}
                                    </Typography>
                                    {chunk.matched_entities && chunk.matched_entities.length > 0 && (
                                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.3, mt: 0.5 }}>
                                            {chunk.matched_entities.map((ent, i) => (
                                                <Chip
                                                    key={i}
                                                    label={ent}
                                                    size="small"
                                                    sx={{ height: 18, fontSize: '0.6rem', bgcolor: 'rgba(99,102,241,0.1)' }}
                                                />
                                            ))}
                                        </Box>
                                    )}
                                </Paper>
                            ))}
                        </Collapse>
                    </Box>
                </Collapse>
            </Paper>
        </Box>
    )
}
