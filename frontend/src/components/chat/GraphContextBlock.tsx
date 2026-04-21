import React, { useState } from 'react'
import {
    Box,
    Typography,
    Collapse,
    IconButton,
    Paper,
    Chip,
    Divider,
    CircularProgress,
    Tooltip,
} from '@mui/material'
import {
    ExpandMore as ExpandMoreIcon,
    ExpandLess as ExpandLessIcon,
    AccountTree as GraphIcon,
    ArrowForward as ArrowIcon,
} from '@mui/icons-material'

export interface GraphEntity {
    name: string
    type: string
    score: number
    source?: string  // 'graph_expansion' for discovered entities
}

export interface GraphRelationship {
    source: string
    target: string
    relation: string
    score: number
}

interface GraphContextBlockProps {
    entities: GraphEntity[]
    relationships: GraphRelationship[]
    isLoading?: boolean
    onEntityClick?: (entityName: string, entityType: string, anchorEl: HTMLElement) => void
    onAskAbout?: (entityName: string) => void
}

// Color palette for entity types
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

export const GraphContextBlock: React.FC<GraphContextBlockProps> = ({
    entities,
    relationships,
    isLoading = false,
    onEntityClick,
    onAskAbout: _onAskAbout,
}) => {
    const [expanded, setExpanded] = useState(true)

    if (entities.length === 0 && relationships.length === 0 && !isLoading) {
        return null
    }

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
                        bgcolor: 'action.hover',
                        cursor: 'pointer',
                    }}
                    onClick={() => setExpanded(!expanded)}
                >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <GraphIcon fontSize="small" sx={{ color: '#10B981' }} />
                        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                            🕸️ Knowledge Graph Context
                        </Typography>
                        {entities.length > 0 && (
                            <Chip
                                label={`${entities.length} entities`}
                                size="small"
                                variant="outlined"
                                sx={{ height: 20, fontSize: '0.7rem' }}
                            />
                        )}
                        {relationships.length > 0 && (
                            <Chip
                                label={`${relationships.length} relations`}
                                size="small"
                                variant="outlined"
                                sx={{ height: 20, fontSize: '0.7rem' }}
                            />
                        )}
                    </Box>
                    <IconButton size="small">
                        {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                    </IconButton>
                </Box>

                {/* Loading State */}
                {isLoading && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 2 }}>
                        <CircularProgress size={16} />
                        <Typography variant="body2" color="text.secondary">
                            Querying knowledge graph...
                        </Typography>
                    </Box>
                )}

                {/* Content */}
                <Collapse in={expanded && !isLoading}>
                    <Box sx={{ p: 2 }}>
                        {/* Entities as colored chips */}
                        {entities.length > 0 && (() => {
                            const extracted = entities.filter(e => e.source !== 'graph_expansion')
                            const discovered = entities.filter(e => e.source === 'graph_expansion')
                            return (
                                <Box sx={{ mb: relationships.length > 0 ? 2 : 0 }}>
                                    {extracted.length > 0 && (
                                        <>
                                            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                                                Extracted Entities
                                            </Typography>
                                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: discovered.length > 0 ? 1.5 : 0 }}>
                                                {extracted.map((entity, index) => {
                                                    const eColor = getEntityColor(entity.type)
                                                    return (
                                                        <Tooltip key={`${entity.name}-${index}`} title={onEntityClick ? 'Click to explore connections' : ''}>
                                                            <Chip
                                                                label={`${entity.name} (${entity.type})`}
                                                                size="small"
                                                                onClick={onEntityClick ? (e: any) => onEntityClick(entity.name, entity.type, e.currentTarget) : undefined}
                                                                sx={{
                                                                    bgcolor: `${eColor}20`,
                                                                    color: eColor,
                                                                    borderColor: `${eColor}40`,
                                                                    border: '1px solid',
                                                                    fontSize: '0.75rem',
                                                                    height: 24,
                                                                    cursor: onEntityClick ? 'pointer' : 'default',
                                                                    transition: 'all 0.15s',
                                                                    '&:hover': onEntityClick ? {
                                                                        bgcolor: `${eColor}35`,
                                                                        transform: 'translateY(-1px)',
                                                                        boxShadow: `0 2px 8px ${eColor}30`,
                                                                    } : {},
                                                                    '& .MuiChip-label': { px: 1 },
                                                                }}
                                                            />
                                                        </Tooltip>
                                                    )
                                                })}
                                            </Box>
                                        </>
                                    )}
                                    {discovered.length > 0 && (
                                        <>
                                            <Typography variant="caption" sx={{ mb: 0.5, display: 'block', color: '#8B5CF6', fontWeight: 600 }}>
                                                ✨ Discovered via Graph
                                            </Typography>
                                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                                                {discovered.map((entity, index) => {
                                                    const eColor = getEntityColor(entity.type)
                                                    return (
                                                        <Tooltip key={`disc-${entity.name}-${index}`} title={onEntityClick ? 'Discovered via graph expansion — click to explore' : 'Discovered via graph expansion'}>
                                                            <Chip
                                                                label={`${entity.name} (${entity.type})`}
                                                                size="small"
                                                                onClick={onEntityClick ? (e: any) => onEntityClick(entity.name, entity.type, e.currentTarget) : undefined}
                                                                sx={{
                                                                    bgcolor: `${eColor}10`,
                                                                    color: eColor,
                                                                    borderColor: `${eColor}50`,
                                                                    border: '1px dashed',
                                                                    fontSize: '0.75rem',
                                                                    height: 24,
                                                                    cursor: onEntityClick ? 'pointer' : 'default',
                                                                    transition: 'all 0.15s',
                                                                    '&:hover': onEntityClick ? {
                                                                        bgcolor: `${eColor}25`,
                                                                        transform: 'translateY(-1px)',
                                                                        boxShadow: `0 2px 8px ${eColor}20`,
                                                                        borderStyle: 'solid',
                                                                    } : {},
                                                                    '& .MuiChip-label': { px: 1 },
                                                                }}
                                                            />
                                                        </Tooltip>
                                                    )
                                                })}
                                            </Box>
                                        </>
                                    )}
                                </Box>
                            )
                        })()}

                        {/* Relationships */}
                        {relationships.length > 0 && (
                            <Box>
                                {entities.length > 0 && <Divider sx={{ mb: 1.5 }} />}
                                <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                                    Relationships
                                </Typography>
                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                                    {relationships.map((rel, index) => (
                                        <Box
                                            key={index}
                                            sx={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: 0.5,
                                                py: 0.25,
                                                fontSize: '0.8rem',
                                            }}
                                        >
                                            <Typography
                                                variant="body2"
                                                onClick={onEntityClick ? (e) => onEntityClick(rel.source, 'ENTITY', e.currentTarget as HTMLElement) : undefined}
                                                sx={{
                                                    fontWeight: 500, color: '#10B981',
                                                    cursor: onEntityClick ? 'pointer' : 'default',
                                                    '&:hover': onEntityClick ? { textDecoration: 'underline' } : {},
                                                }}
                                            >
                                                {rel.source}
                                            </Typography>
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25 }}>
                                                <ArrowIcon sx={{ fontSize: 14, color: 'text.secondary' }} />
                                                <Chip
                                                    label={rel.relation}
                                                    size="small"
                                                    sx={{
                                                        height: 18,
                                                        fontSize: '0.65rem',
                                                        bgcolor: 'action.selected',
                                                    }}
                                                />
                                                <ArrowIcon sx={{ fontSize: 14, color: 'text.secondary' }} />
                                            </Box>
                                            <Typography
                                                variant="body2"
                                                onClick={onEntityClick ? (e) => onEntityClick(rel.target, 'ENTITY', e.currentTarget as HTMLElement) : undefined}
                                                sx={{
                                                    fontWeight: 500, color: '#3B82F6',
                                                    cursor: onEntityClick ? 'pointer' : 'default',
                                                    '&:hover': onEntityClick ? { textDecoration: 'underline' } : {},
                                                }}
                                            >
                                                {rel.target}
                                            </Typography>
                                            <Typography variant="caption" color="text.secondary">
                                                ({(rel.score * 100).toFixed(0)}%)
                                            </Typography>
                                        </Box>
                                    ))}
                                </Box>
                            </Box>
                        )}
                    </Box>
                </Collapse>
            </Paper>
        </Box>
    )
}

export default GraphContextBlock
