import React, { useState, useEffect } from 'react'
import {
    Box,
    Typography,
    Popover,
    Paper,
    Chip,
    IconButton,
    CircularProgress,
    Divider,
    Button,
    Tooltip,
    alpha,
} from '@mui/material'
import {
    Close as CloseIcon,
    ArrowForward as ArrowIcon,
    ArrowBack as ArrowBackIcon,
    QuestionAnswer as AskIcon,
    AccountTree as GraphIcon,
    Hub as HubIcon,
} from '@mui/icons-material'

// Reuse from GraphContextBlock
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
    TECHNOLOGY: '#EC4899',
    CONCEPT: '#8B5CF6',
    PROCESS: '#F97316',
    CUSTOM: '#6B7280',
    ENTITY: '#6B7280',
    UNKNOWN: '#6B7280',
}

const getEntityColor = (type: string): string => {
    return TYPE_COLORS[type.toUpperCase()] || TYPE_COLORS.ENTITY
}

export interface NeighborEntity {
    name: string
    type: string
    occurrence?: number
    description?: string
}

export interface NeighborRelationship {
    source: string
    target: string
    relation: string
    direction: 'incoming' | 'outgoing'
    weight?: number
}

interface EntityNeighborhoodData {
    center: {
        name: string
        type: string
        description?: string
        occurrence?: number
    }
    neighbors: NeighborEntity[]
    relationships: NeighborRelationship[]
    found_in_graph: boolean
}

interface EntityExplorerProps {
    anchorEl: HTMLElement | null
    entityName: string
    entityType: string
    onClose: () => void
    onEntityClick: (entityName: string, entityType: string) => void
    onAskAbout: (entityName: string) => void
}

export const EntityExplorer: React.FC<EntityExplorerProps> = ({
    anchorEl,
    entityName,
    entityType,
    onClose,
    onEntityClick,
    onAskAbout,
}) => {
    const [data, setData] = useState<EntityNeighborhoodData | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [history, setHistory] = useState<{ name: string; type: string }[]>([])

    const open = Boolean(anchorEl)

    useEffect(() => {
        if (open && entityName) {
            fetchNeighborhood(entityName)
        }
    }, [entityName, open])

    const fetchNeighborhood = async (name: string) => {
        setLoading(true)
        setError(null)

        try {
            const response = await fetch('/api/v1/graphrag/entity-neighborhood', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ entity_name: name, limit: 15 }),
            })

            if (!response.ok) {
                throw new Error(`Failed to fetch: ${response.statusText}`)
            }

            const result = await response.json()
            setData(result)
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch neighborhood')
        } finally {
            setLoading(false)
        }
    }

    const handleNeighborClick = (neighbor: NeighborEntity) => {
        // Push current entity to history for back navigation
        setHistory(prev => [...prev, { name: entityName, type: entityType }])
        onEntityClick(neighbor.name, neighbor.type)
    }

    const handleBack = () => {
        if (history.length > 0) {
            const prev = history[history.length - 1]
            setHistory(h => h.slice(0, -1))
            onEntityClick(prev.name, prev.type)
        }
    }

    // Group relationships by type
    const groupedRels: Record<string, { neighbors: string[], direction: string }> = {}
    if (data?.relationships) {
        for (const rel of data.relationships) {
            const key = rel.relation
            if (!groupedRels[key]) {
                groupedRels[key] = { neighbors: [], direction: rel.direction }
            }
            const neighbor = rel.direction === 'outgoing' ? rel.target : rel.source
            if (!groupedRels[key].neighbors.includes(neighbor)) {
                groupedRels[key].neighbors.push(neighbor)
            }
        }
    }

    const entityColor = getEntityColor(entityType)

    return (
        <Popover
            open={open}
            anchorEl={anchorEl}
            onClose={onClose}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
            transformOrigin={{ vertical: 'top', horizontal: 'left' }}
            slotProps={{
                paper: {
                    sx: {
                        width: 380,
                        maxHeight: 480,
                        overflow: 'hidden',
                        display: 'flex',
                        flexDirection: 'column',
                        borderRadius: 2,
                        border: '1px solid',
                        borderColor: alpha(entityColor, 0.3),
                        bgcolor: 'background.paper',
                    }
                }
            }}
        >
            {/* Header */}
            <Box sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                px: 2,
                py: 1.5,
                borderBottom: '1px solid',
                borderColor: 'divider',
                bgcolor: alpha(entityColor, 0.05),
            }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1, minWidth: 0 }}>
                    {history.length > 0 && (
                        <IconButton size="small" onClick={handleBack} sx={{ mr: 0.5 }}>
                            <ArrowBackIcon fontSize="small" />
                        </IconButton>
                    )}
                    <HubIcon sx={{ color: entityColor, fontSize: 20 }} />
                    <Box sx={{ minWidth: 0 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 700, lineHeight: 1.2 }} noWrap>
                            {entityName}
                        </Typography>
                        <Chip
                            label={entityType}
                            size="small"
                            sx={{
                                height: 16,
                                fontSize: '0.6rem',
                                bgcolor: alpha(entityColor, 0.15),
                                color: entityColor,
                            }}
                        />
                    </Box>
                </Box>
                <IconButton size="small" onClick={onClose}>
                    <CloseIcon fontSize="small" />
                </IconButton>
            </Box>

            {/* Content */}
            <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
                {loading && (
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', py: 4, gap: 1 }}>
                        <CircularProgress size={20} />
                        <Typography variant="body2" color="text.secondary">
                            Exploring neighborhood...
                        </Typography>
                    </Box>
                )}

                {error && (
                    <Typography variant="body2" color="error" sx={{ py: 2 }}>
                        {error}
                    </Typography>
                )}

                {!loading && data && (
                    <>
                        {/* Description */}
                        {data.center.description && (
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 2, fontSize: '0.8rem' }}>
                                {data.center.description}
                            </Typography>
                        )}

                        {/* Not found in graph */}
                        {!data.found_in_graph && (
                            <Box sx={{ textAlign: 'center', py: 2 }}>
                                <GraphIcon sx={{ fontSize: 32, color: 'text.disabled', mb: 1 }} />
                                <Typography variant="body2" color="text.secondary">
                                    Not found in knowledge graph
                                </Typography>
                                <Typography variant="caption" color="text.disabled">
                                    This entity hasn't been indexed yet
                                </Typography>
                            </Box>
                        )}

                        {/* Neighbors grouped by relationship */}
                        {data.found_in_graph && data.neighbors.length === 0 && (
                            <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                                No connections found
                            </Typography>
                        )}

                        {Object.entries(groupedRels).map(([relType, group]) => (
                            <Box key={relType} sx={{ mb: 1.5 }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                                    {group.direction === 'outgoing' ? (
                                        <ArrowIcon sx={{ fontSize: 12, color: 'text.disabled' }} />
                                    ) : (
                                        <ArrowBackIcon sx={{ fontSize: 12, color: 'text.disabled' }} />
                                    )}
                                    <Typography variant="caption" sx={{
                                        textTransform: 'uppercase',
                                        fontWeight: 600,
                                        color: 'text.secondary',
                                        fontSize: '0.65rem',
                                        letterSpacing: '0.5px',
                                    }}>
                                        {relType.replace(/_/g, ' ')}
                                    </Typography>
                                </Box>
                                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, pl: 2 }}>
                                    {group.neighbors.map(neighborName => {
                                        const neighborData = data.neighbors.find(
                                            n => n.name.toLowerCase() === neighborName.toLowerCase()
                                        )
                                        const nType = neighborData?.type || 'concept'
                                        const nColor = getEntityColor(nType)
                                        return (
                                            <Tooltip
                                                key={neighborName}
                                                title={neighborData?.description || `${neighborName} (${nType})`}
                                            >
                                                <Chip
                                                    label={neighborName}
                                                    size="small"
                                                    onClick={() => handleNeighborClick({
                                                        name: neighborName,
                                                        type: nType,
                                                    })}
                                                    sx={{
                                                        height: 24,
                                                        fontSize: '0.75rem',
                                                        bgcolor: alpha(nColor, 0.1),
                                                        color: nColor,
                                                        border: '1px solid',
                                                        borderColor: alpha(nColor, 0.3),
                                                        cursor: 'pointer',
                                                        transition: 'all 0.15s',
                                                        '&:hover': {
                                                            bgcolor: alpha(nColor, 0.2),
                                                            transform: 'translateY(-1px)',
                                                        },
                                                        '& .MuiChip-label': { px: 1 },
                                                    }}
                                                />
                                            </Tooltip>
                                        )
                                    })}
                                </Box>
                            </Box>
                        ))}

                        {/* Stats */}
                        {data.found_in_graph && data.neighbors.length > 0 && (
                            <>
                                <Divider sx={{ my: 1.5 }} />
                                <Typography variant="caption" color="text.disabled" sx={{ display: 'block', textAlign: 'center' }}>
                                    {data.neighbors.length} connected entities · {data.relationships.length} relationships
                                </Typography>
                            </>
                        )}
                    </>
                )}
            </Box>

            {/* Action Bar */}
            <Box sx={{
                display: 'flex',
                gap: 1,
                p: 1.5,
                borderTop: '1px solid',
                borderColor: 'divider',
                bgcolor: 'action.hover',
            }}>
                <Button
                    size="small"
                    variant="contained"
                    startIcon={<AskIcon />}
                    onClick={() => {
                        onAskAbout(entityName)
                        onClose()
                    }}
                    sx={{
                        flex: 1,
                        textTransform: 'none',
                        fontSize: '0.75rem',
                        bgcolor: entityColor,
                        '&:hover': { bgcolor: entityColor, filter: 'brightness(0.9)' },
                    }}
                >
                    Ask about {entityName.length > 15 ? entityName.slice(0, 15) + '…' : entityName}
                </Button>
            </Box>
        </Popover>
    )
}

export default EntityExplorer
