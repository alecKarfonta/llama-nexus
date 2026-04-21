import React, { useState } from 'react'
import {
    Box,
    Typography,
    Chip,
    IconButton,
    Collapse,
    Tooltip,
    Badge,
    alpha,
} from '@mui/material'
import {
    AccountTree as GraphIcon,
    ExpandMore as ExpandMoreIcon,
    ExpandLess as ExpandLessIcon,
} from '@mui/icons-material'

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

interface ConversationEntitiesProps {
    entities: Array<{ name: string; type: string; score?: number }>
    onEntityClick: (entityName: string, entityType: string, anchorEl: HTMLElement) => void
}

export const ConversationEntities: React.FC<ConversationEntitiesProps> = ({
    entities,
    onEntityClick,
}) => {
    const [expanded, setExpanded] = useState(false)

    if (entities.length === 0) return null

    // Group by type
    const grouped: Record<string, Array<{ name: string; type: string }>> = {}
    for (const entity of entities) {
        const type = entity.type || 'ENTITY'
        if (!grouped[type]) grouped[type] = []
        grouped[type].push(entity)
    }

    return (
        <Box sx={{
            position: 'sticky',
            top: 0,
            zIndex: 10,
            bgcolor: 'background.paper',
            borderBottom: '1px solid',
            borderColor: 'divider',
        }}>
            <Box
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    px: 2,
                    py: 0.75,
                    cursor: 'pointer',
                    '&:hover': { bgcolor: 'action.hover' },
                }}
                onClick={() => setExpanded(!expanded)}
            >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <GraphIcon sx={{ fontSize: 16, color: '#10B981' }} />
                    <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                        Conversation Entities
                    </Typography>
                    <Badge
                        badgeContent={entities.length}
                        color="primary"
                        sx={{ '& .MuiBadge-badge': { fontSize: 10, height: 16, minWidth: 16 } }}
                    >
                        <Box />
                    </Badge>
                </Box>
                <IconButton size="small">
                    {expanded ? <ExpandLessIcon sx={{ fontSize: 16 }} /> : <ExpandMoreIcon sx={{ fontSize: 16 }} />}
                </IconButton>
            </Box>

            <Collapse in={expanded}>
                <Box sx={{ px: 2, pb: 1.5, maxHeight: 200, overflow: 'auto' }}>
                    {Object.entries(grouped).map(([type, typeEntities]) => (
                        <Box key={type} sx={{ mb: 1 }}>
                            <Typography variant="caption" sx={{
                                fontSize: '0.6rem',
                                textTransform: 'uppercase',
                                fontWeight: 700,
                                color: getEntityColor(type),
                                letterSpacing: '0.5px',
                                display: 'block',
                                mb: 0.25,
                            }}>
                                {type} ({typeEntities.length})
                            </Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                                {typeEntities.map((entity) => {
                                    const eColor = getEntityColor(entity.type)
                                    return (
                                        <Tooltip key={entity.name} title="Click to explore">
                                            <Chip
                                                label={entity.name}
                                                size="small"
                                                onClick={(e) => onEntityClick(entity.name, entity.type, e.currentTarget)}
                                                sx={{
                                                    height: 22,
                                                    fontSize: '0.7rem',
                                                    bgcolor: alpha(eColor, 0.1),
                                                    color: eColor,
                                                    border: '1px solid',
                                                    borderColor: alpha(eColor, 0.25),
                                                    cursor: 'pointer',
                                                    transition: 'all 0.15s',
                                                    '&:hover': {
                                                        bgcolor: alpha(eColor, 0.2),
                                                        transform: 'translateY(-1px)',
                                                    },
                                                    '& .MuiChip-label': { px: 0.75 },
                                                }}
                                            />
                                        </Tooltip>
                                    )
                                })}
                            </Box>
                        </Box>
                    ))}
                </Box>
            </Collapse>
        </Box>
    )
}

export default ConversationEntities
