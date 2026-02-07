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
    CircularProgress,
    Button,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
} from '@mui/material'
import {
    ExpandMore as ExpandMoreIcon,
    ExpandLess as ExpandLessIcon,
    Article as ArticleIcon,
    NavigateBefore as PrevIcon,
    NavigateNext as NextIcon,
    OpenInNew as OpenIcon,
    ContentCopy as CopyIcon,
} from '@mui/icons-material'

export interface RAGChunk {
    id: string
    content: string
    score: number
    document_id: string
    document_title?: string
    chunk_index: number
    metadata?: Record<string, any>
}

interface RAGContextBlockProps {
    chunks: RAGChunk[]
    isLoading?: boolean
    onFetchNeighbors?: (documentId: string, chunkIndex: number, direction: 'before' | 'after') => Promise<RAGChunk[]>
}

export const RAGContextBlock: React.FC<RAGContextBlockProps> = ({
    chunks,
    isLoading = false,
    onFetchNeighbors,
}) => {
    const [expanded, setExpanded] = useState(true)
    const [selectedChunk, setSelectedChunk] = useState<RAGChunk | null>(null)
    const [neighborChunks, setNeighborChunks] = useState<{ before: RAGChunk[], after: RAGChunk[] }>({ before: [], after: [] })
    const [loadingNeighbors, setLoadingNeighbors] = useState(false)

    const handleOpenChunkExplorer = async (chunk: RAGChunk) => {
        setSelectedChunk(chunk)
        setNeighborChunks({ before: [], after: [] })

        if (onFetchNeighbors) {
            setLoadingNeighbors(true)
            try {
                const [before, after] = await Promise.all([
                    onFetchNeighbors(chunk.document_id, chunk.chunk_index, 'before'),
                    onFetchNeighbors(chunk.document_id, chunk.chunk_index, 'after'),
                ])
                setNeighborChunks({ before, after })
            } catch (error) {
                console.warn('Failed to fetch neighbor chunks:', error)
            } finally {
                setLoadingNeighbors(false)
            }
        }
    }

    const handleCopyContent = (content: string) => {
        navigator.clipboard.writeText(content)
    }

    if (chunks.length === 0 && !isLoading) {
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
                        <ArticleIcon fontSize="small" color="primary" />
                        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                            📚 Retrieved Context
                        </Typography>
                        <Chip
                            label={`${chunks.length} chunks`}
                            size="small"
                            variant="outlined"
                        />
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
                            Searching documents...
                        </Typography>
                    </Box>
                )}

                {/* Chunks List */}
                <Collapse in={expanded && !isLoading}>
                    <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
                        {chunks.map((chunk, index) => (
                            <Box key={chunk.id || index}>
                                {index > 0 && <Divider />}
                                <Box
                                    sx={{
                                        p: 2,
                                        '&:hover': { bgcolor: 'action.hover' },
                                        cursor: 'pointer',
                                    }}
                                    onClick={() => handleOpenChunkExplorer(chunk)}
                                >
                                    {/* Chunk Header */}
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                        <Typography variant="caption" color="primary" sx={{ fontWeight: 600 }}>
                                            {chunk.document_title || `Document ${chunk.document_id.slice(0, 8)}`}
                                        </Typography>
                                        <Chip
                                            label={`Score: ${(chunk.score * 100).toFixed(1)}%`}
                                            size="small"
                                            color={chunk.score > 0.8 ? 'success' : chunk.score > 0.6 ? 'warning' : 'default'}
                                            sx={{ height: 20, fontSize: '0.7rem' }}
                                        />
                                        <Chip
                                            label={`Chunk #${chunk.chunk_index}`}
                                            size="small"
                                            variant="outlined"
                                            sx={{ height: 20, fontSize: '0.7rem' }}
                                        />
                                        <Tooltip title="Explore chunks">
                                            <IconButton size="small">
                                                <OpenIcon fontSize="small" />
                                            </IconButton>
                                        </Tooltip>
                                    </Box>

                                    {/* Chunk Content Preview */}
                                    <Typography
                                        variant="body2"
                                        sx={{
                                            color: 'text.secondary',
                                            display: '-webkit-box',
                                            WebkitLineClamp: 3,
                                            WebkitBoxOrient: 'vertical',
                                            overflow: 'hidden',
                                            textOverflow: 'ellipsis',
                                            lineHeight: 1.5,
                                        }}
                                    >
                                        {chunk.content}
                                    </Typography>
                                </Box>
                            </Box>
                        ))}
                    </Box>
                </Collapse>
            </Paper>

            {/* Chunk Explorer Dialog */}
            <Dialog
                open={selectedChunk !== null}
                onClose={() => setSelectedChunk(null)}
                maxWidth="md"
                fullWidth
            >
                <DialogTitle>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <ArticleIcon color="primary" />
                        <Typography variant="h6">
                            {selectedChunk?.document_title || 'Chunk Explorer'}
                        </Typography>
                    </Box>
                </DialogTitle>
                <DialogContent dividers>
                    {loadingNeighbors ? (
                        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                            <CircularProgress />
                        </Box>
                    ) : (
                        <Box>
                            {/* Before Chunks */}
                            {neighborChunks.before.length > 0 && (
                                <Box sx={{ mb: 2, opacity: 0.7 }}>
                                    <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
                                        <PrevIcon fontSize="small" />
                                        Previous chunks
                                    </Typography>
                                    {neighborChunks.before.map((chunk, i) => (
                                        <Paper key={i} sx={{ p: 2, mb: 1, bgcolor: 'action.hover' }}>
                                            <Typography variant="caption" color="text.secondary">
                                                Chunk #{chunk.chunk_index}
                                            </Typography>
                                            <Typography variant="body2">{chunk.content}</Typography>
                                        </Paper>
                                    ))}
                                </Box>
                            )}

                            {/* Selected Chunk */}
                            {selectedChunk && (
                                <Paper
                                    sx={{
                                        p: 2,
                                        mb: 2,
                                        border: '2px solid',
                                        borderColor: 'primary.main',
                                        bgcolor: 'primary.50',
                                    }}
                                >
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                            <Typography variant="subtitle2" color="primary">
                                                Chunk #{selectedChunk.chunk_index}
                                            </Typography>
                                            <Chip
                                                label={`Match: ${(selectedChunk.score * 100).toFixed(1)}%`}
                                                size="small"
                                                color="primary"
                                            />
                                        </Box>
                                        <Tooltip title="Copy content">
                                            <IconButton size="small" onClick={() => handleCopyContent(selectedChunk.content)}>
                                                <CopyIcon fontSize="small" />
                                            </IconButton>
                                        </Tooltip>
                                    </Box>
                                    <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                                        {selectedChunk.content}
                                    </Typography>

                                    {/* Metadata */}
                                    {selectedChunk.metadata && Object.keys(selectedChunk.metadata).length > 0 && (
                                        <Box sx={{ mt: 2, pt: 1, borderTop: '1px solid', borderColor: 'divider' }}>
                                            <Typography variant="caption" color="text.secondary">Metadata:</Typography>
                                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                                                {Object.entries(selectedChunk.metadata).map(([key, value]) => (
                                                    <Chip key={key} label={`${key}: ${value}`} size="small" variant="outlined" />
                                                ))}
                                            </Box>
                                        </Box>
                                    )}
                                </Paper>
                            )}

                            {/* After Chunks */}
                            {neighborChunks.after.length > 0 && (
                                <Box sx={{ opacity: 0.7 }}>
                                    <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
                                        <NextIcon fontSize="small" />
                                        Following chunks
                                    </Typography>
                                    {neighborChunks.after.map((chunk, i) => (
                                        <Paper key={i} sx={{ p: 2, mb: 1, bgcolor: 'action.hover' }}>
                                            <Typography variant="caption" color="text.secondary">
                                                Chunk #{chunk.chunk_index}
                                            </Typography>
                                            <Typography variant="body2">{chunk.content}</Typography>
                                        </Paper>
                                    ))}
                                </Box>
                            )}
                        </Box>
                    )}
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setSelectedChunk(null)}>Close</Button>
                </DialogActions>
            </Dialog>
        </Box>
    )
}

export default RAGContextBlock
