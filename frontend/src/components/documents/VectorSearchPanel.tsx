import React, { useEffect, useState, useCallback } from 'react';
import {
    Box,
    Paper,
    Typography,
    TextField,
    Button,
    IconButton,
    Chip,
    Slider,
    CircularProgress,
    Collapse,
    ToggleButton,
    ToggleButtonGroup,
    Tooltip,
    alpha,
} from '@mui/material';
import {
    Search as SearchIcon,
    TravelExplore as GlobalIcon,
    Folder as DomainIcon,
    ExpandMore as ExpandIcon,
    ExpandLess as CollapseIcon,
    ContentCopy as CopyIcon,
    Close as CloseIcon,
} from '@mui/icons-material';
import { apiService as api } from '../../services/api';
import { useClipboard } from '@/hooks/useClipboard';

interface Domain {
    id: string;
    name: string;
}

interface SearchResult {
    chunk_id: string;
    document_id: string;
    content: string;
    score: number;
    domain_id?: string;
    document_name?: string;
    chunk_index?: number;
    total_chunks?: number;
    section_header?: string;
    metadata?: Record<string, unknown>;
}

interface VectorSearchPanelProps {
    selectedDomain: Domain | null;
    defaultExpanded?: boolean;
}

function getScoreColor(score: number): string {
    if (score >= 0.8) return '#10B981';
    if (score >= 0.6) return '#06B6D4';
    if (score >= 0.4) return '#F59E0B';
    return '#EF4444';
}

function getScoreLabel(score: number): string {
    if (score >= 0.8) return 'Excellent';
    if (score >= 0.6) return 'Good';
    if (score >= 0.4) return 'Fair';
    return 'Low';
}

const VectorSearchPanel: React.FC<VectorSearchPanelProps> = ({
    selectedDomain,
    defaultExpanded = true,
}) => {
    const [expanded, setExpanded] = useState(defaultExpanded);
    const [userToggledExpanded, setUserToggledExpanded] = useState(false);
    const [query, setQuery] = useState('');
    const [topK, setTopK] = useState(5);
    const [scope, setScope] = useState<'global' | 'domain'>('global');
    const [results, setResults] = useState<SearchResult[]>([]);
    const [searching, setSearching] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [expandedResults, setExpandedResults] = useState<Set<number>>(new Set());
    const [hasSearched, setHasSearched] = useState(false);
    const copyToClipboard = useClipboard();

    const handleSearch = useCallback(async () => {
        if (!query.trim()) return;

        const effectiveScope = scope === 'domain' && selectedDomain ? 'domain' : 'global';

        setSearching(true);
        setError(null);
        setHasSearched(true);

        try {
            const body: Record<string, unknown> = {
                query: query.trim(),
                top_k: topK,
            };
            if (effectiveScope === 'domain' && selectedDomain) {
                body.domain_id = selectedDomain.id;
            }

            const res = await api.post('/api/v1/rag/search', body);
            setResults(res.results || []);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : 'Search failed';
            setError(message);
            setResults([]);
        } finally {
            setSearching(false);
        }
    }, [query, topK, scope, selectedDomain]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    };

    const toggleResultExpanded = (index: number) => {
        setExpandedResults(prev => {
            const next = new Set(prev);
            if (next.has(index)) {
                next.delete(index);
            } else {
                next.add(index);
            }
            return next;
        });
    };

    const copyContent = (content: string) => {
        copyToClipboard(content, {
            successMessage: 'Search result content copied.',
            successTitle: 'Content copied',
        });
    };

    useEffect(() => {
        if (!userToggledExpanded) {
            setExpanded(defaultExpanded);
        }
    }, [defaultExpanded, userToggledExpanded]);

    return (
        <Paper
            sx={{
                mb: 0,
                bgcolor: 'background.paper',
                border: '1px solid',
                borderColor: expanded ? alpha('#06B6D4', 0.36) : 'divider',
                borderRadius: '8px',
                overflow: 'hidden',
            }}
        >
            {/* Header - always visible */}
            <Box
                onClick={() => {
                    setUserToggledExpanded(true);
                    setExpanded((current) => !current);
                }}
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    px: 1.5,
                    py: 1,
                    cursor: 'pointer',
                    '&:hover': { bgcolor: 'action.hover' },
                }}
            >
                <SearchIcon sx={{
                    color: '#06B6D4',
                    fontSize: 20,
                }} />
                <Typography variant="subtitle2" sx={{ fontWeight: 600, flex: 1 }}>
                    Vector Search
                </Typography>
                {results.length > 0 && !expanded && (
                    <Chip
                        label={`${results.length} results`}
                        size="small"
                        sx={{
                            height: 22,
                            bgcolor: alpha('#06B6D4', 0.2),
                            color: '#06B6D4',
                            '& .MuiChip-label': { px: 1, fontSize: 11 },
                        }}
                    />
                )}
                {expanded ? <CollapseIcon fontSize="small" /> : <ExpandIcon fontSize="small" />}
            </Box>

            {/* Search interface */}
            <Collapse in={expanded}>
                <Box sx={{ px: 1.5, pb: 1.5 }}>
                    {/* Search input row */}
                    <Box sx={{ display: 'flex', gap: 1, mb: 1, flexWrap: { xs: 'wrap', md: 'nowrap' } }}>
                        <TextField
                            size="small"
                            fullWidth
                            placeholder="Semantic search across your documents..."
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            onKeyDown={handleKeyDown}
                            sx={{
                                '& .MuiOutlinedInput-root': {
                                    '& fieldset': { borderColor: 'rgba(6, 182, 212, 0.3)' },
                                    '&:hover fieldset': { borderColor: 'rgba(6, 182, 212, 0.5)' },
                                    '&.Mui-focused fieldset': { borderColor: '#06B6D4' },
                                },
                            }}
                        />
                        <Button
                            variant="contained"
                            size="small"
                            onClick={handleSearch}
                            disabled={searching || !query.trim()}
                            sx={{
                                minWidth: { xs: '100%', md: 92 },
                                height: 40,
                                borderRadius: '8px',
                                fontWeight: 800,
                            }}
                        >
                            {searching ? <CircularProgress size={20} color="inherit" /> : 'Search'}
                        </Button>
                    </Box>

                    {/* Controls row */}
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1.25, flexWrap: 'wrap' }}>
                        {/* Scope toggle */}
                        <ToggleButtonGroup
                            size="small"
                            value={scope}
                            exclusive
                            onChange={(_, v) => { if (v) setScope(v); }}
                            sx={{
                                '& .MuiToggleButton-root': {
                                    fontSize: 12,
                                    px: 1.5,
                                    py: 0.5,
                                    textTransform: 'none',
                                    borderColor: 'rgba(255,255,255,0.15)',
                                    color: 'text.secondary',
                                    '&.Mui-selected': {
                                        bgcolor: alpha('#06B6D4', 0.15),
                                        color: '#06B6D4',
                                        borderColor: 'rgba(6, 182, 212, 0.4)',
                                    },
                                },
                            }}
                        >
                            <ToggleButton value="global">
                                <GlobalIcon sx={{ fontSize: 16, mr: 0.5 }} />
                                All Domains
                            </ToggleButton>
                            <ToggleButton value="domain" disabled={!selectedDomain}>
                                <DomainIcon sx={{ fontSize: 16, mr: 0.5 }} />
                                {selectedDomain ? selectedDomain.name : 'Select Domain'}
                            </ToggleButton>
                        </ToggleButtonGroup>

                        {/* Top-K slider */}
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: '1 1 180px', maxWidth: 240 }}>
                            <Typography variant="caption" color="text.secondary" sx={{ whiteSpace: 'nowrap' }}>
                                Top-K:
                            </Typography>
                            <Slider
                                value={topK}
                                onChange={(_, v) => setTopK(v as number)}
                                min={1}
                                max={20}
                                size="small"
                                valueLabelDisplay="auto"
                                sx={{
                                    color: '#06B6D4',
                                    '& .MuiSlider-thumb': { width: 14, height: 14 },
                                }}
                            />
                            <Typography variant="caption" sx={{ fontWeight: 600, minWidth: 16, textAlign: 'right' }}>
                                {topK}
                            </Typography>
                        </Box>

                        {/* Clear results */}
                        {results.length > 0 && (
                            <Tooltip title="Clear results">
                                <IconButton
                                    size="small"
                                    onClick={() => { setResults([]); setHasSearched(false); }}
                                    sx={{ color: 'text.secondary' }}
                                >
                                    <CloseIcon fontSize="small" />
                                </IconButton>
                            </Tooltip>
                        )}
                    </Box>

                    {/* Error */}
                    {error && (
                        <Typography variant="body2" color="error" sx={{ mb: 1 }}>
                            {error}
                        </Typography>
                    )}

                    {/* Results */}
                    {hasSearched && !searching && results.length === 0 && !error && (
                        <Box sx={{ textAlign: 'center', py: 3 }}>
                            <Typography variant="body2" color="text.secondary">
                                No results found. Try a different query or broaden the search scope.
                            </Typography>
                        </Box>
                    )}

                    {results.length > 0 && (
                        <Box sx={{ mt: 1 }}>
                            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                                {results.length} result{results.length !== 1 ? 's' : ''} found
                            </Typography>

                            {results.map((result, index) => {
                                const isExpanded = expandedResults.has(index);
                                const preview = result.content.length > 200
                                    ? result.content.slice(0, 200) + '...'
                                    : result.content;
                                const domainName = result.metadata?.domain_name as string || '';

                                return (
                                    <Paper
                                        key={result.chunk_id || index}
                                        sx={{
                                            p: 1.5,
                                            mb: 1,
                                            bgcolor: alpha('#000', 0.3),
                                            border: '1px solid',
                                            borderColor: alpha(getScoreColor(result.score), 0.3),
                                            borderRadius: 1.5,
                                            cursor: 'pointer',
                                            transition: 'all 0.2s',
                                            '&:hover': {
                                                bgcolor: alpha('#000', 0.4),
                                                borderColor: alpha(getScoreColor(result.score), 0.5),
                                            },
                                        }}
                                        onClick={() => toggleResultExpanded(index)}
                                    >
                                        {/* Result header */}
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                                            {/* Score badge */}
                                            <Tooltip title={`Similarity: ${(result.score * 100).toFixed(1)}% — ${getScoreLabel(result.score)}`}>
                                                <Chip
                                                    label={`${(result.score * 100).toFixed(0)}%`}
                                                    size="small"
                                                    sx={{
                                                        height: 22,
                                                        fontWeight: 700,
                                                        fontSize: 11,
                                                        bgcolor: alpha(getScoreColor(result.score), 0.2),
                                                        color: getScoreColor(result.score),
                                                        borderColor: alpha(getScoreColor(result.score), 0.4),
                                                        border: '1px solid',
                                                        '& .MuiChip-label': { px: 1 },
                                                    }}
                                                />
                                            </Tooltip>

                                            {/* Document name */}
                                            <Typography variant="body2" sx={{ fontWeight: 600, flex: 1, fontSize: 13 }}>
                                                {result.document_name || 'Unknown Document'}
                                            </Typography>

                                            {/* Domain chip (for global search) */}
                                            {domainName && (
                                                <Chip
                                                    label={domainName}
                                                    size="small"
                                                    icon={<DomainIcon sx={{ fontSize: '14px !important' }} />}
                                                    sx={{
                                                        height: 20,
                                                        bgcolor: alpha('#F59E0B', 0.15),
                                                        color: '#F59E0B',
                                                        '& .MuiChip-label': { px: 0.5, fontSize: 10 },
                                                        '& .MuiChip-icon': { color: '#F59E0B', ml: 0.5 },
                                                    }}
                                                />
                                            )}

                                            {/* Chunk index */}
                                            {result.chunk_index !== null && result.chunk_index !== undefined && (
                                                <Chip
                                                    label={`Chunk ${result.chunk_index + 1}`}
                                                    size="small"
                                                    sx={{
                                                        height: 20,
                                                        bgcolor: alpha('#6366F1', 0.15),
                                                        color: '#6366F1',
                                                        '& .MuiChip-label': { px: 0.5, fontSize: 10 },
                                                    }}
                                                />
                                            )}

                                            <Tooltip title="Copy content">
                                                <IconButton
                                                    size="small"
                                                    onClick={(e) => { e.stopPropagation(); copyContent(result.content); }}
                                                    sx={{ color: 'text.disabled', '&:hover': { color: 'text.secondary' } }}
                                                >
                                                    <CopyIcon sx={{ fontSize: 14 }} />
                                                </IconButton>
                                            </Tooltip>
                                        </Box>

                                        {/* Section header */}
                                        {result.section_header && (
                                            <Chip
                                                label={result.section_header}
                                                size="small"
                                                sx={{ mb: 0.5, height: 18, bgcolor: alpha('#6366F1', 0.15), '& .MuiChip-label': { fontSize: 10, px: 0.5 } }}
                                            />
                                        )}

                                        {/* Content */}
                                        <Typography
                                            variant="body2"
                                            sx={{
                                                fontSize: 12,
                                                color: 'text.secondary',
                                                whiteSpace: 'pre-wrap',
                                                wordBreak: 'break-word',
                                                lineHeight: 1.5,
                                                maxHeight: isExpanded ? 'none' : 80,
                                                overflow: 'hidden',
                                                fontFamily: 'monospace',
                                            }}
                                        >
                                            {isExpanded ? result.content : preview}
                                        </Typography>

                                        {result.content.length > 200 && (
                                            <Typography
                                                variant="caption"
                                                sx={{
                                                    color: '#06B6D4',
                                                    mt: 0.5,
                                                    display: 'block',
                                                    fontSize: 11,
                                                }}
                                            >
                                                {isExpanded ? 'Click to collapse' : 'Click to expand full content'}
                                            </Typography>
                                        )}
                                    </Paper>
                                );
                            })}
                        </Box>
                    )}
                </Box>
            </Collapse>
        </Paper>
    );
};

export default VectorSearchPanel;
