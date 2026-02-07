import React, { useState, useEffect } from 'react'
import {
    Box,
    Typography,
    Switch,
    FormControlLabel,
    Slider,
    Chip,
    CircularProgress,
    Collapse,
    IconButton,
    FormControl,
    RadioGroup,
    Radio,
    Autocomplete,
    TextField,
    Tooltip,
} from '@mui/material'
import {
    MenuBook as MenuBookIcon,
    ExpandMore as ExpandMoreIcon,
    ExpandLess as ExpandLessIcon,
    Public as GlobalIcon,
    FolderSpecial as DomainIcon,
} from '@mui/icons-material'

interface RAGDomain {
    id: string
    name: string
    description?: string
    document_count: number
}

interface RAGSettingsPanelProps {
    ragEnabled: boolean
    ragSearchMode: 'global' | 'domains'
    ragSelectedDomains: string[]
    ragTopK: number
    ragShowContext: boolean
    onSettingsChange: (settings: {
        ragEnabled?: boolean
        ragSearchMode?: 'global' | 'domains'
        ragSelectedDomains?: string[]
        ragTopK?: number
        ragShowContext?: boolean
    }) => void
}

export const RAGSettingsPanel: React.FC<RAGSettingsPanelProps> = ({
    ragEnabled,
    ragSearchMode,
    ragSelectedDomains,
    ragTopK,
    ragShowContext,
    onSettingsChange,
}) => {
    const [expanded, setExpanded] = useState(ragEnabled)
    const [domains, setDomains] = useState<RAGDomain[]>([])
    const [loadingDomains, setLoadingDomains] = useState(false)

    // Fetch domains when panel expands or RAG is enabled
    useEffect(() => {
        if (expanded || ragEnabled) {
            fetchDomains()
        }
    }, [expanded, ragEnabled])

    const fetchDomains = async () => {
        setLoadingDomains(true)
        try {
            const response = await fetch('/api/v1/rag/domains')
            if (response.ok) {
                const data = await response.json()
                setDomains(data)
            }
        } catch (error) {
            console.warn('Failed to fetch RAG domains:', error)
        } finally {
            setLoadingDomains(false)
        }
    }

    const handleToggleRAG = (enabled: boolean) => {
        onSettingsChange({ ragEnabled: enabled })
        if (enabled) setExpanded(true)
    }

    const selectedDomainObjects = domains.filter(d => ragSelectedDomains.includes(d.id))

    return (
        <Box sx={{ mb: 2 }}>
            {/* Header */}
            <Box
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    cursor: 'pointer',
                    py: 1,
                }}
                onClick={() => setExpanded(!expanded)}
            >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <MenuBookIcon sx={{ color: ragEnabled ? 'primary.main' : 'text.secondary' }} />
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                        RAG Context
                    </Typography>
                    {ragEnabled && (
                        <Chip
                            label={ragSearchMode === 'global' ? 'Global' : `${ragSelectedDomains.length} domains`}
                            size="small"
                            color="primary"
                            variant="outlined"
                        />
                    )}
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Switch
                        checked={ragEnabled}
                        onChange={(e) => {
                            e.stopPropagation()
                            handleToggleRAG(e.target.checked)
                        }}
                        size="small"
                    />
                    <IconButton size="small">
                        {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                    </IconButton>
                </Box>
            </Box>

            {/* Expanded Settings */}
            <Collapse in={expanded}>
                <Box sx={{ pl: 4, pr: 1, pb: 2 }}>
                    {/* Search Mode */}
                    <FormControl component="fieldset" sx={{ mb: 2 }}>
                        <Typography variant="caption" color="text.secondary" sx={{ mb: 1 }}>
                            Search Mode
                        </Typography>
                        <RadioGroup
                            value={ragSearchMode}
                            onChange={(e) => onSettingsChange({ ragSearchMode: e.target.value as 'global' | 'domains' })}
                            row
                        >
                            <FormControlLabel
                                value="global"
                                control={<Radio size="small" />}
                                label={
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                        <GlobalIcon fontSize="small" />
                                        <span>Global</span>
                                    </Box>
                                }
                            />
                            <FormControlLabel
                                value="domains"
                                control={<Radio size="small" />}
                                label={
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                        <DomainIcon fontSize="small" />
                                        <span>Specific Domains</span>
                                    </Box>
                                }
                            />
                        </RadioGroup>
                    </FormControl>

                    {/* Domain Selection */}
                    {ragSearchMode === 'domains' && (
                        <Box sx={{ mb: 2 }}>
                            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                                Select Domains
                            </Typography>
                            {loadingDomains ? (
                                <CircularProgress size={20} />
                            ) : (
                                <Autocomplete
                                    multiple
                                    options={domains}
                                    getOptionLabel={(option) => option.name}
                                    value={selectedDomainObjects}
                                    onChange={(_, newValue) => {
                                        onSettingsChange({ ragSelectedDomains: newValue.map(d => d.id) })
                                    }}
                                    renderInput={(params) => (
                                        <TextField
                                            {...params}
                                            placeholder="Search domains..."
                                            size="small"
                                            variant="outlined"
                                        />
                                    )}
                                    renderOption={(props, option) => (
                                        <li {...props}>
                                            <Box>
                                                <Typography variant="body2">{option.name}</Typography>
                                                <Typography variant="caption" color="text.secondary">
                                                    {option.document_count} documents
                                                </Typography>
                                            </Box>
                                        </li>
                                    )}
                                    renderTags={(value, getTagProps) =>
                                        value.map((option, index) => (
                                            <Chip
                                                {...getTagProps({ index })}
                                                key={option.id}
                                                label={option.name}
                                                size="small"
                                            />
                                        ))
                                    }
                                    size="small"
                                    sx={{ maxWidth: 400 }}
                                />
                            )}
                        </Box>
                    )}

                    {/* Top K Slider */}
                    <Box sx={{ mb: 2 }}>
                        <Typography variant="caption" color="text.secondary">
                            Results to retrieve: {ragTopK}
                        </Typography>
                        <Slider
                            value={ragTopK}
                            onChange={(_, value) => onSettingsChange({ ragTopK: value as number })}
                            min={1}
                            max={20}
                            step={1}
                            marks={[
                                { value: 1, label: '1' },
                                { value: 10, label: '10' },
                                { value: 20, label: '20' },
                            ]}
                            size="small"
                            sx={{ maxWidth: 300 }}
                        />
                    </Box>

                    {/* Show Context Toggle */}
                    <Tooltip title="Display retrieved chunks in the chat interface">
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={ragShowContext}
                                    onChange={(e) => onSettingsChange({ ragShowContext: e.target.checked })}
                                    size="small"
                                />
                            }
                            label={
                                <Typography variant="body2">
                                    Show retrieved context
                                </Typography>
                            }
                        />
                    </Tooltip>
                </Box>
            </Collapse>
        </Box>
    )
}

export default RAGSettingsPanel
