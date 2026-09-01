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
    Inventory2 as ArtifactsIcon,
    Hub as HubIcon,
    ArrowForward as ArrowIcon,
    Search as SearchIcon,
    AccountTree as LinkIcon,
} from '@mui/icons-material'
import { formatGraphRAGReasoningLocation } from '@/utils/graphragReasoningResults'

// ── Types ─────────────────────────────────────────────────────────────

export interface UnifiedChunk {
    id: string
    content: string
    score: number
    boosted_score?: number
    entity_boost?: number
    graph_reasoning_boost?: number
    matched_entities?: string[]
    matched_graph_path_terms?: string[]
    retrieval_reasons?: string[]
    retrieval_queries?: string[]
    document_id?: string
    document_title?: string
    chunk_index?: number
    graph_context?: Array<{
        source?: string
        relation?: string
        target?: string
        evidence?: string
        text?: string
        provenance?: Record<string, unknown>
    }>
    graph_paths?: Array<{
        source?: string
        bridge?: string
        target?: string
        nodes?: string[]
        hops?: number
        score?: number
        chunk_match_score?: number
        chunk_match_reasons?: string[]
        evidence_families?: string[]
        provenance_summary?: string[]
        explanation?: string
        answer_hint?: string
        text?: string
    }>
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
    provenance?: string
    metadata?: Record<string, unknown>
    matched_query_terms?: string[]
}

export interface CrossReference {
    chunk_index: number
    chunk_id: string
    matched_entities: string[]
    matched_graph_path_terms?: string[]
    overlap_score: number
    graph_path_overlap_score?: number
}

export interface QualitySignals {
    avg_chunk_relevance: number
    entity_coverage: number
    graph_expansion_count: number
    cross_reference_count: number
    graph_path_count?: number
    graph_layout_path_count?: number
    graph_semantic_path_count?: number
    direct_graph_relationship_count?: number
    graph_matched_query_terms?: string[]
    graph_evidence_chunk_count?: number
    graph_augmented_query_used?: boolean
    graph_augmented_retrieval_terms?: string[]
    supplemental_chunk_count?: number
    graph_augmented_chunk_count?: number
    remote_graph_used?: boolean
    local_pipeline_graph_run_count?: number
    local_pipeline_graph_edge_count?: number
    local_pipeline_context_chunk_count?: number
    local_pipeline_context_run_count?: number
    local_pipeline_artifact_run_count?: number
    local_pipeline_artifact_count?: number
    local_pipeline_structured_artifact_count?: number
    local_pipeline_source_artifact_count?: number
    local_pipeline_graph_context_chunk_count?: number
    local_pipeline_graph_context_relationship_count?: number
    local_pipeline_graph_context_matched_chunk_count?: number
    local_pipeline_graph_path_context_chunk_count?: number
    local_pipeline_graph_path_context_count?: number
    local_pipeline_graph_path_context_matched_chunk_count?: number
    local_pipeline_evidence_context_chunk_count?: number
    local_pipeline_interaction_handoff_count?: number
    local_pipeline_handoff_graph_anchor_count?: number
    local_pipeline_handoff_capability_card_count?: number
    local_pipeline_handoff_starter_prompt_count?: number
    local_pipeline_handoff_retrieval_rehearsal_count?: number
    local_pipeline_handoff_rehearsed_chunk_match_count?: number
    local_pipeline_handoff_rehearsed_anchor_match_count?: number
    local_pipeline_handoff_modes?: string[]
    local_pipeline_handoff_citation_scopes?: string[]
    graph_overview_relationship_count?: number
    graph_overview_top_relationship_count?: number
    graph_overview_shared_entity_count?: number
    graph_overview_bridge_entity_count?: number
    graph_overview_cross_document_path_count?: number
    graph_answer_strategy?: string
    graph_answer_evidence_card_count?: number
    graph_answer_provenance_anchor_count?: number
    pipeline_evidence_family_count?: number
    pipeline_evidence_total_count?: number
    pipeline_evidence_top_families?: string[]
    pipeline_evidence_guidance_count?: number
    total_chunks: number
    total_entities: number
}

export interface GraphReasoningPath {
    source: string
    target: string
    hops: number
    nodes: string[]
    relationships: Array<{
        source: string
        relation: string
        target: string
        direction?: string
        weight?: number
        evidence?: string
        provenance?: string
        provenance_summary?: string
        evidence_families?: string[]
        rationale?: string
        metadata?: Record<string, unknown>
    }>
    score?: number
    average_weight?: number
    path_type?: string
    matched_query_terms?: string[]
    evidence_families?: string[]
    provenance_summary?: string[]
    explanation?: string
    answer_hint?: string
    text: string
}

export interface GraphAnswerHop {
    index?: number
    source?: string
    relation?: string
    target?: string
    direction?: string
    relationship_type?: string
    provenance?: string
    provenance_summary?: string
    evidence_families?: string[]
    rationale?: string
    evidence?: string | null
    text?: string
}

export interface GraphAnswerEvidenceCard {
    id?: string
    index?: number
    role?: string
    kind?: string
    title?: string
    source?: string
    relation?: string
    target?: string
    relationship_type?: string
    citation_label?: string
    provenance?: string
    provenance_summary?: string
    evidence_families?: string[]
    rationale?: string
    evidence?: string
    location?: Record<string, unknown>
    text?: string
}

export interface GraphAnswerPlan {
    strategy?: string
    instruction?: string
    primary_path?: Partial<GraphReasoningPath> | null
    hops?: GraphAnswerHop[]
    bridge_entities?: string[]
    supporting_relationships?: GraphAnswerHop[]
    evidence_cards?: GraphAnswerEvidenceCard[]
    provenance_anchors?: string[]
    fallback?: string | null
}

export interface GraphReasoningBridgeEntity {
    name: string
    path_count: number
    score?: number
    text?: string
    relationships?: string[]
    path_types?: string[]
    matched_query_terms?: string[]
}

export interface GraphReasoningSummary {
    mode?: string
    answer_guidance?: string
    summary_text?: string
    evidence?: string[]
    key_entities?: string[]
    evidence_terms?: string[]
    bridge_entities?: string[]
    path_count?: number
    direct_relationship_count?: number
    bridge_entity_count?: number
    overview_relationship_count?: number
    shared_entity_count?: number
}

export interface GraphReasoningOverviewRelationship {
    source?: string
    target?: string
    relation?: string
    score?: number
    weight?: number
    relationship_type?: string
    evidence?: string
    provenance?: string
    metadata?: Record<string, unknown>
    matched_query_terms?: string[]
}

export interface GraphReasoningSharedEntity {
    name?: string
    run_count?: number
    filenames?: string[]
    relationships?: string[]
}

export interface GraphReasoningOverviewBridgeEntity {
    name?: string
    run_count?: number
    filenames?: string[]
    relationship_count?: number
    relationships?: string[]
    score?: number
}

export interface GraphReasoningOverviewPath {
    bridge?: string
    source?: string
    target?: string
    nodes?: string[]
    hops?: number
    score?: number
    filenames?: string[]
    run_ids?: string[]
    text?: string
    relationships?: GraphReasoningOverviewRelationship[]
}

export interface GraphReasoningOverview {
    source_count?: number
    relationship_count?: number
    relationship_type_counts?: Record<string, number>
    top_relationships?: GraphReasoningOverviewRelationship[]
    shared_entities?: GraphReasoningSharedEntity[]
    bridge_entity_count?: number
    bridge_entities?: GraphReasoningOverviewBridgeEntity[]
    cross_document_path_count?: number
    cross_document_paths?: GraphReasoningOverviewPath[]
}

export interface GraphReasoning {
    status: string
    reasoning_needed: boolean
    entity_names?: string[]
    entity_pairs?: Array<{ source: string; target: string }>
    paths?: GraphReasoningPath[]
    bridge_entities?: GraphReasoningBridgeEntity[]
    direct_relationships?: UnifiedRelationship[]
    query_complexity?: Record<string, unknown>
    graph_sources?: {
        remote_graph?: boolean
        local_pipeline_runs?: Array<{
            run_id?: string
            filename?: string
            domain?: string | null
            status?: string
            edge_count?: number
            updated_at?: string
        }>
    }
    summary?: GraphReasoningSummary
    graph_overview?: GraphReasoningOverview
    answer_plan?: GraphAnswerPlan
}

export interface PipelineArtifactDescriptor {
    kind?: string
    label?: string
    content_type?: string
    filename?: string
    endpoint?: string
}

export interface PipelineArtifactRun {
    run_id?: string
    filename?: string
    domain?: string | null
    status?: string
    artifact_count?: number
    artifacts?: PipelineArtifactDescriptor[]
    manifest_endpoint?: string
    structured_ingest?: {
        filename?: string
        char_count?: number
        sha256?: string
        truncated?: boolean
        excerpt?: string
    } | null
    local_graph?: {
        node_count?: number
        edge_count?: number
    } | null
    local_context?: {
        chunk_count?: number
        graph_context_relationship_count?: number
        graph_path_context_count?: number
        evidence_card_count?: number
    } | null
    local_graph_reasoning?: {
        bridge_entity_count?: number
        path_candidate_count?: number
    } | null
    source_artifacts?: Array<{
        kind?: string
        filename?: string
        mime_type?: string
        parsed_child_elements?: number
    }>
    evidence_cards?: Array<{
        chunk_id?: string
        category?: string
        categories?: string[]
        title?: string
        reason?: string
        scope_type?: string
        scope_label?: string
        token_estimate?: number
        graph_context_relationship_count?: number
        graph_path_context_count?: number
        evidence_preview?: string
        evidence_families?: string[]
        location_hints?: string[]
        source_locations?: Record<string, unknown>[]
        metadata?: Record<string, unknown>
    }>
}

export interface PipelineArtifactContext {
    runs?: PipelineArtifactRun[]
    run_count?: number
    artifact_count?: number
    structured_artifact_count?: number
    source_artifact_count?: number
}

interface UnifiedContextBlockProps {
    chunks: UnifiedChunk[]
    entities: UnifiedEntity[]
    relationships: UnifiedRelationship[]
    graphConnections?: string[]
    graphReasoning?: GraphReasoning | null
    pipelineArtifacts?: PipelineArtifactContext | null
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

const formatOverviewLabel = (value: string): string =>
    value.replace(/_/g, ' ').replace(/\b\w/g, character => character.toUpperCase())

const formatEvidenceFamilies = (families?: string[]): string[] =>
    Array.from(new Set((families || []).filter(Boolean))).slice(0, 8).map(formatOverviewLabel)

const metadataStringList = (value: unknown): string[] =>
    Array.isArray(value)
        ? Array.from(new Set(value.map(item => String(item || '').trim()).filter(Boolean)))
        : []

const chunkPipelineEvidenceTerms = (chunk: UnifiedChunk): string[] =>
    metadataStringList(chunk.metadata?.matched_local_pipeline_evidence_terms).slice(0, 8)

const overviewRelationshipProvenance = (relationship: GraphReasoningOverviewRelationship): string => {
    if (relationship.provenance) return relationship.provenance
    const metadata = relationship.metadata || {}
    const filename = typeof metadata.local_graph_filename === 'string' ? metadata.local_graph_filename : ''
    const slide = metadata.slide ?? metadata.slide_number ?? metadata.page ?? metadata.page_number ?? metadata.sheet
    const element = typeof metadata.element_type === 'string' ? metadata.element_type : ''
    const parts = [
        filename,
        slide !== undefined && slide !== null && String(slide).trim() ? `location ${slide}` : '',
        element ? formatOverviewLabel(element) : '',
    ].filter(Boolean)
    return parts.join(' | ')
}

const formatChunkGraphRelationship = (relationship: NonNullable<UnifiedChunk['graph_context']>[number]): string => {
    if (relationship.text) return relationship.text
    const source = relationship.source || 'Unknown source'
    const relation = formatOverviewLabel(relationship.relation || 'related_to')
    const target = relationship.target || 'Unknown target'
    return `${source} --[${relation}]--> ${target}`
}

const formatChunkGraphPath = (path: NonNullable<UnifiedChunk['graph_paths']>[number]): string => {
    if (path.text) return path.text
    const nodes = Array.isArray(path.nodes) ? path.nodes.filter(Boolean) : []
    if (nodes.length >= 3) return nodes.join(' -> ')
    const source = path.source || 'Source'
    const bridge = path.bridge || 'Bridge'
    const target = path.target || 'Target'
    return `${source} -> ${bridge} -> ${target}`
}

// ── Component ─────────────────────────────────────────────────────────

export const UnifiedContextBlock: React.FC<UnifiedContextBlockProps> = ({
    chunks,
    entities,
    relationships,
    graphConnections = [],
    graphReasoning = null,
    pipelineArtifacts = null,
    crossReferences = [],
    qualitySignals,
    isLoading = false,
    onEntityClick,
    onAskAbout: _onAskAbout,
}) => {
    const [expanded, setExpanded] = useState(true)
    const [showChunks, setShowChunks] = useState(true)
    const graphPaths = graphReasoning?.paths || []
    const graphBridgeEntities = graphReasoning?.bridge_entities || []
    const directGraphRelationships = graphReasoning?.direct_relationships || []
    const graphSummary = graphReasoning?.summary
    const graphAnswerPlan = graphReasoning?.answer_plan
    const graphPrimaryPath = graphAnswerPlan?.primary_path || null
    const graphAnswerHops = graphAnswerPlan?.hops || []
    const graphAnswerEvidenceCards = graphAnswerPlan?.evidence_cards || []
    const graphOverview = graphReasoning?.graph_overview
    const overviewRelationships = graphOverview?.top_relationships || []
    const overviewSharedEntities = graphOverview?.shared_entities || []
    const overviewBridgeEntities = graphOverview?.bridge_entities || []
    const overviewPaths = graphOverview?.cross_document_paths || []
    const relationshipTypeCounts = Object.entries(graphOverview?.relationship_type_counts || {})
    const localGraphSources = graphReasoning?.graph_sources?.local_pipeline_runs || []
    const artifactRuns = pipelineArtifacts?.runs || []
    const hasPipelineArtifacts = artifactRuns.length > 0 || (qualitySignals?.local_pipeline_artifact_run_count || 0) > 0
    const hasGraphSummary = Boolean(graphSummary?.summary_text || graphSummary?.evidence?.length || graphSummary?.answer_guidance)
    const hasGraphOverview = Boolean(
        (graphOverview?.relationship_count || 0) > 0 ||
        overviewRelationships.length > 0 ||
        overviewSharedEntities.length > 0 ||
        overviewBridgeEntities.length > 0 ||
        overviewPaths.length > 0,
    )

    const hasGraphAnswerPlan = Boolean(graphAnswerPlan?.strategy || graphPrimaryPath || graphAnswerHops.length > 0 || graphAnswerEvidenceCards.length > 0)
    const hasContent = chunks.length > 0 || entities.length > 0 || relationships.length > 0 || graphPaths.length > 0 || graphBridgeEntities.length > 0 || directGraphRelationships.length > 0 || hasGraphSummary || hasGraphOverview || hasGraphAnswerPlan || hasPipelineArtifacts
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
                        {graphPaths.length > 0 && (
                            <Chip
                                label={`${graphPaths.length} paths`}
                                size="small"
                                sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'rgba(16,185,129,0.14)' }}
                            />
                        )}
                        {graphBridgeEntities.length > 0 && (
                            <Chip
                                label={`${graphBridgeEntities.length} bridges`}
                                size="small"
                                sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'rgba(13,148,136,0.14)' }}
                            />
                        )}
                        {graphSummary?.mode && (
                            <Chip
                                label={graphSummary.mode.replace(/_/g, ' ')}
                                size="small"
                                sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'rgba(59,130,246,0.12)' }}
                            />
                        )}
                        {(qualitySignals?.graph_answer_evidence_card_count || 0) > 0 && (
                            <Tooltip title={`${qualitySignals?.graph_answer_provenance_anchor_count || 0} provenance anchor(s) available for graph answer planning`}>
                                <Chip
                                    label={`${qualitySignals?.graph_answer_evidence_card_count} answer evidence`}
                                    size="small"
                                    sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'rgba(37,99,235,0.14)' }}
                                />
                            </Tooltip>
                        )}
                        {hasGraphOverview && (
                            <Tooltip title={`${graphOverview?.relationship_count || overviewRelationships.length} relationships across ${graphOverview?.source_count || localGraphSources.length} pipeline trace${(graphOverview?.source_count || localGraphSources.length) === 1 ? '' : 's'}`}>
                                <Chip
                                    label={`${graphOverview?.relationship_count || overviewRelationships.length} graph overview`}
                                    size="small"
                                    sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'rgba(217,119,6,0.14)' }}
                                />
                            </Tooltip>
                        )}
                        {overviewSharedEntities.length > 0 && (
                            <Tooltip title={overviewSharedEntities.map(entity => entity.name).filter(Boolean).join(', ')}>
                                <Chip
                                    label={`${overviewSharedEntities.length} shared entities`}
                                    size="small"
                                    sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'rgba(22,163,74,0.14)' }}
                                />
                            </Tooltip>
                        )}
                        {(qualitySignals?.graph_evidence_chunk_count || 0) > 0 && (
                            <Chip
                                label={`${qualitySignals?.graph_evidence_chunk_count} evidence chunks`}
                                size="small"
                                sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'rgba(20,184,166,0.14)' }}
                            />
                        )}
                        {(qualitySignals?.graph_augmented_chunk_count || 0) > 0 && (
                            <Chip
                                label={`${qualitySignals?.graph_augmented_chunk_count} graph retrieved`}
                                size="small"
                                sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'rgba(14,165,233,0.14)' }}
                            />
                        )}
                        {(qualitySignals?.local_pipeline_evidence_context_chunk_count || 0) > 0 && (
                            <Tooltip title={`Retrieved from recent pipeline chunks using extracted evidence families: ${(qualitySignals?.pipeline_evidence_top_families || []).slice(0, 8).map(formatOverviewLabel).join(', ') || 'structured evidence'}`}>
                                <Chip
                                    label={`${qualitySignals?.local_pipeline_evidence_context_chunk_count} evidence-aware`}
                                    size="small"
                                    sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'rgba(236,72,153,0.14)' }}
                                />
                            </Tooltip>
                        )}
                        {localGraphSources.length > 0 && (
                            <Tooltip title={localGraphSources.map(source => source.filename || source.run_id).filter(Boolean).join(', ')}>
                                <Chip
                                    label={`${localGraphSources.length} local graph`}
                                    size="small"
                                    sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'rgba(245,158,11,0.14)' }}
                                />
                            </Tooltip>
                        )}
                        {(qualitySignals?.local_pipeline_interaction_handoff_count || 0) > 0 && (
                            <Tooltip title={`${qualitySignals?.local_pipeline_handoff_graph_anchor_count || 0} graph anchor(s), ${qualitySignals?.local_pipeline_handoff_starter_prompt_count || 0} starter prompt(s), ${qualitySignals?.local_pipeline_handoff_retrieval_rehearsal_count || 0} retrieval rehearsal(s), ${qualitySignals?.local_pipeline_handoff_rehearsed_chunk_match_count || 0} rehearsed chunk match(es)`}>
                                <Chip
                                    label={`${qualitySignals?.local_pipeline_interaction_handoff_count} handoff`}
                                    size="small"
                                    sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'rgba(34,197,94,0.14)' }}
                                />
                            </Tooltip>
                        )}
                        {(qualitySignals?.local_pipeline_context_chunk_count || 0) > 0 && (
                            <Chip
                                label={`${qualitySignals?.local_pipeline_context_chunk_count} local chunks`}
                                size="small"
                                sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'rgba(251,146,60,0.14)' }}
                            />
                        )}
                        {hasPipelineArtifacts && (
                            <Tooltip title={`${pipelineArtifacts?.run_count || qualitySignals?.local_pipeline_artifact_run_count || artifactRuns.length} pipeline run(s), ${pipelineArtifacts?.artifact_count || qualitySignals?.local_pipeline_artifact_count || 0} exportable artifact(s)`}>
                                <Chip
                                    icon={<ArtifactsIcon sx={{ fontSize: 12 }} />}
                                    label={`${pipelineArtifacts?.artifact_count || qualitySignals?.local_pipeline_artifact_count || 0} artifacts`}
                                    size="small"
                                    sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'rgba(168,85,247,0.14)' }}
                                />
                            </Tooltip>
                        )}
                        {(qualitySignals?.local_pipeline_graph_context_relationship_count || 0) > 0 && (
                            <Chip
                                label={`${qualitySignals?.local_pipeline_graph_context_relationship_count} chunk graph edges`}
                                size="small"
                                sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'rgba(13,148,136,0.14)' }}
                            />
                        )}
                        {(qualitySignals?.local_pipeline_graph_path_context_count || 0) > 0 && (
                            <Tooltip title={`${qualitySignals?.local_pipeline_graph_path_context_matched_chunk_count || 0} retrieved chunk(s) matched the query through local graph paths`}>
                                <Chip
                                    label={`${qualitySignals?.local_pipeline_graph_path_context_count} chunk graph paths`}
                                    size="small"
                                    sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'rgba(20,184,166,0.14)' }}
                                />
                            </Tooltip>
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
                                {(qualitySignals.graph_path_count || 0) > 0 && (
                                    <Tooltip title="Knowledge graph paths connecting query entities">
                                        <Chip
                                            label={`${qualitySignals.graph_path_count} graph paths`}
                                            size="small"
                                            sx={{ height: 22, fontSize: '0.7rem', bgcolor: 'rgba(16,185,129,0.12)' }}
                                        />
                                    </Tooltip>
                                )}
                                {qualitySignals.graph_answer_strategy && (
                                    <Tooltip title="Graph answer strategy selected from paths, direct relationships, overview evidence, or no-path fallback">
                                        <Chip
                                            label={`answer: ${formatOverviewLabel(qualitySignals.graph_answer_strategy)}`}
                                            size="small"
                                            sx={{ height: 22, fontSize: '0.7rem', bgcolor: 'rgba(59,130,246,0.12)' }}
                                        />
                                    </Tooltip>
                                )}
                                {(qualitySignals.graph_answer_evidence_card_count || 0) > 0 && (
                                    <Tooltip title={`${qualitySignals.graph_answer_provenance_anchor_count || 0} provenance anchor(s) from graph hops and supporting relationships`}>
                                        <Chip
                                            label={`${qualitySignals.graph_answer_evidence_card_count} answer evidence`}
                                            size="small"
                                            sx={{ height: 22, fontSize: '0.7rem', bgcolor: 'rgba(37,99,235,0.12)' }}
                                        />
                                    </Tooltip>
                                )}
                                {(qualitySignals.graph_layout_path_count || 0) > 0 && (
                                    <Tooltip title="Graph paths grounded in document structure such as slides, pages, tables, attachments, or archive entries">
                                        <Chip
                                            label={`${qualitySignals.graph_layout_path_count} layout paths`}
                                            size="small"
                                            sx={{ height: 22, fontSize: '0.7rem', bgcolor: 'rgba(34,197,94,0.12)' }}
                                        />
                                    </Tooltip>
                                )}
                                {(qualitySignals.graph_matched_query_terms || []).length > 0 && (
                                    <Tooltip title={`Graph terms matched from relationships and provenance: ${(qualitySignals.graph_matched_query_terms || []).slice(0, 8).join(', ')}`}>
                                        <Chip
                                            label={`matched: ${(qualitySignals.graph_matched_query_terms || []).slice(0, 3).join(', ')}`}
                                            size="small"
                                            sx={{ height: 22, fontSize: '0.7rem', bgcolor: 'rgba(168,85,247,0.12)' }}
                                        />
                                    </Tooltip>
                                )}
                                {(qualitySignals.graph_evidence_chunk_count || 0) > 0 && (
                                    <Tooltip title="Retrieved passages that matched graph path or relationship evidence">
                                        <Chip
                                            label={`${qualitySignals.graph_evidence_chunk_count} graph evidence`}
                                            size="small"
                                            sx={{ height: 22, fontSize: '0.7rem', bgcolor: 'rgba(20,184,166,0.12)' }}
                                        />
                                    </Tooltip>
                                )}
                                {(qualitySignals.graph_augmented_chunk_count || 0) > 0 && (
                                    <Tooltip title={`Extra retrieval seeded by graph terms: ${(qualitySignals.graph_augmented_retrieval_terms || []).slice(0, 6).join(', ')}`}>
                                        <Chip
                                            label={`${qualitySignals.graph_augmented_chunk_count} graph retrieved`}
                                            size="small"
                                            sx={{ height: 22, fontSize: '0.7rem', bgcolor: 'rgba(14,165,233,0.12)' }}
                                        />
                                    </Tooltip>
                                )}
                                {(qualitySignals.local_pipeline_evidence_context_chunk_count || 0) > 0 && (
                                    <Tooltip title={[
                                        `${qualitySignals.pipeline_evidence_family_count || 0} evidence family/families`,
                                        `${qualitySignals.pipeline_evidence_total_count || 0} extracted evidence item(s)`,
                                        (qualitySignals.pipeline_evidence_top_families || []).length > 0
                                            ? `families: ${(qualitySignals.pipeline_evidence_top_families || []).slice(0, 8).map(formatOverviewLabel).join(', ')}`
                                            : '',
                                    ].filter(Boolean).join(' · ')}>
                                        <Chip
                                            label={`${qualitySignals.local_pipeline_evidence_context_chunk_count} evidence-aware chunk${qualitySignals.local_pipeline_evidence_context_chunk_count === 1 ? '' : 's'}`}
                                            size="small"
                                            sx={{ height: 22, fontSize: '0.7rem', bgcolor: 'rgba(236,72,153,0.12)' }}
                                        />
                                    </Tooltip>
                                )}
                                {(qualitySignals.local_pipeline_graph_run_count || 0) > 0 && (
                                    <Tooltip title={`${qualitySignals.local_pipeline_graph_edge_count || 0} candidate relationships from recent pipeline traces`}>
                                        <Chip
                                            label={`${qualitySignals.local_pipeline_graph_run_count} local graph run${qualitySignals.local_pipeline_graph_run_count === 1 ? '' : 's'}`}
                                            size="small"
                                            sx={{ height: 22, fontSize: '0.7rem', bgcolor: 'rgba(245,158,11,0.12)' }}
                                        />
                                    </Tooltip>
                                )}
                                {(qualitySignals.local_pipeline_interaction_handoff_count || 0) > 0 && (
                                    <Tooltip title={[
                                        `${qualitySignals.local_pipeline_handoff_graph_anchor_count || 0} graph anchor(s)`,
                                        `${qualitySignals.local_pipeline_handoff_capability_card_count || 0} capability card(s)`,
                                        `${qualitySignals.local_pipeline_handoff_starter_prompt_count || 0} starter prompt(s)`,
                                        `${qualitySignals.local_pipeline_handoff_retrieval_rehearsal_count || 0} retrieval rehearsal(s)`,
                                        `${qualitySignals.local_pipeline_handoff_rehearsed_chunk_match_count || 0} rehearsed chunk match(es)`,
                                        `${qualitySignals.local_pipeline_handoff_rehearsed_anchor_match_count || 0} rehearsed anchor match(es)`,
                                        (qualitySignals.local_pipeline_handoff_modes || []).length > 0
                                            ? `modes: ${(qualitySignals.local_pipeline_handoff_modes || []).slice(0, 4).join(', ')}`
                                            : '',
                                        (qualitySignals.local_pipeline_handoff_citation_scopes || []).length > 0
                                            ? `scopes: ${(qualitySignals.local_pipeline_handoff_citation_scopes || []).slice(0, 6).join(', ')}`
                                            : '',
                                    ].filter(Boolean).join(' · ')}>
                                        <Chip
                                            label={`${qualitySignals.local_pipeline_interaction_handoff_count} interaction handoff${qualitySignals.local_pipeline_interaction_handoff_count === 1 ? '' : 's'}`}
                                            size="small"
                                            sx={{ height: 22, fontSize: '0.7rem', bgcolor: 'rgba(34,197,94,0.12)' }}
                                        />
                                    </Tooltip>
                                )}
                                {(qualitySignals.local_pipeline_context_chunk_count || 0) > 0 && (
                                    <Tooltip title="Retrieved chunk text directly from recent pipeline traces before remote ingestion completed">
                                        <Chip
                                            label={`${qualitySignals.local_pipeline_context_chunk_count} local context chunk${qualitySignals.local_pipeline_context_chunk_count === 1 ? '' : 's'}`}
                                            size="small"
                                            sx={{ height: 22, fontSize: '0.7rem', bgcolor: 'rgba(251,146,60,0.12)' }}
                                        />
                                    </Tooltip>
                                )}
                                {(qualitySignals.local_pipeline_artifact_count || 0) > 0 && (
                                    <Tooltip title={`${qualitySignals.local_pipeline_artifact_run_count || 0} run package(s), ${qualitySignals.local_pipeline_structured_artifact_count || 0} structured ingest artifact(s), ${qualitySignals.local_pipeline_source_artifact_count || 0} source artifact(s)`}>
                                        <Chip
                                            label={`${qualitySignals.local_pipeline_artifact_count} pipeline artifact${qualitySignals.local_pipeline_artifact_count === 1 ? '' : 's'}`}
                                            size="small"
                                            sx={{ height: 22, fontSize: '0.7rem', bgcolor: 'rgba(168,85,247,0.12)' }}
                                        />
                                    </Tooltip>
                                )}
                                {(qualitySignals.graph_overview_relationship_count || 0) > 0 && (
                                    <Tooltip title={`${qualitySignals.graph_overview_top_relationship_count || 0} top relationships, ${qualitySignals.graph_overview_shared_entity_count || 0} shared entities, ${qualitySignals.graph_overview_bridge_entity_count || 0} bridge entities, and ${qualitySignals.graph_overview_cross_document_path_count || 0} cross-document paths from the local pipeline graph overview`}>
                                        <Chip
                                            label={`${qualitySignals.graph_overview_relationship_count} overview relationship${qualitySignals.graph_overview_relationship_count === 1 ? '' : 's'}`}
                                            size="small"
                                            sx={{ height: 22, fontSize: '0.7rem', bgcolor: 'rgba(217,119,6,0.12)' }}
                                        />
                                    </Tooltip>
                                )}
                                {(qualitySignals.local_pipeline_graph_context_relationship_count || 0) > 0 && (
                                    <Tooltip title={`${qualitySignals.local_pipeline_graph_context_matched_chunk_count || 0} retrieved chunk(s) matched query terms through local pipeline graph context`}>
                                        <Chip
                                            label={`${qualitySignals.local_pipeline_graph_context_relationship_count} chunk graph edge${qualitySignals.local_pipeline_graph_context_relationship_count === 1 ? '' : 's'}`}
                                            size="small"
                                            sx={{ height: 22, fontSize: '0.7rem', bgcolor: 'rgba(13,148,136,0.12)' }}
                                        />
                                    </Tooltip>
                                )}
                                {(qualitySignals.local_pipeline_graph_path_context_count || 0) > 0 && (
                                    <Tooltip title={`${qualitySignals.local_pipeline_graph_path_context_matched_chunk_count || 0} retrieved chunk(s) matched query terms through local pipeline graph paths`}>
                                        <Chip
                                            label={`${qualitySignals.local_pipeline_graph_path_context_count} chunk graph path${qualitySignals.local_pipeline_graph_path_context_count === 1 ? '' : 's'}`}
                                            size="small"
                                            sx={{ height: 22, fontSize: '0.7rem', bgcolor: 'rgba(20,184,166,0.12)' }}
                                        />
                                    </Tooltip>
                                )}
                            </Box>
                        )}

                        {artifactRuns.length > 0 && (
                            <Box
                                sx={{
                                    mb: 2,
                                    p: 1,
                                    border: '1px solid',
                                    borderColor: 'rgba(168,85,247,0.26)',
                                    bgcolor: 'rgba(168,85,247,0.04)',
                                    borderRadius: 1,
                                }}
                            >
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 0.75 }}>
                                    <ArtifactsIcon sx={{ fontSize: 14, color: 'text.secondary' }} />
                                    <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', display: 'block' }}>
                                        Pipeline Artifacts
                                    </Typography>
                                    <Chip
                                        label={`${pipelineArtifacts?.run_count || artifactRuns.length} run${(pipelineArtifacts?.run_count || artifactRuns.length) === 1 ? '' : 's'}`}
                                        size="small"
                                        variant="outlined"
                                        sx={{ height: 20, fontSize: '0.65rem' }}
                                    />
                                    <Chip
                                        label={`${pipelineArtifacts?.artifact_count || 0} artifact${(pipelineArtifacts?.artifact_count || 0) === 1 ? '' : 's'}`}
                                        size="small"
                                        variant="outlined"
                                        sx={{ height: 20, fontSize: '0.65rem' }}
                                    />
                                </Box>
                                {artifactRuns.slice(0, 3).map((run) => {
                                    const artifacts = run.artifacts || []
                                    const structured = run.structured_ingest || null
                                    const localGraph = run.local_graph || null
                                    const localContext = run.local_context || null
                                    const reasoning = run.local_graph_reasoning || null
                                    const sourceArtifacts = run.source_artifacts || []

                                    return (
                                        <Box
                                            key={run.run_id || run.filename}
                                            sx={{
                                                mt: 1,
                                                pt: 1,
                                                borderTop: '1px solid',
                                                borderColor: 'divider',
                                            }}
                                        >
                                            <Typography variant="caption" sx={{ display: 'block', fontWeight: 700, color: 'text.primary' }}>
                                                {run.filename || 'Pipeline upload'}{run.run_id ? ` · ${run.run_id}` : ''}
                                            </Typography>
                                            {artifacts.length > 0 && (
                                                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                                                    {artifacts.slice(0, 8).map((artifact) => (
                                                        <Tooltip key={`${run.run_id}-${artifact.kind}`} title={artifact.endpoint || artifact.content_type || artifact.label || ''}>
                                                            <Chip
                                                                label={artifact.label || artifact.kind || 'Artifact'}
                                                                size="small"
                                                                variant="outlined"
                                                                sx={{ height: 20, fontSize: '0.65rem' }}
                                                            />
                                                        </Tooltip>
                                                    ))}
                                                </Box>
                                            )}
                                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mt: 0.75 }}>
                                                {structured && (
                                                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                                        Structured: {structured.filename || 'Markdown'} ({(structured.char_count || 0).toLocaleString()} chars)
                                                    </Typography>
                                                )}
                                                {localGraph && (
                                                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                                        Graph: {localGraph.node_count || 0} nodes / {localGraph.edge_count || 0} edges
                                                    </Typography>
                                                )}
                                                {localContext && (
                                                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                                        Context: {localContext.chunk_count || 0} chunks, {localContext.graph_context_relationship_count || 0} edges, {localContext.graph_path_context_count || 0} paths{localContext.evidence_card_count ? `, ${localContext.evidence_card_count} evidence cards` : ''}
                                                    </Typography>
                                                )}
                                                {reasoning && (
                                                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                                        Reasoning: {reasoning.bridge_entity_count || 0} bridges / {reasoning.path_candidate_count || 0} paths
                                                    </Typography>
                                                )}
                                            </Box>
                                            {structured?.excerpt && (
                                                <Box
                                                    component="pre"
                                                    sx={{
                                                        mt: 0.75,
                                                        mb: 0,
                                                        p: 0.75,
                                                        maxHeight: 120,
                                                        overflow: 'auto',
                                                        borderRadius: 1,
                                                        bgcolor: 'rgba(0,0,0,0.12)',
                                                        color: 'text.secondary',
                                                        fontSize: '0.68rem',
                                                        lineHeight: 1.45,
                                                        whiteSpace: 'pre-wrap',
                                                        wordBreak: 'break-word',
                                                    }}
                                                >
                                                    {structured.excerpt}
                                                </Box>
                                            )}
                                            {sourceArtifacts.length > 0 && (
                                                <Typography variant="caption" sx={{ display: 'block', mt: 0.75, color: 'text.secondary' }}>
                                                    Source artifacts: {sourceArtifacts.slice(0, 4).map((artifact) => `${artifact.kind || 'artifact'} ${artifact.filename || ''}`.trim()).filter(Boolean).join(', ')}
                                                </Typography>
                                            )}
                                        </Box>
                                    )
                                })}
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

                        {hasGraphOverview && (
                            <Box
                                sx={{
                                    mb: 2,
                                    p: 1,
                                    border: '1px solid',
                                    borderColor: 'rgba(217,119,6,0.26)',
                                    bgcolor: 'rgba(217,119,6,0.04)',
                                    borderRadius: 1,
                                }}
                            >
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 0.75 }}>
                                    <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', display: 'block' }}>
                                        Local Pipeline Graph Overview
                                    </Typography>
                                    <Chip
                                        label={`${graphOverview?.source_count || localGraphSources.length || 0} trace${(graphOverview?.source_count || localGraphSources.length || 0) === 1 ? '' : 's'}`}
                                        size="small"
                                        variant="outlined"
                                        sx={{ height: 20, fontSize: '0.65rem' }}
                                    />
                                    <Chip
                                        label={`${graphOverview?.relationship_count || overviewRelationships.length} relationship${(graphOverview?.relationship_count || overviewRelationships.length) === 1 ? '' : 's'}`}
                                        size="small"
                                        variant="outlined"
                                        sx={{ height: 20, fontSize: '0.65rem' }}
                                    />
                                    {(graphOverview?.bridge_entity_count || overviewBridgeEntities.length) > 0 && (
                                        <Chip
                                            label={`${graphOverview?.bridge_entity_count || overviewBridgeEntities.length} bridge${(graphOverview?.bridge_entity_count || overviewBridgeEntities.length) === 1 ? '' : 's'}`}
                                            size="small"
                                            variant="outlined"
                                            sx={{ height: 20, fontSize: '0.65rem' }}
                                        />
                                    )}
                                    {(graphOverview?.cross_document_path_count || overviewPaths.length) > 0 && (
                                        <Chip
                                            label={`${graphOverview?.cross_document_path_count || overviewPaths.length} cross-doc path${(graphOverview?.cross_document_path_count || overviewPaths.length) === 1 ? '' : 's'}`}
                                            size="small"
                                            variant="outlined"
                                            sx={{ height: 20, fontSize: '0.65rem' }}
                                        />
                                    )}
                                </Box>

                                {overviewSharedEntities.length > 0 && (
                                    <Box sx={{ mb: 0.75 }}>
                                        <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', mb: 0.35, fontSize: '0.7rem' }}>
                                            Shared entities
                                        </Typography>
                                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.35 }}>
                                            {overviewSharedEntities.slice(0, 8).map((entity, idx) => (
                                                <Tooltip
                                                    key={`${entity.name || 'entity'}-${idx}`}
                                                    title={[
                                                        ...(entity.filenames || []).slice(0, 4),
                                                        ...(entity.relationships || []).slice(0, 4).map(relationship => formatOverviewLabel(relationship)),
                                                    ].join(' | ')}
                                                >
                                                    <Chip
                                                        label={`${entity.name || 'Entity'}${entity.run_count ? ` (${entity.run_count})` : ''}`}
                                                        size="small"
                                                        sx={{ height: 20, fontSize: '0.65rem', bgcolor: 'rgba(22,163,74,0.1)' }}
                                                    />
                                                </Tooltip>
                                            ))}
                                        </Box>
                                    </Box>
                                )}

                                {overviewBridgeEntities.length > 0 && (
                                    <Box sx={{ mb: 0.75 }}>
                                        <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', mb: 0.35, fontSize: '0.7rem' }}>
                                            Bridge entities
                                        </Typography>
                                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.35 }}>
                                            {overviewBridgeEntities.slice(0, 8).map((bridge, idx) => (
                                                <Tooltip
                                                    key={`${bridge.name || 'bridge'}-${idx}`}
                                                    title={[
                                                        ...(bridge.filenames || []).slice(0, 4),
                                                        ...(bridge.relationships || []).slice(0, 4).map(relationship => formatOverviewLabel(relationship)),
                                                    ].join(' | ')}
                                                >
                                                    <Chip
                                                        label={`${bridge.name || 'Bridge'}${bridge.relationship_count ? ` (${bridge.relationship_count} links)` : ''}`}
                                                        size="small"
                                                        sx={{ height: 20, fontSize: '0.65rem', bgcolor: 'rgba(14,165,233,0.1)' }}
                                                    />
                                                </Tooltip>
                                            ))}
                                        </Box>
                                    </Box>
                                )}

                                {overviewPaths.length > 0 && (
                                    <Box sx={{ mb: 0.75 }}>
                                        <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', mb: 0.35, fontSize: '0.7rem' }}>
                                            Cross-document paths
                                        </Typography>
                                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.45 }}>
                                            {overviewPaths.slice(0, 4).map((path, idx) => (
                                                <Box
                                                    key={`${path.source || 'source'}-${path.bridge || 'bridge'}-${path.target || 'target'}-${idx}`}
                                                    sx={{
                                                        px: 0.75,
                                                        py: 0.6,
                                                        border: '1px solid rgba(14,165,233,0.18)',
                                                        bgcolor: 'rgba(14,165,233,0.05)',
                                                        borderRadius: 1,
                                                    }}
                                                >
                                                    <Typography variant="caption" sx={{ display: 'block', fontWeight: 600, fontSize: '0.7rem' }}>
                                                        {path.text || `${path.source || 'Source'} -> ${path.bridge || 'Bridge'} -> ${path.target || 'Target'}`}
                                                    </Typography>
                                                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontSize: '0.65rem' }}>
                                                        {(path.filenames || []).slice(0, 3).join(' + ')}
                                                        {typeof path.score === 'number' ? ` · score ${path.score.toFixed(2)}` : ''}
                                                    </Typography>
                                                </Box>
                                            ))}
                                        </Box>
                                    </Box>
                                )}

                                {relationshipTypeCounts.length > 0 && (
                                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.35, mb: 0.75 }}>
                                        {relationshipTypeCounts.slice(0, 6).map(([type, count]) => (
                                            <Chip
                                                key={type}
                                                label={`${formatOverviewLabel(type)} ${count}`}
                                                size="small"
                                                sx={{ height: 20, fontSize: '0.62rem', bgcolor: 'rgba(217,119,6,0.1)' }}
                                            />
                                        ))}
                                    </Box>
                                )}

                                {overviewRelationships.length > 0 && (
                                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.6 }}>
                                        {overviewRelationships.slice(0, 6).map((relationship, idx) => {
                                            const provenance = overviewRelationshipProvenance(relationship)
                                            return (
                                                <Box key={`${relationship.source || 'source'}-${relationship.target || 'target'}-${idx}`} sx={{ minWidth: 0 }}>
                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap' }}>
                                                        <Typography variant="caption" sx={{ color: 'text.primary', fontWeight: 600, fontSize: '0.7rem', overflowWrap: 'anywhere' }}>
                                                            {relationship.source || 'Unknown source'}
                                                        </Typography>
                                                        <ArrowIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
                                                        <Chip
                                                            label={formatOverviewLabel(relationship.relation || 'related_to')}
                                                            size="small"
                                                            sx={{ height: 20, fontSize: '0.62rem', maxWidth: '100%' }}
                                                        />
                                                        <ArrowIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
                                                        <Typography variant="caption" sx={{ color: 'text.primary', fontWeight: 600, fontSize: '0.7rem', overflowWrap: 'anywhere' }}>
                                                            {relationship.target || 'Unknown target'}
                                                        </Typography>
                                                        {relationship.relationship_type && (
                                                            <Chip
                                                                label={relationship.relationship_type === 'layout' ? 'layout' : 'semantic'}
                                                                size="small"
                                                                variant="outlined"
                                                                color={relationship.relationship_type === 'layout' ? 'success' : 'default'}
                                                                sx={{ height: 20, fontSize: '0.62rem' }}
                                                            />
                                                        )}
                                                        {typeof relationship.score === 'number' && (
                                                            <Chip
                                                                label={`score ${relationship.score.toFixed(2)}`}
                                                                size="small"
                                                                variant="outlined"
                                                                sx={{ height: 20, fontSize: '0.62rem' }}
                                                            />
                                                        )}
                                                    </Box>
                                                    {provenance && (
                                                        <Typography variant="caption" sx={{ display: 'block', color: 'text.disabled', pl: 0.5, fontSize: '0.65rem', overflowWrap: 'anywhere' }}>
                                                            {provenance}
                                                        </Typography>
                                                    )}
                                                    {(relationship.matched_query_terms || []).length > 0 && (
                                                        <Typography variant="caption" sx={{ display: 'block', color: 'text.disabled', pl: 0.5, fontSize: '0.65rem' }}>
                                                            matched: {(relationship.matched_query_terms || []).slice(0, 5).join(', ')}
                                                        </Typography>
                                                    )}
                                                    {relationship.evidence && (
                                                        <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', pl: 0.5, fontSize: '0.65rem', lineHeight: 1.35, overflowWrap: 'anywhere' }}>
                                                            {relationship.evidence.length > 180 ? relationship.evidence.slice(0, 180) + '...' : relationship.evidence}
                                                        </Typography>
                                                    )}
                                                </Box>
                                            )
                                        })}
                                    </Box>
                                )}
                            </Box>
                        )}

                        {hasGraphSummary && (
                            <Box
                                sx={{
                                    mb: 2,
                                    p: 1,
                                    border: '1px solid',
                                    borderColor: 'rgba(59,130,246,0.24)',
                                    bgcolor: 'rgba(59,130,246,0.04)',
                                    borderRadius: 1,
                                }}
                            >
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 0.5 }}>
                                    <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', display: 'block' }}>
                                        Graph Reasoning Summary
                                    </Typography>
                                    {graphSummary?.mode && (
                                        <Chip
                                            label={graphSummary.mode.replace(/_/g, ' ')}
                                            size="small"
                                            variant="outlined"
                                            sx={{ height: 20, fontSize: '0.65rem' }}
                                        />
                                    )}
                                </Box>
                                {graphSummary?.summary_text && (
                                    <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', lineHeight: 1.45, mb: 0.5 }}>
                                        {graphSummary.summary_text}
                                    </Typography>
                                )}
                                {graphSummary?.answer_guidance && (
                                    <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', lineHeight: 1.45, mb: 0.5 }}>
                                        {graphSummary.answer_guidance}
                                    </Typography>
                                )}
                                {(graphSummary?.evidence || []).slice(0, 3).map((item, idx) => (
                                    <Typography key={idx} variant="caption" sx={{ display: 'block', color: 'text.secondary', pl: 1, fontSize: '0.7rem' }}>
                                        - {item}
                                    </Typography>
                                ))}
                                {(graphSummary?.evidence_terms || []).length > 0 && (
                                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.3, mt: 0.75 }}>
                                        {(graphSummary?.evidence_terms || []).slice(0, 8).map((term, idx) => (
                                            <Chip
                                                key={`${term}-${idx}`}
                                                label={term}
                                                size="small"
                                                sx={{ height: 18, fontSize: '0.6rem', bgcolor: 'rgba(59,130,246,0.1)' }}
                                            />
                                        ))}
                                    </Box>
                                )}
                            </Box>
                        )}

                        {hasGraphAnswerPlan && (
                            <Box
                                sx={{
                                    mb: 2,
                                    p: 1,
                                    border: '1px solid',
                                    borderColor: 'rgba(37,99,235,0.26)',
                                    bgcolor: 'rgba(37,99,235,0.04)',
                                    borderRadius: 1,
                                }}
                            >
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 0.75 }}>
                                    <HubIcon sx={{ fontSize: 14, color: 'text.secondary' }} />
                                    <Typography variant="caption" sx={{ fontWeight: 700, color: 'text.secondary', display: 'block' }}>
                                        Graph Answer Plan
                                    </Typography>
                                    {graphAnswerPlan?.strategy && (
                                        <Chip
                                            label={formatOverviewLabel(graphAnswerPlan.strategy)}
                                            size="small"
                                            color="primary"
                                            variant="outlined"
                                            sx={{ height: 20, fontSize: '0.65rem' }}
                                        />
                                    )}
                                    {graphAnswerHops.length > 0 && (
                                        <Chip
                                            label={`${graphAnswerHops.length} hop${graphAnswerHops.length === 1 ? '' : 's'}`}
                                            size="small"
                                            variant="outlined"
                                            sx={{ height: 20, fontSize: '0.65rem' }}
                                        />
                                    )}
                                    {graphAnswerEvidenceCards.length > 0 && (
                                        <Chip
                                            label={`${graphAnswerEvidenceCards.length} evidence card${graphAnswerEvidenceCards.length === 1 ? '' : 's'}`}
                                            size="small"
                                            variant="outlined"
                                            sx={{ height: 20, fontSize: '0.65rem' }}
                                        />
                                    )}
                                </Box>

                                {graphAnswerPlan?.instruction && (
                                    <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', mb: 0.6, fontSize: '0.7rem', lineHeight: 1.45 }}>
                                        {graphAnswerPlan.instruction}
                                    </Typography>
                                )}

                                {graphPrimaryPath && (
                                    <Box sx={{ mb: 0.8 }}>
                                        <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', fontWeight: 700, mb: 0.35 }}>
                                            Primary path
                                        </Typography>
                                        <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', fontSize: '0.7rem', lineHeight: 1.4, overflowWrap: 'anywhere' }}>
                                            {graphPrimaryPath.text || (Array.isArray(graphPrimaryPath.nodes) && graphPrimaryPath.nodes.length > 0 ? graphPrimaryPath.nodes.join(' -> ') : `${graphPrimaryPath.source || 'Source'} -> ${graphPrimaryPath.target || 'Target'}`)}
                                        </Typography>
                                        {graphPrimaryPath.explanation && (
                                            <Typography variant="caption" sx={{ display: 'block', color: 'text.primary', fontSize: '0.72rem', lineHeight: 1.45, mt: 0.35, overflowWrap: 'anywhere' }}>
                                                {graphPrimaryPath.explanation}
                                            </Typography>
                                        )}
                                        {graphPrimaryPath.answer_hint && (
                                            <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', fontSize: '0.66rem', lineHeight: 1.35, mt: 0.25, overflowWrap: 'anywhere' }}>
                                                {graphPrimaryPath.answer_hint}
                                            </Typography>
                                        )}
                                        {formatEvidenceFamilies(graphPrimaryPath.evidence_families).length > 0 && (
                                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.35, mt: 0.5 }}>
                                                {formatEvidenceFamilies(graphPrimaryPath.evidence_families).map(family => (
                                                    <Chip
                                                        key={family}
                                                        label={family}
                                                        size="small"
                                                        color="info"
                                                        variant="outlined"
                                                        sx={{ height: 18, fontSize: '0.6rem' }}
                                                    />
                                                ))}
                                            </Box>
                                        )}
                                    </Box>
                                )}

                                {graphAnswerHops.length > 0 && (
                                    <Box sx={{ mb: 0.8 }}>
                                        <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', fontWeight: 700, mb: 0.35 }}>
                                            Hop rationales
                                        </Typography>
                                        {graphAnswerHops.slice(0, 4).map((hop, idx) => (
                                            <Box key={`${hop.source || 'source'}-${hop.target || 'target'}-${idx}`} sx={{ mb: idx < Math.min(graphAnswerHops.length, 4) - 1 ? 0.45 : 0 }}>
                                                <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', fontSize: '0.68rem', lineHeight: 1.35, overflowWrap: 'anywhere' }}>
                                                    {hop.index || idx + 1}. {hop.source || 'Source'} --[{hop.relation || 'related_to'}]-- {hop.target || 'Target'}
                                                </Typography>
                                                {hop.rationale && (
                                                    <Typography variant="caption" sx={{ display: 'block', color: 'text.disabled', fontSize: '0.62rem', lineHeight: 1.3, overflowWrap: 'anywhere' }}>
                                                        {hop.rationale}
                                                    </Typography>
                                                )}
                                            </Box>
                                        ))}
                                    </Box>
                                )}

                                {graphAnswerEvidenceCards.length > 0 && (
                                    <Box>
                                        <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', fontWeight: 700, mb: 0.35 }}>
                                            Evidence checklist
                                        </Typography>
                                        {graphAnswerEvidenceCards.slice(0, 4).map((card, idx) => {
                                            const citation = card.citation_label || card.provenance || formatGraphRAGReasoningLocation(card.location)
                                            return (
                                                <Box key={`${card.id || card.role || 'evidence'}-${idx}`} sx={{ mb: idx < Math.min(graphAnswerEvidenceCards.length, 4) - 1 ? 0.45 : 0 }}>
                                                    <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', fontSize: '0.67rem', lineHeight: 1.35, overflowWrap: 'anywhere' }}>
                                                        {card.source && card.target
                                                            ? `${card.source} --[${card.relation || 'related_to'}]-- ${card.target}`
                                                            : card.text || card.role || 'Graph evidence'}
                                                    </Typography>
                                                    {citation && (
                                                        <Typography variant="caption" sx={{ display: 'block', color: 'text.disabled', fontSize: '0.61rem', lineHeight: 1.3, overflowWrap: 'anywhere' }}>
                                                            cite: {citation}
                                                        </Typography>
                                                    )}
                                                </Box>
                                            )
                                        })}
                                    </Box>
                                )}
                            </Box>
                        )}

                        {graphBridgeEntities.length > 0 && (
                            <Box sx={{ mb: 2 }}>
                                <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', mb: 0.5, display: 'block' }}>
                                    Bridge Entities
                                </Typography>
                                {graphBridgeEntities.slice(0, 5).map((bridge, idx) => (
                                    <Paper
                                        key={`${bridge.name}-${idx}`}
                                        elevation={0}
                                        sx={{
                                            p: 1,
                                            mb: 0.75,
                                            border: '1px solid',
                                            borderColor: 'rgba(13,148,136,0.26)',
                                            bgcolor: 'rgba(13,148,136,0.04)',
                                            borderRadius: 1,
                                        }}
                                    >
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 0.35 }}>
                                            <Chip label={bridge.name} size="small" sx={{ height: 20, fontSize: '0.65rem' }} />
                                            <Chip
                                                label={`${bridge.path_count} path${bridge.path_count === 1 ? '' : 's'}`}
                                                size="small"
                                                variant="outlined"
                                                sx={{ height: 20, fontSize: '0.65rem' }}
                                            />
                                            {typeof bridge.score === 'number' && (
                                                <Chip
                                                    label={`score ${bridge.score.toFixed(2)}`}
                                                    size="small"
                                                    variant="outlined"
                                                    sx={{ height: 20, fontSize: '0.65rem' }}
                                                />
                                            )}
                                        </Box>
                                        {bridge.text && (
                                            <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', fontSize: '0.7rem', lineHeight: 1.4 }}>
                                                {bridge.text}
                                            </Typography>
                                        )}
                                        {(bridge.relationships || []).slice(0, 3).map((relationship, relIdx) => (
                                            <Typography key={relIdx} variant="caption" sx={{ display: 'block', color: 'text.secondary', pl: 0.5, fontSize: '0.68rem' }}>
                                                {relationship}
                                            </Typography>
                                        ))}
                                    </Paper>
                                ))}
                            </Box>
                        )}

                        {/* Graph reasoning paths */}
                        {graphPaths.length > 0 && (
                            <Box sx={{ mb: 2 }}>
                                <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', mb: 0.5, display: 'block' }}>
                                    Knowledge Graph Paths
                                </Typography>
                                {graphPaths.slice(0, 4).map((path, idx) => (
                                    <Paper
                                        key={`${path.source}-${path.target}-${idx}`}
                                        elevation={0}
                                        sx={{
                                            p: 1,
                                            mb: 0.75,
                                            border: '1px solid',
                                            borderColor: 'rgba(16,185,129,0.28)',
                                            bgcolor: 'rgba(16,185,129,0.04)',
                                            borderRadius: 1,
                                        }}
                                    >
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 0.5 }}>
                                            <Chip label={path.source} size="small" sx={{ height: 20, fontSize: '0.65rem' }} />
                                            <ArrowIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
                                            <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.65rem' }}>
                                                {path.hops} hop{path.hops === 1 ? '' : 's'}
                                            </Typography>
                                            {typeof path.score === 'number' && (
                                                <Chip
                                                    label={`score ${path.score.toFixed(2)}`}
                                                    size="small"
                                                    variant="outlined"
                                                    sx={{ height: 20, fontSize: '0.65rem' }}
                                                />
                                            )}
                                            {path.path_type && (
                                                <Chip
                                                    label={path.path_type === 'layout' ? 'layout path' : path.path_type}
                                                    size="small"
                                                    variant="outlined"
                                                    color={path.path_type === 'layout' ? 'success' : 'default'}
                                                    sx={{ height: 20, fontSize: '0.65rem' }}
                                                />
                                            )}
                                            <ArrowIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
                                            <Chip label={path.target} size="small" sx={{ height: 20, fontSize: '0.65rem' }} />
                                        </Box>
                                        {path.matched_query_terms && path.matched_query_terms.length > 0 && (
                                            <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', pl: 0.5, mb: 0.25, fontSize: '0.65rem' }}>
                                                matched: {path.matched_query_terms.slice(0, 5).join(', ')}
                                            </Typography>
                                        )}
                                        {path.explanation && (
                                            <Typography variant="caption" sx={{ display: 'block', color: 'text.primary', pl: 0.5, mb: 0.25, fontSize: '0.7rem', lineHeight: 1.4, overflowWrap: 'anywhere' }}>
                                                {path.explanation}
                                            </Typography>
                                        )}
                                        {formatEvidenceFamilies(path.evidence_families).length > 0 && (
                                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.3, pl: 0.5, mb: 0.45 }}>
                                                {formatEvidenceFamilies(path.evidence_families).map(family => (
                                                    <Chip
                                                        key={family}
                                                        label={family}
                                                        size="small"
                                                        variant="outlined"
                                                        sx={{ height: 18, fontSize: '0.6rem' }}
                                                    />
                                                ))}
                                            </Box>
                                        )}
                                        {path.relationships.slice(0, 5).map((rel, relIdx) => (
                                            <Box key={relIdx} sx={{ pl: 0.5, mb: rel.provenance ? 0.25 : 0 }}>
                                                <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', fontSize: '0.7rem' }}>
                                                    {rel.source} --[{rel.relation}]-- {rel.target}
                                                </Typography>
                                                {rel.provenance && (
                                                    <Typography variant="caption" sx={{ display: 'block', color: 'text.disabled', fontSize: '0.65rem' }}>
                                                        {rel.provenance}
                                                    </Typography>
                                                )}
                                                {rel.rationale && (
                                                    <Typography variant="caption" sx={{ display: 'block', color: 'text.disabled', fontSize: '0.62rem' }}>
                                                        {rel.rationale}
                                                    </Typography>
                                                )}
                                            </Box>
                                        ))}
                                    </Paper>
                                ))}
                            </Box>
                        )}

                        {directGraphRelationships.length > 0 && (
                            <Box sx={{ mb: 2 }}>
                                <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', mb: 0.5, display: 'block' }}>
                                    Direct Graph Relationships
                                </Typography>
                                {directGraphRelationships.slice(0, 6).map((rel, idx) => (
                                    <Box key={idx} sx={{ pl: 1, mb: rel.provenance ? 0.25 : 0 }}>
                                        <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', fontSize: '0.7rem' }}>
                                            {rel.source} --[{rel.relation}]-- {rel.target}
                                        </Typography>
                                        {rel.provenance && (
                                            <Typography variant="caption" sx={{ display: 'block', color: 'text.disabled', fontSize: '0.65rem' }}>
                                                {rel.provenance}
                                            </Typography>
                                        )}
                                        {rel.matched_query_terms && rel.matched_query_terms.length > 0 && (
                                            <Typography variant="caption" sx={{ display: 'block', color: 'text.disabled', fontSize: '0.65rem' }}>
                                                matched: {rel.matched_query_terms.slice(0, 5).join(', ')}
                                            </Typography>
                                        )}
                                    </Box>
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
                                        borderColor: chunk.matched_graph_path_terms?.length
                                            ? 'rgba(20,184,166,0.34)'
                                            : chunk.retrieval_reasons?.includes('local_pipeline_evidence_context')
                                                ? 'rgba(236,72,153,0.3)'
                                            : chunk.matched_entities?.length
                                                ? 'rgba(99,102,241,0.3)'
                                                : 'divider',
                                        borderRadius: 1,
                                        bgcolor: chunk.matched_graph_path_terms?.length
                                            ? 'rgba(20,184,166,0.04)'
                                            : chunk.retrieval_reasons?.includes('local_pipeline_evidence_context')
                                                ? 'rgba(236,72,153,0.035)'
                                            : chunk.matched_entities?.length
                                                ? 'rgba(99,102,241,0.03)'
                                                : 'transparent',
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
                                                        +entity
                                                    </Typography>
                                                </Tooltip>
                                            )}
                                            {chunk.graph_reasoning_boost !== undefined && chunk.graph_reasoning_boost > 0 && (
                                                <Tooltip title={`Graph path boost: +${(chunk.graph_reasoning_boost * 100).toFixed(0)}%`}>
                                                    <Typography variant="caption" sx={{ fontSize: '0.6rem', color: '#0F766E' }}>
                                                        +path
                                                    </Typography>
                                                </Tooltip>
                                            )}
                                            {chunk.retrieval_reasons?.includes('graph_augmented') && (
                                                <Tooltip title="Retrieved by a query expanded with graph path terms">
                                                    <Typography variant="caption" sx={{ fontSize: '0.6rem', color: '#0369A1' }}>
                                                        +graph retrieval
                                                    </Typography>
                                                </Tooltip>
                                            )}
                                            {chunk.retrieval_reasons?.includes('local_pipeline_graph_path_context') && (
                                                <Tooltip title="Retrieved because local pipeline graph paths matched the query">
                                                    <Typography variant="caption" sx={{ fontSize: '0.6rem', color: '#0F766E' }}>
                                                        +local path
                                                    </Typography>
                                                </Tooltip>
                                            )}
                                            {chunk.retrieval_reasons?.includes('local_pipeline_evidence_context') && (
                                                <Tooltip title={`Retrieved because extracted evidence families matched this passage: ${chunkPipelineEvidenceTerms(chunk).slice(0, 6).join(', ') || 'structured evidence'}`}>
                                                    <Typography variant="caption" sx={{ fontSize: '0.6rem', color: '#BE185D' }}>
                                                        +evidence
                                                    </Typography>
                                                </Tooltip>
                                            )}
                                        </Box>
                                    </Box>
                                    <Typography variant="body2" sx={{ fontSize: '0.8rem', lineHeight: 1.5, color: 'text.primary' }}>
                                        {chunk.content.length > 300 ? chunk.content.slice(0, 300) + '...' : chunk.content}
                                    </Typography>
                                    {chunk.graph_context && chunk.graph_context.length > 0 && (
                                        <Box
                                            sx={{
                                                mt: 0.75,
                                                p: 0.75,
                                                border: '1px solid',
                                                borderColor: 'rgba(13,148,136,0.22)',
                                                bgcolor: 'rgba(13,148,136,0.035)',
                                                borderRadius: 1,
                                            }}
                                        >
                                            <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', fontWeight: 700, mb: 0.35 }}>
                                                Local graph relationships
                                            </Typography>
                                            {chunk.graph_context.slice(0, 3).map((relationship, graphIdx) => (
                                                <Typography
                                                    key={`${relationship.source || 'source'}-${relationship.target || 'target'}-${graphIdx}`}
                                                    variant="caption"
                                                    sx={{ display: 'block', color: 'text.secondary', fontSize: '0.66rem', lineHeight: 1.35, overflowWrap: 'anywhere' }}
                                                >
                                                    {formatChunkGraphRelationship(relationship)}
                                                </Typography>
                                            ))}
                                        </Box>
                                    )}
                                    {chunk.graph_paths && chunk.graph_paths.length > 0 && (
                                        <Box
                                            sx={{
                                                mt: 0.75,
                                                p: 0.75,
                                                border: '1px solid',
                                                borderColor: 'rgba(20,184,166,0.22)',
                                                bgcolor: 'rgba(20,184,166,0.035)',
                                                borderRadius: 1,
                                            }}
                                        >
                                            <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', fontWeight: 700, mb: 0.35 }}>
                                                Local graph paths
                                            </Typography>
                                            {chunk.graph_paths.slice(0, 3).map((path, pathIdx) => {
                                                const score = path.chunk_match_score ?? path.score
                                                const detailBits = [
                                                    path.hops !== undefined ? `${path.hops} hop${path.hops === 1 ? '' : 's'}` : '',
                                                    score !== undefined ? `score ${Number(score).toFixed(2)}` : '',
                                                    ...(path.chunk_match_reasons || []).slice(0, 2),
                                                ].filter(Boolean)

                                                return (
                                                    <Box key={`${formatChunkGraphPath(path)}-${pathIdx}`} sx={{ mb: pathIdx < Math.min(chunk.graph_paths?.length || 0, 3) - 1 ? 0.45 : 0 }}>
                                                        <Typography
                                                            variant="caption"
                                                            sx={{ display: 'block', color: 'text.secondary', fontSize: '0.66rem', lineHeight: 1.35, overflowWrap: 'anywhere' }}
                                                        >
                                                            {formatChunkGraphPath(path)}
                                                        </Typography>
                                                        {detailBits.length > 0 && (
                                                            <Typography
                                                                variant="caption"
                                                                sx={{ display: 'block', color: 'text.disabled', fontSize: '0.6rem', lineHeight: 1.3, overflowWrap: 'anywhere' }}
                                                            >
                                                                {detailBits.join(' | ')}
                                                            </Typography>
                                                        )}
                                                        {path.explanation && (
                                                            <Typography
                                                                variant="caption"
                                                                sx={{ display: 'block', color: 'text.secondary', fontSize: '0.62rem', lineHeight: 1.35, overflowWrap: 'anywhere', mt: 0.15 }}
                                                            >
                                                                {path.explanation}
                                                            </Typography>
                                                        )}
                                                        {formatEvidenceFamilies(path.evidence_families).length > 0 && (
                                                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.25, mt: 0.25 }}>
                                                                {formatEvidenceFamilies(path.evidence_families).slice(0, 5).map(family => (
                                                                    <Chip
                                                                        key={family}
                                                                        label={family}
                                                                        size="small"
                                                                        variant="outlined"
                                                                        sx={{ height: 16, fontSize: '0.55rem' }}
                                                                    />
                                                                ))}
                                                            </Box>
                                                        )}
                                                    </Box>
                                                )
                                            })}
                                        </Box>
                                    )}
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
                                    {chunk.matched_graph_path_terms && chunk.matched_graph_path_terms.length > 0 && (
                                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.3, mt: 0.5 }}>
                                            {chunk.matched_graph_path_terms.slice(0, 5).map((term, i) => (
                                                <Chip
                                                    key={i}
                                                    label={term}
                                                    size="small"
                                                    sx={{ height: 18, fontSize: '0.6rem', bgcolor: 'rgba(20,184,166,0.12)' }}
                                                />
                                            ))}
                                        </Box>
                                    )}
                                    {chunkPipelineEvidenceTerms(chunk).length > 0 && (
                                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.3, mt: 0.5 }}>
                                            {chunkPipelineEvidenceTerms(chunk).slice(0, 6).map((term, i) => (
                                                <Chip
                                                    key={`${term}-${i}`}
                                                    label={term}
                                                    size="small"
                                                    sx={{ height: 18, fontSize: '0.6rem', bgcolor: 'rgba(236,72,153,0.12)' }}
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
