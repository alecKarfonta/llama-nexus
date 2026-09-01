import React, { useEffect, useMemo, useRef, useState } from 'react'
import * as pdfjsLib from 'pdfjs-dist'
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url'
import type { PDFDocumentProxy, RenderTask, TextItem } from 'pdfjs-dist/types/src/display/api'
import type { PageViewport } from 'pdfjs-dist/types/src/display/display_utils'
import {
    Alert,
    Box,
    Button,
    Chip,
    CircularProgress,
    Drawer,
    IconButton,
    Paper,
    Stack,
    TextField,
    Tooltip,
    ToggleButton,
    ToggleButtonGroup,
    Typography,
} from '@mui/material'
import { alpha, useTheme, type Theme } from '@mui/material/styles'
import {
    AccountTree as GraphIcon,
    AltRoute as PathIcon,
    Article as PassageIcon,
    Check as CheckIcon,
    ChevronLeft as PreviousIcon,
    ChevronRight as NextIcon,
    Close as CloseIcon,
    ContentCopy as CopyIcon,
    Description as DocumentIcon,
    OpenInNew as OpenIcon,
    PictureAsPdf as PdfIcon,
    Search as SearchIcon,
    ZoomIn as ZoomInIcon,
    ZoomOut as ZoomOutIcon,
} from '@mui/icons-material'
import { cleanDocumentTextForReading } from '@/utils/chatReferenceText'
import { buildChatReferenceDocumentReview } from '@/utils/chatReferenceDocumentText'
import { buildChatReferenceSourceBrief } from '@/utils/chatReferenceSourceBrief'
import {
    DEFAULT_REFERENCE_DRAWER_LIMIT,
    selectReferenceDrawerRows,
} from '@/utils/chatReferenceDrawerList'
import {
    STRUCTURAL_REVIEW_FAMILY_PATTERNS,
    isTechnicalReviewReference,
    normalizeReferenceFacetText as normalizeFacetText,
    preferredReviewReference,
    referenceAnswerFitScore,
    referenceReviewScore,
} from '@/utils/chatReferenceReviewScoring'
import {
    briefGraphRelationshipTitle,
    cleanGraphReferenceEvidenceText,
    cleanGraphReferenceEndpointLabel,
    cleanGraphReferenceLabel,
    formatGraphReferenceRelation,
    graphRelationshipEvidenceMapNodes,
    graphRelationshipEvidenceParts,
    graphRelationshipEvidenceLead,
} from '@/utils/graphragReferenceDisplay'
import {
    buildGraphRAGReferenceFilterChips,
    type GraphRAGReferenceFilterValue as ReferenceFilterValue,
} from '@/utils/graphragReferenceFilters'
import { displayPathNodes } from '@/utils/graphragPathDisplay'
import { pageNumberFromReference, pageNumbersFromReferences } from '@/utils/referencePdfPage'

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl

export type ChatReferenceKind = 'passage' | 'document' | 'relationship' | 'path' | 'artifact'
export type ChatReferenceSourceRole =
    | 'answer_evidence'
    | 'retrieved_passage'
    | 'reasoning_path'
    | 'source_document'
    | 'graph_relationship'

export interface ChatReferenceLocation {
    label: string
    value: string
    category?: string
}

export interface ChatReferenceItem {
    id: string
    kind: ChatReferenceKind
    sourceRole?: ChatReferenceSourceRole
    title: string
    subtitle?: string
    body?: string
    citationLabel?: string
    score?: number
    documentId?: string
    chunkIndex?: number
    endpoint?: string
    contentType?: string
    filename?: string
    runId?: string
    terms?: string[]
    evidenceFamilies?: string[]
    locationHints?: string[]
    sourceLocations?: ChatReferenceLocation[]
    matchedTerms?: string[]
    relationship?: {
        source?: string
        relation?: string
        target?: string
    }
    path?: {
        nodes?: string[]
        hops?: number
        score?: number
        relationships?: Array<{
            source?: string
            relation?: string
            target?: string
            evidence?: string
        }>
    }
}

interface ReferenceExplorerProps {
    references: ChatReferenceItem[]
    query?: string
    density?: 'cards' | 'tray'
}

interface ArtifactState {
    status: 'loading' | 'loaded' | 'error'
    text?: string
    contentType?: string
    error?: string
}

export interface PdfHighlightRect {
    left: number
    top: number
    width: number
    height: number
    label: string
    kind?: 'text' | 'location'
}

interface ReferenceDocumentGroup {
    key: string
    title: string
    subtitle?: string
    references: ChatReferenceItem[]
    counts: Partial<Record<ChatReferenceKind, number>>
    bestReference: ChatReferenceItem
    evidenceLabels: string[]
    locationLabels: string[]
}

const MAX_VISIBLE_REFERENCES = 4
const INLINE_TRAY_REFERENCE_LIMIT = 3
const DRAWER_SEARCH_REFERENCE_THRESHOLD = DEFAULT_REFERENCE_DRAWER_LIMIT + 15
const DRAWER_SEARCH_SOURCE_THRESHOLD = 2
const STOP_WORDS = new Set([
    'about',
    'after',
    'also',
    'and',
    'are',
    'because',
    'been',
    'between',
    'could',
    'from',
    'have',
    'into',
    'that',
    'their',
    'there',
    'these',
    'this',
    'were',
    'what',
    'when',
    'where',
    'which',
    'with',
    'would',
])

const HIGHLIGHT_STOP_TERMS = new Set([
    'chunk',
    'content',
    'document',
    'evidence',
    'extract',
    'extracted',
    'file',
    'focus',
    'general',
    'heading',
    'hidden',
    'image',
    'images',
    'layout',
    'line',
    'logo',
    'markdown',
    'media',
    'microsoft',
    'paragraph',
    'passage',
    'part',
    'powerpoint',
    'scope',
    'section',
    'semantic',
    'slide',
    'slides',
    'source',
    'structured',
    'text',
    'visibility',
])

const asRecord = (value: unknown): Record<string, unknown> =>
    value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}

const asString = (value: unknown): string =>
    typeof value === 'string' ? value.trim() : ''

const uniqueStrings = (values: Array<string | undefined | null>): string[] => {
    const seen = new Set<string>()
    const output: string[] = []

    values.forEach(value => {
        const normalized = String(value || '').trim()
        const key = normalized.toLowerCase()
        if (!normalized || seen.has(key)) return
        seen.add(key)
        output.push(normalized)
    })

    return output
}

const queryTerms = (query?: string): string[] =>
    uniqueStrings(
        String(query || '')
            .split(/[^A-Za-z0-9_-]+/)
            .map(term => term.trim())
            .filter(term => term.length >= 4 && !STOP_WORDS.has(term.toLowerCase()))
            .slice(0, 10),
    )

const LOW_VALUE_SOURCE_QUERY_TERMS = new Set([
    'about',
    'answer',
    'citation',
    'cite',
    'deck',
    'document',
    'evidence',
    'extracted',
    'source',
    'sources',
    'strongest',
    'summarize',
    'summary',
])

const escapeRegExp = (value: string): string =>
    value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

const highlightPattern = (term: string): string => {
    const escaped = escapeRegExp(term)
    return /^[A-Za-z0-9_-]+$/.test(term) ? `\\b${escaped}\\b` : escaped
}

const normalizeEndpoint = (endpoint?: string): string => {
    const value = String(endpoint || '').trim()
    if (!value) return ''
    if (/^https?:\/\//i.test(value)) return value
    return value.startsWith('/') ? value : `/${value}`
}

const referenceBasename = (value: string): string =>
    value.split(/[\\/]/).filter(Boolean).pop() || value

const QUERY_FILENAME_PATTERN =
    /\b[\w.-]+\.(?:csv|docx?|html?|json|md|pdf|pptx?|txt|xlsx?|xml)\b/gi
const REFERENCE_FILENAME_LABEL_PATTERN =
    /\b[\w.-]+\.(?:csv|docx?|html?|json|md|pdf|pptx?|txt|xlsx?|xml)\b/i

const normalizeFilenameMatch = (value: string): string =>
    referenceBasename(value)
        .replace(/[)\],.;:!?]+$/g, '')
        .trim()
        .toLowerCase()

const queryFilenames = (query?: string): string[] =>
    uniqueStrings(
        Array.from(String(query || '').matchAll(QUERY_FILENAME_PATTERN))
            .map(match => normalizeFilenameMatch(match[0]))
            .filter(Boolean),
    )

const referenceFilenameCandidates = (reference: ChatReferenceItem): string[] =>
    uniqueStrings([
        reference.filename,
        reference.documentId,
        reference.title,
        reference.subtitle,
        ...(reference.sourceLocations || []).flatMap(location => [
            location.value,
            location.label,
        ]),
        ...(reference.locationHints || []),
    ])
        .map(normalizeFilenameMatch)
        .filter(Boolean)

const isFilenameLikeReferenceLabel = (value?: string): boolean =>
    REFERENCE_FILENAME_LABEL_PATTERN.test(String(value || '').trim())

const isSyntheticFilenameRelationshipReference = (
    reference?: ChatReferenceItem | null,
): boolean => {
    if (!reference || reference.kind !== 'relationship') return false
    const source = reference.relationship?.source || ''
    const target = reference.relationship?.target || ''
    return isFilenameLikeReferenceLabel(source) && isFilenameLikeReferenceLabel(target)
}

const matchingQueryFilename = (
    reference: ChatReferenceItem,
    filenames: string[],
): string => {
    if (!filenames.length) return ''
    const candidates = referenceFilenameCandidates(reference)
    return filenames.find(filename =>
        candidates.some(candidate =>
            candidate === filename ||
            candidate.endsWith(`/${filename}`) ||
            candidate.includes(filename),
        ),
    ) || ''
}

const namedFileContextScore = (
    reference: ChatReferenceItem,
    query?: string,
): number => {
    const filenames = queryFilenames(query)
    const filename = matchingQueryFilename(reference, filenames)
    if (!filename) return 0
    const haystack = referenceSearchText(reference)
    const matchingLines = String(query || '')
        .split(/\n+/)
        .map(line => line.trim())
        .filter(line => normalizeFilenameMatch(line).includes(filename))
    const contextText = matchingLines.join(' ')
    if (!contextText) return 0
    return queryTerms(contextText)
        .filter(term => !LOW_VALUE_SOURCE_QUERY_TERMS.has(term.toLowerCase()))
        .reduce((score, term) => (
            haystack.includes(term.toLowerCase()) ? score + 1 : score
        ), 0)
}

const namedFileConfigTokenScore = (
    reference: ChatReferenceItem,
    query?: string,
): number => {
    const haystack = uniqueStrings([
        reference.title,
        reference.subtitle,
        reference.body,
        reference.filename,
        reference.documentId,
        reference.citationLabel,
        ...(reference.terms || []),
        ...(reference.matchedTerms || []),
        ...(reference.locationHints || []),
    ]).join(' ').toLowerCase()
    return uniqueStrings(
        String(query || '').match(/\b[A-Z][A-Z0-9_]{2,}\b/g) || [],
    )
        .filter(token => /[_0-9]/.test(token))
        .reduce((score, token) => (
        haystack.includes(token.toLowerCase()) ? score + 1 : score
    ), 0)
}

const namedFileReferenceScore = (
    reference: ChatReferenceItem,
    query?: string,
): number => {
    const kindScore: Record<ChatReferenceKind, number> = {
        passage: 64,
        document: 36,
        relationship: 0,
        path: 1,
        artifact: 1,
    }
    return (
        (reference.sourceRole === 'answer_evidence' ? 8 : 0) +
        (reference.sourceRole === 'retrieved_passage' ? 5 : 0) +
        (reference.sourceRole === 'source_document' ? 3 : 0) +
        (isRepresentativeDocumentChunkReference(reference) ? 16 : 0) +
        namedFileConfigTokenScore(reference, query) * 120 +
        namedFileContextScore(reference, query) * 14 +
        kindScore[reference.kind] +
        referenceAnswerFitScore(reference, query) +
        referenceQueryRelevanceScore(reference, query) +
        referenceReviewScore(reference) +
        (reference.score || 0)
    )
}

const namedFileReferenceKindPriority = (reference: ChatReferenceItem): number => {
    if (reference.kind === 'passage') return 0
    if (reference.kind === 'document') return 0
    if (reference.kind === 'artifact') return 1
    if (reference.kind === 'path') return 2
    return 4
}

const isPdfReference = (reference?: ChatReferenceItem | null): boolean => {
    if (!reference) return false
    const contentType = String(reference.contentType || '').toLowerCase()
    if (contentType) return contentType.includes('application/pdf')
    const filename = `${reference.filename || reference.title || reference.endpoint || ''}`.toLowerCase()
    return /\.pdf(?:$|[?#])/.test(filename)
}

const isExternalUrlLabel = (value?: string): boolean =>
    /^https?:\/\//i.test(String(value || '').trim())

const isExternalUrlRelationshipReference = (reference: ChatReferenceItem): boolean =>
    Boolean(
        reference.relationship &&
        (isExternalUrlLabel(reference.relationship.source) || isExternalUrlLabel(reference.relationship.target)),
    )

const referenceExternalUrlValues = (reference: ChatReferenceItem): string[] =>
    uniqueStrings([
        reference.relationship?.source,
        reference.relationship?.target,
        ...(reference.path?.nodes || []),
        ...(reference.path?.relationships || []).flatMap(relationship => [
            relationship.source,
            relationship.target,
        ]),
        ...(reference.sourceLocations || []).flatMap(location => [
            location.label,
            location.value,
        ]),
    ]).filter(isExternalUrlLabel)

export const selectReferencePrimaryExternalUrl = (
    reference?: ChatReferenceItem | null,
): string =>
    reference ? referenceExternalUrlValues(reference)[0] || '' : ''

export const selectReferencePdfViewerReference = (
    selected?: ChatReferenceItem | null,
    references: ChatReferenceItem[] = [],
): ChatReferenceItem | null => {
    const seen = new Set<string>()
    const candidates = [selected, ...references].filter(Boolean) as ChatReferenceItem[]

    for (const candidate of candidates) {
        const key = candidate.id || `${candidate.endpoint || ''}:${candidate.filename || candidate.title || ''}`
        if (seen.has(key)) continue
        seen.add(key)

        if (isPdfReference(candidate) && normalizeEndpoint(candidate.endpoint)) {
            return candidate
        }
    }

    return null
}

const isStructuredSourceReference = (reference?: ChatReferenceItem | null): boolean => {
    if (!reference) return false
    const contentType = String(reference.contentType || '').toLowerCase()
    const filename = `${reference.filename || reference.endpoint || ''}`.toLowerCase()
    return (
        contentType.includes('text/markdown') ||
        filename.includes('structured_ingest') ||
        /\.structured\.md(?:$|[?#])/.test(filename)
    )
}

const isSourceDocumentContainerReference = (reference?: ChatReferenceItem | null): boolean =>
    Boolean(reference && reference.sourceRole === 'source_document' && !reference.relationship && !reference.path)

const isGeneratedSourceMetadataPreview = (value: string): boolean => {
    const text = String(value || '').replace(/\s+/g, ' ').trim()
    if (!text) return false
    return (
        /^document\s*[-:]/i.test(text) ||
        /\b(?:filename|content type|extraction method|structured text|layout_[a-z_]+)\b/i.test(text) ||
        /\blayout\b.{0,32}\d+(?:\.\d+)?/i.test(text) ||
        /^[-;\d.\s]+(?:layout|layou)\b/i.test(text)
    )
}

const friendlyReferenceSourceType = (reference?: ChatReferenceItem | null): string => {
    if (!reference) return ''
    if (isPdfReference(reference)) return 'PDF source'
    if (isStructuredSourceReference(reference)) return 'Evidence text'

    const contentType = String(reference.contentType || '').toLowerCase()
    const filename = `${reference.filename || reference.endpoint || ''}`.toLowerCase()
    if (contentType.includes('presentation') || /\.(?:ppt|pptx)(?:$|[?#])/.test(filename)) return 'Slide deck'
    if (contentType.includes('spreadsheet') || /\.(?:xls|xlsx|csv)(?:$|[?#])/.test(filename)) return 'Spreadsheet'
    if (contentType.includes('wordprocessing') || /\.(?:doc|docx|odt)(?:$|[?#])/.test(filename)) return 'Document'
    if (contentType.includes('image/')) return 'Image source'
    if (contentType.includes('json')) return 'Structured data'
    if (contentType.includes('markdown')) return 'Markdown'
    if (contentType.includes('text/')) return 'Text source'
    return ''
}

const withDownloadFlag = (endpoint: string): string => {
    if (!endpoint) return ''
    if (/[?&]download=/.test(endpoint)) return endpoint
    return `${endpoint}${endpoint.includes('?') ? '&' : '?'}download=false`
}

const referenceIcon = (kind: ChatReferenceKind, pdf = false) => {
    if (pdf) return <PdfIcon fontSize="small" />
    if (kind === 'path') return <PathIcon fontSize="small" />
    if (kind === 'relationship') return <GraphIcon fontSize="small" />
    if (kind === 'document' || kind === 'artifact') return <DocumentIcon fontSize="small" />
    return <PassageIcon fontSize="small" />
}

const hasRequestedEvidenceFamilyMarker = (reference: ChatReferenceItem): boolean =>
    (reference.evidenceFamilies || []).some(family =>
        normalizeFacetText(family).replace(/\s+/g, '_') === 'requested_evidence_family',
    )

const isRequestedEvidenceFamilyReference = (reference: ChatReferenceItem): boolean =>
    reference.sourceRole === 'answer_evidence' && hasRequestedEvidenceFamilyMarker(reference)

const RICH_EVIDENCE_FAMILY_PATTERN =
    /\b(?:chart|charts|connector|connectors|arrow|arrows|flow|flows|form field|form fields|forms|styled text|tables?|bold|italic|emphasis)\b/i

const isRichEvidenceFamilyReference = (
    reference: ChatReferenceItem,
    query?: string,
): boolean => {
    if (!requestedEvidenceFamilyQuestionQuery(query)) return false
    if (hasRequestedEvidenceFamilyMarker(reference)) return true
    if (reference.sourceRole !== 'answer_evidence') return false
    if (isSyntheticFilenameRelationshipReference(reference)) return false
    const text = [
        reference.title,
        reference.subtitle,
        reference.citationLabel,
        reference.body,
        ...(reference.evidenceFamilies || []),
        ...(reference.terms || []),
        ...(reference.matchedTerms || []),
    ].join(' ')
    return RICH_EVIDENCE_FAMILY_PATTERN.test(text)
}

const REQUESTED_EVIDENCE_FAMILY_ALIASES: Array<{
    key: string
    patterns: RegExp[]
}> = [
    { key: 'charts', patterns: [/\bcharts?\b/i, /\bplots?\b/i, /\bseries\b/i, /\baxis\b/i, /\btrendline\b/i] },
    { key: 'tables', patterns: [/\btables?\b/i, /\brows?\b/i, /\bcolumns?\b/i, /\bcells?\b/i] },
    { key: 'form_fields', patterns: [/\bform fields?\b/i, /\bforms?\b/i, /\bfields?\b/i, /\bcheckbox(?:es)?\b/i] },
    { key: 'styled_text', patterns: [/\bstyled text\b/i, /\bstyled-text\b/i, /\bformatted text\b/i, /\bemphasis\b/i, /\bemphasized\b/i, /\bbold\b/i, /\bitalic\b/i, /\bhighlight\b/i] },
    { key: 'connectors', patterns: [/\bconnectors?\b/i, /\barrows?\b/i, /\bflows?\b/i, /\brelationship line\b/i] },
    { key: 'speaker_notes', patterns: [/\bspeaker notes?\b/i, /\bpresenter notes?\b/i, /\bnotes slide\b/i] },
    { key: 'comments', patterns: [/\bcomments?\b/i, /\bannotations?\b/i, /\breviewer\b/i] },
    { key: 'media', patterns: [/\bmedia\b/i, /\baudio\b/i, /\bvideo\b/i, /\brecording\b/i, /\bembedded media\b/i] },
    { key: 'attachments', patterns: [/\battachments?\b/i, /\bembedded files?\b/i, /\bembedded objects?\b/i, /\battached documents?\b/i] },
    { key: 'links', patterns: [/\blinks?\b/i, /\burls?\b/i, /\buris?\b/i, /\bhref\b/i] },
    { key: 'images', patterns: [/\bimages?\b/i, /\bpictures?\b/i, /\bfigures?\b/i, /\bscreenshots?\b/i] },
    { key: 'layout', patterns: [/\blayout\b/i, /\bspatial\b/i, /\bvisual(?:ly)? near\b/i, /\bpositions?\b/i] },
]

const DEFAULT_REQUESTED_EVIDENCE_FAMILY_ORDER = REQUESTED_EVIDENCE_FAMILY_ALIASES
    .map(definition => definition.key)

const INLINE_TRAY_BREADTH_FAMILY_PRIORITY = [
    'images',
    'media',
    'attachments',
    'speaker_notes',
    'comments',
    'connectors',
    'styled_text',
    'form_fields',
    'tables',
    'charts',
]

const REQUESTED_EVIDENCE_FAMILY_NORMALIZATION: Record<string, string> = {
    attachment: 'attachments',
    attachments: 'attachments',
    chart: 'charts',
    chart_points: 'charts',
    charts: 'charts',
    comment: 'comments',
    comments: 'comments',
    connector: 'connectors',
    connectors: 'connectors',
    embedded_file: 'attachments',
    embedded_files: 'attachments',
    embedded_object: 'attachments',
    embedded_objects: 'attachments',
    form: 'form_fields',
    form_field: 'form_fields',
    form_fields: 'form_fields',
    forms: 'form_fields',
    image: 'images',
    images: 'images',
    link: 'links',
    links: 'links',
    media: 'media',
    note: 'speaker_notes',
    notes: 'speaker_notes',
    speaker_note: 'speaker_notes',
    speaker_notes: 'speaker_notes',
    styled: 'styled_text',
    styled_text: 'styled_text',
    table: 'tables',
    tables: 'tables',
}

const normalizeRequestedEvidenceFamilyKey = (value: string): string => {
    const normalized = normalizeFacetText(value).replace(/\s+/g, '_')
    return REQUESTED_EVIDENCE_FAMILY_NORMALIZATION[normalized] || normalized
}

const requestedEvidenceFamilyOrder = (query?: string): Map<string, number> => {
    const text = queryPromptText(query)
    if (!text) return new Map()
    const matches: Array<{ key: string; index: number }> = []
    REQUESTED_EVIDENCE_FAMILY_ALIASES.forEach(({ key, patterns }) => {
        const indexes = patterns
            .map(pattern => {
                const match = text.match(pattern)
                return match?.index ?? -1
            })
            .filter(index => index >= 0)
        if (indexes.length > 0) {
            matches.push({ key, index: Math.min(...indexes) })
        }
    })

    return new Map(
        matches
            .sort((left, right) => left.index - right.index)
            .map((match, index) => [match.key, index]),
    )
}

const referenceRequestedEvidenceFamilyKeys = (reference: ChatReferenceItem): string[] =>
    uniqueStrings([
        ...(reference.evidenceFamilies || []),
        reference.title,
        reference.citationLabel,
        ...(reference.terms || []),
        ...(reference.matchedTerms || []),
    ])
        .map(normalizeRequestedEvidenceFamilyKey)
        .filter(key => key && key !== 'requested_evidence_family')

const requestedEvidenceFamilyOrderRank = (
    reference: ChatReferenceItem,
    query?: string,
): number => {
    const queryOrder = requestedEvidenceFamilyOrder(query)
    const familyKeys = referenceRequestedEvidenceFamilyKeys(reference)
    const queryRanks = familyKeys
        .map(key => queryOrder.get(key))
        .filter((rank): rank is number => rank !== undefined)
    if (queryRanks.length > 0) return Math.min(...queryRanks)

    const defaultRanks = familyKeys
        .map(key => DEFAULT_REQUESTED_EVIDENCE_FAMILY_ORDER.indexOf(key))
        .filter(rank => rank >= 0)
    return defaultRanks.length > 0 ? Math.min(...defaultRanks) + 100 : Number.MAX_SAFE_INTEGER
}

const primaryRequestedEvidenceFamilyKey = (
    reference: ChatReferenceItem,
    query?: string,
): string => {
    const keys = referenceRequestedEvidenceFamilyKeys(reference)
        .filter(key => key && key !== 'requested_evidence_family')
    if (!keys.length) return ''

    const queryOrder = requestedEvidenceFamilyOrder(query)
    return keys.slice().sort((left, right) => {
        const leftQueryRank = queryOrder.get(left) ?? Number.MAX_SAFE_INTEGER
        const rightQueryRank = queryOrder.get(right) ?? Number.MAX_SAFE_INTEGER
        const leftDefaultRank = DEFAULT_REQUESTED_EVIDENCE_FAMILY_ORDER.indexOf(left)
        const rightDefaultRank = DEFAULT_REQUESTED_EVIDENCE_FAMILY_ORDER.indexOf(right)

        return (
            leftQueryRank - rightQueryRank ||
            (leftDefaultRank >= 0 ? leftDefaultRank : Number.MAX_SAFE_INTEGER) -
                (rightDefaultRank >= 0 ? rightDefaultRank : Number.MAX_SAFE_INTEGER) ||
            left.localeCompare(right)
        )
    })[0]
}

const compareRequestedEvidenceFamilyOrder = (
    left: ChatReferenceItem,
    right: ChatReferenceItem,
    query?: string,
): number => {
    if (!requestedEvidenceFamilyQuestionQuery(query)) return 0
    const leftRequested =
        hasRequestedEvidenceFamilyMarker(left) || isRichEvidenceFamilyReference(left, query)
    const rightRequested =
        hasRequestedEvidenceFamilyMarker(right) || isRichEvidenceFamilyReference(right, query)
    if (!leftRequested && !rightRequested) return 0
    if (leftRequested !== rightRequested) return leftRequested ? -1 : 1
    const leftMarker = hasRequestedEvidenceFamilyMarker(left)
    const rightMarker = hasRequestedEvidenceFamilyMarker(right)
    if (leftMarker !== rightMarker) return leftMarker ? -1 : 1
    return requestedEvidenceFamilyOrderRank(left, query) - requestedEvidenceFamilyOrderRank(right, query)
}

const clampNumber = (value: number, min: number, max: number): number =>
    Math.min(max, Math.max(min, value))

const PDF_VIEWER_MIN_SCALE = 0.35
const PDF_VIEWER_MAX_SCALE = 2
const PDF_VIEWER_ZOOM_STEP = 0.18

export const pdfViewerRenderScale = (
    basePageWidth: number,
    containerWidth: number,
    viewportWidth: number,
    zoomSteps = 0,
): number => {
    if (basePageWidth <= 0 || containerWidth <= 0) return 1
    const visibleContainerWidth = Math.min(containerWidth, viewportWidth || containerWidth)
    const availableWidth = Math.max(220, visibleContainerWidth - 36)
    const fitScale = availableWidth / basePageWidth
    return clampNumber(
        fitScale * (1 + zoomSteps * PDF_VIEWER_ZOOM_STEP),
        PDF_VIEWER_MIN_SCALE,
        PDF_VIEWER_MAX_SCALE,
    )
}

const isPdfTextItem = (value: unknown): value is TextItem => {
    const record = asRecord(value)
    return (
        typeof record.str === 'string' &&
        Array.isArray(record.transform) &&
        record.transform.length >= 6 &&
        typeof record.width === 'number'
    )
}

const pdfHighlightTerms = (terms: string[]): string[] =>
    uniqueStrings(terms)
        .map(term => term.toLowerCase().replace(/\s+/g, ' ').trim())
        .filter(term => term.length >= 3 && !HIGHLIGHT_STOP_TERMS.has(term))
        .slice(0, 18)

const buildPdfHighlights = (
    items: unknown[],
    viewport: PageViewport,
    terms: string[],
): PdfHighlightRect[] => {
    const usableTerms = pdfHighlightTerms(terms)
    if (!usableTerms.length) return []

    return items
        .filter(isPdfTextItem)
        .flatMap((item): PdfHighlightRect[] => {
            const text = item.str.replace(/\s+/g, ' ').trim()
            if (!text) return []
            if (PREVIEW_PROMPT_META_PATTERN.test(text)) return []
            const matchedTerm = usableTerms.find(term => text.toLowerCase().includes(term))
            if (!matchedTerm) return []

            const transformed = pdfjsLib.Util.transform(viewport.transform, item.transform)
            const left = transformed[4]
            const baselineTop = transformed[5]
            const height = clampNumber(
                Math.abs((item.height || 10) * viewport.scale) || Math.abs(transformed[3]) || 10,
                8,
                38,
            )
            const width = clampNumber(
                (item.width || text.length * 5) * viewport.scale,
                18,
                Math.max(18, viewport.width - left),
            )

            return [{
                left: clampNumber(left, 0, viewport.width),
                top: clampNumber(baselineTop - height, 0, viewport.height),
                width,
                height,
                label: matchedTerm,
                kind: 'text',
            }]
        })
        .slice(0, 32)
}

const PDF_POINTS_PER_INCH = 72

export const pdfLocationHighlightFromValue = (
    value: string,
    viewport: Pick<PageViewport, 'width' | 'height' | 'scale'>,
    label = 'source location',
): PdfHighlightRect | null => {
    const text = String(value || '')
    const numbers = text
        .match(/-?\d+(?:\.\d+)?/g)
        ?.map(Number)
        .filter(Number.isFinite) || []
    if (numbers.length < 4) return null

    const [leftRaw, topRaw, widthRaw, heightRaw] = numbers
    const hasInchUnits = /(?:\d|\.)\s*in(?:ch(?:es)?)?\b/i.test(text)
    const percentLike = !hasInchUnits && Math.max(...numbers.slice(0, 4).map(Math.abs)) <= 100
    const unitScale = hasInchUnits ? PDF_POINTS_PER_INCH * viewport.scale : viewport.scale
    const left = percentLike ? (leftRaw / 100) * viewport.width : leftRaw * unitScale
    const top = percentLike ? (topRaw / 100) * viewport.height : topRaw * unitScale
    const width = percentLike ? (widthRaw / 100) * viewport.width : widthRaw * unitScale
    const height = percentLike ? (heightRaw / 100) * viewport.height : heightRaw * unitScale

    return {
        left: clampNumber(left, 0, viewport.width),
        top: clampNumber(top, 0, viewport.height),
        width: clampNumber(width, 20, viewport.width),
        height: clampNumber(height, 10, viewport.height),
        label,
        kind: 'location',
    }
}

const isBroadPdfLocationHighlight = (
    highlight: PdfHighlightRect,
    viewport: PageViewport,
    label: string,
): boolean => {
    if (/\bzone\b/i.test(label)) return true
    const viewportArea = Math.max(1, viewport.width * viewport.height)
    const highlightArea = highlight.width * highlight.height
    return highlightArea / viewportArea > 0.18
}

export const buildPdfLocationHighlights = (
    references: ChatReferenceItem[],
    viewport: PageViewport,
    pageNumber: number,
): PdfHighlightRect[] => {
    const highlights: PdfHighlightRect[] = []

    references.forEach(reference => {
        const referencedPage = pageNumberFromReference(reference)
        if (referencedPage && referencedPage !== pageNumber) return

        ;(reference.sourceLocations || []).forEach(location => {
            const label = `${location.label} ${location.category || ''}`.toLowerCase()
            if (!label.includes('region') && !label.includes('bbox')) return

            const highlight = pdfLocationHighlightFromValue(
                String(location.value || ''),
                viewport,
                compactSourceLocationLabel(location),
            )
            if (highlight && !isBroadPdfLocationHighlight(highlight, viewport, label)) {
                highlights.push(highlight)
            }
        })
    })

    return highlights.slice(0, 24)
}

const pdfRectsOverlapVertically = (left: PdfHighlightRect, right: PdfHighlightRect): boolean => {
    const overlap = Math.min(left.top + left.height, right.top + right.height) - Math.max(left.top, right.top)
    return overlap > Math.min(left.height, right.height) * 0.45 || Math.abs(left.top - right.top) <= 5
}

const pdfHighlightRectKey = (highlight: PdfHighlightRect): string =>
    [
        highlight.kind,
        Math.round(highlight.left),
        Math.round(highlight.top),
        Math.round(highlight.width),
        Math.round(highlight.height),
    ].join(':')

export const mergePdfHighlightRects = (highlights: PdfHighlightRect[]): PdfHighlightRect[] => {
    const locationSeen = new Set<string>()
    const locationHighlights: PdfHighlightRect[] = []
    const textHighlights = highlights
        .filter(highlight => highlight.kind === 'text')
        .sort((left, right) => left.top - right.top || left.left - right.left)

    highlights
        .filter(highlight => highlight.kind === 'location')
        .forEach(highlight => {
            const key = pdfHighlightRectKey(highlight)
            if (locationSeen.has(key)) return
            locationSeen.add(key)
            locationHighlights.push(highlight)
        })

    const mergedTextHighlights: PdfHighlightRect[] = []
    textHighlights.forEach(highlight => {
        const existing = mergedTextHighlights.find(candidate => pdfRectsOverlapVertically(candidate, highlight))
        if (!existing) {
            mergedTextHighlights.push({ ...highlight })
            return
        }

        const right = Math.max(existing.left + existing.width, highlight.left + highlight.width)
        const bottom = Math.max(existing.top + existing.height, highlight.top + highlight.height)
        existing.left = Math.min(existing.left, highlight.left)
        existing.top = Math.min(existing.top, highlight.top)
        existing.width = right - existing.left
        existing.height = bottom - existing.top
        existing.label = uniqueStrings([existing.label, highlight.label]).join(', ')
    })

    const visibleLocationHighlights = mergedTextHighlights.length > 0
        ? []
        : locationHighlights

    return [...visibleLocationHighlights, ...mergedTextHighlights]
        .sort((left, right) => left.top - right.top || left.left - right.left)
}

const referenceKindLabel = (reference: ChatReferenceItem): string => {
    if (isPdfReference(reference)) return 'PDF'
    if (hasRequestedEvidenceFamilyMarker(reference)) return 'Evidence'
    if (reference.sourceRole === 'answer_evidence') return 'Citation'
    if (reference.sourceRole === 'source_document' && reference.kind === 'passage') return 'Evidence'
    if (reference.kind === 'path') return 'Path'
    if (reference.kind === 'relationship') return 'Relation'
    if (reference.kind === 'artifact') return 'Artifact'
    if (reference.kind === 'document') return 'Document'
    return 'Passage'
}

const referenceAccentColor = (
    reference: Pick<ChatReferenceItem, 'kind'> | null | undefined,
    theme: Theme,
): string => {
    if (reference?.kind === 'path') return theme.palette.secondary.main
    if (reference?.kind === 'relationship') return theme.palette.success.main
    if (reference?.kind === 'document' || reference?.kind === 'artifact') return theme.palette.warning.main
    return theme.palette.info.main
}

const COORDINATE_ONLY_REFERENCE_TITLE_PATTERN =
    /^(?:\s*(?:row|line|page|slide|sheet|table|cell|location)\b\s*[=:]?\s*[\w.-]+[;,\s]*)+$/i

const isCoordinateOnlyReferenceTitle = (value: string): boolean =>
    COORDINATE_ONLY_REFERENCE_TITLE_PATTERN.test(value.trim())

const isCoordinateDumpReferenceLabel = (value: string): boolean =>
    /\b(?:row|line|page|slide|sheet|table|cell|location)\s*=\s*[\w.-]+(?:\s*;\s*(?:row|line|page|slide|sheet|table|cell|location)\s*=\s*[\w.-]+)+/i.test(value)

const formatReferenceFacet = (value: string): string =>
    value
        .replace(/[_-]+/g, ' ')
        .replace(/\b\w/g, letter => letter.toUpperCase())

const EVIDENCE_BUNDLE_LABELS: Record<string, { singular: string; plural: string }> = {
    chart_points: { singular: 'chart point', plural: 'chart points' },
    charts: { singular: 'chart', plural: 'charts' },
    code_blocks: { singular: 'code block', plural: 'code blocks' },
    comments: { singular: 'comment', plural: 'comments' },
    connectors: { singular: 'connector', plural: 'connectors' },
    diagram_relationships: { singular: 'diagram edge', plural: 'diagram edges' },
    embedded_files: { singular: 'embedded file', plural: 'embedded files' },
    forms: { singular: 'form field', plural: 'form fields' },
    images: { singular: 'image', plural: 'images' },
    links: { singular: 'link', plural: 'links' },
    media: { singular: 'media item', plural: 'media items' },
    notes: { singular: 'note', plural: 'notes' },
    ocr: { singular: 'OCR result', plural: 'OCR results' },
    tables: { singular: 'table', plural: 'tables' },
}

const normalizeEvidenceBundleKey = (value: string): string =>
    value
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '_')
        .replace(/^_+|_+$/g, '')

const formatEvidenceBundleNoun = (value: string, count?: number): string => {
    const mapped = EVIDENCE_BUNDLE_LABELS[normalizeEvidenceBundleKey(value)]
    if (mapped) return count === 1 ? mapped.singular : mapped.plural
    return value.replace(/[_-]+/g, ' ').toLowerCase()
}

const evidenceFamilyDisplayScore = (value: string): number => {
    const normalized = normalizeFacetText(value)
    if (!normalized) return 0

    if (/\bsemantic\b/.test(normalized)) return 120
    if (/\bconnector\b/.test(normalized)) return 112
    if (/\bchart\b/.test(normalized)) return 104
    if (/\bform\b/.test(normalized)) return 98
    if (/\btable\b/.test(normalized)) return 94
    if (/\bdiagram\b/.test(normalized)) return 88
    if (/\bstyled\b/.test(normalized)) return 82
    if (/\bnotes?\b/.test(normalized)) return 78
    if (/\blink\b/.test(normalized)) return 46
    if (/\bocr\b/.test(normalized)) return 70
    if (/\bcode\b/.test(normalized)) return 66
    if (STRUCTURAL_REVIEW_FAMILY_PATTERNS.some(pattern => pattern.test(normalized))) return -24
    return 20
}

const isHighSignalEvidenceLabel = (value: string): boolean =>
    evidenceFamilyDisplayScore(value) >= 60

const referenceEvidenceLabels = (reference?: ChatReferenceItem | null): string[] =>
    uniqueStrings(reference?.evidenceFamilies || [])
        .filter(family => !isCoordinateDumpReferenceLabel(family))
        .filter(family =>
            !reference ||
            !hasRequestedEvidenceFamilyMarker(reference) ||
            normalizeFacetText(family).replace(/\s+/g, '_') !== 'requested_evidence_family',
        )
        .map(formatReferenceFacet)
        .sort((left, right) =>
            evidenceFamilyDisplayScore(right) - evidenceFamilyDisplayScore(left) ||
            left.localeCompare(right),
        )
        .filter((label, _index, labels) => {
            const hasHighSignal = labels.some(isHighSignalEvidenceLabel)
            return !hasHighSignal || isHighSignalEvidenceLabel(label)
        })
        .slice(0, 5)

const OFFICE_PACKAGE_PATH_FRAGMENT_PATTERN = /\b(?:ppt|xl|word)\/[^\s|;,)]+/gi
const OFFICE_PACKAGE_PATH_FRAGMENT_TEST_PATTERN = /\b(?:ppt|xl|word)\/[^\s|;,)]+/i

const cleanOfficePackagePathFragments = (value: string): string =>
    value.replace(OFFICE_PACKAGE_PATH_FRAGMENT_PATTERN, match =>
        cleanGraphReferenceLabel(match, 'document package evidence'),
    )

const cleanOfficePackageProvenanceText = (value: string): string =>
    cleanOfficePackagePathFragments(value)
        .replace(/\bembedded[_\s-]*package\b/gi, 'embedded data')
        .replace(/\b(?:media|document|workbook|chart|table|sheet|slide)\s+part\s+(embedded data|chart evidence|table evidence|sheet evidence|slide evidence|media evidence)\b/gi, '$1')
        .replace(/\b(?:media|document|workbook)\s+embedded data\b/gi, 'embedded data')
        .replace(/\b(?:chart|table|sheet|slide|media)\s+(chart evidence|table evidence|sheet evidence|slide evidence|media evidence)\b/gi, '$1')
        .replace(/\s+/g, ' ')
        .trim()

const humanizeVisibleGraphArrows = (value: string): string =>
    value
        .replace(/\s+--?\[[^\]]+\]--?>\s+/g, ' to ')
        .replace(/\s*->\s*/g, ' to ')
        .replace(/\s*→\s*/g, ' to ')
        .replace(/(\S)[.,;:!?]+\s+to\s+/g, '$1 to ')
        .replace(/\s+/g, ' ')
        .trim()

const cleanSourceLocationValue = (value: string): string =>
    humanizeVisibleGraphArrows(cleanOfficePackageProvenanceText(value))

const isGeometryLocationValue = (label: string, value: string): boolean => {
    const normalizedLabel = label.toLowerCase()
    if (!/\b(?:bbox|bounding box|coordinates?|layout bbox|region)\b/.test(normalizedLabel)) return false
    return (
        /\b(?:x|y|w|h)=\d+(?:\.\d+)?(?:in)?\b/i.test(value) ||
        /^\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?){2,}$/.test(value.trim())
    )
}

const referenceStructuredLocations = (reference?: ChatReferenceItem | null): ChatReferenceLocation[] => {
    const seen = new Set<string>()
    const output: ChatReferenceLocation[] = []
    const sourceLocations = reference?.sourceLocations || []

    sourceLocations.forEach(location => {
        const label = cleanSourceLocationValue(location.label)
        const value = cleanSourceLocationValue(location.value)
        if (!label || !value) return
        const key = `${label.toLowerCase()}::${value.toLowerCase()}`
        if (seen.has(key)) return
        seen.add(key)
        output.push({
            label,
            value,
            category: cleanSourceLocationValue(location.category || ''),
        })
    })

    return output.slice(0, 12)
}

const compactSourceLocationLabel = (location: ChatReferenceLocation): string => {
    const label = formatReferenceFacet(location.label)
    const value = cleanSourceLocationValue(location.value)
    if (!value) return label
    if (isGeometryLocationValue(label, value)) return ''
    if (/^(?:slide title|page title|section title|title)$/i.test(label)) {
        return compactReferenceText(value.replace(/^slide\s+\d+\s*:\s*/i, ''), 34)
    }
    if (/^(?:source shape|target shape|shape)$/i.test(label)) {
        const shape = value.match(/\bshape\s+(.+)$/i)
        return `Shape ${compactReferenceText((shape ? shape[1] : value).replace(/^#/, ''), 24)}`
    }
    if (/^zone$/i.test(label)) {
        return `${formatReferenceFacet(value)} zone`
    }
    if (/^(?:anchor|location|source location)$/i.test(label)) {
        const cell = value.match(/\bcell\s+(R\d+C\d+)\b/i)
        if (cell) return `Cell ${cell[1].toUpperCase()}`
        const table = value.match(/\btable\s+(\d+)\b/i)
        if (table) return `Table ${table[1]}`
        const block = value.match(/\bblock\s+(\d+)\b/i)
        if (block) return `Block ${block[1]}`
        const slide = value.match(/\bslide\s+(\d+)\b/i)
        if (slide) return `Slide ${slide[1]}`
        const page = value.match(/\bpage\s+(\d+)\b/i)
        if (page) return `Page ${page[1]}`
    }
    if (/\b(?:embedded data|package|part)\b/i.test(`${label} ${location.value}`)) {
        return cleanOfficePackageProvenanceText(value || label)
    }
    if (label.toLowerCase() === value.toLowerCase()) return label
    const compactValue = value.length <= 38 ? value : `${value.slice(0, 37).trim()}...`
    return `${label} ${compactValue}`
}

const referenceEvidenceBundleLabel = (reference?: ChatReferenceItem | null): string => {
    const locations = referenceStructuredLocations(reference)
    const evidenceType = locations.find(location => location.label.toLowerCase() === 'evidence type')?.value || ''
    const evidenceCountText = locations.find(location => location.label.toLowerCase() === 'evidence count')?.value || ''
    const evidenceCount = Number(evidenceCountText)
    const count = Number.isFinite(evidenceCount) && evidenceCount > 0 ? evidenceCount : undefined
    const noun = evidenceType ? formatEvidenceBundleNoun(evidenceType, count) : ''

    if (count !== undefined && noun) return `${count} ${noun}`
    if (noun) return formatReferenceFacet(noun)
    if (count !== undefined) return `${count} evidence ${count === 1 ? 'item' : 'items'}`
    return ''
}

const cleanLocationLabel = (value: string): string => {
    const text = value
        .replace(/\btable\s+(\d+)\b/gi, 'Table $1')
        .replace(/\bcell\s+(R\d+C\d+)\b/gi, 'Cell $1')
        .replace(/\bstyle\s+summary\s+bold\b/gi, 'Bold')
        .replace(/\bstyle\s+summary\s+italic\b/gi, 'Italic')
        .replace(/\bstyle\s+summary\s+underline(?:d)?\b/gi, 'Underlined')
        .replace(/\bstyle\s+bold\b/gi, 'Bold')
        .replace(/\bstyle\s+italic\b/gi, 'Italic')
        .replace(/\bstyle\s+underline(?:d)?\b/gi, 'Underlined')
        .replace(/^Value\s+True$/i, 'Selected')
        .replace(/^Value\s+False$/i, 'Not selected')
        .replace(/^Checked\s+(?:checked|true|yes)$/i, 'Selected')
        .replace(/^Checked\s+(?:false|no)$/i, 'Not selected')
        .replace(/\b(Table\s+\d+)\s+\(table\)(?:\s*;\s*\1)?/gi, '$1')
        .replace(/\b(Table\s+\d+)(?:\s*;\s*\1\b)+/gi, '$1')
        .replace(/^Slide\s+(\d+)\s+[A-Za-z]{1,4}\.\.\.?$/i, 'Slide $1')
        .replace(/^Slide\s+(\d+)\s*:\s*[A-Za-z]{1,4}\.\.\.?$/i, 'Slide $1')
        .replace(/\s+/g, ' ')
        .trim()
    if (!text) return ''
    if (/^[A-Za-z]{1,4}\.\.\.?$/i.test(text)) return ''
    if (/^(?:region|bbox|bounding box|coordinates?|layout bbox)\s+\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?){2,}$/i.test(text)) {
        return ''
    }

    if (/^file=/i.test(text)) {
        const fileMatch = text.match(/^file=([^;]+)/i)
        const tableMatch = text.match(/\btable=([^;]+)/i)
        const file = fileMatch ? referenceBasename(fileMatch[1].trim()) : ''
        const table = tableMatch ? `table ${tableMatch[1].trim()}` : ''
        return [file, table].filter(Boolean).join(' / ')
    }

    if (/^local pipeline /i.test(text) || /^pipeline handoff /i.test(text)) {
        const parts = text.split(';').map(part => part.trim()).filter(Boolean)
        const section = parts.find(part => /^section\b/i.test(part))
        const element = parts.find(part => /^element\b/i.test(part))
        const line = parts.find(part => /^line\b/i.test(part))
        const run = parts.find(part => /^run\b/i.test(part))
        return [section, element, line, run]
            .filter((part): part is string => Boolean(part))
            .map(part => part.replace(/^section\s+/i, '').replace(/^element\s+/i, ''))
            .join(' / ') || 'document context'
    }

    return text
}

const isFileOnlyLocationLabel = (value: string): boolean =>
    /^file\b/i.test(value.trim()) ||
    /\.(?:pdf|pptx?|docx?|xlsx?|csv|md|txt|json|xml)(?:$|[,\s])/i.test(value.trim())

export const referenceLocationLabels = (reference?: ChatReferenceItem | null): string[] => {
    const labels = uniqueStrings([
        referenceEvidenceBundleLabel(reference),
        ...referenceStructuredLocations(reference)
            .filter(location => String(location.category || '').toLowerCase() !== 'evidence')
            .map(compactSourceLocationLabel),
        ...(reference?.locationHints || []),
    ])
        .map(cleanLocationLabel)
        .filter(label => Boolean(label) && !isCoordinateDumpReferenceLabel(label))
    const highSignal = labels.filter(label => !isFileOnlyLocationLabel(label))
    if (isSourceDocumentContainerReference(reference)) {
        return highSignal.slice(0, 4)
    }

    const lowSignal = labels.filter(isFileOnlyLocationLabel)
    return [...highSignal, ...lowSignal].slice(0, 4)
}

const referenceMatchLabels = (reference?: ChatReferenceItem | null): string[] =>
    uniqueStrings(reference?.matchedTerms || [])
        .map(value => value.replace(/\s+/g, ' ').trim())
        .slice(0, 6)

const referenceDocumentKey = (reference: ChatReferenceItem): string => {
    const rawKey =
        reference.filename ||
        reference.runId ||
        reference.documentId ||
        (reference.relationship
            ? [
                'relationship',
                reference.relationship.source,
                reference.relationship.relation,
                reference.relationship.target,
            ].filter(Boolean).join(':')
            : '') ||
        (reference.path?.nodes?.length ? `path:${reference.path.nodes.join(':')}` : '') ||
        reference.endpoint ||
        reference.title ||
        reference.id
    return `${referenceBasename(String(rawKey || reference.id)).toLowerCase()}`
}

const referenceDocumentTitle = (reference: ChatReferenceItem): string =>
    reference.filename ||
    reference.documentId ||
    (reference.relationship
        ? [
            reference.relationship.source || 'Source',
            reference.relationship.target || 'Target',
        ].filter(Boolean).join(' to ')
        : '') ||
    (reference.path?.nodes?.length ? referencePathTitle(displayPathNodes(reference.path.nodes)) : '') ||
    reference.title ||
    reference.subtitle ||
    'Source document'

const referenceDocumentSubtitle = (reference: ChatReferenceItem): string =>
    uniqueStrings([
        friendlyReferenceSourceType(reference),
        reference.endpoint ? 'source available' : '',
    ])
        .slice(0, 2)
        .join(' · ')

const formatReferenceCount = (count: number, singular: string, plural = `${singular}s`): string =>
    `${count} ${count === 1 ? singular : plural}`

const formatOptionalCount = (
    count: number | undefined,
    singular: string,
    plural = `${singular}s`,
): string =>
    count ? formatReferenceCount(count, singular, plural) : ''

const compactReferenceText = (value: string, limit = 72): string => {
    const text = value.replace(/\s+/g, ' ').trim()
    if (text.length <= limit) return text
    return `${text.slice(0, Math.max(1, limit - 1)).trim()}...`
}

const normalizedPreviewKey = (value: string): string =>
    value
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()

const PREVIEW_LOW_VALUE_QUERY_TERMS = new Set([
    'across',
    'answer',
    'both',
    'chat',
    'cite',
    'compare',
    'describe',
    'document',
    'documents',
    'evidence',
    'files',
    'graph',
    'name',
    'source',
    'sources',
    'strongest',
    'uploaded',
])

const PREVIEW_PROMPT_META_PATTERN =
    /^(?:(?:page|slide)\s+\d+(?:\s+block\s+\d+)?\s*:\s*)?(?:question target|graph intent hints?|generated query|user query|the user explicitly asked|requested evidence|answer instruction|use this path)\b/i

const previewQueryWords = (query?: string): string[] =>
    String(query || '')
        .toLowerCase()
        .replace(/[^a-z0-9.-]+/g, ' ')
        .split(/\s+/)
        .map(word => word.replace(/[.,;:!?]+$/g, '').trim())
        .filter(word =>
            word.length >= 4 &&
            !STOP_WORDS.has(word) &&
            !LOW_VALUE_SOURCE_QUERY_TERMS.has(word) &&
            !PREVIEW_LOW_VALUE_QUERY_TERMS.has(word) &&
            !/\.(?:pdf|pptx?|docx?|xlsx?|csv|md|txt|json|xml)$/i.test(word),
        )

const previewQueryPhrases = (query?: string): string[] => {
    const text = String(query || '')
    const capitalized = Array.from(
        text.matchAll(/\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)+\b/g),
    ).map(match => match[0])
    const words = previewQueryWords(query)
    const ngrams: string[] = []
    for (let width = 4; width >= 2; width -= 1) {
        words.forEach((_, index) => {
            const phrase = words.slice(index, index + width)
            if (phrase.length === width) ngrams.push(phrase.join(' '))
        })
    }
    return uniqueStrings([...capitalized, ...ngrams])
        .map(value => normalizedPreviewKey(value))
        .filter(value => value.length >= 4)
}

const isRedundantPreview = (candidate: string, anchor: string): boolean => {
    const candidateKey = normalizedPreviewKey(candidate)
    const anchorKey = normalizedPreviewKey(anchor)
    if (!candidateKey || !anchorKey) return false
    return (
        candidateKey === anchorKey ||
        (candidateKey.length >= 28 && anchorKey.includes(candidateKey)) ||
        (anchorKey.length >= 28 && candidateKey.includes(anchorKey))
    )
}

export const dedupeReferencePreviewAgainstSelectedEvidence = (
    preview: string,
    selectedEvidenceLead: string,
): string =>
    selectedEvidenceLead && isRedundantPreview(preview, selectedEvidenceLead)
        ? ''
        : preview

const cleanStructuredEvidenceValue = (value: string): string =>
    dedupeStructuredEvidenceFields(value)
        .replace(/\s*;\s*(?:table|sheet|slide|chart|document|media)?\s*source\s+semantic_[a-z_]+\b[^;\n)]*/gi, '')
        .replace(/\s+\bvisibility\s+(?:visible|hidden)\b/gi, '')
        .replace(/\s+\bsemantic_[a-z_]+\b/gi, '')
        .replace(/\s+/g, ' ')
        .trim()

const dedupeStructuredEvidenceFields = (value: string): string => {
    const parts = String(value || '').split(/\s*;\s*/).map(part => part.trim())
    if (parts.length <= 1) return value

    const seen = new Set<string>()
    return parts
        .filter(part => {
            if (!part) return false
            const match = part.match(/^([A-Za-z][A-Za-z0-9 _-]{1,48})\s*[:=]\s*(.+)$/)
            if (!match) return true
            const key = match[1].replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase()
            const fieldValue = match[2].replace(/\s+/g, ' ').trim().toLowerCase()
            const dedupeKey = `${key}:${fieldValue}`
            if (seen.has(dedupeKey)) return false
            seen.add(dedupeKey)
            return true
        })
        .join('; ')
}

const cleanStructuredEvidencePreview = (
    value: string,
    relationship?: {
        source?: string
        relation?: string
        target?: string
        filename?: string
    },
): string => {
    const raw = String(value || '').replace(/\s+/g, ' ').trim()
    const evidenceMatch = raw.match(/\bevidence\s*[:=]\s*([^;\n]+)/i)
    const evidence = cleanStructuredEvidenceValue(evidenceMatch?.[1] || '')
    if (!evidence) return ''

    const table = raw.match(/\btable\s*[:=#]?\s*(\d+)\b/i)?.[1]
    const row = raw.match(/\brow\s*[:=#]?\s*(\d+)\b/i)?.[1]
    const cell =
        raw.match(/\bcell\s+(R\d+C\d+)\b/i)?.[1] ||
        raw.match(/\b(R\d+C\d+)\s*:/i)?.[1]
    const formField = cleanStructuredEvidenceValue(
        raw.match(/\blabel\s*[:=]\s*([^;]+)/i)?.[1] ||
        raw.match(/\bform\s+field\s*[:=]\s*([^;]+)/i)?.[1] ||
        '',
    )
    const style =
        /\b(?:style\s+bold|bold)\b/i.test(raw) ? 'Bold text'
        : /\b(?:style\s+italic|italic)\b/i.test(raw) ? 'Italic text'
        : /\b(?:style\s+underline(?:d)?|underlined)\b/i.test(raw) ? 'Underlined text'
        : ''
    const chart = cleanStructuredEvidenceValue(
        raw.match(/\bchart\s*[:=]?\s*([^;]+)/i)?.[1] || '',
    )

    const tableLocation = uniqueStrings([
        table ? `Table ${table}` : '',
        row ? `row ${row}` : '',
        cell ? `Cell ${cell}` : '',
    ]).join(' ')
    const prefix = uniqueStrings([
        tableLocation,
        formField && !/^[^:]{2,80}:\s*\S/.test(evidence) ? `Form field ${formField}` : '',
        chart ? `Chart ${chart}` : '',
        style,
    ]).join(' · ')

    return humanizeVisibleGraphArrows(cleanGraphReferenceEvidenceText(
        prefix ? `${prefix}: ${evidence}` : evidence,
        relationship,
    ))
}

const cleanReferenceEvidencePreview = (
    value: string,
    relationship?: {
        source?: string
        relation?: string
        target?: string
        filename?: string
    },
): string => {
    const structuredEvidence = cleanStructuredEvidencePreview(value, relationship)
    if (structuredEvidence) return structuredEvidence

    const stripGeneratedSourceEnvelope = (text: string): string => {
        const hasGeneratedEnvelope =
            /\b(?:document\s*-\s*)?filename\s*:/i.test(text) &&
            /\b(?:content\s+type|extraction\s+method)\s*:/i.test(text)
        if (!hasGeneratedEnvelope) return text

        return text
            .replace(/\b(?:document|source document)\s*-\s*/gi, '')
            .replace(/\b(?:filename|content\s+type|extraction\s+method|structured\s+text|characters|char\s+count|truncated|run|domain|generated|warnings|sha256|hash)\s*:\s*[^#;\n]+(?:\s*[-;]\s*)?/gi, '')
            .replace(/#{1,6}\s*(?:extracted structure|document text|source)\b[:\s]*/gi, '')
            .replace(/\s*;\s*(?:layout_[a-z_]+|layout\s+\w+|source|file|page|slide|zone|location)\s*[:=]?\s*[^;.\n]+/gi, '')
            .replace(/\s{2,}/g, ' ')
            .replace(/^\s*[-;:]+\s*/g, '')
            .trim()
    }

    return humanizeVisibleGraphArrows(cleanGraphReferenceEvidenceText(
        dedupeStructuredEvidenceFields(stripGeneratedSourceEnvelope(cleanDocumentTextForReading(value)))
        .replace(/\blocal pipeline\s+[^;()]+;\s*/gi, '')
        .replace(/\b\d+(?:\.\d+)?%\s+match\b/gi, '')
        .replace(/\bscope\s*:\s*\)?\s*/gi, '')
        .replace(/\btable\s+(\d+)\b/gi, 'Table $1')
        .replace(/\bcell\s+(R\d+C\d+)\b/gi, 'Cell $1')
        .replace(/\bstyle\s+bold\b/gi, 'Bold')
        .replace(/\bstyle\s+italic\b/gi, 'Italic')
        .replace(/\bstyle\s+underline(?:d)?\b/gi, 'Underlined')
        .replace(/\b(Table\s+\d+)\s+\(table\)(?:\s*;\s*\1)?/gi, '$1')
        .replace(/\b(Table\s+\d+)(?:\s*;\s*\1\b)+/gi, '$1')
        .replace(/\s*;\s*(?:table|sheet|slide|chart|document|media)?\s*source\s+semantic_[a-z_]+\b[^;.)]*/gi, '')
        .replace(/\s*;\s*(?:row|line|page|slide|sheet|table|cell|location)\s*=\s*[^;.)]+/gi, '')
        .replace(/\s+\bvisibility\s+(?:visible|hidden)\b/gi, '')
        .replace(/\s+\bsemantic_[a-z_]+\b/gi, '')
        .replace(/\bkey refs?\s*:\s*.*$/gi, '')
        .replace(/\s*\|\s*(?:layout_bbox|layout_zone|slide_title|style_su\w*|relationship_id|image_relationship_id)=[^|;.)]+/gi, '')
        .replace(/\s*;\s*(?:layout_bbox|layout_zone|slide_title|style_su\w*|relationship_id|image_relationship_id)=[^;.)]+/gi, '')
        .replace(/\bevidence types?\s*:\s*[^.;|]+/gi, '')
        .replace(/\bprovenance\s*:\s*/gi, '')
        .replace(/\s*\bfile=[^;)]+/gi, '')
        .replace(/\bfile\s+[\w./-]+\s*/gi, '')
        .replace(/\s*;\s*x=\d+(?:\.\d+)?(?:in)?\s+y=\d+(?:\.\d+)?(?:in)?\s+w=\d+(?:\.\d+)?(?:in)?\s+h=\d+(?:\.\d+)?(?:in)?/gi, '')
        .replace(/\s*;?\s*layout\s+[^;.)]*\bx=\d+(?:\.\d+)?(?:in)?\s+y=\d+(?:\.\d+)?(?:in)?\s+w=\d+(?:\.\d+)?(?:in)?\s+h=\d+(?:\.\d+)?(?:in)?/gi, '')
        .replace(/\s+\bx=\d+(?:\.\d+)?(?:in)?\s+y=\d+(?:\.\d+)?(?:in)?\s+w=\d+(?:\.\d+)?(?:in)?\s+h=\d+(?:\.\d+)?(?:in)?/gi, '')
        .replace(/\s*;\s*(?:source|file|slide|page|sheet|table|chart|visibility|parse_status|relationship_id|image_relationship_id)=[^;)]+/gi, '')
        .replace(/\s*;\s*(?:visibility|parse status)\s*:\s*[^;)]+/gi, '')
        .replace(/\s*;\s*(?:form\s+field|field|kind|link\s+uri|uri)\s+[^;)]+/gi, '')
        .replace(/\s+\bslide\s+\d+\s*:\s*([^;.)]+)/gi, (match, detail) =>
            /\b(?:table|chart|cell|form|styled|bold|italic|underlined)\b/i.test(String(detail || ''))
                ? match
                : '',
        )
        .replace(/\s+\)/g, ')')
        .replace(/\(\s+/g, '(')
        .replace(/\s+/g, ' ')
        .replace(/\s*·\s*;\s*/g, ' · ')
        .replace(/\(\s*;\s*/g, '(')
        .replace(/\(\s*\)/g, '')
        .trim(),
        relationship,
    ))
}

export const cleanRequestedEvidencePreview = (
    value: string,
    filename?: string,
): string => {
    const sourceLabelPattern = filename
        ? referenceBasename(filename).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
        : '[\\w./ -]+\\.(?:pdf|pptx?|docx?|xlsx?|csv|md|txt|json|xml)'
    const raw = String(value || '')
        .replace(/\s+/g, ' ')
        .replace(new RegExp(`\\s+Source:\\s*${sourceLabelPattern}\\b[\\s\\S]*$`, 'i'), '')
        .replace(/\s+source document,\s*(?:slide|page|sheet|table|chart)\b[\s\S]*$/i, '')
        .replace(/\s*\([^()]+?\s+is\s+(?:above|below|left of|right of|near|overlapping|overlaps)\s+[^()]+?\)/gi, '')
        .trim()
    const structuredEvidence = cleanStructuredEvidencePreview(raw, { filename })
    if (structuredEvidence) return structuredEvidence

    const evidenceMatch = raw.match(/\bevidence=([^;\n]+)/i)
    const leadingEvidence = evidenceMatch?.[1]
    const withoutArrow = raw.replace(/^.*?--[^;\n]*?->\s*/i, '')
    const withoutProvenance = (leadingEvidence || withoutArrow || raw)
        .replace(/\s*;\s*provenance=[^;\n]+/gi, '')
        .replace(/\s*;\s*source=[^;\n]+/gi, '')
        .trim()
    return cleanReferenceEvidencePreview(withoutProvenance, { filename })
}

const cleanGraphLabel = (value: string | undefined, fallback = 'Source'): string => {
    return cleanGraphReferenceLabel(value, fallback)
}

const concisePathNodeLabel = (value: string): string => {
    if (!isExternalUrlLabel(value)) return value
    try {
        const parsed = new URL(value)
        const leaf = parsed.pathname.split('/').filter(Boolean).pop()
        return leaf ? `${parsed.hostname}/${leaf}` : parsed.hostname
    } catch {
        return value
    }
}

const referencePathTitle = (nodes: string[]): string => {
    let displayNodes = nodes.length > 2 && /^source document$/i.test(nodes[0])
        ? nodes.slice(1)
        : nodes
    if (displayNodes.length > 2 && /^source document$/i.test(displayNodes[displayNodes.length - 1])) {
        displayNodes = displayNodes.slice(0, -1)
    }
    const conciseNodes = displayNodes.map(concisePathNodeLabel)
    if (conciseNodes.length <= 4) return conciseNodes.join(' to ')
    return `${conciseNodes.slice(0, 3).join(' to ')} to ... to ${conciseNodes[conciseNodes.length - 1]}`
}

const referenceGroupSummary = (group: ReferenceDocumentGroup): string =>
    [
        group.counts.passage ? formatReferenceCount(group.counts.passage, 'passage') : '',
        group.counts.path ? formatReferenceCount(group.counts.path, 'path') : '',
        group.counts.relationship ? formatReferenceCount(group.counts.relationship, 'relation') : '',
        group.counts.document || group.counts.artifact
            ? formatReferenceCount((group.counts.document || 0) + (group.counts.artifact || 0), 'file')
            : '',
    ].filter(Boolean).join(' · ')

export const isSourceNavigationReference = (reference: ChatReferenceItem): boolean =>
    Boolean(
        reference.filename ||
        reference.documentId ||
        reference.endpoint ||
        isPdfReference(reference) ||
        isSourceDocumentContainerReference(reference) ||
        ((reference.kind === 'document' || reference.kind === 'artifact') && reference.runId),
    )

const isSourceNavigationGroup = (group: ReferenceDocumentGroup): boolean =>
    group.references.some(isSourceNavigationReference)

const WEAK_GRAPH_RELATION_TITLE_PATTERN =
    /^(?:appears?_with|co_occurs(?:_with)?|mentioned?_with|mentions?|related(?:_to|_text)?|contains(?:_text)?)$/i

const CONCRETE_RELATIONSHIP_EVIDENCE_TITLE_PATTERN =
    /\b(?:evidence|owns|highlights|connects|gates|blocks|required|approved|workflow)\b/i

const isWeakGraphRelationTitle = (value?: string): boolean =>
    WEAK_GRAPH_RELATION_TITLE_PATTERN.test(
        normalizeFacetText(value || '').replace(/\s+/g, '_'),
    )

const weakRelationshipEvidenceTitle = (reference: ChatReferenceItem): string => {
    if (!reference.relationship || !isWeakGraphRelationTitle(reference.relationship.relation)) return ''

    const relationshipTitle = normalizedPreviewKey(briefGraphRelationshipTitle(reference.relationship))
    const relationshipLead = normalizedPreviewKey(graphRelationshipEvidenceLead(reference.relationship))
    const candidates = uniqueStrings([
        reference.title,
        reference.body,
        reference.citationLabel,
    ])
        .map(value => cleanReferenceEvidencePreview(value || '', {
            ...reference.relationship,
            filename: reference.filename,
        }))
        .map(value => compactReferenceText(value, 160))
        .filter(value => {
            const key = normalizedPreviewKey(value)
            if (value.length < 18 || !CONCRETE_RELATIONSHIP_EVIDENCE_TITLE_PATTERN.test(value)) return false
            if (PREVIEW_PROMPT_META_PATTERN.test(value)) return false
            if (relationshipTitle && key === relationshipTitle) return false
            if (relationshipLead && key.includes(relationshipLead)) return false
            return !/(?:--|->|\bevidence=|\bprovenance=)/i.test(value)
        })

    return candidates[0] || ''
}

const requestedEvidenceRelationshipTitle = (value: string): string => {
    const match = String(value || '').replace(/\s+/g, ' ').match(
        /\bSource:\s*([^;|]+)[;|]\s*Relationship:\s*([^;|]+)[;|]\s*Target:\s*([^;|]+)/i,
    )
    if (!match) return ''
    const source = cleanGraphReferenceLabel(match[1], '')
    const relation = formatGraphReferenceRelation(match[2] || '')
    const target = cleanGraphReferenceLabel(match[3], '')
    return compactReferenceText([source, relation, target].filter(Boolean).join(' '), 120)
}

const readableEvidenceAssetName = (value: string): string => {
    const cleaned = cleanOfficePackageProvenanceText(String(value || ''))
        .split(/[\\/]/)
        .filter(Boolean)
        .pop()
        ?.replace(/\.[a-z0-9]{1,8}$/i, '') || ''
    if (!cleaned) return ''
    return formatReferenceFacet(cleaned).replace(/\b(?:Png|Jpe?g|Gif|Webp|Svg|Mp4|Mov|M4v|Mp3|Wav|Xlsx?|Csv|Pdf|Pptx?|Docx?)\b/g, '').replace(/\s+/g, ' ').trim()
}

const readableEvidenceMimeLabel = (value: string): string => {
    const mime = String(value || '').trim().toLowerCase()
    if (!mime) return ''
    if (mime === 'image/png') return 'PNG image'
    if (mime === 'image/jpeg' || mime === 'image/jpg') return 'JPEG image'
    if (mime === 'image/svg+xml') return 'SVG image'
    if (mime.startsWith('image/')) return 'Image'
    if (mime.startsWith('video/')) return 'Video'
    if (mime.startsWith('audio/')) return 'Audio'
    if (/spreadsheet|excel|xlsx/.test(mime)) return 'Spreadsheet'
    if (/pdf/.test(mime)) return 'PDF'
    return ''
}

export const referenceBriefTitle = (reference: ChatReferenceItem): string => {
    if (hasRequestedEvidenceFamilyMarker(reference)) {
        const title = reference.title || reference.citationLabel || ''
        const preview = cleanRequestedEvidencePreview(reference.body || reference.citationLabel || '', reference.filename)
        if (title && isCoordinateOnlyReferenceTitle(title) && preview) {
            return requestedEvidenceRelationshipTitle(reference.body || reference.citationLabel || '') ||
                compactReferenceText(preview.split(/(?<=[.!?])\s+/)[0] || preview, 120)
        }
        return compactReferenceText(title || preview || 'Evidence')
    }

    if (reference.relationship?.source || reference.relationship?.target) {
        const evidenceTitle = weakRelationshipEvidenceTitle(reference)
        if (evidenceTitle) return evidenceTitle
        return compactReferenceText(briefGraphRelationshipTitle(reference.relationship))
    }

    if (reference.path?.nodes && reference.path.nodes.length > 1) {
        const nodes = displayPathNodes(reference.path.nodes)
        return compactReferenceText(referencePathTitle(nodes), 120)
    }

    const title = reference.title.replace(/^(?:local pipeline|pipeline handoff)\s+/i, '')
    if (title && isCoordinateOnlyReferenceTitle(title)) {
        const relationshipTitle = requestedEvidenceRelationshipTitle(reference.body || reference.citationLabel || '')
        if (relationshipTitle) return relationshipTitle
        const preview = cleanReferenceEvidencePreview(reference.body || reference.citationLabel || '', {
            filename: reference.filename,
        })
        if (preview) return compactReferenceText(preview.split(/(?<=[.!?])\s+/)[0] || preview, 120)
    }

    return compactReferenceText(title)
}

export const referencePreviewText = (reference: ChatReferenceItem): string => {
    if (hasRequestedEvidenceFamilyMarker(reference)) {
        return compactReferenceText(
            cleanRequestedEvidencePreview(reference.body || reference.citationLabel || '', reference.filename),
            220,
        )
    }

    if (reference.relationship?.source || reference.relationship?.target) {
        if (isWeakGraphRelationTitle(reference.relationship.relation)) {
            const [concreteStatement] = relationshipConcreteEvidenceStatements(reference)
            if (concreteStatement) return compactReferenceText(concreteStatement, 180)
        }
        const evidenceTitle = weakRelationshipEvidenceTitle(reference)
        if (evidenceTitle) return compactReferenceText(evidenceTitle, 180)
        const leadText = graphRelationshipEvidenceLead(reference.relationship)
        const bodyText = cleanReferenceEvidencePreview(reference.body || '', {
            ...reference.relationship,
            filename: reference.filename,
        })
        const preview = bodyText && !bodyText.toLowerCase().includes(leadText.toLowerCase())
            ? `${leadText}. ${bodyText}`
            : bodyText || leadText
        return compactReferenceText(preview, 180)
    }

    if (reference.path?.nodes && reference.path.nodes.length > 1) {
        return compactReferenceText(buildReferencePathPreviewText(reference), 180)
    }

    const preview = cleanReferenceEvidencePreview(reference.body || '', { filename: reference.filename })
    if (isSourceDocumentContainerReference(reference) && isGeneratedSourceMetadataPreview(preview)) {
        return ''
    }

    return compactReferenceText(
        preview,
        220,
    )
}

export const requestedEvidenceTrayPreviewText = (reference: ChatReferenceItem): string => {
    const rawSource = String(reference.body || reference.citationLabel || '')
    const raw = cleanRequestedEvidencePreview(rawSource, reference.filename)
    if (!raw) return ''

    const normalizedRaw = raw.replace(/\s+/g, ' ').trim()
    const parts = uniqueStrings([
        ...rawSource.replace(/\s+/g, ' ').trim().split(/\s*;\s*/),
        ...normalizedRaw.split(/\s*;\s*/),
    ])
        .map(part => cleanOfficePackageProvenanceText(part)
            .replace(/\bForms\.CheckBox\.1\b/gi, 'Checkbox')
            .replace(/\bchecked\s*(?:true|yes|1)?\b/gi, 'Selected')
            .replace(/\bvalue\s+1\b/gi, 'Selected')
            .replace(/\s+/g, ' ')
            .replace(/^[\s:.)-]+|[\s:.)-]+$/g, '')
            .trim(),
        )
        .filter(Boolean)

    const field = (...labels: string[]) => {
        const patterns = labels.map(label => new RegExp(`^${label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*[:=]?\\s*(.+)$`, 'i'))
        for (const part of parts) {
            for (const pattern of patterns) {
                const match = part.match(pattern)
                const value = match?.[1]?.trim()
                if (value) return value
            }
        }
        return ''
    }

    const family = referenceEvidenceLabels(reference)[0] || referenceBriefTitle(reference)
    const chart = field('chart', 'chart title')
    const point = field('point', 'category', 'series name')
    const axis = field('axis')
    const value = field('value')
    const table = field('table')
    const cell = field('cell')
    const rowHeader = field('row header')
    const columnHeader = field('column header')
    const file = field('embedded file filename', 'embedded file name', 'image part', 'media part', 'filename')
    const mime = field('image mime', 'media mime', 'embedded file mime')
    const assetName = readableEvidenceAssetName(file)
    const mimeLabel = readableEvidenceMimeLabel(mime)

    const familyKey = normalizeFacetText(family)
    const compactParts =
        /\bchart/.test(familyKey)
            ? uniqueStrings([
                chart ? `${chart} chart` : '',
                point,
                axis && value ? `${axis}: ${value}` : value ? `Value ${value}` : axis,
            ])
        : /\btable/.test(familyKey)
            ? uniqueStrings([
                table ? `Table ${table}` : '',
                rowHeader && columnHeader ? `${rowHeader} / ${columnHeader}` : rowHeader || columnHeader,
                cell ? `Cell ${cell}` : '',
            ])
        : /\bform/.test(familyKey)
            ? uniqueStrings([
                normalizedRaw.match(/^\s*([^:;]{2,80})\s*:\s*(?:checked|selected|true|1)\b/i)?.[1]?.trim() || field('label', 'form field', 'field'),
                field('type') || (/checkbox/i.test(normalizedRaw) ? 'Checkbox' : ''),
                /(?:checked|selected|value\s+1)\b/i.test(normalizedRaw) ? 'Selected' : '',
            ])
        : /\bimages?\b/.test(familyKey)
            ? uniqueStrings([
                assetName ? `${assetName} image` : mimeLabel,
            ])
        : /\bmedia\b/.test(familyKey)
            ? uniqueStrings([
                assetName ? `${assetName} media` : mimeLabel,
            ])
        : /\b(?:attachments?|embedded)\b/.test(familyKey)
            ? uniqueStrings([
                assetName ? `${assetName} attachment` : mimeLabel,
            ])
        : []

    const compact = compactParts.filter(Boolean).slice(0, 3).join(' · ')
    return compactReferenceText(compact || normalizedRaw, 118)
}

const referenceTrayPreviewTextBase = (reference: ChatReferenceItem): string =>
    compactReferenceText(
        hasRequestedEvidenceFamilyMarker(reference)
            ? requestedEvidenceTrayPreviewText(reference)
            : reference.relationship
            ? graphRelationshipEvidenceLead(reference.relationship)
            : referencePreviewText(reference),
        170,
    )

const isRawScoreSubtitlePart = (value: string): boolean =>
    /\b(?:\d+(?:\.\d+)?%\s+(?:score|match)|score\s*[:=]\s*\d+(?:\.\d+)?|(?:path|graph|ranking)?\s*score\s+of\s+\d+(?:\.\d+)?)\b/i.test(value)

const isRawLayoutSubtitlePart = (value: string): boolean =>
    /\b(?:x|y|w|h)=\d+(?:\.\d+)?(?:in)?\b/i.test(value) ||
    /^(?:top|middle|bottom)_(?:left|center|right)$/i.test(value.trim())

const isLowValueReferenceTraySubtitlePart = (value: string): boolean =>
    /^(?:answer citation|primary path hop)$/i.test(value.trim()) ||
    /^(?:fill|stroke|line|border|color|font color|highlight)\s+#[0-9a-f]{3,8}\b/i.test(value.trim()) ||
    /^line\s+#[a-z0-9_-]+\b/i.test(value.trim())

const isRawOfficePackageSubtitlePart = (value: string, hasPackageProvenance: boolean): boolean => {
    const text = value.replace(/\s+/g, ' ').trim()
    if (!text) return true
    if (/^(?:media|document|workbook|chart|table|sheet|slide)\s+(?:part|embedded data|package|embedded package)$/i.test(text)) {
        return true
    }
    if (!hasPackageProvenance) return false
    if (/\b(?:slide|page|sheet|table|chart|form|styled|text|evidence|citation|path|hop)\b/i.test(text)) {
        return false
    }
    return /^[A-Z][A-Za-z0-9 &-]{1,42}$/.test(text)
}

const cleanReferenceTraySubtitlePart = (value: string): string =>
    (() => {
        const hasPackageProvenance =
            OFFICE_PACKAGE_PATH_FRAGMENT_TEST_PATTERN.test(value) ||
            /\b(?:embedded[_\s-]*package|(?:media|document|workbook|chart|table|sheet|slide)\s+part)\b/i.test(value)
        const rawParts = value.split(/\s*[|;]\s*/)
        const hasSlideScope = rawParts.some(part => /\bslide\s+\d+\s*:/i.test(part))
        return uniqueStrings(
            rawParts
                .map(part => {
                    const cleaned = part
                    .replace(/\b\d+(?:\.\d+)?%\s+match\b/gi, '')
                    .replace(/\bscope\s*:\s*\)?\s*/gi, '')
                    .replace(/\btable\s+(\d+)\b/gi, 'Table $1')
                    .replace(/\bcell\s+(R\d+C\d+)\b/gi, 'Cell $1')
                    .replace(/\bstyle\s+summary\s+bold\b/gi, 'Bold')
                    .replace(/\bstyle\s+summary\s+italic\b/gi, 'Italic')
                    .replace(/\bstyle\s+summary\s+underline(?:d)?\b/gi, 'Underlined')
                    .replace(/\bstyle\s+bold\b/gi, 'Bold')
                    .replace(/\bstyle\s+italic\b/gi, 'Italic')
                    .replace(/\bstyle\s+underline(?:d)?\b/gi, 'Underlined')
                    .replace(/\bfield\s+([^|;]+)/gi, '$1')
                    .replace(/\btype\s+Forms\.CheckBox\.1\b/gi, 'Checkbox')
                    .replace(/\bvalue\s+(true|false)\b/gi, (_match, value: string) =>
                        value.toLowerCase() === 'true' ? 'Selected' : 'Not selected',
                    )
                    .replace(/\bchecked\s+checked\b/gi, 'Selected')
                    .replace(/\bchecked\s+(?:true|yes)\b/gi, 'Selected')
                    .replace(/\bchecked\s+(?:false|no)\b/gi, 'Not selected')
                    .replace(/\s+/g, ' ')
                    .replace(/^[\s:.)-]+|[\s:.)-]+$/g, '')
                    .trim()
                    if (hasSlideScope && /^\d+$/.test(cleaned)) return `Table ${cleaned}`
                    return cleaned
                })
                .filter(part => part && !isRawLayoutSubtitlePart(part))
                .map(cleanOfficePackageProvenanceText)
                .map(humanizeVisibleGraphArrows)
                .filter(part =>
                    part &&
                    !/^(?:scope|match)$/i.test(part) &&
                    !isLowValueReferenceTraySubtitlePart(part) &&
                    !isRawScoreSubtitlePart(part) &&
                    !isRawOfficePackageSubtitlePart(part, hasPackageProvenance),
                ),
        )
            .map(part => part
                .replace(/^slide\s+(\d+)\s*:\s*[A-Za-z]{1,4}\.?$/i, 'Slide $1')
                .replace(/^slide\s+(\d+)/i, 'Slide $1')
                .replace(/\bvalue\s+([^|;]+)/gi, 'Value $1'),
            )
            .join(' | ')
            .trim()
    })()

export const buildReferenceTraySubtitle = (reference: ChatReferenceItem): string =>
    uniqueStrings((reference.subtitle || '').split(/\s*·\s*/))
        .map(cleanReferenceTraySubtitlePart)
        .filter(part => {
            if (!part || isRawScoreSubtitlePart(part) || isLowValueReferenceTraySubtitlePart(part)) return false
            return !reference.filename || part.toLowerCase() !== referenceBasename(reference.filename).toLowerCase()
        })
        .join(' · ') || referenceKindLabel(reference)

export const buildReferencePathEvidenceLabels = (reference?: ChatReferenceItem | null): string[] =>
    uniqueStrings([
        reference?.path?.hops !== undefined ? `${reference.path.hops} hops` : '',
    ])

export type DisplayPathRelationship = NonNullable<NonNullable<ChatReferenceItem['path']>['relationships']>[number]

export const displayPathRelationships = (
    relationships?: NonNullable<ChatReferenceItem['path']>['relationships'],
): DisplayPathRelationship[] => {
    const seen = new Set<string>()
    const output: DisplayPathRelationship[] = []

    ;(relationships || []).forEach(relationship => {
        const source = cleanGraphReferenceEndpointLabel(relationship.source, '').toLowerCase()
        const relation = formatGraphReferenceRelation(relationship.relation).toLowerCase()
        const target = cleanGraphReferenceEndpointLabel(relationship.target, '').toLowerCase()
        const key = [source, relation, target].join('|')
        if (!source || !target || seen.has(key)) return
        seen.add(key)
        output.push(relationship)
    })

    return output
}

export const buildReferencePathPreviewText = (reference?: ChatReferenceItem | null): string => {
    const relationships = displayPathRelationships(reference?.path?.relationships)
    if (relationships.length > 0) {
        const hops = relationships.slice(0, 3).map(relationship =>
            [
                cleanGraphReferenceEndpointLabel(relationship.source, 'Source'),
                formatGraphReferenceRelation(relationship.relation),
                cleanGraphReferenceEndpointLabel(relationship.target, 'Target'),
            ].join(' '),
        )
        const suffix = relationships.length > hops.length ? ` · +${relationships.length - hops.length} hops` : ''
        return `${hops.join(' · ')}${suffix}`
    }

    const nodes = displayPathNodes(reference?.path?.nodes)
    return nodes.length > 1 ? referencePathTitle(nodes) : ''
}

export interface ReferenceDocumentPreviewSummary {
    eyebrow: string
    title: string
    subtitle: string
    excerpt: string
    chips: string[]
    outline: string[]
}

export interface ReferenceLayoutMapBox {
    id: string
    label: string
    detail: string
    left: number
    top: number
    width: number
    height: number
    tone: 'relationship' | 'chart' | 'shape'
}

export interface ReferenceLayoutMapSummary {
    title: string
    subtitle: string
    boxes: ReferenceLayoutMapBox[]
}

export interface ReferenceLayoutMapViewport {
    left: number
    top: number
    width: number
    height: number
    focused: boolean
}

const DEFAULT_SLIDE_WIDTH_INCHES = 10
const DEFAULT_SLIDE_HEIGHT_INCHES = 5.625

const parseLayoutRegionValue = (value: string): Pick<ReferenceLayoutMapBox, 'left' | 'top' | 'width' | 'height'> | null => {
    const text = String(value || '').trim()
    if (!text) return null

    const named = ['x', 'y', 'w', 'h'].map(key => {
        const match = text.match(new RegExp(`\\b${key}\\s*=\\s*(-?\\d+(?:\\.\\d+)?)`, 'i'))
        return match ? Number(match[1]) : Number.NaN
    })
    const rawNumbers = named.every(Number.isFinite)
        ? named
        : text.match(/-?\d+(?:\.\d+)?/g)?.map(Number).filter(Number.isFinite).slice(0, 4) || []
    if (rawNumbers.length < 4) return null

    const [leftRaw, topRaw, widthRaw, heightRaw] = rawNumbers
    const hasInchUnits = /(?:\d|\.)\s*in(?:ch(?:es)?)?\b/i.test(text)
    const percentLike = !hasInchUnits && Math.max(...rawNumbers.map(Math.abs)) <= 100
    const left = percentLike ? leftRaw : (leftRaw / DEFAULT_SLIDE_WIDTH_INCHES) * 100
    const top = percentLike ? topRaw : (topRaw / DEFAULT_SLIDE_HEIGHT_INCHES) * 100
    const width = percentLike ? widthRaw : (widthRaw / DEFAULT_SLIDE_WIDTH_INCHES) * 100
    const height = percentLike ? heightRaw : (heightRaw / DEFAULT_SLIDE_HEIGHT_INCHES) * 100

    return {
        left: clampNumber(left, 0, 100),
        top: clampNumber(top, 0, 100),
        width: clampNumber(width, 4, 100),
        height: clampNumber(height, 4, 100),
    }
}

const layoutMapLabelKey = (value: string): string =>
    value.trim().toLowerCase()

const LAYOUT_BOX_METADATA_KEYS = new Set([
    'source label',
    'target label',
    'connector',
    'connector title',
    'connector note',
    'shape',
    'shape title',
    'shape note',
    'chart',
    'chart title',
    'table',
    'row',
    'cell',
])

const layoutFacetLabel = (prefix: 'Table' | 'row' | 'Cell', value?: string): string => {
    const text = cleanReferenceEvidencePreview(String(value || ''))
    if (!text) return ''
    if (prefix === 'Table' && /^table\b/i.test(text)) return text.replace(/^table\b/i, 'Table')
    if (prefix === 'row' && /^row\b/i.test(text)) return text.replace(/^row\b/i, 'row')
    if (prefix === 'Cell' && /^cell\b/i.test(text)) return text.replace(/^cell\b/i, 'Cell')
    if (prefix === 'Cell') {
        const cellMatch = text.match(/\b(R\d+C\d+)\b/i)
        if (cellMatch) return `Cell ${cellMatch[1].toUpperCase()}`
    }
    if (/^\d+$/.test(text)) return `${prefix} ${text}`
    return text
}

const layoutTableEvidenceLabel = (values: Record<string, string>): string =>
    uniqueStrings([
        layoutFacetLabel('Table', values.table),
        layoutFacetLabel('row', values.row),
        layoutFacetLabel('Cell', values.cell),
    ]).join(' ')

const buildLayoutBoxLabel = (
    box: ReferenceLayoutMapBox & { values: Record<string, string> },
    reference: ChatReferenceItem,
    index: number,
): ReferenceLayoutMapBox => {
    const pathRelationships = reference.path?.relationships || []
    const hop = pathRelationships[index]
    const source = cleanGraphReferenceEndpointLabel(box.values['source label'] || reference.relationship?.source || '', '')
    const target = cleanGraphReferenceEndpointLabel(box.values['target label'] || reference.relationship?.target || '', '')
    const hopSource = cleanGraphReferenceEndpointLabel(hop?.source || '', '')
    const hopTarget = cleanGraphReferenceEndpointLabel(hop?.target || '', '')
    const connectorTitle = box.values['connector title'] || box.values.connector || ''
    const connectorNote = box.values['connector note'] || ''
    const shapeTitle = box.values['shape title'] || box.values.shape || ''
    const chartTitle = cleanGraphReferenceLabel(box.values['chart title'] || box.values.chart || '', '')
    const tableLabel = layoutTableEvidenceLabel(box.values)
    const label = compactReferenceText(
        (source && target ? `${source} to ${target}` : '') ||
        (hopSource && hopTarget ? `${hopSource} to ${hopTarget}` : '') ||
        connectorTitle ||
        shapeTitle ||
        chartTitle ||
        tableLabel ||
        reference.relationship?.source ||
        (reference.kind === 'path' ? '' : reference.title) ||
        `Evidence region ${index + 1}`,
        42,
    )
    const detail = compactReferenceText(
        connectorNote ||
        hop?.evidence ||
        box.values['shape note'] ||
        chartTitle ||
        tableLabel ||
        reference.relationship?.relation ||
        hop?.relation ||
        '',
        86,
    )
    const tone: ReferenceLayoutMapBox['tone'] =
        source || target || connectorTitle || connectorNote
            ? 'relationship'
            : chartTitle
                ? 'chart'
                : 'shape'

    return {
        ...box,
        label,
        detail,
        tone,
    }
}

export const buildReferenceLayoutMap = (reference: ChatReferenceItem): ReferenceLayoutMapSummary | null => {
    const locations = reference.sourceLocations || []
    let slide = ''
    let slideTitle = ''
    let file = reference.filename ? referenceBasename(reference.filename) : ''
    const boxes: Array<ReferenceLayoutMapBox & { values: Record<string, string> }> = []
    let current: (ReferenceLayoutMapBox & { values: Record<string, string> }) | null = null
    let pendingBoxValues: Record<string, string> = {}

    locations.forEach(location => {
        const labelKey = layoutMapLabelKey(location.label)
        const value = cleanSourceLocationValue(location.value)
        if (!value) return

        if (labelKey === 'file' && !file) file = referenceBasename(value)
        if (labelKey === 'slide' && !slide) slide = value
        if (labelKey === 'slide title' && !slideTitle) slideTitle = value

        if (labelKey === 'region') {
            const region = parseLayoutRegionValue(value)
            if (!region) return
            current = {
                id: `layout-region-${boxes.length}`,
                label: '',
                detail: '',
                tone: 'shape',
                values: { ...pendingBoxValues },
                ...region,
            }
            pendingBoxValues = {}
            boxes.push(current)
            return
        }

        if (!current && LAYOUT_BOX_METADATA_KEYS.has(labelKey)) {
            pendingBoxValues[labelKey] = value
            return
        }

        if (!current) return
        if (LAYOUT_BOX_METADATA_KEYS.has(labelKey)) {
            current.values[labelKey] = value
        }
    })

    const mappedBoxes = boxes
        .map((box, index) => buildLayoutBoxLabel(box, reference, index))
        .filter(box => box.width > 0 && box.height > 0)
        .slice(0, 6)
        .map((box, index, visibleBoxes) => {
            const labelKey = normalizedPreviewKey(box.label)
            const duplicateLabel = labelKey && visibleBoxes
                .slice(0, index)
                .some(previous => normalizedPreviewKey(previous.label) === labelKey)
            return duplicateLabel
                ? {
                    ...box,
                    label: `Evidence region ${index + 1}`,
                }
                : box
        })
    if (mappedBoxes.length === 0) return null

    return {
        title: slide && slideTitle ? `Slide ${slide}: ${slideTitle}` : slideTitle || (slide ? `Slide ${slide}` : 'Evidence layout'),
        subtitle: uniqueStrings([file, formatReferenceCount(mappedBoxes.length, 'region')]).join(' · '),
        boxes: mappedBoxes,
    }
}

const referenceLocationValueByLabel = (
    locations: ChatReferenceLocation[],
    labels: string[],
): string => {
    const normalizedLabels = new Set(labels.map(label => label.toLowerCase()))
    return locations.find(location => normalizedLabels.has(location.label.toLowerCase()))?.value || ''
}

const referenceAnchorLabel = (locations: ChatReferenceLocation[]): string => {
    const slide = referenceLocationValueByLabel(locations, ['Slide'])
    const slideTitle = referenceLocationValueByLabel(locations, ['Slide title'])
    if (slide && slideTitle) return `Slide ${slide}: ${slideTitle}`
    if (slide) return `Slide ${slide}`
    if (slideTitle) return slideTitle

    const page = referenceLocationValueByLabel(locations, ['Page'])
    const section = referenceLocationValueByLabel(locations, ['Section'])
    if (page && section) return `Page ${page}: ${section}`
    if (page) return `Page ${page}`
    if (section) return section

    const sheet = referenceLocationValueByLabel(locations, ['Sheet'])
    const table = referenceLocationValueByLabel(locations, ['Table'])
    if (sheet && table) return `${sheet} / ${table}`
    return sheet || table || ''
}

const cleanDocumentPreviewLine = (value: string): string =>
    cleanReferenceEvidencePreview(
        value
            .replace(/^#{1,6}\s*/, '')
            .replace(/^\s*(?:[-*]|\d+\.)\s*/, '')
            .replace(/^\[[^\]]+\]\s*/i, '')
            .replace(/^(?:scope|evidence types?|provenance|location|source locations?)\s*:\s*/i, '')
            .replace(/\s*;\s*(?:visibility|parse_status|parse status|slide_part|relationship_id|target_mode)[:=]\s*[^;)]+/gi, '')
            .trim(),
    )

const documentPreviewExcerpt = (
    text: string,
    reference: ChatReferenceItem,
    anchorLabel: string,
    query?: string,
): string => {
    const queryPhraseAnchors = previewQueryPhrases(query)
    const queryTermAnchors = previewQueryWords(query).map(value => normalizedPreviewKey(value)).filter(value => value.length >= 4)
    const queryAnchors = uniqueStrings([...queryPhraseAnchors, ...queryTermAnchors])
    const anchors = uniqueStrings([
        ...queryAnchors,
        anchorLabel,
        reference.relationship?.source,
        reference.relationship?.target,
        ...(reference.path?.nodes || []),
        ...(reference.matchedTerms || []),
    ])
        .map(value => normalizedPreviewKey(value))
        .filter(value => value.length >= 4)
    const queryAnchorSet = new Set(queryAnchors)
    const queryPhraseAnchorSet = new Set(queryPhraseAnchors)
    const previewSegments = (value: string): string[] => {
        const normalized = value.replace(/\s+/g, ' ').trim()
        if (!normalized) return []
        const tagged = normalized
            .split(/(?=\[[^\]]+\]\s+(?:page|slide|table|chart)\s+\d+\b)/i)
            .map(part => part.trim())
            .filter(Boolean)
        return tagged.length > 1 ? tagged : [value]
    }
    const excerptAroundAnchor = (line: string): string => {
        const searchableLine = line.toLowerCase().replace(/[_-]+/g, ' ')
        const queryMatchIndex = queryAnchors
            .map(anchor => searchableLine.indexOf(anchor))
            .filter(index => index >= 0)
            .sort((left, right) => left - right)[0]
        const anchorTerms = uniqueStrings(
            anchors.flatMap(anchor => [anchor, ...anchor.split(/\s+/)])
                .map(anchor => anchor.trim().toLowerCase())
                .filter(anchor => anchor.length >= 4),
        )
        const anchorMatchIndex = anchorTerms
            .map(anchor => searchableLine.indexOf(anchor))
            .filter(index => index >= 0)
            .sort((left, right) => left - right)[0]
        const matchIndex = queryMatchIndex ?? anchorMatchIndex

        if (matchIndex === undefined || line.length <= 340) return line

        const start = Math.max(0, matchIndex - (queryMatchIndex !== undefined ? 72 : 96))
        const end = Math.min(line.length, start + 340)
        const excerpt = line.slice(start, end).trim().replace(/^[=\s.:;-]+/, '')
        return `${start > 0 ? '... ' : ''}${excerpt}${end < line.length ? ' ...' : ''}`
    }
    const anchorScore = (key: string, anchor: string): number => {
        if (!key.includes(anchor)) return 0
        if (queryPhraseAnchorSet.has(anchor)) {
            return 8 + Math.min(6, anchor.split(/\s+/).filter(Boolean).length * 2)
        }
        return queryAnchorSet.has(anchor) ? 2 : 1
    }

    const rows = text
        .split('\n')
        .flatMap(previewSegments)
        .map(cleanDocumentPreviewLine)
        .filter(line =>
            line.length >= 18 &&
            !/^structured graphrag ingest\b/i.test(line) &&
            !/^filename\b/i.test(line) &&
            !/^extraction method\b/i.test(line),
        )
        .map(line => {
            const key = normalizedPreviewKey(line)
            const promptMetaPenalty = PREVIEW_PROMPT_META_PATTERN.test(line) ? -500 : 0
            const score = anchors.reduce(
                (total, anchor) => total + anchorScore(key, anchor),
                0,
            ) + promptMetaPenalty
            return { line: score > 0 ? excerptAroundAnchor(line) : line, score }
        })
        .sort((left, right) => right.score - left.score || left.line.length - right.line.length)

    const usefulRows = rows.filter(row => row.score > 0)
    return compactReferenceText(
        (usefulRows.length > 0 ? usefulRows : rows)
            .slice(0, 2)
            .map(row => row.line)
            .join(' '),
        360,
    )
}

export const referencePreviewTextForQuery = (
    reference: ChatReferenceItem,
    query?: string,
): string => {
    if (hasRequestedEvidenceFamilyMarker(reference)) {
        return requestedEvidenceTrayPreviewText(reference) || referencePreviewText(reference)
    }

    if (reference.kind === 'passage' && query) {
        const excerpt = documentPreviewExcerpt(reference.body || '', reference, '', query)
        if (excerpt) return excerpt
    }

    return referencePreviewText(reference)
}

const rawEvidencePreviewSegments = (value: string): string[] => {
    const normalized = String(value || '').replace(/\r/g, '\n').trim()
    if (!normalized) return []

    return normalized
        .split(/\n+/)
        .flatMap(line =>
            line
                .split(/(?=\[[^\]]+\]\s+(?:page|slide|table|chart)\s+\d+\b)/i)
                .flatMap(part =>
                    part.split(
                        /(?=\b(?:page|slide|table|chart)\s+\d+(?:\s+(?:block|paragraph|line|connector|chart|table|row|shape|control)\s+\d+(?:\.\d+)?)?\s*:)/i,
                    ),
                )
                .flatMap(part => part.split(/(?<=[.!?])\s+(?=[A-Z[])/)),
        )
        .map(part => part.replace(/\s+/g, ' ').trim())
        .filter(Boolean)
}

const RELATIONSHIP_EVIDENCE_VERB_PATTERN =
    /\b(?:evidence|owns|highlights|connects|gates|blocks|required|approved|approval|workflow|review|risk|map|launch)\b/i

const RELATIONSHIP_EVIDENCE_ACTION_PATTERN =
    /\b(?:evidence|owns|highlights|connects|gates|blocks|required|approved|workflow)\b/i

const stripRepeatedLeadingWordSequence = (value: string): string => {
    const words = value.split(/\s+/).filter(Boolean)
    for (let width = Math.min(8, Math.floor(words.length / 2)); width >= 1; width -= 1) {
        const first = words.slice(0, width).join(' ').toLowerCase()
        const second = words.slice(width, width * 2).join(' ').toLowerCase()
        if (first && first === second) {
            return words.slice(width).join(' ')
        }
    }
    return value
}

const stripLeadingGraphArrowEvidencePrefix = (
    value: string,
    relationship?: ChatReferenceItem['relationship'],
): string => {
    const text = value
        .replace(
            /^.*?(?:--\[[^\]]+\]-->|--[^;.\n]*?->|->)\s*.*?(?=\b(?:page|slide|table|chart)\s+\d+\b)/i,
            '',
        )
        .trim()
    const source = relationship?.source ? escapeRegExp(relationship.source.trim()) : ''
    const target = relationship?.target ? escapeRegExp(relationship.target.trim()) : ''
    if (source && target) {
        const targetPrefixedEvidence = text.match(
            new RegExp(`^\\s*${source}\\s*(?:--\\[[^\\]]+\\]-->|--[^;.\\n]*?->|->)\\s*${target}\\s+(.+)$`, 'i'),
        )?.[1]?.trim()
        if (targetPrefixedEvidence && targetPrefixedEvidence.length >= 18) {
            return targetPrefixedEvidence
        }
    }
    const afterArrow = text.match(/(?:--\[[^\]]+\]-->|--[^;.\n]*?->|->)\s*(.+)$/i)?.[1]?.trim()
    if (afterArrow) {
        const evidence = stripRepeatedLeadingWordSequence(afterArrow).trim()
        if (evidence.length >= 18 && RELATIONSHIP_EVIDENCE_ACTION_PATTERN.test(evidence)) {
            return evidence
        }
    }
    return text
}

const relationshipEvidenceTextScore = (
    text: string,
    reference: ChatReferenceItem,
    query?: string,
): number => {
    const key = normalizedPreviewKey(text)
    const queryAnchors = previewQueryPhrases(query)
    const queryWordsForScore = previewQueryWords(query).map(normalizedPreviewKey)
    const focusTerms = uniqueStrings([
        reference.relationship?.source,
        reference.relationship?.target,
        ...(reference.matchedTerms || []),
    ])
        .map(normalizedPreviewKey)
        .filter(value => value.length >= 4)

    let score = 0
    if (RELATIONSHIP_EVIDENCE_VERB_PATTERN.test(text)) {
        score += 8
    }
    queryAnchors.forEach(anchor => {
        if (anchor && key.includes(anchor)) score += 6
    })
    queryWordsForScore.forEach(anchor => {
        if (anchor && key.includes(anchor)) score += 2
    })
    focusTerms.forEach(term => {
        if (key.includes(term)) score += 1
    })
    if (/\b(?:page|slide|table|chart)\s+\d+/i.test(text)) score += 1
    if (PREVIEW_PROMPT_META_PATTERN.test(text)) score -= 500
    return score
}

export const relationshipConcreteEvidenceStatements = (
    reference: ChatReferenceItem | null,
    query?: string,
): string[] => {
    if (!reference?.relationship) return []

    const relationshipLead = normalizedPreviewKey(graphRelationshipEvidenceLead(reference.relationship))
    const relationshipTitle = normalizedPreviewKey(briefGraphRelationshipTitle(reference.relationship))
    const sourceKey = normalizedPreviewKey(reference.relationship.source || '')
    const targetKey = normalizedPreviewKey(reference.relationship.target || '')
    const sourceLocationEvidence = (reference.sourceLocations || [])
        .filter(location =>
            /\b(?:evidence|text|connector note|shape note|cell text|chart title|speaker notes?|comment|paragraph|block)\b/i.test(location.label),
        )
        .map(location => location.value)

    const candidates = uniqueStrings([
        ...sourceLocationEvidence,
        reference.body,
        reference.citationLabel,
    ])
        .flatMap(rawEvidencePreviewSegments)
        .map(value => ({
            raw: value,
            value: cleanPdfEvidenceHighlightPhrase(stripLeadingGraphArrowEvidencePrefix(value, reference.relationship)),
        }))
        .filter(({ raw, value }) => {
            const key = normalizedPreviewKey(value)
            if (value.length < 18 || !key) return false
            if (PREVIEW_PROMPT_META_PATTERN.test(value)) return false
            if (
                /(?:--|->)/i.test(raw) &&
                !/\b(?:page|slide|table|chart)\s+\d+\b/i.test(raw) &&
                !RELATIONSHIP_EVIDENCE_ACTION_PATTERN.test(value)
            ) {
                return false
            }
            if (/(?:--|->|\bevidence=|\bprovenance=)/i.test(value)) return false
            if (relationshipLead && (key === relationshipLead || key.includes(relationshipLead))) return false
            if (relationshipTitle && key === relationshipTitle) return false
            if (sourceKey && targetKey && key === `${sourceKey} ${targetKey}`) return false
            return /[a-z]/i.test(value)
        })
        .map(({ value }) => ({
            value: compactReferenceText(value, 220),
            score: relationshipEvidenceTextScore(value, reference, query),
        }))
        .sort((left, right) => right.score - left.score || left.value.length - right.value.length)

    return uniqueStrings(candidates.map(candidate => candidate.value)).slice(0, 4)
}

const referenceTrayPreviewText = (
    reference: ChatReferenceItem,
    query?: string,
): string => {
    if (
        !hasRequestedEvidenceFamilyMarker(reference) &&
        !reference.relationship &&
        query &&
        (reference.kind === 'passage' || isPdfReference(reference) || isSourceDocumentContainerReference(reference))
    ) {
        const excerpt = documentPreviewExcerpt(reference.body || '', reference, '', query)
        if (excerpt) return compactReferenceText(excerpt, 170)
    }

    return referenceTrayPreviewTextBase(reference)
}

const highSignalPreviewChips = (
    labels: string[],
    maxCount: number,
): string[] =>
    uniqueStrings(labels)
        .filter(label => {
            const normalized = normalizeFacetText(label)
            return (
                !normalized ||
                !/\b(?:file|region|visibility|slide|page|sheet|source brief|part|relationship id|source shape|target shape|zone)\b/.test(normalized)
            )
        })
        .slice(0, maxCount)

export const buildReferenceDocumentPreviewSummary = (
    reference: ChatReferenceItem,
    options: {
        documentText?: string
        fallbackPreview?: string
        outline?: string[]
        evidenceLabels?: string[]
        locationLabels?: string[]
        sourceTypeLabel?: string
        query?: string
    } = {},
): ReferenceDocumentPreviewSummary => {
    const locations = referenceStructuredLocations(reference)
    const sourceType = options.sourceTypeLabel || friendlyReferenceSourceType(reference) || referenceKindLabel(reference)
    const anchor = referenceAnchorLabel(locations)
    const sourceTitle = referenceBasename(
        reference.filename ||
        referenceLocationValueByLabel(locations, ['File']) ||
        referenceDocumentTitle(reference),
    )
    const connectorNote = referenceLocationValueByLabel(locations, ['Connector note'])
    const shapeNote = referenceLocationValueByLabel(locations, ['Shape note'])
    const chartTitle = referenceLocationValueByLabel(locations, ['Chart title', 'Chart'])
    const cellText = referenceLocationValueByLabel(locations, ['Cell text', 'Cell'])
    const rawDocumentExcerpt = documentPreviewExcerpt(options.documentText || '', reference, anchor, options.query)
    const documentExcerpt =
        isSourceDocumentContainerReference(reference) && isGeneratedSourceMetadataPreview(rawDocumentExcerpt)
            ? ''
            : rawDocumentExcerpt
    const requestedEvidence = hasRequestedEvidenceFamilyMarker(reference)
    const rawFallbackPreview =
        requestedEvidence
            ? cleanRequestedEvidencePreview(
                options.fallbackPreview || reference.body || reference.citationLabel || '',
                reference.filename,
            )
            : reference.kind === 'path' && documentExcerpt
            ? ''
            : cleanReferenceEvidencePreview(options.fallbackPreview || '')
    const fallbackPreview =
        isSourceDocumentContainerReference(reference) && isGeneratedSourceMetadataPreview(rawFallbackPreview)
            ? ''
            : rawFallbackPreview
    const concreteRelationshipEvidencePreview =
        reference.relationship && isWeakGraphRelationTitle(reference.relationship.relation)
            ? relationshipConcreteEvidenceStatements(reference, options.query)[0] || ''
            : ''
    const queryCenteredPassageExcerpt =
        reference.kind === 'passage' && options.query
            ? fallbackPreview || documentExcerpt
            : ''
    const excerpt = compactReferenceText(
        requestedEvidence
            ? fallbackPreview || documentExcerpt
            : connectorNote ||
              shapeNote ||
              (chartTitle && !anchor.includes(chartTitle) ? chartTitle : '') ||
              cellText ||
              concreteRelationshipEvidencePreview ||
              queryCenteredPassageExcerpt ||
              fallbackPreview ||
              documentExcerpt,
        360,
    )
    const relationshipLabel = reference.relationship
        ? compactReferenceText(briefGraphRelationshipTitle(reference.relationship), 96)
        : ''
    const focusTerms = uniqueStrings([
        reference.relationship?.source,
        reference.relationship?.target,
        ...(reference.path?.nodes || []),
        ...(reference.matchedTerms || []),
    ])
        .map(value => normalizedPreviewKey(value))
        .filter(value => value.length >= 4)
    const focusedOutline = uniqueStrings(options.outline || [])
        .filter(item => {
            const key = normalizedPreviewKey(item)
            if (isLowSignalDocumentOutlineItem(item)) return false
            if (!key || /\b(?:source brief|sheet|workbook|title|https|uri)\b/.test(key)) return false
            return focusTerms.length === 0 || focusTerms.some(term => key.includes(term))
        })
    const excerptKey = normalizedPreviewKey(excerpt)
    const relationshipCoveredByExcerpt = Boolean(
        reference.relationship?.source &&
        reference.relationship?.target &&
        excerptKey.includes(normalizedPreviewKey(reference.relationship.source)) &&
        excerptKey.includes(normalizedPreviewKey(reference.relationship.target)),
    )

    return {
        eyebrow: anchor ? sourceType : 'Document preview',
        title: requestedEvidence
            ? referenceBriefTitle(reference)
            : anchor || referenceDocumentTitle(reference),
        subtitle: uniqueStrings([
            requestedEvidence
                ? sourceTitle
                : sourceTitle !== (anchor || referenceDocumentTitle(reference)) ? sourceTitle : '',
            requestedEvidence
                ? anchor
                : relationshipLabel && !isRedundantPreview(relationshipLabel, anchor || '') ? relationshipLabel : '',
        ]).join(' · '),
        excerpt,
        chips: highSignalPreviewChips([
            ...(options.evidenceLabels || referenceEvidenceLabels(reference)),
            ...(options.locationLabels || referenceLocationLabels(reference)),
        ], 4),
        outline: highSignalPreviewChips([
            relationshipCoveredByExcerpt ? '' : relationshipLabel,
            ...focusedOutline,
        ], 3).filter(item => !excerpt || !isRedundantPreview(item, excerpt)),
    }
}

const referenceSearchText = (reference: ChatReferenceItem): string =>
    uniqueStrings([
        reference.title,
        reference.subtitle,
        reference.body,
        reference.filename,
        reference.relationship?.source,
        reference.relationship?.relation,
        reference.relationship?.target,
        ...(reference.evidenceFamilies || []),
        ...(reference.matchedTerms || []),
        ...(reference.locationHints || []),
        ...(reference.sourceLocations || []).flatMap(location => [
            location.label,
            location.value,
            location.category,
        ]),
        ...(reference.path?.nodes || []),
        ...(reference.path?.relationships || []).flatMap(relationship => [
            relationship.source,
            relationship.relation,
            relationship.target,
            relationship.evidence,
        ]),
    ]).join(' ').toLowerCase()

const queryConnectionPhrases = (query?: string): string[] => {
    const text = String(query || '').replace(/\s+/g, ' ').trim()
    if (!text) return []

    const phrasePairs: string[] = []
    const addPair = (left?: string, right?: string) => {
        [left, right].forEach(value => {
            const cleaned = String(value || '')
                .replace(/\b(?:use|using|cite|with|from|through|via|and)\b.*$/i, '')
                .replace(/[?.!,;:]+$/g, '')
                .trim()
            if (cleaned.length >= 3) phrasePairs.push(cleaned)
        })
    }

    const connectMatch = text.match(/\b(?:how\s+does|how\s+do|trace|connect|connecting|connection\s+between)\b\s+(.+?)\s+(?:connect(?:s|ed)?\s+to|relate(?:s|d)?\s+to|with|and|->|→)\s+(.+?)(?:[?.]|$)/i)
    if (connectMatch) addPair(connectMatch[1], connectMatch[2])

    const arrowMatch = text.match(/\b(.+?)\s*(?:->|→)\s*(.+?)(?:[?.]|$)/)
    if (arrowMatch) addPair(arrowMatch[1], arrowMatch[2])

    return uniqueStrings(phrasePairs)
        .map(phrase => phrase.replace(/^["'`]+|["'`]+$/g, '').trim())
        .filter(phrase => phrase.split(/\s+/).some(term => !LOW_VALUE_SOURCE_QUERY_TERMS.has(term.toLowerCase())))
        .slice(0, 2)
}

const queryPromptText = (query?: string): string => {
    const lines = String(query || '')
        .split(/\n+/)
        .map(line => line.replace(/\s+/g, ' ').trim())
        .filter(Boolean)
    return lines.length > 0 ? lines[lines.length - 1] : ''
}

const relationshipCitationQuery = (query?: string): boolean =>
    /\b(?:cite|citation|connect|connects|connection|direct|gate|gates|href|link|linked|links|relationship|relationships|source|sources|support|supports|supported|uri|url|visual|visuals)\b/i.test(String(query || ''))

const graphRAGSummaryQuestionQuery = (query?: string): boolean => {
    const text = String(query || '').replace(/\s+/g, ' ').trim()
    if (!text) return false
    return (
        /\bsummar(?:y|ize|ise|izes|ises|ized|ised|izing|ising)\b/i.test(text) ||
        /\bkey evidence,\s*entities,\s*relationships\b/i.test(text) ||
        /\bnamed Entities,\s*Top relationships,\s*Best path,\s*and Source proof\b/i.test(text) ||
        /\bsource proof\b/i.test(text)
    )
}

const graphRelationshipQuestionQuery = (query?: string): boolean => {
    const text = String(query || '').replace(/\s+/g, ' ').trim()
    if (!text) return false

    return (
        /\b(?:graph\s+)?(?:edge|edges|relation|relations|relationship|relationships)\b/i.test(text) ||
        /\b(?:bridge|bridges|connect|connected|connection|connections|gate|gates|link|linked|links|visual|visuals)\b/i.test(text) ||
        /(?:--|\b[A-Za-z0-9][^.\n]{0,140}?)(?:->|→)/.test(text)
    )
}

const linkedEvidenceQuery = (query?: string): boolean =>
    /\b(?:href|link|linked|links|url|uri)\b|\bevidence\s+(?:link|packet|source|url)\b|\bopen\s+evidence\b/i.test(String(query || ''))

const graphPathQuestionQuery = (query?: string): boolean => {
    const text = queryPromptText(query)
    if (!text) return false

    return (
        /\b(?:path|paths|trace|traces|hop|hops|chain|through|via)\b/i.test(text) ||
        /\bhow\s+(?:does|do)\b.+\bconnect(?:s|ed|ing)?\s+to\b/i.test(text) ||
        (/\b(?:explain|find|show)\b/i.test(text) && queryConnectionPhrases(text).length > 0)
    )
}

const requestedEvidenceFamilyQuestionQuery = (query?: string): boolean => {
    const text = String(query || '').replace(/\s+/g, ' ').trim()
    if (!text) return false

    return (
        /\b(?:attachment|attachments|chart|charts|comment|comments|connector|connectors|arrow|arrows|embedded file|embedded files|flow|flows|form field|form fields|image|images|media|speaker note|speaker notes|styled text|table|tables)\b/i.test(text) ||
        /\b(?:available|each|every|extracted|list|show)\s+(?:evidence\s+)?(?:families|family|kinds|types)\b/i.test(text) ||
        /\b(?:evidence\s+)?(?:families|family|kinds|types)\s+(?:available|extracted|present)\b/i.test(text)
    )
}

const requestedEvidenceFamilyInspectionQuery = (query?: string): boolean => {
    const text = queryPromptText(query)
    if (!text || !requestedEvidenceFamilyQuestionQuery(text)) return false
    if (requestedEvidenceFamilyOrder(text).size < 3) return false
    return /\b(?:audit|cite|extract|extracted|extraction|inspect|list|review|show)\b/i.test(text)
}

const isBrandingUrlReference = (reference: ChatReferenceItem): boolean =>
    /\b(?:asset|badge|brand\s*kit|brand-kit|icon|logo|watermark)\b/i.test(
        [
            reference.title,
            reference.subtitle,
            reference.body,
            reference.relationship?.source,
            reference.relationship?.target,
            ...(reference.terms || []),
        ].join(' '),
    )

const isLinkedEvidenceReferenceForQuery = (
    reference: ChatReferenceItem,
    query?: string,
): boolean =>
    linkedEvidenceQuery(query) &&
    referenceExternalUrlValues(reference).length > 0 &&
    !isBrandingUrlReference(reference)

const linkedEvidenceSpecificityScore = (reference: ChatReferenceItem): number => {
    const relationship = reference.relationship
    const sourceIsUrl = isExternalUrlLabel(relationship?.source)
    const targetIsUrl = isExternalUrlLabel(relationship?.target)
    const hasExternalUrl = referenceExternalUrlValues(reference).length > 0
    if (!hasExternalUrl) return 0

    return (
        (targetIsUrl && !sourceIsUrl ? 36 : 0) +
        (relationship?.source && !sourceIsUrl ? 10 : 0) +
        (relationship?.relation ? 4 : 0) +
        (reference.body ? 4 : 0) +
        ((reference.sourceLocations || []).length ? 4 : 0) +
        ((reference.evidenceFamilies || []).length ? 2 : 0)
    )
}

const shouldSuppressExternalUrlReference = (
    reference: ChatReferenceItem,
    query?: string,
): boolean =>
    (isExternalUrlRelationshipReference(reference) || referenceExternalUrlValues(reference).length > 0) &&
    !isLinkedEvidenceReferenceForQuery(reference, query)

const isRepresentativeDocumentChunkReference = (reference: ChatReferenceItem): boolean => {
    const title = normalizeFacetText(reference.title || '')
    const body = normalizeFacetText(reference.body || '')
    return (
        title === 'document' &&
        /\brepresentative document chunk\b/.test(body)
    )
}

const SUMMARY_REFERENCE_STRUCTURAL_NODE_PATTERN =
    /\b(?:artifact|body|chunk|document|file|layout|location|page\s*\d*|paragraph|section|slide\s*\d*|source|table|text|title)\b|(?:^|[\s/\\])[\w.-]+\.(?:csv|docx?|html?|md|pdf|pptx?|txt|xlsx?)(?:$|[\s?#])/i

const SUMMARY_REFERENCE_META_NODE_PATTERN =
    /\b(?:best path|document evidence|extracted evidence|graph paths?|key evidence|knowledge graph paths?|retrieved context|semantic relationships?|source proof|top relationships?)\b/i

const SUMMARY_REFERENCE_LOW_VALUE_RELATION_PATTERN =
    /^(?:appears?_with|causes?|co_occurs(?:_with)?|contains?|contains_.+|has|has_.+|in_.+|mentioned?_with|mentions?|related_to|uses?|works?_at)$/i

const isHighSignalSummaryRelationshipReference = (
    reference: ChatReferenceItem,
    query?: string,
): boolean => {
    if (reference.kind !== 'relationship' || !reference.relationship) return false
    if (isTechnicalReviewReference(reference)) return false
    if (shouldSuppressExternalUrlReference(reference, query)) return false

    const relationKey = normalizeFacetText(reference.relationship.relation)
        .replace(/[^a-z0-9]+/g, '_')
        .replace(/^_+|_+$/g, '')
    if (!relationKey || SUMMARY_REFERENCE_LOW_VALUE_RELATION_PATTERN.test(relationKey)) return false

    const source = String(reference.relationship.source || '').trim()
    const target = String(reference.relationship.target || '').trim()
    if (!source || !target) return false

    const sourceLabel = referenceBasename(source)
    const targetLabel = referenceBasename(target)
    return ![
        source,
        target,
        sourceLabel,
        targetLabel,
    ].some(value =>
        SUMMARY_REFERENCE_STRUCTURAL_NODE_PATTERN.test(value) ||
        SUMMARY_REFERENCE_META_NODE_PATTERN.test(value),
    )
}

export const isLowSignalGraphRAGSummaryReference = (
    reference: ChatReferenceItem,
    query?: string,
): boolean => {
    if (!graphRAGSummaryQuestionQuery(query)) return false
    if (isHighSignalSummaryRelationshipReference(reference, query)) return false
    if (reference.kind !== 'relationship') return false

    const relationKey = normalizeFacetText(reference.relationship?.relation)
        .replace(/[^a-z0-9]+/g, '_')
        .replace(/^_+|_+$/g, '')
    const values = [
        reference.title,
        reference.subtitle,
        reference.body,
        reference.relationship?.source,
        reference.relationship?.target,
    ].map(value => String(value || '').trim())
    const structuralNodeCount = [
        reference.relationship?.source,
        reference.relationship?.target,
    ].filter(value =>
        SUMMARY_REFERENCE_STRUCTURAL_NODE_PATTERN.test(String(value || '')) ||
        SUMMARY_REFERENCE_STRUCTURAL_NODE_PATTERN.test(referenceBasename(String(value || ''))),
    ).length

    return (
        values.some(value => SUMMARY_REFERENCE_META_NODE_PATTERN.test(value)) ||
        structuralNodeCount >= 2 ||
        (structuralNodeCount > 0 && SUMMARY_REFERENCE_LOW_VALUE_RELATION_PATTERN.test(relationKey))
    )
}

const orderedUniqueReferences = (
    references: Array<ChatReferenceItem | undefined | null>,
): ChatReferenceItem[] => {
    const seen = new Set<string>()
    const output: ChatReferenceItem[] = []
    references.forEach(reference => {
        if (!reference || seen.has(reference.id)) return
        seen.add(reference.id)
        output.push(reference)
    })
    return output
}

const referenceDisplayKeyText = (value?: string): string =>
    normalizeFacetText(referenceBasename(String(value || '')))

const referenceDocumentDisplayKey = (reference: ChatReferenceItem): string =>
    referenceDisplayKeyText(
        reference.filename ||
        reference.documentId ||
        reference.runId ||
        reference.endpoint ||
        '',
    )

export const referencePathQueryOrderScore = (reference: ChatReferenceItem, query?: string): number => {
    if (reference.kind !== 'path' || !query) return 0
    const queryText = normalizeFacetText(query)
    if (!queryText) return 0
    const nodes = displayPathNodes(reference.path?.nodes || [])
        .map(node => normalizeFacetText(referenceBasename(node)))
        .filter(node => node.length >= 3)
    if (nodes.length < 2) return 0

    const mentioned = nodes
        .map((node, index) => ({
            index,
            position: queryText.indexOf(node),
        }))
        .filter(item => item.position >= 0)
    if (mentioned.length < 2) return 0

    const orderedTransitions = mentioned.slice(1).reduce(
        (count, item, index) => count + (item.position > mentioned[index].position ? 1 : 0),
        0,
    )
    const mentionedEveryNode = mentioned.length === nodes.length
    const allMentionedNodesInPathOrder = mentionedEveryNode && orderedTransitions === nodes.length - 1
    const mentionedNodesInPathOrder = orderedTransitions === mentioned.length - 1

    return (
        mentioned.length * 3 +
        orderedTransitions * 8 +
        (allMentionedNodesInPathOrder ? 24 : mentionedNodesInPathOrder ? 8 : 0) +
        (mentionedEveryNode ? 4 : 0)
    )
}

export const referencePathQueryCoverageScore = (reference: ChatReferenceItem, query?: string): number => {
    if (reference.kind !== 'path' || !query) return 0
    const queryText = normalizeFacetText(query)
    if (!queryText) return 0
    const nodes = displayPathNodes(reference.path?.nodes || [])
        .map(node => normalizeFacetText(referenceBasename(node)))
        .filter(node => node.length >= 3)
    if (nodes.length < 2) return 0

    const mentionedCount = nodes.filter(node => queryText.includes(node)).length
    if (mentionedCount < 2) return 0
    return mentionedCount * 24
}

const STRONG_GRAPH_PATH_RELATIONS = new Set([
    'approves',
    'blocks',
    'calculates',
    'causes',
    'connects_to',
    'depends_on',
    'drives',
    'gates',
    'groups_visual',
    'influences',
    'links_to',
    'maps_to',
    'owns',
    'references',
    'requires',
    'supports',
    'unblocks',
    'uses',
])

const WEAK_GRAPH_PATH_RELATIONS = new Set([
    'appears_with',
    'co_occurs',
    'co_occurs_with',
    'cooccurs',
    'mentioned_with',
    'mentions',
    'related',
    'related_text',
    'related_to',
])

const normalizeGraphPathRelationKey = (value?: string): string =>
    normalizeFacetText(value)
        .replace(/[^a-z0-9]+/g, '_')
        .replace(/^_+|_+$/g, '')

const isStructuralGraphPathRelation = (relationKey: string): boolean =>
    Boolean(relationKey && (
        relationKey.startsWith('contains_') ||
        relationKey.startsWith('has_') ||
        relationKey.startsWith('in_') ||
        ['appears_in', 'belongs_to', 'located_in', 'part_of'].includes(relationKey)
    ))

const isStructuralGraphPathNode = (value?: string): boolean =>
    SUMMARY_REFERENCE_STRUCTURAL_NODE_PATTERN.test(String(value || '')) ||
    SUMMARY_REFERENCE_STRUCTURAL_NODE_PATTERN.test(referenceBasename(String(value || '')))

export const referencePathSemanticStrengthScore = (reference: ChatReferenceItem): number => {
    if (reference.kind !== 'path') return 0
    const relationships = displayPathRelationships(reference.path?.relationships)
    if (relationships.length === 0) return 0

    const score = relationships.reduce((total, relationship) => {
        const relationKey = normalizeGraphPathRelationKey(relationship.relation)
        const semanticEndpoints =
            !isStructuralGraphPathNode(relationship.source) &&
            !isStructuralGraphPathNode(relationship.target)

        if (!relationKey) return total
        if (WEAK_GRAPH_PATH_RELATIONS.has(relationKey)) {
            return total - (semanticEndpoints ? 30 : 38)
        }
        if (isStructuralGraphPathRelation(relationKey)) {
            return total - (semanticEndpoints ? 8 : 18)
        }
        if (STRONG_GRAPH_PATH_RELATIONS.has(relationKey)) {
            return total + (semanticEndpoints ? 34 : 24)
        }
        return total + (semanticEndpoints ? 18 : 8)
    }, 0)

    const multiHopSignal = relationships.length >= 2 ? 10 : 0
    const weakOnlyPenalty = score < 0 ? 14 : 0
    return score + Math.min(10, relationships.length * 2) + multiHopSignal - weakOnlyPenalty
}

const normalizedGraphPathEndpointLabel = (value?: string): string =>
    normalizeFacetText(referenceBasename(String(value || '')))

const referenceEndpointLabels = (reference: ChatReferenceItem): [string, string] | null => {
    if (reference.relationship?.source && reference.relationship?.target) {
        const source = normalizedGraphPathEndpointLabel(reference.relationship.source)
        const target = normalizedGraphPathEndpointLabel(reference.relationship.target)
        return source && target ? [source, target] : null
    }

    const nodes = displayPathNodes(reference.path?.nodes || [])
        .map(normalizedGraphPathEndpointLabel)
        .filter(Boolean)
    if (nodes.length >= 2) {
        return [nodes[0], nodes[nodes.length - 1]]
    }

    return null
}

const endpointPairsMatch = (
    left: [string, string] | null,
    right: [string, string] | null,
): boolean =>
    Boolean(
        left &&
        right &&
        (
            (left[0] === right[0] && left[1] === right[1]) ||
            (left[0] === right[1] && left[1] === right[0])
        ),
    )

const isWeakGraphPathShortcutReference = (reference: ChatReferenceItem): boolean => {
    if (reference.kind === 'relationship') {
        return WEAK_GRAPH_PATH_RELATIONS.has(normalizeGraphPathRelationKey(reference.relationship?.relation))
    }

    if (reference.kind !== 'path') return false
    const relationships = displayPathRelationships(reference.path?.relationships)
    return (
        relationships.length > 0 &&
        relationships.every(relationship =>
            WEAK_GRAPH_PATH_RELATIONS.has(normalizeGraphPathRelationKey(relationship.relation)),
        )
    )
}

const selectedSemanticPathCoversReference = (
    selectedReferences: ChatReferenceItem[],
    candidate: ChatReferenceItem,
): boolean => {
    if (!isWeakGraphPathShortcutReference(candidate)) return false
    const candidateEndpoints = referenceEndpointLabels(candidate)
    if (!candidateEndpoints) return false

    return selectedReferences.some(reference =>
        reference.kind === 'path' &&
        referencePathSemanticStrengthScore(reference) > 0 &&
        endpointPairsMatch(referenceEndpointLabels(reference), candidateEndpoints),
    )
}

const graphFocusLabel = (value?: string): string =>
    normalizedGraphPathEndpointLabel(value)

const referenceGraphFocusLabels = (reference?: ChatReferenceItem | null): string[] => {
    if (!reference) return []

    return uniqueStrings([
        reference.relationship?.source,
        reference.relationship?.target,
        ...(reference.path?.nodes || []),
        ...(reference.path?.relationships || []).flatMap(relationship => [
            relationship.source,
            relationship.target,
        ]),
    ].map(graphFocusLabel).filter(Boolean))
}

const graphRelationshipFocusKey = (
    source?: string,
    relation?: string,
    target?: string,
): string => {
    const sourceKey = graphFocusLabel(source)
    const relationKey = normalizeGraphPathRelationKey(relation)
    const targetKey = graphFocusLabel(target)
    return sourceKey && relationKey && targetKey
        ? `${sourceKey}|${relationKey}|${targetKey}`
        : ''
}

const referenceGraphRelationshipFocusKeys = (reference?: ChatReferenceItem | null): string[] => {
    if (!reference) return []
    const relationships = [
        ...(reference.relationship ? [reference.relationship] : []),
        ...displayPathRelationships(reference.path?.relationships),
    ]

    return uniqueStrings(
        relationships
            .map(relationship => graphRelationshipFocusKey(
                relationship.source,
                relationship.relation,
                relationship.target,
            ))
            .filter(Boolean),
    )
}

export const selectFocusedReferenceReviewReferences = (
    selected?: ChatReferenceItem | null,
    references: ChatReferenceItem[] = [],
    answerEvidenceReferences: ChatReferenceItem[] = [],
): ChatReferenceItem[] => {
    if (!selected) return []

    const selectedDisplayKey = chatReferenceDisplayDedupeKey(selected)
    const selectedRelationshipKeys = new Set(referenceGraphRelationshipFocusKeys(selected))
    const selectedIsGraphReference =
        selected.kind === 'path' ||
        selected.kind === 'relationship' ||
        selectedRelationshipKeys.size > 0 ||
        referenceGraphFocusLabels(selected).length > 1

    return orderedUniqueReferences([
        selected,
        ...answerEvidenceReferences,
        ...references,
    ]).filter(candidate => {
        if (chatReferenceDisplayDedupeKey(candidate) === selectedDisplayKey) return true
        if (!selectedIsGraphReference) return false

        const candidateRelationshipKeys = referenceGraphRelationshipFocusKeys(candidate)
        return (
            candidateRelationshipKeys.length > 0 &&
            candidateRelationshipKeys.some(key => selectedRelationshipKeys.has(key))
        )
    }).slice(0, 8)
}

export const chatReferenceDisplayDedupeKey = (reference: ChatReferenceItem): string => {
    const fileKey = referenceDocumentDisplayKey(reference)
    const pageKey = pageNumberFromReference(reference)
        ? `page:${pageNumberFromReference(reference)}`
        : ''

    if (reference.relationship?.source || reference.relationship?.target) {
        return [
            'relationship',
            referenceDisplayKeyText(reference.relationship.source),
            referenceDisplayKeyText(reference.relationship.relation),
            referenceDisplayKeyText(reference.relationship.target),
            fileKey,
            pageKey,
        ].filter(Boolean).join('|') || reference.id
    }

    if (reference.path?.nodes?.length) {
        const nodeKey = reference.path.nodes.map(referenceDisplayKeyText).filter(Boolean).join('>')
        const relationshipKey = (reference.path.relationships || [])
            .map(relationship => [
                referenceDisplayKeyText(relationship.source),
                referenceDisplayKeyText(relationship.relation),
                referenceDisplayKeyText(relationship.target),
            ].filter(Boolean).join('~'))
            .filter(Boolean)
            .join('>')
        return [
            'path',
            nodeKey,
            relationshipKey,
            fileKey,
            pageKey,
        ].filter(Boolean).join('|') || reference.id
    }

    if (reference.kind === 'passage') {
        const chunkKey = reference.chunkIndex !== undefined ? `chunk:${reference.chunkIndex}` : ''
        return [
            'passage',
            fileKey,
            pageKey,
            chunkKey,
            !chunkKey ? referenceDisplayKeyText(reference.title || reference.body?.slice(0, 120)) : '',
        ].filter(Boolean).join('|') || reference.id
    }

    return [
        reference.kind,
        fileKey,
        pageKey,
        referenceDisplayKeyText(reference.title),
    ].filter(Boolean).join('|') || reference.id
}

export const dedupeChatReferencesForDisplay = (
    references: Array<ChatReferenceItem | undefined | null>,
): ChatReferenceItem[] => {
    const seen = new Set<string>()
    const output: ChatReferenceItem[] = []

    references.forEach(reference => {
        if (!reference) return
        const key = chatReferenceDisplayDedupeKey(reference)
        if (seen.has(key)) return
        seen.add(key)
        output.push(reference)
    })

    return output
}

const sortReferenceOpeningCandidates = (
    references: ChatReferenceItem[],
    query?: string,
): ChatReferenceItem[] =>
    [...references].sort((left, right) =>
        compareRequestedEvidenceFamilyOrder(left, right, query) ||
        referencePathQueryCoverageScore(right, query) - referencePathQueryCoverageScore(left, query) ||
        referencePathSemanticStrengthScore(right) - referencePathSemanticStrengthScore(left) ||
        referencePathQueryOrderScore(right, query) - referencePathQueryOrderScore(left, query) ||
        referenceAnswerFitScore(right, query) - referenceAnswerFitScore(left, query) ||
        referenceQueryRelevanceScore(right, query) - referenceQueryRelevanceScore(left, query) ||
        referenceReviewScore(right) - referenceReviewScore(left) ||
        (right.score || 0) - (left.score || 0),
    )

export const sortReferenceExplorerDrawerReferences = (
    references: ChatReferenceItem[],
    query?: string,
): ChatReferenceItem[] => {
    const summaryQuestion = graphRAGSummaryQuestionQuery(query)
    const pathQuestion = graphPathQuestionQuery(query)
    const relationshipQuestion = graphRelationshipQuestionQuery(query)
    const requestedEvidenceQuestion = requestedEvidenceFamilyQuestionQuery(query)
    const requestedEvidenceInspection = requestedEvidenceFamilyInspectionQuery(query)
    const priorityFor = (reference: ChatReferenceItem): number => {
        if (requestedEvidenceQuestion && (!pathQuestion || requestedEvidenceInspection)) {
            if (hasRequestedEvidenceFamilyMarker(reference)) return 0
            if (reference.kind === 'passage' && reference.sourceRole === 'answer_evidence') return 1
            if (reference.kind === 'document') return 2
            if (reference.kind === 'relationship') return 3
            if (reference.kind === 'path') return 4
            return 5
        }

        if (summaryQuestion) {
            if (isHighSignalSummaryRelationshipReference(reference, query)) return 0
            if (reference.kind === 'passage' && reference.sourceRole === 'answer_evidence') return 1
            if (isLowSignalGraphRAGSummaryReference(reference, query)) return 6
            if (reference.kind === 'relationship') return 2
            if (reference.kind === 'path') return 3
            if (reference.kind === 'passage') return 4
            if (reference.kind === 'document') return 5
            return 6
        }

        if (pathQuestion) {
            if (reference.kind === 'path') return 0
            if (reference.kind === 'relationship') return 1
            if (reference.kind === 'passage') return 2
            if (reference.kind === 'document') return 3
            return 4
        }

        if (relationshipQuestion) {
            if (reference.kind === 'relationship') return 0
            if (reference.kind === 'path') return 1
            if (reference.kind === 'passage') return 2
            if (reference.kind === 'document') return 3
            return 4
        }

        if (reference.kind === 'passage') return 0
        if (reference.kind === 'document') return 1
        if (reference.kind === 'relationship') return 2
        if (reference.kind === 'path') return 3
        return 4
    }

    return [...references].sort((left, right) =>
        priorityFor(left) - priorityFor(right) ||
        compareRequestedEvidenceFamilyOrder(left, right, query) ||
        referencePathQueryCoverageScore(right, query) - referencePathQueryCoverageScore(left, query) ||
        referencePathSemanticStrengthScore(right) - referencePathSemanticStrengthScore(left) ||
        referencePathQueryOrderScore(right, query) - referencePathQueryOrderScore(left, query) ||
        Number(left.sourceRole !== 'answer_evidence') - Number(right.sourceRole !== 'answer_evidence') ||
        referenceAnswerFitScore(right, query) - referenceAnswerFitScore(left, query) ||
        referenceQueryRelevanceScore(right, query) - referenceQueryRelevanceScore(left, query) ||
        referenceReviewScore(right) - referenceReviewScore(left) ||
        (right.score || 0) - (left.score || 0),
    )
}

const referenceDrawerDiversityBucket = (reference: ChatReferenceItem): string => {
    if (reference.kind === 'path') return 'path'
    if (isPdfReference(reference)) return 'pdf'
    if (hasRequestedEvidenceFamilyMarker(reference)) {
        return `requested:${referenceEvidenceBundleLabel(reference)}`
    }
    if (reference.kind === 'relationship') {
        return `relationship:${formatGraphReferenceRelation(reference.relationship?.relation || '') || 'edge'}`
    }
    if (reference.kind === 'passage') return `passage:${referenceEvidenceBundleLabel(reference)}`
    if (reference.kind === 'document' || reference.kind === 'artifact') return 'source'
    return reference.kind
}

export const diversifyReferenceExplorerDrawerReferences = (
    references: ChatReferenceItem[],
    query?: string,
    previewLimit = DEFAULT_REFERENCE_DRAWER_LIMIT,
): ChatReferenceItem[] => {
    const sorted = dedupeChatReferencesForDisplay(
        sortReferenceExplorerDrawerReferences(references, query),
    )
    if (sorted.length <= 1) return sorted

    const limit = Math.max(1, Math.min(previewLimit, sorted.length))
    const sourceKeys = uniqueStrings(sorted.map(referenceDocumentKey))
    if (sourceKeys.length <= 1) return sorted

    const output: ChatReferenceItem[] = []
    const seen = new Set<string>()
    const seenSourceBuckets = new Set<string>()
    const sourceCounts = new Map<string, number>()
    const maxRowsPerSource = Math.max(1, Math.ceil(limit / Math.min(sourceKeys.length, limit)))
    const add = (
        reference: ChatReferenceItem | undefined,
        options: { allowRepeatedBucket?: boolean; allowSourceOverflow?: boolean } = {},
    ): boolean => {
        if (!reference) return false
        const key = chatReferenceDisplayDedupeKey(reference)
        if (seen.has(key)) return false

        const sourceKey = referenceDocumentKey(reference)
        const bucket = `${sourceKey}:${referenceDrawerDiversityBucket(reference)}`
        if (
            !options.allowSourceOverflow &&
            output.length < limit &&
            (sourceCounts.get(sourceKey) || 0) >= maxRowsPerSource
        ) {
            return false
        }
        if (!options.allowRepeatedBucket && seenSourceBuckets.has(bucket)) return false

        seen.add(key)
        seenSourceBuckets.add(bucket)
        sourceCounts.set(sourceKey, (sourceCounts.get(sourceKey) || 0) + 1)
        output.push(reference)
        return true
    }

    sourceKeys.slice(0, limit).forEach(sourceKey => {
        add(sorted.find(reference => referenceDocumentKey(reference) === sourceKey), {
            allowRepeatedBucket: true,
            allowSourceOverflow: true,
        })
    })

    const passes: Array<(reference: ChatReferenceItem) => boolean> = [
        reference => reference.kind === 'path',
        reference => isRequestedEvidenceFamilyReference(reference),
        reference => isPdfReference(reference),
        reference => reference.kind === 'relationship',
        reference => reference.kind === 'passage',
        reference => reference.kind === 'document' || reference.kind === 'artifact',
    ]

    for (const pass of passes) {
        for (const reference of sorted) {
            if (output.length >= limit) break
            if (pass(reference)) add(reference)
        }
        if (output.length >= limit) break
    }

    for (const reference of sorted) {
        if (output.length >= limit) break
        add(reference, { allowRepeatedBucket: true, allowSourceOverflow: true })
    }

    return [
        ...output,
        ...sorted.filter(reference => !seen.has(chatReferenceDisplayDedupeKey(reference))),
    ]
}

export const selectReferenceExplorerOpeningReference = (
    orderedReferences: ChatReferenceItem[],
    query?: string,
    options: {
        trayReferences?: ChatReferenceItem[]
        answerEvidenceReferences?: ChatReferenceItem[]
        fallback?: ChatReferenceItem | null
    } = {},
): ChatReferenceItem | null => {
    const rawCandidates = dedupeChatReferencesForDisplay(orderedUniqueReferences([
        ...(options.trayReferences || []),
        ...(options.answerEvidenceReferences || []),
        ...orderedReferences,
    ]))
    const curatedCandidates = rawCandidates.filter(reference => !isSyntheticFilenameRelationshipReference(reference))
    const candidates = curatedCandidates.length > 0 ? curatedCandidates : rawCandidates
    if (!candidates.length) return options.fallback || null

    const requestedEvidenceCandidates = candidates.filter(hasRequestedEvidenceFamilyMarker)
    const requestedFilenames = queryFilenames(query)

    if (requestedEvidenceCandidates.length > 0 && requestedEvidenceFamilyInspectionQuery(query)) {
        const trayRequestedEvidence = dedupeChatReferencesForDisplay(options.trayReferences || [])
            .find(hasRequestedEvidenceFamilyMarker)
        if (trayRequestedEvidence) return trayRequestedEvidence

        return sortReferenceOpeningCandidates(requestedEvidenceCandidates, query)[0]
    }

    if (requestedFilenames.length > 0 && !graphPathQuestionQuery(query)) {
        const namedFileCandidates = candidates
            .filter(reference =>
                matchingQueryFilename(reference, requestedFilenames) &&
                !shouldSuppressExternalUrlReference(reference, query) &&
                !isRepresentativeDocumentChunkReference(reference) &&
                !isRequestedEvidenceFamilyReference(reference),
            )
        if (namedFileCandidates.length > 0) {
            const configTokenCandidates = namedFileCandidates
                .filter(reference => namedFileConfigTokenScore(reference, query) > 0)
                .sort((left, right) =>
                    namedFileConfigTokenScore(right, query) - namedFileConfigTokenScore(left, query) ||
                    namedFileReferenceScore(right, query) - namedFileReferenceScore(left, query),
                )
            if (configTokenCandidates.length > 0) return configTokenCandidates[0]

            return [...namedFileCandidates].sort((left, right) =>
                namedFileReferenceKindPriority(left) - namedFileReferenceKindPriority(right) ||
                namedFileReferenceScore(right, query) - namedFileReferenceScore(left, query),
            )[0]
        }
    }

    if (graphPathQuestionQuery(query)) {
        const trayPathCandidates = dedupeChatReferencesForDisplay(options.trayReferences || [])
            .filter(reference => reference.kind === 'path')
        if (trayPathCandidates.length > 0) {
            return trayPathCandidates[0]
        }

        const pathCandidates = candidates.filter(reference => reference.kind === 'path')
        if (pathCandidates.length > 0) {
            return sortReferenceOpeningCandidates(pathCandidates, query)[0]
        }
    }

    if (requestedEvidenceCandidates.length > 0 && requestedEvidenceFamilyQuestionQuery(query)) {
        const trayRequestedEvidence = dedupeChatReferencesForDisplay(options.trayReferences || [])
            .find(hasRequestedEvidenceFamilyMarker)
        if (trayRequestedEvidence) return trayRequestedEvidence

        return sortReferenceOpeningCandidates(requestedEvidenceCandidates, query)[0]
    }

    if (graphRAGSummaryQuestionQuery(query)) {
        const summaryRelationshipCandidates = candidates.filter(reference =>
            reference.sourceRole === 'answer_evidence' &&
            isHighSignalSummaryRelationshipReference(reference, query),
        )
        if (summaryRelationshipCandidates.length > 0) {
            return sortReferenceOpeningCandidates(summaryRelationshipCandidates, query)[0]
        }
    }

    if (relationshipCitationQuery(query) && graphRelationshipQuestionQuery(query)) {
        const directRelationshipCandidates = candidates.filter(reference =>
            reference.kind === 'relationship' &&
            reference.sourceRole === 'answer_evidence' &&
            !shouldSuppressExternalUrlReference(reference, query),
        )
        if (directRelationshipCandidates.length > 0) {
            return sortReferenceOpeningCandidates(directRelationshipCandidates, query)[0]
        }
    }

    if (!graphRelationshipQuestionQuery(query) && !graphPathQuestionQuery(query)) {
        const sourceEvidenceCandidates = candidates.filter(reference =>
            reference.kind === 'passage' &&
            reference.sourceRole === 'answer_evidence' &&
            !shouldSuppressExternalUrlReference(reference, query) &&
            !isRequestedEvidenceFamilyReference(reference),
        )
        if (sourceEvidenceCandidates.length > 0) {
            return sortReferenceOpeningCandidates(sourceEvidenceCandidates, query)[0]
        }

        const passageCandidates = candidates.filter(reference =>
            reference.kind === 'passage' &&
            !shouldSuppressExternalUrlReference(reference, query) &&
            !isRepresentativeDocumentChunkReference(reference) &&
            !isRequestedEvidenceFamilyReference(reference),
        )
        if (passageCandidates.length > 0) {
            return sortReferenceOpeningCandidates(passageCandidates, query)[0]
        }
    }

    return sortReferenceOpeningCandidates(candidates, query)[0] || options.fallback || null
}

const normalizedPathSelectionNodes = (reference: ChatReferenceItem): string[] =>
    uniqueStrings([
        ...displayPathNodes(reference.path?.nodes || []),
        ...String(reference.title || '')
            .split(/\s*(?:->|→)\s*/)
            .map(node => node.trim()),
    ])
        .map(node => normalizeFacetText(referenceBasename(node)))
        .filter(Boolean)

const referencePathSelectionScore = (
    candidate: ChatReferenceItem,
    target: ChatReferenceItem,
): number => {
    const candidateNodes = normalizedPathSelectionNodes(candidate)
    const targetNodes = normalizedPathSelectionNodes(target)
    if (candidateNodes.length === 0 || targetNodes.length === 0) return 0

    const targetSet = new Set(targetNodes)
    const overlap = candidateNodes.filter(node => targetSet.has(node)).length
    const sameStart = candidateNodes[0] === targetNodes[0] ? 1 : 0
    const sameEnd = candidateNodes[candidateNodes.length - 1] === targetNodes[targetNodes.length - 1] ? 1 : 0
    const middleOverlap = candidateNodes
        .slice(1, -1)
        .filter(node => targetSet.has(node)).length

    return overlap * 4 + sameStart * 3 + sameEnd * 3 + middleOverlap * 2
}

export const resolveReferenceExplorerSelectionReference = (
    orderedReferences: ChatReferenceItem[],
    reference: ChatReferenceItem,
    query?: string,
): ChatReferenceItem => {
    if (reference.kind === 'path') {
        const [bestPathMatch] = orderedReferences
            .filter(candidate => candidate.kind === 'path')
            .map(candidate => ({
                candidate,
                score: referencePathSelectionScore(candidate, reference),
            }))
            .filter(item => item.score > 0)
            .sort((left, right) =>
                right.score - left.score ||
                referencePathQueryCoverageScore(right.candidate, query) - referencePathQueryCoverageScore(left.candidate, query) ||
                referencePathSemanticStrengthScore(right.candidate) - referencePathSemanticStrengthScore(left.candidate) ||
                referencePathQueryOrderScore(right.candidate, query) - referencePathQueryOrderScore(left.candidate, query) ||
                referenceAnswerFitScore(right.candidate, query) - referenceAnswerFitScore(left.candidate, query) ||
                referenceReviewScore(right.candidate) - referenceReviewScore(left.candidate),
            )
        if (bestPathMatch) return bestPathMatch.candidate
    }

    const exact = orderedReferences.find(candidate => candidate.id === reference.id)
    if (exact) return exact

    const referenceKey = chatReferenceDisplayDedupeKey(reference)
    const deduped = orderedReferences.find(candidate => chatReferenceDisplayDedupeKey(candidate) === referenceKey)
    if (deduped) return deduped

    return reference
}

export const selectReferenceExplorerActiveReference = (
    orderedReferences: ChatReferenceItem[],
    selectedId: string,
    selectedOverrideReference?: ChatReferenceItem | null,
): ChatReferenceItem | null => {
    if (selectedOverrideReference && selectedOverrideReference.id === selectedId) {
        return selectedOverrideReference
    }
    return orderedReferences.find(reference => reference.id === selectedId) || orderedReferences[0] || null
}

const isReferenceTrayGraphContext = (reference: ChatReferenceItem): boolean => {
    if (reference.kind === 'path') return true
    if (reference.kind !== 'relationship') return false
    if (isTechnicalReviewReference(reference) || isBrandingUrlReference(reference)) return false
    return !isExternalUrlRelationshipReference(reference)
}

const referenceTrayGraphContextScore = (
    reference: ChatReferenceItem,
    query?: string,
): number => {
    const families = (reference.evidenceFamilies || []).join(' ')
    const visualEvidence =
        /\b(?:connector|group|layout|semantic|shape|shape_group|visual)\b/i.test(families) ||
        /\b(?:connector|group|layout|semantic|shape|visual)\b/i.test(referenceSearchText(reference))

    return (
        (reference.kind === 'path' ? 36 : 0) +
        (reference.sourceRole === 'reasoning_path' ? 10 : 0) +
        (reference.sourceRole === 'answer_evidence' ? 8 : 0) +
        (visualEvidence ? 8 : 0) +
        referencePathQueryCoverageScore(reference, query) +
        referencePathSemanticStrengthScore(reference) +
        referencePathQueryOrderScore(reference, query) +
        referenceAnswerFitScore(reference, query) +
        referenceQueryRelevanceScore(reference, query) +
        referenceReviewScore(reference) +
        (reference.score || 0)
    )
}

export const selectReferenceExplorerTrayReferences = ({
    orderedReferences,
    visibleReferences,
    answerEvidenceReferences,
    query,
}: {
    orderedReferences: ChatReferenceItem[]
    visibleReferences: ChatReferenceItem[]
    answerEvidenceReferences: ChatReferenceItem[]
    query?: string
}): ChatReferenceItem[] => {
    const output: ChatReferenceItem[] = []
    const selectedIds = new Set<string>()
    const selectedExternalUrls = new Set<string>()
    const addReference = (reference?: ChatReferenceItem) => {
        if (!reference) return
        if (isSyntheticFilenameRelationshipReference(reference)) return
        const key = chatReferenceDisplayDedupeKey(reference)
        if (selectedIds.has(key)) return
        if (selectedSemanticPathCoversReference(output, reference)) return

        const externalUrl = selectReferencePrimaryExternalUrl(reference).toLowerCase()
        if (
            externalUrl &&
            selectedExternalUrls.has(externalUrl) &&
            isLinkedEvidenceReferenceForQuery(reference, query)
        ) {
            return
        }

        output.push(reference)
        selectedIds.add(key)
        if (externalUrl) selectedExternalUrls.add(externalUrl)
    }

    const allReferenceCandidates = dedupeChatReferencesForDisplay(orderedUniqueReferences([
        ...answerEvidenceReferences,
        ...visibleReferences,
        ...orderedReferences,
    ]))
    const relationshipReferences = answerEvidenceReferences
        .filter(reference => reference.kind === 'relationship' || reference.kind === 'path')
    const graphCandidateReferences = allReferenceCandidates.filter(reference =>
        (reference.kind === 'relationship' || reference.kind === 'path') &&
        !shouldSuppressExternalUrlReference(reference, query),
    )
    const focusedRelationshipReferences = relationshipReferences.filter(reference => !shouldSuppressExternalUrlReference(reference, query))
    const primaryRelationshipReferences = focusedRelationshipReferences.length >= 2
        ? focusedRelationshipReferences
        : relationshipReferences
    const requestedEvidenceQuestion = requestedEvidenceFamilyQuestionQuery(query)
    const graphPathQuestion = graphPathQuestionQuery(query)
    const graphRelationshipQuestion = graphRelationshipQuestionQuery(query)
    const summaryQuestion = graphRAGSummaryQuestionQuery(query)
    const requestedFilenames = queryFilenames(query)
    const requestedEvidenceInspection = requestedEvidenceFamilyInspectionQuery(query)
    const requestedEvidenceReferences = allReferenceCandidates
        .filter(reference =>
            isRequestedEvidenceFamilyReference(reference) ||
            (requestedEvidenceInspection && hasRequestedEvidenceFamilyMarker(reference)),
        )
        .sort((left, right) =>
            compareRequestedEvidenceFamilyOrder(left, right, query) ||
            referenceAnswerFitScore(right, query) - referenceAnswerFitScore(left, query) ||
            referenceReviewScore(right) - referenceReviewScore(left),
        )
    const richEvidenceFamilyReferences = allReferenceCandidates
        .filter(reference => isRichEvidenceFamilyReference(reference, query))
        .sort((left, right) =>
        compareRequestedEvidenceFamilyOrder(left, right, query) ||
        referenceAnswerFitScore(right, query) - referenceAnswerFitScore(left, query) ||
        referenceReviewScore(right) - referenceReviewScore(left),
    )
    const requestedEvidenceTrayLimit = requestedEvidenceInspection
        ? Math.min(9, Math.max(6, requestedEvidenceFamilyOrder(query).size || 0))
        : 6
    const maxTrayReferences =
        requestedEvidenceQuestion &&
        (!graphPathQuestion || requestedEvidenceInspection) &&
        richEvidenceFamilyReferences.length >= 3
            ? requestedEvidenceTrayLimit
            : 2
    const preferRequestedEvidenceTray =
        requestedEvidenceQuestion &&
        (!graphPathQuestion || requestedEvidenceInspection) &&
        richEvidenceFamilyReferences.length > 0

    if (requestedFilenames.length > 0 && !graphPathQuestion && !preferRequestedEvidenceTray) {
        const namedFileCandidates = dedupeChatReferencesForDisplay(orderedUniqueReferences([
            ...answerEvidenceReferences,
            ...visibleReferences,
            ...orderedReferences,
        ]))
            .filter(reference =>
                !shouldSuppressExternalUrlReference(reference, query) &&
                !isSyntheticFilenameRelationshipReference(reference) &&
                !isRequestedEvidenceFamilyReference(reference) &&
                Boolean(matchingQueryFilename(reference, requestedFilenames)),
            )
            .sort((left, right) =>
                namedFileReferenceKindPriority(left) - namedFileReferenceKindPriority(right) ||
                namedFileReferenceScore(right, query) - namedFileReferenceScore(left, query) ||
                referenceDocumentTitle(left).localeCompare(referenceDocumentTitle(right)),
            )

        namedFileCandidates
            .filter(reference => namedFileConfigTokenScore(reference, query) > 0)
            .sort((left, right) =>
                namedFileConfigTokenScore(right, query) - namedFileConfigTokenScore(left, query) ||
                namedFileReferenceScore(right, query) - namedFileReferenceScore(left, query),
            )
            .slice(0, 1)
            .forEach(addReference)

        requestedFilenames
            .map(filename => namedFileCandidates.find(reference =>
                matchingQueryFilename(reference, [filename]) === filename,
            ))
            .filter((reference): reference is ChatReferenceItem => Boolean(reference))
            .sort((left, right) =>
                namedFileReferenceKindPriority(left) - namedFileReferenceKindPriority(right) ||
                namedFileReferenceScore(right, query) - namedFileReferenceScore(left, query) ||
                requestedFilenames.indexOf(matchingQueryFilename(left, requestedFilenames)) -
                    requestedFilenames.indexOf(matchingQueryFilename(right, requestedFilenames)),
            )
            .forEach(reference => {
            if (output.length >= 2) return
            addReference(reference)
            })

        if (output.length < Math.min(2, requestedFilenames.length)) {
            const representedFilenames = new Set(
                output
                    .map(reference => matchingQueryFilename(reference, requestedFilenames))
                    .filter(Boolean),
            )
            namedFileCandidates.forEach(reference => {
                if (output.length >= Math.min(2, requestedFilenames.length)) return
                const filenameKey = matchingQueryFilename(reference, requestedFilenames)
                if (
                    filenameKey &&
                    representedFilenames.has(filenameKey) &&
                    representedFilenames.size < requestedFilenames.length
                ) {
                    return
                }
                const previousLength = output.length
                addReference(reference)
                if (filenameKey && output.length > previousLength) {
                    representedFilenames.add(filenameKey)
                }
            })
        }

        if (output.length >= Math.min(2, requestedFilenames.length)) {
            return output.slice(0, 2)
        }
    }

    if (requestedEvidenceQuestion && (!graphPathQuestion || requestedEvidenceInspection)) {
        dedupeChatReferencesForDisplay(orderedUniqueReferences([
            ...requestedEvidenceReferences,
            ...richEvidenceFamilyReferences,
        ]))
            .sort((left, right) =>
                compareRequestedEvidenceFamilyOrder(left, right, query) ||
                referenceAnswerFitScore(right, query) - referenceAnswerFitScore(left, query) ||
                referenceReviewScore(right) - referenceReviewScore(left),
            )
            .slice(0, maxTrayReferences)
            .forEach(addReference)
    }

    if (linkedEvidenceQuery(query)) {
        const [bestLinkedEvidence] = relationshipReferences
            .filter(reference => isLinkedEvidenceReferenceForQuery(reference, query))
            .sort((left, right) =>
                linkedEvidenceSpecificityScore(right) - linkedEvidenceSpecificityScore(left) ||
                referenceAnswerFitScore(right, query) - referenceAnswerFitScore(left, query) ||
                referenceQueryRelevanceScore(right, query) - referenceQueryRelevanceScore(left, query) ||
                referenceReviewScore(right) - referenceReviewScore(left),
            )
        addReference(bestLinkedEvidence)
    }

    if (graphPathQuestion) {
        const [bestPath] = graphCandidateReferences
            .filter(reference => reference.kind === 'path')
            .sort((left, right) =>
                referencePathQueryCoverageScore(right, query) - referencePathQueryCoverageScore(left, query) ||
                referencePathSemanticStrengthScore(right) - referencePathSemanticStrengthScore(left) ||
                referencePathQueryOrderScore(right, query) - referencePathQueryOrderScore(left, query) ||
                referenceAnswerFitScore(right, query) - referenceAnswerFitScore(left, query) ||
                referenceQueryRelevanceScore(right, query) - referenceQueryRelevanceScore(left, query) ||
                referenceReviewScore(right) - referenceReviewScore(left),
            )
        addReference(bestPath)
    }

    if (requestedEvidenceQuestion && graphPathQuestion) {
        requestedEvidenceReferences
            .slice(0, 1)
            .forEach(addReference)
    }

    if (summaryQuestion) {
        graphCandidateReferences
            .filter(reference => isHighSignalSummaryRelationshipReference(reference, query))
            .sort((left, right) =>
                referenceAnswerFitScore(right, query) - referenceAnswerFitScore(left, query) ||
                referenceQueryRelevanceScore(right, query) - referenceQueryRelevanceScore(left, query) ||
                referenceReviewScore(right) - referenceReviewScore(left),
            )
            .slice(0, 2)
            .forEach(addReference)
    }

    if (relationshipCitationQuery(query) && graphRelationshipQuestion && !graphPathQuestion) {
        const directRelationshipSource =
            graphCandidateReferences.filter(reference => reference.kind === 'relationship').length > 0
                ? graphCandidateReferences
                : primaryRelationshipReferences
        const [bestDirectRelationship] = directRelationshipSource
            .filter(reference => reference.kind === 'relationship')
            .sort((left, right) =>
                referencePathQueryCoverageScore(right, query) - referencePathQueryCoverageScore(left, query) ||
                referencePathSemanticStrengthScore(right) - referencePathSemanticStrengthScore(left) ||
                referencePathQueryOrderScore(right, query) - referencePathQueryOrderScore(left, query) ||
                referenceAnswerFitScore(right, query) - referenceAnswerFitScore(left, query) ||
                referenceQueryRelevanceScore(right, query) - referenceQueryRelevanceScore(left, query) ||
                referenceReviewScore(right) - referenceReviewScore(left),
            )
        addReference(bestDirectRelationship)
    }

    const connectionPhrases = queryConnectionPhrases(query)
    if (connectionPhrases.length > 0) {
        const coverageCandidates = uniqueStrings([
            ...answerEvidenceReferences.map(reference => reference.id),
            ...visibleReferences.map(reference => reference.id),
            ...orderedReferences.map(reference => reference.id),
        ])
            .map(id => orderedReferences.find(reference => reference.id === id))
            .filter((reference): reference is ChatReferenceItem => {
                if (!reference) return false
                return !shouldSuppressExternalUrlReference(reference, query)
            })

        connectionPhrases.forEach(phrase => {
            const [best] = coverageCandidates
                .filter(reference => referenceMatchesPhrase(reference, phrase))
                .sort((left, right) =>
                    referencePathQueryCoverageScore(right, query) - referencePathQueryCoverageScore(left, query) ||
                    referencePathSemanticStrengthScore(right) - referencePathSemanticStrengthScore(left) ||
                    referencePathQueryOrderScore(right, query) - referencePathQueryOrderScore(left, query) ||
                    referenceAnswerFitScore(right, query) - referenceAnswerFitScore(left, query) ||
                    referenceQueryRelevanceScore(right, query) - referenceQueryRelevanceScore(left, query) ||
                    referenceReviewScore(right) - referenceReviewScore(left),
                )
            if (best) addReference(best)
        })
    }

    if (graphRelationshipQuestion || graphPathQuestion) {
        const relationshipSource =
            graphCandidateReferences.length > 0
                ? graphCandidateReferences
                : primaryRelationshipReferences
        relationshipSource
            .slice()
            .sort((left, right) =>
                referencePathQueryCoverageScore(right, query) - referencePathQueryCoverageScore(left, query) ||
                referencePathSemanticStrengthScore(right) - referencePathSemanticStrengthScore(left) ||
                referencePathQueryOrderScore(right, query) - referencePathQueryOrderScore(left, query) ||
                referenceAnswerFitScore(right, query) - referenceAnswerFitScore(left, query) ||
                referenceQueryRelevanceScore(right, query) - referenceQueryRelevanceScore(left, query) ||
                referenceReviewScore(right) - referenceReviewScore(left),
            )
            .filter(reference => !selectedSemanticPathCoversReference(output, reference))
            .slice(0, focusedRelationshipReferences.length >= 2 ? 2 : 3)
            .forEach(addReference)
    } else {
        dedupeChatReferencesForDisplay(orderedUniqueReferences([
            ...answerEvidenceReferences,
            ...visibleReferences,
            ...orderedReferences,
        ]))
            .filter(reference =>
                reference.kind === 'passage' &&
                !shouldSuppressExternalUrlReference(reference, query) &&
                !isRepresentativeDocumentChunkReference(reference) &&
                !isRequestedEvidenceFamilyReference(reference),
            )
            .slice(0, 2)
            .forEach(addReference)
        if (output.length < 2) {
            primaryRelationshipReferences.slice(0, 1).forEach(addReference)
        }
        dedupeChatReferencesForDisplay(orderedUniqueReferences([
            ...visibleReferences,
            ...answerEvidenceReferences,
            ...orderedReferences,
        ]))
            .filter(reference =>
                !shouldSuppressExternalUrlReference(reference, query) &&
                reference.kind !== 'passage' &&
                reference.kind !== 'document' &&
                !isRepresentativeDocumentChunkReference(reference) &&
                !isRequestedEvidenceFamilyReference(reference),
            )
            .slice(0, 2)
            .forEach(addReference)
    }

    if (focusedRelationshipReferences.length < 2 && (graphRelationshipQuestion || graphPathQuestion)) {
        answerEvidenceReferences
            .filter(reference =>
                reference.kind === 'passage' &&
                !shouldSuppressExternalUrlReference(reference, query) &&
                (requestedEvidenceQuestion || !isRequestedEvidenceFamilyReference(reference)),
            )
            .slice(0, 1)
            .forEach(addReference)
        visibleReferences
            .filter(reference =>
                !shouldSuppressExternalUrlReference(reference, query) &&
                (requestedEvidenceQuestion || !isRequestedEvidenceFamilyReference(reference)),
            )
            .slice(0, 3)
            .forEach(addReference)
    }

    if ((relationshipCitationQuery(query) && graphRelationshipQuestion) || graphPathQuestion) {
        const hasContextualGraphReference = output.some(isReferenceTrayGraphContext)
        if (!hasContextualGraphReference) {
            const [bestGraphContext] = dedupeChatReferencesForDisplay(orderedUniqueReferences([
                ...answerEvidenceReferences,
                ...visibleReferences,
                ...orderedReferences,
            ]))
                .filter(reference => isReferenceTrayGraphContext(reference))
                .filter(reference => !selectedIds.has(chatReferenceDisplayDedupeKey(reference)))
                .sort((left, right) =>
                    referenceTrayGraphContextScore(right, query) - referenceTrayGraphContextScore(left, query),
                )

            if (bestGraphContext) {
                const lowValueIndex = output.findIndex(reference =>
                    reference.kind === 'passage' ||
                    reference.kind === 'document' ||
                    reference.kind === 'artifact',
                )
                const replaceableIndex = lowValueIndex >= 0
                    ? lowValueIndex
                    : output.findIndex((reference, index) => index > 0 && !isReferenceTrayGraphContext(reference))

                if (replaceableIndex >= 0) {
                    output[replaceableIndex] = bestGraphContext
                } else if (output.length < 2) {
                    output.push(bestGraphContext)
                }
            }
        }
    }

    return output.slice(0, maxTrayReferences)
}

export const selectInlineReferenceExplorerTrayReferences = (
    trayReferences: ChatReferenceItem[],
    query?: string,
    limit = INLINE_TRAY_REFERENCE_LIMIT,
): ChatReferenceItem[] => {
    const normalizedLimit = Math.max(0, Math.floor(limit))
    if (normalizedLimit <= 0) return []
    if (trayReferences.length <= normalizedLimit) return trayReferences.slice(0, normalizedLimit)
    if (!requestedEvidenceFamilyInspectionQuery(query)) return trayReferences.slice(0, normalizedLimit)

    const familyLeaders: Array<{ family: string; reference: ChatReferenceItem; index: number }> = []
    const seenFamilies = new Set<string>()

    trayReferences.forEach((reference, index) => {
        const family = primaryRequestedEvidenceFamilyKey(reference, query)
        const key = family || chatReferenceDisplayDedupeKey(reference)
        if (seenFamilies.has(key)) return
        seenFamilies.add(key)
        familyLeaders.push({ family: key, reference, index })
    })

    if (familyLeaders.length <= normalizedLimit) return trayReferences.slice(0, normalizedLimit)

    const selected = new Set<string>()
    const selectedLeaders: Array<{ family: string; reference: ChatReferenceItem; index: number }> = []
    const addLeader = (leader?: { family: string; reference: ChatReferenceItem; index: number }) => {
        if (!leader || selectedLeaders.length >= normalizedLimit) return
        const key = chatReferenceDisplayDedupeKey(leader.reference)
        if (selected.has(key)) return
        selected.add(key)
        selectedLeaders.push(leader)
    }

    addLeader(familyLeaders[0])
    addLeader(familyLeaders[1])

    INLINE_TRAY_BREADTH_FAMILY_PRIORITY
        .map(family => familyLeaders.find(leader => leader.family === family && leader.index >= normalizedLimit))
        .forEach(addLeader)

    if (selectedLeaders.length < normalizedLimit) {
        [...familyLeaders]
            .reverse()
            .filter(leader => leader.index >= normalizedLimit)
            .forEach(addLeader)
    }

    if (selectedLeaders.length < normalizedLimit) {
        familyLeaders.forEach(addLeader)
    }

    return selectedLeaders
        .slice(0, normalizedLimit)
        .sort((left, right) => left.index - right.index)
        .map(leader => leader.reference)
}

export const shouldShowReferenceExplorerDrawerSearch = ({
    referenceCount,
    sourceGroupCount,
    drawerSearch,
    kindFilter,
    compact = false,
}: {
    referenceCount: number
    sourceGroupCount: number
    drawerSearch?: string
    kindFilter?: ReferenceFilterValue
    compact?: boolean
}): boolean => {
    if (String(drawerSearch || '').trim()) return true
    if (kindFilter && kindFilter !== 'all') return true
    if (compact) return false
    return referenceCount > DRAWER_SEARCH_REFERENCE_THRESHOLD || sourceGroupCount > DRAWER_SEARCH_SOURCE_THRESHOLD
}

const referenceMatchesPhrase = (reference: ChatReferenceItem, phrase: string): boolean => {
    const normalizedPhrase = phrase.toLowerCase()
    if (!normalizedPhrase) return false
    const haystack = referenceSearchText(reference)
    if (haystack.includes(normalizedPhrase)) return true
    const phraseTerms = normalizedPhrase
        .split(/[^a-z0-9_-]+/i)
        .map(term => term.trim())
        .filter(term => term.length >= 3 && !LOW_VALUE_SOURCE_QUERY_TERMS.has(term))
    return phraseTerms.length > 0 && phraseTerms.every(term => haystack.includes(term))
}

const referenceQueryRelevanceScore = (reference: ChatReferenceItem, query?: string): number => {
    const terms = queryTerms(query).filter(term => !LOW_VALUE_SOURCE_QUERY_TERMS.has(term.toLowerCase()))
    if (!terms.length) return 0

    const haystack = referenceSearchText(reference)
    const connectionPhraseScore = queryConnectionPhrases(query).reduce((score, phrase) => (
        referenceMatchesPhrase(reference, phrase) ? score + 3 : score
    ), 0)
    const baseScore = terms.reduce((score, term) => (
        haystack.includes(term.toLowerCase()) ? score + 1 : score
    ), 0)
    const structuralEvidenceQuery = terms.some(term => /^(?:chart|charts|table|tables|form|forms|diagram|diagrams|connector|connectors)$/.test(term.toLowerCase()))
    const relationshipText = [
        reference.relationship?.source,
        reference.relationship?.relation,
        reference.relationship?.target,
    ].join(' ').toLowerCase()
    const relationshipMatchesQuery = terms.some(term => relationshipText.includes(term.toLowerCase()))
    const supportingArtifactFact =
        structuralEvidenceQuery &&
        reference.kind === 'relationship' &&
        !relationshipMatchesQuery &&
        /\b(?:artifact|attachment|embedded|owner|ready|status|workbook)\b/.test(relationshipText)

    return baseScore + connectionPhraseScore - (supportingArtifactFact ? 3 : 0)
}

const cleanDocumentOutlinePreview = (value: string): string =>
    value
        .replace(/\s*;\s*(?:visibility|parse status|parse_status)\s*[:=]\s*[^;)]+/gi, '')
        .replace(/\s*;\s*(?:table|sheet|slide|chart|document|media)?\s*source\s+semantic_[a-z_]+\b[^;)]+/gi, '')
        .replace(/\s*;\s*(?:part|package path|relationship|source|file)\s*[:=]\s*[^;)]+/gi, '')
        .replace(/\s*;\s*$/g, '')
        .replace(/\s+/g, ' ')
        .trim()

const isLowSignalDocumentOutlineItem = (value: string): boolean => {
    const normalized = normalizeFacetText(value)
    return (
        !normalized ||
        ((/^source\s*(?:\||:)/i.test(value) || /^(?:source|relationship|target|evidence)\s*\|/i.test(value)) &&
            /\b(?:relationship|target|evidence)\b/i.test(value)) ||
        /\b(?:semantic_[a-z_]+|visibility\s+(?:visible|hidden)|parse_status|relationship_id|image_relationship_id)\b/i.test(value)
    )
}

const collectTerms = (reference: ChatReferenceItem | null, query?: string): string[] =>
    uniqueStrings([
        ...(reference?.terms || []),
        ...(reference?.matchedTerms || []),
        ...(reference?.evidenceFamilies || []),
        ...(reference?.locationHints || []),
        ...(reference?.sourceLocations || []).flatMap(location => [
            location.label,
            location.value,
            location.category,
        ]),
        reference?.relationship?.source,
        reference?.relationship?.target,
        ...queryTerms(query),
    ])
        .filter(term => {
            const normalized = term.toLowerCase().replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, '')
            if (normalized.length < 3) return false
            if (/^\d+$/.test(normalized)) return false
            if (/^(?:line|paragraph|heading|section|chunk)[-_ ]*\d*$/.test(normalized)) return false
            if (/\b(?:line|paragraph|heading|section|chunk)\b/.test(normalized)) return false
            return !HIGHLIGHT_STOP_TERMS.has(normalized)
        })
        .slice(0, 28)

const cleanPdfEvidenceHighlightPhrase = (value: string): string =>
    cleanReferenceEvidencePreview(value)
        .replace(/^\[[^\]]+\]\s*/i, '')
        .replace(
            /^(?:page|slide)\s+\d+(?:\s+(?:block|paragraph|line|connector|chart|table|row|shape|control)\s+\d+(?:\.\d+)?)?\s*:\s*/i,
            '',
        )
        .replace(/\b(?:Source|Citation)\s*:\s*[^.]+$/i, '')
        .replace(/\s+/g, ' ')
        .trim()

const pdfEvidenceHighlightFragments = (phrase: string): string[] => {
    const words = phrase
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, ' ')
        .split(/\s+/)
        .map(word => word.trim())
        .filter(word =>
            word.length >= 4 &&
            !STOP_WORDS.has(word) &&
            !HIGHLIGHT_STOP_TERMS.has(word),
        )
    const fragments: string[] = []
    for (let width = Math.min(5, words.length); width >= 2; width -= 1) {
        words.forEach((_, index) => {
            const fragment = words.slice(index, index + width)
            if (fragment.length === width) fragments.push(fragment.join(' '))
        })
    }
    return fragments.filter(fragment =>
        /\b(?:evidence|owns|highlights|connects|gates|blocks|required|workflow|review|board|risk|map)\b/i.test(fragment),
    )
}

const referencePdfEvidenceHighlightPhrases = (
    reference: ChatReferenceItem | null,
    query?: string,
): string[] => {
    if (!reference || reference.kind !== 'relationship') return []
    const concreteStatements = relationshipConcreteEvidenceStatements(reference, query)
    const phrases = concreteStatements.length > 0 ? concreteStatements : uniqueStrings([
        referencePreviewTextForQuery(reference, query),
        reference.body,
        reference.citationLabel,
    ])
        .flatMap(value =>
            value
                .split(/\n+/)
                .flatMap(line =>
                    line
                        .replace(/\s+/g, ' ')
                        .split(/(?=\[[^\]]+\]\s+(?:page|slide|table|chart)\s+\d+\b)|(?<=[.!?])\s+/i),
                ),
        )
        .map(cleanPdfEvidenceHighlightPhrase)
        .filter(value =>
            value.length >= 18 &&
            !PREVIEW_PROMPT_META_PATTERN.test(value) &&
            !/(?:--|->|\bevidence=|\bprovenance=)/i.test(value),
        )
    return uniqueStrings([
        ...phrases,
        ...phrases.flatMap(pdfEvidenceHighlightFragments),
    ]).slice(0, 10)
}

export const buildReferencePdfHighlightTerms = (
    reference: ChatReferenceItem | null,
    query?: string,
): string[] => {
    const evidencePhrases = referencePdfEvidenceHighlightPhrases(reference, query)
    if (evidencePhrases.length > 0) return evidencePhrases

    const graphAnchorTerms = uniqueStrings([
        ...(reference?.path?.nodes || []),
        ...(reference?.path?.relationships || []).flatMap(relationship => [
            relationship.source,
            relationship.target,
        ]),
        reference?.relationship?.source,
        reference?.relationship?.target,
    ])
    if (graphAnchorTerms.length > 0) {
        return graphAnchorTerms.slice(0, 18)
    }

    return uniqueStrings([
        ...collectTerms(reference, query),
        ...graphAnchorTerms,
        ...queryTerms(query),
    ]).slice(0, 18)
}

const extractArtifactText = (value: unknown): string => {
    if (typeof value === 'string') return value

    const record = asRecord(value)
    for (const key of ['content', 'text', 'markdown', 'structured_text', 'document_text', 'excerpt']) {
        const text = asString(record[key])
        if (text) return text
    }

    const chunks = Array.isArray(record.chunks) ? record.chunks : []
    const chunkText = chunks
        .map(chunk => {
            const chunkRecord = asRecord(chunk)
            return asString(chunkRecord.content) || asString(chunkRecord.text)
        })
        .filter(Boolean)
        .join('\n\n')
    if (chunkText) return chunkText

    const stages = Array.isArray(record.stages) ? record.stages : []
    for (const stage of stages) {
        const stageRecord = asRecord(stage)
        if (stageRecord.name !== 'structured_ingest_document') continue
        const sampleOutputs = asRecord(stageRecord.sample_outputs)
        const text = asString(sampleOutputs.text)
        if (text) return text
    }

    return JSON.stringify(value, null, 2)
}

const fetchReferenceDocument = async (endpoint: string): Promise<{ text: string; contentType: string }> => {
    const response = await fetch(withDownloadFlag(endpoint))
    if (!response.ok) {
        throw new Error(`Unable to load document (${response.status})`)
    }

    const contentType = response.headers.get('content-type') || ''
    if (contentType.includes('application/json')) {
        const json = await response.json()
        return { text: extractArtifactText(json), contentType }
    }

    return { text: await response.text(), contentType }
}

const PdfReferenceViewer: React.FC<{
    reference: ChatReferenceItem
    terms: string[]
    references?: ChatReferenceItem[]
}> = ({ reference, terms, references }) => {
    const theme = useTheme()
    const canvasRef = useRef<HTMLCanvasElement | null>(null)
    const containerRef = useRef<HTMLDivElement | null>(null)
    const documentRef = useRef<PDFDocumentProxy | null>(null)
    const renderTaskRef = useRef<RenderTask | null>(null)
    const [status, setStatus] = useState<'loading' | 'loaded' | 'error'>('loading')
    const [error, setError] = useState('')
    const [pageNumber, setPageNumber] = useState(1)
    const [pageCount, setPageCount] = useState(0)
    const [pageSize, setPageSize] = useState({ width: 0, height: 0 })
    const [highlights, setHighlights] = useState<PdfHighlightRect[]>([])
    const [containerWidth, setContainerWidth] = useState(0)
    const [rendering, setRendering] = useState(false)
    const [zoomSteps, setZoomSteps] = useState(0)

    const endpoint = normalizeEndpoint(reference.endpoint)
    const highlightReferences = useMemo(
        () => references && references.length > 0 ? references : [reference],
        [reference, references],
    )
    const citedPageNumber = useMemo(
        () => {
            for (const item of highlightReferences) {
                const page = pageNumberFromReference(item)
                if (page) return page
            }
            return null
        },
        [highlightReferences],
    )
    const citedPageNumbers = useMemo(
        () => pageNumbersFromReferences(highlightReferences),
        [highlightReferences],
    )
    const fileLevelEvidenceLabels = useMemo(
        () => uniqueStrings(
            highlightReferences.flatMap(item => {
                if (pageNumberFromReference(item)) return []
                const labels = referenceEvidenceLabels(item)
                if (labels.length > 0) return labels
                const title = referenceBriefTitle(item)
                return title ? [title] : []
            }),
        ).slice(0, 2),
        [highlightReferences],
    )

    useEffect(() => {
        const element = containerRef.current
        if (!element) return undefined

        const updateWidth = () => setContainerWidth(element.getBoundingClientRect().width)
        updateWidth()

        if (typeof ResizeObserver === 'undefined') {
            window.addEventListener('resize', updateWidth)
            return () => window.removeEventListener('resize', updateWidth)
        }

        const observer = new ResizeObserver(updateWidth)
        observer.observe(element)
        return () => observer.disconnect()
    }, [])

    useEffect(() => {
        let active = true
        const previousDocument = documentRef.current
        documentRef.current = null
        if (previousDocument) {
            void previousDocument.destroy().catch(() => undefined)
        }

        renderTaskRef.current?.cancel()
        renderTaskRef.current = null
        setStatus('loading')
        setError('')
        setPageNumber(1)
        setPageCount(0)
        setHighlights([])
        setPageSize({ width: 0, height: 0 })
        setZoomSteps(0)

        const loadPdf = async () => {
            if (!endpoint) throw new Error('PDF source is missing')
            const response = await fetch(withDownloadFlag(endpoint))
            if (!response.ok) throw new Error(`Unable to load PDF (${response.status})`)
            const pdf = await pdfjsLib.getDocument({ data: await response.arrayBuffer() }).promise
            if (!active) {
                await pdf.destroy().catch(() => undefined)
                return
            }
            documentRef.current = pdf
            setPageCount(pdf.numPages)
            setPageNumber(clampNumber(citedPageNumber || 1, 1, pdf.numPages))
            setStatus('loaded')
        }

        loadPdf().catch((loadError: unknown) => {
            if (!active) return
            setStatus('error')
            setError(loadError instanceof Error ? loadError.message : 'Unable to render PDF source')
        })

        return () => {
            active = false
        }
    }, [citedPageNumber, endpoint])

    useEffect(() => {
        if (status !== 'loaded' || !pageCount || !citedPageNumber) return
        setPageNumber(clampNumber(citedPageNumber, 1, pageCount))
    }, [citedPageNumber, pageCount, status])

    useEffect(() => {
        if (status !== 'loaded' || !documentRef.current || !canvasRef.current || containerWidth <= 0) return undefined

        let active = true
        renderTaskRef.current?.cancel()
        renderTaskRef.current = null
        setRendering(true)
        setHighlights([])

        const renderPage = async () => {
            const pdf = documentRef.current
            const canvas = canvasRef.current
            if (!pdf || !canvas) return

            const page = await pdf.getPage(pageNumber)
            if (!active) return

            const baseViewport = page.getViewport({ scale: 1 })
            const scale = pdfViewerRenderScale(
                baseViewport.width,
                containerWidth,
                typeof window !== 'undefined' ? window.innerWidth : containerWidth,
                zoomSteps,
            )
            const viewport = page.getViewport({ scale })
            const context = canvas.getContext('2d')
            if (!context) throw new Error('Canvas rendering is not available')

            const pixelRatio = window.devicePixelRatio || 1
            canvas.width = Math.floor(viewport.width * pixelRatio)
            canvas.height = Math.floor(viewport.height * pixelRatio)
            canvas.style.width = `${Math.floor(viewport.width)}px`
            canvas.style.height = `${Math.floor(viewport.height)}px`
            context.setTransform(1, 0, 0, 1, 0, 0)
            context.clearRect(0, 0, canvas.width, canvas.height)

            const renderTask = page.render({
                canvasContext: context,
                viewport,
                transform: pixelRatio !== 1 ? [pixelRatio, 0, 0, pixelRatio, 0, 0] : undefined,
            })
            renderTaskRef.current = renderTask
            await renderTask.promise
            if (!active) return

            const textContent = await page.getTextContent()
            if (!active) return

            const textHighlights = buildPdfHighlights(textContent.items, viewport, terms)
            const locationHighlights = textHighlights.length > 0
                ? []
                : buildPdfLocationHighlights(highlightReferences, viewport, pageNumber)
            setPageSize({ width: viewport.width, height: viewport.height })
            setHighlights(mergePdfHighlightRects([
                ...textHighlights,
                ...locationHighlights,
            ]).slice(0, 40))
            setRendering(false)
        }

        renderPage().catch((renderError: unknown) => {
            if (!active) return
            if (renderError instanceof Error && renderError.name === 'RenderingCancelledException') return
            setStatus('error')
            setError(renderError instanceof Error ? renderError.message : 'Unable to render PDF page')
            setRendering(false)
        })

        return () => {
            active = false
            renderTaskRef.current?.cancel()
        }
    }, [containerWidth, highlightReferences, pageNumber, status, terms, zoomSteps])

    useEffect(() => () => {
        renderTaskRef.current?.cancel()
        if (documentRef.current) {
            void documentRef.current.destroy().catch(() => undefined)
        }
    }, [])

    const openPdf = () => {
        if (!endpoint) return
        window.open(withDownloadFlag(endpoint), '_blank', 'noopener,noreferrer')
    }

    return (
        <Paper
            ref={containerRef}
            variant="outlined"
            sx={{
                height: '100%',
                minHeight: 0,
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
                borderColor: alpha(theme.palette.error.main, 0.16),
                bgcolor: 'background.paper',
            }}
        >
            <Box
                sx={{
                    px: 0.85,
                    py: 0.65,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 0.65,
                    borderBottom: '1px solid',
                    borderColor: 'divider',
                    flexWrap: 'wrap',
                }}
            >
                <Chip
                    size="small"
                    icon={<PdfIcon />}
                    label={referenceBasename(reference.filename || reference.title || 'PDF')}
                    sx={{
                        height: 24,
                        maxWidth: { xs: 190, sm: 360 },
                        bgcolor: alpha(theme.palette.error.main, 0.08),
                        '& .MuiChip-label': {
                            minWidth: 0,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                        },
                    }}
                />
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25 }}>
                    <Tooltip title="Previous page">
                        <span>
                            <IconButton
                                size="small"
                                disabled={pageNumber <= 1 || status !== 'loaded'}
                                onClick={() => setPageNumber(value => Math.max(1, value - 1))}
                            >
                                <PreviousIcon fontSize="small" />
                            </IconButton>
                        </span>
                    </Tooltip>
                    <Chip
                        size="small"
                        label={pageCount ? `Page ${pageNumber} / ${pageCount}` : 'Page'}
                        variant="outlined"
                        sx={{ height: 22, fontSize: '0.68rem' }}
                    />
                    <Tooltip title="Next page">
                        <span>
                            <IconButton
                                size="small"
                                disabled={pageNumber >= pageCount || status !== 'loaded'}
                                onClick={() => setPageNumber(value => Math.min(pageCount || value, value + 1))}
                            >
                                <NextIcon fontSize="small" />
                            </IconButton>
                        </span>
                    </Tooltip>
                </Box>
                {citedPageNumbers.length > 0 && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.35, flexWrap: 'wrap' }}>
                        <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.66rem', fontWeight: 800 }}>
                            Cited
                        </Typography>
                        {citedPageNumbers.map(page => (
                            <Chip
                                key={`pdf-cited-page-${page}`}
                                size="small"
                                label={`p. ${page}`}
                                clickable={status === 'loaded'}
                                color={page === pageNumber ? 'warning' : 'default'}
                                variant={page === pageNumber ? 'filled' : 'outlined'}
                                onClick={() => setPageNumber(clampNumber(page, 1, pageCount || page))}
                                sx={{
                                    height: 22,
                                    fontSize: '0.66rem',
                                    borderColor: page === pageNumber
                                        ? alpha(theme.palette.warning.main, 0.46)
                                        : alpha(theme.palette.divider, 0.8),
                                    bgcolor: page === pageNumber
                                        ? alpha(theme.palette.warning.main, 0.22)
                                        : alpha(theme.palette.background.paper, 0.22),
                                }}
                            />
                        ))}
                    </Box>
                )}
                {highlights.length > 0 && (
                    <Chip
                        size="small"
                        label={`${highlights.length} highlights`}
                        variant="outlined"
                        sx={{ height: 22, fontSize: '0.68rem', bgcolor: alpha(theme.palette.warning.main, 0.08) }}
                    />
                )}
                {citedPageNumbers.length === 0 && fileLevelEvidenceLabels.length > 0 && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.35, flexWrap: 'wrap' }}>
                        <Chip
                            size="small"
                            label="File-level evidence"
                            variant="outlined"
                            sx={{
                                height: 22,
                                fontSize: '0.66rem',
                                borderColor: alpha(theme.palette.info.main, 0.34),
                                bgcolor: alpha(theme.palette.info.main, 0.08),
                            }}
                        />
                        {fileLevelEvidenceLabels.map(label => (
                            <Chip
                                key={`pdf-file-level-${label}`}
                                size="small"
                                label={label}
                                variant="outlined"
                                sx={{
                                    height: 22,
                                    maxWidth: { xs: 150, sm: 220 },
                                    fontSize: '0.66rem',
                                    '& .MuiChip-label': {
                                        minWidth: 0,
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                    },
                                }}
                            />
                        ))}
                    </Box>
                )}
                <Box sx={{ flex: 1 }} />
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.15 }}>
                    <Tooltip title="Zoom out">
                        <span>
                            <IconButton
                                size="small"
                                disabled={status !== 'loaded' || zoomSteps <= -2}
                                onClick={() => setZoomSteps(value => Math.max(-2, value - 1))}
                            >
                                <ZoomOutIcon fontSize="small" />
                            </IconButton>
                        </span>
                    </Tooltip>
                    <Chip
                        size="small"
                        label={zoomSteps === 0 ? 'Fit' : `${zoomSteps > 0 ? '+' : ''}${zoomSteps}`}
                        variant="outlined"
                        sx={{ height: 22, minWidth: 44, fontSize: '0.66rem' }}
                    />
                    <Tooltip title="Zoom in">
                        <span>
                            <IconButton
                                size="small"
                                disabled={status !== 'loaded' || zoomSteps >= 3}
                                onClick={() => setZoomSteps(value => Math.min(3, value + 1))}
                            >
                                <ZoomInIcon fontSize="small" />
                            </IconButton>
                        </span>
                    </Tooltip>
                </Box>
                <Tooltip title="Open raw PDF">
                    <IconButton size="small" onClick={openPdf}>
                        <OpenIcon fontSize="small" />
                    </IconButton>
                </Tooltip>
            </Box>

            <Box
                sx={{
                    flex: 1,
                    minHeight: 0,
                    overflow: 'auto',
                    p: { xs: 1, sm: 1.25 },
                    bgcolor: alpha(theme.palette.common.black, theme.palette.mode === 'dark' ? 0.14 : 0.035),
                }}
            >
                {status === 'loading' && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 1.5 }}>
                        <CircularProgress size={18} />
                        <Typography variant="body2" color="text.secondary">
                            Loading PDF source
                        </Typography>
                    </Box>
                )}
                {status === 'error' && (
                    <Alert
                        severity="warning"
                        action={
                            endpoint ? (
                                <Button size="small" onClick={openPdf}>
                                    Open PDF
                                </Button>
                            ) : undefined
                        }
                    >
                        {error || 'Unable to render PDF source.'}
                    </Alert>
                )}
                {status === 'loaded' && (
                    <Box
                        sx={{
                            width: pageSize.width || '100%',
                            minHeight: pageSize.height || 360,
                            mx: 'auto',
                            position: 'relative',
                            borderRadius: '8px',
                            boxShadow: theme.shadows[3],
                            bgcolor: 'background.paper',
                        }}
                    >
                        {rendering && (
                            <Box
                                sx={{
                                    position: 'absolute',
                                    inset: 0,
                                    zIndex: 2,
                                    display: 'flex',
                                    alignItems: 'flex-start',
                                    justifyContent: 'flex-end',
                                    p: 1,
                                    pointerEvents: 'none',
                                }}
                            >
                                <Chip
                                    size="small"
                                    icon={<CircularProgress size={12} />}
                                    label="Rendering"
                                    sx={{ height: 24, bgcolor: alpha(theme.palette.background.paper, 0.9) }}
                                />
                            </Box>
                        )}
                        <Box
                            component="canvas"
                            ref={canvasRef}
                            sx={{
                                display: 'block',
                                width: pageSize.width ? `${Math.floor(pageSize.width)}px` : '100%',
                                height: pageSize.height ? `${Math.floor(pageSize.height)}px` : 360,
                                borderRadius: '8px',
                            }}
                        />
                        {highlights.map((highlight, index) => (
                            <Tooltip key={`${highlight.label}-${index}`} title={highlight.label}>
                                <Box
                                    sx={{
                                        position: 'absolute',
                                        left: highlight.left,
                                        top: highlight.top,
                                        width: highlight.width,
                                        height: highlight.height,
                                        borderRadius: '3px',
                                        bgcolor: highlight.kind === 'location'
                                            ? 'rgba(250, 204, 21, 0.055)'
                                            : 'rgba(250, 204, 21, 0.2)',
                                        border: highlight.kind === 'location'
                                            ? '1px solid rgba(245, 158, 11, 0.5)'
                                            : '1px solid rgba(250, 204, 21, 0.6)',
                                        boxShadow: highlight.kind === 'location'
                                            ? 'inset 0 0 0 1px rgba(255, 255, 255, 0.28)'
                                            : '0 0 0 1px rgba(30, 41, 59, 0.14)',
                                        pointerEvents: 'auto',
                                    }}
                                />
                            </Tooltip>
                        ))}
                    </Box>
                )}
            </Box>
        </Paper>
    )
}

const HighlightedText: React.FC<{ text: string; terms: string[] }> = ({ text, terms }) => {
    const usableTerms = useMemo(
        () => uniqueStrings(terms)
            .filter(term => term.length >= 2)
            .sort((a, b) => b.length - a.length)
            .slice(0, 32),
        [terms],
    )

    if (!text || usableTerms.length === 0) return <>{text}</>

    const matcher = new RegExp(`(${usableTerms.map(highlightPattern).join('|')})`, 'gi')
    const lowerTerms = new Set(usableTerms.map(term => term.toLowerCase()))

    return (
        <>
            {text.split(matcher).map((part, index) => {
                if (!part) return null
                const isMatch = lowerTerms.has(part.toLowerCase())
                return isMatch ? (
                    <Box
                        key={`${part}-${index}`}
                        component="mark"
                        sx={{
                            px: 0.25,
                            borderRadius: 0.5,
                            color: 'inherit',
                            bgcolor: 'rgba(250, 204, 21, 0.28)',
                            boxShadow: 'inset 0 -1px 0 rgba(250, 204, 21, 0.65)',
                            boxDecorationBreak: 'clone',
                            WebkitBoxDecorationBreak: 'clone',
                        }}
                    >
                        {part}
                    </Box>
                ) : part
            })}
        </>
    )
}

const DocumentTextViewer: React.FC<{ text: string; terms: string[] }> = ({ text, terms }) => {
    const theme = useTheme()
    const review = useMemo(
        () => buildChatReferenceDocumentReview(text, terms, { contextRadius: 1, maxRows: 180 }),
        [terms, text],
    )
    const [mode, setMode] = useState<'matches' | 'all'>(review.matchCount > 0 ? 'matches' : 'all')

    useEffect(() => {
        setMode(review.matchCount > 0 ? 'matches' : 'all')
    }, [review.matchCount, text])

    const showMatches = mode === 'matches' && review.matchCount > 0
    const textPanelMaxHeight = { xs: 360, md: 'calc(100vh - 470px)' }

    return (
        <Stack
            data-testid="reference-document-text-viewer"
            spacing={0.75}
            sx={{ height: '100%', minHeight: 0 }}
        >
            <Box
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 0.75,
                    flexWrap: 'wrap',
                    px: { xs: 1, sm: 1.15 },
                    pt: 0.85,
                }}
            >
                <ToggleButtonGroup
                    size="small"
                    exclusive
                    value={mode}
                    onChange={(_event, value: 'matches' | 'all' | null) => {
                        if (value) setMode(value)
                    }}
                    sx={{
                        '& .MuiToggleButton-root': {
                            minHeight: 26,
                            px: 1,
                            py: 0.25,
                            textTransform: 'none',
                            fontSize: '0.7rem',
                            fontWeight: 850,
                        },
                    }}
                >
                    <ToggleButton value="matches" disabled={review.matchCount === 0}>
                        Nearby
                    </ToggleButton>
                    <ToggleButton value="all">
                        Full text
                    </ToggleButton>
                </ToggleButtonGroup>
                <Box sx={{ display: 'flex', gap: 0.4, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                    <Chip
                        size="small"
                        label={`${review.lineCount} ${review.lineCount === 1 ? 'line' : 'lines'}`}
                        variant="outlined"
                        sx={{ height: 22, fontSize: '0.66rem' }}
                    />
                    {review.matchCount > 0 && (
                        <Chip
                            size="small"
                            label={`${review.matchCount} ${review.matchCount === 1 ? 'match' : 'matches'}`}
                            sx={{ height: 22, fontSize: '0.66rem', bgcolor: alpha(theme.palette.warning.main, 0.12) }}
                        />
                    )}
                </Box>
            </Box>

            {showMatches ? (
                <Box
                    sx={{
                        flex: 1,
                        minHeight: 0,
                        maxHeight: textPanelMaxHeight,
                        overflow: 'auto',
                        px: { xs: 1, sm: 1.15 },
                        pb: 1.15,
                    }}
                >
                    <Stack spacing={0.35}>
                        {review.rows.map((row, index) => (
                            row.type === 'gap' ? (
                                <Typography
                                    key={`document-review-gap-${index}-${row.hiddenLines}`}
                                    variant="caption"
                                    color="text.secondary"
                                    sx={{
                                        display: 'block',
                                        px: 0.75,
                                        py: 0.25,
                                        fontSize: '0.66rem',
                                    }}
                                >
                                    {row.hiddenLines} {row.hiddenLines === 1 ? 'line' : 'lines'} skipped
                                </Typography>
                            ) : (
                                <Box
                                    key={`document-review-line-${row.lineNumber}`}
                                    sx={{
                                        position: 'relative',
                                        display: 'grid',
                                        gridTemplateColumns: { xs: '30px minmax(0, 1fr)', sm: '38px minmax(0, 1fr)' },
                                        gap: { xs: 0.5, sm: 0.7 },
                                        alignItems: 'flex-start',
                                        px: 0.75,
                                        py: 0.42,
                                        borderRadius: '6px',
                                        border: '1px solid',
                                        borderColor: row.matched
                                            ? alpha(theme.palette.warning.main, 0.22)
                                            : 'transparent',
                                        bgcolor: row.matched
                                            ? alpha(theme.palette.warning.main, 0.072)
                                            : alpha(theme.palette.background.default, 0.16),
                                        '&::before': row.matched
                                            ? {
                                                content: '""',
                                                position: 'absolute',
                                                left: 0,
                                                top: 6,
                                                bottom: 6,
                                                width: 2,
                                                borderRadius: 999,
                                                bgcolor: alpha(theme.palette.warning.main, 0.58),
                                            }
                                            : undefined,
                                    }}
                                >
                                    <Typography
                                        variant="caption"
                                        color="text.secondary"
                                        sx={{
                                            fontFamily: 'monospace',
                                            textAlign: 'right',
                                            lineHeight: 1.55,
                                            userSelect: 'none',
                                        }}
                                    >
                                        {row.lineNumber}
                                    </Typography>
                                    <Typography
                                        component="pre"
                                        variant="body2"
                                        sx={{
                                            m: 0,
                                            minWidth: 0,
                                            whiteSpace: 'pre-wrap',
                                            wordBreak: 'break-word',
                                            fontFamily: theme.typography.fontFamily,
                                            fontSize: { xs: '0.78rem', sm: '0.82rem' },
                                            lineHeight: 1.55,
                                        }}
                                    >
                                        <HighlightedText text={row.text || ' '} terms={terms} />
                                    </Typography>
                                </Box>
                            )
                        ))}
                    </Stack>
                </Box>
            ) : (
                <Box
                    component="pre"
                    sx={{
                        m: 0,
                        flex: 1,
                        minHeight: 0,
                        maxHeight: textPanelMaxHeight,
                        overflow: 'auto',
                        p: { xs: 1.1, sm: 1.5 },
                        pt: 0.75,
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                        color: 'text.primary',
                        fontFamily: theme.typography.fontFamily,
                        fontSize: '0.86rem',
                        lineHeight: 1.62,
                    }}
                >
                    <HighlightedText text={text || 'No document text available.'} terms={terms} />
                </Box>
            )}
        </Stack>
    )
}

const layoutMapToneColor = (
    tone: ReferenceLayoutMapBox['tone'],
    theme: Theme,
    accentColor: string,
): string => {
    if (tone === 'relationship') return accentColor
    if (tone === 'chart') return theme.palette.warning.main
    return theme.palette.info.main
}

export const layoutMapCalloutPosition = (
    box: Pick<ReferenceLayoutMapBox, 'left' | 'top' | 'height'>,
    index: number,
): { left: number; top: number } => {
    const stagger = index % 2 === 0 ? 0 : 5
    const top = box.top <= 16
        ? clampNumber(box.top + box.height + 3 + stagger, 4, 88)
        : clampNumber(box.top - 13 + stagger, 4, 88)

    return {
        left: clampNumber(box.left, 2, 72),
        top,
    }
}

export const layoutMapPinPosition = (
    box: Pick<ReferenceLayoutMapBox, 'left' | 'top' | 'width' | 'height'>,
    index: number,
): { left: number; top: number } => {
    const visibleWidth = Math.min(box.width, 100 - box.left)
    const visibleHeight = Math.min(box.height, 100 - box.top)
    const offsets = [
        { left: -2.6, top: -1.8 },
        { left: 2.6, top: 1.8 },
        { left: 0, top: -5 },
        { left: 0, top: 5 },
        { left: -5, top: 3 },
        { left: 5, top: -3 },
    ]
    const offset = offsets[index % offsets.length]

    return {
        left: clampNumber(box.left + visibleWidth / 2 + offset.left, 3, 97),
        top: clampNumber(box.top + visibleHeight / 2 + offset.top, 5, 95),
    }
}

export const layoutMapViewport = (boxes: Array<Pick<ReferenceLayoutMapBox, 'left' | 'top' | 'width' | 'height'>>): ReferenceLayoutMapViewport => {
    if (boxes.length === 0) {
        return { left: 0, top: 0, width: 100, height: 100, focused: false }
    }

    const minLeft = Math.min(...boxes.map(box => box.left))
    const minTop = Math.min(...boxes.map(box => box.top))
    const maxRight = Math.max(...boxes.map(box => box.left + box.width))
    const maxBottom = Math.max(...boxes.map(box => box.top + box.height))
    const rawWidth = Math.max(1, maxRight - minLeft)
    const rawHeight = Math.max(1, maxBottom - minTop)
    const shouldFocus = rawWidth < 72 || rawHeight < 62
    if (!shouldFocus) return { left: 0, top: 0, width: 100, height: 100, focused: false }

    const padX = rawWidth < 24 ? 16 : 10
    const padY = rawHeight < 20 ? 18 : 12
    const left = clampNumber(minLeft - padX, 0, 96)
    const top = clampNumber(minTop - padY, 0, 96)
    const right = clampNumber(maxRight + padX, left + 4, 100)
    const bottom = clampNumber(maxBottom + padY, top + 4, 100)

    return {
        left,
        top,
        width: Math.max(4, right - left),
        height: Math.max(4, bottom - top),
        focused: true,
    }
}

export const layoutMapViewportBox = (
    box: ReferenceLayoutMapBox,
    viewport: ReferenceLayoutMapViewport,
): ReferenceLayoutMapBox => ({
    ...box,
    left: clampNumber(((box.left - viewport.left) / viewport.width) * 100, 0, 100),
    top: clampNumber(((box.top - viewport.top) / viewport.height) * 100, 0, 100),
    width: clampNumber((box.width / viewport.width) * 100, 4, 100),
    height: clampNumber((box.height / viewport.height) * 100, 4, 100),
})

const ReferenceLayoutMapView: React.FC<{
    map: ReferenceLayoutMapSummary
    accentColor: string
}> = ({ map, accentColor }) => {
    const theme = useTheme()
    const viewport = layoutMapViewport(map.boxes)
    const visibleBoxes = map.boxes.map(box => layoutMapViewportBox(box, viewport))
    const showRegionCallouts = visibleBoxes.length <= 3

    return (
        <Box sx={{ px: { xs: 1, sm: 1.1 }, pt: 0.85, pb: 1 }}>
            <Box
                sx={{
                    display: 'grid',
                    gridTemplateColumns: { xs: '1fr', sm: 'minmax(0, 1fr) auto' },
                    gap: 0.75,
                    alignItems: 'center',
                    mb: 0.75,
                }}
            >
                <Box sx={{ minWidth: 0 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontWeight: 850 }}>
                        Slide map
                    </Typography>
                    <Typography variant="caption" noWrap title={map.title} sx={{ display: 'block', fontWeight: 900 }}>
                        {map.title}
                    </Typography>
                </Box>
                {map.subtitle && (
                    <Box sx={{ justifySelf: { sm: 'end' }, display: 'flex', alignItems: 'center', gap: 0.45, minWidth: 0, maxWidth: { xs: '100%', sm: 360 } }}>
                        {viewport.focused && (
                            <Chip
                                size="small"
                                label="Focused"
                                variant="outlined"
                                sx={{ height: 20, flexShrink: 0, fontSize: '0.62rem', color: accentColor, borderColor: alpha(accentColor, 0.36) }}
                            />
                        )}
                        <Typography
                            variant="caption"
                            color="text.secondary"
                            noWrap
                            sx={{ minWidth: 0 }}
                        >
                            {map.subtitle}
                        </Typography>
                    </Box>
                )}
            </Box>
            <Box
                data-testid="reference-layout-map"
                sx={{
                    position: 'relative',
                    height: { xs: 148, sm: 188 },
                    minHeight: { xs: 148, sm: 188 },
                    flex: '0 0 auto',
                    borderRadius: '8px',
                    overflow: 'hidden',
                    border: '1px solid',
                    borderColor: alpha(accentColor, 0.24),
                    bgcolor: alpha(theme.palette.common.white, 0.035),
                    backgroundImage: `
                        linear-gradient(${alpha(theme.palette.common.white, 0.045)} 1px, transparent 1px),
                        linear-gradient(90deg, ${alpha(theme.palette.common.white, 0.045)} 1px, transparent 1px)
                    `,
                    backgroundSize: '12.5% 25%, 12.5% 25%',
                }}
            >
                {visibleBoxes.map((box, index) => {
                    const color = layoutMapToneColor(box.tone, theme, accentColor)
                    return (
                        <Tooltip key={box.id} title={[box.label, box.detail].filter(Boolean).join(': ')}>
                            <Box
                                data-testid={`reference-layout-map-region-${index + 1}`}
                                sx={{
                                    position: 'absolute',
                                    left: `${box.left}%`,
                                    top: `${box.top}%`,
                                    width: `${Math.min(box.width, 100 - box.left)}%`,
                                    height: `${Math.min(box.height, 100 - box.top)}%`,
                                    minWidth: 24,
                                    minHeight: 20,
                                    zIndex: map.boxes.length - index,
                                    border: '2px solid',
                                    borderColor: alpha(color, index === 0 ? 0.92 : 0.62),
                                    bgcolor: alpha(color, index === 0 ? 0.2 : 0.1),
                                    boxShadow: `0 0 0 1px ${alpha(theme.palette.background.paper, 0.72)}`,
                                    borderRadius: '6px',
                                }}
                            />
                        </Tooltip>
                    )
                })}
                {visibleBoxes.map((box, index) => {
                    const color = layoutMapToneColor(box.tone, theme, accentColor)
                    const pin = layoutMapPinPosition(box, index)
                    return (
                        <Tooltip key={`layout-map-pin-${box.id}`} title={[box.label, box.detail].filter(Boolean).join(': ')}>
                            <Box
                                data-testid={`reference-layout-map-pin-${index + 1}`}
                                sx={{
                                    position: 'absolute',
                                    left: `${pin.left}%`,
                                    top: `${pin.top}%`,
                                    transform: 'translate(-50%, -50%)',
                                    zIndex: 16 + index,
                                    minWidth: 20,
                                    height: 20,
                                    px: 0.35,
                                    borderRadius: '6px',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    bgcolor: alpha(theme.palette.background.paper, 0.88),
                                    border: '1px solid',
                                    borderColor: alpha(color, 0.72),
                                    color: 'text.primary',
                                    fontWeight: 950,
                                    fontSize: '0.66rem',
                                    lineHeight: 1,
                                    boxShadow: `0 4px 12px ${alpha(theme.palette.common.black, 0.2)}`,
                                }}
                            >
                                {index + 1}
                            </Box>
                        </Tooltip>
                    )
                })}
                {showRegionCallouts && visibleBoxes.map((box, index) => {
                    const color = layoutMapToneColor(box.tone, theme, accentColor)
                    const callout = layoutMapCalloutPosition(box, index)
                    return (
                        <Tooltip key={`layout-map-callout-${box.id}`} title={[box.label, box.detail].filter(Boolean).join(': ')}>
                            <Box
                                data-testid={`reference-layout-map-callout-${index + 1}`}
                                sx={{
                                    position: 'absolute',
                                    left: `${callout.left}%`,
                                    top: `${callout.top}%`,
                                    zIndex: 20 + index,
                                    maxWidth: { xs: 158, sm: 220 },
                                    minHeight: 22,
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 0.45,
                                    px: 0.55,
                                    py: 0.25,
                                    borderRadius: '7px',
                                    border: '1px solid',
                                    borderColor: alpha(color, 0.42),
                                    bgcolor: alpha(theme.palette.background.paper, 0.9),
                                    boxShadow: `0 5px 16px ${alpha(theme.palette.common.black, 0.18)}`,
                                    pointerEvents: 'auto',
                                }}
                            >
                                <Typography
                                    variant="caption"
                                    sx={{
                                        width: 16,
                                        height: 16,
                                        borderRadius: '4px',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        flexShrink: 0,
                                        fontSize: '0.58rem',
                                        fontWeight: 950,
                                        color,
                                        bgcolor: alpha(color, 0.12),
                                    }}
                                >
                                    {index + 1}
                                </Typography>
                                <Typography
                                    variant="caption"
                                    noWrap
                                    sx={{
                                        minWidth: 0,
                                        fontSize: { xs: '0.62rem', sm: '0.66rem' },
                                        fontWeight: 850,
                                    }}
                                >
                                    {box.label}
                                </Typography>
                            </Box>
                        </Tooltip>
                    )
                })}
            </Box>
            {map.boxes.length > 1 && (
                <Box
                    sx={{
                        display: 'grid',
                        gridTemplateColumns: { xs: '1fr', sm: 'auto minmax(0, 1fr)' },
                        gap: 0.55,
                        alignItems: 'start',
                        mt: 0.65,
                    }}
                >
                    <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 850, pt: 0.15 }}>
                        Evidence route
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 0.45, flexWrap: 'wrap', minWidth: 0 }}>
                        {map.boxes.slice(0, 4).map((box, index) => {
                            const color = layoutMapToneColor(box.tone, theme, accentColor)
                            return (
                                <Chip
                                    key={`layout-map-legend-${box.id}`}
                                    size="small"
                                    label={`${index + 1}. ${box.label}`}
                                    variant="outlined"
                                    sx={{
                                        height: 22,
                                        maxWidth: { xs: 190, sm: 250 },
                                        fontSize: '0.64rem',
                                        borderColor: alpha(color, 0.32),
                                        bgcolor: alpha(color, 0.055),
                                        '& .MuiChip-label': {
                                            minWidth: 0,
                                            overflow: 'hidden',
                                            textOverflow: 'ellipsis',
                                        },
                                    }}
                                />
                            )
                        })}
                        {map.boxes.length > 4 && (
                            <Chip
                                size="small"
                                label={`+${map.boxes.length - 4}`}
                                variant="outlined"
                                sx={{ height: 22, fontSize: '0.64rem' }}
                            />
                        )}
                    </Box>
                </Box>
            )}
        </Box>
    )
}

export const ReferenceExplorer: React.FC<ReferenceExplorerProps> = ({ references, query, density = 'cards' }) => {
    const theme = useTheme()
    const [open, setOpen] = useState(false)
    const [selectedId, setSelectedId] = useState<string>('')
    const [selectedOverrideReference, setSelectedOverrideReference] = useState<ChatReferenceItem | null>(null)
    const [kindFilter, setKindFilter] = useState<ReferenceFilterValue>('all')
    const [documentFilter, setDocumentFilter] = useState<string>('all')
    const [drawerSearch, setDrawerSearch] = useState('')
    const [artifactCache, setArtifactCache] = useState<Record<string, ArtifactState>>({})
    const [copied, setCopied] = useState(false)
    const [sourceTextExpanded, setSourceTextExpanded] = useState(false)
    const [showAllDrawerReferences, setShowAllDrawerReferences] = useState(false)
    const requestedEndpoints = useRef(new Set<string>())

    const orderedReferences = useMemo(() => {
        const priority: Record<ChatReferenceKind, number> = {
            path: 0,
            relationship: 1,
            passage: 2,
            document: 3,
            artifact: 4,
        }
        return dedupeChatReferencesForDisplay([...references].sort(
            (left, right) =>
                Number(hasRequestedEvidenceFamilyMarker(right)) - Number(hasRequestedEvidenceFamilyMarker(left)) ||
                referencePathQueryCoverageScore(right, query) - referencePathQueryCoverageScore(left, query) ||
                referencePathSemanticStrengthScore(right) - referencePathSemanticStrengthScore(left) ||
                referencePathQueryOrderScore(right, query) - referencePathQueryOrderScore(left, query) ||
                referenceAnswerFitScore(right, query) - referenceAnswerFitScore(left, query) ||
                (referenceQueryRelevanceScore(right, query) * 42) - (referenceQueryRelevanceScore(left, query) * 42) ||
                referenceReviewScore(right) - referenceReviewScore(left) ||
            priority[left.kind] - priority[right.kind] ||
                (right.score || 0) - (left.score || 0),
        ))
    }, [query, references])
    const reviewReferences = useMemo(() => {
        const curated = orderedReferences.filter(reference => !isSyntheticFilenameRelationshipReference(reference))
        return curated.length > 0 ? curated : orderedReferences
    }, [orderedReferences])
    const referenceCounts = useMemo(() => reviewReferences.reduce(
        (counts, reference) => ({
            ...counts,
            [reference.kind]: (counts[reference.kind] || 0) + 1,
        }),
        {} as Record<ChatReferenceKind, number>,
    ), [reviewReferences])
    const topEvidenceFamilies = useMemo(() => {
        const counts = new Map<string, number>()
        reviewReferences.forEach(reference => {
            referenceEvidenceLabels(reference).forEach(label => {
                counts.set(label, (counts.get(label) || 0) + 1)
            })
        })

        const entries = Array.from(counts.entries())
        const hasHighSignal = entries.some(([label]) => isHighSignalEvidenceLabel(label))
        return entries
            .filter(([label]) => !hasHighSignal || isHighSignalEvidenceLabel(label))
            .sort((a, b) =>
                evidenceFamilyDisplayScore(b[0]) - evidenceFamilyDisplayScore(a[0]) ||
                b[1] - a[1] ||
                a[0].localeCompare(b[0]),
            )
            .slice(0, 4)
    }, [reviewReferences])
    const technicalReferenceCount = useMemo(
        () => reviewReferences.filter(reference =>
            isTechnicalReviewReference(reference) ||
            isLowSignalGraphRAGSummaryReference(reference, query),
        ).length,
        [reviewReferences, query],
    )
    const primaryReferenceCount = Math.max(0, reviewReferences.length - technicalReferenceCount)
    const defaultReferenceCount = primaryReferenceCount > 0 ? primaryReferenceCount : reviewReferences.length
    const referenceFilterChips = useMemo(
        () => buildGraphRAGReferenceFilterChips({
            defaultReferenceCount,
            referenceCounts,
            technicalReferenceCount,
        }),
        [defaultReferenceCount, referenceCounts, technicalReferenceCount],
    )
    const sourceGroups = useMemo<ReferenceDocumentGroup[]>(() => {
        const groups = new Map<string, ReferenceDocumentGroup>()

        reviewReferences.forEach(reference => {
            const key = referenceDocumentKey(reference)
            const existing = groups.get(key)
            const evidenceLabels = referenceEvidenceLabels(reference)
            const locationLabels = referenceLocationLabels(reference)

            if (existing) {
                existing.references.push(reference)
                existing.counts[reference.kind] = (existing.counts[reference.kind] || 0) + 1
                existing.evidenceLabels = uniqueStrings([...existing.evidenceLabels, ...evidenceLabels]).slice(0, 4)
                existing.locationLabels = uniqueStrings([...existing.locationLabels, ...locationLabels]).slice(0, 3)
                return
            }

            groups.set(key, {
                key,
                title: referenceDocumentTitle(reference),
                subtitle: referenceDocumentSubtitle(reference),
                references: [reference],
                counts: { [reference.kind]: 1 },
                bestReference: reference,
                evidenceLabels: evidenceLabels.slice(0, 4),
                locationLabels: locationLabels.slice(0, 3),
            })
        })

        const requestedFileSet = new Set(queryFilenames(query))
        return Array.from(groups.values()).map(group => {
            const namedFileBestReference = requestedFileSet.has(group.key)
                ? [...group.references].sort((left, right) =>
                    namedFileReferenceKindPriority(left) - namedFileReferenceKindPriority(right) ||
                    namedFileReferenceScore(right, query) - namedFileReferenceScore(left, query) ||
                    referenceReviewScore(right) - referenceReviewScore(left),
                )[0]
                : undefined
            return {
                ...group,
                bestReference:
                    namedFileBestReference ||
                    preferredReviewReference(group.references) ||
                    group.bestReference,
            }
        }).sort(
            (left, right) =>
                referenceReviewScore(right.bestReference) - referenceReviewScore(left.bestReference) ||
                right.references.length - left.references.length ||
                left.title.localeCompare(right.title),
        )
    }, [query, reviewReferences])
    const sourceNavigationGroups = useMemo(() => {
        const documentGroups = sourceGroups.filter(isSourceNavigationGroup)
        return documentGroups.length > 0 ? documentGroups : sourceGroups
    }, [sourceGroups])
    const showDrawerSearch = shouldShowReferenceExplorerDrawerSearch({
        referenceCount: reviewReferences.length,
        sourceGroupCount: sourceNavigationGroups.length,
        drawerSearch,
        kindFilter,
    })
    const showCompactDrawerSearch = shouldShowReferenceExplorerDrawerSearch({
        referenceCount: reviewReferences.length,
        sourceGroupCount: sourceNavigationGroups.length,
        drawerSearch,
        kindFilter,
        compact: true,
    })
    const filteredReferences = useMemo(() => {
        const normalizedSearch = drawerSearch.trim().toLowerCase()
        const hideTechnicalByDefault = kindFilter === 'all' && !normalizedSearch && primaryReferenceCount > 0
        const references = reviewReferences.filter(reference => {
            const technical =
                isTechnicalReviewReference(reference) ||
                isLowSignalGraphRAGSummaryReference(reference, query)
            if (kindFilter === 'technical' && !technical) return false
            if (kindFilter !== 'all' && kindFilter !== 'technical' && reference.kind !== kindFilter) return false
            if (hideTechnicalByDefault && technical) return false
            if (documentFilter !== 'all' && referenceDocumentKey(reference) !== documentFilter) return false
            if (!normalizedSearch) return true
            return [
                reference.title,
                reference.subtitle,
                reference.filename,
                reference.body,
                reference.relationship?.source,
                reference.relationship?.relation,
                reference.relationship?.target,
                ...(reference.sourceLocations || []).flatMap(location => [
                    location.label,
                    location.value,
                    location.category,
                ]),
                ...(reference.path?.nodes || []),
                ...(reference.path?.relationships || []).flatMap(relationship => [
                    relationship.source,
                    relationship.relation,
                    relationship.target,
                    relationship.evidence,
                ]),
            ].some(value => String(value || '').toLowerCase().includes(normalizedSearch))
        })

        if (normalizedSearch || kindFilter !== 'all') return references

        return diversifyReferenceExplorerDrawerReferences(references, query)
    }, [documentFilter, drawerSearch, kindFilter, primaryReferenceCount, query, reviewReferences])
    const drawerReferenceSelection = selectReferenceDrawerRows(filteredReferences, {
        showAll: showAllDrawerReferences,
        search: drawerSearch,
        kindFilter,
    })
    const drawerReferenceRows = drawerReferenceSelection.rows
    const hiddenDrawerReferenceCount = drawerReferenceSelection.hiddenCount
    const visibleReferences = useMemo(() => {
        const selectedReferences: ChatReferenceItem[] = []
        const selectedIds = new Set<string>()
        const preferredKinds: ChatReferenceKind[] = ['path', 'relationship', 'passage', 'document', 'artifact']

        preferredKinds.forEach(kind => {
            const reference = reviewReferences.find(item => item.kind === kind && !selectedIds.has(chatReferenceDisplayDedupeKey(item)))
            if (!reference) return
            selectedReferences.push(reference)
            selectedIds.add(chatReferenceDisplayDedupeKey(reference))
        })

        reviewReferences.forEach(reference => {
            const key = chatReferenceDisplayDedupeKey(reference)
            if (selectedReferences.length >= MAX_VISIBLE_REFERENCES || selectedIds.has(key)) return
            selectedReferences.push(reference)
            selectedIds.add(key)
        })

        return selectedReferences.slice(0, MAX_VISIBLE_REFERENCES)
    }, [reviewReferences])
    const answerEvidenceReferences = useMemo(
        () => reviewReferences.filter(reference => reference.sourceRole === 'answer_evidence'),
        [reviewReferences],
    )
    const selected = selectReferenceExplorerActiveReference(
        reviewReferences,
        selectedId,
        selectedOverrideReference,
    )
    const selectedIsRequestedEvidence = selected ? hasRequestedEvidenceFamilyMarker(selected) : false
    const selectedSourceGroup = selected
        ? sourceGroups.find(group => group.key === referenceDocumentKey(selected))
        : sourceNavigationGroups[0] || sourceGroups[0]
    const selectedSourceGroupIsNavigable = Boolean(
        selectedSourceGroup && sourceNavigationGroups.some(group => group.key === selectedSourceGroup.key),
    )
    const selectedReviewReferences = useMemo(
        () => selectFocusedReferenceReviewReferences(
            selected,
            selectedSourceGroup?.references || [],
            answerEvidenceReferences,
        ),
        [answerEvidenceReferences, selected, selectedSourceGroup],
    )
    const terms = useMemo(() => collectTerms(selected, query), [query, selected])
    const selectedEndpoint = normalizeEndpoint(selected?.endpoint)
    const selectedPdfViewerReference = useMemo(
        () => selectReferencePdfViewerReference(selected, selectedSourceGroup?.references || []),
        [selected, selectedSourceGroup],
    )
    const selectedPdfViewerEndpoint = normalizeEndpoint(selectedPdfViewerReference?.endpoint)
    const selectedHasPdfViewer = Boolean(selectedPdfViewerReference && selectedPdfViewerEndpoint)
    const selectedIsPdf = isPdfReference(selected) || selectedHasPdfViewer
    const selectedSourceViewReference = selectedHasPdfViewer ? selectedPdfViewerReference : selected
    const selectedSourceViewEndpoint = selectedHasPdfViewer ? selectedPdfViewerEndpoint : selectedEndpoint
    const selectedSourceViewLabel = selectedSourceViewEndpoint
        ? friendlyReferenceSourceType(selectedSourceViewReference) || 'Source view'
        : ''
    const selectedArtifactState = selectedEndpoint ? artifactCache[selectedEndpoint] : undefined
    const fallbackText = selected?.body || ''
    const documentText = cleanDocumentTextForReading(selectedArtifactState?.text || fallbackText)
    const selectedDocumentPreview = useMemo(() => {
        const preview = selected ? referencePreviewTextForQuery(selected, query) : ''
        if (selected && hasRequestedEvidenceFamilyMarker(selected) && preview) return preview
        if (preview && preview.length >= 24) return preview

        return compactReferenceText(
            documentText
                .split('\n')
                .map(line => line.replace(/^#{1,6}\s*/, '').trim())
                .filter(Boolean)
                .slice(0, 10)
                .join(' '),
            620,
        )
    }, [documentText, query, selected])
    const selectedDocumentOutline = useMemo(
        () =>
            uniqueStrings(
                documentText
                    .split('\n')
                    .map(line => cleanDocumentOutlinePreview(line.replace(/^#{1,6}\s*/, '').trim()))
                    .filter(line =>
                        !isLowSignalDocumentOutlineItem(line) &&
                        /^(?:slide|page|sheet|table|chart|section|title|source|formula|metric)\b/i.test(line),
                    )
                    .slice(0, 10),
            ).slice(0, 6),
        [documentText],
    )
    const loadingDocument = Boolean(selectedEndpoint && !selectedHasPdfViewer && !isPdfReference(selected) && selectedArtifactState?.status === 'loading')
    const documentError = selectedArtifactState?.status === 'error' ? selectedArtifactState.error : ''
    const selectedEvidenceLabels = referenceEvidenceLabels(selected)
    const selectedStructuredLocations = referenceStructuredLocations(selected)
    const selectedVisibleStructuredLocations = selectedStructuredLocations
        .map(location => ({
            location,
            label: compactSourceLocationLabel(location),
        }))
        .filter(item => item.label)
    const selectedStructuredLocationOverflowCount = Math.max(0, selectedVisibleStructuredLocations.length - 10)
    const selectedEvidenceBundleLabel = referenceEvidenceBundleLabel(selected)
    const selectedLinkTarget = selectReferencePrimaryExternalUrl(selected)
    const selectedLocationLabels = referenceLocationLabels(selected)
    const selectedMatchLabels = referenceMatchLabels(selected)
    const selectedPathRelationships = displayPathRelationships(selected?.path?.relationships)
    const selectedPathNodes = selected?.path?.nodes
        ? displayPathNodes(selected.path.nodes)
        : []
    const selectedRelationshipIsWeak = Boolean(
        selected?.relationship && isWeakGraphRelationTitle(selected.relationship.relation),
    )
    const selectedRelationshipParts = selected?.relationship
        ? graphRelationshipEvidenceParts(selected.relationship)
        : null
    const selectedRelationshipLead = selectedRelationshipParts
        ? cleanReferenceEvidencePreview(selectedRelationshipParts.lead)
        : ''
    const selectedConcreteEvidenceStatements = useMemo(
        () => selected?.relationship && !selectedIsRequestedEvidence
            ? relationshipConcreteEvidenceStatements(selected, query)
            : [],
        [query, selected, selectedIsRequestedEvidence],
    )
    const rawSelectedConcreteEvidenceLead = selectedConcreteEvidenceStatements[0] || ''
    const selectedConcreteEvidenceLead = isCoordinateOnlyReferenceTitle(rawSelectedConcreteEvidenceLead)
        ? ''
        : rawSelectedConcreteEvidenceLead
    const selectedRelationshipRelationLabel =
        selectedRelationshipParts && selectedRelationshipIsWeak && selectedConcreteEvidenceLead
            ? 'evidence'
            : selectedRelationshipParts?.relationLabel || ''
    const selectedVisibleRelationshipLead =
        selectedRelationshipParts && selectedRelationshipIsWeak && selectedConcreteEvidenceLead
            ? ''
            : selectedRelationshipParts?.lead || ''
    const selectedDocumentQueryIsGraph = graphRelationshipQuestionQuery(query) || graphPathQuestionQuery(query)
    const selectedDocumentTerms = useMemo(
        () => uniqueStrings([
            ...terms,
            ...(selectedDocumentQueryIsGraph
                ? selectedReviewReferences.flatMap(reference => collectTerms(reference, query).slice(0, 12))
                : []),
        ]).slice(0, 48),
        [query, selectedDocumentQueryIsGraph, selectedReviewReferences, terms],
    )
    const selectedPdfTerms = useMemo(
        () => buildReferencePdfHighlightTerms(selected, query),
        [query, selected],
    )
    const selectedLocationMapNodes = uniqueStrings([
        selectedEvidenceBundleLabel,
        ...selectedStructuredLocations
            .filter(location => ['section', 'slide', 'page', 'table', 'chart', 'form'].includes(String(location.category || '').toLowerCase()))
            .slice(0, 2)
            .map(compactSourceLocationLabel),
    ]).slice(0, 3)
    const selectedRelationshipMapNodes = selected?.relationship && !selectedIsRequestedEvidence
        ? [
            ...graphRelationshipEvidenceMapNodes(selected.relationship),
            ...selectedLocationMapNodes,
        ]
        : []
    const selectedPathMapNodes = selectedPathNodes.length > 1
        ? [
            selectedPathNodes[0],
            selectedPathNodes.length > 2 ? selectedPathNodes[Math.floor(selectedPathNodes.length / 2)] : '',
            selectedPathNodes[selectedPathNodes.length - 1],
            ...selectedLocationMapNodes,
        ]
        : []
    const selectedReferenceMapNodes = uniqueStrings([
        ...(selectedRelationshipMapNodes.length > 0 ? selectedRelationshipMapNodes : selectedPathMapNodes),
        selectedEvidenceBundleLabel && selectedRelationshipMapNodes.length === 0 ? 'Evidence bundle' : '',
        selectedEvidenceBundleLabel && selectedRelationshipMapNodes.length === 0 ? selectedEvidenceBundleLabel : '',
        selected?.filename && selectedRelationshipMapNodes.length === 0 ? referenceBasename(selected.filename) : '',
        selected?.title && !selected?.relationship ? referenceBriefTitle(selected) : '',
    ]).filter(Boolean).slice(0, 5)
    const selectedReferenceDetailCount =
        selectedEvidenceLabels.length +
        selectedLocationLabels.length +
        selectedMatchLabels.length +
        selectedStructuredLocations.length +
        (terms.length > 0 && selectedMatchLabels.length === 0 ? Math.min(terms.length, 10) : 0) +
        (selectedPathNodes.length > 0 ? 1 : 0) +
        (selectedSourceGroupIsNavigable && sourceNavigationGroups.length > 1 ? 1 : 0)
    const selectedEvidenceLead = (() => {
        if (selectedConcreteEvidenceLead) {
            return ''
        }
        if (selectedRelationshipLead) {
            return selectedRelationshipLead
        }
        const preview = selected ? referencePreviewTextForQuery(selected, query) : ''
        if (preview) return preview
        return ''
    })()
    const selectedEvidenceSummaryTitle = selectedConcreteEvidenceLead
        ? selectedConcreteEvidenceLead
        : selectedRelationshipParts
        ? `${selectedRelationshipParts.sourceLabel} ${selectedRelationshipParts.relationLabel} ${selectedRelationshipParts.targetLabel}`
        : selectedPathNodes.length > 1
            ? referencePathTitle(selectedPathNodes)
            : selected ? referenceBriefTitle(selected) : 'Selected evidence'
    const selectedEvidenceContext = uniqueStrings([
        selected?.filename ? referenceBasename(selected.filename) : '',
        selectedLocationLabels.find(label => /\b(?:slide|page|sheet|table|chart)\b/i.test(label)) || '',
        selectedEvidenceLabels.slice(0, 2).join(' · '),
    ]).join(' · ')
    const selectedPreviewIsRedundant = isRedundantPreview(
        selectedDocumentPreview,
        [selectedEvidenceSummaryTitle, selectedEvidenceLead, selectedRelationshipLead].filter(Boolean).join(' '),
    )
    const visibleSelectedDocumentPreview = selectedPreviewIsRedundant ? '' : selectedDocumentPreview
    const selectedDocumentPreviewSummary = selected
        ? buildReferenceDocumentPreviewSummary(selected, {
            documentText,
            fallbackPreview: visibleSelectedDocumentPreview,
            outline: selectedDocumentOutline,
            evidenceLabels: selectedEvidenceLabels,
            locationLabels: selectedLocationLabels,
            sourceTypeLabel: selectedSourceViewLabel,
            query,
        })
        : null
    const selectedDocumentPreviewOutline = selectedDocumentPreviewSummary
        ? selectedDocumentPreviewSummary.outline
        : selectedDocumentOutline
    const selectedDocumentPreviewText =
        selected?.kind === 'passage' && !selectedDocumentQueryIsGraph
            ? selectedEvidenceLead || selectedDocumentPreviewSummary?.excerpt || visibleSelectedDocumentPreview
            : selectedDocumentPreviewSummary?.excerpt || visibleSelectedDocumentPreview
    const dedupedSelectedDocumentPreviewText = dedupeReferencePreviewAgainstSelectedEvidence(
        selectedDocumentPreviewText,
        [selectedEvidenceSummaryTitle, selectedEvidenceLead].filter(Boolean).join(' '),
    )
    const selectedLayoutMap = selected ? buildReferenceLayoutMap(selected) : null
    const selectedDocumentPreviewTextForDisplay =
        dedupedSelectedDocumentPreviewText ||
        (selectedIsRequestedEvidence && !selectedLayoutMap
            ? selectedDocumentPreviewSummary?.excerpt || selectedEvidenceLead
            : '')
    const showSelectedDocumentPreviewCard = Boolean(
        selectedLayoutMap ||
        loadingDocument ||
        documentError ||
        selectedDocumentPreviewTextForDisplay ||
        selectedDocumentPreviewOutline.length > 0,
    )
    const activeReference = selected || reviewReferences[0] || references[0]
    const reviewReference = preferredReviewReference(reviewReferences) || activeReference
    const activeAccentColor = referenceAccentColor(activeReference, theme)
    const selectedSourceBrief = buildChatReferenceSourceBrief(
        selectedSourceGroup?.references || (activeReference ? [activeReference] : []),
    )
    const trayReferences = selectReferenceExplorerTrayReferences({
        orderedReferences: reviewReferences,
        visibleReferences,
        answerEvidenceReferences,
        query,
    })
    const inlineTrayReferences = selectInlineReferenceExplorerTrayReferences(
        trayReferences,
        query,
        INLINE_TRAY_REFERENCE_LIMIT,
    )
    const requestedEvidenceTrayQuestion = requestedEvidenceFamilyQuestionQuery(query)
    const requestedEvidenceTrayInspection = requestedEvidenceFamilyInspectionQuery(query)
    const hiddenInlineTrayReferenceCount = Math.max(
        0,
        trayReferences.length - inlineTrayReferences.length,
    )
    const drawerOpeningReference = (() => {
        return selectReferenceExplorerOpeningReference(reviewReferences, query, {
            trayReferences,
            answerEvidenceReferences,
            fallback: reviewReference,
        }) || reviewReference
    })()
    const openingReferenceId = drawerOpeningReference?.id || reviewReferences[0]?.id || ''

    useEffect(() => {
        if (!reviewReferences.length) return
        if (selectedOverrideReference?.id === selectedId) return
        if (!selectedId || !reviewReferences.some(reference => reference.id === selectedId)) {
            setSelectedOverrideReference(null)
            setSelectedId(openingReferenceId)
        }
    }, [openingReferenceId, reviewReferences, selectedId, selectedOverrideReference])

    useEffect(() => {
        if (!filteredReferences.length) return
        if (selectedOverrideReference?.id === selectedId) return
        if (!filteredReferences.some(reference => reference.id === selectedId)) {
            setSelectedOverrideReference(null)
            setSelectedId(
                filteredReferences.find(reference => reference.id === openingReferenceId)?.id ||
                filteredReferences[0].id,
            )
        }
    }, [filteredReferences, openingReferenceId, selectedId, selectedOverrideReference])

    useEffect(() => {
        if (documentFilter === 'all') return
        if (!sourceNavigationGroups.some(group => group.key === documentFilter)) {
            setDocumentFilter('all')
        }
    }, [documentFilter, sourceNavigationGroups])

    useEffect(() => {
        setSourceTextExpanded(false)
    }, [open, selected?.id])

    useEffect(() => {
        setShowAllDrawerReferences(false)
    }, [documentFilter, drawerSearch, kindFilter, open])

    useEffect(() => {
        if (!open || !selectedEndpoint || selectedIsPdf || requestedEndpoints.current.has(selectedEndpoint)) return

        requestedEndpoints.current.add(selectedEndpoint)
        setArtifactCache(previous => ({
            ...previous,
            [selectedEndpoint]: { status: 'loading' },
        }))

        let active = true
        fetchReferenceDocument(selectedEndpoint)
            .then(result => {
                if (!active) return
                setArtifactCache(previous => ({
                    ...previous,
                    [selectedEndpoint]: {
                        status: 'loaded',
                        text: result.text,
                        contentType: result.contentType,
                    },
                }))
            })
            .catch(error => {
                if (!active) return
                setArtifactCache(previous => ({
                    ...previous,
                    [selectedEndpoint]: {
                        status: 'error',
                        error: error instanceof Error ? error.message : 'Unable to load document',
                    },
                }))
            })

        return () => {
            active = false
        }
    }, [open, selectedEndpoint, selectedIsPdf])

    if (!references.length) return null

    const referenceSummary = [
        sourceNavigationGroups.length ? `${sourceNavigationGroups.length} ${sourceNavigationGroups.length === 1 ? 'source' : 'sources'}` : '',
        formatOptionalCount(answerEvidenceReferences.length, 'citation'),
        formatOptionalCount(referenceCounts.path, 'path'),
    ].filter(Boolean).join(' · ') || `${reviewReferences.length} references`
    const requestedReferenceFilenames = queryFilenames(query)
    const trayOpeningReference =
        requestedEvidenceTrayQuestion
            ? trayReferences[0] || drawerOpeningReference
            : graphPathQuestionQuery(query)
            ? trayReferences.find(reference => reference.kind === 'path') || drawerOpeningReference
            : sourceNavigationGroups[0]?.bestReference || trayReferences[0] || drawerOpeningReference

    const selectReference = (reference: ChatReferenceItem) => {
        const resolved = resolveReferenceExplorerSelectionReference(reviewReferences, reference, query)
        const selectedReference = reference.kind === 'path' ? reference : resolved
        setSelectedOverrideReference(selectedReference)
        setSelectedId(selectedReference.id)
    }

    const openReference = (reference: ChatReferenceItem) => {
        selectReference(reference)
        setOpen(true)
    }

    const selectSourceGroup = (group: ReferenceDocumentGroup) => {
        setDocumentFilter(group.key)
        selectReference(preferredReviewReference(group.references) || group.bestReference)
    }

    const clearSourceGroup = () => {
        setDocumentFilter('all')
    }

    const copyDocument = async () => {
        if (!documentText) return
        await navigator.clipboard.writeText(documentText)
        setCopied(true)
        window.setTimeout(() => setCopied(false), 1200)
    }

    const openRawDocument = () => {
        if (!selectedSourceViewEndpoint) return
        window.open(withDownloadFlag(selectedSourceViewEndpoint), '_blank', 'noopener,noreferrer')
    }

    return (
        <>
            {density === 'tray' ? (
                <Paper
                    variant="outlined"
                    data-testid="reference-tray"
                    data-requested-evidence-question={requestedEvidenceTrayQuestion ? 'true' : 'false'}
                    data-requested-evidence-inspection={requestedEvidenceTrayInspection ? 'true' : 'false'}
                    sx={{
                        overflow: 'hidden',
                        borderColor: alpha(theme.palette.info.main, 0.18),
                        bgcolor: alpha(theme.palette.background.paper, 0.78),
                        borderRadius: '8px',
                        display: 'flex',
                        flexDirection: 'column',
                        maxHeight: { xs: 340, sm: 440 },
                        minHeight: 0,
                    }}
                >
                    <Box
                        sx={{
                            px: 0.9,
                            py: 0.55,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            gap: 0.75,
                            minWidth: 0,
                            borderBottom: '1px solid',
                            borderColor: alpha(theme.palette.divider, 0.62),
                            flexWrap: { xs: 'nowrap', sm: 'wrap' },
                        }}
                    >
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.65, minWidth: 0, flex: 1, overflow: 'hidden', flexWrap: { xs: 'nowrap', sm: 'wrap' } }}>
                            <SearchIcon sx={{ fontSize: 17, color: 'info.main', flexShrink: 0 }} />
                            <Typography variant="caption" noWrap sx={{ fontWeight: 900, flexShrink: 0 }}>
                                References
                            </Typography>
                            <Typography variant="caption" color="text.secondary" noWrap sx={{ minWidth: 0, display: { xs: 'none', sm: 'block' } }}>
                                {referenceSummary}
                            </Typography>
                        </Box>
                        <Button
                            size="small"
                            data-testid="reference-tray-review"
                            aria-label="Review evidence"
                            startIcon={<OpenIcon />}
                            onClick={() => openReference(trayOpeningReference)}
                            sx={{
                                minHeight: 28,
                                borderRadius: '8px',
                                justifySelf: { xs: 'flex-end', sm: 'flex-end' },
                                flexShrink: 0,
                                px: { xs: 0.8, sm: 1 },
                                textTransform: 'none',
                                whiteSpace: 'nowrap',
                                '& .MuiButton-startIcon': {
                                    mr: { xs: 0.45, sm: 0.65 },
                                },
                            }}
                        >
                            <Box component="span" sx={{ display: { xs: 'none', sm: 'inline' } }}>
                                Review evidence
                            </Box>
                            <Box component="span" sx={{ display: { xs: 'inline', sm: 'none' } }}>
                                Review
                            </Box>
                        </Button>
                    </Box>
                    {inlineTrayReferences.length > 0 && (
                        <Box
                            sx={{
                                display: 'grid',
                                gridTemplateColumns: {
                                xs: '1fr',
                                sm: inlineTrayReferences.length === 1
                                    ? '1fr'
                                    : 'repeat(2, minmax(0, 1fr))',
                                },
                                gap: 0.65,
                                p: 0.65,
                                minWidth: 0,
                                maxWidth: '100%',
                                minHeight: 0,
                                overflowY: 'auto',
                                overflowX: 'hidden',
                                overscrollBehavior: 'contain',
                            }}
                        >
                            {inlineTrayReferences.map((reference) => {
                                const accentColor = referenceAccentColor(reference, theme)
                                const locationLabels = referenceLocationLabels(reference).slice(0, 2)
                                const evidenceLabels = referenceEvidenceLabels(reference).slice(0, 2)
                                const pathNodes = displayPathNodes(reference.path?.nodes).slice(0, 3)
                                const requestedFilename = matchingQueryFilename(reference, requestedReferenceFilenames)
                                const requestedFilenameTitle = requestedFilename
                                    ? referenceBasename(reference.filename || requestedFilename)
                                    : ''
                                const baseTrayTitle = reference.kind === 'path' && pathNodes.length > 1
                                    ? `${reference.path?.hops || Math.max(pathNodes.length - 1, 1)}-hop graph path`
                                    : referenceBriefTitle(reference)
                                const filenameLedTrayTitle =
                                    requestedFilenameTitle &&
                                    reference.kind !== 'path' &&
                                    !hasRequestedEvidenceFamilyMarker(reference)
                                        ? requestedFilenameTitle
                                        : ''
                                const trayTitle = filenameLedTrayTitle || baseTrayTitle
                                const trayInsight = reference.kind === 'path'
                                    ? buildReferencePathPreviewText(reference)
                                    : referenceTrayPreviewText(reference, query)
                                const traySubtitle = filenameLedTrayTitle
                                    ? uniqueStrings([
                                        buildReferenceTraySubtitle(reference),
                                        baseTrayTitle !== filenameLedTrayTitle ? baseTrayTitle : '',
                                    ]).join(' · ')
                                    : buildReferenceTraySubtitle(reference)

                                return (
                                    <Box
                                        key={`tray-reference-${reference.id}`}
                                        component="button"
                                        type="button"
                                        data-testid="reference-tray-card"
                                        data-reference-title={referenceBriefTitle(reference)}
                                        onClick={() => openReference(reference)}
                                        sx={{
                                            width: '100%',
                                            maxWidth: '100%',
                                            minWidth: 0,
                                            minHeight: 96,
                                            p: 0.75,
                                            border: '1px solid',
                                            borderColor: alpha(accentColor, 0.22),
                                            borderRadius: '8px',
                                            bgcolor: alpha(theme.palette.background.default, 0.5),
                                            color: 'text.primary',
                                            cursor: 'pointer',
                                            font: 'inherit',
                                            textAlign: 'left',
                                            display: 'flex',
                                            flexDirection: 'column',
                                            gap: 0.5,
                                            overflow: 'hidden',
                                            transition: 'background-color 120ms ease, border-color 120ms ease',
                                            '&:hover': {
                                                borderColor: accentColor,
                                                bgcolor: alpha(accentColor, 0.06),
                                            },
                                        }}
                                    >
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.55, minWidth: 0 }}>
                                            <Box sx={{ color: accentColor, display: 'flex', flexShrink: 0 }}>
                                                {referenceIcon(reference.kind, isPdfReference(reference))}
                                            </Box>
                                            <Typography variant="caption" noWrap title={referenceBriefTitle(reference)} sx={{ flex: 1, minWidth: 0, fontWeight: 900 }}>
                                                {trayTitle}
                                            </Typography>
                                            <Chip
                                                size="small"
                                                label={referenceKindLabel(reference)}
                                                sx={{ height: 18, fontSize: '0.62rem', flexShrink: 0 }}
                                            />
                                        </Box>
                                        <Typography
                                            variant="caption"
                                            color="text.secondary"
                                            noWrap
                                            sx={{
                                                display: 'block',
                                                width: '100%',
                                                minWidth: 0,
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                fontSize: '0.68rem',
                                            }}
                                        >
                                            {traySubtitle}
                                        </Typography>
                                        {trayInsight && (
                                            <Typography
                                                variant="caption"
                                                sx={{
                                                    color: 'text.secondary',
                                                    display: '-webkit-box',
                                                    minWidth: 0,
                                                    overflow: 'hidden',
                                                    WebkitBoxOrient: 'vertical',
                                                    WebkitLineClamp: 3,
                                                    fontSize: '0.7rem',
                                                    lineHeight: 1.35,
                                                    wordBreak: 'break-word',
                                                }}
                                            >
                                                {trayInsight}
                                            </Typography>
                                        )}
                                        {reference.kind !== 'path' && pathNodes.length > 1 && (
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.35, minWidth: 0, maxWidth: '100%', overflow: 'hidden' }}>
                                                {pathNodes.map((node, index) => (
                                                    <React.Fragment key={`${reference.id}-tray-node-${index}-${node}`}>
                                                        {index > 0 && (
                                                            <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0 }}>
                                                                to
                                                            </Typography>
                                                        )}
                                                        <Chip
                                                            size="small"
                                                            label={node}
                                                            sx={{
                                                                height: 18,
                                                                minWidth: 0,
                                                                maxWidth: 96,
                                                                flexShrink: 1,
                                                                fontSize: '0.62rem',
                                                                bgcolor: alpha(accentColor, 0.1),
                                                                '& .MuiChip-label': {
                                                                    minWidth: 0,
                                                                    overflow: 'hidden',
                                                                    textOverflow: 'ellipsis',
                                                                },
                                                            }}
                                                        />
                                                    </React.Fragment>
                                                ))}
                                            </Box>
                                        )}
                                        {(locationLabels.length > 0 || evidenceLabels.length > 0) && (
                                            <Box sx={{ display: 'flex', gap: 0.35, flexWrap: 'wrap', mt: 'auto', minHeight: 18 }}>
                                                {locationLabels.map(label => (
                                                    <Tooltip key={`${reference.id}-location-${label}`} title={label}>
                                                        <Chip
                                                            size="small"
                                                            label={label}
                                                            sx={{
                                                                height: 18,
                                                                maxWidth: 150,
                                                                fontSize: '0.62rem',
                                                                bgcolor: alpha(theme.palette.warning.main, 0.1),
                                                                '& .MuiChip-label': {
                                                                    minWidth: 0,
                                                                    overflow: 'hidden',
                                                                    textOverflow: 'ellipsis',
                                                                },
                                                            }}
                                                        />
                                                    </Tooltip>
                                                ))}
                                                {locationLabels.length === 0 && evidenceLabels.map(label => (
                                                    <Chip
                                                        key={`${reference.id}-evidence-${label}`}
                                                        size="small"
                                                        label={label}
                                                        sx={{ height: 18, maxWidth: 150, fontSize: '0.62rem', bgcolor: alpha(theme.palette.info.main, 0.1) }}
                                                    />
                                                ))}
                                            </Box>
                                        )}
                                    </Box>
                                )
                            })}
                            {hiddenInlineTrayReferenceCount > 0 && (
                                <Button
                                    size="small"
                                    variant="outlined"
                                    onClick={() => openReference(trayOpeningReference)}
                                    sx={{
                                        display: 'inline-flex',
                                        gridColumn: '1 / -1',
                                        minHeight: 30,
                                        borderRadius: '8px',
                                        textTransform: 'none',
                                        fontSize: '0.72rem',
                                    }}
                                >
                                    {requestedEvidenceTrayInspection
                                        ? 'Review all evidence'
                                        : `Review ${hiddenInlineTrayReferenceCount} more reference${hiddenInlineTrayReferenceCount === 1 ? '' : 's'}`}
                                </Button>
                            )}
                        </Box>
                    )}
                </Paper>
            ) : (
                <Paper
                    variant="outlined"
                    sx={{
                        overflow: 'hidden',
                        borderColor: alpha(theme.palette.info.main, 0.2),
                        bgcolor: alpha(theme.palette.background.paper, 0.72),
                    }}
                >
                    <Box
                        sx={{
                            px: 1,
                            py: 0.8,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            gap: 1,
                            borderBottom: '1px solid',
                            borderColor: alpha(theme.palette.divider, 0.7),
                        }}
                    >
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, minWidth: 0 }}>
                            <SearchIcon sx={{ fontSize: 17, color: 'info.main' }} />
                            <Typography variant="subtitle2" sx={{ fontSize: '0.84rem', fontWeight: 750 }}>
                                Evidence
                            </Typography>
                            <Chip
                                size="small"
                                label={formatReferenceCount(reviewReferences.length, 'source')}
                                sx={{ height: 19, fontSize: '0.68rem' }}
                            />
                        </Box>
                        <Button
                            size="small"
                            startIcon={<OpenIcon />}
                            onClick={() => openReference(reviewReference)}
                            sx={{ minHeight: 28, textTransform: 'none' }}
                        >
                            Review
                        </Button>
                    </Box>

                    <Box
                        sx={{
                            display: 'grid',
                            gridTemplateColumns: {
                                xs: 'repeat(2, minmax(0, 1fr))',
                                sm: 'repeat(4, minmax(0, 1fr))',
                            },
                            borderBottom: '1px solid',
                            borderColor: alpha(theme.palette.divider, 0.7),
                        }}
                    >
                        {[
                            { label: 'Passages', value: referenceCounts.passage || 0, color: theme.palette.info.main },
                            { label: 'Relations', value: referenceCounts.relationship || 0, color: theme.palette.success.main },
                            { label: 'Paths', value: referenceCounts.path || 0, color: theme.palette.secondary.main },
                            { label: 'Documents', value: (referenceCounts.document || 0) + (referenceCounts.artifact || 0), color: theme.palette.warning.main },
                        ].map(item => (
                            <Box
                                key={item.label}
                                sx={{
                                    px: 1,
                                    py: 0.8,
                                    borderRight: { sm: '1px solid' },
                                    borderBottom: { xs: '1px solid', sm: 0 },
                                    borderColor: alpha(theme.palette.divider, 0.55),
                                    '&:last-of-type': { borderRight: 0 },
                                }}
                            >
                                <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block', fontSize: '0.66rem' }}>
                                    {item.label}
                                </Typography>
                                <Typography sx={{ fontSize: '1rem', fontWeight: 850, lineHeight: 1.15, color: item.color }}>
                                    {item.value}
                                </Typography>
                            </Box>
                        ))}
                    </Box>

                    {topEvidenceFamilies.length > 0 && (
                        <Box
                            sx={{
                                px: 0.75,
                                py: 0.6,
                                display: 'flex',
                                gap: 0.45,
                                flexWrap: 'wrap',
                                borderBottom: '1px solid',
                                borderColor: alpha(theme.palette.divider, 0.7),
                            }}
                        >
                            {topEvidenceFamilies.map(([label, count]) => (
                                <Chip
                                    key={label}
                                    size="small"
                                    label={`${label}${count > 1 ? ` ${count}` : ''}`}
                                    sx={{ height: 20, fontSize: '0.66rem' }}
                                />
                            ))}
                        </Box>
                    )}

                    <Box
                        sx={{
                            display: 'grid',
                            gridTemplateColumns: {
                                xs: '1fr',
                                md: 'repeat(2, minmax(0, 1fr))',
                            },
                            gap: 0.6,
                            p: 0.75,
                        }}
                    >
                        {visibleReferences.map(reference => {
                            const pdf = isPdfReference(reference)
                            const accentColor = referenceAccentColor(reference, theme)
                            const evidenceLabels = referenceEvidenceLabels(reference)
                            const locationLabels = referenceLocationLabels(reference)
                            const pathNodes = displayPathNodes(reference.path?.nodes)
                            const referenceSubtitle = reference.subtitle ? buildReferenceTraySubtitle(reference) : ''
                            return (
                                <Box
                                    key={reference.id}
                                    component="button"
                                    type="button"
                                    onClick={() => openReference(reference)}
                                    sx={{
                                        p: 0.75,
                                        minHeight: 82,
                                        border: '1px solid',
                                        borderColor: alpha(accentColor, 0.24),
                                        borderRadius: '8px',
                                        bgcolor: alpha(theme.palette.background.default, 0.58),
                                        color: 'text.primary',
                                        cursor: 'pointer',
                                        font: 'inherit',
                                        textAlign: 'left',
                                        display: 'flex',
                                        flexDirection: 'column',
                                        gap: 0.42,
                                        minWidth: 0,
                                        maxWidth: '100%',
                                        overflow: 'hidden',
                                        transition: 'border-color 120ms ease, background-color 120ms ease',
                                        '&:hover': {
                                            borderColor: accentColor,
                                            bgcolor: alpha(accentColor, 0.06),
                                        },
                                    }}
                                >
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.6, minWidth: 0 }}>
                                        <Box sx={{ color: pdf ? 'error.main' : accentColor, display: 'flex' }}>
                                            {referenceIcon(reference.kind, pdf)}
                                        </Box>
                                        <Typography variant="caption" noWrap sx={{ flex: 1, minWidth: 0, fontWeight: 750 }}>
                                            {referenceBriefTitle(reference)}
                                        </Typography>
                                        <Chip
                                            size="small"
                                            label={referenceKindLabel(reference)}
                                            sx={{ height: 18, fontSize: '0.62rem' }}
                                        />
                                    </Box>
                                    {reference.relationship && (
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.45, minWidth: 0 }}>
                                            <Chip size="small" label={cleanGraphLabel(reference.relationship.source, 'Source')} sx={{ height: 18, fontSize: '0.62rem', maxWidth: 92 }} />
                                            <Typography variant="caption" color="text.secondary" noWrap>
                                                {formatGraphReferenceRelation(reference.relationship.relation)}
                                            </Typography>
                                            <Chip size="small" label={cleanGraphLabel(reference.relationship.target, 'Target')} sx={{ height: 18, fontSize: '0.62rem', maxWidth: 92 }} />
                                        </Box>
                                    )}
                                    {pathNodes.length > 0 && (
                                        <Box
                                            sx={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: 0.35,
                                                minWidth: 0,
                                                overflow: 'hidden',
                                            }}
                                        >
                                            {pathNodes.slice(0, 4).map((node, nodeIndex) => (
                                                <React.Fragment key={`${reference.id}-node-${nodeIndex}-${node}`}>
                                                    {nodeIndex > 0 && (
                                                        <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0 }}>
                                                            to
                                                        </Typography>
                                                    )}
                                                    <Chip
                                                        size="small"
                                                        label={node}
                                                        sx={{
                                                            height: 18,
                                                            fontSize: '0.62rem',
                                                            maxWidth: nodeIndex === 0 ? 110 : 92,
                                                            bgcolor: alpha(accentColor, 0.1),
                                                            '& .MuiChip-label': {
                                                                minWidth: 0,
                                                                overflow: 'hidden',
                                                                textOverflow: 'ellipsis',
                                                            },
                                                        }}
                                                    />
                                                </React.Fragment>
                                            ))}
                                            {pathNodes.length > 4 && (
                                                <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0 }}>
                                                    +{pathNodes.length - 4}
                                                </Typography>
                                            )}
                                        </Box>
                                    )}
                                    {referenceSubtitle && (
                                        <Typography
                                            variant="caption"
                                            color="text.secondary"
                                            noWrap
                                            sx={{
                                                display: 'block',
                                                minWidth: 0,
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                            }}
                                        >
                                            {referenceSubtitle}
                                        </Typography>
                                    )}
                                    {(evidenceLabels.length > 0 || locationLabels.length > 0) && (
                                        <Box sx={{ display: 'flex', gap: 0.35, flexWrap: 'wrap', minHeight: 20 }}>
                                            {evidenceLabels.slice(0, 2).map(label => (
                                                <Chip
                                                    key={`${reference.id}-evidence-${label}`}
                                                    size="small"
                                                    label={label}
                                                    sx={{ height: 18, fontSize: '0.62rem', bgcolor: alpha(theme.palette.info.main, 0.1) }}
                                                />
                                            ))}
                                            {locationLabels.slice(0, 1).map(label => (
                                                <Tooltip key={`${reference.id}-location-${label}`} title={label}>
                                                    <Chip
                                                        size="small"
                                                        label={label}
                                                        sx={{
                                                            height: 18,
                                                            fontSize: '0.62rem',
                                                            maxWidth: 160,
                                                            bgcolor: alpha(theme.palette.warning.main, 0.1),
                                                            '& .MuiChip-label': {
                                                                minWidth: 0,
                                                                overflow: 'hidden',
                                                                textOverflow: 'ellipsis',
                                                            },
                                                        }}
                                                    />
                                                </Tooltip>
                                            ))}
                                        </Box>
                                    )}
                                    <Typography
                                        variant="body2"
                                        sx={{
                                            color: 'text.secondary',
                                            fontSize: '0.73rem',
                                            lineHeight: 1.38,
                                            display: '-webkit-box',
                                            WebkitLineClamp: 2,
                                            WebkitBoxOrient: 'vertical',
                                            overflow: 'hidden',
                                            wordBreak: 'break-word',
                                        }}
                                    >
                                        <HighlightedText text={referencePreviewTextForQuery(reference, query)} terms={collectTerms(reference, query)} />
                                    </Typography>
                                </Box>
                            )
                        })}
                    </Box>
                </Paper>
            )}

            <Drawer
                anchor="right"
                open={open}
                onClose={() => setOpen(false)}
                sx={{ zIndex: theme.zIndex.modal + 2 }}
                PaperProps={{
                    sx: {
                        width: { xs: '100vw', md: 'min(940px, 82vw)' },
                        maxWidth: '100vw',
                        zIndex: theme.zIndex.modal + 3,
                        bgcolor: 'background.default',
                    },
                }}
            >
                <Box sx={{ height: '100%', display: 'flex', minWidth: 0 }}>
                    <Box
                        sx={{
                            width: { xs: 0, md: 250 },
                            display: { xs: 'none', md: 'flex' },
                            flexDirection: 'column',
                            borderRight: '1px solid',
                            borderColor: 'divider',
                            minHeight: 0,
                        }}
                    >
                        <Box sx={{ px: 1.25, py: 1, borderBottom: '1px solid', borderColor: 'divider' }}>
                            <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>
                                References
                            </Typography>
                            {showDrawerSearch && (
                                <TextField
                                    value={drawerSearch}
                                    onChange={event => setDrawerSearch(event.target.value)}
                                    placeholder="Find source"
                                    size="small"
                                    fullWidth
                                    sx={{
                                        mt: 0.8,
                                        '& .MuiOutlinedInput-root': {
                                            height: 32,
                                            fontSize: '0.76rem',
                                            borderRadius: '8px',
                                        },
                                    }}
                                />
                            )}
                            <Box
                                sx={{
                                    display: kindFilter !== 'all' || drawerSearch.trim() ? 'flex' : 'none',
                                    gap: 0.4,
                                    flexWrap: 'wrap',
                                    mt: 0.75,
                                }}
                            >
                                {referenceFilterChips.slice(0, 4).map(filter => (
                                    <Chip
                                        key={filter.value}
                                        label={`${filter.label} ${filter.count}`}
                                        size="small"
                                        clickable
                                        color={kindFilter === filter.value ? 'info' : 'default'}
                                        variant={kindFilter === filter.value ? 'filled' : 'outlined'}
                                        onClick={() => setKindFilter(filter.value)}
                                        sx={{
                                            height: 22,
                                            fontSize: '0.66rem',
                                            opacity: filter.tone === 'trace' && kindFilter !== filter.value ? 0.78 : 1,
                                        }}
                                    />
                                ))}
                            </Box>
                        </Box>
                        <Stack spacing={0.5} sx={{ p: 0.75, overflow: 'auto', minHeight: 0 }}>
                            {sourceNavigationGroups.length > 1 && (
                                <Box sx={{ pb: 0.75, mb: 0.25, borderBottom: '1px solid', borderColor: alpha(theme.palette.divider, 0.62) }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 0.75, px: 0.25, mb: 0.45 }}>
                                        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 800 }}>
                                            {sourceNavigationGroups.length > 1 ? 'Documents' : 'Source'}
                                        </Typography>
                                        {sourceNavigationGroups.length > 1 && documentFilter !== 'all' && (
                                            <Button
                                                size="small"
                                                onClick={clearSourceGroup}
                                                sx={{ minHeight: 22, px: 0.6, fontSize: '0.66rem', textTransform: 'none' }}
                                            >
                                                Show all
                                            </Button>
                                        )}
                                    </Box>
                                    <Stack spacing={0.45}>
                                        {sourceNavigationGroups.map(group => {
                                            const selectedGroup = sourceNavigationGroups.length === 1 || documentFilter === group.key
                                            const groupAccent = referenceAccentColor(group.bestReference, theme)
                                            return (
                                                <Box
                                                    key={`source-group-${group.key}`}
                                                    component="button"
                                                    type="button"
                                                    onClick={() => selectSourceGroup(group)}
                                                    sx={{
                                                        width: '100%',
                                                        p: 0.7,
                                                        border: '1px solid',
                                                        borderColor: selectedGroup ? groupAccent : alpha(groupAccent, 0.18),
                                                        borderRadius: '8px',
                                                        bgcolor: selectedGroup ? alpha(groupAccent, 0.08) : alpha(theme.palette.background.paper, 0.5),
                                                        color: 'text.primary',
                                                        cursor: 'pointer',
                                                        font: 'inherit',
                                                        textAlign: 'left',
                                                        maxWidth: '100%',
                                                        overflow: 'hidden',
                                                        '&:hover': {
                                                            borderColor: groupAccent,
                                                            bgcolor: alpha(groupAccent, 0.06),
                                                        },
                                                    }}
                                                >
                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.55, minWidth: 0 }}>
                                                        <Box sx={{ color: groupAccent, display: 'flex', flexShrink: 0 }}>
                                                            {referenceIcon(group.bestReference.kind, isPdfReference(group.bestReference))}
                                                        </Box>
                                                        <Typography variant="caption" noWrap sx={{ flex: 1, minWidth: 0, fontWeight: 800 }}>
                                                            {group.title}
                                                        </Typography>
                                                        <Chip size="small" label={group.references.length} sx={{ height: 18, fontSize: '0.62rem' }} />
                                                    </Box>
                                                    <Typography
                                                        variant="caption"
                                                        color="text.secondary"
                                                        noWrap
                                                        sx={{
                                                            display: 'block',
                                                            minWidth: 0,
                                                            overflow: 'hidden',
                                                            textOverflow: 'ellipsis',
                                                            mt: 0.2,
                                                        }}
                                                    >
                                                        {referenceGroupSummary(group) || group.subtitle || 'Evidence'}
                                                    </Typography>
                                                    {(group.evidenceLabels.length > 0 || group.locationLabels.length > 0) && (
                                                        <Box sx={{ display: 'flex', gap: 0.3, flexWrap: 'wrap', mt: 0.45 }}>
                                                            {group.locationLabels.slice(0, 1).map(label => (
                                                                <Chip
                                                                    key={`source-group-location-${group.key}-${label}`}
                                                                    size="small"
                                                                    label={label}
                                                                    variant="outlined"
                                                                    sx={{
                                                                        height: 18,
                                                                        maxWidth: 130,
                                                                        fontSize: '0.58rem',
                                                                        borderColor: alpha(theme.palette.warning.main, 0.28),
                                                                        bgcolor: alpha(theme.palette.warning.main, 0.055),
                                                                        '& .MuiChip-label': {
                                                                            px: 0.58,
                                                                            minWidth: 0,
                                                                            overflow: 'hidden',
                                                                            textOverflow: 'ellipsis',
                                                                        },
                                                                    }}
                                                                />
                                                            ))}
                                                            {group.evidenceLabels.slice(0, group.locationLabels.length > 0 ? 1 : 2).map(label => (
                                                                <Chip
                                                                    key={`source-group-evidence-${group.key}-${label}`}
                                                                    size="small"
                                                                    label={label}
                                                                    variant="outlined"
                                                                    sx={{
                                                                        height: 18,
                                                                        maxWidth: 130,
                                                                        fontSize: '0.58rem',
                                                                        borderColor: alpha(theme.palette.info.main, 0.28),
                                                                        bgcolor: alpha(theme.palette.info.main, 0.055),
                                                                        '& .MuiChip-label': {
                                                                            px: 0.58,
                                                                            minWidth: 0,
                                                                            overflow: 'hidden',
                                                                            textOverflow: 'ellipsis',
                                                                        },
                                                                    }}
                                                                />
                                                            ))}
                                                        </Box>
                                                    )}
                                                </Box>
                                            )
                                        })}
                                    </Stack>
                                </Box>
                            )}
                            {drawerReferenceRows.map(reference => {
                                const referenceSelected = selected?.id === reference.id
                                const referenceAccent = referenceAccentColor(reference, theme)
                                const locationLabels = referenceLocationLabels(reference).slice(0, 2)
                                const evidenceLabels = referenceEvidenceLabels(reference).slice(0, locationLabels.length > 0 ? 1 : 2)

                                return (
                                    <Box
                                        key={`drawer-${reference.id}`}
                                        component="button"
                                        type="button"
                                        data-testid={isPdfReference(reference) ? 'reference-drawer-row-pdf' : 'reference-drawer-row'}
                                        data-reference-title={referenceBriefTitle(reference)}
                                        onClick={() => selectReference(reference)}
                                        sx={{
                                            p: 0.8,
                                            border: '1px solid',
                                            borderColor: referenceSelected ? referenceAccent : alpha(referenceAccent, 0.12),
                                            borderRadius: '8px',
                                            bgcolor: referenceSelected ? alpha(referenceAccent, 0.09) : 'transparent',
                                            color: 'text.primary',
                                            cursor: 'pointer',
                                            font: 'inherit',
                                            textAlign: 'left',
                                            '&:hover': { bgcolor: alpha(referenceAccent, 0.065), borderColor: alpha(referenceAccent, 0.34) },
                                            '&:focus-visible': {
                                                outline: 'none',
                                                borderColor: referenceAccent,
                                                boxShadow: `0 0 0 2px ${alpha(referenceAccent, 0.16)}`,
                                            },
                                        }}
                                    >
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 0 }}>
                                            <Box sx={{ color: referenceAccent, display: 'flex', flexShrink: 0 }}>
                                                {referenceIcon(reference.kind, isPdfReference(reference))}
                                            </Box>
                                            <Typography variant="caption" noWrap sx={{ flex: 1, minWidth: 0, fontWeight: 850 }}>
                                                {referenceBriefTitle(reference)}
                                            </Typography>
                                            {reference.sourceRole === 'answer_evidence' && (
                                                <Chip
                                                    size="small"
                                                    label="cited"
                                                    sx={{
                                                        height: 17,
                                                        fontSize: '0.58rem',
                                                        color: referenceAccent,
                                                        bgcolor: alpha(referenceAccent, 0.1),
                                                        flexShrink: 0,
                                                        '& .MuiChip-label': { px: 0.55 },
                                                    }}
                                                />
                                            )}
                                        </Box>
                                        <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block', mt: 0.16 }}>
                                            {[
                                                referenceKindLabel(reference),
                                                reference.chunkIndex !== undefined ? `chunk ${reference.chunkIndex}` : '',
                                                sourceNavigationGroups.length > 1 ? compactReferenceText(referenceDocumentTitle(reference), 34) : '',
                                            ].filter(Boolean).join(' · ')}
                                        </Typography>
                                        {(locationLabels.length > 0 || evidenceLabels.length > 0) && (
                                            <Box sx={{ display: 'flex', gap: 0.32, flexWrap: 'wrap', mt: 0.5 }}>
                                                {locationLabels.map(label => (
                                                    <Tooltip key={`drawer-location-${reference.id}-${label}`} title={label}>
                                                        <Chip
                                                            size="small"
                                                            label={label}
                                                            variant="outlined"
                                                            sx={{
                                                                height: 18,
                                                                maxWidth: 118,
                                                                fontSize: '0.58rem',
                                                                borderColor: alpha(theme.palette.warning.main, 0.28),
                                                                bgcolor: alpha(theme.palette.warning.main, 0.055),
                                                                '& .MuiChip-label': {
                                                                    px: 0.58,
                                                                    minWidth: 0,
                                                                    overflow: 'hidden',
                                                                    textOverflow: 'ellipsis',
                                                                },
                                                            }}
                                                        />
                                                    </Tooltip>
                                                ))}
                                                {evidenceLabels.map(label => (
                                                    <Tooltip key={`drawer-evidence-${reference.id}-${label}`} title={label}>
                                                        <Chip
                                                            size="small"
                                                            label={label}
                                                            variant="outlined"
                                                            sx={{
                                                                height: 18,
                                                                maxWidth: 118,
                                                                fontSize: '0.58rem',
                                                                borderColor: alpha(theme.palette.info.main, 0.28),
                                                                bgcolor: alpha(theme.palette.info.main, 0.055),
                                                                '& .MuiChip-label': {
                                                                    px: 0.58,
                                                                    minWidth: 0,
                                                                    overflow: 'hidden',
                                                                    textOverflow: 'ellipsis',
                                                                },
                                                            }}
                                                        />
                                                    </Tooltip>
                                                ))}
                                            </Box>
                                        )}
                                    </Box>
                                )
                            })}
                            {hiddenDrawerReferenceCount > 0 && (
                                <Button
                                    size="small"
                                    variant="outlined"
                                    onClick={() => setShowAllDrawerReferences(true)}
                                    sx={{
                                        minHeight: 30,
                                        borderRadius: '8px',
                                        textTransform: 'none',
                                        fontSize: '0.72rem',
                                    }}
                                >
                                    Show {hiddenDrawerReferenceCount} more reference{hiddenDrawerReferenceCount === 1 ? '' : 's'}
                                </Button>
                            )}
                            {filteredReferences.length === 0 && (
                                <Typography variant="caption" color="text.secondary" sx={{ p: 0.75 }}>
                                    No matching references.
                                </Typography>
                            )}
                        </Stack>
                    </Box>

                    <Box sx={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                        <Box
                            sx={{
                                px: { xs: 1, sm: 1.5 },
                                py: 1,
                                display: 'flex',
                                alignItems: 'flex-start',
                                gap: 1,
                                borderBottom: '1px solid',
                                borderColor: 'divider',
                            }}
                        >
                            <Box sx={{ color: selectedIsPdf ? 'error.main' : activeAccentColor, display: 'flex', flexShrink: 0, mt: 0.2 }}>
                                {referenceIcon(selected?.kind || 'passage', selectedIsPdf)}
                            </Box>
                            <Box sx={{ minWidth: 0, flex: 1 }}>
                                <Typography
                                    variant="subtitle1"
                                    title={selected ? referenceBriefTitle(selected) : 'Reference'}
                                    sx={{
                                        fontWeight: 850,
                                        lineHeight: 1.22,
                                        display: '-webkit-box',
                                        overflow: 'hidden',
                                        WebkitBoxOrient: 'vertical',
                                        WebkitLineClamp: { xs: 2, sm: 1 },
                                        overflowWrap: 'anywhere',
                                    }}
                                >
                                    {selected ? referenceBriefTitle(selected) : 'Reference'}
                                </Typography>
                                <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    sx={{
                                        display: '-webkit-box',
                                        minWidth: 0,
                                        overflow: 'hidden',
                                        WebkitBoxOrient: 'vertical',
                                        WebkitLineClamp: { xs: 2, sm: 1 },
                                        lineHeight: 1.25,
                                    }}
                                >
                                    {uniqueStrings([
                                        referenceKindLabel(activeReference),
                                        selected
                                            ? selected.filename
                                                ? referenceBasename(selected.filename)
                                                : buildReferenceTraySubtitle(selected)
                                            : '',
                                    ]).map(value => compactReferenceText(value, 42)).join(' · ')}
                                </Typography>
                            </Box>
                            {selectedSourceViewEndpoint && (
                                <Tooltip title={`Open ${selectedSourceViewLabel.toLowerCase() || 'source view'}`}>
                                    <IconButton size="small" onClick={openRawDocument} sx={{ flexShrink: 0 }}>
                                        <OpenIcon fontSize="small" />
                                    </IconButton>
                                </Tooltip>
                            )}
                            {!selectedIsPdf && documentText && (
                                <Tooltip title="Copy document text">
                                    <IconButton size="small" onClick={copyDocument} sx={{ flexShrink: 0 }}>
                                        {copied ? <CheckIcon fontSize="small" /> : <CopyIcon fontSize="small" />}
                                    </IconButton>
                                </Tooltip>
                            )}
                            <Tooltip title="Close">
                                <IconButton size="small" onClick={() => setOpen(false)} sx={{ flexShrink: 0 }}>
                                    <CloseIcon fontSize="small" />
                                </IconButton>
                            </Tooltip>
                        </Box>

                        {!selectedIsRequestedEvidence && (selectedRelationshipParts || selectedPathNodes.length > 1 || selectedReferenceMapNodes.length > 1) && (
                            <Box
                                sx={{
                                    display: { xs: 'none', sm: 'block' },
                                    px: { xs: 1, sm: 1.5 },
                                    py: 1,
                                    borderBottom: '1px solid',
                                    borderColor: 'divider',
                                    bgcolor: alpha(activeAccentColor, 0.035),
                                }}
                            >
                                <Stack spacing={0.75}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.55, minWidth: 0, flexWrap: 'wrap' }}>
                                        {selectedPathNodes.length > 1 ? (
                                            <PathIcon sx={{ fontSize: 16, color: activeAccentColor, flexShrink: 0 }} />
                                        ) : (
                                            <GraphIcon sx={{ fontSize: 16, color: activeAccentColor, flexShrink: 0 }} />
                                        )}
                                        <Typography variant="caption" sx={{ fontWeight: 900 }}>
                                            {selectedPathNodes.length > 1 ? 'Graph path' : 'Graph relationship'}
                                        </Typography>
                                        <Chip
                                            size="small"
                                            label={referenceKindLabel(activeReference)}
                                            variant="outlined"
                                            sx={{ height: 20, fontSize: '0.64rem' }}
                                        />
                                        {selected?.path?.hops !== undefined && (
                                            <Chip
                                                size="small"
                                                label={`${selected.path.hops} hops`}
                                                sx={{ height: 20, fontSize: '0.64rem', bgcolor: alpha(activeAccentColor, 0.1) }}
                                            />
                                        )}
                                    </Box>

                                    {selectedRelationshipParts ? (
                                        <Box
                                            sx={{
                                                display: 'grid',
                                                gridTemplateColumns: { xs: '1fr', sm: 'minmax(0, 1fr) auto minmax(0, 1fr)' },
                                                alignItems: 'center',
                                                gap: 0.55,
                                            }}
                                        >
                                            {[
                                                {
                                                    key: 'source',
                                                    label: selectedRelationshipParts.sourceLabel,
                                                    detail: selectedRelationshipParts.sourceDetail,
                                                },
                                                {
                                                    key: 'target',
                                                    label: selectedRelationshipParts.targetLabel,
                                                    detail: selectedRelationshipParts.targetDetail,
                                                },
                                            ].map((item, index) => (
                                                <React.Fragment key={`selected-relationship-${item.key}`}>
                                                    {index > 0 && (
                                                        <Typography
                                                            variant="caption"
                                                            color="text.secondary"
                                                            noWrap
                                                            sx={{ display: { xs: 'none', sm: 'block' }, textAlign: 'center', fontWeight: 850 }}
                                                        >
                                                            {selectedRelationshipRelationLabel}
                                                        </Typography>
                                                    )}
                                                    <Box
                                                        sx={{
                                                            minWidth: 0,
                                                            border: '1px solid',
                                                            borderColor: alpha(activeAccentColor, 0.2),
                                                            borderRadius: '8px',
                                                            bgcolor: alpha(theme.palette.background.paper, 0.72),
                                                            px: 0.85,
                                                            py: 0.65,
                                                        }}
                                                    >
                                                        <Typography variant="caption" noWrap sx={{ display: 'block', fontWeight: 900 }}>
                                                            {item.label}
                                                        </Typography>
                                                        <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block', minHeight: '1.3em' }}>
                                                            {item.detail || (index === 0 ? 'Source' : 'Target')}
                                                        </Typography>
                                                    </Box>
                                                </React.Fragment>
                                            ))}
                                            <Typography
                                                variant="caption"
                                                color="text.secondary"
                                                sx={{ display: { xs: 'block', sm: 'none' }, fontWeight: 850 }}
                                            >
                                                Relationship: {selectedRelationshipRelationLabel}
                                            </Typography>
                                        </Box>
                                    ) : selectedPathNodes.length > 1 ? (
                                        selectedPathRelationships.length > 0 ? (
                                            <Stack spacing={0.5} sx={{ minWidth: 0 }}>
                                                {selectedPathRelationships.slice(0, 3).map((relationship, index) => (
                                                    <Box
                                                        key={`reference-path-hop-map-${index}-${relationship.source}-${relationship.target}`}
                                                        sx={{
                                                            display: 'grid',
                                                            gridTemplateColumns: { xs: '1fr', md: 'minmax(0, 1fr) auto minmax(0, 1fr)' },
                                                            alignItems: 'center',
                                                            gap: 0.45,
                                                            minWidth: 0,
                                                        }}
                                                    >
                                                        <Chip
                                                            size="small"
                                                            label={cleanGraphLabel(relationship.source, 'Source')}
                                                            sx={{
                                                                height: 22,
                                                                minWidth: 0,
                                                                maxWidth: { xs: '100%', md: 250 },
                                                                bgcolor: alpha(activeAccentColor, index === 0 ? 0.14 : 0.08),
                                                                '& .MuiChip-label': {
                                                                    minWidth: 0,
                                                                    overflow: 'hidden',
                                                                    textOverflow: 'ellipsis',
                                                                },
                                                            }}
                                                        />
                                                        <Chip
                                                            size="small"
                                                            label={formatGraphReferenceRelation(relationship.relation)}
                                                            variant="outlined"
                                                            sx={{
                                                                height: 22,
                                                                maxWidth: { xs: '100%', md: 180 },
                                                                justifySelf: { xs: 'start', md: 'center' },
                                                                borderColor: alpha(activeAccentColor, 0.34),
                                                                color: activeAccentColor,
                                                                fontSize: '0.64rem',
                                                                '& .MuiChip-label': {
                                                                    minWidth: 0,
                                                                    overflow: 'hidden',
                                                                    textOverflow: 'ellipsis',
                                                                },
                                                            }}
                                                        />
                                                        <Chip
                                                            size="small"
                                                            label={cleanGraphLabel(relationship.target, 'Target')}
                                                            sx={{
                                                                height: 22,
                                                                minWidth: 0,
                                                                maxWidth: { xs: '100%', md: 250 },
                                                                bgcolor: alpha(activeAccentColor, 0.08),
                                                                '& .MuiChip-label': {
                                                                    minWidth: 0,
                                                                    overflow: 'hidden',
                                                                    textOverflow: 'ellipsis',
                                                                },
                                                            }}
                                                        />
                                                    </Box>
                                                ))}
                                                {selectedPathRelationships.length > 3 && (
                                                    <Chip
                                                        size="small"
                                                        label={`+${selectedPathRelationships.length - 3} hops`}
                                                        variant="outlined"
                                                        sx={{ height: 22, width: 'fit-content', fontSize: '0.66rem' }}
                                                    />
                                                )}
                                            </Stack>
                                        ) : (
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.45, flexWrap: 'wrap', minWidth: 0 }}>
                                                {selectedPathNodes.slice(0, 6).map((node, index) => (
                                                    <React.Fragment key={`reference-path-map-${index}-${node}`}>
                                                        {index > 0 && (
                                                            <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0 }}>
                                                                to
                                                            </Typography>
                                                        )}
                                                        <Chip
                                                            size="small"
                                                            label={node}
                                                            sx={{
                                                                height: 22,
                                                                maxWidth: { xs: 160, sm: 220 },
                                                                bgcolor: alpha(activeAccentColor, index === 0 ? 0.14 : 0.08),
                                                                '& .MuiChip-label': {
                                                                    minWidth: 0,
                                                                    overflow: 'hidden',
                                                                    textOverflow: 'ellipsis',
                                                                },
                                                            }}
                                                        />
                                                    </React.Fragment>
                                                ))}
                                                {selectedPathNodes.length > 6 && (
                                                    <Chip
                                                        size="small"
                                                        label={`+${selectedPathNodes.length - 6}`}
                                                        variant="outlined"
                                                        sx={{ height: 22, fontSize: '0.66rem' }}
                                                    />
                                                )}
                                            </Box>
                                        )
                                    ) : (
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.55, flexWrap: 'wrap', minWidth: 0 }}>
                                            {selectedReferenceMapNodes.map((node, index) => (
                                                <React.Fragment key={`reference-map-${index}-${node}`}>
                                                    {index > 0 && (
                                                        <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0 }}>
                                                            to
                                                        </Typography>
                                                    )}
                                                    <Chip
                                                        size="small"
                                                        label={node}
                                                        sx={{
                                                            height: 22,
                                                            maxWidth: { xs: 180, sm: 260 },
                                                            bgcolor: alpha(activeAccentColor, index === 0 ? 0.14 : 0.08),
                                                            '& .MuiChip-label': {
                                                                minWidth: 0,
                                                                overflow: 'hidden',
                                                                textOverflow: 'ellipsis',
                                                            },
                                                        }}
                                                    />
                                                </React.Fragment>
                                            ))}
                                        </Box>
                                    )}

                                    {selectedVisibleRelationshipLead && (
                                        <Typography
                                            variant="caption"
                                            color="text.secondary"
                                            sx={{
                                                display: '-webkit-box',
                                                overflow: 'hidden',
                                                WebkitBoxOrient: 'vertical',
                                                WebkitLineClamp: 2,
                                                lineHeight: 1.35,
                                            }}
                                        >
                                            {selectedVisibleRelationshipLead}
                                        </Typography>
                                    )}
                                </Stack>
                            </Box>
                        )}

                        <Stack
                            spacing={0.75}
                            sx={{
                                px: { xs: 1, sm: 1.5 },
                                py: 0.85,
                                borderBottom: '1px solid',
                                borderColor: 'divider',
                            }}
                        >
                            <Box sx={{ display: { xs: 'block', md: 'none' } }}>
                                {showCompactDrawerSearch && (
                                    <TextField
                                        value={drawerSearch}
                                        onChange={event => setDrawerSearch(event.target.value)}
                                        placeholder="Find source"
                                        size="small"
                                        fullWidth
                                        sx={{
                                            '& .MuiOutlinedInput-root': {
                                                height: 32,
                                                fontSize: '0.76rem',
                                                borderRadius: '8px',
                                            },
                                        }}
                                    />
                                )}
                                <Box
                                    sx={{
                                        display: kindFilter !== 'all' || drawerSearch.trim() ? 'flex' : 'none',
                                        gap: 0.4,
                                        flexWrap: 'wrap',
                                        mt: 0.65,
                                    }}
                                >
                                    {referenceFilterChips.slice(0, 4).map(filter => (
                                        <Chip
                                            key={`mobile-${filter.value}`}
                                            label={`${filter.label} ${filter.count}`}
                                            size="small"
                                            clickable
                                            color={kindFilter === filter.value ? 'info' : 'default'}
                                            variant={kindFilter === filter.value ? 'filled' : 'outlined'}
                                            onClick={() => setKindFilter(filter.value)}
                                            sx={{
                                                height: 22,
                                                fontSize: '0.66rem',
                                                opacity: filter.tone === 'trace' && kindFilter !== filter.value ? 0.78 : 1,
                                            }}
                                        />
                                    ))}
                                </Box>
                                {sourceNavigationGroups.length > 1 && (
                                    <Box
                                        data-testid="reference-mobile-source-selector"
                                        sx={{
                                            mt: 0.65,
                                            minWidth: 0,
                                        }}
                                    >
                                        <Box
                                            sx={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'space-between',
                                                gap: 0.75,
                                                mb: 0.45,
                                            }}
                                        >
                                            <Typography
                                                variant="caption"
                                                color="text.secondary"
                                                sx={{ display: 'block', fontWeight: 850 }}
                                            >
                                                Documents
                                            </Typography>
                                            <Chip
                                                label={formatReferenceCount(sourceNavigationGroups.length, 'source')}
                                                size="small"
                                                variant="outlined"
                                                sx={{ height: 19, fontSize: '0.62rem', flexShrink: 0 }}
                                            />
                                        </Box>
                                        <Box
                                            sx={{
                                                display: 'flex',
                                                gap: 0.45,
                                                overflowX: 'auto',
                                                overflowY: 'hidden',
                                                pb: 0.2,
                                                pr: 0.15,
                                                scrollSnapType: 'x proximity',
                                                WebkitOverflowScrolling: 'touch',
                                            }}
                                        >
                                            <Chip
                                                label="All documents"
                                                size="small"
                                                clickable
                                                color={documentFilter === 'all' ? 'info' : 'default'}
                                                variant={documentFilter === 'all' ? 'filled' : 'outlined'}
                                                onClick={clearSourceGroup}
                                                sx={{
                                                    height: 28,
                                                    maxWidth: 156,
                                                    flex: '0 0 auto',
                                                    justifyContent: 'flex-start',
                                                    scrollSnapAlign: 'start',
                                                    '& .MuiChip-label': {
                                                        minWidth: 0,
                                                        overflow: 'hidden',
                                                        textOverflow: 'ellipsis',
                                                    },
                                                }}
                                            />
                                            {sourceNavigationGroups.map(group => {
                                                const selectedGroup = documentFilter === group.key
                                                return (
                                                    <Chip
                                                        key={`mobile-group-${group.key}`}
                                                        label={`${group.title} (${group.references.length})`}
                                                        size="small"
                                                        clickable
                                                        icon={referenceIcon(group.bestReference.kind, isPdfReference(group.bestReference))}
                                                        color={selectedGroup ? 'info' : 'default'}
                                                        variant={selectedGroup ? 'filled' : 'outlined'}
                                                        onClick={() => selectSourceGroup(group)}
                                                        sx={{
                                                            height: 28,
                                                            maxWidth: selectedGroup ? 220 : 190,
                                                            minWidth: selectedGroup ? 148 : 116,
                                                            flex: '0 0 auto',
                                                            justifyContent: 'flex-start',
                                                            scrollSnapAlign: 'start',
                                                            '& .MuiChip-label': {
                                                                minWidth: 0,
                                                                overflow: 'hidden',
                                                                textOverflow: 'ellipsis',
                                                            },
                                                        }}
                                                    />
                                                )
                                            })}
                                        </Box>
                                    </Box>
                                )}
                                <Box
                                    data-testid="reference-mobile-evidence-list"
                                    sx={{
                                        mt: 0.75,
                                        minWidth: 0,
                                    }}
                                >
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 0.75, mb: 0.45 }}>
                                        <Typography
                                            variant="caption"
                                            color="text.secondary"
                                            sx={{ display: 'block', fontWeight: 850 }}
                                        >
                                            Evidence
                                        </Typography>
                                        <Chip
                                            size="small"
                                            label={formatReferenceCount(drawerReferenceRows.length + hiddenDrawerReferenceCount, 'reference')}
                                            variant="outlined"
                                            sx={{ height: 19, fontSize: '0.62rem', flexShrink: 0 }}
                                        />
                                    </Box>
                                    <Box
                                        sx={{
                                            display: 'flex',
                                            gap: 0.45,
                                            overflowX: 'auto',
                                            overflowY: 'hidden',
                                            pb: 0.2,
                                            pr: 0.15,
                                            scrollSnapType: 'x proximity',
                                            WebkitOverflowScrolling: 'touch',
                                        }}
                                    >
                                        {drawerReferenceRows.map(reference => {
                                            const referenceSelected = selected?.id === reference.id
                                            return (
                                                <Chip
                                                    key={`mobile-source-${reference.id}`}
                                                    label={referenceBriefTitle(reference)}
                                                    size="small"
                                                    clickable
                                                    data-testid={isPdfReference(reference) ? 'reference-drawer-row-pdf' : 'reference-drawer-row'}
                                                    data-reference-title={referenceBriefTitle(reference)}
                                                    icon={referenceIcon(reference.kind, isPdfReference(reference))}
                                                    color={referenceSelected ? 'info' : 'default'}
                                                    variant={referenceSelected ? 'filled' : 'outlined'}
                                                    onClick={() => selectReference(reference)}
                                                    sx={{
                                                        height: 30,
                                                        maxWidth: 260,
                                                        minWidth: referenceSelected ? 190 : 128,
                                                        flex: '0 0 auto',
                                                        justifyContent: 'flex-start',
                                                        scrollSnapAlign: 'start',
                                                        borderColor: referenceSelected ? alpha(activeAccentColor, 0.5) : undefined,
                                                        bgcolor: referenceSelected ? alpha(activeAccentColor, 0.14) : undefined,
                                                        '& .MuiChip-label': {
                                                            minWidth: 0,
                                                            overflow: 'hidden',
                                                            textOverflow: 'ellipsis',
                                                            fontWeight: referenceSelected ? 850 : 650,
                                                        },
                                                    }}
                                                />
                                            )
                                        })}
                                        {hiddenDrawerReferenceCount > 0 && (
                                            <Button
                                                size="small"
                                                variant="outlined"
                                                onClick={() => setShowAllDrawerReferences(true)}
                                                sx={{
                                                    minHeight: 30,
                                                    borderRadius: '8px',
                                                    textTransform: 'none',
                                                    fontSize: '0.72rem',
                                                    flex: '0 0 auto',
                                                    whiteSpace: 'nowrap',
                                                }}
                                            >
                                                Show {hiddenDrawerReferenceCount} more
                                            </Button>
                                        )}
                                    </Box>
                                </Box>
                            </Box>
                            {selected && (selectedEvidenceSummaryTitle || selectedEvidenceLead) && (
                                <Box
                                    data-testid="reference-selected-evidence"
                                    sx={{
                                        border: '1px solid',
                                        borderColor: alpha(activeAccentColor, 0.18),
                                        borderRadius: '8px',
                                        bgcolor: alpha(activeAccentColor, 0.045),
                                        overflow: 'hidden',
                                    }}
                                >
                                    <Box
                                        sx={{
                                            display: 'grid',
                                            gridTemplateColumns: { xs: '1fr', sm: 'minmax(0, 1fr) auto' },
                                            gap: 0.75,
                                            alignItems: 'flex-start',
                                            px: 0.9,
                                            py: 0.75,
                                            borderBottom: selectedEvidenceLead || selectedLinkTarget ? '1px solid' : 0,
                                            borderColor: alpha(theme.palette.divider, 0.6),
                                        }}
                                    >
                                        <Box sx={{ minWidth: 0 }}>
                                            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontWeight: 850 }}>
                                                Selected evidence
                                            </Typography>
                                            <Typography
                                                variant="body2"
                                                title={selectedEvidenceSummaryTitle}
                                                sx={{
                                                    fontWeight: 900,
                                                    display: '-webkit-box',
                                                    overflow: 'hidden',
                                                    WebkitBoxOrient: 'vertical',
                                                    WebkitLineClamp: 2,
                                                    lineHeight: 1.35,
                                                }}
                                            >
                                                {selectedEvidenceSummaryTitle}
                                            </Typography>
                                            {(selectedEvidenceContext || selectedSourceBrief?.subtitle) && (
                                                <Typography
                                                    variant="caption"
                                                    color="text.secondary"
                                                    noWrap
                                                    sx={{
                                                        display: 'block',
                                                        minWidth: 0,
                                                        overflow: 'hidden',
                                                        textOverflow: 'ellipsis',
                                                    }}
                                                >
                                                    {selectedEvidenceContext || selectedSourceBrief?.subtitle}
                                                </Typography>
                                            )}
                                        </Box>
                                    </Box>
                                    {selectedLinkTarget && (
                                        <Box
                                            sx={{
                                                px: 0.9,
                                                py: 0.7,
                                                borderBottom: selectedEvidenceLead ? '1px solid' : 0,
                                                borderColor: alpha(theme.palette.divider, 0.55),
                                                bgcolor: alpha(theme.palette.info.main, 0.035),
                                            }}
                                        >
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.6, mb: 0.35 }}>
                                                <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 850, flex: 1 }}>
                                                    Link target
                                                </Typography>
                                                <Tooltip title="Open link target">
                                                    <IconButton
                                                        size="small"
                                                        component="a"
                                                        href={selectedLinkTarget}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        sx={{ p: 0.35 }}
                                                    >
                                                        <OpenIcon sx={{ fontSize: 15 }} />
                                                    </IconButton>
                                                </Tooltip>
                                                <Tooltip title="Copy link target">
                                                    <IconButton
                                                        size="small"
                                                        onClick={async () => {
                                                            await navigator.clipboard.writeText(selectedLinkTarget)
                                                            setCopied(true)
                                                            window.setTimeout(() => setCopied(false), 1200)
                                                        }}
                                                        sx={{ p: 0.35 }}
                                                    >
                                                        {copied ? <CheckIcon sx={{ fontSize: 15 }} /> : <CopyIcon sx={{ fontSize: 15 }} />}
                                                    </IconButton>
                                                </Tooltip>
                                            </Box>
                                            <Typography
                                                component="a"
                                                href={selectedLinkTarget}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                variant="body2"
                                                sx={{
                                                    color: 'info.light',
                                                    display: 'block',
                                                    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                                                    fontSize: { xs: '0.78rem', sm: '0.8rem' },
                                                    lineHeight: 1.35,
                                                    overflowWrap: 'anywhere',
                                                    textDecoration: 'none',
                                                    '&:hover': { textDecoration: 'underline' },
                                                }}
                                            >
                                                {selectedLinkTarget}
                                            </Typography>
                                        </Box>
                                    )}
                                    {selectedEvidenceLead && (
                                        <Typography
                                            variant="body2"
                                            sx={{
                                                px: 0.9,
                                                pt: 0.75,
                                                pb: selectedEvidenceLabels.length || selectedLocationLabels.length ? 0.55 : 0.8,
                                                display: '-webkit-box',
                                                overflow: 'hidden',
                                                WebkitBoxOrient: 'vertical',
                                                WebkitLineClamp: 3,
                                                color: 'text.primary',
                                                fontSize: '0.82rem',
                                                lineHeight: 1.45,
                                            }}
                                        >
                                            <HighlightedText text={selectedEvidenceLead} terms={terms} />
                                        </Typography>
                                    )}
                                    {(selectedEvidenceLabels.length > 0 || selectedLocationLabels.length > 0) && (
                                        <Box
                                            sx={{
                                                display: 'flex',
                                                gap: 0.4,
                                                flexWrap: 'wrap',
                                                px: 0.9,
                                                pb: 0.8,
                                            }}
                                        >
                                            {selectedLocationLabels.slice(0, 3).map(label => (
                                                <Tooltip key={`source-brief-location-${label}`} title={label}>
                                                    <Chip
                                                        size="small"
                                                        label={label}
                                                        variant="outlined"
                                                        sx={{
                                                            height: 21,
                                                            maxWidth: { xs: 180, sm: 240 },
                                                            fontSize: '0.64rem',
                                                            borderColor: alpha(theme.palette.warning.main, 0.28),
                                                            bgcolor: alpha(theme.palette.warning.main, 0.06),
                                                            '& .MuiChip-label': {
                                                                minWidth: 0,
                                                                overflow: 'hidden',
                                                                textOverflow: 'ellipsis',
                                                            },
                                                        }}
                                                    />
                                                </Tooltip>
                                            ))}
                                            {selectedEvidenceLabels.slice(0, 3).map(label => (
                                                <Tooltip key={`source-brief-evidence-${label}`} title={label}>
                                                    <Chip
                                                        size="small"
                                                        label={label}
                                                        variant="outlined"
                                                        sx={{
                                                            height: 21,
                                                            maxWidth: { xs: 180, sm: 220 },
                                                            fontSize: '0.64rem',
                                                            borderColor: alpha(theme.palette.info.main, 0.26),
                                                            bgcolor: alpha(theme.palette.info.main, 0.055),
                                                            '& .MuiChip-label': {
                                                                minWidth: 0,
                                                                overflow: 'hidden',
                                                                textOverflow: 'ellipsis',
                                                            },
                                                        }}
                                                    />
                                                </Tooltip>
                                            ))}
                                        </Box>
                                    )}
                                </Box>
                            )}
                            {selectedReferenceDetailCount > 0 && (
                                <Box
                                    component="details"
                                    sx={{
                                        border: '1px solid',
                                        borderColor: alpha(activeAccentColor, 0.16),
                                        borderRadius: '8px',
                                        bgcolor: alpha(activeAccentColor, 0.035),
                                        px: 0.75,
                                        py: 0.65,
                                        '& summary': {
                                            cursor: 'pointer',
                                            listStyle: 'none',
                                        },
                                        '& summary::-webkit-details-marker': {
                                            display: 'none',
                                        },
                                    }}
                                >
                                    <Box
                                        component="summary"
                                        sx={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'space-between',
                                            gap: 0.75,
                                            minWidth: 0,
                                        }}
                                    >
                                        <Typography variant="caption" sx={{ fontWeight: 850 }}>
                                            Evidence details
                                        </Typography>
                                        <Chip
                                            size="small"
                                            label={formatReferenceCount(selectedReferenceDetailCount, 'field')}
                                            variant="outlined"
                                            sx={{ height: 20, flexShrink: 0, fontSize: '0.64rem' }}
                                        />
                                    </Box>
                                    <Stack spacing={0.65} sx={{ mt: 0.75 }}>
                            {(selectedEvidenceLabels.length > 0 || selectedLocationLabels.length > 0) && (
                                <Box sx={{ display: 'flex', gap: 0.45, flexWrap: 'wrap' }}>
                                    {selectedSourceViewLabel && (
                                        <Chip
                                            size="small"
                                            label={selectedSourceViewLabel}
                                            variant="outlined"
                                            sx={{
                                                height: 22,
                                                maxWidth: { xs: 150, sm: 190 },
                                                fontSize: '0.68rem',
                                                '& .MuiChip-label': {
                                                    minWidth: 0,
                                                    overflow: 'hidden',
                                                    textOverflow: 'ellipsis',
                                                },
                                            }}
                                        />
                                    )}
                                    {selectedEvidenceLabels.map(label => (
                                        <Chip
                                            key={`selected-evidence-${label}`}
                                            size="small"
                                            label={label}
                                            sx={{
                                                height: 22,
                                                maxWidth: { xs: 170, sm: 230 },
                                                fontSize: '0.68rem',
                                                bgcolor: alpha(theme.palette.info.main, 0.1),
                                                '& .MuiChip-label': {
                                                    minWidth: 0,
                                                    overflow: 'hidden',
                                                    textOverflow: 'ellipsis',
                                                },
                                            }}
                                        />
                                    ))}
                                    {selectedLocationLabels.map(label => (
                                        <Tooltip key={`selected-location-${label}`} title={label}>
                                            <Chip
                                                size="small"
                                                label={label}
                                                sx={{
                                                    height: 22,
                                                    fontSize: '0.68rem',
                                                    maxWidth: { xs: 190, sm: 260 },
                                                    bgcolor: alpha(theme.palette.warning.main, 0.1),
                                                    '& .MuiChip-label': {
                                                        minWidth: 0,
                                                        overflow: 'hidden',
                                                        textOverflow: 'ellipsis',
                                                    },
                                                }}
                                            />
                                        </Tooltip>
                                    ))}
                                </Box>
                            )}
                            {selectedMatchLabels.length > 0 && (
                                <Box sx={{ display: 'flex', gap: 0.45, flexWrap: 'wrap' }}>
                                    {selectedMatchLabels.map(label => (
                                        <Chip
                                            key={`selected-match-${label}`}
                                            size="small"
                                            label={label}
                                            variant="outlined"
                                            sx={{
                                                height: 21,
                                                maxWidth: { xs: 170, sm: 230 },
                                                fontSize: '0.66rem',
                                                '& .MuiChip-label': {
                                                    minWidth: 0,
                                                    overflow: 'hidden',
                                                    textOverflow: 'ellipsis',
                                                },
                                            }}
                                        />
                                    ))}
                                </Box>
                            )}
                            {selectedVisibleStructuredLocations.length > 0 && (
                                <Box
                                    sx={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 0.45,
                                        flexWrap: 'wrap',
                                        p: 0.65,
                                        border: '1px solid',
                                        borderColor: alpha(theme.palette.warning.main, 0.16),
                                        borderRadius: '8px',
                                        bgcolor: alpha(theme.palette.warning.main, 0.045),
                                    }}
                                >
                                    <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 800 }}>
                                        Location
                                    </Typography>
                                    {selectedVisibleStructuredLocations.slice(0, 10).map(({ location, label }) => (
                                        <Tooltip
                                            key={`selected-source-location-${location.label}-${location.value}`}
                                            title={[formatReferenceFacet(location.label), location.value].filter(Boolean).join(': ')}
                                        >
                                            <Chip
                                                size="small"
                                                label={label}
                                                variant="outlined"
                                                sx={{
                                                    height: 22,
                                                    maxWidth: { xs: 190, sm: 260 },
                                                    fontSize: '0.66rem',
                                                    bgcolor: alpha(theme.palette.background.paper, 0.54),
                                                    '& .MuiChip-label': {
                                                        minWidth: 0,
                                                        overflow: 'hidden',
                                                        textOverflow: 'ellipsis',
                                                    },
                                                }}
                                            />
                                        </Tooltip>
                                    ))}
                                    {selectedStructuredLocationOverflowCount > 0 && (
                                        <Chip
                                            size="small"
                                            label={`+${selectedStructuredLocationOverflowCount} more`}
                                            variant="outlined"
                                            sx={{
                                                height: 22,
                                                fontSize: '0.66rem',
                                                color: 'text.secondary',
                                                bgcolor: alpha(theme.palette.background.paper, 0.54),
                                            }}
                                        />
                                    )}
                                </Box>
                            )}
                            {terms.length > 0 && selectedMatchLabels.length === 0 && (
                                <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                                    {terms.slice(0, 10).map(term => (
                                        <Chip
                                            key={term}
                                            size="small"
                                            label={term}
                                            sx={{
                                                height: 20,
                                                maxWidth: { xs: 160, sm: 220 },
                                                fontSize: '0.66rem',
                                                '& .MuiChip-label': {
                                                    minWidth: 0,
                                                    overflow: 'hidden',
                                                    textOverflow: 'ellipsis',
                                                },
                                            }}
                                        />
                                    ))}
                                </Box>
                            )}
                            {selectedPathNodes.length > 0 && (
                                <Box
                                    sx={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 0.45,
                                        flexWrap: 'wrap',
                                    }}
                                >
                                    {selectedPathNodes.map((node, nodeIndex) => (
                                        <React.Fragment key={`selected-path-${nodeIndex}-${node}`}>
                                            {nodeIndex > 0 && (
                                                <Typography variant="caption" color="text.secondary">
                                                    to
                                                </Typography>
                                            )}
                                            <Chip
                                                size="small"
                                                label={node}
                                                sx={{
                                                    height: 22,
                                                    maxWidth: { xs: 190, sm: 280 },
                                                    bgcolor: alpha(activeAccentColor, 0.1),
                                                    '& .MuiChip-label': {
                                                        minWidth: 0,
                                                        overflow: 'hidden',
                                                        textOverflow: 'ellipsis',
                                                    },
                                                }}
                                            />
                                        </React.Fragment>
                                    ))}
                                </Box>
                            )}
                            {selectedSourceGroup && selectedSourceGroupIsNavigable && sourceNavigationGroups.length > 1 && (
                                <Box
                                    sx={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 0.5,
                                        flexWrap: 'wrap',
                                        p: 0.75,
                                        border: '1px solid',
                                        borderColor: alpha(activeAccentColor, 0.18),
                                        borderRadius: '8px',
                                        bgcolor: alpha(activeAccentColor, 0.045),
                                    }}
                                >
                                    <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 800 }}>
                                        Source
                                    </Typography>
                                    <Chip
                                        size="small"
                                        label={selectedSourceGroup.title}
                                        sx={{
                                            height: 22,
                                            maxWidth: { xs: 190, sm: 320 },
                                            bgcolor: alpha(activeAccentColor, 0.1),
                                            '& .MuiChip-label': {
                                                minWidth: 0,
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                            },
                                        }}
                                    />
                                    <Chip
                                        size="small"
                                        label={formatReferenceCount(selectedSourceGroup.references.length, 'reference')}
                                        variant="outlined"
                                        sx={{ height: 22, fontSize: '0.68rem' }}
                                    />
                                    {referenceGroupSummary(selectedSourceGroup) && (
                                        <Typography variant="caption" color="text.secondary">
                                            {referenceGroupSummary(selectedSourceGroup)}
                                        </Typography>
                                    )}
                                </Box>
                            )}
                                    </Stack>
                                </Box>
                            )}
                        </Stack>

                        {selectedPathRelationships.length > 0 && !selectedHasPdfViewer && (
                            <Box
                                component="details"
                                sx={{
                                    px: { xs: 1, sm: 1.5 },
                                    py: 0.85,
                                    borderBottom: '1px solid',
                                    borderColor: 'divider',
                                    bgcolor: alpha(activeAccentColor, 0.035),
                                    '& summary': {
                                        cursor: 'pointer',
                                        listStyle: 'none',
                                    },
                                    '& summary::-webkit-details-marker': {
                                        display: 'none',
                                    },
                                }}
                            >
                                {(() => {
                                    const pathEvidenceLabels = buildReferencePathEvidenceLabels(selected)
                                    return (
                                <Box
                                    component="summary"
                                    sx={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'space-between',
                                        gap: 0.75,
                                        minWidth: 0,
                                    }}
                                >
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.6, minWidth: 0, flexWrap: 'wrap' }}>
                                        <PathIcon sx={{ fontSize: 16, color: activeAccentColor, flexShrink: 0 }} />
                                        <Typography variant="caption" sx={{ fontWeight: 850 }}>
                                            Path evidence
                                        </Typography>
                                        {pathEvidenceLabels.map(label => (
                                            <Chip
                                                key={`selected-path-evidence-${label}`}
                                                size="small"
                                                label={label}
                                                sx={{ height: 20, fontSize: '0.66rem', bgcolor: alpha(activeAccentColor, 0.11) }}
                                            />
                                        ))}
                                    </Box>
                                    <Chip
                                        size="small"
                                        label="Show hops"
                                        variant="outlined"
                                        sx={{ height: 20, flexShrink: 0, fontSize: '0.64rem' }}
                                    />
                                </Box>
                                    )
                                })()}
                                <Stack spacing={0.5} sx={{ mt: 0.75 }}>
                                    {selectedPathRelationships.slice(0, 5).map((relationship, index) => (
                                            <Box
                                                key={`selected-hop-${index}-${relationship.source}-${relationship.target}`}
                                                sx={{
                                                    display: 'grid',
                                                    gridTemplateColumns: { xs: '1fr', sm: 'minmax(0, 1fr) auto minmax(0, 1fr)' },
                                                    alignItems: 'center',
                                                    gap: 0.5,
                                                    px: 0.75,
                                                    py: 0.6,
                                                    borderRadius: '8px',
                                                    border: '1px solid',
                                                    borderColor: alpha(activeAccentColor, 0.2),
                                                    bgcolor: alpha(theme.palette.background.paper, 0.72),
                                                }}
                                            >
                                                <Chip
                                                    size="small"
                                                    label={cleanGraphLabel(relationship.source, 'Source')}
                                                    sx={{
                                                        minWidth: 0,
                                                        maxWidth: { xs: '100%', sm: 220 },
                                                        justifySelf: 'stretch',
                                                        '& .MuiChip-label': {
                                                            minWidth: 0,
                                                            overflow: 'hidden',
                                                            textOverflow: 'ellipsis',
                                                        },
                                                    }}
                                                />
                                                <Typography
                                                    variant="caption"
                                                    color="text.secondary"
                                                    sx={{
                                                        minWidth: 0,
                                                        textAlign: { xs: 'left', sm: 'center' },
                                                        wordBreak: 'break-word',
                                                    }}
                                                >
                                                    {formatGraphReferenceRelation(relationship.relation)}
                                                </Typography>
                                                <Chip
                                                    size="small"
                                                    label={cleanGraphLabel(relationship.target, 'Target')}
                                                    sx={{
                                                        minWidth: 0,
                                                        maxWidth: { xs: '100%', sm: 220 },
                                                        justifySelf: 'stretch',
                                                        '& .MuiChip-label': {
                                                            minWidth: 0,
                                                            overflow: 'hidden',
                                                            textOverflow: 'ellipsis',
                                                        },
                                                    }}
                                                />
                                                {relationship.evidence && (
                                                    <Typography
                                                        variant="caption"
                                                        color="text.secondary"
                                                        sx={{
                                                            gridColumn: { xs: '1', sm: '1 / -1' },
                                                            display: '-webkit-box',
                                                            overflow: 'hidden',
                                                            WebkitBoxOrient: 'vertical',
                                                            WebkitLineClamp: 2,
                                                        }}
                                                    >
                                                        {relationship.evidence}
                                                    </Typography>
                                                )}
                                            </Box>
                                    ))}
                                </Stack>
                            </Box>
                        )}

                        <Box
                            sx={{
                                flex: 1,
                                minHeight: 0,
                                overflow: sourceTextExpanded ? 'auto' : 'hidden',
                                p: { xs: 1, sm: 1.5 },
                            }}
                        >
                            {selectedHasPdfViewer && selectedPdfViewerReference ? (
                                <PdfReferenceViewer
                                    reference={selectedPdfViewerReference}
                                    terms={selectedPdfTerms}
                                    references={selectedReviewReferences}
                                />
                            ) : (
                                <Stack
                                    spacing={1}
                                    sx={{
                                        height: { xs: sourceTextExpanded ? 'auto' : '100%', md: '100%' },
                                        minHeight: 0,
                                    }}
                                >
                                    {showSelectedDocumentPreviewCard && (
                                        <Paper
                                            variant="outlined"
                                            sx={{
                                                order: sourceTextExpanded ? 2 : 1,
                                                borderColor: alpha(activeAccentColor, 0.2),
                                                bgcolor: alpha(activeAccentColor, 0.035),
                                                overflow: 'hidden',
                                            }}
                                        >
                                            <Box
                                                sx={{
                                                    display: 'grid',
                                                    gridTemplateColumns: { xs: '1fr', sm: 'minmax(0, 1fr) auto' },
                                                    gap: 0.75,
                                                    alignItems: 'flex-start',
                                                    px: { xs: 1, sm: 1.1 },
                                                    py: 0.9,
                                                    borderBottom: '1px solid',
                                                    borderColor: alpha(theme.palette.divider, 0.66),
                                                }}
                                            >
                                                <Box sx={{ minWidth: 0 }}>
                                                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontWeight: 850 }}>
                                                        {selectedDocumentPreviewSummary?.eyebrow || 'Document preview'}
                                                    </Typography>
                                                    <Typography
                                                        variant="body2"
                                                        noWrap
                                                        title={selectedDocumentPreviewSummary?.title || referenceDocumentTitle(activeReference)}
                                                        sx={{
                                                            fontWeight: 900,
                                                            overflow: 'hidden',
                                                            textOverflow: 'ellipsis',
                                                        }}
                                                    >
                                                        {selectedDocumentPreviewSummary?.title || referenceDocumentTitle(activeReference)}
                                                    </Typography>
                                                    {selectedDocumentPreviewSummary?.subtitle && (
                                                        <Typography
                                                            variant="caption"
                                                            color="text.secondary"
                                                            noWrap
                                                            sx={{
                                                                display: 'block',
                                                                minWidth: 0,
                                                                overflow: 'hidden',
                                                                textOverflow: 'ellipsis',
                                                            }}
                                                        >
                                                            {selectedDocumentPreviewSummary.subtitle}
                                                        </Typography>
                                                    )}
                                                </Box>
                                                <Box sx={{ display: 'flex', gap: 0.4, flexWrap: 'wrap', justifyContent: { xs: 'flex-start', sm: 'flex-end' } }}>
                                                    {(selectedDocumentPreviewSummary?.chips?.length
                                                        ? selectedDocumentPreviewSummary.chips
                                                        : selectedSourceViewLabel
                                                            ? [selectedSourceViewLabel]
                                                            : []
                                                    ).map(label => (
                                                        <Chip
                                                            key={`document-preview-chip-${label}`}
                                                            size="small"
                                                            label={label}
                                                            variant="outlined"
                                                            sx={{
                                                                height: 22,
                                                                maxWidth: { xs: 150, sm: 190 },
                                                                fontSize: '0.66rem',
                                                                '& .MuiChip-label': {
                                                                    minWidth: 0,
                                                                    overflow: 'hidden',
                                                                    textOverflow: 'ellipsis',
                                                                },
                                                            }}
                                                        />
                                                    ))}
                                                </Box>
                                            </Box>
                                            {loadingDocument && (
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, px: 1.1, py: 0.9 }}>
                                                    <CircularProgress size={16} />
                                                    <Typography variant="body2" color="text.secondary">
                                                        Loading document
                                                    </Typography>
                                                </Box>
                                            )}
                                            {documentError && (
                                                <Alert severity="warning" sx={{ borderRadius: 0 }}>
                                                    {documentError}
                                                </Alert>
                                            )}
                                            {selectedLayoutMap && (
                                                <ReferenceLayoutMapView
                                                    map={selectedLayoutMap}
                                                    accentColor={activeAccentColor}
                                                />
                                            )}
                                            {selectedDocumentPreviewTextForDisplay && (
                                                <Typography
                                                    variant="body2"
                                                    sx={{
                                                        px: { xs: 1, sm: 1.1 },
                                                        pt: selectedLayoutMap ? 0.1 : 0.95,
                                                        pb: 0.95,
                                                        color: 'text.primary',
                                                        lineHeight: 1.55,
                                                    }}
                                                >
                                                    <HighlightedText
                                                        text={selectedDocumentPreviewTextForDisplay}
                                                        terms={selectedDocumentTerms}
                                                    />
                                                </Typography>
                                            )}
                                            {selectedDocumentPreviewOutline.length > 0 && (
                                                <Box
                                                    sx={{
                                                        display: 'flex',
                                                        gap: 0.45,
                                                        flexWrap: 'wrap',
                                                        px: { xs: 1, sm: 1.1 },
                                                        pb: 1,
                                                    }}
                                                >
                                                    {selectedDocumentPreviewOutline.map(item => (
                                                        <Chip
                                                            key={`source-outline-${item}`}
                                                            size="small"
                                                            label={compactReferenceText(item, 42)}
                                                            variant="outlined"
                                                            sx={{
                                                                height: 22,
                                                                maxWidth: 220,
                                                                fontSize: '0.66rem',
                                                                bgcolor: alpha(theme.palette.background.paper, 0.48),
                                                                '& .MuiChip-label': {
                                                                    minWidth: 0,
                                                                    overflow: 'hidden',
                                                                    textOverflow: 'ellipsis',
                                                                },
                                                            }}
                                                        />
                                                    ))}
                                                </Box>
                                            )}
                                        </Paper>
                                    )}
                                    <Paper
                                        variant="outlined"
                                        sx={{
                                            order: sourceTextExpanded ? 1 : 2,
                                            flex: '0 0 auto',
                                            minHeight: sourceTextExpanded ? { xs: 320, md: 360 } : 0,
                                            borderColor: alpha(theme.palette.divider, 0.82),
                                            borderRadius: '8px',
                                            bgcolor: 'background.paper',
                                            overflow: 'hidden',
                                        }}
                                    >
                                        <Button
                                            fullWidth
                                            onClick={() => setSourceTextExpanded(value => !value)}
                                            aria-expanded={sourceTextExpanded}
                                            aria-label={sourceTextExpanded ? 'Hide full document text' : 'Show full document text'}
                                            sx={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'space-between',
                                                gap: 0.75,
                                                px: 1,
                                                py: 0.85,
                                                minHeight: 0,
                                                textAlign: 'left',
                                                textTransform: 'none',
                                                borderRadius: 0,
                                                borderBottom: '1px solid',
                                                borderColor: alpha(theme.palette.divider, 0.66),
                                                color: 'text.primary',
                                            }}
                                        >
                                            <Box sx={{ minWidth: 0 }}>
                                                <Typography variant="caption" sx={{ display: 'block', fontWeight: 900 }}>
                                                    Source text
                                                </Typography>
                                                <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block' }}>
                                                    {sourceTextExpanded ? 'Highlighted source lines' : 'Open highlighted source lines'}
                                                </Typography>
                                            </Box>
                                            <Chip
                                                size="small"
                                                label={sourceTextExpanded ? 'Hide' : 'Open'}
                                                variant="outlined"
                                                sx={{ height: 22, flexShrink: 0 }}
                                            />
                                        </Button>
                                        {sourceTextExpanded && (
                                            <DocumentTextViewer
                                                text={documentText || 'No document text available.'}
                                                terms={selectedDocumentTerms}
                                            />
                                        )}
                                    </Paper>
                                </Stack>
                            )}
                        </Box>
                    </Box>
                </Box>
            </Drawer>
        </>
    )
}

export default ReferenceExplorer
