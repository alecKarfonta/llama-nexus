import { cleanDocumentTextForReading } from './chatReferenceText'
import {
    normalizeReferenceFacetText,
    preferredReviewReference,
    type ReviewReference,
    type ReviewReferenceKind,
} from './chatReferenceReviewScoring'

export interface ChatReferenceSourceLocation {
    label?: string
    value?: string
    category?: string
}

export interface ChatReferenceSourceBriefReference extends ReviewReference {
    documentId?: string
    runId?: string
    chunkIndex?: number
    locationHints?: string[]
    sourceLocations?: ChatReferenceSourceLocation[]
    path?: ReviewReference['path'] & {
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

export interface ChatReferenceSourceBrief {
    title: string
    subtitle: string
    lead: string
    countLabels: string[]
    evidenceLabels: string[]
    locationLabels: string[]
    referenceCount: number
    bestReferenceId?: string
}

const HIGH_SIGNAL_EVIDENCE_PATTERNS = [
    /\bsemantic\b/,
    /\bconnector\b/,
    /\bchart\b/,
    /\bform\b/,
    /\bdiagram\b/,
    /\btable\b/,
    /\bstyled\b/,
    /\bnotes?\b/,
    /\bocr\b/,
    /\bcode\b/,
]

const STRUCTURAL_EVIDENCE_PATTERNS = [
    /\blayout\b/,
    /\bmetadata\b/,
    /\bpresentation\s+structure\b/,
    /\bworkbook\s+structure\b/,
    /\bsource\s+document\b/,
    /\bsource\s+file\b/,
    /\bsection\b/,
]

const uniqueStrings = (values: Array<string | undefined | null>): string[] => {
    const seen = new Set<string>()
    const output: string[] = []

    values.forEach(value => {
        const normalized = String(value || '').replace(/\s+/g, ' ').trim()
        const key = normalized.toLowerCase()
        if (!normalized || seen.has(key)) return
        seen.add(key)
        output.push(normalized)
    })

    return output
}

const compactText = (value: string, limit = 220): string => {
    const text = value.replace(/\s+/g, ' ').trim()
    if (text.length <= limit) return text
    return `${text.slice(0, Math.max(1, limit - 1)).trim()}...`
}

const titleCaseFacet = (value: string): string =>
    value
        .replace(/[_-]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
        .replace(/\b\w/g, letter => letter.toUpperCase())

const referenceBasename = (value: string): string =>
    value.split(/[\\/]/).filter(Boolean).pop() || value

const isGeometryLocationValue = (label: string, value: string): boolean => {
    const normalizedLabel = label.toLowerCase()
    if (!/\b(?:bbox|bounding box|coordinates?|layout bbox|region)\b/.test(normalizedLabel)) return false
    return (
        /\b(?:x|y|w|h)=\d+(?:\.\d+)?(?:in)?\b/i.test(value) ||
        /^\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?){2,}$/.test(value.trim())
    )
}

const friendlyReferenceSourceType = (reference: ChatReferenceSourceBriefReference): string => {
    const contentType = String(reference.contentType || '').toLowerCase()
    const filename = `${reference.filename || reference.endpoint || ''}`.toLowerCase()

    if (contentType.includes('application/pdf') || /\.pdf(?:$|[?#])/.test(filename)) return 'PDF source'
    if (
        contentType.includes('text/markdown') ||
        filename.includes('structured_ingest') ||
        /\.structured\.md(?:$|[?#])/.test(filename)
    ) return 'Evidence text'
    if (contentType.includes('presentation') || /\.(?:ppt|pptx)(?:$|[?#])/.test(filename)) return 'Slide deck'
    if (contentType.includes('spreadsheet') || /\.(?:xls|xlsx|csv)(?:$|[?#])/.test(filename)) return 'Spreadsheet'
    if (contentType.includes('wordprocessing') || /\.(?:doc|docx|odt)(?:$|[?#])/.test(filename)) return 'Document'
    if (contentType.includes('image/')) return 'Image source'
    if (contentType.includes('json')) return 'Structured data'
    if (contentType.includes('markdown')) return 'Markdown'
    if (contentType.includes('text/')) return 'Text source'
    return ''
}

const documentTitle = (reference: ChatReferenceSourceBriefReference): string =>
    reference.filename ||
    reference.documentId ||
    (reference.relationship
        ? [
            reference.relationship.source || 'Source',
            reference.relationship.target || 'Target',
        ].filter(Boolean).join(' -> ')
        : '') ||
    (reference.path?.nodes?.length ? reference.path.nodes.join(' -> ') : '') ||
    reference.title ||
    reference.subtitle ||
    'Source document'

const documentSubtitle = (reference: ChatReferenceSourceBriefReference): string =>
    uniqueStrings([
        friendlyReferenceSourceType(reference),
        reference.endpoint ? 'source available' : '',
    ]).slice(0, 2).join(' · ')

const formatCount = (count: number, singular: string, plural = `${singular}s`): string =>
    `${count} ${count === 1 ? singular : plural}`

const kindCountLabels = (references: ChatReferenceSourceBriefReference[]): string[] => {
    const counts = references.reduce(
        (result, reference) => ({
            ...result,
            [reference.kind]: (result[reference.kind] || 0) + 1,
            answer: result.answer + (reference.sourceRole === 'answer_evidence' ? 1 : 0),
        }),
        {
            answer: 0,
            passage: 0,
            relationship: 0,
            path: 0,
            document: 0,
            artifact: 0,
        } as Record<ReviewReferenceKind | 'answer', number>,
    )

    return [
        counts.answer ? formatCount(counts.answer, 'citation') : '',
        counts.passage ? formatCount(counts.passage, 'passage') : '',
        counts.relationship ? formatCount(counts.relationship, 'relation') : '',
        counts.path ? formatCount(counts.path, 'path') : '',
        counts.document + counts.artifact
            ? formatCount(counts.document + counts.artifact, 'document')
            : '',
    ].filter(Boolean)
}

const evidenceScore = (value: string): number => {
    const normalized = normalizeReferenceFacetText(value)
    if (!normalized) return 0
    const highSignal = HIGH_SIGNAL_EVIDENCE_PATTERNS.some(pattern => pattern.test(normalized))
    const structural = STRUCTURAL_EVIDENCE_PATTERNS.some(pattern => pattern.test(normalized))
    if (highSignal) return 100
    if (structural) return -20
    return 10
}

const evidenceLabels = (references: ChatReferenceSourceBriefReference[]): string[] => {
    const counts = new Map<string, number>()
    references.forEach(reference => {
        uniqueStrings(reference.evidenceFamilies || []).forEach(label => {
            const formatted = titleCaseFacet(label)
            counts.set(formatted, (counts.get(formatted) || 0) + 1)
        })
    })

    const entries = Array.from(counts.entries())
    const hasHighSignal = entries.some(([label]) => evidenceScore(label) >= 100)
    return entries
        .filter(([label]) => !hasHighSignal || evidenceScore(label) >= 100)
        .sort((left, right) =>
            evidenceScore(right[0]) - evidenceScore(left[0]) ||
            right[1] - left[1] ||
            left[0].localeCompare(right[0]),
        )
        .slice(0, 5)
        .map(([label, count]) => count > 1 ? `${label} ${count}` : label)
}

const cleanLocationHint = (value: string): string => {
    const text = value.replace(/\s+/g, ' ').trim()
    if (!text) return ''
    if (/^(?:region|bbox|bounding box|coordinates?|layout bbox)\s+\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?){2,}$/i.test(text)) {
        return ''
    }
    if (/^file=/i.test(text)) {
        const fileMatch = text.match(/^file=([^;]+)/i)
        const pageMatch = text.match(/\bpage=([^;]+)/i)
        const slideMatch = text.match(/\bslide=([^;]+)/i)
        const tableMatch = text.match(/\btable=([^;]+)/i)
        return [
            fileMatch ? referenceBasename(fileMatch[1].trim()) : '',
            pageMatch ? `page ${pageMatch[1].trim()}` : '',
            slideMatch ? `slide ${slideMatch[1].trim()}` : '',
            tableMatch ? `table ${tableMatch[1].trim()}` : '',
        ].filter(Boolean).join(' / ')
    }
    return text
}

const compactLocation = (location: ChatReferenceSourceLocation): string => {
    const label = titleCaseFacet(String(location.label || ''))
    const value = String(location.value || '').replace(/\s+/g, ' ').trim()
    if (!label && !value) return ''
    if (isGeometryLocationValue(label, value)) return ''
    if (!value || label.toLowerCase() === value.toLowerCase()) return label || value
    return compactText(`${label} ${value}`, 64)
}

const locationLabels = (references: ChatReferenceSourceBriefReference[]): string[] =>
    uniqueStrings(
        references.flatMap(reference => [
            ...(reference.sourceLocations || []).map(compactLocation),
            ...(reference.locationHints || []).map(cleanLocationHint),
        ]),
    ).slice(0, 6)

const relationshipLead = (reference: ChatReferenceSourceBriefReference): string => {
    const relationship = reference.relationship
    if (!relationship) return ''
    const source = relationship.source || 'Source'
    const target = relationship.target || 'Target'
    const relation = relationship.relation ? titleCaseFacet(relationship.relation) : 'connects to'
    return `${source} ${relation.toLowerCase()} ${target}.`
}

const pathLead = (reference: ChatReferenceSourceBriefReference): string => {
    const nodes = reference.path?.nodes || []
    if (nodes.length > 1) return `${nodes.join(' -> ')}.`
    const relationships = reference.path?.relationships || []
    if (!relationships.length) return ''
    return compactText(
        relationships
            .map(item => [
                item.source || 'Source',
                item.relation ? titleCaseFacet(item.relation).toLowerCase() : 'connects to',
                item.target || 'Target',
            ].join(' '))
            .join('; '),
        220,
    )
}

const cleanBriefBodyText = (value: string): string =>
    cleanDocumentTextForReading(value)
        .replace(/\blocal pipeline\s+[^;)]+;?\s*/gi, '')
        .replace(/\s*\bfile=[^;)]+/gi, '')
        .replace(/\s*;\s*(?:x|y|w|h)=\d+(?:\.\d+)?(?:in)?/gi, '')
        .replace(/\s*;\s*[^;.()]*\bvisibility\s+(?:visible|hidden)\b/gi, '')
        .replace(/\s*;\s*sheet\s+(?:part|relationship)\s+[^;)]+/gi, '')
        .replace(/\s*;\s*(?:source|file|slide|page|sheet|table|chart|visibility|parse_status|relationship_id|image_relationship_id)=[^;)]+/gi, '')
        .replace(/\s*;\s*(?:visibility|parse status)\s*:\s*[^;)]+/gi, '')
        .replace(/\s+\)/g, ')')
        .replace(/\(\s+/g, '(')
        .replace(/\(\s*;\s*/g, '(')
        .replace(/\(\s*\)/g, '')
        .replace(/\s+/g, ' ')
        .trim()

const referenceLead = (reference: ChatReferenceSourceBriefReference): string => {
    const graphLead = relationshipLead(reference) || pathLead(reference)
    const rawBody = cleanBriefBodyText(reference.body || '')
    if (graphLead && /--\[[^\]]+\]--/.test(rawBody)) return compactText(graphLead, 320)

    const body = compactText(rawBody, graphLead ? 180 : 260)
    return compactText([graphLead, body].filter(Boolean).join(' '), 320)
}

export const buildChatReferenceSourceBrief = (
    references: ChatReferenceSourceBriefReference[],
): ChatReferenceSourceBrief | null => {
    if (!references.length) return null

    const bestReference = preferredReviewReference(references) || references[0]
    const title = documentTitle(bestReference)
    const subtitle = documentSubtitle(bestReference)
    const lead = referenceLead(bestReference)

    return {
        title,
        subtitle,
        lead,
        countLabels: kindCountLabels(references),
        evidenceLabels: evidenceLabels(references),
        locationLabels: locationLabels(references),
        referenceCount: references.length,
        bestReferenceId: bestReference.id,
    }
}
