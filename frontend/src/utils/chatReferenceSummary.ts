export interface ChatReferenceSummaryReference {
    kind?: string
    sourceRole?: string
    title?: string
    filename?: string
    documentId?: string
    runId?: string
    evidenceFamilies?: string[]
}

export interface ChatReferenceContextSummaryInput {
    references?: ChatReferenceSummaryReference[]
    attachedDocumentCount?: number
    passageCount?: number
    fallbackPassageCount?: number
    relationshipCount?: number
    pathCount?: number
}

export const formatChatReferenceCount = (
    count: number,
    singular: string,
    plural = `${singular}s`,
): string => `${count} ${count === 1 ? singular : plural}`

const uniqueSourceCount = (references: ChatReferenceSummaryReference[]): number => {
    const keys = new Set<string>()

    references.forEach(reference => {
        const key =
            reference.filename ||
            reference.documentId ||
            reference.runId ||
            (reference.kind === 'document' ? reference.title : '')
        if (key) keys.add(key)
    })

    return keys.size
}

const evidenceLabel = (value: string): string => {
    const normalized = value.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase()
    if (!normalized) return ''
    if (/\btable\b/.test(normalized)) return 'table evidence'
    if (/\bchart\b/.test(normalized)) return 'chart evidence'
    if (/\b(?:diagram|connector|shape)\b/.test(normalized)) return 'diagram evidence'
    if (/\bform\b/.test(normalized)) return 'form evidence'
    if (/\b(?:ocr|image)\b/.test(normalized)) return 'OCR evidence'
    if (/\bcode\b/.test(normalized)) return 'code evidence'
    if (/\bnotes?\b/.test(normalized)) return 'notes evidence'
    return ''
}

const primaryEvidenceLabel = (references: ChatReferenceSummaryReference[]): string => {
    const counts = new Map<string, number>()

    references.forEach(reference => {
        (reference.evidenceFamilies || []).forEach(family => {
            const label = evidenceLabel(family)
            if (!label) return
            counts.set(label, (counts.get(label) || 0) + 1)
        })
    })

    return Array.from(counts.entries())
        .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
        .map(([label]) => label)[0] || ''
}

export const buildChatReferenceContextSummary = ({
    references = [],
    attachedDocumentCount = 0,
    passageCount = 0,
    fallbackPassageCount = 0,
    relationshipCount,
    pathCount,
}: ChatReferenceContextSummaryInput): string => {
    const answerCitationCount = references.filter(
        reference => reference.sourceRole === 'answer_evidence',
    ).length
    const sourceCount = uniqueSourceCount(references)
    const resolvedPassageCount =
        passageCount || fallbackPassageCount || references.filter(reference => reference.kind === 'passage').length
    const resolvedRelationshipCount =
        relationshipCount ?? references.filter(reference => reference.kind === 'relationship').length
    const resolvedPathCount =
        pathCount ?? references.filter(reference => reference.kind === 'path').length

    const labels = [
        attachedDocumentCount > 1
            ? formatChatReferenceCount(attachedDocumentCount, 'document ready', 'documents ready')
            : attachedDocumentCount === 1
                ? 'Document ready'
                : references.length
                    ? 'Evidence ready'
                    : '',
        answerCitationCount
            ? formatChatReferenceCount(answerCitationCount, 'citation')
            : sourceCount
                ? formatChatReferenceCount(sourceCount, 'source')
                : '',
        answerCitationCount && sourceCount ? formatChatReferenceCount(sourceCount, 'source') : '',
        resolvedPathCount ? formatChatReferenceCount(resolvedPathCount, 'graph path') : '',
        !resolvedPathCount && resolvedRelationshipCount
            ? formatChatReferenceCount(resolvedRelationshipCount, 'relation')
            : '',
        !resolvedPathCount && !resolvedRelationshipCount && resolvedPassageCount
            ? formatChatReferenceCount(resolvedPassageCount, 'passage')
            : '',
        primaryEvidenceLabel(references),
    ].filter(Boolean)

    return labels.slice(0, 5).join(' · ')
}
