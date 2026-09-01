export type ReviewReferenceKind = 'passage' | 'document' | 'relationship' | 'path' | 'artifact'
export type ReviewReferenceSourceRole =
    | 'answer_evidence'
    | 'retrieved_passage'
    | 'reasoning_path'
    | 'source_document'
    | 'graph_relationship'

export interface ReviewReference {
    id: string
    kind: ReviewReferenceKind
    sourceRole?: ReviewReferenceSourceRole
    title?: string
    subtitle?: string
    body?: string
    score?: number
    endpoint?: string
    contentType?: string
    filename?: string
    terms?: string[]
    evidenceFamilies?: string[]
    matchedTerms?: string[]
    relationship?: {
        source?: string
        relation?: string
        target?: string
    }
    path?: {
        nodes?: string[]
        hops?: number
        relationships?: Array<{
            source?: string
            relation?: string
            target?: string
        }>
    }
}

export const STRUCTURAL_REVIEW_FAMILY_PATTERNS = [
    /\blayout\b/,
    /\bmetadata\b/,
    /\bpresentation\s+structure\b/,
    /\bworkbook\s+structure\b/,
    /\bsource\s+document\b/,
    /\bsource\s+file\b/,
    /\bsection\b/,
]

const STRUCTURAL_REFERENCE_RELATIONS = new Set([
    'appears_in',
    'belongs_to',
    'co_occurs',
    'cooccurs',
    'contains',
    'contains_chunk',
    'contains_document',
    'contains_entity',
    'contains_heading',
    'contains_page',
    'contains_paragraph',
    'contains_section',
    'contains_slide',
    'contains_text',
    'has_body',
    'has_chunk',
    'has_content',
    'has_heading',
    'has_page',
    'has_section',
    'has_slide',
    'has_text',
    'has_title',
    'in_document',
    'in_page',
    'in_section',
    'in_slide',
    'located_in',
    'mentioned_in',
])

const GROUPED_VISUAL_REFERENCE_RELATIONS = new Set([
    'contains_shape_group',
    'groups_visual',
])

const GATE_REFERENCE_RELATIONS = new Set([
    'gate',
    'gates',
])

const GENERIC_REFERENCE_RELATIONS = new Set([
    'answer_evidence',
    'associated_with',
    'connected_to',
    'co_occurs',
    'co_occurs_with',
    'cooccurs',
    'appears_with',
    'mentioned_with',
    'mentions',
    'references_chunk',
    'related',
    'related_text',
    'related_to',
])

const STRONG_REVIEW_FAMILY_PATTERNS = [
    /\bsemantic\b/,
    /\bconnector\b/,
    /\bshape(?:s|\s+group)?\b/,
    /\bgroup(?:ed)?\b/,
    /\bvisual\b/,
    /\bchart\b/,
    /\bform\b/,
    /\bdiagram\b/,
    /\btable\b/,
    /\bstyled\b/,
    /\bnotes?\b/,
    /\bocr\b/,
    /\bcode\b/,
]

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

export const normalizeReferenceFacetText = (value?: string): string =>
    String(value || '').replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase()

const normalizeRelationKey = (value?: string): string =>
    String(value || '')
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '_')
        .replace(/^_+|_+$/g, '')

const isExternalUrlLabel = (value?: string): boolean =>
    /^https?:\/\//i.test(String(value || '').trim())

const queryAsksForLinkedEvidence = (query?: string): boolean =>
    /\b(?:href|link|linked|links|url|uri)\b|\bevidence\s+(?:link|packet|source|url)\b|\bopen\s+evidence\b/i.test(String(query || ''))

const isPdfReference = (reference?: ReviewReference | null): boolean => {
    if (!reference) return false
    const contentType = String(reference.contentType || '').toLowerCase()
    if (contentType) return contentType.includes('application/pdf')
    const filename = `${reference.filename || reference.title || reference.endpoint || ''}`.toLowerCase()
    return /\.pdf(?:$|[?#])/.test(filename)
}

const isOriginalSourceReference = (reference?: ReviewReference | null): boolean => {
    if (!reference?.endpoint) return false
    if (reference.sourceRole === 'source_document') return true
    if (reference.kind === 'document' && isPdfReference(reference)) return true
    return reference.kind === 'artifact' && /source[_-]?document|original/i.test(reference.endpoint)
}

const preferredDocumentReference = <T extends ReviewReference>(references: T[]): T | undefined =>
    references.find(reference => isOriginalSourceReference(reference)) ||
    references.find(reference => reference.kind === 'document' && Boolean(reference.endpoint)) ||
    references.find(reference => reference.kind === 'artifact' && Boolean(reference.endpoint)) ||
    references.find(reference => reference.kind === 'passage' && reference.sourceRole === 'source_document') ||
    references[0]

const isLinkLikeRelation = (relation?: string): boolean => {
    const key = normalizeRelationKey(relation)
    return Boolean(key && (
        key.includes('link') ||
        key.includes('url') ||
        key.includes('uri') ||
        key.includes('href') ||
        key === 'references' ||
        key === 'points_to'
    ))
}

const isBrandingAssetLabel = (value?: string): boolean =>
    /\b(?:asset|badge|brand\s*kit|brand-kit|icon|logo|watermark)\b/i.test(String(value || ''))

const isEmbeddedOfficeArtifactLabel = (value?: string): boolean => {
    const text = String(value || '').trim()
    if (!text) return false
    return (
        /(?:^|[/\\])(?:ppt|xl|word)[/\\](?:charts|drawings|embeddings|media|oleObjects|slides|worksheets|_rels)[/\\]/i.test(text) ||
        /\b(?:chartWorkbook|Microsoft_Excel_Worksheet|oleObject|embedding)\b/i.test(text) ||
        /\.(?:bin|rels|xml)(?:$|[?#])/i.test(text)
    )
}

const isChartAnnotationLabel = (value?: string): boolean =>
    /\b(?:axis|data\s*labels?|legend|trendline)\b/i.test(String(value || ''))

const isDocumentContainerRelationship = (
    relationship?: NonNullable<ReviewReference['relationship']>,
): boolean => {
    if (!relationship) return false
    const relation = relationship.relation
    return (
        isStructuralReferenceRelation(relation) &&
        (isStructuralEndpointLabel(relationship.source) || isStructuralEndpointLabel(relationship.target))
    )
}

const referenceLowSignalPenalty = (reference: ReviewReference): number => {
    const relationships = [
        reference.relationship,
        ...(reference.path?.relationships || []),
    ].filter(Boolean)
    let penalty = 0

    relationships.forEach(relationship => {
        const source = relationship?.source
        const target = relationship?.target
        const external = isExternalUrlLabel(source) || isExternalUrlLabel(target)
        const brandedAsset = isBrandingAssetLabel(source) || isBrandingAssetLabel(target)
        const linkLike = isLinkLikeRelation(relationship?.relation)
        const embeddedArtifact = isEmbeddedOfficeArtifactLabel(source) || isEmbeddedOfficeArtifactLabel(target)
        const chartAnnotation = isChartAnnotationLabel(source) || isChartAnnotationLabel(target)

        if (external && brandedAsset) {
            penalty += 96
        } else if (external && linkLike) {
            penalty += 76
        } else if (external) {
            penalty += 52
        } else if (brandedAsset && linkLike) {
            penalty += 34
        }

        if (embeddedArtifact) {
            penalty += 44
        }
        if (chartAnnotation) {
            penalty += 24
        }
    })

    const familyText = uniqueStrings(reference.evidenceFamilies || [])
        .map(normalizeReferenceFacetText)
        .join(' ')
    if (reference.kind === 'relationship' && /\blink\b/.test(familyText)) {
        penalty += 12
    }

    return penalty
}

const reviewFamilyScore = (reference: ReviewReference): number =>
    uniqueStrings(reference.evidenceFamilies || []).reduce((score, family) => {
        const normalized = normalizeReferenceFacetText(family)
        if (!normalized) return score

        const strongMatches = STRONG_REVIEW_FAMILY_PATTERNS.filter(pattern => pattern.test(normalized)).length
        const structuralMatches = STRUCTURAL_REVIEW_FAMILY_PATTERNS.filter(pattern => pattern.test(normalized)).length
        const linkPenalty = /\blink\b/.test(normalized) ? 10 : 0
        return score + (strongMatches * 16) - (structuralMatches * 7) - linkPenalty
    }, 0)

const isStructuralReferenceRelation = (relation?: string): boolean => {
    const key = normalizeRelationKey(relation)
    if (!key) return false
    return (
        STRUCTURAL_REFERENCE_RELATIONS.has(key) ||
        key.startsWith('contains_') ||
        key.startsWith('has_') ||
        key.startsWith('in_')
    )
}

const isGenericReferenceRelation = (relation?: string): boolean => {
    const key = normalizeRelationKey(relation)
    return Boolean(key && GENERIC_REFERENCE_RELATIONS.has(key))
}

const isSemanticReferenceRelation = (relation?: string): boolean => {
    const key = normalizeRelationKey(relation)
    return Boolean(key && !isStructuralReferenceRelation(key) && !isGenericReferenceRelation(key))
}

const isStructuralEndpointLabel = (value?: string): boolean => {
    const normalized = String(value || '').replace(/\s+/g, ' ').trim().toLowerCase()
    if (!normalized) return true
    if (/\.(?:csv|docx?|html?|md|pdf|pptx?|txt|xlsx?)$/i.test(normalized)) return true
    return /^(?:artifact|body|cell|chunk|column|connector|document|file|heading|layout|line|page|paragraph|row|section|shape|sheet|slide|source|table|text|text box|title)\b/.test(normalized)
}

const referenceRelationshipScore = (reference: ReviewReference): number => {
    const relationships = [
        reference.relationship,
        ...(reference.path?.relationships || []),
    ].filter(Boolean)

    return relationships.reduce((score, relationship) => {
        const relation = relationship?.relation
        const relationKey = normalizeRelationKey(relation)
        const source = relationship?.source
        const target = relationship?.target
        const semantic = isSemanticReferenceRelation(relation)
        const structural = isStructuralReferenceRelation(relation)
        const generic = isGenericReferenceRelation(relation)
        const semanticEndpoints = !isStructuralEndpointLabel(source) && !isStructuralEndpointLabel(target)

        if (GATE_REFERENCE_RELATIONS.has(relationKey) && semanticEndpoints) return score + 68
        if (GATE_REFERENCE_RELATIONS.has(relationKey)) return score + 42
        if (GROUPED_VISUAL_REFERENCE_RELATIONS.has(relationKey)) return score + 48
        if (semantic && semanticEndpoints) return score + 38
        if (semantic) return score + 20
        if (generic) return score - 12
        if (structural && semanticEndpoints) return score - 6
        if (structural) return score - 30
        return score
    }, 0)
}

const ANSWER_FIT_LOW_VALUE_TERMS = new Set([
    'about',
    'does',
    'document',
    'evidence',
    'explain',
    'graph',
    'influence',
    'question',
    'reference',
    'references',
    'relationship',
    'relationships',
    'source',
    'sources',
    'summarize',
    'trace',
])

const queryAnswerTerms = (query?: string): string[] =>
    uniqueStrings(
        String(query || '')
            .split(/[^A-Za-z0-9_-]+/)
            .map(term => term.trim().toLowerCase())
            .filter(term => term.length >= 4 && !ANSWER_FIT_LOW_VALUE_TERMS.has(term)),
    ).slice(0, 12)

const queryAnswerPhrases = (query?: string): string[] => {
    const text = String(query || '').replace(/\s+/g, ' ').trim()
    if (!text) return []

    const capitalizedPhrases = text.match(/\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)+\b/g) || []
    return uniqueStrings(capitalizedPhrases)
        .map(phrase => phrase.trim().toLowerCase())
        .filter(phrase => {
            const terms = phrase.split(/\s+/)
            return terms.length >= 2 && terms.some(term => !ANSWER_FIT_LOW_VALUE_TERMS.has(term))
        })
        .slice(0, 8)
}

const relationshipList = (reference: ReviewReference): Array<NonNullable<ReviewReference['relationship']>> =>
    [
        reference.relationship,
        ...(reference.path?.relationships || []),
    ].filter((relationship): relationship is NonNullable<ReviewReference['relationship']> => Boolean(relationship))

const textMatchesPhrase = (text: string, phrase: string): boolean => {
    if (!text || !phrase) return false
    if (text.includes(phrase)) return true
    const terms = phrase.split(/\s+/).filter(term => term.length >= 3)
    return terms.length > 0 && terms.every(term => text.includes(term))
}

const referenceAnswerSearchText = (reference: ReviewReference): string =>
    uniqueStrings([
        reference.title,
        reference.subtitle,
        reference.body,
        reference.filename,
        ...(reference.terms || []),
        ...(reference.matchedTerms || []),
        ...(reference.evidenceFamilies || []),
        reference.relationship?.source,
        reference.relationship?.relation,
        reference.relationship?.target,
        ...(reference.path?.nodes || []),
        ...relationshipList(reference).flatMap(relationship => [
            relationship.source,
            relationship.relation,
            relationship.target,
        ]),
    ]).join(' ').toLowerCase()

const referenceAnswerNodeText = (reference: ReviewReference): string =>
    uniqueStrings([
        reference.relationship?.source,
        reference.relationship?.target,
        ...(reference.path?.nodes || []),
        ...relationshipList(reference).flatMap(relationship => [
            relationship.source,
            relationship.target,
        ]),
    ]).join(' ').toLowerCase()

const referenceHasExternalUrlEvidence = (reference: ReviewReference): boolean =>
    [
        reference.title,
        reference.subtitle,
        reference.body,
        reference.relationship?.source,
        reference.relationship?.target,
        ...(reference.path?.nodes || []),
        ...relationshipList(reference).flatMap(relationship => [
            relationship.source,
            relationship.target,
        ]),
    ].some(isExternalUrlLabel)

export const referenceAnswerFitScore = (reference: ReviewReference, query?: string): number => {
    const phrases = queryAnswerPhrases(query)
    const terms = queryAnswerTerms(query)
    if (!phrases.length && !terms.length) return 0

    const haystack = referenceAnswerSearchText(reference)
    const nodeText = referenceAnswerNodeText(reference)
    const phraseCoverage = phrases.filter(phrase => textMatchesPhrase(haystack, phrase)).length
    const nodePhraseCoverage = phrases.filter(phrase => textMatchesPhrase(nodeText, phrase)).length
    const termCoverage = terms.filter(term => haystack.includes(term)).length
    const relationships = relationshipList(reference)
    const explicitRelationshipCount = relationships.filter(relationship =>
        isSemanticReferenceRelation(relationship.relation),
    ).length
    const familyText = uniqueStrings(reference.evidenceFamilies || [])
        .map(normalizeReferenceFacetText)
        .join(' ')
    const tableEvidence = /\btable\b/.test(familyText) ? 8 : 0
    const semanticEvidence = /\bsemantic\b/.test(familyText) ? 4 : 0
    const queryText = normalizeReferenceFacetText(query)
    const relationshipKeys = relationships.map(relationship => normalizeRelationKey(relationship.relation))
    const asksForVisualSupport = /\b(?:connector|connectors|group|grouped|shape|shapes|support|supports|supported|supporting|visual|visuals)\b/.test(queryText)
    const asksForGate = /\b(?:approval|gate|gates|gated|launch|workflow)\b/.test(queryText)
    const groupedVisualRelationship =
        relationshipKeys.some(key => GROUPED_VISUAL_REFERENCE_RELATIONS.has(key)) ||
        /\b(?:contains shape group|groups visual|grouped visual|shape group)\b/.test(haystack) ||
        /\b(?:shape group|grouped visual)\b/.test(familyText)
    const directGateRelationship = relationshipKeys.some(key => GATE_REFERENCE_RELATIONS.has(key))
    const gateLabelMention = !directGateRelationship && /\bgates?\b/.test(haystack)
    const directRelationshipQuestion = /\b(?:cite|citation|connect|connects|connection|direct|relationship|relationships|source|sources|support|supports|through|via|visual|visuals)\b/.test(queryText)
    const groupedVisualEvidence = asksForVisualSupport && groupedVisualRelationship ? 40 : 0
    const linkedEvidencePriority =
        queryAsksForLinkedEvidence(query) && referenceHasExternalUrlEvidence(reference)
            ? reference.kind === 'path'
                ? 128
                : 116
            : 0
    const gateEvidence = asksForGate
        ? directGateRelationship
            ? 52
            : gateLabelMention && reference.kind === 'relationship'
                ? 12
                : 0
        : 0
    const directRelationshipEvidence =
        directRelationshipQuestion &&
        reference.kind === 'relationship' &&
        (directGateRelationship || groupedVisualRelationship)
            ? 34
            : 0
    const directGatePriority =
        asksForGate && directGateRelationship && reference.kind === 'relationship' ? 38 : 0
    const indirectGateLabelPenalty =
        asksForGate && reference.kind === 'path' && gateLabelMention && !directGateRelationship ? 18 : 0
    const fullPhraseCoverage = phrases.length > 0 && phraseCoverage === phrases.length ? 18 : 0
    const fullNodeCoverage = phrases.length > 0 && nodePhraseCoverage === phrases.length ? 28 : 0
    const pathDepth =
        reference.kind === 'path'
            ? Math.min(18, Math.max(0, (reference.path?.nodes?.length || 0) - 2) * 5 + relationships.length * 2)
            : 0
    const genericOnlyPenalty =
        reference.kind === 'path' &&
        relationships.length > 0 &&
        explicitRelationshipCount === 0 &&
        phrases.length > 0 &&
        nodePhraseCoverage < phrases.length
            ? 10
            : 0

    return (
        phraseCoverage * 7 +
        nodePhraseCoverage * 12 +
        termCoverage * 1.5 +
        explicitRelationshipCount * 10 +
        tableEvidence +
        semanticEvidence +
        groupedVisualEvidence +
        linkedEvidencePriority +
        gateEvidence +
        directRelationshipEvidence +
        directGatePriority +
        fullPhraseCoverage +
        fullNodeCoverage +
        pathDepth -
        genericOnlyPenalty -
        indirectGateLabelPenalty
    )
}

export const isTechnicalReviewReference = (reference: ReviewReference): boolean => {
    if (reference.kind === 'artifact' && !isOriginalSourceReference(reference)) return true
    if (referenceLowSignalPenalty(reference) >= 44) return true
    if (
        [
            reference.title,
            reference.subtitle,
            reference.body,
            reference.endpoint,
        ].some(value => isEmbeddedOfficeArtifactLabel(value))
    ) {
        return true
    }

    const relationships = [
        reference.relationship,
        ...(reference.path?.relationships || []),
    ].filter(Boolean)

    return relationships.some(relationship => {
        const source = relationship?.source
        const target = relationship?.target
        const embeddedArtifact = isEmbeddedOfficeArtifactLabel(source) || isEmbeddedOfficeArtifactLabel(target)
        const externalLink = (isExternalUrlLabel(source) || isExternalUrlLabel(target)) && isLinkLikeRelation(relationship?.relation)
        return embeddedArtifact || externalLink || isDocumentContainerRelationship(relationship)
    })
}

export const referenceReviewScore = (reference: ReviewReference): number => {
    const roleScore: Record<ReviewReferenceSourceRole, number> = {
        answer_evidence: 80,
        reasoning_path: 42,
        graph_relationship: 34,
        retrieved_passage: 16,
        source_document: -18,
    }
    const kindScore: Record<ReviewReferenceKind, number> = {
        relationship: 34,
        path: 24,
        passage: 8,
        document: -24,
        artifact: -30,
    }
    const matchedSignal = Math.min(18, ((reference.matchedTerms || []).length + (reference.terms || []).length) * 2)
    const sourceAvailability = reference.endpoint ? 4 : 0
    const documentPenalty = isOriginalSourceReference(reference) ? -18 : 0

    return (
        (reference.sourceRole ? roleScore[reference.sourceRole] : 0) +
        kindScore[reference.kind] +
        referenceRelationshipScore(reference) +
        reviewFamilyScore(reference) +
        matchedSignal +
        sourceAvailability +
        documentPenalty -
        referenceLowSignalPenalty(reference) +
        (reference.score || 0)
    )
}

export const preferredReviewReference = <T extends ReviewReference>(references: T[]): T | undefined => {
    if (!references.length) return undefined

    const [best] = references
        .map((reference, index) => ({
            reference,
            index,
            score: referenceReviewScore(reference),
        }))
        .sort((left, right) =>
            right.score - left.score ||
            (right.reference.score || 0) - (left.reference.score || 0) ||
            left.index - right.index,
        )

    return best?.reference || preferredDocumentReference(references)
}
