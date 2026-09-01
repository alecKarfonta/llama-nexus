export interface CleanGraphRAGAssistantAnswerOptions {
    sourceLabel?: string
    documentFirst?: boolean
}

const compactWhitespace = (value: string): string =>
    value
        .replace(/[ \t]+\n/g, '\n')
        .replace(/\n{3,}/g, '\n\n')
        .trim()

const stripRawScoreText = (value: string): string =>
    value
        .replace(/^\s*(?:[-*]\s*)?(?:\*\*)?(?:path\s+|graph\s+|ranking\s+)?score(?:(?:\*\*)?\s*[:=]|[:=]\s*(?:\*\*)?)\s*\d+(?:\.\d+)?\s*$/gim, '')
        .replace(/\s*\([^)]*\bscore\s*[:=]\s*\d+(?:\.\d+)?[^)]*\)/gi, '')
        .replace(/\s*,?\s*with\s+(?:a\s+)?(?:lower|higher|path|graph|ranking)?\s*score\s+of\s+\d+(?:\.\d+)?/gi, '')
        .replace(/\s*,?\s*(?:path|graph|ranking)\s+score\s+of\s+\d+(?:\.\d+)?/gi, '')
        .replace(/\b\d+(?:\.\d+)?%\s+score\b/gi, '')
        .replace(/\bscore\s*[:=]\s*\d+(?:\.\d+)?\b/gi, '')
        .replace(/\bhighest-scored\b/gi, 'strongest')
        .replace(/\bhigher-scored\b/gi, 'stronger')
        .replace(/\blower-scored\b/gi, 'weaker')
        .replace(/\bconfidence\s+scores?\b/gi, 'confidence')
        .replace(/\ba\s+(higher|lower|stronger|weaker)\s+confidence\b/gi, '$1 confidence')
        .replace(/\s+\bwith\s+(?:and\s*)?\./gi, '.')
        .replace(/\s+\band\s*\./gi, '.')
        .replace(/,\s*,/g, ',')
        .replace(/\s+\./g, '.')
        .replace(/\(\s*\)/g, '')

const formatGraphToken = (value: string): string => {
    const normalized = value.trim().toLowerCase().replace(/[\s-]+/g, '_')
    const labels: Record<string, string> = {
        answer_evidence: 'answer evidence',
        contains_chart: 'contains chart',
        co_occurs_with: 'co-occurs with',
        contains_image: 'contains image',
        contains_link: 'contains link',
        contains_shape_group: 'contains shape group',
        contains_table: 'contains table',
        graph_path_reasoning: 'graph reasoning',
        groups_visual: 'groups visual',
        has_section: 'has section',
        has_slide: 'has slide',
        links_to: 'links to',
        references_chunk: 'supports',
        retrieved_passage: 'retrieved passage',
        source_document: 'source document',
        source_document_context: 'source context',
    }
    return labels[normalized] || value.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase()
}

const humanizeGraphTokens = (value: string): string =>
    value
        .replace(/--\[\s*([a-z][a-z0-9_-]*)\s*\]--?>/gi, (_match, relation: string) =>
            `--[${formatGraphToken(relation)}]-->`,
        )
        .replace(
            /\b(?:answer_evidence|co_occurs_with|contains_chart|contains_link|contains_shape_group|contains_table|graph_path_reasoning|groups_visual|has_section|has_slide|links_to|references_chunk|retrieved_passage|source_document|source_document_context)\b/gi,
            match => formatGraphToken(match),
        )
        .replace(
            /\bcontains_image\b/gi,
            match => formatGraphToken(match),
        )

const stripGraphRAGInlineCode = (value: string): string =>
    value
        .replace(/`(Graph path:)`/gi, '$1')
        .replace(/`([^`\n]*(?:https?:\/\/|--\[|\.pdf|\.pptx?|\.docx?|\.xlsx?|\.csv|\.tsv|\.md|\.txt)[^`\n]*)`/gi, '$1')
        .replace(
            /`(answer evidence|co-occurs with|contains chart|contains image|contains link|contains shape group|contains table|graph reasoning|groups visual|has section|has slide|links to|retrieved passage|source context|source document|supports)`/gi,
            '$1',
        )
        .replace(/`(\[[a-z][a-z0-9 _-]{2,40}\])`/gi, '$1')
        .replace(/`([A-Z][A-Za-z0-9]*(?:[ .:/-]+[A-Z0-9][A-Za-z0-9]*){0,8})`/g, '$1')

const softenGraphPathSyntax = (value: string): string =>
    value
        .replace(/\s*--\[\s*([^\]]+?)\s*\]--?>\s*/g, (_match, relation: string) =>
            ` -> ${formatGraphToken(relation)} -> `,
        )
        .replace(/\s*(?:→|->)\s*\[\s*([^\]]+?)\s*\]\s*(?:→|->)\s*/g, (_match, relation: string) =>
            ` -> ${formatGraphToken(relation)} -> `,
        )
        .replace(/\s+->\s+/g, ' -> ')

const normalizeGraphAnswerHeadings = (value: string): string =>
    value.replace(
        /^(\s*(?:\*\*)?(?:Direct Connection|Direct Relationship|Evidence and Provenance|Graph Path|Graph path|Provenance and Evidence|Provenance|Relationship|Slide Context|Strongest Direct Relationship|Strongest Multi-Hop Path(?:\s*\([^)]+\))?|Summary|Supporting Visuals|Visual Grouping)(?:\*\*)?)\s+(?=(?:[A-Z]|source document\b))/gm,
        '$1: ',
    )

const cleanDocumentFirstGraphBullets = (value: string): string =>
    value
        .replace(
            /^\s*(?:[-*]\s*)?(?:\*\*)?(?:Direct Graph Relationship|Direct Relationship|Relevant Graph Relationship|Supporting Graph Path|Supporting Graph Relationship|Graph Path)(?:\*\*)?\s*:\s*.*(?:\n|$)/gim,
            '',
        )
        .replace(/^\s*(?:Source Evidence|Evidence)\s*:\s*$/gim, 'Source:')

export const cleanGraphRAGAssistantAnswer = (
    content: string,
    options: CleanGraphRAGAssistantAnswerOptions = {},
): string => {
    const sourceLabel = options.sourceLabel?.trim()
    const sourceText = sourceLabel || 'the selected document'

    const cleaned = content
        .replace(
            /^\s*Based on the GraphRAG pipeline trace for\s+[^,\n]+,?\s*/i,
            `From ${sourceText}, `,
        )
        .replace(/^Pipeline Trace:\s*.*$/gim, '')
        .replace(/\bGraphRAG pipeline trace\b/gi, 'document evidence')
        .replace(/\bpipeline trace\b/gi, 'document evidence')
        .replace(/\bgraph trace\b/gi, 'graph evidence')
        .replace(/(Graph path:\s*.+?)\s+Evidence:/i, '$1\n\nEvidence:')
        .replace(/((?:\*\*)?Graph path:(?:\*\*)?\s*Direct(?: relationship)?(?:\s*\([^)]*\))?)\s+(?=(?:The|This|It)\b)/i, '$1.\n\n')
        .replace(/\bTrace Evidence\s*&\s*Provenance\b/gi, 'Evidence & Provenance')
        .replace(/\bTrace Evidence\b/gi, 'Evidence')
        .replace(/\brun\s+graphrag_ingest_[a-z0-9_-]+\b/gi, 'the document extraction')
        .replace(/\bgraph_path_reasoning\b/gi, 'graph reasoning')
        .replace(
            /^(\s*Source Provenance:\s*)([^,\n]+)(.*)$/gim,
            (_match, prefix: string, _filename: string, suffix: string) =>
                `${prefix}${sourceText}${suffix}`,
        )

    const documentModeCleaned = options.documentFirst
        ? cleanDocumentFirstGraphBullets(cleaned)
        : cleaned

    return compactWhitespace(
        normalizeGraphAnswerHeadings(
            softenGraphPathSyntax(
                stripGraphRAGInlineCode(
                    humanizeGraphTokens(stripRawScoreText(documentModeCleaned)),
                ),
            ),
        ),
    )
}
