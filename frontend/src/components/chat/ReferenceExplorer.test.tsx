import { describe, expect, it } from 'vitest'
import {
    buildReferenceDocumentPreviewSummary,
    buildPdfLocationHighlights,
    buildReferencePathEvidenceLabels,
    cleanRequestedEvidencePreview,
    buildReferencePdfHighlightTerms,
    buildReferenceTraySubtitle,
    buildReferenceLayoutMap,
    buildReferencePathPreviewText,
    chatReferenceDisplayDedupeKey,
    dedupeReferencePreviewAgainstSelectedEvidence,
    dedupeChatReferencesForDisplay,
    displayPathRelationships,
    diversifyReferenceExplorerDrawerReferences,
    layoutMapCalloutPosition,
    layoutMapPinPosition,
    layoutMapViewport,
    layoutMapViewportBox,
    mergePdfHighlightRects,
    requestedEvidenceTrayPreviewText,
    isLowSignalGraphRAGSummaryReference,
    isSourceNavigationReference,
    pdfLocationHighlightFromValue,
    pdfViewerRenderScale,
    referenceBriefTitle,
    referencePreviewText,
    referencePreviewTextForQuery,
    relationshipConcreteEvidenceStatements,
    referencePathQueryCoverageScore,
    referencePathQueryOrderScore,
    referencePathSemanticStrengthScore,
    resolveReferenceExplorerSelectionReference,
    referenceLocationLabels,
    selectReferencePrimaryExternalUrl,
    selectReferenceExplorerActiveReference,
    selectInlineReferenceExplorerTrayReferences,
    selectReferenceExplorerOpeningReference,
    selectFocusedReferenceReviewReferences,
    selectReferencePdfViewerReference,
    selectReferenceExplorerTrayReferences,
    shouldShowReferenceExplorerDrawerSearch,
    sortReferenceExplorerDrawerReferences,
    type ChatReferenceItem,
} from './ReferenceExplorer'
import {
    isTechnicalReviewReference,
    preferredReviewReference,
    referenceAnswerFitScore,
    referenceReviewScore,
    type ReviewReference,
} from '@/utils/chatReferenceReviewScoring'
import {
    pageNumberFromReference,
    pageNumbersFromReferences,
    type ReferencePageTarget,
} from '@/utils/referencePdfPage'
import {
    DEFAULT_REFERENCE_DRAWER_LIMIT,
    selectReferenceDrawerRows,
} from '@/utils/chatReferenceDrawerList'
import { displayPathNodes } from '@/utils/graphragPathDisplay'

const makeReference = (
    id: string,
    relationship: NonNullable<ReviewReference['relationship']>,
    evidenceFamilies: string[],
): ReviewReference => ({
    id,
    kind: 'relationship',
    sourceRole: 'answer_evidence',
    title: `${relationship.source} -> ${relationship.target}`,
    evidenceFamilies,
    relationship,
})

describe('ReferenceExplorer reference ranking', () => {
    it('removes document preview text that repeats the selected evidence lead', () => {
        expect(
            dedupeReferencePreviewAgainstSelectedEvidence(
                'The approval path is Product Revenue to Regional Demand to Risk Map.',
                'The approval path is Product Revenue to Regional Demand to Risk Map.',
            ),
        ).toBe('')

        expect(
            dedupeReferencePreviewAgainstSelectedEvidence(
                'The surrounding section names Ada Lovelace as the Forecast Risk reviewer.',
                'The approval path is Product Revenue to Regional Demand to Risk Map.',
            ),
        ).toContain('Ada Lovelace')
    })

    it('prefers subject-matter evidence over external branding links', () => {
        const semantic = makeReference(
            'semantic-revenue',
            {
                source: 'Product Revenue',
                relation: 'drives',
                target: 'Regional Demand',
            },
            ['semantic'],
        )
        const logoLink = makeReference(
            'logo-link',
            {
                source: 'Llama Nexus Logo',
                relation: 'links_to',
                target: 'https://example.com/brand-kit',
            },
            ['link'],
        )

        expect(referenceReviewScore(semantic)).toBeGreaterThan(referenceReviewScore(logoLink))
        expect(preferredReviewReference([logoLink, semantic])?.id).toBe('semantic-revenue')
    })

    it('surfaces explicit URL evidence when the question asks for linked evidence', () => {
        const query = 'What linked evidence belongs to the Launch Evidence Cluster? Cite the visual group and the URL.'
        const groupedVisual: ChatReferenceItem = {
            id: 'grouped-visual',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Launch Evidence Cluster -> Evidence Snapshot',
            body: 'Launch Evidence Cluster groups Evidence Snapshot on slide 1.',
            evidenceFamilies: ['shape', 'layout'],
            relationship: {
                source: 'Launch Evidence Cluster',
                relation: 'groups_visual',
                target: 'Evidence Snapshot',
            },
        }
        const evidenceUrl: ChatReferenceItem = {
            id: 'evidence-url',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Evidence Snapshot -> https://example.com/evidence/approval-gate',
            body: 'Launch Evidence Cluster contains the Evidence Snapshot link https://example.com/evidence/approval-gate.',
            evidenceFamilies: ['link', 'image', 'layout'],
            terms: ['Launch Evidence Cluster', 'Evidence Snapshot'],
            relationship: {
                source: 'Evidence Snapshot',
                relation: 'links_to',
                target: 'https://example.com/evidence/approval-gate',
            },
        }
        const brandUrl: ChatReferenceItem = {
            id: 'brand-url',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Llama Nexus Logo -> https://example.com/brand-kit',
            body: 'Logo asset link.',
            evidenceFamilies: ['link'],
            relationship: {
                source: 'Llama Nexus Logo',
                relation: 'links_to',
                target: 'https://example.com/brand-kit',
            },
        }

        expect(referenceAnswerFitScore(evidenceUrl, query)).toBeGreaterThan(
            referenceAnswerFitScore(groupedVisual, query),
        )
        expect(
            selectReferenceExplorerOpeningReference(
                [groupedVisual, evidenceUrl, brandUrl],
                query,
            { answerEvidenceReferences: [groupedVisual, evidenceUrl, brandUrl] },
            )?.id,
        ).toBe('evidence-url')
    })

    it('selects the primary external URL from relationship and provenance metadata', () => {
        expect(selectReferencePrimaryExternalUrl({
            id: 'relationship-url',
            kind: 'relationship',
            title: 'Evidence Snapshot link',
            relationship: {
                source: 'Evidence Snapshot',
                relation: 'links_to',
                target: 'https://example.com/evidence/approval-gate',
            },
        })).toBe('https://example.com/evidence/approval-gate')

        expect(selectReferencePrimaryExternalUrl({
            id: 'location-url',
            kind: 'passage',
            title: 'Evidence text',
            sourceLocations: [
                { label: 'Slide', value: '1' },
                { label: 'URI', value: 'https://example.com/source-packet' },
            ],
        })).toBe('https://example.com/source-packet')
    })

    it('collapses duplicate linked evidence URLs in the inline source tray', () => {
        const query = 'Trace the evidence link to https://example.com/evidence/approval-gate and cite the relationship.'
        const primaryLink: ChatReferenceItem = {
            id: 'primary-link',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Evidence Snapshot -> https://example.com/evidence/approval-gate',
            body: 'Evidence Snapshot links to https://example.com/evidence/approval-gate on Slide 1.',
            filename: 'launch-evidence.pptx',
            evidenceFamilies: ['link', 'layout'],
            relationship: {
                source: 'Evidence Snapshot',
                relation: 'links_to',
                target: 'https://example.com/evidence/approval-gate',
            },
        }
        const duplicateUrlReference: ChatReferenceItem = {
            id: 'duplicate-url',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'https://example.com/evidence/approval-gate',
            body: 'The approval gate URL is cited by the same slide evidence.',
            filename: 'launch-evidence.pptx',
            evidenceFamilies: ['link'],
            relationship: {
                source: 'https://example.com/evidence/approval-gate',
                relation: 'cited_by',
                target: 'launch-evidence.pptx',
            },
        }
        const supportingGroup: ChatReferenceItem = {
            id: 'supporting-group',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Launch Evidence Cluster -> Evidence Snapshot',
            body: 'Launch Evidence Cluster groups the Evidence Snapshot visual.',
            filename: 'launch-evidence.pptx',
            evidenceFamilies: ['layout'],
            relationship: {
                source: 'Launch Evidence Cluster',
                relation: 'groups_visual',
                target: 'Evidence Snapshot',
            },
        }

        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: [primaryLink, duplicateUrlReference, supportingGroup],
            visibleReferences: [primaryLink, duplicateUrlReference, supportingGroup],
            answerEvidenceReferences: [primaryLink, duplicateUrlReference, supportingGroup],
            query,
        })

        expect(trayReferences.map(reference => reference.id)).toContain('primary-link')
        expect(trayReferences.filter(reference =>
            selectReferencePrimaryExternalUrl(reference) === 'https://example.com/evidence/approval-gate',
        )).toHaveLength(1)
    })

    it('orders requested evidence family cards by the user question in the inline source tray', () => {
        const query = 'List every extracted evidence family: charts, tables, form fields, styled text, comments, and speaker notes.'
        const chartEvidence: ChatReferenceItem = {
            id: 'requested-chart',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Charts',
            body: 'Revenue Plan chart with Q2 18 and Q3 31.',
            filename: 'launch-approval.pptx',
            evidenceFamilies: ['requested_evidence_family', 'charts'],
        }
        const tableEvidence: ChatReferenceItem = {
            id: 'requested-table',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Tables',
            body: 'Finance Review Board approved the launch in approval.csv.',
            filename: 'launch-approval.pptx',
            evidenceFamilies: ['requested_evidence_family', 'tables'],
        }
        const formEvidence: ChatReferenceItem = {
            id: 'requested-form',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Form Fields',
            body: 'Approval required checkbox is checked.',
            filename: 'launch-approval.pptx',
            evidenceFamilies: ['requested_evidence_family', 'form_fields'],
        }
        const styledEvidence: ChatReferenceItem = {
            id: 'requested-styled',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Styled Text',
            body: 'Approval required is bold and highlighted.',
            filename: 'launch-approval.pptx',
            evidenceFamilies: ['requested_evidence_family', 'styled_text'],
        }
        const commentEvidence: ChatReferenceItem = {
            id: 'requested-comment',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Comments',
            body: 'Maya Reviewer requested approval evidence.',
            filename: 'launch-approval.pptx',
            evidenceFamilies: ['requested_evidence_family', 'comments'],
        }
        const speakerNotesEvidence: ChatReferenceItem = {
            id: 'requested-speaker-notes',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Speaker Notes',
            body: 'Speaker note: cite the Finance Review Board approval evidence.',
            filename: 'launch-approval.pptx',
            evidenceFamilies: ['requested_evidence_family', 'speaker_notes'],
        }
        const directCitation: ChatReferenceItem = {
            id: 'direct-citation',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Finance Review Board -> Approved',
            body: 'Finance Review Board decision Approved.',
            filename: 'launch-approval.pptx',
            evidenceFamilies: ['semantic'],
            relationship: {
                source: 'Finance Review Board',
                relation: 'decision',
                target: 'Approved',
            },
        }

        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: [
                commentEvidence,
                speakerNotesEvidence,
                directCitation,
                tableEvidence,
                formEvidence,
                styledEvidence,
                chartEvidence,
            ],
            visibleReferences: [
                commentEvidence,
                speakerNotesEvidence,
                directCitation,
                tableEvidence,
                formEvidence,
                styledEvidence,
                chartEvidence,
            ],
            answerEvidenceReferences: [
                commentEvidence,
                speakerNotesEvidence,
                directCitation,
                tableEvidence,
                formEvidence,
                styledEvidence,
                chartEvidence,
            ],
            query,
        })

        expect(trayReferences.map(reference => reference.id)).toEqual([
            'requested-chart',
            'requested-table',
            'requested-form',
            'requested-styled',
            'requested-comment',
            'requested-speaker-notes',
        ])
    })

    it('leads graph path questions with the path covering every requested node', () => {
        const query =
            'How does Approval connect to Finance Review Board and Risk Map in this PDF? Cite the strongest source evidence and graph path.'
        const directHop: ChatReferenceItem = {
            id: 'direct-hop',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Approval to Finance Review Board',
            body: 'Approval mentions Finance Review Board.',
            filename: 'llama-nexus-rich.pdf',
            score: 0.95,
            evidenceFamilies: ['semantic'],
            path: {
                nodes: ['Approval', 'Finance Review Board'],
                hops: 1,
                relationships: [
                    {
                        source: 'Approval',
                        relation: 'mentions',
                        target: 'Finance Review Board',
                        evidence: 'Finance Review Board owns approval evidence for the launch.',
                    },
                ],
            },
        }
        const fullPath: ChatReferenceItem = {
            id: 'full-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Approval to Finance Review Board to Risk Map',
            body: 'Approval mentions Finance Review Board; Finance Review Board mentions Risk Map.',
            filename: 'llama-nexus-rich.pdf',
            score: 0.62,
            evidenceFamilies: ['semantic'],
            path: {
                nodes: ['Approval', 'Finance Review Board', 'Risk Map'],
                hops: 2,
                relationships: [
                    {
                        source: 'Approval',
                        relation: 'mentions',
                        target: 'Finance Review Board',
                        evidence: 'Finance Review Board owns approval evidence for the launch.',
                    },
                    {
                        source: 'Finance Review Board',
                        relation: 'mentions',
                        target: 'Risk Map',
                        evidence: 'Risk Map highlights regional blockers before release.',
                    },
                ],
            },
        }
        const documentCitation: ChatReferenceItem = {
            id: 'document-citation',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Finance Review Board to Risk Map',
            body: 'Finance Review Board mentions Risk Map.',
            filename: 'llama-nexus-rich.pdf',
            relationship: {
                source: 'Finance Review Board',
                relation: 'mentions',
                target: 'Risk Map',
            },
        }

        expect(referencePathQueryCoverageScore(fullPath, query)).toBeGreaterThan(
            referencePathQueryCoverageScore(directHop, query),
        )

        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: [directHop, documentCitation, fullPath],
            visibleReferences: [directHop, documentCitation, fullPath],
            answerEvidenceReferences: [directHop, documentCitation, fullPath],
            query,
        })

        expect(trayReferences[0]?.id).toBe('full-path')
        expect(selectInlineReferenceExplorerTrayReferences(trayReferences, query, 2)[0]?.id).toBe('full-path')
    })

    it('keeps requested evidence families first when the question also asks for relationship paths', () => {
        const query =
            'Show the chart, table, form field, styled text, connector, speaker notes, comment, media, and attachment evidence extracted from this PowerPoint. Cite the strongest source locations and relationship paths.'
        const requestedChart: ChatReferenceItem = {
            id: 'requested-chart',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Charts',
            body: 'Chart Revenue Plan: Approval with Revenue impact axis metadata.',
            filename: 'llama-nexus-rich.pptx',
            evidenceFamilies: ['requested_evidence_family', 'charts'],
        }
        const requestedTable: ChatReferenceItem = {
            id: 'requested-table',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Tables',
            body: 'Table row cites Finance Review Board and approval.csv.',
            filename: 'llama-nexus-rich.pptx',
            evidenceFamilies: ['requested_evidence_family', 'tables'],
        }
        const requestedForm: ChatReferenceItem = {
            id: 'requested-form',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Form Fields',
            body: 'Approval required checkbox is checked.',
            filename: 'llama-nexus-rich.pptx',
            evidenceFamilies: ['requested_evidence_family', 'form_fields'],
        }
        const requestedStyled: ChatReferenceItem = {
            id: 'requested-styled',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Styled Text',
            body: 'Approval required is bold and highlighted.',
            filename: 'llama-nexus-rich.pptx',
            evidenceFamilies: ['requested_evidence_family', 'styled_text'],
        }
        const requestedConnector: ChatReferenceItem = {
            id: 'requested-connector',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Connectors',
            body: 'Approval required before launch workflow gates Launch workflow.',
            filename: 'llama-nexus-rich.pptx',
            evidenceFamilies: ['requested_evidence_family', 'connectors'],
        }
        const chartPointQ2: ChatReferenceItem = {
            id: 'chart-point-q2',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Revenue Plan to Q2: 18',
            body: 'Revenue Plan shows category value Q2: 18.',
            filename: 'llama-nexus-rich.pptx',
            evidenceFamilies: ['chart_points'],
            relationship: {
                source: 'Revenue Plan',
                relation: 'shows_category_value',
                target: 'Q2: 18',
            },
        }
        const chartPointQ3: ChatReferenceItem = {
            id: 'chart-point-q3',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Revenue Plan to Q3: 31',
            body: 'Revenue Plan shows category value Q3: 31.',
            filename: 'llama-nexus-rich.pptx',
            evidenceFamilies: ['chart_points'],
            relationship: {
                source: 'Revenue Plan',
                relation: 'shows_category_value',
                target: 'Q3: 31',
            },
        }
        const directDecision: ChatReferenceItem = {
            id: 'direct-decision',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Finance Review Board to Approved',
            body: 'Finance Review Board decision Approved.',
            filename: 'llama-nexus-rich.pptx',
            evidenceFamilies: ['semantic'],
            relationship: {
                source: 'Finance Review Board',
                relation: 'decision',
                target: 'Approved',
            },
        }

        const references = [
            chartPointQ2,
            chartPointQ3,
            requestedChart,
            directDecision,
            requestedConnector,
            requestedStyled,
            requestedForm,
            requestedTable,
        ]
        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: references,
            visibleReferences: references,
            answerEvidenceReferences: references,
            query,
        })

        expect(trayReferences.map(reference => reference.id).slice(0, 5)).toEqual([
            'requested-chart',
            'requested-table',
            'requested-form',
            'requested-styled',
            'requested-connector',
        ])
    })

    it('promotes requested evidence families even when only low-level citations are answer evidence', () => {
        const query = [
            '**Charts**: Slide 1: Launch Approval Packet | Chart Revenue Plan: Approval; semantic axis: Value axis; Revenue impact.',
            '**Tables**: Slide 1: Launch Approval Packet | approval.csv | embedded data.',
            'Show the chart, table, form field, styled text, connector, speaker notes, comment, media, and attachment evidence extracted from this PowerPoint. Cite the strongest source locations and relationship paths.',
        ].join('\n')
        const requestedChart: ChatReferenceItem = {
            id: 'requested-chart',
            kind: 'passage',
            sourceRole: 'source_document',
            title: 'Charts',
            body: 'Requested evidence · Slide 1: Launch Approval Packet | Chart Revenue Plan: Approval.',
            filename: 'llama-nexus-rich.pptx',
            evidenceFamilies: ['requested_evidence_family', 'charts'],
        }
        const requestedTable: ChatReferenceItem = {
            id: 'requested-table',
            kind: 'passage',
            sourceRole: 'source_document',
            title: 'Tables',
            body: 'Requested evidence · Slide 1: Launch Approval Packet | approval.csv | embedded data.',
            filename: 'llama-nexus-rich.pptx',
            evidenceFamilies: ['requested_evidence_family', 'tables'],
        }
        const requestedForm: ChatReferenceItem = {
            id: 'requested-form',
            kind: 'passage',
            sourceRole: 'source_document',
            title: 'Form Fields',
            body: 'Requested evidence · Approval required checkbox is checked.',
            filename: 'llama-nexus-rich.pptx',
            evidenceFamilies: ['requested_evidence_family', 'form_fields'],
        }
        const requestedStyled: ChatReferenceItem = {
            id: 'requested-styled',
            kind: 'passage',
            sourceRole: 'source_document',
            title: 'Styled Text',
            body: 'Requested evidence · Approval required is bold and highlighted.',
            filename: 'llama-nexus-rich.pptx',
            evidenceFamilies: ['requested_evidence_family', 'styled_text'],
        }
        const requestedConnector: ChatReferenceItem = {
            id: 'requested-connector',
            kind: 'passage',
            sourceRole: 'source_document',
            title: 'Connectors',
            body: 'Requested evidence · Approval required gates Launch workflow.',
            filename: 'llama-nexus-rich.pptx',
            evidenceFamilies: ['requested_evidence_family', 'connectors'],
        }
        const chartPointQ2: ChatReferenceItem = {
            id: 'chart-point-q2',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Revenue Plan to Q2: 18',
            body: 'Revenue Plan shows category value Q2: 18.',
            filename: 'llama-nexus-rich.pptx',
            evidenceFamilies: ['chart_points'],
            relationship: {
                source: 'Revenue Plan',
                relation: 'shows_category_value',
                target: 'Q2: 18',
            },
        }
        const chartPointQ3: ChatReferenceItem = {
            id: 'chart-point-q3',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Revenue Plan to Q3: 31',
            body: 'Revenue Plan shows category value Q3: 31.',
            filename: 'llama-nexus-rich.pptx',
            evidenceFamilies: ['chart_points'],
            relationship: {
                source: 'Revenue Plan',
                relation: 'shows_category_value',
                target: 'Q3: 31',
            },
        }

        const familyReferences = [
            requestedChart,
            requestedTable,
            requestedForm,
            requestedStyled,
            requestedConnector,
        ]
        const answerReferences = [chartPointQ2, chartPointQ3]
        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: [...answerReferences, ...familyReferences],
            visibleReferences: answerReferences,
            answerEvidenceReferences: answerReferences,
            query,
        })

        expect(trayReferences.map(reference => reference.id).slice(0, 5)).toEqual([
            'requested-chart',
            'requested-table',
            'requested-form',
            'requested-styled',
            'requested-connector',
        ])
    })

    it('keeps bridge path evidence ahead of requested evidence cards for generic citation wording', () => {
        const query =
            'Trace the path from Approval required before launch workflow to Launch workflow through Approval Dependency Cluster. Show each hop label and cite the slide evidence.'
        const path: ChatReferenceItem = {
            id: 'approval-bridge-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Approval required before launch workflow -> Approval Dependency Cluster -> Launch workflow',
            body: 'Approval required before launch workflow --groups visual--> Approval Dependency Cluster --groups visual--> Launch workflow.',
            filename: 'launch-approval.pptx',
            evidenceFamilies: ['layout', 'semantic'],
            path: {
                nodes: [
                    'Approval required before launch workflow',
                    'Approval Dependency Cluster',
                    'Launch workflow',
                ],
                hops: 2,
                relationships: [
                    {
                        source: 'Approval required before launch workflow',
                        relation: 'groups_visual',
                        target: 'Approval Dependency Cluster',
                    },
                    {
                        source: 'Approval Dependency Cluster',
                        relation: 'groups_visual',
                        target: 'Launch workflow',
                    },
                ],
            },
        }
        const requestedEvidence: ChatReferenceItem = {
            id: 'requested-speaker-notes',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Speaker Notes',
            body: 'Speaker note: cite the Finance Review Board approval evidence.',
            filename: 'launch-approval.pptx',
            evidenceFamilies: ['requested_evidence_family', 'speaker_notes'],
        }
        const directRelationship: ChatReferenceItem = {
            id: 'approval-direct-gate',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Approval required before launch workflow -> Launch workflow',
            body: 'Approval required before launch workflow gates Launch workflow.',
            filename: 'launch-approval.pptx',
            evidenceFamilies: ['semantic'],
            relationship: {
                source: 'Approval required before launch workflow',
                relation: 'gates',
                target: 'Launch workflow',
            },
        }

        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: [requestedEvidence, directRelationship, path],
            visibleReferences: [requestedEvidence, directRelationship, path],
            answerEvidenceReferences: [requestedEvidence, directRelationship, path],
            query,
        })

        expect(trayReferences[0].id).toBe('approval-bridge-path')
        expect(trayReferences.map(reference => reference.id)).not.toContain('requested-speaker-notes')
        expect(selectReferenceExplorerOpeningReference(
            [requestedEvidence, directRelationship, path],
            query,
            {
                trayReferences,
                answerEvidenceReferences: [requestedEvidence, directRelationship, path],
            },
        )?.id).toBe('approval-bridge-path')
    })

    it('opens generated PDF summaries on semantic citations instead of page scaffolding', () => {
        const query =
            'Use the recent GraphRAG pipeline trace for graphrag-live-pdf-evidence-ui-clean-1784816080.pdf to summarize the key evidence, entities, relationships, and any meaningful graph paths. Keep the answer bounded and complete: return exactly 4 bullets, under 180 words total, named Entities, Top relationships, Best path, and Source proof.'
        const pageScaffold: ChatReferenceItem = {
            id: 'pdf-page-scaffold',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'page 1',
            body: 'Chunk preserves direct subject-relation-object evidence for graph search and path answers.',
            filename: 'graphrag-live-pdf-evidence-ui-clean-1784816080.pdf',
            contentType: 'application/pdf',
            evidenceFamilies: ['semantic_relation'],
            relationship: {
                source: 'graphrag-live-pdf-evidence-ui-clean-1784816080.pdf',
                relation: 'semantic_relation',
                target: 'page 1',
            },
        }
        const sourcePdf: ChatReferenceItem = {
            id: 'pdf-source-document',
            kind: 'document',
            sourceRole: 'source_document',
            title: 'graphrag-live-pdf-evidence-ui-clean-1784816080.pdf',
            filename: 'graphrag-live-pdf-evidence-ui-clean-1784816080.pdf',
            contentType: 'application/pdf',
            endpoint: '/api/v1/graphrag/ingest/pipeline/run-pdf/artifacts/source_document',
        }
        const revenueRelationship: ChatReferenceItem = {
            id: 'revenue-demand',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Product Revenue -> Regional Demand',
            body: 'Product Revenue drives Regional Demand on page 1.',
            filename: 'graphrag-live-pdf-evidence-ui-clean-1784816080.pdf',
            contentType: 'application/pdf',
            evidenceFamilies: ['semantic'],
            relationship: {
                source: 'Product Revenue',
                relation: 'drives',
                target: 'Regional Demand',
            },
            sourceLocations: [
                { label: 'Page', value: '1', category: 'page' },
            ],
        }
        const riskRelationship: ChatReferenceItem = {
            id: 'demand-risk',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Regional Demand -> Risk Map',
            body: 'Regional Demand references Risk Map for launch gating.',
            filename: 'graphrag-live-pdf-evidence-ui-clean-1784816080.pdf',
            contentType: 'application/pdf',
            evidenceFamilies: ['semantic'],
            relationship: {
                source: 'Regional Demand',
                relation: 'references',
                target: 'Risk Map',
            },
            sourceLocations: [
                { label: 'Page', value: '1', category: 'page' },
            ],
        }

        const references = [
            pageScaffold,
            sourcePdf,
            revenueRelationship,
            riskRelationship,
        ]
        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: references,
            visibleReferences: references,
            answerEvidenceReferences: [pageScaffold, revenueRelationship, riskRelationship],
            query,
        })

        expect(selectReferenceExplorerOpeningReference(references, query, {
            trayReferences,
            answerEvidenceReferences: [pageScaffold, revenueRelationship, riskRelationship],
        })?.id).toBe('revenue-demand')
        expect(sortReferenceExplorerDrawerReferences(references, query)[0].id).toBe('revenue-demand')
        expect(isLowSignalGraphRAGSummaryReference(pageScaffold, query)).toBe(true)
        expect(isLowSignalGraphRAGSummaryReference(revenueRelationship, query)).toBe(false)
        expect(sortReferenceExplorerDrawerReferences(references, query).at(-1)?.id).toBe('pdf-page-scaffold')
        expect(trayReferences.map(reference => reference.id)).toEqual([
            'revenue-demand',
            'demand-risk',
        ])
    })

    it('uses requested evidence text as the source detail excerpt', () => {
        const reference: ChatReferenceItem = {
            id: 'requested-speaker-notes',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Speaker Notes',
            body: 'Launch Approval Packet --contains speaker notes--> Speaker note: cite the Finance Review Board approval evidence.; evidence=Speaker note: cite the Finance Review Board approval evidence.; provenance=slide 1: Launch Approval Packet',
            filename: 'launch-approval.pptx',
            evidenceFamilies: ['requested_evidence_family', 'speaker_notes'],
            sourceLocations: [
                { label: 'File', value: 'launch-approval.pptx', category: 'document' },
                { label: 'Slide', value: '1', category: 'slide' },
                { label: 'Slide title', value: 'Launch Approval Packet', category: 'slide' },
            ],
        }

        const preview = buildReferenceDocumentPreviewSummary(reference, {
            fallbackPreview: reference.body,
            evidenceLabels: ['Speaker Notes'],
        })

        expect(preview.title).toBe('Speaker Notes')
        expect(preview.subtitle).toContain('launch-approval.pptx')
        expect(preview.subtitle).toContain('Slide 1: Launch Approval Packet')
        expect(preview.excerpt).toBe('Speaker note: cite the Finance Review Board approval evidence.')
        expect(preview.chips).toContain('Speaker Notes')
    })

    it('shows graph context before generic passages for linked visual evidence', () => {
        const query = 'Trace evidence link'
        const directCitation: ChatReferenceItem = {
            id: 'direct-citation',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'https://example.com/evidence/approval-gate',
            body: 'https://example.com/evidence/approval-gate is supported by source evidence.',
            filename: 'graphrag-grouped-visual-link.pptx',
            evidenceFamilies: ['link'],
            relationship: {
                source: 'Evidence Snapshot',
                relation: 'links_to',
                target: 'https://example.com/evidence/approval-gate',
            },
        }
        const genericPassage: ChatReferenceItem = {
            id: 'generic-passage',
            kind: 'passage',
            sourceRole: 'retrieved_passage',
            title: 'graphrag-grouped-visual-link.pptx',
            body: 'Scope:.png (image) Evidence types: shape, image, layout. Link refs: https://example.com/evidence/approval-gate',
            filename: 'graphrag-grouped-visual-link.pptx',
            chunkIndex: 1,
            score: 1,
        }
        const visualPath: ChatReferenceItem = {
            id: 'visual-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Launch Evidence Cluster -> Evidence Snapshot -> https://example.com/evidence/approval-gate',
            body: 'A grouped visual contains the Evidence Snapshot, which links to the approval gate URL.',
            filename: 'graphrag-grouped-visual-link.pptx',
            evidenceFamilies: ['shape_group', 'layout', 'link'],
            path: {
                nodes: [
                    'Launch Evidence Cluster',
                    'Evidence Snapshot',
                    'https://example.com/evidence/approval-gate',
                ],
                hops: 2,
                relationships: [
                    {
                        source: 'Launch Evidence Cluster',
                        relation: 'groups_visual',
                        target: 'Evidence Snapshot',
                    },
                    {
                        source: 'Evidence Snapshot',
                        relation: 'links_to',
                        target: 'https://example.com/evidence/approval-gate',
                    },
                ],
            },
        }

        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: [directCitation, genericPassage, visualPath],
            visibleReferences: [directCitation, genericPassage, visualPath],
            answerEvidenceReferences: [directCitation, genericPassage],
            query,
        })

        expect(trayReferences.map(reference => reference.id)).toEqual([
            'direct-citation',
            'visual-path',
        ])
    })

    it('keeps reference previews readable by removing raw extraction metadata', () => {
        const reference: ChatReferenceItem = {
            id: 'link-preview',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Evidence Snapshot -> https://example.com/evidence/approval-gate',
            body: 'Evidence Snapshot -> https://example.com/evidence/approval-gate slide 1: Launch Evidence; layout top_left x=1.20in y=1.30in w=2.00in h=1.00in; form field Evidence Snapshot; link uri https://example.com/evidence/approval-gate kind click_action',
            filename: 'launch-evidence.pptx',
            evidenceFamilies: ['link', 'layout'],
            relationship: {
                source: 'Evidence Snapshot',
                relation: 'links_to',
                target: 'https://example.com/evidence/approval-gate',
            },
        }

        const preview = buildReferenceDocumentPreviewSummary(reference, {
            fallbackPreview: reference.body,
        })

        expect(preview.excerpt).toContain('Evidence Snapshot')
        expect(preview.excerpt).not.toMatch(/form field|link uri|kind click|x=1\.20in/i)
    })

    it('hides generated source-document envelopes in PDF previews', () => {
        const reference: ChatReferenceItem = {
            id: 'pdf-source-document',
            kind: 'document',
            sourceRole: 'source_document',
            title: 'direct-brief.pdf',
            body: 'document - Filename: source document - Content type: application/pdf - Extraction method: pdf_blocks ## Extracted Structure Product Revenue drives Regional Demand on page 1.; layout_x_inches=1.0; layout_y_inches=0.79',
            filename: 'direct-brief.pdf',
            contentType: 'application/pdf',
        }

        const preview = buildReferenceDocumentPreviewSummary(reference, {
            fallbackPreview: reference.body,
            sourceTypeLabel: 'PDF source',
        })

        expect(preview.title).toBe('direct-brief.pdf')
        expect(preview.excerpt).toContain('Product Revenue drives Regional Demand on page 1')
        expect(preview.excerpt).not.toMatch(/Filename|Content type|Extraction method|layout_x/i)
    })

    it('centers PDF chunk previews on specific query anchors instead of prompt metadata', () => {
        const reference: ChatReferenceItem = {
            id: 'pdf-risk-map-chunk',
            kind: 'passage',
            sourceRole: 'retrieved_passage',
            title: 'llama-nexus-rich.pdf',
            body:
                '[paragraph] page 1 block 3: Finance Review Board owns approval evidence for the launch. [paragraph] page 1 block 4: Launch workflow connects to revenue impact through approval gates. [paragraph] page 1 block 5: Risk Map highlights regional blockers before release. [paragraph] page 1 block 6: Question target: cite Approval required, Finance Review Board, Risk Map, and gates.',
            filename: 'llama-nexus-rich.pdf',
            contentType: 'application/pdf',
            sourceLocations: [
                { label: 'Page', value: '1', category: 'page' },
                { label: 'Block', value: '1', category: 'page' },
            ],
        }

        const preview = buildReferenceDocumentPreviewSummary(reference, {
            documentText: reference.body,
            query: 'What does the Risk Map say in this PDF?',
        })

        expect(preview.excerpt).toContain('Risk Map highlights regional blockers before release')
        expect(preview.excerpt).not.toContain('Question target')
    })

    it('suppresses residual layout-only PDF container previews', () => {
        const reference: ChatReferenceItem = {
            id: 'pdf-source-document-layout-residue',
            kind: 'document',
            sourceRole: 'source_document',
            title: 'direct-brief.pdf',
            body: 'document -0.791.907.267.453.925; layou.',
            filename: 'direct-brief.pdf',
            contentType: 'application/pdf',
        }

        const preview = buildReferenceDocumentPreviewSummary(reference, {
            fallbackPreview: reference.body,
            sourceTypeLabel: 'PDF source',
        })

        expect(preview.title).toBe('direct-brief.pdf')
        expect(preview.excerpt).toBe('')
    })

    it('does not show structured artifact files as source-document chips', () => {
        const reference: ChatReferenceItem = {
            id: 'pdf-source-document-chip',
            kind: 'document',
            sourceRole: 'source_document',
            title: 'direct-brief.pdf',
            filename: 'direct-brief.pdf',
            contentType: 'application/pdf',
            sourceLocations: [
                { label: 'File', value: 'direct-brief.structured.md', category: 'document' },
                { label: 'Page', value: '1', category: 'page' },
            ],
        }

        expect(referenceLocationLabels(reference)).toEqual(['Page 1'])
    })

    it('does not show half-truncated slide title chips in reference cards', () => {
        const reference: ChatReferenceItem = {
            id: 'truncated-slide-chip',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Charts',
            filename: 'llama-nexus-rich.pptx',
            locationHints: ['Slide 1', 'Laun...'],
            evidenceFamilies: ['requested_evidence_family', 'charts'],
        }

        expect(referenceLocationLabels(reference)).toEqual(['Slide 1'])
        expect(referenceLocationLabels(reference).join(' ')).not.toContain('Laun')
    })

    it('summarizes packed table evidence instead of showing raw graph serialization', () => {
        const reference: ChatReferenceItem = {
            id: 'table-evidence-preview',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Tables',
            body: 'table: 1; row: 1; slide: Revenue Architecture; Revenue Architecture --contains table- -> R1C1: Product; evidence=Product; provenance=slide 1: Revenue Architecture; Table 1; Cell R1C1; Bold; location slide 1 Table 1 cell 1.1',
            filename: 'graphrag-ux-rich-e2e.pptx',
            evidenceFamilies: ['tables'],
            sourceLocations: [
                { label: 'Slide', value: '1', category: 'slide' },
                { label: 'Slide title', value: 'Revenue Architecture', category: 'slide' },
                { label: 'Table', value: '1', category: 'table' },
                { label: 'Row', value: '1', category: 'table' },
            ],
        }

        const preview = buildReferenceDocumentPreviewSummary(reference, {
            fallbackPreview: reference.body,
            evidenceLabels: ['Tables'],
        })

        expect(preview.excerpt).toBe('Table 1 row 1 Cell R1C1 · Bold text: Product')
        expect(preview.excerpt).not.toMatch(/--contains|evidence=|provenance=|location slide/i)
        expect(referenceLocationLabels(reference)).toContain('Revenue Architecture')
        expect(referenceLocationLabels(reference)).not.toContain('Slide Title Revenue Architecture')
    })

    it('promotes evidence relationship text over coordinate-only citation titles', () => {
        const reference: ChatReferenceItem = {
            id: 'requested-coordinate-title',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'row=1; row=2; row=3',
            body: 'Approval Gate gates Launch Workflow. Launch requires approval before release; row=2; row=3 Approval Gate Gates Launch Workflow.',
            filename: 'launch-evidence.csv',
            evidenceFamilies: ['semantic', 'table', 'row=1; row=2; row=3'],
            sourceLocations: [
                { label: 'Location', value: 'row=1; row=2; row=3', category: 'table' },
                { label: 'Row', value: '2', category: 'table' },
            ],
        }

        const summary = buildReferenceDocumentPreviewSummary(reference, {
            fallbackPreview: reference.body,
            evidenceLabels: undefined,
        })

        expect(referenceBriefTitle({
            ...reference,
            body: 'Source: Approval Gate; Relationship: gates; Target: Launch Workflow; Evidence: Launch requires approval before release.',
        })).toBe('Approval Gate gates Launch Workflow')
        expect(summary.excerpt).toContain('Launch requires approval before release')
        expect(summary.excerpt).not.toContain('row=2')
        expect(summary.chips).toContain('Semantic')
        expect(summary.chips).toContain('Table')
        expect(referenceLocationLabels(reference)).toContain('Row 2')
        expect(referenceLocationLabels(reference).join(' ')).not.toContain('row=1')
        expect(summary.chips.join(' ')).not.toContain('row=1')
    })

    it('cleans malformed scope fragments from retrieved passage previews', () => {
        const reference: ChatReferenceItem = {
            id: 'table-passage-preview',
            kind: 'passage',
            sourceRole: 'retrieved_passage',
            title: 'graphrag-ux-rich-e2e.pptx',
            body: 'Chunk 14 · 100% match Scope:); Slide 1: Revenue Architecture table 1 (table); table 1; cell R1C1; style bold',
            filename: 'graphrag-ux-rich-e2e.pptx',
            chunkIndex: 14,
            score: 1,
        }

        const preview = buildReferenceDocumentPreviewSummary(reference, {
            fallbackPreview: reference.body,
        })

        expect(preview.excerpt).toContain('Table 1')
        expect(preview.excerpt).toContain('Cell R1C1')
        expect(preview.excerpt).toContain('Bold')
        expect(preview.excerpt).not.toContain('Scope')
        expect(preview.excerpt).not.toContain('100% match')
        expect(preview.excerpt).not.toContain('Table 1 (table); Table 1')
        expect(preview.excerpt).not.toContain('Table 1; Table 1')
        expect(preview.excerpt).not.toContain(');')
    })

    it('keeps embedded Office artifacts and chart annotations behind semantic relationships', () => {
        const semantic = makeReference(
            'semantic-topic',
            {
                source: 'Product Revenue',
                relation: 'drives',
                target: 'Regional Demand',
            },
            ['semantic'],
        )
        const embeddedWorkbook = makeReference(
            'embedded-workbook',
            {
                source: 'Revenue',
                relation: 'references',
                target: 'ppt/embeddings/chartWorkbook.xlsx',
            },
            ['chart'],
        )
        const trendline = makeReference(
            'trendline',
            {
                source: 'Revenue',
                relation: 'explains',
                target: 'Revenue linear trendline',
            },
            ['chart'],
        )

        expect(referenceReviewScore(semantic)).toBeGreaterThan(referenceReviewScore(embeddedWorkbook))
        expect(referenceReviewScore(semantic)).toBeGreaterThan(referenceReviewScore(trendline))
        expect(preferredReviewReference([embeddedWorkbook, trendline, semantic])?.id).toBe('semantic-topic')
    })

    it('prefers relationship facts over high-scoring document container references', () => {
        const semantic = makeReference(
            'semantic-relationship',
            {
                source: 'Product Revenue',
                relation: 'drives',
                target: 'Regional Demand',
            },
            ['semantic', 'connector'],
        )
        semantic.score = 0.68

        const documentContainer = makeReference(
            'document-container',
            {
                source: 'graphrag-complex-office-source.pptx',
                relation: 'contains_chart',
                target: 'Regional Revenue Forecast',
            },
            ['chart', 'presentation_structure'],
        )
        documentContainer.score = 0.99

        expect(referenceReviewScore(semantic)).toBeGreaterThan(referenceReviewScore(documentContainer))
        expect(preferredReviewReference([documentContainer, semantic])?.id).toBe('semantic-relationship')
    })

    it('prefers a complete document path over a shorter co-occurrence path for entity-rich questions', () => {
        const query = 'How does Product Revenue influence Regional Demand, Risk Map, and Launch Priority?'
        const shorterPath: ReviewReference = {
            id: 'shorter-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Product Revenue -> Risk Map -> Regional Demand',
            score: 0.61,
            evidenceFamilies: ['semantic', 'layout'],
            path: {
                nodes: ['Product Revenue', 'Risk Map', 'Regional Demand'],
                hops: 2,
                relationships: [
                    { source: 'Product Revenue', relation: 'co_occurs_with', target: 'Risk Map' },
                    { source: 'Risk Map', relation: 'co_occurs_with', target: 'Regional Demand' },
                ],
            },
        }
        const completePath: ReviewReference = {
            id: 'complete-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Product Revenue -> Regional Demand -> Risk Map -> Launch Priority',
            score: 0.54,
            evidenceFamilies: ['semantic', 'layout'],
            path: {
                nodes: ['Product Revenue', 'Regional Demand', 'Risk Map', 'Launch Priority'],
                hops: 3,
                relationships: [
                    { source: 'Product Revenue', relation: 'co_occurs_with', target: 'Regional Demand' },
                    { source: 'Regional Demand', relation: 'co_occurs_with', target: 'Risk Map' },
                    { source: 'Risk Map', relation: 'co_occurs_with', target: 'Launch Priority' },
                ],
            },
        }

        expect(referenceAnswerFitScore(completePath, query)).toBeGreaterThan(
            referenceAnswerFitScore(shorterPath, query),
        )
    })

    it('prefers grouped visual gate evidence over chart fragments for source questions', () => {
        const query = 'What is the approval gate relationship and what grouped visual supports it?'
        const groupedVisual: ReviewReference = {
            id: 'grouped-visual-gate',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Approval Dependency Cluster -> Approval required before launch workflow.',
            body: 'Grouped PowerPoint visual with a gates connector into the Launch workflow.',
            evidenceFamilies: ['shape_group', 'semantic', 'connector'],
            terms: ['Approval Dependency Cluster', 'Launch workflow', 'gates'],
            relationship: {
                source: 'Approval Dependency Cluster',
                relation: 'Groups Visual',
                target: 'Approval required before launch workflow.',
            },
        }
        const chartFragment: ReviewReference = {
            id: 'chart-fragment',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Approval -> 31',
            body: 'Chart value extracted from a PowerPoint series.',
            evidenceFamilies: ['chart', 'semantic'],
            relationship: {
                source: 'Approval',
                relation: 'Q3 Value',
                target: '31',
            },
        }

        expect(referenceAnswerFitScore(groupedVisual, query)).toBeGreaterThan(
            referenceAnswerFitScore(chartFragment, query),
        )
        expect(referenceReviewScore(groupedVisual)).toBeGreaterThan(referenceReviewScore(chartFragment))
        expect(preferredReviewReference([chartFragment, groupedVisual])?.id).toBe('grouped-visual-gate')
    })

    it('prefers the direct gates edge over a broader path when the question asks for the relationship', () => {
        const query = 'Explain how Approval required before launch workflow connects to Launch workflow through gates and cite the source.'
        const directGate: ReviewReference = {
            id: 'direct-gate',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Approval required before launch workflow. -> Launch workflow',
            body: 'Shape connector on Slide 1: Launch Approval Packet.',
            evidenceFamilies: ['connector', 'semantic', 'shape'],
            relationship: {
                source: 'Approval required before launch workflow.',
                relation: 'gates',
                target: 'Launch workflow',
            },
        }
        const broaderPath: ReviewReference = {
            id: 'broader-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Approval required before launch workflow. -> Approval Dependency Cluster -> Launch workflow',
            evidenceFamilies: ['semantic', 'layout'],
            path: {
                nodes: [
                    'Approval required before launch workflow.',
                    'Approval Dependency Cluster',
                    'Launch workflow',
                ],
                hops: 2,
                relationships: [
                    {
                        source: 'Approval required before launch workflow.',
                        relation: 'groups_visual',
                        target: 'Approval Dependency Cluster',
                    },
                    {
                        source: 'Approval Dependency Cluster',
                        relation: 'groups_visual',
                        target: 'Launch workflow',
                    },
                ],
            },
        }

        expect(referenceAnswerFitScore(directGate, query)).toBeGreaterThan(
            referenceAnswerFitScore(broaderPath, query),
        )
    })

    it('classifies internal artifacts and document containers as technical drill-down references', () => {
        const semantic = makeReference(
            'semantic-relationship',
            {
                source: 'Product Revenue',
                relation: 'drives',
                target: 'Regional Demand',
            },
            ['semantic', 'connector'],
        )
        const embeddedWorkbook = makeReference(
            'embedded-workbook',
            {
                source: 'Regional Revenue Forecast',
                relation: 'references',
                target: 'ppt/embeddings/chartWorkbook.xlsx',
            },
            ['chart'],
        )
        const documentContainer = makeReference(
            'document-container',
            {
                source: 'graphrag-complex-office-source.pptx',
                relation: 'contains_chart',
                target: 'Regional Revenue Forecast',
            },
            ['chart', 'presentation_structure'],
        )

        expect(isTechnicalReviewReference(semantic)).toBe(false)
        expect(isTechnicalReviewReference(embeddedWorkbook)).toBe(true)
        expect(isTechnicalReviewReference(documentContainer)).toBe(true)
    })
})

const makePdfReference = (overrides: Partial<ReferencePageTarget>): ReferencePageTarget => ({
    title: 'Annual report citation',
    ...overrides,
})

describe('ReferenceExplorer PDF page targeting', () => {
    it('uses a selected renderable PDF reference directly', () => {
        const pdfReference: ChatReferenceItem = {
            id: 'annual-report-source',
            kind: 'document',
            title: 'annual-report.pdf',
            filename: 'annual-report.pdf',
            contentType: 'application/pdf',
            endpoint: '/api/v1/graphrag/artifacts/annual-report.pdf',
        }

        expect(selectReferencePdfViewerReference(pdfReference, [])?.id).toBe('annual-report-source')
    })

    it('inherits a renderable PDF from the selected evidence source group', () => {
        const relationshipReference: ChatReferenceItem = {
            id: 'risk-relationship',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Risk Map -> Launch Priority',
            filename: 'annual-report.pdf',
            relationship: {
                source: 'Risk Map',
                relation: 'gates',
                target: 'Launch Priority',
            },
            sourceLocations: [
                { label: 'Page', value: '2', category: 'page' },
            ],
        }
        const pdfSource: ChatReferenceItem = {
            id: 'annual-report-source',
            kind: 'document',
            title: 'annual-report.pdf',
            filename: 'annual-report.pdf',
            contentType: 'application/pdf',
            endpoint: '/api/v1/graphrag/artifacts/annual-report.pdf',
        }

        expect(selectReferencePdfViewerReference(relationshipReference, [pdfSource])?.id).toBe('annual-report-source')
    })

    it('does not choose a PDF-looking citation when no renderable endpoint exists', () => {
        const citationOnly: ChatReferenceItem = {
            id: 'citation-only',
            kind: 'relationship',
            title: 'Risk Map -> Launch Priority',
            filename: 'annual-report.pdf',
            relationship: {
                source: 'Risk Map',
                relation: 'gates',
                target: 'Launch Priority',
            },
        }

        expect(selectReferencePdfViewerReference(citationOnly, [])).toBeNull()
    })

    it('keeps PDF canvas highlights focused on the selected path and query', () => {
        const selectedPath: ChatReferenceItem = {
            id: 'selected-path',
            kind: 'path',
            title: 'Product Revenue -> Regional Demand -> Risk Map',
            filename: 'annual-report.pdf',
            matchedTerms: ['pdf', 'risk'],
            path: {
                nodes: ['Product Revenue', 'Regional Demand', 'Risk Map'],
                hops: 2,
                relationships: [
                    {
                        source: 'Product Revenue',
                        relation: 'drives',
                        target: 'Regional Demand',
                    },
                    {
                        source: 'Regional Demand',
                        relation: 'references',
                        target: 'Risk Map',
                    },
                ],
            },
        }
        const terms = buildReferencePdfHighlightTerms(
            selectedPath,
            'How does Product Revenue connect to Risk Map in this PDF?',
        )

        expect(terms).toEqual(expect.arrayContaining([
            'Product Revenue',
            'Regional Demand',
            'Risk Map',
        ]))
        expect(terms).not.toContain('risk')
        expect(terms).not.toContain('Ada Lovelace')
        expect(terms).not.toContain('Forecast Risk')
    })

    it('uses cited PDF relationship evidence instead of broad endpoint terms for highlights', () => {
        const selectedRelationship: ChatReferenceItem = {
            id: 'approval-finance-pdf',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Approval to Finance Review Board',
            filename: 'llama-nexus-rich.pdf',
            contentType: 'application/pdf',
            body:
                'Approval --[co_occurs_with]--> Finance Review Board\npage 1 block 3: Finance Review Board owns approval evidence for the launch.',
            relationship: {
                source: 'Approval',
                relation: 'related text',
                target: 'Finance Review Board',
            },
        }

        const terms = buildReferencePdfHighlightTerms(
            selectedRelationship,
            'How does Approval connect to Finance Review Board and Risk Map?',
        )

        expect(terms).toContain('Finance Review Board owns approval evidence for the launch.')
        expect(terms).not.toContain('Approval')
        expect(terms).not.toContain('Finance Review Board')
        expect(terms).not.toContain('Risk Map')
    })

    it('promotes concrete relationship evidence over endpoint-only drawer labels', () => {
        const selectedRelationship: ChatReferenceItem = {
            id: 'approval-finance-pdf',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Approval to Finance Review Board',
            filename: 'llama-nexus-rich.pdf',
            contentType: 'application/pdf',
            body:
                'Approval --[co_occurs_with]--> Finance Review Board Finance Review Board owns approval evidence for the launch. Question target: cite Approval required, Finance Review Board, Risk Map, and gates.',
            relationship: {
                source: 'Approval',
                relation: 'related text',
                target: 'Finance Review Board',
            },
        }

        const statements = relationshipConcreteEvidenceStatements(
            selectedRelationship,
            'How does Approval connect to Finance Review Board and Risk Map?',
        )

        expect(statements[0]).toBe('Finance Review Board owns approval evidence for the launch.')
        expect(statements.join(' ')).not.toContain('Approval related text Finance Review Board')
        expect(statements.join(' ')).not.toContain('Question target')
    })

    it('strips graph-arrow prefixes when the relationship target contains the evidence sentence', () => {
        const selectedRelationship: ChatReferenceItem = {
            id: 'approval-finance-pdf-folded-target',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Approval to Finance Review Board',
            filename: 'llama-nexus-rich.pdf',
            contentType: 'application/pdf',
            body:
                'Approval --[co_occurs_with]--> Finance Review Board Finance Review Board owns approval evidence for the launch.',
            relationship: {
                source: 'Approval',
                relation: 'co_occurs_with',
                target: 'Finance Review Board owns approval evidence for the launch',
            },
        }

        const statements = relationshipConcreteEvidenceStatements(
            selectedRelationship,
            'How does Approval connect to Finance Review Board and Risk Map?',
        )

        expect(statements[0]).toBe('Finance Review Board owns approval evidence for the launch.')
        expect(statements.join(' ')).not.toContain('Approval co occurs with Finance Review Board')
    })

    it('uses concrete evidence as the brief title for weak PDF relationship references', () => {
        const selectedRelationship: ChatReferenceItem = {
            id: 'approval-finance-pdf-answer-card',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Finance Review Board owns approval evidence for the launch.',
            body: 'Finance Review Board owns approval evidence for the launch.',
            filename: 'llama-nexus-rich.pdf',
            contentType: 'application/pdf',
            relationship: {
                source: 'Approval',
                relation: 'Co Occurs With',
                target: 'Finance Review Board',
            },
        }

        expect(referenceBriefTitle(selectedRelationship)).toBe(
            'Finance Review Board owns approval evidence for the launch.',
        )
        expect(referenceBriefTitle(selectedRelationship)).not.toBe('Approval to Finance Review Board')
    })

    it('uses concrete evidence as the drawer preview for weak PDF relationships', () => {
        const selectedRelationship: ChatReferenceItem = {
            id: 'approval-finance-pdf-preview',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Approval to Finance Review Board',
            body:
                'Approval --[co_occurs_with]--> Finance Review Board Finance Review Board owns approval evidence for the launch.',
            filename: 'llama-nexus-rich.pdf',
            contentType: 'application/pdf',
            relationship: {
                source: 'Approval',
                relation: 'Co Occurs With',
                target: 'Finance Review Board',
            },
        }

        expect(referencePreviewText(selectedRelationship)).toBe(
            'Finance Review Board owns approval evidence for the launch.',
        )
        expect(referencePreviewText(selectedRelationship)).not.toContain(
            'Approval related text Finance Review Board',
        )
        expect(
            buildReferenceDocumentPreviewSummary(selectedRelationship, {
                documentText: selectedRelationship.body,
                query: 'How does Approval connect to Finance Review Board?',
            }).excerpt,
        ).toBe('Finance Review Board owns approval evidence for the launch.')
    })

    it('keeps PDF review references focused on selected path hops', () => {
        const selectedPath: ChatReferenceItem = {
            id: 'selected-path',
            kind: 'path',
            title: 'Product Revenue -> Regional Demand -> Risk Map',
            filename: 'annual-report.pdf',
            contentType: 'application/pdf',
            sourceLocations: [
                { label: 'Page', value: '1', category: 'page' },
            ],
            path: {
                nodes: ['Product Revenue', 'Regional Demand', 'Risk Map'],
                hops: 2,
                relationships: [
                    {
                        source: 'Product Revenue',
                        relation: 'drives',
                        target: 'Regional Demand',
                    },
                    {
                        source: 'Regional Demand',
                        relation: 'references',
                        target: 'Risk Map',
                    },
                ],
            },
        }
        const firstHop: ChatReferenceItem = {
            id: 'product-regional-demand',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Product Revenue -> Regional Demand',
            filename: 'annual-report.pdf',
            contentType: 'application/pdf',
            relationship: {
                source: 'Product Revenue',
                relation: 'drives',
                target: 'Regional Demand',
            },
        }
        const secondHop: ChatReferenceItem = {
            id: 'regional-demand-risk-map',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Regional Demand -> Risk Map',
            filename: 'annual-report.pdf',
            contentType: 'application/pdf',
            relationship: {
                source: 'Regional Demand',
                relation: 'references',
                target: 'Risk Map',
            },
        }
        const weakShortcut: ChatReferenceItem = {
            id: 'weak-direct',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Product Revenue -> Risk Map',
            filename: 'annual-report.pdf',
            contentType: 'application/pdf',
            relationship: {
                source: 'Product Revenue',
                relation: 'related_text',
                target: 'Risk Map',
            },
        }
        const unrelatedCitation: ChatReferenceItem = {
            id: 'unrelated-risk-owner',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Ada Lovelace -> Forecast Risk',
            filename: 'annual-report.pdf',
            contentType: 'application/pdf',
            relationship: {
                source: 'Ada Lovelace',
                relation: 'owns',
                target: 'Forecast Risk',
            },
        }
        const sourcePdf: ChatReferenceItem = {
            id: 'annual-report-source',
            kind: 'document',
            sourceRole: 'source_document',
            title: 'annual-report.pdf',
            filename: 'annual-report.pdf',
            contentType: 'application/pdf',
            endpoint: '/api/v1/graphrag/artifacts/annual-report.pdf',
        }

        const focusedReferences = selectFocusedReferenceReviewReferences(
            selectedPath,
            [sourcePdf, firstHop, secondHop, weakShortcut, unrelatedCitation],
            [firstHop, secondHop, weakShortcut, unrelatedCitation],
        )

        expect(focusedReferences.map(reference => reference.id)).toEqual([
            'selected-path',
            'product-regional-demand',
            'regional-demand-risk-map',
        ])
    })

    it('merges same-line PDF text fragments and suppresses coordinate boxes when text is available', () => {
        const highlights = mergePdfHighlightRects([
            { kind: 'text', label: 'Product Revenue', left: 10, top: 20, width: 80, height: 10 },
            { kind: 'text', label: 'Regional Demand', left: 94, top: 21, width: 92, height: 10 },
            { kind: 'text', label: 'Risk Map', left: 12, top: 48, width: 64, height: 10 },
            { kind: 'location', label: 'Page region', left: 8, top: 18, width: 188, height: 16 },
            { kind: 'location', label: 'Page region', left: 8.2, top: 18.2, width: 188.1, height: 16.1 },
            { kind: 'location', label: 'Image region', left: 220, top: 90, width: 80, height: 60 },
        ])

        expect(highlights).toHaveLength(2)
        expect(highlights.filter(highlight => highlight.kind === 'location')).toHaveLength(0)
        expect(highlights.find(highlight => highlight.label.includes('Product Revenue'))).toMatchObject({
            kind: 'text',
            left: 10,
            top: 20,
            width: 176,
        })
    })

    it('keeps deduped PDF coordinate boxes as a fallback when text matching finds nothing', () => {
        const highlights = mergePdfHighlightRects([
            { kind: 'location', label: 'Page region', left: 8, top: 18, width: 188, height: 16 },
            { kind: 'location', label: 'Page region', left: 8.2, top: 18.2, width: 188.1, height: 16.1 },
            { kind: 'location', label: 'Image region', left: 220, top: 90, width: 80, height: 60 },
        ])

        expect(highlights).toHaveLength(2)
        expect(highlights.map(highlight => highlight.label)).toEqual(['Page region', 'Image region'])
    })

    it('defaults PDF drawers to fit-to-width and leaves zoom-in explicit', () => {
        const basePageWidth = 612

        expect(pdfViewerRenderScale(basePageWidth, 690, 1440)).toBeCloseTo(654 / basePageWidth)
        expect(pdfViewerRenderScale(basePageWidth, 372, 390)).toBeCloseTo(336 / basePageWidth)
        expect(pdfViewerRenderScale(basePageWidth, 372, 390, 2)).toBeGreaterThan(
            pdfViewerRenderScale(basePageWidth, 372, 390),
        )
        expect(pdfViewerRenderScale(basePageWidth, 372, 390, 20)).toBe(2)
    })

    it('reads a page number from ordinary citation text', () => {
        expect(pageNumberFromReference(makePdfReference({
            subtitle: 'Source passage · page 3',
        }))).toBe(3)
    })

    it('reads a page number from structured source locations', () => {
        expect(pageNumberFromReference(makePdfReference({
            sourceLocations: [
                { label: 'File', value: 'annual-report.pdf' },
                { label: 'Page', value: '2', category: 'page' },
            ],
        }))).toBe(2)
    })

    it('reads page key-value hints emitted by GraphRAG metadata', () => {
        expect(pageNumberFromReference(makePdfReference({
            locationHints: ['file=annual-report.pdf; page=4; section=Revenue'],
        }))).toBe(4)
    })

    it('converts zero-based page index locations to a visible PDF page', () => {
        expect(pageNumberFromReference(makePdfReference({
            sourceLocations: [
                { label: 'page_index', value: '0', category: 'page' },
            ],
        }))).toBe(1)
    })

    it('collects cited pages in reading order without duplicates', () => {
        expect(pageNumbersFromReferences([
            makePdfReference({ subtitle: 'Source passage · page 7' }),
            makePdfReference({ locationHints: ['file=annual-report.pdf; page=2; section=Revenue'] }),
            makePdfReference({ body: 'Repeated reference on p. 7' }),
            makePdfReference({ title: 'No page metadata' }),
        ])).toEqual([2, 7])
    })

    it('maps inch-based GraphRAG PDF regions into rendered PDF points', () => {
        const viewport = { width: 612, height: 792, scale: 1 }
        const highlight = pdfLocationHighlightFromValue(
            'x=1.25in y=1.86in w=5.56in h=0.21in',
            viewport,
            'Region',
        )

        expect(highlight?.label).toBe('Region')
        expect(highlight?.left).toBeCloseTo(90)
        expect(highlight?.top).toBeCloseTo(133.92)
        expect(highlight?.width).toBeCloseTo(400.32)
        expect(highlight?.height).toBeCloseTo(15.12)
    })

    it('keeps percent-based regions percent-relative', () => {
        const viewport = { width: 600, height: 800, scale: 1.5 }
        const highlight = pdfLocationHighlightFromValue(
            '12.5, 20, 50, 10',
            viewport,
            'Region',
        )

        expect(highlight).toMatchObject({
            left: 75,
            top: 160,
            width: 300,
            height: 80,
        })
    })

    it('ignores broad PDF zone rectangles while keeping precise regions', () => {
        const viewport = { width: 600, height: 800, scale: 1.5 } as any
        const highlights = buildPdfLocationHighlights([
            {
                id: 'broad-zone',
                kind: 'relationship',
                title: 'Product Revenue -> Regional Demand',
                sourceLocations: [
                    { label: 'Zone top_left', value: '0, 0, 50, 50', category: 'region' },
                ],
            },
            {
                id: 'precise-bbox',
                kind: 'relationship',
                title: 'Regional Demand -> Risk Map',
                sourceLocations: [
                    { label: 'BBox', value: '10, 20, 30, 5', category: 'region' },
                ],
            },
        ], viewport, 1)

        expect(highlights).toHaveLength(1)
        expect(highlights[0]).toMatchObject({
            kind: 'location',
            left: 60,
            top: 160,
            width: 180,
            height: 40,
        })
    })
})

describe('ReferenceExplorer document preview summaries', () => {
    it('keeps requested evidence text available for sparse table and form drilldowns', () => {
        const tableReference: ChatReferenceItem = {
            id: 'requested-table-evidence',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Tables',
            filename: 'graphrag-ux-rich-e2e.pptx',
            body: 'Table 1 row 1: Metric: Forecast Risk',
            evidenceFamilies: ['requested_evidence_family', 'tables'],
            sourceLocations: [
                { label: 'Slide', value: '1', category: 'slide' },
                { label: 'Slide title', value: 'Revenue Architecture', category: 'slide' },
                { label: 'Sheet', value: 'ChartData', category: 'table' },
            ],
        }

        const summary = buildReferenceDocumentPreviewSummary(tableReference, {
            fallbackPreview: tableReference.body,
            evidenceLabels: ['Tables'],
            locationLabels: ['Slide 1', 'Sheet ChartData'],
            sourceTypeLabel: 'Evidence text',
        })

        expect(summary.eyebrow).toBe('Evidence text')
        expect(summary.title).toBe('Tables')
        expect(summary.subtitle).toContain('graphrag-ux-rich-e2e.pptx')
        expect(summary.subtitle).toContain('Slide 1: Revenue Architecture')
        expect(summary.excerpt).toBe('Table 1 row 1: Metric: Forecast Risk')
        expect(summary.chips).toContain('Tables')
        expect(summary.chips).not.toContain('Slide 1')
    })

    it('keeps semantic table row scaffolding out of evidence previews and outline chips', () => {
        const rawEvidenceRow = 'Source: Approval Gate; Relationship: gates; Target: Launch Workflow; Evidence: Gate tracked in spreadsheet Launch Evidence visibility visible; table source semantic_table_row Approval Gate Gates Launch Workflow'
        const reference: ChatReferenceItem = {
            id: 'semantic-table-row',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Approval Gate to Launch Workflow',
            filename: 'launch-workbook.xlsx',
            body: rawEvidenceRow,
            evidenceFamilies: ['semantic', 'table row'],
            relationship: {
                source: 'Approval Gate',
                relation: 'gates',
                target: 'Launch Workflow',
            },
        }

        const summary = buildReferenceDocumentPreviewSummary(reference, {
            documentText: [
                'Source | Relationship | Target | Evidence',
                rawEvidenceRow,
            ].join('\n'),
            fallbackPreview: rawEvidenceRow,
            outline: [
                'Source | Relationship | Target | Evidence',
                rawEvidenceRow,
            ],
            sourceTypeLabel: 'Evidence text',
            query: 'How does Approval Gate relate to Launch Workflow?',
        })

        const visibleText = [summary.excerpt, ...summary.outline].join(' ')
        expect(summary.excerpt).toBe('Gate tracked in spreadsheet Launch Evidence')
        expect(visibleText).toContain('Approval Gate to Launch Workflow')
        expect(visibleText).not.toContain('visibility visible')
        expect(visibleText).not.toContain('table source')
        expect(visibleText).not.toContain('semantic_table_row')
        expect(visibleText).not.toContain('Source | Relationship | Target | Evidence')
    })

    it('promotes PowerPoint slide context above low-level layout metadata', () => {
        const reference: ChatReferenceItem = {
            id: 'ppt-revenue-demand',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Product Revenue -> Regional Demand',
            filename: 'graphrag-live-readiness-1784793224.pptx',
            contentType: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            body: 'Product Revenue -> Regional Demand: drives',
            evidenceFamilies: ['semantic', 'connector', 'layout'],
            relationship: {
                source: 'Product Revenue',
                relation: 'drives',
                target: 'Regional Demand',
            },
            sourceLocations: [
                { label: 'Slide', value: '1', category: 'slide' },
                { label: 'Slide title', value: 'Revenue Architecture', category: 'slide' },
                { label: 'Connector note', value: 'Product revenue drives regional demand', category: 'layout' },
                { label: 'Region', value: 'x=3.00in y=2.40in w=1.00in h=0.30in', category: 'layout' },
            ],
        }

        const summary = buildReferenceDocumentPreviewSummary(reference, {
            locationLabels: ['Slide 1', 'Region x=3.00in y=2.40in w=1.00in h=0.30in'],
            evidenceLabels: ['Semantic', 'Connector', 'Layout'],
            sourceTypeLabel: 'Slide deck',
        })

        expect(summary.eyebrow).toBe('Slide deck')
        expect(summary.title).toBe('Slide 1: Revenue Architecture')
        expect(summary.subtitle).toContain('graphrag-live-readiness-1784793224.pptx')
        expect(summary.excerpt).toBe('Product revenue drives regional demand')
        expect(summary.chips).toContain('Semantic')
        expect(summary.chips).toContain('Connector')
        expect(summary.chips.join(' ')).not.toContain('x=3.00in')
        expect(summary.chips.join(' ')).not.toContain('Slide')
        expect(summary.outline).toEqual([])
    })

    it('builds a slide layout map from PowerPoint region metadata', () => {
        const reference: ChatReferenceItem = {
            id: 'ppt-layout-region',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Product Revenue -> Regional Demand',
            filename: 'graphrag-live-readiness-1784793224.pptx',
            relationship: {
                source: 'Product Revenue',
                relation: 'drives',
                target: 'Regional Demand',
            },
            sourceLocations: [
                { label: 'File', value: 'graphrag-live-readiness-1784793224.pptx', category: 'document' },
                { label: 'Slide', value: '1', category: 'slide' },
                { label: 'Slide title', value: 'Revenue Architecture', category: 'slide' },
                { label: 'Region', value: 'x=3.00in y=2.40in w=1.00in h=0.30in', category: 'layout' },
                { label: 'Source label', value: 'Product Revenue', category: 'layout' },
                { label: 'Target label', value: 'Regional Demand', category: 'layout' },
                { label: 'Connector note', value: 'Product revenue drives regional demand', category: 'layout' },
            ],
        }

        const map = buildReferenceLayoutMap(reference)

        expect(map?.title).toBe('Slide 1: Revenue Architecture')
        expect(map?.subtitle).toContain('1 region')
        expect(map?.boxes).toHaveLength(1)
        expect(map?.boxes[0]).toMatchObject({
            label: 'Product Revenue to Regional Demand',
            detail: 'Product revenue drives regional demand',
            tone: 'relationship',
        })
        expect(map?.boxes[0].left).toBeCloseTo(30)
        expect(map?.boxes[0].top).toBeCloseTo(42.67, 1)
        expect(map?.boxes[0].width).toBeCloseTo(10)
    })

    it('labels table layout regions with table, row, and cell context', () => {
        const reference: ChatReferenceItem = {
            id: 'ppt-table-layout-region',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Tables',
            filename: 'graphrag-ux-rich-e2e.pptx',
            body: 'Table 1 row 1 Cell R1C1: Product',
            sourceLocations: [
                { label: 'Slide', value: '1', category: 'slide' },
                { label: 'Slide title', value: 'Revenue Architecture', category: 'slide' },
                { label: 'Table', value: '1', category: 'table' },
                { label: 'Row', value: '1', category: 'table' },
                { label: 'Cell', value: 'R1C1: Product', category: 'table' },
                { label: 'Region', value: 'x=1.40in y=2.20in w=4.20in h=0.60in', category: 'layout' },
            ],
        }

        const map = buildReferenceLayoutMap(reference)

        expect(map?.boxes[0]).toMatchObject({
            label: 'Table 1 row 1 Cell R1C1',
            detail: 'Table 1 row 1 Cell R1C1',
        })
    })

    it('labels unlabeled layout regions from graph path hops', () => {
        const reference: ChatReferenceItem = {
            id: 'ppt-path-layout-region',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Regional Demand -> Product Revenue -> Revenue Architecture',
            filename: 'graphrag-live-readiness-1784793224.pptx',
            sourceLocations: [
                { label: 'Slide', value: '1', category: 'slide' },
                { label: 'Slide title', value: 'Revenue Architecture', category: 'slide' },
                { label: 'Region', value: 'x=1.00in y=2.20in w=2.00in h=0.60in', category: 'layout' },
            ],
            path: {
                nodes: ['Regional Demand', 'Product Revenue', 'Revenue Architecture'],
                hops: 2,
                relationships: [
                    {
                        source: 'Product Revenue',
                        relation: 'contains shape',
                        target: 'Revenue Architecture',
                        evidence: 'Product Revenue shape on the Revenue Architecture slide',
                    },
                ],
            },
        }

        const map = buildReferenceLayoutMap(reference)

        expect(map?.boxes[0].label).toBe('Product Revenue to Revenue Architecture')
        expect(map?.boxes[0].detail).toBe('Product Revenue shape on the Revenue Architecture slide')
    })

    it('does not repeat graph path labels on extra unlabeled layout regions', () => {
        const reference: ChatReferenceItem = {
            id: 'ppt-path-extra-layout-regions',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Regional Demand -> Product Revenue -> Revenue Architecture',
            filename: 'graphrag-live-readiness-1784793224.pptx',
            sourceLocations: [
                { label: 'Slide', value: '1', category: 'slide' },
                { label: 'Slide title', value: 'Revenue Architecture', category: 'slide' },
                { label: 'Region', value: 'x=1.00in y=2.20in w=2.00in h=0.60in', category: 'layout' },
                { label: 'Region', value: 'x=3.00in y=2.20in w=1.80in h=0.60in', category: 'layout' },
                { label: 'Region', value: 'x=5.00in y=2.20in w=1.80in h=0.60in', category: 'layout' },
            ],
            path: {
                nodes: ['Regional Demand', 'Product Revenue', 'Revenue Architecture'],
                hops: 1,
                relationships: [
                    {
                        source: 'Product Revenue',
                        relation: 'drives',
                        target: 'Regional Demand',
                        evidence: 'Product revenue drives regional demand',
                    },
                ],
            },
        }

        const map = buildReferenceLayoutMap(reference)

        expect(map?.boxes.map(box => box.label)).toEqual([
            'Product Revenue to Regional Demand',
            'Evidence region 2',
            'Evidence region 3',
        ])
        expect(map?.boxes.map(box => box.detail)).toEqual([
            'Product revenue drives regional demand',
            '',
            '',
        ])
    })

    it('dedupes repeated explicit region labels in the slide map legend', () => {
        const reference: ChatReferenceItem = {
            id: 'ppt-path-duplicate-layout-labels',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Product Revenue -> Regional Demand -> Risk Map',
            filename: 'graphrag-live-upload-audit.pptx',
            sourceLocations: [
                { label: 'Slide', value: '1', category: 'slide' },
                { label: 'Slide title', value: 'Revenue Architecture', category: 'slide' },
                { label: 'Region', value: 'x=1.00in y=2.20in w=2.00in h=0.60in', category: 'layout' },
                { label: 'Source label', value: 'Product Revenue', category: 'layout' },
                { label: 'Target label', value: 'Regional Demand', category: 'layout' },
                { label: 'Region', value: 'x=3.00in y=2.20in w=1.80in h=0.60in', category: 'layout' },
                { label: 'Source label', value: 'Regional Demand', category: 'layout' },
                { label: 'Target label', value: 'Risk Map', category: 'layout' },
                { label: 'Region', value: 'x=5.00in y=2.20in w=1.80in h=0.60in', category: 'layout' },
                { label: 'Source label', value: 'Regional Demand', category: 'layout' },
                { label: 'Target label', value: 'Risk Map', category: 'layout' },
            ],
        }

        const map = buildReferenceLayoutMap(reference)

        expect(map?.boxes.map(box => box.label)).toEqual([
            'Product Revenue to Regional Demand',
            'Regional Demand to Risk Map',
            'Evidence region 3',
        ])
    })

    it('uses readable labels for Office package parts in slide-map routes', () => {
        const reference: ChatReferenceItem = {
            id: 'ppt-package-layout-labels',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Revenue Architecture -> ppt/charts/chart1.xml -> source document',
            filename: 'graphrag-live-upload-audit.pptx',
            sourceLocations: [
                { label: 'Slide', value: '1', category: 'slide' },
                { label: 'Slide title', value: 'Revenue Architecture', category: 'slide' },
                { label: 'Region', value: 'x=1.00in y=0.60in w=8.00in h=0.40in', category: 'layout' },
                { label: 'Region', value: 'x=1.00in y=2.50in w=4.00in h=1.20in', category: 'layout' },
            ],
            path: {
                nodes: ['Revenue Architecture', 'ppt/charts/chart1.xml', 'source document'],
                hops: 2,
                relationships: [
                    {
                        source: 'Revenue Architecture',
                        relation: 'uses_chart_source',
                        target: 'ppt/charts/chart1.xml',
                    },
                    {
                        source: 'ppt/charts/chart1.xml',
                        relation: 'source_document_context',
                        target: 'source document',
                    },
                ],
            },
        }

        const map = buildReferenceLayoutMap(reference)

        expect(map?.boxes.map(box => box.label)).toEqual([
            'Revenue Architecture to chart evidence',
            'chart evidence to source document',
        ])
    })

    it('positions slide-map callouts near regions without leaving the canvas', () => {
        expect(layoutMapCalloutPosition(
            { left: 30, top: 42, height: 8 },
            0,
        )).toEqual({ left: 30, top: 29 })

        expect(layoutMapCalloutPosition(
            { left: 1, top: 4, height: 8 },
            1,
        )).toEqual({ left: 2, top: 20 })

        expect(layoutMapCalloutPosition(
            { left: 94, top: 98, height: 6 },
            0,
        )).toEqual({ left: 72, top: 85 })
    })

    it('staggers slide-map pins so overlapping regions keep visible route numbers', () => {
        const first = layoutMapPinPosition(
            { left: 30, top: 42, width: 10, height: 4 },
            0,
        )
        const second = layoutMapPinPosition(
            { left: 30, top: 42, width: 10, height: 4 },
            1,
        )

        expect(first).not.toEqual(second)
        expect(first).toEqual({ left: 32.4, top: 42.2 })
        expect(second).toEqual({ left: 37.6, top: 45.8 })
    })

    it('focuses sparse slide maps around the cited evidence regions', () => {
        const viewport = layoutMapViewport([
            { left: 45, top: 27, width: 10, height: 4 },
            { left: 55, top: 27, width: 30, height: 18 },
        ])

        expect(viewport.focused).toBe(true)
        expect(viewport.left).toBeLessThan(45)
        expect(viewport.width).toBeLessThan(70)

        const transformed = layoutMapViewportBox({
            id: 'region-1',
            label: 'Approval required before launch workflow.',
            detail: 'gates',
            left: 45,
            top: 27,
            width: 10,
            height: 4,
            tone: 'relationship',
        }, viewport)

        expect(transformed.left).toBeGreaterThan(0)
        expect(transformed.width).toBeGreaterThan(10)
    })
})

describe('ReferenceExplorer drawer reference curation', () => {
    const references = Array.from({ length: 9 }, (_value, index) => ({
        id: `reference-${index + 1}`,
    }))

    it('opens with a capped list of strongest references by default', () => {
        const result = selectReferenceDrawerRows(references)

        expect(result.capped).toBe(true)
        expect(result.rows).toHaveLength(DEFAULT_REFERENCE_DRAWER_LIMIT)
        expect(result.rows.map(reference => reference.id)).toEqual([
            'reference-1',
            'reference-2',
            'reference-3',
            'reference-4',
            'reference-5',
        ])
        expect(result.hiddenCount).toBe(4)
    })

    it('shows all references once the user searches, filters, or expands', () => {
        expect(selectReferenceDrawerRows(references, { search: 'ada' }).rows).toHaveLength(9)
        expect(selectReferenceDrawerRows(references, { kindFilter: 'relationship' }).rows).toHaveLength(9)
        expect(selectReferenceDrawerRows(references, { showAll: true }).rows).toHaveLength(9)
    })

    it('keeps drawer search out of compact evidence sets unless the user is already filtering', () => {
        expect(shouldShowReferenceExplorerDrawerSearch({
            referenceCount: 5,
            sourceGroupCount: 1,
            kindFilter: 'all',
            compact: true,
        })).toBe(false)
        expect(shouldShowReferenceExplorerDrawerSearch({
            referenceCount: 5,
            sourceGroupCount: 1,
            drawerSearch: 'revenue',
            kindFilter: 'all',
            compact: true,
        })).toBe(true)
        expect(shouldShowReferenceExplorerDrawerSearch({
            referenceCount: 5,
            sourceGroupCount: 1,
            kindFilter: 'relationship',
            compact: true,
        })).toBe(true)
    })

    it('shows drawer search for large evidence sets or broader source navigation', () => {
        expect(shouldShowReferenceExplorerDrawerSearch({
            referenceCount: 21,
            sourceGroupCount: 1,
            kindFilter: 'all',
        })).toBe(true)
        expect(shouldShowReferenceExplorerDrawerSearch({
            referenceCount: 4,
            sourceGroupCount: 3,
            kindFilter: 'all',
        })).toBe(true)
        expect(shouldShowReferenceExplorerDrawerSearch({
            referenceCount: 21,
            sourceGroupCount: 3,
            kindFilter: 'all',
            compact: true,
        })).toBe(false)
    })

    it('diversifies the default drawer rows across sources and evidence kinds', () => {
        const query = 'Trace Product Revenue via Regional Demand and cite the source documents.'
        const richPath: ChatReferenceItem = {
            id: 'rich-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Product Revenue -> Regional Demand -> Risk Map',
            filename: 'direct-rich.pptx',
            path: {
                nodes: ['Product Revenue', 'Regional Demand', 'Risk Map'],
                hops: 2,
            },
        }
        const richConnector: ChatReferenceItem = {
            id: 'rich-connector',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Product Revenue -> Regional Demand',
            filename: 'direct-rich.pptx',
            evidenceFamilies: ['connector'],
            relationship: {
                source: 'Product Revenue',
                relation: 'drives',
                target: 'Regional Demand',
            },
        }
        const richTable: ChatReferenceItem = {
            id: 'rich-table',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Tables',
            filename: 'direct-rich.pptx',
            evidenceFamilies: ['requested_evidence_family', 'tables'],
            body: 'Table 1 row 1 Cell R1C1: Product.',
        }
        const richStyled: ChatReferenceItem = {
            id: 'rich-styled',
            kind: 'passage',
            title: 'direct-rich.pptx',
            filename: 'direct-rich.pptx',
            evidenceFamilies: ['styled_text'],
            chunkIndex: 3,
            body: 'Styled text from slide 1.',
        }
        const richSource: ChatReferenceItem = {
            id: 'rich-source',
            kind: 'document',
            sourceRole: 'source_document',
            title: 'direct-rich.pptx',
            filename: 'direct-rich.pptx',
            body: 'Source package.',
        }
        const pdfPage: ChatReferenceItem = {
            id: 'pdf-page',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'page 1',
            filename: 'direct-brief.pdf',
            contentType: 'application/pdf',
            endpoint: '/api/source/direct-brief.pdf',
            sourceLocations: [{ label: 'Page', value: '1', category: 'page' }],
            body: 'Product Revenue drives Regional Demand on page 1.',
        }
        const pdfSource: ChatReferenceItem = {
            id: 'pdf-source',
            kind: 'document',
            sourceRole: 'source_document',
            title: 'direct-brief.pdf',
            filename: 'direct-brief.pdf',
            contentType: 'application/pdf',
            endpoint: '/api/source/direct-brief.pdf',
            body: 'PDF source.',
        }

        const diversified = diversifyReferenceExplorerDrawerReferences([
            richConnector,
            richTable,
            richStyled,
            richSource,
            pdfSource,
            pdfPage,
            richPath,
        ], query)
        const firstScreen = selectReferenceDrawerRows(diversified).rows as ChatReferenceItem[]

        expect(firstScreen.some(reference => reference.filename === 'direct-brief.pdf')).toBe(true)
        expect(firstScreen.some(reference => reference.kind === 'path')).toBe(true)
        expect(firstScreen.some(reference => reference.evidenceFamilies?.includes('requested_evidence_family'))).toBe(true)
        expect(firstScreen.filter(reference => reference.filename === 'direct-rich.pptx').length).toBeLessThan(
            DEFAULT_REFERENCE_DRAWER_LIMIT,
        )
        expect(firstScreen.map(reference => reference.id)).not.toEqual([
            'rich-path',
            'rich-connector',
            'rich-table',
            'rich-styled',
            'rich-source',
        ])
    })

    it('treats uploaded files as source navigation entries without promoting evidence-family labels', () => {
        expect(isSourceNavigationReference({
            id: 'connector-family',
            kind: 'passage',
            title: 'Connectors',
            evidenceFamilies: ['connector'],
            body: 'Product Revenue drives Regional Demand.',
        })).toBe(false)

        expect(isSourceNavigationReference({
            id: 'pdf-page',
            kind: 'passage',
            title: 'page 1',
            filename: 'direct-brief.pdf',
            contentType: 'application/pdf',
            endpoint: '/api/source/direct-brief.pdf',
        })).toBe(true)

        expect(isSourceNavigationReference({
            id: 'source-document',
            kind: 'document',
            sourceRole: 'source_document',
            title: 'direct-rich.pptx',
            filename: 'direct-rich.pptx',
        })).toBe(true)
    })

    it('puts graph paths before relationship rows for explicit path questions', () => {
        const query = 'Trace the path from Approval required before launch workflow to Launch workflow through Approval Dependency Cluster.'
        const path: ChatReferenceItem = {
            id: 'approval-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Approval required before launch workflow -> Approval Dependency Cluster -> Launch workflow',
            path: {
                nodes: ['Approval required before launch workflow', 'Approval Dependency Cluster', 'Launch workflow'],
                hops: 2,
            },
        }
        const relationship: ChatReferenceItem = {
            id: 'approval-edge',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Approval required before launch workflow -> Launch workflow',
            relationship: {
                source: 'Approval required before launch workflow',
                relation: 'gates',
                target: 'Launch workflow',
            },
        }
        const passage: ChatReferenceItem = {
            id: 'approval-passage',
            kind: 'passage',
            title: 'graphrag-rich-form-fixture.pptx',
            body: 'Slide 1: Launch Approval Packet',
        }

        expect(sortReferenceExplorerDrawerReferences([relationship, passage, path], query).map(reference => reference.id)).toEqual([
            'approval-path',
            'approval-edge',
            'approval-passage',
        ])
    })

    it('puts passages before incidental relationships for document citation questions', () => {
        const query = 'What validation result does this document report? Cite the source evidence.'
        const passage: ChatReferenceItem = {
            id: 'validation-passage',
            kind: 'passage',
            title: 'turboquant_validation_output.txt',
            body: 'TURBOQUANT PHASE A: REAL MODEL VALIDATION Model: Qwen/Qwen3-1.7B.',
        }
        const relationship: ChatReferenceItem = {
            id: 'warning-edge',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Warning -> You',
            relationship: {
                source: 'Warning',
                relation: 'co_occurs_with',
                target: 'You',
            },
        }

        expect(sortReferenceExplorerDrawerReferences([relationship, passage], query).map(reference => reference.id)).toEqual([
            'validation-passage',
            'warning-edge',
        ])
    })

    it('opens relationship/source questions on the direct cited relationship before paths', () => {
        const query =
            'Use this selected GraphRAG relationship as the primary evidence: Approval Dependency Cluster --[groups_visual]--> Approval required before launch workflow.'
        const path: ChatReferenceItem = {
            id: 'selected-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Approval Dependency Cluster -> Launch workflow -> Approval required before launch workflow.',
            score: 0.95,
            path: {
                nodes: [
                    'Approval Dependency Cluster',
                    'Launch workflow',
                    'Approval required before launch workflow.',
                ],
                hops: 2,
                relationships: [
                    {
                        source: 'Approval Dependency Cluster',
                        relation: 'groups_visual',
                        target: 'Launch workflow',
                    },
                    {
                        source: 'Launch workflow',
                        relation: 'gates',
                        target: 'Approval required before launch workflow.',
                    },
                ],
            },
        }
        const directRelationship: ChatReferenceItem = {
            id: 'direct-selected-relationship',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Approval Dependency Cluster -> Approval required before launch workflow.',
            score: 0.68,
            evidenceFamilies: ['semantic', 'layout', 'presentation structure'],
            relationship: {
                source: 'Approval Dependency Cluster',
                relation: 'groups_visual',
                target: 'Approval required before launch workflow.',
            },
        }

        expect(selectReferenceExplorerOpeningReference(
            [path, directRelationship],
            query,
            {
                trayReferences: [directRelationship, path],
                answerEvidenceReferences: [directRelationship],
            },
        )?.id).toBe('direct-selected-relationship')
    })

    it('opens graph-first connection questions on the path before requested table evidence', () => {
        const query =
            'How does Product Revenue connect to Regional Demand and Risk Map in this deck? Cite the slide, chart, or table evidence.'
        const requestedTableEvidence: ChatReferenceItem = {
            id: 'requested-table-evidence',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Tables',
            filename: 'graphrag-ux-rich-e2e.pptx',
            body: 'Requested evidence: Slide 1 table lists Forecast Risk and Product.',
            evidenceFamilies: ['requested_evidence_family', 'tables'],
            sourceLocations: [
                { label: 'Slide', value: '1', category: 'slide' },
            ],
        }
        const graphPath: ChatReferenceItem = {
            id: 'product-risk-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Product Revenue -> Regional Demand -> Risk Map',
            filename: 'graphrag-ux-rich-e2e.pptx',
            path: {
                nodes: ['Product Revenue', 'Regional Demand', 'Risk Map'],
                hops: 2,
                relationships: [
                    {
                        source: 'Product Revenue',
                        relation: 'drives',
                        target: 'Regional Demand',
                    },
                    {
                        source: 'Regional Demand',
                        relation: 'references',
                        target: 'Risk Map',
                    },
                ],
            },
        }
        const directRelationship: ChatReferenceItem = {
            id: 'product-regional-demand',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Product Revenue -> Regional Demand',
            filename: 'graphrag-ux-rich-e2e.pptx',
            relationship: {
                source: 'Product Revenue',
                relation: 'drives',
                target: 'Regional Demand',
            },
        }

        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: [requestedTableEvidence, directRelationship, graphPath],
            visibleReferences: [requestedTableEvidence, graphPath, directRelationship],
            answerEvidenceReferences: [requestedTableEvidence, directRelationship],
            query,
        })

        expect(trayReferences[0].id).toBe('product-risk-path')
        expect(trayReferences.map(reference => reference.id)).toContain('requested-table-evidence')
        expect(selectReferenceExplorerOpeningReference(
            [requestedTableEvidence, directRelationship, graphPath],
            query,
            {
                trayReferences,
                answerEvidenceReferences: [requestedTableEvidence, directRelationship],
            },
        )?.id).toBe('product-risk-path')
    })

    it('keeps extracted connector evidence questions focused on requested evidence cards', () => {
        const query = [
            'Extracted evidence: Connectors: Revenue Architecture -> Product Revenue -> Regional Demand: drives.',
            'Show the chart, table, form field, styled text, and connector evidence extracted from this PowerPoint.',
        ].join('\n')
        const tableEvidence: ChatReferenceItem = {
            id: 'requested-tables',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Tables',
            filename: 'graphrag-ux-rich-e2e.pptx',
            body: 'Table 1 row 1 Cell R1C1: Product.',
            evidenceFamilies: ['requested_evidence_family', 'tables'],
            sourceLocations: [{ label: 'Slide', value: '1', category: 'slide' }],
        }
        const connectorEvidence: ChatReferenceItem = {
            id: 'requested-connectors',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Connectors',
            filename: 'graphrag-ux-rich-e2e.pptx',
            body: 'Product Revenue -> Regional Demand: drives.',
            evidenceFamilies: ['requested_evidence_family', 'connectors'],
            sourceLocations: [{ label: 'Slide', value: '1', category: 'slide' }],
        }
        const metadataPath: ChatReferenceItem = {
            id: 'metadata-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'graphrag-ux-rich-e2e.pptx -> Microsoft PowerPoint -> Global Footer',
            filename: 'graphrag-ux-rich-e2e.pptx',
            path: {
                nodes: ['graphrag-ux-rich-e2e.pptx', 'Microsoft PowerPoint', 'Global Footer'],
                hops: 2,
                relationships: [
                    {
                        source: 'graphrag-ux-rich-e2e.pptx',
                        relation: 'created_with',
                        target: 'Microsoft PowerPoint',
                    },
                    {
                        source: 'Microsoft PowerPoint',
                        relation: 'contains_shape',
                        target: 'Global Footer',
                    },
                ],
            },
        }

        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: [metadataPath, tableEvidence, connectorEvidence],
            visibleReferences: [metadataPath, tableEvidence, connectorEvidence],
            answerEvidenceReferences: [tableEvidence, connectorEvidence],
            query,
        })

        expect(trayReferences.map(reference => reference.id)).toEqual([
            'requested-tables',
            'requested-connectors',
        ])
        expect(sortReferenceExplorerDrawerReferences(
            [metadataPath, connectorEvidence, tableEvidence],
            query,
        ).map(reference => reference.id).slice(0, 2).sort()).toEqual([
            'requested-connectors',
            'requested-tables',
        ].sort())
        expect(selectReferenceExplorerOpeningReference(
            [metadataPath, tableEvidence, connectorEvidence],
            query,
            {
                trayReferences,
                answerEvidenceReferences: [tableEvidence, connectorEvidence],
            },
        )?.id).toBe('requested-tables')
    })

    it('keeps query-ordered PDF paths ahead of weaker reordered paths', () => {
        const query =
            'Explain the Product Revenue to Regional Demand to Risk Map path. Cite the PDF page evidence.'
        const reorderedPath: ChatReferenceItem = {
            id: 'pdf-reordered-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Product Revenue -> Risk Map -> Regional Demand',
            filename: 'graphrag-live-pdf-evidence-ui-clean.pdf',
            contentType: 'application/pdf',
            score: 0.99,
            evidenceFamilies: ['semantic', 'layout'],
            path: {
                nodes: ['Product Revenue', 'Risk Map', 'Regional Demand'],
                hops: 2,
                relationships: [
                    {
                        source: 'Product Revenue',
                        relation: 'related text',
                        target: 'Risk Map',
                    },
                    {
                        source: 'Regional Demand',
                        relation: 'references',
                        target: 'Risk Map',
                    },
                ],
            },
        }
        const orderedPath: ChatReferenceItem = {
            id: 'pdf-ordered-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Product Revenue -> Regional Demand -> Risk Map',
            filename: 'graphrag-live-pdf-evidence-ui-clean.pdf',
            contentType: 'application/pdf',
            score: 0.67,
            evidenceFamilies: ['semantic', 'layout'],
            path: {
                nodes: ['Product Revenue', 'Regional Demand', 'Risk Map'],
                hops: 2,
                relationships: [
                    {
                        source: 'Product Revenue',
                        relation: 'drives',
                        target: 'Regional Demand',
                    },
                    {
                        source: 'Regional Demand',
                        relation: 'references',
                        target: 'Risk Map',
                    },
                ],
            },
        }

        expect(referencePathQueryOrderScore(orderedPath, query)).toBeGreaterThan(
            referencePathQueryOrderScore(reorderedPath, query),
        )
        expect(sortReferenceExplorerDrawerReferences([reorderedPath, orderedPath], query)[0].id).toBe('pdf-ordered-path')

        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: [reorderedPath, orderedPath],
            visibleReferences: [reorderedPath, orderedPath],
            answerEvidenceReferences: [],
            query,
        })

        expect(trayReferences[0].id).toBe('pdf-ordered-path')
        expect(selectReferenceExplorerOpeningReference(
            [reorderedPath, orderedPath],
            query,
            { trayReferences },
        )?.id).toBe('pdf-ordered-path')
    })

    it('prefers semantic multi-hop paths over weak direct related-text shortcuts', () => {
        const query = 'What path connects Product Revenue to Risk Map?'
        const weakShortcut: ChatReferenceItem = {
            id: 'weak-direct-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Product Revenue -> Risk Map',
            filename: 'graphrag-upload-smoke.md',
            score: 0.99,
            evidenceFamilies: ['semantic'],
            path: {
                nodes: ['Product Revenue', 'Risk Map'],
                hops: 1,
                relationships: [
                    {
                        source: 'Product Revenue',
                        relation: 'related_text',
                        target: 'Risk Map',
                    },
                ],
            },
        }
        const semanticPath: ChatReferenceItem = {
            id: 'semantic-two-hop-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Product Revenue -> Regional Demand -> Risk Map',
            filename: 'graphrag-upload-smoke.md',
            score: 0.42,
            evidenceFamilies: ['semantic'],
            path: {
                nodes: ['Product Revenue', 'Regional Demand', 'Risk Map'],
                hops: 2,
                relationships: [
                    {
                        source: 'Product Revenue',
                        relation: 'drives',
                        target: 'Regional Demand',
                    },
                    {
                        source: 'Regional Demand',
                        relation: 'references',
                        target: 'Risk Map',
                    },
                ],
            },
        }
        const weakDirectRelationship: ChatReferenceItem = {
            id: 'weak-direct-relationship',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Product Revenue -> Risk Map',
            filename: 'graphrag-upload-smoke.md',
            score: 0.97,
            evidenceFamilies: ['semantic'],
            relationship: {
                source: 'Product Revenue',
                relation: 'related_text',
                target: 'Risk Map',
            },
        }
        const firstHop: ChatReferenceItem = {
            id: 'product-regional-demand',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Product Revenue -> Regional Demand',
            filename: 'graphrag-upload-smoke.md',
            relationship: {
                source: 'Product Revenue',
                relation: 'drives',
                target: 'Regional Demand',
            },
        }
        const secondHop: ChatReferenceItem = {
            id: 'regional-demand-risk-map',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Regional Demand -> Risk Map',
            filename: 'graphrag-upload-smoke.md',
            relationship: {
                source: 'Regional Demand',
                relation: 'references',
                target: 'Risk Map',
            },
        }

        expect(referencePathQueryOrderScore(weakShortcut, query)).toBeGreaterThan(
            referencePathQueryOrderScore(semanticPath, query),
        )
        expect(referencePathSemanticStrengthScore(semanticPath)).toBeGreaterThan(
            referencePathSemanticStrengthScore(weakShortcut),
        )
        expect(sortReferenceExplorerDrawerReferences([weakShortcut, semanticPath], query)[0].id).toBe('semantic-two-hop-path')

        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: [
                weakShortcut,
                weakDirectRelationship,
                firstHop,
                secondHop,
                semanticPath,
            ],
            visibleReferences: [
                weakShortcut,
                weakDirectRelationship,
                firstHop,
                secondHop,
                semanticPath,
            ],
            answerEvidenceReferences: [
                weakDirectRelationship,
                firstHop,
                secondHop,
            ],
            query,
        })

        expect(trayReferences[0].id).toBe('semantic-two-hop-path')
        expect(trayReferences.map(reference => reference.id)).not.toContain('weak-direct-path')
        expect(trayReferences.map(reference => reference.id)).not.toContain('weak-direct-relationship')
        expect(trayReferences.some(reference =>
            ['product-regional-demand', 'regional-demand-risk-map'].includes(reference.id),
        )).toBe(true)
        expect(selectReferenceExplorerOpeningReference(
            [
                weakShortcut,
                weakDirectRelationship,
                firstHop,
                secondHop,
                semanticPath,
            ],
            query,
            { trayReferences },
        )?.id).toBe('semantic-two-hop-path')
    })

    it('keeps passage evidence first for ordinary document citation questions', () => {
        const query = 'What validation result does this document report? Cite the source evidence.'
        const passage: ChatReferenceItem = {
            id: 'warning-passage',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'turboquant_validation_output.txt',
            body: 'Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN.',
            filename: 'turboquant_validation_output.txt',
            chunkIndex: 1,
        }
        const warningRelationship: ChatReferenceItem = {
            id: 'warning-you',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Warning -> You',
            body: 'Warning co occurs with You',
            filename: 'turboquant_validation_output.txt',
            relationship: {
                source: 'Warning',
                relation: 'co_occurs_with',
                target: 'You',
            },
        }
        const hubRelationship: ChatReferenceItem = {
            id: 'you-hub',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'You -> HF Hub',
            body: 'You co occurs with HF Hub',
            filename: 'turboquant_validation_output.txt',
            relationship: {
                source: 'You',
                relation: 'co_occurs_with',
                target: 'HF Hub',
            },
        }
        const representativeDocumentChunk: ChatReferenceItem = {
            id: 'representative-document-chunk',
            kind: 'passage',
            title: 'document',
            body: 'Representative document chunk',
            filename: 'turboquant_validation_output.txt',
        }

        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: [warningRelationship, hubRelationship, representativeDocumentChunk, passage],
            visibleReferences: [warningRelationship, hubRelationship, representativeDocumentChunk, passage],
            answerEvidenceReferences: [warningRelationship, hubRelationship, representativeDocumentChunk, passage],
            query,
        })

        expect(trayReferences[0].id).toBe('warning-passage')
        expect(trayReferences.map(reference => reference.id)).not.toContain('representative-document-chunk')
        expect(selectReferenceExplorerOpeningReference(
            [warningRelationship, hubRelationship, representativeDocumentChunk, passage],
            query,
            {
                trayReferences,
                answerEvidenceReferences: [warningRelationship, hubRelationship, representativeDocumentChunk, passage],
            },
        )?.id).toBe('warning-passage')

        const trayReferencesWithVisiblePassage = selectReferenceExplorerTrayReferences({
            orderedReferences: [warningRelationship, hubRelationship, representativeDocumentChunk, passage],
            visibleReferences: [warningRelationship, hubRelationship, representativeDocumentChunk, passage],
            answerEvidenceReferences: [warningRelationship, hubRelationship],
            query,
        })

        expect(trayReferencesWithVisiblePassage[0].id).toBe('warning-passage')
        expect(selectReferenceExplorerOpeningReference(
            [warningRelationship, hubRelationship, representativeDocumentChunk, passage],
            query,
            {
                trayReferences: trayReferencesWithVisiblePassage,
                answerEvidenceReferences: [warningRelationship, hubRelationship],
            },
        )?.id).toBe('warning-passage')
    })

    it('leads with explicitly named files instead of unrelated rich graph cards', () => {
        const query = 'Across README.md and turboquant_validation_output.txt, what should I know before running this project locally? Cite the relevant documents.'
        const readmeReference: ChatReferenceItem = {
            id: 'readme-local-backend',
            kind: 'passage',
            sourceRole: 'source_document',
            title: 'README.md',
            body: 'Start the local GraphRAG document-chat backend with scripts/start-graphrag-backend-local.sh.',
            filename: 'README.md',
        }
        const validationReference: ChatReferenceItem = {
            id: 'turboquant-hf-token',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'turboquant_validation_output.txt',
            body: 'Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN.',
            filename: 'turboquant_validation_output.txt',
        }
        const unrelatedRichCard: ChatReferenceItem = {
            id: 'pptx-rich-table',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Revenue Architecture table 1',
            body: 'Chunk preserves table structure as subject-predicate-object evidence.',
            filename: 'graphrag-ux-rich-e2e.pptx',
            evidenceFamilies: ['table', 'structured'],
            relationship: {
                source: 'Product Revenue',
                relation: 'drives',
                target: 'Regional Demand',
            },
        }
        const syntheticFilenameRelation: ChatReferenceItem = {
            id: 'filename-works-at',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'README.md -> turboquant_validation_output.txt',
            body: 'README.md works at turboquant_validation_output.txt.',
            filename: 'README.md',
            relationship: {
                source: 'README.md',
                relation: 'Works At',
                target: 'turboquant_validation_output.txt',
            },
        }
        const validationGraphResidue: ChatReferenceItem = {
            id: 'validation-graph-residue',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'TURBOQUANT PHASE A -> REAL MODEL VALIDATION Model',
            body: 'TURBOQUANT PHASE A related text REAL MODEL VALIDATION Model',
            filename: 'turboquant_validation_output.txt',
            relationship: {
                source: 'TURBOQUANT PHASE A',
                relation: 'related_text',
                target: 'REAL MODEL VALIDATION Model',
            },
        }

        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: [unrelatedRichCard, syntheticFilenameRelation, validationGraphResidue, readmeReference, validationReference],
            visibleReferences: [unrelatedRichCard, syntheticFilenameRelation, validationGraphResidue, readmeReference, validationReference],
            answerEvidenceReferences: [unrelatedRichCard, syntheticFilenameRelation, validationGraphResidue, validationReference],
            query,
        })

        expect([...trayReferences.map(reference => reference.id)].sort()).toEqual([
            'turboquant-hf-token',
            'readme-local-backend',
        ].sort())
        expect(selectReferenceExplorerOpeningReference(
            [unrelatedRichCard, syntheticFilenameRelation, validationGraphResidue, readmeReference, validationReference],
            query,
            {
                trayReferences,
                answerEvidenceReferences: [unrelatedRichCard, syntheticFilenameRelation, validationGraphResidue, validationReference],
            },
        )?.id).not.toBe('validation-graph-residue')
    })

    it('allows representative source chunks when an explicitly named text file has no richer passage', () => {
        const query = 'What does turboquant_validation_output.txt warn me to configure?'
        const representativeChunk: ChatReferenceItem = {
            id: 'plain-text-representative-chunk',
            kind: 'passage',
            sourceRole: 'source_document',
            title: 'document',
            subtitle: 'Representative document chunk',
            body: 'Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN.',
            filename: 'turboquant_validation_output.txt',
        }
        const graphResidue: ChatReferenceItem = {
            id: 'plain-text-graph-residue',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'TURBOQUANT PHASE A -> REAL MODEL VALIDATION Model',
            body: 'TURBOQUANT PHASE A related text REAL MODEL VALIDATION Model',
            filename: 'turboquant_validation_output.txt',
            relationship: {
                source: 'TURBOQUANT PHASE A',
                relation: 'related_text',
                target: 'REAL MODEL VALIDATION Model',
            },
        }

        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: [graphResidue, representativeChunk],
            visibleReferences: [graphResidue, representativeChunk],
            answerEvidenceReferences: [graphResidue],
            query,
        })

        expect(trayReferences[0].id).toBe('plain-text-representative-chunk')
        expect(selectReferenceExplorerOpeningReference(
            [graphResidue, representativeChunk],
            query,
            { trayReferences, answerEvidenceReferences: [graphResidue] },
        )?.id).toBe('plain-text-representative-chunk')
    })

    it('uses answer-specific terms to prioritize the cited named-file source', () => {
        const query = [
            'Across README.md and turboquant_validation_output.txt, what should I know before running this project locally?',
            'The answer cites turboquant_validation_output.txt for the HF_TOKEN warning and README.md for local Docker setup.',
        ].join('\n')
        const readmeSetup: ChatReferenceItem = {
            id: 'readme-setup',
            kind: 'passage',
            sourceRole: 'source_document',
            title: 'README.md',
            body: 'Clone the repository and start services with docker compose up -d --build.',
            filename: 'README.md',
            chunkIndex: 6,
        }
        const validationWarning: ChatReferenceItem = {
            id: 'validation-warning',
            kind: 'document',
            sourceRole: 'source_document',
            title: 'turboquant_validation_output.txt',
            body: 'Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN.',
            filename: 'turboquant_validation_output.txt',
        }
        const validationPath: ChatReferenceItem = {
            id: 'validation-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'TURBOQUANT PHASE A -> REAL MODEL VALIDATION Model',
            body: 'TURBOQUANT PHASE A related text REAL MODEL VALIDATION Model',
            filename: 'turboquant_validation_output.txt',
            path: {
                nodes: ['TURBOQUANT PHASE A', 'REAL MODEL VALIDATION Model'],
                hops: 1,
            },
        }

        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: [readmeSetup, validationPath, validationWarning],
            visibleReferences: [readmeSetup, validationPath, validationWarning],
            answerEvidenceReferences: [validationPath],
            query,
        })

        expect(trayReferences[0].id).toBe('validation-warning')
        expect(selectReferenceExplorerOpeningReference(
            [readmeSetup, validationPath, validationWarning],
            query,
            { trayReferences, answerEvidenceReferences: [validationPath] },
        )?.id).toBe('validation-warning')
    })

    it('surfaces rich evidence family cards for chart table form styled-text and connector questions', () => {
        const query = [
            'Extracted evidence: Tables, Charts, Form Fields, Styled Text, and Connectors all cite graphrag-ux-rich-e2e.pptx, slide 1.',
            'What chart, table, form, styled-text, and connector evidence was extracted from this deck? Cite the strongest source locations.',
        ].join('\n')
        const table: ChatReferenceItem = {
            id: 'tables',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Tables',
            body: 'Table 1 row 1 contains Product and Forecast Risk.',
            filename: 'graphrag-ux-rich-e2e.pptx',
            evidenceFamilies: ['table'],
        }
        const chart: ChatReferenceItem = {
            id: 'charts',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Charts',
            body: 'Regional Revenue Forecast chart includes Quarter and Revenue axes.',
            filename: 'graphrag-ux-rich-e2e.pptx',
            evidenceFamilies: ['chart'],
        }
        const form: ChatReferenceItem = {
            id: 'forms',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Form Fields',
            body: 'Approved for launch is checked.',
            filename: 'graphrag-ux-rich-e2e.pptx',
            evidenceFamilies: ['form_fields'],
        }
        const styled: ChatReferenceItem = {
            id: 'styled',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Styled Text',
            body: 'Template Governance Note is bold and Forecast Risk is emphasized.',
            filename: 'graphrag-ux-rich-e2e.pptx',
            evidenceFamilies: ['styled_text'],
        }
        const connector: ChatReferenceItem = {
            id: 'connectors',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Connectors',
            body: 'Product Revenue -> Regional Demand: drives connector is preserved.',
            filename: 'graphrag-ux-rich-e2e.pptx',
            evidenceFamilies: ['connector'],
        }
        const genericDeckPassage: ChatReferenceItem = {
            id: 'generic-deck-passage',
            kind: 'passage',
            sourceRole: 'source_document',
            title: 'graphrag-ux-rich-e2e.pptx',
            body: 'Slide 1: Revenue Architecture Table 1 Cell R1C1 Bold',
            filename: 'graphrag-ux-rich-e2e.pptx',
        }

        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: [genericDeckPassage, table, styled, chart, form, connector],
            visibleReferences: [genericDeckPassage, table, styled, chart, form, connector],
            answerEvidenceReferences: [table, styled, chart, form, connector],
            query,
        })

        expect(trayReferences.map(reference => reference.id)).toEqual([
            'charts',
            'tables',
            'forms',
            'styled',
            'connectors',
        ])
        expect(selectInlineReferenceExplorerTrayReferences(trayReferences, query).map(reference => reference.id)).toEqual([
            'charts',
            'tables',
            'connectors',
        ])
    })

    it('uses broad inline evidence cards to expose visual document depth without adding more tray cards', () => {
        const query = [
            'Extracted evidence: chart, table, form field, styled text, connector, speaker notes, media, image, and embedded files.',
            'Show the chart, table, form field, styled text, connector, speaker notes, media, image, and embedded files extracted from this PowerPoint.',
        ].join('\n')
        const evidence = (id: string, title: string, family: string, body: string): ChatReferenceItem => ({
            id,
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title,
            body,
            filename: 'launch-approval.pptx',
            evidenceFamilies: ['requested_evidence_family', family],
        })
        const references = [
            evidence('tables', 'Tables', 'tables', 'Table 1 row 2 has approval evidence.'),
            evidence('images', 'Images', 'images', 'Risk map image appears on slide 1.'),
            evidence('attachments', 'Attachments', 'attachments', 'Embedded approval workbook is attached.'),
            evidence('forms', 'Form Fields', 'form_fields', 'Approval required checkbox is checked.'),
            evidence('charts', 'Charts', 'charts', 'Revenue Plan chart includes Q2 and Q3.'),
            evidence('media', 'Media', 'media', 'Embedded media evidence appears on slide 1.'),
            evidence('styled', 'Styled Text', 'styled_text', 'Approval is emphasized.'),
            evidence('speaker-notes', 'Speaker Notes', 'speaker_notes', 'Speaker notes cite approval evidence.'),
            evidence('connectors', 'Connectors', 'connectors', 'Approval connector gates launch.'),
        ]

        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: references,
            visibleReferences: references,
            answerEvidenceReferences: references,
            query,
        })

        expect(trayReferences.map(reference => reference.id)).toEqual([
            'charts',
            'tables',
            'forms',
            'styled',
            'connectors',
            'speaker-notes',
            'media',
            'images',
            'attachments',
        ])
        expect(selectInlineReferenceExplorerTrayReferences(trayReferences, query).map(reference => reference.id)).toEqual([
            'charts',
            'tables',
            'images',
        ])
    })

    it('prioritizes the direct gates citation over surrounding visual grouping in the tray', () => {
        const query = 'What gates the launch workflow, and what source evidence supports that relationship?'
        const groupedLaunch: ChatReferenceItem = {
            id: 'grouped-launch',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Approval Dependency Cluster -> Launch workflow',
            body: 'Approval Dependency Cluster groups visual Launch workflow',
            filename: 'graphrag-rich-form-fixture.pptx',
            evidenceFamilies: ['shape_group', 'layout'],
            relationship: {
                source: 'Approval Dependency Cluster',
                relation: 'groups_visual',
                target: 'Launch workflow',
            },
        }
        const groupedGate: ChatReferenceItem = {
            id: 'grouped-gate',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Approval Dependency Cluster -> gates',
            body: 'Approval Dependency Cluster groups visual gates',
            filename: 'graphrag-rich-form-fixture.pptx',
            evidenceFamilies: ['shape_group', 'layout'],
            relationship: {
                source: 'Approval Dependency Cluster',
                relation: 'groups_visual',
                target: 'gates',
            },
        }
        const directGate: ChatReferenceItem = {
            id: 'direct-gates',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Approval required before launch workflow. -> Launch workflow',
            body: 'Approval required before launch workflow. gates Launch workflow',
            filename: 'graphrag-rich-form-fixture.pptx',
            evidenceFamilies: ['connector', 'shape'],
            relationship: {
                source: 'Approval required before launch workflow.',
                relation: 'gates',
                target: 'Launch workflow',
            },
        }

        const trayReferences = selectReferenceExplorerTrayReferences({
            orderedReferences: [groupedLaunch, groupedGate, directGate],
            visibleReferences: [groupedLaunch, groupedGate, directGate],
            answerEvidenceReferences: [groupedLaunch, groupedGate, directGate],
            query,
        })

        expect(trayReferences[0].id).toBe('direct-gates')
    })

    it('opens path questions on the strongest graph path before single-hop citations', () => {
        const query =
            'How does Regional Demand connect to Revenue Architecture? Explain the graph path and cite the document evidence.'
        const directRelationship: ChatReferenceItem = {
            id: 'regional-demand-risk-map-citation',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Regional Demand -> Risk Map',
            score: 0.74,
            filename: 'graphrag-live-readiness.pptx',
            evidenceFamilies: ['semantic', 'connector', 'layout'],
            relationship: {
                source: 'Regional Demand',
                relation: 'references',
                target: 'Risk Map',
            },
        }
        const path: ChatReferenceItem = {
            id: 'regional-demand-revenue-architecture-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Regional Demand -> Risk Map -> Revenue Architecture',
            score: 0.68,
            filename: 'graphrag-live-readiness.pptx',
            evidenceFamilies: ['semantic', 'layout'],
            path: {
                nodes: ['Regional Demand', 'Risk Map', 'Revenue Architecture'],
                hops: 2,
                relationships: [
                    {
                        source: 'Regional Demand',
                        relation: 'references',
                        target: 'Risk Map',
                    },
                    {
                        source: 'Risk Map',
                        relation: 'links_to',
                        target: 'Revenue Architecture',
                    },
                ],
            },
        }
        const alternatePath: ChatReferenceItem = {
            ...path,
            id: 'alternate-regional-demand-revenue-architecture-path',
            title: 'Regional Demand -> Product Revenue -> Revenue Architecture',
            score: 0.95,
            path: {
                nodes: ['Regional Demand', 'Product Revenue', 'Revenue Architecture'],
                hops: 2,
                relationships: [
                    {
                        source: 'Regional Demand',
                        relation: 'driven_by',
                        target: 'Product Revenue',
                    },
                    {
                        source: 'Product Revenue',
                        relation: 'appears_in',
                        target: 'Revenue Architecture',
                    },
                ],
            },
        }

        expect(selectReferenceExplorerOpeningReference(
            [alternatePath, directRelationship, path],
            query,
            {
                trayReferences: [directRelationship, path],
                answerEvidenceReferences: [directRelationship],
            },
        )?.id).toBe('regional-demand-revenue-architecture-path')
    })

    it('resolves deduped visible path cards to the closest selectable drawer path', () => {
        const visiblePathCard: ChatReferenceItem = {
            id: 'drawer-product-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Regional Demand -> Risk Map -> https://example.com/risk-register -> Revenue Architecture',
            path: {
                nodes: ['Regional Demand', 'Risk Map', 'https://example.com/risk-register', 'Revenue Architecture'],
                hops: 3,
            },
        }
        const productPath: ChatReferenceItem = {
            id: 'drawer-product-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Regional Demand -> Product Revenue -> Revenue Architecture',
            path: {
                nodes: ['Regional Demand', 'Product Revenue', 'Revenue Architecture'],
                hops: 2,
            },
        }
        const riskMapPath: ChatReferenceItem = {
            id: 'drawer-risk-map-path',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Regional Demand -> Risk Map -> Revenue Architecture',
            path: { hops: 2 },
        }

        expect(resolveReferenceExplorerSelectionReference(
            [productPath, riskMapPath],
            visiblePathCard,
            'How does Regional Demand connect to Revenue Architecture?',
        ).id).toBe('drawer-risk-map-path')
    })

    it('keeps the clicked path object active when it shares an id with a sorted drawer path', () => {
        const sortedDrawerPath: ChatReferenceItem = {
            id: 'shared-path-id',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Regional Demand -> Product Revenue -> Revenue Architecture',
            path: {
                nodes: ['Regional Demand', 'Product Revenue', 'Revenue Architecture'],
                hops: 2,
            },
        }
        const clickedAnswerPath: ChatReferenceItem = {
            id: 'shared-path-id',
            kind: 'path',
            sourceRole: 'reasoning_path',
            title: 'Regional Demand -> Risk Map -> Revenue Architecture',
            path: {
                nodes: ['Regional Demand', 'Risk Map', 'Revenue Architecture'],
                hops: 2,
            },
        }

        expect(selectReferenceExplorerActiveReference(
            [sortedDrawerPath],
            'shared-path-id',
            clickedAnswerPath,
        )?.title).toBe('Regional Demand -> Risk Map -> Revenue Architecture')
    })

    it('collapses repeated relationship citations that point to the same document fact', () => {
        const first: ChatReferenceItem = {
            id: 'risk-map-citation-1',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Risk Map -> Launch Priority',
            filename: 'graphrag-pdf-strategy.pdf',
            sourceLocations: [
                { label: 'Page', value: '1', category: 'page' },
            ],
            relationship: {
                source: 'Risk Map',
                relation: 'gates',
                target: 'Launch Priority',
            },
        }
        const duplicate: ChatReferenceItem = {
            ...first,
            id: 'risk-map-citation-2',
            sourceRole: 'graph_relationship',
            title: 'Risk Map gates Launch Priority',
            locationHints: ['file=graphrag-pdf-strategy.pdf; page=1; block=5'],
        }
        const adjacentFact: ChatReferenceItem = {
            ...first,
            id: 'risk-map-hold',
            title: 'Risk Map -> Launch Priority Hold',
            relationship: {
                source: 'Risk Map',
                relation: 'gates',
                target: 'Launch Priority Hold launch until approved',
            },
        }

        expect(chatReferenceDisplayDedupeKey(first)).toBe(chatReferenceDisplayDedupeKey(duplicate))
        expect(dedupeChatReferencesForDisplay([first, duplicate, adjacentFact]).map(reference => reference.id)).toEqual([
            'risk-map-citation-1',
            'risk-map-hold',
        ])
    })

    it('keeps the same relationship visible when it appears on different pages', () => {
        const pageOne: ChatReferenceItem = {
            id: 'risk-page-1',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Risk Map -> Launch Priority',
            filename: 'graphrag-pdf-strategy.pdf',
            sourceLocations: [
                { label: 'Page', value: '1', category: 'page' },
            ],
            relationship: {
                source: 'Risk Map',
                relation: 'gates',
                target: 'Launch Priority',
            },
        }
        const pageTwo: ChatReferenceItem = {
            ...pageOne,
            id: 'risk-page-2',
            sourceLocations: [
                { label: 'Page', value: '2', category: 'page' },
            ],
        }

        expect(dedupeChatReferencesForDisplay([pageOne, pageTwo]).map(reference => reference.id)).toEqual([
            'risk-page-1',
            'risk-page-2',
        ])
    })
})

describe('ReferenceExplorer path display', () => {
    it('keeps document bridge steps visible without exposing raw filenames', () => {
        expect(displayPathNodes([
            'Product Revenue',
            'graphrag-ux-rich-e2e.pptx',
            'Ada Lovelace',
            'Forecast Risk',
        ])).toEqual([
            'Product Revenue',
            'source evidence',
            'Ada Lovelace',
            'Forecast Risk',
        ])
    })

    it('does not invent a semantic hop when adjacent document nodes are compacted', () => {
        expect(displayPathNodes([
            'Product Revenue',
            'first.pdf',
            'second.docx',
            'Forecast Risk',
        ])).toEqual([
            'Product Revenue',
            'source evidence',
            'Forecast Risk',
        ])
    })

    it('renames Office package parts inside visible graph paths', () => {
        expect(displayPathNodes([
            'Revenue Architecture',
            'ppt/charts/chart1.xml',
            'ppt/embeddings/operations.xlsx',
            'source document',
        ])).toEqual([
            'Revenue Architecture',
            'chart evidence',
            'embedded data',
            'source document',
        ])
    })

    it('dedupes repeated path hops in the evidence drill-down', () => {
        const hops = displayPathRelationships([
            {
                source: 'Product Revenue',
                relation: 'drives',
                target: 'Regional Demand',
            },
            {
                source: 'Regional Demand',
                relation: 'references',
                target: 'Risk Map',
            },
            {
                source: 'regional demand',
                relation: 'References',
                target: 'Risk Map',
                evidence: 'Duplicate alternate provenance should not repeat the visible hop.',
            },
        ])

        expect(hops).toHaveLength(2)
        expect(hops.map(hop => `${hop.source} -> ${hop.target}`)).toEqual([
            'Product Revenue -> Regional Demand',
            'Regional Demand -> Risk Map',
        ])
    })

    it('includes relationship labels in compact path previews', () => {
        const preview = buildReferencePathPreviewText({
            id: 'approval-path',
            kind: 'path',
            title: 'Approval required before launch workflow. -> Launch workflow',
            path: {
                nodes: [
                    'Approval required before launch workflow.',
                    'Launch workflow',
                ],
                hops: 1,
                relationships: [
                    {
                        source: 'Approval required before launch workflow.',
                        relation: 'gates',
                        target: 'Launch workflow',
                    },
                ],
            },
        })

        expect(preview).toContain('Approval required before launch workflow')
        expect(preview).toContain('gates')
        expect(preview).toContain('Launch workflow')
        expect(preview).not.toContain('--gates-->')
    })
})

describe('ReferenceExplorer tray subtitle', () => {
    it('keeps graph path cards readable without raw scores in the default tray', () => {
        const subtitle = buildReferenceTraySubtitle({
            id: 'path-1',
            kind: 'path',
            sourceRole: 'answer_evidence',
            title: 'Approval Gate -> Launch Workflow',
            filename: 'semantic-clean.csv',
            subtitle: 'semantic-clean.csv · 1 hop · Semantic · 79% score',
            path: {
                nodes: ['Approval Gate', 'Launch Workflow'],
                score: 0.788,
            },
        })

        expect(subtitle).toBe('1 hop · Semantic')
    })

    it('keeps source tray subtitles free of raw slide geometry', () => {
        const subtitle = buildReferenceTraySubtitle({
            id: 'relationship-geometry',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Product Revenue -> Regional Demand',
            filename: 'graphrag-live-upload-audit.pptx',
            subtitle: 'answer citation · graphrag-live-upload-audit.pptx · Primary Path Hop · slide 1: Revenue Architecture | x=3.00in y=2.40in w=1.00in h=0.30in | middle_center',
            relationship: {
                source: 'Product Revenue',
                relation: 'drives',
                target: 'Regional Demand',
            },
        })

        expect(subtitle).toBe('Slide 1: Revenue Architecture')
        expect(subtitle).not.toContain('x=3.00in')
        expect(subtitle).not.toContain('middle_center')
    })

    it('keeps Office package internals out of source tray subtitles', () => {
        const subtitle = buildReferenceTraySubtitle({
            id: 'embedded-workbook-hop',
            kind: 'relationship',
            sourceRole: 'answer_evidence',
            title: 'Revenue Architecture -> chart evidence',
            filename: 'graphrag-live-upload-audit.pptx',
            subtitle: 'Primary Path Hop · Slide 1: Revenue Architecture | Operations | media embedded_package | media part ppt/embeddings/operations.xlsx',
            relationship: {
                source: 'Revenue Architecture',
                relation: 'references',
                target: 'ppt/embeddings/operations.xlsx',
            },
        })

        expect(subtitle).toBe('Slide 1: Revenue Architecture | embedded data')
        expect(subtitle).not.toContain('ppt/embeddings')
        expect(subtitle).not.toContain('operations.xlsx')
        expect(subtitle).not.toContain('media embedded_package')
    })

    it('turns malformed scope and match fragments into compact table evidence subtitles', () => {
        const subtitle = buildReferenceTraySubtitle({
            id: 'summary-passage',
            kind: 'passage',
            sourceRole: 'retrieved_passage',
            title: 'graphrag-ux-rich-e2e.pptx',
            filename: 'graphrag-ux-rich-e2e.pptx',
            subtitle: 'Chunk 14 · 100% match Scope:); table 1; cell R1C1; style bold',
        })

        expect(subtitle).toBe('Chunk 14 · Table 1 | Cell R1C1 | Bold')
        expect(subtitle).not.toContain('Scope')
        expect(subtitle).not.toContain('100% match')
        expect(subtitle).not.toContain(');')
    })

    it('labels requested table evidence with table context instead of orphan numeric fragments', () => {
        const subtitle = buildReferenceTraySubtitle({
            id: 'requested-table',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Tables',
            filename: 'graphrag-ux-rich-e2e.pptx',
            subtitle: 'Requested evidence · Slide 1: Revenue Architecture | 1 · Tables · graphrag-ux-rich-e2e.pptx',
            evidenceFamilies: ['requested_evidence_family', 'tables'],
        })

        expect(subtitle).toBe('Requested evidence · Slide 1: Revenue Architecture | Table 1 · Tables')
        expect(subtitle).not.toContain('| 1')
    })

    it('turns raw form control metadata into readable requested evidence subtitles', () => {
        const subtitle = buildReferenceTraySubtitle({
            id: 'requested-form',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Form Fields',
            filename: 'graphrag-ux-rich-e2e.pptx',
            subtitle: 'Requested evidence · Slide 1: Revenue Architecture | field Approved for launch | type Forms.CheckBox.1 | value True | checked checked · Form Fields',
            evidenceFamilies: ['requested_evidence_family', 'form_fields'],
        })

        expect(subtitle).toBe('Requested evidence · Slide 1: Revenue Architecture | Approved for launch | Checkbox | Selected · Form Fields')
        expect(subtitle).not.toContain('checked checked')
        expect(subtitle).not.toContain('Forms.CheckBox.1')
    })

    it('turns raw styled text metadata into readable requested evidence subtitles', () => {
        const subtitle = buildReferenceTraySubtitle({
            id: 'requested-styled',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Styled Text',
            filename: 'graphrag-ux-rich-e2e.pptx',
            subtitle: 'Requested evidence · Slide 1: Revenue Architecture | style summary bold · Styled Text',
            evidenceFamilies: ['requested_evidence_family', 'styled_text'],
        })

        expect(subtitle).toBe('Requested evidence · Slide 1: Revenue Architecture | Bold · Styled Text')
        expect(subtitle).not.toContain('style summary')
    })

    it('keeps compact requested evidence subtitles free of visual style internals', () => {
        const subtitle = buildReferenceTraySubtitle({
            id: 'requested-chart-style',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Charts',
            filename: 'llama-nexus-rich.pptx',
            subtitle: 'Requested evidence · Slide 1: Launch Approval Packet | value 18 | Revenue Plan | fill #00A884 | line #123456 · Charts',
            evidenceFamilies: ['requested_evidence_family', 'charts'],
        })

        expect(subtitle).toBe('Requested evidence · Slide 1: Launch Approval Packet | Value 18 | Revenue Plan · Charts')
        expect(subtitle).not.toContain('fill #')
        expect(subtitle).not.toContain('line #')
    })

    it('drops half-truncated slide titles from compact requested evidence subtitles', () => {
        const subtitle = buildReferenceTraySubtitle({
            id: 'requested-chart-truncated-slide',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Charts',
            filename: 'llama-nexus-rich.pptx',
            subtitle: 'Requested evidence · Slide 1: Laun | value 18 | Revenue Plan · Charts',
            evidenceFamilies: ['requested_evidence_family', 'charts'],
        })

        expect(subtitle).toBe('Requested evidence · Slide 1 | Value 18 | Revenue Plan · Charts')
        expect(subtitle).not.toContain('Laun')
    })

    it('formats requested table tray previews without spreadsheet-style x separators', () => {
        const preview = requestedEvidenceTrayPreviewText({
            id: 'requested-table-preview',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Tables',
            filename: 'llama-nexus-rich.pptx',
            body: 'table: 1; row header: Approval required; column header: Relation; cell: R2C2; evidence=Approval required x Relation: gates',
            evidenceFamilies: ['requested_evidence_family', 'tables'],
        })

        expect(preview).toBe('Table 1 · Approval required / Relation · Cell R2C2')
        expect(preview).not.toContain(' x ')
    })

    it('formats requested image tray previews without raw package field names', () => {
        const reference: ChatReferenceItem = {
            id: 'requested-image-preview',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Images',
            filename: 'llama-nexus-rich.pptx',
            body: 'image part: risk-map.png; image mime: image/png; slide: Launch Approval Packet',
            evidenceFamilies: ['requested_evidence_family', 'images'],
        }
        const preview = requestedEvidenceTrayPreviewText(reference)

        expect(preview).toBe('Risk Map image')
        expect(referencePreviewTextForQuery(reference, 'Show the extracted image evidence.')).toBe('Risk Map image')
        expect(preview).not.toContain('image part:')
        expect(preview).not.toContain('image mime:')
    })

    it('formats requested media and attachment tray previews as human assets', () => {
        const mediaPreview = requestedEvidenceTrayPreviewText({
            id: 'requested-media-preview',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Media',
            filename: 'llama-nexus-rich.pptx',
            body: 'media part: demo-clip.mp4; media mime: video/mp4; slide: Launch Approval Packet',
            evidenceFamilies: ['requested_evidence_family', 'media'],
        })
        const attachmentPreview = requestedEvidenceTrayPreviewText({
            id: 'requested-attachment-preview',
            kind: 'passage',
            sourceRole: 'answer_evidence',
            title: 'Attachments',
            filename: 'llama-nexus-rich.pptx',
            body: 'embedded file filename: approval-workbook.xlsx; embedded file mime: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            evidenceFamilies: ['requested_evidence_family', 'attachments'],
        })

        expect(mediaPreview).toBe('Demo Clip media')
        expect(attachmentPreview).toBe('Approval Workbook attachment')
        expect(mediaPreview).not.toContain('media part:')
        expect(attachmentPreview).not.toContain('embedded file filename:')
    })

    it('dedupes repeated chart fields while preserving useful requested evidence details', () => {
        const preview = cleanRequestedEvidencePreview(
            'chart: Revenue Plan; chart: Revenue Plan; series name: Approval; point: Approval Q2; category: Q2; value: 18; axis: Revenue impact; slide: Launch Approval Packet',
            'llama-nexus-rich.pptx',
        )

        expect(preview.match(/chart: Revenue Plan/gi)).toHaveLength(1)
        expect(preview).toContain('series name: Approval')
        expect(preview).toContain('value: 18')
        expect(preview).toContain('axis: Revenue impact')
    })

    it('removes broken arrow prefixes from requested form evidence previews', () => {
        const preview = cleanRequestedEvidencePreview(
            'Form Fields: label: Approved for launch; field type: Forms.CheckBox.1; checked: true; value: True; Revenue Architecture --contains form field--> Launch region; evidence=Launch region: Europe; provenance=slide 1: Revenue Architecture; value Europe; form Launch region',
            'graphrag-ux-rich-e2e.pptx',
        )

        expect(preview).toBe('Launch region: Europe')
        expect(preview).not.toContain('->')
        expect(preview).not.toContain('Form field -')
    })

    it('keeps requested layout evidence previews focused on the relationship', () => {
        const preview = cleanRequestedEvidencePreview(
            'Regional Revenue Forecast is visually near Slide 1 Table 1 (Slide 1 Table 1 is below Regional Revenue Forecast); distance 21.3%; zones bottom left -> bottom left Source: direct-rich.pptx, slide 1: Revenue Architecture, chart Regional Revenue Forecast',
            'direct-rich.pptx',
        )
        const sourceDocumentPreview = cleanRequestedEvidencePreview(
            'Regional Revenue Forecast is visually near Slide 1 Table 1 (Slide 1 Table 1 is below Regional Revenue Forecast); distance 21.3%; zones bottom left -> bottom left source document, slide 1: Revenue Architecture, chart Regional Revenue Forecast',
            'direct-rich.pptx',
        )

        expect(preview).toBe(
            'Regional Revenue Forecast is visually near Slide 1 Table 1; distance 21.3%; zones bottom left to bottom left',
        )
        expect(sourceDocumentPreview).toBe(preview)
        expect(preview).not.toContain('is below Regional Revenue Forecast')
        expect(preview).not.toContain('Source:')
        expect(preview).not.toContain('source document')
        expect(preview).not.toContain('direct-rich.pptx')
    })

    it('keeps raw path scores out of default path evidence labels', () => {
        const labels = buildReferencePathEvidenceLabels({
            id: 'path-with-score',
            kind: 'path',
            title: 'Product Revenue -> Regional Demand -> Risk Map',
            path: {
                nodes: ['Product Revenue', 'Regional Demand', 'Risk Map'],
                hops: 2,
                score: 0.65,
            },
        })

        expect(labels).toEqual(['2 hops'])
        expect(labels.join(' ')).not.toContain('score')
        expect(labels.join(' ')).not.toContain('0.65')
    })
})
