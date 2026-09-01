import { describe, expect, it } from 'vitest'
import { buildChatReferenceSourceBrief } from './chatReferenceSourceBrief'

describe('chat reference source brief', () => {
    it('summarizes source evidence counts, locations, and the lead citation', () => {
        const brief = buildChatReferenceSourceBrief([
            {
                id: 'rel-1',
                kind: 'relationship',
                sourceRole: 'answer_evidence',
                title: 'Regional Demand drives Revenue Forecast',
                filename: 'strategy-review.pdf',
                body: 'The forecast section says demand changes drive the revenue model.',
                evidenceFamilies: ['semantic', 'chart_points'],
                relationship: {
                    source: 'Regional Demand',
                    relation: 'drives',
                    target: 'Revenue Forecast',
                },
                sourceLocations: [
                    { label: 'Page', value: '4', category: 'page' },
                    { label: 'Chart', value: 'Revenue model', category: 'chart' },
                ],
            },
            {
                id: 'passage-1',
                kind: 'passage',
                sourceRole: 'answer_evidence',
                title: 'Revenue passage',
                filename: 'strategy-review.pdf',
                body: 'Revenue forecast excerpt.',
                evidenceFamilies: ['semantic'],
                locationHints: ['file=strategy-review.pdf; page=4; section=Revenue'],
            },
            {
                id: 'doc-1',
                kind: 'document',
                sourceRole: 'source_document',
                title: 'strategy-review.pdf',
                filename: 'strategy-review.pdf',
                contentType: 'application/pdf',
                endpoint: '/api/v1/graphrag/artifacts/strategy-review.pdf',
            },
        ])

        expect(brief).toMatchObject({
            title: 'strategy-review.pdf',
            countLabels: ['2 citations', '1 passage', '1 relation', '1 document'],
            referenceCount: 3,
        })
        expect(brief?.lead).toContain('Regional Demand drives Revenue Forecast')
        expect(brief?.lead).toContain('forecast section')
        expect(brief?.evidenceLabels).toEqual(['Semantic 2', 'Chart Points'])
        expect(brief?.locationLabels).toContain('Page 4')
        expect(brief?.locationLabels).toContain('strategy-review.pdf / page 4')
    })

    it('keeps graph paths readable at the source level', () => {
        const brief = buildChatReferenceSourceBrief([
            {
                id: 'path-1',
                kind: 'path',
                sourceRole: 'answer_evidence',
                title: 'Forecast path',
                filename: 'deck.pptx',
                evidenceFamilies: ['diagram_relationships'],
                path: {
                    nodes: ['Revenue Forecast', 'Regional Demand', 'Launch Risk'],
                    hops: 2,
                },
            },
        ])

        expect(brief?.title).toBe('deck.pptx')
        expect(brief?.countLabels).toEqual(['1 citation', '1 path'])
        expect(brief?.lead).toBe('Revenue Forecast -> Regional Demand -> Launch Risk.')
        expect(brief?.evidenceLabels).toEqual(['Diagram Relationships'])
    })

    it('keeps raw pipeline metadata out of relationship source briefs', () => {
        const brief = buildChatReferenceSourceBrief([
            {
                id: 'rel-metadata',
                kind: 'relationship',
                sourceRole: 'answer_evidence',
                title: 'Forecast Risk owner',
                filename: 'strategy.pptx',
                body: 'Forecast Risk --[source_owner]-- Ada Lovelace (local pipeline strategy.pptx; slide 1: Revenue Architecture; ChartData visibility visible; sheet part xl/worksheets/sheet1.xml; sheet relationship rId1)',
                evidenceFamilies: ['semantic'],
                relationship: {
                    source: 'Forecast Risk',
                    relation: 'source_owner',
                    target: 'Ada Lovelace',
                },
            },
        ])

        expect(brief?.lead).toBe('Forecast Risk source owner Ada Lovelace.')
        expect(brief?.lead).not.toContain('local pipeline')
        expect(brief?.lead).not.toContain('xl/worksheets')
        expect(brief?.lead).not.toContain('visibility visible')
    })

    it('keeps numeric geometry out of source-level location chips', () => {
        const brief = buildChatReferenceSourceBrief([
            {
                id: 'rel-geometry',
                kind: 'relationship',
                sourceRole: 'answer_evidence',
                title: 'Forecast Risk owner',
                filename: 'strategy.pptx',
                body: 'Forecast Risk source owner Ada Lovelace.',
                evidenceFamilies: ['semantic'],
                relationship: {
                    source: 'Forecast Risk',
                    relation: 'source_owner',
                    target: 'Ada Lovelace',
                },
                sourceLocations: [
                    { label: 'Region', value: '10, 39.1, 20, 10.7', category: 'region' },
                    { label: 'Slide', value: '1', category: 'slide' },
                ],
                locationHints: ['Region 10, 39.1, 20, 10.7'],
            },
        ])

        expect(brief?.locationLabels).toEqual(['Slide 1'])
    })
})
