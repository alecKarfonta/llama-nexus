import { describe, expect, it } from 'vitest'
import {
    buildChatReferenceContextSummary,
    formatChatReferenceCount,
} from './chatReferenceSummary'

describe('chat reference summary', () => {
    it('keeps the document chat status compact and user-facing', () => {
        const summary = buildChatReferenceContextSummary({
            attachedDocumentCount: 1,
            references: [
                {
                    kind: 'relationship',
                    sourceRole: 'answer_evidence',
                    filename: 'launch-plan.csv',
                    evidenceFamilies: ['semantic', 'table', 'direct_relationship'],
                },
                {
                    kind: 'path',
                    sourceRole: 'answer_evidence',
                    filename: 'launch-plan.csv',
                    evidenceFamilies: ['semantic'],
                },
            ],
            passageCount: 1,
            relationshipCount: 1,
            pathCount: 3,
        })

        expect(summary).toBe('Document ready · 2 citations · 1 source · 3 graph paths · table evidence')
    })

    it('uses readable singular labels', () => {
        expect(formatChatReferenceCount(1, 'citation')).toBe('1 citation')
        expect(formatChatReferenceCount(1, 'passage')).toBe('1 passage')
        expect(formatChatReferenceCount(1, 'graph path')).toBe('1 graph path')
    })

    it('falls back to passages when there are no graph paths or relations', () => {
        const summary = buildChatReferenceContextSummary({
            attachedDocumentCount: 1,
            references: [
                {
                    kind: 'passage',
                    sourceRole: 'retrieved_passage',
                    filename: 'notes.md',
                    evidenceFamilies: ['semantic'],
                },
            ],
            passageCount: 1,
        })

        expect(summary).toBe('Document ready · 1 source · 1 passage')
    })
})
