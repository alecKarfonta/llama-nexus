import { describe, expect, it } from 'vitest'

import { cleanGraphRAGAssistantAnswer } from './graphragAnswerText'

describe('GraphRAG answer text cleanup', () => {
    it('keeps document-chat answers user-facing without raw trace details', () => {
        const answer = cleanGraphRAGAssistantAnswer(
            [
                'Based on the GraphRAG pipeline trace for semantic-clean-1807582440.csv, the connection is direct.',
                '',
                'Source Provenance: semantic-clean-1807582440.csv, Row 2',
                'Pipeline Trace: run graphrag_ingest_9e29aad6eccf (mode: graph_path_reasoning).',
                'The direct relationship provides the most precise and highest-scored evidence (score: 0.788) for this graph trace.',
            ].join('\n'),
            { sourceLabel: 'semantic-clean-1784807582440.csv' },
        )

        expect(answer).toContain('From semantic-clean-1784807582440.csv, the connection is direct.')
        expect(answer).toContain('Source Provenance: semantic-clean-1784807582440.csv, Row 2')
        expect(answer).not.toContain('GraphRAG pipeline trace')
        expect(answer).not.toContain('Pipeline Trace:')
        expect(answer).toContain('graph evidence')
        expect(answer).not.toContain('graph trace')
        expect(answer).not.toContain('graphrag_ingest_')
        expect(answer).not.toContain('score: 0.788')
        expect(answer).not.toContain('highest-scored')
    })

    it('removes compact score labels without dropping source wording', () => {
        const answer = cleanGraphRAGAssistantAnswer(
            'Graph path: Direct relationship Evidence: The path is shown in document evidence with 79% score and score=0.94. This is strongest, with a path score of 0.788, and no intermediate hops are needed. While other paths exist via Ready with a lower score of 0.641, the direct relation has a higher confidence score and alternatives have lower confidence scores.',
        )

        expect(answer).toBe('Graph path: Direct relationship\n\nEvidence: The path is shown in document evidence. This is strongest, and no intermediate hops are needed. While other paths exist via Ready, the direct relation has higher confidence and alternatives have lower confidence.')
    })

    it('removes standalone path score lines from complex document answers', () => {
        const answer = cleanGraphRAGAssistantAnswer(
            [
                'Strongest Multi-Hop Path (Structural Context):',
                '',
                'Path: source document --[contains_shape_group]--> Launch Evidence Cluster --[groups_visual]--> Evidence Snapshot --[links_to]--> https://example.com/evidence/approval-gate',
                'Score: 0.78',
                '- **Score:** 0.798',
                'Graph Path: source document → [has_slide] → Slide 1 → [contains_image] → Evidence Snapshot → [links_to] → https://example.com/evidence/approval-gate.',
                'Explanation: This path illustrates the visual hierarchy within the presentation.',
            ].join('\n'),
        )

        expect(answer).toContain('Strongest Multi-Hop Path')
        expect(answer).toContain('Explanation: This path illustrates')
        expect(answer).toContain('source document -> contains shape group -> Launch Evidence Cluster -> groups visual -> Evidence Snapshot -> links to ->')
        expect(answer).toContain('source document -> has slide -> Slide 1 -> contains image -> Evidence Snapshot -> links to ->')
        expect(answer).not.toContain('Score: 0.78')
        expect(answer).not.toContain('Score:** 0.798')
        expect(answer).not.toContain('score')
        expect(answer).not.toContain('contains_shape_group')
        expect(answer).not.toContain('contains_image')
        expect(answer).not.toContain('groups_visual')
        expect(answer).not.toContain('links_to')
        expect(answer).not.toContain('--[')
    })

    it('separates compressed graph path headings in visual-link answers', () => {
        const answer = cleanGraphRAGAssistantAnswer(
            [
                '**Graph path:** Direct This connection is established directly from the source document to the URL.',
                '',
                'Provenance and Trace Evidence:',
                'Source Location: grouped-visual.pptx, Slide 1.',
            ].join('\n'),
        )

        expect(answer).toContain('**Graph path:** Direct.\n\nThis connection is established directly')
        expect(answer).toContain('Provenance and Evidence:')
        expect(answer).not.toContain('Trace Evidence')
    })

    it('removes code-chip markdown from document labels, URLs, and relation labels', () => {
        const answer = cleanGraphRAGAssistantAnswer(
            [
                'Direct Connection The file `graphrag-grouped-visual-link.pptx` connects to `https://example.com/evidence/approval-gate`.',
                '',
                'Relationship: `source_document --[links_to]--> https://example.com/evidence/approval-gate`',
                'Graph Path source document → `[has_slide]` → Slide 1 → `[contains_link]` → Evidence Snapshot → `[links_to]` → `https://example.com/evidence/approval-gate`.',
                'The relation is `links_to`.',
                'The visual anchor is `Evidence Snapshot` inside `Launch Evidence Cluster`.',
            ].join('\n'),
        )

        expect(answer).toContain('Direct Connection: The file graphrag-grouped-visual-link.pptx connects to https://example.com/evidence/approval-gate.')
        expect(answer).toContain('Relationship: source document -> links to -> https://example.com/evidence/approval-gate')
        expect(answer).toContain('Graph Path: source document -> has slide -> Slide 1 -> contains link -> Evidence Snapshot -> links to -> https://example.com/evidence/approval-gate.')
        expect(answer).toContain('The relation is links to.')
        expect(answer).toContain('The visual anchor is Evidence Snapshot inside Launch Evidence Cluster.')
        expect(answer).not.toContain('`')
        expect(answer).not.toContain('links_to')
        expect(answer).not.toContain('source_document')
    })

    it('humanizes extraction graph relation labels in generated paths', () => {
        const answer = cleanGraphRAGAssistantAnswer(
            '`Graph path:` Slide 1 table (contains_table) -> Slide 1 (has_section) -> Revenue Architecture (contains_chart) -> source_document.',
        )

        expect(answer).toContain('Graph path:')
        expect(answer).toContain('contains table')
        expect(answer).toContain('has section')
        expect(answer).toContain('contains chart')
        expect(answer).toContain('source document')
        expect(answer).not.toContain('`Graph path:`')
        expect(answer).not.toContain('contains_table')
        expect(answer).not.toContain('has_section')
        expect(answer).not.toContain('contains_chart')
        expect(answer).not.toContain('source_document')
    })

    it('removes graph-first evidence bullets from ordinary document answers', () => {
        const answer = cleanGraphRAGAssistantAnswer(
            [
                'The document reports a Real Model Validation for Qwen/Qwen3-1.7B.',
                '',
                'Source Evidence:',
                '- Direct Graph Relationship: REAL MODEL VALIDATION Model -> co-occurs with -> Qwen/Qwen3-1.7B | Source: turboquant_validation_output.txt',
                '- Supporting Graph Path: The validation context links through Warning, You, HF Hub, Please, and TURBOQUANT PHASE A.',
                '- Specific Text: "TURBOQUANT PHASE A: REAL MODEL VALIDATION Model: Qwen/Qwen3-1.7B (head_dim=128, same as Qwen 27B)"',
            ].join('\n'),
            { documentFirst: true },
        )

        expect(answer).toContain('The document reports a Real Model Validation')
        expect(answer).toContain('Source:')
        expect(answer).toContain('Specific Text: "TURBOQUANT PHASE A')
        expect(answer).not.toContain('Direct Graph Relationship')
        expect(answer).not.toContain('Supporting Graph Path')
    })
})
