import { describe, expect, it } from 'vitest'
import { buildChatReferenceDocumentReview } from './chatReferenceDocumentText'

describe('chat reference document text review', () => {
  it('builds contextual matched-line rows with gap markers', () => {
    const review = buildChatReferenceDocumentReview(
      [
        'Title: Revenue Strategy',
        'Overview',
        'Product Revenue appears on the architecture slide.',
        'Background',
        'Operations',
        'Forecast Risk is owned by Ada Lovelace.',
      ].join('\n'),
      ['Product Revenue', 'Forecast Risk'],
      { contextRadius: 0 },
    )

    expect(review.lineCount).toBe(6)
    expect(review.matchCount).toBe(2)
    expect(review.rows).toEqual([
      {
        type: 'line',
        lineNumber: 3,
        text: 'Product Revenue appears on the architecture slide.',
        matched: true,
      },
      {
        type: 'gap',
        hiddenLines: 2,
      },
      {
        type: 'line',
        lineNumber: 6,
        text: 'Forecast Risk is owned by Ada Lovelace.',
        matched: true,
      },
    ])
  })

  it('keeps surrounding context for matched lines', () => {
    const review = buildChatReferenceDocumentReview(
      ['alpha', 'before', 'Ada Lovelace owns the metric.', 'after', 'omega'].join('\n'),
      ['Ada Lovelace'],
      { contextRadius: 1 },
    )

    expect(review.rows).toEqual([
      { type: 'line', lineNumber: 2, text: 'before', matched: false },
      { type: 'line', lineNumber: 3, text: 'Ada Lovelace owns the metric.', matched: true },
      { type: 'line', lineNumber: 4, text: 'after', matched: false },
    ])
  })

  it('ignores generic source terms so everything is not highlighted', () => {
    const review = buildChatReferenceDocumentReview(
      ['Source document', 'Source evidence', 'Useful detail'].join('\n'),
      ['source', 'document'],
    )

    expect(review.matchCount).toBe(0)
    expect(review.rows).toEqual([])
  })

  it('ignores document-format boilerplate terms', () => {
    const review = buildChatReferenceDocumentReview(
      [
        'Application: Microsoft PowerPoint',
        'Slides: 2',
        'Layout Governance',
        'Product Revenue appears here.',
      ].join('\n'),
      ['PowerPoint', 'Slides', 'Layout', 'Product Revenue'],
      { contextRadius: 0 },
    )

    expect(review.matchCount).toBe(1)
    expect(review.rows).toEqual([
      {
        type: 'line',
        lineNumber: 4,
        text: 'Product Revenue appears here.',
        matched: true,
      },
    ])
  })

  it('keeps metadata out of the highlighted match review', () => {
    const review = buildChatReferenceDocumentReview(
      [
        'Title: GraphRAG Revenue Strategy',
        'Creator: Ada Lovelace',
        'Keywords: GraphRAG; Revenue; Risk',
        'Company: Llama Nexus Labs',
        'Slide 1: Revenue Architecture',
        'Product Revenue drives Regional Demand.',
        'Risk Map shows launch blockers.',
      ].join('\n'),
      ['GraphRAG', 'Revenue', 'Risk', 'Product Revenue'],
      { contextRadius: 1 },
    )

    expect(review.matchCount).toBe(3)
    expect(review.rows).toEqual([
      { type: 'line', lineNumber: 5, text: 'Slide 1: Revenue Architecture', matched: true },
      { type: 'line', lineNumber: 6, text: 'Product Revenue drives Regional Demand.', matched: true },
      { type: 'line', lineNumber: 7, text: 'Risk Map shows launch blockers.', matched: true },
    ])
  })

  it('keeps semantic table row scaffolding out of highlighted evidence lines', () => {
    const review = buildChatReferenceDocumentReview(
      [
        'Source | Relationship | Target | Evidence',
        'Source: Approval Gate; Relationship: gates; Target: Launch Workflow; Evidence: Gate tracked in spreadsheet Launch Evidence visibility visible; table source semantic_table_row Approval Gate Gates Launch Workflow',
      ].join('\n'),
      ['Approval Gate', 'Launch Workflow', 'Launch Evidence'],
      { contextRadius: 0 },
    )

    const text = review.rows
      .filter(row => row.type === 'line')
      .map(row => row.text)
      .join(' ')

    expect(text).toContain('Gate tracked in spreadsheet Launch Evidence')
    expect(text).not.toContain('visibility visible')
    expect(text).not.toContain('table source')
    expect(text).not.toContain('semantic_table_row')
  })
})
