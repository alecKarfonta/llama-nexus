import { describe, expect, it } from 'vitest';
import {
  formatGraphRAGReasoningLocation,
  normalizeGraphRAGReasoningResult,
} from './graphragReasoningResults';

describe('graphrag reasoning result normalization', () => {
  it('normalizes local pipeline path reasoning payloads', () => {
    const normalized = normalizeGraphRAGReasoningResult({
      reasoning_source: 'local_pipeline',
      reasoning_task: 'multi_hop',
      warnings: ['Remote GraphRAG reasoning unavailable.'],
      local_pipeline: {
        run_count: 1,
        matched_path_count: 1,
        matched_relationship_count: 2,
      },
      reasoning_path: [
        {
          hop: 1,
          source: 'Strategy Deck',
          relation: 'has_slide',
          target: 'Slide 1: Revenue Architecture',
          confidence: 0.72,
          metadata: {
            local_graph_filename: 'strategy.pptx',
            slide: 1,
            slide_title: 'Revenue Architecture',
          },
        },
      ],
      paths: [
        {
          text: 'Strategy Deck --[has_slide]-- Slide 1',
          nodes: ['Strategy Deck', 'Slide 1: Revenue Architecture'],
          score: 0.82,
          path_type: 'layout',
        },
      ],
      sources: [
        {
          content: 'Strategy Deck --[has_slide]--> Slide 1',
          source: 'strategy.pptx',
          score: 0.82,
          metadata: {
            local_graph_filename: 'strategy.pptx',
            slide: 1,
            slide_title: 'Revenue Architecture',
            layout_bbox: 'x=3.00in y=2.40in w=1.00in h=0.30in',
            layout_zone: 'middle_center',
            table_cell_reference: 'R3C1',
            table_cell_style_name: 'GovernanceCell',
            table_cell_fill_color: '#FFF2CC',
            table_cell_shading_pattern: 'pct20',
            table_cell_vertical_align: 'center',
            table_cell_text_direction: 'tbRl',
            table_cell_width_value: '4320',
            table_cell_width_inches: 3,
            table_cell_width_type: 'dxa',
            table_cell_visual_summary: 'fill #FFF2CC; text anchor ctr',
            label: 'Launch Owner',
            field_type: 'dropdown',
            value: 'Operations',
            form_field_selected_option: 'Operations',
            form_field_options: ['Operations', 'Security', 'Support'],
            chart_title: 'Regional Revenue Forecast',
            chart_series_visual_summary: 'fill #5B9BD5; line #1F4E79 2.00pt',
            chart_trendline_type: 'linear',
            chart_trendline_visual_summary: 'line #7030A0 1.50pt dash',
            diagram_part: 'ppt/diagrams/data1.xml',
            diagram_node_id: 'risk',
            diagram_node_type: 'doc',
            diagram_order: 2,
            diagram_level: 1,
            parent_label: 'Launch Plan',
            diagram_connection_type: 'parOf',
            source_label: 'Launch Plan',
            target_label: 'Risk Review',
            notes_part: 'ppt/notesSlides/notesSlide1.xml',
            notes_relationship_id: 'rIdNotes1',
            comment_id: 'cm-7',
            comment_kind: 'threaded',
            author: 'Ada Reviewer',
            comment_anchor_text: 'Launch Risk',
            revision_type: 'insert',
            revision_id: 'rev-3',
            annotation_type: 'highlight',
            annotation_subject: 'Launch decision',
            emphasis_term: 'Launch Risk',
            emphasis_styles: ['bold', 'underline'],
            bullet_levels: [0, 1],
            style_summary: 'bold; underline; font 18pt',
            media_kind: 'video',
            media_part: 'ppt/media/media1.mp4',
            media_mime: 'video/mp4',
            media_size_bytes: 12048,
          },
        },
      ],
    });

    expect(normalized.reasoningSource).toBe('local_pipeline');
    expect(normalized.reasoningTask).toBe('multi_hop');
    expect(normalized.warnings).toHaveLength(1);
    expect(normalized.hops[0]).toMatchObject({
      entity: 'Slide 1: Revenue Architecture',
      relation: 'has_slide',
      confidence: 0.72,
    });
    expect(normalized.paths[0]).toMatchObject({
      pathType: 'layout',
      score: 0.82,
    });
    expect(normalized.evidence[0].content).toContain('Strategy Deck');
    const location = formatGraphRAGReasoningLocation(normalized.evidence[0].metadata);
    expect(location).toContain('strategy.pptx');
    expect(location).toContain('x=3.00in');
    expect(location).toContain('fill #FFF2CC');
    expect(location).toContain('style GovernanceCell');
    expect(location).toContain('width 3in dxa');
    expect(location).toContain('direction tbRl');
    expect(location).toContain('field Launch Owner');
    expect(location).toContain('options Operations, Security, Support');
    expect(location).toContain('line #7030A0');
    expect(location).toContain('diagram part ppt/diagrams/data1.xml');
    expect(location).toContain('diagram node risk');
    expect(location).toContain('diagram node type doc');
    expect(location).toContain('diagram order 2');
    expect(location).toContain('diagram level 1');
    expect(location).toContain('diagram parent Launch Plan');
    expect(location).toContain('diagram edge Launch Plan to Risk Review');
    expect(location).toContain('diagram connection parOf');
    expect(location).toContain('speaker notes ppt/notesSlides/notesSlide1.xml');
    expect(location).toContain('notes relationship rIdNotes1');
    expect(location).toContain('comment cm-7');
    expect(location).toContain('comment kind threaded');
    expect(location).toContain('author Ada Reviewer');
    expect(location).toContain('anchor Launch Risk');
    expect(location).toContain('revision insert');
    expect(location).toContain('revision id rev-3');
    expect(location).toContain('annotation highlight');
    expect(location).toContain('annotation subject Launch decision');
    expect(location).toContain('emphasis Launch Risk');
    expect(location).toContain('styles bold, underline');
    expect(location).toContain('bullet levels 0, 1');
    expect(location).toContain('style summary bold; underline; font 18pt');
    expect(location).toContain('media video');
    expect(location).toContain('media part ppt/media/media1.mp4');
    expect(location).toContain('media type video/mp4');
    expect(location).toContain('media size 12048 bytes');
  });

  it('merges remote results with attached local pipeline reasoning', () => {
    const normalized = normalizeGraphRAGReasoningResult({
      answer: 'Remote answer',
      sources: [{ content: 'Remote graph source', source: 'remote' }],
      local_pipeline_reasoning: {
        reasoning_source: 'local_pipeline',
        warnings: ['Local trace attached.'],
        direct_relationships: [
          {
            source: 'Slide 1',
            relation: 'contains_connector',
            target: 'Product Revenue -> Regional Demand: drives',
            score: 0.91,
          },
        ],
        sources: [{ content: 'Local connector evidence', source: 'strategy.pptx' }],
        local_pipeline: { run_count: 1 },
      },
    });

    expect(normalized.reasoningSource).toBe('local_pipeline');
    expect(normalized.relationships[0].text).toBe('Slide 1 --[contains_connector]--> Product Revenue -> Regional Demand: drives');
    expect(normalized.evidence.map(item => item.content)).toEqual([
      'Remote graph source',
      'Local connector evidence',
    ]);
    expect(normalized.localPipeline?.run_count).toBe(1);
  });

  it('formats email message provenance for reasoning locations', () => {
    const location = formatGraphRAGReasoningLocation({
      local_graph_filename: 'launch.eml',
      email_subject: 'GraphRAG Launch Evidence',
      email_message_id: '<launch-evidence@example.com>',
      email_from: 'Ada Lovelace <ada@example.com>',
      email_to: 'Grace Hopper <grace@example.com>',
      email_cc: 'Finance Review <finance@example.com>',
      email_bcc: 'Audit Archive <audit@example.com>',
      email_reply_to: 'Launch Desk <launch@example.com>',
      email_date_iso: '2026-07-14T09:30:00-04:00',
      email_header: 'Subject',
      email_part_index: 1,
      email_part_kind: 'plain',
      email_content_type: 'text/plain',
      email_attachment_index: 1,
      email_attachment_filename: 'launch-evidence.csv',
      email_attachment_mime: 'text/csv',
      email_in_reply_to: '<prior@example.com>',
      email_references: ['<thread-root@example.com>', '<prior@example.com>'],
    });

    expect(location).toContain('launch.eml');
    expect(location).toContain('message GraphRAG Launch Evidence');
    expect(location).toContain('from Ada Lovelace <ada@example.com>');
    expect(location).toContain('to Grace Hopper <grace@example.com>');
    expect(location).toContain('cc Finance Review <finance@example.com>');
    expect(location).toContain('bcc Audit Archive <audit@example.com>');
    expect(location).toContain('sent 2026-07-14T09:30:00-04:00');
    expect(location).toContain('message id <launch-evidence@example.com>');
    expect(location).toContain('email header Subject');
    expect(location).toContain('message body part 1 plain text/plain');
    expect(location).toContain('email attachment launch-evidence.csv text/csv index 1');
    expect(location).toContain('reply-to Launch Desk <launch@example.com>');
    expect(location).toContain('in reply to <prior@example.com>');
    expect(location).toContain('references <thread-root@example.com>, <prior@example.com>');
  });

  it('formats legacy path arrays without object leakage', () => {
    const normalized = normalizeGraphRAGReasoningResult({
      paths: [['Entity A', 'Bridge B', 'Entity C']],
    });

    expect(normalized.paths[0].text).toBe('Entity A -> Bridge B -> Entity C');
    expect(normalized.paths[0].text).not.toContain('[object Object]');
    expect(normalized.paths[0].hops).toBe(2);
  });
});
