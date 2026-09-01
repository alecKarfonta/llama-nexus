import { describe, expect, it } from "vitest";
import { buildGraphRAGReferenceSourceLocations } from "./graphragReferenceLocations";

const labelValues = (locations: ReturnType<typeof buildGraphRAGReferenceSourceLocations>) =>
  locations.map((location) => `${location.label}:${location.value}`);

describe("graphrag reference source locations", () => {
  it("normalizes rich document provenance into compact drawer locations", () => {
    const locations = buildGraphRAGReferenceSourceLocations({
      local_graph_filename: "strategy.pptx",
      slide: 2,
      slide_title: "Hidden Assumptions",
      slide_visibility: "hidden",
      slide_part: "ppt/slides/slide2.xml",
      chart_label: "Revenue Architecture chart 1",
      chart_point_label: "Revenue Q2",
      pivot_table: "QualityPivot",
      pivot_table_range: "H1:J6",
      form_kind: "pptx_control",
      label: "Approval required",
      diagram_part: "ppt/diagrams/data1.xml",
      diagram_connection_type: "parOf",
      source_diagram_node_id: "root",
      target_diagram_node_id: "risk",
      markdown_code_language: "python",
      markdown_code_block_index: 2,
      markdown_start_line: 18,
      markdown_end_line: 21,
      comment_id: "cm-7",
      comment_author: "Ada Reviewer",
      notes_part: "ppt/notesSlides/notesSlide1.xml",
      emphasis_term: "Launch Risk",
      emphasis_styles: ["bold", "underline"],
      media_kind: "video",
      media_part: "ppt/media/media1.mp4",
      email_subject: "GraphRAG Launch Evidence",
      layout_left_pct: 12.5,
      layout_top_pct: 8,
      layout_width_pct: 33.3,
      layout_height_pct: 14.1,
    });

    expect(locations[0]).toEqual({
      label: "Lines",
      value: "18-21",
      category: "code",
    });
    expect(labelValues(locations).slice(0, 16)).toEqual([
      "Lines:18-21",
      "Region:12.5, 8, 33.3, 14.1",
      "File:strategy.pptx",
      "Slide:2",
      "Slide title:Hidden Assumptions",
      "Slide visibility:hidden",
      "Slide part:ppt/slides/slide2.xml",
      "Chart:Revenue Architecture chart 1",
      "Chart point:Revenue Q2",
      "Pivot:QualityPivot",
      "Pivot range:H1:J6",
      "Form:pptx_control",
      "Label:Approval required",
      "Diagram:ppt/diagrams/data1.xml",
      "Diagram edge:parOf",
      "Source diagram node:root",
    ]);
  });

  it("deduplicates equivalent fields across metadata records", () => {
    const locations = buildGraphRAGReferenceSourceLocations(
      {
        filename: "launch.pdf",
        local_graph_filename: "launch.pdf",
        page: 4,
      },
      {
        filename: "launch.pdf",
        page: 4,
        table_cell_reference: "B7",
      },
    );

    expect(labelValues(locations)).toEqual([
      "File:launch.pdf",
      "Page:4",
      "Cell:B7",
    ]);
  });

  it("surfaces detailed Office chart, table, form, and media anchors", () => {
    const locations = buildGraphRAGReferenceSourceLocations({
      filename: "approval-deck.pptx",
      table_display_name: "Owner Scorecard",
      table_cell_text_anchor: "Budget exception approved",
      chart_type: "bar",
      chart_point_label: "Q3 approval time",
      series_name: "Approval velocity",
      category: "Q3",
      value: 42,
      chart_axis_type: "value",
      chart_axis_number_format: "$#,##0",
      chart_label_fields: ["value", "category name"],
      chart_label_position: "outEnd",
      chart_trendline_name: "approval velocity",
      chart_trendline_type: "linear",
      form_field_checked: true,
      form_field_selected_option: "Approved",
      control_name: "Approval checkbox",
      control_part: "ppt/ctrlProps/ctrlProp1.xml",
      embedded_file_mime: "text/csv",
    });

    expect(labelValues(locations)).toContain("Table:Owner Scorecard");
    expect(labelValues(locations)).toContain("Cell text:Budget exception approved");
    expect(labelValues(locations)).toContain("Chart type:bar");
    expect(labelValues(locations)).toContain("Chart point:Q3 approval time");
    expect(labelValues(locations)).toContain("Series:Approval velocity");
    expect(labelValues(locations)).toContain("Category:Q3");
    expect(labelValues(locations)).toContain("Value:42");
    expect(labelValues(locations)).toContain("Chart axis type:value");
    expect(labelValues(locations)).toContain("Chart labels:value, category name");
    expect(labelValues(locations)).toContain("Trendline:approval velocity");
    expect(labelValues(locations)).toContain("Checked:yes");
    expect(labelValues(locations)).toContain("Selected:Approved");
  });

  it("surfaces form dates, text boxes, and connector provenance for drilldown", () => {
    const locations = buildGraphRAGReferenceSourceLocations({
      filename: "intake-form.docx",
      form_kind: "content_control",
      field_type: "date",
      label: "Renewal date",
      form_field_date_format: "MMMM d, yyyy",
      form_field_date_full_value: "July 22, 2026",
      form_field_calendar: "gregorian",
      form_field_options: ["July 22", "July 29", "August 5"],
      control_name: "Renewal date picker",
      shape_id: "shape-17",
      shape_title: "Renewal callout",
      shape_description: "Highlights the decision owner",
      text_box_kind: "callout",
      text_box_part: "word/document.xml#textbox-17",
      connector_id: "connector-2",
      connector_name: "approval flow",
      connector_title: "Owner to decision",
      connector_description: "Connects approval owner to final decision",
      source_label: "Owner",
      target_label: "Decision",
    });

    expect(labelValues(locations)).toContain("Date format:MMMM d, yyyy");
    expect(labelValues(locations)).toContain("Date value:July 22, 2026");
    expect(labelValues(locations)).toContain("Calendar:gregorian");
    expect(labelValues(locations)).toContain("Options:July 22, July 29, August 5");
    expect(labelValues(locations)).toContain("Control:Renewal date picker");
    expect(labelValues(locations)).toContain("Shape title:Renewal callout");
    expect(labelValues(locations)).toContain("Shape note:Highlights the decision owner");
    expect(labelValues(locations)).toContain("Text box type:callout");
    expect(labelValues(locations)).toContain("Text box part:word/document.xml#textbox-17");
    expect(labelValues(locations)).toContain("Connector:connector-2");
    expect(labelValues(locations)).toContain("Connector:approval flow");
    expect(labelValues(locations)).toContain("Connector title:Owner to decision");
    expect(labelValues(locations)).toContain("Connector note:Connects approval owner to final decision");
  });

  it("surfaces grouped PowerPoint visual provenance for graph relationships", () => {
    const locations = buildGraphRAGReferenceSourceLocations({
      filename: "approval-deck.pptx",
      slide: 1,
      slide_title: "Launch Approval Packet",
      group_label: "Approval Dependency Cluster",
      group_id: "20",
      group_child_labels: [
        "Approval required before launch workflow.",
        "Launch workflow",
        "gates",
      ],
      group_child_count: 3,
      parent_group_label: "Approval Dependency Cluster",
      parent_group_id: "20",
      grouped_visual_label: "Launch workflow",
      connector_title: "gates",
      source_shape_id: "3",
      target_shape_id: "4",
    });

    expect(labelValues(locations)).toContain("Group:Approval Dependency Cluster");
    expect(labelValues(locations)).toContain("Group id:20");
    expect(labelValues(locations)).toContain(
      "Group items:Approval required before launch workflow., Launch workflow, gates",
    );
    expect(labelValues(locations)).toContain("Group size:3");
    expect(labelValues(locations)).toContain("Parent group:Approval Dependency Cluster");
    expect(labelValues(locations)).toContain("Grouped visual:Launch workflow");
    expect(labelValues(locations)).toContain("Connector title:gates");
    expect(labelValues(locations)).toContain("Source shape:3");
    expect(labelValues(locations)).toContain("Target shape:4");
  });

  it("surfaces aggregated pipeline evidence as readable citation anchors", () => {
    const locations = buildGraphRAGReferenceSourceLocations({
      evidence_family: "diagram_relationships",
      evidence_count: 5,
    });

    expect(labelValues(locations).slice(0, 2)).toEqual([
      "Evidence type:diagram relationships",
      "Evidence count:5",
    ]);
  });
});
