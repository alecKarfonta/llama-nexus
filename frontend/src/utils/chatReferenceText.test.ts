import { describe, expect, it } from "vitest";
import { cleanDocumentTextForReading } from "./chatReferenceText";

describe("cleanDocumentTextForReading", () => {
  it("keeps readable source text and hides generated GraphRAG diagnostics", () => {
    const text = cleanDocumentTextForReading(`
# Structured GraphRAG Ingest: launch.md
## Source
- File: launch.md
## Extracted Structure
Launch Approval Packet
[table_row] line 11: Approval required | gates | Launch workflow
[paragraph] line 14: The approval field is the evidence that the launch can proceed.

## Candidate Entities
- Launch Approval Packet (section) score=0.75 mentions=5
## Candidate Graph Relationships
- Approval required --[has_column_2]--> gates
## Embedding Chunks
### chunk_0000
Graph context: noisy generated context
`);

    expect(text).not.toContain("## Document Text");
    expect(text).not.toContain("## Extracted Structure");
    expect(text).toContain("Launch Approval Packet");
    expect(text).toContain("Approval required | gates | Launch workflow");
    expect(text).toContain("The approval field is the evidence");
    expect(text).not.toContain("Candidate Entities");
    expect(text).not.toContain("Candidate Graph Relationships");
    expect(text).not.toContain("Embedding Chunks");
    expect(text).not.toContain("Graph context: noisy generated context");
  });

  it("removes rich Office provenance brackets without hiding extracted content", () => {
    const text = cleanDocumentTextForReading(`
# Structured GraphRAG Ingest: strategy.pptx
## Extracted Structure
1. [document_metadata | core metadata title | metadata_part=docProps/core.xml; metadata_field=title; metadata_value=GraphRAG Revenue Strategy] Title: GraphRAG Revenue Strategy
12. [slide_title | slide 1 | shape_index=1; layout_bbox=x=0.50in y=0.30in w=9.00in h=0.60in; slide_title=Revenue Architecture] Revenue Architecture
13. [slide_text | slide 1 | shape_index=2; layout_bbox=x=1.00in y=1.00in w=5.00in h=0.50in] Northwind connects product revenue to regional demand.
14. [slide_text | slide 1 | emphasis_terms=["Forecast Risk"]; layout_x_emu=914400; layout_y_emu=1371600] Forecast Risk
; control_id=_x0000_s1026; control_name=LaunchRegion; slide=1; slide_title=Revenue Architecture] Launch region: Europe
[shape_connector] slide 1 connector 1: Product Revenue -> Regional Demand: drives

## Candidate Entities
- noisy generated entity list
`);

    expect(text).toContain("Title: GraphRAG Revenue Strategy");
    expect(text).toContain("Revenue Architecture");
    expect(text).toContain("Northwind connects product revenue to regional demand.");
    expect(text).toContain("Forecast Risk");
    expect(text).toContain("Launch region: Europe");
    expect(text).toContain("slide 1 connector 1: Product Revenue -> Regional Demand: drives");
    expect(text).not.toContain("[document_metadata");
    expect(text).not.toContain("[slide_title");
    expect(text).not.toContain("layout_bbox");
    expect(text).not.toContain("layout_x_emu");
    expect(text).not.toContain("control_id");
    expect(text).not.toContain("Candidate Entities");
  });

  it("removes inline Office package metadata from expanded document text", () => {
    const text = cleanDocumentTextForReading(`
## Document Text
Slide 1: Revenue Architecture; visibility: visible; part: ppt/slides/slide1.xml
Product Revenue
Regional Demand
Value axis; Revenue ($M); position l; format $#,##0
Revenue data labels show series name, category name, value: Q1: 12, Q2: 18; position outEnd; format $#,##0.0
Sheet ChartData; visibility: visible; part: xl/worksheets/sheet1.xml
Form field Region; layout x=1.00in y=2.00in w=3.00in h=0.40in; relationship id rId9
`);

    expect(text).toContain("Slide 1: Revenue Architecture");
    expect(text).not.toContain("## Document Text");
    expect(text).toContain("Value axis; Revenue ($M)");
    expect(text).toContain("Revenue data labels show series name, category name, value: Q1: 12, Q2: 18");
    expect(text).toContain("Sheet ChartData");
    expect(text).toContain("Form field Region");
    expect(text).not.toContain("visibility");
    expect(text).not.toContain("ppt/slides");
    expect(text).not.toContain("xl/worksheets");
    expect(text).not.toContain("position outEnd");
    expect(text).not.toContain("format $");
    expect(text).not.toContain("x=1.00in");
    expect(text).not.toContain("relationship id");
  });

  it("removes semantic table row scaffolding from expanded evidence lines", () => {
    const text = cleanDocumentTextForReading(`
Source: Approval Gate; Relationship: gates; Target: Launch Workflow; Evidence: Gate tracked in spreadsheet Launch Evidence visibility visible; table source semantic_table_row Approval Gate Gates Launch Workflow
`);

    expect(text).toContain("Gate tracked in spreadsheet Launch Evidence");
    expect(text).not.toContain("visibility visible");
    expect(text).not.toContain("table source");
    expect(text).not.toContain("semantic_table_row");
  });

  it("removes image-only dimensions and MIME fragments from expanded document text", () => {
    const text = cleanDocumentTextForReading(`
31 Risk Map; Risk map highlights regional launch blockers; Risk Map Image; 1x1 image; image/png
`);

    expect(text).toContain("Risk Map; Risk map highlights regional launch blockers; Risk Map Image");
    expect(text).not.toContain("1x1 image");
    expect(text).not.toContain("image/png");
  });

  it("deduplicates adjacent repeated extracted lines", () => {
    const text = cleanDocumentTextForReading(`
## Document Text
Slide 1: Launch Approval Packet
Launch Approval Packet
Launch Approval Packet
Approval required before launch workflow.
`);

    expect(text).toContain("Slide 1: Launch Approval Packet");
    expect(text).toContain("Approval required before launch workflow.");
    expect(text.match(/^Launch Approval Packet$/gm) || []).toHaveLength(1);
  });

  it("removes extraction table header scaffolding near labeled rows", () => {
    const text = cleanDocumentTextForReading(`
## Document Text
Owner | Decision
Owner: Finance Review Board; Decision: Approved
Source | Relation
Source: Approval required; Relation: gates Launch workflow
Source
Relation
Approval required before launch workflow.
`);

    expect(text).toContain("Owner: Finance Review Board; Decision: Approved");
    expect(text).toContain("Source: Approval required; Relation: gates Launch workflow");
    expect(text).toContain("Approval required before launch workflow.");
    expect(text).not.toMatch(/^Owner \| Decision$/m);
    expect(text).not.toMatch(/^Source \| Relation$/m);
    expect(text).not.toMatch(/^Source$/m);
    expect(text).not.toMatch(/^Relation$/m);
  });
});
