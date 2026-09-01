import { describe, expect, it } from "vitest";
import {
  buildGraphRAGDocumentEvidenceAnswer,
  buildGraphRAGDocumentFallbackAnswer,
} from "./graphragDocumentAnswerFallback";

describe("graphrag document answer fallback", () => {
  it("builds an instant ordered path answer from graph reasoning evidence", () => {
    const answer = buildGraphRAGDocumentEvidenceAnswer({
      query: "How does Product Revenue connect to Regional Demand and Risk Map? Cite the evidence.",
      sourceLabel: "graphrag-upload-smoke.md",
      graphReasoning: {
        paths: [
          {
            nodes: ["Product Revenue", "Regional Demand", "Risk Map"],
            relationships: [
              {
                source: "Product Revenue",
                relation: "drives",
                target: "Regional Demand",
                evidence: "Product Revenue drives Regional Demand.",
                metadata: { location: "line 3" },
              },
              {
                source: "Regional Demand",
                relation: "references",
                target: "Risk Map",
                evidence: "Regional Demand references Risk Map for launch gating.",
                metadata: { location: "line 4" },
              },
            ],
          },
        ],
      },
    });

    expect(answer).toContain(
      "Path: **Product Revenue to Regional Demand to Risk Map**.",
    );
    expect(answer).toContain(
      "Relationships: Product Revenue drives Regional Demand; Regional Demand references Risk Map.",
    );
    expect(answer).toContain("Bridge: Regional Demand.");
    expect(answer).toContain("Source: graphrag-upload-smoke.md, line 3.");
    expect(answer).toContain("Source: graphrag-upload-smoke.md, line 4.");
    expect(answer).not.toContain("--[");
    expect(answer).not.toContain("model endpoint");
  });

  it("explains cross-document graph paths from the displayed path endpoints", () => {
    const answer = buildGraphRAGDocumentEvidenceAnswer({
      query:
        "Across these uploaded documents, compare launch approval evidence in the PowerPoint and the PDF. Cite both files and describe the graph path connecting Finance Review Board, Approval required, Risk Map, and launch workflow.",
      sourceLabel: "selected batch",
      chunks: [
        {
          document_title: "llama-nexus-rich.pdf",
          metadata: {
            filename: "llama-nexus-rich.pdf",
            location: "page 1 block 1",
            semantic_text:
              "[paragraph] page 1 block 3: Finance Review Board owns approval evidence for the launch. [paragraph] page 1 block 4: Launch workflow connects to revenue impact through approval gates. [paragraph] page 1 block 5: Risk Map highlights regional blockers before release. [paragraph] page 1 block 6: Question target: cite Approval required, Finance Review Board, Risk Map, and gates.",
          },
        },
      ],
      graphReasoning: {
        answer_plan: {
          strategy: "walk_strongest_path",
          primary_path: {
            nodes: ["Approved", "Finance Review Board", "Approval"],
            display_nodes: ["Approved", "Finance Review Board", "Approval"],
            explanation:
              "Finance Review Board connects Approved and Approval through table, media, attachment, presentation structure, semantic evidence: Approved --[decision]-- Finance Review Board --[co_occurs_with]-- Approval Cite llama-nexus-rich.pptx.",
          },
          hops: [
            {
              source: "Finance Review Board",
              relation: "decision",
              target: "Approved",
              evidence:
                "Owner: Finance Review Board; Decision: Approved.",
              provenance_summary:
                "llama-nexus-rich.pptx, slide 1: Launch Approval Packet",
            },
            {
              source: "Approval",
              relation: "related text",
              target: "Finance Review Board",
              evidence:
                "Finance Review Board owns approval evidence for the launch.",
              provenance_summary:
                "llama-nexus-rich.pdf, page 1 block 3",
            },
          ],
        },
        paths: [
          {
            nodes: ["Approved", "Finance Review Board", "Approval"],
            relationships: [
              {
                source: "Finance Review Board",
                relation: "decision",
                target: "Approved",
                evidence:
                  "Owner: Finance Review Board; Decision: Approved.",
                provenance_summary:
                  "llama-nexus-rich.pptx, slide 1: Launch Approval Packet",
              },
              {
                source: "Approval",
                relation: "related text",
                target: "Finance Review Board",
                evidence:
                  "Finance Review Board owns approval evidence for the launch.",
                provenance_summary:
                  "llama-nexus-rich.pdf, page 1 block 3",
              },
            ],
          },
          {
            nodes: ["Approval", "Finance Review Board", "Risk Map"],
            relationships: [
              {
                source: "Approval",
                relation: "related text",
                target: "Finance Review Board",
                evidence:
                  "Finance Review Board owns approval evidence for the launch.",
                provenance_summary:
                  "llama-nexus-rich.pdf, page 1 block 3",
              },
              {
                source: "Finance Review Board",
                relation: "related text",
                target: "Risk Map",
                score: 0.99,
                evidence:
                  "Question target: cite Approval required, Finance Review Board, Risk Map, and gates.",
                provenance_summary:
                  "llama-nexus-rich.pdf, page 1 block 1",
              },
            ],
          },
          {
            nodes: ["Approval required", "Launch workflow"],
            relationships: [
              {
                source: "Approval required before launch workflow",
                relation: "gates",
                target: "Launch workflow",
                evidence:
                  "Approval required before launch workflow. -> Launch workflow: gates",
                provenance_summary:
                  "llama-nexus-rich.pptx, slide 1: Launch Approval Packet",
              },
            ],
          },
        ],
      },
    });

    expect(answer).toContain(
      "Path: **Approved to Finance Review Board to Approval**.",
    );
    expect(answer).toContain("Bridge: Finance Review Board.");
    expect(answer).toContain(
      "Meaning: Approved connects to Approval through Finance Review Board.",
    );
    expect(answer).toContain("Related requested context:");
    expect(answer).toContain("Risk Map highlights regional blockers before release");
    expect(answer).toContain("Finance Review Board owns approval evidence for the launch");
    expect(answer).toContain("Approval required before launch workflow gates Launch workflow");
    expect(answer).toContain("llama-nexus-rich.pptx");
    expect(answer).toContain("llama-nexus-rich.pdf");
    expect(answer).not.toContain("Question target");
    expect(answer).not.toContain("Approval related text Finance Review Board");
    expect(answer).not.toContain(
      "Finance Review Board connects to Finance Review Board through Finance Review Board",
    );
  });

  it("prefers the graph path covering all requested entities over a shorter direct PDF hop", () => {
    const answer = buildGraphRAGDocumentEvidenceAnswer({
      query:
        "How does Approval connect to Finance Review Board and Risk Map in this PDF? Cite the strongest source evidence and graph path.",
      sourceLabel: "llama-nexus-rich.pdf",
      graphReasoning: {
        answer_plan: {
          strategy: "walk_strongest_path",
          primary_path: {
            nodes: ["Finance Review Board", "Risk Map"],
            display_nodes: ["Finance Review Board", "Risk Map"],
            explanation:
              "Finance Review Board connects to Risk Map through layout, semantic evidence.",
            relationships: [
              {
                source: "Finance Review Board",
                relation: "related_text",
                target: "Risk Map",
                score: 0.95,
                evidence:
                  "Risk Map highlights regional blockers before release.",
                provenance_summary: "llama-nexus-rich.pdf, page 1 block 6",
              },
            ],
          },
          hops: [
            {
              source: "Finance Review Board",
              relation: "related_text",
              target: "Risk Map",
              score: 0.95,
              evidence:
                "Risk Map highlights regional blockers before release.",
              provenance_summary: "llama-nexus-rich.pdf, page 1 block 6",
            },
          ],
        },
        paths: [
          {
            nodes: ["Finance Review Board", "Risk Map"],
            score: 0.95,
            relationships: [
              {
                source: "Finance Review Board",
                relation: "related_text",
                target: "Risk Map",
                score: 0.95,
                evidence:
                  "Risk Map highlights regional blockers before release.",
                provenance_summary: "llama-nexus-rich.pdf, page 1 block 6",
              },
            ],
          },
          {
            nodes: ["Approval", "Finance Review Board", "Risk Map"],
            score: 0.72,
            relationships: [
              {
                source: "Approval",
                relation: "related_text",
                target: "Finance Review Board",
                evidence:
                  "Finance Review Board owns approval evidence for the launch.",
                provenance_summary: "llama-nexus-rich.pdf, page 1 block 3",
              },
              {
                source: "Finance Review Board",
                relation: "related_text",
                target: "Risk Map",
                evidence:
                  "Risk Map highlights regional blockers before release.",
                provenance_summary: "llama-nexus-rich.pdf, page 1 block 6",
              },
            ],
          },
        ],
      },
    });

    expect(answer).toContain(
      "Path: **Approval to Finance Review Board to Risk Map**.",
    );
    expect(answer).toContain("Bridge: Finance Review Board.");
    expect(answer).toContain("Finance Review Board owns approval evidence for the launch");
    expect(answer).toContain("Risk Map highlights regional blockers before release");
    expect(answer).not.toContain("Path: **Finance Review Board to Risk Map**.");
    expect(answer).not.toContain("related text");
  });

  it("lists every uploaded source when a batch question asks for each filename", () => {
    const answer = buildGraphRAGDocumentEvidenceAnswer({
      query:
        "For each uploaded filename, summarize what relationships or evidence it contributed and cite the filename.",
      sourceLabel: "Document set",
      chunks: [
        {
          content:
            "line 3: Approval Gate gates Launch Workflow. Finance Review Board owns approval evidence.",
          metadata: { filename: "launch-notes.md", location: "line 3" },
        },
        {
          content:
            "Approval Gate gates Launch Workflow. Finance Review Board owns Approval Evidence.",
          metadata: { filename: "launch-evidence.csv", location: "row 2" },
        },
        {
          content:
            "Risk Map blocks Launch Workflow when regional launch blockers remain open.",
          metadata: { filename: "launch-page.html", location: "section Risk Map" },
        },
        {
          content:
            "Launch Memo requires Approval Gate before Launch Workflow can proceed.",
          metadata: { filename: "launch-memo.docx", location: "table 1 row 2" },
        },
        {
          content:
            "Workbook evidence package supports Finance Review Board approval.",
          metadata: { filename: "launch-workbook.xlsx", location: "sheet 1 row 3" },
        },
      ],
      graphReasoning: {
        paths: [
          {
            nodes: ["Risk Map", "Launch Approval Notes", "Approval Gate"],
            relationships: [
              {
                source: "Launch Approval Notes",
                relation: "has_subsection",
                target: "Risk Map",
                evidence: "Risk Map.",
                metadata: { filename: "launch-notes.md", location: "line 3" },
              },
              {
                source: "Approval Gate",
                relation: "gates",
                target: "Launch Workflow",
                evidence:
                  "Approval Gate gates Launch Workflow. Finance Review Board owns approval evidence.",
                metadata: { filename: "launch-notes.md", location: "line 3" },
              },
            ],
          },
        ],
        direct_relationships: [
          {
            source: "Approval Gate",
            relation: "gates",
            target: "Launch Workflow",
            evidence:
              "Launch requires an approved finance gate before workflow release.",
            metadata: { filename: "launch-evidence.csv", location: "row 2" },
          },
          {
            source: "Risk Map",
            relation: "blocks",
            target: "Launch Workflow",
            evidence:
              "Regional launch blockers remain open until risk review completes.",
            metadata: {
              filename: "launch-page.html",
              location: "section Risk Map",
            },
          },
          {
            source: "Launch Memo",
            relation: "requires",
            target: "Approval Gate",
            evidence:
              "Memo approval table requires the gate before launch workflow starts.",
            metadata: {
              filename: "launch-memo.docx",
              location: "table 1 row 2",
            },
          },
          {
            source: "Approval Evidence",
            relation: "supports",
            target: "Finance Review Board",
            evidence:
              "Workbook evidence package supports the board's approval decision.",
            metadata: {
              filename: "launch-workbook.xlsx",
              location: "sheet 1 row 3",
            },
          },
        ],
      },
    });

    expect(answer).toContain("Per-file evidence:");
    expect(answer).toContain("launch-notes.md");
    expect(answer).toContain("launch-evidence.csv");
    expect(answer).toContain("launch-page.html");
    expect(answer).toContain("launch-memo.docx");
    expect(answer).toContain("launch-workbook.xlsx");
    expect(answer).toContain("Risk Map blocks Launch Workflow");
    expect(answer).toContain("Launch Memo requires Approval Gate");
    expect(answer).toMatch(
      /\*\*launch-notes\.md\*\*: Finance Review Board owns approval evidence/,
    );
    expect(answer).not.toMatch(/\*\*launch-notes\.md\*\*: Risk Map\./);
    expect(answer).not.toMatch(/\*\*launch-notes\.md\*\*: line 3:/);
    expect(answer).not.toContain("Source evidence:");
  });

  it("condenses structured chunk field labels into a readable per-file relationship", () => {
    const answer = buildGraphRAGDocumentFallbackAnswer({
      query:
        "Across these uploaded files, list each filename and identify the strongest launch approval relationship.",
      sourceLabel: "Document set",
      chunks: [
        {
          content:
            "paragraph 1: Launch Approval Memo paragraph 2: Finance Review Board owns approval evidence before launch. paragraph 3: Source [paragraph] paragraph 4: Relationship [paragraph] paragraph 5: Target [paragraph] paragraph 6: Evidence [paragraph] paragraph 7: Approval Gate [paragraph] paragraph 8: gates [paragraph] paragraph 9: Launch Workflow [paragraph] paragraph 10: Memo approval table requires the gate before launch workflow starts.",
          metadata: { filename: "launch-memo.docx", location: "paragraph 1" },
        },
        {
          content:
            "Source: Finance Review Board; Relationship: owns; Target: Approval Evidence; Evidence: Board owns evidence package.",
          metadata: { filename: "launch-evidence.csv", location: "row 3" },
        },
        {
          content:
            "Source: Approval Gate; Relationship: gates; Target: Launch Workflow; Evidence: Gate tracked in spreadsheet Launch Evidence sheet 1 row 3: Source: Finance Review Board",
          metadata: { filename: "launch-workbook.xlsx", location: "sheet 1 row 3" },
        },
      ],
    });

    expect(answer).toContain("launch-memo.docx");
    expect(answer).toContain("launch-workbook.xlsx");
    expect(answer).toContain("Gate tracked in spreadsheet Launch Evidence");
    expect(answer).toContain("Approval Gate gates Launch Workflow");
    expect(answer).toContain(
      "Memo approval table requires the gate before launch workflow starts",
    );
    expect(answer).not.toContain("paragraph 3: Source");
    expect(answer).not.toContain("sheet 1 row 3: Source");
    expect(answer).not.toContain("[paragraph]");
  });

  it("uses the answer plan path before noisy URL link paths", () => {
    const answer = buildGraphRAGDocumentEvidenceAnswer({
      query:
        "How does Product Revenue connect to Risk Map? Cite the slide evidence and explain the strongest graph path.",
      sourceLabel: "graphrag-ui-smoke-rich-deep.pptx",
      graphReasoning: {
        answer_plan: {
          strategy: "walk_strongest_path",
          primary_path: {
            source: "Product Revenue",
            target: "Risk Map",
            nodes: ["Product Revenue", "Regional Demand", "Risk Map"],
            display_text:
              "Product Revenue --[drives]-- Regional Demand --[references]-- Risk Map",
          },
          supporting_relationships: [
            {
              source: "Regional Demand",
              relation: "references",
              target: "Risk Map",
              evidence: "Regional Demand -> Risk Map: references",
              location: {
                filename: "graphrag-ui-smoke-rich-deep.pptx",
                slide: 1,
                slide_title: "Revenue Architecture",
                element_type: "shape_connector",
                location: "slide 1 connector 2",
              },
            },
            {
              source: "Product Revenue",
              relation: "drives",
              target: "Regional Demand",
              evidence: "Product Revenue -> Regional Demand: drives",
              location: {
                filename: "graphrag-ui-smoke-rich-deep.pptx",
                slide: 1,
                slide_title: "Revenue Architecture",
                element_type: "shape_connector",
                location: "slide 1 connector 1",
              },
            },
          ],
          evidence_cards: [
            {
              role: "interaction_handoff_anchor",
              source: "Risk Map",
              relation: "links_to",
              target: "https://example.com/risk-register",
              evidence: "Risk Map links to https://example.com/risk-register",
            },
          ],
        },
        paths: [
          {
            score: 0.99,
            nodes: [
              "Product Revenue",
              "Regional Demand",
              "Slide 1: Revenue Architecture",
              "Risk Map",
            ],
            relationships: [
              {
                source: "Product Revenue",
                relation: "drives",
                target: "Regional Demand",
                evidence: "Product Revenue -> Regional Demand: drives",
              },
              {
                source: "Slide 1: Revenue Architecture",
                relation: "contains_link",
                target: "Risk Map",
                evidence:
                  "Risk Map links to https://example.com/risk-register",
              },
            ],
          },
        ],
      },
    });

    expect(answer).toContain(
      "Path: **Product Revenue to Regional Demand to Risk Map**.",
    );
    expect(answer).toContain(
      "Relationships: Product Revenue drives Regional Demand; Regional Demand references Risk Map.",
    );
    expect(answer).toContain("Bridge: Regional Demand.");
    expect(answer).toContain(
      "Source: graphrag-ui-smoke-rich-deep.pptx, slide 1: Revenue Architecture.",
    );
    expect(answer).not.toContain("https://example.com/risk-register");
    expect(answer).not.toContain("contains link");
    expect(answer).not.toContain("Slide 1: Revenue Architecture --[contains");
    expect(answer).not.toContain("--[");
  });

  it("prioritizes requested rich-document evidence cards over unrelated graph paths", () => {
    const answer = buildGraphRAGDocumentEvidenceAnswer({
      query: "What chart, table, form field, styled text, connector, and layout relationship evidence did this PowerPoint extract? Cite slide evidence.",
      sourceLabel: "graphrag-ux-rich-e2e.pptx",
      graphReasoning: {
        answer_plan: {
          evidence_cards: [
            {
              role: "requested_evidence_family",
              kind: "charts",
              title: "Charts",
              evidence:
                "chart: Regional Revenue Forecast; axis: Quarter; slide: Revenue Architecture; Revenue Architecture --contains chart--> Value axis: Revenue ($M); evidence=Value axis; Revenue ($M); position l; format $#,##0; provenance=slide 1: Revenue Architecture; chart Regional Revenue Forecast; axis Revenue ($M); location slide 1 chart 1 axis 2",
              provenance_summary:
                "graphrag-ux-rich-e2e.pptx, slide 1: Revenue Architecture, chart Regional Revenue Forecast",
              location: {
                filename: "graphrag-ux-rich-e2e.pptx",
                slide: 1,
                slide_title: "Revenue Architecture",
                chart_title: "Regional Revenue Forecast",
                chart_axis_title: "Quarter",
              },
            },
            {
              role: "requested_evidence_family",
              kind: "tables",
              title: "Tables",
              evidence:
                "table: 1; row: 1; slide: Revenue Architecture; Revenue Architecture --contains table--> R1C1: Product; evidence=Product; provenance=slide 1: Revenue Architecture; table 1; cell R1C1; style bold; location slide 1 table 1 cell 1.1; Revenue Architecture --contains table--> Metric: Forecast Risk; Source Owner: Ada Lovelace; Confidence: 0.92 (formula: SUM(0.4,0.52)); evidence=Metric: Forecast Risk; Source Owner: Ada Lovelace; Confidence: 0.92 (formula: SUM(0.4,0.52)); provenance=slide 1: Revenue Architecture; sheet ChartData",
              provenance_summary:
                "graphrag-ux-rich-e2e.pptx, slide 1: Revenue Architecture, table 1",
              location: {
                filename: "graphrag-ux-rich-e2e.pptx",
                slide: 1,
                slide_title: "Revenue Architecture",
                table: 1,
                row: 1,
              },
            },
            {
              role: "requested_evidence_family",
              kind: "form_fields",
              title: "Form Fields",
              evidence:
                "label: Approved for launch; field type: Forms.CheckBox.1; checked: true; value: True; form kind: pptx control; tag: LaunchApproval; alias: Approved for launch; slide: Revenue Architecture; Revenue Architecture --contains form field--> Launch region; evidence=Launch region: Europe; provenance=slide 1: Revenue Architecture; value Europe; form Launch region; location slide 1 control 2; Launch region --has value--> Europe",
              provenance_summary:
                "graphrag-ux-rich-e2e.pptx, slide 1: Revenue Architecture",
              location: {
                filename: "graphrag-ux-rich-e2e.pptx",
                slide: 1,
                slide_title: "Revenue Architecture",
                label: "Approved for launch",
                form_field_checked: true,
                value: "True",
              },
            },
            {
              role: "requested_evidence_family",
              kind: "styled_text",
              title: "Styled Text",
              evidence:
                "Slide 1: Revenue Architecture --emphasizes--> Template Governance Note; evidence=Template Governance Note | style: bold; provenance=slide 1: Revenue Architecture; style bold; location slide 1 layout text 1; style: bold; emphasis: Forecast Risk; slide: Revenue Architecture",
              provenance_summary:
                "graphrag-ux-rich-e2e.pptx, slide 1: Revenue Architecture",
              location: {
                filename: "graphrag-ux-rich-e2e.pptx",
                slide: 1,
                slide_title: "Revenue Architecture",
                style_summary: "bold",
                emphasis_terms: ["Forecast Risk"],
              },
            },
            {
              role: "requested_evidence_family",
              kind: "connectors",
              title: "Connectors",
              evidence:
                "Product Revenue -> Regional Demand: drives; description: Product revenue drives regional demand; slide 1: Revenue Architecture",
              provenance_summary:
                "graphrag-ux-rich-e2e.pptx, slide 1: Revenue Architecture",
              location: {
                filename: "graphrag-ux-rich-e2e.pptx",
                slide: 1,
                slide_title: "Revenue Architecture",
                connector_title: "drives",
                connector_description: "Product revenue drives regional demand",
                source_label: "Product Revenue",
                target_label: "Regional Demand",
              },
            },
            {
              role: "requested_evidence_family",
              kind: "layout",
              title: "Layout Relationships",
              evidence:
                "Regional Revenue Forecast is visually near Slide 1 table 1 (Slide 1 table 1 is below Regional Revenue Forecast); distance 21.3%; zones center -> bottom center; slide 1: Revenue Architecture",
              provenance_summary:
                "graphrag-ux-rich-e2e.pptx, slide 1: Revenue Architecture",
              source: "Regional Revenue Forecast",
              relation: "visually_near",
              target: "Slide 1 table 1",
              location: {
                filename: "graphrag-ux-rich-e2e.pptx",
                slide: 1,
                slide_title: "Revenue Architecture",
                layout_relation: "below",
                related_label: "Slide 1 table 1",
                visual_distance_pct: 21.3,
                layout_zone: "center",
                target_layout_zone: "bottom_center",
              },
            },
          ],
        },
        paths: [
          {
            nodes: ["Slide 1: Revenue Architecture", "https://example.com/brand-kit"],
            relationships: [
              {
                source: "Slide 1: Revenue Architecture",
                relation: "contains_link",
                target: "https://example.com/brand-kit",
                evidence: "Llama Nexus Logo links to https://example.com/brand-kit",
              },
            ],
          },
        ],
      },
    });

    expect(answer).toContain("Extracted evidence:");
    expect(answer).toContain("**Charts**");
    expect(answer).toContain("**Tables**");
    expect(answer).toContain("**Form Fields**");
    expect(answer).toContain("**Styled Text**");
    expect(answer).toContain("**Connectors**");
    expect(answer).toContain("**Layout Relationships**");
    expect(answer).toContain("slide 1: Revenue Architecture");
    expect(answer).toContain("Revenue (USD M)");
    expect(answer).toContain("cell R1C1 = Product");
    expect(answer).toContain("Approved for launch is checked");
    expect(answer).toContain("Template Governance Note is bold");
    expect(answer).toContain("Product Revenue drives Regional Demand");
    expect(answer).toContain(
      "Regional Revenue Forecast is visually near Slide 1 table 1",
    );
    expect(answer).toContain("distance 21.3%");
    expect(answer).toContain("zones center to bottom center");
    expect(answer).not.toContain("Slide 1 table 1 is below Regional Revenue Forecast");
    expect(answer).not.toContain("--contains");
    expect(answer).not.toContain("--emphasizes");
    expect(answer).not.toContain("--visually near");
    expect(answer).not.toContain("slide 1: Revenue Architecture (slide 1: Revenue Architecture)");
    expect(answer).not.toContain("Source:");
    expect(answer).not.toContain("evidence=");
    expect(answer).not.toContain("provenance=");
    expect(answer).not.toContain("graphrag-ux-rich-e2e.pptx, graphrag-ux-rich-e2e.pptx");
    expect(answer).not.toContain("Graph path:");
    expect(answer).not.toContain("brand-kit");
  });

  it("orders rich PowerPoint evidence by the question and includes chart axis metadata", () => {
    const answer = buildGraphRAGDocumentEvidenceAnswer({
      query:
        "Show the chart, table, form field, styled text, connector, speaker notes, comment, media, image, and attachment evidence extracted from this PowerPoint. Cite the strongest source locations and relationship paths.",
      sourceLabel: "llama-nexus-rich.pptx",
      graphReasoning: {
        answer_plan: {
          evidence_cards: [
            {
              role: "requested_evidence_family",
              kind: "comments",
              title: "Comments",
              evidence:
                "Launch Approval Packet --contains comment--> Maya Reviewer: Finance needs approval evidence before launch.; evidence=Maya Reviewer: Finance needs approval evidence before launch.",
              provenance_summary: "slide 1: Launch Approval Packet",
            },
            {
              role: "requested_evidence_family",
              kind: "speaker_notes",
              title: "Speaker Notes",
              evidence:
                "[slide_text] slide 1: Launch workflow [shape_connector] slide 1 connector 1: Approval required before launch workflow. -> Launch workflow: gates [speaker_notes] slide 1 speaker notes: Speaker note: cite the Finance Review Board approval evidence.; evidence=Speaker note: cite the Finance Review Board approval evidence.",
              provenance_summary: "slide 1: Launch Approval Packet",
            },
            {
              role: "requested_evidence_family",
              kind: "tables",
              title: "Tables",
              evidence:
                "table: approval.csv; table: 1; row: 1; slide: Launch Approval Packet; Finance Review Board --decision--> Approved; evidence=Owner: Finance Review Board; Decision: Approved",
              provenance_summary:
                "slide 1: Launch Approval Packet, row 2, table approval.csv",
              location: { table: 1, row: 1 },
            },
            {
              role: "requested_evidence_family",
              kind: "charts",
              title: "Charts",
              evidence:
                "chart: Revenue Plan; axis: Quarter; slide: Launch Approval Packet; evidence=Approval; Q2: 18, Q3: 31",
              provenance_summary:
                "slide 1: Laun...",
              metadata: {
                chart_title: "Revenue Plan",
                semantic_text:
                  "[chart_axis] slide 1 chart 1 axis 2: Value axis; Revenue impact; position l; format $#,##0",
              },
            },
            {
              role: "requested_evidence_family",
              kind: "form_fields",
              title: "Form Fields",
              evidence:
                "Approval required: checked; type Forms.CheckBox.1; slide 1: Launch Approval Packet",
              provenance_summary: "slide 1: Launch Approval Packet",
              location: { label: "Approval required", form_field_checked: true },
            },
            {
              role: "requested_evidence_family",
              kind: "styled_text",
              title: "Styled Text",
              evidence:
                "Slide 1: Launch Approval Packet --emphasizes--> Approval required; evidence=Approval required | style: bold; color #00A884",
              provenance_summary: "slide 1: Launch Approval Packet",
              location: { style_summary: "bold", emphasis_terms: ["Approval required"] },
            },
            {
              role: "requested_evidence_family",
              kind: "connectors",
              title: "Connectors",
              evidence:
                "Approval required before launch workflow. -> Launch workflow: gates; slide 1: Launch Approval Packet",
              provenance_summary: "slide 1: Launch Approval Packet",
            },
            {
              role: "requested_evidence_family",
              kind: "media",
              title: "Media",
              evidence:
                "media kind: embedded media; media mime: image/png; media part: risk-map.png; slide: Launch Approval Packet; evidence text: image: slide 1 image 1: Risk Map; Risk map highlights regional launch blockers before release.",
              provenance_summary: "llama-nexus-rich.pptx, slide 1: Launch Approval Packet",
            },
            {
              role: "requested_evidence_family",
              kind: "images",
              title: "Images",
              evidence:
                "image part: risk-map.png; image mime: image/png; slide: Launch Approval Packet; evidence text: image: slide 1 image 1: Risk Map; Risk map highlights regional launch blockers before release.",
              provenance_summary: "llama-nexus-rich.pptx, slide 1: Launch Approval Packet",
            },
            {
              role: "requested_evidence_family",
              kind: "attachments",
              title: "Attachments",
              evidence:
                "embedded file filename: approval.csv; embedded file mime: text/csv",
              provenance_summary: "llama-nexus-rich.pptx, embedded file approval.csv",
            },
          ],
        },
      },
    });

    expect(answer).toContain("Extracted evidence:");
    expect(answer).toContain("Charts**: Revenue Plan chart; axes include Quarter and Revenue impact (slide 1: Launch Approval Packet).");
    expect(answer).toMatch(
      /\*\*Charts\*\*[\s\S]*\*\*Tables\*\*[\s\S]*\*\*Form Fields\*\*[\s\S]*\*\*Styled Text\*\*[\s\S]*\*\*Connectors\*\*[\s\S]*\*\*Speaker Notes\*\*[\s\S]*\*\*Comments\*\*[\s\S]*\*\*Media\*\*[\s\S]*\*\*Images\*\*[\s\S]*\*\*Attachments\*\*/,
    );
    expect(answer).toContain("Revenue Plan chart");
    expect(answer).toContain("Revenue impact");
    expect(answer).toContain("Finance Review Board");
    expect(answer).toContain("Approval required is checked");
    expect(answer).toContain("Approval required before launch workflow gates Launch workflow");
    expect(answer).toContain("cite the Finance Review Board approval evidence");
    expect(answer).toContain("Maya Reviewer: Finance needs approval evidence before launch");
    expect(answer).toContain("Embedded image Risk Map");
    expect(answer).not.toContain("Embedded Image risk-map.png");
    expect(answer).not.toContain("Embedded image risk-map.png");
    expect(answer).toContain("Risk Map image");
    expect(answer).not.toContain("Risk Map image (image/png)");
    expect(answer).toContain("Risk map highlights regional launch blockers before release");
    expect(answer).toContain("Embedded file approval.csv");
    expect(answer).not.toContain("Embedded file approval.csv (text/csv)");
    expect(answer).toContain("approval.csv");
    expect(answer).not.toContain('["Launch Approval Packet"]');
    expect(answer).not.toContain("media mime");
    expect(answer).not.toContain("media part");
    expect(answer).not.toContain("image part");
    expect(answer).not.toContain("image mime");
    expect(answer).not.toContain("embedded file filename");
    expect(answer).not.toContain("embedded file mime");
    expect(answer).not.toContain("evidence text: image");
    expect(answer).not.toContain("[speaker_notes]");
    expect(answer).not.toContain("[image]");
    expect(answer).not.toContain("--contains comment-->");
    expect(answer).not.toContain("Maya Reviewer: Maya Reviewer");
    expect(answer).not.toContain("ppt/media");
    expect(answer).not.toContain("1x1 image");
    expect(answer).not.toContain("Laun...");
  });

  it("summarizes generated document handoffs instead of treating evidence cards as an extraction audit", () => {
    const answer = buildGraphRAGDocumentEvidenceAnswer({
      query:
        "Use the recent GraphRAG pipeline trace for graphrag-ux-rich-e2e.pptx to summarize the key evidence, entities, relationships, and any meaningful graph paths. Keep the answer bounded and complete: return exactly 4 bullets, under 180 words total, named Entities, Top relationships, Best path, and Source proof.",
      sourceLabel: "graphrag-ux-rich-e2e.pptx",
      graphReasoning: {
        answer_plan: {
          evidence_cards: [
            {
              role: "requested_evidence_family",
              kind: "charts",
              title: "Charts",
              evidence: "chart: Regional Revenue Forecast; axis: Quarter",
              provenance_summary:
                "graphrag-ux-rich-e2e.pptx, slide 1: Revenue Architecture, chart Regional Revenue Forecast",
            },
          ],
        },
        direct_relationships: [
          {
            source: "Product Revenue",
            relation: "drives",
            target: "Regional Demand",
            evidence: "Product Revenue drives Regional Demand.",
            metadata: {
              filename: "graphrag-ux-rich-e2e.pptx",
              slide: 1,
              slide_title: "Revenue Architecture",
            },
          },
          {
            source: "Regional Demand",
            relation: "references",
            target: "Risk Map",
            evidence: "Regional Demand references Risk Map for launch gating.",
            metadata: {
              filename: "graphrag-ux-rich-e2e.pptx",
              slide: 1,
              slide_title: "Revenue Architecture",
            },
          },
          {
            source: "Operational Workbook",
            relation: "status",
            target: "Ready",
            evidence: "Operational Workbook status Ready.",
            metadata: {
              filename: "graphrag-ux-rich-e2e.pptx",
              slide: 1,
              slide_title: "Revenue Architecture",
            },
          },
        ],
        paths: [
          {
            score: 99,
            nodes: [
              "Layout Governance",
              "Slide 1: Revenue Architecture",
              "Revenue Architecture",
              "Source Brief",
            ],
            relationships: [
              {
                source: "Layout Governance",
                relation: "contains_shape",
                target: "Slide 1: Revenue Architecture",
                evidence: "Layout Governance appears on slide 1.",
              },
              {
                source: "Revenue Architecture",
                relation: "contains_text",
                target: "Source Brief",
                evidence: "Source Brief appears as slide text.",
              },
            ],
          },
          {
            nodes: ["Product Revenue", "Regional Demand", "Risk Map"],
            relationships: [
              {
                source: "Product Revenue",
                relation: "drives",
                target: "Regional Demand",
                evidence: "Product Revenue drives Regional Demand.",
                metadata: { slide: 1, slide_title: "Revenue Architecture" },
              },
              {
                source: "Regional Demand",
                relation: "references",
                target: "Risk Map",
                evidence: "Regional Demand references Risk Map for launch gating.",
                metadata: { slide: 1, slide_title: "Revenue Architecture" },
              },
            ],
          },
        ],
      },
    });

    expect(answer).toContain("**Entities**");
    expect(answer).toContain("**Top relationships**");
    expect(answer).toContain("**Best path**");
    expect(answer).toContain("**Source proof**");
    expect(answer).toContain("Product Revenue");
    expect(answer).toContain("Regional Demand");
    expect(answer).toContain("Risk Map");
    expect(answer).toContain("Operational Workbook status is Ready");
    expect(answer).toContain("Product Revenue to Regional Demand to Risk Map");
    expect(answer).toContain(
      "Source proof**: graphrag-ux-rich-e2e.pptx, slide 1: Revenue Architecture",
    );
    expect(answer).not.toContain("Layout Governance -> Slide 1");
    expect(answer).not.toContain("graphrag-ux-rich-e2e.pptx, slide 1: Revenue Architecture; graphrag-ux-rich-e2e.pptx");
    expect(answer).not.toContain("Extracted evidence:");
    expect(answer).not.toContain("**Charts**");
  });

  it("surfaces requested PDF links alongside forms and attachments", () => {
    const answer = buildGraphRAGDocumentEvidenceAnswer({
      query:
        "What PDF evidence connects GraphRAG evidence, Ada Lovelace, the approval form, links, and the source data attachment?",
      sourceLabel: "llama-nexus-rich.pdf",
      graphReasoning: {
        answer_plan: {
          evidence_cards: [
            {
              role: "requested_evidence_family",
              kind: "form_fields",
              title: "Form Fields",
              evidence:
                "label: Approved By; field type: Text; value: Ada Lovelace",
              provenance_summary: "llama-nexus-rich.pdf, page 1",
              location: {
                filename: "llama-nexus-rich.pdf",
                page: 1,
                label: "Approved By",
                value: "Ada Lovelace",
              },
            },
            {
              role: "requested_evidence_family",
              kind: "links",
              title: "Links",
              evidence:
                "https://example.com/evidence links to https://example.com/evidence-index; Annual Strategy links to https://example.com/graphrag; page 1 link 1",
              provenance_summary: "llama-nexus-rich.pdf, page 1",
              location: {
                filename: "llama-nexus-rich.pdf",
                page: 1,
                uri: "https://example.com/graphrag",
              },
            },
            {
              role: "requested_evidence_family",
              kind: "attachments",
              title: "Attachments",
              evidence:
                "embedded file filename: source-data.csv; embedded file mime: text/csv",
              provenance_summary: "llama-nexus-rich.pdf, embedded file 1",
            },
          ],
        },
      },
    });

    expect(answer).toContain("Extracted evidence:");
    expect(answer).toContain("**Form Fields**");
    expect(answer).toContain("Approved By = Ada Lovelace");
    expect(answer).toContain("**Links**");
    expect(answer).toContain("https://example.com/evidence links to");
    expect(answer).toContain("https://example.com/graphrag");
    expect(answer).toContain("https://example.com/evidence-index");
    expect(answer).not.toMatch(/\*\*Links\*\*:\s+com\/evidence links/);
    expect(answer).toContain("**Attachments**");
    expect(answer).toContain("source-data.csv");
    expect(answer).not.toContain("text/csv");
    expect(answer).not.toContain("evidence text: link:");
  });

  it("omits generated retrieval meta relationships from PDF summaries", () => {
    const answer = buildGraphRAGDocumentEvidenceAnswer({
      query:
        "Use the recent GraphRAG pipeline trace for graphrag-live-pdf-evidence-ui-clean-1784816080.pdf to summarize the key evidence, entities, relationships, and any meaningful graph paths. Keep the answer bounded and complete: return exactly 4 bullets, under 180 words total, named Entities, Top relationships, Best path, and Source proof.",
      sourceLabel: "graphrag-live-pdf-evidence-ui-clean-1784816080.pdf",
      graphReasoning: {
        direct_relationships: [
          {
            source: "Product Revenue",
            relation: "drives",
            target: "Regional Demand",
            evidence: "Product Revenue drives Regional Demand on page 1.",
            metadata: {
              filename: "graphrag-live-pdf-evidence-ui-clean-1784816080.pdf",
              location: "page 1",
            },
          },
          {
            source: "graphrag-live-pdf-evidence-ui-clean-1784816080.pdf",
            relation: "causes",
            target: "graph paths",
            evidence: "Generated query mentioned graph paths.",
          },
          {
            source: "graphrag-live-pdf-evidence-ui-clean-1784816080.pdf",
            relation: "uses",
            target: "graph paths",
            evidence: "Generated query mentioned graph paths.",
          },
          {
            source: "graphrag-live-pdf-evidence-ui-clean-1784816080.pdf",
            relation: "works_at",
            target: "graph paths",
            evidence: "Generated query mentioned graph paths.",
          },
        ],
      },
    });

    expect(answer).toContain("Product Revenue drives Regional Demand");
    expect(answer).toContain("graphrag-live-pdf-evidence-ui-clean-1784816080.pdf, page 1");
    expect(answer).not.toContain("graph paths");
    expect(answer).not.toContain("causes graph paths");
    expect(answer).not.toContain("uses graph paths");
    expect(answer).not.toContain("works at graph paths");
  });

  it("answers validation model questions from cleaned document chunks", () => {
    const answer = buildGraphRAGDocumentFallbackAnswer({
      query: "What validation model does this document report? Cite the source evidence.",
      sourceLabel: "turboquant_validation_output.txt",
      chunks: [
        {
          content:
            "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads. ====================================================================== TURBOQUANT PHASE A: REAL MODEL VALIDATION Model: Qwen/Qwen3-1.7B (head_dim=128, same as Qwen 27B)",
          document_title: "turboquant_validation_output.txt",
          chunk_index: 0,
          metadata: {
            filename: "turboquant_validation_output.txt",
            location: "block 1",
            source_locations: [{ location: "block 1", element_type: "paragraph" }],
          },
        },
      ],
      graphReasoning: {
        direct_relationships: [
          {
            source: "Warning",
            relation: "co_occurs_with",
            target: "You",
            evidence: "Warning: You are sending unauthenticated requests to the HF Hub.",
          },
        ],
      },
    });

    expect(answer).toContain("Qwen/Qwen3-1.7B");
    expect(answer).toContain("turboquant_validation_output.txt, block 1");
    expect(answer).not.toContain("HF_TOKEN");
    expect(answer).not.toContain("Warning -> You");
  });

  it("answers gate questions from the strongest direct graph relationship", () => {
    const answer = buildGraphRAGDocumentFallbackAnswer({
      query: "What gates the launch workflow, and what source evidence supports that relationship?",
      sourceLabel: "graphrag-rich-form-fixture.pptx",
      relationships: [
        {
          source: "Approval Dependency Cluster",
          relation: "groups_visual",
          target: "Launch workflow",
          evidence:
            "Approval Dependency Cluster groups Approval required before launch workflow., Launch workflow, gates",
          metadata: {
            filename: "graphrag-rich-form-fixture.pptx",
            slide: 1,
            slide_title: "Launch Approval Packet",
          },
        },
        {
          source: "Approval required before launch workflow.",
          relation: "gates",
          target: "Launch workflow",
          evidence: "Approval required before launch workflow. -> Launch workflow: gates",
          metadata: {
            filename: "graphrag-rich-form-fixture.pptx",
            slide: 1,
            slide_title: "Launch Approval Packet",
          },
        },
      ],
    });

    expect(answer).toContain("Approval required before launch workflow gates Launch workflow.");
    expect(answer).toContain("slide 1: Launch Approval Packet");
    expect(answer).toContain("Evidence:");
  });
});
