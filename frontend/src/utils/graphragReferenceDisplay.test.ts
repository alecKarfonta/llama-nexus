import { describe, expect, it } from "vitest";
import {
  briefGraphRelationshipTitle,
  cleanGraphReferenceEvidenceText,
  cleanGraphReferenceEndpointLabel,
  cleanGraphReferenceLabel,
  formatGraphReferenceRelation,
  graphRelationshipEvidenceMapNodes,
  graphRelationshipEvidenceParts,
  graphRelationshipEvidenceLead,
  isDocumentLikeGraphLabel,
} from "./graphragReferenceDisplay";

describe("graphrag reference display", () => {
  it("turns internal handoff relations into readable source-map labels", () => {
    expect(formatGraphReferenceRelation("handoff_path")).toBe("graph path");
    expect(formatGraphReferenceRelation("pipeline_handoff")).toBe("handoff");
    expect(formatGraphReferenceRelation("pipeline_evidence_family")).toBe("evidence");
    expect(formatGraphReferenceRelation("source_owner")).toBe("source owner");
    expect(formatGraphReferenceRelation("co_occurs_with")).toBe("mentions");
    expect(formatGraphReferenceRelation("related_text")).toBe("mentions");
  });

  it("titles document-to-entity citations around the cited entity", () => {
    expect(
      briefGraphRelationshipTitle({
        source: "graphrag-strategy-rich-fixture.pptx",
        relation: "handoff_path",
        target: "Regional Revenue Forecast",
      }),
    ).toBe("Regional Revenue Forecast");
  });

  it("preserves meaningful entity-to-entity relationships", () => {
    expect(
      briefGraphRelationshipTitle({
        source: "Forecast Risk",
        relation: "source_owner",
        target: "Ada Lovelace",
      }),
    ).toBe("Forecast Risk to Ada Lovelace");
  });

  it("infers the visible verb when generic relationship metadata puts it in the target", () => {
    const relationship = {
      source: "Product Revenue",
      relation: "relationship",
      target: "drives Regional Demand",
    };

    expect(briefGraphRelationshipTitle(relationship)).toBe(
      "Product Revenue to Regional Demand",
    );
    expect(graphRelationshipEvidenceLead(relationship)).toBe(
      "Product Revenue drives Regional Demand",
    );
    expect(graphRelationshipEvidenceMapNodes(relationship)).toEqual([
      "Product Revenue",
      "drives",
      "Regional Demand",
    ]);
    expect(graphRelationshipEvidenceParts(relationship)).toMatchObject({
      sourceLabel: "Product Revenue",
      relationLabel: "drives",
      targetLabel: "Regional Demand",
      lead: "Product Revenue drives Regional Demand",
    });
  });

  it("trims sentence punctuation from graph endpoint labels without changing source text cleanup", () => {
    const relationship = {
      source: "Approval required before launch workflow.",
      relation: "gates",
      target: "Launch workflow",
    };

    expect(cleanGraphReferenceEndpointLabel(relationship.source)).toBe(
      "Approval required before launch workflow",
    );
    expect(briefGraphRelationshipTitle(relationship)).toBe(
      "Approval required before launch workflow to Launch workflow",
    );
    expect(graphRelationshipEvidenceLead(relationship)).toBe(
      "Approval required before launch workflow gates Launch workflow",
    );
    expect(
      cleanGraphReferenceEvidenceText(
        "Approval required before launch workflow. -> Launch workflow: gates",
        relationship,
      ),
    ).toBe("Approval required before launch workflow. -> Launch workflow: gates");
  });

  it("cleans document and table labels without hiding semantic labels", () => {
    expect(isDocumentLikeGraphLabel("reports/board-review.pdf")).toBe(true);
    expect(cleanGraphReferenceLabel("source | relation | target")).toBe("Relationship table");
    expect(cleanGraphReferenceLabel("ppt/slides/slide1.xml")).toBe("slide evidence");
    expect(cleanGraphReferenceLabel("ppt/charts/chart1.xml")).toBe("chart evidence");
    expect(cleanGraphReferenceLabel("ppt/embeddings/operations.xlsx")).toBe("embedded data");
    expect(cleanGraphReferenceLabel("xl/worksheets/sheet1.xml")).toBe("sheet evidence");
    expect(cleanGraphReferenceLabel("Regional Demand")).toBe("Regional Demand");
  });

  it("uses a readable lead for document-backed relationship evidence", () => {
    expect(
      graphRelationshipEvidenceLead({
        source: "graphrag-strategy-rich-fixture.pptx",
        relation: "handoff_path",
        target: "Regional Revenue Forecast",
      }),
    ).toBe("Regional Revenue Forecast is connected by a graph path from source evidence");
  });

  it("does not repeat generic evidence nouns in relationship leads", () => {
    expect(
      graphRelationshipEvidenceLead({
        source: "Revenue Architecture",
        relation: "contains_chart",
        target: "ppt/charts/chart1.xml",
      }),
    ).toBe("Revenue Architecture contains chart evidence");
  });

  it("phrases inverted evidence-to-container relationships naturally", () => {
    expect(
      graphRelationshipEvidenceLead({
        source: "Slide 1: Revenue Architecture table",
        relation: "contains_table",
        target: "Slide 1: Revenue Architecture",
      }),
    ).toBe("Slide 1: Revenue Architecture table is table evidence from Slide 1: Revenue Architecture");
  });

  it("keeps evidence maps concise without losing the source-document cue", () => {
    expect(
      graphRelationshipEvidenceMapNodes({
        source: "graphrag-strategy-rich-fixture.pptx",
        relation: "handoff_path",
        target: "Regional Revenue Forecast",
      }),
    ).toEqual(["Source document", "graph path", "Regional Revenue Forecast"]);
  });

  it("builds explicit source-relation-target parts for graph citations", () => {
    expect(
      graphRelationshipEvidenceParts({
        source: "graphrag-strategy-rich-fixture.pptx",
        relation: "handoff_path",
        target: "Regional Revenue Forecast",
      }),
    ).toMatchObject({
      sourceLabel: "Source document",
      relationLabel: "graph path",
      targetLabel: "Regional Revenue Forecast",
      sourceDetail: "graphrag-strategy-rich-fixture.pptx",
      targetDetail: "",
      lead: "Regional Revenue Forecast is connected by a graph path from source evidence",
      sourceIsDocumentBacked: true,
      targetIsDocumentBacked: false,
    });
  });

  it("removes repeated source filenames from citation previews", () => {
    expect(
      cleanGraphReferenceEvidenceText(
        "graphrag-strategy-rich-fixture.pptx supports Regional Revenue Forecast, graphrag-strategy-rich-fixture.pptx -> Revenue -> Regional Revenue Forecast Revenue connects the forecast chart.",
        {
          source: "graphrag-strategy-rich-fixture.pptx",
          relation: "handoff_path",
          target: "Regional Revenue Forecast",
          filename: "graphrag-strategy-rich-fixture.pptx",
        },
      ),
    ).toBe("Revenue connects the forecast chart.");
  });

  it("replaces standalone source filenames with a plain evidence label", () => {
    expect(
      cleanGraphReferenceEvidenceText(
        "Revenue connects graphrag-strategy-rich-fixture.pptx and Regional Revenue Forecast through chart data.",
        {
          source: "graphrag-strategy-rich-fixture.pptx",
          relation: "handoff_path",
          target: "Regional Revenue Forecast",
          filename: "graphrag-strategy-rich-fixture.pptx",
        },
      ),
    ).toBe("Revenue connects source document and Regional Revenue Forecast through chart data.");
  });

  it("removes raw Office pipeline metadata from citation previews", () => {
    expect(
      cleanGraphReferenceEvidenceText(
        "Operational Workbook --[owner]-- Grace Hopper (local pipeline graphrag-strategy-rich-fixture.pptx; slide 1: Revenue Architecture; Operations visibility visible; sheet part xl/worksheets/sheet1.xml; sheet relationship rId1) Artifact details",
        {
          source: "Operational Workbook",
          relation: "owner",
          target: "Grace Hopper",
          filename: "graphrag-strategy-rich-fixture.pptx",
        },
      ),
    ).toBe("Operational Workbook owner Grace Hopper (slide 1: Revenue Architecture) Artifact details");
  });

  it("keeps directed GraphRAG edge labels readable in citation previews", () => {
    expect(
      cleanGraphReferenceEvidenceText(
        "Approval Dependency Cluster --[groups_visual]--> Approval required before launch workflow. Approval Dependency Cluster groups Approval required before launch workflow., Launch workflow, gates; layout=x=0.60in y=1.20in w=5.40in h=1.20in",
        {
          source: "Approval Dependency Cluster",
          relation: "groups_visual",
          target: "Approval required before launch workflow.",
          filename: "graphrag-grouped-visuals.pptx",
        },
      ),
    ).toBe("Approval Dependency Cluster groups visual Approval required before launch workflow. Approval Dependency Cluster groups Approval required before launch workflow., Launch workflow, gates");
  });

  it("humanizes embedded package evidence in citation previews", () => {
    expect(
      cleanGraphReferenceEvidenceText(
        "Finance Review Board decision Approved Owner: Finance Review Board; Decision: Approved; table 1 row 2; table approval.csv; media embedded_package part Finance Review Board Decision.",
        {
          source: "Finance Review Board",
          relation: "decision",
          target: "Approved",
          filename: "graphrag-rich-form-fixture.pptx",
        },
      ),
    ).toBe("Finance Review Board decision Approved Owner: Finance Review Board; Decision: Approved; table 1 row 2; table approval.csv; embedded package Finance Review Board Decision.");
  });

  it("replaces legacy source package labels with user-facing source evidence", () => {
    expect(
      cleanGraphReferenceEvidenceText(
        "image part: source package part; image mime: image/png; evidence text: Risk Map; source package part",
        {
          source: "llama-nexus-rich.pptx",
          relation: "contains_image",
          target: "Risk Map",
          filename: "llama-nexus-rich.pptx",
        },
      ),
    ).toBe("image part: source evidence; image mime: image/png; evidence text: Risk Map; source evidence");
  });
});
