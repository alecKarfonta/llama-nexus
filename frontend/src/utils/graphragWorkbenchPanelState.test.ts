import { describe, expect, it } from "vitest";
import {
  buildGraphRAGWorkbenchPanelState,
  graphRAGDocumentWorkspaceActiveStepTarget,
  graphRAGDocumentWorkspaceMobileGridAreas,
  graphRAGDocumentWorkspaceStepTargets,
} from "./graphragWorkbenchPanelState";

describe("graphrag workbench panel state", () => {
  it("shows selected upload state instead of stale selected-document chat", () => {
    const state = buildGraphRAGWorkbenchPanelState({
      uploadFileCount: 1,
      selectedFileNames: "new-strategy.pptx",
      uploadPrimarySummary: "1 file - 27.4 KB - Slides",
      uploadPipelineStepCount: 4,
      activeDocumentTitle: "old-document.docx",
      activeDocumentRunCount: 3,
      canAskContext: true,
    });

    expect(state).toMatchObject({
      uploadActive: true,
      uploadQueued: true,
      evidenceTitle: "new-strategy.pptx",
      evidenceChipLabel: "1 selected",
      documentHistoryLabel: "",
      eyebrow: "Selected upload",
      title: "Ready to process document",
      subtitle: "1 file - 27.4 KB - Slides",
      canAsk: false,
      showDocumentInsights: false,
      primaryAction: "process",
      primaryActionLabel: "Process",
    });
    expect(state.stats).toEqual([
      { label: "Files", value: 1 },
      { label: "Steps", value: 4 },
    ]);
  });

  it("locks the panel while a batch upload is processing", () => {
    const state = buildGraphRAGWorkbenchPanelState({
      uploadFileCount: 2,
      uploadInProgress: true,
      selectedFileNames: "deck.pptx, plan.pdf",
      uploadPipelineStepCount: 4,
      hasMultipleUploadFiles: true,
      canAskContext: true,
      formatCount: (value) => value.toLocaleString("en-US"),
    });

    expect(state).toMatchObject({
      evidenceTitle: "deck.pptx, plan.pdf",
      evidenceChipLabel: "Processing",
      eyebrow: "Preparing",
      title: "Processing 2 documents",
      subtitle: "Reading text, tables, relationships, and source locations.",
      canAsk: false,
      showDocumentInsights: false,
      primaryAction: "processing",
      primaryActionLabel: "Processing",
    });
  });

  it("restores document chat controls once upload selection is clear", () => {
    const state = buildGraphRAGWorkbenchPanelState({
      uploadFileCount: 0,
      activeDocumentTitle: "strategy.pptx",
      activeDocumentRunCount: 4,
      readyStats: [
        { label: "Chunks", value: 27 },
        { label: "Entities", value: 40 },
        { label: "Relations", value: 220 },
        { label: "Paths", value: 0 },
      ],
      canAskContext: true,
    });

    expect(state).toMatchObject({
      uploadActive: false,
      uploadQueued: false,
      evidenceTitle: "strategy.pptx",
      evidenceChipLabel: "Ready",
      documentHistoryLabel: "",
      eyebrow: "Document",
      title: "Ask this document",
      subtitle: "strategy.pptx",
      canAsk: true,
      showDocumentInsights: true,
      primaryAction: "chat",
      primaryActionLabel: "Open chat",
    });
    expect(state.stats).toEqual([
      { label: "Chunks", value: 27 },
      { label: "Entities", value: 40 },
      { label: "Relations", value: 220 },
    ]);
  });

  it("labels a selected batch as a multi-document chat target", () => {
    const state = buildGraphRAGWorkbenchPanelState({
      uploadFileCount: 0,
      hasBatchResult: true,
      batchProcessedCount: 12,
      activeDocumentTitle: "Quarterly batch",
      readyStats: [{ label: "Relations", value: 900 }],
      canAskContext: true,
      formatCount: (value) => value.toLocaleString("en-US"),
    });

    expect(state).toMatchObject({
      evidenceChipLabel: "12 docs",
      eyebrow: "Selected set",
      title: "Ask across these documents",
      canAsk: true,
      showDocumentInsights: true,
      primaryActionLabel: "Open chat",
    });
  });

  it("moves the active mobile workspace section to the top", () => {
    expect(graphRAGDocumentWorkspaceMobileGridAreas("upload")).toBe(
      `"upload" "documents" "selected"`,
    );
    expect(graphRAGDocumentWorkspaceMobileGridAreas("documents")).toBe(
      `"documents" "selected" "upload"`,
    );
    expect(graphRAGDocumentWorkspaceMobileGridAreas("selected")).toBe(
      `"selected" "documents" "upload"`,
    );
  });

  it("keeps workspace steps in the document ingestion sequence", () => {
    expect(graphRAGDocumentWorkspaceStepTargets("upload")).toEqual([
      "upload",
      "documents",
      "selected",
    ]);
    expect(graphRAGDocumentWorkspaceStepTargets("documents")).toEqual([
      "upload",
      "documents",
      "selected",
    ]);
    expect(graphRAGDocumentWorkspaceStepTargets("selected")).toEqual([
      "upload",
      "documents",
      "selected",
    ]);
  });

  it("keeps the step rail aligned with the visible mobile section", () => {
    expect(
      graphRAGDocumentWorkspaceActiveStepTarget({
        workflowStep: 3,
        desktopLayout: false,
        mobileLeadSection: "upload",
      }),
    ).toBe("upload");

    expect(
      graphRAGDocumentWorkspaceActiveStepTarget({
        workflowStep: 3,
        desktopLayout: false,
        mobileLeadSection: "selected",
      }),
    ).toBe("selected");
  });

  it("keeps the desktop step rail aligned with document readiness", () => {
    expect(
      graphRAGDocumentWorkspaceActiveStepTarget({
        workflowStep: 3,
        desktopLayout: true,
        mobileLeadSection: "upload",
      }),
    ).toBe("selected");

    expect(
      graphRAGDocumentWorkspaceActiveStepTarget({
        workflowStep: 1,
        desktopLayout: true,
        mobileLeadSection: "selected",
      }),
    ).toBe("upload");
  });

  it("lets an explicit workspace jump own the active step", () => {
    expect(
      graphRAGDocumentWorkspaceActiveStepTarget({
        workflowStep: 3,
        desktopLayout: true,
        mobileLeadSection: "upload",
        mobileWorkspaceSection: "documents",
      }),
    ).toBe("documents");
  });
});
