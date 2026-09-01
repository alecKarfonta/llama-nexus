import { describe, expect, it } from "vitest";
import {
  buildGraphRAGRichEvidenceOutputSummary,
  groupGraphRAGDocumentPipelineRuns,
  preferredGraphRAGPipelineRunForDocumentChat,
  selectGraphRAGWorkbenchPipelineResult,
  summarizeGraphRAGDocumentLibrary,
  summarizeGraphRAGDocumentPipelineRun,
} from "./graphragDocumentRunSummary";

describe("graphrag document run summary", () => {
  it("summarizes rich document evidence families for pipeline output cards", () => {
    const summary = buildGraphRAGRichEvidenceOutputSummary({
      quick_counts: {
        charts: 8,
        tables: 2,
        forms: 2,
        connectors: 2,
        speaker_notes: 1,
        styled_text_blocks: 15,
      },
      graph_evidence: {
        chart_relationship_count: 51,
        table_fact_count: 3,
        styled_text_relationship_count: 4,
        speaker_notes_relationship_count: 3,
      },
    });

    expect(summary).toMatchObject({
      count: 61,
      title: "8 charts · 2 tables · 2 forms · 15 styled text · 2 connectors",
      body: "51 chart links · 3 table facts · 4 style links · 3 note links",
      sampleGroups: [
        "8 charts",
        "2 tables",
        "2 forms",
        "15 styled text",
        "2 connectors",
      ],
    });
  });

  it("does not create a rich evidence output card for empty profiles", () => {
    expect(buildGraphRAGRichEvidenceOutputSummary({ quick_counts: {} })).toBeNull();
  });

  it("summarizes chat readiness, evidence counts, and stage samples", () => {
    const summary = summarizeGraphRAGDocumentPipelineRun({
      status: "complete",
      summary: {
        chunk_count: 27,
        entity_candidate_count: 40,
        relationship_candidate_count: 220,
      },
      stages: [
        {
          name: "structure_extraction",
          status: "complete",
          sample_outputs: {
            elements: [{ text: "Revenue Forecast" }, { text: "Risk Map" }],
          },
        },
        {
          name: "graph_candidates",
          status: "complete",
          sample_outputs: {
            relationships: [{ source: "Revenue", target: "Demand" }],
            selected_endpoint: "/graphrag",
          },
        },
      ],
    });

    expect(summary).toMatchObject({
      readyForChat: true,
      chunkCount: 27,
      entityCount: 40,
      relationshipCount: 220,
      stageCount: 2,
      completedStageCount: 2,
      sampleCount: 4,
    });
  });

  it("recognizes handoff readiness before a run is fully complete", () => {
    const summary = summarizeGraphRAGDocumentPipelineRun({
      status: "preview_only",
      interaction_handoff: { ready: true },
      stages: [
        {
          name: "chunking",
          status: "warning",
          sample_outputs: { chunks: [{ id: "chunk-1" }] },
        },
      ],
    });

    expect(summary.readyForChat).toBe(true);
    expect(summary.completedStageCount).toBe(0);
    expect(summary.sampleCount).toBe(1);
  });

  it("does not treat a raw submitted run as chat-ready without an explicit handoff", () => {
    const submittedOnly = summarizeGraphRAGDocumentPipelineRun({
      status: "submitted",
      stages: [
        {
          name: "upstream_ingest",
          status: "submitted",
          sample_outputs: { selected_endpoint: "/api/v1/graphrag/ingest" },
        },
      ],
    });

    const submittedWithLocalHandoff = summarizeGraphRAGDocumentPipelineRun({
      status: "submitted",
      resume_handoff: { ready: true },
      stages: [
        {
          name: "graph_interaction_plan",
          status: "complete",
          sample_outputs: { graph_anchors: [{ source: "Revenue" }] },
        },
      ],
    });

    expect(submittedOnly.readyForChat).toBe(false);
    expect(submittedOnly.completedStageCount).toBe(0);
    expect(submittedWithLocalHandoff.readyForChat).toBe(true);
    expect(submittedWithLocalHandoff.completedStageCount).toBe(1);
  });

  it("groups duplicate imports but rolls library totals from the preferred latest run", () => {
    const olderRun = {
      run_id: "older",
      status: "complete",
      completed_at: "2026-07-20T10:00:00Z",
      document: { filename: "strategy.md", domain: "general" },
      summary: {
        chunk_count: 9,
        entity_candidate_count: 20,
        relationship_candidate_count: 200,
      },
    };
    const latestRun = {
      run_id: "latest",
      status: "complete",
      completed_at: "2026-07-21T10:00:00Z",
      document: { filename: "strategy.md", domain: "general" },
      summary: {
        chunk_count: 2,
        entity_candidate_count: 8,
        relationship_candidate_count: 15,
      },
    };
    const separateDocument = {
      run_id: "other",
      status: "preview_only",
      interaction_handoff: { ready: true },
      document: { filename: "launch.md", domain: "general" },
      summary: {
        chunk_count: 3,
        entity_candidate_count: 4,
        relationship_candidate_count: 12,
      },
    };

    const groups = groupGraphRAGDocumentPipelineRuns([
      olderRun,
      latestRun,
      separateDocument,
    ]);
    const library = summarizeGraphRAGDocumentLibrary(groups);

    expect(groups).toHaveLength(2);
    expect(groups[0]).toMatchObject({
      runCount: 2,
      chunks: 11,
      relationships: 215,
      chatReady: true,
    });
    expect(preferredGraphRAGPipelineRunForDocumentChat(groups[0].runs)?.run_id).toBe("latest");
    expect(library).toMatchObject({
      chunks: 5,
      entities: 12,
      relationships: 27,
      chatReadyCount: 2,
      hasHistory: true,
    });
  });

  it("can collapse same-filename imports across domains while keeping preferred latest run", () => {
    const generalRun = {
      run_id: "general-run",
      status: "complete",
      completed_at: "2026-07-21T10:00:00Z",
      document: { filename: "strategy.pdf", domain: "general" },
      summary: {
        chunk_count: 4,
        entity_candidate_count: 5,
        relationship_candidate_count: 10,
      },
    };
    const domainSpecificRun = {
      run_id: "pdf-live-run",
      status: "complete",
      completed_at: "2026-07-22T10:00:00Z",
      document: { filename: "strategy.pdf", domain: "pdf-live" },
      summary: {
        chunk_count: 6,
        entity_candidate_count: 8,
        relationship_candidate_count: 18,
      },
    };

    const defaultGroups = groupGraphRAGDocumentPipelineRuns([
      generalRun,
      domainSpecificRun,
    ]);
    const collapsedGroups = groupGraphRAGDocumentPipelineRuns(
      [generalRun, domainSpecificRun],
      { groupAcrossDomains: true },
    );

    expect(defaultGroups).toHaveLength(2);
    expect(collapsedGroups).toHaveLength(1);
    expect(collapsedGroups[0]).toMatchObject({
      run: { run_id: "pdf-live-run" },
      runCount: 2,
      domains: ["pdf-live", "general"],
      relationships: 28,
    });
  });

  it("keeps an explicit selected document ahead of a stale hydrated run", () => {
    const staleLoadedRun = {
      run_id: "pptx-run",
      document: { filename: "old-deck.pptx" },
      stages: [{ name: "graph", sample_outputs: { relationships: [] } }],
    };
    const selectedSummaryRun = {
      run_id: "pdf-run",
      document: { filename: "selected.pdf" },
      summary: { relationship_candidate_count: 12 },
    };

    const result = selectGraphRAGWorkbenchPipelineResult({
      corpusResult: { pipeline_run: staleLoadedRun },
      selectedPipelineRunId: "pdf-run",
      selectedRecentRun: selectedSummaryRun,
    });

    expect(result).toMatchObject({
      run_id: "pdf-run",
      document: { filename: "selected.pdf" },
    });
  });

  it("uses the hydrated selected run once it matches the selected document", () => {
    const selectedSummaryRun = {
      run_id: "pdf-run",
      document: { filename: "selected.pdf" },
      summary: { relationship_candidate_count: 12 },
    };
    const hydratedSelectedRun = {
      ...selectedSummaryRun,
      stages: [
        {
          name: "graph_candidates",
          sample_outputs: { relationships: [{ source: "A", target: "B" }] },
        },
      ],
    };

    const result = selectGraphRAGWorkbenchPipelineResult({
      corpusResult: { pipeline_run: hydratedSelectedRun },
      selectedPipelineRunId: "pdf-run",
      selectedRecentRun: selectedSummaryRun,
    });

    expect(result).toBe(hydratedSelectedRun);
  });

  it("prefers hydrated selected run details over recovered batch summaries", () => {
    const recoveredBatchSummary = {
      run_id: "pdf-run",
      document: { filename: "selected.pdf" },
      stages: [
        {
          name: "graph_candidates",
          status: "complete",
          sample_outputs: {},
        },
      ],
    };
    const hydratedSelectedRun = {
      ...recoveredBatchSummary,
      stages: [
        {
          name: "graph_candidates",
          status: "complete",
          sample_outputs: { relationships: [{ source: "A", target: "B" }] },
        },
      ],
    };

    const result = selectGraphRAGWorkbenchPipelineResult({
      corpusResult: { pipeline_run: hydratedSelectedRun },
      selectedPipelineRunId: "pdf-run",
      selectedBatchRun: recoveredBatchSummary,
      selectedRecentRun: hydratedSelectedRun,
    });

    expect(result).toBe(hydratedSelectedRun);
  });

  it("keeps upload and preview results ahead of document library selection", () => {
    const selectedRun = {
      run_id: "selected-run",
      document: { filename: "selected.pdf" },
    };
    const uploadRun = {
      run_id: "upload-run",
      document: { filename: "fresh-upload.pdf" },
    };

    const result = selectGraphRAGWorkbenchPipelineResult({
      corpusResult: {
        pipeline_upload: uploadRun,
        pipeline_preview: { run_id: "preview-run" },
        pipeline_run: { run_id: "loaded-run" },
      },
      selectedPipelineRunId: "selected-run",
      selectedRecentRun: selectedRun,
    });

    expect(result).toBe(uploadRun);
  });
});
