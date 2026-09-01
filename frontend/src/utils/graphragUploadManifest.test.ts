import { describe, expect, it } from "vitest";
import {
  buildGraphRAGUploadManifest,
  formatGraphRAGUploadFileKind,
  formatGraphRAGUploadFileSize,
} from "./graphragUploadManifest";

describe("graphrag upload manifest", () => {
  it("summarizes mixed files into a graph-ready import plan", () => {
    const manifest = buildGraphRAGUploadManifest(
      [
        { name: "forecast.pdf", size: 2_500_000 },
        { name: "board-deck.pptx", size: 8_000_000 },
        { name: "notes.md", size: 12_000 },
      ],
      { buildKnowledgeGraph: true, useSemanticChunking: true },
    );

    expect(manifest).toMatchObject({
      fileCount: 3,
      totalBytes: 10_512_000,
      totalSizeLabel: "10.0 MB",
      kindSummary: "Markdown + PDF + Slides",
      primarySummary: "3 files - 10.0 MB - Markdown + PDF + Slides",
      richFormatCount: 2,
      richFormatLabel: "2 rich files",
      modeLabel: "Batch import",
      chatModeLabel: "Relationship chat",
      chunkingLabel: "Semantic chunks",
      pipelineSummary: "Extract -> Chunk -> Graph -> Chat",
    });
    expect(manifest.pipelineSteps.map((step) => step.label)).toEqual([
      "Extract",
      "Chunk",
      "Graph",
      "Chat",
    ]);
  });

  it("switches the plan when graph extraction is off", () => {
    const manifest = buildGraphRAGUploadManifest(
      [{ name: "transcript.txt", size: 2048 }],
      { buildKnowledgeGraph: false, useSemanticChunking: false },
    );

    expect(manifest).toMatchObject({
      fileCount: 1,
      kindSummary: "Text",
      modeLabel: "Single import",
      chatModeLabel: "Document chat",
      chunkingLabel: "Standard chunks",
      pipelineSummary: "Extract -> Chunk -> Embed -> Chat",
      richFormatLabel: null,
    });
  });

  it("normalizes common document kinds and sizes", () => {
    expect(formatGraphRAGUploadFileKind("analysis.xlsx")).toBe("Table");
    expect(formatGraphRAGUploadFileKind("slides.potx")).toBe("Slides");
    expect(formatGraphRAGUploadFileKind("archive.zip")).toBe("Archive");
    expect(formatGraphRAGUploadFileKind("message.eml")).toBe("Email");
    expect(formatGraphRAGUploadFileSize(1024)).toBe("1.0 KB");
    expect(formatGraphRAGUploadFileSize(-10)).toBe("0 B");
  });
});
