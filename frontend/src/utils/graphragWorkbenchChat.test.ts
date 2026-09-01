import { describe, expect, it } from "vitest";
import {
  buildGraphRAGPipelineEvidenceSamplePrompt,
  buildSelectedGraphRAGContextOverride,
  buildSelectedGraphRAGPrompt,
  cleanGraphRAGPromptText,
} from "./graphragWorkbenchChat";

describe("graphrag workbench chat helpers", () => {
  it("builds path prompts from structured source, bridge, and target fields", () => {
    const prompt = buildSelectedGraphRAGPrompt(
      {
        kind: "path",
        title: "Noisy imported path title with internal labels",
        source: " Product Revenue ",
        bridge: "Regional Demand",
        target: "Risk Map",
        relation: "drives",
      },
      "direct-rich.pptx",
    );

    expect(prompt).toContain(
      "selected GraphRAG path Product Revenue -> Regional Demand -> Risk Map",
    );
    expect(prompt).toContain("Walk every hop in order");
    expect(prompt).toContain("source provenance for each hop");
    expect(prompt).not.toContain("Noisy imported path title");
  });

  it("passes source, bridge, and target for path questions without forcing one relation", () => {
    expect(
      buildSelectedGraphRAGContextOverride({
        kind: "path",
        source: "Product Revenue",
        bridge: "Regional Demand",
        target: "Risk Map",
        relation: "drives",
      }),
    ).toEqual({
      source: "Product Revenue",
      bridge: "Regional Demand",
      target: "Risk Map",
    });
  });

  it("preserves the displayed path when a parsed bridge is unavailable", () => {
    const prompt = buildSelectedGraphRAGPrompt(
      {
        kind: "path",
        title: "Product Revenue -> Regional Demand -> Supply Risk -> Risk Map",
        source: "Product Revenue",
        target: "Risk Map",
      },
      "direct-rich.pptx",
    );

    expect(prompt).toContain(
      "Product Revenue -> Regional Demand -> Supply Risk -> Risk Map",
    );
  });

  it("keeps the relation for direct relationship questions", () => {
    const row = {
      kind: "relationship" as const,
      source: "Product Revenue",
      relation: "drives",
      target: "Regional Demand",
    };

    expect(buildSelectedGraphRAGPrompt(row, "direct-rich.pptx")).toContain(
      "Product Revenue --[drives]--> Regional Demand",
    );
    expect(buildSelectedGraphRAGContextOverride(row)).toEqual({
      source: "Product Revenue",
      target: "Regional Demand",
      bridge: undefined,
      relation: "drives",
    });
  });

  it("turns relationship evidence samples into graph tracing prompts", () => {
    const prompt = buildGraphRAGPipelineEvidenceSamplePrompt(
      {
        label: "Relationships",
        title: "Drives: Product Revenue -> Regional Demand",
        body: "Product Revenue drives Regional Demand from the slide connector.",
      },
      "direct-rich.pptx",
    );

    expect(prompt).toContain("trace this relationship evidence");
    expect(prompt).toContain("direct graph edge");
    expect(prompt).toContain("related graph paths connecting the entities");
    expect(prompt).toContain("relationship labels");
    expect(prompt).toContain("source provenance");
  });

  it("turns reasoning path samples into hop-by-hop graph prompts", () => {
    const prompt = buildGraphRAGPipelineEvidenceSamplePrompt(
      {
        label: "Reasoning paths",
        title: "Product Revenue -> Regional Demand -> Risk Map",
        body: "A two-hop path connects revenue to risk.",
      },
      "direct-rich.pptx",
    );

    expect(prompt).toContain("reasoning path evidence");
    expect(prompt).toContain("Walk every hop in order");
    expect(prompt).toContain("direct and indirect connections");
    expect(prompt).toContain("source provenance for each hop");
  });

  it("keeps citable chunk prompts grounded in source evidence while surfacing graph use", () => {
    const prompt = buildGraphRAGPipelineEvidenceSamplePrompt(
      {
        label: "Citable chunks",
        title: "Slide 1: Revenue Architecture",
        body: "Northwind connects product revenue to regional demand.",
      },
      "direct-rich.pptx",
    );

    expect(prompt).toContain("citable chunks evidence");
    expect(prompt).toContain("Cite the exact document source");
    expect(prompt).toContain("entities or relationships grounded by the chunk");
    expect(prompt).toContain("GraphRAG chat");
  });

  it("normalizes prompt text for compact buttons and route state", () => {
    expect(cleanGraphRAGPromptText("  Trace\n\tProduct   Revenue  ")).toBe(
      "Trace Product Revenue",
    );
  });
});
