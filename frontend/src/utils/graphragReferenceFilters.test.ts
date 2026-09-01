import { describe, expect, it } from "vitest";
import { buildGraphRAGReferenceFilterChips } from "./graphragReferenceFilters";

describe("graphrag reference filters", () => {
  it("prioritizes answer evidence filters and labels raw details as trace", () => {
    const filters = buildGraphRAGReferenceFilterChips({
      defaultReferenceCount: 6,
      referenceCounts: {
        passage: 9,
        relationship: 8,
        path: 6,
        document: 1,
      },
      technicalReferenceCount: 18,
    });

    expect(filters.map((filter) => `${filter.label}:${filter.count}`)).toEqual([
      "Best:6",
      "Relations:8",
      "Passages:9",
      "Paths:6",
      "Docs:1",
      "Trace:18",
    ]);
  });

  it("omits empty drill-down filters", () => {
    const filters = buildGraphRAGReferenceFilterChips({
      defaultReferenceCount: 2,
      referenceCounts: {
        relationship: 2,
      },
      technicalReferenceCount: 0,
    });

    expect(filters.map((filter) => filter.value)).toEqual(["all", "relationship"]);
  });
});
