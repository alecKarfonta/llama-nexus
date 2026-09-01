import { describe, expect, it } from "vitest";
import {
  chatReferenceRelationshipKey,
  selectAnswerEvidenceDirectRelationships,
} from "./chatReferenceEvidenceRanking";

describe("chat reference evidence ranking", () => {
  it("promotes only direct relationships that the answer plan uses as evidence", () => {
    const relationships = [
      {
        source: "Approval Gate",
        relation: "gates",
        target: "Launch Workflow",
      },
      {
        source: "Approval Gate",
        relation: "owner",
        target: "Operations",
      },
    ];
    const evidenceCards = [
      {
        source: "Approval Gate",
        relation: "gates",
        target: "Launch Workflow",
      },
    ];

    expect(selectAnswerEvidenceDirectRelationships(relationships, evidenceCards)).toEqual([
      relationships[0],
    ]);
  });

  it("keeps high-signal direct relationships even when the answer plan emphasizes visual context", () => {
    const relationships = [
      {
        source: "Approval Dependency Cluster",
        relation: "groups_visual",
        target: "Launch workflow",
      },
      {
        source: "Approval required before launch workflow.",
        relation: "gates",
        target: "Launch workflow",
      },
      {
        source: "Warning",
        relation: "co_occurs_with",
        target: "You",
      },
    ];
    const evidenceCards = [
      {
        source: "Approval Dependency Cluster",
        relation: "groups_visual",
        target: "Launch workflow",
      },
    ];

    expect(selectAnswerEvidenceDirectRelationships(relationships, evidenceCards)).toEqual([
      relationships[0],
      relationships[1],
    ]);
  });

  it("keeps the existing direct relationship fallback when there is no answer plan", () => {
    const relationships = [
      { source: "A", relation: "related_to", target: "B" },
      { source: "A", relation: "mentions", target: "C" },
    ];

    expect(selectAnswerEvidenceDirectRelationships(relationships, [], 1)).toEqual([
      relationships[0],
    ]);
  });

  it("normalizes relation spellings when matching evidence cards", () => {
    expect(
      chatReferenceRelationshipKey({
        source: "Approval Gate",
        relationship_type: "Launch-Gate",
        target: "Release Plan",
      }),
    ).toBe("approval gate|launch gate|release plan");
  });
});
