export type GraphRAGReferenceKind =
  | "passage"
  | "document"
  | "relationship"
  | "path"
  | "artifact";

export type GraphRAGReferenceFilterValue =
  | "all"
  | "technical"
  | GraphRAGReferenceKind;

export interface GraphRAGReferenceFilterChip {
  value: GraphRAGReferenceFilterValue;
  label: string;
  count: number;
  tone?: "primary" | "trace";
}

export const buildGraphRAGReferenceFilterChips = ({
  defaultReferenceCount,
  referenceCounts,
  technicalReferenceCount,
}: {
  defaultReferenceCount: number;
  referenceCounts: Partial<Record<GraphRAGReferenceKind, number>>;
  technicalReferenceCount: number;
}): GraphRAGReferenceFilterChip[] => {
  const filters: GraphRAGReferenceFilterChip[] = [
    {
      value: "all",
      label: "Best",
      count: defaultReferenceCount,
      tone: "primary",
    },
    {
      value: "relationship",
      label: "Relations",
      count: referenceCounts.relationship || 0,
    },
    {
      value: "passage",
      label: "Passages",
      count: referenceCounts.passage || 0,
    },
    {
      value: "path",
      label: "Paths",
      count: referenceCounts.path || 0,
    },
    {
      value: "document",
      label: "Docs",
      count: referenceCounts.document || 0,
    },
    {
      value: "artifact",
      label: "Files",
      count: referenceCounts.artifact || 0,
    },
    {
      value: "technical",
      label: "Trace",
      count: technicalReferenceCount,
      tone: "trace",
    },
  ];

  return filters.filter((filter) => filter.value === "all" || filter.count > 0);
};
