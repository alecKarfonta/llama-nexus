import type { GraphRAGTraceChatContextRequest } from "./graphragPipelinePrompts";

export type GraphRAGWorkbenchGraphRowKind = "path" | "relationship";

export interface GraphRAGWorkbenchGraphRow {
  kind?: GraphRAGWorkbenchGraphRowKind;
  title?: string;
  text?: string;
  source?: string;
  bridge?: string;
  target?: string;
  relation?: string;
}

export type GraphRAGWorkbenchChatContextOverride = Partial<GraphRAGTraceChatContextRequest>;

export interface GraphRAGWorkbenchEvidenceSampleRow {
  label: string;
  title: string;
  body?: string;
}

export const cleanGraphRAGPromptText = (value?: string): string =>
  (value || "").replace(/\s+/g, " ").trim();

const buildStructuredPathText = (row: GraphRAGWorkbenchGraphRow): string => {
  const source = cleanGraphRAGPromptText(row.source);
  const bridge = cleanGraphRAGPromptText(row.bridge);
  const target = cleanGraphRAGPromptText(row.target);
  const nodes = [source, bridge, target].filter(Boolean);

  if (source && bridge && target) {
    return nodes.join(" -> ");
  }

  return (
    cleanGraphRAGPromptText(row.title) ||
    cleanGraphRAGPromptText(row.text) ||
    (source && target
      ? [source, target].join(" -> ")
      : [source || "Source", target || "Target"].join(" -> "))
  );
};

export const buildSelectedGraphRAGPrompt = (
  row: GraphRAGWorkbenchGraphRow,
  evidenceTitle: string,
): string => {
  const source = cleanGraphRAGPromptText(row.source) || "Source";
  const target = cleanGraphRAGPromptText(row.target) || "Target";
  const relation = cleanGraphRAGPromptText(row.relation) || "related_to";
  const evidenceLabel = cleanGraphRAGPromptText(evidenceTitle) || "this document";

  if (row.kind === "relationship") {
    const edge = `${source} --[${relation}]--> ${target}`;
    return `Explain the direct GraphRAG relationship ${edge} in ${evidenceLabel}. Include the relationship label, supporting document evidence, and source provenance.`;
  }

  const path = buildStructuredPathText(row);
  return `Explain the selected GraphRAG path ${path} in ${evidenceLabel}. Walk every hop in order, name each relationship label, cite supporting document evidence, and include source provenance for each hop.`;
};

export const buildGraphRAGPipelineEvidenceSamplePrompt = (
  row: GraphRAGWorkbenchEvidenceSampleRow,
  evidenceTitle: string,
): string => {
  const label = cleanGraphRAGPromptText(row.label) || "pipeline";
  const normalizedLabel = label.toLowerCase();
  const title = cleanGraphRAGPromptText(row.title);
  const body = cleanGraphRAGPromptText(row.body);
  const evidence = [title, body && body !== title ? body : ""]
    .filter(Boolean)
    .join(" - ");
  const evidenceLabel = cleanGraphRAGPromptText(evidenceTitle) || "this document";

  if (normalizedLabel.includes("relationship")) {
    return `Use the GraphRAG context for ${evidenceLabel} to trace this relationship evidence: ${evidence}. Identify the direct graph edge, any related graph paths connecting the entities, relationship labels, and cite the supporting document evidence with source provenance.`;
  }

  if (normalizedLabel.includes("path")) {
    return `Use the GraphRAG context for ${evidenceLabel} to explain this reasoning path evidence: ${evidence}. Walk every hop in order, name each relationship label, compare direct and indirect connections, and cite source provenance for each hop.`;
  }

  if (normalizedLabel.includes("chunk")) {
    return `Explain the citable chunks evidence in ${evidenceLabel}: ${evidence}. Cite the exact document source, identify the entities or relationships grounded by the chunk, and describe how it supports GraphRAG chat.`;
  }

  if (normalizedLabel.includes("chat")) {
    return `Use the GraphRAG chat context in ${evidenceLabel}: ${evidence}. Explain which retrieval chunks, graph anchors, relationships, or starter questions are ready for conversation and cite the source provenance.`;
  }

  return `Explain the ${normalizedLabel} evidence in ${evidenceLabel}: ${evidence}. Cite the exact document source and describe how it is used for GraphRAG chat.`;
};

export const buildSelectedGraphRAGContextOverride = (
  row?: GraphRAGWorkbenchGraphRow,
): GraphRAGWorkbenchChatContextOverride => {
  if (!row) return {};

  const override: GraphRAGWorkbenchChatContextOverride = {
    source: cleanGraphRAGPromptText(row.source) || undefined,
    target: cleanGraphRAGPromptText(row.target) || undefined,
    bridge: cleanGraphRAGPromptText(row.bridge) || undefined,
  };

  if (row.kind === "relationship") {
    override.relation = cleanGraphRAGPromptText(row.relation) || undefined;
  }

  return override;
};
